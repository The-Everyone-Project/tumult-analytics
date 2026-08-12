"""When two frames from two backends count as the same answer.

:func:`assert_frame_equal_across_backends` is the assertion a parity test ends
with. It exists because :func:`pandas.testing.assert_frame_equal` answers a
different question: it compares *pandas* frames, dtype for dtype, and it treats
``None`` and ``NaN`` in an ``object`` column as the same missing value. Across
backends both of those are wrong. The dtypes legitimately differ -- a count
column comes back ``Int64`` from one backend and ``int64`` from the other, which
is a difference in how the column is stored and not in what it says -- while
``None`` and ``NaN`` are two different answers, and a backend that returns one
where the other returns the other has a bug this suite exists to find.

What the assertion does, then:

* It ignores column *order*, comparing by name.
* It ignores row order when told which columns to sort by, which is what a
  groupby result needs: neither backend promises an order. ``sort_by=[]`` sorts
  by every column, making the comparison a multiset comparison.
* It ignores dtypes, and compares *values*: an ``Int64`` 5 equals an ``int64``
  5, and both are compared as integers, so no value is rounded through a float
  on the way to the comparison.
* It distinguishes a null from a NaN everywhere, including in an ``object``
  column, which is the one pandas column that can hold both.

Numbers of different types still compare equal when they *are* equal --
``1 == 1.0`` -- because that is Python's own equality on the values, and nothing
here converts one to the other. ``2**53 + 1`` does not equal ``float(2**53 + 1)``
for exactly the same reason: the float is a different number.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from test.backend_testing.frames import (
    AnyFrame,
    Dtypes,
    SchemaLike,
    is_null_value,
    to_pandas,
)


class _Sentinel:
    """A value that is only equal to itself, for the two flavors of missing."""

    def __init__(self, name: str):
        """Constructor.

        Args:
            name: How the sentinel prints.
        """
        self._name = name

    def __repr__(self) -> str:
        """Returns the sentinel's name."""
        return self._name


NULL = _Sentinel("NULL")
"""What every null flavor -- ``None``, ``pd.NA``, ``NaT`` -- compares as."""

NAN = _Sentinel("NaN")
"""What a floating point NaN compares as. Never equal to :data:`NULL`."""


def value_key(value: Any) -> Any:
    """Returns the value a comparison should use in place of a cell.

    The key collapses the *flavors* of null onto :data:`NULL` -- ``None`` from a
    Spark frame and ``pd.NA`` from a nullable pandas column are the same answer
    -- and every NaN onto :data:`NAN`, which is not equal to :data:`NULL`. Numpy
    scalars become their Python equivalents so that a value is compared by what
    it is rather than by which library boxed it.

    Nothing else is merged. In particular an integer stays an integer: it is
    never widened to a float, so two integers that differ above ``2**53`` still
    differ.

    Args:
        value: A cell of a frame.

    Returns:
        The key to compare with.
    """
    if is_null_value(value):
        return NULL
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return NAN if math.isnan(value) else value
    if isinstance(value, Decimal):
        return NAN if value.is_nan() else value
    if isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
        return NULL if timestamp is pd.NaT else timestamp.to_pydatetime()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


_TYPE_RANKS: Sequence[Tuple[Any, int]] = (
    (bool, 1),
    ((int, float, Decimal), 2),
    (str, 3),
    (bytes, 4),
    (datetime.datetime, 5),
    (datetime.date, 6),
)
"""Sort ranks, so that a column holding several types still has a total order."""


def _sort_key(key: Any) -> Tuple[int, Any]:
    """Returns a totally-ordered sort key for a value key.

    Sorting only has to be *consistent* between the two frames being compared,
    never meaningful, so this ranks by type first and falls back to the value's
    text where the type is not one that orders naturally.

    Args:
        key: A value key, as returned by :func:`value_key`.

    Returns:
        A ``(rank, key)`` pair.
    """
    if key is NULL:
        return (-2, "")
    if key is NAN:
        return (-1, "")
    for types, rank in _TYPE_RANKS:
        if isinstance(key, types):
            return (rank, key)
    return (99, repr(key))


def _rows_of(frame: pd.DataFrame, columns: Sequence[str]) -> List[Tuple[Any, ...]]:
    """Returns a frame's rows as tuples of value keys, in the given column order.

    Args:
        frame: The frame to read.
        columns: The column order to read it in.

    Returns:
        One tuple per row.
    """
    by_column = {
        name: [value_key(value) for value in frame[name].to_numpy(dtype=object)]
        for name in columns
    }
    return [
        tuple(by_column[name][index] for name in columns) for index in range(len(frame))
    ]


def _sorted_rows(
    rows: List[Tuple[Any, ...]], columns: Sequence[str], sort_by: Sequence[str]
) -> List[Tuple[Any, ...]]:
    """Returns rows sorted by the given columns, then by the rest.

    The remaining columns are used as a tiebreak so that the order is total and
    the two frames being compared are put in the same order even when several
    rows share a group key.

    Args:
        rows: The rows to sort.
        columns: The column order the rows are in.
        sort_by: The columns to sort by first.

    Returns:
        The sorted rows.
    """
    positions = [columns.index(name) for name in sort_by]
    positions += [index for index in range(len(columns)) if index not in positions]
    return sorted(
        rows, key=lambda row: tuple(_sort_key(row[position]) for position in positions)
    )


def _describe(row: Tuple[Any, ...], columns: Sequence[str]) -> str:
    """Renders a row for an assertion message.

    Args:
        row: The row's value keys.
        columns: The column names, in the row's order.

    Returns:
        The rendering.
    """
    return (
        "{" + ", ".join(f"{name}: {value!r}" for name, value in zip(columns, row)) + "}"
    )


def assert_frame_equal_across_backends(
    result: AnyFrame,
    expected: AnyFrame,
    schema: Optional[SchemaLike] = None,
    *,
    sort_by: Optional[Sequence[str]] = None,
    dtypes: Optional[Dtypes] = None,
    max_reported: int = 5,
) -> None:
    """Asserts that two frames hold the same answer, whichever backend built them.

    Both frames are brought to pandas through
    :func:`~test.backend_testing.frames.to_pandas` first, so either may be a
    Spark frame or a pandas one, and the values survive the trip. See the module
    docstring for what "the same answer" means here -- dtypes and column order
    do not count, a null and a NaN do.

    Args:
        result: The frame under test.
        expected: The frame it should equal.
        schema: The Analytics schema both frames are expected to have. Passing
            it makes the conversion exact for a frame that carries no usable
            type information of its own; without it, a Spark frame's own schema
            and a pandas frame's own dtypes are used.
        sort_by: Columns to sort both frames by before comparing, for a result
            whose row order is not meaningful -- the groupby keys, typically.
            The other columns break ties, so the order is total. Pass ``[]`` to
            sort by every column, which compares the frames as multisets. Pass
            nothing to compare in the order the frames are already in.
        dtypes: Per-column pandas dtypes for the conversion, overriding
            ``schema``.
        max_reported: How many differing rows to show in the failure message.

    Raises:
        AssertionError: If the frames' columns, lengths, or values differ.
    """
    result_frame = to_pandas(result, schema=schema, dtypes=dtypes)
    try:
        expected_frame = to_pandas(expected, schema=schema, dtypes=dtypes)
    except ValueError as error:
        raise AssertionError(
            f"The expected frame does not fit the schema it was compared under: {error}"
        ) from error

    result_columns = {str(name) for name in result_frame.columns}
    expected_columns = {str(name) for name in expected_frame.columns}
    if result_columns != expected_columns:
        missing = sorted(expected_columns - result_columns)
        extra = sorted(result_columns - expected_columns)
        raise AssertionError(
            "Frames have different columns: "
            f"missing {missing or 'nothing'}, unexpected {extra or 'nothing'}."
        )
    # Column order is not part of the answer, so both frames are read in one
    # canonical order rather than in whichever order they happen to be in.
    columns = sorted(result_columns)

    if len(result_frame) != len(expected_frame):
        raise AssertionError(
            f"Frames have different lengths: {len(result_frame)} rows, expected "
            f"{len(expected_frame)}."
        )

    result_rows = _rows_of(result_frame, columns)
    expected_rows = _rows_of(expected_frame, columns)
    if sort_by is not None:
        unknown = [name for name in sort_by if name not in columns]
        if unknown:
            raise AssertionError(
                f"Cannot sort by column(s) the frames do not have: {unknown}."
            )
        result_rows = _sorted_rows(result_rows, columns, sort_by)
        expected_rows = _sorted_rows(expected_rows, columns, sort_by)

    differences = [
        (index, got, want)
        for index, (got, want) in enumerate(zip(result_rows, expected_rows))
        if got != want
    ]
    if differences:
        shown = differences[:max_reported]
        lines = [
            f"  row {index}: {_describe(got, columns)} != {_describe(want, columns)}"
            for index, got, want in shown
        ]
        if len(differences) > len(shown):
            lines.append(f"  ... and {len(differences) - len(shown)} more")
        ordering = (
            f"sorted by {list(sort_by)}" if sort_by is not None else "in frame order"
        )
        raise AssertionError(
            f"Frames differ in {len(differences)} of {len(result_rows)} rows "
            f"({ordering}):\n" + "\n".join(lines)
        )


def frame_as_rows(frame: AnyFrame, **kwargs: Any) -> List[Dict[str, Any]]:
    """Returns a frame's rows as dicts of value keys, for ad-hoc assertions.

    A convenience for the cases where an assertion is about a handful of values
    rather than about a whole frame; the values are the same keys
    :func:`assert_frame_equal_across_backends` compares, so a null reads as
    :data:`NULL` and a NaN as :data:`NAN`.

    Args:
        frame: The frame to read.
        **kwargs: Passed to :func:`~test.backend_testing.frames.to_pandas`.

    Returns:
        One dict per row.
    """
    converted = to_pandas(frame, **kwargs)
    columns = [str(name) for name in converted.columns]
    return [dict(zip(columns, row)) for row in _rows_of(converted, columns)]
