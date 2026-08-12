"""The frame operations KeySet materialization is written in terms of.

A :class:`~tmlt.analytics.keyset._ops._base.KeySetOp` says *what* a set of keys
is; it says nothing about what holds one. Materializing an op-tree produces a
frame, and which kind of frame that is -- a Spark
:class:`~pyspark.sql.DataFrame` or a :class:`pandas.DataFrame` -- is the
caller's choice, spelled as the :class:`~tmlt.analytics._backends.Backend`
handed to :meth:`~tmlt.analytics.keyset._ops._base.KeySetOp.dataframe`. The
:class:`~tmlt.analytics.keyset.KeySet` itself stays backend-free: one op-tree
materializes on either backend, and the two results hold the same keys.

Why the frame kind, and not callables on the backend
----------------------------------------------------

Every other backend-dependent thing the compiler does reaches Core through a
slot on the :class:`~tmlt.analytics._backends.Backend` descriptor. KeySet
materialization does not, and deliberately: the functions that would fill those
slots are the op implementations themselves, which live here under
:mod:`tmlt.analytics.keyset`, and this package imports
:mod:`tmlt.analytics._backends` to name the descriptor. A descriptor holding
them would have to import back, and since importing a submodule runs its
package's ``__init__``, the two would deadlock on whichever was imported first.

So the dependency stays one-way. A backend already declares
``dataframe_type`` -- the type its tables are carried in -- and that is exactly
the choice materialization has to make, so :func:`frame_kind` reads the answer
off the descriptor that already exists. Nothing here needs a new
:class:`~tmlt.analytics._backends.Backend` field, and a backend gets KeySet
materialization by declaring the frame type it carries, with no change to any
of this.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import enum
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

from tmlt.analytics._backends import Backend, NotSupportedByBackend
from tmlt.analytics._schema import (
    ColumnDescriptor,
    Schema,
    analytics_to_pandas_dtypes,
)

KeySetFrame = Union[DataFrame, pd.DataFrame]
"""A materialized KeySet, in whichever kind of frame it was asked for."""

TaggedValue = Tuple[str, Any]
"""One value of one key, tagged so that a null and a NaN stay apart."""


class FrameKind(enum.Enum):
    """The kind of frame a :class:`KeySetOp` is materialized into."""

    SPARK = "Spark"
    PANDAS = "pandas"

    def __str__(self) -> str:
        """The name this frame kind is called by in error messages."""
        return self.value


def frame_kind(backend: Backend) -> FrameKind:
    """Returns the kind of frame the given backend's tables are carried in.

    Args:
        backend: The backend to read the frame kind off.

    Raises:
        NotSupportedByBackend: If the backend's tables are neither Spark nor
            pandas dataframes, so that a KeySet cannot be materialized for it.
    """
    if backend.dataframe_type is pd.DataFrame:
        return FrameKind.PANDAS
    if backend.dataframe_type is DataFrame:
        return FrameKind.SPARK
    raise NotSupportedByBackend.for_op(
        "KeySet materialization",
        backend.name,
        "A KeySet can be materialized as a Spark or a pandas dataframe, and"
        f" this backend's tables are neither ({backend.dataframe_type!r}).",
    )


def frame_count(frame: KeySetFrame) -> int:
    """Returns the number of rows in a frame of either kind."""
    if isinstance(frame, pd.DataFrame):
        return len(frame.index)
    return frame.count()


def frame_is_empty(frame: KeySetFrame) -> bool:
    """Returns whether a frame of either kind has no rows."""
    if isinstance(frame, pd.DataFrame):
        return len(frame.index) == 0
    return frame.isEmpty()


def select_columns(frame: KeySetFrame, columns: Sequence[str]) -> KeySetFrame:
    """Returns a frame with the given columns, in the given order.

    A frame with no columns is left alone rather than subscripted: it is the
    total-aggregation KeySet, which has no rows to reorder columns of, and the
    two backends spell "select nothing" differently enough that it is clearer
    not to.
    """
    if not columns:
        return frame
    if isinstance(frame, pd.DataFrame):
        return frame[list(columns)]
    return frame.select(*columns)


def tagged_value(value: Any) -> TaggedValue:
    """Returns a comparable tag for one value of a key.

    A null and a NaN are tagged differently. They are different keys on Spark,
    and Core's pandas utilities keep them apart too, so a comparison that
    conflated them would not notice one turning into the other. Both are also
    tagged with a fixed payload rather than themselves, since ``nan != nan``
    would otherwise make a row unequal to itself.

    Args:
        value: The value to tag.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return ("null", None)
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return ("nan", None)
    return ("value", value)


def frame_rows(
    frame: KeySetFrame, columns: Sequence[str]
) -> Counter[Tuple[TaggedValue, ...]]:
    """Returns the multiset of a frame's rows, as tagged Python values.

    This is what "these two frames hold the same keys" is decided on: a multiset
    rather than a set, so that a duplicated row is a difference; of tagged
    values, so that a null and a NaN are not confused for one another.

    Rows are read as Python values -- ``collect`` on Spark, ``itertuples`` on
    pandas -- rather than by converting one frame into the other, so that no
    conversion gets to decide what a null is on the way. ``toPandas`` on a Spark
    frame is exactly such a conversion: it turns a nullable integer column into
    a ``float64`` one, and a null in it arrives as a NaN.

    A frame with no columns is the total-aggregation KeySet, whose rows are all
    the empty row; only how many there are can differ.

    Args:
        frame: The frame to read.
        columns: The columns to read, in the order they are compared in.
    """
    if isinstance(frame, pd.DataFrame):
        if not columns:
            return Counter([()] * len(frame.index))
        records: Iterable[Sequence[Any]] = (
            tuple(record) for record in frame[list(columns)].itertuples(index=False)
        )
    else:
        records = (tuple(row[column] for column in columns) for row in frame.collect())
    return Counter(tuple(tagged_value(value) for value in record) for record in records)


def pandas_frame_from_tuples(
    rows: Iterable[tuple], column_descriptors: Mapping[str, ColumnDescriptor]
) -> pd.DataFrame:
    """Returns the pandas frame holding the given rows, one tuple per row.

    Each column is built directly in the dtype
    :func:`~tmlt.analytics._schema.analytics_to_pandas_dtypes` gives it, rather
    than letting pandas infer one and casting afterwards. Inference is not
    merely slower here: a nullable integer column inferred from Python values
    comes back as ``float64``, and an integer above 2**53 does not survive the
    round trip through it.

    A schema with no columns is the total-aggregation KeySet. It gets the empty
    frame -- no rows and no columns -- which is what the Spark path produces for
    it, and what Core's pandas ``GroupBy`` reads as "aggregate the whole table".

    Args:
        rows: The rows, as tuples aligned with ``column_descriptors``.
        column_descriptors: The columns, in the order the tuples give them in.
    """
    columns = list(column_descriptors)
    if not columns:
        return pd.DataFrame()

    dtypes = analytics_to_pandas_dtypes(Schema(dict(column_descriptors)))
    values = list(rows)
    return pd.DataFrame(
        {
            column: pd.Series([row[position] for row in values], dtype=dtypes[column])
            for position, column in enumerate(columns)
        },
        columns=columns,
    )


def cast_to_schema(
    frame: pd.DataFrame, schema: Mapping[str, ColumnDescriptor]
) -> pd.DataFrame:
    """Returns a frame whose columns have the canonical dtype for their descriptor.

    Materialization mostly produces those dtypes already, since every column
    starts out in one and no operation below changes a column it does not have
    to. The exception is a join column, whose descriptor says ``allow_null=False``
    when *both* sides forbid nulls but whose values come from the left side
    alone -- a nullable left column joined to a non-nullable right one can only
    match on non-null values, so the result holds no nulls but arrives in the
    left side's nullable dtype. This casts it back, and is a no-op otherwise.

    Args:
        frame: The frame to cast.
        schema: The descriptors its columns should match.
    """
    dtypes = analytics_to_pandas_dtypes(Schema(dict(schema)))
    if all(frame[column].dtype == dtype for column, dtype in dtypes.items()):
        return frame
    return frame.astype(dtypes)


def unsupported(
    op: Any, kind: FrameKind, hint: Optional[str] = None
) -> NotSupportedByBackend:
    """Returns the error for an operation that cannot materialize as this frame kind."""
    return NotSupportedByBackend.for_op(type(op).__name__, str(kind), hint)
