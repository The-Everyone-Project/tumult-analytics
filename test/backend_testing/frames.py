"""Null-safe conversion between a backend's frames and pandas.

This is the module that keeps values intact while they cross the backend
boundary. Two directions:

* :func:`pandas_frame_from_rows` builds a pandas frame from plain Python values
  under an explicit dtype per column, which is how both the pandas twins of the
  standard tables and a test's *expected* frame should be built.
* :func:`to_pandas` brings a result frame back to pandas so that one assertion
  can compare results from either backend.

Why not ``toPandas()``
======================

``DataFrame.toPandas()`` is the obvious implementation of :func:`to_pandas` and
the wrong one. Verified against this branch's Spark (3.5) and pandas (1.5):

* A nullable ``LongType`` column comes back as ``float64``. Every integer above
  ``2**53`` is then a different number -- ``2**53 + 1`` becomes ``2**53`` -- and
  the corruption hides from a naive assertion, because comparing the resulting
  ``numpy.float64`` with the original Python ``int`` promotes the int to float
  and reports equality. It shows up only once the value is read back out as an
  int.
* A ``NULL`` in a floating point column comes back as ``NaN``, which conflates
  the two: a ``float64`` column has nowhere to put a null.

:func:`to_pandas` instead collects the frame row-wise -- ``collect()`` hands
back Python objects, where a ``NULL`` is ``None`` and a NaN is a float NaN --
and rebuilds each column at the dtype the column's Analytics descriptor calls
for. That dtype is the *nullable* extension dtype exactly when the column admits
nulls, so a null has a mask bit to live in and never has to impersonate a NaN.

The same trap sits one layer down, inside pandas: ``astype("Float64")`` on an
``object`` column turns a float ``NaN`` into ``pd.NA``. Every column here is
therefore built by constructing the masked array directly -- values and mask
separately -- rather than by casting.

Row-wise collection is O(rows) in Python and is only appropriate for the small
frames a parity test compares. It is not a data path.

Timestamps
==========

Spark renders a ``TimestampType`` using ``spark.sql.session.timeZone``, so a
frame of timestamps only means the same wall clock on both backends when that
setting is UTC. None of the standard tables has a timestamp column; a suite that
adds one has to pin the session timezone, as Core's harness does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

from tmlt.analytics._schema import (
    ColumnDescriptor,
    Schema,
    analytics_to_pandas_dtypes,
    pandas_dtypes_to_analytics_columns,
    spark_schema_to_analytics_columns,
)

AnyFrame = Union[SparkDataFrame, pd.DataFrame]
"""A frame of either backend."""

SchemaLike = Union[Schema, Mapping[str, ColumnDescriptor]]
"""An Analytics schema, or the column descriptors that make one up."""

Dtypes = Mapping[str, Any]
"""A pandas dtype per column name."""


def is_null_value(value: Any) -> bool:
    """Returns whether a value is a null, as opposed to a float NaN.

    The harness's null taxonomy, stated once: ``None``, :data:`pandas.NA` and
    :data:`pandas.NaT` are nulls, and a float ``NaN`` is a *value*. It matches
    the taxonomy Core's parity harness uses and the one the pandas backend's own
    modules document, and it is what makes "None must not equal NaN" a statement
    with content.

    Args:
        value: The value to classify.

    Returns:
        Whether the value is a null.
    """
    return value is None or value is pd.NA or value is pd.NaT


def as_schema(schema: SchemaLike) -> Schema:
    """Returns an Analytics schema, from either a schema or its columns.

    Args:
        schema: A :class:`~tmlt.analytics._schema.Schema`, or a mapping of
            column name to :class:`~tmlt.analytics._schema.ColumnDescriptor`.

    Returns:
        The schema.
    """
    return schema if isinstance(schema, Schema) else Schema(dict(schema))


def pandas_dtypes_for(schema: SchemaLike) -> Dict[str, Any]:
    """Returns the canonical pandas dtype of every column of a schema.

    This is :func:`~tmlt.analytics._schema.analytics_to_pandas_dtypes`, taking
    column descriptors as well as a schema. A column that admits nulls gets the
    nullable extension dtype, which is what gives a null somewhere to live that
    is not a NaN.

    Args:
        schema: The schema to convert.

    Returns:
        The dtype of each column.
    """
    return analytics_to_pandas_dtypes(as_schema(schema))


def analytics_columns(frame: AnyFrame) -> Dict[str, ColumnDescriptor]:
    """Returns the Analytics columns a frame's own type information implies.

    For a Spark frame this reads the frame's schema; for a pandas frame it reads
    its dtypes (and, for an ``object`` column, its values, which is the only
    thing that distinguishes a ``VARCHAR`` column from a ``DATE`` one). The two
    do not always agree -- see
    :func:`~test.backend_testing.materialize.expected_columns` for the rule.

    Args:
        frame: The frame to describe.

    Returns:
        One :class:`~tmlt.analytics._schema.ColumnDescriptor` per column.

    Raises:
        TypeError: If the frame is of neither backend's type.
    """
    if isinstance(frame, SparkDataFrame):
        return spark_schema_to_analytics_columns(frame.schema)
    if isinstance(frame, pd.DataFrame):
        return pandas_dtypes_to_analytics_columns(frame)
    raise TypeError(f"Not a frame of either backend: {type(frame).__name__}.")


################################################################################
# Building a pandas frame from Python values
################################################################################


def _is_masked_dtype(dtype: Any) -> bool:
    """Returns whether a dtype is one of pandas' masked extension dtypes.

    ``Int64``, ``Float64`` and their siblings store their values in a numpy
    array with a separate boolean mask, which is what lets them hold a null that
    is not a NaN. They are recognized by ``numpy_dtype``, which no other
    extension dtype has, rather than by importing ``BaseMaskedDtype``, whose
    module has moved between the pandas versions this branch supports.

    Args:
        dtype: The dtype to classify.

    Returns:
        Whether values of this dtype are stored with a mask.
    """
    return isinstance(dtype, pd.api.extensions.ExtensionDtype) and hasattr(
        dtype, "numpy_dtype"
    )


def _masked_arrays(values: Sequence[Any], numpy_dtype: Any, fill: Any):
    """Splits values into the data and mask a pandas masked array needs.

    Args:
        values: The column's values, with nulls as any of the null flavors.
        numpy_dtype: The dtype of the data array.
        fill: The value to put where a null is; it is masked out, so it is never
            read, but it has to be of the data array's type.

    Returns:
        A ``(data, mask)`` pair.
    """
    mask = np.array([is_null_value(value) for value in values], dtype=bool)
    data = np.array(
        [fill if is_null_value(value) else value for value in values], dtype=numpy_dtype
    )
    return data, mask


def _require_integral(name: str, values: Sequence[Any], dtype: Any) -> None:
    """Raises if a value would be truncated on its way into an integer column.

    numpy truncates silently: ``np.array([1.5], dtype="int64")`` is ``[1]``, and
    a NaN becomes an arbitrary integer. Either would turn a real difference
    between the backends into a passing test, so a non-integral value is an
    error instead. A float that happens to be integral (``5.0``) is fine, since
    nothing is lost.

    Args:
        name: The column's name, for the error message.
        values: The column's values.
        dtype: The integer dtype the column is being built at.

    Raises:
        ValueError: If a value is not an integer and not an integral float.
    """
    for value in values:
        if is_null_value(value) or isinstance(value, (int, np.integer)):
            continue
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            continue
        raise ValueError(
            f"Column '{name}' holds {value!r}, which is not an integer, but the "
            f"column's dtype is {dtype}. Building it would silently change the "
            "value."
        )


def _column(name: str, values: Sequence[Any], dtype: Any) -> pd.Series:
    """Builds one pandas column of the given dtype from Python values.

    Args:
        name: The column's name, for error messages.
        values: The column's values, with ``None`` (or another null flavor) for
            a null and a float ``NaN`` for a NaN.
        dtype: The pandas dtype to build.

    Returns:
        The column.

    Raises:
        ValueError: If the values contain a null the dtype cannot hold, or the
            dtype is not one Analytics uses for a table column.
    """
    has_null = any(is_null_value(value) for value in values)
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        if _is_masked_dtype(dtype):
            if np.issubdtype(dtype.numpy_dtype, np.integer):
                _require_integral(name, values, dtype)
            # Constructing the masked array from its two halves is the whole
            # point: astype() to a nullable float dtype turns a NaN into pd.NA,
            # which is exactly the distinction this harness exists to keep.
            data, mask = _masked_arrays(values, dtype.numpy_dtype, 0)
            return pd.Series(dtype.construct_array_type()(data, mask), name=name)
        # A non-masked extension dtype (a pandas "string" column, say) is not
        # one Analytics builds, so there is no established null handling for it.
        raise ValueError(
            f"Column '{name}': {dtype} is not a dtype this harness builds. Use "
            "the dtype analytics_to_pandas_dtypes gives for the column."
        )
    dtype = np.dtype(dtype)
    if dtype == np.dtype(object):
        # Assigning into an object array, rather than passing the list to
        # pandas, keeps None as None and NaN as NaN instead of letting pandas
        # decide which of them a "missing" value is.
        data = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            data[index] = value
        return pd.Series(data, dtype=object, name=name)
    if np.issubdtype(dtype, np.datetime64):
        return pd.Series(list(values), dtype=dtype, name=name)
    if np.issubdtype(dtype, np.integer):
        if has_null:
            raise ValueError(
                f"Column '{name}' contains a null, which the numpy dtype {dtype} "
                "cannot hold. A column that admits nulls must be declared "
                "allow_null=True, which gives it the nullable Int64 dtype."
            )
        _require_integral(name, values, dtype)
        return pd.Series(np.array(list(values), dtype=dtype), name=name)
    if np.issubdtype(dtype, np.floating):
        if has_null:
            raise ValueError(
                f"Column '{name}' contains a null, which the numpy dtype {dtype} "
                "cannot hold without turning it into a NaN. A floating point "
                "column that admits nulls must be declared allow_null=True, "
                "which gives it the nullable Float64 dtype."
            )
        return pd.Series(np.array(list(values), dtype=dtype), name=name)
    raise ValueError(f"Column '{name}': unsupported dtype {dtype}.")


def pandas_frame_from_columns(
    columns: Mapping[str, Sequence[Any]], dtypes: Dtypes
) -> pd.DataFrame:
    """Builds a pandas frame from Python values, one column at a time.

    Args:
        columns: The values of each column, in column order.
        dtypes: The pandas dtype to build each column at. Every column must
            have one.

    Returns:
        The frame, with the requested dtypes and no null-versus-NaN confusion.

    Raises:
        ValueError: If a column has no dtype, or its values do not fit it.
    """
    missing = [name for name in columns if name not in dtypes]
    if missing:
        raise ValueError(f"No dtype given for column(s): {', '.join(missing)}.")
    built = {
        name: _column(name, list(values), dtypes[name])
        for name, values in columns.items()
    }
    # An empty dict of columns still has to produce a frame with no rows rather
    # than one pandas invents an index for.
    frame = pd.DataFrame(built, columns=list(columns))
    return frame


def pandas_frame_from_rows(
    columns: Sequence[str], rows: Sequence[Sequence[Any]], dtypes: Dtypes
) -> pd.DataFrame:
    """Builds a pandas frame from rows of Python values.

    The row-oriented counterpart of :func:`pandas_frame_from_columns`, for data
    that reads better written out as rows -- a :class:`TableSpec`'s, or a test's
    expected result.

    Args:
        columns: The column names, in order.
        rows: The rows, each holding one value per column, in that order.
        dtypes: The pandas dtype to build each column at.

    Returns:
        The frame.

    Raises:
        ValueError: If a row has the wrong number of values, or a column has no
            dtype, or a column's values do not fit its dtype.
    """
    for index, row in enumerate(rows):
        if len(row) != len(columns):
            raise ValueError(
                f"Row {index} has {len(row)} values, but there are "
                f"{len(columns)} columns."
            )
    by_column: Dict[str, List[Any]] = {
        name: [row[position] for row in rows] for position, name in enumerate(columns)
    }
    return pandas_frame_from_columns(by_column, dtypes)


################################################################################
# Bringing a result back to pandas
################################################################################


def _columns_of_pandas(frame: pd.DataFrame) -> Dict[str, List[Any]]:
    """Returns a pandas frame's values as Python objects, column by column.

    Args:
        frame: The frame to read.

    Returns:
        The values of each column.
    """
    return {
        str(name): list(frame[name].to_numpy(dtype=object)) for name in frame.columns
    }


def _columns_of_spark(frame: SparkDataFrame) -> Dict[str, List[Any]]:
    """Returns a Spark frame's values as Python objects, column by column.

    ``collect()`` is what makes this faithful: it hands back Python objects, so
    a ``NULL`` arrives as ``None`` and a NaN as a float NaN, and an integer is
    still an integer however large.

    Args:
        frame: The frame to read.

    Returns:
        The values of each column.
    """
    names = list(frame.columns)
    rows = frame.collect()
    return {
        name: [row[position] for row in rows] for position, name in enumerate(names)
    }


def to_pandas(
    frame: AnyFrame,
    *,
    schema: Optional[SchemaLike] = None,
    dtypes: Optional[Dtypes] = None,
) -> pd.DataFrame:
    """Returns a backend's frame as pandas, without destroying its values.

    Use this on anything a Session hands back, whichever backend it came from,
    before comparing it to anything. See the module docstring for what
    ``toPandas()`` would do instead.

    The target dtype of each column is taken from, in order of precedence:
    ``dtypes``, then ``schema``, then -- for a Spark frame -- the frame's own
    Spark schema, read through
    :func:`~tmlt.analytics._schema.spark_schema_to_analytics_columns` and
    :func:`~tmlt.analytics._schema.analytics_to_pandas_dtypes`. A column that
    admits nulls therefore lands in a nullable extension dtype, which is what
    keeps a null distinguishable from a NaN, and an empty frame keeps those
    dtypes rather than collapsing to whatever pandas infers from no rows.

    A pandas frame with no ``schema`` and no ``dtypes`` is returned *unchanged*
    -- the same object, not a copy -- since there is nothing to convert.

    Args:
        frame: The frame to convert.
        schema: The Analytics schema the result is expected to have, if the
            frame's own type information should not be trusted or does not
            exist.
        dtypes: The pandas dtype for particular columns, overriding ``schema``
            for those columns. Columns not named here are unaffected.

    Returns:
        The pandas rendering of the frame.

    Raises:
        TypeError: If the frame is of neither backend's type.
        ValueError: If a column's values do not fit the dtype asked for.
    """
    if isinstance(frame, pd.DataFrame):
        if schema is None and dtypes is None:
            return frame
        targets: Dict[str, Any] = dict(frame.dtypes)
        columns = _columns_of_pandas(frame)
    elif isinstance(frame, SparkDataFrame):
        targets = pandas_dtypes_for(spark_schema_to_analytics_columns(frame.schema))
        columns = _columns_of_spark(frame)
    else:
        raise TypeError(f"Not a frame of either backend: {type(frame).__name__}.")
    if schema is not None:
        targets.update(pandas_dtypes_for(schema))
    if dtypes is not None:
        targets.update(dtypes)
    return pandas_frame_from_columns(columns, targets)
