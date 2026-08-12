"""Logic for coercing pandas DataFrames into forms usable by Tumult Analytics.

This is the pandas counterpart of :mod:`~tmlt.analytics._coerce_spark_schema`.
Where Spark has one type per kind of value, pandas has several dtypes that hold
the same values, and one -- ``object`` -- that says nothing about them at all.
This module pins which of them Analytics accepts, which it silently widens, and
what an ``object`` column must contain.

The dtype contract
------------------

Every supported dtype maps onto exactly one
:class:`~tmlt.analytics._schema.ColumnType`, and agrees with the descriptors in
:mod:`tmlt.core.domains.pandas_domains`:

.. list-table::
   :header-rows: 1

   * - pandas dtype
     - Coerced to
     - Analytics type
   * - ``int64``
     - ``int64``
     - ``INTEGER``
   * - ``int8``, ``int16``, ``int32``, ``uint8``, ``uint16``, ``uint32``
     - ``int64``
     - ``INTEGER``
   * - ``Int64``
     - ``Int64``
     - ``INTEGER``
   * - ``Int8``, ``Int16``, ``Int32``
     - ``Int64``
     - ``INTEGER``
   * - ``float64``
     - ``float64``
     - ``DECIMAL``
   * - ``float32``
     - ``float64``
     - ``DECIMAL``
   * - ``Float64``
     - ``Float64``
     - ``DECIMAL``
   * - ``Float32``
     - ``Float64``
     - ``DECIMAL``
   * - ``object`` of :class:`str`
     - ``object``
     - ``VARCHAR``
   * - ``string``
     - ``object``
     - ``VARCHAR``
   * - ``object`` of :class:`datetime.date`
     - ``object``
     - ``DATE``
   * - timezone-naive ``datetime64``
     - unchanged
     - ``TIMESTAMP``
   * - anything else
     - rejected
     - --

``uint64`` is rejected rather than widened: its upper half does not fit in an
``int64``, and ``astype`` would wrap those values silently. ``bool``,
``category``, ``timedelta64``, ``period``, ``interval``, ``complex``, and
timezone-aware ``datetime64`` are rejected outright, as are ``object`` columns
holding anything other than strings-or-nulls or dates-or-nulls.

Widening preserves nullability rather than erasing it: a numpy integer column
widens to the numpy ``int64``, and a nullable extension column widens to
``Int64``, because which of the two a column uses is exactly what says whether
it can hold a null (see below). Timezone-naive ``datetime64`` columns are left
in whatever unit they arrived in -- pandas 2 supports ``s``, ``ms``, ``us`` and
``ns``, and
:class:`~tmlt.core.domains.pandas_domains.PandasTimestampColumnDescriptor`
accepts all of them. Converting the wider units to the canonical ``ns`` would
raise on any timestamp outside 1677--2262, which is the reason those units
exist.

The nullability decision
------------------------

Spark and pandas disagree about nullability, and Analytics resolves the
disagreement in favour of pandas.

``SparkSession.createDataFrame(pdf)`` marks *every* field it infers as
``nullable=True``, whatever the pandas dtype it came from -- a plain numpy
``int64`` column, which cannot represent a null at all, included. So
:func:`~tmlt.analytics._schema.spark_schema_to_analytics_columns` applied to an
inferred Spark schema reports ``allow_null=True`` everywhere.

:func:`~tmlt.analytics._schema.pandas_dtypes_to_analytics_columns` is instead
precise: it reports ``allow_null=True`` exactly when the dtype can represent a
null. A numpy ``int64`` or ``float64`` column cannot -- in a numpy float column
a NaN is a NaN, not a null -- and so gets ``allow_null=False``; a nullable
extension dtype (``Int64``, ``Float64``), ``object`` (which holds ``None``), and
``datetime64`` (which holds ``NaT``) all can, and get ``allow_null=True``.

The documented cross-backend relationship is therefore:

    the two backends infer equal
    :class:`~tmlt.analytics._schema.ColumnDescriptor` objects, except that
    ``allow_null`` may be ``False`` on pandas where Spark inference says
    ``True``. It is never ``True`` on pandas and ``False`` on Spark.

In practice that means the two differ only for numpy ``int64`` and ``float64``
columns, and agree exactly for everything else. ``test_pandas_schema_conversion``
pins this per dtype.

Two things follow from deciding it this way. Being precise is strictly more
information, and it is information a downstream query can use: a column that
genuinely cannot be null does not need the null-handling that Spark's blanket
``nullable=True`` would force on it. And the answer depends only on the dtype,
never on the values, so two frames with the same dtypes always get the same
schema. That second property matters more than it looks: a schema derived by
scanning for nulls would differ between one batch of data and the next, and
would itself reveal whether the data contains a null.

The same reasoning fixes ``allow_nan`` and ``allow_inf``, which
:func:`~tmlt.analytics._schema.pandas_dtypes_to_analytics_columns` reports as
``True`` for every ``DECIMAL`` column without looking at the values -- matching
what :func:`~tmlt.analytics._schema.spark_schema_to_analytics_columns` does with
a Spark schema, which carries no such information to begin with.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from typing import Collection, Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd
from tmlt.core.domains.pandas_domains import (
    PandasDtype,
    PandasTimestampColumnDescriptor,
)


def string_dtypes() -> Tuple[PandasDtype, ...]:
    """Returns every pandas ``string`` extension dtype constructible here.

    That dtype has a storage backend -- ``python``, plus ``pyarrow`` when
    pyarrow is installed -- and two ``string`` dtypes with different storage are
    not equal to each other, so both have to be listed for a lookup by dtype to
    find either. They coerce to the same ``object`` column regardless.
    """
    dtypes: List[PandasDtype] = [pd.StringDtype("python")]
    try:
        dtypes.append(pd.StringDtype("pyarrow"))
    except (ImportError, ValueError):  # pragma: no cover - depends on pyarrow
        pass
    return tuple(dtypes)


CANONICAL_PANDAS_DTYPES: FrozenSet[PandasDtype] = frozenset(
    {
        np.dtype("int64"),
        pd.Int64Dtype(),
        np.dtype("float64"),
        pd.Float64Dtype(),
        np.dtype(object),
        *PandasTimestampColumnDescriptor().accepted_dtypes,
    }
)
"""The dtypes a coerced DataFrame's columns may have.

These are the dtypes :func:`coerce_pandas_schema_or_fail` produces, and the ones
the descriptors in :mod:`tmlt.core.domains.pandas_domains` accept. The
``datetime64`` entries are taken from
:attr:`~tmlt.core.domains.pandas_domains.PandasTimestampColumnDescriptor.accepted_dtypes`
so that the two cannot drift apart; on pandas 1 only the ``ns`` unit is
reachable, since pandas 1 stores every datetime column in it.
"""

TYPE_COERCION_MAP: Dict[PandasDtype, PandasDtype] = {
    # Narrower integers widen to the 64-bit integer dtype of the same
    # nullability: a numpy column stays numpy, an extension column stays an
    # extension column. uint64 is absent deliberately -- see the module
    # docstring.
    np.dtype("int8"): np.dtype("int64"),
    np.dtype("int16"): np.dtype("int64"),
    np.dtype("int32"): np.dtype("int64"),
    np.dtype("uint8"): np.dtype("int64"),
    np.dtype("uint16"): np.dtype("int64"),
    np.dtype("uint32"): np.dtype("int64"),
    pd.Int8Dtype(): pd.Int64Dtype(),
    pd.Int16Dtype(): pd.Int64Dtype(),
    pd.Int32Dtype(): pd.Int64Dtype(),
    # Same for floats.
    np.dtype("float32"): np.dtype("float64"),
    pd.Float32Dtype(): pd.Float64Dtype(),
    # pandas' string extension dtype becomes an object column of str, which is
    # the single representation Core's string descriptor accepts. Its pd.NA
    # nulls become None on the way; see _to_object_column.
    **{dtype: np.dtype(object) for dtype in string_dtypes()},
}
"""Mapping describing how pandas dtypes are coerced by Tumult Analytics."""

SUPPORTED_PANDAS_DTYPES: FrozenSet[PandasDtype] = CANONICAL_PANDAS_DTYPES | frozenset(
    TYPE_COERCION_MAP
)
"""Set of pandas dtypes supported by Tumult Analytics.

A column with one of these dtypes is accepted; one whose dtype is in
:data:`TYPE_COERCION_MAP` is first widened to the dtype it maps to.
"""

_OBJECT_ELEMENT_TYPES: Tuple[type, ...] = (str, datetime.date)
"""The types an ``object`` column's non-null values may have.

A column's values must all be of one of these, not a mixture of both. Note that
:class:`datetime.datetime` is a :class:`datetime.date` by subclassing but is not
one of these types; see
:class:`~tmlt.core.domains.pandas_domains.PandasDateColumnDescriptor` for why it
is rejected.
"""


def _format_dtypes(dtypes: Collection[PandasDtype]) -> str:
    """Returns a deterministic, readable rendering of a collection of dtypes."""
    return ", ".join(sorted({str(dtype) for dtype in dtypes}))


def _unsupported_dtype_hint(column_name: str, dtype: PandasDtype) -> str:
    """Returns advice for a column with an unsupported dtype, or "" if there is none."""
    if isinstance(dtype, pd.DatetimeTZDtype):
        return (
            f" Column '{column_name}' is timezone-aware; Analytics timestamps are"
            " timezone-naive, so convert it with"
            f" df['{column_name}'].dt.tz_convert('UTC').dt.tz_localize(None) to get"
            " naive UTC timestamps."
        )
    if isinstance(dtype, pd.CategoricalDtype):
        return (
            f" Column '{column_name}' is categorical; convert it with"
            f" df['{column_name}'].astype(object) to keep its values."
        )
    if dtype == np.dtype(bool) or isinstance(dtype, pd.BooleanDtype):
        return (
            f" Column '{column_name}' is a boolean column; convert it with"
            f" df['{column_name}'].astype('int64') to keep it as an INTEGER column."
        )
    if dtype == np.dtype("uint64") or isinstance(dtype, pd.UInt64Dtype):
        return (
            f" Column '{column_name}' is unsigned 64-bit, whose largest values do not"
            " fit in an int64; convert it with"
            f" df['{column_name}'].astype('int64') once you have checked that its"
            " values do fit."
        )
    return ""


def _fail_if_dataframe_has_invalid_column_names(dataframe: pd.DataFrame) -> None:
    """Raises an error if the DataFrame's column names are unusable.

    pandas, unlike Spark, allows a column to be named by any hashable object and
    allows two columns to share a name. Analytics column names are strings, and
    a duplicated name makes ``dataframe[name]`` a DataFrame rather than a
    Series, so both are rejected here.
    """
    columns = list(dataframe.columns)

    non_string = [column for column in columns if not isinstance(column, str)]
    if non_string:
        raise ValueError(
            "This DataFrame has column names that are not strings:"
            f" {non_string}. Rename them with DataFrame.rename before loading it."
        )

    if "" in columns:
        raise ValueError('This DataFrame contains a column named "" (the empty string)')

    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(
            f"This DataFrame contains duplicate column names: {duplicates}. Rename"
            " them so that every column has a distinct name."
        )


def _fail_if_dataframe_contains_unsupported_dtypes(dataframe: pd.DataFrame) -> None:
    """Raises an error if DataFrame contains unsupported pandas dtypes."""
    unsupported = [
        (str(name), dataframe[name].dtype)
        for name in dataframe.columns
        if dataframe[name].dtype not in SUPPORTED_PANDAS_DTYPES
    ]

    if unsupported:
        hints = "".join(
            _unsupported_dtype_hint(name, dtype) for name, dtype in unsupported
        )
        raise ValueError(
            "Unsupported pandas dtype: Tumult Analytics does not support the pandas"
            " dtypes of the following columns:"
            f" {[(name, str(dtype)) for name, dtype in unsupported]}."
            + hints
            + " Consider converting these columns into one of the supported pandas"
            f" dtypes: {_format_dtypes(SUPPORTED_PANDAS_DTYPES)}."
        )


def object_column_element_type(column: pd.Series, column_name: str) -> type:
    """Returns the type of the non-null values of an ``object`` column.

    An ``object`` column's dtype says nothing about what it holds, so this makes
    one pass over its non-null values and requires them to be all
    :class:`str` (or a subclass of it, such as :class:`numpy.str_`), giving a
    ``VARCHAR`` column, or all exactly :class:`datetime.date`, giving a ``DATE``
    column. Nulls are whatever :meth:`pandas.Series.isna` reports: ``None``,
    ``float("nan")`` and ``pd.NA``.

    A column with no non-null values to inspect -- one that is empty, or all
    null -- is taken to be a :class:`str` column, since ``None`` is how a
    missing string is written and there is nothing else to go on.

    Args:
        column: The column to inspect. Must have the ``object`` dtype.
        column_name: The name to use in error messages.

    Raises:
        ValueError: If the column's values are of any other type, or of more
            than one of these types.
    """
    element_types = column[~column.isna()].map(type).unique()

    for candidate in _OBJECT_ELEMENT_TYPES:
        if all(
            _is_valid_element_type(element_type, candidate)
            for element_type in element_types
        ):
            return candidate

    offenders = sorted(
        {
            f"{element_type.__module__}.{element_type.__qualname__}"
            for element_type in element_types
        }
    )
    message = (
        f"Unsupported values in object column: column '{column_name}' must contain"
        " only str values, or only datetime.date values, with None for missing"
        f" values; it contains {', '.join(offenders)}."
    )
    if any(element_type is datetime.datetime for element_type in element_types):
        message += (
            " A datetime.datetime carries a time of day that a DATE column cannot"
            f" represent; convert the column with df['{column_name}'].map("
            "datetime.datetime.date) to make it a DATE column, or with"
            f" pd.to_datetime(df['{column_name}']) to make it a TIMESTAMP column."
        )
    raise ValueError(message)


def _is_valid_element_type(element_type: type, candidate: type) -> bool:
    """Returns True if ``element_type`` is acceptable for a ``candidate`` column.

    A :class:`str` column accepts subclasses of :class:`str`, matching
    :class:`~tmlt.core.domains.pandas_domains.PandasStringColumnDescriptor`. A
    :class:`datetime.date` column accepts only :class:`datetime.date` itself,
    matching
    :class:`~tmlt.core.domains.pandas_domains.PandasDateColumnDescriptor`.
    """
    if candidate is datetime.date:
        return element_type is datetime.date
    return issubclass(element_type, candidate)


def _to_object_column(column: pd.Series) -> pd.Series:
    """Returns ``column`` as an ``object`` column whose nulls are all ``None``.

    pandas has three values that :meth:`pandas.Series.isna` calls null --
    ``None``, ``float("nan")`` and ``pd.NA`` -- and an object column can hold
    any of them. ``pd.NA`` in particular arrives whenever a ``string`` column is
    converted with ``astype(object)``, and it does not behave like the other
    two: comparing it propagates ``NA`` instead of returning ``False``, so a
    filter or a join against such a column silently drops rows rather than
    treating the null as unequal. Normalizing every null to ``None`` gives an
    object column one spelling of "missing", the one Spark's converters also
    produce.
    """
    values = column.to_numpy(dtype=object, copy=True)
    values[column.isna().to_numpy()] = None
    return pd.Series(values, index=column.index, name=column.name, dtype=object)


def coerce_pandas_schema_or_fail(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Returns a new DataFrame where all column dtypes are supported.

    In particular, this function raises an error:
        * if ``dataframe`` has a column name that is not a string, is ``""``
          (the empty string), or is shared with another column
        * if ``dataframe`` contains a column whose dtype is not listed in
          :data:`SUPPORTED_PANDAS_DTYPES`
        * if ``dataframe`` contains an ``object`` column whose values are not
          all strings-or-nulls or all dates-or-nulls

    This function returns a DataFrame where all column dtypes
        * are coerced according to :data:`TYPE_COERCION_MAP` if necessary
        * have ``None`` -- rather than ``pd.NA`` or ``float("nan")`` -- as the
          null in an ``object`` column

    The returned DataFrame is always a **copy**, even when nothing needed
    coercing, and it never shares buffers with the caller's frame. This is the
    copy-on-ingest privacy boundary: from here on the data belongs to the
    Session, and neither can be changed by writing to the other. A pandas
    DataFrame is mutable and the caller keeps a reference to theirs, so without
    the copy a caller could alter data the Session had already spent privacy
    budget on -- or observe, through their own frame, a transformation the
    Session performed. :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`
    documents the matching convention on the Core side: a component may read the
    frame it is given, and copies it before writing.

    Args:
        dataframe: The DataFrame to coerce. It is not modified.

    Raises:
        ValueError: If the DataFrame cannot be coerced, as described above.
    """
    _fail_if_dataframe_has_invalid_column_names(dataframe)
    _fail_if_dataframe_contains_unsupported_dtypes(dataframe)

    coerced = dataframe.copy()
    errors: List[str] = []
    for name in list(coerced.columns):
        column = coerced[name]
        target_dtype = TYPE_COERCION_MAP.get(column.dtype, column.dtype)
        if target_dtype == np.dtype(object):
            object_column = (
                column if column.dtype == np.dtype(object) else column.astype(object)
            )
            try:
                object_column_element_type(object_column, str(name))
            except ValueError as error:
                errors.append(str(error))
                continue
            coerced[name] = _to_object_column(object_column)
        elif target_dtype != column.dtype:
            coerced[name] = column.astype(target_dtype)

    if errors:
        raise ValueError(" ".join(errors))

    return coerced
