"""Schema management for private and public tables.

The schema represents the column types of the underlying table. This allows
for seamless transitions of the data representation type.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping as MappingType,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tmlt.core.domains.base import Domain
from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasDtype,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from typeguard import check_type, typechecked

from tmlt.analytics._coerce_pandas_schema import (
    object_column_element_type,
    string_dtypes,
)


class keyValuePair(NamedTuple):
    """A key-value pair for the FrozenDict class."""

    key: Hashable
    value: Hashable


@dataclass(frozen=True)
class FrozenDict(Mapping):
    """A mapping that is immutable and hashable.

    Like Python's built-in dict, FrozenDict maintains the order of elements when
    iterating and converting to/from dict. *Unlike* Python's built-in dict,
    FrozenDict considers this ordering when checking for equality -- two
    FrozenDicts are only equal if they have the same key-value pairs *in the
    same order*.

    This is needed to replace the mutable mappings in some QueryExprs and other
    immutable objects.

    Note: This will not have the same performance characteristics as a normal dict.
    Mainly, really large FrozenDicts will be slower than dicts to access. This is not
    a major concern at the moment though because all use cases have a key referring
    to a single column, and we don't expect users to have dataframes with large
    quantities of columns.
    """

    elements: Tuple[keyValuePair, ...]

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.elements, Tuple[keyValuePair, ...])

    @staticmethod
    @typechecked
    def from_dict(dictionary: MappingType[Any, Any]) -> "FrozenDict":
        """Returns a FrozenDict from a dictionary."""
        elements = tuple(keyValuePair(key, value) for key, value in dictionary.items())
        return FrozenDict(elements)

    def __len__(self) -> int:
        """Returns the number of elements in the FrozenDict."""
        return len(self.elements)

    def __iter__(self):
        """Returns each key in the FrozenDict."""
        for element in self.elements:
            yield element.key

    def __getitem__(self, key):
        """Returns the value of the key if it exists, otherwise raise KeyError."""
        for pair in self.elements:
            if pair.key == key:
                return pair.value
        raise KeyError(key)

    def __eq__(self, other):
        """Determines if two FrozenDicts are equal based on their elements."""
        if not isinstance(other, FrozenDict):
            return False
        return self.elements == other.elements

    def __hash__(self):
        """Hashes a FrozenDict based on the tuple of elements."""
        return hash(self.elements)

    def __or__(self, other):
        """Merge a dict or FrozenDict with this one, producing a new FrozenDict."""
        if not isinstance(other, (dict, FrozenDict)):
            raise TypeError(
                "unsupported operand type(s) for |: "
                f"'{type(self).__name__}' and '{type(other).__name__}'"
            )
        return FrozenDict.from_dict(dict(self) | dict(other))


class ColumnType(Enum):
    """The supported SQL92 column types used by Tumult Analytics.

    Support for Spark data types is currently as follows.

    .. list-table::
       :header-rows: 1

       * - Spark type
         - Corresponding Tumult Analytics type
       * - :class:`~pyspark.sql.types.LongType`
         - ``INTEGER``
       * - :class:`~pyspark.sql.types.IntegerType`
         - ``INTEGER``
       * - :class:`~pyspark.sql.types.DoubleType`
         - ``DECIMAL``
       * - :class:`~pyspark.sql.types.FloatType`
         - ``DECIMAL``
       * - :class:`~pyspark.sql.types.StringType`
         - ``VARCHAR``
       * - :class:`~pyspark.sql.types.DateType`
         - ``DATE``
       * - :class:`~pyspark.sql.types.TimestampType`
         - ``TIMESTAMP``
       * - Other Spark types
         - Not supported in Tumult Analytics

    Columns with unsupported types must be removed or converted to supported ones before
    loading the data into a :class:`~tmlt.analytics.Session`.
    """

    INTEGER = int
    """Integer column type."""
    DECIMAL = float
    """Floating-point column type."""
    VARCHAR = str
    """String column type."""
    DATE = datetime.date
    """Date column type."""
    TIMESTAMP = datetime.datetime
    """Timestamp column type."""

    def __str__(self) -> str:
        """Return a printable version of a ColumnType."""
        return str(self.name)

    def __repr__(self) -> str:
        """Return a string representation of a ColumnType."""
        return "ColumnType." + self.name


@dataclass(frozen=True)
class ColumnDescriptor:
    """Information about a column.

    ColumnDescriptors have the following attributes:

    Attributes:
        column_type: A :class:`ColumnType`, specifying what type this column has.
        allow_null: :class:`bool`. If ``True``, this column allows null values.
        allow_nan: :class:`bool`. If ``True``, this column allows NaN values.
        allow_inf: :class:`bool`. If ``True``, this column allows infinite values.
    """

    column_type: ColumnType
    allow_null: bool = False
    allow_nan: bool = False
    allow_inf: bool = False


class Schema(Mapping):
    """Schema class describing the column information of the data.

    The following SQL92 types are currently supported:
      INTEGER, DECIMAL, VARCHAR, DATE, TIMESTAMP
    """

    def __init__(
        self,
        column_descs: MappingType[str, Union[str, ColumnType, ColumnDescriptor]],
        grouping_column: Optional[str] = None,
        id_column: Optional[str] = None,
        id_space: Optional[str] = None,
        default_allow_null: bool = False,
        default_allow_nan: bool = False,
        default_allow_inf: bool = False,
    ):
        """Constructor.

        Args:
            column_descs: Mapping from column names to supported types.
            grouping_column: Optional column that must be grouped by in this query.
            id_column: The ID column on this table, if one exists.
            id_space: The ID space for this table, if one exists.
            default_allow_null: When a ColumnType or string is used as the value
                in the ColumnDescriptors mapping, the column will allow_null if
                default_allow_null is True.
            default_allow_nan: When a ColumnType or string is used as the value
                in the ColumnDescriptors mapping, the column will allow_nan if
                default_allow_nan is True.
            default_allow_inf: When a ColumnType or string is used as the value
                in the ColumnDescriptors mapping, the column will allow_inf if
                default_allow_inf is True.
        """
        # TODO(#1539): update Schema interface to use ColumnDescriptor everywhere.
        if "" in column_descs:
            raise ValueError('"" (the empty string) is not a supported column name')
        if grouping_column is not None and grouping_column not in column_descs:
            raise KeyError(
                f"Grouping column '{grouping_column}' is not one of the provided"
                " columns"
            )
        self._grouping_column = grouping_column
        if id_column is not None and id_column not in column_descs:
            raise KeyError(
                f"ID column '{id_column}' is not one of the provided columns"
            )
        self._id_column = id_column
        self._id_space = id_space

        supported_types: List[str] = [t.name for t in list(ColumnType)]
        column_types: List[str] = []
        for cd in column_descs.values():
            if isinstance(cd, ColumnDescriptor):
                column_types.append(cd.column_type.name)
            elif isinstance(cd, ColumnType):
                column_types.append(cd.name)
            else:
                column_types.append(cd)
        invalid_types = set(column_types) - set(supported_types)
        if invalid_types:
            raise ValueError(
                f"Column types {invalid_types} not supported; "
                f"use supported types {supported_types}."
            )

        updated_column_types: Dict[str, ColumnDescriptor] = {}
        for col, ty in column_descs.items():
            if isinstance(ty, ColumnDescriptor):
                updated_column_types[col] = ty
            elif isinstance(ty, ColumnType):
                updated_column_types[col] = ColumnDescriptor(
                    ty,
                    allow_null=default_allow_null,
                    allow_nan=default_allow_nan,
                    allow_inf=default_allow_inf,
                )
            else:
                updated_column_types[col] = ColumnDescriptor(
                    column_type=ColumnType[ty],
                    allow_null=default_allow_null,
                    allow_nan=default_allow_nan,
                    allow_inf=default_allow_inf,
                )
        self._column_descs: FrozenDict = FrozenDict.from_dict(updated_column_types)

    @property
    def columns(self):
        """Return the names of the columns in the schema."""
        return dict(self._column_descs).keys()

    @property
    def column_descs(self) -> Dict[str, ColumnDescriptor]:
        """Returns a mapping from column name to column descriptor."""
        return dict(self._column_descs)

    @property
    def column_types(self) -> Dict[str, str]:
        """Returns a mapping from column name to column type."""
        # TODO(#1539): Remove this
        return {col: desc.column_type.name for col, desc in self.column_descs.items()}

    @property
    def grouping_column(self) -> Optional[str]:
        """Returns the optional column that must be grouped by."""
        return self._grouping_column

    @property
    def id_column(self) -> Optional[str]:
        """Return whether the grouping column is an ID column."""
        return self._id_column

    @property
    def id_space(self) -> Optional[str]:
        """Return the ID space for this schema."""
        return self._id_space

    def __eq__(self, other: object) -> bool:
        """Returns True if schemas are equal.

        Args:
            other: Schema to check against.
        """
        if isinstance(other, Schema):
            return (
                self.column_descs == other.column_descs
                and self.grouping_column == other.grouping_column
                and self.id_column == other.id_column
                and self.id_space == other.id_space
            )
        return False

    def __getitem__(self, column: str) -> ColumnDescriptor:
        """Returns the data type for the given column.

        Args:
            column: The column to get the data type for.
        """
        return self.column_descs[column]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the columns in the schema."""
        return iter(self.column_descs)

    def __len__(self) -> int:
        """Return the number of columns in the schema."""
        return len(self.column_descs)

    def __repr__(self) -> str:
        """Return a string representation of self."""
        out = f"Schema({self.column_descs}"
        if self.grouping_column:
            out += f", grouping_column={self.grouping_column}"
        if self.id_column:
            out += f", id_column={self.id_column}"
        if self.id_space:
            out += f", id_space={self.id_space}"
        out += ")"
        return out

    def __hash__(self) -> int:
        """Hashes the schema based on the column_descs mapping.

        This function is sound, it relies on the equality of schemas being complete.
        """
        return hash(self._column_descs)


_SPARK_TO_ANALYTICS: Dict[DataType, ColumnType] = {
    IntegerType(): ColumnType.INTEGER,
    LongType(): ColumnType.INTEGER,
    DoubleType(): ColumnType.DECIMAL,
    FloatType(): ColumnType.DECIMAL,
    StringType(): ColumnType.VARCHAR,
    DateType(): ColumnType.DATE,
    TimestampType(): ColumnType.TIMESTAMP,
}
"""Mapping from Spark type to supported Analytics column types."""

_ANALYTICS_TO_SPARK = {
    "INTEGER": LongType(),
    "DECIMAL": DoubleType(),
    "VARCHAR": StringType(),
    "DATE": DateType(),
    "TIMESTAMP": TimestampType(),
}
"""Mapping from Analytics column types to Spark types."""

_ANALYTICS_TYPE_TO_COLUMN_DESCRIPTOR = {
    ColumnType.INTEGER: SparkIntegerColumnDescriptor,
    ColumnType.DECIMAL: SparkFloatColumnDescriptor,
    ColumnType.VARCHAR: SparkStringColumnDescriptor,
    ColumnType.DATE: SparkDateColumnDescriptor,
    ColumnType.TIMESTAMP: SparkTimestampColumnDescriptor,
}
"""Mapping from Analytics column types to Spark columns descriptor.

More information regarding Spark columns descriptor can be found in
:class:`~tmlt.core.domains.spark_domains.SparkColumnDescriptor`"""

_PANDAS_TO_ANALYTICS_ELEMENT_TYPE: Dict[type, ColumnType] = {
    str: ColumnType.VARCHAR,
    datetime.date: ColumnType.DATE,
}
"""Mapping from the element type of an ``object`` column to its Analytics type.

The keys are what :func:`~tmlt.analytics._coerce_pandas_schema.object_column_element_type`
returns."""

_PANDAS_DTYPE_TO_ANALYTICS: Dict[PandasDtype, ColumnType] = {
    np.dtype("int8"): ColumnType.INTEGER,
    np.dtype("int16"): ColumnType.INTEGER,
    np.dtype("int32"): ColumnType.INTEGER,
    np.dtype("int64"): ColumnType.INTEGER,
    np.dtype("uint8"): ColumnType.INTEGER,
    np.dtype("uint16"): ColumnType.INTEGER,
    np.dtype("uint32"): ColumnType.INTEGER,
    pd.Int8Dtype(): ColumnType.INTEGER,
    pd.Int16Dtype(): ColumnType.INTEGER,
    pd.Int32Dtype(): ColumnType.INTEGER,
    pd.Int64Dtype(): ColumnType.INTEGER,
    np.dtype("float32"): ColumnType.DECIMAL,
    np.dtype("float64"): ColumnType.DECIMAL,
    pd.Float32Dtype(): ColumnType.DECIMAL,
    pd.Float64Dtype(): ColumnType.DECIMAL,
    **{dtype: ColumnType.VARCHAR for dtype in string_dtypes()},
    **{
        dtype: ColumnType.TIMESTAMP
        for dtype in PandasTimestampColumnDescriptor().accepted_dtypes
    },
}
"""Mapping from pandas dtype to supported Analytics column types.

Like :data:`_SPARK_TO_ANALYTICS`, this covers the dtypes that
:func:`~tmlt.analytics._coerce_pandas_schema.coerce_pandas_schema_or_fail`
accepts, not only the ones it produces, so that it can be applied to an
uncoerced DataFrame.

The ``object`` dtype is deliberately absent: it may be either a ``VARCHAR`` or a
``DATE`` column, and only its values say which. See
:func:`~tmlt.analytics._coerce_pandas_schema.object_column_element_type`."""

_ANALYTICS_TYPE_TO_PANDAS_COLUMN_DESCRIPTOR: Dict[ColumnType, type] = {
    ColumnType.INTEGER: PandasIntegerColumnDescriptor,
    ColumnType.DECIMAL: PandasFloatColumnDescriptor,
    ColumnType.VARCHAR: PandasStringColumnDescriptor,
    ColumnType.DATE: PandasDateColumnDescriptor,
    ColumnType.TIMESTAMP: PandasTimestampColumnDescriptor,
}
"""Mapping from Analytics column types to pandas columns descriptor.

More information regarding pandas columns descriptor can be found in
:class:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor`"""

_PANDAS_COLUMN_DESCRIPTOR_TO_ANALYTICS: Dict[type, ColumnType] = {
    descriptor: column_type
    for column_type, descriptor in _ANALYTICS_TYPE_TO_PANDAS_COLUMN_DESCRIPTOR.items()
}
"""Mapping from pandas columns descriptor to Analytics column types.

Derived from :data:`_ANALYTICS_TYPE_TO_PANDAS_COLUMN_DESCRIPTOR` rather than
written out, so that the two cannot come to disagree. It is keyed by descriptor
class rather than by dtype, as the Spark family is, because the string and date
descriptors both describe an ``object`` column."""


def column_type_to_py_type(column_type: ColumnType) -> type:
    """Converts a ColumnType to a python type."""
    return column_type.value


def analytics_to_py_types(analytics_schema: Schema) -> Dict[str, type]:
    """Returns the mapping from column names to supported python types."""
    return {
        column_name: ColumnType[column_desc.column_type.name].value
        for column_name, column_desc in analytics_schema.column_descs.items()
    }


def analytics_to_spark_schema(analytics_schema: Schema) -> StructType:
    """Convert an Analytics schema to a Spark schema."""
    return StructType(
        [
            StructField(
                column_name,
                _ANALYTICS_TO_SPARK[column_desc.column_type.name],
                nullable=column_desc.allow_null,
            )
            for column_name, column_desc in analytics_schema.column_descs.items()
        ]
    )


def analytics_to_spark_columns_descriptor(
    analytics_schema: Schema,
) -> SparkColumnsDescriptor:
    """Convert a schema in Analytics representation to a Spark columns descriptor."""
    out: Dict[str, SparkColumnDescriptor] = {}
    for column_name, column_desc in analytics_schema.column_descs.items():
        if column_desc.column_type == ColumnType.DECIMAL:
            out[column_name] = SparkFloatColumnDescriptor(
                allow_nan=column_desc.allow_nan,
                allow_inf=column_desc.allow_inf,
                allow_null=column_desc.allow_null,
            )
        else:
            out[column_name] = _ANALYTICS_TYPE_TO_COLUMN_DESCRIPTOR[
                ColumnType[column_desc.column_type.name]
            ](allow_null=column_desc.allow_null)
    return out


def spark_schema_to_analytics_columns(
    spark_schema: StructType,
) -> Dict[str, ColumnDescriptor]:
    """Convert Spark schema to Analytics columns."""
    column_descs = {
        field.name: ColumnDescriptor(
            column_type=_SPARK_TO_ANALYTICS[field.dataType],
            allow_null=field.nullable,
            # Spark doesn't contain any information on whether a field contains NaNs,
            # so just assume that it does
            allow_nan=(_SPARK_TO_ANALYTICS[field.dataType] == ColumnType.DECIMAL),
            # Same for infinite values
            allow_inf=(_SPARK_TO_ANALYTICS[field.dataType] == ColumnType.DECIMAL),
        )
        for field in spark_schema
    }
    return column_descs


def spark_dataframe_domain_to_analytics_columns(
    domain: Domain,
) -> Dict[str, ColumnDescriptor]:
    """Convert a Spark dataframe domain to Analytics columns."""
    column_descs: Dict[str, ColumnDescriptor] = {}
    for column_name, descriptor in cast(SparkDataFrameDomain, domain).schema.items():
        if isinstance(descriptor, SparkFloatColumnDescriptor):
            column_descs[column_name] = ColumnDescriptor(
                ColumnType.DECIMAL,
                allow_null=descriptor.allow_null,
                allow_nan=descriptor.allow_nan,
                allow_inf=descriptor.allow_inf,
            )
        else:
            column_descs[column_name] = ColumnDescriptor(
                column_type=_SPARK_TO_ANALYTICS[descriptor.data_type],
                allow_null=descriptor.allow_null,
            )
    return column_descs


def analytics_to_pandas_columns_descriptor(
    analytics_schema: Schema,
) -> Dict[str, PandasColumnDescriptor]:
    """Convert a schema in Analytics representation to a pandas columns descriptor.

    This is the pandas counterpart of
    :func:`analytics_to_spark_columns_descriptor`, and describes the same values:
    for every column, ``descriptor.to_spark_descriptor()`` equals what that
    function produces for it.
    """
    out: Dict[str, PandasColumnDescriptor] = {}
    for column_name, column_desc in analytics_schema.column_descs.items():
        if column_desc.column_type == ColumnType.DECIMAL:
            out[column_name] = PandasFloatColumnDescriptor(
                allow_nan=column_desc.allow_nan,
                allow_inf=column_desc.allow_inf,
                allow_null=column_desc.allow_null,
            )
        else:
            out[column_name] = _ANALYTICS_TYPE_TO_PANDAS_COLUMN_DESCRIPTOR[
                ColumnType[column_desc.column_type.name]
            ](allow_null=column_desc.allow_null)
    return out


def analytics_to_pandas_dtypes(analytics_schema: Schema) -> Dict[str, PandasDtype]:
    """Convert an Analytics schema to the pandas dtype of each of its columns.

    This is the pandas counterpart of :func:`analytics_to_spark_schema`. The
    dtype returned for a column is the canonical one for it -- the dtype a
    DataFrame built to match this schema should use -- which for an integer or
    a decimal column depends on whether it allows nulls, as only the pandas
    nullable extension dtypes can hold one. A column may validly have another
    dtype; see
    :attr:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.accepted_dtypes`.

    Which dtype is canonical is Core's answer, not one restated here: this asks
    the descriptor :func:`analytics_to_pandas_columns_descriptor` builds, which
    is the descriptor the domain will validate the column against.
    """
    return {
        column_name: descriptor.pandas_dtype
        for column_name, descriptor in analytics_to_pandas_columns_descriptor(
            analytics_schema
        ).items()
    }


def pandas_dtypes_to_analytics_columns(
    dataframe: pd.DataFrame,
) -> Dict[str, ColumnDescriptor]:
    """Convert the dtypes of a pandas DataFrame to Analytics columns.

    This is the pandas counterpart of :func:`spark_schema_to_analytics_columns`,
    and takes a DataFrame rather than a schema because pandas has none: an
    ``object`` column's values are the only thing that says whether it is a
    ``VARCHAR`` or a ``DATE`` column.

    Only the *types* are inspected, never the values -- apart from the one pass
    over an ``object`` column that is unavoidable. In particular:

    * ``allow_null`` says whether the dtype **can** represent a null, not
      whether the column happens to contain one. The nullable extension dtypes,
      ``object`` and ``datetime64`` can; the numpy integer and float dtypes
      cannot, and in a numpy float column a NaN is a NaN rather than a null.
    * ``allow_nan`` and ``allow_inf`` are ``True`` for every ``DECIMAL`` column,
      as they are in :func:`spark_schema_to_analytics_columns`, which has no
      information about them at all. Scanning for them would make the schema of
      a table depend on its contents.

    :mod:`~tmlt.analytics._coerce_pandas_schema` documents why, and what this
    implies for the relationship between the two backends' inferred schemas.

    Args:
        dataframe: The DataFrame to describe. Its dtypes must be supported; see
            :data:`~tmlt.analytics._coerce_pandas_schema.SUPPORTED_PANDAS_DTYPES`.

    Raises:
        ValueError: If a column's dtype is not supported, or an ``object``
            column's values are not all strings-or-nulls or all dates-or-nulls.
    """
    column_descs: Dict[str, ColumnDescriptor] = {}
    for column_name in dataframe.columns:
        column = dataframe[column_name]
        dtype = column.dtype
        if dtype == np.dtype(object):
            column_type = _PANDAS_TO_ANALYTICS_ELEMENT_TYPE[
                object_column_element_type(column, str(column_name))
            ]
        elif dtype in _PANDAS_DTYPE_TO_ANALYTICS:
            column_type = _PANDAS_DTYPE_TO_ANALYTICS[dtype]
        else:
            raise ValueError(
                f"Unsupported pandas dtype: column '{column_name}' has dtype"
                f" {dtype}, which Tumult Analytics does not support."
            )
        column_descs[column_name] = ColumnDescriptor(
            column_type=column_type,
            allow_null=_dtype_allows_null(dtype),
            # pandas doesn't contain any information on whether a column
            # contains NaNs, so just assume that it does
            allow_nan=(column_type == ColumnType.DECIMAL),
            # Same for infinite values
            allow_inf=(column_type == ColumnType.DECIMAL),
        )
    return column_descs


def _dtype_allows_null(dtype: PandasDtype) -> bool:
    """Returns True if a column of this dtype can represent a null value.

    A pandas nullable extension dtype carries a mask, an ``object`` column can
    hold ``None``, and a ``datetime64`` column can hold ``NaT``. A numpy integer
    column has nowhere to put a null, and a numpy float column's missing-looking
    value is a NaN, which
    :class:`~tmlt.core.domains.pandas_domains.PandasFloatColumnDescriptor` gates
    with ``allow_nan`` rather than with ``allow_null``.
    """
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return True
    return dtype == np.dtype(object) or np.issubdtype(dtype, np.datetime64)


def pandas_dataframe_domain_to_analytics_columns(
    domain: Domain,
) -> Dict[str, ColumnDescriptor]:
    """Convert a pandas table domain to Analytics columns.

    This is the inverse of :func:`analytics_to_pandas_columns_descriptor` on the
    descriptors that function produces; see
    ``test_pandas_schema_conversion`` for the exact domain on which the
    round-trip is an identity.
    """
    column_descs: Dict[str, ColumnDescriptor] = {}
    for column_name, descriptor in cast(PandasTableDomain, domain).schema.items():
        if isinstance(descriptor, PandasFloatColumnDescriptor):
            column_descs[column_name] = ColumnDescriptor(
                ColumnType.DECIMAL,
                allow_null=descriptor.allow_null,
                allow_nan=descriptor.allow_nan,
                allow_inf=descriptor.allow_inf,
            )
        else:
            column_descs[column_name] = ColumnDescriptor(
                column_type=_PANDAS_COLUMN_DESCRIPTOR_TO_ANALYTICS[type(descriptor)],
                allow_null=descriptor.allow_null,
            )
    return column_descs
