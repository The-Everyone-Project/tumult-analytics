"""The Spark backend: Analytics' original and default backend.

This module is where the compiler's dependency on Core's Spark transformations
now lives. Nothing else under
:mod:`~tmlt.analytics._query_expr_compiler` imports
``tmlt.core.transformations.spark_transformations``; the compiler reaches those
classes through :data:`SPARK`, and so would reach another backend's classes
through that backend's descriptor.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructType,
    TimestampType,
)
from tmlt.core.domains.base import Domain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
    SparkRowDomain,
)
from tmlt.core.measurements.aggregations import (
    create_average_measurement,
    create_bounds_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_partition_selection_measurement,
    create_quantile_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
)
from tmlt.core.transformations.converters import (
    HammingDistanceToSymmetricDifference,
    UnwrapIfGroupedBy,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    DropInfsValue,
    DropNaNsValue,
    DropNullsValue,
    FilterValue,
    FlatMapByKeyValue,
    FlatMapValue,
    LimitKeysPerGroupValue,
    LimitRowsPerGroupValue,
    LimitRowsPerKeyPerGroupValue,
    MapValue,
    PersistValue,
    PublicJoinValue,
    RenameValue,
    ReplaceInfsValue,
    ReplaceNaNsValue,
    ReplaceNullsValue,
    SelectValue,
    UnpersistValue,
)
from tmlt.core.transformations.spark_transformations.filter import Filter
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoin,
    PrivateJoinOnKey,
    PublicJoin,
    TruncationStrategy,
)
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
    FlatMapByKey,
    GroupingFlatMap,
    Map,
    RowsToRowsTransformation,
    RowToRowsTransformation,
    RowToRowTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropInfs,
    DropNaNs,
    DropNulls,
    ReplaceInfs,
    ReplaceNaNs,
    ReplaceNulls,
)
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.transformations.spark_transformations.persist import Persist, Unpersist
from tmlt.core.transformations.spark_transformations.rename import Rename
from tmlt.core.transformations.spark_transformations.select import Select
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitKeysPerGroup,
    LimitRowsPerGroup,
    LimitRowsPerKeyPerGroup,
)

from tmlt.analytics._backends._base import Backend, Ops
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import (
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_dataframe_domain_to_analytics_columns,
)
from tmlt.analytics._utils import AnalyticsInternalError

if TYPE_CHECKING:
    from tmlt.analytics.keyset import KeySet


def _spark_dataframe_domain(schema: Schema) -> SparkDataFrameDomain:
    """Build the Spark domain describing tables with the given Analytics schema."""
    return SparkDataFrameDomain(analytics_to_spark_columns_descriptor(schema))


def _spark_sample_keyset(input_domain: Domain, columns: Sequence[str]) -> "KeySet":
    """Build a one-row KeySet over the given columns of a Spark table domain.

    See :attr:`~tmlt.analytics._backends._base.Backend.sample_keyset` for what
    this is for. The row's values are per-type defaults; nothing reads them, so
    the only thing that matters about them is that they are in the column's
    type.
    """
    # Imported here rather than at the top because tmlt.analytics.keyset imports
    # the top-level tmlt.analytics package, which imports this module by way of
    # tmlt.analytics.constraints: at the top it would be an import cycle.
    from tmlt.analytics.keyset import KeySet  # noqa: PLC0415

    if not isinstance(input_domain, SparkDataFrameDomain):
        raise AnalyticsInternalError(
            "The Spark backend cannot build a sample KeySet from a "
            f"{type(input_domain).__name__}."
        )
    schema = StructType(
        [field for field in input_domain.spark_schema if field.name in columns]
    )
    spark = SparkSession.builder.getOrCreate()

    # KeySets with no columns aren't allowed to have rows.
    if not schema.fields:
        return KeySet.from_dataframe(spark.createDataFrame([], schema=schema))

    default_values = []
    for field in schema.fields:
        default_value: Any
        if isinstance(field.dataType, (ByteType, ShortType, IntegerType, LongType)):
            default_value = 0
        elif isinstance(field.dataType, (FloatType, DoubleType)):
            default_value = 0.0
        elif isinstance(field.dataType, StringType):
            default_value = ""
        elif isinstance(field.dataType, BooleanType):
            default_value = False
        elif isinstance(field.dataType, DateType):
            default_value = datetime.strptime("1970-01-01", "%Y-%m-%d").date()
        elif isinstance(field.dataType, TimestampType):
            default_value = datetime.strptime(
                "1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
            )
        elif isinstance(field.dataType, DecimalType):
            default_value = Decimal("0.0")
        else:
            raise ValueError(f"Unsupported data type {field.dataType}")
        default_values.append(default_value)

    # Create a DataFrame with a single row using the default values
    df = spark.createDataFrame([tuple(default_values)], schema=schema)
    if df.schema != schema:
        raise AnalyticsInternalError(
            f"Failed to create a DataFrame with schema {schema}."
        )
    return KeySet.from_dataframe(df)


SPARK = Backend(
    name="Spark",
    dataframe_domain_type=SparkDataFrameDomain,
    row_domain_type=SparkRowDomain,
    grouped_domain_type=SparkGroupedDataFrameDomain,
    dataframe_type=DataFrame,
    ops=Ops(
        Rename=Rename,
        Filter=Filter,
        Select=Select,
        Map=Map,
        FlatMap=FlatMap,
        FlatMapByKey=FlatMapByKey,
        GroupingFlatMap=GroupingFlatMap,
        PrivateJoin=PrivateJoin,
        PrivateJoinOnKey=PrivateJoinOnKey,
        PublicJoin=PublicJoin,
        TruncationStrategy=TruncationStrategy,
        DropInfs=DropInfs,
        DropNaNs=DropNaNs,
        DropNulls=DropNulls,
        ReplaceInfs=ReplaceInfs,
        ReplaceNaNs=ReplaceNaNs,
        ReplaceNulls=ReplaceNulls,
        Persist=Persist,
        Unpersist=Unpersist,
        PartitionByKeys=PartitionByKeys,
        RowToRowTransformation=RowToRowTransformation,
        RowToRowsTransformation=RowToRowsTransformation,
        RowsToRowsTransformation=RowsToRowsTransformation,
        GroupBy=GroupBy,
        UnwrapIfGroupedBy=UnwrapIfGroupedBy,
        HammingDistanceToSymmetricDifference=HammingDistanceToSymmetricDifference,
        LimitRowsPerGroup=LimitRowsPerGroup,
        LimitKeysPerGroup=LimitKeysPerGroup,
        LimitRowsPerKeyPerGroup=LimitRowsPerKeyPerGroup,
        RenameValue=RenameValue,
        FilterValue=FilterValue,
        SelectValue=SelectValue,
        MapValue=MapValue,
        FlatMapValue=FlatMapValue,
        FlatMapByKeyValue=FlatMapByKeyValue,
        PublicJoinValue=PublicJoinValue,
        DropInfsValue=DropInfsValue,
        DropNaNsValue=DropNaNsValue,
        DropNullsValue=DropNullsValue,
        ReplaceInfsValue=ReplaceInfsValue,
        ReplaceNaNsValue=ReplaceNaNsValue,
        ReplaceNullsValue=ReplaceNullsValue,
        PersistValue=PersistValue,
        UnpersistValue=UnpersistValue,
        LimitRowsPerGroupValue=LimitRowsPerGroupValue,
        LimitKeysPerGroupValue=LimitKeysPerGroupValue,
        LimitRowsPerKeyPerGroupValue=LimitRowsPerKeyPerGroupValue,
        create_count_measurement=create_count_measurement,
        create_count_distinct_measurement=create_count_distinct_measurement,
        create_sum_measurement=create_sum_measurement,
        create_average_measurement=create_average_measurement,
        create_variance_measurement=create_variance_measurement,
        create_standard_deviation_measurement=create_standard_deviation_measurement,
        create_quantile_measurement=create_quantile_measurement,
        create_bounds_measurement=create_bounds_measurement,
        create_partition_selection_measurement=create_partition_selection_measurement,
    ),
    dataframe_domain=_spark_dataframe_domain,
    columns_descriptor=analytics_to_spark_columns_descriptor,
    domain_to_analytics_columns=spark_dataframe_domain_to_analytics_columns,
    coerce_schema_or_fail=coerce_spark_schema_or_fail,
    sample_keyset=_spark_sample_keyset,
)
"""The Spark backend, and the default everywhere a backend can be chosen."""
