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

from pyspark.sql import DataFrame
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

from tmlt.analytics._backends._base import Backend, Ops, spark_domain_from_dataframe
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import (
    Schema,
    analytics_to_spark_columns_descriptor,
    spark_dataframe_domain_to_analytics_columns,
)


def _spark_dataframe_domain(schema: Schema) -> SparkDataFrameDomain:
    """Build the Spark domain describing tables with the given Analytics schema."""
    return SparkDataFrameDomain(analytics_to_spark_columns_descriptor(schema))


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
    domain_from_dataframe=spark_domain_from_dataframe,
)
"""The Spark backend, and the default everywhere a backend can be chosen."""
