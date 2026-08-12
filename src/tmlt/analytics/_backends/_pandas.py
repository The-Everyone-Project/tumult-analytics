"""The pandas backend: Analytics on in-memory frames, without Spark.

This is the second implementation of the :class:`~._base.Backend` seam, and it
exists to make the seam's claim true: a backend is a table of Core artifacts,
and swapping the table swaps the engine. Everything here is the pandas
counterpart of something in :mod:`~tmlt.analytics._backends._spark`, in the same
order, so the two modules can be read side by side.

Which operations are missing, and why
=====================================

The pandas stack in Core is younger than the Spark one, so several
:class:`~._base.Ops` slots have nothing to bind yet. Those are ``None``, and
:data:`PANDAS_UNSUPPORTED_OPS` records every one of them with the reason -- a
slot is allowed to be empty, but not silently. :meth:`Backend.require` turns a
use of one into a :class:`~._base.NotSupportedByBackend` naming the operation,
so a query that needs a missing piece fails where it is written rather than
somewhere inside Core.

Importing this module without the Core it needs
===============================================

Analytics runs against a Core that may not have the pandas stack at all. The
imports below are therefore guarded: a Core without
``tmlt.core.transformations.pandas_transformations`` makes importing this module
raise :class:`~._base.BackendUnavailable`, which names the missing artifact and
what provides it, instead of an ``ImportError`` about a Core submodule the user
has never heard of. :mod:`~tmlt.analytics._backends` imports this module lazily
for the same reason: a Spark-only install must be able to import Analytics.

The named artifact stands for the whole pandas stack rather than being the only
thing imported -- the count measurements come from
``tmlt.core.measurements.pandas_aggregations``, and the domains from
``tmlt.core.domains.pandas_domains``. They ship together, in the same Core
build, so naming one of them is enough to say which build is wanted.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import pandas as pd
from tmlt.core.domains.base import Domain
from tmlt.core.exceptions import OutOfDomainError

from tmlt.analytics._backends._base import (
    Backend,
    BackendUnavailable,
    Ops,
    pandas_domain_from_dataframe,
    pandas_suppress_below,
)
from tmlt.analytics._coerce_pandas_schema import coerce_pandas_schema_or_fail
from tmlt.analytics._schema import (
    Schema,
    analytics_to_pandas_columns_descriptor,
    pandas_dataframe_domain_to_analytics_columns,
)
from tmlt.analytics._utils import AnalyticsInternalError

if TYPE_CHECKING:
    from tmlt.analytics.keyset import KeySet

_REQUIRED_CORE_ARTIFACT = "tmlt.core.transformations.pandas_transformations"

try:
    from tmlt.core.domains.pandas_domains import (
        PandasColumnDescriptor,
        PandasDateColumnDescriptor,
        PandasFloatColumnDescriptor,
        PandasGroupedTableDomain,
        PandasIntegerColumnDescriptor,
        PandasRowDomain,
        PandasStringColumnDescriptor,
        PandasTableDomain,
        PandasTimestampColumnDescriptor,
    )
    from tmlt.core.measurements.pandas_aggregations import (
        create_count_distinct_measurement,
        create_count_measurement,
    )
    from tmlt.core.transformations.pandas_transformations.add_remove_keys import (
        LimitKeysPerGroupValue,
        LimitRowsPerGroupValue,
        LimitRowsPerKeyPerGroupValue,
        MapValue,
        RenameValue,
        SelectValue,
    )
    from tmlt.core.transformations.pandas_transformations.groupby import GroupBy
    from tmlt.core.transformations.pandas_transformations.join import (
        PrivateJoin,
        PrivateJoinOnKey,
    )
    from tmlt.core.transformations.pandas_transformations.map import (
        Map,
        RowToRowTransformation,
    )
    from tmlt.core.transformations.pandas_transformations.rename import Rename
    from tmlt.core.transformations.pandas_transformations.select import Select
    from tmlt.core.transformations.pandas_transformations.truncation import (
        LimitKeysPerGroup,
        LimitRowsPerGroup,
        LimitRowsPerKeyPerGroup,
    )

    # TruncationStrategy names a strategy and holds no dataframe, so Core ships
    # one enum for both backends rather than two that mean the same thing and
    # compare unequal. The pandas joins take these very members; importing it
    # from the Spark module here is the same choice Core's pandas join module
    # makes, and binding the same object is what keeps the compiler's
    # `backend.require("TruncationStrategy").TRUNCATE` meaningful across
    # backends.
    from tmlt.core.transformations.spark_transformations.join import TruncationStrategy
except ImportError as exc:
    raise BackendUnavailable(
        "The pandas backend requires a Tumult Core that provides"
        f" '{_REQUIRED_CORE_ARTIFACT}', and the installed Core does not."
        " That module ships in the Everyone Project Core build"
        " (tmlt.core 0.19.1+ep.pandas.1 or later); no official Core release"
        " contains it. Install that build, or use the default Spark backend.",
        backend="pandas",
        required=_REQUIRED_CORE_ARTIFACT,
    ) from exc


PANDAS_UNSUPPORTED_OPS: Dict[str, str] = {
    # Core's pandas stack has no counterpart yet.
    "Filter": (
        "Core has no pandas filter transformation; row filtering is phase 2 of"
        " the pandas backend."
    ),
    "FlatMap": (
        "Core's pandas map module has only the one-row-to-one-row pair"
        " (RowToRowTransformation and Map); the flat-map family is phase 2."
    ),
    "FlatMapByKey": "As FlatMap: Core's pandas map module has no flat-map family.",
    "GroupingFlatMap": "As FlatMap: Core's pandas map module has no flat-map family.",
    "RowToRowsTransformation": (
        "As FlatMap: this is the row-level piece FlatMap is built from, and"
        " Core's pandas map module does not have it."
    ),
    "RowsToRowsTransformation": (
        "As FlatMap: this is the row-level piece FlatMapByKey is built from, and"
        " Core's pandas map module does not have it."
    ),
    "DropInfs": "Core has no pandas nan module; the null/NaN handling is phase 2.",
    "DropNaNs": "Core has no pandas nan module; the null/NaN handling is phase 2.",
    "DropNulls": "Core has no pandas nan module; the null/NaN handling is phase 2.",
    "ReplaceInfs": "Core has no pandas nan module; the null/NaN handling is phase 2.",
    "ReplaceNaNs": "Core has no pandas nan module; the null/NaN handling is phase 2.",
    "ReplaceNulls": "Core has no pandas nan module; the null/NaN handling is phase 2.",
    "PartitionByKeys": (
        "Core has no pandas partition transformation. Session.partition_and_create"
        " rejects a pandas Session up front rather than reaching this slot."
    ),
    # Out of scope by design rather than by absence.
    "PublicJoin": (
        "A pandas Session has no public tables in phase 1, so nothing can build"
        " a public join. Session.add_public_dataframe and"
        " Builder.with_public_dataframe reject a pandas Session up front."
    ),
    "Persist": (
        "There is nothing to persist: a pandas frame is already materialized in"
        " memory, and Spark's persist exists to stop a lazy plan being"
        " recomputed. persist_table treats an empty slot as a no-op."
    ),
    "Unpersist": "As Persist: unpersist_table treats an empty slot as a no-op.",
    "UnwrapIfGroupedBy": (
        "Core's tmlt.core.transformations.converters.UnwrapIfGroupedBy takes a"
        " SparkDataFrameDomain, so despite living outside the Spark package it"
        " is Spark-only. A pandas counterpart is phase 2."
    ),
    "HammingDistanceToSymmetricDifference": (
        "As UnwrapIfGroupedBy: Core's converter is typed to SparkDataFrameDomain."
    ),
    # The AddRemoveKeys value wrappers Core's pandas add_remove_keys module does
    # not have. Each one wraps a transformation, so a value wrapper is missing
    # for exactly the reason the transformation it wraps is missing, and the
    # entries below say so rather than restating it.
    "FilterValue": "As Filter, which is what it wraps.",
    "FlatMapValue": "As FlatMap, which is what it wraps.",
    "FlatMapByKeyValue": "As FlatMapByKey, which is what it wraps.",
    "PublicJoinValue": "As PublicJoin: a pandas Session has no public tables.",
    "DropInfsValue": "As DropInfs, which is what it wraps.",
    "DropNaNsValue": "As DropNaNs, which is what it wraps.",
    "DropNullsValue": "As DropNulls, which is what it wraps.",
    "ReplaceInfsValue": "As ReplaceInfs, which is what it wraps.",
    "ReplaceNaNsValue": "As ReplaceNaNs, which is what it wraps.",
    "ReplaceNullsValue": "As ReplaceNulls, which is what it wraps.",
    "PersistValue": (
        "As Persist: there is nothing to persist, and persist_table treats an"
        " empty slot as a no-op on the AddRemoveKeys path too."
    ),
    "UnpersistValue": "As Unpersist: unpersist_table treats an empty slot as a no-op.",
    # The remaining aggregations are phase 2: Core's create_*_measurement
    # factories in tmlt.core.measurements.aggregations build on Spark domains
    # and Spark measurements throughout.
    **{
        op: (
            "Phase 2: Core's aggregation factory is built on Spark domains and"
            " Spark measurements, and has no pandas counterpart yet."
        )
        for op in (
            "create_sum_measurement",
            "create_average_measurement",
            "create_variance_measurement",
            "create_standard_deviation_measurement",
            "create_quantile_measurement",
            "create_bounds_measurement",
            "create_partition_selection_measurement",
        )
    },
}
"""Every :class:`~._base.Ops` slot :data:`PANDAS` leaves empty, and why.

Read this as the pandas backend's to-do list and its scope statement at once.
``test_backends.py`` checks it against :data:`PANDAS` in both directions, so a
slot cannot quietly go missing and an entry cannot quietly go stale.
"""


_SAMPLE_VALUES: Tuple[Tuple[type, Any], ...] = (
    (PandasIntegerColumnDescriptor, 0),
    (PandasFloatColumnDescriptor, 0.0),
    (PandasStringColumnDescriptor, ""),
    (PandasDateColumnDescriptor, datetime.date(1970, 1, 1)),
    (PandasTimestampColumnDescriptor, datetime.datetime(1970, 1, 1, 0, 0, 0)),
)
"""The value a sample KeySet uses for each column descriptor.

These are the same defaults the Spark backend uses -- zero, the empty string,
the epoch -- so that the two backends' sample KeySets differ only in how they
were built. A list of pairs rather than a dict keyed by descriptor type because
a descriptor is matched with ``isinstance``, so that a future subclass of one of
these gets its base's default rather than no default at all.
"""


def _sample_value(descriptor: PandasColumnDescriptor) -> Any:
    """Return the stand-in value for one column of a sample KeySet."""
    for descriptor_type, value in _SAMPLE_VALUES:
        if isinstance(descriptor, descriptor_type):
            return value
    raise ValueError(f"Unsupported column descriptor {descriptor}")


def _pandas_dataframe_domain(schema: Schema) -> PandasTableDomain:
    """Build the pandas domain describing tables with the given Analytics schema."""
    return PandasTableDomain(analytics_to_pandas_columns_descriptor(schema))


def _pandas_sample_keyset(input_domain: Domain, columns: Sequence[str]) -> "KeySet":
    """Build a one-row KeySet over the given columns of a pandas table domain.

    See :attr:`~tmlt.analytics._backends._base.Backend.sample_keyset` for what
    this is for, and :data:`_SAMPLE_VALUES` for the row it puts in.

    The KeySet is built with :meth:`~tmlt.analytics.KeySet.from_tuples` rather
    than from the sample frame: a KeySet built from a DataFrame holds that
    DataFrame, and on this backend that is a pandas frame, which
    :meth:`~tmlt.analytics.KeySet.from_dataframe` does not take. The frame is
    still built, and validated against the domain, as the pandas counterpart of
    the Spark implementation's check that Spark gave it the schema it asked for:
    it is what catches a stand-in value that does not actually belong to the
    column it stands in for.
    """
    # Imported here rather than at the top because tmlt.analytics.keyset imports
    # the top-level tmlt.analytics package, which imports the backends package:
    # at the top it would be an import cycle.
    from tmlt.analytics.keyset import KeySet  # noqa: PLC0415

    if not isinstance(input_domain, PandasTableDomain):
        raise AnalyticsInternalError(
            "The pandas backend cannot build a sample KeySet from a "
            f"{type(input_domain).__name__}."
        )
    # project() keeps the domain's column order, as the Spark implementation's
    # filter over spark_schema does.
    domain = input_domain.project([c for c in input_domain.schema if c in columns])

    # KeySets with no columns aren't allowed to have rows.
    if not domain.schema:
        return KeySet.from_tuples([], columns=[])

    row = {column: _sample_value(desc) for column, desc in domain.schema.items()}
    sample = pd.DataFrame([row]).astype(domain.pandas_dtypes)
    try:
        domain.validate(sample)
    except OutOfDomainError as exc:
        raise AnalyticsInternalError(
            f"Failed to create a DataFrame in domain {domain}."
        ) from exc
    return KeySet.from_tuples([tuple(row.values())], columns=list(row))


PANDAS = Backend(
    name="pandas",
    dataframe_domain_type=PandasTableDomain,
    row_domain_type=PandasRowDomain,
    grouped_domain_type=PandasGroupedTableDomain,
    dataframe_type=pd.DataFrame,
    ops=Ops(
        Rename=Rename,
        Select=Select,
        Map=Map,
        PrivateJoin=PrivateJoin,
        PrivateJoinOnKey=PrivateJoinOnKey,
        TruncationStrategy=TruncationStrategy,
        RowToRowTransformation=RowToRowTransformation,
        GroupBy=GroupBy,
        LimitRowsPerGroup=LimitRowsPerGroup,
        LimitKeysPerGroup=LimitKeysPerGroup,
        LimitRowsPerKeyPerGroup=LimitRowsPerKeyPerGroup,
        RenameValue=RenameValue,
        SelectValue=SelectValue,
        MapValue=MapValue,
        LimitRowsPerGroupValue=LimitRowsPerGroupValue,
        LimitKeysPerGroupValue=LimitKeysPerGroupValue,
        LimitRowsPerKeyPerGroupValue=LimitRowsPerKeyPerGroupValue,
        create_count_measurement=create_count_measurement,
        create_count_distinct_measurement=create_count_distinct_measurement,
        # Every other slot is left at its None default; PANDAS_UNSUPPORTED_OPS
        # above says which and why.
    ),
    dataframe_domain=_pandas_dataframe_domain,
    columns_descriptor=analytics_to_pandas_columns_descriptor,
    domain_to_analytics_columns=pandas_dataframe_domain_to_analytics_columns,
    coerce_schema_or_fail=coerce_pandas_schema_or_fail,
    sample_keyset=_pandas_sample_keyset,
    domain_from_dataframe=pandas_domain_from_dataframe,
    suppress_below=pandas_suppress_below,
)
"""The pandas backend: Analytics on in-memory frames, without Spark."""
