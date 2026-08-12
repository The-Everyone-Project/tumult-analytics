"""Which query features a backend can answer, checked before anything is built.

Every construction site in the compiler reaches its Core class through
:meth:`~tmlt.analytics._backends.Backend.require`, so a query that needs an
operation the backend lacks *does* fail with
:class:`~tmlt.analytics._backends.NotSupportedByBackend` rather than a
``TypeError`` about ``None``. But it fails late: by the time the visitor reaches
that slot, the query has already been validated, rewritten, and half compiled.
On a pandas Session "validated" is the expensive part -- ``Filter._validate``
compiles its condition against a real ``SparkSession``, so a query the pandas
backend was always going to refuse would boot a JVM on the way to being refused.

This module is the gate that fires first. It is a table from
:class:`~tmlt.analytics._query_expr.QueryExpr` type to the
:class:`~tmlt.analytics._backends.Ops` slots that type is built from, plus a
walk over the query tree that checks each node against the backend's table
before any of the query is touched. :func:`check_supported` is called from
:meth:`~tmlt.analytics._query_expr_compiler.QueryExprCompiler.query_schema`,
which is the one step ``evaluate``, ``create_view`` and ``describe`` all pass
through, and it is called *before* the schema is computed -- so nothing in
``_validate`` runs, no ``SparkSession`` is asked for, and no budget is spent.

Where the check may not live
============================

Not inside a Core :class:`~tmlt.core.transformations.base.Transformation` or
:class:`~tmlt.core.measurements.base.Measurement`.
:class:`~tmlt.analytics._backends.NotSupportedByBackend` is a
``NotImplementedError``, and Core's ``ChainTT``, ``Composition`` and
``DictMetric`` bodies catch ``NotImplementedError`` to mean "this component has
no closed-form privacy function, fall back to the relation". A rejection raised
in there would be swallowed and turned into a confusing failure somewhere else.
The gate therefore runs in compiler code, before Core is called at all.

Being wrong in the safe direction
=================================

A missing table entry, or an entry that lists too few slots, means a query is
*not* rejected here -- it goes on to fail at the ``require`` call inside the
visitor, which is still compile time and still spends no budget. An entry that
lists too many slots would reject a query the backend could actually answer, and
that is the failure this table must not have: each entry lists only the slots
every use of that query type needs, never the ones a particular argument might
pull in. The ``*Value`` slots of the AddRemoveKeys variants are deliberately
absent for that reason -- whether a query needs ``Select`` or ``SelectValue``
depends on the table's protected change, which the gate does not look at.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple, Type

from tmlt.analytics._backends import Backend, NotSupportedByBackend
from tmlt.analytics._query_expr import (
    DropInfinity,
    DropNullAndNan,
    EnforceConstraint,
    Filter,
    FlatMap,
    FlatMapByID,
    GetBounds,
    GetGroups,
    GroupByBoundedAverage,
    GroupByBoundedStdev,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
    JoinPrivate,
    JoinPublic,
    Map,
    PrivateSource,
    QueryExpr,
    Rename,
    ReplaceInfinity,
    ReplaceNullAndNan,
    Select,
    SingleChildQueryExpr,
    SuppressAggregates,
)
from tmlt.analytics._utils import AnalyticsInternalError
from tmlt.analytics.keyset import KeySet


@dataclass(frozen=True)
class Feature:
    """What one query feature needs from a backend.

    One of these per :class:`~tmlt.analytics._query_expr.QueryExpr` type, in
    :data:`REQUIRED_OPS`.
    """

    api: str
    """The call that builds this query expression, named as the user wrote it.

    The type name says what was rejected; this says where it came from, which is
    what a user needs in order to go and change it."""

    ops: Tuple[str, ...] = ()
    """Every :class:`~tmlt.analytics._backends.Ops` slot this feature needs.

    Empty for the query expressions that build nothing of their own -- reading a
    private source, enforcing a constraint (the constraint builds its own
    truncation, and requires its own slots), suppressing aggregates (which
    post-processes the answer through the backend rather than building a
    pipeline stage)."""


REQUIRED_OPS: Dict[Type[QueryExpr], Feature] = {
    # Reading a table builds nothing.
    PrivateSource: Feature("QueryBuilder()"),
    # Transformations.
    Rename: Feature("QueryBuilder.rename()", ("Rename",)),
    Filter: Feature("QueryBuilder.filter()", ("Filter",)),
    Select: Feature("QueryBuilder.select()", ("Select",)),
    Map: Feature("QueryBuilder.map()", ("Map", "RowToRowTransformation")),
    FlatMap: Feature("QueryBuilder.flat_map()", ("FlatMap", "RowToRowsTransformation")),
    FlatMapByID: Feature(
        "QueryBuilder.flat_map_by_id()", ("FlatMapByKey", "RowsToRowsTransformation")
    ),
    JoinPrivate: Feature(
        "QueryBuilder.join_private()", ("PrivateJoin", "TruncationStrategy")
    ),
    JoinPublic: Feature("QueryBuilder.join_public()", ("PublicJoin",)),
    ReplaceNullAndNan: Feature(
        "QueryBuilder.replace_null_and_nan()", ("ReplaceNulls", "ReplaceNaNs")
    ),
    ReplaceInfinity: Feature("QueryBuilder.replace_infinity()", ("ReplaceInfs",)),
    DropNullAndNan: Feature(
        "QueryBuilder.drop_null_and_nan()", ("DropNulls", "DropNaNs")
    ),
    DropInfinity: Feature("QueryBuilder.drop_infinity()", ("DropInfs",)),
    # A constraint builds its own truncation and requires its own slots; the
    # query expression that carries it builds nothing.
    EnforceConstraint: Feature("QueryBuilder.enforce()"),
    # Aggregations. Every one of them groups -- an ungrouped aggregation is a
    # group-by over a keyset with no columns -- so they all need GroupBy, and
    # each needs the factory that builds its own measurement.
    GroupByCount: Feature(
        "QueryBuilder.count()", ("GroupBy", "create_count_measurement")
    ),
    GroupByCountDistinct: Feature(
        "QueryBuilder.count_distinct()",
        ("GroupBy", "create_count_distinct_measurement"),
    ),
    GroupByBoundedSum: Feature(
        "QueryBuilder.sum()", ("GroupBy", "create_sum_measurement")
    ),
    GroupByBoundedAverage: Feature(
        "QueryBuilder.average()", ("GroupBy", "create_average_measurement")
    ),
    GroupByBoundedVariance: Feature(
        "QueryBuilder.variance()", ("GroupBy", "create_variance_measurement")
    ),
    GroupByBoundedStdev: Feature(
        "QueryBuilder.stdev()", ("GroupBy", "create_standard_deviation_measurement")
    ),
    GroupByQuantile: Feature(
        "QueryBuilder.quantile()", ("GroupBy", "create_quantile_measurement")
    ),
    GetBounds: Feature(
        "QueryBuilder.get_bounds()", ("GroupBy", "create_bounds_measurement")
    ),
    GetGroups: Feature(
        "QueryBuilder.get_groups()",
        ("Select", "create_partition_selection_measurement"),
    ),
    # Suppression filters the answer rather than the data, and does it through
    # the backend, so it needs no ops slot of its own.
    SuppressAggregates: Feature("Query.suppress()"),
}
"""Every :class:`~tmlt.analytics._query_expr.QueryExpr` type, and what it needs.

Every concrete type has an entry, including the ones that need nothing, so that
a type added without a decision about backends is caught by the test that checks
this table against :class:`~tmlt.analytics._query_expr.QueryExpr`'s subclasses
rather than quietly defaulting to "supported everywhere".
"""

AUTOMATIC_PARTITION_SELECTION = Feature(
    "QueryBuilder.groupby() with a list of columns",
    ("Select", "create_partition_selection_measurement"),
)
"""What a group-by that has to discover its own keys needs.

Grouping by a list of columns rather than by a
:class:`~tmlt.analytics.KeySet` means the keys are not known in advance, and
finding them privately is a measurement of its own -- one the aggregation's own
:class:`Feature` says nothing about. It is a feature in its own right, named as
one, so that a backend which has an aggregation but no partition selection
rejects the group-by rather than the aggregation.
"""


def _children(expr: QueryExpr) -> Tuple[QueryExpr, ...]:
    """The query expressions whose output this one is computed from.

    Mirrors :func:`~tmlt.analytics._query_expr_compiler._rewrite_rules.depth_first`:
    the two are the same walk, one rewriting and one checking, and they must
    agree about what the tree is.

    Raises:
        AnalyticsInternalError: If the query expression's shape is unrecognized.
    """
    if isinstance(expr, PrivateSource):
        return ()
    if isinstance(expr, SingleChildQueryExpr):
        return (expr.child,)
    if isinstance(expr, JoinPrivate):
        return (expr.left_child, expr.right_child)
    raise AnalyticsInternalError(
        f"Unrecognized QueryExpr subtype {type(expr).__qualname__}."
    )


def _required_features(expr: QueryExpr) -> Iterator[Tuple[str, Feature]]:
    """The features one query expression uses, and the name each is rejected as.

    Usually one -- the entry :data:`REQUIRED_OPS` holds for the expression's
    type, named after the type. A group-by that has to find its own keys uses a
    second one, which is named after itself rather than after the aggregation
    that asked for it, and is yielded first: how a query gets its groups is a
    more specific answer than what it then computes over them, in the same way
    and for the same reason that :func:`_check_keyset` runs first.
    """
    if isinstance(getattr(expr, "groupby_keys", None), tuple):
        yield "Automatic partition selection", AUTOMATIC_PARTITION_SELECTION
    feature = REQUIRED_OPS.get(type(expr))
    if feature is not None:
        yield type(expr).__name__, feature


def _unsupported_reason(feature: Feature, backend: Backend) -> Optional[str]:
    """Why this backend cannot provide the feature, or ``None`` if it can.

    The one place a feature is compared against a backend: both the gate and
    :func:`unsupported_features` ask this, so that what is rejected and what is
    reported as unavailable cannot drift apart.

    Args:
        feature: The feature to look for.
        backend: The backend to look for it in.
    """
    for op in feature.ops:
        if getattr(backend.ops, op) is None:
            return (
                f"{feature.api} needs the '{op}' operation, which this backend"
                " does not provide."
            )
    return None


def _check_keyset(expr: QueryExpr, backend: Backend) -> None:
    """Check a group-by's keyset, if it has one, against the backend.

    A :class:`~tmlt.analytics.KeySet` is an operation tree of its own, and some
    of those operations exist only on Spark -- one built by filtering, or from a
    Spark DataFrame. Walking the tree for them names every offending operation
    at once, and does it without materializing any of the keyset.

    This is checked before the expression's own ops, because a keyset the
    backend cannot build is a more specific answer than the aggregation it was
    handed to: the aggregation would still be wrong with a different keyset.

    Raises:
        NotSupportedByBackend: If the keyset uses operations this backend lacks.
    """
    keys = getattr(expr, "groupby_keys", None)
    if not isinstance(keys, KeySet):
        return
    # pylint: disable=protected-access
    unsupported = sorted(keys._op_tree.unsupported_ops(backend))
    if not unsupported:
        return
    raise NotSupportedByBackend.for_op(
        unsupported[0],
        backend.name,
        f"{type(expr).__name__} groups by a KeySet built with"
        f" {', '.join(unsupported)}, which this backend cannot materialize.",
    )


def _check_expr(expr: QueryExpr, backend: Backend) -> None:
    """Check one query expression against one backend.

    Raises:
        NotSupportedByBackend: If the backend cannot answer this expression.
    """
    _check_keyset(expr, backend)
    for feature_name, feature in _required_features(expr):
        reason = _unsupported_reason(feature, backend)
        if reason is not None:
            raise NotSupportedByBackend.for_op(feature_name, backend.name, reason)


def check_supported(query: QueryExpr, backend: Backend) -> None:
    """Reject a query the backend cannot answer, before any of it is built.

    Walks the query tree from the leaves up -- the order the compiler itself
    visits it in, so the operation named here is the one the compiler would have
    tripped over -- and checks each node against the backend.

    Args:
        query: The query to check.
        backend: The backend it would be compiled for.

    Raises:
        NotSupportedByBackend: If any part of the query, or of a
            :class:`~tmlt.analytics.KeySet` it groups by, needs something this
            backend does not have.
    """
    for child in _children(query):
        check_supported(child, backend)
    _check_expr(query, backend)


def unsupported_features(backend: Backend) -> Dict[str, str]:
    """Every query feature this backend cannot answer, and why.

    The same table :func:`check_supported` reads, turned around to be read
    backend-first: a feature matrix, computed rather than written down. Keyed by
    the name a rejection would use, valued by the sentence it would carry.

    Args:
        backend: The backend to describe.
    """
    features = {
        expr_type.__name__: feature for expr_type, feature in REQUIRED_OPS.items()
    }
    features["Automatic partition selection"] = AUTOMATIC_PARTITION_SELECTION
    reasons = {name: _unsupported_reason(f, backend) for name, f in features.items()}
    return {name: reason for name, reason in reasons.items() if reason is not None}
