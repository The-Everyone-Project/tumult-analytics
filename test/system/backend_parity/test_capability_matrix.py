"""What each backend can and cannot do -- classified once, in one place.

The negative half of parity. Two backends that answer the same queries the same
way are still not interchangeable unless it is *known* which queries each one
answers, and the failure this module exists to prevent is the quiet one: a query
feature added to Analytics later, supported on Spark, and neither implemented on
pandas nor recorded as missing. Such a feature would fail at some point inside
the pandas compiler with a message about ``None``, and no test would have said so.

Nothing here is hand-listed
===========================

The matrix is driven from
:func:`~tmlt.analytics._query_expr_compiler._backend_support.unsupported_features`
and :data:`~tmlt.analytics._query_expr_compiler._backend_support.REQUIRED_OPS` --
the tables the compiler's own rejection gate reads -- so what this suite asserts
and what the engine actually does cannot drift apart. Two consequences:

* Every feature the table says pandas lacks has to have a case here, or
  :func:`test_every_unsupported_feature_has_a_case` fails.
* Every :class:`~tmlt.analytics._query_expr.QueryExpr` type has to be either
  exercised as *supported* by the cases here or listed as unsupported, or
  :func:`test_every_query_expr_type_is_classified` fails. And "exercised" is not
  a claim: the supported types are collected by walking the query trees the cases
  below actually build, and each is then evaluated on both backends.

What this adds to ``test/unit/test_unsupported_surface.py``
==========================================================

That module is the unit-level surface: it checks how a pandas Session refuses
each feature -- the exception type, the operation named, the documentation hint,
and that refusing costs no budget -- and it asks the *gate* whether Spark would
accept. This module is the system-level pair of that: every feature is put
through a real Spark Session and answered, and every supported feature is put
through both backends and answered. Together they say a rejection is well-formed,
and that what is rejected on one backend really is available on the other.

Two features cannot be answered under an ordinary budget
========================================================

``get_groups`` and grouping by a list of columns both discover their own keys,
which is a measurement with a failure probability and therefore needs an
:class:`~tmlt.analytics.ApproxDPBudget`; under a pure-DP or zCDP budget the engine
refuses them for that reason, on either backend. They are given an
``ApproxDPBudget`` here, and grouping by a list of columns additionally needs the
``auto_partition_selection`` feature flag to be *built* at all.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple, Type

import pytest

from tmlt.analytics import (
    ApproxDPBudget,
    KeySet,
    MaxRowsPerID,
    PrivacyBudget,
    Query,
    QueryBuilder,
    RhoZCDPBudget,
    Session,
)
from tmlt.analytics._backends import PANDAS, SPARK, NotSupportedByBackend
from tmlt.analytics._query_expr import QueryExpr
from tmlt.analytics._query_expr_compiler._backend_support import (
    REQUIRED_OPS,
    _children,
    check_supported,
    unsupported_features,
)
from tmlt.analytics._schema import Schema
from tmlt.analytics.config import config
from tmlt.analytics.truncation_strategy import TruncationStrategy

from test.backend_testing import BackendFixture
from test.system.backend_parity.tables import (
    FEATURES,
    FEATURES_ARK,
    FEATURES_PUBLIC_SPEC,
    JOIN,
    build_session,
)

# The matrix's contents are recorded once, in the unit suite that is about the
# refusal itself; this module checks the same set rather than a second spelling
# of it. See PANDAS_UNSUPPORTED_FEATURES there.
from test.unit.test_unsupported_surface import PANDAS_UNSUPPORTED_FEATURES

_BUDGET = RhoZCDPBudget(100)
"""The budget the matrix's Sessions hold. Finite, so a spend is observable."""

_SPEND = RhoZCDPBudget(1)
"""What one query of the matrix spends."""

_APPROX_BUDGET = ApproxDPBudget(100, 1e-5)
"""The Session budget for the two partition-selection features."""

_APPROX_SPEND = ApproxDPBudget(1, 1e-6)
"""What one partition-selection query spends."""

_KEYS = KeySet.from_dict({"g": ["a", "b", "c", "d"]})
"""A keyset that reaches nothing unsupported, so a case is rejected for itself."""


################################################################################
# The cases
################################################################################


@dataclass(frozen=True)
class _Case:
    """One query feature, and how to ask a Session for it.

    Attributes:
        build: Builds the query. A callable rather than a query, because building
            one may need a feature flag held open.
        layout: The Session layout to run it in.
        approx: Whether the feature needs an ``ApproxDPBudget``.
        auto_partition: Whether the query needs the ``auto_partition_selection``
            flag in order to be built.
        public_tables: Public tables the Spark Session must hold for it.
        spark_evaluates: Whether a Spark Session can answer it. ``False`` is not
            used today -- every feature in the matrix is answerable on Spark --
            and exists so that a future one that is not can be recorded rather
            than dropped.
    """

    build: Callable[[], Query]
    layout: str = FEATURES
    approx: bool = False
    auto_partition: bool = False
    public_tables: Tuple[Any, ...] = ()
    spark_evaluates: bool = True

    def budget(self) -> PrivacyBudget:
        """The total budget a Session for this case needs."""
        return _APPROX_BUDGET if self.approx else _BUDGET

    def spend(self) -> PrivacyBudget:
        """What one evaluation of this case spends."""
        return _APPROX_SPEND if self.approx else _SPEND

    def flags(self) -> Iterator[Any]:
        """The feature flags this case must be built and run inside."""
        if self.auto_partition:
            yield config.features.auto_partition_selection.enabled()


def _flat_map(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row in, one row out, for the ``flat_map`` case.

    Args:
        row: The row to map.

    Returns:
        The new rows.
    """
    return [{"D": 1}]


def _flat_map_by_id(rows: Any) -> List[Dict[str, Any]]:
    """All of an ID's rows in, one row out, for the ``flat_map_by_id`` case.

    Args:
        rows: The ID's rows.

    Returns:
        The new rows.
    """
    return [{"D": len(rows)}]


def _double(row: Dict[str, Any]) -> Dict[str, Any]:
    """Doubles a row's ``v``, for the supported chain case.

    Args:
        row: The row to map.

    Returns:
        The new column.
    """
    return {"double": row["v"] * 2}


_UNSUPPORTED_CASES: Dict[str, _Case] = {
    # ------------------------------------------------------- transformations
    # Each is counted over, so that there is a query to evaluate. The count is
    # not what is rejected: the gate walks a query from its leaves up, so it is
    # the transformation underneath that fails first.
    "Filter": _Case(lambda: QueryBuilder("t").filter("v > 1").groupby(_KEYS).count()),
    "FlatMap": _Case(
        lambda: (
            QueryBuilder("t")
            .flat_map(_flat_map, {"D": "INTEGER"}, augment=True, max_rows=1)
            .groupby(_KEYS)
            .count()
        )
    ),
    "FlatMapByID": _Case(
        lambda: (
            QueryBuilder("t")
            .flat_map_by_id(_flat_map_by_id, {"D": "INTEGER"})
            .enforce(MaxRowsPerID(2))
            .groupby(KeySet.from_dict({"D": [1, 2, 3]}))
            .count()
        ),
        layout=FEATURES_ARK,
    ),
    "JoinPublic": _Case(
        lambda: (
            QueryBuilder("t")
            .join_public("parity_features_public")
            .groupby(_KEYS)
            .count()
        ),
        public_tables=(FEATURES_PUBLIC_SPEC,),
    ),
    "ReplaceNullAndNan": _Case(
        lambda: QueryBuilder("t").replace_null_and_nan().groupby(_KEYS).count()
    ),
    "ReplaceInfinity": _Case(
        lambda: (
            QueryBuilder("t")
            .replace_infinity({"f": (-100.0, 100.0)})
            .groupby(_KEYS)
            .count()
        )
    ),
    "DropNullAndNan": _Case(
        lambda: QueryBuilder("t").drop_null_and_nan(["f"]).groupby(_KEYS).count()
    ),
    "DropInfinity": _Case(
        lambda: QueryBuilder("t").drop_infinity(["f"]).groupby(_KEYS).count()
    ),
    # ---------------------------------------------------------- aggregations
    "GroupByBoundedSum": _Case(
        lambda: QueryBuilder("t").groupby(_KEYS).sum("v", low=0, high=10)
    ),
    "GroupByBoundedAverage": _Case(
        lambda: QueryBuilder("t").groupby(_KEYS).average("v", low=0, high=10)
    ),
    "GroupByBoundedVariance": _Case(
        lambda: QueryBuilder("t").groupby(_KEYS).variance("v", low=0, high=10)
    ),
    "GroupByBoundedStdev": _Case(
        lambda: QueryBuilder("t").groupby(_KEYS).stdev("v", low=0, high=10)
    ),
    "GroupByQuantile": _Case(
        lambda: QueryBuilder("t").groupby(_KEYS).quantile("v", 0.5, low=0, high=10)
    ),
    "GetBounds": _Case(lambda: QueryBuilder("t").groupby(_KEYS).get_bounds("v")),
    # ------------------------------------------------- partition selection
    "GetGroups": _Case(lambda: QueryBuilder("t").get_groups(["g"]), approx=True),
    "Automatic partition selection": _Case(
        lambda: QueryBuilder("t").groupby(["g"]).count(),
        approx=True,
        auto_partition=True,
    ),
}
"""Every feature the pandas backend refuses, as a query a Session can be asked.

Keyed by the name the rejection uses, which is the key
:func:`~tmlt.analytics._query_expr_compiler._backend_support.unsupported_features`
reports -- usually the ``QueryExpr`` type's name, and for one feature the name of
the feature itself, because how a group-by finds its keys is not a property of the
aggregation over them.
"""

_SUPPORTED_CASES: Dict[str, _Case] = {
    # Between them, these cover every QueryExpr type the pandas backend does
    # support; test_every_query_expr_type_is_classified checks that, by walking
    # the trees rather than by trusting this comment.
    "count": _Case(lambda: QueryBuilder("t").groupby(_KEYS).count()),
    "count_distinct": _Case(
        lambda: QueryBuilder("t").groupby(_KEYS).count_distinct(["v"])
    ),
    "rename_map_select": _Case(
        lambda: (
            QueryBuilder("t")
            .rename({"g": "grp"})
            .map(
                _double,
                new_column_types=Schema({"double": "INTEGER"}),
                augment=True,
            )
            .select(["grp", "double"])
            .groupby(KeySet.from_dict({"grp": ["a", "b", "c", "d"]}))
            .count()
        )
    ),
    "enforce": _Case(
        lambda: QueryBuilder("t").enforce(MaxRowsPerID(2)).groupby(_KEYS).count(),
        layout=FEATURES_ARK,
    ),
    "join_private": _Case(
        lambda: (
            QueryBuilder("l")
            .join_private(
                "r",
                truncation_strategy_left=TruncationStrategy.DropExcess(2),
                truncation_strategy_right=TruncationStrategy.DropExcess(2),
            )
            .groupby(KeySet.from_dict({"g": ["a", "b"]}))
            .count()
        ),
        layout=JOIN,
    ),
    "suppress": _Case(lambda: QueryBuilder("t").groupby(_KEYS).count().suppress(2)),
}
"""One query per feature the pandas backend does support, for the other half.

Not a value grid -- :mod:`~test.system.backend_parity.test_answers` is that --
but the completeness half of the matrix: these are the queries whose
``QueryExpr`` types count as *classified as supported*, and each is evaluated on
both backends so that the classification is earned rather than declared.
"""


################################################################################
# Walking a query
################################################################################


def _tree(expr: QueryExpr) -> Iterator[QueryExpr]:
    """Every node of a query expression tree, leaves first.

    Uses the compiler's own
    :func:`~tmlt.analytics._query_expr_compiler._backend_support._children`, so
    that "what the tree is" cannot drift between this walk and the gate's.

    Args:
        expr: The root.

    Yields:
        Each node.
    """
    for child in _children(expr):
        yield from _tree(child)
    yield expr


def _types_in(case: _Case) -> Set[Type[QueryExpr]]:
    """The QueryExpr types a case's query is built from.

    Args:
        case: The case to walk.

    Returns:
        Every node type in its tree.
    """
    with ExitStack() as stack:
        for flag in case.flags():
            stack.enter_context(flag)
        query = case.build()
    return {type(node) for node in _tree(query._query_expr)}


def _supported_types() -> Set[Type[QueryExpr]]:
    """Every QueryExpr type the supported half of the matrix exercises."""
    found: Set[Type[QueryExpr]] = set()
    for case in _SUPPORTED_CASES.values():
        found |= _types_in(case)
    return found


def _unsupported_types() -> Set[Type[QueryExpr]]:
    """Every QueryExpr type the table says the pandas backend cannot build."""
    names = set(unsupported_features(PANDAS))
    return {expr_type for expr_type in REQUIRED_OPS if expr_type.__name__ in names}


################################################################################
# The matrix's own consistency
################################################################################


def test_every_unsupported_feature_has_a_case():
    """Every feature pandas lacks is asked for by a query in this module.

    A feature added to the unsupported table without a case would otherwise be
    untested on both halves at once: nothing would check that pandas really
    refuses it, and nothing would check that Spark really provides it.
    """
    assert set(_UNSUPPORTED_CASES) == set(unsupported_features(PANDAS))


def test_every_query_expr_type_is_classified():
    """No QueryExpr type is neither exercised nor recorded as missing.

    This is the test that makes the matrix a matrix. A type added to the query
    language lands in :data:`REQUIRED_OPS` -- ``test_unsupported_surface`` insists
    on that -- and from there it must either be answered by a case in
    :data:`_SUPPORTED_CASES`, which every backend then has to answer, or be listed
    as unavailable on pandas. Until somebody decides which, this fails.
    """
    classified = _supported_types() | _unsupported_types()
    unclassified = set(REQUIRED_OPS) - classified
    assert not unclassified, (
        "These QueryExpr types are neither exercised by a supported case nor "
        "listed as unsupported on pandas: "
        f"{sorted(t.__name__ for t in unclassified)}. Add a case to "
        "_SUPPORTED_CASES, or record the missing ops in REQUIRED_OPS."
    )


def test_the_two_halves_do_not_overlap():
    """A type is either exercised or unsupported, never both.

    An overlap would mean a supported case reaches a feature pandas cannot
    provide, which would make that case fail on pandas -- confusingly, since it
    is in the *supported* half.
    """
    overlap = _supported_types() & _unsupported_types()
    assert not overlap, sorted(t.__name__ for t in overlap)


def test_the_pandas_backend_lacks_exactly_these():
    """The matrix's contents, recorded, so that a change to it is deliberate.

    Against the same set ``test_unsupported_surface`` checks, rather than
    against a count of it: two suites recording the same matrix in two
    spellings is how they come to record different matrices, and a length is
    the spelling that says least about which features moved.

    That Spark lacks nothing -- the anchor that makes "unsupported" here mean
    "unsupported on pandas" -- is asserted in that module too.
    """
    unsupported = unsupported_features(PANDAS)
    assert set(unsupported) == PANDAS_UNSUPPORTED_FEATURES
    assert {expr_type.__name__ for expr_type in _unsupported_types()} == (
        PANDAS_UNSUPPORTED_FEATURES - {"Automatic partition selection"}
    )
    # Every reason names the ops slot that is missing, so a user is told what to
    # change rather than only that something is wrong.
    for feature, reason in unsupported.items():
        assert "does not provide" in reason, feature


@pytest.mark.parametrize("feature", sorted(_UNSUPPORTED_CASES))
def test_an_unsupported_cases_query_really_uses_the_feature(feature: str):
    """Each case's query contains the node it is meant to be rejected for.

    Without this, a case could be renamed or rewritten into a query that no
    longer reaches its feature, and the rejection test below would still
    pass -- on some *other* unsupported node in the same tree.
    """
    if feature == "Automatic partition selection":
        # Not a QueryExpr type: it is what a group-by needs when its keys are a
        # tuple of column names rather than a KeySet. Check that shape instead.
        with config.features.auto_partition_selection.enabled():
            expr = _UNSUPPORTED_CASES[feature].build()._query_expr
        assert isinstance(getattr(expr, "groupby_keys", None), tuple)
        return
    names = {expr_type.__name__ for expr_type in _types_in(_UNSUPPORTED_CASES[feature])}
    assert feature in names, f"{feature}'s query is built from {sorted(names)}."


################################################################################
# The matrix itself
################################################################################


@pytest.mark.parametrize("feature", sorted(_UNSUPPORTED_CASES))
def test_pandas_refuses_the_feature_and_charges_nothing(feature: str):
    """A pandas Session refuses each unsupported feature, for free.

    The refusal has to be free as well as correct: a gate that spent budget
    before deciding it could not answer would leak the shape of a query that was
    never answered.
    """
    case = _UNSUPPORTED_CASES[feature]
    backend = BackendFixture(name="pandas")
    with ExitStack() as stack:
        stack.enter_context(backend.feature_flag())
        for flag in case.flags():
            stack.enter_context(flag)
        session = build_session(backend, case.layout, case.budget())
        with pytest.raises(NotSupportedByBackend) as raised:
            session.evaluate(case.build(), case.spend())
        assert session.remaining_privacy_budget == case.budget()

    assert raised.value.op == feature
    assert raised.value.backend == "pandas"
    assert unsupported_features(PANDAS)[feature] in str(raised.value)


@pytest.mark.parametrize("feature", sorted(_UNSUPPORTED_CASES))
def test_spark_answers_the_feature(spark, feature: str):
    """A real Spark Session answers every feature pandas refuses.

    The other half of "unsupported on pandas": not merely that the capability
    table says Spark has the ops, but that a Session built on Spark runs the
    query and charges for it. That is what makes the pandas rejection a statement
    about the backends rather than about the query.

    Args:
        spark: The Spark session.
        feature: The feature under test.
    """
    case = _UNSUPPORTED_CASES[feature]
    backend = BackendFixture(name="spark", spark=spark)
    with ExitStack() as stack:
        for flag in case.flags():
            stack.enter_context(flag)
        session = build_session(
            backend, case.layout, case.budget(), public_tables=case.public_tables
        )
        # The gate accepts it, which is the claim the capability table makes ...
        check_supported(case.build()._query_expr, SPARK)
        assert case.spark_evaluates
        # ... and the Session then really answers it.
        answer = session.evaluate(case.build(), case.spend())
        assert answer.columns
        assert session.remaining_privacy_budget != case.budget()


@pytest.mark.parametrize("feature", sorted(_SUPPORTED_CASES))
def test_both_backends_answer_a_supported_feature(
    backend: BackendFixture, feature: str
):
    """Every feature the matrix classifies as supported is answered by both.

    The positive half. It is what earns :data:`_SUPPORTED_CASES` its place in
    :func:`test_every_query_expr_type_is_classified`: a type is not "classified as
    supported" because a comment says so, but because a Session on each backend
    answered a query containing it.
    """
    case = _SUPPORTED_CASES[feature]
    with ExitStack() as stack:
        for flag in case.flags():
            stack.enter_context(flag)
        session = build_session(backend, case.layout, case.budget())
        answer = session.evaluate(case.build(), case.spend())
    # Both backends' frame types expose `columns`, so the answer's shape can be
    # checked without asking which backend produced it.
    assert list(answer.columns)
    assert session.remaining_privacy_budget != case.budget()


@pytest.mark.parametrize("feature", sorted(_SUPPORTED_CASES))
def test_the_gate_accepts_a_supported_feature_on_both_backends(feature: str):
    """The gate agrees with the Sessions above, and needs no Session to say so.

    Asked directly and for both backends at once, which a Session cannot be: this
    is the capability table's own answer, and it is checked here so that a table
    entry loosened or tightened by accident shows up as a difference from the
    evaluations rather than as a silently widened surface.
    """
    case = _SUPPORTED_CASES[feature]
    with ExitStack() as stack:
        for flag in case.flags():
            stack.enter_context(flag)
        query = case.build()
    for descriptor in (PANDAS, SPARK):
        check_supported(query._query_expr, descriptor)
