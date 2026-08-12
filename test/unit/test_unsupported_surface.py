"""What a pandas Session refuses at compile time, and how it refuses it.

Every feature the pandas backend does not have in phase 1 has to fail the same
way: a :class:`~tmlt.analytics._backends.NotSupportedByBackend` naming the
feature and the backend, raised while the query is still being read. Never an
``AttributeError`` from an empty ops slot, never a budget spent on a query that
was never going to run, and -- the one that is easy to get wrong -- never after
booting a JVM.

That last one is what :func:`fixture_forbid_spark` is for. ``Filter._validate``
checks its condition by building an empty Spark DataFrame and filtering it, so a
gate placed anywhere after query validation would start a Spark session before
deciding the query was unsupported. The fixture makes any reach for a
``SparkSession`` an immediate failure, so these tests fail loudly if the gate
ever moves behind validation. It replaces ``SparkSession.Builder.getOrCreate``
rather than only ``launch_gateway`` (which is what Core's ``TMLT_FORBID_JVM``
guard replaces) because the unit-test process may already have a JVM running
from another module's session-scoped ``spark`` fixture: with one running, asking
for a session succeeds without launching anything, and only the ask itself
distinguishes a Spark-free code path from a Spark-using one.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import inspect
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Set

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tmlt.analytics import (
    AddOneRow,
    AddRowsWithID,
    KeySet,
    PureDPBudget,
    QueryBuilder,
    Session,
)
from tmlt.analytics._backends import (
    FEATURE_MATRIX_HINT,
    PANDAS,
    SPARK,
    NotSupportedByBackend,
)
from tmlt.analytics._query_expr import QueryExpr
from tmlt.analytics._query_expr_compiler._backend_support import (
    REQUIRED_OPS,
    check_supported,
    unsupported_features,
)
from tmlt.analytics.config import config

_DATA = pd.DataFrame(
    {"A": ["a", "b", "a"], "B": [1, 2, 3], "C": [1.0, 2.0, 3.0], "id": ["x", "y", "z"]}
)

_KEYS = KeySet.from_dict({"A": ["a", "b"]})
"""The group-by keys the cases below use.

Built from a dict, which is an operation both backends have: a case is meant to
be rejected for the one feature it is about, not for its keyset."""

_COUNT_FACTORY_BOUND = PANDAS.ops.create_count_measurement is not None
"""Whether Core's pandas count factory has landed and been bound yet.

A handful of features are unsupported today only because counting is: a
histogram is a binned column plus a count. Those cases are skipped once counting
works, because there is then nothing left in them for the gate to reject."""


###############################################################################
# Fixtures.
###############################################################################


@pytest.fixture(name="feature_flags", scope="module", autouse=True)
def fixture_feature_flags() -> Iterator[None]:
    """The flags the cases here need in order to be built at all.

    The pandas backend is one; grouping by a list of columns rather than by a
    KeySet is the other, and one of the cases below is exactly that.
    """
    with config.features.pandas_backend.enabled():
        with config.features.auto_partition_selection.enabled():
            yield


@pytest.fixture(name="pandas_session", scope="module")
def fixture_pandas_session() -> Session:
    """A pandas Session with a plain table and an AddRowsWithID one.

    Module-scoped: no test here spends any budget -- that is half of what they
    are checking -- so one Session serves all of them.
    """
    return (
        Session.Builder()
        .with_privacy_budget(PureDPBudget(10))
        .with_id_space("a")
        .with_private_dataframe("t", _DATA, AddOneRow())
        .with_private_dataframe("ids", _DATA, AddRowsWithID("id", "a"))
        .build()
    )


@pytest.fixture(name="forbid_spark")
def fixture_forbid_spark(monkeypatch) -> None:
    """Make any attempt to reach a SparkSession fail the test.

    See this module's docstring for why this is the guard rather than a check
    that no JVM is running.
    """

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "This code path asked for a SparkSession. A query the backend"
            " cannot answer must be rejected before anything reaches Spark."
        )

    monkeypatch.setattr(SparkSession.Builder, "getOrCreate", _forbidden)
    # Belt and braces: if some path builds a session another way, this catches
    # the JVM launch itself. The name has to be replaced on every pyspark module
    # that imported it, not only where it is defined.
    for name, module in list(sys.modules.items()):
        if name != "pyspark" and not name.startswith("pyspark."):
            continue
        if getattr(module, "launch_gateway", None) is not None:
            monkeypatch.setattr(module, "launch_gateway", _forbidden)


###############################################################################
# The negative matrix.
###############################################################################


@dataclass(frozen=True)
class _Case:
    """One unsupported feature, and how to ask a Session for it."""

    id: str
    """The name this case is reported under."""

    build: Callable[[], Any]
    """Builds the query. A callable rather than a query because building one
    needs the feature flags the module fixture holds open."""

    op: str
    """The feature the rejection has to name."""


def _flat_map(row):  # pragma: no cover -- the query is never evaluated.
    """A flat-map function, never called: the query it is in is rejected."""
    return [{"D": 1}]


def _flat_map_by_id(rows):  # pragma: no cover -- as above.
    """A flat-map-by-id function, never called."""
    return [{"D": len(rows)}]


_TRANSFORMATION_CASES: List[_Case] = [
    _Case("filter", lambda: QueryBuilder("t").filter("B > 1"), "Filter"),
    _Case(
        "flat_map",
        lambda: QueryBuilder("t").flat_map(
            _flat_map, {"D": "INTEGER"}, augment=True, max_rows=1
        ),
        "FlatMap",
    ),
    _Case(
        "flat_map_by_id",
        lambda: QueryBuilder("ids").flat_map_by_id(_flat_map_by_id, {"D": "INTEGER"}),
        "FlatMapByID",
    ),
    _Case(
        # The public table does not exist on this Session -- it cannot, a pandas
        # Session takes none -- and the point is that the gate says so before
        # anything looks the table up: the feature is unavailable whatever it
        # would have been joined against.
        "join_public",
        lambda: QueryBuilder("t").join_public("pub"),
        "JoinPublic",
    ),
    _Case(
        "replace_null_and_nan",
        lambda: QueryBuilder("t").replace_null_and_nan(),
        "ReplaceNullAndNan",
    ),
    _Case(
        "replace_infinity",
        lambda: QueryBuilder("t").replace_infinity({"C": (-100.0, 100.0)}),
        "ReplaceInfinity",
    ),
    _Case(
        "drop_null_and_nan",
        lambda: QueryBuilder("t").drop_null_and_nan(["C"]),
        "DropNullAndNan",
    ),
    _Case(
        "drop_infinity", lambda: QueryBuilder("t").drop_infinity(["C"]), "DropInfinity"
    ),
]
"""The unsupported transformations, as transformations.

A transformation on its own is a view rather than a query, so these are the
cases ``create_view`` can be given; the matrix below counts over each of them to
make a query out of it.
"""

_AGGREGATION_CASES: List[_Case] = [
    _Case(
        "sum",
        lambda: QueryBuilder("t").groupby(_KEYS).sum("B", low=0, high=10),
        "GroupByBoundedSum",
    ),
    _Case(
        "average",
        lambda: QueryBuilder("t").groupby(_KEYS).average("B", low=0, high=10),
        "GroupByBoundedAverage",
    ),
    _Case(
        "variance",
        lambda: QueryBuilder("t").groupby(_KEYS).variance("B", low=0, high=10),
        "GroupByBoundedVariance",
    ),
    _Case(
        "stdev",
        lambda: QueryBuilder("t").groupby(_KEYS).stdev("B", low=0, high=10),
        "GroupByBoundedStdev",
    ),
    _Case(
        "quantile",
        lambda: QueryBuilder("t").groupby(_KEYS).quantile("B", 0.5, low=0, high=10),
        "GroupByQuantile",
    ),
    _Case(
        "median",
        lambda: QueryBuilder("t").groupby(_KEYS).median("B", low=0, high=10),
        "GroupByQuantile",
    ),
    _Case(
        "min",
        lambda: QueryBuilder("t").groupby(_KEYS).min("B", low=0, high=10),
        "GroupByQuantile",
    ),
    _Case(
        "max",
        lambda: QueryBuilder("t").groupby(_KEYS).max("B", low=0, high=10),
        "GroupByQuantile",
    ),
    _Case(
        "get_bounds",
        lambda: QueryBuilder("t").groupby(_KEYS).get_bounds("B"),
        "GetBounds",
    ),
    _Case("get_groups", lambda: QueryBuilder("t").get_groups(["A"]), "GetGroups"),
    _Case(
        "automatic_partition_selection",
        lambda: QueryBuilder("t").groupby(["A"]).count(),
        "Automatic partition selection",
    ),
    _Case(
        "keyset_built_by_filtering",
        lambda: QueryBuilder("t").groupby(_KEYS.filter("A = 'a'")).count(),
        "Filter",
    ),
    _Case(
        "histogram",
        lambda: QueryBuilder("t").histogram("B", [0, 2, 4]),
        "GroupByCount",
    ),
    _Case("count", lambda: QueryBuilder("t").groupby(_KEYS).count(), "GroupByCount"),
]

_COUNT_DEPENDENT = {"histogram", "count"}
"""The cases that are only unsupported for as long as counting is."""


def _counted(case: _Case) -> _Case:
    """Turn a transformation case into a query, by counting over it."""
    return _Case(case.id, lambda: case.build().groupby(_KEYS).count(), case.op)


_CASES: List[_Case] = [
    case
    for case in [_counted(c) for c in _TRANSFORMATION_CASES] + _AGGREGATION_CASES
    if not (_COUNT_FACTORY_BOUND and case.id in _COUNT_DEPENDENT)
]
"""Every unsupported feature, as a query a Session can be asked to answer.

A transformation case is counted over so that there is something to evaluate;
the count is not what is being rejected -- the transformation under it is, since
the gate walks the query from its leaves up.
"""


def _check_rejection(excinfo, case: _Case) -> None:
    """Check one rejection: the right feature, the backend, and the doc."""
    error = excinfo.value
    assert isinstance(error, NotSupportedByBackend)
    assert error.op == case.op
    assert case.op in str(error)
    assert error.backend == "pandas"
    assert "pandas backend" in str(error)
    assert FEATURE_MATRIX_HINT in str(error)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_evaluate_rejects_unsupported_features(
    pandas_session: Session, forbid_spark, case: _Case
):
    """Evaluating an unsupported query names the feature, and costs nothing."""
    # pylint: disable=unused-argument
    query = case.build()
    before = pandas_session.remaining_privacy_budget
    with pytest.raises(NotSupportedByBackend) as excinfo:
        pandas_session.evaluate(query, PureDPBudget(1))
    _check_rejection(excinfo, case)
    assert pandas_session.remaining_privacy_budget == before


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_describe_reraises_unsupported_features(
    pandas_session: Session, forbid_spark, case: _Case, capsys
):
    """Describing an unsupported query raises rather than describing it.

    This is the contract ``Session._describe_query_obj`` pins with its separate
    ``except NotSupportedByBackend: raise`` clause: describing a query that
    cannot be evaluated would tell the user it is fine. Nothing is printed on
    the way out.
    """
    # pylint: disable=unused-argument
    query = case.build()
    with pytest.raises(NotSupportedByBackend) as excinfo:
        pandas_session.describe(query)
    _check_rejection(excinfo, case)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "case", _TRANSFORMATION_CASES, ids=[c.id for c in _TRANSFORMATION_CASES]
)
def test_create_view_rejects_unsupported_features(
    pandas_session: Session, forbid_spark, case: _Case
):
    """A view is compiled through the same gate as a query.

    ``create_view`` reaches the gate by a different route -- it compiles a
    transformation rather than a measurement -- and has to refuse the same
    things, leaving no view behind.
    """
    # pylint: disable=unused-argument
    query = case.build()
    with pytest.raises(NotSupportedByBackend) as excinfo:
        pandas_session.create_view(query, "view", cache=False)
    _check_rejection(excinfo, case)
    assert "view" not in pandas_session.private_sources


def test_a_supported_transformation_is_not_rejected(
    pandas_session: Session, forbid_spark
):
    """The gate refuses what is missing, not everything.

    A select is built from an ops slot the pandas backend has, so the gate lets
    it through -- as a view, which is where a transformation on its own can be
    compiled -- and does so without reaching Spark either.
    """
    # pylint: disable=unused-argument
    pandas_session.create_view(QueryBuilder("t").select(["A"]), "selected", cache=False)
    assert "selected" in pandas_session.private_sources
    pandas_session.delete_view("selected")


def test_keyset_from_a_spark_dataframe_is_rejected(pandas_session: Session, spark):
    """A KeySet holding a Spark DataFrame cannot be a pandas group-by.

    This is the one case that cannot run under ``forbid_spark``: building the
    keyset needs the Spark DataFrame it is built from. The rejection still
    happens at compile time, before the keyset is materialized.
    """
    keyset = KeySet.from_dataframe(spark.createDataFrame(pd.DataFrame({"A": ["a"]})))
    query = QueryBuilder("t").groupby(keyset).count()
    before = pandas_session.remaining_privacy_budget
    with pytest.raises(NotSupportedByBackend) as excinfo:
        pandas_session.evaluate(query, PureDPBudget(1))
    assert excinfo.value.op == "FromSparkDataFrame"
    assert excinfo.value.backend == "pandas"
    assert FEATURE_MATRIX_HINT in str(excinfo.value)
    assert pandas_session.remaining_privacy_budget == before


def test_partition_and_create_spends_nothing_and_needs_no_spark(
    pandas_session: Session, forbid_spark
):
    """Partitioning is refused first of all, before budget or Spark.

    ``test_session_pandas.py`` pins that the refusal comes before the budget
    math; this pins the other half of the same ordering, that it also comes
    before anything reaches Spark.
    """
    # pylint: disable=unused-argument
    before = pandas_session.remaining_privacy_budget
    with pytest.raises(NotSupportedByBackend) as excinfo:
        pandas_session.partition_and_create(
            "t", privacy_budget=PureDPBudget(1), column="A", splits={"p0": "a"}
        )
    assert excinfo.value.backend == "pandas"
    assert FEATURE_MATRIX_HINT in str(excinfo.value)
    assert pandas_session.remaining_privacy_budget == before


###############################################################################
# Positive controls: the Spark backend is untouched.
###############################################################################


@pytest.mark.parametrize(
    "case",
    _CASES + _TRANSFORMATION_CASES,
    ids=[c.id for c in _CASES] + [f"{c.id}_view" for c in _TRANSFORMATION_CASES],
)
def test_spark_rejects_nothing(case: _Case, forbid_spark):
    """Every query the pandas backend refuses, the Spark backend accepts.

    The gate is asked directly rather than through a Session: what is being
    checked is that it says yes, and evaluating these for real would need a
    JVM, a budget, and tables that some of them name and a Session here does
    not have. Asking it directly is also what lets this run under
    ``forbid_spark``, which pins that saying yes is as Spark-free as saying no
    -- the gate must not be doing any of the work it is gating.
    """
    # pylint: disable=protected-access,unused-argument
    check_supported(case.build()._query_expr, SPARK)


def test_the_spark_backend_is_missing_nothing():
    """No feature in the table is unavailable on Spark.

    The table is written for the backends that lack things; this is the check
    that nothing was written into it that quietly disables a feature everywhere.
    """
    assert unsupported_features(SPARK) == {}


def test_describing_a_filter_still_works_on_spark(spark):
    """The end-to-end path the gate sits in is unchanged on Spark."""
    session = Session.from_dataframe(
        PureDPBudget(1),
        "t",
        spark.createDataFrame(_DATA),
        protected_change=AddOneRow(),
    )
    session.describe(QueryBuilder("t").filter("B > 1"))


###############################################################################
# The table itself.
###############################################################################


def _concrete_query_expr_types() -> Set[type]:
    """Every QueryExpr type the query language itself defines.

    The abstract ones are dropped -- ``QueryExpr`` and ``SingleChildQueryExpr``
    exist to be inherited from and never appear in a query -- and so are the
    ones defined outside :mod:`tmlt.analytics._query_expr`, which are the stand
    -ins other test modules subclass ``QueryExpr`` to make.
    """
    found: Set[type] = set()
    pending: List[type] = [QueryExpr]
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass not in found:
                found.add(subclass)
                pending.append(subclass)
    return {
        subclass
        for subclass in found
        if not inspect.isabstract(subclass)
        and subclass.__module__ == QueryExpr.__module__
    }


def test_every_query_expr_type_is_in_the_table():
    """A new QueryExpr type has to say what it needs from a backend.

    Without this, a type added to the query language would default to "every
    backend can do this", which is the one answer that cannot be checked.
    """
    assert _concrete_query_expr_types() == set(REQUIRED_OPS)


def test_the_table_only_names_real_ops():
    """Every ops slot the table asks for is one a backend could bind."""
    # pylint: disable=protected-access
    slots = set(type(SPARK.ops)._fields)
    for expr_type, feature in REQUIRED_OPS.items():
        unknown = set(feature.ops) - slots
        assert not unknown, f"{expr_type.__name__} requires unknown ops {unknown}"


def test_pandas_is_missing_the_features_it_should_be():
    """The pandas feature matrix says what phase 1 said it would.

    Spelled out rather than derived, so that a slot bound by accident -- or a
    table entry loosened -- shows up here as a difference rather than as a
    silently widened surface.
    """
    expected = {
        "Filter",
        "FlatMap",
        "FlatMapByID",
        "JoinPublic",
        "ReplaceNullAndNan",
        "ReplaceInfinity",
        "DropNullAndNan",
        "DropInfinity",
        "GroupByBoundedSum",
        "GroupByBoundedAverage",
        "GroupByBoundedVariance",
        "GroupByBoundedStdev",
        "GroupByQuantile",
        "GetBounds",
        "GetGroups",
        "Automatic partition selection",
    }
    if not _COUNT_FACTORY_BOUND:
        expected |= {"GroupByCount", "GroupByCountDistinct"}
    assert set(unsupported_features(PANDAS)) == expected
