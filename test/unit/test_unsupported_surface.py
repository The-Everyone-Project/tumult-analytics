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
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tmlt.analytics import (
    AddOneRow,
    AddRowsWithID,
    ApproxDPBudget,
    ColumnType,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PureDPBudget,
    QueryBuilder,
    Session,
    TruncationStrategy,
)
from tmlt.analytics._backends import (
    FEATURE_MATRIX_HINT,
    PANDAS,
    SPARK,
    Backend,
    NotSupportedByBackend,
)
from tmlt.analytics._query_expr import QueryExpr
from tmlt.analytics._query_expr_compiler._backend_support import (
    REQUIRED_OPS,
    _children,
    _required_features,
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


PANDAS_UNSUPPORTED_FEATURES = {
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
} | (set() if _COUNT_FACTORY_BOUND else {"GroupByCount", "GroupByCountDistinct"})
"""What the pandas backend does not have in phase 1, named rather than counted.

Spelled out rather than derived, so that a slot bound by accident -- or a table
entry loosened -- shows up as a difference rather than as a silently widened
surface. Fifteen ``QueryExpr`` types plus automatic partition selection, which
is a feature of how a group-by finds its keys rather than a type of its own.

This is the one statement of the matrix's contents.
:mod:`test.system.backend_parity.test_capability_matrix` reads it from here, so
that the parity suite and this one cannot come to expect different matrices."""


def test_pandas_is_missing_the_features_it_should_be():
    """The pandas feature matrix says what phase 1 said it would."""
    assert set(unsupported_features(PANDAS)) == PANDAS_UNSUPPORTED_FEATURES


###############################################################################
# The table against the visitors.
###############################################################################
#
# REQUIRED_OPS is a second statement of something the visitors already say: the
# visitor asks for an ops slot by calling backend.require(), and the table says
# in advance which slots a query of that type will ask for. Two statements of
# one fact drift, and the drift is invisible for any op both backends bind --
# which today is most of them. What follows compiles one small query per table
# row and records what require() was actually asked for, so that the two are
# checked against each other rather than only against the backends that happen
# to exist.
#
# Drift in the two directions is not equally bad, and is not checked the same
# way. A row that lists an op the compile never asks for would reject a query
# the backend could have answered -- the failure the module docstring of
# _backend_support says the table must not have -- so that is an assertion. A
# row that omits one the compile does ask for merely delays the rejection to
# the require() call, which is still compile time and still costs no budget; it
# is checked against a named list, so that an omission is a decision on record
# rather than an oversight.


@dataclass(frozen=True)
class _GateCase:
    """One query, as a probe of what compiling its shape really requires."""

    id: str
    """The name this case is reported under."""

    build: Callable[[], Any]
    """Builds the query, once the feature flags are open."""

    approx_dp: bool = False
    """Whether this query needs the ApproxDP Session rather than the PureDP one."""

    evaluate: bool = False
    """Whether the ops have to be recorded from an evaluation rather than a compile.

    Automatic partition selection is the one feature whose Core objects are
    built inside the adaptive composition's callback (``perform_groupby_agg`` in
    the base measurement visitor), which does not run until the measurement
    does. Compiling such a query asks for none of them; evaluating it asks for
    all of them.
    """


def _map_fn(row):  # pragma: no cover -- shape only; the answers are discarded.
    """A map function returning one new column."""
    return {"D": 1}


def _flat_map_fn(row):  # pragma: no cover -- as above.
    """A flat-map function returning one row."""
    return [{"D": 1}]


def _flat_map_by_id_fn(rows):  # pragma: no cover -- as above.
    """A flat-map-by-id function returning one row per ID."""
    return [{"D": 1}]


_GATE_CASES: List[_GateCase] = [
    # Reading a table, and the transformations, each in both shapes: a query on
    # a plain table compiles through the DictMetric path, and the same query on
    # an AddRowsWithID table through the AddRemoveKeys path, which reaches the
    # *Value member of the same op family. Both shapes matter, because a row
    # that named an op only one of the two paths uses would be over-listing for
    # the other -- which is exactly what REQUIRED_OPS[JoinPrivate] used to do.
    _GateCase("private_source", lambda: QueryBuilder("t").groupby(_KEYS).count()),
    _GateCase(
        "private_source_ark",
        lambda: QueryBuilder("ids").enforce(MaxRowsPerID(2)).groupby(_KEYS).count(),
    ),
    _GateCase(
        "rename", lambda: QueryBuilder("t").rename({"B": "B2"}).groupby(_KEYS).count()
    ),
    _GateCase(
        "rename_ark",
        lambda: (
            QueryBuilder("ids")
            .rename({"B": "B2"})
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "filter", lambda: QueryBuilder("t").filter("B > 1").groupby(_KEYS).count()
    ),
    _GateCase(
        "filter_ark",
        lambda: (
            QueryBuilder("ids")
            .filter("B > 1")
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "select", lambda: QueryBuilder("t").select(["A", "B"]).groupby(_KEYS).count()
    ),
    _GateCase(
        "select_ark",
        lambda: (
            QueryBuilder("ids")
            .select(["A", "id"])
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "map",
        lambda: (
            QueryBuilder("t")
            .map(_map_fn, {"D": ColumnType.INTEGER}, augment=True)
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "map_ark",
        lambda: (
            QueryBuilder("ids")
            .map(_map_fn, {"D": ColumnType.INTEGER}, augment=True)
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "flat_map",
        lambda: (
            QueryBuilder("t")
            .flat_map(_flat_map_fn, {"D": "INTEGER"}, augment=True, max_rows=1)
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "flat_map_ark",
        lambda: (
            QueryBuilder("ids")
            .flat_map(_flat_map_fn, {"D": "INTEGER"}, augment=True)
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    # flat_map_by_id has only the one shape: it is defined on ID tables alone.
    _GateCase(
        "flat_map_by_id",
        lambda: (
            QueryBuilder("ids")
            .flat_map_by_id(_flat_map_by_id_fn, {"D": "INTEGER"})
            .enforce(MaxRowsPerID(2))
            .count()
        ),
    ),
    _GateCase(
        "join_private",
        lambda: (
            QueryBuilder("t")
            .join_private(
                QueryBuilder("t2"),
                truncation_strategy_left=TruncationStrategy.DropExcess(1),
                truncation_strategy_right=TruncationStrategy.DropExcess(1),
            )
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "join_private_ark",
        lambda: (
            QueryBuilder("ids")
            .join_private(QueryBuilder("ids2"))
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "join_public",
        lambda: QueryBuilder("t").join_public("pub").groupby(_KEYS).count(),
    ),
    _GateCase(
        "join_public_ark",
        lambda: (
            QueryBuilder("ids")
            .join_public("pub")
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "replace_null_and_nan",
        lambda: QueryBuilder("t").replace_null_and_nan().groupby(_KEYS).count(),
    ),
    _GateCase(
        "replace_null_and_nan_ark",
        lambda: (
            QueryBuilder("ids")
            .replace_null_and_nan()
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "replace_infinity",
        lambda: (
            QueryBuilder("t")
            .replace_infinity({"C": (-1.0, 1.0)})
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "replace_infinity_ark",
        lambda: (
            QueryBuilder("ids")
            .replace_infinity({"C": (-1.0, 1.0)})
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "drop_null_and_nan",
        lambda: QueryBuilder("t").drop_null_and_nan(["C"]).groupby(_KEYS).count(),
    ),
    _GateCase(
        "drop_null_and_nan_ark",
        lambda: (
            QueryBuilder("ids")
            .drop_null_and_nan(["C"])
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    _GateCase(
        "drop_infinity",
        lambda: QueryBuilder("t").drop_infinity(["C"]).groupby(_KEYS).count(),
    ),
    _GateCase(
        "drop_infinity_ark",
        lambda: (
            QueryBuilder("ids")
            .drop_infinity(["C"])
            .enforce(MaxRowsPerID(2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    # The three ID truncations, which are what an EnforceConstraint builds.
    _GateCase(
        "enforce_max_rows_per_id",
        lambda: QueryBuilder("ids").enforce(MaxRowsPerID(2)).groupby(_KEYS).count(),
    ),
    _GateCase(
        "enforce_max_groups_per_id",
        lambda: (
            QueryBuilder("ids")
            .enforce(MaxGroupsPerID("A", 2))
            .enforce(MaxRowsPerGroupPerID("A", 2))
            .groupby(_KEYS)
            .count()
        ),
    ),
    # The aggregations.
    _GateCase("count", lambda: QueryBuilder("t").groupby(_KEYS).count()),
    _GateCase(
        "count_distinct", lambda: QueryBuilder("t").groupby(_KEYS).count_distinct()
    ),
    _GateCase("sum", lambda: QueryBuilder("t").groupby(_KEYS).sum("B", low=0, high=10)),
    _GateCase(
        "average", lambda: QueryBuilder("t").groupby(_KEYS).average("B", low=0, high=10)
    ),
    _GateCase(
        "variance",
        lambda: QueryBuilder("t").groupby(_KEYS).variance("B", low=0, high=10),
    ),
    _GateCase(
        "stdev", lambda: QueryBuilder("t").groupby(_KEYS).stdev("B", low=0, high=10)
    ),
    _GateCase(
        "quantile",
        lambda: QueryBuilder("t").groupby(_KEYS).quantile("B", 0.5, low=0, high=10),
    ),
    _GateCase("get_bounds", lambda: QueryBuilder("t").groupby(_KEYS).get_bounds("B")),
    _GateCase(
        "get_groups", lambda: QueryBuilder("t").get_groups(["A"]), approx_dp=True
    ),
    _GateCase(
        "automatic_partition_selection",
        lambda: QueryBuilder("t").groupby(["A"]).count(),
        approx_dp=True,
        evaluate=True,
    ),
    _GateCase("suppress", lambda: QueryBuilder("t").groupby(_KEYS).count().suppress(5)),
]
"""One query per table row, in each shape that row compiles through."""

_CONDITIONAL_OPS = {
    # The two private-join paths. Which one a join takes is decided by the
    # tables' protected change, so neither is needed by every private join; see
    # REQUIRED_OPS[JoinPrivate].
    "PrivateJoin",
    "PrivateJoinOnKey",
    # The ID truncations. EnforceConstraint's row is empty because the
    # constraint, not the query expression, builds the truncation -- and which
    # truncation depends on which constraint.
    "LimitRowsPerGroup",
    "LimitRowsPerGroupValue",
    "LimitKeysPerGroup",
    "LimitKeysPerGroupValue",
    "LimitRowsPerKeyPerGroup",
    "LimitRowsPerKeyPerGroupValue",
    # A numeric aggregation drops nulls from its measure column first, but only
    # when that column is nullable -- which the gate, which sees a query and not
    # a schema, cannot know.
    "DropNulls",
    "DropNullsValue",
}
"""Ops a compile may ask for that the table does not name, and why.

Every one of these is conditional on something the gate deliberately does not
look at: the tables' protected change, or a column's nullability. Listing them
here is what keeps "the table omits this on purpose" distinct from "the table
forgot this", which is the failure the check below would otherwise be blind to.
"""


def _value_twin(op: str) -> Optional[str]:
    """The ``*Value`` member of an op's family, if it has one.

    Most transformations exist twice in :class:`~tmlt.analytics._backends.Ops`:
    ``Select`` transforms a table, and ``SelectValue`` transforms the table
    inside an AddRemoveKeys dictionary. They are one operation as far as
    REQUIRED_OPS is concerned -- the table names the family, and which member a
    query reaches depends on the protected change of the table it runs on, which
    the gate does not look at (see the module docstring of ``_backend_support``).
    That is sound only because no backend has one member without the other,
    which :func:`test_value_twins_are_bound_together` pins.
    """
    twin = f"{op}Value"
    # pylint: disable=protected-access
    return twin if twin in type(SPARK.ops)._fields else None


def _walk(expr: QueryExpr) -> Iterator[QueryExpr]:
    """Every node of a query tree, the gate's own walk."""
    # pylint: disable=protected-access
    yield expr
    for child in _children(expr):
        yield from _walk(child)


def _features_of(query: Any) -> Set[str]:
    """Every ops slot REQUIRED_OPS says this whole query needs.

    Read through the gate's own :func:`_required_features`, so that a feature
    the gate attributes to a node -- automatic partition selection, which
    belongs to no query expression type of its own -- is counted here too.
    """
    # pylint: disable=protected-access
    ops: Set[str] = set()
    for node in _walk(query._query_expr):
        for _, feature in _required_features(node):
            ops |= set(feature.ops)
    return ops


@pytest.fixture(name="gate_tables", scope="module")
def fixture_gate_tables(spark) -> Dict[str, Any]:
    """The Spark tables the probe queries run on."""
    return {
        "t": spark.createDataFrame(_DATA),
        "t2": spark.createDataFrame(pd.DataFrame({"A": ["a", "b"], "D": [1, 2]})),
        "ids": spark.createDataFrame(_DATA),
        "ids2": spark.createDataFrame(
            pd.DataFrame({"id": ["x", "y"], "A": ["a", "b"], "E": [1, 2]})
        ),
        "pub": spark.createDataFrame(pd.DataFrame({"A": ["a", "b"], "F": [7, 8]})),
    }


def _gate_session(tables: Dict[str, Any], budget) -> Session:
    """A Spark Session holding every table the probe queries name."""
    return (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_id_space("space")
        .with_private_dataframe("t", tables["t"], AddOneRow())
        .with_private_dataframe("t2", tables["t2"], AddOneRow())
        .with_private_dataframe("ids", tables["ids"], AddRowsWithID("id", "space"))
        .with_private_dataframe("ids2", tables["ids2"], AddRowsWithID("id", "space"))
        .with_public_dataframe("pub", tables["pub"])
        .build()
    )


@pytest.fixture(name="required_ops", scope="module")
def fixture_required_ops(gate_tables) -> Dict[str, Set[str]]:
    """What each probe query's compile actually asks ``require()`` for.

    Computed once for every case, rather than per test, because it is the
    expensive half: two Sessions and one compile each. ``Backend.require`` is
    the single funnel every construction site in the compiler goes through --
    that is the property the whole gate rests on -- so recording calls to it
    records exactly the set the gate is trying to predict.
    """
    pure_dp = _gate_session(gate_tables, PureDPBudget(1000))
    approx_dp = _gate_session(gate_tables, ApproxDPBudget(1000, 0.1))
    recorded: Dict[str, Set[str]] = {}
    original = Backend.require
    for case in _GATE_CASES:
        seen: Set[str] = set()

        def spy(self, op_name: str, _seen: Set[str] = seen) -> Any:
            _seen.add(op_name)
            return original(self, op_name)

        session = approx_dp if case.approx_dp else pure_dp
        budget = ApproxDPBudget(1, 1e-6) if case.approx_dp else PureDPBudget(1)
        query = case.build()
        Backend.require = spy  # type: ignore[method-assign]
        try:
            if case.evaluate:
                session.evaluate(query, budget)
            else:
                # pylint: disable=protected-access
                session._compile_and_get_info(query._query_expr, budget)
        finally:
            Backend.require = original  # type: ignore[method-assign]
        recorded[case.id] = seen
    return recorded


@pytest.mark.parametrize("case", _GATE_CASES, ids=[c.id for c in _GATE_CASES])
def test_the_table_lists_only_ops_the_compiler_asks_for(
    required_ops: Dict[str, Set[str]], case: _GateCase
):
    """Every op the table demands for this query, the compiler really needs.

    This is the direction that has to hold: an op listed but never asked for
    means the gate refuses a query on a backend that could have answered it. It
    is also the direction no backend catches, since a slot both backends bind is
    never missing at the point where it would show.

    An op counts as needed if either member of its family was asked for -- see
    :func:`_value_twin`.
    """
    recorded = required_ops[case.id]
    query = case.build()
    for op in _features_of(query):
        twin = _value_twin(op)
        assert op in recorded or (twin is not None and twin in recorded), (
            f"REQUIRED_OPS demands '{op}' for this query, but compiling it never"
            f" asked for it (or for {twin}). The table over-lists: a backend"
            f" without '{op}' would be refused a query it could answer. Asked"
            f" for: {sorted(recorded)}."
        )


@pytest.mark.parametrize("case", _GATE_CASES, ids=[c.id for c in _GATE_CASES])
def test_the_table_accounts_for_every_op_the_compiler_asks_for(
    required_ops: Dict[str, Set[str]], case: _GateCase
):
    """Nothing the compiler needs is missing from the table by accident.

    An op the table does not name is not a correctness problem -- the query
    still fails at ``require()``, at compile time, having spent nothing -- but
    it is a later and vaguer failure than the gate's, so each omission has to be
    a decision. :data:`_CONDITIONAL_OPS` is where those decisions are written
    down; anything else reaching here is one nobody made.
    """
    query = case.build()
    listed = _features_of(query)
    accounted = set(listed)
    for op in listed:
        twin = _value_twin(op)
        if twin is not None:
            accounted.add(twin)
    unaccounted = required_ops[case.id] - accounted - _CONDITIONAL_OPS
    assert not unaccounted, (
        f"Compiling this query asked for {sorted(unaccounted)}, which no"
        " REQUIRED_OPS row names and _CONDITIONAL_OPS does not excuse. Add the"
        " op to the row if every query of that type needs it, or to"
        " _CONDITIONAL_OPS -- with a reason -- if it depends on something the"
        " gate does not look at."
    )


@pytest.mark.parametrize("case", _GATE_CASES, ids=[c.id for c in _GATE_CASES])
def test_the_gate_rejects_on_every_op_it_lists(case: _GateCase):
    """Removing any one listed op is enough for the gate to refuse the query.

    The other half of the agreement: the test above says the compiler needs
    every op the table lists, and this says the gate acts on every op the table
    lists. A row could otherwise name an op that nothing ever reads.

    Nothing is compiled here -- a backend with a hole in it is handed straight
    to the gate -- so this half needs no Spark.
    """
    # pylint: disable=protected-access
    query = case.build()
    for op in _features_of(query):
        backend = replace(SPARK, ops=SPARK.ops._replace(**{op: None}))
        with pytest.raises(NotSupportedByBackend) as excinfo:
            check_supported(query._query_expr, backend)
        assert op in str(excinfo.value)


def test_every_table_row_has_a_probe_query():
    """Every row of the table is exercised by one of the cases above.

    A row nothing probes is a row the checks above say nothing about, which
    would make them quietly weaker as the query language grows.
    """
    covered: Set[type] = set()
    for case in _GATE_CASES:
        # pylint: disable=protected-access
        covered |= {type(node) for node in _walk(case.build()._query_expr)}
    assert covered == set(REQUIRED_OPS)


def test_value_twins_are_bound_together():
    """No backend has one member of an op family without the other.

    :func:`_value_twin` treats ``Select`` and ``SelectValue`` as one requirement,
    which is only sound while this holds. If a backend ever binds one without
    the other, the table has to name both members explicitly instead.
    """
    # pylint: disable=protected-access
    for backend in (SPARK, PANDAS):
        for op in type(backend.ops)._fields:
            twin = _value_twin(op)
            if twin is None:
                continue
            assert (getattr(backend.ops, op) is None) == (
                getattr(backend.ops, twin) is None
            ), f"{backend.name} binds exactly one of {op} and {twin}"


def test_every_conditional_op_is_a_real_slot():
    """The named exceptions name ops that exist."""
    # pylint: disable=protected-access
    unknown = _CONDITIONAL_OPS - set(type(SPARK.ops)._fields)
    assert not unknown, f"_CONDITIONAL_OPS names unknown ops {unknown}"
