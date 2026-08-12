"""The same noise, at the same price, over the same schema.

The P3 bar. :mod:`~test.system.backend_parity.test_answers` shows the two
backends compute the same *answer* at an infinite budget; that says nothing about
what happens at a finite one, where the answer is a random variable and the two
things a user can actually inspect are the noise the engine says it will add and
the budget it says it has left. Both are compared here, as objects, with no
tolerance: a backend that reached the same distribution by a different route
would still be a different engine, and a backend that charged a different price
for the same query would be a privacy bug.

The three things checked
========================

* **The mechanism and its parameter**, through ``Session._noise_info``, which
  reports what a query would cost without answering it. Compared directly
  between backends, and also pinned per backend so that the pandas half of the
  comparison can run without a JVM.
* **The budget ledger**, after a sequence of queries. Not arithmetic: an engine
  that answered from a fresh accountant every time would give exactly the same
  answers as this one and would have no privacy guarantee at all.
* **The schema each backend reports** for its tables, which is the one place the
  two are known not to agree, and where the interesting assertion is therefore
  that they disagree in *exactly* the documented way and nowhere else.

A note on exact arithmetic
==========================

Budgets are held as exact rationals, not floats, so
``RhoZCDPBudget(3)`` minus ``RhoZCDPBudget(0.1)`` is not
``RhoZCDPBudget(2.9)``: spending ``0.1`` subtracts the exact rational the float
``0.1`` denotes, and ``2.9`` is a different rational from ``3`` minus that one,
even though both print as ``2.9`` and the two floats are equal. The
cross-backend comparisons are unaffected -- both backends do the same exact
arithmetic -- but the *pinned* ledger test spends dyadic amounts, which are exact
in both representations, so that its expected values can be written down at all.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from tmlt.analytics import (
    KeySet,
    MaxRowsPerID,
    PrivacyBudget,
    PureDPBudget,
    Query,
    QueryBuilder,
    RhoZCDPBudget,
    Session,
)
from tmlt.analytics._noise_info import _NoiseMechanism
from tmlt.analytics._schema import ColumnType

from test.backend_testing import (
    BACKEND_NAMES,
    STANDARD_TABLES,
    BackendFixture,
    TableSpec,
    all_nullability_divergences,
    analytics_columns,
    nullability_divergences,
)
from test.system.backend_parity.tables import (
    ARK,
    CHOICE,
    CHOICE_SPEC,
    G_KEYS,
    ID_SPEC,
    JOIN,
    JOIN_LEFT_SPEC,
    JOIN_RIGHT_SPEC,
    PLAIN,
    PLAIN_MAX_ROWS,
    ROWS_SPEC,
    build_session,
)

################################################################################
# The noise a query would add
################################################################################

_AMOUNTS = (0.1, 0.5, 2.0)
"""The finite budgets the calibration bar is measured at."""

_MECHANISMS = {
    "rho-zCDP": _NoiseMechanism.DISCRETE_GAUSSIAN,
    "pure-DP": _NoiseMechanism.GEOMETRIC,
}
"""Which mechanism each budget flavor selects."""


def _budget(kind: str, amount: float) -> PrivacyBudget:
    """Builds a budget of the given flavor.

    Args:
        kind: ``"rho-zCDP"`` or ``"pure-DP"``.
        amount: The rho or epsilon.

    Raises:
        ValueError: If the flavor is neither.
    """
    if kind == "rho-zCDP":
        return RhoZCDPBudget(amount)
    if kind == "pure-DP":
        return PureDPBudget(amount)
    raise ValueError(f"Unknown budget flavor {kind!r}.")


_NOISE_PARAMETER: Dict[Tuple[str, str, float], float] = {
    # AddOneRow: sensitivity 1. For zCDP the parameter is sigma squared,
    # 1 / (2 rho); for pure DP it is the geometric alpha, 1 / epsilon. The
    # inexact-looking values are exactly what the engine returns -- the float
    # arithmetic behind them lands one ulp below the integer -- and are pinned as
    # returned rather than rounded, because "equal as returned" is the property.
    (PLAIN, "rho-zCDP", 0.1): 4.999999999999999,
    (PLAIN, "rho-zCDP", 0.5): 1,
    (PLAIN, "rho-zCDP", 2.0): 0.25,
    (PLAIN, "pure-DP", 0.1): 9.999999999999998,
    (PLAIN, "pure-DP", 0.5): 2,
    (PLAIN, "pure-DP", 2.0): 0.5,
    # AddMaxRows(3): sensitivity 3, so nine times the zCDP parameter and three
    # times the pure-DP one.
    (PLAIN_MAX_ROWS, "rho-zCDP", 0.1): 44.99999999999999,
    (PLAIN_MAX_ROWS, "rho-zCDP", 0.5): 9,
    (PLAIN_MAX_ROWS, "rho-zCDP", 2.0): 2.25,
    (PLAIN_MAX_ROWS, "pure-DP", 0.1): 29.99999999999999,
    (PLAIN_MAX_ROWS, "pure-DP", 0.5): 6,
    (PLAIN_MAX_ROWS, "pure-DP", 2.0): 1.5,
    # AddRowsWithID under MaxRowsPerID(2): an ID contributes at most two rows, so
    # the sensitivity is 2 under both norms and the two flavors' parameters
    # coincide -- 4 / (2 rho) and 2 / epsilon.
    (ARK, "rho-zCDP", 0.1): 19.99999999999999,
    (ARK, "rho-zCDP", 0.5): 4,
    (ARK, "rho-zCDP", 2.0): 1,
    (ARK, "pure-DP", 0.1): 19.99999999999999,
    (ARK, "pure-DP", 0.5): 4,
    (ARK, "pure-DP", 2.0): 1,
}
"""The noise parameter each layout, flavor and amount produces.

Independent of how the query groups and of which count it is, which is why the
key does not mention either: the parameter depends on the sensitivity the
protected change and the constraints buy, and on the budget spent, and on
nothing else. :func:`test_the_noise_parameter_does_not_depend_on_the_grouping`
is what keeps that claim honest.
"""


def _query(layout: str, grouping: Optional[KeySet], aggregation: str) -> Query:
    """Builds the query the calibration tests measure.

    Args:
        layout: The Session layout, which decides whether a constraint is needed.
        grouping: The keyset to group by, or ``None`` for an ungrouped count.
        aggregation: ``"count"`` or ``"count_distinct"``.

    Returns:
        The query.
    """
    builder = QueryBuilder("t")
    if layout == ARK:
        builder = builder.enforce(MaxRowsPerID(2))
    grouped = builder if grouping is None else builder.groupby(grouping)
    return grouped.count() if aggregation == "count" else grouped.count_distinct(["v"])


_LAYOUTS = (PLAIN, PLAIN_MAX_ROWS, ARK)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("kind", sorted(_MECHANISMS))
@pytest.mark.parametrize("amount", _AMOUNTS)
def test_the_noise_a_query_would_add(
    backend: BackendFixture, layout: str, kind: str, amount: float
):
    """Each backend reports exactly the mechanism and parameter expected.

    The two backends assemble different Core measurements -- pandas ones and
    Spark ones -- so this is not a formality: they have to agree about the noise
    despite computing it in different code.
    """
    session = build_session(backend, layout, _budget(kind, 1000))
    query = _query(layout, G_KEYS, "count")
    info = session._noise_info(query, _budget(kind, amount))
    assert list(info) == [
        {
            "noise_mechanism": _MECHANISMS[kind],
            "noise_parameter": _NOISE_PARAMETER[(layout, kind, amount)],
        }
    ]
    # Asking what a query would cost costs nothing.
    assert session.remaining_privacy_budget == _budget(kind, 1000)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("kind", sorted(_MECHANISMS))
@pytest.mark.parametrize("amount", _AMOUNTS)
def test_the_noise_is_the_same_on_both_backends(
    spark, layout: str, kind: str, amount: float
):
    """``_noise_info`` is equal between the backends, as returned.

    The direct form of the comparison above, with nothing written down for the
    two to agree with: whatever ``_noise_info`` returns, the two backends return
    the same thing, including the exact float.

    Args:
        spark: The Spark session; this test needs both backends at once.
        layout: The Session layout.
        kind: The budget flavor.
        amount: The budget spent.
    """
    query = _query(layout, G_KEYS, "count")
    reported = {}
    for name in BACKEND_NAMES:
        fixture = BackendFixture(name=name, spark=spark if name == "spark" else None)
        with fixture.feature_flag():
            session = build_session(fixture, layout, _budget(kind, 1000))
            reported[name] = list(session._noise_info(query, _budget(kind, amount)))
    assert reported["pandas"] == reported["spark"]


@pytest.mark.parametrize("kind", sorted(_MECHANISMS))
def test_the_noise_parameter_does_not_depend_on_the_grouping(
    backend: BackendFixture, kind: str
):
    """One parameter per (layout, flavor, budget), whatever the query groups by.

    :data:`_NOISE_PARAMETER` is keyed as though the grouping and the choice of
    count made no difference. They do not, and this is what says so -- without
    it, that table would be quietly asserting less than it looks like it does.
    """
    session = build_session(backend, ARK, _budget(kind, 1000))
    budget = _budget(kind, 0.5)
    expected = [
        {
            "noise_mechanism": _MECHANISMS[kind],
            "noise_parameter": _NOISE_PARAMETER[(ARK, kind, 0.5)],
        }
    ]
    for grouping in (None, G_KEYS):
        for aggregation in ("count", "count_distinct"):
            info = session._noise_info(_query(ARK, grouping, aggregation), budget)
            assert list(info) == expected, (grouping, aggregation)


################################################################################
# The budget ledger
################################################################################

_DYADIC_SPENDS = (0.5, 1.0, 2.0)
"""The amounts the pinned ledger test spends, from a total of four.

Dyadic rationals, so that the remaining budget after each is a value that can be
written down: see the module docstring on exact arithmetic.
"""

_DYADIC_REMAINING = (3.5, 2.5, 0.5)
"""What is left after each of :data:`_DYADIC_SPENDS`, in order."""


@pytest.mark.parametrize("kind", sorted(_MECHANISMS))
def test_the_ledger_after_a_sequence_of_queries(backend: BackendFixture, kind: str):
    """Each query costs what it was given, and the ledger keeps the running total.

    Compared as budget objects rather than as numbers, which is the comparison
    that would catch a backend whose accountant tracked the same value in a
    different flavor.
    """
    session = build_session(backend, ARK, _budget(kind, 4.0))
    query = _query(ARK, G_KEYS, "count")
    assert session.remaining_privacy_budget == _budget(kind, 4.0)

    for spend, remaining in zip(_DYADIC_SPENDS, _DYADIC_REMAINING):
        session.evaluate(query, _budget(kind, spend))
        assert session.remaining_privacy_budget == _budget(kind, remaining)

    # And the ledger is enforced, not just reported.
    with pytest.raises(RuntimeError, match="budget"):
        session.evaluate(query, _budget(kind, 1.0))


@pytest.mark.parametrize("kind", sorted(_MECHANISMS))
def test_the_ledger_is_the_same_on_both_backends(spark, kind: str):
    """The two accountants charge the same price for the same three queries.

    Run with the budgets the calibration bar uses rather than the dyadic ones, so
    that the comparison covers the case where the remaining budget is a rational
    no float denotes -- the case a hand-written expectation cannot state, and the
    one where two implementations could most easily drift apart.

    Args:
        spark: The Spark session; this test needs both backends at once.
        kind: The budget flavor.
    """
    query = _query(ARK, G_KEYS, "count")
    ledgers: Dict[str, List[PrivacyBudget]] = {}
    for name in BACKEND_NAMES:
        fixture = BackendFixture(name=name, spark=spark if name == "spark" else None)
        with fixture.feature_flag():
            session = build_session(fixture, ARK, _budget(kind, sum(_AMOUNTS)))
            remaining = []
            for amount in _AMOUNTS:
                session.evaluate(query, _budget(kind, amount))
                remaining.append(session.remaining_privacy_budget)
            ledgers[name] = remaining

    assert ledgers["pandas"] == ledgers["spark"]

    # The sequence really ran the budget down, rather than agreeing about three
    # untouched budgets: it strictly decreases, and a fourth query of the same
    # size is refused. Note what it does *not* end at -- exactly zero. The total
    # was the float sum of the three amounts, and subtracting each amount as an
    # exact rational leaves a residue of 3/36028797018963968, because the float
    # sum is not the exact sum. Both backends leave the same residue, which is
    # the assertion above; this one only needs it to be spent down.
    values = [budget.value for budget in ledgers["pandas"]]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values)
    for name in BACKEND_NAMES:
        fixture = BackendFixture(name=name, spark=spark if name == "spark" else None)
        with fixture.feature_flag():
            session = build_session(fixture, ARK, _budget(kind, sum(_AMOUNTS)))
            for amount in _AMOUNTS:
                session.evaluate(query, _budget(kind, amount))
            with pytest.raises(RuntimeError, match="budget"):
                session.evaluate(query, _budget(kind, min(_AMOUNTS)))


################################################################################
# The schema each backend reports
################################################################################

_SPECS: Mapping[str, Tuple[str, TableSpec]] = {
    "rows": (PLAIN, ROWS_SPEC),
    "ids": (ARK, ID_SPEC),
    "choice": (CHOICE, CHOICE_SPEC),
}
"""One table of each shape the suite uses, with the layout that holds it."""


@pytest.mark.parametrize("case", sorted(_SPECS))
def test_a_session_reports_the_schema_the_harness_predicts(
    backend: BackendFixture, case: str
):
    """``get_schema`` is exactly what the harness's oracle says it will be.

    :func:`~test.backend_testing.materialize.expected_columns` is that oracle:
    the spec's own schema for Spark, and the spec's schema with ``allow_null``
    widened on every ``VARCHAR``, ``DATE`` and ``TIMESTAMP`` column for pandas,
    because those types' pandas dtypes always admit a null. Asserting against it
    rather than against the spec is the difference between testing the backends
    and testing a fiction that holds on neither.
    """
    layout, spec = _SPECS[case]
    session = build_session(backend, layout, RhoZCDPBudget(1))
    assert dict(session.get_schema("t")) == backend.expected_columns(spec)


def test_the_join_tables_schemas_are_predicted_too(backend: BackendFixture):
    """The same, for the two-table layout."""
    session = build_session(backend, JOIN, RhoZCDPBudget(1))
    assert dict(session.get_schema("l")) == backend.expected_columns(JOIN_LEFT_SPEC)
    assert dict(session.get_schema("r")) == backend.expected_columns(JOIN_RIGHT_SPEC)


@pytest.mark.parametrize("case", sorted(_SPECS))
def test_the_schemas_differ_in_exactly_the_documented_columns(spark, case: str):
    """The backends' schemas disagree where the harness says, and nowhere else.

    This is the assertion the divergence exists for. It is not "the schemas are
    equal" -- they are not, and cannot be -- and it is not "the schemas differ",
    which would pass if they differed everywhere. It is that the set of columns
    they differ on is exactly
    :func:`~test.backend_testing.materialize.nullability_divergences`, so a new
    disagreement of any kind fails this test.

    Args:
        spark: The Spark session; this test needs both backends at once.
        case: The table under test.
    """
    layout, spec = _SPECS[case]
    schemas = {}
    for name in BACKEND_NAMES:
        fixture = BackendFixture(name=name, spark=spark if name == "spark" else None)
        with fixture.feature_flag():
            session = build_session(fixture, layout, RhoZCDPBudget(1))
            schemas[name] = dict(session.get_schema("t"))

    assert set(schemas["pandas"]) == set(schemas["spark"])
    differing = {
        column
        for column in schemas["pandas"]
        if schemas["pandas"][column] != schemas["spark"][column]
    }
    assert differing == {column for column, _ in nullability_divergences(spec)}

    # And every difference is in the same direction: pandas widens a
    # non-nullable column to nullable, never the other way round.
    for column in differing:
        assert schemas["pandas"][column].allow_null is True
        assert schemas["spark"][column].allow_null is False
        assert (
            schemas["pandas"][column].column_type
            is schemas["spark"][column].column_type
        )


def test_the_standard_tables_diverge_exactly_where_documented():
    """The harness's own list of divergences is the one its docstring states.

    Three columns among the standard tables, and no others. Stated here as well
    as in the harness so that a change to either is a failure rather than a
    documentation drift.
    """
    divergences = all_nullability_divergences(list(STANDARD_TABLES.values()))
    assert {
        table: [column for column, _ in columns]
        for table, columns in divergences.items()
    } == {"id3": ["group"], "id4": ["group"], "rows1": ["A"]}


def test_an_answers_integer_columns_are_nullable_on_spark_and_not_on_pandas(spark):
    """The divergence in the other direction, which the results all show.

    Every ``INTEGER`` column of an *answer* -- the count, and an integer group-by
    key -- is reported ``allow_null=True`` by the Spark backend and
    ``allow_null=False`` by the pandas one. That is the converse of the input
    divergence above, and the harness's own docstring says it cannot happen to a
    materialized spec; it happens to every result, because Core leaves a Spark
    ``LongType`` field at its permissive default while a pandas answer arrives in
    a plain ``int64``, whose dtype cannot hold a null.

    Neither backend is wrong about the data: no count is ever null. But it is a
    difference in the schema a user reads back, so it is asserted rather than
    left to be discovered.

    Args:
        spark: The Spark session; this test needs both backends at once.
    """
    budget = RhoZCDPBudget(float("inf"))
    query = (
        QueryBuilder("t")
        .enforce(MaxRowsPerID(2))
        .groupby(KeySet.from_dict({"g": ["a", "b"], "v": [1, 2]}))
        .count()
    )
    columns: Dict[str, Dict[str, Any]] = {}
    for name in BACKEND_NAMES:
        fixture = BackendFixture(name=name, spark=spark if name == "spark" else None)
        with fixture.feature_flag():
            session = build_session(fixture, CHOICE, budget)
            columns[name] = dict(analytics_columns(session.evaluate(query, budget)))

    integer_columns = sorted(
        name
        for name, descriptor in columns["pandas"].items()
        if descriptor.column_type is ColumnType.INTEGER
    )
    # The count and the integer group-by key: both of them, not just the count.
    assert integer_columns == ["count", "v"]
    for column in integer_columns:
        assert columns["pandas"][column].allow_null is False, column
        assert columns["spark"][column].allow_null is True, column

    # The string group-by key is the one column the two agree about, and they
    # agree on the permissive answer.
    assert columns["pandas"]["g"].allow_null is True
    assert columns["spark"]["g"].allow_null is True


def test_the_session_is_the_backend_under_test(backend: BackendFixture):
    """A finite-budget Session is built on the backend the test is running for.

    Every expectation in this module is per-backend, so a Session that had
    quietly chosen Spark would satisfy the Spark ones and hide the pandas half of
    the suite entirely.
    """
    session = build_session(backend, ARK, RhoZCDPBudget(1))
    assert isinstance(session, Session)
    assert session._backend is backend.descriptor
