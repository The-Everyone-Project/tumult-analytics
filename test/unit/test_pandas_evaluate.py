"""End-to-end tests: a pandas Session answering queries.

This is where the pandas backend stops being a table of Core artifacts and
starts being an engine. Everything here goes through the public path a user
takes -- ``Session.Builder`` with a pandas frame, a
:class:`~tmlt.analytics.QueryBuilder` query, and
:meth:`~tmlt.analytics.Session.evaluate` -- and checks the answer, not just that
nothing raised.

Why the answers can be checked exactly
======================================

Every budget here is infinite. A differentially private count at infinite budget
adds noise of scale zero, so the answer is the true one, and the tests can
compare against counts worked out by hand rather than against a tolerance. That
makes them tests of the *compiler*: what they would catch is a query compiled
into the wrong pipeline -- a truncation applied twice, a groupby that drops its
zero-count keys, a keyset materialized in the wrong column order -- none of
which a noisy assertion could distinguish from bad luck.

The one place a budget is finite is
:func:`test_evaluating_twice_spends_the_budget_twice`, which is about the
accountant rather than the arithmetic.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Iterator, List

import pandas as pd
import pytest

from tmlt.analytics import (
    AddMaxRows,
    AddRowsWithID,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PrivacyBudget,
    PureDPBudget,
    Query,
    QueryBuilder,
    RhoZCDPBudget,
    Session,
)
from tmlt.analytics._backends import PANDAS
from tmlt.analytics._noise_info import _NoiseMechanism
from tmlt.analytics._schema import Schema
from tmlt.analytics.config import config

INF = float("inf")

_UNBOUNDED_BUDGETS = [
    pytest.param(RhoZCDPBudget(INF), id="rho-zCDP"),
    pytest.param(PureDPBudget(INF), id="pure-DP"),
]
"""The budgets every exact-answer test is run under.

Two budgets rather than one because the budget picks the noise mechanism --
discrete Gaussian for zCDP, geometric for pure DP -- and so picks which of
Core's pandas measurements the count factory assembles. At infinite budget both
add nothing, so both must produce the same true answer.
"""

###############################################################################
# The data.
###############################################################################

# A: the groupby column, with one key ("d") that appears in no row, so that
#    zero-filling is exercised by every groupby test.
# G: a second groupby column, for the two-column keyset.
# B: a value column with repeats inside a group, so that count_distinct and
#    count differ.
ROWS = pd.DataFrame(
    {
        "A": ["a", "a", "a", "b", "c", "c", "c"],
        "G": ["x", "x", "y", "x", "y", "y", "y"],
        "B": [1, 1, 2, 3, 4, 5, 5],
    }
)

# Every row of an ID belongs to one group, deliberately: then a per-ID
# truncation's effect on a per-group count does not depend on *which* rows Core
# drops, only on how many, and the expected counts below are exact without
# reaching into Core's truncation order.
#
#   i1: three "a" rows      i2: one "b" row
#   i3: two "c" rows        i4: four "c" rows
ID_ROWS = pd.DataFrame(
    {
        "id": ["i1", "i1", "i1", "i2", "i3", "i3", "i4", "i4", "i4", "i4"],
        "A": ["a", "a", "a", "b", "c", "c", "c", "c", "c", "c"],
        "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
)

A_KEYS = KeySet.from_dict({"A": ["a", "b", "c", "d"]})


@pytest.fixture(name="pandas_enabled", autouse=True)
def fixture_pandas_enabled() -> Iterator[None]:
    """Run every test in this module with the pandas backend enabled."""
    with config.features.pandas_backend.enabled():
        yield


def _session(budget: PrivacyBudget) -> Session:
    """A pandas Session over ``ROWS``, protecting one row."""
    session = (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_private_dataframe("t", ROWS, AddMaxRows(1))
        .build()
    )
    # Every expectation below is a pandas one; a Session that had quietly
    # chosen Spark would still answer, and answer correctly.
    assert session._backend is PANDAS  # pylint: disable=protected-access
    return session


def _id_session(budget: PrivacyBudget) -> Session:
    """A pandas Session over ``ID_ROWS``, protecting one ID."""
    session = (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_id_space("ids")
        .with_private_dataframe("t", ID_ROWS, AddRowsWithID("id", "ids"))
        .build()
    )
    assert session._backend is PANDAS  # pylint: disable=protected-access
    return session


def _normalized(frame: pd.DataFrame, sort_by: List[str]) -> pd.DataFrame:
    """A frame in a form two of them can be compared by.

    An answer's row order is not part of it -- neither backend promises one --
    and neither is the index a groupby happens to leave behind.
    """
    return frame.sort_values(sort_by, ignore_index=True)[list(frame.columns)]


def _expected(rows: List[tuple], columns: List[str]) -> pd.DataFrame:
    """The frame a hand-computed answer is compared against."""
    return pd.DataFrame(rows, columns=columns)


def _assert_answer(
    actual: pd.DataFrame, expected: pd.DataFrame, sort_by: List[str]
) -> None:
    """Assert an answer is exactly the expected frame, and is a pandas one."""
    assert isinstance(actual, pd.DataFrame), (
        "A pandas Session must answer with a pandas frame, but answered with a "
        f"{type(actual).__name__}."
    )
    pd.testing.assert_frame_equal(
        _normalized(actual, sort_by), _normalized(expected, sort_by)
    )


###############################################################################
# (a) Groupby count and count_distinct over a public KeySet.
###############################################################################


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_groupby_count(budget: PrivacyBudget):
    """A grouped count is the true count in every group, and zero elsewhere."""
    session = _session(budget)
    answer = session.evaluate(QueryBuilder("t").groupby(A_KEYS).count(), budget)
    _assert_answer(
        answer,
        _expected([("a", 3), ("b", 1), ("c", 3), ("d", 0)], ["A", "count"]),
        ["A"],
    )


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_groupby_count_distinct(budget: PrivacyBudget):
    """A grouped count_distinct counts distinct values, not rows.

    Group "a" has three rows but two distinct ``B`` values, and group "c" has
    three rows and two distinct values, so an answer that counted rows would be
    caught here rather than agreeing by accident.
    """
    session = _session(budget)
    answer = session.evaluate(
        QueryBuilder("t").groupby(A_KEYS).count_distinct(["B"]), budget
    )
    _assert_answer(
        answer,
        _expected([("a", 2), ("b", 1), ("c", 2), ("d", 0)], ["A", "count_distinct(B)"]),
        ["A"],
    )


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_groupby_two_columns_fills_every_cell(budget: PrivacyBudget):
    """A two-column keyset answers for its whole cross product.

    Four of these eight cells hold no rows at all, including a whole value of
    ``A``. Every one of them has to come back as a zero: a groupby that answered
    only for the cells it saw would be a disclosure, since the missing rows
    would say which combinations exist.
    """
    session = _session(budget)
    keys = KeySet.from_dict({"A": ["a", "b", "c", "d"], "G": ["x", "y"]})
    answer = session.evaluate(QueryBuilder("t").groupby(keys).count(), budget)
    _assert_answer(
        answer,
        _expected(
            [
                ("a", "x", 2),
                ("a", "y", 1),
                ("b", "x", 1),
                ("b", "y", 0),
                ("c", "x", 0),
                ("c", "y", 3),
                ("d", "x", 0),
                ("d", "y", 0),
            ],
            ["A", "G", "count"],
        ),
        ["A", "G"],
    )


def test_the_answer_is_a_pandas_frame_of_the_right_dtypes():
    """The answer is a pandas frame, and its columns are typed as they should be.

    A count is an ``INTEGER`` column, which
    :mod:`~tmlt.analytics._coerce_pandas_schema` maps to ``int64``, and the
    grouping column keeps the dtype it was ingested with. Checking the dtypes
    and not only the values is what catches a count coming back as an object
    column of Python ints, which compares equal cell by cell.
    """
    session = _session(RhoZCDPBudget(INF))
    answer = session.evaluate(
        QueryBuilder("t").groupby(A_KEYS).count(), RhoZCDPBudget(INF)
    )
    assert isinstance(answer, pd.DataFrame)
    assert list(answer.columns) == ["A", "count"]
    assert answer["A"].dtype == ROWS["A"].dtype
    assert answer["count"].dtype == "int64"


###############################################################################
# (b) The scalar path: a total count, with no groupby.
###############################################################################


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_total_count(budget: PrivacyBudget):
    """A count with no groupby counts the whole table."""
    session = _session(budget)
    answer = session.evaluate(QueryBuilder("t").count(), budget)
    _assert_answer(answer, _expected([(len(ROWS),)], ["count"]), ["count"])


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_total_count_distinct(budget: PrivacyBudget):
    """A count_distinct with no groupby counts distinct rows of the whole table.

    ``ROWS`` has one duplicated row -- the two ``(a, x, 1)`` rows -- so this is
    one less than :func:`test_total_count`.
    """
    session = _session(budget)
    answer = session.evaluate(QueryBuilder("t").count_distinct(), budget)
    _assert_answer(
        answer,
        _expected([(len(ROWS.drop_duplicates()),)], ["count_distinct"]),
        ["count_distinct"],
    )


###############################################################################
# (d) Rename, select and map, on a table with no ID column.
###############################################################################


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_rename_map_select_chain(budget: PrivacyBudget):
    """A chain of transformations compiles and answers on pandas.

    The point is the three visitor paths, not the arithmetic: the query renames
    the grouping column, maps a new one out of ``B``, drops the rest, and groups
    by what is left. The counts are the same as
    :func:`test_groupby_count`'s, because none of those steps changes how many
    rows there are.
    """
    session = _session(budget)
    query = (
        QueryBuilder("t")
        .rename({"A": "grp"})
        .map(
            lambda row: {"C": row["B"] * 2},
            new_column_types=Schema({"C": "INTEGER"}),
            augment=True,
        )
        .select(["grp", "C"])
        .groupby(KeySet.from_dict({"grp": ["a", "b", "c", "d"]}))
        .count()
    )
    answer = session.evaluate(query, budget)
    _assert_answer(
        answer,
        _expected([("a", 3), ("b", 1), ("c", 3), ("d", 0)], ["grp", "count"]),
        ["grp"],
    )


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_map_result_is_counted(budget: PrivacyBudget):
    """The mapped column is real: grouping by it counts what the map produced."""
    session = _session(budget)
    query = (
        QueryBuilder("t")
        .map(
            lambda row: {"C": row["B"] * 2},
            new_column_types=Schema({"C": "INTEGER"}),
            augment=True,
        )
        .groupby(KeySet.from_dict({"C": [2, 4, 6, 8, 10, 12]}))
        .count()
    )
    answer = session.evaluate(query, budget)
    # B is [1, 1, 2, 3, 4, 5, 5], so C is [2, 2, 4, 6, 8, 10, 10].
    _assert_answer(
        answer,
        _expected([(2, 2), (4, 1), (6, 1), (8, 1), (10, 2), (12, 0)], ["C", "count"]),
        ["C"],
    )


###############################################################################
# (c) The AddRowsWithID path: constraints, and the *Value wrappers they build.
###############################################################################


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_ark_max_rows_per_id(budget: PrivacyBudget):
    """Enforcing MaxRowsPerID truncates each ID, then the count is exact.

    The truncation keeps at most two rows per ID, and every ID's rows are in one
    group, so the answer follows from the ID sizes alone: "a" loses one of its
    three rows, "b" keeps its one, and "c" keeps two from each of the two IDs
    that contribute to it.
    """
    session = _id_session(budget)
    query = QueryBuilder("t").enforce(MaxRowsPerID(2)).groupby(A_KEYS).count()
    answer = session.evaluate(query, budget)
    _assert_answer(
        answer,
        _expected([("a", 2), ("b", 1), ("c", 4), ("d", 0)], ["A", "count"]),
        ["A"],
    )


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_ark_max_groups_and_rows_per_group_per_id(budget: PrivacyBudget):
    """The per-group ID constraints truncate as they say they do.

    Each ID here is in exactly one group, so ``MaxGroupsPerID`` drops nothing
    and ``MaxRowsPerGroupPerID(1)`` leaves each ID one row: the count of a group
    becomes the number of IDs in it.
    """
    session = _id_session(budget)
    query = (
        QueryBuilder("t")
        .enforce(MaxGroupsPerID("A", 1))
        .enforce(MaxRowsPerGroupPerID("A", 1))
        .groupby(A_KEYS)
        .count()
    )
    answer = session.evaluate(query, budget)
    _assert_answer(
        answer,
        _expected([("a", 1), ("b", 1), ("c", 2), ("d", 0)], ["A", "count"]),
        ["A"],
    )


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_ark_rename_map_select_before_enforce(budget: PrivacyBudget):
    """Transformations applied to a table that still has its ID column.

    This is the same chain as :func:`test_rename_map_select_chain`, but on a
    table whose protected change is ``AddRowsWithID``. Nothing about the query
    says so, and that is the point: the compiler reaches the ``*Value`` wrappers
    -- ``RenameValue``, ``MapValue``, ``SelectValue`` -- rather than the plain
    transformations, because the table is a value of the ID dictionary.
    """
    session = _id_session(budget)
    query = (
        QueryBuilder("t")
        .rename({"A": "grp"})
        .map(
            lambda row: {"C": row["B"] * 2},
            new_column_types=Schema({"C": "INTEGER"}),
            augment=True,
        )
        .select(["id", "grp", "C"])
        .enforce(MaxRowsPerID(2))
        .groupby(KeySet.from_dict({"grp": ["a", "b", "c", "d"]}))
        .count()
    )
    answer = session.evaluate(query, budget)
    _assert_answer(
        answer,
        _expected([("a", 2), ("b", 1), ("c", 4), ("d", 0)], ["grp", "count"]),
        ["grp"],
    )


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_create_view_over_an_id_table(budget: PrivacyBudget):
    """A view over an ``AddRowsWithID`` table can be created and queried.

    Regression test: of the table-plumbing helpers ``create_view`` chains,
    ``rename_table`` was the one call in ``session.py`` that did not pass the
    Session's backend. On the ID path the rename is a ``RenameValue``, so the
    default built Spark's against a pandas domain and the view failed on
    exactly this shape -- while a view over a non-ID table sailed through.
    """
    session = _id_session(budget)
    session.create_view(QueryBuilder("t").enforce(MaxRowsPerID(2)), "v", cache=False)
    answer = session.evaluate(QueryBuilder("v").groupby(A_KEYS).count(), budget)
    _assert_answer(
        answer,
        _expected([("a", 2), ("b", 1), ("c", 4), ("d", 0)], ["A", "count"]),
        ["A"],
    )


@pytest.mark.parametrize("budget", _UNBOUNDED_BUDGETS)
def test_describe_reports_the_group_count(
    budget: PrivacyBudget, capsys: pytest.CaptureFixture
):
    """``describe`` of a grouped query counts the groups on this backend.

    Regression test: the group count in the description sizes the KeySet, and
    was the one KeySet call in ``session.py`` that did not pass the Session's
    backend. The keyset here is a join, whose size cannot be read off its
    operands -- sizing it materializes it, so on a pandas Session the Spark
    default booted the JVM this module's test lane forbids. (A ``from_dict``
    keyset would not catch this: its size is structural, on any backend.)
    """
    session = _session(budget)
    joined = A_KEYS.join(KeySet.from_dict({"A": ["a", "b"], "G": ["x", "y"]}))
    session.describe(QueryBuilder("t").groupby(joined))
    assert "(4 groups)" in capsys.readouterr().out


def test_the_ark_path_uses_the_pandas_value_wrappers():
    """The ``*Value`` wrappers the ARK path builds are Core's pandas ones.

    :func:`test_ark_rename_map_select_before_enforce` proves the answers are
    right, which it would also be if the wrappers came from somewhere else and
    happened to work. This checks the binding itself, since a ``*Value`` wrapper
    is the one kind of op whose backend cannot be read off the answer.
    """
    for op in (
        "RenameValue",
        "SelectValue",
        "MapValue",
        "LimitRowsPerGroupValue",
        "LimitKeysPerGroupValue",
        "LimitRowsPerKeyPerGroupValue",
    ):
        wrapper = PANDAS.require(op)
        assert wrapper.__module__ == (
            "tmlt.core.transformations.pandas_transformations.add_remove_keys"
        ), f"{op} is bound to {wrapper.__module__}"


def test_the_count_factories_are_the_pandas_ones():
    """The count measurements come from Core's pandas aggregations."""
    for op in ("create_count_measurement", "create_count_distinct_measurement"):
        factory = PANDAS.require(op)
        assert factory.__module__ == "tmlt.core.measurements.pandas_aggregations"


###############################################################################
# The accountant, and the noise it reports.
###############################################################################


def test_noise_info_is_reachable():
    """A pandas query can say what noise it would add, without answering it."""
    session = _session(PureDPBudget(10))
    # pylint: disable-next=protected-access
    info = session._noise_info(
        QueryBuilder("t").groupby(A_KEYS).count(), PureDPBudget(1)
    )
    assert info == [
        {"noise_mechanism": _NoiseMechanism.GEOMETRIC, "noise_parameter": 1}
    ]
    # Asking what a query would cost costs nothing.
    assert session.remaining_privacy_budget == PureDPBudget(10)


def test_evaluate_with_noise_info_answers_and_describes_in_one_compile():
    """One call gives the same answer and the same noise info as two.

    ``_noise_info`` then ``evaluate`` is what a caller who wants to report the
    noise alongside the answer has to write, and it compiles the query twice.
    This gives both from one compilation -- and, more to the point, from one
    measurement, so the noise reported is necessarily the noise the answer
    carries.
    """
    query = QueryBuilder("t").groupby(A_KEYS).count()
    session = _session(RhoZCDPBudget(INF))
    # pylint: disable=protected-access
    answer, info = session._evaluate_with_noise_info(query, RhoZCDPBudget(INF))

    reference = _session(RhoZCDPBudget(INF))
    assert info == reference._noise_info(query, RhoZCDPBudget(INF))
    _assert_answer(
        answer, reference.evaluate(query, RhoZCDPBudget(INF)), list(A_KEYS.columns())
    )


def test_evaluate_with_noise_info_spends_the_budget_once():
    """Answering and describing in one call costs one answer's worth of budget."""
    session = _session(PureDPBudget(2))
    query = QueryBuilder("t").groupby(A_KEYS).count()
    # pylint: disable=protected-access
    session._evaluate_with_noise_info(query, PureDPBudget(1))
    assert session.remaining_privacy_budget == PureDPBudget(1)


def test_evaluating_twice_spends_the_budget_twice():
    """The accountant behind a pandas Session is the real one.

    Not an arithmetic test: an engine that answered from a fresh accountant each
    time would give exactly the same answers as this one, and would have no
    privacy guarantee at all.
    """
    session = _session(PureDPBudget(2))
    query = QueryBuilder("t").groupby(A_KEYS).count()

    assert session.remaining_privacy_budget == PureDPBudget(2)
    session.evaluate(query, PureDPBudget(1))
    assert session.remaining_privacy_budget == PureDPBudget(1)
    session.evaluate(query, PureDPBudget(1))
    assert session.remaining_privacy_budget == PureDPBudget(0)

    with pytest.raises(RuntimeError, match="budget"):
        session.evaluate(query, PureDPBudget(1))


###############################################################################
# (e) Suppression, through Backend.suppress_below. The pandas implementation
# must not reuse the Spark spelling: pandas' DataFrame.filter selects columns
# by label, so the Spark one-liner silently returns every row and no columns.
###############################################################################


def test_suppress_below_threshold():
    """Suppression drops the groups whose count is under the threshold."""
    session = _session(RhoZCDPBudget(INF))
    answer = session.evaluate(
        QueryBuilder("t").groupby(A_KEYS).count().suppress(2), RhoZCDPBudget(INF)
    )
    _assert_answer(answer, _expected([("a", 3), ("c", 3)], ["A", "count"]), ["A"])


###############################################################################
# (3) Cross-backend parity: a smoke check, ahead of the exhaustive one.
###############################################################################

_PARITY_QUERIES = [
    pytest.param(lambda: QueryBuilder("t").count(), ["count"], id="total-count"),
    pytest.param(
        lambda: QueryBuilder("t").groupby(A_KEYS).count(), ["A"], id="groupby-count"
    ),
    pytest.param(
        lambda: QueryBuilder("t").groupby(A_KEYS).count_distinct(["B"]),
        ["A"],
        id="groupby-count-distinct",
    ),
    pytest.param(
        lambda: (
            QueryBuilder("t")
            .rename({"A": "grp"})
            .map(
                lambda row: {"C": row["B"] * 2},
                new_column_types=Schema({"C": "INTEGER"}),
                augment=True,
            )
            .select(["grp", "C"])
            .groupby(KeySet.from_dict({"grp": ["a", "b", "c", "d"]}))
            .count()
        ),
        ["grp"],
        id="rename-map-select",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("build_query,sort_by", _PARITY_QUERIES)
def test_the_two_backends_agree(spark, build_query, sort_by: List[str]):
    """The same query over the same data answers the same on either backend.

    A smoke check rather than a survey -- the exhaustive cross-backend
    comparison is its own work -- but it is the check that matters most: the
    hand-computed answers above say the pandas backend is self-consistent, and
    this says it agrees with the backend that has been in production for years.
    """
    budget = RhoZCDPBudget(INF)
    query: Query = build_query()

    pandas_answer = _session(budget).evaluate(query, budget)

    spark_session = (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_private_dataframe("t", spark.createDataFrame(ROWS), AddMaxRows(1))
        .build()
    )
    spark_answer = spark_session.evaluate(query, budget).toPandas()

    assert isinstance(pandas_answer, pd.DataFrame)
    assert list(pandas_answer.columns) == list(spark_answer.columns)
    pd.testing.assert_frame_equal(
        _normalized(pandas_answer, sort_by), _normalized(spark_answer, sort_by)
    )


@pytest.mark.slow
def test_the_two_backends_agree_on_the_ark_path(spark):
    """The ID-truncation path agrees across backends too.

    The one query family whose answer depends on Core dropping rows, rather than
    only on counting them. It agrees exactly because every ID's rows are in one
    group, so the two backends' truncations need not drop the *same* rows to
    produce the same counts -- which is all the privacy guarantee asks of them.
    """
    budget = RhoZCDPBudget(INF)
    query = QueryBuilder("t").enforce(MaxRowsPerID(2)).groupby(A_KEYS).count()

    pandas_answer = _id_session(budget).evaluate(query, budget)

    spark_session = (
        Session.Builder()
        .with_privacy_budget(budget)
        .with_id_space("ids")
        .with_private_dataframe(
            "t", spark.createDataFrame(ID_ROWS), AddRowsWithID("id", "ids")
        )
        .build()
    )
    spark_answer = spark_session.evaluate(query, budget).toPandas()

    pd.testing.assert_frame_equal(
        _normalized(pandas_answer, ["A"]), _normalized(spark_answer, ["A"])
    )
