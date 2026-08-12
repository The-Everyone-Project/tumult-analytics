"""The grid: same query, same data, same answer, on either backend.

This is the structural (P1) and exact-value (P2) bar. Every test here runs at an
infinite budget, where both noise mechanisms add nothing, so the answer is the
true one and can be written down.

What the grid crosses
=====================

* **Budget flavor** -- :class:`~tmlt.analytics.RhoZCDPBudget` and
  :class:`~tmlt.analytics.PureDPBudget`, because the budget chooses the noise
  mechanism and so chooses which of Core's pandas measurements is assembled.
* **Aggregation** -- ``count`` and ``count_distinct``, on data where they differ.
* **Grouping** -- no group-by at all, one keyset column, two keyset columns with
  cells that must be zero-filled, and a keyset with a ``None`` key.
* **Protected change** -- ``AddOneRow``, ``AddMaxRows(3)``, and
  ``AddRowsWithID`` under each of the three truncation constraints. The first two
  change the metric the pipeline is built under; the third changes the pipeline,
  since the table becomes a value of an ID dictionary.

Plus three shapes that are not part of the cross product because they are about
composition rather than about a cell of it: a transformation chain
(``rename`` -> ``map`` -> ``select`` -> ``groupby``), a private join under each
truncation strategy and between two ID tables, and suppression.

Why the comparison is against an expected frame
===============================================

Each test body runs once per backend and compares that backend's answer to one
hand-checked frame. Two backends that both equal the same frame agree, so this
proves parity -- and it proves something stronger at the same time, that they
agree on the *right* answer rather than sharing a mistake. It also means only
half of each test needs a JVM, which is what lets the ``test-nojvm`` lane run the
pandas halves.

The expected answers assume nothing about which rows a truncation keeps: see
:data:`~test.system.backend_parity.tables.ID_SPEC` for how the ID table is built
to make that so, and :mod:`~test.system.backend_parity.test_truncation` for the
suite that drops the assumption.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from tmlt.analytics import (
    Constraint,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PrivacyBudget,
    Query,
    QueryBuilder,
    Session,
)
from tmlt.analytics._schema import ColumnType, Schema
from tmlt.analytics.truncation_strategy import TruncationStrategy

from test.backend_testing import (
    BackendFixture,
    analytics_columns,
    assert_frame_equal_across_backends,
    pandas_frame_from_rows,
)
from test.system.backend_parity.tables import (
    ARK,
    ARK_JOIN,
    G_G2_KEYS,
    G_KEYS,
    G_NULL_KEYS,
    JOIN,
    JOIN_G_KEYS,
    PLAIN,
    PLAIN_MAX_ROWS,
    UNBOUNDED_BUDGETS,
    ParitySessions,
    count_column,
    expected_dtypes,
)

################################################################################
# The dimensions
################################################################################

_BUDGETS = [pytest.param(budget, id=name) for name, budget in UNBOUNDED_BUDGETS]

_GROUPINGS: Dict[str, Optional[KeySet]] = {
    "total": None,
    "one-col": G_KEYS,
    "two-col": G_G2_KEYS,
    "null-key": G_NULL_KEYS,
}
"""The four ways the grid groups, by the name its expected answers use."""

_GROUP_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "total": (),
    "one-col": ("g",),
    "two-col": ("g", "g2"),
    "null-key": ("g",),
}
"""The columns each grouping puts in the answer, in the keyset's order."""

_AGGREGATIONS = ("count", "count_distinct")

_CONSTRAINTS: Dict[str, Tuple[Constraint, ...]] = {
    # A bound on rows per ID, and nothing else. Every ID with more than two rows
    # loses some.
    "max-rows-per-id": (MaxRowsPerID(2),),
    # The other admissible shape: a bound on groups per ID plus one on rows per
    # group per ID. Neither is accepted alone -- a query needs *some* bound on an
    # ID's total contribution -- so each of the three constraints is exercised
    # here in a combination the engine accepts.
    "max-groups-then-rows": (MaxGroupsPerID("g", 1), MaxRowsPerGroupPerID("g", 3)),
    # The same pair, enforced in the other order and with the bounds swapped, so
    # that neither the order nor one particular bound is what is being tested.
    "rows-per-group-then-groups": (
        MaxRowsPerGroupPerID("g", 1),
        MaxGroupsPerID("g", 2),
    ),
}

_PLAIN_CHANGES = {"add-one-row": PLAIN, "add-max-rows-3": PLAIN_MAX_ROWS}
"""The two non-ID protected changes, as Session layout keys."""


################################################################################
# The expected answers
#
# Every number below was computed by hand from the table it reads, and then
# checked against both backends. A cell that is zero is a keyset key the data
# does not reach, and it is in the answer because a group-by over a public keyset
# answers for every declared key -- which is the property a backend that only
# reported the groups it found would fail.
################################################################################

_Rows = Tuple[Tuple[Any, ...], ...]

_EXPECTED: Dict[Tuple[str, str, str], _Rows] = {
    # ------------------------------------------------------------------ plain
    # ROWS_SPEC: g = a,a,a,b,NULL,NULL,c and v = 1,1,2,3,4,4,5.
    # The answer does not depend on the protected change, because nothing
    # truncates a table with no ID column: AddMaxRows(3) buys a larger
    # sensitivity, not a smaller table.
    ("plain", "total", "count"): ((7,),),
    ("plain", "total", "count_distinct"): ((5,),),
    ("plain", "one-col", "count"): (("a", 3), ("b", 1), ("c", 1), ("d", 0)),
    # Group a has three rows but two distinct v (1, 1, 2), which is what makes
    # this row different from the one above rather than a duplicate of it.
    ("plain", "one-col", "count_distinct"): (("a", 2), ("b", 1), ("c", 1), ("d", 0)),
    ("plain", "two-col", "count"): (
        ("a", "x", 2),
        ("a", "y", 1),
        ("b", "x", 1),
        # (b, y) is empty although both keys appear elsewhere in the data: the
        # cell a backend that zero-filled only the missing *keys* would drop.
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    ("plain", "two-col", "count_distinct"): (
        ("a", "x", 1),
        ("a", "y", 1),
        ("b", "x", 1),
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    # The null key selects the two rows whose g is NULL, and they hold one
    # distinct v between them.
    ("plain", "null-key", "count"): (("a", 3), ("b", 1), (None, 2), ("d", 0)),
    ("plain", "null-key", "count_distinct"): (
        ("a", 2),
        ("b", 1),
        (None, 1),
        ("d", 0),
    ),
    # -------------------------------------------------------- MaxRowsPerID(2)
    # ID_SPEC's IDs hold 3, 1, 2, 4 and 2 rows, so truncating to two leaves
    # 2 + 1 + 2 + 2 + 2 = 9 rows: i1 and i4 both lose some. Every v is unique,
    # so a count and a count_distinct of v agree everywhere.
    ("max-rows-per-id", "total", "count"): ((9,),),
    ("max-rows-per-id", "total", "count_distinct"): ((9,),),
    ("max-rows-per-id", "one-col", "count"): (
        ("a", 3),  # i1 keeps 2 of 3, i2 keeps its 1
        ("b", 2),  # i3 keeps both
        ("c", 2),  # i4 keeps 2 of 4
        ("d", 0),
    ),
    ("max-rows-per-id", "one-col", "count_distinct"): (
        ("a", 3),
        ("b", 2),
        ("c", 2),
        ("d", 0),
    ),
    ("max-rows-per-id", "two-col", "count"): (
        ("a", "x", 2),
        ("a", "y", 1),
        ("b", "x", 2),
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    ("max-rows-per-id", "two-col", "count_distinct"): (
        ("a", "x", 2),
        ("a", "y", 1),
        ("b", "x", 2),
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    ("max-rows-per-id", "null-key", "count"): (
        ("a", 3),
        ("b", 2),
        (None, 2),  # i5's two rows both survive
        ("d", 0),
    ),
    ("max-rows-per-id", "null-key", "count_distinct"): (
        ("a", 3),
        ("b", 2),
        (None, 2),
        ("d", 0),
    ),
    # ------------------------- MaxGroupsPerID(g, 1) + MaxRowsPerGroupPerID(g, 3)
    # Every ID is in exactly one g, so the group bound drops nothing and the row
    # bound leaves each ID min(rows, 3): 3 + 1 + 2 + 3 + 2 = 11.
    ("max-groups-then-rows", "total", "count"): ((11,),),
    ("max-groups-then-rows", "total", "count_distinct"): ((11,),),
    ("max-groups-then-rows", "one-col", "count"): (
        ("a", 4),  # i1 keeps all 3, i2 keeps its 1
        ("b", 2),
        ("c", 3),  # i4 keeps 3 of 4
        ("d", 0),
    ),
    ("max-groups-then-rows", "one-col", "count_distinct"): (
        ("a", 4),
        ("b", 2),
        ("c", 3),
        ("d", 0),
    ),
    ("max-groups-then-rows", "two-col", "count"): (
        ("a", "x", 3),
        ("a", "y", 1),
        ("b", "x", 2),
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    ("max-groups-then-rows", "two-col", "count_distinct"): (
        ("a", "x", 3),
        ("a", "y", 1),
        ("b", "x", 2),
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    ("max-groups-then-rows", "null-key", "count"): (
        ("a", 4),
        ("b", 2),
        (None, 2),
        ("d", 0),
    ),
    ("max-groups-then-rows", "null-key", "count_distinct"): (
        ("a", 4),
        ("b", 2),
        (None, 2),
        ("d", 0),
    ),
    # ------------------------- MaxRowsPerGroupPerID(g, 1) + MaxGroupsPerID(g, 2)
    # One row per ID per group, and every ID is in one group, so each ID keeps
    # exactly one row: five rows, and a group's count becomes its number of IDs.
    ("rows-per-group-then-groups", "total", "count"): ((5,),),
    ("rows-per-group-then-groups", "total", "count_distinct"): ((5,),),
    ("rows-per-group-then-groups", "one-col", "count"): (
        ("a", 2),  # i1 and i2
        ("b", 1),  # i3
        ("c", 1),  # i4
        ("d", 0),
    ),
    ("rows-per-group-then-groups", "one-col", "count_distinct"): (
        ("a", 2),
        ("b", 1),
        ("c", 1),
        ("d", 0),
    ),
    ("rows-per-group-then-groups", "two-col", "count"): (
        ("a", "x", 1),
        ("a", "y", 1),
        ("b", "x", 1),
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    ("rows-per-group-then-groups", "two-col", "count_distinct"): (
        ("a", "x", 1),
        ("a", "y", 1),
        ("b", "x", 1),
        ("b", "y", 0),
        ("d", "x", 0),
        ("d", "y", 0),
    ),
    ("rows-per-group-then-groups", "null-key", "count"): (
        ("a", 2),
        ("b", 1),
        (None, 1),
        ("d", 0),
    ),
    ("rows-per-group-then-groups", "null-key", "count_distinct"): (
        ("a", 2),
        ("b", 1),
        (None, 1),
        ("d", 0),
    ),
}


################################################################################
# Assertions shared by every test here
################################################################################


def _assert_structure(
    result: Any,
    backend: BackendFixture,
    group_columns: Sequence[str],
    value_column: str,
    group_type: ColumnType = ColumnType.VARCHAR,
) -> None:
    """Asserts a result's columns and their types, per backend (the P1 bar).

    The values are checked by :func:`_assert_answer`; this is about the shape of
    the frame, and about the one place the two backends describe the same answer
    differently.

    **The documented divergence.** Every ``INTEGER`` column of a result -- the
    aggregation's output, and an integer group-by key -- is reported
    ``allow_null=False`` by the pandas backend and ``allow_null=True`` by the
    Spark one. Neither is wrong about the data: a count is never null on either
    backend. It is a difference in what the two frame types can *say*. A Spark
    ``LongType`` field carries a ``nullable`` flag and Core leaves it at its
    permissive default, while a pandas column's nullability is read back off its
    dtype and the answer arrives in a plain ``int64``, which cannot hold a null.
    ``VARCHAR`` columns agree, at ``allow_null=True``, for a related reason from
    the other direction: an ``object`` column can always hold a ``None``.

    Args:
        result: The frame the Session answered with.
        backend: The backend that answered.
        group_columns: The group-by columns the answer should carry.
        value_column: The aggregation's output column.
        group_type: The Analytics type of the group-by columns.
    """
    columns = analytics_columns(result)
    assert set(columns) == {*group_columns, value_column}, (
        f"The {backend.name} backend answered with columns {sorted(columns)}, "
        f"expected {sorted({*group_columns, value_column})}."
    )

    value = columns[value_column]
    assert value.column_type is ColumnType.INTEGER
    assert value.allow_null is (backend.name == "spark"), (
        f"A count column's allow_null is expected to be False on pandas and "
        f"True on Spark; the {backend.name} backend reported {value.allow_null}."
    )

    for name in group_columns:
        descriptor = columns[name]
        assert descriptor.column_type is group_type
        if group_type is ColumnType.INTEGER:
            assert descriptor.allow_null is (backend.name == "spark")
        else:
            assert descriptor.allow_null is True


def _assert_answer(
    result: Any,
    expected_rows: _Rows,
    group_columns: Sequence[str],
    value_column: str,
    group_dtype: Any = object,
) -> None:
    """Asserts a result holds exactly the expected cells (the P2 bar).

    Args:
        result: The frame the Session answered with.
        expected_rows: The rows it should hold, in ``(*group keys, value)`` order.
        group_columns: The group-by columns.
        value_column: The aggregation's output column.
        group_dtype: The pandas dtype the group columns are compared at.
    """
    columns = [*group_columns, value_column]
    dtypes = expected_dtypes(group_columns, value_column, group_dtype)
    expected = pandas_frame_from_rows(columns, list(expected_rows), dtypes)
    assert_frame_equal_across_backends(
        result, expected, sort_by=list(group_columns), dtypes=dtypes
    )


def _aggregate(builder: QueryBuilder, grouping: str, aggregation: str) -> Query:
    """Builds the aggregation the grid asks for, over an optional group-by.

    An ungrouped aggregation is not a special case of the grouped one at the
    Analytics level -- it is a different call, and it returns a one-row frame
    rather than a frame per key -- so both spellings are in the grid.

    Args:
        builder: The query built so far.
        grouping: The grouping's name, a key of :data:`_GROUPINGS`.
        aggregation: ``"count"`` or ``"count_distinct"``.

    Returns:
        The query.
    """
    keys = _GROUPINGS[grouping]
    grouped = builder if keys is None else builder.groupby(keys)
    if aggregation == "count":
        return grouped.count()
    return grouped.count_distinct(["v"])


def _enforced(builder: QueryBuilder, constraint_name: str) -> QueryBuilder:
    """Applies a named constraint combination to a query.

    Args:
        builder: The query built so far.
        constraint_name: A key of :data:`_CONSTRAINTS`.

    Returns:
        The query, with each constraint enforced in order.
    """
    for constraint in _CONSTRAINTS[constraint_name]:
        builder = builder.enforce(constraint)
    return builder


################################################################################
# The cross product
################################################################################


@pytest.mark.parametrize("budget", _BUDGETS)
@pytest.mark.parametrize("change", sorted(_PLAIN_CHANGES))
@pytest.mark.parametrize("grouping", sorted(_GROUPINGS))
@pytest.mark.parametrize("aggregation", _AGGREGATIONS)
def test_counts_on_a_table_with_no_id_column(
    backend: BackendFixture,
    sessions: ParitySessions,
    budget: PrivacyBudget,
    change: str,
    grouping: str,
    aggregation: str,
):
    """Both backends give the true answer for every cell of the plain grid."""
    session = sessions.get(backend, _PLAIN_CHANGES[change], budget)
    result = session.evaluate(
        _aggregate(QueryBuilder("t"), grouping, aggregation), budget
    )

    group_columns = _GROUP_COLUMNS[grouping]
    value_column = count_column(aggregation)
    _assert_structure(result, backend, group_columns, value_column)
    _assert_answer(
        result, _EXPECTED[("plain", grouping, aggregation)], group_columns, value_column
    )


@pytest.mark.parametrize("budget", _BUDGETS)
@pytest.mark.parametrize("constraint", sorted(_CONSTRAINTS))
@pytest.mark.parametrize("grouping", sorted(_GROUPINGS))
@pytest.mark.parametrize("aggregation", _AGGREGATIONS)
def test_counts_on_a_table_protected_by_id(
    backend: BackendFixture,
    sessions: ParitySessions,
    budget: PrivacyBudget,
    constraint: str,
    grouping: str,
    aggregation: str,
):
    """Both backends give the true answer for every cell of the ID grid.

    The ID path is a different pipeline, not a different metric: the table is a
    value of an ID dictionary, so every stage reaches Core's ``*Value`` wrapper
    and the constraint builds a truncation of its own before the aggregation
    sees anything.
    """
    session = sessions.get(backend, ARK, budget)
    query = _aggregate(_enforced(QueryBuilder("t"), constraint), grouping, aggregation)
    result = session.evaluate(query, budget)

    group_columns = _GROUP_COLUMNS[grouping]
    value_column = count_column(aggregation)
    _assert_structure(result, backend, group_columns, value_column)
    _assert_answer(
        result,
        _EXPECTED[(constraint, grouping, aggregation)],
        group_columns,
        value_column,
    )


################################################################################
# Composition: chains, joins, suppression
################################################################################

_CHAIN_KEYS = KeySet.from_dict({"grp": ["a", "b", "c", "d"]})
"""The chain renames ``g`` to ``grp``, so its group-by is keyed on the new name."""


def _double(row: Dict[str, Any]) -> Dict[str, Any]:
    """Doubles a row's ``v``, for the ``map`` in the chains below.

    Args:
        row: The row to map.

    Returns:
        The new column.
    """
    return {"double": row["v"] * 2}


@pytest.mark.parametrize("budget", _BUDGETS)
def test_a_transformation_chain(
    backend: BackendFixture, sessions: ParitySessions, budget: PrivacyBudget
):
    """Rename -> map -> select -> groupby, on a table with no ID column.

    A chain rather than three separate tests because the interesting failure is
    a stage that works alone and composes wrongly: a rename that the map's row
    dict does not see, or a select that drops the column the group-by needs.
    """
    session = sessions.get(backend, PLAIN, budget)
    query = (
        QueryBuilder("t")
        .rename({"g": "grp"})
        .map(_double, new_column_types=Schema({"double": "INTEGER"}), augment=True)
        .select(["grp", "double"])
        .groupby(_CHAIN_KEYS)
        .count()
    )
    result = session.evaluate(query, budget)
    _assert_structure(result, backend, ("grp",), "count")
    # The chain renames and adds a column but drops no row, so the counts are
    # the plain one-column counts under the new name.
    _assert_answer(result, (("a", 3), ("b", 1), ("c", 1), ("d", 0)), ("grp",), "count")


@pytest.mark.parametrize("budget", _BUDGETS)
def test_a_transformation_chain_on_a_table_protected_by_id(
    backend: BackendFixture, sessions: ParitySessions, budget: PrivacyBudget
):
    """The same chain over an ``AddRowsWithID`` table, then a constraint.

    Nothing in the query says the table is protected by ID, and that is the
    point: the compiler has to reach ``RenameValue``, ``MapValue`` and
    ``SelectValue`` rather than the plain transformations, because the table is
    a value of the ID dictionary. The ``count_distinct`` is over the *mapped*
    column, so the map has to have run for the answer to come out right.
    """
    session = sessions.get(backend, ARK, budget)
    query = (
        QueryBuilder("t")
        .rename({"g": "grp"})
        .map(_double, new_column_types=Schema({"double": "INTEGER"}), augment=True)
        .select(["id", "grp", "double"])
        .enforce(MaxRowsPerID(2))
        .groupby(_CHAIN_KEYS)
        .count_distinct(["double"])
    )
    result = session.evaluate(query, budget)
    value_column = count_column("count_distinct", "double")
    _assert_structure(result, backend, ("grp",), value_column)
    # Doubling is injective, so the distinct count of `double` after truncating
    # to two rows per ID is the distinct count of `v`: the MaxRowsPerID(2) row
    # of the grid above, minus the null group, which this keyset omits.
    _assert_answer(
        result, (("a", 3), ("b", 2), ("c", 2), ("d", 0)), ("grp",), value_column
    )


_JOIN_STRATEGIES = [
    pytest.param(TruncationStrategy.DropExcess(2), "drop-excess-2", id="drop-excess"),
    pytest.param(
        TruncationStrategy.DropNonUnique(), "drop-non-unique", id="drop-non-unique"
    ),
]
"""Both truncation strategies a private join can be given.

Both are available on both backends -- the pandas backend binds Core's shared
:class:`~tmlt.core.transformations.spark_transformations.join.TruncationStrategy`
enum, not a copy of it -- so both belong in the grid.
"""

_JOIN_EXPECTED: Dict[str, _Rows] = {
    # The left table has k1 twice; the right has every key once, and a k4 that
    # matches nothing. DropExcess(2) drops nothing at all, so the join is
    # (k1, a), (k1, b), (k2, a), (k3, b) against one right row each.
    "drop-excess-2": (("a", 2), ("b", 2), ("z", 0)),
    # DropNonUnique drops the left's two k1 rows and keeps the right whole, so
    # only k2 and k3 join.
    "drop-non-unique": (("a", 1), ("b", 1), ("z", 0)),
}


@pytest.mark.parametrize("budget", _BUDGETS)
@pytest.mark.parametrize("strategy,expected_key", _JOIN_STRATEGIES)
def test_a_private_join(
    backend: BackendFixture,
    sessions: ParitySessions,
    budget: PrivacyBudget,
    strategy: TruncationStrategy.Type,
    expected_key: str,
):
    """A join between two private tables, under each truncation strategy."""
    session = sessions.get(backend, JOIN, budget)
    query = (
        QueryBuilder("l")
        .join_private(
            "r", truncation_strategy_left=strategy, truncation_strategy_right=strategy
        )
        .groupby(JOIN_G_KEYS)
        .count()
    )
    result = session.evaluate(query, budget)
    _assert_structure(result, backend, ("g",), "count")
    _assert_answer(result, _JOIN_EXPECTED[expected_key], ("g",), "count")


@pytest.mark.parametrize("budget", _BUDGETS)
def test_a_private_join_between_two_id_tables(
    backend: BackendFixture, sessions: ParitySessions, budget: PrivacyBudget
):
    """A join on the ID, which takes no truncation strategy at all.

    Two tables in one ID space join on the ID, and the result is still an ID
    table, so the truncation happens afterwards as a constraint. It is a second
    Core operation -- ``PrivateJoinOnKey`` rather than ``PrivateJoin`` -- and
    therefore a second thing to compare.
    """
    session = sessions.get(backend, ARK_JOIN, budget)
    query = (
        QueryBuilder("l")
        .join_private("r")
        .enforce(MaxRowsPerID(2))
        .groupby(JOIN_G_KEYS)
        .count()
    )
    result = session.evaluate(query, budget)
    _assert_structure(result, backend, ("g",), "count")
    # i1 joins its two rows (a, b) against one right row; i2 joins its one row
    # (a) against two, giving four rows before truncation and, after keeping two
    # per ID, (a, a) from i2 and one of i1's -- i3 and i4 join nothing. Which of
    # i1's two rows survives is not observable here, because the count is over
    # the group column and i1 contributes one row to whichever group it keeps:
    # a + b = 3 either way. test_truncation.py is where that is pinned down.
    _assert_answer(result, (("a", 3), ("b", 1), ("z", 0)), ("g",), "count")


@pytest.mark.parametrize("budget", _BUDGETS)
def test_suppression(
    backend: BackendFixture, sessions: ParitySessions, budget: PrivacyBudget
):
    """``suppress`` drops the low groups, and drops the same ones on both.

    Suppression is post-processing rather than a pipeline stage, and it goes
    through :meth:`~tmlt.analytics._backends.Backend.suppress_below` because the
    Spark spelling of it does the wrong thing on a pandas frame -- pandas'
    ``DataFrame.filter`` selects *columns* by label, so the Spark one-liner
    would return every row and no columns. That failure is silent in a test that
    only checks the surviving counts, so this checks the surviving *columns*
    too, which :func:`_assert_structure` does.
    """
    session = sessions.get(backend, PLAIN, budget)
    query = QueryBuilder("t").groupby(G_KEYS).count().suppress(2)
    result = session.evaluate(query, budget)
    _assert_structure(result, backend, ("g",), "count")
    # Of a=3, b=1, c=1, d=0, only a clears a threshold of 2.
    _assert_answer(result, (("a", 3),), ("g",), "count")


@pytest.mark.parametrize("budget", _BUDGETS)
def test_suppression_on_a_table_protected_by_id(
    backend: BackendFixture, sessions: ParitySessions, budget: PrivacyBudget
):
    """Suppression after a truncation, on the ID path."""
    session = sessions.get(backend, ARK, budget)
    query = (
        QueryBuilder("t").enforce(MaxRowsPerID(2)).groupby(G_KEYS).count().suppress(3)
    )
    result = session.evaluate(query, budget)
    _assert_structure(result, backend, ("g",), "count")
    # Of a=3, b=2, c=2, d=0, only a clears a threshold of 3.
    _assert_answer(result, (("a", 3),), ("g",), "count")


################################################################################
# The grid's own bookkeeping
################################################################################


def test_every_grid_cell_has_an_expected_answer():
    """No cell of the cross product is silently missing its oracle.

    The parametrizations above read :data:`_EXPECTED` by key, so a missing entry
    would fail as a ``KeyError`` in one test rather than as a gap in the grid.
    This says the same thing once, up front, and also catches the opposite
    mistake: an entry left behind by a dimension that was removed.
    """
    cases = ["plain", *sorted(_CONSTRAINTS)]
    wanted = {
        (case, grouping, aggregation)
        for case in cases
        for grouping in _GROUPINGS
        for aggregation in _AGGREGATIONS
    }
    assert set(_EXPECTED) == wanted, (
        f"missing {sorted(wanted - set(_EXPECTED))}, "
        f"unexpected {sorted(set(_EXPECTED) - wanted)}"
    )


def test_the_grid_is_the_size_it_claims():
    """The grid is 2 budgets x 4 groupings x 2 aggregations x 5 protected cases.

    Guarding the count keeps a parametrization from quietly collapsing -- a
    dimension reduced to one value still passes every test in the file.
    """
    assert len(_BUDGETS) == 2
    assert len(_GROUPINGS) == len(_GROUP_COLUMNS) == 4
    assert len(_AGGREGATIONS) == 2
    assert len(_PLAIN_CHANGES) == 2
    assert len(_CONSTRAINTS) == 3
    # One expected answer per (case, grouping, aggregation), where the cases are
    # "plain" -- shared by both non-ID protected changes -- and the three ID
    # constraint combinations.
    assert len(_EXPECTED) == (1 + len(_CONSTRAINTS)) * len(_GROUPINGS) * len(
        _AGGREGATIONS
    )


def test_the_expected_frames_are_all_buildable():
    """Every expected frame can be built at the dtypes it is compared under.

    :func:`~test.backend_testing.frames.pandas_frame_from_rows` refuses values
    that would be silently changed by their dtype -- a null in a plain ``int64``
    column, a non-integral float in an integer one -- so building each frame here
    turns a typo in the table above into a failure of this test rather than of
    whichever grid cell reads it.
    """
    built: List[Any] = []
    for (case, grouping, aggregation), rows in _EXPECTED.items():
        group_columns = _GROUP_COLUMNS[grouping]
        value_column = count_column(aggregation)
        columns = [*group_columns, value_column]
        frame = pandas_frame_from_rows(
            columns,
            list(rows),
            expected_dtypes(group_columns, value_column),
        )
        assert list(frame.columns) == columns, (case, grouping, aggregation)
        assert len(frame) == len(rows)
        built.append(frame)
    assert len(built) == len(_EXPECTED)


def test_sessions_are_built_on_the_backend_under_test(
    backend: BackendFixture, sessions: ParitySessions
):
    """Every Session in the grid really is a Session of the backend under test.

    A Session that had quietly chosen Spark would answer, and answer correctly,
    so every expected value in this file would still pass. This is the check
    that the pandas half of the suite is testing the pandas backend.
    """
    for key in (PLAIN, PLAIN_MAX_ROWS, ARK, JOIN, ARK_JOIN):
        session = sessions.get(backend, key, UNBOUNDED_BUDGETS[0][1])
        assert isinstance(session, Session)
        assert session._backend is backend.descriptor, (
            f"The {key} Session was built on {session._backend.name}, "
            f"not on {backend.descriptor.name}."
        )
