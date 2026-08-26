"""Which rows a truncation keeps -- the same ones, on both backends.

This is the module the acceptance rests on, and the gap A4b's smoke tests flagged
in their own docstring: their ID data was built so that every ID's rows sat in
one group, which makes a per-group count depend only on *how many* rows a
truncation keeps and never on *which*. That is all the privacy guarantee asks --
any ``k`` of an ID's rows is as good as any other ``k`` -- so a suite that stops
there has proved something true and weaker than it sounds. Two backends can keep
different rows, agree on every count, and disagree the moment a query looks at
the values in the rows that survived.

Core's parity suite settles the underlying question: its truncation utilities are
bit-identical across the two backends. What this module settles is that the
property survives the whole Analytics compile -- the constraint machinery, the
``*Value`` wrappers of the ID path, the rewriter, the group-by -- and reaches the
answer a user sees.

How "which rows" becomes an answer
==================================

By grouping on the row rather than on the group column. Grouping
:data:`~test.system.backend_parity.tables.CHOICE_SPEC` by
:data:`~test.system.backend_parity.tables.CHOICE_ROW_KEYS` -- the product of
every ``g`` and every ``v`` in the table -- makes each declared cell's count the
number of surviving rows with that exact ``(g, v)``. Two backends that kept
different rows then produce different frames even where their per-group totals
match. A ``count_distinct(v)`` per group is the second view of the same thing,
and the one that shows the difference is not an artifact of the keyset: ``v``
repeats inside ``i1`` and ``i3``, so the number of distinct values among an ID's
survivors depends on which of its rows survived.

Why the expected survivors are pinned, and what that does and does not claim
===========================================================================

:data:`_SURVIVORS` says which rows Core actually keeps. That is a stronger
statement than the privacy guarantee needs, and it is written down deliberately,
because a *change* in the row choice is something this suite should notice: it
would be a legitimate change to Core, and it would have to happen on both
backends at once. If it ever happens, these tables change and the two backends
keep agreeing -- which is the property under test.

To keep that from being the only thing asserted,
:func:`test_the_engine_keeps_an_admissible_set_of_rows` checks the pinned
survivors against *every* choice the constraint permits, computed here from the
data rather than from the engine, and asserts that more than one such choice
exists. A test that pinned an answer no other choice could have produced would
pass for the wrong reason.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import itertools
from collections import Counter
from typing import Any, Dict, FrozenSet, Iterator, List, Sequence, Tuple

import pytest

from tmlt.analytics import (
    Constraint,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PrivacyBudget,
    QueryBuilder,
    RhoZCDPBudget,
)

from test.backend_testing import (
    BACKEND_NAMES,
    BackendFixture,
    assert_frame_equal_across_backends,
    pandas_frame_from_rows,
)
from test.system.backend_parity.tables import (
    CHOICE,
    CHOICE_G_KEYS,
    CHOICE_GROUPS,
    CHOICE_ROW_KEYS,
    CHOICE_SPEC,
    CHOICE_VALUES,
    UNBOUNDED_BUDGETS,
    ParitySessions,
)

_BUDGETS = [pytest.param(budget, id=name) for name, budget in UNBOUNDED_BUDGETS]

_ALL_CELLS: Tuple[Tuple[str, int], ...] = tuple(
    itertools.product(CHOICE_GROUPS, CHOICE_VALUES)
)
"""Every cell :data:`CHOICE_ROW_KEYS` declares, in the order its answer has them.

Built from the same two lists the keyset is, so the expected frame below always
declares exactly the cells the query asks for; a keyset that grew a key without
this growing with it would fail on the frame lengths rather than pass by
comparing a subset.
"""

_CONSTRAINTS: Dict[str, Tuple[Constraint, ...]] = {
    # i1 has five rows and keeps two; i2 has four and keeps two; i3 has three and
    # keeps two. Three separate choices, in one query.
    "max-rows-per-id-2": (MaxRowsPerID(2),),
    # A looser bound, so that the choice is a different one rather than a subset
    # of the same one.
    "max-rows-per-id-3": (MaxRowsPerID(3),),
    # i2 is in four groups and may keep two, so a *group* is chosen here as well
    # as a row. Its per-group rows are one each, so the row bound only bites on
    # i1 and i3.
    "max-groups-2-rows-3": (MaxGroupsPerID("g", 2), MaxRowsPerGroupPerID("g", 3)),
    # The group bound is loose enough to drop nothing, and the row bound leaves
    # each (ID, group) exactly one row: the choice is entirely a row choice.
    "max-groups-4-rows-1": (MaxGroupsPerID("g", 4), MaxRowsPerGroupPerID("g", 1)),
    # The tightest group bound: i2 keeps one of its four groups.
    "max-groups-1-rows-2": (MaxGroupsPerID("g", 1), MaxRowsPerGroupPerID("g", 2)),
}

_Survivor = Tuple[Any, int, int]
"""One surviving cell: ``(g, v, count)``, with the count above zero."""

_SURVIVORS: Dict[str, Tuple[_Survivor, ...]] = {
    # i1 (v = 1, 1, 2, 3, 4) keeps v=1 and v=4 -- two distinct values. Keeping
    # its two v=1 rows instead would leave one, which is what makes the
    # count_distinct below a real check and not a restatement of the count.
    # i3 (v = 7, 7, 8) keeps 7 and 8; i2 keeps b and d.
    "max-rows-per-id-2": (
        ("a", 1, 1),
        ("a", 4, 1),
        ("b", 7, 1),
        ("b", 8, 1),
        ("b", 12, 1),
        ("d", 14, 1),
    ),
    # i1 keeps v = 1, 3, 4; i3 keeps all three of its rows, so the (b, 7) cell
    # holds two -- a count above one, which a set-based comparison would lose.
    "max-rows-per-id-3": (
        ("a", 1, 1),
        ("a", 3, 1),
        ("a", 4, 1),
        ("a", 11, 1),
        ("b", 7, 2),
        ("b", 8, 1),
        ("b", 12, 1),
        ("d", 14, 1),
    ),
    # i2 keeps groups a and b out of a, b, c, d -- so (c, 13) here comes from
    # nowhere else, and (d, 14) is gone.
    "max-groups-2-rows-3": (
        ("a", 1, 1),
        ("a", 3, 1),
        ("a", 4, 1),
        ("b", 7, 2),
        ("b", 8, 1),
        ("b", 12, 1),
        ("c", 13, 1),
    ),
    "max-groups-4-rows-1": (
        ("a", 1, 1),
        ("a", 11, 1),
        ("b", 7, 1),
        ("b", 12, 1),
        ("c", 13, 1),
        ("d", 14, 1),
    ),
    # i2 keeps exactly one of its four groups, and it is b.
    "max-groups-1-rows-2": (
        ("a", 1, 1),
        ("a", 4, 1),
        ("b", 7, 1),
        ("b", 8, 1),
        ("b", 12, 1),
    ),
}
"""The rows that survive each constraint, as the row-keyed group-by sees them.

Cells not listed are zero. Checked against both backends, and against the set of
choices the constraint permits; see the module docstring.
"""

_DISTINCT_PER_GROUP: Dict[str, Tuple[Tuple[Any, int], ...]] = {
    "max-rows-per-id-2": (("a", 2), ("b", 3), ("c", 0), ("d", 1)),
    "max-rows-per-id-3": (("a", 4), ("b", 3), ("c", 0), ("d", 1)),
    "max-groups-2-rows-3": (("a", 3), ("b", 3), ("c", 1), ("d", 0)),
    "max-groups-4-rows-1": (("a", 2), ("b", 2), ("c", 1), ("d", 1)),
    "max-groups-1-rows-2": (("a", 2), ("b", 3), ("c", 0), ("d", 0)),
}
"""``count_distinct(v)`` per group after each truncation.

The same fact as :data:`_SURVIVORS` read through an aggregation instead of
through the keyset, which is what a user would actually write. It is here because
the row-keyed group-by is an unusual query: a backend could get the survivor
frame right and still lose the row identity somewhere between the truncation and
an ordinary aggregation.
"""


################################################################################
# What the constraints permit
################################################################################

_RowIndex = int
_Choice = FrozenSet[_RowIndex]
"""One admissible set of surviving rows, as indices into the spec's rows."""


def _rows_by_id() -> Dict[Any, List[_RowIndex]]:
    """Returns the indices of :data:`CHOICE_SPEC`'s rows, grouped by ID."""
    columns = CHOICE_SPEC.columns
    id_position = columns.index("id")
    by_id: Dict[Any, List[_RowIndex]] = {}
    for index, row in enumerate(CHOICE_SPEC.rows):
        by_id.setdefault(row[id_position], []).append(index)
    return by_id


def _group_of(index: _RowIndex) -> Any:
    """Returns the ``g`` value of one of :data:`CHOICE_SPEC`'s rows.

    Args:
        index: The row's position in the spec.
    """
    return CHOICE_SPEC.rows[index][CHOICE_SPEC.columns.index("g")]


def _value_of(index: _RowIndex) -> int:
    """Returns the ``v`` value of one of :data:`CHOICE_SPEC`'s rows.

    Args:
        index: The row's position in the spec.
    """
    return CHOICE_SPEC.rows[index][CHOICE_SPEC.columns.index("v")]


def _keep_at_most(
    indices: Sequence[_RowIndex], most: int
) -> Iterator[Tuple[_RowIndex, ...]]:
    """Every way of keeping at most ``most`` of the given rows.

    Args:
        indices: The rows to choose from.
        most: The bound. Keeping fewer than the bound is never admissible -- a
            truncation drops only what it must -- so this yields exactly the
            subsets of size ``min(most, len(indices))``.

    Yields:
        One tuple of kept indices per admissible choice.
    """
    yield from itertools.combinations(indices, min(most, len(indices)))


def _admissible_choices(constraint_name: str) -> Iterator[_Choice]:
    """Every set of rows a constraint combination could legitimately leave.

    Computed from :data:`CHOICE_SPEC` and the constraint's bounds alone --
    nothing here asks the engine anything -- so it is an independent statement of
    what the constraint means, and the engine's answer has to be one of these.

    Args:
        constraint_name: A key of :data:`_CONSTRAINTS`.

    Yields:
        One admissible set of surviving row indices at a time.

    Raises:
        ValueError: If the constraint combination is not one this enumerator
            understands.
    """
    constraints = _CONSTRAINTS[constraint_name]
    by_id = _rows_by_id()

    if len(constraints) == 1 and isinstance(constraints[0], MaxRowsPerID):
        per_id = [
            list(_keep_at_most(indices, constraints[0].max))
            for indices in by_id.values()
        ]
        for combination in itertools.product(*per_id):
            yield frozenset(itertools.chain.from_iterable(combination))
        return

    groups_per_id = next(
        (c for c in constraints if isinstance(c, MaxGroupsPerID)), None
    )
    rows_per_group = next(
        (c for c in constraints if isinstance(c, MaxRowsPerGroupPerID)), None
    )
    if groups_per_id is None or rows_per_group is None:
        raise ValueError(
            f"Cannot enumerate the choices {constraint_name!r} permits: it is "
            "neither a MaxRowsPerID nor a MaxGroupsPerID/MaxRowsPerGroupPerID "
            "pair."
        )

    per_id = []
    for indices in by_id.values():
        by_group: Dict[Any, List[_RowIndex]] = {}
        for index in indices:
            by_group.setdefault(_group_of(index), []).append(index)
        choices: List[Tuple[_RowIndex, ...]] = []
        for kept_groups in _keep_at_most(sorted(by_group), groups_per_id.max):
            within = [
                list(_keep_at_most(by_group[group], rows_per_group.max))
                for group in kept_groups
            ]
            for combination in itertools.product(*within):
                choices.append(tuple(itertools.chain.from_iterable(combination)))
        per_id.append(choices)
    for combination in itertools.product(*per_id):
        yield frozenset(itertools.chain.from_iterable(combination))


def _cells(choice: _Choice) -> Tuple[Tuple[Any, int, int], ...]:
    """Renders a set of surviving rows the way :data:`_SURVIVORS` states it.

    Args:
        choice: The surviving row indices.

    Returns:
        The non-empty ``(g, v, count)`` cells, sorted.
    """
    counted = Counter((_group_of(index), _value_of(index)) for index in choice)
    return tuple(sorted((g, v, count) for (g, v), count in counted.items()))


################################################################################
# The tests
################################################################################


@pytest.mark.parametrize("budget", _BUDGETS)
@pytest.mark.parametrize("constraint", sorted(_CONSTRAINTS))
def test_the_truncation_keeps_exactly_these_rows(
    backend: BackendFixture,
    sessions: ParitySessions,
    budget: PrivacyBudget,
    constraint: str,
):
    """Both backends' truncations leave the same rows, cell for cell.

    The keyset declares every ``(g, v)`` the table could hold, so the answer is a
    complete census of the truncated table: a surviving row shows up as a cell of
    one, a row dropped shows up as a zero, and a cell of two says the truncation
    kept a duplicated value twice.
    """
    session = sessions.get(backend, CHOICE, budget)
    builder = QueryBuilder("t")
    for each in _CONSTRAINTS[constraint]:
        builder = builder.enforce(each)
    result = session.evaluate(builder.groupby(CHOICE_ROW_KEYS).count(), budget)

    surviving = {(g, v): count for g, v, count in _SURVIVORS[constraint]}
    expected = pandas_frame_from_rows(
        ["g", "v", "count"],
        [(g, v, surviving.get((g, v), 0)) for g, v in _ALL_CELLS],
        {"g": object, "v": "int64", "count": "int64"},
    )
    assert_frame_equal_across_backends(
        result,
        expected,
        sort_by=["g", "v"],
        dtypes={"g": object, "v": "int64", "count": "int64"},
    )


@pytest.mark.parametrize("budget", _BUDGETS)
@pytest.mark.parametrize("constraint", sorted(_CONSTRAINTS))
def test_a_count_distinct_sees_the_same_row_choice(
    backend: BackendFixture,
    sessions: ParitySessions,
    budget: PrivacyBudget,
    constraint: str,
):
    """An ordinary aggregation over the truncated table agrees too.

    The row-keyed group-by above is an unusual query. This is the query a user
    would write, over a column whose values repeat inside an ID, so it is the
    same fact reached by the ordinary path: the number of distinct ``v`` in a
    group depends on which of an ID's rows the truncation kept.
    """
    session = sessions.get(backend, CHOICE, budget)
    builder = QueryBuilder("t")
    for each in _CONSTRAINTS[constraint]:
        builder = builder.enforce(each)
    result = session.evaluate(
        builder.groupby(CHOICE_G_KEYS).count_distinct(["v"]), budget
    )
    expected = pandas_frame_from_rows(
        ["g", "count_distinct(v)"],
        list(_DISTINCT_PER_GROUP[constraint]),
        {"g": object, "count_distinct(v)": "int64"},
    )
    assert_frame_equal_across_backends(
        result,
        expected,
        sort_by=["g"],
        dtypes={"g": object, "count_distinct(v)": "int64"},
    )


@pytest.mark.parametrize("constraint", sorted(_CONSTRAINTS))
def test_the_engine_keeps_an_admissible_set_of_rows(constraint: str):
    """The pinned survivors are one of the choices the constraint permits.

    And there is more than one, which is the part that gives the two tests above
    their content: if the constraint's bounds forced a single answer, they would
    agree whatever the two backends did.

    Nothing here touches a Session. The admissible sets are enumerated from
    :data:`~test.system.backend_parity.tables.CHOICE_SPEC` and the constraint's
    own numbers, so this is an independent reading of what the constraint means.
    """
    choices = list(_admissible_choices(constraint))
    answers = {_cells(choice) for choice in choices}
    assert len(answers) > 1, (
        f"{constraint} permits only one surviving set on this data, so the "
        "parity assertion has nothing to catch. Make the data richer."
    )
    assert _SURVIVORS[constraint] in answers, (
        f"The rows {constraint} was recorded as keeping are not a set it is "
        f"allowed to keep. Recorded: {_SURVIVORS[constraint]}."
    )


@pytest.mark.parametrize("constraint", sorted(_CONSTRAINTS))
def test_the_truncation_actually_drops_rows(constraint: str):
    """Each constraint really bites on this data.

    A constraint whose bounds were loose enough to drop nothing would make its
    row of :data:`_SURVIVORS` the whole table, and the parity claim would be
    about a truncation that never happened.
    """
    kept = sum(count for _, _, count in _SURVIVORS[constraint])
    assert kept < len(CHOICE_SPEC.rows), (
        f"{constraint} keeps all {kept} rows of the table, so it truncates nothing."
    )


@pytest.mark.parametrize("constraint", sorted(_CONSTRAINTS))
def test_the_distinct_counts_follow_from_the_surviving_rows(constraint: str):
    """:data:`_DISTINCT_PER_GROUP` is what :data:`_SURVIVORS` implies.

    The two tables are two views of one fact and are written out separately, so
    this is the arithmetic that ties them together -- without it, a typo in one
    would be invisible until a backend happened to disagree with it.
    """
    distinct: Dict[Any, int] = {
        group: 0 for group, _ in _DISTINCT_PER_GROUP[constraint]
    }
    for group, _value, _count in _SURVIVORS[constraint]:
        distinct[group] += 1
    assert tuple(sorted(distinct.items(), key=lambda pair: str(pair[0]))) == tuple(
        sorted(_DISTINCT_PER_GROUP[constraint], key=lambda pair: str(pair[0]))
    )


@pytest.mark.parametrize("constraint", sorted(_CONSTRAINTS))
def test_the_two_backends_choose_the_same_rows(spark, constraint: str):
    """The same claim, with nothing written down for the backends to agree with.

    :func:`test_the_truncation_keeps_exactly_these_rows` compares each backend to
    :data:`_SURVIVORS`, which proves they agree *and* pins what they agree on. If
    Core ever changes its row choice, that test fails on both backends until the
    table is updated -- and this one keeps testing the property that actually
    matters in the meantime.

    Args:
        spark: The Spark session, requested directly because this test needs both
            backends at once rather than one per run.
        constraint: The constraint combination under test.
    """
    budget = RhoZCDPBudget(float("inf"))
    builder = QueryBuilder("t")
    for each in _CONSTRAINTS[constraint]:
        builder = builder.enforce(each)
    query = builder.groupby(CHOICE_ROW_KEYS).count()

    answers = {}
    cache = ParitySessions()
    for name in BACKEND_NAMES:
        fixture = BackendFixture(name=name, spark=spark if name == "spark" else None)
        with fixture.feature_flag():
            answers[name] = cache.get(fixture, CHOICE, budget).evaluate(query, budget)

    assert_frame_equal_across_backends(
        answers["pandas"],
        answers["spark"],
        sort_by=["g", "v"],
        dtypes={"g": object, "v": "int64", "count": "int64"},
    )
