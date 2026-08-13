"""The tables the parity suite runs on, and the Sessions it runs them in.

The standard tables in :mod:`test.backend_testing.data` are the tables the
existing suite already uses, restated. They are the right input for a test that
wants to compare against what the Spark suite has always done, and three of the
things this suite has to compare need something they cannot express:

* **A null group key.** No standard table has a null in a column that a group-by
  would key on, so none of them can show that a ``KeySet`` containing ``None``
  selects the null rows on both backends -- the case where the two backends'
  different null representations (a Spark ``NULL`` and an ``object`` column's
  ``None``) would show up if they were going to.
* **A value column with repeats inside a group.** ``count`` and
  ``count_distinct`` differ only where a value repeats, and a test whose two
  aggregations return the same number cannot tell them apart.
* **Truncation that bites, in a way the answer can see.** Every ID in
  :data:`ID_SPEC` sits in one group and holds only distinct values, which makes
  its answers independent of *which* rows a truncation keeps -- deliberately, so
  that the grid in :mod:`~test.system.backend_parity.test_answers` tests the
  cross product and nothing else. :data:`CHOICE_SPEC` is the opposite: it is
  built so that keeping a different row changes the answer, and it is what
  :mod:`~test.system.backend_parity.test_truncation` uses.

Everything here is a :class:`~test.backend_testing.data.TableSpec`, so it is one
table said once and materialized twice by
:meth:`~test.backend_testing.backends.BackendFixture.materialize`; nothing in
this module builds a frame or imports pyspark.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tmlt.analytics import (
    AddMaxRows,
    AddOneRow,
    AddRowsWithID,
    KeySet,
    PrivacyBudget,
    PureDPBudget,
    RhoZCDPBudget,
    Session,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.protected_change import ProtectedChange

from test.backend_testing import (
    BackendFixture,
    TableSpec,
    integer,
    spark_frame,
    varchar,
)

INF = float("inf")

UNBOUNDED_BUDGETS: Tuple[Tuple[str, PrivacyBudget], ...] = (
    ("rho-zCDP", RhoZCDPBudget(INF)),
    ("pure-DP", PureDPBudget(INF)),
)
"""The two budget flavors every exact-answer test is run under.

The budget picks the noise mechanism -- discrete Gaussian for zCDP, geometric
for pure DP -- and so picks which of Core's pandas measurements the count
factory assembles, which means it picks a different pipeline to compare. At an
infinite budget both mechanisms add nothing, so both must return the true
answer.
"""

ID_SPACE = "ids"
"""The one ID space every ``AddRowsWithID`` table here lives in."""


################################################################################
# The tables
################################################################################

ROWS_SPEC = TableSpec(
    name="parity_rows",
    schema=Schema(
        {
            "g": varchar(allow_null=True),
            "g2": varchar(allow_null=False),
            "v": integer(allow_null=False),
        }
    ),
    rows=(
        ("a", "x", 1),
        ("a", "x", 1),
        ("a", "y", 2),
        ("b", "x", 3),
        (None, "x", 4),
        (None, "y", 4),
        ("c", "y", 5),
    ),
)
"""The table with no ID column, for ``AddOneRow`` and ``AddMaxRows``.

Three properties are deliberate. ``g`` holds nulls, so a ``KeySet`` with a
``None`` key has something to select. ``v`` repeats inside group ``a`` and
inside the null group, so ``count`` and ``count_distinct`` return different
numbers there rather than agreeing by accident. And no row has ``g == "d"``, so
every group-by over :data:`G_KEYS` has a cell to zero-fill.
"""

ID_SPEC = TableSpec(
    name="parity_ids",
    schema=Schema(
        {
            "id": varchar(allow_null=False),
            "g": varchar(allow_null=True),
            "g2": varchar(allow_null=False),
            "v": integer(allow_null=False),
        }
    ),
    rows=(
        ("i1", "a", "x", 1),
        ("i1", "a", "x", 2),
        ("i1", "a", "x", 3),
        ("i2", "a", "y", 4),
        ("i3", "b", "x", 5),
        ("i3", "b", "x", 6),
        ("i4", "c", "y", 7),
        ("i4", "c", "y", 8),
        ("i4", "c", "y", 9),
        ("i4", "c", "y", 10),
        ("i5", None, "x", 11),
        ("i5", None, "x", 12),
    ),
)
"""The ``AddRowsWithID`` table for the grid, with row-choice-free answers.

The IDs have 3, 1, 2, 4 and 2 rows, so every truncation constraint the grid
enforces really drops rows from some of them. But each ID sits in exactly one
``(g, g2)`` cell and every ``v`` in the table is unique, so how many rows an ID
keeps determines the answer completely and *which* rows it keeps cannot. That
is what makes the grid's expected answers writable by hand -- and it is exactly
the assumption :mod:`~test.system.backend_parity.test_truncation` drops.

``i5`` has a null ``g``, so the ID path has a null group key too.
"""

CHOICE_SPEC = TableSpec(
    name="parity_choice",
    schema=Schema(
        {
            "id": varchar(allow_null=False),
            "g": varchar(allow_null=False),
            "v": integer(allow_null=False),
        }
    ),
    rows=(
        # Five rows in one group, and v repeats: keeping the first two rows
        # leaves one distinct value, keeping any other two leaves two.
        ("i1", "a", 1),
        ("i1", "a", 1),
        ("i1", "a", 2),
        ("i1", "a", 3),
        ("i1", "a", 4),
        # Four groups for one ID, so MaxGroupsPerID has to choose two of them.
        ("i2", "a", 11),
        ("i2", "b", 12),
        ("i2", "c", 13),
        ("i2", "d", 14),
        # Three rows in one group, with a repeat, for a second bite.
        ("i3", "b", 7),
        ("i3", "b", 7),
        ("i3", "b", 8),
    ),
)
"""The table where a truncation's *choice* of rows changes the answer.

Every constraint in :mod:`~test.system.backend_parity.test_truncation` bites on
this table -- ``i1`` has more rows than ``MaxRowsPerID`` allows, ``i2`` spans
more groups than ``MaxGroupsPerID`` allows -- and the surviving rows are
observable: grouping by ``(g, v)`` rather than by ``g`` turns "which rows
survived" into a frame of counts, and ``v`` repeats inside ``i1`` and ``i3`` so
a ``count_distinct`` sees the difference too.
"""

JOIN_LEFT_SPEC = TableSpec(
    name="parity_join_left",
    schema=Schema({"k": varchar(allow_null=False), "g": varchar(allow_null=False)}),
    rows=(("k1", "a"), ("k1", "b"), ("k2", "a"), ("k3", "b")),
)
"""The left side of the private join. ``k1`` repeats, so truncation applies."""

JOIN_RIGHT_SPEC = TableSpec(
    name="parity_join_right",
    schema=Schema({"k": varchar(allow_null=False), "w": integer(allow_null=False)}),
    rows=(("k1", 10), ("k2", 20), ("k3", 30), ("k4", 40)),
)
"""The right side of the private join.

Each key appears once, and ``k4`` appears on no left row, so the join drops it.
Keeping the right side key-unique is what makes both truncation strategies'
answers independent of which rows they keep: ``DropNonUnique`` drops the left's
``k1`` rows and nothing else, and ``DropExcess(2)`` drops nothing at all.
"""

ARK_JOIN_LEFT_SPEC = TableSpec(
    name="parity_ark_join_left",
    schema=Schema({"id": varchar(allow_null=False), "g": varchar(allow_null=False)}),
    rows=(("i1", "a"), ("i1", "b"), ("i2", "a"), ("i3", "b")),
)
"""The left side of the join between two ``AddRowsWithID`` tables."""

FEATURES_SPEC = TableSpec(
    name="parity_features",
    schema=Schema(
        {
            "id": varchar(allow_null=False),
            "g": varchar(allow_null=False),
            "v": integer(allow_null=False),
            "f": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
            ),
        }
    ),
    rows=(
        ("i1", "a", 1, 1.0),
        ("i1", "a", 2, float("nan")),
        ("i2", "a", 3, float("inf")),
        ("i2", "b", 4, None),
        ("i3", "b", 5, 5.0),
        ("i4", "c", 6, 6.0),
    ),
)
"""The table the capability matrix asks every query feature of.

It carries an ``id``, a group column, an integer to aggregate and a floating
point column holding all three of the values the drop/replace features exist
for -- a null, a NaN and an infinity -- because a query like
``replace_infinity`` cannot be built at all without a ``DECIMAL`` column to
name.

Its counts are of no interest: the matrix asks whether a feature is *available*,
not what it computes. See :mod:`~test.system.backend_parity.test_answers` for the
suite that cares about values.
"""

FEATURES_PUBLIC_SPEC = TableSpec(
    name="parity_features_public",
    schema=Schema(
        {"g": varchar(allow_null=False), "extra": integer(allow_null=False)}
    ),
    rows=(("a", 1), ("b", 2)),
)
"""A public table for the ``join_public`` case of the capability matrix.

Public tables are Spark frames on both backends -- a pandas Session takes none at
all -- so this is only ever materialized for Spark. The pandas half of the matrix
needs no public table, because the gate refuses ``join_public`` before anything
looks the table up.
"""

ARK_JOIN_RIGHT_SPEC = TableSpec(
    name="parity_ark_join_right",
    schema=Schema({"id": varchar(allow_null=False), "w": integer(allow_null=False)}),
    rows=(("i1", 10), ("i2", 20), ("i2", 30), ("i4", 40)),
)
"""The right side of that join. ``i4`` is on no left row; ``i2`` has two rows.

A join between two tables in the same ID space joins on the ID rather than on a
column, and needs no truncation strategy -- it is
:attr:`~tmlt.analytics._backends.Ops.PrivateJoinOnKey` rather than
:attr:`~tmlt.analytics._backends.Ops.PrivateJoin`, which is a second Core
operation to compare.
"""


################################################################################
# The keysets
################################################################################

G_KEYS = KeySet.from_dict({"g": ["a", "b", "c", "d"]})
"""One grouping column. ``"d"`` is on no row, so it is zero-filled."""

G_G2_KEYS = KeySet.from_dict({"g": ["a", "b", "d"], "g2": ["x", "y"]})
"""Two grouping columns, as a product: six declared cells, three of them empty.

``(b, y)`` is empty because the data has no such row while both keys appear
elsewhere, and both ``(d, x)`` and ``(d, y)`` are empty because ``"d"`` is on no
row at all. A backend that only zero-filled the cells its data happened to
produce would be caught by the first of those.
"""

G_NULL_KEYS = KeySet.from_dict({"g": ["a", "b", None, "d"]})
"""One grouping column, with a null key among the four.

The null key is the case the standard tables cannot express, and the one where
the backends' representations differ underneath: on Spark the key matches a SQL
``NULL``, on pandas a ``None`` in an ``object`` column.
"""

JOIN_G_KEYS = KeySet.from_dict({"g": ["a", "b", "z"]})
"""The grouping for the join results. ``"z"`` is never produced by a join."""

CHOICE_GROUPS: Tuple[str, ...] = ("a", "b", "c", "d")
"""Every ``g`` in :data:`CHOICE_SPEC`, all four of which its ``i2`` touches."""

CHOICE_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 7, 8, 11, 12, 13, 14)
"""Every ``v`` in :data:`CHOICE_SPEC`, deduplicated and in order."""

CHOICE_G_KEYS = KeySet.from_dict({"g": list(CHOICE_GROUPS)})
"""The groups of :data:`CHOICE_SPEC`."""

CHOICE_ROW_KEYS = KeySet.from_dict({"g": list(CHOICE_GROUPS), "v": list(CHOICE_VALUES)})
"""Every ``(g, v)`` pair :data:`CHOICE_SPEC` could hold, as a product.

Grouping by this rather than by ``g`` is what makes "which rows survived the
truncation" an *answer*: each declared cell's count is the number of surviving
rows with that ``(g, v)``, so two backends that kept different rows produce
different frames even when their per-group totals match.
"""


################################################################################
# The Sessions
################################################################################

PLAIN = "plain"
"""Session key: :data:`ROWS_SPEC` as ``t``, protected with ``AddOneRow``."""

PLAIN_MAX_ROWS = "plain-max-rows"
"""Session key: :data:`ROWS_SPEC` as ``t``, protected with ``AddMaxRows(3)``."""

ARK = "ark"
"""Session key: :data:`ID_SPEC` as ``t``, protected with ``AddRowsWithID``."""

CHOICE = "choice"
"""Session key: :data:`CHOICE_SPEC` as ``t``, protected with ``AddRowsWithID``."""

JOIN = "join"
"""Session key: the two join tables as ``l`` and ``r``, with ``AddOneRow``."""

ARK_JOIN = "ark-join"
"""Session key: the two ID join tables as ``l`` and ``r``, in one ID space."""

FEATURES = "features"
"""Session key: :data:`FEATURES_SPEC` as ``t``, protected with ``AddOneRow``."""

FEATURES_ARK = "features-ark"
"""Session key: :data:`FEATURES_SPEC` as ``t``, protected with ``AddRowsWithID``."""


@dataclass(frozen=True)
class _Layout:
    """One named set of tables to put in a Session, and how to protect them."""

    tables: Mapping[str, TableSpec]
    protected_change: ProtectedChange
    id_spaces: Tuple[str, ...] = ()


_LAYOUTS: Dict[str, _Layout] = {
    PLAIN: _Layout({"t": ROWS_SPEC}, AddOneRow()),
    PLAIN_MAX_ROWS: _Layout({"t": ROWS_SPEC}, AddMaxRows(3)),
    ARK: _Layout({"t": ID_SPEC}, AddRowsWithID("id", ID_SPACE), (ID_SPACE,)),
    CHOICE: _Layout({"t": CHOICE_SPEC}, AddRowsWithID("id", ID_SPACE), (ID_SPACE,)),
    JOIN: _Layout({"l": JOIN_LEFT_SPEC, "r": JOIN_RIGHT_SPEC}, AddOneRow()),
    ARK_JOIN: _Layout(
        {"l": ARK_JOIN_LEFT_SPEC, "r": ARK_JOIN_RIGHT_SPEC},
        AddRowsWithID("id", ID_SPACE),
        (ID_SPACE,),
    ),
    FEATURES: _Layout({"t": FEATURES_SPEC}, AddOneRow()),
    FEATURES_ARK: _Layout(
        {"t": FEATURES_SPEC}, AddRowsWithID("id", ID_SPACE), (ID_SPACE,)
    ),
}
"""Every Session layout the suite asks for, by key."""

PROTECTED_CHANGE_KEYS: Tuple[str, ...] = (PLAIN, PLAIN_MAX_ROWS, ARK)
"""The protected-change dimension of the grid, as Session keys.

``AddOneRow`` and ``AddMaxRows(3)`` bound a row's contribution and change the
metric the pipeline is built under; ``AddRowsWithID`` changes the shape of the
pipeline entirely, since the table becomes a value of an ID dictionary and every
transformation over it reaches a ``*Value`` wrapper instead.
"""


def protected_change_for(key: str) -> ProtectedChange:
    """Returns the protected change a Session layout uses.

    Args:
        key: The layout's key, one of the module's ``PLAIN``-style constants.
    """
    return _LAYOUTS[key].protected_change


@dataclass
class ParitySessions:
    """Sessions at an infinite budget, built once and shared.

    A Spark Session is expensive to build and the suite wants dozens of queries
    against each of a handful of them, so they are cached here rather than built
    per test. This is the lifecycle the engine documents -- one Session, one
    budget, every query against it -- and at an infinite budget the cache hides
    nothing: no query exhausts the budget, so no test can be affected by which
    tests ran before it. Anything that cares about the ledger builds its own
    Session; see :mod:`~test.system.backend_parity.test_calibration`.

    Attributes:
        cache: The Sessions built so far, keyed by backend, layout and budget.
    """

    cache: Dict[Tuple[str, str, str], Session] = field(default_factory=dict)

    def get(self, backend: BackendFixture, key: str, budget: PrivacyBudget) -> Session:
        """Returns the Session for a layout on a backend, building it if needed.

        Args:
            backend: The backend under test, as the ``backend`` fixture yields.
            key: The layout's key, one of the module's ``PLAIN``-style constants.
            budget: The Session's total budget. Must be infinite: a finite one
                would be shared between tests, and then a test's answer would
                depend on which tests spent from it first.

        Returns:
            The Session.

        Raises:
            ValueError: If the budget is finite.
        """
        if not budget.is_infinite:
            raise ValueError(
                "ParitySessions caches Sessions across tests, so it only holds "
                f"ones with an infinite budget; {budget} is finite. Build a "
                "finite-budget Session in the test that spends it."
            )
        cache_key = (backend.name, key, repr(budget))
        if cache_key not in self.cache:
            layout = _LAYOUTS[key]
            self.cache[cache_key] = backend.build_session(
                layout.tables,
                budget=budget,
                protected_change=layout.protected_change,
                id_spaces=layout.id_spaces,
            )
        return self.cache[cache_key]


def build_session(
    backend: BackendFixture,
    key: str,
    budget: PrivacyBudget,
    *,
    tables: Optional[Mapping[str, TableSpec]] = None,
    public_tables: Sequence[TableSpec] = (),
) -> Session:
    """Builds a fresh, uncached Session for a layout.

    For the finite-budget tests, where the point is what the Session's own
    accountant does and sharing one would be wrong.

    Args:
        backend: The backend under test.
        key: The layout's key.
        budget: The Session's total budget; may be finite.
        tables: Tables to use instead of the layout's own.
        public_tables: Public tables to add, keyed by their specs' own names.
            Only the Spark backend can be given one -- a pandas Session takes no
            public table at all -- so passing these for pandas is an error rather
            than a silently different Session.

    Returns:
        The Session.

    Raises:
        ValueError: If public tables are asked for on a backend that cannot hold
            them.
    """
    layout = _LAYOUTS[key]
    specs = layout.tables if tables is None else tables
    if not public_tables:
        return backend.build_session(
            specs,
            budget=budget,
            protected_change=layout.protected_change,
            id_spaces=layout.id_spaces,
        )
    if backend.is_pandas:
        raise ValueError(
            "A pandas Session cannot hold a public table, so asking for one "
            "would build a Session that is not the one under test."
        )
    spark = backend.require_spark()
    with backend.feature_flag():
        builder = backend.session_builder(budget, layout.id_spaces)
        for name, spec in specs.items():
            builder = builder.with_private_dataframe(
                name,
                backend.materialize(spec),
                protected_change=layout.protected_change,
            )
        for spec in public_tables:
            builder = builder.with_public_dataframe(spec.name, spark_frame(spec, spark))
        return builder.build()


################################################################################
# Expected frames
################################################################################


def count_column(aggregation: str, column: str = "v") -> str:
    """Returns the name a count aggregation gives its output column.

    Args:
        aggregation: Either ``"count"`` or ``"count_distinct"``.
        column: The column counted, for ``count_distinct``.

    Raises:
        ValueError: If the aggregation is neither.
    """
    if aggregation == "count":
        return "count"
    if aggregation == "count_distinct":
        return f"count_distinct({column})"
    raise ValueError(f"Not a count aggregation: {aggregation!r}.")


def expected_dtypes(
    group_columns: Sequence[str], value_column: str, group_dtype: Any = object
) -> Dict[str, Any]:
    """Returns the dtypes a result should be compared under.

    Both frames in a comparison are brought to these dtypes, which is what makes
    the comparison a comparison of *values*: the two backends report a count
    column's nullability differently (see
    :mod:`~test.system.backend_parity.test_calibration`), and passing an explicit
    dtype keeps that difference from deciding whether the answers match. It is
    asserted separately, where it belongs.

    Args:
        group_columns: The group-by columns, if any.
        value_column: The aggregation's output column.
        group_dtype: The dtype the group columns hold. ``object`` for a string
            column, whether or not it holds a null.

    Returns:
        One dtype per column.
    """
    dtypes: Dict[str, Any] = {name: group_dtype for name in group_columns}
    dtypes[value_column] = "int64"
    return dtypes
