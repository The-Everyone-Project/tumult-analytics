"""The standard test tables, as data rather than as frames.

A :class:`TableSpec` is one table said once: its rows as plain Python values,
and the Analytics :class:`~tmlt.analytics._schema.Schema` those rows are meant
to have. Nothing here imports pyspark or builds a frame --
:mod:`test.backend_testing.materialize` turns a spec into whichever backend's
frame type is wanted, and it is the only module that knows how.

The specs below are the tables the existing suite already runs on, restated:

* ``id1``, ``id2``, ``id3``, ``id4`` and ``rows1`` are the frames the
  ``_session_data`` fixture in ``test/system/conftest.py`` builds, and that
  ``test/system/session/conftest.py`` turns into the ``session`` fixture's
  ``id_a1``/``id_a2``/``id_a3``/``id_a4``/``id_b1``/``id_b2``/``rows_1`` tables.
* ``private_id_data`` is the one deterministic frame of the ``_session_data``
  fixture in ``test/conftest.py``.

The other three frames of that second fixture -- ``private_data``,
``join_private_data`` and ``join_public_data`` -- are **not** restated here.
They are unseeded ``np.random.choice`` draws, so they are different data on
every run and cannot be an oracle for a parity test; a parity suite that wants
random input should draw it with a seeded generator of its own. Public tables
are not restated for a second reason: they are Spark tables on both backends
(a pandas Session refuses to take one at all), so they have no pandas twin to
compare against.

Schemas are stated as the *Spark* truth -- the descriptors
:func:`~tmlt.analytics._schema.spark_schema_to_analytics_columns` yields for the
frame the existing fixture builds. Where the pandas twin cannot reproduce one,
:func:`~test.backend_testing.materialize.expected_columns` says what pandas
reports instead; see that module for the divergence rule.

Two details of the restatement, both deliberate:

* ``id3`` and ``id4`` declare their ``id`` column as ``INTEGER``, which
  materializes on Spark as a ``LongType``, where the fixture writes
  ``IntegerType``. The Session never sees the difference:
  :mod:`~tmlt.analytics._coerce_spark_schema` widens the column to ``bigint``
  on the way in. These specs describe the coerced table.
* ``id1``, ``id2`` and ``private_id_data`` are all-nullable because the fixture
  builds them with ``spark.createDataFrame(pandas_df)``, and Spark infers
  ``nullable=True`` for every column it is not told about. That is not a
  property of the data -- none of those columns contains a null -- but it is
  the schema the tests run against.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema

Row = Tuple[Any, ...]
"""One row of a :class:`TableSpec`, as plain Python values.

``None`` is a SQL ``NULL``. A float ``NaN`` is a NaN, never a null: the two are
different values on both backends, and the whole harness keeps them apart."""


@dataclass(frozen=True)
class TableSpec:
    """One logical table: rows, and the Analytics schema they have.

    A spec is backend-neutral. It is turned into a frame by
    :func:`~test.backend_testing.materialize.frame_for`, which is where the two
    backends' representations are decided.

    Attributes:
        name: The table's name, as the existing fixtures key it.
        schema: The Analytics schema of the table, stated as the Spark truth.
        rows: The rows, in column order, as plain Python values.
    """

    name: str
    schema: Schema
    rows: Tuple[Row, ...]

    @property
    def columns(self) -> Tuple[str, ...]:
        """The table's column names, in order."""
        return tuple(self.schema.column_descs)

    def column_descs(self) -> Dict[str, ColumnDescriptor]:
        """The table's Analytics column descriptors, as a plain dict."""
        return dict(self.schema.column_descs)


def _schema(**columns: ColumnDescriptor) -> Schema:
    """Builds a Schema from keyword arguments, for the specs below.

    Args:
        **columns: One :class:`~tmlt.analytics._schema.ColumnDescriptor` per
            column, in column order.

    Returns:
        The schema.
    """
    return Schema(columns)


def integer(*, allow_null: bool) -> ColumnDescriptor:
    """An ``INTEGER`` column descriptor.

    Args:
        allow_null: Whether the column admits nulls.

    Returns:
        The descriptor.
    """
    return ColumnDescriptor(ColumnType.INTEGER, allow_null=allow_null)


def varchar(*, allow_null: bool) -> ColumnDescriptor:
    """A ``VARCHAR`` column descriptor.

    Args:
        allow_null: Whether the column admits nulls.

    Returns:
        The descriptor.
    """
    return ColumnDescriptor(ColumnType.VARCHAR, allow_null=allow_null)


def decimal(*, allow_null: bool) -> ColumnDescriptor:
    """A ``DECIMAL`` column descriptor.

    ``allow_nan`` and ``allow_inf`` are always ``True``, because that is what
    both backends' schema inference reports for a floating point column: neither
    a Spark schema nor a pandas dtype records whether the column holds NaNs or
    infinities, so both assume it might. A spec that said otherwise would
    describe a table neither backend can build.

    Args:
        allow_null: Whether the column admits nulls.

    Returns:
        The descriptor.
    """
    return ColumnDescriptor(
        ColumnType.DECIMAL, allow_null=allow_null, allow_nan=True, allow_inf=True
    )


ID1 = TableSpec(
    name="id1",
    schema=_schema(
        id=integer(allow_null=True),
        group=varchar(allow_null=True),
        group2=varchar(allow_null=True),
        n=integer(allow_null=True),
        float_n=decimal(allow_null=True),
    ),
    rows=(
        (1, "A", "X", 4, 4.0),
        (1, "A", "Y", 5, 5.0),
        (1, "A", "X", 6, 6.0),
        (2, "A", "Y", 7, 7.0),
        (3, "A", "X", 8, 8.0),
        (3, "B", "Y", 9, 9.0),
    ),
)
"""The ``id1`` frame: an ID table with two grouping columns and a float column."""

ID2 = TableSpec(
    name="id2",
    schema=_schema(
        id=integer(allow_null=True),
        group=varchar(allow_null=True),
        x=integer(allow_null=True),
    ),
    rows=(
        (1, "A", 12),
        (1, "B", 15),
        (1, "A", 18),
        (2, "B", 21),
        (3, "A", 24),
        (3, "B", 27),
    ),
)
"""The ``id2`` frame: a second ID table, for joins against ``id1``."""

ID3 = TableSpec(
    name="id3",
    schema=_schema(
        id=integer(allow_null=True),
        group=varchar(allow_null=False),
        x=integer(allow_null=True),
    ),
    rows=(
        (1, "A", 12),
        (None, "B", 15),
        (1, "A", 18),
        (2, "B", None),
        (3, "A", 24),
        (3, "B", 27),
        (None, "A", 30),
    ),
)
"""The ``id3`` frame: nulls in the ID column and in a measure column."""

ID4 = TableSpec(
    name="id4",
    schema=_schema(
        id=integer(allow_null=False),
        group=varchar(allow_null=False),
        x=integer(allow_null=False),
    ),
    rows=(
        (1, "A", 12),
        (1, "B", 15),
        (1, "A", 18),
        (2, "B", 21),
        (3, "A", 24),
        (3, "B", 27),
    ),
)
"""The ``id4`` frame: the same shape as ``id3``, with nothing nullable."""

ROWS1 = TableSpec(
    name="rows1",
    schema=_schema(
        A=varchar(allow_null=False),
        B=integer(allow_null=False),
        X=integer(allow_null=False),
    ),
    rows=(("0", 0, 0), ("0", 0, 1), ("0", 1, 2), ("1", 0, 3)),
)
"""The ``rows1`` frame: the non-ID table, protected with ``AddMaxRows``."""

PRIVATE_ID_DATA = TableSpec(
    name="private_id_data",
    schema=_schema(
        id=integer(allow_null=True),
        A=integer(allow_null=True),
        B=integer(allow_null=True),
        X=varchar(allow_null=True),
    ),
    rows=(
        (1, 4, 100, "X"),
        (1, 5, 100, "Y"),
        (1, 6, 100, "X"),
        (2, 7, 100, "Y"),
        (3, 8, 100, "X"),
        (3, 9, 100, "Y"),
    ),
)
"""The deterministic ID frame of ``test/conftest.py``'s ``_session_data``."""

STANDARD_TABLES: Mapping[str, TableSpec] = {
    spec.name: spec for spec in (ID1, ID2, ID3, ID4, ROWS1, PRIVATE_ID_DATA)
}
"""Every standard table, keyed by the name the existing fixtures use."""

ID_TABLES: Sequence[TableSpec] = (ID1, ID2, ID3, ID4, PRIVATE_ID_DATA)
"""The standard tables that have an ``id`` column, for ``AddRowsWithID``."""
