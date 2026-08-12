"""Backend parity on the neighboring-relation path.

A neighboring relation is where a Session says what it protects, and
:class:`~tmlt.analytics._neighboring_relation_visitor.NeighboringRelationCoreVisitor`
is where that statement becomes the three things the accountant is launched
with: an input domain, an input metric, and ``d_in``. It is the only place those
are minted, so if the pandas backend produced a domain that described the data
even slightly differently -- a column that could hold a null where the Spark
domain said it could not, an ID column harmonized in one backend and not the
other -- every guarantee downstream would be about a different table than the
one the Session holds.

These tests therefore build each protected change twice, once per backend, and
require the two results to correspond exactly. The metric and ``d_in`` are
engine-neutral objects and are compared with ``==``. The domains are not
comparable directly, so they are bridged with
:meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.to_spark_descriptor`,
which renders a pandas column descriptor as the Spark descriptor for the same
values, and compared field by field.

Building the two inputs
-----------------------

Each table is written out once as rows, and each backend's version is built from
those rows independently: the Spark frame from an explicit
:class:`~pyspark.sql.types.StructType`, the pandas frame by ``astype`` onto
explicit dtypes. Nothing converts one frame into the other, so the two
descriptions have to agree on their own.

The Spark schemas are explicit because ``createDataFrame`` marks every field it
*infers* as nullable, whatever it was given -- which is the one documented
disagreement between the backends' inferred schemas (see
:mod:`~tmlt.analytics._coerce_pandas_schema`). Writing the schema by hand states
the nullability the pandas dtype genuinely has, and the correspondence is then
exact:

.. list-table::
   :header-rows: 1

   * - pandas dtype
     - Spark field
   * - ``int64``
     - ``LongType``, not nullable
   * - ``Int64``
     - ``LongType``, nullable
   * - ``object`` of :class:`str`
     - ``StringType``, nullable
   * - ``object`` of :class:`datetime.date`
     - ``DateType``, nullable
   * - ``float64``
     - ``DoubleType``, not nullable
   * - ``datetime64[ns]``
     - ``TimestampType``, nullable
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from typing import Any, ClassVar, Dict, List, NamedTuple, Tuple

import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.pandas_domains import PandasTableDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    AddRemoveKeys as CoreAddRemoveKeys,
    DictMetric,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics._backends import PANDAS, SPARK
from tmlt.analytics._neighboring_relation import (
    AddRemoveKeys,
    AddRemoveRows,
    AddRemoveRowsAcrossGroups,
    Conjunction,
)
from tmlt.analytics._neighboring_relation_visitor import NeighboringRelationCoreVisitor
from tmlt.analytics._table_identifier import NamedTable, TableCollection

DATE = datetime.date(2024, 1, 15)
OTHER_DATE = datetime.date(2024, 2, 20)
TIMESTAMP = datetime.datetime(2024, 1, 15, 9, 30)
OTHER_TIMESTAMP = datetime.datetime(2024, 2, 20, 17, 45)


class Table(NamedTuple):
    """One logical table, described so that each backend can build it itself."""

    rows: List[Tuple[Any, ...]]
    """The table's contents, shared by both backends."""

    spark_schema: StructType
    """The Spark schema, written out rather than inferred."""

    pandas_dtypes: Dict[str, Any]
    """The dtype of every column, written out rather than inferred."""

    def spark(self, spark: Any) -> DataFrame:
        """Returns this table as a Spark DataFrame."""
        return spark.createDataFrame(self.rows, schema=self.spark_schema)

    def pandas(self) -> pd.DataFrame:
        """Returns this table as a pandas DataFrame."""
        frame = pd.DataFrame(
            self.rows, columns=[field.name for field in self.spark_schema]
        )
        return frame.astype(self.pandas_dtypes)


EPISODES = Table(
    rows=[
        ("resident-1", 10, DATE, 1250.5),
        ("resident-1", 11, OTHER_DATE, 90.25),
        ("resident-2", 12, DATE, 400.0),
    ],
    spark_schema=StructType(
        [
            StructField("resident_id", StringType(), nullable=True),
            StructField("episode_id", LongType(), nullable=False),
            StructField("start_date", DateType(), nullable=True),
            StructField("cost", DoubleType(), nullable=False),
        ]
    ),
    pandas_dtypes={
        "resident_id": object,
        "episode_id": "int64",
        "start_date": object,
        "cost": "float64",
    },
)
"""A table with an ID column, exercising four of the five column types."""

ANSWERS = Table(
    rows=[
        ("resident-1", "q1", 3),
        ("resident-2", "q1", 5),
        ("resident-2", "q2", 1),
    ],
    spark_schema=StructType(
        [
            StructField("resident_id", StringType(), nullable=True),
            StructField("question", StringType(), nullable=True),
            StructField("score", LongType(), nullable=False),
        ]
    ),
    pandas_dtypes={"resident_id": object, "question": object, "score": "int64"},
)
"""A second table in the same ID space as :data:`EPISODES`."""

PROJECTS = Table(
    rows=[("project-1", "north"), ("project-2", "south")],
    spark_schema=StructType(
        [
            StructField("project_id", StringType(), nullable=True),
            StructField("region", StringType(), nullable=True),
        ]
    ),
    pandas_dtypes={"project_id": object, "region": object},
)
"""The only table in a second ID space -- the project rollup."""

METRICS = Table(
    rows=[("visits", 7, TIMESTAMP), ("referrals", 2, OTHER_TIMESTAMP)],
    spark_schema=StructType(
        [
            StructField("metric_name", StringType(), nullable=True),
            StructField("value", LongType(), nullable=False),
            StructField("recorded_at", TimestampType(), nullable=True),
        ]
    ),
    pandas_dtypes={
        "metric_name": object,
        "value": "int64",
        "recorded_at": "datetime64[ns]",
    },
)
"""A table protected by row count rather than by ID, with a TIMESTAMP column."""

FLOAT_KEYED = Table(
    rows=[(1.5, "a"), (2.5, "b")],
    spark_schema=StructType(
        [
            StructField("key", DoubleType(), nullable=False),
            StructField("label", StringType(), nullable=True),
        ]
    ),
    pandas_dtypes={"key": "float64", "label": object},
)
"""A table whose only candidate key column is a DECIMAL, which is not allowed."""

TIMESTAMP_KEYED = Table(
    rows=[(TIMESTAMP, "a"), (OTHER_TIMESTAMP, "b")],
    spark_schema=StructType(
        [
            StructField("key", TimestampType(), nullable=True),
            StructField("label", StringType(), nullable=True),
        ]
    ),
    pandas_dtypes={"key": "datetime64[ns]", "label": object},
)
"""A table whose only candidate key column is a TIMESTAMP, which is not allowed."""

NON_NULLABLE_ID = Table(
    rows=[(1, "a", 4), (2, "b", 7)],
    spark_schema=StructType(
        [
            StructField("id", LongType(), nullable=False),
            StructField("label", StringType(), nullable=True),
            StructField("count", LongType(), nullable=False),
        ]
    ),
    pandas_dtypes={"id": "int64", "label": object, "count": "int64"},
)
"""An INTEGER ID column that cannot hold a null: ``int64``, ``nullable=False``.

``count`` is a second column that cannot hold one either, and is not the ID
column; harmonization must leave it alone.
"""

NULLABLE_ID = Table(
    rows=[(1, "x", 5), (None, "y", 6)],
    spark_schema=StructType(
        [
            StructField("id", LongType(), nullable=True),
            StructField("label", StringType(), nullable=True),
            StructField("count", LongType(), nullable=False),
        ]
    ),
    pandas_dtypes={"id": "Int64", "label": object, "count": "int64"},
)
"""The same table with a nullable ID column: ``Int64``, ``nullable=True``."""


def bridge(domain: Domain) -> Any:
    """Renders a domain in Spark terms, so that two backends' domains compare.

    A :class:`~tmlt.core.domains.collections.DictDomain` becomes a dictionary of
    bridged domains, keyed by the same identifiers; a table domain becomes its
    columns descriptor, with a pandas one converted column by column. Anything
    else fails, so that a domain neither backend was expected to build cannot
    slip through as an incidental match.
    """
    if isinstance(domain, DictDomain):
        return {key: bridge(value) for key, value in domain.key_to_domain.items()}
    if isinstance(domain, SparkDataFrameDomain):
        return dict(domain.schema)
    if isinstance(domain, PandasTableDomain):
        return {
            column: descriptor.to_spark_descriptor()
            for column, descriptor in domain.schema.items()
        }
    raise AssertionError(f"Unexpected domain type {type(domain).__name__}")


def assert_outputs_correspond(
    spark_output: NeighboringRelationCoreVisitor.Output,
    pandas_output: NeighboringRelationCoreVisitor.Output,
) -> None:
    """Asserts that the two backends built the same accountant inputs."""
    assert bridge(pandas_output.domain) == bridge(spark_output.domain)
    assert pandas_output.metric == spark_output.metric
    assert pandas_output.distance == spark_output.distance


def spark_tables(spark: Any, tables: Dict[str, Table]) -> Dict[str, Any]:
    """Returns the named tables as Spark DataFrames."""
    return {name: table.spark(spark) for name, table in tables.items()}


def pandas_tables(tables: Dict[str, Table]) -> Dict[str, Any]:
    """Returns the named tables as pandas DataFrames."""
    return {name: table.pandas() for name, table in tables.items()}


def visitors(
    spark: Any, tables: Dict[str, Table], output_measure: Any = PureDP()
) -> Tuple[NeighboringRelationCoreVisitor, NeighboringRelationCoreVisitor]:
    """Returns a visitor per backend over the same logical tables."""
    return (
        NeighboringRelationCoreVisitor(
            spark_tables(spark, tables), output_measure, SPARK
        ),
        NeighboringRelationCoreVisitor(pandas_tables(tables), output_measure, PANDAS),
    )


class TestBackendsAreDistinct:
    """Checks that the parity assertions are comparing two different things."""

    def test_each_backend_builds_its_own_domain_type(self, spark: Any) -> None:
        """The bridge would be vacuous if both sides built Spark domains."""
        spark_visitor, pandas_visitor = visitors(spark, {"episodes": EPISODES})
        relation = AddRemoveRows("episodes", n=1)

        assert isinstance(relation.accept(spark_visitor).domain, SparkDataFrameDomain)
        assert isinstance(relation.accept(pandas_visitor).domain, PandasTableDomain)

    def test_the_default_backend_is_spark(self, spark: Any) -> None:
        """Omitting the backend leaves the visitor exactly as it was."""
        tables = spark_tables(spark, {"episodes": EPISODES})
        relation = AddRemoveRows("episodes", n=1)

        defaulted = relation.accept(NeighboringRelationCoreVisitor(tables, PureDP()))
        explicit = relation.accept(
            NeighboringRelationCoreVisitor(tables, PureDP(), SPARK)
        )
        assert defaulted == explicit


class TestAddRemoveRowsParity:
    """AddMaxRows: one table, SymmetricDifference, d_in = n."""

    @pytest.mark.parametrize("n", [1, 5])
    def test_parity(self, spark: Any, n: int) -> None:
        """Both backends build the same domain, metric and d_in."""
        spark_visitor, pandas_visitor = visitors(spark, {"metrics": METRICS})
        relation = AddRemoveRows("metrics", n=n)

        spark_output = relation.accept(spark_visitor)
        pandas_output = relation.accept(pandas_visitor)

        assert_outputs_correspond(spark_output, pandas_output)
        assert pandas_output.metric == SymmetricDifference()
        assert pandas_output.distance == ExactNumber(n)

    def test_the_data_is_carried_through_untouched(self) -> None:
        """Each backend's output carries that backend's own frame."""
        tables = pandas_tables({"metrics": METRICS})
        visitor = NeighboringRelationCoreVisitor(tables, PureDP(), PANDAS)

        assert AddRemoveRows("metrics", n=1).accept(visitor).data is tables["metrics"]

    def test_validate_input_accepts_pandas_frames(self) -> None:
        """The relation validates a pandas table as readily as a Spark one."""
        assert AddRemoveRows("metrics", n=1).validate_input(
            pandas_tables({"metrics": METRICS})
        )


class TestAddRemoveRowsAcrossGroupsParity:
    """AddMaxRowsInMaxGroups: IfGroupedBy, and a measure-dependent d_in."""

    @pytest.mark.parametrize(
        "output_measure,agg_metric,distance",
        [
            (PureDP(), SumOf(SymmetricDifference()), ExactNumber(2 * 3)),
            (
                RhoZCDP(),
                RootSumOfSquared(SymmetricDifference()),
                ExactNumber(3 * ExactNumber(sp.sqrt(2))),
            ),
        ],
        ids=["PureDP", "RhoZCDP"],
    )
    def test_parity(
        self, spark: Any, output_measure: Any, agg_metric: Any, distance: ExactNumber
    ) -> None:
        """Both backends build the same domain, metric and d_in."""
        spark_visitor, pandas_visitor = visitors(
            spark, {"answers": ANSWERS}, output_measure
        )
        relation = AddRemoveRowsAcrossGroups("answers", "question", 2, 3)

        spark_output = relation.accept(spark_visitor)
        pandas_output = relation.accept(pandas_visitor)

        assert_outputs_correspond(spark_output, pandas_output)
        assert pandas_output.metric == IfGroupedBy(["question"], agg_metric)
        assert pandas_output.distance == distance

    @pytest.mark.parametrize("grouping_column", ["question", "score"])
    def test_validate_input_accepts_pandas_frames(self, grouping_column: str) -> None:
        """A VARCHAR and an INTEGER grouping column are both allowed."""
        assert AddRemoveRowsAcrossGroups(
            "answers", grouping_column, 1, 1
        ).validate_input(pandas_tables({"answers": ANSWERS}))


class TestAddRemoveKeysParity:
    """AddRowsWithID: a DictDomain of tables sharing one ID column."""

    def test_parity(self, spark: Any) -> None:
        """Both backends build the same DictDomain, metric and d_in."""
        tables = {"episodes": EPISODES, "answers": ANSWERS}
        spark_visitor, pandas_visitor = visitors(spark, tables)
        relation = AddRemoveKeys(
            "resident_id",
            {"episodes": "resident_id", "answers": "resident_id"},
            max_keys=3,
        )

        spark_output = relation.accept(spark_visitor)
        pandas_output = relation.accept(pandas_visitor)

        assert_outputs_correspond(spark_output, pandas_output)
        assert pandas_output.metric == CoreAddRemoveKeys(
            {
                NamedTable("episodes"): "resident_id",
                NamedTable("answers"): "resident_id",
            }
        )
        assert pandas_output.distance == ExactNumber(3)
        assert set(pandas_output.domain.key_to_domain) == {
            NamedTable("episodes"),
            NamedTable("answers"),
        }

    def test_validate_input_accepts_pandas_frames(self) -> None:
        """The relation validates pandas tables sharing an ID column."""
        assert AddRemoveKeys(
            "resident_id", {"episodes": "resident_id", "answers": "resident_id"}
        ).validate_input(pandas_tables({"episodes": EPISODES, "answers": ANSWERS}))

    def test_mixing_the_two_backends_is_rejected(self, spark: Any) -> None:
        """The other multi-table relation rejects a mixture of its own."""
        mixed = pandas_tables({"episodes": EPISODES})
        mixed["answers"] = ANSWERS.spark(spark)
        relation = AddRemoveKeys(
            "resident_id", {"episodes": "resident_id", "answers": "resident_id"}
        )

        with pytest.raises(ValueError, match="mixes backends"):
            relation.validate_input(mixed)


class TestNullableIDHarmonizationParity:
    """The one place the visitor rewrites a domain it has already built.

    An ``AddRemoveKeys`` metric only supports a domain whose ID columns agree
    about nullability, so when any one of them is nullable they all become
    nullable. The rule is engine-neutral -- widen ``allow_null`` on the ID
    column of every table in the ID space -- and these tests require the pandas
    path to apply it identically, including the case where nothing needs
    widening.
    """

    RELATION = AddRemoveKeys("space", {"strict": "id", "loose": "id"}, max_keys=2)

    def test_a_nullable_id_widens_every_table_in_the_space(self, spark: Any) -> None:
        """``int64`` beside ``Int64`` widens exactly as Spark's does."""
        tables = {"strict": NON_NULLABLE_ID, "loose": NULLABLE_ID}
        spark_visitor, pandas_visitor = visitors(spark, tables)

        spark_output = self.RELATION.accept(spark_visitor)
        pandas_output = self.RELATION.accept(pandas_visitor)

        assert_outputs_correspond(spark_output, pandas_output)
        for table in ("strict", "loose"):
            domain = pandas_output.domain.key_to_domain[NamedTable(table)]
            assert domain.schema["id"].allow_null is True
            # The widening is confined to the ID column: 'count' is an int64
            # column too, and it stays as it was.
            assert domain.schema["count"].allow_null is False

    def test_the_widened_domain_still_describes_the_data(self) -> None:
        """The rewritten pandas domain accepts the frames it describes."""
        tables = pandas_tables({"strict": NON_NULLABLE_ID, "loose": NULLABLE_ID})
        visitor = NeighboringRelationCoreVisitor(tables, PureDP(), PANDAS)

        output = self.RELATION.accept(visitor)
        for table in ("strict", "loose"):
            output.domain.key_to_domain[NamedTable(table)].validate(tables[table])

    def test_all_non_nullable_ids_are_left_alone(self, spark: Any) -> None:
        """With nothing to harmonize, the domains are untouched on both sides."""
        tables = {"strict": NON_NULLABLE_ID, "other": NON_NULLABLE_ID}
        relation = AddRemoveKeys("space", {"strict": "id", "other": "id"}, max_keys=2)
        spark_visitor, pandas_visitor = visitors(spark, tables)

        spark_output = relation.accept(spark_visitor)
        pandas_output = relation.accept(pandas_visitor)

        assert_outputs_correspond(spark_output, pandas_output)
        domain = pandas_output.domain.key_to_domain[NamedTable("strict")]
        assert domain.schema["id"].allow_null is False


class TestConjunctionParity:
    """The shape a real Session builds: several protected changes at once."""

    TABLES: ClassVar[Dict[str, Table]] = {
        "episodes": EPISODES,
        "answers": ANSWERS,
        "projects": PROJECTS,
        "metrics": METRICS,
    }
    """Two tables in one ID space, a rollup table in a second, and a row table."""

    RELATION = Conjunction(
        AddRemoveKeys(
            "resident_id",
            {"episodes": "resident_id", "answers": "resident_id"},
            max_keys=2,
        ),
        AddRemoveKeys("project_id", {"projects": "project_id"}, max_keys=1),
        AddRemoveRows("metrics", n=1),
    )

    def test_parity(self, spark: Any) -> None:
        """Both backends build the same nested domain, metric and d_in."""
        spark_visitor, pandas_visitor = visitors(spark, self.TABLES)

        spark_output = self.RELATION.accept(spark_visitor)
        pandas_output = self.RELATION.accept(pandas_visitor)

        assert_outputs_correspond(spark_output, pandas_output)

    def test_the_structure_is_the_one_the_accountant_expects(self, spark: Any) -> None:
        """A DictMetric over the two ID spaces and the row-protected table."""
        _, pandas_visitor = visitors(spark, self.TABLES)

        output = self.RELATION.accept(pandas_visitor)

        assert output.metric == DictMetric(
            {
                TableCollection("resident_id"): CoreAddRemoveKeys(
                    {
                        NamedTable("episodes"): "resident_id",
                        NamedTable("answers"): "resident_id",
                    }
                ),
                TableCollection("project_id"): CoreAddRemoveKeys(
                    {NamedTable("projects"): "project_id"}
                ),
                NamedTable("metrics"): SymmetricDifference(),
            }
        )
        assert output.distance == {
            TableCollection("resident_id"): ExactNumber(2),
            TableCollection("project_id"): ExactNumber(1),
            NamedTable("metrics"): ExactNumber(1),
        }
        # The two ID spaces are separate sub-dictionaries; the row-protected
        # table sits beside them rather than inside either.
        assert set(output.domain.key_to_domain) == {
            TableCollection("resident_id"),
            TableCollection("project_id"),
            NamedTable("metrics"),
        }
        assert isinstance(
            output.domain.key_to_domain[NamedTable("metrics")], PandasTableDomain
        )

    def test_validate_input_accepts_pandas_frames(self) -> None:
        """The whole mixed relation validates against pandas tables."""
        assert self.RELATION.validate_input(pandas_tables(self.TABLES))

    def test_mixing_the_two_backends_is_rejected(self, spark: Any) -> None:
        """A relation covers one accountant, and so one representation."""
        mixed = pandas_tables(self.TABLES)
        mixed["metrics"] = METRICS.spark(spark)

        with pytest.raises(ValueError, match="mixes backends"):
            self.RELATION.validate_input(mixed)


class TestKeyColumnTypeParity:
    """The ID-column type allowlist, and the errors it raises."""

    @staticmethod
    def _error(relation: Any, tables: Dict[str, Any]) -> str:
        with pytest.raises(ValueError) as excinfo:
            relation.validate_input(tables)
        return str(excinfo.value)

    @pytest.mark.parametrize(
        "table", [FLOAT_KEYED, TIMESTAMP_KEYED], ids=["DECIMAL", "TIMESTAMP"]
    )
    def test_a_disallowed_key_column_is_rejected_identically(
        self, spark: Any, table: Table
    ) -> None:
        """Both backends reject it, with the same message."""
        relation = AddRemoveKeys("space", {"table": "key"})

        spark_error = self._error(relation, spark_tables(spark, {"table": table}))
        pandas_error = self._error(relation, pandas_tables({"table": table}))

        assert spark_error == pandas_error
        assert "is not of a type allowed for keys" in pandas_error
        assert "DATE, INTEGER, VARCHAR" in pandas_error

    @pytest.mark.parametrize(
        "table", [FLOAT_KEYED, TIMESTAMP_KEYED], ids=["DECIMAL", "TIMESTAMP"]
    )
    def test_a_disallowed_grouping_column_is_rejected_identically(
        self, spark: Any, table: Table
    ) -> None:
        """The grouping column uses the same allowlist, on both backends."""
        relation = AddRemoveRowsAcrossGroups("table", "key", 1, 1)

        spark_error = self._error(relation, spark_tables(spark, {"table": table}))
        pandas_error = self._error(relation, pandas_tables({"table": table}))

        assert spark_error == pandas_error
        assert "grouping is supported" in pandas_error

    def test_key_types_must_match_across_tables_on_both_backends(
        self, spark: Any
    ) -> None:
        """A VARCHAR key beside an INTEGER key is rejected either way."""
        tables = {"episodes": EPISODES, "ids": NON_NULLABLE_ID}
        relation = AddRemoveKeys("space", {"episodes": "resident_id", "ids": "id"})

        spark_error = self._error(relation, spark_tables(spark, tables))
        pandas_error = self._error(relation, pandas_tables(tables))

        assert spark_error == pandas_error
        assert "Key types must match across tables" in pandas_error

    def test_int64_and_Int64_are_the_same_key_type(self) -> None:
        """Nullability is not part of the key type, as it is not on Spark."""
        assert AddRemoveKeys("space", {"strict": "id", "loose": "id"}).validate_input(
            pandas_tables({"strict": NON_NULLABLE_ID, "loose": NULLABLE_ID})
        )

    def test_a_missing_key_column_is_reported_before_its_type(self) -> None:
        """The clearer error wins, on pandas as on Spark."""
        relation = AddRemoveKeys("space", {"table": "nonexistent"})

        error = self._error(relation, pandas_tables({"table": FLOAT_KEYED}))

        assert "does not exist in the input" in error


class TestUnsupportedPandasInput:
    """Tables a pandas Session could not load are rejected during validation."""

    def test_an_unsupported_dtype_is_rejected(self) -> None:
        """A dtype with no Analytics column type stops the relation."""
        frame = pd.DataFrame({"A": pd.Series([True, False], dtype="bool")})

        with pytest.raises(ValueError, match="Unsupported pandas dtype"):
            AddRemoveRows("table", n=1).validate_input({"table": frame})

    def test_a_mixed_object_column_is_rejected(self) -> None:
        """An object column has to hold one kind of value."""
        frame = pd.DataFrame({"A": pd.Series(["a", DATE], dtype=object)})

        with pytest.raises(ValueError, match="Unsupported values in object column"):
            AddRemoveRows("table", n=1).validate_input({"table": frame})

    def test_a_duplicated_column_name_is_rejected(self) -> None:
        """Two columns of one name make the frame's schema ambiguous."""
        frame = pd.DataFrame([[1, 2]], columns=["A", "A"])

        with pytest.raises(ValueError, match="duplicate column names"):
            AddRemoveRows("table", n=1).validate_input({"table": frame})
