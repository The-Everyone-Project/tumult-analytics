"""Tests for the backend descriptors and the functions that choose between them."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import builtins
import datetime
import importlib
import sys
from typing import Any, List, Sequence, Tuple

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import (
    PandasGroupedTableDomain,
    PandasRowDomain,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
    SparkRowDomain,
)

import tmlt.analytics._backends as backends
from tmlt.analytics import AnalyticsInternalError, KeySet
from tmlt.analytics._backends import (
    PANDAS,
    SPARK,
    Backend,
    BackendUnavailable,
    NotSupportedByBackend,
    Ops,
    backend_for_dataframe,
    backend_for_domain,
)
from tmlt.analytics._backends._pandas import PANDAS_UNSUPPORTED_OPS
from tmlt.analytics._schema import (
    Schema,
    analytics_to_pandas_columns_descriptor,
    analytics_to_spark_columns_descriptor,
)

###############################################################################
# The ops table: what each backend can do, and what it says it cannot.
###############################################################################


def test_spark_ops_are_complete():
    """The Spark backend binds every operation in the ops table.

    Spark is the reference backend: a slot it does not fill is a slot the
    compiler cannot use at all, which would be a mistake rather than a choice.
    """
    assert [field for field in Ops._fields if getattr(SPARK.ops, field) is None] == []


def test_pandas_ops_are_bound_or_documented():
    """Every empty pandas slot has a documented reason, and no reason is stale."""
    empty = {field for field in Ops._fields if getattr(PANDAS.ops, field) is None}
    documented = set(PANDAS_UNSUPPORTED_OPS)
    assert empty - documented == set(), (
        "These pandas ops are None but PANDAS_UNSUPPORTED_OPS does not say why: "
        f"{sorted(empty - documented)}"
    )
    assert documented - empty == set(), (
        "PANDAS_UNSUPPORTED_OPS documents these ops as missing, but they are "
        f"bound: {sorted(documented - empty)}"
    )


def test_pandas_unsupported_ops_names_real_ops():
    """Every documented gap names a real ops-table field, with a real reason."""
    assert set(PANDAS_UNSUPPORTED_OPS) <= set(Ops._fields)
    for op, reason in PANDAS_UNSUPPORTED_OPS.items():
        assert reason.strip(), f"'{op}' has an empty reason"


def test_pandas_binds_the_shared_truncation_strategy():
    """Both backends take the very same TruncationStrategy enum.

    Core ships one enum rather than one per backend, so that a strategy chosen
    against one backend is the same value the other's joins compare against.
    """
    assert PANDAS.ops.TruncationStrategy is SPARK.ops.TruncationStrategy


@pytest.mark.parametrize("op", sorted(PANDAS_UNSUPPORTED_OPS))
def test_require_explains_a_missing_pandas_op(op: str):
    """Asking the pandas backend for an op it lacks names the op and the backend."""
    with pytest.raises(NotSupportedByBackend) as excinfo:
        PANDAS.require(op)
    assert excinfo.value.op == op
    assert excinfo.value.backend == "pandas"
    assert op in str(excinfo.value)


def test_pandas_descriptor_fields():
    """The pandas backend describes pandas tables."""
    assert PANDAS.name == "pandas"
    assert PANDAS.dataframe_type is pd.DataFrame
    assert PANDAS.dataframe_domain_type is PandasTableDomain
    assert PANDAS.row_domain_type is PandasRowDomain
    assert PANDAS.grouped_domain_type is PandasGroupedTableDomain


def test_pandas_schema_conversions_round_trip():
    """The pandas backend's schema conversions are each other's inverse."""
    schema = Schema({"A": "VARCHAR", "B": "INTEGER", "C": "DECIMAL"})
    domain = PANDAS.dataframe_domain(schema)
    assert isinstance(domain, PandasTableDomain)
    assert domain.schema == PANDAS.columns_descriptor(schema)
    assert domain.schema == analytics_to_pandas_columns_descriptor(schema)
    assert PANDAS.domain_to_analytics_columns(domain) == schema.column_descs


def test_pandas_coercion_copies():
    """The pandas backend's coercion is the copy-on-ingest boundary."""
    frame = pd.DataFrame({"A": ["a"], "B": [1]})
    coerced = PANDAS.coerce_schema_or_fail(frame)
    assert coerced is not frame
    pd.testing.assert_frame_equal(coerced, frame)


###############################################################################
# Choosing a backend from data.
###############################################################################


def test_backend_for_dataframe_spark(spark):
    """A Spark DataFrame chooses the Spark backend."""
    assert (
        backend_for_dataframe(spark.createDataFrame(pd.DataFrame({"A": [1]}))) is SPARK
    )


def test_backend_for_dataframe_pandas():
    """A pandas DataFrame chooses the pandas backend."""
    assert backend_for_dataframe(pd.DataFrame({"A": [1]})) is PANDAS


@pytest.mark.parametrize("value", [None, 17, "a frame", [1, 2, 3], pd.Series([1])])
def test_backend_for_dataframe_rejects_other_types(value: Any):
    """Anything that is not a table of some backend is rejected by name."""
    with pytest.raises(AnalyticsInternalError, match="No backend handles tables"):
        backend_for_dataframe(value)


def test_backend_for_domain_spark():
    """Every Spark domain family member chooses the Spark backend."""
    descriptor = analytics_to_spark_columns_descriptor(Schema({"A": "VARCHAR"}))
    assert backend_for_domain(SparkDataFrameDomain(descriptor)) is SPARK
    assert backend_for_domain(SparkRowDomain(descriptor)) is SPARK
    assert (
        backend_for_domain(
            SparkGroupedDataFrameDomain(descriptor, groupby_columns=["A"])
        )
        is SPARK
    )


def test_backend_for_domain_pandas():
    """Every pandas domain family member chooses the pandas backend."""
    descriptor = analytics_to_pandas_columns_descriptor(Schema({"A": "VARCHAR"}))
    assert backend_for_domain(PandasTableDomain(descriptor)) is PANDAS
    assert backend_for_domain(PandasRowDomain(descriptor)) is PANDAS
    assert (
        backend_for_domain(PandasGroupedTableDomain(descriptor, groupby_columns=["A"]))
        is PANDAS
    )


def test_backend_for_domain_rejects_other_domains():
    """A domain that is not a table domain of either backend is rejected."""
    with pytest.raises(AnalyticsInternalError, match="No backend handles tables in"):
        backend_for_domain(NumpyIntegerDomain())


def test_backend_for_dataframe_and_domain_agree():
    """The two ways of choosing a backend choose the same one."""
    schema = Schema({"A": "VARCHAR", "B": "INTEGER"})
    for backend in (SPARK, PANDAS):
        assert backend_for_domain(backend.dataframe_domain(schema)) is backend


###############################################################################
# Importing the pandas backend, and the guard that explains a Core without it.
###############################################################################


def test_pandas_is_resolved_lazily_and_cached():
    """`PANDAS` is the module attribute, resolved once and reused."""
    assert backends.PANDAS is PANDAS
    assert backends.PANDAS is backends.PANDAS
    assert isinstance(backends.PANDAS, Backend)


def test_backends_module_rejects_unknown_attributes():
    """The lazy attribute hook does not invent attributes."""
    with pytest.raises(AttributeError, match="no attribute 'NOT_A_BACKEND'"):
        _ = backends.NOT_A_BACKEND


def test_import_guard_names_the_missing_core_artifact(monkeypatch):
    """A Core without the pandas transformations gives an actionable error.

    The import is made to fail the way a stock Core makes it fail, and the
    module is re-imported from scratch, so this exercises the guard itself
    rather than a stand-in for it.
    """
    module_name = "tmlt.analytics._backends._pandas"
    missing = "tmlt.core.transformations.pandas_transformations"
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith(missing):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(BackendUnavailable) as excinfo:
        importlib.import_module(module_name)

    assert excinfo.value.backend == "pandas"
    assert excinfo.value.required == missing
    message = str(excinfo.value)
    assert missing in message
    # The error has to say what would fix it, not only what is wrong: which
    # build line has the module, and that getting it means repointing the
    # source rather than raising a bound, since no specifier can name a local
    # version.
    assert "ep.backend" in message
    assert "tool.uv.sources" in message
    assert isinstance(excinfo.value, ImportError)


###############################################################################
# sample_keyset: the same stand-in keyset, built each backend's way.
###############################################################################

_SAMPLE_SCHEMAS: Tuple[Tuple[str, Schema, Sequence[str]], ...] = (
    ("one integer", Schema({"A": "INTEGER"}), ["A"]),
    ("one string", Schema({"A": "VARCHAR"}), ["A"]),
    ("one date", Schema({"A": "DATE"}), ["A"]),
    (
        "every keyset type",
        Schema({"i": "INTEGER", "s": "VARCHAR", "d": "DATE"}),
        ["i", "s", "d"],
    ),
    (
        "a subset of the columns",
        Schema({"i": "INTEGER", "s": "VARCHAR", "d": "DATE"}),
        ["s"],
    ),
    (
        "columns named out of domain order",
        Schema({"i": "INTEGER", "s": "VARCHAR"}),
        ["s", "i"],
    ),
    ("no columns at all", Schema({"i": "INTEGER"}), []),
)


def _reference_sample_keyset(schema: StructType) -> KeySet:
    """The sample-keyset builder as it was before it moved behind the backend.

    Kept here verbatim so that the move can be shown not to have changed what
    the Spark backend produces.
    """
    spark = SparkSession.builder.getOrCreate()
    if not schema.fields:
        return KeySet.from_dataframe(spark.createDataFrame([], schema=schema))
    default_values: List[Any] = []
    for field in schema.fields:
        if isinstance(field.dataType, LongType):
            default_values.append(0)
        elif isinstance(field.dataType, StringType):
            default_values.append("")
        elif isinstance(field.dataType, DateType):
            default_values.append(datetime.date(1970, 1, 1))
        else:
            raise ValueError(f"Unsupported data type {field.dataType}")
    return KeySet.from_dataframe(
        spark.createDataFrame([tuple(default_values)], schema=schema)
    )


def _as_rows(keyset: KeySet) -> List[Tuple[Any, ...]]:
    """The keyset's rows, in a form two KeySets can be compared by."""
    columns = keyset.columns()
    return sorted(
        tuple(row[column] for column in columns) for row in keyset.dataframe().collect()
    )


@pytest.mark.parametrize(
    "schema,columns",
    [(schema, columns) for _, schema, columns in _SAMPLE_SCHEMAS],
    ids=[name for name, _, _ in _SAMPLE_SCHEMAS],
)
def test_spark_sample_keyset_matches_the_helper_it_replaced(
    spark, schema: Schema, columns: Sequence[str]
):
    """Moving the sample-keyset builder into the backend did not change it."""
    # pylint: disable=unused-argument
    domain = SPARK.dataframe_domain(schema)
    assert isinstance(domain, SparkDataFrameDomain)
    expected_fields = StructType(
        [field for field in domain.spark_schema if field.name in columns]
    )
    expected = _reference_sample_keyset(expected_fields)
    actual = SPARK.sample_keyset(domain, columns)
    assert actual.columns() == expected.columns()
    assert actual.schema() == expected.schema()
    assert _as_rows(actual) == _as_rows(expected)


@pytest.mark.parametrize(
    "schema,columns",
    [(schema, columns) for _, schema, columns in _SAMPLE_SCHEMAS],
    ids=[name for name, _, _ in _SAMPLE_SCHEMAS],
)
def test_pandas_sample_keyset_matches_spark(
    spark, schema: Schema, columns: Sequence[str]
):
    """The two backends build the same sample keyset from the same schema."""
    # pylint: disable=unused-argument
    from_spark = SPARK.sample_keyset(SPARK.dataframe_domain(schema), columns)
    from_pandas = PANDAS.sample_keyset(PANDAS.dataframe_domain(schema), columns)
    assert from_pandas.columns() == from_spark.columns()
    assert from_pandas.schema() == from_spark.schema()
    assert _as_rows(from_pandas) == _as_rows(from_spark)


def test_pandas_sample_keyset_is_not_backed_by_a_dataframe():
    """The pandas sample keyset is built from tuples, never from a DataFrame.

    A KeySet built from a dataframe holds that dataframe, and the only
    dataframe a KeySet takes is a Spark one -- which is exactly what a pandas
    Session is meant not to need.
    """
    domain = PANDAS.dataframe_domain(Schema({"A": "VARCHAR", "B": "INTEGER"}))
    keyset = PANDAS.sample_keyset(domain, ["A", "B"])
    assert keyset.columns() == ["A", "B"]
    # pylint: disable-next=protected-access
    assert "FromSparkDataFrame" not in repr(keyset._op_tree)


def test_pandas_sample_keyset_rejects_a_foreign_domain():
    """The pandas builder says so when handed a domain it does not describe."""
    domain = SPARK.dataframe_domain(Schema({"A": "VARCHAR"}))
    with pytest.raises(AnalyticsInternalError, match="cannot build a sample KeySet"):
        PANDAS.sample_keyset(domain, ["A"])


def test_spark_sample_keyset_rejects_a_foreign_domain():
    """The Spark builder says so when handed a domain it does not describe."""
    domain = PANDAS.dataframe_domain(Schema({"A": "VARCHAR"}))
    with pytest.raises(AnalyticsInternalError, match="cannot build a sample KeySet"):
        SPARK.sample_keyset(domain, ["A"])
