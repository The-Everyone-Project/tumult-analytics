"""Tests for the backend-parity test harness itself.

The harness is the oracle a parity suite is judged against, so it needs its own
tests: a comparison helper that quietly treats a null as a NaN, or a conversion
that rounds a large integer, would turn a real disagreement between the backends
into a green test. Everything here is about
:mod:`test.backend_testing` and nothing about Analytics' own behavior.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from tmlt.analytics import AddMaxRows, PureDPBudget
from tmlt.analytics._backends import SPARK
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.config import config

from test.backend_testing import (
    ID1,
    ID2,
    ID3,
    ID4,
    NAN,
    NULL,
    PRIVATE_ID_DATA,
    ROWS1,
    STANDARD_TABLES,
    BackendFixture,
    TableSpec,
    all_nullability_divergences,
    analytics_columns,
    assert_frame_equal_across_backends,
    frame_as_rows,
    pandas_dtypes_for,
    pandas_frame,
    pandas_frame_from_rows,
    spark_frame,
    to_pandas,
    value_key,
)

BIG = 2**53 + 1
"""The smallest integer a float64 cannot represent."""

SPECS: List[TableSpec] = list(STANDARD_TABLES.values())
SPEC_IDS = [spec.name for spec in SPECS]


###############################################################################
# to_pandas: the conversion traps.
###############################################################################


@pytest.fixture(name="lossy_frame")
def fixture_lossy_frame(spark):
    """A Spark frame holding every value the naive conversion damages."""
    return spark.createDataFrame(
        [[BIG, 1.0, "a"], [None, float("nan"), None], [3, None, "c"]],
        schema=StructType(
            [
                StructField("i", LongType(), nullable=True),
                StructField("d", DoubleType(), nullable=True),
                StructField("s", StringType(), nullable=True),
            ]
        ),
    )


def test_large_integers_survive_the_conversion(lossy_frame):
    """An integer above 2**53 comes back exactly, in a nullable integer column."""
    converted = to_pandas(lossy_frame)
    assert converted["i"].dtype == pd.Int64Dtype()
    # Read back out as a Python int: comparing the raw cell to BIG would promote
    # it to a float and report equality even if it had been corrupted.
    assert int(converted["i"][0]) == BIG
    assert converted["i"][1] is pd.NA
    assert int(converted["i"][2]) == 3


def test_topandas_is_the_lossy_conversion_this_replaces(lossy_frame):
    """The trap the shim exists for, pinned rather than described.

    Not asserted unconditionally: this documents PySpark's behavior, and the
    point of the shim is that Analytics' tests no longer depend on it.
    """
    naive = lossy_frame.toPandas()
    if pd.api.types.is_float_dtype(naive["i"].dtype):
        assert int(naive["i"][0]) != BIG, (
            "toPandas() no longer widens a nullable long to float64; the shim is "
            "still correct, but this note is out of date."
        )
    if pd.api.types.is_float_dtype(naive["d"].dtype):
        # The NaN in row 1 and the NULL in row 2 have become the same value.
        assert naive["d"].isna().tolist() == [False, True, True]


def test_null_and_nan_stay_apart_in_a_floating_point_column(lossy_frame):
    """A NULL becomes pd.NA and a NaN stays a NaN, in one Float64 column."""
    converted = to_pandas(lossy_frame)
    assert converted["d"].dtype == pd.Float64Dtype()
    assert converted["d"][0] == 1.0
    assert np.isnan(converted["d"][1])
    assert converted["d"][2] is pd.NA


def test_strings_and_nulls_survive(lossy_frame):
    """An object column keeps its Nones as Nones."""
    converted = to_pandas(lossy_frame)
    assert converted["s"].dtype == np.dtype(object)
    assert list(converted["s"]) == ["a", None, "c"]


def test_empty_frames_keep_their_dtypes(lossy_frame):
    """An empty result still says what its columns are."""
    empty = lossy_frame.filter("i = -1")
    converted = to_pandas(empty)
    assert len(converted) == 0
    assert list(converted.columns) == ["i", "d", "s"]
    assert converted["i"].dtype == pd.Int64Dtype()
    assert converted["d"].dtype == pd.Float64Dtype()
    assert converted["s"].dtype == np.dtype(object)


def test_non_nullable_columns_use_the_numpy_dtypes(spark):
    """A column Spark says cannot be null does not need the masked dtype."""
    frame = spark.createDataFrame(
        [[1, 2.5]],
        schema=StructType(
            [
                StructField("i", LongType(), nullable=False),
                StructField("d", DoubleType(), nullable=False),
            ]
        ),
    )
    converted = to_pandas(frame)
    assert converted["i"].dtype == np.dtype("int64")
    assert converted["d"].dtype == np.dtype("float64")


def test_an_explicit_schema_overrides_the_frames_own(spark):
    """The caller can say what the result's schema is meant to be."""
    frame = spark.createDataFrame(
        [[1]], schema=StructType([StructField("i", LongType(), nullable=False)])
    )
    schema = Schema({"i": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)})
    assert to_pandas(frame, schema=schema)["i"].dtype == pd.Int64Dtype()
    assert to_pandas(frame, dtypes={"i": np.dtype("int64")})["i"].dtype == np.dtype(
        "int64"
    )


def test_to_pandas_is_the_identity_on_a_pandas_frame():
    """A pandas frame with nothing to convert is returned as it is."""
    frame = pd.DataFrame({"a": [1, 2]})
    assert to_pandas(frame) is frame


def test_to_pandas_retypes_a_pandas_frame_given_a_schema():
    """Given a schema, a pandas frame is rebuilt at that schema's dtypes."""
    frame = pd.DataFrame({"a": np.array([1, 2], dtype="int64")})
    schema = Schema({"a": ColumnDescriptor(ColumnType.INTEGER, allow_null=True)})
    converted = to_pandas(frame, schema=schema)
    assert converted["a"].dtype == pd.Int64Dtype()
    assert [int(value) for value in converted["a"]] == [1, 2]


def test_to_pandas_rejects_things_that_are_not_frames():
    """A value of neither backend's frame type is an error, not a guess.

    The call carries no ``# type: ignore``: pandas ships no ``py.typed`` marker,
    so ``AnyFrame`` is ``Any`` to mypy and a wrong argument type is invisible to
    it. Which is why the check exists at runtime.
    """
    with pytest.raises(TypeError, match="Not a frame of either backend"):
        to_pandas([1, 2, 3])


###############################################################################
# Building pandas frames from Python values.
###############################################################################


def test_a_null_in_a_non_nullable_column_is_refused():
    """Silently turning the null into a NaN is exactly what must not happen."""
    with pytest.raises(ValueError, match="allow_null"):
        pandas_frame_from_rows(["d"], [(1.0,), (None,)], {"d": np.dtype("float64")})
    with pytest.raises(ValueError, match="allow_null"):
        pandas_frame_from_rows(["i"], [(None,)], {"i": np.dtype("int64")})


def test_rows_must_be_the_right_width():
    """A row with a missing value is a bug in the test, and says so."""
    with pytest.raises(ValueError, match="2 values"):
        pandas_frame_from_rows(["a", "b", "c"], [(1, 2)], {})


def test_every_column_needs_a_dtype():
    """Nothing is inferred: an unlisted column is an error."""
    with pytest.raises(ValueError, match="No dtype given"):
        pandas_frame_from_rows(["a"], [(1,)], {})


def test_a_non_integral_value_in_an_integer_column_is_refused():
    """Truncation is silent in numpy, and would hide a real difference."""
    for dtype in (np.dtype("int64"), pd.Int64Dtype()):
        with pytest.raises(ValueError, match="not an integer"):
            pandas_frame_from_rows(["i"], [(1.5,)], {"i": dtype})
        with pytest.raises(ValueError, match="not an integer"):
            pandas_frame_from_rows(["i"], [(float("nan"),)], {"i": dtype})
    # A float that loses nothing is fine.
    assert list(
        pandas_frame_from_rows(["i"], [(5.0,)], {"i": pd.Int64Dtype()})["i"]
    ) == [5]


def test_nan_and_null_can_share_a_nullable_float_column():
    """The distinction the whole harness rests on, built rather than cast."""
    frame = pandas_frame_from_rows(
        ["d"], [(1.0,), (float("nan"),), (None,)], {"d": pd.Float64Dtype()}
    )
    assert np.isnan(frame["d"][1])
    assert frame["d"][2] is pd.NA


###############################################################################
# The comparison.
###############################################################################


def _frame(**columns: Any) -> pd.DataFrame:
    """Builds a small pandas frame, inferring dtypes as pandas would.

    Args:
        **columns: One column per keyword argument.

    Returns:
        The frame.
    """
    return pd.DataFrame(columns)


def test_value_key_taxonomy():
    """Nulls collapse onto one sentinel, NaN onto another, numbers onto values."""
    assert value_key(None) is NULL
    assert value_key(pd.NA) is NULL
    assert value_key(pd.NaT) is NULL
    assert value_key(float("nan")) is NAN
    assert value_key(np.float64("nan")) is NAN
    assert NULL != NAN
    assert value_key(np.int64(5)) == 5
    assert value_key(BIG) == BIG
    assert value_key(BIG) != float(BIG)


def test_equal_frames_compare_equal():
    """Dtypes and column order are not part of the answer."""
    result = pd.DataFrame(
        {"count": pd.array([5, 1], dtype="Int64"), "group": ["A", "B"]}
    )
    expected = pd.DataFrame(
        {"group": ["A", "B"], "count": np.array([5, 1], dtype="int64")}
    )
    assert_frame_equal_across_backends(result, expected)


def test_none_does_not_equal_nan_in_an_object_column():
    """The comparison pandas' own assert_frame_equal gets wrong."""
    result = pandas_frame_from_rows(["x"], [(None,)], {"x": np.dtype(object)})
    expected = pandas_frame_from_rows(["x"], [(float("nan"),)], {"x": np.dtype(object)})
    with pytest.raises(AssertionError, match="differ in 1 of 1 rows"):
        assert_frame_equal_across_backends(result, expected)


def test_null_does_not_equal_nan_in_a_nullable_float_column():
    """The same distinction, in the dtype a nullable DECIMAL column uses."""
    result = pandas_frame_from_rows(["d"], [(None,)], {"d": pd.Float64Dtype()})
    expected = pandas_frame_from_rows(
        ["d"], [(float("nan"),)], {"d": pd.Float64Dtype()}
    )
    with pytest.raises(AssertionError):
        assert_frame_equal_across_backends(result, expected)


def test_integers_are_compared_exactly():
    """Two integers that differ only above 2**53 are two different answers."""
    result = pandas_frame_from_rows(["i"], [(BIG,)], {"i": pd.Int64Dtype()})
    expected = pandas_frame_from_rows(["i"], [(2**53,)], {"i": np.dtype("int64")})
    with pytest.raises(AssertionError):
        assert_frame_equal_across_backends(result, expected)
    same = pandas_frame_from_rows(["i"], [(BIG,)], {"i": np.dtype("int64")})
    assert_frame_equal_across_backends(result, same)


def test_row_order_matters_unless_sorting_is_asked_for():
    """A groupby result has no order, so the test has to say what to sort by."""
    result = _frame(group=["B", "A"], count=[1, 5])
    expected = _frame(group=["A", "B"], count=[5, 1])
    with pytest.raises(AssertionError, match="in frame order"):
        assert_frame_equal_across_backends(result, expected)
    assert_frame_equal_across_backends(result, expected, sort_by=["group"])
    # An empty sort_by sorts by every column: a multiset comparison.
    assert_frame_equal_across_backends(result, expected, sort_by=[])


def test_sorting_by_a_group_key_is_stable_across_duplicate_keys():
    """Rows sharing a group key are ordered by the rest of the row, not by luck."""
    result = _frame(group=["A", "A"], count=[2, 1])
    expected = _frame(group=["A", "A"], count=[1, 2])
    assert_frame_equal_across_backends(result, expected, sort_by=["group"])


def test_differing_columns_are_reported():
    """A missing or unexpected column names itself."""
    with pytest.raises(AssertionError, match="different columns"):
        assert_frame_equal_across_backends(_frame(a=[1]), _frame(b=[1]))


def test_differing_lengths_are_reported():
    """A lost or duplicated row is a failure, not a truncated comparison."""
    with pytest.raises(AssertionError, match="different lengths"):
        assert_frame_equal_across_backends(_frame(a=[1, 2]), _frame(a=[1]))


def test_sorting_by_an_unknown_column_is_an_error():
    """A typo in sort_by must not silently compare in frame order."""
    with pytest.raises(AssertionError, match="Cannot sort by"):
        assert_frame_equal_across_backends(
            _frame(a=[1]), _frame(a=[1]), sort_by=["nope"]
        )


def test_frame_as_rows_reads_a_frame_the_way_the_comparison_does():
    """The ad-hoc reader and the assertion agree on what a value is."""
    frame = pandas_frame_from_rows(
        ["d"], [(None,), (float("nan"),), (1.5,)], {"d": pd.Float64Dtype()}
    )
    assert frame_as_rows(frame) == [{"d": NULL}, {"d": NAN}, {"d": 1.5}]


###############################################################################
# The standard tables, materialized.
###############################################################################


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_a_spec_materializes_with_the_schema_its_backend_reports(
    backend: BackendFixture, spec: TableSpec
):
    """Each backend's frame has the columns expected_columns says it has."""
    frame = backend.materialize(spec)
    assert isinstance(frame, backend.descriptor.dataframe_type)
    assert analytics_columns(frame) == backend.expected_columns(spec)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_the_two_materializations_hold_the_same_values(spark, spec: TableSpec):
    """The pandas twin of each standard table is the same data as the Spark one."""
    assert_frame_equal_across_backends(
        spark_frame(spec, spark), pandas_frame(spec), sort_by=[]
    )


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_the_spark_materialization_reproduces_the_spec(spark, spec: TableSpec):
    """A Spark frame reproduces a spec's schema exactly, nullability included."""
    assert analytics_columns(spark_frame(spec, spark)) == spec.column_descs()


def test_the_documented_nullability_divergences_are_the_only_ones():
    """Where the backends cannot agree is a fixed, named list.

    A new divergence -- or one that has been fixed -- must be an edit to
    :mod:`test.backend_testing.materialize`'s docstring, not a surprise in a
    parity test.
    """
    divergences = all_nullability_divergences(SPECS)
    assert {
        name: [column for column, _ in columns] for name, columns in divergences.items()
    } == {"id3": ["group"], "id4": ["group"], "rows1": ["A"]}


def test_pandas_dtypes_follow_nullability():
    """The nullable extension dtypes are used exactly where nulls are allowed."""
    assert pandas_dtypes_for(ID3.schema) == {
        "id": pd.Int64Dtype(),
        "group": np.dtype(object),
        "x": pd.Int64Dtype(),
    }
    assert pandas_dtypes_for(ID4.schema) == {
        "id": np.dtype("int64"),
        "group": np.dtype(object),
        "x": np.dtype("int64"),
    }
    assert pandas_dtypes_for(ID1.schema)["float_n"] == pd.Float64Dtype()


###############################################################################
# The specs are the frames the existing suite already runs on.
###############################################################################


def _fixture_frames(spark) -> Dict[str, Any]:
    """Builds the standard frames the way the existing fixtures build them.

    A copy of the constructions in ``test/system/conftest.py`` and (for
    ``private_id_data``) ``test/conftest.py``, so that the specs can be checked
    against them. It is a copy rather than a call because those are
    module-scoped fixtures of another directory, and because the point is to
    compare against what the suite actually writes.

    Args:
        spark: The Spark session.

    Returns:
        The frames, keyed as the fixtures key them.
    """
    return {
        "id1": spark.createDataFrame(
            pd.DataFrame(
                [
                    [1, "A", "X", 4, 4.0],
                    [1, "A", "Y", 5, 5.0],
                    [1, "A", "X", 6, 6.0],
                    [2, "A", "Y", 7, 7.0],
                    [3, "A", "X", 8, 8.0],
                    [3, "B", "Y", 9, 9.0],
                ],
                columns=["id", "group", "group2", "n", "float_n"],
            )
        ),
        "id2": spark.createDataFrame(
            pd.DataFrame(
                [
                    [1, "A", 12],
                    [1, "B", 15],
                    [1, "A", 18],
                    [2, "B", 21],
                    [3, "A", 24],
                    [3, "B", 27],
                ],
                columns=["id", "group", "x"],
            )
        ),
        "id3": spark.createDataFrame(
            [
                [1, "A", 12],
                [None, "B", 15],
                [1, "A", 18],
                [2, "B", None],
                [3, "A", 24],
                [3, "B", 27],
                [None, "A", 30],
            ],
            schema=StructType(
                [
                    StructField("id", IntegerType(), nullable=True),
                    StructField("group", StringType(), nullable=False),
                    StructField("x", LongType(), nullable=True),
                ]
            ),
        ),
        "id4": spark.createDataFrame(
            [
                [1, "A", 12],
                [1, "B", 15],
                [1, "A", 18],
                [2, "B", 21],
                [3, "A", 24],
                [3, "B", 27],
            ],
            schema=StructType(
                [
                    StructField("id", IntegerType(), nullable=False),
                    StructField("group", StringType(), nullable=False),
                    StructField("x", LongType(), nullable=False),
                ]
            ),
        ),
        "rows1": spark.createDataFrame(
            [["0", 0, 0], ["0", 0, 1], ["0", 1, 2], ["1", 0, 3]],
            schema=StructType(
                [
                    StructField("A", StringType(), nullable=False),
                    StructField("B", LongType(), nullable=False),
                    StructField("X", LongType(), nullable=False),
                ]
            ),
        ),
        "private_id_data": spark.createDataFrame(
            pd.DataFrame(
                [
                    [1, 4, 100, "X"],
                    [1, 5, 100, "Y"],
                    [1, 6, 100, "X"],
                    [2, 7, 100, "Y"],
                    [3, 8, 100, "X"],
                    [3, 9, 100, "Y"],
                ],
                columns=["id", "A", "B", "X"],
            )
        ),
    }


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_a_spec_restates_the_frame_the_existing_fixtures_build(spark, spec: TableSpec):
    """Each spec is the table the suite already uses, said differently.

    Compared after Analytics' own Spark coercion, which is what a Session sees:
    that is where ``id3`` and ``id4``'s ``IntegerType`` ID columns become
    ``bigint``, which is why the specs declare them as plain ``INTEGER``.
    """
    fixture_frame = SPARK.coerce_schema_or_fail(_fixture_frames(spark)[spec.name])
    assert analytics_columns(fixture_frame) == spec.column_descs()
    assert_frame_equal_across_backends(
        fixture_frame, spark_frame(spec, spark), sort_by=[]
    )


###############################################################################
# The backend fixture.
###############################################################################


def test_the_fixture_names_a_real_backend(backend: BackendFixture):
    """The descriptor and the name agree, on both parameters.

    Case-insensitively: the descriptors spell themselves ``"Spark"`` and
    ``"pandas"``, and those strings appear in user-facing error messages, so the
    harness lowercases rather than changing them.
    """
    assert backend.name in ("spark", "pandas")
    assert backend.descriptor.name.lower() == backend.name


def test_only_the_spark_parameter_carries_a_session(backend: BackendFixture):
    """The pandas parameter has no Spark session, and says so when asked."""
    if backend.name == "spark":
        assert backend.spark is not None
        assert backend.require_spark() is backend.spark
    else:
        assert backend.spark is None
        with pytest.raises(RuntimeError, match="carries no Spark session"):
            backend.require_spark()


def test_the_pandas_parameter_does_not_ask_for_a_spark_session(
    backend: BackendFixture, request: pytest.FixtureRequest
):
    """The JVM stays out of the pandas lane.

    The fixture resolves Spark with ``getfixturevalue`` inside its Spark branch,
    so ``spark`` is not in a pandas run's fixture closure: nothing about
    collecting or running this test can start a JVM.
    """
    if backend.is_pandas:
        assert "spark" not in request.fixturenames


def test_the_pandas_feature_flag_is_open_for_the_pandas_parameter(
    backend: BackendFixture,
):
    """A pandas table can be added to a builder inside a test taking the fixture."""
    assert bool(config.features.pandas_backend) == backend.is_pandas


def test_a_session_builds_on_either_backend(backend: BackendFixture):
    """The fixture builds a Session, and the Session is on the right backend.

    This stops at ``build()``: evaluating a query on the pandas backend needs
    the operations binding that lands with a4b.
    """
    session = backend.build_session(
        {"t": ROWS1},
        budget=PureDPBudget(float("inf")),
        protected_change=AddMaxRows(2),
    )
    assert session._backend is backend.descriptor
    assert session.private_sources == ["t"]
    assert session.get_schema("t") == backend.expected_columns(ROWS1)


def test_a_session_can_hold_several_tables(backend: BackendFixture):
    """Specs may be passed as a sequence, keyed by their own names."""
    session = backend.build_session(
        [ID2, ROWS1],
        budget=PureDPBudget(float("inf")),
        protected_change={"id2": AddMaxRows(1), "rows1": AddMaxRows(2)},
    )
    assert sorted(session.private_sources) == ["id2", "rows1"]


def test_the_builder_carries_the_budget(backend: BackendFixture):
    """A builder from the fixture is ready to have tables added to it."""
    session = (
        backend.session_builder(PureDPBudget(3))
        .with_private_dataframe(
            "t", backend.materialize(ROWS1), protected_change=AddMaxRows(1)
        )
        .build()
    )
    assert session.remaining_privacy_budget == PureDPBudget(3)
