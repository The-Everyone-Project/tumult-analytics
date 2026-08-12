"""Tests for how a builder chooses, and enforces, its backend.

Nothing here calls ``build()`` on a pandas builder except the one test at the
bottom that documents why it cannot yet: the neighboring-relation visitor is
still Spark-only, so a pandas ``build()`` fails inside it. Everything the
builder itself decides -- which backend a table puts it on, which combinations
it refuses, and which coercion it routes through -- is decided before ``build()``
is reached, and is tested here without it.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import numpy as np
import pandas as pd
import pytest

from tmlt.analytics import (
    AddMaxRows,
    AddMaxRowsInMaxGroups,
    AddOneRow,
    AddRowsWithID,
    PureDPBudget,
    Session,
)
from tmlt.analytics._backends import PANDAS, SPARK, NotSupportedByBackend
from tmlt.analytics._base_builder import BaseBuilder, DataFrameMixin
from tmlt.analytics.config import config
from tmlt.analytics.protected_change import ProtectedChange


class _DataFrameBuilder(DataFrameMixin, BaseBuilder):
    """A builder that is nothing but its dataframes, for testing the mixin."""

    def build(self):
        return self._private_dataframes, self._public_dataframes, self._backend


@pytest.fixture(name="pandas_frame")
def fixture_pandas_frame() -> pd.DataFrame:
    """A small pandas table."""
    return pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})


@pytest.fixture(name="spark_frame")
def fixture_spark_frame(spark):
    """The same table, in Spark."""
    return spark.createDataFrame(pd.DataFrame({"A": ["a", "b"], "B": [1, 2]}))


@pytest.fixture(name="pandas_enabled")
def fixture_pandas_enabled():
    """Run the test with the pandas backend feature flag enabled."""
    with config.features.pandas_backend.enabled():
        yield


###############################################################################
# The feature flag.
###############################################################################


def test_pandas_frame_requires_the_feature_flag(pandas_frame):
    """A pandas table is refused while the feature flag is disabled."""
    builder = _DataFrameBuilder()
    with config.features.pandas_backend.disabled():
        with pytest.raises(RuntimeError, match="pandas_backend"):
            builder.with_private_dataframe("t", pandas_frame, AddOneRow())
    # The refused table left no trace.
    assert builder._private_dataframes == {}
    assert builder._backend is None


def test_spark_frame_needs_no_feature_flag(spark_frame):
    """The Spark backend is unaffected by the pandas feature flag."""
    with config.features.pandas_backend.disabled():
        builder = _DataFrameBuilder().with_private_dataframe(
            "t", spark_frame, AddOneRow()
        )
    assert builder._backend is SPARK


###############################################################################
# Inference.
###############################################################################


def test_first_private_frame_chooses_the_backend(pandas_frame, pandas_enabled):
    """A pandas table puts the builder on the pandas backend."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder().with_private_dataframe("t", pandas_frame, AddOneRow())
    assert builder._backend is PANDAS


def test_spark_frame_chooses_spark(spark_frame):
    """A Spark table puts the builder on the Spark backend."""
    builder = _DataFrameBuilder().with_private_dataframe("t", spark_frame, AddOneRow())
    assert builder._backend is SPARK


def test_backend_is_unset_before_the_first_private_frame():
    """A builder with no private table yet has no backend."""
    assert _DataFrameBuilder()._backend is None


def test_several_pandas_frames_stay_on_pandas(pandas_frame, pandas_enabled):
    """More tables of the same kind do not change the backend."""
    # pylint: disable=unused-argument
    builder = (
        _DataFrameBuilder()
        .with_private_dataframe("t1", pandas_frame, AddOneRow())
        .with_private_dataframe("t2", pandas_frame, AddMaxRows(3))
    )
    assert builder._backend is PANDAS
    assert set(builder._private_dataframes) == {"t1", "t2"}


###############################################################################
# Mixing backends.
###############################################################################


def test_pandas_after_spark_is_rejected(spark_frame, pandas_frame, pandas_enabled):
    """A pandas table cannot join a Spark Session."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder().with_private_dataframe("t1", spark_frame, AddOneRow())
    with pytest.raises(ValueError) as excinfo:
        builder.with_private_dataframe("t2", pandas_frame, AddOneRow())
    message = str(excinfo.value)
    assert "'t2'" in message
    assert "DataFrame" in message
    assert "Spark backend" in message
    assert builder._backend is SPARK
    assert set(builder._private_dataframes) == {"t1"}


def test_spark_after_pandas_is_rejected(spark_frame, pandas_frame, pandas_enabled):
    """A Spark table cannot join a pandas Session."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder().with_private_dataframe(
        "t1", pandas_frame, AddOneRow()
    )
    with pytest.raises(ValueError) as excinfo:
        builder.with_private_dataframe("t2", spark_frame, AddOneRow())
    message = str(excinfo.value)
    assert "'t2'" in message
    assert "pandas backend" in message
    assert builder._backend is PANDAS
    assert set(builder._private_dataframes) == {"t1"}


###############################################################################
# Protected changes.
###############################################################################


@pytest.mark.parametrize(
    "protected_change",
    [AddOneRow(), AddMaxRows(5), AddRowsWithID("id")],
    ids=["AddOneRow", "AddMaxRows", "AddRowsWithID"],
)
def test_supported_protected_changes(
    pandas_frame, pandas_enabled, protected_change: ProtectedChange
):
    """The protected changes the pandas backend can protect are accepted."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder().with_private_dataframe(
        "t", pandas_frame, protected_change
    )
    assert builder._private_dataframes["t"].protected_change == protected_change


def test_add_max_rows_in_max_groups_is_rejected(pandas_frame, pandas_enabled):
    """AddMaxRowsInMaxGroups needs grouped truncation the pandas backend lacks."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder()
    with pytest.raises(NotSupportedByBackend) as excinfo:
        builder.with_private_dataframe(
            "t",
            pandas_frame,
            AddMaxRowsInMaxGroups("A", max_groups=2, max_rows_per_group=3),
        )
    assert excinfo.value.backend == "pandas"
    assert "'t'" in str(excinfo.value)
    assert builder._private_dataframes == {}


def test_add_max_rows_in_max_groups_is_fine_on_spark(spark_frame):
    """The same protected change is untouched on the Spark backend."""
    builder = _DataFrameBuilder().with_private_dataframe(
        "t", spark_frame, AddMaxRowsInMaxGroups("A", max_groups=2, max_rows_per_group=3)
    )
    assert builder._backend is SPARK


###############################################################################
# Public tables.
###############################################################################


def test_public_dataframe_is_rejected_on_a_pandas_builder(
    pandas_frame, spark_frame, pandas_enabled
):
    """A pandas Session has no join-public, so it takes no public tables."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder().with_private_dataframe("t", pandas_frame, AddOneRow())
    with pytest.raises(NotSupportedByBackend) as excinfo:
        builder.with_public_dataframe("pub", spark_frame)
    assert excinfo.value.backend == "pandas"
    assert builder._public_dataframes == {}


def test_pandas_public_dataframe_is_rejected(pandas_frame, pandas_enabled):
    """A pandas frame is never a public table, even on a fresh builder."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder()
    with pytest.raises(NotSupportedByBackend) as excinfo:
        builder.with_public_dataframe("pub", pandas_frame)
    assert excinfo.value.backend == "pandas"
    assert builder._public_dataframes == {}


def test_pandas_private_frame_is_rejected_after_a_public_table(
    pandas_frame, spark_frame, pandas_enabled
):
    """Adding public tables first does not sneak them onto a pandas Session."""
    # pylint: disable=unused-argument
    builder = _DataFrameBuilder().with_public_dataframe("pub", spark_frame)
    with pytest.raises(NotSupportedByBackend) as excinfo:
        builder.with_private_dataframe("t", pandas_frame, AddOneRow())
    assert excinfo.value.backend == "pandas"
    assert builder._private_dataframes == {}


def test_spark_public_dataframe_still_works(spark_frame):
    """The Spark path is unchanged."""
    builder = (
        _DataFrameBuilder()
        .with_private_dataframe("t", spark_frame, AddOneRow())
        .with_public_dataframe("pub", spark_frame)
    )
    assert set(builder._public_dataframes) == {"pub"}


###############################################################################
# Coercion.
###############################################################################


def test_coercion_routes_through_the_pandas_backend(pandas_enabled):
    """A pandas table is coerced by the pandas coercion, not the Spark one."""
    # pylint: disable=unused-argument
    frame = pd.DataFrame({"A": np.array([1, 2], dtype=np.int32)})
    builder = _DataFrameBuilder().with_private_dataframe("t", frame, AddOneRow())
    stored = builder._private_dataframes["t"].dataframe
    assert isinstance(stored, pd.DataFrame)
    # The pandas coercion narrows int32 to the int64 Analytics supports, and
    # always returns a copy -- the privacy boundary between the caller's frame
    # and the Session's.
    assert stored["A"].dtype == np.dtype("int64")
    assert stored is not frame
    pd.testing.assert_frame_equal(stored, PANDAS.coerce_schema_or_fail(frame))


def test_unsupported_pandas_dtype_is_rejected(pandas_enabled):
    """The pandas coercion's rejections reach the caller."""
    # pylint: disable=unused-argument
    frame = pd.DataFrame({"A": [1 + 2j]})
    with pytest.raises(ValueError):
        _DataFrameBuilder().with_private_dataframe("t", frame, AddOneRow())


###############################################################################
# The one thing that needs build().
###############################################################################


def test_end_to_end_pandas_build(pandas_frame, pandas_enabled):
    """A pandas Session can be built end to end."""
    # pylint: disable=unused-argument
    session = (
        Session.Builder()
        .with_privacy_budget(PureDPBudget(1))
        .with_private_dataframe("t", pandas_frame, AddOneRow())
        .build()
    )
    assert session._backend is PANDAS
    assert session.private_sources == ["t"]
