"""Unit tests for :mod:`~tmlt.analytics._coerce_pandas_schema`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
import math
import re
from typing import Any, List

import numpy as np
import pandas as pd
import pytest
from tmlt.core.domains.pandas_domains import (
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)

from tmlt.analytics._coerce_pandas_schema import (
    CANONICAL_PANDAS_DTYPES,
    SUPPORTED_PANDAS_DTYPES,
    TYPE_COERCION_MAP,
    coerce_pandas_schema_or_fail,
    object_column_element_type,
    string_dtypes,
)

DATE = datetime.date(2020, 1, 1)
OTHER_DATE = datetime.date(2020, 1, 2)
TIMESTAMPS = pd.to_datetime(pd.Series(["2020-01-01 12:00:00", "2020-01-02 00:00:00"]))


def assert_values_equal(actual: pd.Series, expected: List[Any]) -> None:
    """Asserts that a Series holds ``expected``, treating all nulls as equal.

    This avoids comparing Series objects directly, which would also compare
    dtypes -- the thing under test here is usually meant to change.
    """
    actual_values = list(actual)
    assert len(actual_values) == len(expected)
    for got, want in zip(actual_values, expected):
        if want is None or (isinstance(want, float) and math.isnan(want)):
            assert pd.isna(got), f"expected a null, got {got!r}"
        else:
            assert got == want


class TestDtypeTables:
    """The dtype tables themselves are part of the contract."""

    def test_supported_is_canonical_plus_coercible(self) -> None:
        """Every supported dtype is either canonical or has a coercion."""
        assert SUPPORTED_PANDAS_DTYPES == CANONICAL_PANDAS_DTYPES | frozenset(
            TYPE_COERCION_MAP
        )

    def test_coercion_targets_are_canonical(self) -> None:
        """Coercion always lands on a dtype a coerced frame may have."""
        assert set(TYPE_COERCION_MAP.values()) <= CANONICAL_PANDAS_DTYPES

    def test_canonical_dtypes_are_never_coerced(self) -> None:
        """Coercion is idempotent because no canonical dtype is a source."""
        assert not CANONICAL_PANDAS_DTYPES & frozenset(TYPE_COERCION_MAP)

    def test_coercion_map_is_as_documented(self) -> None:
        """The coercion table is exactly the one the module docstring lists."""
        expected = {
            np.dtype("int8"): np.dtype("int64"),
            np.dtype("int16"): np.dtype("int64"),
            np.dtype("int32"): np.dtype("int64"),
            np.dtype("uint8"): np.dtype("int64"),
            np.dtype("uint16"): np.dtype("int64"),
            np.dtype("uint32"): np.dtype("int64"),
            pd.Int8Dtype(): pd.Int64Dtype(),
            pd.Int16Dtype(): pd.Int64Dtype(),
            pd.Int32Dtype(): pd.Int64Dtype(),
            np.dtype("float32"): np.dtype("float64"),
            pd.Float32Dtype(): pd.Float64Dtype(),
            **{dtype: np.dtype(object) for dtype in string_dtypes()},
        }
        assert TYPE_COERCION_MAP == expected

    def test_canonical_dtypes_are_as_documented(self) -> None:
        """A coerced frame's columns have one of these dtypes."""
        assert np.dtype("int64") in CANONICAL_PANDAS_DTYPES
        assert pd.Int64Dtype() in CANONICAL_PANDAS_DTYPES
        assert np.dtype("float64") in CANONICAL_PANDAS_DTYPES
        assert pd.Float64Dtype() in CANONICAL_PANDAS_DTYPES
        assert np.dtype(object) in CANONICAL_PANDAS_DTYPES
        assert np.dtype("datetime64[ns]") in CANONICAL_PANDAS_DTYPES

    def test_canonical_dtypes_agree_with_core(self) -> None:
        """Every dtype Core's descriptors call canonical is canonical here."""
        descriptors = [
            PandasIntegerColumnDescriptor(allow_null=False),
            PandasIntegerColumnDescriptor(allow_null=True),
            PandasFloatColumnDescriptor(allow_null=False),
            PandasFloatColumnDescriptor(allow_null=True),
            PandasStringColumnDescriptor(),
            PandasDateColumnDescriptor(),
            PandasTimestampColumnDescriptor(),
        ]
        for descriptor in descriptors:
            assert descriptor.pandas_dtype in CANONICAL_PANDAS_DTYPES

    def test_unsupported_dtypes_are_not_listed(self) -> None:
        """The dtypes the module rejects are genuinely absent from the table."""
        for dtype in (
            np.dtype(bool),
            np.dtype("uint64"),
            np.dtype("float16"),
            np.dtype("complex128"),
            np.dtype("timedelta64[ns]"),
            pd.CategoricalDtype(["a"]),
            pd.BooleanDtype(),
            pd.UInt64Dtype(),
            pd.DatetimeTZDtype(tz="UTC"),
            pd.PeriodDtype("D"),
        ):
            assert dtype not in SUPPORTED_PANDAS_DTYPES


class TestAcceptedDtypes:
    """Every row of the dtype table, in the coercion direction."""

    @pytest.mark.parametrize(
        "series,expected_dtype",
        [
            # Integers: narrower numpy widths widen to numpy int64...
            pytest.param(pd.Series([1, 2], dtype="int8"), np.dtype("int64"), id="int8"),
            pytest.param(
                pd.Series([1, 2], dtype="int16"), np.dtype("int64"), id="int16"
            ),
            pytest.param(
                pd.Series([1, 2], dtype="int32"), np.dtype("int64"), id="int32"
            ),
            pytest.param(
                pd.Series([1, 2], dtype="int64"), np.dtype("int64"), id="int64"
            ),
            pytest.param(
                pd.Series([1, 2], dtype="uint8"), np.dtype("int64"), id="uint8"
            ),
            pytest.param(
                pd.Series([1, 2], dtype="uint16"), np.dtype("int64"), id="uint16"
            ),
            pytest.param(
                pd.Series([1, 2], dtype="uint32"), np.dtype("int64"), id="uint32"
            ),
            # ...and narrower nullable widths widen to the nullable Int64.
            pytest.param(pd.Series([1, 2], dtype="Int8"), pd.Int64Dtype(), id="Int8"),
            pytest.param(pd.Series([1, 2], dtype="Int16"), pd.Int64Dtype(), id="Int16"),
            pytest.param(pd.Series([1, 2], dtype="Int32"), pd.Int64Dtype(), id="Int32"),
            pytest.param(pd.Series([1, 2], dtype="Int64"), pd.Int64Dtype(), id="Int64"),
            # Floats, the same way.
            pytest.param(
                pd.Series([1.5, 2.5], dtype="float32"),
                np.dtype("float64"),
                id="float32",
            ),
            pytest.param(
                pd.Series([1.5, 2.5], dtype="float64"),
                np.dtype("float64"),
                id="float64",
            ),
            pytest.param(
                pd.Series([1.5, 2.5], dtype="Float32"), pd.Float64Dtype(), id="Float32"
            ),
            pytest.param(
                pd.Series([1.5, 2.5], dtype="Float64"), pd.Float64Dtype(), id="Float64"
            ),
            # Strings, dates and timestamps.
            pytest.param(
                pd.Series(["a", "b"], dtype="string"), np.dtype(object), id="string"
            ),
            pytest.param(
                pd.Series(["a", "b"], dtype=object), np.dtype(object), id="object_str"
            ),
            pytest.param(
                pd.Series([DATE, OTHER_DATE], dtype=object),
                np.dtype(object),
                id="object_date",
            ),
            pytest.param(TIMESTAMPS, np.dtype("datetime64[ns]"), id="datetime64_ns"),
        ],
    )
    def test_dtype_is_coerced_as_documented(
        self, series: pd.Series, expected_dtype: Any
    ) -> None:
        """Each supported dtype ends up as the documented coerced dtype."""
        original_values = list(series)
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame({"A": series}))
        assert coerced["A"].dtype == expected_dtype
        assert_values_equal(coerced["A"], original_values)

    @pytest.mark.parametrize("dtype", ["int64", "Int64", "float64", "Float64"])
    def test_canonical_numeric_dtypes_pass_through(self, dtype: str) -> None:
        """A column that is already canonical keeps its dtype exactly."""
        series = pd.Series([1, 2], dtype=dtype)
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame({"A": series}))
        assert coerced["A"].dtype == series.dtype

    def test_coercion_is_idempotent(self) -> None:
        """Coercing an already-coerced frame changes nothing."""
        dataframe = pd.DataFrame(
            {
                "int": pd.Series([1, 2], dtype="int32"),
                "nullable_int": pd.Series([1, None], dtype="Int8"),
                "float": pd.Series([1.5, 2.5], dtype="float32"),
                "string": pd.Series(["a", None], dtype="string"),
                "date": pd.Series([DATE, None], dtype=object),
                "timestamp": TIMESTAMPS,
            }
        )
        once = coerce_pandas_schema_or_fail(dataframe)
        twice = coerce_pandas_schema_or_fail(once)
        pd.testing.assert_frame_equal(once, twice)

    def test_coerced_frame_is_in_a_core_domain(self) -> None:
        """A coerced frame validates against the matching Core domain."""
        dataframe = pd.DataFrame(
            {
                "int": pd.Series([1, 2], dtype="int32"),
                "nullable_int": pd.Series([1, None], dtype="Int8"),
                "float": pd.Series([1.5, 2.5], dtype="float32"),
                "string": pd.Series(["a", None], dtype="string"),
                "date": pd.Series([DATE, None], dtype=object),
                "timestamp": TIMESTAMPS,
            }
        )
        coerced = coerce_pandas_schema_or_fail(dataframe)
        domain = PandasTableDomain(
            {
                "int": PandasIntegerColumnDescriptor(),
                "nullable_int": PandasIntegerColumnDescriptor(allow_null=True),
                "float": PandasFloatColumnDescriptor(allow_nan=True, allow_inf=True),
                "string": PandasStringColumnDescriptor(allow_null=True),
                "date": PandasDateColumnDescriptor(allow_null=True),
                "timestamp": PandasTimestampColumnDescriptor(allow_null=True),
            }
        )
        domain.validate(coerced)

    def test_empty_dataframe_is_accepted(self) -> None:
        """A frame with no rows is coerced on its dtypes alone."""
        dataframe = pd.DataFrame(
            {"A": pd.Series([], dtype="int32"), "B": pd.Series([], dtype=object)}
        )
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"].dtype == np.dtype("int64")
        assert coerced["B"].dtype == np.dtype(object)
        assert len(coerced) == 0

    def test_no_columns_is_accepted(self) -> None:
        """A frame with no columns at all is coerced to an equal frame."""
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame())
        assert list(coerced.columns) == []

    def test_column_order_is_preserved(self) -> None:
        """Coercion never reorders or renames columns."""
        dataframe = pd.DataFrame(
            {
                "z": pd.Series([1], dtype="int32"),
                "a": pd.Series(["x"], dtype="string"),
                "m": pd.Series([1.0], dtype="float32"),
            }
        )
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert list(coerced.columns) == ["z", "a", "m"]


class TestNullNormalization:
    """``object`` columns come out with ``None`` as their only null."""

    def test_string_dtype_na_becomes_none(self) -> None:
        """pd.NA from a ``string`` column becomes None, not pd.NA."""
        dataframe = pd.DataFrame({"A": pd.Series(["a", None], dtype="string")})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"].dtype == np.dtype(object)
        assert list(coerced["A"]) == ["a", None]
        assert coerced["A"][1] is None

    @pytest.mark.parametrize("storage", ["python", "pyarrow"])
    def test_string_dtype_storages_are_both_accepted(self, storage: str) -> None:
        """Both backings of the ``string`` dtype coerce the same way."""
        try:
            dtype = pd.StringDtype(storage)
        except (ImportError, ValueError):  # pragma: no cover - depends on pyarrow
            pytest.skip(f"pandas string storage {storage!r} is unavailable")
        dataframe = pd.DataFrame({"A": pd.Series(["a", None], dtype=dtype)})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"].dtype == np.dtype(object)
        assert list(coerced["A"]) == ["a", None]

    def test_object_na_becomes_none(self) -> None:
        """A pd.NA sitting in an object column is normalized too."""
        dataframe = pd.DataFrame({"A": pd.Series(["a", pd.NA], dtype=object)})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"][1] is None

    def test_object_nan_becomes_none(self) -> None:
        """A float NaN used as a missing string is normalized too."""
        dataframe = pd.DataFrame({"A": pd.Series(["a", float("nan")], dtype=object)})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"][1] is None

    def test_date_column_nulls_are_normalized(self) -> None:
        """A date column's nulls get the same treatment as a string column's."""
        dataframe = pd.DataFrame({"A": pd.Series([DATE, pd.NA], dtype=object)})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"].dtype == np.dtype(object)
        assert coerced["A"][0] == DATE
        assert coerced["A"][1] is None

    def test_nullable_int_keeps_its_null(self) -> None:
        """Widening a nullable integer column keeps the null a null."""
        dataframe = pd.DataFrame({"A": pd.Series([1, None], dtype="Int8")})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"].dtype == pd.Int64Dtype()
        assert coerced["A"][0] == 1
        assert pd.isna(coerced["A"][1])

    def test_float_nan_is_not_turned_into_a_null(self) -> None:
        """A NaN in a numpy float column is a NaN, and stays one."""
        dataframe = pd.DataFrame({"A": pd.Series([1.0, float("nan")], dtype="float32")})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced["A"].dtype == np.dtype("float64")
        assert math.isnan(coerced["A"][1])

    def test_float_infinities_are_preserved(self) -> None:
        """Widening a float column does not disturb its infinities."""
        dataframe = pd.DataFrame(
            {"A": pd.Series([float("inf"), float("-inf")], dtype="float32")}
        )
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert list(coerced["A"]) == [float("inf"), float("-inf")]

    def test_timestamp_nat_is_preserved(self) -> None:
        """NaT is the null of a datetime64 column and is left alone."""
        dataframe = pd.DataFrame({"A": pd.to_datetime(pd.Series(["2020-01-01", None]))})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert pd.isna(coerced["A"][1])


class TestObjectColumnContents:
    """``object`` columns are checked against what they hold."""

    @pytest.mark.parametrize(
        "values,expected_type",
        [
            pytest.param(["a", "b"], str, id="strings"),
            pytest.param(["a", None], str, id="strings_and_none"),
            pytest.param([np.str_("a")], str, id="numpy_strings"),
            pytest.param([DATE, OTHER_DATE], datetime.date, id="dates"),
            pytest.param([DATE, None], datetime.date, id="dates_and_none"),
            pytest.param([None, None], str, id="all_null_is_a_string_column"),
            pytest.param([], str, id="empty_is_a_string_column"),
        ],
    )
    def test_valid_object_columns(self, values: List[Any], expected_type: type) -> None:
        """An object column of one supported type is accepted and classified."""
        column = pd.Series(values, dtype=object)
        assert object_column_element_type(column, "A") is expected_type
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame({"A": column}))
        assert coerced["A"].dtype == np.dtype(object)

    @pytest.mark.parametrize(
        "values,expected_offenders",
        [
            pytest.param([b"bytes"], ["builtins.bytes"], id="bytes"),
            pytest.param(["a", 1], ["builtins.int"], id="mixed_str_and_int"),
            pytest.param([1, 2], ["builtins.int"], id="ints_in_object_column"),
            pytest.param([["a"]], ["builtins.list"], id="lists"),
            pytest.param(
                ["a", DATE], ["builtins.str", "datetime.date"], id="mixed_str_and_date"
            ),
            pytest.param(
                [datetime.datetime(2020, 1, 1)],
                ["datetime.datetime"],
                id="datetimes",
            ),
        ],
    )
    def test_invalid_object_columns(
        self, values: List[Any], expected_offenders: List[str]
    ) -> None:
        """An object column of anything else is rejected, naming the offenders."""
        column = pd.Series(values, dtype=object)
        with pytest.raises(ValueError) as excinfo:
            object_column_element_type(column, "A")
        message = str(excinfo.value)
        assert "column 'A'" in message
        for offender in expected_offenders:
            assert offender in message

        with pytest.raises(ValueError, match="column 'A'"):
            coerce_pandas_schema_or_fail(pd.DataFrame({"A": column}))

    def test_datetime_in_a_date_column_suggests_the_conversion(self) -> None:
        """The datetime.datetime case is common enough to get its own advice."""
        column = pd.Series([DATE, datetime.datetime(2020, 1, 1)], dtype=object)
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(pd.DataFrame({"A": column}))
        message = str(excinfo.value)
        assert "datetime.datetime.date" in message
        assert "pd.to_datetime" in message

    def test_every_bad_object_column_is_named(self) -> None:
        """Two bad columns produce one error mentioning both."""
        dataframe = pd.DataFrame(
            {
                "good": pd.Series(["a"], dtype=object),
                "bad_one": pd.Series([1], dtype=object),
                "bad_two": pd.Series([b"x"], dtype=object),
            }
        )
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(dataframe)
        message = str(excinfo.value)
        assert "column 'bad_one'" in message
        assert "column 'bad_two'" in message
        assert "column 'good'" not in message


class TestRejectedDtypes:
    """Unsupported dtypes are rejected with an actionable error."""

    @pytest.mark.parametrize(
        "series",
        [
            pytest.param(pd.Series([True, False], dtype=bool), id="bool"),
            pytest.param(pd.Series([True, None], dtype="boolean"), id="boolean_ext"),
            pytest.param(pd.Series(["a", "b"], dtype="category"), id="category"),
            pytest.param(
                pd.Series(pd.to_timedelta([1, 2], unit="s")), id="timedelta64"
            ),
            pytest.param(
                pd.Series(pd.period_range("2020-01-01", periods=2, freq="D")),
                id="period",
            ),
            pytest.param(pd.Series(pd.interval_range(0, 2)), id="interval"),
            pytest.param(pd.Series([1 + 2j, 3 + 4j]), id="complex"),
            pytest.param(pd.Series([1, 2], dtype="uint64"), id="uint64"),
            pytest.param(pd.Series([1, 2], dtype="UInt64"), id="UInt64_ext"),
            pytest.param(pd.Series([1, 2], dtype="UInt32"), id="UInt32_ext"),
            pytest.param(pd.Series([1.0, 2.0], dtype="float16"), id="float16"),
            pytest.param(
                pd.to_datetime(pd.Series(["2020-01-01"])).dt.tz_localize("UTC"),
                id="tz_aware",
            ),
        ],
    )
    def test_unsupported_dtype_is_rejected(self, series: pd.Series) -> None:
        """The error names the column, its dtype, and the supported dtypes."""
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(pd.DataFrame({"offending": series}))
        message = str(excinfo.value)
        assert "offending" in message
        assert str(series.dtype) in message
        assert "int64" in message

    def test_timezone_aware_error_says_how_to_convert(self) -> None:
        """A tz-aware column is common, so its error carries the incantation."""
        series = pd.to_datetime(pd.Series(["2020-01-01"])).dt.tz_localize(
            "America/New_York"
        )
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(pd.DataFrame({"when": series}))
        message = str(excinfo.value)
        assert "when" in message
        assert "tz_convert('UTC')" in message
        assert "tz_localize(None)" in message

    def test_boolean_error_says_how_to_convert(self) -> None:
        """A boolean column names itself and suggests a cast."""
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(
                pd.DataFrame({"flag": pd.Series([True, False])})
            )
        message = str(excinfo.value)
        assert "'flag'" in message
        assert "astype('int64')" in message

    def test_categorical_error_says_how_to_convert(self) -> None:
        """A categorical column names itself and suggests a cast."""
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(
                pd.DataFrame({"kind": pd.Series(["a", "b"], dtype="category")})
            )
        message = str(excinfo.value)
        assert "'kind'" in message
        assert "astype(object)" in message

    def test_uint64_error_explains_the_overflow(self) -> None:
        """uint64 is rejected rather than silently wrapped."""
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(
                pd.DataFrame({"big": pd.Series([1], dtype="uint64")})
            )
        message = str(excinfo.value)
        assert "'big'" in message
        assert "int64" in message

    def test_every_unsupported_column_is_listed(self) -> None:
        """One error lists every offending column, as the Spark version does."""
        dataframe = pd.DataFrame(
            {
                "ok": pd.Series([1, 2], dtype="int64"),
                "flag": pd.Series([True, False]),
                "kind": pd.Series(["a", "b"], dtype="category"),
            }
        )
        with pytest.raises(ValueError) as excinfo:
            coerce_pandas_schema_or_fail(dataframe)
        message = str(excinfo.value)
        assert "flag" in message
        assert "kind" in message
        assert "'ok'" not in message

    def test_dtype_errors_are_deterministic(self) -> None:
        """The list of supported dtypes is sorted, so messages do not vary."""
        dataframe = pd.DataFrame({"flag": pd.Series([True])})
        messages = set()
        for _ in range(3):
            with pytest.raises(ValueError) as excinfo:
                coerce_pandas_schema_or_fail(dataframe)
            messages.add(str(excinfo.value))
        assert len(messages) == 1


class TestColumnNames:
    """Column names pandas allows but Analytics does not."""

    def test_empty_column_name_is_rejected(self) -> None:
        """Mirrors the Spark coercion's rejection of the empty name."""
        dataframe = pd.DataFrame({"": [1, 2]})
        with pytest.raises(
            ValueError, match=re.escape('contains a column named "" (the empty string)')
        ):
            coerce_pandas_schema_or_fail(dataframe)

    def test_non_string_column_name_is_rejected(self) -> None:
        """A pandas name may be any hashable; Analytics names are strings."""
        dataframe = pd.DataFrame({0: [1, 2]})
        with pytest.raises(ValueError, match="column names that are not strings"):
            coerce_pandas_schema_or_fail(dataframe)

    def test_duplicate_column_names_are_rejected(self) -> None:
        """A duplicated name has no single dtype, so it cannot be described."""
        dataframe = pd.DataFrame([[1, 2]], columns=["A", "A"])
        with pytest.raises(ValueError, match="duplicate column names"):
            coerce_pandas_schema_or_fail(dataframe)


class TestCopySemantics:
    """The returned frame is a copy: the copy-on-ingest privacy boundary."""

    @staticmethod
    def _mixed_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "int": pd.Series([1, 2], dtype="int64"),
                "narrow_int": pd.Series([1, 2], dtype="int32"),
                "float": pd.Series([1.5, 2.5], dtype="float64"),
                "string": pd.Series(["a", "b"], dtype=object),
                "date": pd.Series([DATE, OTHER_DATE], dtype=object),
                "timestamp": TIMESTAMPS,
            }
        )

    def test_result_is_a_new_object(self) -> None:
        """Even a frame that needs no coercion at all is copied."""
        dataframe = pd.DataFrame({"A": pd.Series([1, 2], dtype="int64")})
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert coerced is not dataframe

    @pytest.mark.parametrize(
        "column", ["int", "narrow_int", "float", "string", "date", "timestamp"]
    )
    def test_result_shares_no_buffers(self, column: str) -> None:
        """No column of the result shares memory with the caller's frame."""
        dataframe = self._mixed_frame()
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert not np.shares_memory(
            dataframe[column].to_numpy(), coerced[column].to_numpy()
        )

    def test_writing_to_the_result_does_not_change_the_input(self) -> None:
        """The Session cannot disturb the caller's frame."""
        dataframe = self._mixed_frame()
        coerced = coerce_pandas_schema_or_fail(dataframe)
        coerced.loc[0, "int"] = 999
        coerced.loc[0, "string"] = "changed"
        assert dataframe.loc[0, "int"] == 1
        assert dataframe.loc[0, "string"] == "a"

    def test_writing_to_the_input_does_not_change_the_result(self) -> None:
        """The caller cannot disturb data the Session has already ingested."""
        dataframe = self._mixed_frame()
        coerced = coerce_pandas_schema_or_fail(dataframe)
        dataframe.loc[0, "int"] = 999
        dataframe.loc[0, "string"] = "changed"
        assert coerced.loc[0, "int"] == 1
        assert coerced.loc[0, "string"] == "a"

    def test_input_dtypes_are_unchanged(self) -> None:
        """Coercion happens on the copy, never on the caller's frame."""
        dataframe = self._mixed_frame()
        before = dict(dataframe.dtypes)
        coerce_pandas_schema_or_fail(dataframe)
        assert dict(dataframe.dtypes) == before

    def test_input_object_column_keeps_its_nulls(self) -> None:
        """Null normalization does not reach back into the caller's frame."""
        dataframe = pd.DataFrame({"A": pd.Series(["a", pd.NA], dtype=object)})
        coerce_pandas_schema_or_fail(dataframe)
        assert dataframe["A"][1] is pd.NA

    def test_index_is_preserved(self) -> None:
        """Coercion does not reindex; Core's domains ignore the index anyway."""
        dataframe = pd.DataFrame(
            {"A": pd.Series([1, 2], dtype="int32")}, index=["x", "y"]
        )
        coerced = coerce_pandas_schema_or_fail(dataframe)
        assert list(coerced.index) == ["x", "y"]
