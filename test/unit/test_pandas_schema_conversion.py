"""Unit tests for the pandas conversions in :mod:`~tmlt.analytics._schema`.

The load-bearing test here is :class:`TestRoundTripIdentity`: an Analytics
schema, turned into pandas column descriptors and back, must come out
unchanged. Every later claim about the pandas backend describing the same table
as the Spark one rests on it.

:class:`TestSparkInferenceComparison` pins the one place where the two backends
deliberately disagree. It needs a Spark session, and is the only thing in this
module that does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import types as spark_types
from tmlt.core.domains.pandas_domains import (
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)

from tmlt.analytics._coerce_pandas_schema import coerce_pandas_schema_or_fail
from tmlt.analytics._schema import (
    _SPARK_TO_ANALYTICS,
    ColumnDescriptor,
    ColumnType,
    Schema,
    analytics_to_pandas_columns_descriptor,
    analytics_to_pandas_dtypes,
    analytics_to_spark_columns_descriptor,
    pandas_dataframe_domain_to_analytics_columns,
    pandas_dtypes_to_analytics_columns,
    spark_schema_to_analytics_columns,
)

DATE = datetime.date(2020, 1, 1)
OTHER_DATE = datetime.date(2020, 1, 2)

ALL_COLUMN_TYPES = [
    ColumnType.INTEGER,
    ColumnType.DECIMAL,
    ColumnType.VARCHAR,
    ColumnType.DATE,
    ColumnType.TIMESTAMP,
]


def _decimal(allow_null: bool, allow_nan: bool, allow_inf: bool) -> ColumnDescriptor:
    return ColumnDescriptor(
        ColumnType.DECIMAL,
        allow_null=allow_null,
        allow_nan=allow_nan,
        allow_inf=allow_inf,
    )


ROUND_TRIP_DESCRIPTORS: List[ColumnDescriptor] = [
    *(
        ColumnDescriptor(column_type, allow_null=allow_null)
        for column_type in ALL_COLUMN_TYPES
        if column_type != ColumnType.DECIMAL
        for allow_null in (False, True)
    ),
    *(
        _decimal(allow_null, allow_nan, allow_inf)
        for allow_null, allow_nan, allow_inf in itertools.product(
            (False, True), repeat=3
        )
    ),
]
"""Every ColumnDescriptor the round-trip is an identity on.

Four types with two nullabilities each, plus DECIMAL with all eight
combinations of its three flags: sixteen in all. See
:class:`TestRoundTripIdentity` for the descriptors this deliberately leaves
out.
"""


def _descriptor_id(descriptor: ColumnDescriptor) -> str:
    return (
        f"{descriptor.column_type.name}"
        f"-null{int(descriptor.allow_null)}"
        f"-nan{int(descriptor.allow_nan)}"
        f"-inf{int(descriptor.allow_inf)}"
    )


class TestAnalyticsToPandasColumnsDescriptor:
    """Analytics schema to pandas column descriptors."""

    def test_all_types(self) -> None:
        """Each Analytics type maps to its pandas descriptor."""
        schema = Schema(
            {
                "1": "INTEGER",
                "2": "DECIMAL",
                "3": "VARCHAR",
                "4": "DATE",
                "5": "TIMESTAMP",
            }
        )
        assert analytics_to_pandas_columns_descriptor(schema) == {
            "1": PandasIntegerColumnDescriptor(allow_null=False),
            "2": PandasFloatColumnDescriptor(
                allow_nan=False, allow_inf=False, allow_null=False
            ),
            "3": PandasStringColumnDescriptor(allow_null=False),
            "4": PandasDateColumnDescriptor(allow_null=False),
            "5": PandasTimestampColumnDescriptor(allow_null=False),
        }

    def test_all_types_with_null(self) -> None:
        """allow_null is carried onto every descriptor."""
        schema = Schema(
            {
                name: ColumnDescriptor(column_type, allow_null=True)
                for name, column_type in zip("12345", ALL_COLUMN_TYPES)
            }
        )
        descriptors = analytics_to_pandas_columns_descriptor(schema)
        assert all(descriptor.allow_null for descriptor in descriptors.values())

    @pytest.mark.parametrize(
        "allow_nan,allow_inf", list(itertools.product((False, True), repeat=2))
    )
    def test_decimal_flags(self, allow_nan: bool, allow_inf: bool) -> None:
        """A DECIMAL column's NaN and infinity flags are carried across."""
        schema = Schema(
            {"A": _decimal(allow_null=False, allow_nan=allow_nan, allow_inf=allow_inf)}
        )
        assert analytics_to_pandas_columns_descriptor(schema) == {
            "A": PandasFloatColumnDescriptor(
                allow_nan=allow_nan, allow_inf=allow_inf, allow_null=False
            )
        }

    @pytest.mark.parametrize("descriptor", ROUND_TRIP_DESCRIPTORS, ids=_descriptor_id)
    def test_agrees_with_the_spark_descriptor(
        self, descriptor: ColumnDescriptor
    ) -> None:
        """The two backends describe the same values.

        Core's ``to_spark_descriptor`` is the bridge: the pandas descriptor
        Analytics builds for a column must convert to exactly the Spark
        descriptor it builds for the same column.
        """
        schema = Schema({"A": descriptor})
        pandas_descriptor = analytics_to_pandas_columns_descriptor(schema)["A"]
        spark_descriptor = analytics_to_spark_columns_descriptor(schema)["A"]
        assert pandas_descriptor.to_spark_descriptor() == spark_descriptor

    def test_empty_schema(self) -> None:
        """An empty schema converts to an empty descriptor mapping."""
        assert analytics_to_pandas_columns_descriptor(Schema({})) == {}


class TestAnalyticsToPandasDtypes:
    """Analytics schema to the canonical pandas dtype of each column."""

    def test_non_null_columns(self) -> None:
        """A non-nullable column uses the numpy dtypes."""
        schema = Schema(
            {
                "1": "INTEGER",
                "2": "DECIMAL",
                "3": "VARCHAR",
                "4": "DATE",
                "5": "TIMESTAMP",
            }
        )
        assert analytics_to_pandas_dtypes(schema) == {
            "1": np.dtype("int64"),
            "2": np.dtype("float64"),
            "3": np.dtype(object),
            "4": np.dtype(object),
            "5": np.dtype("datetime64[ns]"),
        }

    def test_nullable_columns(self) -> None:
        """Only integers and decimals change dtype when they allow nulls."""
        schema = Schema(
            {
                name: ColumnDescriptor(column_type, allow_null=True)
                for name, column_type in zip("12345", ALL_COLUMN_TYPES)
            }
        )
        assert analytics_to_pandas_dtypes(schema) == {
            "1": pd.Int64Dtype(),
            "2": pd.Float64Dtype(),
            "3": np.dtype(object),
            "4": np.dtype(object),
            "5": np.dtype("datetime64[ns]"),
        }

    @pytest.mark.parametrize("descriptor", ROUND_TRIP_DESCRIPTORS, ids=_descriptor_id)
    def test_agrees_with_the_core_domain(self, descriptor: ColumnDescriptor) -> None:
        """The dtype table matches what Core calls the canonical dtype."""
        schema = Schema({"A": descriptor})
        domain = PandasTableDomain(analytics_to_pandas_columns_descriptor(schema))
        assert analytics_to_pandas_dtypes(schema) == domain.pandas_dtypes


class TestPandasDtypesToAnalyticsColumns:
    """DataFrame dtypes to Analytics columns, the inference direction."""

    @pytest.mark.parametrize(
        "series,expected",
        [
            # Integers. A numpy column cannot hold a null, an extension column
            # can, and that is the whole of the allow_null decision.
            pytest.param(
                pd.Series([1, 2], dtype="int8"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                id="int8",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="int16"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                id="int16",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="int32"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                id="int32",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="int64"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                id="int64",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="uint8"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                id="uint8",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="uint16"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                id="uint16",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="uint32"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=False),
                id="uint32",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="Int8"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                id="Int8",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="Int16"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                id="Int16",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="Int32"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                id="Int32",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="Int64"),
                ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
                id="Int64",
            ),
            # Decimals. allow_nan and allow_inf are always True.
            pytest.param(
                pd.Series([1.5], dtype="float32"),
                ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=False,
                    allow_nan=True,
                    allow_inf=True,
                ),
                id="float32",
            ),
            pytest.param(
                pd.Series([1.5], dtype="float64"),
                ColumnDescriptor(
                    ColumnType.DECIMAL,
                    allow_null=False,
                    allow_nan=True,
                    allow_inf=True,
                ),
                id="float64",
            ),
            pytest.param(
                pd.Series([1.5], dtype="Float32"),
                ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
                ),
                id="Float32",
            ),
            pytest.param(
                pd.Series([1.5], dtype="Float64"),
                ColumnDescriptor(
                    ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=True
                ),
                id="Float64",
            ),
            # Strings, dates and timestamps all hold a null in any case.
            pytest.param(
                pd.Series(["a"], dtype="string"),
                ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                id="string",
            ),
            pytest.param(
                pd.Series(["a", "b"], dtype=object),
                ColumnDescriptor(ColumnType.VARCHAR, allow_null=True),
                id="object_str",
            ),
            pytest.param(
                pd.Series([DATE, OTHER_DATE], dtype=object),
                ColumnDescriptor(ColumnType.DATE, allow_null=True),
                id="object_date",
            ),
            pytest.param(
                pd.to_datetime(pd.Series(["2020-01-01"])),
                ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
                id="datetime64_ns",
            ),
        ],
    )
    def test_dtype_matrix(self, series: pd.Series, expected: ColumnDescriptor) -> None:
        """Each supported dtype infers the documented ColumnDescriptor."""
        dataframe = pd.DataFrame({"A": series})
        assert pandas_dtypes_to_analytics_columns(dataframe) == {"A": expected}

    def test_inference_does_not_depend_on_the_values(self) -> None:
        """Two frames with the same dtypes get the same schema.

        A schema that changed with the data would leak whether the data
        contains a null, which is the reason allow_null is read off the dtype.
        """
        without_nulls = pd.DataFrame(
            {
                "int": pd.Series([1, 2], dtype="Int64"),
                "float": pd.Series([1.5, 2.5], dtype="float64"),
                "string": pd.Series(["a", "b"], dtype=object),
            }
        )
        with_nulls = pd.DataFrame(
            {
                "int": pd.Series([1, None], dtype="Int64"),
                "float": pd.Series([float("nan"), float("inf")], dtype="float64"),
                "string": pd.Series(["a", None], dtype=object),
            }
        )
        assert pandas_dtypes_to_analytics_columns(
            without_nulls
        ) == pandas_dtypes_to_analytics_columns(with_nulls)

    def test_empty_object_column_is_a_string_column(self) -> None:
        """With no values to inspect, an object column is taken to be VARCHAR."""
        dataframe = pd.DataFrame({"A": pd.Series([None, None], dtype=object)})
        assert pandas_dtypes_to_analytics_columns(dataframe) == {
            "A": ColumnDescriptor(ColumnType.VARCHAR, allow_null=True)
        }

    def test_unsupported_dtype_is_rejected(self) -> None:
        """An uncoerced frame with an unsupported dtype raises."""
        dataframe = pd.DataFrame({"A": pd.Series([True, False])})
        with pytest.raises(ValueError, match="Unsupported pandas dtype"):
            pandas_dtypes_to_analytics_columns(dataframe)

    def test_invalid_object_column_is_rejected(self) -> None:
        """An object column of something else raises, naming the column."""
        dataframe = pd.DataFrame({"A": pd.Series([1, 2], dtype=object)})
        with pytest.raises(ValueError, match="column 'A'"):
            pandas_dtypes_to_analytics_columns(dataframe)

    def test_column_order_is_preserved(self) -> None:
        """The resulting mapping is in the frame's column order."""
        dataframe = pd.DataFrame(
            {"z": [1], "a": [1.0], "m": pd.Series(["x"], dtype=object)}
        )
        assert list(pandas_dtypes_to_analytics_columns(dataframe)) == ["z", "a", "m"]


class TestPandasDataFrameDomainToAnalyticsColumns:
    """Pandas table domain to Analytics columns."""

    def test_all_descriptors(self) -> None:
        """Each pandas descriptor maps back to its Analytics type."""
        domain = PandasTableDomain(
            {
                "1": PandasIntegerColumnDescriptor(allow_null=True),
                "2": PandasFloatColumnDescriptor(
                    allow_nan=True, allow_inf=False, allow_null=True
                ),
                "3": PandasStringColumnDescriptor(allow_null=False),
                "4": PandasDateColumnDescriptor(allow_null=False),
                "5": PandasTimestampColumnDescriptor(allow_null=True),
            }
        )
        assert pandas_dataframe_domain_to_analytics_columns(domain) == {
            "1": ColumnDescriptor(ColumnType.INTEGER, allow_null=True),
            "2": ColumnDescriptor(
                ColumnType.DECIMAL, allow_null=True, allow_nan=True, allow_inf=False
            ),
            "3": ColumnDescriptor(ColumnType.VARCHAR, allow_null=False),
            "4": ColumnDescriptor(ColumnType.DATE, allow_null=False),
            "5": ColumnDescriptor(ColumnType.TIMESTAMP, allow_null=True),
        }

    @pytest.mark.parametrize("size", [32, 64])
    def test_descriptor_size_is_dropped(self, size: int) -> None:
        """Analytics has one INTEGER and one DECIMAL type, whatever the width."""
        domain = PandasTableDomain(
            {
                "i": PandasIntegerColumnDescriptor(size=size),
                "f": PandasFloatColumnDescriptor(size=size),
            }
        )
        columns = pandas_dataframe_domain_to_analytics_columns(domain)
        assert columns["i"].column_type == ColumnType.INTEGER
        assert columns["f"].column_type == ColumnType.DECIMAL

    def test_empty_domain(self) -> None:
        """An empty domain converts to an empty column mapping."""
        assert pandas_dataframe_domain_to_analytics_columns(PandasTableDomain({})) == {}


class TestRoundTripIdentity:
    """Schema to pandas descriptors and back is an identity.

    This is the invariant the pandas backend's ``validate_transformation`` will
    rest on: a domain built from a schema describes that schema and no other,
    so a transformation's output domain can be checked against the schema
    Analytics expects.

    The identity holds on :data:`ROUND_TRIP_DESCRIPTORS`, which is every
    descriptor Analytics itself produces. It does not hold on a ColumnDescriptor
    that sets ``allow_nan`` or ``allow_inf`` on a non-DECIMAL column, since no
    pandas descriptor has anywhere to record those; that region is tested below
    to pin what happens instead. Separately, the descriptors *inferred from a
    DataFrame* are a strict subset of the ones that round-trip -- see
    :meth:`test_descriptors_reachable_from_a_frame_round_trip`.
    """

    @pytest.mark.parametrize("descriptor", ROUND_TRIP_DESCRIPTORS, ids=_descriptor_id)
    def test_round_trip_is_an_identity(self, descriptor: ColumnDescriptor) -> None:
        """Every descriptor Analytics produces survives the round trip."""
        schema = Schema({"A": descriptor})
        domain = PandasTableDomain(analytics_to_pandas_columns_descriptor(schema))
        assert pandas_dataframe_domain_to_analytics_columns(domain) == (
            schema.column_descs
        )

    def test_round_trip_of_a_whole_schema(self) -> None:
        """All sixteen descriptors at once, in one schema."""
        schema = Schema(
            {
                f"col{index}": descriptor
                for index, descriptor in enumerate(ROUND_TRIP_DESCRIPTORS)
            }
        )
        domain = PandasTableDomain(analytics_to_pandas_columns_descriptor(schema))
        assert pandas_dataframe_domain_to_analytics_columns(domain) == (
            schema.column_descs
        )

    def test_round_trip_preserves_column_order(self) -> None:
        """The columns come back in the order they went in."""
        schema = Schema({"z": "INTEGER", "a": "DECIMAL", "m": "VARCHAR"})
        domain = PandasTableDomain(analytics_to_pandas_columns_descriptor(schema))
        assert list(pandas_dataframe_domain_to_analytics_columns(domain)) == [
            "z",
            "a",
            "m",
        ]

    @pytest.mark.parametrize(
        "column_type",
        [
            column_type
            for column_type in ALL_COLUMN_TYPES
            if column_type != ColumnType.DECIMAL
        ],
    )
    @pytest.mark.parametrize(
        "allow_nan,allow_inf", [(True, False), (False, True), (True, True)]
    )
    def test_outside_the_domain_of_validity(
        self, column_type: ColumnType, allow_nan: bool, allow_inf: bool
    ) -> None:
        """A non-DECIMAL column's NaN and infinity flags are dropped.

        ColumnDescriptor lets these be set on any column type, but they only
        describe floating-point values, and only PandasFloatColumnDescriptor has
        anywhere to put them. The round trip returns them as False rather than
        failing -- exactly as the Spark family does -- so this documents the
        boundary of the identity above rather than reporting a defect.
        """
        descriptor = ColumnDescriptor(
            column_type, allow_null=False, allow_nan=allow_nan, allow_inf=allow_inf
        )
        schema = Schema({"A": descriptor})
        domain = PandasTableDomain(analytics_to_pandas_columns_descriptor(schema))
        result = pandas_dataframe_domain_to_analytics_columns(domain)["A"]
        assert result != descriptor
        assert result == ColumnDescriptor(column_type, allow_null=False)

    @pytest.mark.parametrize(
        "series",
        [
            pytest.param(pd.Series([1, 2], dtype="int64"), id="int64"),
            pytest.param(pd.Series([1, None], dtype="Int64"), id="Int64"),
            pytest.param(pd.Series([1.5], dtype="float64"), id="float64"),
            pytest.param(pd.Series([1.5, None], dtype="Float64"), id="Float64"),
            pytest.param(pd.Series(["a", None], dtype=object), id="object_str"),
            pytest.param(pd.Series([DATE, None], dtype=object), id="object_date"),
            pytest.param(
                pd.to_datetime(pd.Series(["2020-01-01", None])), id="datetime64_ns"
            ),
        ],
    )
    def test_descriptors_reachable_from_a_frame_round_trip(
        self, series: pd.Series
    ) -> None:
        """The whole ingest path is consistent, end to end.

        Coerce a frame, infer its schema, build the domain that schema
        describes: the domain must both accept the frame and convert back to
        the same schema.
        """
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame({"A": series}))
        schema = Schema(pandas_dtypes_to_analytics_columns(coerced))
        domain = PandasTableDomain(analytics_to_pandas_columns_descriptor(schema))
        domain.validate(coerced)
        assert pandas_dataframe_domain_to_analytics_columns(domain) == (
            schema.column_descs
        )

    def test_the_reachable_descriptors_are_a_strict_subset(self) -> None:
        """Not every round-trippable descriptor can be inferred from a frame.

        Inference always reports allow_nan and allow_inf as True on a DECIMAL
        column, and object and datetime64 columns always allow nulls, so those
        descriptors are unreachable from a DataFrame even though the round trip
        is an identity on them.
        """
        frames = [
            pd.DataFrame({"A": pd.Series([1], dtype=dtype)})
            for dtype in ("int64", "Int64", "float64", "Float64")
        ] + [
            pd.DataFrame({"A": pd.Series(["a"], dtype=object)}),
            pd.DataFrame({"A": pd.Series([DATE], dtype=object)}),
            pd.DataFrame({"A": pd.to_datetime(pd.Series(["2020-01-01"]))}),
        ]
        reachable = {pandas_dtypes_to_analytics_columns(frame)["A"] for frame in frames}
        assert reachable < set(ROUND_TRIP_DESCRIPTORS)
        # A DECIMAL column that forbids NaNs is round-trippable but never
        # inferred, and a VARCHAR column that forbids nulls likewise.
        assert (
            _decimal(allow_null=False, allow_nan=False, allow_inf=False)
            in set(ROUND_TRIP_DESCRIPTORS) - reachable
        )
        assert (
            ColumnDescriptor(ColumnType.VARCHAR, allow_null=False)
            in set(ROUND_TRIP_DESCRIPTORS) - reachable
        )


SPARK_COMPARISON_CASES = [
    pytest.param(pd.Series([1, 2], dtype="int8"), False, id="int8"),
    pytest.param(pd.Series([1, 2], dtype="int16"), False, id="int16"),
    pytest.param(pd.Series([1, 2], dtype="int32"), False, id="int32"),
    pytest.param(pd.Series([1, 2], dtype="int64"), False, id="int64"),
    pytest.param(pd.Series([1, 2], dtype="uint8"), False, id="uint8"),
    pytest.param(pd.Series([1, 2], dtype="uint16"), False, id="uint16"),
    pytest.param(pd.Series([1, 2], dtype="uint32"), False, id="uint32"),
    pytest.param(pd.Series([1, None], dtype="Int8"), True, id="Int8"),
    pytest.param(pd.Series([1, None], dtype="Int16"), True, id="Int16"),
    pytest.param(pd.Series([1, None], dtype="Int32"), True, id="Int32"),
    pytest.param(pd.Series([1, None], dtype="Int64"), True, id="Int64"),
    pytest.param(pd.Series([1.5, 2.5], dtype="float32"), False, id="float32"),
    pytest.param(pd.Series([1.5, 2.5], dtype="float64"), False, id="float64"),
    pytest.param(pd.Series([1.5, None], dtype="Float32"), True, id="Float32"),
    pytest.param(pd.Series([1.5, None], dtype="Float64"), True, id="Float64"),
    pytest.param(pd.Series(["a", None], dtype="string"), True, id="string"),
    pytest.param(pd.Series(["a", "b"], dtype=object), True, id="object_str"),
    pytest.param(pd.Series([DATE, OTHER_DATE], dtype=object), True, id="object_date"),
    pytest.param(
        pd.to_datetime(pd.Series(["2020-01-01", "2020-01-02"])),
        True,
        id="datetime64_ns",
    ),
]
"""Each supported dtype, with the allow_null the pandas backend infers for it.

Spark's inference reports ``allow_null=True`` for every one of them.
"""


class TestSparkInferenceComparison:
    """Pins the relationship between the two backends' inferred schemas.

    The comparison is made on a *coerced* frame, which is what Analytics would
    actually hand to Spark: an uncoerced narrow integer column becomes a Spark
    ``ByteType`` or ``ShortType``, which Analytics has no type for at all. See
    :meth:`test_uncoerced_narrow_integers_are_not_representable_in_analytics`.
    """

    @staticmethod
    def _spark_columns(spark: Any, frame: pd.DataFrame) -> Dict[str, ColumnDescriptor]:
        return spark_schema_to_analytics_columns(spark.createDataFrame(frame).schema)

    @pytest.mark.parametrize("series,pandas_allow_null", SPARK_COMPARISON_CASES)
    def test_the_documented_relationship_holds(
        self, spark: Any, series: pd.Series, pandas_allow_null: bool
    ) -> None:
        """Equal, except allow_null may be False on pandas where Spark says True."""
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame({"A": series}))
        pandas_column = pandas_dtypes_to_analytics_columns(coerced)["A"]
        spark_column = self._spark_columns(spark, coerced)["A"]

        assert pandas_column.column_type == spark_column.column_type
        assert pandas_column.allow_nan == spark_column.allow_nan
        assert pandas_column.allow_inf == spark_column.allow_inf
        # The one permitted disagreement, in the one permitted direction.
        assert spark_column.allow_null is True
        assert pandas_column.allow_null is pandas_allow_null
        assert not (pandas_column.allow_null and not spark_column.allow_null)

    @pytest.mark.parametrize("series,pandas_allow_null", SPARK_COMPARISON_CASES)
    def test_spark_infers_everything_as_nullable(
        self, spark: Any, series: pd.Series, pandas_allow_null: bool
    ) -> None:
        """Spark marks every inferred field nullable, whatever the dtype.

        This is the fact the nullability decision turns on: Spark's
        ``nullable`` here carries no information about the pandas column it came
        from, so matching it would mean discarding what the dtype does say.
        """
        del pandas_allow_null
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame({"A": series}))
        field = spark.createDataFrame(coerced).schema["A"]
        assert field.nullable is True

    @pytest.mark.parametrize(
        "series,expected_spark_type",
        [
            pytest.param(
                pd.Series([1, 2], dtype="int8"), spark_types.LongType(), id="int8"
            ),
            pytest.param(
                pd.Series([1, None], dtype="Int64"), spark_types.LongType(), id="Int64"
            ),
            pytest.param(
                pd.Series([1.5], dtype="float32"),
                spark_types.DoubleType(),
                id="float32",
            ),
            pytest.param(
                pd.Series([1.5, None], dtype="Float64"),
                spark_types.DoubleType(),
                id="Float64",
            ),
            pytest.param(
                pd.Series(["a", None], dtype="string"),
                spark_types.StringType(),
                id="string",
            ),
            pytest.param(
                pd.Series([DATE], dtype=object),
                spark_types.DateType(),
                id="object_date",
            ),
            pytest.param(
                pd.to_datetime(pd.Series(["2020-01-01"])),
                spark_types.TimestampType(),
                id="datetime64_ns",
            ),
        ],
    )
    def test_coercion_lands_on_the_spark_type_analytics_expects(
        self, spark: Any, series: pd.Series, expected_spark_type: Any
    ) -> None:
        """Coercing first makes Spark infer the type Analytics uses for it."""
        coerced = coerce_pandas_schema_or_fail(pd.DataFrame({"A": series}))
        assert (
            spark.createDataFrame(coerced).schema["A"].dataType == expected_spark_type
        )

    @pytest.mark.parametrize("dtype", ["int8", "int16", "Int8", "Int16"])
    def test_uncoerced_narrow_integers_are_not_representable_in_analytics(
        self, spark: Any, dtype: str
    ) -> None:
        """Why the comparison is made after coercion.

        Spark infers a ``ByteType`` or ``ShortType`` from a narrow pandas
        integer column, and Analytics has no column type for either; coercing
        the frame first is what keeps the two backends comparable.
        """
        uncoerced = pd.DataFrame({"A": pd.Series([1, 2], dtype=dtype)})
        spark_type = spark.createDataFrame(uncoerced).schema["A"].dataType
        assert spark_type in (spark_types.ByteType(), spark_types.ShortType())
        assert spark_type not in _SPARK_TO_ANALYTICS
