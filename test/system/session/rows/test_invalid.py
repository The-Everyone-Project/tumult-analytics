"""Tests for invalid session configurations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Dict, Tuple, Type, Union
from unittest.mock import Mock, patch

import pytest
from pyspark.sql import DataFrame
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import DictMetric, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics import (
    AddOneRow,
    AnalyticsInternalError,
    ApproxDPBudget,
    KeySet,
    PrivacyBudget,
    PureDPBudget,
    Query,
    QueryBuilder,
    RhoZCDPBudget,
    Session,
)
from tmlt.analytics._table_identifier import NamedTable
from tmlt.analytics.session import _format_insufficient_budget_msg


@pytest.mark.usefixtures("session_data")
class TestInvalidSession:
    """Tests for Invalid Sessions."""

    sdf: DataFrame
    sdf_col_types: Dict[str, str]
    sdf_input_domain: SparkDataFrameDomain

    @pytest.mark.parametrize(
        "query,error_type,expected_error_msg",
        [
            (
                QueryBuilder("private_source_not_in_catalog")
                .groupby(KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}))
                .count(),
                ValueError,
                "Query references nonexistent table 'private_source_not_in_catalog'",
            )
        ],
    )
    def test_invalid_queries_evaluate(
        self,
        query: Query,
        error_type: Type[Exception],
        expected_error_msg: str,
    ):
        """Evaluate raises error on invalid queries."""
        mock_accountant = Mock()
        mock_accountant.output_measure = PureDP()
        mock_accountant.input_metric = DictMetric(
            {NamedTable("private"): SymmetricDifference()}
        )
        mock_accountant.input_domain = DictDomain(
            {NamedTable("private"): self.sdf_input_domain}
        )
        mock_accountant.d_in = {NamedTable("private"): ExactNumber(1)}
        mock_accountant.privacy_budget = ExactNumber(float("inf"))

        session = Session(accountant=mock_accountant, public_sources={})
        session.create_view(QueryBuilder("private"), "view", cache=False)
        with pytest.raises(error_type, match=expected_error_msg):
            session.evaluate(query, privacy_budget=PureDPBudget(float("inf")))

    @pytest.mark.parametrize(
        "requested,remaining,budget_type,expected_msg",
        [
            (
                (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
                (ExactNumber(2), ExactNumber.from_float(0.4, round_up=True)),
                ApproxDPBudget(2, 0.4),
                "\nRequested: ε=3.000, δ=0.500\nRemaining:"
                " ε=2.000, δ=0.400\nDifference: ε=1.000, δ=0.100",
            ),
            (
                (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
                (ExactNumber(2), ExactNumber.from_float(0.5, round_up=True)),
                ApproxDPBudget(2, 0.5),
                "\nRequested: ε=3.000, δ=0.500\nRemaining:"
                " ε=2.000, δ=0.500\nDifference: ε=1.000",
            ),
            (
                (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
                (ExactNumber(3), ExactNumber.from_float(0.4, round_up=True)),
                ApproxDPBudget(3, 0.4),
                "\nRequested: ε=3.000, δ=0.500\nRemaining:"
                " ε=3.000, δ=0.400\nDifference: δ=0.100",
            ),
            (
                (ExactNumber(3), ExactNumber.from_float(0.5, round_up=True)),
                (ExactNumber(3), ExactNumber.from_float(0.41, round_up=True)),
                ApproxDPBudget(3, 0.41),
                "\nRequested: ε=3.000, δ=0.500\nRemaining:"
                " ε=3.000, δ=0.410\nDifference: δ=9.000e-02",
            ),
            (
                ExactNumber(3),
                ExactNumber(2),
                PureDPBudget(2),
                "\nRequested: ε=3.000\nRemaining: ε=2.000\nDifference: ε=1.000",
            ),
            (
                ExactNumber(3),
                ExactNumber.from_float(2.91, round_up=True),
                PureDPBudget(2.91),
                "\nRequested: ε=3.000\nRemaining: ε=2.910\nDifference: ε=9.000e-02",
            ),
            (
                ExactNumber(3),
                ExactNumber(2),
                RhoZCDPBudget(2),
                "\nRequested: 𝝆=3.000\nRemaining: 𝝆=2.000\nDifference: 𝝆=1.000",
            ),
            (
                ExactNumber(3),
                ExactNumber.from_float(2.91, round_up=True),
                RhoZCDPBudget(2.91),
                "\nRequested: 𝝆=3.000\nRemaining: 𝝆=2.910\nDifference: 𝝆=9.000e-02",
            ),
        ],
    )
    def test_format_insufficient_budget_msg(
        self,
        requested: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
        remaining: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
        budget_type: PrivacyBudget,
        expected_msg: str,
    ):
        """Tests that InsufficientBudgetError is formatted correctly."""
        assert repr(
            _format_insufficient_budget_msg(requested, remaining, budget_type)
        ) == repr(expected_msg)

    @pytest.mark.parametrize("output_measure", [(PureDP()), (ApproxDP()), (RhoZCDP())])
    def test_invalid_privacy_budget_evaluate_and_create(
        self, output_measure: Union[PureDP, RhoZCDP]
    ):
        """Evaluate and create functions raise error on invalid privacy_budget."""
        one_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        two_budget: Union[PureDPBudget, ApproxDPBudget, RhoZCDPBudget]
        if output_measure == PureDP():
            one_budget = PureDPBudget(1)
            two_budget = PureDPBudget(2)
        elif output_measure == ApproxDP():
            one_budget = ApproxDPBudget(1, 0.5)
            two_budget = ApproxDPBudget(2, 0.5)
        elif output_measure == RhoZCDP():
            one_budget = RhoZCDPBudget(1)
            two_budget = RhoZCDPBudget(2)
        else:
            pytest.fail(
                f"must use PureDP, ApproxDP, or RhoZCDP, found {output_measure}"
            )

        query_expr = (
            QueryBuilder("private")
            .groupby(KeySet.from_dict({"A": ["0", "1"], "B": [0, 1]}))
            .count()
        )
        session = Session.from_dataframe(
            privacy_budget=one_budget,
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        with pytest.raises(
            RuntimeError,
            match="Cannot answer query without exceeding the Session privacy budget",
        ):
            session.evaluate(query_expr, privacy_budget=two_budget)

        with pytest.raises(
            RuntimeError,
            match="Cannot perform this partition without "
            "exceeding the Session privacy budget",
        ):
            session.partition_and_create(
                "private",
                privacy_budget=two_budget,
                column="A",
                splits={"part_0": "0", "part_1": "1"},
            )

    @staticmethod
    def _budgets_for(
        output_measure: Union[PureDP, ApproxDP, RhoZCDP],
    ) -> Tuple[PrivacyBudget, PrivacyBudget, PrivacyBudget]:
        """Returns (session budget, query budget, larger budget) for a measure."""
        if output_measure == PureDP():
            return PureDPBudget(10), PureDPBudget(1), PureDPBudget(2)
        if output_measure == ApproxDP():
            return ApproxDPBudget(10, 0.5), ApproxDPBudget(1, 0), ApproxDPBudget(2, 0)
        return RhoZCDPBudget(10), RhoZCDPBudget(1), RhoZCDPBudget(2)

    def _multi_table_session(self, session_budget: PrivacyBudget) -> Session:
        builder = Session.Builder().with_privacy_budget(session_budget)
        for i in range(3):
            builder = builder.with_private_dataframe(
                f"private_{i}", self.sdf, protected_change=AddOneRow()
            )
        return builder.build()

    @pytest.mark.parametrize("output_measure", [(PureDP()), (ApproxDP()), (RhoZCDP())])
    def test_evaluate_rejects_measurement_exceeding_budget(
        self, output_measure: Union[PureDP, ApproxDP, RhoZCDP]
    ):
        """Evaluate rejects a measurement whose privacy loss exceeds the budget.

        Session.evaluate relies on the privacy accountant to check the privacy
        relation of the compiled measurement. Simulate a compiler bug that returns a
        measurement using more budget than requested, and check that it is rejected
        with an internal error, before any budget is spent.
        """
        session_budget, query_budget, larger_budget = self._budgets_for(output_measure)
        session = self._multi_table_session(session_budget)
        query = QueryBuilder("private_1").count()
        compile_and_get_info = session._compile_and_get_info
        too_expensive_measurement, _, noise_info = compile_and_get_info(
            query._query_expr, larger_budget
        )
        _, adjusted_budget, _ = compile_and_get_info(query._query_expr, query_budget)
        with patch.object(
            session,
            "_compile_and_get_info",
            return_value=(too_expensive_measurement, adjusted_budget, noise_info),
        ):
            with pytest.raises(
                AnalyticsInternalError,
                match=r"similar inputs will \*not\* produce similar outputs",
            ) as exc_info:
                session.evaluate(query, privacy_budget=query_budget)
        # The accountant's own check produced the error.
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "does not satisfy the privacy relation" in str(exc_info.value.__cause__)
        assert session.remaining_privacy_budget == session_budget

        # The Session is still usable, and correct measurements are accepted.
        session.evaluate(query, privacy_budget=query_budget)
        assert session.remaining_privacy_budget != session_budget

    def test_evaluate_rejects_mismatched_measurement(self):
        """Evaluate rejects a measurement built for a different input metric.

        This is not a privacy relation failure, so the accountant's error is raised
        unchanged, and no budget is spent.
        """
        session = self._multi_table_session(PureDPBudget(10))
        other_session = Session.from_dataframe(
            privacy_budget=PureDPBudget(10),
            source_id="private_1",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )
        query = QueryBuilder("private_1").count()
        mismatched = other_session._compile_and_get_info(
            query._query_expr, PureDPBudget(1)
        )
        with patch.object(session, "_compile_and_get_info", return_value=mismatched):
            with pytest.raises(ValueError, match="does not match") as exc_info:
                session.evaluate(query, privacy_budget=PureDPBudget(1))
        assert not isinstance(exc_info.value, AnalyticsInternalError)
        assert session.remaining_privacy_budget == PureDPBudget(10)

    def test_invalid_grouping_with_view(self):
        """Tests that grouping flatmap + rename fails if not used in a later groupby."""
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        grouping_flatmap = QueryBuilder("private").flat_map(
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            new_column_types={"Repeat": "INTEGER"},
            grouping=True,
            augment=True,
            max_rows=1,
        )

        session.create_view(
            grouping_flatmap.rename({"Repeat": "repeated"}),
            "grouping_flatmap_renamed",
            cache=False,
        )

        with pytest.raises(
            ValueError,
            match=(
                "Column 'repeated' produced by grouping transformation is not in "
                r"groupby columns \['A'\]"
            ),
        ):
            invalid_query = (
                QueryBuilder("grouping_flatmap_renamed")
                .replace_null_and_nan(replace_with={})
                .groupby(KeySet.from_dict({"A": ["0", "1"]}))
                .sum("X", 0, 3)
            )
            session.evaluate(
                query_expr=invalid_query,
                privacy_budget=PureDPBudget(10),
            )

    def test_invalid_double_grouping_with_view(self):
        """Tests that multiple grouping transformations aren't allowed."""
        session = Session.from_dataframe(
            privacy_budget=PureDPBudget(float("inf")),
            source_id="private",
            dataframe=self.sdf,
            protected_change=AddOneRow(),
        )

        grouping_flatmap = QueryBuilder("private").flat_map(
            f=lambda row: [{"Repeat": 1 if row["A"] == "0" else 2}],
            new_column_types={"Repeat": "INTEGER"},
            grouping=True,
            augment=True,
            max_rows=1,
        )

        session.create_view(grouping_flatmap, "grouping_flatmap", cache=False)

        grouping_flatmap_2 = QueryBuilder("grouping_flatmap").flat_map(
            f=lambda row: [{"i": row["X"]} for _ in range(row["Repeat"])],
            new_column_types={"i": "INTEGER"},
            grouping=True,
            augment=True,
            max_rows=2,
        )

        with pytest.raises(
            ValueError,
            match=(
                "Multiple grouping transformations are used in this query. "
                "Only one grouping transformation is allowed."
            ),
        ):
            session.create_view(grouping_flatmap_2, "grouping_flatmap_2", cache=False)
