"""Defines a visitor for creating noisy measurements from query expressions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import List, Tuple

from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.metrics import HammingDistance, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.misc import get_nonconflicting_string

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._noise_info import NoiseInfo, _noise_from_measurement
from tmlt.analytics._query_expr import GetGroups, QueryExpr
from tmlt.analytics._query_expr_compiler._base_measurement_visitor import (
    BaseMeasurementVisitor,
)
from tmlt.analytics._query_expr_compiler._transformation_visitor import (
    TransformationVisitor,
)
from tmlt.analytics._table_reference import TableReference
from tmlt.analytics._transformation_utils import get_table_from_ref
from tmlt.analytics.constraints import Constraint
from tmlt.analytics.privacy_budget import ApproxDPBudget


class MeasurementVisitor(BaseMeasurementVisitor):
    """A visitor to create a measurement from a DP query expression."""

    def _visit_child_transformation(
        self, expr: QueryExpr, mechanism: NoiseMechanism
    ) -> Tuple[Transformation, TableReference, List[Constraint]]:
        """Visit a child transformation, producing a transformation."""
        tv = TransformationVisitor(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            mechanism=mechanism,
            catalog=self.catalog,
            backend=self.backend,
        )
        child, reference, constraints = expr.accept(tv)

        tv.validate_transformation(expr, child, reference, self.catalog)

        return child, reference, constraints

    def _handle_enforce(
        self,
        constraint: Constraint,
        child_transformation: Transformation,
        child_ref: TableReference,
        **kwargs,
    ) -> Tuple[Transformation, TableReference]:
        """Enforce a constraint after a child transformation."""
        return constraint._enforce(
            child_transformation, child_ref, backend=self.backend, **kwargs
        )

    def visit_get_groups(self, expr: GetGroups) -> Tuple[Measurement, NoiseInfo]:
        """Create a measurement from a GetGroups query expression."""
        if not isinstance(self.budget, ApproxDPBudget):
            raise ValueError("GetGroups is only supported with ApproxDPBudgets.")

        schema = expr.child.schema(self.catalog)

        # Set the columns if no columns were provided.
        if expr.columns:
            columns = expr.columns
        else:
            columns = tuple(
                col for col in schema.column_descs.keys() if col != schema.id_column
            )

        # Check if ID column is one of the columns in get_groups
        # Note: if get_groups columns is None or empty, all of the columns in the table
        # is used for partition selection, hence that needs to be checked as well
        if schema.id_column and (not columns or (schema.id_column in columns)):
            raise RuntimeError(
                "get_groups cannot be used on the privacy ID column"
                f" ({schema.id_column}) of a table with the AddRowsWithID protected"
                " change."
            )

        child_transformation, child_ref = self._truncate_table(
            *self._visit_child_transformation(expr.child, NoiseMechanism.GEOMETRIC),
            grouping_columns=tuple(),
        )

        transformation = get_table_from_ref(child_transformation, child_ref)
        # The domain is checked again after each step below, because each step
        # replaces it.
        self.backend.table_domain(
            transformation.output_domain, "the truncated table's domain"
        )

        # squares the sensitivity in zCDP, which is a worst-case analysis
        # that we may be able to improve.
        if isinstance(transformation.output_metric, IfGroupedBy):
            transformation |= self.backend.require("UnwrapIfGroupedBy")(
                transformation.output_domain, transformation.output_metric
            )
        self.backend.table_domain(
            transformation.output_domain, "the unwrapped table's domain"
        )
        if not isinstance(
            transformation.output_metric,
            (IfGroupedBy, HammingDistance, SymmetricDifference),
        ):
            raise AnalyticsInternalError(
                "Unrecognized metric in GetGroups transformation."
                f"{transformation.output_metric} instead."
            )

        transformation |= self.backend.require("Select")(
            transformation.output_domain, transformation.output_metric, list(columns)
        )

        mid_stability = transformation.stability_function(self.stability)
        output_domain = self.backend.table_domain(
            transformation.output_domain, "the selected table's domain"
        )
        count_column = "count"
        if count_column in set(output_domain.schema):
            count_column = get_nonconflicting_string(list(output_domain.schema))

        epsilon, delta = self.budget.value
        agg = self.backend.require("create_partition_selection_measurement")(
            input_domain=output_domain,
            epsilon=epsilon,
            delta=delta,
            d_in=mid_stability,
            count_column=count_column,
        )

        self._validate_measurement(agg, mid_stability)

        measurement = PostProcess(
            transformation | agg, lambda result: result.drop(count_column)
        )
        noise_info = _noise_from_measurement(measurement)
        return measurement, noise_info
