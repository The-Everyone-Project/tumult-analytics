"""Operation for subtracting a KeySet from a KeySet or Plan."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass, replace
from typing import Collection, Literal, Optional, overload

import pandas as pd
from pyspark.sql import DataFrame
from tmlt.core.utils.join import join
from tmlt.core.utils.pandas_join import join as pandas_join

from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp
from ._frames import frame_count, frame_is_empty


@dataclass(frozen=True)
class Subtract(KeySetOp):
    """Subtract a (non-plan) KeySetOp from another KeySetOp.

    The columns of ``right`` must be a subset of the columns of ``left``.
    The result is a plan iff ``left`` is a plan.
    """

    left: KeySetOp
    right: KeySetOp

    def __post_init__(self):
        """Validation."""
        if self.right.is_plan():
            raise ValueError(
                "Cannot subtract a KeySetPlan from a KeySet or KeySetPlan."
            )

        non_subset_columns = self.right.columns() - self.left.columns()
        if non_subset_columns:
            raise ValueError(
                "Unable to subtract KeySets, right hand side has columns that "
                f"do not exist in the left hand side: {', '.join(non_subset_columns)}"
            )

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.left.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return self.left.schema()

    def children(self) -> tuple[KeySetOp, ...]:
        """The operations whose outputs this one is computed from."""
        return (self.left, self.right)

    def with_children(self, children: tuple[KeySetOp, ...]) -> KeySetOp:
        """This subtraction over the given left and right operations."""
        left, right = children
        return replace(self, left=left, right=right)

    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return join(
            self.left._spark_dataframe(),
            self.right._spark_dataframe(),
            on=list(self.right.columns()),
            how="left_anti",
            nulls_are_equal=True,
        )

    def _pandas_dataframe(self) -> pd.DataFrame:
        """Generate the pandas dataframe corresponding to this operation.

        A subtraction is a left anti-join on both backends, and Core's pandas
        implementation of one keeps the left side's rows and dtypes untouched,
        as Spark's does.
        """
        return pandas_join(
            self.left._pandas_dataframe(),
            self.right._pandas_dataframe(),
            on=list(self.right.columns()),
            how="left_anti",
            nulls_are_equal=True,
        )

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        This operation may be expensive.
        """
        return self.left.is_empty() or frame_is_empty(self.dataframe())

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.left.is_plan()

    @overload
    def size(self, fast: Literal[True], backend: Backend = SPARK) -> Optional[int]: ...

    @overload
    def size(self, fast: Literal[False], backend: Backend = SPARK) -> int: ...

    @overload
    def size(self, fast: bool, backend: Backend = SPARK) -> Optional[int]: ...

    def size(self, fast, backend=SPARK):
        """Determine the size of the KeySet resulting from this operation.

        Subtract cannot determine its size fast.
        """
        if fast:
            return None
        return frame_count(self.dataframe(backend))

    def __str__(self):
        """Human-readable string representation."""
        return (
            "Subtract\n"
            + textwrap.indent(str(self.left), "  ")
            + "\n"
            + textwrap.indent(str(self.right), "  ")
        )

    def decompose(
        self, split_columns: Collection[str]
    ) -> tuple[list[KeySetOp], list[KeySetOp]]:
        """Decompose this KeySetOp into a collection of factors and subtracted values.

        See :meth:`KeySet._decompose` for details.
        """
        left_factors, left_svs = self.left.decompose(split_columns)

        # If a factor contains all of the columns being subtracted, absorb the
        # subtraction into the factor instead of adding a subtracted value.
        merged_subtraction = False
        new_factors: list[KeySetOp] = []
        for f in left_factors:
            if self.right.columns() <= f.columns():
                new_factors.append(Subtract(f, self.right))
                merged_subtraction = True
            else:
                new_factors.append(f)

        if merged_subtraction:
            return new_factors, left_svs
        return new_factors, left_svs + [self.right]
