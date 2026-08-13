"""Operation for computing the union of two KeySets or Plans."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass, replace
from typing import Collection, Literal, Optional, overload

import pandas as pd
from pyspark.sql import DataFrame
from tmlt.core.utils.pandas_grouping import distinct_rows

from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._schema import ColumnDescriptor, Schema, analytics_to_pandas_dtypes

from ._base import KeySetOp
from ._frames import frame_count


@dataclass(frozen=True)
class Union(KeySetOp):
    """Compute the union of two KeySetOps.

    The schemas of ``left`` and ``right`` must match exactly aside from
    nullability; if a column is nullable in either operand, it will be nullable
    in the union. The result of this operation is concrete iff both operands are
    concrete.
    """

    left: KeySetOp
    right: KeySetOp

    def __post_init__(self):
        """Validation."""
        if self.left.columns() != self.right.columns():
            raise ValueError(
                "KeySet union operands must have the same columns:\n"
                f"Left:  {' '.join(sorted(self.left.columns()))}\n"
                f"Right: {' '.join(sorted(self.right.columns()))}"
            )

        if not (self.left.is_plan() or self.right.is_plan()):
            mismatched_columns = {}
            for c in self.columns():
                if (
                    self.left.schema()[c].column_type
                    != self.right.schema()[c].column_type
                ):
                    mismatched_columns[c] = (
                        self.left.schema()[c].column_type,
                        self.right.schema()[c].column_type,
                    )
            if mismatched_columns:
                raise ValueError(
                    "KeySet union operands have mismatched column types:\n"
                    + "\n".join(
                        f"{c}: {left} / {right}"
                        for c, (left, right) in mismatched_columns.items()
                    )
                )

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.left.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        left_schema = self.left.schema()
        right_schema = self.right.schema()
        return {
            c: ColumnDescriptor(
                left_schema[c].column_type,
                allow_null=left_schema[c].allow_null or right_schema[c].allow_null,
                allow_nan=left_schema[c].allow_nan or right_schema[c].allow_nan,
                allow_inf=left_schema[c].allow_inf or right_schema[c].allow_inf,
            )
            for c in left_schema
        }

    def children(self) -> tuple[KeySetOp, ...]:
        """The operations whose outputs this one is computed from."""
        return (self.left, self.right)

    def with_children(self, children: tuple[KeySetOp, ...]) -> KeySetOp:
        """This union over the given left and right operations."""
        left, right = children
        return replace(self, left=left, right=right)

    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        left_df = self.left._spark_dataframe()
        right_df = self.right._spark_dataframe()
        return left_df.unionByName(right_df).distinct()

    def _pandas_dataframe(self) -> pd.DataFrame:
        """Generate the pandas dataframe corresponding to this operation.

        The two operands are concatenated one column at a time, each side cast
        to the union's dtype for that column first. Concatenating the frames
        whole would let pandas resolve any dtype disagreement itself, and its
        resolution of an object column against a numeric one is an object column
        of NaNs -- which is a different row from an object column of ``None``
        under the null-safe deduplication that follows, and a different row from
        what Spark's ``unionByName`` produces.
        """
        schema = self.schema()
        columns = list(schema)
        if not columns:
            return pd.DataFrame()

        dtypes = analytics_to_pandas_dtypes(Schema(schema))
        operands = [self.left._pandas_dataframe(), self.right._pandas_dataframe()]
        combined = pd.DataFrame(
            {
                column: pd.concat(
                    [operand[column].astype(dtypes[column]) for operand in operands],
                    ignore_index=True,
                )
                for column in columns
            },
            columns=columns,
        )
        return distinct_rows(combined)

    def is_empty(self, backend: Backend = SPARK) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        This operation may be expensive.
        """
        return self.left.is_empty(backend) and self.right.is_empty(backend)

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.left.is_plan() or self.right.is_plan()

    @overload
    def size(self, fast: Literal[True], backend: Backend = SPARK) -> Optional[int]: ...

    @overload
    def size(self, fast: Literal[False], backend: Backend = SPARK) -> int: ...

    @overload
    def size(self, fast: bool, backend: Backend = SPARK) -> Optional[int]: ...

    def size(self, fast, backend=SPARK):
        """Determine the size of the KeySet resulting from this operation."""
        # There's no shortcut to get this count due to deduplication
        if fast:
            return None
        return frame_count(self.dataframe(backend))

    def __str__(self):
        """Human-readable string representation."""
        return (
            "Union\n"
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
        # Union doesn't naturally decompose into factors, so treat it as atomic
        return [self], []
