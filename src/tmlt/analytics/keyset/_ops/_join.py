"""Operation for constructing a KeySet by joining two factors."""

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
class Join(KeySetOp):
    """Construct a KeySet from the inner natural join of two existing KeySetOps.

    The ``left`` and ``right`` factors must have at least one overlapping
    column. ``Join`` is concrete if both of the factors are concrete.
    """

    left: KeySetOp
    right: KeySetOp

    def __post_init__(self):
        """Validation."""
        if not self.join_columns():
            raise ValueError(
                "Unable to join KeySets, they have no "
                f"overlapping columns (left: {' '.join(self.left.columns())}, "
                f"right: {' '.join(self.right.columns())})"
            )

        if not self.is_plan():
            for c in self.join_columns():
                l_type = self.left.schema()[c]
                r_type = self.right.schema()[c]
                if l_type.column_type != r_type.column_type:
                    raise ValueError(
                        f"Unable to join KeySets, join column {c} does not have the "
                        f"same type in both KeySets (left: {l_type}, right: {r_type})."
                    )

    def join_columns(self) -> set[str]:
        """The columns being joined on."""
        return set(self.left.columns()) & set(self.right.columns())

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.left.columns() | self.right.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        schema = self.left.schema().copy()
        for c, desc in self.right.schema().items():
            # For columns in the join columns, nulls are allowed if they are
            # allowed in both joined KeySets. For other columns (those appearing
            # only on the right), just add the column descriptor to the schema
            # as-is.
            if c in self.join_columns():
                schema[c] = ColumnDescriptor(
                    schema[c].column_type,
                    allow_null=schema[c].allow_null and desc.allow_null,
                    allow_nan=schema[c].allow_nan and desc.allow_nan,
                    allow_inf=schema[c].allow_inf and desc.allow_inf,
                )
            else:
                schema[c] = desc

        return schema

    def children(self) -> tuple[KeySetOp, ...]:
        """The operations whose outputs this one is computed from."""
        return (self.left, self.right)

    def with_children(self, children: tuple[KeySetOp, ...]) -> KeySetOp:
        """This join over the given left and right operations."""
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
            on=list(self.join_columns()),
            nulls_are_equal=True,
        )

    def _pandas_dataframe(self) -> pd.DataFrame:
        """Generate the pandas dataframe corresponding to this operation.

        Core's :func:`~tmlt.core.utils.pandas_join.join` answers the same
        question its Spark counterpart above does -- which rows pair up, what
        the output columns are called and in what order -- so the two backends
        produce the same keys.
        """
        return pandas_join(
            self.left._pandas_dataframe(),
            self.right._pandas_dataframe(),
            on=list(self.join_columns()),
            nulls_are_equal=True,
        )

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return frame_is_empty(self.dataframe())

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
        if fast:
            return None
        return frame_count(self.dataframe(backend))

    def __str__(self):
        """Human-readable string representation."""
        return (
            "Join\n"
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
        if self.join_columns() <= set(split_columns):
            l_fs, l_svs = self.left.decompose(split_columns)
            r_fs, r_svs = self.right.decompose(split_columns)

            return l_fs + r_fs, l_svs + r_svs
        else:
            return [self], []
