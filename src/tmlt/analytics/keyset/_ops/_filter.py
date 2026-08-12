"""Operation for filtering the rows in a KeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass
from typing import Literal, NoReturn, Optional, Union, overload

from pyspark.sql import Column, DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp
from ._frames import (
    FrameKind,
    frame_count,
    frame_is_empty,
    frame_kind,
    unsupported,
)


@dataclass(frozen=True)
class Filter(KeySetOp):
    """Filter the rows of a KeySet."""

    child: KeySetOp
    condition: Union[Column, str]

    def __post_init__(self):
        """Validation."""
        if not isinstance(self.child, KeySetOp):
            raise AnalyticsInternalError(
                "Child of Project KeySetOp must be a KeySetOp, "
                f"not {type(self.child).__qualname__}."
            )

        if isinstance(self.condition, str) and self.condition == "":
            raise ValueError("A KeySet cannot be filtered by an empty condition.")

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return self.child.columns()

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return self.child.schema()

    def children(self) -> tuple[KeySetOp, ...]:
        """The operations whose outputs this one is computed from."""
        return (self.child,)

    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return self.child._spark_dataframe().filter(self.condition)

    def _pandas_dataframe(self) -> NoReturn:
        """Raises: a filter condition can only be evaluated by Spark.

        See :meth:`unsupported_ops`.
        """
        raise unsupported(
            self,
            FrameKind.PANDAS,
            "A KeySet filter condition is a Spark SQL expression; build the"
            " filtered KeySet with KeySet.from_tuples instead.",
        )

    def unsupported_frame_ops(self, kind: FrameKind) -> set[str]:
        """The operations in this op-tree that cannot produce a frame of this kind.

        A filter condition is a Spark SQL expression or a
        :class:`~pyspark.sql.Column`, and evaluating one means asking Spark. It
        is a piece of Spark the user wrote, not a piece Analytics chose, so
        there is nothing to translate it into.
        """
        unsupported_ops = super().unsupported_frame_ops(kind)
        if kind is FrameKind.PANDAS:
            unsupported_ops.add(type(self).__name__)
        return unsupported_ops

    def is_empty(self, backend: Backend = SPARK) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        This operation may be expensive.
        """
        return frame_is_empty(self.dataframe(backend))

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return self.child.is_plan()

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
        return f"Filter {self.condition}\n" + textwrap.indent(str(self.child), "  ")
