"""Operation for constructing a KeySet from a Spark DataFrame."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from dataclasses import dataclass
from typing import Any, Literal, NoReturn, Optional, overload

from pyspark.sql import DataFrame

from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import ColumnDescriptor, spark_schema_to_analytics_columns

from ._base import KeySetOp
from ._frames import FrameKind, frame_kind, unsupported
from ._utils import validate_schema


@dataclass(frozen=True)
class FromSparkDataFrame(KeySetOp):
    """Construct a KeySet from a Spark DataFrame."""

    df: DataFrame

    def __post_init__(self):
        """Validation."""
        validate_schema(self.schema())
        if len(self.columns()) == 0 and not self.df.isEmpty():
            raise ValueError("A KeySet with no columns must not have any rows.")

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return set(self.df.columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return spark_schema_to_analytics_columns(self.df.schema)

    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return coerce_spark_schema_or_fail(self.df.dropDuplicates())

    def _pandas_dataframe(self) -> NoReturn:
        """Raises: a Spark dataframe is not collected into memory implicitly.

        See :meth:`unsupported_ops`.
        """
        raise unsupported(
            self,
            FrameKind.PANDAS,
            "Collecting a Spark dataframe into memory is the caller's decision"
            " to make; build the KeySet with KeySet.from_tuples instead.",
        )

    def unsupported_frame_ops(self, kind: FrameKind) -> set[str]:
        """The operations in this op-tree that cannot produce a frame of this kind.

        A KeySet built from a Spark dataframe can only be materialized as one.
        Collecting a distributed frame into the driver's memory is not something
        to do because a backend was switched: it is unbounded work on data whose
        size nobody has looked at, and if it is the right thing to do it is the
        caller's to decide. Build the KeySet with
        :meth:`~tmlt.analytics.KeySet.from_tuples` instead.
        """
        unsupported_ops = super().unsupported_frame_ops(kind)
        if kind is FrameKind.PANDAS:
            unsupported_ops.add(type(self).__name__)
        return unsupported_ops

    def is_empty(self, backend: Backend = SPARK) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        The answer is a property of the Spark dataframe this operation holds, so
        it is the same whatever backend asks for it.
        """
        return self.df.isEmpty()

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return False

    @overload
    def size(self, fast: Literal[True], backend: Backend = SPARK) -> Optional[int]: ...

    @overload
    def size(self, fast: Literal[False], backend: Backend = SPARK) -> int: ...

    @overload
    def size(self, fast: bool, backend: Backend = SPARK) -> Optional[int]: ...

    def size(self, fast, backend=SPARK):
        """Determine the size of the KeySet resulting from this operation.

        Like :meth:`is_empty`, this counts the Spark dataframe this operation
        holds whatever backend asks: the count is a property of that frame, and
        producing it does not hand the frame to another backend.
        """
        if not self.columns():
            return 1
        if fast:
            return None
        return self._spark_dataframe().count()

    def __eq__(self, other: Any):
        """Determine if this KeySetOp is equal to another."""
        if not isinstance(other, FromSparkDataFrame):
            return False

        return self.df.schema == other.df.schema and self.df.sameSemantics(other.df)

    def __str__(self):
        """Human-readable string representation."""
        return f"FromSparkDataFrame {self.df}"
