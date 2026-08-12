"""Operation for detecting the KeySet for a group of columns."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Literal, Optional, overload

import pandas as pd
from pyspark.sql import DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp
from ._frames import FrameKind
from ._utils import validate_column_names


@dataclass(frozen=True)
class Detect(KeySetOp):
    """Detect a KeySet from a group of columns on a Session."""

    detect_columns: frozenset[str]

    def __post_init__(self):
        """Validation."""
        if len(self.detect_columns) == 0:
            raise ValueError(
                "Detect must be used on a non-empty collection of columns."
            )
        validate_column_names(self.detect_columns)

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return set(self.detect_columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation.

        Raises ``AnalyticsInternalError``, as the schema is not known until
        fixed values are supplied for detected columns.
        """
        raise AnalyticsInternalError("KeySetPlan does not have a fixed schema.")

    def unsupported_frame_ops(self, kind: FrameKind) -> set[str]:
        """The operations in this op-tree that cannot produce a frame of this kind.

        No backend can materialize a plan, so this one names itself for every
        kind of frame rather than for a particular one.
        """
        unsupported_ops = super().unsupported_frame_ops(kind)
        unsupported_ops.add(type(self).__name__)
        return unsupported_ops

    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        Raises ``AnalyticsInternalError``, as the dataframe is not known until
        fixed values are supplied. That no frame can be built here is not a
        shortcoming of any backend, so it is not a
        :class:`~tmlt.analytics._backends.NotSupportedByBackend`.
        """
        raise AnalyticsInternalError(
            "KeySetPlan does not have a fixed dataframe representation."
        )

    def _pandas_dataframe(self) -> pd.DataFrame:
        """Generate the pandas dataframe corresponding to this operation.

        Raises ``AnalyticsInternalError``, for the same reason
        :meth:`_spark_dataframe` does.
        """
        raise AnalyticsInternalError(
            "KeySetPlan does not have a fixed dataframe representation."
        )

    def is_empty(self, backend: Backend = SPARK) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty.

        Raises ``AnalyticsInternalError``, as whether the operation's output
        dataframe is empty is not known until fixed values are supplied.
        """
        raise AnalyticsInternalError("KeySetPlan does not have a fixed size.")

    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""
        return True

    @overload
    def size(self, fast: Literal[True], backend: Backend = SPARK) -> Optional[int]: ...

    @overload
    def size(self, fast: Literal[False], backend: Backend = SPARK) -> int: ...

    @overload
    def size(self, fast: bool, backend: Backend = SPARK) -> Optional[int]: ...

    def size(self, fast, backend=SPARK):
        """Determine the size of the KeySet resulting from this operation.

        Raises ``AnalyticsInternalError``, as the operation's output dataframe
        size is not known until fixed values are supplied.
        """
        raise AnalyticsInternalError("KeySetPlan does not have a fixed size.")

    def __str__(self):
        """Human-readable string representation."""
        return f"Detect {', '.join(self.detect_columns)}"

    def decompose(
        self, split_columns: Collection[str]
    ) -> tuple[list[KeySetOp], list[KeySetOp]]:
        """Decompose this KeySetOp into a collection of factors and subtracted values.

        See :meth:`KeySet._decompose` for details. Raises ``AnalyticsInternalError``, as
        the operation cannot be decomposed until fixed values are supplied.
        """
        raise AnalyticsInternalError("KeySetPlan cannot be decomposed.")
