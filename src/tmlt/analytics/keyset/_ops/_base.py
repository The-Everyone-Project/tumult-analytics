"""Base class for KeySet operations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Literal, Optional, overload

import pandas as pd
from pyspark.sql import DataFrame

from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._schema import ColumnDescriptor

from ._frames import FrameKind, KeySetFrame, frame_kind, unsupported


class KeySetOp(ABC):
    """Base class for operations used to define KeySets."""

    @abstractmethod
    def columns(self) -> set[str]:
        r"""Get a list of the columns included in the output of this operation.

        The column order of a :class:`KeySetOp` is an implementation detail, and
        should not be considered when deciding whether two :class:`KeySetOp`\ s
        are equivalent.
        """

    @abstractmethod
    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation.

        If this operation is a plan (i.e. ``self.is_plan()`` returns True), this
        method will raise ``AnalyticsInternalError``.
        """

    def children(self) -> tuple[KeySetOp, ...]:
        """The operations whose outputs this one is computed from.

        Empty for the operations that introduce data rather than transform it.
        """
        return ()

    def dataframe(self, backend: Backend = SPARK) -> KeySetFrame:
        """Generate the dataframe corresponding to this operation.

        This operation may be computationally expensive, even though a Spark
        dataframe is not evaluated until it is used elsewhere.

        If this operation is a plan (i.e. ``self.is_plan()`` returns True), this
        method will raise ``AnalyticsInternalError``.

        Args:
            backend: The backend whose kind of dataframe to produce. Defaults to
                Spark, so a caller that says nothing gets what it always got.

        Raises:
            NotSupportedByBackend: If this operation cannot produce that kind of
                dataframe. :meth:`unsupported_ops` names every such operation in
                an op-tree without materializing any of it.
        """
        return self.frame(frame_kind(backend))

    def frame(self, kind: FrameKind) -> KeySetFrame:
        """Generate the frame of the given kind corresponding to this operation.

        This is :meth:`dataframe` with the backend already resolved to the one
        thing materialization reads off it; see
        :mod:`~tmlt.analytics.keyset._ops._frames`.
        """
        if kind is FrameKind.PANDAS:
            return self._pandas_dataframe()
        return self._spark_dataframe()

    @abstractmethod
    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation."""

    def _pandas_dataframe(self) -> pd.DataFrame:
        """Generate the pandas dataframe corresponding to this operation.

        Operations that have no pandas implementation inherit this, which says
        so; :meth:`unsupported_ops` reports the same operations without
        evaluating anything.
        """
        raise unsupported(self, FrameKind.PANDAS)

    def unsupported_ops(self, backend: Backend) -> set[str]:
        """The names of the operations in this op-tree the backend cannot perform.

        Walking the tree for these lets a caller fail with every offending
        operation named at once, before any of the tree has been evaluated,
        rather than with whichever one :meth:`dataframe` reached first.

        Args:
            backend: The backend the tree would be materialized on.
        """
        return self.unsupported_frame_ops(frame_kind(backend))

    def unsupported_frame_ops(self, kind: FrameKind) -> set[str]:
        """The operations in this op-tree that cannot produce a frame of this kind.

        This is :meth:`unsupported_ops` with the backend already resolved to the
        one thing the answer depends on, and stands to it as :meth:`frame` does
        to :meth:`dataframe`. It is the form the overrides are written against,
        and the form a caller with no backend in hand -- :meth:`KeySet.__eq__`,
        which cannot be given one -- asks the question in.

        Args:
            kind: The kind of frame the tree would be materialized as.
        """
        unsupported_ops: set[str] = set()
        for child in self.children():
            unsupported_ops |= child.unsupported_frame_ops(kind)
        return unsupported_ops

    @abstractmethod
    def is_empty(self, backend: Backend = SPARK) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""

    @abstractmethod
    def is_plan(self) -> bool:
        """Determine whether this plan has any parts requiring partition selection."""

    @overload
    def size(self, fast: Literal[True], backend: Backend = SPARK) -> Optional[int]: ...

    @overload
    def size(self, fast: Literal[False], backend: Backend = SPARK) -> int: ...

    # Needed to make mypy happy, as it won't automatically combine the above
    # overloads to understand that fast is a bool.
    #   https://github.com/python/mypy/issues/10194
    @overload
    def size(self, fast: bool, backend: Backend = SPARK) -> Optional[int]: ...

    @abstractmethod
    def size(self, fast, backend=SPARK):
        """Determine the size of the KeySet resulting from this operation."""

    def decompose(
        self, split_columns: Collection[str]
    ) -> tuple[list[KeySetOp], list[KeySetOp]]:
        """Decompose this KeySetOp into a collection of factors and subtracted values.

        See :meth:`KeySet._decompose` for details.
        """
        _ = split_columns
        return [self], []
