"""Base class for KeySet operations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Collection, Literal, Optional, overload

import pandas as pd
from pyspark.sql import DataFrame

from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._schema import ColumnDescriptor
from tmlt.analytics._utils import AnalyticsInternalError

from ._frames import FrameKind, KeySetFrame, frame_kind, unsupported


class KeySetOp(ABC):
    """Base class for operations used to define KeySets."""

    _no_pandas_hint: ClassVar[Optional[str]] = None
    """Why this operation cannot be materialized in memory, if it cannot.

    An operation with no pandas path has to say so twice -- once when asked for
    a frame, and once when :meth:`unsupported_frame_ops` walks the tree looking
    for exactly such operations before evaluating any of it. The two answers
    have to agree, and nothing but this attribute makes them: setting it is the
    whole of declaring that an operation is Spark-only, and the default of None
    is the whole of declaring that it is not.

    The value is the hint the error carries, so it should say what to do
    instead rather than only what went wrong."""

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

    def with_children(self, children: tuple[KeySetOp, ...]) -> KeySetOp:
        """This operation over the given children in place of its own.

        The companion of :meth:`children`, and its inverse:
        ``op.with_children(op.children())`` equals ``op``. Together the two are
        everything a tree walk needs -- take an operation apart, rewrite the
        pieces, put it back together -- which is how the rewrite rules in
        :mod:`~tmlt.analytics.keyset._ops._rules` walk a tree without knowing
        what any node in it is. An operation that overrides both is walked
        correctly without anything else being told it exists.

        The default is the one the operations that introduce data want: they
        have no children, so there is nothing to replace.

        Args:
            children: The replacements: as many as :meth:`children` returns, in
                the same order.
        """
        if children:
            raise AnalyticsInternalError(
                f"{type(self).__qualname__} has no children, but"
                f" {len(children)} were passed to with_children."
            )
        return self

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
        so, with the reason from :attr:`_no_pandas_hint`;
        :meth:`unsupported_ops` reports the same operations from the same
        attribute, without evaluating anything.
        """
        raise unsupported(self, FrameKind.PANDAS, self._no_pandas_hint)

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
        if kind is FrameKind.PANDAS and self._no_pandas_hint is not None:
            unsupported_ops.add(type(self).__name__)
        for child in self.children():
            unsupported_ops |= child.unsupported_frame_ops(kind)
        return unsupported_ops

    @abstractmethod
    def is_empty(self) -> bool:
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
