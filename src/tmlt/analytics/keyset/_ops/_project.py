"""Operation for projecting columns out of a KeySet."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import textwrap
from dataclasses import dataclass, replace
from typing import Literal, Optional, overload

import pandas as pd
from pyspark.sql import DataFrame
from tmlt.core.utils.pandas_grouping import distinct_rows

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._schema import ColumnDescriptor

from ._base import KeySetOp
from ._frames import frame_count
from ._utils import validate_column_names


@dataclass(frozen=True)
class Project(KeySetOp):
    """Project a set of columns out of a KeySet."""

    child: KeySetOp
    projected_columns: frozenset[str]

    def __post_init__(self):
        """Validation."""
        if not isinstance(self.child, KeySetOp):
            raise AnalyticsInternalError(
                "Child of Project KeySetOp must be a KeySetOp, "
                f"not {type(self.child).__qualname__}."
            )
        if not isinstance(self.projected_columns, frozenset):
            raise AnalyticsInternalError(
                "Project KeySetOp's columns must be a frozenset, "
                f"not {type(self.projected_columns).__qualname__}."
            )
        validate_column_names(self.projected_columns)

        if len(self.projected_columns) == 0:
            raise ValueError(
                "At least one column must be kept when subscripting a KeySet."
            )

        missing_columns = self.projected_columns - set(self.child.columns())
        if len(missing_columns) == 1:
            raise ValueError(
                f"Column {list(missing_columns)[0]} is not present in KeySet, "
                f"available columns are: {', '.join(self.child.columns())}"
            )
        if len(missing_columns) > 1:
            raise ValueError(
                f"Columns {', '.join(sorted(missing_columns))} are not present in "
                f"KeySet, available columns are: {', '.join(self.child.columns())}"
            )

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return set(self.projected_columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        child_schema = self.child.schema()
        return {c: child_schema[c] for c in self.projected_columns}

    def children(self) -> tuple[KeySetOp, ...]:
        """The operations whose outputs this one is computed from."""
        return (self.child,)

    def with_children(self, children: tuple[KeySetOp, ...]) -> KeySetOp:
        """This projection over the given child operation."""
        (child,) = children
        return replace(self, child=child)

    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        This operation may be computationally expensive, even though the full
        dataframe is not evaluated until it is used elsewhere.
        """
        return (
            self.child._spark_dataframe()
            .select(*self.projected_columns)
            .dropDuplicates()
        )

    def _pandas_dataframe(self) -> pd.DataFrame:
        """Generate the pandas dataframe corresponding to this operation.

        Deduplication is Core's :func:`~tmlt.core.utils.pandas_grouping.distinct_rows`
        rather than :meth:`pandas.DataFrame.drop_duplicates`, so that a null and
        a NaN in an object column stay two rows, as they are on Spark.
        """
        projected = self.child._pandas_dataframe()[list(self.projected_columns)]
        return distinct_rows(projected)

    def is_empty(self) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return self.child.is_empty()

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
        return f"Project {', '.join(self.projected_columns)}\n" + textwrap.indent(
            str(self.child), "  "
        )
