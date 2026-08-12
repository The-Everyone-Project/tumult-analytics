"""Operation for constructing a KeySet from a pandas DataFrame."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from dataclasses import dataclass
from typing import Any, Literal, Optional, overload

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from tmlt.core.utils.pandas_grouping import distinct_rows

from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._coerce_pandas_schema import coerce_pandas_schema_or_fail
from tmlt.analytics._schema import (
    ColumnDescriptor,
    Schema,
    analytics_to_spark_schema,
    pandas_dtypes_to_analytics_columns,
)

from ._base import KeySetOp
from ._utils import validate_schema


@dataclass(frozen=True)
class FromPandasDataFrame(KeySetOp):
    """Construct a KeySet from a pandas DataFrame.

    This is the counterpart of
    :class:`~tmlt.analytics.keyset._ops._from_dataframe.FromSparkDataFrame`, and
    unlike it, it materializes on either backend: the keys are already in
    memory, so handing them to Spark costs no more than any other KeySet built
    from values does.
    """

    df: pd.DataFrame

    def __post_init__(self):
        """Validation."""
        validate_schema(self.schema())
        if len(self.columns()) == 0 and len(self.df.index) > 0:
            raise ValueError("A KeySet with no columns must not have any rows.")

    def columns(self) -> set[str]:
        """Get a list of the columns included in the output of this operation."""
        return set(self.df.columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Get the schema of the output of this operation."""
        return pandas_dtypes_to_analytics_columns(self.df)

    def _pandas_dataframe(self) -> pd.DataFrame:
        """Generate the pandas dataframe corresponding to this operation.

        Deduplication is Core's
        :func:`~tmlt.core.utils.pandas_grouping.distinct_rows`, so that a null
        and a NaN in an object column stay two rows, as ``dropDuplicates`` leaves
        them on Spark.
        """
        return distinct_rows(coerce_pandas_schema_or_fail(self.df))

    def _spark_dataframe(self) -> DataFrame:
        """Generate the Spark dataframe corresponding to this operation.

        The rows are handed to Spark as Python values, taken from the frame the
        pandas path produces, so that the two backends are the same keys by
        construction. ``createDataFrame`` on the pandas frame itself would not
        do: it reads a column through numpy, and Spark's ``LongType`` rejects
        the ``numpy.int64`` that comes back as not being an integer.
        """
        frame = self._pandas_dataframe()
        columns = list(frame.columns)
        # Going through an object column is what turns each value into a Python
        # one: a nullable extension column's ``tolist`` yields numpy scalars and
        # ``pd.NA``, neither of which Spark accepts, while ``astype(object)``
        # yields Python ints, and the ``where`` writes each dtype's own spelling
        # of a null back as the ``None`` Spark reads as one.
        values = [
            frame[column].astype(object).where(frame[column].notna(), None).tolist()
            for column in columns
        ]
        rows = list(zip(*values)) if columns else []

        schema = analytics_to_spark_schema(Schema(self.schema()))
        spark = SparkSession.builder.getOrCreate()
        return spark.createDataFrame(
            spark.sparkContext.parallelize(rows, numSlices=2 + len(rows) // 1024),
            schema=schema,
        )

    def is_empty(self, backend: Backend = SPARK) -> bool:
        """Determine whether the dataframe corresponding to this operation is empty."""
        return len(self.df.index) == 0

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

        Counting means deduplicating, which is a pass over the frame, so a fast
        size is unavailable for the same reason it is on a Spark dataframe.
        """
        if not self.columns():
            return 1
        if fast:
            return None
        return len(self._pandas_dataframe().index)

    def __eq__(self, other: Any):
        """Determine if this KeySetOp is equal to another.

        Two of these are equal when they hold the same frame. Two distinct
        frames holding the same keys are *not* reported equal even though they
        are: a pandas frame's ``==`` is elementwise, and the cheap comparisons
        that could stand in for it -- :meth:`pandas.DataFrame.equals` among them
        -- do not distinguish a null from a NaN, which are different keys. A
        conservative answer here costs a full comparison in
        :meth:`~tmlt.analytics.KeySet.__eq__`, which is where a caller asked for
        one; a wrong one would silently merge two different KeySets.
        """
        if not isinstance(other, FromPandasDataFrame):
            return False
        return self.df is other.df

    def __hash__(self):
        """Hash this KeySetOp, consistently with :meth:`__eq__`."""
        return hash(id(self.df))

    def __str__(self):
        """Human-readable string representation."""
        cols = "\n  ".join(
            f"{col}: {desc.column_type}{' not NULL' if not desc.allow_null else ''}"
            for col, desc in self.schema().items()
        )
        return f"FromPandasDataFrame ({len(self.df.index)} rows)\n  {cols}"
