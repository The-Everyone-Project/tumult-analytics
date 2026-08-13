"""User-facing KeySet classes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import datetime
from collections.abc import Sequence
from functools import reduce
from typing import Any, Collection, Iterable, Mapping, Optional, cast, overload

import pandas as pd
from pyspark.sql import Column, DataFrame

from tmlt.analytics import AnalyticsInternalError
from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._coerce_pandas_schema import coerce_pandas_schema_or_fail
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, FrozenDict

from ._ops import (
    CrossJoin,
    Detect,
    Filter,
    FromPandasDataFrame,
    FromSparkDataFrame,
    FromTuples,
    Join,
    KeySetOp,
    Project,
    Subtract,
    Union,
    rewrite,
)
from ._ops._frames import (
    FrameKind,
    KeySetFrame,
    cast_to_schema,
    frame_count,
    frame_kind,
    frame_rows,
    select_columns,
)


class KeySet:
    """A class containing a set of values for specific columns.

       An introduction to KeySet initialization and manipulation can be found in
       the :ref:`group-by-queries` tutorial.

    .. warning::
        If a column has null values dropped or replaced, then Analytics
        will raise an error if you use a KeySet that contains a null value for
        that column.
    """

    def __init__(self, op_tree: KeySetOp, columns: Sequence[str]):
        """Constructor."""
        if not isinstance(op_tree, KeySetOp):
            raise ValueError(
                "KeySets should not be initialized using their constructor, "
                "use one of the various static initializer methods instead."
            )
        if op_tree.is_plan():
            raise AnalyticsInternalError(
                "KeySet should not be generated with a plan "
                "including partition selection."
            )
        if len(columns) != len(set(columns)):
            column_counts: dict[str, int] = {}
            for c in columns:
                column_counts[c] = column_counts.get(c, 0) + 1
            raise AnalyticsInternalError(
                "KeySet columns are not all distinct, duplicates are: "
                + " ".join(c for c, count in column_counts.items() if count > 1)
            )
        if op_tree.columns() != set(columns):
            raise AnalyticsInternalError(
                f"KeySet columns {columns} do not match "
                f"the columns of its op-tree {op_tree.columns()}."
            )
        self._op_tree = rewrite(op_tree)
        self._columns = columns
        # One materialized frame per kind of frame, rather than one frame: the
        # two hold the same keys but are different objects, and a caller that
        # asks for both should not pay for either twice. Note that a pandas
        # frame cannot be tested for with `if frame:` -- pandas raises on a
        # frame's truth value -- so these are always compared against None.
        self._dataframes: dict[FrameKind, KeySetFrame] = {}
        self._size: Optional[int] = None
        self._cached = False

    @staticmethod
    def from_dataframe(df: DataFrame) -> KeySet:
        """Creates a KeySet from a dataframe.

        This DataFrame should contain every combination of values being selected
        in the KeySet. If there are duplicate rows in the dataframe, only one
        copy of each will be kept.

        When creating KeySets with this method, it is the responsibility of the
        caller to ensure that the given dataframe remains valid for the lifetime
        of the KeySet. If the dataframe becomes invalid, for example because its
        Spark session is closed, this method or any uses of the resulting
        dataframe may raise exceptions or have other unanticipated effects.
        """
        return KeySet(FromSparkDataFrame(df), columns=df.columns)

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> KeySet:
        """Creates a KeySet from a pandas dataframe.

        This is :meth:`from_dataframe` for keys that are already in memory. The
        dataframe should contain every combination of values being selected in
        the KeySet; if there are duplicate rows, only one copy of each is kept.

        Unlike :meth:`from_dataframe`, this method copies the dataframe it is
        given, so the KeySet is unaffected by later changes to it -- and the
        resulting KeySet can be materialized on either backend, since its keys
        are in memory either way.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"A": ["a1", "a2"], "B": [1, 2]})
            >>> keyset = KeySet.from_pandas(df)
            >>> keyset.dataframe().sort("A", "B").toPandas()
                A  B
            0  a1  1
            1  a2  2
        """
        # coerce_pandas_schema_or_fail always returns a copy: from here on the
        # keys belong to the KeySet, and neither it nor the caller can change
        # what the other sees.
        coerced = coerce_pandas_schema_or_fail(df)
        return KeySet(FromPandasDataFrame(coerced), columns=list(coerced.columns))

    @staticmethod
    def from_tuples(
        tuples: Iterable[tuple[str | int | datetime.date | None, ...]],
        columns: Sequence[str],
    ) -> KeySet:
        """Creates a KeySet from a list of tuples and column names.

        Example:
            >>> tuples = [
            ...   ("a1", "b1"),
            ...   ("a2", "b1"),
            ...   ("a3", "b3"),
            ... ]
            >>> keyset = KeySet.from_tuples(tuples, ["A", "B"])
            >>> keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a2  b1
            2  a3  b3
        """
        # Deduplicate the tuples
        tuple_set = frozenset(tuples)
        for t in tuple_set:
            if not isinstance(t, tuple):
                raise ValueError(
                    "Each element of tuples must be a tuple, but got "
                    f"{type(t)} instead."
                )
            if len(t) != len(columns):
                raise ValueError(
                    "Tuples must contain the same number of values "
                    "as there are columns.\n"
                    f"Columns: {', '.join(columns)}\n"
                    f"Mismatched tuple: {', '.join(map(str, t))}"
                )

        column_types: dict[str, set[type]] = {col: set() for col in columns}
        for t in tuple_set:
            for i, col in enumerate(columns):
                column_types[col].add(type(t[i]))

        schema = {}
        for col in column_types:
            types = column_types[col]
            if types == set():
                raise ValueError(
                    "Unable to infer column types for an empty collection of values."
                )
            if types == {type(None)}:
                raise ValueError(
                    f"Column '{col}' contains only null values, unable to "
                    "infer its type."
                )
            if len(types - {type(None)}) != 1:
                raise ValueError(
                    f"Column '{col}' contains values of multiple types: "
                    f"{', '.join(t.__name__ for t in types)}"
                )

            (col_type,) = types - {type(None)}
            # KeySets can't include floats, so no need to worry about nan/inf values.
            schema[col] = ColumnDescriptor(
                ColumnType(col_type), allow_null=type(None) in types
            )

        return KeySet(
            FromTuples(tuple_set, FrozenDict.from_dict(schema)), columns=columns
        )

    @staticmethod
    def from_dict(
        domains: Mapping[
            str,
            Iterable[Optional[str]]
            | Iterable[Optional[int]]
            | Iterable[Optional[datetime.date]],
        ],
    ) -> KeySet:
        """Creates a KeySet from a dictionary.

        The ``domains`` dictionary should map column names to the desired values
        for those columns. The KeySet returned is the cross-product of those
        columns. Duplicate values in the column domains are allowed, but only
        one of the duplicates is kept.

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": ["b1", "b2"],
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
        """
        if len(domains) == 0:
            return KeySet.from_tuples([], columns=[])

        domain_keysets = (
            KeySet.from_tuples(((v,) for v in values), columns=[col])
            for col, values in domains.items()
        )
        return reduce(lambda l, r: l * r, domain_keysets)

    # TODO(opendp/tumult-analytics#124): Make this public and fill in its docstring
    #     with an example of usage.
    @staticmethod
    def _detect(columns: Sequence[str]) -> KeySetPlan:
        """Detect the keys for a collection of columns."""
        return KeySetPlan(Detect(frozenset(columns)), columns=columns)

    @overload
    def __mul__(self, other: KeySet) -> KeySet: ...

    @overload
    def __mul__(self, other: KeySetPlan) -> KeySetPlan: ...

    def __mul__(self, other):
        r"""The Cartesian product of the two KeySet factors.

        Example:
            >>> keyset1 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset2 = KeySet.from_tuples([("b1",), ("b2",)], columns=["B"])
            >>> product = keyset1 * keyset2
            >>> product.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
        """
        # TODO(opendp/tumult-analytics#124): Mention the behavior of this method in
        #     terms of its interation with KeySetPlans in the docstring and
        #     the below error message.

        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet multiplication expected another KeySet, not "
                f"{type(other).__qualname__}, as right-hand value."
            )
        if isinstance(other, KeySet):
            return KeySet(
                CrossJoin((self._op_tree, other._op_tree)),
                columns=self.columns() + other.columns(),
            )
        else:
            return KeySetPlan(
                CrossJoin((self._op_tree, other._op_tree)),
                columns=self.columns() + other.columns(),
            )

    def __sub__(self, other: KeySet) -> KeySet:
        """Remove rows in this that match rows in another KeySet.

        Equivalent to a left anti-join between this KeySet and ``other``.

        ``other`` must have a subset of the columns of this KeySet. Any rows in this
        KeySet where the values in those columns match values in ``other`` are removed.

        Example:
            >>> keyset1 = KeySet.from_dict({"A": [1, 2], "B": ["a", "b"]})
            >>> result = keyset1 - KeySet.from_tuples([(1, "b")], columns=["A", "B"])
            >>> result.dataframe().sort("A", "B").toPandas()
               A  B
            0  1  a
            1  2  a
            2  2  b
        """
        return KeySet(Subtract(self._op_tree, other._op_tree), self.columns())

    def __getitem__(self, desired_columns: str | Sequence[str]) -> KeySet:
        """``KeySet[col, col, ...]`` returns a KeySet with those columns only.

        The returned KeySet contains all unique combinations of values in the
        given columns that were present in the original KeySet.

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": ["b1", "b2"],
            ...     "C": ["c1", "c2"],
            ...     "D": [0, 1, 2, 3]
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> a_b_keyset = keyset["A", "B"]
            >>> a_b_keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
            >>> a_b_keyset = keyset[["A", "B"]]
            >>> a_b_keyset.dataframe().sort("A", "B").toPandas()
                A   B
            0  a1  b1
            1  a1  b2
            2  a2  b1
            3  a2  b2
            >>> a_keyset = keyset["A"]
            >>> a_keyset.dataframe().sort("A").toPandas()
                A
            0  a1
            1  a2
        """
        if isinstance(desired_columns, str):
            desired_columns = [desired_columns]

        if len(desired_columns) != len(set(desired_columns)):
            column_counts: dict[str, int] = {}
            for c in desired_columns:
                column_counts[c] = column_counts.get(c, 0) + 1
            raise ValueError(
                "Selected columns are not all distinct, duplicates are: "
                + " ".join(c for c, count in column_counts.items() if count > 1)
            )

        return KeySet(
            Project(self._op_tree, frozenset(desired_columns)), columns=desired_columns
        )

    @overload
    def join(self, other: KeySet) -> KeySet: ...

    @overload
    def join(self, other: KeySetPlan) -> KeySetPlan: ...

    def join(self, other):
        r"""The inner natural join of two KeySet objects.

        The two KeySets are inner joined on columns with matching names,
        treating nulls as equal to one another.

        Example:
            >>> keyset1 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset2 = KeySet.from_tuples(
            ...     [("a2", "b1"), ("a3", "b2")], columns=["A", "B"]
            ... )
            >>> keyset1.join(keyset2).dataframe().sort("A", "B").toPandas()
                A   B
            0  a2  b1
        """
        # TODO(opendp/tumult-analytics#124): Mention the behavior of this method in
        #     terms of its interation with KeySetPlans in the docstring and
        #     the below error message.

        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                f"KeySet join expected another KeySet, not {type(other).__qualname__}."
            )
        if isinstance(other, KeySet):
            return KeySet(
                Join(self._op_tree, other._op_tree),
                columns=list(dict.fromkeys(self.columns() + other.columns())),
            )
        else:
            return KeySetPlan(
                Join(self._op_tree, other._op_tree),
                columns=list(dict.fromkeys(self.columns() + other.columns())),
            )

    def filter(self, condition: Column | str) -> KeySet:
        """Filters this KeySet using some condition.

        This method accepts the same syntax as
        :meth:`pyspark.sql.DataFrame.filter`: valid conditions are those that
        can be used in a `WHERE clause
        <https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-where.html>`__
        in Spark SQL. Examples of valid conditions include:

        * ``age < 42``
        * ``age BETWEEN 17 AND 42``
        * ``age < 42 OR (age < 60 AND gender IS NULL)``
        * ``LENGTH(name) > 17``
        * ``favorite_color IN ('blue', 'red')``

        Example:
            >>> domains = {
            ...     "A": ["a1", "a2"],
            ...     "B": [0, 1, 2, 3],
            ... }
            >>> keyset = KeySet.from_dict(domains)
            >>> filtered_keyset = keyset.filter("B < 2")
            >>> filtered_keyset.dataframe().sort("A", "B").toPandas()
                A  B
            0  a1  0
            1  a1  1
            2  a2  0
            3  a2  1
            >>> import pyspark.sql.functions as sf
            >>> filtered_keyset = keyset.filter(sf.col("A") != "a1")
            >>> filtered_keyset.dataframe().sort("A", "B").toPandas()
                A  B
            0  a2  0
            1  a2  1
            2  a2  2
            3  a2  3

        Args:
            condition: A string of SQL expressions or a PySpark
                :class:`~pyspark.sql.Column` specifying the filter to apply to
                the data.
        """
        return KeySet(Filter(self._op_tree, condition), columns=self.columns())

    @overload
    def union(self, other: KeySet) -> KeySet: ...

    @overload
    def union(self, other: KeySetPlan) -> KeySetPlan: ...

    def union(self, other):
        """Compute the union of this KeySet with another KeySet.

        The two KeySets must have exactly the same columns. The result contains
        all unique rows that appear in either KeySet.

        Example:
            >>> keyset1 = KeySet.from_tuples([(1, "a"), (2, "b")], columns=["A", "B"])
            >>> keyset2 = KeySet.from_tuples([(2, "b"), (3, "c")], columns=["A", "B"])
            >>> union_result = keyset1.union(keyset2)
            >>> union_result.dataframe().sort("A", "B").toPandas()
               A  B
            0  1  a
            1  2  b
            2  3  c
        """
        # TODO(opendp/tumult-analytics#124): Mention the behavior of this method in
        #     terms of its interation with KeySetPlans in the docstring and
        #     the below error message.

        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                f"KeySet union expected another KeySet, not {type(other).__qualname__}."
            )
        if isinstance(other, KeySet):
            return KeySet(
                Union(self._op_tree, other._op_tree),
                columns=self.columns(),
            )
        else:
            return KeySetPlan(
                Union(self._op_tree, other._op_tree),
                columns=self.columns(),
            )

    def columns(self) -> list[str]:
        """Returns the list of columns used in this KeySet."""
        return list(self._columns)

    def schema(self) -> dict[str, ColumnDescriptor]:
        """Returns the KeySet's schema.

        Example:
            >>> keys = [
            ...     ("a1", 0),
            ...     ("a2", None),
            ... ]
            >>> keyset = KeySet.from_tuples(keys, columns=["A", "B"])
            >>> schema = keyset.schema()
            >>> schema # doctest: +NORMALIZE_WHITESPACE
            {'A': ColumnDescriptor(column_type=ColumnType.VARCHAR, allow_null=False, allow_nan=False, allow_inf=False),
             'B': ColumnDescriptor(column_type=ColumnType.INTEGER, allow_null=True, allow_nan=False, allow_inf=False)}
        """
        schema = self._op_tree.schema()
        return {c: schema[c] for c in self.columns()}  # Reorder to match self.columns()

    def _materialize(self, kind: FrameKind) -> KeySetFrame:
        """Returns this KeySet's keys as a frame of the given kind.

        The frame is cached per kind, so asking twice evaluates the op-tree
        once, and asking for the other kind does not discard the first.
        """
        frame = self._dataframes.get(kind)
        if frame is not None:
            return frame

        frame = self._op_tree.frame(kind)
        if set(frame.columns) != set(self.columns()):
            raise AnalyticsInternalError(
                f"KeySet op-tree dataframe produced columns {list(frame.columns)} "
                f"that do not match its expected columns {self.columns()}."
            )
        # Reorder to match self.columns()
        frame = select_columns(frame, self.columns())
        if isinstance(frame, pd.DataFrame):
            # ...and put every column in the canonical dtype for its descriptor,
            # so that the frame is in the domain this KeySet's schema describes.
            frame = cast_to_schema(frame, self.schema())
        elif self._cached:
            frame.cache()

        self._dataframes[kind] = frame
        return frame

    def dataframe(self, backend: Backend = SPARK) -> DataFrame:
        """Returns the dataframe associated with this KeySet.

        This dataframe contains every combination of values being selected in
        the KeySet, and its rows are guaranteed to be unique.

        Args:
            backend: The backend whose kind of dataframe to produce. Defaults to
                Spark, and a Spark :class:`~pyspark.sql.DataFrame` is what this
                method is for; :meth:`to_pandas` is the typed way to ask for the
                keys in memory.
        """
        return cast(DataFrame, self._materialize(frame_kind(backend)))

    def to_pandas(self) -> pd.DataFrame:
        """Returns this KeySet's keys as a pandas dataframe.

        This is :meth:`dataframe` with the keys built in memory instead of in
        Spark: the same keys, in the same column order, with each column in the
        dtype this KeySet's :meth:`schema` calls for.

        Not every KeySet can be built this way. One made from a Spark dataframe
        cannot -- collecting a distributed frame is a decision for its owner to
        make -- and neither can one that has been filtered with :meth:`filter`,
        since a filter condition is a Spark expression. Both raise
        :class:`~tmlt.analytics._backends.NotSupportedByBackend`.

        Example:
            >>> keyset = KeySet.from_tuples([("a1", 1), ("a2", 2)], ["A", "B"])
            >>> keyset.to_pandas().sort_values(["A", "B"], ignore_index=True)
                A  B
            0  a1  1
            1  a2  2
        """
        return cast(pd.DataFrame, self._materialize(FrameKind.PANDAS))

    def size(self, backend: Backend = SPARK) -> int:
        """Returns the number of groups included in this KeySet.

        Note that in some situations this method may need to count the elements
        in the KeySet's dataframe, which can be extremely slow.

        Args:
            backend: The backend to count on, where counting requires
                materializing the KeySet. The answer does not depend on it.
        """
        if self._size is None:
            # A frame this KeySet has already built has the answer in it, and
            # asking the op-tree instead would build a second one -- on pandas,
            # redoing the whole coerce-and-deduplicate. Which frame does not
            # matter: both hold the same keys.
            #
            # Except for a KeySet with no columns, which is the total
            # aggregation: it stands for the one group that is the whole table,
            # and is spelled as a frame with no rows. Its size is one and its
            # frame's row count is zero, so that one has to be asked of the
            # op-tree.
            cached = next(iter(self._dataframes.values()), None)
            if cached is not None and self.columns():
                self._size = frame_count(cached)
            else:
                self._size = self._op_tree.size(fast=False, backend=backend)
        return self._size

    def cache(self) -> None:
        """Caches the KeySet's dataframe in memory.

        This is a Spark cache, and applies to the Spark dataframe alone. A
        pandas frame is already in memory, and :meth:`to_pandas` already keeps
        the one it built.
        """
        # Caching an already-cached dataframe produces a warning, so avoid doing
        # it by only caching the dataframe when the KeySet isn't already cached.
        if not self._cached:
            self._cached = True
            spark_dataframe = self._dataframes.get(FrameKind.SPARK)
            if spark_dataframe is not None:
                cast(DataFrame, spark_dataframe).cache()

    def uncache(self) -> None:
        """Removes the KeySet's dataframe from memory and disk.

        Like :meth:`cache`, this concerns the Spark dataframe alone.
        """
        self._cached = False
        spark_dataframe = self._dataframes.get(FrameKind.SPARK)
        if spark_dataframe is not None:
            cast(DataFrame, spark_dataframe).unpersist()

    def is_equivalent(self, other: KeySet | KeySetPlan) -> Optional[bool]:
        """Determine if another KeySet is equivalent to this one, if possible.

        This method is an alternative to :meth:`KeySet.__eq__` which is
        guaranteed to never evaluate the full KeySet dataframe. This ensures
        that it is never time-consuming to call, but also means that it cannot
        always determine if two KeySets are equivalent. If the KeySets are
        neither definitely equivalent nor easily shown to not be equivalent,
        this method returns ``None``.
        """
        # A KeySet and a KeySetPlan can't be equivalent, but allowing either to
        # be passed could avoid some user confusion about what is/is not a plan.
        if not isinstance(other, KeySet):
            return False

        if self._op_tree == other._op_tree:
            return True

        # Differing column nullability doesn't necessarily mean that two KeySets
        # are different, as the one with the nullable column might not actually
        # contain any nulls. Thus, only consider column type when comparing
        # schemas.
        if {c: d.column_type for c, d in self.schema().items()} != {
            c: d.column_type for c, d in other.schema().items()
        }:
            return False

        return None

    def __eq__(self, other: Any):
        """Determine if another KeySet is equal to this one.

        Two KeySets are equal if they contain the same values for the same
        columns; the rows and columns may appear in any order.

        When :meth:`is_equivalent` cannot answer from the op-trees alone, the
        keys have to be built and compared. Which engine builds them is decided
        here rather than passed in -- ``==`` takes no arguments, and a KeySet is
        compared in places that know nothing about backends: two frozen
        :class:`~tmlt.analytics.Query` dataclasses holding KeySets, or a KeySet
        looked up in a dict, whose :meth:`__hash__` hashes only the schema and
        so leaves every collision to be settled here.

        So the choice is inferred. If both op-trees can be materialized in
        memory -- a pure walk of the two trees, which builds nothing -- the
        comparison is done on pandas frames; otherwise it is done on Spark, as
        it always was. That keeps a comparison of two ordinary KeySets from
        starting a JVM, which on a pandas Session was the whole cost: the trees
        are usually small literal data that was in memory to begin with, and
        the Spark path would have shipped it out to be brought back.

        Example:
            >>> ks1 = KeySet.from_dict({"A": [1, 2], "B": [3, 4]})
            >>> ks2 = KeySet.from_dict({"B": [3, 4], "A": [1, 2]})
            >>> ks3 = KeySet.from_dict({"B": [4, 3], "A": [2, 1]})
            >>> ks4 = KeySet.from_dict({"B": [4, 5], "A": [1, 2]})
            >>> ks1 == ks2
            True
            >>> ks1 == ks3
            True
            >>> ks1 == ks4
            False
        """
        if not isinstance(other, KeySet):
            return False

        equivalent = self.is_equivalent(other)
        if equivalent is not None:
            return equivalent

        # Reorder columns between the two dataframes to match
        columns = self.columns()
        if not (
            self._op_tree.unsupported_frame_ops(FrameKind.PANDAS)
            or other._op_tree.unsupported_frame_ops(FrameKind.PANDAS)
        ):
            # Multisets of null-tagged rows, so that a duplicated row and a null
            # turning into a NaN are both differences; see frame_rows.
            return frame_rows(self.to_pandas(), columns) == frame_rows(
                other.to_pandas(), columns
            )

        self_df = self.dataframe().select(*columns)
        other_df = other.dataframe().select(*columns)
        # other_df should contain all rows in self_df
        if self_df.exceptAll(other_df).count() != 0:
            return False
        # and vice versa
        if other_df.exceptAll(self_df).count() != 0:
            return False
        return True

    def __hash__(self):
        """Hash the KeySet based on its schema."""
        return hash(FrozenDict.from_dict(self.schema()))

    def _decompose(
        self, split_columns: Optional[Collection[str]] = None
    ) -> tuple[list[KeySet], list[KeySet]]:
        """Decompose this KeySet into a collection of factors and subtracted values.

        Express this KeySet as a collection of factors and subtracted values.
        When ``split_columns`` is provided, this decomposition will assume that
        the factors are independent of those columns. For example, splitting on
        ``B`` would allow this decomposition:

        .. code-block::

           AB.join(BC) -> factors=[AB, BC], subtracted_values=[]

        Without splitting, this would instead decompose as

        .. code-block::

           AB.join(BC) -> factors=[AB.join(BC)], subtracted_values=[]


        If ``split_columns`` is not provided, the ``(factors, subtracted_values)``
        tuple has the property that

        .. code-block::

           reduce(
               lambda l,r: l - r,
               subtracted_values,
               initial=reduce(lambda l,r: l * r, factors)
           )

        produces the same set of keys the original KeySet contained. If it is
        provided, then a similar property applies, but ``l * r`` must be
        replaced with ``l.join(r)`` if ``l`` and ``r`` have overlapping
        columns. These overlapping columns must be a subset of the split
        columns, other columns must still only appear in one factor.
        """

        def as_keyset(op: KeySetOp) -> KeySet:
            return KeySet(op, columns=[c for c in self.columns() if c in op.columns()])

        f_optrees, sv_optrees = self._op_tree.decompose(split_columns or [])
        return [as_keyset(f) for f in f_optrees], [as_keyset(sv) for sv in sv_optrees]


class KeySetPlan:
    """A plan for computing a KeySet based on values in a table.

    A :class:`.KeySetPlan` describes a plan for computing a set of group keys
    that may be used when computing a group-by query. This is similar to what a
    :class:`.KeySet` represents, with one key difference: a :class:`.KeySetPlan`
    requires spending some privacy budget with a :class:`.Session` to get back a
    specific :class:`.KeySet` for a particular table. The :class:`.KeySetPlan`
    alone cannot produce an equivalent dataframe and doesn't have a fixed
    schema.
    """

    def __init__(self, op_tree: KeySetOp, columns: Sequence[str]):
        """Constructor."""
        if not isinstance(op_tree, KeySetOp):
            raise ValueError(
                "KeySets should not be initialized using their constructor, "
                "use one of the various static initializer methods instead."
            )
        if not op_tree.is_plan():
            raise AnalyticsInternalError(
                "KeySetPlan must be generated with a plan "
                "including partition selection."
            )
        if op_tree.columns() != set(columns):
            raise AnalyticsInternalError(
                f"KeySet columns {columns} do not match "
                f"the columns of its op-tree {op_tree.columns()}."
            )
        self._op_tree = rewrite(op_tree)
        self._columns = columns

        if not self._op_tree.is_plan():
            raise AnalyticsInternalError(
                "KeySetPlan op-tree unexpectedly became not a plan."
            )

    def columns(self) -> list[str]:
        """Returns the list of columns used in this KeySetPlan."""
        return list(self._columns)

    def __mul__(self, other: KeySet | KeySetPlan) -> KeySetPlan:
        """The Cartesian product of the two KeySet or KeySetPlan factors.

        Example:
            >>> keyset1 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset2 = KeySet._detect(["B"])
            >>> product = keyset1 * keyset2
            >>> product.columns()
            ['A', 'B']
        """
        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet multiplication expected another KeySet or KeySetPlan, not "
                f"{type(other).__qualname__}, as right-hand value."
            )
        return KeySetPlan(
            CrossJoin((self._op_tree, other._op_tree)),
            columns=self.columns() + other.columns(),
        )

    def __sub__(self, other: KeySet) -> KeySetPlan:
        """Remove rows in this that match rows in another KeySet.

        Equivalent to a left anti-join between this and other.

        other must have a subset of the columns of this KeySet. Any rows in this
        KeySet where the values in those columns match values in other are removed.

        Example:
            >>> keyset1 = KeySet._detect(["A", "B"])
            >>> result = keyset1 - KeySet.from_tuples([(1, "b")], columns=["A", "B"])
            >>> result.columns()
            ['A', 'B']
        """
        return KeySetPlan(
            Subtract(self._op_tree, other._op_tree), columns=self.columns()
        )

    def __getitem__(self, desired_columns: str | Sequence[str]) -> KeySet | KeySetPlan:
        """``KeySetPlan[col, col, ...]`` returns a KeySetPlan with those columns only.

        The returned KeySetPlan contains all unique combinations of values in the
        given columns that were present in the original KeySetPlan.

        Example:
            >>> keyset = KeySet._detect(["A", "B", "C"])
            >>> keyset["A", "B"].columns()
            ['A', 'B']
            >>> keyset[["A", "B"]].columns()
            ['A', 'B']
            >>> keyset["A"].columns()
            ['A']
        """
        if isinstance(desired_columns, str):
            desired_columns = [desired_columns]

        if len(desired_columns) != len(set(desired_columns)):
            column_counts: dict[str, int] = {}
            for c in desired_columns:
                column_counts[c] = column_counts.get(c, 0) + 1
            raise ValueError(
                "Selected columns are not all distinct, duplicates are: "
                + " ".join(c for c, count in column_counts.items() if count > 1)
            )
        # Projecting is the only operation that can turn a KeySetPlan into a
        # KeySet (by dropping the detected columns without a join or similar
        # operation that would force them to be computed anyway), so it works a
        # bit differently. Do the op-tree rewrite early and use its output to
        # figure out whether the result is a plan or not.
        op_tree = rewrite(Project(self._op_tree, frozenset(desired_columns)))
        if op_tree.is_plan():
            return KeySetPlan(op_tree, columns=desired_columns)
        return KeySet(op_tree, columns=desired_columns)

    def join(self, other: KeySet | KeySetPlan) -> KeySetPlan:
        r"""The inner natural join of two KeySet or KeySetPlan objects.

        The two KeySets are inner joined on columns with matching names,
        treating nulls as equal to one another. Joining a :class:`KeySetPlan` to
        any other value always produces a :class:`KeySetPlan`.

        Example:
            >>> keyset1 = KeySet._detect(["A", "B"])
            >>> keyset2 = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
            >>> keyset1.join(keyset2).columns()
            ['A', 'B']
        """
        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet join expected another KeySet or KeySetPlan, not "
                f"{type(other).__qualname__}."
            )

        return KeySetPlan(
            Join(self._op_tree, other._op_tree),
            columns=list(dict.fromkeys(self.columns() + other.columns())),
        )

    def filter(self, condition: Column | str) -> KeySetPlan:
        """Filters this KeySetPlan using some condition.

        This method accepts the same syntax as
        :meth:`pyspark.sql.DataFrame.filter`: valid conditions are those that
        can be used in a `WHERE clause
        <https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-where.html>`__
        in Spark SQL. Examples of valid conditions include:

        * ``age < 42``
        * ``age BETWEEN 17 AND 42``
        * ``age < 42 OR (age < 60 AND gender IS NULL)``
        * ``LENGTH(name) > 17``
        * ``favorite_color IN ('blue', 'red')``

        Example:
            >>> keyset = KeySet._detect(["A", "B"])
            >>> filtered_keyset = keyset.filter("B < 2")
            >>> filtered_keyset.columns()
            ['A', 'B']
        """
        return KeySetPlan(Filter(self._op_tree, condition), columns=self.columns())

    def union(self, other: KeySet | KeySetPlan) -> KeySetPlan:
        """Compute the union of this KeySetPlan with another KeySet or KeySetPlan.

        The two KeySets must have exactly the same columns. The result contains
        all unique rows that appear in either KeySet (duplicates are removed,
        following SQL UNION behavior). Unioning a :class:`KeySetPlan` with
        any other value always produces a :class:`KeySetPlan`.

        Example:
            >>> keyset1 = KeySet._detect(["A", "B"])
            >>> keyset2 = KeySet.from_tuples([(1, "a")], columns=["A", "B"])
            >>> keyset1.union(keyset2).columns()
            ['A', 'B']
        """
        if not isinstance(other, (KeySet, KeySetPlan)):
            raise ValueError(
                "KeySet union expected another KeySet or KeySetPlan, not "
                f"{type(other).__qualname__}."
            )

        return KeySetPlan(
            Union(self._op_tree, other._op_tree),
            columns=self.columns(),
        )

    def is_equivalent(self, other: KeySet | KeySetPlan) -> Optional[bool]:
        """Determine if another KeySetPlan is equivalent to this one, if possible.

        This method is guaranteed to never evaluate any underlying dataframe,
        ensuring that it is never time-consuming to call. However, this means
        means that it cannot always determine if two KeySetPlans are
        equivalent. If the KeySetPlans are neither definitely equivalent nor
        easily shown to not be equivalent, this method returns ``None``.

        Example:
            >>> ks1 = KeySet._detect(["A", "B"]) * KeySet.from_dict({"C": [1,2]})
            >>> ks2 = KeySet.from_dict({"C": [1,2]}) * KeySet._detect(["A", "B"])
            >>> ks1.is_equivalent(ks2)
            True
        """
        # A KeySet and a KeySetPlan can't be equivalent, but allowing either to
        # be passed could avoid some user confusion about what is/is not a plan.
        if not isinstance(other, KeySetPlan):
            return False

        if self._op_tree == other._op_tree:
            return True

        if self.columns() != other.columns():
            return False

        return None

    def __eq__(self, other: Any):
        r"""Determine if another :class:`KeySetPlan` is equal to this one.

        Unlike for :meth:`KeySet.__eq__`, there is no fallback full-dataframe
        comparison here -- this method only relies on :meth:`is_equivalent`, and
        returns ``False`` if it cannot determine whether the two
        :class:`KeySetPlan`\ s are equivalent.
        """
        if not isinstance(other, KeySetPlan):
            return False

        return bool(self.is_equivalent(other))
