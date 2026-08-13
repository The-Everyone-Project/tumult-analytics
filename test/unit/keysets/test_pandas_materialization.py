"""Unit tests for materializing a KeySet as a pandas dataframe.

A :class:`~tmlt.analytics.KeySet` describes a set of keys without saying what
holds them, and the same op-tree materializes on either backend. What has to be
true of the two results is that they are the *same keys* -- so most of what is
tested here is an equality between backends, computed over a corpus of KeySets
that between them use every operation that can be materialized.

Comparing them takes some care. ``toPandas`` on a Spark frame turns a nullable
integer column into a ``float64`` one, so a null arrives as a NaN and the two
become indistinguishable; and a pandas frame spells a null three ways
(``None``, ``pd.NA``, ``float("nan")``), only two of which mean "null". Both
sides are therefore reduced to a multiset of rows of tagged Python values by
:func:`~tmlt.analytics.keyset._ops._frames.frame_rows`, which keeps a null and a
NaN apart, before being compared. That function lives in ``src`` rather than
here because :meth:`KeySet.__eq__` decides equality with it.

Half of this module is such a comparison, and so needs a Spark session; the other
half asks only about the pandas frame. The comparisons carry ``@pytest.mark.spark``
so that the ``test-nojvm`` nox session can deselect them and run the pandas half of
the file. The marker is written by hand here rather than left to the collection
hook in ``test/conftest.py``, which infers it from the ``spark`` fixture: these
tests reach Spark through ``KeySet.dataframe()`` inside the test body, which no
fixture closure shows. One of them -- :func:`test_size_matches` -- is marked as a
whole although a few of its corpus cases answer without Spark, because per-case
marking would mean marks on a corpus that four tests share.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import datetime
import math
from collections import Counter
from dataclasses import replace
from typing import Any, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame
from tmlt.core.domains.pandas_domains import (
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.pandas_transformations.groupby import GroupBy
from tmlt.core.utils.testing import Case, parametrize

from tmlt.analytics import KeySet
from tmlt.analytics._backends import SPARK, Backend, NotSupportedByBackend
from tmlt.analytics._schema import Schema, analytics_to_pandas_dtypes
from tmlt.analytics.keyset._ops import KeySetOp
from tmlt.analytics.keyset._ops._frames import FrameKind, frame_kind, frame_rows

PANDAS = replace(SPARK, name="pandas", dataframe_type=pd.DataFrame)
"""A stand-in for the pandas backend descriptor, until the real one exists.

KeySet materialization reads exactly two things off a backend: the type its
tables are carried in, which is what selects the implementation, and its name,
which appears in the error when an operation has no implementation for it. The
rest of this descriptor is still Spark's, and nothing below reads it -- which is
also why binding the real pandas backend needs no change to the keyset code.
"""

NEITHER = replace(SPARK, name="something else", dataframe_type=int)
"""A backend whose tables are in neither kind of frame."""


################################################################################
# Comparing the two backends' output
################################################################################

_rows = frame_rows
"""The multiset-of-tagged-rows comparison, which lives in src.

It is what :meth:`KeySet.__eq__` decides equality on when both op-trees can be
built in memory, so it is engine code rather than test code; the tests here read
it from where it lives so that the two cannot come apart. Aliased because it is
used in nearly every assertion below.
"""


def assert_same_keys(keyset: KeySet) -> None:
    """Asserts that a KeySet holds the same keys on both backends."""
    columns = keyset.columns()
    spark_rows = _rows(keyset.dataframe(), columns)
    pandas_rows = _rows(keyset.to_pandas(), columns)
    assert pandas_rows == spark_rows, (
        f"KeySet materialized differently on the two backends.\n"
        f"Only on Spark:  {spark_rows - pandas_rows}\n"
        f"Only on pandas: {pandas_rows - spark_rows}"
    )


################################################################################
# The corpus
################################################################################

_DATE = datetime.date(2024, 3, 17)
_OTHER_DATE = datetime.date(1999, 12, 31)

# A keyset of every column type a keyset may hold, with a null in each column
# that can have one.
MIXED = KeySet.from_tuples(
    [
        ("a1", 1, _DATE),
        ("a2", None, _OTHER_DATE),
        (None, 3, _DATE),
        ("a1", 3, None),
    ],
    columns=["A", "B", "C"],
)
NULLABLE_A = KeySet.from_tuples([("a1",), ("a2",), (None,)], columns=["A"])
NON_NULL_B = KeySet.from_tuples([(1,), (2,), (3,)], columns=["B"])
# Large enough that the in-memory cross-join rule leaves it alone, so that
# CrossJoin's own implementation is what gets exercised.
WIDE_D = KeySet.from_tuples([(value,) for value in range(200)], columns=["D"])
TOTAL = KeySet.from_tuples([], columns=[])
FROM_PANDAS = KeySet.from_pandas(
    pd.DataFrame(
        {
            "A": ["a1", "a2", None],
            "B": pd.array([1, None, 3], dtype="Int64"),
            "C": [_DATE, _OTHER_DATE, None],
        }
    )
)


def _corpus() -> list[Case]:
    """One case per way of building a KeySet that can be materialized."""
    return [
        Case("from_tuples")(keyset=MIXED),
        Case("from_tuples_nulls")(keyset=NULLABLE_A),
        Case("from_tuples_empty")(
            keyset=KeySet.from_tuples([("a1",)], columns=["A"])
            - KeySet.from_tuples([("a1",)], columns=["A"])
        ),
        Case("total_aggregation")(keyset=TOTAL),
        Case("in_memory_cross_join")(keyset=NULLABLE_A * NON_NULL_B),
        Case("cross_join")(keyset=MIXED["A", "B"] * WIDE_D),
        Case("cross_join_with_total")(keyset=NULLABLE_A * TOTAL),
        Case("project")(keyset=MIXED["A", "C"]),
        Case("project_one_column")(keyset=MIXED["B"]),
        Case("union")(
            keyset=MIXED["A", "B"].union(
                KeySet.from_tuples([("a3", 9), ("a1", 1)], columns=["A", "B"])
            )
        ),
        Case("union_mixed_nullability")(
            keyset=NON_NULL_B.union(KeySet.from_tuples([(4,), (None,)], columns=["B"]))
        ),
        Case("join")(keyset=NULLABLE_A.join(MIXED)),
        Case("join_on_nullable_column")(
            keyset=MIXED["A", "B"].join(
                KeySet.from_tuples([(None, "x"), ("a1", "y")], columns=["A", "E"])
            )
        ),
        Case("subtract")(keyset=MIXED - KeySet.from_tuples([("a1",)], columns=["A"])),
        Case("subtract_null_key")(
            keyset=NULLABLE_A - KeySet.from_tuples([("a1",), (None,)], columns=["A"])
        ),
        Case("nested")(
            keyset=(NULLABLE_A * NON_NULL_B)
            .join(MIXED)["A", "B"]
            .union(NULLABLE_A * NON_NULL_B)
        ),
        Case("from_pandas")(keyset=FROM_PANDAS),
        Case("from_pandas_projected")(keyset=FROM_PANDAS["A", "B"]),
        Case("from_pandas_joined")(keyset=FROM_PANDAS.join(NULLABLE_A)),
        Case("from_pandas_empty")(keyset=KeySet.from_pandas(pd.DataFrame())),
    ]


@parametrize(_corpus())
@pytest.mark.spark
def test_same_keys_on_both_backends(keyset: KeySet) -> None:
    """Every KeySet in the corpus holds the same keys on both backends."""
    assert_same_keys(keyset)


@parametrize(_corpus())
def test_column_order_matches(keyset: KeySet) -> None:
    """The pandas frame's columns are the KeySet's columns, in its order."""
    assert list(keyset.to_pandas().columns) == keyset.columns()


@parametrize(_corpus())
def test_dtypes_match_schema(keyset: KeySet) -> None:
    """Every column's dtype is the canonical one for its column descriptor."""
    assert keyset.to_pandas().dtypes.to_dict() == analytics_to_pandas_dtypes(
        Schema(keyset.schema())
    )


@parametrize(_corpus())
@pytest.mark.spark
def test_size_matches(keyset: KeySet) -> None:
    """A KeySet's size is the same computed on either backend."""
    spark_size = KeySet(keyset._op_tree, keyset.columns()).size(SPARK)
    pandas_size = KeySet(keyset._op_tree, keyset.columns()).size(PANDAS)
    assert pandas_size == spark_size
    if keyset.columns():
        assert pandas_size == sum(_rows(keyset.to_pandas(), keyset.columns()).values())


def test_corpus_covers_every_materializable_op() -> None:
    """The corpus really does exercise every operation with a pandas path."""

    def op_names(op: KeySetOp) -> set[str]:
        names = {type(op).__name__}
        for child in op.children():
            names |= op_names(child)
        return names

    covered: set[str] = set()
    for case in _corpus():
        covered |= op_names(case.args["keyset"]._op_tree)
    assert covered == {
        "CrossJoin",
        "FromPandasDataFrame",
        "FromTuples",
        "InMemoryCrossJoin",
        "Join",
        "Project",
        "Subtract",
        "Union",
    }


################################################################################
# R-K1: the total-aggregation KeySet
################################################################################


@pytest.mark.spark
def test_total_aggregation_shape() -> None:
    """A KeySet with no columns materializes to an empty frame on both backends.

    A KeySet with no columns is a total aggregation: it stands for the single
    group that is the whole table, so its size is one even though it has no
    rows. Spark spells that as a dataframe with no rows and no columns, and the
    pandas materialization has to produce exactly the same shape.
    """
    assert TOTAL.size() == 1

    spark_frame = TOTAL.dataframe()
    assert (spark_frame.count(), len(spark_frame.columns)) == (0, 0)

    pandas_frame = TOTAL.to_pandas()
    assert pandas_frame.shape == (0, 0)
    assert list(pandas_frame.columns) == []


def test_total_aggregation_is_what_core_expects() -> None:
    """Core's pandas GroupBy reads the empty frame as a total aggregation.

    ``GroupBy`` and ``PandasGroupedTable`` both take group keys with no columns
    to mean "aggregate the whole table", and reject a frame that has rows but no
    columns. The frame produced above is the accepted spelling: grouping by it
    yields no groupby columns and a single output row.
    """
    groupby = GroupBy(
        input_domain=PandasTableDomain({"A": PandasStringColumnDescriptor()}),
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=TOTAL.to_pandas(),
    )
    assert groupby.groupby_columns == []
    assert groupby.group_keys is None

    aggregated = groupby(pd.DataFrame({"A": ["x", "y"]})).agg(
        len, fill_value=0, output_column="count"
    )
    assert aggregated["count"].tolist() == [2]


def test_cross_join_of_total_aggregations() -> None:
    """Crossing two total aggregations is still a total aggregation."""
    assert (TOTAL * TOTAL).to_pandas().shape == (0, 0)


################################################################################
# Nulls
################################################################################


@parametrize(
    Case("null_string")(values=[("a1",), (None,)], column="A"),
    Case("null_integer")(values=[(1,), (None,)], column="A"),
    Case("null_date")(values=[(_DATE,), (None,)], column="A"),
)
@pytest.mark.spark
def test_null_survives_materialization(values: list[tuple], column: str) -> None:
    """A null key is a null in the pandas frame, not a NaN and not dropped."""
    keyset = KeySet.from_tuples(values, columns=[column])
    frame = keyset.to_pandas()
    assert len(frame.index) == 2
    tagged = {tag for (tag,) in _rows(frame, [column])}
    assert ("null", None) in tagged
    assert_same_keys(keyset)


def test_null_and_nan_are_distinguished_by_the_comparison() -> None:
    """The comparison this module's assertions rest on tells the two apart.

    A test that could not see the difference between a null and a NaN would
    pass whether or not the materialization preserved one, so pin that it can.
    """
    null_frame = pd.DataFrame({"A": [None]}, dtype=object)
    nan_frame = pd.DataFrame({"A": [float("nan")]}, dtype=object)
    assert _rows(null_frame, ["A"]) != _rows(nan_frame, ["A"])


@pytest.mark.spark
def test_join_on_null_keys_matches_them() -> None:
    """Nulls join to each other, as they do on Spark, rather than being dropped."""
    left = KeySet.from_tuples([("a1", 1), (None, 2)], columns=["A", "B"])
    right = KeySet.from_tuples([("a2", 7), (None, 8)], columns=["A", "C"])
    joined = left.join(right)
    assert _rows(joined.to_pandas(), ["A", "B", "C"]) == Counter(
        {(("null", None), ("value", 2), ("value", 8)): 1}
    )
    assert_same_keys(joined)


@pytest.mark.spark
def test_subtract_removes_null_keys() -> None:
    """A null on the right of a subtraction removes the null row on the left."""
    left = KeySet.from_tuples([("a1",), ("a2",), (None,)], columns=["A"])
    result = left - KeySet.from_tuples([("a1",), (None,)], columns=["A"])
    assert _rows(result.to_pandas(), ["A"]) == Counter({(("value", "a2"),): 1})
    assert_same_keys(result)


@pytest.mark.spark
def test_nullable_join_column_is_cast_back() -> None:
    """A join column both sides forbid nulls in comes back non-nullable.

    The join's schema says a join column allows nulls only when both sides do,
    but the values come from the left side, whose column may be the nullable
    dtype. Nothing can survive there -- a null on the left has nothing to match
    on the right -- so the result is cast back to the dtype its descriptor calls
    for.
    """
    nullable = KeySet.from_tuples([("a1", 1), ("a2", None)], columns=["A", "B"])
    non_null = KeySet.from_tuples([(1, "x"), (5, "y")], columns=["B", "C"])
    joined = nullable.join(non_null)

    assert joined.schema()["B"].allow_null is False
    assert joined.to_pandas()["B"].dtype == np.dtype("int64")
    assert_same_keys(joined)


@pytest.mark.spark
def test_large_integers_are_not_rounded() -> None:
    """An integer key too large for a float survives the pandas materialization.

    Letting pandas infer the dtype of a nullable integer column built from
    Python values produces a ``float64`` column, which silently rounds anything
    above 2**53. The columns are built in their target dtype instead.
    """
    large = 2**62 + 1
    keyset = KeySet.from_tuples([(large,), (None,)], columns=["A"])
    assert keyset.to_pandas()["A"].dropna().tolist() == [large]
    assert_same_keys(keyset)


################################################################################
# KeySet.from_pandas
################################################################################


@pytest.mark.spark
def test_from_pandas_deduplicates() -> None:
    """Duplicate rows in the given frame become one key."""
    keyset = KeySet.from_pandas(pd.DataFrame({"A": ["a1", "a1", "a2"], "B": [1, 1, 2]}))
    assert keyset.size() == 2
    assert_same_keys(keyset)


def test_from_pandas_copies_the_frame() -> None:
    """Changing the frame afterwards does not change the KeySet.

    This is the copy-on-ingest boundary: a pandas frame is mutable and the
    caller keeps theirs, so a KeySet holding a reference to it could have its
    keys changed underneath it.
    """
    frame = pd.DataFrame({"A": ["a1", "a2"]})
    keyset = KeySet.from_pandas(frame)

    frame.loc[0, "A"] = "changed"

    assert sorted(keyset.to_pandas()["A"]) == ["a1", "a2"]


@pytest.mark.spark
def test_from_pandas_widens_narrow_dtypes() -> None:
    """A narrow input dtype is coerced to the one Analytics uses."""
    keyset = KeySet.from_pandas(pd.DataFrame({"A": np.array([5, 6], dtype="int32")}))
    assert keyset.schema()["A"].allow_null is False
    assert keyset.to_pandas()["A"].dtype == np.dtype("int64")
    assert_same_keys(keyset)


def test_from_pandas_rejects_types_a_keyset_cannot_hold() -> None:
    """A column of a type KeySets do not allow is rejected on construction."""
    with pytest.raises(ValueError, match="has type DECIMAL"):
        KeySet.from_pandas(pd.DataFrame({"A": [1.5, 2.5]}))


def test_from_pandas_zero_columns_is_a_total_aggregation() -> None:
    """An empty frame is the total-aggregation KeySet, as no columns always is."""
    keyset = KeySet.from_pandas(pd.DataFrame())
    assert keyset.size() == 1
    assert keyset.to_pandas().shape == (0, 0)


@pytest.mark.spark
def test_from_pandas_equals_the_same_keys_from_tuples() -> None:
    """A KeySet built from a frame equals one built from the same tuples."""
    frame = pd.DataFrame(
        {"A": ["a1", "a2", None], "B": pd.array([1, 2, None], "Int64")}
    )
    assert KeySet.from_pandas(frame) == KeySet.from_tuples(
        [("a1", 1), ("a2", 2), (None, None)], columns=["A", "B"]
    )


def test_from_pandas_op_equality_is_conservative() -> None:
    """Two ops over distinct frames are not reported equivalent without comparing.

    :meth:`KeySet.is_equivalent` may answer None -- "cannot tell cheaply" -- and
    that is the right answer for two frames it would have to walk to compare.
    """
    frame = pd.DataFrame({"A": ["a1"]})
    same_frame = KeySet.from_pandas(frame)

    assert same_frame.is_equivalent(KeySet.from_pandas(frame.copy())) is None
    # ...but the op-tree of a KeySet is equivalent to itself.
    assert same_frame.is_equivalent(same_frame) is True


################################################################################
# Caching
################################################################################


@pytest.mark.spark
def test_materialized_frames_are_cached_per_backend() -> None:
    """Each backend's frame is built once, and neither displaces the other."""
    keyset = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])

    pandas_frame = keyset.to_pandas()
    spark_frame = keyset.dataframe()

    assert keyset.to_pandas() is pandas_frame
    assert keyset.dataframe() is spark_frame


@pytest.mark.spark
def test_cache_does_not_disturb_the_pandas_frame() -> None:
    """cache() and uncache() are Spark's, and leave the pandas frame alone."""
    keyset = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
    pandas_frame = keyset.to_pandas()

    keyset.cache()
    assert keyset.to_pandas() is pandas_frame
    assert keyset.dataframe().is_cached

    keyset.uncache()
    assert keyset.to_pandas() is pandas_frame
    assert keyset.dataframe().is_cached is False


@pytest.mark.spark
def test_cache_before_materializing_still_caches_spark() -> None:
    """A KeySet cached before either frame exists still caches the Spark one."""
    keyset = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
    keyset.cache()

    assert len(keyset.to_pandas().index) == 2
    assert keyset.dataframe().is_cached


################################################################################
# Operations with no pandas implementation
################################################################################


def test_filter_is_not_supported(spark) -> None:
    """A filtered KeySet cannot be materialized in pandas."""
    keyset = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"]).filter("A = 'a1'")

    with pytest.raises(NotSupportedByBackend, match="Filter is not supported"):
        keyset.to_pandas()
    assert keyset._op_tree.unsupported_ops(PANDAS) == {"Filter"}
    assert keyset._op_tree.unsupported_ops(SPARK) == set()
    # It is still perfectly materializable on Spark.
    assert keyset.dataframe().count() == 1


def test_from_dataframe_is_not_supported(spark) -> None:
    """A KeySet built from a Spark dataframe cannot be materialized in pandas.

    Collecting a distributed frame into the driver is unbounded work on data
    whose size nobody has looked at, so it is not something a backend switch
    does silently.
    """
    keyset = KeySet.from_dataframe(spark.createDataFrame(pd.DataFrame({"A": ["a1"]})))

    with pytest.raises(
        NotSupportedByBackend, match="FromSparkDataFrame is not supported"
    ):
        keyset.to_pandas()
    assert keyset._op_tree.unsupported_ops(PANDAS) == {"FromSparkDataFrame"}
    assert keyset._op_tree.unsupported_ops(SPARK) == set()
    assert keyset.dataframe().count() == 1


def test_unsupported_ops_names_all_of_them(spark) -> None:
    """The walker reports every offending operation in a tree, not just one."""
    from_spark = KeySet.from_dataframe(
        spark.createDataFrame(pd.DataFrame({"A": ["x"]}))
    )
    filtered = KeySet.from_tuples([(1,), (2,)], columns=["B"]).filter("B = 1")
    combined = from_spark * filtered

    assert combined._op_tree.unsupported_ops(PANDAS) == {
        "Filter",
        "FromSparkDataFrame",
    }
    assert combined._op_tree.unsupported_ops(SPARK) == set()


def test_unsupported_ops_is_empty_for_the_corpus() -> None:
    """Nothing in the corpus is reported as unsupported on either backend."""
    for case in _corpus():
        op_tree = case.args["keyset"]._op_tree
        assert op_tree.unsupported_ops(PANDAS) == set()
        assert op_tree.unsupported_ops(SPARK) == set()


def test_a_plan_is_unsupported_everywhere() -> None:
    """A KeySetPlan cannot be materialized on any backend."""
    plan = KeySet._detect(["A"])
    assert plan._op_tree.unsupported_ops(PANDAS) == {"Detect"}
    assert plan._op_tree.unsupported_ops(SPARK) == {"Detect"}


################################################################################
# Choosing a backend
################################################################################


@parametrize(
    Case("spark")(backend=SPARK, expected=FrameKind.SPARK),
    Case("pandas")(backend=PANDAS, expected=FrameKind.PANDAS),
)
def test_frame_kind(backend: Backend, expected: FrameKind) -> None:
    """A backend's frame kind is read off the frame type it carries."""
    assert frame_kind(backend) is expected


def test_frame_kind_rejects_other_backends() -> None:
    """A backend carrying neither kind of frame cannot materialize a KeySet."""
    with pytest.raises(NotSupportedByBackend, match="KeySet materialization"):
        frame_kind(NEITHER)


@pytest.mark.spark
def test_dataframe_defaults_to_spark() -> None:
    """Saying nothing about a backend gets a Spark dataframe, as it always did."""
    keyset = KeySet.from_tuples([("a1",)], columns=["A"])
    assert isinstance(keyset.dataframe(), DataFrame)
    assert isinstance(keyset._op_tree.dataframe(), DataFrame)


@pytest.mark.spark
def test_op_dataframe_takes_a_backend() -> None:
    """An op-tree materializes on whichever backend it is handed."""
    keyset = KeySet.from_tuples([("a1",)], columns=["A"])
    assert isinstance(keyset._op_tree.dataframe(PANDAS), pd.DataFrame)
    assert isinstance(keyset._op_tree.dataframe(SPARK), DataFrame)


def test_keyset_dataframe_with_a_pandas_backend() -> None:
    """KeySet.dataframe honours a pandas backend, though to_pandas is the way to ask."""
    keyset = KeySet.from_tuples([("a1",)], columns=["A"])
    frame: Any = keyset.dataframe(PANDAS)
    assert isinstance(frame, pd.DataFrame)
    assert frame is keyset.to_pandas()


################################################################################
# Equality, and which engine decides it
################################################################################
#
# KeySet.__eq__ falls back to comparing the keys themselves whenever
# is_equivalent cannot answer from the op-trees. That fallback used to be a
# Spark exceptAll on frames built at the Spark default, so comparing two
# KeySets built from Python literals started a JVM -- and __eq__ is reached
# from places that never mention a backend: a frozen Query dataclass holding a
# KeySet, or a dict lookup, since __hash__ hashes the schema alone and leaves
# every collision to be settled by ==. The tests below are unmarked, so the
# no-JVM lane runs them and the claim is checked rather than asserted.


def test_equality_of_in_memory_keysets_needs_no_spark() -> None:
    """Two KeySets whose trees can be built in memory compare in memory.

    The four keysets are the ones in :meth:`KeySet.__eq__`'s own docstring: the
    same keys in a different column order, the same keys in a different row
    order, and a different set of keys.
    """
    ks1 = KeySet.from_dict({"A": [1, 2], "B": [3, 4]})
    ks2 = KeySet.from_dict({"B": [3, 4], "A": [1, 2]})
    ks3 = KeySet.from_dict({"B": [4, 3], "A": [2, 1]})
    ks4 = KeySet.from_dict({"B": [4, 5], "A": [1, 2]})
    assert ks1 == ks2
    assert ks1 == ks3
    assert ks1 != ks4
    assert ks4 != ks1


@parametrize(
    Case("subtraction")(
        left=KeySet.from_dict({"A": ["a1", "a2", "a3"]})
        - KeySet.from_tuples([("a3",)], columns=["A"]),
        right=KeySet.from_tuples([("a1",), ("a2",)], columns=["A"]),
        equal=True,
    ),
    Case("subtraction_unequal")(
        left=KeySet.from_dict({"A": ["a1", "a2", "a3"]})
        - KeySet.from_tuples([("a3",)], columns=["A"]),
        right=KeySet.from_tuples([("a1",), ("a3",)], columns=["A"]),
        equal=False,
    ),
    Case("cross_join")(
        left=KeySet.from_dict({"A": ["a1", "a2"]}) * KeySet.from_dict({"B": [1, 2]}),
        right=KeySet.from_tuples(
            [("a1", 1), ("a1", 2), ("a2", 1), ("a2", 2)], columns=["A", "B"]
        ),
        equal=True,
    ),
    Case("union")(
        left=KeySet.from_dict({"A": ["a1"]}).union(KeySet.from_dict({"A": ["a2"]})),
        right=KeySet.from_dict({"A": ["a1", "a2"]}),
        equal=True,
    ),
    Case("a_null_key_is_a_key")(
        left=KeySet.from_dict({"A": ["a1", "a2", None]})
        - KeySet.from_tuples([("a2",)], columns=["A"]),
        right=KeySet.from_tuples([("a1",), (None,)], columns=["A"]),
        equal=True,
    ),
    Case("a_null_key_is_not_a_missing_key")(
        left=KeySet.from_dict({"A": ["a1", None]}),
        right=KeySet.from_tuples([("a1",)], columns=["A"]),
        equal=False,
    ),
    Case("from_pandas_against_from_tuples")(
        left=KeySet.from_pandas(
            pd.DataFrame(
                {"A": ["a1", "a2", None], "B": pd.array([1, 2, None], "Int64")}
            )
        ),
        right=KeySet.from_tuples(
            [("a1", 1), ("a2", 2), (None, None)], columns=["A", "B"]
        ),
        equal=True,
    ),
)
def test_value_equality_without_spark(left: KeySet, right: KeySet, equal: bool) -> None:
    """Value-level equality is answered in memory, and answered correctly.

    Every case here is one ``is_equivalent`` cannot settle from the op-trees, so
    each really does take the fallback -- which is the path this is about.
    """
    assert left.is_equivalent(right) is None
    assert (left == right) is equal
    assert (right == left) is equal


@pytest.mark.spark
def test_equality_against_a_spark_tree_uses_spark(spark) -> None:
    """A comparison one side of which needs Spark is done on Spark.

    A KeySet built from a Spark DataFrame cannot be materialized in memory --
    collecting a distributed frame is its owner's decision -- so the in-memory
    path is not available for it, whatever the other side is. The comparison
    still has to give the right answer, and it does: both sides go to Spark, as
    they always did.
    """
    from_spark = KeySet.from_dataframe(
        spark.createDataFrame(pd.DataFrame({"A": ["a1", "a2"]}))
    )
    in_memory = KeySet.from_tuples([("a1",), ("a2",)], columns=["A"])
    different = KeySet.from_tuples([("a1",), ("a3",)], columns=["A"])

    assert from_spark._op_tree.unsupported_frame_ops(FrameKind.PANDAS)
    assert not in_memory._op_tree.unsupported_frame_ops(FrameKind.PANDAS)

    assert from_spark == in_memory
    assert in_memory == from_spark
    assert from_spark != different
    assert different != from_spark
    # The in-memory side was compared without being built in memory.
    assert FrameKind.PANDAS not in in_memory._dataframes


@pytest.mark.spark
def test_a_filtered_keyset_is_compared_on_spark(spark) -> None:
    """The other irreducibly-Spark tree: one built by filtering."""
    # pylint: disable=unused-argument
    filtered = KeySet.from_dict({"A": ["a1", "a2"]}).filter("A = 'a1'")
    in_memory = KeySet.from_tuples([("a1",)], columns=["A"])
    assert filtered._op_tree.unsupported_frame_ops(FrameKind.PANDAS) == {"Filter"}
    assert filtered == in_memory
    assert in_memory == filtered
    assert filtered != KeySet.from_tuples([("a2",)], columns=["A"])
