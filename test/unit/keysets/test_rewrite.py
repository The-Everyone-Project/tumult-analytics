"""Tests for KeySetOp tree rewriting operations.

This logic is to some degree tested by the other KeySet tests, but these tests
explicitly cover that rewrite rules don't change the output dataframes, and aim
to hit known-tricky pieces of the rewriting logic.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from inspect import isabstract
from typing import Callable, Dict, List, Optional, Type, cast
from unittest.mock import patch

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

from tmlt.analytics import AnalyticsInternalError, KeySet
from tmlt.analytics.keyset._ops._base import KeySetOp
from tmlt.analytics.keyset._ops._cross_join import CrossJoin, InMemoryCrossJoin
from tmlt.analytics.keyset._ops._detect import Detect
from tmlt.analytics.keyset._ops._filter import Filter
from tmlt.analytics.keyset._ops._from_tuples import FromTuples
from tmlt.analytics.keyset._ops._join import Join
from tmlt.analytics.keyset._ops._project import Project
from tmlt.analytics.keyset._ops._subtract import Subtract
from tmlt.analytics.keyset._ops._union import Union


def _from_df(data: dict, spark: SparkSession) -> KeySet:
    return KeySet.from_dataframe(spark.createDataFrame(pd.DataFrame(data)))


# Be careful using anything that boils down to cross-joining FromTuples in these
# tests, as apply_cross_joins_in_memory will hide lots of other optimizations.
@parametrize(
    Case("from_tuples_crossjoin")(
        ks=lambda spark: (
            KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"])
            * KeySet.from_tuples([(5,), (6,), (7,)], columns=["C"])
        )
    ),
    Case("crossjoin_reorder")(
        ks=lambda spark: KeySet.from_dict({"A": [1], "C": [2], "B": [3]})
    ),
    Case("crossjoin_reorder_df")(
        ks=lambda spark: (
            _from_df({"A": [1]}, spark)
            * _from_df({"C": [2]}, spark)
            * _from_df({"B": [3]}, spark)
        )
    ),
    Case("crossjoin_merge")(
        ks=lambda spark: (
            (KeySet.from_dict({"A": [1]}) * KeySet.from_dict({"C": [3]}))
            * (KeySet.from_dict({"D": [4]}) * KeySet.from_dict({"B": [2]}))
        )
    ),
    Case("crossjoin_merge_df")(
        ks=lambda spark: (
            (_from_df({"A": [1]}, spark) * _from_df({"C": [3]}, spark))
            * (_from_df({"D": [4]}, spark) * _from_df({"B": [2]}, spark))
        )
    ),
    Case("crossjoin_merge_mixed")(
        ks=lambda spark: (
            (KeySet.from_dict({"A": [1]}) * _from_df({"C": [3]}, spark))
            * (KeySet.from_dict({"D": [4]}) * _from_df({"B": [2]}, spark))
        )
    ),
    Case("join_reorder")(
        ks=lambda spark: KeySet.from_dict({"B": [2], "C": [3]}).join(
            KeySet.from_dict({"A": [1], "B": [2]})
        )
    ),
    Case("join_linearize")(
        ks=lambda spark: (
            KeySet.from_dict({"B": [2], "C": [3]})
            .join(KeySet.from_dict({"A": [1], "B": [2]}))
            .join(
                KeySet.from_dict({"C": [3], "D": [4]}).join(
                    KeySet.from_dict({"D": [4], "E": [5]})
                )
            )
        )
    ),
    Case("union_reorder")(
        ks=lambda spark: KeySet.from_dict({"A": [1, 2]}).union(
            KeySet.from_dict({"A": [2, 3]})
        ),
        # Because the ordering depends on hashes, this may or may not change the
        # resulting op-tree.
        allow_unchanged=True,
    ),
    Case("union_linearize")(
        ks=lambda spark: (
            KeySet.from_dict({"A": [1, 2]})
            .union(KeySet.from_dict({"A": [2, 3]}))
            .union(KeySet.from_dict({"A": [4]}).union(KeySet.from_dict({"A": [5]})))
        ),
    ),
    Case("nested_project")(
        ks=lambda spark: KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"])[
            "A", "B"
        ]["A"]
    ),
    Case("noop_project")(
        ks=lambda spark: KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"])[
            "A", "B", "C"
        ]
    ),
    Case("crossjoin_project_left")(
        ks=lambda spark: (
            KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"])
            * KeySet.from_tuples([(4, 5, 6)], columns=["D", "E", "F"])
        )["A", "B"]
    ),
    Case("crossjoin_project_left_df")(
        ks=lambda spark: (
            _from_df({"A": [1], "B": [2], "C": [3]}, spark)
            * _from_df({"D": [4], "E": [5], "F": [6]}, spark)
        )["A", "B"]
    ),
    Case("crossjoin_project_right")(
        ks=lambda spark: (
            KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"])
            * KeySet.from_tuples([(4, 5, 6)], columns=["D", "E", "F"])
        )["D", "E"]
    ),
    Case("crossjoin_project_right_df")(
        ks=lambda spark: (
            _from_df({"A": [1], "B": [2], "C": [3]}, spark)
            * _from_df({"D": [4], "E": [5], "F": [6]}, spark)
        )["D", "E"]
    ),
    Case("crossjoin_project_both")(
        ks=lambda spark: (
            KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"])
            * KeySet.from_tuples([(4, 5, 6)], columns=["D", "E", "F"])
        )["C", "D", "E"]
    ),
    Case("crossjoin_project_both_df")(
        ks=lambda spark: (
            _from_df({"A": [1], "B": [2], "C": [3]}, spark)
            * _from_df({"D": [4], "E": [5], "F": [6]}, spark)
        )["C", "D", "E"]
    ),
    Case("crossjoin_project_nested")(
        ks=lambda spark: (
            KeySet.from_tuples([(1, 2, 3)], columns=["A", "B", "C"])
            * KeySet.from_tuples([(4, 5, 6)], columns=["D", "E", "F"])
            * KeySet.from_tuples([(7, 8, 9)], columns=["G", "H", "I"])
        )["C", "D", "H"]
    ),
    Case("crossjoin_project_nested_df")(
        ks=lambda spark: (
            _from_df({"A": [1], "B": [2], "C": [3]}, spark)
            * _from_df({"D": [4], "E": [5], "F": [6]}, spark)
            * _from_df({"G": [7], "H": [8], "I": [9]}, spark)
        )["C", "D", "H"]
    ),
    Case("subtract_reorder")(
        ks=lambda spark: (
            KeySet.from_dict({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
            - KeySet.from_tuples([(1, 7)], columns=["A", "C"])
            - KeySet.from_tuples([(5, 2), (6, 2)], columns=["B", "A"])
            - KeySet.from_tuples([(4, 7)], columns=["B", "C"])
        )
    ),
    Case("subtract_reorder_nested")(
        ks=lambda spark: (
            KeySet.from_dict({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
            - KeySet.from_tuples([(1, 7)], columns=["A", "C"])
            - (
                KeySet.from_tuples([(5, 2), (6, 2)], columns=["B", "A"])
                - KeySet.from_tuples([(6, 2)], columns=["B", "A"])
            )
            - KeySet.from_tuples([(4, 7)], columns=["B", "C"])
        )
    ),
    Case("extract_crossjoin_from_join_left")(
        ks=lambda spark: (
            KeySet.from_tuples([(2, 8), (4, 10)], columns=["B", "C"])
            * KeySet.from_dict({"D": [1, 2], "E": [3]})
        ).join(KeySet.from_tuples([(1, 2), (3, 4), (5, 6)], columns=["A", "B"]))
    ),
    Case("extract_crossjoin_from_join_right")(
        ks=lambda spark: KeySet.from_tuples(
            [(1, 2), (3, 4), (5, 6)], columns=["A", "B"]
        ).join(
            KeySet.from_tuples([(2, 8), (4, 10)], columns=["B", "C"])
            * KeySet.from_dict({"D": [1, 2], "E": [3]})
        )
    ),
    Case("extract_crossjoin_from_join_both")(
        ks=lambda spark: (
            KeySet.from_tuples([(1, 2), (3, 4), (5, 6)], columns=["A", "B"])
            * KeySet.from_dict({"D": [1, 2]})
        ).join(
            KeySet.from_tuples([(2, 8), (4, 10)], columns=["B", "C"])
            * KeySet.from_dict({"E": [3]})
        )
    ),
    Case("extract_crossjoin_from_join_neither")(
        ks=lambda spark: (
            KeySet.from_tuples([(1, 2), (3, 4)], columns=["A", "B"])
            * KeySet.from_tuples([(5, 6), (7, 8)], columns=["C", "D"])
        ).join(
            KeySet.from_tuples([(1, 11), (3, 12)], columns=["A", "E"])
            * KeySet.from_tuples([(13, 6), (14, 8)], columns=["F", "D"])
        ),
        allow_unchanged=True,
    ),
    Case("extract_crossjoin_from_subtract")(
        ks=lambda spark: (
            KeySet.from_dict({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
            - KeySet.from_tuples([(5, 7), (6, 8)], columns=["B", "C"])
        )
    ),
)
def test_rewrite_equality(
    ks: Callable[[SparkSession], KeySet], allow_unchanged: Optional[bool], spark
):
    """Rewritten KeySets have the same semantics as the original ones."""
    ks_rewritten = ks(spark)
    with patch("tmlt.analytics.keyset._keyset.rewrite", lambda op: op):
        ks_original = ks(spark)

    if not allow_unchanged:
        # Ensure that rewriting actually happened
        assert ks_rewritten._op_tree != ks_original._op_tree

    assert ks_rewritten.columns() == ks_original.columns()
    assert ks_rewritten.schema() == ks_original.schema()
    assert_dataframe_equal(ks_rewritten.dataframe(), ks_original.dataframe())


###############################################################################
# The walk itself: children() and with_children() are what the walkers know.
###############################################################################


def _concrete_op_classes() -> List[Type[KeySetOp]]:
    """Every concrete KeySetOp class, found by walking the class hierarchy."""

    # Typed as plain `type` so that the abstract base can be passed in; the
    # abstract classes are then filtered out, which is what makes the cast true.
    def descendants(cls: type) -> List[type]:
        found: List[type] = []
        for subclass in cls.__subclasses__():
            found.append(subclass)
            found.extend(descendants(subclass))
        return found

    return [
        cast(Type[KeySetOp], cls)
        for cls in descendants(KeySetOp)
        if not isabstract(cls)
    ]


def test_an_operation_with_children_can_rebuild_itself():
    """Any op that has children can be given new ones.

    ``depth_first`` and ``breadth_first`` rewrite a tree by taking each node
    apart with :meth:`KeySetOp.children` and putting it back together with
    :meth:`KeySetOp.with_children`. They ask nothing else about a node, so an
    operation that overrides one of the pair and not the other would be walked
    as though it had no children -- silently, and only in the rewriter. This is
    what makes adding an operation a matter of writing the operation.
    """
    missing = [
        cls.__qualname__
        for cls in _concrete_op_classes()
        if cls.children is not KeySetOp.children
        and cls.with_children is KeySetOp.with_children
    ]
    assert missing == []


def _tuples(rows: List[tuple], columns: List[str]) -> FromTuples:
    """The FromTuples operation a KeySet of literal rows is built from."""
    # pylint: disable=protected-access
    return cast(FromTuples, KeySet.from_tuples(rows, columns=columns)._op_tree)


_A = _tuples([(1,), (2,)], ["A"])
_B = _tuples([(3,), (4,)], ["B"])
_A2 = _tuples([(1,), (5,)], ["A"])

_OPS: Dict[str, KeySetOp] = {
    "from_tuples": _A,
    "detect": Detect(frozenset({"A"})),
    "cross_join": CrossJoin((_A, _B)),
    "in_memory_cross_join": InMemoryCrossJoin((_A, _B)),
    "join": Join(_A, CrossJoin((_A, _B))),
    "project": Project(CrossJoin((_A, _B)), frozenset({"A"})),
    "filter": Filter(_A, "A > 1"),
    "subtract": Subtract(_A, _A2),
    "union": Union(_A, _A2),
}
"""One op of each kind that can be built without a Spark session."""


@pytest.mark.parametrize("op", _OPS.values(), ids=_OPS.keys())
def test_with_children_inverts_children(op: KeySetOp):
    """Putting an op back together with its own children returns an equal op."""
    rebuilt = op.with_children(op.children())
    assert rebuilt == op
    assert type(rebuilt) is type(op)


def _other_leaf(op: KeySetOp) -> KeySetOp:
    """A different operation over the same columns as the given one."""
    columns = sorted(op.columns())
    rows = [tuple(range(100, 100 + len(columns)))]
    return KeySet.from_tuples(rows, columns=columns)._op_tree


@pytest.mark.parametrize("op", _OPS.values(), ids=_OPS.keys())
def test_with_children_replaces_the_children(op: KeySetOp):
    """The new children are the ones the rebuilt op reports."""
    replacements = tuple(_other_leaf(c) for c in op.children())
    assert op.with_children(replacements).children() == replacements


def test_with_children_rejects_children_for_a_leaf():
    """An op that introduces data has no children to replace."""
    with pytest.raises(AnalyticsInternalError):
        _A.with_children((_B,))
