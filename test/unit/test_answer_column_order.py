"""The answer's column order is the table's, in every process.

A group-by answer names its group columns in the order its ``GroupBy`` was
built with, and Core takes that order from the transformation feeding it. Where
Analytics builds that transformation from a ``set``, the order is whatever the
set iterates in -- which depends on ``PYTHONHASHSEED``, so it is stable within a
process and varies between them. An ordinary in-process assertion cannot catch
that: it would pass or fail according to the seed the suite happened to run
under, and pytest's default seed of 0 is one of the ones that passes.

So the check here runs the query in subprocesses, one per seed, and asserts they
all agree with the table. The subprocesses run concurrently because each one
pays a fresh interpreter's import cost and they do not interact.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

_TABLE_COLUMNS = ("g", "g2", "v", "w")
"""The private table's columns, in the order the table declares them."""

_PROGRAM = f"""
import pandas as pd
import sympy as sp
from tmlt.core.domains.collections import DictDomain
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountant,
    SequentialComposition,
)
from tmlt.core.measures import PureDP
from tmlt.core.metrics import DictMetric, SymmetricDifference

from tmlt.analytics import KeySet, PureDPBudget, QueryBuilder, Session
from tmlt.analytics._backends import PANDAS
from tmlt.analytics._schema import Schema
from tmlt.analytics._table_identifier import NamedTable

columns = {list(_TABLE_COLUMNS)!r}
schema = Schema(dict(zip(columns, ["VARCHAR", "VARCHAR", "INTEGER", "INTEGER"])))
frame = pd.DataFrame(
    dict(zip(columns, [["a", "a", "b"], ["x", "y", "x"], [1, 1, 2], [7, 8, 9]]))
)

domain = {{NamedTable("t"): PANDAS.dataframe_domain(schema)}}
accountant = PrivacyAccountant.launch(
    SequentialComposition(
        input_domain=DictDomain(domain),
        input_metric=DictMetric({{key: SymmetricDifference() for key in domain}}),
        d_in={{key: 1 for key in domain}},
        privacy_budget=sp.oo,
        output_measure=PureDP(),
    ),
    {{NamedTable("t"): frame}},
)
session = Session(accountant=accountant, public_sources={{}})

# The group-by columns are the table's first two, and the counted columns are
# its last two named backwards, so neither list on its own is the answer: only
# a Select that reads the table's order gets "g" before "g2".
keys = KeySet.from_dict({{"g": ["a", "b"], "g2": ["x", "y"]}})
query = QueryBuilder("t").groupby(keys).count_distinct(columns=["w", "v"], name="c")
answer = session.evaluate(query, PureDPBudget(float("inf")))
print(" ".join(answer.columns))
"""
"""A whole count-distinct query, as a program a fresh interpreter can run.

It is a string rather than a module so that the seeds under test and the order
they must produce stay in one file with the assertion about them.
"""

_SEEDS: Tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6", "7")
"""The hash seeds to run under.

Eight is margin, not a measurement: before the fix five of the first twelve
seeds produced a swapped answer, so any handful of them would have caught it,
and which handful will vary with the interpreter.
"""

_EXPECTED = ["g", "g2", "c"]
"""The group columns in the table's order, then the count."""


def _column_order(seed: str) -> List[str]:
    """Return the answer's columns from a subprocess run under one hash seed."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.split()


def test_answer_column_order_is_the_table_order_under_every_hash_seed():
    """A count-distinct answer names its group columns in the table's order."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        orders: Dict[str, List[str]] = dict(
            zip(_SEEDS, pool.map(_column_order, _SEEDS))
        )
    assert orders == {seed: _EXPECTED for seed in _SEEDS}
