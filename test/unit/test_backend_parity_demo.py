"""One query, run on both backends and compared: the harness end to end.

This is a demonstration rather than a suite. It exists so that the three pieces
of :mod:`test.backend_testing` -- the ``backend`` fixture, the standard tables,
and the comparison shim -- are exercised together by something a parity suite
can be copied from.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import numpy as np

from tmlt.analytics import AddMaxRows, KeySet, PureDPBudget, QueryBuilder

from test.backend_testing import (
    ID1,
    BackendFixture,
    assert_frame_equal_across_backends,
    pandas_frame_from_rows,
)

INF_BUDGET = PureDPBudget(float("inf"))
"""An infinite budget adds no noise, so the counts below are exact."""

GROUPS = KeySet.from_dict({"group": ["A", "B"]})

EXPECTED = pandas_frame_from_rows(
    ["group", "count"],
    [("A", 5), ("B", 1)],
    {"group": np.dtype(object), "count": np.dtype("int64")},
)
"""``ID1`` has five rows in group A and one in group B."""


def test_groupby_count_agrees_across_backends(backend: BackendFixture):
    """The same groupby count, on the same data, on either backend."""
    session = backend.build_session(
        {"t": ID1}, budget=INF_BUDGET, protected_change=AddMaxRows(1)
    )
    result = session.evaluate(QueryBuilder("t").groupby(GROUPS).count(), INF_BUDGET)
    assert_frame_equal_across_backends(result, EXPECTED, sort_by=["group"])
