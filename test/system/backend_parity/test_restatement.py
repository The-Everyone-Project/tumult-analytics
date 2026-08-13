"""The standard tables really are the existing fixtures, restated.

:mod:`test.backend_testing.data` states each of the existing suite's tables a
second time -- its rows and its Analytics schema, as data rather than as a frame
-- so that the parity suite can build the same table on either backend. Every
parity result rests on that restatement being faithful, and it is a claim about
frames built somewhere else, by fixtures that know nothing about it: a fixture
edited without its spec would leave the parity suite comparing two backends on
data the rest of the suite had stopped using, and saying nothing about it.

Checked rather than made
========================

The obvious alternative is to have one build the other, leaving one copy of the
rows. Both directions were rejected.

Building the fixtures from the specs would mean rewriting fixtures this fork has
deliberately left alone -- ``test/conftest.py``'s especially, whose whole
pandas-backend diff is additive by design -- and it would erase a difference the
two spellings carry: the ``id3`` and ``id4`` fixtures give their ID column
Spark's ``IntegerType``, where a schema derived from a spec gives ``LongType``.
Those two fixtures are therefore the suite's only tables that exercise
Analytics' widening of a 32-bit integer column, and building them from their
specs would silently stop testing it.

Building the specs from the fixtures would make them need a Spark session, which
is the one thing :mod:`test.backend_testing.data` is careful not to need: the
whole point of a spec is to be materializable on a backend that has no JVM.

So the two stay independent and are held to agreeing about the table -- its
columns in order, their Analytics types and nullability, and its rows -- which is
everything a parity test reads. The Spark type width is exactly what they are
allowed to differ about, and :func:`analytics_columns` is the comparison that
allows it: ``IntegerType`` and ``LongType`` are both ``INTEGER``.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Any, Dict, List

import pytest
from pyspark.sql import SparkSession

from test.backend_testing import (
    ID1,
    ID2,
    ID3,
    ID4,
    ROWS1,
    AnyFrame,
    TableSpec,
    analytics_columns,
    frame_as_rows,
    spark_frame,
)

_SYSTEM_TABLES: Dict[str, TableSpec] = {
    "id1": ID1,
    "id2": ID2,
    "id3": ID3,
    "id4": ID4,
    "rows1": ROWS1,
}
"""The tables ``test/system/conftest.py``'s ``_session_data`` builds, by its keys.

``private_id_data``, the sixth standard table, comes from the fixture of the same
name in ``test/conftest.py``, which this fixture shadows. Its restatement is
checked in ``test/unit/test_backend_testing.py``, where that fixture is the one
in scope.
"""


def _rows(frame: AnyFrame) -> List[str]:
    """A frame's rows, as strings that sort, so row order does not matter.

    The values are the keys :func:`frame_as_rows` produces, so a null reads as
    the null sentinel rather than comparing unequal to itself.
    """
    return sorted(repr(sorted(row.items())) for row in frame_as_rows(frame))


@pytest.mark.parametrize("name", sorted(_SYSTEM_TABLES))
def test_the_spec_restates_the_system_fixture(
    name: str, _session_data: Dict[str, Any], spark: SparkSession
):
    """A standard table's spec describes the fixture frame of the same name."""
    spec = _SYSTEM_TABLES[name]
    fixture = _session_data[name]

    assert list(fixture.columns) == list(spec.columns)
    assert analytics_columns(fixture) == spec.column_descs()
    assert _rows(fixture) == _rows(spark_frame(spec, spark))
