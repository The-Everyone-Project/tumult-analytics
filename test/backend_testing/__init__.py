"""Test infrastructure for comparing the Spark and pandas backends.

Analytics is growing a pandas backend beside its Spark one. Wherever both exist,
the interesting question is not "does this work" but "do the two agree", and
answering it needs one test body that can be handed either backend. This package
is the machinery for that, and nothing else: a fixture that yields each backend
in turn, the standard test tables said once and materialized twice, and a
definition of when two results count as the same answer.

Everything here is backend-*neutral*. Anything that knows what a particular
query does belongs in the suite testing it.

Using it
========

::

    from test.backend_testing import (
        ID1,
        BackendFixture,
        assert_frame_equal_across_backends,
    )

    def test_something(backend: BackendFixture):
        session = backend.build_session({"t": ID1}, budget=PureDPBudget(float("inf")))
        result = session.evaluate(query, PureDPBudget(float("inf")))
        assert_frame_equal_across_backends(result, expected, sort_by=["group"])

The ``backend`` fixture itself is made available to a directory by importing it
into that directory's ``conftest.py``; see
:mod:`test.backend_testing.fixtures`. ``test/unit/conftest.py`` already does.

The three pieces
================

* :mod:`test.backend_testing.data` -- the standard tables as
  :class:`~test.backend_testing.data.TableSpec` values: rows of plain Python
  values plus the Analytics schema they have. No frames, no pyspark.
* :mod:`test.backend_testing.materialize` -- one spec, two frames, and the
  nullability rule that says where the two cannot be made to agree. **Read its
  docstring before writing a test that asserts a schema.**
* :mod:`test.backend_testing.frames` and
  :mod:`test.backend_testing.comparison` -- getting a result back to pandas
  without destroying it, and comparing two of them. **Read
  :mod:`~test.backend_testing.frames` before reaching for ``toPandas()``**: it
  is lossy in two ways that make a parity test lie.

What the helpers guarantee
==========================

* :func:`~test.backend_testing.frames.to_pandas` preserves values. Integers stay
  integers however large, a null stays distinguishable from a NaN, and an empty
  frame keeps its dtypes. It is the identity on a pandas frame given no schema.
* :func:`~test.backend_testing.comparison.assert_frame_equal_across_backends`
  compares *answers*: it ignores column order, ignores row order when told which
  columns to sort by, and ignores dtypes -- but a null never equals a NaN, and
  integers are never compared through a float.
* :func:`~test.backend_testing.materialize.expected_columns` is the oracle for
  "what schema does this backend report", and it is not always the spec:
  ``VARCHAR``, ``DATE`` and ``TIMESTAMP`` columns are always nullable on pandas.
* The ``backend`` fixture starts no JVM for its pandas parameter, and holds the
  ``pandas_backend`` feature flag open for it.

The names re-exported below are the surface a parity suite should use; the
modules' other names are implementation details.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from test.backend_testing.backends import BACKEND_NAMES, BackendFixture, Tables
from test.backend_testing.comparison import (
    NAN,
    NULL,
    assert_frame_equal_across_backends,
    frame_as_rows,
    value_key,
)
from test.backend_testing.data import (
    ID1,
    ID2,
    ID3,
    ID4,
    ID_TABLES,
    PRIVATE_ID_DATA,
    ROWS1,
    STANDARD_TABLES,
    Row,
    TableSpec,
)
from test.backend_testing.fixtures import backend
from test.backend_testing.frames import (
    AnyFrame,
    analytics_columns,
    is_null_value,
    pandas_dtypes_for,
    pandas_frame_from_columns,
    pandas_frame_from_rows,
    to_pandas,
)
from test.backend_testing.materialize import (
    PANDAS_ALWAYS_NULLABLE_TYPES,
    BackendLike,
    all_nullability_divergences,
    expected_columns,
    frame_for,
    nullability_divergences,
    pandas_frame,
    spark_frame,
)

__all__ = [
    "BACKEND_NAMES",
    "ID1",
    "ID2",
    "ID3",
    "ID4",
    "ID_TABLES",
    "NAN",
    "NULL",
    "PANDAS_ALWAYS_NULLABLE_TYPES",
    "PRIVATE_ID_DATA",
    "ROWS1",
    "STANDARD_TABLES",
    "AnyFrame",
    "BackendFixture",
    "BackendLike",
    "Row",
    "TableSpec",
    "Tables",
    "all_nullability_divergences",
    "analytics_columns",
    "assert_frame_equal_across_backends",
    "backend",
    "expected_columns",
    "frame_as_rows",
    "frame_for",
    "is_null_value",
    "nullability_divergences",
    "pandas_dtypes_for",
    "pandas_frame",
    "pandas_frame_from_columns",
    "pandas_frame_from_rows",
    "spark_frame",
    "to_pandas",
    "value_key",
]
