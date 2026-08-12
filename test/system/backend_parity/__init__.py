"""The evidence that the pandas backend computes the same DP as the Spark one.

This package is the acceptance suite for the pandas backend. Everything else in
the branch shows that the pandas path *works*; this shows that it works the
*same way*, which is the only claim a privacy engine's second backend can be
accepted on.

Why ``test/system``
===================

Every test here builds a real :class:`~tmlt.analytics.Session` and evaluates
real queries through it. That is the system tree's job, and it is also the only
place the question can be asked: the property under test is not "does this
transformation agree" -- Core's own parity suite settles that, bit for bit --
but "does the agreement survive the whole Analytics compile", from
``QueryBuilder`` through the rewriter, the constraint machinery, the accountant
and the noise.

The five bars
=============

Each is one module, and each answers a different question.

* :mod:`~test.system.backend_parity.test_answers` -- **structure and value.**
  The grid: two budget flavors, two aggregations, four ways of grouping, three
  protected changes with three truncation constraints between them, plus
  chains, private joins and suppression. Every cell runs on both backends and
  is compared against one hand-checked answer, so agreement is proved *through*
  correctness rather than by comparing two possibly-identical mistakes.
* :mod:`~test.system.backend_parity.test_truncation` -- **which rows survive.**
  The load-bearing module, and the gap A4b's smoke tests flagged: a count per
  group agreeing does not prove the two backends *kept the same rows*. Here the
  data is built so that a different choice would change a downstream answer.
* :mod:`~test.system.backend_parity.test_calibration` -- **the same noise, the
  same price.** ``_noise_info`` and the budget ledger, compared as objects; and
  the schema each backend reports, including the one place they cannot agree.
* :mod:`~test.system.backend_parity.test_noise` -- **the noise is really
  there.** Draw counting (pandas only -- Spark's draws happen in UDF workers and
  cannot be counted from the test process), and a distributional gate that
  checks the noise the pandas backend adds has the moments ``_noise_info``
  promises, and the same shape as Spark's.
* :mod:`~test.system.backend_parity.test_capability_matrix` -- **what pandas
  refuses.** Driven from
  :func:`~tmlt.analytics._query_expr_compiler._backend_support.unsupported_features`
  rather than from a hand-written list, so a query feature added later fails
  this suite until somebody classifies it.

How a test is written
=====================

Through the ``backend`` fixture from :mod:`test.backend_testing`, which runs the
test body once per backend, and
:func:`~test.backend_testing.comparison.assert_frame_equal_across_backends`,
which compares answers rather than frames. Two consequences worth knowing:

* A parity test is *two* tests, and only the Spark half needs a JVM. That is
  what lets the ``test-nojvm`` nox session run the pandas halves of this suite
  with ``-m "not spark"`` and prove they never boot one.
* The cross-backend comparison is usually transitive: both backends are compared
  against the same expected frame. Where a property has no expected value that
  can honestly be written down -- which rows a truncation keeps, what
  ``_noise_info`` returns -- the two backends are compared directly instead, and
  those tests need Spark.

Sessions are shared, on purpose
===============================

Building a Spark Session is the expensive part of a system test, so
:class:`~test.system.backend_parity.tables.ParitySessions` caches one per
(backend, table set, protected change, budget) and every test at infinite budget
runs against the cached one. This is the lifecycle the engine documents -- build
the Session once with the total budget, then run every query against it -- and
an infinite budget is never exhausted, so nothing is hidden by the sharing. The
finite-budget modules build their own Sessions, since there the ledger *is* the
subject.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025
