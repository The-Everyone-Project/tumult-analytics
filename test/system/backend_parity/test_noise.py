"""The noise is really added, once per cell, with the promised distribution.

The P4 bar and the distributional gate.
:mod:`~test.system.backend_parity.test_calibration` shows the two backends
*report* the same noise; this shows the pandas backend actually draws it.

Why this half is pandas-only
============================

Draws cannot be counted on Spark from a test process. A Spark aggregation adds
its noise inside a pandas UDF, which runs in a separate Python worker process, so
a patch installed in the test process is not the function the worker calls: the
count would come back zero and the test would pass for the wrong reason. Core's
own suite counts draws on its pandas measurements for exactly this reason, and
this module follows it.

That leaves a gap, and the distributional gate is what covers it: a *sample* of
Spark's answers can be collected from the test process even though its draws
cannot be counted, so the two backends' noise can be compared as distributions.
Between them, the two halves say the pandas backend draws the right number of
samples and that those samples come from the right distribution as the Spark
ones.

The gate's tolerances
=====================

The distributional tests are statistical and therefore can fail on a run where
nothing is wrong. Every threshold below is set so that this is unlikely enough to
be ignored -- a few parts in ten thousand per assertion -- and each is written as
a multiple of the relevant standard error rather than as a round number, so that
the reasoning is visible and survives a change to the sample size. They are
marked ``slow``.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import math
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tmlt.analytics import (
    KeySet,
    MaxRowsPerID,
    PureDPBudget,
    Query,
    QueryBuilder,
    RhoZCDPBudget,
    Session,
)

from test.backend_testing import BackendFixture
from test.system.backend_parity.tables import ARK, G_KEYS, build_session

################################################################################
# Where the samplers live
################################################################################

_DGAUSS = "tmlt.core.measurements.noise_mechanisms.sample_dgauss"
"""The discrete Gaussian sampler, called once per noisy value.

The same function Core's own draw-count tests patch."""

_DLAPLACE = "tmlt.core.measurements.noise_mechanisms._sample_dlaplace"
"""One of the two double-sided geometric samplers: the large-scale branch.

``AddGeometricNoise`` picks between two samplers by noise scale, because one is
faster for small scales and the other for large. At a scale of 10 or more it
calls this once per value; below 10 it calls
:data:`_GEOMETRIC_EXP` *twice*, since a double-sided geometric is the difference
of two one-sided ones. Both branches are exercised below rather than one, because
"the noise is drawn once per cell" has to be true of whichever sampler runs.
"""

_GEOMETRIC_EXP = "tmlt.core.measurements.noise_mechanisms._sample_geometric_exp_slow"
"""The other double-sided geometric sampler: the small-scale branch."""

_GEOMETRIC_BRANCH_SCALE = 10
"""The noise scale at which ``AddGeometricNoise`` switches samplers."""

_GEOMETRIC_DRAWS_PER_VALUE = 2
"""How many :data:`_GEOMETRIC_EXP` calls one double-sided geometric sample takes."""


################################################################################
# Draw counting
################################################################################

_FIVE_GROUPS = ("a", "b", "c", "d", "no-such-group")
"""The keys :data:`_FIVE_KEYS` declares, kept as a tuple so they can be counted.

Counting them off the ``KeySet`` instead would mean materializing it, and
:meth:`~tmlt.analytics.KeySet.dataframe` materializes to *Spark* -- which would
start a JVM in a module that exists to be run without one.
"""

_FIVE_KEYS = KeySet.from_dict({"g": list(_FIVE_GROUPS)})
"""Five declared keys, two of which no row reaches.

The count of draws has to be five, not three: a declared key with no rows must
not be distinguishable from one with a few, so a zero-filled cell is noised like
any other. A backend that noised only the groups its data produced would leak
which keys were empty, and would be caught here.
"""


def _ark_query(keys: KeySet) -> Query:
    """A grouped count over the ID table, truncated to two rows per ID.

    Args:
        keys: The keyset to group by.

    Returns:
        The query.
    """
    return QueryBuilder("t").enforce(MaxRowsPerID(2)).groupby(keys).count()


@pytest.mark.parametrize("cells", [1, 2, 5])
def test_a_grouped_count_draws_one_sample_per_declared_key(cells: int):
    """A zCDP grouped count draws exactly as many samples as it has cells.

    Parametrized over the number of cells rather than run once, so that the
    assertion is "one draw per cell" and not "some fixed number of draws".
    """
    backend = BackendFixture(name="pandas")
    keys = KeySet.from_dict({"g": [f"key-{index}" for index in range(cells)]})
    with backend.feature_flag():
        session = build_session(backend, ARK, RhoZCDPBudget(100))
        with patch(_DGAUSS, return_value=0) as sampler:
            result = session.evaluate(_ark_query(keys), RhoZCDPBudget(0.5))
    assert sampler.call_count == cells
    assert len(result) == cells


def test_an_ungrouped_count_draws_exactly_one_sample():
    """A total count is one noisy value, so it is one draw.

    An ungrouped aggregation is a group-by over a keyset with no columns, and the
    interesting failure would be a backend that noised the scalar twice -- once
    inside the grouped measurement it is built from and once on the way out.
    """
    backend = BackendFixture(name="pandas")
    with backend.feature_flag():
        session = build_session(backend, ARK, RhoZCDPBudget(100))
        query = QueryBuilder("t").enforce(MaxRowsPerID(2)).count()
        with patch(_DGAUSS, return_value=0) as sampler:
            session.evaluate(query, RhoZCDPBudget(0.5))
    assert sampler.call_count == 1


def test_an_infinite_budget_draws_nothing():
    """At an infinite budget the mechanism short-circuits and draws no samples.

    This is what makes every exact-value test in this suite meaningful: the
    answers there are the true answers because no sample was taken, not because
    the samples happened to be zero.
    """
    backend = BackendFixture(name="pandas")
    budget = RhoZCDPBudget(float("inf"))
    with backend.feature_flag():
        session = build_session(backend, ARK, budget)
        with patch(_DGAUSS, return_value=0) as sampler:
            session.evaluate(_ark_query(_FIVE_KEYS), budget)
    assert sampler.call_count == 0


def test_an_infinite_pure_dp_budget_draws_nothing():
    """The same short circuit on the geometric mechanism.

    Both samplers are patched, because which one a scale of zero would have
    reached is not the point -- neither must be called at all.
    """
    backend = BackendFixture(name="pandas")
    budget = PureDPBudget(float("inf"))
    with backend.feature_flag():
        session = build_session(backend, ARK, budget)
        with (
            patch(_DLAPLACE, return_value=0) as large,
            patch(_GEOMETRIC_EXP, return_value=0) as small,
        ):
            session.evaluate(_ark_query(_FIVE_KEYS), budget)
    assert large.call_count == 0
    assert small.call_count == 0


@pytest.mark.parametrize(
    "epsilon",
    [
        # Under MaxRowsPerID(2) the geometric scale is 2 / epsilon, so this is a
        # scale of 20 -- the large-scale branch -- and ...
        pytest.param(0.1, id="large-scale-branch"),
        # ... this is a scale of 4, the small-scale one.
        pytest.param(0.5, id="small-scale-branch"),
    ],
)
def test_a_pure_dp_grouped_count_draws_one_geometric_sample_per_key(epsilon: float):
    """Whichever geometric sampler runs, it runs once per declared key.

    The scale decides which of Core's two samplers ``AddGeometricNoise`` calls,
    and the small-scale one is called twice per value rather than once. Both
    branches are covered, and the expected count is derived from the scale the
    engine itself reports rather than from the branch being assumed.
    """
    backend = BackendFixture(name="pandas")
    cells = len(_FIVE_GROUPS)
    with backend.feature_flag():
        session = build_session(backend, ARK, PureDPBudget(100))
        query = _ark_query(_FIVE_KEYS)
        scale = session._noise_info(query, PureDPBudget(epsilon))[0]["noise_parameter"]
        with (
            patch(_DLAPLACE, return_value=0) as large,
            patch(_GEOMETRIC_EXP, return_value=0) as small,
        ):
            session.evaluate(query, PureDPBudget(epsilon))

    if scale >= _GEOMETRIC_BRANCH_SCALE:
        assert (large.call_count, small.call_count) == (cells, 0)
    else:
        assert (large.call_count, small.call_count) == (
            0,
            cells * _GEOMETRIC_DRAWS_PER_VALUE,
        )


def test_counting_draws_on_spark_would_count_nothing(spark):
    """Why the draw-count bar is pandas-only, asserted rather than asserted-in-prose.

    Spark adds its noise in a pandas UDF, which runs in a separate Python worker,
    so a patch installed here is not the function that runs. The count comes back
    zero even though the query really was noised -- which is exactly why a
    draw-count test on Spark would be a test that always passes. Pinning that
    keeps somebody from "fixing" the pandas-only bar by extending it to Spark.

    The answer is checked to be noisy, so that the zero count is evidence about
    where the sampler ran and not about whether it ran.

    Args:
        spark: The Spark session.
    """
    backend = BackendFixture(name="spark", spark=spark)
    query = _ark_query(_FIVE_KEYS)
    with patch(_DGAUSS, return_value=0) as sampler:
        session = build_session(backend, ARK, RhoZCDPBudget(100))
        # A tiny budget, so that the noise is overwhelmingly likely to move at
        # least one of the five cells off its true value.
        answer = session.evaluate(query, RhoZCDPBudget(0.001)).toPandas()
    assert sampler.call_count == 0

    truth = {"a": 3, "b": 2, "c": 2, "d": 0, "no-such-group": 0}
    noisy = [row["count"] - truth[row["g"]] for row in answer.to_dict("records")]
    assert any(residual != 0 for residual in noisy), (
        "At rho=0.001 the five cells should not all land on their true values; "
        "if they did, this test has not shown anything about where the sampler "
        f"ran. Residuals: {noisy}."
    )


################################################################################
# The distributional gate
################################################################################

_GATE_BUDGET = RhoZCDPBudget(0.5)
"""The budget each repeat of the distributional gate spends."""

_GATE_TRUTH: Dict[Any, int] = {"a": 3, "b": 2, "c": 2, "d": 0}
"""The true counts of :data:`~test.system.backend_parity.tables.ID_SPEC` under
``MaxRowsPerID(2)``, grouped by ``g`` over
:data:`~test.system.backend_parity.tables.G_KEYS`.

Taken from the exact-value grid -- these are the numbers
:mod:`~test.system.backend_parity.test_answers` proves both backends return at an
infinite budget -- so the residuals below are the noise and nothing else.
"""

_PANDAS_REPEATS = 500
"""How many times the gate evaluates the query on pandas."""

_SPARK_REPEATS = 40
"""How many times on Spark.

Fewer, because a Spark evaluate is about fifty times slower than a pandas one and
the two-sample test does not need balanced samples. It costs power, not validity.
"""

_SIGMA_TOLERANCE_SES = 4.0
"""How many standard errors of slack each moment assertion allows.

Four, on both moments. For the mean of ``n`` residuals the standard error is
``sigma / sqrt(n)``, and for the sample standard deviation it is approximately
``sigma / sqrt(2 (n - 1))``; a four-standard-error band is exceeded with
probability about 6e-5 per assertion under the null, which is rare enough to
ignore in a suite this size and still tight enough to catch a mechanism whose
scale is wrong by a few percent.
"""

_KS_CRITICAL_COEFFICIENT = 1.949
"""The two-sample Kolmogorov-Smirnov coefficient for alpha = 0.001.

The critical value is this times ``sqrt(1/n + 1/m)``. A tighter alpha than the
usual 0.01 because a flaky acceptance test is worse than a slightly less powerful
one, and because the two samples are large enough that the loss of power is small.

The statistic is conservative here for a second reason: both samples are drawn
from a *discrete* distribution, and the two-sample KS null distribution assumes a
continuous one, which makes the true false-positive rate lower than alpha rather
than higher.
"""


def _residuals(session: Session, repeats: int) -> np.ndarray:
    """Evaluates the gate's query repeatedly and returns the noise it added.

    Args:
        session: The Session to evaluate in. Its budget must cover ``repeats``
            evaluations at :data:`_GATE_BUDGET`.
        repeats: How many times to evaluate.

    Returns:
        One residual per cell per repeat: the noisy count minus the true one.
    """
    query = _ark_query(G_KEYS)
    collected: List[float] = []
    for _ in range(repeats):
        answer = session.evaluate(query, _GATE_BUDGET)
        frame = answer if isinstance(answer, pd.DataFrame) else answer.toPandas()
        for row in frame.to_dict("records"):
            collected.append(float(row["count"] - _GATE_TRUTH[row["g"]]))
    return np.array(collected, dtype=float)


def _gate_session(backend: BackendFixture, repeats: int) -> Session:
    """Builds a Session with enough budget for the gate's repeats.

    One Session for the whole sample, which is both the documented lifecycle and
    the only way to run the experiment: a fresh Session per repeat would reset the
    accountant, which is the thing a privacy engine must never do.

    Args:
        backend: The backend under test.
        repeats: How many evaluations the Session must afford.

    Returns:
        The Session.
    """
    rho = _GATE_BUDGET.rho * repeats
    return build_session(backend, ARK, RhoZCDPBudget(rho))


def _reported_sigma(session: Session) -> float:
    """The standard deviation the engine says it will add, per noisy value.

    ``_noise_info`` reports the discrete Gaussian's parameter as sigma *squared*,
    so the standard deviation is its square root. Reading it from the engine
    rather than deriving it here is deliberate: the gate then tests that the noise
    matches what the engine *claims*, which is the number a user calibrates
    against.

    Args:
        session: The Session the gate runs in.

    Returns:
        The reported sigma.
    """
    info = session._noise_info(_ark_query(G_KEYS), _GATE_BUDGET)
    return math.sqrt(float(info[0]["noise_parameter"]))


def _ks_statistic(first: np.ndarray, second: np.ndarray) -> float:
    """The two-sample Kolmogorov-Smirnov statistic: the largest ECDF gap.

    Computed here rather than taken from scipy so that what is being compared,
    and at what threshold, is visible in this file.

    Args:
        first: One sample.
        second: The other.

    Returns:
        The maximum absolute difference between the two empirical CDFs.
    """
    left = np.sort(np.asarray(first, dtype=float))
    right = np.sort(np.asarray(second, dtype=float))
    points = np.union1d(left, right)
    left_cdf = np.searchsorted(left, points, side="right") / len(left)
    right_cdf = np.searchsorted(right, points, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _ks_critical(first_size: int, second_size: int) -> float:
    """The KS critical value at :data:`_KS_CRITICAL_COEFFICIENT`'s alpha.

    Args:
        first_size: The first sample's size.
        second_size: The second's.

    Returns:
        The threshold the statistic must stay below.
    """
    return _KS_CRITICAL_COEFFICIENT * math.sqrt(1 / first_size + 1 / second_size)


@pytest.mark.slow
def test_the_pandas_noise_has_the_moments_the_engine_promises():
    """The noise pandas adds is centered, and as wide as ``_noise_info`` says.

    A draw count says a sample was taken; it says nothing about what was
    sampled -- Core's ``sample_dgauss`` could be called the right number of times
    with the wrong scale, or with the scale of a different mechanism, and every
    other test in this suite would still pass. This is the test that looks at the
    numbers.
    """
    backend = BackendFixture(name="pandas")
    with backend.feature_flag():
        session = _gate_session(backend, _PANDAS_REPEATS)
        sigma = _reported_sigma(session)
        residuals = _residuals(session, _PANDAS_REPEATS)

    count = len(residuals)
    assert count == _PANDAS_REPEATS * len(_GATE_TRUTH)

    mean_bound = _SIGMA_TOLERANCE_SES * sigma / math.sqrt(count)
    assert abs(residuals.mean()) < mean_bound, (
        f"The noise is not centered: mean {residuals.mean():+.4f}, bound "
        f"+/-{mean_bound:.4f} (sigma={sigma}, n={count})."
    )

    deviation = float(residuals.std(ddof=1))
    deviation_bound = _SIGMA_TOLERANCE_SES * sigma / math.sqrt(2 * (count - 1))
    assert abs(deviation - sigma) < deviation_bound, (
        f"The noise is the wrong width: sd {deviation:.4f}, expected "
        f"{sigma} +/- {deviation_bound:.4f} (n={count})."
    )

    # A discrete Gaussian's samples are integers, and a count plus an integer is
    # an integer: a residual that was not would mean the noise had been added
    # through a float somewhere.
    assert np.all(residuals == np.round(residuals))


@pytest.mark.slow
def test_the_two_backends_noise_has_the_same_distribution(spark):
    """Spark's residuals and pandas' residuals look like the same distribution.

    The half of the noise bar that Spark can be held to. Its draws cannot be
    counted from here, but its answers can be collected, so the comparison is
    made where the difference would show up: in the distribution of the noise
    the two backends actually added to the same query over the same data.

    Args:
        spark: The Spark session; this test needs both backends at once.
    """
    samples: Dict[str, np.ndarray] = {}
    reported: Dict[str, float] = {}
    for name, repeats in (("pandas", _PANDAS_REPEATS), ("spark", _SPARK_REPEATS)):
        fixture = BackendFixture(name=name, spark=spark if name == "spark" else None)
        with fixture.feature_flag():
            session = _gate_session(fixture, repeats)
            reported[name] = _reported_sigma(session)
            samples[name] = _residuals(session, repeats)

    # Both backends promised the same scale -- test_calibration proves that in
    # general; here it is a precondition for reading the comparison below.
    assert reported["pandas"] == reported["spark"]

    statistic = _ks_statistic(samples["pandas"], samples["spark"])
    critical = _ks_critical(len(samples["pandas"]), len(samples["spark"]))
    assert statistic < critical, (
        "The two backends' noise does not look like the same distribution: "
        f"KS statistic {statistic:.4f}, critical value {critical:.4f} at "
        f"alpha=0.001 (n={len(samples['pandas'])}, m={len(samples['spark'])})."
    )

    # And Spark's own noise has the promised scale, to the precision its smaller
    # sample allows. Stated separately from the two-sample test because that test
    # would also pass if *both* backends were wrong in the same way.
    spark_residuals = samples["spark"]
    sigma = reported["spark"]
    bound = _SIGMA_TOLERANCE_SES * sigma / math.sqrt(2 * (len(spark_residuals) - 1))
    assert abs(float(spark_residuals.std(ddof=1)) - sigma) < bound


@pytest.mark.slow
def test_the_gate_would_notice_the_wrong_scale():
    """The moment assertions are tight enough to reject noise of the wrong width.

    A tolerance test rather than an engine test: the two gates above pass, and
    the question this answers is whether they would have failed if the noise had
    been wrong. It builds samples at the reported sigma and at sigma scaled by a
    small factor, and checks that the second is outside the band the first is
    inside.

    Uses numpy's generator rather than the engine, seeded, so that it is a
    statement about the *tolerance* and not another sample from the mechanism.
    """
    backend = BackendFixture(name="pandas")
    with backend.feature_flag():
        session = _gate_session(backend, 1)
        sigma = _reported_sigma(session)

    count = _PANDAS_REPEATS * len(_GATE_TRUTH)
    bound = _SIGMA_TOLERANCE_SES * sigma / math.sqrt(2 * (count - 1))
    generator = np.random.default_rng(20250101)

    right = np.round(generator.normal(0.0, sigma, count))
    assert abs(float(right.std(ddof=1)) - sigma) < bound

    # Ten percent is the smallest error worth calling a bug in a noise scale, and
    # the band above is about 3% of sigma wide, so it is comfortably rejected.
    wrong = np.round(generator.normal(0.0, sigma * 1.1, count))
    assert abs(float(wrong.std(ddof=1)) - sigma) > bound
