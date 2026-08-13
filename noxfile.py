"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.
"""

import platform
import sys
from pathlib import Path

import nox
from nox import session as session
from tmlt.nox_utils import DependencyConfiguration, SessionManager, install_group

nox.options.default_venv_backend = "uv|virtualenv"

CWD = Path(".").resolve()

PACKAGE_NAME = "tmlt.analytics"
"""Name of the package."""
PACKAGE_GITHUB = "opendp/tumult-analytics"
"""GitHub organization/project."""
SMOKETEST_SCRIPT = """
from tmlt.analytics.utils import check_installation
check_installation()
"""
"""Python script to run as a quick self-test."""

MIN_COVERAGE = 75
"""For test suites where we track coverage (i.e. the fast tests and the full
test suite), fail if test coverage falls below this percentage."""

# NOTE (The-Everyone-Project fork): the no-JVM test lane. Mirrors the session of
# the same name in the Core fork's noxfile.
NOJVM_TEST_PATHS = [
    # The backend parity suite. Every test in it that takes the `backend` fixture
    # is half Spark and half pandas; the Spark halves carry the `spark` marker and
    # are deselected, so what runs here is the pandas half of the acceptance
    # suite -- including its draw counts and its distributional gate, neither of
    # which needs Spark at all.
    CWD / "test" / "system" / "backend_parity",
    # The pandas backend's own suites.
    CWD / "test" / "unit" / "test_pandas_evaluate.py",
    CWD / "test" / "unit" / "test_backends.py",
    CWD / "test" / "unit" / "test_unsupported_surface.py",
    CWD / "test" / "unit" / "test_neighboring_relation_pandas.py",
    CWD / "test" / "unit" / "test_session_pandas.py",
    CWD / "test" / "unit" / "test_coerce_pandas_schema.py",
    CWD / "test" / "unit" / "test_pandas_schema_conversion.py",
    # The hash-seed check on the answer's column order. It answers on pandas, in
    # subprocesses of its own, and neither they nor the parent reach for a JVM.
    CWD / "test" / "unit" / "test_answer_column_order.py",
    # KeySet materialization. Half of this file compares the pandas frame against
    # the Spark one, and half asks only about the pandas frame. The comparisons
    # carry the `spark` marker written by hand: they reach Spark through
    # KeySet.dataframe() inside the test body, where the collection hook -- which
    # reads fixture closures -- cannot see it.
    CWD / "test" / "unit" / "keysets" / "test_pandas_materialization.py",
    # The parity harness's self-tests. Core's lane includes its own for the same
    # reason: the harness is what every test above is written against, so a lane
    # that ran them without it would not be checking the same thing.
    CWD / "test" / "unit" / "test_backend_testing.py",
    # The check that the documented feature matrix is the computed one: it reads
    # the same tables the compiler's rejection gate reads, and the docs sources.
    # No data and no engine, so no JVM.
    CWD / "test" / "unit" / "test_docs_feature_matrix.py",
]
"""Test paths the test-nojvm session runs.

Every path here has been checked to pass with ``TMLT_FORBID_JVM=1`` and
``-m "not spark"``.
"""


def is_mac():
    """Returns true if the current system is a mac."""
    return sys.platform == "darwin"


# NOTE (The-Everyone-Project fork): the "oldest" cells below used to pin
# `"tmlt.core": "==0.19.1"`, which resolves the *upstream PyPI* 0.19.1 wheel.
# That wheel does not contain `tmlt.core.utils.pandas_truncation`, which the
# pandas backend depends on, so those two matrix cells would install a Core that
# cannot run this branch's tests.
#
# `"==0.19.1+ep.pandas.1"` is not a fix: the fork build is deliberately a PEP 440
# local version, and PyPI rejects local versions, so there is nothing to resolve
# against. Instead we point those cells at the published fork wheel as a PEP 508
# direct reference. tmlt.nox_utils builds each pin by plain string concatenation
# (`sess.install(pkg + version)`, see nox_utils._session_manager), so a value
# starting with " @ <url>" produces the valid requirement
# `tmlt.core @ https://.../tmlt_core-...whl`. This is the least invasive fix that
# still exercises the real floor version.
#
# The fork wheels are `py3-none-<platform>` (pure-Python tag, platform-specific
# vendored libs), so one URL per platform covers Python 3.10-3.12.
#
# TODO: the `>=0.19.1` cells still resolve from PyPI. They are currently
# harmless -- `uv pip install` keeps the already-installed 0.19.1+ep.pandas.1,
# which satisfies the constraint -- but if upstream ever publishes a 0.19.x
# newer than the fork base, those cells will silently upgrade to a Core without
# the pandas modules. Repoint them at the fork wheel if that happens.
_EP_CORE_WHEEL_BASE = (
    "https://github.com/The-Everyone-Project/tumult-core/releases/download/"
    "0.19.1-ep-pandas-1/tmlt_core-0.19.1+ep.pandas.1-py3-none-"
)


def ep_core_wheel():
    """Direct-reference pin for the Everyone Project tmlt.core fork build.

    Returned as a `" @ <url>"` suffix so that nox_utils' `pkg + version`
    concatenation yields a valid PEP 508 requirement.
    """
    if is_mac():
        arch = "arm64" if platform.machine() == "arm64" else "x86_64"
        return f" @ {_EP_CORE_WHEEL_BASE}macosx_11_0_{arch}.whl"
    return f" @ {_EP_CORE_WHEEL_BASE}manylinux_2_17_x86_64.manylinux2014_x86_64.whl"


DEPENDENCY_MATRIX = [
    DependencyConfiguration(
        id="3.10-oldest",
        python="3.10",
        packages={
            "pyspark[sql]": "==3.3.1" if not is_mac() else "==3.5.0",
            "sympy": "==1.8",
            "pandas": "==1.4.0",
            "tmlt.core": ep_core_wheel(),
        },
    ),
    DependencyConfiguration(
        id="3.10-newest",
        python="3.10",
        packages={
            "pyspark[sql]": "==3.5.8",
            "sympy": "==1.9",
            "pandas": "==1.5.3",
            "tmlt.core": ">=0.19.1",
        },
    ),
    DependencyConfiguration(
        id="3.11-oldest",
        python="3.11",
        packages={
            "pyspark[sql]": "==3.4.0" if not is_mac() else "==3.5.0",
            "sympy": "==1.8",
            "pandas": "==1.5.0",
            "tmlt.core": ep_core_wheel(),
        },
    ),
    DependencyConfiguration(
        id="3.11-newest",
        python="3.11",
        packages={
            "pyspark[sql]": "==3.5.8",
            "sympy": "==1.12",
            "pandas": "==1.5.3",
            "tmlt.core": ">=0.19.1",
        },
    ),
    DependencyConfiguration(
        id="3.12-oldest",
        python="3.12",
        packages={
            "pyspark[sql]": "==4.0.0",
             "sympy": "==1.8",
             "pandas": "==2.2.0",
             "numpy": "==1.26.0",
             "scipy": "==1.11.2",
             "randomgen": "==1.26.0",
             "pyarrow": "==18.0.0",
             "tmlt.core": ">=0.19.1",
        },
    ),
    DependencyConfiguration(
        id="3.12-newest",
        python="3.12",
        packages={
            "pyspark[sql]": "==4.1.0",
             "sympy": "==1.12",
             "pandas": "==2.3.3",
             "numpy": "==1.26.4",
             "scipy": "==1.17.1",
             "randomgen": "==2.3.0",
             "pyarrow": "==18.1.0",
             "tmlt.core": ">=0.19.1",
        },
    ),
]

AUDIT_VERSIONS = ["3.10", "3.11", "3.12"]
AUDIT_SUPPRESSIONS = [
    "PYSEC-2023-228",
    # Affects: pip<23.3
    # Notice: Command Injection in pip when used with Mercurial
    # Link: https://github.com/advisories/GHSA-mq26-g339-26xf
    # Impact: None, we don't use Mercurial, and in any case we assume that users will
    #         have their own pip installations -- it is not a dependency of Analytics.
    "PYSEC-2017-147",
    # Affects: PySpark 1.6 through 2.1
    # Link: https://nvd.nist.gov/vuln/detail/CVE-2017-12612
    # Impact: None, we don't support these versions of PySpark. This appears to
    #         be showing up due to a bad data import into the PyPA vulnerability
    #         database [0], which they are aware of and working to fix [1], but
    #         in the mean time we are also ignoring it here.
    # [0] https://github.com/pypa/advisory-database/commit/c9b8e1f96953321b54b796baef731c8f72587115
    # [1] https://github.com/pypa/advisory-database/issues/207#issuecomment-2491830484
]

# Dictionary mapping benchmark paths to the corresponding timeouts, in minutes.
# The goal is to make sure we don't have major performance regression, so the
# timeouts have been set in https://github.com/opendp/tumult-analytics/pull/49
# to 25% longer than one run on the GitHub runners (rounded up).
BENCHMARK_TO_TIMEOUT = {
    "keyset_projection": 14 * 60,
    "keyset_cross_product_per_size": 45 * 60,
    "keyset_cross_product_per_factors": 37 * 60,
}

sm = SessionManager(
    package=PACKAGE_NAME,
    package_github=PACKAGE_GITHUB,
    directory=CWD,
    default_python_version="3.10",
    smoketest_script=SMOKETEST_SCRIPT,
    parallel_tests=False,
    min_coverage=MIN_COVERAGE,
    audit_versions=AUDIT_VERSIONS,
    audit_suppressions=AUDIT_SUPPRESSIONS,
)

sm.build()

sm.ruff_format()
sm.ruff_check()
sm.mypy()

sm.smoketest()
sm.release_smoketest()
sm.test()
sm.test_fast()
sm.test_slow()
sm.test_doctest()


# NOTE (The-Everyone-Project fork): the no-JVM lane, mirroring Core's.
@session(name="test-nojvm", tags=["test"], python="3.10")
@sm._install_package  # noqa: SLF001
@install_group("test")
def test_nojvm(sess):
    """Run the tests that must not start a JVM.

    pyspark is installed here exactly as it is everywhere else -- it is an
    unconditional dependency, and modules like tmlt.analytics._backends import it
    at module scope. What this session checks is the stronger, and more useful,
    property that the pandas code paths never *start* one: TMLT_FORBID_JVM makes
    the guard in test/conftest.py replace pyspark's launch_gateway, so any test
    that reaches for a Spark session fails loudly instead of quietly booting a
    JVM. The Spark-dependent tests are deselected by '-m "not spark"', which the
    same conftest applies structurally rather than test by test.

    This reuses SessionManager's private helpers rather than duplicating its
    pytest invocation. noxfile.py is not linted, and keeping the argument list in
    one place is worth the private access.

    Args:
        sess: The nox session.
    """
    sess.env["TMLT_FORBID_JVM"] = "1"
    sm._test(  # noqa: SLF001
        sess, "not spark", min_coverage=0, test_paths=NOJVM_TEST_PATHS
    )


sm.docs_linkcheck()
sm.docs_doctest()
sm.docs()

for name, timeout in BENCHMARK_TO_TIMEOUT.items():
    sm.benchmark(Path("benchmark") / f"{name}.py", timeout)

sm.audit()

sm.make_release()

sm.test_dependency_matrix(DEPENDENCY_MATRIX)
