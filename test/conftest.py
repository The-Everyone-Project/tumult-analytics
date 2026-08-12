"""Creates a Spark Context to use for each testing session."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

# TODO(#2206): Import these fixtures from core once it is rewritten

import atexit
import logging
import os
import sys
from typing import Any, Dict, Iterator, List, NoReturn, TypeVar, Union, cast, overload
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd
import pytest
from pyspark import java_gateway
from pyspark.context import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType
from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import Measure, PureDP
from tmlt.core.metrics import AbsoluteDifference, Metric
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils import cleanup as core_cleanup
from tmlt.core.utils.exact_number import ExactNumber

from tmlt.analytics import (
    ApproxDPBudget,
    BinningSpec,
    KeySet,
    MaxGroupsPerID,
    MaxRowsPerGroupPerID,
    MaxRowsPerID,
    PrivacyBudget,
    PureDPBudget,
    QueryBuilder,
    RhoZCDPBudget,
)
from tmlt.analytics._schema import ColumnDescriptor, ColumnType, Schema
from tmlt.analytics.truncation_strategy import TruncationStrategy

SIMPLE_TRANSFORMATION_QUERIES = [
    QueryBuilder("private_data").rename({"D": "new"}),
    QueryBuilder("private_data").filter("C>1"),
    QueryBuilder("private_data").select(["A", "B", "C"]),
    QueryBuilder("private_data").map(
        f=lambda row: {"F": 1},
        new_column_types=Schema({"F": "INTEGER"}),
        augment=True,
    ),
    QueryBuilder("private_data").flat_map(
        f=lambda row: [{"F": 1}],
        new_column_types=Schema({"F": "INTEGER"}),
        augment=True,
        max_rows=2,
    ),
    QueryBuilder("private_data").join_private(
        "join_private_data",
        truncation_strategy_left=TruncationStrategy.DropExcess(1),
        truncation_strategy_right=TruncationStrategy.DropNonUnique(),
    ),
    QueryBuilder("private_data").join_public("join_public_data"),
    QueryBuilder("private_data").replace_null_and_nan(),
    QueryBuilder("private_data").replace_infinity({"C": (-100, 100)}),
    QueryBuilder("private_data").drop_null_and_nan(["C"]),
    QueryBuilder("private_data").drop_infinity(["C"]),
    QueryBuilder("private_data").bin_column(
        column="A", spec=BinningSpec(bin_edges=[1000 * i for i in range(0, 10)])
    ),
]

KEY_SET = KeySet.from_dict(
    {
        "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True).tolist(),
        "B": np.random.choice(np.arange(0, 100, 1), 100, replace=True).tolist(),
    }
)
KEY_SET.cache()


GROUPBY_AGGREGATION_QUERIES = [
    lambda x: x.groupby(KEY_SET).count("measure_col"),
    lambda x: x.groupby(KEY_SET).sum("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).variance("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).stdev("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).min("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).max("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).median("C", low=0, high=1000, name="measure_col"),
    lambda x: x.groupby(KEY_SET).count_distinct(name="measure_col"),
    lambda x: x.groupby(KEY_SET).quantile(
        "C", 0.5, low=0, high=1000, name="measure_col"
    ),
    lambda x: x.groupby(KEY_SET).count(name="measure_col").suppress(1),
    # TODO(#3342): Enable after get_bounds core slowness is fixed
    # lambda x: x.groupby(KEY_SET).get_bounds("C", lower_bound_column="measure_col"),
]

NON_GROUPBY_AGGREGATION_QUERIES = [
    QueryBuilder("private_data").count(name="measure_col"),
    QueryBuilder("private_data").count_distinct(columns=["A", "B"], name="measure_col"),
    QueryBuilder("private_data").quantile("A", 0.5, 0, 1000, name="measure_col"),
    QueryBuilder("private_data").min("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").max("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").median("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").sum("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").average("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").variance("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").stdev("A", 0, 1000, name="measure_col"),
    QueryBuilder("private_data").get_groups(["A"]),
    QueryBuilder("private_data").histogram(
        column="A",
        bin_edges=BinningSpec(bin_edges=[1000 * i for i in range(0, 10)]),
        name="measure_col",
    ),
    # TODO(#3342): Enable after get_bounds core slowness is fixed
    # QueryBuilder("private_data").get_bounds("C", lower_bound_column="measure_col"),
]

ID_QUERIES = [
    QueryBuilder("id_a_private_data").enforce(MaxRowsPerID(1)).count(),
    QueryBuilder("id_a_private_data")
    .enforce(MaxRowsPerID(100))
    .filter("id >= 2")
    .groupby(KEY_SET)
    .count("measure_col"),
    QueryBuilder("id_a_private_data")
    .enforce(MaxGroupsPerID("X", 1))
    .enforce(MaxRowsPerGroupPerID("X", 1))
    .count(),
    QueryBuilder("id_a_private_data")
    .flat_map_by_id(
        lambda rows: [{"per_id_sum": sum(r["A"] for r in rows)}],
        new_column_types={
            "per_id_sum": ColumnDescriptor(
                ColumnType.INTEGER,
                allow_null=False,
            )
        },
    )
    .enforce(MaxRowsPerID(1))
    .sum("per_id_sum", low=0, high=5, name="sum"),
]


def quiet_py4j():
    """Remove noise in the logs irrelevant to testing."""
    print("Calling PySparkTest:suppress_py4j_logging")
    logger = logging.getLogger("py4j")
    # This is to silence py4j.java_gateway: DEBUG logs.
    logger.setLevel(logging.ERROR)


# this initializes one shared spark session for the duration of the test session.
# another option may be to set the scope to "module", which changes the duration to
# one session per module
@pytest.fixture(scope="session", name="spark")
def pyspark():
    """Setup a context to execute pyspark tests."""
    quiet_py4j()
    print("Setting up spark session.")
    spark = (
        SparkSession.builder.appName("analytics-test")
        .master("local[4]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.eventLog.enabled", "false")
        .config("spark.driver.allowMultipleContexts", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.default.parallelism", "5")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "16g")
        .config("spark.driver.memory", "2g")
        .config("spark.port.maxRetries", "30")
        .config("spark.sql.shuffle.partitions", "1")
        # Disable Spark UI / Console display
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.dagGraph.retainedRootRDDs", "1")
        .config("spark.ui.retainedJobs", "1")
        .config("spark.ui.retainedStages", "1")
        .config("spark.ui.retainedTasks", "1")
        .config("spark.sql.ui.retainedExecutions", "1")
        .config("spark.worker.ui.retainedExecutors", "1")
        .config("spark.worker.ui.retainedDrivers", "1")
        .getOrCreate()
    )
    # This is to silence pyspark logs.
    spark.sparkContext.setLogLevel("OFF")
    return spark


@pytest.fixture(scope="function", name="spark_with_progress")
def pyspark_with_progress():
    """A context to execute pyspark tests, with spark.ui.showConsoleProgress enabled."""
    quiet_py4j()
    print("Setting up spark session.")
    spark = (
        SparkSession.builder.appName("analytics-test-with-progress")
        .master("local[4]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.eventLog.enabled", "false")
        .config("spark.driver.allowMultipleContexts", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.default.parallelism", "5")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "16g")
        .config("spark.driver.memory", "2g")
        .config("spark.port.maxRetries", "30")
        .config("spark.sql.shuffle.partitions", "1")
        # Disable Spark UI, leave console display enabled
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.dagGraph.retainedRootRDDs", "1")
        .config("spark.ui.retainedJobs", "1")
        .config("spark.ui.retainedStages", "1")
        .config("spark.ui.retainedTasks", "1")
        .config("spark.sql.ui.retainedExecutions", "1")
        .config("spark.worker.ui.retainedExecutors", "1")
        .config("spark.worker.ui.retainedDrivers", "1")
        .getOrCreate()
    )
    # This is to silence pyspark logs.
    spark.sparkContext.setLogLevel("OFF")
    return spark


def create_mock_measurement(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_measure: Measure = PureDP(),
    is_interactive: bool = False,
    return_value: Any = np.int64(0),
    privacy_function_implemented: bool = False,
    privacy_function_return_value: Any = ExactNumber(1),
    privacy_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Measurement with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_measure: Output measure for the mock.
        is_interactive: Whether the mock should be interactive.
        return_value: Return value for the Measurement's __call__.
        privacy_function_implemented: If True, raises a :class:`NotImplementedError`
            with the message "TEST" when the privacy function is called.
        privacy_function_return_value: Return value for the Measurement's privacy
            function.
        privacy_relation_return_value: Return value for the Measurement's privacy
            relation.
    """
    measurement = create_autospec(spec=Measurement, instance=True)
    measurement.input_domain = input_domain
    measurement.input_metric = input_metric
    measurement.output_measure = output_measure
    measurement.is_interactive = is_interactive
    measurement.return_value = return_value
    measurement.privacy_function.return_value = privacy_function_return_value
    measurement.privacy_relation.return_value = privacy_relation_return_value
    if not privacy_function_implemented:
        measurement.privacy_function.side_effect = NotImplementedError("TEST")
    return measurement


def create_mock_transformation(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_domain: Domain = NumpyIntegerDomain(),
    output_metric: Metric = AbsoluteDifference(),
    return_value: Any = 0,
    stability_function_implemented: bool = False,
    stability_function_return_value: Any = ExactNumber(1),
    stability_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Transformation with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_domain: Output domain for the mock.
        output_metric: Output metric for the mock.
        return_value: Return value for the Transformation's __call__.
        stability_function_implemented: If False, raises a :class:`NotImplementedError`
            with the message "TEST" when the stability function is called.
        stability_function_return_value: Return value for the Transformation's stability
            function.
        stability_relation_return_value: Return value for the Transformation's stability
            relation.
    """
    transformation = create_autospec(spec=Transformation, instance=True)
    transformation.input_domain = input_domain
    transformation.input_metric = input_metric
    transformation.output_domain = output_domain
    transformation.output_metric = output_metric
    transformation.return_value = return_value
    transformation.stability_function.return_value = stability_function_return_value
    transformation.stability_relation.return_value = stability_relation_return_value
    transformation.__or__ = Transformation.__or__
    if not stability_function_implemented:
        transformation.stability_function.side_effect = NotImplementedError("TEST")
    return transformation


T = TypeVar("T", bound=PrivacyBudget)


def assert_approx_equal_budgets(
    budget1: T, budget2: T, atol: float = 1e-8, rtol: float = 1e-5
):
    """Asserts that two budgets are approximately equal.

    Args:
        budget1: The first budget.
        budget2: The second budget.
        atol: The absolute tolerance for the comparison.
        rtol: The relative tolerance for the comparison.
    """
    if not isinstance(budget1, type(budget2)) or not isinstance(budget2, type(budget1)):
        raise AssertionError(
            f"Budgets are not of the same type: {type(budget1)} and {type(budget2)}"
        )
    if isinstance(budget1, PureDPBudget) and isinstance(budget2, PureDPBudget):
        if not np.allclose(budget1.epsilon, budget2.epsilon, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Epsilon values are not approximately equal: {budget1} and {budget2}"
            )
        return
    if isinstance(budget1, ApproxDPBudget) and isinstance(budget2, ApproxDPBudget):
        if not np.allclose(budget1.epsilon, budget2.epsilon, atol=atol, rtol=rtol):
            raise AssertionError(
                "Epsilon values are not approximately equal: "
                f"{budget1.epsilon} and {budget2.epsilon}"
            )
        if not np.allclose(budget1.delta, budget2.delta, atol=atol, rtol=rtol):
            raise AssertionError(
                "Delta values are not approximately equal: "
                f"{budget1.delta} and {budget2.delta}"
            )
        return
    if isinstance(budget1, RhoZCDPBudget) and isinstance(budget2, RhoZCDPBudget):
        if not np.allclose(budget1.rho, budget2.rho, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Rho values are not approximately equal: "
                f"{budget1.rho} and {budget2.rho}"
            )
        return
    raise AssertionError(f"Budget type not recognized: {type(budget1)}")


@overload
def create_empty_input(domain: DictDomain) -> Dict: ...


@overload
def create_empty_input(domain: SparkDataFrameDomain) -> DataFrame: ...


def create_empty_input(domain):
    """Returns an empty input for a given domain.

    Args:
        domain: The domain for which to create an empty input.
    """
    spark = SparkSession.builder.getOrCreate()
    if isinstance(domain, DictDomain):
        return {
            k: create_empty_input(cast(Union[DictDomain, SparkDataFrameDomain], v))
            for k, v in domain.key_to_domain.items()
        }
    if isinstance(domain, SparkDataFrameDomain):
        # TODO(#3092): the row is only necessary b/c of a bug in core for empty dfs
        row: List[Any] = []
        for field in domain.spark_schema.fields:
            if field.dataType.simpleString() == "string":
                row.append("")
            elif field.dataType.simpleString() == "integer":
                row.append(0)
            elif field.dataType.simpleString() == "double":
                row.append(0.0)
            elif field.dataType.simpleString() == "boolean":
                row.append(False)
            elif field.dataType.simpleString() == "bigint":
                row.append(0)
            else:
                raise ValueError(
                    f"Unsupported field type: {field.dataType.simpleString()}"
                )
        return spark.createDataFrame([row], domain.spark_schema)
    raise ValueError(f"Unsupported domain type: {type(domain)}")


def pyspark_schema_from_pandas(df: pd.DataFrame) -> StructType:
    """Create a pyspark schema corresponding to a pandas dataframe."""

    def convert_type(dtype):
        if dtype == np.int64:
            return LongType()
        elif dtype == float:
            return FloatType()
        elif dtype == str:
            return StringType()
        raise NotImplementedError("Type not implemented yet.")

    return StructType(
        [
            StructField(colname, convert_type(dtype))
            for colname, dtype in df.dtypes.items()
        ]
    )


@pytest.fixture(scope="module")
def _session_data(spark):
    base_private_data = pd.DataFrame(
        {
            "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "B": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "C": np.random.choice(np.arange(0, 100, 0.5), 100, replace=True),
            "D": np.random.choice(np.arange(0, 100, 0.5), 100, replace=True),
        }
    )
    private_id_data = pd.DataFrame(
        [
            [1, 4, 100, "X"],
            [1, 5, 100, "Y"],
            [1, 6, 100, "X"],
            [2, 7, 100, "Y"],
            [3, 8, 100, "X"],
            [3, 9, 100, "Y"],
        ],
        columns=["id", "A", "B", "X"],
    )
    join_private_data = pd.DataFrame(
        {
            "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "Y": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
        }
    )
    join_public_data = pd.DataFrame(
        {
            "A": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
            "Z": np.random.choice(np.arange(0, 100, 1), 100, replace=True),
        }
    )
    private_sdf = spark.createDataFrame(base_private_data)
    private_id_sdf = spark.createDataFrame(private_id_data)
    join_private_sdf = spark.createDataFrame(join_private_data)
    join_public_sdf = spark.createDataFrame(join_public_data)
    return {
        "private_data": private_sdf,
        "private_id_data": private_id_sdf,
        "join_private_data": join_private_sdf,
        "join_public_data": join_public_sdf,
    }


################################################################################
# NOTE (The-Everyone-Project fork): keeping the JVM out of the pandas test lane.
#
# Everything below is additive. It defines two new things -- a collection hook
# that marks the Spark-dependent tests, and an opt-in guard that turns starting a
# JVM into a failure -- and changes nothing above: no existing fixture is
# replaced, rewired, or renamed, and with TMLT_FORBID_JVM unset the guard does
# nothing at all, so an ordinary `pytest` run behaves exactly as it did.
#
# This mirrors the same block in the Core fork's test/conftest.py, which the
# `test-nojvm` nox session there has been running against for as long as the
# pandas stack has existed. See noxfile.py's test_nojvm for what the lane is for.
################################################################################


def _requires_spark(item: pytest.Item) -> bool:
    """Returns whether a collected test needs a Spark session to run.

    Two routes, because the suite has two:

    * The ``spark`` fixture (or another fixture that requests it) appearing
      anywhere in the test's fixture closure. ``item.fixturenames`` is the whole
      closure, so this covers ``spark_with_progress`` and ``_session_data`` as
      well as a direct request.
    * The ``backend`` fixture's ``spark`` parameter. A test parametrized over
      backends is *half* a Spark test: the fixture resolves its Spark session with
      ``getfixturevalue`` precisely so that the pandas parameter never starts a
      JVM, which also keeps ``spark`` out of the static closure above. Reading the
      parameter is what tells the two halves apart.

    Args:
        item: The collected test item.

    Returns:
        Whether the item needs a Spark session.
    """
    if "spark" in getattr(item, "fixturenames", ()):
        return True
    callspec = getattr(item, "callspec", None)
    params = getattr(callspec, "params", {}) if callspec is not None else {}
    if params.get("backend") == "spark":
        return True
    backend = params.get("backend")
    return getattr(backend, "name", None) == "spark"


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    """Applies the ``spark`` marker to every test that needs a Spark session.

    Marking structurally rather than by annotating each test keeps this in one
    place: there are hundreds of them, and one that was meant to be marked but
    was not would quietly boot a JVM in the ``test-nojvm`` lane. Tests that reach
    Spark by some third route are not detected here -- :func:`forbid_jvm` is what
    catches those, at runtime.

    Args:
        items: The collected test items, marked in place.
    """
    for item in items:
        if _requires_spark(item):
            item.add_marker(pytest.mark.spark)


FORBID_JVM_ENV_VAR = "TMLT_FORBID_JVM"
"""Setting this to 1 forbids the test process from starting a JVM.

The ``test-nojvm`` nox session sets it; see :func:`forbid_jvm`."""

_FORBID_JVM_MESSAGE = (
    f"{FORBID_JVM_ENV_VAR} is set, but something tried to start a JVM.\n"
    "\n"
    "This test lane runs with pyspark installed and is meant to prove that the "
    "pandas code paths never boot it. If the test that triggered this really "
    "does need a Spark session, make sure it requests the `spark` fixture (or "
    "runs on the `backend` fixture's spark parameter) so that the collection "
    "hook marks it and -m 'not spark' deselects it. Otherwise, a code path that "
    "is supposed to be Spark-free reached for a SparkSession."
)


def _forbidden_launch_gateway(*_args: Any, **_kwargs: Any) -> NoReturn:
    """Stands in for pyspark's ``launch_gateway`` and refuses to start a JVM.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(_FORBID_JVM_MESSAGE)


@pytest.fixture(scope="session", autouse=True)
def forbid_jvm() -> Iterator[None]:
    """Turns any attempt to start a JVM into a loud failure, when opted in.

    Only active when ``TMLT_FORBID_JVM`` is set, so ordinary test runs are
    unaffected -- which is what makes this additive.

    ``launch_gateway`` is the one function that actually spawns the JVM
    (``SparkSession.builder.getOrCreate()`` reaches it through
    ``SparkContext._ensure_initialized``), so replacing it catches every route
    into a Spark session whichever API built it. It has to be replaced on every
    pyspark module that imported the *name*, not just on the one that defines it:
    ``pyspark.context`` does ``from pyspark.java_gateway import launch_gateway``,
    and that binding is the one ``getOrCreate()`` calls.

    The replacement is deliberately never undone. Nothing after the last test may
    start a JVM either, and ``atexit`` hooks -- which is where Core's temporary
    table cleanup lives -- run long after fixture teardown.

    Yields:
        Nothing.

    Raises:
        AssertionError: If a JVM was already running before the first test.
        RuntimeError: If pyspark's ``launch_gateway`` could not be replaced.
    """
    if os.environ.get(FORBID_JVM_ENV_VAR, "") not in ("1", "true", "True"):
        yield
        return

    # A JVM started while test modules were being imported would predate this
    # fixture, so check for one rather than assume.
    if SparkContext._gateway is not None:
        raise AssertionError(
            f"{FORBID_JVM_ENV_VAR} is set, but a JVM was already running before "
            "the first test started -- something booted one during collection."
        )

    for name, module in list(sys.modules.items()):
        if name != "pyspark" and not name.startswith("pyspark."):
            continue
        if getattr(module, "launch_gateway", None) is not None:
            setattr(module, "launch_gateway", _forbidden_launch_gateway)

    # Importing tmlt.core.utils.cleanup registers an atexit hook that calls
    # SparkSession.builder.getOrCreate() to drop Core's temporary database. In
    # this lane no session ever exists, so there is no temporary database to
    # drop -- but the hook would still boot a JVM, and it runs after pytest has
    # returned, where raising only prints "Exception ignored in atexit callback"
    # and leaves the exit code untouched. Since the lane could not fail on it,
    # drop it. Note that this is the guard working around library behaviour, not
    # proving anything about it.
    atexit.unregister(core_cleanup._cleanup_temp)

    if java_gateway.launch_gateway is not _forbidden_launch_gateway:
        raise RuntimeError(
            "Could not install the no-JVM guard: pyspark.java_gateway does not "
            "have a launch_gateway to replace."
        )
    yield
