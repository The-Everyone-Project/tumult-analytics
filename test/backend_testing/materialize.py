"""Turning a :class:`~test.backend_testing.data.TableSpec` into a real frame.

One spec, two frames: :func:`spark_frame` builds the Spark one under an explicit
schema, :func:`pandas_frame` builds the pandas one at the dtypes that schema
implies, and :func:`frame_for` picks between them from a backend.

Nullability, and where the two backends cannot be made to agree
===============================================================

The Spark frame is built from
:func:`~tmlt.analytics._schema.analytics_to_spark_schema`, which sets each
field's ``nullable`` from the column's ``allow_null``, so a Spark frame
reproduces a spec's schema exactly.

pandas has no schema, only dtypes, and ``allow_null`` is read back off the
dtype: a column whose dtype *can* hold a null admits nulls, whether or not it
contains one (see
:func:`~tmlt.analytics._schema.pandas_dtypes_to_analytics_columns`). For two of
the five Analytics types that is a faithful round trip, and for three it is not:

.. list-table::
   :header-rows: 1

   * - Analytics type
     - pandas dtype
     - ``allow_null`` reproduced?
   * - ``INTEGER``
     - ``Int64`` when nullable, ``int64`` when not
     - yes
   * - ``DECIMAL``
     - ``Float64`` when nullable, ``float64`` when not
     - yes
   * - ``VARCHAR``
     - ``object``
     - **no** -- always reported nullable
   * - ``DATE``
     - ``object``
     - **no** -- always reported nullable
   * - ``TIMESTAMP``
     - ``datetime64[ns]``
     - **no** -- always reported nullable

An ``object`` column can always hold ``None`` and a ``datetime64`` column can
always hold ``NaT``; there is no pandas dtype for "string, but never missing".
So a *non-nullable* ``VARCHAR``, ``DATE`` or ``TIMESTAMP`` column has no pandas
twin, and the pandas backend will report it as ``allow_null=True``.

This is not papered over. :func:`expected_columns` returns the columns the given
backend's schema inference actually yields for a spec, and
:func:`nullability_divergences` names the columns where the two disagree, so a
test can assert the real behavior of each backend instead of a fiction that
holds on neither. Among the standard tables the affected columns are
``id3.group``, ``id4.group`` and ``rows1.A``.

The converse divergence -- pandas reporting ``allow_null=False`` where Spark
reports ``True`` -- cannot happen here, because the dtype for a nullable column
is chosen to be one that admits nulls. It can happen to a *user's* frame, which
is the "never True on pandas and False on Spark" asymmetry the schema layer
documents: ``spark.createDataFrame(pandas_df)`` marks every column nullable,
while the pandas frame it was built from reports its numpy integer columns as
non-nullable.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession

from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    analytics_to_spark_schema,
)

from test.backend_testing.data import TableSpec
from test.backend_testing.frames import (
    AnyFrame,
    pandas_dtypes_for,
    pandas_frame_from_rows,
)

PANDAS_ALWAYS_NULLABLE_TYPES = frozenset(
    {ColumnType.VARCHAR, ColumnType.DATE, ColumnType.TIMESTAMP}
)
"""Analytics types whose pandas dtypes always admit a null.

A column of one of these types is reported ``allow_null=True`` by the pandas
backend whatever the spec says; see the module docstring."""


class BackendLike(Protocol):
    """Anything that names a backend.

    Every function here reads only ``name``, so it takes the
    :class:`~test.backend_testing.backends.BackendFixture` the ``backend``
    fixture yields, or any other object carrying the same field. Names are
    ``"spark"`` and ``"pandas"``.
    """

    @property
    def name(self) -> str:
        """The backend's name."""
        ...  # pragma: no cover


BackendArg = Union[str, BackendLike]
"""A backend, named directly or carried by an object that has a name."""


def backend_name(backend: BackendArg) -> str:
    """Returns the name of a backend, given either it or its name.

    Args:
        backend: The backend, or its name.

    Returns:
        The name.
    """
    return backend if isinstance(backend, str) else backend.name


def spark_frame(spec: TableSpec, spark: SparkSession) -> SparkDataFrame:
    """Builds the Spark frame a spec describes.

    The frame is built from Python row tuples under the schema
    :func:`~tmlt.analytics._schema.analytics_to_spark_schema` derives from the
    spec, rather than from a pandas frame: going through pandas would hand the
    schema over to Spark's inference, which is what made the existing fixtures'
    columns all-nullable in the first place.

    Args:
        spec: The table to build.
        spark: The Spark session to build it in.

    Returns:
        The Spark frame.
    """
    return spark.createDataFrame(
        [list(row) for row in spec.rows], schema=analytics_to_spark_schema(spec.schema)
    )


def pandas_frame(spec: TableSpec) -> pd.DataFrame:
    """Builds the pandas frame a spec describes.

    Each column is built at the dtype
    :func:`~tmlt.analytics._schema.analytics_to_pandas_dtypes` gives for it,
    which is the nullable extension dtype exactly when the column admits nulls.
    Where that cannot reproduce the spec's nullability, see the module
    docstring and :func:`nullability_divergences`.

    Args:
        spec: The table to build.

    Returns:
        The pandas frame.
    """
    return pandas_frame_from_rows(
        list(spec.columns), list(spec.rows), pandas_dtypes_for(spec.schema)
    )


def frame_for(
    spec: TableSpec, backend: BackendArg, spark: Optional[SparkSession] = None
) -> AnyFrame:
    """Builds the frame a spec describes, for the given backend.

    Args:
        spec: The table to build.
        backend: The backend to build it for, or its name.
        spark: The Spark session, for the Spark backend. May be omitted if
            ``backend`` carries one, as
            :class:`~test.backend_testing.backends.BackendFixture` does.

    Returns:
        A Spark frame or a pandas frame, as the backend calls for.

    Raises:
        ValueError: If the backend is not one this harness knows.
        RuntimeError: If the Spark backend was asked for without a session.
    """
    name = backend_name(backend)
    if name == "pandas":
        return pandas_frame(spec)
    if name == "spark":
        session = spark if spark is not None else getattr(backend, "spark", None)
        if session is None:
            raise RuntimeError(
                "Building a Spark frame needs a Spark session; pass spark=, or a "
                "backend that carries one."
            )
        return spark_frame(spec, session)
    raise ValueError(f"Unknown backend {name!r}.")


def expected_columns(
    spec: TableSpec, backend: BackendArg
) -> Dict[str, ColumnDescriptor]:
    """Returns the Analytics columns a backend reports for a spec's frame.

    For Spark this is the spec's own schema. For pandas it is the spec's schema
    with ``allow_null`` forced to ``True`` on every ``VARCHAR``, ``DATE`` and
    ``TIMESTAMP`` column, because those dtypes always admit a null -- see the
    module docstring.

    Use this, rather than ``spec.schema``, wherever a test asserts what a
    Session says a table's schema is.

    Args:
        spec: The table.
        backend: The backend, or its name.

    Returns:
        One descriptor per column.

    Raises:
        ValueError: If the backend is not one this harness knows.
    """
    name = backend_name(backend)
    if name == "spark":
        return spec.column_descs()
    if name != "pandas":
        raise ValueError(f"Unknown backend {name!r}.")
    columns: Dict[str, ColumnDescriptor] = {}
    for column, descriptor in spec.column_descs().items():
        columns[column] = (
            ColumnDescriptor(
                descriptor.column_type,
                allow_null=True,
                allow_nan=descriptor.allow_nan,
                allow_inf=descriptor.allow_inf,
            )
            if descriptor.column_type in PANDAS_ALWAYS_NULLABLE_TYPES
            else descriptor
        )
    return columns


def nullability_divergences(spec: TableSpec) -> List[Tuple[str, str]]:
    """Returns the columns of a spec the two backends describe differently.

    Args:
        spec: The table.

    Returns:
        One ``(column, reason)`` pair per column where the backends' inferred
        ``allow_null`` disagree, in column order.
    """
    spark_columns = expected_columns(spec, "spark")
    pandas_columns = expected_columns(spec, "pandas")
    return [
        (
            column,
            f"{descriptor.column_type.name} is non-nullable on Spark, but its "
            "pandas dtype always admits a null",
        )
        for column, descriptor in spark_columns.items()
        if descriptor != pandas_columns[column]
    ]


def all_nullability_divergences(
    specs: Sequence[TableSpec],
) -> Dict[str, List[Tuple[str, str]]]:
    """Returns :func:`nullability_divergences` for every spec that has one.

    Args:
        specs: The tables to check.

    Returns:
        The divergences, keyed by table name; tables with none are absent.
    """
    found = {spec.name: nullability_divergences(spec) for spec in specs}
    return {name: divergences for name, divergences in found.items() if divergences}
