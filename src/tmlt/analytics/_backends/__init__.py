"""Backend descriptors: which engine the compiler builds its pipeline on.

:class:`~tmlt.analytics._backends._base.Backend` describes one backend,
:data:`~tmlt.analytics._backends._spark.SPARK` is the Spark one -- the default
wherever a backend can be chosen, so that code which says nothing about backends
behaves exactly as it did before this seam existed -- and
:data:`~tmlt.analytics._backends._pandas.PANDAS` is the pandas one.

``PANDAS`` is resolved lazily, on first attribute access. It binds Core
artifacts that only the Everyone Project Core build ships, so importing it
eagerly here would make a Spark-only install fail to import Analytics at all;
deferring it means that install pays nothing and only a program that actually
asks for the pandas backend sees
:class:`~tmlt.analytics._backends._base.BackendUnavailable`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import TYPE_CHECKING, Any, Tuple

import pandas as pd
from tmlt.core.domains.base import Domain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
    SparkRowDomain,
)

from tmlt.analytics._backends._base import (
    DATAFRAME_DOMAIN_TYPES,
    FEATURE_MATRIX_HINT,
    AnyDataFrame,
    Backend,
    BackendUnavailable,
    NotSupportedByBackend,
    Op,
    Ops,
    TableDomain,
)
from tmlt.analytics._backends._spark import SPARK
from tmlt.analytics._utils import AnalyticsInternalError

if TYPE_CHECKING:
    from tmlt.analytics._backends._pandas import PANDAS

__all__ = [
    "DATAFRAME_DOMAIN_TYPES",
    "FEATURE_MATRIX_HINT",
    "PANDAS",
    "SPARK",
    "AnyDataFrame",
    "Backend",
    "BackendUnavailable",
    "NotSupportedByBackend",
    "Op",
    "Ops",
    "TableDomain",
    "backend_for_dataframe",
    "backend_for_domain",
]

_SPARK_DOMAIN_TYPES: Tuple[type, ...] = (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
    SparkRowDomain,
)
"""Every domain type the Spark backend claims, table or not."""

# Every domain type the pandas backend claims, table or not.
#
# Recognizing a domain is cheaper than building the backend that owns it: this
# needs only Core's pandas *domains*, where the backend needs its pandas
# transformations too. Keeping the two apart is what lets backend_for_domain
# answer "no backend handles this" for a Spark-family domain without importing
# the pandas backend and risking a BackendUnavailable that has nothing to do
# with the question asked.
#
# PandasDataFrameDomain is deliberately absent: it describes a frame through its
# columns' numpy domains, and is not a domain this backend's tables live in.
try:
    from tmlt.core.domains.pandas_domains import (
        PandasGroupedTableDomain,
        PandasRowDomain,
        PandasTableDomain,
    )

    _PANDAS_DOMAIN_TYPES: Tuple[type, ...] = (
        PandasTableDomain,
        PandasGroupedTableDomain,
        PandasRowDomain,
    )
except ImportError:  # pragma: no cover -- needs a Core without pandas domains.
    _PANDAS_DOMAIN_TYPES = ()


def _pandas_backend() -> Backend:
    """Return the pandas backend, importing it on first use.

    Raises:
        BackendUnavailable: If the installed Core cannot provide it.
    """
    # Deferred deliberately: see this module's docstring. Importing it at the
    # top would make a Spark-only install fail to import Analytics.
    from tmlt.analytics._backends._pandas import (  # noqa: PLC0415
        PANDAS as pandas_backend,
    )

    return pandas_backend


def __getattr__(name: str) -> Any:
    """Resolve ``PANDAS`` on first access.

    See this module's docstring for why it is not imported eagerly.
    """
    if name == "PANDAS":
        backend = _pandas_backend()
        # Cache it, so that later accesses are plain attribute lookups and every
        # caller gets the same object -- backends are compared by identity.
        globals()["PANDAS"] = backend
        return backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def backend_for_dataframe(dataframe: Any) -> Backend:
    """Return the backend whose tables the given dataframe is one of.

    This is where a :class:`~tmlt.analytics.Session` decides which backend it is
    on: the user does not name a backend, they hand over a dataframe, and the
    type of that dataframe is the choice.

    Args:
        dataframe: A dataframe of some backend's table type.

    Raises:
        BackendUnavailable: If the dataframe is a pandas one and the installed
            Core cannot provide the pandas backend.
        AnalyticsInternalError: If no backend claims this dataframe type.
    """
    if isinstance(dataframe, SPARK.dataframe_type):
        return SPARK
    if isinstance(dataframe, pd.DataFrame):
        return _pandas_backend()
    raise AnalyticsInternalError(
        f"No backend handles tables of type {type(dataframe).__name__}."
    )


def backend_for_domain(domain: Domain) -> Backend:
    """Return the backend whose tables the given domain describes.

    Used where the backend has to be recovered from the data rather than passed
    in -- inside a :class:`~tmlt.analytics.constraints.Constraint`, which is a
    user-facing value with no backend of its own, and in
    :class:`~tmlt.analytics.Session`, which is handed an accountant whose domain
    is the only record of which backend built it.

    Args:
        domain: A table domain.

    Raises:
        BackendUnavailable: If the domain is a pandas one and the installed Core
            cannot provide the pandas backend.
        AnalyticsInternalError: If no backend claims this domain type.
    """
    if isinstance(domain, _SPARK_DOMAIN_TYPES):
        return SPARK
    if _PANDAS_DOMAIN_TYPES and isinstance(domain, _PANDAS_DOMAIN_TYPES):
        return _pandas_backend()
    raise AnalyticsInternalError(
        f"No backend handles tables in a {type(domain).__name__}."
    )
