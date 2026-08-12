"""Backend descriptors: which engine the compiler builds its pipeline on.

:class:`~tmlt.analytics._backends._base.Backend` describes one backend, and
:data:`~tmlt.analytics._backends._spark.SPARK` is the Spark one -- the default
wherever a backend can be chosen, so that code which says nothing about backends
behaves exactly as it did before this seam existed.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from tmlt.core.domains.base import Domain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain

from tmlt.analytics._backends._base import (
    DATAFRAME_DOMAIN_TYPES,
    Backend,
    NotSupportedByBackend,
    Op,
    Ops,
)
from tmlt.analytics._backends._spark import SPARK
from tmlt.analytics._utils import AnalyticsInternalError

__all__ = [
    "DATAFRAME_DOMAIN_TYPES",
    "SPARK",
    "Backend",
    "NotSupportedByBackend",
    "Op",
    "Ops",
    "backend_for_domain",
]


def backend_for_domain(domain: Domain) -> Backend:
    """Return the backend whose tables the given domain describes.

    Used where the backend has to be recovered from the data rather than passed
    in -- inside a :class:`~tmlt.analytics.constraints.Constraint`, which is a
    user-facing value with no backend of its own.

    Args:
        domain: A table domain.

    Raises:
        AnalyticsInternalError: If no backend claims this domain type.
    """
    if isinstance(domain, SparkDataFrameDomain):
        return SPARK
    # TODO(#pandas-backend): map PandasTableDomain to the pandas backend once
    # that backend exists.
    raise AnalyticsInternalError(
        f"No backend handles tables in a {type(domain).__name__}."
    )
