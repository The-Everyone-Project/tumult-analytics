"""Defines the base :class:`Constraint` class."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from abc import ABC, abstractmethod
from typing import Tuple

from tmlt.core.transformations.base import Transformation

from tmlt.analytics._backends import SPARK, Backend
from tmlt.analytics._table_reference import TableReference


class Constraint(ABC):
    """Base class representing a known, enforceable fact about a table.

    Constraints provide information about the contents of a table to help
    produce differentially-private results. For example, a constraint might say
    that each ID in a table corresponds to no more than two rows in that table
    (the :class:`~tmlt.analytics.MaxRowsPerID`
    constraint). Constraints are applied via the :meth:`QueryBuilder.enforce()
    <tmlt.analytics.QueryBuilder.enforce>` method.

    This class is a base class for all constraints, and cannot be used directly.
    """

    @abstractmethod
    def _enforce(
        self,
        child_transformation: Transformation,
        child_ref: TableReference,
        *,
        backend: Backend = SPARK,
    ) -> Tuple[Transformation, TableReference]:
        """Append this constraint's truncation to a transformation.

        Args:
            child_transformation: The transformation whose output to truncate.
            child_ref: Which of its tables to truncate.
            backend: The backend to build the truncation with. A constraint is a
                value a user constructs, with no backend in it, so the backend
                comes from whoever enforces it -- the visitor compiling the
                query, or the Session partitioning a table -- each of which
                already knows which one it is on.
        """
        raise NotImplementedError()
