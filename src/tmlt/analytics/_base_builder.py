"""Building blocks of modular builders used by various Analytics objects."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Set, Union

import pandas as pd
from pyspark.sql import DataFrame
from typeguard import check_type, typechecked

from tmlt.analytics._backends import (
    SPARK,
    AnyDataFrame,
    Backend,
    NotSupportedByBackend,
    backend_for_dataframe,
)
from tmlt.analytics._utils import assert_is_identifier
from tmlt.analytics.config import config
from tmlt.analytics.privacy_budget import PrivacyBudget
from tmlt.analytics.protected_change import (
    AddMaxRowsInMaxGroups,
    AddRowsWithID,
    ProtectedChange,
)


class BaseBuilder(ABC):
    """A base for various builders of privacy-tracking objects."""

    @abstractmethod
    def build(self) -> Any:
        """Constructs the type that this builder builds."""


class PrivacyBudgetMixin:
    """Adds support for setting the privacy budget for a builder."""

    __budget: Optional[PrivacyBudget] = None

    def __init__(self):
        """Constructor.

        @nodoc
        """
        super().__init__()
        self.__budget = None

    @typechecked
    def with_privacy_budget(self, privacy_budget: PrivacyBudget):
        """Set the privacy budget for the object being built."""
        check_type(privacy_budget, PrivacyBudget)
        if self.__budget is not None:
            raise ValueError("This builder already has a privacy budget set")
        self.__budget = privacy_budget
        return self

    @property
    def _privacy_budget(self) -> PrivacyBudget:
        if self.__budget is None:
            raise ValueError("This builder must have a privacy budget set")
        return self.__budget


# Which backend's frame a builder is handed is how it learns which backend to
# build on; see DataFrameMixin.with_private_dataframe.
class PrivateDataFrame(NamedTuple):
    """A private dataframe and its protected change."""

    dataframe: AnyDataFrame
    protected_change: ProtectedChange


class DataFrameMixin:
    """Adds private and public dataframe support to a builder.

    A builder is also where the backend is chosen. Nobody names one: the first
    private dataframe's type is the choice, and every table added afterwards
    must agree with it, because a Session's tables all live in one accountant
    and an accountant's domain is a domain of one backend's tables.
    """

    __private_dataframes: Dict[str, PrivateDataFrame]
    __public_dataframes: Dict[str, DataFrame]
    __id_spaces: Set[str]
    __backend: Optional[Backend]

    def __init__(self):
        """Constructor.

        @nodoc
        """
        super().__init__()
        self.__private_dataframes = {}
        self.__public_dataframes = {}
        self.__id_spaces = set()
        self.__backend = None

    @typechecked
    def with_private_dataframe(
        self,
        source_id: str,
        dataframe: AnyDataFrame,
        protected_change: ProtectedChange,
    ):
        """Adds a DataFrame as a private source.

        The dataframe may be a Spark DataFrame or, with the ``pandas_backend``
        feature flag enabled, a pandas one. The first private dataframe decides
        which backend the Session runs on, and every table added after it must
        be of the same kind.

        Not all column types are supported in private sources; see
        :class:`~tmlt.analytics.ColumnType` for information about which types are
        supported.

        Args:
            source_id: Source id for the private source dataframe.
            dataframe: Private source dataframe to perform queries on,
                corresponding to the ``source_id``.
            protected_change: A
                :class:`~tmlt.analytics.ProtectedChange`
                specifying what changes to the input data should be protected.
        """
        assert_is_identifier(source_id)
        if (
            source_id in self.__private_dataframes
            or source_id in self.__public_dataframes
        ):
            raise ValueError(f"Table '{source_id}' already exists")

        backend = self.__backend_for(source_id, dataframe)
        if backend.ops.PublicJoin is None and self.__public_dataframes:
            raise NotSupportedByBackend.for_op(
                "Public tables",
                backend.name,
                f"Table '{source_id}' is a {type(dataframe).__name__}, but this"
                " builder already has public tables, and this backend has no"
                " join-public, so a public table on it could never be read.",
            )
        # Unlike the public-table refusal above, this one has no op slot to read:
        # the backends that lack it lack it in Core's aggregations rather than in
        # a transformation the compiler would ask for by name.
        if backend is not SPARK and isinstance(protected_change, AddMaxRowsInMaxGroups):
            raise NotSupportedByBackend.for_op(
                "The AddMaxRowsInMaxGroups protected change",
                backend.name,
                f"Table '{source_id}' cannot use it. Protecting it needs"
                " grouped truncation, which this backend does not have yet."
                " AddOneRow, AddMaxRows and AddRowsWithID are supported.",
            )

        dataframe = backend.coerce_schema_or_fail(dataframe)
        self.__private_dataframes[source_id] = PrivateDataFrame(
            dataframe, protected_change
        )
        self.__backend = backend
        return self

    @typechecked
    def with_public_dataframe(self, source_id: str, dataframe: AnyDataFrame):
        """Adds a public dataframe.

        Public tables are Spark-only: a pandas Session has no join-public, so a
        public table on one could never be read.
        """
        assert_is_identifier(source_id)
        if (
            source_id in self.__private_dataframes
            or source_id in self.__public_dataframes
        ):
            raise ValueError(f"Table '{source_id}' already exists")

        # Either the frame or the builder can be the reason this is refused, and
        # the message should name whichever it is.
        backend = backend_for_dataframe(dataframe)
        unsupported: Optional[Backend] = None
        if backend.ops.PublicJoin is None:
            unsupported = backend
        elif self.__backend is not None and self.__backend.ops.PublicJoin is None:
            unsupported = self.__backend
        if unsupported is not None:
            raise NotSupportedByBackend.for_op(
                "Public tables",
                unsupported.name,
                f"Table '{source_id}' cannot be added: this backend has no"
                " join-public, so a public table on it could never be read.",
            )

        dataframe = SPARK.coerce_schema_or_fail(dataframe)
        self.__public_dataframes[source_id] = dataframe
        return self

    def __backend_for(self, source_id: str, dataframe: AnyDataFrame) -> Backend:
        """Return the backend a new private dataframe puts this builder on.

        Raises:
            RuntimeError: If it is a pandas dataframe and the ``pandas_backend``
                feature flag is disabled.
            ValueError: If it disagrees with the backend an earlier private
                dataframe established.
        """
        backend = backend_for_dataframe(dataframe)
        # The flag gates the feature as a whole, so it is checked before the
        # narrower question of whether this table agrees with the others: a
        # caller who has not enabled the pandas backend is told that first,
        # rather than being told how to mix backends they cannot use.
        if backend is not SPARK:
            config.features.pandas_backend.raise_if_disabled()
        if self.__backend is not None and backend is not self.__backend:
            raise ValueError(
                f"Table '{source_id}' is a {type(dataframe).__name__}, which is"
                f" a table of the {backend.name} backend, but this Session is"
                f" being built on the {self.__backend.name} backend -- an"
                " earlier private table established that. All of a Session's"
                " tables must be on one backend, because they share one privacy"
                " accountant. Convert the table before adding it, or start a"
                " separate Session for it."
            )
        return backend

    @typechecked
    def with_id_space(self, id_space: str):
        """Adds an identifier space.

        This defines a space of identifiers that map 1-to-1 to the identifiers
        being protected by a table with the :class:`~.AddRowsWithID` protected
        change. Any table with such a protected change must be a member of some
        identifier space.
        """
        assert_is_identifier(id_space)
        if id_space in self.__id_spaces:
            raise ValueError(f"ID space '{id_space}' already exists")
        self.__id_spaces.add(id_space)
        return self

    def _add_id_space_if_one_private_df(self):
        """If there's only one private dataframe, add its ID space.

        This only has any effect if:
        - there is only one private DataFrame, and
        - that private DataFrame uses the :class:`~.AddRowsWithID` protected change, and
        - this builder does not already have the associated ID space.
        """
        if len(self._private_dataframes) != 1:
            return self
        only_protected_change = list(self._private_dataframes.values())[
            0
        ].protected_change
        if not isinstance(only_protected_change, AddRowsWithID):
            return self
        id_space = only_protected_change.id_space
        if id_space in self._id_spaces:
            return self
        self.with_id_space(id_space)
        return self

    @property
    def _private_dataframes(self) -> Dict[str, PrivateDataFrame]:
        return dict(self.__private_dataframes)

    @property
    def _public_dataframes(self) -> Dict[str, DataFrame]:
        return dict(self.__public_dataframes)

    @property
    def _id_spaces(self) -> Set[str]:
        return self.__id_spaces

    @property
    def _backend(self) -> Optional[Backend]:
        """The backend this builder is on, or None before the first private table."""
        return self.__backend


class ParameterMixin:
    """Adds support for setting parameters to a builder."""

    __parameters: Dict[str, Any]

    def __init__(self):
        """Constructor.

        @nodoc
        """
        super().__init__()
        self.__parameters = {}

    @typechecked
    def with_parameter(self, name: str, value: Any):
        """Set the value of a parameter."""
        check_type(name, str)
        if name in self.__parameters:
            raise ValueError(f"Parameter '{name}' has already been set")
        self.__parameters[name] = value
        return self

    @property
    def _parameters(self) -> Dict[str, Any]:
        return dict(self.__parameters)
