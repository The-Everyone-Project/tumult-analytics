"""The value the ``backend`` fixture yields: one backend, ready to test.

A :class:`BackendFixture` is the handle a parity test works through. It names
the backend, carries the Spark session when there is one, and knows the three
things a test needs to do with a backend that are not the test's own subject:
get the :class:`~tmlt.analytics._backends.Backend` descriptor, turn a
:class:`~test.backend_testing.data.TableSpec` into a frame of the right type,
and build a :class:`~tmlt.analytics.Session` on it.

It deliberately does *not* know anything about queries, results or comparisons.
A suite that wants a richer object -- a dispatch table of the operations it is
testing -- should override the ``backend`` fixture in its own ``conftest.py``
and build that object from the one this yields, so that the parametrization,
the Spark laziness and the feature flag keep living in one place.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd
from pyspark.sql import SparkSession

from tmlt.analytics import AddOneRow, PrivacyBudget, Session
from tmlt.analytics._schema import ColumnDescriptor
from tmlt.analytics.config import config
from tmlt.analytics.protected_change import ProtectedChange

from test.backend_testing.data import TableSpec
from test.backend_testing.frames import AnyFrame, to_pandas
from test.backend_testing.materialize import expected_columns, frame_for

if TYPE_CHECKING:
    from tmlt.analytics._backends import Backend

BACKEND_NAMES: Tuple[str, ...] = ("spark", "pandas")
"""The backends the ``backend`` fixture is parametrized over, in fixture order.

Read this rather than spelling the names out, so that adding a backend is one
edit."""

Tables = Union[Mapping[str, TableSpec], Sequence[TableSpec]]
"""Tables to put in a Session: named, or keyed by their specs' own names."""


def _as_mapping(tables: Tables) -> Dict[str, TableSpec]:
    """Returns tables keyed by name, from either accepted form.

    Args:
        tables: The tables, as a mapping or as a sequence of specs.

    Returns:
        The tables, keyed by the name each will have in the Session.
    """
    if isinstance(tables, Mapping):
        return dict(tables)
    return {spec.name: spec for spec in tables}


@dataclass(frozen=True)
class BackendFixture:
    """One backend under test, as yielded by the ``backend`` fixture.

    Attributes:
        name: The backend's name, one of :data:`BACKEND_NAMES`. Always
            lowercase, which the :class:`~tmlt.analytics._backends.Backend`
            descriptor's own ``name`` is not -- it spells itself ``"Spark"``,
            and that string reaches users in error messages, so the harness
            lowercases rather than asking for it to change.
        spark: The Spark session, for the Spark backend; ``None`` for pandas.
            It is carried rather than requested as a fixture parameter so that
            the pandas run of a test never starts a JVM.
    """

    name: str
    spark: Optional[SparkSession] = None

    @property
    def descriptor(self) -> "Backend":
        """The :class:`~tmlt.analytics._backends.Backend` this fixture names.

        Imported on use rather than at module import, because the pandas
        descriptor binds Core artifacts that only some Core builds ship.

        Raises:
            ValueError: If this fixture names a backend that does not exist.
        """
        # Deferred: importing the pandas descriptor is what raises
        # BackendUnavailable on a Core that cannot provide it.
        from tmlt.analytics._backends import PANDAS, SPARK  # noqa: PLC0415

        if self.name == "spark":
            return SPARK
        if self.name == "pandas":
            return PANDAS
        raise ValueError(f"Unknown backend {self.name!r}.")

    @property
    def is_pandas(self) -> bool:
        """Whether this is the pandas backend."""
        return self.name == "pandas"

    def require_spark(self) -> SparkSession:
        """Returns this backend's Spark session, or raises if it has none.

        Returns:
            The Spark session.

        Raises:
            RuntimeError: If this backend carries no Spark session.
        """
        if self.spark is None:
            raise RuntimeError(
                f"The {self.name} backend carries no Spark session; only the "
                "spark backend does."
            )
        return self.spark

    def materialize(self, spec: TableSpec) -> AnyFrame:
        """Builds the frame a spec describes, in this backend's frame type.

        Args:
            spec: The table to build.

        Returns:
            A Spark frame or a pandas frame, as this backend calls for.
        """
        return frame_for(spec, self)

    def expected_columns(self, spec: TableSpec) -> Dict[str, ColumnDescriptor]:
        """Returns the Analytics columns this backend reports for a spec.

        This is the spec's schema for Spark, and the spec's schema with the
        documented nullability widening for pandas; see
        :func:`~test.backend_testing.materialize.expected_columns`.

        Args:
            spec: The table.

        Returns:
            One descriptor per column.
        """
        return expected_columns(spec, self)

    def feature_flag(self) -> ContextManager[Any]:
        """A context manager inside which this backend may be used.

        The pandas backend is behind the ``pandas_backend`` feature flag, so a
        builder only accepts a pandas table inside this. The ``backend`` fixture
        already holds it open for the whole of a test that takes the fixture;
        this is for code that builds a Session without one.

        Returns:
            The feature flag's context manager for pandas, and a do-nothing one
            for every other backend.
        """
        if self.is_pandas:
            return config.features.pandas_backend.enabled()
        return nullcontext()

    def session_builder(
        self,
        budget: Optional[PrivacyBudget] = None,
        id_spaces: Sequence[str] = (),
    ) -> Session.Builder:
        """Returns a Session builder set up for this backend.

        The builder infers its backend from the first private table it is given,
        so "set up for this backend" means: the budget and ID spaces are
        already on it, and the pandas feature flag is expected to be enabled --
        which the ``backend`` fixture does for the whole test. Add tables with
        :meth:`materialize`, or use :meth:`build_session`, which does both.

        Args:
            budget: The Session's total privacy budget, if it should be set now.
            id_spaces: ID spaces to declare, for tables protected with
                :class:`~tmlt.analytics.AddRowsWithID`.

        Returns:
            The builder.
        """
        builder = Session.Builder()
        if budget is not None:
            builder = builder.with_privacy_budget(budget)
        for id_space in id_spaces:
            builder = builder.with_id_space(id_space)
        return builder

    def build_session(
        self,
        tables: Tables,
        *,
        budget: PrivacyBudget,
        protected_change: Union[
            ProtectedChange, Mapping[str, ProtectedChange], None
        ] = None,
        id_spaces: Sequence[str] = (),
    ) -> Session:
        """Builds a Session over the given specs, on this backend.

        Args:
            tables: The tables, as ``{name: spec}`` or as specs keyed by their
                own names.
            budget: The Session's total privacy budget.
            protected_change: The protected change for every table, or one per
                table name. Defaults to :class:`~tmlt.analytics.AddOneRow`.
            id_spaces: ID spaces to declare.

        Returns:
            The Session.

        Raises:
            KeyError: If a per-table ``protected_change`` mapping is missing a
                table.
        """
        specs = _as_mapping(tables)
        if protected_change is None:
            protected_change = AddOneRow()
        with self.feature_flag():
            builder = self.session_builder(budget, id_spaces)
            for name, spec in specs.items():
                change = (
                    protected_change[name]
                    if isinstance(protected_change, Mapping)
                    else protected_change
                )
                builder = builder.with_private_dataframe(
                    name, self.materialize(spec), protected_change=change
                )
            return builder.build()

    def to_pandas(self, frame: AnyFrame, **kwargs: Any) -> pd.DataFrame:
        """Brings a result frame back to pandas without destroying its values.

        A convenience for :func:`~test.backend_testing.frames.to_pandas`, which
        needs no backend: whichever backend produced the frame, the frame says
        which it was.

        Args:
            frame: The frame to convert.
            **kwargs: Passed through to
                :func:`~test.backend_testing.frames.to_pandas`.

        Returns:
            The pandas rendering of the frame.
        """
        return to_pandas(frame, **kwargs)
