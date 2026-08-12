"""The :class:`Backend` descriptor: the seam between the compiler and Core.

The compiler used to name Core's Spark classes directly -- ``Rename(...)``,
``SparkDataFrameDomain(...)``, ``analytics_to_spark_columns_descriptor(...)``.
Every one of those names is a choice of *backend*, and there was no single place
where that choice was made, so adding a second backend would have meant editing
every construction site.

A :class:`Backend` is that single place. It is a plain description of one
backend -- its domain types, its dataframe type, the Core classes it builds
tables with, and the handful of Schema conversions the compiler needs -- and the
compiler reaches everything through the descriptor it was handed. The Spark
descriptor lives in :mod:`~tmlt.analytics._backends._spark`; it is the default
everywhere, so a caller that says nothing gets exactly the behavior it had
before.

Not every backend can do everything. An :class:`Ops` slot may be ``None``, which
says "this backend has no such operation"; :meth:`Backend.require` turns that
into a :class:`NotSupportedByBackend` naming the operation and the backend,
rather than a :class:`TypeError` about ``None`` not being callable.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from tmlt.core.domains.base import Domain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain

from tmlt.analytics._schema import ColumnDescriptor, Schema
from tmlt.analytics._utils import AnalyticsInternalError

if TYPE_CHECKING:
    # For typing only, to keep this module importable on its own: KeySet is only
    # named in an annotation here, and the concrete backend modules -- which do
    # have to build one -- import it for themselves.
    from tmlt.analytics.keyset import KeySet

try:
    from tmlt.core.domains.pandas_domains import PandasTableDomain

    _PANDAS_TABLE_DOMAIN_TYPES: Tuple[type, ...] = (PandasTableDomain,)
except ImportError:  # pragma: no cover
    # A Core without the pandas table domains: the Spark backend still works.
    _PANDAS_TABLE_DOMAIN_TYPES = ()

# The annotation is a string so that it is not evaluated at import time, which
# would raise NameError on a Core whose import above failed. It is spelled as a
# tuple of type objects rather than as Tuple[type, ...] because mypy only
# narrows `isinstance(x, TUPLE)` when it can see what the tuple's items are
# types *of*; with Tuple[type, ...] the guards below would stop narrowing.
DATAFRAME_DOMAIN_TYPES: "Tuple[Type[Union[SparkDataFrameDomain, PandasTableDomain]], ...]" = (
    (SparkDataFrameDomain,) + _PANDAS_TABLE_DOMAIN_TYPES
)
"""Every domain type that describes a table, for use in isinstance guards.

Use this where a guard asks "is this a table domain at all", and the narrow
domain type where the code that follows reads something only one backend's
domain has (``SparkDataFrameDomain.spark_schema``, say -- ``.schema`` is common
to both families by design)."""


class NotSupportedByBackend(NotImplementedError):
    """An operation the backend in use cannot perform.

    Raised by :meth:`Backend.require` when the requested :class:`Ops` slot is
    ``None``. It subclasses :class:`NotImplementedError` so that callers who
    already catch that keep working.
    """

    def __init__(
        self,
        message: str,
        *,
        op: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        """Constructor.

        Args:
            message: The human-readable message.
            op: The name of the unsupported operation, if known.
            backend: The name of the backend that does not support it, if known.
        """
        super().__init__(message)
        self.op = op
        self.backend = backend

    @classmethod
    def for_op(
        cls, op: str, backend: str, hint: Optional[str] = None
    ) -> "NotSupportedByBackend":
        """Build the standard "this backend has no such operation" error.

        Args:
            op: The name of the unsupported operation.
            backend: The name of the backend that does not support it.
            hint: An optional sentence appended to the message, suggesting what
                the caller might do instead.
        """
        message = f"{op} is not supported by the {backend} backend."
        if hint is not None:
            message = f"{message} {hint}"
        return cls(message, op=op, backend=backend)


class BackendUnavailable(ImportError):
    """A backend Analytics knows about, but the installed Core cannot provide.

    A backend descriptor is only as real as the Core artifacts it binds. When
    those are missing -- an older or stock Core against a newer Analytics --
    importing the descriptor's module raises this rather than letting an
    ``ImportError`` for some deeply-nested Core module reach the user, who has
    no way to tell from it which backend is unavailable or what would fix it.

    It subclasses :class:`ImportError` because that is what it stands in for,
    and because ``except ImportError`` around an optional-backend import is the
    natural thing for a caller to write.
    """

    def __init__(
        self,
        message: str,
        *,
        backend: Optional[str] = None,
        required: Optional[str] = None,
    ):
        """Constructor.

        Args:
            message: The human-readable message.
            backend: The name of the unavailable backend, if known.
            required: The Core artifact whose absence makes it unavailable.
        """
        super().__init__(message)
        self.backend = backend
        self.required = required


Op = Callable[..., Any]
"""A Core class or factory the compiler constructs something with.

Spelled as a callable rather than as a ``type`` because the aggregation slots
hold Core's ``create_*_measurement`` factory functions, not classes."""


class Ops(NamedTuple):
    """The Core classes and factories one backend builds its pipeline from.

    One slot per thing the compiler constructs. A slot is ``None`` when the
    backend has no equivalent; read slots through :meth:`Backend.require` rather
    than directly, so that a missing one produces a
    :class:`NotSupportedByBackend` instead of a ``TypeError``.

    The names are Core's own, so that a construction site in the compiler reads
    the same as it did before the seam went in.
    """

    # Table transformations.
    Rename: Optional[Op] = None
    Filter: Optional[Op] = None
    Select: Optional[Op] = None
    Map: Optional[Op] = None
    FlatMap: Optional[Op] = None
    FlatMapByKey: Optional[Op] = None
    GroupingFlatMap: Optional[Op] = None
    PrivateJoin: Optional[Op] = None
    PrivateJoinOnKey: Optional[Op] = None
    PublicJoin: Optional[Op] = None
    # The enum of truncation strategies PrivateJoin's arguments come from. It is
    # a value rather than a class the compiler constructs, but it ships with the
    # join transformations and belongs with them.
    TruncationStrategy: Optional[Op] = None
    DropInfs: Optional[Op] = None
    DropNaNs: Optional[Op] = None
    DropNulls: Optional[Op] = None
    ReplaceInfs: Optional[Op] = None
    ReplaceNaNs: Optional[Op] = None
    ReplaceNulls: Optional[Op] = None
    Persist: Optional[Op] = None
    Unpersist: Optional[Op] = None
    PartitionByKeys: Optional[Op] = None

    # Row-level transformations, the pieces Map and FlatMap are built from.
    RowToRowTransformation: Optional[Op] = None
    RowToRowsTransformation: Optional[Op] = None
    RowsToRowsTransformation: Optional[Op] = None

    # Grouping and metric conversion.
    GroupBy: Optional[Op] = None
    UnwrapIfGroupedBy: Optional[Op] = None
    HammingDistanceToSymmetricDifference: Optional[Op] = None

    # Truncation, for tables with the AddRowsWithID protected change.
    LimitRowsPerGroup: Optional[Op] = None
    LimitKeysPerGroup: Optional[Op] = None
    LimitRowsPerKeyPerGroup: Optional[Op] = None

    # The AddRemoveKeys variants: the same operations, applied to one value of a
    # dictionary of tables that share an ID column.
    RenameValue: Optional[Op] = None
    FilterValue: Optional[Op] = None
    SelectValue: Optional[Op] = None
    MapValue: Optional[Op] = None
    FlatMapValue: Optional[Op] = None
    FlatMapByKeyValue: Optional[Op] = None
    PublicJoinValue: Optional[Op] = None
    DropInfsValue: Optional[Op] = None
    DropNaNsValue: Optional[Op] = None
    DropNullsValue: Optional[Op] = None
    ReplaceInfsValue: Optional[Op] = None
    ReplaceNaNsValue: Optional[Op] = None
    ReplaceNullsValue: Optional[Op] = None
    PersistValue: Optional[Op] = None
    UnpersistValue: Optional[Op] = None
    LimitRowsPerGroupValue: Optional[Op] = None
    LimitKeysPerGroupValue: Optional[Op] = None
    LimitRowsPerKeyPerGroupValue: Optional[Op] = None

    # Aggregation factories. These are functions rather than classes: each picks
    # a noise mechanism and assembles the measurement that goes with it.
    create_count_measurement: Optional[Op] = None
    create_count_distinct_measurement: Optional[Op] = None
    create_sum_measurement: Optional[Op] = None
    create_average_measurement: Optional[Op] = None
    create_variance_measurement: Optional[Op] = None
    create_standard_deviation_measurement: Optional[Op] = None
    create_quantile_measurement: Optional[Op] = None
    create_bounds_measurement: Optional[Op] = None
    create_partition_selection_measurement: Optional[Op] = None

    # TODO(#keyset-backend): the KeySet materialization operations (building a
    # sample keyset for noise info, sampling one, and suppress-below) still name
    # Spark directly; they move behind this table with the rest of the KeySet
    # work rather than being guessed at here.


@dataclass(frozen=True)
class Backend:
    """One backend, described well enough for the compiler to target it.

    Everything the compiler needs to know about where its data lives: the domain
    types that describe a table, the Core classes that transform one, and how to
    turn an Analytics :class:`~tmlt.analytics._schema.Schema` into the domain and
    the columns descriptor this backend speaks.
    """

    name: str
    """The backend's name, as it appears in error messages."""

    dataframe_domain_type: type
    """The domain type describing a whole table."""

    row_domain_type: type
    """The domain type describing a single row of a table."""

    grouped_domain_type: type
    """The domain type describing a grouped table."""

    dataframe_type: type
    """The type this backend's tables are carried in."""

    ops: Ops
    """The Core classes and factories this backend builds its pipeline from."""

    dataframe_domain: Callable[[Schema], Domain]
    """Builds this backend's table domain from an Analytics schema."""

    columns_descriptor: Callable[[Schema], Mapping[str, Any]]
    """Converts an Analytics schema to this backend's columns descriptor."""

    domain_to_analytics_columns: Callable[[Domain], Dict[str, ColumnDescriptor]]
    """Converts one of this backend's table domains back to Analytics columns."""

    coerce_schema_or_fail: Callable[[Any], Any]
    """Coerces a dataframe to a schema Analytics supports, or raises."""

    sample_keyset: Callable[[Domain, Sequence[str]], "KeySet"]
    """Builds a one-row :class:`~tmlt.analytics.KeySet` over the given columns.

    Automatic partition selection has to report the noise its query will add
    before it knows what the groupby keys are, and the privacy analysis of a
    keyset does not depend on its contents -- only on its columns. So the noise
    is computed against a stand-in keyset with one arbitrary row, which this
    builds. The column *types* come from the table's domain, which is where the
    backend enters: only that backend's domain knows how to name them.
    """

    def require(self, op_name: str) -> Op:
        """Return an operation, or say clearly that this backend lacks it.

        Args:
            op_name: The name of an :class:`Ops` field.

        Raises:
            NotSupportedByBackend: If this backend has no such operation.
        """
        if op_name not in Ops._fields:
            raise AnalyticsInternalError(
                f"'{op_name}' is not an operation in the backend ops table."
            )
        op = getattr(self.ops, op_name)
        if op is None:
            raise NotSupportedByBackend.for_op(op_name, self.name)
        return op
