"""Module containing supported variants of neighboring relations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

import pandas as pd
from typeguard import check_type

from tmlt.analytics._backends._base import AnyDataFrame
from tmlt.analytics._coerce_pandas_schema import (
    _fail_if_dataframe_has_invalid_column_names,
)
from tmlt.analytics._coerce_spark_schema import coerce_spark_schema_or_fail
from tmlt.analytics._schema import (
    ColumnDescriptor,
    ColumnType,
    pandas_dtypes_to_analytics_columns,
    spark_schema_to_analytics_columns,
)

ALLOWED_ID_COLUMN_TYPES: FrozenSet[ColumnType] = frozenset(
    {ColumnType.INTEGER, ColumnType.VARCHAR, ColumnType.DATE}
)
"""The Analytics column types a grouping column or an ID column may have.

Both :class:`AddRemoveRowsAcrossGroups` and :class:`AddRemoveKeys` restrict the
column they protect to this set, and always have: on Spark the check was
``dataType in [LongType(), StringType(), DateType()]``, applied to a coerced
frame, and these are the three Analytics types those Spark types stand for. A
``DECIMAL`` column is excluded because grouping on floating-point values is not
meaningful, and a ``TIMESTAMP`` column because Analytics has never allowed it
here.

Stating the rule in Analytics types rather than in one engine's is what lets the
same check serve both backends. On pandas it admits ``int64`` and ``Int64``
(``INTEGER``), an ``object`` column of :class:`str` and the ``string`` extension
dtype (``VARCHAR``), and an ``object`` column of :class:`datetime.date`
(``DATE``) -- exactly the pandas columns whose Spark counterparts the Spark rule
admits.
"""

_ALLOWED_ID_COLUMN_TYPES_STR = ", ".join(
    sorted(column_type.name for column_type in ALLOWED_ID_COLUMN_TYPES)
)
"""The allowed types, rendered for an error message."""


def _analytics_columns(df: AnyDataFrame) -> Dict[str, ColumnDescriptor]:
    """Returns the Analytics columns of a table, as they are once coerced.

    The relations are engine-neutral, but the tables they validate are not, so
    this is where the two backends' schemas are brought onto common ground: the
    caller gets :class:`~tmlt.analytics._schema.ColumnDescriptor` objects and
    never sees a Spark type or a pandas dtype. Which backend a table belongs to
    is read off the table rather than passed in, since a relation is validated
    against the very data it will protect.

    Calling this also validates the table the way loading it into a
    :class:`~tmlt.analytics.Session` would, and raises the same errors: an
    unsupported Spark type or pandas dtype, an unusable column name, or an
    ``object`` column holding something other than strings or dates.

    Neither branch materializes coerced data. The Spark branch coerces as it
    always did, which on Spark builds a query plan and reads nothing; the pandas
    branch does not, because coercion only ever widens a column within its
    Analytics column type -- ``IntegerType`` to ``LongType``, ``Int8`` to
    ``Int64`` -- so it cannot change the answer, and
    :func:`~tmlt.analytics._coerce_pandas_schema.coerce_pandas_schema_or_fail`
    would copy the whole frame to tell us something we already know. Nothing it
    checks is skipped: its column-name check is called directly below, and its
    dtype and ``object``-column checks are part of
    :func:`~tmlt.analytics._schema.pandas_dtypes_to_analytics_columns`.

    Args:
        df: The table to describe.

    Raises:
        ValueError: If the table is not one Analytics can load.
    """
    if isinstance(df, pd.DataFrame):
        # The private helper is the single source of truth for what makes a
        # pandas column name usable; coerce_pandas_schema_or_fail, the only
        # public caller, applies it before doing the copy this avoids.
        _fail_if_dataframe_has_invalid_column_names(df)
        return pandas_dtypes_to_analytics_columns(df)
    return spark_schema_to_analytics_columns(coerce_spark_schema_or_fail(df).schema)


def _fail_if_backends_are_mixed(dfs: Dict[str, AnyDataFrame]) -> None:
    """Raises an error if the tables are not all in one backend's representation.

    A relation describes one accountant, and an accountant has a single input
    domain, built by a single backend. A mixture would produce a domain
    describing tables that are not the tables the Session holds, so it is
    rejected here rather than allowed to become a confusing failure inside Core.
    """
    pandas_tables = {name for name, df in dfs.items() if isinstance(df, pd.DataFrame)}
    other_tables = set(dfs) - pandas_tables
    if pandas_tables and other_tables:
        raise ValueError(
            "The provided input mixes backends: tables"
            f" {sorted(pandas_tables)} are pandas DataFrames, while tables"
            f" {sorted(other_tables)} are not. Every table in a relation must be"
            " in the same representation."
        )


class NeighboringRelation(ABC):
    """Base class for a NeighboringRelation."""

    @abstractmethod
    def validate_input(self, dfs: Dict[str, AnyDataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception.

        Used only for top-level validation.

        Exception types and common reasons:
           - TypeError: Input dictionary does not map table names to DataFrames
           - ValueError: Input dictionary contains an invalid number of items or
             contains invalid values.
           - KeyError: Relation table doesn't exist in input dictionary
        """

    @abstractmethod
    def _validate(self, dfs: Dict[str, AnyDataFrame]) -> Any:
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.

        Returns a list of table names from the input.
        """

    @abstractmethod
    def accept(self, visitor) -> Any:
        """Returns the result of a visit to core for this relation."""


@dataclass(frozen=True)
class AddRemoveRows(NeighboringRelation):
    """A relation of tables differing by a limited number of rows.

    Two tables are considered neighbors under this relation if
    they differ by at most n rows.
    """

    table: str
    """The name of the table in this relation."""
    n: int = field(default=1)
    """The max number of rows which may be differ for two instances of the table to
     be neighbors.
     """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.table, str)
        check_type(self.n, int)

    def validate_input(self, dfs: Dict[str, AnyDataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception.

        Used only for top-level validation.
        """
        check_type(dfs, Dict[str, AnyDataFrame])
        if len(dfs) > 1:
            raise ValueError(
                f"The provided input contains too many items: {dfs.items()}."
                " The AddRemoveRows relation requires input with one item."
            )
        self._validate(dfs)
        return True

    def _validate(self, dfs: Dict[str, AnyDataFrame]) -> List[str]:
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.
        """
        # validation checks that can be called by other relations. This
        # just verifies that the initialized table is in the dfs input,
        # and that it points to a dataframe object in the Dict.
        #
        # Looking the table up is what rejects a missing one, with a KeyError
        # that callers rely on, so it has to stay ahead of the check below --
        # which is consequently unreachable.
        _analytics_columns(dfs[self.table])
        if self.table not in dfs.keys():
            raise ValueError(
                f"""The provided input doesn't contain the relation table
                Input table names: {dfs.keys()}
                Relation table name: {self.table}"""
            )
        return [self.table]

    def accept(self, visitor: "NeighboringRelationVisitor") -> Any:
        """Visit this NeighboringRelation with a Visitor."""
        return visitor.visit_add_remove_rows(self)


@dataclass(frozen=True)
class AddRemoveRowsAcrossGroups(NeighboringRelation):
    """A relation of tables differing by a limited number of groups.

    Two tables are considered neighbors under this relation if they differ by
     at most n groups, with each group differing by no more than m rows.
    """

    table: str
    """The name of the table in this relation."""
    grouping_column: str
    """The column that must be grouped over for the privacy guarantee to hold."""
    max_groups: int
    """The maximum number of groups which may differ for two instances of the table
     to be neighbors.
    """
    per_group: int
    """The max number of rows in any single group that may differ for two instances of
     the table to be neighbors.
     """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.table, str)
        check_type(self.grouping_column, str)
        check_type(self.max_groups, int)
        check_type(self.per_group, int)

    def validate_input(self, dfs: Dict[str, AnyDataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception.

        Used only for top-level validation.
        """
        # checks to be done in any case: the table is in the relation,
        # and the columns exist and have appropriate types
        # private checks to be done if this is a top-level call:
        # the input dict is of length one, is a DataFrame + public checks.
        check_type(dfs, Dict[str, AnyDataFrame])
        if len(dfs) > 1:
            raise ValueError(
                f"The provided input contains too many items: {dfs.items()}."
                " The AddRemoveRowsAcrossGroups relation requires input with one item."
            )

        self._validate(dfs)
        return True

    def _validate(self, dfs: Dict[str, AnyDataFrame]) -> List[str]:
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.
        """
        # checks needed here is that the table is a DataFrame,
        # the grouping column exists
        # and is appropriately typed (i.e. is a supported type for grouping)
        if self.table not in dfs.keys():
            raise KeyError(
                f"""The provided input doesn't contain the relation table
                Input table names: {dfs.keys()}
                Relation table name: {self.table}"""
            )
        if self.grouping_column not in dfs[self.table].columns:
            raise ValueError(
                f"Grouping column '{self.grouping_column}' does not exist in the input."
                f" Available columns: {', '.join(dfs[self.table].columns)}"
            )

        columns = _analytics_columns(dfs[self.table])
        if columns[self.grouping_column].column_type not in ALLOWED_ID_COLUMN_TYPES:
            raise ValueError(
                f"Grouping column '{self.grouping_column}' is not of a type on which"
                " grouping is supported. Supported types for grouping:"
                f" {_ALLOWED_ID_COLUMN_TYPES_STR}"
            )
        return [self.table]

    def accept(self, visitor: "NeighboringRelationVisitor") -> Any:
        """Visit this NeighboringRelation with a Visitor."""
        return visitor.visit_add_remove_rows_across_groups(self)


@dataclass(frozen=True)
class AddRemoveKeys(NeighboringRelation):
    """A relation of tables differing by a certain number of keys.

    Two tables are considered neighbors under this definition if they
    differ only by the addition/removal of all rows with max_keys distinct values under
    the columns indicated.

    Note that AddRemoveKeys is a neighboring relation that covers *multiple*
    tables.
    """

    id_space: str
    """The identifier space protected in the relation."""
    table_to_key_column: Dict[str, str]
    """A dictionary mapping table names to key columns."""
    max_keys: int = field(default=1)
    """The maximum number of keys which may differ for two instances of the table
    to be neighbors.
    """

    def __post_init__(self) -> None:
        """Checks arguments to constructor."""
        check_type(self.id_space, str)
        check_type(self.table_to_key_column, Dict[str, str])
        check_type(self.max_keys, int)
        if self.id_space == "":
            raise ValueError("id space must be non-empty")
        if len(self.table_to_key_column) == 0:
            raise ValueError("table_to_key_column must contain at least one table")
        if self.max_keys < 1:
            raise ValueError("max_keys must be positive")

    def validate_input(self, dfs: Dict[str, AnyDataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception.

        Used only for top-level validation.
        """
        # This is the other relation that covers several tables, so it is the
        # other one that can be handed a mixture of backends. The check is here
        # rather than in _validate because a Conjunction has already made it
        # over the whole input by the time it calls that.
        check_type(dfs, Dict[str, AnyDataFrame])
        _fail_if_backends_are_mixed(dfs)
        self._validate(dfs)
        return True

    def _validate(self, dfs: Dict[str, AnyDataFrame]) -> List[str]:
        """Private validation checks.

        These are the validation checks to be done in all cases
        (regardless of whether the relation is top level).
        """
        # checks needed here:
        # - input type
        check_type(dfs, Dict[str, AnyDataFrame])
        # - all tables present in table_to_key_column are in the input tables
        difference = set(self.table_to_key_column.keys()).difference(set(dfs.keys()))
        if difference:
            raise ValueError(
                "It appears that the list of input tables doesn't contain some of the"
                " tables used in the relation. Tables that appear only in the relation:"
                f" {difference}"
            )
        key_type: Optional[ColumnType] = None
        for table_name, df in dfs.items():
            # check that each table has the requisite column
            # and that the column is the requisite type
            if table_name in self.table_to_key_column:
                key_column = self.table_to_key_column[table_name]
                if key_column not in df.columns:
                    raise ValueError(
                        f"Key column '{key_column}' does not exist in the input."
                        f" Available columns: {', '.join(df.columns)}"
                    )
                column_type = _analytics_columns(df)[key_column].column_type
                if column_type not in ALLOWED_ID_COLUMN_TYPES:
                    raise ValueError(
                        f"Key column '{key_column}' is not of a type allowed for"
                        f" keys. Supported types are: {_ALLOWED_ID_COLUMN_TYPES_STR}."
                    )
                if key_type is None:
                    key_type = column_type
                elif column_type != key_type:
                    raise ValueError(
                        f"Key column '{key_column}' has type "
                        f"{column_type}, but in another"
                        f" table it has type {key_type}. Key types"
                        " must match across tables"
                    )

        return list(self.table_to_key_column.keys())

    def accept(self, visitor: "NeighboringRelationVisitor") -> Any:
        """Visit this NeighboringRelation with a Visitor."""
        return visitor.visit_add_remove_keys(self)


@dataclass(init=False, frozen=True)
class Conjunction(NeighboringRelation):
    """A conjunction composed of other neighboring relations."""

    children: List[NeighboringRelation]
    """Other neighboring relations to build the Conjunction.
     Args can be provided as a single list or as separate arguments.

     If more than one list is provided, Conjunction will only use the first.
     """

    def __init__(self, *children) -> None:
        """Constructor."""
        # flatten the (potentially) nested list
        # since frozen is set to True, we need to subvert it to flatten.
        if isinstance(children[0], list):
            object.__setattr__(self, "children", self._flatten(children[0]))
        else:
            object.__setattr__(self, "children", self._flatten(list(children)))
        # post_init is not automatically called if a dataclass has
        # init=False in its decorator
        self.__post_init__()

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type(self.children, List[NeighboringRelation])

    def validate_input(self, dfs: Dict[str, AnyDataFrame]) -> bool:
        """Does nothing if input is valid, otherwise raises an informative exception."""
        # checks that the provided input maps tables names to DataFrames
        # checks that every table belongs to the same backend
        # checks that every input table in dfs is covered in the relation
        # checks each table is covered only once in the relation
        # validation checks pass for each of the children
        check_type(dfs, Dict[str, AnyDataFrame])
        _fail_if_backends_are_mixed(dfs)
        covered_tables: List[str] = []
        for child in self.children:
            relation_table_names = child._validate(dfs)
            # the sets have at least one common member, which means
            # the table(s) is/are already being used in the relation
            if set(covered_tables) & set(relation_table_names):
                raise ValueError(
                    f"""It appears a table is used more than once in the relation.
                    Duplicate table(s): {
                        set(covered_tables) & set(relation_table_names)
                    }"""
                )
            covered_tables.extend(relation_table_names)
        # the sets don't match, meaning a table was left out of either the relation
        # or the input dict
        if set(dfs.keys()) ^ set(covered_tables):
            raise ValueError(
                f"""It appears that the list of input tables doesn't match the list of
                tables used in the relation.
                Tables that appear in only one list: {
                    set(dfs.keys()) ^ set(covered_tables)
                }
                """
            )
        return True

    def accept(self, visitor: "NeighboringRelationVisitor") -> Any:
        """Visit this NeighboringRelation with a Visitor."""
        return visitor.visit_conjunction(self)

    def _validate(self, dfs: Dict[str, AnyDataFrame]):
        """Private validation checks.

        These are the validation checks to be done
        in any case, i.e. regardless of if the relation is top-level.
        """
        # not necessary for conjunction, since it's always a top-level call
        return

    def _flatten(self, children_list):
        """Recursively flatten a Conjunction's list of child NeighboringRelations."""
        flat_list = []
        for element in children_list:
            if isinstance(element, Conjunction):
                flat_list.extend(element._flatten(element.children))
            else:
                flat_list.append(element)
        return flat_list


class NeighboringRelationVisitor(ABC):
    """A base class for implementing visitors for :class:`NeighboringRelation`."""

    @abstractmethod
    def visit_add_remove_rows(self, relation: AddRemoveRows) -> Any:
        """Visit a :class:`AddRemoveRows`."""

    @abstractmethod
    def visit_add_remove_rows_across_groups(
        self, relation: AddRemoveRowsAcrossGroups
    ) -> Any:
        """Visit a :class:`AddRemoveRowsAcrossGroups`."""

    @abstractmethod
    def visit_add_remove_keys(self, relation: AddRemoveKeys) -> Any:
        """Visit a :class:`AddRemoveKeys`."""

    @abstractmethod
    def visit_conjunction(self, relation: Conjunction) -> Any:
        """Visit a :class:`Conjunction`."""
