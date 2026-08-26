"""Tests for a Session on the pandas backend.

The Sessions here are built by hand, from an accountant over a dictionary of
pandas table domains, rather than through ``Session.Builder``: the builder's
``build()`` goes through the neighboring-relation visitor, which is still
Spark-only. Everything a Session decides from its accountant -- which backend it
is on, and what it therefore refuses -- is decided from that domain, so building
the accountant directly tests the real thing.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Dict

import pandas as pd
import pytest
import sympy as sp
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.pandas_domains import PandasTableDomain
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountant,
    SequentialComposition,
)
from tmlt.core.measures import PureDP
from tmlt.core.metrics import DictMetric, SymmetricDifference
from typeguard import TypeCheckError

from tmlt.analytics import (
    AnalyticsInternalError,
    ColumnType,
    PureDPBudget,
    QueryBuilder,
    Session,
)
from tmlt.analytics._backends import PANDAS, SPARK, NotSupportedByBackend
from tmlt.analytics._catalog import Catalog
from tmlt.analytics._query_expr_compiler import QueryExprCompiler
from tmlt.analytics._schema import Schema
from tmlt.analytics._table_identifier import NamedTable

_SCHEMA = Schema({"A": "VARCHAR", "B": "INTEGER"})
_DATA = pd.DataFrame({"A": ["a", "b", "a"], "B": [1, 2, 3]})


def _pandas_session(tables: Dict[str, pd.DataFrame], budget: float = 1) -> Session:
    """Build a Session over pandas tables, without going through the builder."""
    domains = {NamedTable(name): PANDAS.dataframe_domain(_SCHEMA) for name in tables}
    accountant = PrivacyAccountant.launch(
        SequentialComposition(
            input_domain=DictDomain(domains),
            input_metric=DictMetric({key: SymmetricDifference() for key in domains}),
            d_in={key: 1 for key in domains},
            privacy_budget=sp.Integer(budget),
            output_measure=PureDP(),
        ),
        {NamedTable(name): frame for name, frame in tables.items()},
    )
    return Session(accountant=accountant, public_sources={})


@pytest.fixture(name="session")
def fixture_session() -> Session:
    """A one-table pandas Session."""
    return _pandas_session({"t": _DATA})


###############################################################################
# Deriving the backend from the accountant.
###############################################################################


def test_backend_comes_from_the_accountant_domain(session: Session):
    """A Session over pandas tables is on the pandas backend."""
    # pylint: disable=protected-access
    assert session._backend is PANDAS


def test_backend_agrees_across_several_tables():
    """Several pandas tables still make one pandas Session."""
    session = _pandas_session({"t1": _DATA, "t2": _DATA})
    # pylint: disable=protected-access
    assert session._backend is PANDAS
    assert sorted(session.private_sources) == ["t1", "t2"]


def test_mixed_backend_domain_is_rejected(spark):
    """An accountant whose tables are of two backends is not a Session.

    This cannot be reached through the builder, which refuses to mix backends,
    but the Session takes an accountant from anywhere and has to say so rather
    than picking one of the two arbitrarily.
    """
    domains = {
        NamedTable("pandas_table"): PANDAS.dataframe_domain(_SCHEMA),
        NamedTable("spark_table"): SPARK.dataframe_domain(_SCHEMA),
    }
    spark_frame = spark.createDataFrame(_DATA)
    accountant = PrivacyAccountant.launch(
        SequentialComposition(
            input_domain=DictDomain(domains),
            input_metric=DictMetric({key: SymmetricDifference() for key in domains}),
            d_in={key: 1 for key in domains},
            privacy_budget=sp.Integer(1),
            output_measure=PureDP(),
        ),
        {
            NamedTable("pandas_table"): _DATA,
            NamedTable("spark_table"): spark_frame,
        },
    )
    with pytest.raises(AnalyticsInternalError, match="must all be on one backend"):
        Session(accountant=accountant, public_sources={})


def test_query_schema_checks_the_compilers_own_backend():
    """A query is validated against the compiler's backend, not the catalog's.

    A :class:`~tmlt.analytics._catalog.Catalog` records no backend, so there is
    nothing there for a compiler to be talked out of its own by. It used to hold
    one that defaulted to Spark, and a pandas compiler handed such a catalog --
    which is every catalog not built by a Session -- validated its queries
    against Spark's feature set, and so refused nothing.
    """
    catalog = Catalog()
    catalog.add_private_table("t", _SCHEMA.column_descs, constraints=[])
    # pylint: disable=protected-access
    query = QueryBuilder("t").filter("A = 'a'")._query_expr
    with pytest.raises(NotSupportedByBackend) as excinfo:
        QueryExprCompiler(PureDP(), backend=PANDAS).query_schema(query, catalog)
    assert excinfo.value.op == "Filter"
    assert excinfo.value.backend == "pandas"


def test_get_schema_routes_through_the_backend(session: Session):
    """Reading a table's schema uses the pandas domain conversion."""
    assert session.get_schema("t") == _SCHEMA.column_descs
    assert session.get_column_types("t") == {
        "A": ColumnType.VARCHAR,
        "B": ColumnType.INTEGER,
    }


###############################################################################
# What a pandas Session refuses.
###############################################################################


def test_partition_and_create_is_rejected(session: Session):
    """Partitioning is refused, and refused before any budget is spent."""
    before = session.remaining_privacy_budget
    with pytest.raises(NotSupportedByBackend) as excinfo:
        session.partition_and_create(
            "t", privacy_budget=PureDPBudget(0.5), column="A", splits={"p0": "a"}
        )
    assert excinfo.value.backend == "pandas"
    assert "No privacy budget has been spent" in str(excinfo.value)
    assert session.remaining_privacy_budget == before


def test_partition_and_create_rejects_before_validating_its_arguments(
    session: Session,
):
    """The rejection comes first, so a bad partition cannot spend budget either."""
    before = session.remaining_privacy_budget
    with pytest.raises(NotSupportedByBackend):
        session.partition_and_create(
            "t",
            privacy_budget=PureDPBudget(0.5),
            column="A",
            splits={"not an identifier": "a"},
        )
    assert session.remaining_privacy_budget == before


def test_add_public_dataframe_is_rejected(session: Session, spark):
    """A pandas Session takes no public tables."""
    with pytest.raises(NotSupportedByBackend) as excinfo:
        session.add_public_dataframe("pub", spark.createDataFrame(_DATA))
    assert excinfo.value.backend == "pandas"
    assert session.public_sources == []


def test_add_public_pandas_dataframe_is_rejected(session: Session):
    """A public table is a Spark table on every backend, and only a Spark one.

    ``add_public_dataframe`` keeps its Spark-only signature -- a public table
    exists to be joined against, and ``join_public`` is Spark's -- so a pandas
    frame is refused by the type check rather than by the backend.
    """
    with pytest.raises(TypeCheckError, match='"dataframe"'):
        session.add_public_dataframe("pub", _DATA)


###############################################################################
# Views.
###############################################################################


def test_create_view_works(session: Session):
    """A view built from operations the pandas backend has is created."""
    session.create_view(QueryBuilder("t").select(["A"]), "v", cache=False)
    assert sorted(session.private_sources) == ["t", "v"]
    assert session.get_column_types("v") == {"A": ColumnType.VARCHAR}


def test_create_view_with_cache_warns_and_still_builds(session: Session):
    """Caching is a no-op here, and says so rather than pretending."""
    with pytest.warns(UserWarning, match="nothing to cache"):
        session.create_view(QueryBuilder("t").select(["A"]), "v", cache=True)
    assert sorted(session.private_sources) == ["t", "v"]


def test_delete_view(session: Session):
    """A view can be deleted again; the unpersist step is a no-op."""
    session.create_view(QueryBuilder("t").select(["A"]), "v", cache=False)
    session.delete_view("v")
    assert session.private_sources == ["t"]


###############################################################################
# Describing queries.
###############################################################################


def test_describe_a_supported_query(session: Session, capsys):
    """Describing a query the backend can run works as it does on Spark."""
    session.describe(QueryBuilder("t").select(["A"]))
    printed = capsys.readouterr().out
    assert "A" in printed
    assert "VARCHAR" in printed


def test_describe_reraises_unsupported_operations(session: Session):
    """An operation this backend lacks is reported, not swallowed.

    ``NotSupportedByBackend`` is a ``NotImplementedError``, and the describe
    path catches that to mean "this query is a measurement, so it has no
    constraints". Without the separate clause, an unsupported query would be
    described as though it were fine.
    """
    with pytest.raises(NotSupportedByBackend) as excinfo:
        session.describe(QueryBuilder("t").filter("A = 'a'"))
    assert excinfo.value.op == "Filter"
    assert excinfo.value.backend == "pandas"
