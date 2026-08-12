.. _backends:

Backends: Spark and pandas
==========================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2025

A :class:`~tmlt.analytics.Session` evaluates queries on some engine. Historically
there was only one -- Spark -- and it was named directly throughout the query
compiler. A *backend* is that choice, made in one place: the domains a table lives
in, the operations a query is built from, and the kind of dataframe answers come
back in.

Two backends exist. Spark is the default, and was the only one before this seam
existed, so code that never mentions a backend behaves exactly as it did. The
pandas backend runs the same queries in memory, in the same process, with no JVM
and no cluster -- useful when the data is small enough that Spark's startup
dominates the work, or when a Spark installation is not available at all. It is
experimental, and answers a *subset* of the query surface: see
:ref:`the matrix<backend-feature-matrix>` below.

What is *not* different between them is the privacy guarantee. Both backends build
Tumult Core measurements, accounted by the same
:class:`~tmlt.analytics.Session`; a query answered on either spends the same
budget, and its noise is calibrated to the same sensitivity. What differs is which
queries can be answered at all -- and a query that cannot be is refused, never
answered a different way.

.. note::

   The pandas backend is behind the ``pandas_backend``
   :class:`~tmlt.analytics.FeatureFlag`, and needs a Tumult Core build that ships
   the pandas transformations and measurements it binds. On a Core without them,
   asking for the backend raises ``BackendUnavailable`` rather than failing
   somewhere deeper.

Choosing a backend
------------------

Nobody names a backend. You hand a builder its tables, and the type of the first
private dataframe *is* the choice: a :class:`pyspark.sql.DataFrame` puts the
Session on Spark, and a :class:`pandas.DataFrame` puts it on pandas.

.. testcode::
    :hide:

    # Hidden block for imports to make the examples testable.
    import pandas as pd
    from tmlt.analytics import (
        AddOneRow,
        KeySet,
        PureDPBudget,
        QueryBuilder,
        Session,
    )
    from tmlt.analytics.config import config

    members = pd.DataFrame(
        {
            "name": ["alice", "bob", "carol", "dave"],
            "city": ["Alameda", "Berkeley", "Alameda", "Concord"],
        }
    )

.. testcode::

    with config.features.pandas_backend.enabled():
        session = (
            Session.Builder()
            .with_privacy_budget(PureDPBudget(2))
            .with_private_dataframe(
                "members", members, protected_change=AddOneRow()
            )
            .build()
        )

Three rules follow from the choice being the data:

* **Every table in a Session is on one backend.** A Session's tables share one
  privacy accountant, and an accountant's domain is a domain of one backend's
  tables, so a builder that is handed a pandas frame after a Spark one -- or the
  reverse -- refuses it with a :class:`ValueError` naming both. Convert the table
  first, or build a second Session for it.
* **The feature flag gates the feature, not the table.** It is checked before the
  question of whether the table agrees with the others, so a caller who has not
  enabled the pandas backend is told that rather than being told how to mix
  backends they cannot use yet.
* **The one-table constructor is Spark-only.**
  :meth:`~tmlt.analytics.Session.from_dataframe` takes a Spark DataFrame by
  signature; a Session on any other backend is built through
  :class:`~tmlt.analytics.Session.Builder`.

Everything else about the Session is the same. It describes its tables, spends its
budget, and answers queries through :meth:`~tmlt.analytics.Session.evaluate` --
which returns a dataframe of the backend's own kind, so a pandas Session answers
with a pandas dataframe rather than one you have to collect:

.. testcode::

    counts = session.evaluate(
        QueryBuilder("members")
        .groupby(KeySet.from_dict({"city": ["Alameda", "Berkeley"]}))
        .count(),
        PureDPBudget(1),
    )
    print(type(counts))

.. testoutput::

    <class 'pandas.core.frame.DataFrame'>

.. _backend-feature-matrix:

The feature matrix
------------------

The table below is generated when this page is built, from the same tables the
query compiler's own rejection gate reads. It is therefore not a description of
what the backends supported when someone last wrote it down: it is what they
support now. "Written as" is the call that builds the feature, and "Feature" is
the name a rejection message gives it.

.. backend-feature-matrix::

A query that needs a feature its backend lacks is refused at compile time, before
any of it is built and before any budget is spent, with the message naming the
feature, the call, and the missing operation:

.. testcode::

    try:
        session.evaluate(
            QueryBuilder("members").filter("city = 'Alameda'").count(),
            PureDPBudget(1),
        )
    except NotImplementedError as error:
        print(error)

.. testoutput::

    Filter is not supported by the pandas backend. QueryBuilder.filter() needs the 'Filter' operation, which this backend does not provide. See the backend feature matrix in the documentation.

The gate runs before the query's schema is computed, which matters for more than
tidiness: validating a filter condition compiles it against a real
:class:`~pyspark.sql.SparkSession`, so a query the pandas backend was always going
to refuse would otherwise start a JVM on its way to being refused.

What the pandas backend supports, in one sentence: reading a table, ``select``,
``rename``, ``map``, ``join_private``, ``enforce`` with any of the row and group
constraints, ``count`` and ``count_distinct`` -- ungrouped, or grouped by a
:class:`~tmlt.analytics.KeySet` -- and ``suppress`` on the result. Aggregations
over a column's values -- ``sum``, ``average``, ``variance``, ``stdev``,
``quantile``, ``get_bounds`` -- and everything that needs a row-level Spark
expression -- ``filter``, ``flat_map``, ``flat_map_by_id``, ``join_public``, the
null, NaN and infinity transformations -- are not there yet.

Beyond the query surface
------------------------

Four things a Session can do are not query features, and so are not in the table
above. On the pandas backend:

* **Public tables are Spark-only.** A public table exists to be joined against,
  and there is no pandas ``join_public``, so a public table on a pandas Session
  could never be read. Both
  :meth:`~tmlt.analytics.Session.add_public_dataframe` and the builder's
  ``with_public_dataframe`` refuse rather than accept one that nothing could use.
* **Protected changes:** :class:`~tmlt.analytics.AddOneRow`,
  :class:`~tmlt.analytics.AddMaxRows` and :class:`~tmlt.analytics.AddRowsWithID`
  are supported. :class:`~tmlt.analytics.AddMaxRowsInMaxGroups` is not --
  protecting it needs grouped truncation, which the pandas backend does not have
  yet -- and is refused by the builder, at the table it was given for.
* **Partitioning is not available.**
  :meth:`~tmlt.analytics.Session.partition_and_create` is refused before anything
  is built, so a rejected partition spends no budget.
* **Caching a view does nothing.** ``cache=True`` on
  :meth:`~tmlt.analytics.Session.create_view` is ignored, with a warning. It is
  not an error: caching is a performance request, and this backend's answer to it
  is that its tables are already materialized in memory.

Two :class:`~tmlt.analytics.KeySet` operations also have no pandas
implementation, and a group-by whose keys use one is refused with every offending
operation named:

* a KeySet built from a Spark dataframe with
  :meth:`~tmlt.analytics.KeySet.from_dataframe`. Collecting a distributed frame
  into the driver is unbounded work on data whose size nobody has looked at; if
  it is the right thing to do, it is the caller's decision to make, not a side
  effect of a backend switch. Use :meth:`~tmlt.analytics.KeySet.from_pandas` or
  :meth:`~tmlt.analytics.KeySet.from_tuples`.
* a KeySet narrowed with :meth:`~tmlt.analytics.KeySet.filter`. The condition is
  a Spark SQL expression or a :class:`~pyspark.sql.Column` -- a piece of Spark
  the user wrote rather than one Analytics chose, so there is nothing to
  translate. Build the narrowed KeySet with
  :meth:`~tmlt.analytics.KeySet.from_tuples` instead.

Every other way of building a KeySet materializes on either backend, and
:meth:`~tmlt.analytics.KeySet.to_pandas` asks for the keys in memory --
the same keys, in the same column order, as
:meth:`~tmlt.analytics.KeySet.dataframe` gives on Spark.

Things that look like backend differences and are not
-----------------------------------------------------

The backend parity suite compares the two backends query by query. Four of its
findings are worth stating here, because each is easy to mistake for a gap in the
pandas backend:

* **Private joins work on pandas,** with either
  :class:`~tmlt.analytics.TruncationStrategy` -- ``DropExcess`` and
  ``DropNonUnique`` both. ``join_private`` is in the supported half of the matrix
  above, not an exception to it.
* **Discovering group keys needs an approximate-DP budget on every backend.**
  :meth:`~tmlt.analytics.QueryBuilder.get_groups`, and grouping by a list of
  column names rather than by a KeySet, both find their own keys -- a measurement
  with a failure probability, so the engine refuses them under a pure-DP or zCDP
  budget and asks for an :class:`~tmlt.analytics.ApproxDPBudget`, whatever the
  backend. That is an engine requirement; that they are unsupported on pandas as
  well is what the matrix records.
* **Privacy budgets are exact rationals.** Adding budgets given as floats leaves
  residues -- a total spent in two halves may not compare equal to the total
  asked for -- because the arithmetic is done exactly rather than in floating
  point. This is true on either backend and has nothing to do with which one is
  in use.
* **Both backends answer the same query with the same distribution.** Where a
  query is supported on both, the parity suite draws it many times on each and
  gates the two sets of draws against each other, so "the same answer" means the
  same noise distribution rather than merely the same shape.

Schemas and nullability
-----------------------

Spark carries a schema; pandas carries dtypes. Analytics reads a column's type and
its nullability off whichever of the two it is given, and the answers are the same
except about ``allow_null``, in two places worth knowing about.

**Input tables.** A dtype decides nullability, and it decides it without looking
at any value -- so the same dtypes always give the same schema, and no schema ever
reveals whether the data happens to contain a null. Compared against a Spark
dataframe built by inference from the same pandas frame, pandas is the *more*
precise of the two: ``SparkSession.createDataFrame`` marks every field it infers
``nullable=True``, including a numpy ``int64`` column that cannot hold a null,
where pandas reports ``allow_null=False``. Compared against a Spark dataframe
whose schema was *declared* with a non-nullable string, date or timestamp column,
the difference runs the other way: pandas holds those types in ``object`` and
``datetime64`` columns, which always admit a null, so Analytics reports
``allow_null=True``. There is no non-nullable pandas dtype for them to be held in.

.. testcode::

    session.describe("members")

.. testoutput::

    Column Name    Column Type    Nullable
    -------------  -------------  ----------
    name           VARCHAR        True
    city           VARCHAR        True

**Answers.** An ``INTEGER`` column of a *result* -- a ``count``, say -- reads back
``allow_null=True`` from a Spark answer and ``allow_null=False`` from a pandas
one. This follows from the two representations' defaults: Core's Spark output
schema uses a nullable ``LongType``, while the pandas output column is a numpy
``int64``. It is benign in both cases -- a count is never null -- but a test that
asserts on a result's descriptors has to expect it.

When a query is refused
-----------------------

.. currentmodule:: tmlt.analytics._backends

.. py:exception:: NotSupportedByBackend
    :canonical: tmlt.analytics._backends._base.NotSupportedByBackend

    Raised when the backend in use cannot perform an operation a query, a table
    or a Session method needs.

    It subclasses :class:`NotImplementedError`, so code that already catches that
    keeps working. Every message it carries names the feature, names the backend,
    and ends with a pointer to this page. Two attributes carry the same
    information for a caller that would rather branch than parse: ``op``, the
    name of the operation that was refused, and ``backend``, the name of the
    backend that refused it.

    A rejection is not a failed query: nothing has been built and no privacy
    budget has been spent when it is raised.

.. py:class:: Backend
    :canonical: tmlt.analytics._backends._base.Backend

    A descriptor of one backend: the domains its tables live in, the Core classes
    it builds a pipeline from, and the kind of dataframe it carries. It is an
    internal type, and there is nothing to construct -- the two that exist are the
    module constants ``SPARK`` and ``PANDAS``
    (``from tmlt.analytics._backends import PANDAS, SPARK``, a private module that
    may move), and a Session chooses between them from the tables it is given
    rather than being told.

    It appears in this documentation only as the type of the optional ``backend``
    argument of :meth:`~tmlt.analytics.KeySet.dataframe` and
    :meth:`~tmlt.analytics.KeySet.size`, where the two constants are the values to
    pass, and where :meth:`~tmlt.analytics.KeySet.to_pandas` is the typed way to
    ask for the other backend's frame.
