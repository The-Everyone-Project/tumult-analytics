"""The backend feature matrix, read off the tables the compiler enforces.

NOTE (The-Everyone-Project fork): added with the pandas backend.

Every ``NotSupportedByBackend`` message ends with
``FEATURE_MATRIX_HINT`` -- "See the backend feature matrix in the documentation."
-- so the documentation owes the reader a matrix that is *right*. A matrix written
by hand would be right on the day it was written, and the way it would go wrong is
the quiet way: a feature implemented on pandas but still documented as missing,
or worse, one still documented as present.

So the table is not written down. It is built here, from
``REQUIRED_OPS``/``AUTOMATIC_PARTITION_SELECTION`` and ``unsupported_features()``
in :mod:`tmlt.analytics._query_expr_compiler._backend_support` -- the same tables
the compiler's own rejection gate reads -- and rendered at docs-build time by the
directive in :mod:`backend_matrix`.

This half is kept free of any Sphinx or docutils import so that
``test/unit/test_docs_feature_matrix.py`` can check it in a test environment,
which installs neither.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

BACKEND_NAMES: Tuple[str, ...] = ("Spark", "pandas")
"""The backends the matrix has a column for, in column order."""

AUTOMATIC_PARTITION_SELECTION_NAME = "Automatic partition selection"
"""The one matrix row that is not a query expression type.

Grouping by a list of columns rather than by a KeySet means the keys have to be
discovered, which is a measurement of its own; the compiler names it as a feature
in its own right, and so does the matrix."""


class FeatureRow(NamedTuple):
    """One row of the matrix: one query feature, and each backend's answer.

    Attributes:
        feature: The feature's name, as a rejection message spells it -- the
            ``QueryExpr`` type's name for most features, and a phrase for the one
            that is not a query type.
        api: The call a user writes to reach it.
        supported: Whether each backend, by name, can answer it.
        missing: The operations each backend is missing, by name; empty for a
            backend that supports the feature.
    """

    feature: str
    api: str
    supported: Dict[str, bool]
    missing: Dict[str, Tuple[str, ...]]


def backends() -> Dict[str, Any]:
    """Returns the backend descriptors the matrix describes, by column name.

    Imported inside the function rather than at module scope so that the import
    error from a Core that cannot provide the pandas backend arrives while a
    document is being built, where Sphinx reports which document asked for it.
    """
    from tmlt.analytics._backends import PANDAS, SPARK  # noqa: PLC0415

    return {"Spark": SPARK, "pandas": PANDAS}


def feature_rows() -> List[FeatureRow]:
    """Returns the matrix, one row per query feature, in the compiler's order.

    The order is ``REQUIRED_OPS``' own -- reading a table, then transformations,
    then aggregations -- with the one feature that is not a query expression type
    last.
    """
    # Deferred for the same reason as backends(), and because importing the
    # compiler is what pulls in the query expression types.
    from tmlt.analytics._query_expr_compiler._backend_support import (  # noqa: PLC0415
        AUTOMATIC_PARTITION_SELECTION,
        REQUIRED_OPS,
        unsupported_features,
    )

    descriptors = backends()
    unsupported = {
        name: unsupported_features(backend) for name, backend in descriptors.items()
    }

    features = [
        (expr_type.__name__, feature) for expr_type, feature in REQUIRED_OPS.items()
    ]
    features.append((AUTOMATIC_PARTITION_SELECTION_NAME, AUTOMATIC_PARTITION_SELECTION))

    rows = []
    for name, feature in features:
        supported = {
            backend: name not in unsupported[backend] for backend in descriptors
        }
        missing = {
            backend: tuple(
                op
                for op in feature.ops
                if getattr(descriptors[backend].ops, op) is None
            )
            for backend in descriptors
        }
        rows.append(FeatureRow(name, feature.api, supported, missing))
    return rows


def cell(row: FeatureRow, backend: str) -> str:
    """Returns the table cell for one feature on one backend."""
    if row.supported[backend]:
        return "yes"
    if not row.missing[backend]:
        # Unreachable through unsupported_features, which decides "unsupported"
        # by finding a missing operation. Spelled rather than asserted so that a
        # future rejection reason that is not a missing slot still renders.
        return "no"
    return "no -- needs " + ", ".join(f"``{op}``" for op in row.missing[backend])


def table_lines(rows: Sequence[FeatureRow]) -> List[str]:
    """Returns the reStructuredText of the matrix table."""
    lines = [
        ".. list-table:: Query features, by backend",
        "   :header-rows: 1",
        "   :widths: 25 30 15 30",
        "",
        "   * - Feature",
        "     - Written as",
    ]
    lines += [f"     - {backend}" for backend in BACKEND_NAMES]
    for row in rows:
        lines += [
            f"   * - ``{row.feature}``",
            f"     - ``{row.api}``",
        ]
        lines += [f"     - {cell(row, backend)}" for backend in BACKEND_NAMES]
    lines.append("")
    return lines
