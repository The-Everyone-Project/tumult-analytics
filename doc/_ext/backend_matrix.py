"""Sphinx extension that renders the backend feature matrix.

NOTE (The-Everyone-Project fork): added with the pandas backend.

The matrix itself -- which query feature each backend can answer, and what it is
missing when it cannot -- is computed in :mod:`backend_matrix_data` from the tables
the query compiler's rejection gate reads. This module is only the rendering half:
it turns those rows into a table wherever a document writes

.. code-block:: rst

    .. backend-feature-matrix::

so that the page every "not supported by this backend" message points at cannot
disagree with the engine. See :mod:`backend_matrix_data` for why the two halves are
separate files.

A generated table cannot go stale, but it can be deleted from the page it was
generated into; ``test/unit/test_docs_feature_matrix.py`` guards against that.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, List

from backend_matrix_data import feature_rows, table_lines
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx

DIRECTIVE_NAME = "backend-feature-matrix"
"""The directive this extension adds."""


class BackendFeatureMatrix(SphinxDirective):
    """Renders the backend feature matrix as a table."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> List[nodes.Node]:
        """Returns the table's nodes.

        Returns:
            The parsed table.
        """
        rows = feature_rows()

        # Rebuild the page when what the table is generated from changes: its
        # source is Python, which Sphinx would not otherwise know this document
        # depends on.
        from tmlt.analytics._query_expr_compiler import (  # noqa: PLC0415
            _backend_support,
        )

        sources = [
            inspect.getsourcefile(_backend_support),
            inspect.getsourcefile(feature_rows),
            __file__,
        ]
        for source in sources:
            if source is not None:
                self.env.note_dependency(source)

        content = StringList(table_lines(rows), source=f"<{DIRECTIVE_NAME}>")
        container = nodes.container()
        container.document = self.state.document
        self.state.nested_parse(content, self.content_offset, container)
        return container.children


def setup(app: "Sphinx") -> Dict[str, Any]:
    """Registers the directive.

    Args:
        app: The Sphinx application.

    Returns:
        The extension's metadata.
    """
    app.add_directive(DIRECTIVE_NAME, BackendFeatureMatrix)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
