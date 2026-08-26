"""Tests that the documented backend feature matrix is the computed one.

NOTE (The-Everyone-Project fork): added with the pandas backend.

Every :class:`~tmlt.analytics._backends.NotSupportedByBackend` message ends with
:data:`~tmlt.analytics._backends.FEATURE_MATRIX_HINT`, which promises the reader a
feature matrix in the documentation. The matrix is generated at docs-build time by
``doc/_ext/backend_matrix.py`` from the tables the compiler's rejection gate reads,
so it cannot say a backend supports something the engine refuses.

What a generated table *can* do is stop being generated: the directive can be
deleted from the page, the page dropped from the toctree, or the extension
unregistered, and every one of those leaves a docs build that succeeds while the
promise in the error message is broken. That is what this module checks, along with
the rows themselves -- the sync of the table to the code is what the whole
arrangement is for, and it is worth one assertion rather than none.

The docs sources are read as text. Nothing here builds documentation; the docs
build (``nox -s docs``) is what checks that the page renders.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tmlt.analytics._backends import FEATURE_MATRIX_HINT, PANDAS, SPARK
from tmlt.analytics._query_expr_compiler._backend_support import (
    AUTOMATIC_PARTITION_SELECTION,
    REQUIRED_OPS,
    unsupported_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
"""The repository root: test/unit/<this file>."""

DOC_ROOT = REPO_ROOT / "doc"

MATRIX_PAGE = DOC_ROOT / "topic-guides" / "backends.rst"
"""The page FEATURE_MATRIX_HINT promises."""

EXTENSION_DIR = DOC_ROOT / "_ext"

DIRECTIVE = ".. backend-feature-matrix::"
"""How the page asks for the generated table."""


def _load(name: str) -> ModuleType:
    """Imports a module from ``doc/_ext`` by path.

    Loaded by path rather than by name because ``doc/_ext`` is on the path only
    while Sphinx is running, where ``doc/conf.py`` puts it.

    Args:
        name: The module's name, without ``.py``.

    Returns:
        The imported module.
    """
    path = EXTENSION_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="matrix", scope="module")
def fixture_matrix() -> ModuleType:
    """The module that computes the matrix.

    Kept free of Sphinx imports precisely so that it can be imported here; see its
    docstring.
    """
    return _load("backend_matrix_data")


################################################################################
# The rows are the engine's own answer
################################################################################


def test_every_query_feature_has_a_row(matrix: ModuleType) -> None:
    """The matrix covers every feature the compiler's gate knows about."""
    rows = matrix.feature_rows()
    expected = {expr_type.__name__ for expr_type in REQUIRED_OPS} | {
        matrix.AUTOMATIC_PARTITION_SELECTION_NAME
    }
    assert {row.feature for row in rows} == expected
    assert len(rows) == len(expected), "a feature is in the matrix twice"


def test_rows_say_what_each_api_is_written_as(matrix: ModuleType) -> None:
    """Each row names the call that builds its feature, as the gate does."""
    apis = {row.feature: row.api for row in matrix.feature_rows()}
    for expr_type, feature in REQUIRED_OPS.items():
        assert apis[expr_type.__name__] == feature.api
    assert (
        apis[matrix.AUTOMATIC_PARTITION_SELECTION_NAME]
        == AUTOMATIC_PARTITION_SELECTION.api
    )


@pytest.mark.parametrize("backend_name", ["Spark", "pandas"])
def test_unsupported_rows_are_exactly_the_unsupported_features(
    matrix: ModuleType, backend_name: str
) -> None:
    """A row says "no" exactly when the engine would refuse the feature."""
    backend = {"Spark": SPARK, "pandas": PANDAS}[backend_name]
    refused = set(unsupported_features(backend))
    documented = {
        row.feature for row in matrix.feature_rows() if not row.supported[backend_name]
    }
    assert documented == refused


def test_a_refused_feature_names_the_operation_it_needs(matrix: ModuleType) -> None:
    """Every "no" cell names at least one operation the backend really lacks."""
    for row in matrix.feature_rows():
        for backend_name, backend in (("Spark", SPARK), ("pandas", PANDAS)):
            if row.supported[backend_name]:
                assert not row.missing[backend_name]
                continue
            assert row.missing[backend_name], row.feature
            for op in row.missing[backend_name]:
                assert getattr(backend.ops, op) is None


def test_the_table_has_a_line_for_every_row(matrix: ModuleType) -> None:
    """The rendered table lists every feature and every backend column."""
    rows = matrix.feature_rows()
    rendered = "\n".join(matrix.table_lines(rows))
    assert rendered.startswith(".. list-table::")
    for row in rows:
        assert f"``{row.feature}``" in rendered
        assert f"``{row.api}``" in rendered
    for backend_name in matrix.BACKEND_NAMES:
        assert f"     - {backend_name}" in rendered


################################################################################
# The page the error messages promise
################################################################################


def test_the_hint_promises_a_page_and_the_page_exists() -> None:
    """The message every rejection ends with points at a page that is there."""
    assert "feature matrix" in FEATURE_MATRIX_HINT
    assert MATRIX_PAGE.is_file(), f"{MATRIX_PAGE} is what the hint promises"


def test_the_page_generates_the_matrix() -> None:
    """The page renders the table with the directive, rather than writing one."""
    assert DIRECTIVE in MATRIX_PAGE.read_text()


def test_the_page_is_in_the_toctree() -> None:
    """A page that is not in a toctree is a page nobody can navigate to."""
    index = (MATRIX_PAGE.parent / "index.rst").read_text()
    assert MATRIX_PAGE.stem in index.split()


def test_the_extension_is_registered() -> None:
    """The directive the page uses comes from an extension conf.py loads."""
    conf = (DOC_ROOT / "conf.py").read_text()
    assert '"backend_matrix"' in conf
    assert (EXTENSION_DIR / "backend_matrix.py").is_file()
