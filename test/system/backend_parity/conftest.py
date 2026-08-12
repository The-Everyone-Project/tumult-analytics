"""Fixtures for the backend parity suite.

Two things, both of which exist so that no test module has to know how a backend
is set up: the ``backend`` fixture from :mod:`test.backend_testing`, imported
rather than redefined, and the shared Session cache it is used with.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Iterator

import pytest

from test.backend_testing.fixtures import backend
from test.system.backend_parity.tables import ParitySessions

__all__ = ["backend", "fixture_sessions"]


@pytest.fixture(name="sessions", scope="module")
def fixture_sessions() -> Iterator[ParitySessions]:
    """The module's cache of infinite-budget Sessions.

    Module-scoped so that the Spark Sessions a module needs are built once
    between them rather than once per test; see
    :class:`~test.system.backend_parity.tables.ParitySessions` for why sharing
    them is sound.

    Yields:
        The cache, empty at the start of each module.
    """
    yield ParitySessions()
