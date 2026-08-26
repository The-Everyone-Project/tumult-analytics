"""Fixtures shared by the unit tests.

The only thing here is the ``backend`` fixture, made available to this directory
by importing it. It is additive: a test that does not request it is unaffected,
and no existing fixture is rewired through it -- the ``session`` fixtures keep
building exactly the Spark Sessions they always did.

See :mod:`test.backend_testing` for what the fixture yields and how a parity
test uses it. To run another directory's tests on both backends, add the same
import to that directory's ``conftest.py``.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from test.backend_testing.fixtures import backend

__all__ = ["backend"]
