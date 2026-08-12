"""The ``backend`` fixture.

Kept in its own module so that a ``conftest.py`` anywhere in the suite can make
it available with one import::

    from test.backend_testing.fixtures import backend  # noqa: F401

``test/unit/conftest.py`` does that. A directory whose tests should also run on
both backends adds the same line to its own conftest; nothing else in the suite
is affected, because a fixture that no test requests is never set up.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from typing import Iterator

import pytest

from tmlt.analytics.config import config

from test.backend_testing.backends import BACKEND_NAMES, BackendFixture


@pytest.fixture(params=list(BACKEND_NAMES))
def backend(request: pytest.FixtureRequest) -> Iterator[BackendFixture]:
    """Runs the test once per backend.

    Two properties of this fixture are load-bearing and must survive any change
    to it:

    * **The Spark session is resolved lazily.** It is fetched with
      ``getfixturevalue`` inside the ``spark`` parameter's branch rather than
      declared as a parameter of this fixture, so the pandas run of a test never
      starts a JVM, and neither does collection. A test that needs Spark for
      something other than the backend under test should request the ``spark``
      fixture itself.
    * **The pandas feature flag is held open for the pandas parameter**, because
      that is what a builder checks before accepting a pandas table, and it has
      to still be enabled when the test body adds one. It is restored on the way
      out, and the Spark parameter runs with the flag untouched.

    This fixture is deliberately independent of the suite's other fixtures: it
    does not use, replace or rewire the ``session`` fixtures, which keep
    building exactly the Spark Sessions they always did.

    Args:
        request: The pytest request, carrying the backend name.

    Yields:
        The backend to test.
    """
    if request.param == "spark":
        yield BackendFixture(name="spark", spark=request.getfixturevalue("spark"))
        return
    with config.features.pandas_backend.enabled():
        yield BackendFixture(name=request.param)
