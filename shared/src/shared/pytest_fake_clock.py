"""The ``fake_clock`` pytest fixture: a fresh ``shared.testing_fake_clock.FakeClock`` per test.

This is pytest glue, so it imports pytest and is not a pure stdlib leaf.  Each
package's tests/conftest.py binds the fixture by importing it.
"""

from __future__ import annotations

import pytest

from shared.testing_fake_clock import FakeClock

__all__ = ['fake_clock']


@pytest.fixture
def fake_clock() -> FakeClock:
    """A FakeClock at 0.0, private to the requesting test."""
    return FakeClock()
