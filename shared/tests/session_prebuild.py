"""Build an expensive session fixture's value at collection time, outside every item's timer.

pytest-timeout (``func_only=False``) arms its timer over an item's whole runtest
protocol, so a lazily built session fixture is billed to whichever item happens
to request it first. ``pytest_collection_finish`` runs after collection and
before ``pytest_runtestloop`` starts any item's protocol, so a value built there
is billed to no item at all. Bound in ``shared/tests/conftest.py::first_party_tree``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, NoReturn, TypeVar

import pytest

T = TypeVar('T')


@dataclass(frozen=True)
class _Built(Generic[T]):
    """A build that returned: its value is handed to every consumer."""

    value: T

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True)
class _Failed:
    """A build that raised: the SAME exception is re-raised to every consumer.

    The original traceback is kept so each re-raise shows the build's frames
    once, rather than accumulating one consumer's frames per earlier raise.
    """

    error: Exception
    traceback: TracebackType | None

    def unwrap(self) -> NoReturn:
        raise self.error.with_traceback(self.traceback)


class SessionPrebuild(Generic[T]):
    """One session fixture's value, built at most once per session at collection time.

    The outcome lives in the session's stash, never on the instance, so one
    module-level instance serves any number of sessions.
    """

    def __init__(self, fixture_name: str, build: Callable[[], T]) -> None:
        self.fixture_name = fixture_name
        self._build = build
        self._key: pytest.StashKey[_Built[T] | _Failed] = pytest.StashKey()

    def prebuild_if_requested(self, session: pytest.Session) -> None:
        """Build now if a collected item statically requests the fixture.

        Nothing is built under ``--collect-only``, where no item will run. A
        build that raises is captured for :meth:`result` to re-raise, so every
        consumer sees the setup error the lazy fixture used to raise and the
        build is never retried.
        """
        if session.config.getoption('collectonly'):
            return
        if not any(
            self.fixture_name in getattr(item, 'fixturenames', ())
            for item in session.items
        ):
            return
        try:
            outcome: _Built[T] | _Failed = _Built(self._build())
        except Exception as exc:
            outcome = _Failed(exc, exc.__traceback__)
        session.stash[self._key] = outcome

    def result(self, session: pytest.Session) -> T:
        """The prebuilt value, or the exception its build raised."""
        outcome = session.stash.get(self._key, None)
        if outcome is None:
            raise RuntimeError(
                f'{self.fixture_name!r} was never prebuilt for this session. It is '
                f'built in pytest_collection_finish, and only when a collected item '
                f'statically requests it. Either it was requested only dynamically '
                f'(request.getfixturevalue is invisible at collection: take it as a '
                f'fixture argument instead), or conftest no longer binds '
                f'pytest_collection_finish to prebuild_if_requested.'
            )
        return outcome.unwrap()
