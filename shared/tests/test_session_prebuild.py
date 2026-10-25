"""SessionPrebuild builds a session fixture's value at collection time, exactly once.

Driven entirely through fakes — a session carrying fake items, a stash and a
controllable ``--collect-only`` flag, and a ``build`` that counts its calls — so
no real source tree is read. Assertions are on call counts and object identity,
never on timing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from session_prebuild import SessionPrebuild

_FIXTURE = 'expensive_tree'


class _CountingBuild:
    """A ``build`` that records every call and returns a fresh object (or raises)."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.built: list[object] = []
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        value = object()
        self.built.append(value)
        return value


def _session(*items: object, collectonly: bool = False) -> pytest.Session:
    options = {'collectonly': collectonly}
    config = SimpleNamespace(getoption=lambda name: options[name])
    return cast(
        pytest.Session,
        SimpleNamespace(items=list(items), config=config, stash=pytest.Stash()),
    )


def _item(*fixturenames: str) -> SimpleNamespace:
    return SimpleNamespace(fixturenames=fixturenames)


def test_a_requested_fixture_is_built_once_and_handed_out_by_identity():
    build = _CountingBuild()
    prebuild = SessionPrebuild(_FIXTURE, build)
    session = _session(_item('tmp_path'), _item('monkeypatch', _FIXTURE))

    prebuild.prebuild_if_requested(session)

    assert build.calls == 1
    (built,) = build.built
    assert prebuild.result(session) is built
    assert prebuild.result(session) is built
    assert build.calls == 1


def test_an_unrequested_fixture_is_never_built_and_result_says_why():
    build = _CountingBuild()
    prebuild = SessionPrebuild(_FIXTURE, build)
    session = _session(_item('tmp_path'), SimpleNamespace())

    prebuild.prebuild_if_requested(session)

    assert build.calls == 0
    with pytest.raises(RuntimeError) as excinfo:
        prebuild.result(session)
    message = str(excinfo.value)
    assert _FIXTURE in message
    assert 'getfixturevalue' in message
    assert 'pytest_collection_finish' in message
    assert build.calls == 0


def test_collect_only_builds_nothing():
    build = _CountingBuild()
    prebuild = SessionPrebuild(_FIXTURE, build)
    session = _session(_item(_FIXTURE), collectonly=True)

    prebuild.prebuild_if_requested(session)

    assert build.calls == 0


def test_a_failed_build_is_reraised_to_every_consumer_without_retry():
    err = OSError('unreadable')
    build = _CountingBuild(error=err)
    prebuild = SessionPrebuild(_FIXTURE, build)
    session = _session(_item(_FIXTURE))

    prebuild.prebuild_if_requested(session)

    for _ in range(2):
        with pytest.raises(OSError) as excinfo:
            prebuild.result(session)
        assert excinfo.value is err
    assert build.calls == 1


def test_an_item_without_fixturenames_is_tolerated():
    build = _CountingBuild()
    prebuild = SessionPrebuild(_FIXTURE, build)
    session = _session(SimpleNamespace(), _item(_FIXTURE))

    prebuild.prebuild_if_requested(session)

    assert build.calls == 1
    assert prebuild.result(session) is build.built[0]


def test_outcomes_live_in_the_session_not_the_instance():
    build = _CountingBuild()
    prebuild = SessionPrebuild(_FIXTURE, build)
    built_session = _session(_item(_FIXTURE))
    other_session = _session(_item('tmp_path'))

    prebuild.prebuild_if_requested(built_session)
    prebuild.prebuild_if_requested(other_session)

    assert prebuild.result(built_session) is build.built[0]
    with pytest.raises(RuntimeError, match=_FIXTURE):
        prebuild.result(other_session)
    assert build.calls == 1
