"""Tests for orchestrator.repo_paths — dark-factory tooling-root resolution.

Task 3605 (census 2026-08-02 §1.3, codebook entry-cand-20260722-3): the watcher
rotation spawn must inject DARK_FACTORY_ROOT so a cross-project rotation can run
`cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh ...` instead of guessing a
path (or expanding an unset var to `/scripts/...`).

Steps covered by this file:
  step-1: TestResolveDarkFactoryRoot — the __file__-anchored ascent
  step-3: TestDarkFactoryRootEnvOverride — DARK_FACTORY_ROOT precedence + validation
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from orchestrator.repo_paths import DARK_FACTORY_ROOT_ENV, resolve_dark_factory_root

_LOGGER_NAME = 'orchestrator.repo_paths'


def _make_fake_checkout(base: Path) -> Path:
    """A directory that validates as a DF checkout (carries the rearm marker)."""
    root = base.resolve()
    script = root / 'scripts' / 'watcher-rearm.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.touch()
    return root


def _repo_paths_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """WARNING messages emitted on the resolver's own logger."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno >= logging.WARNING
    ]


class TestResolveDarkFactoryRoot:
    """resolve_dark_factory_root() contract."""

    def test_walks_up_to_a_checkout_containing_the_rearm_script(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no env override, the ascent yields a checkout carrying the rearm script.

        Asserted on the MARKER FILE, never on a hardcoded /home/leo/src/dark-factory
        literal: pytest may import the package from the primary checkout or from an
        editable worktree install, and both carry scripts/watcher-rearm.sh (it is
        git-tracked), so a marker assertion is stable under both layouts.
        """
        monkeypatch.delenv('DARK_FACTORY_ROOT', raising=False)

        root = resolve_dark_factory_root()

        assert root is not None, 'resolver must find the DF checkout it is running from'
        assert isinstance(root, Path), f'must return a Path, got {type(root).__name__}'
        assert root.is_dir(), f'resolved root {root} is not a directory'
        assert (root / 'scripts' / 'watcher-rearm.sh').is_file(), (
            f'resolved root {root} does not carry scripts/watcher-rearm.sh — '
            'a root that cannot satisfy the rearm guard must not be returned'
        )


class TestDarkFactoryRootEnvOverride:
    """The DARK_FACTORY_ROOT operator override: precedence, validation, fallthrough.

    Task 3605.  An operator export must keep steering the resolver, but an export
    that does not validate must degrade to the auto-resolved root (loudly) rather
    than being propagated — a stale export shipped verbatim into a rotation's
    environment trips scripts/watcher-rearm.sh:150-152 (exit 2), which is verbatim
    census sighting #4 and the exact failure this task exists to remove.
    """

    def test_env_var_name_is_named_once(self) -> None:
        """The env-var spelling is a module constant, not a respelled literal."""
        assert DARK_FACTORY_ROOT_ENV == 'DARK_FACTORY_ROOT'

    def test_valid_override_wins_over_the_walk_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A validating DARK_FACTORY_ROOT beats the __file__ ascent.

        Set via the CONSTANT, not a hardcoded 'DARK_FACTORY_ROOT' string, so a
        rename cannot half-land (constant renamed, resolver still reading the
        old name, test still passing against the old name).
        """
        fake = _make_fake_checkout(tmp_path / 'fake-df')
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(fake))

        assert resolve_dark_factory_root() == fake

    def test_nonexistent_override_falls_through_and_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A DARK_FACTORY_ROOT naming a path that is not there is rejected, not propagated."""
        missing = tmp_path / 'nope'
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(missing))

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            root = resolve_dark_factory_root()

        assert root != missing, 'a non-existent override must never be propagated'
        assert root is not None, 'must fall through to the walk-up, not disable injection'
        assert (root / 'scripts' / 'watcher-rearm.sh').is_file()
        warnings = _repo_paths_warnings(caplog)
        assert warnings, 'a rejected override must be visible in the orchestrator log'
        assert any(str(missing) in m for m in warnings), (
            f'the warning must name the discarded value {missing}; got {warnings!r}'
        )

    def test_empty_override_falls_through_silently(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """DARK_FACTORY_ROOT='' is the normal orchestrator-process state, not an anomaly."""
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, '')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            root = resolve_dark_factory_root()

        assert root is not None
        assert (root / 'scripts' / 'watcher-rearm.sh').is_file()
        assert _repo_paths_warnings(caplog) == [], (
            'an unset/empty var must not warn — it is the normal state, and warning '
            'on it would train operators to ignore the real rejection warnings'
        )

    def test_whitespace_only_override_is_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`export DARK_FACTORY_ROOT= ` counts as unset (convention from reify_checkout)."""
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, '   ')

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            root = resolve_dark_factory_root()

        assert root is not None
        assert (root / 'scripts' / 'watcher-rearm.sh').is_file()
        assert _repo_paths_warnings(caplog) == []

    def test_real_dir_without_the_marker_falls_through_and_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A real directory that is not a DF checkout is rejected, not propagated."""
        not_df = tmp_path / 'some-other-repo'
        not_df.mkdir()
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(not_df))

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            root = resolve_dark_factory_root()

        assert root != not_df, 'a root without watcher-rearm.sh cannot serve a rotation'
        assert root is not None
        assert (root / 'scripts' / 'watcher-rearm.sh').is_file()
        warnings = _repo_paths_warnings(caplog)
        assert any(str(not_df) in m for m in warnings), (
            f'the warning must name the discarded value {not_df}; got {warnings!r}'
        )
