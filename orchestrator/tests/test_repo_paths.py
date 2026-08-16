"""Tests for orchestrator.repo_paths — dark-factory tooling-root resolution.

Task 3605 (census 2026-08-02 §1.3, codebook entry-cand-20260722-3): the watcher
rotation spawn must inject DARK_FACTORY_ROOT so a cross-project rotation can run
`cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh ...` instead of guessing a
path (or expanding an unset var to `/scripts/...`).

Steps covered by this file:
  step-1: TestResolveDarkFactoryRoot — the __file__-anchored ascent
  step-3: TestDarkFactoryRootEnvOverride — DARK_FACTORY_ROOT precedence + validation
  step-5: TestDarkFactoryRootUnresolvable — the terminal None branch degrades loudly
  (review amendment) TestRejectedDarkFactoryRootOverride — telling "unset" apart
      from "inherited and known-bad", which the None-branch caller must not conflate
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.repo_paths import (
    DARK_FACTORY_ROOT_ENV,
    rejected_dark_factory_root_override,
    resolve_dark_factory_root,
)

_LOGGER_NAME = 'orchestrator.repo_paths'

#: The ascent seam.  Patched by every test that is really about ENV-OVERRIDE
#: semantics, so those tests assert an exact path rather than depending on
#: pytest happening to run from an editable install inside a git checkout that
#: carries the marker.  Under a non-editable/site-packages layout — the very
#: layout the None branch exists for — an unpatched ascent legitimately returns
#: None and an unpatched fallthrough test would fail for a reason unrelated to
#: what it claims to test.
_ASCENT = 'orchestrator.repo_paths._ascend_for_rearm_marker'


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

        The single deliberately-INTEGRATION test in this file: it does not patch
        the ascent, so it proves the real __file__-anchored walk finds the real
        checkout under the layout pytest actually runs in.  Its siblings patch
        `_ASCENT`, because their subject is override precedence, not the walk.

        Asserted on the MARKER FILE, never on a hardcoded /home/leo/src/dark-factory
        literal: pytest may import the package from the primary checkout or from an
        editable worktree install, and both carry scripts/watcher-rearm.sh (it is
        git-tracked), so a marker assertion is stable under both layouts.
        """
        monkeypatch.delenv(DARK_FACTORY_ROOT_ENV, raising=False)

        root = resolve_dark_factory_root()

        assert root is not None, 'resolver must find the DF checkout it is running from'
        assert isinstance(root, Path), f'must return a Path, got {type(root).__name__}'
        assert root.is_dir(), f'resolved root {root} is not a directory'
        assert (root / 'scripts' / 'watcher-rearm.sh').is_file(), (
            f'resolved root {root} does not carry scripts/watcher-rearm.sh — '
            'a root that cannot satisfy the rearm guard must not be returned'
        )

    def test_ascent_walks_past_a_git_ancestor_lacking_the_marker(
        self, tmp_path: Path
    ) -> None:
        """The walk is keyed on the MARKER, so an inner `.git` cannot terminate it.

        Regression pin for the reason this module does not reuse
        verify_runner.resolve_local_df_checkout: that helper returns the FIRST
        ancestor containing `.git` and stops.  Under a dark-factory checkout
        nested inside an outer git repo (vendored tree, repo-of-repos, an
        editable install under a parent workspace) the first `.git` hit carries
        no watcher-rearm.sh, and terminating there would resolve to None even
        though a perfectly good tooling root sits directly above it.
        """
        from orchestrator.repo_paths import _ascend_for_rearm_marker

        df_root = _make_fake_checkout(tmp_path / 'df')
        inner = df_root / 'vendor' / 'inner-repo'
        (inner / '.git').mkdir(parents=True)
        nested = inner / 'pkg' / 'mod.py'
        nested.parent.mkdir(parents=True)
        nested.touch()

        assert _ascend_for_rearm_marker(start=nested) == df_root

    def test_ascent_returns_none_when_no_ancestor_carries_the_marker(
        self, tmp_path: Path
    ) -> None:
        """A tree with no marker anywhere above it resolves to None, not to `/`."""
        from orchestrator.repo_paths import _ascend_for_rearm_marker

        # tmp_path has no scripts/watcher-rearm.sh at any level up to `/`.
        assert _ascend_for_rearm_marker(start=tmp_path) is None


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
        ascent = _make_fake_checkout(tmp_path / 'ascent')
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(missing))

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER_NAME),
            patch(_ASCENT, return_value=ascent),
        ):
            root = resolve_dark_factory_root()

        assert root == ascent, (
            'a non-existent override must be discarded in favour of the ascent; '
            f'got {root!r}'
        )
        warnings = _repo_paths_warnings(caplog)
        assert warnings, 'a rejected override must be visible in the orchestrator log'
        assert any(str(missing) in m for m in warnings), (
            f'the warning must name the discarded value {missing}; got {warnings!r}'
        )

    def test_empty_override_falls_through_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """DARK_FACTORY_ROOT='' is the normal orchestrator-process state, not an anomaly."""
        ascent = _make_fake_checkout(tmp_path / 'ascent')
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, '')

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER_NAME),
            patch(_ASCENT, return_value=ascent),
        ):
            root = resolve_dark_factory_root()

        assert root == ascent
        assert _repo_paths_warnings(caplog) == [], (
            'an unset/empty var must not warn — it is the normal state, and warning '
            'on it would train operators to ignore the real rejection warnings'
        )

    def test_whitespace_only_override_is_treated_as_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """`export DARK_FACTORY_ROOT= ` counts as unset (convention from reify_checkout)."""
        ascent = _make_fake_checkout(tmp_path / 'ascent')
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, '   ')

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER_NAME),
            patch(_ASCENT, return_value=ascent),
        ):
            root = resolve_dark_factory_root()

        assert root == ascent
        assert _repo_paths_warnings(caplog) == []

    def test_real_dir_without_the_marker_falls_through_and_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A real directory that is not a DF checkout is rejected, not propagated."""
        not_df = tmp_path / 'some-other-repo'
        not_df.mkdir()
        ascent = _make_fake_checkout(tmp_path / 'ascent')
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(not_df))

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER_NAME),
            patch(_ASCENT, return_value=ascent),
        ):
            root = resolve_dark_factory_root()

        assert root == ascent, (
            'a root without watcher-rearm.sh cannot serve a rotation and must be '
            f'discarded in favour of the ascent; got {root!r}'
        )
        warnings = _repo_paths_warnings(caplog)
        assert any(str(not_df) in m for m in warnings), (
            f'the warning must name the discarded value {not_df}; got {warnings!r}'
        )


def _assert_unresolvable(root: object) -> None:
    """The terminal miss must be `None` — never an empty or root-ish path.

    Anti-regression for the census-sighted expansion: a falsy-but-present value
    would let the caller inject DARK_FACTORY_ROOT='' and turn the rotation's
    `cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh` into `cd  && ...` /
    `/scripts/...` — strictly worse than unset, because it also defeats the
    rearm script's own `[ -z ... ]` guard diagnostic at the receiving end.
    """
    assert root is None, f'unresolvable must be None, got {root!r}'


class TestDarkFactoryRootUnresolvable:
    """The terminal branch: no override, no validating ascent (task 3605)."""

    def test_returns_none_and_warns_when_ascent_finds_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No ancestor carries the marker: degrade to None, loudly."""
        monkeypatch.delenv(DARK_FACTORY_ROOT_ENV, raising=False)

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER_NAME),
            patch(_ASCENT, return_value=None),
        ):
            root = resolve_dark_factory_root()

        _assert_unresolvable(root)
        assert _repo_paths_warnings(caplog), (
            'an unresolvable root must be visible in the orchestrator log — the '
            'caller must be able to tell "unresolvable" from "resolved to nothing"'
        )

    def test_returns_none_when_a_bad_override_has_no_ascent_to_fall_back_to(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A rejected override plus a missing ascent is still None, never the override."""
        not_df = tmp_path / 'some-other-repo'
        not_df.mkdir()
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(not_df))

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER_NAME),
            patch(_ASCENT, return_value=None),
        ):
            root = resolve_dark_factory_root()

        _assert_unresolvable(root)
        warnings = _repo_paths_warnings(caplog)
        assert any(str(not_df) in m for m in warnings), (
            f'the discarded override must still be named in the log; got {warnings!r}'
        )


class TestRejectedDarkFactoryRootOverride:
    """rejected_dark_factory_root_override(): "is an inherited export known-bad?"

    Task 3605 (review amendment).  Omitting DARK_FACTORY_ROOT from a spawned
    agent's env_overrides does NOT unset it in the child — cli_invoke seeds the
    subprocess env from os.environ first — so a caller reporting an unresolvable
    root must be able to tell "genuinely unset" from "inherited and known-bad"
    rather than asserting the former and being wrong.
    """

    def test_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DARK_FACTORY_ROOT_ENV, raising=False)
        assert rejected_dark_factory_root_override() is None

    def test_none_when_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blank counts as unset here exactly as it does in the resolver."""
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, '   ')
        assert rejected_dark_factory_root_override() is None

    def test_none_when_the_override_validates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A GOOD export is not a rejection — it is what the resolver returns."""
        good = _make_fake_checkout(tmp_path / 'fake-df')
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(good))
        assert rejected_dark_factory_root_override() is None

    def test_returns_the_raw_value_when_the_override_is_bad(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A set-but-invalid export is reported VERBATIM so a caller can name it."""
        bad = tmp_path / 'not-a-checkout'
        bad.mkdir()
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(bad))
        assert rejected_dark_factory_root_override() == str(bad)

    def test_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """It must not warn: resolve_dark_factory_root already logged this rejection."""
        bad = tmp_path / 'not-a-checkout'
        bad.mkdir()
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(bad))

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            rejected_dark_factory_root_override()

        assert _repo_paths_warnings(caplog) == [], (
            'double-logging one misconfiguration trains operators to ignore it'
        )
