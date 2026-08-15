"""Tests for orchestrator.repo_paths — dark-factory tooling-root resolution.

Task 3605 (census 2026-08-02 §1.3, codebook entry-cand-20260722-3): the watcher
rotation spawn must inject DARK_FACTORY_ROOT so a cross-project rotation can run
`cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh ...` instead of guessing a
path (or expanding an unset var to `/scripts/...`).

Steps covered by this file:
  step-1: TestResolveDarkFactoryRoot — the __file__-anchored ascent
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.repo_paths import resolve_dark_factory_root


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
