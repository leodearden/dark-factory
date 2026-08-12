"""Unit tests for shared.reify_checkout — layout-independent reify discovery.

The single source of reify-checkout resolution for every reify-dependent test in
dark-factory (task 3978, promoting task 3843's resolver out of
``fused-memory/tests/test_lock_charter_guard.py``).

Step 1 (RED -> step-2 GREEN): ``resolve_reify_root`` + ``REIFY_ROOT_ENV``
Step 3 (RED -> step-4 GREEN): ``reify_skip_reason``
"""

from __future__ import annotations

from pathlib import Path

from shared.reify_checkout import REIFY_ROOT_ENV, reify_skip_reason, resolve_reify_root

# The two real markers in play, one per consumer call site.  The resolver is
# parameterized on these precisely because the two consumers gate on different
# scripts and each needs a checkout that actually carries ITS script.
_VERIFY_MARKER = Path('scripts') / 'verify.sh'
_GUARD_MARKER = Path('scripts') / 'lock-charter-guard.sh'


def _plant(root: Path, tests_dir_relpath: str, marker: str | Path = _VERIFY_MARKER) -> Path:
    """Create a synthetic reify checkout under *root* plus a tests dir.

    Creates ``root/reify/<marker>`` as a real file (content is irrelevant) and
    ``root/tests_dir_relpath`` as a directory, then returns a fake test-file
    path inside that tests dir.  ``resolve_reify_root`` only inspects ancestor
    DIRECTORIES, so the returned test-file path itself need not exist on disk.

    Ported from fused-memory/tests/test_lock_charter_guard.py:234-248 (task
    3843), generalized to write an arbitrary marker relpath.
    """
    marker_file = root / 'reify' / marker
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_text('#!/bin/sh\necho stub\n')
    tests_dir = root / tests_dir_relpath
    tests_dir.mkdir(parents=True, exist_ok=True)
    return tests_dir / 'test_x.py'


def _plant_marker(root: Path, marker: str | Path = _VERIFY_MARKER) -> Path:
    """Create ``root/<marker>`` as a real file and return *root*."""
    marker_file = root / marker
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_text('#!/bin/sh\necho stub\n')
    return root


def test_env_var_name_is_single_sourced():
    """The env-var name lives here, not respelled at each call site."""
    assert REIFY_ROOT_ENV == 'REIFY_ROOT'


class TestResolveReifyRootDiscovery:
    """Ancestor discovery, independent of checkout layout.

    Every case delenvs REIFY_ROOT so discovery — not an operator's ambient
    override — is what is under test.  Precedence is pinned separately below.
    """

    def test_resolves_from_bare_checkout_layout(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        assert resolve_reify_root(_VERIFY_MARKER, start=start) == (src / 'reify').resolve()

    def test_resolves_from_worktree_layout(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        bare_start = _plant(src, 'dark-factory/orchestrator/tests')
        worktree_start = _plant(src, 'dark-factory/.worktrees/3978/orchestrator/tests')
        expected = (src / 'reify').resolve()

        bare_result = resolve_reify_root(_VERIFY_MARKER, start=bare_start)
        worktree_result = resolve_reify_root(_VERIFY_MARKER, start=worktree_start)

        assert worktree_result == expected
        assert worktree_result == bare_result, (
            'worktree and bare-checkout layouts must resolve to the same reify '
            "root: '.worktrees/<id>' adds exactly two path segments relative to "
            'the bare checkout, so no single fixed parents[N] index can satisfy '
            f'both (got worktree={worktree_result!r} vs bare={bare_result!r})'
        )

    def test_resolves_from_arbitrary_extra_nesting(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/.worktrees/3978/orchestrator/tests/sub')
        assert resolve_reify_root(_VERIFY_MARKER, start=start) == (src / 'reify').resolve()

    def test_returns_none_when_no_ancestor_carries_the_marker(self, tmp_path, monkeypatch):
        """STRUCTURAL PIN against a hardcoded default.

        A hardcoded ``/home/leo/src/reify`` default cannot satisfy this on this
        machine, where that path really exists: it would answer with the
        developer's own checkout for a tree that has none in its ancestry.
        """
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        start = tmp_path / 'a' / 'b' / 'c' / 'test_x.py'
        assert resolve_reify_root(_VERIFY_MARKER, start=start) is None

    def test_picks_the_nearest_ancestor(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        outer = tmp_path
        _plant(outer, 'unused-outer-tests-dir')
        inner = outer / 'inner'
        start = _plant(inner, 'dark-factory/orchestrator/tests')
        assert resolve_reify_root(_VERIFY_MARKER, start=start) == (inner / 'reify').resolve()

    def test_ignores_an_ancestor_reify_dir_without_the_marker(self, tmp_path, monkeypatch):
        """A stray marker-less ``reify`` dir must not shadow the real checkout."""
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        higher = tmp_path
        _plant(higher, 'unused-higher-tests-dir')
        (higher / 'src' / 'reify').mkdir(parents=True)
        start_dir = higher / 'src' / 'dark-factory' / 'orchestrator' / 'tests'
        start_dir.mkdir(parents=True)
        start = start_dir / 'test_x.py'
        assert resolve_reify_root(_VERIFY_MARKER, start=start) == (higher / 'reify').resolve()


class TestResolveReifyRootMarker:
    """The marker relpath is the one generalization over task 3843 — pin it."""

    def test_discovery_is_marker_specific(self, tmp_path, monkeypatch):
        """A checkout carrying only verify.sh does not answer for the guard script.

        Each consumer needs a checkout that actually carries ITS script; a tree
        with only the other one is not a usable checkout for this caller.
        """
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests', marker=_VERIFY_MARKER)

        assert resolve_reify_root(_VERIFY_MARKER, start=start) == (src / 'reify').resolve()
        assert resolve_reify_root(_GUARD_MARKER, start=start) is None, (
            'discovery must key on the calling site marker: a reify checkout carrying '
            'only scripts/verify.sh cannot satisfy a caller gating on '
            'scripts/lock-charter-guard.sh'
        )

    def test_accepts_a_str_marker(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        assert resolve_reify_root('scripts/verify.sh', start=start) == (src / 'reify').resolve()

    def test_accepts_a_path_marker(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        assert resolve_reify_root(Path('scripts/verify.sh'), start=start) == (
            (src / 'reify').resolve()
        )


class TestResolveReifyRootOverride:
    """REIFY_ROOT precedence — task 3843's semantics, carried over verbatim."""

    def test_override_wins_over_discovery(self, tmp_path, monkeypatch):
        src = tmp_path / 'src'
        start = _plant(src, 'dark-factory/orchestrator/tests')
        other = tmp_path / 'other' / 'reify-elsewhere'
        other.mkdir(parents=True)
        monkeypatch.setenv(REIFY_ROOT_ENV, str(other))

        result = resolve_reify_root(_VERIFY_MARKER, start=start)

        assert result == other.resolve()
        assert result != (src / 'reify').resolve(), (
            'REIFY_ROOT must win over a discoverable ancestor, not be shadowed by it'
        )

    def test_override_is_honored_verbatim_when_absent_on_disk(self, tmp_path, monkeypatch):
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        missing = tmp_path / 'does-not-exist' / 'reify-typo'
        monkeypatch.setenv(REIFY_ROOT_ENV, str(missing))

        result = resolve_reify_root(_VERIFY_MARKER, start=start)

        assert result is not None, (
            'an absent REIFY_ROOT must resolve to the named path, not None — '
            'None would route callers to the discovery-miss skip, hiding that '
            'the operator named a path at all'
        )
        assert result == missing.resolve(), (
            'a REIFY_ROOT typo must surface downstream as a skip naming the bad '
            'path, not silently fall back to a discovered checkout that answers '
            'for a different repo than the operator named'
        )
        assert not result.exists()

    def test_override_returns_an_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REIFY_ROOT_ENV, 'relative/reify-path')
        start = tmp_path / 'a' / 'b' / 'test_x.py'

        result = resolve_reify_root(_VERIFY_MARKER, start=start)

        assert isinstance(result, Path)
        assert result.is_absolute(), 'callers append the marker relpath unconditionally'

    def test_empty_env_var_falls_back_to_discovery(self, tmp_path, monkeypatch):
        """``export REIFY_ROOT=`` is a shell accident, not an intent."""
        monkeypatch.setenv(REIFY_ROOT_ENV, '')
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')

        result = resolve_reify_root(_VERIFY_MARKER, start=start)

        assert result == (src / 'reify').resolve()
        assert result != Path.cwd(), 'an empty override must not resolve to the process CWD'

    def test_whitespace_only_env_var_falls_back_to_discovery(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REIFY_ROOT_ENV, '   ')
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')

        result = resolve_reify_root(_VERIFY_MARKER, start=start)

        assert result == (src / 'reify').resolve()
        assert result != Path.cwd(), 'a blank override must not resolve to the process CWD'

    def test_unset_env_var_falls_back_to_discovery(self, tmp_path, monkeypatch):
        """Regression pin: honoring REIFY_ROOT must not break plain discovery."""
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        assert resolve_reify_root(_VERIFY_MARKER, start=start) == (src / 'reify').resolve()

    def test_override_arm_is_marker_independent(self, tmp_path, monkeypatch):
        """One ``export REIFY_ROOT=`` steers EVERY reify-dependent test.

        The task's cross-consumer requirement only holds if the override arm
        ignores the marker — it returns before the walk — so orchestrator (gating
        on scripts/verify.sh) and fused-memory (gating on
        scripts/lock-charter-guard.sh) are steered to the SAME root by a single
        export, regardless of which script each one gates on.
        """
        named = tmp_path / 'operator-named-reify'
        named.mkdir()
        monkeypatch.setenv(REIFY_ROOT_ENV, str(named))
        start_a = tmp_path / 'a' / 'orchestrator' / 'tests' / 'test_x.py'
        start_b = tmp_path / 'b' / 'fused-memory' / 'tests' / 'test_y.py'

        result_a = resolve_reify_root(_VERIFY_MARKER, start=start_a)
        result_b = resolve_reify_root(_GUARD_MARKER, start=start_b)

        assert result_a == named.resolve()
        assert result_a == result_b, (
            'the REIFY_ROOT override arm must be marker-INDEPENDENT so a single '
            'export steers every reify-dependent consumer to the same checkout '
            f'(got {result_a!r} for {_VERIFY_MARKER} vs {result_b!r} for {_GUARD_MARKER})'
        )


class TestReifySkipReason:
    """The pure skip-reason builder — WORK item 3.

    Returns a reason STRING rather than calling ``pytest.skip`` (shared/src may
    not import pytest); each call site turns it into the skip mechanism it
    actually needs.  The two non-None arms are deliberately distinct: a
    discovery MISS ("nobody has reify checked out") must never be confused with
    an operator's REIFY_ROOT naming a path that is not there.
    """

    def test_returns_none_when_the_marker_is_present(self, tmp_path):
        """The gate must RUN — no skip — when the checkout really carries it."""
        root = _plant_marker(tmp_path / 'reify')
        assert reify_skip_reason(_VERIFY_MARKER, root) is None

    def test_discovery_miss_names_the_marker_and_the_override(self):
        reason = reify_skip_reason(_VERIFY_MARKER, None)

        assert isinstance(reason, str) and reason, (
            'a falsy reason would silently DISABLE the skipif at the call sites, '
            'which gate on truthiness / `is not None`'
        )
        assert str(_VERIFY_MARKER) in reason, (
            'the discovery-miss reason must name the marker relpath that was '
            f'searched for, so the reader knows what was missing (got {reason!r})'
        )
        assert REIFY_ROOT_ENV in reason, (
            'the discovery-miss reason must name REIFY_ROOT as the way to '
            f'override it (got {reason!r})'
        )

    def test_discovery_miss_does_not_claim_a_path_was_named(self):
        """A None root must not be stringified into the message.

        The concrete bug this pins shut: formatting the set-but-absent arm's
        message with ``root=None`` yields 'reify checkout at None has no ...',
        which reads as though the operator named a path when nobody did.
        """
        reason = reify_skip_reason(_VERIFY_MARKER, None)
        assert 'None' not in reason, (
            f'discovery-miss reason must not stringify the absent root: {reason!r}'
        )

    def test_set_but_absent_names_the_bad_path_verbatim(self, tmp_path):
        bad_root = tmp_path / 'does-not-exist' / 'reify-typo'

        reason = reify_skip_reason(_VERIFY_MARKER, bad_root)

        assert isinstance(reason, str) and reason
        assert str(bad_root) in reason, (
            'the bad path must appear verbatim so it is self-evident in the '
            f'pytest -rs output which path was wrong (got {reason!r})'
        )

    def test_the_two_arms_are_distinguishable(self, tmp_path):
        """Keeps 'nobody has reify' from being read as 'you typed the path wrong'."""
        bad_root = tmp_path / 'does-not-exist' / 'reify-typo'

        miss_reason = reify_skip_reason(_VERIFY_MARKER, None)
        named_reason = reify_skip_reason(_VERIFY_MARKER, bad_root)

        assert miss_reason != named_reason
        assert str(bad_root) in named_reason
        assert str(bad_root) not in miss_reason

    def test_existing_dir_without_the_marker_is_the_set_but_absent_arm(self, tmp_path):
        """A checkout present but without the script is NOT a discovery miss."""
        root = tmp_path / 'reify'
        root.mkdir()

        reason = reify_skip_reason(_VERIFY_MARKER, root)

        assert isinstance(reason, str) and reason
        assert str(root) in reason
        assert reason != reify_skip_reason(_VERIFY_MARKER, None)

    def test_marker_specific_when_the_root_carries_only_the_other_script(self, tmp_path):
        root = _plant_marker(tmp_path / 'reify', marker=_VERIFY_MARKER)

        assert reify_skip_reason(_VERIFY_MARKER, root) is None
        guard_reason = reify_skip_reason(_GUARD_MARKER, root)
        assert isinstance(guard_reason, str) and guard_reason
        assert str(_GUARD_MARKER) in guard_reason

    def test_accepts_a_str_marker(self, tmp_path):
        root = _plant_marker(tmp_path / 'reify')
        assert reify_skip_reason('scripts/verify.sh', root) is None

        miss = reify_skip_reason('scripts/verify.sh', None)
        assert isinstance(miss, str) and miss

    def test_accepts_a_path_marker(self, tmp_path):
        root = _plant_marker(tmp_path / 'reify')
        assert reify_skip_reason(Path('scripts/verify.sh'), root) is None
