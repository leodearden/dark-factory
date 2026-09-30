"""Unit tests for shared.reify_checkout — layout-independent reify discovery.

The single source of reify-checkout resolution for every reify-dependent test in
dark-factory (task 3978, promoting task 3843's resolver out of
``fused-memory/tests/test_lock_charter_guard.py``).  These cases OWN the
resolver's semantics: the two consumers exercise it through thin marker-bound
adapters and pin only their own wiring, so a semantic change is edited here.

Step 1 (RED -> step-2 GREEN): ``resolve_reify_checkout`` + ``REIFY_ROOT_ENV``
Step 3 (RED -> step-4 GREEN): ``reify_skip_reason``

Task 4259 step 3 (RED -> step-4 GREEN): ``checkout_skip_reason`` — the weaker
CHECKOUT-reachability gate the corpus sweeps need, whose two arms were
hand-rolled identically in shared/tests/test_locking.py and
fused-memory/tests/test_lock_charter_guard.py before it existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import shared.reify_checkout
from shared.reify_checkout import (
    REIFY_ROOT_ENV,
    ReifyCheckout,
    checkout_skip_reason,
    reify_skip_reason,
    resolve_reify_checkout,
)

# The two real markers in play, one per consumer call site.  The resolver is
# parameterized on these precisely because the two consumers gate on different
# scripts and each needs a checkout that actually carries ITS script.
_VERIFY_MARKER = Path('scripts') / 'verify.sh'
_GUARD_MARKER = Path('scripts') / 'lock-charter-guard.sh'


def _plant(root: Path, tests_dir_relpath: str, marker: str | Path = _VERIFY_MARKER) -> Path:
    """Create a synthetic reify checkout under *root* plus a tests dir.

    Creates ``root/reify/<marker>`` as a real file (content is irrelevant) and
    ``root/tests_dir_relpath`` as a directory, then returns a fake test-file
    path inside that tests dir.  ``resolve_reify_checkout`` only inspects
    ancestor DIRECTORIES, so the returned test-file path itself need not exist
    on disk.

    Ported from fused-memory/tests/test_lock_charter_guard.py (task 3843),
    generalized to write an arbitrary marker relpath.
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


def _resolved_root(marker: str | Path, start: Path) -> Path | None:
    """The resolved root alone — for cases that are not about provenance."""
    return resolve_reify_checkout(marker, start=start).root


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
        assert _resolved_root(_VERIFY_MARKER, start) == (src / 'reify').resolve()

    def test_resolves_from_worktree_layout(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        bare_start = _plant(src, 'dark-factory/orchestrator/tests')
        worktree_start = _plant(src, 'dark-factory/.worktrees/3978/orchestrator/tests')
        expected = (src / 'reify').resolve()

        bare_result = _resolved_root(_VERIFY_MARKER, bare_start)
        worktree_result = _resolved_root(_VERIFY_MARKER, worktree_start)

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
        assert _resolved_root(_VERIFY_MARKER, start) == (src / 'reify').resolve()

    def test_returns_none_when_no_ancestor_carries_the_marker(self, tmp_path, monkeypatch):
        """STRUCTURAL PIN against a hardcoded default.

        A hardcoded ``/home/leo/src/reify`` default cannot satisfy this on this
        machine, where that path really exists: it would answer with the
        developer's own checkout for a tree that has none in its ancestry.
        """
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        start = tmp_path / 'a' / 'b' / 'c' / 'test_x.py'
        assert _resolved_root(_VERIFY_MARKER, start) is None

    def test_picks_the_nearest_ancestor(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        outer = tmp_path
        _plant(outer, 'unused-outer-tests-dir')
        inner = outer / 'inner'
        start = _plant(inner, 'dark-factory/orchestrator/tests')
        assert _resolved_root(_VERIFY_MARKER, start) == (inner / 'reify').resolve()

    def test_ignores_an_ancestor_reify_dir_without_the_marker(self, tmp_path, monkeypatch):
        """A stray marker-less ``reify`` dir must not shadow the real checkout."""
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        higher = tmp_path
        _plant(higher, 'unused-higher-tests-dir')
        (higher / 'src' / 'reify').mkdir(parents=True)
        start_dir = higher / 'src' / 'dark-factory' / 'orchestrator' / 'tests'
        start_dir.mkdir(parents=True)
        start = start_dir / 'test_x.py'
        assert _resolved_root(_VERIFY_MARKER, start) == (higher / 'reify').resolve()


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

        assert _resolved_root(_VERIFY_MARKER, start) == (src / 'reify').resolve()
        assert _resolved_root(_GUARD_MARKER, start) is None, (
            'discovery must key on the calling site marker: a reify checkout carrying '
            'only scripts/verify.sh cannot satisfy a caller gating on '
            'scripts/lock-charter-guard.sh'
        )

    def test_accepts_a_str_marker(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        assert _resolved_root('scripts/verify.sh', start) == (src / 'reify').resolve()

    def test_accepts_a_path_marker(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        assert _resolved_root(Path('scripts/verify.sh'), start) == (src / 'reify').resolve()


class TestResolveReifyRootOverride:
    """REIFY_ROOT precedence — task 3843's semantics, carried over verbatim."""

    def test_override_wins_over_discovery(self, tmp_path, monkeypatch):
        src = tmp_path / 'src'
        start = _plant(src, 'dark-factory/orchestrator/tests')
        other = tmp_path / 'other' / 'reify-elsewhere'
        other.mkdir(parents=True)
        monkeypatch.setenv(REIFY_ROOT_ENV, str(other))

        result = _resolved_root(_VERIFY_MARKER, start)

        assert result == other.resolve()
        assert result != (src / 'reify').resolve(), (
            'REIFY_ROOT must win over a discoverable ancestor, not be shadowed by it'
        )

    def test_override_is_honored_verbatim_when_absent_on_disk(self, tmp_path, monkeypatch):
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        missing = tmp_path / 'does-not-exist' / 'reify-typo'
        monkeypatch.setenv(REIFY_ROOT_ENV, str(missing))

        result = _resolved_root(_VERIFY_MARKER, start)

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

        result = _resolved_root(_VERIFY_MARKER, start)

        assert isinstance(result, Path)
        assert result.is_absolute(), 'callers append the marker relpath unconditionally'

    def test_empty_env_var_falls_back_to_discovery(self, tmp_path, monkeypatch):
        """``export REIFY_ROOT=`` is a shell accident, not an intent."""
        monkeypatch.setenv(REIFY_ROOT_ENV, '')
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')

        result = _resolved_root(_VERIFY_MARKER, start)

        assert result == (src / 'reify').resolve()
        assert result != Path.cwd(), 'an empty override must not resolve to the process CWD'

    def test_whitespace_only_env_var_falls_back_to_discovery(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REIFY_ROOT_ENV, '   ')
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')

        result = _resolved_root(_VERIFY_MARKER, start)

        assert result == (src / 'reify').resolve()
        assert result != Path.cwd(), 'a blank override must not resolve to the process CWD'

    def test_unset_env_var_falls_back_to_discovery(self, tmp_path, monkeypatch):
        """Regression pin: honoring REIFY_ROOT must not break plain discovery."""
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        src = tmp_path
        start = _plant(src, 'dark-factory/orchestrator/tests')
        assert _resolved_root(_VERIFY_MARKER, start) == (src / 'reify').resolve()

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

        result_a = _resolved_root(_VERIFY_MARKER, start_a)
        result_b = _resolved_root(_GUARD_MARKER, start_b)

        assert result_a == named.resolve()
        assert result_a == result_b, (
            'the REIFY_ROOT override arm must be marker-INDEPENDENT so a single '
            'export steers every reify-dependent consumer to the same checkout '
            f'(got {result_a!r} for {_VERIFY_MARKER} vs {result_b!r} for {_GUARD_MARKER})'
        )


class TestReifyCheckoutProvenance:
    """The resolution reports WHERE its answer came from, so no caller re-derives it.

    An ambient ``os.environ`` re-read at message-formatting time is precisely
    the defect this field removes: it would attribute a DISCOVERED root to the
    operator's REIFY_ROOT whenever that env var happened to be set.
    """

    def test_discovery_is_not_named_by_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        start = _plant(tmp_path, 'dark-factory/orchestrator/tests')

        checkout = resolve_reify_checkout(_VERIFY_MARKER, start=start)

        assert checkout.root == (tmp_path / 'reify').resolve()
        assert checkout.named_by_env is False

    def test_override_is_named_by_env(self, tmp_path, monkeypatch):
        named = tmp_path / 'operator-named-reify'
        monkeypatch.setenv(REIFY_ROOT_ENV, str(named))
        start = _plant(tmp_path, 'dark-factory/orchestrator/tests')

        checkout = resolve_reify_checkout(_VERIFY_MARKER, start=start)

        assert checkout.root == named.resolve()
        assert checkout.named_by_env is True

    def test_discovery_miss_is_not_named_by_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        start = tmp_path / 'a' / 'b' / 'test_x.py'

        checkout = resolve_reify_checkout(_VERIFY_MARKER, start=start)

        assert checkout == ReifyCheckout(None, False)

    def test_result_unpacks_as_root_then_provenance(self, tmp_path, monkeypatch):
        """Field ORDER is part of the contract — call sites unpack positionally."""
        monkeypatch.delenv(REIFY_ROOT_ENV, raising=False)
        start = _plant(tmp_path, 'dark-factory/orchestrator/tests')

        root, named_by_env = resolve_reify_checkout(_VERIFY_MARKER, start=start)

        assert root == (tmp_path / 'reify').resolve()
        assert named_by_env is False


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
        assert reify_skip_reason(_VERIFY_MARKER, root, named_by_env=False) is None

    def test_discovery_miss_names_the_marker_and_the_override(self):
        reason = reify_skip_reason(_VERIFY_MARKER, None, named_by_env=False)

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
        reason = reify_skip_reason(_VERIFY_MARKER, None, named_by_env=False)
        assert isinstance(reason, str) and reason
        assert 'None' not in reason, (
            f'discovery-miss reason must not stringify the absent root: {reason!r}'
        )

    def test_set_but_absent_names_the_bad_path_verbatim(self, tmp_path):
        bad_root = tmp_path / 'does-not-exist' / 'reify-typo'

        reason = reify_skip_reason(_VERIFY_MARKER, bad_root, named_by_env=True)

        assert isinstance(reason, str) and reason
        assert str(bad_root) in reason, (
            'the bad path must appear verbatim so it is self-evident in the '
            f'pytest -rs output which path was wrong (got {reason!r})'
        )

    def test_the_two_arms_are_distinguishable(self, tmp_path):
        """Keeps 'nobody has reify' from being read as 'you typed the path wrong'."""
        bad_root = tmp_path / 'does-not-exist' / 'reify-typo'

        miss_reason = reify_skip_reason(_VERIFY_MARKER, None, named_by_env=False)
        named_reason = reify_skip_reason(_VERIFY_MARKER, bad_root, named_by_env=True)

        assert isinstance(miss_reason, str) and miss_reason
        assert isinstance(named_reason, str) and named_reason
        assert miss_reason != named_reason
        assert str(bad_root) in named_reason
        assert str(bad_root) not in miss_reason

    def test_existing_dir_without_the_marker_is_the_set_but_absent_arm(self, tmp_path):
        """A checkout present but without the script is NOT a discovery miss."""
        root = tmp_path / 'reify'
        root.mkdir()

        reason = reify_skip_reason(_VERIFY_MARKER, root, named_by_env=True)

        assert isinstance(reason, str) and reason
        assert str(root) in reason
        assert reason != reify_skip_reason(_VERIFY_MARKER, None, named_by_env=False)

    def test_marker_specific_when_the_root_carries_only_the_other_script(self, tmp_path):
        root = _plant_marker(tmp_path / 'reify', marker=_VERIFY_MARKER)

        assert reify_skip_reason(_VERIFY_MARKER, root, named_by_env=False) is None
        guard_reason = reify_skip_reason(_GUARD_MARKER, root, named_by_env=False)
        assert isinstance(guard_reason, str) and guard_reason
        assert str(_GUARD_MARKER) in guard_reason

    def test_accepts_a_str_marker(self, tmp_path):
        root = _plant_marker(tmp_path / 'reify')
        assert reify_skip_reason('scripts/verify.sh', root, named_by_env=False) is None

        miss = reify_skip_reason('scripts/verify.sh', None, named_by_env=False)
        assert isinstance(miss, str) and miss

    def test_accepts_a_path_marker(self, tmp_path):
        root = _plant_marker(tmp_path / 'reify')
        assert reify_skip_reason(Path('scripts/verify.sh'), root, named_by_env=False) is None


class TestReifySkipReasonProvenance:
    """The message must never blame the operator for a path they did not name."""

    def test_named_by_env_credits_the_override(self, tmp_path):
        bad_root = tmp_path / 'does-not-exist' / 'reify-typo'

        reason = reify_skip_reason(_VERIFY_MARKER, bad_root, named_by_env=True)

        assert isinstance(reason, str) and reason
        assert REIFY_ROOT_ENV in reason, (
            'when the operator named the path, the message must say so — that is '
            f'the difference between "your typo" and "your machine" (got {reason!r})'
        )

    def test_discovered_root_is_not_blamed_on_the_operator(self, tmp_path, monkeypatch):
        """Provenance comes from the RESOLUTION, never from an ambient env re-read.

        REIFY_ROOT is deliberately set here while the root is a DISCOVERED one:
        a re-read at formatting time would tack '(named by REIFY_ROOT)' onto a
        path the operator never named, which is the misattribution the explicit
        parameter exists to prevent.
        """
        monkeypatch.setenv(REIFY_ROOT_ENV, str(tmp_path / 'unrelated-operator-path'))
        discovered = tmp_path / 'reify'
        discovered.mkdir()

        reason = reify_skip_reason(_VERIFY_MARKER, discovered, named_by_env=False)

        assert isinstance(reason, str) and reason
        assert str(discovered) in reason
        assert REIFY_ROOT_ENV not in reason, (
            'a DISCOVERED root must not be attributed to REIFY_ROOT just because '
            f'the env var happens to be set in this process (got {reason!r})'
        )

    def test_provenance_is_the_only_difference_between_the_two_messages(self, tmp_path):
        bad_root = tmp_path / 'does-not-exist' / 'reify-typo'

        named = reify_skip_reason(_VERIFY_MARKER, bad_root, named_by_env=True)
        unnamed = reify_skip_reason(_VERIFY_MARKER, bad_root, named_by_env=False)

        assert isinstance(named, str) and isinstance(unnamed, str)
        assert named != unnamed
        for reason in (named, unnamed):
            assert str(bad_root) in reason
            assert str(_VERIFY_MARKER) in reason

    def test_resolution_and_reason_agree_when_sourced_from_one_call(
        self, tmp_path, monkeypatch
    ):
        """The intended call shape: resolve once, format from that same result."""
        missing = tmp_path / 'does-not-exist' / 'reify-typo'
        monkeypatch.setenv(REIFY_ROOT_ENV, str(missing))
        start = _plant(tmp_path, 'dark-factory/orchestrator/tests')

        checkout = resolve_reify_checkout(_VERIFY_MARKER, start=start)
        reason = reify_skip_reason(
            _VERIFY_MARKER, checkout.root, named_by_env=checkout.named_by_env
        )

        assert isinstance(reason, str) and reason
        assert str(missing) in reason
        assert REIFY_ROOT_ENV in reason


class TestCheckoutSkipReason:
    """`checkout_skip_reason` — the CHECKOUT-reachability gate — WORK item 4.

    A weaker precondition than `reify_skip_reason`'s, and deliberately so.  The
    corpus sweeps that use it need only a git CHECKOUT to be on disk; they do
    not need the guard script the Tier-2 gates need, so their set-but-absent
    arm must NOT borrow the marker-based wording — a reason built around the
    stronger marker would be wrong for them.  Its ``None`` arm, by contrast, IS
    the very same discovery MISS `reify_skip_reason` describes, so it delegates
    there rather than restating it.

    Both arms were hand-rolled, identically, in shared/tests/test_locking.py
    and fused-memory/tests/test_lock_charter_guard.py before task 4259; the
    recomputed-equality assertions below are what prove the wording is
    DELEGATED rather than merely similar-looking.

    Contract, identical to its sibling: a reason string or ``None``, never
    ``''`` — a falsy reason silently disables a caller that gates on
    truthiness, turning a skip into a phantom pass.
    """

    def test_is_exported(self):
        assert 'checkout_skip_reason' in shared.reify_checkout.__all__

    def test_none_root_is_the_shared_discovery_miss_reason(self):
        """The ``None`` arm must BE `reify_skip_reason`'s, not a lookalike.

        Two ways, matching this file's established method: the independent
        invariants first (the reason names the override var and the marker),
        then the recomputed equality — which is the half that actually proves
        delegation, since a hand-rolled string can satisfy the invariants and
        still drift from what the Tier-2 skipifs say about the same condition.
        """
        reason = checkout_skip_reason('reify', None, marker=_GUARD_MARKER)

        assert isinstance(reason, str) and reason
        assert REIFY_ROOT_ENV in reason, (
            f'the discovery-miss reason must name the override: {reason!r}'
        )
        assert str(_GUARD_MARKER) in reason, (
            f'the discovery-miss reason must name the marker: {reason!r}'
        )
        assert reason == reify_skip_reason(_GUARD_MARKER, None, named_by_env=False), (
            'the discovery-miss wording must come from the shared builder, not '
            'a second hand-rolled string that can drift from it'
        )

    @pytest.mark.parametrize('marker', [_VERIFY_MARKER, _GUARD_MARKER])
    def test_none_root_carries_the_callers_own_marker(self, marker):
        """The marker is a parameter, not a default baked in from one call site."""
        reason = checkout_skip_reason('reify', None, marker=marker)

        assert isinstance(reason, str) and reason
        assert str(marker) in reason

    def test_missing_path_names_the_repo_and_the_path(self, tmp_path):
        """A REIFY_ROOT typo must be self-evident in ``pytest -rs`` output.

        This is the exact rendering both suites hand-rolled; it is pinned ONCE,
        here.
        """
        missing = tmp_path / 'no-such-reify-checkout'

        reason = checkout_skip_reason('reify', missing, marker=_GUARD_MARKER)

        assert isinstance(reason, str) and reason
        assert str(missing) in reason, (
            f'the reason must name the path verbatim: {reason!r}'
        )
        assert reason == f'reify checkout not present at {missing}'

    def test_a_regular_file_is_not_a_checkout(self, tmp_path):
        """The arm is ``is_dir()``, not ``exists()``.

        A path that exists but is a FILE is not a checkout; an ``exists()``
        check would ADMIT it and the sweep would then fail deep inside git
        rather than skip with a reason naming the bad path.
        """
        not_a_dir = tmp_path / 'reify-is-a-file'
        not_a_dir.write_text('not a checkout\n')

        reason = checkout_skip_reason('reify', not_a_dir, marker=_GUARD_MARKER)

        assert isinstance(reason, str) and reason
        assert str(not_a_dir) in reason
        assert reason == f'reify checkout not present at {not_a_dir}'

    def test_real_directory_admits_the_run(self, tmp_path):
        """A real checkout directory must return None — the gate must ADMIT.

        Asserted as ``is None`` rather than falsiness: an empty string is falsy
        too, and that is exactly the phantom-pass this contract forbids.
        """
        assert checkout_skip_reason('reify', tmp_path, marker=_GUARD_MARKER) is None

    @pytest.mark.parametrize('repo', ['reify', 'dark-factory'])
    def test_repo_label_is_used_verbatim(self, repo, tmp_path):
        """The label is a parameter — the builder is not hardcoded to 'reify'."""
        missing = tmp_path / 'no-such-checkout'

        reason = checkout_skip_reason(repo, missing, marker=_GUARD_MARKER)

        assert isinstance(reason, str) and reason
        assert reason.startswith(f'{repo} checkout not present at'), (
            f'the caller-supplied repo label must be used verbatim: {reason!r}'
        )

    @pytest.mark.parametrize('repo', ['reify', 'dark-factory'])
    def test_the_set_but_absent_arm_is_never_empty(self, repo, tmp_path):
        """Anti-vacuity on the arm every label may take.

        A falsy-but-not-None reason would pass ``reason is not None`` and then
        skip with no message — or, at a truthiness-gating call site, not skip
        at all.

        Only the set-but-absent arm is swept over both labels: the ``None`` arm
        is reify-ONLY (it raises for any other label — see
        `test_a_none_root_for_another_repo_is_refused`), so feeding
        ``dark-factory`` through it here would have asserted non-emptiness on a
        reason that names the WRONG repo.
        """
        reason = checkout_skip_reason(repo, tmp_path / 'absent', marker=_GUARD_MARKER)

        assert reason is not None
        assert reason != ''

    def test_the_none_arm_is_never_empty(self):
        """Anti-vacuity on the reify-only ``None`` arm."""
        reason = checkout_skip_reason('reify', None, marker=_GUARD_MARKER)

        assert reason is not None
        assert reason != ''

    @pytest.mark.parametrize('repo', ['dark-factory', 'graphiti', 'REIFY'])
    def test_a_none_root_for_another_repo_is_refused(self, repo):
        """The ``None`` arm cannot honor a non-reify label, so it REFUSES one.

        The wording it delegates to names reify and REIFY_ROOT literally.
        Answering a ``dark-factory`` caller with it would report the wrong repo
        and prescribe ``export REIFY_ROOT=...`` — a remedy that cannot fix a
        dark-factory problem.  That is the module's own conflation failure one
        label over, so it raises rather than misattributes.

        Unreachable from today's callers (every non-reify root is resolved from
        ``__file__`` and so is never None); this guard is what keeps it so.  The
        label match is exact — ``'REIFY'`` is not ``'reify'``, and a reason that
        renders a label the caller did not pass is the same defect.
        """
        with pytest.raises(ValueError) as excinfo:
            checkout_skip_reason(repo, None, marker=_GUARD_MARKER)

        message = str(excinfo.value)
        assert repo in message, (
            f'the refusal must name the label it cannot serve: {message!r}'
        )
        assert REIFY_ROOT_ENV in message, (
            f'the refusal must say WHY — the wording is REIFY_ROOT-specific: '
            f'{message!r}'
        )

    def test_reify_is_still_accepted_on_the_none_arm(self):
        """The guard must not have closed the arm its callers actually use."""
        assert checkout_skip_reason('reify', None, marker=_GUARD_MARKER)

    def test_a_falsy_delegated_reason_raises_rather_than_passing_it_on(
        self, monkeypatch
    ):
        """The ``None`` arm's anti-phantom-pass guard, exercised.

        The guard is unreachable through the public surface — `reify_skip_reason`
        returns a non-empty literal on every path where *root* is None — so the
        only way to reach it is to break the delegate, which is exactly the
        defect it exists to catch (a future edit inside this module).  Stubbing
        it here is what makes the branch covered rather than permanently
        untested dead code.

        It must RAISE, not return: a falsy reason reaching a caller that gates
        on truthiness does not skip at all, turning the skip into a phantom
        pass.  ``RuntimeError``, not ``assert`` — an assert is stripped under
        ``-O``, which is precisely when nobody is watching.
        """
        monkeypatch.setattr(shared.reify_checkout, 'reify_skip_reason', lambda *a, **k: '')

        with pytest.raises(RuntimeError) as excinfo:
            checkout_skip_reason('reify', None, marker=_GUARD_MARKER)

        assert 'reify' in str(excinfo.value), (
            f'the failure must name the repo whose gate it just protected: '
            f'{excinfo.value!r}'
        )
