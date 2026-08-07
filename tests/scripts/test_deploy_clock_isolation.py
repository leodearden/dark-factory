"""Guard tests: a test run can never falsify a REAL deploy clock (task 3797).

``scripts/restart-all-orchestrators.sh`` resolves its ``CLOCK_FILE`` from
``$ORCH_FLEET_DEPLOY_CLOCK``, defaulting to
``$REPO_DIR/data/orchestrator/last_redeploy_orchestrator.json`` — the live
checkout the script sits in — and stamps it on its verified-fresh exit-0 path.
``scripts/tests/test_restart_all_orchestrators.py`` drove that path against a
fake ``systemctl`` without setting the env var, so an ordinary green test run
wrote a REAL "the fleet just redeployed" stamp.  Nothing distinguishes it from
a genuine one: ``scripts/orchestrator-watchdog.py`` reads the file and SKIPS
its staleness pass for ``ORCH_RESTART_MIN_INTERVAL_SECS`` (28800s = 8h), so the
fleet-staleness backstop was silently disarmed for the rest of the day.

The fix for that one suite is a conftest redirect.  This module covers the
suite-wide, opt-out-impossible SECOND layer in the root ``df_pytest_isolation``
module — the one that catches the next spawner that forgets:

* two pure helpers, ``deploy_clock_snapshot`` / ``deploy_clock_violation_reason``,
  directly testable without a nested pytest run; and
* a session-scoped autouse fixture that snapshots the protected clocks around
  the whole session and fails loudly if any of them moved.

Shaped like ``test_basetemp_git_isolation.py``, the module covering this repo's
other suite-wide isolation defence, so the two read as one family.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import df_pytest_isolation  # noqa: E402
from df_pytest_isolation import (  # noqa: E402
    PROTECTED_DEPLOY_CLOCK_RELPATHS,
    deploy_clock_guard_roots,
    deploy_clock_snapshot,
    deploy_clock_violation_reason,
)

# NOT `from df_pytest_isolation import _df_deploy_clocks_unwritten`. Importing a
# fixture into a TEST module binds it as a module-scoped fixture that SHADOWS
# the conftest's — which would make the liveness test below resolve its own
# import and pass even with the conftest wiring removed, i.e. exactly the dead
# defence it exists to detect. Reach it through the module instead.
_GUARD_NAME = '_df_deploy_clocks_unwritten'

# pytest's fixture marker is private and has MOVED: <=8.x hangs it off the
# decorated function as `_pytestfixturefunction`, 9.x wraps the function in a
# `FixtureFunctionDefinition` carrying `_fixture_function_marker`. Both spellings
# are accepted, and neither-found is an explicit failure rather than a skipped
# assertion — a private-API pin that silently stops finding its target is worse
# than no pin, because it still reads as coverage.
_MARKER_ATTRS = ('_fixture_function_marker', '_pytestfixturefunction')


def _fixture_marker(fixture: object) -> Any:
    """Return pytest's fixture marker, whatever this pytest version calls it.

    `Any`, not `object`, is the honest annotation and is load-bearing for the
    type gate: the two `_MARKER_ATTRS` spellings above hang DIFFERENT private
    classes off the fixture, neither of which pytest exports, so there is no
    real static type that covers both — and `object` makes every `.scope` /
    `.autouse` read below a `reportAttributeAccessIssue`. Do not "tighten" this
    back to `object`; pin the attributes with assertions instead, as the caller
    does.
    """
    for attr in _MARKER_ATTRS:
        marker = getattr(fixture, attr, None)
        if marker is not None:
            return marker
    pytest.fail(
        f'cannot find pytest\'s fixture marker on {fixture!r} under any of '
        f'{_MARKER_ATTRS}. pytest moved its private fixture API again — find the '
        'new spelling and add it, do NOT delete this assertion.',
        pytrace=False,
    )

_FLEET_RELPATH = 'data/orchestrator/last_redeploy_orchestrator.json'
_FM_RELPATH = 'data/fused-memory/last_redeploy_fused_memory.json'


def _write(root: Path, relpath: str, body: bytes) -> Path:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return target


class TestProtectedRelpaths:
    """The list of clocks the guard watches."""

    def test_it_is_a_non_empty_tuple(self) -> None:
        """A tuple, not a list: a module-level mutable would let one suite
        quietly shrink the protected set for every later one in the process.
        """
        assert isinstance(PROTECTED_DEPLOY_CLOCK_RELPATHS, tuple)
        assert PROTECTED_DEPLOY_CLOCK_RELPATHS

    def test_it_covers_the_orchestrator_fleet_clock(self) -> None:
        assert _FLEET_RELPATH in PROTECTED_DEPLOY_CLOCK_RELPATHS

    def test_it_covers_the_fused_memory_deploy_clock(self) -> None:
        """The identical defect class, not an adjacent one.

        Same watchdog, same min-interval semantics (FM_RESTART_MIN_INTERVAL_SECS,
        8h), and it is stamped from a chained ``--stamp-fm-deploy-clock``
        SUBPROCESS — the same shape (a spawned process resolving a path from the
        environment) that produced the fleet-clock bug.
        """
        assert _FM_RELPATH in PROTECTED_DEPLOY_CLOCK_RELPATHS

    def test_the_relpaths_are_relative(self) -> None:
        """They are joined onto a root by the snapshot; an absolute entry would
        silently escape that root and point at the real checkout.
        """
        for relpath in PROTECTED_DEPLOY_CLOCK_RELPATHS:
            assert not Path(relpath).is_absolute(), relpath


# The gitignored worktree-root vocabulary, .gitignore:15-17.  Spelled out here
# rather than imported so the test pins the values independently of the module.
_WORKTREE_ROOTS = ('.worktrees', '.worktrees-orphaned', '.eval-worktrees')


class TestDeployClockGuardRoots:
    """WHICH checkouts the guard watches — one root is not enough."""

    def test_a_plain_checkout_yields_only_itself(self, tmp_path: Path) -> None:
        """No ``.worktrees`` parent (the main checkout's own shape) — exactly one
        root, so a main-checkout run does not snapshot the same file twice.
        """
        root = tmp_path / 'dark-factory'
        root.mkdir()

        assert deploy_clock_guard_roots(root) == (root.resolve(),)

    @pytest.mark.parametrize('worktree_dir', _WORKTREE_ROOTS)
    def test_a_worktree_also_yields_the_enclosing_main_checkout(
        self, tmp_path: Path, worktree_dir: str,
    ) -> None:
        """The whole point of this helper.

        ``scripts/orchestrator-watchdog.py`` HARDCODES ``REPO_DIR =
        "/home/leo/src/dark-factory"`` (line 96), so its ``FM_DEPLOY_CLOCK_PATH``
        default and the fleet restart script it spawns resolve into the MAIN
        checkout no matter which worktree the suite runs from.  A guard rooted
        only at the worktree would report all-clear while the REAL production
        clock was falsified — the fused-memory entry in the protected set would
        be watching a path nothing under a worktree can even write.
        """
        main = tmp_path / 'dark-factory'
        worktree = main / worktree_dir / '3797'
        worktree.mkdir(parents=True)

        assert deploy_clock_guard_roots(worktree) == (worktree.resolve(), main.resolve())

    def test_the_run_s_own_checkout_comes_first(self, tmp_path: Path) -> None:
        """Ordering is load-bearing for the failure MESSAGE: when a test falsifies
        both, the reader should be pointed at the checkout they are working in.
        """
        main = tmp_path / 'dark-factory'
        worktree = main / '.worktrees' / '3797'
        worktree.mkdir(parents=True)

        assert deploy_clock_guard_roots(worktree)[0] == worktree.resolve()

    def test_the_roots_are_deduped(self, tmp_path: Path) -> None:
        """Snapshotting a root twice would compare it against itself twice and
        double every failure message; it must be impossible by construction.
        """
        worktree = tmp_path / 'dark-factory' / '.worktrees' / '3797'
        worktree.mkdir(parents=True)

        for root in (worktree, tmp_path / 'plain'):
            roots = deploy_clock_guard_roots(root)
            assert len(roots) == len(set(roots)), roots

    def test_a_lookalike_directory_name_is_not_a_worktree(self, tmp_path: Path) -> None:
        """Matching is on the exact parent NAME, never a substring: a directory
        merely containing the word is an ordinary, unrelated checkout, and
        promoting its parent to a guarded root would watch some stranger's tree.
        """
        root = tmp_path / 'my.worktrees-backup' / 'dark-factory'
        root.mkdir(parents=True)

        assert deploy_clock_guard_roots(root) == (root.resolve(),)

    def test_the_roots_are_absolute(self, tmp_path: Path) -> None:
        """The snapshot joins relpaths onto these; a relative root would resolve
        against whatever cwd the session happened to leave behind.
        """
        (tmp_path / 'dark-factory').mkdir()

        for root in deploy_clock_guard_roots(Path('dark-factory')):
            assert root.is_absolute(), root

    def test_this_checkout_is_guarded_in_this_run(self) -> None:
        """Liveness, not just shape: whatever checkout this file lives in — main
        or worktree — must be among the roots the guard would snapshot.
        """
        assert REPO_ROOT.resolve() in deploy_clock_guard_roots(REPO_ROOT)


class TestDeployClockSnapshot:
    """What the guard records at session start."""

    def test_absent_clocks_snapshot_as_none(self, tmp_path: Path) -> None:
        """Absence is a first-class value, not an omitted key.

        The common case IS absence — a fresh worktree has no ``data/`` dir at
        all — so "file created during the run", the exact 3797 shape, is only
        detectable if absence is recorded rather than skipped.
        """
        snapshot = deploy_clock_snapshot(tmp_path)

        assert set(snapshot) == set(PROTECTED_DEPLOY_CLOCK_RELPATHS)
        assert all(value is None for value in snapshot.values())

    def test_a_missing_data_dir_does_not_raise(self, tmp_path: Path) -> None:
        """No ``data/`` at all is the ordinary state of a task worktree."""
        assert not (tmp_path / 'data').exists()

        deploy_clock_snapshot(tmp_path)  # must not raise

    def test_a_present_clock_snapshots_as_bytes_and_mtime(self, tmp_path: Path) -> None:
        clock = _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        snapshot = deploy_clock_snapshot(tmp_path)

        assert snapshot[_FLEET_RELPATH] == (clock.read_bytes(), clock.stat().st_mtime_ns)

    def test_every_protected_path_is_keyed_independently(self, tmp_path: Path) -> None:
        _write(tmp_path, _FLEET_RELPATH, b'fleet')

        snapshot = deploy_clock_snapshot(tmp_path)

        assert snapshot[_FLEET_RELPATH] is not None
        assert snapshot[_FM_RELPATH] is None


class TestDeployClockViolationReason:
    """What counts as a falsified clock."""

    def test_unchanged_absent_is_clean(self, tmp_path: Path) -> None:
        before = deploy_clock_snapshot(tmp_path)

        assert deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path)) is None

    def test_unchanged_present_is_clean(self, tmp_path: Path) -> None:
        """A suite that merely READS the clock must stay green — the guard
        watches for writes, not for access.
        """
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)

        assert deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path)) is None

    def test_a_created_clock_is_a_violation(self, tmp_path: Path) -> None:
        """The exact 3797 shape: absent before, stamped by a test, present after."""
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1786033966, "iso": "..."}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_changed_bytes_are_a_violation(self, tmp_path: Path) -> None:
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 2, "iso": "y"}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_identical_bytes_with_a_moved_mtime_are_a_violation(self, tmp_path: Path) -> None:
        """NOT pedantry — the case a bytes-only guard would silently miss.

        ``stamp_fleet_deploy_clock`` writes ``{"ts": <int seconds>, "iso": ...}``
        at ONE-SECOND resolution, so two stamps landing inside the same second
        are byte-identical.  A test suite restamping in a tight loop produces
        exactly that, and mtime is the only signal left.
        """
        clock = _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)
        stat = clock.stat()
        os.utime(clock, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        after = deploy_clock_snapshot(tmp_path)
        before_entry, after_entry = before[_FLEET_RELPATH], after[_FLEET_RELPATH]
        # Bound before subscripting: the snapshot value is `tuple | None`, and
        # a None on either side would mean the clock vanished rather than being
        # restamped — a different violation, and one that would make the
        # bytes-identical premise below vacuous rather than false.
        assert before_entry is not None and after_entry is not None
        assert after_entry[0] == before_entry[0], 'bytes must be identical'

        reason = deploy_clock_violation_reason(before, after)

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_a_deleted_clock_is_a_violation(self, tmp_path: Path) -> None:
        """Deletion is as damaging as a stamp, in the other direction: it makes
        the watchdog see "never redeployed" and lose the real last-deploy time.
        """
        clock = _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)
        clock.unlink()

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_the_fm_clock_is_watched_too(self, tmp_path: Path) -> None:
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FM_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FM_RELPATH in reason

    def test_the_reason_names_the_remedy_and_the_benign_alternative(
        self, tmp_path: Path,
    ) -> None:
        """The message has to be actionable at 3am by whoever sees it fail.

        Both readings must be spelled out: a test that forgot the env var (the
        common case, remedy = point it at a tmp file) AND a genuine concurrent
        fleet redeploy in a machine-operated checkout (not a bug at all).  A
        bare "the clock changed" invites the reader to assume whichever one
        they thought of first.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert 'ORCH_FLEET_DEPLOY_CLOCK' in reason
        assert 'redeploy' in reason.lower()

    def test_the_reason_prints_the_observed_bodies(self, tmp_path: Path) -> None:
        """Triage must be possible from the failure OUTPUT alone.

        Telling the reader to "compare the {ts, iso} body against the deploy you
        expect" — the step that separates a genuine concurrent redeploy from a
        test bug — is useless if the bodies are not printed: by the time anyone
        reads the ERROR the file may have moved again, so re-reading it is not the
        same observation.  Both readings are already in hand in the snapshots.
        """
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 111, "iso": "2026-08-06T00:00:00+00:00"}')
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 222, "iso": "2026-08-07T00:00:00+00:00"}')
        after = deploy_clock_snapshot(tmp_path)

        reason = deploy_clock_violation_reason(before, after)

        assert reason is not None
        assert '"ts": 111' in reason, reason
        assert '"ts": 222' in reason, reason
        before_entry, after_entry = before[_FLEET_RELPATH], after[_FLEET_RELPATH]
        # Bound before subscripting: a snapshot value is `tuple | None` by design.
        assert before_entry is not None and after_entry is not None
        assert str(before_entry[1]) in reason, 'the before mtime_ns must be printed'
        assert str(after_entry[1]) in reason, 'the after mtime_ns must be printed'

    def test_the_reason_reports_absence_rather_than_a_bare_none(
        self, tmp_path: Path,
    ) -> None:
        """The created case (the 3797 shape) has no before body to print, and
        ``None`` would read as "the guard failed to look" rather than "the file
        did not exist".
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert 'absent' in reason, reason

    def test_a_root_names_the_absolute_file(self, tmp_path: Path) -> None:
        """A run guards MORE THAN ONE checkout (see deploy_clock_guard_roots), so
        a bare relpath leaves the reader unable to tell which one was falsified.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}')
        after = deploy_clock_snapshot(tmp_path)

        reason = deploy_clock_violation_reason(before, after, root=tmp_path)

        assert reason is not None
        assert str(tmp_path / _FLEET_RELPATH) in reason, reason

    def test_the_detector_is_not_vacuous_end_to_end(self, tmp_path: Path) -> None:
        """Self-test: snapshot and detector agree on a dict shape that WORKS.

        Every case above builds its snapshots through ``deploy_clock_snapshot``
        precisely so the two helpers cannot drift into mutually incompatible
        shapes while each stays green on hand-built dicts.  A guard that
        silently matches nothing is worse than no guard, because it reads as
        coverage.  This asserts both directions in one test: clean stays clean,
        and a single real write flips it.
        """
        clean = deploy_clock_snapshot(tmp_path)
        assert deploy_clock_violation_reason(clean, deploy_clock_snapshot(tmp_path)) is None

        _write(tmp_path, _FLEET_RELPATH, b'stamped by a test\n')

        assert deploy_clock_violation_reason(clean, deploy_clock_snapshot(tmp_path)) is not None


class TestGuardIsLiveInThisRun:
    """The fixture is WIRED, not merely defined.

    Everything above tests pure functions against ``tmp_path`` roots; all of it
    would stay green if the fixture were never loaded by any conftest.  This
    class is the only assertion that the defence is actually armed in the
    process running it — the difference between a wired defence and a dead one.
    """

    def test_the_guard_fixture_exists(self) -> None:
        assert hasattr(df_pytest_isolation, _GUARD_NAME), (
            f'df_pytest_isolation defines no {_GUARD_NAME}; the pure helpers '
            'above protect nothing on their own.'
        )

    def test_the_guard_fixture_is_session_scoped_and_autouse(self) -> None:
        """Both properties pinned STRUCTURALLY, not by inspection of behaviour.

        A function-scoped or non-autouse variant would still import cleanly and
        keep every test above green while protecting nothing: function scope
        would miss writes from module-/session-scoped fixtures (where expensive
        subprocess setup tends to live), and without ``autouse`` nothing would
        ever request it.
        """
        marker = _fixture_marker(getattr(df_pytest_isolation, _GUARD_NAME))

        assert marker.scope == 'session', f'scope is {marker.scope!r}, expected session'
        assert marker.autouse is True, 'the guard must be autouse — nothing requests it'

    def test_the_guard_fixture_is_registered_in_this_run(self, request) -> None:
        """The conftest binding is real WIRING, not a dormant definition.

        pytest only collects fixtures bound into a conftest's namespace, which
        is why df_pytest_isolation's fixtures are imported there under
        ``# noqa: F401 — the binding IS the wiring``. Deleting that import
        breaks nothing visible except this assertion.
        """
        try:
            request.getfixturevalue(_GUARD_NAME)
        except pytest.FixtureLookupError:
            pytest.fail(
                f'{_GUARD_NAME} is not registered for this rootdir. Wire the '
                'test-root conftest to import it from df_pytest_isolation '
                '(`# noqa: F401 — the binding IS the wiring`); without that, '
                'this whole suite runs with no deploy-clock guard.',
                pytrace=False,
            )


# ---------------------------------------------------------------------------
# The guard's FAILURE contract, exercised through a real nested pytest run.
#
# Everything above pins the helpers, the marker and the registration; none of it
# pins what a violation actually DOES to the run. A refactor that warned instead
# of failing, or moved the check into a fixture nobody requests, would leave
# every test in this file green while the defence emitted nothing an exit code
# could carry. Only a nested run can observe "the process exits non-zero even
# though every test passed", because a fixture cannot fail its own session.
# ---------------------------------------------------------------------------

# Minimal ini so the nested run's rootdir is the tmp tree and NOT this repo:
# without it pytest walks up looking for an inifile and would inherit this
# repo's addopts (`--import-mode=importlib -m 'not smoke ...'`).
_NESTED_INI = '[pytest]\n'

_NESTED_CONFTEST = '''\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from df_pytest_isolation import _df_deploy_clocks_unwritten  # noqa: F401
'''

_NESTED_STAMP_BODY = '{"ts": 1786033966, "iso": "2026-08-06T16:32:46+00:00"}'


def _nested_test_source(*, writes_clock: bool) -> str:
    """Source for the nested test module — which PASSES either way.

    Built by concatenation rather than ``str.format``: the stamp body is JSON, so
    a template would have to escape its braces, and a mis-escaped one would
    silently write a different file than the guard watches.
    """
    body = (
        (
            '    clock = Path(__file__).resolve().parent / RELPATH\n'
            '    clock.parent.mkdir(parents=True, exist_ok=True)\n'
            f'    clock.write_text({_NESTED_STAMP_BODY!r})\n'
        )
        if writes_clock
        else '    pass\n'
    )
    return (
        'from pathlib import Path\n'
        '\n'
        f'RELPATH = {_FLEET_RELPATH!r}\n'
        '\n'
        '\n'
        'def test_a_forgetful_spawner():\n'
        '    """PASSES. The damage is to the checkout, not to this result."""\n'
        + body
    )


def _nested_run(tmp_path: Path, *, writes_clock: bool) -> subprocess.CompletedProcess[str]:
    """Run a throwaway pytest session wired to the guard, in its own tmp checkout.

    The copied ``df_pytest_isolation.py`` sits at the tmp tree's root, so the
    guard's ``Path(__file__).resolve().parent`` resolves THERE and the protected
    clocks it watches are the tmp ones — the real checkout is never involved.
    """
    root = tmp_path / ('violating' if writes_clock else 'clean')
    root.mkdir()
    shutil.copy2(Path(df_pytest_isolation.__file__), root / 'df_pytest_isolation.py')
    (root / 'pytest.ini').write_text(_NESTED_INI)
    (root / 'conftest.py').write_text(_NESTED_CONFTEST)
    (root / 'test_forgetful.py').write_text(_nested_test_source(writes_clock=writes_clock))
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(root)],
        cwd=root, capture_output=True, text=True, timeout=300,
    )


class TestTheGuardFailsTheRunEndToEnd:
    """A violation must cost the RUN, not merely log something."""

    def test_a_stamped_clock_fails_a_session_whose_tests_all_passed(
        self, tmp_path: Path,
    ) -> None:
        result = _nested_run(tmp_path, writes_clock=True)
        combined = result.stdout + result.stderr

        assert result.returncode != 0, (
            'a run that falsified a deploy clock exited 0 — the guard is inert. '
            f'stdout={result.stdout!r}'
        )
        assert '1 passed' in combined, (
            'the nested TEST must still pass, or this proves nothing about a '
            f'green suite being caught. output={combined!r}'
        )
        assert 'falsified a REAL deploy clock' in combined, combined
        # The BASENAME, not the relpath: the message names the absolute file, and
        # asserting on a 48-char segment of a long tmp path would be hostage to
        # terminal-width wrapping rather than to the guard's behaviour.
        assert Path(_FLEET_RELPATH).name in combined, combined

    def test_the_same_harness_without_the_write_exits_zero(self, tmp_path: Path) -> None:
        """Non-vacuity control: the identical nested tree, minus the clock write.

        Without it the failure above could be any nested-harness breakage — a
        bad ini, an unimportable module, a missing pytest.
        """
        result = _nested_run(tmp_path, writes_clock=False)

        assert result.returncode == 0, (
            f'the control run failed for an unrelated reason: '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert 'falsified a REAL deploy clock' not in result.stdout
