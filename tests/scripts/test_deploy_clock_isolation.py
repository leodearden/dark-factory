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
