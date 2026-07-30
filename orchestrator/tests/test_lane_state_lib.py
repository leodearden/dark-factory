"""Contract for ``orchestrator/scripts/warm-lane/lib_lane_state.sh``.

Task 3074 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` leaf β, Phase 1).

The warm-lane scripts need two facts that dark-factory — not reify — owns:

1. **A lane's lifecycle state**, from the orchestrator's own durable record at
   ``<worktree_base>/.lane-state/<lane>.json`` (``LaneState`` in
   ``orchestrator/src/orchestrator/lane_lifecycle.py``).
2. **Which worktree bands are protected** from a reclaim sweep
   (``PROTECTED_PREFIXES`` in ``orchestrator/src/orchestrator/git_ops.py``).

Before this leaf, (1) existed only as a private reader inside
``warm-lane-audit.sh`` and (2) existed only as a hand-copied glob literal in
``warm-lane-gc.sh`` — an INV-5 (no-lockstep-duplication) violation whose
failure mode is silent: a band added to ``PROTECTED_PREFIXES`` that nobody
mirrors into the sweep's default becomes a live managed worktree the reaper
will happily remove.

These tests drive the REAL shipped bash via subprocess against a synthetic
``tmp_path`` mount — the same "exercise the file that actually ships" pattern
as ``test_warm_lane_scripts_shipped.py`` — never a fixture copy of it.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

# Reused verbatim from the sibling shipped-script contract rather than
# re-derived: WARM_LANE_SCRIPT_DIR is resolved from __file__ (not the process
# CWD, which differs between the merge-verify harness and a plain
# `pytest orchestrator/tests`), and _sanitized_env strips the resolution-hostile
# keys (REIFY_WARM_LANE_MOUNT, GIT_DIR, GIT_WORK_TREE) that git hooks and
# `git rebase --exec` export — without which a passing result here could be an
# artifact of the ambient checkout instead of the synthetic mount.
from test_warm_lane_scripts_shipped import (  # noqa: E402
    WARM_LANE_SCRIPT_DIR,
    _sanitized_env,
)

LIB = WARM_LANE_SCRIPT_DIR / 'lib_lane_state.sh'


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def _run_sourced(
    snippet: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Source the SHIPPED lib in a fresh bash and run *snippet* against it."""
    script = f'source {shlex.quote(str(LIB))}\n{snippet}\n'
    return subprocess.run(
        ['bash', '-c', script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(cwd if cwd is not None else WARM_LANE_SCRIPT_DIR),
        env=_sanitized_env() if env is None else env,
    )


def _mount(base: Path, lanes: dict[str, dict | str | None]) -> Path:
    """Build a synthetic warm-lane mount under *base*.

    Each key is a lane directory name; the value is its ``.lane-state`` record
    — a dict (written as JSON), a raw string (written verbatim, for corrupt /
    truncated records), or ``None`` for a lane with NO record at all.  The
    ``.lane-state`` directory is created only if at least one lane has a
    record, so "no state dir" is expressible.
    """
    base.mkdir(parents=True, exist_ok=True)
    for lane, record in lanes.items():
        (base / lane).mkdir(exist_ok=True)
        if record is None:
            continue
        state_dir = base / '.lane-state'
        state_dir.mkdir(exist_ok=True)
        text = record if isinstance(record, str) else json.dumps(record, indent=2)
        (state_dir / f'{lane}.json').write_text(text)
    return base


def _globals(stdout: str) -> dict[str, str]:
    """Parse the ``KEY=value`` block emitted by :data:`_ECHO_GLOBALS`."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if '=' in line:
            key, _, value = line.partition('=')
            out[key] = value
    return out


#: Appended to a snippet to publish the three globals for inspection.  Printed
#: one per line with no quoting so an empty value is an empty tail.
_ECHO_GLOBALS = (
    "printf 'RAW=%s\\nTASK=%s\\nCAUSE=%s\\n' "
    '"$LANE_STATE_RAW" "$LANE_STATE_TASK_ID" "$LANE_STATE_CAUSE"'
)

#: A realistic record: the shape ``LaneRecord`` actually serializes.
_ASSIGNED_RECORD = {
    'state': 'assigned',
    'task_id': '5551',
    'title': 'a lane held by a task',
    'branch': 'task/5551',
    'seeded_from_sha': 'abc123',
    'updated_at': '2026-07-30T07:00:00+00:00',
}


class TestLaneStateReadHappyPath:
    """``lane_state_read`` resolves a real record, and fails open without one.

    This reproduces the task's live-pool signal hermetically: on the operational
    pool ``_lane-28`` reads ``assigned 5551`` while a recordless ``_iact-*``
    reads ``unknown``.  Asserting it against the live pool would be asserting
    against mutable state that CI does not have, so the same two shapes are
    built from a synthetic record instead.
    """

    def test_reads_state_and_task_id_from_the_record(self, tmp_path: Path) -> None:
        base = _mount(tmp_path / 'mount', {'_lane-28': _ASSIGNED_RECORD})
        proc = _run_sourced(f'lane_state_read {shlex.quote(str(base / "_lane-28"))}')
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert proc.stdout == 'assigned 5551\n', (
            'lane_state_read must print "<raw> <task_id>" for an assigned lane; '
            f'got {proc.stdout!r}'
        )

    def test_recordless_lane_reads_unknown(self, tmp_path: Path) -> None:
        base = _mount(
            tmp_path / 'mount',
            {'_lane-28': _ASSIGNED_RECORD, '_iact-demo': None},
        )
        proc = _run_sourced(f'lane_state_read {shlex.quote(str(base / "_iact-demo"))}')
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert proc.stdout == 'unknown\n', (
            'a lane with no state record must fail OPEN to a bare "unknown" — '
            'every recordless _iact-* and manual operator worktree takes this '
            f'path; got {proc.stdout!r}'
        )

    def test_publishes_the_three_globals_for_an_assigned_lane(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(tmp_path / 'mount', {'_lane-28': _ASSIGNED_RECORD})
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-28"))} >/dev/null\n'
            f'{_ECHO_GLOBALS}',
        )
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        published = _globals(proc.stdout)
        assert published == {'RAW': 'assigned', 'TASK': '5551', 'CAUSE': ''}, (
            'the three globals must describe ONE observation of ONE record; '
            f'got {published!r}'
        )

    def test_publishes_the_three_globals_for_a_recordless_lane(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(tmp_path / 'mount', {'_iact-demo': None})
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_iact-demo"))} >/dev/null\n'
            f'{_ECHO_GLOBALS}',
        )
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        published = _globals(proc.stdout)
        assert published['RAW'] == 'unknown', (
            f'LANE_STATE_RAW must default to "unknown"; got {published!r}'
        )
        assert published['TASK'] == '', (
            f'no record means no task id to publish; got {published!r}'
        )

    def test_accepts_a_bare_lane_name_against_an_explicit_state_dir(
        self, tmp_path: Path,
    ) -> None:
        """The lane argument may be a bare name, not only a full directory path.

        ``warm-lane-audit.sh`` iterates lane NAMES and carries its own
        ``STATE_DIR``, so the name + explicit-state-dir form is the shape the
        audit's adapter needs.
        """
        base = _mount(tmp_path / 'mount', {'_lane-28': _ASSIGNED_RECORD})
        proc = _run_sourced(
            f'lane_state_read _lane-28 {shlex.quote(str(base / ".lane-state"))}',
        )
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert proc.stdout == 'assigned 5551\n', (
            f'a bare lane name + explicit state dir must resolve; got {proc.stdout!r}'
        )
