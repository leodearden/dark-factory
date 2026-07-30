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

import pytest

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


class TestLaneStateReadFailsOpenWithAnAttributableCause:
    """An unreadable record degrades that lane, never the run — and says why.

    ``LANE_STATE_CAUSE`` distinguishes the two causes because they send an
    operator to two different places: ``no-readable-record`` is a
    filesystem/permissions question (and is also the ordinary reading for every
    recordless ``_iact-*`` and manual operator worktree, which leaf γ routes to
    the /proc fallback), while ``unparseable-record`` means the record IS there
    and readable but corrupt, truncated, or reshaped.  Collapsing them into one
    cause would point triage at a missing file that is often sitting right
    there.
    """

    def test_corrupt_record_reads_unknown_with_unparseable_cause(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(
            tmp_path / 'mount',
            {'_lane-9': '{"state": "assig'},   # truncated mid-write
        )
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-9"))}\n{_ECHO_GLOBALS}',
        )
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert proc.stdout.startswith('unknown\n'), (
            f'a corrupt record must fail open to "unknown"; got {proc.stdout!r}'
        )
        published = _globals(proc.stdout)
        assert published['CAUSE'] == 'unparseable-record', (
            'a record that IS present and readable but yields no `state` string '
            f'is unparseable, not missing; got {published!r}'
        )

    def test_record_with_no_state_key_reads_unparseable(self, tmp_path: Path) -> None:
        base = _mount(tmp_path / 'mount', {'_lane-9': {'task_id': '77'}})
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-9"))} >/dev/null\n'
            f'{_ECHO_GLOBALS}',
        )
        published = _globals(proc.stdout)
        assert published['RAW'] == 'unknown'
        assert published['CAUSE'] == 'unparseable-record', (
            f'a readable record with no `state` string is unparseable; {published!r}'
        )

    def test_absent_state_dir_reads_no_readable_record(self, tmp_path: Path) -> None:
        base = _mount(tmp_path / 'mount', {'_iact-demo': None})
        assert not (base / '.lane-state').exists(), 'fixture must have no state dir'
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_iact-demo"))} >/dev/null\n'
            f'{_ECHO_GLOBALS}',
        )
        published = _globals(proc.stdout)
        assert published['RAW'] == 'unknown'
        assert published['CAUSE'] == 'no-readable-record', (
            f'no state dir at all is a missing record, not a corrupt one; {published!r}'
        )

    def test_present_state_dir_missing_record_reads_no_readable_record(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(
            tmp_path / 'mount',
            {'_lane-28': _ASSIGNED_RECORD, '_lane-29': None},
        )
        assert (base / '.lane-state').is_dir(), 'fixture must have a state dir'
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-29"))} >/dev/null\n'
            f'{_ECHO_GLOBALS}',
        )
        published = _globals(proc.stdout)
        assert published['RAW'] == 'unknown'
        assert published['CAUSE'] == 'no-readable-record', (
            f'a state dir with no record for this lane is missing; {published!r}'
        )

    def test_null_task_id_reads_empty_without_losing_the_state(
        self, tmp_path: Path,
    ) -> None:
        """``"task_id": null`` is the shape an UNASSIGNED lane's record carries.

        It must read as an empty task id — the audit's ``pin`` column falls back
        to the branch-derived id on exactly this signal — while the lane's state
        still resolves normally.
        """
        base = _mount(
            tmp_path / 'mount',
            {'_lane-7': '{"state": "released", "task_id": null}'},
        )
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-7"))}\n{_ECHO_GLOBALS}',
        )
        assert proc.stdout.startswith('released\n'), (
            f'a null task_id must not suppress the state; got {proc.stdout!r}'
        )
        published = _globals(proc.stdout)
        assert published == {'RAW': 'released', 'TASK': '', 'CAUSE': ''}, published

    def test_absent_task_id_key_reads_empty_without_losing_the_state(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(tmp_path / 'mount', {'_lane-7': {'state': 'seed'}})
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-7"))}\n{_ECHO_GLOBALS}',
        )
        assert proc.stdout.startswith('seed\n'), proc.stdout
        published = _globals(proc.stdout)
        assert published == {'RAW': 'seed', 'TASK': '', 'CAUSE': ''}, published


class TestLaneStateReadIsNonCreating:
    """The read never brings the thing it is reading into existence.

    Same A1 guarantee the audit's ``_lane_record`` carries, and it matters for
    the same reason: this lib is about to be called by a reclaim sweep, and a
    read that MINTED an empty ``.lane-state/<lane>.json`` would manufacture the
    very record the sweep consults to decide whether a lane is free.
    """

    def test_reading_against_a_mount_with_no_state_dir_creates_nothing(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(tmp_path / 'mount', {'_iact-demo': None})
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_iact-demo"))}',
        )
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert not (base / '.lane-state').exists(), (
            'lane_state_read created the state directory — no mkdir is permitted'
        )
        assert sorted(p.name for p in base.iterdir()) == ['_iact-demo'], (
            f'lane_state_read created something under the mount: '
            f'{sorted(p.name for p in base.iterdir())}'
        )

    def test_reading_a_lane_with_no_record_creates_no_record(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(
            tmp_path / 'mount',
            {'_lane-28': _ASSIGNED_RECORD, '_lane-29': None},
        )
        state_dir = base / '.lane-state'
        proc = _run_sourced(f'lane_state_read {shlex.quote(str(base / "_lane-29"))}')
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert not (state_dir / '_lane-29.json').exists(), (
            'lane_state_read created the missing record — no `>`-open, no touch'
        )
        assert sorted(p.name for p in state_dir.iterdir()) == ['_lane-28.json'], (
            f'state dir gained an entry: {sorted(p.name for p in state_dir.iterdir())}'
        )


class TestLaneStateReadStateDirOverride:
    """An explicit ``<state-dir>`` is honoured verbatim, including outside the mount.

    ``warm-lane-audit.sh`` already supports ``--state-dir`` /
    ``REIFY_WARM_LANE_AUDIT_STATE_DIR``, documented as possibly pointing
    anywhere including outside the mount.  Deriving the state dir solely from
    the lane dir would silently drop that override during the refactor —
    changing observable audit behaviour under the banner of an extraction.
    """

    def test_explicit_state_dir_outside_the_mount_wins(self, tmp_path: Path) -> None:
        base = _mount(tmp_path / 'mount', {'_lane-28': None})
        elsewhere = tmp_path / 'somewhere-else'
        elsewhere.mkdir()
        (elsewhere / '_lane-28.json').write_text(json.dumps(_ASSIGNED_RECORD))

        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-28"))} '
            f'{shlex.quote(str(elsewhere))}',
        )
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert proc.stdout == 'assigned 5551\n', (
            'an explicit state dir outside the mount must be honoured verbatim; '
            f'got {proc.stdout!r}'
        )

    def test_explicit_state_dir_is_not_silently_ignored_for_a_derivable_lane(
        self, tmp_path: Path,
    ) -> None:
        """The override WINS — it does not merely fill in for a missing default.

        The lane's derived state dir here holds a DIFFERENT record, so a read
        that answered from it would pass a weaker test while ignoring the
        operator's override.
        """
        base = _mount(tmp_path / 'mount', {'_lane-28': _ASSIGNED_RECORD})
        elsewhere = tmp_path / 'override'
        elsewhere.mkdir()
        (elsewhere / '_lane-28.json').write_text(
            json.dumps({'state': 'quarantined', 'task_id': '9999'}),
        )
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-28"))} '
            f'{shlex.quote(str(elsewhere))}',
        )
        assert proc.stdout == 'quarantined 9999\n', (
            'the derived default answered instead of the explicit override; '
            f'got {proc.stdout!r}'
        )


class TestLaneStateReadNeverInheritsAPredecessorsValues:
    """Every lane's triple is resolved from scratch.

    The globals are process-wide, so without a reset at entry a lane whose
    record cannot be read would report the PREVIOUS lane's task id — which in
    the audit's ``pin`` column reads as "lane X is held by task N", a claim
    about a lane the record never made.
    """

    def test_recordless_lane_after_an_assigned_lane_publishes_no_task_id(
        self, tmp_path: Path,
    ) -> None:
        base = _mount(
            tmp_path / 'mount',
            {'_lane-28': _ASSIGNED_RECORD, '_iact-demo': None},
        )
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_lane-28"))} >/dev/null\n'
            f'lane_state_read {shlex.quote(str(base / "_iact-demo"))} >/dev/null\n'
            f'{_ECHO_GLOBALS}',
        )
        published = _globals(proc.stdout)
        assert published['TASK'] == '', (
            'the recordless lane inherited its predecessor\'s task id — the '
            f'globals must be reset at entry; got {published!r}'
        )
        assert published['RAW'] == 'unknown', published
        assert published['CAUSE'] == 'no-readable-record', published

    def test_a_clean_read_clears_a_previous_lanes_cause(self, tmp_path: Path) -> None:
        base = _mount(
            tmp_path / 'mount',
            {'_iact-demo': None, '_lane-28': _ASSIGNED_RECORD},
        )
        proc = _run_sourced(
            f'lane_state_read {shlex.quote(str(base / "_iact-demo"))} >/dev/null\n'
            f'lane_state_read {shlex.quote(str(base / "_lane-28"))} >/dev/null\n'
            f'{_ECHO_GLOBALS}',
        )
        published = _globals(proc.stdout)
        assert published == {'RAW': 'assigned', 'TASK': '5551', 'CAUSE': ''}, (
            'a successful read must clear the previous lane\'s UNKNOWN cause, or '
            f'the audit would warn about a lane that resolved fine; {published!r}'
        )


class TestLaneStateClass:
    """``lane_state_class`` is the ONE normative raw-state -> column mapping.

    Before this leaf the table lived only inside ``warm-lane-audit.sh``'s
    ``_read_lane_assignment``.  It is normative because downstream consumers
    partition the whole pool on it: ASSIGNED means reserved for a task,
    RELEASED means in the pool and not reserved, QUARANTINED means withheld,
    and UNKNOWN is the fail-open bucket that is counted free (conservative).
    """

    @pytest.mark.parametrize(
        ('raw', 'column'),
        [
            ('assigned', 'ASSIGNED'),
            ('in_use', 'ASSIGNED'),
            ('released', 'RELEASED'),
            ('seed', 'RELEASED'),
            ('registered', 'RELEASED'),
            ('quarantined', 'QUARANTINED'),
            ('', 'UNKNOWN'),
            ('unknown', 'UNKNOWN'),
            ('reticulating', 'UNKNOWN'),
            ('ASSIGNED', 'UNKNOWN'),   # raw values are lowercase; no folding
        ],
    )
    def test_mapping_table(self, raw: str, column: str) -> None:
        proc = _run_sourced(f'lane_state_class {shlex.quote(raw)}')
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert proc.stdout == f'{column}\n', (
            f'lane_state_class {raw!r} must be {column}; got {proc.stdout!r}'
        )

    def test_every_lane_state_enum_member_maps_to_a_known_column(self) -> None:
        """The drift gate: a new dark-factory ``LaneState`` fails LOUDLY here.

        This is what makes the bash mapping a genuine consumer of the Python
        enum rather than a hand-copy of it.  Without this guard, adding a
        seventh ``LaneState`` member would silently degrade every lane carrying
        it to ``assigned=UNKNOWN`` — and a pool-wide UNKNOWN spike is
        indistinguishable from a real state-dir outage, which is a triage trap.
        Green today (all six members are mapped); RED the moment one is added
        without teaching the mapping about it.
        """
        from orchestrator.lane_lifecycle import LaneState

        unmapped = []
        for member in LaneState:
            proc = _run_sourced(f'lane_state_class {shlex.quote(member.value)}')
            column = proc.stdout.strip()
            if column == 'UNKNOWN' or proc.returncode != 0:
                unmapped.append((member.name, member.value, column))
        assert not unmapped, (
            'LaneState members that lane_state_class does not recognise: '
            f'{unmapped!r}. Teach the `case` in '
            'orchestrator/scripts/warm-lane/lib_lane_state.sh about them — '
            'leaving them UNKNOWN silently degrades every lane in that state.'
        )
