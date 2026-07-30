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
import shutil
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

#: Resolved ONCE, absolutely, so a case may replace ``PATH`` wholesale to hide
#: ``python3`` from the library without also hiding the interpreter this harness
#: needs to launch it.
_BASH = shutil.which('bash') or '/bin/bash'


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def _run_sourced(
    snippet: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Source the SHIPPED lib in a fresh bash and run *snippet* against it.

    *timeout* is generous by default and raised further by the
    ``lane_protect_glob`` cases: unlike the pure-bash lane-state half, those
    pay a real ``python3`` interpreter start plus the ``orchestrator.git_ops``
    import chain, which on a loaded host (this repo's own fleet saturates it)
    measures single-digit seconds cold.
    """
    script = f'source {shlex.quote(str(LIB))}\n{snippet}\n'
    return subprocess.run(
        [_BASH, '-c', script],
        capture_output=True,
        text=True,
        timeout=timeout,
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


# ---------------------------------------------------------------------------
# warm-lane-audit.sh: the extract-and-unify contract (PRD §8)
# ---------------------------------------------------------------------------

AUDIT = WARM_LANE_SCRIPT_DIR / 'warm-lane-audit.sh'

#: A stub task-status oracle: echoes back the id it was asked about.  That is
#: what makes the record's task_id OBSERVABLE in the audit's `pin` column,
#: which otherwise reports the pinning task's STATUS rather than its id.
_STATUS_STUB = '#!/usr/bin/env bash\nprintf \'status-%s\\n\' "$1"\n'


def _git_worktree(path: Path) -> None:
    """Make *path* pass the audit's ``_is_git_worktree`` gate."""
    subprocess.run(
        ['git', 'init', '-q', '-b', 'main'],
        cwd=str(path),
        check=True,
        timeout=60,
        env=_sanitized_env(),
    )


def _audit_json(
    base: Path, *, status_cmd: Path, state_dir: Path | None = None,
) -> tuple[dict, str]:
    """Run the SHIPPED warm-lane-audit.sh over *base* and parse its JSON."""
    extra = {'REIFY_LANE_LEAK_STATUS_CMD': str(status_cmd)}
    if state_dir is not None:
        extra['REIFY_WARM_LANE_AUDIT_STATE_DIR'] = str(state_dir)
    proc = subprocess.run(
        [str(AUDIT), '--mount', str(base), '--format', 'json'],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(WARM_LANE_SCRIPT_DIR),
        env=_sanitized_env(extra=extra),
    )
    assert proc.returncode == 0, (
        f'warm-lane-audit.sh must never abort (advisory-only); '
        f'rc={proc.returncode} stderr={proc.stderr!r}'
    )
    return json.loads(proc.stdout), proc.stderr


#: One mount covering every branch of the reader: each column, both task-id
#: shapes, and all three UNKNOWN causes.
_AUDIT_LANES: dict[str, dict | str | None] = {
    '_lane-1': {'state': 'assigned', 'task_id': '5551'},
    '_lane-2': {'state': 'in_use', 'task_id': '42'},
    '_lane-3': {'state': 'released', 'task_id': None},
    '_lane-4': {'state': 'quarantined', 'task_id': None},
    '_lane-5': '{"state": "assig',                         # unparseable-record
    '_lane-6': None,                                       # no-readable-record
    '_lane-7': {'state': 'reticulating', 'task_id': '9'},  # unrecognized-state
    '_lane-8': {'state': 'assigned', 'task_id': None},     # null id -> fallback
}


@pytest.fixture
def audit_mount(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic mount whose lanes are real git worktrees, plus the stub."""
    base = _mount(tmp_path / 'mount', _AUDIT_LANES)
    for lane in _AUDIT_LANES:
        _git_worktree(base / lane)
    status_cmd = tmp_path / 'status-stub.sh'
    status_cmd.write_text(_STATUS_STUB)
    status_cmd.chmod(0o755)
    return base, status_cmd


class TestAuditLaneStateBehaviourIsUnchanged:
    """The audit's observable per-lane verdict survives the extraction.

    Characterization test, written against the PRE-refactor script and required
    to keep passing after it: an "extraction" that changed what an operator
    reads off the audit would not be an extraction.  Every expected value here
    was captured from the reader as it shipped in warm-lane-audit.sh.
    """

    @pytest.mark.parametrize(
        ('lane', 'assigned', 'pin'),
        [
            ('_lane-1', 'ASSIGNED', 'status-5551'),
            ('_lane-2', 'ASSIGNED', 'status-42'),
            ('_lane-3', 'RELEASED', '-'),
            ('_lane-4', 'QUARANTINED', '-'),
            ('_lane-5', 'UNKNOWN', '-'),
            ('_lane-6', 'UNKNOWN', '-'),
            ('_lane-7', 'UNKNOWN', '-'),
            # `"task_id": null` on an ASSIGNED lane: the pin falls back to the
            # branch-derived id, which is absent here, so the column reports
            # `unknown` rather than inventing a holder.
            ('_lane-8', 'ASSIGNED', 'unknown'),
        ],
    )
    def test_assigned_column_and_pin(
        self,
        audit_mount: tuple[Path, Path],
        lane: str,
        assigned: str,
        pin: str,
    ) -> None:
        base, status_cmd = audit_mount
        report, _ = _audit_json(base, status_cmd=status_cmd)
        rows = {row['lane']: row for row in report['lanes']}
        assert lane in rows, f'{lane} missing from the report: {sorted(rows)}'
        assert rows[lane]['assigned'] == assigned, rows[lane]
        assert rows[lane]['pin'] == pin, rows[lane]

    @pytest.mark.parametrize(
        ('lane', 'cause'),
        [
            ('_lane-5', 'unparseable-record'),
            ('_lane-6', 'no-readable-record'),
            ('_lane-7', 'unrecognized-state:reticulating'),
        ],
    )
    def test_the_three_unknown_causes_are_still_named_on_stderr(
        self, audit_mount: tuple[Path, Path], lane: str, cause: str,
    ) -> None:
        """All three causes survive, including the one the lib does NOT set.

        ``unrecognized-state:<raw>`` is derived by the audit from a UNKNOWN
        class with a non-empty raw, not published by ``lane_state_read`` — and
        it is the load-bearing one: a mass state_unknown spike carrying one
        repeated raw value is the SCHEMA-DRIFT signal, and no other cause looks
        like that.
        """
        base, status_cmd = audit_mount
        _, stderr = _audit_json(base, status_cmd=status_cmd)
        assert f'lane={lane}: assignment state unknown ({cause})' in stderr, (
            f'the {cause!r} warning for {lane} is gone from stderr:\n{stderr}'
        )

    def test_headroom_counters_still_partition_the_pool(
        self, audit_mount: tuple[Path, Path],
    ) -> None:
        base, status_cmd = audit_mount
        report, _ = _audit_json(base, status_cmd=status_cmd)
        headroom = report['headroom']
        assert headroom['resident'] == len(_AUDIT_LANES), headroom
        assert headroom['assigned'] == 3, headroom       # _lane-1, -2, -8
        assert headroom['quarantined'] == 1, headroom    # _lane-4
        assert headroom['state_unknown'] == 3, headroom  # _lane-5, -6, -7
        assert headroom['pinned'] == 3, headroom

    def test_the_state_dir_override_still_wins(
        self, audit_mount: tuple[Path, Path], tmp_path: Path,
    ) -> None:
        """``REIFY_WARM_LANE_AUDIT_STATE_DIR`` may point outside the mount.

        Documented behaviour the extraction must not silently drop — which is
        exactly what deriving the state dir solely from the lane dir would do.
        """
        base, status_cmd = audit_mount
        elsewhere = tmp_path / 'state-elsewhere'
        elsewhere.mkdir()
        (elsewhere / '_lane-1.json').write_text(
            json.dumps({'state': 'quarantined', 'task_id': '777'}),
        )
        report, _ = _audit_json(base, status_cmd=status_cmd, state_dir=elsewhere)
        rows = {row['lane']: row for row in report['lanes']}
        assert rows['_lane-1']['assigned'] == 'QUARANTINED', (
            f'the override was ignored in favour of the derived default: '
            f'{rows["_lane-1"]}'
        )
        # Every other lane's record lives in the mount's own .lane-state, which
        # the override displaces — so they all go no-readable-record.
        assert rows['_lane-2']['assigned'] == 'UNKNOWN', rows['_lane-2']


class TestAuditReadsThroughTheLib:
    """The INV-5 guard: ONE definition site, and the audit is wired to it.

    The behavioural class above would pass just as well against a second copy
    of the reader living inside the audit — which is precisely the duplication
    this leaf exists to close.  These assertions are what make it an extraction
    rather than an addition.
    """

    #: The distinctive tail of the sed scalar-extraction idiom.  Matching on the
    #: idiom rather than the function NAME is deliberate: a copy that renamed
    #: the function would still be a copy.
    _SCALAR_IDIOM = '[[:space:]]*:[[:space:]]*\\"([^\\"]*)\\".*'

    def test_the_scalar_extraction_idiom_has_exactly_one_definition_site(
        self,
    ) -> None:
        hits = {
            path.name: path.read_text().count(self._SCALAR_IDIOM)
            for path in sorted(WARM_LANE_SCRIPT_DIR.glob('*.sh'))
            if self._SCALAR_IDIOM in path.read_text()
        }
        assert hits == {'lib_lane_state.sh': 1}, (
            'the lane-state record parser must have exactly ONE definition site '
            f'across the shipped warm-lane scripts; found {hits!r}'
        )

    def test_warm_lane_audit_sources_the_lib(self) -> None:
        text = AUDIT.read_text()
        assert 'source "$SCRIPT_DIR/lib_lane_state.sh"' in text, (
            'warm-lane-audit.sh does not source lib_lane_state.sh — its reader '
            'is still its own copy'
        )

    def test_warm_lane_audit_fails_loud_when_the_lib_is_missing(
        self, tmp_path: Path,
    ) -> None:
        """A missing lib is a DEPLOYMENT fault: exit 2, not a degraded run.

        The audit's "never abort" rule is about lane-level data problems (an
        unreadable record degrades that lane to UNKNOWN).  It is not about
        wiring: nothing about the invocation could have avoided an absent
        sibling and no retry fixes it, so this takes warm-lane-gc.sh's
        established exit-2 shape.  Degrading instead would report every lane as
        UNKNOWN — indistinguishable from a real pool-wide state-dir outage.
        """
        staged_dir = tmp_path / 'incomplete-deploy'
        staged_dir.mkdir()
        staged = staged_dir / 'warm-lane-audit.sh'
        staged.write_bytes(AUDIT.read_bytes())
        staged.chmod(0o755)
        # lib_portable.sh travels (it is a different guard's subject); the
        # lane-state lib deliberately does not.
        (staged_dir / 'lib_portable.sh').write_bytes(
            (WARM_LANE_SCRIPT_DIR / 'lib_portable.sh').read_bytes(),
        )
        proc = subprocess.run(
            [str(staged), '--help'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(staged_dir),
            env=_sanitized_env(),
        )
        assert proc.returncode == 2, (
            'an absent lib_lane_state.sh must be the wiring sentinel exit 2, '
            f'not a degraded run; rc={proc.returncode} stderr={proc.stderr!r}'
        )
        assert 'lib_lane_state.sh not found next to warm-lane-audit.sh' in proc.stderr, (
            f'the fail-loud message must name the missing sibling; {proc.stderr!r}'
        )


# ---------------------------------------------------------------------------
# render_protect_glob: PROTECTED_PREFIXES, rendered for a bash consumer
# ---------------------------------------------------------------------------

class TestRenderProtectGlob:
    """The pure renderer that turns the band registry into a sweep's glob.

    ``PROTECTED_PREFIXES`` cannot be text-scraped from bash: five of its keys
    are computed constants and one more is config-driven, so any sed/awk parse
    would silently UNDER-render — and the failure mode of under-rendering is "a
    live managed worktree is no longer protected from the reaper".  A real
    import is the only faithful read, and this is what it renders.
    """

    def test_a_prefix_key_renders_as_a_glob_and_an_exact_name_verbatim(
        self,
    ) -> None:
        """The two documented key semantics, and nothing invented on top.

        ``git_ops.py``'s registry comment defines them: a key ending in ``-`` is
        a PREFIX (matched with ``str.startswith``); a key NOT ending in ``-`` is
        an EXACT worktree name (matched with ``==``).  Rendering an exact name
        as ``<name>*`` would silently widen the protected set — and rendering a
        prefix verbatim would silently narrow it, which is the direction that
        gets a live worktree reclaimed.
        """
        from orchestrator.git_ops import render_protect_glob

        rendered = render_protect_glob(
            {'_merge-': 'merge-queue', '_merge-verify': 'persistent-merge-verify'},
        )
        assert rendered == '_merge-*,_merge-verify', rendered

    def test_owned_bands_are_excluded(self) -> None:
        from orchestrator.git_ops import render_protect_glob

        rendered = render_protect_glob(
            {'_lane-': 'warm-lane-pool', '_merge-': 'merge-queue'},
            owned=('_lane-',),
        )
        assert rendered == '_merge-*', rendered

    def test_the_pool_bands_are_excluded_under_the_owned_pool_constant(
        self,
    ) -> None:
        """LOAD-BEARING, not cosmetic.

        A naive render would hand ``warm-lane-gc.sh`` ``_lane-*,_spec-*`` as
        PROTECTED.  gc would then skip every pool lane in both passes, reclaim
        would stop entirely, and the pool would accrete straight back to the
        2026-07-10 ENOSPC outage the sweep exists to prevent.
        """
        from orchestrator.git_ops import (
            PROTECT_GLOB_OWNED_POOL_BANDS,
            render_protect_glob,
        )

        rendered = render_protect_glob(owned=PROTECT_GLOB_OWNED_POOL_BANDS)
        bands = rendered.split(',')
        assert '_lane-*' not in bands, rendered
        assert '_spec-*' not in bands, rendered
        assert PROTECT_GLOB_OWNED_POOL_BANDS == frozenset({'_lane-', '_spec-'}), (
            f'the owned-pool band set changed: {PROTECT_GLOB_OWNED_POOL_BANDS!r}'
        )

    def test_output_is_comma_joined_in_registry_order_and_stable(self) -> None:
        from orchestrator.git_ops import PROTECTED_PREFIXES, render_protect_glob

        first = render_protect_glob()
        assert first == render_protect_glob(), 'render is not deterministic'
        assert ' ' not in first, f'the glob must carry no spaces: {first!r}'

        # Registry order, not sorted order — a bash consumer splits on comma and
        # the order is what an operator diffs against the previous default.
        registry_order = [
            key if not key.endswith('-') else f'{key}*' for key in PROTECTED_PREFIXES
        ]
        rendered = first.split(',')
        assert rendered[: len(registry_order)] == registry_order, (
            f'rendered order {rendered!r} does not follow PROTECTED_PREFIXES '
            f'order {registry_order!r}'
        )

    def test_no_mapping_renders_every_registry_key_plus_the_default_iact_band(
        self,
    ) -> None:
        """``None`` means the DEFAULT band map, iact band included.

        A module constant alone cannot capture the authoritative map — the iact
        band is config-shaped — so the renderer's default has to come from
        ``default_protected_prefixes()`` rather than ``PROTECTED_PREFIXES``.
        """
        from orchestrator.git_ops import (
            PROTECTED_PREFIXES,
            default_protected_prefixes,
            render_protect_glob,
        )

        rendered = set(render_protect_glob().split(','))
        for key in PROTECTED_PREFIXES:
            expected = f'{key}*' if key.endswith('-') else key
            assert expected in rendered, f'{key!r} is missing from {rendered!r}'
        assert '_iact-*' in rendered, (
            f'the default interactive band is missing from {rendered!r}'
        )
        assert default_protected_prefixes()['_iact-'] == 'interactive'
        assert set(default_protected_prefixes()) == set(PROTECTED_PREFIXES) | {
            '_iact-',
        }

    def test_default_protected_prefixes_does_not_mutate_the_registry(self) -> None:
        from orchestrator.git_ops import PROTECTED_PREFIXES, default_protected_prefixes

        before = dict(PROTECTED_PREFIXES)
        merged = default_protected_prefixes()
        merged['_scribble-'] = 'nope'
        assert dict(PROTECTED_PREFIXES) == before, (
            'default_protected_prefixes() returned a view of the module registry'
        )


@pytest.fixture
def git_ops(tmp_path: Path):
    """A bare ``GitOps`` — ``protected_prefixes()`` reads only ``self.config``.

    Deliberately local rather than reusing ``test_git_ops.py``'s fixture, which
    drags in a real initialized repo this contract does not need.
    """
    from orchestrator.config import GitConfig
    from orchestrator.git_ops import GitOps

    return GitOps(GitConfig(), tmp_path)


class TestProtectedPrefixesInstanceViewIsUnchanged:
    """``GitOps.protected_prefixes()`` keeps its per-instance contract.

    It is refactored onto ``default_protected_prefixes()`` so the static-
    registry + iact-band merge exists in ONE place, but the observable answer —
    including a per-deployment ``iact_prefix`` override winning — must not move.
    """

    def test_returns_the_registry_merged_with_this_instances_iact_band(
        self, git_ops,
    ) -> None:
        from orchestrator.git_ops import PROTECTED_PREFIXES

        prefixes = git_ops.protected_prefixes()
        for key, owner in PROTECTED_PREFIXES.items():
            assert prefixes[key] == owner, key
        assert prefixes[git_ops.config.iact_prefix] == 'interactive'

    def test_an_overridden_iact_prefix_still_wins(self, git_ops) -> None:
        from orchestrator.git_ops import default_protected_prefixes

        git_ops.config.iact_prefix = '_custom-iact-'
        prefixes = git_ops.protected_prefixes()
        assert prefixes['_custom-iact-'] == 'interactive', prefixes
        assert '_iact-' not in prefixes or prefixes.get('_iact-') != 'interactive', (
            'the DEFAULT iact band leaked into an instance that overrode it — '
            f'{prefixes!r}'
        )
        # The default map is unaffected by the instance override.
        assert default_protected_prefixes()['_iact-'] == 'interactive'


# ---------------------------------------------------------------------------
# lane_protect_glob — the bash -> python bridge
# ---------------------------------------------------------------------------

#: `lane_protect_glob` pays a python3 start plus the `orchestrator.git_ops`
#: import chain.  Measured on this host under fleet load: ~2.4s warm, ~10s cold.
#: The ceiling is deliberately far above both — a timeout here would be a flaky
#: failure indistinguishable from the real bridge break the test exists to catch.
_BRIDGE_TIMEOUT = 120


def _stub_path_dir(tmp_path: Path, name: str, *, body: str | None = None) -> Path:
    """A directory fit to be the WHOLE ``PATH``, holding at most one executable.

    With *body* ``None`` the directory is EMPTY, which is how "``python3`` is
    absent" is expressed — the systemd-timer reading of this failure, where the
    unit's ``PATH`` need not carry the interpreter the interactive shell has.
    """
    stub = tmp_path / f'stub-bin-{name}'
    stub.mkdir(parents=True, exist_ok=True)
    if body is not None:
        exe = stub / name
        exe.write_text(body)
        exe.chmod(0o755)
    return stub


def _warn_lines(stderr: str) -> list[str]:
    """The ``[warn]``-prefixed lines, ignoring any other stderr noise.

    Bash's own ``python3: command not found`` is not part of the contract — the
    library's ONE attributable warning is.
    """
    return [line for line in stderr.splitlines() if line.lstrip().startswith('[warn]')]


class TestLaneProtectGlobBridge:
    """``lane_protect_glob`` is the "readable from bash" half of this leaf.

    ``PROTECTED_PREFIXES`` cannot be faithfully text-scraped — five of its keys
    are computed constants and the ``_iact-`` band is config-driven — so the
    bridge shells a real import.  These cases run it end to end against the real
    shipped repo layout, from a HOSTILE cwd and with the resolution-hostile env
    keys stripped, so a pass cannot be an artifact of the ambient checkout.
    """

    def test_owned_pool_bands_render_exactly_as_python_computes_them(
        self, tmp_path: Path,
    ) -> None:
        """The contract the whole leaf exists for: bash sees what python sees."""
        from orchestrator.git_ops import (
            PROTECT_GLOB_OWNED_POOL_BANDS,
            render_protect_glob,
        )

        expected = render_protect_glob(owned=PROTECT_GLOB_OWNED_POOL_BANDS)
        proc = _run_sourced(
            'lane_protect_glob _lane- _spec-', cwd=tmp_path, timeout=_BRIDGE_TIMEOUT,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == expected, (
            f'bash rendered {proc.stdout.strip()!r}, python renders {expected!r}'
        )
        assert _warn_lines(proc.stderr) == [], proc.stderr
        # The load-bearing exclusion: handing a pool sweep its own bands as
        # PROTECTED would stop reclaim outright (2026-07-10 ENOSPC).
        rendered = set(proc.stdout.strip().split(','))
        assert '_lane-*' not in rendered and '_spec-*' not in rendered, rendered

    def test_with_no_arguments_it_renders_the_full_set(self, tmp_path: Path) -> None:
        """No ``owned`` means nothing is excluded — including the pool bands."""
        from orchestrator.git_ops import render_protect_glob

        proc = _run_sourced(
            'lane_protect_glob', cwd=tmp_path, timeout=_BRIDGE_TIMEOUT,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == render_protect_glob()
        rendered = set(proc.stdout.strip().split(','))
        assert {'_lane-*', '_spec-*'} <= rendered, rendered

    @pytest.mark.parametrize(
        ('label', 'body'),
        [
            ('python3-absent', None),
            ('python3-exits-nonzero', '#!/bin/sh\nexit 1\n'),
            ('python3-prints-nothing', '#!/bin/sh\nexit 0\n'),
        ],
    )
    def test_a_broken_bridge_fails_loud_and_never_answers_empty(
        self, tmp_path: Path, label: str, body: str | None,
    ) -> None:
        """Fail-open, but NEVER silently.

        An empty stdout with a zero exit would read downstream as "nothing is
        protected" — the exact silent widening of a reclaim sweep this leaf
        exists to close.  So a broken bridge prints nothing, returns non-zero
        (which aborts the ``set -euo pipefail`` callers rather than letting them
        proceed unprotected) and leaves exactly ONE attributable ``[warn]``
        line, the same warn-loud/fail-open stance ``_live_refs_env_probe``
        already established in this directory.
        """
        stub = _stub_path_dir(tmp_path, 'python3', body=body)
        proc = _run_sourced(
            'lane_protect_glob _lane- _spec-',
            cwd=tmp_path,
            env=_sanitized_env(extra={'PATH': str(stub)}),
            timeout=_BRIDGE_TIMEOUT,
        )
        assert proc.returncode != 0, (
            f'{label}: a broken bridge returned 0 — a caller checking only the '
            f'exit status would treat {proc.stdout!r} as authoritative'
        )
        assert proc.stdout.strip() == '', f'{label}: {proc.stdout!r}'
        warns = _warn_lines(proc.stderr)
        assert len(warns) == 1, f'{label}: expected exactly one [warn], got {warns!r}'
        assert 'python3' in warns[0], f'{label}: warning does not name the stage: {warns[0]!r}'
