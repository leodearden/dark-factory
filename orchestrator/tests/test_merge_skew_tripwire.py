"""Tests for orchestrator.merge_skew_tripwire (task 2382, merge-skew δ).

M3 of plans/merge-skew-attribution-prd.md — the proactive pipeline-landing
tripwire (PRD task δ, invariant I6, boundary rows 5-6): on each successful
merge landing, if the landing's changed files trip a project-configured
load-bearing oracle, emit exactly ONE advisory info escalation naming the
landing sha and the in-flight tasks whose branch diffs overlap the landing's
changed set, and attach a steward-visible note to those tasks' metadata.

Each test class imports the module under test LOCALLY inside its test
methods (not at module scope) so a not-yet-implemented symbol never breaks
collection of the rest of this file during earlier RED steps — mirrors
test_merge_queue_lifecycle_registry.py / test_merge_queue_request_liveness.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest


class TestLoadBearingOracle:
    """Unit tests for _run_load_bearing_oracle(project_root, oracle_cmd, changed_files).

    Mirrors TestVerifyPipelineGuard (test_verify.py) — real executable bash
    scripts written into tmp_path so the real git_ops._run subprocess
    executes them; no subprocess mock.
    """

    def _write_oracle_script(
        self, project_root: Path, script_content: str, *, executable: bool = True,
    ) -> Path:
        script = project_root / 'oracle.sh'
        script.write_text(script_content)
        if executable:
            script.chmod(0o755)
        return script

    @pytest.mark.asyncio
    async def test_script_exits_0_returns_true_and_receives_changed_files(
        self, tmp_path: Path,
    ) -> None:
        """Script present, exits 0 → True; receives the trailing changed-file args."""
        from orchestrator.merge_skew_tripwire import _run_load_bearing_oracle

        sentinel = tmp_path / 'received_args.txt'
        script = self._write_oracle_script(tmp_path, f"""\
#!/usr/bin/env bash
echo "$@" > {sentinel}
exit 0
""")

        result = await _run_load_bearing_oracle(
            tmp_path, ['bash', str(script)], ['src/a.py', 'src/b.py'],
        )

        assert result is True
        received = sentinel.read_text().strip()
        assert received == 'src/a.py src/b.py', (
            f'Oracle script received unexpected args: {received!r}'
        )

    @pytest.mark.asyncio
    async def test_script_exits_nonzero_returns_false(self, tmp_path: Path) -> None:
        """Script present but exits non-zero → False (not load-bearing)."""
        from orchestrator.merge_skew_tripwire import _run_load_bearing_oracle

        script = self._write_oracle_script(tmp_path, """\
#!/usr/bin/env bash
exit 1
""")

        result = await _run_load_bearing_oracle(
            tmp_path, ['bash', str(script)], ['src/a.py'],
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_oracle_cmd_returns_false_without_spawning(
        self, tmp_path: Path,
    ) -> None:
        """oracle_cmd == [] → False without consulting any script (fail-open)."""
        from orchestrator.merge_skew_tripwire import _run_load_bearing_oracle

        result = await _run_load_bearing_oracle(tmp_path, [], ['src/a.py'])
        assert result is False

    @pytest.mark.asyncio
    async def test_none_oracle_cmd_returns_false_without_spawning(
        self, tmp_path: Path,
    ) -> None:
        """oracle_cmd is None → False without consulting any script (fail-open)."""
        from orchestrator.merge_skew_tripwire import _run_load_bearing_oracle

        result = await _run_load_bearing_oracle(tmp_path, None, ['src/a.py'])
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_changed_files_returns_false_without_spawning(
        self, tmp_path: Path,
    ) -> None:
        """changed_files == [] → False without spawning (fail-open)."""
        from orchestrator.merge_skew_tripwire import _run_load_bearing_oracle

        # A script that would exit 0 if invoked — must NOT be spawned at all.
        script = self._write_oracle_script(tmp_path, """\
#!/usr/bin/env bash
exit 0
""")

        result = await _run_load_bearing_oracle(tmp_path, ['bash', str(script)], [])
        assert result is False

    @pytest.mark.asyncio
    async def test_missing_script_returns_false(self, tmp_path: Path) -> None:
        """oracle_cmd names a script that doesn't exist → False (fail-open)."""
        from orchestrator.merge_skew_tripwire import _run_load_bearing_oracle

        missing = tmp_path / 'does-not-exist.sh'
        result = await _run_load_bearing_oracle(
            tmp_path, [str(missing)], ['src/a.py'],
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_non_executable_script_returns_false(self, tmp_path: Path) -> None:
        """Non-executable script → False (fail-open); exception absorbed by broad except."""
        from orchestrator.merge_skew_tripwire import _run_load_bearing_oracle

        script = self._write_oracle_script(tmp_path, """\
#!/usr/bin/env bash
exit 0
""", executable=False)

        result = await _run_load_bearing_oracle(
            tmp_path, [str(script)], ['src/a.py'],
        )
        assert result is False


class TestComputeTripwireOverlap:
    """Unit tests for the pure ``compute_tripwire_overlap(landing_changed_files,
    inflight_diffs)``.

    ``inflight_diffs`` is a list of ``(task_id, branch, branch_changed_files)``
    tuples. Encodes the boundary-row-5 core: an in-flight task whose branch
    diff shares ≥1 file with the landing set is named in the result; a
    non-overlapping task is silently absent (not a zero-overlap hit).
    """

    def test_overlapping_task_yields_hit_with_sorted_overlap_files(self) -> None:
        from orchestrator.merge_skew_tripwire import TripwireHit, compute_tripwire_overlap

        landing_changed_files = ['src/a.py', 'src/b.py', 'src/c.py']
        inflight_diffs = [
            ('101', 'task/101', ['src/c.py', 'src/a.py', 'unrelated.py']),
        ]

        hits = compute_tripwire_overlap(landing_changed_files, inflight_diffs)

        assert hits == [
            TripwireHit(task_id='101', branch='task/101', overlap_files=('src/a.py', 'src/c.py')),
        ]

    def test_non_overlapping_task_yields_no_hit(self) -> None:
        from orchestrator.merge_skew_tripwire import compute_tripwire_overlap

        landing_changed_files = ['src/a.py']
        inflight_diffs = [
            ('202', 'task/202', ['unrelated1.py', 'unrelated2.py']),
        ]

        hits = compute_tripwire_overlap(landing_changed_files, inflight_diffs)

        assert hits == []

    def test_mixed_overlapping_and_non_overlapping(self) -> None:
        from orchestrator.merge_skew_tripwire import TripwireHit, compute_tripwire_overlap

        landing_changed_files = ['src/a.py']
        inflight_diffs = [
            ('101', 'task/101', ['src/a.py']),
            ('202', 'task/202', ['unrelated.py']),
        ]

        hits = compute_tripwire_overlap(landing_changed_files, inflight_diffs)

        assert hits == [
            TripwireHit(task_id='101', branch='task/101', overlap_files=('src/a.py',)),
        ]
        assert not any(h.task_id == '202' for h in hits), (
            f'non-overlapping task 202 must be absent from hits; got {hits!r}'
        )

    def test_empty_landing_set_yields_no_hits(self) -> None:
        from orchestrator.merge_skew_tripwire import compute_tripwire_overlap

        hits = compute_tripwire_overlap([], [('101', 'task/101', ['src/a.py'])])

        assert hits == []

    def test_empty_inflight_diffs_yields_no_hits(self) -> None:
        from orchestrator.merge_skew_tripwire import compute_tripwire_overlap

        hits = compute_tripwire_overlap(['src/a.py'], [])

        assert hits == []

    def test_ordering_is_deterministic_and_matches_input_order(self) -> None:
        from orchestrator.merge_skew_tripwire import compute_tripwire_overlap

        landing_changed_files = ['src/a.py', 'src/b.py']
        inflight_diffs = [
            ('301', 'task/301', ['src/b.py']),
            ('101', 'task/101', ['src/a.py']),
            ('202', 'task/202', ['src/a.py', 'src/b.py']),
        ]

        hits = compute_tripwire_overlap(landing_changed_files, inflight_diffs)

        assert [h.task_id for h in hits] == ['301', '101', '202'], (
            'hits must preserve inflight_diffs input order, not be resorted'
        )


class _FakeEscalationQueue:
    """Minimal fake escalation queue (per-file duplication convention — see
    test_merge_queue_lifecycle_registry.py:86 / test_multihost_verify_integration.py:639).

    ``by_task`` seeds what ``get_by_task`` returns for a given sentinel
    task_id (defaults to ``[]``, i.e. no pre-existing open escalation — the
    common case in these tests).
    """

    def __init__(self, by_task: dict[str, list] | None = None) -> None:
        self._by_task = by_task or {}
        self._seq = 0
        self.submitted: list[Any] = []
        self.get_by_task_calls: list[tuple[str, str | None]] = []

    def get_by_task(self, task_id: str, status: str | None = None) -> list:
        self.get_by_task_calls.append((task_id, status))
        return self._by_task.get(task_id, [])

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{task_id}-{self._seq}'

    def submit(self, esc: Any) -> None:
        self.submitted.append(esc)


def _write_oracle_script(project_root: Path, *, exit_code: int) -> Path:
    """Write a real executable bash oracle script exiting *exit_code*."""
    script = project_root / 'oracle.sh'
    script.write_text(f"""\
#!/usr/bin/env bash
exit {exit_code}
""")
    script.chmod(0o755)
    return script


@pytest.mark.asyncio
class TestEmitTripwire:
    """Unit tests for ``emit_pipeline_landing_tripwire`` — the orchestration
    entrypoint covering boundary rows 5-6 of
    plans/merge-skew-attribution-prd.md.
    """

    async def test_oracle_negative_no_escalation(self, tmp_path: Path) -> None:
        """Boundary row 6: oracle-negative landing → zero escalations, zero updates."""
        from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire

        script = _write_oracle_script(tmp_path, exit_code=1)  # not load-bearing
        fake_eq = _FakeEscalationQueue()
        get_branch_diff = AsyncMock(return_value=['src/a.py'])
        update_task = AsyncMock(return_value=True)

        await emit_pipeline_landing_tripwire(
            project_root=tmp_path,
            oracle_cmd=['bash', str(script)],
            escalation_queue=fake_eq,
            landing_sha='deadbeef' * 5,
            landing_task_id='999',
            landing_changed_files=['src/a.py'],
            inflight=[('101', 'task/101')],
            get_branch_diff=get_branch_diff,
            update_task=update_task,
        )

        assert fake_eq.submitted == [], (
            f'oracle-negative landing must submit no escalation; got {fake_eq.submitted!r}'
        )
        update_task.assert_not_awaited()
        get_branch_diff.assert_not_awaited()

    async def test_oracle_positive_one_escalation_names_only_overlapping(
        self, tmp_path: Path,
    ) -> None:
        """Boundary row 5: oracle-positive landing with one overlapping and one
        non-overlapping in-flight task → exactly one escalation naming only the
        overlapping task, and a metadata note attached only to that task.
        """
        from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire

        script = _write_oracle_script(tmp_path, exit_code=0)  # load-bearing
        fake_eq = _FakeEscalationQueue()
        update_task = AsyncMock(return_value=True)

        branch_diffs = {
            'task/101': ['src/a.py', 'unrelated_101.py'],  # overlaps landing's src/a.py
            'task/202': ['unrelated_202.py'],  # no overlap
        }

        async def get_branch_diff(branch: str) -> list[str] | None:
            return branch_diffs.get(branch)

        landing_sha = 'a' * 40
        await emit_pipeline_landing_tripwire(
            project_root=tmp_path,
            oracle_cmd=['bash', str(script)],
            escalation_queue=fake_eq,
            landing_sha=landing_sha,
            landing_task_id='999',
            landing_changed_files=['src/a.py'],
            inflight=[('101', 'task/101'), ('202', 'task/202')],
            get_branch_diff=get_branch_diff,
            update_task=update_task,
        )

        assert len(fake_eq.submitted) == 1, (
            f'expected exactly one escalation; got {fake_eq.submitted!r}'
        )
        esc = fake_eq.submitted[0]
        assert esc.severity == 'info'
        assert esc.level == 0
        assert esc.category == 'risk_identified'
        assert landing_sha[:12] in esc.summary or landing_sha[:12] in esc.detail, (
            f'expected landing sha in summary/detail; got summary={esc.summary!r} '
            f'detail={esc.detail!r}'
        )
        assert '101' in esc.summary or '101' in esc.detail, (
            'expected overlapping task_id 101 named in summary/detail'
        )
        assert '202' not in esc.summary and '202' not in esc.detail, (
            'non-overlapping task_id 202 must NOT appear in the escalation'
        )

        update_task.assert_awaited_once()
        call = update_task.await_args
        assert call is not None
        assert call.args[0] == '101', (
            f'update_task must be called for the overlapping task only; got {call.args[0]!r}'
        )
        note = call.args[1]
        assert 'merge_skew_tripwire' in note
        assert note['merge_skew_tripwire']['landing_sha'] == landing_sha
        assert note['merge_skew_tripwire']['overlap_files'] == ['src/a.py']
        assert call.kwargs.get('metadata_mode') == 'merge'

    async def test_empty_or_none_oracle_cmd_never_consults_downstream(
        self, tmp_path: Path,
    ) -> None:
        """oracle_cmd falsy (None or []) → short-circuits before touching
        get_branch_diff/update_task at all — asserted by making both raise
        if they were ever (incorrectly) invoked."""
        from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire

        get_branch_diff = AsyncMock(side_effect=AssertionError('get_branch_diff must not be called'))
        update_task = AsyncMock(side_effect=AssertionError('update_task must not be called'))

        for oracle_cmd in (None, []):
            fake_eq = _FakeEscalationQueue()
            await emit_pipeline_landing_tripwire(
                project_root=tmp_path,
                oracle_cmd=oracle_cmd,
                escalation_queue=fake_eq,
                landing_sha='f' * 40,
                landing_task_id='999',
                landing_changed_files=['src/a.py'],
                inflight=[('101', 'task/101')],
                get_branch_diff=get_branch_diff,
                update_task=update_task,
            )
            assert fake_eq.submitted == [], (
                f'oracle_cmd={oracle_cmd!r} must yield zero escalations'
            )

        get_branch_diff.assert_not_awaited()
        update_task.assert_not_awaited()

    async def test_dedup_skips_submit_when_open_escalation_exists(
        self, tmp_path: Path,
    ) -> None:
        """I6 (≤1 escalation per landing): a pre-existing pending escalation
        for this landing's sentinel task_id suppresses a duplicate submit —
        mirrors _alarm_verify_worktree_contention's dedup guard."""
        from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire

        script = _write_oracle_script(tmp_path, exit_code=0)
        landing_sha = 'b' * 40
        sentinel = f'pipeline-landing-tripwire-{landing_sha[:12]}'
        fake_eq = _FakeEscalationQueue(by_task={sentinel: [object()]})
        update_task = AsyncMock(return_value=True)

        async def get_branch_diff(branch: str) -> list[str] | None:
            return ['src/a.py']

        await emit_pipeline_landing_tripwire(
            project_root=tmp_path,
            oracle_cmd=['bash', str(script)],
            escalation_queue=fake_eq,
            landing_sha=landing_sha,
            landing_task_id='999',
            landing_changed_files=['src/a.py'],
            inflight=[('101', 'task/101')],
            get_branch_diff=get_branch_diff,
            update_task=update_task,
        )

        assert fake_eq.submitted == [], (
            'a pre-existing pending escalation for this landing must suppress a duplicate submit'
        )
        update_task.assert_not_awaited()

    async def test_get_branch_diff_raising_for_one_task_still_processes_others(
        self, tmp_path: Path,
    ) -> None:
        """A raising get_branch_diff for one in-flight task must not prevent
        the remaining in-flight tasks from being processed, and must not
        propagate out of emit (fail-open)."""
        from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire

        script = _write_oracle_script(tmp_path, exit_code=0)
        fake_eq = _FakeEscalationQueue()
        update_task = AsyncMock(return_value=True)

        async def get_branch_diff(branch: str) -> list[str] | None:
            if branch == 'task/101':
                raise RuntimeError('git boom')
            return ['src/a.py']

        # Must not raise.
        await emit_pipeline_landing_tripwire(
            project_root=tmp_path,
            oracle_cmd=['bash', str(script)],
            escalation_queue=fake_eq,
            landing_sha='c' * 40,
            landing_task_id='999',
            landing_changed_files=['src/a.py'],
            inflight=[('101', 'task/101'), ('202', 'task/202')],
            get_branch_diff=get_branch_diff,
            update_task=update_task,
        )

        assert len(fake_eq.submitted) == 1
        esc = fake_eq.submitted[0]
        assert '101' not in esc.summary and '101' not in esc.detail, (
            'the raising task must be skipped entirely, not named in the escalation'
        )
        assert '202' in esc.summary or '202' in esc.detail
        update_task.assert_awaited_once()
        call = update_task.await_args
        assert call is not None
        assert call.args[0] == '202'

    async def test_update_task_raising_is_swallowed_escalation_still_submitted(
        self, tmp_path: Path,
    ) -> None:
        """update_task raising must not propagate — the escalation itself
        (already submitted before update_task runs) is unaffected."""
        from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire

        script = _write_oracle_script(tmp_path, exit_code=0)
        fake_eq = _FakeEscalationQueue()
        update_task = AsyncMock(side_effect=RuntimeError('update boom'))

        async def get_branch_diff(branch: str) -> list[str] | None:
            return ['src/a.py']

        # Must not raise.
        await emit_pipeline_landing_tripwire(
            project_root=tmp_path,
            oracle_cmd=['bash', str(script)],
            escalation_queue=fake_eq,
            landing_sha='d' * 40,
            landing_task_id='999',
            landing_changed_files=['src/a.py'],
            inflight=[('101', 'task/101')],
            get_branch_diff=get_branch_diff,
            update_task=update_task,
        )

        assert len(fake_eq.submitted) == 1, (
            'escalation must still be submitted despite update_task raising'
        )

    async def test_escalation_submit_raising_is_swallowed(self, tmp_path: Path) -> None:
        """escalation_queue.submit raising must not propagate out of emit."""
        from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire

        script = _write_oracle_script(tmp_path, exit_code=0)

        class _RaisingSubmitQueue(_FakeEscalationQueue):
            def submit(self, esc: Any) -> None:
                raise RuntimeError('submit boom')

        fake_eq = _RaisingSubmitQueue()
        update_task = AsyncMock(return_value=True)

        async def get_branch_diff(branch: str) -> list[str] | None:
            return ['src/a.py']

        # Must not raise.
        await emit_pipeline_landing_tripwire(
            project_root=tmp_path,
            oracle_cmd=['bash', str(script)],
            escalation_queue=fake_eq,
            landing_sha='e' * 40,
            landing_task_id='999',
            landing_changed_files=['src/a.py'],
            inflight=[('101', 'task/101')],
            get_branch_diff=get_branch_diff,
            update_task=update_task,
        )
