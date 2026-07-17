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
