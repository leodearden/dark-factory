"""Tests for orchestrator.substrate_gate module.

Covers: extract_probe_set, build_checker_argv, run_substrate_recheck verdict
mapping, and (in later steps) the harness-level _run_substrate_gate /
_block_and_escalate_substrate_flip integration.

PRD: prd-gate-exec D4 — dispatch-time substrate re-diff.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared scaffolding (pre-1)
# ---------------------------------------------------------------------------

def fake_checker(rc: int, stdout: str = '', stderr: str = ''):
    """Return a run_subprocess-shaped callable that records invocations.

    The returned callable matches the signature used by substrate_gate:
        (argv, *, cwd, timeout) -> (rc, stdout, stderr)

    The ``calls`` attribute on the returned callable accumulates each
    (argv, cwd) pair so tests can assert what was passed.
    """
    calls: list[tuple[list[str], Any]] = []

    def _run(argv, *, cwd, timeout):
        calls.append((argv, cwd))
        return (rc, stdout, stderr)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def make_probe_task(
    checker: list[str] | None = None,
    probe_set: str | None = None,
    extra_meta: dict | None = None,
    *,
    metadata_as_json_string: bool = False,
) -> dict:
    """Return a task dict carrying ``metadata.substrate_probe``.

    Args:
        checker: argv prefix for the checker command.  Defaults to
            ``['python', '-m', 'checker']``.
        probe_set: committed probe-set path.  Defaults to
            ``'probes/my_probes.json'``.
        extra_meta: additional keys merged into ``metadata``.
        metadata_as_json_string: when True the ``metadata`` value is a
            JSON-encoded string (exercising normalization code paths that
            mirror Scheduler._normalize_task_metadata).
    """
    if checker is None:
        checker = ['python', '-m', 'checker']
    if probe_set is None:
        probe_set = 'probes/my_probes.json'

    substrate_probe = {'probe_set': probe_set, 'checker': checker}
    meta: dict = {'substrate_probe': substrate_probe}
    if extra_meta:
        meta.update(extra_meta)

    raw_meta: Any = json.dumps(meta) if metadata_as_json_string else meta
    return {
        'id': '42',
        'title': 'Test task',
        'status': 'pending',
        'metadata': raw_meta,
    }


# ---------------------------------------------------------------------------
# Step-1 RED: extract_probe_set and build_checker_argv
# ---------------------------------------------------------------------------


class TestExtractProbeSet:
    """Tests for ``orchestrator.substrate_gate.extract_probe_set``."""

    def test_returns_none_when_no_metadata(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'title': 'Plain task'}
        assert extract_probe_set(task) is None

    def test_returns_none_when_metadata_is_none(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': None}
        assert extract_probe_set(task) is None

    def test_returns_none_when_metadata_empty_dict(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': {}}
        assert extract_probe_set(task) is None

    def test_returns_none_when_no_substrate_probe_key(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': {'other_key': 'value'}}
        assert extract_probe_set(task) is None

    def test_returns_none_when_substrate_probe_missing_probe_set(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': {'substrate_probe': {'checker': ['py']}}}
        assert extract_probe_set(task) is None

    def test_returns_none_when_substrate_probe_is_not_dict(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': {'substrate_probe': 'not-a-dict'}}
        assert extract_probe_set(task) is None

    def test_returns_none_when_substrate_probe_probe_set_is_empty_string(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': {'substrate_probe': {'probe_set': '', 'checker': ['py']}}}
        assert extract_probe_set(task) is None

    def test_returns_descriptor_when_present(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = make_probe_task(probe_set='probes/foo.json')
        result = extract_probe_set(task)
        assert result is not None
        assert result['probe_set'] == 'probes/foo.json'
        assert result['checker'] == ['python', '-m', 'checker']

    def test_returns_descriptor_when_metadata_is_json_string(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = make_probe_task(probe_set='probes/bar.json', metadata_as_json_string=True)
        result = extract_probe_set(task)
        assert result is not None
        assert result['probe_set'] == 'probes/bar.json'

    def test_returns_none_when_metadata_is_malformed_json_string(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': '{not valid json'}
        assert extract_probe_set(task) is None

    def test_returns_none_when_metadata_json_string_decodes_to_non_dict(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': json.dumps([1, 2, 3])}
        assert extract_probe_set(task) is None

    def test_returns_none_when_metadata_is_integer(self):
        from orchestrator.substrate_gate import extract_probe_set
        task = {'id': '1', 'metadata': 42}
        assert extract_probe_set(task) is None


class TestBuildCheckerArgv:
    """Tests for ``orchestrator.substrate_gate.build_checker_argv``."""

    def test_basic_argv_concatenation(self):
        from orchestrator.substrate_gate import build_checker_argv
        descriptor = {'probe_set': 'probes/foo.json', 'checker': ['python', '-m', 'checker']}
        result = build_checker_argv(descriptor)
        assert result == ['python', '-m', 'checker', 'probes/foo.json']

    def test_single_element_checker(self):
        from orchestrator.substrate_gate import build_checker_argv
        descriptor = {'probe_set': 'probes/bar.json', 'checker': ['./run_check.sh']}
        result = build_checker_argv(descriptor)
        assert result == ['./run_check.sh', 'probes/bar.json']

    def test_placeholder_substitution_in_checker(self):
        """Checker template items containing ``{probe_set}`` are substituted."""
        from orchestrator.substrate_gate import build_checker_argv
        descriptor = {
            'probe_set': 'probes/baz.json',
            'checker': ['python', '-m', 'checker', '--probes={probe_set}'],
        }
        result = build_checker_argv(descriptor)
        assert result == ['python', '-m', 'checker', '--probes=probes/baz.json']

    def test_returns_none_when_no_checker_key(self):
        from orchestrator.substrate_gate import build_checker_argv
        descriptor = {'probe_set': 'probes/foo.json'}
        assert build_checker_argv(descriptor) is None

    def test_returns_none_when_checker_is_empty_list(self):
        from orchestrator.substrate_gate import build_checker_argv
        descriptor = {'probe_set': 'probes/foo.json', 'checker': []}
        assert build_checker_argv(descriptor) is None

    def test_returns_none_when_checker_is_none(self):
        from orchestrator.substrate_gate import build_checker_argv
        descriptor = {'probe_set': 'probes/foo.json', 'checker': None}
        assert build_checker_argv(descriptor) is None

    def test_returns_none_when_probe_set_missing(self):
        from orchestrator.substrate_gate import build_checker_argv
        descriptor = {'checker': ['python', '-m', 'checker']}
        assert build_checker_argv(descriptor) is None

    def test_returns_none_when_descriptor_is_none(self):
        from orchestrator.substrate_gate import build_checker_argv
        assert build_checker_argv(None) is None

    def test_returns_none_when_descriptor_not_dict(self):
        from orchestrator.substrate_gate import build_checker_argv
        assert build_checker_argv('not-a-dict') is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step-3 RED: run_substrate_recheck exit-code→verdict mapping
# ---------------------------------------------------------------------------


class TestRunSubstrateRecheck:
    """Tests for ``orchestrator.substrate_gate.run_substrate_recheck``."""

    def test_rc0_returns_pass_not_flipped(self):
        """rc=0: all probes pass → PASS, not flipped."""
        from orchestrator.substrate_gate import PASS, run_substrate_recheck
        task = make_probe_task()
        checker = fake_checker(rc=0, stdout='all good', stderr='')
        verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == PASS
        assert verdict.flipped is False
        assert verdict.exit_code == 0

    def test_rc1_returns_flip(self):
        """rc=1: ≥1 FAIL → FLIP."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = make_probe_task()
        checker = fake_checker(rc=1, stderr='probe failed')
        verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == FLIP
        assert verdict.flipped is True
        assert verdict.exit_code == 1
        assert 'FAIL' in verdict.reason

    def test_rc2_returns_flip_unprovable(self):
        """rc=2: ≥1 UNPROVABLE → FLIP with UNPROVABLE reason."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = make_probe_task()
        checker = fake_checker(rc=2)
        verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == FLIP
        assert verdict.flipped is True
        assert verdict.exit_code == 2
        assert 'UNPROVABLE' in verdict.reason

    def test_rc127_returns_flip_unverifiable(self):
        """rc=127 (command not found) → FLIP with unverifiable reason."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = make_probe_task()
        checker = fake_checker(rc=127, stderr='command not found')
        verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == FLIP
        assert verdict.flipped is True
        assert verdict.exit_code == 127
        assert 'rc=127' in verdict.reason or 'unverifiable' in verdict.reason

    def test_other_nonzero_rc_returns_flip_unverifiable(self):
        """Any other non-zero rc → FLIP with 'unverifiable' reason."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = make_probe_task()
        checker = fake_checker(rc=255)
        verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == FLIP
        assert verdict.flipped is True
        assert verdict.exit_code == 255
        assert 'unverifiable' in verdict.reason.lower() or 'rc=255' in verdict.reason

    def test_no_descriptor_returns_skip(self):
        """Task with no substrate_probe → SKIP, not flipped (gate no-op)."""
        from orchestrator.substrate_gate import SKIP, run_substrate_recheck
        task = {'id': '1', 'title': 'plain', 'metadata': {}}
        checker = fake_checker(rc=0)
        verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == SKIP
        assert verdict.flipped is False
        # Checker must NOT have been invoked for a task with no descriptor
        assert checker.calls == []

    def test_descriptor_but_no_checker_returns_flip(self):
        """Descriptor present but no resolvable checker → FLIP."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = {'id': '1', 'metadata': {'substrate_probe': {'probe_set': 'probes/foo.json', 'checker': []}}}
        checker = fake_checker(rc=0)
        verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == FLIP
        assert verdict.flipped is True
        assert 'no checker command' in verdict.reason or 'probe set declared' in verdict.reason

    def test_checker_called_with_correct_argv_and_cwd(self):
        """Checker is invoked with the built argv and cwd=worktree."""
        from orchestrator.substrate_gate import run_substrate_recheck
        task = make_probe_task(checker=['run_check'], probe_set='probes/suite.json')
        checker = fake_checker(rc=0)
        run_substrate_recheck(task=task, worktree='/wt/gate', run_subprocess=checker)
        assert len(checker.calls) == 1
        called_argv, called_cwd = checker.calls[0]
        assert called_argv == ['run_check', 'probes/suite.json']
        assert called_cwd == '/wt/gate'

    def test_prd_s10_two_way_boundary_pass(self):
        """PRD §10 boundary: same descriptor, rc=0 → PASS (author-time PASS holds)."""
        from orchestrator.substrate_gate import PASS, run_substrate_recheck
        task = make_probe_task()
        verdict = run_substrate_recheck(
            task=task, worktree='/gate', run_subprocess=fake_checker(rc=0)
        )
        assert verdict.verdict == PASS
        assert not verdict.flipped

    def test_prd_s10_two_way_boundary_flip(self):
        """PRD §10 boundary: same descriptor, rc=1 → FLIP (4352 drift case)."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = make_probe_task()
        verdict = run_substrate_recheck(
            task=task, worktree='/gate', run_subprocess=fake_checker(rc=1)
        )
        assert verdict.verdict == FLIP
        assert verdict.flipped

    def test_stdout_stderr_captured_in_verdict(self):
        """Stdout/stderr from checker are captured in the verdict."""
        from orchestrator.substrate_gate import run_substrate_recheck
        task = make_probe_task()
        checker = fake_checker(rc=1, stdout='some output', stderr='some error')
        verdict = run_substrate_recheck(task=task, worktree='/wt', run_subprocess=checker)
        assert verdict.stdout == 'some output'
        assert verdict.stderr == 'some error'

    def test_metadata_as_json_string_works(self):
        """Task with metadata as JSON string is handled (normalization)."""
        from orchestrator.substrate_gate import PASS, run_substrate_recheck
        task = make_probe_task(metadata_as_json_string=True)
        verdict = run_substrate_recheck(
            task=task, worktree='/gate', run_subprocess=fake_checker(rc=0)
        )
        assert verdict.verdict == PASS


# ---------------------------------------------------------------------------
# Step-7 RED: harness _run_substrate_gate + _block_and_escalate_substrate_flip
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path):
    """Build a bare Harness with mocked internals for substrate-gate tests.

    Patches:
    - McpLifecycle, OverrideStore, Scheduler, BriefingAssembler (construction)
    - harness.git_ops.resolve_branch_sha → returns a fake main SHA
    - harness.git_ops.worktree_base → tmp_path / '.worktrees'
    - harness.scheduler.set_task_status → AsyncMock
    """
    from orchestrator.config import OrchestratorConfig
    from orchestrator.harness import Harness

    config = OrchestratorConfig(project_root=tmp_path, max_per_module=1)
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(config)

    # Wire stub scheduler
    h.scheduler = MagicMock()
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.carries_substrate_probe = MagicMock(return_value=True)

    # Wire stub git_ops
    h.git_ops = MagicMock()
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' * 5)  # 40-char SHA
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.git_ops.project_root = tmp_path

    # No escalation queue by default — tests that need one attach it explicitly
    h._escalation_queue = None

    return h


def _make_assignment(task_id: str = '42', probe: bool = True):
    """Build a minimal TaskAssignment-like object."""
    task = make_probe_task() if probe else {'id': task_id, 'title': 'Plain', 'metadata': {}}
    task['id'] = task_id

    from unittest.mock import MagicMock
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = task
    return assignment


def _pass_verdict():
    from orchestrator.substrate_gate import PASS, SubstrateVerdict
    return SubstrateVerdict(
        verdict=PASS,
        exit_code=0,
        checker_argv=['run_check', 'probes/foo.json'],
        probe_set='probes/foo.json',
        reason='all probes PASS',
    )


def _flip_verdict():
    from orchestrator.substrate_gate import FLIP, SubstrateVerdict
    return SubstrateVerdict(
        verdict=FLIP,
        exit_code=1,
        checker_argv=['run_check', 'probes/foo.json'],
        probe_set='probes/foo.json',
        reason='PASS→FAIL flip detected',
    )


def _skip_verdict():
    from orchestrator.substrate_gate import SKIP, SubstrateVerdict
    return SubstrateVerdict(
        verdict=SKIP,
        exit_code=None,
        checker_argv=None,
        probe_set=None,
        reason='no descriptor',
    )


class TestRunSubstrateGate:
    """Unit tests for ``Harness._run_substrate_gate``."""

    @pytest.mark.asyncio
    async def test_returns_true_on_pass(self, tmp_path: Path, monkeypatch):
        """PASS verdict → gate returns True (dispatch allowed)."""
        h = _make_harness(tmp_path)
        assignment = _make_assignment()

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )
        # Patch asyncio subprocess calls for worktree add/remove
        with patch('asyncio.create_subprocess_exec', new=AsyncMock(
            return_value=_fake_proc(0)
        )):
            result = await h._run_substrate_gate(assignment)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_on_skip(self, tmp_path: Path, monkeypatch):
        """SKIP verdict (no descriptor) → gate returns True (dispatch allowed)."""
        h = _make_harness(tmp_path)
        assignment = _make_assignment(probe=False)

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _skip_verdict(),
        )
        with patch('asyncio.create_subprocess_exec', new=AsyncMock(
            return_value=_fake_proc(0)
        )):
            result = await h._run_substrate_gate(assignment)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_flip_and_calls_escalate(self, tmp_path: Path, monkeypatch):
        """FLIP verdict → gate returns False AND invokes _block_and_escalate_substrate_flip."""
        h = _make_harness(tmp_path)
        assignment = _make_assignment()
        escalate_calls: list = []

        async def _fake_escalate(task_id, *, verdict):
            escalate_calls.append((task_id, verdict))

        h._block_and_escalate_substrate_flip = _fake_escalate

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _flip_verdict(),
        )
        with patch('asyncio.create_subprocess_exec', new=AsyncMock(
            return_value=_fake_proc(0)
        )):
            result = await h._run_substrate_gate(assignment)

        assert result is False
        assert len(escalate_calls) == 1
        assert escalate_calls[0][0] == '42'

    @pytest.mark.asyncio
    async def test_gate_worktree_torn_down_in_finally(self, tmp_path: Path, monkeypatch):
        """Gate worktree is removed in finally even when run_substrate_recheck raises."""
        h = _make_harness(tmp_path)
        assignment = _make_assignment()
        remove_calls: list = []

        def _track_create_subprocess_exec(*args, **kwargs):
            # Track 'git worktree remove' calls to verify cleanup
            if 'remove' in args:
                remove_calls.append(args)
            return _fake_proc(0)

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            MagicMock(side_effect=RuntimeError('checker crashed')),
        )

        proc_mock = AsyncMock(return_value=_fake_proc(0))

        with patch('asyncio.create_subprocess_exec', proc_mock), contextlib.suppress(Exception):
            # Should not propagate the error (gate catches it and returns False or raises)
            await h._run_substrate_gate(assignment)

        # Regardless of how _run_substrate_gate handles the exception,
        # verify that 'git worktree remove' was called (finally ran)
        all_calls = [list(c.args) for c in proc_mock.call_args_list]
        remove_seen = any('remove' in str(call) for call in all_calls)
        assert remove_seen, (
            f'Expected git worktree remove to be called in finally; calls={all_calls!r}'
        )


def _fake_proc(returncode: int):
    """Return a minimal asyncio.Process mock."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(b'', b''))
    proc.wait = AsyncMock(return_value=returncode)
    return proc


class TestBlockAndEscalateSubstrateFlip:
    """Unit tests for ``Harness._block_and_escalate_substrate_flip``."""

    @pytest.mark.asyncio
    async def test_sets_task_blocked(self, tmp_path: Path):
        """set_task_status('blocked') is called with the task_id."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')

        verdict = _flip_verdict()
        await h._block_and_escalate_substrate_flip('99', verdict=verdict)

        h.scheduler.set_task_status.assert_awaited_once_with('99', 'blocked')  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_files_l1_escalation_design_concern(self, tmp_path: Path):
        """Files exactly one L1 with category='design_concern' and severity='blocking'."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        verdict = _flip_verdict()
        await h._block_and_escalate_substrate_flip('77', verdict=verdict)

        pending = esc_queue.get_pending()
        l1s = [e for e in pending if e.task_id == '77' and e.level == 1]
        assert len(l1s) == 1, f'Expected exactly 1 L1 for task 77; got {l1s!r}'
        esc = l1s[0]
        assert esc.category == 'design_concern'
        assert esc.severity == 'blocking'

    @pytest.mark.asyncio
    async def test_deduped_by_has_open_l1(self, tmp_path: Path):
        """Second call with open L1 is suppressed (no duplicate filed)."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        verdict = _flip_verdict()
        await h._block_and_escalate_substrate_flip('55', verdict=verdict)
        await h._block_and_escalate_substrate_flip('55', verdict=verdict)

        pending = esc_queue.get_pending()
        l1s = [e for e in pending if e.task_id == '55' and e.level == 1]
        assert len(l1s) == 1, f'Expected exactly 1 L1 after dedup; got {l1s!r}'

    @pytest.mark.asyncio
    async def test_noop_when_no_escalation_queue(self, tmp_path: Path):
        """No-ops gracefully when _escalation_queue is None (bare-harness tests)."""
        h = _make_harness(tmp_path)
        h._escalation_queue = None  # bare-harness scenario

        verdict = _flip_verdict()
        # Should not raise
        await h._block_and_escalate_substrate_flip('33', verdict=verdict)

        # set_task_status is still called (blocking the task is unconditional)
        h.scheduler.set_task_status.assert_awaited_once_with('33', 'blocked')  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Step-9 RED: dispatch-level wiring through _run_slot (PRD §10 two-way boundary)
# ---------------------------------------------------------------------------


def _make_slot_harness(tmp_path: Path):
    """Build a Harness whose _run_slot can be called directly in tests.

    Extends _make_harness with:
    - scheduler.release = MagicMock (non-async, mirroring production)
    - scheduler._dispatched = set()
    - _run_id = None (so _apply_retry_cap fast-returns)
    - event_store = None (so event emit branches are skipped)
    """
    h = _make_harness(tmp_path)
    h.scheduler.release = MagicMock()
    h.scheduler._dispatched = set()
    h._run_id = None
    h.event_store = None
    return h


def _make_mock_workflow():
    """Return a minimal AsyncMock workflow whose run() is awaitable."""
    from orchestrator.workflow import WorkflowOutcome
    wf = AsyncMock()
    wf.run = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf.metrics = MagicMock(
        total_cost_usd=0.0,
        total_duration_ms=0,
        agent_invocations=0,
        execute_iterations=0,
        verify_attempts=0,
        review_cycles=0,
    )
    wf._steward = None
    wf._last_block_reason = None
    wf._last_block_detail = None
    wf._last_block_phase = None
    return wf


class TestRunSlotSubstrateGateWiring:
    """PRD §10 two-way boundary: _run_slot wiring for the substrate gate.

    Tests verify the gate integration in _run_slot BEFORE TaskWorkflow construction:
    (a) FLIP → TaskWorkflow never constructed / workflow.run never awaited, cooldown armed.
    (b) PASS → TaskWorkflow IS constructed and workflow.run IS awaited.
    (c) NO-PROBE → gate never invoked (carries_substrate_probe=False), workflow proceeds.
    """

    @pytest.mark.asyncio
    async def test_flip_blocks_workflow_construction(self, tmp_path: Path):
        """(a) FLIP: TaskWorkflow must NOT be constructed when gate returns False."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(probe=True)
        h.scheduler.carries_substrate_probe = MagicMock(return_value=True)
        h._run_substrate_gate = AsyncMock(return_value=False)
        # _block_and_escalate_substrate_flip is called inside _run_substrate_gate (already mocked);
        # stub it here too so the gate-returning-False path is clean.
        h._block_and_escalate_substrate_flip = AsyncMock()

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
            MockWorkflow.return_value = _make_mock_workflow()
            await h._run_slot(assignment, sem)

        # (a1) TaskWorkflow must NOT have been constructed.
        assert not MockWorkflow.called, (
            'TaskWorkflow must NOT be constructed on a substrate flip — '
            'agent must not spin up'
        )

        # (a2) _run_substrate_gate must have been awaited.
        h._run_substrate_gate.assert_awaited_once()

        # (a3) Scheduler.release must be called with requeued=True (cooldown armed).
        h.scheduler.release.assert_called_once()  # type: ignore[attr-defined]
        call_kwargs = h.scheduler.release.call_args  # type: ignore[attr-defined]
        requeued = call_kwargs.kwargs.get('requeued', None)
        assert requeued is True, (
            f'scheduler.release must be called with requeued=True on flip; got {call_kwargs!r}'
        )

    @pytest.mark.asyncio
    async def test_pass_allows_workflow_construction(self, tmp_path: Path):
        """(b) PASS: TaskWorkflow MUST be constructed and workflow.run awaited."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(probe=True)
        h.scheduler.carries_substrate_probe = MagicMock(return_value=True)
        h._run_substrate_gate = AsyncMock(return_value=True)

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
            mock_wf = _make_mock_workflow()
            MockWorkflow.return_value = mock_wf
            await h._run_slot(assignment, sem)

        # (b1) TaskWorkflow must have been constructed.
        assert MockWorkflow.called, 'TaskWorkflow must be constructed on PASS gate verdict'

        # (b2) workflow.run must have been awaited.
        mock_wf.run.assert_awaited_once()

        # (b3) _run_substrate_gate must have been awaited.
        h._run_substrate_gate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_probe_skips_gate(self, tmp_path: Path):
        """(c) NO-PROBE: gate must NOT be invoked when carries_substrate_probe is False."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(probe=False)
        h.scheduler.carries_substrate_probe = MagicMock(return_value=False)
        gate_mock = AsyncMock(return_value=True)
        h._run_substrate_gate = gate_mock

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
            mock_wf = _make_mock_workflow()
            MockWorkflow.return_value = mock_wf
            await h._run_slot(assignment, sem)

        # (c1) _run_substrate_gate must NOT have been called.
        gate_mock.assert_not_awaited()

        # (c2) Workflow still runs (non-probe tasks unaffected).
        mock_wf.run.assert_awaited_once()
