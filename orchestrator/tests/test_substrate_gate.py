"""Tests for orchestrator.substrate_gate module.

Covers: extract_probe_set, build_checker_argv, run_substrate_recheck verdict
mapping, and (in later steps) the harness-level _run_substrate_gate /
_block_and_escalate_substrate_flip integration.

PRD: prd-gate-exec D4 — dispatch-time substrate re-diff.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
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
# Step-1 RED: carries_substrate_probe + fail-closed malformed-present cases
# ---------------------------------------------------------------------------


class TestCarriesSubstrateProbe:
    """Tests for ``orchestrator.substrate_gate.carries_substrate_probe``."""

    def test_returns_true_when_substrate_probe_present_as_dict(self):
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'metadata': {'substrate_probe': {'probe_set': 'foo.json', 'checker': ['py']}}}
        assert carries_substrate_probe(task) is True

    def test_returns_true_when_metadata_is_json_string_with_substrate_probe(self):
        from orchestrator.substrate_gate import carries_substrate_probe
        meta = {'substrate_probe': {'probe_set': 'foo.json', 'checker': ['py']}}
        task = {'id': '1', 'metadata': json.dumps(meta)}
        assert carries_substrate_probe(task) is True

    def test_returns_false_when_metadata_absent(self):
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'title': 'Plain task'}
        assert carries_substrate_probe(task) is False

    def test_returns_false_when_metadata_is_none(self):
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'metadata': None}
        assert carries_substrate_probe(task) is False

    def test_returns_false_when_metadata_not_dict_and_not_str(self):
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'metadata': 42}
        assert carries_substrate_probe(task) is False

    def test_returns_false_when_corrupt_json_string_metadata(self):
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'metadata': '{not valid json'}
        assert carries_substrate_probe(task) is False

    def test_returns_false_when_substrate_probe_key_absent(self):
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'metadata': {'other_key': 'value'}}
        assert carries_substrate_probe(task) is False

    def test_returns_true_when_substrate_probe_value_is_malformed(self):
        """Key 'substrate_probe' present even if its value is malformed → True."""
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'metadata': {'substrate_probe': 'oops'}}
        assert carries_substrate_probe(task) is True

    def test_returns_true_when_substrate_probe_has_empty_probe_set(self):
        """substrate_probe dict present but probe_set empty → still carries it."""
        from orchestrator.substrate_gate import carries_substrate_probe
        task = {'id': '1', 'metadata': {'substrate_probe': {'probe_set': '', 'checker': ['py']}}}
        assert carries_substrate_probe(task) is True


class TestRunSubstrateRecheckFailClosed:
    """RED tests: malformed-present substrate_probe → FLIP + WARNING (fail-closed)."""

    def test_malformed_substrate_probe_not_dict_returns_flip_with_warning(self, caplog):
        """substrate_probe present but NOT a dict → FLIP + WARNING (fail-closed)."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = {'id': '42', 'metadata': {'substrate_probe': 'oops'}}
        checker = fake_checker(rc=0)
        with caplog.at_level(logging.WARNING, logger='orchestrator.substrate_gate'):
            verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == FLIP, f'Expected FLIP, got {verdict.verdict!r}'
        assert verdict.flipped is True
        # Checker must NOT have been invoked
        assert checker.calls == []  # type: ignore[attr-defined]
        # A WARNING must have been emitted
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            f'Expected a WARNING; got: {[r.message for r in caplog.records]!r}'
        )

    def test_substrate_probe_dict_with_empty_probe_set_returns_flip_with_warning(self, caplog):
        """substrate_probe is a dict but probe_set empty → FLIP + WARNING (fail-closed)."""
        from orchestrator.substrate_gate import FLIP, run_substrate_recheck
        task = {'id': '42', 'metadata': {'substrate_probe': {'probe_set': '', 'checker': ['py']}}}
        checker = fake_checker(rc=0)
        with caplog.at_level(logging.WARNING, logger='orchestrator.substrate_gate'):
            verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == FLIP, f'Expected FLIP, got {verdict.verdict!r}'
        assert verdict.flipped is True
        assert checker.calls == []  # type: ignore[attr-defined]
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            f'Expected a WARNING; got: {[r.message for r in caplog.records]!r}'
        )

    def test_no_substrate_probe_key_returns_skip_no_warning(self, caplog):
        """Task with NO substrate_probe key → SKIP, no WARNING (regression guard)."""
        from orchestrator.substrate_gate import SKIP, run_substrate_recheck
        task = {'id': '42', 'metadata': {'other': 'value'}}
        checker = fake_checker(rc=0)
        with caplog.at_level(logging.WARNING, logger='orchestrator.substrate_gate'):
            verdict = run_substrate_recheck(task=task, worktree='/gate/wt', run_subprocess=checker)
        assert verdict.verdict == SKIP, f'Expected SKIP, got {verdict.verdict!r}'
        assert verdict.flipped is False
        assert checker.calls == []  # type: ignore[attr-defined]
        # No WARNING for a genuinely-absent probe
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
            f'Expected no WARNING for genuinely-absent probe; got: {[r.message for r in caplog.records]!r}'
        )


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
        assert checker.calls == []  # type: ignore[attr-defined]

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
        assert len(checker.calls) == 1  # type: ignore[attr-defined]
        called_argv, called_cwd = checker.calls[0]  # type: ignore[attr-defined]
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
    # is_deterministic was added after these tests were written (task 1899);
    # mock it False so _run_slot stays on the normal (non-deterministic) path.
    h.scheduler.is_deterministic = MagicMock(return_value=False)

    # Wire stub git_ops
    h.git_ops = MagicMock()
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' * 5)  # 40-char SHA
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.git_ops.project_root = tmp_path
    h.git_ops.prune_worktrees = AsyncMock(return_value=None)

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


    @pytest.mark.asyncio
    async def test_checker_runs_off_event_loop_thread(self, tmp_path: Path, monkeypatch):
        """run_substrate_recheck must execute on a worker thread, not the event-loop thread.

        step-11 RED: with the current inline call the checker runs on the same
        event-loop thread (idents equal) → this assertion fails.
        step-12 GREEN: after wrapping with asyncio.to_thread the checker executes
        on a worker thread (idents differ) → passes.

        The test is implementation-agnostic: any offload mechanism (to_thread,
        run_in_executor, etc.) that moves execution off the event-loop thread
        satisfies it.
        """
        h = _make_harness(tmp_path)
        assignment = _make_assignment()

        # Capture the event-loop thread identity at call time.
        event_loop_thread_ident = threading.get_ident()

        # Replace run_substrate_recheck with a recorder that captures its own
        # thread ident when invoked, then returns PASS so the gate returns True.
        checker_thread_idents: list[int] = []

        def _recording_recheck(**kw):
            checker_thread_idents.append(threading.get_ident())
            return _pass_verdict()

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            _recording_recheck,
        )

        with patch('asyncio.create_subprocess_exec', new=AsyncMock(
            return_value=_fake_proc(0)
        )):
            result = await h._run_substrate_gate(assignment)

        assert result is True, 'gate should return True (PASS verdict)'
        assert len(checker_thread_idents) == 1, (
            'run_substrate_recheck should have been called exactly once'
        )
        # KEY assertion: the checker must NOT run on the event-loop thread.
        assert checker_thread_idents[0] != event_loop_thread_ident, (
            'run_substrate_recheck ran on the event-loop thread — it must be '
            'offloaded to a worker thread (asyncio.to_thread) to avoid blocking '
            'the entire asyncio event loop during a 120s subprocess.run call'
        )

    @pytest.mark.asyncio
    async def test_worktree_add_uses_resolved_sha_and_gate_path(
        self, tmp_path: Path, monkeypatch
    ):
        """git worktree add is called with the resolved main SHA and the correct gate path.

        Regression guard: a wrong SHA variable or wrong gate_path in the argv would
        build the worktree at the wrong commit or path and silently let drift through.
        """
        h = _make_harness(tmp_path)
        assignment = _make_assignment()

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )

        exec_mock = AsyncMock(return_value=_fake_proc(0))
        with patch('asyncio.create_subprocess_exec', exec_mock):
            result = await h._run_substrate_gate(assignment)

        assert result is True

        # Locate the 'git worktree add' call among all subprocess calls
        # (pre-cleanup remove/prune + actual add + finally remove).
        all_calls = exec_mock.call_args_list
        add_calls = [c for c in all_calls if 'add' in c.args]
        assert len(add_calls) == 1, (
            f'Expected exactly 1 worktree add call; found {len(add_calls)}; '
            f'calls={[c.args for c in all_calls]!r}'
        )
        add_args = add_calls[0].args

        # The resolved SHA (from _make_harness: resolve_branch_sha returns 'deadbeef' * 5)
        expected_sha = 'deadbeef' * 5
        # The expected gate path (worktree_base / '_substrate-gate-<task_id>')
        expected_gate_path = str(tmp_path / '.worktrees' / '_substrate-gate-42')

        assert expected_sha in add_args, (
            f'Resolved main SHA not found in worktree add argv: {add_args!r}'
        )
        assert expected_gate_path in add_args, (
            f'Gate path not found in worktree add argv: {add_args!r}'
        )

    @pytest.mark.asyncio
    async def test_worktree_add_uses_symbolic_ref_fallback_when_sha_none(
        self, tmp_path: Path, monkeypatch
    ):
        """When resolve_branch_sha returns None the fallback branch name is used as ref.

        Exercises the None-fallback branch: the gate should still proceed (dispatch
        is allowed when the worktree add succeeds on the fallback ref), and the
        worktree add argv should contain the branch name rather than a resolved SHA.
        """
        h = _make_harness(tmp_path)
        # Override the mock so resolve_branch_sha returns None → triggers fallback
        h.git_ops.resolve_branch_sha = AsyncMock(return_value=None)
        assignment = _make_assignment()

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )

        exec_mock = AsyncMock(return_value=_fake_proc(0))
        with patch('asyncio.create_subprocess_exec', exec_mock):
            result = await h._run_substrate_gate(assignment)

        # Gate still returns True — PASS verdict proceeds even on the fallback path.
        assert result is True, 'gate should return True (PASS) even with SHA-None fallback'

        # Find the 'git worktree add' call
        all_calls = exec_mock.call_args_list
        add_calls = [c for c in all_calls if 'add' in c.args]
        assert len(add_calls) == 1, (
            f'Expected exactly 1 worktree add call; found {len(add_calls)}; '
            f'calls={[c.args for c in all_calls]!r}'
        )
        add_args = add_calls[0].args

        # With the fallback, the configured main_branch name ('main') is used instead
        # of a resolved SHA.  It must appear as the ref argument to 'git worktree add'.
        assert 'main' in add_args, (
            f'Fallback branch name ("main") not found in worktree add argv: {add_args!r}'
        )
        # No 40-char hex SHA should appear (SHA was None → fallback is the branch name)
        assert 'deadbeef' * 5 not in add_args, (
            f'Resolved SHA should NOT appear in worktree add argv when resolve returned None: '
            f'{add_args!r}'
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
    from orchestrator.workflow import TerminalReport, WorkflowOutcome, WorkflowState
    wf = AsyncMock()
    # W9-γ: run() returns a TerminalReport (TR-1), not a bare WorkflowOutcome.
    wf.run = AsyncMock(return_value=TerminalReport(
        outcome=WorkflowOutcome.DONE, reason='', phase=WorkflowState.DONE,
        detail='', category=None,
    ))
    wf.metrics = MagicMock(
        total_cost_usd=0.0,
        total_duration_ms=0,
        agent_invocations=0,
        execute_iterations=0,
        verify_attempts=0,
        review_cycles=0,
    )
    wf._steward = None
    return wf


class TestRunSlotSubstrateGateWiring:
    """PRD §10 two-way boundary: _run_slot wiring for the substrate gate.

    Tests verify the gate integration in _run_slot BEFORE TaskWorkflow construction:
    (a) FLIP → TaskWorkflow never constructed / workflow.run never awaited, cooldown armed.
    (b) PASS → TaskWorkflow IS constructed and workflow.run IS awaited.
    (c) NO-PROBE → gate never invoked (task metadata carries no substrate_probe key), workflow proceeds.

    Task 2121: _run_slot's dispatch gate now calls substrate_gate.carries_substrate_probe
    (module-level, key-presence) directly on assignment.task instead of consulting
    Scheduler.carries_substrate_probe (deleted — it diverged from the module predicate
    and is why task 2121 exists). Gate routing below is therefore driven entirely by the
    task metadata _make_assignment(probe=...) builds; no scheduler-level stub is needed
    or consulted.
    """

    @pytest.mark.asyncio
    async def test_flip_blocks_workflow_construction(self, tmp_path: Path):
        """(a) FLIP: TaskWorkflow must NOT be constructed when gate returns False."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(probe=True)
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


# ---------------------------------------------------------------------------
# Task 2121 RED: dispatch-level fail-closed via the REAL production gate
# ---------------------------------------------------------------------------


def _stub_old_carries_substrate_probe(h) -> None:
    """Stub the still-present (pre-deletion) Scheduler.carries_substrate_probe.

    Replicates the OLD divergent semantics (``extract_probe_set(task) is not
    None``) that the harness's production dispatch gate consulted BEFORE task
    2121's rewire, WITHOUT referencing the ``Scheduler.carries_substrate_probe``
    symbol itself (it is deleted as part of this task's impl step).

    On base (pre-rewire) harness.py calls ``self.scheduler.carries_substrate_probe``
    directly, so this stub makes the gate observably skip for a malformed-but-
    present descriptor — reproducing the actual defect (RED).  After the
    rewire, harness.py calls ``substrate_gate.carries_substrate_probe`` (module-
    level) directly instead, so this stub is never consulted — it goes dead
    but harmless (GREEN).

    A plain ``MagicMock()`` scheduler attribute would return a truthy MagicMock
    for any task regardless of predicate choice, which would make the base
    dispatch gate run unconditionally and mask the defect — hence the explicit
    ``side_effect`` rather than a bare mock.
    """
    from orchestrator.substrate_gate import extract_probe_set  # noqa: PLC0415
    h.scheduler.carries_substrate_probe = MagicMock(
        side_effect=lambda t: extract_probe_set(t) is not None
    )


class TestRunSlotSubstrateFailClosed:
    """Task 2121: the production dispatch gate must fail CLOSED, not skip.

    Unlike ``TestRunSlotSubstrateGateWiring`` above (which mocks BOTH
    ``carries_substrate_probe`` and ``_run_substrate_gate``), these tests
    drive the REAL ``_run_substrate_gate`` / ``run_substrate_recheck`` end to
    end through ``_run_slot`` — the actual production dispatch path — so they
    exercise the real predicate-choice decision at harness.py's dispatch gate
    rather than a mocked stand-in.

    (a)/(b): a declared-but-malformed ``substrate_probe`` descriptor must
    BLOCK dispatch (agent never spun up, task blocked, exactly one L1
    'design_concern' escalation filed) rather than silently skip the gate.
    (c): a task with no ``substrate_probe`` key at all is an unaffected
    companion case — dispatch proceeds normally, no block, no escalation.
    """

    async def _run_and_assert_blocked(
        self, tmp_path: Path, *, task_id: str, metadata: dict,
    ) -> None:
        """Shared body for the two malformed-descriptor cases (a) and (b)."""
        from escalation.queue import EscalationQueue  # noqa: PLC0415

        h = _make_slot_harness(tmp_path)
        _stub_old_carries_substrate_probe(h)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')

        assignment = MagicMock()
        assignment.task_id = task_id
        assignment.task = {'id': task_id, 'title': 'fail-closed test', 'metadata': metadata}

        sem = MagicMock()
        sem.release = MagicMock()

        proc_mock = AsyncMock(return_value=_fake_proc(0))
        with (
            patch('asyncio.create_subprocess_exec', proc_mock),
            patch('orchestrator.harness.TaskWorkflow') as MockWorkflow,
        ):
            MockWorkflow.return_value = _make_mock_workflow()
            await h._run_slot(assignment, sem)

        assert not MockWorkflow.called, (
            f'TaskWorkflow must NOT be constructed for malformed descriptor '
            f'{metadata!r} — the dispatch gate must fail CLOSED, not skip'
        )
        h.scheduler.set_task_status.assert_awaited_once_with(task_id, 'blocked')  # type: ignore[attr-defined]

        pending = h._escalation_queue.get_pending()
        l1s = [e for e in pending if e.task_id == task_id and e.level == 1]
        assert len(l1s) == 1, f'Expected exactly 1 L1 escalation for task {task_id}; got {l1s!r}'
        assert l1s[0].category == 'design_concern', f'Expected design_concern, got {l1s[0].category!r}'
        assert l1s[0].summary.startswith('SUBSTRATE_FLIP'), f'Unexpected summary: {l1s[0].summary!r}'

        h.scheduler.release.assert_called_once()  # type: ignore[attr-defined]
        release_call = h.scheduler.release.call_args  # type: ignore[attr-defined]
        assert release_call.kwargs.get('requeued') is True, (
            f'scheduler.release must be called with requeued=True on a fail-closed '
            f'block; got {release_call!r}'
        )

    @pytest.mark.asyncio
    async def test_malformed_string_descriptor_blocks_dispatch(self, tmp_path: Path):
        """(a) substrate_probe='garbage' (string) — must block, not skip."""
        await self._run_and_assert_blocked(
            tmp_path, task_id='fc-a', metadata={'substrate_probe': 'garbage'},
        )

    @pytest.mark.asyncio
    async def test_empty_dict_descriptor_blocks_dispatch(self, tmp_path: Path):
        """(b) substrate_probe={} (empty dict) — must block, not skip."""
        await self._run_and_assert_blocked(
            tmp_path, task_id='fc-b', metadata={'substrate_probe': {}},
        )

    @pytest.mark.asyncio
    async def test_no_probe_key_allows_dispatch(self, tmp_path: Path):
        """(c) companion: no substrate_probe key at all — dispatch proceeds normally."""
        from escalation.queue import EscalationQueue  # noqa: PLC0415

        h = _make_slot_harness(tmp_path)
        _stub_old_carries_substrate_probe(h)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')

        task_id = 'fc-c'
        assignment = MagicMock()
        assignment.task_id = task_id
        assignment.task = {'id': task_id, 'title': 'no probe here', 'metadata': {}}

        sem = MagicMock()
        sem.release = MagicMock()

        proc_mock = AsyncMock(return_value=_fake_proc(0))
        with (
            patch('asyncio.create_subprocess_exec', proc_mock),
            patch('orchestrator.harness.TaskWorkflow') as MockWorkflow,
        ):
            mock_wf = _make_mock_workflow()
            MockWorkflow.return_value = mock_wf
            await h._run_slot(assignment, sem)

        assert MockWorkflow.called, 'TaskWorkflow must be constructed when no probe is declared'
        mock_wf.run.assert_awaited_once()

        blocked_calls = [
            call for call in h.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            if len(call.args) >= 2 and call.args[1] == 'blocked'
        ]
        assert not blocked_calls, (
            f'set_task_status must never be called with "blocked" for a non-probe '
            f'task; got {blocked_calls!r}'
        )

        pending = h._escalation_queue.get_pending()
        l1s = [e for e in pending if e.task_id == task_id]
        assert not l1s, f'Expected no escalation for non-probe task {task_id}; got {l1s!r}'


# ---------------------------------------------------------------------------
# TestSubstrateGateForeignBandGuard — harness pre-clean band-ownership guard
# (gitops-chokepoints ε, task 2205 step-13/14).
#
# _run_substrate_gate's best-effort pre-clean loop removes any stale gate
# worktree left by a prior interrupted run before adding a fresh one. This
# guards the path-scoped `git worktree remove --force` against ever
# targeting a foreign protected band, consulting the same
# `GitOps._refuse_foreign_band` primitive used by the git_ops.py sweeps and
# self-heal sites. The real gate_path is always `_substrate-gate-{task_id}`
# (the owned band), so a genuine refusal is unreachable through real call
# patterns — proven here via a forced-True spy, mirroring the other
# point-site wiring tests in test_protected_prefixes.py.
# ---------------------------------------------------------------------------


async def _init_gate_repo(repo: Path) -> None:
    from orchestrator.git_ops import _run

    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


def _make_real_git_harness(repo: Path):
    """Build a Harness whose ``git_ops`` is a REAL ``GitOps`` on *repo*.

    Unlike ``_make_harness`` (which replaces ``h.git_ops`` with a bare
    MagicMock), this keeps the real instance so ``_refuse_foreign_band`` can
    actually be exercised / spied on — needed to prove the substrate-gate
    pre-clean site consults the shared band-ownership guard.
    """
    from orchestrator.config import OrchestratorConfig
    from orchestrator.harness import Harness

    config = OrchestratorConfig(project_root=repo, max_per_module=1)
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(config)

    h.scheduler = MagicMock()
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.is_deterministic = MagicMock(return_value=False)
    h._escalation_queue = None
    return h


@pytest.mark.asyncio
class TestSubstrateGateForeignBandGuard:
    """RED — _run_substrate_gate's pre-clean consults _refuse_foreign_band
    before its path-scoped `git worktree remove --force` (step-13/14)."""

    async def test_refusal_prevents_preclean_removal(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A forced-True spy must stop the pre-clean removal (the stale gate
        worktree survives intact) and must be consulted with
        (gate_path, frozenset({'_substrate-gate-'}), 'substrate-gate-cleanup')."""
        from orchestrator.git_ops import _run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_gate_repo(repo)
        h = _make_real_git_harness(repo)
        assignment = _make_assignment()

        gate_path = h.git_ops.worktree_base / '_substrate-gate-42'
        h.git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(gate_path), 'main'],
            cwd=repo,
        )
        assert rc == 0, f'failed to plant stale gate worktree fixture: {err}'
        assert gate_path.exists()

        calls: list[tuple] = []

        def _spy(path, owned, context):
            calls.append((path, owned, context))
            return True

        monkeypatch.setattr(h.git_ops, '_refuse_foreign_band', _spy)
        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )

        result = await h._run_substrate_gate(assignment)

        assert gate_path.exists(), (
            'expected the stale gate worktree to survive the pre-clean sweep '
            '(refused), but it was removed from disk'
        )
        assert calls, (
            'expected _refuse_foreign_band to be consulted before the '
            'pre-clean remove'
        )
        called_path, called_owned, called_context = calls[0]
        assert called_path == gate_path
        assert called_owned == frozenset({'_substrate-gate-'}), (
            f'expected the pre-clean guard to pass owned={{"_substrate-gate-"}}; '
            f'got {called_owned!r}'
        )
        assert called_context == 'substrate-gate-cleanup'
        # Side effect of forcing a refusal on this always-owned path: the
        # still-registered stale worktree makes the subsequent (real) `git
        # worktree add` fail, which the gate maps to FLIP.
        assert result is False

    async def test_owned_band_preclean_removal_proceeds_unchanged(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Regression: with the REAL (unpatched) guard, a stale worktree at
        the owned `_substrate-gate-<id>` path is still cleaned up by the
        pre-clean sweep, so the gate succeeds exactly as before the guard was
        wired (if the guard wrongly refused its own band, `git worktree add`
        would fail on the still-registered path and the gate would FLIP)."""
        from orchestrator.git_ops import _run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_gate_repo(repo)
        h = _make_real_git_harness(repo)
        assignment = _make_assignment()

        gate_path = h.git_ops.worktree_base / '_substrate-gate-42'
        h.git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(gate_path), 'main'],
            cwd=repo,
        )
        assert rc == 0, f'failed to plant stale gate worktree fixture: {err}'

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )

        result = await h._run_substrate_gate(assignment)

        assert result is True, (
            'expected the gate to succeed — the pre-existing stale worktree '
            'at the owned gate_path must still be cleaned up by the '
            'pre-clean sweep'
        )


# ---------------------------------------------------------------------------
# TestSubstrateGatePruneChokepoint — gitops-chokepoints β (task 2190).
#
# The substrate-gate pre-clean's prune half must route through the guarded
# GitOps.prune_worktrees(context=...) chokepoint (added by α, task 2185)
# rather than issuing a raw ('git', 'worktree', 'prune') subprocess argv.
# ---------------------------------------------------------------------------


class TestSubstrateGatePruneChokepoint:
    """gitops-chokepoints β: the substrate-gate pre-clean's prune half routes
    through ``GitOps.prune_worktrees(context='substrate-gate-cleanup')``
    instead of a raw ``git worktree prune`` argv (mocked ``git_ops``)."""

    @pytest.mark.asyncio
    async def test_preclean_calls_prune_worktrees_with_context(
        self, tmp_path: Path, monkeypatch,
    ):
        """The pre-clean loop calls the guarded chokepoint with the
        'substrate-gate-cleanup' context, not a raw prune subprocess — and
        does so only after the path-scoped remove has run."""
        h = _make_harness(tmp_path)
        # h.git_ops is a bare MagicMock, so the default
        # `refuse_foreign_band(...)` return is a truthy MagicMock and
        # `not truthy` is False — force it False so the guarded path-scoped
        # `git worktree remove` branch is actually exercised by this test,
        # not silently skipped.
        h.git_ops.refuse_foreign_band = MagicMock(return_value=False)
        assignment = _make_assignment()

        exec_mock = AsyncMock(return_value=_fake_proc(0))
        # Record how many subprocess calls had happened by the time the prune
        # chokepoint fires, so we can confirm remove-then-prune ordering
        # (not just that both happened somewhere during the gate run).
        exec_calls_before_prune: list[int] = []
        h.git_ops.prune_worktrees.side_effect = (  # type: ignore[attr-defined]
            lambda **kw: exec_calls_before_prune.append(len(exec_mock.call_args_list))
        )

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )
        with patch('asyncio.create_subprocess_exec', new=exec_mock):
            result = await h._run_substrate_gate(assignment)

        assert result is True
        h.git_ops.prune_worktrees.assert_awaited_once_with(  # type: ignore[attr-defined]
            context='substrate-gate-cleanup',
        )
        assert exec_calls_before_prune == [1], (
            f'expected exactly one subprocess call (the path-scoped `git '
            f'worktree remove`) to precede the prune chokepoint call; '
            f'got {exec_calls_before_prune!r}'
        )

    @pytest.mark.asyncio
    async def test_no_raw_prune_argv_in_subprocess_calls(
        self, tmp_path: Path, monkeypatch,
    ):
        """No subprocess call in the gate path issues a raw
        ('git', 'worktree', 'prune') argv — prune only happens via the
        chokepoint delegate, asserted above."""
        h = _make_harness(tmp_path)
        # Force the guard False (see test_preclean_calls_prune_worktrees_with_
        # context above) so the path-scoped remove branch actually runs here
        # too, and this test's "no raw prune argv" sweep covers that call.
        h.git_ops.refuse_foreign_band = MagicMock(return_value=False)
        assignment = _make_assignment()

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )
        exec_mock = AsyncMock(return_value=_fake_proc(0))
        with patch('asyncio.create_subprocess_exec', exec_mock):
            result = await h._run_substrate_gate(assignment)

        assert result is True
        prune_argv_calls = [
            c for c in exec_mock.call_args_list
            if tuple(c.args) == ('git', 'worktree', 'prune')
        ]
        assert not prune_argv_calls, (
            f'expected no raw ("git", "worktree", "prune") subprocess call; '
            f'found {prune_argv_calls!r}'
        )

    @pytest.mark.asyncio
    async def test_stale_gate_worktree_self_heal_still_works(
        self, tmp_path: Path, monkeypatch,
    ):
        """Regression (real GitOps): a stale gate worktree left by a prior
        interrupted run is still cleaned up before 'worktree add', so the
        gate proceeds normally rather than FLIPping on 'already exists'."""
        from orchestrator.git_ops import _run

        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_gate_repo(repo)
        h = _make_real_git_harness(repo)
        assignment = _make_assignment()

        gate_path = h.git_ops.worktree_base / '_substrate-gate-42'
        h.git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(gate_path), 'main'],
            cwd=repo,
        )
        assert rc == 0, f'failed to plant stale gate worktree fixture: {err}'
        assert gate_path.exists()

        monkeypatch.setattr(
            'orchestrator.substrate_gate.run_substrate_recheck',
            lambda **kw: _pass_verdict(),
        )

        result = await h._run_substrate_gate(assignment)

        assert result is True, (
            'expected the gate to succeed — the pre-existing stale gate '
            'worktree must still be cleaned up (remove + chokepoint prune) '
            'so `git worktree add` does not fail with "already exists"'
        )
