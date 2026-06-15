"""Tests for orchestrator.substrate_gate module.

Covers: extract_probe_set, build_checker_argv, run_substrate_recheck verdict
mapping, and (in later steps) the harness-level _run_substrate_gate /
_block_and_escalate_substrate_flip integration.

PRD: prd-gate-exec D4 — dispatch-time substrate re-diff.
"""

from __future__ import annotations

import json
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
        verdict = run_substrate_recheck(task=task, worktree='/wt/gate', run_subprocess=checker)
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
