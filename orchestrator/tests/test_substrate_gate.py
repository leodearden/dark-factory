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
