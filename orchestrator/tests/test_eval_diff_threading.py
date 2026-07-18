"""Focused tests: each eval diff caller threads its base commit into get_diff.

Guards D1 (task 2469): the judge, compare, and rereview call sites must pass
the authoritative base commit through to ``snapshots.get_diff`` rather than
relying on the removed ``metadata.json`` read / uncommitted-diff fallback.

Follows the test_snapshots.py convention: drive the async entrypoints with
``asyncio.run(...)`` (no pytest-asyncio marker).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.evals.judge import run_judge


def test_run_judge_threads_pre_task_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_judge must call get_diff(worktree, task['pre_task_commit']).

    RED before the fix: run_judge calls ``get_diff(Path(...))`` with a single
    positional arg, so the 2-arg recorder raises TypeError before appending —
    ``recorded`` never reaches two BASESHA-bearing calls.
    """
    recorded: list[tuple[Path, str]] = []

    async def rec(worktree_path: Path, base_commit: str) -> str:
        recorded.append((worktree_path, base_commit))
        return 'SENTINEL'

    async def fake_invoke_agent(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            structured_output={
                'winner': 'A', 'confidence': 0.9, 'reasoning': 'r',
            },
            output='{"winner": "A", "confidence": 0.9, "reasoning": "r"}',
        )

    monkeypatch.setattr('orchestrator.evals.judge.get_diff', rec)
    monkeypatch.setattr(
        'orchestrator.evals.judge.invoke_agent', fake_invoke_agent,
    )

    # No 'diff' key on either result → get_diff IS invoked for both.
    result_a = {'worktree_path': '/tmp/wt_a', 'config_name': 'A'}
    result_b = {'worktree_path': '/tmp/wt_b', 'config_name': 'B'}
    task = {
        'id': 't',
        'name': 't',
        'pre_task_commit': 'BASESHA',
        'task_definition': {'description': 'd'},
    }

    asyncio.run(run_judge(result_a, result_b, task))

    assert len(recorded) == 2
    assert all(base == 'BASESHA' for _, base in recorded)
