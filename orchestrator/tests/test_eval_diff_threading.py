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

from orchestrator.evals.compare import _assess_task
from orchestrator.evals.judge import run_judge
from orchestrator.evals.runner import EvalResult


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


def test_assess_task_threads_pre_task_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compare._assess_task must thread task['pre_task_commit'] into get_diff.

    RED before the fix: _assess_task calls ``get_diff(wt)`` one-arg; the 2-arg
    recorder raises TypeError, which _assess_task's try/except swallows, so
    nothing is recorded and ``len(recorded) == 2`` fails.
    """
    recorded: list[tuple[Path, str]] = []

    async def rec(worktree_path: Path, base_commit: str) -> str:
        recorded.append((worktree_path, base_commit))
        return 'SENTINEL'

    async def fake_invoke_agent(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            structured_output={
                'winner': 'A',
                'confidence': 0.9,
                'summary': 's',
                'strengths_a': [],
                'weaknesses_a': [],
                'strengths_b': [],
                'weaknesses_b': [],
                'circumstances': '',
            },
            output='{}',
        )

    monkeypatch.setattr('orchestrator.evals.compare.get_diff', rec)
    monkeypatch.setattr(
        'orchestrator.evals.compare.invoke_agent', fake_invoke_agent,
    )

    # Real dirs so wt.is_dir() is True and the get_diff branch is reached.
    wt_a = tmp_path / 'wt_a'
    wt_b = tmp_path / 'wt_b'
    wt_a.mkdir()
    wt_b.mkdir()

    result_a = EvalResult(
        task_id='t', config_name='A', outcome='done',
        metrics={}, worktree_path=str(wt_a),
    )
    result_b = EvalResult(
        task_id='t', config_name='B', outcome='done',
        metrics={}, worktree_path=str(wt_b),
    )
    task = {
        'id': 't',
        'name': 't',
        'pre_task_commit': 'BASESHA',
        'task_definition': {'description': 'd'},
    }

    asyncio.run(_assess_task(task, result_a, result_b, 'A', 'B'))

    assert len(recorded) == 2
    assert all(base == 'BASESHA' for _, base in recorded)
