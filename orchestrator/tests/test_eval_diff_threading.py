"""Focused tests: each eval diff caller threads its base commit into get_diff.

Guards D1 (task 2469): the judge, compare, and rereview call sites must pass
the authoritative base commit through to ``snapshots.get_diff`` rather than
relying on the removed ``metadata.json`` read / uncommitted-diff fallback.

Follows the test_snapshots.py convention: drive the async entrypoints with
``asyncio.run(...)`` (no pytest-asyncio marker).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.evals.compare import _assess_task
from orchestrator.evals.judge import run_judge
from orchestrator.evals.rereview import Candidate, rereview_one
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


def test_assess_task_missing_base_skips_get_diff_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A task lacking pre_task_commit must fail loud, not silently empty-diff.

    This is the exact D1 failure mode (task 2469): compare._assess_task can
    receive the compare_models ``tasks.get(task_id, {'id':.., 'name':..})``
    fallback dict, which lacks 'pre_task_commit'. The fix warns loudly and
    skips get_diff entirely rather than diffing against a missing base and
    silently grading an empty diff. This locks that behavior in:

    - get_diff is NEVER called (the recorder stays empty),
    - both diffs stay '' → the prompt renders the '(no diff available)'
      placeholder and the get_diff SENTINEL never leaks in,
    - the 'No pre_task_commit' warning fires.

    Worktree dirs are REAL (is_dir() True) so the *only* reason get_diff is
    skipped is the falsy base — not a missing worktree.
    """
    recorded: list[tuple[Path, str]] = []

    async def rec(worktree_path: Path, base_commit: str) -> str:
        recorded.append((worktree_path, base_commit))
        return 'SENTINEL'

    captured_prompts: list[str] = []

    async def fake_invoke_agent(**kwargs: object) -> SimpleNamespace:
        captured_prompts.append(str(kwargs.get('prompt', '')))
        return SimpleNamespace(
            structured_output={
                'winner': 'tie',
                'confidence': 0.0,
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

    # Real dirs so wt.is_dir() is True: the ONLY reason get_diff is skipped
    # is the missing base, not a missing worktree.
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
    # NO 'pre_task_commit' key → the degraded, fail-loud branch.
    task = {
        'id': 't',
        'name': 't',
        'task_definition': {'description': 'd'},
    }

    with caplog.at_level(logging.WARNING, logger='orchestrator.evals.compare'):
        asyncio.run(_assess_task(task, result_a, result_b, 'A', 'B'))

    # get_diff was never invoked — no silent diff against a missing base.
    assert recorded == []
    # Both diffs stayed empty → prompt shows the empty-diff placeholder twice
    # (one per model) and the SENTINEL (only returned if get_diff ran) is absent.
    assert len(captured_prompts) == 1
    assert captured_prompts[0].count('(no diff available)') == 2
    assert 'SENTINEL' not in captured_prompts[0]
    # The miss is loud, not swallowed.
    assert any('No pre_task_commit' in m for m in caplog.messages)


def test_rereview_one_threads_candidate_base_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rereview_one must call get_diff(cand.worktree_path, cand.base_commit).

    The recorder returns '' so rereview_one short-circuits to a ``skip``
    outcome, avoiding the real reviewer call and all file writes.

    RED before the fix: rereview_one calls ``get_diff(cand.worktree_path)``
    one-arg; its try/except turns the TypeError into an ``error`` outcome, so
    ``recorded`` stays empty and ``outcome.status`` is 'error', not 'skip'.
    """
    recorded: list[tuple[Path, str]] = []

    async def rec(worktree_path: Path, base_commit: str) -> str:
        recorded.append((worktree_path, base_commit))
        return ''

    monkeypatch.setattr('orchestrator.evals.rereview.get_diff', rec)

    cand = Candidate(
        result_path=Path('/tmp/none.json'),
        task_id='t',
        config_name='c',
        run_id='r',
        worktree_path=Path('/tmp/wt'),
        base_commit='BASESHA',
        current_blocking=0,
        current_suggestions=0,
        mtime=datetime(2026, 1, 1, tzinfo=UTC),
    )

    outcome = asyncio.run(rereview_one(cand))

    assert recorded == [(Path('/tmp/wt'), 'BASESHA')]
    assert outcome.status == 'skip'
