"""One eval fixture task, a config's finished result on it, and an assessment of it that measures.

What drives the judges and ``compare_models`` through a single fixture
without an LLM. Each builder returns a fresh value, so no test can mutate
another's copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.evals.results import EvalResult

FIXTURE_ID = 'df_task_9001'


def fixture_task() -> dict[str, Any]:
    """The task JSON the judges and the assessor read for ``FIXTURE_ID``."""
    return {
        'id': FIXTURE_ID,
        'name': 'eval probe',
        'pre_task_commit': 'a' * 40,
        'task_definition': {'description': 'd'},
    }


def eval_result(config_name: str, worktree: Path) -> EvalResult:
    """*config_name*'s finished run of the fixture, with its worktree created at *worktree*."""
    worktree.mkdir()
    return EvalResult(task_id=FIXTURE_ID, config_name=config_name, outcome='done',
                      metrics={}, worktree_path=str(worktree))


def a_wins_assessment() -> dict[str, Any]:
    """An assessor answer carrying every field ``ASSESSOR_SCHEMA`` requires, naming A the winner."""
    return {
        'winner': 'A', 'confidence': 0.9, 'summary': 's',
        'strengths_a': [], 'weaknesses_a': [], 'strengths_b': [], 'weaknesses_b': [],
        'circumstances': '',
    }
