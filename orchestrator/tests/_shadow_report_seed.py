"""Seeds a real project for the shadow-report tests: cells, their result JSONs and production spend.

One home for what ``test_shadow_report.py`` and ``test_eval_shadow_cli.py``
share. Everything is written through the public ``RunStore``, ``ResultStore``
and ``CostStore`` APIs, so a seeded store is one the production writers could
have produced. A module rather than ``conftest.py`` for the reason
``_briefing_helpers.py`` gives.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from shared.cost_store import CostStore
from shared.project_layout import runs_db

from orchestrator.evals.metrics import EvalMetrics
from orchestrator.evals.results import EvalResult, ResultStore
from orchestrator.run_store import RunStore, ShadowCell
from orchestrator.shadow_types import CostBasis, ReferenceKind, ShadowCellState, ShadowShape

PROJECT_ID = 'demo'
CANDIDATE = 'glm-5.3-endpoint'
INCUMBENT = 'claude-opus-high'
OPENED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
SETTLED_AT = OPENED_AT + timedelta(hours=2)
LANDED_SHA = 'b' * 40

USABLE_METRICS: Mapping[str, Any] = {
    'tests_pass': True,
    'lint_clean': True,
    'typecheck_clean': True,
    'review_blocking_issues': 0,
}

_ids = itertools.count(1)


@dataclass(frozen=True)
class SeededProject:
    """A project root laid out as the orchestrator lays one out, with an open run store."""

    project_root: Path
    db_path: Path
    results_dir: Path
    store: RunStore


def make_project(tmp_path: Path) -> SeededProject:
    project_root = tmp_path / 'project'
    db_path = runs_db(project_root)
    return SeededProject(project_root, db_path, tmp_path / 'results', RunStore(db_path))


def build_cell(**overrides: Any) -> ShadowCell:
    """A coherent cell with a fresh id and task: a settled one is stamped and referenced."""
    serial = next(_ids)
    fields: dict[str, Any] = {
        'cell_id': f'cell-{serial:04d}',
        'project_id': PROJECT_ID,
        'task_id': f'{9000 + serial}',
        'shape': ShadowShape.IMPLEMENTER,
        'candidate': CANDIDATE,
        'incumbent': INCUMBENT,
        'stratum': ('df', 'feature', 'full'),
        'base_sha': 'a' * 40,
        'state': ShadowCellState.SETTLED,
        'cost_basis': CostBasis.METERED,
        'opened_at': OPENED_AT,
        'settle_deadline': OPENED_AT + timedelta(hours=168),
    }
    fields.update(overrides)
    return ShadowCell(**_coherent(fields))


def _coherent(fields: dict[str, Any]) -> dict[str, Any]:
    state: ShadowCellState = fields['state']
    if state.requires_reason:
        fields.setdefault('reason', f'seeded {state.value}')
    if state.stamps_settled_at:
        fields.setdefault('settled_at', SETTLED_AT)
    if state is ShadowCellState.SETTLED:
        fields.setdefault('reference_kind', ReferenceKind.LANDED)
        if fields['reference_kind'] is ReferenceKind.LANDED:
            fields.setdefault('reference_sha', LANDED_SHA)
    return fields


def seed_cell(store: RunStore, **overrides: Any) -> ShadowCell:
    cell = build_cell(**overrides)
    store.insert_shadow_cell(cell)
    return cell


def write_result(results_dir: Path, cell: ShadowCell, **metric_overrides: Any) -> Path:
    """Persist *cell*'s EvalResult; the metrics are usable unless overridden."""
    metrics = EvalMetrics(
        **{**USABLE_METRICS, **metric_overrides},
        shadow_cell_id=cell.cell_id,
        reference_kind=cell.reference_kind,
    )
    result = EvalResult(
        task_id=cell.task_id,
        config_name=cell.candidate,
        outcome='done',
        metrics=metrics.to_dict(),
        worktree_path='',
        run_id=cell.cell_id,
    )
    return ResultStore(results_dir).save(result)


def seed_scored_cell(
    project: SeededProject, *, metrics: Mapping[str, Any] | None = None, **overrides: Any,
) -> ShadowCell:
    """A settled cell whose result JSON is written first, so the row points at it."""
    cell = build_cell(**overrides)
    path = write_result(project.results_dir, cell, **(metrics or {}))
    scored = replace(cell, result_path=path)
    project.store.insert_shadow_cell(scored)
    return scored


async def seed_production_cost(
    db_path: Path, project_id: str, task_id: str, cost_usd: float, completed_at: datetime,
) -> None:
    stamp = completed_at.astimezone(UTC).isoformat()
    async with CostStore(db_path) as costs:
        await costs.save_invocation(
            run_id='run-production', task_id=task_id, project_id=project_id,
            account_name='max-a', model='opus', role='implementer',
            cost_usd=cost_usd, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100,
            capped=False, started_at=stamp, completed_at=stamp,
        )


def write_config(tmp_path: Path, project_root: Path, n_min: int) -> Path:
    path = tmp_path / 'orchestrator.yaml'
    path.write_text(yaml.safe_dump({
        'project_root': str(project_root),
        'fused_memory': {'project_id': PROJECT_ID},
        'shadow_eval': {'n_min': n_min},
    }))
    return path
