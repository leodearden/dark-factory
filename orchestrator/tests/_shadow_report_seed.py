"""Seeds a real project for the shadow-report tests, and reads the rendered table back.

One home for what ``test_shadow_report.py`` and ``test_eval_shadow_cli.py``
share: the seeded cells, their result JSONs and production spend, and the C5
table's contracted columns with a parser for the rendered rows. Everything is
written through the public ``RunStore``, ``ResultStore`` and ``CostStore``
APIs, so a seeded store is one the production writers could have produced. A
module rather than ``conftest.py`` for the reason ``_briefing_helpers.py``
gives.
"""

from __future__ import annotations

import itertools
import re
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
OTHER_CANDIDATE = 'kimi-k2-endpoint'
INCUMBENT = 'claude-opus-high'
OPENED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
SETTLED_AT = OPENED_AT + timedelta(hours=2)
LANDED_SHA = 'b' * 40

# The contracted C5 columns, spelled out as the pin rather than imported from the report.
CONTRACTED_COLUMNS = (
    'shape', 'candidate', 'incumbent', 'pairs', 'unpaired', 'cap_excluded', 'expired',
    'failed', 'usable \u0394 [95% CI]', '$/usable cand', '$/usable inc',
    'credits/usable cand', 'credits/usable inc', 'terminal_kind', 'tags',
)

USABLE_METRICS: Mapping[str, Any] = {
    'tests_pass': True,
    'lint_clean': True,
    'typecheck_clean': True,
    'review_blocking_issues': 0,
}

_ids = itertools.count(1)


@dataclass(frozen=True)
class SeededProject:
    """A project root laid out as the orchestrator lays one out, with an open run store.

    The root is ``tmp_path`` itself, the value conftest's autouse
    ``_isolate_orch_config`` pins ``ORCH_PROJECT_ROOT`` to, so a written YAML
    and the environment agree on it.
    """

    project_root: Path
    db_path: Path
    results_dir: Path
    store: RunStore


def make_project(tmp_path: Path) -> SeededProject:
    db_path = runs_db(tmp_path)
    return SeededProject(tmp_path, db_path, tmp_path / 'results', RunStore(db_path))


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


def seed_two_lines(project: SeededProject) -> ShadowCell:
    """Four metered pairs for CANDIDATE and two subscription pairs for OTHER_CANDIDATE.

    Returns the first metered cell.
    """
    metered = [seed_scored_cell(project, cost_usd=1.0) for _ in range(4)]
    for credits in (4.0, 2.0):
        seed_scored_cell(
            project, candidate=OTHER_CANDIDATE, cost_basis=CostBasis.SUBSCRIPTION,
            cost_usd=0.5, cost_credits=credits,
        )
    return metered[0]


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
    path = tmp_path / 'dark-factory-orchestrator.yaml'
    path.write_text(yaml.safe_dump({
        'project_root': str(project_root),
        'fused_memory': {'project_id': PROJECT_ID},
        'shadow_eval': {'n_min': n_min},
    }))
    return path


def _row_cells(line: str) -> list[str]:
    return re.split(r'\s{2,}', line.strip())


def _header_index(lines: list[str]) -> int:
    [index] = [i for i, line in enumerate(lines) if _row_cells(line)[0] == 'shape']
    return index


def table_header(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    return tuple(_row_cells(lines[_header_index(lines)]))


def table_rows(text: str) -> list[dict[str, str]]:
    """The rendered table's rows, each keyed by the header's column names."""
    lines = text.splitlines()
    start = _header_index(lines)
    rows = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        rows.append(dict(zip(_row_cells(lines[start]), _row_cells(line), strict=True)))
    return rows


def table_row(text: str, candidate: str) -> dict[str, str]:
    [row] = [row for row in table_rows(text) if row['candidate'] == candidate]
    return row
