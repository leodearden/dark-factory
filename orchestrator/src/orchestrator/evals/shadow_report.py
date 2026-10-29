"""The shadow-eval report: one paired line per candidate over a project's shadow_cells rows.

Contract: plans/live-shadow-eval-prd.md C5.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from shared.cost_store import CostStore

from orchestrator.evals.metrics import EvalMetrics, terminal_kind_of
from orchestrator.evals.paired_stats import PairedDifference, bootstrap_mean_ci
from orchestrator.evals.results import ResultLoadError, load_result
from orchestrator.run_store import RunStore, ShadowCell, ShadowCellQuery
from orchestrator.shadow_eval import PRODUCTION_COST_BASIS
from orchestrator.shadow_scoring import ShadowReference, score_implementer_cell
from orchestrator.shadow_types import CostBasis, ReferenceKind, ShadowCellState, ShadowShape

__all__ = [
    'CostPerUsable',
    'LineKey',
    'LineTag',
    'MissingResult',
    'ReportLine',
    'RunsDbMissingError',
    'ShadowReport',
    'UnsupportedShapeError',
    'read_shadow_report',
    'render_text',
]

UNMEASURED_KIND = 'unmeasured'
_NO_FIGURE = '-'
_COLUMN_GAP = '  '
_COLUMNS = (
    'shape', 'candidate', 'incumbent', 'pairs', 'unpaired', 'cap_excluded', 'expired',
    'failed', 'usable \u0394 [95% CI]', '$/usable cand', '$/usable inc',
    'credits/usable cand', 'credits/usable inc', 'terminal_kind', 'tags',
)
# On a pair production landed, so production's side of the difference is usable.
_PRODUCTION_USABLE_ON_A_PAIR = 1.0


class LineTag(StrEnum):
    UNDERPOWERED = 'UNDERPOWERED'
    RESULT_MISSING = 'result_missing'


class RunsDbMissingError(FileNotFoundError):
    """The runs.db to report on does not exist; reading never creates one."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(f'no runs.db at {db_path}')
        self.db_path = db_path


class UnsupportedShapeError(ValueError):
    """The report has no rule for the requested shape; the message names the supported ones."""


@dataclass(frozen=True, order=True)
class LineKey:
    shape: ShadowShape
    candidate: str
    incumbent: str
    cost_basis: CostBasis

    @classmethod
    def of(cls, cell: ShadowCell) -> LineKey:
        return cls(cell.shape, cell.candidate, cell.incumbent, cell.cost_basis)


@dataclass(frozen=True)
class CostPerUsable:
    """One side's spend per usable outcome; its dollars are imputed exactly when its basis says so."""

    usd_per_usable: float | None
    credits_per_usable: float | None
    cost_basis: CostBasis

    def __post_init__(self) -> None:
        if self.credits_per_usable is not None and not self.cost_basis.imputes_dollars:
            raise ValueError(
                f'CostPerUsable: a {self.cost_basis.value} side meters dollars, never credits '
                f'(got credits_per_usable={self.credits_per_usable!r})'
            )

    @property
    def usd_imputed(self) -> bool:
        return self.cost_basis.imputes_dollars

    def to_dict(self) -> dict[str, Any]:
        return {
            'usd_per_usable': self.usd_per_usable,
            'usd_imputed': self.usd_imputed,
            'credits_per_usable': self.credits_per_usable,
            'cost_basis': self.cost_basis.value,
        }


@dataclass(frozen=True)
class MissingResult:
    """A settled cell whose result could not be read, so it is neither paired nor unpaired."""

    cell_id: str
    task_id: str
    result_path: Path | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'cell_id': self.cell_id,
            'task_id': self.task_id,
            'result_path': None if self.result_path is None else str(self.result_path),
            'detail': self.detail,
        }


@dataclass(frozen=True)
class _ScoredCell:
    """A settled cell read back from its result: whether it pairs, and its outcome."""

    cell: ShadowCell
    paired: bool
    usable: bool
    terminal_kind: str


@dataclass(frozen=True)
class ReportLine:
    key: LineKey
    unpaired: int
    cap_excluded: int
    expired: int
    failed: int
    primary: PairedDifference
    candidate_cost: CostPerUsable
    incumbent_cost: CostPerUsable
    terminal_kinds: tuple[tuple[str, int], ...]
    result_missing: tuple[MissingResult, ...]
    n_min: int

    @property
    def pairs(self) -> int:
        return self.primary.n

    @property
    def tags(self) -> tuple[LineTag, ...]:
        flagged = (
            (LineTag.UNDERPOWERED, self.pairs < self.n_min),
            (LineTag.RESULT_MISSING, bool(self.result_missing)),
        )
        return tuple(tag for tag, raised in flagged if raised)

    def to_dict(self) -> dict[str, Any]:
        return {
            'shape': self.key.shape.value,
            'candidate': self.key.candidate,
            'incumbent': self.key.incumbent,
            'cost_basis': self.key.cost_basis.value,
            'pairs': self.pairs,
            'unpaired': self.unpaired,
            'cap_excluded': self.cap_excluded,
            'expired': self.expired,
            'failed': self.failed,
            'primary_outcome': {
                'name': _shape_rule(self.key.shape).outcome_name,
                'n': self.primary.n,
                'mean_difference': self.primary.mean,
                'ci95_low': self.primary.ci_low,
                'ci95_high': self.primary.ci_high,
            },
            'candidate_cost': self.candidate_cost.to_dict(),
            'incumbent_cost': self.incumbent_cost.to_dict(),
            'terminal_kind': dict(self.terminal_kinds),
            'tags': [tag.value for tag in self.tags],
            'result_missing': [gap.to_dict() for gap in self.result_missing],
        }


@dataclass(frozen=True)
class ShadowReport:
    project_id: str
    db_path: Path
    shape: ShadowShape
    n_min: int
    lines: tuple[ReportLine, ...]

    @property
    def complete(self) -> bool:
        return not any(line.result_missing for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            'project_id': self.project_id,
            'db_path': str(self.db_path),
            'shape': self.shape.value,
            'n_min': self.n_min,
            'complete': self.complete,
            'lines': [line.to_dict() for line in self.lines],
        }


@dataclass(frozen=True)
class _ShapeRule:
    """How a settled cell of one shape becomes a paired outcome."""

    outcome_name: str
    needs_reference: bool
    usable: Callable[[EvalMetrics, ShadowReference], bool]

    def pairs(self, reference: ShadowReference) -> bool:
        return not self.needs_reference or reference.kind is ReferenceKind.LANDED


def _implementer_usable(metrics: EvalMetrics, reference: ShadowReference) -> bool:
    """The gates production clears to land: tests, lint, typecheck, and no blocking review."""
    score = score_implementer_cell(metrics, reference)
    gates = (score.tests_pass, score.lint_clean, score.typecheck_clean)
    return all(gate is True for gate in gates) and score.review_blocking_issues == 0


_SHAPE_RULES: Mapping[ShadowShape, _ShapeRule] = MappingProxyType({
    ShadowShape.IMPLEMENTER: _ShapeRule(
        outcome_name='usable', needs_reference=True, usable=_implementer_usable,
    ),
})


def _shape_rule(shape: ShadowShape) -> _ShapeRule:
    try:
        return _SHAPE_RULES[shape]
    except KeyError:
        raise UnsupportedShapeError(
            f'the shadow report has no rule for shape {shape.value!r}; '
            f'supported shapes: {", ".join(sorted(_SHAPE_RULES))}'
        ) from None


async def read_shadow_report(
    db_path: Path,
    *,
    project_id: str,
    shape: ShadowShape,
    candidate: str | None,
    n_min: int,
) -> ShadowReport:
    """Report *project_id*'s *shape* cells in the runs.db at *db_path*, optionally one candidate's."""
    rule = _shape_rule(shape)
    if not db_path.is_file():
        raise RunsDbMissingError(db_path)
    cells = RunStore(db_path).find_shadow_cells(
        ShadowCellQuery(project_id, shape=shape, candidate=candidate)
    )
    readings = {
        cell.cell_id: _read_settled_cell(cell, rule)
        for cell in cells
        if cell.state is ShadowCellState.SETTLED
    }
    pairs = [r.cell for r in readings.values() if isinstance(r, _ScoredCell) and r.paired]
    production_costs = await _production_costs(db_path, project_id, pairs)
    lines = tuple(
        _summarise_line(key, group, readings, production_costs, n_min)
        for key, group in sorted(_group_by_line(cells).items())
    )
    return ShadowReport(project_id, db_path, shape, n_min, lines)


def _read_settled_cell(cell: ShadowCell, rule: _ShapeRule) -> _ScoredCell | MissingResult:
    reference = _reference_of(cell)
    if cell.result_path is None:
        return MissingResult(cell.cell_id, cell.task_id, None, 'the row names no result_path')
    try:
        metrics, raw = _load_metrics(cell.result_path)
    except ResultLoadError as e:
        return MissingResult(cell.cell_id, cell.task_id, cell.result_path, e.detail)
    kind = terminal_kind_of(raw)
    return _ScoredCell(
        cell=cell,
        paired=rule.pairs(reference),
        usable=rule.usable(metrics, reference),
        terminal_kind=UNMEASURED_KIND if kind is None else kind,
    )


def _reference_of(cell: ShadowCell) -> ShadowReference:
    if cell.reference_kind is None:
        raise ValueError(f'shadow cell {cell.cell_id!r} is settled but carries no reference_kind')
    try:
        return ShadowReference(cell.reference_kind, cell.reference_sha)
    except ValueError as e:
        raise ValueError(f'shadow cell {cell.cell_id!r}: {e}') from e


def _load_metrics(path: Path) -> tuple[EvalMetrics, dict[str, Any]]:
    raw = load_result(path).metrics
    try:
        return EvalMetrics.from_dict(raw), raw
    except (ValueError, TypeError, AttributeError) as e:
        raise ResultLoadError(path, f'unreadable metrics: {type(e).__name__}: {e}') from e


async def _production_costs(
    db_path: Path, project_id: str, pairs: Sequence[ShadowCell],
) -> dict[str, float]:
    """Production's spend on each pair's task after the cell opened, keyed by cell id."""
    if not pairs:
        return {}
    async with CostStore(db_path) as costs:
        return {
            cell.cell_id: await costs.task_invocation_cost(
                project_id, cell.task_id, since=cell.opened_at,
            )
            for cell in pairs
        }


def _group_by_line(cells: Iterable[ShadowCell]) -> dict[LineKey, list[ShadowCell]]:
    groups: defaultdict[LineKey, list[ShadowCell]] = defaultdict(list)
    for cell in cells:
        groups[LineKey.of(cell)].append(cell)
    return groups


def _summarise_line(
    key: LineKey,
    cells: Sequence[ShadowCell],
    readings: Mapping[str, _ScoredCell | MissingResult],
    production_costs: Mapping[str, float],
    n_min: int,
) -> ReportLine:
    """One line from its cells, their settled readings and production's cost on each pair."""
    read = [readings[cell.cell_id] for cell in cells if cell.cell_id in readings]
    scored = [r for r in read if isinstance(r, _ScoredCell)]
    pairs = [r for r in scored if r.paired]
    states = Counter(cell.state for cell in cells)
    return ReportLine(
        key=key,
        unpaired=len(scored) - len(pairs),
        cap_excluded=states[ShadowCellState.CAP_EXCLUDED],
        expired=states[ShadowCellState.EXPIRED],
        failed=states[ShadowCellState.FAILED],
        primary=bootstrap_mean_ci(
            [float(pair.usable) - _PRODUCTION_USABLE_ON_A_PAIR for pair in pairs]
        ),
        candidate_cost=_candidate_cost(key.cost_basis, pairs),
        incumbent_cost=_incumbent_cost([production_costs[p.cell.cell_id] for p in pairs]),
        terminal_kinds=tuple(sorted(Counter(r.terminal_kind for r in scored).items())),
        result_missing=tuple(r for r in read if isinstance(r, MissingResult)),
        n_min=n_min,
    )


def _candidate_cost(basis: CostBasis, pairs: Sequence[_ScoredCell]) -> CostPerUsable:
    """The row's own spend is the authority (PRD decision 11), never the result JSON's."""
    usable = sum(pair.usable for pair in pairs)
    credits = sum(pair.cell.cost_credits for pair in pairs)
    return CostPerUsable(
        usd_per_usable=_per(sum(pair.cell.cost_usd for pair in pairs), usable),
        credits_per_usable=_per(credits, usable) if basis.imputes_dollars else None,
        cost_basis=basis,
    )


def _incumbent_cost(production_usd: Sequence[float]) -> CostPerUsable:
    return CostPerUsable(
        usd_per_usable=_per(sum(production_usd), len(production_usd)),
        credits_per_usable=None,
        cost_basis=PRODUCTION_COST_BASIS,
    )


def _per(total: float, count: int) -> float | None:
    return total / count if count else None


def render_text(report: ShadowReport) -> str:
    """The report as a preamble, a column-aligned table and any result_missing detail."""
    sections = [_preamble(report)]
    if report.lines:
        sections.append(_aligned([_COLUMNS, *(_row_cells(line) for line in report.lines)]))
    else:
        sections.append('no shadow cells match')
    trailer = _missing_trailer(report)
    if trailer:
        sections.append('\n'.join(trailer))
    return '\n\n'.join(sections) + '\n'


def _preamble(report: ShadowReport) -> str:
    return '\n'.join((
        f'shadow-eval report: project {report.project_id}, shape {report.shape.value}',
        f'runs.db: {report.db_path}',
        f'n_min {report.n_min}: a line with fewer pairs is UNDERPOWERED '
        f'(provisional, PRD open question 2; not a power calculation)',
    ))


def _row_cells(line: ReportLine) -> tuple[str, ...]:
    return (
        line.key.shape.value,
        line.key.candidate,
        line.key.incumbent,
        *(str(count) for count in (
            line.pairs, line.unpaired, line.cap_excluded, line.expired, line.failed,
        )),
        _difference(line.primary),
        _dollars(line.candidate_cost),
        _dollars(line.incumbent_cost),
        _credits(line.candidate_cost),
        _credits(line.incumbent_cost),
        _joined(f'{kind}:{count}' for kind, count in line.terminal_kinds),
        _joined(tag.value for tag in line.tags),
    )


def _difference(primary: PairedDifference) -> str:
    if primary.mean is None:
        return _NO_FIGURE
    if primary.ci_low is None or primary.ci_high is None:
        return f'{primary.mean:+.3f}'
    return f'{primary.mean:+.3f} [{primary.ci_low:+.3f}, {primary.ci_high:+.3f}]'


def _dollars(cost: CostPerUsable) -> str:
    """The only formatter of a dollar figure, so an imputed one can never print untagged."""
    if cost.usd_per_usable is None:
        return _NO_FIGURE
    figure = f'{cost.usd_per_usable:.2f}'
    return f'{figure} [imputed]' if cost.usd_imputed else figure


def _credits(cost: CostPerUsable) -> str:
    return _NO_FIGURE if cost.credits_per_usable is None else f'{cost.credits_per_usable:.2f}'


def _joined(items: Iterable[str]) -> str:
    return ','.join(items) or _NO_FIGURE


def _aligned(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return '\n'.join(
        _COLUMN_GAP.join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    )


def _missing_trailer(report: ShadowReport) -> list[str]:
    trailer = [
        f'result_missing: cell {gap.cell_id} task {gap.task_id} ({line.key.candidate}) '
        f'{_NO_FIGURE if gap.result_path is None else gap.result_path}: {gap.detail}'
        for line in report.lines
        for gap in line.result_missing
    ]
    if not report.complete:
        trailer.append(
            f'INCOMPLETE: {len(trailer)} settled cell(s) have no readable result and are '
            f'counted in no column; this is not a complete measurement'
        )
    return trailer
