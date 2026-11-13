"""Rendering one link-heal sweep run's records as a JSON and Markdown report.

The renderer is pure and writes no files: the sweep decides what a run did,
builds a :class:`SweepRunView` of it, and writes what :func:`render_run_report`
returns. The records themselves stay in the ledger; a report counts them and
names the ledger rows they are read from.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fused_memory.maintenance.link_heal import RunCounts
from fused_memory.maintenance.link_heal_executor import undo_command
from fused_memory.maintenance.link_heal_ledger import NO_ACTION_TAKEN, CandidateRecord
from fused_memory.maintenance.link_heal_store import CreatedWindow

SWEEP_REPORT_FORMAT = 'link-heal-sweep-report/1'

#: The verdict column of a record no adjudication decided.
NO_VERDICT = 'none'


class WritesWithheld(StrEnum):
    """Why a sweep made no apply run."""

    #: ``link_heal.writes`` is false.
    WRITES_OFF = 'writes_off'
    #: The run was asked for with ``--dry-run``.
    DRY_RUN = 'dry_run'
    #: No calibration of the adjudicator's model and prompt version exists.
    UNCALIBRATED = 'uncalibrated'
    #: The latest calibration of the adjudicator's model and prompt version failed.
    CALIBRATION_FAILED = 'calibration_failed'

    @property
    def is_dry_run(self) -> bool:
        return self in (WritesWithheld.WRITES_OFF, WritesWithheld.DRY_RUN)


@dataclass(frozen=True)
class LedgerRun:
    """A ledger run by id, with the counts it disclosed."""

    run_id: str
    counts: RunCounts

    def as_json(self) -> dict[str, Any]:
        return {'run_id': self.run_id, 'counts': self.counts.as_json()}


@dataclass(frozen=True)
class CalibrationStatus:
    """The calibration a sweep's writes were gated on: the adjudicator it asked
    for, and the latest calibration of it, if any."""

    model: str
    prompt_sha256: str
    calibration_run_id: str | None
    passed: bool

    def __post_init__(self) -> None:
        if self.passed and self.calibration_run_id is None:
            raise ValueError('a passed calibration names its calibration run')

    def as_json(self) -> dict[str, Any]:
        return {
            'model': self.model,
            'prompt_sha256': self.prompt_sha256,
            'calibration_run_id': self.calibration_run_id,
            'passed': self.passed,
        }


@dataclass(frozen=True)
class SweepRunView:
    """Everything one sweep run's report says: its plan run and records, its apply
    run or why it made none, its window and calibration, and the ledger."""

    plan: LedgerRun
    apply: LedgerRun | None
    records: tuple[CandidateRecord, ...]
    window: CreatedWindow
    calibration: CalibrationStatus
    withheld: WritesWithheld | None
    ledger_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, 'records', tuple(self.records))
        if (self.apply is None) == (self.withheld is None):
            raise ValueError('a sweep run has an apply run exactly when no write was withheld')

    @property
    def undo_run_id(self) -> str | None:
        """The run to undo: the apply run, once it applied a heal."""
        if self.apply is None or self.apply.counts.applied == 0:
            return None
        return self.apply.run_id

    @property
    def runs(self) -> tuple[LedgerRun, ...]:
        return (self.plan,) if self.apply is None else (self.plan, self.apply)


def render_run_report(view: SweepRunView) -> tuple[str, str]:
    """The report's JSON text and Markdown page; the same view renders the same bytes."""
    document = _document(view)
    return json.dumps(document, sort_keys=True, indent=2) + '\n', _markdown(view, document)


def report_paths(reports_dir: Path, now: datetime, plan_run_id: str) -> tuple[Path, Path]:
    """Where a sweep run's JSON and Markdown reports go, dated by *now*'s UTC day."""
    if now.tzinfo is None:
        raise ValueError(f'a report is dated by a timezone-aware now; got {now.isoformat()}')
    stem = f'link-heal-sweep-{now.astimezone(UTC).date().isoformat()}-{plan_run_id[:8]}'
    return reports_dir / f'{stem}.json', reports_dir / f'{stem}.md'


_CROSS_TAB_KEYS = ('project', 'decision', 'verdict', 'action_taken', 'action_state')


def _cells(record: CandidateRecord) -> tuple[str, str, str, str, str]:
    verdict = NO_VERDICT if record.verdict is None else record.verdict.value
    state = NO_ACTION_TAKEN if record.action_state is None else record.action_state.value
    return record.project_id, record.decision.value, verdict, record.action_taken, state


def run_headline(view: SweepRunView) -> dict[str, Any]:
    """The facts a run's report leads with: its runs, its window and its writes."""
    return {
        'plan_run_id': view.plan.run_id,
        'apply_run_id': None if view.apply is None else view.apply.run_id,
        'undo_run_id': view.undo_run_id,
        'window': {
            'since': view.window.since.isoformat(), 'until': view.window.until.isoformat(),
        },
        'writes': {
            'mode': _writes_mode(view),
            'withheld': None if view.withheld is None else view.withheld.value,
        },
    }


def _document(view: SweepRunView) -> dict[str, Any]:
    return {
        'format': SWEEP_REPORT_FORMAT,
        **run_headline(view),
        'calibration': view.calibration.as_json(),
        'runs': {
            'plan': view.plan.as_json(),
            'apply': None if view.apply is None else view.apply.as_json(),
        },
        'counts': _counts(view.records),
        'escapes': {
            'filed': [
                {'run_id': run.run_id, **entry} for run in view.runs for entry in run.counts.escaped
            ],
            'would_escape': [
                {'run_id': run.run_id, 'anchor': anchor}
                for run in view.runs
                for anchor in run.counts.would_escape
            ],
        },
        'records_ref': {'ledger': str(view.ledger_path), 'plan_run_id': view.plan.run_id},
    }


def _writes_mode(view: SweepRunView) -> str:
    if view.withheld is None:
        return 'applied'
    return 'dry_run' if view.withheld.is_dry_run else 'withheld'


def _counts(records: Sequence[CandidateRecord]) -> dict[str, Any]:
    cells = Counter(_cells(record) for record in records)
    by_project: dict[str, Counter[str]] = {}
    for (project, decision, _verdict, _action, _state), count in cells.items():
        by_project.setdefault(project, Counter())[decision] += count
    return {
        'records': len(records),
        'by_project': {project: dict(tally) for project, tally in by_project.items()},
        'by_verdict': dict(_tally(cells, 2)),
        'by_action': dict(_tally(cells, 3)),
        'by_action_state': dict(_tally(cells, 4)),
        'cross_tab': [
            {**dict(zip(_CROSS_TAB_KEYS, cell, strict=True)), 'count': cells[cell]}
            for cell in sorted(cells)
        ],
    }


def _tally(cells: Mapping[tuple[str, ...], int], column: int) -> Counter[str]:
    tally: Counter[str] = Counter()
    for cell, count in cells.items():
        tally[cell[column]] += count
    return tally


def _markdown(view: SweepRunView, document: Mapping[str, Any]) -> str:
    plan_counts = view.plan.counts
    lines = [
        f'# Link-heal sweep {view.plan.run_id[:8]}',
        '',
        f'- window: `{document["window"]["since"]}` to `{document["window"]["until"]}`',
        f'- plan run: `{view.plan.run_id}`: examined {plan_counts.examined}, '
        f'staged {plan_counts.planned}, not enabled {plan_counts.not_enabled}, '
        f'adjudication failed {plan_counts.adjudication_failed}, '
        f'read failed {plan_counts.read_failed}',
        f'- calibration: {_calibration_line(view.calibration)}',
        f'- Writes: {_writes_line(view)}',
        *([] if view.undo_run_id is None else [
            f'- Undo: `{undo_command(view.undo_run_id)}`',
        ]),
        f'- records: ledger `{view.ledger_path}`, plan run `{view.plan.run_id}`',
        '',
        '## Escapes',
        '',
        *(_escape_lines(document['escapes']) or ['None.']),
        '',
        *_project_tables(document['counts']['cross_tab']),
    ]
    return '\n'.join(lines)


def _calibration_line(calibration: CalibrationStatus) -> str:
    adjudicator = f'model `{calibration.model}`, prompt `{calibration.prompt_sha256}`'
    if calibration.calibration_run_id is None:
        return f'{adjudicator}: none'
    outcome = 'passed' if calibration.passed else 'failed'
    return f'{adjudicator}: {outcome} (run `{calibration.calibration_run_id}`)'


def _writes_line(view: SweepRunView) -> str:
    apply, withheld = view.apply, view.withheld
    if apply is not None:
        applied = f'applied: apply run `{apply.run_id}` applied {apply.counts.applied} heal(s)'
        withheld_heals = apply.counts.withheld_uncalibrated
        if not withheld_heals:
            return applied
        return f'{applied}, withheld {withheld_heals} heal(s) whose adjudicator is uncalibrated'
    if withheld is not None and withheld.is_dry_run:
        return f'dry-run ({withheld})'
    return f'withheld ({withheld})'


def _escape_lines(escapes: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    return [
        *(
            f'- filed: `{entry["anchor"]}` as `{entry["escalation_id"]}` (run `{entry["run_id"][:8]}`)'
            for entry in escapes['filed']
        ),
        *(
            f'- would escape: `{entry["anchor"]}` (run `{entry["run_id"][:8]}`)'
            for entry in escapes['would_escape']
        ),
    ]


def _project_tables(cross_tab: Iterable[Mapping[str, Any]]) -> list[str]:
    rows_by_project: dict[str, list[Mapping[str, Any]]] = {}
    for row in cross_tab:
        rows_by_project.setdefault(row['project'], []).append(row)
    if not rows_by_project:
        return ['No links were examined.', '']
    lines: list[str] = []
    for project in sorted(rows_by_project):
        lines += [
            f'## {project}',
            '',
            '| decision | verdict | action | action state | count |',
            '|---|---|---|---|---|',
            *(
                f'| {row["decision"]} | {row["verdict"]} | {row["action_taken"]} '
                f'| {row["action_state"]} | {row["count"]} |'
                for row in rows_by_project[project]
            ),
            '',
        ]
    return lines
