"""One scheduled, windowed link-heal sweep run (task 7159).

A sweep plans from the link adjudicator's verdicts on the links whose child was
created since the last complete sweep, staging only the heals it is enabled for,
then heals what it staged, but only if it may write. This module decides whether
a sweep attempts writes: it does when ``link_heal.writes`` is on, the run is not
a dry run, and the exact model and prompt version the adjudicator asks under has
a passing calibration. At write time,
``link_heal_executor.py::run_apply`` re-checks that each sweep heal's verdict
came from an adjudicator with a passing calibration, and withholds any other.
Every run, writing or not, leaves its records in the ledger, one report on disk
and one info escalation naming the report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from shared.safe_io import atomic_write_text

from fused_memory.maintenance.link_adjudicator import LinkAdjudicator
from fused_memory.maintenance.link_heal import HealAction
from fused_memory.maintenance.link_heal_executor import (
    EscapeFiler,
    RunLimits,
    RunReport,
    run_adjudicator_plan,
    run_apply,
    run_report_escape,
)
from fused_memory.maintenance.link_heal_ledger import (
    AdjudicatorCalibration,
    LinkHealLedger,
    RunSource,
    plan_document_path,
)
from fused_memory.maintenance.link_heal_report import (
    CalibrationStatus,
    LedgerRun,
    SweepRunView,
    WritesWithheld,
    render_run_report,
    report_paths,
    run_headline,
)
from fused_memory.maintenance.link_heal_store import CreatedWindow, LinkCensus, LinkHealStore

if TYPE_CHECKING:
    from fused_memory.config.schema import LinkHealConfig


@dataclass(frozen=True)
class SweepSettings:
    """What one sweep covers, which heals it may stage, and whether it asks to write.

    *presence_projects* always includes *projects*: they are where a parent missing
    from its child's project is looked for, so a sweep over a subset of the
    configured projects reports a parent hosted in another one instead of
    detaching it as absent.
    """

    projects: tuple[str, ...]
    writes: bool
    dry_run: bool
    actions: frozenset[HealAction]
    window_hours: int
    presence_projects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'projects', tuple(self.projects))
        object.__setattr__(
            self, 'presence_projects',
            tuple(dict.fromkeys((*self.projects, *self.presence_projects))),
        )
        object.__setattr__(self, 'actions', frozenset(self.actions))
        if not self.projects:
            raise ValueError(
                'a link-heal sweep covers at least one project: set link_heal.projects '
                'or name one with --project',
            )

    @classmethod
    def from_config(
        cls,
        config: LinkHealConfig,
        *,
        projects: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> SweepSettings:
        """The settings *config* gives, covering *projects* instead when they are named."""
        return cls(
            projects=tuple(config.projects if projects is None else projects),
            writes=config.writes,
            dry_run=dry_run,
            actions=_heal_actions(config.sweep_actions),
            window_hours=config.sweep_window_hours,
            presence_projects=tuple(config.projects),
        )

    def withheld(self, calibration: AdjudicatorCalibration) -> WritesWithheld | None:
        """Why this sweep may not write, given its adjudicator's *calibration*;
        ``None`` when it may."""
        if not self.writes:
            return WritesWithheld.WRITES_OFF
        if self.dry_run:
            return WritesWithheld.DRY_RUN
        if calibration.calibration is None:
            return WritesWithheld.UNCALIBRATED
        if not calibration.passed:
            return WritesWithheld.CALIBRATION_FAILED
        return None


def _heal_actions(values: Sequence[str]) -> frozenset[HealAction]:
    known = {action.value: action for action in HealAction}
    unknown = [value for value in values if value not in known]
    if unknown:
        raise ValueError(
            f'link_heal.sweep_actions names unknown heal action(s) {unknown}; '
            f'the heal actions are {list(known)}',
        )
    return frozenset(known[value] for value in values)


def resolve_window(
    ledger: LinkHealLedger,
    *,
    now: datetime,
    since: datetime | None,
    window_hours: int,
    projects: Sequence[str],
) -> CreatedWindow:
    """``[since, now)``. With no *since*, the window starts at the earliest start over
    *projects*: each project's sweep watermark, or *window_hours* before *now* for a
    project no sweep has completed. Re-examining a project's links from an earlier
    start is harmless: adjudications are reused and pending heals are deduped."""
    if since is not None:
        return CreatedWindow(since=since, until=now)
    fallback = now - timedelta(hours=window_hours)
    start = min(ledger.sweep_watermark(project) or fallback for project in projects)
    return CreatedWindow(since=start, until=now)


@dataclass(frozen=True)
class SweepReport:
    """One sweep run: what its report says, where the report is, and its escalation."""

    view: SweepRunView
    json_path: Path
    report_path: Path
    escalation_id: str | None

    @property
    def plan_run_id(self) -> str:
        return self.view.plan.run_id

    @property
    def apply_run_id(self) -> str | None:
        return None if self.view.apply is None else self.view.apply.run_id

    @property
    def complete(self) -> bool:
        """Whether every link was read and judged and every staged heal was attempted."""
        return all(run.counts.complete for run in self.view.runs)


async def run_sweep(
    *,
    adjudicate: LinkAdjudicator,
    store: LinkHealStore,
    census: LinkCensus,
    ledger: LinkHealLedger,
    limits: RunLimits,
    settings: SweepSettings,
    filer: EscapeFiler,
    reports_dir: Path,
    now: datetime,
    since: datetime | None = None,
) -> SweepReport:
    """Plan a sweep over the links created in its window, heal what it staged when it
    may write, and report the run.

    The window is recorded for each project the sweep covers, and only a complete
    plan run moves a project's watermark, so links a failed read or adjudication
    left undecided, a project the sweep did not cover, and a gap before an explicit
    *since* are all examined by a later sweep. Examining a link again is harmless:
    adjudications are reused and pending heals are deduped. A sweep that may write
    files its adjudicator storm. One that may not only forecasts it.
    """
    window = resolve_window(
        ledger, now=now, since=since, window_hours=settings.window_hours,
        projects=settings.projects,
    )
    calibration = ledger.adjudicator_calibration(adjudicate.model, adjudicate.prompt_sha256)
    withheld = settings.withheld(calibration)
    plan = await run_adjudicator_plan(
        adjudicate=adjudicate,
        store=store,
        census=census,
        ledger=ledger,
        limits=limits,
        projects=settings.projects,
        plan_path_for=lambda run_id: plan_document_path(ledger.db_path.parent, run_id),
        source=RunSource.SWEEP,
        enabled_actions=settings.actions,
        created_in=window,
        filer=filer if withheld is None else None,
        presence_projects=settings.presence_projects,
    )
    ledger.record_sweep_window(
        plan.run_id, window, projects=settings.projects, complete=plan.counts.complete,
    )
    applied = None if withheld is not None else await run_apply(
        store=store, ledger=ledger, limits=limits, filer=filer, source=RunSource.SWEEP,
    )
    view = SweepRunView(
        plan=_ledger_run(plan),
        apply=None if applied is None else _ledger_run(applied),
        records=tuple(ledger.run_records(plan.run_id)),
        window=window,
        calibration=_calibration_status(adjudicate, calibration),
        withheld=withheld,
        ledger_path=ledger.db_path,
    )
    json_path, report_path = report_paths(reports_dir, now, plan.run_id)
    json_text, markdown = render_run_report(view)
    atomic_write_text(json_path, json_text, mkdir=True)
    atomic_write_text(report_path, markdown, mkdir=True)
    escalation_id = filer(
        run_report_escape({**run_headline(view), 'report_json': str(json_path)}, report_path),
    )
    return SweepReport(
        view=view, json_path=json_path, report_path=report_path, escalation_id=escalation_id,
    )


def _ledger_run(report: RunReport) -> LedgerRun:
    return LedgerRun(run_id=report.run_id, counts=report.counts)


def _calibration_status(
    adjudicate: LinkAdjudicator, calibration: AdjudicatorCalibration,
) -> CalibrationStatus:
    latest = calibration.calibration
    return CalibrationStatus(
        model=adjudicate.model,
        prompt_sha256=adjudicate.prompt_sha256,
        calibration_run_id=None if latest is None else latest.run_id,
        passed=calibration.passed,
    )
