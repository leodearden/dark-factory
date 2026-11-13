"""Link-heal runs (plans/write-triage-link-healing-prd.md H1, H2).

A plan run decides every live link, ledgers the heals it stands behind with one
examination per decided link, and writes the plan document: a deterministic
rendering of every pending heal, whose sha256 an operator can approve. It writes nothing to the store, and
files nothing: every escape it would raise is recorded as ``would_escape``.
Its verdicts come from the hand-link corpus (:func:`run_plan`) or from the
link adjudicator (:func:`run_adjudicator_plan`), which is asked only about the
links no deterministic row decides and that this adjudicator (model and prompt
version) has not judged at their current texts, so a model or brief change
asks again and never inherits older verdicts; its verdicts are ledgered as
adjudications. A deterministic
plan is :func:`run_plan` with no verdicts at all: only the deterministic rows
plan a heal, and every other link is reported unexamined.

An apply run drains the pending heals oldest first. Each heal re-reads its
record live, and is skipped when the record no longer shows what it was
planned against; otherwise it sends exactly one write, then re-reads the
record to verify it. An unattended run makes at most ``max_actions_per_run``
write attempts and escalates a backlog beyond its multiple; an operator who
approves the exact pending plan by sha lifts both. Any run stops, and
escalates, after ``write_failure_streak`` consecutive failed heals. An apply
whose pending heals rest on adjudications that are too often misfile or
CORRECTS writes nothing and escalates; only raising the ceiling lifts that.
A heal of a calibration-gated source (the sweep) is written only when the
adjudicator behind its verdict has a passing calibration; any other is
withheld, never written, attended or not.

An undo run takes every heal one apply run applied back to its pre-image,
newest first. Each step is corroborated against the image it expects and
verified after its one write; the heal's row is marked undone by the undo run
only once every step applied, and that row keeps a re-plan from proposing the
heal again at the same texts. A heal an earlier undo left part-way is taken
back from the last step that applied.

The escape anchors below belong to these filers alone; no other filer may
share them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shared.safe_io import atomic_write_text
from shared.streak import KeyedStreaks

from fused_memory.maintenance.link_adjudicator import LinkAdjudicator, LinkPair, LinkVerdict
from fused_memory.maintenance.link_heal import (
    COMPLETION_ACTIONS,
    MISFILE_VERDICTS,
    BasisSource,
    HealAction,
    LinkBasis,
    LinkDecision,
    LinkImage,
    LinkReads,
    LinkState,
    Plan,
    PlannedAction,
    RunCounts,
    StaleField,
    Verdict,
    apply_change,
    build_plan,
    corroboration_mismatch,
    corroboration_reads_parent,
    journal_reason,
    needs_verdict,
    plan_links,
    read_links,
    undo_changes,
    verification_mismatch,
)
from fused_memory.maintenance.link_heal_ledger import (
    UNDO_ACTION,
    ActionRow,
    ActionState,
    AdjudicationRecord,
    AdjudicationRow,
    Disposition,
    ExaminationRecord,
    LinkHealLedger,
    RunSource,
    UndoTarget,
    WriteRecords,
)
from fused_memory.maintenance.link_heal_store import (
    CreatedWindow,
    LinkCensus,
    LinkHealStore,
    LiveRecord,
    MetadataChange,
    StoreReadFailed,
)
from fused_memory.middleware._folded_escalation import file_folded_escalation

if TYPE_CHECKING:
    from fused_memory.config.schema import LinkHealConfig

logger = logging.getLogger(__name__)

BACKLOG_ANCHOR = 'link-heal-backlog'
WRITE_FAILURE_ANCHOR = 'link-heal-write-failure'
MISFILE_SHARE_ANCHOR = 'link-heal-misfile-share'
CORRECTS_SHARE_ANCHOR = 'link-heal-corrects-share'
ADJUDICATOR_ANCHOR = 'link-heal-adjudicator'
CALIBRATION_ANCHOR = 'link-heal-calibration'
RUN_REPORT_ANCHOR = 'link-heal-run-report'

#: The link-heal CLI, from the repo root.
LINK_HEAL_SCRIPT = 'fused-memory/scripts/link_heal.py'


def undo_command(run_id: str) -> str:
    """The one spelling of the command that takes back apply run *run_id*'s heals,
    as operators are shown it."""
    return f'{LINK_HEAL_SCRIPT} undo --run {run_id}'

#: A share ceiling judges only a run with at least this many adjudicated pairs.
SHARE_ESCAPE_MIN_PAIRS = 20

PLAN_FORMAT = 'link-heal-plan/1'

CAP_BIT = 'max_actions_per_run'
STREAK_STOP = 'write_failure_streak'
SHARE_STOP = 'adjudicated_share_ceiling'
PLAN_DOCUMENT_STOP = 'plan_document_unwritten'

#: The run sources whose heals are written only while the adjudicator behind each
#: heal's verdict has a passing calibration. An apply of one withholds every other
#: heal; so does a plan run of one that files its escapes, which is a run about to
#: be applied, before it stages its own heals.
CALIBRATION_GATED_SOURCES = frozenset({RunSource.SWEEP})


@dataclass(frozen=True)
class RunLimits:
    max_actions_per_run: int
    backlog_multiplier: int
    write_failure_streak: int
    misfile_share_ceiling: float
    corrects_share_ceiling: float

    @classmethod
    def from_config(cls, config: LinkHealConfig) -> RunLimits:
        return cls(
            max_actions_per_run=config.max_actions_per_run,
            backlog_multiplier=config.backlog_multiplier,
            write_failure_streak=config.write_failure_streak,
            misfile_share_ceiling=config.misfile_share_ceiling,
            corrects_share_ceiling=config.corrects_share_ceiling,
        )


@dataclass(frozen=True)
class Escape:
    """One condition a run escalates rather than absorbs."""

    anchor: str
    summary: str
    detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'detail', MappingProxyType(dict(self.detail)))


#: Files an :class:`Escape`; returns the escalation id, or ``None`` when none was filed.
EscapeFiler = Callable[[Escape], str | None]


def backlog_escape(
    pending_count: int, limits: RunLimits, *, projects: Sequence[str],
) -> Escape | None:
    """The backlog escape, when more heals are pending than the cap drains in its multiple."""
    threshold = limits.max_actions_per_run * limits.backlog_multiplier
    if pending_count <= threshold:
        return None
    return Escape(
        anchor=BACKLOG_ANCHOR,
        summary=(
            f'link-heal backlog: {pending_count} pending heals exceed {threshold} '
            f'(max_actions_per_run x backlog_multiplier)'
        ),
        detail={
            'pending': pending_count,
            'threshold': threshold,
            'max_actions_per_run': limits.max_actions_per_run,
            'backlog_multiplier': limits.backlog_multiplier,
            'projects': list(projects),
        },
    )


def write_failure_escape(
    run_id: str, *, streak: int, not_attempted: int, last_failure: Mapping[str, Any] | None,
    projects: Sequence[str],
) -> Escape:
    """The escape a run files when it stops on a failure streak."""
    return Escape(
        anchor=WRITE_FAILURE_ANCHOR,
        summary=f'link-heal run {run_id[:8]} stopped after {streak} consecutive failed heals',
        detail={
            'run_id': run_id,
            'consecutive_failures': streak,
            'not_attempted': not_attempted,
            'last_failure': dict(last_failure or {}),
            'projects': list(projects),
        },
    )


def share_escapes(
    verdicts: Sequence[Verdict], limits: RunLimits, *, projects: Sequence[str],
) -> tuple[Escape, ...]:
    """The share escapes *verdicts* trip: a misfile or CORRECTS share over its ceiling."""
    if len(verdicts) < SHARE_ESCAPE_MIN_PAIRS:
        return ()
    escapes = (
        _share_escape(
            verdicts, MISFILE_VERDICTS, limits.misfile_share_ceiling,
            anchor=MISFILE_SHARE_ANCHOR, label='misfile', projects=projects,
        ),
        _share_escape(
            verdicts, frozenset({Verdict.CORRECTS}), limits.corrects_share_ceiling,
            anchor=CORRECTS_SHARE_ANCHOR, label='CORRECTS', projects=projects,
        ),
    )
    return tuple(escape for escape in escapes if escape is not None)


def adjudicated_share_escapes(
    heals: Sequence[ActionRow],
    ledger: LinkHealLedger,
    limits: RunLimits,
    *,
    projects: Sequence[str],
) -> tuple[Escape, ...]:
    """The share escapes of the adjudications behind *heals*: every one, made or reused,
    that the plan runs which staged them rest on. A plan run forecasts with it; apply gates on it."""
    verdicts = ledger.adjudication_verdicts({row.run_id for row in heals})
    return share_escapes(verdicts, limits, projects=projects)


def _share_escape(
    verdicts: Sequence[Verdict],
    counted: frozenset[Verdict],
    ceiling: float,
    *,
    anchor: str,
    label: str,
    projects: Sequence[str],
) -> Escape | None:
    count = sum(verdict in counted for verdict in verdicts)
    share = count / len(verdicts)
    if share <= ceiling:
        return None
    return Escape(
        anchor=anchor,
        summary=(
            f'link-heal: {count} of {len(verdicts)} adjudicated pairs are {label} '
            f'(share {share:.2f} > ceiling {ceiling})'
        ),
        detail={
            'n': len(verdicts),
            'count': count,
            'share': share,
            'ceiling': ceiling,
            'projects': list(projects),
        },
    )


def adjudicator_storm_escape(
    summary: Mapping[str, Any], *, projects: Sequence[str],
) -> Escape:
    """The escape for a link adjudicator that stopped on consecutive failed shards."""
    return Escape(
        anchor=ADJUDICATOR_ANCHOR,
        summary=(
            f'link adjudicator ({summary.get("model")}) stopped after '
            f'{summary.get("count")} consecutive failed shards'
        ),
        detail={**summary, 'projects': list(projects)},
    )


def run_report_escape(summary: Mapping[str, Any], report_path: Path) -> Escape:
    """The info record that surfaces one sweep run's report; *summary* is the run's
    headline facts, carried as the detail beside the report's path."""
    return Escape(
        anchor=RUN_REPORT_ANCHOR,
        summary=f'link-heal sweep report: {report_path}',
        detail={**summary, 'report': str(report_path)},
    )


def render_escape_detail(detail: Mapping[str, Any]) -> str:
    """Sorted ``key: value`` lines; a non-string value renders as JSON."""
    return '\n'.join(
        f'{key}: {value if isinstance(value, str) else json.dumps(value, sort_keys=True)}'
        for key, value in sorted(detail.items())
    )


@dataclass(frozen=True)
class _EscapeRoute:
    """Where an anchor's escapes land. ``dedupe`` folds a filing into the anchor's open
    record; a route whose every filing carries its own payload must not fold."""

    category: str
    suggested_action: str
    severity: str = 'blocking'
    dedupe: bool = True


_ESCAPE_ROUTES: Mapping[str, _EscapeRoute] = MappingProxyType({
    BACKLOG_ANCHOR: _EscapeRoute(
        category='risk_identified',
        suggested_action=(
            'Run `fused-memory/scripts/link_heal.py status` and review the plan document; '
            'apply a reviewed plan with `apply --approved-plan-sha`, or raise '
            'link_heal.max_actions_per_run.'
        ),
    ),
    WRITE_FAILURE_ANCHOR: _EscapeRoute(
        category='infra_issue',
        suggested_action=(
            'Read the failed actions of the named run in link_heal.db (their detail '
            'carries the failure and error_type). Check that the server is healthy and '
            'that mem0_update.metadata_patch_allowed_agent_prefixes admits link-heal-, '
            'then re-run apply.'
        ),
    ),
    MISFILE_SHARE_ANCHOR: _EscapeRoute(
        category='risk_identified',
        suggested_action=(
            'Review the plan document (`fused-memory/scripts/link_heal.py status`) and the '
            'adjudicator plan runs\' verdicts in link_heal.db\'s adjudications table. If the '
            'misfile share is real, raise link_heal.misfile_share_ceiling and re-run apply; '
            'otherwise fix the adjudicator and re-plan.'
        ),
    ),
    CORRECTS_SHARE_ANCHOR: _EscapeRoute(
        category='risk_identified',
        suggested_action=(
            'Review the plan document (`fused-memory/scripts/link_heal.py status`) and the '
            'adjudicator plan runs\' verdicts in link_heal.db\'s adjudications table. If the '
            'CORRECTS share is real, raise link_heal.corrects_share_ceiling and re-run apply; '
            'otherwise fix the adjudicator and re-plan.'
        ),
    ),
    ADJUDICATOR_ANCHOR: _EscapeRoute(
        category='infra_issue',
        suggested_action=(
            'The link adjudicator\'s Claude CLI shards kept failing (the detail\'s labels '
            'name how). Check that the CLI is logged in and that link_heal.adjudicator_model '
            'names a usable model, then re-run `link_heal.py plan --from-adjudicator`.'
        ),
    ),
    CALIBRATION_ANCHOR: _EscapeRoute(
        category='risk_identified',
        suggested_action=(
            'Read the calibration report the detail names: it gives each failed floor and '
            'the confusion behind it. Either fix the adjudicator (its model or the rater '
            'brief) and re-run `link_heal.py calibrate`, or deliberately re-base '
            'link_heal.calibration_kappa_margin. Sweeps do not write until a calibration '
            'of the exact model and prompt passes.'
        ),
    ),
    RUN_REPORT_ANCHOR: _EscapeRoute(
        category='link_heal_run_report',
        suggested_action=(
            'Read the sweep report the detail names (Markdown, with the JSON beside it): '
            'counts by project, decision, verdict and action, and the escapes. To take back '
            f'every heal the run applied, run `{undo_command("<undo_run_id>")}` with the '
            'undo id the report gives.'
        ),
        severity='info',
        dedupe=False,
    ),
})


class FoldedEscapeFiler:
    """Files an :class:`Escape` into *project_root*'s queue, folded under its anchor.

    Every project's escapes land in this one queue, naming the project in the detail.
    """

    AGENT_ROLE = 'fused-memory/link-heal'

    def __init__(self, project_root: str | None, log: logging.Logger = logger) -> None:
        self._project_root = project_root
        self._log = log

    def __call__(self, escape: Escape) -> str | None:
        route = _ESCAPE_ROUTES[escape.anchor]
        return file_folded_escalation(
            self._project_root,
            anchor_task_id=escape.anchor,
            agent_role=self.AGENT_ROLE,
            category=route.category,
            severity=route.severity,
            summary=escape.summary,
            detail=render_escape_detail(escape.detail),
            suggested_action=route.suggested_action,
            logger=self._log,
            log_label='link_heal',
            level=1,
            dedupe=route.dedupe,
        )


def _file(filer: EscapeFiler, escape: Escape | None) -> tuple[Mapping[str, Any], ...]:
    """File *escape*, if any, and the ``escaped`` entry that discloses it."""
    if escape is None:
        return ()
    return ({'anchor': escape.anchor, 'escalation_id': filer(escape)},)


@dataclass(frozen=True)
class RunReport:
    run_id: str
    counts: RunCounts
    plan_sha256: str | None = None
    plan_path: Path | None = None


def render_plan_document(rows: Iterable[ActionRow]) -> bytes:
    """The canonical plan document for *rows*: the same rows always render the same bytes."""
    document = {'format': PLAN_FORMAT, 'actions': [_plan_entry(row) for row in rows]}
    return (json.dumps(document, sort_keys=True, indent=2) + '\n').encode('utf-8')


def plan_sha256(document: bytes) -> str:
    return hashlib.sha256(document).hexdigest()


def pending_plan_sha256(pending: Sequence[ActionRow]) -> str:
    """The sha an --approved-plan-sha is checked against: that of the plan document *pending* renders."""
    return plan_sha256(render_plan_document(pending))


def _plan_entry(row: ActionRow) -> dict[str, Any]:
    heal = row.planned
    return {
        'action_id': row.action_id,
        'project_id': heal.project_id,
        'child_id': heal.child_id,
        'action': heal.action.value,
        'pre_image': heal.pre_image.as_dict(),
        'post_image': heal.post_image.as_dict(),
        'basis_source': heal.basis_source.value,
        'basis_key': heal.basis_key,
        'child_sha256': heal.child_sha256,
        'parent_sha256': heal.parent_sha256,
    }


#: The dispositions of a heal the plan stands behind: new, or pending from before.
_STANDING = frozenset({Disposition.STAGED, Disposition.ALREADY_PENDING})


def _stage(
    decisions: Iterable[LinkDecision],
    ledger: LinkHealLedger,
    enabled_actions: frozenset[HealAction] | None = None,
) -> tuple[ExaminationRecord, ...]:
    """Each decision's examination: its heal dropped when it is not among
    *enabled_actions* (``None`` enables every action) or an undo took it back,
    matched to its row when already pending, else staged as new."""
    return tuple(_examine(decision, ledger, enabled_actions) for decision in decisions)


def _examine(
    decision: LinkDecision,
    ledger: LinkHealLedger,
    enabled_actions: frozenset[HealAction] | None,
) -> ExaminationRecord:
    planned = decision.planned
    if planned is None:
        return _examination(decision, Disposition.REPORTED)
    if enabled_actions is not None and planned.action not in enabled_actions:
        return _examination(decision, Disposition.NOT_ENABLED)
    if ledger.is_undo_suppressed(planned):
        return _examination(decision, Disposition.UNDO_SUPPRESSED)
    pending = ledger.find_pending(planned)
    if pending is None:
        return _examination(decision, Disposition.STAGED, staged=planned)
    return _examination(decision, Disposition.ALREADY_PENDING, pending_action_id=pending.action_id)


def _examination(
    decision: LinkDecision,
    disposition: Disposition,
    *,
    staged: PlannedAction | None = None,
    pending_action_id: int | None = None,
) -> ExaminationRecord:
    state = decision.state
    basis_source, basis_key = decision.basis_ref
    return ExaminationRecord(
        project_id=state.project_id,
        child=state.child,
        parent_id=state.link.parent_id,
        parent=state.parent,
        basis_source=basis_source,
        basis_key=basis_key,
        decision=decision.decision,
        disposition=disposition,
        staged=staged,
        pending_action_id=pending_action_id,
    )


def _staged_counts(counts: RunCounts, examinations: Sequence[ExaminationRecord]) -> RunCounts:
    dispositions = Counter(examination.disposition for examination in examinations)
    standing = [
        examination.decision.value
        for examination in examinations
        if examination.disposition in _STANDING
    ]
    return replace(
        counts,
        planned=len(standing),
        planned_by_action=Counter(standing),
        already_pending=dispositions[Disposition.ALREADY_PENDING],
        undo_suppressed=dispositions[Disposition.UNDO_SUPPRESSED],
        not_enabled=dispositions[Disposition.NOT_ENABLED],
    )


async def run_plan(
    bases: Iterable[LinkBasis],
    *,
    store: LinkHealStore,
    census: LinkCensus,
    ledger: LinkHealLedger,
    limits: RunLimits,
    projects: Sequence[str],
    source: RunSource,
    plan_path_for: Callable[[str], Path],
) -> RunReport:
    """A non-writing run: ledger the plan, write its document, report what would escape.

    The document goes, atomically, to ``plan_path_for(run_id)``.

    Every read is made before the run row exists, so a read that raises leaves no
    run. The new heals are committed only once their document is written: a
    document that cannot be written leaves nothing new pending, and the run
    finishes stopped by it.
    """
    plan = await build_plan(bases, store=store, census=census, projects=projects)
    run_id = ledger.start_run(source, writes=False)
    return await _publish_plan(
        plan, run_id,
        ledger=ledger, limits=limits, projects=projects, source=source,
        plan_path=plan_path_for(run_id),
    )


async def _publish_plan(
    plan: Plan,
    run_id: str,
    *,
    ledger: LinkHealLedger,
    limits: RunLimits,
    projects: Sequence[str],
    source: RunSource,
    plan_path: Path,
    escapes: Sequence[Escape] = (),
    escaped: Sequence[Mapping[str, Any]] = (),
    enabled_actions: frozenset[HealAction] | None = None,
    withhold_uncalibrated: bool = False,
) -> RunReport:
    """Stage *plan* against the ledger, write its document, and finish *run_id*.

    Its examinations commit with its new heals, or roll back with them; a heal
    outside *enabled_actions* is recorded and never staged. *escapes*, and the
    backlog and share escapes an apply of *source*'s pending heals would raise
    now, are recorded as ``would_escape``; *escaped* discloses the escapes the
    run has already filed.

    With *withhold_uncalibrated*, every pending heal of *source* whose verdict came
    from an adjudicator with no passing calibration is withheld first, so a fresh
    verdict's identical heal is staged as new rather than matched to it.
    """
    withheld = (
        _withhold_uncalibrated(ledger.pending_actions(source), ledger=ledger, run_id=run_id)[1]
        if withhold_uncalibrated
        else 0
    )
    examinations = _stage(plan.decisions, ledger, enabled_actions)
    counts = replace(
        _staged_counts(plan.counts, examinations),
        escaped=tuple(escaped), withheld_uncalibrated=withheld,
    )
    try:
        with ledger.publishing_planned(run_id, examinations, source) as pending:
            document = render_plan_document(pending)
            await asyncio.to_thread(atomic_write_text, plan_path, document.decode('utf-8'))
    except OSError as failure:
        logger.error('link-heal plan %s: cannot write %s: %s', run_id[:8], plan_path, failure)
        counts = replace(counts, stopped_by=PLAN_DOCUMENT_STOP)
        ledger.finish_run(run_id, counts=counts.as_json())
        return RunReport(run_id=run_id, counts=counts)
    sha = plan_sha256(document)
    backlog = backlog_escape(len(pending), limits, projects=projects)
    shares = adjudicated_share_escapes(pending, ledger, limits, projects=projects)
    would_escape = tuple(
        escape.anchor for escape in (backlog, *shares, *escapes) if escape is not None
    )
    counts = replace(counts, would_escape=would_escape)
    ledger.finish_run(run_id, counts=counts.as_json(), plan_sha256=sha, plan_path=plan_path)
    return RunReport(run_id=run_id, counts=counts, plan_sha256=sha, plan_path=plan_path)


@dataclass(frozen=True)
class _Candidate:
    """A link the verdict rows decide, with the parent the verdict judges it against."""

    state: LinkState
    parent: LiveRecord

    @property
    def child_id(self) -> str:
        return self.state.child.memory_id

    @property
    def key(self) -> str:
        return f'{self.state.project_id}:{self.child_id}'

    def pair(self) -> LinkPair:
        return LinkPair(key=self.key, child_text=self.state.child.text, parent_text=self.parent.text)

    def judged(
        self, ledger: LinkHealLedger, adjudicate: LinkAdjudicator,
    ) -> AdjudicationRow | None:
        return ledger.adjudication_at(
            self.state.project_id, self.child_id, self.parent.memory_id,
            self.state.child.text_sha256, self.parent.text_sha256,
            model=adjudicate.model, prompt_sha256=adjudicate.prompt_sha256,
        )

    def record(self, verdict: LinkVerdict, prompt_sha256: str) -> AdjudicationRecord | None:
        if verdict.verdict is None:
            return None
        return AdjudicationRecord(
            project_id=self.state.project_id,
            child_id=self.child_id,
            parent_id=self.parent.memory_id,
            child_sha256=self.state.child.text_sha256,
            parent_sha256=self.parent.text_sha256,
            verdict=verdict.verdict,
            reason=verdict.reason,
            model=verdict.model,
            prompt_sha256=prompt_sha256,
        )


def _candidates(reads: LinkReads) -> list[_Candidate]:
    return [
        _Candidate(state=state, parent=state.parent)
        for state in reads.states
        if state.parent is not None and needs_verdict(state)
    ]


def _adjudicator_basis(adjudication_id: int, record: AdjudicationRecord) -> LinkBasis:
    return LinkBasis(
        project_id=record.project_id,
        child_id=record.child_id,
        parent_id=record.parent_id,
        verdict=record.verdict,
        child_sha256=record.child_sha256,
        parent_sha256=record.parent_sha256,
        source=BasisSource.ADJUDICATOR,
        key=str(adjudication_id),
    )


@dataclass(frozen=True)
class _Adjudicated:
    """What a plan run learned about its candidates, before anything is ledgered."""

    reused: tuple[AdjudicationRow, ...]
    records: tuple[AdjudicationRecord, ...]
    failed: int
    storms: tuple[Mapping[str, Any], ...]

    def storm_escapes(self, projects: Sequence[str]) -> tuple[Escape, ...]:
        return tuple(adjudicator_storm_escape(storm, projects=projects) for storm in self.storms)


async def _adjudicate(
    candidates: Sequence[_Candidate], ledger: LinkHealLedger, adjudicate: LinkAdjudicator,
) -> _Adjudicated:
    """Reuse each candidate's verdict by *adjudicate* at its current texts; ask about the rest."""
    reused: list[AdjudicationRow] = []
    asked: list[_Candidate] = []
    for candidate in candidates:
        row = candidate.judged(ledger, adjudicate)
        if row is None:
            asked.append(candidate)
        else:
            reused.append(row)
    storms: list[Mapping[str, Any]] = []
    verdicts = (
        await adjudicate([candidate.pair() for candidate in asked], on_failure_storm=storms.append)
        if asked
        else []
    )
    answered = (
        candidate.record(verdict, adjudicate.prompt_sha256)
        for candidate, verdict in zip(asked, verdicts, strict=True)
    )
    records = tuple(record for record in answered if record is not None)
    return _Adjudicated(
        reused=tuple(reused),
        records=records,
        failed=len(verdicts) - len(records),
        storms=tuple(storms),
    )


async def run_adjudicator_plan(
    *,
    adjudicate: LinkAdjudicator,
    store: LinkHealStore,
    census: LinkCensus,
    ledger: LinkHealLedger,
    limits: RunLimits,
    projects: Sequence[str],
    plan_path_for: Callable[[str], Path],
    source: RunSource = RunSource.ADJUDICATOR,
    enabled_actions: frozenset[HealAction] | None = None,
    created_in: CreatedWindow | None = None,
    filer: EscapeFiler | None = None,
    presence_projects: Sequence[str] = (),
) -> RunReport:
    """A non-writing run of *source* planning heals from the link adjudicator's verdicts.

    Only the links whose child *created_in* admits are read, when it is given.
    A parent is looked for in *presence_projects* as well as *projects*.
    The adjudicator is asked only about the links the verdict rows decide and
    that this adjudicator (model and prompt version) has not judged at their
    current texts, so a model or brief change asks again. Every read
    and every question is made before the run row exists. The run rests on the
    adjudications it makes and on those it reuses. A failed verdict is counted,
    never ledgered. Only heals in *enabled_actions* are staged (``None`` stages
    every one). A share over its ceiling is recorded as ``would_escape``, never
    filed; so is a failure storm, unless *filer* is given, which files it and
    discloses it as ``escaped``. A run given a *filer* is about to be applied, so
    for a source in :data:`CALIBRATION_GATED_SOURCES` it first withholds the
    pending heals whose adjudicator has no passing calibration (see
    :func:`_publish_plan`).
    """
    reads = await read_links(
        store, census, projects, created_in, presence_projects=presence_projects,
    )
    adjudicated = await _adjudicate(_candidates(reads), ledger, adjudicate)
    run_id = ledger.start_run(source, writes=False)
    new_ids = ledger.add_adjudications(run_id, adjudicated.records)
    ledger.reuse_adjudications(run_id, [row.adjudication_id for row in adjudicated.reused])
    bases = [
        *(_adjudicator_basis(row.adjudication_id, row.record) for row in adjudicated.reused),
        *(
            _adjudicator_basis(adjudication_id, record)
            for adjudication_id, record in zip(new_ids, adjudicated.records, strict=True)
        ),
    ]
    plan = plan_links(reads, bases)
    counts = replace(
        plan.counts,
        adjudications_reused=len(adjudicated.reused),
        adjudication_failed=adjudicated.failed,
    )
    storms = adjudicated.storm_escapes(projects)
    escaped = () if filer is None else tuple(
        entry for storm in storms for entry in _file(filer, storm)
    )
    return await _publish_plan(
        replace(plan, counts=counts), run_id,
        ledger=ledger, limits=limits, projects=projects, source=source,
        plan_path=plan_path_for(run_id), escapes=storms if filer is None else (),
        escaped=escaped, enabled_actions=enabled_actions,
        withhold_uncalibrated=filer is not None and source in CALIBRATION_GATED_SOURCES,
    )


def link_heal_agent_id(run_id: str) -> str:
    """The ``agent_id`` every write of run *run_id* carries."""
    return f'link-heal-{run_id[:8]}'


@dataclass(frozen=True)
class _Outcome:
    """One heal's ledger state; ``wrote`` says whether a write was attempted, and
    ``records`` holds the reads the state was decided on."""

    state: ActionState
    detail: Mapping[str, Any] | None = None
    wrote: bool = False
    records: WriteRecords | None = None


def _read_failure(phase: str, failure: StoreReadFailed) -> dict[str, Any]:
    return {
        'failure': phase,
        'tool': failure.tool,
        'memory_id': failure.memory_id,
        'error_type': failure.error_type,
        'error': failure.detail,
    }


@dataclass(frozen=True)
class _Corroboration:
    """The live reads a heal was corroborated against, and how they differ from its plan."""

    child: LiveRecord | None
    parent: LiveRecord | None
    mismatch: StaleField | None

    @property
    def records(self) -> WriteRecords:
        return WriteRecords(before_child=self.child, before_parent=self.parent)


async def _corroborate(store: LinkHealStore, heal: PlannedAction) -> _Corroboration:
    """Re-read the child, its parent when the heal rests on it, and (for a
    completion) its children, live."""
    child = await store.read(heal.project_id, heal.child_id)
    parent_id = heal.pre_image.parent_id
    parent = (
        await store.read(heal.project_id, parent_id)
        if parent_id is not None and corroboration_reads_parent(heal)
        else None
    )
    child_count = (
        await store.count_children(heal.project_id, heal.child_id)
        if heal.action in COMPLETION_ACTIONS
        else None
    )
    return _Corroboration(
        child=child,
        parent=parent,
        mismatch=corroboration_mismatch(heal, child, parent, child_count),
    )


async def _verify(
    store: LinkHealStore, project_id: str, child_id: str, image: LinkImage,
) -> tuple[StaleField | None, LiveRecord | None]:
    """Whether the record now shows *image*, and the record as read."""
    record = await store.read(project_id, child_id)
    return verification_mismatch(image, record), record


async def _write_and_verify(
    store: LinkHealStore,
    project_id: str,
    child_id: str,
    change: MetadataChange,
    after: LinkImage,
    *,
    run_id: str,
    reason: str,
    before: WriteRecords,
) -> _Outcome:
    """Send *change* as one write, then confirm the record shows *after*.

    The outcome's records are *before* plus the child the verification read.
    """
    reply = await store.write(
        project_id, child_id, change,
        agent_id=link_heal_agent_id(run_id), causation_id=run_id, reason=reason,
    )
    if not reply.landed:
        detail = {'failure': 'write', 'error_type': reply.error_type, 'error': reply.error}
        return _Outcome(ActionState.FAILED, detail, wrote=True, records=before)
    try:
        mismatch, verified = await _verify(store, project_id, child_id, after)
    except StoreReadFailed as failure:
        return _Outcome(
            ActionState.FAILED, _read_failure('verify', failure), wrote=True, records=before,
        )
    records = replace(before, after_child=verified)
    if mismatch is not None:
        detail = {'failure': 'verify', 'field': mismatch.value}
        return _Outcome(ActionState.FAILED, detail, wrote=True, records=records)
    return _Outcome(ActionState.APPLIED, wrote=True, records=records)


async def _heal_one(heal: PlannedAction, *, store: LinkHealStore, run_id: str) -> _Outcome:
    try:
        corroboration = await _corroborate(store, heal)
    except StoreReadFailed as failure:
        return _Outcome(ActionState.FAILED, _read_failure('read', failure))
    if corroboration.mismatch is not None:
        return _Outcome(
            ActionState.SKIPPED_STALE,
            {'stale_field': corroboration.mismatch.value},
            records=corroboration.records,
        )
    return await _write_and_verify(
        store, heal.project_id, heal.child_id, heal.change, heal.post_image,
        run_id=run_id,
        reason=journal_reason(run_id[:8], heal.action, heal.pre_image),
        before=corroboration.records,
    )


class ApprovalMismatch(ValueError):
    """The approved plan sha is not the sha of the plan pending now."""

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f'approved plan sha256 {expected} is not the pending plan\'s sha256 {actual}; '
            'the pending heals changed since that plan was reviewed — re-plan and review it',
        )


def _check_approval(approved: str, pending: Sequence[ActionRow]) -> None:
    actual = pending_plan_sha256(pending)
    if actual != approved:
        raise ApprovalMismatch(expected=approved, actual=actual)


@dataclass
class _Drain:
    """What one apply run's drain has come to so far; the drain is one failure run."""

    failures: KeyedStreaks[None]
    outcomes: Counter[ActionState] = field(default_factory=Counter)
    writes: int = 0
    last_failure: Mapping[str, Any] | None = None
    not_attempted: int = 0
    stopped_by: str | None = None

    @classmethod
    def for_limit(cls, limit: int) -> _Drain:
        return cls(failures=KeyedStreaks(threshold=limit))

    @property
    def streak(self) -> int:
        return self.failures.count(None)

    def record(self, outcome: _Outcome) -> bool:
        """Count *outcome*; True when this failure brings the streak to the limit.

        A failure extends the streak, an applied heal resets it, a skip leaves it.
        """
        self.outcomes[outcome.state] += 1
        if outcome.wrote:
            self.writes += 1
        if outcome.state is ActionState.FAILED:
            self.last_failure = outcome.detail
            return self.failures.failure(None).fired
        if outcome.state is ActionState.APPLIED:
            self.failures.success(None)
        return False

    def stop(self, *, not_attempted: int) -> None:
        self.stopped_by = STREAK_STOP
        self.not_attempted = not_attempted


async def _drain(
    pending: Sequence[ActionRow],
    *,
    store: LinkHealStore,
    ledger: LinkHealLedger,
    run_id: str,
    cap: int | None,
    streak_limit: int,
) -> _Drain:
    """Heal *pending* in order until *cap* write attempts or *streak_limit* failures."""
    drain = _Drain.for_limit(streak_limit)
    for index, row in enumerate(pending):
        if cap is not None and drain.writes >= cap:
            outcome = _Outcome(ActionState.SKIPPED_CAP)
        else:
            outcome = await _heal_one(row.planned, store=store, run_id=run_id)
        ledger.set_outcome(
            row.action_id, outcome.state, run_id, outcome.detail, records=outcome.records,
        )
        if drain.record(outcome):
            drain.stop(not_attempted=len(pending) - index - 1)
            break
    return drain


def _projects(pending: Sequence[ActionRow]) -> list[str]:
    return list(dict.fromkeys(row.planned.project_id for row in pending))


def _withhold_uncalibrated(
    pending: Sequence[ActionRow], *, ledger: LinkHealLedger, run_id: str,
) -> tuple[list[ActionRow], int]:
    """Withhold, under *run_id*, each heal in *pending* whose verdict came from an
    adjudicator with no passing calibration; the heals admitted, and how many were withheld."""
    admitted: list[ActionRow] = []
    withheld = 0
    for row in pending:
        calibration = ledger.basis_calibration(row)
        if calibration is None or calibration.passed:
            admitted.append(row)
            continue
        ledger.set_outcome(
            row.action_id, ActionState.WITHHELD_UNCALIBRATED, run_id, calibration.as_detail(),
        )
        withheld += 1
    return admitted, withheld


async def run_apply(
    *,
    store: LinkHealStore,
    ledger: LinkHealLedger,
    limits: RunLimits,
    filer: EscapeFiler,
    source: RunSource,
    approved_plan_sha256: str | None = None,
) -> RunReport:
    """A writing run: heal the pending rows of *source*, oldest first.

    Raises :class:`ApprovalMismatch` before any run row when an approved sha
    is not the pending plan's. For a source in :data:`CALIBRATION_GATED_SOURCES`
    it first withholds, attended or not, every heal whose verdict came from an
    adjudicator with no passing calibration; the share ceilings, the backlog
    escape and the drain then see only the heals it admitted. When the
    adjudications behind those trip a share ceiling, it files each share escape
    and writes nothing, attended or not. ``planned`` counts every pending heal
    the run took on, withheld ones included.
    """
    pending = ledger.pending_actions(source)
    attended = approved_plan_sha256 is not None
    if approved_plan_sha256 is not None:
        _check_approval(approved_plan_sha256, pending)
    run_id = ledger.start_run(source, writes=True)
    admitted, withheld = (
        _withhold_uncalibrated(pending, ledger=ledger, run_id=run_id)
        if source in CALIBRATION_GATED_SOURCES
        else (list(pending), 0)
    )
    projects = _projects(admitted)
    refusal = _share_refusal(
        admitted, taken_on=pending, withheld_uncalibrated=withheld, ledger=ledger,
        limits=limits, filer=filer, projects=projects, approved_plan_sha256=approved_plan_sha256,
    )
    if refusal is not None:
        ledger.finish_run(run_id, counts=refusal.as_json())
        return RunReport(run_id=run_id, counts=refusal)
    escaped = () if attended else _file(
        filer, backlog_escape(len(admitted), limits, projects=projects),
    )
    drain = await _drain(
        admitted, store=store, ledger=ledger, run_id=run_id,
        cap=None if attended else limits.max_actions_per_run,
        streak_limit=limits.write_failure_streak,
    )
    if drain.stopped_by is not None:
        escaped += _file(filer, write_failure_escape(
            run_id, streak=drain.streak, not_attempted=drain.not_attempted,
            last_failure=drain.last_failure, projects=projects,
        ))
    counts = _drain_counts(
        pending, drain, escaped=escaped, approved_plan_sha256=approved_plan_sha256,
        withheld_uncalibrated=withheld,
    )
    ledger.finish_run(run_id, counts=counts.as_json())
    return RunReport(run_id=run_id, counts=counts)


def _share_refusal(
    admitted: Sequence[ActionRow],
    *,
    taken_on: Sequence[ActionRow],
    withheld_uncalibrated: int,
    ledger: LinkHealLedger,
    limits: RunLimits,
    filer: EscapeFiler,
    projects: Sequence[str],
    approved_plan_sha256: str | None,
) -> RunCounts | None:
    """File the share escapes the adjudications behind *admitted* trip; ``None`` when
    none does. *taken_on* is every pending heal the run took on."""
    escapes = adjudicated_share_escapes(admitted, ledger, limits, projects=projects)
    if not escapes:
        return None
    return RunCounts(
        planned=len(taken_on),
        planned_by_action=Counter(row.planned.action.value for row in taken_on),
        not_attempted=len(admitted),
        withheld_uncalibrated=withheld_uncalibrated,
        escaped=tuple(entry for escape in escapes for entry in _file(filer, escape)),
        stopped_by=SHARE_STOP,
        approved_plan_sha256=approved_plan_sha256,
    )


def _drain_counts(
    heals: Sequence[ActionRow],
    drain: _Drain,
    *,
    escaped: tuple[Mapping[str, Any], ...] = (),
    approved_plan_sha256: str | None = None,
    withheld_uncalibrated: int = 0,
) -> RunCounts:
    """A writing run's disclosure: the heals it took on, and what became of them."""
    skipped_cap = drain.outcomes[ActionState.SKIPPED_CAP]
    return RunCounts(
        planned=len(heals),
        planned_by_action=Counter(row.planned.action.value for row in heals),
        applied=drain.outcomes[ActionState.APPLIED],
        skipped_stale=drain.outcomes[ActionState.SKIPPED_STALE],
        skipped_cap=skipped_cap,
        failed=drain.outcomes[ActionState.FAILED],
        not_attempted=drain.not_attempted,
        withheld_uncalibrated=withheld_uncalibrated,
        escaped=escaped,
        caps_bit=(CAP_BIT,) if skipped_cap else (),
        stopped_by=drain.stopped_by,
        approved_plan_sha256=approved_plan_sha256,
    )


async def _undo_step(
    heal: PlannedAction,
    change: MetadataChange,
    before: LinkImage,
    after: LinkImage,
    *,
    store: LinkHealStore,
    run_id: str,
) -> _Outcome:
    """Corroborate that the record shows *before*, write *change*, verify *after*."""
    try:
        stale, current = await _verify(store, heal.project_id, heal.child_id, before)
    except StoreReadFailed as failure:
        return _Outcome(ActionState.FAILED, _read_failure('read', failure))
    records = WriteRecords(before_child=current)
    if stale is not None:
        return _Outcome(ActionState.SKIPPED_STALE, {'stale_field': stale.value}, records=records)
    return await _write_and_verify(
        store, heal.project_id, heal.child_id, change, after,
        run_id=run_id,
        reason=journal_reason(run_id[:8], UNDO_ACTION, before),
        before=records,
    )


def _undo_start(row: ActionRow, ledger: LinkHealLedger) -> LinkImage:
    """Where an undo of *row* starts: past the steps an earlier undo already applied."""
    resumed = ledger.last_applied_undo_step(row.action_id)
    return row.planned.post_image if resumed is None else resumed.after


async def _undo_heal(
    row: ActionRow, *, store: LinkHealStore, ledger: LinkHealLedger, run_id: str,
) -> _Outcome:
    """Take one heal back step by step; it is undone only once every step applied.

    Every step is ledgered; the heal's outcome is its first step that did not apply.
    """
    heal = row.planned
    before = _undo_start(row, ledger)
    wrote = False
    for change in undo_changes(before, heal.pre_image):
        after = apply_change(before, change)
        outcome = await _undo_step(heal, change, before, after, store=store, run_id=run_id)
        ledger.add_undo_step(
            run_id, row, before=before, after=after, state=outcome.state, detail=outcome.detail,
            records=outcome.records,
        )
        wrote = wrote or outcome.wrote
        if outcome.state is not ActionState.APPLIED:
            return replace(outcome, wrote=wrote)
        before = after
    ledger.mark_undone(row.action_id, run_id)
    return _Outcome(ActionState.APPLIED, wrote=wrote)


async def run_undo(
    target: UndoTarget, *, store: LinkHealStore, ledger: LinkHealLedger, limits: RunLimits,
) -> RunReport:
    """A writing run taking back every heal *target* still has applied, newest first."""
    heals = target.heals[::-1]
    run_id = ledger.start_run(RunSource.UNDO, writes=True)
    drain = _Drain.for_limit(limits.write_failure_streak)
    for index, row in enumerate(heals):
        if drain.record(await _undo_heal(row, store=store, ledger=ledger, run_id=run_id)):
            drain.stop(not_attempted=len(heals) - index - 1)
            break
    counts = _drain_counts(heals, drain)
    ledger.finish_run(run_id, counts=counts.as_json())
    return RunReport(run_id=run_id, counts=counts)
