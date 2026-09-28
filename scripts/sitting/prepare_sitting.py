#!/usr/bin/env python3
"""Prepare a sitting: sweep, gate and number every open question, and render the brief Leo answers by number.

The seam that keeps judgement in the loop: carve-out gates 1-4 and
``pins_recovery`` arrive as agent-supplied facts from the preparation store and
default to unknown, so ``--apply-closes`` alone never makes anything closeable.
The agent must first ``record`` evidence for each gate it asserts. The script
measures only what the stores state outright: the pin markers, the DO-NOT-CLOSE
companions, and the member chain read by resolution text.

Every store of record is read, never written. The only files this writes are
the ``--ledger`` and ``--preparation`` paths it is handed; each mutation of a
store is a pre-built payload that the agent applies.

Exit codes: 0 done (a degraded store is a stated shortfall, not a failure);
2 a configuration or validation error; 3 ``resolve-answers`` must ask back.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

# Bind scripts/ and orchestrator/ to THIS checkout, never an editable install's (tasks 2881/2882).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / 'orchestrator' / 'src', _REPO_ROOT / 'scripts'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _task_db_scan import discover_project_roots  # noqa: E402
from escalation.models import SEVERITY_RANK, Escalation  # noqa: E402
from orchestrator.session_registry import (  # noqa: E402
    DecisionState,
    decisions_dir,
    fleet_root,
    list_decisions,
    sessions_dir,
)
from sitting import brief, gates, inventory, ownership, payloads  # noqa: E402
from sitting import ledger as ledgers  # noqa: E402
from sitting import preparation as preparations  # noqa: E402
from sitting.gates import CarveoutFacts, Fact, MemberOutcome, Verdict  # noqa: E402
from sitting.inventory import (  # noqa: E402
    Glossary,
    Inventory,
    OpenItem,
    Shortfall,
    cited_ids,
    key_str,
    parse_key,
)
from sitting.ownership import OwnershipFinding, SessionIndex, TaskRows  # noqa: E402
from sitting.payloads import ApplyPayload  # noqa: E402
from sitting.preparation import (  # noqa: E402
    EscalationClosed,
    GateFacts,
    Manual,
    Preparation,
    PreparationStore,
    PreparationStoreCorrupt,
    ReleasePredicate,
    Standing,
    TaskStatusIs,
)

Bucket = Literal['live', 'standing', 'closeable', 'report_only']
BUCKETS: tuple[Bucket, ...] = ('live', 'standing', 'closeable', 'report_only')
NUMBERED_BUCKETS = frozenset({'live', 'report_only'})

EXIT_OK, EXIT_CONFIG, EXIT_ASK_BACK = 0, 2, 3

CARVEOUT_ACTION = 'resume'
"""The usual C1 action for a carve-out close, per ``skills/escalation-watcher/SKILL.md::Ruled-elsewhere check (answered-but-unrecorded)``."""
CARVEOUT_RESOLVED_BY = 'sitting-preparer'
CARVEOUT_RESOLUTION_TURNS = 1
"""A carve-out close takes one preparer turn and no Leo turn."""
OWNER_RELEASE_STATUSES: tuple[str, ...] = tuple(sorted(ownership.TERMINAL_TASK_STATUSES))
PIN_RELEASE = Manual('a human spends the pin (acknowledge_declared_pins) or withdraws it')

DEFAULT_PREPARATION = _REPO_ROOT / preparations.DEFAULT_PREPARATION_PATH


@dataclass(frozen=True)
class PendingRecords:
    """A queue's root-tier pending records by task id; ``complete`` is False when any root record was unreadable."""

    by_task: Mapping[str, tuple[Escalation, ...]]
    complete: bool


@dataclass(frozen=True)
class Sources:
    """What the stores of record said, read once per run."""

    inventory: Inventory
    queues_scanned: tuple[str, ...]
    pending: Mapping[str, PendingRecords]
    sessions: SessionIndex
    task_rows: Mapping[str, TaskRows]
    handover_path: Path | None
    preparations: PreparationStore
    shortfalls: tuple[Shortfall, ...]

    def rows_for(self, item: OpenItem) -> TaskRows:
        rows = self.task_rows.get(item.project)
        if rows is None:
            return TaskRows(MappingProxyType({}), f'no project root is known for {item.project or "an unscoped item"}')
        return rows


@dataclass(frozen=True)
class Classified:
    item: OpenItem
    bucket: Bucket
    ownership: OwnershipFinding
    preparation: Preparation | None
    verdict: Verdict
    standing: Standing | None = None
    standing_source: str = ''
    payloads: tuple[ApplyPayload, ...] = ()


@dataclass(frozen=True)
class Sitting:
    generated_at: str
    queues_scanned: tuple[str, ...]
    shortfalls: tuple[Shortfall, ...]
    ledger: ledgers.Ledger
    classified: tuple[Classified, ...]
    numbered: tuple[brief.BriefEntry, ...]
    standing: tuple[brief.StandingEntry, ...]
    done: tuple[brief.DoneEntry, ...]
    closes: tuple[brief.CloseRecord, ...]
    glossary: Glossary
    docket_reason: str | None

    def number(self, item: OpenItem) -> int:
        return self.ledger.entries[key_str(item.key)].number


def read_sources(
    *,
    project_roots: Sequence[str] | None,
    decisions_root: Path,
    sessions_root: Path,
    handover_path: Path | None,
    preparation_path: Path,
    project: str | None,
    now: datetime,
) -> Sources:
    roots = discover_project_roots() if project_roots is None else list(project_roots)
    shortfalls = _queueless(roots)
    if not decisions_dir(decisions_root).is_dir():
        shortfalls.append(Shortfall('decision_registry', str(decisions_dir(decisions_root)), 'no decisions directory'))
    queues = inventory.escalation_queue_dirs(roots, list_decisions(decisions_root))
    found = inventory.collect_open_items(queue_dirs=queues, decisions_root=decisions_root, now=now, project=project)
    sessions = ownership.index_sessions(sessions_root)
    if not sessions.available:
        shortfalls.append(Shortfall('session_registry', str(sessions_root), 'no sessions root'))
    task_rows = _task_rows(queues)
    stored, problem = _load_preparations(preparation_path)
    shortfalls += [*found.shortfalls, *sessions.shortfalls, *([problem] if problem else [])]
    shortfalls += [Shortfall('tasks_db', project, rows.unavailable) for project, rows in task_rows.items()
                   if rows.unavailable]
    return Sources(
        inventory=found,
        queues_scanned=queues,
        pending=MappingProxyType({q: _pending_records(q, index) for q, index in found.escalation_index.items()}),
        sessions=sessions,
        task_rows=task_rows,
        handover_path=handover_path,
        preparations=stored,
        shortfalls=tuple(shortfalls),
    )


def gather_facts(item: OpenItem, sources: Sources, preparation: Preparation | None) -> CarveoutFacts:
    """The agent-supplied gate facts (unknown unless recorded) merged with what the stores state outright."""
    agent = preparation.gate_facts if preparation is not None else GateFacts()
    members, sideways = _member_chain(item, sources.inventory)
    return CarveoutFacts(
        escalation_id=item.escalation_id or '',
        ruling=agent.ruling,
        names_this_record=agent.names_this_record,
        executed=agent.executed,
        session_terminated=agent.session_terminated,
        pins_recovery=agent.pins_recovery,
        pin_declared_by=item.pin_declared_by,
        root_cause=item.root_cause,
        do_not_close_companions=_do_not_close_companions(item, sources.pending),
        sideways=sideways,
        members=members,
    )


def classify(
    sources: Sources, *, recommend_only: bool, ledger_standing: Mapping[str, Standing], now: datetime,
) -> tuple[Classified, ...]:
    """Sweep for an owner, then run the six gates, then place each item in exactly one bucket."""
    classified = []
    for item in sources.inventory.items:
        finding = ownership.sweep(
            item, sessions=sources.sessions, task_rows=sources.rows_for(item),
            handover_path=sources.handover_path, now=now,
        )
        prep = sources.preparations.get(item.key)
        facts = gather_facts(item, sources, prep)
        verdict = gates.evaluate_carveout(facts, recommend_only=recommend_only)
        standing, source = _standing(item, finding, prep, ledger_standing.get(key_str(item.key)))
        bucket = _bucket(standing, verdict, facts)
        classified.append(Classified(
            item, bucket, finding, prep, verdict, standing, source,
            _close_payloads(item, verdict) if bucket == 'closeable' else (),
        ))
    return tuple(classified)


def reprobe_release_predicate(
    predicate: ReleasePredicate, item: OpenItem, sources: Sources, now: datetime,
) -> brief.ReleaseProbe | None:
    """Measure a machine-checkable release now; a Manual one is not probeable and yields None."""
    measured_at = now.isoformat()
    if isinstance(predicate, Manual):
        return None
    if isinstance(predicate, TaskStatusIs):
        rows = sources.rows_for(item)
        if rows.unavailable:
            return brief.ReleaseProbe(f'unmeasured: {rows.unavailable}', False, measured_at)
        row = rows.rows.get(predicate.task_id)
        observed = row.status if row is not None else 'not in the store'
        return brief.ReleaseProbe(observed, observed in predicate.statuses, measured_at)
    esc = _read_in_queue(sources.inventory, item.queue_dir, predicate.esc_id)
    if esc is None:
        return brief.ReleaseProbe(f'not readable in {item.queue_dir or "any scanned queue"}', False, measured_at)
    return brief.ReleaseProbe(esc.status, esc.status in gates.TERMINAL_STATUSES, measured_at)


def compose(
    sources: Sources, ledger: ledgers.Ledger, *, recommend_only: bool, multi_sitting: bool, now: datetime,
) -> Sitting:
    """Number every open item in *ledger*, classify it, and build every part the brief renders."""
    by_key = {key_str(item.key): item for item in sources.inventory.items}
    numbered_ledger = ledgers.assign(ledger, by_key, lambda key: _urgency(by_key[key]), now.isoformat())
    ledger_standing = {key: entry.standing for key, entry in numbered_ledger.in_state('standing') if entry.standing}
    classified = classify(sources, recommend_only=recommend_only, ledger_standing=ledger_standing, now=now)

    def number(item: OpenItem) -> int:
        return numbered_ledger.entries[key_str(item.key)].number

    entries = tuple(
        brief.BriefEntry(number(c.item), c.item, c.preparation, c.ownership)
        for c in classified if c.bucket in NUMBERED_BUCKETS
    )
    standing = tuple(
        brief.StandingEntry(number(c.item), c.item, c.standing,
                            reprobe_release_predicate(c.standing.release_predicate, c.item, sources, now))
        for c in classified if c.standing is not None
    )
    done = tuple(
        brief.DoneEntry(entry.number, parse_key(key), entry.done_at or '')
        for key, entry in numbered_ledger.in_state('done')
    )
    closes = tuple(
        brief.CloseRecord(c.item, c.verdict, c.payloads)
        for c in sorted(classified, key=lambda c: number(c.item))
        if c.bucket == 'closeable' or (c.bucket == 'report_only' and c.verdict.demoted_by == gates.RECOMMEND_ONLY)
    )
    glossary = _glossary(sources, classified, done)
    return Sitting(
        generated_at=now.isoformat(),
        queues_scanned=sources.queues_scanned,
        shortfalls=(*sources.shortfalls, *glossary.shortfalls),
        ledger=numbered_ledger,
        classified=classified,
        numbered=entries,
        standing=standing,
        done=done,
        closes=closes,
        glossary=glossary,
        docket_reason=brief.docket_reason(entries, multi_sitting=multi_sitting),
    )


def render_text(sitting: Sitting) -> str:
    parts = [brief.render_brief(
        sitting.numbered, sitting.standing, sitting.done, glossary=sitting.glossary, generated_at=sitting.generated_at,
    )]
    if sitting.docket_reason:
        parts.append(f'docket page recommended: {sitting.docket_reason}\n')
    if sitting.closes:
        parts.append(brief.render_closes(sitting.closes, glossary=sitting.glossary))
    missed = [c for c in sitting.classified if c.bucket == 'report_only' and c.verdict.missed_gates]
    if missed:
        parts.append(_report_only_section(sitting, missed))
    parts.append(_shortfall_section(sitting))
    return '\n'.join(parts)


def classification_json(sitting: Sitting) -> dict[str, Any]:
    """The classification ``return_brief`` consumes, so it never re-runs the gates."""
    releases = {key_str(entry.item.key): entry.release for entry in sitting.standing}
    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
    for c in sorted(sitting.classified, key=lambda c: sitting.number(c.item)):
        buckets[c.bucket].append(_classified_row(c, sitting.number(c.item), releases.get(key_str(c.item.key))))
    return {
        'generated_at': sitting.generated_at,
        'queues_scanned': list(sitting.queues_scanned),
        'shortfalls': [asdict(shortfall) for shortfall in sitting.shortfalls],
        **buckets,
        'done': [_done_row(entry) for entry in sitting.done],
        'docket_recommended': sitting.docket_reason,
    }


NIGHTLY_COMMANDS = frozenset({'brief', 'record'})


def nightly_confinement_breach(args: argparse.Namespace) -> str | None:
    """Why *args* would write or close outside what an unattended run may touch, or None."""
    if args.command not in NIGHTLY_COMMANDS:
        return f'{args.command!r} is not a nightly subcommand ({", ".join(sorted(NIGHTLY_COMMANDS))})'
    if args.preparation.resolve() != DEFAULT_PREPARATION.resolve():
        return f'--preparation must be the default store {DEFAULT_PREPARATION}'
    if getattr(args, 'apply_closes', False):
        return '--apply-closes is never available to the unattended run'
    if getattr(args, 'ledger', None) is not None:
        return '--ledger is never available to the unattended run'
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.environ.get(preparations.NIGHTLY_CONFINEMENT_ENV) and (breach := nightly_confinement_breach(args)):
        print(f'refused under {preparations.NIGHTLY_CONFINEMENT_ENV}: {breach}', file=sys.stderr)
        return EXIT_CONFIG
    try:
        return args.handler(args)
    except ledgers.LedgerCorrupt as exc:
        print(exc, file=sys.stderr)
        return EXIT_CONFIG


def _queueless(roots: Iterable[str]) -> list[Shortfall]:
    return [
        Shortfall('escalation_queue', str(Path(root) / inventory.QUEUE_SUBDIRS[0]),
                  'no escalation queue under this project root')
        for root in roots if not any((Path(root) / subdir).is_dir() for subdir in inventory.QUEUE_SUBDIRS)
    ]


def _task_rows(queues: Iterable[str]) -> Mapping[str, TaskRows]:
    roots = {inventory.queue_project(q): root for q in queues if (root := inventory.queue_project_root(q)) is not None}
    return MappingProxyType({project: ownership.load_task_rows(root) for project, root in roots.items()})


def _load_preparations(path: Path) -> tuple[PreparationStore, Shortfall | None]:
    try:
        return preparations.load(path), None
    except PreparationStoreCorrupt as exc:
        return PreparationStore(), Shortfall('preparation', str(exc.path), exc.reason)


def _pending_records(queue_dir: str, index: Mapping[str, Path]) -> PendingRecords:
    by_task: dict[str, list[Escalation]] = defaultdict(list)
    complete = True
    for path in index.values():
        if path.parent != Path(queue_dir):
            continue
        esc, problem = inventory.read_escalation(path)
        complete = complete and problem is None
        if esc is not None and esc.status == 'pending':
            by_task[esc.task_id].append(esc)
    return PendingRecords(MappingProxyType({task: tuple(found) for task, found in by_task.items()}), complete)


def _read_in_queue(found: Inventory, queue_dir: str, esc_id: str) -> Escalation | None:
    path = found.escalation_index.get(queue_dir, {}).get(esc_id)
    return inventory.read_escalation(path)[0] if path is not None else None


def _member_chain(item: OpenItem, found: Inventory) -> tuple[tuple[MemberOutcome, ...], Fact]:
    """Each member located across the archive and read by resolution text, and the gate-6 sideways finding."""
    if item.kind != 'esc' or item.queue_dir not in found.escalation_index:
        return (), Fact(None, 'not a pending escalation in a scanned queue; the sideways check cannot run')
    outcomes, unreadable = [], []
    for member_id in item.members:
        member = _read_in_queue(found, item.queue_dir, member_id)
        if member is None:
            unreadable.append(member_id)
        else:
            outcomes.append(MemberOutcome(member_id, member.status, member.resolution or ''))
    twins = sorted(
        other.escalation_id or '' for other in found.items
        if other.kind == 'esc' and other.queue_dir == item.queue_dir and other.key != item.key
        and set(other.members) & set(item.members)
    )
    if unreadable:
        return tuple(outcomes), Fact(None, f'members {unreadable} are not readable in {item.queue_dir}')
    if twins:
        return tuple(outcomes), Fact(False, f'pending L2s {twins} share a member; disposition them in this sitting')
    chain = ', '.join(f'{m.member_id} {m.status} ({gates.classify_member(m.status, m.resolution)})'
                      for m in outcomes) or 'no members'
    return tuple(outcomes), Fact(True, f'{chain}; no other pending L2 in {item.queue_dir} shares a member')


def _do_not_close_companions(item: OpenItem, pending: Mapping[str, PendingRecords]) -> tuple[str, ...] | None:
    records = pending.get(item.queue_dir)
    if item.kind != 'esc' or not item.task_id or records is None or not records.complete:
        return None
    return tuple(
        esc.id for esc in records.by_task.get(item.task_id, ())
        if esc.id != item.escalation_id and esc.root_cause.startswith(gates.VETO_PIN_PREFIX)
    )


def _standing(
    item: OpenItem, finding: OwnershipFinding, prep: Preparation | None, recorded: Standing | None,
) -> tuple[Standing | None, str]:
    """The item's standing and its source, most explicit first: the session ledger, the preparation, then the stores."""
    if recorded is not None:
        return recorded, 'ledger'
    if prep is not None and prep.standing is not None:
        return prep.standing, 'preparation'
    if item.pin_declared_by or item.root_cause.startswith(gates.VETO_PIN_PREFIX):
        return Standing(
            'pin', ', '.join(item.pin_declared_by) or item.root_cause, PIN_RELEASE,
            f'pin_declared_by={list(item.pin_declared_by)} root_cause={item.root_cause!r}',
        ), 'pin'
    if finding.owned:
        owner = finding.owners[0]
        release: ReleasePredicate = (
            TaskStatusIs(owner.owner_task_id, OWNER_RELEASE_STATUSES) if owner.owner_task_id
            else Manual(f'{owner.owner} finishes or hands the question back')
        )
        return Standing('owned', owner.owner, release, f'{owner.probe}: {owner.evidence or owner.owner}'), 'ownership'
    return None, ''


def _bucket(standing: Standing | None, verdict: Verdict, facts: CarveoutFacts) -> Bucket:
    if standing is not None:
        return 'standing'
    if verdict.disposition == 'closeable':
        return 'closeable'
    ruled_member = any(gates.classify_member(m.status, m.resolution) == 'ruled' for m in facts.members)
    return 'report_only' if facts.ruling.held is True or ruled_member else 'live'


def _close_payloads(item: OpenItem, verdict: Verdict) -> tuple[ApplyPayload, ...]:
    """``resolve_issue`` first, so a server refusal aborts before the registry records an answer."""
    close = payloads.close_decision_argv(item, DecisionState.ANSWERED, gates.closing_evidence(verdict))
    if not item.escalation_id:
        return (close,)
    resolution = (
        'Closed under the ruled-elsewhere carve-out; all six gates held.\n'
        f'Ruling: {verdict.gate("ruling_is_leos_own").evidence}\n'
        f'Executed: {verdict.gate("ruling_was_executed").evidence}'
    )
    resolve = payloads.resolve_issue_payload(
        item.escalation_id, resolution, CARVEOUT_ACTION,
        resolved_by=CARVEOUT_RESOLVED_BY, resolution_turns=CARVEOUT_RESOLUTION_TURNS,
    )
    return (resolve, close)


def _urgency(item: OpenItem) -> tuple[int, float]:
    return (-SEVERITY_RANK.get(item.severity, 0), -(item.age_days or 0.0))


def _glossary(sources: Sources, classified: Iterable[Classified], done: Iterable[brief.DoneEntry]) -> Glossary:
    """Gloss every id any rendered part cites, each in its item's own queue and project."""
    esc_wanted: dict[str, set[str]] = defaultdict(set)
    task_wanted: dict[str, set[str]] = defaultdict(set)
    for c in classified:
        esc_ids, task_ids = _rendered_citations(c)
        esc_wanted[c.item.queue_dir] |= esc_ids
        task_wanted[c.item.project] |= task_ids
    for entry in done:
        if entry.key[0] == 'esc':
            esc_wanted[entry.key[1]].add(entry.key[-1])
    return inventory.gloss(esc_wanted, task_wanted, sources.inventory.escalation_index)


def _rendered_citations(c: Classified) -> tuple[set[str], set[str]]:
    item_esc, item_tasks = inventory.item_citations(c.item)
    esc_ids, task_ids = set(item_esc), set(item_tasks)
    texts = [
        *(text for probe in c.ownership.probes for text in (probe.owner, probe.evidence)),
        *(text for gate in c.verdict.gates for text in (gate.evidence, gate.note)),
        *(json.dumps(payload.args, ensure_ascii=False) for payload in c.payloads),
    ]
    for standing in (c.standing, c.preparation.standing if c.preparation else None):
        if standing is not None:
            texts += [standing.owner, standing.evidence]
            _add_release_ids(standing.release_predicate, esc_ids, task_ids)
    if c.preparation is not None:
        texts += _preparation_texts(c.preparation)
        esc_ids |= set(c.preparation.cites.escalation_ids)
        task_ids |= set(c.preparation.cites.task_ids)
    for text in texts:
        cited_esc, cited_tasks = cited_ids(text)
        esc_ids |= cited_esc
        task_ids |= cited_tasks
    return esc_ids, task_ids


def _preparation_texts(prep: Preparation) -> list[str]:
    recommendation = prep.recommendation
    reasoning = (recommendation.reason if isinstance(recommendation, preparations.NoLean)
                 else recommendation.evidence_chain)
    return [prep.question, prep.on_apply, reasoning,
            *(text for option in prep.options for text in (option.text, option.ramification))]


def _add_release_ids(predicate: ReleasePredicate, esc_ids: set[str], task_ids: set[str]) -> None:
    if isinstance(predicate, TaskStatusIs):
        task_ids.add(predicate.task_id)
    elif isinstance(predicate, EscalationClosed):
        esc_ids.add(predicate.esc_id)
    else:
        esc_ids |= cited_ids(predicate.text)[0]


def _report_only_section(sitting: Sitting, missed: Sequence[Classified]) -> str:
    lines = ['## Carve-out not met (report-only)', '']
    for c in sorted(missed, key=lambda c: sitting.number(c.item)):
        label = _cite_item(c.item, sitting.glossary)
        lines.append(f'- **{sitting.number(c.item)}.** {label} — missed: {", ".join(c.verdict.missed_gates)}')
    return '\n'.join(lines) + '\n'


def _cite_item(item: OpenItem, glossary: Glossary) -> str:
    if item.kind != 'esc' or not item.escalation_id:
        return f'decision {item.decision_id}'
    return brief.cite(brief.Citation('esc', item.escalation_id, item.queue_dir), glossary)


def _shortfall_section(sitting: Sitting) -> str:
    lines = ['## Shortfalls', '']
    lines += [f'- {s.source} {s.path}: {s.reason}' for s in sitting.shortfalls] or ['None: every store read cleanly.']
    lines += ['', f'queues scanned: {", ".join(sitting.queues_scanned) or "none"}']
    return '\n'.join(lines) + '\n'


def _classified_row(c: Classified, number: int, release: brief.ReleaseProbe | None) -> dict[str, Any]:
    item = c.item
    row: dict[str, Any] = {
        'number': number,
        'key': list(item.key),
        'escalation_id': item.escalation_id,
        'decision_id': item.decision_id,
        'project': item.project,
        'task_id': item.task_id,
        'queue_dir': item.queue_dir,
        'prepared': c.preparation is not None,
    }
    if c.standing is not None:
        row.update(source=c.standing_source, standing=preparations.standing_to_json(c.standing),
                   release=asdict(release) if release is not None else None)
    if c.bucket in ('closeable', 'report_only'):
        row.update(gates=[asdict(gate) for gate in c.verdict.gates], missed_gates=list(c.verdict.missed_gates),
                   demoted_by=c.verdict.demoted_by)
    if c.bucket == 'closeable':
        row['payloads'] = [{'tool': payload.tool, 'args': payload.args} for payload in c.payloads]
    return row


def _done_row(entry: brief.DoneEntry) -> dict[str, Any]:
    kind, record_id = entry.key[0], entry.key[-1]
    return {
        'number': entry.number,
        'key': list(entry.key),
        'escalation_id': record_id if kind == 'esc' else None,
        'decision_id': record_id if kind == 'decision' else None,
        'done_at': entry.done_at,
    }


def _now(args: argparse.Namespace) -> datetime:
    return args.now or datetime.now(UTC)


def handover_file(explicit: Path | None) -> Path | None:
    """*explicit*, else this checkout's newest handover; None unless it is a file."""
    handover = explicit if explicit is not None else ownership.resolve_handover_path(_REPO_ROOT)
    return handover if handover is not None and handover.is_file() else None


def _sources(args: argparse.Namespace) -> Sources:
    return read_sources(
        project_roots=args.project_roots,
        decisions_root=fleet_root(args.decisions_root),
        sessions_root=args.sessions_root if args.sessions_root is not None else sessions_dir(),
        handover_path=handover_file(args.handover),
        preparation_path=args.preparation,
        project=args.project,
        now=_now(args),
    )


def _cmd_brief(args: argparse.Namespace) -> int:
    now = _now(args)
    base = (ledgers.load(args.ledger) if args.ledger else None) or ledgers.new_sitting(now.isoformat())
    sitting = compose(_sources(args), base, recommend_only=not args.apply_closes,
                      multi_sitting=args.multi_sitting, now=now)
    if args.ledger:
        ledgers.save(args.ledger, sitting.ledger)
    if args.json:
        print(json.dumps(classification_json(sitting), indent=2, ensure_ascii=False))
    elif args.docket_json:
        print(json.dumps(brief.docket_rows(sitting.numbered), indent=2, ensure_ascii=False))
    else:
        print(render_text(sitting), end='')
    return EXIT_OK


def _cmd_new_sitting(args: argparse.Namespace) -> int:
    seed = ledgers.load(args.seed) if args.seed else None
    sitting = ledgers.new_sitting(_now(args).isoformat(), seed=seed)
    ledgers.save(args.ledger, sitting)
    carried = f'{len(sitting.entries)} item(s) carried from {args.seed}' if seed else 'no seed'
    print(f'{sitting.sitting_id} at {args.ledger}: {carried}')
    return EXIT_OK


def _cmd_record(args: argparse.Namespace) -> int:
    try:
        text = sys.stdin.read() if args.source == '-' else Path(args.source).read_text(encoding='utf-8')
        data = json.loads(text)
        stored = preparations.record(args.preparation, data if isinstance(data, list) else [data])
    except (OSError, TypeError, ValueError, PreparationStoreCorrupt) as exc:
        print(f'record refused, nothing written: {exc}', file=sys.stderr)
        return EXIT_CONFIG
    print(f'recorded; {len(stored.entries)} preparation(s) in {args.preparation}')
    return EXIT_OK


def _cmd_resolve_answers(args: argparse.Namespace) -> int:
    current = ledgers.load(args.ledger)
    if current is None:
        print(f'no sitting at {args.ledger}; run new-sitting first', file=sys.stderr)
        return EXIT_CONFIG
    now = _now(args)
    sitting = compose(_sources(args), current, recommend_only=True, multi_sitting=False, now=now)
    working = sitting.ledger
    for entry in sitting.standing:
        if working.entries[key_str(entry.item.key)].state != 'standing':
            working = ledgers.set_standing(working, key_str(entry.item.key), entry.standing)
    options = {key_str(entry.item.key): brief.options_by_label(entry) for entry in sitting.numbered}
    answers = [_answer(token, args.source, now.isoformat()) for token in args.answer]
    answered, resolved = ledgers.resolve_answers(working, answers, options)
    persisted = _with_answer_rounds(sitting.ledger, answered)
    ledgers.save(args.ledger, persisted)
    print(ledgers.render_echo_table(resolved), end='')
    for row in resolved.rows:
        print(f'item {row.number} resolution_turns={ledgers.resolution_turns(persisted, row.key)}')
    return EXIT_ASK_BACK if resolved.unresolved else EXIT_OK


def _cmd_summary(args: argparse.Namespace) -> int:
    current = ledgers.load(args.ledger)
    if current is None:
        print(f'no sitting at {args.ledger}', file=sys.stderr)
        return EXIT_CONFIG
    print(json.dumps(asdict(ledgers.sitting_summary(current)), indent=2))
    return EXIT_OK


def _answer(token: str, source: str, answered_at: str) -> ledgers.Answer:
    """One ``N=OPTION[:note]`` token; anything else keeps the whole token as its item ref, so it is asked back."""
    item_ref, has_option, rest = token.partition('=')
    if not has_option:
        return ledgers.Answer(token, '', '', answered_at, source)
    option_ref, _, note = rest.partition(':')
    return ledgers.Answer(item_ref, option_ref, note, answered_at, source)


def _with_answer_rounds(base: ledgers.Ledger, answered: ledgers.Ledger) -> ledgers.Ledger:
    """*base* carrying *answered*'s answer bookkeeping, without the in-memory standings used to resolve."""
    return replace(base, entries={
        key: replace(entry, answer_rounds=answered.entries[key].answer_rounds,
                     first_answered_at=answered.entries[key].first_answered_at,
                     last_answered_at=answered.entries[key].last_answered_at)
        for key, entry in base.entries.items()
    })


def parse_instant(text: str) -> datetime:
    parsed = inventory.parse_stamp(text)
    if parsed is None:
        raise argparse.ArgumentTypeError(f'{text!r} is not an ISO-8601 timestamp')
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='prepare_sitting.py', description='Prepare a sitting brief Leo answers by number.')
    commands = parser.add_subparsers(dest='command', required=True)
    clock = argparse.ArgumentParser(add_help=False)
    clock.add_argument('--now', type=parse_instant, help='ISO-8601 instant to run at (default: now)')
    stores = argparse.ArgumentParser(add_help=False, parents=[clock])
    stores.add_argument('--project-root', action='append', dest='project_roots',
                        help='repeatable; default: _task_db_scan.discover_project_roots()')
    stores.add_argument('--decisions-root', type=Path, help='fleet root holding decisions/ (default: fleet_root())')
    stores.add_argument('--sessions-root', type=Path, help='default: session_registry.sessions_dir()')
    stores.add_argument('--handover', type=Path, help='default: ownership.resolve_handover_path(<this checkout>)')
    stores.add_argument('--preparation', type=Path, default=DEFAULT_PREPARATION)
    stores.add_argument('--project', help='keep only this project, folded through normalize_project_token')

    brief_cmd = commands.add_parser('brief', parents=[stores], help='print the sitting brief')
    brief_cmd.add_argument('--ledger', type=Path, help='session ledger: stable numbers, written back')
    brief_cmd.add_argument('--apply-closes', action='store_true', help='let all-six-gate items be closeable')
    brief_cmd.add_argument('--multi-sitting', action='store_true', help='the decisions span more than one sitting')
    output = brief_cmd.add_mutually_exclusive_group()
    output.add_argument('--json', action='store_true', help='emit the classification instead')
    output.add_argument('--docket-json', action='store_true', help='emit the docket rows instead')
    brief_cmd.set_defaults(handler=_cmd_brief)

    new = commands.add_parser('new-sitting', parents=[clock], help='create or reset a session ledger')
    new.add_argument('--ledger', type=Path, required=True)
    new.add_argument('--seed', type=Path, help='the nightly ledger whose open numbers carry forward')
    new.set_defaults(handler=_cmd_new_sitting)

    record = commands.add_parser('record', help='validate and merge preparations')
    record.add_argument('--preparation', type=Path, default=DEFAULT_PREPARATION)
    record.add_argument('--from', dest='source', required=True, help="a JSON file of preparations, or '-' for stdin")
    record.set_defaults(handler=_cmd_record)

    resolve = commands.add_parser('resolve-answers', parents=[stores], help="resolve Leo's numbered answers")
    resolve.add_argument('--ledger', type=Path, required=True)
    resolve.add_argument('--answer', action='append', required=True, help="repeatable 'N=OPTION[:note]' token")
    resolve.add_argument('--source', choices=sorted(ledgers.ANSWER_SOURCES), default='terminal')
    resolve.set_defaults(handler=_cmd_resolve_answers)

    summary = commands.add_parser('summary', help='print the sitting-time instrument inputs')
    summary.add_argument('--ledger', type=Path, required=True)
    summary.set_defaults(handler=_cmd_summary)
    return parser


if __name__ == '__main__':
    sys.exit(main())
