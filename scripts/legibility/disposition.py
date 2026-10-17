"""How a verified census finding becomes work, and how that work links back to
its codebook entry: docs/quality-findings-contract.md §6 (route), §8 (the
dedup protocol before filing) and the C1 ``filed_tickets`` back-links
(plans/census-incremental-prd.md L4).
"""
from __future__ import annotations

import copy
import enum
import logging
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Bound as codebook.py binds it: no module-level assignment precedes the
# imports below, so they need no E402 suppression.
if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from legibility import codebook, task_store
from shared.toolcall_markup import escape_envelope_literals

logger = logging.getLogger("legibility.disposition")

CENSUS_SOURCE = "legibility_census"
FINDING_KEY_METADATA = "x_finding_key"
ACCEPTANCE_REASON_METADATA = "x_acceptance_reason"
SEMANTIC_SCORE_THRESHOLD = 0.6
RESOLVE_BUDGET_SECS = 300.0
RESOLVE_CALL_CAP_SECS = 115.0
LIVE_STATUSES = frozenset({"pending", "in-progress", "blocked", "deferred", "merge-deferred"})
_DONE_STATUS = "done"
_CANCELLED_STATUS = "cancelled"


def task_title(cluster_title: str) -> str:
    """The title a census task is filed under for a cluster titled *cluster_title*."""
    return escape_envelope_literals(f"[legibility census] {cluster_title}")


class Step(enum.StrEnum):
    """The contract §8 step that decided a finding."""

    KEY = "key"
    SEMANTIC = "semantic"
    CURATOR = "curator"


class Resolution(enum.StrEnum):
    CREATED = "created"
    COMBINED = "combined"
    REFUSED = "refused"
    FAILED = "failed"
    EXISTING = "existing"


class Standing(enum.StrEnum):
    """What a task found for a finding says about it (contract §7, §8 step 1)."""

    FILED = "filed"
    ACCEPTED = "accepted"
    DONE = "done"


def _require_the_codebook_vocabulary(name: str, stated: Iterable[str], required: Iterable[str]) -> None:
    if sorted(stated) != sorted(required):
        raise RuntimeError(
            f"disposition.{name} must state exactly codebook's {sorted(required)}, got {sorted(stated)}"
        )


_RESOLUTION_OF = {
    task_store.TicketStatus.CREATED: Resolution.CREATED,
    task_store.TicketStatus.COMBINED: Resolution.COMBINED,
    task_store.TicketStatus.REFUSED: Resolution.REFUSED,
    task_store.TicketStatus.FAILED: Resolution.FAILED,
}

_require_the_codebook_vocabulary("Step", (step.value for step in Step), codebook.FILED_TICKET_STEPS)
_require_the_codebook_vocabulary(
    "Resolution", (resolution.value for resolution in Resolution), codebook.FILED_TICKET_RESOLUTIONS,
)


def ticket_id_from_submit_result(result: object) -> str | None:
    """The ticket id one curator-path ``submit_task`` call answered with, or
    ``None`` if it filed nothing usable.

    The seam's contract (``fused_memory/server/tools.py::submit_task``,
    ``fused_memory/middleware/task_interceptor.py::TaskInterceptor``): success
    is ``{"ticket": "tkt_<id>"}``, a TICKET id; ``resolve_ticket`` later yields
    the task id. A rejection (``{"error": ..., "error_type": ...}``), a
    non-dict, or a missing, empty or non-string ticket gives ``None``.
    """
    if not isinstance(result, dict):
        return None
    ticket_id = result.get("ticket")
    return ticket_id if isinstance(ticket_id, str) and ticket_id else None


@dataclass(frozen=True)
class Filing:
    """One verified finding the census would file, with the curator payload
    that files it and the codebook entry its back-link lands on."""

    entry_id: str | None
    finding_key: str | None
    title: str
    statement: str
    payload: Mapping[str, Any]

    @property
    def project_root(self) -> str:
        return self.payload["project_root"]

    @property
    def query(self) -> str:
        return "\n".join(dict.fromkeys(text for text in (self.title, self.statement) if text))


@dataclass(frozen=True)
class BackLink:
    """One C1 ``filed_tickets`` row, held to ``codebook.filed_ticket_errors``."""

    entry_id: str
    ticket: str | None
    project_root: str
    run_id: str
    finding_key: str | None
    task_id: int | None
    resolution: Resolution | None
    step: Step

    def __post_init__(self) -> None:
        errors = codebook.filed_ticket_errors(self.to_row())
        if errors:
            raise ValueError(f"back-link for entry {self.entry_id!r}: {'; '.join(errors)}")

    def to_row(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "project_root": self.project_root,
            "run_id": self.run_id,
            "finding_key": self.finding_key,
            "task_id": self.task_id,
            "resolution": None if self.resolution is None else self.resolution.value,
            "step": self.step.value,
        }


@dataclass(frozen=True)
class Decision:
    """What the dedup protocol decided for one Filing, and at which step."""

    filing: Filing
    step: Step
    standing: Standing | None = None
    ticket: str | None = None
    task_id: int | None = None
    resolution: Resolution | None = None
    note: str | None = None

    @property
    def routed(self) -> bool:
        """Work exists for the finding: a task, or a ticket the curator holds."""
        return self.ticket is not None or self.task_id is not None

    def disposition(self) -> str:
        """The contract §7 disposition this decision gives the finding."""
        if self.standing is Standing.ACCEPTED:
            return f"accepted:{self.task_id}"
        if self.task_id is not None:
            return f"filed:{self.task_id}"
        if self.resolution is not None:
            return f"open (ticket {self.ticket} {self.resolution.value})"
        if self.ticket is not None:
            return f"open (ticket {self.ticket} pending)"
        return "open"

    def with_outcome(self, outcome: task_store.TicketOutcome) -> Decision:
        """This decision once its curator ticket reached *outcome*."""
        resolution = _RESOLUTION_OF[outcome.status]
        unfiled_reason = None if outcome.task_id is not None else f"{resolution.value}: {outcome.reason}"
        return replace(
            self,
            resolution=resolution,
            task_id=outcome.task_id,
            standing=None if outcome.task_id is None else Standing.FILED,
            note=_notes(self.note, unfiled_reason),
        )

    def back_link(self, run_id: str) -> BackLink | None:
        if self.filing.entry_id is None or not self.routed:
            return None
        return BackLink(
            entry_id=self.filing.entry_id,
            ticket=self.ticket,
            project_root=self.filing.project_root,
            run_id=run_id,
            finding_key=self.filing.finding_key,
            task_id=self.task_id,
            resolution=self.resolution,
            step=self.step,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "entry_id": self.filing.entry_id,
            "finding_key": self.filing.finding_key,
            "title": self.filing.title,
            "project_root": self.filing.project_root,
            "step": self.step.value,
            "standing": None if self.standing is None else self.standing.value,
            "ticket": self.ticket,
            "task_id": self.task_id,
            "resolution": None if self.resolution is None else self.resolution.value,
            "note": self.note,
            "disposition": self.disposition(),
        }


@dataclass(frozen=True)
class Disposal:
    """The protocol's answer for a run's filings. *unfiled* holds every filing
    that reached no decision: all of them when step 1 was *unreachable*, and
    the curator-bound ones under a dry run."""

    decisions: tuple[Decision, ...]
    unfiled: tuple[Filing, ...]
    unreachable: str | None = None

    def resolved(self, resolutions: Resolutions) -> Disposal:
        """This disposal with each decision whose ticket resolved carrying its outcome."""
        outcomes = {resolved.ticket: resolved.outcome for resolved in resolutions.resolved}
        return replace(self, decisions=tuple(
            decision.with_outcome(outcomes[decision.ticket]) if decision.ticket in outcomes else decision
            for decision in self.decisions
        ))


@dataclass(frozen=True)
class _Found:
    standing: Standing
    task_id: int


_PRECEDENCE = (Standing.FILED, Standing.ACCEPTED, Standing.DONE)


def _standing_of(hit: task_store.TaskHit, *, store: task_store.TaskStore, project_root: str) -> Standing | None:
    if hit.status in LIVE_STATUSES:
        return Standing.FILED
    if hit.status == _DONE_STATUS:
        return Standing.DONE
    if hit.status == _CANCELLED_STATUS:
        accepting = store.get_task(project_root, hit.task_id).metadata.get(ACCEPTANCE_REASON_METADATA)
        return Standing.ACCEPTED if accepting else None
    return None


def _key_lookup(filing: Filing, store: task_store.TaskStore) -> _Found | None:
    """Contract §8 step 1. Raises StoreUnreachable when it cannot be performed."""
    if filing.finding_key is None:
        return None
    hits = store.find_tasks_by_metadata(filing.project_root, key=FINDING_KEY_METADATA, value=filing.finding_key)
    found = [
        _Found(standing, hit.task_id)
        for hit in hits
        if (standing := _standing_of(hit, store=store, project_root=filing.project_root)) is not None
    ]
    if not found:
        return None
    return min(found, key=lambda each: (_PRECEDENCE.index(each.standing), -each.task_id))


@dataclass(frozen=True)
class _SemanticMatch:
    task_id: int | None
    note: str | None = None


def _semantic_lookup(filing: Filing, store: task_store.TaskStore) -> _SemanticMatch:
    """Contract §8 step 2: the top plausible match, read back with get_task;
    only a live task stops the filing."""
    try:
        hits = store.search_tasks(filing.project_root, query=filing.query, score_threshold=SEMANTIC_SCORE_THRESHOLD)
        if not hits:
            return _SemanticMatch(None)
        record = store.get_task(filing.project_root, max(hits, key=lambda hit: hit.score).task_id)
    except task_store.StoreUnreachable as exc:
        logger.warning("disposition: §8 step 2 could not run for %r: %s", filing.title, exc)
        return _SemanticMatch(None, f"§8 step 2 could not run: {exc}")
    return _SemanticMatch(record.task_id if record.status in LIVE_STATUSES else None)


def _notes(*notes: str | None) -> str | None:
    return "; ".join(note for note in notes if note) or None


def _file_with_curator(filing: Filing, submit: Callable[..., Any], note: str | None) -> Decision:
    """Contract §8 step 3."""
    try:
        result = submit(**filing.payload)
    except Exception as exc:
        logger.warning("disposition: submit_task raised for %r: %s", filing.title, exc)
        return Decision(filing, Step.CURATOR, note=_notes(note, f"submit_task raised {type(exc).__name__}: {exc}"))
    ticket = ticket_id_from_submit_result(result)
    if ticket is None:
        logger.warning("disposition: submit_task returned no ticket for %r: %r", filing.title, result)
        return Decision(filing, Step.CURATOR, note=_notes(note, f"submit_task returned no ticket: {result!r}"))
    return Decision(filing, Step.CURATOR, ticket=ticket, note=note)


def _decide(
    filing: Filing,
    found: _Found | None,
    *,
    store: task_store.TaskStore,
    submit: Callable[..., Any] | None,
) -> Decision | None:
    if found is not None:
        return Decision(filing, Step.KEY, found.standing, task_id=found.task_id, resolution=Resolution.EXISTING)
    match = _semantic_lookup(filing, store)
    if match.task_id is not None:
        return Decision(filing, Step.SEMANTIC, Standing.FILED, task_id=match.task_id, resolution=Resolution.EXISTING)
    if submit is None:
        return None
    return _file_with_curator(filing, submit, match.note)


def dispose(
    filings: Iterable[Filing],
    *,
    store: task_store.TaskStore,
    submit: Callable[..., Any] | None,
) -> Disposal:
    """Run contract §8 over *filings*. Every step-1 lookup runs before any
    filing, so a store that cannot answer one files nothing at all. With
    *submit* None (a dry run) the protocol stops before step 3."""
    filings = tuple(filings)
    try:
        found = [_key_lookup(filing, store) for filing in filings]
    except task_store.StoreUnreachable as exc:
        logger.warning("disposition: §8 step 1 could not run, so nothing is filed: %s", exc)
        return Disposal(decisions=(), unfiled=filings, unreachable=str(exc))
    decisions: list[Decision] = []
    unfiled: list[Filing] = []
    for filing, hit in zip(filings, found, strict=True):
        decision = _decide(filing, hit, store=store, submit=submit)
        if decision is None:
            unfiled.append(filing)
        else:
            decisions.append(decision)
    return Disposal(tuple(decisions), tuple(unfiled))



def _row_identity(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("ticket"), row.get("task_id"), row.get("step")


def with_back_links(cb: dict, links: Iterable[BackLink]) -> dict:
    """A new codebook with each link's C1 row on its entry; a row already
    there is not added twice."""
    result = copy.deepcopy(cb)
    entries = {entry.get("id"): entry for entry in result.get("entries") or []}
    for link in links:
        entry = entries.get(link.entry_id)
        if entry is None:
            raise ValueError(f"with_back_links: no codebook entry {link.entry_id!r} to link to")
        row = link.to_row()
        rows = entry.setdefault("filed_tickets", [])
        if all(_row_identity(existing) != _row_identity(row) for existing in rows):
            rows.append(row)
    codebook.assert_no_deletion(cb, result)
    return result


@dataclass(frozen=True)
class ResolvedTicket:
    entry_id: str
    ticket: str
    project_root: str
    outcome: task_store.TicketOutcome
    carried: bool


@dataclass(frozen=True)
class Resolutions:
    """One resolve_tickets pass. *pending* names the tickets still undecided,
    including any the budget left unsent; they resolve on a later run."""

    resolved: tuple[ResolvedTicket, ...] = ()
    pending: tuple[str, ...] = ()
    budget_exhausted: bool = False
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _OpenTicket:
    entry_id: str
    row: dict[str, Any]
    carried: bool


def _open_tickets(cb: dict, run_id: str) -> list[_OpenTicket]:
    """Every row of *cb* holding a ticket with no resolution, carried-over rows first."""
    open_tickets = [
        _OpenTicket(entry["id"], row, carried=row.get("run_id") != run_id)
        for entry in cb.get("entries") or []
        for row in entry.get("filed_tickets") or []
        if row.get("ticket") is not None and row.get("resolution") is None
    ]
    return sorted(open_tickets, key=lambda open_ticket: not open_ticket.carried)


def resolve_tickets(
    cb: dict,
    *,
    store: task_store.TaskStore,
    run_id: str,
    budget_secs: float = RESOLVE_BUDGET_SECS,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict, Resolutions]:
    """Resolve every unresolved ticket row in *cb* within *budget_secs* in
    total (D7), writing each decided outcome onto its row of a new codebook.
    It never writes task metadata (T4)."""
    result = copy.deepcopy(cb)
    deadline = clock() + budget_secs
    open_tickets = _open_tickets(result, run_id)
    resolved: list[ResolvedTicket] = []
    pending: list[str] = []
    errors: list[str] = []
    budget_exhausted = False
    for index, open_ticket in enumerate(open_tickets):
        remaining = deadline - clock()
        if remaining <= 0:
            budget_exhausted = True
            pending.extend(unsent.row["ticket"] for unsent in open_tickets[index:])
            break
        row = open_ticket.row
        try:
            outcome = store.resolve_ticket(
                row["project_root"], row["ticket"], timeout_secs=min(remaining, RESOLVE_CALL_CAP_SECS),
            )
        except task_store.StoreUnreachable as exc:
            errors.append(f"{row['ticket']}: {exc}")
            continue
        if outcome.undecided:
            pending.append(row["ticket"])
            continue
        row.update(resolution=_RESOLUTION_OF[outcome.status].value, task_id=outcome.task_id)
        resolved.append(ResolvedTicket(open_ticket.entry_id, row["ticket"], row["project_root"], outcome, open_ticket.carried))
    codebook.assert_no_deletion(cb, result)
    return result, Resolutions(tuple(resolved), tuple(pending), budget_exhausted, tuple(errors))
