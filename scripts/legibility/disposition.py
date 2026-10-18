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
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

# Bound as codebook.py binds it: no module-level assignment precedes the
# imports below, so they need no E402 suppression.
if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from legibility import codebook, task_store, verdict
from shared.toolcall_markup import escape_envelope_literals

logger = logging.getLogger("legibility.disposition")

CENSUS_SOURCE = "legibility_census"
SOURCE_METADATA = "source"
FINDING_KEY_METADATA = "x_finding_key"
ACCEPTANCE_REASON_METADATA = "x_acceptance_reason"
SUPERSEDES_TASK_METADATA = "x_supersedes_task"
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
    """What a task found for a finding says about it (contract §7, §8 step 1).
    ABANDONED is a task cancelled without an acceptance reason: no work and no
    acceptance, so the §8 key lookup treats it as absent."""

    FILED = "filed"
    ACCEPTED = "accepted"
    DONE = "done"
    ABANDONED = "abandoned"


class DispositionKind(enum.StrEnum):
    OPEN = "open"
    FILED = "filed"
    ACCEPTED = "accepted"
    REFUTED = "refuted"
    FIXED = "fixed"


_REFERENCE_OF = {
    DispositionKind.OPEN: (type(None), "no reference"),
    DispositionKind.FILED: (int, "a task id"),
    DispositionKind.ACCEPTED: (int, "a task id"),
    DispositionKind.REFUTED: (type(None), "no reference"),
    DispositionKind.FIXED: (str, "a commit sha"),
}


@dataclass(frozen=True)
class Disposition:
    """A finding's contract §7 disposition: ``open``, ``filed:<task id>``,
    ``accepted:<task id>``, ``refuted`` or ``fixed:<sha>``."""

    kind: DispositionKind
    ref: int | str | None = None

    def __post_init__(self) -> None:
        expected, described = _REFERENCE_OF[self.kind]
        if type(self.ref) is not expected:
            raise ValueError(f"a {self.kind.value} disposition takes {described}, got {self.ref!r}")

    def __str__(self) -> str:
        return self.kind.value if self.ref is None else f"{self.kind.value}:{self.ref}"


_OPEN = Disposition(DispositionKind.OPEN)


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


def files(found: verdict.Verdict) -> bool:
    """Contract §6 (D6): only a mechanical finding is filed; a structural one
    is recorded in the codebook and goes to deliberation."""
    return found.route is verdict.Route.MECHANICAL


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
    that files it and the codebook entry its back-link lands on.
    *verified_present*: this run's verifier found the finding present at
    as_of_sha, which is the re-verify contract §8 step 1c asks of a done task."""

    entry_id: str | None
    finding_key: str | None
    title: str
    statement: str
    payload: Mapping[str, Any]
    verified_present: bool = False

    @property
    def project_root(self) -> str:
        return self.payload["project_root"]

    @property
    def query(self) -> str:
        return "\n".join(dict.fromkeys(text for text in (self.title, self.statement) if text))

    @property
    def supersedes_task(self) -> int | None:
        return (self.payload.get("metadata") or {}).get(SUPERSEDES_TASK_METADATA)

    def superseding(self, task_id: int) -> Filing:
        """This filing superseding done task *task_id*: the one writer of
        x_supersedes_task and of the title that says so (contract §8 step 1c)."""
        suffix = escape_envelope_literals(f" (supersedes done task {task_id}: the finding is still present)")
        metadata = {**(self.payload.get("metadata") or {}), SUPERSEDES_TASK_METADATA: task_id}
        return replace(self, payload={**self.payload, "title": task_title(self.title) + suffix, "metadata": metadata})


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
    """What the dedup protocol decided for one Filing, and at which step.
    *awaiting_ticket* is an earlier run's ticket for the filing's entry that
    the curator still holds, which stopped this one before steps 2-3."""

    filing: Filing
    step: Step
    standing: Standing | None = None
    ticket: str | None = None
    task_id: int | None = None
    resolution: Resolution | None = None
    note: str | None = None
    awaiting_ticket: str | None = None

    @property
    def routed(self) -> bool:
        """Work exists for the finding: a task, or a ticket the curator holds."""
        return self.ticket is not None or self.task_id is not None

    def disposition(self) -> str:
        """The contract §7 disposition this decision gives the finding, with a
        note when it is open for a reason the decision knows."""
        if self.standing is Standing.DONE:
            return f"{_OPEN} (task {self.task_id} done; §8 step 1c re-verify not yet run)"
        if self.task_id is not None:
            kind = DispositionKind.ACCEPTED if self.standing is Standing.ACCEPTED else DispositionKind.FILED
            return str(Disposition(kind, self.task_id))
        if self.resolution is not None:
            return f"{_OPEN} (ticket {self.ticket} {self.resolution.value})"
        pending = self.ticket if self.ticket is not None else self.awaiting_ticket
        if pending is not None:
            return f"{_OPEN} (ticket {pending} pending)"
        return str(_OPEN)

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


_PRECEDENCE = (Standing.FILED, Standing.ACCEPTED, Standing.DONE, Standing.ABANDONED)


def _rank(standing: Standing, task_id: int) -> tuple[int, int]:
    """Which of several tasks found for one finding governs it: the first
    standing in _PRECEDENCE, the newest task first."""
    return _PRECEDENCE.index(standing), -task_id


def _standing(status: str, metadata: Mapping[str, Any]) -> Standing | None:
    """The one rule from a task's status, and a cancelled task's metadata, to
    its Standing; None for a status this module does not recognise."""
    if status in LIVE_STATUSES:
        return Standing.FILED
    if status == _DONE_STATUS:
        return Standing.DONE
    if status == _CANCELLED_STATUS:
        return Standing.ACCEPTED if metadata.get(ACCEPTANCE_REASON_METADATA) else Standing.ABANDONED
    return None


def _hit_standing(hit: task_store.TaskHit, *, store: task_store.TaskStore, project_root: str) -> Standing | None:
    metadata = store.get_task(project_root, hit.task_id).metadata if hit.status == _CANCELLED_STATUS else {}
    return _standing(hit.status, metadata)


def _key_lookup(filing: Filing, store: task_store.TaskStore) -> _Found | None:
    """Contract §8 step 1; an abandoned task counts as absent. Raises
    StoreUnreachable when it cannot be performed."""
    if filing.finding_key is None:
        return None
    hits = store.find_tasks_by_metadata(filing.project_root, key=FINDING_KEY_METADATA, value=filing.finding_key)
    found = [
        _Found(standing, hit.task_id)
        for hit in hits
        if (standing := _hit_standing(hit, store=store, project_root=filing.project_root))
        not in (None, Standing.ABANDONED)
    ]
    if not found:
        return None
    return min(found, key=lambda each: _rank(each.standing, each.task_id))


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


def _past_step_one(
    filing: Filing,
    *,
    store: task_store.TaskStore,
    submit: Callable[..., Any] | None,
    note: str | None = None,
) -> Decision | None:
    """Contract §8 steps 2 and 3."""
    match = _semantic_lookup(filing, store)
    if match.task_id is not None:
        return Decision(
            filing, Step.SEMANTIC, Standing.FILED, task_id=match.task_id, resolution=Resolution.EXISTING, note=note,
        )
    if submit is None:
        return None
    return _file_with_curator(filing, submit, _notes(note, match.note))


def _decide(
    filing: Filing,
    found: _Found | None,
    *,
    store: task_store.TaskStore,
    submit: Callable[..., Any] | None,
    awaiting: str | None,
) -> Decision | None:
    """Contract §8 for one filing after its step-1 lookup found *found*. A
    filing that would go on to steps 2-3 while its entry's earlier ticket
    *awaiting* is undecided stops there: that ticket may yet be its task."""
    onward, note = filing, None
    if found is not None:
        if found.standing is not Standing.DONE or not filing.verified_present:
            return Decision(filing, Step.KEY, found.standing, task_id=found.task_id, resolution=Resolution.EXISTING)
        onward = filing if filing.supersedes_task is not None else filing.superseding(found.task_id)
        note = f"§8 step 1c: done task {found.task_id} carries the key and the finding is still present"
    if awaiting is not None:
        held = f"§8 dedup: ticket {awaiting}, filed for this entry by an earlier run, still awaits the curator"
        return Decision(onward, Step.CURATOR, awaiting_ticket=awaiting, note=_notes(note, held))
    return _past_step_one(onward, store=store, submit=submit, note=note)


def _same_key_as(filing: Filing, earlier: Decision) -> Decision:
    """*filing* linked to the work *earlier*, a filing of this run carrying
    its key, was routed to."""
    return replace(
        earlier,
        filing=filing,
        step=Step.KEY,
        note=f"same finding key as {earlier.filing.title!r}, routed earlier this run",
    )


def dispose(
    filings: Iterable[Filing],
    *,
    store: task_store.TaskStore,
    submit: Callable[..., Any] | None,
    awaiting: Mapping[str, str] | None = None,
) -> Disposal:
    """Run contract §8 over *filings*. Every step-1 lookup runs before any
    filing, so a store that cannot answer one files nothing at all. A filing
    whose key an earlier filing of this run was already routed under is
    linked to that work, never filed again; nor is one whose entry is in
    *awaiting* (entry id -> the earlier run's ticket the curator has not
    decided, :attr:`Outcomes.awaiting`) once step 1 sends it onward. With
    *submit* None (a dry run) the protocol stops before step 3."""
    filings = tuple(filings)
    held_tickets = awaiting or {}
    try:
        found = [_key_lookup(filing, store) for filing in filings]
    except task_store.StoreUnreachable as exc:
        logger.warning("disposition: §8 step 1 could not run, so nothing is filed: %s", exc)
        return Disposal(decisions=(), unfiled=filings, unreachable=str(exc))
    decisions: list[Decision] = []
    unfiled: list[Filing] = []
    routed_by_key: dict[str, Decision] = {}
    for filing, hit in zip(filings, found, strict=True):
        earlier = None if filing.finding_key is None else routed_by_key.get(filing.finding_key)
        held_by = None if filing.entry_id is None else held_tickets.get(filing.entry_id)
        decision = (
            _decide(filing, hit, store=store, submit=submit, awaiting=held_by)
            if earlier is None
            else _same_key_as(filing, earlier)
        )
        if decision is None:
            unfiled.append(filing)
            continue
        decisions.append(decision)
        if decision.routed and filing.finding_key is not None:
            routed_by_key.setdefault(filing.finding_key, decision)
    return Disposal(tuple(decisions), tuple(unfiled))


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, Any]:
    return row.get("ticket"), row.get("task_id")


def with_back_links(cb: dict, links: Iterable[BackLink]) -> dict:
    """A new codebook with each link's C1 row on its entry; a row naming a
    ticket and task already there is not added twice, whichever step
    decided it."""
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


def _awaits_curator(row: Mapping[str, Any]) -> bool:
    """The one rule for a C1 row whose ticket the curator has not decided."""
    return row.get("ticket") is not None and row.get("resolution") is None


def _open_tickets(cb: dict, run_id: str) -> list[_OpenTicket]:
    """Every row of *cb* awaiting the curator, carried-over rows first."""
    open_tickets = [
        _OpenTicket(entry["id"], row, carried=row.get("run_id") != run_id)
        for entry in cb.get("entries") or []
        for row in entry.get("filed_tickets") or []
        if _awaits_curator(row)
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
    A ticket several rows hold is asked about once. It never writes task
    metadata (T4)."""
    result = copy.deepcopy(cb)
    deadline = clock() + budget_secs
    outcomes: dict[tuple[str, str], task_store.TicketOutcome] = {}
    resolved: list[ResolvedTicket] = []
    pending: list[str] = []
    errors: list[str] = []
    budget_exhausted = False
    for open_ticket in _open_tickets(result, run_id):
        row = open_ticket.row
        asked = (row["project_root"], row["ticket"])
        if asked not in outcomes:
            remaining = deadline - clock()
            if remaining <= 0:
                budget_exhausted = True
                pending.append(row["ticket"])
                continue
            try:
                outcomes[asked] = store.resolve_ticket(
                    row["project_root"], row["ticket"], timeout_secs=min(remaining, RESOLVE_CALL_CAP_SECS),
                )
            except task_store.StoreUnreachable as exc:
                errors.append(f"{row['ticket']}: {exc}")
                continue
        outcome = outcomes[asked]
        if outcome.undecided:
            pending.append(row["ticket"])
            continue
        row.update(resolution=_RESOLUTION_OF[outcome.status].value, task_id=outcome.task_id)
        resolved.append(ResolvedTicket(open_ticket.entry_id, row["ticket"], row["project_root"], outcome, open_ticket.carried))
    codebook.assert_no_deletion(cb, result)
    return result, Resolutions(tuple(resolved), tuple(dict.fromkeys(pending)), budget_exhausted, tuple(errors))


@dataclass(frozen=True)
class Backfill:
    """The historical census tasks one backfill linked, as (task id, entry id),
    and those whose title matched no entry. *done_in_run* names the earlier
    run whose backfill already did this, when this one looked nothing up."""

    linked: tuple[tuple[int, str], ...] = ()
    unmatched: tuple[task_store.TaskHit, ...] = ()
    error: str | None = None
    done_in_run: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "linked": [{"task_id": task_id, "entry_id": entry_id} for task_id, entry_id in self.linked],
            "unmatched": [{"task_id": hit.task_id, "title": hit.title} for hit in self.unmatched],
            "error": self.error,
            "done_in_run": self.done_in_run,
        }


def _entry_ids_by_filed_title(cb: Mapping[str, Any]) -> dict[str, str]:
    """Each entry under the task title its own title files as, and under that
    of any candidate promoted to it, which a later correction may have
    retitled away from. Where titles collide the last entry wins, as in
    census._find_adjudicated_candidate."""
    promoted_titles: dict[str, list[str]] = {}
    for candidate in cb.get("candidates") or []:
        if isinstance(candidate.get("promoted_to"), str) and isinstance(candidate.get("title"), str):
            promoted_titles.setdefault(candidate["promoted_to"], []).append(candidate["title"])
    by_title: dict[str, str] = {}
    for entry in cb.get("entries") or []:
        for title in (*promoted_titles.get(entry["id"], ()), entry.get("title")):
            if isinstance(title, str):
                by_title[task_title(title)] = entry["id"]
    return by_title


def _rows_of(cb: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for entry in cb.get("entries") or []:
        for row in entry.get("filed_tickets") or []:
            yield entry["id"], row


def backfill(
    cb: dict, *, store: task_store.TaskStore, project_root: str, run_id: str,
) -> tuple[dict, Backfill]:
    """Link every ``source=legibility_census`` task to the entry its title was
    filed under (plans/census-incremental-prd.md L4), once: the codebook's
    ``codebook.HISTORICAL_BACKFILL`` marker records the run that did, and any
    later run looks nothing up. A task linked anywhere is skipped, and so is
    an entry whose open ticket row the ticket pass owns. An unreachable store
    links nothing, sets no marker and never raises."""
    done = cb.get(codebook.HISTORICAL_BACKFILL)
    if done is not None:
        return cb, Backfill(done_in_run=done["run_id"])
    try:
        hits = store.find_tasks_by_metadata(project_root, key=SOURCE_METADATA, value=CENSUS_SOURCE)
    except task_store.StoreUnreachable as exc:
        logger.warning("disposition: backfill skipped: %s", exc)
        return cb, Backfill(error=str(exc))
    by_title = _entry_ids_by_filed_title(cb)
    linked_task_ids = {row.get("task_id") for _, row in _rows_of(cb)}
    ticketed = {entry_id for entry_id, row in _rows_of(cb) if _awaits_curator(row)}
    keys = {entry["id"]: entry.get("finding_key") for entry in cb.get("entries") or []}
    links: list[BackLink] = []
    unmatched: list[task_store.TaskHit] = []
    for hit in hits:
        entry_id = by_title.get(hit.title)
        if entry_id is None:
            unmatched.append(hit)
        elif hit.task_id not in linked_task_ids and entry_id not in ticketed:
            linked_task_ids.add(hit.task_id)
            links.append(BackLink(
                entry_id=entry_id, ticket=None, project_root=project_root, run_id=run_id,
                finding_key=keys[entry_id], task_id=hit.task_id,
                resolution=Resolution.CREATED, step=Step.CURATOR,
            ))
    linked = tuple((link.task_id, link.entry_id) for link in links if link.task_id is not None)
    result = with_back_links(cb, links)
    result[codebook.HISTORICAL_BACKFILL] = {
        "run_id": run_id, "unmatched_task_ids": [hit.task_id for hit in unmatched],
    }
    return result, Backfill(linked, tuple(unmatched))


@dataclass(frozen=True)
class TaskRef:
    """A task, named in the project whose ids it is numbered in."""

    project_root: str
    task_id: int


@dataclass(frozen=True)
class TaskOutcome:
    """One back-linked task as this run read it. *ref* None is a ticket that
    names no task; a *ref* with *standing* None was not observed, and *note*
    says why."""

    ref: TaskRef | None
    ticket: str | None
    status: str | None
    standing: Standing | None
    note: str | None = None


@dataclass(frozen=True)
class EntryOutcome:
    """What one codebook entry's back-linked tasks say, read this run.
    *awaiting* holds the tickets of its rows the curator has not decided, in
    row order."""

    entry_id: str
    finding_key: str | None
    entry_status: str
    tasks: tuple[TaskOutcome, ...]
    awaiting: tuple[str, ...] = ()

    @property
    def hold(self) -> str | None:
        """Why the entry is not acted on this run, or None. Corroborate
        before acting (INV-3): a ticket awaiting the curator may still become
        the live task, and so may a linked task this run did not observe."""
        if self.awaiting:
            return f"ticket {self.awaiting[-1]} awaits the curator and may become the live task"
        if any(task.ref is not None and task.standing is None for task in self.tasks):
            return "another linked task was not observed"
        return None

    @property
    def governing(self) -> TaskOutcome | None:
        """The task whose standing decides the entry, by the §8 step 1
        precedence; None when there is no task, or while the entry is on
        :attr:`hold`."""
        if self.hold is not None:
            return None
        ranked = [
            (_rank(task.standing, task.ref.task_id), task)
            for task in self.tasks
            if task.standing is not None and task.ref is not None
        ]
        return min(ranked, key=lambda each: each[0])[1] if ranked else None


@dataclass(frozen=True)
class Reverify:
    """An entry whose governing task is done, owed one re-verify (D8)."""

    entry_id: str
    task: TaskRef


_ACCEPTED_STATUS = "accepted"
_FIXED_STATUS = "fixed"
_RETIRED_STATUS = "retired"
_OPEN_STATUS = "open"
_SETTLED_STATUSES = frozenset({_FIXED_STATUS, _RETIRED_STATUS})


@dataclass(frozen=True)
class Outcomes:
    """Every back-linked entry's outcome, from one run's read; *statuses_error*
    says why the get_statuses snapshot was unavailable, when it was."""

    entries: tuple[EntryOutcome, ...] = ()
    statuses_error: str | None = None

    @property
    def any_back_links(self) -> bool:
        return bool(self.entries)

    @property
    def awaiting(self) -> dict[str, str]:
        """Each entry's newest ticket awaiting the curator, by entry id: the
        work an earlier run filed for it that the curator still holds."""
        return {entry.entry_id: entry.awaiting[-1] for entry in self.entries if entry.awaiting}

    @property
    def due(self) -> tuple[Reverify, ...]:
        """The re-verify queue, in codebook order: each entry governed by a
        done task that is not already fixed or retired."""
        due: list[Reverify] = []
        for entry in self.entries:
            governing = entry.governing
            if governing is None or governing.ref is None or governing.standing is not Standing.DONE:
                continue
            if entry.entry_status not in _SETTLED_STATUSES:
                due.append(Reverify(entry.entry_id, governing.ref))
        return tuple(due)


def _read_outcome(
    ref: TaskRef,
    ticket: str | None,
    *,
    statuses: Mapping[str, Any] | None,
    statuses_error: str | None,
    observed_root: str,
    store: task_store.TaskStore,
) -> TaskOutcome:
    """*ref*'s standing: from the snapshot for the observed project, with a
    cancelled task read back for its acceptance reason, and from get_task
    alone for another project's task, whose id the snapshot does not number."""
    status = None
    if ref.project_root == observed_root:
        if statuses is None:
            return TaskOutcome(ref, ticket, None, None, f"get_statuses could not be read: {statuses_error}")
        status = statuses.get(str(ref.task_id))
        if status is None:
            return TaskOutcome(ref, ticket, None, None, f"task {ref.task_id} is not in the get_statuses snapshot")
        if status != _CANCELLED_STATUS:
            return _outcome_of(ref, ticket, status, {})
    try:
        record = store.get_task(ref.project_root, ref.task_id)
    except task_store.StoreUnreachable as exc:
        return TaskOutcome(ref, ticket, status, None, f"could not be read back: {exc}")
    return _outcome_of(ref, ticket, record.status, record.metadata)


def _outcome_of(ref: TaskRef, ticket: str | None, status: str, metadata: Mapping[str, Any]) -> TaskOutcome:
    standing = _standing(status, metadata)
    return TaskOutcome(ref, ticket, status, standing, None if standing else f"unrecognised task status {status!r}")


def _ticket_only(row: Mapping[str, Any]) -> TaskOutcome:
    note = "awaiting the curator" if _awaits_curator(row) else f"{row.get('resolution')}; it names no task"
    return TaskOutcome(None, row.get("ticket"), None, None, note)


def _entry_outcome(entry: Mapping[str, Any], read: Callable[[TaskRef, str | None], TaskOutcome]) -> EntryOutcome:
    rows = entry.get("filed_tickets") or []
    tasks: list[TaskOutcome] = []
    seen: set[TaskRef] = set()
    for row in rows:
        if row.get("task_id") is None:
            tasks.append(_ticket_only(row))
            continue
        ref = TaskRef(row["project_root"], row["task_id"])
        if ref not in seen:
            seen.add(ref)
            tasks.append(read(ref, row.get("ticket")))
    awaiting = tuple(row["ticket"] for row in rows if _awaits_curator(row))
    return EntryOutcome(entry["id"], entry.get("finding_key"), entry["status"], tuple(tasks), awaiting)


def read_outcomes(
    cb: Mapping[str, Any],
    *,
    statuses: Mapping[str, Any] | None,
    statuses_error: str | None,
    observed_root: str,
    store: task_store.TaskStore,
) -> Outcomes:
    """Each back-linked entry's tasks, read from one get_statuses snapshot
    (*statuses*, None when it could not be read, *statuses_error* saying why)
    and get_task where the snapshot cannot answer. Never raises on a store
    failure: an unread task is unobserved, and its note says why."""

    def read(ref: TaskRef, ticket: str | None) -> TaskOutcome:
        return _read_outcome(
            ref, ticket, statuses=statuses, statuses_error=statuses_error, observed_root=observed_root, store=store,
        )

    entries = tuple(_entry_outcome(entry, read) for entry in cb.get("entries") or [] if entry.get("filed_tickets"))
    unobserved = [
        f"task {task.ref.task_id} in {task.ref.project_root}: {task.note}"
        for entry in entries
        for task in entry.tasks
        if task.ref is not None and task.standing is None
    ]
    if unobserved:
        logger.warning(
            "disposition: %d back-linked task(s) were not observed, so their entries keep their status: %s",
            len(unobserved),
            "; ".join(unobserved),
        )
    return Outcomes(entries, statuses_error)


REVERIFY_ENTRY = "x_reverify_entry"
_CAP_DEFERRAL = "deferred by the operator verify cap (max_verify_clusters); re-queued next run"


class ReverifyResult(enum.StrEnum):
    GONE = "gone"
    PRESENT = "present"
    INCONCLUSIVE = "inconclusive"
    DEFERRED = "deferred"


_VERIFIED_BY_RESULT = {ReverifyResult.GONE: False, ReverifyResult.PRESENT: True}


@dataclass(frozen=True)
class Reverification:
    """One due entry's re-verify at as_of_sha (D8). INV-3: GONE carries a
    refuting Verdict and PRESENT a verified one, so a failed verify call can
    never read as a landed fix."""

    entry_id: str
    task: TaskRef
    result: ReverifyResult
    found: verdict.Verdict | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        required = _VERIFIED_BY_RESULT.get(self.result)
        if required is not None and (self.found is None or self.found.verified is not required):
            carried = "no Verdict" if self.found is None else f"a Verdict with verified={self.found.verified}"
            raise ValueError(
                f"reverification of entry {self.entry_id!r}: INV-3 requires {self.result.value} to carry "
                f"a Verdict with verified={required}, got {carried}"
            )


def _clusters_by_entry(clusters: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(clusters, list | tuple):
        return {}
    return {
        cluster[REVERIFY_ENTRY]: cluster
        for cluster in clusters
        if isinstance(cluster, Mapping) and isinstance(cluster.get(REVERIFY_ENTRY), str)
    }


def _reverification(
    due: Reverify, lists: Sequence[tuple[str, Mapping[str, Mapping[str, Any]]]], error: str | None,
) -> Reverification:
    if error is not None:
        return Reverification(
            due.entry_id, due.task, ReverifyResult.INCONCLUSIVE, reason=f"the re-verify call did not complete: {error}",
        )
    for name, clusters in lists:
        cluster = clusters.get(due.entry_id)
        if cluster is None:
            continue
        found = cluster.get("verdict")
        result = ReverifyResult.PRESENT if name == "verified" else ReverifyResult.GONE
        if isinstance(found, verdict.Verdict) and found.verified is _VERIFIED_BY_RESULT[result]:
            return Reverification(due.entry_id, due.task, result, found)
        reason = (
            f"{name} without a verdict: the verify call failed"
            if not isinstance(found, verdict.Verdict)
            else f"{name}, but its verdict says verified={found.verified}"
        )
        return Reverification(due.entry_id, due.task, ReverifyResult.INCONCLUSIVE, reason=reason)
    return Reverification(
        due.entry_id, due.task, ReverifyResult.INCONCLUSIVE, reason="the verifier returned no result for it",
    )


def reverifications(
    result: object,
    *,
    offered: Sequence[Reverify],
    deferred: Sequence[Reverify],
    error: str | None = None,
) -> dict[str, Reverification]:
    """Each re-verify, by entry id, typed from one verify_fn *result* whose
    clusters carry REVERIFY_ENTRY: *offered* ones by what the verifier
    answered (*error* set when the call did not complete), *deferred* ones as
    the verify cap left them. Never raises on a malformed or partial result."""
    answered: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
    lists = (
        ("verified", _clusters_by_entry(answered.get("verified"))),
        ("rejected", _clusters_by_entry(answered.get("rejected"))),
    )
    typed = {due.entry_id: _reverification(due, lists, error) for due in offered}
    for due in deferred:
        typed[due.entry_id] = Reverification(due.entry_id, due.task, ReverifyResult.DEFERRED, reason=_CAP_DEFERRAL)
    return typed


@dataclass(frozen=True)
class StatusChange:
    """One entry the projection changed, with the accepting task or fix sha it set."""

    entry_id: str
    from_status: str
    to_status: str
    accepted_by: int | None = None
    fixed_at_sha: str | None = None


@dataclass(frozen=True)
class _Projected:
    status: str
    accepted_by: int | None = None
    fixed_at_sha: str | None = None


def _projection(
    entry: Mapping[str, Any], outcome: EntryOutcome, again: Reverification | None, *, as_of_sha: str,
) -> _Projected | None:
    """The one statement of how a task's outcome sets its entry's status
    (INV-9); None leaves the entry as it is. Only a re-verify that found the
    finding gone at *as_of_sha* fixes it (INV-3)."""
    governing = outcome.governing
    status = entry["status"]
    if status == _RETIRED_STATUS or governing is None or governing.ref is None:
        return None
    if governing.standing is Standing.ACCEPTED:
        return _Projected(_ACCEPTED_STATUS, accepted_by=governing.ref.task_id)
    if (
        governing.standing is Standing.DONE
        and again is not None
        and again.result is ReverifyResult.GONE
        and again.task == governing.ref
    ):
        return _Projected(_FIXED_STATUS, fixed_at_sha=as_of_sha)
    if status == _FIXED_STATUS and governing.standing is not Standing.FILED:
        return None
    return _Projected(_OPEN_STATUS if status in (_ACCEPTED_STATUS, _FIXED_STATUS) else status)


def _projected_state(entry: Mapping[str, Any]) -> _Projected:
    return _Projected(entry["status"], entry.get("accepted_by"), entry.get("fixed_at_sha"))


def _apply(entry: dict[str, Any], projected: _Projected) -> None:
    entry["status"] = projected.status
    for field, value in (("accepted_by", projected.accepted_by), ("fixed_at_sha", projected.fixed_at_sha)):
        if value is None:
            entry.pop(field, None)
        else:
            entry[field] = value


def project(
    cb: dict, outcomes: Outcomes, reverified: Mapping[str, Reverification], *, as_of_sha: str,
) -> tuple[dict, tuple[StatusChange, ...]]:
    """A new codebook whose back-linked entries carry the status their tasks'
    outcomes project, re-derived from the task store on every run, and the
    entries that changed."""
    result = copy.deepcopy(cb)
    by_entry = {outcome.entry_id: outcome for outcome in outcomes.entries}
    changes: list[StatusChange] = []
    for entry in result.get("entries") or []:
        outcome = by_entry.get(entry["id"])
        projected = (
            None if outcome is None
            else _projection(entry, outcome, reverified.get(entry["id"]), as_of_sha=as_of_sha)
        )
        if projected is None or projected == _projected_state(entry):
            continue
        changes.append(StatusChange(
            entry["id"], entry["status"], projected.status, projected.accepted_by, projected.fixed_at_sha,
        ))
        _apply(entry, projected)
    codebook.assert_no_deletion(cb, result)
    return result, tuple(changes)


@dataclass(frozen=True)
class Refiling:
    """An entry to file again this run: still present after its done task
    (superseding it, contract §8 step 1c), or abandoned and sighted again."""

    entry_id: str
    supersedes_task: int | None
    verified_present: bool
    found: verdict.Verdict | None = None


def refilings(
    outcomes: Outcomes, reverified: Mapping[str, Reverification], *, sighted: Collection[str],
) -> tuple[Refiling, ...]:
    """The entries to file again, in codebook order. A present finding is
    re-filed only when its verdict routes mechanical (contract §6); an
    abandoned one only when it recurred this run (in *sighted*)."""
    refiled: list[Refiling] = []
    for outcome in outcomes.entries:
        again = reverified.get(outcome.entry_id)
        governing = outcome.governing
        if again is not None and again.result is ReverifyResult.PRESENT and again.found is not None:
            if files(again.found):
                refiled.append(Refiling(outcome.entry_id, again.task.task_id, True, again.found))
        elif governing is not None and governing.standing is Standing.ABANDONED and outcome.entry_id in sighted:
            refiled.append(Refiling(outcome.entry_id, None, False))
    return tuple(refiled)


def disposition_of(entry: Mapping[str, Any], standings: Mapping[TaskRef, Standing]) -> Disposition:
    """*entry*'s contract §7 disposition, from its (projected) status and its
    ``filed_tickets`` read against *standings*: fixed at its sha, accepted by
    its task, filed under its newest live task, else open."""
    status = entry.get("status")
    if status == _FIXED_STATUS and entry.get("fixed_at_sha"):
        return Disposition(DispositionKind.FIXED, entry["fixed_at_sha"])
    if status == _ACCEPTED_STATUS and entry.get("accepted_by") is not None:
        return Disposition(DispositionKind.ACCEPTED, entry["accepted_by"])
    live = [
        row["task_id"]
        for row in entry.get("filed_tickets") or []
        if row.get("task_id") is not None
        and standings.get(TaskRef(row["project_root"], row["task_id"])) is Standing.FILED
    ]
    return Disposition(DispositionKind.FILED, max(live)) if live else _OPEN


def entry_dispositions(cb: Mapping[str, Any], standings: Mapping[TaskRef, Standing]) -> dict[str, Disposition]:
    """Every entry's contract §7 disposition, by :func:`disposition_of`."""
    return {entry["id"]: disposition_of(entry, standings) for entry in cb.get("entries") or []}


def task_standings(outcomes: Outcomes, decisions: Iterable[Decision]) -> dict[TaskRef, Standing]:
    """Every task standing this run knows: the outcome read's, then each
    routed decision's, a curator ticket's created or combined task among them."""
    standings = {
        task.ref: task.standing
        for entry in outcomes.entries
        for task in entry.tasks
        if task.ref is not None and task.standing is not None
    }
    for decision in decisions:
        if decision.task_id is not None and decision.standing is not None:
            standings[TaskRef(decision.filing.project_root, decision.task_id)] = decision.standing
    return standings


@dataclass(frozen=True)
class StructuralFinding:
    entry_id: str
    title: str
    finding_key: str | None
    anchor: str | None


def structural_findings(cb: Mapping[str, Any], entry_ids: Iterable[str]) -> tuple[StructuralFinding, ...]:
    """The structural findings promoted to *entry_ids*, read from their entries."""
    entries = {entry.get("id"): entry for entry in cb.get("entries") or []}
    found: list[StructuralFinding] = []
    for entry_id in entry_ids:
        entry = entries.get(entry_id)
        if entry is None:
            raise ValueError(f"structural_findings: no codebook entry {entry_id!r}")
        found.append(StructuralFinding(entry_id, entry["title"], entry.get("finding_key"), entry.get("finding_anchor")))
    return tuple(found)


def _filing_record(filing: Filing) -> dict[str, Any]:
    return {
        "entry_id": filing.entry_id,
        "finding_key": filing.finding_key,
        "title": filing.title,
        "project_root": filing.project_root,
    }


def _resolved_record(resolved: ResolvedTicket) -> dict[str, Any]:
    return {
        "entry_id": resolved.entry_id,
        "ticket": resolved.ticket,
        "project_root": resolved.project_root,
        "status": resolved.outcome.status.value,
        "task_id": resolved.outcome.task_id,
        "reason": resolved.outcome.reason,
    }


@dataclass(frozen=True)
class FilingReport:
    """Everything one run's filing produced, for the run record: the
    *disposal* as dispose returned it, the ticket pass's *resolutions*, the
    *backfill*, and the *structural* findings recorded but not filed."""

    disposal: Disposal
    resolutions: Resolutions
    backfill: Backfill
    structural: tuple[StructuralFinding, ...] = ()

    def to_record(self) -> dict[str, Any]:
        resolved = self.disposal.resolved(self.resolutions)
        return {
            "decisions": [decision.to_record() for decision in resolved.decisions],
            "unfiled": [_filing_record(filing) for filing in resolved.unfiled],
            "unreachable": resolved.unreachable,
            "carried": [_resolved_record(each) for each in self.resolutions.resolved if each.carried],
            "pending_tickets": list(self.resolutions.pending),
            "budget_exhausted": self.resolutions.budget_exhausted,
            "errors": list(self.resolutions.errors),
            "backfill": self.backfill.to_record(),
            "structural": [asdict(finding) for finding in self.structural],
        }


FILED_TASK_COLUMNS = (
    "finding_key", "title", "step", "resolution", "task_id", "ticket", "disposition", "project_root", "note",
)
STRUCTURAL_COLUMNS = ("finding_key", "entry_id", "title", "anchor")
_PENDING_NOTE = "pending with the curator; resolves on the next run"


def _cell(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split()).replace("|", "\\|")


def _table(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(_cell(row[column]) for column in columns) + " |" for row in rows),
    ]


def _finding_label(item: Mapping[str, Any]) -> str:
    return f"{item['title']} (`{item['finding_key'] or 'unkeyed'}`)"


def _decision_row(decision: Mapping[str, Any], observed_root: str) -> dict[str, Any]:
    pending = decision["ticket"] is not None and decision["resolution"] is None
    return {
        **decision,
        "project_root": None if decision["project_root"] == observed_root else decision["project_root"],
        "note": _notes(decision["note"], _PENDING_NOTE if pending else None),
    }


def _decision_lines(record: Mapping[str, Any], observed_root: str) -> list[str]:
    decisions = record["decisions"]
    if not decisions:
        return []
    return _table(FILED_TASK_COLUMNS, (_decision_row(decision, observed_root) for decision in decisions))


def _unfiled_lines(record: Mapping[str, Any]) -> list[str]:
    unfiled = record["unfiled"]
    if record["unreachable"] is not None:
        return [
            "- Contract §8 step 1 (key lookup) could not run, so NOTHING was filed: "
            f"{record['unreachable']}",
            *(f"  - {_finding_label(item)}" for item in unfiled),
        ]
    if not unfiled:
        return []
    return [
        f"- {len(unfiled)} finding(s) stopped before the curator (§8 step 3) and were not filed:",
        *(f"  - {_finding_label(item)}" for item in unfiled),
    ]


def _resolution_lines(record: Mapping[str, Any]) -> list[str]:
    lines = []
    if record["carried"]:
        lines.append("- Tickets from earlier runs resolved this run:")
        lines.extend(
            f"  - {item['ticket']} (entry {item['entry_id']}): {item['status']}"
            + (f" task {item['task_id']}" if item["task_id"] is not None else "")
            + (f" ({item['reason']})" if item["reason"] else "")
            for item in record["carried"]
        )
    if record["budget_exhausted"]:
        lines.append(
            f"- The {RESOLVE_BUDGET_SECS:g} s resolve budget ran out; these tickets resolve on the next run: "
            + ", ".join(record["pending_tickets"])
        )
    if record["errors"]:
        lines.append("- resolve_ticket could not answer for: " + "; ".join(record["errors"]))
    return lines


def _backfill_lines(record: Mapping[str, Any]) -> list[str]:
    backfill = record["backfill"]
    lines = []
    if backfill["linked"]:
        lines.append(
            f"- Backfill linked {len(backfill['linked'])} historical census task(s): "
            + ", ".join(f"{item['task_id']} -> {item['entry_id']}" for item in backfill["linked"])
        )
    if backfill["unmatched"]:
        lines.append(
            f"- Backfill matched no entry for {len(backfill['unmatched'])} census task(s): "
            + "; ".join(f"{item['task_id']} {item['title']}" for item in backfill["unmatched"])
        )
    if backfill["error"]:
        lines.append(f"- Backfill could not run: {backfill['error']}")
    return lines


def filed_tasks_lines(record: Mapping[str, Any], *, observed_root: str) -> list[str]:
    """The ``## Filed Tasks`` body, without its heading, rendered from
    :meth:`FilingReport.to_record`: each finding's deciding step (contract
    §8), its task, ticket and contract §7 disposition."""
    blocks = [
        block
        for block in (
            _decision_lines(record, observed_root),
            _unfiled_lines(record),
            _resolution_lines(record),
            _backfill_lines(record),
        )
        if block
    ]
    if not blocks:
        return ["_none filed._"]
    lines = list(blocks[0])
    for block in blocks[1:]:
        lines.extend(["", *block])
    return lines


def structural_lines(record: Mapping[str, Any]) -> list[str]:
    """The report's ``## Structural`` section, always present (contract §6, D6)."""
    heading = ["", "## Structural", ""]
    structural = record["structural"]
    if not structural:
        return [*heading, "_no structural findings this run._"]
    return [
        *heading,
        "These verified findings are structural (contract §6, D6): recorded in the codebook and NOT filed; "
        "each goes to deliberation with the human, then a program doc and /prd.",
        "",
        *_table(STRUCTURAL_COLUMNS, structural),
    ]


class RowOutcome(enum.StrEnum):
    """What one back-linked ticket or task did to its entry this run."""

    LIVE = "live"
    ACCEPTED = "accepted"
    ABANDONED = "abandoned"
    GONE = "gone"
    PRESENT = "present"
    PRESENT_STRUCTURAL = "present-structural"
    INCONCLUSIVE = "inconclusive"
    DEFERRED = "deferred"
    SETTLED = "settled"
    OUTRANKED = "outranked"
    HELD = "held"
    UNOBSERVED = "unobserved"
    TICKET = "ticket"


_STANDING_OUTCOME = {Standing.FILED: RowOutcome.LIVE, Standing.ACCEPTED: RowOutcome.ACCEPTED}
_REVERIFY_OUTCOME = {
    ReverifyResult.GONE: RowOutcome.GONE,
    ReverifyResult.INCONCLUSIVE: RowOutcome.INCONCLUSIVE,
    ReverifyResult.DEFERRED: RowOutcome.DEFERRED,
}


@dataclass(frozen=True)
class OutcomeReport:
    """One run's outcome feedback, for the run record: the *outcomes* read,
    the re-verifies by entry, the projection's *changes*, the entries
    *refiled*, and every entry's contract §7 disposition after the run."""

    outcomes: Outcomes
    reverified: Mapping[str, Reverification]
    changes: tuple[StatusChange, ...]
    refiled: tuple[Refiling, ...]
    dispositions: Mapping[str, Disposition]

    def _done_outcome(self, entry: EntryOutcome) -> tuple[RowOutcome, str | int | None]:
        if entry.entry_status in _SETTLED_STATUSES:
            return RowOutcome.SETTLED, entry.entry_status
        again = self.reverified.get(entry.entry_id)
        if again is None:
            return RowOutcome.INCONCLUSIVE, "no re-verify ran"
        if again.result is ReverifyResult.PRESENT:
            mechanical = again.found is not None and files(again.found)
            return (RowOutcome.PRESENT, again.task.task_id) if mechanical else (RowOutcome.PRESENT_STRUCTURAL, None)
        if again.result is ReverifyResult.GONE:
            fixed_at = {change.entry_id: change.fixed_at_sha for change in self.changes}
            return RowOutcome.GONE, fixed_at.get(entry.entry_id)
        return _REVERIFY_OUTCOME[again.result], again.reason

    def _outcome(self, entry: EntryOutcome, task: TaskOutcome) -> tuple[RowOutcome, str | int | None]:
        if task.ref is None:
            return RowOutcome.TICKET, task.note
        if task.standing is None:
            return RowOutcome.UNOBSERVED, task.note
        governing = entry.governing
        if governing is None or governing.ref is None:
            return RowOutcome.HELD, entry.hold
        if governing.ref != task.ref:
            return RowOutcome.OUTRANKED, governing.ref.task_id
        if task.standing is Standing.ABANDONED:
            refiled = any(refiling.entry_id == entry.entry_id for refiling in self.refiled)
            return RowOutcome.ABANDONED, "re-filed this run" if refiled else "not sighted this run"
        if task.standing is Standing.DONE:
            return self._done_outcome(entry)
        return _STANDING_OUTCOME[task.standing], None

    def _row(self, entry: EntryOutcome, task: TaskOutcome) -> dict[str, Any]:
        outcome, detail = self._outcome(entry, task)
        return {
            "entry_id": entry.entry_id,
            "finding_key": entry.finding_key,
            "project_root": None if task.ref is None else task.ref.project_root,
            "ticket": task.ticket,
            "task_id": None if task.ref is None else task.ref.task_id,
            "task_status": task.status,
            "outcome": outcome.value,
            "detail": detail,
            "disposition": str(self.dispositions[entry.entry_id]),
        }

    def _reverify_counts(self) -> dict[str, int]:
        results = [again.result for again in self.reverified.values()]
        return {
            "due": len(self.outcomes.due),
            "offered": sum(1 for result in results if result is not ReverifyResult.DEFERRED),
            **{result.value: results.count(result) for result in ReverifyResult},
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "any_back_links": self.outcomes.any_back_links,
            "statuses_error": self.outcomes.statuses_error,
            "rows": [self._row(entry, task) for entry in self.outcomes.entries for task in entry.tasks],
            "reverify": self._reverify_counts(),
            "changes": [asdict(change) for change in self.changes],
            "refiled": [
                {"entry_id": refiling.entry_id, "supersedes_task": refiling.supersedes_task,
                 "verified_present": refiling.verified_present}
                for refiling in self.refiled
            ],
        }


DISPOSITION_COLUMNS = (
    "entry_id", "finding_key", "ticket", "task_id", "task_status", "outcome", "disposition", "project_root",
)
_NOT_YET_LINKED = "_no back-linked tickets yet._"
_NOTHING_CHANGED = "_no outcome changed this run._"
_OUTCOME_CELLS = {
    RowOutcome.LIVE: "live",
    RowOutcome.ACCEPTED: "accepted",
    RowOutcome.ABANDONED: "abandoned (eligible to re-file; {detail})",
    RowOutcome.GONE: "done -> re-verified gone (fixed at {detail})",
    RowOutcome.PRESENT: "done -> re-verified present (re-filed, supersedes {detail})",
    RowOutcome.PRESENT_STRUCTURAL: "done -> re-verified present but structural (contract §6: not re-filed)",
    RowOutcome.INCONCLUSIVE: "done -> re-verify inconclusive ({detail})",
    RowOutcome.DEFERRED: "done -> re-verify deferred by the verify cap (re-queued next run)",
    RowOutcome.SETTLED: "done -> entry already {detail}",
    RowOutcome.OUTRANKED: "{task_status} (task {detail} decides the entry)",
    RowOutcome.HELD: "{task_status} (not acted on: {detail})",
    RowOutcome.UNOBSERVED: "unobserved ({detail})",
    RowOutcome.TICKET: "ticket {detail}",
}


def _disposition_row(row: Mapping[str, Any], observed_root: str) -> dict[str, Any]:
    template = _OUTCOME_CELLS[RowOutcome(row["outcome"])]
    return {
        **row,
        "outcome": template.format(detail=row["detail"], task_status=row["task_status"]),
        "project_root": None if row["project_root"] in (None, observed_root) else row["project_root"],
    }


def _change_line(change: Mapping[str, Any]) -> str:
    line = f"- {change['entry_id']}: {change['from_status']} -> {change['to_status']}"
    if change["fixed_at_sha"] is not None:
        line += f", fixed at {change['fixed_at_sha']}"
    if change["accepted_by"] is not None:
        line += f", accepted by task {change['accepted_by']}"
    return line


def dispositions_lines(record: Mapping[str, Any], *, observed_root: str) -> list[str]:
    """The report's ``## Dispositions`` section, always present, rendered
    from :meth:`OutcomeReport.to_record`: each previous ticket -> its task ->
    that task's outcome for its entry, then what changed."""
    heading = ["", "## Dispositions", ""]
    if not record["any_back_links"]:
        return [*heading, _NOT_YET_LINKED]
    notes = []
    if record["statuses_error"] is not None:
        notes.append(
            f"- get_statuses could not be read ({record['statuses_error']}); "
            "no status changed for the tasks it could not see."
        )
    counts = record["reverify"]
    notes.append(
        f"- Re-verify at as_of_sha (D8): {counts['due']} due, {counts['offered']} offered, "
        f"{counts['gone']} gone, {counts['present']} present, {counts['inconclusive']} inconclusive, "
        f"{counts['deferred']} deferred by the verify cap."
    )
    changes = [_change_line(change) for change in record["changes"]] or [_NOTHING_CHANGED]
    rows = (_disposition_row(row, observed_root) for row in record["rows"])
    return [*heading, *notes, "", *_table(DISPOSITION_COLUMNS, rows), "", *changes]
