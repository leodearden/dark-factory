"""One census run, preflight to persist, over injected effects."""
from __future__ import annotations

import copy
import functools
import json
import logging
import textwrap
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from legibility import (
    census_filing,
    census_findings,
    census_headroom,
    census_mining,
    census_report,
    census_state,
    census_trigger,
    census_verdicts,
    census_window,
    codebook,
    codebook_lifecycle,
    coder,
    config,
    digest,
    disposition,
    embeddings,
    filing_policy,
    finding_area,
    invariants,
    landing_gate,
    prescreen,
    session_ledger,
    similarity,
    synthesis,
    task_store,
    unlanded,
)
from legibility import verdict as verdict_mod

logger = logging.getLogger("legibility.census.run")


CANDIDATE_KEY = "candidate_id"
"""The verify-cluster key naming the candidate a verdict resolves to: the
reply is matched back by it, never by title."""


def _payload_sighting(session: str | None, payload: Mapping[str, Any]) -> dict:
    return {
        "session": session,
        "origin_phase": payload.get("origin_phase") or "unknown",
        "manifested_phase": payload.get("manifested_phase") or "unknown",
    }


@dataclass(frozen=True)
class _ResolvedClusters:
    """The verification cluster of every candidate a to-verify payload
    resolved to, by candidate id in first-payload order; the ids of the
    uncatalogued ones, in mining order; and how many to-verify payloads the
    merge skipped, which are never offered."""

    clusters: Mapping[str, dict]
    uncatalogued: tuple[str, ...]
    unmerged: int


def _resolved_clusters(screen: prescreen.Screen, merge: codebook.MergeResult) -> _ResolvedClusters:
    """Group *screen*'s to-verify payloads by the candidate *merge* resolved
    each to: one cluster per candidate, its fields the first payload's, seen
    in every payload's session."""
    payloads: dict[str, list[tuple[str | None, Mapping[str, Any]]]] = {}
    unmerged = 0
    for ref in screen.to_verify:
        resolution = merge.resolution_at(ref)
        if resolution is None:
            unmerged += 1
            continue
        record = screen.records[ref.record]
        payloads.setdefault(resolution.candidate_id, []).append(
            (record.get("session"), record["candidates"][ref.candidate]),
        )
    uncatalogued = (merge.resolution_at(ref) for ref in screen.uncatalogued)
    return _ResolvedClusters(
        clusters=MappingProxyType({
            candidate_id: _candidate_cluster(candidate_id, session_payloads)
            for candidate_id, session_payloads in payloads.items()
        }),
        uncatalogued=tuple(dict.fromkeys(
            resolution.candidate_id for resolution in uncatalogued if resolution is not None
        )),
        unmerged=unmerged,
    )


def _candidate_cluster(
    candidate_id: str, session_payloads: Sequence[tuple[str | None, Mapping[str, Any]]],
) -> dict:
    """The verification cluster of *candidate_id* from the (session, payload)
    pairs resolved to it: the first payload's fields, one sighting per
    distinct session."""
    first_session, first = session_payloads[0]
    sightings: dict[str | None, dict] = {}
    for session, payload in session_payloads:
        sightings.setdefault(session, _payload_sighting(session, payload))
    first_sighting = sightings[first_session]
    evidence_quote = first.get("evidence_quote")
    return {
        **census_filing.cluster_record(
            title=first.get("title"),
            cause=first.get("cause"),
            area=first.get("area"),
            origin_phase=first_sighting["origin_phase"],
            manifested_phase=first_sighting["manifested_phase"],
            evidence=[evidence_quote] if evidence_quote else [],
            sightings=list(sightings.values()),
        ),
        CANDIDATE_KEY: candidate_id,
    }


def _read_statuses(status_fetcher) -> tuple[Mapping[str, Any] | None, str | None]:
    """The run's one get_statuses snapshot (``census_trigger.extract_statuses``),
    or None and why it could not be read: an unread snapshot leaves every
    outcome unobserved, and never fails the run."""
    try:
        return census_trigger.extract_statuses(status_fetcher()), None
    except Exception as exc:
        return None, str(exc)


@dataclass(frozen=True)
class _Reverified:
    """The re-verify call's reverifications, and the headroom probes it reported."""

    reverifications: dict[str, disposition.Reverification]
    probes: int = 0


def _reverify(
    due: Sequence[disposition.Reverify],
    *,
    cb: dict,
    verify_fn,
    model: str,
    remaining: int | None,
) -> _Reverified:
    """Re-verify each *due* entry at as_of_sha (D8) in ONE *verify_fn* call,
    the first *remaining* of them when the verify cap sets that, the rest
    deferred. A cap hit inside the call leaves the offered ones inconclusive
    and never defers the run: an unrun re-verify loses nothing, because the
    next run's outcome read queues it again."""
    offered = tuple(due) if remaining is None else tuple(due[:remaining])
    deferred = tuple(due[len(offered):])
    if deferred:
        logger.warning(
            "census: operator verify cap reached -- %d of %d re-verification(s) DEFERRED to the next run",
            len(deferred), len(due),
        )
    if not offered:
        return _Reverified(disposition.reverifications({}, offered=(), deferred=deferred))
    entries = {entry["id"]: entry for entry in cb.get("entries") or []}
    clusters = [
        {
            **{key: value for key, value in census_filing.entry_cluster(cb, entries[item.entry_id]).items() if key != "finding"},
            disposition.REVERIFY_ENTRY: item.entry_id,
        }
        for item in offered
    ]
    try:
        result = verify_fn(clusters, model=model) or {}
    except census_headroom.CensusHeadroomExhausted as exc:
        logger.warning("census: headroom exhausted during re-verification, so it is inconclusive: %s", exc.reason)
        return _Reverified(disposition.reverifications(
            {}, offered=offered, deferred=deferred, error=f"headroom exhausted: {exc.reason}",
        ))
    return _Reverified(
        disposition.reverifications(result, offered=offered, deferred=deferred), _reported_probes(result),
    )


def _reported_probes(result) -> int:
    """The headroom probes a *verify_fn* result reports spending; 0 for a
    result without an int count."""
    reported = result.get("headroom_probes") if isinstance(result, Mapping) else None
    return reported if isinstance(reported, int) else 0


def _pending_cluster(cb: dict, queued: similarity.QueuedCluster) -> dict:
    """The verify cluster of queue cluster *queued*: its representative
    candidate's, seen in every member's sessions."""
    candidates = cb.get("candidates") or []
    representative = codebook.record_by_id(candidates, queued.representative.key, kind="queue representative")
    members = [
        representative,
        *(codebook.record_by_id(candidates, item.key, kind="queue member") for item in queued.others),
    ]
    by_session: dict[str | None, dict] = {}
    for member in members:
        for sighting in member.get("sightings") or []:
            by_session.setdefault(sighting.get("session"), copy.deepcopy(sighting))
    sightings = list(by_session.values())
    first = next(iter(representative.get("sightings") or []), {})
    return {
        **census_filing.cluster_record(
            title=representative["title"],
            cause=representative.get("cause"),
            area=representative.get("area"),
            origin_phase=first.get("origin_phase") or "unknown",
            manifested_phase=first.get("manifested_phase") or "unknown",
            evidence=[sighting["evidence_quote"] for sighting in sightings if sighting.get("evidence_quote")],
            sightings=sightings,
        ),
        CANDIDATE_KEY: representative["id"],
    }


def _escalate_skipped_similarity(reason: str, *, project_id: str, escalate_fn) -> None:
    """Say once, loudly, that this run's similarity stage was skipped."""
    escalate_fn(
        summary=f"legibility census ({project_id}): similarity stage skipped -- {reason}",
        detail=(
            f"census: similarity stage skipped -- {reason}. No similarity attach and no queue "
            "adjudication happened this run; the pending queue is unchanged, and the next run "
            "retries the whole stage."
        ),
        level=logging.WARNING,
    )


_SUMMARY_REASON_CHARS = 200
"""How much of a refused synthesis reply's reason an escalation summary
quotes; its detail carries all of it."""


def _escalate_unavailable_synthesis(reason: str, *, project_id: str, escalate_fn) -> None:
    """Say once, loudly, that this run's synthesis reply was refused."""
    shown = textwrap.shorten(reason, _SUMMARY_REASON_CHARS, placeholder=" ...")
    escalate_fn(
        summary=f"legibility census ({project_id}): synthesis unavailable -- {shown}",
        detail=(
            f"census: synthesis unavailable -- {reason}. Nothing in the reply was applied; every "
            "data section of the report is complete without it, and its ## Synthesis section "
            "says synthesis was unavailable."
        ),
        level=logging.WARNING,
    )


def _synthesize(
    brief: synthesis.Brief, *, synthesize_fn, model: str,
) -> synthesis.Synthesis | synthesis.Unavailable | None:
    """The validated synthesis of *brief*, or why the reply was refused; None,
    with no call made, when *brief* is empty. A raising *synthesize_fn*
    propagates (see :func:`census_headroom.stage_headroom_gate`)."""
    if brief.is_empty:
        return None
    return synthesis.parse(synthesize_fn(brief, model=model), brief)


def _accepted(result: synthesis.Synthesis | synthesis.Unavailable | None) -> synthesis.Synthesis:
    """The document *result* accepted, or the empty one when none was."""
    return result if isinstance(result, synthesis.Synthesis) else synthesis.Synthesis()


@dataclass(frozen=True)
class Synthesized:
    """A run's synthesis before its corrections: the brief, the document
    accepted or why none was (None when no call was made), the run's
    codebook and verified clusters with its phase refinements applied, and
    the origin x manifestation matrix of those clusters."""

    brief: synthesis.Brief
    result: synthesis.Synthesis | synthesis.Unavailable | None
    refined: synthesis.Refined
    matrix_md: str


def _synthesis_stage(
    brief: synthesis.Brief,
    cb: dict,
    *,
    standing: Mapping[str, Any],
    clusters: Sequence[tuple[dict, str | None]],
    synthesize_fn,
    model: str,
    project_id: str,
    escalate_fn,
) -> Synthesized:
    """Synthesize *brief*, then refine the phases of the run's placed
    *clusters* and of every entry of *cb* that is not in *standing*, the
    codebook before the run, and render the matrix of the refined clusters.
    A refused reply is one info escalation and refines nothing."""
    result = _synthesize(brief, synthesize_fn=synthesize_fn, model=model)
    if isinstance(result, synthesis.Unavailable):
        _escalate_unavailable_synthesis(result.reason, project_id=project_id, escalate_fn=escalate_fn)
    stood = {entry["id"] for entry in standing.get("entries") or []}
    refined = synthesis.apply_refinements(
        cb,
        _accepted(result),
        clusters=clusters,
        created=[entry["id"] for entry in cb.get("entries") or [] if entry["id"] not in stood],
    )
    matrix_md = census_report.render_matrix(census_report.compute_matrix(
        [sighting for cluster in refined.clusters for sighting in cluster.get("sightings") or []],
    ))
    return Synthesized(brief, result, refined, matrix_md)


def _apply_synthesis_corrections(
    cb: dict,
    synthesized: Synthesized,
    *,
    standing: Mapping[str, Any],
    run_id: str,
    date: str,
    project_id: str,
    invariant_slugs: Collection[str],
) -> tuple[dict, synthesis.SynthesisReport]:
    """*cb* with the synthesis corrections merged as one coding record, and
    the run's synthesis report, counted against *standing*, the codebook
    before the run."""
    corrections = synthesis.corrections_record(
        _accepted(synthesized.result), run_id=run_id, date=date, project=project_id,
    )
    counts = codebook.MergeCounts()
    if corrections is not None:
        merged = codebook.apply_coding_records(cb, [corrections], invariant_slugs=invariant_slugs)
        cb, counts = merged.codebook, merged.counts
    return cb, synthesis.SynthesisReport(
        brief=synthesized.brief,
        result=synthesized.result,
        applied=synthesis.Applied.of(synthesized.refined, counts),
        standing=synthesis.standing_count(standing, cb),
    )


def _detect_mass_rejection(offered: int, verified: int, *, clusters: str, escalate_fn) -> census_report.MassRejection | None:
    """The anomaly record when *offered* clusters went to one verify call and
    none was verified, logged and escalated; None otherwise. *clusters* names
    what was offered.

    The default verifier fails CLOSED per cluster, so any systemic failure
    (model unreachable mid-run, a permission denial, persistently unparseable
    verdicts) presents as an ordinary all-rejected run: the signature of the
    2026-08-03 sandbox incident, where the verify subprocess was rooted outside
    the censused tree. An all-rejected run is legitimately possible, so this
    never fails or defers a run whose mining is paid for. The returned record
    is the durable trace: the committed report renders it."""
    if not offered or verified:
        return None
    suspect = (
        f"census: ALL {offered} {clusters} cluster(s) were "
        "REJECTED and none survived -- suspect a systemic verifier failure "
        "(model unreachable, tool access denied, or unparseable verdicts) rather "
        "than genuinely unfounded claims. This is the observable signature of the "
        "2026-08-03 sandbox incident, where the verify subprocess was rooted "
        "outside the censused tree and every read was permission-denied. Check "
        "the per-cluster 'verify failed' warnings above; a run with real findings "
        "is being reported as an empty census if this is systemic."
    )
    escalate_fn(
        summary=f"legibility census: all {offered} {clusters} cluster(s) rejected -- possible systemic verifier failure",
        detail=suspect,
        level=logging.WARNING,
    )
    return census_report.MassRejection(offered=offered)


@dataclass(frozen=True)
class QueueOutcome:
    """What the one queue verify call decided, or why it was inconclusive."""

    verified: tuple[dict, ...] = ()
    rejected: tuple[dict, ...] = ()
    inconclusive: str | None = None
    probes: int = 0


def _adjudicate_queue(
    offered: Sequence[similarity.QueuedCluster],
    *,
    cb: dict,
    verify_fn,
    model: str,
    project_root: str,
) -> QueueOutcome:
    """Verify each *offered* queue cluster's representative in ONE
    *verify_fn* call. A cap hit inside it leaves every offered candidate
    pending and never defers the run: the next run queues them again."""
    if not offered:
        return QueueOutcome()
    clusters = [_pending_cluster(cb, queued) for queued in offered]
    try:
        result = verify_fn(clusters, model=model) or {}
    except census_headroom.CensusHeadroomExhausted as exc:
        logger.warning("census: headroom exhausted during queue adjudication, so it is inconclusive: %s", exc.reason)
        return QueueOutcome(inconclusive=f"headroom exhausted: {exc.reason}")
    verified = _offered_replies(result.get("verified") or [], clusters, verdict="verified")
    return QueueOutcome(
        verified=tuple(census_verdicts.verified_with_verdicts(verified, project_root=project_root)),
        rejected=tuple(_offered_replies(result.get("rejected") or [], clusters, verdict="rejected")),
        probes=_reported_probes(result),
    )


def _with_finding(cluster: dict, area_map: finding_area.AreaMap) -> dict:
    """*cluster* carrying the finding identity and severity its Verdict gives;
    a cluster without a Verdict is returned as it is."""
    found = cluster.get("verdict")
    if not isinstance(found, verdict_mod.Verdict):
        return cluster
    return {
        **cluster,
        "finding": census_findings.identity_of(found, area_map),
        "severity": found.severity,
    }


def _promoted_entry_id(cb: dict, candidate: Mapping[str, Any]) -> str | None:
    """The entry promoted *candidate* stands for, when it resolves."""
    entry_id = candidate.get("promoted_to") if candidate.get("disposition") == "promoted" else None
    return entry_id if any(entry.get("id") == entry_id for entry in cb.get("entries") or []) else None


def _offered_replies(replies: Sequence[dict], offered: Sequence[Mapping[str, Any]], *, verdict: str) -> list[dict]:
    """The *replies* of one verify call that name, by CANDIDATE_KEY, a
    candidate of the clusters it was *offered*, each carrying its offered
    cluster's fields over whatever the reply restated, so only what the
    verifier added (its verdict, its remediation) is the reply's; each other
    one is WARNed and discarded before its verdict is read."""
    offered_by_id = {cluster[CANDIDATE_KEY]: cluster for cluster in offered}
    kept = []
    for reply in replies:
        if reply.get(CANDIDATE_KEY) in offered_by_id:
            kept.append({**reply, **offered_by_id[reply[CANDIDATE_KEY]]})
        else:
            logger.warning(
                "census: a %s verdict for %r names no candidate this call offered (%s=%r), so it is discarded",
                verdict, reply.get("title"), CANDIDATE_KEY, reply.get(CANDIDATE_KEY),
            )
    return kept


def _report_dropped_verdicts(
    records: list[census_report.DroppedVerdict], *, disposition_conflicts: int = 0,
) -> None:
    """Announce the run's dropped verdicts: one WARNING per record, naming its
    standing candidate, then ONE run-summary line sizing the total, emitted
    only when either count is non-zero.

    *disposition_conflicts* is the merge's ``candidate_disposition_conflicts``,
    its count of payloads routed to a standing candidate; it rides the same
    line so the two counts read side by side."""
    for record in records:
        logger.warning("census: %s", census_report.dropped_verdict_message(record))
    if not records and not disposition_conflicts:
        return
    logger.warning(
        "census: %d unresolved verdict(s), %d candidate disposition conflict(s) "
        "across the merge -- verdicts that found no pending candidate and were "
        "dropped. These were PAID FOR and went nowhere: a prior adjudication of "
        "the same title is standing and only a hand re-open will change it. See "
        "the per-cluster warnings above for which titles.",
        len(records),
        disposition_conflicts,
    )


def _repo_relative(path, project_root) -> str:
    """*path* relative to *project_root*, both resolved; ``ValueError``
    naming both when *path* lies outside it."""
    try:
        return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        raise ValueError(
            f"report_path {path} is not under project_root {project_root}"
        ) from None


def _require_valid(cb: dict, stage: str) -> None:
    """Raise, before anything is written, if *stage* left *cb* invalid."""
    validation_errors = codebook.validate(cb)
    if validation_errors:
        raise RuntimeError(
            f"census: codebook {stage} produced an invalid codebook: {validation_errors}"
        )


def _ensure_output_parents(*paths) -> None:
    """Create the parent directory of every non-``None`` output path."""
    for path in paths:
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Deferred:
    """A headroom gate failed at *stage*; no census output or state was
    persisted, so a re-run loses nothing."""

    stage: census_headroom.CensusStage
    reason: str
    counts: census_headroom.DeferralCounts = census_headroom.DeferralCounts()


@dataclass(frozen=True)
class Ledgered:
    """A landed run's coded sessions went into the session ledger."""

    rows: int


@dataclass(frozen=True)
class LedgerFailed:
    """A landed run's ledger write failed, so the next census re-mines its
    coded sessions."""

    error: str


@dataclass(frozen=True)
class CrossProjectTicket:
    """A curator ticket this run filed into a project OTHER than the one it
    observed (``filing_policy.resolve_target``). ``resolve_ticket`` must be
    pointed at *project_root* to learn its task id."""

    ticket_id: str
    project_root: str


@dataclass(frozen=True)
class Done:
    """The full pipeline ran and its commit landed. ``ledger`` is ``None``
    when the run's batch source kept no session ledger."""

    report_path: str
    record_path: str
    run_id: str
    stop_reason: str
    gate: landing_gate.GateVerdict
    filing: disposition.FilingReport
    filed_ticket_ids: tuple[str, ...] = ()
    cross_project_tickets: tuple[CrossProjectTicket, ...] = ()
    released_entry_ids: tuple[str, ...] = ()
    dropped_verdicts: tuple[census_report.DroppedVerdict, ...] = ()
    withheld: tuple[filing_policy.WithheldCluster, ...] = ()
    skipped_deletions: tuple[codebook.SkippedRecord, ...] = ()
    dry_run: census_report.DryRunFiling | None = None
    ledger: Ledgered | LedgerFailed | None = None

    @property
    def unresolved_verdicts(self) -> int:
        return len(self.dropped_verdicts)


@dataclass(frozen=True)
class Unlanded:
    """The pipeline ran but its commit did not land, so every written output
    was rolled back; ``rollback`` says where each one is now.
    ``filed_ticket_ids`` names the tickets filed before the commit, if any;
    ``dry_run.path`` names where the payloads file was WRITTEN."""

    rollback: unlanded.Rollback
    gate: landing_gate.GateVerdict
    filed_ticket_ids: tuple[str, ...] = ()
    dry_run: census_report.DryRunFiling | None = None


CensusOutcome = Deferred | Done | Unlanded


def _defer(
    stage: census_headroom.CensusStage,
    reason: str | None,
    *,
    escalate_fn,
    counts: census_headroom.DeferralCounts,
) -> Deferred:
    """Abort the census at *stage*: raise one WARNING alarm, return the outcome.

    ONE owner of the deferral shape, so the preflight gate, the
    stage-boundary gate and the in-verify abort cannot drift into three
    slightly different alarms/outcomes (design-invariants:
    one mechanism, one spelling).
    """
    reason = reason or f"headroom gate failed at the {stage} stage"
    verified, rejected, unverified = counts.verified, counts.rejected, counts.unverified

    # Derived here, not threaded from the raise sites, so they cannot
    # disagree about what "total" means (CensusHeadroomExhausted's INVARIANT
    # guarantees the three terms sum to the clusters offered at every site).
    total = counts.total

    detail = reason
    if stage is not census_headroom.CensusStage.PREFLIGHT:
        # All THREE counts plus the derived total, so the numbers account for
        # every offered cluster. A REJECTION is adjudication already paid for,
        # not an absence: with only verified/unverified, "1 verified, 3
        # unverified" of a 5-cluster run left an operator unable to tell work
        # already spent from a bookkeeping bug. The stage-boundary gate still
        # reports 0 verified and 0 rejected -- nothing was adjudicated there.
        detail = (
            f"{reason}\n\n"
            f"Of {total} novel cluster(s) offered: "
            f"{verified} were verified before the cap and "
            f"{rejected} were rejected before it (both are adjudication "
            f"already paid for); {unverified} were NOT verified. Nothing was "
            "persisted: no report, no matrix, no codebook merge, no filed "
            "tasks, and last_census_at was NOT advanced -- so this window "
            "WILL be re-mined and these sightings are not lost. The mining "
            "spend for this run is sunk. Recovery is a plain re-run once "
            "capacity returns; re-mining is idempotent (apply_coding_records "
            "dedups sightings by session)."
        )

    escalate_fn(
        summary=(
            f"legibility census deferred at the {stage} stage "
            f"({verified} cluster(s) verified, {rejected} rejected, "
            f"{unverified} unverified of {total} offered): {reason}"
            if stage is not census_headroom.CensusStage.PREFLIGHT
            else f"legibility census deferred: {reason}"
        ),
        detail=detail,
        level=logging.WARNING,
    )

    return Deferred(stage=stage, reason=reason, counts=counts)


@dataclass(frozen=True)
class _Refusal:
    """Why a census did not land: the *headline* its escalation summary
    opens with, and the *cause* its detail opens with."""

    headline: str
    cause: str


def _commit_refusal(landing: unlanded.CommitResult) -> _Refusal:
    return _Refusal(
        "legibility census commit did not land",
        f"git commit did not land after {landing.attempts} attempt(s): {landing.stderr}",
    )


def _gate_refusal(verdict: landing_gate.GateVerdict) -> _Refusal:
    cause = verdict.describe()
    if verdict.output_tail:
        cause += f"; its output ends:\n{verdict.output_tail}"
    return _Refusal("legibility census refused by its landing gate", cause)


def _unlanded(
    refusal: _Refusal,
    landed: Done,
    *,
    commit_paths: list[str],
    roll_back,
    escalate_fn,
    project_id: str,
) -> Unlanded:
    """Put back a census that did not land, for *refusal*'s reason: roll
    back, raise one ERROR alarm, return the outcome.

    ONE owner of the ``Unlanded`` shape, as ``_defer`` is of the deferral
    shape: a landing-gate refusal and a commit that did not land differ only
    in *refusal*. *landed* is the outcome the run would have returned had it
    landed; every one of its outputs -- report and dry-run payloads file
    alike -- is rolled back.
    """
    rollback = roll_back(paths=commit_paths)
    if rollback.restored:
        summary = (
            f"{refusal.headline} ({project_id}): outputs "
            "rolled back, census-state NOT advanced"
        )
        state_line = (
            "last_census_at was NOT advanced, so the census trigger will "
            "fire again."
        )
    else:
        summary = (
            f"{refusal.headline} ({project_id}): rollback "
            "INCOMPLETE, outputs still dirty in the checkout"
        )
        state_line = (
            "census-state.json may still carry the advanced last_census_at "
            "in the working tree."
        )
    detail = (
        f"{refusal.cause}\n\n"
        f"{rollback.describe()}\n\n"
        "Remediation tickets filed before the commit: "
        f"{', '.join(landed.filed_ticket_ids) or 'none'}.\n"
        f"{state_line}"
    )
    escalate_fn(summary=summary, detail=detail, level=logging.ERROR)
    return Unlanded(
        rollback=rollback, gate=landed.gate, filed_ticket_ids=landed.filed_ticket_ids, dry_run=landed.dry_run,
    )


def _write_text(path: Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class CensusEffects:
    """Every effect run_census has on the world, each an injected seam."""

    batch_source: Iterable[Sequence[Any]]
    selection_of: Callable[[], census_window.SessionSelection | None]
    invoke: Callable[[str, str], str]
    verify_fn: Callable[..., Mapping[str, Any] | None]
    synthesize_fn: Callable[..., str]
    embedder: embeddings.Embedder
    submit_fn: Callable[..., Any]
    task_store: task_store.TaskStore
    status_fetcher: Callable[[], Any]
    escalate_fn: Callable[..., object]
    gate: Callable[[], landing_gate.GateVerdict]
    commit: Callable[..., unlanded.CommitResult]
    roll_back: Callable[..., unlanded.Rollback]
    write_text: Callable[[Path, str], object] = _write_text
    dump_codebook: Callable[[dict, Path | str], object] = codebook.dump
    advance_state: Callable[..., object] = census_state.advance
    record_codings: Callable[[Path, Sequence[session_ledger.LedgerRow]], int] = session_ledger.record_codings


@dataclass(frozen=True)
class CensusRun:
    """One census run: its identity, the codebook it starts from, its caps and
    its output paths."""

    standing: dict
    config: config.LegibilityConfig
    project_root: str
    project_id: str
    codebook_path: Path | str
    census_state_path: Path | str
    report_path: Path | str
    date: str
    run_id: str
    as_of_sha: str
    since: str | None
    force: bool = False
    max_batches: int | None = None
    max_verify_clusters: int | None = None
    dry_run_payloads_path: Path | str | None = None
    harness_project: filing_policy.ProjectRef | None = None
    started: float = field(default_factory=time.monotonic)
    report_relpath: str = field(init=False)

    def __post_init__(self) -> None:
        if self.max_batches is not None and self.max_batches < 1:
            raise ValueError(
                f"max_batches must be >= 1 when set, got {self.max_batches!r} -- omit it "
                "entirely for unbounded mining"
            )
        if self.max_verify_clusters is not None and self.max_verify_clusters < 1:
            raise ValueError(
                f"max_verify_clusters must be >= 1 when set, got {self.max_verify_clusters!r} "
                "-- a negative cap would slice novel_clusters[:N] from the END rather "
                "than bounding it; omit it entirely to verify every novel cluster"
            )
        object.__setattr__(self, "report_relpath", _repo_relative(self.report_path, self.project_root))


@dataclass(frozen=True)
class Mined:
    """What mining coded, which sessions its source selected, and the project
    inputs read beside it."""

    result: census_mining.MiningResult
    selection: census_window.SessionSelection | None
    invariant_slugs: tuple[str, ...]
    area_map: finding_area.AreaMap
    instrument_findings: prescreen.InstrumentFindings

    @property
    def sessions(self) -> frozenset[str]:
        return frozenset(record["session"] for record in self.result.records if record.get("session"))


def mine(run: CensusRun, effects: CensusEffects) -> Mined:
    """Mine to saturation, then read which sessions the source selected."""
    invariant_slugs = invariants.read_slugs(run.project_root)
    area_map = finding_area.load_area_map(run.project_root)
    for note in area_map.notes:
        logger.warning("census: finding area map: %s", note)
    instrument_findings = prescreen.load_instrument_findings(run.project_root, run.project_id)
    result = census_mining.mine_to_saturation(
        effects.batch_source,
        run.standing,
        project=run.project_id,
        model=run.config.models.census_miner,
        config=run.config.census.saturation,
        invariant_slugs=invariant_slugs,
        invoke=effects.invoke,
        max_batches=run.max_batches,
    )
    return Mined(result, effects.selection_of(), invariant_slugs, area_map, instrument_findings)


@dataclass(frozen=True)
class Merged:
    """The mined batch screened and merged purely into the standing codebook,
    and what of it this run verifies: the novel clusters the cap admits, the
    similarity queue it leaves room for, and the budget left to re-verify."""

    screen: prescreen.Screen
    merge: codebook.MergeResult
    view: similarity.View
    novel: tuple[dict, ...]
    to_verify: tuple[dict, ...]
    coverage: census_report.VerifyCoverage | None
    offered: tuple[similarity.QueuedCluster, ...]
    reverify_budget: int | None
    session_screened: int


def merge(run: CensusRun, effects: CensusEffects, mined: Mined) -> Merged:
    """Screen and merge the mined batch, screen it by similarity, and cap
    what is verified."""
    # The pre-screen (plans/census-incremental-prd.md L6): a cluster already
    # catalogued by title, or from a session already sighted, costs no verify
    # call, and the merge attaches each title-screened sighting through its
    # existing routing. The cap, the stage gate and verify see what is left;
    # a surfaced cluster is verified, never similarity-screened
    # (prescreen.py::Screen.uncatalogued).
    screen = prescreen.screen(mined.result.records, prescreen.Catalogue.of(run.standing), run_id=run.run_id)
    merged = codebook.apply_coding_records(run.standing, screen.records, invariant_slugs=mined.invariant_slugs)
    logger.info("census: merge: %s", merged.summary())
    if merged.skipped_deletions:
        logger.warning(
            "census: %d mined coding record(s) carried a deletion directive and "
            "were skipped: %s",
            len(merged.skipped_deletions),
            "; ".join(skipped.describe() for skipped in merged.skipped_deletions),
        )
    codebook.warn_unknown_invariant_slugs(merged.invariant_slugs, mined.invariant_slugs, run=run.run_id)
    resolved = _resolved_clusters(screen, merged)
    if resolved.unmerged:
        logger.warning(
            "census: %d to-verify payload(s) belong to mined record(s) the merge skipped, so they are not verified",
            resolved.unmerged,
        )
    view = similarity.build_view(
        run.standing,
        {candidate_id: resolved.clusters[candidate_id] for candidate_id in resolved.uncatalogued},
        embedder=effects.embedder,
        settings=run.config.census.adjudication,
    )
    if view.skipped is not None:
        _escalate_skipped_similarity(view.skipped, project_id=run.project_id, escalate_fn=effects.escalate_fn)
    novel = tuple(cluster for candidate_id, cluster in resolved.clusters.items() if not view.screens(candidate_id))
    to_verify, coverage = _verify_cap(novel, run.max_verify_clusters)
    # The pending queue's representatives are verified inside what the verify
    # cap leaves after the novel clusters, and their verdicts take the novel
    # clusters' one path from here on.
    view = view.without({cluster[CANDIDATE_KEY] for cluster in to_verify})
    cap = run.max_verify_clusters
    remaining = None if cap is None else max(0, cap - len(to_verify))
    queue_cap = run.config.census.adjudication.max_clusters_per_run
    offered = view.select(queue_cap if remaining is None else min(queue_cap, remaining))
    return Merged(
        screen=screen,
        merge=merged,
        view=view,
        novel=novel,
        to_verify=to_verify,
        coverage=coverage,
        offered=offered,
        reverify_budget=None if remaining is None else remaining - len(offered),
        session_screened=sum(1 for item in screen.screened if item.rule is prescreen.Rule.SESSION),
    )


def _verify_cap(novel: tuple[dict, ...], cap: int | None) -> tuple[tuple[dict, ...], census_report.VerifyCoverage | None]:
    """The first *cap* of the *novel* clusters, and that cap's coverage; a
    deferred remainder is logged."""
    if cap is None:
        return novel, None
    to_verify = novel[:cap]
    deferred_count = len(novel) - len(to_verify)
    if deferred_count:
        logger.warning(
            "census: operator verify cap (--max-verify-clusters=%d) reached -- "
            "%d of %d novel cluster(s) DEFERRED, not verified this run; they still "
            "merge into the codebook as pending candidates (deferred, never "
            "dropped), but their sessions are ledgered as coded, so a later "
            "census re-adjudicates them only if the same confusion RECURS",
            cap, deferred_count, len(novel),
        )
    return to_verify, census_report.VerifyCoverage(novel=len(novel), offered=len(to_verify), cap=cap)


@dataclass(frozen=True)
class Verdicts:
    """What verification decided for the novel clusters and the queue, the
    alarms it raised, and the headroom probes it spent."""

    verified: tuple[dict, ...]
    rejected: tuple[dict, ...]
    fixed_entry_ids: tuple[str, ...]
    queue: QueueOutcome
    mass_rejection: census_report.MassRejection | None
    queue_mass_rejection: census_report.MassRejection | None
    normalisations: dict[str, int]
    probes: int


def adjudicate(run: CensusRun, effects: CensusEffects, mined: Mined, merged: Merged) -> Verdicts | Deferred:
    """Verify the capped novel clusters and the queue behind a headroom gate,
    or defer at the verify stage."""
    probes = 0
    if merged.to_verify or merged.offered:
        probes += 1
        stage_headroom = census_headroom.stage_headroom_gate(
            effects.invoke, model=run.config.models.trickle, stage=census_headroom.CensusStage.VERIFY,
        )
        if not stage_headroom.ok:
            return _defer(
                census_headroom.CensusStage.VERIFY,
                "headroom gate failed at the verify stage boundary: "
                f"{stage_headroom.reason or 'no headroom'}",
                escalate_fn=effects.escalate_fn,
                counts=census_headroom.DeferralCounts(unverified=len(merged.to_verify)),
            )
    try:
        verify_result = effects.verify_fn(list(merged.to_verify), model=run.config.models.census_verify) or {}
    except census_headroom.CensusHeadroomExhausted as exc:
        return _defer(
            census_headroom.CensusStage.VERIFY,
            f"headroom exhausted during verification: {exc.reason}",
            escalate_fn=effects.escalate_fn,
            counts=exc.counts,
        )
    probes += _reported_probes(verify_result)

    verified = census_verdicts.verified_with_verdicts(
        _offered_replies(verify_result.get("verified") or [], merged.to_verify, verdict="verified"),
        project_root=run.project_root,
    )
    rejected = _offered_replies(verify_result.get("rejected") or [], merged.to_verify, verdict="rejected")
    mass_rejection = _detect_mass_rejection(
        len(merged.to_verify), len(verified), clusters="verified-candidate", escalate_fn=effects.escalate_fn,
    )
    queue = _adjudicate_queue(
        merged.offered,
        cb=run.standing,
        verify_fn=effects.verify_fn,
        model=run.config.models.census_verify,
        project_root=run.project_root,
    )
    queue_mass_rejection = None if queue.inconclusive is not None else _detect_mass_rejection(
        len(merged.offered), len(queue.verified), clusters="pending-queue", escalate_fn=effects.escalate_fn,
    )
    verified = [*verified, *queue.verified]
    rejected = [*rejected, *queue.rejected]
    return Verdicts(
        verified=tuple(_with_finding(cluster, mined.area_map) for cluster in verified),
        rejected=tuple(_with_finding(cluster, mined.area_map) for cluster in rejected),
        fixed_entry_ids=tuple(verify_result.get("fixed") or []),
        queue=queue,
        mass_rejection=mass_rejection,
        queue_mass_rejection=queue_mass_rejection,
        normalisations=verdict_mod.count_normalisations(
            cluster["verdict"]
            for cluster in (*verified, *rejected)
            if isinstance(cluster.get("verdict"), verdict_mod.Verdict)
        ),
        probes=probes + queue.probes,
    )


@dataclass(frozen=True)
class _VerifiedPlacements:
    """Where each verified cluster landed, and what landing it decided."""

    placed: tuple[tuple[dict, str | None], ...]
    fileable: tuple[tuple[dict, str | None], ...]
    held_back: tuple[tuple[filing_policy.WithheldCluster, str], ...]
    promoted: tuple[census_findings.Adjudication, ...]
    dropped: tuple[census_report.DroppedVerdict, ...]
    key_attaches: tuple[prescreen.KeyAttach, ...]
    surfaced: tuple[prescreen.Surfaced, ...]
    re_observed: tuple[prescreen.ReObserved, ...]


def _place_verified(
    cb: dict, clusters: Sequence[dict], *, run_id: str, instrument_findings: prescreen.InstrumentFindings,
) -> tuple[dict, _VerifiedPlacements]:
    """Promote, key-attach or drop each verified cluster, gating its filing at
    the distinct sessions of the record it lands on."""
    # Each verified cluster's filing is gated at the distinct sessions of the
    # record it lands on: its candidate, counted BEFORE promotion so a withheld
    # entry is marked as it is created; its entry, after a key attach; or the
    # standing record a dropped verdict met. A structural cluster (contract
    # §6, D6) is promoted but never filed.
    placed: list[tuple[dict, str | None]] = []
    fileable: list[tuple[dict, str | None]] = []
    held_back: list[tuple[filing_policy.WithheldCluster, str]] = []
    promoted: list[census_findings.Adjudication] = []
    dropped: list[census_report.DroppedVerdict] = []
    key_attaches: list[prescreen.KeyAttach] = []
    surfaced: list[prescreen.Surfaced] = []
    re_observed: list[prescreen.ReObserved] = []
    for cluster in clusters:
        cand_id = cluster[CANDIDATE_KEY]
        candidate = codebook.record_by_id(cb["candidates"], cand_id, kind="verified candidate")
        if candidate["disposition"] != "pending":
            placed.append((cluster, None))
            dropped.append(census_report.dropped_verdict(
                verdict="verified", cluster=cluster, standing=candidate, contradicting_disposition="rejected",
            ))
            standing_entry_id = _promoted_entry_id(cb, candidate)
            holder = (
                candidate if standing_entry_id is None
                else codebook.record_by_id(cb["entries"], standing_entry_id, kind="standing entry")
            )
            files, _ = census_filing.filing_gate(cluster, census_filing.distinct_sessions(holder))
            if files:
                fileable.append((cluster, standing_entry_id))
            continue
        found = cluster["verdict"]
        # The key and cross-instrument rules (plans/census-incremental-prd.md
        # L6): a verdict at a live entry's (area, anchor), this run's
        # promotions included, joins that entry, which keeps its key; any
        # other is promoted, under the key of another instrument's finding
        # at its location when there is one.
        placement = prescreen.place(cb, cluster["title"], cluster["finding"], instrument_findings)
        if placement.surfaced is not None:
            surfaced.append(placement.surfaced)
        if placement.standing_entry_id is not None:
            cb, key_attach = prescreen.attach_by_key(
                cb, cand_id, placement.standing_entry_id, found, run_id=run_id,
            )
            key_attaches.append(key_attach)
            entry_id = key_attach.entry_id
            attached_entry = codebook.record_by_id(cb["entries"], entry_id, kind="key-attached entry")
            files, held = census_filing.filing_gate(cluster, census_filing.distinct_sessions(attached_entry))
            if entry_id not in {adjudication.record_id for adjudication in promoted}:
                promoted.append(census_findings.Adjudication(entry_id, found, cluster.get("cause")))
        else:
            entry_id = f"entry-{cand_id}"
            files, held = census_filing.filing_gate(cluster, census_filing.distinct_sessions(candidate))
            if placement.adoption is not None:
                re_observed.append(prescreen.ReObserved(cluster["title"], entry_id, placement.adoption))
            entry_fields = {
                "id": entry_id,
                "title": cluster.get("title"),
                "severity": found.severity,
                "severity_reason": found.severity_reason,
                "status": "open",
                "origin_phase": cluster.get("origin_phase") or "unknown",
                "manifested_phase": cluster.get("manifested_phase") or "unknown",
                **census_findings.first_sighting_fields(run_id),
                "first_seen_run": placement.first_seen_run(run_id),
            }
            cb = codebook_lifecycle.promote_candidate(
                cb, cand_id, entry_fields, finding=placement.identity,
                supersedes=placement.supersedes, withheld=held is not None,
            )
            promoted.append(census_findings.Adjudication(entry_id, found, cluster.get("cause")))
        placed.append((cluster, entry_id))
        if held is not None:
            held_back.append((held, entry_id))
        if files:
            fileable.append(({**cluster, "finding": placement.identity}, entry_id))
    return cb, _VerifiedPlacements(
        placed=tuple(placed),
        fileable=tuple(fileable),
        held_back=tuple(held_back),
        promoted=tuple(promoted),
        dropped=tuple(dropped),
        key_attaches=tuple(key_attaches),
        surfaced=tuple(surfaced),
        re_observed=tuple(re_observed),
    )


@dataclass(frozen=True)
class Placed:
    """The codebook with every verdict placed, and what filing, synthesis and
    the record read of that placement."""

    codebook: dict
    placed: tuple[tuple[dict, str | None], ...]
    fileable: tuple[tuple[dict, str | None], ...]
    withheld: tuple[filing_policy.WithheldCluster, ...]
    structural: tuple[disposition.StructuralFinding, ...]
    promoted: tuple[census_findings.Adjudication, ...]
    refuted: tuple[census_findings.Adjudication, ...]
    re_observed: tuple[prescreen.ReObserved, ...]
    dropped_verdicts: tuple[census_report.DroppedVerdict, ...]
    sighted: tuple[str, ...]
    attached: tuple[similarity.AppliedAttach, ...]
    adjudicated: tuple[similarity.Adjudicated, ...]
    screen_report: prescreen.ScreenReport
    adjudication_report: similarity.AdjudicationReport


def place_verdicts(run: CensusRun, mined: Mined, merged: Merged, verdicts: Verdicts) -> Placed:
    """Apply every verdict to the merged codebook: similarity attaches,
    promotions and rejections, queue members, retirements and this run's
    sightings."""
    cb, attached = similarity.attach(merged.merge.codebook, merged.view, run_id=run.run_id)
    cb, verified = _place_verified(
        cb, verdicts.verified, run_id=run.run_id, instrument_findings=mined.instrument_findings,
    )
    # A held cluster whose entry files this run anyway, through a cluster
    # that joined it, is not withheld.
    filed_entry_ids = {entry_id for _, entry_id in verified.fileable}
    withheld = tuple(held for held, entry_id in verified.held_back if entry_id not in filed_entry_ids)
    structural = disposition.structural_findings(
        cb, [adjudication.record_id for adjudication in verified.promoted if not disposition.files(adjudication.found)],
    )
    if withheld:
        logger.info(
            "census: %d verified cluster(s) recorded but not filed (no in-tree "
            "remediation, fewer than %d sightings): %s",
            len(withheld), filing_policy.MIN_UNREMEDIATED_SIGHTINGS,
            [cluster.title for cluster in withheld],
        )

    dropped_verdicts = list(verified.dropped)
    refuted: list[census_findings.Adjudication] = []
    for cluster in verdicts.rejected:
        cand_id = cluster[CANDIDATE_KEY]
        candidate = codebook.record_by_id(cb["candidates"], cand_id, kind="rejected candidate")
        if candidate["disposition"] != "pending":
            dropped_verdicts.append(census_report.dropped_verdict(
                verdict="rejected", cluster=cluster, standing=candidate, contradicting_disposition="promoted",
            ))
            continue
        cb = codebook_lifecycle.reject_candidate(cb, cand_id, finding=cluster.get("finding"))
        if cluster.get("finding") is not None:
            refuted.append(census_findings.Adjudication(cand_id, cluster["verdict"], cluster.get("cause")))

    cb, adjudicated = similarity.adjudicate_members(cb, merged.offered)

    for entry_id in dict.fromkeys(verdicts.fixed_entry_ids):
        if codebook.record_by_id(cb["entries"], entry_id, kind="fixed entry").get("status") == "retired":
            logger.warning("census: the verifier reported entry %r fixed, but it is already retired", entry_id)
            continue
        cb = codebook_lifecycle.retire_entry(cb, entry_id)

    cb, sighted = census_findings.mark_seen(cb, run_id=run.run_id, sessions=mined.sessions)

    _report_dropped_verdicts(
        dropped_verdicts,
        disposition_conflicts=merged.merge.counts.candidate_disposition_conflicts,
    )
    return Placed(
        codebook=cb,
        placed=verified.placed,
        fileable=verified.fileable,
        withheld=withheld,
        structural=structural,
        promoted=verified.promoted,
        refuted=tuple(refuted),
        re_observed=verified.re_observed,
        dropped_verdicts=tuple(dropped_verdicts),
        sighted=sighted,
        attached=attached,
        adjudicated=adjudicated,
        screen_report=prescreen.ScreenReport(
            merged.screen, verified.surfaced, verified.key_attaches, verified.re_observed,
            mined.instrument_findings,
        ),
        adjudication_report=similarity.AdjudicationReport(
            merged.view, attached, merged.offered, adjudicated, verdicts.queue.inconclusive,
            verdicts.queue_mass_rejection is not None,
        ),
    )


def synthesize(
    run: CensusRun, effects: CensusEffects, mined: Mined, verdicts: Verdicts, placed: Placed,
) -> Synthesized:
    """Synthesize this run's findings, refine its phases, render its matrix,
    and hold the refined codebook to the schema."""
    # Synthesis (plans/census-incremental-prd.md L10, C6) runs after the key
    # rule, the attaches and adjudication, so the brief's keys and its new /
    # re-observed split are exact. Its phase refinements reach this run's
    # verified clusters, which the matrix is rendered from, and the entries
    # this run created, before the codebook is validated, filed from or
    # written.
    brief = synthesis.Brief.of(
        run_id=run.run_id,
        before=run.standing,
        after=placed.codebook,
        verdicts=placed.promoted,
        adopted={item.entry_id: item.adoption.external.run_id for item in placed.re_observed},
        run_sessions=mined.sessions,
        counts={
            "mined": len(mined.result.records),
            "confirmed": len(verdicts.verified),
            "refuted": len(verdicts.rejected),
            **{f"screened_{rule}": count for rule, count in placed.screen_report.counts().items()},
            **placed.adjudication_report.counts(),
        },
    )
    synthesized = _synthesis_stage(
        brief, placed.codebook, standing=run.standing, clusters=placed.placed, synthesize_fn=effects.synthesize_fn,
        model=run.config.models.census_synthesis, project_id=run.project_id, escalate_fn=effects.escalate_fn,
    )
    _require_valid(synthesized.refined.codebook, "merge")
    return synthesized


@dataclass(frozen=True)
class FilingPlan:
    """Everything persist files and writes: the run's stage values, the
    codebook before filing, the filings to dispose, and the outcome feedback
    that shaped them."""

    run: CensusRun
    mined: Mined
    merged: Merged
    verdicts: Verdicts
    placed: Placed
    synthesized: Synthesized
    codebook: dict
    filings: tuple[disposition.Filing, ...]
    withheld_entry_ids: frozenset[str]
    outcomes: disposition.Outcomes
    reverified: Mapping[str, disposition.Reverification]
    status_changes: tuple[disposition.StatusChange, ...]
    refiled: tuple[disposition.Refiling, ...]
    backfilled: disposition.Backfill
    reverify_probes: int


def plan_filing(
    run: CensusRun,
    effects: CensusEffects,
    mined: Mined,
    merged: Merged,
    verdicts: Verdicts,
    placed: Placed,
    synthesized: Synthesized,
) -> FilingPlan:
    """Link historical tasks, read and re-verify the outcomes of earlier
    filings, and queue this run's filings with their payloads."""
    # Historical census tasks link to their entries before this run files
    # anything (plans/census-incremental-prd.md L4), in dry runs too: it is a
    # read plus a codebook update. disposition.backfill never raises and logs
    # its own failure.
    cb, backfilled = disposition.backfill(
        synthesized.refined.codebook, store=effects.task_store, project_root=run.project_root, run_id=run.run_id,
    )
    # Outcome feedback (plans/census-incremental-prd.md L5, D8), after the
    # backfill so the tasks it linked are read this run. Re-verifies take
    # only what the verify cap leaves after the novel clusters.
    statuses, statuses_error = _read_statuses(effects.status_fetcher)
    outcomes = disposition.read_outcomes(
        cb, statuses=statuses, statuses_error=statuses_error,
        observed_root=run.project_root, store=effects.task_store,
    )
    reverified = _reverify(
        outcomes.due,
        cb=cb,
        verify_fn=effects.verify_fn,
        model=run.config.models.census_verify,
        remaining=merged.reverify_budget,
    )
    cb, status_changes = disposition.project(cb, outcomes, reverified.reverifications, as_of_sha=run.as_of_sha)
    refiled = disposition.refilings(outcomes, reverified.reverifications, sighted=placed.sighted)
    queue = census_filing.with_refilings(census_filing.filing_queue(list(placed.fileable), cb), refiled, cb)
    payloads = census_filing.build_task_payloads(
        [pending.cluster for pending in queue],
        project_root=run.project_root, project_id=run.project_id, run_id=run.run_id,
        harness=run.harness_project,
    )
    return FilingPlan(
        run=run,
        mined=mined,
        merged=merged,
        verdicts=verdicts,
        placed=placed,
        synthesized=synthesized,
        codebook=cb,
        filings=tuple(pending.filing(payload) for pending, payload in zip(queue, payloads, strict=True)),
        withheld_entry_ids=frozenset(
            pending.withheld_entry_id for pending in queue if pending.withheld_entry_id is not None
        ),
        outcomes=outcomes,
        reverified=reverified.reverifications,
        status_changes=status_changes,
        refiled=refiled,
        backfilled=backfilled,
        reverify_probes=reverified.probes,
    )


def _after_filing(plan: FilingPlan, disposal: disposition.Disposal) -> tuple[dict, tuple[str, ...]]:
    """The plan's codebook with every withheld entry this run routed marked
    filed, each routed decision's back-link and the run's follow-ups; and the
    ids of those released entries."""
    run = plan.run
    released_entry_ids = tuple(dict.fromkeys(
        decision.filing.entry_id
        for decision in disposal.decisions
        if decision.routed and decision.filing.entry_id in plan.withheld_entry_ids
    ))
    cb = plan.codebook
    for entry_id in released_entry_ids:
        cb = codebook_lifecycle.mark_entry_filed(cb, entry_id)
    if released_entry_ids:
        logger.info(
            "census: %d withheld codebook entr(ies) recurred and were routed: %s",
            len(released_entry_ids), list(released_entry_ids),
        )
    cb = disposition.with_back_links(
        cb, [link for decision in disposal.decisions if (link := decision.back_link(run.run_id)) is not None],
    )
    cb = disposition.with_follow_ups(
        cb, reverified=plan.reverified, refiled=plan.refiled, decisions=disposal.decisions,
        run_id=run.run_id, as_of_sha=run.as_of_sha,
    )
    return cb, released_entry_ids


def _storm_batch_indices(mining_result: census_mining.MiningResult) -> list[int]:
    """The indices of the mined batches whose coding storm-failed, logged."""
    indices = [s.index for s in mining_result.batch_stats if s.status == coder.CodingStatus.FAILURE]
    if indices:
        logger.warning(
            "census: %d storm batch(es) (>50%% coding failures, PRD §8.6) at "
            "indices %s -- dup_rate for those batches was excluded from the "
            "saturation decision but is still recorded in the report",
            len(indices), indices,
        )
    return indices


def _census_record(
    plan: FilingPlan,
    cb: dict,
    *,
    filing_report: disposition.FilingReport,
    synthesis_report: synthesis.SynthesisReport,
    dry_run: census_report.DryRunFiling | None,
) -> dict:
    """The run's census record, over *cb*, the codebook it persists."""
    run, mined, merged, verdicts, placed = plan.run, plan.mined, plan.merged, plan.verdicts, plan.placed
    config, mining_result = run.config, mined.result
    standings = disposition.task_standings(plan.outcomes, filing_report.disposal.resolved(
        filing_report.resolutions,
    ).decisions)
    dispositions_record = disposition.OutcomeReport(
        plan.outcomes, plan.reverified, plan.status_changes, plan.refiled,
        disposition.entry_dispositions(cb, standings),
    ).to_record()
    verify_calls = len(merged.to_verify) + len(merged.offered) + dispositions_record["reverify"]["offered"]
    probe_calls = 1 + verdicts.probes + plan.reverify_probes
    storm_batch_indices = _storm_batch_indices(mining_result)
    # verify is ONE call PER CLUSTER (census_verify.build_verify_fn), not one call
    # total -- and the per-cluster count, novel clusters plus queue clusters
    # plus re-verifies, is precisely what an operator using
    # --max-verify-clusters reads this line to check. headroom-probe is the
    # run's REAL probe spend: the preflight and stage gate PLUS whatever the
    # verifier reported spending on its own in-verify detectors.
    cost_note = (
        f"invoke calls: {config.models.census_miner} miner="
        f"{sum(s.total for s in mining_result.batch_stats)}, "
        f"{config.models.census_verify} verify={verify_calls}, "
        f"{config.models.census_synthesis} synthesis={synthesis_report.calls}, "
        f"{config.models.trickle} headroom-probe={probe_calls}"
    )
    if storm_batch_indices:
        cost_note += (
            f"; WARNING: {len(storm_batch_indices)} storm batch(es) at indices "
            f"{storm_batch_indices} (>50% coding failures -- degraded dup-rate "
            "signal, excluded from the saturation decision)"
        )
    method = census_report.build_method(
        run_id=run.run_id,
        as_of_sha=run.as_of_sha,
        since=run.since,
        selection=mined.selection,
        mined=len(mining_result.records),
        confirmed=len(verdicts.verified),
        refuted=len(verdicts.rejected),
        unverified=len(merged.novel) - len(merged.to_verify) + merged.session_screened,
        miner_calls=sum(s.total for s in mining_result.batch_stats),
        verify_calls=verify_calls,
        synthesis_calls=synthesis_report.calls,
        probe_calls=probe_calls,
        wall_clock_secs=round(time.monotonic() - run.started, 3),
        verify_normalisations=verdicts.normalisations,
        reverify=dispositions_record["reverify"],
        screened=placed.screen_report.counts(),
        inputs_consumed=placed.screen_report.inputs.consumed,
        inputs_consumed_note=placed.screen_report.inputs.note,
        embedding_calls=merged.view.requests,
        adjudication=placed.adjudication_report.counts(),
        slug_tally=merged.merge.invariant_slugs,
    )
    return census_report.build_census_record(
        method=method,
        date=run.date,
        project_id=run.project_id,
        force=run.force,
        matrix_md=plan.synthesized.matrix_md,
        mining_result=mining_result,
        synthesis=synthesis_report.to_record(),
        project_root=run.project_root,
        filing=filing_report.to_record(),
        dispositions=dispositions_record,
        screened=placed.screen_report.to_record(),
        adjudication=placed.adjudication_report.to_record(),
        cost_note=cost_note,
        verify_coverage=merged.coverage,
        dry_run=dry_run,
        dropped_verdicts=tuple(placed.dropped_verdicts),
        mass_rejection=verdicts.mass_rejection,
        withheld=placed.withheld,
        findings=[
            row.to_record()
            for row in census_findings.finding_rows(
                cb,
                promoted=placed.promoted,
                refuted=placed.refuted,
                sighted=placed.sighted,
                run_id=run.run_id,
                cause_of=functools.partial(codebook.entry_field, cb, field="cause"),
                disposition_of=lambda entry: str(disposition.disposition_of(entry, standings)),
            )
        ],
        area_map=mined.area_map.to_record(),
    )


def _landed(
    plan: FilingPlan,
    *,
    filing_report: disposition.FilingReport,
    released_entry_ids: tuple[str, ...],
    dry_run: census_report.DryRunFiling | None,
    gate: landing_gate.GateVerdict,
) -> Done:
    """The outcome the run returns once its commit lands."""
    run = plan.run
    decisions = filing_report.disposal.decisions
    return Done(
        report_path=str(run.report_path),
        run_id=run.run_id,
        record_path=str(Path(run.report_path).with_suffix(".json")),
        filed_ticket_ids=tuple(decision.ticket for decision in decisions if decision.ticket is not None),
        stop_reason=plan.mined.result.stop_reason,
        dry_run=dry_run,
        dropped_verdicts=tuple(plan.placed.dropped_verdicts),
        withheld=plan.placed.withheld,
        cross_project_tickets=tuple(
            CrossProjectTicket(ticket_id=decision.ticket, project_root=decision.filing.project_root)
            for decision in decisions
            if decision.ticket is not None and decision.filing.project_root != run.project_root
        ),
        released_entry_ids=released_entry_ids,
        skipped_deletions=plan.merged.merge.skipped_deletions,
        filing=filing_report,
        gate=gate,
    )


def persist(plan: FilingPlan, effects: CensusEffects) -> Done | Unlanded:
    """File the plan's tickets, write its record, report, codebook and census
    state, gate and commit them or roll them back, then ledger its sessions."""
    run = plan.run
    disposal = disposition.dispose(
        list(plan.filings),
        store=effects.task_store,
        submit=None if run.dry_run_payloads_path is not None else effects.submit_fn,
        awaiting=plan.outcomes.awaiting,
    )
    dry_run = None
    if run.dry_run_payloads_path is not None:
        dry_run = census_filing.write_dry_run_payloads(
            Path(run.dry_run_payloads_path),
            [dict(filing.payload) for filing in disposal.unfiled],
            withheld_offered=sum(1 for filing in disposal.unfiled if filing.entry_id in plan.withheld_entry_ids),
            write_text=effects.write_text,
        )
    cb, released_entry_ids = _after_filing(plan, disposal)
    cb, resolutions = disposition.resolve_tickets(cb, store=effects.task_store, run_id=run.run_id)
    # Synthesis corrections land after filing, so this run's filing pairing
    # and sighting counts read the codebook the run verified.
    cb, synthesis_report = _apply_synthesis_corrections(
        cb, plan.synthesized, standing=run.standing, run_id=run.run_id, date=run.date,
        project_id=run.project_id, invariant_slugs=plan.mined.invariant_slugs,
    )
    _require_valid(cb, "filing")
    filing_report = disposition.FilingReport(disposal, resolutions, plan.backfilled, plan.placed.structural)
    record = _census_record(plan, cb, filing_report=filing_report, synthesis_report=synthesis_report, dry_run=dry_run)

    record_path = Path(run.report_path).with_suffix(".json")
    record_text = json.dumps(record, indent=2) + "\n"
    effects.write_text(record_path, record_text)
    effects.write_text(Path(run.report_path), census_report.render_report(json.loads(record_text)))
    effects.dump_codebook(cb, run.codebook_path)
    selection = plan.mined.selection
    effects.advance_state(
        run.census_state_path,
        census_at=datetime.fromisoformat(run.date).date(),
        run_id=run.run_id,
        report_path=run.report_relpath,
        as_of_sha=run.as_of_sha,
        session_watermark=selection.window.end if selection is not None else None,
    )

    verdict = effects.gate()
    landed = _landed(
        plan, filing_report=filing_report, released_entry_ids=released_entry_ids, dry_run=dry_run, gate=verdict,
    )
    commit_paths = [str(run.report_path), str(record_path), str(run.codebook_path), str(run.census_state_path)]
    if dry_run is not None:
        commit_paths.append(dry_run.path)
    unlanded_as = functools.partial(
        _unlanded, landed=landed, commit_paths=commit_paths, roll_back=effects.roll_back,
        escalate_fn=effects.escalate_fn, project_id=run.project_id,
    )
    if not verdict.passed:
        return unlanded_as(_gate_refusal(verdict))
    landing = effects.commit(paths=commit_paths, message=f"legibility census {run.date}")
    if not landing.ok:
        return unlanded_as(_commit_refusal(landing))
    if selection is None:
        return landed
    return _ledger_census_codings(
        landed, list(plan.merged.merge.merged_records), run_id=run.run_id, ledger=selection.ledger,
        record_codings=effects.record_codings,
    )


def run_census(
    *,
    effects: CensusEffects,
    codebook_dict: dict,
    config,
    project_root: str,
    project_id: str,
    codebook_path,
    census_state_path,
    report_path,
    date: str,
    run_id: str,
    as_of_sha: str,
    since: str | None,
    force: bool = False,
    max_batches: int | None = None,
    max_verify_clusters: int | None = None,
    dry_run_payloads_path: str | Path | None = None,
    harness_project: filing_policy.ProjectRef | None = None,
) -> CensusOutcome:
    """Run one periodic legibility census end to end: preflight, then
    ``mine``, ``merge``, ``adjudicate``, ``place_verdicts``, ``synthesize``,
    ``plan_filing`` and ``persist``.

    Only ``persist`` writes census outputs and state, so a ``Deferred`` -- at
    preflight or at the verify gate -- leaves none behind and a re-run loses
    nothing. Before ``persist`` the run writes only the output directories
    and, in ``merge``, the embedder's vector cache. What the
    operator caps (*max_batches*, *max_verify_clusters*) and
    *dry_run_payloads_path* cost and leave behind is stated in ``census.main``'s
    flag help and in ``census_report.VerifyCoverage``.
    """
    run = CensusRun(
        standing=codebook_dict,
        config=config,
        project_root=project_root,
        project_id=project_id,
        codebook_path=codebook_path,
        census_state_path=census_state_path,
        report_path=report_path,
        date=date,
        run_id=run_id,
        as_of_sha=as_of_sha,
        since=since,
        force=force,
        max_batches=max_batches,
        max_verify_clusters=max_verify_clusters,
        dry_run_payloads_path=dry_run_payloads_path,
        harness_project=harness_project,
    )
    headroom = census_headroom.preflight_headroom(effects.invoke, model=config.models.trickle)
    if not headroom.ok:
        return _defer(
            census_headroom.CensusStage.PREFLIGHT,
            headroom.reason or "headroom preflight failed",
            escalate_fn=effects.escalate_fn,
            counts=census_headroom.DeferralCounts(),
        )

    # Before mining, so an uncreatable output path fails fast; after
    # preflight, so a defer creates nothing.
    _ensure_output_parents(
        report_path, codebook_path, census_state_path, dry_run_payloads_path,
    )

    mined = mine(run, effects)
    merged = merge(run, effects, mined)
    verdicts = adjudicate(run, effects, mined, merged)
    if isinstance(verdicts, Deferred):
        return verdicts
    placed = place_verdicts(run, mined, merged, verdicts)
    synthesized = synthesize(run, effects, mined, verdicts, placed)
    return persist(plan_filing(run, effects, mined, merged, verdicts, placed, synthesized), effects)


def _ledger_census_codings(
    landed: Done,
    records: list[dict],
    *,
    run_id: str,
    ledger: session_ledger.LedgerSnapshot,
    record_codings: Callable[[Path, Sequence[session_ledger.LedgerRow]], int],
) -> Done:
    """Ledger every session this landed run coded, so the next census skips
    it. A failure costs only a re-mine of those sessions, so it is reported
    on the outcome rather than raised."""
    rows = session_ledger.rows_for(
        records,
        coded_by=session_ledger.CodedBy.CENSUS,
        run_ref=run_id,
        instrument_version=digest.DIGEST_INSTRUMENT_VERSION,
        coded_at=datetime.now(UTC),
    )
    try:
        written = record_codings(ledger.path, rows)
    except session_ledger.LedgerError as exc:
        logger.warning(
            "census: %d coded session(s) NOT ledgered, the next census re-mines them: %s",
            len(rows), exc,
        )
        return replace(landed, ledger=LedgerFailed(str(exc)))
    return replace(landed, ledger=Ledgered(written))
