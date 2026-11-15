"""Which clusters a census run files, and the submit_task payloads it files them with."""
from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from legibility import (
    census_findings,
    census_report,
    census_verdicts,
    codebook,
    disposition,
    filing_policy,
)
from legibility import verdict as verdict_mod
from shared.toolcall_markup import escape_envelope_literals

logger = logging.getLogger("legibility.census.filing")


def _cluster_description(cluster: dict, *, observed_project_id: str) -> str:
    """Factual cluster summary + evidence -- no prose routing intent
    (lesson `prose-routing-intent`: routing is expressed structurally, via
    payload fields, never as English directives embedded in the text)."""
    lines = [
        cluster.get("summary")
        or cluster.get("title")
        or "Confusion cluster observed by the periodic legibility census."
    ]

    evidence = cluster.get("evidence") or []
    if evidence:
        lines.append("")
        lines.append("Evidence:")
        lines.extend(f"- {quote}" for quote in evidence)

    sightings = cluster.get("sightings") or []
    if sightings:
        lines.append("")
        lines.append(f"Observed in {len(sightings)} sighting(s) (project: {observed_project_id}).")

    remediation = filing_policy.proposed_remediation(cluster)
    if remediation is not None:
        lines.append("")
        lines.append(f"Proposed remediation: {remediation['path']} -- {remediation['change']}")

    return "\n".join(lines)


def build_task_payloads(
    clusters,
    *,
    project_root: str,
    project_id: str,
    run_id: str,
    harness: filing_policy.ProjectRef | None = None,
) -> list[dict]:
    """Map each verified cluster to one curator-path ``submit_task`` kwarg
    dict. ``task_kind`` is always ``"normal"``; ``planning_mode`` is
    deliberately OMITTED (defaults False at the submit_task layer) --
    curator dedup against already-filed remediation is the point (PRD
    decision 9), and ``planning_mode`` is exactly the curator-bypassing
    path that would defeat it.

    *project_root*/*project_id* name the OBSERVED project. Which project each
    payload files into is decided by ``filing_policy.resolve_target``; a
    payload routed to *harness* carries the matched markers under
    ``metadata.x_fix_surface``, and ``metadata.origin_project_id`` always names
    the observed project.

    Every payload names its run under ``metadata.x_finding_run``, and a cluster
    carrying a ``finding`` identity names its key under
    ``metadata.x_finding_key`` (quality-findings contract §8);
    ``x_supersedes_task`` and its title are written afterwards, by
    ``disposition.Filing.superseding`` (contract §8 step 1c). ``priority`` is
    the cluster's severity (contract §4).

    ``title`` and ``description`` pass through
    ``shared.toolcall_markup::escape_envelope_literals`` because cluster
    evidence may QUOTE tool-call envelope literals, which the boundary
    markup guard refuses (task 5907).

    Pure function -- returns payloads only; ``disposition.dispose`` decides
    which of them ``submit_fn`` files.
    """
    observed = filing_policy.ProjectRef(project_root=project_root, project_id=project_id)
    payloads = []
    for cluster in clusters:
        target = filing_policy.resolve_target(cluster, observed=observed, harness=harness)
        metadata: dict = {
            "source": disposition.CENSUS_SOURCE,
            "origin_project_id": project_id,
            "x_finding_run": run_id,
        }
        finding = cluster.get("finding")
        if finding is not None:
            metadata["x_finding_key"] = finding.key
        if target.fix_surface:
            metadata["x_fix_surface"] = [
                {"component": match.component, "evidence": match.evidence}
                for match in target.fix_surface
            ]
        payloads.append(
            {
                "project_root": target.project.project_root,
                "title": disposition.task_title(_cluster_title(cluster)),
                "description": escape_envelope_literals(
                    _cluster_description(cluster, observed_project_id=project_id)
                ),
                "task_kind": "normal",
                "priority": cluster.get("severity", verdict_mod.DEFAULT_SEVERITY),
                "metadata": metadata,
            }
        )
    return payloads


def _cluster_title(cluster: Mapping[str, Any]) -> str:
    return cluster.get("title") or "Untitled confusion cluster"


def cluster_record(
    *, title, cause, area, origin_phase, manifested_phase, evidence, sightings,
) -> dict:
    """The one shape of a verification/filing cluster, whichever record it
    was built from."""
    return {
        "title": title,
        "summary": cause or title,
        "cause": cause,
        "area": area,
        "origin_phase": origin_phase,
        "manifested_phase": manifested_phase,
        "evidence": evidence,
        "sightings": sightings,
    }


def distinct_sessions(record: Mapping[str, Any]) -> int:
    """How many distinct sessions *record*'s own sightings carry, whatever its
    disposition or status: the singleton gate asks only whether a confusion
    recurred, and this run's verifier has already judged whether it is real."""
    return len(set(codebook.sessions_of(record)))


def filing_gate(cluster: dict, sighting_count: int) -> tuple[bool, filing_policy.WithheldCluster | None]:
    """Whether this run files verified *cluster*, seen in *sighting_count*
    distinct sessions and at least in those its own sightings carry, and the
    cluster as held back when it does not: a structural one never files and
    is never held (``disposition.files``, contract §6), a mechanical one files
    by ``filing_policy.is_fileable``."""
    carried = {sighting.get("session") for sighting in cluster.get("sightings") or [] if sighting.get("session")}
    seen = max(sighting_count, len(carried))
    mechanical = disposition.files(cluster["verdict"])
    files = mechanical and filing_policy.is_fileable(cluster, sighting_count=seen)
    if not mechanical or files:
        return files, None
    return False, filing_policy.WithheldCluster(title=cluster.get("title"), sighting_count=seen)


_RELEASABLE_ENTRY_STATUSES = frozenset({"open", "partially"})
"""Entry statuses a withheld entry may still be filed from: a fixed or
retired confusion needs no ticket."""


def entry_cluster(cb: dict, entry: dict) -> dict:
    """A fileable cluster rebuilt from codebook *entry*, by ``codebook.entry_field``;
    its severity and finding identity are the entry's own."""
    sightings = list(entry.get("sightings") or [])
    cluster = cluster_record(
        title=entry["title"],
        cause=codebook.entry_field(cb, entry, "cause"),
        area=codebook.entry_field(cb, entry, "area"),
        origin_phase=entry["origin_phase"],
        manifested_phase=entry["manifested_phase"],
        evidence=[s["evidence_quote"] for s in sightings if s.get("evidence_quote")],
        sightings=sightings,
    )
    return {
        **cluster,
        "severity": entry["severity"],
        "finding": census_findings.FindingIdentity.from_record(entry),
    }


@dataclass(frozen=True)
class PendingFiling:
    """One cluster to file this run. Work routed for it releases the withheld
    codebook entry *withheld_entry_id* names, if any; its back-link lands on
    *entry_id*. *verified_present*: this run's verifier found it present at
    as_of_sha; *supersedes_task*: the done task a re-filing supersedes."""

    cluster: dict
    withheld_entry_id: str | None = None
    entry_id: str | None = None
    verified_present: bool = False
    supersedes_task: int | None = None

    def filing(self, payload: Mapping[str, Any]) -> disposition.Filing:
        """This cluster as the dedup protocol files it, by *payload*."""
        finding = self.cluster.get("finding")
        filing = disposition.Filing(
            entry_id=self.entry_id,
            finding_key=None if finding is None else finding.key,
            title=_cluster_title(self.cluster),
            statement=self.cluster.get("summary") or "",
            payload=payload,
            verified_present=self.verified_present,
        )
        return filing if self.supersedes_task is None else filing.superseding(self.supersedes_task)


def filing_queue(fileable: list[tuple[dict, str | None]], cb: dict) -> list[PendingFiling]:
    """Every cluster to file this run, from the post-merge codebook *cb*.

    Each *fileable* cluster, given with the id of the entry it landed on --
    promoted to this run, the standing promotion its recurrence was routed
    to, or a key attach -- or None, releases that entry when it is releasable
    withheld, so a recurrence files one ticket for its entry, not two. Every
    other releasable withheld entry that ``filing_policy.is_fileable`` now
    accepts at its own distinct-session count follows. An entry withheld THIS
    run is never among them: the gate that withheld it used that same count.

    Each is paired with the entry its back-link lands on: the withheld entry
    it releases, else the entry it was promoted to. A *fileable* cluster was
    verified present at as_of_sha this run; a released withheld entry was
    verified by an earlier run, so it is not."""
    releasable = [
        entry for entry in cb.get("entries") or []
        if entry.get(codebook.ENTRY_FILING_KEY) == codebook.ENTRY_FILING_WITHHELD
        and entry.get("status") in _RELEASABLE_ENTRY_STATUSES
    ]
    releasable_ids = {entry["id"] for entry in releasable}

    queue = []
    for cluster, promoted_entry_id in fileable:
        withheld_entry_id = promoted_entry_id if promoted_entry_id in releasable_ids else None
        queue.append(PendingFiling(
            cluster, withheld_entry_id, withheld_entry_id or promoted_entry_id, verified_present=True,
        ))
    paired = {pending.withheld_entry_id for pending in queue}
    for entry in releasable:
        if entry["id"] in paired:
            continue
        cluster = entry_cluster(cb, entry)
        if filing_policy.is_fileable(cluster, sighting_count=distinct_sessions(entry)):
            queue.append(PendingFiling(cluster, entry["id"], entry["id"]))
    return queue


def _carrying(pending: PendingFiling, refiling: disposition.Refiling) -> PendingFiling:
    """*pending* carrying *refiling*: the done task it supersedes and, for a
    cluster this run's verifier did not see, the re-verify Verdict's
    remediation."""
    cluster = pending.cluster
    if refiling.found is not None and not pending.verified_present:
        cluster = census_verdicts.with_verdict(cluster, refiling.found)
    return replace(
        pending,
        cluster=cluster,
        verified_present=pending.verified_present or refiling.verified_present,
        supersedes_task=refiling.supersedes_task,
    )


def with_refilings(
    queue: list[PendingFiling], refiled: Sequence[disposition.Refiling], cb: dict,
) -> list[PendingFiling]:
    """*queue* with each *refiled* entry carried onto its queued filing, and
    every other one appended as its entry's cluster."""
    entries = {entry["id"]: entry for entry in cb.get("entries") or []}
    unqueued = {refiling.entry_id: refiling for refiling in refiled}
    merged = []
    for pending in queue:
        refiling = None if pending.entry_id is None else unqueued.pop(pending.entry_id, None)
        merged.append(pending if refiling is None else _carrying(pending, refiling))
    appended = [
        _carrying(PendingFiling(entry_cluster(cb, entries[entry_id]), entry_id=entry_id), refiling)
        for entry_id, refiling in unqueued.items()
    ]
    return [*merged, *appended]


def _free_payloads_path(path: Path, *, limit: int = 1000) -> Path:
    """Return *path* if it is free, else the first unused numbered sibling
    (``{stem}-2{suffix}``, ``{stem}-3{suffix}``, ...).

    A ``--dry-run-filing`` payload file is a human-review deliverable AND
    the only remaining handle on that run's remediation work -- the run
    advances the codebook and census-state, so nothing can regenerate it
    (see the dry-run WARNING in ``write_dry_run_payloads``). A second dry
    run on the same date must therefore never overwrite the first.

    The probe is bounded by *limit*; exhausting it raises ``RuntimeError``
    naming the directory. Raising here leaves nothing persisted, because
    ``census_run.persist`` writes the payloads before any other output.
    """
    if not path.exists():
        return path
    for n in range(2, limit + 1):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"census: could not find a free dry-run payload path near {path} -- "
        f"{limit} numbered siblings already exist in {path.parent}; clear out "
        "the reviewed ones before running another dry-run census"
    )


def write_dry_run_payloads(
    requested_path: Path,
    payloads: list[dict],
    *,
    withheld_offered: int,
    write_text: Callable[[Path, str], object],
) -> census_report.DryRunFiling:
    """Write *payloads* -- what a real run would hand ``submit_fn`` -- for a
    human to review, at *requested_path* or its free sibling, and say so
    loudly; *withheld_offered* of them release earlier-withheld entries."""
    resolved_path = _free_payloads_path(requested_path)
    if resolved_path != requested_path:
        logger.warning(
            "census: dry-run payload path %s already exists and was left "
            "UNTOUCHED (an earlier review artifact no re-run can "
            "regenerate) -- this run's payloads were written to %s instead",
            requested_path, resolved_path,
        )
    write_text(resolved_path, json.dumps(payloads, indent=2) + "\n")
    # The WARNING deliberately does NOT offer a repeat census as the
    # recovery path, because that path is dead: the same confusions code as
    # `matches` against the dumped codebook, and the sessions are ledgered as
    # coded, so no later census mines them again. Hand filing is the only
    # remaining handle on the work; do not reintroduce the "just run it again
    # without the flag" guidance.
    logger.warning(
        "census: --dry-run-filing -- %d task payload(s) written to %s; "
        "NOTHING was filed into a live task tree. This run HAS ALREADY "
        "advanced the codebook and census-state, so filing those payloads "
        "by hand is the only remaining way to land the work: a later "
        "census will NOT re-file them (these confusions now code as "
        "matches, not candidates, and their sessions are ledgered as coded).",
        len(payloads), resolved_path,
    )
    if withheld_offered:
        logger.warning(
            "census: --dry-run-filing -- %d of those payload(s) are withheld "
            "codebook entries that have since recurred; they stay marked "
            "withheld, so the next real census re-offers them.",
            withheld_offered,
        )
    # Every downstream consumer -- the report's Filed Tasks section, the
    # commit paths and the outcome -- reads the path off this one object,
    # so they all name the file actually written.
    return census_report.DryRunFiling(path=str(resolved_path), payload_count=len(payloads))
