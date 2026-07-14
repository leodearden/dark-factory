#!/usr/bin/env python3
"""scripts/legibility/census.py — periodic legibility census runner.

Task eta (η) of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.7 "Census runner", contract §7, decisions 4/5/6/9, boundary test §8.7).
The LEAF that consumes every prior legibility module: beta's (β)
inventory/sampling, alpha's (α) digest, delta's (δ) trickle coder, gamma's
(γ) codebook merger, zeta's (ζ) census trigger/state.

Flow (§5.7): usage-headroom preflight -> stratified-random saturation
mining (Sonnet miners) until >=dup_rate duplicates for consecutive_batches
-> Sonnet verification of novel clusters vs current main -> Fable
synthesis into a dated ``plans/confusion-census-<date>.md`` report with
the origin x manifestation matrix -> curator-path ``submit_task``
remediation filing -> codebook update (promote/reject candidates, retire
fixed, never delete) -> advance ``docs/legibility/census-state.json``.
``--force`` for operator-initiated runs.

Every LLM / MCP / git side effect in this module is an INJECTED seam
(``invoke``, ``verify_fn``, ``synthesize_fn``, ``submit_fn``,
``escalate_fn``, ``status_fetcher``, ``commit``, ``batch_source``) --
mirrors delta's ``coder.code_digests(invoke=)`` and zeta's
``census_trigger(status_fetcher=)``. The scripts/ test env (``uv run
--project shared``) has no MCP client / httpx / live models, so every
seam is ALWAYS faked in this module's own test suite; the deterministic
core (duplicate/dup_rate, the mining batch loop + saturation stop, the
origin x manifestation matrix, census-state advance, codebook lifecycle
transforms, report rendering) is unit-tested with no network.

Model routing (ratified static policy, PRD §5/§12 -- deliberately NOT the
adaptive ``resolve_route`` ladder): Sonnet for mining + verification,
Fable ONLY for synthesis, read from ``config.Models``
(``census_miner``/``census_verify``/``census_synthesis``).

Codebook handling (PRD decision 1 -- sole writer, never delete): new
mining sightings/candidates merge through the EXISTING
``codebook.apply_coding_record``. Census-only lifecycle transitions
(promote/reject/retire) are deterministic in-memory transforms in this
module, persisted via ``codebook.dump()`` after ``codebook.validate()``
and ``codebook.assert_no_deletion()`` confirm the write is safe.
``codebook.py`` itself is NOT modified.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Self-bootstrap for standalone `python scripts/legibility/census.py` runs
# -- must run BEFORE the `legibility.*` imports below, since a direct
# script invocation puts only scripts/legibility/ (not scripts/) on
# sys.path. Skipped under pytest/normal package import: __name__ is
# 'legibility.census' or 'census', never '__main__' (mirrors
# sampling.py:37-38).
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codebook  # noqa: E402
import coder  # noqa: E402
import config  # noqa: E402
import digest  # noqa: E402
import inventory  # noqa: E402
from legibility import census_trigger  # noqa: E402


# ---------------------------------------------------------------------------
# is_duplicate / batch_dup_rate — the saturation-mining novelty signal
# ---------------------------------------------------------------------------

def is_duplicate(record: dict) -> bool:
    """A §7.3 coding record is a *duplicate* (a re-observation of an
    already-known cause) iff it carries zero candidates -- a candidate is
    the sole novelty signal in a coding record; matches-only or entirely
    empty (``{"matches": [], "candidates": []}``) both mean "nothing new
    here"."""
    return len(record.get("candidates") or []) == 0


def batch_dup_rate(records: list[dict]) -> float:
    """Fraction of *records* that are duplicates (``is_duplicate`` True).
    Returns ``0.0`` for an empty batch rather than raising
    ZeroDivisionError -- an empty/all-failed batch has no signal either
    way, and the saturation loop must be able to treat it as "not
    saturated" without crashing."""
    if not records:
        return 0.0
    return sum(1 for record in records if is_duplicate(record)) / len(records)


# ---------------------------------------------------------------------------
# mine_to_saturation — stratified-random batch loop + saturation stop
# ---------------------------------------------------------------------------

@dataclass
class BatchStats:
    """Per-batch mining tally: how one ``coder.code_digests`` batch scored
    against the saturation threshold."""

    index: int
    total: int
    succeeded: int
    failed: int
    dup_rate: float
    saturated: bool


@dataclass
class MiningResult:
    """Outcome of ``mine_to_saturation``: every successfully-coded record
    across every consumed batch, per-batch stats, and why mining stopped
    (``"saturated"`` -- ``config.consecutive_batches`` consecutive batches
    at/above ``config.dup_rate``; ``"exhausted"`` -- ``batch_source`` ran
    out first)."""

    records: list[dict] = field(default_factory=list)
    batch_stats: list[BatchStats] = field(default_factory=list)
    stop_reason: str = "exhausted"


def mine_to_saturation(
    batch_source, codebook_dict: dict, *, project: str, model: str, config, invoke,
) -> MiningResult:
    """Code batches from *batch_source* against *codebook_dict* via
    ``coder.code_digests`` until novelty saturates.

    Iterates *batch_source* lazily -- a batch is pulled only when the loop
    is ready to code it, so a source that stops yielding once mining
    saturates never has its later batches materialized. Per batch: code
    via ``coder.code_digests(batch, codebook_dict, project=project,
    model=model, invoke=invoke)``, compute this batch's ``dup_rate`` over
    its own successful records (``batch_dup_rate`` -- failed codings are
    excluded from the denominator per that function's own contract), and
    track a consecutive-saturated-batch counter: incremented when
    ``dup_rate >= config.dup_rate``, reset to 0 otherwise. Mining stops
    (``stop_reason="saturated"``) the moment the counter reaches
    ``config.consecutive_batches`` -- right after that Nth consecutive
    saturated batch, so no further batch is pulled. If *batch_source*
    exhausts before that, ``stop_reason="exhausted"``.

    *config* is a ``config.Saturation``-shaped object (``.dup_rate``,
    ``.consecutive_batches`` -- i.e. a project's
    ``LegibilityConfig.census.saturation``), not the whole
    ``LegibilityConfig``. *model* is the caller's already-resolved model
    id (Sonnet miner routing per the ratified static policy -- this
    function does not read ``config.Models`` itself).
    """
    result = MiningResult()
    consecutive_saturated = 0

    for index, batch in enumerate(batch_source):
        run_result = coder.code_digests(
            list(batch), codebook_dict, project=project, model=model, invoke=invoke,
        )
        result.records.extend(run_result.records)

        dup_rate = batch_dup_rate(run_result.records)
        saturated = dup_rate >= config.dup_rate
        result.batch_stats.append(
            BatchStats(
                index=index,
                total=run_result.total,
                succeeded=run_result.succeeded,
                failed=run_result.failed,
                dup_rate=dup_rate,
                saturated=saturated,
            )
        )

        consecutive_saturated = consecutive_saturated + 1 if saturated else 0
        if consecutive_saturated >= config.consecutive_batches:
            result.stop_reason = "saturated"
            return result

    result.stop_reason = "exhausted"
    return result


# ---------------------------------------------------------------------------
# compute_matrix / render_matrix — origin x manifestation matrix (PRD
# decision 6: the `unknown` phase is explicit, never inferred)
# ---------------------------------------------------------------------------

def compute_matrix(sightings: list[dict]) -> dict[str, dict[str, int]]:
    """Tally *sightings* into an origin x manifestation count structure.

    A sighting with a missing, ``None``, or empty ``origin_phase`` /
    ``manifested_phase`` is bucketed under the explicit ``"unknown"``
    phase -- NEVER inferred to a concrete one (PRD decision 6). Returns a
    nested dict ``{origin: {manifested: count}}`` covering exactly the
    phases actually observed (rows/cols this sparse by construction --
    ``"unknown"`` only appears when at least one sighting actually landed
    there), row/col-ordered per ``codebook.PHASES`` for a stable,
    deterministic iteration order. Every observed row carries every
    observed column key (0 for a combination that was never seen), so a
    caller never needs a ``.get(..., 0)`` fallback. An empty *sightings*
    list returns ``{}``.
    """
    counts: dict[tuple[str, str], int] = {}
    origins: set[str] = set()
    manifesteds: set[str] = set()

    for sighting in sightings:
        origin = sighting.get("origin_phase") or "unknown"
        manifested = sighting.get("manifested_phase") or "unknown"
        origins.add(origin)
        manifesteds.add(manifested)
        key = (origin, manifested)
        counts[key] = counts.get(key, 0) + 1

    ordered_origins = [p for p in codebook.PHASES if p in origins]
    ordered_manifesteds = [p for p in codebook.PHASES if p in manifesteds]

    return {
        origin: {
            manifested: counts.get((origin, manifested), 0)
            for manifested in ordered_manifesteds
        }
        for origin in ordered_origins
    }


def render_matrix(matrix: dict[str, dict[str, int]]) -> str:
    """Render *matrix* (``compute_matrix``'s output shape) as a
    deterministic markdown table: one row per origin phase, one column
    per manifestation phase, PHASES-ordered, 0 for any cell with no
    count. Returns a fixed placeholder (never an empty string or a
    header-only table) when *matrix* is empty."""
    if not matrix:
        return "_No sightings recorded._\n"

    manifesteds_seen: set[str] = set()
    for row in matrix.values():
        manifesteds_seen.update(row.keys())
    manifesteds = [p for p in codebook.PHASES if p in manifesteds_seen]
    origins = [p for p in codebook.PHASES if p in matrix]

    header = "| origin \\ manifested | " + " | ".join(manifesteds) + " |"
    separator = "| --- | " + " | ".join("---" for _ in manifesteds) + " |"
    rows = [
        "| " + origin + " | " + " | ".join(str(matrix[origin].get(m, 0)) for m in manifesteds) + " |"
        for origin in origins
    ]
    return "\n".join([header, separator, *rows]) + "\n"


# ---------------------------------------------------------------------------
# preflight_headroom — cheap usage-headroom probe (PRD decision 5: no
# usage API assumed -- one tiny call, scan its reply for a banner)
# ---------------------------------------------------------------------------

_HEADROOM_BANNER_MARKERS = (
    "usage limit",
    "rate limit",
    "please run /login",
    "invalid api key",
)
"""Case-insensitive substrings that mark a usage-limit/auth banner reply
from the headless `claude` CLI, rather than a genuine model response."""

_HEADROOM_PROBE_PROMPT = "ping"
"""The tiny probe prompt -- a cheap single round trip, not a real mining
call. Its content doesn't matter; only whether the reply carries a banner
marker or raises."""


@dataclass
class HeadroomResult:
    """Verdict from ``preflight_headroom``: ``ok=True`` means the probe
    round-tripped cleanly and mining may proceed; ``ok=False`` means it
    should be deferred to the next trigger evaluation, with ``reason``
    explaining why (a banner marker, or an invocation error)."""

    ok: bool
    reason: str | None = None


def preflight_headroom(invoke, *, model: str) -> HeadroomResult:
    """Issue one tiny probe via ``invoke(prompt, model)`` and decide
    whether the census has headroom to proceed.

    No usage API is assumed to exist (PRD decision 5) -- this is a cheap
    preflight probe, not a quota lookup. The reply is scanned
    case-insensitively for a known usage-limit/auth banner marker
    (``_HEADROOM_BANNER_MARKERS``); a match defers. An invocation error
    raised by *invoke* (e.g. a ``CoderInvocationError``-shaped failure)
    is also treated as a deferral -- fail-safe, never a crash, since a
    probe failure is exactly the kind of "the model isn't reachable right
    now" signal this preflight exists to catch. Makes no mining decisions
    itself -- just the ok/deferred verdict + reason.
    """
    try:
        reply = invoke(_HEADROOM_PROBE_PROMPT, model)
    except Exception as exc:  # noqa: BLE001 - any probe failure must fail safe
        return HeadroomResult(ok=False, reason=f"headroom probe invocation failed: {exc}")

    lowered = (reply or "").lower()
    for marker in _HEADROOM_BANNER_MARKERS:
        if marker in lowered:
            return HeadroomResult(
                ok=False, reason=f"headroom probe reply carries a banner marker: {marker!r}",
            )

    return HeadroomResult(ok=True, reason=None)


# ---------------------------------------------------------------------------
# build_task_payloads — verified clusters -> curator-path submit_task
# kwargs (PRD decision 9: normal task_kind, never planning_mode; PRD
# decision 4: a harness-rooted cluster may target dark_factory)
# ---------------------------------------------------------------------------

def _cluster_description(cluster: dict, *, project_id: str) -> str:
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
        lines.append(f"Observed in {len(sightings)} sighting(s) (project: {project_id}).")

    return "\n".join(lines)


def build_task_payloads(clusters, *, project_root: str, project_id: str) -> list[dict]:
    """Map each verified cluster to one curator-path ``submit_task`` kwarg
    dict. ``task_kind`` is always ``"normal"``; ``planning_mode`` is
    deliberately OMITTED (defaults False at the submit_task layer) --
    curator dedup against already-filed remediation is the point (PRD
    decision 9), and ``planning_mode`` is exactly the curator-bypassing
    path that would defeat it.

    A cluster observed in a hosted project whose root cause is
    harness-rooted may carry ``target_project_root``/``target_project_id``
    overrides to file into dark_factory instead of the census's own
    project (PRD decision 4, same fused-memory, different project_root);
    absent those, the payload targets the census's own *project_root*/
    *project_id*.

    Pure function -- returns payloads only; the actual ``submit_fn`` call
    happens in ``run_census``.
    """
    payloads = []
    for cluster in clusters:
        title = cluster.get("title") or "Untitled confusion cluster"
        target_project_root = cluster.get("target_project_root", project_root)
        target_project_id = cluster.get("target_project_id", project_id)
        payloads.append(
            {
                "project_root": target_project_root,
                "title": f"[legibility census] {title}",
                "description": _cluster_description(cluster, project_id=target_project_id),
                "task_kind": "normal",
                "priority": cluster.get("priority", "medium"),
                "metadata": {
                    "source": "legibility_census",
                    "origin_project_id": project_id,
                },
            }
        )
    return payloads


# ---------------------------------------------------------------------------
# promote_candidate / reject_candidate / retire_entry — census-only
# codebook lifecycle transforms (PRD decision 1: deterministic, in-memory,
# reuse gamma's validator + never-delete assertion + sole-writer dump;
# codebook.py itself is NOT modified)
# ---------------------------------------------------------------------------

def _find_by_id(items: list[dict], item_id: str, *, kind: str) -> dict:
    for item in items:
        if item.get("id") == item_id:
            return item
    raise ValueError(f"{kind}: no id {item_id!r} found")


def promote_candidate(cb: dict, cand_id: str, entry_fields: dict) -> dict:
    """Promote candidate *cand_id* in *cb* to a new codebook entry.

    Returns a NEW codebook (a ``copy.deepcopy`` of *cb*, never mutated in
    place). Appends an entry built from *entry_fields* (expected to carry
    ``id``/``title``/``severity``/``status``/``origin_phase``/
    ``manifested_phase`` -- everything ``codebook.py``'s v2 entry schema
    requires except ``sightings``) with ``sightings`` set to the
    candidate's OWN sightings (the candidate is the source of truth for
    what was actually observed, not *entry_fields*). The candidate itself
    is stamped ``disposition="promoted"`` and ``promoted_to`` naming the
    new entry id -- RETAINED, never removed (PRD decision 3).
    ``codebook.assert_no_deletion`` is run against the pre-transform *cb*
    before returning, as a construction-independent safety net.
    """
    result = copy.deepcopy(cb)
    candidate = _find_by_id(result.get("candidates") or [], cand_id, kind="promote_candidate")

    new_entry_id = entry_fields["id"]
    entry = dict(entry_fields)
    entry["sightings"] = copy.deepcopy(candidate.get("sightings") or [])
    result.setdefault("entries", []).append(entry)

    candidate["disposition"] = "promoted"
    candidate["promoted_to"] = new_entry_id

    codebook.assert_no_deletion(cb, result)
    return result


def reject_candidate(cb: dict, cand_id: str) -> dict:
    """Reject candidate *cand_id* in *cb*: stamps ``disposition="rejected"``
    in place on a deep copy, RETAINED (never removed). Returns a NEW
    codebook; *cb* is never mutated."""
    result = copy.deepcopy(cb)
    candidate = _find_by_id(result.get("candidates") or [], cand_id, kind="reject_candidate")
    candidate["disposition"] = "rejected"

    codebook.assert_no_deletion(cb, result)
    return result


def retire_entry(cb: dict, entry_id: str) -> dict:
    """Retire entry *entry_id* in *cb*: stamps ``status="retired"`` in
    place on a deep copy, RETAINED (never removed). Returns a NEW
    codebook; *cb* is never mutated."""
    result = copy.deepcopy(cb)
    entry = _find_by_id(result.get("entries") or [], entry_id, kind="retire_entry")
    entry["status"] = "retired"

    codebook.assert_no_deletion(cb, result)
    return result


# ---------------------------------------------------------------------------
# advance_census_state — §7.5 census-state.json sole writer (zeta/2579
# MUST-persist contract: last_census_done_count is ALWAYS present)
# ---------------------------------------------------------------------------

def advance_census_state(
    path, *, now_iso: str, report_path: str, done_count: int,
) -> None:
    """Write the §7.5 census-state dict to *path*, atomically.

    Always writes exactly ``{"last_census_at": now_iso,
    "last_census_report": report_path, "last_census_done_count":
    done_count}`` -- ``last_census_done_count`` is NEVER conditionally
    omitted, even when ``done_count == 0`` (falsy but a real baseline).
    zeta's ``census_trigger.compute_tasks_landed()`` returns ``None`` --
    and its condition (b) permanently fails safe -- whenever that key is
    absent (census_trigger.py:410-416), and this module is
    census-state.json's SOLE writer, so this is the one place that
    baseline can ever be supplied.

    Uses the same ``tempfile.mkstemp`` + ``os.replace`` atomic-write
    pattern as ``codebook.dump`` (temp file in the same directory as
    *path*, then an atomic rename): a crash or kill mid-write can never
    leave a partial state file for the next ``load_census_state`` to trip
    over, and a pre-existing state file is fully replaced, never merged.
    """
    state = {
        "last_census_at": now_iso,
        "last_census_report": report_path,
        "last_census_done_count": done_count,
    }
    directory = os.path.dirname(os.fspath(path)) or "."
    fd, tmp_file = tempfile.mkstemp(prefix=".census-state-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp_file, path)
    except BaseException:
        try:
            os.remove(tmp_file)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# render_report — dated plans/confusion-census-<date>.md markdown assembly
# ---------------------------------------------------------------------------

def render_report(
    *,
    date: str,
    project_id: str,
    force: bool,
    matrix_md: str,
    mining_result: MiningResult,
    synthesis_md: str,
    filed_task_ids: list[str],
    cost_note: str,
) -> str:
    """Assemble the dated census report as markdown, purely from the
    pieces passed in -- no clock, no model call, no I/O. *date* and every
    piece of LLM-produced prose (*synthesis_md*, *matrix_md*) are inputs,
    so the same inputs always render byte-identical output.
    """
    lines = [f"# confusion census {date}", "", f"Project: {project_id}"]

    if force:
        lines.append("")
        lines.append("_--force: operator-initiated run._")

    lines.append("")
    lines.append("## Saturation")
    lines.append("")
    lines.append(f"- batches: {len(mining_result.batch_stats)}")
    lines.append(f"- stop reason: {mining_result.stop_reason}")
    for stats in mining_result.batch_stats:
        lines.append(
            f"  - batch {stats.index}: dup_rate={stats.dup_rate:.2f} "
            f"(total={stats.total}, succeeded={stats.succeeded}, failed={stats.failed}, "
            f"saturated={stats.saturated})"
        )

    lines.append("")
    lines.append("## Origin x Manifestation Matrix")
    lines.append("")
    lines.append(matrix_md)

    lines.append("## Synthesis")
    lines.append("")
    lines.append(synthesis_md)

    lines.append("")
    lines.append("## Filed Tasks")
    lines.append("")
    if filed_task_ids:
        lines.extend(f"- {task_id}" for task_id in filed_task_ids)
    else:
        lines.append("_none filed._")

    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(cost_note)
    lines.append("")

    return "\n".join(lines)
