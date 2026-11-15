"""A census run's JSON record, and the markdown report rendered from it alone."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import yaml
from legibility import (
    census_findings,
    census_mining,
    census_window,
    codebook,
    disposition,
    filing_policy,
    prescreen,
    session_ledger,
    similarity,
    synthesis,
)


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


@dataclass
class VerifyCoverage:
    """Coverage record for the operator verify cap (``--max-verify-clusters``).

    ``novel`` is how many novel clusters this run's mining actually
    produced; ``offered`` is how many of them were handed to ``verify_fn``
    (one Sonnet call each -- the cost being bounded); ``cap`` is the
    operator cap that produced the split. The ``novel - offered``
    remainder is DEFERRED, not dropped: those clusters' candidates still
    merge into the codebook as ``pending``.

    That pickup is CONDITIONAL, not automatic, and the report says so: the
    sessions behind a deferred cluster were coded, so they are ledgered and
    never re-mined. Pending candidates are absent from
    ``coder.build_codebook_index`` (entries only), so a RECURRENCE re-emerges
    as a novel cluster and is promoted against the pending id; a one-off
    deferred by the cap sits pending until a human adjudicates it.

    ``offered``, deliberately, and NOT ``verified``: this is a COST
    record, counting verify_fn calls made, and a cluster handed to the
    verifier may well come back FALSE. The rendered line says "handed" for
    the same reason. Naming it ``verified`` is what let the capped report
    print "verified all N novel cluster(s)" directly beneath the
    ``MassRejection`` notice saying not one of them survived -- a report
    contradicting itself on the one run whose report must not lie.

    ``None`` in place of this record means no verify cap was used, so no
    coverage line is rendered. The ``## Verification`` section itself may
    still appear on the ``MassRejection`` path below -- the two signals are
    independent and can render together."""

    novel: int
    offered: int
    cap: int | None = None


@dataclass
class MassRejection:
    """Anomaly record: clusters were offered for verification and NOT ONE
    survived.

    ``offered`` is how many were handed to ``verify_fn`` -- the same
    quantity ``VerifyCoverage.offered`` counts, and named identically on
    purpose. This is the observable signature of the 2026-08-03 sandbox
    incident, where the verify subprocess was rooted outside the censused
    tree and every read was permission-denied -- a run with real findings
    reported as an empty census. It is deliberately NOT folded into
    ``VerifyCoverage``: that record is a COST record and renders on every
    capped run, while this one is an ANOMALY record and renders only when
    the verifier returned nothing. Two different questions, so two
    records; they can and do render together.

    ``None`` in place of this record means the run did not mass-reject.
    An all-rejected run is legitimately possible, so this is a SUSPICION
    to be checked, never a failure."""

    offered: int


@dataclass
class DryRunFiling:
    """Outcome record for ``--dry-run-filing``: every would-be
    ``submit_task`` payload this run built was written to ``path`` as JSON
    for human review, and NOTHING was filed into a live task tree.

    ``payload_count`` is how many payloads the file holds. Reused by both
    ``render_report`` (which must not let a run that filed nothing read as a
    normal run that had nothing to file) and ``census_run.Done``/``census_run.Unlanded``
    (so ``census.main``'s summary line can name the review file instead of printing a
    misleading ``filed_tickets=0``). ``None`` in place of this record means
    the run filed normally."""

    path: str
    payload_count: int


SECTION_HEADER = "header"
SECTION_FORCE_MARKER = "force-marker"
SECTION_METHOD = "method"
SECTION_SATURATION = "saturation"
SECTION_SCREENED = "screened"
SECTION_ADJUDICATION = "adjudication"
SECTION_VERIFICATION = "verification"
SECTION_UNRESOLVED_VERDICTS = "unresolved-verdicts"
SECTION_MATRIX = "matrix"
SECTION_SYNTHESIS = "synthesis"
SECTION_FINDINGS = "findings"
SECTION_FILED_TASKS = "filed-tasks"
SECTION_STRUCTURAL = "structural"
SECTION_DISPOSITIONS = "dispositions"
SECTION_WITHHELD = "withheld-clusters"
SECTION_COST = "cost"
"""Stable machine keys for the blocks :func:`census_report_sections` emits.

Never rendered -- they exist so a caller can ask WHICH blocks a report
carries and in what order without matching on the English inside them."""


@dataclass(frozen=True)
class ReportSection:
    """One block of the census report, under a STABLE machine key.

    The key is never rendered. It exists so a caller can ask WHICH blocks a
    report carries and in what order without matching on the English inside
    them. Prose is the part of this function expected to be reworded, and a
    check that keys on prose constrains wording rather than behaviour; keying
    on the structure keeps the NO-SILENT-CAPS disclosure guarantees
    falsifiable instead -- a section that stops being emitted, or is emitted
    on the wrong run, or lands below the thing it qualifies, fails, and a copy
    edit does not.

    ``lines`` is the section's own slice of the report, including its own
    leading blank line, so :func:`join_report_sections` is a plain
    concatenation. A single element may itself be a multi-line blob (an
    embedded ``matrix_md``).

    Convention, not invention: esc-3208-4, memories
    53e61951-0704-4436-94bd-bf12ae66c23b and
    538183c6-6a83-44b6-8a2f-290b75a545d6, shipped in
    ``fused-memory/scripts/memory_eval_retrieval_probe.py`` and
    ``memory_eval_staleness_sweep.py``, which each define the dataclass
    locally as this does.
    """

    key: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


CENSUS_RECORD_VERSION = 2
"""The shape of :func:`build_census_record`'s value. 2 replaced version 1's
``synthesis_md`` prose with the ``synthesis`` mapping
(``synthesis.SynthesisReport.to_record``); :func:`census_report_sections`
renders this version only."""


def _selection_evidence(
    selection: census_window.SessionSelection | None, *, mined: int,
) -> dict[str, Any]:
    if selection is None:
        return {
            "window": None,
            "sessions_enumerated": None,
            "skipped_coded": None,
            "skipped_zero_signal": None,
            "mined": mined,
            "ledger_rows": None,
        }
    return {
        "window": selection.window.to_record(),
        "sessions_enumerated": selection.sessions_enumerated,
        "skipped_coded": selection.skipped_coded,
        "skipped_zero_signal": selection.skipped_zero_signal,
        "mined": mined,
        "ledger_rows": selection.ledger.total_rows,
    }


def _ledger_extra(selection: census_window.SessionSelection | None) -> dict[str, Any]:
    if selection is None:
        return {
            "ledger_created_this_run": None,
            "ledger_state": None,
            "ledger_pruned": None,
            "ledger_error": None,
        }
    ledger = selection.ledger
    return {
        "ledger_created_this_run": ledger.state == session_ledger.LedgerState.CREATED,
        "ledger_state": ledger.state.value,
        "ledger_pruned": ledger.pruned,
        "ledger_error": ledger.error,
    }


def build_method(
    *,
    run_id: str,
    as_of_sha: str,
    since: str | None,
    selection: census_window.SessionSelection | None,
    mined: int,
    confirmed: int,
    refuted: int,
    unverified: int,
    miner_calls: int,
    verify_calls: int,
    synthesis_calls: int,
    probe_calls: int,
    wall_clock_secs: float,
    verify_normalisations: Mapping[str, int],
    reverify: Mapping[str, int],
    screened: Mapping[str, int],
    inputs_consumed: Sequence[str],
    inputs_consumed_note: str | None,
    embedding_calls: int,
    adjudication: Mapping[str, int],
    slug_tally: codebook.SlugTally,
) -> dict[str, Any]:
    """The JSON-safe ``## Method`` block: the seven keys of the
    quality-findings contract §5, then ``extra``
    (plans/census-incremental-prd.md §4.3). *verify_calls* includes the
    offered re-verifies counted in *reverify*, and *inputs_consumed* are the
    run ids of the other instruments' records the pre-screen read.
    *selection* is ``None`` for a run with no mining window, which nulls the
    evidence and ledger fields it would have supplied; an unreadable ledger
    counts ``ledger_rows`` and ``ledger_pruned`` as ``None``, never 0."""
    return {
        "run_id": run_id,
        "as_of_sha": as_of_sha,
        "since": since or "none",
        "evidence": _selection_evidence(selection, mined=mined),
        "verification": {
            "confirmed": confirmed, "weakened": 0, "refuted": refuted, "unverified": unverified,
        },
        "cost": {
            "miner_calls": miner_calls,
            "verify_calls": verify_calls,
            "synthesis_calls": synthesis_calls,
            "probe_calls": probe_calls,
            "embedding_calls": embedding_calls,
            "wall_clock_secs": wall_clock_secs,
        },
        "inputs_consumed": list(inputs_consumed),
        "extra": {
            "inputs_consumed_note": inputs_consumed_note,
            "screened": dict(screened),
            "attached": adjudication["attached"],
            "adjudicated": adjudication["adjudicated"],
            **_ledger_extra(selection),
            "verify_normalisations": dict(verify_normalisations),
            "reverify": dict(reverify),
            "slug_rejections": slug_tally.rejections_record(),
        },
    }


def _optional_asdict(value) -> dict | None:
    return asdict(value) if value is not None else None


def build_census_record(
    *,
    method: dict[str, Any],
    date: str,
    project_id: str,
    force: bool,
    matrix_md: str,
    mining_result: census_mining.MiningResult,
    synthesis: Mapping[str, Any],
    project_root: str,
    filing: Mapping[str, Any],
    dispositions: Mapping[str, Any],
    screened: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    cost_note: str,
    verify_coverage: VerifyCoverage | None = None,
    dry_run: DryRunFiling | None = None,
    dropped_verdicts: tuple[DroppedVerdict, ...] = (),
    mass_rejection: MassRejection | None = None,
    withheld: tuple[filing_policy.WithheldCluster, ...] = (),
    findings: Sequence[Mapping[str, Any]] = (),
    area_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The run's JSON record: everything :func:`census_report_sections`
    renders, as plain JSON-safe values. The mining records themselves are
    not carried; the codebook holds what they merged. *findings* are the
    contract §1 rows (``census_findings.FindingRow.to_record``) and
    *area_map* the ``finding_area.AreaMap.to_record`` that keyed them.
    *filing* is ``disposition.FilingReport.to_record``, *dispositions*
    ``disposition.OutcomeReport.to_record``, *screened*
    ``prescreen.ScreenReport.to_record``, *adjudication*
    ``similarity.AdjudicationReport.to_record``, *synthesis*
    ``synthesis.SynthesisReport.to_record``, and *project_root* the observed
    project their decisions and outcomes are reported against."""
    return {
        "record_version": CENSUS_RECORD_VERSION,
        "run_id": method["run_id"],
        "date": date,
        "project_id": project_id,
        "project_root": project_root,
        "force": force,
        "method": method,
        "mining": {
            "batch_stats": [asdict(stats) for stats in mining_result.batch_stats],
            "stop_reason": mining_result.stop_reason,
            "max_batches": mining_result.max_batches,
        },
        "verify_coverage": _optional_asdict(verify_coverage),
        "mass_rejection": _optional_asdict(mass_rejection),
        "dropped_verdicts": [asdict(record) for record in dropped_verdicts],
        "matrix_md": matrix_md,
        "synthesis": dict(synthesis),
        "filing": dict(filing),
        "dispositions": dict(dispositions),
        "screened": dict(screened),
        "adjudication": dict(adjudication),
        "dry_run": _optional_asdict(dry_run),
        "withheld": [asdict(cluster) for cluster in withheld],
        "findings": [dict(row) for row in findings],
        "area_map": None if area_map is None else dict(area_map),
        "cost_note": cost_note,
    }


def _from_record(cls, value: Mapping[str, Any] | None):
    return cls(**value) if value is not None else None


def census_report_sections(record: Mapping[str, Any]) -> tuple[ReportSection, ...]:
    """The census report, decomposed, rendered from *record*
    (:func:`build_census_record`) alone: the record is the source and the
    markdown is never edited by hand (quality-findings contract §5).
    :func:`render_report` joins what this returns, so the two cannot drift.

    NO SILENT CAPS: when the operator bounded this run, the report says so in
    as many words. Every cost-control rendering is gated on its flag being
    set, so a FLAGLESS run renders exactly the golden lock
    (``test_render_report_flagless_output_is_byte_identical_golden``).
    ``## Verification`` also renders, uncapped, when every offered cluster
    was rejected (*mass_rejection*): that anomaly must not commit a report
    indistinguishable from a clean census.
    """
    if record.get("record_version") != CENSUS_RECORD_VERSION:
        raise ValueError(
            f"census record version {record.get('record_version')!r} is not "
            f"{CENSUS_RECORD_VERSION}, the only shape census_report_sections renders"
        )
    mining = record["mining"]
    batch_stats = [census_mining.BatchStats.from_record(stats) for stats in mining["batch_stats"]]
    stop_reason = mining["stop_reason"]
    max_batches = mining["max_batches"]
    verify_coverage = _from_record(VerifyCoverage, record["verify_coverage"])
    mass_rejection = _from_record(MassRejection, record["mass_rejection"])
    dry_run = _from_record(DryRunFiling, record["dry_run"])
    dropped_verdicts = [DroppedVerdict(**item) for item in record["dropped_verdicts"]]
    withheld = [filing_policy.WithheldCluster(**item) for item in record["withheld"]]

    sections: list[ReportSection] = []

    def emit(key: str, lines: list[str]) -> None:
        sections.append(ReportSection(key=key, lines=tuple(lines)))

    emit(
        SECTION_HEADER,
        [f"# confusion census {record['date']}", "", f"Project: {record['project_id']}"],
    )

    if record["force"]:
        emit(SECTION_FORCE_MARKER, ["", "_--force: operator-initiated run._"])

    method_yaml = yaml.safe_dump(record["method"], sort_keys=False, default_flow_style=False)
    emit(SECTION_METHOD, ["", "## Method", "", "```yaml", *method_yaml.splitlines(), "```"])

    saturation = [
        "",
        "## Saturation",
        "",
        f"- batches: {len(batch_stats)}",
        f"- stop reason: {stop_reason}",
    ]
    if max_batches is not None:
        # Deliberately states only counts this function was actually handed:
        # the total number of ENUMERATED sessions is not knowable here
        # (batch_source is a generic injected iterable), and claiming
        # "X of Y (enumerated)" would assert a number the code never
        # measured.
        if stop_reason == "capped":
            # Reports `succeeded`, not `total`, as the coverage count:
            # `total` counts digests DRAWN into a batch, including any that
            # failed to code and so contributed no signal to the codebook (a
            # storm batch, PRD §8.6, is the extreme case) -- `succeeded` is
            # what this run actually coded. `drawn` is still reported
            # alongside it, labelled, so nothing previously visible is lost.
            # Computed here, not at the outer `max_batches is not None`
            # scope, because the "not reached" branch below never uses them.
            coded = sum(stats.succeeded for stats in batch_stats)
            drawn = sum(stats.total for stats in batch_stats)
            coverage_line = (
                f"- coverage: coded {coded} of {drawn} session digest(s) drawn across "
                f"{len(batch_stats)} batch(es); operator batch cap = "
                f"{max_batches} batch(es) -- mining was BOUNDED BY THE CAP, "
                "not run to saturation: sessions beyond the cap were NOT mined, so this "
                "census is PARTIAL coverage, not a full sweep."
            )
            # Gated on the count discrepancy itself, not on any batch's
            # "failure" status: sub-storm coding failures (below the >50%
            # threshold, PRD §8.6) still make coded < drawn and still
            # overstate coverage if left unsaid, and this line's job is the
            # count, not the storm classification (run_census reports storm
            # indices separately). Kept inside the same coverage line so the
            # shortfall can never be read apart from the claim it qualifies;
            # a healthy capped run (coded == drawn) renders no clause at all
            # -- a line trained to be ignored is a line that will be ignored
            # on the run that matters.
            if drawn > coded:
                coverage_line += (
                    f" {drawn - coded} digest(s) FAILED TO CODE and contributed no "
                    "signal (see the per-batch tallies below)."
                )
            saturation.append(coverage_line)
            # Only coded sessions are ledgered, so the capped-away ones are
            # still eligible next run.
            saturation.append(
                "- RESUMED NEXT RUN: the capped-away sessions were not mined, so they "
                "were not ledgered as coded; the next census mines them while they "
                "remain inside its retention window. No rollback is needed."
            )
        else:
            saturation.append(
                f"- operator batch cap: {max_batches} batch(es) "
                f"(not reached -- mining stopped by: {stop_reason})"
            )
    for stats in batch_stats:
        saturation.append(
            f"  - batch {stats.index}: dup_rate={stats.dup_rate:.2f} "
            f"(total={stats.total}, succeeded={stats.succeeded}, failed={stats.failed}, "
            f"saturated={stats.saturated})"
        )
    emit(SECTION_SATURATION, saturation)
    emit(SECTION_SCREENED, prescreen.report_lines(record["screened"]))
    emit(SECTION_ADJUDICATION, similarity.report_lines(record["adjudication"]))

    if verify_coverage is not None or mass_rejection is not None:
        verification = ["", "## Verification", ""]
    else:
        verification = []

    if mass_rejection is not None:
        # FIRST inside the section: a cap line is routine, this is not. On a
        # flagless run the mere PRESENCE of a ## Verification section is
        # itself the anomaly -- a stronger signal than one more always-present
        # line an operator learns to skim past. Same voice as the `suspect`
        # string in _detect_mass_rejection, deliberately.
        verification.append(
            f"- **ALL {mass_rejection.offered} verified-candidate cluster(s) were "
            "REJECTED and none survived.** Suspect a SYSTEMIC verifier failure "
            "(model unreachable, tool access denied, or unparseable verdicts) "
            "rather than genuinely unfounded claims: this is the observable "
            "signature of the 2026-08-03 sandbox incident, in which the verify "
            "subprocess was rooted outside the censused tree and every read was "
            "permission-denied. Check the run's per-cluster 'verify failed' "
            "warnings before reading this census as unremarkable -- a run with "
            "real findings is being reported as an empty one if this is systemic."
        )

    if verify_coverage is not None:
        deferred = verify_coverage.novel - verify_coverage.offered
        # "handed ... to the verifier", never "verified": this is the cap's
        # COST, and a handed cluster may come back FALSE. The old "verified N
        # of M" spelling read as an outcome, and directly under the
        # mass-rejection notice above it contradicted it outright.
        if deferred > 0:
            verification.append(
                f"- handed {verify_coverage.offered} of {verify_coverage.novel} novel "
                f"clusters to the verifier (operator verify cap: {verify_coverage.cap}); "
                f"{deferred} deferred as pending candidates -- merged into the codebook "
                "by this run but NOT verified; adjudication deferred to a later census."
            )
            # "a later census" is conditional, not automatic: the deferred
            # clusters' sessions were coded, so they are ledgered and skipped.
            verification.append(
                "- a deferred candidate is re-adjudicated only if the same confusion "
                "RECURS: its sessions were coded this run and are ledgered as coded, "
                "so they are not re-mined. A one-off deferred by the cap stays "
                "pending until it is adjudicated by hand."
            )
        else:
            # A cap that was SET BUT NOT REACHED must not emit the deferral
            # clause -- nothing was deferred and nothing went unverified.
            verification.append(
                f"- handed all {verify_coverage.novel} novel cluster(s) to the "
                f"verifier; operator verify cap: {verify_coverage.cap} (not reached)."
            )

    if verification:
        emit(SECTION_VERIFICATION, verification)

    if dropped_verdicts:
        # Same NO-SILENT-CAPS reasoning as the coverage lines above, applied to
        # the other thing this run silently paid for: stdout and the WARNING log
        # of an unattended nightly run are gone by morning, and the dated report
        # is the artifact an operator actually reads afterwards. "This run paid
        # for N adjudications that went nowhere" belongs in the channel that
        # survives. Gated on a non-empty list so a clean run stays byte-identical
        # to the golden.
        unresolved = [
            "",
            "## Unresolved Verdicts",
            "",
            f"- {len(dropped_verdicts)} verdict(s) this run PAID FOR resolved to no "
            "pending candidate and were DROPPED. Nothing is lost from the codebook -- "
            "a standing prior adjudication of the same title holds, which is the "
            "correct outcome -- but the adjudication spend went nowhere and only a "
            "hand re-open will change that.",
        ]
        unresolved.extend(
            f"  - {dropped_verdict_message(record)}" for record in dropped_verdicts
        )
        emit(SECTION_UNRESOLVED_VERDICTS, unresolved)

    emit(SECTION_MATRIX, ["", "## Origin x Manifestation Matrix", "", record["matrix_md"]])

    # NO leading blank line, deliberately: `matrix_md` is embedded verbatim and
    # carries its own trailing newline, so `## Synthesis` follows it
    # immediately. The golden pins `matrix\n## Synthesis`; normalising this
    # during a refactor is the one plausible way to break byte-identity while
    # every structural assertion still passes.
    emit(SECTION_SYNTHESIS, synthesis.report_lines(record["synthesis"]))

    emit(SECTION_FINDINGS, census_findings.report_lines(record))

    filed_tasks = ["", "## Filed Tasks", ""]
    if dry_run is not None:
        # First: a dry run files nothing by construction, and a bare
        # "_none filed._" would read as a normal run with nothing to file.
        filed_tasks.extend([
            f"_dry-run: {dry_run.payload_count} payload(s) written to {dry_run.path} "
            "-- NOTHING filed; review before filing._",
            "",
        ])
    filed_tasks.extend(
        disposition.filed_tasks_lines(record["filing"], observed_root=record["project_root"])
    )
    emit(SECTION_FILED_TASKS, filed_tasks)
    emit(SECTION_STRUCTURAL, disposition.structural_lines(record["filing"]))
    emit(
        SECTION_DISPOSITIONS,
        disposition.dispositions_lines(record["dispositions"], observed_root=record["project_root"]),
    )

    if withheld:
        recorded_not_filed = [
            "",
            "## Recorded, Not Filed",
            "",
            f"- {len(withheld)} verified cluster(s) were promoted into the codebook "
            f"and marked withheld, not filed: each has fewer than "
            f"{filing_policy.MIN_UNREMEDIATED_SIGHTINGS} sightings and no in-tree "
            "remediation from the verifier. Each files automatically once a later "
            f"census counts {filing_policy.MIN_UNREMEDIATED_SIGHTINGS} sightings of it.",
        ]
        recorded_not_filed.extend(
            f"  - {cluster.title} (sightings: {cluster.sighting_count})" for cluster in withheld
        )
        emit(SECTION_WITHHELD, recorded_not_filed)

    # The trailing "" is the report's final newline, which the join would
    # otherwise not supply.
    emit(SECTION_COST, ["", "## Cost", "", record["cost_note"], ""])

    return tuple(sections)


def join_report_sections(sections: tuple[ReportSection, ...]) -> str:
    """Render *sections* to the markdown an operator reads.

    Each section carries its own leading blank line, so this is a plain
    concatenation -- there is no separator policy here that could disagree
    with what a section believes its own shape is."""
    return "\n".join(line for section in sections for line in section.lines)


def render_report(record: Mapping[str, Any]) -> str:
    """The census report markdown for *record*: a pure join of
    :func:`census_report_sections`, which owns the structure and every
    NO-SILENT-CAPS gate. No clock, no model call, no I/O."""
    return join_report_sections(census_report_sections(record))


DROP_CONTRADICTION = "contradicts-standing"
DROP_AGREEMENT = "agrees-with-standing"
"""Stable machine keys for whether the standing candidate a dropped verdict
met contradicts or agrees with it: what an operator must DO differs, so it is
carried as data, like the ``SECTION_*`` report keys, not matched out of a log
line."""


@dataclass(frozen=True)
class DroppedVerdict:
    """One verdict this run PAID FOR whose offered candidate was no longer
    pending, so it was discarded: a surfaced cluster the merge routed to a
    standing adjudicated candidate, which keeps its verdict (task 4144)."""

    verdict: str
    """``"verified"`` or ``"rejected"`` -- which adjudication loop paid for it.
    Inverts against ``standing_disposition``; see :func:`dropped_verdict`."""

    title: str | None
    kind: str
    """``DROP_CONTRADICTION`` / ``DROP_AGREEMENT``."""

    standing_id: str
    standing_disposition: str
    standing_first_seen: str
    """The standing candidate, named by the id the verdict resolved to."""


def dropped_verdict(
    *, verdict: str, cluster: Mapping[str, Any], standing: Mapping[str, Any], contradicting_disposition: str,
) -> DroppedVerdict:
    """Classify one dropped *verdict* against its *standing* candidate.

    The two adjudication loops differ only in *contradicting_disposition*: a
    standing PROMOTION contradicts a fresh REJECT, and a standing REJECT
    contradicts a fresh VERIFY."""
    return DroppedVerdict(
        verdict=verdict,
        title=cluster.get("title"),
        kind=DROP_CONTRADICTION if standing["disposition"] == contradicting_disposition else DROP_AGREEMENT,
        standing_id=standing["id"],
        standing_disposition=standing["disposition"],
        standing_first_seen=standing["first_seen"],
    )


_DROPPED_VERDICT_LOSS = {
    "verified": (
        "this run CONFIRMED a pattern a prior verdict called unfounded, so it enters "
        "NO codebook entry and is invisible to every later census"
    ),
    "rejected": (
        "a live codebook entry stands for a title this run judged unfounded"
    ),
}
"""What a CONTRADICTED drop actually COSTS, per loop -- the one clause
:func:`dropped_verdict_message` cannot render uniformly, because the two
losses differ in substance and not merely in wording."""


def dropped_verdict_message(record: DroppedVerdict) -> str:
    """The operator-facing prose for *record* -- rendered FROM the record, so
    the wording is free to change without breaking anything that asserts on
    the drop.

    Carries no ``census:`` prefix and no leading bullet: it is the ONE
    rendering, read both from the run's WARNING log and from the persisted
    report's ``## Unresolved Verdicts`` section, and each of those supplies
    its own framing."""
    standing = (
        f"id={record.standing_id}, disposition={record.standing_disposition}, "
        f"first_seen={record.standing_first_seen}"
    )
    if record.kind == DROP_CONTRADICTION:
        return (
            f"{record.verdict} cluster {record.title!r} resolved to no pending "
            f"candidate -- this verdict is DROPPED and CONTRADICTS the standing "
            f"record ({standing}): {_DROPPED_VERDICT_LOSS[record.verdict]}. Nothing "
            "reconciles the two; only a hand re-open will change it."
        )
    return (
        f"{record.verdict} cluster {record.title!r} resolved to no pending "
        f"candidate -- this verdict is DROPPED and AGREES with the standing record "
        f"({standing}). Nothing to do; only the verify call was spent."
    )
