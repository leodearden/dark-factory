"""Coding batches until novelty saturates."""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from legibility import coder

logger = logging.getLogger("legibility.census.mining")


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


MAX_PER_DIGEST_CODER_WARNINGS_PER_BATCH = 3
"""How many of ``coder.code_digests``' per-digest WARNINGs may survive ONE
census mining batch before the rest are bounded away in favour of the
batch aggregate.

Small, but deliberately NOT zero. Zero would mean a census run silently
swallows another module's log records -- the exact silent-degradation
shape this fix exists to close. A small allowance keeps the raw,
unaggregated line shape visible for the common one-or-two-failure batch
(where there never was a flood), makes the suppression obviously PARTIAL
rather than total, and still turns a 20-digest storm from 20 lines into
4. The aggregate line states how many lines were bounded away, so the
drop stays loud."""


@contextlib.contextmanager
def _bounded_coder_warnings(limit: int):
    """Bound ``legibility.coder``'s per-digest WARNINGs to *limit* records
    for the duration of the ``with`` block, yielding the counting filter
    so the caller can read ``.suppressed``.

    Deliberately CALLER-SCOPED, and removed in a ``finally`` so an
    exception mid-batch cannot leave it attached. ``code_digests``'
    per-digest WARNING is the only sink some failures ever reach for
    ``nightly.run_nightly``'s trickle -- one or two failures a night,
    where the per-digest shape is exactly right -- so it must never be
    silenced globally, dropped to DEBUG, or bounded beyond the census's
    own call.

    A counting ``logging.Filter`` rather than either alternative: raising
    the coder logger's LEVEL would suppress any FUTURE non-digest coder
    warning too, and matching on the message template would couple census
    to a string in ``coder.py`` -- a file this task holds no lock on and
    which is free to reword. Counting is surgical enough because the bound
    applies a WARNING FLOOR of its own: sub-WARNING records pass through
    untouched and unconsumed, so this is a bound on the per-digest warning
    funnel and not on whatever else ``coder.py`` may one day log."""

    class _CountingFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self.seen = 0
            self.suppressed = 0

        def filter(self, record):
            if record.levelno < logging.WARNING:
                # The contract in the names around this filter is
                # per-digest WARNINGs, so the filter ENFORCES that floor
                # rather than leaning on "coder.py currently logs nothing
                # else" (true today, unenforceable tomorrow). Without it a
                # future logger.info/debug in coder.py would eat the budget,
                # push real per-digest WARNINGs out of a census run, and
                # falsify the `suppressed` count reported in the aggregate.
                return True
            self.seen += 1
            if self.seen > limit:
                self.suppressed += 1
                return False
            return True

    coder_logger = logging.getLogger("legibility.coder")
    counting_filter = _CountingFilter()
    coder_logger.addFilter(counting_filter)
    try:
        yield counting_filter
    finally:
        coder_logger.removeFilter(counting_filter)


@dataclass
class BatchStats:
    """Per-batch mining tally: how one ``coder.code_digests`` batch scored
    against the saturation threshold.

    ``status`` mirrors ``coder.RunResult.status`` (``coder.CodingStatus``;
    FAILURE is a storm, PRD §8.6, when this batch's failed/total fraction
    strictly exceeds 0.5): surfaced so a caller (``census_run.run_census``,
    or any future one) can tell a batch's ``dup_rate`` was drawn from a
    small, skewed, storm-degraded sample rather than a healthy one, even
    though ``saturated`` already accounts for it (a storm batch is never
    ``saturated=True`` -- see ``mine_to_saturation``)."""

    index: int
    total: int
    succeeded: int
    failed: int
    dup_rate: float
    saturated: bool
    status: coder.CodingStatus = coder.CodingStatus.OK

    @classmethod
    def from_record(cls, stats: Mapping[str, Any]) -> BatchStats:
        """The batch as the census record holds it (:func:`census_report.build_census_record`),
        its status a JSON string; one outside ``coder.CodingStatus`` raises
        ``ValueError``."""
        return cls(**{**stats, "status": coder.CodingStatus(stats["status"])})


@dataclass
class MiningResult:
    """Outcome of ``mine_to_saturation``: every successfully-coded record
    across every consumed batch, per-batch stats, and why mining stopped
    (``"saturated"`` -- ``config.consecutive_batches`` consecutive batches
    at/above ``config.dup_rate``; ``"exhausted"`` -- ``batch_source`` ran
    out first; ``"capped"`` -- the operator's ``max_batches`` cap was
    reached, so coverage is deliberately PARTIAL).

    ``max_batches`` echoes the operator cap this run was given (``None``
    when uncapped -- the default). It travels on the result so
    ``census_report.render_report`` can state the cap and its coverage consequence from
    the mining facts it is already handed, without a second parameter."""

    records: list[dict] = field(default_factory=list)
    batch_stats: list[BatchStats] = field(default_factory=list)
    stop_reason: str = "exhausted"
    max_batches: int | None = None


def mine_to_saturation(
    batch_source, codebook_dict: dict, *, project: str, model: str, config,
    invariant_slugs: Sequence[str], invoke, max_batches: int | None = None,
) -> MiningResult:
    """Code batches from *batch_source* against *codebook_dict* via
    ``coder.code_digests`` until novelty saturates.

    Iterates *batch_source* lazily -- a batch is pulled only when the loop
    is ready to code it, so a source that stops yielding once mining
    saturates never has its later batches materialized. Per batch: code
    via ``coder.code_digests(batch, codebook_dict, project=project,
    model=model, invariant_slugs=invariant_slugs, invoke=invoke)``,
    compute this batch's ``dup_rate`` over its own successful records
    (``batch_dup_rate`` -- failed codings are
    excluded from the denominator per that function's own contract), and
    track a consecutive-saturated-batch counter: incremented when
    ``dup_rate >= config.dup_rate`` AND the batch is not a storm, reset to
    0 otherwise. Mining stops (``stop_reason="saturated"``) the moment the
    counter reaches ``config.consecutive_batches`` -- right after that Nth
    consecutive saturated batch, so no further batch is pulled. If
    *batch_source* exhausts before that, ``stop_reason="exhausted"``.

    A STORM batch (``coder.CodingStatus.FAILURE`` -- PRD §8.6,
    more than half the batch's digests failed to code) never counts toward
    saturation, however high its ``dup_rate`` computes: that rate is drawn
    from whatever handful of records happened to succeed, too small and
    too skewed a sample to trust, and a run of storms would otherwise be
    able to trip the saturation stop on degraded data rather than genuine
    novelty exhaustion. The batch's ``dup_rate`` is still recorded on its
    ``BatchStats`` (with ``status=CodingStatus.FAILURE``) for visibility, but
    ``saturated`` is forced False and the consecutive-saturated counter is
    reset, exactly as if the batch scored below threshold.

    A batch with ANY coding failures emits exactly ONE aggregated WARNING
    naming the batch index, ``failed/total``, how many DISTINCT failure
    reasons there were, and per reason its count plus one example session
    id. A batch that coded cleanly stays silent. This is the caller-side
    summary ``coder.code_digests``' own docstring points at: that function
    logs one WARNING per failed digest, which is the right shape for the
    nightly trickle (one or two failures a night) and a flood for a census
    batch of 20. Reasons are grouped by EXACT string equality with no
    normalization -- the property those per-digest lines exist for is
    telling 20 identical ENOENTs apart from 20 distinct model errors, and
    any canonicalization is a guess that can collapse precisely that
    distinction. Output stays bounded without a cap on distinct reasons
    because a batch has at most ``census_window.DEFAULT_BATCH_SIZE`` digests, so
    the worst case is one long line rather than N lines.

    The aggregate does not merely ADD to the flood: the
    ``coder.code_digests`` call is wrapped in ``_bounded_coder_warnings``,
    which caps its per-digest WARNINGs at
    ``MAX_PER_DIGEST_CODER_WARNINGS_PER_BATCH`` for the duration of THIS
    call only, and the aggregate names how many lines that bound dropped.
    The bound is caller-scoped by construction and removed in a ``finally``
    -- ``nightly.run_nightly``'s trickle keeps full per-digest visibility.

    *max_batches* is the OPERATOR COST CAP (``--max-batches``): mining
    stops with ``stop_reason="capped"`` once that many batches have been
    coded. The cap is enforced here, inside the loop, rather than by
    islicing *batch_source* upstream, for two reasons: only this loop
    knows WHY it stopped, so ``"capped"`` stays distinguishable from a
    source that genuinely ran dry (an islice wrapper is indistinguishable
    from ``"exhausted"`` -- exactly the silent-cap failure the cap must
    not introduce), and the ``return`` happens before the next batch is
    pulled, so a capped-away batch costs no digest render and no
    ``coder.code_digests`` call. A capped run is deliberately PARTIAL
    coverage: sessions beyond the cap were never mined, and
    ``census_report.render_report`` says so in as many words. The saturation check
    deliberately PRECEDES the cap check, so a run that saturates on
    exactly the capped batch reports the stronger, more informative
    ``"saturated"`` (novelty genuinely exhausted -- coverage was
    sufficient regardless of the cap) rather than under-claiming as
    partial. ``max_batches=None`` (the default) is exactly today's
    behavior: no cap, no extra rendering, unbounded mining. A cap below 1
    raises ``ValueError`` rather than degrading silently: because the cap
    is checked AFTER a batch is coded, ``max_batches=0`` would still spend
    one full ``coder.code_digests`` call and then render a
    self-contradictory "mined 1 batch(es); operator batch cap = 0" line --
    a silently mis-honored cap on a parameter whose whole purpose is to be
    an explicit, legible bound.

    *config* is a ``config.Saturation``-shaped object (``.dup_rate``,
    ``.consecutive_batches`` -- i.e. a project's
    ``LegibilityConfig.census.saturation``), not the whole
    ``LegibilityConfig``. *model* is the caller's already-resolved model
    id (Sonnet miner routing per the ratified static policy -- this
    function does not read ``config.Models`` itself).
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError(
            f"max_batches must be >= 1 when set, got {max_batches!r} -- a cap of "
            "0 or less cannot be honored (the cap is checked after a batch is "
            "coded, so it would still spend one full coder.code_digests call) "
            "and would render a self-contradictory coverage line; omit it "
            "entirely for unbounded mining"
        )

    result = MiningResult(max_batches=max_batches)
    consecutive_saturated = 0

    for index, batch in enumerate(batch_source):
        # ONLY this call is bounded -- see _bounded_coder_warnings' docstring
        # for why the bound must never outlive it.
        with _bounded_coder_warnings(
            MAX_PER_DIGEST_CODER_WARNINGS_PER_BATCH
        ) as bounded:
            run_result = coder.code_digests(
                list(batch), codebook_dict, project=project, model=model,
                invariant_slugs=invariant_slugs, invoke=invoke,
            )
        result.records.extend(run_result.records)

        dup_rate = batch_dup_rate(run_result.records)
        # A storm (coder.CodingStatus.FAILURE) never satisfies
        # saturation, no matter how high dup_rate computes over its few
        # successful records -- see the storm paragraph above.
        saturated = (
            dup_rate >= config.dup_rate
            and run_result.status != coder.CodingStatus.FAILURE
        )
        result.batch_stats.append(
            BatchStats(
                index=index,
                total=run_result.total,
                succeeded=run_result.succeeded,
                failed=run_result.failed,
                dup_rate=dup_rate,
                saturated=saturated,
                status=run_result.status,
            )
        )

        # The CALLER-SIDE aggregate `coder.code_digests`' docstring points
        # at: that function emits one WARNING per failed digest, which is
        # the right shape for the nightly trickle's one-or-two failures and
        # a flood for a 20-digest census batch. This line is the batch-level
        # summary that a flood cannot be read as. Grouping is EXACT-STRING by
        # design -- the property the per-digest lines exist for is telling N
        # identical ENOENTs apart from N genuinely distinct model errors, and
        # any normalization is a guess that can collapse the two. `coder.py`
        # is deliberately NOT modified (its per-digest line is the only sink
        # some failures ever reach for callers other than this one).
        if run_result.failed:
            by_reason: dict[str, list[str]] = {}
            for session, reason in run_result.failures:
                by_reason.setdefault(reason, []).append(session)
            breakdown = "; ".join(
                f"{len(sessions)}x {reason!r} (e.g. session={sessions[0]})"
                for reason, sessions in by_reason.items()
            )
            # State the bound's own cost: a suppressed line is a dropped
            # record, and a silent drop is the defect this whole line closes.
            bound_note = (
                f" [{bounded.suppressed} per-digest coder line(s) suppressed by "
                f"this batch's bound of {MAX_PER_DIGEST_CODER_WARNINGS_PER_BATCH}]"
                if bounded.suppressed else ""
            )
            logger.warning(
                "mining batch %d: %d/%d digest(s) failed to code, %d distinct "
                "reason(s): %s%s",
                index, run_result.failed, run_result.total, len(by_reason),
                breakdown, bound_note,
            )

        consecutive_saturated = consecutive_saturated + 1 if saturated else 0
        if consecutive_saturated >= config.consecutive_batches:
            result.stop_reason = "saturated"
            return result

        # Checked AFTER saturation, deliberately -- see the max_batches
        # paragraph above. Returning here means batch N+1 is never pulled.
        if max_batches is not None and len(result.batch_stats) >= max_batches:
            logger.warning(
                "mining stopped at the operator batch cap: %d batch(es) mined "
                "(--max-batches=%d) -- coverage is PARTIAL, not saturated; "
                "sessions beyond the cap were NOT mined",
                len(result.batch_stats), max_batches,
            )
            result.stop_reason = "capped"
            return result

    result.stop_reason = "exhausted"
    return result
