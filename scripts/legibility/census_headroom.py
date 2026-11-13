"""The census's usage-headroom probe, and the deferral it signals."""
from __future__ import annotations

import enum
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from shared.cap_markers import looks_like_blocking_banner

logger = logging.getLogger("legibility.census.headroom")


HEADROOM_PROBE_PROMPT = "ping"
"""The tiny probe prompt -- a cheap single round trip, not a real mining
call. Its content doesn't matter; only whether the reply carries a banner
marker or raises."""


class CensusStage(enum.StrEnum):
    """The headroom gates at which a census defers."""

    PREFLIGHT = "preflight"
    VERIFY = "verify"


@dataclass(frozen=True)
class DeferralCounts:
    """What a verify-stage deferral had adjudicated when the cap hit."""

    verified: int = 0
    rejected: int = 0
    unverified: int = 0

    @property
    def total(self) -> int:
        return self.verified + self.rejected + self.unverified


class CensusHeadroomExhausted(Exception):
    """Raised when a headroom limit is detected DURING a census stage.

    Not an error in the census -- an external condition that makes the
    stage's output untrustworthy. It exists so an infra failure can never
    be laundered into a verdict: ``census_run.adjudicate`` catches it and defers
    before any persistence, instead of the cap-poisoned clusters landing
    in the codebook as ordinary rejections.

    Carries the counts an operator needs to size what was interrupted:
    *verified* clusters that came back TRUE before the hit, *rejected*
    ones adjudicated FALSE before it, and *unverified* ones left (the
    hitting cluster plus every one never attempted). A rejection is real
    work already spent, not an absence -- omitting it made the counts fail
    to account for the run on exactly the interrupted runs an operator
    most needs to size.

    INVARIANT, true at every raise site: ``verified + rejected +
    unverified`` equals the number of clusters offered. At the two
    pre-append sites (invocation error, unparseable banner) the hitting
    cluster is not yet adjudicated, so ``verified + rejected == index``
    and ``unverified == remaining``; at the backstop site it already IS
    adjudicated, so ``verified + rejected == index + 1`` and ``unverified
    == remaining - 1``. The total is therefore derivable by any reader and
    is NOT threaded separately -- see ``census_run.py::_defer``, which sums it, so the
    raise sites cannot disagree about what "total" means.
    """

    def __init__(
        self, *, stage: CensusStage, reason: str, verified: int = 0, rejected: int = 0,
        unverified: int = 0,
    ):
        super().__init__(f"census headroom exhausted during {stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.verified = verified
        self.rejected = rejected
        self.unverified = unverified

    @property
    def counts(self) -> DeferralCounts:
        return DeferralCounts(verified=self.verified, rejected=self.rejected, unverified=self.unverified)


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
    case-insensitively for a known usage-limit/auth banner marker via
    ``shared.cap_markers.looks_like_blocking_banner`` (capacity OR auth --
    either means no useful model output is coming); a match defers, and
    the reason quotes the marker that fired. An invocation error
    raised by *invoke* (e.g. a ``session_runner.InvocationFailed``-shaped failure)
    is also treated as a deferral -- fail-safe, never a crash, since a
    probe failure is exactly the kind of "the model isn't reachable right
    now" signal this preflight exists to catch. Makes no mining decisions
    itself -- just the ok/deferred verdict + reason.

    In production *invoke* is the pooled runner's mining stage, so a capped
    account fails over to the next INSIDE the probe; only a pool with no
    headroom left raises (``session_runner.NoHeadroom``) and defers the
    census (task 5515).
    """
    try:
        reply = invoke(HEADROOM_PROBE_PROMPT, model)
    except Exception as exc:  # noqa: BLE001 - any probe failure must fail safe
        return HeadroomResult(ok=False, reason=f"headroom probe invocation failed: {exc}")

    marker = looks_like_blocking_banner(reply or "")
    if marker is not None:
        return HeadroomResult(
            ok=False, reason=f"headroom probe reply carries a banner marker: {marker!r}",
        )

    return HeadroomResult(ok=True, reason=None)


def stage_headroom_gate(invoke, *, model: str, stage: CensusStage) -> HeadroomResult:
    """Re-probe headroom at a stage boundary, logging which stage asked.

    Deliberately NOT a second probe implementation -- it is a thin wrapper
    over ``preflight_headroom`` that adds the stage name to the log and
    nothing else, so the "one tiny call, no usage API assumed" decision
    (PRD decision 5) and its fail-safe exception handling hold identically
    at every gate.

    Gates exist only where an unnoticed cap CORRUPTS output, not at every
    stage transition. Today that means the preflight and this one, before
    verification. There is deliberately NO gate before synthesis: a cap
    there makes ``synthesize_fn`` RAISE, which propagates into ``census.main()``'s
    hard-failure path and is already loud and already distinguishable from
    a verification result -- so a gate there would only abort a run whose
    ``verified`` list is complete and honest, discarding work that was
    genuinely paid for. Mining is the same: a storm batch is already
    detected and reported. Do not "complete the pattern" by adding them.
    """
    result = preflight_headroom(invoke, model=model)
    if not result.ok:
        logger.warning(
            "census headroom gate FAILED at the %s stage boundary: %s",
            stage, result.reason,
        )
    return result
