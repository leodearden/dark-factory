"""Whether a sighting attached to a codebook entry exhibits the entry's defining
mechanism, and so counts toward the singleton filing gate
(plans/census-incremental-prd.md C1)."""
from __future__ import annotations

import copy
import enum
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from legibility import codebook, filing_policy, prescreen

logger = logging.getLogger("legibility.census.sighting_fit")


class Standing(enum.StrEnum):
    """What one session's sightings on an entry are worth to its filing gate."""

    FOUNDING = "founding"
    ATTACHED_BY_RULE = "attached_by_rule"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNCHECKED = "unchecked"
    UNEVIDENCED = "unevidenced"


COUNTED = frozenset({Standing.FOUNDING, Standing.ATTACHED_BY_RULE, Standing.CONFIRMED})

_RULE_METHODS = frozenset({prescreen.Rule.TITLE, prescreen.Rule.KEY})


def founding_sessions(cb: Mapping[str, Any], entry: Mapping[str, Any]) -> frozenset[str]:
    """The sessions of the candidate *entry* was promoted from, or, for an
    entry with none, of its first sighting."""
    candidate = codebook.promoted_from(cb, entry["id"])
    sessions = codebook.sessions_of(candidate) if candidate else []
    return frozenset(sessions or codebook.sessions_of(entry)[:1])


def _verdicts(entry: Mapping[str, Any]) -> dict[str, bool]:
    return {row["session"]: row["exhibits"] for row in entry.get(codebook.SIGHTING_FITS_KEY) or []}


def _sightings_by_session(entry: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    by_session: dict[str, list[Mapping[str, Any]]] = {}
    for sighting in entry.get("sightings") or []:
        session = sighting.get("session")
        if isinstance(session, str):
            by_session.setdefault(session, []).append(sighting)
    return by_session


def _standing(
    session: str,
    sightings: Sequence[Mapping[str, Any]],
    *,
    founding: frozenset[str],
    verdicts: Mapping[str, bool],
) -> Standing:
    if session in founding:
        return Standing.FOUNDING
    if any((sighting.get("attached_by") or {}).get("method") in _RULE_METHODS for sighting in sightings):
        return Standing.ATTACHED_BY_RULE
    if session in verdicts:
        return Standing.CONFIRMED if verdicts[session] else Standing.REFUTED
    if any(sighting.get("evidence_quote") for sighting in sightings):
        return Standing.UNCHECKED
    return Standing.UNEVIDENCED


def standings(cb: Mapping[str, Any], entry: Mapping[str, Any]) -> Mapping[str, Standing]:
    """The Standing of each session *entry*'s sightings carry."""
    founding = founding_sessions(cb, entry)
    verdicts = _verdicts(entry)
    return {
        session: _standing(session, sightings, founding=founding, verdicts=verdicts)
        for session, sightings in _sightings_by_session(entry).items()
    }


def counted_sessions(cb: Mapping[str, Any], entry: Mapping[str, Any]) -> frozenset[str]:
    """The sessions that count toward *entry*'s singleton filing gate: its
    founding ones, rule-attached ones and fit-confirmed ones."""
    return frozenset(session for session, standing in standings(cb, entry).items() if standing in COUNTED)


@dataclass(frozen=True)
class OfferedSighting:
    session: str
    evidence_quote: str


@dataclass(frozen=True)
class FitCheck:
    """One entry whose release a fit verdict decides: what the judge sees.
    Only verbatim evidence is offered, never a coder's note, which is the
    coder's own claim of fit."""

    entry_id: str
    title: str
    cause: str | None
    founding_evidence: tuple[str, ...]
    offered: tuple[OfferedSighting, ...]


@dataclass(frozen=True)
class FitVerdict:
    session: str
    exhibits: bool
    reason: str


FitFn = Callable[..., Sequence[FitVerdict]]
"""``fit_fn(check, *, model) -> verdicts``: judges a FitCheck's offered sightings."""


@dataclass(frozen=True)
class FitOutcome:
    entry_id: str
    title: str
    confirmed: tuple[str, ...]
    refuted: tuple[str, ...]
    unchecked: tuple[str, ...]
    error: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "title": self.title,
            "confirmed": list(self.confirmed),
            "refuted": list(self.refuted),
            "unchecked": list(self.unchecked),
            "error": self.error,
        }


def _first_quote(sightings: Iterable[Mapping[str, Any]]) -> str | None:
    return next((sighting["evidence_quote"] for sighting in sightings if sighting.get("evidence_quote")), None)


def _fit_check(cb: Mapping[str, Any], entry: Mapping[str, Any], unchecked: Sequence[str]) -> FitCheck:
    by_session = _sightings_by_session(entry)
    founding = founding_sessions(cb, entry)
    founding_evidence = tuple(
        quote
        for session, sightings in by_session.items()
        if session in founding and (quote := _first_quote(sightings)) is not None
    )
    offered = tuple(
        OfferedSighting(session=session, evidence_quote=quote)
        for session in unchecked
        if (quote := _first_quote(by_session[session])) is not None
    )
    return FitCheck(
        entry_id=entry["id"],
        title=entry["title"],
        cause=codebook.entry_field(cb, entry, "cause"),
        founding_evidence=founding_evidence,
        offered=offered,
    )


def at_stake(cb: Mapping[str, Any], entries: Iterable[Mapping[str, Any]]) -> tuple[FitCheck, ...]:
    """A FitCheck for each of *entries* below the singleton gate that its
    unchecked sightings could carry over it: the only entries whose filing a
    fit verdict decides."""
    checks = []
    for entry in entries:
        entry_standings = standings(cb, entry)
        counted = sum(1 for standing in entry_standings.values() if standing in COUNTED)
        unchecked = [session for session, standing in entry_standings.items() if standing is Standing.UNCHECKED]
        if counted < filing_policy.MIN_UNREMEDIATED_SIGHTINGS <= counted + len(unchecked):
            checks.append(_fit_check(cb, entry, unchecked))
    return tuple(checks)


def _judge(fit_check: FitCheck, fit_fn: FitFn, model: str) -> tuple[dict[str, FitVerdict], str | None]:
    """The first verdict *fit_fn* gives each offered session, or none and the
    error when it fails: an unrun check only delays a filing."""
    try:
        verdicts = tuple(fit_fn(fit_check, model=model))
    except Exception as exc:  # any failure leaves the sightings unchecked, never crashes the census
        logger.warning("sighting fit check for entry %r failed: %s", fit_check.entry_id, exc)
        return {}, f"{type(exc).__name__}: {exc}"
    offered = {offer.session for offer in fit_check.offered}
    chosen: dict[str, FitVerdict] = {}
    for verdict in verdicts:
        if verdict.session in offered:
            chosen.setdefault(verdict.session, verdict)
    return chosen, None


def _stamp(entry: dict[str, Any], verdicts: Sequence[FitVerdict], run_id: str) -> None:
    entry.setdefault(codebook.SIGHTING_FITS_KEY, []).extend(
        {"session": verdict.session, "exhibits": verdict.exhibits, "run_id": run_id, "reason": verdict.reason}
        for verdict in verdicts
    )
    errors = codebook.sighting_fit_errors(entry)
    if errors:
        raise ValueError(f"sighting fit rows stamped on entry {entry['id']!r} are malformed: {errors}")


def check(
    cb: Mapping[str, Any],
    checks: Sequence[FitCheck],
    *,
    fit_fn: FitFn,
    model: str,
    run_id: str,
) -> tuple[dict, tuple[FitOutcome, ...]]:
    """Run one *fit_fn* call per check and stamp its verdicts as
    ``sighting_fits`` rows on a copy of *cb*; *cb* itself is unchanged."""
    stamped = copy.deepcopy(dict(cb))
    outcomes = []
    for fit_check in checks:
        chosen, error = _judge(fit_check, fit_fn, model)
        sessions = tuple(offer.session for offer in fit_check.offered)
        verdicts = [chosen[session] for session in sessions if session in chosen]
        if verdicts:
            entry = codebook.record_by_id(stamped["entries"], fit_check.entry_id, kind="entry")
            _stamp(entry, verdicts, run_id)
        outcomes.append(
            FitOutcome(
                entry_id=fit_check.entry_id,
                title=fit_check.title,
                confirmed=tuple(verdict.session for verdict in verdicts if verdict.exhibits),
                refuted=tuple(verdict.session for verdict in verdicts if not verdict.exhibits),
                unchecked=tuple(session for session in sessions if session not in chosen),
                error=error,
            )
        )
    codebook.assert_no_deletion(dict(cb), stamped)
    return stamped, tuple(outcomes)
