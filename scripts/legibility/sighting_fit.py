"""Whether a sighting attached to a codebook entry exhibits the entry's defining
mechanism, and so counts toward the singleton filing gate
(plans/census-incremental-prd.md C1)."""
from __future__ import annotations

import copy
import enum
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from legibility import codebook, coder, filing_policy, prescreen, report_text, session_runner

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


Releases = Callable[[Mapping[str, Any], int], bool]
"""``releases(entry, sighting_count)``: whether the filing gate files *entry*
at *sighting_count* counted sessions."""


def at_stake(
    cb: Mapping[str, Any], entries: Iterable[Mapping[str, Any]], *, releases: Releases,
) -> tuple[FitCheck, ...]:
    """A FitCheck for each of *entries* the gate *releases* holds back at its
    counted sessions but files once its unchecked sightings count too: the
    only entries whose filing a fit verdict decides."""
    checks = []
    for entry in entries:
        entry_standings = standings(cb, entry)
        counted = sum(1 for standing in entry_standings.values() if standing in COUNTED)
        unchecked = [session for session, standing in entry_standings.items() if standing is Standing.UNCHECKED]
        if unchecked and not releases(entry, counted) and releases(entry, counted + len(unchecked)):
            checks.append(_fit_check(cb, entry, unchecked))
    return tuple(checks)


def _judge(fit_check: FitCheck, fit_fn: FitFn, model: str) -> tuple[dict[str, FitVerdict], str | None]:
    """The first verdict *fit_fn* gives each offered session, or none and the
    error when the judge was unreachable or gave no judgment: an unrun check
    only delays a filing. Any other exception is a defect and propagates."""
    try:
        verdicts = tuple(fit_fn(fit_check, model=model))
    except (session_runner.InvocationFailed, FitReplyError) as exc:
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


class FitReplyError(ValueError):
    """A fit judge's reply that is no judgment of the offered sightings."""


_REPLY_SHAPE = {"sightings": [{"session": "...", "exhibits": True, "reason": "..."}]}


def _fit_prompt(fit_check: FitCheck) -> str:
    """The judge's task and the check it judges. Asks for an observation from
    the quoted text, never a diagnosis."""
    payload = {
        "entry_id": fit_check.entry_id,
        "title": fit_check.title,
        "cause": fit_check.cause,
        "founding_evidence": list(fit_check.founding_evidence),
        "sightings": [
            {"session": offer.session, "evidence_quote": offer.evidence_quote} for offer in fit_check.offered
        ],
    }
    return (
        "You are the sighting fit judge for the dark-factory agent-confusion "
        "codebook (plans/census-incremental-prd.md C1). The entry below names "
        "one confusion mechanism in its title, and its founding evidence shows "
        "that mechanism. Each sighting under it was attached to the entry by a "
        "cheaper judgment. For each sighting, decide from its quoted evidence "
        "alone whether it exhibits the mechanism the entry's title names, not "
        "merely a related theme. Judge only what the quoted text shows: an "
        "observation, never a diagnosis or a guess about what the session may "
        "also have done. When the quote does not show the mechanism, exhibits "
        "is false.\n\n"
        "Respond with STRICT JSON ONLY (no prose, no markdown fences), one item "
        "per sighting, exactly this shape:\n"
        + json.dumps(_REPLY_SHAPE) + "\n\n"
        "=== ENTRY AND SIGHTINGS ===\n" + json.dumps(payload, ensure_ascii=False)
    )


def _verdict(item: object) -> FitVerdict | None:
    if not isinstance(item, dict):
        return None
    session, exhibits, reason = item.get("session"), item.get("exhibits"), item.get("reason")
    if not isinstance(session, str) or not isinstance(exhibits, bool):
        return None
    return FitVerdict(session=session, exhibits=exhibits, reason=reason if isinstance(reason, str) else "")


def _parse_reply(raw: str) -> tuple[FitVerdict, ...]:
    try:
        parsed = coder.parse_coder_output(raw)
    except coder.CoderParseError as exc:
        raise FitReplyError(f"the fit reply is no JSON object: {exc}") from exc
    items = parsed.get("sightings")
    if not isinstance(items, list):
        raise FitReplyError(f"the fit reply's 'sightings' is {type(items).__name__}, not a list")
    return tuple(verdict for item in items if (verdict := _verdict(item)) is not None)


def build_fit_fn(invoke: Callable[[str, str], str]) -> FitFn:
    """The production ``fit_fn(check, *, model)``: one *invoke(prompt, model)*
    call per check, its reply parsed as coder.parse_coder_output parses the
    coder's. An item that is no verdict is dropped; a reply that is no
    judgment raises FitReplyError, which check() records as unchecked."""

    def fit_fn(fit_check: FitCheck, *, model: str) -> tuple[FitVerdict, ...]:
        return _parse_reply(invoke(_fit_prompt(fit_check), model))

    return fit_fn


def _sessions(sessions: Sequence[str]) -> str:
    return ", ".join(sessions) or "none"


def report_lines(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The report's ``## Sighting Fit`` section, rendered from the run
    record's ``sighting_fit`` value (FitOutcome.to_record per check)."""
    lines = [
        "",
        "## Sighting Fit",
        "",
        f"- {len(records)} withheld entr(ies) could reach "
        f"{filing_policy.MIN_UNREMEDIATED_SIGHTINGS} sightings only through attached "
        "sightings no rule placed. Each such sighting's quoted evidence was judged "
        "against the mechanism its entry's title names; only a confirmed one counts.",
    ]
    for record in records:
        lines.append(
            f"  - {report_text.quoted_title(record['title'])} ({record['entry_id']}): "
            f"confirmed {_sessions(record['confirmed'])}; refuted {_sessions(record['refuted'])}; "
            f"unchecked {_sessions(record['unchecked'])}"
        )
        if record["error"]:
            lines.append(f"    - the fit check failed, so its sightings stay unchecked: {record['error']}")
    return lines
