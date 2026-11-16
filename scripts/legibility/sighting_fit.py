"""Whether a sighting attached to a codebook entry exhibits the entry's defining
mechanism, and so counts toward the singleton filing gate
(plans/census-incremental-prd.md C1)."""
from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from typing import Any

from legibility import codebook, prescreen


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
