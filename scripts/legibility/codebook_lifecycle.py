"""The codebook's legal-transition table, and the census lifecycle transforms
built on it.

A candidate's disposition, and an entry's status and filing marker, change
only through :func:`transit`, which refuses a transition from a state its
:class:`Rule` does not list. The merger's ``corrections`` op is outside the
table: it carries an operator's adjudication in a coding record
(``codebook.py::_apply_coding_record``).
"""
from __future__ import annotations

import copy
import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from legibility import census_findings, codebook


class Transition(enum.StrEnum):
    PROMOTE = "promote"
    REJECT = "reject"
    RETIRE = "retire"
    ACCEPT = "accept"
    FIX = "fix"
    REOPEN = "reopen"
    WITHHOLD = "withhold"
    FILE = "file"


@dataclass(frozen=True)
class Rule:
    """A transition writes *target* into a *kind* record's *field*, from any
    of *sources*; a None source is the field's absence."""

    kind: str
    field: str
    sources: frozenset[str | None]
    target: str


_LIVE_STATUSES = frozenset(codebook.STATUSES) - {"retired"}

TABLE: Mapping[Transition, Rule] = MappingProxyType({
    Transition.PROMOTE: Rule("candidate", "disposition", frozenset({"pending"}), "promoted"),
    Transition.REJECT: Rule("candidate", "disposition", frozenset({"pending"}), "rejected"),
    Transition.RETIRE: Rule("entry", "status", _LIVE_STATUSES, "retired"),
    Transition.ACCEPT: Rule("entry", "status", _LIVE_STATUSES, "accepted"),
    Transition.FIX: Rule("entry", "status", _LIVE_STATUSES, "fixed"),
    Transition.REOPEN: Rule("entry", "status", frozenset({"accepted", "fixed"}), "open"),
    Transition.WITHHOLD: Rule(
        "entry", codebook.ENTRY_FILING_KEY, frozenset({None}), codebook.ENTRY_FILING_WITHHELD
    ),
    Transition.FILE: Rule(
        "entry",
        codebook.ENTRY_FILING_KEY,
        frozenset({codebook.ENTRY_FILING_WITHHELD}),
        codebook.ENTRY_FILING_FILED,
    ),
})


class IllegalTransition(ValueError):
    pass


def transit(record: dict, transition: Transition) -> None:
    """Move *record* along *transition* in place, inside a codebook its caller
    has already copied; raise IllegalTransition, writing nothing, from a state
    the transition's rule does not list."""
    rule = TABLE[transition]
    current = record.get(rule.field)
    if current not in rule.sources:
        legal = ", ".join(sorted(repr(source) for source in rule.sources))
        raise IllegalTransition(
            f"{transition}: {rule.kind} {record.get('id')!r} has {rule.field} {current!r}; "
            f"legal only from {legal}"
        )
    record[rule.field] = rule.target


def promote_candidate(
    cb: dict, cand_id: str, entry_fields: dict, *,
    finding: census_findings.FindingIdentity | None = None,
    supersedes: Sequence[str] = (),
    withheld: bool = False,
) -> dict:
    """A new codebook in which pending candidate *cand_id* is promoted to a new
    entry built from *entry_fields* and the candidate's own sightings.

    *entry_fields* carries every field the v2 entry schema requires except
    ``sightings``. The candidate stays, naming the entry in ``promoted_to``. A
    *finding* identity is stamped on the candidate, *supersedes* joins its
    chain, and the entry carries both (``census_findings.carried_identity``).
    A *withheld* entry is marked withheld by the singleton filing gate.
    """
    result = copy.deepcopy(cb)
    candidate = codebook.record_by_id(result.get("candidates") or [], cand_id, kind="candidate")
    transit(candidate, Transition.PROMOTE)
    if finding is not None:
        candidate.update(census_findings.stamped(candidate, finding))
    if supersedes:
        candidate["supersedes"] = list(dict.fromkeys([*candidate.get("supersedes", []), *supersedes]))

    entry = {**entry_fields, **census_findings.carried_identity(candidate)}
    entry["sightings"] = copy.deepcopy(candidate.get("sightings") or [])
    if withheld:
        transit(entry, Transition.WITHHOLD)
    result.setdefault("entries", []).append(entry)
    candidate["promoted_to"] = entry_fields["id"]

    codebook.assert_no_deletion(cb, result)
    return result


def reject_candidate(
    cb: dict, cand_id: str, *, finding: census_findings.FindingIdentity | None = None,
) -> dict:
    """A new codebook in which pending candidate *cand_id* is rejected, and
    stamped with *finding* when given. The candidate stays."""
    result = copy.deepcopy(cb)
    candidate = codebook.record_by_id(result.get("candidates") or [], cand_id, kind="candidate")
    transit(candidate, Transition.REJECT)
    if finding is not None:
        candidate.update(census_findings.stamped(candidate, finding))

    codebook.assert_no_deletion(cb, result)
    return result


def retire_entry(cb: dict, entry_id: str) -> dict:
    """A new codebook in which entry *entry_id* is retired. The entry stays."""
    result = copy.deepcopy(cb)
    entry = codebook.record_by_id(result.get("entries") or [], entry_id, kind="entry")
    transit(entry, Transition.RETIRE)

    codebook.assert_no_deletion(cb, result)
    return result


def mark_entry_filed(cb: dict, entry_id: str) -> dict:
    """A new codebook in which withheld entry *entry_id* is marked filed."""
    result = copy.deepcopy(cb)
    entry = codebook.record_by_id(result.get("entries") or [], entry_id, kind="entry")
    transit(entry, Transition.FILE)

    codebook.assert_no_deletion(cb, result)
    return result
