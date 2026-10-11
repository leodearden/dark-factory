"""The census's side of docs/quality-findings-contract.md §1: the finding
identity a Verdict gives a codebook record, and the §1 row that record reports
(plans/census-incremental-prd.md leaf L3).
"""
from __future__ import annotations

import copy
import enum
import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bound as codebook.py binds it: no module-level assignment precedes the
# imports below, so they need no E402 suppression.
if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from legibility import codebook, finding_area, verdict
from shared.finding_key import finding_key


def _raise_on(errors: list[codebook.FindingIdentityError], location: str) -> None:
    if errors:
        raise ValueError("; ".join(error.at(location) for error in errors))


@dataclass(frozen=True)
class FindingIdentity:
    """A record's finding identity, held to ``codebook.finding_identity_errors``."""

    key: str
    area: str
    anchor: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tags, tuple):
            raise ValueError(f"FindingIdentity/tags must be a tuple, got {self.tags!r}")
        _raise_on(codebook.finding_identity_errors(self._fields()), "FindingIdentity")

    @classmethod
    def mint(cls, area: str, anchor: str, tags: tuple[str, ...]) -> FindingIdentity:
        return cls(key=finding_key(area, anchor, tags[0]), area=area, anchor=anchor, tags=tuple(tags))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FindingIdentity | None:
        """The identity *record* carries, or None when it carries none.

        Raises ValueError on a partial or malformed identity, which
        ``codebook.validate`` reports for the same record."""
        if not any(field in record for field in codebook.FINDING_IDENTITY_FIELDS):
            return None
        _raise_on(codebook.finding_identity_errors(record), f"record {record.get('id')!r}")
        return cls(
            key=record["finding_key"],
            area=record["finding_area"],
            anchor=record["finding_anchor"],
            tags=tuple(record["finding_tags"]),
        )

    def _fields(self) -> dict[str, Any]:
        values = (self.key, self.area, self.anchor, self.tags)
        return dict(zip(codebook.FINDING_IDENTITY_FIELDS, values, strict=True))

    def to_fields(self) -> dict[str, Any]:
        return {**self._fields(), "finding_tags": list(self.tags)}


def identity_of(found: verdict.Verdict, area_map: finding_area.AreaMap) -> FindingIdentity:
    return FindingIdentity.mint(area_map.area_of(found.anchor), found.anchor, found.tags)


def stamped(record: Mapping[str, Any], identity: FindingIdentity) -> dict[str, Any]:
    """*record* carrying *identity*, by contract §2's supersedes rule.

    An unkeyed record gains the identity. A record keyed at the same area and
    anchor keeps its standing key, whatever the primary tag. A moved anchor
    takes the new key, and the old one is recorded under ``supersedes``.
    """
    standing = FindingIdentity.from_record(record)
    if standing is None:
        return {**record, **identity.to_fields()}
    if (standing.area, standing.anchor) == (identity.area, identity.anchor) or standing.key == identity.key:
        return dict(record)
    supersedes = [*record.get("supersedes", []), standing.key]
    return {**record, **identity.to_fields(), "supersedes": supersedes}


def carried_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """The identity fields and supersedes chain a record promoted from *record*
    carries, so :func:`stamped` alone decides the key of both."""
    identity = FindingIdentity.from_record(record)
    if identity is None:
        return {}
    return {**identity.to_fields(), "supersedes": list(record.get("supersedes") or [])}


def first_sighting_fields(run_id: str) -> dict[str, Any]:
    return {"first_seen_run": run_id, "last_seen_runs": [run_id]}


def _sighted(entry: Mapping[str, Any], sessions: Collection[str]) -> bool:
    return any(sighting.get("session") in sessions for sighting in entry.get("sightings") or [])


def mark_seen(cb: dict, *, run_id: str, sessions: Collection[str]) -> tuple[dict, tuple[str, ...]]:
    """Append *run_id* to ``last_seen_runs`` on every entry sighted in *sessions*.

    Returns a new codebook and the sighted entry ids; *cb* is never mutated.
    """
    result = copy.deepcopy(cb)
    sighted: list[str] = []
    for entry in result.get("entries") or []:
        if not _sighted(entry, sessions):
            continue
        sighted.append(entry["id"])
        runs = entry.setdefault("last_seen_runs", [])
        if run_id not in runs:
            runs.append(run_id)
    codebook.assert_no_deletion(cb, result)
    return result, tuple(sighted)


FINDING_ROW_FIELDS = (
    "id",
    "key",
    "area",
    "sub_area",
    "anchor",
    "tags",
    "severity",
    "evidence_source",
    "statement",
    "proposal",
    "verdict",
    "disposition",
    "first_seen",
    "last_seen",
    "supersedes",
)
"""The keys of :meth:`FindingRow.to_record`, in order: the display id, then contract §1's fields."""


class FindingVerdict(enum.StrEnum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class FindingRow:
    record_id: str
    identity: FindingIdentity
    severity: str
    statement: str
    proposal: str | None
    verdict: FindingVerdict
    first_seen: str | None
    last_seen: tuple[str, ...]
    supersedes: tuple[str, ...]

    @property
    def evidence_source(self) -> list[str]:
        if self.verdict is FindingVerdict.CONFIRMED:
            return ["agent-transcripts", "present-tree"]
        return ["agent-transcripts"]

    @property
    def disposition(self) -> str:
        return "refuted" if self.verdict is FindingVerdict.REFUTED else "open"

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "key": self.identity.key,
            "area": self.identity.area,
            "sub_area": None,
            "anchor": self.identity.anchor,
            "tags": list(self.identity.tags),
            "severity": self.severity,
            "evidence_source": self.evidence_source,
            "statement": self.statement,
            "proposal": self.proposal,
            "verdict": self.verdict.value,
            "disposition": self.disposition,
            "first_seen": self.first_seen,
            "last_seen": list(self.last_seen),
            "supersedes": list(self.supersedes),
        }


def _keyed(record: Mapping[str, Any]) -> FindingIdentity:
    identity = FindingIdentity.from_record(record)
    if identity is None:
        raise ValueError(f"record {record.get('id')!r} carries no finding_key; it has no §1 row")
    return identity


def _statement(record: Mapping[str, Any], cause: str | None, reason: str | None) -> str:
    statement = cause or record.get("title") or ""
    return f"{statement} Verifier: {reason}" if reason else statement


def _entry_row(
    entry: Mapping[str, Any], *, statement: str, proposal: str | None, finding_verdict: FindingVerdict,
) -> FindingRow:
    return FindingRow(
        record_id=entry["id"],
        identity=_keyed(entry),
        severity=entry["severity"],
        statement=statement,
        proposal=proposal,
        verdict=finding_verdict,
        first_seen=entry.get("first_seen_run"),
        last_seen=tuple(entry.get("last_seen_runs") or ()),
        supersedes=tuple(entry.get("supersedes") or ()),
    )


def confirmed_row(entry: Mapping[str, Any], found: verdict.Verdict, cause: str | None) -> FindingRow:
    return _entry_row(
        entry,
        statement=_statement(entry, cause, found.reason),
        proposal=None if found.remediation is None else found.remediation.change,
        finding_verdict=FindingVerdict.CONFIRMED,
    )


def observed_row(entry: Mapping[str, Any], cause: str | None) -> FindingRow:
    return _entry_row(
        entry,
        statement=_statement(entry, cause, None),
        proposal=None,
        finding_verdict=FindingVerdict.UNVERIFIED,
    )


def refuted_row(
    candidate: Mapping[str, Any], found: verdict.Verdict, cause: str | None, *, run_id: str,
) -> FindingRow:
    return FindingRow(
        record_id=candidate["id"],
        identity=_keyed(candidate),
        severity=found.severity,
        statement=_statement(candidate, cause, found.reason),
        proposal=None,
        verdict=FindingVerdict.REFUTED,
        first_seen=run_id,
        last_seen=(run_id,),
        supersedes=(),
    )
