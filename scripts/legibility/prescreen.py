"""Decides whether an already-catalogued finding absorbs a novel census
cluster, so the census spends no verify call on what it already knows
(plans/census-incremental-prd.md L6)."""
from __future__ import annotations

import copy
import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from legibility import census_findings, codebook, verdict


class Rule(enum.StrEnum):
    """The pre-screen's rules. A rule that attaches a sighting writes its
    value as that sighting's ``attached_by.method``."""

    TITLE = "title"
    SESSION = "session"
    KEY = "key"
    CROSS_INSTRUMENT = "cross_instrument"


def normalised_title(title: str) -> str:
    return " ".join(title.split()).casefold()


def _attached_by(rule: Rule, run_id: str) -> dict[str, str]:
    return {"method": rule.value, "run_id": run_id}


@dataclass(frozen=True)
class TitleMatch:
    """The catalogued records a title matches once normalised.

    ``title`` is the exact title the merger routes by: the preferred
    candidate's, or, when no candidate carries the title, the entry's, whose
    id is then ``entry_id``."""

    title: str
    matched: tuple[str, ...]
    entry_id: str | None
    fixed_entry_ids: tuple[str, ...]

    @property
    def surfaced(self) -> bool:
        """A fixed entry re-observed is verified, never screened away."""
        return bool(self.fixed_entry_ids)


@dataclass(frozen=True)
class CataloguedTitle:
    record_id: str
    title: str
    fixed_entry_id: str | None


def _preferred(records: Sequence[CataloguedTitle], title: str) -> CataloguedTitle:
    return next((record for record in records if record.title == title), records[0])


def _sessions_of(record: Mapping[str, Any]) -> list[str]:
    return [
        sighting["session"]
        for sighting in record.get("sightings") or []
        if isinstance(sighting, Mapping) and isinstance(sighting.get("session"), str)
    ]


@dataclass(frozen=True)
class Catalogue:
    """The codebook as the pre-screen reads it before this run's merge: every
    entry's and candidate's normalised title, and every sighted session."""

    candidate_titles: Mapping[str, tuple[CataloguedTitle, ...]]
    entry_titles: Mapping[str, tuple[CataloguedTitle, ...]]
    sessions: Mapping[str, tuple[str, ...]]

    @classmethod
    def of(cls, cb: Mapping[str, Any]) -> Catalogue:
        entries = [entry for entry in cb.get("entries") or [] if isinstance(entry, Mapping)]
        candidates = [candidate for candidate in cb.get("candidates") or [] if isinstance(candidate, Mapping)]
        fixed = {entry.get("id") for entry in entries if entry.get("status") == "fixed"}

        def fixed_entry_of(record: Mapping[str, Any], *, is_entry: bool) -> str | None:
            if is_entry:
                return record.get("id") if record.get("id") in fixed else None
            if record.get("disposition") == "promoted" and record.get("promoted_to") in fixed:
                return record.get("promoted_to")
            return None

        def index(records: Iterable[Mapping[str, Any]], *, is_entry: bool) -> dict[str, tuple[CataloguedTitle, ...]]:
            titles: dict[str, list[CataloguedTitle]] = {}
            for record in records:
                title, record_id = record.get("title"), record.get("id")
                if isinstance(title, str) and title and isinstance(record_id, str):
                    titled = CataloguedTitle(record_id, title, fixed_entry_of(record, is_entry=is_entry))
                    titles.setdefault(normalised_title(title), []).append(titled)
            return {key: tuple(value) for key, value in titles.items()}

        sessions: dict[str, list[str]] = {}
        for record in (*candidates, *entries):
            for session in dict.fromkeys(_sessions_of(record)):
                sessions.setdefault(session, []).append(record["id"])
        return cls(
            candidate_titles=index(candidates, is_entry=False),
            entry_titles=index(entries, is_entry=True),
            sessions={session: tuple(ids) for session, ids in sessions.items()},
        )

    def title_match(self, title: str) -> TitleMatch | None:
        key = normalised_title(title)
        candidates = self.candidate_titles.get(key, ())
        entries = self.entry_titles.get(key, ())
        if not candidates and not entries:
            return None
        matched = (*candidates, *entries)
        routed = _preferred(candidates, title) if candidates else _preferred(entries, title)
        return TitleMatch(
            title=routed.title,
            matched=tuple(record.record_id for record in matched),
            entry_id=None if candidates else routed.record_id,
            fixed_entry_ids=tuple(
                dict.fromkeys(record.fixed_entry_id for record in matched if record.fixed_entry_id)
            ),
        )

    def attachable(self, title: str) -> TitleMatch | None:
        """The one title-rule decision: the match a cluster of *title*
        attaches through, or None when it is novel or surfaced."""
        match = self.title_match(title)
        return None if match is None or match.surfaced else match

    def sighted(self, sessions: Iterable[str]) -> tuple[str, ...]:
        """Every record already carrying a sighting from one of *sessions*."""
        return tuple(dict.fromkeys(
            record_id for session in sessions for record_id in self.sessions.get(session, ())
        ))


@dataclass(frozen=True)
class Screened:
    """A cluster no verify call is spent on, and the records that absorbed it."""

    rule: Rule
    title: str
    session: str | None
    matched: tuple[str, ...]


@dataclass(frozen=True)
class Surfaced:
    """A cluster re-observing a fixed entry: verified, and reported."""

    title: str
    fixed_entry_ids: tuple[str, ...]


@dataclass(frozen=True)
class Screen:
    to_verify: tuple[dict, ...]
    screened: tuple[Screened, ...]
    surfaced: tuple[Surfaced, ...]


def screen(clusters: Sequence[dict], catalogue: Catalogue) -> Screen:
    """Split novel *clusters* into those still to verify, in order, and those
    the title rule or the session rule screens."""
    to_verify: list[dict] = []
    screened: list[Screened] = []
    surfaced: list[Surfaced] = []
    for cluster in clusters:
        title = cluster["title"]
        sessions = _sessions_of(cluster)
        session = sessions[0] if sessions else None
        if (attach := catalogue.attachable(title)) is not None:
            screened.append(Screened(Rule.TITLE, title, session, attach.matched))
        elif (match := catalogue.title_match(title)) is not None:
            surfaced.append(Surfaced(title, match.fixed_entry_ids))
            to_verify.append(cluster)
        elif sighted := catalogue.sighted(sessions):
            screened.append(Screened(Rule.SESSION, title, session, sighted))
        else:
            to_verify.append(cluster)
    return Screen(tuple(to_verify), tuple(screened), tuple(surfaced))


_MATCH_FIELDS = ("origin_phase", "manifested_phase", "evidence_quote")


def routed_records(records: Sequence[dict], catalogue: Catalogue, *, run_id: str) -> list[dict]:
    """New copies of *records* in which every title-screened candidate payload
    attaches through the merger's existing routing: renamed to the matched
    candidate's exact title, or turned into a ``matches`` op on the one entry
    carrying the title. Each carries ``attached_by``; no other payload changes."""
    routed = []
    for record in records:
        record = copy.deepcopy(record)
        candidates, matches = [], list(record.get("matches") or [])
        for payload in record.get("candidates") or []:
            title = payload.get("title") if isinstance(payload, dict) else None
            attach = catalogue.attachable(title) if isinstance(title, str) and title else None
            if attach is None:
                candidates.append(payload)
            elif attach.entry_id is None:
                candidates.append({**payload, "title": attach.title, "attached_by": _attached_by(Rule.TITLE, run_id)})
            else:
                matches.append({
                    "entry_id": attach.entry_id,
                    **{field: payload[field] for field in _MATCH_FIELDS if field in payload},
                    "attached_by": _attached_by(Rule.TITLE, run_id),
                })
        if "candidates" in record:
            record["candidates"] = candidates
        if "matches" in record or matches:
            record["matches"] = matches
        routed.append(record)
    return routed


def keyed_entry(cb: Mapping[str, Any], identity: census_findings.FindingIdentity) -> dict | None:
    """The first entry whose finding identity stands at *identity*'s area and
    anchor, whatever its primary tag (plans/census-incremental-prd.md T2)."""
    for entry in cb.get("entries") or []:
        standing = census_findings.FindingIdentity.from_record(entry)
        if standing is not None and (standing.area, standing.anchor) == (identity.area, identity.anchor):
            return entry
    return None


@dataclass(frozen=True)
class KeyAttach:
    """A verified cluster's candidate promoted onto the entry already standing
    at its location, which keeps its key: *identity* is that entry's."""

    title: str
    candidate_id: str
    entry_id: str
    identity: census_findings.FindingIdentity
    verifier_tags: tuple[str, ...]

    @property
    def tag_disagreement(self) -> tuple[str, str] | None:
        """The standing and the verifier's primary tag, when they differ."""
        standing, offered = self.identity.tags[0], self.verifier_tags[0]
        return None if standing == offered else (standing, offered)


def _record_by_id(records: Iterable[dict], record_id: str, *, kind: str) -> dict:
    for record in records:
        if record.get("id") == record_id:
            return record
    raise ValueError(f"attach_by_key: no {kind} {record_id!r}")


def attach_by_key(
    cb: dict, candidate_id: str, entry: Mapping[str, Any], found: verdict.Verdict, *, run_id: str,
) -> tuple[dict, KeyAttach]:
    """A new codebook in which pending *candidate_id* is promoted onto
    *entry* instead of to an entry of its own, and the attach it made.

    The candidate takes the entry's standing identity; each of its sightings
    from a session the entry does not carry is appended to the entry, marked
    with the attach and the verifier's tags."""
    identity = census_findings.FindingIdentity.from_record(entry)
    if identity is None:
        raise ValueError(f"attach_by_key: entry {entry.get('id')!r} carries no finding identity")
    result = copy.deepcopy(cb)
    candidate = _record_by_id(result.get("candidates") or [], candidate_id, kind="candidate")
    target = _record_by_id(result.get("entries") or [], entry["id"], kind="entry")
    attached_by = _attached_by(Rule.KEY, run_id)

    candidate.update(census_findings.stamped(candidate, identity))
    candidate.update(disposition="promoted", promoted_to=target["id"], attached_by=attached_by)
    sightings = target.setdefault("sightings", [])
    carried = set(_sessions_of(target))
    for sighting in candidate.get("sightings") or []:
        if sighting.get("session") in carried:
            continue
        carried.add(sighting.get("session"))
        sightings.append({
            **copy.deepcopy(sighting), "attached_by": dict(attached_by), "verifier_tags": list(found.tags),
        })

    codebook.assert_no_deletion(cb, result)
    return result, KeyAttach(
        title=candidate.get("title") or "",
        candidate_id=candidate_id,
        entry_id=target["id"],
        identity=identity,
        verifier_tags=found.tags,
    )
