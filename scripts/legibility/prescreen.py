"""Decides whether an already-catalogued finding absorbs a novel census
cluster, so the census spends no verify call on what it already knows
(plans/census-incremental-prd.md L6)."""
from __future__ import annotations

import copy
import dataclasses
import enum
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legibility import census_findings, codebook, finding_area, verdict


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


FIXED_STATUSES = frozenset({"fixed", "retired"})
"""Entry statuses a re-observation contradicts: fixed by a task (L5), or
retired because the verifier reported it fixed (``census.py::retire_entry``).
No rule attaches to such an entry; a cluster re-observing one is surfaced
(plans/census-incremental-prd.md L5, consumers)."""


def is_fixed(entry: Mapping[str, Any]) -> bool:
    return entry.get("status") in FIXED_STATUSES


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


def sessions_of(record: Mapping[str, Any]) -> list[str]:
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
        fixed = {entry.get("id") for entry in entries if is_fixed(entry)}

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
            for session in dict.fromkeys(sessions_of(record)):
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
    """A cluster re-observing a fixed entry, by *rule*'s match: verified, never
    attached to that entry, and reported."""

    rule: Rule
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
        sessions = sessions_of(cluster)
        session = sessions[0] if sessions else None
        if (attach := catalogue.attachable(title)) is not None:
            screened.append(Screened(Rule.TITLE, title, session, attach.matched))
        elif (match := catalogue.title_match(title)) is not None:
            surfaced.append(Surfaced(Rule.TITLE, title, match.fixed_entry_ids))
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


def _entries_at(
    cb: Mapping[str, Any], identity: census_findings.FindingIdentity,
) -> list[tuple[dict, census_findings.FindingIdentity]]:
    """Every entry whose finding identity stands at *identity*'s area and
    anchor, whatever its primary tag (plans/census-incremental-prd.md T2),
    with that standing identity."""
    location = (identity.area, identity.anchor)
    return [
        (entry, standing) for entry in cb.get("entries") or []
        if (standing := census_findings.FindingIdentity.from_record(entry)) is not None
        and (standing.area, standing.anchor) == location
    ]


def _joinable(
    cb: Mapping[str, Any], identity: census_findings.FindingIdentity,
) -> tuple[dict, census_findings.FindingIdentity] | None:
    """The first entry at *identity*'s location that is not fixed: the one a
    verdict there joins."""
    return next(((entry, standing) for entry, standing in _entries_at(cb, identity) if not is_fixed(entry)), None)


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
    cb: dict, candidate_id: str, entry_id: str, found: verdict.Verdict, *, run_id: str,
) -> tuple[dict, KeyAttach]:
    """A new codebook in which pending *candidate_id* is promoted onto entry
    *entry_id* instead of to an entry of its own, and the attach it made.

    The candidate takes the entry's standing identity; each of its sightings
    from a session the entry does not carry is appended to the entry, marked
    with the attach and the verifier's tags."""
    result = copy.deepcopy(cb)
    candidate = _record_by_id(result.get("candidates") or [], candidate_id, kind="candidate")
    target = _record_by_id(result.get("entries") or [], entry_id, kind="entry")
    identity = census_findings.FindingIdentity.from_record(target)
    if identity is None:
        raise ValueError(f"attach_by_key: entry {entry_id!r} carries no finding identity")
    attached_by = _attached_by(Rule.KEY, run_id)

    candidate.update(census_findings.stamped(candidate, identity))
    candidate.update(disposition="promoted", promoted_to=target["id"], attached_by=attached_by)
    sightings = target.setdefault("sightings", [])
    carried = set(sessions_of(target))
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


# ---------------------------------------------------------------------------
# Other instruments' findings (docs/quality-findings-contract.md §5, §9)
# ---------------------------------------------------------------------------

_SHAPE_PLACEHOLDERS = {
    "<YYYYMMDD>": r"(?P<date>\d{8})",
    "<YYYY-MM-DD>": r"(?P<date>\d{4}-\d{2}-\d{2})",
    "[-<n>]": r"(?:-(?P<n>\d+))?",
}
_SHAPE_PLACEHOLDER_RE = re.compile("|".join(re.escape(placeholder) for placeholder in _SHAPE_PLACEHOLDERS))


def _basename_regex(shape: str) -> re.Pattern[str]:
    parts, position = [], 0
    for placeholder in _SHAPE_PLACEHOLDER_RE.finditer(shape):
        parts += [re.escape(shape[position:placeholder.start()]), _SHAPE_PLACEHOLDERS[placeholder.group()]]
        position = placeholder.end()
    return re.compile("".join([*parts, re.escape(shape[position:])]))


@dataclass(frozen=True)
class InstrumentHome:
    """Where an instrument writes its run records, and their basename shape
    as its report-format names it: a date and an optional ``-<n>``."""

    name: str
    directory: str
    shape: str

    def newest(self, project_root: Path) -> Path | None:
        """The record of the latest run: the greatest (date, n), read as
        numbers, since a bare basename sorts after its own ``-2`` as text."""
        pattern = _basename_regex(self.shape)
        directory = project_root / self.directory
        runs = []
        if directory.is_dir():
            for path in directory.iterdir():
                found = pattern.fullmatch(path.name)
                if found is not None and path.is_file():
                    order = (int(found["date"].replace("-", "")), int(found["n"] or 1))
                    runs.append((order, path))
        return max(runs)[1] if runs else None

    def where(self) -> str:
        return f"{self.directory}/{self.shape}"


def instrument_homes(project_id: str) -> tuple[InstrumentHome, ...]:
    return (
        InstrumentHome("/review", "review/reports", f"review-{project_id}-<YYYYMMDD>[-<n>].json"),
        InstrumentHome("/hotspot-survey", "plans", "bug-hotspot-survey-<YYYY-MM-DD>[-<n>]-full-findings.json"),
        InstrumentHome("/review-all", "plans", f"review-all-{project_id}-<YYYYMMDD>[-<n>].json"),
    )


@dataclass(frozen=True)
class ExternalFinding:
    """A finding another instrument's newest record carries."""

    key: str
    area: str
    anchor: str
    first_seen: str | None
    supersedes: tuple[str, ...]
    run_id: str

    @classmethod
    def of(cls, finding: object, *, run_id: str) -> ExternalFinding | None:
        """The finding, or None when it has no valid key, area and anchor."""
        if not isinstance(finding, Mapping):
            return None
        key, area, anchor = finding.get("key"), finding.get("area"), finding.get("anchor")
        if not (
            isinstance(key, str) and codebook.is_finding_key(key)
            and isinstance(area, str) and re.fullmatch(finding_area.AREA_KEY_PATTERN, area)
            and isinstance(anchor, str) and verdict.is_anchor(anchor)
        ):
            return None
        first_seen = finding.get("first_seen")
        supersedes = finding.get("supersedes")
        return cls(
            key=key,
            area=area,
            anchor=anchor,
            first_seen=first_seen if isinstance(first_seen, str) and first_seen else None,
            supersedes=tuple(
                old for old in supersedes if isinstance(old, str) and codebook.is_finding_key(old)
            ) if isinstance(supersedes, list) else (),
            run_id=run_id,
        )


class AdoptedBy(enum.StrEnum):
    KEY = "key"
    LOCATION = "location"


@dataclass(frozen=True)
class Adoption:
    """A census promotion re-observing *external*: it takes the external key
    in place of its minted one, and the external supersedes and first_seen."""

    identity: census_findings.FindingIdentity
    external: ExternalFinding
    matched_by: AdoptedBy

    @property
    def supersedes(self) -> tuple[str, ...]:
        return self.external.supersedes

    @property
    def first_seen(self) -> str | None:
        return self.external.first_seen


@dataclass(frozen=True)
class InstrumentFindings:
    """What the census read of the other instruments: the run_id of every
    record it consumed, why any instrument gave it none, and their findings."""

    consumed: tuple[str, ...]
    note: str | None
    findings: tuple[ExternalFinding, ...]

    def match(self, identity: census_findings.FindingIdentity) -> Adoption | None:
        """The finding *identity* re-observes: an equal key first, else an
        equal (area, anchor)."""
        location = (identity.area, identity.anchor)
        by_key = next((found for found in self.findings if found.key == identity.key), None)
        by_location = next((found for found in self.findings if (found.area, found.anchor) == location), None)
        external, matched_by = (by_key, AdoptedBy.KEY) if by_key is not None else (by_location, AdoptedBy.LOCATION)
        if external is None:
            return None
        return Adoption(
            identity=dataclasses.replace(identity, key=external.key),
            external=external,
            matched_by=matched_by,
        )


@dataclass(frozen=True)
class _Read:
    run_id: str | None
    findings: tuple[ExternalFinding, ...]
    note: str | None


def _read_record(home: InstrumentHome, project_root: Path) -> _Read:
    path = home.newest(project_root)
    if path is None:
        return _Read(None, (), f"{home.name}: no record matching {home.where()}")
    where = path.relative_to(project_root).as_posix()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _Read(None, (), f"{home.name}: {where} is unreadable: {exc}")
    method = record.get("method") if isinstance(record, dict) else None
    run_id = method.get("run_id") if isinstance(method, dict) else None
    if not isinstance(run_id, str) or not run_id:
        return _Read(None, (), f"{home.name}: {where} is not a contract record: it has no method.run_id")
    raw = record.get("findings")
    if not isinstance(raw, list):
        return _Read(None, (), f"{home.name}: {where} is not a contract record: its findings is not a list")
    findings = tuple(found for item in raw if (found := ExternalFinding.of(item, run_id=run_id)) is not None)
    skipped = len(raw) - len(findings)
    note = f"{home.name}: skipped {skipped} finding(s) in {where} with no valid key, area or anchor" if skipped else None
    return _Read(run_id, findings, note)


def load_instrument_findings(project_root: str | Path, project_id: str) -> InstrumentFindings:
    """Read each other instrument's newest record under *project_root*.

    Never raises: a missing, unreadable or non-contract record, or a finding
    with no usable identity, becomes a line of the note instead."""
    reads = [_read_record(home, Path(project_root)) for home in instrument_homes(project_id)]
    notes = [read.note for read in reads if read.note is not None]
    return InstrumentFindings(
        consumed=tuple(read.run_id for read in reads if read.run_id is not None),
        note="; ".join(notes) if notes else None,
        findings=tuple(found for read in reads for found in read.findings),
    )


# ---------------------------------------------------------------------------
# Where a verified cluster enters the codebook: the key and cross-instrument
# rules, decided once per cluster
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """Where one verified cluster's pending candidate enters the codebook.

    With *standing_entry_id* it joins that entry (:func:`attach_by_key`);
    otherwise it is promoted to an entry of its own, adopting *adoption*'s
    key when another instrument already keys its location. *identity* is the
    one it files under either way. *surfaced* names the fixed entries at its
    location that it re-observes rather than joins."""

    identity: census_findings.FindingIdentity
    standing_entry_id: str | None
    adoption: Adoption | None
    surfaced: Surfaced | None

    @property
    def supersedes(self) -> tuple[str, ...]:
        return () if self.adoption is None else self.adoption.supersedes

    def first_seen_run(self, run_id: str) -> str:
        """The run a new entry was first seen in: the adopted finding's, else
        *run_id*."""
        return (None if self.adoption is None else self.adoption.first_seen) or run_id


def place(
    cb: Mapping[str, Any], title: str, identity: census_findings.FindingIdentity, inputs: InstrumentFindings,
) -> Placement:
    """Where the verified cluster *title*, at minted *identity*, enters *cb*:
    onto the entry standing at its location, which keeps its key (the key
    rule), else as a new entry under the key of the finding in *inputs* it
    re-observes (the cross-instrument rule), else under *identity*."""
    joinable = _joinable(cb, identity)
    if joinable is not None:
        entry, standing = joinable
        return Placement(standing, entry["id"], None, None)
    fixed_entry_ids = tuple(entry["id"] for entry, _ in _entries_at(cb, identity) if is_fixed(entry))
    adoption = inputs.match(identity)
    return Placement(
        identity=identity if adoption is None else adoption.identity,
        standing_entry_id=None,
        adoption=adoption,
        surfaced=Surfaced(Rule.KEY, title, fixed_entry_ids) if fixed_entry_ids else None,
    )


# ---------------------------------------------------------------------------
# The run's ## Screened section
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReObserved:
    """A cluster promoted under another instrument's finding key."""

    title: str
    entry_id: str
    adoption: Adoption


@dataclass(frozen=True)
class ScreenReport:
    screen: Screen
    surfaced_by_key: tuple[Surfaced, ...]
    attached: tuple[KeyAttach, ...]
    re_observed: tuple[ReObserved, ...]
    inputs: InstrumentFindings

    def counts(self) -> dict[str, int]:
        screened = [item.rule for item in self.screen.screened]
        return {
            Rule.TITLE.value: screened.count(Rule.TITLE),
            Rule.SESSION.value: screened.count(Rule.SESSION),
            Rule.KEY.value: len(self.attached),
            Rule.CROSS_INSTRUMENT.value: len(self.re_observed),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "inputs": {"consumed": list(self.inputs.consumed), "note": self.inputs.note},
            "screened": [
                {"rule": item.rule.value, "title": item.title, "session": item.session, "matched": list(item.matched)}
                for item in self.screen.screened
            ],
            "surfaced": [
                {"rule": item.rule.value, "title": item.title, "fixed_entry_ids": list(item.fixed_entry_ids)}
                for item in (*self.screen.surfaced, *self.surfaced_by_key)
            ],
            "attached": [
                {
                    "title": item.title,
                    "candidate_id": item.candidate_id,
                    "entry_id": item.entry_id,
                    "key": item.identity.key,
                    "verifier_tags": list(item.verifier_tags),
                    "tag_disagreement": None if item.tag_disagreement is None else list(item.tag_disagreement),
                }
                for item in self.attached
            ],
            "re_observed": [
                {
                    "title": item.title,
                    "entry_id": item.entry_id,
                    "key": item.adoption.identity.key,
                    "run_id": item.adoption.external.run_id,
                    "matched_by": item.adoption.matched_by.value,
                    "supersedes": list(item.adoption.supersedes),
                    "first_seen": item.adoption.first_seen,
                }
                for item in self.re_observed
            ],
        }


def _title(title: str) -> str:
    return '"' + " ".join(title.split()) + '"'


def _attached_line(item: Mapping[str, Any]) -> str:
    line = f"- attached by key: {_title(item['title'])} ({item['candidate_id']}) → {item['entry_id']} under {item['key']}"
    if item["tag_disagreement"] is None:
        return line
    standing, offered = item["tag_disagreement"]
    return f"{line}; primary tag disagreement: standing {standing}, verifier {offered}"


def report_lines(record: Mapping[str, Any]) -> list[str]:
    """The report's ``## Screened`` section, rendered from the run record's
    ``screened`` value (:meth:`ScreenReport.to_record`)."""
    counts = ", ".join(f"{rule} {count}" for rule, count in record["counts"].items())
    inputs = record["inputs"]
    consumed = ", ".join(inputs["consumed"]) or "none"
    return [
        "",
        "## Screened",
        "",
        f"- per rule: {counts}",
        f"- inputs consumed: {consumed}",
        *([f"- inputs note: {inputs['note']}"] if inputs["note"] else []),
        *(
            f"- screened by {item['rule']}: {_title(item['title'])} → {', '.join(item['matched'])}"
            for item in record["screened"]
        ),
        *(
            f"- surfaced by {item['rule']}, verified (re-observes fixed {', '.join(item['fixed_entry_ids'])}): "
            f"{_title(item['title'])}"
            for item in record["surfaced"]
        ),
        *(_attached_line(item) for item in record["attached"]),
        *(
            f"- re-observed from {item['run_id']} by {item['matched_by']}: {_title(item['title'])} → "
            f"{item['entry_id']} under {item['key']}"
            for item in record["re_observed"]
        ),
    ]
