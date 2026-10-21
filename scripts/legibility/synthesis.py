"""The census synthesis (plans/census-incremental-prd.md C6, leaf L10): the
brief the model is given, the one validator of the document it returns, that
document's application to the run, and the report's ``## Synthesis`` section
rendered from the run record.

The brief holds only what this run observed: the keyed entries it created,
and the keyed entries that gained a session. Standing findings with no new
evidence are counted by the runner and never sent to the model.
"""
from __future__ import annotations

import reprlib
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from legibility import census_findings, codebook, coder, verdict


@dataclass(frozen=True)
class Judgement:
    """This run's verdict on a finding, as the brief states it."""

    severity: str
    severity_reason: str
    reason: str
    proposal: str | None
    route: str

    @classmethod
    def of(cls, found: verdict.Verdict) -> Judgement:
        return cls(
            severity=found.severity,
            severity_reason=found.severity_reason,
            reason=found.reason,
            proposal=None if found.remediation is None else found.remediation.change,
            route=found.route.value,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "severity_reason": self.severity_reason,
            "reason": self.reason,
            "proposal": self.proposal,
            "route": self.route,
        }


def _judgement_record(judgement: Judgement | None) -> dict[str, Any] | None:
    return None if judgement is None else judgement.to_record()


@dataclass(frozen=True)
class Standing:
    """A re-observed entry's severity, status and phases before this run."""

    severity: str
    status: str
    origin_phase: str
    manifested_phase: str

    @classmethod
    def of(cls, entry: Mapping[str, Any]) -> Standing:
        return cls(entry["severity"], entry["status"], entry["origin_phase"], entry["manifested_phase"])

    def to_record(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "status": self.status,
            "origin_phase": self.origin_phase,
            "manifested_phase": self.manifested_phase,
        }


@dataclass(frozen=True)
class NewFinding:
    """A keyed entry this run created under its own key."""

    key: str
    entry_id: str
    title: str
    cause: str | None
    area: str
    anchor: str
    tags: tuple[str, ...]
    origin_phase: str
    manifested_phase: str
    sessions: tuple[str, ...]
    verdict: Judgement | None

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "entry_id": self.entry_id,
            "title": self.title,
            "cause": self.cause,
            "area": self.area,
            "anchor": self.anchor,
            "tags": list(self.tags),
            "origin_phase": self.origin_phase,
            "manifested_phase": self.manifested_phase,
            "sessions": list(self.sessions),
            "verdict": _judgement_record(self.verdict),
        }


@dataclass(frozen=True)
class ReObservedFinding:
    """A keyed entry that gained a session this run: one that stood before
    it, or one promoted under the key of another instrument's finding,
    *adopted_from* that instrument's run."""

    key: str
    entry_id: str
    title: str
    cause: str | None
    gained_sessions: tuple[str, ...]
    standing: Standing
    verdict: Judgement | None
    adopted_from: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "entry_id": self.entry_id,
            "title": self.title,
            "cause": self.cause,
            "gained_sessions": list(self.gained_sessions),
            "standing": self.standing.to_record(),
            "verdict": _judgement_record(self.verdict),
            "adopted_from": self.adopted_from,
        }


def _by_id(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {record["id"]: record for record in records}


def _require_entries(entries: Mapping[str, Any], named: Iterable[str], *, what: str) -> None:
    unknown = sorted(set(named) - set(entries))
    if unknown:
        raise ValueError(f"synthesis.Brief.of: {what} names no entry of the run's codebook: {unknown}")


def _gained_sessions(standing: Mapping[str, Any] | None, entry: Mapping[str, Any]) -> tuple[str, ...]:
    seen = set(codebook.sessions_of(standing or {}))
    return tuple(dict.fromkeys(session for session in codebook.sessions_of(entry) if session not in seen))


def _invariant_counts(cb: Mapping[str, Any], run_sessions: Collection[str]) -> tuple[tuple[str, int], ...]:
    pairs = {
        (sighting["invariant_violated"], sighting["session"])
        for record in (*(cb.get("entries") or []), *(cb.get("candidates") or []))
        for sighting in record.get("sightings") or []
        if sighting.get("session") in run_sessions
        and isinstance(sighting.get("invariant_violated"), str)
        and sighting["invariant_violated"]
    }
    return tuple(sorted(Counter(slug for slug, _ in pairs).items()))


def _run_counts(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    for name, value in counts.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"synthesis.Brief.of: counts must map str to int, got {name!r}: {value!r}")
    return tuple(counts.items())


@dataclass(frozen=True)
class Brief:
    """What the synthesis model is given about one run, and nothing else."""

    run_id: str
    new: tuple[NewFinding, ...]
    re_observed: tuple[ReObservedFinding, ...]
    unkeyed_re_observed: int
    invariant_counts: tuple[tuple[str, int], ...]
    counts: tuple[tuple[str, int], ...]

    @classmethod
    def of(
        cls,
        *,
        run_id: str,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        verdicts: Sequence[census_findings.Adjudication],
        adopted: Mapping[str, str],
        run_sessions: Collection[str],
        counts: Mapping[str, int],
    ) -> Brief:
        """The brief of the run that turned codebook *before* into *after*.

        *verdicts* are the run's promotions, each naming the entry it created
        or key-attached to; *adopted* maps each entry promoted under another
        instrument's key to that instrument's run; *run_sessions* are the
        sessions this run mined."""
        standing_entries = _by_id(before.get("entries") or [])
        entries = _by_id(after.get("entries") or [])
        judgements = {adjudication.record_id: Judgement.of(adjudication.found) for adjudication in verdicts}
        _require_entries(entries, judgements, what="a verdict")
        _require_entries(entries, adopted, what="an adoption")
        new: list[NewFinding] = []
        re_observed: list[ReObservedFinding] = []
        unkeyed = 0
        for entry_id, entry in entries.items():
            standing = standing_entries.get(entry_id)
            identity = census_findings.FindingIdentity.from_record(entry)
            gained = _gained_sessions(standing, entry)
            if identity is not None and standing is None and entry_id not in adopted:
                new.append(NewFinding(
                    key=identity.key,
                    entry_id=entry_id,
                    title=entry["title"],
                    cause=codebook.entry_field(after, entry, "cause"),
                    area=identity.area,
                    anchor=identity.anchor,
                    tags=identity.tags,
                    origin_phase=entry["origin_phase"],
                    manifested_phase=entry["manifested_phase"],
                    sessions=gained,
                    verdict=judgements.get(entry_id),
                ))
            elif gained and identity is None:
                unkeyed += 1
            elif gained and identity is not None:
                re_observed.append(ReObservedFinding(
                    key=identity.key,
                    entry_id=entry_id,
                    title=entry["title"],
                    cause=codebook.entry_field(after, entry, "cause"),
                    gained_sessions=gained,
                    standing=Standing.of(standing or entry),
                    verdict=judgements.get(entry_id),
                    adopted_from=adopted.get(entry_id),
                ))
        return cls(
            run_id=run_id,
            new=tuple(new),
            re_observed=tuple(re_observed),
            unkeyed_re_observed=unkeyed,
            invariant_counts=_invariant_counts(after, run_sessions),
            counts=_run_counts(counts),
        )

    @property
    def is_empty(self) -> bool:
        return not self.new and not self.re_observed

    @property
    def correctable_entry_ids(self) -> frozenset[str]:
        """The entries a correction may name: re-observed ones that stood
        before this run."""
        return frozenset(finding.entry_id for finding in self.re_observed if finding.adopted_from is None)

    @property
    def refinable_keys(self) -> frozenset[str]:
        """The keys a phase refinement may name: the findings judged this run."""
        return frozenset(finding.key for finding in (*self.new, *self.re_observed) if finding.verdict is not None)

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "new": [finding.to_record() for finding in self.new],
            "re_observed": [finding.to_record() for finding in self.re_observed],
            "unkeyed_re_observed": self.unkeyed_re_observed,
            "invariant_counts": dict(self.invariant_counts),
            "counts": dict(self.counts),
        }


# ---------------------------------------------------------------------------
# The C6 document and its one validator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewObservation:
    key: str
    observation: str

    def to_record(self) -> dict[str, str]:
        return {"key": self.key, "observation": self.observation}


@dataclass(frozen=True)
class Change:
    """What this run's evidence changes about a re-observed finding."""

    field: str
    from_value: str
    to_value: str
    why: str

    def to_record(self) -> dict[str, str]:
        return {"field": self.field, "from": self.from_value, "to": self.to_value, "why": self.why}


@dataclass(frozen=True)
class ReObservation:
    key: str
    sessions: tuple[str, ...]
    changed: tuple[Change, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sessions": list(self.sessions),
            "changed": [change.to_record() for change in self.changed],
        }


@dataclass(frozen=True)
class PhaseRefinement:
    key: str
    origin_phase: str
    manifested_phase: str
    why: str

    def to_record(self) -> dict[str, str]:
        return {
            "key": self.key,
            "origin_phase": self.origin_phase,
            "manifested_phase": self.manifested_phase,
            "why": self.why,
        }


@dataclass(frozen=True)
class Correction:
    """A withdrawn framing of a standing entry: its replacement title and/or
    cause, and the note saying why."""

    entry_id: str
    note: str
    title: str | None
    cause: str | None

    def to_record(self) -> dict[str, str]:
        written = {name: value for name, value in (("title", self.title), ("cause", self.cause)) if value is not None}
        return {"entry_id": self.entry_id, "note": self.note, **written}


@dataclass(frozen=True)
class Note:
    """An unverified claim: synthesis runs with no tools."""

    claim: str
    evidence: str

    def to_record(self) -> dict[str, Any]:
        return {"claim": self.claim, "evidence": self.evidence, "verified": False}


@dataclass(frozen=True)
class Synthesis:
    """A C6 document :func:`parse` accepted whole."""

    new: tuple[NewObservation, ...] = ()
    re_observed: tuple[ReObservation, ...] = ()
    phase_refinements: tuple[PhaseRefinement, ...] = ()
    corrections: tuple[Correction, ...] = ()
    notes: tuple[Note, ...] = ()

    def to_record(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "new": [item.to_record() for item in self.new],
            "re_observed": [item.to_record() for item in self.re_observed],
            "phase_refinements": [item.to_record() for item in self.phase_refinements],
            "corrections": [item.to_record() for item in self.corrections],
            "notes": [item.to_record() for item in self.notes],
        }


@dataclass(frozen=True)
class Unavailable:
    """A reply :func:`parse` refused, and every reason it refused it."""

    reason: str


_DOCUMENT_KEYS = ("new", "re_observed", "phase_refinements", "corrections", "notes")
_FINDING_VERDICTS = tuple(finding_verdict.value for finding_verdict in census_findings.FindingVerdict)
_PHASES = tuple(codebook.PHASES)
_CHANGE_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "severity": verdict.SEVERITIES,
    "verdict": _FINDING_VERDICTS,
    "origin_phase": _PHASES,
    "manifested_phase": _PHASES,
}
_CORRECTION_WRITES = ("title", "cause")
_NOT_CORRECTABLE = {
    "status": "lifecycle status is census-owned",
    "origin_phase": "phases change through phase_refinements",
    "manifested_phase": "phases change through phase_refinements",
}
_NO_TOOLS = "synthesis runs with no tools, so it verifies nothing"

_BOUNDED = reprlib.Repr()
_BOUNDED.maxstring = _BOUNDED.maxother = 120
_UNSHOWN = object()
_Item = TypeVar("_Item")


class _Validator:
    """Every violation of one document against *brief*, each stated as
    ``<path>: <problem> (<bounded value>)``."""

    def __init__(self, brief: Brief) -> None:
        self.problems: list[str] = []
        self._new_keys = frozenset(finding.key for finding in brief.new)
        self._gained = {finding.key: finding.gained_sessions for finding in brief.re_observed}
        self._refinable = brief.refinable_keys
        self._correctable = brief.correctable_entry_ids

    def _add(self, path: str, problem: str, value: object = _UNSHOWN) -> None:
        shown = "" if value is _UNSHOWN else f" ({_BOUNDED.repr(value)})"
        self.problems.append(f"{path}: {problem}{shown}")

    def _object(
        self, value: object, path: str, keys: Sequence[str], *, optional: Sequence[str] = (),
        refused: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """*value* when it is an object of exactly *keys* plus any of *optional*."""
        if not isinstance(value, dict):
            self._add(path, "must be an object", value)
            return None
        before = len(self.problems)
        for name in keys:
            if name not in value:
                self._add(path, f"missing key {name!r}")
        for name in value:
            if name not in keys and name not in optional:
                why = (refused or {}).get(name)
                self._add(path, f"unexpected key {name!r}" + (f": {why}" if why else ""))
        return value if len(self.problems) == before else None

    def _text(self, value: object, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self._add(path, "must be a non-empty string", value)
            return ""
        return value

    def _one_of(self, value: object, path: str, vocabulary: Sequence[str]) -> str:
        if not isinstance(value, str) or value not in vocabulary:
            self._add(path, f"must be one of {list(vocabulary)}", value)
            return ""
        return value

    def _reference(self, value: object, path: str, known: Collection[str], problem: str) -> str:
        if not isinstance(value, str) or value not in known:
            self._add(path, problem, value)
            return ""
        return value

    def _each(
        self, value: object, path: str, item_of: Callable[[object, str], _Item | None], *, unique: str | None = None,
    ) -> tuple[_Item, ...]:
        if not isinstance(value, list):
            self._add(path, "must be a list", value)
            return ()
        items: list[_Item] = []
        seen: set[object] = set()
        for index, raw in enumerate(value):
            item_path = f"{path}[{index}]"
            item = item_of(raw, item_path)
            if item is None:
                continue
            if unique is not None:
                identifier = getattr(item, unique)
                if identifier in seen:
                    self._add(f"{item_path}.{unique}", "duplicate in this list", identifier)
                seen.add(identifier)
            items.append(item)
        return tuple(items)

    def document(self, raw: object) -> Synthesis | None:
        document = self._object(raw, "$", _DOCUMENT_KEYS)
        if document is None:
            return None
        return Synthesis(
            new=self._each(document["new"], "new", self._new, unique="key"),
            re_observed=self._each(document["re_observed"], "re_observed", self._re_observed, unique="key"),
            phase_refinements=self._each(
                document["phase_refinements"], "phase_refinements", self._refinement, unique="key",
            ),
            corrections=self._each(document["corrections"], "corrections", self._correction, unique="entry_id"),
            notes=self._each(document["notes"], "notes", self._note),
        )

    def _new(self, raw: object, path: str) -> NewObservation | None:
        item = self._object(raw, path, ("key", "observation"))
        if item is None:
            return None
        return NewObservation(
            key=self._reference(item["key"], f"{path}.key", self._new_keys, "not a new finding of this run"),
            observation=self._text(item["observation"], f"{path}.observation"),
        )

    def _re_observed(self, raw: object, path: str) -> ReObservation | None:
        item = self._object(raw, path, ("key", "sessions", "changed"))
        if item is None:
            return None
        key = self._reference(item["key"], f"{path}.key", self._gained, "not a re-observed finding of this run")
        gained = self._gained.get(key, ())

        def session(value: object, session_path: str) -> str:
            return self._reference(value, session_path, gained, "not a session this finding gained")

        return ReObservation(
            key=key,
            sessions=self._each(item["sessions"], f"{path}.sessions", session),
            changed=self._each(item["changed"], f"{path}.changed", self._change),
        )

    def _change(self, raw: object, path: str) -> Change | None:
        item = self._object(raw, path, ("field", "from", "to", "why"))
        if item is None:
            return None
        field = self._one_of(item["field"], f"{path}.field", tuple(_CHANGE_VOCABULARIES))
        vocabulary = _CHANGE_VOCABULARIES.get(field)
        if vocabulary is None:
            return None
        return Change(
            field=field,
            from_value=self._one_of(item["from"], f"{path}.from", vocabulary),
            to_value=self._one_of(item["to"], f"{path}.to", vocabulary),
            why=self._text(item["why"], f"{path}.why"),
        )

    def _refinement(self, raw: object, path: str) -> PhaseRefinement | None:
        item = self._object(raw, path, ("key", "origin_phase", "manifested_phase", "why"))
        if item is None:
            return None
        return PhaseRefinement(
            key=self._reference(
                item["key"], f"{path}.key", self._refinable, "no verdict this run, so its phases are not refinable",
            ),
            origin_phase=self._one_of(item["origin_phase"], f"{path}.origin_phase", _PHASES),
            manifested_phase=self._one_of(item["manifested_phase"], f"{path}.manifested_phase", _PHASES),
            why=self._text(item["why"], f"{path}.why"),
        )

    def _correction(self, raw: object, path: str) -> Correction | None:
        item = self._object(raw, path, ("entry_id", "note"), optional=_CORRECTION_WRITES, refused=_NOT_CORRECTABLE)
        if item is None:
            return None
        if not any(name in item for name in _CORRECTION_WRITES):
            self._add(path, "writes neither title nor cause")
        written = {name: self._text(item[name], f"{path}.{name}") for name in _CORRECTION_WRITES if name in item}
        return Correction(
            entry_id=self._reference(
                item["entry_id"], f"{path}.entry_id", self._correctable,
                "not a correctable entry (a re-observed entry that stood before this run)",
            ),
            note=self._text(item["note"], f"{path}.note"),
            title=written.get("title"),
            cause=written.get("cause"),
        )

    def _note(self, raw: object, path: str) -> Note | None:
        item = self._object(raw, path, ("claim", "evidence", "verified"))
        if item is None:
            return None
        if item["verified"] is not False:
            self._add(f"{path}.verified", f"must be false: {_NO_TOOLS}", item["verified"])
        return Note(claim=self._text(item["claim"], f"{path}.claim"), evidence=self._text(item["evidence"], f"{path}.evidence"))


def parse(raw: str, brief: Brief) -> Synthesis | Unavailable:
    """The C6 document in the model's *raw* reply, held to *brief*, or why
    it is unavailable. Nothing is accepted from a document with any problem."""
    try:
        document = coder.parse_coder_output(raw)
    except coder.CoderParseError as exc:
        return Unavailable(f"reply is not a JSON object: {exc}")
    validator = _Validator(brief)
    synthesis = validator.document(document)
    if synthesis is None or validator.problems:
        return Unavailable("; ".join(validator.problems))
    return synthesis


def _quoted(values: Iterable[str]) -> str:
    return " | ".join(f'"{value}"' for value in values)


def reply_instructions() -> str:
    """The synthesis prompt's statement of the C6 reply: its JSON shape and
    every vocabulary and reference :func:`parse` holds it to."""
    vocabularies = "".join(
        f"  {field}: {_quoted(vocabulary)}\n" for field, vocabulary in _CHANGE_VOCABULARIES.items()
    )
    refused = "; ".join(f"never {name} ({why})" for name, why in _NOT_CORRECTABLE.items())
    return (
        "Respond with STRICT JSON ONLY (no prose, no markdown fences), exactly this shape, every key "
        "present and no other key; any list may be empty:\n"
        '{"new": [{"key": "fk-...", "observation": "..."}], '
        '"re_observed": [{"key": "fk-...", "sessions": ["..."], '
        '"changed": [{"field": "...", "from": "...", "to": "...", "why": "..."}]}], '
        '"phase_refinements": [{"key": "fk-...", "origin_phase": "...", "manifested_phase": "...", "why": "..."}], '
        '"corrections": [{"entry_id": "...", "note": "...", "title": "...", "cause": "..."}], '
        '"notes": [{"claim": "...", "evidence": "...", "verified": false}]}\n\n'
        "new: at most one item per key of the brief's new findings.\n"
        "re_observed: at most one item per key of the brief's re_observed findings; sessions are some of "
        "that finding's gained_sessions; each changed item names a field, and from and to are values "
        "of that field:\n"
        + vocabularies
        + "phase_refinements: at most one item per key of a brief finding whose verdict is not null; "
        f"origin_phase and manifested_phase are each one of {_quoted(_PHASES)}.\n"
        "corrections: at most one item per entry_id, which is that of a brief re_observed finding whose "
        "adopted_from is null; note says why its framing is withdrawn; title and/or cause, at least one, "
        f"replaces it; {refused}.\n"
        f"notes: verified is always false: {_NO_TOOLS}.\n"
        "Every observation, why, note, title, cause, claim and evidence is a non-empty string.\n"
    )
