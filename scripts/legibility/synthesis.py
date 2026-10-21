"""The census synthesis (plans/census-incremental-prd.md C6, leaf L10): the
brief the model is given, the one validator of the document it returns, that
document's application to the run, and the report's ``## Synthesis`` section
rendered from the run record.

The brief holds only what this run observed: the keyed entries it created,
and the keyed entries that gained a session. Standing findings with no new
evidence are counted by the runner and never sent to the model.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from legibility import census_findings, codebook, verdict


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
