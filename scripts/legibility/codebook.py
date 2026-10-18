"""scripts/legibility/codebook.py — confusion-codebook v2 schema, validator,
and deterministic sole-writer merger.

See plans/confusion-reduction-prd.md §7.1 (codebook v2 contract) and §7.3
(coding record contract). Per PRD decision 1, this module is the SOLE
WRITER of `docs/legibility/confusion-codebook.yaml`: the LLM trickle coder
and census miner never edit the YAML directly — they emit coding records
(§7.3) that this module's `apply_coding_records()` merges in idempotently,
append-only, never-delete.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# Self-bootstrap for a standalone `python scripts/legibility/codebook.py` run,
# which puts only scripts/legibility/ (not scripts/) on sys.path. Mirrors
# coder.py's identical guard.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bind `shared` to this checkout, as coder.py::_SHARED_SRC does, so a worktree
# run validates keys against its own shared.finding_key. No module-level
# assignment precedes the imports below, so they need no E402 suppression.
if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from legibility import finding_area, invariants
from shared.finding_key import KEY_PATTERN, is_finding_key

logger = logging.getLogger("legibility.codebook")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase enum, shared by entry/sighting/coding-record schemas (PRD §7.1).
PHASES = [
    "prd",
    "decompose",
    "architect",
    "implement",
    "verify",
    "review",
    "merge",
    "recon",
    "ops",
    "unknown",
]

STATUSES = ["open", "partially", "fixed", "retired", "mined-unverified", "accepted"]

DISPOSITIONS = ["pending", "promoted", "rejected"]

_CANDIDATE_ID_RE = re.compile(r"^cand-\d{8}-\d+$")

HEADER = """\
# Agent-confusion codebook — persistent cause registry for legibility surveys.
# version 2. MERGER-OWNED FILE: written solely by scripts/legibility/codebook.py
# (plans/confusion-reduction-prd.md §7.1/§7.3). Append-only, never hand-edited:
# entries are never deleted (retire via status: retired); sightings/candidates
# are appended by the deterministic merger's apply_coding_records(). Hand edits
# will be normalized/overwritten on the next merge.
"""

# Removal-directive markers a §7.3 coding record must never carry (PRD §8.3
# never-delete). Presence of any top-level key, or any match op with one of
# these actions, is rejected outright by apply_coding_records() before it
# mutates anything.
_REMOVAL_KEYS = {"delete", "remove", "retract", "drop"}
_REMOVAL_ACTIONS = {"delete", "remove"}


class NeverDeleteError(Exception):
    """Raised when a codebook write would drop an entry or a candidate, or
    shrink one's sighting history. The confusion codebook is append-only (PRD
    §6/§8.3): entries are retired via status='retired', never removed."""

# ---------------------------------------------------------------------------
# v2 schema (structural). Strict on v2-relevant fields; permissive on v1
# free-form fields (area/cause/fix/fix_where/affected/filed_tasks/
# known_cause_match/sightings_2026_06/upstream) by simply not constraining
# them here — additionalProperties defaults to permissive (open-world).
# ---------------------------------------------------------------------------

_SIGHTING_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "project": {"type": "string"},
        "session": {"type": "string"},
        "origin_phase": {"enum": PHASES},
        "manifested_phase": {"enum": PHASES},
        # A free string here, so validate() keeps the legacy free-text values
        # valid. New writes are screened against the project's invariant
        # slugs by screen_invariant_slugs inside _apply_coding_record.
        "invariant_violated": {"type": ["string", "null"]},
        "note": {"type": "string"},
        "evidence_quote": {"type": "string"},
    },
    "required": ["date", "project", "session", "origin_phase", "manifested_phase"],
}

# Contract run history and ticket fields (plans/census-incremental-prd.md C1;
# docs/quality-findings-contract.md §1, §2, §8). The finding identity fields
# are checked by finding_identity_errors, not by these schemas.
_FINDING_KEY_SCHEMA = {"type": "string", "pattern": KEY_PATTERN}

_RUN_ID_SCHEMA = {"type": "string", "minLength": 1}

FILED_TICKET_RESOLUTIONS = ("created", "combined", "refused", "failed", "existing")

FILED_TICKET_STEPS = ("key", "semantic", "curator")

_FILED_TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "ticket": {"type": ["string", "null"]},
        "project_root": {"type": "string"},
        "run_id": _RUN_ID_SCHEMA,
        # null: filed before finding keys existed (plans/census-incremental-prd.md L4 backfill)
        "finding_key": {"type": ["string", "null"], "pattern": KEY_PATTERN},
        "task_id": {"type": ["integer", "null"]},
        "resolution": {"enum": [*FILED_TICKET_RESOLUTIONS, None]},
        "step": {"enum": list(FILED_TICKET_STEPS)},
    },
    "required": ["ticket", "project_root", "run_id", "finding_key", "task_id", "resolution", "step"],
}

_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "severity": {"enum": ["high", "medium", "low"]},
        "status": {"enum": STATUSES},
        "origin_phase": {"enum": PHASES},
        "manifested_phase": {"enum": PHASES},
        "sightings": {"type": "array", "items": _SIGHTING_SCHEMA},
        "severity_reason": {"type": "string"},
        "first_seen_run": _RUN_ID_SCHEMA,
        "last_seen_runs": {"type": "array", "items": _RUN_ID_SCHEMA},
        "supersedes": {"type": "array", "items": _FINDING_KEY_SCHEMA},
        "filed_tickets": {"type": "array", "items": _FILED_TICKET_SCHEMA},
        "fixed_at_sha": {"type": "string", "pattern": r"^[0-9a-f]{7,40}$"},
        "accepted_by": {"type": "integer"},
    },
    "required": [
        "id",
        "title",
        "severity",
        "status",
        "origin_phase",
        "manifested_phase",
        "sightings",
    ],
}

_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "first_seen": {"type": "string"},
        "disposition": {"enum": DISPOSITIONS},
        "sightings": {"type": "array", "items": _SIGHTING_SCHEMA},
        "attached_by": {
            "type": "object",
            "properties": {
                "method": {"enum": ["session", "title", "key", "similarity"]},
                "score": {"type": "number"},
                "run_id": _RUN_ID_SCHEMA,
            },
            "required": ["method", "run_id"],
        },
        "adjudicated_with": {"type": "string", "pattern": _CANDIDATE_ID_RE.pattern},
    },
    "required": ["id", "title", "first_seen", "disposition", "sightings"],
}

HISTORICAL_BACKFILL = "historical_backfill"
"""The top-level key recording the one run whose backfill linked the census
tasks filed before back-links existed (plans/census-incremental-prd.md L4),
and the task ids it matched to no entry."""

_HISTORICAL_BACKFILL_SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": _RUN_ID_SCHEMA,
        "unmatched_task_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["run_id", "unmatched_task_ids"],
}

V2_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "version": {"const": 2},
        "entries": {"type": "array", "items": _ENTRY_SCHEMA},
        "candidates": {"type": "array", "items": _CANDIDATE_SCHEMA},
        HISTORICAL_BACKFILL: _HISTORICAL_BACKFILL_SCHEMA,
    },
    "required": ["version", "entries"],
}

# ---------------------------------------------------------------------------
# §7.3 coding-record schema (coder output -> merger input)
# ---------------------------------------------------------------------------

_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "entry_id": {"type": "string"},
        "origin_phase": {"enum": PHASES},
        "manifested_phase": {"enum": PHASES},
        "invariant_violated": {"type": ["string", "null"]},
        "note": {"type": "string"},
    },
    "required": ["entry_id"],
}

_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entry_id": {"type": "string"},
        # Required AND non-empty: an entry's framing may never change without
        # an audit trail saying why it was withdrawn, and `_build_sighting`
        # emits `note` only when truthy — so `""` would land the rewrite with
        # a bare, unexplained sighting, defeating the op's one safety
        # property. Required-key-only enforcement let exactly that through.
        # The asymmetry with title/cause below is deliberate: for those, empty
        # means "not supplied" and the correction loop skips the field (a
        # correction can never CLEAR one); for `note` there is no such
        # fallback, so emptiness invalidates the whole record.
        "note": {"type": "string", "minLength": 1},
        "title": {"type": "string"},
        "cause": {"type": "string"},
        "status": {"enum": STATUSES},
        "origin_phase": {"enum": PHASES},
        "manifested_phase": {"enum": PHASES},
    },
    "required": ["entry_id", "note"],
}

_CANDIDATE_RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "cause": {"type": "string"},
        "area": {"type": "string"},
        "origin_phase": {"enum": PHASES},
        "manifested_phase": {"enum": PHASES},
        "evidence_quote": {"type": "string"},
    },
    "required": ["title"],
}

CODING_RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "session": {"type": "string"},
        "date": {"type": "string"},
        "project": {"type": "string"},
        "agent_class": {"type": "string"},
        "matches": {"type": "array", "items": _MATCH_SCHEMA},
        "candidates": {"type": "array", "items": _CANDIDATE_RECORD_SCHEMA},
        # Optional, like its two siblings — every record the coder already
        # emits stays valid without it.
        "corrections": {"type": "array", "items": _CORRECTION_SCHEMA},
    },
    "required": ["session", "date", "project", "agent_class"],
}


def _type_matches(value, type_spec) -> bool:
    """Check `value` against a JSON-Schema `type` keyword (a string, or a
    list of strings meaning "any of"). Supports exactly the primitive types
    this module's schemas use."""
    types = type_spec if isinstance(type_spec, list) else [type_spec]
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "null": lambda v: v is None,
    }
    return any(checks[t](value) for t in types if t in checks)


def _validate_node(instance, schema: dict, path: list[str]) -> list[str]:
    """Minimal recursive validator for the small JSON-Schema subset this
    module's schemas use (type/properties/required/items/enum/const/
    minLength/minItems/pattern).

    Not a general-purpose JSON-Schema implementation: the `jsonschema` PyPI
    package is NOT available in this module's actual verify environment
    (`scripts/tests/` runs via `uv run --project shared`, and
    shared/pyproject.toml does not declare jsonschema — it's only a
    transitive dependency of other subprojects, e.g. fused-memory). This
    hand-rolled walker replaces it without adding a new dependency, over the
    same declarative schema dicts (V2_SCHEMA, CODING_RECORD_SCHEMA, etc.).
    Unknown/extra properties are permitted (open-world), matching plain
    JSON-Schema's default `additionalProperties` behavior.
    """
    loc = "/".join(path) or "<root>"
    errors: list[str] = []

    if "const" in schema:
        if instance != schema["const"]:
            errors.append(f"{loc}: {instance!r} does not equal {schema['const']!r}")
        return errors

    if "enum" in schema:
        if instance not in schema["enum"]:
            errors.append(f"{loc}: {instance!r} is not one of {schema['enum']!r}")
        return errors

    schema_type = schema.get("type")
    if schema_type is not None and not _type_matches(instance, schema_type):
        errors.append(f"{loc}: {instance!r} is not of type {schema_type!r}")
        return errors

    min_length = schema.get("minLength")
    if min_length is not None and isinstance(instance, str) and len(instance) < min_length:
        errors.append(
            f"{loc}: {instance!r} is shorter than the required minLength {min_length}"
        )
        return errors

    pattern = schema.get("pattern")
    if pattern is not None and isinstance(instance, str) and not re.search(pattern, instance):
        errors.append(f"{loc}: {instance!r} does not match the pattern {pattern!r}")
        return errors

    min_items = schema.get("minItems")
    if min_items is not None and isinstance(instance, list) and len(instance) < min_items:
        errors.append(f"{loc}: {instance!r} has fewer than the required minItems {min_items}")
        return errors

    if "properties" in schema and isinstance(instance, dict):
        for required in schema.get("required", []):
            if required not in instance:
                errors.append(f"{loc}: {required!r} is a required property")
        for prop, subschema in schema["properties"].items():
            if prop in instance:
                errors.extend(_validate_node(instance[prop], subschema, path + [prop]))

    if "items" in schema and isinstance(instance, list):
        for i, item in enumerate(instance):
            errors.extend(_validate_node(item, schema["items"], path + [str(i)]))

    return errors


def _schema_errors(instance: dict, schema: dict) -> list[str]:
    return _validate_node(instance, schema, [])


def filed_ticket_errors(row: Mapping[str, object]) -> list[str]:
    """The one statement of the C1 ``filed_tickets`` row invariant: validate()
    applies it to every entry's rows, and disposition.BackLink to itself."""
    return _validate_node(row, _FILED_TICKET_SCHEMA, ["filed_ticket"])


FINDING_IDENTITY_FIELDS = ("finding_key", "finding_area", "finding_anchor", "finding_tags")


@dataclass(frozen=True)
class FindingIdentityError:
    field: str
    problem: str

    def at(self, location: str) -> str:
        return f"{location}/{self.field}: {self.problem}"


def finding_identity_errors(record: Mapping[str, object]) -> list[FindingIdentityError]:
    """The finding identity a codebook record carries (contract §1, §2):
    either none of FINDING_IDENTITY_FIELDS, or all four, holding a §2 key, a
    plain area key, a §1 anchor and a non-empty list of §1 tags.

    The one statement of that invariant: validate() applies it to every
    entry and candidate, and census_findings.FindingIdentity to itself.
    """
    # verdict owns the §1 anchor and tag vocabulary and imports this module.
    from legibility import verdict

    present = [field for field in FINDING_IDENTITY_FIELDS if field in record]
    missing = [field for field in FINDING_IDENTITY_FIELDS if field not in record]
    if not present:
        return []
    if missing:
        return [
            FindingIdentityError(
                field,
                f"is missing beside {', '.join(present)}: a finding identity is all four fields or none",
            )
            for field in missing
        ]
    checks: tuple[tuple[str, Callable[[object], bool], str], ...] = (
        ("finding_key", is_finding_key, "a finding key fk-<12 hex>"),
        (
            "finding_area",
            lambda value: isinstance(value, str) and re.fullmatch(finding_area.AREA_KEY_PATTERN, value) is not None,
            "a plain area key",
        ),
        (
            "finding_anchor",
            lambda value: isinstance(value, str) and verdict.is_anchor(value),
            "a contract §1 anchor: slug:<kebab> or a repo-relative path[::symbol] with no line pin",
        ),
        (
            "finding_tags",
            lambda value: isinstance(value, (list, tuple))
            and bool(value)
            and all(isinstance(tag, str) and verdict.is_finding_tag(tag) for tag in value),
            "a non-empty list of contract §1 tags",
        ),
    )
    return [
        FindingIdentityError(field, f"{record[field]!r} is not {expected}")
        for field, holds, expected in checks
        if not holds(record[field])
    ]


def _finding_identity_errors_at(records: object, path: str) -> list[str]:
    if not isinstance(records, list):
        return []
    return [
        error.at(f"{path}/{index}")
        for index, record in enumerate(records)
        if isinstance(record, dict)
        for error in finding_identity_errors(record)
    ]


def validate(codebook: dict) -> list[str]:
    """Structural + semantic validation of a v2 codebook.

    Returns a list of human-readable error strings; an empty list means the
    codebook is valid. Strict on v2-relevant fields (version, entry
    id/title/severity/status, origin_phase/manifested_phase enums, sighting
    shape, candidate id regex + disposition, the optional C1 finding fields,
    finding_identity_errors, accepted ⇒ accepted_by); permissive on v1
    free-form fields (validate() never rejects on their presence, absence,
    or type).
    """
    errors = _schema_errors(codebook, V2_SCHEMA)
    if isinstance(codebook, dict):
        errors.extend(_finding_identity_errors_at(codebook.get("entries"), "entries"))
        errors.extend(_finding_identity_errors_at(codebook.get("candidates"), "candidates"))

    entries = codebook.get("entries", []) if isinstance(codebook, dict) else []
    if isinstance(entries, list):
        seen_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if entry_id is None:
                continue
            if entry_id in seen_ids:
                errors.append(f"duplicate entry id: {entry_id!r}")
            else:
                seen_ids.add(entry_id)
            if entry.get("status") == "accepted" and "accepted_by" not in entry:
                errors.append(f"entry {entry_id!r}: status 'accepted' requires accepted_by (contract §7)")

    candidates = codebook.get("candidates", []) if isinstance(codebook, dict) else []
    if isinstance(candidates, list):
        seen_cand_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            cand_id = candidate.get("id")
            if not isinstance(cand_id, str):
                continue
            if not _CANDIDATE_ID_RE.match(cand_id):
                errors.append(
                    f"candidate id does not match cand-<yyyymmdd>-<n>: {cand_id!r}"
                )
            if cand_id in seen_cand_ids:
                errors.append(f"duplicate candidate id: {cand_id!r}")
            else:
                seen_cand_ids.add(cand_id)

    return errors


def _build_sighting(payload: dict, *, session: str, date: str, project: str) -> dict:
    """Build a sighting dict from a match/candidate payload + record header
    fields. invariant_violated/note/evidence_quote are included only when
    truthy (never emitted as an explicit null/empty key)."""
    sighting = {
        "date": date,
        "project": project,
        "session": session,
        "origin_phase": payload.get("origin_phase", "unknown"),
        "manifested_phase": payload.get("manifested_phase", "unknown"),
    }
    invariant_violated = payload.get("invariant_violated")
    if invariant_violated:
        sighting["invariant_violated"] = invariant_violated
    note = payload.get("note")
    if note:
        sighting["note"] = note
    evidence_quote = payload.get("evidence_quote")
    if evidence_quote:
        sighting["evidence_quote"] = evidence_quote
    return sighting


def _deletion_directive(record: object) -> str | None:
    """Why `record` is deletion-shaped, or None when it is not: a top-level
    delete/remove/retract/drop key, or a match/correction carrying a
    delete/remove action."""
    if not isinstance(record, dict):
        return None
    for key in _REMOVAL_KEYS:
        if key in record:
            return f"coding record carries a removal directive: top-level {key!r} key"
    for match in record.get("matches", []) or []:
        if isinstance(match, dict) and match.get("action") in _REMOVAL_ACTIONS:
            return f"coding record match carries a removal action: {match.get('action')!r}"
    for correction in record.get("corrections", []) or []:
        if isinstance(correction, dict) and correction.get("action") in _REMOVAL_ACTIONS:
            return (
                "coding record correction carries a removal action: "
                f"{correction.get('action')!r}"
            )
    return None


def assert_no_deletion(before: dict, after: dict) -> None:
    """Structural never-delete post-condition, independent of how `after`
    was constructed: every entry id present in `before` must still be
    present in `after`, and every session recorded in an entry's sightings
    in `before` must still be present in that same entry's sightings in
    `after`. The same two checks apply to top-level `candidates` — id
    presence and per-candidate sighting sessions — since candidates hold
    real mined signal (pending novel causes awaiting promotion) and must
    be covered by the same construction-independent safety net as entries.
    Retirement (status -> 'retired') is unaffected — it changes a field,
    not entry/sighting presence. Raises NeverDeleteError otherwise.
    """
    before_entries = {
        e["id"]: e
        for e in (before.get("entries", []) or [])
        if isinstance(e, dict) and "id" in e
    }
    after_entries = {
        e["id"]: e
        for e in (after.get("entries", []) or [])
        if isinstance(e, dict) and "id" in e
    }

    missing = set(before_entries) - set(after_entries)
    if missing:
        raise NeverDeleteError(f"entries dropped from codebook: {sorted(missing)!r}")

    for entry_id, before_entry in before_entries.items():
        before_sessions = {
            s.get("session") for s in (before_entry.get("sightings", []) or [])
        }
        after_sessions = {
            s.get("session")
            for s in (after_entries[entry_id].get("sightings", []) or [])
        }
        if not before_sessions <= after_sessions:
            raise NeverDeleteError(
                f"entry {entry_id!r} lost sightings for session(s): "
                f"{sorted(s for s in (before_sessions - after_sessions) if s is not None)!r}"
            )

    before_candidates = {
        c["id"]: c
        for c in (before.get("candidates", []) or [])
        if isinstance(c, dict) and "id" in c
    }
    after_candidates = {
        c["id"]: c
        for c in (after.get("candidates", []) or [])
        if isinstance(c, dict) and "id" in c
    }

    missing_candidates = set(before_candidates) - set(after_candidates)
    if missing_candidates:
        raise NeverDeleteError(
            f"candidates dropped from codebook: {sorted(missing_candidates)!r}"
        )

    for cand_id, before_candidate in before_candidates.items():
        before_sessions = {
            s.get("session") for s in (before_candidate.get("sightings", []) or [])
        }
        after_sessions = {
            s.get("session")
            for s in (after_candidates[cand_id].get("sightings", []) or [])
        }
        if not before_sessions <= after_sessions:
            raise NeverDeleteError(
                f"candidate {cand_id!r} lost sightings for session(s): "
                f"{sorted(s for s in (before_sessions - after_sessions) if s is not None)!r}"
            )


_SIGHTING_OPS = ("matches", "candidates", "corrections")


@dataclass(frozen=True)
class SlugTally:
    """``invariant_violated`` values screened on write: how many named a
    declared slug, and the values that did not (stored as absent)."""

    valid: int = 0
    rejected: tuple[str, ...] = ()

    def plus(self, other: SlugTally) -> SlugTally:
        return SlugTally(
            valid=self.valid + other.valid,
            rejected=self.rejected + other.rejected,
        )


@dataclass(frozen=True)
class MergeCounts:
    """What merged coding records did, op by op. The field names are the keys
    of the `codebook.py apply` summary line."""

    matched: int = 0
    candidates_applied: int = 0
    candidate_disposition_conflicts: int = 0
    corrections_applied: int = 0
    correction_skipped: int = 0
    skipped_unknown_entry: int = 0

    def plus(self, other: MergeCounts) -> MergeCounts:
        return MergeCounts(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )

    @property
    def mutations(self) -> int:
        """The ops that wrote the codebook: every appended sighting (a recurrence
        on an adjudicated title included) and every correction."""
        return (
            self.matched
            + self.candidates_applied
            + self.candidate_disposition_conflicts
            + self.corrections_applied
        )


@dataclass(frozen=True)
class SkippedRecord:
    """A record a merge left out whole: its position in the batch, its
    session when it has one, and why."""

    index: int
    session: str | None
    reason: str

    @classmethod
    def of(cls, index: int, record: object, reason: str) -> SkippedRecord:
        session = record.get("session") if isinstance(record, dict) else None
        return cls(index=index, session=session if isinstance(session, str) else None, reason=reason)

    def describe(self) -> str:
        return f"session={self.session}: {self.reason}"


@dataclass(frozen=True)
class MergeResult:
    """One batch merged by `apply_coding_records`.

    `merged_records` are the records that went through the merge, in batch
    order, an already-absorbed re-apply included; `skipped_deletions` and
    `invalid_records` are the ones left out; `counts` and `invariant_slugs` sum
    over the merged ones.
    """

    codebook: dict
    counts: MergeCounts = MergeCounts()
    merged_records: tuple[dict, ...] = ()
    skipped_deletions: tuple[SkippedRecord, ...] = ()
    invalid_records: tuple[SkippedRecord, ...] = ()
    invariant_slugs: SlugTally = SlugTally()

    @property
    def mutated(self) -> bool:
        return self.counts.mutations > 0

    def summary(self) -> str:
        values = {f.name: getattr(self.counts, f.name) for f in fields(self.counts)}
        values.update(
            record_invalid=len(self.invalid_records),
            deletion_directive=len(self.skipped_deletions),
            invariant_slugs_valid=self.invariant_slugs.valid,
            invariant_slugs_rejected=len(self.invariant_slugs.rejected),
        )
        return " ".join(f"{key}={value}" for key, value in values.items())


def screen_invariant_slugs(
    record: dict, invariant_slugs: Collection[str]
) -> tuple[dict, SlugTally]:
    """A copy of *record* keeping only ``invariant_violated`` values that are
    one of *invariant_slugs*, and the tally of what was kept and dropped.

    Covers every op a sighting is built from. An empty value is neither
    valid nor rejected: it is already absent.
    """
    screened = copy.deepcopy(record)
    valid = 0
    rejected: list[str] = []
    for key in _SIGHTING_OPS:
        for op in screened.get(key) or []:
            if not isinstance(op, dict):
                continue
            value = op.get("invariant_violated")
            if not value:
                continue
            if isinstance(value, str) and value in invariant_slugs:
                valid += 1
                continue
            del op["invariant_violated"]
            rejected.append(value if isinstance(value, str) else repr(value))
    return screened, SlugTally(valid=valid, rejected=tuple(rejected))


_MAX_REPORTED_REJECTED_SLUGS = 10
_MAX_REPORTED_SLUG_CHARS = 80


def _bounded_rejected_slugs(rejected: Collection[str]) -> str:
    """The distinct *rejected* values, sorted, as one bounded rendering: they
    are LLM free text, so both their number and their length are capped."""
    distinct = sorted(set(rejected))
    shown = [
        value if len(value) <= _MAX_REPORTED_SLUG_CHARS
        else value[:_MAX_REPORTED_SLUG_CHARS] + "..."
        for value in distinct[:_MAX_REPORTED_REJECTED_SLUGS]
    ]
    hidden = len(distinct) - len(shown)
    return f"{shown} +{hidden} more" if hidden else f"{shown}"


def warn_unknown_invariant_slugs(
    tally: SlugTally, invariant_slugs: Collection[str], *, run: str
) -> None:
    """The one WARNING a run emits for its rejected ``invariant_violated``
    values, if it rejected any. The count is exact; the values are bounded."""
    if not tally.rejected:
        return
    declared = (
        ", ".join(invariant_slugs)
        if invariant_slugs
        else f"none (the project declares no slug in {invariants.DOC_RELPATH})"
    )
    logger.warning(
        "codebook: run %s stored %d invariant_violated value(s) as absent "
        "because they name no declared invariant slug: %s; declared slugs: %s",
        run, len(tally.rejected), _bounded_rejected_slugs(tally.rejected), declared,
    )


def apply_coding_records(
    codebook: dict, records: Iterable[dict], *, invariant_slugs: Collection[str]
) -> MergeResult:
    """Merge a batch of §7.3 coding records into a copy of *codebook*.

    Never mutates or aliases *codebook*. A deletion-shaped record lands in
    `skipped_deletions` and one failing `validate_coding_record` in
    `invalid_records`, each skipped whole, before it can touch the copy, while
    the rest of the batch merges.

    Raises ValueError for a non-dict *codebook*. Raises NeverDeleteError only
    when `assert_no_deletion()`, the construction-independent safety net run
    over the whole merge, finds a dropped entry or sighting: that is a merger
    bug, never a property of a record.
    """
    if not isinstance(codebook, dict):
        raise ValueError(
            "apply_coding_records: expected a dict-shaped codebook, got "
            f"{type(codebook).__name__} (empty or malformed YAML file?)"
        )
    merged = copy.deepcopy(codebook)
    counts, slugs = MergeCounts(), SlugTally()
    merged_records: list[dict] = []
    skipped_deletions: list[SkippedRecord] = []
    invalid_records: list[SkippedRecord] = []
    for index, record in enumerate(records):
        directive = _deletion_directive(record)
        if directive is not None:
            skipped_deletions.append(SkippedRecord.of(index, record, directive))
            continue
        errors = validate_coding_record(record)
        if errors:
            invalid_records.append(SkippedRecord.of(index, record, "; ".join(errors)))
            continue
        record_counts, record_slugs = _apply_coding_record(
            merged, record, invariant_slugs=invariant_slugs
        )
        counts, slugs = counts.plus(record_counts), slugs.plus(record_slugs)
        merged_records.append(record)
    assert_no_deletion(codebook, merged)
    return MergeResult(
        codebook=merged,
        counts=counts,
        merged_records=tuple(merged_records),
        skipped_deletions=tuple(skipped_deletions),
        invalid_records=tuple(invalid_records),
        invariant_slugs=slugs,
    )


def _apply_coding_record(
    codebook: dict, record: dict, *, invariant_slugs: Collection[str]
) -> tuple[MergeCounts, SlugTally]:
    """Apply one §7.3 coding record to a v2 codebook, in place. Sole-writer
    merge. Preconditions: *codebook* is the batch's own copy, and *record* is
    valid and not deletion-shaped; the batch, `apply_coding_records`, ensures
    all three.

    Three ops: `matches` (append a sighting to an existing entry),
    `candidates` (file a novel cause for the census to adjudicate), and
    `corrections` (withdraw an entry's refuted framing). They are applied in
    the order corrections → matches → candidates (see the correction-handling
    paragraph for why corrections go first).

    Returns what each op did and the record's `SlugTally`.

    The record is first screened by `screen_invariant_slugs()` against
    *invariant_slugs*, the observed project's declared vocabulary: an
    `invariant_violated` value that is not one of them is stored as absent
    rather than failing the record. Emitting the run's one WARNING is the
    caller's job (`warn_unknown_invariant_slugs()`).

    Match handling: an entry_id that doesn't resolve to an existing entry
    is counted as skipped and never fabricated into a new entry — only the
    census promotes candidates to entries. A sighting is appended to the
    matched entry only if no existing sighting on that entry already
    carries this session (dedup key: (session, entry_id), the entry being
    implicit in "that entry's sightings list").

    Candidate handling: a candidate title is resolved against the existing
    `candidates` list. A still-`pending` record of that title absorbs the
    sighting (`candidates_applied`); a genuinely unseen title creates a new
    pending record (`candidates_applied`); a title whose records are ALL
    already adjudicated (rejected/promoted) takes neither path — no pending
    twin is fabricated, and the recurrence sighting is routed to whichever
    record a consumer will actually read:

    - `promoted` with a `promoted_to` naming an existing entry → the
      sighting is appended to THAT ENTRY (session-deduped, counted in
      `matched`, exactly like the match path). A promoted candidate's own
      `sightings` list is a dead field: census.promote_candidate deep-copies
      it into the new entry once at promotion time and nothing re-reads it
      afterwards, so filing a recurrence there would lose the signal while
      reporting it preserved.
    - `rejected`, or `promoted` with a `promoted_to` that resolves to no
      entry → the sighting is appended to the existing candidate record and
      counted in `candidate_disposition_conflicts`. The merger never
      fabricates an entry (only the census promotes), so the candidate is
      the only place left to keep it.

    Correction handling: a `corrections` op writes an entry's `title`,
    `cause` and/or `status` in place, and appends one note-bearing sighting
    as the audit trail. This is the one op that is NOT a pure append, and the
    boundary is deliberate: a sighting is an immutable dated observation and
    stays append-only, while an entry's `title`/`cause` is the CURRENT best
    explanation of a cause — and is the only part of an entry that
    `scripts/legibility/coder.py::build_codebook_index` renders to the coder,
    so a refuted framing can only be withdrawn there. A field is written only
    when the correction supplies it AND the value is truthy, so a correction
    can never clear one; `note`, by contrast, is schema-enforced non-empty,
    so the rewrite can never land without its audit trail. An unresolvable
    `entry_id` is counted in `skipped_unknown_entry` exactly as in the match
    path — the merger never fabricates an entry. The whole op (field writes
    and sighting alike) is gated by the same per-entry `session` dedup the
    match path uses, so re-running `apply` over an already-merged record is
    exactly a no-op.

    That dedup admits one sighting per (session, entry), so a record carrying
    two ops for the SAME entry can land only one of them. Corrections are
    therefore applied FIRST: a correction is the only op that can withdraw a
    refuted framing, and it carries its own note-bearing sighting, so letting
    a sibling `matches` op win would trade the whole withdrawal for one
    interchangeable sighting. A sibling match for an entry a correction just
    corrected is absorbed (no second sighting, not counted in `matched`).
    Two corrections for one entry in one record still collide, and the loser
    is reported rather than dropped: `correction_skipped` counts it and a
    WARNING names the entry and session — distinguishing it from the silent,
    correct no-op of re-applying an already-merged record.

    `status` remains census-owned for lifecycle transitions
    (`scripts/legibility/census.py::retire_entry`); a correction writing it
    is an operator adjudication CARRIED BY the record, not a
    merger-initiated transition.

    Never-resurrect: the merger must never re-open a census verdict. Only
    the census writes `disposition` (census.py promote_candidate/
    reject_candidate), so an adjudicated-title collision leaves that field
    untouched and surfaces itself via `candidate_disposition_conflicts`
    rather than by manufacturing a byte-identical-title pending record —
    which is precisely how rejected `cand-20260722-28` reappeared as
    pending `cand-20260724-2` in the live registry.
    """
    matched = candidates_applied = candidate_disposition_conflicts = 0
    corrections_applied = correction_skipped = skipped_unknown_entry = 0
    record, slugs = screen_invariant_slugs(record, invariant_slugs)

    session = record["session"]
    date = record["date"]
    project = record["project"]

    entries_by_id = {e["id"]: e for e in codebook.get("entries", [])}

    # Entries that ALREADY carried this session before this record was applied.
    # Everything below deduplicates on (session, entry), so this set is what
    # separates "already merged on an earlier run" — the idempotent re-apply
    # the dedup exists for, correctly silent — from "a sibling op in THIS
    # record got there first", which is a collision and must not be silent.
    premerged = {
        entry_id
        for entry_id, entry in entries_by_id.items()
        if session in {s.get("session") for s in entry.get("sightings", []) or []}
    }

    # `corrections` runs BEFORE `matches`, deliberately: both ops write at most
    # one sighting per (session, entry), so when one record carries both for the
    # same entry exactly one can land. The correction is the one that must —
    # it is the only op that can withdraw a refuted framing, while a match adds
    # a sighting whose own note the correction's sighting stands in for.
    for correction in record.get("corrections", []) or []:
        entry_id = correction.get("entry_id")
        entry = entries_by_id.get(entry_id)
        if entry is None:
            skipped_unknown_entry += 1
            continue
        sightings = entry.setdefault("sightings", [])
        if session in {s.get("session") for s in sightings}:
            if entry_id not in premerged:
                # Another correction for this same entry_id already consumed
                # this record's one sighting slot, so this one's field writes
                # cannot land. Silently dropping a framing withdrawal is the
                # one outcome this op must never have.
                correction_skipped += 1
                logger.warning(
                    "codebook: correction for entry %r dropped — session %r was "
                    "already recorded on that entry by an earlier op in the SAME "
                    "record, so its title/cause/status writes did not land; split "
                    "the record or merge the corrections",
                    entry_id, session,
                )
            continue
        for field in ("title", "cause", "status"):
            value = correction.get(field)
            # Supplied AND truthy: a correction can never CLEAR a field, since
            # an emptied title/cause is a deletion in disguise.
            if value:
                entry[field] = value
        sightings.append(
            _build_sighting(correction, session=session, date=date, project=project)
        )
        corrections_applied += 1

    for match in record.get("matches", []) or []:
        entry = entries_by_id.get(match.get("entry_id"))
        if entry is None:
            skipped_unknown_entry += 1
            continue
        sightings = entry.setdefault("sightings", [])
        existing_sessions = {s.get("session") for s in sightings}
        if session in existing_sessions:
            continue
        sightings.append(_build_sighting(match, session=session, date=date, project=project))
        matched += 1

    candidates = codebook.setdefault("candidates", [])
    date_prefix = f"cand-{date.replace('-', '')}-"

    for candidate_payload in record.get("candidates", []) or []:
        title = candidate_payload.get("title")
        same_title = [c for c in candidates if c.get("title") == title]

        already_seen = any(
            session in {s.get("session") for s in c.get("sightings", [])}
            for c in same_title
        )
        if already_seen:
            continue

        sighting = _build_sighting(candidate_payload, session=session, date=date, project=project)
        pending_match = next(
            (c for c in same_title if c.get("disposition") == "pending"), None
        )
        if pending_match is not None:
            pending_match.setdefault("sightings", []).append(sighting)
            candidates_applied += 1
        elif same_title:
            # Every same-title candidate is already adjudicated (rejected/
            # promoted). Fabricating a fresh pending twin here would silently
            # undo that verdict (this is exactly how rejected
            # cand-20260722-28 came back as pending cand-20260724-2 in the
            # live codebook). So: never touch `disposition` -- promote/reject
            # transitions are census-owned (census.py promote_candidate/
            # reject_candidate) -- and route the recurrence sighting to
            # wherever it will actually be READ.
            promoted_entry = None
            for c in reversed(same_title):
                if c.get("disposition") == "promoted":
                    promoted_entry = entries_by_id.get(c.get("promoted_to"))
                    if promoted_entry is not None:
                        break
            if promoted_entry is not None:
                # The census already turned this title into a real entry.
                # promote_candidate deep-copies the candidate's `sightings`
                # into the entry ONCE and never again, and the entry is what
                # the census files from: census.py::_entry_cluster rebuilds a
                # withheld entry's task, evidence included, from ENTRY
                # sightings alone. Filing the recurrence on the candidate
                # would leave it out of that task -- so it goes on the entry,
                # deduped by session and counted exactly as the match path
                # above counts its own appends.
                entry_sightings = promoted_entry.setdefault("sightings", [])
                if session not in {s.get("session") for s in entry_sightings}:
                    entry_sightings.append(sighting)
                    matched += 1
            else:
                # Rejected, or promoted with a `promoted_to` that names no
                # existing entry (hand-edited/pre-promote_candidate record).
                # The merger never fabricates an entry -- only the census
                # does -- so the sighting lands on the candidate so the
                # signal is not lost, and the conflict is surfaced via
                # counts. `same_title[-1]` (last in list order = most recently
                # added) is the deterministic tie-break when more than one
                # adjudicated record shares a title.
                same_title[-1].setdefault("sightings", []).append(sighting)
                candidate_disposition_conflicts += 1
        else:
            n = 1 + sum(
                1 for c in candidates if str(c.get("id", "")).startswith(date_prefix)
            )
            candidates.append(
                {
                    "id": f"{date_prefix}{n}",
                    "title": title,
                    "cause": candidate_payload.get("cause"),
                    "area": candidate_payload.get("area"),
                    "first_seen": date,
                    "disposition": "pending",
                    "sightings": [sighting],
                }
            )
            candidates_applied += 1

    return MergeCounts(
        matched=matched,
        candidates_applied=candidates_applied,
        candidate_disposition_conflicts=candidate_disposition_conflicts,
        corrections_applied=corrections_applied,
        correction_skipped=correction_skipped,
        skipped_unknown_entry=skipped_unknown_entry,
    ), slugs


def validate_coding_record(record: dict) -> list[str]:
    """Validate a §7.3 coding record (coder output -> merger input).

    Returns a list of human-readable error strings; an empty list means the
    record is valid. Required: session/date/project/agent_class (strings).
    matches/candidates are optional lists; invariant_violated accepts null
    or any string here. An unknown slug is not an error: apply_coding_records
    stores it as absent through screen_invariant_slugs.
    """
    return _schema_errors(record, CODING_RECORD_SCHEMA)


# ---------------------------------------------------------------------------
# load / dump — deterministic canonical YAML I/O
# ---------------------------------------------------------------------------

def load(path: str | os.PathLike) -> dict:
    """Load a codebook YAML file into a plain dict via yaml.safe_load."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump(codebook: dict, path: str | os.PathLike) -> None:
    """Write `codebook` to `path` in canonical, deterministic form.

    ruamel.yaml (comment-preserving round-trip) is NOT available in this
    environment, so — per PRD decision 1 (merger is the sole writer) — this
    normalizes to a fixed canonical block-style form: sort_keys=False
    (preserve dict insertion order), default_flow_style=False (block style,
    no inline `{...}`), allow_unicode=True, a wide width (avoid
    nondeterministic line wrapping), prefixed with the fixed HEADER comment.
    Byte-stable given byte-stable input, so a no-change night commits
    nothing (PRD §6.7).

    Written atomically: the full document is assembled in a temp file
    created in the same directory as `path`, then moved into place with
    os.replace() (an atomic rename on POSIX). `path` therefore only ever
    holds a complete, previously-flushed document — a crash or kill
    between truncate and write can no longer leave a partial/corrupt file
    behind for the next load()/validate() to trip over. This matters
    because this file is the single canonical, sole-writer registry for
    the whole legibility pipeline (PRD decision 1), rewritten by the
    nightly/trickle merger.
    """
    body = yaml.safe_dump(
        codebook,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=4096,
    )
    directory = os.path.dirname(os.fspath(path)) or "."
    fd, tmp_file = tempfile.mkstemp(prefix=".codebook-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(HEADER)
            f.write(body)
        os.replace(tmp_file, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_file)
        raise


# ---------------------------------------------------------------------------
# migrate_v1_to_v2 — in-place-shaped v1 → v2 upgrade
# ---------------------------------------------------------------------------

def migrate_v1_to_v2(codebook: dict) -> dict:
    """Return a v2-shaped copy of `codebook` (does not mutate the input).

    version -> 2; each entry gains origin_phase/manifested_phase='unknown'
    and sightings=[] (only if absent — idempotent on already-migrated
    input); status 'yes' (not a v2 enum member) maps to 'open' (the v1
    'yes' entries are verified-real-but-unfixed causes, which is exactly
    'open' semantics); every other v1 field, including sightings_2026_06,
    is retained verbatim; entry order is preserved. Top-level candidates
    defaults to [] (only if absent). The top-level `updated` field (if
    present) is DROPPED rather than retained or re-stamped: under the
    sole-writer design dump()/apply_coding_records() never touch it, so a
    retained value would freeze at the migration date and mislead readers
    as sightings accumulate, while a mutable timestamp would break dump()'s
    byte-stable no-change-night guarantee (PRD §6.7). Output passes
    validate() (V2_SCHEMA never required `updated`).

    Raises ValueError if `codebook` is not a dict (e.g. an empty or
    malformed YAML file loads via yaml.safe_load as None, or as some
    non-mapping) — mirrors validate()'s isinstance guard so malformed
    input fails loudly with a clear message here instead of an
    AttributeError/TypeError deep in the mutation logic below.
    """
    if not isinstance(codebook, dict):
        raise ValueError(
            "migrate_v1_to_v2: expected a dict-shaped codebook, got "
            f"{type(codebook).__name__} (empty or malformed YAML file?)"
        )
    result = copy.deepcopy(codebook)
    result["version"] = 2
    result.pop("updated", None)

    for entry in result.get("entries", []):
        entry.setdefault("origin_phase", "unknown")
        entry.setdefault("manifested_phase", "unknown")
        entry.setdefault("sightings", [])
        # YAML 1.1 (PyYAML's default resolver) coerces an unquoted `yes` to
        # the Python bool True, so a real v1 file's `status: yes` loads as
        # True, not the string "yes" (confirmed against the live
        # docs/legibility/confusion-codebook.yaml) -- both spellings map to
        # 'open'.
        if entry.get("status") in ("yes", True):
            entry["status"] = "open"

    result.setdefault("candidates", [])

    return result


# ---------------------------------------------------------------------------
# main(argv) — CLI: validate / migrate / apply. Fail-loud: every subcommand
# returns 0 iff its operation succeeded and the resulting file (if any) is
# schema-valid; errors go to stderr.
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> int:
    codebook = load(args.path)
    errors = validate(codebook)
    for error in errors:
        print(error, file=sys.stderr)
    return 0 if not errors else 1


def _cmd_migrate(args: argparse.Namespace) -> int:
    codebook = load(args.path)
    if not isinstance(codebook, dict):
        print(
            f"{args.path}: expected a dict-shaped codebook, got "
            f"{type(codebook).__name__} (empty or malformed YAML file?)",
            file=sys.stderr,
        )
        return 1
    migrated = migrate_v1_to_v2(codebook)
    errors = validate(migrated)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    dump(migrated, args.path)
    return 0


def _read_jsonl(path: str) -> tuple[list[tuple[int, Any]], int]:
    """The (line number, value) of each non-blank line of *path*, and how many
    lines were not JSON. Each of those is named on stderr and skipped."""
    parsed: list[tuple[int, Any]] = []
    malformed = 0
    with open(path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed.append((lineno, json.loads(line)))
            except json.JSONDecodeError as exc:
                print(f"{path}:{lineno}: malformed JSON, skipping: {exc}", file=sys.stderr)
                malformed += 1
    return parsed, malformed


def _cmd_apply(args: argparse.Namespace) -> int:
    """Merge a JSONL batch of §7.3 coding records into a v2 codebook file.

    A malformed JSON line, a deletion-shaped record and a schema-invalid record
    are each skipped, named on stderr by `path:lineno:`, and counted on the
    stdout `applied:` line; the rest of the batch still lands. Returns 1,
    leaving the file untouched, only for a non-dict codebook or a merged
    codebook that fails `validate()`.
    """
    invariant_slugs = invariants.read_slugs(args.project_root)
    codebook = load(args.codebook)
    if not isinstance(codebook, dict):
        print(
            f"{args.codebook}: expected a dict-shaped codebook, got "
            f"{type(codebook).__name__} (empty or malformed YAML file?)",
            file=sys.stderr,
        )
        return 1
    parsed, malformed = _read_jsonl(args.records)
    merge = apply_coding_records(
        codebook, [record for _, record in parsed], invariant_slugs=invariant_slugs
    )
    for label, skips in (
        ("deletion directive rejected", merge.skipped_deletions),
        ("invalid coding record", merge.invalid_records),
    ):
        for skipped in skips:
            lineno = parsed[skipped.index][0]
            print(
                f"{args.records}:{lineno}: {label}, skipping: {skipped.describe()}",
                file=sys.stderr,
            )
    warn_unknown_invariant_slugs(merge.invariant_slugs, invariant_slugs, run=f"apply {args.records}")

    errors = validate(merge.codebook)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    dump(merge.codebook, args.codebook)
    print(f"applied: {merge.summary()} malformed_json={malformed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codebook",
        description="confusion-codebook v2 schema validator, v1->v2 migrator, "
        "and deterministic sole-writer coding-record merger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a v2 codebook file")
    validate_parser.add_argument("path")
    validate_parser.set_defaults(func=_cmd_validate)

    migrate_parser = subparsers.add_parser(
        "migrate", help="migrate a v1 codebook file to v2, in place"
    )
    migrate_parser.add_argument("path")
    migrate_parser.set_defaults(func=_cmd_migrate)

    apply_parser = subparsers.add_parser(
        "apply", help="apply a JSONL file of §7.3 coding records to a codebook file"
    )
    apply_parser.add_argument(
        "--project-root",
        required=True,
        help="the observed project, whose design-invariants doc declares the "
        "invariant slugs a sighting may name",
    )
    apply_parser.add_argument("codebook")
    apply_parser.add_argument("records")
    apply_parser.set_defaults(func=_cmd_apply)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
