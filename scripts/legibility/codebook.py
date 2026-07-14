"""scripts/legibility/codebook.py — confusion-codebook v2 schema, validator,
and deterministic sole-writer merger.

See plans/confusion-reduction-prd.md §7.1 (codebook v2 contract) and §7.3
(coding record contract). Per PRD decision 1, this module is the SOLE
WRITER of `docs/legibility/confusion-codebook.yaml`: the LLM trickle coder
and census miner never edit the YAML directly — they emit coding records
(§7.3) that this module's `apply_coding_record()` merges in idempotently,
append-only, never-delete.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

import yaml

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

STATUSES = ["open", "partially", "fixed", "retired", "mined-unverified"]

DISPOSITIONS = ["pending", "promoted", "rejected"]

_CANDIDATE_ID_RE = re.compile(r"^cand-\d{8}-\d+$")

HEADER = """\
# Agent-confusion codebook — persistent cause registry for legibility surveys.
# version 2. MERGER-OWNED FILE: written solely by scripts/legibility/codebook.py
# (plans/confusion-reduction-prd.md §7.1/§7.3). Append-only, never hand-edited:
# entries are never deleted (retire via status: retired); sightings/candidates
# are appended by the deterministic merger's apply_coding_record(). Hand edits
# will be normalized/overwritten on the next merge.
"""

# Removal-directive markers a §7.3 coding record must never carry (PRD §8.3
# never-delete). Presence of any top-level key, or any match op with one of
# these actions, is rejected outright by apply_coding_record() before it
# mutates anything.
_REMOVAL_KEYS = {"delete", "remove", "retract", "drop"}
_REMOVAL_ACTIONS = {"delete", "remove"}


class NeverDeleteError(Exception):
    """Raised when a coding record carries a deletion-shaped directive, or
    when a merge result would drop a codebook entry or shrink an entry's
    sighting history. The confusion codebook is append-only (PRD §6/§8.3):
    entries are retired via status='retired', never removed."""

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
        # Free string, NOT slug-checked against design-invariants.md's INV
        # vocabulary — this PRD ships the field, the sibling PRD owns the gate.
        "invariant_violated": {"type": ["string", "null"]},
        "note": {"type": "string"},
        "evidence_quote": {"type": "string"},
    },
    "required": ["date", "project", "session", "origin_phase", "manifested_phase"],
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
    },
    "required": ["id", "title", "first_seen", "disposition", "sightings"],
}

V2_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "version": {"const": 2},
        "entries": {"type": "array", "items": _ENTRY_SCHEMA},
        "candidates": {"type": "array", "items": _CANDIDATE_SCHEMA},
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
        "null": lambda v: v is None,
    }
    return any(checks[t](value) for t in types if t in checks)


def _validate_node(instance, schema: dict, path: list[str]) -> list[str]:
    """Minimal recursive validator for the small JSON-Schema subset this
    module's schemas use (type/properties/required/items/enum/const).

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


def validate(codebook: dict) -> list[str]:
    """Structural + semantic validation of a v2 codebook.

    Returns a list of human-readable error strings; an empty list means the
    codebook is valid. Strict on v2-relevant fields (version, entry
    id/title/severity/status, origin_phase/manifested_phase enums, sighting
    shape, candidate id regex + disposition); permissive on v1 free-form
    fields (validate() never rejects on their presence, absence, or type).
    """
    errors = _schema_errors(codebook, V2_SCHEMA)

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


def _reject_deletion_directive(record: dict) -> None:
    """Raise NeverDeleteError if `record` is deletion-shaped: a top-level
    delete/remove/retract/drop key, or a match carrying a delete/remove
    action. Called before apply_coding_record touches anything."""
    if not isinstance(record, dict):
        return
    for key in _REMOVAL_KEYS:
        if key in record:
            raise NeverDeleteError(
                f"coding record carries a removal directive: top-level {key!r} key"
            )
    for match in record.get("matches", []) or []:
        if isinstance(match, dict) and match.get("action") in _REMOVAL_ACTIONS:
            raise NeverDeleteError(
                f"coding record match carries a removal action: {match.get('action')!r}"
            )


def assert_no_deletion(before: dict, after: dict) -> None:
    """Structural never-delete post-condition, independent of how `after`
    was constructed: every entry id present in `before` must still be
    present in `after`, and every session recorded in an entry's sightings
    in `before` must still be present in that same entry's sightings in
    `after`. Retirement (status -> 'retired') is unaffected — it changes a
    field, not entry/sighting presence. Raises NeverDeleteError otherwise.
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


def apply_coding_record(codebook: dict, record: dict) -> tuple[dict, dict]:
    """Apply one §7.3 coding record to a v2 codebook. Sole-writer merge.

    Operates on a deep copy — never mutates `codebook` in place. Returns
    `(new_codebook, stats)` where stats has keys `matched`,
    `skipped_unknown_entry`, `candidates_applied`, `record_invalid`.

    A record that fails `validate_coding_record()` is skipped WHOLE (never
    partially applied): the input codebook comes back unchanged (deep
    copy) and `stats['record_invalid']` is True.

    Match handling: an entry_id that doesn't resolve to an existing entry
    is counted as skipped and never fabricated into a new entry — only the
    census promotes candidates to entries. A sighting is appended to the
    matched entry only if no existing sighting on that entry already
    carries this session (dedup key: (session, entry_id), the entry being
    implicit in "that entry's sightings list").

    Never-delete: a deletion-shaped `record` (top-level delete/remove/
    retract/drop key, or a match with a delete/remove action) raises
    NeverDeleteError immediately, before `codebook` is even copied.
    `assert_no_deletion()` is also run as a construction-independent
    safety net just before returning.
    """
    _reject_deletion_directive(record)

    stats = {
        "matched": 0,
        "skipped_unknown_entry": 0,
        "candidates_applied": 0,
        "record_invalid": False,
    }

    result = copy.deepcopy(codebook)

    record_errors = validate_coding_record(record)
    if record_errors:
        stats["record_invalid"] = True
        assert_no_deletion(codebook, result)
        return result, stats

    session = record["session"]
    date = record["date"]
    project = record["project"]

    entries_by_id = {e["id"]: e for e in result.get("entries", [])}

    for match in record.get("matches", []) or []:
        entry = entries_by_id.get(match.get("entry_id"))
        if entry is None:
            stats["skipped_unknown_entry"] += 1
            continue
        sightings = entry.setdefault("sightings", [])
        existing_sessions = {s.get("session") for s in sightings}
        if session in existing_sessions:
            continue
        sightings.append(_build_sighting(match, session=session, date=date, project=project))
        stats["matched"] += 1

    candidates = result.setdefault("candidates", [])
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
        stats["candidates_applied"] += 1

    assert_no_deletion(codebook, result)
    return result, stats


def validate_coding_record(record: dict) -> list[str]:
    """Validate a §7.3 coding record (coder output -> merger input).

    Returns a list of human-readable error strings; an empty list means the
    record is valid. Required: session/date/project/agent_class (strings).
    matches/candidates are optional lists; invariant_violated accepts null
    or any string (not slug-checked — see PHASES/HEADER module docstring).
    """
    return _schema_errors(record, CODING_RECORD_SCHEMA)


# ---------------------------------------------------------------------------
# load / dump — deterministic canonical YAML I/O
# ---------------------------------------------------------------------------

def load(path: str | os.PathLike) -> dict:
    """Load a codebook YAML file into a plain dict via yaml.safe_load."""
    with open(path, "r", encoding="utf-8") as f:
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
    """
    body = yaml.safe_dump(
        codebook,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=4096,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER)
        f.write(body)


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
    defaults to [] (only if absent). Output passes validate().
    """
    result = copy.deepcopy(codebook)
    result["version"] = 2

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
    migrated = migrate_v1_to_v2(codebook)
    errors = validate(migrated)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    dump(migrated, args.path)
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    codebook = load(args.codebook)
    totals = {
        "matched": 0,
        "skipped_unknown_entry": 0,
        "candidates_applied": 0,
        "record_invalid": 0,
    }

    with open(args.records, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            codebook, stats = apply_coding_record(codebook, record)
            totals["matched"] += stats["matched"]
            totals["skipped_unknown_entry"] += stats["skipped_unknown_entry"]
            totals["candidates_applied"] += stats["candidates_applied"]
            totals["record_invalid"] += int(stats["record_invalid"])

    errors = validate(codebook)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    dump(codebook, args.codebook)
    print(
        "applied: matched={matched} candidates_applied={candidates_applied} "
        "skipped_unknown_entry={skipped_unknown_entry} "
        "record_invalid={record_invalid}".format(**totals)
    )
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
    apply_parser.add_argument("codebook")
    apply_parser.add_argument("records")
    apply_parser.set_defaults(func=_cmd_apply)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
