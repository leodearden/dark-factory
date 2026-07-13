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

import os
import re

import jsonschema
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


def _schema_errors(instance: dict, schema: dict) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(instance), key=str):
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{location}: {err.message}")
    return errors


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
