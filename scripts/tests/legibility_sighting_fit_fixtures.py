"""The reify regression codebook for the sighting fit rule (task 6787): entry
entry-cand-20260926-5 as it stood in reify's codebook, withheld with one
verified founding session and three coder ``matches`` that do not exhibit its
mechanism.

A plain module rather than a conftest fixture, like
``legibility_carrier_fixtures``: conftest appends this directory to
``sys.path``, so the test files import it by bare name. Each function returns
a fresh codebook.
"""
from __future__ import annotations

ENTRY_ID = "entry-cand-20260926-5"
CANDIDATE_ID = "cand-20260926-5"
TITLE = (
    "Guessed scripts subdirectory probed with stderr suppressed, "
    "so a nonexistent path surfaces only as a bare exit 2"
)
CAUSE = (
    "The agent chained a directory listing with a guessed subdirectory "
    "(scripts/legibility) and redirected stderr to /dev/null. The path does not "
    "exist, so the call returned exit 2 with no error message. The failure showed "
    "up as a tool_error with no explanation, and the same kind of missing-path "
    "error recurred later in the session (not-found signals at turns 50 and 111)."
)

FOUNDING_SESSION = "b91ec332-5092-4a0b-9a92-51cf37f2ae1d"
FOUNDING_EVIDENCE = (
    r"ls scripts | head -200 | tr '\n' ' '; echo; ls scripts/legibility 2>/dev/null"
    " -> [exit 2] Exit code 2"
)

DATA_PROBE_SESSION = "ec5077fb-2d01-4cd0-a78e-4ac101f39339"
SIDECAR_PROBE_SESSION = "0a3c2225-030d-428a-abc4-0e465bd52eb9"
DOCS_PROBE_SESSION = "16ffd98f-72bb-4eea-83df-3a4ede488aa0"
MISATTACHED_SESSIONS = (DATA_PROBE_SESSION, SIDECAR_PROBE_SESSION, DOCS_PROBE_SESSION)

MISATTACHED_NOTES = {
    DATA_PROBE_SESSION: (
        "Agent probes data directory (ls data 2>/dev/null) and git check-ignore with "
        "stderr suppressed; path-not-found errors surface only as bare exit codes "
        "without diagnostic context"
    ),
    SIDECAR_PROBE_SESSION: (
        "Turn 163: Probed path 'gui/src-tauri/sidecar/' surfaces only as bare exit 2 "
        "with stderr redirected (2>&1), concealing 'No such file or directory' "
        "diagnostic detail from the exit code alone"
    ),
    DOCS_PROBE_SESSION: (
        "A guessed docs path was probed with stderr suppressed, and the nonexistent "
        "path surfaced only as a bare exit 2 from ls after a 2>/dev/null grep."
    ),
}

MISATTACHED_EVIDENCE = {
    DATA_PROBE_SESSION: (
        "ls data 2>/dev/null | head; git check-ignore -v data/x 2>/dev/null"
        " -> [exit 1] Exit code 1"
    ),
    SIDECAR_PROBE_SESSION: (
        "ls gui/src-tauri/sidecar/ 2>&1"
        " -> ls: cannot access 'gui/src-tauri/sidecar/': No such file or directory"
    ),
    DOCS_PROBE_SESSION: (
        "ls /home/leo/src/reify/docs/task-authoring.md "
        "/home/leo/src/dark-factory/docs/task-authoring.md 2>&1"
    ),
}

_MISATTACHED_HEADERS = (
    (DATA_PROBE_SESSION, "2026-10-04", "unknown"),
    (SIDECAR_PROBE_SESSION, "2026-10-04", "verify"),
    (DOCS_PROBE_SESSION, "2026-10-08", "unknown"),
)


def _founding_sighting(project: str) -> dict:
    return {
        "date": "2026-09-26",
        "project": project,
        "session": FOUNDING_SESSION,
        "origin_phase": "unknown",
        "manifested_phase": "unknown",
        "evidence_quote": FOUNDING_EVIDENCE,
    }


def _misattached_sightings(project: str, *, quoted: bool) -> list[dict]:
    sightings = []
    for session, date, phase in _MISATTACHED_HEADERS:
        sighting = {
            "date": date,
            "project": project,
            "session": session,
            "origin_phase": phase,
            "manifested_phase": phase,
            "note": MISATTACHED_NOTES[session],
        }
        if quoted:
            sighting["evidence_quote"] = MISATTACHED_EVIDENCE[session]
        sightings.append(sighting)
    return sightings


def _codebook(project: str, *, quoted: bool) -> dict:
    return {
        "version": 2,
        "entries": [
            {
                "id": ENTRY_ID,
                "title": TITLE,
                "severity": "medium",
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "filing": "withheld",
                "sightings": [
                    _founding_sighting(project),
                    *_misattached_sightings(project, quoted=quoted),
                ],
            }
        ],
        "candidates": [
            {
                "id": CANDIDATE_ID,
                "title": TITLE,
                "cause": CAUSE,
                "area": "tool-use / path discovery",
                "first_seen": "2026-09-26",
                "disposition": "promoted",
                "sightings": [_founding_sighting(project)],
                "promoted_to": ENTRY_ID,
            }
        ],
    }


def reify_codebook_at_rest(*, project: str = "reify") -> dict:
    """The entry as reify's codebook holds it: the three mis-attached
    sightings carry only the coder's note."""
    return _codebook(project, quoted=False)


def reify_codebook_with_quotes(*, project: str = "reify") -> dict:
    """The same entry, with each mis-attached sighting also quoting the
    command its session actually ran."""
    return _codebook(project, quoted=True)
