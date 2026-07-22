"""Tests for scripts/legibility/check_transcript_persistence.py — the
registry↔transcript reconciliation detector (task 2893).

The detector cross-checks fleet session-registry records
(``~/.claude/fleet/sessions/<slug>/record.json``) against
``~/.claude/projects`` transcripts and alarms when a COMPLETED
spawn-launched interactive session produced no plausibly-matching
transcript (the "session ran, no transcript" regression).

Every test injects a tmp ``fleet_root`` (so ``session_registry`` never
touches the real ``~/.claude`` tree), a tmp ``projects_root``, a FIXED
``now``, a written fixture ``legibility.yaml``, and a fake ``poster`` — no
real ``~/.claude``, no live escalation server, and no dependency on the
separately-landing ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE`` preventer.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from legibility import check_transcript_persistence as mod
from orchestrator import session_registry


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FIXED_NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _write_session_record(
    fleet_root: Path,
    slug: str,
    *,
    status: session_registry.Status,
    prompt: str,
    cwd: str = "/home/leo/src/dark-factory",
    start_ts: datetime | None = None,
    exit_code: int | None = 0,
) -> session_registry.SessionRecord:
    """Write a SessionRecord under *fleet_root* via the canonical writer."""
    rec = session_registry.SessionRecord(
        session_slug=slug,
        status=status,
        prompt=prompt,
        cwd=cwd,
        start_ts=_iso(start_ts) if start_ts is not None else _iso(FIXED_NOW),
        exit_code=exit_code,
    )
    session_registry.write_record(rec, root=fleet_root)
    return rec


def _write_config(
    root: Path,
    *,
    project_id: str = "proj_a",
    escalation_port: int = 8199,
    cwd_prefixes=None,
) -> Path:
    """Write a minimal valid docs/legibility/legibility.yaml under *root*."""
    cwd_prefixes = cwd_prefixes if cwd_prefixes is not None else [str(root / "work")]
    legibility_dir = root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    config_path = legibility_dir / "legibility.yaml"
    lines = [
        f"project_id: {project_id}",
        f"project_root: {root}",
        f"escalation_port: {escalation_port}",
        "cwd_prefixes:",
    ]
    lines += [f"  - {prefix}" for prefix in cwd_prefixes]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _write_transcript(projects_root: Path, cwd: str, name: str, first_user_text: str) -> Path:
    """Write a *.jsonl transcript under the encoded-cwd dir for *cwd*.

    The transcript's first non-sidechain/non-meta user turn carries
    *first_user_text* — the signal ``find_matching_transcript`` matches a
    record's prompt prefix against.
    """
    enc = mod.inventory.encode_cwd(cwd)
    session_dir = projects_root / enc
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / name
    lines = [
        json.dumps({"type": "user", "message": {"content": first_user_text}}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# step-1/2: registry enumeration + completed-spawn filtering
# ---------------------------------------------------------------------------

def test_find_completed_spawn_records_filters(tmp_path):
    fleet = tmp_path / "fleet"

    # (KEEP) a completed EXITED spawn with a usable prompt, in window.
    _write_session_record(
        fleet, "good-exited", status=session_registry.Status.EXITED,
        prompt="Investigate the flaky merge on task 2500 and report back.",
        start_ts=FIXED_NOW - timedelta(hours=1),
    )
    # (KEEP) FAILED_TO_START is also terminal.
    _write_session_record(
        fleet, "good-failed", status=session_registry.Status.FAILED_TO_START,
        prompt="Run the unblock skill for df#2085.",
        start_ts=FIXED_NOW - timedelta(hours=5),
    )
    # (EXCLUDE) empty prompt.
    _write_session_record(
        fleet, "empty-prompt", status=session_registry.Status.EXITED,
        prompt="", start_ts=FIXED_NOW - timedelta(hours=1),
    )
    # (EXCLUDE) non-terminal (RUNNING) — transcript may still be flushing.
    _write_session_record(
        fleet, "still-running", status=session_registry.Status.RUNNING,
        prompt="A long-running interactive session.",
        start_ts=FIXED_NOW - timedelta(hours=1),
    )
    # (EXCLUDE) terminal but older than the 48h lookback window.
    _write_session_record(
        fleet, "too-old", status=session_registry.Status.EXITED,
        prompt="An old completed session outside the window.",
        start_ts=FIXED_NOW - timedelta(hours=72),
    )

    # (TOLERATE) a corrupt record.json — skipped, no raise.
    corrupt_dir = fleet / "sessions" / "corrupt-sess"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "record.json").write_text("{not valid json", encoding="utf-8")
    # (TOLERATE) an empty slug dir with no record.json — skipped, no raise.
    (fleet / "sessions" / "empty-dir").mkdir(parents=True)

    kept = mod.find_completed_spawn_records(
        mod.iter_registry_records(fleet_root=fleet),
        now=FIXED_NOW,
        lookback=timedelta(hours=48),
    )

    assert {r.session_slug for r in kept} == {"good-exited", "good-failed"}
