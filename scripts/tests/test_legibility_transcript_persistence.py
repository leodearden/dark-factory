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
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from legibility import check_transcript_persistence as mod
from legibility.config import load_config
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


# ---------------------------------------------------------------------------
# step-3/4: STRONG prompt-prefix matching (find_matching_transcript)
# ---------------------------------------------------------------------------

_USABLE_PROMPT = (
    "Investigate the flaky merge on task 2500 and report your findings "
    "back to the orchestrator."
)


def _spawn_record(slug: str, cwd: str, prompt: str, *, start_offset_hours: int = 1):
    return session_registry.SessionRecord(
        session_slug=slug,
        status=session_registry.Status.EXITED,
        prompt=prompt,
        cwd=cwd,
        start_ts=_iso(FIXED_NOW - timedelta(hours=start_offset_hours)),
        exit_code=0,
    )


def test_find_matching_transcript_strong_prompt_prefix(tmp_path):
    projects = tmp_path / "projects"
    cwd = "/home/leo/src/dark-factory/.worktrees/2500"
    rec = _spawn_record("sess-strong", cwd, _USABLE_PROMPT)

    match_path = _write_transcript(projects, cwd, "sess-strong.jsonl", _USABLE_PROMPT)

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got == match_path


def test_find_matching_transcript_sibling_only_returns_none(tmp_path):
    # Same-cwd confound: a SIBLING headless-agent transcript in the SAME
    # encoded-cwd dir, with a DIFFERENT first user turn, must NOT satisfy the
    # per-session match — the usable-prompt record is correctly flagged MISSING.
    projects = tmp_path / "projects"
    cwd = "/home/leo/src/dark-factory/.worktrees/2500"
    rec = _spawn_record("sess-missing", cwd, _USABLE_PROMPT)

    _write_transcript(
        projects, cwd, "sibling-agent.jsonl",
        "You are a TDD implementer. Execute the structured plan step by step.",
    )

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got is None


# ---------------------------------------------------------------------------
# step-5/6: WEAK file-mtime time-window fallback (short/empty prompts only)
# ---------------------------------------------------------------------------

_SHORT_PROMPT = "continue"  # < MIN_MATCH_LEN -> not usable -> weak fallback


def test_weak_mtime_fallback_matches_in_window(tmp_path):
    projects = tmp_path / "projects"
    cwd = "/home/leo/src/dark-factory/.worktrees/2600"
    rec = _spawn_record("sess-short", cwd, _SHORT_PROMPT)  # start_ts = now - 1h

    path = _write_transcript(projects, cwd, "sess-short.jsonl", "unrelated first turn text")
    _set_mtime(path, FIXED_NOW - timedelta(hours=1))  # within [start-skew, now+skew]

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got == path


def test_weak_mtime_fallback_out_of_window_returns_none(tmp_path):
    projects = tmp_path / "projects"
    cwd = "/home/leo/src/dark-factory/.worktrees/2600"
    rec = _spawn_record("sess-short2", cwd, _SHORT_PROMPT)

    path = _write_transcript(projects, cwd, "sess-short2.jsonl", "unrelated first turn text")
    _set_mtime(path, FIXED_NOW - timedelta(hours=100))  # far before the window

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got is None


def test_weak_fallback_does_not_fire_for_usable_prompt(tmp_path):
    # A usable-prompt record whose ONLY transcript is a time-in-window sibling
    # that fails the content match must STILL return None — the weak fallback
    # is gated on a non-usable prompt, so the same-cwd confound stays defeated.
    projects = tmp_path / "projects"
    cwd = "/home/leo/src/dark-factory/.worktrees/2600"
    rec = _spawn_record("sess-usable", cwd, _USABLE_PROMPT)

    path = _write_transcript(
        projects, cwd, "sibling-inwindow.jsonl",
        "You are a TDD implementer. A different first turn.",
    )
    _set_mtime(path, FIXED_NOW - timedelta(hours=1))  # in the time window

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got is None


def _set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


# ---------------------------------------------------------------------------
# step-7/8: find_missing_transcripts end-to-end
# ---------------------------------------------------------------------------

def test_find_missing_transcripts_end_to_end(tmp_path):
    projects = tmp_path / "projects"
    project_prefix = "/home/leo/src/dark-factory"
    cwd_prefixes = [project_prefix]

    member_present = "/home/leo/src/dark-factory/.worktrees/2700"
    member_missing = "/home/leo/src/dark-factory/.worktrees/2701"
    foreign_cwd = "/home/leo/src/some-other-project"
    missing_prompt = (
        "Diagnose the stuck reconciliation on task 2701 and summarise the root cause."
    )

    # (a) completed in-window spawn WITH a matching transcript -> no finding.
    present = _spawn_record("sess-present", member_present, _USABLE_PROMPT)
    _write_transcript(projects, member_present, "sess-present.jsonl", _USABLE_PROMPT)

    # (b) completed in-window spawn with NO matching transcript -> one finding.
    missing = _spawn_record("sess-missing", member_missing, missing_prompt)

    # (c) in-flight (non-terminal) spawn with no transcript -> tolerated.
    inflight = session_registry.SessionRecord(
        session_slug="sess-inflight",
        status=session_registry.Status.RUNNING,
        prompt="An in-flight interactive session still running.",
        cwd=member_present,
        start_ts=_iso(FIXED_NOW - timedelta(hours=1)),
    )

    # (d) completed spawn whose cwd is a FOREIGN project -> excluded.
    foreign = _spawn_record(
        "sess-foreign", foreign_cwd,
        "A completed spawn in a different project entirely, definitely not a member.",
    )

    findings = mod.find_missing_transcripts(
        [present, missing, inflight, foreign], projects, cwd_prefixes,
        now=FIXED_NOW, lookback=timedelta(hours=48),
    )

    assert len(findings) == 1
    found = findings[0]
    assert found.session_slug == "sess-missing"
    assert found.cwd == member_missing
    assert found.start_ts == missing.start_ts
    assert found.exit_code == 0
    assert "Diagnose the stuck reconciliation" in found.prompt_prefix
    assert found.expected_dir == projects / mod.inventory.encode_cwd(member_missing)


def test_find_matching_transcript_missing_dir_returns_none(tmp_path):
    projects = tmp_path / "projects"
    projects.mkdir()
    rec = _spawn_record(
        "sess-nodir", "/home/leo/src/some-other-project",
        "A substantive prompt that is definitely long enough to be usable.",
    )

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got is None


# ---------------------------------------------------------------------------
# step-9/10: pure preventer guard — payload_exports_force_persistence
# (fixture strings ONLY — never the real committed spawn-claude.sh)
# ---------------------------------------------------------------------------

def test_payload_exports_force_persistence_true_for_export_form():
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1\n"
        'claude --prompt "$PROMPT"\n'
    )
    assert mod.payload_exports_force_persistence(script) is True


def test_payload_exports_force_persistence_true_for_plain_and_quoted_forms():
    assert mod.payload_exports_force_persistence(
        "CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1\n"
    ) is True
    assert mod.payload_exports_force_persistence(
        'export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE="1"\n'
    ) is True


def test_payload_exports_force_persistence_false_when_absent():
    script = (
        "#!/usr/bin/env bash\n"
        "export CLAUDE_CODE_CHILD_SESSION=1\n"
        'claude --prompt "$PROMPT"\n'
    )
    assert mod.payload_exports_force_persistence(script) is False


def test_payload_exports_force_persistence_false_when_zero():
    assert mod.payload_exports_force_persistence(
        "export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=0\n"
    ) is False


# ---------------------------------------------------------------------------
# step-11/12: escalation envelope + best-effort poster
# ---------------------------------------------------------------------------

def _missing_finding(
    slug: str = "sess-lost",
    cwd: str = "/home/leo/src/dark-factory/.worktrees/2701",
) -> "mod.MissingTranscript":
    return mod.MissingTranscript(
        session_slug=slug,
        cwd=cwd,
        prompt_prefix="Diagnose the stuck reconciliation on task 2701.",
        start_ts=_iso(FIXED_NOW - timedelta(hours=1)),
        exit_code=0,
        expected_dir=Path("/tmp/projects") / mod.inventory.encode_cwd(cwd),
    )


class _RecordingPoster:
    """A fake poster capturing (url, envelope) calls (never posts anything)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, envelope: dict) -> None:
        self.calls.append((url, envelope))


def test_build_escalation_arguments_shape(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id="proj_a"))
    findings = [
        _missing_finding("sess-lost-1", "/home/leo/src/dark-factory/.worktrees/2701"),
        _missing_finding("sess-lost-2", "/home/leo/src/dark-factory/.worktrees/2702"),
    ]

    args = mod._build_escalation_arguments(findings, cfg, force_persistence_ok=None)

    assert args["task_id"] == "legibility-transcript-check-proj_a"
    assert args["agent_role"] == "legibility-transcript-check"
    assert args["category"] == "infra_issue"
    assert args["severity"] == "info"
    # summary names the count of missing transcripts.
    assert "2" in args["summary"]
    # detail names each finding's slug AND cwd.
    assert "sess-lost-1" in args["detail"]
    assert "sess-lost-2" in args["detail"]
    assert "/home/leo/src/dark-factory/.worktrees/2701" in args["detail"]
    assert "/home/leo/src/dark-factory/.worktrees/2702" in args["detail"]


def test_post_findings_posts_envelope_and_returns_true(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id="proj_a", escalation_port=8271))
    findings = [_missing_finding()]
    poster = _RecordingPoster()

    ok = mod.post_findings(cfg, findings, poster=poster)

    assert ok is True
    assert len(poster.calls) == 1
    url, envelope = poster.calls[0]
    assert url == "http://localhost:8271/mcp"
    # A JSON-RPC tools/call envelope for the escalate_info tool.
    assert envelope["jsonrpc"] == "2.0"
    assert envelope["method"] == "tools/call"
    assert envelope["params"]["name"] == "escalate_info"
    assert envelope["params"]["arguments"]["agent_role"] == "legibility-transcript-check"
    assert envelope["params"]["arguments"]["task_id"] == "legibility-transcript-check-proj_a"


def test_post_findings_best_effort_swallows_poster_exception(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    findings = [_missing_finding()]

    def _raising(url: str, envelope: dict) -> None:
        raise RuntimeError("escalation server down")

    # Best-effort contract: a raising poster -> False, never propagates.
    ok = mod.post_findings(cfg, findings, poster=_raising)
    assert ok is False


# ---------------------------------------------------------------------------
# step-13/14: run_check integration + CheckResult + main()
# ---------------------------------------------------------------------------

def _build_fleet_and_projects(tmp_path, *, include_lost: bool):
    """Set up a tmp fleet_root + projects_root with a canonical record mix.

    Always present: a completed spawn WITH a matching transcript (no finding),
    an in-flight non-terminal spawn (tolerated), and a foreign-project
    completed spawn (excluded). When *include_lost*, also a completed
    member-project spawn with NO transcript (the one finding).

    Returns ``(fleet_root, projects_root, config_path)``.
    """
    fleet = tmp_path / "fleet"
    projects = tmp_path / "projects"
    member_present = "/home/leo/src/dark-factory/.worktrees/2700"
    member_lost = "/home/leo/src/dark-factory/.worktrees/2701"
    foreign_cwd = "/home/leo/src/some-other-project"

    # completed + transcript present (strong match) -> no finding.
    _write_session_record(
        fleet, "sess-present", status=session_registry.Status.EXITED,
        prompt=_USABLE_PROMPT, cwd=member_present,
        start_ts=FIXED_NOW - timedelta(hours=1),
    )
    _write_transcript(projects, member_present, "sess-present.jsonl", _USABLE_PROMPT)

    # in-flight (non-terminal) -> tolerated, no finding.
    _write_session_record(
        fleet, "sess-inflight", status=session_registry.Status.RUNNING,
        prompt="An in-flight interactive session still running right now.",
        cwd=member_present, start_ts=FIXED_NOW - timedelta(hours=1),
    )

    # foreign-project completed spawn -> excluded (not is_member).
    _write_session_record(
        fleet, "sess-foreign", status=session_registry.Status.EXITED,
        prompt="A completed spawn in a different project entirely, not a member here.",
        cwd=foreign_cwd, start_ts=FIXED_NOW - timedelta(hours=1),
    )

    if include_lost:
        _write_session_record(
            fleet, "sess-lost", status=session_registry.Status.EXITED,
            prompt=(
                "Diagnose the stuck reconciliation on task 2701 and summarise "
                "the root cause."
            ),
            cwd=member_lost, start_ts=FIXED_NOW - timedelta(hours=1),
        )

    config_path = _write_config(
        tmp_path, project_id="proj_a", escalation_port=8199,
        cwd_prefixes=["/home/leo/src/dark-factory"],
    )
    return fleet, projects, config_path


def test_run_check_flags_lost_and_escalates(tmp_path):
    fleet, projects, config_path = _build_fleet_and_projects(tmp_path, include_lost=True)
    poster = _RecordingPoster()

    result = mod.run_check(
        config_path=config_path, fleet_root=fleet, projects_root=projects,
        now=FIXED_NOW, lookback=timedelta(hours=48), poster=poster,
    )

    assert result.exit_code == 1
    assert [m.session_slug for m in result.missing] == ["sess-lost"]
    assert result.escalated is True
    assert len(poster.calls) == 1


def test_run_check_healthy_no_escalation(tmp_path):
    fleet, projects, config_path = _build_fleet_and_projects(tmp_path, include_lost=False)
    poster = _RecordingPoster()

    result = mod.run_check(
        config_path=config_path, fleet_root=fleet, projects_root=projects,
        now=FIXED_NOW, lookback=timedelta(hours=48), poster=poster,
    )

    assert result.exit_code == 0
    assert result.missing == []
    assert result.escalated is False
    assert poster.calls == []


def test_run_check_preventer_guard_toggles_force_persistence_ok(tmp_path):
    # A tmp spawn-script fixture MISSING the token -> force_persistence_ok False;
    # a fixture WITH the token -> True. Fixtures only — never the real
    # committed spawn-claude.sh (which does not carry the token on this branch).
    fleet, projects, config_path = _build_fleet_and_projects(tmp_path, include_lost=False)

    missing_token = tmp_path / "spawn-no-token.sh"
    missing_token.write_text(
        "#!/usr/bin/env bash\nexport CLAUDE_CODE_CHILD_SESSION=1\nclaude\n",
        encoding="utf-8",
    )
    present_token = tmp_path / "spawn-with-token.sh"
    present_token.write_text(
        "#!/usr/bin/env bash\nexport CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1\nclaude\n",
        encoding="utf-8",
    )

    missing_result = mod.run_check(
        config_path=config_path, fleet_root=fleet, projects_root=projects,
        now=FIXED_NOW, lookback=timedelta(hours=48), poster=_RecordingPoster(),
        check_preventer=True, spawn_script_path=missing_token,
    )
    assert missing_result.force_persistence_ok is False
    # A failed preventer guard is itself a loud (non-zero) signal.
    assert missing_result.exit_code == 1

    present_result = mod.run_check(
        config_path=config_path, fleet_root=fleet, projects_root=projects,
        now=FIXED_NOW, lookback=timedelta(hours=48), poster=_RecordingPoster(),
        check_preventer=True, spawn_script_path=present_token,
    )
    assert present_result.force_persistence_ok is True
    assert present_result.exit_code == 0


def test_main_healthy_returns_zero(tmp_path):
    # main() wiring: argparse -> run_check -> exit code, fully offline (no
    # findings -> no escalation POST, no --check-preventer -> no real-file read).
    fleet, projects, config_path = _build_fleet_and_projects(tmp_path, include_lost=False)

    rc = mod.main([
        "--config", str(config_path),
        "--fleet-root", str(fleet),
        "--projects-root", str(projects),
        "--lookback-hours", "48",
    ])
    assert rc == 0
