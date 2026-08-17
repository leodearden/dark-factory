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
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from legibility import check_transcript_persistence as mod
from legibility.config import load_config

from orchestrator import session_registry

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FIXED_NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)


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


def test_find_matching_transcript_tolerates_whitespace_reflow(tmp_path):
    # A transcript that stores the SAME prompt words but with reflowed
    # whitespace (runs of spaces / newlines instead of single spaces) must
    # still match — the STRONG containment check normalizes whitespace on both
    # sides, so a raw substring mismatch does not FALSE-POSITIVE a present
    # transcript as MISSING (reviewer_comprehensive/correctness, task 2893).
    projects = tmp_path / "projects"
    cwd = "/home/leo/src/dark-factory/.worktrees/2500"
    rec = _spawn_record("sess-reflow", cwd, _USABLE_PROMPT)

    reflowed = "   ".join(_USABLE_PROMPT.split())  # single spaces -> triple spaces
    assert reflowed != _USABLE_PROMPT  # the raw substring check would miss this
    match_path = _write_transcript(projects, cwd, "sess-reflow.jsonl", reflowed)

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got == match_path


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
# task 3272: underscore cwds must resolve to their REAL on-disk encoded dir
#
# ``find_matching_transcript`` uses ``inventory.encode_cwd`` as a DIRECT
# LOOKUP KEY (``session_dir.is_dir()`` -> None), with no ``is_member``
# re-check to save it — unlike inventory.py's two superset pre-filters. So an
# encoder that drops a character silently turns every present transcript of
# an underscore-bearing member cwd into a FALSE-POSITIVE "session ran, no
# transcript" finding, which escalates, and reports a non-existent
# ``expected_dir`` to the human reading it.
#
# The fixture dir below is a HARD-CODED literal copied from a real
# ``~/.claude/projects`` entry. It is deliberately NOT built via
# ``_write_transcript`` (which derives its dir through the encoder under
# test): a fixture encoded by the buggy function lands in the same wrong
# place the lookup looks, so the assertion passes vacuously. That is exactly
# how this divergence survived a fully green suite.
# ---------------------------------------------------------------------------

_UNDERSCORE_MEMBER_CWD = "/home/leo/src/dark-factory/.eval-worktrees/df_task_12/run-5383f6a8"
_UNDERSCORE_MEMBER_DIR = "-home-leo-src-dark-factory--eval-worktrees-df-task-12-run-5383f6a8"


def _write_transcript_in_literal_dir(
    projects_root: Path,
    encoded_dir: str,
    name: str,
    first_user_text: str,
    *,
    cwd: str | None = None,
) -> Path:
    """Write a transcript into an EXPLICITLY NAMED encoded dir.

    Sibling of :func:`_write_transcript` that takes the encoded dir name as a
    literal instead of deriving it via ``inventory.encode_cwd``, so the
    fixture cannot track a bug in the encoder it is meant to test.

    When *cwd* is given it is recorded on the user line, the way a real
    transcript carries its session's REAL cwd. That is the field
    ``inventory.session_cwd`` reads, and the only evidence
    ``mod.resolve_session_dir``'s degrade path will accept as confirmation
    that an unexpectedly-named dir actually belongs to a cwd.
    """
    session_dir = projects_root / encoded_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / name
    line: dict = {"type": "user", "message": {"content": first_user_text}}
    if cwd is not None:
        line["cwd"] = cwd
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    return path


def test_find_matching_transcript_resolves_underscore_cwd(tmp_path):
    projects = tmp_path / "projects"
    rec = _spawn_record("sess-underscore", _UNDERSCORE_MEMBER_CWD, _USABLE_PROMPT)

    match_path = _write_transcript_in_literal_dir(
        projects, _UNDERSCORE_MEMBER_DIR, "sess-underscore.jsonl", _USABLE_PROMPT
    )

    got = mod.find_matching_transcript(
        rec, projects, now=FIXED_NOW, skew=timedelta(hours=6),
    )
    assert got == match_path


def test_find_missing_transcripts_no_false_positive_for_underscore_cwd(tmp_path):
    # The live impact: a member session whose transcript IS on disk must not
    # be reported MISSING just because its cwd contains an underscore.
    projects = tmp_path / "projects"
    rec = _spawn_record("sess-underscore", _UNDERSCORE_MEMBER_CWD, _USABLE_PROMPT)

    _write_transcript_in_literal_dir(
        projects, _UNDERSCORE_MEMBER_DIR, "sess-underscore.jsonl", _USABLE_PROMPT
    )

    findings = mod.find_missing_transcripts(
        [rec], projects, ["/home/leo/src/dark-factory"],
        now=FIXED_NOW, lookback=timedelta(hours=48),
    )

    assert findings == []


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


def test_payload_exports_force_persistence_false_when_commented_out():
    # A commented-out / documentary reference must NOT satisfy the guard — the
    # match is anchored to an assignment context, not an unanchored substring
    # (reviewer_comprehensive/robustness, task 2893).
    script = (
        "#!/usr/bin/env bash\n"
        "# do NOT set CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1 in the child env\n"
        "  ## CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1  (historical note)\n"
        'claude --prompt "$PROMPT"\n'
    )
    assert mod.payload_exports_force_persistence(script) is False


def test_payload_exports_force_persistence_false_when_embedded_in_identifier():
    # The token embedded in a LARGER identifier is not a real assignment.
    assert mod.payload_exports_force_persistence(
        "MY_CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1\n"
    ) is False


def test_payload_exports_force_persistence_false_when_space_after_equals():
    # `VAR= 1` is not a valid bash assignment; the `\\s*`-after-`=` gap is gone.
    assert mod.payload_exports_force_persistence(
        "export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE= 1\n"
    ) is False


def test_payload_exports_force_persistence_true_for_indented_assignment():
    # A genuinely indented assignment (inside an if/block) still matches.
    assert mod.payload_exports_force_persistence(
        "if true; then\n    export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1\nfi\n"
    ) is True


def test_payload_exports_force_persistence_true_for_midline_after_semicolon():
    # A `;`-separated export mid-line (no line-start anchor) must still match
    # (task 2923 regression: the line-start-only regex missed this).
    assert mod.payload_exports_force_persistence(
        "trap 'exit 143' TERM; export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1; "
        "cd /tmp && claude\n"
    ) is True


def test_payload_exports_force_persistence_true_for_midline_after_interpolation():
    # Mirrors skills/spawn/spawn-claude.sh:350's real construction: prior
    # `${..._export}` vars (each either empty or `export VAR=val; `)
    # concatenate directly before `export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1`
    # inside an eval'd string — neither at line start nor preceded by
    # whitespace, just a `}` boundary from the preceding interpolation
    # (task 2923).
    assert mod.payload_exports_force_persistence(
        'inner="trap foo EXIT; '
        '${spawn_id_export}${parent_id_export}${result_export}${wm_title_export}'
        'export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1; cd $q_cwd && claude '
        '$flags $q_prompt; ec=\\$?; exit \\$ec"\n'
    ) is True


def test_payload_exports_force_persistence_false_for_midline_without_export_keyword():
    # Mid-line WITHOUT the `export` keyword must still be rejected — the
    # mandatory-`export` branch exists precisely so a bare mid-line `VAR=1`
    # substring embedded in prose can't false-positive.
    assert mod.payload_exports_force_persistence(
        "trap foo; CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1; cd /tmp && claude\n"
    ) is False


def test_payload_exports_force_persistence_false_for_commented_midline_continuation():
    # A comment continuing mid-line after a semicolon (a space, not a
    # boundary char, precedes the token) must not match either.
    assert mod.payload_exports_force_persistence(
        "cd /tmp; # do not export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1 here\n"
    ) is False


def test_payload_exports_force_persistence_false_for_quoted_echo():
    # A merely-quoted/echoed token must NOT satisfy the guard — `"`/`'` are
    # deliberately excluded from the mid-line boundary set, since including
    # them would match an `echo "export ...=1"` that never actually exports
    # anything (reviewer_comprehensive/robustness, task 2923 amendment).
    assert mod.payload_exports_force_persistence(
        'echo "export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1"\n'
    ) is False
    assert mod.payload_exports_force_persistence(
        "echo 'export CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1'\n"
    ) is False


# ---------------------------------------------------------------------------
# step-11/12: escalation envelope + best-effort poster
# ---------------------------------------------------------------------------

def _missing_finding(
    slug: str = "sess-lost",
    cwd: str = "/home/leo/src/dark-factory/.worktrees/2701",
) -> mod.MissingTranscript:
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


class _FakeHttpxResponse:
    """A 200 `application/json` reply with an empty JSON-RPC body.

    `status_code`, `headers` and `json()` were added for task 3644:
    `_default_poster` now delegates to `census_trigger.post_mcp_envelope`,
    which reads `status_code` (to detect the stateful escalation server's 400
    "Missing session ID") and `content-type` (to detect an SSE-framed body)
    before decoding. Without them this fake would die on an AttributeError
    and stop exercising the real code path -- so the task-2953 header
    contract below keeps being asserted against the transport that actually
    ships, not against a fake the implementation has outgrown."""

    status_code = 200
    headers = {"content-type": "application/json"}

    def raise_for_status(self):
        pass

    def json(self):
        return {}


def test_default_poster_sends_streamable_http_accept_headers(install_fake_httpx):
    """Task 2953: the streamable-HTTP MCP transport 406s any tools/call POST
    lacking an Accept header covering both application/json and
    text/event-stream (verified live against a local MCP /mcp endpoint).
    `_default_poster`'s httpx import is lazy, but httpx IS importable here --
    a direct dependency of `shared` (shared/pyproject.toml, `httpx>=0.27`,
    task 2965) -- so an un-faked call would really hit the network. The
    shared `install_fake_httpx` fixture substitutes a stub so the outbound
    request shape is assertable independent of any live listener.
    Mirrors nightly.py's identical test for its own `_default_poster`."""
    captured_kwargs = {}

    def _fake_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeHttpxResponse()

    install_fake_httpx(_fake_post)

    mod._default_poster('http://localhost:8199/mcp', {'jsonrpc': '2.0'})

    headers = captured_kwargs.get('headers') or {}
    assert 'application/json' in headers.get('Accept', '')
    assert 'text/event-stream' in headers.get('Accept', '')
    # Content-Type is part of the same transport contract -- pin it too so a
    # future edit dropping it can't pass on the Accept assertions alone.
    assert headers.get('Content-Type') == 'application/json'
    # The envelope must still ride along unchanged on the same POST.
    assert captured_kwargs.get('json') == {'jsonrpc': '2.0'}


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


def test_run_check_enriches_escalation_detail_with_preventer_verdict(tmp_path):
    # run_check with BOTH lost findings AND --check-preventer: the escalation
    # detail is enriched with the preventer verdict via force_persistence_ok,
    # so the loud alarm names the known regression cause when it is present
    # (reviewer_comprehensive/test-coverage, task 2893).
    fleet, projects, config_path = _build_fleet_and_projects(tmp_path, include_lost=True)
    poster = _RecordingPoster()
    missing_token = tmp_path / "spawn-no-token.sh"
    missing_token.write_text(
        "#!/usr/bin/env bash\nexport CLAUDE_CODE_CHILD_SESSION=1\nclaude\n",
        encoding="utf-8",
    )

    result = mod.run_check(
        config_path=config_path, fleet_root=fleet, projects_root=projects,
        now=FIXED_NOW, lookback=timedelta(hours=48), poster=poster,
        check_preventer=True, spawn_script_path=missing_token,
    )

    assert result.exit_code == 1
    assert result.force_persistence_ok is False
    assert len(poster.calls) == 1
    _url, envelope = poster.calls[0]
    detail = envelope["params"]["arguments"]["detail"]
    assert "preventer guard" in detail
    # The verdict for a missing token is the loud 'MISSING' marker.
    assert "MISSING" in detail


def test_main_errors_when_no_config_or_project_id():
    # Neither --config nor --project-id -> parser.error -> SystemExit(2).
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--lookback-hours", "48"])
    assert excinfo.value.code == 2


def test_load_config_via_project_id_lazy_resolution(tmp_path, monkeypatch):
    # The --project-id branch of _load_config lazily imports
    # nightly.resolve_config_path and loads the resolved legibility.yaml.
    config_path = _write_config(tmp_path, project_id="proj_lazy")
    from legibility import nightly

    monkeypatch.setattr(nightly, "resolve_config_path", lambda pid: config_path)

    cfg = mod._load_config(config_path=None, project_id="proj_lazy")
    assert cfg.project_id == "proj_lazy"


def test_load_config_requires_config_or_project_id():
    # Neither given at the function boundary -> ValueError (main() guards this
    # earlier via parser.error, but the loader is defensive too).
    with pytest.raises(ValueError):
        mod._load_config(config_path=None, project_id=None)


def test_read_spawn_script_unreadable_returns_empty(tmp_path):
    # A missing/unreadable spawn script fails SAFE: '' -> guard False, i.e. a
    # missing script is NOT evidence the preventer is in place
    # (reviewer_comprehensive/test-coverage, task 2893).
    missing = tmp_path / "does-not-exist.sh"
    assert mod._read_spawn_script(missing) == ""
    assert mod.payload_exports_force_persistence(mod._read_spawn_script(missing)) is False

    # A directory path also raises OSError on read_text -> same fail-safe.
    a_dir = tmp_path / "a_dir"
    a_dir.mkdir()
    assert mod._read_spawn_script(a_dir) == ""


# ---------------------------------------------------------------------------
# task 3644: the transcript-loss alarm must survive a STATEFUL server
# ---------------------------------------------------------------------------
#
# Same defect as nightly.py's and census.py's posters: `_default_poster`
# bare-POSTed `tools/call`, the escalation server rejects it at the transport
# layer with `400 Bad Request` / "Missing session ID" (captured live
# 2026-08-05), and `post_findings` swallows that best-effort and returns
# False -- so the transcript-loss alarm never reached anyone. Fixed by the
# same single-sourced transport in census_trigger.


class _FakeStatefulResponse:
    """An `httpx.Response` stand-in for the stateful-server handshake."""

    def __init__(self, *, status_code=200, headers=None, payload=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            # A plain RuntimeError, not httpx.HTTPStatusError: the shared
            # install_fake_httpx stub exposes only `post` and pytest.fails on
            # any other attribute (task 3376).
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


def _stateful_escalation_post(recorded, *, session_id="sid-transcript"):
    """A fake `httpx.post` behaving like the STATEFUL escalation server:
    session-less `tools/call` -> 400; `initialize` -> 200 + the assigned id
    as a response header; `notifications/initialized` -> 202; `tools/call`
    WITH that header -> 200."""
    def _post(url, **kwargs):
        recorded.append((url, kwargs))
        envelope = kwargs.get("json") or {}
        method = envelope.get("method")
        if method == "initialize":
            return _FakeStatefulResponse(
                headers={"mcp-session-id": session_id,
                         "content-type": "application/json"},
                payload={"jsonrpc": "2.0", "id": 1, "result": {}},
            )
        if method == "notifications/initialized":
            return _FakeStatefulResponse(status_code=202)
        if (kwargs.get("headers") or {}).get("mcp-session-id") != session_id:
            return _FakeStatefulResponse(
                status_code=400,
                headers={"content-type": "application/json"},
                payload={"jsonrpc": "2.0", "id": "server-error",
                         "error": {"code": -32600,
                                   "message": "Bad Request: Missing session ID"}},
            )
        return _FakeStatefulResponse(
            headers={"content-type": "application/json"},
            payload={"jsonrpc": "2.0", "id": 1,
                     "result": {"structuredContent": {"id": "esc-11", "status": "queued"}}},
        )

    return _post


def _recording_delete(deleted):
    """A fake `httpx.delete` recording the MCP session-termination call the
    transport makes after a handshake, so the long-lived escalation server does
    not leak a session per transcript-loss alarm."""
    def _delete(url, **kwargs):
        deleted.append((url, kwargs))
        return _FakeStatefulResponse(status_code=200)

    return _delete


def test_post_findings_lands_against_a_stateful_server(tmp_path, install_fake_httpx):
    """The DEFAULT poster (no `poster=` injection) must handshake and land.

    `post_findings` -- NOT `post_escalation`; this module's escalation
    entrypoint takes a `Sequence[MissingTranscript]` -- so a minimal non-empty
    findings list is what reaches the POST."""
    cfg = load_config(_write_config(tmp_path, project_id="proj_a"))
    recorded = []
    deleted = []
    install_fake_httpx(
        _stateful_escalation_post(recorded), delete=_recording_delete(deleted),
    )

    assert mod.post_findings(cfg, [_missing_finding("sess-lost-1")]) is True

    methods = [(kwargs.get("json") or {}).get("method") for _url, kwargs in recorded]
    assert methods == [
        "tools/call", "initialize", "notifications/initialized", "tools/call",
    ]
    # The retried call carries the server-assigned session id...
    assert (recorded[-1][1].get("headers") or {}).get("mcp-session-id") == "sid-transcript"
    # ...and is still the escalate_info the alarm meant to file.
    params = (recorded[-1][1].get("json") or {})["params"]
    assert params["name"] == "escalate_info"
    assert "sess-lost-1" in params["arguments"]["detail"]
    # ...and the session opened to file it is released again.
    assert [(kw.get("headers") or {}).get("mcp-session-id") for _u, kw in deleted] == [
        "sid-transcript"
    ]


def test_post_findings_reports_false_on_a_tool_error_envelope(
    tmp_path, install_fake_httpx, caplog
):
    """HTTP 200 is not success. `post_findings` discards the response body and
    reports True on "no exception", so a tool-level failure
    (`result.isError: true`) would otherwise read as a landed alarm -- the same
    green-on-paper/nothing-filed failure task 3644 exists to close, one layer
    up from the transport."""
    cfg = load_config(_write_config(tmp_path, project_id="proj_a"))

    def _post(url, **kwargs):
        return _FakeStatefulResponse(
            headers={"content-type": "application/json"},
            payload={"jsonrpc": "2.0", "id": 1, "result": {
                "isError": True,
                "content": [{"type": "text", "text": "unknown category 'nope'"}],
            }},
        )

    install_fake_httpx(_post)

    with caplog.at_level(logging.WARNING):
        assert mod.post_findings(cfg, [_missing_finding("sess-lost-1")]) is False

    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("escalation post failed" in m for m in warned), warned
    assert any("isError" in m or "reported an error" in m for m in warned), warned
