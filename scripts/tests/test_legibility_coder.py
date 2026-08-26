"""Tests for scripts/legibility/coder.py — the Haiku trickle coder (digest
-> strict-JSON §7.3 coding record).

See plans/confusion-reduction-prd.md §7.3 (coding record contract), §8.1
(consumer-side boundary test), §8.6 (coder failure storm). Task delta of
the confusion-reduction PRD decomposition.

Import the target module as `import coder as mod`, resolved via
scripts/tests/conftest.py's sys.path insertion (both scripts/ and
scripts/legibility/ are on sys.path; no package __init__ needed) — mirrors
test_legibility_digest.py's import style.

The LLM is ALWAYS mocked in this file: every code_digest/code_digests call
below injects a fake `invoke` callable (or monkeypatches mod._invoke_cli)
rather than ever shelling out to a real `claude` process.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

import codebook as codebook_mod
import coder as mod
import digest as digest_mod
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_CODEBOOK_PATH = _REPO_ROOT / "docs" / "legibility" / "confusion-codebook.yaml"

# ---------------------------------------------------------------------------
# Shared fixture helpers — synthetic transcript -> real digest text, mirrors
# test_legibility_digest.py's helper shapes (own copies, per this repo's
# convention of each test file owning its scaffolding, e.g. test_codebook.py's
# _minimal_v2()).
# ---------------------------------------------------------------------------

_SESSION_ID = "cafe1234-0000-4000-8000-000000000000"
_CWD = "/home/leo/src/dark-factory"
_TIMESTAMP = "2026-07-14T06:02:29.796Z"


def _user_text(s, *, cwd=_CWD, session_id=_SESSION_ID, timestamp=_TIMESTAMP):
    """Build a synthetic genuine (non-meta, non-sidechain) human user turn
    carrying real-transcript-shaped top-level cwd/sessionId/timestamp
    fields (observed on every non-queue-operation record in a real Claude
    Code transcript)."""
    return {
        "type": "user",
        "message": {"role": "user", "content": s},
        "isSidechain": False,
        "isMeta": False,
        "cwd": cwd,
        "sessionId": session_id,
        "timestamp": timestamp,
    }


def _write_jsonl(tmp_path, records, name="transcript.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")
    return path


def _build_digest_text(tmp_path, *, session_id=_SESSION_ID, agent_class="interactive", name="transcript.jsonl"):
    """Build a real digest string via digest.build_digest on a minimal
    synthetic single-user-turn transcript — session/date/agent_class in
    the resulting frontmatter are deterministic from the inputs given
    here."""
    records = [_user_text("Please fix this, it is wrong.", session_id=session_id)]
    path = _write_jsonl(tmp_path, records, name=name)
    return digest_mod.build_digest(path, agent_class_override=agent_class)


def _hand_digest(session_id, body_marker, *, date="2026-07-14", agent_class="interactive"):
    """Hand-written minimal digest text: a leading frontmatter block with
    exactly the fields code_digest reads (session/date/agent_class) plus a
    one-line body. Used where a test's focus is code_digest's own logic
    (never-fabricate handling, batch tallying), not digest.py's rendering
    fidelity — that round trip is separately covered by
    _build_digest_text (steps 3/9/17, per plan.json reuse item 2)."""
    return (
        "---\n"
        f'session: "{session_id}"\n'
        f'date: "{date}"\n'
        f'agent_class: "{agent_class}"\n'
        "---\n\n"
        f"## User Corrections\n- {body_marker}\n"
    )


def _tiny_codebook():
    """A minimal but real v2 codebook fixture, distinct from the live
    docs/legibility/confusion-codebook.yaml used by the step-9 happy-path
    test — used where a test's focus doesn't require grounding against
    the live entry set (never-fabricate, batch tallying, CLI plumbing)."""
    return {
        "version": 2,
        "entries": [
            {
                "id": "one-shot-subagent-contract",
                "title": "Silent no-op one-shot subagent contracts",
                "cause": "Sub-agents are given contracts their runtime cannot honor.",
                "severity": "high",
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [],
            },
        ],
        "candidates": [],
    }


# ---------------------------------------------------------------------------
# step-1: RED — build_codebook_index() compact rendering
# ---------------------------------------------------------------------------

def test_build_codebook_index_compact_one_line_per_entry():
    codebook = {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Short title A",
                "cause": (
                    "Paragraph one explaining the root cause in some "
                    "detail.\n\nParagraph two continues on and on with "
                    "even more detail than anyone could possibly need "
                    "here, well past the one-line summary cap so "
                    "truncation must kick in eventually for sure yes "
                    "indeed this needs to be quite a bit longer still "
                    "to actually cross the two hundred character cap."
                ),
                "fix": "apply patch XYZ",
                "fix_where": ["some/file.py:12"],
                "status": "open",
                "severity": "high",
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "sightings": [
                    {
                        "date": "2026-01-01", "project": "p", "session": "s",
                        "origin_phase": "implement", "manifested_phase": "merge",
                    }
                ],
            },
            {
                "id": "entry-b",
                "title": "Title B",
                "cause": "one line cause for B",
                "status": "retired",
                "severity": "low",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [],
            },
        ],
        "candidates": [
            {
                "id": "cand-20260101-1",
                "title": "SECRET_CANDIDATE_TITLE_MARKER",
                "cause": "a novel candidate cause, never in the entry index",
                "first_seen": "2026-01-01",
                "disposition": "pending",
                "sightings": [],
            }
        ],
    }

    index = mod.build_codebook_index(codebook)
    lines = [line for line in index.splitlines() if line.strip()]

    assert len(lines) == 2, f"expected exactly one line per entry, got: {lines!r}"
    assert "entry-a" in lines[0] and "Short title A" in lines[0]
    assert "entry-b" in lines[1] and "Title B" in lines[1]

    # cause collapsed to a single line -- no embedded newlines anywhere.
    assert "\n" not in lines[0]
    assert "\n" not in lines[1]

    # retired entries are still included (a census re-observes pre-fix
    # traces, so a live sighting can still match a retired cause).
    assert "entry-b" in index

    # heavy fields (and candidates) never leak into the compact index.
    for banned in (
        "fix_where", "apply patch XYZ", "some/file.py", "sightings",
        "2026-01-01", "SECRET_CANDIDATE_TITLE_MARKER",
    ):
        assert banned not in index, f"heavy/candidate field leaked into index: {banned!r}"


def test_build_codebook_index_empty_entries_yields_empty_or_header_only():
    codebook = {"version": 2, "entries": [], "candidates": []}
    index = mod.build_codebook_index(codebook)
    assert index.strip() == ""


# ---------------------------------------------------------------------------
# step-3: RED — parse_frontmatter() digest frontmatter -> meta dict
# ---------------------------------------------------------------------------

def test_parse_frontmatter_returns_session_date_agent_class(tmp_path):
    digest_text = _build_digest_text(tmp_path, agent_class="orchestrated-task")

    meta = mod.parse_frontmatter(digest_text)

    assert meta["session"] == _SESSION_ID
    assert meta["date"] == "2026-07-14"
    assert meta["agent_class"] == "orchestrated-task"


def test_parse_frontmatter_raises_on_missing_delimiters():
    with pytest.raises(mod.CoderParseError):
        mod.parse_frontmatter("no frontmatter here, just prose")


def test_parse_frontmatter_raises_on_non_mapping_block():
    malformed = "---\n- just\n- a\n- list\n---\nbody\n"
    with pytest.raises(mod.CoderParseError):
        mod.parse_frontmatter(malformed)


# ---------------------------------------------------------------------------
# step-5: RED — build_prompt() embeds digest + codebook index + phase enum
# ---------------------------------------------------------------------------

def test_build_prompt_embeds_digest_and_index_verbatim():
    digest_text = '---\nsession: "s1"\n---\n\n## User Corrections\n- unique marker UC123'
    index = "- entry-a: Title A — cause a\n- entry-b: Title B — cause b"

    prompt = mod.build_prompt(digest_text, index)

    assert digest_text in prompt
    assert index in prompt


def test_build_prompt_embeds_phase_enum_including_unknown():
    prompt = mod.build_prompt("digest text", "codebook index")

    for phase in codebook_mod.PHASES:
        assert phase in prompt
    assert "unknown" in prompt


# ---------------------------------------------------------------------------
# step-7: RED — parse_coder_output() raw LLM stdout -> judgment dict
# ---------------------------------------------------------------------------

def test_parse_coder_output_clean_json_object():
    raw = '{"matches": [], "candidates": []}'
    assert mod.parse_coder_output(raw) == {"matches": [], "candidates": []}


def test_parse_coder_output_json_fenced_block():
    raw = '```json\n{"matches": [], "candidates": []}\n```'
    assert mod.parse_coder_output(raw) == {"matches": [], "candidates": []}


def test_parse_coder_output_json_embedded_in_prose():
    raw = (
        "Sure, here is my judgment:\n"
        '{"matches": [], "candidates": []}\n'
        "Let me know if you need anything else."
    )
    assert mod.parse_coder_output(raw) == {"matches": [], "candidates": []}


def test_parse_coder_output_pure_garbage_raises():
    with pytest.raises(mod.CoderParseError):
        mod.parse_coder_output("I cannot help with that request, sorry.")


def test_parse_coder_output_top_level_array_raises():
    with pytest.raises(mod.CoderParseError):
        mod.parse_coder_output('[{"entry_id": "x"}]')


def test_parse_coder_output_top_level_scalar_raises():
    with pytest.raises(mod.CoderParseError):
        mod.parse_coder_output('"just a string"')


# ---------------------------------------------------------------------------
# step-9: RED — code_digest() happy path, against the real live codebook
# ---------------------------------------------------------------------------

def test_code_digest_happy_path_success(tmp_path):
    digest_text = _build_digest_text(tmp_path, agent_class="orchestrated-task")
    live_codebook = codebook_mod.load(_LIVE_CODEBOOK_PATH)
    meta = mod.parse_frontmatter(digest_text)

    captured = {}
    judgment = {
        "matches": [
            {
                "entry_id": "one-shot-subagent-contract",
                "origin_phase": "implement",
                "manifested_phase": "verify",
                "invariant_violated": None,
                "note": "matched on silent no-op subagent contract",
            }
        ],
        "candidates": [
            {
                "title": "novel confusion shape",
                "cause": "something new observed in this session",
                "area": "orchestrator",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "evidence_quote": "quote from the digest",
            }
        ],
    }

    def fake_invoke(prompt, model):
        captured["prompt"] = prompt
        captured["model"] = model
        return json.dumps(judgment)

    result = mod.code_digest(
        digest_text, live_codebook, project="dark_factory", model="haiku",
        invoke=fake_invoke,
    )

    assert result.ok is True
    # result.record is Optional on the CodingResult — an unparseable or
    # schema-invalid coding record is SKIPPED and leaves it None. ok is True
    # here, so it is populated; narrowed rather than subscripted blind.
    assert result.record is not None
    assert result.record["session"] == meta["session"]
    assert result.record["date"] == meta["date"]
    assert result.record["agent_class"] == meta["agent_class"]
    assert result.record["project"] == "dark_factory"
    assert result.record["matches"] == judgment["matches"]
    assert result.record["candidates"] == judgment["candidates"]
    assert codebook_mod.validate_coding_record(result.record) == []

    assert captured["model"] == "haiku"
    assert digest_text in captured["prompt"]
    index = mod.build_codebook_index(live_codebook)
    assert index in captured["prompt"]


# ---------------------------------------------------------------------------
# step-11: RED — code_digest() never-fabricate + empty-vs-failure
# ---------------------------------------------------------------------------

def test_code_digest_unparseable_output_is_failure_not_fabricated():
    digest_text = _hand_digest(_SESSION_ID, "a confusing correction happened here")
    codebook = _tiny_codebook()

    def fake_invoke(prompt, model):
        return "I'm sorry, I cannot help with that request."

    result = mod.code_digest(
        digest_text, codebook, project="dark_factory", model="haiku",
        invoke=fake_invoke,
    )

    assert result.ok is False
    assert result.record is None
    assert result.reason, "a failure reason must be recorded"
    assert result.session == _SESSION_ID


def test_code_digest_schema_invalid_record_is_failure_not_fabricated():
    digest_text = _hand_digest(_SESSION_ID, "a confusing correction happened here")
    codebook = _tiny_codebook()

    # A match with an out-of-enum origin_phase, and separately a candidate
    # missing its required `title` -- two independent schema violations in
    # one judgment.
    judgment = {
        "matches": [
            {
                "entry_id": "one-shot-subagent-contract",
                "origin_phase": "not_a_real_phase",
            }
        ],
        "candidates": [
            {"cause": "a candidate missing its required title field"}
        ],
    }

    def fake_invoke(prompt, model):
        return json.dumps(judgment)

    result = mod.code_digest(
        digest_text, codebook, project="dark_factory", model="haiku",
        invoke=fake_invoke,
    )

    assert result.ok is False
    assert result.record is None
    assert result.reason is not None
    assert "origin_phase" in result.reason, result.reason
    assert "title" in result.reason, result.reason
    assert result.session == _SESSION_ID


def test_code_digest_invocation_error_is_failure_not_fabricated():
    digest_text = _hand_digest(_SESSION_ID, "a confusing correction happened here")
    codebook = _tiny_codebook()

    def fake_invoke(prompt, model):
        raise mod.CoderInvocationError(
            "claude CLI exited 1 (model='haiku'): simulated backend outage"
        )

    result = mod.code_digest(
        digest_text, codebook, project="dark_factory", model="haiku",
        invoke=fake_invoke,
    )

    assert result.ok is False
    assert result.record is None
    assert result.reason, "a failure reason must be recorded"
    assert result.session == _SESSION_ID


def test_code_digest_malformed_frontmatter_is_failure_not_raised():
    # code_digest itself must never let a CoderParseError from
    # parse_frontmatter propagate uncaught -- a direct caller (not just
    # code_digests' batch-level try/except) must get a clean per-digest
    # failure result, per the function's own never-fabricate docstring.
    digest_text = "no frontmatter here, just prose"
    codebook = _tiny_codebook()

    def fake_invoke(prompt, model):
        raise AssertionError("invoke must never be reached when frontmatter is unparseable")

    result = mod.code_digest(
        digest_text, codebook, project="dark_factory", model="haiku",
        invoke=fake_invoke,
    )

    assert result.ok is False
    assert result.record is None
    assert result.reason, "a failure reason must be recorded"
    assert result.session is None


def test_code_digest_empty_judgment_is_success_not_failure():
    digest_text = _hand_digest(_SESSION_ID, "nothing confusing happened this session")
    codebook = _tiny_codebook()

    def fake_invoke(prompt, model):
        return json.dumps({"matches": [], "candidates": []})

    result = mod.code_digest(
        digest_text, codebook, project="dark_factory", model="haiku",
        invoke=fake_invoke,
    )

    assert result.ok is True
    assert result.reason is None
    assert result.record is not None
    assert result.record["matches"] == []
    assert result.record["candidates"] == []
    assert codebook_mod.validate_coding_record(result.record) == []


# ---------------------------------------------------------------------------
# step-13: RED — code_digests() batch + storm threshold
# ---------------------------------------------------------------------------

def _make_batch_invoke(fail_sessions):
    """Fake invoke: returns unparseable garbage for any digest whose
    prompt embeds a session id in `fail_sessions`, else a valid empty
    judgment. Session ids are looked up as substrings of the prompt,
    which embeds the digest text (and therefore its frontmatter
    `session: "..."` line) verbatim."""
    def fake_invoke(prompt, model):
        for session_id in fail_sessions:
            if session_id in prompt:
                return "not parseable as json, sorry"
        return json.dumps({"matches": [], "candidates": []})
    return fake_invoke


def _batch_digests(n):
    return [_hand_digest(f"batch-sess-{i}", f"body marker {i}") for i in range(n)]


def test_code_digests_all_succeed_batch_status_ok():
    digests = _batch_digests(3)
    codebook = _tiny_codebook()

    result = mod.code_digests(
        digests, codebook, project="dark_factory", model="haiku",
        invoke=_make_batch_invoke(fail_sessions=set()),
    )

    assert result.status == "ok"
    assert result.total == 3
    assert result.succeeded == 3
    assert result.failed == 0
    assert len(result.records) == 3
    assert result.failures == []
    for record in result.records:
        assert codebook_mod.validate_coding_record(record) == []


def test_code_digests_majority_failure_batch_status_failure():
    digests = _batch_digests(4)
    codebook = _tiny_codebook()
    fail_sessions = {"batch-sess-0", "batch-sess-1", "batch-sess-2"}

    result = mod.code_digests(
        digests, codebook, project="dark_factory", model="haiku",
        invoke=_make_batch_invoke(fail_sessions),
    )

    assert result.status == "failure"
    assert result.total == 4
    assert result.succeeded == 1
    assert result.failed == 3
    assert len(result.records) == 1
    assert len(result.failures) == 3
    failure_sessions = {session for session, _reason in result.failures}
    assert failure_sessions == fail_sessions
    for _session, reason in result.failures:
        assert reason


def test_code_digests_exactly_half_failure_is_not_a_storm():
    digests = _batch_digests(4)
    codebook = _tiny_codebook()
    fail_sessions = {"batch-sess-0", "batch-sess-1"}

    result = mod.code_digests(
        digests, codebook, project="dark_factory", model="haiku",
        invoke=_make_batch_invoke(fail_sessions),
    )

    assert result.status == "ok"
    assert result.total == 4
    assert result.succeeded == 2
    assert result.failed == 2
    assert len(result.records) == 2
    assert len(result.failures) == 2
    failure_sessions = {session for session, _reason in result.failures}
    assert failure_sessions == fail_sessions


# ---------------------------------------------------------------------------
# task 4511: code_digests announces EVERY per-digest failure as it happens.
#
# Not merely a nicer rendering of the aggregate nightly.py already escalates.
# A SUB-STORM batch -- failed/total <= 0.5, e.g. 2 of 4 -- returns
# status="ok", so run_nightly escalates nothing and those failures reach NO
# sink at all today: not the journal, not an escalation, nowhere. Per-digest
# lines also separate a storm of 38 identical ENOENTs from 38 distinct model
# errors, a distinction the single joined aggregate detail flattens.
# ---------------------------------------------------------------------------

def _coder_warnings(caplog):
    """Records at >= WARNING on coder.py's OWN logger, filtered by name so a
    sibling module's records can never be mistaken for these."""
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "legibility.coder"
    ]


def _make_crashing_invoke(crash_sessions):
    """Fake invoke that raises a BARE RuntimeError (not CoderInvocationError)
    for the named sessions -- the exception class code_digest does NOT catch,
    so it escapes to code_digests' own isolating `except Exception` and lands
    as a `(None, reason)` failure."""
    def fake_invoke(prompt, model):
        for session_id in crash_sessions:
            if session_id in prompt:
                raise RuntimeError(f"unexpected explosion coding {session_id}")
        return json.dumps({"matches": [], "candidates": []})
    return fake_invoke


def test_code_digests_logs_every_failure_in_a_sub_storm_batch(caplog):
    """THE MOTIVATING CASE: 2 of 4 fail, which does not STRICTLY exceed the
    0.5 storm threshold, so status stays "ok" and nightly.py escalates
    nothing. These WARNINGs are the only sink those two failures ever
    reach."""
    digests = _batch_digests(4)
    fail_sessions = {"batch-sess-0", "batch-sess-1"}

    with caplog.at_level(logging.DEBUG, logger="legibility.coder"):
        result = mod.code_digests(
            digests, _tiny_codebook(), project="dark_factory", model="haiku",
            invoke=_make_batch_invoke(fail_sessions),
        )

    assert result.status == "ok", (
        "if this ever became a storm the test would be pinning the wrong "
        "case -- the whole point is that nightly.py stays silent here"
    )

    warned = _coder_warnings(caplog)
    assert len(warned) == 2, (
        f"expected one WARNING per failed digest; got "
        f"{[r.getMessage() for r in warned]}"
    )
    messages = [r.getMessage() for r in warned]
    for session_id in sorted(fail_sessions):
        assert any(session_id in m for m in messages), (
            f"{session_id!r} never reached the journal; got {messages}"
        )
    for session_id in ("batch-sess-2", "batch-sess-3"):
        assert not any(session_id in m for m in messages), (
            f"a SUCCEEDING digest was reported as a failure: {messages}"
        )
    for message in messages:
        assert "could not parse a JSON object" in message, (
            f"the REASON is the diagnosis, not just the session id; got "
            f"{message!r}"
        )


def test_code_digests_logs_one_record_per_failure_in_a_storm(caplog):
    """3 of 4 -- a genuine storm. One line per failure, so an operator can
    tell three identical ENOENTs from three distinct model errors; the
    aggregate nightly.py escalates joins them into one string and loses
    that."""
    digests = _batch_digests(4)
    fail_sessions = {"batch-sess-0", "batch-sess-1", "batch-sess-2"}

    with caplog.at_level(logging.DEBUG, logger="legibility.coder"):
        result = mod.code_digests(
            digests, _tiny_codebook(), project="dark_factory", model="haiku",
            invoke=_make_batch_invoke(fail_sessions),
        )

    assert result.status == "failure"

    warned = _coder_warnings(caplog)
    assert len(warned) == 3, (
        f"expected one WARNING per failed digest; got "
        f"{[r.getMessage() for r in warned]}"
    )
    messages = [r.getMessage() for r in warned]
    for session_id in sorted(fail_sessions):
        assert any(session_id in m for m in messages), (
            f"{session_id!r} never reached the journal; got {messages}"
        )


def test_code_digests_logs_the_isolated_crash_path_too(caplog):
    """The OTHER failure path -- code_digests' own isolating `except
    Exception`, which yields `(None, reason)` because the crash happened
    before a session could be attributed. Both paths must reach the journal,
    and the batch must keep going."""
    digests = _batch_digests(3)

    with caplog.at_level(logging.DEBUG, logger="legibility.coder"):
        result = mod.code_digests(
            digests, _tiny_codebook(), project="dark_factory", model="haiku",
            invoke=_make_crashing_invoke({"batch-sess-1"}),
        )

    # The batch kept going: the other two digests still coded.
    assert result.status == "ok"
    assert result.total == 3
    assert result.succeeded == 2
    assert len(result.failures) == 1
    session, reason = result.failures[0]
    assert session is None
    assert "unexpected explosion coding batch-sess-1" in reason

    warned = _coder_warnings(caplog)
    assert len(warned) == 1, (
        f"expected exactly one WARNING; got {[r.getMessage() for r in warned]}"
    )
    message = warned[0].getMessage()
    assert "unexpected explosion coding batch-sess-1" in message
    assert "None" in message, (
        f"an unattributable crash must SAY the session is unknown rather "
        f"than omitting it; got {message!r}"
    )


# ---------------------------------------------------------------------------
# step-15: RED — _invoke_cli() via the fake-`claude`-binary-on-PATH idiom
# (per tests/scripts/test_spawn_claude.py's _write_fake_claude* idiom)
# ---------------------------------------------------------------------------

def _write_fake_claude_capturing(bin_dir, *, argv_file, stdin_file, stdout_path):
    """Fake `claude` binary: records its own argv (one per line, via "$@",
    which excludes $0/the binary path itself) and its stdin to files, then
    echoes the contents of *stdout_path* to stdout and exits 0."""
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > "{argv_file}"\n'
        f'cat > "{stdin_file}"\n'
        f'cat "{stdout_path}"\n'
    )
    p.chmod(0o755)


def _write_fake_claude_recording_cwd(bin_dir, *, cwd_file, stdout_path):
    """Fake `claude` binary: records the directory it was RUN IN to
    *cwd_file*, then echoes the contents of *stdout_path* and exits 0.

    Separate from _write_fake_claude_capturing so the cwd tests assert the
    one thing they are about; `pwd` is the honest probe here because the
    working directory is a property of the spawned process, not of anything
    _invoke_cli could report about itself."""
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        f'pwd > "{cwd_file}"\n'
        f'cat > /dev/null\n'
        f'cat "{stdout_path}"\n'
    )
    p.chmod(0o755)


def _write_fake_claude_failing(bin_dir, *, exit_code=1, stderr_text="simulated failure"):
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "{stderr_text}" >&2\n'
        f"exit {exit_code}\n"
    )
    p.chmod(0o755)


def _write_fake_claude_sleeping(bin_dir, *, sleep_secs):
    """Fake `claude` binary: sleeps past any reasonable test timeout before
    ever producing output, to exercise _invoke_cli's
    subprocess.TimeoutExpired -> CoderInvocationError path."""
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        f"sleep {sleep_secs}\n"
        'printf \'{"matches": [], "candidates": []}\'\n'
    )
    p.chmod(0o755)


def test_invoke_cli_argv_and_prompt_delivery(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv.txt"
    stdin_file = tmp_path / "stdin.txt"
    stdout_path = tmp_path / "stdout.txt"
    stdout_path.write_text('{"matches": [], "candidates": []}')

    _write_fake_claude_capturing(
        bin_dir, argv_file=argv_file, stdin_file=stdin_file, stdout_path=stdout_path,
    )

    raw = mod._invoke_cli(
        "the prompt text UNIQUE_MARKER_UC999", "haiku",
        claude_bin=str(bin_dir / "claude"), timeout=10,
    )

    argv = argv_file.read_text().splitlines()
    assert "-p" in argv, argv
    assert "--model" in argv, argv
    assert "haiku" in argv, argv

    # Prompt delivery: stdin, per code_digest's `input=prompt` contract --
    # not necessarily present in argv.
    assert "the prompt text UNIQUE_MARKER_UC999" in stdin_file.read_text()

    assert raw == '{"matches": [], "candidates": []}'


def test_invoke_cli_nonzero_exit_raises_invocation_error(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude_failing(bin_dir, exit_code=1, stderr_text="boom, the model backend is down")

    with pytest.raises(mod.CoderInvocationError):
        mod._invoke_cli(
            "prompt text", "haiku",
            claude_bin=str(bin_dir / "claude"), timeout=10,
        )


# ---------------------------------------------------------------------------
# task 4736 / GAP 2: a non-zero exit must carry BOTH streams, each labelled
#
# The 2026-08-24 incident: the claude CLI wrote its cap banner to STDOUT and
# exited 1, and _invoke_cli embedded only `(proc.stderr or "")[-2000:]`.  With
# stderr empty, the reason string that reached the journal, the escalation and
# `run.failures` was the bare `claude CLI exited 1 (model='haiku', ...): ` on
# 17 of 20 digests -- the CLI had SAID exactly what was wrong and the coder
# discarded it.
# ---------------------------------------------------------------------------

def _write_fake_claude_failing_on_both_streams(
    bin_dir, *, stdout_text="", stderr_text="", exit_code=1,
):
    """Fake `claude` binary: emit *stdout_text* on STDOUT and *stderr_text* on
    STDERR, then exit *exit_code*.

    Payloads travel through sidecar FILES that the script ``cat``s, never
    interpolated into the shell source the way _write_fake_claude_failing
    does it.  Not fastidiousness: the task-4736 cap tests parametrize this
    writer over ``shared.cap_markers.REAL_CLI_CAP_MESSAGES``, whose entries
    already carry apostrophes and a U+00B7, and the next transcript-cited
    entry may carry a ``"``, a ``$`` or a backtick that the shell would eat
    or that would split the script outright.  The bytes the fake CLI emits
    have to be the corpus's bytes, or a green test would be proving something
    other than what the CLI actually said.
    """
    out_file = bin_dir / "fake_stdout.txt"
    err_file = bin_dir / "fake_stderr.txt"
    out_file.write_text(stdout_text, encoding="utf-8")
    err_file.write_text(stderr_text, encoding="utf-8")
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        # Drain the prompt so a large stdin can never EPIPE the fake before
        # it has emitted its payloads.
        "cat > /dev/null\n"
        f'cat "{out_file}"\n'
        f'cat "{err_file}" >&2\n'
        f"exit {exit_code}\n"
    )
    p.chmod(0o755)


def test_invoke_cli_nonzero_exit_carries_both_streams_labelled(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude_failing_on_both_streams(
        bin_dir,
        stdout_text="STDOUT_MARKER_SO7412 the CLI's own diagnostic",
        stderr_text="STDERR_MARKER_SE7412 the backend's complaint",
    )

    with pytest.raises(mod.CoderInvocationError) as excinfo:
        mod._invoke_cli(
            "prompt text", "haiku",
            claude_bin=str(bin_dir / "claude"), timeout=10, cwd=str(tmp_path),
        )

    message = str(excinfo.value)

    # (a) BOTH streams reach the reader.  Dropping either one is the defect.
    assert "STDOUT_MARKER_SO7412" in message, (
        f"the CLI's stdout must reach the error text -- this is the exact "
        f"byte the 2026-08-24 incident discarded; got {message!r}"
    )
    assert "STDERR_MARKER_SE7412" in message, message

    # (b) Each stream is LABELLED, so a reader can tell which stream said
    # what.  Two unlabelled blobs concatenated would carry the bytes but not
    # the provenance, and "the CLI wrote its banner to STDOUT" is precisely
    # the fact this incident turned on.
    assert "stdout=" in message, (
        f"each stream must be labelled so the reader knows which one carried "
        f"the diagnostic; got {message!r}"
    )
    assert "stderr=" in message, message

    # (c) The pre-existing invocation context is still there -- this change
    # ADDS a stream, it does not trade one diagnostic for another.
    assert "model='haiku'" in message, message
    assert "claude_bin=" in message, message
    assert "cwd=" in message, message
    assert "exited 1" in message, message


def test_invoke_cli_nonzero_exit_with_empty_stderr_still_says_why(tmp_path):
    """The exact 17-of-20 shape from 2026-08-24: everything on stdout,
    stderr EMPTY, exit 1.  Before this change the operator-visible reason was
    the empty-tailed `...cwd=None): ` -- a failure that named no cause."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude_failing_on_both_streams(
        bin_dir,
        stdout_text="ONLY_ON_STDOUT_OS7412",
        stderr_text="",
    )

    with pytest.raises(mod.CoderInvocationError) as excinfo:
        mod._invoke_cli(
            "prompt text", "haiku",
            claude_bin=str(bin_dir / "claude"), timeout=10,
        )

    message = str(excinfo.value)
    assert "ONLY_ON_STDOUT_OS7412" in message, (
        f"with stderr empty the stdout text is the ONLY diagnostic the CLI "
        f"produced; an error that omits it names no cause at all -- which is "
        f"what reached the journal on 17 of 20 digests; got {message!r}"
    )
    # And the reason is no longer effectively empty after the colon.
    assert not message.rstrip().endswith(":"), message


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_invoke_cli_nonzero_exit_tail_bounds_each_stream_keeping_the_tail(
    tmp_path, stream,
):
    """Each stream is bounded, to the SAME constant, keeping its TAIL.

    Parametrized over both streams rather than source-grepping for a stray
    ``[-2000:]``: the bound that matters is the one the code APPLIES, and a
    grep cannot tell a live literal from the docstrings that deliberately
    cite the old one-stream slice as the incident's provenance.  Asserting
    the behaviour on both streams pins "one shared bound, symmetrically
    applied" directly -- an unbounded stream or a head-truncated one fails
    here whatever the source happens to spell.
    """
    bound = mod._ERROR_STREAM_TAIL_CHARS
    assert isinstance(bound, int) and bound > 0

    payload = "HEAD_MARKER_HM7412" + ("x" * (bound * 2)) + "TAIL_MARKER_TM7412"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude_failing_on_both_streams(
        bin_dir,
        stdout_text=payload if stream == "stdout" else "",
        stderr_text=payload if stream == "stderr" else "",
    )

    with pytest.raises(mod.CoderInvocationError) as excinfo:
        mod._invoke_cli(
            "prompt text", "haiku",
            claude_bin=str(bin_dir / "claude"), timeout=10,
        )

    message = str(excinfo.value)
    assert "TAIL_MARKER_TM7412" in message, (
        f"{stream} must be truncated to its TAIL, not its head -- a CLI's "
        f"last words are its diagnostic ones; got {message!r}"
    )
    assert "HEAD_MARKER_HM7412" not in message, (
        f"an unbounded {stream} would blow up every journal line, "
        f"run.failures entry and escalation body it lands in; the bound must "
        f"actually bind"
    )
    # The bound BINDS, and binds to the shared constant: the surviving run of
    # x's cannot exceed it.  (The message also carries the invocation context
    # and the other, empty stream, so an exact length check would be pinning
    # the prose rather than the bound.)
    runs = [len(m) for m in re.findall(r"x+", message)]
    assert runs, message
    assert max(runs) <= bound, (
        f"{stream}'s surviving payload ({max(runs)} chars) exceeds the shared "
        f"bound of {bound}"
    )


def test_invoke_cli_timeout_raises_invocation_error(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude_sleeping(bin_dir, sleep_secs=2)

    with pytest.raises(mod.CoderInvocationError):
        mod._invoke_cli(
            "prompt text", "haiku",
            claude_bin=str(bin_dir / "claude"), timeout=0.2,
        )


def _cwd_probe(tmp_path, monkeypatch):
    """Set up the two-sibling-directory fixture the cwd tests share:
    chdir into `launcher/`, and return (target_dir, cwd_file, claude_bin)
    for a fake `claude` that records where it ran."""
    bin_dir = tmp_path / "bin"
    launcher = tmp_path / "launcher"
    target = tmp_path / "target"
    for d in (bin_dir, launcher, target):
        d.mkdir()

    cwd_file = tmp_path / "cwd.txt"
    stdout_path = tmp_path / "stdout.txt"
    stdout_path.write_text('{"matches": [], "candidates": []}')
    _write_fake_claude_recording_cwd(bin_dir, cwd_file=cwd_file, stdout_path=stdout_path)

    monkeypatch.chdir(launcher)
    return launcher, target, cwd_file, str(bin_dir / "claude")


def test_invoke_cli_runs_in_given_cwd(tmp_path, monkeypatch):
    """A caller can scope the headless CLI subprocess to a directory other
    than the launcher's — the knob census needs to verify a project that is
    not the one it was launched from."""
    launcher, target, cwd_file, claude_bin = _cwd_probe(tmp_path, monkeypatch)

    mod._invoke_cli(
        "prompt text", "haiku", claude_bin=claude_bin, timeout=10, cwd=str(target),
    )

    # .resolve() on both sides: a symlinked tmp dir (/tmp -> /private/tmp on
    # non-Linux hosts) must not make this flaky.
    recorded = Path(cwd_file.read_text().strip()).resolve()
    assert recorded == target.resolve()
    assert recorded != launcher.resolve()


def test_invoke_cli_without_cwd_inherits_the_launcher_cwd(tmp_path, monkeypatch):
    """The new parameter's default is EXACTLY today's behavior, so the
    existing callers (code_digest, the trickle, nightly) are provably
    unchanged by its introduction."""
    launcher, target, cwd_file, claude_bin = _cwd_probe(tmp_path, monkeypatch)

    mod._invoke_cli("prompt text", "haiku", claude_bin=claude_bin, timeout=10)

    recorded = Path(cwd_file.read_text().strip()).resolve()
    assert recorded == launcher.resolve()
    assert recorded != target.resolve()


def test_invoke_cli_missing_cwd_raises_invocation_error(tmp_path, monkeypatch):
    """A cwd that does not exist must fail as a CoderInvocationError, not
    as a raw FileNotFoundError escaping the documented contract.

    This is not cosmetic typing. census's first invoke is the headroom
    probe, and preflight_headroom folds ANY probe exception into
    HeadroomResult(ok=False) — so an unwrapped OSError from a typo'd
    --project-root would exit 0 as "census deferred", indistinguishable
    from a usage-limit banner.
    """
    _launcher, target, _cwd_file, claude_bin = _cwd_probe(tmp_path, monkeypatch)

    with pytest.raises(mod.CoderInvocationError) as excinfo:
        mod._invoke_cli(
            "prompt text", "haiku",
            claude_bin=claude_bin, timeout=10, cwd=str(target / "does-not-exist"),
        )

    # The message must NAME the directory, so an operator reading the
    # failure does not have to infer which path was rejected.
    assert "does-not-exist" in str(excinfo.value)


def test_invoke_cli_cwd_that_is_a_file_raises_invocation_error(tmp_path, monkeypatch):
    """Same contract for the not-a-directory case (NotADirectoryError on
    Linux), which is a different OSError subclass than the missing-dir
    one — pinning both keeps the except-arm from being narrowed to
    FileNotFoundError alone."""
    _launcher, target, _cwd_file, claude_bin = _cwd_probe(tmp_path, monkeypatch)
    not_a_dir = target / "regular-file.txt"
    not_a_dir.write_text("i am not a directory", encoding="utf-8")

    with pytest.raises(mod.CoderInvocationError):
        mod._invoke_cli(
            "prompt text", "haiku",
            claude_bin=claude_bin, timeout=10, cwd=str(not_a_dir),
        )


# ---------------------------------------------------------------------------
# task 4510: characterization pins over _invoke_cli's LEGIBILITY_CLAUDE_BIN
# branch — the seam scripts/legibility-trickle@.service's Environment= line
# depends on. NOT a RED: coder.py already implements this branch. These make
# the pin undeletable from the CODE side, so dropping the env-var lookup
# (which would silently re-open the 2026-08-18 outage even with the unit line
# present) fails here.
# ---------------------------------------------------------------------------

def _scrub_path_of_claude(tmp_path, monkeypatch):
    """Point PATH somewhere the REAL `claude` is NOT resolvable, and prove it.

    Load-bearing test SAFETY, not tidiness. The real
    /home/leo/.local/bin/claude is on the test runner's PATH, so if
    _invoke_cli's resolution order (`claude_bin or
    os.environ.get(_CLAUDE_BIN_ENV_VAR) or "claude"`) ever regresses, an
    unscrubbed PATH would let the bare-name fallback spawn the GENUINE claude
    CLI — real LLM spend, real wall-clock, and a test that passes for the
    wrong reason, silently breaking this module's docstring promise that the
    LLM is ALWAYS mocked here. With `claude` unresolvable, that same
    regression instead ENOENTs into CoderInvocationError: loud and cheap.

    Deliberately NOT a fully empty PATH, though that is the obvious spelling.
    The fake binaries above are `#!/usr/bin/env bash` scripts and `env` needs
    PATH to find `bash`, so an empty PATH makes every fake die with exit 127
    ("env: 'bash': No such file or directory") — failing these tests for a
    reason with nothing to do with the branch under test. PATH therefore keeps
    a stdlib bin dir and drops ~/.local/bin, and the assertion below pins the
    property that actually matters instead of trusting the spelling to imply
    it.
    """
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", f"{empty_bin}{os.pathsep}/usr/bin")
    assert shutil.which("claude") is None, (
        "PATH scrub failed: a real `claude` is still resolvable, so a "
        "regression in _invoke_cli's env-var branch would silently spawn the "
        "GENUINE CLI (real spend) instead of failing loudly"
    )


def test_invoke_cli_honours_claude_bin_env_var(tmp_path, monkeypatch):
    """With no explicit claude_bin=, the binary comes from
    LEGIBILITY_CLAUDE_BIN — the exact seam the trickle systemd unit pins so
    the coder survives a `systemd --user` manager whose PATH lacks
    ~/.local/bin (2026-08-18: 6/6 selected digests ENOENT'd on reify, 38/38
    on dark_factory)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv.txt"
    stdin_file = tmp_path / "stdin.txt"
    stdout_path = tmp_path / "stdout.txt"
    stdout_path.write_text('{"matches": [], "candidates": []}')
    _write_fake_claude_capturing(
        bin_dir, argv_file=argv_file, stdin_file=stdin_file, stdout_path=stdout_path,
    )

    _scrub_path_of_claude(tmp_path, monkeypatch)
    monkeypatch.setenv(mod._CLAUDE_BIN_ENV_VAR, str(bin_dir / "claude"))

    raw = mod._invoke_cli("the prompt text UNIQUE_MARKER_ENV777", "haiku", timeout=10)

    # The env-var-resolved binary actually RAN, and got the real argv/stdin.
    argv = argv_file.read_text().splitlines()
    assert "-p" in argv, argv
    assert "--model" in argv, argv
    assert "haiku" in argv, argv
    assert "the prompt text UNIQUE_MARKER_ENV777" in stdin_file.read_text()
    assert raw == '{"matches": [], "candidates": []}'


def test_invoke_cli_explicit_claude_bin_beats_the_env_var(tmp_path, monkeypatch):
    """Pins the precedence order _invoke_cli's docstring promises: explicit
    argument > env var > bare name. The env var points at a FAILING fake, so
    if precedence ever inverted this would raise CoderInvocationError."""
    good_dir = tmp_path / "good-bin"
    bad_dir = tmp_path / "bad-bin"
    good_dir.mkdir()
    bad_dir.mkdir()

    argv_file = tmp_path / "argv.txt"
    stdin_file = tmp_path / "stdin.txt"
    stdout_path = tmp_path / "stdout.txt"
    stdout_path.write_text('{"matches": [], "candidates": []}')
    # Separate directories on purpose: both helpers write a file literally
    # named "claude", so a shared bin_dir would have the loser overwrite the
    # winner and the test would prove nothing.
    _write_fake_claude_capturing(
        good_dir, argv_file=argv_file, stdin_file=stdin_file, stdout_path=stdout_path,
    )
    _write_fake_claude_failing(bad_dir, exit_code=1, stderr_text="env var fake must not win")

    _scrub_path_of_claude(tmp_path, monkeypatch)
    monkeypatch.setenv(mod._CLAUDE_BIN_ENV_VAR, str(bad_dir / "claude"))

    raw = mod._invoke_cli(
        "the prompt text UNIQUE_MARKER_PREC555", "haiku",
        claude_bin=str(good_dir / "claude"), timeout=10,
    )

    assert raw == '{"matches": [], "candidates": []}'
    assert "the prompt text UNIQUE_MARKER_PREC555" in stdin_file.read_text()


# ---------------------------------------------------------------------------
# step-17: RED — main(argv) end-to-end, LLM mocked via monkeypatch of
# mod._invoke_cli (never a real subprocess)
# ---------------------------------------------------------------------------

def _write_main_digest(tmp_path, session_id, marker, name):
    """Build a real digest file (via digest.build_digest, mirroring
    _build_digest_text) and write it to tmp_path/name. Returns the Path."""
    records = [_user_text(marker, session_id=session_id)]
    transcript_path = _write_jsonl(tmp_path, records, name=f"{name}.transcript.jsonl")
    text = digest_mod.build_digest(transcript_path, agent_class_override="interactive")
    digest_path = tmp_path / f"{name}.md"
    digest_path.write_text(text, encoding="utf-8")
    return digest_path


def test_main_happy_path_writes_valid_jsonl_and_returns_0(tmp_path, monkeypatch, capsys):
    digest1 = _write_main_digest(tmp_path, "main-sess-1", "first session confusion", "d1")
    digest2 = _write_main_digest(tmp_path, "main-sess-2", "second session confusion", "d2")

    codebook_path = tmp_path / "codebook.yaml"
    codebook_mod.dump(_tiny_codebook(), codebook_path)

    def fake_invoke_cli(prompt, model, **kwargs):
        return json.dumps({"matches": [], "candidates": []})

    monkeypatch.setattr(mod, "_invoke_cli", fake_invoke_cli)

    out_path = tmp_path / "out.jsonl"
    rc = mod.main([
        str(digest1), str(digest2),
        "--codebook", str(codebook_path),
        "--project", "dark_factory",
        "--out", str(out_path),
    ])

    assert rc == 0
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    sessions = set()
    for line in lines:
        record = json.loads(line)
        assert codebook_mod.validate_coding_record(record) == []
        sessions.add(record["session"])
    assert sessions == {"main-sess-1", "main-sess-2"}

    captured = capsys.readouterr()
    assert "total=2" in captured.err
    assert "failed=0" in captured.err


def test_main_storm_writes_zero_records_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    digests = [
        _write_main_digest(tmp_path, f"main-storm-{i}", f"session {i} confusion", f"storm{i}")
        for i in range(3)
    ]

    codebook_path = tmp_path / "codebook.yaml"
    codebook_mod.dump(_tiny_codebook(), codebook_path)

    def fake_invoke_cli(prompt, model, **kwargs):
        return "not parseable as json, sorry"

    monkeypatch.setattr(mod, "_invoke_cli", fake_invoke_cli)

    out_path = tmp_path / "out.jsonl"
    rc = mod.main([
        *[str(d) for d in digests],
        "--codebook", str(codebook_path),
        "--project", "dark_factory",
        "--out", str(out_path),
    ])

    assert rc != 0
    # zero downstream coding records written (PRD §8.6 storm fixture)
    assert not out_path.exists() or out_path.read_text(encoding="utf-8").strip() == ""

    captured = capsys.readouterr()
    assert captured.err, "expected a failure summary on stderr"
    assert "3/3" in captured.err or "failed=3" in captured.err


def test_main_storm_truncates_stale_out_from_prior_run(tmp_path, monkeypatch, capsys):
    # A storm must never leave a --out from a PRIOR successful run lying
    # around looking like this run's (empty) output -- a downstream
    # consumer that reads the file instead of gating on the exit code
    # must see the current run's true outcome.
    digests = [
        _write_main_digest(tmp_path, f"main-stale-{i}", f"session {i} confusion", f"stale{i}")
        for i in range(3)
    ]

    codebook_path = tmp_path / "codebook.yaml"
    codebook_mod.dump(_tiny_codebook(), codebook_path)

    out_path = tmp_path / "out.jsonl"
    out_path.write_text(
        json.dumps({
            "session": "prior-run-session", "date": "2026-07-01",
            "project": "dark_factory", "agent_class": "interactive",
            "matches": [], "candidates": [],
        }) + "\n",
        encoding="utf-8",
    )

    def fake_invoke_cli(prompt, model, **kwargs):
        return "not parseable as json, sorry"

    monkeypatch.setattr(mod, "_invoke_cli", fake_invoke_cli)

    rc = mod.main([
        *[str(d) for d in digests],
        "--codebook", str(codebook_path),
        "--project", "dark_factory",
        "--out", str(out_path),
    ])

    assert rc != 0
    assert out_path.read_text(encoding="utf-8") == "", (
        "a stale --out from a prior run must be truncated on a storm, "
        "never left holding a previous run's records"
    )


# ---------------------------------------------------------------------------
# main(--digests DIR) -- amendment: iterdir() must skip subdirectories and
# other non-file entries rather than crash on read_text(IsADirectoryError)
# ---------------------------------------------------------------------------

def test_main_digests_dir_skips_subdirectories(tmp_path, monkeypatch, capsys):
    digests_dir = tmp_path / "digests"
    digests_dir.mkdir()

    records = [_user_text("a confusing correction", session_id="dir-sess-1")]
    transcript_path = _write_jsonl(tmp_path, records, name="dir.transcript.jsonl")
    text = digest_mod.build_digest(transcript_path, agent_class_override="interactive")
    (digests_dir / "d1.md").write_text(text, encoding="utf-8")

    # A stray subdirectory alongside the real digest file -- iterdir()
    # yields it too; read_text() on a directory raises IsADirectoryError
    # if it isn't filtered out first.
    (digests_dir / "a_subdir").mkdir()

    codebook_path = tmp_path / "codebook.yaml"
    codebook_mod.dump(_tiny_codebook(), codebook_path)

    def fake_invoke_cli(prompt, model, **kwargs):
        return json.dumps({"matches": [], "candidates": []})

    monkeypatch.setattr(mod, "_invoke_cli", fake_invoke_cli)

    out_path = tmp_path / "out.jsonl"
    rc = mod.main([
        "--digests", str(digests_dir),
        "--codebook", str(codebook_path),
        "--project", "dark_factory",
        "--out", str(out_path),
    ])

    assert rc == 0
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["session"] == "dir-sess-1"
