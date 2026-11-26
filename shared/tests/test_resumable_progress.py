"""Tests for the cap-hit resume progress predicate (task 4274).

The class under guard: a TOP-LEVEL session killed by the usage cap before it
made any tool call, then resumed with CAP_HIT_RESUME_PROMPT ("Continue where
you left off and complete your task") when its only captured output was one
sentence of stated intent — a continuity claim the transcript does not support.
The cap-hit branch decided resume-vs-fresh on ONE question: is the transcript
REACHABLE?  A transcript that EXISTS was assumed to carry resumable work.

PROVENANCE — and its limit.  Legibility census 2026-08-16 §1.2 (session
4396db7a) NAMED this failure mode, but its own specimen is NOT covered by this
predicate and task 4274 does not close that finding.  4396db7a was an Agent-tool
SUB-AGENT kill; a sub-agent's turns go to a sidecar
``<session_id>/subagents/agent-*.jsonl`` under the PARENT's sessionId, so the
parent transcript ``read_transcript_records`` reads necessarily carries the
Task/Agent ``tool_use`` that spawned it and ``detect_resumable_progress``
returns True on it.  Census §4 declined to file a task for 1.2 and left the
remedy with task **2561**'s runner-side persistence protocol; that ownership
stands.  These tests therefore pin the ADJACENT top-level-session class only.

``detect_resumable_progress`` is the pure half of the answer to the next
question — does the transcript record work to CONTINUE? — and
``resumable_progress_for_session`` is its session-level wrapper.

FAIL-SAFE DIRECTION (the load-bearing property these tests pin).  The predicate
can only ever cause a resume→fresh DOWNGRADE, and a wrong downgrade DISCARDS
REAL AGENT WORK — strictly worse than the confusing-but-harmless prompt this
fixes.  So every ambiguous input must resolve to True (resume, today's
behaviour); only the affirmatively-proven-empty case may return False.

Structured on ``test_cli_invoke_background.py``, the direct precedent for a
transcript predicate + session-wrapper pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.cli_invoke import (
    detect_resumable_progress,
    resumable_progress_for_session,
)

# ── Fixture builders ────────────────────────────────────────────────────────
# Hand-authored transcript record dicts mirroring the on-disk JSONL shape: an
# assistant record carries content blocks either nested under
# ``record['message']['content']`` (the real CLI shape) or flat under
# ``record['content']`` (the shape the existing read_transcript_records tests
# use).  Both nestings are supported and both are covered here.


def _assistant(blocks: list, *, nested: bool = True) -> dict:
    """An assistant transcript record wrapping *blocks*.

    ``nested=True`` → record['message']['content'] (real CLI shape).
    ``nested=False`` → flat record['content'].
    """
    if nested:
        return {'type': 'assistant', 'message': {'role': 'assistant', 'content': blocks}}
    return {'type': 'assistant', 'content': blocks}


def _text(text: str = 'I will start by reading the plan.') -> dict:
    return {'type': 'text', 'text': text}


def _tool_use(name: str = 'Read', **inp) -> dict:
    return {'type': 'tool_use', 'name': name, 'input': inp or {'file_path': '/tmp/x.py'}}


def _user(text: str = 'do the task') -> dict:
    return {'type': 'user', 'content': text}


def _write_transcript(base: Path, session_id: str, records: list) -> Path:
    """Write *records* as a JSONL transcript under
    base/projects/<slug>/<session_id>.jsonl — the layout
    ``_resolve_transcript_path`` globs for."""
    slug_dir = base / 'projects' / 'myproject'
    slug_dir.mkdir(parents=True, exist_ok=True)
    transcript = slug_dir / f'{session_id}.jsonl'
    transcript.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    return transcript


class TestDetectResumableProgress:
    """The pure predicate returns False ONLY when it can affirmatively prove
    there is nothing to continue: a non-None record list with ZERO assistant
    ``tool_use`` blocks AND at most one assistant record.  Everything else —
    including every malformed or ambiguous shape — returns True."""

    # ── The two proven-empty shapes (False) ─────────────────────────────────

    def test_empty_list_is_false(self) -> None:
        """No records at all → nothing to continue → False."""
        assert detect_resumable_progress([]) is False

    def test_single_text_only_assistant_turn_is_false(self) -> None:
        """The covered shape: a top-level session capped before its first tool
        call — one sentence of stated intent, nothing else.  This is the whole
        point of the predicate.  (NOT census 1.2's own specimen, which carried
        an Agent-tool tool_use and stays with task 2561 — see the module
        docstring.)"""
        records = [
            _user(),
            _assistant([_text('I will start by reading the task plan.')]),
        ]
        assert detect_resumable_progress(records) is False

    def test_single_text_only_assistant_turn_flat_nesting_is_false(self) -> None:
        """Same proven-empty shape under the flat ``record['content']`` nesting."""
        records = [
            _user(),
            _assistant([_text()], nested=False),
        ]
        assert detect_resumable_progress(records) is False

    def test_user_records_alone_do_not_count_as_progress(self) -> None:
        """Only ASSISTANT turns are evidence the agent did anything; a pile of
        user records is still a session that produced nothing."""
        records = [_user('do the task'), _user('are you there?'), _user('hello?')]
        assert detect_resumable_progress(records) is False

    # ── Real progress (True) ────────────────────────────────────────────────

    def test_assistant_tool_use_is_true(self) -> None:
        """A tool call is durable evidence the agent DID something."""
        records = [
            _user(),
            _assistant([_text('reading the plan'), _tool_use('Read')]),
        ]
        assert detect_resumable_progress(records) is True

    def test_assistant_tool_use_flat_nesting_is_true(self) -> None:
        """tool_use must be found under the flat nesting too."""
        records = [_user(), _assistant([_tool_use('Bash', command='ls')], nested=False)]
        assert detect_resumable_progress(records) is True

    def test_two_text_only_assistant_turns_is_true(self) -> None:
        """Deliberately protects prose-only workers (synthesis, judge, review
        agents) whose accumulated reasoning IS the thing worth resuming."""
        records = [
            _user(),
            _assistant([_text('first, the trade-offs are ...')]),
            _assistant([_text('on balance I recommend ...')]),
        ]
        assert detect_resumable_progress(records) is True

    def test_many_text_only_assistant_turns_is_true(self) -> None:
        """A long prose-only run must never be discarded."""
        records = [_user()] + [_assistant([_text(f'para {i}')]) for i in range(12)]
        assert detect_resumable_progress(records) is True

    def test_tool_use_in_single_assistant_turn_beats_the_turn_count(self) -> None:
        """<=1 assistant turn is only half the False condition — a lone turn
        that made a tool call is genuine progress."""
        assert detect_resumable_progress([_assistant([_tool_use('Write')])]) is True

    # ── Fail-safe: ambiguity NEVER yields False ─────────────────────────────

    def test_none_records_is_true(self) -> None:
        """An unreadable transcript is ambiguous, not proven-empty → resume."""
        assert detect_resumable_progress(None) is True

    def test_non_dict_records_are_true(self) -> None:
        """Garbage records cannot prove emptiness → resume, and never raise."""
        # list[Any] because these records are DELIBERATELY off-contract: the
        # annotation keeps the hostile intent visible rather than letting the
        # checker narrow to whatever this particular garbage happens to be.
        records: list[Any] = ['nonsense', 42, None, ['x']]
        assert detect_resumable_progress(records) is True

    def test_mixed_garbage_and_text_turn_is_true(self) -> None:
        """A single text-only turn ALONGSIDE unparseable records is ambiguous:
        the garbage may itself have been an assistant turn."""
        records = [_user(), 'garbage', _assistant([_text()])]
        assert detect_resumable_progress(records) is True

    def test_assistant_with_non_list_content_is_true(self) -> None:
        """An unrecognised content shape is ambiguous → resume."""
        assert detect_resumable_progress(
            [{'type': 'assistant', 'message': {'content': 'a bare string'}}]
        ) is True

    def test_assistant_with_no_content_key_is_true(self) -> None:
        """A record with no content at all is an unknown shape → resume."""
        assert detect_resumable_progress([{'type': 'assistant'}]) is True

    def test_non_dict_blocks_are_true(self) -> None:
        """Blocks that are not dicts are ambiguous → resume, and never raise."""
        assert detect_resumable_progress([_assistant(['a string block', 7, None])]) is True

    def test_blocks_missing_type_key_are_true(self) -> None:
        """A block with no 'type' could be anything, including a tool call."""
        assert detect_resumable_progress(
            [_assistant([{'name': 'Read', 'input': {'file_path': '/x'}}])]
        ) is True

    def test_unknown_block_type_is_true(self) -> None:
        """A block type this predicate does not model must not be read as
        emptiness.  'thinking' is NOT a future shape -- it is present in all 89
        measured orchestrator transcripts -- so this arm is live today; see the
        block-classification comment in
        ``shared/src/shared/cli_invoke.py::detect_resumable_progress``."""
        assert detect_resumable_progress(
            [_assistant([{'type': 'server_tool_use', 'name': 'WebSearch'}])]
        ) is True

    def test_malformed_input_never_raises(self) -> None:
        """Totality contract: every hostile shape returns a bool, never raises."""
        # list[Any] for the same reason as test_non_dict_records_are_true: every
        # entry below is an intentionally off-contract shape, and the annotation
        # says so instead of narrowing to the union they accidentally form.
        hostile: tuple[list[Any] | None, ...] = (
            None,
            [],
            [None],
            [{}],
            [{'type': 'assistant', 'message': None}],
            [{'type': 'assistant', 'message': {'content': None}}],
            [{'type': 'assistant', 'content': {'not': 'a list'}}],
            [{'type': None}],
            ['', 0, False, ()],
        )
        for records in hostile:
            assert isinstance(detect_resumable_progress(records), bool)


class TestResumableProgressForSession:
    """The session-level wrapper: locate the on-disk transcript via
    ``read_transcript_records``, apply the pure predicate, never raise."""

    def test_text_only_session_has_no_resumable_progress(self, tmp_path) -> None:
        """The covered shape on disk: a reachable top-level transcript whose
        agent only stated an intention, never calling a tool → False, so the
        cap-hit branch retries FRESH.  (Census 1.2's own specimen is not this
        shape — see the module docstring.)"""
        _write_transcript(
            tmp_path,
            'sess-empty',
            [_user(), _assistant([_text('I will start by reading the plan.')])],
        )
        assert resumable_progress_for_session(tmp_path, 'sess-empty') is False

    def test_tool_use_session_has_resumable_progress(self, tmp_path) -> None:
        """A transcript with a real tool call → True → resume as today."""
        _write_transcript(
            tmp_path,
            'sess-worked',
            [_user(), _assistant([_text('reading'), _tool_use('Read')])],
        )
        assert resumable_progress_for_session(tmp_path, 'sess-worked') is True

    def test_multi_turn_prose_session_has_resumable_progress(self, tmp_path) -> None:
        """A prose-only worker's accumulated reasoning is worth resuming."""
        _write_transcript(
            tmp_path,
            'sess-prose',
            [_user(), _assistant([_text('first ...')]), _assistant([_text('therefore ...')])],
        )
        assert resumable_progress_for_session(tmp_path, 'sess-prose') is True

    def test_missing_transcript_is_true(self, tmp_path) -> None:
        """No transcript file on disk → read_transcript_records returns None →
        True.  The fail-safe direction is INVERTED relative to the background
        detector: an unreadable transcript must resume rather than silently
        discard work.  (The cap-hit branch's own reachability arm handles a
        genuinely absent transcript first, with its more specific message.)"""
        assert resumable_progress_for_session(tmp_path, 'sess-nonexistent') is True

    def test_truncated_final_line_is_tolerated(self, tmp_path) -> None:
        """A SIGKILL leaves a half-written final line.  The tolerant parse in
        read_transcript_records skips it; the surviving tool_use still counts,
        and nothing raises."""
        transcript = _write_transcript(
            tmp_path,
            'sess-truncated',
            [_user(), _assistant([_tool_use('Read')])],
        )
        with transcript.open('a', encoding='utf-8') as fh:
            fh.write('{"type": "assistant", "message": {"cont')
        assert resumable_progress_for_session(tmp_path, 'sess-truncated') is True

    def test_truncated_final_line_on_empty_session_is_tolerated(self, tmp_path) -> None:
        """Same truncation over a text-only session: the parse still must not
        raise, and the proven-empty verdict still stands."""
        transcript = _write_transcript(
            tmp_path,
            'sess-truncated-empty',
            [_user(), _assistant([_text()])],
        )
        with transcript.open('a', encoding='utf-8') as fh:
            fh.write('{"type": "assis')
        assert resumable_progress_for_session(tmp_path, 'sess-truncated-empty') is False

    def test_empty_transcript_file_is_false(self, tmp_path) -> None:
        """A transcript that exists but parsed to zero records is proven-empty
        — this is precisely the 'reachable but carries nothing' gap."""
        slug_dir = tmp_path / 'projects' / 'myproject'
        slug_dir.mkdir(parents=True, exist_ok=True)
        (slug_dir / 'sess-blank.jsonl').write_text('')
        assert resumable_progress_for_session(tmp_path, 'sess-blank') is False

    def test_never_raises_on_unreadable_config_dir(self, tmp_path) -> None:
        """Totality contract inherited from read_transcript_records."""
        assert isinstance(
            resumable_progress_for_session(tmp_path / 'does' / 'not' / 'exist', 'sess-x'),
            bool,
        )
