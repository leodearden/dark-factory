"""Tests for the background-task abandonment detector + downgrade (task 2761).

Layer 2 of the headless ``--print`` background-task footgun fix: detect when an
otherwise-successful ``claude --print`` run ended its turn while a backgrounded
Bash command was still pending (launched via ``Bash run_in_background=true``,
never subsequently polled with ``BashOutput`` or killed with
``KillShell``/``KillBash``), and downgrade ``success``→failure so existing
non-success handling retries/resumes instead of proceeding on a half-done tree.

RCA: Reify 5164's amender ended its turn (681s, 19 turns, subtype=success,
timed_out=false) "to wait for the completion notification" while a 2700s
backgrounded OCCT test was still pending — the CLI treated the conversation as
complete and exited success, silently abandoning the work mid-task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.cli_invoke import (
    AgentResult,
    _parse_claude_output,
    _SubprocessResult,
    detect_ended_awaiting_background,
    ended_awaiting_background_for_session,
)


# ── Fixture builders ────────────────────────────────────────────────────────
# Hand-authored assistant/user record dicts mirroring the on-disk transcript
# JSONL shape: an assistant record carries content blocks either nested under
# ``record['message']['content']`` (the real CLI shape) or flat under
# ``record['content']`` (the shape used by the existing read_transcript_records
# tests).  tool_use blocks carry ``name`` + ``input``.


def _assistant(blocks: list, *, nested: bool = True) -> dict:
    """An assistant transcript record wrapping *blocks*.

    ``nested=True`` → record['message']['content'] (real CLI shape).
    ``nested=False`` → flat record['content'].
    """
    if nested:
        return {'type': 'assistant', 'message': {'role': 'assistant', 'content': blocks}}
    return {'type': 'assistant', 'content': blocks}


def _bash_launch(*, background: bool = True, command: str = 'sleep 9999') -> dict:
    return {
        'type': 'tool_use',
        'name': 'Bash',
        'input': {'command': command, 'run_in_background': background},
    }


def _bash_output(shell_id: str = 'sh-1') -> dict:
    return {'type': 'tool_use', 'name': 'BashOutput', 'input': {'bash_id': shell_id}}


def _kill_shell(shell_id: str = 'sh-1', *, name: str = 'KillShell') -> dict:
    return {'type': 'tool_use', 'name': name, 'input': {'shell_id': shell_id}}


def _text(text: str = 'thinking') -> dict:
    return {'type': 'text', 'text': text}


def _write_transcript(base: Path, session_id: str, records: list) -> Path:
    """Write *records* as a JSONL transcript under
    base/projects/<slug>/<session_id>.jsonl — mirrors the fixture style of the
    existing read_transcript_records / count_transcript_turns tests."""
    slug_dir = base / 'projects' / 'myproject'
    slug_dir.mkdir(parents=True, exist_ok=True)
    transcript = slug_dir / f'{session_id}.jsonl'
    transcript.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    return transcript


class TestDetectEndedAwaitingBackground:
    """The pure detector fires (True) iff the session's final
    background-management action was a launch never followed by a poll/kill —
    i.e. index(last background launch) > index(last reap)."""

    def test_empty_list_is_false(self) -> None:
        """No records → no launch → False (fail-safe)."""
        assert detect_ended_awaiting_background([]) is False

    def test_no_background_launch_is_false(self) -> None:
        """A foreground Bash (run_in_background falsy/absent) is not a launch → False."""
        records = [
            _assistant([_text('run the tests')]),
            _assistant([_bash_launch(background=False, command='pytest -q')]),
            _assistant([{'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'ls'}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_single_abandoned_launch_is_true(self) -> None:
        """A single background launch with nothing after it → True (the RCA)."""
        records = [
            _assistant([_text('kick off the long build')]),
            _assistant([_bash_launch(command='./occt-test.sh')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_launch_then_bashoutput_is_false(self) -> None:
        """Launch later polled by BashOutput → engaged → False."""
        records = [
            _assistant([_bash_launch()]),
            _assistant([_bash_output()]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_then_killshell_is_false(self) -> None:
        """Launch later killed by KillShell → engaged → False."""
        records = [
            _assistant([_bash_launch()]),
            _assistant([_kill_shell()]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_then_killbash_is_false(self) -> None:
        """KillBash is also a reap (older CLI tool name) → False."""
        records = [
            _assistant([_bash_launch()]),
            _assistant([_kill_shell(name='KillBash')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_second_launch_with_reap_before_it_is_true(self) -> None:
        """Two launches; the reap sits BEFORE the second launch, which is never
        reaped → last launch after last reap → True."""
        records = [
            _assistant([_bash_launch(command='job-a')]),
            _assistant([_kill_shell()]),
            _assistant([_bash_launch(command='job-b')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_launch_reaped_then_second_launch_abandoned_is_true(self) -> None:
        """launch → BashOutput (reaps first) → a SECOND launch left abandoned → True."""
        records = [
            _assistant([_bash_launch(command='job-a')]),
            _assistant([_bash_output()]),
            _assistant([_bash_launch(command='job-b')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_tolerant_to_mixed_nesting_and_malformed_blocks(self) -> None:
        """Nested + flat records, plus malformed/missing blocks interleaved →
        no exception, correct verdict (True — the launch is the last action)."""
        records = [
            {'type': 'assistant', 'message': {'role': 'assistant'}},  # message w/o content
            {'type': 'assistant', 'content': None},  # content not a list
            _assistant([_text('planning')], nested=False),  # flat text
            {'type': 'user', 'content': [{'type': 'tool_result', 'content': 'x'}]},
            _assistant(['not-a-dict-block', 42, {'type': 'text'}], nested=False),  # junk blocks
            _assistant([{'type': 'tool_use', 'name': 'Bash'}]),  # Bash tool_use w/o input
            _assistant([_bash_launch(command='./occt-test.sh')], nested=True),  # the launch
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_malformed_records_only_is_false(self) -> None:
        """None / non-dict records and empty blocks never raise → False."""
        records = [
            None,  # type: ignore[list-item]
            'garbage',  # type: ignore[list-item]
            {'type': 'assistant'},  # no content at all
            {'no_type': True, 'content': []},
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_reap_after_launch_across_nesting_styles_is_false(self) -> None:
        """A launch (nested) later reaped by a BashOutput authored in the flat
        shape → still recognised as a reap → False."""
        records = [
            _assistant([_bash_launch()], nested=True),
            _assistant([_bash_output()], nested=False),
        ]
        assert detect_ended_awaiting_background(records) is False


class TestEndedAwaitingBackgroundForSession:
    """File-reading seam: mirrors count_transcript_turns' shape — delegate to
    read_transcript_records, then the pure detector, mapping None (unresolvable
    / missing transcript) → False (fail-safe)."""

    def test_missing_transcript_is_false(self, tmp_path: Path) -> None:
        """No matching transcript file → read_transcript_records None → False."""
        (tmp_path / 'projects' / 'myproject').mkdir(parents=True, exist_ok=True)
        assert (
            ended_awaiting_background_for_session(tmp_path, 'no-such-session') is False
        )

    def test_abandoned_launch_transcript_is_true(self, tmp_path: Path) -> None:
        """A transcript whose tail launched but never reaped a background task → True."""
        sid = 'sess-bg-abandoned'
        _write_transcript(
            tmp_path,
            sid,
            [
                _assistant([_text('start the long build')]),
                _assistant([_bash_launch(command='./occt-test.sh')]),
            ],
        )
        assert ended_awaiting_background_for_session(tmp_path, sid) is True

    def test_polled_launch_transcript_is_false(self, tmp_path: Path) -> None:
        """A transcript where the launch is later polled by BashOutput → False."""
        sid = 'sess-bg-polled'
        _write_transcript(
            tmp_path,
            sid,
            [
                _assistant([_bash_launch()]),
                _assistant([_bash_output()]),
            ],
        )
        assert ended_awaiting_background_for_session(tmp_path, sid) is False

    def test_empty_transcript_is_false(self, tmp_path: Path) -> None:
        """A located-but-empty transcript → records [] → False (never raises)."""
        sid = 'sess-bg-empty'
        _write_transcript(tmp_path, sid, [])
        assert ended_awaiting_background_for_session(tmp_path, sid) is False


# A clean success envelope: subtype='success', is_error absent, returncode 0 —
# the exact shape a run that ended-awaiting-background produces (Reify 5164).
_SUCCESS_ENVELOPE = json.dumps(
    {
        'result': 'done',
        'subtype': 'success',
        'cost_usd': 0.02,
        'duration_ms': 681_000,
        'num_turns': 19,
        'session_id': 'sess-bg-downgrade',
    }
)


class TestEndedAwaitingBackgroundField:
    """The new boolean outcome flag mirrors schema_salvaged / schema_tool_denied:
    it defaults False on both AgentResult and _SubprocessResult."""

    def test_agent_result_defaults_false(self) -> None:
        """AgentResult.ended_awaiting_background defaults to False."""
        result = AgentResult(success=True, output='ok')
        assert result.ended_awaiting_background is False

    def test_subprocess_result_defaults_false(self) -> None:
        """_SubprocessResult.ended_awaiting_background defaults to False."""
        result = _SubprocessResult(stdout='', stderr='', returncode=0, duration_ms=0)
        assert result.ended_awaiting_background is False


class TestParseClaudeOutputDowngrade:
    """_parse_claude_output downgrades success→failure when the subprocess was
    flagged ended_awaiting_background, and propagates the flag on every path."""

    def test_success_envelope_flagged_is_downgraded(self) -> None:
        """subtype=success + ended_awaiting_background=True → success False, flag True."""
        sub = _SubprocessResult(
            stdout=_SUCCESS_ENVELOPE,
            stderr='',
            returncode=0,
            duration_ms=681_000,
            ended_awaiting_background=True,
        )
        agent = _parse_claude_output(sub)
        assert agent.success is False
        assert agent.ended_awaiting_background is True

    def test_success_envelope_unflagged_stays_success(self) -> None:
        """subtype=success + ended_awaiting_background=False → success stays True, flag False."""
        sub = _SubprocessResult(
            stdout=_SUCCESS_ENVELOPE,
            stderr='',
            returncode=0,
            duration_ms=681_000,
            ended_awaiting_background=False,
        )
        agent = _parse_claude_output(sub)
        assert agent.success is True
        assert agent.ended_awaiting_background is False

    def test_flag_propagated_on_empty_stdout_without_flipping_success(self) -> None:
        """An already-failing empty-stdout result + flag=True → success stays
        False and the flag is still propagated (downgrade is an idempotent no-op
        on an already-failing result; no crash)."""
        sub = _SubprocessResult(
            stdout='',
            stderr='boom',
            returncode=1,
            duration_ms=100,
            ended_awaiting_background=True,
        )
        agent = _parse_claude_output(sub)
        assert agent.success is False
        assert agent.ended_awaiting_background is True
