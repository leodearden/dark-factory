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

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    _parse_claude_output,
    _SubprocessResult,
    classify_agent_failure,
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


def _bash_launch(
    *,
    background: bool = True,
    command: str = 'sleep 9999',
    tool_id: str = 'toolu_1',
) -> dict:
    """A Bash tool_use block.  Real CLI blocks carry an ``id``; it is emitted
    unconditionally (with a default) so the pre-existing tests are unaffected
    while the bg-log-reap tests can correlate a launch with its tool_result."""
    return {
        'type': 'tool_use',
        'id': tool_id,
        'name': 'Bash',
        'input': {'command': command, 'run_in_background': background},
    }


def _bg_log_path(task_id: str = 'bz89cdzmh') -> str:
    """The CLI's background-output path shape: the task id is a substring of it."""
    return f'/tmp/claude-1000/-w-3639/sess/tasks/{task_id}.output'


def _bg_launch_result(
    tool_id: str = 'toolu_1',
    task_id: str = 'bz89cdzmh',
    log_path: str | None = None,
    *,
    structured: bool = True,
    in_text: bool = True,
) -> dict:
    """The ``user`` tool_result record the CLI emits for a background launch.

    Mirrors the REAL shape sampled from ``~/.claude/projects/*/*.jsonl``: the
    record carries BOTH a structured ``toolUseResult.backgroundTaskId`` and a
    result-text sentence naming the output file.  ``structured=False`` drops the
    former and ``in_text=False`` drops the latter, so the detector's two
    token-extraction sources can be exercised independently.
    """
    log_path = log_path if log_path is not None else _bg_log_path(task_id)
    text = f'Command running in background with ID: {task_id}.'
    if in_text:
        text += (
            f' Output is being written to: {log_path}.'
            ' You will be notified when it completes.'
            ' To check interim output, use Read on that file path.'
        )
    record: dict = {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [
                {'type': 'tool_result', 'tool_use_id': tool_id, 'content': text},
            ],
        },
    }
    if structured:
        record['toolUseResult'] = {'stdout': '', 'backgroundTaskId': task_id}
    return record


def _fg_bash(command: str) -> dict:
    """A FOREGROUND Bash tool_use (no run_in_background) — not a launch."""
    return {'type': 'tool_use', 'name': 'Bash', 'input': {'command': command}}


# A second bg-log path used by the malformed-record tolerance test.
LOG_ALT = _bg_log_path('zz99alt00')


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

    def test_poll_once_still_running_then_end_is_false_documented_gap(self) -> None:
        """DOCUMENTED, ACCEPTED false negative (design tradeoff, not a bug): an
        agent that launches a background task, polls it ONCE with BashOutput
        while it is still running, then ends its turn "to wait for the
        completion notification" → the poll is the last background action (reap
        after launch) → False.

        This is arguably the *primary* Reify-5164 abandonment shape — the amender
        described ending its turn to wait for completion, and a single natural
        poll before ending lands exactly here. The detector cannot distinguish
        "polled, saw it FINISHED, moved on" (a genuine clear) from "polled, saw
        it STILL RUNNING, ended anyway" (this abandonment) without parsing
        free-form BashOutput result text, which is deliberately avoided (fragile
        across CLI versions). It errs toward False because a success→failure
        downgrade must never re-run a possibly-complete task. Pinned here so this
        coverage boundary is explicit and reviewed, not only prose-documented in
        the detector's docstring. (Cf. test_launch_then_bashoutput_is_false,
        which pins the same verdict for the FINISHED-then-moved-on shape.)
        """
        records = [
            _assistant([_text('kick off the long OCCT test')]),
            _assistant([_bash_launch(command='./occt-test.sh')]),
            _assistant([_bash_output()]),  # polled once — still running
            _assistant([_text('I will wait for the background test to finish')]),
        ]
        assert detect_ended_awaiting_background(records) is False


class TestForegroundBgLogReadIsAReap:
    """A FOREGROUND read of a background launch's output file is a reap (task
    3639).

    The `_lane-31` specimen: an agent launches `cargo test` with
    ``run_in_background=true``, then reads the CLI-reported bg-log with a
    foreground ``tail``/``cat``/``grep`` — or with ``Read``, which the CLI's own
    launch message explicitly recommends — and never calls ``BashOutput``.  That
    session demonstrably engaged with its pending work, so the abandonment
    verdict must not fire.  Identity comes from the launch's tool_result: the
    structured ``toolUseResult.backgroundTaskId`` and the path captured from the
    ``Output is being written to: <path>`` sentence.
    """

    LOG = _bg_log_path()

    def test_foreground_tail_of_bg_log_is_false(self) -> None:
        """The `_lane-31` shape: launch → its tool_result → foreground `tail` of
        the reported bg-log → engaged → False."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(),
            _assistant([_fg_bash(f'tail -50 {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_foreground_cat_of_bg_log_is_false(self) -> None:
        """`cat <bg-log>` is the same reap, with no shell-verb enumeration."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(),
            _assistant([_fg_bash(f'cat {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_foreground_grep_of_bg_log_is_false(self) -> None:
        """`grep -c FAILED <bg-log>` likewise."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(),
            _assistant([_fg_bash(f'grep -c FAILED {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_read_tool_of_bg_log_is_false(self) -> None:
        """`Read` on the bg-log path — the reap the CLI's own launch message
        recommends ("use Read on that file path") → False."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(),
            _assistant([{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': self.LOG}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_grep_tool_of_bg_log_is_false(self) -> None:
        """`Grep` with the bg-log as ``path`` → tool-agnostic token match → False."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(),
            _assistant(
                [
                    {
                        'type': 'tool_use',
                        'name': 'Grep',
                        'input': {'pattern': 'test result', 'path': self.LOG},
                    }
                ]
            ),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_token_from_structured_field_only_is_false(self) -> None:
        """Token recovered from ``toolUseResult.backgroundTaskId`` alone (the
        result text carries no ``Output is being written to:`` sentence) — the
        task id is a substring of the log path, so the later `tail` matches."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(structured=True, in_text=False),
            _assistant([_fg_bash(f'tail -50 {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_token_from_result_text_only_is_false(self) -> None:
        """Token recovered from the result TEXT alone (no structured
        ``toolUseResult``) — tolerates CLI versions emitting only one source."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(structured=False, in_text=True),
            _assistant([_fg_bash(f'tail -50 {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_with_nothing_after_still_true(self) -> None:
        """2761 PRESERVATION: launch → its tool_result → nothing.  The genuine
        Reify-5164 abandonment shape must still fire."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_unrelated_foreground_activity_still_true(self) -> None:
        """2761 PRESERVATION: post-launch activity that never references the
        bg-log token is not a reap → still True (no blanket "did anything
        afterwards" mute)."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            _bg_launch_result(),
            _assistant([_fg_bash('git status')]),
            _assistant(
                [{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/etc/hosts'}}]
            ),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_second_unreaped_launch_after_reaped_first_is_true(self) -> None:
        """launch A → A's result → read A's log → launch B → B's result →
        nothing: the reap is positional, so B's abandonment still fires."""
        records = [
            _assistant([_bash_launch(command='job-a', tool_id='toolu_a')]),
            _bg_launch_result(tool_id='toolu_a', task_id='aaa11111'),
            _assistant([_fg_bash(f'tail -50 {_bg_log_path("aaa11111")}')]),
            _assistant([_bash_launch(command='job-b', tool_id='toolu_b')]),
            _bg_launch_result(tool_id='toolu_b', task_id='bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_malformed_user_records_never_raise(self) -> None:
        """Malformed ``user`` records (toolUseResult not a dict, content not a
        list, tool_use_id absent, backgroundTaskId empty/non-string) are skipped
        rather than raising, and a text-derived token from a well-formed sibling
        still clears the verdict → False."""
        records = [
            _assistant([_bash_launch(command='cargo test --all')]),
            {'type': 'user', 'toolUseResult': 'not-a-dict', 'message': {'content': 'not-a-list'}},
            {'type': 'user', 'toolUseResult': {'backgroundTaskId': ''}, 'content': None},
            {'type': 'user', 'toolUseResult': {'backgroundTaskId': 42}, 'content': [None, 7]},
            {
                'type': 'user',
                'content': [
                    {'type': 'tool_result', 'content': f'Output is being written to: {LOG_ALT}.'}
                ],
            },
            _assistant([_fg_bash(f'tail -50 {LOG_ALT}')]),
        ]
        assert detect_ended_awaiting_background(records) is False


class TestTaskToolReaps:
    """``TaskOutput`` and ``TaskStop`` are reaps (task 3639) — the Task-tool
    analogues of ``BashOutput``/``KillShell``: one collects a backgrounded
    Task/subagent's result, the other terminates it.  Both are equally
    conclusive evidence the session engaged with its pending work."""

    def test_task_output_after_launch_is_false(self) -> None:
        """TaskOutput collects a backgrounded task's result → reap → False."""
        records = [
            _assistant([_bash_launch()]),
            _assistant([{'type': 'tool_use', 'name': 'TaskOutput', 'input': {'task_id': 't1'}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_task_stop_after_launch_is_false(self) -> None:
        """TaskStop terminates a backgrounded task → reap → False."""
        records = [
            _assistant([_bash_launch()]),
            _assistant([{'type': 'tool_use', 'name': 'TaskStop', 'input': {'task_id': 't1'}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_task_output_before_second_launch_is_true(self) -> None:
        """The reap sits BEFORE a second, unreaped launch → still True (mirrors
        test_second_launch_with_reap_before_it_is_true)."""
        records = [
            _assistant([_bash_launch(command='job-a')]),
            _assistant([{'type': 'tool_use', 'name': 'TaskOutput', 'input': {'task_id': 't1'}}]),
            _assistant([_bash_launch(command='job-b')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_task_output_in_flat_record_shape_is_false(self) -> None:
        """The reap authored in the flat (nested=False) record shape is still
        recognised (mirrors test_reap_after_launch_across_nesting_styles_is_false)."""
        records = [
            _assistant([_bash_launch()], nested=True),
            _assistant(
                [{'type': 'tool_use', 'name': 'TaskOutput', 'input': {'task_id': 't1'}}],
                nested=False,
            ),
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

    def test_lane31_specimen_transcript_is_false(self, tmp_path: Path) -> None:
        """End-to-end replay of the `_lane-31` specimen that motivated task 3639:
        FOUR background `cargo test` launches, each followed by its launch
        tool_result and a FOREGROUND `tail` of its own bg-log, ZERO
        BashOutput/KillShell/KillBash anywhere, final action a review-verdict
        submission → the run completed and must NOT be downgraded."""
        sid = 'sess-lane31'
        records: list = [_assistant([_text('run the crate test suites in parallel')])]
        for n in range(4):
            tool_id = f'toolu_{n}'
            task_id = f'bg{n}0000{n}'
            records.append(
                _assistant([_bash_launch(command=f'cargo test -p crate{n}', tool_id=tool_id)])
            )
            records.append(_bg_launch_result(tool_id=tool_id, task_id=task_id))
            records.append(_assistant([_fg_bash(f'tail -50 {_bg_log_path(task_id)}')]))
        records.append(
            _assistant(
                [
                    {
                        'type': 'tool_use',
                        'name': 'mcp__verdict-tools__submit_review_verdict',
                        'input': {'verdict': 'APPROVE'},
                    }
                ]
            )
        )
        _write_transcript(tmp_path, sid, records)
        assert ended_awaiting_background_for_session(tmp_path, sid) is False

    def test_lane31_shaped_transcript_ending_in_unreaped_launch_is_true(
        self, tmp_path: Path
    ) -> None:
        """Control for the specimen replay: the same transcript shape whose TAIL
        is an unreaped launch → the genuine abandonment still fires (True)."""
        sid = 'sess-lane31-abandoned'
        records: list = [
            _assistant([_bash_launch(command='cargo test -p crate0', tool_id='toolu_0')]),
            _bg_launch_result(tool_id='toolu_0', task_id='bg000000'),
            _assistant([_fg_bash(f'tail -50 {_bg_log_path("bg000000")}')]),
            _assistant([_bash_launch(command='cargo test -p crate1', tool_id='toolu_1')]),
            _bg_launch_result(tool_id='toolu_1', task_id='bg111111'),
            _assistant([_text('I will wait for the background test to finish')]),
        ]
        _write_transcript(tmp_path, sid, records)
        assert ended_awaiting_background_for_session(tmp_path, sid) is True


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


class TestClassifyEndedAwaitingBackground:
    """The classification layer routes a downgraded ended-awaiting-background
    result to a dedicated kind, placed after the OK/SUCCESS check (so it never
    shadows a genuine success) and before the timed_out rule (which it must not
    shadow either, since the two flags are mutually exclusive by construction)."""

    def test_failure_kind_exists(self) -> None:
        """AgentFailureKind.ENDED_AWAITING_BACKGROUND exists."""
        assert AgentFailureKind.ENDED_AWAITING_BACKGROUND

    def test_downgraded_result_classified_as_ended_awaiting_background(self) -> None:
        """success=False + ended_awaiting_background=True → dedicated kind, summary
        mentions background."""
        result = AgentResult(
            success=False,
            output='done',
            subtype='success',
            turns=19,
            ended_awaiting_background=True,
        )
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.ENDED_AWAITING_BACKGROUND
        assert 'background' in cls.summary.lower()

    def test_plain_success_not_misfired(self) -> None:
        """A genuine success (flag False) stays SUCCESS — the new rule, placed
        after the OK check, cannot misfire."""
        result = AgentResult(
            success=True, output='ok', subtype='success', turns=3,
            ended_awaiting_background=False,
        )
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.SUCCESS

    def test_timed_out_not_shadowed(self) -> None:
        """A timed-out result (flag False) still classifies as TIMED_OUT — the
        new rule does not shadow the timeout rule below it."""
        result = AgentResult(
            success=False, output='', subtype='error_empty_output',
            duration_ms=300_000, timed_out=True, ended_awaiting_background=False,
        )
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.TIMED_OUT

    def test_diagnostic_detail_still_renders(self) -> None:
        """diagnostic_detail keeps its full signal dump (transcript_turns line)."""
        result = AgentResult(
            success=False, output='done', subtype='success',
            ended_awaiting_background=True, transcript_turns=19,
        )
        cls = classify_agent_failure(result)
        assert 'transcript_turns=' in cls.diagnostic_detail
