"""Tests for the background-task abandonment detector + downgrade (task 2761).

Layer 2 of the headless ``--print`` background-task footgun fix: detect when an
otherwise-successful ``claude --print`` run ended its turn while a background
task it started was still pending, and downgrade ``success``→failure so existing
non-success handling retries/resumes instead of proceeding on a half-done tree.
What counts as a launch is defined once, by the contract in
``shared.cli_invoke.detect_ended_awaiting_background``'s docstring.

A session reaps by calling a background-management tool named in
``_BACKGROUND_REAP_TOOLS``, by issuing any tool_use whose input references the
task's id or output-file path (the tool-agnostic clause, task 3639), or by
having a task-notification naming the task delivered to it (task 5986).
The tool names appearing in the fixtures below are test DATA standing in for
whatever a CLI build might emit, not a claim that each is live — only
``TaskStop`` is (task 5332).  Exercising the accept set's full membership is
the coverage that should stay; see the comment on ``_BACKGROUND_REAP_TOOLS``
for why membership is kept rather than resynced, and
orchestrator/tests/test_roles_harness_tool_inventory.py::MEASURED_ABSENT_TOOLS
for the measurement that says which names are live.

RCA: Reify 5164's amender ended its turn (681s, 19 turns, subtype=success,
timed_out=false) "to wait for the completion notification" while a 2700s
backgrounded OCCT test was still pending — the CLI treated the conversation as
complete and exited success, silently abandoning the work mid-task.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from _transcript_fixtures import (
    assistant_record,
    bash_launch,
    bg_launch_result,
    bg_log_path,
    confirmed_bash_launch,
)

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    SubprocessResult,
    _content_blocks,
    _iter_result_texts,
    classify_agent_failure,
    detect_ended_awaiting_background,
    ended_awaiting_background_for_session,
    parse_claude_output,
)

# ── Fixture builders ────────────────────────────────────────────────────────
# The confirmed-launch records come from _transcript_fixtures (shared with the
# other suites that build a transcript ending on one); the builders below are
# this suite's own.


def _user_result(tool_id: str, content: str, tool_use_result: object, *, is_error: bool) -> dict:
    return {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': tool_id,
                    'content': content,
                    'is_error': is_error,
                },
            ],
        },
        'toolUseResult': tool_use_result,
    }


def _ran_in_foreground_result(
    tool_id: str = 'toolu_1', stdout: str = 'PASS: 17 | FAIL: 0 | SKIP: 3'
) -> dict:
    """The result of a run_in_background call the CLI ran in the FOREGROUND:
    finished output, no background id.  Shape of reify 4996 lane-10, session
    a88d703e-2307-40b9-8460-8448202ab597."""
    return _user_result(
        tool_id,
        stdout,
        {
            'stdout': stdout,
            'stderr': '',
            'interrupted': False,
            'isImage': False,
            'noOutputExpected': False,
        },
        is_error=False,
    )


def _rejected_result(tool_id: str = 'toolu_1') -> dict:
    """The result of a call the user REJECTED.  Shape of dark-factory 3395,
    session 34bc8499-8e42-41cc-b726-f5c9797ce123."""
    return _user_result(
        tool_id,
        "The user doesn't want to proceed with this tool use. The tool use was"
        ' rejected (eg. if it was a file edit, the new_string was NOT written to'
        ' the file). STOP what you are doing and wait for the user to tell you'
        ' how to proceed.',
        'User rejected tool use',
        is_error=True,
    )


def _auto_backgrounded_result(tool_id: str = 'toolu_fg', task_id: str = 'bd9xuykdj') -> dict:
    """The result of a FOREGROUND Bash the CLI moved to the background after its
    timeout.  Shape of dark-factory 3799, session
    14a8ad8b-2a6b-405a-afcc-ae084b3dce9a."""
    return _user_result(
        tool_id,
        'Command did not complete within its 120s timeout and was moved to the'
        f' background (ID: {task_id}). Output is being written to:'
        f' {bg_log_path(task_id)}. You will be notified when it completes.'
        ' To check interim output, use Read on that file path.',
        {
            'stdout': '',
            'stderr': '',
            'interrupted': False,
            'isImage': False,
            'noOutputExpected': False,
            'backgroundTaskId': task_id,
            'timedOutAfterMs': 120000,
        },
        is_error=False,
    )


def _fg_bash(command: str, *, tool_id: str | None = None) -> dict:
    """A FOREGROUND Bash tool_use (no run_in_background) — not a launch."""
    block: dict = {'type': 'tool_use', 'name': 'Bash', 'input': {'command': command}}
    if tool_id is not None:
        block['id'] = tool_id
    return block


def _verdict_submit(name: str) -> dict:
    """A terminal-submission tool_use block, named exactly as the transcript
    records it (server-prefixed or bare)."""
    return {'type': 'tool_use', 'name': name, 'input': {'verdict': 'APPROVE'}}


# A second bg-log path used by the malformed-record tolerance test.
LOG_ALT = bg_log_path('zz99alt00')


def _bash_output(shell_id: str = 'sh-1') -> dict:
    return {'type': 'tool_use', 'name': 'BashOutput', 'input': {'bash_id': shell_id}}


def _kill_shell(shell_id: str = 'sh-1', *, name: str = 'KillShell') -> dict:
    return {'type': 'tool_use', 'name': name, 'input': {'shell_id': shell_id}}


def _text(text: str = 'thinking') -> dict:
    return {'type': 'text', 'text': text}


MONITOR_TASK_ID = 'byk7waubd'


def _monitor_call(
    *,
    tool_id: str = 'toolu_mon1',
    command: str = 'until [ -s /tmp/x.log ]; do sleep 10; done; tail -8 /tmp/x.log',
    timeout_ms: int = 900000,
) -> dict:
    """A ``Monitor`` tool_use block, in the CLI's measured input shape."""
    return {
        'type': 'tool_use',
        'id': tool_id,
        'name': 'Monitor',
        'input': {
            'command': command,
            'description': 'test run completion',
            'timeout_ms': timeout_ms,
            'persistent': False,
        },
    }


def _monitor_armed(
    tool_id: str = 'toolu_mon1',
    task_id: object = MONITOR_TASK_ID,
    timeout_ms: int = 900000,
    *,
    nested: bool = True,
) -> dict:
    """The ``user`` record confirming a Monitor arm: a tool_result naming the
    call plus the structured ``toolUseResult`` carrying the armed task's id.
    ``nested=False`` puts the content list flat under ``record['content']``."""
    content = [
        {
            'type': 'tool_result',
            'tool_use_id': tool_id,
            'content': (
                f'Monitor started (task {task_id}, timeout {timeout_ms}ms).'
                ' You will be notified on each event.'
                ' Keep working — do not poll or sleep.'
            ),
        },
    ]
    record: dict = {
        'type': 'user',
        'toolUseResult': {'taskId': task_id, 'timeoutMs': timeout_ms, 'persistent': False},
    }
    if nested:
        record['message'] = {'role': 'user', 'content': content}
    else:
        record['content'] = content
    return record


def _monitor_rejected(tool_id: str = 'toolu_mon1') -> dict:
    """The ``user`` record for a Monitor call the CLI REJECTED: nothing armed."""
    return {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': tool_id,
                    'is_error': True,
                    'content': (
                        '<tool_use_error>InputValidationError: Monitor failed due to'
                        ' the following issues:\nThe required parameter `description`'
                        ' is missing</tool_use_error>'
                    ),
                },
            ],
        },
        'toolUseResult': 'InputValidationError: [...]',
    }


# The call and its confirming result, for each launch kind.
_EACH_LAUNCH_KIND = [
    pytest.param(bash_launch(), bg_launch_result(), id='bash'),
    pytest.param(_monitor_call(), _monitor_armed(), id='monitor'),
]


def _monitor_event_text(
    task_id: str = MONITOR_TASK_ID,
    event: str = '4953 passed, 1 skipped in 262.80s',
    summary: str = 'test sweep result',
) -> str:
    """The text of a Monitor event (or expiry) notification."""
    return (
        '<task-notification>\n'
        f'<task-id>{task_id}</task-id>\n'
        f'<summary>Monitor event: "{summary}"</summary>\n'
        f'<event>{event}</event>\n'
        '</task-notification>'
    )


def _bash_completion_text(task_id: str = 'bz89cdzmh', tool_use_id: str = 'toolu_1') -> str:
    """The text of a background Bash completion notification."""
    return (
        '<task-notification>\n'
        f'<task-id>{task_id}</task-id>\n'
        f'<tool-use-id>{tool_use_id}</tool-use-id>\n'
        f'<output-file>{bg_log_path(task_id)}</output-file>\n'
        '<status>completed</status>\n'
        '<summary>Background command "cargo test" completed (exit code 0)</summary>\n'
        '</task-notification>'
    )


def _delivered_mid_turn(text: str) -> dict:
    """A notification delivered to the model inside the running turn."""
    return {
        'type': 'attachment',
        'attachment': {
            'type': 'queued_command',
            'prompt': text,
            'commandMode': 'task-notification',
        },
    }


def _delivered_new_turn(text: str, *, as_blocks: bool = False) -> dict:
    """A notification delivered to the model as the prompt of a new turn.
    ``as_blocks=True`` carries the text as a ``text`` sub-block list."""
    content: object = [{'type': 'text', 'text': text}] if as_blocks else text
    return {
        'type': 'user',
        'message': {'role': 'user', 'content': content},
        'origin': {'kind': 'task-notification', 'producer': 'session-task'},
    }


def _enqueued_only(text: str) -> dict:
    """Queue bookkeeping for a notification: NOT a delivery to the model."""
    return {'type': 'queue-operation', 'operation': 'enqueue', 'content': text}


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
            assistant_record([_text('run the tests')]),
            assistant_record([bash_launch(background=False, command='pytest -q')]),
            assistant_record([{'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'ls'}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_single_abandoned_launch_is_true(self) -> None:
        """A single background launch with nothing after it → True (the RCA)."""
        records = [
            assistant_record([_text('kick off the long build')]),
            *confirmed_bash_launch(command='./occt-test.sh'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_launch_then_bashoutput_is_false(self) -> None:
        """Launch later polled by BashOutput → engaged → False."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_bash_output()]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_then_killshell_is_false(self) -> None:
        """Launch later killed by KillShell → engaged → False."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_kill_shell()]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_then_killbash_is_false(self) -> None:
        """KillBash is also a reap (older CLI tool name) → False."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_kill_shell(name='KillBash')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_second_launch_with_reap_before_it_is_true(self) -> None:
        """Two launches; the reap sits BEFORE the second launch, which is never
        reaped → last launch after last reap → True."""
        records = [
            *confirmed_bash_launch(command='job-a', tool_id='toolu_a', task_id='aaa11111'),
            assistant_record([_kill_shell()]),
            *confirmed_bash_launch(command='job-b', tool_id='toolu_b', task_id='bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_launch_reaped_then_second_launch_abandoned_is_true(self) -> None:
        """launch → BashOutput (reaps first) → a SECOND launch left abandoned → True."""
        records = [
            *confirmed_bash_launch(command='job-a', tool_id='toolu_a', task_id='aaa11111'),
            assistant_record([_bash_output()]),
            *confirmed_bash_launch(command='job-b', tool_id='toolu_b', task_id='bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_tolerant_to_mixed_nesting_and_malformed_blocks(self) -> None:
        """Nested + flat records, plus malformed/missing blocks interleaved →
        no exception, correct verdict (True — the launch is the last action)."""
        records = [
            {'type': 'assistant', 'message': {'role': 'assistant'}},  # message w/o content
            {'type': 'assistant', 'content': None},  # content not a list
            assistant_record([_text('planning')], nested=False),  # flat text
            {'type': 'user', 'content': [{'type': 'tool_result', 'content': 'x'}]},
            assistant_record(['not-a-dict-block', 42, {'type': 'text'}], nested=False),  # junk blocks
            assistant_record([{'type': 'tool_use', 'name': 'Bash'}]),  # Bash tool_use w/o input
            *confirmed_bash_launch(command='./occt-test.sh', nested=True),  # the launch
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
            *confirmed_bash_launch(nested=True),
            assistant_record([_bash_output()], nested=False),
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
            assistant_record([_text('kick off the long OCCT test')]),
            *confirmed_bash_launch(command='./occt-test.sh'),
            assistant_record([_bash_output()]),  # polled once — still running
            assistant_record([_text('I will wait for the background test to finish')]),
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

    LOG = bg_log_path()

    def test_foreground_tail_of_bg_log_is_false(self) -> None:
        """The `_lane-31` shape: launch → its tool_result → foreground `tail` of
        the reported bg-log → engaged → False."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([_fg_bash(f'tail -50 {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_foreground_cat_of_bg_log_is_false(self) -> None:
        """`cat <bg-log>` is the same reap, with no shell-verb enumeration."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([_fg_bash(f'cat {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_foreground_grep_of_bg_log_is_false(self) -> None:
        """`grep -c FAILED <bg-log>` likewise."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([_fg_bash(f'grep -c FAILED {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_read_tool_of_bg_log_is_false(self) -> None:
        """`Read` on the bg-log path — the reap the CLI's own launch message
        recommends ("use Read on that file path") → False.

        This is also the shape
        orchestrator/src/orchestrator/agents/harness_guidance.py::WAIT_PATTERN_GUIDANCE
        now steers every Bash-capable role into (task 5332), which is why
        correcting those prompts needed no change to the detector. Keep it: it
        is the executable form of that claim, and narrowing
        ``_iter_input_strings`` to a ``command`` key fails HERE rather than
        silently downgrading every correctly-behaved session to failure.
        """
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': self.LOG}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_grep_tool_of_bg_log_is_false(self) -> None:
        """`Grep` with the bg-log as ``path`` → tool-agnostic token match → False."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record(
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
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(in_text=False),
            assistant_record([_fg_bash(f'tail -50 {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_token_from_result_text_only_is_false(self) -> None:
        """Token recovered from the result TEXT alone: the structured id names
        another task, so only the reported output path matches the `tail`."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(task_id='qq77other', log_path=self.LOG),
            assistant_record([_fg_bash(f'tail -50 {self.LOG}')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_with_nothing_after_still_true(self) -> None:
        """2761 PRESERVATION: launch → its tool_result → nothing.  The genuine
        Reify-5164 abandonment shape must still fire."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_unrelated_foreground_activity_still_true(self) -> None:
        """2761 PRESERVATION: post-launch activity that never references the
        bg-log token is not a reap → still True (no blanket "did anything
        afterwards" mute)."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([_fg_bash('git status')]),
            assistant_record(
                [{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/etc/hosts'}}]
            ),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_second_unreaped_launch_after_reaped_first_is_true(self) -> None:
        """launch A → A's result → read A's log → launch B → B's result →
        nothing: the reap is positional, so B's abandonment still fires."""
        records = [
            assistant_record([bash_launch(command='job-a', tool_id='toolu_a')]),
            bg_launch_result(tool_id='toolu_a', task_id='aaa11111'),
            assistant_record([_fg_bash(f'tail -50 {bg_log_path("aaa11111")}')]),
            assistant_record([bash_launch(command='job-b', tool_id='toolu_b')]),
            bg_launch_result(tool_id='toolu_b', task_id='bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_malformed_user_records_never_raise(self) -> None:
        """Malformed ``user`` records (toolUseResult not a dict, content not a
        list, tool_use_id absent, backgroundTaskId empty/non-string) are skipped
        rather than raising, and a text-derived token from a well-formed sibling
        still clears the verdict → False."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(in_text=False),
            {'type': 'user', 'toolUseResult': 'not-a-dict', 'message': {'content': 'not-a-list'}},
            {'type': 'user', 'toolUseResult': {'backgroundTaskId': ''}, 'content': None},
            {'type': 'user', 'toolUseResult': {'backgroundTaskId': 42}, 'content': [None, 7]},
            {
                'type': 'user',
                'content': [
                    {'type': 'tool_result', 'content': f'Output is being written to: {LOG_ALT}.'}
                ],
            },
            assistant_record([_fg_bash(f'tail -50 {LOG_ALT}')]),
        ]
        assert detect_ended_awaiting_background(records) is False


class TestTaskToolReaps:
    """``TaskOutput`` and ``TaskStop`` are reaps (task 3639): one collects a
    backgrounded Task/subagent's result, the other terminates it.  Both are
    equally conclusive evidence the session engaged with its pending work.

    Deliberately NOT described as the analogues of
    ``BashOutput``/``KillShell``, as this docstring used to: that framing is
    doubly wrong.  ``BashOutput`` was never live in this fleet, and
    ``TaskOutput`` has since been removed from the registry too (task 5332; the
    measurement is cited from the module docstring).  Both remain in the accept
    set on purpose; these cases pin that membership."""

    def test_task_output_after_launch_is_false(self) -> None:
        """TaskOutput collects a backgrounded task's result → reap → False."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([{'type': 'tool_use', 'name': 'TaskOutput', 'input': {'task_id': 't1'}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_task_stop_after_launch_is_false(self) -> None:
        """TaskStop terminates a backgrounded task → reap → False."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([{'type': 'tool_use', 'name': 'TaskStop', 'input': {'task_id': 't1'}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_task_output_before_second_launch_is_true(self) -> None:
        """The reap sits BEFORE a second, unreaped launch → still True (mirrors
        test_second_launch_with_reap_before_it_is_true)."""
        records = [
            *confirmed_bash_launch(command='job-a', tool_id='toolu_a', task_id='aaa11111'),
            assistant_record([{'type': 'tool_use', 'name': 'TaskOutput', 'input': {'task_id': 't1'}}]),
            *confirmed_bash_launch(command='job-b', tool_id='toolu_b', task_id='bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_task_output_in_flat_record_shape_is_false(self) -> None:
        """The reap authored in the flat (nested=False) record shape is still
        recognised (mirrors test_reap_after_launch_across_nesting_styles_is_false)."""
        records = [
            *confirmed_bash_launch(nested=True),
            assistant_record(
                [{'type': 'tool_use', 'name': 'TaskOutput', 'input': {'task_id': 't1'}}],
                nested=False,
            ),
        ]
        assert detect_ended_awaiting_background(records) is False


class TestVerdictSubmissionClearsVerdict:
    """A terminal verdict submission is a reap (task 3639).

    The four ``mcp__verdict-tools__`` entry points share the property that makes
    the inference sound: calling one writes the role's whole deliverable to
    ``verdicts/<role>.json``, so the role's job is finished and nothing can
    still be pending.  Modelled as a REAP at the submission's position (not a
    "was the last action a verdict?" special case), so every ordering gets the
    right answer for free.  Deliberately NOT a loose ``submit_*`` prefix rule:
    ``submit_task`` is called mid-session to file follow-up work while a
    background command is genuinely still running.
    """

    def test_review_verdict_submission_is_false(self) -> None:
        """The `_lane-31` final action: launch → submit_review_verdict → False."""
        records = [
            *confirmed_bash_launch(command='cargo test --all'),
            assistant_record([_verdict_submit('mcp__verdict-tools__submit_review_verdict')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize(
        'tool_name',
        [
            'mcp__verdict-tools__submit_completion_verdict',
            'mcp__verdict-tools__submit_triage',
            'mcp__verdict-tools__submit_merge_disposition',
        ],
    )
    def test_each_terminal_submission_is_false(self, tool_name: str) -> None:
        """All four verdict-tools submissions clear the verdict, not just review."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_verdict_submit(tool_name)]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_bare_unprefixed_spelling_is_false(self) -> None:
        """Matching is on the segment after the last ``__``, so the server prefix
        is not load-bearing (robust to a rename or an unprefixed exposure)."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_verdict_submit('submit_review_verdict')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_activity_after_submission_still_false(self) -> None:
        """A session that submitted its deliverable is not waiting, regardless of
        what harmless work it does afterwards (a literal last-action check would
        wrongly fire here)."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_verdict_submit('mcp__verdict-tools__submit_review_verdict')]),
            assistant_record([{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/etc/hosts'}}]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_after_submission_is_true(self) -> None:
        """The suppression is POSITIONAL, not a global "verdict anywhere" mute:
        a launch after the submission outranks it → True."""
        records = [
            *confirmed_bash_launch(command='job-a', tool_id='toolu_a', task_id='aaa11111'),
            assistant_record([_verdict_submit('mcp__verdict-tools__submit_review_verdict')]),
            *confirmed_bash_launch(command='job-b', tool_id='toolu_b', task_id='bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    @pytest.mark.parametrize(('launch_call', 'confirmation'), _EACH_LAUNCH_KIND)
    def test_submission_beside_the_launch_call_is_false(
        self, launch_call: dict, confirmation: dict
    ) -> None:
        """A parallel batch.  The CLI writes every tool_use of a message before
        any tool_result, so a submission issued beside a launch call sits
        before the launch's confirmation.  The deliverable is written whatever
        is still pending, so the submission outranks that launch → False."""
        records = [
            assistant_record([launch_call]),
            assistant_record([_verdict_submit('mcp__verdict-tools__submit_review_verdict')]),
            confirmation,
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_submit_task_is_not_a_terminal_submission(self) -> None:
        """NEGATIVE control (no over-widening): filing a follow-up task is not a
        terminal verdict — agents call it mid-session while a background command
        is genuinely still running, which is exactly the 2761 abandonment."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_verdict_submit('mcp__fused-memory__submit_task')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_confirm_plan_is_not_a_terminal_submission(self) -> None:
        """NEGATIVE control: confirm_plan is deliberately outside the set — the
        architect may legitimately keep working after it."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([_verdict_submit('mcp__plan-tools__confirm_plan')]),
        ]
        assert detect_ended_awaiting_background(records) is True


class TestMonitorArmIsALaunch:
    """A CONFIRMED Monitor arm is a launch (task 5986).

    The launch is the ``user`` record whose tool_result names an earlier
    ``Monitor`` call and whose structured ``toolUseResult.taskId`` is a usable
    token — not the call, which the CLI may reject.  The Monitor call itself is
    neither a launch nor a reap, and the armed task's id becomes a reap token.
    """

    def test_armed_monitor_then_turn_ends_is_true(self) -> None:
        """Specimens df 3220 / reify 5690: arm, announce standing by, end."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_text('Monitor armed. Standing by for the ratchet test result.')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_flat_arm_result_is_a_launch(self) -> None:
        """The confirming record is found in the flat content shape too."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(nested=False),
            assistant_record([_text('Monitor armed. Standing by for the ratchet test result.')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_monitor_tailing_a_pending_bg_log_is_true(self) -> None:
        """Specimen df 2889: a Monitor whose command watches the pending Bash
        launch's bg-log is a new wait, not a reap of that launch."""
        log = bg_log_path()
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([_monitor_call(command=f'until grep -q "test result" {log}; do sleep 10; done')]),
            _monitor_armed(),
            assistant_record([_text('Monitor armed. Waiting for the test result.')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_rejected_monitor_naming_the_bg_log_does_not_reap_it(self) -> None:
        """Specimen reify 5536: a REJECTED Monitor naming the bg-log armed
        nothing and engaged with nothing, so the Bash launch stays pending."""
        log = bg_log_path()
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([_monitor_call(command=f'tail -f {log}')]),
            _monitor_rejected(),
            assistant_record([_text('Waiting for the sweep to finish.')]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_unrelated_activity_after_arm_is_true(self) -> None:
        """Activity that never references the armed task is not a reap."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_fg_bash('git status')]),
            assistant_record(
                [{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/etc/hosts'}}]
            ),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_rejected_monitor_alone_is_false(self) -> None:
        """Specimen df 2849: a rejected call armed nothing, so nothing is pending."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_rejected(),
            assistant_record([_text('done')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_monitor_call_with_no_result_is_false(self) -> None:
        """A call with no confirming record is not a launch."""
        records = [
            assistant_record([_monitor_call()]),
            assistant_record([_text('done')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_task_id_result_for_a_non_monitor_call_is_not_a_launch(self) -> None:
        """A ``taskId`` result confirms a launch only when it answers a Monitor call."""
        records = [
            assistant_record([{'type': 'tool_use', 'id': 'toolu_some', 'name': 'SomeTool', 'input': {}}]),
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [
                        {'type': 'tool_result', 'tool_use_id': 'toolu_some', 'content': 'ok'},
                    ],
                },
                'toolUseResult': {'taskId': 'bq9imj9co'},
            },
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize('degenerate_id', ['', 'a', 'xyz', 42])
    def test_degenerate_monitor_task_id_is_not_a_launch(self, degenerate_id: object) -> None:
        """An id no notification could ever be matched on is not a launch:
        honoring it would fire on every such session."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(task_id=degenerate_id),
            assistant_record([_text('Monitor armed.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_task_stop_of_armed_monitor_is_false(self) -> None:
        """Stopping the armed Monitor reaps it."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record(
                [{'type': 'tool_use', 'name': 'TaskStop', 'input': {'task_id': MONITOR_TASK_ID}}]
            ),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_reference_to_the_monitor_task_id_is_false(self) -> None:
        """The arm's ``taskId`` is harvested as a token, so reading the armed
        task's output file reaps it."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record(
                [
                    {
                        'type': 'tool_use',
                        'name': 'Read',
                        'input': {'file_path': f'/tmp/claude-1000/s/tasks/{MONITOR_TASK_ID}.output'},
                    }
                ]
            ),
        ]
        assert detect_ended_awaiting_background(records) is False



class TestBashLaunchNeedsConfirmation:
    """A run_in_background Bash is a launch only once its result confirms it
    (task 11122).

    The launch is the ``user`` record whose tool_result answers the call and
    whose structured ``toolUseResult.backgroundTaskId`` is a usable token: the
    rule ``TestMonitorArmIsALaunch`` pins for Monitor.  The call itself is
    classified like any tool_use, so one naming a pending task's token reaps
    it.

    Census of 2026-11-21 over every archived top-level dark-factory and reify
    session (61,768 files).  Of the detector's 280 fires, 140 ended on an
    unconfirmed launch: 131 ran in the foreground (117 of them on a clean
    end_turn), 8 were rejected or errored, and 1 had no result.  545 of the
    13,489 archived run_in_background results ran in the foreground, and no
    result carried exactly one of the two identity sources (the structured id
    and the output-file sentence).
    """

    def test_launch_the_cli_ran_in_the_foreground_is_false(self) -> None:
        """Specimen reify 4996 lane-10: the CLI ran the call to completion, so
        nothing was left pending."""
        records = [
            assistant_record([bash_launch(command='./run-tests.sh')]),
            _ran_in_foreground_result(),
            assistant_record([_text('All 17 tests pass.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_rejected_launch_is_false(self) -> None:
        """A rejected call started nothing."""
        records = [
            assistant_record([bash_launch()]),
            _rejected_result(),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_launch_with_no_result_is_false(self) -> None:
        """The CLI died mid-call, so no result confirms the launch.

        Decided not a launch, mirroring test_monitor_call_with_no_result_is_false.
        Counting it would add 1 fire in 61,768 archived sessions, and that
        session ended stop_reason=tool_use: a CLI killed mid-call leaves no
        success envelope to downgrade.
        """
        records = [
            assistant_record([_text('kick off the long build')]),
            assistant_record([bash_launch(command='./occt-test.sh')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize('degenerate_id', ['', 'a', 'xyz', 42])
    def test_degenerate_background_task_id_is_not_a_launch(self, degenerate_id: object) -> None:
        """An id no later reference could be matched on is not a launch:
        honoring it would fire on every such session."""
        records = [
            assistant_record([bash_launch()]),
            _user_result(
                'toolu_1',
                'Command running in background with ID: x.',
                {'stdout': '', 'backgroundTaskId': degenerate_id},
                is_error=False,
            ),
            assistant_record([_text('Waiting for the build.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_output_file_sentence_without_structured_id_does_not_confirm(self) -> None:
        """Only the structured id confirms: a command's own output can quote
        the CLI's launch sentence."""
        stdout = (
            '... Output is being written to:'
            ' /tmp/claude-1000/x/tasks/qq77other0.output. ...'
        )
        records = [
            assistant_record([bash_launch(command='grep -n "written to" shared/src/shared/cli_invoke.py')]),
            _ran_in_foreground_result(stdout=stdout),
            assistant_record([_text('Found the sentence.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_foreground_run_after_a_pending_launch_is_true(self) -> None:
        """A later call the CLI ran in the foreground leaves the earlier
        confirmed launch pending."""
        records = [
            *confirmed_bash_launch(command='cargo test'),
            assistant_record([bash_launch(command='git status', tool_id='toolu_2')]),
            _ran_in_foreground_result('toolu_2'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_foreground_run_reading_a_pending_launch_reaps_it(self) -> None:
        """A backgrounded call is classified like any tool_use, so one that
        reads a pending task's log engages it."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([bash_launch(command=f'tail -50 {bg_log_path()}', tool_id='toolu_2')]),
            _ran_in_foreground_result('toolu_2'),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_confirmed_launch_naming_a_pending_launch_is_true(self) -> None:
        """A waiter on a pending task is itself a launch, confirmed after the
        call that names the earlier token, so it is the final event."""
        records = [
            *confirmed_bash_launch(),
            assistant_record([
                bash_launch(
                    command=f'until grep -q done {bg_log_path()}; do sleep 5; done',
                    tool_id='toolu_2',
                ),
            ]),
            bg_launch_result('toolu_2', 'bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_background_task_id_answering_a_non_launch_call_is_not_a_launch(self) -> None:
        """A ``backgroundTaskId`` confirms a launch only when it answers a
        run_in_background Bash call."""
        records = [
            assistant_record([{'type': 'tool_use', 'id': 'toolu_some', 'name': 'SomeTool', 'input': {}}]),
            _user_result('toolu_some', 'ok', {'backgroundTaskId': 'bq9imj9co'}, is_error=False),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize(
        ('launch_call', 'other_kinds_confirmation'),
        [
            pytest.param(
                bash_launch(),
                _user_result(
                    'toolu_1',
                    'Command running in background with ID: bq9imj9co.',
                    {'stdout': '', 'taskId': 'bq9imj9co'},
                    is_error=False,
                ),
                id='bash-answered-by-taskId',
            ),
            pytest.param(
                _monitor_call(),
                _user_result(
                    'toolu_mon1',
                    f'Monitor started (task {MONITOR_TASK_ID}, timeout 900000ms).',
                    {'backgroundTaskId': MONITOR_TASK_ID},
                    is_error=False,
                ),
                id='monitor-answered-by-backgroundTaskId',
            ),
        ],
    )
    def test_a_launch_is_confirmed_only_by_its_own_kinds_field(
        self, launch_call: dict, other_kinds_confirmation: dict
    ) -> None:
        """Each launch kind has one confirming field: ``backgroundTaskId`` for
        a run_in_background Bash, ``taskId`` for a Monitor.  The other kind's
        field confirms nothing."""
        records = [
            assistant_record([launch_call]),
            other_kinds_confirmation,
            assistant_record([_text('Waiting for it to finish.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize(('launch_call', 'confirmation'), _EACH_LAUNCH_KIND)
    def test_reap_beside_the_launch_call_does_not_outrank_it(
        self, launch_call: dict, confirmation: dict
    ) -> None:
        """A launch is ordered at its confirmation.  A reap issued beside its
        call, in the same parallel batch, sits before that confirmation and
        cannot engage a task not yet started, so the launch stays the final
        event → True.  Contrast
        TestVerdictSubmissionClearsVerdict::test_submission_beside_the_launch_call_is_false."""
        records = [
            *confirmed_bash_launch(command='job-a', tool_id='toolu_a', task_id='aaa11111'),
            assistant_record([launch_call]),
            assistant_record(
                [{'type': 'tool_use', 'name': 'TaskStop', 'input': {'task_id': 'aaa11111'}}]
            ),
            confirmation,
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_auto_backgrounded_foreground_bash_is_false_documented_gap(self) -> None:
        """DOCUMENTED, ACCEPTED false negative.  Specimen dark-factory 3799: the
        CLI moved a foreground Bash to the background after its 120s timeout,
        and the session ended waiting on it.  That result answers no launch
        call, so it is not a launch → False.

        Counting it was measured and rejected (task 11122, 2026-11-21).  Over
        61,768 archived sessions it would have added 86 verdicts, 56 on a
        clean end_turn.  36 of the 56 carried the structured auto-background
        id, and only 9 of those were waiting on work they still needed.  27
        had finished, leaving a redundant `find /` or `grep -r` running.  The
        other 20 were foreground output quoting the CLI's launch sentence.  A
        downgrade that is right a quarter of the time is refused.  Sibling of
        test_earlier_completion_after_later_launch_is_false_documented_gap.
        """
        records = [
            assistant_record([_fg_bash('cargo test --all', tool_id='toolu_fg')]),
            _auto_backgrounded_result(),
            assistant_record([_text('Waiting for the test run to finish.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

_EXPIRY = '[Monitor expired after 10m with 2 events delivered. Re-arm it if you still need the watch.]'


class TestDeliveredNotificationIsAReap:
    """A task-notification DELIVERED to the model whose text names a harvested
    token is a reap, for every launch kind (task 5986).

    Delivery takes one of two shapes: a mid-turn ``attachment`` record
    (``queued_command`` with ``commandMode`` ``task-notification``), or a
    ``user`` record whose ``origin.kind`` is ``task-notification`` — the shape
    a headless session that outlived its turn uses.  ``queue-operation``
    records are queue bookkeeping, not delivery.
    """

    def test_monitor_event_delivered_mid_turn_then_more_work_is_false(self) -> None:
        """Specimen df 2991: the Monitor's event reached the model mid-turn."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_fg_bash('git status')]),
            _delivered_mid_turn(_monitor_event_text()),
            assistant_record([_fg_bash('git diff --stat')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_monitor_expiry_delivered_as_new_turn_is_false(self) -> None:
        """Specimen reify 7838: the CLI stayed alive after the turn ended and
        delivered the Monitor's expiry as a new turn."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_text('Standing by for the eval result.')]),
            _delivered_new_turn(_monitor_event_text(event=_EXPIRY)),
            assistant_record([_text('The watch expired after both results arrived. Done.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_bash_completion_delivered_mid_turn_is_false(self) -> None:
        """The dominant false-downgrade shape of a Bash-only reap rule: the
        completion reached the model and the session moved on."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            _delivered_mid_turn(_bash_completion_text()),
            assistant_record(
                [{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/etc/hosts'}}]
            ),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_bash_completion_delivered_as_new_turn_is_false(self) -> None:
        """The session ended its turn to wait, and the CLI delivered the
        completion as a new turn rather than exiting."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            assistant_record([_text("I'll wait for the notification.")]),
            _delivered_new_turn(_bash_completion_text()),
            assistant_record([_text('All tests passed.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize(
        'launch_result',
        [
            bg_launch_result(in_text=False),
            bg_launch_result(task_id='qq77other', log_path=bg_log_path()),
        ],
        ids=['task-id-token', 'log-path-token'],
    )
    def test_completion_matches_on_either_token(self, launch_result: dict) -> None:
        """The completion names both the task id and the output file, so either
        harvested token identifies it."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            launch_result,
            _delivered_mid_turn(_bash_completion_text()),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_text_sub_block_content_is_read(self) -> None:
        """A new-turn delivery whose content is a list of text sub-blocks."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_text('Standing by for the eval result.')]),
            _delivered_new_turn(_monitor_event_text(event=_EXPIRY), as_blocks=True),
            assistant_record([_text('Done.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_streamed_monitor_event_then_end_is_false_documented_gap(self) -> None:
        """DOCUMENTED, ACCEPTED false negative.  Specimen reify 5623: a
        progress-streaming Monitor delivered a mid-stream event and the session
        ended awaiting a later one → the delivery reaps → False.

        No structured flag tells a terminal event from a mid-stream one; only
        the notification text does, and parsing it is the fragility the
        detector refuses.  The Monitor analogue of
        test_poll_once_still_running_then_end_is_false_documented_gap.
        """
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            _delivered_mid_turn(
                _monitor_event_text(event='progress: 74 suites finished, 0 failures/errors so far')
            ),
            assistant_record([_text('Suite still running clean. Waiting for completion.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize(
        'later_launch',
        [
            [assistant_record([_monitor_call()]), _monitor_armed()],
            [
                assistant_record([bash_launch(command='job-b', tool_id='toolu_b')]),
                bg_launch_result(tool_id='toolu_b', task_id='bbb22222'),
            ],
        ],
        ids=['later-monitor-arm', 'later-bash-launch'],
    )
    def test_earlier_completion_after_later_launch_is_false_documented_gap(
        self, later_launch: list
    ) -> None:
        """DOCUMENTED, ACCEPTED false negative.  Reaps are positional, not per
        launch: Bash A launches, a later launch L follows, A's completion is
        delivered, and the turn ends with L still pending → the delivery is
        the final event → False.

        A notification arrives asynchronously, so it can land after a launch
        the agent made later, unlike a tool_use reap the agent issues itself.
        Per-launch tracking was measured and rejected (task 7127,
        2026-11-21).  Over the dark-factory and reify transcript archives it
        would have added 24 verdicts on sessions that ended a clean turn,
        beside the positional rule's 191.  Only 10 of the 24 were waiting on
        work they still needed.  11 had finished, and the pending launch was
        a redundant waiter whose target had already reported.  2 had
        finished with a non-essential check still running.  1 ended on an
        API error.  A downgrade that is wrong about half the time is the
        direction this module refuses.  Sibling of
        test_streamed_monitor_event_then_end_is_false_documented_gap.
        """
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            *later_launch,
            _delivered_mid_turn(_bash_completion_text()),
            assistant_record([_text('Tests passed. Standing by for the second result.')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_notification_for_an_unknown_task_is_not_a_reap(self) -> None:
        """A delivery that names no harvested token engages with nothing pending."""
        records = [
            assistant_record([bash_launch(command='cargo test --all')]),
            bg_launch_result(),
            _delivered_mid_turn(_monitor_event_text(task_id='zz99other')),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_enqueued_but_undelivered_is_not_a_reap(self) -> None:
        """An event enqueued after the final turn, by a process that then
        exited, never reached the agent."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_text('Monitor armed. Standing by.')]),
            _enqueued_only(_monitor_event_text()),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_launch_after_a_delivered_notification_is_true(self) -> None:
        """The reap is positional: a later launch outranks it."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            _delivered_mid_turn(_monitor_event_text()),
            assistant_record([bash_launch(command='job-b', tool_id='toolu_b')]),
            bg_launch_result(tool_id='toolu_b', task_id='bbb22222'),
        ]
        assert detect_ended_awaiting_background(records) is True

    @pytest.mark.parametrize(
        'record',
        [
            {'type': 'attachment', 'attachment': 'junk'},
            {
                'type': 'attachment',
                'attachment': {
                    'type': 'queued_command', 'commandMode': 'task-notification', 'prompt': None,
                },
            },
            {
                'type': 'attachment',
                'attachment': {
                    'type': 'queued_command', 'commandMode': 'task-notification', 'prompt': 42,
                },
            },
            {
                'type': 'attachment',
                'attachment': {
                    'type': 'queued_command', 'commandMode': 'prompt',
                    'prompt': _monitor_event_text(),
                },
            },
            {
                'type': 'user',
                'origin': 'task-notification',
                'message': {'role': 'user', 'content': _monitor_event_text()},
            },
            {
                'type': 'user',
                'origin': {'kind': 'other'},
                'message': {'role': 'user', 'content': _monitor_event_text()},
            },
            {
                'type': 'user',
                'origin': {'kind': 'task-notification'},
                'message': {'role': 'user', 'content': None},
            },
            {
                'type': 'user',
                'origin': {'kind': 'task-notification'},
                'message': {'role': 'user', 'content': 42},
            },
        ],
        ids=[
            'attachment-not-a-dict',
            'prompt-none',
            'prompt-int',
            'command-mode-prompt',
            'origin-not-a-dict',
            'origin-kind-other',
            'content-none',
            'content-int',
        ],
    )
    def test_malformed_notification_records_never_raise(self, record: dict) -> None:
        """Records that are not a well-formed delivery reap nothing and never raise."""
        records = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            record,
        ]
        assert detect_ended_awaiting_background(records) is True


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
                assistant_record([_text('start the long build')]),
                *confirmed_bash_launch(command='./occt-test.sh'),
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
                *confirmed_bash_launch(),
                assistant_record([_bash_output()]),
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
        records: list = [assistant_record([_text('run the crate test suites in parallel')])]
        for n in range(4):
            tool_id = f'toolu_{n}'
            task_id = f'bg{n}0000{n}'
            records.append(
                assistant_record([bash_launch(command=f'cargo test -p crate{n}', tool_id=tool_id)])
            )
            records.append(bg_launch_result(tool_id=tool_id, task_id=task_id))
            records.append(assistant_record([_fg_bash(f'tail -50 {bg_log_path(task_id)}')]))
        records.append(
            assistant_record(
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
            assistant_record([bash_launch(command='cargo test -p crate0', tool_id='toolu_0')]),
            bg_launch_result(tool_id='toolu_0', task_id='bg000000'),
            assistant_record([_fg_bash(f'tail -50 {bg_log_path("bg000000")}')]),
            assistant_record([bash_launch(command='cargo test -p crate1', tool_id='toolu_1')]),
            bg_launch_result(tool_id='toolu_1', task_id='bg111111'),
            assistant_record([_text('I will wait for the background test to finish')]),
        ]
        _write_transcript(tmp_path, sid, records)
        assert ended_awaiting_background_for_session(tmp_path, sid) is True

    def test_monitor_armed_transcript_is_true(self, tmp_path: Path) -> None:
        """Specimen df 3220: the session arms a Monitor and ends its turn
        standing by, so the process exits before the result arrives."""
        sid = 'sess-monitor-armed'
        records: list = [
            assistant_record([_text('arm a watch on the ratchet test')]),
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_text('Monitor armed. Standing by for the ratchet test result.')]),
        ]
        _write_transcript(tmp_path, sid, records)
        assert ended_awaiting_background_for_session(tmp_path, sid) is True

    def test_reify7838_shaped_transcript_is_false(self, tmp_path: Path) -> None:
        """Specimen reify 7838: both Monitor events were delivered mid-turn, the
        turn ended, and the CLI stayed alive to deliver the expiry as a new
        turn.  The session engaged with everything it armed."""
        sid = 'sess-reify7838'
        expiry = _monitor_event_text(event=_EXPIRY)
        records: list = [
            assistant_record([_monitor_call()]),
            _monitor_armed(),
            assistant_record([_fg_bash('cargo run --bin eval')]),
            _delivered_mid_turn(_monitor_event_text(event='eval rc=0')),
            assistant_record([_fg_bash('git commit -am "eval green"')]),
            _delivered_mid_turn(_monitor_event_text(event='stdlib rc=0\n[exited with code 0]')),
            assistant_record([_text('Both suites are green and committed.')]),
            _enqueued_only(expiry),
            _delivered_new_turn(expiry),
            assistant_record([_text('The watch expired after delivering both results.')]),
        ]
        _write_transcript(tmp_path, sid, records)
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
    it defaults False on both AgentResult and SubprocessResult."""

    def test_agent_result_defaults_false(self) -> None:
        """AgentResult.ended_awaiting_background defaults to False."""
        result = AgentResult(success=True, output='ok')
        assert result.ended_awaiting_background is False

    def test_subprocess_result_defaults_false(self) -> None:
        """SubprocessResult.ended_awaiting_background defaults to False."""
        result = SubprocessResult(stdout='', stderr='', returncode=0, duration_ms=0)
        assert result.ended_awaiting_background is False


class TestParseClaudeOutputDowngrade:
    """parse_claude_output downgrades success→failure when the subprocess was
    flagged ended_awaiting_background, and propagates the flag on every path."""

    def test_success_envelope_flagged_is_downgraded(self) -> None:
        """subtype=success + ended_awaiting_background=True → success False, flag True."""
        sub = SubprocessResult(
            stdout=_SUCCESS_ENVELOPE,
            stderr='',
            returncode=0,
            duration_ms=681_000,
            ended_awaiting_background=True,
        )
        agent = parse_claude_output(sub)
        assert agent.success is False
        assert agent.ended_awaiting_background is True

    def test_success_envelope_unflagged_stays_success(self) -> None:
        """subtype=success + ended_awaiting_background=False → success stays True, flag False."""
        sub = SubprocessResult(
            stdout=_SUCCESS_ENVELOPE,
            stderr='',
            returncode=0,
            duration_ms=681_000,
            ended_awaiting_background=False,
        )
        agent = parse_claude_output(sub)
        assert agent.success is True
        assert agent.ended_awaiting_background is False

    def test_flag_propagated_on_empty_stdout_without_flipping_success(self) -> None:
        """An already-failing empty-stdout result + flag=True → success stays
        False and the flag is still propagated (downgrade is an idempotent no-op
        on an already-failing result; no crash)."""
        sub = SubprocessResult(
            stdout='',
            stderr='boom',
            returncode=1,
            duration_ms=100,
            ended_awaiting_background=True,
        )
        agent = parse_claude_output(sub)
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


class TestDetectorTotalityAndTokenHygiene:
    """Amendment pass (reviewer_comprehensive, task 3639).

    Three properties the detector's own docstring promises but nothing pinned:
    a malformed BLOCK never raises (the docstring's "malformed
    records/blocks ... are skipped, never raise" previously covered only
    records), a degenerate identity token cannot mute the whole verdict, and
    the token scan sees the same strings the old whole-input ``json.dumps``
    exposed.  Task 5986 adds a fourth: that scan reaches the bottom of an
    input of any depth.
    """

    def test_non_str_tool_use_name_never_raises(self) -> None:
        """A tool_use block whose ``name`` is unhashable is skipped, not fatal.

        Pre-amendment this raised ``TypeError: unhashable type`` at ``name in
        _BACKGROUND_REAP_TOOLS``.  The call site in ``run_cli_subprocess`` is
        unguarded, so the exception would fail the whole agent invocation —
        a malformed block breaking a completed run.
        """
        records = [
            *confirmed_bash_launch(),
            assistant_record([{'type': 'tool_use', 'name': ['weird'], 'input': {}}]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_non_str_name_block_still_reaps_by_token(self) -> None:
        """Name normalization does not cost the block its token-based reap."""
        records = [
            assistant_record([bash_launch()]),
            bg_launch_result(),
            assistant_record([
                {'type': 'tool_use', 'name': None, 'input': {'file_path': bg_log_path()}},
            ]),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize('degenerate_id', ['a', 'b1', 'xyz'])
    def test_degenerate_task_id_is_not_a_usable_token(self, degenerate_id: str) -> None:
        """A too-short ``backgroundTaskId`` is dropped rather than matched.

        A 1-2 char token is a substring of essentially every serialized tool
        input, so honoring it would mark the next tool_use of ANY kind as a
        reap and silently mute the abandonment verdict.  Here the following
        ``Write`` has nothing to do with the launch, so the run IS abandoned.
        """
        records = [
            *confirmed_bash_launch(),
            {
                'type': 'user',
                'toolUseResult': {'stdout': '', 'backgroundTaskId': degenerate_id},
                'message': {'role': 'user', 'content': []},
            },
            assistant_record([
                {'type': 'tool_use', 'name': 'Write',
                 'input': {'file_path': '/tmp/notes.md', 'content': 'a b1 xyz'}},
            ]),
        ]
        assert detect_ended_awaiting_background(records) is True

    def test_real_length_task_id_is_still_a_usable_token(self) -> None:
        """The length floor is inert for a real id (9 chars, e.g. ``b1ucu6z5t``)."""
        records = [
            assistant_record([bash_launch()]),
            bg_launch_result(task_id='b1ucu6z5t', in_text=False),
            assistant_record([_fg_bash('tail -50 /tmp/x/tasks/b1ucu6z5t.output')]),
        ]
        assert detect_ended_awaiting_background(records) is False

    def test_token_found_in_nested_non_str_input(self) -> None:
        """The token scan walks lists/nested dicts, as ``json.dumps`` did.

        Parity guard for replacing whole-input serialization with a walk over
        the input's individual ``str`` values: a token buried under a list and
        a nested dict must still count as a reap.
        """
        records = [
            assistant_record([bash_launch()]),
            bg_launch_result(),
            assistant_record([{
                'type': 'tool_use',
                'name': 'SomeTool',
                'input': {
                    'n': 3,
                    'flag': True,
                    'nested': {'paths': [None, {'p': bg_log_path()}]},
                },
            }]),
        ]
        assert detect_ended_awaiting_background(records) is False

    @pytest.mark.parametrize(
        ('innermost', 'expected'),
        [(bg_log_path(), False), ('/etc/hosts', True)],
        ids=['token-at-the-bottom-reaps', 'no-token-still-fires'],
    )
    def test_input_nested_past_the_recursion_limit_is_scanned_whole(
        self, innermost: str, expected: bool
    ) -> None:
        """``json.loads`` decodes a transcript line nested far deeper than the
        interpreter's recursion limit, so the token scan must walk such an
        input to the bottom rather than raise or give up partway."""
        depth = sys.getrecursionlimit() + 500
        deep_input = json.loads('[' * depth + json.dumps(innermost) + ']' * depth)
        records = [
            assistant_record([bash_launch()]),
            bg_launch_result(),
            assistant_record([{'type': 'tool_use', 'name': 'SomeTool', 'input': {'deep': deep_input}}]),
        ]
        assert detect_ended_awaiting_background(records) is expected

    def test_input_with_no_strings_is_not_a_reap(self) -> None:
        """An input carrying no ``str`` anywhere cannot match a token."""
        records = [
            assistant_record([bash_launch()]),
            bg_launch_result(),
            assistant_record([
                {'type': 'tool_use', 'name': 'SomeTool', 'input': {'n': 1, 'ok': False}},
            ]),
        ]
        assert detect_ended_awaiting_background(records) is True


class TestContentBlockNestingSeam:
    """``_content_blocks`` is the SINGLE expression of the transcript
    content-nesting tolerance rule (amendment, task 3639) — previously
    duplicated verbatim between ``_iter_result_texts`` and the detector, so a
    CLI nesting change had to be fixed in two places and only one was covered
    by the ``nested=True/False`` parametrization.
    """

    @pytest.mark.parametrize('record', [
        None,
        'not-a-dict',
        {},
        {'message': 'not-a-dict'},
        {'message': {'content': 'not-a-list'}},
        {'content': 'not-a-list'},
        {'message': {}, 'content': None},
    ])
    def test_malformed_records_yield_no_blocks(self, record: object) -> None:
        assert _content_blocks(record) == []

    def test_both_nestings_resolve(self) -> None:
        blocks = [{'type': 'tool_use', 'name': 'Bash'}]
        assert _content_blocks({'message': {'content': blocks}}) is blocks
        assert _content_blocks({'content': blocks}) is blocks

    def test_message_nesting_wins_over_flat(self) -> None:
        """Precedence matches the real CLI shape (message first)."""
        nested = [{'type': 'tool_use', 'name': 'A'}]
        flat = [{'type': 'tool_use', 'name': 'B'}]
        assert _content_blocks({'message': {'content': nested}, 'content': flat}) is nested

    def test_result_text_extraction_shares_the_seam(self) -> None:
        """``_iter_result_texts`` reads both nestings through the same helper."""
        block = {'type': 'tool_result', 'content': 'hello'}
        assert list(_iter_result_texts({'message': {'content': [block]}})) == ['hello']
        assert list(_iter_result_texts({'content': [block]})) == ['hello']
        assert list(_iter_result_texts({'content': 'not-a-list'})) == []
