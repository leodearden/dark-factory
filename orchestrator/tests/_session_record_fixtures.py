"""SessionRecord builders shared by the session-record and session-registry test files."""

from __future__ import annotations

from orchestrator import session_records


def make_record(**overrides: object) -> session_records.SessionRecord:
    """Build a fully-populated SessionRecord for round-trip/identity tests.

    Every field is given a concrete, distinguishable value so a round-trip
    test can catch a field being dropped/mis-typed; ``overrides`` lets a
    test tweak just the field(s) it cares about. Includes the C1 schema
    extensions (parent_session_id/spawn_mode/display/question) and the
    task-4193 hook-owner binding (claude_session_id, claude_owner_pid)
    alongside the original rail fields.
    """
    # Declared as a bare `dict` (not `dict[str, object]`) so pyright treats it
    # as dict[Unknown, Unknown] at the **fields unpack below -- mirrors
    # test_merge_types_invariants.py's _real_kwargs/_decided_kwargs/_entry_kwargs,
    # the established idiom in this suite for a kwargs-builder helper whose
    # result is unpacked into a strictly-typed constructor.
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': session_records.Status.LAUNCHING,
        'title': 'unblock:df#2085 routing-mechanism',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'escalation_id': 'esc-1',
        'prompt': '/unblock 2085',
        'cwd': '/home/leo/src/dark-factory',
        'launcher_pid': 4242,
        'start_ts': '2026-07-07T00:00:00+00:00',
        'exit_code': None,
        'result_file': None,
        'transcript_path': '~/.claude/projects/-home-leo-src-dark-factory',
        'parent_session_id': 'root-df-1-1',
        'spawn_mode': 'sibling',
        'display': session_records.Display(
            kind='wm', wm_title='unblock:df#2085 slug', wm_window_id='0x1a', tmux_target=None
        ),
        'question': session_records.Question(text='approve rollout?', asked_at='2026-07-07T00:00:00+00:00'),
        'claude_session_id': 'uuid-claude-abc123',
        'claude_launch_session_id': 'uuid-claude-launch-0',
        'claude_owner_pid': 424242,
        'self_exit': True,
        't3_end': session_records.T3End.SETTLE,
        'exit_request': session_records.ExitRequest(
            state=session_records.ExitRequestState.ATTEMPTED,
            reason=None,
            pid=424242,
            at='2026-07-07T00:00:00+00:00',
            marker_session_id='uuid-claude-abc123',
        ),
        'exit_requested_at': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return session_records.SessionRecord(**fields)
