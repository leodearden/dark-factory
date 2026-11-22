"""Transcript records for the ``ended_awaiting_background`` detector's suites.

One copy of the confirmed-background-launch shape (a ``run_in_background``
Bash call plus the ``user`` record in which the CLI reports having started it)
for every suite that builds a transcript ending on one, so a change to the
launch contract updates them all at once. Consumers import by bare module name
-- ``conftest.py`` prepends ``shared/tests`` to ``sys.path`` (same convention
as ``_measured_cli_results.py``). The leading underscore keeps pytest from
collecting this module as a test file.

Records mirror the on-disk transcript JSONL shape: an assistant record carries
its content blocks nested under ``record['message']['content']`` (the real CLI
shape) or flat under ``record['content']``.
"""

from __future__ import annotations


def assistant_record(blocks: list, *, nested: bool = True, model: str | None = None) -> dict:
    """An assistant transcript record wrapping *blocks*, served by *model* if given.

    ``nested=True`` → record['message']['content'] (real CLI shape).
    ``nested=False`` → flat record['content'].
    """
    fields: dict = {'content': blocks}
    if model is not None:
        fields['model'] = model
    if nested:
        return {'type': 'assistant', 'message': {'role': 'assistant', **fields}}
    return {'type': 'assistant', **fields}


def bash_launch(
    *,
    background: bool = True,
    command: str = 'sleep 9999',
    tool_id: str = 'toolu_1',
) -> dict:
    """A Bash tool_use block asking for ``run_in_background`` (unless
    ``background=False``), carrying the ``id`` its tool_result answers."""
    return {
        'type': 'tool_use',
        'id': tool_id,
        'name': 'Bash',
        'input': {'command': command, 'run_in_background': background},
    }


def bg_log_path(task_id: str = 'bz89cdzmh') -> str:
    """The CLI's background-output path shape: the task id is a substring of it."""
    return f'/tmp/claude-1000/-w-3639/sess/tasks/{task_id}.output'


def bg_launch_result(
    tool_id: str = 'toolu_1',
    task_id: str = 'bz89cdzmh',
    log_path: str | None = None,
    *,
    in_text: bool = True,
) -> dict:
    """The ``user`` tool_result record the CLI emits for a background launch.

    Mirrors the REAL shape sampled from ``~/.claude/projects/*/*.jsonl``: the
    record carries BOTH a structured ``toolUseResult.backgroundTaskId`` and a
    result-text sentence naming the output file.  ``in_text=False`` drops the
    sentence, and a *log_path* not containing *task_id* separates the two
    tokens, so the detector's two token-extraction sources can be exercised
    independently.
    """
    log_path = log_path if log_path is not None else bg_log_path(task_id)
    text = f'Command running in background with ID: {task_id}.'
    if in_text:
        text += (
            f' Output is being written to: {log_path}.'
            ' You will be notified when it completes.'
            ' To check interim output, use Read on that file path.'
        )
    return {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [
                {'type': 'tool_result', 'tool_use_id': tool_id, 'content': text},
            ],
        },
        'toolUseResult': {'stdout': '', 'backgroundTaskId': task_id},
    }


def confirmed_bash_launch(
    *,
    command: str = 'sleep 9999',
    tool_id: str = 'toolu_1',
    task_id: str = 'bz89cdzmh',
    nested: bool = True,
) -> list[dict]:
    """A CONFIRMED run_in_background launch: the call and the result in which
    the CLI reports having started it."""
    return [
        assistant_record([bash_launch(command=command, tool_id=tool_id)], nested=nested),
        bg_launch_result(tool_id=tool_id, task_id=task_id),
    ]
