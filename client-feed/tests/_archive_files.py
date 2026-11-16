"""Fixture verify-archive files under a root's data/verify-logs/<task id>/, in the orchestrator's filename grammar."""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

TASK_ID = 7
SECRET_CMD = 'uv run pytest /home/someone/private/tests'
SECRET_HINT = 'AssertionError: secret customer value 0.393'
SECRET_MESSAGE = 'assert 0.2 > 0.393'
SECRET_TEXT = 'SECRET-TRACEBACK-TEXT /home/someone/private/test_a.py:12'


def task_dir(root: Path, task_id: int = TASK_ID) -> Path:
    path = root / 'data' / 'verify-logs' / str(task_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stamp(when: datetime) -> str:
    return when.strftime('%Y%m%dT%H%M%S_%fZ')


def qualified(qualifiers: str) -> str:
    return f'.{qualifiers}' if qualifiers else ''


def write_plan(root: Path, when: datetime, *, attempt: int = 1, task_id: int = TASK_ID) -> Path:
    path = task_dir(root, task_id) / f'attempt-{attempt}.plan-{stamp(when)}.json'
    path.write_text(json.dumps({'legs': [{'label': 'test', 'reason': 'full suite'}]}))
    return path


def summary_payload(category: str, commands: Sequence[tuple[str, int]]) -> dict:
    """The orchestrator's verify.py::_build_summary_payload shape."""
    runs = [
        {
            'label': label,
            'cmd': SECRET_CMD,
            'rc': rc,
            'timed_out': False,
            'started_at': '2026-10-01T10:00:00+00:00',
            'duration_secs': 1.5,
            'segments': None,
            'load': None,
            'slot_wait_secs': None,
        }
        for label, rc in commands
    ]
    worst = max(runs, key=lambda run: run['rc'])
    return {
        'role': 'task',
        'category': category,
        'cause_hint': SECRET_HINT,
        'label': worst['label'],
        'rc': worst['rc'],
        'timed_out': False,
        'cmd': SECRET_CMD,
        'started_at': '2026-10-01T10:00:00+00:00',
        'duration_secs': 4.5,
        'commands': runs,
    }


def write_summary(
    root: Path,
    when: datetime,
    *,
    qualifiers: str = 'alpha',
    category: str = 'test_failure',
    commands: Sequence[tuple[str, int]] = (('test', 1), ('lint', 0), ('type', 0)),
    attempt: int = 1,
    infix: str = 'summary',
    task_id: int = TASK_ID,
) -> Path:
    name = f'attempt-{attempt}{qualified(qualifiers)}.{infix}-{stamp(when)}.json'
    path = task_dir(root, task_id) / name
    path.write_text(json.dumps(summary_payload(category, commands)))
    return path


def junit_xml(*testcases: str, suite_time: str | None = '1.5', wrapped: bool = True) -> bytes:
    time_attribute = '' if suite_time is None else f' time="{suite_time}"'
    suite = f'<testsuite name="pytest"{time_attribute}>{"".join(testcases)}</testsuite>'
    document = f'<testsuites name="pytest tests">{suite}</testsuites>' if wrapped else suite
    return f'<?xml version="1.0" encoding="utf-8"?>{document}'.encode()


def passed(name: str, classname: str = 'tests.test_a') -> str:
    return f'<testcase classname="{classname}" name="{name}" time="0.01" />'


def red(name: str, tag: str, type_attribute: str | None = None) -> str:
    type_part = '' if type_attribute is None else f' type="{type_attribute}"'
    return (
        f'<testcase classname="tests.test_a" name="{name}" time="0.5">'
        f'<{tag} message="{SECRET_MESSAGE}"{type_part}>{SECRET_TEXT}</{tag}>'
        '</testcase>'
    )


def write_junit(
    root: Path,
    when: datetime,
    content: bytes,
    *,
    qualifiers: str = 'alpha',
    attempt: int = 1,
    task_id: int = TASK_ID,
) -> Path:
    name = f'attempt-{attempt}{qualified(qualifiers)}.junit-{stamp(when)}.xml.gz'
    path = task_dir(root, task_id) / name
    path.write_bytes(gzip.compress(content))
    return path
