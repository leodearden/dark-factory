"""The follow-up holder marker and its predicate (plans/follow-up-filing-shape-prd.md §4.3, decision 10).

A holder is the task whose metadata carries ``followup_holder``. The predicate reads a task row as
the store hands it out: metadata as a dict or as its JSON string, possibly absent or unreadable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from fused_memory.middleware.followup_hold import FOLLOWUP_HOLDER_METADATA_KEY, is_holder

_MARKER = {'project_id': 'dark_factory', 'minted_at': '2026-11-08T00:00:00+00:00'}


def test_the_marker_key_is_the_contract_fixed_name() -> None:
    assert FOLLOWUP_HOLDER_METADATA_KEY == 'followup_holder'


@pytest.mark.parametrize('metadata', [
    {FOLLOWUP_HOLDER_METADATA_KEY: _MARKER, 'files': ['a.py']},
    json.dumps({FOLLOWUP_HOLDER_METADATA_KEY: _MARKER}),
], ids=['dict', 'json-string'])
def test_a_row_carrying_the_marker_is_a_holder(metadata: Any) -> None:
    assert is_holder({'id': 7, 'status': 'pending', 'metadata': metadata})


@pytest.mark.parametrize('task', [
    {'id': 7, 'metadata': {'files': ['a.py']}},
    {'id': 7, 'metadata': {FOLLOWUP_HOLDER_METADATA_KEY: None}},
    {'id': 7, 'metadata': json.dumps({FOLLOWUP_HOLDER_METADATA_KEY: None})},
    {'id': 7},
    {'id': 7, 'metadata': ''},
    {'id': 7, 'metadata': None},
    {'id': 7, 'metadata': json.dumps([FOLLOWUP_HOLDER_METADATA_KEY])},
    {'id': 7, 'metadata': 'followup_holder: yes'},
], ids=[
    'key-absent', 'value-none', 'json-value-none', 'metadata-absent', 'metadata-empty-string',
    'metadata-none', 'non-object-json', 'garbage-string',
])
def test_a_row_without_a_readable_marker_is_not_a_holder(task: dict[str, Any]) -> None:
    assert not is_holder(task)


_PROBE = (
    'import json, sys\n'
    'import fused_memory.middleware.followup_hold\n'
    'print(json.dumps([name for name in ('
    "'fused_memory.middleware.task_interceptor', 'fused_memory.middleware.task_curator'"
    ') if name in sys.modules]))\n'
)


def test_importing_followup_hold_loads_neither_the_interceptor_nor_the_curator() -> None:
    """curator_types imports this module, so it must stay below both drivers.

    ``-I`` drops PYTHONPATH, the user site and the cwd, so this measures the module's own import
    closure rather than whatever the pytest process loaded.
    """
    result = subprocess.run(
        [sys.executable, '-I', '-c', _PROBE], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [], (
        f'importing followup_hold loaded {result.stdout.strip()}; it must import only the stdlib '
        'and lower layers.'
    )
