"""The L2 skill's ``resolve_issue`` examples close a level-2 record as written.

INV-10 tier 1 (docs/legibility/design-invariants.md): every fenced
``mcp__escalation__resolve_issue(...)`` example in
skills/escalation-watcher/SKILL.md is parsed and EXECUTED over real HTTP
against a pending level-2 record. Only ``escalation_id`` is substituted, and
the call is header-less, the interactive L2 channel. An example that omits
``resolution_class`` or ``ruled_by_human`` teaches a close the server refuses
or misattributes (plans/instruments-honesty-prd.md decisions 8 and 9).
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _escalation_http import escalation_http_call
from _escalation_seed import seed_escalation

from escalation.classify import resolver_tier
from escalation.models import RESOLUTION_CLASS_REQUIRED_CODE, RESOLUTION_CLASSES
from escalation.queue import EscalationQueue

_SKILL = Path(__file__).resolve().parents[2] / 'skills' / 'escalation-watcher' / 'SKILL.md'
_CALL_NAME = 'mcp__escalation__resolve_issue'
_CALL_TOKEN = f'{_CALL_NAME}('
_HEADING = re.compile(r'^#+\s+(.*)$')
_TASK = '8562'


@dataclass(frozen=True)
class _Example:
    heading: str
    line: int
    source: str

    def label(self) -> str:
        return f'{self.heading} (SKILL.md line {self.line})'

    def test_id(self) -> str:
        return re.sub(r'[^a-z0-9_]+', '-', self.heading.lower()).strip('-')


def _fenced_blocks(text: str) -> list[tuple[str, int, list[str]]]:
    """(nearest heading, first body line number, body lines) of every fenced block, indented or not."""
    blocks: list[tuple[str, int, list[str]]] = []
    heading, fence, body, start = '', '', [], 0
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if fence:
            if stripped == fence:
                blocks.append((heading, start, body))
                fence = ''
            else:
                body.append(line)
        elif stripped.startswith('```'):
            fence, body, start = stripped[:len(stripped) - len(stripped.lstrip('`'))], [], number + 1
        elif match := _HEADING.match(line):
            heading = match.group(1).strip()
    assert not fence, f'{_SKILL.name}: a fence opened before line {start} never closes'
    return blocks


def _examples() -> list[_Example]:
    examples = []
    for heading, start, body in _fenced_blocks(_SKILL.read_text(encoding='utf-8')):
        first = next((line.strip() for line in body if line.strip()), '')
        if first.startswith(_CALL_TOKEN):
            examples.append(_Example(heading, start, textwrap.dedent('\n'.join(body))))
    return examples


_EXAMPLES = _examples()


def _kwargs(example: _Example) -> dict[str, Any]:
    """The example's literal keyword arguments, refusing any shape that is not one plain call."""
    where = example.label()
    try:
        tree = ast.parse(example.source, mode='eval')
    except SyntaxError as exc:
        raise AssertionError(f'{where}: not a Python call expression: {exc}') from exc
    call = tree.body
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == _CALL_NAME, (
        f'{where}: the block is not a single {_CALL_NAME}(...) call'
    )
    assert not call.args, f'{where}: positional arguments; every argument must be a keyword'
    kwargs: dict[str, Any] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None, f'{where}: a **mapping splat; every argument must be a literal keyword'
        ellipses = [n for n in ast.walk(keyword.value) if isinstance(n, ast.Constant) and n.value is Ellipsis]
        assert not ellipses, f'{where}: {keyword.arg}= contains a bare ... (an Ellipsis, not a placeholder string)'
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError as exc:
            raise AssertionError(f'{where}: {keyword.arg}= is not a literal: {exc}') from exc
    return kwargs


@pytest.fixture
def served(tmp_path: Path, serve_escalation_mcp) -> tuple[str, EscalationQueue]:
    base_url, _port, queue = serve_escalation_mcp(tmp_path / 'escalations')
    return base_url, queue


async def _run(served: tuple[str, EscalationQueue], kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    base_url, queue = served
    l2 = seed_escalation(queue, level=2, task_id=_TASK)
    result = await escalation_http_call(
        base_url, 'resolve_issue', identity=None, **{**kwargs, 'escalation_id': l2.id},
    )
    return l2.id, result


def test_every_resolve_issue_mention_is_a_parseable_fenced_example():
    occurrences = _SKILL.read_text(encoding='utf-8').count(_CALL_TOKEN)

    assert occurrences >= 1, f'{_SKILL.name} names no {_CALL_TOKEN}'
    assert len(_EXAMPLES) == occurrences, (
        f'{occurrences} mentions of {_CALL_TOKEN} but {len(_EXAMPLES)} fenced blocks open with one: '
        f'found {[e.label() for e in _EXAMPLES]}'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('example', _EXAMPLES, ids=[e.test_id() for e in _EXAMPLES])
async def test_the_example_closes_a_level_2_record_with_its_provenance(served, example: _Example):
    kwargs = _kwargs(example)
    assert kwargs.get('resolution_class') in RESOLUTION_CLASSES, (
        f'{example.label()}: resolution_class={kwargs.get("resolution_class")!r} is not in RESOLUTION_CLASSES'
    )
    assert type(kwargs.get('ruled_by_human')) is bool, (
        f'{example.label()}: ruled_by_human={kwargs.get("ruled_by_human")!r} is not stated as a bool'
    )

    escalation_id, result = await _run(served, kwargs)

    assert 'error' not in result, (example.label(), result)
    record = served[1].get(escalation_id)
    assert record is not None and record.status != 'pending', (example.label(), record)
    assert record.resolution_class == kwargs['resolution_class']
    assert record.ruled_by_human is kwargs['ruled_by_human']
    assert resolver_tier(record) in {'human-ruled', 'attended-seat'}


@pytest.mark.asyncio
@pytest.mark.parametrize('example', _EXAMPLES, ids=[e.test_id() for e in _EXAMPLES])
async def test_the_example_without_its_class_is_refused(served, example: _Example):
    kwargs = {k: v for k, v in _kwargs(example).items() if k != 'resolution_class'}

    escalation_id, result = await _run(served, kwargs)

    assert result.get('code') == RESOLUTION_CLASS_REQUIRED_CODE, (example.label(), result)
    record = served[1].get(escalation_id)
    assert record is not None and record.status == 'pending'
