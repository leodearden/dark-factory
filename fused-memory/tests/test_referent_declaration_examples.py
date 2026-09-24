"""Certifier: every documented ``entities=`` example is one the live gate accepts.

Task 3675 (PRD ``plans/memory-referent-fidelity-prd.md`` leaf kappa).
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

#: The keyword the extractor scans documents for. Pinned below to the LIVE
#: signature of both write tools, so a parameter rename goes red here instead
#: of leaving every documented example "valid" under a dead name.
_DECLARATION_KEYWORD = 'entities'


# ── The scan keyword is a live parameter ─────────────────────────────────────


@pytest.mark.parametrize('tool_name', ['add_memory', 'add_episode'])
def test_declaration_keyword_is_a_live_write_tool_parameter(tool_name):
    tool = create_mcp_server(AsyncMock())._tool_manager.get_tool(tool_name)
    assert tool is not None, f'the {tool_name} tool is not registered on the MCP server'

    parameters = inspect.signature(tool.fn).parameters
    assert _DECLARATION_KEYWORD in parameters, (
        f'{tool_name} declares no {_DECLARATION_KEYWORD!r} parameter (it has '
        f'{sorted(parameters)}). Every documented example this file certifies '
        'teaches that keyword, and leaf delta (task 3669) is the premise that '
        'it exists. If it was renamed, rename _DECLARATION_KEYWORD and every '
        'documented example with it.'
    )


# ── The extractor, against hand-written fixtures only ────────────────────────

_ORIGIN = 'fixture.md'

_HAPPY_DOC = """\
Declare what the write is about:

```
add_memory(
  content="Task 3127's retry loop swallows the timeout",
  category="observations_and_summaries",
  entities=[{'kind': 'task', 'id': 3127}]
)
```

and when nothing applies:

```
add_memory(content="ruff is the only linter", entities=[])
```
"""


def test_happy_doc_yields_every_example_in_order_with_its_own_content():
    examples = _declared_examples(_HAPPY_DOC, origin=_ORIGIN)

    assert [(e.literal, e.content) for e in examples] == [
        ([{'kind': 'task', 'id': 3127}], "Task 3127's retry loop swallows the timeout"),
        ([], 'ruff is the only linter'),
    ]


def test_doc_without_any_declaration_fails_loudly_rather_than_yielding_nothing():
    doc = 'add_memory(content="x", category="temporal_facts")\n'

    with pytest.raises(AssertionError, match=_ORIGIN):
        _declared_examples(doc, origin=_ORIGIN)


@pytest.mark.parametrize(
    'doc',
    [
        pytest.param("add_memory(entities=[{'kind': 'task', 'id': 3127}\n", id='truncated'),
        pytest.param("entities=[{'kind': 'task', 'id': 3127}", id='unbalanced-at-eof'),
    ],
)
def test_unbalanced_list_fails_loudly_rather_than_yielding_a_prefix(doc):
    with pytest.raises(AssertionError, match=_ORIGIN):
        _declared_examples(doc, origin=_ORIGIN)


@pytest.mark.parametrize(
    'doc',
    [
        pytest.param('add_memory(entities=[<ids>])', id='placeholder'),
        pytest.param('add_memory(entities={ids})', id='f-string-field'),
        pytest.param('add_memory(entities=[{task}])', id='f-string-field-in-list'),
        pytest.param('add_memory(entities=...)', id='ellipsis'),
        pytest.param('To skip the check, omit `entities=` entirely.', id='prose-mention'),
    ],
)
def test_non_literal_value_fails_loudly(doc):
    with pytest.raises(AssertionError, match=_ORIGIN):
        _declared_examples(doc, origin=_ORIGIN)


def test_bracket_or_comma_inside_a_quoted_string_does_not_end_the_walk():
    doc = """add_memory(entities=[{'kind': 'task', 'id': 7, 'project_id': "a],b"}])"""

    [example] = _declared_examples(doc, origin=_ORIGIN)

    assert example.literal == [{'kind': 'task', 'id': 7, 'project_id': 'a],b'}]


def test_only_the_bare_keyword_argument_is_matched():
    doc = """\
metadata={"canonical_entities": 1}
add_memory(canonical_entities=[{'kind': 'task', 'id': 1}])
```json
{"memory_hints": {"entities": ["TaskInterceptor"]}}
```
add_memory(content="Task 2 landed", entities=[{'kind': 'task', 'id': 2}])
"""

    examples = _declared_examples(doc, origin=_ORIGIN)

    assert [e.literal for e in examples] == [[{'kind': 'task', 'id': 2}]]


def test_content_pairs_within_the_same_call_never_a_neighbouring_one():
    doc = """\
```
add_memory(content="Task 111 is unrelated", category="temporal_facts")
```

```
add_memory(category="temporal_facts", entities=[{'kind': 'task', 'id': 222}])
```

```
add_episode(source="text", content="Task 333 notes", entities=[{'kind': 'task', 'id': 333}])
```
"""

    examples = _declared_examples(doc, origin=_ORIGIN)

    assert [(e.literal[0]['id'], e.content) for e in examples] == [
        (222, None),
        (333, 'Task 333 notes'),
    ]
