"""Certifier: every documented ``entities=`` example is one the live gate accepts.

Task 3675 (PRD ``plans/memory-referent-fidelity-prd.md`` leaf kappa). Several
sites teach agents to declare a write's referents: the markdown guides, the
review-checkpoint reflection template and the recon prompt fragment. Each one
carries a hand-transcribed ``entities=`` example that must agree with the live
``add_memory`` signature and gate. That is the INV-5 lockstep shape, and this
repo has already seen hand-transcribed prompt text drift twice in one file. So
the agreement is checked by a machine, here, once for every site.

EXECUTE, NOT COMPARE. Each extracted literal is run through the live
``fused_memory.server.entities_gate::entities_gate`` together with the content
of its own call. This is the ``tests/scripts/test_package_source_lookup_convention.py``
doctrine: certify that the recipe works, not that it is still spelled the same.
Rewording the prose around an example never fails here. A renamed key, a label
used as an id, or an id the example's own content contradicts does fail. The
scan keyword is pinned to the live tool signatures as well, so renaming the
parameter fails here too.

WHY THIS SUITE. ``fused-memory/pyproject.toml`` sets
``pythonpath = ["src", "../orchestrator/src"]``, so both the gate and the
orchestrator template are declared imports here. ``tests/scripts/`` runs under
``uv run --project shared``, and that project's declared closure has no
fused_memory: it raises ModuleNotFoundError under ``--isolated``. It is present
only incidentally, in a venv provisioned with ``--all-packages``.
``tests/scripts/test_check_fused_memory_unit_parity.py`` records the same
finding.

KNOWN LIMITATION. A task-role diff that touches only ``.md`` files does not run
this file, because ``orchestrator/src/orchestrator/verify.py::_has_source_files``
counts only ``.py``/``.rs`` files. Merge-role full-breadth verify and review
checkpoints still run it.

Sibling guards on the same contract: ``test_referent_guidance_prompt_drift.py``
checks that the rendered fragment reaches every recon prompt that writes, and
``orchestrator/tests/test_review_checkpoint_reflection_splice.py`` checks that
the reflection template reaches the review prompt.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.entities_gate import entities_gate
from fused_memory.server.tools import create_mcp_server
from fused_memory.utils.referent_resolution import render_referent_declaration_guidance

#: The keyword the extractor scans documents for. Pinned below to the LIVE
#: signature of both write tools, so a parameter rename goes red here instead
#: of leaving every documented example "valid" under a dead name.
_DECLARATION_KEYWORD = 'entities'

_DECLARATION_RE = re.compile(rf'(?<!\w){_DECLARATION_KEYWORD}[ \t]*=(?!=)[ \t]*')
_CONTENT_RE = re.compile(r'(?<!\w)content[ \t]*=(?!=)[ \t]*')
_WRITE_CALL_RE = re.compile(r'add_(?:memory|episode)[ \t]*\(')
_STRING_RE = re.compile(r"""'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*\"""")


@dataclass(frozen=True)
class DeclaredExample:
    """One documented declaration, and the content of the write call it sits in."""

    literal: list[Any]
    content: str | None
    where: str


def _declared_examples(text: str, *, origin: str) -> list[DeclaredExample]:
    """Every ``entities=<list literal>`` in *text*, in document order.

    Fails loudly, naming *origin*, rather than returning ``[]``. An extractor
    that silently finds nothing would make every assertion downstream pass
    while certifying nothing.
    """
    examples = [_declared_example(text, m, origin=origin) for m in _DECLARATION_RE.finditer(text)]
    assert examples, (
        f'{origin}: no `{_DECLARATION_KEYWORD}=` example found. A guarded site '
        'that shows no declaration teaches nothing about it. Add a worked call, '
        'or drop the site from _GUARDED_SITES if it no longer teaches writes.'
    )
    return examples


def _declared_example(text: str, match: re.Match[str], *, origin: str) -> DeclaredExample:
    where = f'{origin}:{text.count(chr(10), 0, match.start()) + 1}'
    return DeclaredExample(
        literal=_list_literal(text, match.end(), where=where),
        content=_call_content(text, match.start(), where=where),
        where=where,
    )


def _list_literal(text: str, start: int, *, where: str) -> list[Any]:
    assert text.startswith('[', start), (
        f'{where}: `{_DECLARATION_KEYWORD}=` is not followed by a list literal. '
        'Every example must be executable: write a concrete list ([] or '
        "[{'kind': 'task', 'id': 3127}]), never a placeholder, and phrase "
        f'omission as "omit `{_DECLARATION_KEYWORD}`", without the "=".'
    )
    end = _balanced_end(text, start, '[', ']')
    assert end is not None, f'{where}: the `{_DECLARATION_KEYWORD}=[` list never closes.'
    span = text[start:end]
    try:
        value = ast.literal_eval(span)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise AssertionError(
            f'{where}: `{_DECLARATION_KEYWORD}={span}` is not a Python literal ({exc}). '
            'Write concrete values; a placeholder cannot be checked against the gate.'
        ) from exc
    assert isinstance(value, list), f'{where}: `{_DECLARATION_KEYWORD}={span}` is not a list.'
    return value


def _call_content(text: str, offset: int, *, where: str) -> str | None:
    """The ``content=`` string of the write call enclosing *offset*, if any."""
    call = _enclosing_write_call(text, offset)
    if call is None:
        return None
    match = _CONTENT_RE.search(text, *call)
    if match is None:
        return None
    string = _STRING_RE.match(text, match.end())
    assert string is not None, (
        f'{where}: the enclosing call passes `content=` something other than a '
        'quoted string, so the declaration cannot be checked against it.'
    )
    return ast.literal_eval(string.group())


def _enclosing_write_call(text: str, offset: int) -> tuple[int, int] | None:
    """``(open, close)`` of the nearest ``add_memory(``/``add_episode(`` spanning *offset*."""
    starts = list(_WRITE_CALL_RE.finditer(text, 0, offset))
    if not starts:
        return None
    open_paren = starts[-1].end() - 1
    close = _balanced_end(text, open_paren, '(', ')')
    if close is None or close <= offset:
        return None
    return open_paren, close


def _balanced_end(text: str, start: int, opener: str, closer: str) -> int | None:
    """Offset just past the *closer* balancing the *opener* at *start*.

    Quoted strings are skipped whole, so a bracket or comma inside one never
    ends the walk. Returns None if the text runs out first.
    """
    depth = 0
    pos = start
    while pos < len(text):
        string = _STRING_RE.match(text, pos)
        if string is not None:
            pos = string.end()
            continue
        if text[pos] == opener:
            depth += 1
        elif text[pos] == closer:
            depth -= 1
        pos += 1
        if depth == 0:
            return pos
    return None


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


# ── Every guarded site teaches declarations the live gate accepts ────────────

_GROUP_ID = 'dark_factory'

#: Content for an example whose call carries none. It names no task, so the
#: gate's content scan is empty and can never conflict: such an example is
#: certified for SHAPE only. Named here so that is a visible choice.
_CONTENT_NAMING_NO_TASK = 'Fallback content for a contentless example; it names no task.'

_GUARDED_SITES = (('recon-prompt-fragment', render_referent_declaration_guidance),)


def _rejection(example: DeclaredExample) -> str | None:
    block = entities_gate(
        example.literal,
        content=example.content if example.content is not None else _CONTENT_NAMING_NO_TASK,
        group_id=_GROUP_ID,
    )
    if block is None:
        return None
    details = {k: block[k] for k in ('conflicts', 'content_referents') if k in block}
    return (
        f'{example.where}: {_DECLARATION_KEYWORD}={example.literal!r} -> '
        f'{block["error_type"]} {details or ""}: {block.get("hint", block["error"])}'
    )


@pytest.mark.parametrize(('site_id', 'load'), _GUARDED_SITES, ids=[s for s, _ in _GUARDED_SITES])
def test_every_guarded_example_is_accepted_by_the_live_gate(site_id, load):
    examples = _declared_examples(load(), origin=site_id)

    assert any(example.literal for example in examples), (
        f'{site_id}: every example declares []. The gate accepts that, but it '
        'teaches nothing about the entry shape. Show at least one non-empty '
        'declaration whose content names what it declares.'
    )
    rejections = [r for r in map(_rejection, examples) if r is not None]
    assert not rejections, (
        f'{site_id} teaches declarations the live entities_gate rejects:\n' + '\n'.join(rejections)
    )


def test_rendered_fragment_names_the_declaration_keyword():
    assert _DECLARATION_KEYWORD in render_referent_declaration_guidance()


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
