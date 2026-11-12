"""Read the trigger chain's submit shapes from the two docs that own them.

skills/_shared/filing-the-trigger-chain.md holds the ``submit_task`` calls an
agent makes, and docs/task-authoring.md §6 holds the metadata exemplars.  Only
fenced code is parsed, never prose.  Anything that cannot be read as a literal
raises :class:`DocShapeError` rather than being skipped, so a doc edit that
breaks extraction fails the suite instead of quietly shrinking it.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

SNIPPET_RELPATH = Path('skills/_shared/filing-the-trigger-chain.md')
AUTHORING_RELPATH = Path('docs/task-authoring.md')

_FENCE = '```'
_SECTION_6_HEADING = '## 6.'
_PLACEHOLDER = re.compile(r'<[^<>]+>')


class DocShapeError(Exception):
    """A doc's fenced code does not yield the shape this suite reads."""


def fenced_blocks(text: str, lang: str) -> list[str]:
    """The bodies of every fence in *text* whose info string is exactly *lang*."""
    blocks: list[str] = []
    body: list[str] | None = None
    keep = False
    for line in text.splitlines():
        stripped = line.strip()
        if body is None:
            if stripped.startswith(_FENCE):
                body, keep = [], stripped[len(_FENCE):].strip() == lang
        elif stripped == _FENCE:
            if keep:
                blocks.append('\n'.join(body))
            body = None
        else:
            body.append(line)
    if body is not None:
        raise DocShapeError(f'an unclosed {_FENCE} fence')
    return blocks


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _frozen(item) for key, item in value.items()})


def _frozen(value: object) -> object:
    if isinstance(value, dict):
        return _frozen_mapping(value)
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _call_kwargs(call: ast.Call) -> Mapping[str, object]:
    where = f'submit_task(...) at fence line {call.lineno}'
    if call.args:
        raise DocShapeError(f'{where} passes a positional argument')
    kwargs: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise DocShapeError(f'{where} passes **kwargs')
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError as exc:
            raise DocShapeError(f'{where}: keyword {keyword.arg!r} is not a literal') from exc
    return _frozen_mapping(kwargs)


def submit_calls(markdown: str) -> tuple[Mapping[str, object], ...]:
    """The keyword arguments of every ``submit_task`` call in *markdown*'s python fences."""
    calls = [
        _call_kwargs(node)
        for block in fenced_blocks(markdown, 'python')
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'submit_task'
    ]
    if not calls:
        raise DocShapeError('no submit_task call in any python fence')
    return tuple(calls)


def snippet_submit_calls(repo_root: Path) -> tuple[Mapping[str, object], ...]:
    return submit_calls((repo_root / SNIPPET_RELPATH).read_text())


def _section_6(text: str) -> str:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(_SECTION_6_HEADING)]
    if len(starts) != 1:
        raise DocShapeError(f'{AUTHORING_RELPATH} has {len(starts)} {_SECTION_6_HEADING!r} headings')
    tail = lines[starts[0] + 1:]
    end = next((i for i, line in enumerate(tail) if line.startswith('## ')), len(tail))
    return '\n'.join(tail[:end])


def authoring_metadata_exemplars(repo_root: Path) -> tuple[Mapping[str, object], ...]:
    """Every ```json metadata exemplar in docs/task-authoring.md §6 and its subsections."""
    blocks = fenced_blocks(_section_6((repo_root / AUTHORING_RELPATH).read_text()), 'json')
    if not blocks:
        raise DocShapeError(f'no json fence in {AUTHORING_RELPATH} §6')
    exemplars = []
    for block in blocks:
        try:
            exemplar = json.loads(block)
        except ValueError as exc:
            raise DocShapeError(f'a {AUTHORING_RELPATH} §6 json fence does not parse') from exc
        if not isinstance(exemplar, dict):
            raise DocShapeError(f'a {AUTHORING_RELPATH} §6 json fence is not an object')
        exemplars.append(_frozen_mapping(exemplar))
    return tuple(exemplars)


def _substitute(text: str, placeholders: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            return placeholders[token[1:-1]]
        except KeyError:
            raise DocShapeError(f'no concrete value for placeholder {token}') from None

    return _PLACEHOLDER.sub(replace, text)


def concretize(value: object, placeholders: Mapping[str, str]) -> Any:
    """A plain JSON-shaped deep copy of *value* with every ``<token>`` filled from *placeholders*.

    *placeholders* is keyed by the token's text without its angle brackets.
    """
    if isinstance(value, str):
        return _substitute(value, placeholders)
    if isinstance(value, Mapping):
        return {key: concretize(item, placeholders) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [concretize(item, placeholders) for item in value]
    return value
