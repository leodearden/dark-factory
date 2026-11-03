"""Render the role prompts' code-quality block from the normative doc.

``code_quality.md`` beside this module is the single normative definition of
code quality. It is package data so it travels with the orchestrator install,
never with the operated repo, whose checkout may lack ``docs/``;
``docs/code-quality.md`` is a symlink to it.

``guidance()`` renders the ``## Definition`` body under a prompt title, then the
``## The fourteen heuristics``, ``## Two stances`` and ``## Do not steer by``
sections verbatim. A rendered section that is missing, empty or carries a brace
raises ``ValueError`` naming it, at import of
``orchestrator/src/orchestrator/agents/roles.py``.
"""

from __future__ import annotations

import importlib.resources
import re

_TITLE = '## Code quality — judge against this definition'
_DEFINITION = '## Definition'
_RENDERED = ('## The fourteen heuristics', '## Two stances', '## Do not steer by')

_NEXT_HEADING_RE = re.compile(r'^## ', re.MULTILINE)
_BRACE_LINE_RE = re.compile(r'^.*[{}].*$', re.MULTILINE)

NORMATIVE_DOC = importlib.resources.files('orchestrator.agents').joinpath('code_quality.md')


def section(text: str, heading: str) -> str:
    """Return the body between the *heading* LINE and the next ``## `` line.

    Raises :class:`ValueError` naming *heading* when no line of *text* is
    exactly that heading — including when it appears only as a substring, e.g.
    demoted to ``###``, which is drift rather than a match.
    """
    start = re.search(rf'^{re.escape(heading)}[ \t]*$', text, re.MULTILINE)
    if start is None:
        raise ValueError(
            f'heading {heading!r} not found as a whole line. Was the section '
            'renamed, removed, or demoted below `## `?'
        )
    terminator = _NEXT_HEADING_RE.search(text, start.end())
    return text[start.end():terminator.start() if terminator else len(text)]


def _body(doc_text: str, heading: str) -> str:
    body = section(doc_text, heading).strip()
    if not body:
        raise ValueError(f'section {heading!r} of the code-quality doc is empty.')
    brace_line = _BRACE_LINE_RE.search(body)
    if brace_line is not None:
        raise ValueError(
            f'section {heading!r} of the code-quality doc carries a literal brace '
            f'on the line {brace_line.group(0)!r}. The rendered block reaches '
            "roles.py::_REVIEWER_HEURISTICS_TEMPLATE's str.format(); rewrite the "
            'line without braces.'
        )
    return body


def render(doc_text: str) -> str:
    """Return the prompt block rendered from *doc_text*.

    Opens with ``\\n## `` so it starts its own section at every splice site, and
    ends with exactly one newline.
    """
    parts = [
        f'{_TITLE}\n\n{_body(doc_text, _DEFINITION)}',
        *(f'{heading}\n\n{_body(doc_text, heading)}' for heading in _RENDERED),
    ]
    return '\n' + '\n\n'.join(parts) + '\n'


def guidance() -> str:
    """Return the block rendered from the packaged normative doc."""
    return render(NORMATIVE_DOC.read_text(encoding='utf-8'))
