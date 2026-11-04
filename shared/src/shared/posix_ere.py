"""POSIX ERE syntax read from Python, for diagnostics that must not
reinterpret a ``git grep -E`` pattern under Python's own rules."""

from __future__ import annotations

import re

_PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

#: Each POSIX character class as Python bracket content over ASCII, every
#: member escaped, so no emitted class can open a Python set operator.
_POSIX_CLASSES: dict[str, str] = {
    'alpha': 'a-zA-Z',
    'digit': '0-9',
    'alnum': 'a-zA-Z0-9',
    'upper': 'A-Z',
    'lower': 'a-z',
    'space': ' \\t\\n\\r\\f\\v',
    'blank': ' \\t',
    'xdigit': '0-9A-Fa-f',
    'punct': ''.join(re.escape(char) for char in _PUNCT),
    'cntrl': '\\x00-\\x1f\\x7f',
    'print': '\\x20-\\x7e',
    'graph': '\\x21-\\x7e',
}


def to_python_regex(pattern: str) -> str | None:
    """*pattern*, a POSIX ERE, as a Python regex that matches the same text.

    Only bracket expressions are rewritten: inside one a backslash is a
    literal, ``[:name:]`` is a character class, and Python's set operators
    (``&&``, ``||``, ``~~``, ``--``, a nested ``[``) mean nothing. ``None``
    when the bracket expression is unterminated or uses a feature with no
    ASCII translation (an unknown class, ``[.x.]``, ``[=x=]``).
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == '\\':
            out.append(pattern[i : i + 2])
            i += 2
            continue
        if char != '[':
            out.append(char)
            i += 1
            continue
        end = _bracket_end(pattern, i)
        if end < 0:
            return None
        translated = _translate_bracket(pattern[i + 1 : end - 1])
        if translated is None:
            return None
        out.append(translated)
        i = end
    return ''.join(out)


def _translate_bracket(body: str) -> str | None:
    """The bracket expression whose text between ``[`` and ``]`` is *body*."""
    negated = body.startswith('^')
    if negated:
        body = body[1:]
    members: list[str] = []
    i = 0
    while i < len(body):
        if _opens_bracket_term(body, i):
            delimiter = body[i + 1]
            close = body.find(delimiter + ']', i + 2)
            content = _POSIX_CLASSES.get(body[i + 2 : close]) if delimiter == ':' else None
            if content is None:
                return None
            members.append(content)
            i = close + 2
            continue
        if i + 2 < len(body) and body[i + 1] == '-':
            if _opens_bracket_term(body, i + 2):
                return None
            members.append(f'{re.escape(body[i])}-{re.escape(body[i + 2])}')
            i += 3
            continue
        members.append(re.escape(body[i]))
        i += 1
    return '[' + ('^' if negated else '') + ''.join(members) + ']'


def _opens_bracket_term(body: str, i: int) -> bool:
    """``[:class:]``, ``[.coll.]`` or ``[=equiv=]`` starts at *body[i]*."""
    return body[i] == '[' and body[i + 1 : i + 2] in (':', '.', '=')


def _bracket_end(pattern: str, start: int) -> int:
    """Index just past the ERE bracket expression opening at *start*, or -1.

    A ``]`` first in the list (after an optional ``^``) is literal, and
    ``[:class:]``, ``[.coll.]`` and ``[=equiv=]`` carry their own ``]``.
    """
    i = start + 1
    if pattern.startswith('^', i):
        i += 1
    if pattern.startswith(']', i):
        i += 1
    while i < len(pattern):
        if pattern[i] == ']':
            return i + 1
        if pattern[i] == '[' and pattern[i + 1 : i + 2] in (':', '.', '='):
            close = pattern.find(pattern[i + 1] + ']', i + 2)
            if close < 0:
                return -1
            i = close + 2
            continue
        i += 1
    return -1


def top_level_arms(pattern: str) -> list[str] | None:
    """The ERE alternatives of *pattern*: split on ``|`` outside groups,
    bracket expressions and escapes, after unwrapping a group that encloses
    the whole pattern. ``None`` when the pattern is not balanced.
    """
    arms: list[str] = []
    depth = 0
    start = 0
    wholly_grouped = pattern.startswith('(')
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == '\\':
            i += 2
            continue
        if char == '[':
            i = _bracket_end(pattern, i)
            if i < 0:
                return None
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return None
            if depth == 0 and i != len(pattern) - 1:
                wholly_grouped = False
        elif char == '|' and depth == 0:
            arms.append(pattern[start:i])
            start = i + 1
        i += 1
    if depth:
        return None
    if wholly_grouped and not arms:
        return top_level_arms(pattern[1:-1])
    arms.append(pattern[start:])
    return arms
