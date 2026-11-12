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


#: The escaped letters both engines read alike outside a bracket: GNU ERE's
#: word and space classes and word boundaries, which Python shares. Every
#: other escaped letter or digit diverges: ``\d`` is a literal ``d`` to
#: glibc and a digit class to Python.
_SHARED_LETTER_ESCAPES = frozenset('wWsSbB')

#: Escaped punctuation GNU ERE reads as an anchor and Python as itself.
_GNU_ANCHOR_ESCAPES = frozenset('<>`\'')

#: The characters that end a quantifier, after which Python 3.11+ reads a
#: ``+`` as possessive and GNU ERE as one more repetition.
_QUANTIFIER_ENDS = frozenset('*+?}')


def to_python_regex(pattern: str) -> str | None:
    """*pattern*, a POSIX ERE, as Python regex source; ``None`` where the two
    engines are known to read it differently.

    Bracket expressions are translated: inside one a backslash is a literal,
    ``[:name:]`` is a character class, and Python's set operators (``&&``,
    ``||``, ``~~``, ``--``, a nested ``[``) mean nothing. Outside a bracket
    nothing is rewritten; a construct there whose meaning differs yields
    ``None``: an escaped letter or digit outside
    :data:`_SHARED_LETTER_ESCAPES`, a GNU anchor escape, a trailing
    backslash, Python's ``(?`` extension syntax, and a ``+`` after a
    quantifier. So do an unterminated bracket expression and one using a
    feature with no ASCII translation (an unknown class, ``[.x.]``,
    ``[=x=]``). Everything else passes through as written, so the result
    serves a diagnostic, never a verdict.
    """
    out: list[str] = []
    after_quantifier = False
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == '\\':
            escaped = pattern[i + 1 : i + 2]
            if not _escape_reads_alike(escaped):
                return None
            out.append(char + escaped)
            i += 2
        elif char == '[':
            end = _bracket_end(pattern, i)
            translated = _translate_bracket(pattern[i + 1 : end - 1]) if end > 0 else None
            if translated is None:
                return None
            out.append(translated)
            i = end
        elif (char == '(' and pattern.startswith('?', i + 1)) or (
            char == '+' and after_quantifier
        ):
            return None
        else:
            out.append(char)
            i += 1
        after_quantifier = char in _QUANTIFIER_ENDS
    return ''.join(out)


def compile_ere(pattern: str) -> re.Pattern[str] | None:
    """*pattern*, a ``git grep -E`` ERE, compiled from
    :func:`to_python_regex`; ``None`` when it cannot be translated or Python
    rejects the translation."""
    translated = to_python_regex(pattern)
    if translated is None:
        return None
    try:
        return re.compile(translated)
    except re.error:
        return None


def _escape_reads_alike(escaped: str) -> bool:
    """A backslash before *escaped* means the same to GNU ERE and Python."""
    if not escaped:
        return False
    if escaped.isascii() and escaped.isalnum():
        return escaped in _SHARED_LETTER_ESCAPES
    return escaped not in _GNU_ANCHOR_ESCAPES


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
