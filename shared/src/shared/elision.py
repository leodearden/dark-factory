"""The house elision marker and the word-safe cut it follows, shared by every package that clips text."""

from __future__ import annotations


def elision_marker(elided: int) -> str:
    return f'…[+{elided} chars elided]'


def word_boundary_head(text: str, budget: int) -> str:
    """At most ``budget`` chars of ``text``, backed up to the last word boundary unless that keeps half or less."""
    window = text[:budget]
    last_ws = next(
        (i for i in range(len(window) - 1, -1, -1) if window[i].isspace()),
        -1,
    )
    head = text[:last_ws] if last_ws > budget // 2 else window
    return head.rstrip()


def elide_at_word(text: str, cap: int) -> str:
    """``text`` if it fits ``cap``, else a word-boundary head plus the counted marker, the whole within ``cap``."""
    if len(text) <= cap:
        return text
    reserve = len(elision_marker(len(text)))
    budget = cap - reserve
    if budget < 1:
        raise ValueError(
            f'elide_at_word cap {cap} cannot hold the {reserve}-char elision marker '
            f'for a {len(text)}-char text'
        )
    head = word_boundary_head(text, budget)
    return head + elision_marker(len(text) - len(head))
