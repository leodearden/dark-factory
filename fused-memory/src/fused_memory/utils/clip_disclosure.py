"""One vocabulary for disclosing a clip, so no reader takes a partial read for a complete one.

Named 'clip' because ``shared/cap_markers.py`` already owns 'cap' for CLI usage banners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from shared.elision import elision_marker, word_boundary_head

# Not 'truncated': that response key already carries a bool elsewhere in fused-memory.
TRUNCATION_FACTS_KEY = 'truncation_facts'


@dataclass(frozen=True)
class TruncationFact:
    """``field`` was cut at ``cap``, losing ``elided``; ``None`` when a retrieval cap hides the count."""

    field: str
    cap: int
    elided: int | None

    def __post_init__(self) -> None:
        if self.cap <= 0:
            raise ValueError(f'TruncationFact.cap must be positive, got {self.cap!r}')
        if self.elided is not None and self.elided <= 0:
            raise ValueError(f'TruncationFact.elided must be positive or None, got {self.elided!r}')


@dataclass(frozen=True)
class TextClip:
    text: str
    fact: TruncationFact | None


def _clip(value: object, cap: int, *, field: str, marked: bool) -> TextClip:
    text = str(value or '')
    elided = len(text) - cap
    if elided <= 0:
        return TextClip(text=text, fact=None)
    head = text[:cap] + elision_marker(elided) if marked else text[:cap]
    return TextClip(text=head, fact=TruncationFact(field=field, cap=cap, elided=elided))


def clip_text(value: object, cap: int, *, field: str) -> TextClip:
    """Cut to ``cap`` with no inline marker: for a structured channel, where the fact rides beside the text."""
    return _clip(value, cap, field=field, marked=False)


def elide_text(value: object, cap: int, *, field: str) -> TextClip:
    """Cut to ``cap`` and append the counted marker: for text an LLM reads as prose."""
    return _clip(value, cap, field=field, marked=True)


def fit_at_word(text: str, limit: int) -> str:
    """Cut to at most ``limit`` chars ending in '…', for bounded text a mid-token cut would garble.

    The cut backs up to the last word boundary unless that would keep half the budget or less.
    """
    if limit <= 0:
        raise ValueError(f'fit_at_word limit must be positive, got {limit!r}')
    if len(text) <= limit:
        return text
    return word_boundary_head(text, limit - 1) + '…'


def truncation_fields(*facts: TruncationFact | None) -> dict[str, list[dict[str, Any]]]:
    """The facts present, as plain dicts under ``TRUNCATION_FACTS_KEY`` for a dict channel; ``{}`` when none."""
    present = [asdict(fact) for fact in facts if fact is not None]
    return {TRUNCATION_FACTS_KEY: present} if present else {}


def clip_note(*, omitted: int, omitted_noun: str, cap_name: str, cap: int, dropped_first: str) -> str:
    """The ``_NOTE:`` line a prompt reads as clipped coverage; '' when nothing was omitted."""
    if omitted <= 0:
        return ''
    return (
        f'_NOTE: {omitted} additional {omitted_noun} were omitted from this render by the '
        f'{cap_name}={cap} cap. Coverage was clipped — NOT complete this cycle; '
        f'{dropped_first} were dropped first._'
    )


def fetch_cap_note(*, fetched: int, noun: str, cap_name: str, cap: int) -> str:
    """The ``_NOTE:`` line for a retrieval that filled its cap, where the omitted count is unknowable."""
    if fetched < cap:
        return ''
    return (
        f'_NOTE: {fetched} {noun} were fetched, reaching the {cap_name}={cap} cap; any beyond it '
        f'were not fetched. Coverage may be clipped — NOT known complete this cycle._'
    )


def window_note(*, shown: int, total: int, noun: str, window: str, cap_name: str, cap: int) -> str:
    """The ``_NOTE:`` line for a section capped by design: it names the window, not a coverage gap."""
    if total <= shown:
        return ''
    return (
        f'_NOTE: showing the {shown} {window} of {total} {noun} ({cap_name}={cap}). '
        f'This section is a window by design, not a complete list._'
    )
