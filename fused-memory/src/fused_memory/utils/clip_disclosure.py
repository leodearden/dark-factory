"""One vocabulary for disclosing a clip, so no reader takes a partial read for a complete one.

Named 'clip' because ``shared/cap_markers.py`` already owns 'cap' for CLI usage banners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

TRUNCATED_KEY = 'truncated'


class TruncationFact(TypedDict):
    field: str
    cap: int
    elided: int | None  # None: a retrieval cap was hit, so how much lay beyond it is unknown


@dataclass(frozen=True)
class TextClip:
    text: str
    fact: TruncationFact | None


def elision_marker(elided: int) -> str:
    return f'…[+{elided} chars elided]'


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


def truncation_facts(*clips: TextClip) -> list[TruncationFact]:
    return [clip.fact for clip in clips if clip.fact is not None]
