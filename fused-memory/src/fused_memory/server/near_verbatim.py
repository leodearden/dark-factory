"""Whether a text repeats a reference near-verbatim, and so adds nothing to it.

Near-verbatim means the same words and symbols, near-identically ordered: only
case, whitespace and small reorderings may differ. A changed number, a dropped
``not``, a flipped operator or an added sentence is never near-verbatim,
however long the text around it. The basis for the rule and its threshold is
calibration/write_triage_near_verbatim_report.md.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

#: Basis: calibration/write_triage_near_verbatim_report.md.
NEAR_VERBATIM_MIN_RATIO: float = 0.98

_TOKEN = re.compile(r'\w+|[^\w\s]')
_WHITESPACE_RUN = re.compile(r'\s+')


def _normalized(text: str) -> str:
    return _WHITESPACE_RUN.sub(' ', text.lower()).strip()


def _token_counts(text: str) -> Counter[str]:
    return Counter(_TOKEN.findall(text.lower()))


def text_similarity(a: str, b: str) -> float:
    """The character similarity of two texts, ignoring case and whitespace runs."""
    return difflib.SequenceMatcher(None, _normalized(a), _normalized(b), autojunk=False).ratio()


def is_near_verbatim(text: str, reference: str) -> bool:
    """True when ``text`` has exactly ``reference``'s words and symbols, near-identically ordered."""
    if _token_counts(text) != _token_counts(reference):
        return False
    return text_similarity(text, reference) >= NEAR_VERBATIM_MIN_RATIO
