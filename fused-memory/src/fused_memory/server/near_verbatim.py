"""Whether a text repeats a reference near-verbatim, and so adds nothing to it.

Near-verbatim means the same words and symbols in the same order: only case and
whitespace may differ. A changed number, a dropped ``not``, a flipped operator,
two swapped words or an added sentence is never near-verbatim, however long the
text around it. The rule has no free parameter and runs in time linear in the
texts' length. The basis for the rule is
calibration/write_triage_near_verbatim_report.md.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r'\w+|[^\w\s]')


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def is_near_verbatim(text: str, reference: str) -> bool:
    """True when ``text`` has exactly ``reference``'s words and symbols, in the same order."""
    return _tokens(text) == _tokens(reference)
