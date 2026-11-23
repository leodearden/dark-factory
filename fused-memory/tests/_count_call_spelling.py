"""Whether agent-facing text spells count_memories_by_metadata with keyword args.

The MCP tool requires the keyword ``filters``; text that spells the call
positionally, ``count_memories_by_metadata(project_id, {...})``, leads agents
to pass the dict under another key, so the call fails validation and the
stage2_suppress gate fails open (task 7715).

Lives outside conftest.py, for the same reason ``_fm_helpers`` does, so test
files can ``from _count_call_spelling import X``.
"""

from __future__ import annotations

import re

CALL_OPENING = re.compile(r'count_memories_by_metadata\(')
KEYWORD_CALL = re.compile(r'count_memories_by_metadata\(project_id=[^,()]*,\s*filters=')

_EXCERPT_LENGTH = 80


def non_keyword_count_calls(text: str) -> list[str]:
    """An excerpt from each call opening not spelled ``project_id=..., filters=...``.

    Empty when every call in ``text`` is keyword-spelled.
    """
    return [
        text[opening.start() : opening.start() + _EXCERPT_LENGTH]
        for opening in CALL_OPENING.finditer(text)
        if not KEYWORD_CALL.match(text, opening.start())
    ]
