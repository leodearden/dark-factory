"""Utility to extract a JSON object from LLM response text.

Handles:
- Markdown code fences (```json...``` or ```...```)
- Nested braces in JSON string values
- Arbitrary nesting depth
- Surrounding text before/after the JSON object
"""

from __future__ import annotations

import re

# Strip markdown code fences: ```json...``` or ```...```
_CODE_FENCE_RE = re.compile(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', re.DOTALL)


def extract_json(text: str) -> str | None:
    """Extract the first balanced JSON object from *text*.

    Strips markdown code fences, then locates the first ``{`` and scans
    forward using brace counting (with string-literal awareness so that
    braces inside quoted values are not counted).  Returns the balanced
    substring, or ``None`` if no complete JSON object is found.
    """
    if not text:
        return None

    # Strip markdown code fences if present
    m = _CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1)

    # Find the first '{' to start scanning
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            # Inside a string literal — braces don't count
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # No balanced object found
    return None
