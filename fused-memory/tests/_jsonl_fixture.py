"""Parse a committed JSONL fixture with the stdlib, one record per non-blank line.

Tests read their fixtures through this rather than through the script under
test's own loader, so a bug in the loader cannot mask a defect in the data
(and vice versa).
"""
from __future__ import annotations

import json
from pathlib import Path


def parse_jsonl_fixture(path: Path) -> list[dict]:
    assert path.exists(), f'fixture missing: {path}'
    out: list[dict] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - data defect
            raise AssertionError(f'{path}:{lineno} is not valid JSON: {exc}') from exc
    return out
