"""The committed tool-call markup corpus, read by ``tool_use_id``.

One reader for every orchestrator suite that drives a real harness payload, so
a change to the corpus's format or location is one edit here.

A small loader rather than an import of
``shared/tests/toolcall_markup_corpus_extract``: the orchestrator package
cannot import another package's test tree. The format is one JSON object per
line, which is the whole of what that helper does for a reader.

Keyed by id rather than by index or by "the shortest one", so a corpus refresh
that adds records cannot silently repoint a test at a different payload. Read
as JSON, which is also how the raw envelope literals reach a test WITHOUT
appearing in its own source text: the corpus escapes every one of them as
``\\u003c``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / 'shared' / 'tests' / 'fixtures' / 'toolcall_markup_corpus.jsonl'
)


def load_corpus() -> list[dict[str, Any]]:
    """Every committed corpus record, in file order."""
    records = [
        json.loads(line)
        for line in CORPUS_PATH.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert records, f'the committed corpus at {CORPUS_PATH} is empty'
    return records


def corpus_specimen(tool_use_id: str) -> dict[str, Any]:
    """The committed corpus record with this ``tool_use_id``."""
    for record in load_corpus():
        if record.get('tool_use_id') == tool_use_id:
            return record
    raise AssertionError(f'specimen {tool_use_id!r} is missing from {CORPUS_PATH}')
