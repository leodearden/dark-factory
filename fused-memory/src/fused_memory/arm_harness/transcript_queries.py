"""Search queries read from a ``memory_eval_transcript_corpus`` ``corpus-<STAMP>.jsonl``.

These are the retrieval probe's query source: real searches agents issued. Only
records whose answer was recovered (``result_status == 'ok'``) are kept, since the
other statuses are gaps in that instrument, not searches the store answered.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ACCEPTED_SCHEMA_VERSION = 1
"""The ``scripts/memory_eval_transcript_corpus.py::SCHEMA_VERSION`` this reader understands."""

OK_STATUS = 'ok'


def iter_transcript_queries(path: Path) -> Iterator[str]:
    """Yield each answered search's query, in file order; blank lines are skipped."""
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            query = _answered_query(_record(line, line_number), line_number)
            if query is not None:
                yield query


def _record(line: str, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f'transcript corpus line {line_number} is not JSON: {error}') from error
    if not isinstance(record, dict):
        raise ValueError(f'transcript corpus line {line_number} is not a JSON object')
    if record.get('schema_version') != ACCEPTED_SCHEMA_VERSION:
        raise ValueError(
            f'transcript corpus line {line_number} has schema_version '
            f'{record.get("schema_version")!r}; this reader accepts {ACCEPTED_SCHEMA_VERSION}'
        )
    return record


def _answered_query(record: dict[str, Any], line_number: int) -> str | None:
    if record.get('result_status') != OK_STATUS:
        return None
    query = record.get('query')
    if not isinstance(query, str):
        raise ValueError(
            f'transcript corpus line {line_number} is an ok record without a string query: '
            f'{query!r}'
        )
    return query
