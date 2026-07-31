#!/usr/bin/env python3
"""Retro replay corpus — mine every memory search out of the archived transcripts.

PRD ``docs/prds/memory-eval-program.md`` §5 leaf θ, decision D9.

One-shot and strictly READ-ONLY. It reads the already-archived agent
transcripts off disk and writes three artifacts under
``fused-memory/data/memory-evals/transcript-corpus/``. It never opens the
memory store, never writes a memory, and never mutates the archive.

Why a retro corpus at all
-------------------------
The going-forward telemetry seam records searches as they happen, but it
started recording recently and PRD §3 measured that 99.7% of its journal rows
are unattributed. The archive, meanwhile, already holds hundreds of real
searches with their real answers — what was asked, what came back, at what
scores. That is the population leaf η's write-after-miss validation needs and
that a shadow-replay harness would otherwise have to synthesise. This script
recovers it once.

What it extracts
----------------
For each ``mcp__fused-memory__search`` call found in a transcript: the full
query, the ids and scores of the results that were actually SHOWN to the
agent, who issued it, and which task/session it came from. Result **content
text is deliberately not copied** — see "Record schema" below.

The transcript shapes it reads (measured, not assumed)
------------------------------------------------------
Archive layout, written by ``shared.transcript_archive``::

    <archive_root>/<task_id>/<enc-cwd>/<session_id>.jsonl.gz
    <archive_root>/<task_id>/<enc-cwd>/<session_id>/subagents/agent-<hex>.jsonl.gz

A search is an ``assistant`` record whose ``message.content`` holds::

    {"type": "tool_use", "id": "toolu_…", "name": "mcp__fused-memory__search",
     "input": {"query": …, "project_id": …, "limit": …},
     "caller": {"type": "direct"}}

Its answer is a LATER ``user`` record holding::

    {"type": "tool_result", "tool_use_id": "toolu_…", "content": "<JSON string>"}

decoding to ``{"results": [{"id", "content", "category", "source_store",
"relevance_score", …}]}``. Non-message records (``queue-operation``,
``attachment``, ``last-prompt``) are interleaved throughout and are skipped.

.. warning::

   **The source ``tool_use`` block carries its own ``caller`` key**, uniformly
   ``{"type": "direct"}`` across every measured call. That is a Claude Code
   harness field; it discriminates nothing and it is **not** agent identity.
   This script's ``caller`` field means *who issued the search* and is
   recovered from the briefing's Agent Identity line — a different thing that
   happens to share a name. Do not conflate them.

Failure semantics
-----------------
A run that extracted nothing must never be ambiguous about why, so coverage
resolves to one of four statuses with distinct exit codes:

===============  ====  ==========================================
status           exit  meaning
===============  ====  ==========================================
``ok``           0     every transcript found was read
``degraded``     0     some read, some unreadable — all disclosed
``no_input``     2     no archive, or an archive with no transcripts
``total_failure``  3   transcripts found, none readable
===============  ====  ==========================================

The two zero-search cases (an empty archive vs. a run where every transcript
failed) therefore differ in BOTH the status string a human reads and the exit
code a wrapper reads. See :func:`coverage_status`.

Usage::

    # whole archive (default root = the MAIN checkout's archive)
    python fused-memory/scripts/memory_eval_transcript_corpus.py

    # one transcript, for debugging
    python fused-memory/scripts/memory_eval_transcript_corpus.py \
        --transcript <path/to/session.jsonl.gz>
"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _PACKAGE_ROOT.parent

# `legibility` is a namespace package under repo-root scripts/ — no __init__.py,
# installed nowhere, and NOT on fused-memory's pytest `pythonpath`. This insert
# is the only way the two mandated readers import at all, so it runs at module
# scope rather than under a __main__ guard (the importlib-loaded test needs it
# too). Idempotent, guarded — the migrate_tasks_json_to_sqlite.py idiom.
#
# `shared` deliberately gets NO such insert: fused-memory/pyproject.toml
# declares dark-factory-shared as a workspace dependency, so `shared.*` already
# resolves. A second mechanism would silently diverge from the declared one.
_SCRIPTS_ROOT = _REPO_ROOT / 'scripts'
if _SCRIPTS_ROOT.exists() and str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from legibility.digest import load_transcript  # noqa: E402
from legibility.inventory import iter_json_lines  # noqa: E402

# INV-5 / D9: TWO existing readers, ONE core, ZERO new parsers. The scan path
# streams via iter_json_lines (memory-bounded across thousands of multi-MB gz
# files); --transcript mode slurps via load_transcript (an ordered list is the
# right shape for one small file). Both feed the identical extract_searches.
# The test asserts these bindings ARE those functions, so a future regression
# that quietly reintroduces a local parser fails rather than passing review.

EVAL_ID = 'transcript-corpus'

DEFAULT_OUT_ROOT = _PACKAGE_ROOT / 'data' / 'memory-evals'
"""Artifact root. ``data/`` is gitignored (fused-memory/.gitignore:9), so a
run's output never lands in a diff by accident."""

SCHEMA_VERSION = 1

SEARCH_TOOL_NAMES = frozenset({'mcp__fused-memory__search'})
"""The tool calls that count as a memory search.

A frozenset rather than a constant string because ``--tool-name`` widens it:
the archive spans months of tool-name history, and a future rename must be
minable without editing this file.
"""


# ---------------------------------------------------------------------------
# Caller identity
# ---------------------------------------------------------------------------

_AGENT_ID_RE = re.compile(r'agent_id:\*\*\s*`([^`]+)`')
"""The briefing's Agent Identity line: ``- **agent_id:** `claude-task-…` ``.

Anchored on the backticks because the surrounding markdown varies between
briefing templates while the backticked value does not.
"""

_TASK_ROLE_RE = re.compile(r'^claude-task-(\d+)-(.+)$')
"""``claude-task-<id>-<role>``. Role is the REMAINDER, so a multi-segment role
(``code-reviewer``) survives instead of truncating at its first hyphen."""


def _empty_caller() -> dict[str, Any]:
    return {'agent_id': None, 'task_id': None, 'role': None}


def _caller_from_text(text: str) -> dict[str, Any] | None:
    """Parse one record's text into a caller block, or None if it holds none.

    Every part degrades independently: an agent_id that does not match
    ``claude-task-<id>-<role>`` (an interactive session, say) is preserved
    verbatim with task_id/role None. Nothing here raises, and nothing here
    can suppress a search record — an unattributed search stays in the corpus
    as a disclosed gap, because dropping it would bias the corpus toward
    whichever roles happen to carry a parseable briefing.
    """
    match = _AGENT_ID_RE.search(text)
    if match is None:
        return None
    agent_id = match.group(1)
    caller = _empty_caller()
    caller['agent_id'] = agent_id
    shape = _TASK_ROLE_RE.match(agent_id)
    if shape is not None:
        caller['task_id'] = shape.group(1)
        caller['role'] = shape.group(2)
    return caller


def _caller_from_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Look for the Agent Identity line in a ``user`` record's text.

    Only user records: the briefing arrives as one, and an assistant record
    quoting the marker back is the agent talking about identity, not carrying
    it. Content may be a plain string (the usual briefing shape) or a block
    list, so both are flattened.
    """
    if record.get('type') != 'user':
        return None
    message = record.get('message')
    if not isinstance(message, Mapping):
        return None
    content = message.get('content')
    if isinstance(content, str):
        return _caller_from_text(content)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping):
                text = block.get('text')
                if isinstance(text, str):
                    caller = _caller_from_text(text)
                    if caller is not None:
                        return caller
    return None


# ---------------------------------------------------------------------------
# The pure extraction core
# ---------------------------------------------------------------------------


def _content_blocks(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The record's ``message.content`` blocks, or [] for anything else.

    Absorbs every non-message shape the archive interleaves: records with no
    ``message`` at all (``queue-operation``), a null message
    (``attachment``), and a plain-string content (most user turns).
    """
    message = record.get('message')
    if not isinstance(message, Mapping):
        return []
    content = message.get('content')
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, Mapping)]


def _project_results(raw_results: Iterable[Any]) -> list[dict[str, Any]]:
    """Project the shown results to ids, scores and sizes — never their text.

    Rank is 1-based and positional: the order the store returned them is the
    order the agent saw them, which is the signal a retrieval eval is after.

    A missing field degrades to None rather than raising, so one malformed
    entry cannot cost us its siblings.
    """
    projected: list[dict[str, Any]] = []
    for rank, entry in enumerate(raw_results, start=1):
        if not isinstance(entry, Mapping):
            projected.append({
                'id': None, 'score': None, 'source_store': None,
                'category': None, 'content_chars': None, 'rank': rank,
            })
            continue
        content = entry.get('content')
        projected.append({
            'id': entry.get('id'),
            'score': entry.get('relevance_score'),
            'source_store': entry.get('source_store'),
            'category': entry.get('category'),
            'content_chars': len(content) if isinstance(content, str) else None,
            'rank': rank,
        })
    return projected


def _parse_tool_result(block: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Decode a ``tool_result`` block into ``(result_status, results)``.

    The payload is a JSON *string* carrying ``{"results": [...]}``.
    """
    content = block.get('content')
    if not isinstance(content, str):
        return 'unparsed', []
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return 'unparsed', []
    if not isinstance(payload, Mapping):
        return 'unparsed', []
    raw_results = payload.get('results')
    if not isinstance(raw_results, list):
        return 'unparsed', []
    return 'ok', _project_results(raw_results)


def extract_searches(
    records: Iterable[Mapping[str, Any]],
    *,
    source: str | None = None,
    tool_names: frozenset[str] = SEARCH_TOOL_NAMES,
) -> list[dict[str, Any]]:
    """Mine one transcript's records into corpus records. Pure, single-pass.

    Walks *records* once. Each matching ``tool_use`` block becomes a pending
    partial keyed by its ``tool_use_id``; the matching ``tool_result`` fills it
    in when it streams past. A search whose answer never arrives keeps the
    ``result_status='missing'`` it was created with rather than being dropped —
    a truncated transcript loses the ANSWER, not the fact that the search
    happened.

    Emission is in ``tool_use`` order, so a search answered late still sits
    where it was issued.

    Args:
        records: the transcript's records, in order.
        source: archive-RELATIVE path of the transcript, stamped on each
            record so the corpus stays portable across machines.
        tool_names: which tool names count as a search.
    """
    emitted: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    caller: dict[str, Any] | None = None

    for record in records:
        if not isinstance(record, Mapping):
            continue
        if caller is None:
            # First match wins and is never overwritten: a later record
            # quoting a different agent_id is one agent discussing another's
            # briefing, not a change of who is running.
            caller = _caller_from_record(record)
        for block in _content_blocks(record):
            block_type = block.get('type')
            if block_type == 'tool_use':
                name = block.get('name')
                if name not in tool_names:
                    continue
                tool_use_id = block.get('id')
                raw_input = block.get('input')
                params = dict(raw_input) if isinstance(raw_input, Mapping) else {}
                query = params.pop('query', None)
                partial = {
                    'schema_version': SCHEMA_VERSION,
                    'transcript': source,
                    'tool_use_id': tool_use_id,
                    'tool_name': name,
                    'query': query,
                    'params': params,
                    'result_status': 'missing',
                    'result_count': 0,
                    'results': [],
                }
                emitted.append(partial)
                if isinstance(tool_use_id, str):
                    pending[tool_use_id] = partial
            elif block_type == 'tool_result':
                partial = pending.pop(block.get('tool_use_id'), None)  # type: ignore[arg-type]
                if partial is None:
                    continue
                status, results = _parse_tool_result(block)
                partial['result_status'] = status
                partial['results'] = results
                partial['result_count'] = len(results)

    # Anything left pending keeps the 'missing' status it was created with —
    # no end-of-walk flush is needed, because the pending partial IS the
    # already-emitted record. Nothing can be dropped by forgetting to flush.
    #
    # Caller is stamped after the walk, not during it: the briefing precedes
    # every search in a real transcript, but relying on that ordering accident
    # would silently unattribute any transcript that violates it.
    resolved = caller or _empty_caller()
    for record in emitted:
        record['caller'] = dict(resolved)
    return emitted
