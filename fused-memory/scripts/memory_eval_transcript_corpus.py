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

Record schema
-------------
The corpus is JSON Lines — one search per line, keys sorted, UTF-8 kept
verbatim. ``schema_version`` is stamped on every RECORD rather than in a
header so a consumer that reads a single line still knows what it is
holding::

    {
      "schema_version": 1,             // SCHEMA_VERSION; bump on any change
      "transcript":     "3214/-home-…/<session>.jsonl.gz",
      "task_id":        "3214",        // ─┐
      "session_id":     "<uuid>",      //  │ ALL path-derived, by
      "is_subagent":    false,         //  │ parse_archive_path — never
      "subagent_id":    null,          // ─┘ read from the record bodies
      "tool_use_id":    "toolu_…",     // correlates the call to its answer
      "tool_name":      "mcp__fused-memory__search",
      "query":          "<the full query text>",
      "params":         {"project_id": "dark_factory", "limit": 5},
      "caller":         {"agent_id": "claude-task-3214-implementer",
                         "task_id": "3214", "role": "implementer"},
      "result_status":  "ok",          // ok | error | unparsed | missing
      "result_count":   2,
      "results": [
        {"rank": 1,                    // 1-based, the order the agent SAW
         "id": "<memory uuid>",
         "score": 0.91,                // the store's relevance_score
         "source_store": "graphiti",
         "category": "decisions_and_rationale",
         "content_chars": 148}         // LENGTH only — see below
      ]
    }

Field provenance, and why each is where it is:

- ``transcript`` is archive-RELATIVE, so the corpus stays portable across
  machines and checkouts. ``--transcript`` mode recovers that same relative
  slice from the absolute path it is given (``archive_relative_slice``),
  falling back to the bare file name only for a transcript that is genuinely
  not in an archive tree — so the two modes emit identical records for the
  same file rather than differing on provenance.
- ``task_id`` / ``session_id`` / ``is_subagent`` / ``subagent_id`` come from
  the archive PATH alone. Records do carry ``isSidechain`` / ``agentId`` /
  ``gitBranch`` hints, but those describe what the agent was doing rather
  than where the transcript was filed, so the path stays authoritative and
  the hints remain available as an independent cross-check.
- ``query`` is lifted out of the tool ``input``; ``params`` is everything
  else that was passed (``project_id``, ``limit``, …). Splitting them keeps
  the query greppable without discarding the call's other arguments.
- ``caller`` is *who issued the search*, recovered from the briefing's Agent
  Identity line — **not** the ``tool_use`` block's own ``caller`` key (see
  the warning above). Its nested ``task_id`` is the id the agent believes it
  is running as; the TOP-LEVEL ``task_id`` is the id the archive filed it
  under. They normally agree, and keeping both means a disagreement is
  visible rather than averaged away.
- ``results[].content_chars`` is a LENGTH, never the text. The corpus records
  *what was retrieved and how well it scored*, which is what a retrieval eval
  measures; copying result bodies would duplicate the live memory store into
  a gitignored file that silently staled the moment a memory was edited.
  A consumer that needs the text looks the ``id`` up in the store.
- ``result_status`` is a closed four-value vocabulary — ``ok``, ``error``,
  ``unparsed``, ``missing`` — recording how well the search's ANSWER was
  recovered. Only ``ok`` is a measurement of the store; the other three are
  gaps in this instrument. They are kept distinct from a genuine zero-hit
  search (``ok`` with ``result_count == 0``) because all four look identical
  to a consumer filtering on the count alone, and a decoding bug would then
  read as a recall collapse. See :data:`_RESULT_STATUS`.

Artifacts (paths named here in TEXT because the lock-charter file lint's
extension allowlist has no ``jsonl`` entry, so they cannot be declared in
``metadata.files``)::

    fused-memory/data/memory-evals/transcript-corpus/corpus-<STAMP>.jsonl
    fused-memory/data/memory-evals/transcript-corpus/coverage-<STAMP>.json
    fused-memory/data/memory-evals/transcript-corpus/report-<STAMP>.txt

``data/`` is gitignored, so a run never dirties a diff.

Declared consumers
------------------
- PRD leaf η's **write-after-miss validation set** — the population of real
  searches with real answers that it would otherwise have to synthesise.
- **Operators**, who read ``report-<STAMP>.txt`` to see how much of the
  archive a run actually covered and what it could not read.
- Future **E3 / E8 runners** and the **shadow-replay harness**, which need a
  fixed, re-runnable set of historical queries to replay against.

Scope boundary (PRD §7): this leaf produces the CORPUS ONLY. No shadow-replay
harness, no E3 golden query set, no E8 re-derivation audit, and no write or
mutation of the live memory corpus — the script reads transcripts off disk
and writes under ``fused-memory/data/memory-evals/``, nothing else.

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

That vocabulary is only closed if a damaged file lands INSIDE it. The archive
is fire-and-forget runtime state written by the running fleet, so a unit
killed mid-write or a file read while still compressing leaves a partially
written ``.gz``, and a flipped byte in stored data damages an intact one — and
an unreadable transcript surfaces as FOUR different exception types, only one
of which is an ``OSError``. ``legibility.inventory.as_unreadable_file_error``
enumerates them and normalizes all four to ``OSError`` preserving the original
message; BOTH readers this script uses funnel through that one helper, so
every shape lands in the counted ``parse_failures`` path — ``degraded`` if
other files read, ``total_failure`` if none did — instead of aborting the run
with a traceback and an exit code outside this table.

A malformed ``--stamp`` (or ``$MEMORY_EVAL_RUN_STAMP``) is the one other way a
run can end without a status, and it too has a code rather than a traceback:
:data:`EXIT_BAD_STAMP` / 4. It is outside the status table on purpose — no
corpus was produced and no coverage could be, so calling it ``total_failure``
would claim an archive was read when none was opened.

Usage::

    # whole archive (default root = the MAIN checkout's archive)
    python fused-memory/scripts/memory_eval_transcript_corpus.py

    # one transcript, for debugging
    python fused-memory/scripts/memory_eval_transcript_corpus.py \
        --transcript <path/to/session.jsonl.gz>
"""
from __future__ import annotations

import argparse
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

# The `# type: ignore[reportMissingImports]` is not a defect being papered
# over: these resolve at RUNTIME via the insert above, but pyright is invoked
# from fused-memory/ whose extraPaths cannot see repo-root scripts/. Same
# situation, same idiom, as escalation/'s imports of orchestrator.*.
from legibility.digest import load_transcript  # type: ignore[reportMissingImports]  # noqa: E402
from legibility.inventory import iter_json_lines  # type: ignore[reportMissingImports]  # noqa: E402
from shared.memory_eval_metrics import (  # noqa: E402
    RUN_STAMP_ENV_VAR,
    MetricSchemaError,
    eval_dir,
    report_artifact_path,
    run_stamp,
    validate_stamp,
)

# `validate_stamp` is the module's one shape check for a run stamp. The
# alternatives here are worse: re-deriving the stamp regex would be a SECOND
# definition of a shape that module documents as the one thing it must
# validate (an override "is going into a filename the window loader will
# order and cut by string comparison"), and not validating at all is the
# defect this fixes. The public `run_stamp()`/`write_metric_series(stamp=)`
# entry points both route through it; this script needs the check WITHOUT
# their side effects, since it stamps three artifacts of its own. Task 3434
# promoted it to that module's `__all__` so this import no longer needs the
# underscore-private workaround.

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
# Archive-path provenance
# ---------------------------------------------------------------------------


def _strip_transcript_suffix(name: str) -> str:
    """``<x>.jsonl.gz`` / ``<x>.jsonl`` -> ``<x>``."""
    if name.endswith('.jsonl.gz'):
        return name[: -len('.jsonl.gz')]
    if name.endswith('.jsonl'):
        return name[: -len('.jsonl')]
    return name


def parse_archive_path(rel_path: str | Path) -> dict[str, Any]:
    """Recover task/session/subagent identity from an archive-relative path.

    The layout is the one ``shared.transcript_archive`` WRITES — named rather
    than line-cited so this reader stays coupled to that producer by name as
    it evolves::

        <task_id>/<enc-cwd>/<session_id>.jsonl.gz
        <task_id>/<enc-cwd>/<session_id>/subagents/agent-<hex>.jsonl.gz

    Every field degrades to None independently rather than raising. An archive
    is append-only runtime state that can acquire a stray file, and one
    unexpected name must not abort a scan over thousands of good ones.

    Note this is the ONLY source of task/session identity the corpus uses.
    Records do carry ``isSidechain``/``gitBranch`` hints, but those are a
    property of what the agent was doing rather than of where the transcript
    was filed, so the path stays authoritative.
    """
    parts = Path(rel_path).parts
    parsed: dict[str, Any] = {
        'task_id': None,
        'session_id': None,
        'is_subagent': False,
        'subagent_id': None,
    }
    if len(parts) >= 1:
        parsed['task_id'] = parts[0] if len(parts) > 1 else None
    if len(parts) == 3:
        # <task_id>/<enc-cwd>/<session_id>.jsonl.gz
        parsed['session_id'] = _strip_transcript_suffix(parts[2])
    elif len(parts) == 5 and parts[3] == 'subagents':
        # <task_id>/<enc-cwd>/<session_id>/subagents/agent-<hex>.jsonl.gz
        parsed['session_id'] = parts[2]
        parsed['is_subagent'] = True
        parsed['subagent_id'] = _strip_transcript_suffix(parts[4])
    return parsed


def _is_encoded_cwd(part: str) -> bool:
    """Does *part* look like ``transcript_archive``'s encoded-cwd directory?

    ``encode_cwd`` maps ``/`` to ``-``, so an absolute working directory
    always encodes to a leading ``-``. Measured across the live archive's 499
    distinct level-2 directories: ALL 499 start with ``-``.

    This is the discriminator :func:`archive_relative_slice` needs and the
    task-id component cannot supply — task dirs are usually numeric but not
    always (the live archive holds ``df_task_12``-style ids from eval runs),
    so "is it a number" would both over- and under-match.
    """
    return part.startswith('-')


def archive_relative_slice(path: str | Path) -> str:
    """The archive-relative tail of *path*, or its bare name if it has none.

    ``--transcript`` mode is handed a path from the operator's shell — usually
    absolute, and therefore carrying the archive root as a prefix that the
    scan path has already stripped. Passing it to :func:`parse_archive_path`
    whole yields the wrong task id (some ancestor directory name); passing
    only ``path.name`` yields a single component, which parses to task_id
    ``None`` and drags session/subagent identity to null with it.

    So recover the slice: if the tail matches one of the two layouts
    ``shared.transcript_archive`` writes, return exactly those 3 (or 5)
    components; otherwise return the bare file name, which parses to the
    all-null provenance that is the honest answer for a transcript sitting
    outside any archive tree.

    WHY this matters beyond tidiness: ``--transcript`` is the documented
    operator DEBUG path, so its output is read side by side with the scan
    path's. Without this, a subagent transcript mined here reported
    ``is_subagent: false`` and a null task — a confidently wrong answer, and
    the exact kind an operator comparing two modes would trust
    (reviewer_comprehensive/correctness, task 3214 amendment pass).

    The layout is matched, never assumed: the encoded-cwd component must
    actually look like one (:func:`_is_encoded_cwd`), so an arbitrary
    ``a/b/c.jsonl`` under some temp dir is NOT silently relabelled with
    ``a`` as its task id. Recovering nothing is correct there; inventing
    provenance is not.
    """
    parts = Path(path).parts
    # <task_id>/<enc-cwd>/<session_id>/subagents/agent-<hex>.jsonl.gz
    if len(parts) >= 5 and parts[-2] == 'subagents' and _is_encoded_cwd(parts[-4]):
        return Path(*parts[-5:]).as_posix()
    # <task_id>/<enc-cwd>/<session_id>.jsonl.gz
    if len(parts) >= 3 and _is_encoded_cwd(parts[-2]):
        return Path(*parts[-3:]).as_posix()
    return Path(path).name


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


def _content_blocks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
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


_RESULT_STATUS = ('ok', 'error', 'unparsed', 'missing')
"""How well the search's ANSWER was recovered — a closed vocabulary.

- ``ok``       the answer was decoded; ``results`` is what the agent saw.
- ``error``    the tool call itself failed (``is_error``).
- ``unparsed`` an answer arrived but could not be decoded.
- ``missing``  no answer appears anywhere (a truncated transcript).

Only ``ok`` is a measurement of the store. The other three are gaps in this
instrument, and they are kept distinct from a genuine zero-hit search (which
is ``ok`` with ``result_count == 0``) because all four look identical to a
consumer that filters on the count alone — a decoding bug would then read as
a recall collapse.
"""


def _result_text(content: Any) -> str | None:
    """Flatten a ``tool_result`` payload to its JSON text, or None.

    Two shapes occur: a bare string (the common one), and a list of blocks
    whose ``text`` carries the payload.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block['text']
            for block in content
            if isinstance(block, Mapping) and isinstance(block.get('text'), str)
        ]
        if parts:
            return ''.join(parts)
    return None


def _parse_tool_result(block: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Decode a ``tool_result`` block into ``(result_status, results)``.

    ``is_error`` is checked FIRST and wins even over a decodable payload: a
    failed call's body is an error message, and reading one as results would
    manufacture a measurement out of a failure.
    """
    if block.get('is_error') is True:
        return 'error', []
    text = _result_text(block.get('content'))
    if text is None:
        return 'unparsed', []
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return 'unparsed', []
    if not isinstance(payload, Mapping):
        return 'unparsed', []
    raw_results = payload.get('results')
    if not isinstance(raw_results, list):
        # A payload that decoded but carries no results list is not an empty
        # search — it is a shape we do not understand, and saying so is the
        # whole point of keeping 'unparsed' separate from 'ok'.
        return 'unparsed', []
    return 'ok', _project_results(raw_results)


def extract_searches(
    records: Iterable[Mapping[str, Any]],
    *,
    source: str | None = None,
    tool_names: frozenset[str] = SEARCH_TOOL_NAMES,
    provenance: Mapping[str, Any] | None = None,
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
        provenance: task/session fields from :func:`parse_archive_path`,
            merged into every emitted record.
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
                partial.update(dict(provenance or {}))
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


# ---------------------------------------------------------------------------
# Archive scan + coverage accounting
# ---------------------------------------------------------------------------

DEFAULT_MAX_FAILURE_EXAMPLES = 20
"""Enough to see a pattern, few enough that the report stays readable. The
COUNT is never capped, and a truncated example list discloses itself."""


def _new_coverage() -> dict[str, Any]:
    """A zeroed coverage mapping. Every key is always present, even at zero:
    an absent counter reads as "not measured", which is a different claim."""
    return {
        'tasks_scanned': 0,
        'transcripts_found': 0,
        'transcripts_read': 0,
        'searches_extracted': 0,
        'searches_unresolved': 0,
        'parse_failures': {
            'count': 0,
            'examples': [],
            'examples_truncated': False,
            'examples_omitted': 0,
        },
    }


def _record_failure(
    coverage: dict[str, Any], rel: str, reason: str, max_examples: int
) -> None:
    """Count an unreadable file, keeping a bounded, self-disclosing example."""
    failures = coverage['parse_failures']
    failures['count'] += 1
    if len(failures['examples']) < max_examples:
        failures['examples'].append({'transcript': rel, 'reason': reason})
    else:
        failures['examples_truncated'] = True
        failures['examples_omitted'] += 1


def iter_transcripts(root: Path) -> list[Path]:
    """Every transcript under *root*, in deterministic (sorted) order.

    Both archive suffixes, both layouts. Sorted so two runs over an unchanged
    archive emit byte-identical corpora — without that, a re-run would diff
    against itself and a real change would be unreadable.
    """
    if not root.is_dir():
        return []
    found = set(root.glob('*/**/*.jsonl.gz')) | set(root.glob('*/**/*.jsonl'))
    return sorted(p for p in found if p.is_file())


def scan_archive(
    root: str | Path,
    *,
    tool_names: frozenset[str] = SEARCH_TOOL_NAMES,
    max_failure_examples: int = DEFAULT_MAX_FAILURE_EXAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mine a whole archive tree. Returns ``(records, coverage)``.

    Each file streams through the PROMOTED ``legibility.inventory.iter_json_lines``
    rather than being slurped: the live archive is thousands of multi-MB gz
    files, so memory has to stay bounded by the largest single record.

    Per that reader's documented contract, a blank or corrupt LINE is skipped
    silently while an unreadable FILE raises ``OSError``. That split is
    exactly the accounting this coverage needs: ``parse_failures`` counts
    files we could not read at all, and a truncated trailing line — which
    every fire-and-forget writer eventually produces — does not inflate it
    into a false alarm.

    All FOUR unreadable-file shapes reach that ``except OSError`` — bad magic,
    a truncated stream, a corrupt body, and a non-UTF-8 byte — because
    ``legibility.inventory.as_unreadable_file_error`` normalizes them at the
    reader seam. The handler is deliberately NOT widened here to also catch
    ``EOFError``/``zlib.error``/``UnicodeDecodeError``: that helper is the one
    answer to "which exceptions mean an unreadable file", and a second answer
    at each call site is exactly what INV-5 forbids. The original message
    survives normalization, so the disclosed ``parse_failures`` example still
    says which shape it was.

    A missing *root* is not an error here; it is zero transcripts found, which
    :func:`coverage_status` turns into ``no_input``.
    """
    root = Path(root)
    coverage = _new_coverage()
    records: list[dict[str, Any]] = []
    tasks: set[str] = set()

    for path in iter_transcripts(root):
        rel = path.relative_to(root).as_posix()
        coverage['transcripts_found'] += 1
        provenance = parse_archive_path(rel)
        if provenance['task_id'] is not None:
            tasks.add(provenance['task_id'])
        try:
            extracted = extract_searches(
                iter_json_lines(path),
                source=rel,
                tool_names=tool_names,
                provenance=provenance,
            )
        except OSError as exc:
            _record_failure(coverage, rel, f'{type(exc).__name__}: {exc}',
                            max_failure_examples)
            continue
        coverage['transcripts_read'] += 1
        records.extend(extracted)

    coverage['tasks_scanned'] = len(tasks)
    coverage['searches_extracted'] = len(records)
    coverage['searches_unresolved'] = sum(
        1 for record in records if record['result_status'] != 'ok'
    )
    return records, coverage


def scan_transcript(
    path: str | Path,
    *,
    tool_names: frozenset[str] = SEARCH_TOOL_NAMES,
    max_failure_examples: int = DEFAULT_MAX_FAILURE_EXAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mine ONE transcript. Same return shape as :func:`scan_archive`.

    Reads via ``legibility.digest.load_transcript`` — the slurping sibling of
    the scan path's streaming reader. An ordered in-memory list is the right
    shape for one small file, and it gives the operator-facing debug path a
    second, independent reader onto the IDENTICAL core: if the two ever
    disagree, a parser has been duplicated (INV-5 / PRD D9).

    The coverage mapping is shape-identical, so ``coverage_status``,
    ``render_report`` and ``write_corpus`` are shared verbatim between modes.
    An unreadable file is accounted for the same way — ``total_failure`` and
    its non-zero exit, not a traceback — for every corruption shape, because
    ``load_transcript`` and its streaming sibling normalize through the SAME
    ``inventory.as_unreadable_file_error``, so they agree by construction
    rather than by two copies staying in step. That agreement is what lets an
    operator reach for either mode on a damaged archive and get the same
    verdict; it is asserted directly, not assumed
    (``test_legibility_digest.TestLoadTranscriptCorruptionShapes``).

    Provenance is recovered here too, not skipped: see
    :func:`archive_relative_slice`. A path that still sits inside a
    well-formed archive tree yields the same ``task_id``/``session_id``/
    ``is_subagent``/``subagent_id`` the scan path would have given it, so the
    debug mode's answer is comparable to the real one rather than uniformly
    null.
    """
    path = Path(path)
    coverage = _new_coverage()
    coverage['transcripts_found'] = 1
    rel = archive_relative_slice(path)
    provenance = parse_archive_path(rel)
    if provenance['task_id'] is not None:
        coverage['tasks_scanned'] = 1

    try:
        records = extract_searches(
            load_transcript(path),
            source=rel,
            tool_names=tool_names,
            provenance=provenance,
        )
    except OSError as exc:
        _record_failure(coverage, rel, f'{type(exc).__name__}: {exc}', max_failure_examples)
        return [], coverage

    coverage['transcripts_read'] = 1
    coverage['searches_extracted'] = len(records)
    coverage['searches_unresolved'] = sum(
        1 for record in records if record['result_status'] != 'ok'
    )
    return records, coverage


# ---------------------------------------------------------------------------
# Run status — the INV-2 / INV-4 seam
# ---------------------------------------------------------------------------

EXIT_CODES = {'ok': 0, 'degraded': 0, 'no_input': 2, 'total_failure': 3}
"""Process exit code per run status.

Distinct non-zero codes for the two zero-search cases, because a scheduler or
wrapper reads the exit code and never opens the artifact. ``degraded`` is 0
deliberately: a partial failure that is fully disclosed (a count plus a
bounded, self-disclosing example list) still yields a usable corpus, while a
wholesale failure does not.
"""

EXIT_BAD_STAMP = 4
"""A malformed ``--stamp`` / ``$MEMORY_EVAL_RUN_STAMP``: no run, no artifacts.

Deliberately OUTSIDE :data:`EXIT_CODES` rather than a fifth status. Those four
describe how much of an archive a completed run covered; this one says the run
never started, so no coverage exists to describe. Folding it into
``total_failure`` would claim transcripts were found and none could be read —
a specific, false statement about the archive.

It exists at all because the alternative is a traceback: a malformed stamp
raises ``MetricSchemaError`` out of the stamp validator, and an uncaught one
exits 1, which is not in this module's documented vocabulary. The same
argument that makes a corrupt transcript land inside the status table makes a
bad stamp land inside the exit-code table.
"""


def coverage_status(coverage: Mapping[str, Any]) -> str:
    """Resolve a coverage mapping to one of :data:`EXIT_CODES`' statuses.

    WHY four statuses rather than a search count plus a failure tally:

    An extraction run that produces zero searches has at least three very
    different causes — the archive is empty, every transcript failed to read,
    or the archive is fine and genuinely contains no searches. All three
    report ``searches_extracted == 0``, so a count cannot tell them apart, and
    a consumer looking only at the count would read a totally broken run as a
    clean empty result. That is precisely the failure this script exists to
    make impossible (design-invariants INV-2 structured-facts-at-failure,
    INV-4 no-silent-fail-soft).

    So the cause is resolved here into a status that a human reads in the
    report and a wrapper reads as an exit code:

    - no transcripts found at all -> ``no_input`` (nothing to measure)
    - transcripts found, none readable -> ``total_failure`` (the instrument
      is broken; whatever the archive holds, we did not see it)
    - some readable, some not -> ``degraded`` (usable, with disclosed gaps)
    - all readable -> ``ok`` (whatever count it reports is a measurement)

    Note ``ok`` says nothing about how many searches were found: an archive
    that was fully read and holds none is a legitimate measurement.
    """
    found = coverage.get('transcripts_found', 0)
    read = coverage.get('transcripts_read', 0)
    if not found:
        return 'no_input'
    if not read:
        return 'total_failure'
    if read < found:
        return 'degraded'
    return 'ok'


def stamp_status(coverage: dict[str, Any]) -> dict[str, Any]:
    """Resolve and stamp ``coverage['status']`` in place, returning *coverage*.

    The artifact must carry the status, not only the process exit code: a
    coverage file read a week later has no exit code attached to it, and
    re-deriving one from the counters is exactly the guesswork the status
    exists to remove.
    """
    coverage['status'] = coverage_status(coverage)
    return coverage


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def render_report(coverage: Mapping[str, Any]) -> str:
    """The human-readable coverage report.

    What an operator reads when pointed at a run. It leads with the STATUS,
    in words, because the counters alone cannot distinguish an empty archive
    from a wholesale failure — and an operator should never have to infer
    "there was no archive" from a zero.
    """
    failures = coverage.get('parse_failures') or {}
    status = coverage.get('status') or coverage_status(coverage)
    lines = [
        f'memory-eval run: {EVAL_ID}',
        f'status:          {status}  (exit {EXIT_CODES.get(status, "?")})',
        '',
        f'tasks scanned:       {coverage.get("tasks_scanned", 0)}',
        f'transcripts found:   {coverage.get("transcripts_found", 0)}',
        f'transcripts read:    {coverage.get("transcripts_read", 0)}',
        f'searches extracted:  {coverage.get("searches_extracted", 0)}',
        f'searches unresolved: {coverage.get("searches_unresolved", 0)}'
        '   (answer not recoverable; see result_status)',
        '',
        f'parse failures (unreadable transcripts): {failures.get("count", 0)}',
    ]

    if status == 'no_input':
        lines.append(
            '  NO INPUT — no transcripts were found. This is NOT a measurement '
            'of an empty archive; nothing was read at all.'
        )
    elif status == 'total_failure':
        lines.append(
            '  TOTAL FAILURE — transcripts were found but NONE could be read. '
            'The zero searches below measure this instrument, not the archive.'
        )
    elif status == 'degraded':
        lines.append(
            '  DEGRADED — some transcripts were unreadable. The corpus is '
            'usable but incomplete by exactly the count above.'
        )

    examples = failures.get('examples') or []
    if examples:
        lines.append('')
        lines.append('  examples:')
        lines.extend(
            f'    {ex.get("transcript")} — {ex.get("reason")}' for ex in examples
        )
    if failures.get('examples_truncated'):
        lines.append(
            f'    … and {failures.get("examples_omitted", 0)} more not shown '
            '(example list capped; the count above is complete).'
        )
    return '\n'.join(lines) + '\n'


def write_corpus(
    records: Iterable[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    out_root: str | Path,
    *,
    stamp: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write the three artifacts. Returns ``(corpus, coverage, report)`` paths.

    Stamping goes through the shared ``run_stamp``/``RUN_STAMP_ENV_VAR``, so
    one logical memory-eval run correlates across leaves.

    An explicit *stamp* gets the SAME shape check ``run_stamp`` applies to its
    env-var route, because overriding is precisely how a caller would
    otherwise sidestep the validated field. All three artifact names carry the
    stamp, and every declared consumer (leaf η, the E3/E8 runners) picks a run
    by ordering those filenames as strings — so ``--stamp latest`` would sort
    outside the chronological sequence and silently select the wrong run. A
    malformed stamp raises ``MetricSchemaError`` here, BEFORE any path is
    built or any directory created, so a rejected run leaves nothing behind.

    The layout is delegated, never restated: ``report_artifact_path`` already
    produces ``<root>/<eval_id>/report-<STAMP>.txt``, so it is called
    verbatim, and the two artifacts with no helper are placed in the directory
    it implies — obtained from ``eval_dir(out_root, EVAL_ID)``, the helper
    ``shared.memory_eval_metrics`` exposes for callers that need only the
    directory. It replaces the older
    ``metrics_artifact_path(..., 'ignored').parent`` idiom, which now raises
    ``MetricSchemaError``: the path helpers validate their stamp, and this
    call site was the one caller they had left stamp-unvalidated for.

    Serialization mirrors ``serialize_metric_series`` — ``sort_keys=True``,
    ``ensure_ascii=False``, trailing newline — so two identical runs are
    byte-identical and a real change diffs cleanly. The corpus drops ``indent``
    only because JSON Lines requires one object per physical line.

    Plain writes, not an atomic writer: this is a one-shot script with no
    concurrent reader, and copying ``memory_eval_metrics._atomic_write_text``
    (itself already a documented copy) would create a third copy of it.
    """
    stamp = validate_stamp(stamp, what='--stamp') if stamp else run_stamp()
    report_path = report_artifact_path(out_root, EVAL_ID, stamp)
    directory = eval_dir(out_root, EVAL_ID)
    directory.mkdir(parents=True, exist_ok=True)

    corpus_path = directory / f'corpus-{stamp}.jsonl'
    coverage_path = directory / f'coverage-{stamp}.json'

    corpus_path.write_text(
        ''.join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n'
            for record in records
        ),
        encoding='utf-8',
    )
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    report_path.write_text(render_report(coverage), encoding='utf-8')
    return corpus_path, coverage_path, report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ARCHIVE_RELPATH = Path('data') / 'orchestrator' / 'agent-transcripts'


def default_archive_root() -> Path:
    """``<main checkout>/data/orchestrator/agent-transcripts``.

    Resolved against the MAIN checkout, not this file's own: the archive is
    untracked runtime state that exists only there, and a git worktree has no
    ``data/`` directory at all. A ``__file__``-relative default would make
    every worktree run of a correct script report ``no_input`` — which is the
    worst failure mode for this particular script, because it would train
    operators to read exit 2 as normal and hollow out the very signal the
    script exists to create. Exit 2 has to keep meaning what it says.

    ``resolve_main_checkout`` raises ``ValueError`` when the path is not
    inside a git working tree or ``git`` is unavailable (a hermetic temp-dir
    test, say); the repo's established fallback still canonicalizes. Called
    lazily so merely importing this module never spawns ``git``.
    """
    from fused_memory.models.scope import resolve_main_checkout  # noqa: PLC0415

    try:
        checkout = Path(resolve_main_checkout(_PACKAGE_ROOT))
    except ValueError:
        checkout = Path(_REPO_ROOT).resolve()
    return checkout / ARCHIVE_RELPATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--archive-root',
        default=None,
        help=(
            'Root of the agent-transcript archive. Defaults to the MAIN '
            "checkout's data/orchestrator/agent-transcripts (resolved via "
            'git worktree list), because the archive is untracked runtime '
            'state that exists only there — a worktree has no data/ dir. '
            'A genuinely absent or empty archive yields no_input / exit 2, '
            'never a silent empty corpus.'
        ),
    )
    parser.add_argument(
        '--transcript',
        default=None,
        help='Mine ONE transcript file instead of the whole archive (debugging).',
    )
    parser.add_argument(
        '--out-root',
        default=str(DEFAULT_OUT_ROOT),
        help='Artifact root (default: %(default)s). Gitignored.',
    )
    parser.add_argument(
        '--stamp',
        default=None,
        help=f'Run stamp; defaults to ${RUN_STAMP_ENV_VAR} or the current UTC '
             'time. Must be a YYYYmmddTHHMMSSZ stamp — artifact names are '
             f'ordered by string comparison. A malformed one exits '
             f'{EXIT_BAD_STAMP} and writes nothing.',
    )
    parser.add_argument(
        '--max-failure-examples',
        type=int,
        default=DEFAULT_MAX_FAILURE_EXAMPLES,
        help='Cap on disclosed failure examples (default: %(default)s). '
             'The COUNT is never capped, and truncation is disclosed.',
    )
    parser.add_argument(
        '--tool-name',
        action='append',
        dest='tool_names',
        default=None,
        help='Tool name counting as a search (repeatable). '
             f'Default: {sorted(SEARCH_TOOL_NAMES)}.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tool_names = frozenset(args.tool_names) if args.tool_names else SEARCH_TOOL_NAMES

    if args.transcript:
        records, coverage = scan_transcript(
            Path(args.transcript),
            tool_names=tool_names,
            max_failure_examples=args.max_failure_examples,
        )
    else:
        root = Path(args.archive_root) if args.archive_root else default_archive_root()
        records, coverage = scan_archive(
            root,
            tool_names=tool_names,
            max_failure_examples=args.max_failure_examples,
        )

    stamp_status(coverage)
    try:
        _, _, report_path = write_corpus(
            records, coverage, args.out_root, stamp=args.stamp
        )
    except MetricSchemaError as exc:
        # The one failure that is a USAGE error rather than a coverage
        # outcome. Caught here so it reads as a message plus a documented
        # code, not a traceback plus exit 1 — the scan is already done and
        # discarded either way, since the stamp names all three artifacts.
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_BAD_STAMP
    report = render_report(coverage)
    print(report, end='')
    print(f'report: {report_path}')
    return EXIT_CODES[coverage['status']]


if __name__ == '__main__':
    sys.exit(main())
