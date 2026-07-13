#!/usr/bin/env python3
"""Deterministic confusion-digest extractor (zero LLM).

Turns a single Claude Code session transcript JSONL file into a 5-15KB
markdown digest with YAML frontmatter: one section per confusion-signal
class (non-sidechain user turns, tool-error neighborhoods, self-corrections,
retry loops), plus a handful of scalar signal counts (not-found, df-guard
trips, interrupts).

Task alpha of the confusion-reduction PRD (plans/confusion-reduction-prd.md
Sec 5.1, contract Sec 7.2, boundary test Sec 8.1 producer side). A rewrite
of the agent-legibility survey's ephemeral scorer logic, properly, with
tests -- not a port of survey hacks.

Pure-function core + a thin `main(argv) -> int` argparse CLI, mirroring the
scripts/analyze_speculation_depth.py convention: every function below is
unit-testable against in-memory record lists (no clock, no network).
Malformed JSONL lines degrade (skip) rather than raise -- real transcripts
are written fire-and-forget and can have a truncated/corrupt trailing line.
"""
from __future__ import annotations

import json
from typing import Any


def load_transcript(path: Any) -> list[dict[str, Any]]:
    """Parse a transcript JSONL file into an ordered list of record dicts.

    Blank lines and lines that fail to parse as JSON are skipped rather
    than raising: fire-and-forget transcript writers can leave a truncated
    or corrupt trailing line, and one bad line must not abort the whole
    read (mirrors analyze_speculation_depth.load_events).
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _message_content(record: dict[str, Any]) -> Any:
    """Return a record's ``message.content``, or None if absent/malformed."""
    message = record.get('message')
    if not isinstance(message, dict):
        return None
    return message.get('content')


def _user_turn_text(content: Any) -> str | None:
    """Extract genuine human-typed text from a user record's content.

    ``content`` is either a plain string (the common case) or a list of
    content blocks. A list contributes only its 'text' blocks -- a user
    record whose content is entirely tool_result blocks (an answer TO the
    agent, not a human speaking) yields no text and is excluded.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block.get('text') for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
            and isinstance(block.get('text'), str)
        ]
        if texts:
            return '\n'.join(texts)
    return None


def _content_to_text(content: Any) -> str:
    """Best-effort flatten of a tool_result/message 'content' field to text.

    ``content`` is either a plain string (the common case) or a list of
    content blocks (dicts). Blocks without a 'text' field (e.g. the
    ``tool_reference`` blocks ToolSearch results carry) contribute nothing
    rather than being guessed at.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get('text') for block in content
            if isinstance(block, dict) and isinstance(block.get('text'), str)
        ]
        return '\n'.join(parts)
    return ''


def _summarize_input(tool_input: Any, *, limit: int = 200) -> str:
    """Canonical, size-capped string summary of a tool_use ``input`` dict."""
    try:
        summary = json.dumps(tool_input, sort_keys=True)
    except TypeError:
        summary = str(tool_input)
    if len(summary) > limit:
        summary = summary[:limit] + '...'
    return summary


def _iter_tool_use_blocks(
    records: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    """Return (record_index, block) for every assistant tool_use block."""
    found = []
    for index, record in enumerate(records):
        if record.get('type') != 'assistant':
            continue
        content = _message_content(record)
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                found.append((index, block))
    return found


def _iter_tool_result_blocks(
    records: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    """Return (record_index, block) for every user tool_result block."""
    found = []
    for index, record in enumerate(records):
        if record.get('type') != 'user':
            continue
        content = _message_content(record)
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                found.append((index, block))
    return found


def iter_error_neighborhoods(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each structured-error tool_result with its preceding attempt.

    Only tool_result blocks carrying a truthy ``is_error`` flag count --
    never a substring match on the result content (the core of the
    decoy-FAIL suppression decision, PRD Sec 13.2). Matching to the
    assistant's attempt is via ``tool_use_id``; an unmatched id (e.g. the
    attempt was truncated off the front of the transcript window) degrades
    to None attempt fields rather than raising.
    """
    attempts_by_id = {
        block.get('id'): block
        for _, block in _iter_tool_use_blocks(records)
    }

    neighborhoods = []
    for index, block in _iter_tool_result_blocks(records):
        if not block.get('is_error'):
            continue
        attempt = attempts_by_id.get(block.get('tool_use_id'))
        neighborhoods.append({
            'index': index,
            'attempt_tool': attempt.get('name') if attempt else None,
            'attempt_input_summary': (
                _summarize_input(attempt.get('input')) if attempt else None
            ),
            'error_content': _content_to_text(block.get('content')),
        })
    return neighborhoods


SELF_CORRECTION_PATTERNS: tuple[str, ...] = (
    "that's wrong",
    'let me fix',
    'my mistake',
    'i was wrong',
    'actually,',
    'correction:',
)
"""Curated self-correction markers, matched case-insensitively against
assistant TEXT blocks only (native-carrier scoping)."""


def _line_context(text: str, pos: int) -> str:
    """Return the newline-delimited line of *text* containing offset *pos*."""
    start = text.rfind('\n', 0, pos) + 1  # rfind returns -1 when absent -> 0
    end = text.find('\n', pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _assistant_text_blocks(
    records: list[dict[str, Any]],
) -> list[tuple[int, str]]:
    """Return (record_index, text) for every assistant 'text' content block.

    Excludes 'thinking' and 'tool_use' blocks -- only genuine assistant
    text shown to the user is the self-correction carrier.
    """
    found = []
    for index, record in enumerate(records):
        if record.get('type') != 'assistant':
            continue
        content = _message_content(record)
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text')
                if isinstance(text, str):
                    found.append((index, text))
    return found


def iter_self_corrections(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect curated self-correction markers in assistant TEXT blocks only.

    Native-carrier scoping: the same phrase inside a tool_result or inside
    a Write/Edit/MultiEdit/NotebookEdit tool_use input (an agent authoring
    test data, not a real correction) is never scanned -- restricting the
    scan to assistant 'text' blocks (see :func:`_assistant_text_blocks`)
    structurally excludes both.
    """
    hits = []
    for index, text in _assistant_text_blocks(records):
        lowered = text.lower()
        for pattern in SELF_CORRECTION_PATTERNS:
            pos = lowered.find(pattern)
            if pos == -1:
                continue
            hits.append({
                'index': index,
                'pattern': pattern,
                'context': _line_context(text, pos),
            })
    return hits


def iter_user_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return genuine non-sidechain, non-meta human user turns.

    Excludes: non-'user' records, isSidechain=True (subagent) turns,
    isMeta=True (system-injected) turns, and user records whose content is
    entirely tool_result blocks. User corrections are gold (PRD Sec 5) --
    this is the highest-priority digest section.
    """
    turns = []
    for index, record in enumerate(records):
        if record.get('type') != 'user':
            continue
        if record.get('isSidechain'):
            continue
        if record.get('isMeta'):
            continue
        text = _user_turn_text(_message_content(record))
        if text is None:
            continue
        turns.append({'index': index, 'text': text})
    return turns
