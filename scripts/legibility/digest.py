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
