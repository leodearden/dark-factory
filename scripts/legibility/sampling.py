#!/usr/bin/env python3
"""Zero-LLM signal scorer + stratified budget sampler (PRD §5.2 point 2, §7.4, §8.4).

Scores a session transcript's confusion signals with a single zero-token
pass (:func:`score_signals`), classifies its agent class
(:func:`classify_agent_class`), collapses near-duplicate shapes
(:func:`shape_fingerprint` / :func:`dedupe_shapes`), and picks a
budget-bounded, stratified subset for the nightly digest
(:func:`stratified_sample`). :func:`main` wires
``config.load_config -> inventory.enumerate_sessions -> score_signals ->
classify_agent_class -> stratified_sample -> render_manifest`` into the
CLI acceptance surface.

Task β of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.2, contract §7.4). Self-contained — does not import task α's
``digest.py`` (owns its own transcript-line iteration and signal-scoring
primitives rather than reaching into another task's module).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterator


def _iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed dict records from a JSONL file, skipping blank/malformed lines.

    A transcript is written fire-and-forget and can have a truncated or
    corrupt trailing line, which must not abort the whole read. Raises
    ``OSError`` if *path* cannot be opened at all (caller's concern).
    """
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


# ---------------------------------------------------------------------------
# score_signals — 5-class zero-token confusion-signal scorer (PRD §7.2)
# ---------------------------------------------------------------------------

_NOT_FOUND_PATTERNS: tuple[str, ...] = (
    'no such file',
    'not found',
    'does not exist',
    'command not found',
)
"""Case-insensitive substrings; matched against any record's flattened text."""

_SELF_CORRECT_PATTERNS: tuple[str, ...] = (
    'actually,',
    'wait,',
    'let me reconsider',
    "that's wrong",
    'my mistake',
)
"""Case-insensitive substrings; matched ONLY against assistant text blocks
(native-carrier scoping) — never a tool_result or a tool_use input, so an
agent authoring test data containing one of these phrases is never
mis-scored as a real self-correction."""

_DF_GUARD_TEXT_PATTERNS: tuple[str, ...] = (
    'phantom-done',
    'phantom_done',
    'premise-lint',
    'premise_lint',
    'false premise',
    'guard trip',
    'scope_violation',
    'done_gate_missing_files',
)
"""Case-insensitive substrings drawn from real dark_factory guard/refusal
vocabulary: the phantom-done gate (orchestrator/scheduler.py,
fused-memory/middleware/task_interceptor.py), the premise-lint guard
(fused-memory/reconciliation/recon_self_model.py,
fused-memory/middleware/premise_lint_guard.py), and escalation
categorization (escalation/server.py)."""

_DF_GUARD_TOOL_NAMES: frozenset[str] = frozenset({
    'mcp__plan-tools__report_false_premise',
    'mcp__plan-tools__report_unactionable_task',
    'mcp__plan-tools__report_task_already_done',
    'mcp__plan-tools__report_blocking_dependency',
})
"""The real plan-tools guard/refusal tool names
(orchestrator/src/orchestrator/mcp/plan_tools.py) — a structural
tool_use.name match, never a text/substring guess."""

_INTERRUPT_PATTERNS: tuple[str, ...] = ('interrupted by user',)


def _message_content(record: dict[str, Any]) -> Any:
    message = record.get('message')
    if not isinstance(message, dict):
        return None
    return message.get('content')


def _record_text(record: dict[str, Any]) -> str:
    """Flatten a record's message.content into one lowercased text blob.

    ``content`` is either a plain string or a list of content blocks; each
    block contributes its 'text' field (assistant/user text, thinking) and
    its 'content' field (tool_result output, itself sometimes a string).
    Used for the plain substring-scan classes (not_found, df_guard text
    patterns, interrupt) — never for self_correct, which is scoped to
    assistant text blocks only via :func:`_assistant_text`.
    """
    content = _message_content(record)
    if isinstance(content, str):
        return content.lower()
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get('text')
            if isinstance(text, str):
                parts.append(text)
            block_content = block.get('content')
            if isinstance(block_content, str):
                parts.append(block_content)
        return '\n'.join(parts).lower()
    return ''


def _assistant_text(record: dict[str, Any]) -> str:
    """Flatten only assistant 'text' content blocks (native-carrier scoping)."""
    if record.get('type') != 'assistant':
        return ''
    content = _message_content(record)
    if not isinstance(content, list):
        return ''
    parts = [
        block.get('text') for block in content
        if isinstance(block, dict) and block.get('type') == 'text'
        and isinstance(block.get('text'), str)
    ]
    return '\n'.join(parts).lower()


def _has_tool_error(record: dict[str, Any]) -> bool:
    """True iff *record* is a user record carrying a tool_result with a truthy is_error.

    Structural only — never a text/"FAIL" substring match (mirrors the
    decoy-FAIL suppression decision applied to task α's digest.py)."""
    if record.get('type') != 'user':
        return False
    content = _message_content(record)
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get('type') == 'tool_result' and block.get('is_error')
        for block in content
    )


def _has_guard_tool_use(record: dict[str, Any]) -> bool:
    """True iff *record* is an assistant record invoking a known guard/refusal tool."""
    if record.get('type') != 'assistant':
        return False
    content = _message_content(record)
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get('type') == 'tool_use'
        and block.get('name') in _DF_GUARD_TOOL_NAMES
        for block in content
    )


@dataclass(frozen=True)
class SignalCounts:
    """Per-class confusion-signal counts for one session (PRD §7.2 signal_counts)."""

    tool_error: int = 0
    not_found: int = 0
    self_correct: int = 0
    df_guard: int = 0
    interrupt: int = 0

    @property
    def total_signal(self) -> int:
        return (
            self.tool_error + self.not_found + self.self_correct
            + self.df_guard + self.interrupt
        )


def score_signals(path: Path) -> SignalCounts:
    """Zero-token single pass over a transcript, counting the 5 confusion-signal classes.

    One increment per class per RECORD (not per pattern match) — a record
    carrying multiple synonymous markers for the same class still counts
    once. ``tool_error`` and the tool-use half of ``df_guard`` are
    structural checks; every other class is a plain case-insensitive
    substring scan. Malformed/unreadable input degrades to an all-zero
    :class:`SignalCounts` rather than raising.
    """
    tool_error = not_found = self_correct = df_guard = interrupt = 0
    try:
        for record in _iter_json_lines(path):
            if _has_tool_error(record):
                tool_error += 1

            text = _record_text(record)
            if text and any(pattern in text for pattern in _NOT_FOUND_PATTERNS):
                not_found += 1
            if text and any(pattern in text for pattern in _INTERRUPT_PATTERNS):
                interrupt += 1

            assistant_text = _assistant_text(record)
            if assistant_text and any(
                pattern in assistant_text for pattern in _SELF_CORRECT_PATTERNS
            ):
                self_correct += 1

            has_guard_text = bool(text) and any(
                pattern in text for pattern in _DF_GUARD_TEXT_PATTERNS
            )
            if _has_guard_tool_use(record) or has_guard_text:
                df_guard += 1
    except OSError:
        pass
    return SignalCounts(
        tool_error=tool_error,
        not_found=not_found,
        self_correct=self_correct,
        df_guard=df_guard,
        interrupt=interrupt,
    )


# ---------------------------------------------------------------------------
# classify_agent_class — 5-stratum classifier (PRD §5.2 point 2)
# ---------------------------------------------------------------------------

STRATA: tuple[str, ...] = (
    'recon', 'curator-classifier', 'watcher', 'orchestrated-task', 'interactive',
)
"""The 5 agent-class strata the sampler groups by."""

_WORKTREE_DIR_MARKERS: tuple[str, ...] = ('--worktrees-', '--claude-worktrees-')

_RECON_HEADER_PREFIXES: tuple[str, ...] = (
    '## Reconciliation Run',
    '## Stage 2: Task-Knowledge Sync',
)
_RECON_SUBSTRING_MARKERS: tuple[str, ...] = ('memory_consolidator',)

_WATCHER_SUBSTRING_MARKERS: tuple[str, ...] = ('recon-escalation-watcher',)
"""skills/recon-escalation-watcher — a real dark_factory skill/slash-command."""

_CURATOR_CLASSIFIER_SUBSTRING_MARKERS: tuple[str, ...] = ('review suggestion classifier',)
"""Verbatim opening of TRIAGE_SYSTEM_PROMPT
(orchestrator/src/orchestrator/agents/triage.py:131), matched
case-insensitively."""


def _first_user_turn_text(record: dict[str, Any] | None) -> str:
    """Flatten a first-non-sidechain-user-turn record's text content to a string.

    *record* is the RAW transcript-line dict the caller already located
    (this function does no file I/O); returns '' when *record* is None
    (no such turn found) or carries no text content.
    """
    if record is None:
        return ''
    content = _message_content(record)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get('text') for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
            and isinstance(block.get('text'), str)
        ]
        return '\n'.join(parts)
    return ''


def classify_agent_class(record: dict[str, Any] | None, path: Path) -> str:
    """Classify a session's agent class (one of :data:`STRATA`).

    Priority: *path*'s encoded-dir SHAPE is checked first — only its
    parent directory's name is inspected, never any file content — a
    ``--worktrees-``/``--claude-worktrees-`` marker means
    ``'orchestrated-task'`` and wins even over a recon-marker turn.
    Otherwise, cheap content markers in *record* (the session's
    already-located first non-sidechain, non-meta user turn) decide:
    recon header / ``memory_consolidator`` -> ``'recon'``;
    ``recon-escalation-watcher`` -> ``'watcher'``; the triage classifier's
    system-prompt opening -> ``'curator-classifier'``. Falls back to
    ``'interactive'`` — a main-dir freeform human turn, or no matching
    user turn at all (*record* is None).
    """
    encoded_dir = path.parent.name
    if any(marker in encoded_dir for marker in _WORKTREE_DIR_MARKERS):
        return 'orchestrated-task'

    text = _first_user_turn_text(record)
    if text.startswith(_RECON_HEADER_PREFIXES) or any(
        marker in text for marker in _RECON_SUBSTRING_MARKERS
    ):
        return 'recon'
    if any(marker in text for marker in _WATCHER_SUBSTRING_MARKERS):
        return 'watcher'
    if any(marker in text.lower() for marker in _CURATOR_CLASSIFIER_SUBSTRING_MARKERS):
        return 'curator-classifier'
    return 'interactive'
