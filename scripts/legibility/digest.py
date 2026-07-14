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


DECOY_MARKER = '# decoy-fail'
"""Inline same-line sentinel: a fixture/test author appends this to a line
to explicitly declare "this is not a real signal, don't count it", without
having to relocate the string out of its otherwise-native carrier. Matched
case-insensitively, like every other pattern in this module."""


def _strip_decoy_lines(text: str) -> str:
    """Return *text* with every line containing DECOY_MARKER removed.

    Applied before pattern matching in the text-pattern signal detectors
    (not_found, df_guard, self_correct) so a same-line decoy marker
    suppresses only that line's occurrence -- other lines in the same
    carrier are unaffected.
    """
    marker = DECOY_MARKER.lower()
    lines = [line for line in text.split('\n') if marker not in line.lower()]
    return '\n'.join(lines)


def _signal_text_sources(
    records: list[dict[str, Any]],
    *,
    tool_result: bool = False,
    assistant_text: bool = False,
    user_text: bool = False,
) -> list[tuple[int, str]]:
    """Yield (record_index, text) pairs from the requested NATIVE carriers
    only: tool_result content, assistant 'text' blocks, and non-sidechain
    user-turn text (incl. isMeta system injections). Carriers are opt-in
    per call, since each text-pattern detector is scoped to its own subset.

    NEVER includes the ``input`` of a Write/Edit/MultiEdit/NotebookEdit (or
    any) tool_use block -- an assistant-authored file mutation is not a
    native signal carrier, so a decoy string planted there is structurally
    excluded rather than merely filtered (the core of the decoy-FAIL
    suppression decision, PRD Sec 13.2).
    """
    sources: list[tuple[int, str]] = []
    if tool_result:
        for index, block in _iter_tool_result_blocks(records):
            sources.append((index, _content_to_text(block.get('content'))))
    if assistant_text:
        sources.extend(_assistant_text_blocks(records))
    if user_text:
        for index, record in enumerate(records):
            if record.get('type') == 'user' and not record.get('isSidechain'):
                text = _user_turn_text(_message_content(record))
                if text:
                    sources.append((index, text))
    return sources


def iter_self_corrections(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect curated self-correction markers in assistant TEXT blocks only.

    Native-carrier scoping: the same phrase inside a tool_result or inside
    a Write/Edit/MultiEdit/NotebookEdit tool_use input (an agent authoring
    test data, not a real correction) is never scanned -- restricting the
    scan to assistant 'text' blocks (see :func:`_assistant_text_blocks`)
    structurally excludes both. A same-line ``# decoy-fail`` sentinel
    suppresses an otherwise-matching line.
    """
    hits = []
    for index, text in _signal_text_sources(records, assistant_text=True):
        stripped = _strip_decoy_lines(text)
        lowered = stripped.lower()
        for pattern in SELF_CORRECTION_PATTERNS:
            pos = lowered.find(pattern)
            if pos == -1:
                continue
            hits.append({
                'index': index,
                'pattern': pattern,
                'context': _line_context(stripped, pos),
            })
    return hits


NOT_FOUND_PATTERNS: tuple[str, ...] = (
    'no such file or directory',
    'modulenotfounderror',
    'command not found',
    'does not exist',
    'not found in',
)
"""Matched case-insensitively against tool_result content only."""

DF_GUARD_PATTERNS: tuple[str, ...] = (
    'blocked:',
    'darkfactorypathscopeviolation',
    'done_gate_missing_files',
    'known-false premise about recon internals',
)
"""Real dark-factory guard TRIP literals -- never the bare, lowercase
category-name mention (e.g. ``category="scope_violation"``). Grounded in
the actual enforcement code: PathGuardVerdict's ``DarkFactoryPathScopeViolation``
error_type (fused_memory/middleware/path_scope_guard.py), the phantom-done
gate's ``done_gate_missing_files`` error code
(fused_memory/middleware/task_interceptor.py, orchestrator/scheduler.py),
and premise_lint_guard's fixed error-message prefix. ``scope_violation`` and
``phantom-done`` alone are common bare mentions (category labels, design-intent
prose/docstrings) and are deliberately NOT matched."""

INTERRUPT_PATTERN = 'request interrupted by user'
"""The literal Claude-Code-injected marker for a user-interrupted tool call
(e.g. "[Request interrupted by user for tool use]")."""


def iter_not_found(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect NOT_FOUND_PATTERNS in tool_result content only.

    A same-line ``# decoy-fail`` sentinel suppresses an otherwise-matching
    line (PRD Sec 13.2 decoy-FAIL suppression).
    """
    hits = []
    for index, text in _signal_text_sources(records, tool_result=True):
        lowered = _strip_decoy_lines(text).lower()
        for pattern in NOT_FOUND_PATTERNS:
            if pattern in lowered:
                hits.append({'index': index, 'pattern': pattern})
    return hits


def iter_df_guards(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect DF_GUARD_PATTERNS (real trip literals) in their native carriers:
    tool_result content, assistant text, and user-turn text (incl. isMeta
    system injections, excluding isSidechain subagent turns).

    A same-line ``# decoy-fail`` sentinel suppresses an otherwise-matching
    line (PRD Sec 13.2 decoy-FAIL suppression).
    """
    hits = []
    for index, text in _signal_text_sources(
        records, tool_result=True, assistant_text=True, user_text=True,
    ):
        lowered = _strip_decoy_lines(text).lower()
        for pattern in DF_GUARD_PATTERNS:
            if pattern in lowered:
                hits.append({'index': index, 'pattern': pattern})
    return hits


def iter_interrupts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect the injected interrupt marker in non-sidechain user turns."""
    hits = []
    for index, record in enumerate(records):
        if record.get('type') != 'user' or record.get('isSidechain'):
            continue
        text = _user_turn_text(_message_content(record))
        if text and INTERRUPT_PATTERN in text.lower():
            hits.append({'index': index, 'pattern': INTERRUPT_PATTERN})
    return hits


RETRY_MIN = 3
"""Minimum repeat count for a same-tool/same-input group to count as a
near-identical retry loop."""


def _input_signature(tool_input: Any) -> str:
    """Canonical, UNTRUNCATED string signature of a tool_use ``input`` dict.

    Used for retry-loop grouping -- unlike :func:`_summarize_input` (a
    size-capped display string), this must never be truncated: truncating
    could collapse two distinct long inputs onto the same prefix and
    produce a false-positive retry-loop group.
    """
    try:
        return json.dumps(tool_input, sort_keys=True)
    except TypeError:
        return str(tool_input)


def find_retry_loops(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group tool_use calls by (name, canonical input signature) and flag
    groups recurring >= RETRY_MIN times as near-identical retry loops.

    Deterministic and dependency-free (sibling to the decoy-FAIL decision,
    PRD Sec 13.2): no fuzzy string similarity, just "same tool, same
    canonical-JSON input, again".
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for index, block in _iter_tool_use_blocks(records):
        key = (block.get('name'), _input_signature(block.get('input')))
        groups.setdefault(key, []).append(index)

    loops = []
    for (name, signature), indices in groups.items():
        if len(indices) >= RETRY_MIN:
            loops.append({
                'tool': name,
                'signature': signature,
                'count': len(indices),
                'indices': indices,
            })
    return loops


def signal_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Assemble the 5-key signal tally required by the frontmatter contract
    (PRD Sec 7.2): ``{tool_error, self_correct, not_found, df_guard,
    interrupt}``. Each value is the hit count from the corresponding
    detector; an absent signal class reports 0 rather than being omitted.
    """
    return {
        'tool_error': len(iter_error_neighborhoods(records)),
        'self_correct': len(iter_self_corrections(records)),
        'not_found': len(iter_not_found(records)),
        'df_guard': len(iter_df_guards(records)),
        'interrupt': len(iter_interrupts(records)),
    }


SIGNAL_WEIGHTS: dict[str, float] = {
    'user_turn': 5.0,
    'self_correct': 3.0,
    'df_guard': 2.0,
    'interrupt': 2.0,
    'tool_error': 1.0,
    'not_found': 1.0,
}
"""Documented weights for :func:`score_signals`. Non-sidechain user turns
are gold (PRD Sec 5) and outweigh any single occurrence of any other
individual signal class. Self-corrections (an explicit "I was wrong"
moment) are the strongest non-gold signal; df_guard/interrupt are
mid-weight structural signals; tool_error/not_found are the lowest-weight,
highest-frequency noise signals. Every weight is strictly positive, which
is what makes score_signals monotonic."""


def score_signals(counts: dict[str, int], n_user_turns: int) -> float:
    """Weighted-sum confusion score used by beta's sampler to rank sessions.

    Strictly monotonic in every input: since every SIGNAL_WEIGHTS entry is
    positive, incrementing any one count (or n_user_turns) by 1 strictly
    raises the score.
    """
    score = SIGNAL_WEIGHTS['user_turn'] * n_user_turns
    for key, weight in SIGNAL_WEIGHTS.items():
        if key == 'user_turn':
            continue
        score += weight * counts.get(key, 0)
    return score


ORCHESTRATED_TASK_MARKERS: tuple[str, ...] = ('task id:', 'worktree:')
"""Injected task-briefing preamble literal
(orchestrator/src/orchestrator/dry_run_unblock.py: f'Task ID: {task_id}\\n'
f'Worktree: {worktree}\\n')."""

RECON_MARKERS: tuple[str, ...] = ('operating in sleep mode',)
"""Phrase shared by all three reconciliation stage system prompts
(fused_memory/reconciliation/prompts/stage{1,2,3}.py: Memory Consolidator /
Task-Knowledge Sync / Integrity Check agents are each described as
"operating in sleep mode")."""

WATCHER_MARKERS: tuple[str, ...] = ('escalation-watcher-auto',)
"""The canonical auto-watcher identity string
(escalation/src/escalation/authority.py: _WATCHER_AUTO_IDENTITY =
'orchestrator-escalation-watcher-auto')."""

CURATOR_CLASSIFIER_MARKERS: tuple[str, ...] = (
    'task curator for the dark-factory orchestrator',
    'code module classifier',
)
"""Literal system-prompt fragments for the task curator
(fused_memory/src/fused_memory/middleware/task_curator.py) and the
code-module classifier (orchestrator/src/orchestrator/harness.py)."""


def classify_agent_class(
    records: list[dict[str, Any]], override: str | None = None,
) -> str:
    """Best-effort agent-class classification from transcript markers.

    A caller-supplied *override* (e.g. the CLI ``--agent-class`` flag, or
    the authoritative class beta already computed) always wins, verbatim --
    alpha never guesses when the caller already knows. Otherwise: a
    genuinely empty transcript classifies as 'unknown'; a non-empty
    transcript with no marker match falls back to 'interactive'.
    """
    if override is not None:
        return override
    if not records:
        return 'unknown'

    haystack = '\n'.join(
        text for _, text in _signal_text_sources(
            records, tool_result=True, assistant_text=True, user_text=True,
        )
    ).lower()

    if any(marker in haystack for marker in ORCHESTRATED_TASK_MARKERS):
        return 'orchestrated-task'
    if any(marker in haystack for marker in RECON_MARKERS):
        return 'recon'
    if any(marker in haystack for marker in WATCHER_MARKERS):
        return 'watcher'
    if any(marker in haystack for marker in CURATOR_CLASSIFIER_MARKERS):
        return 'curator-classifier'
    return 'interactive'


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


def _yaml_dquote(value: Any) -> str:
    """Render *value* as a double-quoted YAML scalar.

    Frontmatter values are always explicitly quoted, never left as bare
    scalars: PyYAML's default (safe) resolver treats an unquoted ISO date
    (e.g. ``2026-07-14``) as a ``datetime.date`` object rather than a
    string, and similarly special-cases bare ``true``/``false``/``null``/
    numeric-looking scalars. Quoting sidesteps that whole implicit-typing
    surface so every string field round-trips as a plain str.
    """
    escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


FRONTMATTER_KEYS: tuple[str, ...] = (
    'session', 'cwd', 'encoded_dir', 'agent_class', 'date', 'size_bytes', 'score',
)
"""Top-level frontmatter keys in the exact PRD Sec 7.2 order (everything
before ``signal_counts``, which is rendered as its own nested block)."""

SIGNAL_COUNT_KEYS: tuple[str, ...] = (
    'tool_error', 'self_correct', 'not_found', 'df_guard', 'interrupt',
)
"""``signal_counts`` nested keys in the exact PRD Sec 7.2 order."""


def render_frontmatter(meta: dict[str, Any]) -> str:
    """Hand-render *meta* as a '---'-delimited YAML frontmatter block.

    Fixed key order (PRD Sec 7.2), explicit hand rendering rather than
    ``yaml.safe_dump`` -- deterministic, byte-stable output for exact tests
    and stable downstream diffs, and avoids a dumper's default surprises
    (key sorting, quoting, anchors, float formatting). String-valued fields
    are explicitly double-quoted (see :func:`_yaml_dquote`); numeric fields
    (size_bytes, score, and every signal_counts value) are emitted bare.
    """
    lines = ['---']
    for key in FRONTMATTER_KEYS:
        value = meta[key]
        if key in ('size_bytes', 'score'):
            lines.append(f'{key}: {value}')
        else:
            lines.append(f'{key}: {_yaml_dquote(value)}')
    lines.append('signal_counts:')
    counts = meta['signal_counts']
    for key in SIGNAL_COUNT_KEYS:
        lines.append(f'  {key}: {counts[key]}')
    lines.append('---')
    return '\n'.join(lines) + '\n'
