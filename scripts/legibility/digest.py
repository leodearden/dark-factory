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

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Self-bootstrap for standalone `python scripts/legibility/digest.py` runs —
# must run BEFORE the `legibility.*` import below, since a direct script
# invocation puts only scripts/legibility/ (not scripts/) on sys.path.
# Skipped under pytest/normal package import: __name__ is 'legibility.digest'.
# Verbatim the sampling.py:43-48 / census.py / nightly.py guard.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The ONE import this module takes from a sibling, and a deliberate narrowing
# of the "self-contained" claim inventory.py's own docstring makes about the
# alpha/beta split: these two names are the shared definition of "this
# transcript is unreadable", and the alternative to importing them is a second
# copy of that answer in each reader — the duplication this replaces.
from legibility.inventory import (  # noqa: E402
    UNREADABLE_FILE_ERRORS,
    as_unreadable_file_error,
)

logger = logging.getLogger('legibility.digest')


def load_transcript(path: Any) -> list[dict[str, Any]]:
    """Parse a transcript JSONL file into an ordered list of record dicts.

    Degradation splits at the file/line boundary, and that split is a
    contract rather than an implementation detail. A corrupt LINE degrades
    silently: blank lines and lines that fail to parse as JSON are skipped,
    because fire-and-forget transcript writers can leave a truncated or
    corrupt trailing line and one bad line must not abort the whole read
    (mirrors analyze_speculation_depth.load_events). An unreadable FILE
    raises ``OSError``, so a caller reporting coverage counts it once
    instead of aborting its walk.

    Making the ``OSError`` half true takes explicit normalization: the
    decode shape is a ``ValueError`` subclass, not an ``OSError``. This
    reader does NOT answer that
    question itself — it funnels through
    ``inventory.as_unreadable_file_error``, the one place that enumerates the
    shapes and explains why each is expected, which is also what
    ``inventory.iter_json_lines`` (the streaming sibling cited above) uses.
    The two readers are therefore interchangeable at this boundary by
    construction rather than by two copies kept in sync;
    ``test_legibility_digest.TestLoadTranscriptCorruptionShapes`` asserts the
    agreement directly, including that they share the one implementation.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        # Wrap only the read iteration — decoding happens lazily HERE, per
        # chunk, not at open() — leaving the JSONDecodeError skip inside the
        # loop so the file-level vs line-level split above is preserved.
        # Catch tuple and wrapping are BOTH the shared inventory
        # helpers, not local copies: that is what makes this reader and
        # iter_json_lines agree structurally instead of by two edits landing
        # together.
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except UNREADABLE_FILE_ERRORS as exc:
            raise as_unreadable_file_error(exc) from exc
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
        # The `isinstance(..., str)` guard already excludes None and non-str at
        # runtime, but binding it with a walrus is what carries that narrowing
        # into the element expression — pyright cannot narrow an unbound
        # `.get()` CALL across the guard, so the un-bound spelling inferred
        # list[Unknown | None] and rejected the join. Same iteration, same
        # filter, same values.
        texts = [
            block_text for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
            and isinstance(block_text := block.get('text'), str)
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
            block_text for block in content
            if isinstance(block, dict) and isinstance(block_text := block.get('text'), str)
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


EXIT_CODE_PATTERN = re.compile(
    r'exit\s*=\s*(\d+)|exit\s+code\s+(\d+)|exit\s+status\s+(\d+)',
    re.IGNORECASE,
)
"""The three exit-code spellings observed in tool_result content: the
machine-readable ``exit=<rc>`` form emitted by scripts/watcher-rearm.sh
(:228-237) and the two prose forms ('exit code N', 'exit status N') shells
and harness wrappers produce. Case-insensitive, like every other pattern in
this module."""

BOUNDED_WAIT_EXIT_CODE = 124
"""``timeout(1)``'s ceiling exit -- the status it returns when the command
was still running when the time limit expired (as opposed to the command's
own status). In this repo a 124 is overwhelmingly produced by a deliberately
bounded poll or wait, not by a broken command."""


def _extract_exit_code(text: str) -> int | None:
    """Return the FIRST exit code mentioned in *text*, or None.

    First-match-wins is deliberate: a tool_result that reports several codes
    leads with the outcome of the invocation being reported (the watcher's
    own ``exit=<rc>`` marker precedes any retry chatter), so the leading code
    is the one that classifies the result.
    """
    match = EXIT_CODE_PATTERN.search(text)
    if match is None:
        return None
    return int(next(g for g in match.groups() if g is not None))


DESIGNED_OUTCOME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r'WATCHER_REARM_OUTCOME:\s*(CEILING|FIRED)\b', re.IGNORECASE),
        'watcher-rearm-declared',
    ),
)
"""Table of (compiled regex, label) pairs recognising a self-DECLARED
designed outcome, so a future contract is a one-line addition.

Seeded with scripts/watcher-rearm.sh's marker (:228-237), which emits four
outcomes and means different things by them: FIRED (exit=0, :228) and
CEILING (exit=124, :231) are designed loop-continuation outcomes, while
KILLED (137|143|144, :234) and ERROR (the catch-all, :237) are REAL
failures. Only the first two are matched here -- deliberately. Suppressing
all four because they share a marker token would trade one over-count for
an under-count and hide genuine watcher breakage, the opposite of the
07-31 census cluster 1.3 finding's intent.

Two traps documented in that script's header (:52-73) bound what this
table can be relied on to mean:
  * usage/guard failures (exit 2 -- missing DARK_FACTORY_ROOT, unresolved
    queue dir, bad flag) return BEFORE the watcher is ever invoked and emit
    NO marker at all (:59-63), so absence of a marker is NOT absence of
    failure -- which is why a bare exit code still classifies below.
  * a SIGTERM delivered to the watcher is converted to a clean exit 0 by
    its own handler, surfacing as ``FIRED exit=0`` with EMPTY stdout
    (:66-73). It is still a designed outcome by the shell contract; a
    caller needing to know whether an escalation actually fired must check
    stdout, which is beyond what a digest signal tally can see.
"""


def classify_designed_outcome(content: str, exit_code: int | None) -> str | None:
    """Return a label when a structured error is a DESIGNED outcome, else None.

    Two tiers, most authoritative first:
      1. a machine-readable self-declaration matching DESIGNED_OUTCOME_PATTERNS;
      2. a bare BOUNDED_WAIT_EXIT_CODE with no declaration at all.
    The label differs per tier so a digest reader can tell which rule fired.

    Tier 2 is a deliberate precision/recall tradeoff, and a deliberately
    cheap one to be wrong about: misclassifying a genuine un-designed
    timeout as designed costs exactly one weight-1.0 noise signal
    (tool_error is joint-lowest in SIGNAL_WEIGHTS) and never a gold user
    turn. Tightening to declaration-only, should a future census show real
    suppressions, is deleting the tier-2 branch -- the table above is what
    keeps the decision reversible rather than diffuse.
    """
    for pattern, label in DESIGNED_OUTCOME_PATTERNS:
        if pattern.search(content):
            return label
    if exit_code == BOUNDED_WAIT_EXIT_CODE:
        return 'bounded-wait-ceiling'
    return None


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
    """Detect the injected interrupt marker in non-sidechain user turns.

    A same-line ``# decoy-fail`` sentinel suppresses an otherwise-matching
    line, for parity with the other text-pattern detectors (not_found,
    df_guard, self_correct) and PRD Sec 13.2's decoy-FAIL suppression
    contract.
    """
    hits = []
    for index, record in enumerate(records):
        if record.get('type') != 'user' or record.get('isSidechain'):
            continue
        text = _user_turn_text(_message_content(record))
        if not text:
            continue
        if INTERRUPT_PATTERN in _strip_decoy_lines(text).lower():
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
    # `str | None` in the key, not `str`: _iter_tool_use_blocks selects on
    # `type == 'tool_use'` only, so a malformed block with no 'name' yields a
    # None here and always has. The declared type is corrected to match rather
    # than the value coerced, which would change what a nameless block renders
    # as in the report for no benefit.
    groups: dict[tuple[str | None, str], list[int]] = {}
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
f'Worktree: {worktree}\\n'). Both markers must co-occur (matched via
all(), not any()) since they are the two lines of a single injected
preamble -- a lone incidental mention of one marker (e.g. bare 'Worktree:'
prose in an ordinary interactive session) must not alone trigger this
classification."""

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

HARNESS_BRIEFING_HEADINGS: tuple[str, ...] = (
    '# context', '## context', '## agent identity', '# task',
)
"""ANCHOR heading literals for an injected orchestrator briefing
(orchestrator/src/orchestrator/agents/briefing.py): ``_get_memory_context``
emits '# Context' (:1118) and, on its memory-unavailable exception path, a
'## Context' sub-block (:1113); ``_agent_identity`` emits '## Agent
Identity' (:98); the role prompt templates emit '# Task' (:188/261/1024).
At least one anchor must be present as a line-anchored heading for a turn
to classify as briefing-injected -- see :func:`is_harness_injected_turn`
for the corroboration rule that goes with it."""

HARNESS_BRIEFING_SUBHEADINGS: tuple[str, ...] = (
    '## project context', '## conventions', '## recent decisions',
    '## task context', '# action',
)
"""CORROBORATING heading literals -- the structural sub-blocks a real
briefing carries alongside an anchor
(orchestrator/src/orchestrator/agents/briefing.py: '## Project Context'
:1091, '## Conventions' :1096, '## Recent Decisions' :1101, '## Task
Context' :1109 inside ``_get_memory_context``'s sections list; '# Action'
in the role prompt templates at :193/:496/:1038). Never sufficient alone:
a human turn headed '## Conventions' carries no anchor and stays gold."""

HARNESS_PROMPT_MARKERS: tuple[str, ...] = (
    'you are the trickle coder for the dark-factory agent-confusion codebook',
)
"""Harness-authored PROSE preamble literals, matched as plain
case-insensitive substrings (unlike the line-anchored heading markers
above): the trickle coder's system prompt
(scripts/legibility/coder.py:174 ``build_prompt``). Extend with future
harness prompt literals as one-line additions."""


def is_harness_injected_turn(text: str) -> bool:
    """True when *text* is harness-injected rather than genuine human-typed
    input: either the orchestrator's briefing preamble or a harness prose
    preamble (any HARNESS_PROMPT_MARKERS substring).

    The briefing rule is ANCHOR + CORROBORATOR: at least one
    HARNESS_BRIEFING_HEADINGS anchor must appear as its own stripped line,
    AND at least two DISTINCT briefing headings (anchors union
    HARNESS_BRIEFING_SUBHEADINGS) must be present in total.

    This replaces the original all-of-three rule ('# Context' AND '## Agent
    Identity' AND '# Task'), which the 07-31 confusion census showed was too
    strict to catch the shape actually injected most often: cluster 1.1(b)
    (plans/confusion-census-2026-07-31.md:79/:85) sighted a context-ONLY
    injection -- '# Context' followed by '## Project Context' and the
    memory-search JSON dump, with neither '## Agent Identity' nor '# Task'
    present -- passing the filter BY CONSTRUCTION and being rendered as a
    fabricated gold "User Correction" in four dispatched-task digests.

    Both false-positive guards the original all() was written to protect
    still hold, for reasons the relaxation preserves rather than tolerates:
    a genuine human turn opening with a lone '# Task' has one heading and
    NO corroborator, so it falls below the >=2 threshold; and a '# Context'
    quoted mid-prose is not a stripped line at all, so it is never a
    heading -- matching stays line-anchored and never substring.
    """
    lowered = text.lower()
    lines = {line.strip() for line in lowered.splitlines()}
    anchors = [h for h in HARNESS_BRIEFING_HEADINGS if h in lines]
    if anchors:
        corroborators = [h for h in HARNESS_BRIEFING_SUBHEADINGS if h in lines]
        if len(anchors) + len(corroborators) >= 2:
            return True
    return any(marker in lowered for marker in HARNESS_PROMPT_MARKERS)


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

    if all(marker in haystack for marker in ORCHESTRATED_TASK_MARKERS):
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
    isMeta=True (system-injected) turns, user records whose content is
    entirely tool_result blocks, and harness-injected briefing/prompt turns
    (see :func:`is_harness_injected_turn`) -- these are typed into the
    transcript as ordinary user-role text (isMeta=False), so isMeta alone
    cannot exclude them. This function is the SINGLE source for both the
    gold user_corrections section and render_digest's n_user_turns score
    component, so this one filter excludes a harness-injected turn from
    both. User corrections are gold (PRD Sec 5) -- this is the
    highest-priority digest section.
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
        if is_harness_injected_turn(text):
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


def _first_present(records: list[dict[str, Any]], key: str) -> Any:
    """Return the first truthy top-level *key* value across *records*.

    Real transcripts carry ``cwd``/``sessionId``/``timestamp`` on every
    non-queue-operation record (confirmed against a live
    ``~/.claude/projects/<enc>/*.jsonl`` file); scanning for the first
    record that has the field makes this robust to a leading
    queue-operation/last-prompt record that lacks it.
    """
    for record in records:
        value = record.get(key)
        if value:
            return value
    return None


def _derive_session(records: list[dict[str, Any]]) -> str:
    return _first_present(records, 'sessionId') or 'unknown'


def _derive_cwd(records: list[dict[str, Any]]) -> str:
    return _first_present(records, 'cwd') or 'unknown'


def _derive_date(records: list[dict[str, Any]]) -> str:
    """Return just the ISO date component of the first record timestamp."""
    timestamp = _first_present(records, 'timestamp')
    if not isinstance(timestamp, str) or not timestamp:
        return 'unknown'
    return timestamp.split('T', 1)[0]


def _encode_cwd(cwd: str) -> str:
    """Mirror Claude Code's own ``~/.claude/projects/<enc>`` encoding.

    '/', '.' AND '_' all map to '-', and case is preserved — same rule as
    orchestrator.session_registry.encode_cwd, the canonical implementation,
    which carries the authoritative statement of the rule and the record of
    the 738 real (encoded-dir, decoded-cwd) pairs it was validated against.
    Reused here as the fallback when the real transcript path isn't
    available to read the ground-truth encoded dir name off disk.

    Agreement with the canonical is not left to goodwill: it is asserted
    row-for-row against real on-disk dir names by
    ``scripts/tests/test_legibility_inventory.py``'s ``TestEncoderLockstep``
    (task 3272, which found this copy and three others all missing '_').
    """
    return cwd.replace('/', '-').replace('.', '-').replace('_', '-')


def _derive_encoded_dir(records: list[dict[str, Any]], cwd: str, path: Any) -> str:
    """Prefer the real transcript file's parent dir name (ground truth);
    fall back to mirror-encoding the derived *cwd* when no *path* is given
    or *cwd* itself could not be derived."""
    if path is not None:
        parent_name = Path(path).parent.name
        if parent_name:
            return parent_name
    if cwd == 'unknown':
        return 'unknown'
    return _encode_cwd(cwd)


SECTION_HEADINGS: dict[str, str] = {
    'user_corrections': 'User Corrections',
    'error_neighborhoods': 'Error Neighborhoods',
    'self_corrections': 'Self-Corrections',
    'retry_loops': 'Retry Loops',
    'not_found': 'Not Found',
    'df_guard': 'Guard Trips',
    'interrupt': 'Interrupts',
}
"""Markdown '## ' heading text per digest section. The first four are the
PRD Sec 7.2 primary sections (verbatim prose order: user turns, error
neighborhoods, self-corrections, retry loops); the last three are the
secondary scalar signals, included only when present (PRD decomposition
Sec 11 alpha observable: "frontmatter + the four signal-class sections")."""

SECTION_PRIORITY: tuple[str, ...] = (
    'retry_loops', 'not_found', 'error_neighborhoods', 'df_guard', 'interrupt',
    'self_corrections', 'user_corrections',
)
"""Section keys in ASCENDING signal priority -- index 0 is dropped/trimmed
FIRST under the 15KB soft cap, the last entry ('user_corrections', gold)
is trimmed LAST (PRD Sec 7.2: "truncate lowest-signal sections last"; PRD
Sec 5: user corrections are gold). Digest bodies render in the REVERSE of
this order (gold first). Ranking mirrors SIGNAL_WEIGHTS magnitude
(self_correct 3.0 > df_guard/interrupt 2.0 > tool_error/not_found 1.0),
with retry_loops lowest since it carries no SIGNAL_WEIGHTS entry at all
(it is a structural section, not one of the 5 scored signal classes)."""


def _render_user_corrections(items: list[dict[str, Any]]) -> list[str]:
    return [f"- (turn {item['index']}) {item['text']}" for item in items]


def _render_error_neighborhoods(items: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in items:
        tool = item['attempt_tool'] or 'unknown'
        summary = item['attempt_input_summary'] or ''
        lines.append(
            f"- (turn {item['index']}) {tool}({summary}) -> {item['error_content']}"
        )
    return lines


def _render_self_corrections(items: list[dict[str, Any]]) -> list[str]:
    return [
        f"- (turn {item['index']}) [{item['pattern']}] {item['context']}"
        for item in items
    ]


def _render_retry_loops(items: list[dict[str, Any]]) -> list[str]:
    return [
        f"- {item['tool']} x{item['count']}: {item['signature']}" for item in items
    ]


def _render_scalar_signal(items: list[dict[str, Any]]) -> list[str]:
    return [f"- (turn {item['index']}) {item['pattern']}" for item in items]


_SECTION_RENDERERS: dict[str, Any] = {
    'user_corrections': (iter_user_turns, _render_user_corrections),
    'error_neighborhoods': (iter_error_neighborhoods, _render_error_neighborhoods),
    'self_corrections': (iter_self_corrections, _render_self_corrections),
    'retry_loops': (find_retry_loops, _render_retry_loops),
    'not_found': (iter_not_found, _render_scalar_signal),
    'df_guard': (iter_df_guards, _render_scalar_signal),
    'interrupt': (iter_interrupts, _render_scalar_signal),
}
"""(detector, item-renderer) pair per section key, keyed identically to
SECTION_HEADINGS/SECTION_PRIORITY."""

ITEM_TRUNCATION_MARKER = '... [item truncated]'
"""Appended to a per-item line that exceeded its byte cap. Explicit and
ASCII (so its own byte length equals its character length), keeping
degradation legible rather than silently lossy."""

MAX_ITEM_BYTES = 2048
"""Hard ceiling on a single rendered item's UTF-8 byte length, regardless
of how large *max_bytes* is."""

MIN_ITEM_BYTES = 256
"""Floor on a single rendered item's cap, even for a very small
*max_bytes* -- an item capped below this would be truncated into
unreadable noise."""

FRONTMATTER_RESERVE_BYTES = 512
"""Bytes reserved for the frontmatter block plus a '## ' section heading
when deriving the per-item cap from *max_bytes* (see
:func:`_item_byte_cap`). Measured frontmatter is ~276-310 bytes;
reserving 512 leaves headroom for a heading line too, so "frontmatter +
heading + one capped item" always fits under any realistic max_bytes --
this is what makes the R2 empty-body pathology structurally unreachable
rather than merely checked (:func:`_warn_if_body_evicted` is the
belt-and-braces backstop for the residual edge cases, e.g. a pathologically
tiny max_bytes)."""


def _item_byte_cap(max_bytes: int) -> int:
    """Derive the per-item byte cap from the digest's overall *max_bytes*
    budget: never above MAX_ITEM_BYTES, never below MIN_ITEM_BYTES,
    otherwise *max_bytes* minus headroom for the frontmatter + heading."""
    return max(MIN_ITEM_BYTES, min(MAX_ITEM_BYTES, max_bytes - FRONTMATTER_RESERVE_BYTES))


def _cap_item(line: str, cap: int) -> str:
    """Return *line* unchanged when its UTF-8 byte length is within *cap*;
    otherwise byte-truncate (never splitting a multi-byte codepoint) and
    append :data:`ITEM_TRUNCATION_MARKER`.

    Byte-wise because the whole digest budget is measured in UTF-8 bytes
    (:func:`_resolve_size_bytes`); ``decode(..., 'ignore')`` drops a
    partial trailing codepoint instead of emitting a U+FFFD replacement
    character, so a truncated item never introduces mojibake into text a
    downstream LLM coder reads verbatim.
    """
    encoded = line.encode('utf-8')
    if len(encoded) <= cap:
        return line
    keep = max(0, cap - len(ITEM_TRUNCATION_MARKER))
    return encoded[:keep].decode('utf-8', 'ignore') + ITEM_TRUNCATION_MARKER


def _build_sections(
    records: list[dict[str, Any]], *, item_max_bytes: int = _item_byte_cap(15360),
) -> dict[str, list[str]]:
    """Run every detector once and render its hits to markdown bullet
    lines, keyed by section key, capping each rendered line to
    *item_max_bytes* (see :func:`_cap_item`) at this single choke point --
    applied uniformly across all seven sections so ONE oversized item
    (e.g. a multi-KB pasted user turn, or a huge echoed tool_result) can
    never evict every sibling section during :func:`_truncate_sections`'
    trim loop. A section with zero hits maps to an empty list --
    render_digest skips emitting a heading for it."""
    sections = {}
    for key, (detector, renderer) in _SECTION_RENDERERS.items():
        sections[key] = [
            _cap_item(line, item_max_bytes) for line in renderer(detector(records))
        ]
    return sections


def _render_body(sections: dict[str, list[str]]) -> str:
    """Render non-empty sections, gold (user_corrections) first, retry_loops
    last -- the REVERSE of the ascending-priority SECTION_PRIORITY order."""
    blocks = []
    for key in reversed(SECTION_PRIORITY):
        lines = sections.get(key) or []
        if not lines:
            continue
        blocks.append('## ' + SECTION_HEADINGS[key] + '\n' + '\n'.join(lines))
    return '\n\n'.join(blocks)


def _resolve_size_bytes(meta: dict[str, Any], body: str) -> str:
    """Render *meta* (sans a final size_bytes) + *body*, resolving
    ``size_bytes`` to the FINAL digest's own UTF-8 byte length via a short
    fixed point.

    size_bytes is self-referential -- embedding the correct value can
    change the digest's own length -- so an initial guess (0) is rendered,
    measured, and re-rendered. Only the field's digit-count can change
    between passes (everything else is fixed), so this converges within a
    couple of iterations for any realistic digest; the loop bound is a
    defensive cap, not a claim that more iterations are ever needed.
    """
    meta = dict(meta)
    meta['size_bytes'] = 0
    digest = render_frontmatter(meta) + '\n' + body
    for _ in range(4):
        size = len(digest.encode('utf-8'))
        if size == meta['size_bytes']:
            return digest
        meta['size_bytes'] = size
        digest = render_frontmatter(meta) + '\n' + body
    return digest


def _truncate_sections(
    meta: dict[str, Any], sections: dict[str, list[str]], max_bytes: int,
) -> tuple[str, dict[str, list[str]]]:
    """Trim *sections* in ASCENDING SECTION_PRIORITY order (lowest-signal
    first) until the fully-rendered digest fits within *max_bytes*, or
    there is nothing left to trim -- PRD Sec 7.2 "truncate lowest-signal
    sections last" (equivalently: trim lowest-signal sections FIRST).

    Each iteration drops exactly one TRAILING item from the current
    lowest-priority non-empty section (an emptied section stops being
    rendered at all -- its heading disappears along with it -- and the
    next-lowest-priority non-empty section becomes the new trim target),
    then re-measures via the :func:`_resolve_size_bytes` fixed point.
    Trimming only ever shrinks the digest, so this always terminates: in
    the worst case every item in every section is removed, at which point
    the loop stops even if the (now section-free) digest is still over
    the cap -- a soft cap can't shrink the frontmatter itself.

    Returns the rendered digest AND the post-trim section dict, so a
    caller can inspect final section state (e.g. render_digest's
    :func:`_warn_if_body_evicted` consistency guard) without re-parsing
    the rendered markdown body.
    """
    sections = {key: list(lines) for key, lines in sections.items()}
    digest = _resolve_size_bytes(meta, _render_body(sections))

    while len(digest.encode('utf-8')) > max_bytes:
        target_key = next((key for key in SECTION_PRIORITY if sections.get(key)), None)
        if target_key is None:
            break
        sections[target_key].pop()
        digest = _resolve_size_bytes(meta, _render_body(sections))

    return digest, sections


def _warn_if_body_evicted(
    meta: dict[str, Any], sections: dict[str, list[str]], max_bytes: int,
) -> None:
    """Emit-time consistency guard: nonzero ``signal_counts`` implies a
    non-empty body. :func:`_item_byte_cap`'s per-item cap makes a
    violation structurally unreachable for any realistic *max_bytes* (see
    FRONTMATTER_RESERVE_BYTES) -- this is the belt-and-braces backstop for
    the residual edge cases (e.g. a pathologically tiny *max_bytes*).

    Never raises: a soft-cap edge case must not turn into a per-session
    failure for census.py/nightly.py callers. Never silent either -- a
    violation is logged LOUDLY, naming the session, the nonzero counts and
    *max_bytes*, so the degenerate digest is never mistaken for a
    genuinely signal-free session.
    """
    counts = meta['signal_counts']
    if not any(counts.values()):
        return
    if any(sections.values()):
        return
    nonzero = {key: value for key, value in counts.items() if value}
    logger.warning(
        'digest body fully evicted despite nonzero signal_counts: '
        'session=%s signal_counts=%s max_bytes=%d',
        meta['session'], nonzero, max_bytes,
    )


def render_digest(
    records: list[dict[str, Any]],
    *,
    agent_class: str,
    path: Any = None,
    max_bytes: int = 15360,
) -> str:
    """Render the full markdown digest: frontmatter + one section per
    non-empty signal class (PRD Sec 7.2), soft-capped at *max_bytes*.

    *agent_class* is the caller's already-resolved class (typically
    :func:`classify_agent_class`'s result) -- render_digest never
    classifies on its own. *path*, when given, sources ``encoded_dir``
    from the transcript file's real parent dir name; otherwise it falls
    back to mirror-encoding the derived ``cwd`` (see
    :func:`_derive_encoded_dir`). When the unrestricted render exceeds
    *max_bytes*, :func:`_truncate_sections` trims lowest-signal sections
    first (gold user-corrections last); a digest already under the cap is
    returned unmodified (zero trim iterations).
    """
    cwd = _derive_cwd(records)
    counts = signal_counts(records)
    meta = {
        'session': _derive_session(records),
        'cwd': cwd,
        'encoded_dir': _derive_encoded_dir(records, cwd, path),
        'agent_class': agent_class,
        'date': _derive_date(records),
        'score': score_signals(counts, len(iter_user_turns(records))),
        'signal_counts': counts,
    }

    sections = _build_sections(records, item_max_bytes=_item_byte_cap(max_bytes))
    digest, trimmed_sections = _truncate_sections(meta, sections, max_bytes)
    _warn_if_body_evicted(meta, trimmed_sections, max_bytes)
    return digest


def build_digest(
    path: Any, *, agent_class_override: str | None = None, max_bytes: int = 15360,
) -> str:
    """Orchestrate the full pipeline for a single transcript file: parse,
    classify, render. This is the headline entry point (PRD Sec 8.1
    boundary test, producer side) -- what beta/delta and :func:`main` call.

    *agent_class_override*, when given, is passed straight through to
    :func:`classify_agent_class` (e.g. beta's already-authoritative class);
    otherwise alpha's best-effort marker heuristics decide.
    """
    records = load_transcript(path)
    agent_class = classify_agent_class(records, override=agent_class_override)
    return render_digest(records, agent_class=agent_class, path=path, max_bytes=max_bytes)


def main(argv: Sequence[str]) -> int:
    """CLI entry point: build a digest for a single transcript file.

    Prints the rendered digest to stdout and returns 0, unless ``--out``
    is given, in which case the digest is written to that file instead
    (nothing is printed to stdout). A transcript path that does not exist
    is reported to stderr with a non-zero return, rather than letting the
    underlying ``open()`` raise -- this is the only handled error case;
    everything else (e.g. an unreadable/permission-denied file) surfaces
    as an uncaught exception, matching the tool's `scripts/` siblings.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Turn a Claude Code session transcript JSONL file into a '
            '5-15KB markdown confusion digest with YAML frontmatter.'
        )
    )
    parser.add_argument('transcript', help='Path to a Claude Code session transcript JSONL file')
    parser.add_argument(
        '--agent-class', dest='agent_class', default=None,
        help="Override alpha's best-effort agent_class classification",
    )
    parser.add_argument(
        '--max-bytes', type=int, default=15360,
        help='Soft byte cap for the rendered digest (default: %(default)s)',
    )
    parser.add_argument(
        '--out', default=None,
        help='Write the digest to this file instead of printing it to stdout',
    )
    args = parser.parse_args(argv)

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f'digest: transcript not found: {transcript_path}', file=sys.stderr)
        return 1

    digest = build_digest(
        transcript_path, agent_class_override=args.agent_class, max_bytes=args.max_bytes,
    )

    if args.out:
        Path(args.out).write_text(digest, encoding='utf-8')
    else:
        print(digest)

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
