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
§5.2, contract §7.4). It owns its own signal-scoring primitives, reusing
only the low-level JSONL-line iterator
(:func:`legibility.inventory._iter_json_lines`) from its own sibling
module rather than duplicating it.

This module was originally "self-contained — does not import task α's
``digest.py``". That is now deliberately relaxed for exactly one reason:
``budgets.max_daily_digest_bytes`` is a DIGEST-OUTPUT budget, and charging
a record's real cost against it means knowing how big its digest will be,
which only the digest renderer can answer (:func:`digest_byte_cost_fn`).
The signal-scoring primitives remain unshared; the dependency is on
α's renderer alone, and it is one-directional (``digest.py`` imports
nothing from ``legibility.*``, so there is no cycle).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

# Self-bootstrap for standalone `python scripts/legibility/sampling.py` runs
# — must run BEFORE the `legibility.*` imports below, since a direct script
# invocation puts only scripts/legibility/ (not scripts/) on sys.path.
# Skipped under pytest/normal package import: __name__ is 'legibility.sampling'.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility import digest  # noqa: E402
from legibility.config import LegibilityConfig, load_config  # noqa: E402
from legibility.inventory import (  # noqa: E402
    SessionRecord,
    _iter_json_lines,
    enumerate_sessions,
    resolve_agent_transcript_roots,
)

logger = logging.getLogger('legibility.sampling')

# _iter_json_lines lives in legibility.inventory — this module reuses that
# single fire-and-forget-transcript-read implementation rather than keeping
# its own byte-for-byte copy (a future fix to the graceful-degrade contract
# then only needs to land in one place).


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


def _score_and_find_first_turn(path: Path) -> tuple[SignalCounts, dict[str, Any] | None]:
    """Single pass over *path*, producing both a signal score and the first user turn.

    :func:`score_signals` already has to read every record in *path* (a
    confusion signal can occur anywhere in the file); :func:`main`'s
    scoring loop separately called :func:`_find_first_user_turn` on the
    same path, re-opening and re-parsing the same transcript purely to
    locate its first non-sidechain, non-meta user turn
    (reviewer_comprehensive/performance, task 2573 amendment pass #2).
    This helper folds both searches into the SAME iteration over
    :func:`legibility.inventory._iter_json_lines`, so a session transcript
    is read once instead of twice. :func:`score_signals` and
    :func:`_find_first_user_turn` both delegate to this function, so every
    other caller (including the direct unit tests of :func:`score_signals`)
    keeps its existing single-purpose signature and exact return value.
    """
    tool_error = not_found = self_correct = df_guard = interrupt = 0
    first_turn: dict[str, Any] | None = None
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

            if (
                first_turn is None
                and record.get('type') == 'user'
                and not record.get('isSidechain')
                and not record.get('isMeta')
            ):
                first_turn = record
    except OSError:
        return SignalCounts(), None

    counts = SignalCounts(
        tool_error=tool_error,
        not_found=not_found,
        self_correct=self_correct,
        df_guard=df_guard,
        interrupt=interrupt,
    )
    return counts, first_turn


def score_signals(path: Path) -> SignalCounts:
    """Zero-token single pass over a transcript, counting the 5 confusion-signal classes.

    One increment per class per RECORD (not per pattern match) — a record
    carrying multiple synonymous markers for the same class still counts
    once. ``tool_error`` and the tool-use half of ``df_guard`` are
    structural checks; every other class is a plain case-insensitive
    substring scan. Malformed/unreadable input degrades to an all-zero
    :class:`SignalCounts` rather than raising. Delegates to
    :func:`_score_and_find_first_turn` (shared with
    :func:`_find_first_user_turn` so :func:`main`'s scoring loop can get
    both in one file read); see its docstring for why.
    """
    counts, _ = _score_and_find_first_turn(path)
    return counts


# ---------------------------------------------------------------------------
# classify_agent_class — 5-stratum classifier (PRD §5.2 point 2)
# ---------------------------------------------------------------------------

STRATA: tuple[str, ...] = (
    'recon', 'curator-classifier', 'watcher', 'orchestrated-task', 'interactive',
)
"""The 5 agent-class strata the sampler groups by."""

_WORKTREE_DIR_MARKERS: tuple[str, ...] = (
    '--worktrees-', '--claude-worktrees-',
    # reify's warm-lane and build-worktree encoded-dir shapes (task 2612):
    # encode_cwd (inventory.py) maps both '/' and '.' to '-', so
    # /home/leo/src/reify/.warm-lanes/worktrees/<id> and
    # .../reify/build-worktrees/<id> never produce the double-dash
    # '--worktrees-'/'--claude-worktrees-' form above.
    # The two are deliberately asymmetric: 'build-worktrees' is a generic
    # enough phrase that an unqualified '-build-worktrees-' marker could
    # false-match an unrelated repo's own build/worktrees dir (e.g. a repo
    # named '...-build'), so it's qualified with the repo name; whereas
    # 'warm-lanes-worktrees' is distinctive enough on its own that no
    # project qualifier is needed.
    '-warm-lanes-worktrees-', '-reify-build-worktrees-',
)

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
    parent directory's name is inspected, never any file content — an
    orchestrated-worktree marker (dark_factory's ``--worktrees-``/
    ``--claude-worktrees-``, or reify's ``-warm-lanes-worktrees-``/
    ``-reify-build-worktrees-``; see :data:`_WORKTREE_DIR_MARKERS`) means
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


# ---------------------------------------------------------------------------
# ScoredRecord, shape_fingerprint, dedupe_shapes — near-duplicate collapsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoredRecord:
    """A :class:`~legibility.inventory.SessionRecord` enriched with its
    signal score and agent-class stratum — the unit :func:`stratified_sample`
    operates on. Produced by :func:`main` via
    ``inventory.enumerate_sessions -> score_signals -> classify_agent_class``,
    or constructed directly for pure in-memory testing (PRD §8.4).
    """

    session: SessionRecord
    stratum: str
    counts: SignalCounts
    first_turn_text: str = ''

    @property
    def score(self) -> int:
        return self.counts.total_signal

    @property
    def size_bytes(self) -> int:
        return self.session.size_bytes

    @property
    def path(self) -> Path:
        return self.session.path


_DIGITS_RE = re.compile(r'\d+')


def _normalize_first_turn(text: str, *, prefix_len: int = 80) -> str:
    """Collapse whitespace, replace digit runs with '#', and cap length.

    Normalizes away the parts of a recon/watcher clone's first turn that
    vary run-to-run (dates, session-specific numbers) while keeping enough
    of the structural prefix to distinguish a genuinely different prompt.
    """
    collapsed = ' '.join(text.split())
    digitless = _DIGITS_RE.sub('#', collapsed)
    return digitless[:prefix_len].lower()


def shape_fingerprint(record: ScoredRecord) -> tuple[str, tuple[bool, ...], str]:
    """A cheap, hashable near-duplicate key: (stratum, signal-shape, first-turn skeleton).

    ``signal-shape`` is a boolean presence-per-class pattern (never exact
    counts) — near-identical recon clones fire the same CLASSES of signal
    night after night even when exact counts drift by one or two. The
    first-turn skeleton (:func:`_normalize_first_turn`) absorbs
    date/number drift between otherwise-identical clone runs.
    """
    counts = record.counts
    signal_shape = (
        bool(counts.tool_error),
        bool(counts.not_found),
        bool(counts.self_correct),
        bool(counts.df_guard),
        bool(counts.interrupt),
    )
    return (record.stratum, signal_shape, _normalize_first_turn(record.first_turn_text))


def dedupe_shapes(records: Sequence[ScoredRecord]) -> list[ScoredRecord]:
    """Collapse near-duplicate shapes, keeping the highest-scoring representative per fingerprint.

    Returned survivors follow first-occurrence order of each fingerprint in
    *records* (stable) — callers needing score order sort separately.
    """
    best: dict[tuple, ScoredRecord] = {}
    order: list[tuple] = []
    for record in records:
        fingerprint = shape_fingerprint(record)
        if fingerprint not in best:
            order.append(fingerprint)
            best[fingerprint] = record
        elif record.score > best[fingerprint].score:
            best[fingerprint] = record
    return [best[fingerprint] for fingerprint in order]


# ---------------------------------------------------------------------------
# stratified_sample — budget-after-stratification sampler (PRD §5.2 point 2, §8.4)
# ---------------------------------------------------------------------------

DEFAULT_DIGEST_MAX_BYTES = 15360
"""The PRD §7.2 per-digest soft cap, in bytes.

``nightly.DEFAULT_MAX_DIGEST_BYTES`` is an ALIAS of this name, not a second
definition — the two spellings are one word-order swap apart, so independent
literals would drift invisibly. :func:`legibility.digest.build_digest`'s own
``max_bytes`` default is still a separate literal (``digest.py`` is outside
task 3268's scope); ``TestDigestByteCostFn`` pins the two equal so drift
fails a test.

Used by :func:`stratified_sample` as its conservative flat per-record charge
when no ``cost_fn`` is injected, and as :func:`digest_byte_cost_fn`'s default
render cap. Charging it flat over-estimates any realistic digest — but do NOT
build a hard-cap invariant on that. §7.2 is a SOFT cap:
:func:`legibility.digest._truncate_sections` stops trimming once every section
is empty "even if the (now section-free) digest is still over the cap — a soft
cap can't shrink the frontmatter itself". A digest whose frontmatter alone
exceeds *max_bytes* (a pathological ``cwd``/``session`` value, or a very small
injected cap) therefore renders LARGER than this constant, and neither the flat
fallback nor this value is a bound in that case."""


def render_cache_key(path: Path, max_bytes: int, renderer) -> tuple:
    """The shared key for a rendered-digest cache entry.

    Defined ONCE, here, because two pipeline stages must agree on it exactly:
    :func:`digest_byte_cost_fn` (which populates the cache while costing) and
    ``nightly.build_digests`` (which reads it instead of re-rendering). The
    key carries *max_bytes* AND the *renderer* itself alongside the path, so a
    cache entry can never be served to a stage using a different soft cap or a
    different (e.g. monkeypatched, or test-injected) builder — a mismatch is a
    structural cache MISS and an honest re-render, not a silently wrong digest.
    """
    return (path, max_bytes, renderer)


def digest_byte_cost_fn(
    *, max_bytes: int = DEFAULT_DIGEST_MAX_BYTES, build=None,
    rendered: dict[tuple, str] | None = None,
) -> Callable[[ScoredRecord], int]:
    """Build the production ``cost_fn`` for :func:`stratified_sample`:
    charge each record the REAL byte size of the digest it renders to.

    This is what makes ``budgets.max_daily_digest_bytes`` mean what it says
    (PRD §5.2 point 2 — "Budget cap, not count cap"): the charge is the
    UTF-8 byte length of ``build(record.path,
    agent_class_override=record.stratum, max_bytes=max_bytes)``. Passing
    *max_bytes* straight through matters — a caller that renders with the
    same constant it charges with (``nightly.DEFAULT_MAX_DIGEST_BYTES``)
    gets a ``bytes_used`` that provably describes the digests actually
    produced, not a parallel estimate free to drift from them. Beta's
    ``record.stratum`` is already authoritative, so alpha never re-guesses
    the agent class.

    *build* defaults to :func:`legibility.digest.build_digest`, resolved at
    CALL time so a test can monkeypatch the module attribute.

    *rendered* is a RENDER CACHE, keyed by :func:`render_cache_key`: every
    successful render is stored in it as the digest TEXT, not just its byte
    count. Its job is cross-STAGE, and it is the only reason this closure
    holds a cache at all — ``stratified_sample`` already memoizes ``cost_fn``
    per record, so a byte-count memo here would be dead weight in the only
    path that uses it (reviewer_comprehensive, task 3268 amendment pass). The
    two caches answer different questions: ``stratified_sample``'s prevents a
    second CALL within one sampler run; this one prevents a second RENDER by
    a later pipeline stage. ``nightly.run_nightly`` passes one dict through
    both :func:`nightly.select_digest_sessions` and
    :func:`nightly.build_digests`, so a candidate rendered to measure its cost
    is not rendered again to emit it. That matters more than it looks:
    :func:`legibility.digest._truncate_sections` is super-linear in signal-item
    count (it pops ONE item and re-renders the whole digest per iteration), so
    an unshared costing render roughly DOUBLED the dominant cost of the nightly
    job — measured on the ``_write_multi_mb_transcript`` fixture shape at
    0.80s / 2.57s / 14.96s per render for 1000 / 2000 / 4000 error lines.
    Passing no *rendered* dict keeps a private one, so a standalone caller
    (:func:`main`) still never renders the same session twice.

    The cache holds digest TEXT (~15KB each, §7.2-capped) for post-dedupe,
    post-``top_fraction`` candidates only — tens of entries, not one per
    enumerated session.

    A raising *build* does NOT propagate: the record is charged ZERO and a
    warning is logged. Zero is the ACCURATE charge — a render that raises
    produces no digest bytes, so it consumes none of the daily budget — and
    it is also the charge that best preserves the loud path: the record
    stays as affordable as possible, so it reaches
    :func:`nightly.build_digests`, raises there again, and is reported
    through the EXISTING structured ``extractor_failures`` -> escalation
    channel (PRD decision 8). Propagating here would instead take down the
    entire night's run with a bare traceback from the sampler, losing every
    other session's digest along with the failure's own structured report.

    This charge was previously the flat *max_bytes* — the most EXPENSIVE
    charge available, which made a failed costing the candidate most likely
    to be cut by the budget (a reserve group skipped whole, or the greedy
    leftover fill halting on it). A cut record never reaches
    ``build_digests``, so no ``extractor_failures`` entry and no escalation
    were produced and the only trace was this warning: precisely the silent
    fail-soft the degrade was written to avoid (reviewer_comprehensive,
    task 3268 amendment pass). Charging zero removes the budget as a way to
    lose the report — though note it is not an absolute guarantee: a
    zero-cost record can still be skipped as part of an unaffordable reserve
    GROUP, or sit behind the greedy fill's strict halt.
    """
    cache: dict[tuple, str] = {} if rendered is None else rendered

    def cost(record: ScoredRecord) -> int:
        renderer = build if build is not None else digest.build_digest
        key = render_cache_key(record.path, max_bytes, renderer)

        cached = cache.get(key)
        if cached is not None:
            return len(cached.encode('utf-8'))

        try:
            text = renderer(
                record.path, agent_class_override=record.stratum, max_bytes=max_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - degrade loud, never propagate
            # Deliberately NOT cached: build_digests must re-attempt, raise
            # again, and report the failure via extractor_failures.
            logger.warning(
                'digest costing failed for %s (%s: %s); charging 0 bytes — a render '
                'that raises produces no digest, and the cheapest possible charge is '
                'what keeps the record affordable enough to reach build_digests, '
                'which reports the failure through extractor_failures',
                record.path.name, type(exc).__name__, exc,
            )
            return 0

        cache[key] = text
        return len(text.encode('utf-8'))

    return cost


@dataclass(frozen=True)
class SampleResult:
    """The result of :func:`stratified_sample`: the selection plus accounting.

    Every record handed to :func:`stratified_sample` leaves by exactly ONE
    of five doors, and each door has its own counter:

    - ``zero_signal_dropped`` — no confusion signal at all.
    - ``dedupe_collapsed`` — a near-duplicate shape collapsed by
      :func:`dedupe_shapes`.
    - ``below_sampling_cut`` — real, distinct signal that ranked below its
      stratum's ``top_fraction``/``per_stratum_min`` cut, so it never
      competed for budget.
    - ``budget_skipped`` — reached the candidate stage and DID compete, but
      was cut solely because the byte budget ran out (a reserved-floor group
      skipped whole, or a leftover candidate excluded by the greedy fill's
      halt).
    - ``selected`` — digested.

    Separating those doors is the whole point: it lets an operator reading
    :func:`main`'s summary tell "there was nothing to sample" apart from
    "the budget cut off real signal" (reviewer_comprehensive/observability,
    task 2573 amendment pass #2) apart from "the sampling cut held real
    signal back" — three states that otherwise look identical from outside.
    The last two have DIFFERENT remedies and must never be conflated:
    raising ``budgets.max_daily_digest_bytes`` recovers a ``budget_skipped``
    record and does nothing whatever for a ``below_sampling_cut`` one, which
    needs ``sampling.top_fraction``/``per_stratum_min`` raised instead.

    CONSERVATION INVARIANT (the full identity, pinned as a test in
    ``test_legibility_sampling.TestSampleResultAccounting``)::

        total_records == zero_signal_dropped + dedupe_collapsed
                         + below_sampling_cut + budget_skipped
                         + len(selected)

    Nothing falls through unaccounted, so the operator line always balances
    and "why did only 3 of 17 sessions get digested?" is answerable from the
    line alone. Its load-bearing corollary is that every record reaching the
    CANDIDATE stage ends in exactly one of ``selected`` or
    ``budget_skipped`` — which is what makes ``not selected and
    budget_skipped > 0`` mean exactly "candidates existed and were ALL
    discarded on the byte budget", the total-suppression predicate
    ``nightly._report_sample_outcome`` escalates on, and the reason a
    budget-suppressed night is no longer indistinguishable from a genuine
    no-change night (task 3270).
    """

    selected: list[ScoredRecord]
    per_stratum_counts: dict[str, int]
    zero_signal_dropped: int

    bytes_used: int
    """Total DIGEST bytes charged for ``selected`` — never raw transcript
    size. ``budgets.max_daily_digest_bytes`` is a digest-OUTPUT budget (PRD
    §5.2 point 2: "fill the daily digest-byte budget in score order. Budget
    cap, not count cap"), so this is the sum of what
    :func:`stratified_sample`'s cost basis charged: the real rendered digest
    size when a ``cost_fn`` is injected, and a conservative flat
    :data:`DEFAULT_DIGEST_MAX_BYTES` per record otherwise. It is NEVER
    ``inventory.SessionRecord.size_bytes`` — a 0.5-6.5MB transcript renders
    to a §7.2-capped ~15KB digest, so charging raw size over-charged the
    budget by 20x-500x and selected nothing at all."""

    dedupe_collapsed: int = 0
    budget_skipped: int = 0

    below_sampling_cut: int = 0
    """Records that survived the zero-signal and dedupe filters but ranked
    below their stratum's candidate cut, so they never reached the budget
    phase at all.

    The cut is ``max(ceil(top_fraction * stratum_size), per_stratum_min)``
    (see :func:`stratified_sample` step 4): at the stock
    ``top_fraction=0.12`` a stratum of 17 distinct real-signal records
    yields 3 candidates and 14 of these. Those 14 landed in NO bucket
    before task 3270's amendment pass, so the summary line simply did not
    balance and an operator could reasonably read the gap as a second,
    undiagnosed suppression mode.

    Deliberately NOT folded into ``budget_skipped``: these records lost to
    the SAMPLING policy, not to the byte budget, and the two have different
    fixes (see the class docstring). Defaulted so hand-built
    ``SampleResult``s in tests stay valid."""

    total_records: int = 0
    """How many records were HANDED TO :func:`stratified_sample` — the
    denominator every other count in this dataclass is a slice of.

    Both production callers build exactly ONE ``ScoredRecord`` per
    ``inventory.enumerate_sessions`` hit (:func:`main`'s scoring loop and
    ``nightly.select_scored_records``), so this is 1:1 with the sessions
    enumerated for the target date — which is why the operator summary line
    may label it ``enumerated=``. It is the number of INPUTS, not of
    survivors and not of candidates: ``total_records - zero_signal_dropped
    - dedupe_collapsed`` is what survived the two pre-budget FILTERS, and
    the stratum sampling cut then narrows those survivors further, so what
    actually reached the candidate stage is that quantity minus
    ``below_sampling_cut`` (equivalently: ``len(selected) +
    budget_skipped``). See the full conservation identity above. Defaulted
    so the existing all-keyword construction sites, and hand-built
    ``SampleResult``s in tests, stay valid."""


def stratified_sample(
    records: Sequence[ScoredRecord],
    config: LegibilityConfig,
    *,
    cost_fn: Callable[[ScoredRecord], int] | None = None,
) -> SampleResult:
    """Pick a budget-bounded, stratified subset of *records* (PRD §5.2 point 2, §8.4).

    The budget is a DIGEST-OUTPUT budget, and so is everything charged
    against it. ``budgets.max_daily_digest_bytes`` bounds the bytes of
    digest this night will PRODUCE (PRD §5.2 point 2: "fill the daily
    digest-byte budget in score order. Budget cap, not count cap"), and each
    digest is itself soft-capped at §7.2's 15KB. The charged quantity — and
    therefore ``SampleResult.bytes_used`` — is always digest bytes, never
    ``inventory.SessionRecord.size_bytes``: a 0.5-6.5MB transcript renders
    to a ~15KB digest, so charging raw transcript size over-charged the
    budget by 20x-500x and made a single real session unaffordable against
    the whole night's cap.

    *cost_fn* is the per-record BYTE-COST seam every budget charge routes
    through. Real callers inject :func:`digest_byte_cost_fn`, which charges
    the ACTUAL rendered digest size. When *cost_fn* is None each record is
    charged a flat :data:`DEFAULT_DIGEST_MAX_BYTES` — correct UNITS and an
    over-estimate for any realistic digest, so the daily cap holds in
    practice. It is NOT a hard bound: §7.2 is a soft cap and the frontmatter
    is not trimmable, so see :data:`DEFAULT_DIGEST_MAX_BYTES` before relying
    on it as one. That fallback is deliberately
    conservative, NOT the intended production basis: the PRD rules out a
    pure count cap, and a flat charge is exactly a count cap in disguise. It
    exists so this function stays injection-free and pure for the §8.4
    boundary test, and so a caller who forgets to inject degrades to a
    coarse budget rather than back to raw transcript bytes.

    This function stays pure with respect to its own inputs — no clock, and
    all I/O confined to the injected *cost_fn* seam, so the PRD §8.4
    boundary test can still drive it entirely in memory. Two guarantees the
    seam makes to an I/O-backed *cost_fn*: each record is costed AT MOST
    ONCE per call (memoized on ``record.path``, which matters because the
    reserve phase reads each group's total twice — sort key, then charge),
    and only CANDIDATES are ever costed (records dropped as zero-signal or
    collapsed by :func:`dedupe_shapes` never reach the budget phase).

    Per stratum: (1) group; (2) drop
    zero-signal records; (3) :func:`dedupe_shapes` the survivors; (4) rank
    by score desc and take ``candidates = top max(ceil(top_fraction *
    stratum_size), per_stratum_min)`` capped at the survivor count (the
    ``top_fraction`` proportion is relative to the stratum's ORIGINAL
    size, including zero-signal/clone records, not just the survivors —
    otherwise a heavily-noisy stratum would sample far less than intended
    of its real signal); (5) RESERVE each non-empty stratum's top
    ``min(per_stratum_min, len(candidates))`` as a GROUP, cheapest
    stratum-floor (by total bytes) first — accumulated BEFORE any
    cross-stratum score comparison, so a stratum of huge high-score
    sessions elsewhere can never evict a cheaper stratum's floor (the §8.4
    "big sessions can't evict a whole stratum" postcondition). The overall
    byte budget still governs even the reserve phase: a stratum's floor is
    skipped whole (never partially) if it would push the running total
    over budget — real ``~/.claude/projects`` session sizes can dwarf a
    conservative daily budget, so "budget-protected" floors must still
    respect the cap, never silently exceed it; (6) fill the remaining
    budget from the leftover (non-reserved) candidates across ALL strata,
    in global score-descending order, stopping outright at the first
    candidate that would exceed the budget (a strict greedy halt, not a
    skip-ahead bin-pack). The final selection is returned in
    score-descending order. ``dedupe_collapsed`` (records collapsed away by
    :func:`dedupe_shapes` in step 3), ``below_sampling_cut`` (survivors the
    step-4 cut left out, which therefore never compete for budget at all)
    and ``budget_skipped`` (candidates excluded solely by the byte cap in
    steps 5-6) are accumulated
    alongside the selection for :func:`main`'s summary — as is
    ``total_records`` (``len(records)``, one per enumerated session in both
    production callers) — see :class:`SampleResult`.
    """
    top_fraction = config.sampling.top_fraction
    per_stratum_min = config.sampling.per_stratum_min
    max_bytes = config.budgets.max_daily_digest_bytes

    cost_memo: dict[Path, int] = {}

    def _cost(record: ScoredRecord) -> int:
        """This call's single budget-charge basis, memoized per session path.

        Two records sharing a path are the same session by construction, so
        the path is a safe memo key — and memoizing is what lets an
        I/O-backed *cost_fn* be invoked once per candidate rather than once
        per budget reference.
        """
        cached = cost_memo.get(record.path)
        if cached is not None:
            return cached
        charged = cost_fn(record) if cost_fn is not None else DEFAULT_DIGEST_MAX_BYTES
        cost_memo[record.path] = charged
        return charged

    by_stratum: dict[str, list[ScoredRecord]] = {}
    for record in records:
        by_stratum.setdefault(record.stratum, []).append(record)

    zero_signal_dropped = 0
    dedupe_collapsed = 0
    below_sampling_cut = 0
    reserved_groups: list[list[ScoredRecord]] = []
    leftover: list[ScoredRecord] = []

    for stratum_records in by_stratum.values():
        stratum_size = len(stratum_records)

        nonzero = [r for r in stratum_records if r.score > 0]
        zero_signal_dropped += stratum_size - len(nonzero)

        deduped = dedupe_shapes(nonzero)
        dedupe_collapsed += len(nonzero) - len(deduped)
        survivors = sorted(deduped, key=lambda r: r.score, reverse=True)

        candidate_count = min(
            max(math.ceil(top_fraction * stratum_size), per_stratum_min),
            len(survivors),
        )
        candidates = survivors[:candidate_count]
        # Survivors the cut left out. Counted rather than dropped on the
        # floor: they are real, distinct signal, and an operator asking why
        # only 3 of 17 sessions were digested has to be able to read the
        # answer off the summary line (see SampleResult.below_sampling_cut).
        below_sampling_cut += len(survivors) - candidate_count

        reserve_count = min(per_stratum_min, len(candidates))
        if reserve_count:
            reserved_groups.append(candidates[:reserve_count])
        leftover.extend(candidates[reserve_count:])

    selected: list[ScoredRecord] = []
    bytes_used = 0
    budget_skipped = 0
    for group in sorted(reserved_groups, key=lambda g: sum(_cost(r) for r in g)):
        group_bytes = sum(_cost(r) for r in group)
        if bytes_used + group_bytes > max_bytes:
            budget_skipped += len(group)
            continue
        selected.extend(group)
        bytes_used += group_bytes

    sorted_leftover = sorted(leftover, key=lambda r: r.score, reverse=True)
    for index, record in enumerate(sorted_leftover):
        record_bytes = _cost(record)
        if bytes_used + record_bytes > max_bytes:
            budget_skipped += len(sorted_leftover) - index
            break
        selected.append(record)
        bytes_used += record_bytes

    selected.sort(key=lambda r: r.score, reverse=True)

    per_stratum_counts: dict[str, int] = {}
    for record in selected:
        per_stratum_counts[record.stratum] = per_stratum_counts.get(record.stratum, 0) + 1

    return SampleResult(
        selected=selected,
        per_stratum_counts=per_stratum_counts,
        zero_signal_dropped=zero_signal_dropped,
        bytes_used=bytes_used,
        dedupe_collapsed=dedupe_collapsed,
        budget_skipped=budget_skipped,
        below_sampling_cut=below_sampling_cut,
        total_records=len(records),
    )


# ---------------------------------------------------------------------------
# render_manifest + main — CLI acceptance surface
# ---------------------------------------------------------------------------

def render_manifest(selected: Sequence[ScoredRecord]) -> str:
    """Render *selected* as JSONL, one ``{session, stratum, score, size}`` object per line.

    ``session`` is the session's transcript path (as a string). Empty
    input yields an empty string (no trailing newline either way — callers
    ``print()`` the result, which adds the final newline).
    """
    lines = [
        json.dumps({
            'session': str(record.path),
            'stratum': record.stratum,
            'score': record.score,
            'size': record.size_bytes,
        })
        for record in selected
    ]
    return '\n'.join(lines)


def format_sample_summary_line(result: SampleResult, *, max_bytes: int) -> str:
    """Render *result*'s accounting as ONE greppable ``key=value`` line.

    The single source for both operator-facing surfaces: :func:`main`'s
    stderr summary block appends it, and ``nightly.run_nightly`` logs it at
    INFO on EVERY run. One formatter rather than two field lists, so the
    diagnostic an operator runs by hand and the line the systemd timer
    writes to the journal can never disagree about what the budget was
    spent on (INV-5 no-lockstep-duplication) — the drift that let a
    totally-budget-suppressed night read exactly like a genuine no-change
    night for 14 nights (2026-07-16..29, task 3270).

    Emits no prefix and no trailing newline: callers own both (nightly
    prefixes the project id and target date, since one journal interleaves
    every project's timer). Every field is on one line because that is what
    an operator's ``journalctl ... | grep`` has to return as a single unit;
    the per-stratum breakdown deliberately stays in :func:`main`'s
    multi-line block, where it does not have to fit.

    ``enumerated`` is :attr:`SampleResult.total_records` — one per
    enumerated session in both production callers.

    The drop fields are emitted in PIPELINE order and, by
    :class:`SampleResult`'s conservation invariant, they BALANCE:
    ``enumerated`` equals the four drop counts plus ``selected``. That is
    load-bearing for the operator reading it — an unaccounted remainder
    would read as a second, undiagnosed suppression mode — so a new drop
    bucket added to the sampler must be added here too.
    """
    return (
        f'enumerated={result.total_records} '
        f'zero_signal_dropped={result.zero_signal_dropped} '
        f'dedupe_collapsed={result.dedupe_collapsed} '
        f'below_sampling_cut={result.below_sampling_cut} '
        f'budget_skipped={result.budget_skipped} '
        f'selected={len(result.selected)} '
        f'bytes_used={result.bytes_used}/{max_bytes}'
    )


def _find_first_user_turn(path: Path) -> dict[str, Any] | None:
    """Find a session's first non-sidechain, non-meta user-turn record.

    Returns the raw record dict (as :func:`classify_agent_class` expects),
    or None if the transcript has no such turn. Malformed/unreadable input
    degrades to None rather than raising. Delegates to
    :func:`_score_and_find_first_turn`; see its docstring for why.
    """
    _, first_turn = _score_and_find_first_turn(path)
    return first_turn


def main(argv: Sequence[str]) -> int:
    """CLI entry point: the full inventory -> score -> classify -> sample -> render pipeline.

    Wires ``config.load_config -> inventory.enumerate_sessions ->
    score_signals -> classify_agent_class -> stratified_sample ->
    render_manifest``. Prints the JSONL manifest to stdout and a
    per-stratum-counts / zero-signal-drops / budget-accounting summary to
    stderr. This is the CLI acceptance surface — a manual run against live
    ``~/.claude/projects`` is the observable, not a pytest.

    Samples with the same real-digest-byte cost basis
    (:func:`digest_byte_cost_fn` at :data:`DEFAULT_DIGEST_MAX_BYTES`) that
    ``nightly.select_digest_sessions`` uses, so this diagnostic and the
    pipeline can never disagree about what the budget was spent on — the
    whole point of reading this CLI is to find out whether the budget is
    why a session was skipped. The summary's byte line is therefore
    labelled "digest bytes used": it is digest OUTPUT, not raw transcript
    size. Note that ``render_manifest``'s per-record ``size`` field is
    deliberately still the raw transcript size — informational, and a
    different question from what the record costs the budget.
    """
    default_config = Path(__file__).resolve().parent.parent.parent / 'docs' / 'legibility' / 'legibility.yaml'
    parser = argparse.ArgumentParser(
        description='Sample yesterday\'s confusion-signal sessions for the nightly digest.'
    )
    parser.add_argument(
        '--config', default=str(default_config),
        help='Path to legibility.yaml (default: %(default)s)',
    )
    parser.add_argument(
        '--projects-root', default=str(Path.home() / '.claude' / 'projects'),
        help='Root of the ~/.claude/projects tree (default: %(default)s)',
    )
    parser.add_argument(
        '--date', default=None,
        help='UTC date to sample, YYYY-MM-DD (default: yesterday UTC).',
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target_date = (
        date.fromisoformat(args.date) if args.date
        else (datetime.now(UTC) - timedelta(days=1)).date()
    )

    sessions = enumerate_sessions(
        args.projects_root, cfg.cwd_prefixes, target_date,
        agent_transcript_roots=resolve_agent_transcript_roots(
            cfg.project_root, cfg.agent_transcript_roots
        ),
    )

    scored: list[ScoredRecord] = []
    for session in sessions:
        counts, first_turn = _score_and_find_first_turn(session.path)
        stratum = classify_agent_class(first_turn, session.path)
        scored.append(
            ScoredRecord(
                session=session,
                stratum=stratum,
                counts=counts,
                first_turn_text=_first_user_turn_text(first_turn),
            )
        )

    result = stratified_sample(
        scored, cfg, cost_fn=digest_byte_cost_fn(max_bytes=DEFAULT_DIGEST_MAX_BYTES),
    )

    print(render_manifest(result.selected))

    summary = ['=== legibility sampler summary ===', '']
    # result.total_records rather than len(sessions): the same number (one
    # ScoredRecord per enumerated session, just above), sourced from the
    # object the greppable line below is also formatted from, so the
    # human block and the one-liner cannot drift apart.
    summary.append(f'sessions enumerated: {result.total_records}')
    summary.append(f'zero-signal dropped: {result.zero_signal_dropped}')
    summary.append(f'near-duplicate clones collapsed: {result.dedupe_collapsed}')
    # Named for its REMEDY, not just its cause: unlike the budget-skipped
    # line below, no amount of extra byte budget recovers these — the
    # stratum sampling cut is what held them back.
    summary.append(
        f'below the stratum sampling cut (raise sampling.top_fraction / '
        f'per_stratum_min to recover): {result.below_sampling_cut}'
    )
    for stratum in sorted(result.per_stratum_counts):
        summary.append(f'  {stratum}: {result.per_stratum_counts[stratum]} selected')
    summary.append(
        f'digest bytes used: {result.bytes_used} / {cfg.budgets.max_daily_digest_bytes}'
    )
    summary.append(f'budget-skipped candidates (would exceed cap): {result.budget_skipped}')
    # The same greppable one-liner ``nightly.run_nightly`` logs to the
    # journal, from the same formatter — so this hand-run diagnostic and the
    # timer's nightly line can never report different numbers (INV-5).
    summary.append('')
    summary.append(
        format_sample_summary_line(result, max_bytes=cfg.budgets.max_daily_digest_bytes)
    )
    print('\n'.join(summary), file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
