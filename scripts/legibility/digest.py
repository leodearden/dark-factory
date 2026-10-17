#!/usr/bin/env python3
"""Deterministic confusion-digest extractor (zero LLM).

Turns a single Claude Code session transcript JSONL file into a 5-15KB
markdown digest with YAML frontmatter: one section per confusion-signal
class (non-sidechain user turns, tool-error neighborhoods, self-corrections,
retry loops), plus a handful of scalar signal counts (not-found, df-guard
trips, interrupts).

Task alpha of the confusion-reduction PRD (plans/confusion-reduction-prd.md
Sec 5.1, contract Sec 7.2, boundary test Sec 8.1 producer side). Signal hits
and the agent class come from :mod:`legibility.detectors` (Sec 7.2.4).

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
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Self-bootstrap for standalone `python scripts/legibility/digest.py` runs —
# must run BEFORE the `legibility.*` import below, since a direct script
# invocation puts only scripts/legibility/ (not scripts/) on sys.path.
# Skipped under pytest/normal package import: __name__ is 'legibility.digest'.
# Verbatim the sampling.py:43-48 / census.py / nightly.py guard.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility.detectors import (  # noqa: E402  # debt: task 6760
    CarrierKind,
    ErrorHit,
    RecordHits,
    SignalClass,
    classify_agent_class,
    error_hits,
    is_non_dialogue_carrier,
    message_content,
    record_hits,
    user_turn_text,
)
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
        content = message_content(record)
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                found.append((index, block))
    return found


def iter_error_neighborhoods(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each structured error (:func:`detectors.error_hits`) with the
    assistant attempt that produced it.

    Matching to the attempt is via ``tool_use_id``; an unmatched id (e.g.
    the attempt was truncated off the front of the transcript window)
    degrades to None attempt fields rather than raising.

    That id is also EXPOSED on each neighborhood as ``tool_use_id``: it is
    the stable join key letting a caller relate a classified result back to
    the assistant attempt that produced it WITHOUT a second scan. It is
    what :func:`find_retry_loops` joins on to annotate a retry group with
    how many of its calls ended in a DESIGNED outcome
    (``plans/confusion-census-2026-08-26.md`` §1.1 / R1, task 4751) --
    a retry group's ``indices`` are tool_use record positions while a
    neighborhood's ``index`` is the tool_result one, so record index cannot
    serve as that key. Because it is read off the RESULT block it survives
    an unmatched attempt, the same degradation contract the None attempt
    fields above already document.

    Each neighborhood carries ``exit_code`` and ``designed_outcome`` (see
    :func:`detectors.classify_error_content`);
    :func:`iter_genuine_errors` and :func:`iter_designed_outcomes` are its
    two partitions.
    """
    return _error_neighborhoods(records, [error_hits(record) for record in records])


def _error_neighborhoods(
    records: list[dict[str, Any]], errors_per_record: Sequence[tuple[ErrorHit, ...]],
) -> list[dict[str, Any]]:
    attempts_by_id = {
        block.get('id'): block
        for _, block in _iter_tool_use_blocks(records)
    }
    neighborhoods = []
    for index, errors in enumerate(errors_per_record):
        for error in errors:
            attempt = attempts_by_id.get(error.tool_use_id)
            neighborhoods.append({
                'index': index,
                'tool_use_id': error.tool_use_id,
                'attempt_tool': attempt.get('name') if attempt else None,
                'attempt_input_summary': (
                    _summarize_input(attempt.get('input')) if attempt else None
                ),
                'error_content': error.content,
                'exit_code': error.exit_code,
                'designed_outcome': error.designed_outcome,
            })
    return neighborhoods


def _neighborhoods_to_partition(
    records: list[dict[str, Any]] | None,
    neighborhoods: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Resolve what a partition helper should filter: an already-computed
    ``neighborhoods`` list when the caller has one, else a fresh scan of
    ``records``. Exactly one is required -- passing neither is a caller
    bug, raised rather than silently returning an empty partition."""
    if neighborhoods is not None:
        return neighborhoods
    if records is None:
        raise TypeError('pass either records or neighborhoods=')
    return iter_error_neighborhoods(records)


def iter_genuine_errors(
    records: list[dict[str, Any]] | None = None,
    *,
    neighborhoods: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The neighborhoods that are REAL failures (``designed_outcome`` None).

    One half of a total, lossless partition of
    :func:`iter_error_neighborhoods`; the other half is
    :func:`iter_designed_outcomes`. This is what ``signal_counts``
    tallies as ``tool_error``.

    Pass ``neighborhoods=`` to partition a scan the caller already has, as
    :func:`_build_sections` does with its one per-build scan.
    """
    return [
        n for n in _neighborhoods_to_partition(records, neighborhoods)
        if n['designed_outcome'] is None
    ]


def iter_designed_outcomes(
    records: list[dict[str, Any]] | None = None,
    *,
    neighborhoods: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The neighborhoods that are DESIGNED outcomes, not failures.

    The complement of :func:`iter_genuine_errors` over
    :func:`iter_error_neighborhoods` -- together the two are total and
    lossless, and ``neighborhoods=`` partitions a caller's existing scan
    exactly as it does there. Reported under its own ``designed_outcome``
    count and its own digest section rather than discarded, so a reader of
    a digest can see that N of its structured errors were bounded-wait
    ceilings rather than failures.

    Reported, but only where a digest exists to report it: a session whose
    ONLY structured errors are designed outcomes scores zero on the
    sampler's ranking scale (``sampling.SignalCounts.total_signal``
    excludes this class by design) and is dropped at the zero-signal gate
    before digest time. The count disambiguates a MIXED session's errors;
    it is not a corpus-wide census of ceilings (PRD Sec 7.2.1).
    """
    return [
        n for n in _neighborhoods_to_partition(records, neighborhoods)
        if n['designed_outcome'] is not None
    ]


_CARRIER_ORDER = {kind: rank for rank, kind in enumerate(CarrierKind)}


def _signal_items(hits: Sequence[RecordHits], signal: SignalClass) -> list[dict[str, Any]]:
    """One text class's hits as digest items ``{index, pattern, context}``.

    Record-major, then a STABLE sort on carrier kind, so a class read in
    several carrier kinds lists every tool_result hit, then every assistant
    text hit, then every user-turn hit. Section trimming pops trailing
    items, so this order decides which items survive the byte cap.
    """
    items = [
        (hit.carrier, {'index': index, 'pattern': hit.pattern, 'context': hit.line})
        for index, record in enumerate(hits)
        for hit in record.patterns[signal]
    ]
    items.sort(key=lambda pair: _CARRIER_ORDER[pair[0]])
    return [item for _, item in items]


def _all_record_hits(records: list[dict[str, Any]]) -> list[RecordHits]:
    return [record_hits(record) for record in records]


def _hits_to_reduce(
    records: list[dict[str, Any]] | None,
    hits: Sequence[RecordHits] | None,
) -> Sequence[RecordHits]:
    """Resolve what a per-class reader reduces: the caller's ``hits`` (one
    :func:`detectors.record_hits` result per record, in record order) when
    given, else a fresh pass over ``records``. Exactly one is required, as
    for :func:`_neighborhoods_to_partition`, so a caller reading several
    classes pays for one detector pass instead of one per class."""
    if hits is not None:
        return hits
    if records is None:
        raise TypeError('pass either records or hits=')
    return _all_record_hits(records)


def iter_self_corrections(
    records: list[dict[str, Any]] | None = None, *, hits: Sequence[RecordHits] | None = None,
) -> list[dict[str, Any]]:
    """Self-correction markers in assistant text blocks."""
    return _signal_items(_hits_to_reduce(records, hits), SignalClass.SELF_CORRECT)


def iter_not_found(
    records: list[dict[str, Any]] | None = None, *, hits: Sequence[RecordHits] | None = None,
) -> list[dict[str, Any]]:
    """Not-found literals in tool_result content."""
    return _signal_items(_hits_to_reduce(records, hits), SignalClass.NOT_FOUND)


def iter_df_guards(
    records: list[dict[str, Any]] | None = None, *, hits: Sequence[RecordHits] | None = None,
) -> list[dict[str, Any]]:
    """Guard trip literals in tool_results, assistant text and user turns."""
    return _signal_items(_hits_to_reduce(records, hits), SignalClass.DF_GUARD)


def iter_interrupts(
    records: list[dict[str, Any]] | None = None, *, hits: Sequence[RecordHits] | None = None,
) -> list[dict[str, Any]]:
    """The injected interrupt marker in non-sidechain user turns."""
    return _signal_items(_hits_to_reduce(records, hits), SignalClass.INTERRUPT)


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


def find_retry_loops(
    records: list[dict[str, Any]],
    *,
    neighborhoods: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Group tool_use calls by (name, canonical input signature), flag
    groups recurring >= RETRY_MIN times as near-identical retry loops, and
    ANNOTATE each with how many of its calls ended in a DESIGNED outcome.

    Grouping stays deterministic and dependency-free (sibling to the
    decoy-FAIL decision, PRD Sec 13.2): no fuzzy string similarity, no
    time or adjacency heuristic, just "same tool, same canonical-JSON
    input, again". The annotation adds no second classifier and no second
    pattern table -- it JOINS, on ``tool_use_id``, against the
    already-enriched :func:`iter_error_neighborhoods` scan that task 3610
    made the single source of truth for "was this result designed?".

    Why (confusion-census-2026-08-26 §1.1 / R1, task 4751): a healthy
    escalation-watcher rotation makes one date-check and one re-arm call
    per ~3600s cycle, so ANY rotation of >= RETRY_MIN cycles necessarily
    crossed this threshold and rendered under "## Retry Loops" -- while the
    same digest's ``signal_counts`` correctly reported ``tool_error: 0``
    and tallied the ceilings under ``designed_outcome``. Two layers of one
    instrument told contradictory stories about one session. Reading ONE
    classification is what stops that.

    Annotation, never suppression. A group is always returned at its true
    ``count``, so a genuine retry storm interleaved with bounded-wait
    ceilings stays fully visible and ambiguity still costs a reader one
    annotated line rather than a hidden failure (PRD Sec 7.2.1's
    fail-toward-genuine principle). A group with no designed results
    reports ``designed_outcome_count`` 0 and an empty ``designed_outcomes``,
    and renders byte-identically to the pre-4751 format.

    ``designed_outcomes`` holds the DISTINCT ``(exit_code, label)`` pairs
    in first-appearance order -- deterministic by construction over the
    group's already-ordered members, and needing no comparison key:
    ``exit_code`` is ``int | None``, so a naive ``sorted()`` would raise
    TypeError the moment a declaration carries no code.

    Pass ``neighborhoods=`` to annotate from a scan the caller already
    holds -- the same contract and the same resolver
    (:func:`_neighborhoods_to_partition`) as :func:`iter_genuine_errors`
    and :func:`iter_designed_outcomes`.
    """
    # `str | None` in the key, not `str`: _iter_tool_use_blocks selects on
    # `type == 'tool_use'` only, so a malformed block with no 'name' yields a
    # None here and always has. The declared type is corrected to match rather
    # than the value coerced, which would change what a nameless block renders
    # as in the report for no benefit.
    #
    # A member is one (record index, tool_use id) PAIR, accumulated once:
    # the two are needed for different things -- the index is a tool_use
    # RECORD position and so cannot address a neighborhood (whose own
    # `index` is the tool_result position), while the id is the only key
    # that can -- and keeping them in ONE list makes drifting them apart
    # structurally impossible rather than a lockstep invariant a later
    # edit could quietly break.
    groups: dict[tuple[str | None, str], list[tuple[int, Any]]] = {}
    for index, block in _iter_tool_use_blocks(records):
        key = (block.get('name'), _input_signature(block.get('input')))
        groups.setdefault(key, []).append((index, block.get('id')))

    # Only the DESIGNED half is indexed: a genuine failure needs no
    # annotation here (it has its own section), and restricting the map is
    # what makes a hit unambiguous. A None id is excluded so a malformed
    # id-less block cannot collide with an orphan neighborhood's None.
    designed_by_id = {
        n['tool_use_id']: n
        for n in _neighborhoods_to_partition(records, neighborhoods)
        if n['designed_outcome'] is not None and n['tool_use_id'] is not None
    }

    loops = []
    for (name, signature), members in groups.items():
        if len(members) < RETRY_MIN:
            continue
        designed_count = 0
        pairs: list[tuple[int | None, str]] = []
        seen: set[tuple[int | None, str]] = set()
        for _index, tool_use_id in members:
            hit = designed_by_id.get(tool_use_id)
            if hit is None:
                continue
            designed_count += 1
            pair = (hit['exit_code'], hit['designed_outcome'])
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        loops.append({
            'tool': name,
            'signature': signature,
            'count': len(members),
            'indices': [index for index, _id in members],
            'designed_outcome_count': designed_count,
            'designed_outcomes': pairs,
        })
    return loops


def signal_counts(
    records: list[dict[str, Any]] | None = None, *, hits: Sequence[RecordHits] | None = None,
) -> dict[str, int]:
    """The frontmatter's ``signal_counts`` (PRD Sec 7.2): every signal
    class's hit total over the transcript, 0 for an absent class.

    ``tool_error`` counts only GENUINE failures; declared-designed outcomes
    are tallied separately under ``designed_outcome`` (task 3610), so one
    real error is not buried among bounded-wait ceilings. The split is total
    and lossless: the two sum to the number of ``is_error`` results.
    """
    reduced = _hits_to_reduce(records, hits)
    return {signal.value: sum(record.count(signal) for record in reduced) for signal in SignalClass}


SIGNAL_WEIGHTS: dict[SignalClass, float] = {
    SignalClass.SELF_CORRECT: 3.0,
    SignalClass.DF_GUARD: 2.0,
    SignalClass.INTERRUPT: 2.0,
    SignalClass.TOOL_ERROR: 1.0,
    SignalClass.NOT_FOUND: 1.0,
}
"""Documented weights for :func:`score_signals`, one per
:data:`detectors.CONFUSION_CLASSES` member. Self-corrections (an explicit
"I was wrong" moment) are the strongest signal; df_guard/interrupt are
mid-weight structural signals; tool_error/not_found are the lowest-weight,
highest-frequency noise signals. Every weight is strictly positive, which
is what makes score_signals monotonic.

``designed_outcome`` is DELIBERATELY absent: a declared-designed outcome is
*reported* in ``signal_counts`` but contributes exactly zero to the
confusion score, because a bounded-poll ceiling is a designed
loop-continuation, not agent confusion. *Reported* is bounded by where a
report exists: the sampler drops a session scoring zero before any digest
is rendered (PRD Sec 7.2.1)."""

USER_TURN_WEIGHT: float = 5.0
"""Weight of one non-sidechain human user turn: gold (PRD Sec 5), outweighing
any single occurrence of any :data:`SIGNAL_WEIGHTS` class."""


def score_signals(counts: Mapping[str, int], n_user_turns: int) -> float:
    """Weighted-sum confusion score used by beta's sampler to rank sessions.

    Strictly monotonic in every input: since every weight is positive,
    incrementing any one count (or n_user_turns) by 1 strictly raises the
    score.
    """
    score = USER_TURN_WEIGHT * n_user_turns
    for signal, weight in SIGNAL_WEIGHTS.items():
        score += weight * counts.get(signal, 0)
    return score


def iter_user_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return genuine non-sidechain, non-meta human user turns.

    Excludes: non-'user' records, isSidechain=True (subagent) turns,
    isMeta=True (system-injected) turns, user records whose content is
    entirely tool_result blocks, a record whose harness provenance names a
    non-human producer (:func:`reingestion.has_non_human_origin` -- a background-task
    notification, an auto-continuation, a coordinator or peer message), and
    re-ingested machine content (:func:`reingestion.is_reingested_content`): a pasted
    coder judgment, and every harness-injected briefing/prompt/report/
    context-block turn -- the orchestrator briefing, the trickle-coder and
    resume prompts, the reconciliation judge's run-review prompt and a lone
    memory-context block alike (see :func:`reingestion.is_harness_injected_turn`). Every
    one of those injected shapes lands in the transcript as ordinary
    user-role text (isMeta unset), so isMeta alone cannot exclude any of
    them. The gold bucket asks the SAME predicate every scalar detector asks
    (:func:`detectors.is_non_dialogue_carrier`) rather than holding a private copy of
    the rule: task 5685's ruling that the fix belongs at the
    content-classification layer, not per bucket.

    This function is the SINGLE source for both the gold user_corrections
    section and render_digest's n_user_turns score component, so this one
    filter excludes such a turn from the body AND the score together --
    which is exactly what confusion-census-2026-07-31 §3.1 asks for, its
    clusters 1.1(b) and 1.2 being one event observed from two surfaces.
    User corrections are gold (PRD Sec 5) -- this is the highest-priority
    digest section.
    """
    turns = []
    for index, record in enumerate(records):
        if record.get('type') != 'user':
            continue
        if record.get('isSidechain'):
            continue
        if record.get('isMeta'):
            continue
        text = user_turn_text(record)
        if text is None:
            continue
        if is_non_dialogue_carrier(record, text):
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


DIGEST_INSTRUMENT_VERSION: int = 5
"""Which generation of this instrument produced a given digest.

BUMP POLICY: increment whenever a signal detector or the gold-turn
filter changes SEMANTICS -- i.e. whenever the same transcript would now
yield different signal_counts, a different gold section, or a different
section partition. Pure refactors, renderer cosmetics and new frontmatter
keys do NOT bump it.

  1 -- the implicit pre-3610 baseline. NEVER emitted: digests rendered
       before task 3610 landed carry no ``instrument_version`` key at
       all, and that ABSENCE is the "predates the filter change" signal a
       census reads. Digests are rendered on demand and cached only
       in-process (sampling.py, nightly.py), so there is no persisted
       corpus to migrate and absence is a free, unambiguous discriminator.
  2 -- the relaxed anchor+corroborator briefing filter
       (:func:`reingestion.is_harness_injected_turn`) plus the genuine/designed error
       split (:func:`iter_genuine_errors`).
  3 -- the re-ingested-content classifier (:func:`reingestion.is_reingested_content`,
       task 5685), consulted by every text-pattern detector and the
       gold-turn filter.
  4 -- the non-human-origin record rule (:func:`reingestion.has_non_human_origin`,
       task 5956), consulted with the content classifier by the gold-turn
       filter and every dialogue carrier.
  5 -- the merger's heading set (reingestion.MERGER_HEADINGS) and the rescoped
       memory-block markers, task 3659.

This answers ``plans/confusion-census-2026-07-31.md:151`` (Sec 6): the
next census must be able to tell a pre-fix trace from a live regression.
"""

FRONTMATTER_KEYS: tuple[str, ...] = (
    'session', 'cwd', 'encoded_dir', 'agent_class', 'date', 'size_bytes', 'score',
    'n_user_turns', 'truncated_items', 'instrument_version',
)
"""Top-level frontmatter keys in the exact PRD Sec 7.2 order (everything
before ``signal_counts``, which is rendered as its own nested block).

``n_user_turns`` follows ``score`` so the block reads score -> its dominant
5.0-weighted basis -> the remaining weighted counts. Rationale
(confusion-census-2026-07-31 R2 / Sec 1.2): :func:`score_signals` adds
``USER_TURN_WEIGHT * n_user_turns`` on top of the weighted
``signal_counts``, so rendering only the counts left the score
unreconstructible from the digest -- a session whose sole "user turn" was a
pasted report showed ``score: 5.0`` beside an all-zero ``signal_counts``.
It is deliberately NOT a sixth ``signal_counts`` entry: that block is
specifically :func:`signal_counts`' per-class detector hit counts (read by
:func:`_warn_if_body_evicted`'s ``any(counts.values())`` guard), and a
user-turn tally is neither.

``truncated_items`` follows it (confusion-census-2026-07-31 R1): the
per-item :data:`ITEM_TRUNCATION_MARKER` is legible only to a reader who
gets that far into the body, so the count is surfaced up front -- a reader
weighing the digest knows how many of its items have had substance capped
out of them. It counts what :func:`_cap_item` actually truncated (tracked
at source by :func:`_build_sections`, never re-derived by scanning the
rendered body for :data:`ITEM_TRUNCATION_MARKER`, which item content can
quote on its own), maintained over the POST-trim sections (see
:func:`_truncate_sections`), so it describes the body that actually
ships.

``instrument_version`` (task 3610) stays LAST of the top-level keys, and
that position is pinned by its own test: a generation marker belongs after
the measurements it describes, and appending rather than inserting is what
keeps downstream frontmatter diffs stable as the contract grows. Task
3614's two keys were added ahead of it, so the byte-identical prefix that
commit claimed now ends at ``score`` -- the append discipline it was
protecting is unchanged, only the prefix it applies to is longer."""

BARE_FRONTMATTER_KEYS: frozenset[str] = frozenset(
    {'size_bytes', 'score', 'n_user_turns', 'truncated_items', 'instrument_version'}
)
"""Top-level frontmatter keys rendered as bare YAML scalars; every other
key in FRONTMATTER_KEYS is double-quoted via :func:`_yaml_dquote`. Named
rather than inlined into :func:`render_frontmatter` so a further numeric
key is a one-word addition here instead of a growing inline literal."""

SIGNAL_COUNT_KEYS: tuple[SignalClass, ...] = tuple(SignalClass)
"""``signal_counts`` nested keys in the exact PRD Sec 7.2 order.

``designed_outcome`` is last (task 3610) so the pre-existing five-key
prefix stays byte-stable for anything diffing rendered frontmatter."""


def render_frontmatter(meta: dict[str, Any]) -> str:
    """Hand-render *meta* as a '---'-delimited YAML frontmatter block.

    Fixed key order (PRD Sec 7.2), explicit hand rendering rather than
    ``yaml.safe_dump`` -- deterministic, byte-stable output for exact tests
    and stable downstream diffs, and avoids a dumper's default surprises
    (key sorting, quoting, anchors, float formatting). String-valued fields
    are explicitly double-quoted (see :func:`_yaml_dquote`); numeric fields
    (:data:`BARE_FRONTMATTER_KEYS` and every signal_counts value) are
    emitted bare.
    """
    lines = ['---']
    for key in FRONTMATTER_KEYS:
        value = meta[key]
        if key in BARE_FRONTMATTER_KEYS:
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
    'designed_outcomes': 'Designed Outcomes',
}
"""Markdown '## ' heading text per digest section. The first four are the
PRD Sec 7.2 primary sections (verbatim prose order: user turns, error
neighborhoods, self-corrections, retry loops); the last three are the
secondary scalar signals, included only when present (PRD decomposition
Sec 11 alpha observable: "frontmatter + the four signal-class sections")."""

SECTION_PRIORITY: tuple[str, ...] = (
    'designed_outcomes',
    'retry_loops', 'not_found', 'error_neighborhoods', 'df_guard', 'interrupt',
    'self_corrections', 'user_corrections',
)
"""Section keys in ASCENDING signal priority -- index 0 is dropped/trimmed
FIRST under the 15KB soft cap, the last entry ('user_corrections', gold)
is trimmed LAST (PRD Sec 7.2: "truncate lowest-signal sections last"; PRD
Sec 5: user corrections are gold). Digest bodies render in the REVERSE of
this order (gold first). Ranking mirrors SIGNAL_WEIGHTS magnitude
(self_correct 3.0 > df_guard/interrupt 2.0 > tool_error/not_found 1.0),
with retry_loops below those since it carries no SIGNAL_WEIGHTS entry at
all (it is a structural section, not one of the 5 scored signal classes).

'designed_outcomes' sits at index 0 -- trimmed FIRST, below even
retry_loops -- because it is explicitly NOT confusion: a declared
bounded-poll ceiling is a designed loop-continuation (task 3610, 07-31
census cluster 1.3), reported for visibility and deliberately unweighted
in SIGNAL_WEIGHTS. When a digest must shed bytes, that is the first
thing a reader can afford to lose.

That ordering was once a hazard: a byte-pressured digest shed the section
EXPLAINING that churn was designed before it shed the section PRESENTING
that churn as loops, stranding the explanation (census 2026-08-26 §1.1).
It no longer is -- 'retry_loops' lines are now SELF-disambiguating,
carrying their own designed-outcome count and the rules that fired (see
:func:`_render_retry_loops`), so the fix is in place rather than in the
priority order and no re-prioritisation is needed."""


def _render_user_corrections(items: list[dict[str, Any]]) -> list[str]:
    return [f"- (turn {item['index']}) {item['text']}" for item in items]


def _exit_marker(exit_code: int | None) -> str:
    """Render the structured '[exit N] ' prefix, or '' when no code was
    extracted. Promoting the code out of the raw error prose is 07-31
    census R3's "record exit codes" ask: it keeps a bare 124 legible even
    when :func:`_cap_item` byte-truncates the content away."""
    return '' if exit_code is None else f'[exit {exit_code}] '


def _render_error_neighborhoods(items: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in items:
        tool = item['attempt_tool'] or 'unknown'
        summary = item['attempt_input_summary'] or ''
        marker = _exit_marker(item['exit_code'])
        lines.append(
            f"- (turn {item['index']}) {tool}({summary}) -> "
            f"{marker}{item['error_content']}"
        )
    return lines


def _render_designed_outcomes(items: list[dict[str, Any]]) -> list[str]:
    """Name the rule that fired plus the exit code, so a reader can tell a
    self-declared WATCHER_REARM_OUTCOME apart from a bare bounded-wait
    124 without re-deriving the classification."""
    lines = []
    for item in items:
        tool = item['attempt_tool'] or 'unknown'
        summary = item['attempt_input_summary'] or ''
        marker = _exit_marker(item['exit_code'])
        lines.append(
            f"- (turn {item['index']}) {tool}({summary}) -> "
            f"{marker}[{item['designed_outcome']}] {item['error_content']}"
        )
    return lines


def _render_self_corrections(items: list[dict[str, Any]]) -> list[str]:
    return [
        f"- (turn {item['index']}) [{item['pattern']}] {item['context']}"
        for item in items
    ]


def _render_retry_loops(items: list[dict[str, Any]]) -> list[str]:
    """Render each retry group, annotating it with how many of its calls
    ended in a DESIGNED outcome (census 2026-08-26 R1, task 4751).

    Two contracts this locks:

    * A group with ZERO designed results takes an explicit early branch
      (``if not designed: ... continue``) that emits the pre-4751 f-string
      verbatim -- byte-identity is a contract, not an accident of an
      annotation that happens to render empty, and spelling it as its own
      branch is what makes "genuine retry storms are untouched" a testable
      literal-f-string assertion rather than a claim about interpolation.
    * The annotation sits to the LEFT of the stable ``": {signature}"``
      terminator, so :func:`_cap_item` byte-truncates the signature first
      and leaves the annotation legible -- exactly the reason
      :func:`_exit_marker` promotes an exit code out of raw prose. The
      groups that most need explaining are the ones with the longest
      commands, i.e. precisely those the cap would otherwise eat.

    The count is always rendered ALONGSIDE the group's full ``count``,
    never in place of it: the annotation explains churn, it never hides
    how much churn there was. Each distinct ``(exit_code, label)`` pair is
    then named with the SAME ``[exit N] [label]`` spelling
    :func:`_render_designed_outcomes` uses, so a reader meets one
    vocabulary in both sections instead of re-deriving the classification
    from the raw command.
    """
    lines = []
    for item in items:
        designed = item['designed_outcome_count']
        if not designed:
            lines.append(f"- {item['tool']} x{item['count']}: {item['signature']}")
            continue
        rules = ', '.join(
            f'{_exit_marker(code)}[{label}]'
            for code, label in item['designed_outcomes']
        )
        lines.append(
            f"- {item['tool']} x{item['count']}"
            f' ({designed} designed-outcome results: {rules})'
            f": {item['signature']}"
        )
    return lines


def _render_scalar_signal(items: list[dict[str, Any]]) -> list[str]:
    return [f"- (turn {item['index']}) {item['pattern']}" for item in items]


@dataclass(frozen=True)
class _DigestScan:
    """What every section is detected from, computed once per build."""

    records: list[dict[str, Any]]
    hits: Sequence[RecordHits]
    neighborhoods: list[dict[str, Any]]


def _scan(records: list[dict[str, Any]], hits: Sequence[RecordHits] | None) -> _DigestScan:
    reduced = _hits_to_reduce(records, hits)
    return _DigestScan(records, reduced, _error_neighborhoods(records, [h.errors for h in reduced]))


_SectionDetector = Callable[[_DigestScan], list[dict[str, Any]]]
_ItemRenderer = Callable[[list[dict[str, Any]]], list[str]]

_SECTION_RENDERERS: dict[str, tuple[_SectionDetector, _ItemRenderer]] = {
    'user_corrections': (
        lambda scan: iter_user_turns(scan.records), _render_user_corrections,
    ),
    'error_neighborhoods': (
        lambda scan: iter_genuine_errors(neighborhoods=scan.neighborhoods),
        _render_error_neighborhoods,
    ),
    'self_corrections': (
        lambda scan: iter_self_corrections(hits=scan.hits),
        _render_self_corrections,
    ),
    'retry_loops': (
        lambda scan: find_retry_loops(scan.records, neighborhoods=scan.neighborhoods),
        _render_retry_loops,
    ),
    'not_found': (
        lambda scan: iter_not_found(hits=scan.hits), _render_scalar_signal,
    ),
    'df_guard': (
        lambda scan: iter_df_guards(hits=scan.hits), _render_scalar_signal,
    ),
    'interrupt': (
        lambda scan: iter_interrupts(hits=scan.hits), _render_scalar_signal,
    ),
    'designed_outcomes': (
        lambda scan: iter_designed_outcomes(neighborhoods=scan.neighborhoods),
        _render_designed_outcomes,
    ),
}
"""(detector, item-renderer) pair per section key, keyed identically to
SECTION_HEADINGS/SECTION_PRIORITY. The genuine and designed halves of the
error partition render in separate sections, so each neighborhood appears
exactly once in the body (task 3610)."""

ITEM_TRUNCATION_MARKER = '... [item truncated'
"""Stable, greppable PREFIX of the tail appended to a per-item line that
exceeded its byte cap; :func:`_truncation_suffix` completes it with the
quantified loss. Explicit and ASCII (so its own byte length equals its
character length), keeping degradation legible rather than silently lossy.

Deliberately a PREFIX rather than the whole literal: a fixed opaque marker
told a reader that something was dropped but never how much, so a 20KB
pasted turn capped to 2KB looked identical to one capped by 40 bytes
(confusion-census-2026-07-31 R1, and §3.4 "rendered surfaces that omit
their own basis"). Keeping the constant a prefix preserves every existing
``ITEM_TRUNCATION_MARKER in ...`` grep, test and scan while the tail
carries the numbers.

NOT a detection predicate: because it is now an open prefix it matches
strictly more text than the old closed literal, and the literal occurs
verbatim in this repo's own data (docs/legibility/confusion-codebook.yaml
evidence quotes, plans/confusion-census-2026-07-31.md), so a session that
merely reads or greps those files renders items containing the marker
without having been capped. ``truncated_items`` is therefore tracked from
what :func:`_cap_item` actually did (see :func:`_build_sections`), never
re-derived by scanning the rendered body."""

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
:func:`_item_byte_cap`). Measured frontmatter is ~318 bytes typical and
~397 with a long session/cwd/encoded_dir and 2-digit counts (task 3610
added ``instrument_version`` and ``designed_outcome``, ~30 bytes, to a
previously-measured ~276-310);
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


def _truncation_suffix(dropped: int, original: int) -> str:
    """Render the full item-truncation tail: the stable
    :data:`ITEM_TRUNCATION_MARKER` prefix plus how much of the item was
    discarded, so the degradation reports its own magnitude rather than
    only its existence."""
    return f'{ITEM_TRUNCATION_MARKER}: {dropped} of {original} bytes dropped]'


def _cap_item(line: str, cap: int) -> str:
    """Return *line* unchanged when its UTF-8 byte length is within *cap*;
    otherwise byte-truncate (never splitting a multi-byte codepoint) and
    append a :func:`_truncation_suffix` quantifying the loss.

    The within-cap return is the SAME object (``return line``, never a
    copy) -- a contract, not an accident: it makes ``capped is not line``
    the ground truth for "this item was actually truncated", which is how
    :func:`_build_sections` tallies ``truncated_items`` without re-deriving
    truncation by scanning rendered text for :data:`ITEM_TRUNCATION_MARKER`
    (which item content can contain on its own).

    Byte-wise because the whole digest budget is measured in UTF-8 bytes
    (:func:`_resolve_size_bytes`); ``decode(..., 'ignore')`` drops a
    partial trailing codepoint instead of emitting a U+FFFD replacement
    character, so a truncated item never introduces mojibake into text a
    downstream LLM coder reads verbatim.

    The suffix is self-referential -- its own byte length depends on the
    digit counts of the numbers it reports, which depend on how many bytes
    the suffix leaves room for -- so it is resolved by the same short
    bounded fixed point :func:`_resolve_size_bytes` already documents for
    ``size_bytes``: render against a guess (dropped=0), measure, re-render.
    Suffix length is monotone non-decreasing across passes (a longer suffix
    retains fewer bytes, which drops more, whose digit count can only grow)
    and is bounded by the digit count of *original*, so this converges in a
    couple of passes; the loop bound is a defensive cap, not a claim that
    more passes are ever needed.

    The result is ``<= cap`` bytes for any ``cap >= len(suffix)``: the
    retained slice is capped at ``cap - len(suffix)`` and decoding can only
    shorten it. :data:`MIN_ITEM_BYTES` (the floor :func:`_item_byte_cap`
    never returns below) comfortably exceeds the longest possible suffix,
    which is what makes that precondition structural.
    """
    encoded = line.encode('utf-8')
    original = len(encoded)
    if original <= cap:
        return line

    suffix = _truncation_suffix(0, original)
    for _ in range(4):
        keep = max(0, cap - len(suffix.encode('utf-8')))
        kept = encoded[:keep].decode('utf-8', 'ignore')
        resolved = _truncation_suffix(original - len(kept.encode('utf-8')), original)
        if resolved == suffix:
            return kept + suffix
        suffix = resolved

    # Unreached in practice (see the convergence argument above); recompute
    # the retained slice against the final suffix so the `<= cap` bound
    # holds even if the loop bound were somehow exhausted.
    keep = max(0, cap - len(suffix.encode('utf-8')))
    return encoded[:keep].decode('utf-8', 'ignore') + suffix


def _build_sections(
    records: list[dict[str, Any]],
    *,
    item_max_bytes: int = _item_byte_cap(15360),
    hits: Sequence[RecordHits] | None = None,
) -> tuple[dict[str, list[str]], set[tuple[str, int]]]:
    """Run every detector once and render its hits to markdown bullet
    lines, keyed by section key, capping each rendered line to
    *item_max_bytes* (see :func:`_cap_item`) at this single choke point --
    applied uniformly across all seven sections so ONE oversized item
    (e.g. a multi-KB pasted user turn, or a huge echoed tool_result) can
    never evict every sibling section during :func:`_truncate_sections`'
    trim loop. A section with zero hits maps to an empty list --
    render_digest skips emitting a heading for it.

    *hits* is the caller's :func:`detectors.record_hits` pass over
    *records*, when it has one (:func:`_hits_to_reduce`).

    Returns ``(sections, truncated)``, where *truncated* is the set of
    ``(section_key, item_index)`` positions :func:`_cap_item` actually
    byte-capped. Recording it HERE, at the only place truncation happens,
    is what makes ``truncated_items`` ground truth: re-deriving it by
    scanning rendered lines for :data:`ITEM_TRUNCATION_MARKER` would also
    count an item whose own content merely quotes the marker (this repo's
    codebook and census files both contain it verbatim, so any session
    that reads them produces such an item), inflating a frontmatter field
    whose whole purpose is to report its own basis honestly
    (confusion-census-2026-07-31 §3.4). Positions stay valid under
    :func:`_truncate_sections` because it only ever pops the TRAILING item
    of a section, never reindexes a surviving one.
    """
    scan = _scan(records, hits)
    sections: dict[str, list[str]] = {}
    truncated: set[tuple[str, int]] = set()
    for key, (detector, renderer) in _SECTION_RENDERERS.items():
        items = detector(scan)
        lines = []
        for index, line in enumerate(renderer(items)):
            capped = _cap_item(line, item_max_bytes)
            if capped is not line:  # _cap_item's documented identity contract
                truncated.add((key, index))
            lines.append(capped)
        sections[key] = lines
    return sections, truncated


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
    meta: dict[str, Any],
    sections: dict[str, list[str]],
    max_bytes: int,
    *,
    truncated: set[tuple[str, int]],
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

    *truncated* is :func:`_build_sections`' ground-truth set of
    ``(section_key, item_index)`` positions it byte-capped. It is
    maintained INCREMENTALLY here -- seeded once, and discarded from when
    a pop removes a position in it -- so ``truncated_items`` always
    describes the body being rendered rather than a pre-trim snapshot
    (popping a truncated item would otherwise leave the frontmatter
    claiming the shipped body contains a truncated item it does not).

    Incremental rather than recomputed per iteration for cost: this trim
    loop is already documented as super-linear in signal-item count
    (nightly.build_digests, sampling.digest_byte_cost_fn: "seconds to
    minutes per multi-MB session"), and rescanning every rendered line of
    every section on each of O(items) iterations would add a second
    Python-level quadratic term on top -- unlike :func:`_render_body` /
    :func:`_resolve_size_bytes`, whose per-iteration work is C-level
    join/encode. Discarding one set member per pop is O(1).

    The termination argument is unchanged: popping an item can only lower
    the count (never raise it), so the frontmatter never grows across
    iterations and the digest still shrinks monotonically.

    Returns the rendered digest AND the post-trim section dict, so a
    caller can inspect final section state (e.g. render_digest's
    :func:`_warn_if_body_evicted` consistency guard) without re-parsing
    the rendered markdown body.
    """
    sections = {key: list(lines) for key, lines in sections.items()}
    remaining = set(truncated)

    def _render(current: dict[str, list[str]]) -> str:
        return _resolve_size_bytes(
            {**meta, 'truncated_items': len(remaining)}, _render_body(current),
        )

    digest = _render(sections)

    while len(digest.encode('utf-8')) > max_bytes:
        target_key = next((key for key in SECTION_PRIORITY if sections.get(key)), None)
        if target_key is None:
            break
        # pop() removes the TRAILING item, so the popped position is the
        # last index -- and no surviving item is reindexed.
        remaining.discard((target_key, len(sections[target_key]) - 1))
        sections[target_key].pop()
        digest = _render(sections)

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
    :func:`detectors.classify_agent_class`'s result) -- render_digest never
    classifies on its own. *path*, when given, sources ``encoded_dir``
    from the transcript file's real parent dir name; otherwise it falls
    back to mirror-encoding the derived ``cwd`` (see
    :func:`_derive_encoded_dir`). When the unrestricted render exceeds
    *max_bytes*, :func:`_truncate_sections` trims lowest-signal sections
    first (gold user-corrections last); a digest already under the cap is
    returned unmodified (zero trim iterations).
    """
    cwd = _derive_cwd(records)
    hits = _all_record_hits(records)
    counts = signal_counts(hits=hits)
    # Bound ONCE and used for both the rendered key and the scored
    # argument, so the frontmatter's stated basis and the score computed
    # from it can never diverge (confusion-census-2026-07-31 R2).
    n_user_turns = len(iter_user_turns(records))
    meta = {
        'session': _derive_session(records),
        'cwd': cwd,
        'encoded_dir': _derive_encoded_dir(records, cwd, path),
        'agent_class': agent_class,
        'date': _derive_date(records),
        'score': score_signals(counts, n_user_turns),
        'n_user_turns': n_user_turns,
        # Seeded so render_frontmatter is never called on a meta missing
        # the key; _truncate_sections resolves the real value against the
        # POST-trim sections it is about to render.
        'truncated_items': 0,
        'instrument_version': DIGEST_INSTRUMENT_VERSION,
        'signal_counts': counts,
    }

    sections, truncated = _build_sections(
        records, item_max_bytes=_item_byte_cap(max_bytes), hits=hits,
    )
    digest, trimmed_sections = _truncate_sections(
        meta, sections, max_bytes, truncated=truncated,
    )
    _warn_if_body_evicted(meta, trimmed_sections, max_bytes)
    return digest


def build_digest(
    path: Any, *, agent_class_override: str | None = None, max_bytes: int = 15360,
) -> str:
    """Orchestrate the full pipeline for a single transcript file: parse,
    classify, render. This is the headline entry point (PRD Sec 8.1
    boundary test, producer side) -- what beta/delta and :func:`main` call.

    *agent_class_override*, when given, is rendered verbatim (e.g. beta's
    already-authoritative stratum); otherwise
    :func:`detectors.classify_agent_class` decides.
    """
    records = load_transcript(path)
    agent_class = (
        agent_class_override if agent_class_override is not None
        else classify_agent_class(Path(path), records)
    )
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
        help='Render this agent_class instead of classifying the transcript',
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
