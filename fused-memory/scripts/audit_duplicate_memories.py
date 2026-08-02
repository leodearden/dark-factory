#!/usr/bin/env python3
"""Corpus-health sweep: find near-duplicate Mem0 memories for a project,
report (or delete) the losers, and emit a metric series for the eval program.

Motivation: Stage-1 reconciliation (finding 2cf1b99f) observed the
worktree-local-venv-vs-shared-checkout-venv gotcha rewritten as a
near-duplicate ``procedural_knowledge`` memory >=13 times -- task-worker
agents write the gotcha ad hoc without first ``search()``-ing Mem0 for
existing coverage, so it recurs faster than any single consolidation pass
absorbs it. This script is the automated backstop.

Three detectors, all reported -- only two may DELETE
----------------------------------------------------
Candidate generation is dual-path; the transitive closure
(``cluster_memories_by_pairs``) is shared, so the two can never disagree
about what a cluster IS -- only about which pairs are candidates.

  - LEXICAL: an O(n^2) ``difflib.SequenceMatcher.ratio()`` scan, tuned by
    ``--threshold``. High precision on the near-verbatim rewrites that
    dominate this corpus, structurally blind to paraphrase.
  - ANN: each record's OWN stored Qdrant vector (fetched by
    ``scroll_by_metadata(..., with_vectors=True)``) used as the query vector
    for ``query_points``. Catches same-meaning/different-wording pairs the
    lexical scan cannot, at zero embedding API calls and with no second
    metric space.

The ANN path is NOT a replacement -- swapping one detector for the other
would lose signal, so the actioned plan clusters over the UNION and every
emitted cluster records which path(s) found it (``found_by``,
``lexical_clustered``, ``ann_max_score``, ``lexical_max_ratio``).

  - LIVENESS-SNAPSHOT RECURRENCE: point-in-time liveness observations that
    re-assert one identical core FACT about the same task, grouped by
    ``(category, subject_task_id, core_fact)``. Keyed on EXACT equality of
    that fact, not on similarity -- there is no threshold to tune. It is
    REPORT-ONLY: its groups feed neither ``delete_candidates`` nor
    ``apply_delete_candidates`` and are never handed to the apply gate, so
    ``--apply`` can never act on them.

Why a third class rather than a lower threshold: measured on the four
solar_challenge records that motivated it (tasks 94/96, Stage-1 finding
724e7be4, run e149def2), the pairwise lexical ratios between the three
liveness snapshots are 0.117-0.233 against a 0.85 threshold. The repeated
core status fact is a single clause; the unique investigative payload
(file/symlink checks, git-HEAD/reflog analysis, per-cycle grep/glob results)
is several times longer and swamps every similarity metric. No cutoff those
records could reach is settable without flooding the corpus with false
clusters -- and under ``--apply`` a false cluster is an irreversible delete.
This is not near-duplicate CONTENT; it is one FACT re-asserted at different
timestamps, a different equivalence relation needing its own detector.

Ownership seam: task 3098 authorizes DETECTION of this pattern only. The
disposition of the specific existing solar_challenge entries is a separate
human-gated decision on solar_challenge task 99.

The ANN cutoff is READ, never invented -- and read PER CATEGORY
---------------------------------------------------------------
It comes from ``config.write_triage`` -- a calibration OUTPUT derived by
``scripts/calibrate_write_triage.py`` from measured similarity
distributions, with ``calibration_report_path`` recording its provenance.
Task 3357 made that calibration per-category, so the resolution is too, in
strict priority per swept category:

  1. ``--ann-threshold``: an explicit operator number, for every category.
  2. ``config.write_triage.t_high_by_category[category]``: the cutoff
     measured on THAT category's own labeled pairs.
  3. ``config.write_triage.t_high``: the POOLED cutoff, used as a disclosed
     fallback and counted as ``ann_pooled_fallback_categories``. The pooled
     corpus is overwhelmingly one category, so a fallback category runs on
     evidence gathered elsewhere -- weaker, hence counted, hence withheld
     from ``--apply`` deletion (see ``restrict_delete_candidates_for_apply``).
  4. Nothing: that category is ABSENT from the resolved mapping -- its ANN
     path is DISABLED, logged at ERROR and counted as
     ``ann_disabled_uncalibrated``. The lexical path still runs.

A category with no derivable cutoff is a real outcome, not a defect: the
committed calibration refuses one for a category with no negative pairs to
separate against, and that refusal is itself the recorded finding. A
plausible stand-in cutoff would silently mis-cluster the corpus -- and
under ``--apply`` that means wrong deletions.

Every cap is counted
--------------------
The ANN path can lose candidates where a full scan cannot, so each loss is a
metric rather than invisible recall: ``top_k_saturated``,
``below_threshold_dropped``, ``unknown_neighbor_dropped``, ``missing_vector``,
``cross_category_dropped``, ``unswept_category_dropped``, ``ann_query_errors``,
per-category ``scan_truncated``, ``ann_disabled_uncalibrated``, and
``ann_pooled_fallback_categories`` (not a lost candidate but a weaker one:
how many categories ran on a cutoff measured on a different population). The
liveness path adds ``liveness_snapshot_untasked`` (a recognised snapshot that
resolves to no subject task, so no group can hold it) and
``liveness_snapshot_unfielded`` (point-in-time framing whose live fields could
not be read).

Only chain-free evidence may DELETE
-----------------------------------
The report shows every cluster both detectors found; ``--apply`` acts on a
strict subset. A cluster contributes deletions only when its lexical pairs
SPAN it (what this detector actioned before the ANN path existed) or its ANN
sub-graph is a CLIQUE. Otherwise a member could be in the cluster purely by
transitivity — A~B and B~C above the cutoff while A~C is far below it — and
under ``--apply`` every non-survivor of that chain is an irreversible delete.
Withheld clusters are reported (``apply_withheld_groups``), never silently
dropped. See ``restrict_delete_candidates_for_apply``.

Scope: all three Mem0 categories (``procedural_knowledge``,
``preferences_and_norms``, ``observations_and_summaries``). Clustering runs
once PER category, so a cross-category union is structurally impossible
rather than merely unlikely. Graphiti-backed categories are out of scope:
this detector reads the Qdrant collection and a Graphiti edge has no vector
here to cluster on.

Artifacts: one M1 metric series per run at
``<--metrics-root>/<eval_id>/metrics-<STAMP>.json`` (default root
``fused-memory/data/memory-evals/``, eval_id ``e6-corpus-health``), with a
human-readable ``report-<STAMP>.txt`` and a ``details-<STAMP>.json`` carrying
the cluster-size histogram, per-topic accretion table, per-event
consolidation table and per-group liveness-recurrence table. Schema lives in
``shared.memory_eval_metrics`` and is never restated here. The stdout plan
additionally carries ``liveness_snapshot_recurrence_groups``,
``liveness_snapshot_recurrence_clusters`` and ``liveness_snapshot_disclosure``
alongside the pre-existing keys, which keep their exact meaning and shape.
Suppress the artifacts with ``--no-metrics``.

ORDERING NOTE -- scheduling and gate-filing belong to task 3136, not here.
This script is a detector with a stable CLI and a stable stdout contract;
3136 owns when it runs and what is done with the report. 3136's timer may
subsume the eval program's leaf-epsilon invocation, so NEITHER owns the only
schedule: adding a second trigger here would create a duplicate, and
assuming 3136 always fires would leave the eval series with holes.

Structural parallel to ``scripts/audit_duplicate_tasks.py`` (union-find
near-duplicate clustering + pick_survivor + dry-run/apply split) and the Mem0
sweep-script family (``scripts/prune_recon_cycle_summaries.py``,
``scripts/sweep_orphan_flag_markers.py``): enumerate via
``memory.mem0.scroll_by_metadata``, delete losers via
``memory.delete_memory`` best-effort.

Read-only against the live corpus except for ``--apply`` deletions and the
artifact directory (memory-eval PRD §7).

Safety carve-outs:
  - Dry-run report is the default; deletion only under explicit ``--apply``.
  - Only near-duplicate CLUSTERS above a high similarity threshold are
    actioned; a survivor (canonical-flagged, else oldest) is always retained
    per cluster.
  - ``--apply`` refuses on an empty scan, on a plan with nothing left to
    delete after the gate, and when ANY category's scan looks truncated -- a
    capped window may have the survivor just outside it, so "delete the
    losers" could delete the last copy.
  - Clusters held together only by an ANN chain are withheld from ``--apply``
    (see "Only chain-free evidence may DELETE" above).
  - A cluster whose members all share one ``metadata.topic`` is task 3136's
    Option-C carve-out: an expected shape, never a delete candidate.
  - Liveness-snapshot recurrence groups are report-only BY CONSTRUCTION --
    they are never added to the delete lists, rather than filtered out of
    them downstream where a later refactor could drop the filter.
  - Missing/unextractable content degrades to ``''``, which never clusters
    and is never deleted.

Usage
-----
  # Dry run (default): print JSON report + write metrics, delete nothing.
  python scripts/audit_duplicate_memories.py --project-id dark_factory

  # Commit the deletions.
  python scripts/audit_duplicate_memories.py --project-id dark_factory --apply

  # Tune the LEXICAL threshold (default 0.85); the ANN cutoff is calibrated.
  python scripts/audit_duplicate_memories.py --project-id dark_factory \\
      --threshold 0.80
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from fused_memory.reconciliation.task_filter import (
    LIVE_TASK_STATUS_RE,
    POINT_IN_TIME_CHECK_RE,
    TASK_REF_RE,
)

logger = logging.getLogger('audit_duplicate_memories')


# ---------------------------------------------------------------------------
# Pure-function core (no I/O — fully testable without a live Mem0)
# ---------------------------------------------------------------------------

def _sort_groups_deterministically(groups: list[list[dict]]) -> list[list[dict]]:
    """Return a new list of groups in deterministic order without mutating *groups*.

    Members within each returned group are sorted by ``str(id)``; the list of
    groups is then sorted by the minimum id (as a string) in each group.
    Unlike Taskmaster task ids (mostly-numeric, handled via ``_id_as_int`` in
    ``audit_duplicate_tasks.py``), Mem0 memory ids are typically UUIDs, so a
    plain string sort key is used instead.
    """
    sorted_groups = [sorted(g, key=lambda m: str(m.get('id', ''))) for g in groups]
    sorted_groups.sort(key=lambda g: str(g[0].get('id', '')))
    return sorted_groups


def cluster_memories_by_pairs(
    memories: list[dict],
    pairs: Iterable[tuple[int, int]],
) -> list[list[dict]]:
    """Transitive closure over candidate *pairs* → deterministic memory groups.

    The shared clustering core. Candidate GENERATION is pluggable — the
    lexical ``difflib`` scan in ``find_near_duplicate_memory_groups`` and the
    ANN neighbour scan both emit ``(i, j)`` index pairs into this one
    function — but the closure itself has a single implementation. Two
    union-finds that must agree is a defect waiting to happen; this is the
    one site.

    Args:
        memories: Memory dicts. Indices in *pairs* address this list.
        pairs: Any iterable (list, generator, ...) of ``(i, j)`` index pairs
            deemed near-duplicates by some candidate generator. Duplicated
            and mirrored pairs are harmless — union is idempotent.

    Returns:
        List of groups (each a list of >= 2 memories) formed by transitive
        closure of *pairs*; singletons are dropped. Groups are sorted by the
        minimum id within the group so output is deterministic. Members are
        the caller's own dict objects (passed through by identity, not
        copied). Does not mutate *memories*.
    """
    n = len(memories)
    if n < 2:
        return []

    # Union-find (path-compressed) over memory indices.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in pairs:
        union(i, j)

    # Materialise groups: collect memory lists per root, drop singletons.
    groups: dict[int, list[dict]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(memories[i])

    result = [g for g in groups.values() if len(g) >= 2]
    return _sort_groups_deterministically(result)


def find_near_duplicate_memory_groups(
    memories: list[dict],
    threshold: float = 0.85,
) -> list[list[dict]]:
    """Find groups of memories with near-duplicate content using SequenceMatcher.

    Args:
        memories: Memory dicts (each with at least an ``'id'`` and
            ``'content'`` key). No category filtering is performed here —
            the caller (``build_sweep_plan``) is responsible for that.
        threshold: Minimum ``difflib.SequenceMatcher.ratio()`` to flag a pair.

    Returns:
        List of groups (each a list of >= 2 memories) formed by transitive
        closure of all pairs whose content similarity >= threshold. Groups
        are sorted by the minimum id within the group so output is
        deterministic. Does not mutate *memories*.

    Complexity:
        O(n^2) pairs x O(L) per ``SequenceMatcher.ratio()`` call (L = content
        length) — mirrors ``audit_duplicate_tasks.find_near_duplicate_groups``.
        Each pair is pre-filtered through ``quick_ratio()``, a documented
        upper bound on ``ratio()`` that is much cheaper to compute, so
        obviously-dissimilar pairs skip the expensive exact comparison
        without changing which pairs end up >= threshold.
    """
    n = len(memories)
    if n < 2:
        return []

    return cluster_memories_by_pairs(
        memories, _lexical_pairs(_normalized_contents(memories), threshold),
    )


def _normalized_contents(memories: list[dict]) -> list[str]:
    """The exact text the lexical path compares — ONE definition.

    Both lexical entry points (``find_near_duplicate_memory_groups`` and
    ``build_sweep_plan``) normalise here rather than each inlining the
    expression. ``_lexical_pairs``' empty-content guard and
    ``_max_lexical_ratio``'s reporting both key off this precise
    normalisation, and the plan prints the two paths' verdicts side by side —
    so a change made at one of two copies would silently desynchronise
    numbers a reader is invited to compare.
    """
    return [(m.get('content') or '').strip().lower() for m in memories]


def _lexical_pairs(
    normalized: list[str],
    threshold: float,
) -> Iterator[tuple[int, int]]:
    """Yield index pairs whose normalised contents are >= *threshold* similar.

    The scores are discarded — see :func:`_lexical_pairs_scored`, which is the
    single implementation both share.
    """
    for i, j, _ratio in _lexical_pairs_scored(normalized, threshold):
        yield (i, j)


def _lexical_pairs_scored(
    normalized: list[str],
    threshold: float,
) -> Iterator[tuple[int, int, float]]:
    """Yield ``(i, j, ratio)`` for pairs >= *threshold* similar.

    The lexical candidate GENERATOR, split out from the clustering core it
    feeds (``cluster_memories_by_pairs``). Behaviour is unchanged from when
    this loop was inline: same O(n^2) scan, same ``quick_ratio`` pre-filter,
    same empty-content guard.

    The exact ``ratio()`` is yielded rather than thrown away because
    ``build_sweep_plan`` reports a per-cluster ``lexical_max_ratio``: without
    it, ``_max_lexical_ratio`` would recompute from scratch precisely the
    comparisons this scan already paid for.
    """
    n = len(normalized)
    for i in range(n):
        # Empty/blank content never clusters and is never deleted (safe
        # degradation). Guard here because SequenceMatcher(None, '', '').ratio()
        # returns 1.0, which would otherwise union two empty-content records
        # and mark one for deletion — a real memory whose content simply could
        # not be extracted. See the docstring safety carve-out.
        if not normalized[i]:
            continue
        # Reuse one SequenceMatcher across the inner loop instead of
        # constructing a fresh one per pair (set_seq1 is fixed per i;
        # set_seq2 varies per j — identical argument order to the original
        # difflib.SequenceMatcher(None, normalized[i], normalized[j])).
        matcher = difflib.SequenceMatcher()
        matcher.set_seq1(normalized[i])
        for j in range(i + 1, n):
            if not normalized[j]:
                continue
            matcher.set_seq2(normalized[j])
            # quick_ratio()/real_quick_ratio() are documented upper bounds on
            # ratio(), so skipping pairs below threshold here never changes
            # which pairs end up unioned — only cheaper on obvious misses.
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold:
                yield (i, j, ratio)


# The live task-table fields a liveness snapshot reports, enumerated ONCE and
# deliberately the SAME four `task_filter.LIVE_TASK_STATUS_RE` gates on, so
# "what counts as a live field" keeps one definition shared with the gate that
# forces these observations into their durable point-in-time form.
#
# That alignment is now LOAD-BEARING, not merely tidy: both call sites below
# prefilter on `LIVE_TASK_STATUS_RE`, which is only semantics-preserving while
# every name here is one the imported gate also recognises in assignment form.
# Adding a field here without the matching `task_filter` change would silently
# start DROPPING real snapshots rather than counting them — the parametrized
# premise guard in `TestLivenessDetectorRegexBudget` exists to catch exactly
# that edit.
_LIVE_FIELD_NAMES: tuple[str, ...] = (
    'status',
    'claimant_run_id',
    'heartbeat_at',
    'pid',
)

# ``<field> = <value>``, as EITHER a fully-quoted value or a bare token — never
# a half-read mixture of the two.
#
# The quoted branch is deliberately all-or-nothing. An earlier single-branch
# form (``"?([A-Za-z0-9_./:+-]+)"?``) let the value class terminate INSIDE a
# quoted string, so `status="in progress"` silently truncated to the key
# `status=in` — a key that groups with nothing real, would group with any other
# value truncating to the same prefix, and fires NEITHER disclosure counter.
# That is exactly the invisible recall loss the disclosure design in this file
# exists to prevent. Requiring the closing quote makes a value that cannot be
# read whole fail to match at all, so it falls through to the counted
# ``liveness_snapshot_unfielded`` branch instead of becoming a bad key.
#
# `|` is excluded from the quoted branch because it is the key's own pair
# delimiter: admitting it would let one value forge two pairs. Such a value
# also reads as unfielded rather than silently corrupting the key.
#
# The bare branch's class stops at the punctuation these observations wrap
# their fields in, so `null,` and `null)` read as `null` rather than carrying
# the delimiter into the key; the characters it does admit (`-`, `.`, `/`,
# `:`, `+`) are the ones real values need — `in-progress`, a path, an ISO
# timestamp.
_LIVE_FIELD_ASSIGNMENT_RE = re.compile(
    r'\b(' + '|'.join(_LIVE_FIELD_NAMES) + r')\s*=\s*'
    r'(?:"([^"|\n]+)"|([A-Za-z0-9_./:+-]+))',
    re.IGNORECASE,
)


def _classify_liveness_snapshot(content: str) -> tuple[bool, str | None]:
    """The ONE liveness-snapshot classifier. Both call sites route through here.

    Deliberately single-copy. The extractor below and
    ``find_liveness_snapshot_recurrences`` need the same two gates but consume
    different parts of the verdict, and an earlier shape had each re-apply the
    gates itself — so the extractor's tests could keep passing while the
    shipped detector diverged. One function means those tests exercise the
    code production actually runs.

    Args:
        content: Raw memory content. Never None — callers coalesce.

    Returns:
        ``(framed, core_fact)``:

        * ``(False, None)`` — not a liveness snapshot, and not a loss either:
          the content never entered this detector's scope.
        * ``(True, None)`` — point-in-time framing over a live-status marker
          whose fields could not be read whole. THE counted
          ``liveness_snapshot_unfielded`` loss; the caller owns that counter,
          so this function stays a pure classifier.
        * ``(True, key)`` — a snapshot, with its canonical key.

        ``POINT_IN_TIME_CHECK_RE`` is consulted AT MOST ONCE per call, and not
        at all when the cheap prefilter rejects — the budget
        ``TestLivenessDetectorRegexBudget`` pins.
    """
    # Cheap gate FIRST. ``POINT_IN_TIME_CHECK_RE``'s two ``(?=.*...)``
    # lookaheads under ``re.DOTALL`` make it quadratic in content length —
    # measured 2.2 ms at 424 chars rising to 794 ms at 4240 on NON-matching
    # input, versus 0.015-0.35 ms for the lookahead-free
    # ``LIVE_TASK_STATUS_RE`` — and a non-match is the overwhelmingly common
    # case: this detector always runs, with no flag to disable it, over a
    # default 5000-records-per-category scan of mostly ordinary prose.
    # The gate is SOUND because ``_LIVE_FIELD_ASSIGNMENT_RE`` is
    # ``LIVE_TASK_STATUS_RE``'s first alternative plus extra obligations, so
    # an assignment hit implies a live-status hit: a record failing this check
    # could never have yielded a key anyway. A record it rejects reports
    # ``framed=False`` even if the point-in-time framing would have matched —
    # which is the intended contract, since such a record is out of scope
    # rather than a loss.
    if not LIVE_TASK_STATUS_RE.search(content):
        return False, None
    if not POINT_IN_TIME_CHECK_RE.search(content):
        return False, None

    pairs: set[str] = set()
    for field, quoted, bare in _LIVE_FIELD_ASSIGNMENT_RE.findall(content):
        # Exactly one branch ever participates; `' '.join(split())` normalises
        # the whitespace a quoted value may now legitimately carry (and is a
        # no-op on a bare token).
        cleaned = ' '.join((quoted or bare).split()).rstrip('./:+-').lower()
        if cleaned:
            pairs.add(f'{field.lower()}={cleaned}')
    if not pairs:
        return True, None
    return True, '|'.join(sorted(pairs))


def extract_liveness_snapshot_fact(record: dict) -> str | None:
    """Canonical key for the core FACT a point-in-time liveness snapshot asserts.

    A third equivalence relation, alongside the lexical and ANN paths: not
    "reads alike" but "asserts the same live fields". Two snapshots taken days
    apart share one clause verbatim and differ everywhere else, so no
    similarity threshold reachable by them could be set without flooding the
    corpus with false clusters — and under ``--apply`` a false cluster is an
    irreversible delete. Keying on an exact repeated fact instead introduces
    no tunable at all.

    A thin projection of ``_classify_liveness_snapshot`` — the shared
    classifier ``find_liveness_snapshot_recurrences`` also runs — so this
    public entry point cannot drift from the shipped detector.

    Classification requires BOTH markers:

    * ``POINT_IN_TIME_CHECK_RE`` — the timestamped "liveness check performed
      <date>" framing. Requiring it is not incidental: gate 2628
      (``task_filter.frames_live_task_status_as_current_fact``) BLOCKS the
      bare current-fact framing in the four gated categories, so every
      liveness observation written since it landed necessarily carries this
      framing. Requiring it therefore costs no recall on the corpus this
      detector sweeps.
    * At least one extractable ``<live_field>=<value>`` assignment. A
      ``LIVE_TASK_STATUS_RE`` paraphrase ("actively driven by") names no
      readable fields, so there is no fact to key on.

    What the framing requirement does NOT do is exclude every non-liveness
    use of ``status=``. ``POINT_IN_TIME_CHECK_RE``'s cheapest alternative is a
    bare ``as of <digit>``, which ordinary prose satisfies routinely, so a
    config note like "As of 2026-07-24 the dispatcher config has
    status=enabled for task 12" DOES classify here (pinned by
    ``test_as_of_prose_with_an_incidental_assignment_still_classifies``). The
    gate is a recall-preserving filter, not a precision guarantee. Over-firing
    is bounded by construction rather than by the pattern: this class is
    report-only, and a subject needs >= 2 records sharing an IDENTICAL key
    before anything is emitted, so an incidental hit falls out as a singleton.

    Returns:
        The canonical key — distinct ``field=value`` pairs, values lowercased
        with the delimiter and any trailing punctuation stripped, sorted (by
        field name, then value) and joined on ``|`` — or None when *record* is
        not a point-in-time liveness snapshot. Compared by EXACT string
        equality downstream: a snapshot reporting a different status produces
        a different key and never groups with this one.
    """
    return _classify_liveness_snapshot(record.get('content') or '')[1]


def liveness_snapshot_subject_task_ids(record: dict) -> set[str]:
    """Which task(s) a liveness snapshot is ABOUT, as a set of ``str`` ids.

    The union of three sources, not ``metadata.task_id`` alone, despite the
    pattern reading as "repeated snapshots of the same task". Measured against
    the live corpus: memory 1eef7df7 is filed under ``task_id='99'`` while its
    content re-verifies tasks 94 AND 96 (it carries
    ``related_task_ids=['94', '96']``). Under a strict ``metadata.task_id``
    grouping the buckets are 94={6b245659}, 96={08aa0017}, 99={1eef7df7} —
    every one a singleton, leaving the detector inert on the exact corpus that
    motivated it.

    Risk posture: a content ref can over-attribute an incidental mention. That
    is acceptable ONLY because this class is report-only and a subject needs
    >= 2 records sharing an IDENTICAL core fact before anything is emitted, so
    an over-attributed single mention falls out as a singleton. Over-firing
    costs a reviewer one glance, never data.

    Every id is coerced to ``str`` and blanks are dropped — the project-wide
    convention ``memory_service._normalize_task_id_metadata`` enforces at every
    write boundary, so an int ``94`` and the string ``'94'`` are one subject
    rather than two. A ``related_task_ids`` that is not a collection degrades
    without raising: a scalar contributes itself, anything else contributes
    nothing. Does not mutate *record*.
    """
    metadata = record.get('metadata') or {}
    subjects: set[str] = set()

    def _add(value: Any) -> None:
        text = str(value).strip() if value is not None else ''
        if text:
            subjects.add(text)

    _add(metadata.get('task_id'))

    related = metadata.get('related_task_ids')
    if isinstance(related, list | tuple | set):
        for item in related:
            _add(item)
    elif isinstance(related, str | int):
        _add(related)

    for ref in TASK_REF_RE.findall(record.get('content') or ''):
        _add(ref)

    return subjects


# The three Mem0-backed categories, enumerated ONCE. Graphiti-backed
# categories (entities_and_relations, temporal_facts, decisions_and_rationale)
# are deliberately absent: this detector reads the Mem0/Qdrant collection, and
# a Graphiti edge has no vector here to cluster on.
_ALL_CATEGORIES: tuple[str, ...] = (
    'procedural_knowledge',
    'preferences_and_norms',
    'observations_and_summaries',
)


# Every way the ANN candidate path can lose a candidate, enumerated ONCE so
# the zero-filled disclosure dict has a single definition. The metrics layer
# emits one metric per key; an absent key downstream is indistinguishable
# from "not measured", so these are always all present, always integers,
# always zero-filled on a clean run.
_ANN_DISCLOSURE_KEYS: tuple[str, ...] = (
    'top_k_saturated',
    'below_threshold_dropped',
    'unknown_neighbor_dropped',
    'missing_vector',
)


# Every way the liveness-recurrence path can lose a candidate, enumerated ONCE.
# Kept as a THIRD sibling of _ANN_DISCLOSURE_KEYS / _PLAN_DISCLOSURE_KEYS
# rather than folded into either, so each generator's disclosure contract stays
# its own. Same contract as its siblings: every key always present, always an
# int, always zero-filled on a clean run.
_LIVENESS_DISCLOSURE_KEYS: tuple[str, ...] = (
    'liveness_snapshot_untasked',
    'liveness_snapshot_unfielded',
)


def find_liveness_snapshot_recurrences(
    memories: list[dict],
    *,
    categories: Iterable[str] = _ALL_CATEGORIES,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Group point-in-time liveness snapshots that re-assert one core FACT.

    The third candidate class, and the only one that is REPORT-ONLY: its
    groups never reach ``delete_candidates``. Task 3098 authorizes DETECTION
    of this pattern, not disposition of the records that exhibit it.

    Buckets by ``(category, subject_task_id, core_fact)`` and emits every
    bucket with >= 2 members — the same >= 2 rule ``cluster_memories_by_pairs``
    already applies, so "what counts as a group" does not acquire a second
    meaning. Grouping is PER CATEGORY for the same reason ``build_sweep_plan``
    clusters per category: a preference and an observation that happen to
    report the same live fields are different kinds of knowledge, and unioning
    them would be cross-store conflation rather than deduplication.

    A record contributes to one bucket per subject it names, so a
    re-verification covering two tasks joins both their groups — which is
    exactly how memory 1eef7df7 links to the earlier 94 and 96 snapshots.

    KNOWN LIMITATION, deliberate: the core fact is built ONCE PER RECORD by
    unioning every assignment in its content, then attributed to every subject
    that record names. When a multi-task re-verification reports DIVERGENT
    values per task, that merged key matches neither single-task snapshot and
    the recurrence goes unreported — pinned, so the behaviour is visible
    rather than discovered later, by
    ``test_divergent_per_task_statuses_do_not_group``. The real motivating
    record (1eef7df7) reports identical values for both its subjects, so the
    detector fires on the corpus that motivated it; a future divergent one
    would be a RECALL gap in a report-only path, never a wrong delete.

    Per-subject key scoping was tried and rejected on evidence, not taste.
    Splitting content into per-task clauses with
    ``task_filter._CLAUSE_SPLIT_RE`` and keying each subject on its own clause
    (the technique ``task_filter.find_conflicting_task_status_ids`` uses)
    yields the EMPTY SET on all four real records: ``_CLAUSE_SPLIT_RE`` splits
    on ``.``, which shatters ``dark-factory-orchestrator.yaml`` and
    ``CLAUDE.md:95`` mid-sentence, and the nearby reference is written
    ``task/94``, which ``TASK_REF_RE``'s ``\\s*#?\\s*`` separator does not
    match. Adopting it would trade a bounded, pinned recall gap for a detector
    inert on its own motivating corpus.

    Args:
        memories: Raw memory list (any/all categories).
        categories: Categories to sweep. Records outside the set are ignored
            entirely — they were never in scope, so they are not a loss.

    Returns:
        ``(groups, disclosure)``. Each group carries ``subject_task_id``,
        ``category``, ``core_fact``, ``member_ids`` (sorted by ``str`` id),
        ``snapshot_count``, ``first_seen``/``last_seen`` (the min/max member
        ``created_at``) and ``span_days``. Timestamps go through
        ``_parsed_timestamp``/``_created_at_sort_key`` so "what counts as a
        usable timestamp" keeps the one definition it shares with
        ``pick_survivor``; a group whose members ALL lack a parseable
        timestamp reports None for all three rather than fabricating a span.
        Groups are sorted by ``(category, subject_task_id, first member id)``
        so two identical runs serialise byte-identically. Does not mutate
        *memories* or its dicts.

        *disclosure* carries every ``_LIVENESS_DISCLOSURE_KEYS`` counter —
        ``liveness_snapshot_untasked`` (a recognised snapshot that resolves to
        no subject, so no bucket can hold it) and
        ``liveness_snapshot_unfielded`` (point-in-time framing over a
        ``LIVE_TASK_STATUS_RE`` marker whose fields could not be read — a
        paraphrase naming no fields, or a value ``_LIVE_FIELD_ASSIGNMENT_RE``
        declines to read half-way, such as an unterminated quote).
        Both are always present, always ints, always 0 on a clean run.
    """
    swept = set(categories)
    disclosure = dict.fromkeys(_LIVENESS_DISCLOSURE_KEYS, 0)
    buckets: dict[tuple[str, str, str], list[dict]] = {}

    for record in memories:
        category = record.get('category')
        if category not in swept:
            continue
        # THE classifier — the same call `extract_liveness_snapshot_fact`
        # makes, so the two can never diverge, and one consult of the
        # quadratic `POINT_IN_TIME_CHECK_RE` per record at most.
        framed, core_fact = _classify_liveness_snapshot(record.get('content') or '')
        if core_fact is None:
            # "Not a snapshot at all" and "a snapshot whose fields could not be
            # read" are different outcomes, and only the second is a loss. The
            # discrimination lives HERE rather than inside the classifier,
            # which stays pure; `framed` is precisely the distinction.
            if framed:
                disclosure['liveness_snapshot_unfielded'] += 1
            continue
        subjects = liveness_snapshot_subject_task_ids(record)
        if not subjects:
            # A recognised snapshot no bucket can hold: real recall, counted
            # rather than invisible.
            disclosure['liveness_snapshot_untasked'] += 1
        for subject in subjects:
            buckets.setdefault((category, subject, core_fact), []).append(record)

    groups: list[dict[str, Any]] = []
    for (category, subject, core_fact), members in buckets.items():
        if len(members) < 2:
            continue
        # (epoch seconds, the raw created_at it came from), parsed ONCE per
        # member. Members whose timestamp does not parse are absent, so an
        # all-unparseable group leaves the span undefined rather than 0.0.
        stamped = [
            (ts, m.get('created_at'))
            for m in members
            if (ts := _parsed_timestamp(m.get('created_at'))) is not None
        ]
        first_seen = last_seen = span_days = None
        if stamped:
            stamped.sort(key=lambda pair: pair[0])
            first_seen = stamped[0][1]
            last_seen = stamped[-1][1]
            span_days = (stamped[-1][0] - stamped[0][0]) / _SECONDS_PER_DAY
        groups.append({
            'subject_task_id': subject,
            'category': category,
            'core_fact': core_fact,
            'member_ids': sorted((m.get('id') for m in members), key=str),
            'snapshot_count': len(members),
            'first_seen': first_seen,
            'last_seen': last_seen,
            'span_days': span_days,
        })

    groups.sort(key=lambda g: (
        str(g['category']), g['subject_task_id'], str(g['member_ids'][0]),
    ))
    return groups, disclosure


def ann_pairs_from_neighbors(
    memories: list[dict],
    neighbors_by_id: dict[Any, list[dict]],
    threshold: float,
    *,
    top_k: int | None = None,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """ANN neighbour lists → candidate index pairs + a full loss disclosure.

    The ANN sibling of ``_lexical_pairs``: both emit ``(i, j)`` index pairs
    into ``cluster_memories_by_pairs``. This one catches the paraphrase class
    the lexical scan structurally cannot — same meaning, almost no shared
    wording — but unlike a full O(n^2) scan it can lose candidates, so every
    loss is counted rather than absorbed.

    Pure: no I/O. The caller supplies the already-fetched neighbour lists
    (see ``fetch_ann_neighbors``), which keeps this fully testable without a
    live Qdrant.

    Args:
        memories: Scanned records, each optionally carrying ``'vector'``.
            Pair indices address this list.
        neighbors_by_id: ``{memory_id: [{'id': ..., 'score': ...}, ...]}``.
            A record absent from this mapping simply contributes no pairs.
        threshold: Minimum score for a hit to become a candidate pair.
            INCLUSIVE — a hit exactly at *threshold* is kept.
        top_k: The per-record neighbour cap the query was issued with. When
            given, a neighbour list filled to the cap is counted as
            ``top_k_saturated``: further matches may have been cut off.

    Returns:
        ``(pairs, disclosure)``. *pairs* are canonical ``(low, high)`` index
        tuples, deduped and sorted so the output is deterministic.
        *disclosure* maps every key in ``_ANN_DISCLOSURE_KEYS`` to an int:

          - ``top_k_saturated`` — records whose neighbour list hit the cap.
            A property of the QUERY, so counted regardless of how many hits
            then survived the threshold.
          - ``below_threshold_dropped`` — hits scored under *threshold*.
          - ``unknown_neighbor_dropped`` — hits naming a record outside the
            scanned set (excluded by the category filter, or written between
            the scroll and the query); it cannot be clustered, so it is lost.
          - ``missing_vector`` — records with no stored vector to query with,
            so they were never used as a query point.

        A self-hit is dropped silently: a record is always its own nearest
        neighbour, which is expected rather than a loss.

    Does not mutate *memories*.
    """
    disclosure = dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0)
    index_by_id: dict[Any, int] = {m.get('id'): i for i, m in enumerate(memories)}
    pairs: set[tuple[int, int]] = set()

    for i, memory in enumerate(memories):
        if memory.get('vector') is None:
            # Never queried — with_vectors was off, or Qdrant returned the
            # point without a vector. Either way its neighbourhood is unknown.
            disclosure['missing_vector'] += 1
            continue

        hits = neighbors_by_id.get(memory.get('id')) or []
        if top_k is not None and len(hits) >= top_k:
            disclosure['top_k_saturated'] += 1

        for hit in hits:
            hit_id = hit.get('id')
            if hit_id == memory.get('id'):
                continue  # self-hit: expected, not a loss
            if (hit.get('score') or 0.0) < threshold:
                disclosure['below_threshold_dropped'] += 1
                continue
            j = index_by_id.get(hit_id)
            if j is None:
                disclosure['unknown_neighbor_dropped'] += 1
                continue
            pairs.add((i, j) if i < j else (j, i))

    return sorted(pairs), disclosure


def ann_scores_for_pairs(
    memories: list[dict],
    neighbors_by_id: dict[Any, list[dict]],
    pairs: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Recover each candidate pair's ANN score from EITHER endpoint's list.

    The natural sibling of ``ann_pairs_from_neighbors``: same inputs, same
    index space, same no-I/O contract, so it is testable without a live
    Qdrant.

    BOTH directions are scanned because ANN neighbour lists are NOT
    symmetric — "the left endpoint's list contains the right" is a false
    invariant, and the two states that break it are ones this detector
    explicitly expects and counts:

      - ``top_k_saturated`` — the right lists the left while the left's cap
        is filled by nearer points, so the left never names the right.
      - ``missing_vector`` — a vector-less record is never a query point
        (``fetch_ann_neighbors`` skips it, so it is absent from
        *neighbors_by_id* entirely), yet it is still reachable as another
        record's hit.

    Scanning one side would therefore silently mis-report precisely the
    cases the disclosure counters exist to make legible.

    Args:
        memories: The scanned records; pair indices address this list.
        neighbors_by_id: ``{memory_id: [{'id': ..., 'score': ...}, ...]}``,
            as ``fetch_ann_neighbors`` returns it.
        pairs: Canonical ``(low, high)`` index pairs to score.

    Returns:
        ``{(low, high): max_score}``. A pair NEITHER endpoint scored is
        ABSENT from the mapping — never defaulted to 0.0. Absence is
        load-bearing: ``build_sweep_plan`` already degrades an unscored pair
        to ``ann_max_score: null``, whereas 0.0 is not None and so would be
        published as a real measurement claiming the cluster was found
        BELOW the cutoff that produced it. A hit carrying no score is
        likewise an unrecorded measurement, not a measurement of zero.

        For pairs ``ann_pairs_from_neighbors`` produced over the SAME
        mapping, a score is by construction always recoverable from at
        least one side (every such pair originated as a scored hit).
        ``None`` is therefore the defensive contract value for a caller
        supplying ``ann_pairs`` without scores, not a routine outcome.

    Does not mutate *memories*.
    """
    index_by_id: dict[Any, int] = {m.get('id'): i for i, m in enumerate(memories)}
    scores: dict[tuple[int, int], float] = {}

    for left, right in pairs:
        found: list[float] = []
        # Both directions: (query point, the endpoint its hit must resolve to).
        for source, target in ((left, right), (right, left)):
            for hit in neighbors_by_id.get(memories[source].get('id')) or []:
                # A self-hit resolves to `source`, never `target`, so it is
                # skipped here without needing a special case.
                if index_by_id.get(hit.get('id')) != target:
                    continue
                score = hit.get('score')
                if score is not None:
                    found.append(score)
        if found:
            scores[(left, right)] = max(found)

    return scores


def _created_at_sort_key(created_at: Any) -> tuple[int, float]:
    """Sort key placing parseable ``created_at`` oldest-first, unparseable last.

    Returns ``(0, timestamp)`` for a parseable ISO datetime string (so the
    oldest instant sorts first), or ``(1, 0.0)`` for ``None``/unparseable
    values — always after every parseable entry, so a record with no usable
    timestamp is never mistakenly picked as "the oldest".
    """
    if not isinstance(created_at, str) or not created_at:
        return (1, 0.0)
    try:
        return (0, datetime.fromisoformat(created_at).timestamp())
    except (ValueError, TypeError):
        return (1, 0.0)


def pick_survivor(group: list[dict]) -> tuple[dict, list[dict]]:
    """Pick the survivor from a near-duplicate memory group.

    Survivor selection (in order):
      1. A member explicitly flagged canonical (``metadata.get('canonical')``
         truthy) wins, regardless of age.
      2. Otherwise, the oldest member by ``created_at`` (ISO string) wins.
         Records with a missing/unparseable ``created_at`` sort last (see
         ``_created_at_sort_key``) so they are never chosen as "oldest"
         unless every member in the group lacks a usable timestamp.
      3. Ties (equal or absent ``created_at``) are broken by the lowest
         ``str(id)``.

    Raises ``ValueError`` for groups with < 2 memories.

    Returns ``(survivor, losers)`` with ``losers`` = all non-survivor members.
    """
    if len(group) < 2:
        raise ValueError(f'pick_survivor requires a group of >= 2 memories, got {len(group)}')

    def _sort_key(m: dict) -> tuple[bool, int, float, str]:
        canonical = bool((m.get('metadata') or {}).get('canonical'))
        bucket, ts = _created_at_sort_key(m.get('created_at'))
        # `not canonical` so canonical=True sorts first (False < True);
        # bucket/ts ascending so the oldest parseable timestamp sorts first;
        # id ascending so the lowest id wins remaining ties.
        return (not canonical, bucket, ts, str(m.get('id', '')))

    ordered = sorted(group, key=_sort_key)
    survivor = ordered[0]
    losers = [m for m in group if m is not survivor]
    return survivor, losers


def _shared_topic(group: list[dict]) -> str | None:
    """The one non-empty ``metadata.topic`` every member shares, else None."""
    topics = {str((m.get('metadata') or {}).get('topic') or '') for m in group}
    if len(topics) != 1:
        return None
    topic = topics.pop()
    return topic or None


def partition_topic_carveout(
    groups: list[list[dict]],
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Split clusters into ``(actionable, carved_out)`` by shared topic.

    OWNERSHIP SEAM — this rule belongs to task 3136, not to this detector.
    3136's amended Option-C carve-out says a cluster whose members all share
    the same non-empty ``metadata.topic`` is an EXPECTED shape: those records
    are deliberately co-topic'd, so deleting the "losers" would destroy
    intended structure rather than remove accidental duplication. It is
    honored here as INPUT FILTERING only — applied between clustering and
    survivor selection, so carved-out clusters never reach ``pick_survivor``,
    whose contract is untouched. Do NOT re-derive this rule elsewhere.

    The boundary is deliberately narrow: only a FULLY shared, non-empty topic
    carves out. Differing topics, a partially-missing topic, or an
    empty-string topic all stay actionable — a cluster is only exempt when
    every member was deliberately tagged the same way.

    Note ``metadata.topic`` has zero corpus footprint until 3195/3201 land,
    so this is inert-but-correct today by design: implemented now so it is
    not reinvented later, and load-bearing the moment topics appear.
    """
    actionable: list[list[dict]] = []
    carved_out: list[list[dict]] = []
    for group in groups:
        (carved_out if _shared_topic(group) else actionable).append(group)
    return actionable, carved_out


_MAX_LEXICAL_RATIO_MEMBERS = 40
"""Members of one cluster scanned exhaustively for ``lexical_max_ratio``.

A WORK bound on a report-only field, never a similarity decision — nothing is
clustered, kept or deleted because of it. It exists because ANN candidates now
feed the same transitive closure, so cluster size is no longer bounded by
near-verbatim similarity: a chained ANN cluster of a few hundred members would
otherwise make this O(k^2) exact-``ratio()`` loop the dominant cost of the
whole run. 40 members = 780 pairs, the same order as the per-record ANN query
cost the run already pays. When it binds, the run SAYS so
(``lexical_max_ratio_sampled``) rather than quietly reporting a sampled maximum
as an exhaustive one.
"""


def _max_lexical_ratio(
    normalized: list[str],
    member_indices: list[int],
    *,
    known: float | None = None,
    max_members: int = _MAX_LEXICAL_RATIO_MEMBERS,
) -> tuple[float | None, bool]:
    """Highest pairwise ``ratio()`` among a cluster's members, ignoring thresholds.

    Reported even when the lexical path LOST (scored below threshold and
    contributed no pair). An operator looking at an ANN-only cluster needs to
    see how the other detector scored it — "ANN 0.905, lexical 0.095" is the
    auditable form of a path disagreement; omitting the loser's score would
    hide exactly the number that explains why the paths differed.

    Args:
        normalized: Normalised contents (see :func:`_normalized_contents`).
        member_indices: The cluster's indices into *normalized*.
        known: The best ratio the lexical scan ALREADY measured for this
            cluster's pairs (see :func:`_lexical_pairs_scored`). Seeding it
            here is free accuracy and free speed: those pairs are a subset of
            the ones scanned below, so the exhaustive maximum is unchanged,
            while the running best starts high enough to prune immediately.
        max_members: Exhaustive-scan bound (see
            :data:`_MAX_LEXICAL_RATIO_MEMBERS`).

    Returns:
        ``(best, sampled)``. *best* is None when no pair could be compared
        (every member's content is empty). *sampled* is True when the cluster
        exceeded *max_members* and the maximum is over a subset — the caller
        publishes it as ``lexical_max_ratio_sampled`` so a bounded number is
        never read as an exhaustive one.
    """
    scanned = member_indices[:max_members]
    sampled = len(scanned) < len(member_indices)

    best = known
    matcher = difflib.SequenceMatcher()
    for a in range(len(scanned)):
        left = normalized[scanned[a]]
        if not left:
            continue
        matcher.set_seq1(left)
        for b in range(a + 1, len(scanned)):
            right = normalized[scanned[b]]
            if not right:
                continue
            matcher.set_seq2(right)
            # real_quick_ratio()/quick_ratio() are documented UPPER BOUNDS on
            # ratio(), so a pair that cannot even reach the running best
            # cannot be the maximum — skipping it leaves the answer identical
            # while avoiding the expensive exact comparison. Same prefilter
            # _lexical_pairs_scored applies, against `best` instead of a
            # fixed threshold.
            if best is not None and (
                matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best
            ):
                continue
            ratio = matcher.ratio()
            best = ratio if best is None else max(best, ratio)
    return best, sampled


# Losses that happen when ANN pairs are REMAPPED onto the per-category sweep,
# owned by build_sweep_plan rather than by ann_pairs_from_neighbors (which has
# no category awareness). Kept as a separate tuple from
# _ANN_DISCLOSURE_KEYS so each generator's disclosure contract stays its own.
_PLAN_DISCLOSURE_KEYS: tuple[str, ...] = (
    'cross_category_dropped',
    'unswept_category_dropped',
)


def _count_ann_pair_drops(
    memories: list[dict],
    ann_pairs: Iterable[tuple[int, int]],
    categories: Iterable[str],
) -> dict[str, int]:
    """Count ANN pairs the per-category sweep will refuse to cluster.

    A pair whose endpoints land in different swept categories, or outside the
    swept set entirely, is dropped by ``build_sweep_plan``'s remap. It
    consumed a ``top_k`` slot and named two real records, so it is a genuine
    recall loss — and it is NOT covered by ``unknown_neighbor_dropped``, which
    only fires for a hit outside the scanned set. Counting it here completes
    the ANN path's loss accounting: every drop is a metric the M2 limits
    evaluator can alarm on, never invisible recall.

    Counted ONCE per canonical pair, deliberately not inside the per-category
    loop: that loop visits the same dropped pair once per swept category, so
    incrementing there would multiply the count by the number of categories.

    An out-of-range index (a caller bug, silently dropped by the remap today)
    reads as an absent category, so it lands in ``unswept_category_dropped``
    rather than raising.
    """
    swept = set(categories)
    counts = dict.fromkeys(_PLAN_DISCLOSURE_KEYS, 0)

    def _category_at(index: int) -> Any:
        return memories[index].get('category') if 0 <= index < len(memories) else None

    seen: set[tuple[int, int]] = set()
    for left, right in ann_pairs:
        key = (left, right) if left < right else (right, left)
        if key in seen:
            continue
        seen.add(key)
        left_category, right_category = _category_at(key[0]), _category_at(key[1])
        if left_category not in swept or right_category not in swept:
            counts['unswept_category_dropped'] += 1
        elif left_category != right_category:
            counts['cross_category_dropped'] += 1
    return counts


def _pairs_span_members(
    member_indices: list[int],
    pairs: Iterable[tuple[int, int]],
) -> bool:
    """Do *pairs* alone connect every member into ONE component?

    "The lexical path found this cluster too" is only true if the lexical
    pairs SPAN it. A single lexical pair inside a cluster that ANN otherwise
    merged out of two unrelated halves would satisfy a naive
    "lexical_clustered" check while the merge itself rests entirely on ANN —
    which is exactly the evidence the apply gate must not accept.
    """
    if len(member_indices) < 2:
        return False
    parent = {i: i for i in member_indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right
    return len({find(i) for i in member_indices}) == 1


def restrict_delete_candidates_for_apply(
    groups: Iterable[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Narrow the reported delete candidates to the ones ``--apply`` may act on.

    The REPORT is unchanged — every cluster both detectors found still appears
    in ``near_duplicate_groups``/``delete_candidates``, because the dry-run
    report is the right place for a lower-confidence signal. This gates only
    the IRREVERSIBLE path.

    Two facts make the ANN path riskier than the lexical one it joins, and
    only on deletion:

      1. Its cutoff (``config.write_triage.t_high``) was calibrated for the
         write-triage decision on a corpus that is overwhelmingly
         ``procedural_knowledge``. Its pairwise precision on the two
         newly-swept categories is thinly evidenced by comparison.
      2. Pairwise precision does not imply CLUSTER precision. The transitive
         closure chains A~B and B~C into one cluster even when A~C is far
         below the cutoff, and under ``--apply`` every non-survivor of that
         chain becomes an irreversible delete.

    So a cluster may contribute deletions only when its evidence is
    chain-free:

      - ``lexical_spanning`` — the lexical pairs alone connect every member,
        i.e. the cluster is one this detector would have actioned before the
        ANN path existed. Pre-existing behaviour, unchanged.
      - ``ann_clique`` — EVERY member pair independently cleared the ANN
        cutoff, so no member is in the cluster by transitivity alone.

    Anything else is withheld, with its ids and the reason reported so the
    withholding is legible rather than a silent shortfall.

    Args:
        groups: The plan's ``near_duplicate_groups`` entries.

    Returns:
        ``(delete_candidates, withheld_groups)``.
    """
    actionable: list[Any] = []
    withheld: list[dict[str, Any]] = []
    for group in groups:
        losers = list(group.get('delete_candidate_ids') or ())
        if not losers:
            continue
        if group.get('lexical_spanning') or group.get('ann_clique'):
            actionable.extend(losers)
            continue
        withheld.append({
            'member_ids': list(group.get('member_ids') or ()),
            'category': group.get('category'),
            'withheld_ids': losers,
            'reason': 'ann_chained_without_lexical_span_or_ann_clique',
        })
    return actionable, withheld


def build_sweep_plan(
    memories: list[dict],
    threshold: float = 0.85,
    *,
    categories: Iterable[str] = _ALL_CATEGORIES,
    ann_pairs: Iterable[tuple[int, int]] | None = None,
    ann_scores: dict[tuple[int, int], float] | None = None,
    ann_disclosure: dict[str, int] | None = None,
    ann_threshold: float | None = None,
) -> dict[str, Any]:
    """Orchestrate filtering -> dual-path clustering -> survivor selection -> report.

    Runs BOTH candidate generators and reports both verdicts. The lexical
    ``difflib`` path is kept, not swapped out for ANN: it is high-precision
    on the near-verbatim rewrites that dominate this corpus, while ANN
    catches the paraphrase class lexical structurally cannot. Silently
    replacing one with the other would lose signal, so the actioned plan
    clusters over the UNION of both pair sets and each emitted cluster
    records which path(s) found it.

    Args:
        memories: Raw memory list (any/all categories).
        threshold: Near-duplicate similarity threshold for the LEXICAL path
            only. The ANN cutoff is applied upstream, when the pairs are
            generated (see ``ann_pairs_from_neighbors``).
        categories: Categories to sweep (default: all three Mem0 categories).
            Clustering runs once PER category, so a cross-category union is
            structurally impossible; records outside the set are ignored.
        ann_pairs: Optional ``(i, j)`` index pairs from the ANN path.
            Indices address *memories* as passed in — they are remapped
            across this function's category filter, and a pair whose
            endpoint was filtered out is dropped.
        ann_scores: Optional ``{(i, j): score}`` for those pairs, in the same
            index space, surfaced per cluster as ``ann_max_score``.
        ann_disclosure: Optional counter dict from the ANN path, echoed into
            the report so cap/loss counts travel with the plan they shaped.
            The remap-owned counters in ``_PLAN_DISCLOSURE_KEYS`` are MERGED
            into the echoed copy (the caller's dict is never mutated), so the
            plan carries the ANN path's complete loss accounting.
        ann_threshold: The ANN cutoff actually in effect, echoed so a reader
            of the report never has to guess which number produced it.

    Returns:
        JSON-serialisable plan dict. Pre-existing keys keep their exact
        meaning and shape (``clusters_total``, ``near_duplicate_groups``
        with the survivor's id/content and full member id list,
        ``delete_candidates`` flattened across clusters), so existing
        consumers are unaffected. Each group additionally carries
        ``found_by`` (``['ann']``, ``['lexical']`` or both),
        ``lexical_clustered``, ``lexical_spanning``, ``ann_clique``,
        ``ann_max_score``, ``lexical_max_ratio``,
        ``lexical_max_ratio_sampled`` and ``delete_candidate_ids``; the plan
        additionally carries ``threshold``, ``ann_threshold``,
        ``ann_disclosure``, ``topic_carveout_clusters`` /
        ``topic_carveout_groups`` (see ``partition_topic_carveout``),
        ``apply_delete_candidates`` / ``apply_withheld_clusters`` /
        ``apply_withheld_groups`` (see
        ``restrict_delete_candidates_for_apply``) and ``path_verdicts``
        (``lexical_only_clusters`` / ``ann_only_clusters`` /
        ``both_paths_clusters``).

        It also carries ``liveness_snapshot_recurrence_groups`` /
        ``liveness_snapshot_recurrence_clusters`` /
        ``liveness_snapshot_disclosure`` (see
        ``find_liveness_snapshot_recurrences``). That class is REPORT-ONLY: it
        never contributes to ``delete_candidates`` or
        ``apply_delete_candidates``, and its groups are never handed to the
        apply gate — it is surfaced for human review and consolidation only.
        Task 3098 authorizes detection of the pattern, not disposition of the
        records that exhibit it.
    """
    near_duplicate_groups: list[dict[str, Any]] = []
    # Mem0/Qdrant point ids can be int or str (ExtendedPointId), so this is
    # not narrowed to list[str].
    delete_candidates: list[Any] = []
    lexical_only = ann_only = both_paths = 0
    clusters_total = 0
    topic_carveout_groups: list[dict[str, Any]] = []

    # Cluster PER CATEGORY. Widening the filter must not merge the corpus: a
    # preference and a procedure that happen to read alike are different
    # kinds of knowledge, and unioning them would be cross-store data loss
    # rather than deduplication. Running the closure once per category makes
    # a cross-category union structurally impossible instead of relying on a
    # downstream check to catch it.
    for category in categories:
        candidate_indices = [
            i for i, m in enumerate(memories) if m.get('category') == category
        ]
        if not candidate_indices:
            continue
        candidates = [memories[i] for i in candidate_indices]
        # Original index -> candidate index, so ANN pairs generated against
        # the unfiltered scan still address the right records after filtering.
        candidate_index = {orig: new for new, orig in enumerate(candidate_indices)}

        normalized = _normalized_contents(candidates)
        # Keep the exact ratios the scan already computed: _max_lexical_ratio
        # seeds its running maximum from them instead of re-measuring pairs
        # this loop just measured.
        lexical_ratio_by_pair = {
            (i, j): ratio for i, j, ratio in _lexical_pairs_scored(normalized, threshold)
        }
        lexical_pairs = set(lexical_ratio_by_pair)

        remapped_ann: set[tuple[int, int]] = set()
        ann_score_by_pair: dict[tuple[int, int], float] = {}
        for pair in ann_pairs or []:
            left, right = candidate_index.get(pair[0]), candidate_index.get(pair[1])
            if left is None or right is None:
                # Endpoint outside this category — either filtered out
                # entirely, or in a DIFFERENT category, in which case the
                # pair is dropped here and cannot union across categories.
                continue
            key = (left, right) if left < right else (right, left)
            remapped_ann.add(key)
            score = (ann_scores or {}).get(pair)
            if score is not None:
                ann_score_by_pair[key] = max(ann_score_by_pair.get(key, score), score)

        clustered = cluster_memories_by_pairs(candidates, lexical_pairs | remapped_ann)
        # Task 3136's Option-C carve-out, applied BEFORE survivor selection so
        # a deliberately co-topic'd cluster can never reach pick_survivor and
        # therefore can never produce a delete candidate.
        groups, carved_out = partition_topic_carveout(clustered)
        for carved in carved_out:
            topic_carveout_groups.append({
                'member_ids': [m.get('id') for m in carved],
                'category': category,
                'topic': _shared_topic(carved),
            })
        clusters_total += len(groups)
        # cluster_memories_by_pairs passes member dicts through by identity,
        # so this recovers each cluster's indices for per-path attribution.
        index_by_identity = {id(m): i for i, m in enumerate(candidates)}
        # Bucket the pair SETS by cluster in one pass over each set, rather
        # than enumerating every cluster's O(k^2) in-group pairs and testing
        # membership: clusters partition the members, so a pair is in-group
        # exactly when both endpoints share a component. Carved-out clusters
        # are absent from `component_of`, so their pairs fall out here.
        component_of: dict[int, int] = {}
        for position, group in enumerate(groups):
            for member in group:
                component_of[index_by_identity[id(member)]] = position
        lexical_by_group: dict[int, list[tuple[int, int]]] = {}
        ann_by_group: dict[int, list[tuple[int, int]]] = {}
        for pair_set, bucket in ((lexical_pairs, lexical_by_group),
                                 (remapped_ann, ann_by_group)):
            for pair in pair_set:
                position = component_of.get(pair[0])
                if position is not None and position == component_of.get(pair[1]):
                    bucket.setdefault(position, []).append(pair)

        for position, group in enumerate(groups):
            member_indices = [index_by_identity[id(m)] for m in group]
            # Every distinct member pair: the denominator a clique must fill.
            in_group_count = len(member_indices) * (len(member_indices) - 1) // 2
            group_lexical = lexical_by_group.get(position, [])
            group_ann = ann_by_group.get(position, [])

            found_by = []
            if group_ann:
                found_by.append('ann')
            if group_lexical:
                found_by.append('lexical')
            if group_ann and group_lexical:
                both_paths += 1
            elif group_ann:
                ann_only += 1
            else:
                lexical_only += 1

            scores = [ann_score_by_pair[p] for p in group_ann if p in ann_score_by_pair]
            known_ratio = max(
                (lexical_ratio_by_pair[p] for p in group_lexical
                 if p in lexical_ratio_by_pair),
                default=None,
            )
            max_ratio, ratio_sampled = _max_lexical_ratio(
                normalized, member_indices, known=known_ratio,
            )

            survivor, losers = pick_survivor(group)
            loser_ids = [m.get('id') for m in losers]
            near_duplicate_groups.append({
                'survivor_id': survivor.get('id'),
                'survivor_content': survivor.get('content'),
                'member_ids': [m.get('id') for m in group],
                'category': category,
                'found_by': found_by,
                'lexical_clustered': bool(group_lexical),
                # Chain-free-evidence flags; see
                # restrict_delete_candidates_for_apply for why --apply needs
                # them and the report does not.
                'lexical_spanning': _pairs_span_members(member_indices, group_lexical),
                'ann_clique': bool(group_ann) and len(group_ann) == in_group_count,
                'ann_max_score': max(scores) if scores else None,
                'lexical_max_ratio': max_ratio,
                'lexical_max_ratio_sampled': ratio_sampled,
                'delete_candidate_ids': loser_ids,
            })
            delete_candidates.extend(loser_ids)

    # The THIRD candidate class, run ONCE over the whole list rather than
    # inside the per-category loop above: it does its own
    # (category, subject, core_fact) bucketing, so the per-category isolation
    # comes from the key. Running it inside the loop would be O(n x
    # categories) and would multiply its disclosure counters by the number of
    # categories — the same double-counting reason _count_ann_pair_drops is
    # called once, outside.
    liveness_groups, liveness_disclosure = find_liveness_snapshot_recurrences(
        memories, categories=categories,
    )

    apply_candidates, apply_withheld = restrict_delete_candidates_for_apply(
        near_duplicate_groups,
    )
    echoed_disclosure: dict[str, int] | None = None
    if ann_disclosure is not None or ann_pairs is not None:
        echoed_disclosure = {
            **dict(ann_disclosure or {}),
            **_count_ann_pair_drops(memories, ann_pairs or (), categories),
        }

    return {
        'clusters_total': clusters_total,
        'near_duplicate_groups': near_duplicate_groups,
        'delete_candidates': delete_candidates,
        'threshold': threshold,
        'ann_threshold': ann_threshold,
        'ann_disclosure': echoed_disclosure,
        'topic_carveout_clusters': len(topic_carveout_groups),
        'topic_carveout_groups': topic_carveout_groups,
        'liveness_snapshot_recurrence_clusters': len(liveness_groups),
        'liveness_snapshot_recurrence_groups': liveness_groups,
        'liveness_snapshot_disclosure': liveness_disclosure,
        'apply_delete_candidates': apply_candidates,
        'apply_withheld_clusters': len(apply_withheld),
        'apply_withheld_groups': apply_withheld,
        'path_verdicts': {
            'lexical_only_clusters': lexical_only,
            'ann_only_clusters': ann_only,
            'both_paths_clusters': both_paths,
        },
    }


# ---------------------------------------------------------------------------
# Cluster metrics + the M1 metric series (pure — no I/O)
# ---------------------------------------------------------------------------

_EVAL_ID = 'e6-corpus-health'
"""This detector's eval id — the artifact directory name under the M1 root."""

_GLOBAL_SCOPE = 'all'
"""metric_id suffix for a run-global counter, where no single category owns it.

Every metric_id is ``<name>.<scope>`` so the suffix is always meaningful and
always present. The ANN disclosures are properties of ONE query pass over the
whole scanned set, not of a category, so inventing a per-category split for
them would be a fabricated attribution.
"""

_SECONDS_PER_DAY = 86400.0

# Metrics whose full shape (histogram, per-topic table, per-event table) is
# too wide for a single scalar and lives in the companion details file.
_DETAILED_METRICS: tuple[str, ...] = (
    'cluster_size_max',
    'cluster_size_mean',
    'topic_accretion_rate_mean',
    'consolidation_net_delta',
    'liveness_snapshot_recurrences',
)


def _metric_id(name: str, scope: str) -> str:
    return f'{name}.{scope}'


def split_plan_by_category(
    plan: dict[str, Any],
    categories: Iterable[str],
) -> dict[str, dict[str, list[dict]]]:
    """Project a sweep plan into per-category slices.

    ``build_sweep_plan`` already clusters ONCE PER CATEGORY and tags every
    emitted group with the category it was clustered under, so this is a
    projection of what the plan states — not a second categorisation that
    could disagree with the one the plan actually used.

    Every requested category gets a slice, empty ones included: an absent key
    downstream would read as "not measured" rather than "measured, found
    nothing".
    """
    slices: dict[str, dict[str, list[dict]]] = {
        category: {
            'near_duplicate_groups': [],
            'topic_carveout_groups': [],
            'liveness_snapshot_recurrence_groups': [],
        }
        for category in categories
    }
    for key in ('near_duplicate_groups', 'topic_carveout_groups',
                'liveness_snapshot_recurrence_groups'):
        for group in plan.get(key) or []:
            slot = slices.get(group.get('category'))
            if slot is not None:
                slot[key].append(group)
    return slices


def _parsed_timestamp(created_at: Any) -> float | None:
    """Epoch seconds for a parseable ``created_at``, else None.

    Delegates to ``_created_at_sort_key``'s tolerance so "what counts as a
    usable timestamp" has one definition shared with survivor selection.
    """
    bucket, ts = _created_at_sort_key(created_at)
    return None if bucket else ts


def _topic_of(record: dict) -> str | None:
    topic = (record.get('metadata') or {}).get('topic')
    return str(topic) if topic else None


def _absorbed_count(metadata: dict) -> int:
    """How many predecessors a consolidation event RECORDS having absorbed.

    Read strictly from ``supersedes``: a list contributes its length, a
    non-empty scalar contributes 1. A record flagged ``canonical`` with no
    ``supersedes`` names no predecessors, so it absorbed 0 *recorded* ones —
    guessing a number for it would put an invented figure into a metric.
    """
    supersedes = metadata.get('supersedes')
    if isinstance(supersedes, str):
        return 1 if supersedes else 0
    if isinstance(supersedes, list | tuple | set):
        return len(supersedes)
    return 1 if supersedes else 0


def _topic_accretion_table(records: list[dict]) -> list[dict[str, Any]]:
    """Per-topic ``members / span_days`` accretion, topics sorted for determinism.

    Only records whose ``created_at`` parses can be placed on the span, so
    ``members`` counts those and unparseable ones are counted separately (see
    the ``unparseable_created_at`` metric) rather than silently inflating a
    rate over a span they did not contribute to.

    A topic spanning zero elapsed time has NO defined rate — ``rate_per_day``
    is None. A rate needs elapsed time; dividing by a fudged epsilon would
    manufacture an enormous rate out of a single record.
    """
    by_topic: dict[str, list[float]] = {}
    for record in records:
        topic = _topic_of(record)
        if topic is None:
            continue
        ts = _parsed_timestamp(record.get('created_at'))
        if ts is not None:
            by_topic.setdefault(topic, []).append(ts)
        else:
            by_topic.setdefault(topic, [])

    table: list[dict[str, Any]] = []
    for topic in sorted(by_topic):
        stamps = sorted(by_topic[topic])
        span_days = (stamps[-1] - stamps[0]) / _SECONDS_PER_DAY if stamps else 0.0
        rate = len(stamps) / span_days if span_days > 0 else None
        table.append({
            'topic': topic,
            'members': len(stamps),
            'span_days': span_days,
            'rate_per_day': rate,
        })
    return table


def _consolidation_events(
    records: list[dict],
    groups: list[dict],
) -> list[dict[str, Any]]:
    """Per-consolidation net delta: what accreted after it minus what it absorbed.

    A consolidation event is a record carrying ``canonical: true`` and/or a
    non-empty ``supersedes``. Its peer set — the population that can accrete
    "after" it — is its shared ``metadata.topic`` when it has one, else the
    near-duplicate cluster it belongs to. An event with neither has no
    identifiable peers, so nothing is attributed to it (0), rather than
    charging it with every later record in the category.

    A negative delta means consolidation is winning; a positive one means the
    topic is re-accreting faster than the consolidation absorbed.

    Every record's timestamp and topic are parsed ONCE per run, and the peer
    sets are pre-bucketed, so this is O(records + sum of peer-set sizes)
    rather than O(events x records) with a fresh ``datetime.fromisoformat``
    inside the inner loop.
    """
    cluster_by_id: dict[Any, frozenset] = {}
    for group in groups:
        members = frozenset(group.get('member_ids') or ())
        for member_id in members:
            cluster_by_id[member_id] = members

    ts_by_index = [_parsed_timestamp(r.get('created_at')) for r in records]
    topic_by_index = [_topic_of(r) for r in records]
    indices_by_topic: dict[str, list[int]] = {}
    indices_by_cluster: dict[frozenset, list[int]] = {}
    for index, record in enumerate(records):
        topic = topic_by_index[index]
        if topic is not None:
            indices_by_topic.setdefault(topic, []).append(index)
        members = cluster_by_id.get(record.get('id'))
        if members is not None:
            indices_by_cluster.setdefault(members, []).append(index)

    events: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        metadata = record.get('metadata') or {}
        if not (metadata.get('canonical') or metadata.get('supersedes')):
            continue
        topic = topic_by_index[index]
        event_ts = ts_by_index[index]
        peers = cluster_by_id.get(record.get('id'))

        accreted = 0
        if event_ts is not None:
            # Peer set: the shared topic when there is one, else the cluster;
            # an event with neither has no identifiable peers, so nothing is
            # attributed to it rather than charging it with the whole category.
            if topic is not None:
                peer_indices: list[int] = indices_by_topic.get(topic, [])
            elif peers is not None:
                peer_indices = indices_by_cluster.get(peers, [])
            else:
                peer_indices = []
            for other in peer_indices:
                if records[other] is record:
                    continue
                other_ts = ts_by_index[other]
                if other_ts is not None and other_ts > event_ts:
                    accreted += 1

        absorbed = _absorbed_count(metadata)
        events.append({
            'memory_id': record.get('id'),
            'topic': topic,
            'absorbed': absorbed,
            'accreted_after': accreted,
            'net_delta': accreted - absorbed,
        })
    return events


def compute_cluster_metrics(
    records_by_category: dict[str, list[dict]],
    plan_by_category: dict[str, dict[str, list[dict]]],
    disclosures: dict[str, Any],
    *,
    details_path: str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Corpus-health metrics for one run, plus the companion details payload.

    Pure: takes the already-fetched records and the already-built plan, so the
    whole metric surface is testable without a live Mem0.

    Kind selection follows M2's rule vocabulary rather than taste. A ``count``
    or ``proportion`` ARMS an exact statistical test, so it is used only where
    a regression is well-defined and directional; everything whose movement is
    not a regression is a ``scalar``, which is reported but never alarmed. In
    particular the topic carve-out count is a scalar: more carve-outs is the
    EXPECTED Option-C shape once 3195/3201 land, so arming a rule on it would
    alarm on the feature working.

    Args:
        records_by_category: ``{category: [record, ...]}`` — the scanned
            corpus. Its per-category lengths are the metric denominators.
        plan_by_category: ``{category: {'near_duplicate_groups': [...],
            'topic_carveout_groups': [...],
            'liveness_snapshot_recurrence_groups': [...]}}`` (see
            :func:`split_plan_by_category`).
        disclosures: ``{counter_name: int | {category: int}}``. An int is a
            run-global counter (emitted as ``<name>.all``); a mapping is
            per-category (one metric per category, e.g. ``scan_truncated``).
            EVERY key is emitted, zeros included — an absent metric is
            indistinguishable from "not measured", and a cap that fired must
            be a countable, alarmable number rather than invisible recall loss.
        details_path: Filename of the companion details artifact, stamped onto
            the metrics whose full shape lives there.

    Returns:
        ``(metrics, details)``. *metrics* is a list of
        ``shared.memory_eval_metrics.Metric``, sorted by ``metric_id`` so two
        identical runs serialise byte-identically. *details* carries the
        per-category histogram, per-topic accretion table and per-event
        consolidation table.

    Note the clustered-member share is WITHHELD for an empty category and
    named in ``details['categories'][cat]['undefined_metrics']``. M1 forbids a
    zero-trial proportion, and emitting 0.0 would assert "0% of this corpus is
    duplicated" about a corpus that contains nothing to measure.
    """
    from shared.memory_eval_metrics import Metric  # noqa: PLC0415

    metrics: list[Any] = []
    details: dict[str, Any] = {'categories': {}}
    total_corpus = sum(len(v) for v in records_by_category.values())

    for category in sorted(records_by_category):
        records = records_by_category[category]
        slice_ = plan_by_category.get(category) or {}
        groups = slice_.get('near_duplicate_groups') or []
        carveouts = slice_.get('topic_carveout_groups') or []
        recurrences = slice_.get('liveness_snapshot_recurrence_groups') or []
        corpus_size = len(records)

        sizes = [len(g.get('member_ids') or ()) for g in groups]
        clustered_members = sum(sizes)
        histogram: dict[str, int] = {}
        for size in sizes:
            histogram[str(size)] = histogram.get(str(size), 0) + 1

        accretion = _topic_accretion_table(records)
        rates = [r['rate_per_day'] for r in accretion if r['rate_per_day'] is not None]
        events = _consolidation_events(records, groups)
        unparseable = sum(
            1 for r in records if _parsed_timestamp(r.get('created_at')) is None
        )
        undefined: list[str] = []

        metrics.append(Metric(
            metric_id=_metric_id('near_duplicate_clusters', category),
            kind='count', value=float(len(groups)), n=corpus_size,
            direction='higher_is_worse',
        ))
        if corpus_size > 0:
            metrics.append(Metric(
                metric_id=_metric_id('clustered_member_share', category),
                kind='proportion', value=clustered_members / corpus_size,
                n=corpus_size, denominator=corpus_size,
                direction='higher_is_worse',
            ))
        else:
            undefined.append(_metric_id('clustered_member_share', category))
        metrics.append(Metric(
            metric_id=_metric_id('cluster_size_max', category),
            kind='scalar', value=float(max(sizes)) if sizes else 0.0,
            n=len(groups), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('cluster_size_mean', category),
            kind='scalar', value=(sum(sizes) / len(sizes)) if sizes else 0.0,
            n=len(groups), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('topic_carveout_clusters', category),
            kind='scalar', value=float(len(carveouts)), n=len(carveouts),
        ))
        # A `count`/`higher_is_worse`, unlike the `scalar` used for the topic
        # carve-out immediately above. M2's rule is that an ARMED kind is for
        # a regression that is well-defined and directional: more carve-outs
        # is the EXPECTED Option-C shape, so alarming on it would alarm on the
        # feature working, whereas more liveness recurrence groups IS the
        # indefinite accretion this detector exists to surface. Emitted for
        # every swept category, empty ones included — an absent metric is
        # indistinguishable from "not measured".
        #
        # Carries `details_path` like every other metric whose full shape
        # lives in the companion file: the scalar says HOW MANY groups, but
        # the per-group `subject_task_id`/`core_fact`/`member_ids` table
        # written below is the one artifact that tells a human WHICH memories
        # to consolidate. Without the pointer a consumer reading only the
        # metric series cannot discover it.
        metrics.append(Metric(
            metric_id=_metric_id('liveness_snapshot_recurrences', category),
            kind='count', value=float(len(recurrences)), n=corpus_size,
            direction='higher_is_worse', details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('topic_accretion_rate_mean', category),
            kind='scalar', value=(sum(rates) / len(rates)) if rates else 0.0,
            n=len(rates), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('consolidation_net_delta', category),
            kind='scalar', value=float(sum(e['net_delta'] for e in events)),
            n=len(events), details_path=details_path,
        ))
        metrics.append(Metric(
            metric_id=_metric_id('unparseable_created_at', category),
            kind='count', value=float(unparseable), n=corpus_size,
            direction='higher_is_worse',
        ))

        details['categories'][category] = {
            'corpus_size': corpus_size,
            'clustered_members': clustered_members,
            'cluster_size_histogram': histogram,
            'topic_accretion': accretion,
            'consolidation_events': events,
            'liveness_snapshot_recurrences': [
                {
                    'subject_task_id': g.get('subject_task_id'),
                    'core_fact': g.get('core_fact'),
                    'member_ids': g.get('member_ids'),
                    'snapshot_count': g.get('snapshot_count'),
                    'span_days': g.get('span_days'),
                }
                for g in recurrences
            ],
            'undefined_metrics': undefined,
        }

    for name in sorted(disclosures):
        value = disclosures[name]
        if isinstance(value, dict):
            for category in sorted(value):
                metrics.append(Metric(
                    metric_id=_metric_id(name, category), kind='count',
                    value=float(value[category]),
                    n=len(records_by_category.get(category) or ()),
                    direction='higher_is_worse',
                ))
        else:
            metrics.append(Metric(
                metric_id=_metric_id(name, _GLOBAL_SCOPE), kind='count',
                value=float(value), n=total_corpus, direction='higher_is_worse',
            ))

    metrics.sort(key=lambda m: m.metric_id)
    return metrics, details


def details_artifact_path(stamp: str) -> str:
    """``details-<STAMP>.json`` — the companion payload's FILENAME.

    A bare filename, not a path: it is both the name written beside
    ``metrics-<STAMP>.json`` and the value stamped onto ``Metric.details_path``,
    and a reader resolves it relative to the metrics file it came from. Naming
    it once keeps those two uses from drifting apart.
    """
    return f'details-{stamp}.json'


def emit_metrics_artifact(
    series: Any,
    details: dict[str, Any],
    root: str | Any,
    stamp: str,
) -> tuple[Any, Any, Any]:
    """Write the run's three artifacts, returning their paths.

    Delegates the metrics + report write to
    ``shared.memory_eval_metrics.write_metric_series``, which VALIDATES BEFORE
    creating anything. The details file is written only afterwards, so a
    rejected series leaves no directory and no partial artifact — a directory
    holding only ``details-<STAMP>.json`` would read to the limits evaluator as
    a run whose metrics file went missing, which is strictly worse than the run
    never having written at all.

    All THREE files go through the same ``_atomic_write_text``
    (temp-in-dir + ``os.replace``). Reusing the metrics writer's own helper
    rather than a plain ``write_text`` is the point: every emitted metric
    carries a ``details_path``, and a reader that follows it must never land on
    a half-written JSON document. A crash mid-write leaves the temp file
    unlinked and no details file at all — visibly absent, which is diagnosable,
    rather than present and truncated, which is not. A failure to write it is
    logged and RE-RAISED: the series' ``details_path`` would otherwise dangle,
    and a silently incomplete artifact set is exactly what M1 rejects at emit
    time.

    Returns:
        ``(metrics_path, report_path, details_path)``.
    """
    from shared.memory_eval_metrics import (  # noqa: PLC0415
        _atomic_write_text,
        write_metric_series,
    )

    metrics_path, report_path = write_metric_series(series, root, stamp=stamp)
    details_path = metrics_path.parent / details_artifact_path(stamp)
    try:
        _atomic_write_text(
            details_path,
            json.dumps(details, indent=2, sort_keys=True, default=str) + '\n',
        )
    except OSError:
        logger.exception(
            'Wrote %s but FAILED to write its companion %s — every metric in '
            'that series names a details file that does not exist.',
            metrics_path, details_path,
        )
        raise
    return metrics_path, report_path, details_path


def build_metric_series(
    project_id: str,
    metrics: Iterable[Any],
    *,
    corpus_counts: dict[str, int],
    eval_id: str = _EVAL_ID,
    run_stamp: str | None = None,
) -> Any:
    """Assemble one run's :class:`MetricSeries`, validating it at BUILD time.

    Everything is routed through ``validate_metric_series`` on the raw payload
    rather than relying on model construction, so a malformed metric — however
    it was produced, ``Metric`` object or hand-built dict — surfaces as one
    ``MetricSchemaError`` here in the runner that computed it. That is M1's
    "rejected at emit time, not read time": by the time the dashboard reads the
    artifact there is nobody left to tell.

    The schema itself lives in ``shared.memory_eval_metrics`` and is never
    restated here (INV-5) — this function only fills it in.

    Args:
        project_id: Scanned project; becomes ``Corpus.project_id``.
        metrics: ``Metric`` models and/or raw metric dicts.
        corpus_counts: Per-category corpus sizes — the denominators behind the
            denominators.
        eval_id: Artifact directory name (default :data:`_EVAL_ID`).
        run_stamp: Run stamp; defaults to ``shared.memory_eval_metrics``'
            helper, which honours ``MEMORY_EVAL_RUN_STAMP``.

    Raises:
        MetricSchemaError: The series is not emittable.
    """
    from shared.memory_eval_metrics import (  # noqa: PLC0415
        SCHEMA_VERSION,
        Metric,
        MetricSeries,
        validate_metric_series,
    )
    from shared.memory_eval_metrics import run_stamp as _run_stamp

    payload = {
        'schema_version': SCHEMA_VERSION,
        'eval_id': eval_id,
        'run_stamp': run_stamp if run_stamp is not None else _run_stamp(),
        'corpus': {'project_id': project_id, 'counts': dict(corpus_counts)},
        'metrics': [
            m.model_dump(mode='json', exclude_none=True) if isinstance(m, Metric) else dict(m)
            for m in metrics
        ],
    }
    # Raises MetricSchemaError (never a bare pydantic error) on any malformed
    # metric, unknown field, or duplicate metric_id. Parsing afterwards cannot
    # fail: the same payload just validated.
    validate_metric_series(payload)
    return MetricSeries.model_validate(payload)


# ---------------------------------------------------------------------------
# I/O layer: fetch (thin, mock-tested)
# ---------------------------------------------------------------------------

# Payload keys tried in order when extracting a Mem0 memory's text content
# from its scroll_by_metadata() 'metadata' payload dict. 'data' is the
# canonical scroll-payload key (mirrors prune_recon_cycle_summaries.py, which
# reads only metadata.get('data')) so it is tried first; 'memory' is a
# search-result-layer key (memory_service.py's MemoryResult.content, built
# from Mem0's search API, not scroll) that can appear stale on a scroll
# payload, so it is only a fallback; 'content' is a defensive third fallback.
_CONTENT_KEYS: tuple[str, ...] = ('data', 'memory', 'content')


async def fetch_memories(
    memory: Any,
    project_id: str,
    *,
    categories: Iterable[str] = _ALL_CATEGORIES,
    scan_limit: int = 5000,
    with_vectors: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Enumerate a project's memories across *categories*, with scan stats.

    Issues ONE ``scroll_by_metadata`` per category. That is not a missed
    optimisation: the primitive builds an AND-equality filter and has no OR,
    so a single widened call cannot express "any of these three categories".

    Each raw record (``{'id', 'created_at', 'metadata'[, 'vector']}``) is
    normalised to ``{'id', 'content', 'category', 'created_at', 'metadata'}``
    (plus ``'vector'`` when *with_vectors*). The top-level ``'category'`` is
    lifted out of the payload because ``build_sweep_plan`` filters on it —
    without the lift every record would read as category ``None`` and the
    sweep would be a silent no-op. Content is extracted by trying
    ``_CONTENT_KEYS`` in order, falling back to ``''``; a record with no
    extractable content never clusters and is never deleted.

    Args:
        memory: Live (or mock) MemoryService instance.
        project_id: Project scope to scan.
        categories: Categories to enumerate (default: all three Mem0 ones).
        scan_limit: Max points PER CATEGORY.
        with_vectors: Fetch each record's stored vector, for ANN candidate
            generation. See ``Mem0Backend.scroll_by_metadata``.

    Returns:
        ``(records, scan_stats)``. *scan_stats* maps each requested category
        to ``{'scanned': n, 'truncated': 0|1}``, counted PER CATEGORY so a cap
        firing on one is never reported as a clean scan of the whole corpus.
        ``truncated`` is an int rather than a bool so it can be summed and
        emitted as a metric directly.

    Raises:
        TimeoutError: A Qdrant read timeout propagates rather than degrading
            to an empty list (``scroll_by_metadata`` no longer swallows it),
            so a timed-out scan is never mistaken for an empty corpus.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    scope = Scope(project_id=project_id)
    records: list[dict[str, Any]] = []
    scan_stats: dict[str, dict[str, int]] = {}

    for category in categories:
        # Defense in depth against a repeated category (the CLI already
        # de-duplicates): re-scrolling one appends a second copy of every
        # record under the SAME id, which self-clusters and makes the id its
        # own delete candidate. Skipping is safe -- scan_stats already holds
        # that category's counts, so nothing is lost from the disclosure.
        if category in scan_stats:
            continue
        raw_records = await memory.mem0.scroll_by_metadata(
            scope, {'category': category}, limit=scan_limit, with_vectors=with_vectors,
        ) or []
        scan_stats[category] = {
            'scanned': len(raw_records),
            'truncated': int(len(raw_records) >= scan_limit),
        }
        for record in raw_records:
            payload = record.get('metadata') or {}
            content = ''
            for key in _CONTENT_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value:
                    content = value
                    break
            normalized: dict[str, Any] = {
                'id': record.get('id'),
                'content': content,
                'category': payload.get('category'),
                'created_at': record.get('created_at'),
                'metadata': payload,
            }
            if with_vectors:
                normalized['vector'] = record.get('vector')
            records.append(normalized)

    return records, scan_stats


async def fetch_procedural_memories(
    memory: Any,
    project_id: str,
    scan_limit: int = 5000,
) -> list[dict[str, Any]]:
    """Single-category ``procedural_knowledge`` fetch — retained for back-compat.

    A thin alias over :func:`fetch_memories`, kept because task 3136 schedules
    this detector and existing callers bind this name. It returns a BARE LIST
    (not the ``(records, scan_stats)`` tuple) and defaults to payload-only, so
    every pre-existing caller is unaffected by the widening to three
    categories and to optional vectors.

    New callers should use :func:`fetch_memories`, which reports per-category
    scan stats the truncation guard and the metrics series both need.
    """
    records, _scan_stats = await fetch_memories(
        memory, project_id,
        categories=('procedural_knowledge',),
        scan_limit=scan_limit,
    )
    return records


# ---------------------------------------------------------------------------
# I/O layer: ANN candidate generation (thin, mock-tested)
# ---------------------------------------------------------------------------

def _calibrated_float(value: Any) -> float | None:
    """A measured cutoff, or ``None`` if this leaf is not one.

    ``near_duplicate_guard.resolve_near_dup_threshold``'s leaf check, hoisted
    so the pooled scalar and every per-category map VALUE are screened by one
    rule. ``bool`` is excluded deliberately (it is an ``int`` subclass, and
    ``True`` would resolve as a cutoff of 1.0), and so is every non-numeric
    object — including the attribute an unspecced Mock auto-generates.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def resolve_ann_threshold(
    memory_service: Any,
    override: float | None = None,
    categories: Iterable[str] = (),
) -> tuple[dict[str, float], dict[str, int]]:
    """Resolve the ANN cosine cutoff PER CATEGORY. READ, never invented.

    Every cutoff is a CALIBRATION OUTPUT derived by
    ``scripts/calibrate_write_triage.py`` from measured similarity
    distributions over the curator-labeled corpus, with
    ``calibration_report_path`` recording its provenance. This detector reads
    that single home; it never restates a number and never supplies a
    fallback of its own.

    Priority per swept category (task 3357):

      1. ``override`` — an explicit operator number, applied to every
         category, including over a populated per-category map.
      2. ``config.write_triage.t_high_by_category[category]`` — the cutoff
         measured on THAT category's own labeled pairs.
      3. ``config.write_triage.t_high`` — the POOLED cutoff, a DISCLOSED
         fallback counted in ``ann_pooled_fallback_categories``. It is real
         evidence, but gathered on a corpus dominated by one category, so a
         fallback category is additionally withheld from ``--apply``
         deletion by ``restrict_delete_candidates_for_apply``.
      4. Neither — the category is ABSENT from the returned mapping. Its ANN
         path is disabled rather than run on a guess.

    Follows ``resolve_near_dup_threshold``'s defensive getattr-at-every-hop
    shape and its ``isinstance(int | float) and not bool`` leaf check (see
    :func:`_calibrated_float`), applied to the map's VALUES too — but returns
    ABSENCE instead of a module default, because substituting an uncalibrated
    stand-in here is precisely the invented threshold that is forbidden. A
    ``t_high_by_category`` that is not a Mapping (an unspecced Mock, a stray
    scalar) is ignored wholesale rather than probed.

    Args:
        memory_service: Live (or mock) MemoryService.
        override: Operator-supplied ``--ann-threshold``. Wins when given,
            including over an uncalibrated config.
        categories: The categories actually being swept. Nothing is resolved
            for a category that was not asked for, so an empty iterable
            yields an empty mapping — the caller passes its swept tuple.

    Returns:
        ``({category: threshold}, disclosure)``. A category is present only
        when a MEASURED cutoff resolved for it; an EMPTY mapping means no
        category resolved and the caller must DISABLE the ANN path entirely
        rather than guess. *disclosure* always carries both
        ``ann_disabled_uncalibrated`` (how many swept categories resolved to
        nothing) and ``ann_pooled_fallback_categories`` (how many ran on the
        pooled number), so neither a disabled path nor weaker evidence is a
        silently missing detector.
    """
    swept = tuple(categories)

    forced = _calibrated_float(override)
    if forced is not None:
        return (
            dict.fromkeys(swept, forced),
            {'ann_disabled_uncalibrated': 0, 'ann_pooled_fallback_categories': 0},
        )

    write_triage = getattr(getattr(memory_service, 'config', None), 'write_triage', None)
    pooled = _calibrated_float(getattr(write_triage, 't_high', None))
    by_category = getattr(write_triage, 't_high_by_category', None)
    if not isinstance(by_category, Mapping):
        by_category = {}

    resolved: dict[str, float] = {}
    pooled_fallback: list[str] = []
    uncalibrated: list[str] = []
    for category in swept:
        own = _calibrated_float(by_category.get(category))
        if own is not None:
            resolved[category] = own
        elif pooled is not None:
            resolved[category] = pooled
            pooled_fallback.append(category)
        else:
            uncalibrated.append(category)

    if pooled_fallback:
        logger.warning(
            'ANN cutoff for %s falls back to the POOLED '
            'config.write_triage.t_high (%r): the calibration derived no '
            'cutoff of their own, so they are gated by a number measured on '
            'a different population. Weaker evidence — these categories are '
            'withheld from --apply deletion.',
            ', '.join(pooled_fallback), pooled,
        )
    if uncalibrated:
        logger.error(
            'ANN path DISABLED for %s: neither config.write_triage'
            '.t_high_by_category nor the pooled t_high is calibrated (pooled '
            'got %r) and no --ann-threshold override was given. Refusing to '
            'invent a similarity cutoff — run scripts/calibrate_write_triage.py, '
            'or pass --ann-threshold explicitly. The lexical path still runs.',
            ', '.join(uncalibrated), getattr(write_triage, 't_high', None),
        )

    return resolved, {
        'ann_disabled_uncalibrated': len(uncalibrated),
        'ann_pooled_fallback_categories': len(pooled_fallback),
    }


_ANN_QUERY_CONCURRENCY = 16
"""ANN queries in flight at once.

A resource knob, not a similarity decision: it changes only how fast the same
queries are issued, never which hits come back. Sequential issue would put a
``--scan-limit``-sized scan across three categories at up to 15,000 SERIALISED
round-trips — minutes of pure latency for a detector task 3136 schedules
periodically. Bounded rather than unbounded because the same Qdrant instance
serves the live memory path, and an unbounded fan-out over a whole corpus
would be a self-inflicted load spike.
"""


async def fetch_ann_neighbors(
    memory: Any,
    project_id: str,
    records: list[dict[str, Any]],
    *,
    top_k: int,
    score_threshold: float,
    concurrency: int = _ANN_QUERY_CONCURRENCY,
) -> tuple[dict[Any, list[dict[str, Any]]], dict[str, int]]:
    """ANN neighbours for each record, queried with the record's OWN vector.

    Mirrors ``task_curator.search_corpus``'s ``query_points`` call, retargeted
    from the curator's own collection to the Mem0 collection. The query vector
    is the embedding Mem0 already stored (fetched by
    ``scroll_by_metadata(..., with_vectors=True)``), NOT a freshly-embedded
    string: the detector therefore makes zero embedding API calls at runtime
    and cannot drift into a second metric space.

    The record's CATEGORY is pushed into the query as a payload filter (the
    same ``Filter``/``FieldCondition``/``MatchValue`` shape
    ``Mem0Backend.scroll_by_metadata`` and ``count_by_metadata`` build). The
    collection holds every category, so an unfiltered ``top_k`` could be spent
    entirely on records the per-category sweep will discard — the genuine
    same-category candidates cut off below them would be lost recall the ANN
    path exists to add. Filtering ALSO makes a cross-category pair structurally
    impossible rather than merely dropped later. A record with no category
    cannot be clustered by ``build_sweep_plan`` at all, so it is queried
    unfiltered exactly as before.

    Args:
        memory: Live (or mock) MemoryService instance.
        project_id: Project scope to query.
        records: Normalised records, each optionally carrying ``'vector'``.
        top_k: Neighbours wanted per record. The query asks for ``top_k + 1``
            so the record's own guaranteed self-hit does not consume a slot.
        score_threshold: Cutoff pushed down into Qdrant.
        concurrency: Max queries in flight (see
            :data:`_ANN_QUERY_CONCURRENCY`).

    Returns:
        ``({memory_id: [{'id', 'score'}, ...]}, disclosure)``, with records in
        input order so two identical runs produce identical output. A record
        with no vector is skipped (``ann_pairs_from_neighbors`` counts it as
        ``missing_vector``). *disclosure* carries ``ann_query_errors``: a
        failure on one record is logged and counted, and the sweep continues,
        so one bad point never costs the whole scan.
    """
    from qdrant_client.http import models as qmodels  # noqa: PLC0415

    from fused_memory.models.scope import Scope  # noqa: PLC0415

    neighbors: dict[Any, list[dict[str, Any]]] = {}
    disclosure = {'ann_query_errors': 0}
    queryable = [r for r in records if r.get('vector') is not None]
    if not queryable:
        return neighbors, disclosure

    scope = Scope(project_id=project_id)
    collection_name = scope.mem0_collection_name(memory.config.mem0.collection_prefix)
    client = await memory.mem0._get_async_qdrant()  # noqa: SLF001

    filter_by_category: dict[Any, Any] = {}

    def _category_filter(category: Any) -> Any:
        if not category:
            return None
        if category not in filter_by_category:
            filter_by_category[category] = qmodels.Filter(must=[
                qmodels.FieldCondition(
                    key='category', match=qmodels.MatchValue(value=category),
                ),
            ])
        return filter_by_category[category]

    semaphore = asyncio.Semaphore(concurrency)

    async def _query(record: dict[str, Any]) -> tuple[Any, list[dict[str, Any]] | None]:
        record_id = record.get('id')
        async with semaphore:
            try:
                result = await client.query_points(
                    collection_name=collection_name,
                    query=record.get('vector'),
                    limit=top_k + 1,  # +1 absorbs the guaranteed self-hit
                    score_threshold=score_threshold,
                    query_filter=_category_filter(record.get('category')),
                    with_payload=False,
                )
            except Exception as exc:
                # Per-record, so one bad point is counted rather than fatal.
                logger.error('ANN query failed for memory %s: %s', record_id, exc)
                return record_id, None
        return record_id, [
            {'id': point.id, 'score': point.score}
            for point in result.points
            if point.id != record_id
        ]

    # gather preserves argument order, so `neighbors` is populated in input
    # order regardless of which query finished first.
    for record_id, hits in await asyncio.gather(*(_query(r) for r in queryable)):
        if hits is None:
            disclosure['ann_query_errors'] += 1
            continue
        neighbors[record_id] = hits

    return neighbors, disclosure


# ---------------------------------------------------------------------------
# I/O layer: apply (thin, mock-tested)
# ---------------------------------------------------------------------------

async def apply_deletions(
    memory: Any,
    project_id: str,
    plan: dict[str, Any],
    *,
    dry_run: bool,
    candidates: Iterable[Any] | None = None,
) -> dict[str, int]:
    """Delete the plan's ``delete_candidates`` from Mem0 (best-effort).

    Dry-run performs no ``delete_memory`` calls and returns zero counts.
    Otherwise, each candidate is deleted individually in a try/except so a
    single failure does not abort the remaining deletes (best-effort partial
    progress) — mirrors ``audit_duplicate_tasks.apply_changes``.

    Args:
        candidates: Explicit ids to delete, overriding the plan's own
            ``delete_candidates``. ``_run`` passes the apply-gated subset
            (see ``restrict_delete_candidates_for_apply``); the default keeps
            every pre-existing caller's behaviour byte-for-byte.

    Returns:
        Dict with ``deleted`` and ``delete_errors`` counts.
    """
    if dry_run:
        return {'deleted': 0, 'delete_errors': 0}

    deleted = 0
    delete_errors = 0
    targets = plan.get('delete_candidates', []) if candidates is None else candidates
    for memory_id in targets:
        try:
            await memory.delete_memory(memory_id, store='mem0', project_id=project_id)
            logger.info('Deleted near-duplicate memory %s', memory_id)
            deleted += 1
        except Exception as exc:
            logger.error('Failed to delete memory %s: %s', memory_id, exc)
            delete_errors += 1

    return {'deleted': deleted, 'delete_errors': delete_errors}


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _apply_refusal_reason(
    records: list[dict],
    scan_stats: dict[str, dict[str, int]],
    scan_limit: int,
) -> str | None:
    """Why ``--apply`` must not run, or None if it may.

    Irreversible-deletion guards (mirrors ``prune_recon_cycle_summaries.py``'s
    scan-completeness guard). Truncation is judged PER CATEGORY: one capped
    category is enough to refuse the whole apply, because a cluster's survivor
    may be sitting just outside that window and "delete the losers" would then
    delete the last copy.
    """
    if not records:
        return 'scan returned 0 records — refusing to apply on an empty scan'
    truncated = sorted(c for c, s in scan_stats.items() if s.get('truncated'))
    if truncated:
        return (
            f'scan looks truncated for {truncated} (>= --scan-limit={scan_limit}) — '
            'refusing to apply. Re-run with a higher --scan-limit'
        )
    return None


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    from shared.memory_eval_metrics import run_stamp  # noqa: PLC0415

    import fused_memory.config.schema as _schema  # noqa: PLC0415
    import fused_memory.services.memory_service as _service  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # De-duplicate while preserving the caller's order. A repeated
    # --categories value would otherwise ingest the same Qdrant points twice
    # under one id; the two identical copies cluster at ratio 1.0 and
    # pick_survivor hands back the second copy as a LOSER, so --apply would
    # delete the id that is also the survivor -- i.e. the last copy of the
    # record. It also doubles every metric denominator.
    categories = tuple(dict.fromkeys(args.categories))
    stamp = run_stamp()

    config = _schema.FusedMemoryConfig()
    memory = _service.MemoryService(config)
    await memory.initialize()
    try:
        # The ANN cutoff is READ from the calibration (or an explicit
        # override), PER CATEGORY. An empty mapping means no category is
        # calibrated: the ANN path is disabled and counted, never run against
        # a guessed threshold.
        ann_thresholds, disclosures = resolve_ann_threshold(
            memory, args.ann_threshold, categories,
        )
        # Floor pushed down to Qdrant: the MIN of the resolved cutoffs cannot
        # drop a candidate any category would have accepted (every cutoff is
        # >= the min). The authoritative per-category filtering happens in the
        # pure ann_pairs_from_neighbors.
        ann_threshold = min(ann_thresholds.values()) if ann_thresholds else None

        records, scan_stats = await fetch_memories(
            memory, args.project_id, categories=categories,
            scan_limit=args.scan_limit, with_vectors=ann_threshold is not None,
        )
        logger.info(
            'Fetched %d memory/memories across %s', len(records), list(categories),
        )

        ann_pairs: list[tuple[int, int]] = []
        ann_scores: dict[tuple[int, int], float] = {}
        if ann_threshold is not None:
            neighbors, query_disclosure = await fetch_ann_neighbors(
                memory, args.project_id, records,
                top_k=args.ann_top_k, score_threshold=ann_threshold,
            )
            disclosures.update(query_disclosure)
            ann_pairs, pair_disclosure = ann_pairs_from_neighbors(
                records, neighbors, ann_threshold, top_k=args.ann_top_k,
            )
            disclosures.update(pair_disclosure)
            # Scans BOTH endpoints' lists: neighbour lists are asymmetric, so
            # a pair is often recorded only on the side that discovered it.
            ann_scores = ann_scores_for_pairs(records, neighbors, ann_pairs)
        else:
            disclosures.update(dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0))
            disclosures['ann_query_errors'] = 0

        plan = build_sweep_plan(
            records, threshold=args.threshold, categories=categories,
            ann_pairs=ann_pairs, ann_scores=ann_scores,
            ann_disclosure=disclosures, ann_threshold=ann_threshold,
        )
        # stdout is task 3136's report contract — unchanged shape, one JSON doc.
        print(json.dumps(plan, indent=2, default=str))

        # build_sweep_plan owns the remap-stage losses (a cross-category or
        # unswept-category ANN pair), so the plan's echoed copy — not the dict
        # handed to it — is the complete accounting the metrics must carry.
        disclosures = plan['ann_disclosure'] or disclosures

        if not args.no_metrics:
            records_by_category = {
                category: [m for m in records if m.get('category') == category]
                for category in categories
            }
            metrics, details = compute_cluster_metrics(
                records_by_category,
                split_plan_by_category(plan, categories),
                {
                    **disclosures,
                    # Plain ints, so the existing branch emits them at
                    # _GLOBAL_SCOPE (`<name>.all`) with no new emission path:
                    # they are properties of ONE pass over the whole scanned
                    # set, not of a category.
                    **(plan['liveness_snapshot_disclosure'] or {}),
                    'scan_truncated': {
                        c: int(scan_stats.get(c, {}).get('truncated', 0))
                        for c in categories
                    },
                },
                details_path=details_artifact_path(stamp),
            )
            series = build_metric_series(
                args.project_id, metrics,
                corpus_counts={c: len(v) for c, v in records_by_category.items()},
                eval_id=args.eval_id, run_stamp=stamp,
            )
            metrics_path, _report, _details = emit_metrics_artifact(
                series, details, args.metrics_root, stamp,
            )
            logger.info('Wrote metrics artifact %s', metrics_path)

        if not args.apply:
            logger.info('Dry run — nothing was modified. Use --apply to commit.')
            return 0

        refusal = _apply_refusal_reason(records, scan_stats, args.scan_limit)
        if refusal:
            logger.error('ABORT: %s.', refusal)
            return 1

        # The apply gate: clusters whose only evidence is an ANN CHAIN stay in
        # the report but never reach an irreversible delete.
        apply_candidates = plan['apply_delete_candidates']
        withheld = plan['apply_withheld_groups']
        if withheld:
            logger.warning(
                '%d cluster(s) withheld from --apply: neither lexically '
                'spanning nor an ANN clique, so membership rests on '
                'transitivity across the ANN cutoff. Still reported, NOT '
                'deleted: %s',
                len(withheld), [g['member_ids'] for g in withheld],
            )
        if not apply_candidates:
            logger.error(
                'ABORT: 0 applicable delete candidates (%d reported, %d '
                'cluster(s) withheld by the apply gate) — nothing to apply.',
                len(plan['delete_candidates']), len(withheld),
            )
            return 1

        result = await apply_deletions(
            memory, args.project_id, plan, dry_run=False,
            candidates=apply_candidates,
        )
        logger.info(
            'Applied: deleted %d/%d memory/memories; %d error(s)',
            result['deleted'], len(apply_candidates), result['delete_errors'],
        )
        return 1 if result['delete_errors'] > 0 else 0
    finally:
        await memory.close()


_DEFAULT_METRICS_ROOT = str(Path(__file__).resolve().parent.parent / 'data' / 'memory-evals')
"""``fused-memory/data/memory-evals`` (M1 §3), resolved off THIS file.

Not off the cwd: a scheduled run's working directory is not guaranteed, and a
relative default would scatter artifacts wherever the scheduler happened to
start — invisible to the limits evaluator, which scans one root.
"""

_DEFAULT_ANN_TOP_K = 10
"""Neighbours requested per record.

A resource knob, not a similarity threshold: it trades query cost against
recall, and when it binds the run says so (``top_k_saturated``) rather than
quietly losing candidates. The cutoff that decides what IS a duplicate is
never set here — see :func:`resolve_ann_threshold`.
"""


def _build_parser() -> argparse.ArgumentParser:
    """The CLI surface. Every flag added after the first release is optional.

    ``--project-id X`` and ``--project-id X --apply`` — the invocation task
    3136 schedules — must keep parsing byte-for-byte, so all of this task's
    flags carry defaults.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-id', dest='project_id', required=True,
        help='Project id to scan for near-duplicate memories',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.85,
        help='LEXICAL near-duplicate similarity threshold (default: 0.85). '
             'Does not affect the ANN path, whose cutoff is calibrated.',
    )
    parser.add_argument(
        '--scan-limit', dest='scan_limit', type=int, default=5000,
        help='Maximum number of memories to scan PER CATEGORY (default: 5000)',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit deletions (default: dry-run, report only)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    parser.add_argument(
        '--categories', nargs='+', default=list(_ALL_CATEGORIES),
        choices=list(_ALL_CATEGORIES),
        help=f'Mem0 categories to sweep (default: {" ".join(_ALL_CATEGORIES)}). '
             'Only Mem0-backed categories are accepted; a Graphiti-backed or '
             'unknown name is rejected at parse time rather than silently '
             'scrolling an empty corpus and reporting a clean sweep.',
    )
    parser.add_argument(
        '--ann-top-k', dest='ann_top_k', type=int, default=_DEFAULT_ANN_TOP_K,
        help=f'ANN neighbours per record (default: {_DEFAULT_ANN_TOP_K}). '
             'Saturation is reported as top_k_saturated.',
    )
    parser.add_argument(
        '--ann-threshold', dest='ann_threshold', type=float, default=None,
        help='Override the ANN cosine cutoff. Default: config.write_triage.t_high, '
             'a calibration output. With neither, the ANN path is DISABLED.',
    )
    parser.add_argument(
        '--metrics-root', dest='metrics_root', default=_DEFAULT_METRICS_ROOT,
        help=f'Root for M1 metric artifacts (default: {_DEFAULT_METRICS_ROOT})',
    )
    parser.add_argument(
        '--eval-id', dest='eval_id', default=_EVAL_ID,
        help=f'Artifact directory name under --metrics-root (default: {_EVAL_ID})',
    )
    parser.add_argument(
        '--no-metrics', dest='no_metrics', action='store_true',
        help='Skip the metrics artifact (report to stdout only)',
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
