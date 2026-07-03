"""Suffix-conflict graph and bounce-state tracker (MQ-refactor task δ=1988).

Extracted verbatim from :mod:`orchestrator.merge_queue`: the two-layer
suffix-conflict machinery — ``SuffixConflictGraph`` (the immutable
conflict-graph dataclass) and ``EMPTY_SUFFIX_CONFLICT_GRAPH`` (its sentinel
empty instance) — plus a NEW :class:`SuffixConflictTracker` class that owns
the state (``graph`` / ``signature`` / ``last_known_main_sha`` /
``bounce_registry``) and logic (``recompute()`` /
``bounce_conflicting_suffix_items()``) that previously lived directly on
``SpeculativeMergeWorker``.  ``merge_queue`` re-exports all three names
through a top-level shim so existing importers
(``from orchestrator.merge_queue import X``, etc.) keep working unchanged.

``SpeculativeMergeWorker`` delegates to a ``SuffixConflictTracker`` instance
(``self._suffix_tracker``) via thin property/method wrappers that preserve
the worker's original attribute names (``_suffix_conflict_graph``,
``_suffix_conflict_signature``, ``_last_known_main_sha``,
``_bounce_registry``) and method signatures
(``recompute_suffix_conflict_graph()``, ``_bounce_conflicting_suffix_items()``)
so the existing conflict-graph and bounce test suites keep passing with zero
churn.  The tracker itself never holds a worker reference: GitOps access,
lane-buffer access, and frozen-prefix reads are injected as narrow
*re-reading* callables (``git_ops``, ``lane_buffers``, ``frozen_prefix``,
``frozen_prefix_tip``) that look up the caller's live attribute on every
call — not a value or bound method captured once at construction — so the
tracker is unit-testable standalone AND stays correct if the caller later
reassigns e.g. ``worker._git_ops`` (amend: reviewer
robustness_stale_reference / consistency, task δ=1988).

A moved method that reads a merge_queue-resident constant (``MERGE_LANES``,
``MERGE_BOUNCE_CAP``, ``NEEDS_REBASE_REASON_PREFIX``) — these stay in
``merge_queue.py`` and are NOT moved — resolves it through a function-local
(deferred) import from :mod:`orchestrator.merge_queue` rather than a
top-level import.  This mirrors the reach-back convention established by
:mod:`orchestrator.merge_liveness` / :mod:`orchestrator.merge_drift` and
keeps this module free of any top-level import of ``merge_queue`` (which
would deadlock module load, since merge_queue's shim needs this module
fully defined first).
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from collections.abc import Callable

from orchestrator.git_ops import GitOps
from orchestrator.merge_types import (
    GroupMergeRequest,
    MergeBounceRegistry,
    MergeOutcome,
    MergeRequest,
)
from orchestrator.overlap_footprint import Footprint, get_overlap_detector

logger = logging.getLogger('orchestrator.merge_queue')


@dataclasses.dataclass(frozen=True)
class SuffixConflictGraph:
    """Immutable conflict-graph over the unfrozen merge-queue suffix (task δ=1889).

    Holds two distinct pair-edge relations and one per-item marker:

    * **footprint_edges** — pairs whose changed-path footprints overlap (γ seam).
      Drives future ζ (ordering) consumers; computed cheaply via path-set
      intersection without forking git.

    * **textual_edges** — pairs with genuine 3-way textual conflicts (β seam).
      Pruned to footprint-overlapping pairs only (textual ⇒ footprint contract).
      Drives future η (bounce) consumers; each entry represents a confirmed
      git merge-tree conflict.

    * **conflicts_with_main** — request_ids of suffix items that conflict with
      the current main tip (the δ user-signal).

    **Node identity** — request_id (the stable per-MergeRequest UUID, e.g.
    ``'mr-a1b2c3d4'``).  Nodes are stored in pick order (high lane before
    normal, FIFO within each lane) so the tuple doubles as the ordered view
    of the suffix.

    **Immutability** — frozen dataclass; every field is a frozenset or tuple
    so the graph can be shared safely across the async event loop without
    copying.

    See also: EMPTY_SUFFIX_CONFLICT_GRAPH (module constant for the zero case).
    """

    nodes: tuple[str, ...]
    """Request IDs in pick order (high lane → normal lane, FIFO within each lane)."""

    textual_edges: frozenset[frozenset[str]]
    """Unordered pairs {rid_a, rid_b} with a confirmed 3-way textual conflict."""

    footprint_edges: frozenset[frozenset[str]]
    """Unordered pairs {rid_a, rid_b} whose changed-path footprints overlap."""

    conflicts_with_main: frozenset[str]
    """Request IDs that conflict with the current main tip."""

    def textual_neighbors(self, rid: str) -> frozenset[str]:
        """Return the set of request_ids connected to *rid* via textual_edges."""
        return frozenset(
            next(iter(edge - {rid}))
            for edge in self.textual_edges
            if rid in edge
        )

    def footprint_neighbors(self, rid: str) -> frozenset[str]:
        """Return the set of request_ids connected to *rid* via footprint_edges."""
        return frozenset(
            next(iter(edge - {rid}))
            for edge in self.footprint_edges
            if rid in edge
        )

    def to_snapshot_dict(self) -> dict:
        """Return a JSON-safe dict representation suitable for heartbeat snapshots.

        Output format:
          nodes: list[str]                 — in pick order
          textual_edges: list[list[str]]   — each inner list sorted; outer sorted
          footprint_edges: list[list[str]] — same shape as textual_edges
          conflicts_with_main: list[str]   — sorted
        """
        return {
            'nodes': list(self.nodes),
            'textual_edges': sorted(sorted(edge) for edge in self.textual_edges),
            'footprint_edges': sorted(sorted(edge) for edge in self.footprint_edges),
            'conflicts_with_main': sorted(self.conflicts_with_main),
        }


EMPTY_SUFFIX_CONFLICT_GRAPH = SuffixConflictGraph(
    nodes=(),
    textual_edges=frozenset(),
    footprint_edges=frozenset(),
    conflicts_with_main=frozenset(),
)
"""Sentinel empty SuffixConflictGraph for the default/zero-suffix case."""


class SuffixConflictTracker:
    """Owns the two-layer suffix-conflict state + logic (task δ=1988).

    Constructed with four narrow accessor callables — ``git_ops``,
    ``lane_buffers``, ``frozen_prefix``, ``frozen_prefix_tip`` — instead of a
    worker reference, so the tracker is fully unit-testable without a
    :class:`SpeculativeMergeWorker`.  All four are captured as callables (not
    their values, and not a bound-method snapshot) so the tracker always
    re-reads the caller's live state at call time, including a
    :class:`~orchestrator.git_ops.GitOps` reassigned on the caller *after*
    this tracker was constructed (e.g. ``worker._git_ops = new_instance``;
    amend: reviewer robustness_stale_reference / consistency).

    Attributes:
        graph: The current :class:`SuffixConflictGraph` (starts at the empty
            sentinel).
        signature: Debounce signature ``(ordered_rids, main_sha)`` from the
            last successful :meth:`recompute`, or ``None`` before the first
            compute (or after invalidation by a bounce).
        last_known_main_sha: The real main SHA cached at the last successful
            :meth:`recompute` call.
        bounce_registry: Per-branch bounce counter for the needs-rebase cap.
    """

    def __init__(
        self,
        git_ops: Callable[[], GitOps],
        *,
        lane_buffers: Callable[[], dict[str, collections.deque[MergeRequest]]],
        frozen_prefix: Callable[[], tuple[str, ...]],
        frozen_prefix_tip: Callable[[str], str],
    ) -> None:
        self._git_ops = git_ops
        self._lane_buffers = lane_buffers
        self._frozen_prefix = frozen_prefix
        self._frozen_prefix_tip = frozen_prefix_tip
        self.graph: SuffixConflictGraph = EMPTY_SUFFIX_CONFLICT_GRAPH
        self.signature: tuple[tuple[str, ...], str] | None = None
        self.last_known_main_sha: str | None = None
        self.bounce_registry: MergeBounceRegistry = MergeBounceRegistry()

    async def recompute(self) -> None:
        """Recompute and store the conflict-graph over the unfrozen suffix (task δ=1889).

        Enumerates _lane_buffers in pick order (high → normal, FIFO within each
        lane), resolves each item's branch head SHA and changed-paths footprint,
        builds footprint_edges (γ path-overlap) and textual_edges (β 3-way merge
        probe), and marks items that conflict with the current main tip.

        **Debounce** — short-circuits when the signature (ordered suffix
        request_ids, main_sha) is unchanged since the last recompute.  Any change
        to the suffix (submit/resubmit → new request_id, landing → item removed +
        main advances, reorder → order changes) breaks the signature and forces a
        full recompute.  Within a recompute, merge_tree_conflicts results are
        memoized by (base_sha, head_sha) to avoid duplicate probes.

        **Fail-open** — any probe error → conservative (treat as conflict/edge);
        items with missing branch refs are skipped without aborting.  Never raises.

        **Textual pruning** — the β/γ contract idealises textual-conflict ⇒
        footprint-overlap.  The footprint is approximated via
        ``get_changed_files(main_sha, head)`` (current-main..head delta) rather
        than the true merge-base..head delta; this can under-report a branch's
        footprint when main has since converged to identical content.
        Downstream ζ/η consumers should treat ``footprint_edges`` as a
        best-effort (not guaranteed-complete) superset.  Only
        footprint-overlapping pairs are probed via merge_tree_conflicts,
        collapsing the O(n²) forks to the overlap subset; a missed footprint
        edge means the textual probe is skipped, so a genuine textual conflict
        could be silently omitted from ``textual_edges``.

        This method is async (forks git subprocesses) while snapshot() stays
        synchronous and cheap (reads the stored graph; no await).
        """
        # Deferred reach-back: MERGE_LANES stays in merge_queue.py (not moved).
        # A top-level import would deadlock module load — see module docstring.
        from orchestrator.merge_queue import MERGE_LANES

        # ── 1. Build ordered suffix list ──────────────────────────────────────
        # ε=1890 defensive exclusion: frozen items (currently verifying) must
        # never appear as suffix-graph nodes — the frozen/suffix partition is
        # invariant (§5.3). Compute the frozen-rid set once, then filter.
        frozen_rids: frozenset[str] = frozenset(self._frozen_prefix())
        suffix: list[MergeRequest] = []
        for lane in MERGE_LANES:  # high → normal
            for req in self._lane_buffers()[lane]:  # FIFO within each lane
                if req.request_id not in frozen_rids:
                    suffix.append(req)

        ordered_rids = tuple(req.request_id for req in suffix)

        # ── 2. Debounce: fetch main_sha once, compare signature ───────────────
        try:
            main_sha = await self._git_ops().get_main_sha()
        except Exception:
            logger.warning('recompute_suffix_conflict_graph: get_main_sha failed; skipping')
            return

        # λ=1895 — cache the real main SHA BEFORE the debounce early-return so
        # it stays fresh even when the signature is unchanged and we return early.
        # snapshot()'s two_layer_invariants() call reads this field, not
        # _newest_frozen_commit() (the frozen-stack tip), to avoid a spurious
        # base-chain violation during normal in-flight verify.
        self.last_known_main_sha = main_sha

        sig = (ordered_rids, main_sha)
        if sig == self.signature:
            return  # Suffix + main unchanged — prior graph is still valid

        # ── 3. Empty suffix → sentinel ────────────────────────────────────────
        if not suffix:
            self.graph = EMPTY_SUFFIX_CONFLICT_GRAPH
            self.signature = sig
            return

        # ── 4. Resolve branch heads + footprints ─────────────────────────────
        # Per-item: (head_sha | None, changed_paths | None)
        heads: list[str | None] = []
        changed_paths_list: list[list[str] | None] = []
        # Initialised exactly once here so that a get_changed_files failure
        # (set to True below) survives into the signature-caching gate at the
        # end of this method.  Do NOT re-initialise this flag in later steps.
        _any_probe_error = False

        branch_prefix = self._git_ops().config.branch_prefix
        for req in suffix:
            ref = None
            head = None
            try:
                # Mirror resolve_queued_branch_ref: try prefixed form first
                # (bare-id contract: task/X wins), then bare.  Capturing the
                # SHA in the same pass avoids a redundant third rev-parse vs
                # the previous resolve_queued_branch_ref + resolve_branch_sha
                # pattern (each call forks a git subprocess).
                prefixed = f'{branch_prefix}{req.branch}'
                sha = await self._git_ops().resolve_branch_sha(prefixed)
                if sha is not None:
                    ref, head = prefixed, sha
                else:
                    sha2 = await self._git_ops().resolve_branch_sha(req.branch)
                    if sha2 is not None:
                        ref, head = req.branch, sha2
            except Exception:
                logger.warning(
                    'recompute_suffix_conflict_graph: resolve_queued_branch_ref(%r) raised; '
                    'treating item as missing ref', req.branch, exc_info=True,
                )
                ref = None
                head = None
            heads.append(head)

            if head is None:
                changed_paths_list.append(None)
                continue
            try:
                # Branch delta = from merge-base to branch head; approximate
                # using branch..main ancestor as from (git diff main..branch).
                # Simpler: get_changed_files(main_sha, head) captures the branch
                # delta relative to current main (sufficient for overlap detection).
                paths = await self._git_ops().get_changed_files(main_sha, head)
            except Exception:
                logger.warning(
                    'recompute_suffix_conflict_graph: get_changed_files failed for %r; '
                    'treating footprint as unknown', ref, exc_info=True,
                )
                paths = None
                _any_probe_error = True
            changed_paths_list.append(paths)

        # ── 5. Build footprint_edges via path-set intersection ────────────────
        detector = get_overlap_detector(None)
        footprint_edges: set[frozenset[str]] = set()

        # Precompute footprints once per item — avoids O(E·n) recomputation
        # inside the O(n²) pairwise loop (each detector.footprint() call
        # allocates a fresh frozenset; hoisting it drops that to O(n)).
        footprints_list: list[Footprint | None] = [
            detector.footprint(p) if p is not None else None
            for p in changed_paths_list
        ]

        for i in range(len(suffix)):
            fp_i = footprints_list[i]
            if fp_i is None:
                continue  # missing ref — skip pairwise comparison for this item
            for j in range(i + 1, len(suffix)):
                fp_j = footprints_list[j]
                if fp_j is None:
                    continue
                try:
                    overlap = detector.overlaps(fp_i, fp_j)
                except Exception:
                    logger.warning(
                        'recompute_suffix_conflict_graph: footprint overlap check '
                        'raised for pair (%s, %s); treating as overlap (fail-open)',
                        suffix[i].request_id, suffix[j].request_id,
                        exc_info=True,
                    )
                    overlap = True
                if overlap:
                    footprint_edges.add(frozenset({
                        suffix[i].request_id,
                        suffix[j].request_id,
                    }))

        # ── 6. Build textual_edges via merge_tree_conflicts (β) ──────────────
        # Only probe footprint-overlapping pairs (textual ⇒ footprint contract
        # from γ; note footprint is an approximation — see docstring caveat —
        # so a genuine textual conflict without a footprint edge is possible).
        # Memoize results by frozenset({head_a, head_b}) within this recompute
        # so a pair is probed at most once even when re-ordered.
        _probe_cache: dict[frozenset[str], bool] = {}
        textual_edges: set[frozenset[str]] = set()

        # Build once — invariant across all footprint-edge iterations (O(n) vs O(E·n)).
        idx_map = {req.request_id: k for k, req in enumerate(suffix)}

        for fp_edge in footprint_edges:
            rids = tuple(fp_edge)
            i_idx = idx_map.get(rids[0])
            j_idx = idx_map.get(rids[1])
            if i_idx is None or j_idx is None:
                continue
            head_i = heads[i_idx]
            head_j = heads[j_idx]
            if head_i is None or head_j is None:
                # Conservative: missing ref → treat as textual conflict
                textual_edges.add(fp_edge)
                continue
            cache_key = frozenset({head_i, head_j})
            if cache_key in _probe_cache:
                has_conflict = _probe_cache[cache_key]
            else:
                try:
                    probe = await self._git_ops().merge_tree_conflicts(head_i, head_j)
                    has_conflict = not probe.clean
                except Exception:
                    logger.warning(
                        'recompute_suffix_conflict_graph: merge_tree_conflicts raised '
                        'for pair (%s, %s); treating as conflict (fail-open)',
                        suffix[i_idx].request_id, suffix[j_idx].request_id,
                        exc_info=True,
                    )
                    has_conflict = True
                    _any_probe_error = True
                _probe_cache[cache_key] = has_conflict
            if has_conflict:
                textual_edges.add(fp_edge)

        # ── 7. Probe each suffix item vs the frozen-prefix tip (η=1892) ──────────
        # η repoints the probe base from bare main_sha to frozen_prefix_tip().
        # When the frozen prefix is non-empty, the tip is the newest frozen
        # item's merge_commit (the speculative stack top that suffix items must
        # stack onto); when the frozen prefix is empty, frozen_prefix_tip()
        # returns main_sha unchanged, so δ's empty-prefix behaviour is
        # byte-identical to before.  This realizes "bounce the younger, let the
        # older proceed": the older item is already in the frozen prefix
        # (verifying), the younger sits in the suffix — probing the suffix item
        # against the frozen tip flags exactly the younger for a needs_rebase
        # bounce before it consumes a verify slot.
        # The _probe_cache is keyed by frozenset({probe_base, head}) so that any
        # pair already probed in the textual-edge step reuses its cached result.
        probe_base = self._frozen_prefix_tip(main_sha)
        conflicts_with_main: set[str] = set()
        for k, req in enumerate(suffix):
            head = heads[k]
            if head is None:
                # Missing ref → skip (branch gone; no conservative mark here
                # because we can't even tell if the branch exists; skipping is
                # safer than always-conflicting for a deleted-branch item).
                continue
            cache_key = frozenset({probe_base, head})
            if cache_key in _probe_cache:
                has_main_conflict = _probe_cache[cache_key]
            else:
                try:
                    probe = await self._git_ops().merge_tree_conflicts(probe_base, head)
                    has_main_conflict = not probe.clean
                except Exception:
                    logger.warning(
                        'recompute_suffix_conflict_graph: merge_tree_conflicts(frozen_tip, %s) '
                        'raised for item %s; treating as conflicts_with_main (fail-open)',
                        head, req.request_id,
                        exc_info=True,
                    )
                    has_main_conflict = True
                    _any_probe_error = True
                _probe_cache[cache_key] = has_main_conflict
            if has_main_conflict:
                conflicts_with_main.add(req.request_id)

        # ── 8. Store ──────────────────────────────────────────────────────────
        self.graph = SuffixConflictGraph(
            nodes=ordered_rids,
            textual_edges=frozenset(textual_edges),
            footprint_edges=frozenset(footprint_edges),
            conflicts_with_main=frozenset(conflicts_with_main),
        )
        # Only cache the debounce signature when every probe succeeded.  A
        # transient error leaves the signature un-stored so the next tick
        # re-probes rather than serving a stale conservative result behind the
        # debounce.
        if not _any_probe_error:
            self.signature = sig

    async def bounce_conflicting_suffix_items(self) -> None:
        """η=1892 graph-time bounce: divert suffix items that conflict with the frozen tip.

        For each request_id in ``graph.conflicts_with_main``:

        1. Skip :class:`GroupMergeRequest` items (trains keep their existing
           TRAIN_REBASE_CONFLICT path).
        2. Emit a structured ``needs_rebase`` log line.
        3. Bump the bounce registry; if the count exceeds :data:`MERGE_BOUNCE_CAP`
           → escalate WITHOUT rebasing (cap exceeded).
        4. Else: attempt a mechanical rebase onto the frozen tip via
           ``rebase_onto_main(req.worktree, onto=frozen_tip)``.
           - True (clean) → leave the item in the lane buffer (re-queue);
             the future and ``merge_first_enqueued_at`` are untouched.
           - False (real conflict) → remove from the lane buffer and escalate.

        After processing any bounce, set ``signature = None`` so the next
        :meth:`recompute` call re-probes reality (prevents a successfully-
        rebased item from being re-bounced off a stale graph).

        Called by :meth:`SpeculativeMergeWorker._acquire_next_request`
        immediately after :meth:`recompute` and before ``_pop_next_pickable()``,
        so bounced/escalated items are diverted before consuming a verify slot.

        Fail-open: if ``get_main_sha()`` raises, log a warning and return
        without touching any item.
        """
        # Deferred reach-back: these three constants stay in merge_queue.py
        # (not moved). A top-level import would deadlock module load — see
        # module docstring.
        from orchestrator.merge_queue import (
            MERGE_BOUNCE_CAP,
            MERGE_LANES,
            NEEDS_REBASE_REASON_PREFIX,
        )

        if not self.graph.conflicts_with_main:
            return  # nothing to bounce — leave signature intact

        # Use the same main_sha that recompute() used to populate
        # conflicts_with_main.  A concurrent asyncio task (e.g. a verify
        # completion that mutates _inflight) can change the frozen prefix between
        # recompute() and here; a freshly-fetched main_sha would give a frozen_tip
        # inconsistent with the probe_base that actually flagged the conflict —
        # causing an unnecessary rebase/escalation off a stale graph (TOCTOU).
        # Reading sig[1] (the main_sha stored in the debounce signature) gives
        # exactly the same base used during recompute().
        # Fall back to a fresh fetch only when the signature is None (cleared by a
        # prior bounce in the same acquire cycle — rare, but handle fail-open).
        if self.signature is not None:
            main_sha = self.signature[1]
        else:
            try:
                main_sha = await self._git_ops().get_main_sha()
            except Exception:
                logger.warning(
                    '_bounce_conflicting_suffix_items: get_main_sha failed; skipping bounce'
                )
                return

        frozen_tip = self._frozen_prefix_tip(main_sha)

        # Build a rid → MergeRequest index over all lane buffers.
        rid_to_req: dict[str, MergeRequest] = {}
        for lane in MERGE_LANES:
            for req in self._lane_buffers()[lane]:
                rid_to_req[req.request_id] = req

        _any_bounced = False

        for rid in self.graph.conflicts_with_main:
            req = rid_to_req.get(rid)
            if req is None:
                continue  # item already gone (escalated or removed)
            if isinstance(req, GroupMergeRequest):
                continue  # trains keep their own TRAIN_REBASE_CONFLICT path

            branch = req.branch
            logger.info(
                '_bounce_conflicting_suffix_items: needs_rebase task_id=%s rid=%s '
                'branch=%s frozen_tip=%s',
                req.task_id, rid, branch, frozen_tip,
            )

            count = self.bounce_registry.record_bounce(branch)
            _any_bounced = True

            if count > MERGE_BOUNCE_CAP:
                # Cap exceeded — escalate WITHOUT rebasing.
                #
                # Trade-off note (robustness_premature_escalation): frozen_tip
                # is speculative — the frozen items are still verifying and may
                # themselves fail verification.  A branch that is clean vs bare
                # main but conflicts with frozen_tip may be escalated for a
                # collision with an item that never lands.  This is accepted: the
                # steward resolves it, and if the frozen item later fails, the
                # branch can be cleanly requeued without prejudice.
                self._lane_buffers()[req.lane].remove(req)
                self.bounce_registry.clear(branch)  # fresh slate on resubmission
                outcome = MergeOutcome(
                    status='blocked',
                    reason=(
                        f'{NEEDS_REBASE_REASON_PREFIX}: branch {branch!r} '
                        f'bounce cap exceeded (count={count}, cap={MERGE_BOUNCE_CAP})'
                    ),
                )
                logger.warning(
                    '_bounce_conflicting_suffix_items: cap exceeded task_id=%s '
                    'branch=%s count=%d cap=%d; escalating without rebase',
                    req.task_id, branch, count, MERGE_BOUNCE_CAP,
                )
                if not req.result.done():
                    req.result.set_result(outcome)
                continue

            # Attempt mechanical rebase onto the frozen tip.
            clean = await self._git_ops().rebase_onto_main(req.worktree, onto=frozen_tip)
            if clean:
                # Clean rebase — item stays in the lane buffer (re-queued).
                # Future and merge_first_enqueued_at are untouched.
                #
                # Downstream path note (design_double_rebase): the next call to
                # _pop_next_pickable() may pick this item for dispatch.  The
                # dispatch loop calls merge_to_main() which performs a
                # ``git merge --no-ff`` — NOT another rebase — so there is no
                # double-rebase hazard.  If actual_main has advanced past
                # frozen_tip since the bounce, the merge integrates those extra
                # main commits; this is expected behaviour for a speculative
                # queue.  pre_rebased is intentionally not updated: skip_verify
                # is always False for single-item merges (task-1724), so updating
                # it would be a no-op.
                logger.info(
                    '_bounce_conflicting_suffix_items: clean rebase; re-queued '
                    'task_id=%s branch=%s onto=%s',
                    req.task_id, branch, frozen_tip,
                )
            else:
                # Real conflict — remove from lane buffer and escalate.
                #
                # Trade-off note (robustness_premature_escalation): frozen_tip
                # is speculative — frozen items are still verifying and may fail.
                # A branch clean vs bare main can thus be escalated for a
                # collision with an item that never lands.  Accepted: the steward
                # resolves it; if the frozen item fails, the branch requeues.
                self._lane_buffers()[req.lane].remove(req)
                self.bounce_registry.clear(branch)  # fresh slate on resubmission
                outcome = MergeOutcome(
                    status='blocked',
                    reason=(
                        f'{NEEDS_REBASE_REASON_PREFIX}: branch {branch!r} '
                        f'has a real rebase conflict onto frozen tip {frozen_tip!r}'
                    ),
                )
                logger.warning(
                    '_bounce_conflicting_suffix_items: rebase conflict task_id=%s '
                    'branch=%s onto=%s; escalating',
                    req.task_id, branch, frozen_tip,
                )
                if not req.result.done():
                    req.result.set_result(outcome)

        if _any_bounced:
            # Invalidate the debounce signature so the next recompute re-probes
            # reality.  A clean rebase mutates the branch HEAD; an escalated
            # item removes an entry — in both cases the cached graph is stale.
            self.signature = None
