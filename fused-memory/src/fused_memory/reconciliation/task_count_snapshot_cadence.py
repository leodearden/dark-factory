"""Task-count snapshot write-cadence hardening — task 2278.

Reify's Mem0 ``task_count_snapshot`` observation (metadata.kind=
'task_count_snapshot', category='observations_and_summaries') is written
per-project by reconciliation Stage 2 (task_knowledge_sync) as its FINAL
action each cycle.  The write depends entirely on the Stage-2 LLM
*remembering* the (memory-stored) Snapshot Discipline norm — there is no
structural guarantee.  This module provides the pure, dependency-free
building blocks for two structural guards:

1. A Stage-2 freshness stat (``report.stats['task_count_snapshot_mem0_written']``)
   computed deterministically in Python from the run-window timestamp of
   existing ``kind='task_count_snapshot'`` Mem0 records — see
   :func:`extract_snapshot_written` and the ``_verify_task_count_snapshot_written``
   helper in ``stages/task_knowledge_sync.py`` that produces the stat.
2. A harness consecutive-full-cycle-miss escalation — see
   :func:`evaluate_snapshot_cadence` and :func:`build_stale_snapshot_finding`,
   wired into ``harness.py``'s ``_maybe_escalate_stale_task_count_snapshot``.

Task 2325 follow-up hardens the write itself rather than only observing it:
:func:`build_task_count_snapshot_content` is the pure renderer for a
DETERMINISTIC Python ``add_memory`` write performed at the end of Stage 2's
``run()`` (the ``_write_task_count_snapshot`` I/O helper in
``stages/task_knowledge_sync.py``), so the write no longer depends on the
Stage-2 LLM remembering the memory-stored "Snapshot Discipline" norm. Gated
on ``not is_snapshot_write_blocked(project_id)`` (see
``reconciliation/policies``) — projects that have never used the per-project
census (currently dark_factory, solar_challenge_platform) are exempted
rather than given a write nobody consumes.

Structural template: mirrors :mod:`fused_memory.reconciliation.stage1_stall_detector`
(threshold constant + pure compute helper + escalation), but per-project (not
per-task) and journal-backed (not Mem0-marker-backed) — see design_decisions
in plan.json for task 2278.

Task 3045 renamed the two persistence-claim-shaped stat keys to carry a
``mem0_`` infix (``task_count_snapshot_mem0_written`` /
``task_count_snapshot_mem0_pruned``). Every ``task_count_snapshot*`` stat
here describes operations on **Mem0 ``observations_and_summaries`` records
only**: per the Snapshot Discipline policy (``prompts/stage1.py``) a
task-count snapshot is NEVER persisted as a Graphiti ``temporal_facts``
edge, for ANY project, so the absence of such an edge is never evidence of
a failed or rejected write. The former un-namespaced spellings were dumped
verbatim into Stage 3's payload and the judge prompt, where both read them
as Graphiti persistence claims and filed a false discrepancy. See
:data:`LEGACY_SNAPSHOT_WRITTEN_STAT_KEY` for the read-only back-compat
alias.

This module has zero imports from ``stages/`` or ``harness`` — it is pure and
side-effect-free so both can import from it without a dependency cycle.
"""

from __future__ import annotations

TASK_COUNT_SNAPSHOT_KIND: str = 'task_count_snapshot'
"""Mem0 metadata ``kind`` tag identifying a task-count snapshot observation."""

SNAPSHOT_WRITTEN_STAT_KEY: str = 'task_count_snapshot_mem0_written'
"""Key under Stage 2's ``report.stats`` recording this cycle's freshness check.

Value is ``1`` when a fresh snapshot was confirmed within the run window,
``0`` when confirmed absent, and the key is omitted entirely when the check
was inconclusive (unknown run window or a transient query failure) — see
:func:`extract_snapshot_written`.

Counts **Mem0 ``observations_and_summaries`` records only** (task 3045).
Per the Snapshot Discipline policy (``prompts/stage1.py``), no Graphiti
``temporal_facts`` edge is ever written or even attempted for a task-count
snapshot — for ANY project, blocked or not — so the absence of such an edge
is never evidence of a failed or rejected write. The ``mem0_`` infix exists
because this key is dumped verbatim into Stage 3's payload and the judge
prompt (``json.dumps(report.stats)`` in ``_format_report``); under the
former un-namespaced spelling ``task_count_snapshot_written`` both readers
parsed it as a Graphiti persistence claim, looked for the (correctly
absent) edge, and filed a false "rejected write" discrepancy.
"""

LEGACY_SNAPSHOT_WRITTEN_STAT_KEY: str = 'task_count_snapshot_written'
"""READ-ONLY back-compat alias for :data:`SNAPSHOT_WRITTEN_STAT_KEY` — task 3045.

The pre-rename spelling. **No producer ever emits this key**; it exists
solely so :func:`extract_snapshot_written` can still read ``stage_reports``
blobs persisted by cycles that ran before the rename (see that function's
docstring for why the harness's miss-streak recomputation depends on it).

:data:`SNAPSHOT_PRUNED_STAT_KEY` deliberately gets no legacy twin: it is
write-only observability with no reader anywhere in the tree (not in
``_COMPUTED_STAT_KEYS``, not in ``journal.get_stats()``, no ``.get(...)``
call site), so an alias for it would be dead code.
"""

SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY: str = 'task_count_snapshot_prune_enumerated'
"""Key under Stage 2's ``report.stats`` recording how many existing
``kind='task_count_snapshot'`` Mem0 records ``_prune_task_count_snapshots``
(in ``stages/task_knowledge_sync.py``) enumerated this cycle via
``get_memories_by_metadata``. ``0`` both when enumeration raised and when it
genuinely found nothing — see :data:`SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY`
to tell those two cases apart (task 2646).

Counts raw enumerated members, including any that lack an ``id`` (those are
excluded from the delete pass, not from this count) — so a nonzero
enumerated value paired with a lower :data:`SNAPSHOT_PRUNED_STAT_KEY` is not
by itself evidence of a delete failure; it may simply mean some enumerated
records had no ``id``. Practically unlikely for Mem0 records, but worth
ruling out before reading the gap as a delete-failure signal (amendment
round, task 2646 review).

Conditional presence: see the note after :data:`SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY`
below — populated only when the prune is actually reached this cycle; read
via ``report.stats.get(...)``, never direct indexing.
"""

SNAPSHOT_PRUNED_STAT_KEY: str = 'task_count_snapshot_mem0_pruned'
"""Key under Stage 2's ``report.stats`` recording how many
``kind='task_count_snapshot'`` Mem0 records ``_prune_task_count_snapshots``
successfully deleted this cycle (excludes per-item delete failures; equal to
the function's own ``int`` return value) — task 2646.

Counts deletions of **Mem0 ``observations_and_summaries`` records only**
(``mem0_`` infix added by task 3045, for the same reason as
:data:`SNAPSHOT_WRITTEN_STAT_KEY`). Per Snapshot Discipline
(``prompts/stage1.py``) no Graphiti ``temporal_facts`` edge is ever written
for a task-count snapshot, so there is likewise no Graphiti edge for this
prune to delete and the absence of one is never evidence of a failed prune.

Conditional presence: see the note after :data:`SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY`
below — populated only when the prune is actually reached this cycle; read
via ``report.stats.get(...)``, never direct indexing.
"""

SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY: str = 'task_count_snapshot_prune_enumeration_ok'
"""Key under Stage 2's ``report.stats``: ``1`` unless the
``get_memories_by_metadata`` enumeration call itself raised, in which case
``0``.

This is the crux stat that distinguishes a silent enumeration failure (0
enumerated, 0 pruned, ``enumeration_ok=0`` — the runtime no-op fingerprint
behind the incident that motivated task 2646) from a genuine empty result (0
enumerated, 0 pruned, ``enumeration_ok=1``) — a distinction a single
delete-count int cannot make.

Conditional presence: see the note after :data:`SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY`
below — populated only when the prune is actually reached this cycle. Absent
is a THIRD state, distinct from both ``0`` and ``1`` here: it means the
prune never ran (e.g. the project is write-blocked, or there is no
taskmaster), not that enumeration failed. Read via ``report.stats.get(...)``,
never direct indexing, or the two states collapse.
"""

SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY: str = 'task_count_snapshot_prune_truncated'
"""Key under Stage 2's ``report.stats``: ``1`` when ``_prune_task_count_snapshots``
hit its ``scroll_limit`` cap (enumerated a full page of ``scroll_limit`` records),
``0`` otherwise.

Distinct from :data:`SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY`: enumeration_ok=1
here too, since the ``get_memories_by_metadata`` call itself did not raise —
but a full page means older stale snapshots may remain unpruned this cycle
(see the scroll-cap saturation WARNING in ``_prune_task_count_snapshots``),
so the prune provably did NOT delete every stale record. Without this flag a
truncated/partial prune is reported identically to a clean, complete one
(amendment round, task 2646 review).

Conditional presence (all four ``task_count_snapshot_prune_*``/``_pruned``
stats above, robustness finding — amendment round, task 2646 review):
populated in ``report.stats`` only when ``_write_task_count_snapshot``'s
prune call is actually reached this cycle — i.e. the project is not
write-blocked (see ``is_snapshot_write_blocked``), ``taskmaster`` is
available, and the pre-prune task fetch/filter step did not raise. On every
short-circuit path the keys are left ABSENT, not zeroed — a bare ``0``
would recreate, one level up, the exact enumeration-vs-absence ambiguity
task 2646 exists to resolve. Downstream readers MUST use
``report.stats.get(key, default)``, never direct indexing — mirrors the
pre-existing :data:`SNAPSHOT_WRITTEN_STAT_KEY` / ``extract_snapshot_written``
convention above, kept for the same reason.
"""

TASK_COUNT_SNAPSHOT_MISS_THRESHOLD: int = 2
"""Number of consecutive full-cycle misses before the harness escalates."""

ESCALATION_CATEGORY: str = 'recon_stale_task_count_snapshot'
"""Escalation category for a sustained task_count_snapshot cadence gap.

Registered in ``harness.py``'s ``_RECON_DEDUP_CONFIG.infra_dedupe_categories``
and the 'info'-severity category tuple in ``_escalate`` — this is low-urgency
process/tooling hardening, not an operator-blocking issue.
"""

TASK_COUNT_SNAPSHOT_CATEGORY: str = 'observations_and_summaries'
"""Mem0 ``category`` for the task_count_snapshot write — task 2325.

Used by the deterministic write helper (``_write_task_count_snapshot`` in
``stages/task_knowledge_sync.py``) so the internal ``memory_service.add_memory``
call lands in the same category the pre-existing (LLM-authored) snapshot
writes already use.
"""


def extract_snapshot_written(stage_report: object) -> bool | None:
    """Read the freshness stat off a Stage-2 report.

    Accepts a real ``StageReport`` (attribute access), a raw dict shape (e.g.
    a journal-reconstructed ``_error`` entry or test double), or ``None``.

    Key precedence (task 3045): :data:`SNAPSHOT_WRITTEN_STAT_KEY` is read
    first, and :data:`LEGACY_SNAPSHOT_WRITTEN_STAT_KEY` is consulted ONLY
    when the new key is ABSENT — not merely falsy. A legitimate ``0`` under
    the new key is a CONFIRMED miss and must never be re-read from the
    legacy key, hence the absence sentinel rather than a truthiness check.

    The fallback exists because ``harness._maybe_escalate_stale_task_count_snapshot``
    recomputes its consecutive-miss streak from ``journal.get_recent_runs``
    — i.e. from ``stage_reports`` blobs persisted by cycles that ran BEFORE
    the rename (see ``journal.py``'s stage-report serialization). Without it
    every such historical row reads as ``None``,
    :func:`compute_snapshot_miss_streak` stops at the first one, and the
    ``recon_stale_task_count_snapshot`` escalation silently never fires
    across the rename boundary — fail-quiet, the worst failure mode for a
    guard whose entire job is to notice an absence. The fallback can be
    dropped once no journal rows older than the rename remain inside the
    harness's ``max(20, threshold * 4)`` lookback window.

    Returns:
        ``True`` when the resolved stat ``== 1``, ``False`` when ``== 0``,
        and ``None`` when the report is ``None``, the ``stats`` dict is
        absent, neither key is present, or the value is unrecognized —
        "unknown", never miscounted as a confirmed miss.
    """
    if stage_report is None:
        return None
    if isinstance(stage_report, dict):
        stats = stage_report.get('stats') or {}
    else:
        stats = getattr(stage_report, 'stats', None) or {}
    _ABSENT = object()
    value = stats.get(SNAPSHOT_WRITTEN_STAT_KEY, _ABSENT)
    if value is _ABSENT:
        value = stats.get(LEGACY_SNAPSHOT_WRITTEN_STAT_KEY)
    if value == 1:
        return True
    if value == 0:
        return False
    return None


def compute_snapshot_miss_streak(recent_flags: list[bool | None]) -> int:
    """Count the leading run of consecutive misses in *recent_flags*.

    *recent_flags* is most-recent-first.  Counts consecutive ``False``
    entries from the start, stopping at the first ``True`` (a written cycle
    resets the streak) or ``None`` (unknown — stop, fail-safe: an
    inconclusive cycle must never be counted as either a miss or a reset).
    """
    streak = 0
    for flag in recent_flags:
        if flag is False:
            streak += 1
        else:
            break
    return streak


def evaluate_snapshot_cadence(
    current_written: bool | None,
    prior_flags: list[bool | None],
    *,
    blocked: bool,
    threshold: int = TASK_COUNT_SNAPSHOT_MISS_THRESHOLD,
) -> dict:
    """Decide whether the current cycle's miss should escalate.

    Fail-safe / fail-open rules, checked before the streak is ever computed:

    - *blocked* (project is in ``SNAPSHOT_WRITE_BLOCKED_PROJECTS``): the
      absence of a snapshot is correct-by-design there, so never escalate.
    - *current_written* is not ``False`` (i.e. ``True`` — a fresh snapshot
      was confirmed this cycle — or ``None`` — the check was inconclusive):
      never escalate. Only a CONFIRMED current miss can trigger escalation.

    Otherwise the streak is ``compute_snapshot_miss_streak(prior_flags) + 1``
    (the "+1" is the current confirmed miss), and ``escalate`` is
    ``streak >= threshold``.

    Returns:
        ``{'streak': int, 'escalate': bool}``.
    """
    if blocked or current_written is not False:
        return {'streak': 0, 'escalate': False}
    streak = compute_snapshot_miss_streak(prior_flags) + 1
    return {'streak': streak, 'escalate': streak >= threshold}


def build_stale_snapshot_finding(project_id: str) -> dict:
    """Build the stable per-project finding dict for a stale-snapshot escalation.

    Shape mirrors ``harness._DEAD_OWNER_STORM_FINDING``: only ``category``,
    ``affected_ids``, and ``description`` are set, and none of them vary
    across calls for the same *project_id* — this stable identity is what lets
    ``_escalate``'s content-fingerprint dedup fold every repeat cycle's call
    into a single pending escalation instead of filing one per cycle.
    """
    return {
        'category': ESCALATION_CATEGORY,
        'affected_ids': [f'{TASK_COUNT_SNAPSHOT_KIND}:{project_id}'],
        'description': (
            f'task_count_snapshot has not been written for project {project_id!r} '
            'within the run window for multiple consecutive full reconciliation cycles'
        ),
    }


def build_task_count_snapshot_content(
    project_id: str,
    *,
    total: int,
    done: int,
    cancelled: int,
    active: int,
    other: int,
    highest_task_id: int,
    as_of: str | None = None,
) -> str:
    """Render the human-readable content string for a task_count_snapshot write.

    Pure and dependency-free — no I/O, no imports from ``stages/`` or
    ``harness``. Counts are the caller's responsibility (see
    ``_write_task_count_snapshot`` in ``stages/task_knowledge_sync.py``,
    which derives them from ``filter_task_tree``'s
    ``FilteredTaskTree.{total_count,done_count,cancelled_count,other_count,
    max_task_id}`` plus ``len(active_tasks)``).

    Args:
        project_id: Project the census is scoped to.
        total: Total task count.
        done: Count of done tasks.
        cancelled: Count of cancelled tasks.
        active: Count of active (non-terminal) tasks.
        other: Count of tasks in an unrecognized/other status.
        highest_task_id: Highest top-level task id observed.
        as_of: ISO date string for the census, or ``None`` when unknown — the
            leading "As of {as_of}:" clause is omitted in that case.

    Returns:
        A concise, non-empty human-readable census line.
    """
    prefix = f'As of {as_of}: ' if as_of else ''
    return (
        f'{prefix}project {project_id} task-count snapshot — '
        f'{total} total, {done} done, {cancelled} cancelled, {active} active, '
        f'{other} other, highest task id {highest_task_id}.'
    )


def build_task_count_snapshot_unavailable_content(
    project_id: str,
    project_root: str,
    *,
    as_of: str | None = None,
) -> str:
    """Render the UNKNOWN-sentinel content string for a task_count_snapshot write.

    Pure and dependency-free — no I/O, no imports from ``stages/`` or
    ``harness`` — mirroring :func:`build_task_count_snapshot_content`.

    Used by ``_write_task_count_snapshot`` (``stages/task_knowledge_sync.py``)
    in place of the numeric renderer when a zero-count project_root is not a
    readable git working tree (``resolve_main_checkout`` raises
    ``ValueError``): ``SqliteTaskBackend.get_tasks`` auto-creates an empty
    ``tasks.db`` and returns ``{'tasks': []}`` for *any* path, so a zero
    count there is a false census, not a genuinely empty project. The
    caller pairs this content with ``metadata['snapshot_status'] =
    'unavailable'`` (task 2738).

    CRITICAL: the returned text must never contain a ``<digits>
    <status-word>`` census pair (e.g. "0 total", "0 done") — such text
    would match ``task_filter.COUNT_SNAPSHOT_RE`` and be mis-parsed as a
    numeric snapshot line by ``is_count_snapshot``/``strip_snapshot_lines``.

    Args:
        project_id: Project the census was attempted for.
        project_root: The project_root that failed the git-working-tree
            check.
        as_of: ISO date string for the cycle, or ``None`` when unknown — the
            leading "As of {as_of}:" clause is omitted in that case.

    Returns:
        A concise, non-empty, non-numeric sentinel string.
    """
    prefix = f'As of {as_of}: ' if as_of else ''
    return (
        f'{prefix}project {project_id} task-count snapshot UNAVAILABLE — '
        f'project_root {project_root} is not a readable git working tree; '
        f'counts not collected this cycle.'
    )
