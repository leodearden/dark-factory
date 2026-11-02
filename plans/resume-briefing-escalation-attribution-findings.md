# Resume briefing: escalation attribution — findings

**Task:** 3754. **Base measured:** `6dcb4016ab` (2026-10-26).

After `run()`'s ESCALATED wait, the orchestrator resumes the implementer with a
briefing built by
`orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_resume_prompt`.
That briefing has two inputs: `## The Issue` and `## Handler's Resolution`. This
record lists what was wrong with both inputs and with the guard that decides
whether to resume at all, what changed, and the deliberate scope fences.

## 1. The 3713 timeline

**Provenance.** This account comes from the task text, not from a re-read of
the escalation records. On 2026-10-26,
`get_task_escalation_history(task_id='3713')` against
`/home/leo/src/dark-factory/data/escalations` returned `count: 0`. The records
are no longer in that store, so the timeline could not be re-measured here. The
code mechanism below was read first-hand at the base commit.

- Task 3713's resumed agent received the resolution prose of `esc-3713-4`.
- `esc-3713-5` was still pending.
- The resume text was built by
  `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._wait_for_resolution`
  as `'\n'.join(e.resolution for e in get_by_task(task_id) if e.status ==
  'resolved' and e.resolution is not None)`. That resolved-only filter means
  the pending `esc-3713-5` contributed nothing, so `esc-3713-4`'s prose was the
  only text. It reached the agent with no escalation id, no header and no
  separator.
- `## The Issue` was blank (defect 2), so nothing told the agent which
  escalation that prose answered. The agent had to infer the mismatch from
  work that was already committed.

## 2. The three defects (measured at the base commit)

### Defect 1: the resolutions were an unattributed, unordered union

- The tail of `TaskWorkflow._wait_for_resolution` joined every resolved record
  in the task's history, archive included.
- It added no ids, no headings and no separators.
- Its order was filesystem glob order:
  `escalation/src/escalation/queue.py::EscalationQueue.get_by_task` globs the
  queue root and then the archive, and never sorts. `Escalation.resolved_at`
  existed and went unused.

### Defect 2: `## The Issue` rendered blank

- The resume closure in `TaskWorkflow.run` passed
  `'\n'.join(e.summary for e in self._check_escalations())`.
- `TaskWorkflow._check_escalations` returns pending records only.
- `_wait_for_resolution` returns only once no pending L0 remains (its loop
  breaks, or the orphans are dismissed on expiry), and an open hand-off raises.
- So in the normal case the argument was `''`.
- A record filed between the end of the wait and the prompt build (the closure
  runs lazily inside `_invoke`) did appear, but only as a bare summary with no
  id.

### Defect 3: the tail stop-the-line guard saw only level-1 hand-offs

- The entry gate applies `orchestrator/src/orchestrator/workflow.py::_needs_human`
  to every pending record, but only at entry.
- The wait loop polls level 0 only.
- The tail checked only level-1 queue hand-offs (`get_by_task(level=1)` matches
  the level exactly).
- So a record that `_needs_human` selects (a born-at-L2 `critical`/`urgent`, or
  a level-2 queue hand-off) filed DURING the wait was caught by none of the
  three checks, and `run()` resumed the implementer past it.
- The merge-entry path already patched this hole for itself with a single-shot
  `_is_gating_escalation` re-check in `TaskWorkflow._handle_merge_gate_escalations`.
  `orchestrator/tests/test_workflow_merge_gating_strand.py::TestGateReFireIsBounded`
  documented the gap.

## 3. What changed

- **New pure module** `orchestrator/src/orchestrator/resume_resolution.py`. It
  takes already-fetched records and does no I/O:
  - `CarriedResolution`, frozen, with `answered_this_round`;
  - `select_carried_resolutions`, which keeps today's resolved-only filter
    exactly and imposes a deterministic order: `resolved_at`, then `timestamp`,
    then `id`. A round is one ESCALATED wait: a record answers this round when
    its id is not in `resolved_before_round`, the ids already resolved as the
    wait began;
  - `render_carried_resolutions`, one block per resolution headed by its
    escalation id and summary, with `resolved_at` / `resolved_by` / action
    shown as `unknown` when absent. This round's resolutions come first under
    their own heading; earlier rounds' follow under a heading saying each may
    already have been applied. A round with no resolution of its own says so;
  - `render_issue_context`, which names this round's answered escalations,
    earlier rounds' answered escalations, and any still pending. It is never
    empty, and it adds a do-not-act-on-mismatched-text instruction only when
    something is still pending.
- **Structured return value.** `_wait_for_resolution` now returns
  `tuple[CarriedResolution, ...]` (`()` without a queue) instead of a joined
  `str`. `run()`'s resume closure renders both briefing sections from that one
  tuple, so they cannot disagree. The pending read stays lazy inside the
  closure, so a record filed in the race window is named as still pending.
  Only records that await an answer (`escalation.pins.is_queue_handoff`, which
  excludes info) are named as pending, so an open info note does not trigger
  the mismatch warning. The selection is not stashed on the instance: it lives
  for one resume, and a return value carries it without temporal coupling.
- **Narrowed `## Action`.** The resumed agent is told to apply the
  resolution(s) that answer this round's escalation(s), not every resolution
  shown.
- **Widened tail.** After the wait, ONE pending read selects
  `_needs_human(e) or (e.level == 1 and is_queue_handoff(e))` over the full
  pending set.
  - The existing D13 attribution is kept: a steward-filed record that was not
    open at entry raises `_StewardReescalated`; anything else raises
    `_NoStewardApplicable`.
  - The block reason names each record under its own tier: `L2` for a record
    that needs a human, otherwise `L1` (`Gated by open L2 <ids> and L1 <ids>`).
  - An info record never stops the resume.
- **Merge-entry hand-off arm stops the steward.** Widening the tail moved the
  chained-L2 merge-entry shape from the post-wait re-check onto the
  `_HumanHandoff` arm of `TaskWorkflow._handle_merge_gate_escalations`, with
  the steward still live. That arm now stops and clears it before parking, as
  the re-check arm and the gate-cleared exit do, through the shared
  `TaskWorkflow._stop_and_clear_steward`. `run()`'s own hand-off arm is
  unchanged: clearing the reference drops the steward's cost from the report
  `harness.py::_run_slot` reads off `workflow._steward`.

## 4. Scope fences, and why

- **No plain-L0 resume gate.** The wait loop already drains plain blocking
  L0s. Gating them again at the tail risks the resume-spin documented at
  `_wait_for_resolution`'s entry gate. A `_needs_human` hit, by contrast, ends
  in `_mark_blocked`, which halts rather than resumes, so it cannot spin.
- **The all-history union is kept.** An earlier round's ruling may still be
  live. `TaskWorkflow._collect_granted_files` keeps the same all-history union
  on purpose, and narrowing to the newest record would silently drop guidance.
  The defect was that the union was unattributed and unordered, not that it was
  a union. Each item is now tagged with its round instead.
- **The round boundary is a snapshot at the wait's entry.** It is taken after
  `_ensure_steward_started`, which in production only schedules the steward's
  loop (`TaskSteward.start`), so nothing this round is resolved before it.
  *Residual:* a steward still live from the previous round, or a resolver in
  another process, that resolves the new record before the wait begins makes
  it read as an earlier round. It is still shown, under the earlier-rounds
  heading.
- **Ordering is imposed at the consumer, not in `EscalationQueue.get_by_task`.**
  That method has many callers, and sorting there would shift behaviour for all
  of them.
- **The merge-entry re-check is retained** as redundant enforcement. It still
  covers a gating record (including a plain blocking L0) filed after the wait's
  tail read, and at merge entry there is no later gate. Only its comment
  changed.
- **How "tail of the joined set" was read.** The task text says the "tail
  stop-the-line guard inspects only the tail of the joined set". This was read
  as: the tail-of-wait guard inspects only the level-1 subset of the open
  records, not the full set. The alternative reading, the "newest record only"
  selector in `TaskWorkflow._check_infra_resume_thrash`, was considered and
  left alone. That function is a thrash counter keyed on category, not a guard
  against stale text, and widening it would change what it counts.
  *Hypothesis:* the task author meant the guard reading, since "stop-the-line"
  is the code's own name for the `_needs_human` gate. The memory record the
  task cites for this phrase could not be consulted (section 5).

## 5. Memory record 997a3e0f

The task text cites mem0 `997a3e0f`. On 2026-10-26,
`get_memory_by_id(project_id='dark_factory', memory_id='997a3e0f')` returned
`found: False`. No id in any namespace (mem0, mem0 tombstone, graphiti node,
graphiti edge) starts with that prefix, and no tombstone records a deliberate
deletion. The interpretation in section 4 is therefore grounded in code, not
in that record.
