# Truth-propagation record mechanics

**Status:** active — 2026-08-24. Code-tier complement to the skill-text SOP
fixes landed the same day (`1e249b3121` skills, `a48a4f491b` INV-9,
`863970f336` member-chain sweep). Scope ratified by Leo in-session
2026-08-24.

> **Code anchors** verified against main `863970f336` (2026-08-24). Main
> moves fast — cite-by-symbol; re-locate at implementation time.

## Goal

A session that rules an escalated question can write the ruling into the
escalation record **without closing it**, cheaply and first-class; a
session that resolves a record is shown its unresolved twins before it
moves on; the decision about whether convention suffices is scheduled,
not forgotten; and machine-read world-claims stop silently enforcing
rotted premises. All four are the code half of INV-9 `one-fact-one-home`
(`docs/legibility/design-invariants.md`): the escalation record + git are
the two homes for a ruling; every other surface points.

## Background

The answered-but-unrecorded investigation (esc-6107-7; watcher field
report 2026-08-22→24) measured five pending L2s whose questions were
already ruled — 30.2 answered-yet-open days, worst case a fully-verified,
Leo-released fix (task 3875) not shipping for 6.8 further days. The class
is **category-agnostic** (3 `design_concern`, 1 `dependency_discovered`,
1 `infra_issue`); nothing in this PRD may key on
`category == 'design_concern'`. All five shared one mechanism: one L1
promoted to L2 more than once; `queue.resolve()` cascades down to the
member, never sideways to sibling L2s. The skill-text fixes (ruling-time
amendment rule in `skills/unblock/SKILL.md`; ruled-elsewhere check,
sideways-at-resolve rule, and self-referential-predicate ban in the two
watcher skills) landed 2026-08-24 and are the **consumers** of the
mechanics below.

## Sketch of approach

Four independent leaves, bare-B (no new architecture; each extends an
existing, verified mechanism):

1. **α — `amend_escalation` MCP verb** (escalation server). Append-only
   `Amendment` on a **pending** record; bumps `updated_at` (the watcher
   re-verify trigger); never touches `status`/`severity`/`level`/
   `members`; shares the `promote_to_l2` fold path's caps and lock
   discipline (reuse the existing append/cap helpers — INV-5, no copied
   constants). Replaces the documented promote-fold workaround in
   `skills/unblock/SKILL.md` Step 4, which has a severity-floor side
   effect and requires re-passing `root_cause`/member ids.
2. **β — sideways census in `resolve_issue`'s response**. Report-only:
   the response gains a structured field listing (a) other pending
   records on the same `task_id`, (b) pending L2s sharing any member id
   with the resolved record. No state change, no auto-close — INV-2
   structured facts, consumed by the watcher's "look sideways at every
   resolve" rule.
3. **γ — backstop decision gate**. A pure human gate (`deterministic` +
   `always_escalates`, no `before_done`) on a **dated** milestone
   (2026-09-28): decide whether the convention tier suffices or a
   mechanical backstop ships (TaskInterceptor amend-on-write hook vs a
   harness-sweep disposition). Metric: answered-yet-open days found by
   `scripts/member-chain-sweep.py` during watcher rotations; baseline
   30.2 days / 5 records pre-convention. Dated (not delayed-after-deps)
   deliberately: if α/β stall, the decision still fires — their landing
   state is itself data for the decision.
4. **δ — `delivered_check` premise-recheck discipline**. World-claim
   descriptors (e.g. `expect: absent`) may carry a dated
   `premise: {stated, as_of}`; write-time validation of the shape; the
   scheduler-side evaluator emits a **warn-never-block** signal (info
   escalation through the existing ladder, deduped per check+premise —
   INV-4: bounded, not per-tick) when it enforces a world-claim whose
   premise age exceeds a configurable threshold. A blocking expiry is
   explicitly rejected: it would convert rot into fresh false blocks.

## Resolved design decisions (Leo, 2026-08-24)

- Escalation record + git are the two homes for a ruling; task/PRD/
  manifest/memory carry dated pointers (`esc-id + commit + date`).
- Convention first; the mechanical backstop is a **decision**, not a
  default — hence γ rather than building the hook now.
- Task **4377** (`pin_declared_by`) is a hard prerequisite for any
  disposition-**suggesting** automation. Nothing in this PRD suggests
  dispositions; γ's brief must restate the prerequisite for whatever it
  spawns.
- Nothing auto-closes on member-chain evidence, ever — pins
  (esc-3105-3) are indistinguishable from answered questions.
- Cockpit/DecisionRecord changes are **out of scope** (Leo is
  redesigning the cockpit separately; do not break it further).

## Pre-conditions

All substrate verified against `863970f336`: `Amendment` model + fold
caps (`escalation/src/escalation/models.py`, `queue.py::add_members_to_l2`
— today's sole `updated_at` writer); `resolve_issue` and queue reads
(`server.py`, `queue.py`); `delivered_checks` write-time validation +
scheduler evaluator (see `docs/task-authoring.md` §"delivered_checks");
milestone/pure-gate presets (`docs/task-authoring.md` §5–6);
`scripts/member-chain-sweep.py` (landed today).

## Cross-PRD relationship

| Seam | Other PRD / owner | Direction | Resolution |
|---|---|---|---|
| Amendments machinery | `plans/escalation-l2-tiering.md` (task 3997 landed `amendments`) | extends | α owned **here**; reuses 3997's model/caps, adds a writer |
| Cockpit DecisionRecord / reaper | `plans/fleet-cockpit-prd.md` | none | explicitly out of scope here; inputs for Leo's redesign are in memory (`cockpit-redesign-inputs`) |
| `delivered_checks` vocabulary | `docs/task-authoring.md` (normative doc, no owning PRD) | extends | δ owned **here**; updates the doc in the same change |

## Decomposition plan

| Leaf | Title | Modules | Observable signal | Prereqs |
|---|---|---|---|---|
| α | Add `amend_escalation` MCP verb (append-only, bumps `updated_at`) | `escalation` | An interactive session calls `mcp__escalation__amend_escalation` on a pending record; `get_escalation` then shows the appended amendment with a queue-stamped timestamp and `updated_at` newer than `triaged_at`, while `status`/`severity`/`level`/`members` are byte-identical to before; the same call against a resolved/archived record refuses without mutation | — |
| β | Return a sideways census from `resolve_issue` (same-task + shared-member pending records) | `escalation` | Resolving a fixture L2 whose task has a second pending record and whose member id appears in a third pending L2 returns both in a structured response field; resolving a record with no twins returns the field empty; no listed record's file changes | — |
| γ | Backstop decision gate: does the convention tier suffice? | none (pure gate) | On 2026-09-28 the gate escalates born-at-L2 carrying the decision brief (metric, baseline 30.2d/5, sweep-script invocation, the 4377 prerequisite, the two candidate mechanisms); the task is `done` only via a human `resume` recording the decision in the resolution | — (dated milestone) |
| δ | `delivered_check` premise-recheck: dated world-claim premises, warn-never-block | `orchestrator`, `fused-memory`, `shared`, `docs/task-authoring.md` | A seeded fixture check with `premise.as_of` older than the threshold still evaluates green AND files exactly one deduped info escalation naming the check and premise age; a fresh premise files nothing; a malformed `premise` is rejected at `submit_task` with a structured ValidationError | — |

No intra-batch dependencies — four independent leaves.

## Out of scope

- Any cockpit/DecisionRecord change (Leo's separate redesign).
- Disposition-suggesting or auto-closing automation (gated on 4377; γ
  decides whether to even propose it).
- Reify-side skill copies — the skills are shared dark-factory assets
  already; nothing to port.
- The TaskInterceptor amend-on-write hook itself (γ's subject, not a
  leaf here).

## Open questions (surfaced but not decided in this session)

1. **β's response field name** (`sideways`, `related_pending`, …).
   Suggested: `related_pending`. Decide during β.
2. **δ's premise field shape** — `{stated: str, as_of: date}` vs reusing
   the `evidence` observation shape. Suggested: the former, minimal.
   Decide during δ.
3. **δ's warn threshold + dedup key** — suggested 14 days,
   `(task_id, check_name, premise.as_of)`. Decide during δ.
