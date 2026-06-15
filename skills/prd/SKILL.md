---
name: prd
description: "Author and decompose PRDs under implementation-chain-completeness gates that prevent incomplete/ill-formed work from reaching the orchestrator. ALWAYS use this skill for: /prd commands, authoring a new PRD, decomposing a committed PRD into tasks, queueing tasks from a PRD into the orchestrator. Triggers on requests like 'let's write a PRD for X', 'draft a PRD', 'decompose this PRD', 'queue tasks from <prd>.md', or any mention of starting/finishing PRD-shaped work. Walks G1 (consumer named), G2 (user-observable leaf signal), G3 (assumed-substrate verified), G4 (cross-PRD seam ownership), G5 (design-first when stakes are high), G6 (premise validity), and the meta-gate 'would this PRD produce a complete/coherent/cohesive/good design under decompose-and-queue without further oversight?' before saving or queueing. Adapts to the current project via an optional overlay at .claude/skills/prd/project.md. This is NOT for: editing existing PRDs without re-running gates, running tasks (use /orchestrate), reviewing landed code (use /review), unblocking tasks (use /unblock)."
---

# PRD Authoring + Decomposition

This skill is the **front-end discipline that runs before any task reaches the orchestrator**. It prevents the failure modes that the dark-factory orchestrator's narrow-file-lock model makes expensive to recover from:

- **Orphan producers** — a mechanism built with no named consumer; the integration task never gets queued (G1).
- **Fake-done leaves** — a task marked done with load-bearing wiring absent because its only "signal" was a synthetic-input unit test (G2).
- **Integration starvation** — cross-crate/cross-module integration steps get starved or never queued under narrow file locks; the fix is upfront contracts + two-way boundary tests on high-stakes seams (G5).
- **Grammar/substrate fictions** — the PRD assumes a substrate capability (parser production, API endpoint, schema, CLI flag, library fn) that doesn't exist (G3).
- **False premises** — a leaf signal asserts a number / exactness / capability / rejection that is impossible, misattributed, or unbacked by an active rejection mechanism (G6).

These are **orchestrator-level** failure modes, not project-specific ones — which is why this skill generalizes. Project-specific knowledge (signal vocabulary, the G3 verifier, exemplars, domain hazards, memory namespace) is supplied by a per-project **overlay**; the gates themselves are universal.

## Step 0 — Load the project overlay (do this first, every invocation)

```bash
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ls "$ROOT/.claude/skills/prd/project.md" 2>/dev/null && echo "overlay present" || echo "no overlay — generic mode"
```

If `<ROOT>/.claude/skills/prd/project.md` exists, **Read it** and treat it as authoritative extensions/overrides to everything below: it supplies `project_id`/`project_root` for fused-memory, the PRD output-path convention, the G2 signal vocabulary, the G3 substrate verifier, project G1 sub-checks, G5 seam list + threshold overrides, the G6 domain flag, exemplar PRDs, anti-triggers, and the project memory namespace. The overlay may ship its own reference files under `<ROOT>/.claude/skills/prd/references/` (e.g. a grammar-gate verifier) — Read those when the overlay points to them.

If no overlay exists, run in **generic mode**: the gates still apply, but you elicit `project_id`/output-path conventions from the user, the G2 menu is the generic one, G3 reduces to a manual "does this assumed capability exist?" check, and G6's numeric/exactness branches fire only when a signal actually asserts a number or an exactness claim.

The overlay schema is documented in `references/project-overlay.md`. To specialize this skill for a new project, write that overlay — **do not** create a competing `SKILL.md` under the project's `.claude/skills/prd/` (a personal skill shadows project skills, and a dir without `SKILL.md` is correctly ignored, so the overlay loads cleanly).

## Modes

Pick from context — bare `/prd` invocation:

- No PRD exists yet for the topic → **author mode**.
- PRD is committed and its tasks aren't queued yet → **decompose mode**.
- Both apply (author finished, session has room) → confirm before transitioning.

### Author mode

A conversational design session that ends with a committed PRD on disk. The PRD is the **output** of the conversation, not a template to fill in. The skill drives discussion through the gates, surfaces design choices, helps resolve them, then writes + commits.

Complete when this can be answered "yes":

> If I decompose and queue this PRD without further oversight, will the architecture of what gets implemented be complete, coherent, cohesive, and **good**?

No open **design** questions remain at PRD-save time. Tactical / implementation-time questions are fine and go in `## Open questions`.

See `references/author-mode.md`.

### Decompose mode

Read a committed PRD, re-walk gates, then file the whole task batch via fused-memory `submit_task` with **`planning_mode=True` on every task, no exceptions** (synchronous, curator-bypassing; lands them as `deferred`, returns `task_id` directly). After filing, wire **all** dependencies, then flip the **entire batch** `deferred` → `pending` in a single bulk `set_task_status` call. Fused-memory owns persistence — no commit step.

See `references/decompose-mode.md`.

## Gates

Each gate has a calibrated response level. See `references/gates.md` for what each catches and the exact application algorithm.

| Gate | What | Level |
|---|---|---|
| **G1** | Consumer named for every mechanism introduced (which other PRD or user surface consumes it) | **block** |
| **G2** | Every leaf task names a user-observable signal proving completion | **block** (decompose only) |
| **G3** | Every assumed substrate capability (syntax, endpoint, schema, flag, library fn) is verified to exist OR queued as an explicit prerequisite | **block** |
| **G4** | Cross-PRD seams have a named owner; reciprocal "the other owns it" patterns resolved | **prompt** |
| **G5** | High-stakes / architecturally-complex PRDs use approach **B + H** (contracts + two-way boundary tests) rather than bare B | **prompt with heuristic** |
| **G6** | Every signal asserting a number/exactness/end-to-end capability/rejection has its premise validated — achievable, true, producible from the task's own dependency set, and rejection-mechanism-backed | **block** |
| **Manifest** | Per-leaf capability→evidence bindings committed beside the PRD (mechanizes G3+G6: anti-orphan/wired, anti-inversion, field-population, grammar-fixture, numeric-floor); any FAIL binding blocks queueing | **block** (decompose) |
| **META** | The "yes" question above | **block** at PRD save |

- `block` — the phase cannot complete until the gap is resolved.
- `prompt` — surface the gap, the user decides.
- `prompt with heuristic` — propose a default with reasoning, the user confirms or overrides.

## Outputs

**Author mode:**
- A saved PRD at the project's PRD path convention (from the overlay; generic default is a path the user names). Committed to git in the same skill turn — task agents run in worktrees branched from main and need the PRD on disk before decompose references it.
- Section structure follows the audit-derived shape (match by content, not literal numbering): consumer + user-observable surface; sketch of approach; pre-conditions; resolved design decisions; out of scope; cross-PRD relationship + seam-owner table; decomposition plan (one bullet per task naming its observable signal); open (tactical) questions.

**Decompose mode:**
- A batch of tasks filed via `submit_task` with `planning_mode=True` (always). Each carries metadata fields `user_observable_signal`, `consumer_ref`, and a substrate-confirmed flag (e.g. `grammar_confirmed`).
- All declared dependencies (intra-batch and out-of-batch, including cross-PRD) wired via `add_dependency` while the batch is still `deferred`.
- A committed **capability manifest** beside the PRD binding each leaf signal's asserted capabilities to evidence (mechanizing G3+G6); any FAIL binding blocks the batch until resolved.
- The whole batch flipped `deferred` → `pending` together in a single bulk `set_task_status` call — never one-at-a-time.
- The orchestrator does **not** currently read the `user_observable_signal` / `consumer_ref` / substrate-confirmed metadata fields; they are substrate for a future tracking-infra session. Surface this in the hand-back.

## Conversational style

Terse, technical. No preamble. Surface design choices as 2–4 way option menus via `AskUserQuestion` when the choice is genuinely independent of context; otherwise raise the question inline. Push back if the framing has an unstated assumption. Do not recommend a single answer unless analysis genuinely converges.

## Anti-triggers

- Editing an existing PRD without re-running gates → not this skill. If the edit changes a load-bearing mechanism, run `/prd` author for a fresh design pass.
- Running tasks → `/orchestrate`.
- Reviewing landed code → `/review`.
- Resolving blocked tasks → `/unblock`.
- (The overlay may add project-specific anti-triggers, e.g. an authoring skill for the project's own artifacts.)

## Reference

- `references/gates.md` — G1–G6 + META detail and application algorithms.
- `references/author-mode.md` — conversational flow.
- `references/decompose-mode.md` — fused-memory filing mechanics.
- `references/project-overlay.md` — the overlay schema and how to specialize this skill for a new project.
