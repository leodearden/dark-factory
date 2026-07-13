# PRD: Design-invariants gate — encode the legibility survey's cross-cutting root causes as checkable design invariants (G7)

**Status**: active — authored 2026-07-13 (AFK spawn session from the agent-legibility
survey; enforcement-ladder placement ratified by owner in that session — not
relitigated here).
**Mode**: bare B with a fixture-based integration-gate leaf (ε) facing both
consumers (light H).
**Sources**: `plans/agent-legibility-survey-2026-07-13.md` §3 (the five
cross-cutting root causes) + the spawn brief
`~/.claude/spawn-briefs/prd-design-invariants-gate-2026-07-13.md`.

## Goal

The survey's five cross-cutting root causes become **named, checkable design
invariants**, enforced where design happens:

1. A short normative doc, `docs/legibility/design-invariants.md`, states each
   invariant as a one-line rule + a checkable design question + survey
   evidence + the house pattern that satisfies it.
2. `/prd` decompose gains **gate G7 "invariants pass"**: every task in a batch
   is walked against the invariant questions before filing; a hit blocks
   queueing until the task is redesigned or explicitly waived with recorded
   rationale.
3. `/review` phase 2 audits landed code against the identical list (same doc,
   loaded at run time — no second copy).
4. The confusion census closes the loop (sibling PRD, seam only): incidents
   tagged with the invariant they violate; a repeatedly-violated invariant is
   an enforcement gap → guard task.

Enforcement evidence for why *this* ladder placement (per the brief): a lesson
recorded in curated memory did NOT stop `skills/unblock/SKILL.md` teaching the
two-dot diff trap; hand-transcribed prompt text drifted twice in one file; a
prose contract idled the fast path for ~7,950 tasks — while the
`ValidationError`+hint guard pattern and pinning tests demonstrably held.
Design-time review has no mechanical substrate, so the gate lives in the
design-time machinery (/prd, /review) with machine-checked enforcement
descending via point lints (tasks 2563, 2558) and census measurement.

## Consumers (G1)

- **`/prd` decompose sessions** — G7's gate text (skills/prd) instructs loading
  the doc and walking every task against it. Mechanical, normative consumer.
- **`/review` sessions** — phase-2 reference instructs the same audit over
  landed code. Mechanical, normative consumer.
- **Confusion census** (sibling PRD "continuous confusion reduction") —
  consumes the invariant **id vocabulary** via the optional
  `invariant_violated` codebook field. Indirect.
- **Every future PRD** — indirect (the gate shapes what gets queued).

## The five invariants (content spec for task α)

The doc is a gate checklist, not an essay — one compact block per invariant.
Stable slugs are load-bearing (the census field references them; G7 waivers
name them). Numeric aliases INV-1…INV-5 for prose convenience.

| id | Rule (one line) | Checkable design question(s) | Survey evidence (cite cluster + counts from §3/§1) | House pattern |
|---|---|---|---|---|
| **INV-1 `contracts-machine-checked`** | Any eligibility/routing/capability contract lives where it's consumed, machine-checked — an enforced schema field or a submit-time lint, never description prose or dispatcher-internal heuristics. | Does this feature introduce a contract (eligibility, routing, capability envelope, tool filter/result-envelope convention) that lives only in prose or a dispatcher's internals? Does a new tool/agent surface declare its envelope where callers see it, or is it discovered by failure? | Simple-task fast path dead ~7,950 tasks behind an unadvertised title regex; `prose-routing-intent` (12); `watcher-capability-envelope` (18). | ValidationError+hint guard at the submit boundary (execution_class_guard 2225; routing lint **2563**); server-stamped identity + level gate (2041-2044). |
| **INV-2 `structured-facts-at-failure`** | Emit structured facts at the failure point; never re-derive stories by log-scraping facts the emitter already had in a variable. | Must any consumer of this feature's output parse logs/prose to recover a fact the emitter knew (exit code, step identity, SHA)? Do its escalations/reports separate raw observation (with `measured_at`) from hypothesis? | `block-report-misattribution` (16); `guards-assert-unverified-diagnoses` (14). | Table-driven FailureCategory ladder (2131); structured `evidence` field + observation/hypothesis split (**2558**); FAIL-anchored excerpting. |
| **INV-3 `corroborate-before-acting`** | State read from a snapshot/cache/metadata is re-corroborated against ground truth (git, DB, live process) before an agent or sweep acts on it. | Does this feature act (dispatch, delete, requeue, merge, rewind) on state that could have changed since read? Where exactly is the re-check? | `merge-state-not-git-corroborated` (13); `unverified-task-premises` (15); phantom-done family. | Merge Tier-3.5 git-authority corroboration (2037); `already_merged` genuine-check (5026); `premise_lint()`. |
| **INV-4 `storm-escape-required`** | Every fail-soft path (suppression, fallback, degradation, retry-absorb) carries a rate/streak-threshold escalation — loud-over-silent applied at design time. | If this feature's fallback fires 100× in an hour, who hears about it, and via what counter? | Judge fallback verdicts hid a total subsystem outage (`one-shot-subagent-contract`, 17); curator degrade-to-create; 1755 storm-counter precedent. | Consecutive-streak gate (`merge_liveness.py`, generalized by **2558**); storm counter (1755); LLM-adjudicated guard failing safe to strict. |
| **INV-5 `no-lockstep-duplication`** | No duplicated lock-step logic: two sites that must agree byte-for-byte are one site plus a call (or render-from-source) — extraction over documentation. | Does this feature copy logic, constants, or prompt text that must stay in agreement with another site? What is the shared-helper / render-from-code alternative? | `canonical_queued_branch_name` un-normalized site (`server.py:1000`); already-merged guard duplicated until 5026; sibling-tool envelope divergence; hand-transcribed prompt text drifted twice in one file. | Extract helper (`canonical_queued_branch_name`); render prompts/examples from live schemas (2559) with drift/pinning tests. |

The doc additionally declares:
- **Census seam**: incident records MAY carry optional
  `invariant_violated: <slug>`; the slug vocabulary is THIS doc; the coding
  pipeline is owned by the continuous-confusion-reduction PRD. A slug violated
  repeatedly across census batches = enforcement gap → file a guard task.
- **Fixture pointer**: calibration fixtures live at
  `docs/legibility/design-invariants-fixtures.md` (task ε).

## G7 gate spec (content spec for task β)

Wire into the repo's `skills/prd/` (note: `~/.claude/skills/prd` is a symlink
into the repo — verified 2026-07-13 — so one edit serves interactive and
project use; no second copy, per INV-5).

1. **`skills/prd/SKILL.md`** — add a row to the Gates table after G6:
   `| **G7** | Every task passes the five design invariants (docs/legibility/design-invariants.md); a hit blocks queueing until redesigned or waived with recorded rationale | **block** (decompose; advisory walk in author mode) |`
2. **`skills/prd/references/gates.md`** — new `## G7 — Design invariants pass`
   section between G6 and the Capability Manifest section:
   - **Level**: block (decompose); advisory in author mode (walk the sketch
     against the questions early — cheapest fix point).
   - **What it catches**: designs that re-introduce the survey's five
     cross-cutting root causes (§3): prose contracts, log-scraped stories,
     uncorroborated action, silent fail-soft, lock-step duplication.
   - **Application (decompose)**: after the G6 re-check, Read
     `docs/legibility/design-invariants.md` (it is the single normative list —
     do NOT restate the invariants in this section beyond their slugs, per
     INV-5). Walk **every task in the batch** (not only leaves — violations
     attach to mechanisms, which intermediates introduce too) against each
     invariant's checkable question. Trigger shapes: adds a
     detector/suppressor/fallback without a storm escape (INV-4)? a tool
     without a declared filter/envelope convention (INV-1)? a contract in
     prose (INV-1)? a log-scrape of emitter-known facts (INV-2)? action on
     snapshot state without corroboration (INV-3)? duplicated lock-step logic
     (INV-5)?
   - **Resolution**: redesign the task (add the streak counter, move the
     contract to a schema field/lint, add the corroboration step, extract the
     helper) — or **waive**: record `G7 waiver: <slug> — <rationale>` in the
     PRD's decomposition-plan row AND stamp
     `metadata.g7_waivers: [{"invariant": <slug>, "rationale": <text>}]` on
     the filed task. An unresolved, unwaived hit blocks the batch.
   - Calibration: `docs/legibility/design-invariants-fixtures.md`.
3. **`skills/prd/references/gates.md` application orders** — decompose order
   gains a step between the G6 re-check and the capability manifest ("G7 walk —
   load the invariants doc, walk every task; resolve or record waivers");
   author-mode order gains "G7 alongside the G2/G6 draft walk".
4. **`skills/prd/references/decompose-mode.md`** — insert the same step
   (numbered 2.7 or equivalent) between the G2/G6 walk and Step 2.5's
   manifest, plus the `g7_waivers` metadata line in the Step 3 filing
   template (only present when a waiver was recorded).

## /review consumption spec (content spec for task γ)

1. **`skills/review/references/phase2-architecture.md`** — new step after
   Step 5 (cross-module consistency): *"Step 5.5: Design-invariants audit — if
   `docs/legibility/design-invariants.md` exists at project root, Read it and
   audit the modules in scope against each invariant's checkable question
   (the doc is normative; do not restate it here). Findings carry the
   invariant slug. Classify severity like stub findings (blast radius).
   Record under `invariant_findings` in the phase-2 JSON:
   `{"invariant": <slug>, "file": ..., "line": ..., "issue": ..., "severity": ...}`."*
   Add `invariant_findings` to the Step 8 report schema example and one line
   to the display summary.
2. **`skills/review/SKILL.md`** — one bullet in the Phase 2 step list
   ("audit against docs/legibility/design-invariants.md when present —
   findings keyed by invariant slug").

## Fixtures spec (content spec for task ε)

`docs/legibility/design-invariants-fixtures.md` — calibration + test vector
for both consumers. Per invariant, two seeded violations with expected
verdicts:

- **PRD-leaf-shaped** (G7-facing): a realistic 2-4-line decomposition-plan row
  that violates exactly one invariant (exemplar from the brief: "add a
  suppression counter with no escalation" → INV-4), annotated with the
  expected G7 disposition (`flag: <slug>` + the redesign that clears it).
- **Code-snippet-shaped** (/review-facing): a short illustrative snippet or
  described module shape carrying the same violation, annotated with the
  expected `invariant_findings` entry.

Plus a **rehearsal verdict table**: the implementing agent walks the as-landed
G7 text (β) and phase-2 step (γ) against every fixture and records
verdict-vs-expected in the doc. All ten must flag with the correct slug —
a miss is a defect in β/γ/α wording to fix within this task's scope (wording
only; scope stays prose).

## Sketch of approach

One normative doc; two thin consumers that Read it at run time; fixtures that
pin the behavior; a one-line CLAUDE.md pointer for discoverability. All
surfaces are prose (docs/, skills/, CLAUDE.md) — no production code paths.
CLAUDE.md gets ONLY the pointer (ratified ladder placement: CLAUDE.md is thin
standing rules; the normative doc lives in docs/legibility/): one line in
CLAUDE.md's `## Reference` list —
`- **Design invariants**: docs/legibility/design-invariants.md — five checkable invariants gating /prd decompose (G7) and /review phase 2`.

## Pre-conditions for activating

All substrate verified on main 2026-07-13 (G3):
- `skills/prd/SKILL.md` + `references/gates.md` G1-G6+Manifest+META machinery — present.
- `skills/review/SKILL.md` + `references/phase2-architecture.md` — present.
- `docs/legibility/` — present (confusion-codebook.yaml committed 0691d13263).
- Survey §3 committed (`plans/agent-legibility-survey-2026-07-13.md`).
- `~/.claude/skills/prd` → repo symlink (no drift surface).
No novel substrate; no G3 prerequisite tasks.

## Resolved design decisions

1. **Single normative copy** (INV-5 by construction): gates.md §G7 and
   phase2-architecture.md reference the doc by path and load it at run time;
   neither restates the invariant list. Fixtures likewise live once, beside
   the doc, referenced from both.
2. **Stable slug ids** (`contracts-machine-checked`, …) as the cross-artifact
   vocabulary — waivers, `invariant_findings`, and the census
   `invariant_violated` field all carry the slug. Numeric aliases are prose
   sugar only.
3. **G7 checks every task in the batch**, not only leaves (violations attach
   to mechanisms; intermediates introduce mechanisms).
4. **Waiver contract**: PRD-recorded rationale is normative
   (`G7 waiver: <slug> — <rationale>` in the decomposition row) + a
   `metadata.g7_waivers` stamp on the filed task. Nothing in the orchestrator
   reads the stamp today (same status as `user_observable_signal`; substrate
   for tracking infra).
5. **G7 is a prose-consumed checklist, deliberately** — considered against
   INV-1 and accepted as the ratified ladder placement: the gate's consumers
   are LLM design sessions; design time has no machine boundary. The
   machine-checked descendants are the point lints (2563 routing, 2558
   evidence schema) and the census measurement loop; a repeatedly-violated
   invariant is the signal to build the next mechanical guard.
6. **Fixtures carry expected verdicts in-file** (rehearsal table) rather than
   a CI harness — there is no runnable substrate for prompt-machinery
   assertions; the census loop is the ongoing drift detector.
7. **CLAUDE.md pointer folded into α** (atomic doc+discoverability; the line
   is one-line-merge-trivial against pending 2547's unrelated CLAUDE.md
   edits).

## Out of scope

- The census coding pipeline, nightly trickle, and the `invariant_violated`
  field's implementation — owned by the continuous-confusion-reduction PRD
  (sibling, authored concurrently). This PRD only exports the slug vocabulary
  and the field-name convention.
- Any mechanical submit-boundary lint for invariant shapes — tasks 2563/2558
  are the live point enforcements; future guards are census-driven.
- Reify or other-project overlay updates (their overlays inherit G7 from the
  shared gates.md automatically; project-specific invariant extensions are
  theirs).
- Retroactive audit of already-queued batches.
- Orchestrator-side reading of `g7_waivers` metadata.

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| Continuous-confusion-reduction PRD (sibling session, uncommitted at authoring) | consumes | `invariant_violated: <slug>` optional codebook field; slug vocabulary = design-invariants.md | field+pipeline: census PRD; vocabulary: **this PRD** | coordinated by brief-level convention; census PRD conforms to the slugs |
| Task **2563** (routing-intent lint) | precedent | point enforcement of INV-1 | 2563 | pending — cited, not re-filed |
| Task **2558** (structured evidence + streak gate) | precedent | point enforcement of INV-2/INV-4 | 2558 | pending — cited, not re-filed |
| Task 1746 (done) | precedent | gate-prose edits to skills/prd/references/gates.md flow through the orchestrator | — | landed 2026-06-15 |
| Capability-delivered-checks PRD (sibling session, uncommitted) | none semantic | both edit skills/prd references (file-level contention only; different sections) | — | orchestrator module locks serialize |
| Verify-scope-inversion PRD (sibling session) | none | — | — | — |

No contested seams; no reciprocal-ownership statements.

## Decomposition plan

| # | Task | Modules | Prereqs | Observable signal |
|---|---|---|---|---|
| α | Author `docs/legibility/design-invariants.md` per §The-five-invariants + CLAUDE.md pointer line | `docs/legibility/design-invariants.md`, `CLAUDE.md` | — | **Intermediate** — unlocks β/γ/ε. Doc on main with all five entries (slug, rule, question, evidence, house pattern), census-seam + fixture-pointer paragraphs; `grep design-invariants CLAUDE.md` hits the Reference line |
| β | Wire G7 into skills/prd per §G7-gate-spec (SKILL.md row; gates.md §G7 + both application orders; decompose-mode.md step + `g7_waivers` template line) | `skills/prd/SKILL.md`, `skills/prd/references/gates.md`, `skills/prd/references/decompose-mode.md` | α | **Intermediate** — unlocks ε. Committed gate text a decompose session loads instructs the G7 walk and points at the doc path that exists on main; no restatement of the invariant list (INV-5) |
| γ | Wire /review consumption per §review-consumption-spec (phase2 reference Step 5.5 + report schema + SKILL.md bullet) | `skills/review/references/phase2-architecture.md`, `skills/review/SKILL.md` | α | **Intermediate** — unlocks ε. Committed phase-2 text instructs the invariant audit with `invariant_findings` keyed by slug |
| ε | Fixtures + rehearsal per §Fixtures-spec | `docs/legibility/design-invariants-fixtures.md` | α, β, γ | **Leaf (the gate)** — fixtures doc committed with 2 seeded violations per invariant and a rehearsal verdict table showing every seeded violation flagged with the correct slug by the as-landed β/γ text; first real-world signal: the next /prd decompose in any sibling batch runs G7 |

G2 notes: α/β/γ are intermediates roped into ε per the C-as-integration-gate
pattern; ε's signal is the committed rehearsal table (product surface here IS
the design-time prose machinery). G6: ε asserts a rejection capability ("G7
flags seeded violations") — bound by authoring the violating fixtures and
observing the flag in the rehearsal, producible entirely from ε's own
dependency set (α, β, γ all upstream). No numeric/exactness premises anywhere
in the batch. G7 self-check of this batch: no detector/fallback added (INV-4
n/a — the census loop is the gate's own miss-detector), no prose contract
consumed by a machine boundary (decision 5), no duplication (decision 1), no
snapshot-state action, no log-scraping. Clean.

## Open questions (tactical)

1. Exact fixture wording/count beyond the 2-per-invariant floor — ε
   implementer's call.
2. gates.md step numbering for the G7 insertion (2.7 vs renumber) — β
   implementer's call; keep diffs minimal.
3. Whether reify's overlay later adds project-specific invariants (element
   locking etc.) — out of scope; overlay owner's call when it bites.
