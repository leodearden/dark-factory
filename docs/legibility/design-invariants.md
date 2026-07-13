# Design invariants

A gate checklist, not an essay. These five invariants encode
the agent-legibility survey's cross-cutting root causes
(`plans/agent-legibility-survey-2026-07-13.md` §3) as named,
checkable design-time questions. They will gate `/prd` decompose (G7,
`skills/prd/references/gates.md`) and `/review` phase 2's cross-module
audit — both consumers Read this doc at run time once sibling tasks
β/γ wire the G7 section and phase-2 step (neither lands with this doc);
it is the single normative copy (no restatement, per INV-5). Stable slug
ids are load-bearing: G7 waivers, `/review`'s `invariant_findings`, and
the confusion census's optional `invariant_violated` field all reference
them. Numeric aliases INV-1..INV-5 are prose convenience only.

## INV-1 `contracts-machine-checked`

**Rule**: Any eligibility/routing/capability contract lives where it's
consumed, machine-checked — an enforced schema field or a submit-time lint,
never description prose or dispatcher-internal heuristics.

**Checkable design question(s)**: Does this feature introduce a contract
(eligibility, routing, capability envelope, tool filter/result-envelope
convention) that lives only in prose or a dispatcher's internals? Does a new
tool/agent surface declare its envelope where callers see it, or is it
discovered by failure?

**Survey evidence**: Simple-task fast path dead ~7,950 tasks behind an
unadvertised title regex; `prose-routing-intent` (12);
`watcher-capability-envelope` (18).

**House pattern**: ValidationError+hint guard at the submit boundary
(execution_class_guard 2225; routing lint 2563); server-stamped identity +
level gate (2041-2044).

## INV-2 `structured-facts-at-failure`

**Rule**: Emit structured facts at the failure point; never re-derive
stories by log-scraping facts the emitter already had in a variable.

**Checkable design question(s)**: Must any consumer of this feature's output
parse logs/prose to recover a fact the emitter knew (exit code, step
identity, SHA)? Do its escalations/reports separate raw observation (with
`measured_at`) from hypothesis?

**Survey evidence**: `block-report-misattribution` (16);
`guards-assert-unverified-diagnoses` (14).

**House pattern**: Table-driven FailureCategory ladder (2131); structured
`evidence` field + observation/hypothesis split (2558); FAIL-anchored
excerpting.

## INV-3 `corroborate-before-acting`

**Rule**: State read from a snapshot/cache/metadata is re-corroborated
against ground truth (git, DB, live process) before an agent or sweep acts
on it.

**Checkable design question(s)**: Does this feature act (dispatch, delete,
requeue, merge, rewind) on state that could have changed since read? Where
exactly is the re-check?

**Survey evidence**: `merge-state-not-git-corroborated` (13);
`unverified-task-premises` (15); phantom-done family.

**House pattern**: Merge Tier-3.5 git-authority corroboration (2037);
`already_merged` genuine-check (5026); `premise_lint()`.

## INV-4 `storm-escape-required`

**Rule**: Every fail-soft path (suppression, fallback, degradation,
retry-absorb) carries a rate/streak-threshold escalation — loud-over-silent
applied at design time.

**Checkable design question(s)**: If this feature's fallback fires 100× in
an hour, who hears about it, and via what counter?

**Survey evidence**: Judge fallback verdicts hid a total subsystem outage
(`one-shot-subagent-contract`, 17); curator degrade-to-create; 1755
storm-counter precedent.

**House pattern**: Consecutive-streak gate (`merge_liveness.py`, generalized
by 2558); storm counter (1755); LLM-adjudicated guard failing safe to
strict.

## INV-5 `no-lockstep-duplication`

**Rule**: No duplicated lock-step logic: two sites that must agree
byte-for-byte are one site plus a call (or render-from-source) —
extraction over documentation.

**Checkable design question(s)**: Does this feature copy logic, constants,
or prompt text that must stay in agreement with another site? What is the
shared-helper / render-from-code alternative?

**Survey evidence**: `canonical_queued_branch_name` un-normalized site
(`server.py:1000`); already-merged guard duplicated until 5026;
sibling-tool envelope divergence; hand-transcribed prompt text drifted
twice in one file.

**House pattern**: Extract helper (`canonical_queued_branch_name`); render
prompts/examples from live schemas (2559) with drift/pinning tests.

## Census seam

Incident records MAY carry an optional `invariant_violated: <slug>` field.
The slug vocabulary is *this* doc — the five ids above. The coding pipeline
that populates the field is owned by `plans/confusion-reduction-prd.md`,
which ships the field in its γ task and names this doc reciprocally in its
§10 (Cross-PRD relationship). A slug violated repeatedly across census
batches is an enforcement gap: file a guard task.

## Fixtures

Calibration fixtures — two seeded violations per invariant plus a rehearsal
verdict table exercising the as-landed G7 and `/review` phase-2 text — will
live at `docs/legibility/design-invariants-fixtures.md` once sibling task ε
lands (file not yet present as of this doc's landing).
