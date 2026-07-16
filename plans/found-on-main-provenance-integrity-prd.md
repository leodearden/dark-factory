# PRD — found_on_main provenance integrity: reopen-freshness gate + attribution tightening

**Status:** active · 2026-07-16 · dark_factory
**Source:** RCA `~/.claude/spawn-briefs/found-on-main-rca-2026-07-16-REPORT.md` (session rca-df-107414; 7 mining lanes + 3 adversarial skeptics; Leo-ratified 2026-07-16). Supersedes cancelled task 2669's refuted premise.

## Goal

Close both faces of the proven task-status-correctness cluster:

- **Face A (fabricated attribution):** `found_on_main` done-stamps citing commits unrelated to the task (tasks 2394 ×2, 2531, +27 historical). A user/operator observes: no new task ever reaches `done` citing a sha that doesn't carry that task's work; when evidence can't be attributed, the task escalates loudly instead of silently completing.
- **Face B (re-derivation clobber):** a legitimate `done→pending` reopen re-marked `done` within one dispatch tick (~20s; task 1175, proven live post-2500/2649). A user/operator observes: a reopen sticks; if a gate believes the task is done anyway, exactly one dedup-guarded L2 `provenance_conflict` escalation appears for human arbitration instead of a silent re-complete.

## Background

The RCA identified a family of ≥6 always-on "already-landed" re-derivation sites sharing one substrate (git-ancestry-class evidence ⇒ `mark_done`) with inconsistent guards:

| Site | Guards today |
|---|---|
| Dispatch-gate ancestry path `harness.py:7383-7451` | FIX 1 + FIX 2 (task 2500) |
| Dispatch-gate merge-marker path `harness.py:7453-7468` | none (clobbered 1175, note at `:7465`) |
| Dispatch-gate content-equivalence path `harness.py:7471-7485` | none; `anchor = citation or get_main_sha()` stamps main's tip |
| Stranded-in-progress sweep `harness.py:3592-3635` | FIX 1 only |
| LandedOutbox `reconcile_landed_row` RC-2 `merge_queue.py:3779-3782` | none; rows never consumed on happy path (task 2155 KNOWN LIMITATION) |
| Coalesce re-drive `merge_queue.py:8980` | `resolve_branch_sha(branch) or main_sha` fallback |

Two structural facts drive the design: (1) **every done-write funnels through one server-side chokepoint** — the fused-memory interceptor `_apply_status_transition` → `_validate_done_provenance` (`task_interceptor.py:987-1030`), which already shells git and holds the pre-write metadata snapshot (incl. `reopen_at`); (2) **task 2500's FIX 1 (`commit_effect_present_in_main`) is a no-op on merge commits** (empty diff-tree ⇒ unconditional True, `git_ops.py:6185-6187`) — and all three observed bad shas were merge commits. Neither face's fix subsumes the other: perfect attribution would not have saved 1175 (its marker sha is genuinely its own), and a freshness guard cannot see a first-time misattribution on a never-reopened task.

## Sketch of approach

**R1 — server-side reopen-freshness gate (chokepoint).** In `_validate_done_provenance`, for commit-bearing kinds (`merged`, `found_on_main`) on a task whose `metadata.reopen_at` is set: resolve the evidence commit's committer date (`git show -s --format=%cI`), TZ-normalize both sides to aware UTC, and **reject** the done-write when evidence-date < `reopen_at`, with typed error `done_evidence_stale` carrying structured facts (task_id, evidence sha, evidence committed-at, reopen_at, caller agent_id). Fail closed on unparseable dates. Commitless kinds (`deterministic-*`, new `operational-verified`) are exempt (they self-evidence). Ships warn-mode behind `reconciliation.reject_stale_done_evidence: warn|enforce` (default `warn`), flipped to `enforce` by a dedicated task after orchestrator-side handling lands.

**R5 — the escalation arm.** Orchestrator done-writers (dispatch gates, sweeps, LandedOutbox reconcile) treat `done_evidence_stale` as **terminal-for-this-tick** (no retry loop, per-task memo) and file **one dedup-guarded born-at-L2** `provenance_conflict` escalation (fingerprint: task_id + evidence sha; storms surface as `dedupe_count` increments on the single record, never log spam). The **sanctioned override** for "the reopen was wrong — it really is done": the arbitrating (non-recon) session resolves that escalation, then issues the done-write with first-class `done_provenance.stale_evidence_override = {escalation_id, reason}`. The interceptor's machine-checked acceptance rule uses existing substrate: the override is accepted **only from non-recon-stage callers** (the caller classification `recon_write_policy` already applies) and only with both fields present — recon-stage callers can never override; the recorded `escalation_id` is audit substrate for Stage-2 recon (fused-memory has no client for the orchestrator's escalation server, so live cross-service verification is deliberately not assumed — G3).

**R2 — attribution tightening (sender side).**
- Delete both silent fallbacks (`or get_main_sha()` `harness.py:7479`; `or main_sha` `merge_queue.py:8980`): no attributable citation ⇒ **do not stamp — escalate** (loud-over-silent).
- Extract **one shared evidence-validation helper** (single site; INV-5) applying citation-lineage (FIX 2) + effect-present (FIX 1′) uniformly across all five orchestrator branches (ancestry, merge-marker, content-equivalence, stranded sweep, coalesce re-drive).
- **FIX 1′:** for a merge-commit evidence sha, diff the merge's **second-parent (branch) content** against main instead of the merge commit's own empty diff-tree, so reverted/absent work fails the effect check (the 1175 "reverted" shape).
- Anchor `DEFAULT_COMMIT_CITATION_PATTERN`'s `task/{tid}` alternative to the subject line so body-prose mentions can't create citations.

**R3 — LandedOutbox hygiene.** Consume the row on happy-path completion (closes the RC-2 stale-row window; startup RC-3 pruning already treats rows as disposable).

**R6 — operational-task closure (adjacent).** New commitless `done_provenance.kind='operational-verified'` requiring `escalation_id` (the resolving escalation, recorded for audit; shape-validated at write, matching the shipped `deterministic-gate` precedent — no live cross-service check) — the honest closure for no-code operational tasks (2648/2650-class), ending the found_on_main-shaped workarounds and the task_kind thrash. Plus a submit-time **warn-only** suggestion when a `task_kind='normal'` submission looks operational (restart/confirm phrasing, no files): suggest `task_kind='deterministic'`/`execution_class`, never coerce.

**Soak.** A `deterministic` delayed-milestone predicate task (+7 days after the fixes deploy) re-runs the provenance audit and asserts **zero new spurious `found_on_main` stamps with stamp-time after the deploy**; non-zero ⇒ `milestone_check_failed` L2.

## Resolved design decisions

1. **Server-side chokepoint, not client-side:** the freshness gate lives in the interceptor so all present and future writers are covered; a `Scheduler.mark_done`-only guard is explicitly rejected (RCA what-NOT-to-do).
2. **Two guards, not one:** attribution tightening (Face A) and reopen-freshness (Face B) are complementary; neither is dropped in favor of the other.
3. **Escalate-on-conflict, not silent re-complete or hard loop:** typed rejection + single dedup-guarded L2; expected volume low single digits/day (audit baseline: 66 found_on_main, 16 ok).
4. **Override is schema'd:** `stale_evidence_override={escalation_id, reason}` with a machine-checked acceptance rule (non-recon-stage caller class + both fields required — INV-1) and the escalation_id recorded as audit substrate; live cross-service escalation verification is explicitly NOT assumed (fused-memory has no orchestrator-escalation client — G3-verified).
5. **Warn→enforce rollout:** `reconciliation.reject_stale_done_evidence` defaults `warn` at α; a dedicated flip task turns on `enforce` only after orchestrator-side handling (β) lands; fused-memory restart is an explicit deterministic gate; orchestrator units ride the normal ≤8h fleet redeploy (soak's 7d delay absorbs it).
6. **One shared helper for evidence validation** across all five orchestrator branches — extraction over five copies (INV-5).
7. **Gates stay:** the already-landed gates catch genuine out-of-band landings (task 2313's purpose); this PRD guards them, never amputates. Also rejected (RCA-proven wrong): re-doing 2649's atomic write, terminal-gating `set_task_status`, branch-deletion-on-reopen, widening task 2085's phrase allowlist.

## Pre-conditions for activating

None blocking: tasks 2500 (partial hardening), 2649 (atomic reopen write), 2645/2648/2667 (audit tooling + backlog correction) are all `done` on main. The audit script `fused-memory/scripts/audit_found_on_main_provenance.py` exists (exercised live by 2648).

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| task 2433 (pending; TargetedReconciliation stale mem0-echo on reopen→redone) | adjacent, disjoint | mem0 echo hygiene — different mechanism, **not superseded** | task 2433 | independent |
| `plans/capability-delivered-checks-prd.md` | consumes (mechanically) | YAML sidecar + `commit_planning` stamper | that PRD | wired (stamper live) |
| escalation-lifecycle dashboard batch (2656-2666) | produces into | new `provenance_conflict` category flows through existing lifecycle/resolution_class machinery | this PRD emits, existing machinery consumes | wired |

No contested-ownership seams.

## Contract (B+H): the `done_evidence_stale` seam

**Producer:** fused-memory interceptor `_apply_status_transition`/`_validate_done_provenance`.
**Consumers:** every orchestrator done-writer reaching `set_task_status(status='done')`.

- Rejection shape: `{'success': False, 'error': 'done_evidence_stale', 'task_id', 'evidence_commit', 'evidence_committed_at', 'reopen_at', 'agent_id'}` — structured facts, no prose-parsing required (INV-2). In `warn` mode: write proceeds, one WARNING census line `task_status.done_evidence_stale_warn task_id=<id> ...` (stable grep anchor).
- Scope: fires only when `metadata.reopen_at` present AND `done_provenance.kind ∈ {merged, found_on_main}` AND evidence committer-date < `reopen_at` (aware-UTC comparison; unparseable ⇒ reject in enforce, warn-line in warn).
- Override: same call shape + `done_provenance.stale_evidence_override={escalation_id, reason}`; interceptor accepts iff caller is non-recon-stage (existing `recon_write_policy` classification) AND both fields are non-empty; otherwise `done_evidence_stale_override_invalid`. The escalation_id is recorded verbatim for Stage-2 audit.
- Consumer obligation: on `done_evidence_stale`, do NOT retry this tick or next until either `reopen_at` changes or the conflict escalation resolves; file/refresh the single dedup-guarded `provenance_conflict` L2 (born-at-L2, severity urgent, fingerprint task_id+evidence sha).
- Ordering: α (producer, warn) → β (consumers + escalation) → γ (enforce flip). Enforce is never on before consumer handling exists.

## Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | 1175-shape clobber blocked | real sqlite backend; task done→pending reopened (reopen_at stamped); gate attempts re-done citing pre-reopen merge marker; enforce on | status stays `pending` on fresh read; response `error=done_evidence_stale` with structured fields; exactly one pending `provenance_conflict` L2; second gate attempt increments `dedupe_count`, files nothing new |
| 2 | Legit crash-recovery re-complete passes | reopened task genuinely re-merged; evidence commit newer than reopen_at | done-write accepted; no escalation |
| 3 | Override path | scenario-1 arbitrated; done-write re-issued with `stale_evidence_override` from an interactive (non-recon) agent_id | accepted, escalation_id recorded; same override from a recon-stage agent_id ⇒ rejected `done_evidence_stale_override_invalid`; missing reason/escalation_id ⇒ rejected |
| 4 | Warn-mode telemetry | same as 1 but `warn` | write proceeds; census WARNING line emitted with same structured fields |
| 5 | FIX 1′ catches reverted merge | merge commit on main whose second-parent content absent at HEAD | shared helper reports effect-absent; gate escalates, does not stamp |
| 6 | No-citation ⇒ no stamp | branch content-equivalent on main, no attributable citation | no done-write; escalation carries branch + probe evidence; grep finds no `or get_main_sha()`/`or main_sha` fallback in gate paths |
| 7 | Deterministic/operational exempt | DeterministicRunner resume; operational-verified close | freshness gate not applied to commitless kinds |

## Decomposition plan

Labels α–κ, θ; deps intra-batch unless noted. All `task_kind="normal"` code_tdd except κ/θ (deterministic).

- **α — Interceptor reopen-freshness gate (warn-mode) + typed error + override validation** · fused-memory (`task_interceptor.py`, config, tests) · prereqs: — · *Signal:* with enforce set in test config, a stale-evidence done-write on a reopened task returns `error=done_evidence_stale` with all structured fields against the real backend; in warn mode the census WARNING line appears; valid `stale_evidence_override` accepted, invalid rejected. (Intermediate → unlocks β, γ, η.)
- **β — Orchestrator handling: terminal-this-tick + dedup-guarded `provenance_conflict` born-at-L2** · orchestrator (`harness.py`, `scheduler.py`, `merge_queue.py` done-write call sites, tests) · prereqs: α · *Signal:* integration test: on stale rejection the gate skips the candidate this tick (no per-tick retry), exactly one pending L2 `provenance_conflict` exists; a repeat rejection increments `dedupe_count` instead of filing a second. (Intermediate → unlocks γ.)
- **γ — Enforce flip: `reject_stale_done_evidence: enforce` default + end-to-end reopen-sticks test** · fused-memory config + cross-package integration test · prereqs: α, β · *Signal:* boundary-test #1 green end-to-end against real backend + real gate code: reopen persists past a gate tick, one L2 filed. (Leaf.)
- **δ — git_ops evidence primitives: FIX 1′ second-parent effect check + subject-anchored citation regex** · orchestrator (`git_ops.py`, tests) · prereqs: — · *Signal:* unit fixtures: `commit_effect_present_in_main` returns False for a merge commit whose branch content is absent at HEAD (True pre-change); a commit whose body prose mentions `task/N` no longer yields a citation (G6-branch-4: absence observed by test). (Intermediate → unlocks ε.)
- **ε — Call-site tightening: shared evidence-validation helper across all five branches; remove `or get_main_sha()`/`or main_sha`; escalate-instead-of-stamp** · orchestrator (`harness.py`, `merge_queue.py`, tests) · prereqs: δ · *Signal:* integration test: branch-absent + historical-marker + deliverable-absent task (1175 shape) is NOT re-marked done — an escalation with structured evidence is filed instead; `git grep -E 'or (await self\.git_ops\.)?get_main_sha\(\)|or main_sha'` over the gate/re-drive paths returns nothing. (Leaf; also unlocks ζ, θ.)
- **ζ — LandedOutbox consume-on-happy-path** · orchestrator (`merge_queue.py`, `landed_outbox.py`, `workflow.py`, tests) · prereqs: ε (file-overlap serialization) · *Signal:* integration test: after a normal completion the task's LandedRow is consumed (absent from `landed_outbox.json` without restart); startup RC-3 prune count for happy-path completions drops to zero in test. (Leaf.)
- **η — `operational-verified` provenance kind + submit-time operational suggestion (warn-only)** · fused-memory (`task_interceptor.py`, submit lint, docs/CLAUDE.md provenance table, tests) · prereqs: α (file-overlap + validator adjacency) · *Signal:* `set_task_status(done, done_provenance={kind:'operational-verified', escalation_id:..., note:...})` accepted for a no-commit task from a non-recon caller with both fields present, and the record is retrievable via `get_task` showing the new kind; missing escalation_id/note, or a recon-stage caller, ⇒ structured rejection; a restart-phrased `task_kind='normal'` submission emits the suggestion WARNING (never coerces). (Leaf.)
- **ι — Predicate wrapper `fused-memory/scripts/check_found_on_main_spurious_rate.py`** · fused-memory scripts + test · prereqs: — · *Signal:* script wraps `audit_found_on_main_provenance.py` with `--since <ISO>`: exit 0 when zero found_on_main stamps newer than `--since` are flagged misattributed/deliverable-absent, exit 1 with a structured stdout summary otherwise; committed executable. (Intermediate → unlocks θ.)
- **κ — Deploy gate: restart fused-memory to load α/γ/η (deterministic, pure gate)** · no code · `task_kind='deterministic'`, `always_escalates=true`, no `before_done`, `execution_class='operational'` · prereqs: γ, η · *Signal:* born-at-L2 filed; resolution records the restart (`systemctl` timestamps newer than γ/η merge times); task `done` with `deterministic-gate` provenance. (Leaf.)
- **θ — Soak: +7d delayed-milestone predicate re-running the audit** · no code · `task_kind='deterministic'`, `metadata.milestone={mode:'delayed', after_secs:604800}`, `always_escalates=true`, no `before_done` (pure gate; RUNBOOK: run ι's script with `--since <κ resolution time>`; exit 0 ⇒ resolve done, else file/keep `milestone_check_failed`) · prereqs: ε, ι, κ · *Signal:* 7 days after deps land, the gate L2 fires; its resolution records the ι script run output showing zero new spurious stamps. (Leaf.)

*(θ/κ are filed as pure gates rather than `before_done` deploy/predicate presets because their scripts/units aren't bindable at submit time — `submit_task` validates `before_done.script` existence at filing; ι's script lands mid-batch. The L2 resolver — auto-watcher or human — runs the by-then-landed script per the RUNBOOK in the task description.)*

## Out of scope

- Task 2433's mem0-echo supersession (separate mechanism, stays filed).
- `FUSED_ROUTING_INTENT_ENFORCE` flip (separate lever; suggestion stays warn-only here).
- Retro-correction of the 48 historical flags (task 2667, done — its script remains the tool).
- Any change to `set_task_status`'s non-terminal-gated reopen semantics (sanctioned corrective path stays).
- task/1175's deliverable (refiled as task 2673, independent).

## Open questions (tactical)

1. **Warn-mode soak duration before γ flips enforce.** Suggested: γ merely depends on α+β (no timed soak) since boundary-test #1 covers the contract; if noise appears in warn census lines, γ's implementer may add a short hold. Decide at γ.
2. **Exact dedupe fingerprint string format** for `provenance_conflict` (reuse existing `dedupe_fingerprint` conventions). Decide at β.
3. **Whether ε's escalate-instead-of-stamp reuses `escalate_info` vs `escalate_blocker`** for the no-citation case (task not blocked, merely not-completed). Decide at ε.
