# confusion census 2026-07-31

Project: dark_factory

## Saturation

- batches: 11
- stop reason: saturated
  - batch 0: dup_rate=0.73 (total=20, succeeded=15, failed=5, saturated=False)
  - batch 1: dup_rate=0.84 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 2: dup_rate=0.83 (total=20, succeeded=18, failed=2, saturated=False)
  - batch 3: dup_rate=1.00 (total=20, succeeded=15, failed=5, saturated=True)
  - batch 4: dup_rate=0.80 (total=20, succeeded=15, failed=5, saturated=False)
  - batch 5: dup_rate=0.85 (total=20, succeeded=13, failed=7, saturated=False)
  - batch 6: dup_rate=1.00 (total=20, succeeded=7, failed=13, saturated=False)
  - batch 7: dup_rate=1.00 (total=20, succeeded=11, failed=9, saturated=True)
  - batch 8: dup_rate=0.80 (total=20, succeeded=15, failed=5, saturated=False)
  - batch 9: dup_rate=0.95 (total=20, succeeded=19, failed=1, saturated=True)
  - batch 10: dup_rate=1.00 (total=20, succeeded=12, failed=8, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | prd | implement | recon | ops | unknown |
| --- | --- | --- | --- | --- | --- |
| prd | 2 | 0 | 0 | 0 | 0 |
| implement | 0 | 1 | 0 | 0 | 0 |
| recon | 0 | 0 | 2 | 0 | 1 |
| ops | 0 | 1 | 0 | 4 | 0 |
| unknown | 0 | 0 | 1 | 0 | 3 |

## Synthesis

# Confusion census — 2026-07-31

**Date:** 2026-07-31
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests → per-finding verification against current main (Sonnet) → this synthesis (Fable). Every finding restated below survived the verification stage; this synthesis adds clustering, counting, and code-reading against current main only — no diagnosis appears here that was not itself verified.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** second completed periodic census. Previous: 2026-07-24 (`plans/confusion-census-2026-07-24.md`, 52 findings / 9 clusters). Saturation statistics and filed-task ids are appended by the census runner outside this synthesis.

## Corpus

- **15 verified findings across 14 unique sessions** (one interactive session, 7a2fed96, contributed two).
- Composition: **12 of 15 findings (80%) concern the legibility pipeline's own digest instrument** — up from 37% on 07-24, but the shape changed. The 07-24 headline cluster (the trickle-coder re-ingesting its *own* prompt, 12 sightings) has **zero sightings this cycle**. What dominates instead is the general form of the same classifier defect, observed on ordinary subject sessions: orchestrated-task workers and recon-review sessions whose digests mislabel injected or pasted content as "User Corrections" and score on it.
- The remaining 3 findings are interactive-session surfaces: two cross-session coordination collisions (one session) and one git/tooling failure during PRD landing.
- Phase-stamp coverage: 4/15 findings carry `origin_phase: unknown`, 4/15 `manifested_phase: unknown`. **Zero findings manifested in merge or verify** — the PRD's motivating architect/implement→merge hypothesis is again untested by this batch, and the batch is small (15 vs 52).

## Executive summary (observations)

1. **The instrument, not the subjects, again produced most of the findings — and one verified mechanism explains the two biggest clusters.** The digest score arithmetic on current main (`scripts/legibility/digest.py`, `SIGNAL_WEIGHTS`/`score_signals`) weighs each non-sidechain user turn at 5.0 — a component not itemized in the rendered `signal_counts`. Every mislabeled "User Correction" in this batch is such a counted turn, so a session whose only "user turn" is an injected briefing or a pasted report both (a) renders that content as a correction and (b) scores exactly 5.0 with all five `signal_counts` at zero — precisely what four independent sightings quote. The sightings' alternative hypothesis (divergent scoring code paths) is superseded by this reading of main.
2. **The "User Corrections" mislabel is now confirmed to be source-general, not trickle-coder self-recursion.** Four sightings are ordinary orchestrated-task sessions whose injected `# Context` / `## Project Context` briefing block was filed as a correction; five are recon-review sessions where a pasted "Reconciliation Run Review" stage-report dump was filed as one. The 07-24 census had flagged one architect-block sighting as a hint of generality; this batch settles it.
3. **A fix for part of this landed mid-window.** Task 3278 (done 2026-07-30) discharged the 07-24 census's R1/R2: `is_harness_injected_turn` (digest.py:538) now excludes the orchestrator briefing block and the trickle-coder prompt from both the "User Corrections" section and the gold user-turn score component. Facet (a) of cluster 1.1 and the associated 5.0 scores are within its stated scope. **Facet (b) — pasted report content in a genuine user turn — is outside its scope on current main** (no marker matches pasted recon-review text). This batch's sightings are pre-fix traces unless the next cycle shows otherwise (§6).
4. **The digest's signal channel has fidelity defects in both directions**: `watcher-rearm.sh`'s designed bounded-wait ceiling (exit 124, self-declared in stdout as a normal outcome) was counted as `tool_error` three times in one session — inflating the error signal 4× and burying the session's one real failure — while another session's three genuine `not_found` events were rendered with no path or command, collapsing to an uninformative repeated template.
5. **Outside the instrument, all three findings involve shared mutable state between concurrent or successive operations**: a sibling planning session consumed a design doc before its addendum landed and encoded the stale version into a PRD; a MEMORY.md edit was rejected because a concurrent writer touched the file between read and write; and the Bash tool's persistent working directory made a relative-pathspec `git commit --only` fail twice while landing PRD docs.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (15 total). `merge` and `verify` kept explicitly to show their zeros.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| prd | 2 | · | · | · | · | · | · | · | · | **2** |
| implement | · | · | 1 | · | · | · | · | · | · | **1** |
| recon | · | · | · | · | · | · | 2 | · | 1 | **3** |
| ops | · | · | 1 | · | · | · | · | 4 | · | **5** |
| unknown | · | · | · | · | · | · | 1 | · | 3 | **4** |
| **total** | **2** | **0** | **2** | **0** | **0** | **0** | **3** | **4** | **4** | **15** |

Readings (observational):

- The diagonal dominates again; no cross-phase propagation is visible in this batch.
- Stamp consistency is weak for identical shapes: the same briefing-block mislabel is stamped ops/ops in three sightings and ops/implement in one; the same pasted-report mislabel spans recon/recon, recon/unknown, unknown/recon, and unknown/unknown. The matrix's cells partly reflect stamping variance, not phenomenon variance (§6).

## 1. Verified clusters

Ordered by sighting count. Session ids abbreviated to 8 chars.

### 1.1 Non-corrective turns filed as "User Corrections" — the classifier defect is source-general (9 sightings, 9 sessions)

**Facet (a) — injected task-briefing blocks in ordinary orchestrated-task sessions (4 sightings: 8860621c, a7805e53, 4c5bf231, b90b9d91).** Each session is `agent_class: orchestrated-task` (tasks/worktrees 3075, 2989, 3208 among them), and the sole flagged "User Correction" is the standard orchestrator briefing injection — `# Context` → `## Project Context` followed by the memory-search JSON dump every dispatched task receives at session start. Prior codebook candidates (entry-cand-20260722-10/-13/-16/-17/-19/-21/-24/-26/-27/-30/-31) documented this mislabeling only as the trickle-coder ingesting its own prompt; these four sightings confirm the heuristic fires on any injected early-turn block regardless of source, putting every dispatched task's digest at risk of fabricated corrections, not just the trickle-coder's meta-sessions.

**Facet (b) — pasted reconciliation run-review dumps in recon-review sessions (5 sightings: 2b93092c, 5871ae29, 47c7fd98, ccf68b39, 47e690d3).** The flagged content is a full "Reconciliation Run Review" — run metadata plus per-stage stats JSON, in one case from a different project's run — pasted into a turn for review/discussion. It is report/tool output, not anyone correcting the agent. One sighting (2b93092c) additionally shows the captured payload truncated mid-field (`...[item truncated]`), discarding whatever substance the turn had.

Fix status, verified against current main: task 3278 (done 2026-07-30) landed `is_harness_injected_turn` (digest.py:538), which excludes facet (a)'s briefing block — matched as the co-occurrence of the three line-anchored headings `# Context` / `## Agent Identity` / `# Task` — and the trickle-coder prompt. No marker on main matches facet (b)'s pasted-report shape; that facet remains live by construction. Whether this census's digests were generated before or after the fix is not established by the sightings (§6).

Evidence (representative): `## User Corrections — (turn 3) # Context / ## Project Context / { "results": [ { "id": "ccf73ca4-...", "content": "Task 1470 wired /audit into /review Phase-2 Architectural Coherence." } ... ] }`; `(turn 3) ## Reconciliation Run Review / ### Run Metadata / Run ID: 84e6ce03-...`

### 1.2 Digest score decoupled from the rendered `signal_counts` (5 sightings, 5 sessions)

Sessions: 2b93092c, 47c7fd98, fffb63d4, 47e690d3 (each quoting `score: 5.0` with `tool_error`, `self_correct`, `not_found`, `df_guard`, `interrupt` all 0), plus ccf68b39 (all counters zero, digest surfaced and a "correction" filed anyway; score not quoted). Four of the five overlap cluster 1.1(b) — the mislabeled turn and the score are the same event.

Reading the score arithmetic on current main resolves the observed decoupling: `score_signals` adds `SIGNAL_WEIGHTS['user_turn'] * n_user_turns` (5.0 per non-sidechain user turn) on top of the weighted counts, and the digest frontmatter renders only the five counts — not the user-turn component. A session whose single counted "user turn" is an injected or pasted block therefore scores exactly 5.0 with an all-zero `signal_counts` block. The confusion for reviewers is real and stands as sighted: the rendered surface implies `signal_counts` is the score's full basis, and a session with no recorded confusion signals surfaces as significant. The sightings' stronger hypothesis — that scorer and extractor run divergent code paths — is not supported by main as read; the two disagree only in what they *render*. Post-3278, filtered briefing turns are excluded from `n_user_turns` as well, so facet (a)'s scores are within the landed fix's scope; facet (b)'s are not.

This is the 07-24 cluster 1.4 shape ("scores decoupled from bodies") recurring with a new facet: nonzero score with a populated-but-mislabeled body, rather than a high score with an empty one.

### 1.3 Signal-channel fidelity: designed outcomes counted as errors; real signals stripped of context (2 sightings, 2 sessions)

- **a189558e** — `watcher-rearm.sh`'s bounded-wait ceiling is a designed loop-continuation outcome, self-declared on stdout (`WATCHER_REARM_OUTCOME: CEILING exit=124`), yet the digest extraction counted each ceiling hit as a `tool_error`: 3 of the session's 4 tool_error signals are this benign shape, identical in rendering to the one real failure (turn 70's exit 2 — the `DARK_FACTORY_ROOT` contract gap already catalogued as entry-cand-20260722-3). The error signal is inflated 4× and the genuine bug is buried among false positives.
- **4541215f** — an orchestrated-task session's digest shows three `no such file or directory` events (turns 15, 28, 117) with no captured command or path — only the bare trigger-phrase template, repeated. This recurs the 07-24 census's cluster-1.4 rendering facet (matched substring dropped, trigger phrase kept), observed here on a real not-found sequence that is consequently undiagnosable from the digest.

The two sightings fail in opposite directions on the same surface: the first admits designed non-failures into the error channel; the second strips the discriminating content from genuine failures.

### 1.4 Concurrent sessions collide on shared mutable documents (2 findings, 1 session: 7a2fed96)

Both findings are from one interactive PRD session and share a shape — no coordination or versioning protocol between concurrent writers of shared artifacts:

- A sibling planning session read this session's eval-design report before a later addendum (§10, stopping criteria) was appended, and wrote `docs/prds/memory-metadata-vocabulary.md` mandating one of the measures based on the stale version. The divergence surfaced only because the user happened to notice and say so.
- An Edit to auto-memory `MEMORY.md`, built from a previously-read snapshot, was rejected (`File has been modified since read`) because a concurrent writer touched the index first — forcing an unplanned re-read-and-retry with no guidance for reconciling the two edits.

Single-session provenance; the cluster is grouped on shape, not on independent recurrence. Adjacent (not identical) to the codebook's `machine-operated-main-checkout` entry, which covers concurrent actors destroying each other's state in the main checkout.

## 2. One-off sightings (1 finding, verified)

- **Bash persistent cwd broke relative-pathspec `git commit --only` during PRD landing** (c7b3e5de, prd→prd). The Bash tool's working directory persists across commands; an earlier command left cwd inside `shared/`, and a later `git commit --only docs/prds/<file>` — following CLAUDE.md's direct-to-main guidance — resolved the relative pathspec against the stale cwd (`could not open directory 'shared/docs/prds/'` … `pathspec did not match any files`), failing two consecutive attempts to land the memory-eval-dashboard PRD and its capability manifest.

## 3. Cross-cutting observations

These restate patterns visible across the verified sightings; they diagnose nothing beyond what the sightings and the cited code show.

1. **One mechanism, two symptom clusters.** Clusters 1.1 and 1.2 are a single root shape observed from two angles: a non-corrective turn admitted as a gold "user turn" simultaneously fabricates a correction *and* mints a 5.0 score invisible in `signal_counts`. Any consumer triaging digests by either surface inherits both errors together.
2. **The mislabeling family completed its arc from "self-recursion quirk" to "general classifier defect" across three cycles**: 07-22 trickle candidates (self-prompt only) → 07-24 census (one architect-block hint) → this census (4 orchestrated-task + 5 pasted-report sightings, zero self-prompt sightings). The landed fix (3278) covers the injected-turn sources by literal marker; the pasted-report source has no marker and remains open.
3. **Fix latency echo, as designed.** The dominant cluster's fix landed 2026-07-30 — inside this census's mining window. The PRD's 5-day census floor exists exactly because early censuses re-observe pre-fix traces; this batch is likely such a re-observation for facet (a), and the next cycle is the discriminating measurement (§6).
4. **Rendered surfaces that omit their own basis** recur across the instrument: the score omits its dominant component (1.2), error counts omit exit-code/outcome semantics (1.3), not-found signals omit the path (1.3), and captured turns are truncated mid-field (1.1b). In each case the digest asserts a conclusion whose supporting detail was discarded before rendering.
5. **The non-instrument findings all reduce to stale-snapshot writes/reads against shared state** (doc version, memory index, working directory) — three surfaces, two sessions, no shared tooling. Thin evidence for a structural claim; recorded as the observation only.

## 4. Remediation candidates

To be filed via the curator path (plain `submit_task`; curator dedup is the protection — PRD §6.9). Known landed work to dedup against: **task 3278** (done 2026-07-30, discharged 07-24 R1/R2 — do not refile its scope).

| # | Candidate | Cluster | Size |
|---|---|---|---|
| R1 | Digest gold-turn filter: extend beyond literal harness markers to pasted non-corrective report/log content in genuine user turns (e.g. recon run-review dumps) — outside `is_harness_injected_turn`'s scope on current main; also stop truncating captured items mid-field or mark truncation prominently | 1.1(b) | M |
| R2 | Digest frontmatter: itemize the score's `user_turn` component (e.g. render `n_user_turns` or a per-component score breakdown) so `signal_counts` stops implying it is the score's full basis | 1.2 | S |
| R3 | Digest error channel: record exit codes with tool_error signals and honor designed-outcome stdout declarations (e.g. `WATCHER_REARM_OUTCOME: CEILING`) so bounded-wait ceilings are distinguishable from failures; capture the failing path/command for not-found signals, not just the trigger phrase | 1.3 | S |
| R4 | CLAUDE.md direct-to-main guidance: pin `git commit --only` to absolute pathspecs or `git -C /home/leo/src/dark-factory`, neutralizing the persistent-cwd hazard | §2 | S |
| R5 | Shared-document coordination minimum: a stated re-read-before-consume / freshness-stamp convention for in-flight design docs and the live-co-authored MEMORY.md (retry protocol exists as feedback memory; nothing is stated where sibling sessions will see it) | 1.4 | S |

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1(a) briefing-block mislabel | Add sightings to the existing mislabel/self-ingestion entry family (entry-cand-20260722-10/-13/-16/-17/-19/-21 et al.); widen the cause text to source-general; record task 3278 in `filed_tasks`; candidate for `status: partially` now, `fixed` only after a post-3278 cycle shows zero new sightings |
| 1.1(b) pasted-report mislabel | New candidate — distinct source (genuine user turn containing non-corrective report content), verified outside the landed fix's scope |
| 1.2 score/signal_counts decoupling | Add sightings to entry-cand-20260722-20 (score decoupled from evidence body); refine its cause with the verified `user_turn`-weight mechanism |
| 1.3 exit-124 counted as tool_error | New candidate; cross-reference entry-cand-20260722-3 (the real failure it buried) and `watcher-loop-harness-mismatch` |
| 1.3 context-stripped not-found rendering | Add sighting to the 07-24 cluster-1.4-family entry covering trigger-phrase-only signal rendering (merger locates the promoted id; new candidate if none exists) |
| 1.4 concurrent-session shared-doc collisions | New candidate (one entry, two facets); cross-reference `machine-operated-main-checkout` (adjacent, distinct surface) |
| Bash cwd pathspec (§2) | Dated one-off entry per existing convention |

## 6. Method notes for the next census

- **The discriminating check for the landed fix:** confirm whether this census's digests were generated pre- or post-3278 (digest cache provenance vs the 07-30 landing). If post-fix digests still show facet (a), the first thing to check is the filter's all-three-headings co-occurrence requirement (`# Context` + `## Agent Identity` + `# Task`) against the real shape of briefing turns — the sightings' quoted turns show only `# Context`/`## Project Context`. Zero facet-(a) sightings next cycle would be the fix-confirmed signal.
- The 07-24 census's other two headline masses — recon self-report auditability and session-start memory-injection quality — show zero recurrences in this batch. With only 15 verified findings, absence is weak evidence; note it, don't conclude from it.
- Same-shape sightings received divergent phase stamps (ops/ops vs ops/implement; recon/recon vs unknown/unknown). A stamping convention for instrument-defect findings — whose origin is the legibility tooling regardless of the subject session's phase — would make the matrix reflect phenomena rather than stamping variance.
- Merge- and verify-manifested sightings: zero for the second consecutive census. The PRD's motivating hypothesis remains untested; checking sampled-session composition against the class strata would distinguish "quiet phases" from "unreached phases".


## Filed Tasks

Recorded retroactively (2026-08-12, by task 3614). This section read `_none filed._` while two tasks had in fact been filed from these candidates — itself an instance of §3.4's finding that rendered surfaces omit their own basis.

| Candidate | Task | Scope actually taken |
|---|---|---|
| R3, and cluster 1.1 facet **(a)** | **3610** — "Census R6 (07-31 R1-R3 refile + 3278 residual)" | Relax `is_harness_injected_turn`'s all-three-headings co-occurrence requirement (the §6 discriminating check, verified still failing on main); add exit-code / designed-outcome awareness to `tool_error` extraction. Its title says "R1-R3", but its FIX list covers only these two — R1 and R2 were left undischarged. |
| R1, R2 | **3614** — "Legibility digest: discharge 07-31 census R1 and R2" | R1: new `is_pasted_report_turn` predicate excluding pasted recon run-review dumps from gold user turns (facet **(b)**, distinct from 3610's facet (a) — see that task's separate-predicate rationale), plus quantified item-truncation marking (`... [item truncated: N of M bytes dropped]`) and a `truncated_items` frontmatter key. R2: `n_user_turns` frontmatter key making `score` reconstructible from the frontmatter alone. |

**R4** (CLAUDE.md `git commit --only` pathspec hazard) and **R5** (shared-document re-read/freshness convention) remain **unfiled** as of this edit — verified against the task corpus 2026-08-12; the nearest matches (3058, 2745, 3060) address adjacent but distinct surfaces and do not cover either candidate.

## Cost

invoke calls: sonnet miner=220, sonnet verify=1, fable synthesis=1, haiku headroom-probe=1; WARNING: 1 storm batch(es) at indices [6] (>50% coding failures -- degraded dup-rate signal, excluded from the saturation decision)
