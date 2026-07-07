# Output contract — survey report, findings JSON, program doc

Merges the strengths of the two proven runs: dark-factory 2026-07-06 (machine-readable findings + adversarial verdicts + contradiction resolutions) and reify 2026-07-05 (churn exonerations, S/M/L+risk tags, invariant registry, triage briefing).

## Artifact 1 — `bug-hotspot-survey-<date>.md` (the synthesis report)

Required sections, in order:

1. **Method header** (unlabeled preamble, ~6 lines): corpora + sizes (commits mined, window, fix-commit ratio, tasks mined, docs mined), agent-team shape, verification stats in the form `N findings: X confirmed / Y weakened / Z refuted`, which mining lanes ran (and any lost to failures), **as-of main SHA**, and the path to the findings JSON.
2. **`## Ranked hotspots`** — table `# | Hotspot (files + size) | Evidence | Root structural cause`. Evidence is quantitative (fix-commit counts, fix ratios, recency, incident/task ids); root cause is a one-line structural diagnosis. Exemplar row:
   > `| 1 | merge-queue (merge_queue.py 9.4k lines + 12 satellites) | 248 fix commits; still #1 churn (180 changes) in last 3 weeks, *after* the 17-task refactor | Lifecycle & permits tracked by census/flags, not ownership; refactor extracted the wrong seam |`
3. **`## Churn exonerations`** — clusters/files that are hot by raw count but healthy (young TDD feature, mechanical renames), with the reason. Prevents downstream sessions filing "fixes" against healthy churn. Sourced from the reviewers' `churn_exoneration` fields. Empty section allowed, but must be present.
4. **`## Latent bugs (fileable now)`** — numbered list from `kind: latent-bug` findings. Each item: file:line anchor(s), **bolded one-sentence consequence**, historical cross-ref (task/commit ids) if any, and an impact tag. These map ~1:1 to quick-stream tasks.
5. **`## Per-hotspot findings`** — one `### N. <hotspot>` per ranked hotspot, fixed two-part shape:
   - **Diagnosis (confirmed).** Dense prose with file:line, adoption-census counts where relevant ("51 inline literals vs 5 uses of the extracted helper"), task-id history.
   - **Proposals (ranked).** Bullets, each: **named mechanism artifact** (type/chokepoint/table/invariant + where enforced), what existing code it makes **deletable**, size (S/M/L) + risk, finding id(s), back-refs to latent bugs it fixes.
   - Mark any **weakened** finding inline (`*(weakened: <skeptic note gist>)*`) — do not let the verdict distinction vanish at the md layer.
6. **`## Cross-system chains`** — numbered, named chains (e.g. "Merge-landed vs task-done atomicity gap"): member subsystems, the missing guarantee, the compensation inventory it spawned, the fundamental fix, pointer into the ranked priorities.
7. **`## Ranked priorities (payoff × feasibility)`** — the single canonical ranking (5–8 items); each names the mechanism and quantifies the collapse ("collapses ~6 detectors across 4 subsystems; 20+ historical fix tasks"). Other sections point here rather than restating.
8. **`## Contradiction resolutions`** — wherever ≥2 reviewers overlapped or conflicted: the conflict, the resolution, the rationale. Empty section allowed but must be present (its absence is a signal). These become the program doc's "resolved decisions — do not relitigate".

**Evidence hard rules (apply before writing):**
- No finding appears without at least one file:line anchor OR an explicit *hypothesis* marker.
- Every finding cites bug history (task ids / commit SHAs / escalation ids) or states "speculative — prophylactic".
- Every architectural proposal states what existing code it makes deletable.
- Pin the surveyed main SHA in the method header; all anchors are relative to it.

**Ranking rubric:** hotspots by evidence (fix-commit count, fix ratio, recency, live incidents); remedies by payoff × feasibility, where payoff ≈ (historical fix tasks in the class) + (compensations deleted) + (open incident classes closed) and feasibility = S/M/L effort + risk; latent bugs by impact.

## Artifact 2 — `bug-hotspot-survey-<date>-full-findings.json`

The workflow's return value, digested. Do not skip this artifact — it is what makes later premise re-verification tractable (a DF program-time G6 catch traced a false survey premise through exactly this file).

Top level: `{ method, hotspot_table, themes, clusters, refuted, cross_system }`.

- `method`: `{as_of_sha, window, corpora, lanes_run, verification: {confirmed, weakened, refuted, unverified}}`.
- `themes`: the mining output, `{subsystem, theme, evidence, count_estimate}` per entry. (No prose `mining_summaries` blobs.)
- `clusters[*].findings[*]`: assign **stable ids** at digest time — `<cluster-key>.<n>` (e.g. `merge-queue.3`) — so the md report, program doc, and PRDs can cite individual findings. Fields per finding: `{id, title, kind, files, anchors, problem, proposal, bug_history_link, impact, effort, verdict, verdict_notes}`. Weakened/refuted verdicts must carry non-empty `verdict_notes`.
- `refuted`: kept separately with `why` — refuted findings are excluded from clusters but preserved for audit.
- `cross_system`: `{chains, top_priorities, contradictions}` verbatim from synthesis.

## Artifact 3 (successor, not written by the survey) — the remediation program doc

Written by the follow-on deliberation/planning session as `bug-hotspot-remediation-program-<date>.md`. The survey hand-off names it as the expected next artifact. Sections (merged from both proven runs):

1. **`## Streams`** — `ID | Slug | Scope | Mode (agent-driven /prd vs spawned interactive /prd) | Wave | Upstream deps`. Streams sized for one /prd session each; quick latent-bug material becomes agent-mode micro-streams.
2. **`## Seam ownership (G4 — authoritative)`** — `Seam/artifact | Owner stream | Consumers (do NOT redefine)`. Every shared mechanism the survey proposed gets exactly one owner; explicit anti-duplication injunctions.
3. **`## Invariant registry`** — from reify: `INV-id | Invariant | Enforcement mechanism (type/test/lint/doc+test) | Status (proposed→enforced) | Owner stream`. Meta-invariant: every filed task cites its INV-id and lands the enforcement in its done-criteria.
4. **`## Resolved design decisions (do not relitigate)`** — the survey's contradiction resolutions promoted to binding rulings, plus operational rulings from deliberation.
5. **`## Shared conventions (every session)`** — filing mechanics (agent_id tagging, `planning_mode=True` + bulk `commit_planning`, metadata.files, dedup-before-filing via `search_tasks` + confirm with `get_task`), AFK autonomy rule, `git commit --only` for racing sessions, and the **survey-freshness clause**: "re-verify any file:line you build a task on — the survey is pinned to `<SHA>` and main moves".
6. **`## Triage briefing`** — from reify: for future /unblock and escalation-triage sessions. Postures that change triage (e.g. fail-closed rollouts make new loud failures working-as-intended), structural changes in flight ("check before trusting a pre-existing task's premise"), things that look stuck but aren't (dependency-gated release gates), dup-check guidance.
7. **`## FILED — program status`** (appended during execution) — per-stream PRD path + task-id anchors, and a coordinator-interventions log (deviations from the survey, premise corrections).

**Release-gate convention:** deferred batches are gated by an **escalate-on-dispatch milestone task** (deterministic pure gate, `always_escalates=true` where the tracker supports it) — never a bare no-op pending task, which a scheduler will dispatch and falsely complete.
