---
name: hotspot-survey
description: "Multi-agent historical bug-hotspot survey — mines git history, fix-task history, and postmortems for where bugs have clustered, deep-reviews each hotspot cluster for the root structural cause, adversarially verifies every finding against the code, and synthesizes cross-system improvement proposals into a ranked report + machine-readable findings JSON ready to feed /prd. ALWAYS use this skill for: /hotspot-survey commands, 'conduct a systematic survey for bug hotspots', 'which parts of the code have been buggy historically', 'mine git history for where bugs cluster', 'what should we refactor next based on bug history', or refreshing a previous hotspot survey. This is a long-running many-agent survey (~25-30 agents, ~2.5M subagent tokens, ~60-90 min wall clock) — state the cost up front, and if the ask is vague ('are there bugs here?') confirm scope before launching. This is NOT for: current-state correctness review (/review — does it run, is it wired), comprehension of a single target (/study), root-causing one specific problem (/deb), authoring the remediation PRDs (/prd — feed it this skill's report), or a project's deterministic detector sweep (e.g. reify /audit)."
argument-hint: "[scope — subsystems/modules to survey; omit for whole repo]"
---

# Bug Hotspot Survey

A longitudinal survey answering: **where have bugs clustered historically, what structural property of the code produces them, and what systemic change would kill the whole class?** Fix-commit density is the core signal — historical hotspots predict future bugs. The deliverable is a ranked, adversarially-verified report plus a machine-readable findings JSON, designed to feed a human deliberation pass and then `/prd` — never direct task filing.

Proven shape (dark-factory 2026-07-06: 28 agents, 2.54M subagent tokens, 64 min, 75 findings → 16 PRD streams / ~110 tasks; reify 2026-07-05: 26 agents → 8 PRDs / ~90 tasks):

```
Phase 0  (you)              scout churn + fix ratios, pick clusters, seed known context
Phase 1  Mine       3 agents  task-db / git-log / postmortems → recurring fix themes
Phase 2  Review     ~12 agents one deep architectural reviewer per hotspot cluster
Phase 3  Verify     ~12 agents one adversarial skeptic per review (pipelined, no barrier)
Phase 4  Synthesize 1 agent   cross-system defect→patch chains, priorities, contradictions
Phase 5  (you)              digest, write report + JSON, commit, hand off to deliberation
```

## Parse invocation

```
/hotspot-survey                        → whole repo
/hotspot-survey --scope orchestrator   → restrict mining + clusters to a subtree/subsystem
/hotspot-survey --since 2026-01-01     → history window (default: ~6 months, or project epoch if younger)
/hotspot-survey --clusters 8           → target cluster count (default 8–12, evidence-driven)
```

Free-text arguments name the scope ("survey the merge queue and its satellites").

## Step 0 — Load the project overlay (do this first, every invocation)

```bash
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ls "$ROOT/.claude/skills/hotspot-survey/project.md" 2>/dev/null && echo "overlay present" || echo "no overlay — generic mode"
```

If the overlay exists, **Read it** — it supplies the fused-memory `project_id`/`agent_id`, the task-tracker source and its mining recipe, the report output directory, a seed subsystem vocabulary, the project's fix-commit grep vocabulary, any deterministic audit CLI to fold in, and the `/prd` hand-off conventions. Overlay schema: `references/project-overlay.md`. Without an overlay, run generic: derive everything in Phase 0 and put reports under `plans/` (or `docs/notes/` if the repo has no `plans/`).

## Cost gate

Before launching Phases 1–4, tell the user what it costs (agent count, token estimate, wall clock). If the invocation was an unmistakable ask (explicit `/hotspot-survey`, or a prose request that names this exact activity), proceed. If scope is genuinely ambiguous **and a human is present**, confirm scope first. Under AFK autonomy, take the whole-repo default and proceed — never block on a question.

## Phase 0 — Inline scouting (you, the coordinator — no agents)

The survey's quality is set here: the workflow is seeded with hard numbers and a hand-authored cluster list, not agent guesses.

1. **Churn + fix-density mining** (adapt paths/grep vocabulary from the overlay):

```bash
# Overall churn since window start
git log --since="<SINCE>" --pretty=format: --name-only | grep -v '^$' | sort | uniq -c | sort -rn | head -60
# Fix-flavored churn (exclude docs/plans)
git log --since="<SINCE>" -i --grep='fix' --grep='bug' --grep='amend' --grep='regression' \
  --pretty=format: --name-only | grep -v '^$' | grep -v '^plans/' | grep -v '^docs/' | sort | uniq -c | sort -rn | head -50
# Total vs fix-flavored commit counts → repo-wide fix ratio
git rev-list --count --since="<SINCE>" HEAD
git rev-list --count --since="<SINCE>" -i --grep='fix' --grep='bug' --grep='amend' --grep='regression' HEAD
# Recent churn (last ~3 weeks, source only) → recency weighting
git log --since="<3 weeks ago>" --pretty=format: --name-only -- '*/src' | grep -v '^$' | sort | uniq -c | sort -rn | head -40
# File sizes of the emerging suspects
wc -l <suspect files> | sort -rn
```

2. **Per-file fix ratio** for the top-churn files (fix-commits touching file ÷ total commits touching file) — this separates "hot because bugs" from "hot because active feature work".

3. **Probe the task tracker** (source per overlay; e.g. `.taskmaster/tasks/tasks.json` structure, count, field names) so the mine:tasks prompt can state the exact shape.

4. **Search memory for known bug classes**: `search(query="recurring bugs, incidents, fix batches", project_id=<overlay>)` plus your own session memory. This becomes per-cluster "known context" — *leads to verify, not gospel*.

5. **Hand-author the cluster list**: 8–12 entries `{key, model, files (with line counts + churn stats), context}` where `context` is a short paragraph of known bug classes ending in pointed "Ask:" questions, and `model` downgrades peripheral clusters to a cheaper tier. Define the **fixed subsystem vocabulary** (cluster keys + `other`) that every phase will share — this shared enum is what makes mining → review → synthesis joinable.

## Phases 1–4 — the workflow

Use the **Workflow tool** (background), adapting the template in `references/orchestration.md` — it contains the full proven script, the four structured-output schemas, and every prompt template. Fallback if the Workflow tool is unavailable: plain parallel Agent fan-out with the same phases (loses the journal/resume and pipelining; reify's run proves it works).

| Stage | Agents | Model | Effort | Notes |
|---|---|---|---|---|
| Mine | 3 | cheap (sonnet) | medium | task-db / git-log / plans+postmortems, in parallel |
| Review | 8–12 | default; cheap for peripheral clusters | high | one per cluster, seeded with mined themes + known context |
| Verify | 1 per review | cheap (sonnet) | medium | pipelined off each review — no barrier |
| Synthesize | 1 | default | high | gets all themes + verified findings, refuted stripped |

Hard rules (each one bought with a real failure — see `references/orchestration.md` §Failure modes):

- **Semantically validate miner outputs** after the mining barrier: minimum theme count, placeholder/degenerate detection (`test`, empty evidence). Schema validation alone is not enough — a miner under schema-rejection pressure once emitted `"test theme"` junk that passed. Re-run a failed miner once; if it fails again, proceed without that lane and say so in the report method line.
- **Pipeline each review straight into its skeptic** (`pipeline()`, not a barrier) — verification of cluster A runs while cluster B is still reviewing.
- **Digest before synthesis**: strip refuted findings, trim fields — keeps the synthesis prompt bounded.
- **Count confirmed/weakened/refuted separately** — "N survived" conflates confirmed with weakened.

## Phase 5 — Digest and report (you, the coordinator)

The workflow result is large (~270KB in the DF run) and arrives truncated in the notification. **Never ingest it raw**: parse the full output file with a small Python digest (index of all findings; full detail only for `impact == high`; chains + priorities), writing intermediates to the scratchpad.

Then produce the two artifacts (format contract: `references/report-format.md`):

- `<output_dir>/bug-hotspot-survey-<date>.md` — the synthesis report. Must open with a **method header** (corpora + sizes, verification stats `X confirmed / Y weakened / Z refuted`, **as-of main SHA**, path to the JSON) and include: ranked-hotspots table, churn exonerations, latent bugs (fileable now, with impact tags), per-hotspot diagnosis + ranked proposals (S/M/L + risk), cross-system chains, one canonical ranked-priorities list, contradiction resolutions.
- `<output_dir>/bug-hotspot-survey-<date>-full-findings.json` — the full structured findings with stable per-finding ids (`<cluster>.<n>`). This is what makes later premise re-verification tractable — do not skip it.

Commit both with `git commit --only <paths>` (a live merge queue races broad commits). Display an in-chat summary: ranked table, latent bugs, top priorities.

## Hand-off — deliberation, then /prd (never auto-file)

The survey ends at the report. Do **not** file tasks or auto-run `/prd`. The proven pipeline is:

1. **Deliberation** — walk the user through the judgement calls: each contested issue, the options, trade-offs and long-term consequences. Invariant candidates get explicit treatment ("look for principled invariants we can make explicit and enforce uniformly").
2. **Remediation program doc** — a coordination contract for the `/prd` sessions: streams table, seam-ownership (G4) table, resolved decisions, shared filing conventions, a survey-freshness clause ("re-verify any file:line you build a task on"), and an invariant registry (INV-ids with enforcement mechanism + status). See `references/report-format.md` §Program doc.
3. **`/prd` per stream** — mechanical streams via an agent team, design-heavy streams via spawned interactive sessions.

Close the survey turn by proposing step 1, with the report path as the anchor.

## Graceful degradation

| Missing | Impact | Behaviour |
|---|---|---|
| fused-memory / task tracker | no fix-task mining lane | run 2 miners (git + postmortems), note in method header |
| Workflow tool | no journal/resume, no pipelining | plain Agent fan-out with the same phases and schemas-as-prose |
| plans//postmortems | no third mining lane | run 2 miners, note it |
| Project overlay | no project specifics | generic mode: derive in Phase 0, elicit output dir if unclear |
| A miner fails semantic validation twice | one evidence lane lost | proceed, state the lost lane in the method header |

Never fail silently — the method header records exactly which lanes ran.

## Writing to memory

At the end, write a survey summary to fused-memory:

```
add_memory(
  content="Bug-hotspot survey <date>: <N> agents, mined <corpora>; <K> clusters reviewed; <X> confirmed / <Y> weakened / <Z> refuted findings; top hotspots: <list with fix ratios>. Report: <md path>, full findings: <json path>. As-of <SHA>.",
  category="observations_and_summaries",
  project_id=<overlay>, agent_id="claude-interactive"
)
```

Record contested design decisions that got resolved in deliberation as separate `decisions_and_rationale` memories.

## Anti-triggers

- Current-state correctness ("does it actually work?") → `/review`.
- Understanding one module deeply → `/study`.
- One specific bug/incident → `/deb` (root-cause investigation of a single problem), not a survey.
- Filing the remediation work → `/prd` (consuming this skill's report).
- Deterministic invariant detectors (phantom-done, orphan symbols) → the project's `/audit` if it has one; this skill *folds in* such results, it doesn't replace them.

## Reference

- `references/orchestration.md` — the workflow template: schemas, prompt templates, model allocation, failure modes.
- `references/report-format.md` — the report + JSON + program-doc output contract.
- `references/project-overlay.md` — overlay schema for specializing this skill per project.
- `references/exemplar-run-df-2026-07-06.js` — the verbatim workflow script from the proven dark-factory run.
