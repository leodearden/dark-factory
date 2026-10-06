---
name: hotspot-survey
description: "Multi-agent historical bug-hotspot survey — mines git history, fix-task history, and postmortems for where bugs have clustered, deep-reviews each hotspot cluster for the root structural cause against docs/code-quality.md, adversarially verifies every finding against the code, and synthesizes cross-system improvement proposals into a ranked report + keyed findings JSON (docs/quality-findings-contract.md) that feeds deliberation and then /prd. Runs in REFRESH mode by default when a previous survey report exists (mines only the commits since its as_of_sha, re-verifies its open findings, re-reviews only materially changed clusters); --full re-derives everything. ALWAYS use this skill for: /hotspot-survey commands, 'conduct a systematic survey for bug hotspots', 'which parts of the code have been buggy historically', 'mine git history for where bugs cluster', 'what should we refactor next based on bug history', refreshing a previous hotspot survey, or a /review-all run asking for a hotspot refresh. Long-running and many-agent (full: ~28 agents, ~2.5M subagent tokens, 60-90 min; refresh: sized from the prior report in the cost gate) — state the cost up front, and if the ask is vague ('are there bugs here?') confirm scope before launching. NOT for: current-state correctness review (/review), comprehension of a single target (/study), root-causing one specific problem (/deb), authoring the remediation PRDs (/prd — feed it this skill's report), or a project's deterministic detector sweep (e.g. reify /audit)."
argument-hint: "[--refresh | --full] [scope — subsystems/modules to survey; omit for whole repo]"
---

# Bug Hotspot Survey

A longitudinal survey answering: **where have bugs clustered historically, what structural property of the code produces them, and what systemic change would remove the whole class?** Fix history chooses where to look; `docs/code-quality.md` is what the code is judged against. The deliverable is a ranked, adversarially verified report plus a findings JSON, both conforming to `docs/quality-findings-contract.md` (cited as "contract §n"; read it before a run, it is not restated here). Mechanical findings it files itself as curator tickets (contract §6); structural findings feed a human deliberation pass and then `/prd`, and the survey files none of them.

Proven shape (dark-factory 2026-07-06: 28 agents, 2.54M subagent tokens, 64 min, 75 findings → 16 PRD streams / ~110 tasks; reify 2026-07-05: 26 agents → 8 PRDs / ~90 tasks). Neither run conforms to the current contract; the DF run is the refresh baseline (§Refresh mode).

```
Phase 0  (you)                 pin as_of_sha, read the prior report + other instruments' reports,
                               fix-signal scouting, pick clusters, select which ones to review
Phase 1  Mine + Re-verify      3 miners (task store / git log / postmortems) ∥ one re-verifier per ≤8
                               prior findings (refresh, or full with a prior report)
Phase 2  Review   ≤12 agents   one deep reviewer per selected cluster
Phase 3  Verify   1 per review skeptic per review, pipelined, blind to proposals
Phase 4  Synthesize  1 agent   chains, ranked remedies, contradictions over deltas + still-open
Phase 5  (you)                 digest, mint keys, file mechanical tickets, write report + JSON,
                               conformance check, commit
Hand-off (you, with the human) deliberation → program doc → /prd; file the trigger chain
```

## Parse invocation

```
/hotspot-survey                        → refresh if a prior report exists, else full; whole repo
/hotspot-survey --full                 → re-mine from the overlay's epoch and review every cluster
/hotspot-survey --refresh              → refresh; refuses (says why) when no prior report exists
/hotspot-survey --scope orchestrator   → restrict mining + clusters to a subtree/subsystem
/hotspot-survey --since <commit|date>  → full-mode window start (default: overlay epoch, else ~6 months)
/hotspot-survey --clusters 8           → target cluster count (default 8–12, evidence-driven)
/hotspot-survey --as-of <sha>          → pin this commit instead of main's tip
/hotspot-survey --launched-from <esc>  → the escalation that launched this run (recorded as extra.launched_from)
/hotspot-survey --findings-only        → write and commit the report; file nothing, no trigger chain
```

Free-text arguments name the scope ("survey the merge queue and its satellites"). `/review-all` invokes `--refresh --findings-only --as-of <its pinned sha>` (contract §9): both runs read one tree, and the outer run owns routing and its own chain. A run started from a `review_due` escalation gets `--launched-from <esc id>` (`skills/_shared/filing-the-trigger-chain.md` §5 step 2). Every run is attended (contract §11).

## Step 0 — Load the project overlay (every invocation)

```bash
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ls "$ROOT/.claude/skills/hotspot-survey/project.md" 2>/dev/null && echo "overlay present" || echo "no overlay — generic mode"
```

If the overlay exists, **Read it**: it supplies the fused-memory identity, the task-store source and its schema probe, the output directory, the cluster seeds with their areas, fix-commit vocabulary additions, the refresh baseline for a pre-contract report, any metrics snapshot, and hand-off conventions. Schema: `references/project-overlay.md`. Without one, run generic: derive everything in Phase 0, reports under `plans/` (or `docs/notes/` if the repo has no `plans/`).

## What each measurement is for

`docs/code-quality.md` lists the fix and amend rate as lagging ("watch but do not steer") and forbids steering by raw line count (contract §10). This survey uses each measurement for exactly one job:

| Measurement | Used for | Never used for |
|---|---|---|
| Fix density (chosen ratio, §Phase 0 step 2) | choosing clusters and recency weighting: where to look | ranking remedies, setting severity, judging success |
| Raw ratio (the old full-message grep) | printed beside the chosen ratio so drift is visible | anything else |
| `amend:` count | review-friction context in mining evidence | fix density |
| Line counts (`wc -l`, diff `--shortstat`) | reviewer reading strategy; refresh material-change selection | ranking, a finding on its own, severity |
| Findings by severity | remedy payoff; the report's priorities | — |
| Open-finding count per area at re-verification, and the complexity pair (max per function falls, module total flat or falls) where a metrics snapshot exists | the **success measure** between runs | — |

## Cost gate

Before launching Phases 1–4, tell the user the cost of the mode you are about to run, and in refresh mode quote the full-mode cost beside it:

- **Full**: ~28 agents, ~2.5M subagent tokens, 60–90 min, plus re-verification when a prior report exists.
- **Refresh**: agents ≈ 3 + ⌈P/8⌉ + 2R + 1 and tokens ≈ 0.3M + ⌈P/8⌉ × 75k + R × 195k + 140k, where P = prior findings to re-verify and R = clusters selected for review. The 75k re-verifier figure is a skeptic's observed cost, not yet measured for re-verification.

A refresh is cheap when few clusters changed; when most did, it costs about as much as a full run and buys key continuity rather than tokens. Say which. If the invocation was an unmistakable ask, proceed. If scope is genuinely ambiguous, confirm scope first. The survey is human-attended: with no human present, stop and say so rather than launching. The `## Method` block's `cost` records what the run actually spent.

## Phase 0 — Inline scouting (you, the coordinator — no agents)

The survey's quality is set here: the workflow is seeded with hard numbers and a hand-authored cluster list, not agent guesses.

1. **Pin the run.** `AS_OF=$(git rev-parse main)`, or the `--as-of` commit; `run_id` per contract §5 (`hotspot-survey-<project_id>-<YYYYMMDD>[-<n>]`). Every agent reads this one tree.

2. **Fix signal.** A *fix commit* is a non-merge commit whose **subject** starts with a conventional `fix`, `bugfix` or `hotfix` prefix (optional scope, optional `!`). `amend:` does **not** count: it marks a review amendment made on a task branch before merge, not a fix of landed code (in dark-factory it has meant that since before the first survey). The overlay may add a marker (e.g. commits that broke main) and fix-origin task-store values. Report the raw full-message ratio beside the chosen one (orchestration.md §Failure modes 7 shows why).

```bash
FIX='^(fix|bugfix|hotfix)(\([^)]*\))?!?:'
RANGE="<SINCE>..$AS_OF"              # refresh: prior as_of_sha; full: the window-start commit
SRC=(-- . ':!plans' ':!docs' ':!*.md' <overlay exclusions, e.g. the task-store dir>)
git log --no-merges --format=%s $RANGE "${SRC[@]}" | wc -l                 # chosen denominator
git log --no-merges --format=%s $RANGE "${SRC[@]}" | grep -cE "$FIX"      # chosen numerator
git log --no-merges --format=%s $RANGE "${SRC[@]}" | grep -cE '^amend(\([^)]*\))?!?:'
git rev-list --count $RANGE; git rev-list --count -i --grep=fix --grep=bug --grep=amend --grep=regression $RANGE  # raw
# fix-flavored churn per file (repeat with --since='3 weeks ago' for recency weighting)
git log --no-merges --format='%x00%s' --name-only $RANGE "${SRC[@]}" \
  | FIX="$FIX" awk -v RS='\0' 'BEGIN { fix = ENVIRON["FIX"] } NR>1 { n=split($0, l, "\n"); if (l[1] ~ fix) for (i=2;i<=n;i++) if (l[i]!="") c[l[i]]++ } END { for (f in c) print c[f], f }' \
  | sort -rn | head -50
```

   Then the **per-file fix ratio** for the top-churn files (fix commits touching the file ÷ non-merge commits touching it): it separates "hot because bugs" from "hot because active feature work". Where the overlay names fix-origin task metadata, add the count of such tasks whose file-level metadata intersects each cluster.

3. **Probe the task store** with the overlay's schema probe (columns, value types, row count) so the mine:tasks prompt states the exact shape. Probe it; never copy a shape from an older doc.

4. **Read what other instruments already know** (contract §9). The latest `/review` report (`review/reports/`), the latest `/review-all` report and census report (output dir), and the confusion codebook (`docs/legibility/confusion-codebook.yaml`, entries not `retired`/`fixed`/`done`, matched to clusters by their file references). Open findings in a cluster's area go into its `context` and, when they carry a key, into `PRIOR` for re-verification. Record each report's `run_id` (or its path, for one that predates the contract) in `inputs_consumed`.

5. **Search memory for known bug classes**: `search(query="recurring bugs, incidents, fix batches", project_id=<overlay>)` plus your own session memory, as per-cluster leads to verify.

6. **Hand-author the cluster list**: 8–12 entries `{key, area, model, review, files, context}` (orchestration.md §Meta). `area` is the `review/briefing.yaml` subproject owning the cluster's core files, or `repo` (report-format.md §Area). `context` ends in pointed "Ask:" questions. The cluster keys plus `other` are the **fixed subsystem vocabulary** every phase shares.

## Refresh mode

The default when the output directory holds a prior `*-full-findings.json`.

1. **Baseline.** Read the latest prior artefact's `method.run_id` and `method.as_of_sha`; that SHA is `since`. A pre-contract artefact has neither: use the overlay's baseline pointer and seed `PRIOR` with
   `python3 <skill>/scripts/findings_artefact.py legacy --input <json> --run-id <prior run_id> --clusters <overlay key=area list> --out <scratch>/seed.json`,
   which maps each finding onto contract fields with a provisional file-grain anchor, a `ref` of `positional:clusters[i].findings[j]`, and the same ref under `supersedes`.
2. **Prior findings to re-verify.** Every prior finding whose disposition is `open`, `filed:` or `accepted:` (all of a pre-contract baseline's), plus other instruments' keyed open findings in the survey's areas. Refresh dispositions first with contract §8 step 1 (key lookup): a live task means `filed:`, a `cancelled` task carrying `metadata.x_acceptance_reason` means `accepted:` (a `cancelled` task without it is treated as absent, contract §8 step 1b), and a `done` task whose finding re-verifies as present is flagged for the hand-off to file with `x_supersedes_task` (contract §8 step 1c). A store that cannot be reached leaves dispositions as the prior report had them, and the report says so.
3. **Re-verification** runs in Phase 1 beside the miners, one re-verifier per ≤8 prior findings of a cluster, blind to `proposal`, at `as_of_sha`, with outcomes `confirmed` / `weakened` / `refuted` / `fixed` and the current `path::symbol` anchor (orchestration.md §Re-verification prompt).
4. **Which clusters get a fresh reviewer read.** Prior clusters keep their `files` (renames resolved with `git diff -M --name-status since as_of`). The trigger is change volume, and it decides only which clusters are read again: it never ranks, grades or measures anything, since `docs/code-quality.md` §Do not steer by rules out raw line count as a measure of quality. Preferred trigger, when metrics snapshots exist for `since` and `as_of`: the cluster's complexity pair moved (its per-function maximum or its module total changed). Otherwise: `git diff --shortstat since as_of -- <its files>` changes (insertions + deletions) at least **20%** of its line count at `since`. Either way, also review a cluster when a carried finding's anchor file was renamed or deleted, and a new cluster: a file whose window fix count would have ranked it into the cluster list but which no prior cluster holds. Reviewers are seeded with the cluster's surviving prior findings ("verify you agree, do not re-report, then look for NEW structure").
5. **Synthesis** sees this run's verified findings plus the re-verified still-open ones.

| Item | Refresh | Full |
|---|---|---|
| `as_of_sha`, `run_id` | new | new |
| Mining window | `since..as_of` | overlay epoch (or `--since`)`..as_of` |
| Cluster list | prior clusters (renames resolved) + new hot files | re-derived from scratch |
| Prior findings | re-verified; keys carried forward | re-verified; keys carried forward |
| Reviews | material-change and new clusters only | every cluster |
| Unreviewed clusters' `architecture_notes` / `churn_exoneration` | copied from the prior artefact, marked with its `run_id` | recomputed |
| Skeptics | this run's new findings | this run's new findings |
| Themes | window only; prior themes not carried | whole window |
| Synthesis input | new verified + still-open carried | everything verified |

## Phases 1–4 — the workflow

Use the **Workflow tool** (background), adapting `references/orchestration.md`: the full script, the schemas and every prompt template. Fallback if the Workflow tool is unavailable: plain parallel Agent fan-out with the same phases (loses journal/resume and pipelining).

| Stage | Agents | Model | Effort | Notes |
|---|---|---|---|---|
| Mine | 3 | cheap (sonnet) | medium | task store / git log / plans+postmortems, over the run's window |
| Re-verify | ⌈P/8⌉ | cheap (sonnet) | medium | in the same parallel stage as the miners; blind to `proposal` |
| Review | selected clusters | default; cheap for peripheral | high | Reads `docs/code-quality.md`; tags every finding |
| Verify | 1 per review | cheap (sonnet) | medium | pipelined off each review; blind to `proposal` |
| Synthesize | 1 | default | high | Reads `docs/code-quality.md`; refuted stripped |

Every judging prompt carries the quality definition and tells its agent to tag each finding with the heuristic it rests on (contract §10): `Read docs/code-quality.md` when the project carries it, else the block `orchestrator/src/orchestrator/agents/code_quality.py::guidance` renders, embedded verbatim (orchestration.md §Meta, `QUALITY_DEF`). The doc is never hand-restated. Hard rules, each bought with a real failure (orchestration.md §Failure modes):

- **Semantically validate miner outputs** after Phase 1: minimum theme count, placeholder detection. Re-run a failed miner once; if it fails again, proceed without that lane and say so in the method header.
- **Pipeline each review straight into its skeptic** (`pipeline()`, not a barrier).
- **Skeptics and re-verifiers see the claim, never the proposal.**
- **Digest before synthesis**: strip refuted findings, trim fields.
- **Count confirmed / weakened / refuted / unverified separately.**

## Phase 5 — Digest and report (you, the coordinator)

The workflow result is large (~270KB in the DF run) and arrives truncated. **Never ingest it raw**: parse the output file with a Python digest in the scratchpad that imports `finding_key` from `<skill>/scripts/findings_artefact.py` and:

- maps each new finding onto contract §1 fields: anchor and primary tag from the skeptic's `anchor`/`primary_tag` (the kind's fixed primary tag wins where report-format.md fixes one), `severity`, `evidence_source` and `route` per report-format.md, `disposition: open` (or `refuted`), `first_seen = run_id`, `last_seen = [run_id]`, `sub_area` = the cluster key, key per contract §2 over the plain `area`;
- applies re-verification outcomes to `PRIOR`: `confirmed`/`weakened` keep the prior key and `first_seen`, append `run_id` to `last_seen`, and get a new key with the old one under `supersedes` only when the anchor moved (a pre-contract seed always mints from its verified symbol anchor, keeping its positional ref under `supersedes`); `fixed` → `disposition: fixed:<sha>`; `refuted` → `refuted`; reviewer `prior_changes` update severity or verdict and are listed in §Carried forward;
- matches new findings against carried and foreign keys: an equal key is a re-observation (contract §9), never a second finding;
- resolves synthesis members (refs and display ids) to keys, and prints an index of all findings with full detail only for `severity == high`.

**File the mechanical findings** (contract §6; skipped under `--findings-only`): each `confirmed` finding with `route: mechanical` (a `latent-bug` with one anchor and no design choice) goes through contract §8's dedup protocol in order, and only a finding no step claims is filed as a curator ticket:

```
submit_task(project_root=<root>, agent_id=<overlay>, priority=<severity>,
  title="<one-line consequence> (<anchor>)", description=<statement>,
  metadata={"x_finding_key": "<key>", "x_finding_run": "<run_id>", "source": "hotspot-survey",
            "files": ["<anchor path>"]})
resolve_ticket(...)   # → the task id, or the curator's drop/combine target
```

Record `filed:<id>` (or the `filed:`/`accepted:` a dedup step found) on the finding, and the deciding step in the report. A store that cannot be reached for step 1 means nothing is filed and the report says so; the findings stay `open`. Weakened findings and every `structural` finding go to deliberation.

Then write the two artefacts (shape: `references/report-format.md`) — `<output_dir>/bug-hotspot-survey-<date>[-<n>].md` and `<output_dir>/bug-hotspot-survey-<date>[-<n>]-full-findings.json`; a rerun never overwrites an earlier report (contract §5) — and run the conformance check:

```bash
python3 <skill>/scripts/findings_artefact.py check --findings <json> --report <md> --repo "$ROOT"
```

Do not commit until it exits 0. Commit both with `git commit --only <paths>` (a live merge queue races broad commits). Display an in-chat summary: ranked table, latent bugs with their filed ids, carried-forward outcomes, top priorities.

## Hand-off — deliberation, program doc, /prd, trigger chain

The survey files nothing on the structural route (contract §6); its mechanical tickets were filed in Phase 5. The pipeline:

1. **Deliberation** — walk the user through the judgement calls: each contested issue, the options, trade-offs and long-term consequences. Invariant candidates get explicit treatment: map each to an existing `docs/legibility/design-invariants.md` id or propose adding one there.
2. **Remediation program doc** — per report-format.md §Artifact 3: streams carrying finding keys, seam ownership, invariants (pointing at `docs/legibility/design-invariants.md`, never minting ids), resolved decisions, shared conventions (filed tasks carry `x_finding_key`/`x_finding_run`, contract §8; findings cited by key), and the FILED table with keys per task.
3. **`/prd` per stream** — agent-mode streams via an agent team, design-heavy streams via spawned interactive sessions. Each runs contract §8's dedup protocol and stamps the keys.
4. **Trigger chain** (contract §11; none under `--findings-only`). Follow `skills/_shared/filing-the-trigger-chain.md` §§0–5 with `skill="hotspot-survey"` (bare name, no slash) and `run_id=<this survey's run_id>`; the calls live only there. What is specific to this survey:
   - The run's filed tasks are Phase 5's mechanical tickets plus whatever the `/prd` sessions file later. File the chain at the end of deliberation under `planning_mode` and leave it `deferred`, in either form (snippet §3's program paragraph, §4); the program doc's shared conventions name the gate id so each `/prd` session adds its own dependencies in the interim form. Commit it only once the FILED table is complete and the run has filed tasks: `scripts/check_run_completion.py` exits 2 (an error) for a run with none.
   - Pick the form by probing the dark-factory checkout, the factory root, never the surveyed project (snippet §2); for reify the probe runs in dark-factory.
   - First supersede any open chain whose `trigger_chain.skill` is `hotspot-survey` (snippet §1, §5). The old human gate's escalation resolves as `ran as <run_id>` when it is the `--launched-from` escalation (snippet §5 step 2), recorded as `extra.launched_from`.
   - Chain tasks carry `metadata.trigger_chain`, never `x_finding_run`. A run that files no tasks files no chain and says so under `extra` (snippet §0).

   Verify with `get_task`, never `search_tasks` (its corpus excludes `deferred`).

Close the survey turn by proposing step 1, with the report path as the anchor.

## Graceful degradation

| Missing | Impact | Behaviour |
|---|---|---|
| Task store / fused-memory | no fix-task lane; dispositions not refreshed; no mechanical tickets; no trigger chain | run 2 miners; keep prior dispositions; leave mechanical findings `open`; say all four in the report |
| Workflow tool | no journal/resume, no pipelining | plain Agent fan-out with the same phases and schemas-as-prose |
| plans/postmortems | no third mining lane | run 2 miners, note it |
| Project overlay | no project specifics | generic mode: derive in Phase 0, elicit output dir if unclear |
| A miner fails semantic validation twice | one evidence lane lost | proceed, state the lost lane in the method header |
| Prior report predates the contract and no overlay baseline | no `since` | run `--full`, still seeding `PRIOR` from the legacy JSON |
| Verification did not run | verdicts `unverified` | set `extra.verification_skipped_reason`; the check enforces it |

Never fail silently: the method header records exactly which lanes ran.

## Writing to memory

At the end, write a survey summary to fused-memory:

```
add_memory(
  content="Bug-hotspot survey <run_id> (<mode>, as_of <sha>, since <sha|none>): <N> agents, <tokens>; mined <corpora>; <K> clusters reviewed of <C>; <X> confirmed / <Y> weakened / <Z> refuted / <U> unverified new findings; carried <c> still open, <f> fixed; fix signal <chosen> (raw <raw>). Report: <md path>, findings: <json path>.",
  category="observations_and_summaries",
  project_id=<overlay>, agent_id="claude-interactive",
  entities=[]   # a task or run this summary is about, e.g. [{'kind': 'task', 'id': <the chain's human gate id>}]
)
```

Record contested design decisions resolved in deliberation as separate `decisions_and_rationale` memories, each with its `entities`.

## Anti-triggers

- Current-state correctness ("does it actually work?") → `/review`.
- Understanding one module deeply → `/study`.
- One specific bug/incident → `/deb`.
- Filing the remediation work → `/prd` (consuming this skill's report).
- Deterministic invariant detectors (phantom-done, orphan symbols) → the project's `/audit` if it has one; this skill *folds in* such results.

## Reference

- `docs/quality-findings-contract.md` — finding fields, keys, severity, report pins, dispositions, dedup, consumption, scheduling.
- `docs/code-quality.md` — what every judging prompt judges against.
- `references/orchestration.md` — the workflow template: schemas, prompts, model allocation, failure modes.
- `references/report-format.md` — survey vocabulary → contract fields, the report + JSON + program-doc shape, the conformance assertions.
- `references/project-overlay.md` — overlay schema.
- `scripts/findings_artefact.py` — key minting, the conformance check, legacy seeding.
- `references/exemplar-run-df-2026-07-06.js` — the verbatim script of the proven DF run (pre-contract).
