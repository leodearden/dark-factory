---
name: review-all
description: "Deep, infrequent, human-attended review of a WHOLE project's architectural and code quality against docs/code-quality.md — pins one tree, takes a whole-repo metrics snapshot (a report, never a gate), refreshes the stale instruments in delta mode (/hotspot-survey --refresh, /review --since) and consumes the census codebook, then runs a heuristic-driven architecture pass: one seat per area judging all fourteen heuristics and both stances, cross-area seats for SPOT / layering / orthogonal dimensions / stateless interactions, a skeptic per seat blinded to proposals, a synthesis joined by finding key with every prior instrument's open findings, and a critic of that synthesis; ends in deliberation with the human, a program doc and /prd per stream, and files the completion-driven trigger for the next run. ALWAYS use this skill for: '/review-all', 'review the whole project against the quality heuristics', 'whole-repo architecture and code-quality review', 'what should we refactor across the whole codebase', 'the quarterly factory self-review', or when the human gate task 'Run /review-all on <project>' dispatches. This is a serious commitment — the first dark-factory run measured 22.9M subagent tokens (8.6M of them in the Phase 2 instrument refreshes), 8 h of machine time for phases 1–4 and one full deliberation sitting plus the /prd work — so state the cost, Phase 2 included, and get a go before phase 2. NOT for: does-it-run correctness of landed work (/review), bug-history hotspots alone (/hotspot-survey), confusion sightings (/census), one module (/study), one bug (/deb), authoring the remediation PRDs (/prd — fed by this skill's program doc), or anything unattended."
argument-hint: "[--plan | --yes] [--areas a,b] [--fable] [--no-file] [--launched-from <esc-id>] [--resume <run_id>] [project-root — omit for the current project]"
---

# /review-all — whole-project quality review

Answers one question about a whole project: **where does the next agent change cost the most, under `docs/code-quality.md`, and what would remove that cost?** Every other quality instrument answers a narrower question from narrower evidence; this one reads them all first (`docs/quality-findings-contract.md` §9), runs only what is stale, adds the one pass nothing else does — judging every area against the fourteen heuristics and both stances — and ends in human deliberation. It supersedes the "quarterly factory self-review" priced in the agent-capacity study. Cadence is task completion, not the calendar (contract §11).

```
Phase 0  (you)                 pin as_of_sha; areas; instrument ages + open findings; cost gate
Phase 1  (you, scripts)        whole-repo metrics snapshot — report, never ranking
Phase 2  (sibling skills)      /hotspot-survey --refresh, /review --since   (census consumed, never run)
Phase 3  Workflow  ~20+4 seats area seats → blinded skeptics → cross-area seats → skeptics
Phase 4  Workflow  2 seats     synthesis → critic (one revision)
Phase 5  (you)                 digest, mint keys, report + findings JSON, file mechanical findings
Phase 6  (you + human)         deliberation → program doc → /prd per stream → trigger chain
```

## Parse invocation

```
/review-all                      → whole project, all phases, human present
/review-all --plan               → phases 0–1 and the roster + cost only; no seats launched
/review-all --yes                → accept the recommended roster at the cost gate and the recommended option at every deliberation item without waiting
/review-all --areas orchestrator,shared   → restrict phase 3 to these briefing areas (cross lenses still run)
/review-all --fable              → pre-authorise the synthesis and critic seats on fable
/review-all --no-file            → report only: no tickets, no trigger chain
/review-all --launched-from <esc-id>   → the review_due escalation that spawned this run (the normal launch path); omit or `none` for a hand launch
/review-all --resume <run_id>    → resume the workflow from its journal (same args file); it re-runs every call after the first divergence in journal order, so a changed or unrun tail is re-run as a standalone workflow over the completed results instead (references/orchestration.md failure mode 16)
```

A trailing path is the project root; otherwise `git rev-parse --show-toplevel`.

`--plan` and `--yes` are mutually exclusive (one stops at the gate, the other walks through it); given both, stop and say so. `--yes` changes no roster and no item shape: the cost gate is still posted in full and the run proceeds at once; in Phase 6 every item is still put in `references/deliberation.md`'s fixed shape with its recommendation, and the recommended option is recorded as the ruling with `ruled_by: lead (--yes)` in the program doc §Resolved decisions and in its `decisions_and_rationale` memory. An item whose analysis does not converge (no recommendation) still stops for the human; `--yes` never auto-accepts an "accept as standing" disposition and never commits the trigger-chain gate (`commit_planning` stays the human's act).

## Step 0 — overlay and prerequisites

```bash
ROOT="$(git rev-parse --show-toplevel)"
ls "$ROOT/.claude/skills/review-all/project.md" 2>/dev/null && echo "overlay present" || echo "no overlay — generic mode"
ls "$ROOT/review/briefing.yaml" "$ROOT/docs/code-quality.md" "$ROOT/docs/quality-findings-contract.md"
```

Read the overlay if present (`references/project-overlay.md`); it supplies identity, path→area rules, the per-language metrics instruments, slice seeds, report homes and hand-off conventions. Read `docs/code-quality.md` in full yourself before anything else: its Definition is the filter every seat applies, and its `Do not steer by` list binds you as the lead.

**No briefing → stop.** Area keys are the briefing's `subprojects` keys plus `repo` and there is no second list (contract §3); a run without them mints no joinable keys. Route to `/review-briefing` and return.

## Cost gate

This is a human-attended skill; AFK autonomy does not apply. After Phase 1, post the roster (seat, model, effort, read budget, expected tokens) with the total token and wall-clock estimate and the human-hours Phase 6 implies, then wait for a go (`--yes`: post it and proceed). A first run on a mature repo is a large commitment by design — say so plainly rather than trimming the roster to look cheap. `--plan` stops here.

The gate prices **both** halves of the run; the phase 3–4 roster alone understates it by more than a third:

- **Phase 2**, from each sibling skill's own gate: `/hotspot-survey` refresh ≈ 3 + ⌈P/8⌉ + 2R + 1 agents and 0.3M + ⌈P/8⌉ × 75k + R × 195k + 140k tokens (its SKILL.md §Cost gate; P = prior findings re-verified, R = clusters reviewed), and `/review` has no formula — quote its last measured run. A **first** refresh after a long gap re-reads every cluster and costs near a full hotspot run: the formula under-estimated the first dark-factory refresh by about 60 %.
- **Phases 3–4**, from the roster in §Phases 3–4 and the per-seat figures in `references/orchestration.md` §Model / effort routing.

Before posting the gate, check the account's remaining weekly usage headroom against the total estimate, Phase 2 included, and say at the gate whether the run fits. Measured on the first reify run (`review-all-reify-20261010`): the weekly limit fired mid-Phase 3 at 11.2M subagent tokens and 37 of 70 agents failed; the resume then re-ran ~1.2M tokens of completed calls (`references/orchestration.md` failure mode 16). A run that does not fit waits for the reset.

Calibration, the first dark-factory run (`review-all-dark_factory-20261008`, measured): Phase 2 8.6M tokens — hotspot refresh 5.66M over 89 min with 42 agents, `/review` full 2.97M over 133 min with 10 agents; phases 3–4 14.2M tokens over 50 seats; total 22.9M; wall clock 8:05 for phases 1–4; Phase 6 one sitting for 26 streams, then six `/prd` seats in wave 1. Quote these beside the formula figures until a second run replaces them.

## Phase 0 — pin, areas, instrument ages (inline)

1. **Pin one tree.** `AS_OF=$(git -C "$ROOT" rev-parse main)`; `run_id = review-all-<project_id>-<YYYYMMDD>[-<n>]`; `since` = the previous `review-all` report's `as_of_sha` or `none`. Create a detached read-only worktree every seat will read — main moves under a live merge queue and a three-hour run must read one tree:
   `git -C "$ROOT" worktree add --detach "$ROOT/.claude/worktrees/review-all-<run_id>" "$AS_OF"`. Never stash (CLAUDE.md §Working in the main checkout).
2. **Areas.** Parse `review/briefing.yaml` `subprojects` keys; add `repo`. One rule maps a path to an area (contract §3): the subproject whose member directory contains it; for paths under no member, the rule the overlay names. A path no rule covers is either a briefing defect (a workspace member or top-level package the briefing omits — report it for `/review-briefing`, exclude it from the run) or, when the overlay says so, `repo`; the report lists both. When the briefing names its members in prose rather than as member lists, the overlay carries the mapping as an ordered rule list — first match wins, a catch-all per area, sub-package rules where one package spans areas by file — and lists every mapping it inferred rather than read from the briefing as a briefing defect for `/review-briefing`; the snapshot records the rules in its `params` so the report can cite them. Measured on the first reify run (`review-all-reify-20261010`): the briefing had purpose prose and no member lists, and two crates (`reify-eval`, `reify-stdlib`) were split across several areas by file.
3. **Instrument ages and open findings.** For each of the latest hotspot findings artefact, `/review` report, census report + `docs/legibility/confusion-codebook.yaml`, and previous `/review-all` record: note `run_id`, `as_of_sha`, `git rev-list --count <its sha>..$AS_OF -- <source globs>`, and collect every finding with `disposition` `open` or `filed:` (→ the prior open set) and `accepted:` (→ `prior.standing`), joined on contract §2 `canonical`. A prior report with no `as_of_sha` (pre-contract) is stale by definition — no commits-since can be counted — and is recorded in `inputs_consumed` as `legacy:<path>`. A legacy markdown-only report (a hotspot survey with no findings JSON) cannot be refreshed at all; convert it by hand — each diagnosis becomes prior findings with file-grain anchors, `sub_area` = the diagnosis label, keys minted by `skills/hotspot-survey/scripts/findings_artefact.py key` — so the area seats re-verify them like any prior, and list it as `legacy:<path>` (first reify run: seven diagnoses became 20 priors; at the interim 7 still present, 4 gone, 8 unchecked). A defect in a sibling instrument found here (its report home gitignored against contract §5, a mis-named worktree) is filed as a task against that skill with `source="review-all"`, never fixed inside this run. A legacy `<area>/<sub>` area becomes the plain area in the key plus `sub_area`, and the finding is assigned to the slice holding the sub-area's files (else the area's first slice) so it is re-verified, never left unchecked. Entries with no key and no anchor become `leads`. Staleness verdict per instrument: hotspot stale at ≥ 50 source commits since or when its human-gate task is `done`; review stale on any source change (its `--since` mode is cheap enough to always run). The census is never run here.
4. **Memory.** `search(query="architecture decisions, standing quality rulings, accepted gaps", project_id=<overlay>)` → `leads` with `source: memory`.
5. **Launch provenance.** Once `plans/completion-driven-triggers-prd.md` lands, the normal way this skill starts is a `review_due` escalation from the previous run's human gate (`metadata.trigger_chain.skill == "review-all"`, the bare skill name), and its id reaches the run as `--launched-from <esc-id>`; record it in the report's `extra.launched_from` — Phase 6 resolves it per `skills/_shared/filing-the-trigger-chain.md` §5 step 2. A hand launch records `none`.

## Phase 1 — metrics snapshot (inline, scripts only; no LLM judgment)

The instrument is `scripts/quality_metrics_snapshot.py` (contract: `plans/quality-metrics-snapshot-prd.md` §Contract), and its measures live in `scripts/source_measures.py`; the overlay names other languages' instruments. It is a report: no verdicts, ranking, averages or thresholds beyond heuristic 14's two marks, and the quality doc's `Do not steer by` list is what it must never be read as.

**A project that is not a Python workspace or single package.** The Python snapshot measures a uv workspace's members, or a single src-layout package (no `[tool.uv.workspace]`: `[project].name` with `src/` and `tests/`), as `scripts/source_measures.py::workspace_domain` defines; on any other project the overlay's Metrics instruments slot names the instrument and it replaces step 2. For a Cargo workspace (with TypeScript, Python and shell beside it) that is `scripts/rust_metrics_snapshot.py` in this skill: `python3 <skill>/scripts/rust_metrics_snapshot.py --root <pinned worktree> --run-id <run_id> --out <scratch>/review-all/<run_id>/` writes `snapshot.json`, `import_graph.json` and `metrics_summary.txt` there, after its PROJECT RULES block is replaced by the overlay's (it has no `--diff` yet). Its complexity figure is lizard cyclomatic complexity (CCN) per function, read as the pair (max per function, file total) — not the quality doc's cognitive complexity, which has no working Rust binary (`rust-code-analysis-cli` does not build) — so the slice brief's cognitive columns carry `ccn_max`/`ccn_total` labelled CCN, and the report's metrics section says so. Measured on the first reify run (`review-all-reify-20261010`): 2,901 files / 1.99M lines in 5m25s, 195 files ≥ 2,000 lines, where the Python default would have measured about 2 % of the code.

1. **Previous snapshot.** The committed `plans/quality-metrics/*.json` whose `as_of_sha` is `since`, else the newest whose `as_of_sha` is an ancestor of `AS_OF` (`git merge-base --is-ancestor`). With none, omit `--diff`; the script then says there was no previous snapshot. The report names the snapshot it diffed.
2. **Measure the pinned worktree** with the main checkout's script and venv, into scratch:
   ```bash
   uv run --directory "$ROOT" python scripts/quality_metrics_snapshot.py \
     --root "$ROOT/.claude/worktrees/review-all-<run_id>" --run-id <run_id> \
     --out <scratch>/review-all/<run_id>/snapshot.json [--diff <previous>] \
     > <scratch>/review-all/<run_id>/snapshot-diff.txt
   ```
   It refuses a dirty domain or a HEAD that moved during the run, exiting 2 with the cause named on stderr. Exit 2 stops Phase 1: report the named cause. There is no inline-probe fallback. If the snapshot's `evidence.complete` is false, the report lists `evidence.unreadable`: those files' measures are unknown, never zero.
3. **The seats' inputs**, under `<scratch>/review-all/<run_id>/`:
   - `import_graph.json`: the snapshot's `import_graph` section copied unchanged, e.g. `python -c 'import json, sys; json.dump(json.load(open(sys.argv[1]))["import_graph"], open(sys.argv[2], "w"))' snapshot.json import_graph.json`. Its nodes are module import names; a file record's `module` joins them to paths.
   - `metrics_summary`: the `--summary <snapshot>` output, followed by the sections of `snapshot-diff.txt`, trimmed to ≤ 4k chars by dropping whole entries from the end of a section (noted `(+N more)`), never re-sorted.
4. **Home.** The script writes and never commits (the metrics PRD's decision 8). The snapshot is copied to `plans/quality-metrics/<run_id>.json` only in Phase 5, immediately before that phase's `git commit --only`, so no uncommitted file sits in the machine-operated checkout.

Task 5414 is done (`cdc15a6b56`, 2026-10-05): its per-package pinning counts live in `scripts/suite_census_pinning.py`, which after the metrics PRD's S1 enumerates through `scripts/source_measures.py::workspace_domain` as the snapshot does — reuse those counts rather than re-deriving them.

**Slices.** Hand-author the slice list (never delegate it): one seat per area; an area over ~40k source lines is split along the import graph's package clusters (overlay seeds first) into slices of ≤ ~40k lines. Pack by **source** lines, never total lines: pack src clusters to the cap and attach a package's tests to the slice holding most of its src (as index-only entries plus `tests_reaching_internals`); a test suite that is itself its area's product (reify's `tests/infra`) counts as src; an area with one small file is read inside a neighbouring slice, its READ FULLY entries carrying that area, and the prompt tells the seat. Measured on the first reify run: packing by total lines gave 61 slices, 20 of them pure-test seats with nothing to read fully; by src lines, 30. Within a slice, `read_fully` is the top ~10k lines (what a seat actually reads; below) ranked by cognitive max, fan-in, reach-back imports, cycle-closing deferred imports (`cycle_closing_imports`), re-exports, patch targets and fix density — never by line count or the raw deferred-import count — with each entry's numbers taken from the snapshot's file records (the mapping is in `references/orchestration.md` §The slice brief), and `index_only` is the rest. `tests_reaching_internals` comes from tests records' `private_patch_targets`; `size_alarms` from records whose `lines` reach `params.h14_alarm_lines`; `prior_open` and `leads` from Phase 0, filtered to the area.

**Rotation.** The ranking instruments move slowly, so without a rotation the same files top every run's read lists and index-only files stay index-only — a blind spot the method itself creates. Read `read_coverage` from the previous `/review-all` record (the newest `plans/review-all-<project_id>-*.json` whose `as_of_sha` is an ancestor of `AS_OF`; none → no rotation): a file a seat listed under `files_read_fully` or `partial` last run has its percentile-rank sum multiplied by 0.5, unless fix-grepped commits have touched it since that record's `as_of_sha`, in which case it keeps its rank. The report's read-coverage table then says how many files were read fully for the first time (`references/report-format.md` §Artefact 1).

**Read shortfall.** Size read lists at what a seat reads, not at the plan. Measured on the first reify run: 30 seats were handed 298 src files / 427k lines (15k each) and reported 234k read — 55 % of the plan, 27 % of 880k src lines; 12 seats read under half their plan, typically 4–8k lines of two 13–15k-line files before stopping; of the 16 files with a per-function CCN max ≥ 60, 9 were on a read list and none was reported read in full. Plan ~8–12k lines per opus/high seat, and give a slice whose read list is a few giant files two seats rather than one. The area prompt states the budget and the large-file protocol (`references/orchestration.md` failure mode 18).

**The generator.** The slice rules are hand-authored, the files are scripted — two scripts under `<scratch>/review-all/<run_id>/`, kept with the run's intermediates (the first dark-factory run's pair is the shape to lift, not code to reuse: its paths and rules are that run's):
1. *Build the full slice records* from `snapshot.json`, the fix-density counts, the Phase 0 prior set and the leads: an ordered list of `(seat, area, path predicate)` rules, applied first-match per source file, with a last catch-all rule per area so no file is unassigned (the script refuses a file no rule covers); a `sub_area → seat` map for priors whose anchor file is not in the tree; per file the snapshot record's measures; the ranking above as a sum of percentile ranks, with the rotation term; `read_fully` cut at the budget; priors assigned to the slice holding their anchor file, else their sub-area's seat, else the area's first slice; leads routed by `fix_where` paths, else keyword rules, capped per seat. It prints the roster (seat, files, lines, read-fully count and lines, index count, tests, alarms, priors, leads) and the unassigned priors for you to check before anything is launched.
2. *Write the seat briefs and the slim args*: one `<scratch>/seats/<seat>/brief.md` per slice in the order `references/orchestration.md` §The slice brief fixes, `<scratch>/prior_open.json` with every prior in full, and `args.json` in the `args` shape — each slice reduced to `seat`, `area`, `brief`, `read_fully` as path + lines, counts, `size_alarms`, `prior_open` as identities, and `prior` to `standing` + `full_path`. Measured: 820 KB with everything inline, 60 KB slim; the inline form cannot be emitted as a tool-call value.

## Phase 2 — stale instruments in delta mode

Run as sibling skill invocations (in this session, or spawned per `skills/spawn/SKILL.md` so they overlap), each against the same pinned tree and in findings-only mode, so the inner runs file no tickets and no trigger chain — this run's Phase 5 triages the merged findings and Phase 6 files the one chain:

- `/hotspot-survey --refresh --as-of $AS_OF --findings-only` when stale — re-verifies its open findings against the pinned tree and writes its findings artefact under the contract.
- `/review --as-of $AS_OF --since <previous review as_of_sha> --findings-only --launched-from none` — its delta mode.

Wait for both artefacts, add their `run_id`s to `inputs_consumed`, and fold their findings into the prior open set (Phase 1's generator writes it per slice and to `prior_open.json`) / `leads` by canonical. Their mechanical candidates are kept for the Phase 5 merge by key. The census is consumed only (its codebook entries carry `finding_key`; entries without one become `leads`).

## Phases 3–4 — the workflow

Pass the slim `args.json` Phase 1's generator wrote (the `args` shape in `references/orchestration.md`; seat briefs and `prior_open.json` stay on disk) and launch the script there through the Workflow tool (background), passing the parsed JSON as `args`. Routing is explicit on every seat; the session model is never inherited:

| Seat | Count (first DF run) | Model | Effort |
|---|---|---|---|
| area review | one per slice (20 on the first run) | opus | high |
| skeptic | one per seat (23 on the first run) | sonnet | high |
| calibration skeptic | 10, only when the area skeptics refute nothing | sonnet | high |
| cross-area lens | 4 (overlay may add) | opus | high |
| synthesis | 1 | fable, else opus / xhigh | high |
| critic | 1 | fable, else opus / xhigh | high |

Fable provenance follows `skills/team/SKILL.md` §Fable: a fable lead routes the two seats; an opus lead needs `--fable`, a statement in the conversation or the overlay's standing ruling, and otherwise runs the fallback and says so in `cost`. Hard rules, each bought with a failure the hotspot skill records: semantic validation of every seat with one re-run (`cleanSeat`/`degenerateSeat`), `pipeline()` from each seat into its skeptic, skeptics blinded to proposals, a barrier only before the cross lenses, digest before synthesis, three verdict classes counted separately, `log()` for everything dropped. Zero refutations from the area skeptics starts the script's `Calibrate` phase: a stratified sample of **ten** confirmed findings — every `high` first, then the rest — each re-verified by a fresh skeptic with per-measurement falsification, beside the cross lenses, returned as `yellow_flag_sample`. You read the sample's outcomes, not the code (§Lead context budget), and the digest copies it to the report's `extra.yellow_flag_sample`; either way compare the result's `calibration.refutation_rate` with the previous runs' `verification.refutation_rate` and say in `## Method` whether the skeptic is drifting. (First run: 536 findings across this run and its two Phase 2 siblings, 0 refuted; five sampled, five reproduced — a sample that small cannot tell "all real" from "skeptic too soft", which is why the skeptic now records a falsification attempt per measurement, the sample is ten, and seats run it.) Fallback when the Workflow tool is unavailable: Agent-tool fan-out with the `team-high` / `team-medium` seat types and the schemas as prose; pipelining and the journal are lost and the report says so.

## Lead context budget

The lead's context is the run's scarcest resource: the first dark-factory run reached ≈850k lead tokens, and reify carries about three times its code. So the lead never reads seat prose; it reads digests and summaries that scripts produce — the generator's roster, the Phase 5 digest of `workflow-result.json`, the rendered report. Calibration is run by seats (the `Calibrate` phase), filing with more than ~10 candidates by one seat (Phase 5), and the digest and the renders are scripts.

## Phase 5 — digest, keys, report, mechanical filings (inline)

The result is 300–600 KB and arrives truncated; write it to `<scratch>/review-all/<run_id>/workflow-result.json` and digest with a script, never by eye. The script has already mapped the synthesis's short ids back to canonicals and returned `short_ids` for the prose that still uses them, and `severity_corrections` with each one's acceptance — the digest applies the accepted ones to the findings record and flags the rest (`references/report-format.md` §Artefact 2); it never expands an abbreviation by hand. The digest mints keys by importing the one key implementation — `shared/src/shared/finding_key.py` once it lands, until then `skills/hotspot-survey/scripts/findings_artefact.py` (`finding_key`, `normalise_anchor`) — never a reimplementation, and the report states which it used; for every finding, prior verdict and refutation it sets `first_seen` (the run that first recorded the key, from the earliest artefact carrying it), appends this `run_id` to the prior's `last_seen` list on re-observation, fills `supersedes` from `supersedes_canonical`, carries `sub_area`, and applies the observation classes (including `unchecked`). It writes the JSON record and renders the markdown from it (its `## Method` opens with the contract §5 yaml block; the markdown is the ~60 KB human-facing form, `high` findings in full and the rest one line each), both named by `run_id`, per `references/report-format.md`, into the overlay's output directory (default `plans/`). Copy `<scratch>/review-all/<run_id>/snapshot.json` to `plans/quality-metrics/<run_id>.json`, then commit both plus that copied snapshot with `git commit --only <paths>` (docs-only: pre-commit skips pyright). Then remove the detached worktree: `git -C "$ROOT" worktree remove "$ROOT/.claude/worktrees/review-all-<run_id>"`.

Filing (skipped under `--no-file`): the mechanical candidates of this run and of the Phase 2 instruments are merged **by key** first (first run: 75 — 30 from this run, 35 from `/review`, 10 from the hotspot refresh — each row recording its sources), then mechanical findings with verdict `confirmed` — and `weakened` ones only after you have read the skeptic's note and still agree the cost is real — go through the contract §8 protocol in order — key lookup via `find_tasks_by_metadata(project_root, key="x_finding_key", value=<key>)`, which matches a list-valued key by membership, where a live task is `filed:<id>`, a `cancelled` task carrying `x_acceptance_reason` is `accepted:<id>` and one without it is absent; then `search_tasks` at `score_threshold ≥ 0.6` confirmed with `get_task`; then `submit_task` (curator path, not `planning_mode`) with `priority = severity`, `metadata.x_finding_key`, `x_finding_run`, `source="review-all"`, `files` file-level. Record the deciding step per finding in the report's Mechanical filings table; a ticket id is recorded as such until `resolve_ticket` names the task. Structural findings are filed by nobody here (contract §6). Store unreachable → file nothing, say so, dispositions stay `open`.

With more than ~10 candidates, hand filing to one seat (sonnet / medium) rather than doing it inline: it runs the contract §8 protocol above by script and returns a `filings.json` (per key: the deciding step and the task or ticket id) for the lead to read. The script is an async client over `scripts/migrate_metadata_modules_to_files.py::FusedMemoryClient` with `call_tool_with_retry` and `write_failure_reason`, in two passes — a search pass writing the `search_tasks` hits per key for the seat to decide step 2 on, then a submit pass that files the undecided keys and resolves their tickets. `resolve_ticket` takes `ticket` (not a task id) and 1–2 min each, with ReadTimeout retries; the first run's 60 tickets took about 50 min, so run the submit pass detached and poll its filings file.

## Phase 6 — deliberation, program, /prd, trigger chain

Close the run turn by proposing deliberation with the report path as the anchor, then follow `references/deliberation.md`: contested items first in a fixed shape (findings, next change, options with trade-offs and long-term consequences, invariant candidates in checkable form; under `--yes` the recommendation is taken as the ruling), rulings recorded as they are made, the program doc per `references/report-format.md` §Artefact 3, `/prd` per stream (agent team for `agent` streams, spawned interactive sessions for `spawn` streams — both lead-sequenced: seats author, the lead commits, seats decompose in seam order, the lead commits again), and the contract §11 trigger chain as the program's last act, filed per `skills/_shared/filing-the-trigger-chain.md` §§0–5 with `skill="review-all"` (bare name, no slash) and this `run_id` (gate identity is `metadata.trigger_chain`). Review-all specifics: the gate is filed when the first `/prd` stream files and each later session adds its `high`/`critical` tasks as dependencies, `commit_planning` only once the FILED table is complete; an older open `review-all` chain is superseded first (§5), and when this run was launched from that chain's `review_due` escalation (`--launched-from`), §5 step 2 resolves it `ran as <run_id>`; a program that files no tasks files no chain and the report's `extra` says so (§0).

## Graceful degradation

| Missing | Behaviour |
|---|---|
| `review/briefing.yaml` | stop; `/review-briefing` first (areas are its keys; no second list) |
| hotspot baseline | no `fix-history` lane: run `/hotspot-survey` whole if the cost gate accepts, else proceed with `leads` from git log only and say so |
| `/review` report | run `/review --since none` (whole) if accepted, else proceed; `inputs_consumed` says which |
| codebook | census consumption skipped; no `agent-transcripts` evidence; say so |
| `radon` / `complexipy` | `scripts/quality_metrics_snapshot.py` exits 2 naming the tool and its pin; run `uv sync --all-packages`. If still absent, the run has no snapshot, `metrics_summary` says so, and nothing is hand-measured in its place |
| project is neither a Python workspace nor a single src-layout package | the Python snapshot cannot run; use the overlay's instrument (Rust: `scripts/rust_metrics_snapshot.py`); never run the Python default and call 2 % of the code the domain |
| overlay | generic mode: Python defaults, areas from the briefing, output under `plans/` |
| fable not authorised | opus / xhigh on synthesis and critic; `cost` line says a fable seat was wanted and why |
| Workflow tool | Agent-tool fan-out with `team-*` seat types; no journal, no pipelining; report says so |
| Phase 2 siblings not run | fix-history and does-it-run evidence missing; neither heals by itself (no baseline, no trigger chain) — schedule each sibling's first whole run as "instrument debt" in the program doc |
| fused-memory / task store | file nothing (no tickets, no trigger chain); dispositions `open`; the next session files |
| `args` too large for a tool-call value (every slice inline) | never pass it; Phase 1's generator writes the seat briefs and `prior_open.json` to disk and a slim `args` (measured 820 KB → 60 KB); a run that cannot write scratch files cannot run |
| a seat echoes a different `seat`/`area` than its slice | the script overwrites both from the slice and logs the drift; the report's seat list is the slice list, and a join on the seat's own echo is never trusted (first run: 2 of 19 seats, 41 priors misreported `unchecked`) |
| `scripts/check_run_completion.py` absent in the dark-factory checkout (the §2 probe; never the target project) | `skills/_shared/filing-the-trigger-chain.md` §4 interim form instead of §3 |
| a seat degenerate twice | lane lost; listed under Dropped and degraded |

Never fail silently: the method header and §Dropped and degraded record exactly what ran.

## Writing to memory

The report is the home of every finding, measurement and proposal (contract §12); memory gets pointers.

- At the end of Phase 5: one `observations_and_summaries` entry — run id, as_of_sha, seats and cost, verification counts, areas with the highest-ranked cells, report and JSON paths — with `project_id` and `agent_id` per the overlay and `entities=` naming the mechanical tasks filed (`[]` if none).
- In Phase 6: one `decisions_and_rationale` entry per ruling, `entities=` naming the tasks or the gate id once they exist (`references/deliberation.md` §Recording).
- Not written: findings, metrics, proposals, the matrix, seat prose.

## Anti-triggers

- Does the software work after the orchestrator ran → `/review`.
- Where have bugs clustered → `/hotspot-survey` (this skill consumes and refreshes it).
- Confusion sightings → `/census` (consumed, never run here).
- One module → `/study`; one bug → `/deb`.
- Authoring remediation → `/prd`, from the program doc.
- Anything unattended: the human gate dispatches to a human or a spawned session; the instrument never runs from it on its own.

## Reference

- `references/orchestration.md` — the workflow template: `args` shape, schemas, prompts, routing, failure modes, design notes.
- `references/report-format.md` — report, findings JSON and program-doc contract.
- `references/deliberation.md` — the deliberation protocol, `/prd` launch, trigger chain.
- `references/project-overlay.md` — overlay schema for specializing this skill per project.
