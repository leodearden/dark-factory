---
name: review-all
description: "Deep, infrequent, human-attended review of a WHOLE project's architectural and code quality against docs/code-quality.md — pins one tree, takes a whole-repo metrics snapshot (a report, never a gate), refreshes the stale instruments in delta mode (/hotspot-survey --refresh, /review --since) and consumes the census codebook, then runs a heuristic-driven architecture pass: one seat per area judging all fourteen heuristics and both stances, cross-area seats for SPOT / layering / orthogonal dimensions / stateless interactions, a skeptic per seat blinded to proposals, a synthesis joined by finding key with every prior instrument's open findings, and a critic of that synthesis; ends in deliberation with the human, a program doc and /prd per stream, and files the completion-driven trigger for the next run. ALWAYS use this skill for: '/review-all', 'review the whole project against the quality heuristics', 'whole-repo architecture and code-quality review', 'what should we refactor across the whole codebase', 'the quarterly factory self-review', or when the human gate task 'Run /review-all on <project>' dispatches. This is a serious commitment — the first dark-factory run is plausibly 5–8M subagent tokens, 4–5 h machine time and 20–30 h of human deliberation and PRD work — so state the cost and get a go before phase 3. NOT for: does-it-run correctness of landed work (/review), bug-history hotspots alone (/hotspot-survey), confusion sightings (/census), one module (/study), one bug (/deb), authoring the remediation PRDs (/prd — fed by this skill's program doc), or anything unattended."
argument-hint: "[--plan] [--areas a,b] [--fable] [--no-file] [--launched-from <esc-id>] [--resume <run_id>] [project-root — omit for the current project]"
---

# /review-all — whole-project quality review

Answers one question about a whole project: **where does the next agent change cost the most, under `docs/code-quality.md`, and what would remove that cost?** Every other quality instrument answers a narrower question from narrower evidence; this one reads them all first (`docs/quality-findings-contract.md` §9), runs only what is stale, adds the one pass nothing else does — judging every area against the fourteen heuristics and both stances — and ends in human deliberation. It supersedes the "quarterly factory self-review" priced in the agent-capacity study. Cadence is task completion, not the calendar (contract §11).

```
Phase 0  (you)                 pin as_of_sha; areas; instrument ages + open findings; cost gate
Phase 1  (you, scripts)        whole-repo metrics snapshot — report, never ranking
Phase 2  (sibling skills)      /hotspot-survey --refresh, /review --since   (census consumed, never run)
Phase 3  Workflow  ~15+4 seats area seats → blinded skeptics → cross-area seats → skeptics
Phase 4  Workflow  2 seats     synthesis → critic (one revision)
Phase 5  (you)                 digest, mint keys, report + findings JSON, file mechanical findings
Phase 6  (you + human)         deliberation → program doc → /prd per stream → trigger chain
```

## Parse invocation

```
/review-all                      → whole project, all phases, human present
/review-all --plan               → phases 0–1 and the roster + cost only; no seats launched
/review-all --areas orchestrator,shared   → restrict phase 3 to these briefing areas (cross lenses still run)
/review-all --fable              → pre-authorise the synthesis and critic seats on fable
/review-all --no-file            → report only: no tickets, no trigger chain
/review-all --launched-from <esc-id>   → the review_due escalation that spawned this run (the normal launch path); omit or `none` for a hand launch
/review-all --resume <run_id>    → resume the workflow from its journal (same args file)
```

A trailing path is the project root; otherwise `git rev-parse --show-toplevel`.

## Step 0 — overlay and prerequisites

```bash
ROOT="$(git rev-parse --show-toplevel)"
ls "$ROOT/.claude/skills/review-all/project.md" 2>/dev/null && echo "overlay present" || echo "no overlay — generic mode"
ls "$ROOT/review/briefing.yaml" "$ROOT/docs/code-quality.md" "$ROOT/docs/quality-findings-contract.md"
```

Read the overlay if present (`references/project-overlay.md`); it supplies identity, path→area rules, the per-language metrics instruments, slice seeds, report homes and hand-off conventions. Read `docs/code-quality.md` in full yourself before anything else: its Definition is the filter every seat applies, and its `Do not steer by` list binds you as the lead.

**No briefing → stop.** Area keys are the briefing's `subprojects` keys plus `repo` and there is no second list (contract §3); a run without them mints no joinable keys. Route to `/review-briefing` and return.

## Cost gate

This is a human-attended skill; AFK autonomy does not apply. After Phase 1, post the roster (seat, model, effort, read budget, expected tokens) with the total token and wall-clock estimate and the human-hours Phase 6 implies, then wait for a go. A first run on a mature repo is a large commitment by design — say so plainly rather than trimming the roster to look cheap. `--plan` stops here.

## Phase 0 — pin, areas, instrument ages (inline)

1. **Pin one tree.** `AS_OF=$(git -C "$ROOT" rev-parse main)`; `run_id = review-all-<project_id>-<YYYYMMDD>[-<n>]`; `since` = the previous `review-all` report's `as_of_sha` or `none`. Create a detached read-only worktree every seat will read — main moves under a live merge queue and a three-hour run must read one tree:
   `git -C "$ROOT" worktree add --detach "$ROOT/.claude/worktrees/review-all-<run_id>" "$AS_OF"`. Never stash (CLAUDE.md §Working in the main checkout).
2. **Areas.** Parse `review/briefing.yaml` `subprojects` keys; add `repo`. One rule maps a path to an area (contract §3): the subproject whose member directory contains it; for paths under no member, the rule the overlay names. A path no rule covers is either a briefing defect (a workspace member or top-level package the briefing omits — report it for `/review-briefing`, exclude it from the run) or, when the overlay says so, `repo`; the report lists both.
3. **Instrument ages and open findings.** For each of the latest hotspot findings artefact, `/review` report, census report + `docs/legibility/confusion-codebook.yaml`, and previous `/review-all` record: note `run_id`, `as_of_sha`, `git rev-list --count <its sha>..$AS_OF -- <source globs>`, and collect every finding with `disposition` `open` or `filed:` (→ `prior.open`) and `accepted:` (→ `prior.standing`), joined on contract §2 `canonical`. A legacy `<area>/<sub>` area becomes the plain area in the key plus `sub_area`, and the finding is assigned to the slice holding the sub-area's files (else the area's first slice) so it is re-verified, never left unchecked. Entries with no key and no anchor become `leads`. Staleness verdict per instrument: hotspot stale at ≥ 50 source commits since or when its human-gate task is `done`; review stale on any source change (its `--since` mode is cheap enough to always run). The census is never run here.
4. **Memory.** `search(query="architecture decisions, standing quality rulings, accepted gaps", project_id=<overlay>)` → `leads` with `source: memory`.
5. **Launch provenance.** Once `plans/completion-driven-triggers-prd.md` lands, the normal way this skill starts is a `review_due` escalation from the previous run's human gate (`metadata.trigger_chain.skill == "review-all"`, the bare skill name), and its id reaches the run as `--launched-from <esc-id>`; record it in `args`/the report's `extra.launched_from` — Phase 6 resolves it per `skills/_shared/filing-the-trigger-chain.md` §5 step 2. A hand launch records `none`.

## Phase 1 — metrics snapshot (inline, scripts only; no LLM judgment)

The quality doc's `What to measure` table is the instrument list; its `Do not steer by` list is what the snapshot must never be read as. Default Python instruments (the overlay names other languages'): `radon raw`, `complexipy`, and the AST helpers of `scripts/merge_lane_metrics.py` — `file_size_measures` (raw and prose lines), `function_local_imports`, `reexport_names`, `file_cognitive_measures`, `tracked_python_files`. `patch_targets` there is lane-scoped; write the generic AST walk (string arguments of `patch(`/`monkeypatch.setattr(` containing a `._` segment) alongside. Install with `uv pip install radon complexipy` if absent.

The snapshot's home is the committed `plans/quality-metrics/<run_id>.json` (the metrics PRD's decision 8: the script writes, the skill commits — this run commits it alongside the report in Phase 5). Produce:

- `plans/quality-metrics/<run_id>.json` — the snapshot in the shape `plans/quality-metrics-snapshot-prd.md` §Contract fixes: per file lines, prose share, cognitive max per function and module total, fan-in, fan-out, reach-back imports, function-local imports, re-export names, test patch targets into privates, private reads; per area and whole repo the totals and the files over `soft_lines` / `alarm_lines`; and its `import_graph` section `{edges, reach_back, deferred, cycles}`. With `scripts/quality_metrics_snapshot.py --run-id <run_id> --out <path> --diff <previous>` once it lands; until then inline probes writing that same shape, and the report says so.
- `--diff` against the one previous snapshot nearest `since` (the `plans/quality-metrics/*.json` whose `as_of_sha` is `since`, else the newest one older than `AS_OF`); same instrument, same rule — the report writes which snapshot it diffed.
- under `<scratch>/review-all/<run_id>/`: `import_graph.json` (the snapshot's section copied unchanged — the path seats read) and `metrics_summary`, ≤ 4k chars of tables for the seat prompts including the delta column.

Task 5414 widens the merge-lane ratchet enumeration — reuse its enumeration when it has landed rather than re-deriving a count it already pins.

**Slices.** Hand-author the slice list (never delegate it): one seat per area; an area over ~40k source lines is split along the import graph's package clusters (overlay seeds first) into slices of ≤ ~40k lines. Within a slice, `read_fully` is the top ~15k lines ranked by cognitive max, fan-in, reach-back imports, re-exports, patch targets and fix density — never by line count — and `index_only` is the rest. `tests_reaching_internals` comes from the patch-target measure; `size_alarms` from the ceilings; `prior_open` and `leads` from Phase 0, filtered to the area.

## Phase 2 — stale instruments in delta mode

Run as sibling skill invocations (in this session, or spawned per `skills/spawn/SKILL.md` so they overlap), each against the same pinned tree and in findings-only mode, so the inner runs file no tickets and no trigger chain — this run's Phase 5 triages the merged findings and Phase 6 files the one chain:

- `/hotspot-survey --refresh --as-of $AS_OF --findings-only` when stale — re-verifies its open findings against the pinned tree and writes its findings artefact under the contract.
- `/review --as-of $AS_OF --since <previous review as_of_sha> --findings-only --launched-from none` — its delta mode.

Wait for both artefacts, add their `run_id`s to `inputs_consumed`, and fold their findings into `prior.open` / `leads` by canonical. The census is consumed only (its codebook entries carry `finding_key`; entries without one become `leads`).

## Phases 3–4 — the workflow

Write `<scratch>/review-all/<run_id>/args.json` to the `args` shape in `references/orchestration.md` and launch the script there through the Workflow tool (background), passing the parsed JSON as `args`. Routing is explicit on every seat; the session model is never inherited:

| Seat | Count (first DF run) | Model | Effort |
|---|---|---|---|
| area review | one per slice (~15) | opus | high |
| skeptic | one per seat (~19) | sonnet | high |
| cross-area lens | 4 (overlay may add) | opus | high |
| synthesis | 1 | fable, else opus / xhigh | high |
| critic | 1 | fable, else opus / xhigh | high |

Fable provenance follows `skills/team/SKILL.md` §Fable: a fable lead routes the two seats; an opus lead needs `--fable`, a statement in the conversation or the overlay's standing ruling, and otherwise runs the fallback and says so in `cost`. Hard rules, each bought with a failure the hotspot skill records: semantic validation of every seat with one re-run (`cleanSeat`/`degenerateSeat`), `pipeline()` from each seat into its skeptic, skeptics blinded to proposals, a barrier only before the cross lenses, digest before synthesis, three verdict classes counted separately, `log()` for everything dropped. A YELLOW FLAG log (zero refutations) means you re-verify five confirmed findings inline before Phase 5 and record the result. Fallback when the Workflow tool is unavailable: Agent-tool fan-out with the `team-high` / `team-medium` seat types and the schemas as prose; pipelining and the journal are lost and the report says so.

## Phase 5 — digest, keys, report, mechanical filings (inline)

The result is 300–600 KB and arrives truncated; write it to `<scratch>/review-all/<run_id>/workflow-result.json` and digest with a script, never by eye. The digest mints keys by importing the one key implementation — `shared/src/shared/finding_key.py` once it lands, until then `skills/hotspot-survey/scripts/findings_artefact.py` (`finding_key`, `normalise_anchor`) — never a reimplementation, and the report states which it used; for every finding, prior verdict and refutation it sets `first_seen` (the run that first recorded the key, from the earliest artefact carrying it), appends this `run_id` to the prior's `last_seen` list on re-observation, fills `supersedes` from `supersedes_canonical`, carries `sub_area`, and applies the observation classes (including `unchecked`). It writes the JSON record and renders the markdown from it (its `## Method` opens with the contract §5 yaml block), both named by `run_id`, per `references/report-format.md`, into the overlay's output directory (default `plans/`). Commit both plus `plans/quality-metrics/<run_id>.json` with `git commit --only <paths>` (docs-only: pre-commit skips pyright). Then remove the detached worktree: `git -C "$ROOT" worktree remove "$ROOT/.claude/worktrees/review-all-<run_id>"`.

Filing (skipped under `--no-file`): mechanical findings with verdict `confirmed` — and `weakened` ones only after you have read the skeptic's note and still agree the cost is real — go through the contract §8 protocol in order — key lookup via `find_tasks_by_metadata` or, until it lands, the read-only forensic query per CLAUDE.md §"Forensic reads of tasks.db" testing membership with `json_each(metadata, '$.x_finding_key')` — the field may be a list, so equality on `json_extract` misses (shape from `scripts/tasks_db_schema.py`), where a live task is `filed:<id>`, a `cancelled` task carrying `x_acceptance_reason` is `accepted:<id>` and one without it is absent; then `search_tasks` at `score_threshold ≥ 0.6` confirmed with `get_task`; then `submit_task` (curator path, not `planning_mode`) with `priority = severity`, `metadata.x_finding_key`, `x_finding_run`, `source="review-all"`, `files` file-level. Record the deciding step per finding in the report's Mechanical filings table; a ticket id is recorded as such until `resolve_ticket` names the task. Structural findings are filed by nobody here (contract §6). Store unreachable → file nothing, say so, dispositions stay `open`.

## Phase 6 — deliberation, program, /prd, trigger chain

Close the run turn by proposing deliberation with the report path as the anchor, then follow `references/deliberation.md`: contested items first in a fixed shape (findings, next change, options with trade-offs and long-term consequences, invariant candidates in checkable form), rulings recorded as they are made, the program doc per `references/report-format.md` §Artefact 3, `/prd` per stream (agent team for `agent` streams, spawned interactive sessions for `spawn` streams), and the contract §11 trigger chain as the program's last act, filed per `skills/_shared/filing-the-trigger-chain.md` §§0–5 with `skill="review-all"` (bare name, no slash) and this `run_id` (gate identity is `metadata.trigger_chain`). Review-all specifics: the gate is filed when the first `/prd` stream files and each later session adds its `high`/`critical` tasks as dependencies, `commit_planning` only once the FILED table is complete; an older open `review-all` chain is superseded first (§5), and when this run was launched from that chain's `review_due` escalation (`--launched-from`), §5 step 2 resolves it `ran as <run_id>`; a program that files no tasks files no chain and the report's `extra` says so (§0).

## Graceful degradation

| Missing | Behaviour |
|---|---|
| `review/briefing.yaml` | stop; `/review-briefing` first (areas are its keys; no second list) |
| hotspot baseline | no `fix-history` lane: run `/hotspot-survey` whole if the cost gate accepts, else proceed with `leads` from git log only and say so |
| `/review` report | run `/review --since none` (whole) if accepted, else proceed; `inputs_consumed` says which |
| codebook | census consumption skipped; no `agent-transcripts` evidence; say so |
| `radon` / `complexipy` | `uv pip install radon complexipy`; if still absent, AST-only measures and cognitive complexity marked unmeasured in the snapshot |
| overlay | generic mode: Python defaults, areas from the briefing, output under `plans/` |
| `scripts/quality_metrics_snapshot.py` | inline probes writing the snapshot shape to `plans/quality-metrics/<run_id>.json`; report says so |
| fable not authorised | opus / xhigh on synthesis and critic; `cost` line says a fable seat was wanted and why |
| Workflow tool | Agent-tool fan-out with `team-*` seat types; no journal, no pipelining; report says so |
| fused-memory / task store | file nothing (no tickets, no trigger chain); dispositions `open`; the next session files |
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
