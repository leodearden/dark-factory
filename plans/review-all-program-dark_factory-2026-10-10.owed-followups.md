# Owed follow-ups of the first dark-factory /review-all program

Companion to `plans/review-all-program-dark_factory-2026-10-10.md` (the program doc; rulings R1–R92).
Run id `review-all-dark_factory-20261008`. Written 2026-11-13.

## What this file is

The program authored and filed all 26 streams (545 tasks). A few filings and edits could not be done at
filing time, because each waits on something landing or deploying first. Until 2026-11-13 they were listed
only in the program lead's private notes. `docs/quality-findings-contract.md` says a disposition lives in
the task store, never in a side ledger, so every item below now has an owner in the store:

- a **human gate task** that fires when the item's trigger holds (most items), or
- the **filed task itself**, when the trigger already held on 2026-11-13, or
- an **existing task** that already carries the work.

Each gate is a pure human gate: `task_kind='deterministic'`, `metadata.always_escalates=true`, no
`before_done` (`docs/task-authoring.md` §5). When its dependencies are terminal it files a born-at-L2
escalation and blocks; whoever picks it up acts from the matching section below, then resolves the
escalation with `resume`, which marks the gate done. Every gate carries `source: review-all` and
`x_finding_run: review-all-dark_factory-20261008`. No gate is a dependency of the program capstone, gate
6698, and nothing depends on any gate.

A dependency on a **local** task that ends `cancelled` satisfies the dependent just as `done` does
(`docs/task-authoring.md` §3.1), so every gate also fires when one of its triggers was cancelled; each
gate's description and section say what to do then. A **cross-project** dependency (`metadata.external_deps`)
that ends `cancelled` does not satisfy: the dependent is blocked with an escalation to a human (§3.2).

| § | Item | Owner in the store | Trigger (depends on) |
|---|---|---|---|
| 1 | S6 ξ and υ: 14-day lifecycle-refusal quiet period, then `hold_lane` on dark-factory (and reify) | gate **9924** | 9186 |
| 2 | S10 σ: 30-day combine-collision check | gate **9925** | 9237 |
| 3 | S9: 14-day L0 settle check | **filed: task 9923** (trigger already held) | 9190 (the check's own dependency) |
| 4 | R37: remove `convert_to_blocked_enforce` from eight other project yamls | gate **9926** (72 h delayed milestone) | 9212 |
| 5 | R42: backfill `metadata.files` on three done tasks | gate **9927** (24 h delayed milestone) | 9204 |
| 6 | Reify: remove the `speculation_probe` stanza before S7 β deploys | **reify task 9997**; dark-factory 9133 depends on `reify:9997` | — |
| 7 | S13 ρ: fortnightly recon category-candidate check (recurring chain) | gate **9928** | 9751 |
| 8 | S2 ξ: three-day check for unidentified escalation authority acts | gate **9929** | 9817, 9843 |
| 9 | S2 ο: the escalation identity denial | filed by gate 9929's act (a second gate) | S2 ξ (not yet filed), `reify:9996` |
| 10 | Amend to 6254, note to 4809 | gate **9930** | 6254, 4809 |
| 11 | Actions on 5258, 5912, 6695, 3498 | gate **9931** | 5258, 5912, 6695, 3498 |
| 12 | Amends to 8143, 8262, 7008, 3678, 4875; decide 5700 | gate **9932** | 8143, 8262, 7008, 3678, 4875, 5700 |
| 13 | Already owned: S24 interval flip; S16 yaml sweep and refusal | tasks 9861; 9718, 9719 | — |
| 14 | Dropped: edge 5153 → 9216 | — (ruled R59) | — |
| 15 | Not covered here | — | — |

## Common procedure for filing a task named below

1. Dedup: `search_tasks` on the title (threshold 0.5); confirm any hit at 0.6 or above with `get_task`.
   Do not rely on `find_tasks_by_metadata(key="x_finding_run", …)` for dedup: it returns hundreds of rows.
2. `submit_task(..., planning_mode=True, project_root="/home/leo/src/dark-factory")` (the task is born
   `deferred`, so the scheduler cannot take it early), then `add_dependency` for each edge, then
   `commit_planning(task_ids="<id>")` to `pending`, then `get_task` to verify status, metadata and edges.
3. A `submit_task` that times out may still have created the task: read the store (`search_tasks` on the
   title, or `find_tasks_by_metadata(key="prd_path", value=<path>)`) before submitting again.
4. A deterministic task's `before_done.script` must exist on main and be executable when it is submitted
   (`fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_script_under`;
   ruling R58). This is why most items below wait for a script task to land.
5. Filing with `prd_path` and `prd_task_label` lets `commit_planning` stamp the plan's capability-manifest
   sidecar block. A sidecar change is a repo write: commit it directly to main
   (`git commit --only <sidecar>`) after `shared/tests/test_capability_manifest.py` passes, read by exit code.
6. Every new task carries `metadata.source: "review-all"` and
   `metadata.x_finding_run: "review-all-dark_factory-20261008"`.

---

## 1. S6 ξ and υ: lifecycle-refusal quiet period, then `hold_lane`

**Owner:** gate 9924, "Review-all follow-up: file S6's 14-day lifecycle-refusal quiet-period check (ξ)
and the dark-factory hold_lane flip (υ)". Depends on 9186.

**Ruling.** R32 (Leo, sitting key S6-5): a refused merge-lane lifecycle transition first escalates and the
lane continues; after the operative registry has run 14 days with zero refusals on dark-factory and reify,
a refusal holds the lane instead (one restart-only setting
`merge_lane.lifecycle_refusal_policy: escalate | hold_lane`, default `escalate`), flipped per project. R58:
the check is filed only once its script is on main. Plan: `plans/review-all-s6-merge-lane-lifecycle-outcome-types-and-provenance-prd.md`
(§Decomposition plan ξ and υ; §Live tasks).

**Trigger.** Task 9186 (S6 φ: the policy setting, the hold arm and `scripts/lifecycle_refusal_census.py`)
is done. Check: `git ls-files -s scripts/lifecycle_refusal_census.py` on main shows mode `100755`.
If 9186 was cancelled: follow its `metadata.superseded_by` to the task that now writes the census script
and the policy; file ξ and υ only once that script is on main; if no task carries them, ask Leo whether
R32's second stage still stands.

**Act.** File three tasks under planning mode, then `commit_planning` each to `pending`:

**ξ** — title `S6 ξ: Lifecycle-refusal quiet period: 14 days, dark-factory and reify`
- `task_kind='deterministic'`, priority medium, `files: []`, `always_escalates: false`.
- `metadata.milestone = {"mode": "delayed", "after_secs": 1209600}` (fires 14 days after 9186 is done).
- `metadata.before_done = {"kind": "predicate", "script": "scripts/lifecycle_refusal_census.py", "args":
  ["--runs-db", "/home/leo/src/dark-factory/data/orchestrator/runs.db", "--runs-db",
  "/home/leo/src/reify/data/orchestrator/runs.db", "--window-days", "14"], "timeout_secs": 120}`.
  Before filing, run the script by hand with those arguments and confirm its flags still match what 9186
  landed.
- `prd_path: "plans/review-all-s6-merge-lane-lifecycle-outcome-types-and-provenance-prd.md"`,
  `prd_task_label: "ξ"`, no finding key. Depends on 9186. Not a gate-6698 dependency.
- Description: "Deterministic predicate check (R32, sitting key S6-5). Fires 14 days after S6 φ 9186 is
  done. Exit 0 = zero lifecycle refusals on dark-factory and reify in the last 14 days, all of them under
  the operative registry: done, which releases υ. Any refusal: a born-at-L2 milestone_check_failed to Leo
  carrying the per-store counts; the policy stays `escalate` until he resolves it (a resume re-runs the
  check) or abandons this task."

**υ** — title `S6 υ: Arm hold_lane on dark-factory`
- `task_kind='normal'`, `metadata.complexity='simple'`, priority medium, about 10 lines.
- Files: `dark-factory-orchestrator.yaml`. Work: set `merge_lane.lifecycle_refusal_policy: hold_lane`
  (the key 9186 added; nest it where 9186's config model reads it).
- `prd_path` as ξ, `prd_task_label: "υ"`, no finding key. Depends on ξ.
- Signal: after the next fleet redeploy (the value is restart-only), `get_merge_queue` on dark-factory
  reports `lifecycle_refusal_policy: hold_lane`.

**Reify flip gate** — title `Review-all follow-up: flip reify's merge_lane.lifecycle_refusal_policy to hold_lane (after S6 ξ)`
- A pure human gate shaped like this one (deterministic, `always_escalates: true`, no `before_done`,
  `files: []`, `source`, `x_finding_run`), depending on ξ.
- Description: "When S6 ξ is done (14 quiet days on dark-factory and reify), make reify's one-line flip by
  hand: in /home/leo/src/reify/dark-factory-orchestrator.yaml set `merge_lane.lifecycle_refusal_policy:
  hold_lane`, one yaml-only `git -C /home/leo/src/reify commit --only dark-factory-orchestrator.yaml` on
  reify main, best effort (R32; S8-7's precedent). Reify's scheduler is halted, so a reify-store task would
  not dispatch. The value is restart-only: it takes effect at reify's orchestrator's next restart. The
  other seven projects stay `escalate`. If ξ was cancelled, do nothing and say so. Resolve citing the sha."

Resolve gate 9924 naming the three new ids.

---

## 2. S10 σ: 30-day combine-collision check

**Owner:** gate 9925, "Review-all follow-up: file S10's 30-day combine-collision check (σ)". Depends on 9237.

**Ruling.** R44 (Leo, sitting key S10-7): a key collision in the curator's combine is an audited refusal
with the token `candidate_key_collision`; 30 days after the combine change, a deterministic check counts
them across all projects, and "frequent" (at least 5 collisions, or at least 1% of combine writes) means
address it more elegantly. R58: filed once the census script is on main. Plan:
`plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md` (§Sketch σ,
§Decomposition plan ρ and σ).

**Trigger.** Task 9237 (S10 ρ, `scripts/combine_collision_census.py`) is done; it depends on 9214 (S10 ζ,
the combine that writes the refusal), so both have landed. Check: `git ls-files -s
scripts/combine_collision_census.py` shows mode `100755`. If 9237 was cancelled: file σ only if the script
is on main by another task; otherwise ask Leo whether R44's count still stands.

**Act.** File **σ** — title `S10 σ: Deterministic 30-day collision check`
- `task_kind='deterministic'`, priority medium, `files: []`, `always_escalates: false`.
- `metadata.before_done = {"script": "scripts/combine_collision_census.py", "args": ["--days", "30"],
  "kind": "predicate", "timeout_secs": 120}`; `metadata.milestone = {"mode": "delayed", "after_secs": 2592000}`.
- `prd_path: "plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md"`,
  `prd_task_label: "σ"`, no finding key of its own. Depends on 9214 and 9237. Not a gate-6698 dependency.
- Description: "Deterministic 30-day check (R44, sitting key S10-7). Its timer starts when S10 ζ 9214 and
  ρ 9237 are both done, so the window holds only combines made after ζ. The script counts
  `candidate_key_collision` combine refusals per project from the shared audit file. Exit 0 (fewer than 5
  collisions and under 1% of the window's combine writes): done, counts in done_provenance. Exit 1: a
  born-at-L2 milestone_check_failed to Leo with the counts; Leo or the L2 watcher files the more elegant
  fix (a deterministic task cannot file it)."

Resolve gate 9925 naming σ's id.

---

## 3. S9: 14-day L0 settle check — FILED 2026-11-13 as task 9923

**Ruling.** R40 (Leo, sitting key S9-7): task 9190 (S9 ε) makes the steward decide every published
outcome of a blocking L0 escalation; 14 days after it lands, a deterministic check must show it did not
strand or wrongly dismiss more L0s. R58: filed once the script is on main. Plan:
`plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md` contract C10 and §Live tasks
("file the C10 check task once π is on main … so it is filed before ε lands and its 14-day clock starts
when ε is done").

**Why it was filed now.** The trigger already held: task 9164 (S9 π) is done and
`scripts/check_l0_settle_outcomes.py` is on main with mode `100755` (checked 2026-11-13; its flags match
C10). The older note that it should wait for 9190 to land contradicted C10, which files it before 9190
lands so the pre-landing window is clean; C10 was followed.

**As filed.** Task 9923, `Check stranded and wrongly dismissed L0s 14 days after the steward-decides-outcome change landed`:
deterministic, medium, depends on 9190, `milestone = {"mode": "delayed", "after_secs": 1209600}`,
`before_done = {"kind": "predicate", "script": "scripts/check_l0_settle_outcomes.py", "args": ["--queue-dir",
"data/escalations", "--window-days", "14", "--baseline-stranded-per-100", "3.6", "--ratio", "2.0",
"--min-stranded", "8", "--max-reaper-wrong", "3", "--max-pending-hours", "24"], "timeout_secs": 120}`,
`x_finding_key: [fk-bab6058b23c8, fk-e39550272df9]`, `prd_path` the S9 plan (no `prd_task_label`: C10 is
not a labelled leaf, so `commit_planning` reported it as an out-of-plan follow-up, as expected). A failure
opens the joint L0 end-of-life redesign with the harness plan (S8). Nothing is owed here any more.

---

## 4. R37: remove `convert_to_blocked_enforce` from the eight other project yamls

**Owner:** gate 9926, "Review-all follow-up: remove convert_to_blocked_enforce from the eight other
project yamls once its default flip is deployed fleet-wide". Depends on 9212; delayed milestone 259,200 s.

**Ruling.** R37 (Leo, sitting key S8-7), in his confirmed order: (1) task 9212 (S8 η) flips the field's
code default to True; (2) after that is deployed fleet-wide, the lead removes the key from the eight other
project yamls, best effort, one yaml-only commit per repository, recording each sha or skip reason on
S8 δ's task; (3) task 9219 (S8 δ) deletes the field and this repository's own line. Plan:
`plans/review-all-s8-harness-composition-root-and-slot-registry-prd.md` D18.

**Trigger.** 9212 is done AND every project's orchestrator has restarted onto it. The 72-hour delayed
milestone covers the longest wait for a unit's next restart on this host (R70). Before acting, run
`scripts/orchestrator-watchdog.py --report` (read-only) or compare each `orchestrator-<project>.service`
`ActiveEnterTimestamp` (`systemctl --user show <unit> -p ActiveEnterTimestamp`) with 9212's merge commit
time; if any started earlier, wait and resume later. Removing a line before its orchestrator runs the new
default would put that project into log mode at its next restart. If 9212 was cancelled: edit nothing;
the default never flipped. Edits become safe again only once 9219 (which deletes the field) is deployed,
when a leftover line merely files one self-closing unknown-key escalation.

**Act.**
1. In each of `autopilot-video`, `know-live`, `pump-web-ui`, `reify`, `scoping-intake`, `solar-challenge`,
   `solar-challenge-platform`, `stream-touch` under `/home/leo/src/`, delete the line
   `convert_to_blocked_enforce: true` from `dark-factory-orchestrator.yaml`, plus any comment lines directly
   above it that describe only that setting. (In `solar-challenge-platform`, `orchestrator.yaml` is a
   symlink to `dark-factory-orchestrator.yaml`: edit the real file once.) Re-grep every project yaml on the
   host first; a project that gained the key since 2026-11-10 is treated the same way.
2. One commit per repository on its `main`: `git -C <repo> commit --only dark-factory-orchestrator.yaml -m
   "config: drop convert_to_blocked_enforce (now the code default; dark-factory R37)"`. Best effort: a repo
   that is dirty on that file, not on main, or refuses the commit is skipped.
3. Append to task 9219's details (`update_task(id="9219", details=…, append=True)`) one line per repository:
   the sha, or `skipped: <why>`.

Resolve gate 9926 with the same list.

---

## 5. R42: backfill `metadata.files` on the three done tasks that differ

**Owner:** gate 9927, "Review-all follow-up: backfill metadata.files from the retired files_to_modify key
on the three done tasks that differ". Depends on 9204; delayed milestone 86,400 s.

**Ruling.** R42 (Leo, sitting key S10-5): `metadata.files_to_modify` is destroyed. Fused-memory refuses an
incoming payload carrying it, every reader reads `files` only, and "the lead backfills the three rows that
differ (dark-factory 2507, reify 4344 and 4865 as of 2026-11-10) after the refusal deploys". Plan:
`plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md` (§Sketch δ;
§Resolved design decisions, the ordering "refusal first, backfill second").

**Trigger.** Task 9204 (S10 δ) is done AND fused-memory has restarted onto it:
`systemctl --user show fused-memory -p ActiveEnterTimestamp` is later than 9204's merge commit time (the
on-merge coordinator restarts it; the one-day milestone leaves it time). If 9204 was cancelled: follow its
`superseded_by`; if no task retires the key, readers still union both keys and the backfill is moot —
resolve with that note.

**Act.**
1. Re-measure every store, read-only: for each project root in
   `shared/src/shared/project_layout.py::known_project_roots`, open
   `<root>/.taskmaster/tasks/tasks.db` read-only (check the shape first with `scripts/tasks_db_schema.py
   --project-root <root>`) and list the rows whose `metadata` has a `files_to_modify` list with any path
   not in `metadata.files`, e.g. `SELECT id, metadata FROM tasks WHERE json_extract(metadata,
   '$.files_to_modify') IS NOT NULL`, then compare the two lists in Python.
2. For each such row: `update_task(id=<id>, project_root=<root>, metadata={"files": <files ∪
   files_to_modify, order kept, duplicates dropped>}, metadata_mode="merge")`. On 2026-11-10 that was
   dark-factory 2507 and reify 4344 and 4865, all `done`; dark-factory 2350 also carries the key but adds no
   path. The dead key stays on the rows (a merge cannot delete a key) and is inert.

Resolve gate 9927 listing the ids updated.

---

## 6. Reify: remove the `speculation_probe` stanza before S7 β deploys — FILED as reify task 9997

**Ruling.** R33 (Leo, sitting key S7-1) keeps deep chain and deletes the variable-depth speculation probe;
lead ruling OQ-CQ2 in `plans/review-all-s7-merge-lane-gates-placement-mechanisms-and-target-adapters-prd.md`
("reify's live yaml still carries `speculation_probe:` … a reify-side change removes that stanza before β
deploys. The lead files it in reify, and β carries it as `metadata.external_deps`").

**Done 2026-11-13.** Reify task **9997** (reify's own store, priority medium, `complexity='simple'`, files
`dark-factory-orchestrator.yaml`), "Remove the retired speculation_probe stanza from reify's
dark-factory-orchestrator.yaml before dark-factory deletes the probe", filed under planning mode and
committed to `pending`. Removing the block is safe at any time: it sets `probe_fraction: 0.0`, which is
also the code default (`orchestrator/src/orchestrator/config.py::SpeculationProbeConfig`), and the probe
has been off since 2026-07-23. Dark-factory task **9133** (S7 β, `pending`, unclaimed when written) now
carries `metadata.external_deps: ["reify:9997"]`, so it cannot dispatch before the stanza is gone.

**Watch.** Reify's scheduler is halted (reify's merge-lane pause), so 9997 may not dispatch on its own,
and 9133 waits on it: 9133 heads S7's chain (S7 α1, α2, γ, ε, ζ and S6's later merge-worker leaves follow
it). If 9133 becomes otherwise ready while 9997 is still `pending`, make the edit by hand as 9997's
description says (one yaml-only commit on reify main) and set 9997 `done` citing the sha. If 9997 is ever
cancelled, the scheduler blocks 9133 with an escalation (a cancelled external dependency does not satisfy):
check the stanza is gone, then remove the edge with `remove_dependency(id="9133", depends_on="reify:9997")`.

---

## 7. S13 ρ: fortnightly recon finding-category candidate check

**Owner:** gate 9928, "Review-all follow-up: file S13's fortnightly recon finding-category candidate check
(ρ, seed link of a recurring chain)". Depends on 9751.

**Ruling.** R67 (Leo, sitting key S13-2). His words: "Record rejections with adequate context for later
analysis + File a delayed recruring task to analyze the rejected findings to see whether our category set
is optimally complete or should be expanded, and escalate if any good candidates occur." Task 9737 (S13 ε)
records refused categories; task 9751 (S13 π) writes the check script; ρ is the seed link of a native
recurrence chain, filed once π is on main (R58). Plan:
`plans/review-all-s13-recon-sweeps-flag-identity-and-recon-report-state-prd.md` (§Decomposition plan π and ρ).

**Trigger.** 9751 is done; `git ls-files -s fused-memory/scripts/check_recon_category_candidates.py` shows
mode `100755`. If 9751 was cancelled: file ρ only if the script is on main by another task; otherwise ask
Leo whether R67's analysis still stands.

**Act.** File **ρ** — title `S13 ρ: Recon finding-category candidates: fortnightly check (seed link of a recurring chain)`
- `task_kind='deterministic'`, priority medium, `files: []`, no finding key. Depends on 9751 (already done;
  the edge records the order). Not a gate-6698 dependency: its links recur after the program.
- `metadata.before_done = {"kind": "predicate", "script": "fused-memory/scripts/check_recon_category_candidates.py",
  "args": ["--window-days", "14", "--min-refusals", "10", "--min-runs", "3", "--max-lost-share", "0.20",
  "--min-sample", "20"], "timeout_secs": 120, "recheck_secs": 3600}`.
- `metadata.milestone = {"mode": "dated", "at": "<filing time + 30 days, UTC ISO-8601>"}`.
- `metadata.recurrence = {"key": "recon-category-candidates", "interval_secs": 1209600}` (no `minted_from`
  on the seed link).
- `prd_path: "plans/review-all-s13-recon-sweeps-flag-identity-and-recon-report-state-prd.md"`,
  `prd_task_label: "ρ"`.
- Description: "Seed link of a recurring chain (R67; docs/task-authoring.md §6.1). Every 14 days after the
  previous link finishes, it runs fused-memory/scripts/check_recon_category_candidates.py over the
  recon-report store's rejected_findings table. The script fires (exit 1, printing each finding) when (a)
  one normalised refused label not passed as --ruled-label is refused at least 10 times across at least 3
  runs and more than half of its refusals are un-retried or retried as `other`, or (b) more than 20% of at
  least 20 refusals were never retried. Exit 0 closes the link done and mints the next. Exit 75 (no
  rejected_findings table yet: fused-memory not redeployed since S13 ε) re-arms the link hourly; 30
  re-arms file milestone_check_stalled. A firing link files a born-at-L2 milestone_check_failed carrying
  the output and blocks the chain until resolved. To resolve: after ruling on a candidate, either add
  `--ruled-label <label>` to this link's before_done.args and resume, or file the follow-up (a new
  FindingCategory value, a prompt synonym) and resume once the condition has cleared; `abandon` ends the
  chain. This is the project's first live recurring chain, so its first runs also test the mechanism."

Resolve gate 9928 naming ρ's id; commit the stamped S13 sidecar if `commit_planning` changed it.

---

## 8. S2 ξ: three-day check for unidentified escalation authority acts

**Owner:** gate 9929, "Review-all follow-up: file S2's three-day check for unidentified escalation
authority acts (ξ), then the gate for the denial (ο)". Depends on 9817 and 9843. Priority high.

**Ruling.** R63 (Leo, sitting key S2-2, widened by residual R-6 = B): a caller that sends no
`X-Escalation-Identity` header loses every authority-bearing escalation action at every level. His words:
"(deploy the MCP configuration to all existing projects before denying, make the denial self-explaining via
the error message, especially for the no identity header case). the headless channel should not be
supported." Order: (1) ε 9760 parses identity, stamps the channel and warns; (2) μ 9817 and ν 9843 deploy
identities everywhere; (3) ξ checks for three days that no authority act came from an unidentified caller
(Leo's "a few days" read as three, confirmed by the lead); (4) ο refuses. R91 sets who keeps authority.
Plan: `plans/review-all-s2-typed-escalation-record-facts-prd.md` (§Sketch μ, ν, "ξ and ο"; §Not filed).

**Trigger.** 9817 and 9843 are done, AND every project's orchestrator and fused-memory run code that
contains 9817's merge, so the window never counts an agent spawned by older code. Check with
`scripts/orchestrator-watchdog.py --report` or each unit's `ActiveEnterTimestamp` against 9817's merge
time; if any is older, wait and resume later. If 9817 or 9843 was cancelled: the deploy did not happen as
planned; file neither ξ nor ο; ask Leo.

**Act.**

Step 1 — the predicate must cover every authority-bearing act. The plan as written has ξ count only
records whose `resolved_via` is `unidentified` (resolve, park, promote). R-6 = B widened the denial to
every act in `escalation/src/escalation/authority.py::AUTHORITY_BEARING`: every `resolve_issue` action
(park included), `promote_to_l2`, `stamp_triage`, `amend_escalation`, `claim_policy_class` and
`declare_pin`. ξ could pass while one of the other four still arrives header-less, so read
`scripts/check_unidentified_escalation_actions.py` (written by 9760). If it already counts every act by an
`UNIDENTIFIED` caller from durable data, go to step 2. If not, file and land first:

- **Widening task** — title `S2 ξ prerequisite: the unidentified-action check counts every authority-bearing escalation act`
  - `task_kind='normal'`, priority high, `source`, `x_finding_run`, `prd_path` the S2 plan, no label, no key.
  - Files (re-measure at filing; S3 ζ 9200 may have moved the tools into `filing_tools.py`,
    `resolve_tools.py` and `read_tools.py`): the module(s) that perform each `AUTHORITY_BEARING` act,
    `escalation/src/escalation/authority.py`, `scripts/check_unidentified_escalation_actions.py`,
    `scripts/tests/test_check_unidentified_escalation_actions.py`, and the tests of the acts touched.
  - Work: every authority-bearing call made by an `UNIDENTIFIED` caller leaves a durable, structured fact
    the script can count (tool, action, level, time; on the escalation record it acted on, or, for
    `declare_pin` and `claim_policy_class`, in the store that act writes), and the script counts every such
    fact in its window across every queue it already reads, listing them per queue and exiting 1 on any.
    Not by parsing log lines (structured data instead of meaningful strings). Today's `resolved_via` stamps
    keep their meaning.
  - Signal: an `UNIDENTIFIED` HTTP `stamp_triage`, `amend_escalation`, `claim_policy_class` and
    `declare_pin` each make the script exit 1 and list the act; a run with only identified callers exits 0.
  - Depends on 9760. File ξ (step 2) depending on it as well.

  The program lead may instead amend task 9760 (still `pending` and unclaimed on 2026-11-13) to write the
  script this way from the start; then step 1 only checks it.

Step 2 — file **ξ** — title `S2 ξ: Check: no escalation authority act came from an unidentified caller for three days after the identity deploy`
- `task_kind='deterministic'`, priority **high**, `files: []`, no finding key.
- `metadata.before_done = {"kind": "predicate", "script": "scripts/check_unidentified_escalation_actions.py",
  "args": ["--window-secs", "259200", "--queue-dir", "/home/leo/src/dark-factory/data/reconciliation/escalations"],
  "env": {"DASHBOARD_KNOWN_PROJECT_ROOTS": "<the host's value>"}, "timeout_secs": 120}`. The reconciliation
  queue lies under no project's `data/escalations`, so it is passed explicitly. Take the
  `DASHBOARD_KNOWN_PROJECT_ROOTS` value the dashboard service runs with, and confirm by hand that the
  script then lists every project's queue (`shared/src/shared/project_layout.py::known_project_roots`).
- `metadata.milestone = {"mode": "delayed", "after_secs": 259200}`.
- `prd_path: "plans/review-all-s2-typed-escalation-record-facts-prd.md"`, `prd_task_label: "ξ"` (the
  sidecar has no ξ block, so `commit_planning` reports it unlabelled: expected).
- Depends on 9817, 9843 and, if filed, the widening task.
- Description: "Deterministic predicate check (R63, Leo's sitting key S2-2 and residual R-6 = B). Three
  days after the identity deploy (S2 μ 9817, ν 9843) and after every project's orchestrator and
  fused-memory restarted onto μ, scripts/check_unidentified_escalation_actions.py must find no
  authority-bearing escalation act (every resolve action incl. park, promote_to_l2, stamp_triage,
  amend_escalation, claim_policy_class, declare_pin) by a caller with no X-Escalation-Identity header, in
  any project's queue or the reconciliation queue. Exit 0: done, which releases the denial ο. Non-zero: a
  born-at-L2 milestone_check_failed names the queues and records; a human finds the client the deploy
  missed, fixes its configuration, and resumes (the check re-runs)."
- After `commit_planning`: `add_dependency(id="6698", depends_on="<ξ id>")`. ξ joins the program's
  capstone gate 6698 (R63, program status), as every high leaf of the program does.

Step 3 — file the gate for ο as §9 says. Resolve gate 9929 naming ξ's id, the widening task's id if any,
and the ο gate's id.

---

## 9. S2 ο: refuse escalation authority to a caller that sends no identity header

**Owner:** not yet in the store. Its trigger is "S2 ξ is done", and ξ does not exist until gate 9929's
act files it; gate 9929's step 3 files the gate below. Ruling and plan as §8; the denial closes
`fk-ece88b7661b7`.

**The ο gate** — title `Review-all follow-up: file S2's escalation identity denial (ο) once the three-day check passes`
- Pure human gate: deterministic, `always_escalates: true`, no `before_done`, `files: []`, priority high,
  `source`, `x_finding_run`. No `x_finding_key` on the gate: ο carries `fk-ece88b7661b7`.
- Depends on ξ, and on `reify:9996` added with `add_dependency(id=<gate>, depends_on="reify:9996")`: reify
  task 9996 makes `reify/scripts/docs-truth-sweep.py` send an identity header on its `promote_to_l2`
  call, and the denial must not land before it does (the lead's reading of R63's "carries it as a
  cross-project dependency if order matters": the denial is where order matters).
- Description: trigger (ξ done and reify 9996 done; if 9996 was cancelled the scheduler blocks this gate
  with an escalation: check by hand that the docs-truth sweep sends the header, or that nothing on reify
  calls an authority-bearing tool header-less, before going on); the act below; a pointer to this section.

**Act when the ο gate fires.** Re-measure the files (S3 ζ 9200 may have moved the escalation tools), then
file **ο** — title `S2 ο: Refuse escalation authority to a caller that sends no identity header, with a self-explaining error`
- `task_kind='normal'`, priority **high**, `x_finding_key: ["fk-ece88b7661b7"]` (it closes the key),
  `prd_path` the S2 plan, `prd_task_label: "ο"`, `source`, `x_finding_run`.
- Depends on ξ and on `reify:9996` (qualified `add_dependency`).
- Files (measured at `e59840cdaf`; re-measure): `escalation/src/escalation/authority.py`, the escalation
  tool module(s), `escalation/tests/test_capability_guard_http.py`,
  `escalation/tests/test_status_authority_gate.py`, `docs/escalation-standing-policy.md`,
  `skills/recon-escalation-watcher/SKILL.md`.
- Work: every `UNIDENTIFIED` caller's `AUTHORITY_BEARING` call returns `{"error": <ε's message>, "code":
  "identity_required"}` and changes nothing, at every level, a header-less `promote_to_l2` included (this
  supersedes lead ruling S2/OQ-9's "no header → allowed"); filings and reads stay open; the five esc-2087-2
  test pins change (`test_capability_guard_http.py` 2, `test_status_authority_gate.py` 3); `authority.py`'s
  module docstring, `docs/escalation-standing-policy.md`'s esc-2087-2 passage and the recon watcher skill's
  paragraph saying a header-less `promote_to_l2` "actually succeeds" are rewritten; the change records that
  it ends the 2026-08-20 human-channel guarantee (esc-2087-2).
- Signal: boundary row 28 of the plan (every authority-bearing call over real HTTP with no headers, at
  levels 0, 1 and 2, returns `identity_required` with the pinned text and leaves the record unchanged;
  `escalate_info` and `get_escalation` succeed); after each project's orchestrator restarts on ο, a
  header-less `resolve_issue` there answers `identity_required`. The escalation server runs inside each
  project's orchestrator, so ο reaches a project at that orchestrator's restart.
- After `commit_planning`: `add_dependency(id="6698", depends_on="<ο id>")` — ο joins gate 6698.

---

## 10. Amend to task 6254 and note to task 4809

**Owner:** gate 9930, "Review-all follow-up: apply the dashboard plan's amend to task 6254 and the server
plan's note to task 4809 once their branches land or are dropped". Depends on 6254 and 4809.

**Source.** The wave-3 clerk pass of 2026-11-13 did not write these, because each task's branch carried
unmerged work (6254: 16 commits on `task/6254`; 4809: 19 on `task/4809`). Rulings: R76 (Leo, sitting key
S22-3: source-text dashboard tests become jsdom render tests); the S15 plan's G4 note on S15 β0 9825
(`plans/review-all-s15-fused-memory-mcp-server-boundary-prd.md`).

**Trigger.** Both tasks are terminal. If a branch is dropped while its task stays open, whoever drops it
applies that task's text at once rather than waiting for the gate.

**Act, per task.** Still open with the branch gone: apply the text verbatim (an amend is appended to the
description as a dated "AMENDED 2026-MM-DD (review-all S22):" paragraph, `docs/task-authoring.md` §9; a
note is appended to `details` with `append=True`). Done: check whether the landed change meets the text's
substance; if not, file a small follow-up task carrying it. Cancelled: follow `superseded_by` and apply the
text to the absorbing task if its substance still applies.

- **6254** (amend, from the S22 plan, `plans/review-all-s22-dashboard-readers-and-aggregation-prd.md`):
  > Test this change through the jsdom harness (dashboard/tests/js/dom_harness.mjs), in dashboard/tests/js/render_memory_tabs.test.mjs and render_overview_orchestrators.test.mjs, which S22 π2, π4 write (plans/review-all-s22-dashboard-readers-and-aggregation-prd.md); add no assertion that reads frontend source text (S22 ο's guard refuses one). dashboard/tests/test_tab_memory.py and dashboard/tests/test_tab_overview.py are deleted by that conversion, so replace them in metadata.files with the render-test file.

  (π2 is task 9881, π4 is 9885; 6254 already depends on both.) The `metadata.files` change: drop
  `dashboard/tests/test_tab_memory.py` and `dashboard/tests/test_tab_overview.py`, add
  `dashboard/tests/js/render_memory_tabs.test.mjs` and `dashboard/tests/js/render_overview_orchestrators.test.mjs`.
- **4809** (note, from the S15 plan):
  > S15 β0 (task 9825) now depends on this task (G4: same dispatch-boundary region). β0 makes every create_mcp_server build install the markup guard, and its row 21b requires a build without a project registry to file nothing; if this task adds an unconditional last-resort preservation floor, make it injectable so registry-less test builds stay hermetic.

  If 4809 landed: check the floor it added is injectable; if not, add the text as a note on 9825 instead.

Resolve gate 9930 saying what was applied where.

---

## 11. Actions held on tasks 5258, 5912, 6695 and 3498

**Owner:** gate 9931, "Review-all follow-up: settle the filing and reconciliation plans' held actions on
tasks 5258, 5912, 6695 and 3498 once their branches land or are dropped". Depends on all four.

**Source.** The wave-3 clerk pass of 2026-11-13. Each task carried unmerged work or a live claimant, so
the clerk did not act: 5258 (20 commits on `task/5258`, tip `707f613ab1`); 5912 (5 commits, tip
`0a3bee697a`, live worktree); 6695 (`in-progress`, live claimant, 7-commit branch); 3498 (7 commits and a
stale claimant from 2026-08-19). Plans: S1 `plans/review-all-s1-one-escalation-filing-entry-point-prd.md`,
S12 `plans/review-all-s12-reconciliation-run-lifecycle-and-stage-interface-prd.md`, S13 (§7's plan).

**Trigger.** All four are terminal. A branch dropped while its task stays open: apply that task's action at once.

**Act.**
- **5258** (S1 amend). Still open: append verbatim
  > Unifying the three markup residue/storm filers: build EscalationDraft values and file through escalation/filing.py::EscalationFiler (S1 C1); do not construct Escalation or call queue.submit (invariant escalation-filing-one-door). Edges 5258 -> 9688, 9769, 9810 already wired. NOTE: 5258's branch task/5258 (tip 707f613ab1, 25 files) predates S1; its markup_records.py approach must be rebased onto the filer when it resumes.

  Done: check its filers go through `escalation/filing.py::EscalationFiler`; if not, file a follow-up with
  the text. Cancelled: nothing.
- **5912** (S12 cancel, held). The planned action was `cancel` into S12 α 9729 with
  `x_acceptance_reason: "subsumed by S12 α/β1: RunError.from_exception is the one run-failure record constructor and the one driver's ladder builds every error record"`.
  Landed (done): nothing to cancel; append a note to 9729's details that 5912's cancellation-record
  routing (`run_full_cycle`'s cancellation and both drivers' `AllAccountsCapped` error records through
  `_run_failure_record`) is on main and 9729 builds on it. Still open with its branch dropped: cancel it
  into 9729 with that reason (`set_task_status(id="5912", status="cancelled", …)` with the reason in
  `metadata.x_acceptance_reason`, and `superseded_by: [9729]`). Already cancelled: nothing.
- **6695** (S12 stamp). Whatever its status: `update_task(id="6695", metadata={"x_finding_key":
  ["fk-0485729893fe"], "x_finding_run": "review-all-dark_factory-20261008"}, metadata_mode="merge")` (if it
  already has an `x_finding_key` list, send the union). 6695 carries the residual of `fk-0485729893fe`
  exactly; S12 β1 9757 depends on it. If it was cancelled, stamp the key on the task that absorbed it
  (`superseded_by`), or on 9757 if none.
- **3498** (S13 note). Still open: append to details verbatim
  > G4: S13 κ2 task 9745 deletes the in-cycle Mem0 flag_for_stage2 drain and κ3 task 9746 drops the script's flag_for_stage2 census probe; decide against the post-κ3 sweep_orphan_flag_markers.py.

  Done: check its decision still holds once 9745 and 9746 land; if it does not, file a follow-up.

Resolve gate 9931 listing the action taken on each.

---

## 12. Amends held on tasks 8143, 8262, 7008, 3678, 4875; the decision on 5700

**Owner:** gate 9932, "Review-all follow-up: apply the config plan's amends to tasks 8143, 8262, 7008,
3678 and 4875 and settle task 5700 once their branches land or are dropped". Depends on all six.

**Source.** The wave-3 clerk pass of 2026-11-13. On that day each branch carried unmerged commits (8143:
33; 8262: 1; 7008: 6; 3678: 14; 4875: 15; 5700: 3), so the clerk reverted the amends it had written and did
not cancel 5700. Plans: S16 `plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md`
(R15 as read by R92: a setting's default has one home, the config field; S16 α 9691 deletes the restating
`defaults.yaml` leaves; S16 β1 9694 and β2 9693 replace the `RELOADABLE_FIELDS` lists with `Reload.GREEN`
on the field); S24 `plans/review-all-s24-fleet-watchdog-and-redeploy-coordination-prd.md`.

**Trigger.** All six are terminal. A branch dropped while its task stays open: apply that task's text at once.

**Act, per task.** Still open with the branch gone: append the amend verbatim to the description (dated
"AMENDED 2026-MM-DD (review-all S16):" paragraph). Done: check the landed change against the text; if it
diverges (a new `defaults.yaml` leaf, a `RELOADABLE_FIELDS` entry after β1/β2 landed, the old check
pattern), file a small follow-up. Cancelled: drop the text unless the absorbing task needs it.

- **8143**: Delivered check on cancel_children.enforce becomes `<old pattern>|<marker S16 β2 (9693) writes>` over fused-memory config/reload.py and config/schema.py; β2 states the marker spelling in its commit.
  (This changes 8143's `metadata.delivered_checks` entry for `cancel_children.enforce`: widen its pattern
  with the marker named in 9693's commit message and add `config/schema.py` to its paths; then mirror it
  in 8143's capability-manifest sidecar.)
- **8262**: Change the defaults on the config.py field defaults only; do not edit defaults.yaml (S16 α 9691 deletes its restating leaves). Update its test to read field defaults.
- **7008**: Flip merge_verify_breadth on the config.py field default only; S16 α (9691) deletes the restating defaults.yaml leaf; put the paired-policy prose in the field description.
- **3678**: declare a new config field's default on the config.py Field only, never as a new defaults.yaml leaf (S16 α 9691 makes a literal leaf fail the config-default-one-home guard); if S16 β1 9694 has landed, mark its hot-reload tier with Reload.GREEN on the field instead of a RELOADABLE_FIELDS entry.
- **4875**: if S16 β2 9693 has landed (fused_memory/config/reload.py deleted), declare the new leaf's tier with Reload.GREEN on its schema.py field instead of a RELOADABLE_FIELDS entry.
- **5700** (S24 cancel, held). Planned: cancel into S24 β 9786 with
  `x_acceptance_reason: "subsumed by S24 leaf β (9786): Systemd.revive returns a typed ReviveOutcome naming any timed-out phase, and callers log the outcome instead of an unconditional restart issued, plans/review-all-s24-fleet-watchdog-and-redeploy-coordination-prd.md"`.
  Landed (done): nothing to cancel; note on 9786's details that 5700's qualified "restart issued" log is on
  main and β replaces it with the typed outcome. Still open with the branch dropped: cancel it into 9786
  with that reason and `superseded_by: [9786]`.

Resolve gate 9932 listing the action taken on each.

---

## 13. Already owned by filed tasks (no gate)

- **S24's flip of fused-memory's restart interval from 4 h to 1 h** (R78; Leo's R-4 answer of 2026-11-13:
  "R-4 C using a drain restart that allows recon runs to finish, and until the shutdown fix (then A)").
  Carried by task **9861** (S24 θ), which depends on gate 9521 (the operator reinstall that puts task 9493's
  single-stop-signal fix into effect), 9764 and 9795. 9493 is done; 9521 was `blocked` at level 2 on
  2026-11-13, waiting on Leo. Nothing more to file.
- **S16's pre-authorised yaml sweep, then the refusal of old setting names** (R70; Leo's R-5 answer: "a L2
  task for the watcher to /spawn a pre-authorised /team session to update all the project yamls before the
  deploy"). Carried by task **9718** (S16 σ, the operator-sweep gate with a 72-hour delayed milestone,
  depending on 9710, 9711, 9712 and 6754) and task **9719** (S16 ξ, the refusal, depending on 9718).
  Nothing more to file.

## 14. Dropped

- **Edge 5153 → 9216** (S8 residual). Leo ruled on 2026-11-11 (R59) that 5153 lands first and the
  slot-registry task 9216 depends on it; that edge was wired the same day. Nothing owed.

## 15. Not covered here

The program doc's "Open — lead" list holds items outside this file's scope, still owned by the lead:
S17's xref on the scheduler's fused-memory task-store client partition (needs a follow-up task or an
acceptance); the moves scan's two residuals (task 9748's sibling `project-policies-config` pattern; task
9790's check, vacuous until task 9768's move lands); the suggestion to drop θ1 9798's stale orphan-substring
clause. Gate 6698 itself and its dependencies are never written by a follow-up of this file, except the
two `add_dependency(6698, …)` calls §8 and §9 name for ξ and ο.
