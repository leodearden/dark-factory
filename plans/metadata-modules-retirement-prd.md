# Metadata.modules Retirement PRD

**Status:** active — retirement ratified by Leo 2026-08-20 after a 4-agent
census + in-session discussion (session `investigate-df-3582900`).
Approach: bare B (G5 prompt resolved: no new seam contract is created —
the work is removals plus one boundary guard on an existing precedent;
the `metadata.files` writer charter stays owned by DF 3260).

## Goal

Retire the task-metadata key `metadata.modules` **for new writes** while
preserving all historical carriers. After this PRD lands, an operator
observes:

- A `submit_task` for a **new** task whose metadata carries `modules` is
  rejected at the fused-memory boundary with a structured error naming
  `metadata.files` as the field to use; the same submission with `files`
  succeeds; `update_task` on an existing carrier still succeeds.
- Live (non-terminal) task records across the seven orchestrator corpora
  no longer carry `metadata.modules`: a migration `--dry-run` reports
  zero pending actions, with residual carriers only in the statuses the
  migration deliberately skips (immutable history).
- Skill-filed tasks (`/prd` decompose, `/unblock`, `/orchestrate`,
  watcher triage, `/review` phase 3) declare sparse `metadata.files`,
  never `modules`.
- `docs/task-authoring.md` and the Tier-A blessed-key list describe
  `modules` as retired-historical; the path-scope guard attests from
  `files`/`files_to_modify` only.

Terminal and deferred records keep the key forever (immutable to
`update_task`; ruling 2026-08-20: the write journal is the recovery path
for original-scope declarations — no done-time stamp is added).

## Background — what `metadata.modules` is today

All measured 2026-08-20 at HEAD `eba215060c` unless marked inherited.

- **The rename already happened.** `fabfa367f5` (2026-05-06) canonicalized
  scope on `metadata.files`; `Scheduler._get_modules`
  (`scheduler.py:8773-8810`) derives locks from `files` only and never
  reads `modules`. The paired migration
  (`scripts/migrate_metadata_modules_to_files.py`) ran **once**, the same
  day (53 copy + 141 drop across 208 then-active tasks), skipping
  terminal tasks by design, and never again.
- **The field is still actively minted.** 5,223 of 12,370 task records
  across 7 corpora carry it (475 on live tasks; DF 1,738 / reify 2,857),
  ~96.6% divergent from `files` as string sets. Writers: six skill
  templates, the auto-eval redo copier (`harness.py:9208`), and ad-hoc
  agent submissions (~23 since 08-01 with no template source). DF 4524 —
  filed **today** by the module-tagger PRD's own decompose — carries it.
- **A modules-only filing is silently under-locked.** Nothing derives
  `files` from `modules` on any path; with the module tagger retired
  (`plans/module-tagger-retirement-prd.md`, same day), a files-less
  filing runs on the synthetic `task-<id>` lock until plan reconcile.
- **Live readers exist** (the "one reader" folklore was wrong):
  `task_interceptor.py:1666` (path-scope-guard attestation — a local
  `modules` entry suppresses the cross-repo advisory, a foreign entry
  vetoes attestation; advisory-only either way), `cli.py:338` (task-list
  display), `scripts/mint_hard_v2_fixtures.py:496` (eval-fixture minting
  reads `modules` deliberately: on done tasks `files` is the merge diff —
  the answer — via `_reconcile_metadata_files_for_done`), and the queued
  consumer task 3659 (memory-briefing-and-fusion PRD D2).
- **Why retire rather than tidy:** the field misled the esc-6068-4
  investigation; it underwrote a functionally inert L1 repair recipe
  (`scope_violation` → extend `modules` — no lock effect since May; DF
  3465, amended 2026-08-20, owns the fix); a modules-only task presents
  exactly like a wiped-`files` specimen to the writer-ownership map,
  misrouting investigations; and `docs/task-authoring.md` §3.2 still
  describes it as a co-equal scope key while dispatch ignores it.

## Sketch of approach

Three leaves plus decompose-session coordination actions:

1. **α — skills sweep**: the template sites that mint `modules` switch to
   sparse, honest `metadata.files` declarations (deliberate
   under-declaration remains policy — defer discovery to the architect).
2. **β — migration fix + fleet run**: fix the migration script's copy
   branch (it now collides with the directory-lock reject gate), test the
   branch logic, and **run it in the same task** across all seven corpora
   (never a standalone op task — the δ/3691 "`--apply` never run"
   precedent, and `execution_class` op tasks silently convert to pure
   gates).
3. **ε — boundary retirement**: drop `modules` from the path-guard
   attestation tuple, reject new-task submissions that carry it, annotate
   the blessing retired-historical, correct the docs.

No key is un-blessed, ever. No terminal record is touched.

## Resolved design decisions

1. **Retire for new writes; keep all historical data** (Leo,
   2026-08-20). On done tasks `files` is the merge diff, so `modules` is
   the only in-record original-scope trace — preserved automatically by
   terminal-task immutability plus the migration's skip set. The write
   journal covers future recovery needs; no done-time declaration stamp.
2. **The Tier-A blessing stays forever, annotated** "retired 2026-08 —
   historical carrier, no live writer; rejected on new submissions"
   (`shared/src/shared/task_metadata.py:872`). Un-blessing rejects
   nothing (warn-only census), breaks two tests, and manufactures
   `unknown_key` noise from 1,553+ immutable carriers — the
   `origin_finding_id` precedent, re-affirmed by the tagger PRD's
   decision 5 the same day.
3. **Reject at submit** (Leo, 2026-08-20; chosen over warn-only and
   docs-only). A NEW `submit_task` carrying `metadata.modules` gets a
   structured error with a use-`files` hint (INV-1 house pattern;
   sibling of `lock_charter_error` / 4524's `simple_task_files_error`).
   The update path stays tolerant — existing carriers are re-written
   whole by routine amendments. Sequenced strictly after the writers are
   gone (α templates, DF 4507's copier fix) so nothing legitimate
   bounces.
4. **3659 / briefing-PRD D2 re-points to `files`-derived area terms**
   (Leo, 2026-08-20). Retrieval needs representativeness, not
   completeness, so deliberately sparse `files` phrases the conventions
   query about as well as coarse `modules` did; the 5-case relevance
   probe is re-run inside 3659 itself; an explicit `metadata.area` key is
   introduced only if that probe degrades (out of scope here).
5. **The migration run is folded into β**, executed by β's implementer
   from the worktree against the live fused-memory HTTP endpoint, with
   before/after counts in the PR and a `--dry-run` re-run proving zero
   pending actions. The script is idempotent, per-task resumable, and
   write-attributed (`migrate-metadata`).
6. **The attestation narrowing is accepted** (Leo, 2026-08-20). With α+β
   landed first, no live task carries `modules`, so dropping it from the
   tuple flips ~nothing; any residue files into the `esc-task-path-guard*`
   advisory family that DF 3465 teaches the L1 watcher to absorb.
7. **The harness copier line rides DF 4507**, whose own "(a)" clause
   already flags it — not a leaf here. The decompose session amends 4507
   to make that clause in-scope-there and modules-dropping (below).
8. **Coordination is performed by the decompose session, not filed as
   tasks** (tagger-PRD pattern): (i) amend task 3659's description — D2
   phrasing source becomes "area terms derived from `metadata.files`
   (path components; fallback task title)", re-probe folded in; (ii) edit
   `docs/prds/memory-briefing-and-fusion.md` D2 (:31, :77, :91 region) to
   match, committed docs-only; (iii) append to DF 4507's details: under
   this PRD, its "(a)" clause is in-scope in 4507 — at the redo-sibling
   creation, **delete** the `'modules'` copy (`harness.py:9208`) and
   route the `files` copy through `sanitize_files_for_persist`, rather
   than "fix or file separately"; (iv) verify DF 3465's 2026-08-20
   amendment is present (it is — done in-session). All performed before
   `commit_planning` flips the batch.
9. **Retained readers, deliberately:** `cli.py:338` (renders historical
   records; empty suffix on new tasks is fine) and
   `mint_hard_v2_fixtures.py:496` (must keep reading `modules` on the
   historical corpus — mining `files` off done tasks would bake the
   merge diff, i.e. the answer, into eval fixtures).

## Pre-conditions for activating

- fused-memory MCP running for decompose filing (standard).
- Decompose coordination actions 8(i)–(iv) complete before the batch
  flips `deferred`→`pending` — this guarantees the 3659/D2 amendment
  strictly precedes β's migration run.

## Cross-PRD / cross-task relationships

| Other | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/memory-briefing-and-fusion.md` (task 3659, D2) | that PRD consumes `modules` today | conventions/task-context query phrasing source | this PRD re-points it (coordination 8(i)/(ii)); 3659 owns the re-probe | amend at decompose |
| `plans/module-tagger-retirement-prd.md` | alignment | post-tagger, files-less filings stay on the synthetic lock ⇒ α must write `files`, and its δ (DF 4524) gates simple fileless submissions; blessing-annotation shape shared (its decision 5) | each PRD its own leaves | active |
| DF 4507 | this PRD needs its "(a)" clause landed | `harness.py:9208` redo-sibling `modules` copy deletion | 4507 (amended per 8(iii)) | ε depends on 4507 |
| DF 3465 (amended 2026-08-20) | owns the `scope_violation` recipe rewrite in both watcher skills | `escalation-watcher-auto/SKILL.md:365-382`, `escalation-watcher/SKILL.md:981-988` | 3465 | α excludes those lines; same-file lock serialization on `escalation-watcher/SKILL.md` |
| DF 3090 / DF 3260 | referenced, unaffected | structural scope auto-expand / `files` writer charter | 3090 / 3260 | — |
| `plans/capability-delivered-checks-prd.md` | consumes | manifest YAML sidecar | that PRD's stamper | standard |

Lock-contention note (not a semantic dep): ε shares
`task_interceptor.py` and `docs/task-authoring.md` with DF 4524
(pending/high) — both add sibling boundary guards; scope ε's edits
narrowly and reuse, don't copy, the error-construction pattern (INV-5).

## Decomposition plan

- **α — Sweep the `metadata.modules` template sites in skills/ to sparse
  `metadata.files`** · high · `task_kind=normal` · no deps.
  Sites at `eba215060c` (re-grep before editing; anchors drift):
  `skills/prd/references/decompose-mode.md:93`,
  `skills/orchestrate/SKILL.md:157`, `skills/unblock/SKILL.md:269`,
  `skills/escalation-watcher/SKILL.md:1061`,
  `skills/review/references/phase3-triage.md:97`; then grep the whole
  `skills/` tree for further `"modules"` keys inside
  `submit_task`/`add_task`/`update_task` templates. **Exclude** the
  `scope_violation` recipe lines
  (`escalation-watcher-auto/SKILL.md:365-382`,
  `escalation-watcher/SKILL.md:981-988`) — DF 3465 owns those.
  `phase1-integration.md:105` is a review-JSON blob, not task metadata —
  leave it. Replace each template's `"modules"` with a sparse `"files"`
  declaration plus a one-line note that under-declaring is fine (the
  architect widens at plan time).
  **Observable signal:** `rg '"modules"' skills/` returns only the
  3465-owned recipe lines (until 3465 lands) and non-task-metadata hits;
  the next task filed via a swept skill carries `metadata.files` and no
  `metadata.modules`, observable in its record via `get_task`.
- **β — Fix the migration's copy branch and run it across the seven
  corpora** · high · `task_kind=normal` · depends on: α.
  Fix `scripts/migrate_metadata_modules_to_files.py`: the copy branch
  (`files` empty ← `modules` verbatim) now collides with the update-path
  directory-lock gate (`_reject_directory_locks_in_update_metadata`,
  `task_interceptor.py:4450` region) for directory-shaped entries — the
  dominant historical shape. Sanitize with the same predicate the charter
  uses (`sanitize_files_for_persist` shape); an all-directory result
  leaves `files` empty (defer-to-architect). Add branch-logic tests to
  `tests/scripts/test_migrate_metadata_modules_to_files.py` (today it
  covers only the HTTP client handshake). Then **run it in this task**
  from the worktree against live fused-memory (`--server-url` default
  `127.0.0.1:8002`) for: dark-factory, reify, autopilot-video, know-live,
  pump-web-ui, solar-challenge, solar-challenge-platform. Paste
  per-project before/after carrier counts into the PR.
  **Observable signal:** a `--dry-run` re-run after the live run reports
  zero pending actions for every project; remaining carriers are only in
  the migration's deliberate skip statuses, with counts stated in the PR.
  If the sandbox blocks the HTTP run, **escalate** — the run is
  load-bearing; do not mark done on the code fix alone (the δ/3691
  failure shape).
- **ε — Drop `modules` from the path-guard tuple, reject it on new
  submissions, annotate it retired** · high · `task_kind=normal` ·
  depends on: α, β, DF 4507.
  (1) `task_interceptor.py:1666` tuple → `('files', 'files_to_modify')`
  + docstring; `path_scope_guard.py` prose comments; update the
  parametrized `modules` cases in
  `fused-memory/tests/test_task_interceptor.py` (:8596, :9032-9125) and
  `fused-memory/tests/test_path_scope_guard.py:900`.
  (2) Submit-boundary rejection: a NEW `submit_task` whose metadata
  carries `modules` returns a structured `retired_key_modules_error`
  with a use-`files` hint. `update_task` and planning-batch commits stay
  tolerant. Reuse the `lock_charter_error` construction pattern —
  coordinate with DF 4524's sibling guard, don't copy (INV-5).
  (3) Docs: `docs/task-authoring.md` §3.2 attestation prose (three keys →
  two) and the Tier-A entry annotated retired-historical;
  `shared/src/shared/task_metadata.py:872` comment.
  (4) Idempotence backstop: re-run the migration `--dry-run`; if
  stragglers appeared since β, run it live once more (idempotent).
  **Observable signal:** a live `submit_task` carrying `metadata.modules`
  returns `retired_key_modules_error` through the MCP response (G6
  branch 4 — the rejection is exercised, not assumed); the same
  submission with `files` succeeds; an `update_task` re-write of an
  existing carrier succeeds; the task-authoring census lists `modules`
  as retired.

**G7 (advisory walk, author mode):** ε's contract is machine-checked at
the boundary, not prose (`contracts-machine-checked` ✓ — this is the
walk's own finding, promoted to decision 3); the rejection is a loud
per-write error, no fail-soft path to storm (`storm-escape-required`
n/a); β re-reads each task through the live MCP before writing, is
idempotent, and both β and ε re-verify with `--dry-run` after acting
(`corroborate-before-acting` ✓ — the race window with concurrent
metadata writers is closed by the ε backstop); α edits six pre-existing
independent prose sites and adds no new lock-step duplication
(`no-lockstep-duplication` — extraction judged over-engineering for
prose templates, accepted); no held states, loops, or status transitions
introduced (INV-6/7/8 n/a). No unresolved hits; decompose re-walks the
full list as the blocking check.

## Out of scope

- **Scrubbing terminal/deferred carriers** — immutable, and deliberately
  preserved as the only in-record original-scope trace (decision 1).
- **`metadata.area`** — only if 3659's re-probe shows `files`-derived
  phrasing degrades retrieval; that PRD's call, not this one's.
- **`cli.py:338` and `mint_hard_v2_fixtures.py:496`** — retained readers
  (decision 9).
- **The `scope_violation` recipe rewrite** (DF 3465), **structural scope
  auto-expand** (DF 3090), **the `files` single-writer charter** (DF
  3260).
- **reify scratch debris** — `.orchestrator-scratch/dep-audit/audit.py`
  reads `modules` from a `tasks.json` that no longer exists; dead
  tooling, not a consumer.

## Open questions (tactical)

1. **Which statuses the β run leaves untouched** — the script's skip set
   is `done`/`cancelled`/`deferred`; `merge-deferred` is processed.
   Enumerate the residual carrier counts by status in β's PR.
2. **Whether ε's rejection ships a deliberate-bypass flag** (an
   `allow_retired_modules` metadata key for archaeology writes).
   Suggested: no — quoting the key in prose never trips the guard, only
   the metadata key itself does. Decide during ε.
3. **Worktree→`127.0.0.1:8002` reachability under the sandbox** for β's
   run — landlock is filesystem-scoped so HTTP should pass; if not, β
   escalates and the run falls to the operator (stated in β's signal).

## Decompose instruction

The decomposing session must **not** stamp `metadata.modules` on this
batch's own tasks — the live decompose template still carries it until α
lands; declare `metadata.files` only. Perform coordination actions
8(i)–(iv) before `commit_planning`.
