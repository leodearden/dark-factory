# Client feed: a per-project, client-safe evidence feed for the client portal

**Status:** authored 2026-10-14 (spawned design session; `/team` + `/prd` author mode; the design was
attacked by an opus critic seat before this PRD was written, see the design note §13).
**Decomposed 2026-10-14:** α 6966, θ 6967, β1 6968, β2 6969, γ 6970, δ 6971, ε 6972, ζ 6973, η 6974;
bindings and the G7 walk in `client-feed-prd.capability-manifest.md` and its stamped YAML sidecar.
**Revised 2026-10-23** (gate 7062, esc-7062-1): the client defect bridge, §14 — a `defect_reported`
inbox event, a narrowly privileged filer outside §8's boundary, and the `defect_filed` feed event
(FEED_VERSION 1.1); σ1a, σ1b, σ2–σ5, with portal tasks P1a, P1b, P2 and P3. Supersedes task 7005.
**Type:** greenfield. A new workspace member, `client-feed/`, that publishes one project's build
evidence to the LDE client portal through file contracts, one orchestrator event the factory does not
record today, and a minimal acceptance-criteria data format.
**Approach:** B+H. G5 applies: the seam crosses two repositories (this one and the portal, today
`~/src/scoping-intake`), client-data isolation is the invariant, and about a dozen mechanisms are
introduced.
**Code anchors** verified against main `b7688370cf` (2026-10-14). Cite-by-symbol; re-locate at
implementation time.
**Design source of truth:** `~/mission-control/consulting/notes/client-dashboard-design.md`
(2026-10-14), decisions logged in `~/mission-control/consulting/decisions.md` under 2026-10-14. This
PRD restates only what an architect needs; where the two differ, the note wins and this PRD is amended.
**Authorisation:** Leo, 2026-10-14: "You're expressly authorised to change what dark factory tracks in
order to facilitate delivery of this client dashboard." Used by θ.

## 1. Goal

A paying client opens their private link on the portal and watches their block take shape: whether
the factory is running, what is in flight right now, every change that landed and when, every test
run with per-test results, the acceptance criteria and their latest outcome, the client-visible source
tree down to file names with line counts, and the files they chose to reveal, updated on every advance
of main. Nothing from any other project, and nothing of the factory's own configuration, ever. The
portal renders it; this PRD produces it.

When this lands, an operator can run, for one client project:

```
client-feed publish --config ~/.config/client-feed/<name>.yaml
```

where the config names one project root, one output directory, one inbox directory and the policy,
and within a few seconds `out/feed.jsonl` and `out/state.json` exist, carry only that project's facts
filtered by the policy, and keep growing as the factory works. An audit-hook test proves the publisher
opened nothing outside the root and its two directories; a two-project canary fixture proves no byte of
the other project reached the feed. The portal's own fixture of the contract validates against this
package's schema, and this package's fixture feed renders in the portal (two-way boundary tests, §9).

## 2. Background

- The engagement design (`block-engagement-design.md`) promises "progress metrics visible throughout"
  and client-visible status; on 2026-10-13 Leo decided the client dashboard is part of the alpha trial
  and on 2026-10-14 that it is unified with the scoping intake into one client portal. The portal is
  LDE's product and holds prospect data; dark-factory is AGPL and must hold neither, so the factory's
  half is a generic publisher and a contract (design note §7).
- Nothing client-facing exists. The operator dashboard is fleet-wide, bound to loopback, has
  unauthenticated write endpoints (`dashboard/src/dashboard/app.py` POST routes under
  `/api/v2/dashboard/scheduler/…` and `/curator/cancel`), never reads the task store directly for its
  tree (it fans out over every registered root through fused-memory
  `dashboard/src/dashboard/data/tasks.py::fetch_tasks`), and identifies a project by its directory
  basename (`dashboard/src/dashboard/data/mcp_fanout.py::project_label`). None of it is reused.
- Every fact the feed needs is recorded, per project, in structured form, except one:
  - the task store at `<root>/.taskmaster/tasks/tasks.db` (`shared.project_layout.tasks_db`; tables
    `tasks` and `dependencies`; PK `tag, id`; `metadata` JSON carries `prd_path`, `prd_task_label`,
    `done_provenance`);
  - the event store at `<root>/data/orchestrator/runs.db` (`shared.project_layout.runs_db`; `events`
    with `event_type`, `task_id`, `phase`, `data`; `merge_finalized` with `data.state == 'done'` and
    `data.merge_sha` is the landed fact; `train_merged` carries `member_task_ids`; `merge_verify` and
    `workflow_verify` carry `passed` and `duration_ms`; `phase_enter`/`phase_exit` carry the phase in
    `events.phase`; `scheduler_paused`/`scheduler_resumed`; `escalation_created`/`escalation_resolved`, whose `data`
    carries `level` on some rows only (452 of 2,675 on 2026-10-14); `task_results.review_cycles`);
  - the verify archive at `<root>/data/verify-logs/<task_id>/` (`attempt-N[.module].junit-<ts>.xml.gz`
    for task legs on every verify — 2,294 reports under 1,885 task directories on 2026-10-14 — and for
    merge runs only under `merge_verify_breadth: full`; `summary-<ts>.json` with `category` and `rc`;
    pruned past 30 days and oldest-first past 500 MB by `orchestrator/src/orchestrator/verify.py::_prune_archive`);
    task legs archive at `project_root / 'data' / 'verify-logs'` (the `run_scoped_verification` call in
    `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._run_scoped_verification_with_infra_retry`);
  - git: merge commits `Merge task/<id> into main`, branches `task/<id>`.
- **Not recorded today:** the review outcome. `TaskWorkflow._execute_verify_review_loop` holds a
  `ReviewAggregation` (`orchestrator/src/orchestrator/artifacts.py::ReviewAggregation`:
  `has_blocking_issues`, `blocking_issues`, `suggestions`, `reviews`, `reviewer_errors`) and never
  writes it; θ emits it as an event with counts only.
- The verify archive and junit injection are pytest-only (`with_junitxml` is a no-op for any other
  tool), so test and criteria evidence exists for pytest projects only; the feed says so.
- No acceptance-suite data format exists; §7.7 defines one.
- `/factory-init` commits the orchestrator config, `.mcp.json`, `.envrc`, Claude settings and a review
  briefing into a target repo, and review reports land in `plans/`; the engagement design strips them
  from delivered history. The client-visible tree is therefore an allowlist (§4.4), never "everything
  tracked".

## 3. Sketch of approach

A long-running publisher per client project, under Leo's account, handed one config file that names
exactly one project root, one output directory (its own), one inbox directory (the portal's) and the
policy. It polls the two stores read-only, watches main's head, reads the verify archive and git
objects, applies the policy and a field allowlist, and appends typed events to `out/feed.jsonl` while
rewriting `out/state.json` atomically as the fold of its own feed. It tails `in/inbox.jsonl` for reveal
requests and reads `in/engagement.json` for the stage and the delivered commit. The portal (other repo)
owns `in/`, writes the inbox and the engagement projection, and renders.

```
client-feed/src/client_feed/
  contract.py        feed events, state document, inbox events, engagement projection (pydantic; FEED_VERSION)
  policy.py          the publisher's config: root, dirs, visibility globs, reveal proportion and floor, free/never globs, counting caps
  feed_io.py         append-only writer (O_APPEND, O_NOFOLLOW, monotonic seq), torn-tail truncation, rotation, atomic state (mkstemp+rename), inbox tailer, cursor file
  fold.py            state = fold(feed) up to a seq watermark; the one reducer both state.json and tests use
  sources/tasks.py   tasks.db read-only projection (tag pinned), incremental by updated_at; waiting_on from open escalations
  sources/events.py  runs.db read-only cursor over events.id → typed events, incl. factory_state and review_completed
  sources/git.py     head, branches, merge stats, visible-tree snapshot with counted lines and kinds, blob content via cat-file
  sources/archive.py junit → test_results (ids, outcomes, durations, exception type); summary → gate legs
  sources/prds.py    PRD title and excerpt from metadata.prd_path, path-redacted
  budget.py          counted set, spent, allowed; reveal identity; hostile-path refusal reasons
  criteria.py        criteria.yaml model, the {nodeid: criterion} map, junit join, results events
  (no pytest plugin of its own: the collection map is written by the defect-tests PRD's injected plugin, see §12)
  publisher.py       the loop: sources → events → feed; heartbeat with degraded reasons
  cli.py             publish, snapshot (one pass), criteria-results, dump-state, validate-feed
client-feed/tests/   contract fixtures, source tests over fixture stores, audit-hook isolation, canary, end-to-end on this repo
```

The publisher never imports `orchestrator`, `fused_memory` or `dashboard`; it depends on `shared` only
for `shared.project_layout` (`tasks_db`, `runs_db`) and `shared.task_metadata` (`done_provenance`), plus, from
§14, `shared.defects` (the ref rule, in the non-vendorable `defect_ref.py`) and `shared.mcp_session` (the filer only),
and on `pydantic`, `pyyaml` and the standard library. It uses no helper that defaults to the factory's
own checkout (`shared.project_layout.default_project_root` is banned in `client_feed` by a test).

## 4. Resolved design decisions

1. **Files are the transport, one writer per directory.** `out/` (owned by Leo's account, group
   `portal`, mode 2750) holds `feed.jsonl`, `state.json`, `cursor.json`; `in/` (owned by the portal
   account, group `leo`-readable, mode 2750) holds `inbox.jsonl` and `engagement.json`. The publisher
   opens with `O_NOFOLLOW`, writes temp files with `mkstemp` in `out/` and renames. The portal cannot
   plant a symlink or replace anything the publisher writes. No network between the halves.
2. **The feed is the record; `state.json` is its fold at a watermark.** `fold.py` is the one reducer;
   the publisher applies it to its own feed, writes `state.json` with `watermark_seq`, and the portal
   resumes from the watermark it last applied. Sources are never a second input to the state (they
   are pruned; the feed is not).
3. **Append-only with recovery rules.** `seq` strictly increasing; on start the publisher truncates a
   torn final line and logs it; a major `FEED_VERSION` bump rotates to `feed.<major>.jsonl`; revealed
   content is emitted only when the blob changed; `test_results.cases` is capped per run (default
   5,000 rows, the rest counted).
4. **Visibility is an allowlist in factory-side policy.** The client-visible tree is the set of
   tracked files matching `policy.visible_globs`; files outside it are neither listed nor revealable.
   The same policy holds the reveal proportion, floor, free globs, never-reveal globs and the counting
   caps. Nothing the portal writes can widen what a client may see.
5. **Direct read-only reads of the two stores, never the fused-memory MCP.** Opened with `mode=ro`
   URIs; `tasks` queried with `tag = 'master'` pinned (the only tag on the live store, checked
   2026-10-14). The publisher has no write path to any store.
6. **Field allowlist, not denylist.** A task contributes `id, title, status, updated_at, depends_on,
   prd_path, prd_task_label, landed_commit, waiting_on`. `description`, `details`, `test_strategy`
   and the rest of `metadata` never leave the publisher. Titles and PRD excerpts have `/home/...` and
   absolute paths redacted. A failing test contributes its id, outcome, duration and exception type,
   never message or output. `merge_blocked.data.reason` is never copied. §14.3 adds exactly two fields to
   the read side: `defects[].ref` values that are `portal:<policy project_id>:` refs, and
   `x_client_defect_gate_for`.
7. **One root per publisher, proven.** The publisher is constructed from one config and opens
   nothing outside the root, `out/` and `in/`; the test installs `sys.addaudithook` and asserts every
   `open`, `sqlite3.connect` and subprocess working directory lies under those three (B1).
8. **Project identity is explicit.** The config names `project_id`; every feed line and `state.json`
   carry it; the portal refuses a feed whose `project_id` is not the one it expected (B4).
9. **Event types (FEED_VERSION "1.0"):** `heartbeat`, `factory_state`, `prd_upserted`,
   `task_upserted`, `task_phase`, `verify_run`, `test_results`, `merge_landed`, `branch_snapshot`,
   `tree_snapshot`, `file_revealed`, `reveal_refused`, `criteria_results`, `criteria_declared`. A new type is a minor
   version; a changed field a major one. The portal ignores unknown types and refuses an unknown major.
   FEED_VERSION "1.1" adds `defect_filed` and the `Source` member `filer` (§14.3). A new member of an enum the
   reader validates as closed is not minor for that reader; §14.3 states the ordering that makes it safe.
10. **Never hide red, from recorded sources.** A failed verify is `verify_run.passed == false`; a
    blocked task is `task_upserted` with that status and `waiting_on: leo` when an open escalation
    whose recorded `level` is 2 exists for it, else `factory` (an open escalation with no recorded
    level is the factory's own ladder, not Leo); `factory_state` carries `scheduler: running|paused`, `last_event_at`
    and `idle_seconds` from the project's own `runs.db`. Where a fact is not recorded (no junit, no
    summary) the event says `not_recorded`, never pass.
11. **The publisher reports its own health.** Every `heartbeat` carries `head`, `as_of`,
    `engagement_version`, `watermark_seq` and `degraded: [{source, reason}]`; a source failure is never
    swallowed (INV-11) and the portal shows a stale or degraded publisher as such, distinct from
    "quiet" (INV-6, INV-13).
12. **Reveal budget** (design note §5): counted set = visible production files not matching free,
    never-reveal or generated globs (`linguist-generated` honoured); lines = non-blank lines, a line
    over `max_line_bytes` counting as ⌈bytes/max⌉; budget = `reveal_proportion × counted lines added or
    changed since delivered_commit` (all counted lines when null); a path in the counted set costs
    `max(its chargeable lines, reveal_floor_lines)` and a path outside it costs 0; a request that adds
    nothing to `spent` (a path outside the counted set, or one already revealed) is granted whatever
    `spent` is, and any other is granted when `spent + cost ≤ budget` (where this is implemented:
    §12, the 2026-10-22 δ entry (b) and the 2026-10-25 entry (a)). Reveal identity is
    the path until a delete event; a recreated path needs a new reveal; renames follow git's default
    detection; a deleted file emits a final `file_revealed {deleted: true}` and stops counting.
13. **Hostile requests are refused with typed reasons.** A requested path must appear in
    `git ls-tree -r <head>` filtered by the visible set; content is read with `git cat-file blob` at
    that commit, never from the working tree; symlink and submodule entries are refused; `..`,
    absolute paths and unknown paths are `reveal_refused {reason: not_in_tree|not_visible|
    not_revealable|over_budget|symlink}`.
14. **`engagement.json` is a projection with one writer.** `{engagement_version, project_id, stage,
    delivered_commit}` written by the portal's admin page (a hand edit only until that page exists);
    the publisher echoes `engagement_version`. The engagement record in the portal is the home (INV-9).
15. **Cursors persist.** `cursor.json` records the last `events.id`, the last `tasks.updated_at`, the
    last head sha, the inbox byte offset and the blob line-count cache. A restart resumes without
    re-emitting (B18).
16. **The factory records the review outcome (θ).** `review_completed` per review cycle with
    `{task_id, cycle, has_blocking_issues, blocking_count, suggestion_count, by_category: {category: n},
    reviewer_count, reviewer_errors}`, counts only, emitted through `EventStore.emit` from
    `TaskWorkflow._execute_verify_review_loop` where the aggregation is decided (both the PASS exit
    and the `review_cycle += 1` branch). `merge_landed.review` carries the last cycle's counts; until θ
    has written a row in production the feed says `not_recorded` (INV-13).
17. **Criteria data is pytest-only and never written into the checkout** (§7.7). The defect-tests
    PRD's injected plugin (`plans/defect-tests-prd.md` β) writes the collection map
    `{nodeid: {defects, criterion}}` beside the junit report; the publisher joins by nodeid; a staging
    run's results are produced by the CLI into `out/`. (Amended 2026-10-16, §12.)
18. **Naming:** package `client-feed/`, module `client_feed`, CLI `client-feed`, unit
    `client-feed@<name>.service`. The portal's name is the other repo's decision.

## 5. Pre-conditions

None in this repo beyond θ for review counts (the feed degrades to `not_recorded` without it). The
portal half (other repo) consumes α's contract and is filed there with a cross-project dependency on
α.

## 6. Cross-PRD relationship

| Seam | Direction | Mechanism | Owner |
|---|---|---|---|
| Feed, inbox and engagement contracts | this PRD produces; the portal PRD consumes | `client_feed.contract` models, the fixture feed, `validate-feed` | **this PRD (α)** |
| The two directories, their ownership, the dedicated account | the portal PRD owns the account and both directories' creation; this PRD's unit writes into `out/` | `deploy/` in the portal repo; `client-feed@.service` here | portal PRD for the account and directories; this PRD (η) for the publisher unit |
| Reveal requests and refusals | portal produces requests; this PRD consumes and answers | `in/inbox.jsonl` `reveal_requested`; `file_revealed` / `reveal_refused` | this PRD (δ) for the rule; portal for the UI |
| Acceptance criteria format | this PRD defines; the client project's suite and the portal consume | `acceptance/criteria.yaml`, the plugin, the map file, `criteria_results` | **this PRD (ε)** |
| Review outcome event | this PRD adds to the orchestrator; this PRD consumes | `review_completed` | **this PRD (θ)** |
| Design tokens | no code seam | `consulting/outputs/design-tokens.css` | mission-control |
| Operator dashboard | none; nothing shared | — | — |
| Client defect bridge: `defect_reported`, the filer, `defect_filed`, the portal's MCP-port rule | see §14.7 | §14.3 | this PRD (σ1a–σ5) for the contract, the acknowledgement and the filer; the portal PRD for the writer, the restater and its deploy rule |

No reciprocal ownership remains.

## 7. Contract

All models live in `client_feed/contract.py` (pydantic v2, `extra='forbid'`, frozen). JSON on disk is
UTF-8, one object per line in `.jsonl` files. Times are ISO-8601 UTC. `FEED_VERSION = "1.0"`.

### 7.1 Envelope

Every feed line: `{seq: int, at: str, project_id: str, feed_version: str, type: str, ...}`. `seq` is
strictly increasing within a feed file; a reader that sees a non-increasing `seq` stops and reports a
corrupt feed. Events with a natural key (`task_upserted.task_id`, `prd_upserted.prd_path`,
`file_revealed.path`, `verify_run.run_key`, `merge_landed.commit`) are idempotent under the fold.

### 7.2 Events

| type | fields | source | emitted when |
|---|---|---|---|
| `heartbeat` | `head`, `as_of`, `engagement_version`, `watermark_seq`, `degraded: [{source, reason}]` | publisher | every pass (default 5 s) and on start |
| `factory_state` | `scheduler: running\|paused`, `since`, `last_event_at`, `idle_seconds` | `scheduler_paused`/`scheduler_resumed`; newest `events.timestamp` | when any field changes |
| `prd_upserted` | `prd_path`, `title`, `excerpt`, `task_counts: {status: n}` | `sources/prds.py`; title = first H1, excerpt = first non-heading paragraph, paths redacted | first seen or counts change |
| `task_upserted` | `task_id`, `title`, `status`, `waiting_on: factory\|leo\|null`, `updated_at`, `depends_on`, `prd_path`, `prd_task_label`, `landed_commit` | `sources/tasks.py`; open escalation from `escalation_created`/`escalation_resolved` | `updated_at` moves past the cursor or an escalation opens/closes |
| `task_phase` | `task_id`, `phase`, `entered: bool`, `at` | `phase_enter`/`phase_exit` | per event |
| `verify_run` | `run_key`, `task_id`, `kind: task\|merge`, `passed`, `duration_ms`, `commit`, `legs: [{name, category, rc}] \| not_recorded` | `workflow_verify`/`merge_verify` + summary JSON | per event |
| `test_results` | `run_key`, `counts: {passed, failed, skipped, errored, truncated}`, `duration_ms`, `cases: [{id, outcome, duration_ms, criterion, exception_type}]` | junit under the task's archive dir matching the run; the criterion map file beside it | when a junit report is found for a `verify_run` |
| `merge_landed` | `commit`, `at`, `task_ids`, `titles`, `files_changed`, `insertions`, `deletions`, `review: {cycles, blocking_count, suggestion_count, by_category} \| not_recorded` | `merge_finalized` (`state == done`), `train_merged`; `git show --stat`; `review_completed` (θ) | per landed merge |
| `branch_snapshot` | `branches: [{name, task_id, ahead, last_commit_at}]` | `git for-each-ref refs/heads/task/*` | when the set or any tip changes |
| `tree_snapshot` | `commit`, `totals: {files, production_lines, test_lines, docs_lines}`, `files: [{path, kind, lines, bytes, last_changed_at}]`, `budget: {proportion, counted_lines, budget_lines, spent_lines, floor_lines}` | `git ls-tree -r -l HEAD` filtered by `visible_globs`; blob line counts cached by sha; last change from the commits walked since the previous snapshot | on every advance of main |
| `file_revealed` | `path`, `commit`, `lines`, `content`, `deleted: bool`, `renamed_from` | `git cat-file blob` | on grant, and on an advance that changes the blob |
| `reveal_refused` | `path`, `reason` (§4.13) | budget | per refused request |
| `criteria_results` | `run_kind: verify\|staging`, `commit`, `run_at`, `results: [{id, outcome: passed\|failed\|not_run, tests: [id]}]` | the junit join for `verify`; the CLI's file in `out/` for `staging` | when the newest result of a run kind changes |
| `criteria_declared` | `criteria: [{id, prd, capability, statement, kind}]` | `acceptance/criteria.yaml` | on start and whenever the file's content hash changes (added 2026-10-15 at the portal PRD's ask: results carry ids only) |

`kind` for a tree file: `test` if the path matches `test_globs` (default `**/tests/**`, `**/test_*.py`,
`**/*_test.py`, `**/conftest.py`), `docs` if `**/*.md`, `docs/**`, else `production`; files matching
`generated_globs` or marked `linguist-generated` are listed with `kind: generated` and excluded from
the budget.

### 7.3 `state.json`

`fold(feed)` up to `watermark_seq`: `{project_id, feed_version, watermark_seq, as_of, head,
engagement_version, degraded, factory, prds: {prd_path: …}, tasks: {task_id: …}, in_flight: [{task_id,
phase, since}], branches, tree, revealed: [{path, commit, lines}], criteria: {run_kind: …},
criteria_declared: {id: …}, recent: [last 200 events]}`. Each `criteria_declared` event replaces the
whole `criteria_declared` map. Rewritten atomically after every pass that appended.

### 7.4 `in/inbox.jsonl` (portal → publisher)

`{seq, at, type: "reveal_requested", path}`. Answered with `file_revealed` or `reveal_refused`. A request
for a path already revealed and still revealable appends nothing, because the feed already holds its
content at head (§12, 2026-10-25 entry (b)). A second line type, `defect_reported`, is added by §14.3.

### 7.5 `in/engagement.json` (portal → publisher)

`{engagement_version: int, project_id, stage: str, delivered_commit: str|null}` (`stage` is a plain string here; its enum's home is the portal). Missing or unparsable: the
publisher runs with reveals disabled and names it in `heartbeat.degraded`.

### 7.6 Publisher policy (`~/.config/client-feed/<name>.yaml`, factory-side)

`{project_id, project_root, out_dir, in_dir, interval_seconds: 5, visible_globs: [...],
reveal_proportion: 0.25, reveal_floor_lines: 50, free_globs: [tests, docs defaults],
never_reveal_globs: [], generated_globs: [], test_globs, docs_globs, max_line_bytes: 400,
max_cases_per_run: 5000}`. Validated by `policy.py`; a config naming a root that is not a git
repository, or an `out_dir` not owned by the running user, refuses to start.

### 7.7 Acceptance criteria data

- `acceptance/criteria.yaml` in the client project:
  `{format: 1, criteria: [{id: "C-12", prd: "plans/x-prd.md", capability, statement, kind: success|failure}]}`.
- `@pytest.mark.criterion("C-12")`. The marker is registered, and the map written, by the
  defect-tests PRD's single-file plugin `_df_defect_plugin` (`plans/defect-tests-prd.md` β,
  decision 19), which the orchestrator injects into every structured pytest verify leg with `-p`;
  a client project needs no conftest edit. At collection it writes
  `<junit path>.tests.json` = `{nodeid: {"defects": [ref, ...], "criterion": str | null}}` beside
  the `--junitxml` target when one is given, so the archived junit has its map beside it. A staging
  run loads the same plugin with `-p _df_defect_plugin` on `PYTHONPATH`. No results file is written
  into the checkout.
- The publisher joins junit and map by nodeid into `criteria_results {run_kind: verify}`.
- `client-feed criteria-results --junit <xml|xml.gz> --map <json> --criteria acceptance/criteria.yaml
  --run-kind staging --commit <sha> --out <out_dir>` writes `out/criteria-staging.json` and the
  publisher emits it as `criteria_results {run_kind: staging}`.

### 7.8 CLI

`client-feed publish --config C` (the loop); `client-feed snapshot --config C` (one pass, exits);
`client-feed dump-state --out-dir D`; `client-feed validate-feed --out-dir D` (the contract check the
portal's tests also run); `client-feed criteria-results …` (§7.7).

## 8. Isolation invariant (normative)

For a publisher configured with root R, output directory O and inbox directory I: every byte it writes
under O is a function of R's git objects, R's `.taskmaster/tasks/tasks.db` rows with `tag='master'`,
R's `data/orchestrator/runs.db`, R's `data/verify-logs/`, R's `acceptance/criteria.yaml`, the policy,
and I's `inbox.jsonl` and `engagement.json`, filtered by the policy's visible set. It opens no other
path. It writes nowhere but O. The portal account reads O and nothing under R.

The client defect filer is a separate process under its own invariant (§14.5); it widens nothing here.

## 9. Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Audit-hook confinement | fixture root A with tasks.db, runs.db, verify-logs, git; a sibling root B and the real dark-factory checkout both reachable on disk | `sys.addaudithook` records every open / sqlite connect / subprocess cwd during `snapshot`; all lie under A, O or I; `default_project_root` is never called |
| B2 | Two-project canary | B's task titles, descriptions, metadata, PRD text, file names and contents carry `CANARY-B-…` | no file under O contains `CANARY-B` after publishing A |
| B3 | Allowlist | a task in A with `description`, `details`, `metadata.dry_run_proposals`, `metadata.external_deps`; a title containing `/home/leo/x` | `task_upserted` and `state.json` carry none of them; the title is redacted |
| B4 | Foreign project refusal | the policy names project X; feed lines name Y | `validate-feed` rejects; the portal's consumer test (other repo) rejects the same fixture |
| B5 | Fixture round trip | the committed fixture feed `tests/fixtures/feed-v1/` | validates under `contract.py`; `fold(feed)` equals the committed `state.json`; the portal's suite validates and renders it |
| B6 | Landed merge with review | `merge_finalized` `state=done` for task 7, two `review_completed` rows, a `Merge task/7 into main` commit | one `merge_landed` with that sha, stat counts from git, `review.cycles == 2` and the last row's counts; with no rows, `review == "not_recorded"` |
| B7 | Red shown | an archived red attempt summary (a leg rc ≠ 0) and a junit with one failure under the task's archive dir | `verify_run.passed == false`; `test_results.counts.failed == 1`; `exception_type` present, no message |
| B8 | Not recorded | a `verify_run` whose task dir has no junit | `test_results` absent; `verify_run.legs == "not_recorded"` |
| B9 | Factory state | `scheduler_paused` then nothing for 3 h | `factory_state.scheduler == paused`, `idle_seconds ≥ 10800` |
| B10 | Waiting on Leo | task 9 blocked with an open `escalation_created` level 2 | `task_upserted.waiting_on == "leo"`; after `escalation_resolved`, `"factory"` |
| B11 | Visible set | tree with `dark-factory-orchestrator.yaml`, `.mcp.json`, `plans/review.md`, `src/a.py`; policy visible `src/**`, `tests/**` | `tree_snapshot.files` lists only `src/a.py`; a reveal of `.mcp.json` is `reveal_refused not_visible` |
| B12 | Reveal within budget | 1,000 counted lines, proportion 0.25, floor 50; request a 200-line production file | `file_revealed` with content; `spent_lines == 200` |
| B13 | Reveal over budget and floor | spent 210 of 250; request a 100-line file; then a 10-line file | both `reveal_refused over_budget` (the first costs 100: 310 > 250; the second costs the floor 50: 260 > 250) |
| B14 | Free set | request a test file | granted; `spent_lines` unchanged |
| B15 | Hostile paths | requests for `../x`, `/etc/passwd`, a symlink entry, an untracked path | four `reveal_refused` with reasons `not_in_tree`, `not_in_tree`, `symlink`, `not_in_tree`; nothing read outside the tree (B1's hook) |
| B16 | Follow head | a revealed file changes in a later commit; another is renamed; a third deleted then recreated | re-emitted; `renamed_from`; `deleted: true` then no content until a new request |
| B17 | Paid code free | `delivered_commit` set; a file unchanged since it | outside the counted set: excluded from counted lines, and its reveal costs 0 |
| B18 | Restart and torn tail | kill mid-append leaving a partial line; restart | the torn line is truncated and logged; no duplicate `seq`; no re-emitted `merge_landed`; `cursor.json` consistent |
| B19 | Degraded source | runs.db unreadable | heartbeat `degraded` names `events`; other sources still publish |
| B20 | Criteria | criteria.yaml with C-1..C-3; a junit and map binding C-1 pass, C-2 fail, none for C-3 | `criteria_results {verify}`: passed, failed, not_run |
| B21 | Review event recorded | a real review cycle on this repo after θ | `SELECT count(*) FROM events WHERE event_type='review_completed'` ≥ 1; the row carries counts and no issue text |
| B22 | End to end on this repo | `snapshot --config` for this checkout with a visible set of `*/src/**`, `*/tests/**` | exits 0; `state.json.prds` non-empty; every `task_upserted.title` matches a live row; `validate-feed` passes; no `description` and no `/home/` anywhere in O |

## 10. Decomposition plan

Sizing per the overlay (300–1,500 LOC per leaf, ≤ 12 declared files). Labels are Greek; ids are
assigned at decompose.

| Label | Task | Modules | Observable signal | Prereqs |
|---|---|---|---|---|
| α | New workspace member `client-feed/` with the contract, the fold, fixtures and `validate-feed` | `client-feed/pyproject.toml`, `client_feed/contract.py`, `client_feed/fold.py`, `client_feed/cli.py` (validate-feed), `client-feed/tests/conftest.py` (the basetemp wiring `tests/scripts/test_basetemp_git_isolation.py` requires), `client-feed/tests/`, root `pyproject.toml` members, `hooks/project-checks` (`PYRIGHT_PACKAGES`, `MAGICMOCK_TEST_DIRS`), `dark-factory-orchestrator.yaml` (`test_command`, `lint_command`, `type_check_command`), `CONTRIBUTING.md` where `tests/scripts/test_contributing_type_check_command_drift.py` pins the member count; the drift tests `test_hook_pyright_packages_drift.py` and `test_hook_magicmock_test_dirs_drift.py` must stay green | `client-feed validate-feed --out-dir client-feed/tests/fixtures/feed-v1` exits 0 and rejects a foreign `project_id` fixture (B4, B5); the package's tests run under the repo verify command | — |
| θ | Orchestrator records each review cycle's outcome as a `review_completed` event | `orchestrator/src/orchestrator/event_store.py` (`EventType`, `EventStore.emit`), `orchestrator/src/orchestrator/workflow.py` (`TaskWorkflow._execute_verify_review_loop`), `orchestrator/tests/` | B21 | — |
| β1 | Store sources: tasks projection with `waiting_on`, events cursor with `factory_state` and `review_completed`, PRD titles | `client_feed/sources/tasks.py`, `sources/events.py`, `sources/prds.py`, tests over fixture stores | `client-feed dump-sources --config` on this repo prints allowlisted, path-redacted tasks and typed events and no `description` (B3, B9, B10); B6's events half | α, θ |
| β2 | Policy, git and archive sources | `client_feed/policy.py`, `sources/git.py`, `sources/archive.py`, tests | a visible-set tree snapshot of this repo with kinds and counted lines (B11's listing half); junit → `test_results` with counts and exception types (B7, B8) | α |
| γ | Publisher loop, feed writer, cursors, heartbeat, CLI | `client_feed/publisher.py`, `feed_io.py`, `cli.py` (publish, snapshot, dump-state), tests | `client-feed snapshot` on this repo writes a valid feed whose `state.json` equals `fold(feed)` (B22 minus reveals); restart and torn tail (B18); degraded heartbeat (B19) | β1, β2 |
| δ | Reveals: engagement projection, budget, inbox, hostile-path refusal, follow-head | `client_feed/budget.py`, `feed_io.py` (inbox tail), `publisher.py`, tests | B11's refusal half, B12–B17 | γ |
| ε | Acceptance criteria format, map join (reading the defect-tests β map), staging CLI | `client_feed/criteria.py`, `cli.py` (criteria-results), `publisher.py`, tests, `acceptance/criteria.yaml` exemplar | B20; `client-feed criteria-results` on a junit and map from this repo's own suite with one marked test | γ |
| ζ | Isolation gate: audit hook, canary, end to end | `client-feed/tests/test_isolation.py`, the two-project fixture builder | B1, B2, B15 and B22 pass on this checkout | δ, ε |
| η | Publisher unit template, operator docs, README and ARCHITECTURE pointers | `scripts/client-feed.service.template`, `scripts/setup-host.sh` (sync the member; render only, no enable), `OPERATIONS.md` §"Client feed", `README.md`, `ARCHITECTURE.md` §2.3 | `systemctl --user cat client-feed@demo.service` renders from the template with the config path substituted; OPERATIONS names the install steps, the directory ownership and the isolation check | γ |

Edges: α, θ → β1; α → β2; β1, β2 → γ; γ → δ, ε, η; δ, ε → ζ. ζ is the integration gate and names
B1, B2, B15 and B22 as its signal. The portal PRD (other repo) depends on α by cross-project
dependency `dark_factory:<α>` and on ζ for its end-to-end test.

G7 walk at decompose; design-time notes: INV-1 (the contract is a pydantic model with committed
fixtures on both sides, and `fold.py` is the one reducer); INV-5 (α edits the lockstep package lists,
which `tests/scripts/test_contributing_type_check_command_drift.py`,
`tests/scripts/test_hook_pyright_packages_drift.py`, `tests/scripts/test_hook_magicmock_test_dirs_drift.py`
and `tests/scripts/test_basetemp_git_isolation.py` already pin); INV-6 and INV-13 (the heartbeat's
age and `degraded` list, and `factory_state`, make "publisher dead", "store never written" and
"factory paused" distinct from "quiet"; `merge_landed.review` renders `not_recorded` until θ has
produced a row); INV-9 (`engagement.json` echoes its version; the record lives in the portal; policy
has one home); INV-11 (no source failure is swallowed; B19); INV-12 (the publisher refuses to start on
a config it cannot validate and raises on a corrupt feed it would otherwise overwrite).

## 11. Out of scope

The portal: pages, streaming, clients, projects, blocks, stages, the admin page, the dedicated account
and both directories' creation (portal PRD, other repo). Running the acceptance suite against staging.
Review issue text (counts only, θ). Test evidence for non-pytest projects. Costs, tokens, model names,
escalation text (never shown). A second feed transport. Per-person links. Failing-test message text
(open in the design note; exception type only here). Raw client defect text (§14.2 decision 2).

## 12. Amendments

- 2026-10-15, from the portal PRD session (`~/src/scoping-intake/docs/prds/client-portal.md`, tasks 45–56, with 54 → `dark_factory:6966` and 56 → `dark_factory:6973`): `criteria_declared` event (α, ε); `stage` as `str` (α); out/ files group-readable 0640 (γ, task 6970); the portal's `in-v1` fixture vendored by δ (6971); a committed B22 snapshot fixture for the portal's κ (ζ, 6973). Task details amended accordingly.

- 2026-10-16, from the defect-tests PRD session (`plans/defect-tests-prd.md`, Leo's go): the pytest
  collection map moves to that PRD's injected plugin (β), which owns the `criterion` and `defect`
  markers and writes `<junit path>.tests.json` = `{nodeid: {defects, criterion}}`; client-feed ships
  no pytest plugin (§3, decision 17, §7.7, ε row). Task 6972 (ε) amended to consume it and made to
  depend on β (7002's contract work does not touch the map, so it is unchanged); θ 6967 adds
  `by_kind` counts to `review_completed`; the ε sidecar block's plugin check is replaced. The defect display
  (`cases[].defects`, `defects_results`) is that PRD's ι, not a task of this one.

- 2026-10-20, from β1 (task 6968), answering the esc-6969-1 note routed to it. These bind γ (6970):
  - (a) Red task verifies come from the verify archive's per-attempt summaries, which
    `orchestrator/src/orchestrator/verify.py::_archive_attempt_summary` writes for every attempt, green
    or red. γ assembles them through `client-feed/src/client_feed/sources/archive.py::read_archived_run`.
    `workflow_verify` cannot be that source: it is emitted only on VERIFY→REVIEW, with `passed=true`
    (0 of 6,377 live rows are false). β1 still maps the recorded `passed` of `workflow_verify` and
    `merge_verify` (`client-feed/src/client_feed/sources/events.py::read_events`), so either source
    flows through unchanged. B7's precondition is amended to match.
  - (b) `waiting_on` (`client-feed/src/client_feed/sources/tasks.py::waiting_on`): done or cancelled
    gives null; an open escalation with recorded level 2 gives `leo`; any other open escalation, or a
    blocked task, gives `factory`; otherwise null. An escalation is closed by `escalation_id`, else by
    `escalation_ids`, else task-wide by an outcome-only `escalation_resolved`
    (`client-feed/src/client_feed/sources/events.py::read_open_escalations`). Level promotions and
    most resolutions are not recorded in runs.db, so `leo` rarely fires today; follow-up ticket
    tkt_0RW1F0587N3HXAH0PN8A4A5TYJ covers recording them.
  - (c) PRD title and excerpt are read from main's git objects, never the working tree (§8;
    `client-feed/src/client_feed/sources/prds.py::read_prds`).
  - (d) `review.cycles` = max(newest `review_completed` cycle, Σ `task_results.review_cycles` for the
    task); a task with no valid row gives `not_recorded`
    (`client-feed/src/client_feed/sources/events.py::read_review_counts`).

- 2026-10-21, from β1's review amendments (task 6968). These bind γ (6970):
  - (e) The tasks cursor (`client-feed/src/client_feed/sources/tasks.py::read_tasks`) is the last
    projected row's raw `updated_at`. A skipped row never moves it, so it is read again until a
    projected row passes it. The cursor is not exactly-once. Rows are ordered as the store spells
    `updated_at` (`.mmmZ` beside microsecond `+00:00`), and the next read takes rows strictly after
    the cursor. So a row stamped at or before the cursor (the producer stamps to the millisecond),
    but committed after the read, is missed until that task is next written. If γ needs every
    write, it re-reads with `since=None` on some cadence and emits only rows that differ from its
    folded state.
  - (f) `client-feed/src/client_feed/cli.py` is no longer vendorable: `dump-sources` pulls in
    `yaml`, `shared.project_layout` and git through `policy` and the sources.
    `client-feed/tests/test_import_boundary.py` still lists `cli` in `VENDORABLE_MODULES` and
    passes, because it checks only each direct import's root package. That file is outside β1's
    scope; ticket tkt_0RW1SJ7XHC6R7VVTB1AERS60T7 drops `cli` from the set.
  - (g) `client-feed/tests/_stores.py` copies the producers' DDL by hand, citing each by
    path::symbol. Nothing executable catches drift, so a renamed producer column would leave the
    fixture tests green and a live read failing. The risk is accepted: `client-feed dump-sources`
    against a live root is the check.

- 2026-10-21, from γ (task 6970). These bind δ (6971), ε (6972), η (6974) and the portal:
  - (a) A task `verify_run` is closed at each verify-phase exit. Its window runs from that task's
    `phase_enter` with phase `verify` to the `phase_exit`, and it is read through
    `client-feed/src/client_feed/sources/archive.py::read_archived_run`
    (`client-feed/src/client_feed/stream_events.py::_task_run`). `passed` is "every archived leg's
    rc is 0". A window with no archived summary emits no `verify_run`, because nothing records
    whether it passed (§4.10). `workflow_verify` is not mapped: it follows the same phase exit, so
    mapping it would report every green run twice. Merge runs come from `merge_verify`, which is
    emitted after its run, so their window is `[at - duration_ms, at]`; with no duration or no
    task id their legs are `not_recorded`
    (`client-feed/src/client_feed/stream_events.py::_merge_run`). `run_key` is
    `'<kind>-<runs.db event id>'`, from the closing `phase_exit` or the `merge_verify` row.
  - (b) `heartbeat.watermark_seq` is the watermark of the `state.json` on disk when the heartbeat
    was appended, i.e. the previous commit's
    (`client-feed/src/client_feed/publisher.py::run_pass`), as in the committed feed-v1 fixture.
  - (c) `factory_state` is emitted only when (`scheduler`, `since`, `last_event_at` floored to the
    minute, `idle_seconds // 60`) differs from the folded state's
    (`client-feed/src/client_feed/value_events.py::factory_event`, `IDLE_RESOLUTION_SECONDS`). So
    the folded `idle_seconds` is at most a minute behind, and a busy factory emits at most one
    line a minute.
  - (d) A train's `merge_landed.review` is `not_recorded`: its members were reviewed separately,
    and summing their counts would report a review that never happened
    (`client-feed/src/client_feed/stream_events.py::_merge_landed`).
  - (e) `cursor.json` adds `through_seq` and `feed_bytes`
    (`client-feed/src/client_feed/feed_io.py::Cursor`). A commit appends and fsyncs, then
    replaces `state.json`, then `cursor.json` (`client-feed/src/client_feed/feed_io.py::OutDir.commit`).
    On restart, the lines past `through_seq` are folded. The first pass skips any stream event
    whose identity is among them: `(type, natural_key)`, or for `task_phase` its payload without
    `seq` (`client-feed/src/client_feed/stream_events.py::identity`,
    `client-feed/src/client_feed/publisher.py::_published`). The stamps of skipped events are not
    reused, so a `seq` gap can follow a crash. The contract requires only that `seq` strictly
    increases.
  - (f) An archive file that cannot be read loses only that run's archived details: a task run is
    not emitted, a merge run's legs are `not_recorded`, and `archive` is named in `degraded`
    (`client-feed/src/client_feed/stream_events.py::_archived`). A merge commit git cannot read holds
    the whole events batch: `events_id` does not advance, and `git` is named in every heartbeat.
    The hold is bounded only for a commit the repository does not hold. Once the landing is more
    than `UNKNOWN_MERGE_GRACE` (one hour) old, and only on a pass where git still reads main's
    head, it is skipped when git, running, reports that the repository does not hold the commit
    (`client-feed/src/client_feed/sources/git.py::has_commit`): no `merge_landed`, an ERROR log
    naming the commit and the runs.db event id, and `skipped N landing(s) whose merge commit is not
    in the repository` under `git` (`client-feed/src/client_feed/publisher.py::_skip_missing_before`,
    `client-feed/src/client_feed/stream_events.py::_missing_before`). A timeout, a failed spawn, a
    killed git or any other failure reading a commit the repository holds keeps holding the batch,
    however old the landing. The rule is stateless: a restart
    neither resets nor extends it, and a broken repository never drops a landing. Git is read
    lazily, in landing order and before any archive, and only for landings not already in the
    uncommitted tail, so a held pass re-reads only the landings ahead of the one held
    (`client-feed/src/client_feed/stream_events.py::_landed`). The stream read is not capped per
    pass: one commit can land in more than one row, and `_first_landings` folds them only within
    one read, so a batch boundary between them would publish a second `merge_landed`. The
    other failure rules: each source unit fails alone, and its typed error becomes
    `Degradation(source, redact_paths(str(exc)))` while any other exception propagates
    (`client-feed/src/client_feed/publisher.py::_guarded`). Unprojectable rows are named as
    `skipped N rows: <reason>=<count>, …` under `events` or `tasks`
    (`client-feed/src/client_feed/publisher.py::_skipped`). While `runs.db` cannot be read, no
    escalation is known: a re-read task whose status the state already records keeps its recorded
    `waiting_on`, so a degraded pass holds the last known value rather than publishing a guess
    (`client-feed/src/client_feed/value_events.py::task_events`).
  - (g) Until δ, `tree_snapshot.budget` uses §4.12's null `delivered_commit` rule. `counted_lines`
    counts the production files that match neither `free_globs` nor `never_reveal_globs`.
    `budget_lines` is `floor(proportion × counted_lines)` and `spent_lines` is 0
    (`client-feed/src/client_feed/value_events.py::_budget`). δ moves the rule into its budget module.
  - (h) Tasks are read from the tasks cursor, and re-read with `since=None` on a process's first
    pass and every `TASKS_RESCAN_PASSES` = 60 passes after it
    (`client-feed/src/client_feed/publisher.py::publish`). A row is emitted only when it differs
    from the folded state, which closes the tasks-cursor gap of β1's amendment (e). `snapshot` always re-reads
    (`client-feed/src/client_feed/publisher.py::snapshot`). For η: `client-feed publish` exits 0 on
    SIGTERM or SIGINT once its pass in progress completes. The loop runs on a worker thread, so the
    handler never sets the stop Event while its own thread holds that Event's lock
    (`client-feed/src/client_feed/cli.py::_publish_until_signalled`). δ's inbox tail is
    `client-feed/src/client_feed/inbox.py::read_inbox`, with its offset in `Cursor.inbox_offset`.
- 2026-10-22, from task 7221 (B22). These bind γ (6970) and ε (6972):
  - (a) A test case id reaches the feed through
    `client-feed/src/client_feed/sources/archive.py::_case_id`, which applies
    `client-feed/src/client_feed/redact.py::redact_paths`. γ emits `read_archived_run`'s case ids as
    they are and never re-derives them.
  - (b) ε's `criteria_results` `results[].tests` carries those same ids. The collection map's nodeids
    are converted to that spelling, redaction included, before the join, so both sides of the join
    pass through the one rule and no raw nodeid is emitted. The staging CLI
    (`client-feed criteria-results --run-kind staging`) follows the same rule.
  - (c) Case ids were never unique, and redaction can merge parametrized ids (measured 2026-10-22 on
    task 5244's archived junits: 90,933 cases, 3,199 duplicate ids before redaction and 3,220
    after). A join must not assume one case per id.

- 2026-10-22, from δ (task 6971). These bind ζ (6973) and the portal:
  - (a) A request is refused for the first reason that holds, in the order `not_in_tree`,
    `not_visible`, `symlink`, `not_revealable` (a `never_reveal_globs` match, or a blob that is not
    UTF-8), `over_budget` (`client-feed/src/client_feed/budget.py::judge`). A submodule is refused as
    `symlink`. Tree membership is the request's raw `path` text looked up among the paths `git
    ls-tree` lists at main's head (`client-feed/src/client_feed/sources/git.py::tree_entries`); it is
    never normalized, joined to a directory or opened on disk. Content is read only after the path,
    visibility and kind checks, by the entry's blob sha through `git cat-file --batch`
    (`client-feed/src/client_feed/sources/git.py::blob_reader`).
  - (b) A path costs `max(lines, reveal_floor_lines)` only inside the counted set, and 0 outside it
    (`client-feed/src/client_feed/budget.py::CountedSet.cost`). The counted set is head's production
    files that match neither `free_globs` nor `never_reveal_globs` and have more than 0 chargeable
    lines (`client-feed/src/client_feed/budget.py::counted_set`). `spent_lines` is not a running
    total: it is recomputed at head from the revealed set, so a path that leaves the counted set
    stops costing (`client-feed/src/client_feed/budget.py::CountedSet.spent`). A path already
    revealed is granted again without a budget check
    (`client-feed/src/client_feed/budget.py::CountedSet.grantable`). With a `delivered_commit`, a
    file's chargeable lines are the lines it added or changed since that commit. Paths are paired by
    `git diff-tree -M`, and each changed blob pair's added lines are counted by
    `client-feed/src/client_feed/policy.py::Policy.counted_lines`, not by numstat
    (`client-feed/src/client_feed/budget.py::lines_since`). A file added since then is charged all
    its counted lines, and a pure rename is charged none.
  - (c) `RevealBudget` is said only inside `tree_snapshot`. When the budget changes and the tree
    does not, the state's tree is re-emitted at the same commit with the new budget
    (`client-feed/src/client_feed/value_events.py::budget_event`). Two `tree_snapshot` lines with one
    `commit` are therefore not a duplicate: the later one's budget is current.
  - (d) The engagement is read first in each pass
    (`client-feed/src/client_feed/engagement.py::read_engagement`). A missing, unreadable (a symlink
    included) or invalid file, or a `delivered_commit` the repository does not hold, is named
    `engagement` in `degraded` (`client-feed/src/client_feed/engagement.py::EngagementUnusable`).
    Reveals are then disabled for that pass: the inbox is not read, its offset is held, nothing is
    followed, and a new tree's budget uses §4.12's null `delivered_commit` rule. When the engagement
    is usable but the reveal unit fails (a git error, say), that unit's source is named in
    `degraded`, the inbox offset and `reveals_head` are held, and a new tree keeps the state's last
    budget instead of switching to the null rule for one pass
    (`client-feed/src/client_feed/publisher.py::_new_tree_budget`). A `project_id` other
    than the policy's is the refusal
    `client-feed/src/client_feed/engagement.py::EngagementProjectMismatch`, a `FeedError`: the pass
    stops before anything is appended, and the CLI exits 1. `heartbeat.engagement_version` echoes
    only a usable engagement's version, and is null otherwise
    (`client-feed/src/client_feed/publisher.py::run_pass`).
  - (e) An inbox that cannot be read (unreadable, a symlink, or shorter than the cursor's offset) is
    `inbox` in `degraded`, and its offset is held
    (`client-feed/src/client_feed/inbox.py::InboxUnusable`). A line that is not a reveal request is
    rejected, never raised: it is counted under `inbox`, and the tail moves past it with the requests
    around it (`client-feed/src/client_feed/publisher.py::_inbox_unit`).
  - (f) Follow-head walks main's first-parent commits after `Cursor.reveals_head` through head,
    oldest first, in one `git log --first-parent -M --raw`
    (`client-feed/src/client_feed/sources/git.py::first_parent_changes`,
    `client-feed/src/client_feed/reveals.py::_tracks`). A revealed file whose blob changed is
    re-emitted at head, and a renamed one is emitted at its new path with `renamed_from`. Its reveal
    ends with a `file_revealed` at its revealed path with `deleted: true` when the path is deleted,
    or when at head it is no longer revealable. A followed path is never refused for budget. A path
    recreated after its delete is not followed, and a change reverted within the walk emits nothing
    (`client-feed/src/client_feed/reveals.py::_follow_events`). When `reveals_head` is null or no
    longer an ancestor of head, each revealed file is diffed from its own `commit` to head instead.
    In one pass the ends come first, then the re-emits, then the inbox's answers in inbox order.
  - (g) The revealed set's only home is `state.revealed` (path, commit, lines), folded from
    `file_revealed` (`client-feed/src/client_feed/fold.py::_apply_file_revealed`). `cursor.json`
    holds `reveals_head`, `inbox_offset` and `added_line_counts`, and no per-path blob shas
    (`client-feed/src/client_feed/feed_io.py::Cursor`). `added_line_counts` caches (b)'s counted
    added lines by old then new blob sha, valid only under the `max_line_bytes` they were counted
    with, as `line_counts` does for whole blobs. A pair already counted is not diffed again, and
    only the pairs of the latest count are kept
    (`client-feed/src/client_feed/budget.py::AddedLineCounts`). All three move only when the reveal
    unit runs and succeeds; an inbox read holding no request moves its offset alone
    (`client-feed/src/client_feed/publisher.py::_advanced`). After a crash between the append and
    the cursor, an answer whose identity (what it says, without its envelope) is in the uncommitted
    tail is consumed there and not emitted again
    (`client-feed/src/client_feed/reveals.py::answer_of`).
  - (h) The portal's `in-v1` fixture is vendored byte for byte at
    `client-feed/tests/fixtures/in-v1` from scoping-intake
    `f5cea40d8354190c9fa0cce260b2e69adc79ea3a`. Its `delivered_commit` is a fixture sha that no test
    repository holds, so read as vendored it disables reveals. A test that needs a usable engagement
    sets it to null after reading the vendored file (`client-feed/tests/test_in_v1_fixture.py`).

- 2026-10-25, from task 7484. These bind ζ (6973) and the portal:
  - (a) A request that adds nothing to `spent_lines` is granted whatever `spent_lines` is: a path
    outside the counted set, or one already revealed. Any other request must fit what is left of
    the budget (`client-feed/src/client_feed/budget.py::CountedSet.grantable`). Follow-head never
    refuses for budget (δ entry (f)), so `spent_lines` can exceed `budget_lines`. A free file is
    still granted then, and B14's `spent_lines` is unchanged by its grant.
  - (b) A granted request for a path already in the revealed set emits no event, so a portal retry
    or a double-click appends nothing. Within a pass, follow-head runs before the inbox is answered,
    and it has already re-emitted at head every revealed file whose blob changed
    (`client-feed/src/client_feed/reveals.py::_follow_events`). The feed therefore already holds
    head's content for every revealed path. The request is still judged first, so a refusal is
    emitted exactly as before (`client-feed/src/client_feed/reveals.py::_answer`). A consumer that
    tracks pending requests therefore settles one when its path is in `state.revealed` or a
    `reveal_refused` answers it, and never waits for a `file_revealed` per request. An explicit
    `already_revealed` acknowledgement would need a contract revision.
  - (c) git output that does not parse is a `GitError` of the command that wrote it, and it
    degrades `git` like any other git failure
    (`client-feed/src/client_feed/sources/git.py::_parsing`). Every `git cat-file --batch` read is
    bounded by `GIT_TIMEOUT_SECS`, as every one-shot git call is. The bound applies per read, so the
    caller's work between reads never counts against it. A read that outlasts it kills the session
    and fails as `timed out` (`client-feed/src/client_feed/sources/git.py::blob_reader`).

- 2026-10-29, from η (task 6974). These bind σ2 and the portal's deploy/ README:
  - (a) The publisher unit is `client-feed@<name>.service`, rendered from
    `scripts/client-feed.service.template`. Its ExecStart is `uv run --no-sync --project client-feed
    client-feed publish --config %h/.config/client-feed/%i.yaml`, so an instance's config is
    `~/.config/client-feed/<name>.yaml`. Operating it: `OPERATIONS.md` §"Client feed".
  - (b) It is rendered through `scripts/render_systemd_unit.py::UNITS` entry `client-feed`, whose
    family is None: it has no Environment= policy and preserves nothing. `scripts/setup-host.sh`
    section 7a renders it, runs `daemon-reload` on success, and never enables or starts an
    instance.
  - (c) setup-host section 3 now runs one `uv sync --all-packages` from the repo root, in place of
    its per-member `uv sync` loop. A member-scoped sync is exact and prunes the other members from
    the one shared `.venv`, so the loop left only the last member's closure installed (measured
    2026-10-29, uv 0.11.6, in a throwaway two-member workspace: after `uv sync --all-packages`,
    `cd b && uv sync` left only `b`). Appending `client-feed` to the loop, as the task text read,
    would have pruned the dashboard.
  - (d) The worked policy for this repository is `client-feed/examples/dark-factory.yaml`,
    validated by `client-feed/tests/test_policy_example.py`.
  - (e) A clean stop exits 0 through `uv run`, so the unit carries no `SuccessExitStatus=`. The
    publisher's handler only sets its stop Event, and the process exits once the pass in progress
    completes (`client-feed/src/client_feed/cli.py::_publish_until_signalled`). A `uv run` child of
    that shape gave uv rc 0 in 6 of 6 trials, whether SIGTERM reached uv alone or the whole process
    group, as systemd's control-group kill sends it (measured 2026-10-29, uv 0.11.6). The shape
    matters: in one trial, a child that called `sys.exit(0)` inside its handler gave 143 when the
    whole group was signalled. A likely cause, not verified: uv forwards the SIGTERM it receives,
    and that duplicate lands after the interpreter has dropped the handler. Re-measure before
    changing the handler.
  - (f) For σ2: the filer's template is another `UNITS` entry with its own explicit `--unit` call
    site in setup-host. `tests/scripts/test_render_systemd_unit.py::test_setup_host_renders_each_unit_from_the_template_its_spec_names`
    pairs each registry key with that call site.

## 13. Open questions (tactical)

1. **Tree-snapshot cost on large repos.** Suggested: snapshot only on head change; blob line counts
   cached by sha in `cursor.json`; last-change dates from the commits walked since the previous
   snapshot. Decide in β2/γ.
2. **One unit per project or one unit with a list.** **Decided by η (6974):** one instance per
   project, `client-feed@<name>.service`, `--config %h/.config/client-feed/<name>.yaml`.
3. **Default globs** for visible, test, docs and generated sets. Suggested as in §7.2 and §7.6, with
   `visible_globs` having no default (a policy must name it). Decide in β2.
4. **`excerpt` rule for PRDs** whose first paragraph is a status line. **Decided by β1 (6968):** the
   first paragraph after the first `## ` heading, skipping subheadings and fenced blocks; the title is
   the first H1, else the path; both are path-redacted
   (`client-feed/src/client_feed/sources/prds.py::summarize_prd`).
5. **Where the plugin writes the map when no `--junitxml` is given.** Suggested: nowhere; the plugin is
   inert without a junit target. Decide in ε.


## 14. Revision 2026-10-23: the client defect bridge

Authored 2026-10-22/23 for gate task 7062 (esc-7062-1; `plans/defect-tests-prd.md` ν, decision 6), with
Leo's rulings taken in the session (§14.2), and attacked by an opus critic seat before commit (18
findings, all folded in). **Code anchors** in this section are verified against main `6885b119db`
(2026-10-23). Cite-by-symbol; re-locate at implementation time. Approach B+H, as for the rest of this
PRD: the seam crosses two repositories, carries untrusted outside text, and touches the isolation
invariant.

### 14.1 Goal and consumers

A client who finds a defect reports it on the portal against a declared criterion of the suite.
Within a minute of the portal's `defect_reported` line:

- a task in that client's project carries one declared defect with `origin: client` and ref
  `portal:<project_id>:<block>:<report>`, status `open`;
- its description is the portal's restatement of the report, screened for prompt injection, or, when
  the report was quarantined, typed facts only;
- an always-escalates human gate that the fix depends on is committed. It reaches Leo's L2 queue when
  the client project's orchestrator dispatches it (§14.2 decision 4). Every report is gated until
  decision 5's flip.

The feed then carries `defect_filed {ref, block, report, task_id, gated}`, and the portal shows the
report as filed. Leo reads the original in the portal, checks or writes the description, records his
verdict, and resumes the gate. The factory then fixes it under the defect-tests rule: a test proven to
catch it, kept.

When the fix lands, the feed's `merge_landed.task_ids` names the fix task. The portal then marks that
report fixed, as actor `system`: the merge presses "Fixed" (decision 13). A block may hold any number
of open reports at once. The client's own words never leave the portal.

What an operator sees:

- `lookup_defect_ref(project_root=R, ref="portal:<id>:<b>:<n>")` returns one live carrier.
- `get_task` on that carrier shows `metadata.defects[0]` with `origin: client`.
- A report not filed within `filer_grace_seconds` makes the publisher's heartbeat name `filer` in
  `degraded`, with the ref.

| Mechanism | Consumer |
|---|---|
| `defect_reported` inbox line (§14.3) | the filer (§14.4); the publisher's unfiled check (§14.3); written by the portal (portal PRD, amended §14.7) |
| The filer, `client-feed file-defects` (§14.4) | the client defect window: the portal's `defect_reported` stage, whose "Fixed" is Leo's (portal PRD decision 5), and the LDE sheet's 90-day promise; Leo's L2 queue (the gate); defect-tests' proof and presence check, which act on the declared defect |
| `defect_filed` feed event and the `filer` degraded source (§14.3) | the portal's project and admin pages (filed / not filed per report); the portal's per-report "Fixed", pressed when `merge_landed.task_ids` names the filed fix (decision 13); defect-tests ι (7057), which builds its typed defect projection on the same ref reader |
| Policy `filer:` block and `filer.auto_dispatch` (§14.3) | the filer; gate σ5 flips `auto_dispatch` |

### 14.2 Resolved design decisions

1. **The premise (Leo, 2026-10-16, "Feed amendment, new writer").** The bridge is an inbox line plus a
   narrowly privileged filer outside the publisher's read-only boundary (§8). Not re-litigated here.
2. **Raw client text never leaves the portal (Leo, 2026-10-23).**
   - Its home is the portal's record: the `DefectReport` input on the block's `defect_reported`
     transition (`scoping_intake/engagement.py::DefectReport`, `::Block.history`). Leo reads and acts
     on it from the admin page.
   - The inbox line has no field for the raw text (`extra='forbid'`). Its restated fields are
     model-written. The portal quarantines a restatement that repeats the report verbatim beyond a
     threshold (`not_restatable`, P1b), so the restater cannot carry the raw text across by echoing
     it.
   - Raw text is written into no task field, prompt, transcript, memory or embedding.
   - The reasons:
     - Every dispatched agent runs as Leo's user and can read the whole filesystem. The OS sandbox
       contains writes only (`plans/os-sandbox-worktree-containment-prd.md`, Out of scope).
     - Agents have the network, and fused-memory binds no caller to a project.
     - Task text, once filed, reaches about ten durable sinks: transcripts, recon memories, the
       curator's Qdrant corpus and its OpenAI embedder, the dashboard, escalations, and others.
     - Injected text in one client's report could therefore reach every client.
3. **A tool-less model restates each report on the portal side (Leo, 2026-10-23).**
   - The model is capable but has no tools. Its prompt is written to watch for prompt injection. For
     each report it gives exactly one of two outcomes:
     - a restatement in three fields (observed, expected, steps);
     - a quarantine with typed reasons.
   - Any restater failure is a quarantine (`restater_declined`, `restater_failed`). Unscreened text is
     never filed.
   - The restater runs on the portal's real-client backend: the `api` backend in
     `scoping_intake/cli.py::BACKENDS`, with structured output. The `claude-code` backend's
     implementation of it returns `restater_declined` without calling the model, because that
     backend's session history keeps every input (`scoping_intake/claude_code.py` module docstring).
   - The portal owns the prompt and the model. The line records `restater.prompt_version`.
   - A kept report is never dropped. The portal's ordering and its reconciler (§14.7) guarantee that
     every kept `DefectReport` transition gets its line, or is shown to Leo with a typed reason why it
     has none.
4. **Every report is filed with a gate (Leo, 2026-10-23).**
   - The filer files two tasks: the fix task, which carries the declared defect, and an
     always-escalates human gate. The fix depends on the gate, and both are committed `pending`.
   - The gate is born at L2 when the client project's orchestrator dispatches it, on that project's
     escalation queue.
   - Leo then:
     - reads the report in the portal;
     - compares it with the fix task's description, editing the description with `update_task` or
       writing it when the report was quarantined;
     - records `x_client_defect_review: accepted | corrected | written` on the gate;
     - resumes the gate, which releases the fix.
   - Who owns the hold, and how long it may last, come from the existing gate ladder (INV-7). Nothing
     is left `deferred`.
5. **Unquarantined reports go ungated only after a soak (Leo, 2026-10-23).**
   - `filer.auto_dispatch` is shipped `false`. With it `true`, an unquarantined restatement files the
     fix `pending` with no gate. A quarantined report, or one filed by decision 7's fallback, is always
     gated.
   - Gate σ5 flips the flag. It counts the gates' recorded `x_client_defect_review` verdicts with
     `find_tasks_by_metadata`, so the soak is a recorded fact, not recollection.
   - This is the shadow-then-enforce pattern of `defect_tests.mode` (`plans/defect-tests-prd.md`
     decision 3).
6. **Filing uses `planning_mode` only.**
   - Every `submit_task` call passes `planning_mode=True`, and the batch is then released with
     `commit_planning`.
   - The curator path is never used. It can drop a candidate or combine it into another task, and
     both discard `metadata.defects`
     (`fused-memory/src/fused_memory/middleware/task_interceptor.py::_dispatch_ticket_decision`,
     `::_execute_combine`; θ1, task 7051, is pending).
   - The declaration is made at creation, through the creation floor
     (`fused-memory/src/fused_memory/backends/defect_records.py::require_creation_floor`).
   - `declare_defect` is not needed: each report has its own ref, so a repeat is a new report, never a
     sighting.
7. **The ref is the idempotency key; filing converges and always reaches Leo.**
   - One filer runs per policy. It holds an exclusive, non-blocking `flock` on a file in its state
     directory for the life of the process, `--once` included, and exits with a typed message when the
     lock is held.
   - Before filing, it asks `lookup_defect_ref`; a carrier means the report was filed. A pass that
     crashed partway is completed by the next one from what the store holds:
     - the gate is found by `metadata.x_client_defect_gate_for == ref` (`find_tasks_by_metadata`,
       every status);
     - the fix is found by the ref;
     - missing edges and commits are added.
   - The end state is one fix and at most one gate per ref, with no task left `deferred`. The filer's
     inbox cursor is an optimisation; correctness never depends on it.
   - A failed call is classified by a closed table keyed on two things: the
     `shared/src/shared/mcp_session.py::McpCallError` kind, and the reply's own code (its `error_type`,
     or for the defect refusals its `error`).
     - Retried with backoff:
       - transport and HTTP-status failures;
       - a backlog refusal.

       Such a report does not hold up later ones; it stays unfiled and the publisher names it.
     - Not retried, but filed in the quarantined shape (the gated fallback):
       - a `TOOL_ERROR`, which is how the MCP-markup boundary guard refuses. The guard deliberately
         over-reports and rejects a restatement quoting `</description>` (`shared/src/shared/toolcall_markup.py`;
         `fused-memory/src/fused_memory/server/main.py::_install_tool_dispatch_guards`);
       - a reply carrying `ValidationError`;
       - a reply carrying `defect_creation_refused`.
     - Any other code is retried and named under `filer`, and falls back after 5 attempts. This is the
       storm escape.
   - The fallback files typed facts only, always gated, with `filer_fallback: {kind, code}` in
     `x_client_defect` and in the gate's description. Every report therefore reaches Leo.
8. **The publisher acknowledges filings; the filer writes no file.** The publisher already reads I's
   inbox and R's `tasks.db` (§8).
   - A **carrier** is a task whose `metadata.defects` holds the ref and whose status is not
     `deferred`. A planning-born row, written before its edge and commit, is not yet a filing.
   - `defect_filed` is keyed by `ref`. It is emitted when a carrier first appears, and emitted again
     whenever its `(task_id, gated)` changes.
   - For the unfiled check, the publisher keeps its own position in the inbox:
     `cursor.json.defect_scan_offset`. This is separate from δ's `inbox_offset`, and it moves on every
     pass where the inbox is readable, even when reveals are disabled (§12, 2026-10-22 (d)).
   - Reports seen without a carrier wait in `cursor.json.pending_reports: {ref: line_at}`. The time is
     the line's own `at`, stamped by the portal, so a re-scan after a crash or outage yields the same
     value. An entry is pruned once a carrier exists.
   - `Cursor` is `extra='forbid'`, so 1.0 code cannot read a cursor that 1.1 code has written.
     Rolling back means deleting `defect_scan_offset` and `pending_reports` first.
   - The heartbeat names `filer` in `degraded`, with the refs, while any entry is older than
     `filer_grace_seconds`.
   - The filer writes nothing under O, I or R, which keeps one writer per directory (§4.1). A dead
     filer is distinct from "no reports" (INV-6, INV-13).
9. **The filer is narrow in code shape and in process.**
   - It runs as its own process and unit, under its own invariant (§14.5).
   - It calls a closed set of six fused-memory tools, with `project_root` fixed to the policy's root.
   - The publisher modules and the filer do not import each other. Neither the ref rule nor the MCP
     client enters a vendorable module (`client-feed/tests/test_import_boundary.py`).
   - This narrowness is a property of the filer's code, not an enforced boundary. fused-memory has no
     caller authentication and listens on all interfaces (`fused-memory/config/config.yaml`,
     `server.host`).
   - The boundary that matters is that the portal can write tasks only through the inbox. It holds
     only when both of these are true:
     - the portal runs as its dedicated account;
     - that account cannot open a TCP connection to any local or loopback address except its own
       listeners.
   - The portal's deploy owns that rule (Leo, 2026-10-23; §14.7). It is default-deny, not a port list,
     because the factory's MCP ports are an open set: fused-memory, the recon queue, one escalation
     server per project, and the operator dashboard's unauthenticated write routes (§2). Gate σ4
     probes it before the filer is enabled for a live client.
10. **Every text field on a filed task is written by the filer.**
    - The fix task's title is `Client defect <ref> against criterion <criterion_id>`. Titles reach
      every prompt and are published to the client (§4.6).
    - The gate's title is `Human gate: client defect <ref>` for every outcome. It is published too, so
      it never says "quarantined".
    - The defect `summary` is `Client report <n> on block <b> against criterion <criterion_id>`. The
      client is never shown it (defect-tests decision 18). Defect-tests γ renders it into briefings.
    - The fix description is a fixed template. It holds the criterion id; a pointer to
      `acceptance/criteria.yaml` in the project; and the restatement's three fields under one heading
      naming their provenance ("restated by the portal from the client's report"). When the report is
      quarantined, or filed by the fallback, the template holds the typed reasons instead, plus "Leo
      writes this description at the gate".
    - The gate's description holds typed facts only: the ref, the fix task id, the outcome kind and
      reasons, the portal page to open, and the verdict to record. The restatement lives on the fix
      task alone (INV-9).
11. **Identity comes from the policy, not from the line.**
    - `<project_id>` is the policy's `project_id`, which equals the portal's `FeedBinding.project_id`
      (`scoping_intake/clients.py::FeedBinding`).
    - `<block>` is the block's number.
    - `<report>` is the count of `DefectReport` transitions in that block's history, the new one
      included. It needs no new portal state.
    - The filer and the publisher both mint the ref through
      `client_feed/defect_ref.py::portal_ref(project_id, block, report)`. This module is non-vendorable
      and calls `shared/src/shared/defects.py::mint_portal_ref`.
    - The portal never parses a ref. It correlates `defect_filed` to its report by the event's typed
      `block` and `report`.
12. **Criterion ids have one rule, and only declared ones cross.**
    - `CriterionId` (§14.3) becomes the id rule of `acceptance/criteria.yaml`: ε (6972) adopts it for
      `client_feed.contract.Criterion.id`, amended at decompose.
    - The portal offers, and writes into a line, only an id the feed has declared in
      `criteria_declared`.
    - A report kept while no criteria are declared gets no line. The admin page names it with a typed
      reason, and the reconciler writes the line once its id is declared. So an id that is legal in
      `criteria.yaml` but rejected by the inbox contract cannot occur.

13. **The merge presses "Fixed", per report, and a block holds any number of open reports (Leo,
    2026-10-23, confirming task 7005's constraint of 2026-10-15).**
    - The portal (P3) joins `merge_landed.task_ids` (§7.2, train members included) with
      `defects_filed[ref].task_id`. It marks the report `(block, report)` fixed as actor `system`,
      through an idempotent per-report record operation: a replayed `merge_landed` changes nothing.
      Only the operation that closes the last open report moves the stage.
    - When a block's last open report is fixed, it leaves `defect_reported`:
      - a paid block goes back to `defect_window`;
      - an unpaid block is re-demonstrated, with a review window of the original length starting on
        the landing date.
    - Leo can still mark a report fixed by hand, for instance after cancelling its fix as not a defect.
    - Under defect-tests `shadow` mode, a fix can land without a proven test. "Fixed" then means
      "merged", and enforce mode adds the proof.
    - A client may report again while other reports are open. Each report gets its own number, ref, fix
      and gate.
    - The factory side needs nothing beyond σ1b's `defect_filed` and γ's `merge_landed`. Task 7005,
      the 2026-10-15 milestone gate for this same bridge, is superseded by this section and cancelled.

### 14.3 Contract

All models live in `client_feed/contract.py` (§7, `extra='forbid'`, frozen), except the ref rule,
which is in `client_feed/defect_ref.py`.

**`in/inbox.jsonl` gains a second line type.** The inbox model becomes a union discriminated on
`type`: `RevealRequested | DefectReported`. `parse_inbox_line` returns the union. δ's reveal path
handles only `RevealRequested`.

```
DefectReported   {seq: Seq, at: UtcTimestamp, type: "defect_reported", block: int ≥ 1, report: int ≥ 1,
                  criterion_id: CriterionId, restater: {prompt_version: Token},
                  outcome: Restated | Quarantined}                      # discriminated on outcome.kind
Restated         {kind: "restated", observed: str 1..2000, expected: str 1..2000,
                  steps: [str 1..500] (0..20 items)}
Quarantined      {kind: "quarantined", reasons: [QuarantineReason] (≥ 1, unique)}
QuarantineReason = suspected_injection | not_restatable | restater_declined | restater_failed
CriterionId      = ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$     # also Criterion.id in acceptance/criteria.yaml (ε)
Token            = ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$

client_feed/defect_ref.py:  portal_ref(project_id, block, report) -> str   # shared.defects.mint_portal_ref(project_id, str(block), report)
```

A line that fails the contract is rejected and counted under `inbox` (§12, 2026-10-22 (e)). A
`defect_reported` line that validates is never counted as rejected.

**Feed (FEED_VERSION "1.1"): one new event type and one new `Source` member.**

| type | fields | source | emitted when |
|---|---|---|---|
| `defect_filed` | `ref`, `block`, `report`, `task_id`, `gated: bool` | carriers (decision 8) in `tasks.db` whose `metadata.defects[].ref` is a `portal:<policy project_id>:` ref; `gated` = the fix depends on a task carrying `x_client_defect_gate_for == ref` | a carrier first appears, or its `(task_id, gated)` changes |

- `state.json` gains `defects_filed: {ref: {block, report, task_id, gated}}`. Natural key: `ref`.
- `Source` gains `filer`. `heartbeat.degraded` names it with the reason
  `N defect report(s) unfiled past <filer_grace_seconds>s: <ref>, …`.
- **Compatibility.** A new type and a new state key are minor changes: the portal ignores both until
  it reads them. A new `Source` member is not minor for a reader that validates `Source` as a closed
  enum, and the portal's `scoping_intake/feed.py::FeedSource` does. P1a therefore adds `filer` to
  `FeedSource`.
- **Ordering.** The publisher names `filer` only for a `defect_reported` line. Only P1b writes such
  lines, and P1b depends on P1a, so P1a is deployed first. This holds while the portal is the inbox's
  only writer (§4.1).

**The ref reader** (`client_feed/sources/tasks.py`) adds to §4 decision 6's allowlist exactly two
fields:

- `metadata.defects[].ref` values that are `portal:<policy project_id>:` refs;
- `metadata.x_client_defect_gate_for`.

Refs of other origins (`review:`, `followup:`, `human:`, …) and other projects' portal refs are never
selected (B3, extended). Defect-tests ι (7057) extends this reader into its typed `defects`
projection; it does not add a second one.

**Policy (§7.6) gains (all in σ1a):** `filer_grace_seconds: 600` (read by the publisher) and

```
filer: {fused_memory_url: "http://127.0.0.1:8002", state_dir: str, auto_dispatch: false,
        interval_seconds: 10, fix_priority: "high"}
```

`state_dir` is the filer's own directory, owned by Leo's account. It must be none of O, I or R, and
must lie under none of them. `policy.py` refuses a policy that breaks this.

**The filer's closed tool set:**

- The tools: `lookup_defect_ref`, `find_tasks_by_metadata`, `get_task`, `submit_task` (always
  `planning_mode=True`), `add_dependency` and `commit_planning`.
- The client: `shared/src/shared/mcp_session.py::SyncMcpSession`. σ2 makes its `httpx` client
  injectable, so the filer passes `trust_env=False` and no proxy setting can redirect it. Every call
  carries `project_root=R`.
- The filer calls no other tool. In particular it never calls `update_task`, `set_task_status`,
  `remove_task`, `declare_defect`, any memory tool or any escalation tool.

**The tasks it files:**

| | fix task | gate (when gated) |
|---|---|---|
| `task_kind` | `normal` | `deterministic`, `metadata.always_escalates: true` |
| title | `Client defect <ref> against criterion <criterion_id>` | `Human gate: client defect <ref>` |
| description | the template of decision 10 | typed facts: ref, fix task id, outcome kind and reasons (or `filer_fallback`), the portal page to open, "edit task <fix> if needed, record `x_client_defect_review`, then resume" |
| metadata | `defects: [{ref, origin: client, summary, declared_at, declared_by: "client-defect-filer", status: open, sightings: 1}]`, `source: "client-defect-filer"`, `x_client_defect: {block, report, criterion_id, outcome_kind, restater_prompt_version, filer_fallback?}` | `source: "client-defect-filer"`, `x_client_defect_gate_for: <ref>`; later `x_client_defect_review` (Leo) |
| priority | `filer.fix_priority` | same |
| dependencies | the gate | none |

### 14.4 Sketch

```
portal (LDE account)                                    factory (Leo's account)
report_defect: keep the move (lock) ─► restate (api backend, no tools; outside the lock)
   │  record keeps the raw text (home)        └─► append the line (lock); the reconciler fills any gap
   └─► in/inbox.jsonl  defect_reported{block, report, criterion_id, restated | quarantined}
                                   │
                    client-feed file-defects (σ2) ─lookup / submit(planning) / add_dependency / commit─►  fused-memory
                                   │                                                                       │
                    client-feed publish reads in/ + tasks.db ◄──────────────────────────────── tasks.db ◄───┘
                                   └─► out/feed.jsonl  defect_filed{ref, block, report, task_id, gated} | degraded: filer
```

`client_feed/filer.py` holds one pass:

1. Tail the inbox from the filer's own cursor.
2. For each `defect_reported` line, converge its filing (decision 7).
3. Advance the cursor past the lines that are filed.

The CLI has three forms:

- `client-feed file-defects --config C` is the loop.
- `--once` runs one pass.
- `--check` writes nothing. It takes the lock, handshakes with U, confirms that U lists the six tools,
  calls `lookup_defect_ref` on a ref the portal can never mint (block token `check`), and prints the
  calls one pass would make over the inbox. It is σ4's install check.

`client_feed/filer.py` uses `client_feed.contract`, `client_feed.defect_ref`, `client_feed.policy` and
`shared.mcp_session`. It imports nothing from `client_feed.publisher`, `client_feed.sources` or
`client_feed.feed_io`. §3's dependency sentence gains `shared.defects` (through `defect_ref.py`) and
`shared.mcp_session` (the filer only).

### 14.5 Filer isolation invariant (normative)

For a filer configured with policy P (root R, inbox directory I, state directory S, fused-memory URL U):

- It opens P, `I/inbox.jsonl` for reading, and S. Python's own and its libraries' reads under
  `sys.prefix`, site-packages and the CA bundle are outside this rule.
- It runs no subprocess. Its only connections are to U, even with proxy environment variables set.
- It calls only the closed tool set (§14.3), every call with `project_root=R`, and `submit_task` only
  with `planning_mode=True`.
- It writes nothing under O, I or R.
- Everything it submits is a function of the `defect_reported` lines in I and of P. No other text
  reaches a task.

§8 is unchanged. The publisher still opens nothing beyond R, O and I. Its new reads, the
`defect_reported` lines and the two allowlisted fields, are inside them.

### 14.6 Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B23 | Inbox contract, both ways | the portal's committed `tests/fixtures/in-v2/` (a restated and a quarantined `defect_reported` line between two `reveal_requested` lines), vendored byte for byte | each line validates under `client_feed.contract`. A `defect_reported` line is rejected in each of these cases: it carries an extra `text` field; its `observed` is empty; it has both outcomes or neither; it has an unknown reason; its criterion id holds a `:`. The portal's suite validates its own writer output against the same fixture (P21's rule) |
| B24 | One ref rule | policy `project_id: acme-widgets`; block 2, report 3 | `defect_ref.portal_ref` gives `portal:acme-widgets:2:3`. The filer and the publisher get it through that one function. The import-boundary test stays green with `contract` vendorable |
| B25 | Filed, gated | one restated line; `auto_dispatch: false` | one fix task with one `defects` entry `{origin: client, status: open, sightings: 1}`, the filer-authored title, and the restatement under the provenance heading; one gate, `deterministic` and `always_escalates`, titled `Human gate: client defect <ref>`; the fix depends on the gate; both `pending`; every `submit_task` passes `planning_mode=True` |
| B26 | Quarantined | one quarantined line; `auto_dispatch: true` | the fix description holds no restated field and names the reasons; a gate is filed regardless |
| B27 | Auto-dispatch | one restated line; `auto_dispatch: true` | the fix is `pending` with no gate |
| B28 | Idempotent, single instance | the same inbox run twice; again with S's cursor deleted; and a second filer started while the first holds the lock | one fix and one gate in total; the second filer exits with its typed lock message |
| B29 | Crash convergence | the filer killed after the gate's submit, after the fix's submit, and before the commit | each following pass ends with one fix and one gate, the edge in place, both `pending`, and no `deferred` task |
| B30 | Narrow by construction | a fake MCP server that records calls; the audit-hook recorder `client-feed/tests/_audit_hook.py`; proxy environment variables set | tool names ⊆ the closed set; every `project_root == R`; outside `sys.prefix`, site-packages and the CA bundle, opens only P, `I/inbox.jsonl` and S; connects only U; no subprocess; nothing written under O, I or R; an import test shows the filer and the publisher modules do not import each other |
| B31 | No raw text | owned by the portal: its P-row P30 (P1b), a raw report carrying `CANARY-RAW-…` | no line under `in/` holds it. On the factory side, B23's `extra` rejection is the guarantee: no line can carry a text field |
| B32 | Acknowledgement | a carrier for the ref in `tasks.db`; then its gate edge added; then a planning-born `deferred` carrier for a second ref | one `defect_filed {ref, block, report, task_id, gated: false}`, then one with `gated: true`; nothing for the deferred carrier; no duplicate after a restart (B18's rules) |
| B33 | Unfiled is loud | a `defect_reported` line read on a pass before its grace expires, with no carrier; then the engagement made unusable; then only a `deferred` carrier | the heartbeat names `filer` with the ref once the grace passes, on every later pass in all three states; it clears once a non-deferred carrier appears |
| B34 | Reveals unaffected | an inbox interleaving both line types | reveal answers as in B12–B16; the filer ignores `reveal_requested`; no valid `defect_reported` line is counted under `inbox` |
| B35 | Validation fallback | a restatement whose `observed` contains `</description>`; the fake returns the guard's real shape (an `isError` reply), and B36 repeats it against the real guard | the restated fix is refused; the filer files the quarantined shape with `filer_fallback` and a gate, even with `auto_dispatch: true` |
| B36 | Real tool layer end to end | the server built as the fixture in `fused-memory/tests/test_defect_write_tools.py` builds it (`SqliteTaskBackend`, `TaskInterceptor`, `create_mcp_server`), plus `fused_memory.server.main._install_tool_dispatch_guards`, its HTTP app served by uvicorn in a background thread on a free 127.0.0.1 port; the filer pointed at that URL; the vendored `in-v2` inbox | B25, B29 and B35 hold on the rows that store wrote and against the real guard; the publisher, run over that same `tasks.db`, emits `defect_filed` for each filed ref |

B3 is extended with a task carrying a `review:` ref and a foreign project's `portal:` ref: neither
reaches O.

### 14.7 Cross-PRD relationship (additions to §6)

| Seam | Direction | Mechanism | Owner | Status |
|---|---|---|---|---|
| `defect_reported` inbox line | the portal writes it; this PRD defines and reads it | `client_feed.contract.DefectReported`; the portal's `in-v2` fixture | **this PRD (σ1a)** for the contract; the portal (P1a, P1b) for the writer, the restater, the fixture and the reconciler | portal PRD amended and portal tasks filed 2026-10-23 |
| Restater prompt, model, quarantine judgement and verbatim-overlap check | portal only | `scoping_intake` `ScopingModel`, api backend | portal PRD (P1b) | filed |
| `defect_filed`, `defects_filed`, `Source.filer` | this PRD produces; the portal reads and renders | feed 1.1 | this PRD (σ1b); the portal (P1a reads, P1b renders) | filed |
| "Fixed" pressed by the merge, per report; concurrent open reports | this PRD produces the join keys; the portal consumes them | `defects_filed[ref].task_id` (σ1b) with `merge_landed.task_ids` (γ) | portal PRD (P3): its edge table, `Block`'s open reports, the join | filed |
| The portal account's TCP to local and loopback addresses | the portal's deploy | default-deny for the portal uid except its own listeners, plus the deploy's own check | portal PRD (P2) | filed; σ4 probes it |
| Criterion id rule | this PRD defines; ε adopts; the portal checks | `CriterionId`; `criteria_declared` | this PRD (σ1a); ε (6972) amended to use it and to depend on σ1a | amended at decompose |
| Declared client defect → proof, presence check, display | this PRD files the declaration; defect-tests acts on it | `metadata.defects`, origin `client` | defect-tests (β1…η, ι) | ι (7057) amended at decompose to build on σ1b's ref reader and to depend on σ1b; if ι adds a feed type it bumps FEED_VERSION to 1.2 |
| Audit-hook recorder | σ2 creates it; ζ uses it | `client-feed/tests/_audit_hook.py` | this PRD (σ2) | ζ (6973) amended at decompose to import it and to depend on σ2 |
| Interim hand filing (θ2's `client` recipe) | superseded per project once σ4 enables the filer for it | `docs/task-authoring.md` | defect-tests θ2 (7056) | unchanged |

The portal PRD's amendment (committed with this revision) changes these places:

- Its inbox contract line gains `defect_reported`, carrying the report id (`block`, `report`) and the
  criterion id. The line is written only from the restater's output, and only with a declared
  criterion id.
- Its `report_defect` gains an order:
  1. keep the move under the lock;
  2. restate outside the lock;
  3. append the line under the lock;
  4. a reconciler appends the line for any kept `DefectReport` transition that has none, idempotent by
     `(block, report)`, and the admin page names any that cannot yet be written.
- Its feed reader gains `FeedSource.filer`, `defect_filed` and `defects_filed`.
- Its pages gain filed / not filed per report.
- Its state machine lets a block hold many open reports, adding a `defect_reported` → `defect_reported`
  edge for another report, and lets the merge press "Fixed" per report (decision 13; P3).
- Its deploy gains the default-deny rule.
- Its P-rows gain P29 to P32.

### 14.8 Decomposition (additions to §10)

Pre-conditions:

- δ (6971) has landed.
- η (6974) defines the unit template and the "Client feed" section of `OPERATIONS.md`, which σ2
  extends.
- Portal task 55 (the report-defect route) is done.

| Label | Task | Modules | Observable signal | Prereqs |
|---|---|---|---|---|
| σ1a | Inbox union with `defect_reported`, `CriterionId`, and the ref rule | `client_feed/contract.py` (`DefectReported`, `Restated`, `Quarantined`, `CriterionId`, the union, `parse_inbox_line`), `client_feed/inbox.py` (`InboxRead` over the union; δ's reveal path takes `RevealRequested` only), `client_feed/defect_ref.py` (new), `client_feed/policy.py` (`filer_grace_seconds` and the `filer:` block's model with its O/I/R checks), `client-feed/tests/test_import_boundary.py` (`shared.defects` allowed), a factory fixture `client-feed/tests/fixtures/in-v2-factory/`, tests | Intermediate: unlocks σ1b, σ2 and ε's adoption of `CriterionId`. B24; B23's rejection cases over the factory fixture; a policy naming its `state_dir` under O is refused; δ's reveal tests stay green | — |
| σ1b | The acknowledgement: `defect_filed`, `defects_filed`, `Source.filer`, the ref reader, the unfiled check, feed 1.1 | `client_feed/contract.py` (`DefectFiled`, `Source.filer`, `FEED_VERSION`), `client_feed/sources/tasks.py` (the ref reader), `client_feed/feed_io.py` (`Cursor.defect_scan_offset`, `pending_reports`), `client_feed/publisher.py`, `client_feed/value_events.py` or `stream_events.py`, `client_feed/fold.py`, the committed 1.1 feed fixture, tests | `client-feed snapshot` over a fixture root whose inbox holds a `defect_reported` line and whose `tasks.db` holds its carrier writes one `defect_filed` and folds it into `state.json`. Without a carrier, and past the grace, the heartbeat names `filer`, with reveals disabled too (B32, B33, B34, B3 extended). `validate-feed` passes on the 1.1 fixture | σ1a, ε (6972) (shares `publisher.py` and `contract.py`) |
| σ2 | The filer: `client-feed file-defects` (`--once`, `--check`), the closed MCP client, convergent gated filing with the validation fallback, the lock, the unit template and operator docs | `client_feed/filer.py` (new), `client_feed/cli.py` (`file-defects`), `shared/src/shared/mcp_session.py` (an injectable `httpx` client), `client-feed/tests/test_import_boundary.py` (`shared.mcp_session` allowed, for the filer only), `client-feed/tests/_audit_hook.py` (new; ζ imports it), `client-feed/tests/test_filer*.py`, `scripts/client-feed-filer.service.template`, `scripts/setup-host.sh` (render only), `OPERATIONS.md` ("Client feed": the filer, `--check`, σ4's preconditions) | Intermediate: unlocks σ3, σ4 and ζ's import. B25–B30 and B35 pass in the client-feed suite | σ1a, ε (6972) (shares `cli.py`), η (6974) |
| σ3 | Integration gate: the real tool layer and the portal's fixture, end to end | `client-feed/tests/fixtures/in-v2/` (vendored, with its source sha), B36's test and its home: where both the filer and `fused_memory.server` import, e.g. `fused-memory/tests/` with `client-feed` as a fused-memory dev dependency (src unchanged). σ3 decides | B23 against the portal's real fixture; B36 against the real fused-memory tool layer and its markup guard, ending with `defect_filed` in the feed | σ1b, σ2, `scoping_intake:<P1a>` |
| σ4 | Human gate: enable the filer for the first live client | `task_kind: deterministic`, `always_escalates` | Leo confirms each of the following, then enables `client-feed-filer@<name>`; the first real report shows `defect_filed` in the feed and no `filer` degradation. **Checks:** (1) the portal runs as its dedicated account; (2) P2's check passes on this host; (3) a probe from the portal account to the fused-memory URL and to every escalation port named in the registered projects' `dark-factory-orchestrator.yaml` is refused; (4) the client project's orchestrator is running and its L2 queue is watched; (5) its feed declares criteria; (6) `client-feed file-defects --config C --check` passes | σ3, `scoping_intake:<P1b>`, `scoping_intake:<P2>` |
| σ5 | Human gate: flip `filer.auto_dispatch` | `task_kind: deterministic`, `always_escalates`, `metadata.milestone {mode: delayed, after_secs: 2592000}` | `find_tasks_by_metadata(x_client_defect_review)` counts at least 10 gated live reports, with no `corrected` verdict on an unquarantined restatement. Leo then sets `filer.auto_dispatch: true` in the policy. If the count is short, the gate is re-armed with a new delayed milestone, not left in an open deferral | σ4 |

Amendments at decompose to tasks already filed:

- ε (6972) adopts `CriterionId` for `Criterion.id`, gains `client_feed/contract.py` in its files, and
  depends on σ1a, with which it shares `contract.py`. σ2 depends on ε, with which it shares `cli.py`.
  σ1b depends on ε, with which it shares `publisher.py` and `contract.py`.
- ζ (6973) imports `_audit_hook.py` and depends on σ2.
- ι (7057) builds on σ1b's ref reader and depends on σ1b.
- 7005, the 2026-10-15 milestone gate for this same bridge, is cancelled as superseded (decision 13).
  It has no dependents.

Portal tasks, filed in scoping-intake's tree, are the portal PRD's ξ1, ξ2, π and ρ:

- **P1a (ξ1).** The `feed.py` contract side:
  - the inbox union;
  - `FeedSource.filer`;
  - readers for `defect_filed`, `defects_filed` and `criteria_declared`;
  - `pending_reveals` filtering by type;
  - the `in-v2` fixture, byte-exact.

  Also vendors the factory's committed 1.1 feed fixture. Depends on `dark_factory:<σ1a>` and
  `dark_factory:<σ1b>`.
- **P1b (ξ2).** The report path:
  - the restater method on `ScopingModel`: an injection-briefed prompt on the api backend, a
    synthetic backend for tests, and `restater_declined` on claude-code;
  - typed quarantine and the verbatim-overlap check;
  - the report number, counted from history;
  - `report_defect`'s order and the reconciler;
  - only declared criteria offered;
  - the canary test (P30, which owns B31);
  - filed / not filed on the project and admin pages, from P1a's readers.

  Depends on P1a, P3 and `dark_factory:6972`.
- **P2 (π).** The deploy's default-deny for the portal uid's TCP to local and loopback addresses
  except its own listeners, with the deploy's check. Depends on the portal's deployment recipe ι (53).
- **P3 (ρ).** Concurrent reports and "Fixed" pressed by the merge (decision 13):
  - `Block`'s open reports replace the single `defect`;
  - the `defect_reported` → `defect_reported` edge for another report;
  - `fix_report`, idempotent per report, which moves the stage only when it closes the last open report;
  - the exits' actor set and their second guard;
  - the `merge_landed` × `defects_filed` join;
  - the re-demonstration window;
  - μ's deemed-acceptance rule over open reports;
  - the edge table in the portal PRD and its P9/P10 tests.

  Depends on P1a and the portal's μ (61).

Edges:

- Within this PRD: σ1a → σ1b, σ2; ε (6972) → σ1b, σ2; η (6974) → σ2; σ1b, σ2, P1a → σ3; σ3, P1b, P2
  → σ4; σ4 → σ5.
- Amendments to filed tasks: σ1a → ε (6972); σ2 → ζ (6973); σ1b → ι (7057).
- Into the portal: σ1a, σ1b → P1a; P1a, μ (61) → P3; P1a, P3, 6972 → P1b; ι (53) → P2.

σ3 is the integration gate and names B23 and B36 as its signal.

G7, design-time notes:

- `no-silent-fail-soft`:
  - A restater failure is a quarantine.
  - A validation refusal is a gated fallback.
  - An unfiled report is `degraded: filer`.
  - A rejected line is counted under `inbox`.
  - A kept report with no line is named on the admin page.
- `holds-owned-and-bounded`: the gate is owned by Leo through the L2 ladder; `deferred` appears only
  inside one convergent pass (B29); σ5 re-arms by milestone.
- `corroborate-before-acting`: `lookup_defect_ref` is called before every filing; the acknowledgement
  is read from the producer's own non-deferred rows.
- `one-fact-one-home`:
  - the raw report lives in the portal record;
  - the restatement lives on the fix task;
  - the ref is minted by one function;
  - `CriterionId` is one pattern, shared with ε;
  - the policy has one file.
- `contracts-machine-checked`:
  - the contract is pydantic, with fixtures on both sides;
  - the tool set is checked by a recording fake;
  - the substrate is checked by B36 against the real tool layer and its markup guard.
- `readers-prove-their-producer`: `defect_filed` is emitted only from real non-deferred `tasks.db`
  rows (B36), and `degraded: filer` separates a silent filer from no reports.
- `guards-exercise-behaviour`: B30 runs the filer; it does not grep its source.
- `storm-escape-required`: one unfileable report does not hold the others. A transport failure is
  retried with backoff and stays visible under `filer`; a validation refusal falls back once.

### 14.9 Out of scope

- Raw client text anywhere on the factory side.
- Disputed reports, such as a change request presented as a defect. These are Leo's call at the gate;
  defect-tests' withdrawal and exemption rules apply to the declared defect.
- Authentication on the factory's MCP servers. The residual is recorded in decision 9, and the
  portal's default-deny is the chosen control.
- Read containment for agents.
- Client channels other than the portal.
- The client-facing display of defect outcomes (defect-tests ι, and the portal).

### 14.10 Open questions (tactical)

1. **Where `filer.state_dir` defaults and what its cursor holds.** Suggested:
   `~/.local/state/client-feed/<name>/filer/`, holding the lock file, the inbox offset and per-ref
   attempt counts. Decide in σ2.
2. **The fix task's priority by stage.** Suggested: `high` in both the unpaid review window and the
   paid defect window. Decide in σ2.
3. **The restater's details.** Its retries before `restater_failed`, its exact prompt, its model and
   its verbatim-overlap threshold are the portal's, in P1b. `prompt_version` records which ones ran.
4. **`filer_grace_seconds`.** Suggested: 600 (60 filer intervals). Decide in σ1b.
5. **The soak threshold of 10 in σ5.** Provisional; σ5 records the count it actually saw.
