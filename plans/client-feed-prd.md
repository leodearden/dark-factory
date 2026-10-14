# Client feed: a per-project, client-safe evidence feed for the client portal

**Status:** authored 2026-10-14 (spawned design session; `/team` + `/prd` author mode; the design was
attacked by an opus critic seat before this PRD was written, see the design note §13). Not yet
decomposed.
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
    `events.phase`; `scheduler_paused`/`scheduler_resumed`; `escalation_created`/`escalation_resolved`
    with a level; `task_results.review_cycles`);
  - the verify archive at `<root>/data/verify-logs/<task_id>/` (`attempt-N[.module].junit-<ts>.xml.gz`
    for task legs on every verify — 2,294 reports under 1,885 task directories on 2026-10-14 — and for
    merge runs only under `merge_verify_breadth: full`; `summary-<ts>.json` with `category` and `rc`;
    pruned past 30 days and oldest-first past 500 MB by `orchestrator/src/orchestrator/verify.py::_prune_archive`);
    task legs archive at `project_root / 'data' / 'verify-logs'` (the `run_scoped_verification` call in
    `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._execute_verify_review_loop`'s verify
    path);
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
  pytest_criterion.py  pytest plugin: @pytest.mark.criterion("C-12") → the map file beside the junit report
  publisher.py       the loop: sources → events → feed; heartbeat with degraded reasons
  cli.py             publish, snapshot (one pass), criteria-results, dump-state, validate-feed
client-feed/tests/   contract fixtures, source tests over fixture stores, audit-hook isolation, canary, end-to-end on this repo
```

The publisher never imports `orchestrator`, `fused_memory` or `dashboard`; it depends on `shared` only
for `shared.project_layout` (`tasks_db`, `runs_db`) and `shared.task_metadata` (`done_provenance`),
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
   never message or output. `merge_blocked.data.reason` is never copied.
7. **One root per publisher, proven.** The publisher is constructed from one config and opens
   nothing outside the root, `out/` and `in/`; the test installs `sys.addaudithook` and asserts every
   `open`, `sqlite3.connect` and subprocess working directory lies under those three (B1).
8. **Project identity is explicit.** The config names `project_id`; every feed line and `state.json`
   carry it; the portal refuses a feed whose `project_id` is not the one it expected (B4).
9. **Event types (FEED_VERSION "1.0"):** `heartbeat`, `factory_state`, `prd_upserted`,
   `task_upserted`, `task_phase`, `verify_run`, `test_results`, `merge_landed`, `branch_snapshot`,
   `tree_snapshot`, `file_revealed`, `reveal_refused`, `criteria_results`. A new type is a minor
   version; a changed field a major one. The portal ignores unknown types and refuses an unknown major.
10. **Never hide red, from recorded sources.** A failed verify is `verify_run.passed == false`; a
    blocked task is `task_upserted` with that status and `waiting_on: leo` when an open escalation
    exists for it, else `factory`; `factory_state` carries `scheduler: running|paused`, `last_event_at`
    and `idle_seconds` from the project's own `runs.db`. Where a fact is not recorded (no junit, no
    summary) the event says `not_recorded`, never pass.
11. **The publisher reports its own health.** Every `heartbeat` carries `head`, `as_of`,
    `engagement_version`, `watermark_seq` and `degraded: [{source, reason}]`; a source failure is never
    swallowed (INV-11) and the portal shows a stale or degraded publisher as such, distinct from
    "quiet" (INV-6, INV-13).
12. **Reveal budget** (design note §5): counted set = visible production files not matching free,
    never-reveal or generated globs (`linguist-generated` honoured); lines = non-blank lines, a line
    over `max_line_bytes` counting as ⌈bytes/max⌉; budget = `reveal_proportion × counted lines added or
    changed since delivered_commit` (all counted lines when null); each reveal costs
    `max(file_lines, reveal_floor_lines)`; granted when `spent + cost ≤ budget`. Reveal identity is
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
17. **Criteria data is pytest-only and never written into the checkout** (§7.7). The plugin writes a
    `{nodeid: criterion}` map beside the junit report; the publisher joins by nodeid; a staging run's
    results are produced by the CLI into `out/`.
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

`kind` for a tree file: `test` if the path matches `test_globs` (default `**/tests/**`, `**/test_*.py`,
`**/*_test.py`, `**/conftest.py`), `docs` if `**/*.md`, `docs/**`, else `production`; files matching
`generated_globs` or marked `linguist-generated` are listed with `kind: generated` and excluded from
the budget.

### 7.3 `state.json`

`fold(feed)` up to `watermark_seq`: `{project_id, feed_version, watermark_seq, as_of, head,
engagement_version, degraded, factory, prds: {prd_path: …}, tasks: {task_id: …}, in_flight: [{task_id,
phase, since}], branches, tree, revealed: [{path, commit, lines}], criteria: {run_kind: …}, recent:
[last 200 events]}`. Rewritten atomically after every pass that appended.

### 7.4 `in/inbox.jsonl` (portal → publisher)

`{seq, at, type: "reveal_requested", path}`. Answered with `file_revealed` or `reveal_refused`.

### 7.5 `in/engagement.json` (portal → publisher)

`{engagement_version: int, project_id, stage, delivered_commit: str|null}`. Missing or unparsable: the
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
- `@pytest.mark.criterion("C-12")` via the plugin `client_feed.pytest_criterion` (loaded with
  `pytest_plugins` in the project's `conftest.py`); at collection it writes
  `<junit path>.criteria.json` = `{nodeid: criterion}` beside the `--junitxml` target when one is
  given, so the archived junit has its map beside it. No results file is written into the checkout.
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

## 9. Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Audit-hook confinement | fixture root A with tasks.db, runs.db, verify-logs, git; a sibling root B and the real dark-factory checkout both reachable on disk | `sys.addaudithook` records every open / sqlite connect / subprocess cwd during `snapshot`; all lie under A, O or I; `default_project_root` is never called |
| B2 | Two-project canary | B's task titles, descriptions, metadata, PRD text, file names and contents carry `CANARY-B-…` | no file under O contains `CANARY-B` after publishing A |
| B3 | Allowlist | a task in A with `description`, `details`, `metadata.dry_run_proposals`, `metadata.external_deps`; a title containing `/home/leo/x` | `task_upserted` and `state.json` carry none of them; the title is redacted |
| B4 | Foreign project refusal | the policy names project X; feed lines name Y | `validate-feed` rejects; the portal's consumer test (other repo) rejects the same fixture |
| B5 | Fixture round trip | the committed fixture feed `tests/fixtures/feed-v1/` | validates under `contract.py`; `fold(feed)` equals the committed `state.json`; the portal's suite validates and renders it |
| B6 | Landed merge with review | `merge_finalized` `state=done` for task 7, two `review_completed` rows, a `Merge task/7 into main` commit | one `merge_landed` with that sha, stat counts from git, `review.cycles == 2` and the last row's counts; with no rows, `review == "not_recorded"` |
| B7 | Red shown | `workflow_verify` `passed=false` and a junit with one failure | `verify_run.passed == false`; `test_results.counts.failed == 1`; `exception_type` present, no message |
| B8 | Not recorded | a `verify_run` whose task dir has no junit | `test_results` absent; `verify_run.legs == "not_recorded"` |
| B9 | Factory state | `scheduler_paused` then nothing for 3 h | `factory_state.scheduler == paused`, `idle_seconds ≥ 10800` |
| B10 | Waiting on Leo | task 9 blocked with an open `escalation_created` level 2 | `task_upserted.waiting_on == "leo"`; after `escalation_resolved`, `"factory"` |
| B11 | Visible set | tree with `dark-factory-orchestrator.yaml`, `.mcp.json`, `plans/review.md`, `src/a.py`; policy visible `src/**`, `tests/**` | `tree_snapshot.files` lists only `src/a.py`; a reveal of `.mcp.json` is `reveal_refused not_visible` |
| B12 | Reveal within budget | 1,000 counted lines, proportion 0.25, floor 50; request a 200-line production file | `file_revealed` with content; `spent_lines == 200` |
| B13 | Reveal over budget and floor | spent 200 of 250; request a 100-line file; then a 10-line file | both `reveal_refused over_budget` (the second costs 50) |
| B14 | Free set | request a test file | granted; `spent_lines` unchanged |
| B15 | Hostile paths | requests for `../x`, `/etc/passwd`, a symlink entry, an untracked path | four `reveal_refused` with reasons `not_in_tree`, `not_in_tree`, `symlink`, `not_in_tree`; nothing read outside the tree (B1's hook) |
| B16 | Follow head | a revealed file changes in a later commit; another is renamed; a third deleted then recreated | re-emitted; `renamed_from`; `deleted: true` then no content until a new request |
| B17 | Paid code free | `delivered_commit` set; a file unchanged since it | excluded from counted lines; its reveal costs 0 |
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
| ε | Acceptance criteria format, pytest plugin, map join, staging CLI | `client_feed/criteria.py`, `pytest_criterion.py`, `cli.py` (criteria-results), `publisher.py`, tests, `acceptance/criteria.yaml` exemplar | B20; `client-feed criteria-results` on a junit and map from this repo's own suite with one marked test | γ |
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
(open in the design note; exception type only here).

## 12. Open questions (tactical)

1. **Tree-snapshot cost on large repos.** Suggested: snapshot only on head change; blob line counts
   cached by sha in `cursor.json`; last-change dates from the commits walked since the previous
   snapshot. Decide in β2/γ.
2. **One unit per project or one unit with a list.** Suggested: `client-feed@<name>.service` instance
   per project, `--config ~/.config/client-feed/<name>.yaml`. Decide in η.
3. **Default globs** for visible, test, docs and generated sets. Suggested as in §7.2 and §7.6, with
   `visible_globs` having no default (a policy must name it). Decide in β2.
4. **`excerpt` rule for PRDs** whose first paragraph is a status line. Suggested: first paragraph after
   the first `## ` heading. Decide in β1.
5. **Where the plugin writes the map when no `--junitxml` is given.** Suggested: nowhere; the plugin is
   inert without a junit target. Decide in ε.
