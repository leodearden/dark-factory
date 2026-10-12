# PRD: the offline lane's red path — keep the evidence, confirm what failed, file on persistence, and remember through the task store

PRD-MARKER:offline-lane-red-path

**Project:** dark-factory (orchestrator; the lane code is shared by dark-factory, reify and solar-challenge). **Status:** active, 2026-10-12.

**Type:** contract resolving three accreted PRDs (`docs/prds/offline-deep-test-lane-worker.md` C3/C4, `docs/prds/offline-deep-test-lane-infra-extension.md` IE-D1, `plans/integration-test-lane-prd.md` §3) and eight piecemeal tasks.

**Approach:** B+H. The lane↔task-store filing seam is load-bearing and has never worked in production; two other PRDs consume what this one changes (the flake ledger and the verify-result cache); three hub files and a ratcheted file move.

**Origin:** task 6782 / esc-6782-2. **Ruling this PRD executes (Leo, 2026-10-12):** the fix proceeds as one short PRD, not as task 5757 re-scoped alone, covering in order: keep the evidence; file only what failed in both runs; hold back transients and outages; lane memory (tasks 6050 and 5747) together with a defined meaning for "cancelled". The bake-off live test is left unchanged.

**Code anchors** verified against main `1c0690552b` (2026-10-12); the cited files are unchanged through `3ed5cf4174`. Main moves fast — cite-by-symbol; re-locate at implementation time.

**Review:** authored under `/team` + `/prd` from four substrate reports. An adversarial critic pass on the first draft returned 25 findings and a re-check of the revision returned 16 more; all are folded in. The three that changed the design: held-back reds are recorded in the flake ledger but do not open de-flake debt here (D6); a pass counts toward persistence only if it started 30 minutes after the last counted one (D5); and what one pass means for one test is a closed table (D5).

## 1. Goal

When a lane command goes red, the lane keeps what it saw, files a fix task only for tests that are really and persistently red, counts every held-back red where it can be seen, and knows which task covers which test because it was told the task id when it filed.

What an operator observes once this lands:

1. **Every lane pass leaves a record.** `data/orchestrator/offline-lane/<command>/<pass-id>/record.json` names the head, both runs' exit codes and durations, and for a red pass each failing test with its kind and first failure line. Full output sits beside it.
2. **A lane-filed task explains itself.** `get_task` on it shows, per test, the failure text from each run of each counted pass, the passes it was red on, the command, and a reproduce line. A reader can classify it without re-running anything.
3. **One red pass files no task.** The episode record shows the test held at 1 of 2. A test reproduced on two spaced passes files exactly one task.
4. **A test that passed the detection run is never filed**, whatever the confirm does.
5. **The lane knows what it filed.** Every lane filing has a task id that resolves through `get_task`, recorded on the episode.
6. **A held-back red is counted.** `flake_occurrence` has rows with `call_site = 'offline_lane'`, and the ledger report shows them under their own call site.
7. **A red that outlives its task is escalated once per task**, 24 hours after filing, naming the task and its status.

Consumers (G1): the fix-task reader (architect and steward agents, and the L2 watcher); the flake-ledger report; the verify-result cache audit (task 5655) through the hook in §5.6; auto-eval (task 5423) and the flake ledger's own filing through the typed submit seam; the soak gate ζ, whose report is the input to the two follow-ups in §9.

## 2. Premise (G6) — measured

Period 2026-09-19 → 2026-10-11, dark-factory, from the orchestrator journal unless stated.

| Fact | Figure | Rule |
|---|---|---|
| Lane passes, `qdrant-integration` | 940, of which 92 red | lines `offline-lane: qdrant-integration sub-run … status=` |
| Red passes cleared by the confirm | 33 (`qdrant-integration`), 16 (`warm-lane-bash`) | `intermittent nondeterminism` lines |
| Confirmed reds | 59 | `submit_fix_task returned no task id` lines |
| Confirmed reds whose filing recorded a task id | 0 of 59 | `filed INFO escalation` lines: 0 |
| Tasks those 59 produced | 8 | `tickets.db`, status `created` |
| Incidents of a test confirmed red | 10 (18 if counted per test) | maximal runs of consecutive passes of one command with the test in the confirm's failing set, grouped by start and length |
| …that were a production code break | 1 (task 6146; red 25 passes in a row, 13.1 h) | commit `86441bf154` |
| …that were provider outages | 2 (OpenAI credits: 9 tests, 26 passes, 8.5 h; Claude weekly cap on the operator's login: 2 passes at one head, 21 min apart) | esc-6280-3; task 6813 |
| …that lasted exactly one pass | 7 | next pass of the command green |
| Consecutive passes that share a head | 309 of 939 | same head as the previous pass of that command |
| Next pass starting under 8 min after a confirm | 14 of 90 | start minus the previous confirm's end |
| Spacing of consecutive passes | median 24.9 min, p90 64.7, max 386.9; 30 of 939 over two hours | start to start |
| After a stall of the bake-off test, re-stalls 8–45 min later | 0 of 12 runs | Qdrant request log, 10 incidents |
| After a stall in a detection run, the confirm minutes later re-stalls | 2 of 9 | same log, lane passes only |
| Longest a test stayed confirmed red after its task was filed | 13.0 h (n = 1, task 6146) | filing → last confirmed-red pass |
| Lane filing → last write, all eight tasks | 4.2, 4.4, 13.1, 14.2, 15.3, 16.4, 27.6, 89.4 h | ticket `created_at` → task `updated_at` |

What the code does today (read at the anchor commit):

- `orchestrator/src/orchestrator/offline_lane.py::OfflineLaneWorker._run_once` keeps a sub-run's exit code and drops its output, on all three legs.
- `::_default_confirm_command` re-runs the whole command serially and returns whatever fails in that run. Nothing compares it with the detection run. `plans/integration-test-lane-prd.md` §3 mandated this.
- `orchestrator/src/orchestrator/harness.py::_OfflineLaneTaskClient.submit_fix_task` returns `''` for every production response shape, because `Harness._extract_task_id` reads none of them. `open_fix_tasks`, `_red_advance_counts`, `_promoted_blockers`, the L0 info and the staged L2 have therefore never run.
- The dedup that did work is fused-memory's curator, with the unique index on `candidate_key` as a backstop only until the files tagger changes the key (16 of 24 lane tasks across three projects; rule: stored key ≠ the title-only key). Both treat a cancelled task as covering nothing.
- A fleet restart that kills a detection run is logged `status=FAIL`.

Basis of each number this PRD sets:

- **`persist_passes = 2`, each counted pass starting at least `recheck_secs` after the previous counted pass started.** Replayed over the ten incidents: it files the code break one pass later (+48 min of a 790 min incident), files the long outage 26 min later than an unspaced count would, and files neither the seven one-pass incidents nor the 21-minute cap incident. `3` would file the same incidents, the code break a further 31 min later and the outage 29 min later. One real break is thin evidence: ζ re-measures, and the knob is hot-reloadable.
- **`recheck_secs = 1800`.** Stalls recur within minutes (2 of 9) and not at 8–45 min (0 of 12); 30 min is past the window where they cluster. Without a recheck a held red waits for the next merge.
- **`red_unlanded_hours = 24`.** About 1.8 × the one measured red-after-filing. Provisional; ζ re-measures.
- **Record retention, 14 days.** Twice ζ's window, and the existing lifetime of the lane's durable state.

The long outage on record still files one task under these settings. §4 D7 says why that is accepted here.

## 3. Sketch of approach

A **pass** is one execution of one lane command (or legacy leg) at one head: a detection run and, if that is red, a confirm.

1. **Record.** Every sub-run streams its output to disk and writes a structured record. A structured pytest command also produces junit, used as per-test evidence.
2. **Confirm.** Where the command can be narrowed, the confirm re-runs only the tests the detection run named, serially. Where it cannot, it re-runs the whole command and keeps only tests that failed in both runs with a compatible kind. Each test is reproduced, cleared or inconclusive; a confirm that cannot answer makes the whole pass unconfirmable.
3. **Hold or file.** A reproduced test starts or extends an **episode**. At `persist_passes` counted passes the lane files one task for the tests that crossed the line and have no live covering task. It files synchronously and is told the task id.
4. **Remember.** One durable record per (command, test) holds the episode, the covering task id and any escalation. Task status is read live at every decision.
5. **Count the held-back.** Every verdict is written to the flake ledger under a new call site. This PRD does not open de-flake debt from it.
6. **Bound.** A test still reproduced 24 hours after its task was filed raises one escalation. A held episode is re-observed after 30 minutes if main has not advanced, and expires if nothing extends it.

## 4. Resolved design decisions

**D1 — Identity is (command, test), not the failing set.** The set fingerprint drifts while one test stays red (on 2026-10-11 three fingerprints overlapped on two tests), and nothing reads `metadata.offline_lane_fingerprint`. Episodes, covering and filing are keyed per test. `workflow.py::compute_failing_test_set_fingerprint` and the three unblessed metadata keys (`failing_tests`, `suspect_ranges`, `offline_lane_fingerprint`) retire; a lane task carries `spawn_context`, `merge_lane` and one `x_offline_lane` object (§5.5).

**D2 — Evidence lives on disk; a digest lives in the task.** Records and full output go under `data/orchestrator/offline-lane/`: not inside the lane worktree, which `git_ops.py::reset_persistent_offline_deep_worktree` cleans every pass, and not under `data/verify-logs/`, whose 500 MiB prune budget is already full. The task's `details` carries a bounded digest that must suffice on its own, because a task agent's sandbox may not reach `data/`. Nothing is lifted from branch `task/4292`: its 2,000-character tail cuts the first failures of a long red, and its scope hint points the wrong way in both decisive cases on record (the one real break touched no file under the command's directory; the outage's range touched six).

**D3 — The confirm narrows when it can and intersects when it cannot.** Narrowing uses `orchestrator/src/orchestrator/verify_cmd.py` (`parse_config_command`, `scope_to`, `serial_pytest`, `with_pytest_timeout`, `render`), which renders `pytest -m integration <ids>` correctly. It does not call `verify.py::confirm_isolated_rerun_verdict`: that function needs a `VerifyResult` and `ModuleConfig`s the lane does not have, runs under verify admission, and scopes with `_scope_to_keyword`, which drops `-m integration` so the re-run selects nothing. The command builder, the verdict vocabulary (`flake_ledger.py::FlakeVerdict`) and the ledger are shared; the runner differs because a lane pass is not a verify. An id that does not survive a round trip through the parser (the stdout parser truncates ids at whitespace) makes the pass unconfirmable with that reason.

**D4 — A confirm that cannot answer says so.** The verdict is a tagged union: `Confirmed(reproduced, cleared, inconclusive)` or `Unconfirmable(reason)`. Unconfirmable: a confirm that exits non-zero naming no test (a thread-mode timeout or a crash); a detection run that is red and names no test, on any leg except the numeric one (D14); a sub-run ended by a signal; an id that fails the round trip. Today the first becomes the `::nonzero-exit` sentinel on the generic leg and a silent flake on the infra leg, and a signal-ended run is logged as a failure. Inconclusive, for one test: it FAILED in detection and ERRORed at setup or collection in the confirm (a confirm ERROR reproduces only a detection ERROR, the rule from task 3880's plan); or the whole-command confirm exited non-zero without completing its session and did not name it. So the confirm's own harness can neither fabricate a reproduction nor clear a real red. An unconfirmable command is a pseudo-test of its own (`<command>::unconfirmable`) with its own episode and a command-level task on persistence. A worker stopped before a pass's verdict is recorded leaves that pass with no effect.

**D5 — File on persistence, over a closed set of observations.** `git.offline_lane_persist_passes`, default 2, hot-reloadable and read at each pass; 1 restores filing on the first reproduced pass. Spacing is measured start to start. What one pass does to a test's episode:

| The pass shows, for test T | Effect on T's episode |
|---|---|
| reproduced, and the pass started at least `recheck_secs` after the last counted pass started | extends (a counted pass) |
| reproduced, sooner than that | none |
| cleared | ends |
| passed: per-test evidence shows T passed; or there is no per-test evidence for T and the detection run exited 0 | ends |
| inconclusive, skipped, deselected or not run; or T is absent from the failing set with no per-test evidence while the detection run exited non-zero | none |
| the pass is unconfirmable | none for T (the command's pseudo-test is reproduced) |
| the covering task is done and the head does not contain its commit | none (a stale head) |

A held episode that no pass extends within `red_unlanded_hours` expires; the one knob serves as both that expiry and D11's deadline. Per-test evidence is junit for a structured pytest command, mapped to node ids by a normaliser β owns and tests, and the per-test result lines for the infra leg; where the mapping fails for T there is no per-test evidence for T.

**D6 — Held-back reds are counted in the flake ledger; they do not open debt here.** Every verdict is recorded with `record_flake_occurrence` under a new call site, and the ledger report's gate-blind and systemic counters are partitioned by call site so lane rows cannot dilute the merge gate's. `open_debt` is not called. Its owner rules were built for a suppression that lands a merge: a cancelled owner is replaced at the next occurrence, a deferred one is re-released, a done one re-enters as a regression. Applied to a lane test that stalls every other day, that restores the loop this PRD ends, at a higher rate; its de-flake task text is hard-coded to the merge gate; and the number of tests it would file for is unmeasured, because detection failing sets were never recorded. The debt rule for lane-held tests, including what a cancelled owner means, is a named follow-up decided on ζ's measured counts (§9). G7 waiver: `storm-escape-required` — held-back reds are counted per test and reported, but nothing files or escalates on their rate until that follow-up; today they are not counted at all.

**D7 — Outages are bounded here, not classified.** An outage that persists across two spaced passes files one task whose evidence shows the provider's error; nothing refiles while that task is open; one escalation follows at 24 hours. Typed classification is deferred to a named follow-up (§9) because its substrate does not exist (no per-failure exception facts in pytest output, no OpenAI health gate, and the one quota predicate is private to `fused_memory`), a per-test classifier would have missed the downstream-symptom failures in the one long outage on record, and two incidents cannot calibrate a hold.

**D8 — The lane files synchronously and deduplicates for itself.** It files the way `flake_ledger.py::_ensure_owner_task` does: `submit_task(planning_mode=True)` then `commit_planning`, through the same task-client protocol. The curator leaves the lane's path: it adds a median 70 s, and over 120 lane tickets in three projects its decisions beyond matching the identical set were 13 (11 overlaps between lane tasks, 2 folds into an unrelated task) plus one `create` for a stale head. Tests that cross the threshold on the same pass share one task; tests that cross on a later pass get their own. The title carries the command and the first red head, so it cannot collide with an earlier filing.

**D9 — One typed submit seam for machine filers.** `Scheduler` gains a submit method returning a tagged outcome — a ticket; a task with its id, status and `combined` flag; or an error — parsed with `shared/src/shared/mcp_envelope.py::parse_tool_result`. `flake_ledger.py::FlakeLedgerTaskClient.submit_task` returns that outcome instead of a bare string, and `_ensure_owner_task` treats `combined` into a terminal task as a failed filing. `Harness._extract_task_id`, `chronic_flake.py::extract_task_id` and the `tkt_` prefix guard retire. Accepting `ticket` in the old extractor (task 6050's framing) would have stored a ticket id as a task id.

**D10 — What the covering task's status means when the test is red again.** The binding from a (command, test) to its covering task outlives the episode: it lasts until that task is terminal, so a test that flaps while its task is open is still covered. Status is classified with the ledger's existing four-way owner verdict (`flake_ledger.py::_owner_liveness`), moved to a shared home, not re-derived.

| Covering task reads | Lane does |
|---|---|
| open (any non-terminal status except deferred, including blocked and infra-hold) | nothing; the record accumulates passes |
| deferred, and the lane's record says `commit_planning` was never confirmed | finishes the filing |
| deferred otherwise | nothing; whoever deferred it holds it |
| done, with no fix commit recorded, or with one the pass's head contains | the fix did not hold: the binding is dropped, a new episode starts at 1 and its task will name the prior one |
| done, and the head does not contain the fix commit | the pass is a stale head for that test |
| cancelled or absent | no refile while the episode lasts. When the episode ends, or the D11 escalation raised for it is resolved, whichever comes first, the binding is dropped and any new episode starts at 1 |
| unreadable, or the ancestry check itself fails | no decision this pass, logged loudly |

"Cancelled" therefore means: that filing is closed. It neither blocks a later, separately evidenced filing nor triggers anything by itself. The repeated false-premise loop ends because a one-pass red no longer files, not because the lane remembers cancellations. A test that stays red after its task was cancelled is escalated once and, after that is resolved by whoever resolves it, may be filed again on two fresh spaced passes: at most one such cycle per 24 hours per test, each needing someone to cancel a task.

**D11 — One bounded escalation replaces three.** The advance-count trigger, the cancelled/deferred arm (`_TERMINAL_NON_DONE`) and the born-at-L2 critical go; none ever fired. In their place: when a test is reproduced on a counted pass at least `git.offline_lane_red_unlanded_hours` (default 24) after its covering task was filed, and that task is not done with its fix contained, the lane raises one escalation per covering task — role `orchestrator-offline-lane`, severity `blocking`, level 1, category `dependency_discovered`, filed against a per-command sentinel id so it does not alter the fix task's own recovery handling, and carrying the task id, its live status and the record paths. The role joins `escalation/src/escalation/authority.py::PROMOTE_SENTINEL_BOUND_ROLES` so a promotion to level 2 stays on the sentinel. Its state is read from the escalation queue. The L0 info at filing goes, and with it the role's entry in `escalation/src/escalation/classify.py::INFO_L0_MECHANICAL_ROLES`.

**D12 — A held red is re-observed.** For a generic command, if an episode is held below the threshold and no pass of that command has started for `git.offline_lane_recheck_secs` (default 1800) since the last counted pass started, the lane runs that command again at the current head — at most `persist_passes − 1` rechecks per command per head, whatever the number of held tests. The legacy legs are re-observed at the next advance. A filing that could not be made is kept as filing-due on the record and retried on the poll tick without re-running the command.

**D13 — All three legs use one engine; no reify or solar-challenge file changes.** The legacy legs keep their own commands and gain records, verdicts, episodes and persistence. The numeric leg's detection names no tests (its failing set lives in a manifest inside the build tree), so its verdict comes from the confirm alone: exit 100 with ids → those are reproduced; exit 0 → nothing reproduced, recorded as one cleared row against the leg's pseudo-test so the rate is visible; any other exit → unconfirmable. The infra leg prints a result line per test, so it intersects like an opaque command and has per-test evidence; a non-zero exit with no failing line is unconfirmable, where today it is silently dropped.

## 5. Contract

**5.1 Run record** (`offline_lane_records.py`, new). One per sub-run: `command`, `leg`, `head`, `phase` (`detect` | `confirm`), `started_at`, `ended_at`, `returncode`, `ended_by` (`exit` | `signal`), `output_path`, `junit_path | None`, `failures: [{node, kind: failed | error, first_line}]`, `ids_available: bool`, `outcomes: {node: passed | skipped} | None`, `session_complete: bool`. Written before the pass's verdict is acted on. Retention: 14 days, plus any record a live episode or a non-terminal covering task refers to.

**5.2 Pass verdict** (`offline_lane_confirm.py`, new). `confirm(command, detect_record) -> Confirmed | Unconfirmable`, with `mode: narrowed | intersected | leg_native`. For `narrowed` and `intersected`, `reproduced`, `cleared` and `inconclusive` partition the detection run's failures. For `leg_native` on the numeric leg, `reproduced` is what the confirm printed and the other two are empty.

**5.3 Episode record** (`offline_lane_episodes.py`, new; stored with `guard_state.py`). Per (command, node): `counted_passes: [pass-id]`, `last_counted_started_at`, `first_red_head`, `last_green_head`, `covering_task_id | None`, `planning_committed: bool`, `filed_at | None`, `filing_due: bool`, `escalation_id | None`, `notes: [EvidenceNote]`, `prior_task_ids: [id]`. Per command: `rechecks: {head: count}`. It is the lane's only red-path state; the three guard sidecars are deleted. Entries are refreshed on every pass while the episode is live or the covering task is non-terminal, and otherwise expire after the lane's existing 14 days. A store that cannot be read degrades to an empty one with a loud log; the cost is a filing one pass late and, if a covering task was open, possibly one duplicate task. History of ended episodes is not kept here: it is the ledger rows of §5.7.

**5.4 Filing.** Ordering: write `filing_due` → submit (planning mode) → record `covering_task_id` → `commit_planning` → record `planning_committed` → clear `filing_due`. A `combined` outcome into an open task makes that task the covering task; into a terminal task it is logged as a swallowed filing and treated as cancelled.

**5.5 The filed task.**

- Title: `offline-lane[<command>]: <n> test(s) red on <k> passes since <first_red_head, 12 chars> — <first test>`. Command-level: `offline-lane[<command>]: exits non-zero with no reproducible failing test since <head>`.
- Description, short: the command and leg, up to ten tests then a count, the suspect range (last all-green head to first red head), and one reproduce line (the working directory and the narrowed command, or the whole command where it cannot be narrowed).
- Details, the digest: for up to ten tests, the kind and the first failure line (300 characters) from the detection run and the confirm of each counted pass; a table of counted passes (pass id, head, start time, verdict mode); any hook notes; prior lane task ids for the test; the record directories.
- Metadata: `spawn_context = 'offline_lane_red'`, `merge_lane = 'normal'`, and `x_offline_lane = {command, leg, nodes, first_red_head, last_green_head, pass_ids, prior_task_ids}`.
- Priority: `LaneCommand.fix_task_priority` for a generic command; `high` for the legacy legs, as today.

A worked digest entry:

```
tests/test_corpus_smoke.py::test_live_population_smoke   kind: failed
  pass 20261001T132104-e935a2 detect : AssertionError: 3 rows carried '[unverified_claim]' ...
  pass 20261001T132104-e935a2 confirm: AssertionError: 3 rows carried '[unverified_claim]' ...
  pass 20261001T140922-7c41d0 detect : AssertionError: 3 rows carried '[unverified_claim]' ...
  pass 20261001T140922-7c41d0 confirm: AssertionError: 3 rows carried '[unverified_claim]' ...
```

**5.6 Red-handler hook** for task 5655: `on_pass_verdict(command, head, verdict, run_records) -> [EvidenceNote]`, called once per red pass after the verdict and before any filing decision. Returned notes (`{source, node | None, text}`) are stored on the named test's episode, or on every reproduced test's when no node is named, and rendered into the digest at filing. A hook that raises costs only its notes.

**5.7 Ledger recording.** Per red pass, one `record_flake_occurrence` per test with `call_site = offline_lane`: reproduced → `fails_in_isolation`; cleared → `passes_in_isolation`; inconclusive → `unconfirmable` with that reason. An unconfirmable pass writes one `unconfirmable` row against the command's pseudo-test; the numeric leg's unnamed clear writes one `passes_in_isolation` row against the leg's pseudo-test. `open_debt` is never called from the lane.

**5.8 Error semantics.** No path treats an unreadable status, a failed submit, a failed ancestry check or an unparsed output as a negative answer. Each is a named state (`unreadable`, `filing_due`, `Unconfirmable`, `inconclusive`) that defers the decision and is visible in the record.

## 6. Boundary-test sketch

Facing the lane (L), the task store (T) and the flake ledger (F). Every L scenario drives the real `OfflineLaneWorker` with a fixture command. Where a scenario has a verdict half and an episode-or-filing half, §9 assigns them to different leaves.

| # | Side | Scenario | Asserts |
|---|---|---|---|
| 1 | L | A red pass and a green pass of a fixture command | each leaves a record; the red one names each failing test with kind and first line; the output file exists |
| 2 | L | A sub-run killed by a signal | record `ended_by: signal`; verdict unconfirmable with the signal as reason |
| 3 | L | A test fails detection and the narrowed confirm, pass 1 | no task; the episode holds one counted pass |
| 4 | L,T | The same test reproduced on a pass that starts 30 min after pass 1 started | exactly one task; the outcome is a task id, not a ticket; `details` shows both passes; the episode has the id and `planning_committed` |
| 5 | L | The same test reproduced on a pass that starts 3 min after pass 1 started | not counted; no task |
| 6 | L | A test fails detection, passes the narrowed confirm | verdict cleared; its episode, if any, ends; no task |
| 7 | L | Opaque command: detection red on A, whole-command confirm red on B only | nothing reproduced; B has no episode |
| 8 | L | Confirm exits non-zero with no ids, on two spaced passes | unconfirmable both times; real tests' episodes untouched; one command-level task on the second |
| 9 | L | Detection FAILED on A; confirm ERROR on A at setup | A is inconclusive: not reproduced, not cleared |
| 10 | L | A held test is skipped on the next pass; then the session is truncated; then it passes | the first two leave the episode unchanged; the third ends it |
| 11 | L | A failing id containing whitespace | the pass is unconfirmable with the round-trip reason |
| 12 | L,T | Tests A and B cross on pass 2; test C crosses on pass 4 while that task is open | one task for A and B; a second for C; no third |
| 13 | L,T | Persistent red with the covering task open, across ten more passes and past 24 h | no second task; no escalation before 24 h; exactly one after; none later |
| 14 | L,T | Covered test goes green, then red twice, while its task is still open | no new task |
| 15 | L,T | Covering task cancelled mid-episode; test stays red | no refile; one escalation at 24 h from filing; once that is resolved, the next red is held at 1 and a second spaced red files a new task naming the prior |
| 16 | L,T | Covering task done with fix commit C; red at a head without C, then on two spaced passes at heads with C | first pass has no effect; a new task after the second, naming the prior |
| 17 | L,T | Covering task done with no fix commit; then the ancestry check fails | the first is treated as contained; the second defers with a loud log |
| 18 | L,T | Task store unreachable at filing | `filing_due` persists; the command is not re-run to retry; one task once reachable |
| 19 | L,T | Restart between submit and `commit_planning` | the deferred task is committed on the next decision; no second task |
| 20 | L | Held episode, main idle, two tests held on one command | one recheck pass starts 30 min after the last counted pass started; it is counted and files if red; no second recheck at that head; a never-extended episode expires after `red_unlanded_hours` |
| 21 | L | Episode store unreadable while a covering task is open | loud log; at most one duplicate task |
| 22 | L | Numeric leg: detection red then confirm exits 100 with ids; exits 0; exits 64. Infra leg crashes printing no failing line | reproduced; nothing reproduced; unconfirmable; unconfirmable |
| 23 | L,T | A registered hook returns a note for test A | the note appears in A's task digest; a raising hook loses only its note |
| 24 | T | The four live `submit_task` responses (ticket; planning-mode create; planning-mode collision; tool error), asserted in fused-memory's tests against the same fixture file the orchestrator's typed parser is tested with | both sides read one fixture |
| 25 | F | The ledger and its report after scenarios 4, 6, 8, 9 and 22 | one row per test per red pass at `offline_lane` with the verdict of §5.7, including the two pseudo-test rows; no debt row; the report lists `offline_lane` separately and the merge-gate counters are unchanged |

## 7. Pre-conditions (G3) — verified against main `1c0690552b`

| Capability | Evidence |
|---|---|
| Narrowed pytest commands | `verify_cmd.py::parse_config_command`, `::scope_to`, `::serial_pytest`, `::with_pytest_timeout`, `::with_junitxml`, `::render` exist; both dark-factory lane commands parse structured and render a runnable `pytest -m integration <ids>` (run 2026-10-12). `worker_id` survives `-p no:xdist` in fused-memory through its own conftest fixture |
| Failure-line parse | `verify.py::_extract_failing_test_ids` exists; it returns ids without kinds and truncates at whitespace. **The typed parse and the round-trip check do not exist** — β builds them by extending that parser |
| Per-test outcomes | `verify.py::_extract_failing_test_ids_from_junit` reads junit failures; reify's `tests/infra/run_all.sh` prints `RESULT: PASS | FAIL | SKIP (name)`. **A junit passed/skipped reader and a junit↔node-id normaliser do not exist** — β builds them |
| Synchronous create | `submit_task(planning_mode=True)` returns `{'task_id', 'status': 'deferred', 'planning_mode': True}`, and on a title collision `{'task_id', 'status', 'combined': True, 'planning_mode': True}`; `commit_planning` exists; tasks filed this way dispatch normally (29 of 30 ledger-filed tasks are done) |
| Task-client protocol | `flake_ledger.py::FlakeLedgerTaskClient` and `chronic_flake.py::SchedulerChronicFlakeTaskClient` exist; the client is constructible from what `Harness._start_offline_lane` holds. **Its `submit_task` returns a bare string** — γ changes it |
| Envelope parser | `shared/src/shared/mcp_envelope.py::parse_tool_result` exists |
| Ledger API | `flake_ledger.py::record_flake_occurrence`, `::_owner_liveness`, `::FlakeCallSite`; `flake_report.py::compute_gate_blind`, `::compute_systemic` exist. **`FlakeCallSite.offline_lane` and the per-call-site partition do not** — ε adds them |
| Fix-contains-head check | `git_ops.py::GitOps.is_ancestor` exists; it returns false for a git error as well as for "not an ancestor". **A read that tells the two apart does not exist** — δ2 adds it. `done_provenance.commit` is optional: 231 of 3,818 done tasks with provenance carry none |
| Status partition | `shared/src/shared/task_statuses.py::TERMINAL`, `::ACTIVE` exist |
| Escalation vocabulary | severities `info`, `blocking`, `critical`, `urgent` (`escalation/src/escalation/models.py::KNOWN_SEVERITIES`); level 1 is the auto-watcher's; `blocking` level-1 escalations against a sentinel id by an `orchestrator-` role already exist in `harness.py`; the lane already holds an `EscalationQueue` |
| Signal-ended runs | a child ended by a signal reports a negative return code; the lane launches through `sh -c`, where a killed grandchild surfaces as 128 + n. α pins both forms in scenario 2 |
| Durable small state | `guard_state.py::PersistentMap` holds JSON values; an entry expires a caller-supplied time after its last write (the lane passes 14 days) |
| Lane knobs hot-reload | `git.offline_lane_red_advances_before_blocker` is in `config.py::RELOADABLE_FIELDS`; the three new knobs join it and that one leaves. No project config sets it |

Not assumed: a task read by metadata (none exists); a structured cancel reason (none exists); narrowing for solar-challenge's commands (they parse opaque; task 6068 would not change that for the three that start with `timeout`); a lane run timeout (`LaneCommand` has none); detection ids on the numeric leg; a harness flag saying a stop is in progress; an orchestrator-side fixture for a real fused-memory task backend (ι uses the typed client over the scenario-24 fixture).

## 8. Cross-PRD relationship (G4)

Every dated pointer below is written by δ3.

| Other PRD or task | Direction | Seam | Owner | Changed by |
|---|---|---|---|---|
| `docs/prds/offline-deep-test-lane-worker.md` | amends | C3 fingerprint dedup, C4 escalation staging, boundary rows B4–B7 | this PRD | δ2; pointer |
| `plans/integration-test-lane-prd.md` | amends | §3 "Confirm (flake filter)": whole-command serial re-run | this PRD | β; pointer |
| `docs/prds/offline-deep-test-lane-infra-extension.md` | amends | IE-D1: infra failures ride the fingerprint dedup unchanged | this PRD | δ2; pointer |
| `plans/flake-ledger-prd.md` | extends | a fourth call site that records only; the gate-blind and systemic counters partitioned by call site; §8.1 "one implementation, all call sites" gains the lane's own runner (D3); `FlakeLedgerTaskClient.submit_task`'s return type (D9); `_owner_liveness` moves to a shared home | flake ledger owns the vocabulary, the debt invariant and the report; this PRD owns the lane call site and the typed outcome | γ, δ1, ε; pointer |
| `plans/verify-result-cache-prd.md` ε (task 5655) | produces for | §5.6 hook; that PRD's "blocker after three confirmed-red advances", "lane engine untouched" and "a fix task carrying the leak record" | this PRD owns the hook; 5655 consumes it | 5655 depends on δ2 in place of 4226, and its work items 2–3 and signal are rewritten at decompose; pointer |
| `plans/fm-task-dedup-prd.md` | relies on, unchanged | cancelled rows never collide; the planning-mode collision response | fm-task-dedup | — |
| `plans/review-all-s14-shared-leaves-prd.md` | consumes | `shared.mcp_envelope` | S14 owns the envelope; this PRD owns the typed submit on `Scheduler` | — |
| `plans/review-all-s11-curator-pipeline-prd.md` | none after this PRD | the lane leaves the ticket path | — | — |
| `plans/info-l0-disposition-router-prd.md` D8 | amends | the lane no longer files L0 info | this PRD | δ2; pointer |
| `plans/config-hot-reload-prd.md` | amends | green-tier list | this PRD | δ3; pointer |
| `plans/warm-lane-infra-repatriation-prd.md` | amends a promise | "a red routes within one advance" becomes "within `persist_passes` spaced passes" | this PRD | δ3; pointer |
| Task 5423 (auto-eval) | consumes | the typed submit seam | γ | 5423 depends on γ and drops its own parse work |

Disposition of existing tasks, applied at decompose:

| Task | Disposition |
|---|---|
| 4292 | superseded by α and δ1; nothing lifted (D2); branch kept as reference |
| 3880 | lane half folded into β; re-scoped to its discriminator half |
| 4226 | folded into β (D4); tasks 5655 and 5412 re-point their dependency to δ2 |
| 6028, 6639 | folded into δ1 (command in the title, digest and metadata; the builder leaves `workflow.py`) |
| 6050, 5747 (and branch), 6654 | superseded by γ, δ1 and δ2 |
| 5757 | superseded: remedy (a) is replaced by β and D5; remedy (b) is not adopted, and D10 says why its concern no longer arises |
| 6650 | kept; depends on β, because its lane call site moves into the new confirm module |
| 6068, 6813, 4078 | kept, independent; 4078's sentinel name becomes the command pseudo-test |

## 9. Decomposition plan

Sizes follow the overlay bands. Labels are Greek; ids are assigned at decompose. "HUB" marks a hand-carry. `offline_lane.py` and the four lane test files are ratcheted; every leaf that touches them moves the run, confirm and red-path code out into the new modules so the file shrinks, and carries `orchestrator/tests/merge_lane_ratchet_baseline.json` and `orchestrator/tests/merge_lane_ratchet_authorized_raises.json`. The chain is a line, so no two leaves edit `offline_lane.py` or an engine module at once. α through ε are intermediates roped into the integration gate ι; ζ is the leaf an operator observes.

- **α — Run records: every lane sub-run leaves evidence (no hub; dispatchable).** [~600 LOC; `offline_lane_records.py`, `offline_lane_runs.py` (new; the four run seams move here and stream to file), `offline_lane.py`, the two ratchet files, tests]. Unlocks β. Signal: scenario 1 and the record half of scenario 2.
- **β — The confirm confirms (no hub; dispatchable).** [~800 LOC; `offline_lane_confirm.py` (new), the typed extension of the failure-line parser with its round-trip check, the junit outcome reader and id normaliser, `offline_lane.py` (confirm seams out), the two ratchet files, tests]. Depends α. Unlocks δ1, δ2. Signal: the verdicts of scenarios 2, 6–11 and 22.
- **γ — One typed submit seam for machine filers (HUB: `harness.py`).** [~600 LOC; `scheduler.py`, `flake_ledger.py` (protocol return type, `_ensure_owner_task`, prefix guard removed), `chronic_flake.py`, `harness.py` (`_extract_task_id` retired, auto-eval call site), twelve test fakes, one fixture file shared with a fused-memory test]. No dependency. Unlocks δ1, task 5423. Signal: scenario 24.
- **δ1 — Episode engine and the filed task (no hub; dispatchable).** [~800 LOC; `offline_lane_episodes.py`, `offline_lane_filing.py` (new), `flake_ledger.py` and `shared/src/shared/task_statuses.py` (`_owner_liveness` moves), unit tests against fake records and a fake typed client]. Depends β, γ. Unlocks δ2. Signal: the D5 and D10 tables as table-driven tests over the engine, and the §5.5 payload against the worked example.
- **δ2 — Cutover (HUB: `harness.py`; hand-carry).** [~1,300 LOC, most of it test migration; `offline_lane.py` (red path replaced by the engine; sidecars, `_maybe_promote_blocker`, `_file_info_escalation`, `_file_blocker_escalation`, `_TERMINAL_NON_DONE` deleted; recheck in `run`; the §5.6 hook), `harness.py` (`_OfflineLaneTaskClient` deleted; the lane wired to the machine-filer client), `git_ops.py` (an ancestry read that reports a git error apart from "no"), `escalation/src/escalation/classify.py`, `escalation/src/escalation/authority.py` and their tests, the four lane test files, `test_restart_durable_guards.py`, the two ratchet files]. Depends α, β, γ, δ1. Policy values are the engine's defaults until δ3. Signal: the episode and filing halves of scenarios 3–6, 8, 10 and 12–21, and scenario 23.
- **δ3 — Knobs, removals and documents (HUB: `config.py`, `workflow.py`; hand-carry).** [~400 LOC; `config.py` (three knobs in, one out, reload list), `offline_lane.py` (policy read from config), `workflow.py` (two helpers removed) and the stale references to them (`chronic_flake.py` docstrings, `shared/src/shared/task_metadata.py`, `test_workflow_offline_lane_fix.py`), `OPERATIONS.md` (a lane red-path runbook section), `ARCHITECTURE.md` §5.4, dated pointers in the eight PRDs marked in §8]. Depends δ2. Signal: a reload changes `persist_passes` on a running worker and the next pass uses it (the existing config-reload test pattern).
- **ε — Lane verdicts in the flake ledger, recording only (no hub; dispatchable).** [~400 LOC; `flake_ledger.py` (`FlakeCallSite.offline_lane`), `flake_report.py` (both counters by call site), the recording call in the engine, tests]. Depends δ3. G7 waiver: `storm-escape-required` — see D6. Signal: scenario 25, on rows the worker wrote.
- **ι — Integration gate (no hub; dispatchable).** [~500 LOC; `orchestrator/tests/test_offline_lane_red_path_integration.py` (new)]. Depends ε. Signal: scenarios 1–23 and 25 execute in one module against the real `OfflineLaneWorker`, the typed client over the scenario-24 fixture and the real ledger; scenario 24's other half runs in fused-memory.
- **ζ — Soak gate (operational; `task_kind: deterministic` where its checks are mechanical).** Depends ι and seven days of the deployed lane. Checks, read from run records, the ledger and the task store, not the journal: every pass since deploy has a record; every lane filing id resolves through `get_task` and carries a non-empty digest; fix tasks filed ≤ tests that reached `persist_passes`; every cleared or reproduced test has a ledger row. Reports, from ledger rows and run records: episodes by length; filing delay for any real break; per test, how often it was held back and whether a one-pass red recurred; the record directory's size; whether a task agent could read the record paths.
- **Two follow-ups filed at decompose, `deferred`, each with a deferral record naming ζ's report as its flip condition:** (1) the debt rule for lane-held tests, including what a cancelled owner means, as an amendment to `plans/flake-ledger-prd.md` §5.9; (2) a typed provider-unavailable outcome for lane tests and a lane hold for it (esc-6280-3 option C; task 6813; solar-challenge task 624).
- **Companion corrections at decompose:** the disposition table in §8; the dependency re-points for 5655, 5412, 5423 and 6650; the rewrite of 5655's work items 2–3 and signal.

Dependency DAG: α → β; β, γ → δ1; α, β, γ, δ1 → δ2; δ2 → δ3 → ε → ι → ζ.

## 10. G7 walk (advisory, author mode)

| Invariant | Reading |
|---|---|
| `contracts-machine-checked` | The submit outcome and the pass verdict become types; scenario 24's fixture is checked on both sides. |
| `structured-facts-at-failure` | Today a reader gets test ids and nothing else; §5.1 keeps the exit code, kind and failure line the emitter knew. ζ reads records and stores, never the journal. |
| `corroborate-before-acting` | Task and escalation status are read live at every decision; an unreadable answer defers (D10, §5.8). |
| `storm-escape-required` | **Waived for D6** (rationale there). A persistent red is bounded by D11; the recheck by D12's per-head cap; a held episode by its expiry. |
| `no-lockstep-duplication` | One command builder, one verdict vocabulary, one ledger, one task-client protocol, one submit parser, one owner-liveness classifier. The lane's runner is separate from verify's on purpose (D3). |
| `status-matches-liveness` | The lane writes task status only by filing. |
| `holds-owned-and-bounded` | A held episode is owned by the lane, bounded by the recheck and the expiry, and visible in the episode record; `filing_due` is bounded by the poll tick; a cancelled-but-red test is bounded by the D11 escalation, whose resolution releases it. |
| `loop-thread-occupancy-bounded` | No whole-corpus task read is added; status reads are by id. Record parsing is per sub-run; the leaf's architect moves it off the loop if it is not trivial. |
| `one-fact-one-home` | Task status lives in the task store and is pointed at; counted passes live in the episode record; ended episodes' history lives in the ledger; the fingerprint and three sidecars, which restated them, go. |
| `guards-exercise-behaviour` | ι runs the real worker against a fixture command; ζ reads production rows. |
| `no-silent-fail-soft` | `Unconfirmable`, `inconclusive`, `unreadable` and `filing_due` replace places where a failure read as "flake", "green" or "nothing to do"; the numeric leg's unnamed clear is counted. |
| `exceptions-owned-or-ratified` | No allow-list is added; one role joins an existing registry and leaves another, both in δ2. |
| `readers-prove-their-producer` | ε's report and ζ's checks read rows the lane wrote. |

## 11. Out of scope

- Changing the bake-off live test, or any test (Leo, 2026-10-12).
- Opening de-flake debt from the lane (§9 follow-up 1).
- Typed outage classification (§9 follow-up 2).
- The live CLI test on the operator's Claude login (task 6813).
- A lane run timeout on `LaneCommand`; `parse_config_command` for `uv run --extra` and `timeout` prefixes (task 6068).
- `fused-memory` changes beyond one test reading the shared fixture: the curator, the unique index and the cancelled-rows rule are untouched.
- The "three mechanisms for one variability axis" finding on `offline_lane.py` (review-all `fk-591206484ea5`); the new modules reduce it and do not claim to close it.
- A cancel provenance on tasks. D10 does not need one.

## 12. Open questions (tactical)

1. **The sentinel id's spelling for the D11 escalation.** **Suggested resolution:** the form `harness.py`'s existing sentinel filers use, one per command. Decide in δ2.
2. **Where the typed failure-line parser lives.** **Suggested resolution:** extend the parser in place, with `_extract_failing_test_ids` as a projection of it. Decide in β.
3. **Whether the recheck pass should be narrowed to the held tests.** **Suggested resolution:** no; a whole-command pass keeps one engine. Revisit on ζ's report.
4. **`human_decomposed=True` on planning-mode filings by a machine.** The flake ledger already files this way and no production code reads the flag. **Suggested resolution:** leave it.
5. **Whether a digest carrying traceback paths trips fused-memory's path-scope advisory.** It never blocks creation but may raise an advisory per filing. **Suggested resolution:** the first live filing shows it; if it fires, the digest keeps the failure line and drops file paths. ζ reports it.
6. **What the auto-watcher does with the D11 escalation.** It may resolve, promote or leave it; any of these is within D10's stated cycle. **Suggested resolution:** none needed; ζ reports what happened to the first one.

## 13. META check

*If this PRD is decomposed and queued without further oversight, is the result complete, coherent, cohesive and good?*

- **Complete.** Each of the four ruled items has a leaf: evidence (α), both-runs (β), transients and outages (D5–D7; δ1, δ2, ε, and the two named follow-ups), memory and "cancelled" (γ, δ1, δ2, D10).
- **Coherent.** One identity (command, test), one engine for three legs, one filing path, and a closed table for what a pass means.
- **Cohesive.** It removes more mechanism than it adds: the fingerprint, three sidecars, three escalation triggers, two extractors and the curator hop go; records, episodes and two types arrive.
- **Good.** Every number has a measured basis and a named re-measurement (ζ). The weakest are `persist_passes = 2` and `red_unlanded_hours = 24`, each resting on one real break.

Two decisions a reader may want to reverse before decompose. D6: lane-held reds are counted but get no owning task until ζ reports. D7: an outage lasting two spaced passes still files one task.
