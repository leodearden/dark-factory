# PRD: the offline lane's red path — keep the evidence, confirm what failed, file on persistence, and remember through the task store

PRD-MARKER:offline-lane-red-path

**Project:** dark-factory (orchestrator; the lane code is shared by dark-factory, reify and solar-challenge). **Status:** active, 2026-10-12; decomposed 2026-10-12 (§14).

**Type:** contract resolving three accreted PRDs (`docs/prds/offline-deep-test-lane-worker.md` C3/C4, `docs/prds/offline-deep-test-lane-infra-extension.md` IE-D1, `plans/integration-test-lane-prd.md` §3) and eight piecemeal tasks.

**Approach:** B+H. The lane↔task-store filing seam is load-bearing and has never worked in production; two other PRDs consume what this one changes (the flake ledger and the verify-result cache); three hub files and a ratcheted file move.

**Origin:** task 6782 / esc-6782-2. **Rulings this PRD executes (Leo, 2026-10-12).** First: the fix proceeds as one short PRD, not as task 5757 re-scoped alone, covering in order: keep the evidence; file only what failed in both runs; hold back transients and outages; lane memory (tasks 6050 and 5747) together with a defined meaning for "cancelled". Second, on reading the first committed version (`c4cfd6691d`): a flaky test is a bug that deserves a fix, so flakes found on the lane enter the flake ledger and are filed as hot-path flakes are, through one common implementation; and an outage is escalated, not filed as a fix task. The second ruling supersedes the first version's "the bake-off live test is left unchanged" and its two decisions that held flakes and outages back.

**Code anchors** verified against main `1c0690552b` (2026-10-12); the cited files are unchanged through `3ed5cf4174`. Main moves fast — cite-by-symbol; re-locate at implementation time.

**Review:** authored under `/team` + `/prd` from four substrate reports. A critic pass on the first draft returned 25 findings and a re-check 16 more; the amendment for the second ruling had its own critic pass (18 findings) and re-check (13 more). All are folded in. Three calls in D6 are the author's, made while folding the amendment's findings, and are marked there.

## 1. Goal

When a lane command goes red, the lane keeps what it saw, files a fix task only for tests that are really and persistently red, gives a flaky test an owner through the flake ledger, tells a human about an outage instead of filing work for it, and knows which task covers which test because it was told the task id when it filed.

What an operator observes once this lands:

1. **Every lane pass leaves a record.** `data/orchestrator/offline-lane/<command>/<pass-id>/record.json` names the head, both runs' exit codes and durations, and for a red pass each failing test with its kind and first failure line. Full output sits beside it.
2. **A lane-filed task explains itself.** `get_task` on it shows, per test, the failure text from each run of each counted pass, the passes it was red on, the command, and a reproduce line. A reader can classify it without re-running anything.
3. **One red pass files no fix task.** The episode record shows the test held at 1 of 2. A test reproduced on two spaced passes files exactly one fix task.
4. **A test that passed the detection run is never filed**, whatever the confirm does.
5. **The lane knows what it filed.** Every lane filing has a task id that resolves through `get_task`, recorded on the episode.
6. **A flaky lane test gets an owner.** The first time the lane clears a test, or a held red proves to be nondeterministic (it both failed and passed at one head) with no fix filed, the flake ledger has a debt row for it and an owning de-flake task that says the lane observed it, carries the failure text and gives a command that runs the test.
7. **A red that outlives its task is escalated once per task**, 24 hours after filing, naming the task and its status.
8. **An outage is escalated, not filed.** When a red pass's dependency probe fails there is one open infrastructure escalation for that command, naming the dependency that failed. No task is filed. The lane closes the escalation when the probe passes again.

Consumers (G1): the fix-task and de-flake-task readers (architect and steward agents, and the L2 watcher); the flake-ledger report; the verify-result cache audit (task 5655) through the hook in §5.7; auto-eval (task 5423) and the flake ledger's own filing through the typed submit seam; the soak gate ζ.

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
| One- or two-pass incidents whose test next passed at a different head | 6 of 8 | head of the first pass that shows the test passing |
| Consecutive passes that share a head | 309 of 939 | same head as the previous pass of that command |
| Next pass starting under 8 min after a confirm | 14 of 90 | start minus the previous confirm's end |
| Spacing of consecutive passes | median 24.9 min, p90 64.7, max 386.9; 30 of 939 over two hours | start to start |
| After a stall of the bake-off test, re-stalls 8–45 min later | 0 of 12 runs | Qdrant request log, 10 incidents |
| After a stall in a detection run, the confirm minutes later re-stalls | 2 of 9 | same log, lane passes only |
| Longest a test stayed confirmed red after its task was filed | 13.0 h (n = 1, task 6146) | filing → last confirmed-red pass |
| Lane filing → last write, all eight tasks | 4.2, 4.4, 13.1, 14.2, 15.3, 16.4, 27.6, 89.4 h | ticket `created_at` → task `updated_at` |
| Hot-path flake debts | 35; the owning task is done for 34 and pending for 1; none has re-entered | `runs.db::flake_debt` joined to task status, 2026-10-12 |
| De-flake tasks, all origins | 78, of which 67 done and 1 cancelled | task titles beginning `De-flake`, 2026-10-12 |
| The merge gate's storm alarm, replayed on the lane's clears | about 8 born-at-L2 criticals in the period | 49 clearing passes over 8 process lifetimes, one alarm per 5 since the last reset |
| The same alarm on the hot path | 5 criticals, 09-23 → 10-06, each closed by hand | escalation history of its sentinel |
| The OpenAI outage's path to a human | lane task filed 23:35:55Z; its architect blocked it 23:55:05Z; escalation 23:56:42Z | task 6280 metadata; esc-6280-3 |
| What told the architect it was an outage | one `POST /v1/embeddings` returning 429 at 23:51:16Z | esc-6280-3 detail |
| What the lane's integration tests depend on | 32 files; 7 name FalkorDB only, 1 Qdrant without OpenAI, 1 the Claude CLI; one uses a chat model | service names in files carrying the `integration` marker |

What the code does today (read at the anchor commit):

- `orchestrator/src/orchestrator/offline_lane.py::OfflineLaneWorker._run_once` keeps a sub-run's exit code and drops its output, on all three legs.
- `::_default_confirm_command` re-runs the whole command serially and returns whatever fails in that run. Nothing compares it with the detection run. `plans/integration-test-lane-prd.md` §3 mandated this.
- `orchestrator/src/orchestrator/harness.py::_OfflineLaneTaskClient.submit_fix_task` returns `''` for every production response shape, because `Harness._extract_task_id` reads none of them. `open_fix_tasks`, `_red_advance_counts`, `_promoted_blockers`, the L0 info and the staged L2 have therefore never run.
- The dedup that did work is fused-memory's curator, with the unique index on `candidate_key` as a backstop only until the files tagger changes the key (16 of 24 lane tasks across three projects; rule: stored key ≠ the title-only key). Both treat a cancelled task as covering nothing.
- A fleet restart that kills a detection run is logged `status=FAIL`.
- The lane's three false-premise filings for the bake-off test were each cancelled because the task named a suspect commit range and carried no output: the reader was asked which commit broke a test that no commit had broken.
- `orchestrator/src/orchestrator/flake_recorder.py::_bump_suppression_streak_and_maybe_escalate` counts suppressions cumulatively in process memory, never resets on a clean verify, and at five files a `critical`, level-2 escalation and resets. Each suppression is a merge that landed on a gate that had gone red.

Basis of each number this PRD sets:

- **`persist_passes = 2`, each counted pass starting at least `recheck_secs` after the previous counted pass started.** Replayed over the ten incidents: it files the code break one pass later (+48 min of a 790 min incident), files the long outage 26 min later than an unspaced count would, and, on the recorded passes, files neither the seven one-pass incidents nor the 21-minute cap incident. `3` would file the same incidents, the code break a further 31 min later and the outage 29 min later. One real break is thin evidence: ζ re-measures, and the knob is hot-reloadable.
- **`recheck_secs = 1800`.** Stalls recur within minutes (2 of 9) and not at 8–45 min (0 of 12); 30 min is past the window where they cluster. Without a recheck a held red waits for the next merge.
- **`red_unlanded_hours = 24`.** About 1.8 × the one measured red-after-filing. Provisional; ζ re-measures.
- **Record retention, 14 days.** Twice ζ's window, and the existing lifetime of the lane's durable state.
- **Probe timeout, 180 s.** Each dependency check inside the probe is bounded at 30 s, the longest per-request bound on the probed path; dark-factory's probe has four; 180 s is 1.5 × their sum.
- **Flake filing cap, 8 per observation.** Retired by ε0 (Leo, 2026-10-13; §14 item 24): the cap and its 120-second budget existed only because the merge waited for the filing, and ε0 moves the filing off the observation path. How many distinct tests the lane will file for is unmeasured (the tests behind the 49 cleared passes were never recorded); ζ reports it.

Outside dark-factory nothing above is measured. reify and solar-challenge get the same engine and the same rules through D13.

## 3. Sketch of approach

A **pass** is one execution of one lane command (or legacy leg) at one head: a detection run and, if that is red, a probe where one is declared and then a confirm.

1. **Record.** Every sub-run streams its output to disk and writes a structured record. A structured pytest command also produces junit, used as per-test evidence.
2. **Probe.** If the command declares a dependency probe, a red detection run is followed by it. A failing probe makes the pass an outage: the lane raises one escalation, does nothing else with that pass, and re-runs the probe until it passes.
3. **Confirm.** Where the command can be narrowed, the confirm re-runs only the tests the detection run named, serially. Where it cannot, it re-runs the whole command and keeps only tests that failed in both runs with a compatible kind. Each test is reproduced, cleared or inconclusive; a confirm that cannot answer makes the whole pass unconfirmable.
4. **Hold or file.** A reproduced test starts or extends an **episode**. At `persist_passes` counted passes the lane files one fix task for the tests that crossed the line and have no live covering task. It files synchronously and is told the task id.
5. **Remember.** One durable record per (command, test) holds the episode, the covering task id and any escalation. Task status is read live at every decision.
6. **Own the flaky.** Every verdict goes to the flake ledger through the recorder the merge gate already uses. A cleared test, and a held red that both failed and passed at one head with no fix filed, opens flake debt, and the ledger files the owning de-flake task with the lane's evidence.
7. **Bound.** A test still reproduced 24 hours after its task was filed raises one escalation. A held episode is re-observed after 30 minutes if main has not advanced, and expires if nothing extends it.

## 4. Resolved design decisions

**D1 — Identity is (command, test), not the failing set.** The set fingerprint drifts while one test stays red (on 2026-10-11 three fingerprints overlapped on two tests), and nothing reads `metadata.offline_lane_fingerprint`. Episodes, covering and filing are keyed per test. `workflow.py::compute_failing_test_set_fingerprint` and the three unblessed metadata keys (`failing_tests`, `suspect_ranges`, `offline_lane_fingerprint`) retire; a lane task carries `spawn_context`, `merge_lane` and one `x_offline_lane` object (§5.5).

**D2 — Evidence lives on disk; a digest lives in the task.** Records and full output go under `data/orchestrator/offline-lane/`: not inside the lane worktree, which `git_ops.py::reset_persistent_offline_deep_worktree` cleans every pass, and not under `data/verify-logs/`, whose 500 MiB prune budget is already full. The task's `details` carries a bounded digest that must suffice on its own, because a task agent's sandbox may not reach `data/`. Nothing is lifted from branch `task/4292`: its 2,000-character tail cuts the first failures of a long red, and its scope hint points the wrong way in both decisive cases on record (the one real break touched no file under the command's directory; the outage's range touched six).

**D3 — The confirm narrows when it can and intersects when it cannot.** Narrowing uses `orchestrator/src/orchestrator/verify_cmd.py` (`parse_config_command`, `scope_to`, `serial_pytest`, `with_pytest_timeout`, `render`), which renders `pytest -m integration <ids>` correctly. It does not call `verify.py::confirm_isolated_rerun_verdict`: that function needs a `VerifyResult` and `ModuleConfig`s the lane does not have, runs under verify admission, and scopes with `_scope_to_keyword`, which drops `-m integration` so the re-run selects nothing. The command builder, the verdict vocabulary (`flake_ledger.py::FlakeVerdict`), the recorder and the ledger are shared (D6); the re-run's runner differs because a lane pass is not a verify. An id that does not survive a round trip through the parser (the stdout parser truncates ids at whitespace) makes the pass unconfirmable with that reason.

**D4 — A red pass ends in exactly one of three outcomes.** `Outage(probe)` (D7), `Confirmed(reproduced, cleared, inconclusive)` or `Unconfirmable(reason)`. Unconfirmable: a confirm that exits non-zero naming no test (a thread-mode timeout or a crash); a detection run that is red and names no test, on any leg except the numeric one (D13); a sub-run ended by a signal; an id that fails the round trip. Today the first becomes the `::nonzero-exit` sentinel on the generic leg and a silent flake on the infra leg, and a signal-ended run is logged as a failure. Inconclusive, for one test: it FAILED in detection and ERRORed at setup or collection in the confirm (a confirm ERROR reproduces only a detection ERROR, the rule from task 3880's plan); or the whole-command confirm exited non-zero without completing its session and did not name it. So the confirm's own harness can neither fabricate a reproduction nor clear a real red. An unconfirmable command is a pseudo-test of its own (`<command>::unconfirmable`) with its own episode and a command-level task on persistence. A worker stopped before a pass's outcome is recorded leaves that pass with no effect.

**D5 — File on persistence, over a closed set of observations.** `git.offline_lane_persist_passes`, default 2, hot-reloadable and read at each pass; 1 restores filing on the first reproduced pass. Spacing is measured start to start. What one pass does to a test's episode:

| The pass shows, for test T | Effect on T's episode |
|---|---|
| reproduced, and the pass started at least `recheck_secs` after the last counted pass started | extends (a counted pass) |
| reproduced, sooner than that | none |
| cleared | ends |
| passed: per-test evidence shows T passed; or there is no per-test evidence for T and the detection run exited 0 | ends |
| inconclusive, skipped, deselected or not run; or T is absent from the failing set with no per-test evidence while the detection run exited non-zero | none |
| the pass is unconfirmable | none for T (the command's pseudo-test is reproduced) |
| the pass is an outage | none, except that the episode is marked as having seen one |
| the covering task is done and the head does not contain its commit | none (a stale head) |

A held episode that no pass extends within `red_unlanded_hours` expires; the one knob serves as both that expiry and D11's deadline. Per-test evidence is junit for a structured pytest command, mapped to node ids by a normaliser β owns and tests, and the per-test result lines for the infra leg; where the mapping fails for T there is no per-test evidence for T.

**D6 — Lane flakes enter the flake ledger and are filed, through the hot path's recorder.**

- *One recorder.* The body of `flake_recorder.py::record_merge_flake_suppression` — occurrence rows and due items first, then the live signals — becomes one function that takes the call site and what that call site supplies (§5.8). Since ε0 (Leo, 2026-10-13) the debt filing is not on that path: the function writes the rows and one item per debt-eligible test on the ledger's due list, in one local transaction, and a background drain calls `open_debt` for each item; the cap and the budget are retired. The merge-gate entry delegates to it. The lane calls it once per red pass that is not an outage, and once more for any pass, of any colour, that turns a held test into a recovered one.
- *Two lane triggers for debt.* A **cleared** test, which is the hot path's `passes_in_isolation`. And a **recovered** test: one whose episode ends with no fix task filed, and which is known to have passed at a head where it was reproduced — either the pass that ends the episode is at such a head, or the last pass that showed it passing before the episode was. Same code, both failed and passed, is nondeterminism by construction, in either order. The episode record keeps the head of every pass that reproduced the test, counted or not. An episode with no such head is not a recovery (it may have been fixed by a commit, or been an outage no probe covers), an episode that expires is not one, and an episode marked as having seen an outage pass is not one. A recovery writes no occurrence row of its own: its trace is the earlier `fails_in_isolation` rows and the owner task's metadata. A reproduced test opens no debt. An unconfirmable pass, an inconclusive test, the command pseudo-test and the numeric leg's unnamed clear open none.
- *A stale head for a de-flake owner* (Leo, 2026-10-13). D10's stale-head rule, applied to the ledger: when a test's debt owner is done and the head of the evidence — the pass's head for a clear, the head where it both failed and passed for a recovery — does not contain the owner's fix commit, the evidence predates the fix and the test is not offered for debt, so the ledger neither re-opens the debt nor raises its regression escalation. Its occurrence row is still written. Decided where the cycle would be re-opened: in the ledger's debt drain, against the due item's head, before `open_debt` can re-enter — for every call site, since the drain runs after the observation and the owner can go done in between (ε0) — with D10's own containment read; no fix commit recorded counts as contained, as in D10, and a failed ancestry read leaves the item due. The lane gives each item the head of its evidence (ε2).
- *No debt behind an open fix task.* Debt is not opened for a test while a lane fix task covering it is open; a test that is simply red is not yet a flake question, and two unlinked tasks for one test help nobody. In the other order both can be live, and the fix task's digest names the de-flake owner of its tests.
- *The trigger is recorded.* `flake_ledger.py::open_debt` takes an optional context — call site, trigger, command, working directory, heads, pass ids and evidence — which the task builder renders and stamps into the owning task's metadata. That metadata is how a reader, and ζ, knows a debt was opened by the lane and why. The lane keeps no due list of its own: it makes the recorder call before it records the episode's end, and the trigger is durable on the ledger's due list from that call (ε0).
- *Evidence reaches the de-flake task.* `flake_ledger.py::FlakeSuppression` gains an optional per-test evidence mapping (kind, first failure line, record directory), written into the existing `flake_occurrence.detail` column. The field and the rendering serve every call site; this PRD populates the lane's, and the merge gate's is a follow-up (§9).

Why. The hot path's de-flake tasks get done (34 of 35 owners; 67 of 78 de-flake tasks, one cancelled). The lane's three filings for the bake-off test were cancelled for a different reason: each asked a regression question about a transient failure and carried no output. With evidence in a task framed as a flake, the ledger's rule that a cancelled owner is replaced is the policy working, not a loop.

Three calls made by the author while folding the amendment's review:

1. **The lane does not join the merge gate's storm alarm.** That alarm is cumulative, and means "five suppressions, each of which let a merge land on a gate that had gone red". Replayed on the lane's clears it would have raised about eight born-at-L2 criticals in 23 days for clears that land nothing. A lane clear is not masked: it opens debt with an owner. A rate escalation for the lane belongs to the ledger's health sweep (task 3792), per call site. G7 waiver: `storm-escape-required` — until task 3792 lands, nothing escalates on the lane's clear rate, and the numeric leg's unnamed clears and inconclusive tests are counted but owned by no task.
2. **A recovered test needs proof at one head.** The wider rule, "any held red that later passes", would have opened debt for 5 tests in the period, 6 of 8 incidents recovering at a different head, the Claude-cap incident among them. Under the rule as written, every test that flaked on the record gets an owner, through a clear or a same-head proof, but three of the eight short incidents open nothing by themselves; ζ reports such incidents.
3. **No debt while a covering fix task is open.**

Consequences, stated plainly: the bake-off test gets a de-flake task the next time the lane clears it; a lane test that is cleared again after its de-flake task landed raises the ledger's existing born-at-L2 regression escalation; the first weeks may open several debts at once; since ε0 nothing caps how many.

**D7 — An outage is escalated, not filed: a declared dependency probe decides.** A generic lane command may declare `outage_probe`, a shell command that exits 0 when every external service the command's tests use is usable and exits 75 naming the dependency that is not, with `outage_probe_timeout_secs` (default 180). The probe runs once, straight after a red detection run and before the confirm, in the command's working directory and environment but at normal priority.

- *Probe exits 75 and names a dependency.* The pass is an outage. The confirm is skipped; no episode changes beyond the outage mark; nothing is written to the flake ledger; no task is filed. The lane opens one escalation per command: role `orchestrator-offline-lane`, category `infra_issue`, severity `blocking`, level 1, on the command's sentinel id, carrying the dependency the probe named and the record path. It amends that record only when the probe's result changes.
- *While the last probe result on the lane's record is an outage* the lane re-runs the probe alone every `recheck_secs`, whether or not main advances and whether or not the escalation is still open. When the probe exits 0 the lane resolves its own level-1 record and any pending level-2 whose only member is that record, and runs the command again. A level-2 with other members is left open, amended to say the lane's outage ended. If someone resolves the escalation while the probe still fails, the loop continues without opening another; the next red pass of the command whose probe fails opens one.
- *Probe exits 0, or none is declared.* The pass proceeds as in D3–D6. Without a probe a persistent outage still files one fix task, as bounded by D5, D10 and D11.
- *Anything else* — another exit code, a crash, no dependency named, the timeout — is a broken probe, not an outage. The pass proceeds to the confirm as if no probe were declared, and the lane keeps one escalation open per command saying the probe is broken, so a probe defect cannot hide the command's reds.

What a probe must do: for each dependency, one bounded check that the service does the work the tests need, not merely that it answers — a write and a read on a scratch name for a store, one embedding through the configured embedder, one completion through the configured chat model — resolving endpoints and models through the same configuration the tests load, and printing which dependency failed. dark-factory's `qdrant-integration` gets such a probe; `warm-lane-bash` has no external dependency and gets none.

The probe is run-level and decided by an exit status, so it covers the tests that fail only as a downstream symptom, which a per-test classifier would miss; and it is what the architect of task 6280 ran by hand. Limits: one probe per command, so a failing dependency holds all of that command's reds, including tests that do not use it, until the probe passes; a probe that wrongly fails holds the command's reds, visibly, behind an open escalation; a blip shorter than the gap between the failure and the probe reads as a flake and goes to D6, whose evidence will show the provider's error; the Claude login cap stays with task 6813, whose test now skips under a cap. Provider health owned by production is the larger answer and a follow-up (§9).

**D8 — The lane files synchronously and deduplicates for itself.** It files the way `flake_ledger.py::_ensure_owner_task` does: `submit_task(planning_mode=True)` then `commit_planning`, through the same task-client protocol. The curator leaves the lane's path: it adds a median 70 s, and over 120 lane tickets in three projects its decisions beyond matching the identical set were 13 (11 overlaps between lane tasks, 2 folds into an unrelated task) plus one `create` for a stale head. Tests that cross the threshold on the same pass share one task; tests that cross on a later pass get their own. The title carries the command and the first red head, so it cannot collide with an earlier filing.

**D9 — One typed submit seam for machine filers.** `Scheduler` gains a submit method returning a tagged outcome — a ticket; a task with its id, status and `combined` flag; or an error — parsed with `shared/src/shared/mcp_envelope.py::parse_tool_result`. `flake_ledger.py::FlakeLedgerTaskClient.submit_task` returns that outcome instead of a bare string, and `_ensure_owner_task` treats `combined` into a terminal task as a failed filing. `Harness._extract_task_id`, `chronic_flake.py::extract_task_id` and the `tkt_` prefix guard retire. Accepting `ticket` in the old extractor (task 6050's framing) would have stored a ticket id as a task id.

**D10 — What the covering task's status means when the test is red again.** The binding from a (command, test) to its covering fix task outlives the episode: it lasts until that task is terminal, so a test that flaps while its task is open is still covered. Status is classified with the ledger's existing four-way owner verdict (`flake_ledger.py::_owner_liveness`), moved to a shared home, not re-derived.

| Covering task reads | Lane does |
|---|---|
| open (any non-terminal status except deferred, including blocked and infra-hold) | nothing; the record accumulates passes |
| deferred, and the lane's record says `commit_planning` was never confirmed | finishes the filing |
| deferred otherwise | nothing; whoever deferred it holds it |
| done, with no fix commit recorded, or with one the pass's head contains | the fix did not hold: the binding is dropped, a new episode starts at 1 and its task will name the prior one |
| done, and the head does not contain the fix commit | the pass is a stale head for that test |
| cancelled or absent | no refile while the episode lasts. When the episode ends, or the D11 escalation raised for it is resolved, whichever comes first, the binding is dropped and any new episode starts at 1 |
| unreadable, or the ancestry check itself fails | no decision this pass, logged loudly |

"Cancelled" therefore means, for a fix task: that filing is closed. It neither blocks a later, separately evidenced filing nor triggers anything by itself. A test that stays red after its fix task was cancelled is escalated once and, after that is resolved by whoever resolves it, may be filed again on two fresh spaced passes: at most one such cycle per 24 hours per test, each needing someone to cancel a task. A de-flake task is the ledger's, and the ledger's own rules govern it (D6).

**D11 — One bounded escalation replaces three.** The advance-count trigger, the cancelled/deferred arm (`_TERMINAL_NON_DONE`) and the born-at-L2 critical go; none ever fired. In their place: when a test is reproduced on a counted pass at least `git.offline_lane_red_unlanded_hours` (default 24) after its covering task was filed, and that task is not done with its fix contained, the lane raises one escalation per covering task — role `orchestrator-offline-lane`, severity `blocking`, level 1, category `dependency_discovered`, filed against a per-command sentinel id so it does not alter the fix task's own recovery handling, and carrying the task id, its live status and the record paths. The role joins `escalation/src/escalation/authority.py::PROMOTE_SENTINEL_BOUND_ROLES` so a promotion to level 2 stays on the sentinel. Its state is read from the escalation queue. The L0 info at filing goes, and with it the role's entry in `escalation/src/escalation/classify.py::INFO_L0_MECHANICAL_ROLES`.

**D12 — A held red is re-observed.** For a generic command, if an episode is held below the threshold, a recheck is due `git.offline_lane_recheck_secs` (default 1800) after its last counted pass started, and the lane then runs that command again at the current head unless a pass of it has started since the recheck fell due — at most `persist_passes − 1` rechecks per command per head, whatever the number of held tests, and none while the command's last probe result is an outage (D7's probe loop runs instead). The legacy legs are re-observed at the next advance. A filing that could not be made is kept as filing-due on the record and retried on the poll tick without re-running the command.

**D13 — All three legs use one engine; no reify or solar-challenge file changes.** The legacy legs keep their own commands and gain records, outcomes, episodes, persistence and ledger recording. The numeric leg's detection names no tests (its failing set lives in a manifest inside the build tree), so its outcome comes from the confirm alone: exit 100 with ids → those are reproduced; exit 0 → nothing reproduced and nothing named; any other exit → unconfirmable. The infra leg prints a result line per test, so it intersects like an opaque command and has per-test evidence; a non-zero exit with no failing line is unconfirmable, where today it is silently dropped. The probe exists on generic commands only; a project adopts one in its own config.

## 5. Contract

**5.1 Run record** (`offline_lane_records.py`, new). One per sub-run: `command`, `leg`, `head`, `phase` (`detect` | `confirm`), `started_at`, `ended_at`, `returncode`, `ended_by` (`exit` | `signal`), `output_path`, `junit_path | None`, `failures: [{node, kind: failed | error, first_line}]`, `ids_available: bool`, `outcomes: {node: passed | skipped} | None`, `session_complete: bool`, and on a detect record `probe: {result: up | outage | broken, returncode, failed_dependency, first_line, duration} | None`. Written before the pass's outcome is acted on. The pass's `record.json` also carries a header written when the pass starts and, written after the engine has acted, the pass outcome and its effect on episodes, tasks and debts (§14); ζ reads them. Retention: 14 days, plus any record a live episode or a non-terminal covering task refers to.

**5.2 Pass outcome** (`offline_lane_confirm.py`, new; β defines all three arms). `confirm(command, detect_record) -> Confirmed | Unconfirmable`, with `mode: narrowed | intersected | leg_native`. For `narrowed` and `intersected`, `reproduced`, `cleared` and `inconclusive` partition the detection run's failures. For `leg_native` on the numeric leg, `reproduced` is what the confirm printed and the other two are empty. `Outage` is decided before `confirm` is called (§5.10).

**5.3 Episode record** (`offline_lane_episodes.py`, new, the engine; stored with `guard_state.py`). Per (command, node): `counted_passes: [pass-id]`, `last_counted_started_at`, `first_red_head`, `reproduced_heads: [head]`, `last_green_head`, `had_outage_pass: bool`, `covering_task_id | None`, `planning_committed: bool`, `filed_at | None`, `filing_due: bool`, `escalation_id | None`, `notes: [EvidenceNote]`, `prior_task_ids: [id]`. Per command: `rechecks: {head: count}`, `outage_escalation_id | None`, `last_probe_result`. It is the lane's only red-path state; the three guard sidecars are deleted. The engine exposes one tick, called from the worker's poll loop, which owns rechecks, the probe loop and the retry of due filings; later leaves extend the tick, not the loop. Entries are refreshed on every pass while the episode is live or the covering task is non-terminal, and otherwise expire after the lane's existing 14 days. A store that cannot be read degrades to an empty one with a loud log; the cost is a filing one pass late and, if a covering task was open, possibly one duplicate task. History of ended episodes is not kept here: it is the ledger rows of §5.8.

**5.4 Filing a fix task.** Ordering: write `filing_due` → submit (planning mode) → record `covering_task_id` → `commit_planning` → record `planning_committed` → clear `filing_due`. A `combined` outcome into an open task makes that task the covering task; into a terminal task it is logged as a swallowed filing and treated as cancelled.

**5.5 The filed fix task.**

- Title: `offline-lane[<command>]: <n> test(s) red on <k> passes since <first_red_head, 12 chars> — <first test>`. Command-level: `offline-lane[<command>]: exits non-zero with no reproducible failing test since <head>`.
- Description, short: the command and leg, up to ten tests then a count, the suspect range (last all-green head to first red head), and one reproduce line (the working directory and the narrowed command, or the whole command where it cannot be narrowed).
- Details, the digest: for up to ten tests, the kind and the first failure line (300 characters) from the detection run and the confirm of each counted pass; a table of counted passes (pass id, head, start time, outcome mode); any hook notes; prior lane task ids for the test; any live de-flake owner of the test; the record directories.
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

**5.6 The de-flake task for a lane debt** (built by `flake_ledger.py::build_deflake_task_arguments` from the debt row and the context of §5.8).

- Title: `Flake debt: <test> — <trigger> on the offline lane (<command>), cycle <open_count>`. The cycle number is in the title for every call site, so a re-entry after a finished owner cannot collide with that owner's title.
- Description: what was observed, in the lane's terms (cleared: failed in a lane detection run and passed an isolated serial re-run at the same head; recovered: failed its re-run and passed on a later pass at the same head); that nothing was suppressed and no merge depended on it; the command, working directory and one reproduce line that selects the test (the narrowed lane command — a bare node id is deselected by the package's default options); the heads and pass ids; for up to five of the most recent occurrences, the kind and first failure line.
- The ledger's binding instructions (`RESPONSIBILITY_FIX_ROOT_DEFECT`, `RESPONSIBILITY_REMOVE_FROM_LEDGER`, `NEVER_WIDEN_A_TIMEOUT`) verbatim, plus one sentence for a test of a live dependency: the root defect may be in how the product handles a slow or failed dependency call, and changing that handling is a fix; relaxing the test is not.
- A note that the merge gate does not run this test, so the task's own run of the reproduce line is its evidence, and the lane will observe it after landing.
- Metadata: the ledger's existing keys, plus `flake_debt_call_site`, `flake_debt_trigger`, `flake_debt_command`. Priority: the ledger's, unchanged.

**5.7 Red-handler hook** for task 5655: `on_pass_outcome(command, head, outcome, run_records) -> [EvidenceNote]`, called once per red pass that reaches a `Confirmed` or `Unconfirmable` outcome, after the outcome and before any filing decision. Returned notes (`{source, node | None, text}`) are stored on the named test's episode, or on every reproduced test's when no node is named, and rendered into the digest at filing. A hook that raises costs only its notes.

**5.8 Ledger recording and flake debt** (the recorder is `flake_recorder.py`'s, shared with the merge gate).

What the shared function takes, and what each call site supplies:

| Merge-specific today | The merge gate supplies | The lane supplies |
|---|---|---|
| the observation carried on a `VerifyResult` | that carrier, unwrapped by its entry point | zero or more `FlakeSuppression`s directly, plus recovered tests |
| `merge_sha` on each row | the merge sha | the pass's head |
| the observing `task_id` | the task being merged | none |
| the `merge_flake_suppressed` event | emitted | not emitted |
| the storm streak and its critical escalation | bumped, as today | not bumped (D6, call 1) |
| the regression escalation's wording | a merge-gate suppression re-opened the debt | a lane clear or recovery re-opened it, naming the pass and head |
| which tests may open debt | every test on a `passes_in_isolation` observation | the debt-eligible tests, passed separately: cleared and recovered tests with no open covering fix task |

- Occurrence rows, per red pass that is not an outage: reproduced → `fails_in_isolation`; cleared → `passes_in_isolation`; inconclusive → `unconfirmable` naming the test and the reason. An unconfirmable pass, and the numeric leg's unnamed clear, each write one `unconfirmable` row under the ledger's existing unknown-test sentinel with the reason. Each `FlakeSuppression` carries `call_site = offline_lane`, the local runner, the host pressure reading and the per-test evidence.
- Debt: every cleared test and every recovered test that has no open covering fix task goes on the ledger's due list with its context; the ledger's drain calls `open_debt(test, context)` for each, with no cap and off any pass or merge (ε0). Never for the unknown-test sentinel or a command pseudo-test.
- Ordering for a debt trigger: the recorder call (the pass's rows and due items, in one transaction) → the engine records the episodes' new state. A restart between the two leaves items the drain files and episodes the next pass ends again, a repeat inside the cycle that files no second task; a restart before the call leaves the pass with no effect (D4).
- Order inside the recorder: rows and due items first, live signals second; the filing is the drain's (ε0). The regression escalation is unchanged and is raised from the drain: a re-entry after a finished de-flake task raises `regressed_after_resolution` once.
- The added evidence field is optional on the wire and on read; a `FlakeSuppression` without it, and an `open_debt` without a context, behave as today.

**5.9 Error semantics.** No path treats an unreadable status, a failed submit, a failed ancestry check, a failed ledger write or an unparsed output as a negative answer. Each is a named state (`unreadable`, `filing_due`, an item on the ledger's due list, `Unconfirmable`, `inconclusive`) or the ledger's own loud degrade, defers the decision, and is visible in the record.

**5.10 Outage probe.** `LaneCommand.outage_probe: str | None`, `outage_probe_timeout_secs: int = 180`. Launched in the command's working directory and environment, without the idle-priority prefix, after a red detection run. Exit 0 → up. Exit 75 with one line naming the failed dependency → `Outage`. Anything else, including the timeout → a broken probe: the pass goes on to the confirm. While `last_probe_result` is an outage the engine's tick runs the probe alone every `recheck_secs`; on exit 0 it resolves the records D7 names and marks the command due for a pass. An escalation the lane cannot write is logged loudly and retried at the next probe; the pass is still an outage.

## 6. Boundary-test sketch

Facing the lane (L), the task store (T) and the flake ledger (F). Every L scenario drives the real `OfflineLaneWorker` with a fixture command. Where a scenario has an outcome half and an episode, filing or ledger half, §9 assigns them to different leaves.

| # | Side | Scenario | Asserts |
|---|---|---|---|
| 1 | L | A red pass and a green pass of a fixture command | each leaves a record; the red one names each failing test with kind and first line; the output file exists |
| 2 | L | A sub-run killed by a signal | record `ended_by: signal`; outcome unconfirmable with the signal as reason |
| 3 | L | A test fails detection and the narrowed confirm, pass 1 | no task; the episode holds one counted pass |
| 4 | L,T | The same test reproduced on a pass that starts 30 min after pass 1 started | exactly one fix task; the outcome is a task id, not a ticket; `details` shows both passes; the episode has the id and `planning_committed` |
| 5 | L | The same test reproduced on a pass that starts 3 min after pass 1 started | not counted; no task |
| 6 | L | A test fails detection, passes the narrowed confirm | outcome cleared; its episode, if any, ends; no fix task |
| 7 | L | Opaque command: detection red on A, whole-command confirm red on B only | nothing reproduced; B has no episode |
| 8 | L | Confirm exits non-zero with no ids, on two spaced passes | unconfirmable both times; real tests' episodes untouched; one command-level task on the second |
| 9 | L | Detection FAILED on A; confirm ERROR on A at setup | A is inconclusive: not reproduced, not cleared |
| 10 | L | A held test is skipped on the next pass; then the session is truncated; then it passes | the first two leave the episode unchanged; the third ends it |
| 11 | L | A failing id containing whitespace | the pass is unconfirmable with the round-trip reason |
| 12 | L,T | Tests A and B cross on pass 2; test C crosses on pass 4 while that task is open | one task for A and B; a second for C; no third |
| 13 | L,T | Persistent red with the covering task open, across ten more passes and past 24 h | no second task; no escalation before 24 h; exactly one after; none later |
| 14 | L,T | Covered test goes green, then red twice, while its task is still open | no new fix task |
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
| 25 | F | The ledger and its report after scenarios 4, 6, 8, 9 and 22 | one row per test per red pass at `offline_lane` with the verdict of §5.8; sentinel rows for the unconfirmable pass and the unnamed clear; the report lists `offline_lane` separately and the merge-gate counters are unchanged |
| 26 | L,F,T | The first clear of a test (scenario 6); then a second clear while its task is open | a debt row with an owner task; the task is §5.6's, naming the lane, the trigger and the failure line, with a reproduce line that selects the test; the second clear files nothing |
| 27 | L,F,T | A test held at 1 passes on a recheck pass at the same head, which is green; separately, a test that passed at head H is reproduced once at H and next passes at another head | each episode ends; the recorder is called for the ending pass, green or not; a debt row and an owning de-flake task open with trigger `recovered`; no new occurrence row is written for the recovery |
| 28 | L,F | A test first reproduced at a head where it never passed next passes at a different head; another held episode expires; a third contained an outage pass | none opens debt |
| 29 | L,F | Nine tests cleared in one pass | nine items on the ledger's due list and no network call on the pass; after the drain runs, nine de-flake tasks |
| 30 | L,F | Six red lane passes that each clear a test, interleaved with five merge-gate suppressions | no storm escalation from the lane's clears; exactly one from the merge gate's five |
| 31 | F,T | A test whose de-flake task is done is cleared again, with that task still on its title-only key | the ledger's regression escalation fires once, worded for the lane; a new owner task is filed with `cycle 2` in its title and is not swallowed |
| 32 | F | A merge-gate suppression with no evidence field and no context, before and after ε1 | the same rows, the same debt, the same storm escalation at five, and a task whose text is the merge gate's apart from the cycle number |
| 33 | L,F,T | A test with an open covering fix task is cleared; later the fix task is done and the test is cleared again | no debt the first time; debt the second |
| 34 | L,F | Restart after the recorder call for a recovered test and before the episode store is written | the drain opens the debt; the next pass ends the episode again and files no second task |
| 35 | L | Red detection; the declared probe exits 75 naming a dependency; a second such pass with the same result | outcome outage; the confirm is not run; held episodes gain the outage mark and nothing else; no ledger or task effect; one escalation naming the dependency; no second escalation and no amendment |
| 36 | L | After 35 with main idle | the probe alone re-runs every `recheck_secs`; when it exits 0 the lane resolves its level-1 record and a level-2 whose only member that record is (minted the way the auto-watcher mints it), then runs the command; a level-2 with a second member is amended, not resolved |
| 37 | L | The escalation is resolved by hand while the probe still fails | the probe loop continues and opens nothing; the next red pass whose probe fails opens exactly one new escalation |
| 38 | L | The probe crashes, exits with another code, names no dependency, or times out; a command with no probe goes red | broken probe: the confirm runs, the pass is handled normally and one escalation says the probe is broken; never outage |
| 39 | L | The probe script against live stores; against an unreachable store; against a store that answers but refuses a write | exit 0; exit 75 naming that store; exit 75 naming that store |
| 40 | L,F,T | A test whose de-flake task is done is cleared on a pass whose head predates that task's fix commit; later it is cleared at a head that contains it | the first clear writes its occurrence row and opens no debt and no regression escalation; the second re-opens the debt and raises the regression escalation once; likewise when the owner goes done after the clear was recorded and before the ledger acts on it |
| 41 | L,F,T | fused-memory unreachable when a lane clear and a merge-gate suppression make debt due | no lane pass and no merge waits on it; both items stay due and are retried with a growing backoff; one escalation names them after 30 minutes; both debts get owners when fused-memory returns, and the drain resolves its escalation |

## 7. Pre-conditions (G3) — verified against main `1c0690552b`

| Capability | Evidence |
|---|---|
| Narrowed pytest commands | `verify_cmd.py::parse_config_command`, `::scope_to`, `::serial_pytest`, `::with_pytest_timeout`, `::with_junitxml`, `::render` exist; both dark-factory lane commands parse structured and render a runnable `pytest -m integration <ids>` (run 2026-10-12). `worker_id` survives `-p no:xdist` in fused-memory through its own conftest fixture |
| Failure-line parse | `verify.py::_extract_failing_test_ids` exists; it returns ids without kinds and truncates at whitespace. **The typed parse and the round-trip check do not exist** — α builds the typed parse by extending that parser, and β the round-trip check (§14) |
| Per-test outcomes | `verify.py::_extract_failing_test_ids_from_junit` reads junit failures; reify's `tests/infra/run_all.sh` prints `RESULT: PASS | FAIL | SKIP (name)`. **A junit passed/skipped reader and a junit↔node-id normaliser do not exist** — β builds them |
| Synchronous create | `submit_task(planning_mode=True)` returns `{'task_id', 'status': 'deferred', 'planning_mode': True}`, and on a title collision `{'task_id', 'status', 'combined': True, 'planning_mode': True}`; `commit_planning` exists; tasks filed this way dispatch normally |
| Task-client protocol | `flake_ledger.py::FlakeLedgerTaskClient` and `chronic_flake.py::SchedulerChronicFlakeTaskClient` exist; the client is constructible from what `Harness._start_offline_lane` holds. **Its `submit_task` returns a bare string** — γ changes it |
| Envelope parser | `shared/src/shared/mcp_envelope.py::parse_tool_result` exists |
| Ledger API | `flake_ledger.py::record_flake_occurrence`, `::open_debt`, `::_owner_liveness`, `::FlakeCallSite`, `::FlakeSuppression`; `flake_report.py::compute_gate_blind`, `::compute_systemic` exist. `flake_occurrence.detail` exists, written as one string per observation; an `unconfirmable` row may name a test; `::flake_suppression_from_wire` projects onto declared fields, so an added optional field is safe in both directions. **`FlakeCallSite.offline_lane`, per-test evidence in `detail`, a context on `open_debt`, and the per-call-site partition do not exist** — ε1 adds them |
| Shared recorder | `flake_recorder.py::record_merge_flake_suppression` exists with one caller (`merge_lane/worker.py`); it takes a `VerifyResult` and is merge-specific in the ways §5.8 lists; its cap applies through `::_unowned_first`. **A call-site-parametrised body does not exist** — ε1 extracts it |
| De-flake task text | `flake_ledger.py::build_deflake_task_arguments` takes a `DebtRow` only; its text says a merge landed and its title is the same on every cycle; `flake_debt` has no call-site column, and none is added (the owner task's metadata carries it). Of 30 `Flake debt:` tasks, 4 still hold their title-only key, so a re-entry's filing could collide. **Context rendering and the cycle in the title do not exist** — ε1 adds them |
| Host pressure reading | `shared/src/shared/psi.py` exists; ε2 uses the read the verify discriminator uses |
| Fix-contains-head check | `git_ops.py::GitOps.is_ancestor` exists; it returns false for a git error as well as for "not an ancestor". **A read that tells the two apart does not exist** — δ2 adds it. `done_provenance.commit` is optional: 231 of 3,818 done tasks with provenance carry none |
| Status partition | `shared/src/shared/task_statuses.py::TERMINAL`, `::ACTIVE` exist |
| Escalation queue | severities `info`, `blocking`, `critical`, `urgent` (`escalation/src/escalation/models.py::KNOWN_SEVERITIES`); category `infra_issue` is in use; `harness.py` already files `blocking` level-1 records on sentinel ids under `orchestrator-` roles and resolves its own; `queue.py::amend` drops identical text and sheds the oldest past 20; the orphan reaper acts on level 0 only and the level-2 revalidation sweep skips `infra_issue`; the auto-watcher's rule for `infra_issue` is to promote, leaving the level-1 record pending as a member of a level-2 that carries the watcher's role and, once the lane's role is sentinel-bound, the sentinel id; `queue.py::resolve` is callable in-process |
| Probe | **`LaneCommand` has no probe field and dark-factory has no probe script** — π adds both. `git.offline_lane_commands` is hot-reloadable, and older code ignores an unknown key |
| Signal-ended runs | a child ended by a signal reports a negative return code; the lane launches through `sh -c`, where a killed grandchild surfaces as 128 + n. α pins both forms in scenario 2 |
| Durable small state | `guard_state.py::PersistentMap` holds JSON values; an entry expires a caller-supplied time after its last write (the lane passes 14 days) |
| Lane knobs hot-reload | `git.offline_lane_red_advances_before_blocker` is in `config.py::RELOADABLE_FIELDS`; the three new knobs join it and that one leaves. No project config sets it |

Not assumed: a task read by metadata (none exists); a structured cancel reason (none exists); narrowing for solar-challenge's commands (they parse opaque; task 6068 would not change that for the three that start with `timeout`); a lane run timeout (`LaneCommand` has none); detection ids on the numeric leg; a harness flag saying a stop is in progress; a provider health gate for OpenAI (none exists; `shared/api_health.py` covers Claude only); failure evidence on merge-gate occurrences (the discriminator does not produce it); the ledger's health sweep (task 3792, pending); a merge gate that runs `integration` tests (it deselects them, so a de-flake task for a lane test is not checked by its own gate); an orchestrator-side fixture for a real fused-memory task backend (ι uses the typed client over the scenario-24 fixture).

## 8. Cross-PRD relationship (G4)

Dated pointers: δ3 writes them, except the flake ledger's (ε1) and the probe fields' addition to the integration-lane pointer (π).

| Other PRD or task | Direction | Seam | Owner | Changed by |
|---|---|---|---|---|
| `docs/prds/offline-deep-test-lane-worker.md` | amends | C3 fingerprint dedup, C4 escalation staging, boundary rows B4–B7 | this PRD | δ2; pointer |
| `plans/integration-test-lane-prd.md` | amends | §3 "Confirm (flake filter)": whole-command serial re-run; `LaneCommand` gains the probe fields | this PRD | β, π; pointer |
| `docs/prds/offline-deep-test-lane-infra-extension.md` | amends | IE-D1: infra failures ride the fingerprint dedup unchanged | this PRD | δ2; pointer |
| `plans/flake-ledger-prd.md` | extends | a fourth call site (§5.3 vocabulary); the recorder's body shared across call sites, with the storm alarm staying the merge gate's (§8.3); a second debt trigger at the lane and a context on `open_debt` (§5.9); optional per-test evidence on the observation, in `detail` and on the wire (§8, §8.4); the de-flake task built from the debt row and that context, with the cycle in its title; both health counters partitioned by call site; §8.1 "one implementation, all call sites" gains the lane's own re-run runner (D3); `FlakeLedgerTaskClient.submit_task`'s return type (D9); `_owner_liveness` moves to a shared home | flake ledger owns the vocabulary, the debt invariant, the recorder and the report; this PRD owns the lane call site, the typed outcome, the context and the evidence field | γ, δ1, ε1, ε2; pointer |
| Open flake-ledger tasks on the same symbols: 3792 (health sweep), 3794, 3795, 4976, 5089 (another call site), 5257, 5850 (the builder's instruction text), 5875 (the report), 6343 | serialises with | `flake_ledger.py`, `flake_recorder.py`, `flake_report.py` | each task keeps its scope; none is folded | decompose wires an edge between ε1 and each that edits the same symbol. For 3792: its three health classes are computed per call site, and the lane's rows never enter the merge gate's rates |
| `plans/local-embedder-cutover-prd.md` (tasks 6797–6810) | relies on | which embedder the tests and the probe use | the cutover PRD | the probe reads the configured embedder, so the cutover needs no probe change |
| `plans/verify-result-cache-prd.md` ε (task 5655) | produces for | §5.7 hook; that PRD's "blocker after three confirmed-red advances", "lane engine untouched" and "a fix task carrying the leak record" | this PRD owns the hook; 5655 consumes it | 5655 depends on δ2 in place of 4226, and its work items 2–3 and signal are rewritten at decompose; pointer |
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
| 5757 | superseded: remedy (a) is replaced by β and D5; remedy (b) is not adopted, and D6 and D10 say why its concern no longer arises |
| 6650 | kept; depends on β, because its lane call site moves into the new confirm module |
| 6068, 6813, 4078 | kept, independent; 4078's sentinel name becomes the command pseudo-test |

## 9. Decomposition plan

Sizes follow the overlay bands. Labels are Greek; ids are assigned at decompose. "HUB" marks a hand-carry. `offline_lane.py`, the four lane test files and `test_merge_queue_flake_recorder.py` are ratcheted; every leaf that touches a ratcheted file carries `orchestrator/tests/merge_lane_ratchet_baseline.json` and `orchestrator/tests/merge_lane_ratchet_authorized_raises.json`, and the lane leaves move the run, confirm and red-path code out into the new modules so `offline_lane.py` shrinks. The chain is a line, so no two leaves edit `offline_lane.py`, an engine module, `config.py` or a ledger module at once. α through ε2 are intermediates roped into the integration gate ι; ζ is the leaf an operator observes.

- **α — Run records: every lane sub-run leaves evidence (no hub; dispatchable).** [~600 LOC; `offline_lane_records.py`, `offline_lane_runs.py` (new; the detection run seams move here, and every sub-run launch streams to file through one launcher), the typed extension of the failure-line parser (kind and first line), `offline_lane.py`, the two ratchet files, tests]. Unlocks β. Signal: scenario 1 and the record half of scenario 2.
- **β — The confirm confirms (no hub; dispatchable).** [~800 LOC; `offline_lane_confirm.py` (new; the three-arm outcome type), the reproduction rule and the round-trip check over α's typed parse, the junit outcome reader and id normaliser, `offline_lane.py` (confirm seams out), the two ratchet files, tests]. Depends α. Unlocks δ1, δ2. Signal: the outcomes of scenarios 2, 6–11 and 22.
- **γ — One typed submit seam for machine filers (HUB: `harness.py`).** [~600 LOC; `scheduler.py`, `flake_ledger.py` (protocol return type, `_ensure_owner_task`, prefix guard removed), `chronic_flake.py`, `harness.py` (the auto-eval call site; `_extract_task_id` loses that caller and is deleted by δ2 with the lane's old client), twelve test fakes, one fixture file shared with a fused-memory test]. No dependency. Unlocks δ1, task 5423. Signal: scenario 24.
- **δ1 — Episode engine and the filed task (no hub; dispatchable).** [~800 LOC; `offline_lane_episodes.py`, `offline_lane_filing.py`, `offline_lane_escalations.py` (new; the last holds the D11 record and the sentinel), `flake_ledger.py` and `shared/src/shared/task_statuses.py` (`_owner_liveness` moves), unit tests against fake records and a fake typed client]. Depends β, γ. Unlocks δ2. Signal: the D5 and D10 tables as table-driven tests over the engine, and the §5.5 payload against the worked example.
- **δ2 — Cutover (HUB: `harness.py`; hand-carry).** [~1,300 LOC, most of it test migration; `offline_lane.py` (red path replaced by the engine; sidecars, `_maybe_promote_blocker`, `_file_info_escalation`, `_file_blocker_escalation`, `_TERMINAL_NON_DONE` deleted; `run` calls the engine's one tick, which owns the recheck), `offline_lane_pass.py` (new; the per-pass pipeline and the §5.7 hook), `harness.py` (`_OfflineLaneTaskClient` and `_extract_task_id` deleted; the lane wired to the machine-filer client), `git_ops.py` (an ancestry read that reports a git error apart from "no"), `escalation/src/escalation/classify.py`, `escalation/src/escalation/authority.py` and their tests, the four lane test files, `test_restart_durable_guards.py`, the two ratchet files]. Depends α, β, γ, δ1. Policy values are the engine's defaults until δ3. Signal: the episode and filing halves of scenarios 3–6, 8, 10 and 12–21, and scenario 23.
- **δ3 — Knobs, removals and documents (HUB: `config.py`, `workflow.py`; hand-carry).** [~400 LOC; `config.py` (three knobs in, one out, reload list), `offline_lane_episodes.py` (policy read from config through δ1's one function), `workflow.py` (two helpers removed) and the stale references to them (`chronic_flake.py` docstrings, `shared/src/shared/task_metadata.py`, `test_workflow_offline_lane_fix.py`), `OPERATIONS.md` (a lane red-path runbook section), `ARCHITECTURE.md` §5.4, dated pointers in seven PRDs marked in §8]. Depends δ2. Signal: a reload changes `persist_passes` on a running worker and the next pass uses it (the existing config-reload test pattern).
- **π — Outage probe (HUB: `config.py`; hand-carry).** [~550 LOC; `config.py` (`LaneCommand.outage_probe`, its timeout), `offline_lane_runs.py` (run the probe), `offline_lane_pass.py` (the probe step), `offline_lane_escalations.py` (the two records), `offline_lane_episodes.py` (the `Outage` and broken-probe handling; the outage mark on held episodes; the probe loop on the engine's tick; the per-command escalation opened, amended on change and resolved with a level-2 whose only member it is), `fused-memory/scripts/lane_dependency_probe.py` (new; D7's per-dependency checks through the tests' own configuration), `dark-factory-orchestrator.yaml` (the probe on `qdrant-integration`), `OPERATIONS.md` (the runbook's outage paragraph), the probe fields in the integration-lane pointer, tests]. Depends δ3. Signal: scenarios 35–39.
- **ε0 — Flake-debt filing leaves the observation path (HUB: `harness.py`; hand-carry; added 2026-10-13, §14 item 24).** [~600 LOC; `flake_ledger.py` (the due list), `flake_recorder.py` (enqueue; cap, budget and task client retired), `flake_debt_drain.py` (new; the drain, the regression escalation moves here, a stale-item escalation), `flake_report.py` (`filing pending`, the due list's age), `harness.py` (registers the drain), `merge_lane/worker.py` (no task client for the recorder), tests, the two ratchet files, the flake-ledger PRD pointer, OPERATIONS.md]. Depends π. Unlocks ε1. Signal: scenario 41's merge-gate half; scenario 29's merge-gate form.
- **ε1 — The ledger side: one recorder body, context and evidence (no hub; dispatchable).** [~700 LOC; `flake_recorder.py` (the shared function of §5.8; the merge entry delegates), `flake_ledger.py` (`FlakeCallSite.offline_lane`, per-test evidence in `detail` and its wire read, the context on `open_debt`, the builder of §5.6 with the cycle in the title), `flake_report.py` (both counters by call site), `plans/flake-ledger-prd.md` (dated pointer), `test_flake_recorder.py`, `test_merge_queue_flake_recorder.py`, `test_flake_ledger.py` (its one-connection-per-call bound either holds or is restated), the two ratchet files]. Depends ε0. Unlocks ε2. Signal: scenario 32; the builder's output for a cleared and a recovered context against §5.6; the report's partition over seeded rows of two call sites.
- **ε2 — The lane side: clears and recoveries reach the recorder (no hub; dispatchable).** [~450 LOC; `offline_lane_episodes.py` (one recorder call per non-outage red pass and per pass that makes a recovery; the one-head rule and `reproduced_heads`; each due item's evidence head; the recorder call before the episode store write; the debt-eligible list; the live de-flake owner in the fix digest), tests]. Depends ε1. G7 waiver: `storm-escape-required` — D6, call 1 (the lane's clear rate escalates nowhere until task 3792 lands). Signal: scenarios 25–31, 33, 34 and 40.
- **ι — Integration gate (no hub; dispatchable).** [~1,500 LOC in two modules; `orchestrator/tests/test_offline_lane_red_path_integration.py` (scenarios 1–23) and `orchestrator/tests/test_offline_lane_ledger_and_outage_integration.py` (scenarios 25–38), both new, with the two ratchet files]. Depends ε2. Signal: those scenarios execute against the real `OfflineLaneWorker`, the typed client over the scenario-24 fixture and the real ledger; scenario 24's other half runs in fused-memory and scenario 39 in π.
- **ζ0 — Soak report script (no hub; dispatchable; added at decompose, §14).** [~400 LOC; `scripts/offline_lane_soak_report.py` and its test]. Depends ε2. Signal: the script prints one verdict per check of ζ, read from the pass records' outcome and effect blocks, with the count it examined and the rule it applied, and exits 0, 1 or 2 (nothing to judge).
- **ζ — Soak gate (operational; a deterministic pure gate with a seven-day delayed milestone, whose runbook runs ζ0's script).** Depends ι, ζ0 and seven days of the deployed lane, dark-factory only. Checks, read from run records, the ledger, the escalation queue and the task store, not the journal: every pass since deploy has a record; every lane filing id resolves through `get_task` and carries a non-empty digest; fix tasks filed ≤ tests that reached `persist_passes`; every cleared or reproduced test has a ledger row; every task whose metadata says the lane opened its debt carries evidence. Reports: episodes by length; filing delay for any real break; debts the lane opened, by test and trigger, and what became of each owner task; clearing passes per day and held reds that ended at a different head (the reds D6 leaves unowned); regression escalations at the lane's call site; outage passes, probe results and how each outage escalation closed; the record directory's size; whether a task agent could read the record paths.
- **Two follow-ups filed at decompose:** (1) an ordinary task: the merge gate's discriminator populates the evidence field, so hot-path de-flake tasks carry failure text too; (2) `deferred`, as a PRD stub with a deferral record naming the embedder cutover's gate (task 6807) as its flip condition: provider health owned by production, for whichever embedder and chat model are configured, which raises the infrastructure escalation itself and which a lane probe can consult (esc-6280-3 option C; solar-challenge task 624).
- **Companion corrections at decompose:** the disposition table in §8; the dependency re-points for 5655, 5412, 5423 and 6650; the rewrite of 5655's work items 2–3 and signal; the edges between ε1 and the open flake-ledger tasks named in §8; a note on task 3792 that its classes are per call site.

Dependency DAG: α → β; β, γ → δ1; α, β, γ, δ1 → δ2; δ2 → δ3 → π → ε0 → ε1 → ε2 → ι → ζ; ε2 → ζ0 → ζ.

## 10. G7 walk (advisory, author mode)

| Invariant | Reading |
|---|---|
| `contracts-machine-checked` | The submit outcome and the pass outcome become types; scenario 24's fixture is checked on both sides; the absence of the evidence field and of the debt context is pinned by scenario 32. |
| `structured-facts-at-failure` | Today a reader gets test ids and nothing else; §5.1 keeps the exit code, kind and failure line the emitter knew, and both task kinds receive them. The outage decision reads an exit status and a named dependency. A debt's origin is task metadata. ζ reads records and stores, never the journal. |
| `corroborate-before-acting` | Task and escalation status are read live at every decision; an unreadable answer defers (D10, §5.9). The probe is an independent check of the dependency before a red is acted on. A recovery needs the same code to have both failed and passed. |
| `storm-escape-required` | **Waived for the lane's clear rate** (D6, call 1): a clear opens owned debt through the ledger's drain, and its rate escalation is task 3792's. A persistent red is bounded by D11; the recheck by D12's per-head cap; a held episode by its expiry; a failing probe by its open escalation. |
| `no-lockstep-duplication` | One command builder, one verdict vocabulary, one recorder, one ledger, one de-flake task builder, one task-client protocol, one submit parser, one owner-liveness classifier. The re-run's runner is separate from verify's on purpose (D3). |
| `status-matches-liveness` | The lane writes task status only by filing. |
| `holds-owned-and-bounded` | A held episode is owned by the lane, bounded by the recheck and the expiry, and visible in the episode record; `filing_due` is bounded by δ1's stuck-write escalation and the ledger's due list by the drain's stale-item escalation (§14 items 16 and 24); a cancelled-but-red test is bounded by the D11 escalation, whose resolution releases it; an outage hold is owned by its open escalation and ended by the lane's own probe loop, which does not wait for a merge. |
| `loop-thread-occupancy-bounded` | No whole-corpus task read is added; status reads are by id. Record parsing is per sub-run; the probe is a subprocess with a timeout; the leaf's architect moves parsing off the loop if it is not trivial. The ledger's debt drain (ε0) handles at most 50 items per pass, stops at the first transport failure and after 5 minutes, and runs its sqlite calls off the loop thread. |
| `one-fact-one-home` | Task status lives in the task store and is pointed at; counted passes live in the episode record; flake history and debt live in the ledger; a debt's origin lives on its owner task; the fingerprint and three sidecars, which restated them, go. |
| `guards-exercise-behaviour` | ι runs the real worker against a fixture command; π runs its script against live, unreachable and write-refusing stores; ζ reads production rows. |
| `no-silent-fail-soft` | `Outage`, `Unconfirmable`, `inconclusive`, `unreadable`, `filing_due` and the ledger's due list replace places where a failure read as "flake", "green" or "nothing to do"; the numeric leg's unnamed clear is counted. |
| `exceptions-owned-or-ratified` | No allow-list is added; one role joins an existing registry and leaves another, both in δ2. |
| `readers-prove-their-producer` | ε1's report partition, ε2's rows and ζ's checks read rows the lane wrote. |

## 11. Out of scope

- Changing any test in this PRD's own leaves. The de-flake tasks the ledger files will change tests or product code, as they do on the hot path.
- A rate escalation for lane clears (task 3792).
- Evidence on merge-gate occurrences (§9 follow-up 1).
- Provider health owned by production (§9 follow-up 2).
- The live CLI test on the operator's Claude login (task 6813).
- Probes for reify and solar-challenge; each project declares its own.
- A lane run timeout on `LaneCommand`; `parse_config_command` for `uv run --extra` and `timeout` prefixes (task 6068).
- `fused-memory` changes beyond one test reading the shared fixture and the probe script: the curator, the unique index and the cancelled-rows rule are untouched.
- The ledger's instruction text itself (task 5850).
- The "three mechanisms for one variability axis" finding on `offline_lane.py` (review-all `fk-591206484ea5`); the new modules reduce it and do not claim to close it.
- A cancel provenance on tasks. D10 does not need one.

## 12. Open questions (tactical)

1. **The sentinel id's spelling for the lane's two escalations.** **Suggested resolution:** the form `harness.py`'s existing sentinel filers use, one per command, shared by D11 and D7 and told apart by category. Settled at decompose (§14): `__offline_lane__<slug of the command name>`, defined in δ1.
2. **Where the typed failure-line parser lives.** **Suggested resolution:** extend the parser in place, with `_extract_failing_test_ids` as a projection of it. Decide in β.
3. **Whether the recheck pass should be narrowed to the held tests.** **Suggested resolution:** no; a whole-command pass keeps one engine. Revisit on ζ's report.
4. **`human_decomposed=True` on planning-mode filings by a machine.** The flake ledger already files this way and no production code reads the flag. **Suggested resolution:** leave it.
5. **Whether a digest carrying traceback paths trips fused-memory's path-scope advisory.** It never blocks creation but may raise an advisory per filing. **Suggested resolution:** the first live filing shows it; if it fires, the digest keeps the failure line and drops file paths. ζ reports it.
6. **Whether reading recent occurrences inside the debt filing keeps the ledger's one-connection-per-call bound.** **Suggested resolution:** pass the evidence in the context so the builder reads nothing. Decide in ε1.
7. **Whether a de-flake agent can run a live integration test from its worktree.** The reproduce line assumes it can reach the stores and has the key. **Suggested resolution:** ζ reports the first lane de-flake task's outcome.

## 13. META check

*If this PRD is decomposed and queued without further oversight, is the result complete, coherent, cohesive and good?*

- **Complete.** Each ruled item has a leaf: evidence (α), both-runs (β), transients (D5; δ1, δ2), flakes (D6; ε1, ε2), outages (D7; π), memory and "cancelled" (γ, δ1, δ2, D10).
- **Coherent.** One identity (command, test), one engine for three legs, one filing path for fix tasks, the hot path's recorder for flakes, and a closed table for what a pass means.
- **Cohesive.** It removes more mechanism than it adds: the fingerprint, three sidecars, three escalation triggers, two extractors and the curator hop go; records, episodes, a probe and two types arrive, and the flake recorder gains a second caller in place of a second damper.
- **Good.** Every number has a measured basis or is inherited from the mechanism it joins, and has a named re-measurement (ζ). The weakest are `persist_passes = 2` and `red_unlanded_hours = 24`, each resting on one real break, and the flake filing volume, which is unmeasured.

Consequences accepted with the second ruling: the bake-off test, and any other lane test the lane clears, gets a de-flake task; a de-flake that does not hold raises the ledger's regression escalation; a lane command without a probe still files one fix task for a persistent outage. The three author's calls in D6 have not been seen by the owner and are the ones to reverse if they are wrong; reversing any of them changes ε1 and ε2 only.

## 14. Decompose record (2026-10-12)

Decomposed by a sibling `/team` + `/prd` session against main `2178c2badb`. Between the anchor commit `1c0690552b` and that head no cited file changed except `shared/src/shared/cli_invoke.py`. Five drafting seats re-read every substrate symbol §7 names. A critic then reviewed the integrated batch before filing: 32 findings, three blocking, all folded in below. Manifest: `plans/offline-lane-red-path-prd.capability-manifest.md`, with its stamped YAML twin beside it.

| Label | Task | Status at filing | Prereqs |
|---|---|---|---|
| α | 6857 | pending | — |
| β | 6858 | pending | α |
| γ | 6859 | pending (hub carry) | — |
| δ1 | 6860 | pending | β, γ |
| δ2 | 6861 | pending (hub carry) | α, β, γ, δ1 |
| δ3 | 6862 | pending (hub carry) | δ2 |
| π | 6863 | pending (hub carry) | δ3 |
| ε0 | 6934 | pending (hub carry; added 2026-10-13) | π |
| ε1 | 6864 | pending | ε0 (was π) |
| ε2 | 6865 | pending (G7 waiver `storm-escape-required`) | ε1 |
| ι | 6866 | pending | ε2 |
| ζ0 | 6867 | pending (added at decompose) | ε2 |
| ζ | 6868 | pending (operator gate; delayed milestone, 7 days) | ι, ζ0 |
| follow-up 1: merge-gate evidence | 6869 | pending | ε2 (both edit `verify.py`) |
| follow-up 2: provider-health PRD stub | 6870 | **deferred** (`x_deferral` names task 6807; a decision gate when flipped) | — |
| follow-up 3: the suite census and lane rows | 6871 | pending | ε2 |

Companion corrections applied in the task store:

- Cancelled as superseded, each with a note saying where its work went: 4292, 6028, 6639, 6050, 5747, 6654, 5757, 4226. Branches `task/4292` and `task/5747` are kept as reference.
- Re-scoped: 3880 (to its discriminator half; depends on β), 5655 (rewritten against the §5.7 hook; depends on δ2 in place of 4226), 5423 (drops its parse work and is retitled; depends on γ).
- Re-pointed or newly dependent: 5412 (4226 → δ2, with a note that two of its acceptance criteria describe the deleted red path), 6650 (→ β), 4078 (→ π, the last leaf that edits its files; its validator now also guards the command name as a file-name component).
- Flake-ledger tasks serialised behind ε1: 3792 (with the per-call-site note), 3794, 4976, 5089, 5875, 6343. No edge, by evidence: 3795 (edits no symbol; after ε1 through 3792 and 3794), 5257 and 5850 (different symbols). Notes on all nine.
- Notes on two tasks §8 did not list: 6526 (edits the function δ1 moves) and 5658 (its boundary row 12 changes with 5655).
- Review finding `fk-0238ff63263c` (task 6639) rides on δ2, the leaf after which its anchor is gone.

Re-scopes made at decompose. None changes a §4 decision.

1. **The typed failure-line parse is α's, not β's.** Scenario 1 is α's signal and asserts kind and first line; as §9 was written the parse that produces them landed in β, downstream. β keeps the reproduction rule, the round-trip check and the junit outcome reader. §7 and §9 updated.
2. **"Four run seams" are six launches.** `offline_lane.py` has three detection seams and three confirm seams, each with its own subprocess launch. α moves the detection seams and gives all six one streaming launcher; β moves the confirm seams.
3. **β leaves a transitional projection.** Between β and δ2 the old red path consumes the new outcome: the reproduced set is filed as today; a confirm that exits non-zero naming no test is filed as today's sentinel was, under its new name; every other unconfirmable pass files nothing, including a detection run that names no test, where today the whole-command confirm decided. δ2 deletes the projection.
4. **γ does not touch the lane's old client.** `Harness._extract_task_id` loses its auto-eval caller in γ and is deleted by δ2 together with its last caller. γ therefore edits one ratcheted test file, by a return annotation only, and needs no edge to the α → β chain. §9 updated.
5. **The per-pass pipeline leaves `offline_lane.py`.** δ2 creates `offline_lane_pass.py::run_pass` (detect → confirm → hooks → `LaneEngine.observe`) and the hook list `lane_pass_outcome_hooks`. π inserts the probe step there and ε2 adds its recorder call inside `LaneEngine.observe`, so no leaf after δ2 edits the ratcheted worker. The engine's tick never runs a lane command: it returns `TickResult.commands_due` and the worker runs them. §9 updated.
6. **§12 question 1 is settled here, in δ1.** The D11 record, the role constant and the sentinel live in a new `offline_lane_escalations.py`. The sentinel is `__offline_lane__<slug of the command name>`, one per command, shared by D11 and D7 and told apart by category; the slug is α's, because the id becomes part of a file name and `LaneCommand.name` is unvalidated.
7. **Policy has one seam.** δ1's `lane_policy(config)` returns the defaults; δ3 changes its body to read the three knobs. δ3 does not edit `offline_lane.py`. §9 updated.
8. **The store is written once per pass.** `PersistentMap` rewrites its file on every assignment, so δ1 keeps a command's whole record under one key and writes it off the event-loop thread (`loop-thread-occupancy-bounded`). One key refreshed every pass never ages out, so δ1 applies §5.3's refresh rule itself: one batched status read per pass over the command's bound tasks, and an ended episode whose task is terminal is dropped 14 days after its last refresh.
9. **§5.3 gains fields.** Per command: `last_pass_started_at`, `last_pass_head`, `last_green_head` (D12 and §5.5 need them) and, in π, `probe_broken_escalation_id` (the outage and broken-probe records share a sentinel and a category). Per episode, `last_green_head` is the head of the last pass that showed that test passing, set when the episode starts.
10. **Retention has a seam.** α's prune takes pinned pass ids; δ1's engine exposes them; δ2 connects the two (§5.1's exception).
11. **δ2 and ι do not both run the matrix.** δ2 rewrites the boundary rows that assert the old machine (scenarios 3, 4, 6, 8, 13, 14, 15, 19, 21) and adds 23; δ1 runs the D5 and D10 tables at the engine's interface; ι runs scenarios 1–23 and 25–38 against the real worker.
12. **ζ0 is new.** ζ's mechanical checks had no producer, and a gate cannot be filed against a script that does not exist. ζ0 delivers `scripts/offline_lane_soak_report.py`; ζ is a pure operator gate whose runbook runs it. "Seven days after deploy" has no machine expression: the milestone counts from ι and ζ0 being done, and the runbook checks the deployed window.
13. **Two of the G7 resolutions.** ε1's report prints a never-written line for a call site with no row, because its partition lands before the lane writes one (`readers-prove-their-producer`). ε2 gives the host-pressure mapping one public home in `shared/psi.py` instead of a third copy (`no-lockstep-duplication`).
14. **ι carries the ratchet.** Its two modules import `orchestrator.offline_lane`, so each adds a row to the ratchet baseline, which must byte-match a fresh measurement.
15. **The pass record says what the pass did.** As §5.1 was written, no durable record said which tests a pass reproduced, cleared or counted, and §5.3 keeps no history of ended episodes, yet ζ's checks need both. α declares two pass-level blocks on `record.json`; δ2 writes the outcome and the effect the engine returns; ε2 adds what became of each debt it offered. α also writes a pass header at allocation, so a pass a restart interrupts is a record and not a bare log. §5.1 updated.
16. **A filing that stays due is escalated.** §10 called `filing_due` "bounded by the poll tick"; a retry period is not a bound, and a filing that failed every time, visible only as a log line, was this PRD's premise. When a due entry has stayed due for `recheck_secs`, δ1 raises one level-1 record on the command's sentinel and resolves it itself (`storm-escape-required`, `holds-owned-and-bounded`). ζ0 checks that no test reached the threshold without a task or that record.
17. **`planning_committed` is read back.** `commit_planning` on the task client returns nothing and never raises, so δ1 sets the flag only after a live read shows the task is no longer deferred.
18. **One renderer for the reproduce line** (β's `reproduce_line`), used by δ1's fix task and ε2's de-flake context.
19. **Regression escalations carry the call site** as a structured evidence entry (ε1). The result type this item first gave the recorder is superseded by item 24: with no cap there is nothing the recorder fails to reach, and whether an item was filed is the due list's to say.
20. **π's outage hold has every exit.** The command losing its probe or being disabled ends the outage state; a level-2 that holds the lane's record is found by membership, not by the sentinel id; a resolved record is archived, so its state is read with `get`. The exit code 75 has one home in `shared`.
21. **Documents follow what exists.** δ3 documents records, episodes, knobs and escalations; π adds the outage text; ε2 adds the flake-ledger text.
22. **Task 3792's classes.** Classes 1 and 3 are per call site, from ε1's counters. Class 2 runs over `flake_debt`, which has no call-site column (§7); it names a debt's origin from the owner task's metadata. The note on 3792 says so.
23. **A stale head for a de-flake owner** (Leo, 2026-10-13, after filing). The decompose critic found that D10's stale-head rule had no ledger counterpart: a lane pass started before a de-flake fix landed would re-open the debt and raise a born-at-L2 regression escalation. D6 gains the rule; δ1 (6860) provides the containment read and a public `flake_ledger.py::resolving_commit`; scenario 40 is new, run by ε2 (6865) and ι (6866). Item 24 moved the check itself from ε2 into ε0's drain.
24. **Flake-debt filing leaves the observation path** (Leo, 2026-10-13, after filing; new leaf ε0, task 6934). The merge waited for de-flake filing, and that wait was the only reason for the filing cap and the 120-second budget (task 4974). Both cost something: a test past the cap or the budget got no debt row at all — for a recovered test, which has no next occurrence, a lost flake — and every merge path had to be handed a task client or its suppressions opened no debt (tasks 6343 and 4976 are two such paths). ε0 writes the rows and one item per debt-eligible test on a due list in the ledger, in one transaction, and a background drain calls today's `open_debt` for each, unchanged, so the flake-ledger PRD's owner invariant (§5.9) and its lazy close keep their meaning. Items leave the list only once the test has an owner that reads live and not deferred; failed items back off exponentially; an item due for more than 30 minutes raises one escalation. The drain calls `open_debt` with the wall clock, not the observation instant, so a retried re-entry cannot report its regression twice. The stale-head rule of item 23 moves from the lane into the drain, for every call site: deferring `open_debt` lets a de-flake owner go done between the observation and the cycle's re-opening, so that is where it must be decided; `open_debt` gains that one rule and is otherwise unchanged. The lane's `debt_due` (§5.3) is retired: ε2 calls the recorder before it records the episode's end. Scenarios 29 and 34 are rewritten and 41 is new. ε0 adds no volume bound: a burst of N cleared tests files N de-flake tasks, as the cap already did over several observations; a policy bound, if wanted, is task 3792's.

Not changed, and for the owner to know:

- **The three author's calls in D6 are still unreviewed.** ε1 rests on call 1; ε2 rests on all three and carries the waiver. ζ's runbook puts them in front of Leo with the soak figures.
- **Between β and δ2 a detection run that names no test files nothing** (re-scope 3). Two of the nine detection stalls on record were followed by a confirm that also stalled; in that window those are logged, not filed.
- **γ makes auto-eval live.** `auto_eval_enabled` defaults on, and today every redo it files is orphaned by the failed id parse. After γ a redo dispatches, while task 5423's rename-rollback and wrong-trigger defects are still open. γ's hand-carry carries a decision point.
- **Two edge directions are close calls.** 4976 (a fix for a live silent loss) and 5089 (high priority, with 5529 behind it) now wait on ε1. Hand-carrying either before ε1 starts and removing its edge is the alternative; ε1's text tolerates both orders.
- **Three gaps in ε2 that §5.8 had, closed by ε0 (item 24).** A restart can no longer lose a pass's occurrence rows (the recorder call comes before the episode store); a recovered test whose owner filing fails stays on the due list and is retried; and no cap leaves a recovered test without a debt row. One narrow loss remains, visible in the pass record: a test not offered because its covering task's status or the stale-head ancestry read was unreadable.
- **D10's stale-head rule had no ledger counterpart.** Resolved on 2026-10-13 (item 23).
- **A reader nobody owned.** `scripts/suite_census_evidence.py::FLAKE_QUERY` reads every `flake_occurrence` row with no call-site filter, so lane rows will enter the suite census once ε2 is live. Follow-up 3 makes the script say what they mean to it.
- **Three unblessed metadata keys.** `flake_debt_call_site`, `flake_debt_trigger` and `flake_debt_command` (§5.6) are machine-read by ζ0 and task 3792 and are not in the blessed set, like the ledger's three existing `flake_debt_*` keys; each filing mints census lines. Accepted as §5.6 names them.
- **δ2 (25 files) and δ3 (26 files, 16 prose-only) are over the 15-file review trigger.** Both are hand-carries and each has one clean cut, named in its task.
- **The deferral record on follow-up 2 is convention-only.** Its enforcement (task 6525) has not landed; the record will need re-stamping when it does.

Delivered checks: 33 mechanical checks (`grep` and `path`) across α–ζ0, each linted against main with `shared.delivered_check_polarity.lint_delivered_checks` before filing: zero findings. ζ's bindings are `manual`.
