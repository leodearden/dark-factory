# D8: adversarial critic of `plans/review-all-program-tooling-prd.md` (+ sidecar, md twin)

Critic seat, 2026-10-19. Read-only: no file in any repo changed, nothing committed. All experiments ran under `<scratchpad>/d8/`: a private tmux socket that was killed afterwards, a throwaway git repo for the regex test, and an in-process FastMCP probe.

Measured against main `fbc5840274`. Main has moved past the author's `3ed84d5c5a`, but nothing the PRD cites has changed between the two. The worktree's working-tree skill files (D1/D2 amendments) were read as they stand.

**What holds, verified:**
- The sidecar loads through `shared.capability_manifest.load_capability_manifest`: 13 blocks, `signal_leaves: [ω]`.
- 22 of the 24 grep/path checks FAIL at `main`.
  - Each multi-path check also fails on every path taken alone.
  - Both of κ's `absent` checks PASS at `main`, so they are vacuous today, as the author says. `lint_delivered_checks` rejects them with `vacuous_absent`.
- 6385's live metadata has `prd_task_label: "L1"` and `dependencies: []`, as ρ's signal assumes.
- `/usr/bin/python3` here has no `httpx`, so the stdlib premise behind decision 3 is real.
- Ordering in the `release` recipe is right: `commit_planning` lints before the hold refusal.
- The `tests/scripts/` collection claims hold, and so does `test_skills_has_no_tests_of_its_own`.

## BLOCKERS

### B1. The "old server rejects `include_rows`" premise is false; INV-11's refusal never fires against a real server, and ω certifies a fake

- **Claim:** H3, α's `reader-refuses-old-server`, B11 and manifest contract 1 say a server that predates ρ rejects the argument. α's reader then raises `StoreReaderUnsupported`. `fake_store.py --no-include-rows` "rejects the argument with an `isError` envelope".
- **Actually:** fused-memory builds its tools on `mcp.server.fastmcp` 1.27.0. Arguments are validated by `ArgModelBase`, which keeps pydantic's default `extra='ignore'`. An undeclared `include_rows` is dropped silently, and the call succeeds with the 5-field rows. The reader then renders a program with no `metadata` and no `dependencies`, which is exactly the silent fallback INV-11 forbids. B11 passes only because the fake rejects; no real server does.
- **Evidence:**
  - Measured in-process. With `@mcp.tool() async def find(key, include_value=False)`, the call `call_tool('find', {'key':'k','include_rows':True})` returned normally, with no error.
  - The live `find_tasks_by_metadata` schema has no `additionalProperties: false`.
  - Task **4867** (pending, high) documents this class: "FastMCP's pydantic model dropped it SILENTLY (extra='ignore')".
  - The handler echoes `key`, `value` and `statuses`, but not the opt-ins.
- **Fold:**
  - ρ echoes `include_rows: true` in the response envelope and always puts `metadata` and `dependencies` on each row.
  - α raises `StoreReaderUnsupported` when the echo is absent or any row lacks either key. It should accept both server behaviours, since 4867 will later turn the drop into a rejection.
  - `fake_store.py --no-include-rows` mimics `extra='ignore'`: it accepts the argument, ignores it, sends no echo and returns 5-field rows. B11 asserts against that.
  - Add 4867 to §8.
- **Touches:** α (`state.py`, `fixture/fake_store.py`), ρ (`server/tools.py`, tests), ω (B11), manifest contract 1, and the α `reader-refuses-old-server` binding text.

### B2. `--prompt-file` as specified inlines the brief into the launcher's command line; tmux refuses it above ~16 KB, and the briefs δ generates are bigger

- **Claim:** H5 and σ1 say the file's text is the prompt "byte-for-byte", with the trailer appended "as today". δ prints `--prompt-file briefs/S2.decompose.md`.
- **Actually:** "As today" means `q_prompt=$(printf %q "$prompt")` baked into `inner`, then `tmux new-window … bash -c "$inner"`, or `<emulator> -- bash -c "$inner"`.
  - tmux 3.4 refuses a command over 16,240 bytes with `command too long`. That size was measured on a private socket here.
  - The emulator lanes cap any single argv string at 128 KiB (MAX_ARG_STRLEN).
  - The payload plus trailer is about 2 KB, so the usable prompt on the tmux lane is about 14 KB.
  - 5 of the 16 review-all briefs in `~/.claude/spawn-briefs/` exceed that: 14.7, 14.9, 17.6, 48.8 and 75.6 KB.
  - A `gen_brief.py` brief carries more than any of those: full statements, filings rows verbatim, rulings, xrefs, and the schema with an example.
- **Evidence:** `skills/spawn/SKILL.md` line 30 already prescribes the opposite: write the plan to a file and "spawn a short prompt pointing at it. Passing a multi-paragraph plan as one shell-quoted CLI arg is fragile." σ3's probe uses a trivial prompt on the tmux lane, and B9's fixture brief is small, so no planned check can see this failure.
- **Fold:**
  - `--prompt-file <abs path>` makes `spawn-claude.sh` compose the short pointer prompt itself (`Read <abs path> and follow it…`), with the trailer appended. `record.prompt` holds that pointer.
  - Drop "byte-for-byte".
  - σ1 test: a 64 KB file through the tmux branch of the fake harness, asserting that the command length does not depend on the file size.
  - B9 asserts an absolute path. The payload `cd`s to `<cwd>`, so a relative `briefs/…` path breaks.
- **Touches:** σ1 (`skills/spawn/spawn-claude.sh`, `skills/spawn/SKILL.md`, `tests/scripts/test_spawn_claude.py`), δ (`gen_brief.py`), σ3 (probe with a brief over 16 KB), H5 text.

## MAJORS

### M1. The ratchet re-hash requirement is refuted (ρ, §6 row 4, contract 4, manifest `loop-blocking-ratchet-row`)

- **Claim:** widening the handler changes its content hash, so ρ re-hashes the `AUDITED_SITES` row and the ledger entry.
- **Actually:** `content_hash = sha256(ast.unparse(call_node))[:12]` hashes the *call* `_normalize_project_root(project_root)`, not the handler body.
- **Evidence:** I computed it: the hash is `81d0d2709341`, the value all 23 handler rows share. Widening the body cannot move it. 6385's three dry_run_proposals were about a *new* call site, first the missing 23rd row and then the missing `LOOP_BLOCKING_LEDGER` entry, not about a hash change.
- **Fold:**
  - Drop the re-hash requirement, and drop `shared/tests/loop_blocking_allowlist.py` and `test_shared_governed_lists.py` from ρ's modules. Including them only adds locks those files don't need; task 5270 already declares them.
  - State the actual rule: a new row is needed only if ρ adds a new blocking call inside a coroutine.

### M2. The grep checks are hollow or fragile (manifest polarity)

- **Multi-path checks pass on any one path.** `git grep … -- p1 p2 p3` exits 0 if any path matches.
  - ρ's `include_rows` over four files passes when `tools.py` alone declares it. That is the "accepted then dropped" hollow-DELIVERED shape that task 3364 built `scripts/check_method_param_wiring.py` to catch. Task 6581 already uses that script for `get_tasks(fields)`.
  - σ1's two- and three-path checks and σ2's two-path `exit_requested_at` check have the same weakness.
  - The md twin's "in the four fused-memory files" misreads the check.
- **The pipe pattern pins an unstated order.** Under ERE, `handed-off\|in-progress` matches only the literal `handed-off|in-progress`. Measured: `…|in-progress|handed-off` and a table-escaped `\|` both miss. The PRD never pins that order.
- **Fold:**
  - ρ: replace the grep with `kind: script` → `check_method_param_wiring.py --file …/server/tools.py --function find_tasks_by_metadata --param include_rows --forwards-to find_tasks_by_metadata`. Add a second one for the interceptor.
  - Make one check per path wherever every path is required.
  - Pin the vocabulary line verbatim (`outcome: done|blocked|abandoned|handed-off|in-progress`), or drop the prose check in favour of σ1's mirror test.

### M3. σ2's placement contradicts itself and forks the one shared body

- **Claim (H5):** the self-termination block sits "after the single `write_record` and `_stamp_session_pointer`", yet `exit_requested_at` is stamped "inside the same single write".
- **Actually:** both cannot be true. The preconditions also read `may_bind` and the launch-window withhold decision. Those are locals of `_run_status_refresh_and_retitle`, which `run_notification` shares, and whose docstring says "ONE BLOCK, ALL LANES … Do NOT re-fork that lane".
- **Evidence:**
  - `session_hooks.py` is 1,960 lines, already over heuristic 14's ~1,500.
  - Task **6922** (pending) extracts `_owning_claude_pid`, `ProcessTree` and `_EventProbes`, the very symbols σ2 binds by name.
- **Fold:**
  - Decide before the write through a Stop-only parameter of the shared body, and stamp onto `record` before the one `write_record`. Signal after `_stamp_session_pointer`.
  - Put the decision and the kill in their own module, for example `orchestrator/session_self_exit.py`. Bind it to wherever 6922 leaves the probes.

### M4. "Logged" refusals have no sink

- **Actually:** `session_hooks.py` installs no logging handler; it only has `logger = getLogger(__name__)`. Hook stderr on an exit-0 Stop is not shown to anyone.
  - σ3's "the hook log names the signalled pid" has nowhere to be read from.
  - The G7 `no-silent-fail-soft` claim ("absent `background_tasks` is logged") has nothing behind it.
- **Fold:**
  - Stamp `exit_request: {state: refused|signalled, reason, pid, at}` on the record inside the same write. That is durable and η can read it.
  - Stay silent when there is no marker. The Stop hook runs on every turn of every session on the host.

### M5. Self-exit becomes the fleet-wide default before it is measured (needs Leo)

- **The trailer is shared.** Every `/spawn` caller gets the same trailer, including escalation-watcher's `/unblock` sessions that Leo drives.
- **σ2 is live the moment it merges.** `stop.sh` runs `/home/leo/src/dark-factory/orchestrator/src/…` directly, with no restart. Once σ1 and σ2 are both on main, every attended spawned session closes itself before σ3 has measured SIGINT.
- **Leo's ruling was narrower.** A8(ii) asked for a feasibility report. D6's recommendation 13 proposed a review-all trailer *variant*, not a new default.
- **Missing guards:**
  - Pid equality proves nothing under a shared `claude agents` host. One is live on this host now (pid 2857696) and runs its sessions in-process. Task **6945** (pending) says `claude_owner_pid` collapses onto that host pid. `session_registry.py::_is_shared_session_host` already exists.
  - `session_crons` non-empty is not checked, so a `/loop` cron would die with the session.
- **Fold:**
  - Make it opt-in: `CLAUDE_SPAWN_SELF_EXIT=1`, set by `gen_brief.py`'s printed command and by σ3's probe. Default-on only by a later ruling.
  - Add the `not _is_shared_session_host(owner_pid)` and empty-`session_crons` preconditions.

### M6. η and σ2 add the 5th and 6th copies of the outcome vocabulary, and a second reader of the record shape

- **Claim:** §3 says "none imports a workspace package". The G7 walk says σ1 "edits all four" vocabulary sites.
- **Actually:**
  - η must restate `TERMINAL_STATUSES`, `RESULT_FILENAME`, the record JSON, the terminal outcome set and the `outcome:` header parser.
  - σ2's hook must also parse `result.md`'s outcome and cannot import `scripts/`.
  - The `session_registry` header says "consumers import this; they never re-derive the record shape".
- **Evidence:** under a bare `env -i /usr/bin/python3` with `sys.path` inserts, both of these import cleanly:
  - `from orchestrator import session_registry`
  - `from sitting.ownership import parse_result_outcome, RESULT_OUTCOMES`

  That is `stop.sh`'s own pattern.
- **Fold:**
  - Restate the rule as "no third-party import".
  - σ1 moves `RESULT_OUTCOMES`, the terminal and owning sets, and `parse_result_outcome` into `session_registry.py`, beside `RESULT_FILENAME` (the T5 contract's home).
  - `ownership.py`, `session_hooks.py` and `watch_sessions.py` import them from there.

### M7. Contract homes multiply (heuristic 11)

- **Return block.** The PRD's §Return block, `state.py` (declared "the one home"), `deliberation.md:54`'s partial field and op list, and the brief embed. A PRD section is not a durable home for a live schema.
- **Held-leaf shapes.** Decision 6 says they are "stated once in report-format.md … imported by the validator as constants". There is no import path from markdown, so `state.py` either restates them or parses prose (heuristic 12). H1 restates them as well.
- **Release recipe.** Four copies: `report-format.md`'s held-leaf bullet, `deliberation.md` step 3, H4, and the clerk prompt.
- **Fold:**
  - `state.py` constants become the machine home, for example `SEAM_WAIT_KEYS`.
  - κ re-points `deliberation.md:54` and `report-format.md` at `state.py::RETURN_SCHEMA`, `validate_return` and the key constants, and deletes the field lists.
  - The clerk prompt constant in `clerk_batch.py` is the recipe's home, and the skill text cites it.

### M8. `gate_check.py` encodes the interim trigger-chain form only

- **Actually:**
  - `filing-the-trigger-chain.md` §3, the final form: `human_gate` depends on `completion_gate` only, and that gate's predicate measures `x_finding_run` tasks.
  - Only §4, the interim form, wires high and critical leaves onto the gate.
  - The final form goes live when `scripts/check_run_completion.py` reaches main; it is not there yet, and tasks 6386–6393 are all pending.
  - Under the final form, gate_check exits 1 forever: every high leaf reads "missing" and the completion gate reads "foreign".
- **Evidence:** §6 cites "§3–§4 … human_gate depends on the program's high/critical tasks | wired". S3's third report has also been dropped: tasks with a program `prd_path` but no `x_finding_run`, which escape the final form's measure.
- **Fold:**
  - Detect the form: look for a `completion_gate` sibling with the same `run_id`.
  - Check the matching rule for each form, and restore the escapee report.
  - Add a final-form row to B7.
  - Name 6389 (`check_run_completion.py --pending`) in §8.

### M9. A rejected return has no repair path; for a decompose return that is a duplicate-filing hazard

- **Actually:**
  - Rejection writes nothing, the inbox file is immutable, and the only stated recovery is "relaunch".
  - A decompose seat has already filed its batch and run `commit_planning` on it. A form error (`notes` at 601 characters, an extra key, a mistyped label) leaves the store full of tasks and the program record empty.
  - Relaunching decompose re-files.
  - After σ2, the spawned session has already ended itself and cannot be asked to fix anything.
- **Fold:**
  - Split rejections into two classes:
    - Form errors: the seat or lead rewrites the inbox file and re-applies.
    - Contradictions with the store: stop for the lead.
  - The brief and trailer tell the seat or session to run `apply_return.py --check` on its own file before it ends or touches the marker.
  - Add `apply_return.py --from-store <S>`. Filed, held and dependency state already live in the store (H1), so the return can be reconstructed instead of relaunched.

### M10. A spawned session's return is applied twice

- **Actually:**
  - H2 has the session write `inbox/<S>.<pass>.json` and also fence the same object in `result.md`.
  - η applies the fence when it first sees a final result, while the lead applies the inbox file.
  - `returns/<S>.<pass>[.<n>]` accepts the second application as `.1`, so the xref, action and ruling entries are appended twice.
- **Fold:** pick one of these.
  - One channel: the session writes the inbox file only, and `result.md` names its path.
  - Or `apply_return` refuses identical content (by hash) for an `<S>.<pass>` already applied.

### M11. δ cannot write `sessions.json.record_dir`

- **Actually:**
  - Record slugs are `<role>-<project>-<pid>` (for example `prd-decompose-solar_challenge-1016051`). `spawn-claude.sh`'s `launching` call mints them after the lead runs the printed command; δ only prints that command.
  - The registry has no lookup by title, and there are about 91,700 record dirs.
  - η's title→record resolution is unspecified.
- **Fold:**
  - δ writes `{title, stream, pass, spawned_after}`.
  - η resolves the record once, by role prefix, `record.title` and an mtime after `spawned_after`, and records it.
  - Name the writer of that field. Today δ is "the only writer of `sessions.json`".

### M12. α's surfaces are incomplete for its six parallel consumers

- **Actually:**
  - `StoreReader` has no query for ε's impact scan: every non-terminal task carrying `delivered_checks`, which is a presence lookup with `include_rows`.
  - `fixture/store.json` must hold every B-row's preconditions:
    - B2: an out-of-program task declaring a touched path.
    - B4: a task whose check anchors a moved path.
    - B7: a gate with one missing high leaf and one foreign dep.
    - B10: the fixture record dirs.
  - Otherwise β, γ, ε, ζ and η will each edit `state.py` and `fixture/` in parallel with no edges.
  - The overlay says same-file leaves need real `add_dependency` edges.
- **Fold:**
  - α enumerates the full `StoreReader` API, adding `tasks_with_delivered_checks`, and a per-B-row fixture inventory.
  - β through η treat `state.py` and `fixture/` as read-only and copy-and-mutate in a temp dir. If that won't hold, chain them.

### M13. ω leaves parts of §1 unexercised and never tests the design's own premise

- **Missing from B1–B11:**
  - `record.py rule --supersedes` (§1 Recording; the newest ruling plus its superseded id in the render).
  - `clerk_batch.py ingest`.
  - `gen_brief.py … author`.
  - The limit-killed case: `outcome: in-progress` with the record `exited` must be reported as "ended at checkpoint, relaunch or resume", not left silent.
- **The stdlib claim is never asserted.** If the walk runs "in one process" under the repo venv, where `httpx` and `shared` import, then an accidental `import httpx` or `from shared…` passes. Decision 3's stdlib/any-cwd claim is never checked.
- **Fold:**
  - Add the missing B-rows.
  - Run every script as a subprocess with `sys.executable -S -I` from a temp cwd outside the repo.
  - Ship `check_review_all_program_fixture.py` executable with a shebang. `kind: script` runs `<root>/<script>` directly, and a non-executable script yields ERRORED, which leaves κ held indefinitely.

### M14. G4 seams the PRD's own seam scan would find are missing from §8

| Leaf | Missing seams |
|---|---|
| ρ | **4867**: the unknown-argument guard (B1). **6581**: `get_tasks(fields=…)` projection, pending; it already delivers OQ4's "fields projection" on the sibling tool. **4390**: an ids / prd-metadata filter on `get_tasks`. **7150**: shares `tools.py` and `docs/task-authoring.md`, with no edge. |
| σ2 | **6922**: extracts the probes σ2 binds. **6945**: owner pid on a shared host. **6080** and **6102**: `_process_comm`. |
| σ1 | **6528**: in progress, holding `ownership.py` now. **6611**: `ownership.py`'s terminal set. **6105**: partitions `escalation-watcher/SKILL.md`, so σ1's §Joining site may move. |
| κ | **6959**: shares `report-format.md`, with no edge. |
| ζ | **6389**. |

The author's G4 "pass" stands only for 7150.

### M15. Renders and stamped sidecars leave the machine-operated checkout dirty for a whole wave

- **Actually:**
  - `program_doc.py` writes the doc and `.state/` into `project_root/<plans_dir>` after every applied step.
  - `commit_planning`'s stamper and the clerk's `repath` and `release` edit sidecars and md twins there too.
  - The commit happens once per wave, which can take hours.
  - PROPOSALS A1's "PRDs are authored [in the home] and copied into the checkout at commit time" was dropped.
- **Why it matters:** with dark-factory as the target, one uncommitted file that overlaps any queued branch's diff halts every merge (`wip_conflict`; measured 1 h 48 m on 2026-09-21).
- **Fold:**
  - Render into the home.
  - `program_doc.py --publish` copies the doc and `.state/` into `plans_dir` immediately before the wave commit and prints the path list.
  - Say where seats write PRDs.
  - `release` and `repath` print the sidecar paths they dirtied.

### M16. The design inputs dangle

- **Actually:**
  - "Leo's rulings of 2026-10-19" and `D6-spawn-feasibility.md` exist only under `/tmp/claude-1000/…/scratchpad`.
  - `plans/review-all-feedback-synthesis-2026-10-17{.md,/}` (S3, S5) is untracked in the main checkout. D7 found the same.
  - σ2's rationale lives in D6 (§Q2 c/d, §Q4 items 4–6), and the open-question answers rest on the rulings. An implementer or decompose seat can read neither.
- **Fold:** commit the synthesis package, D6 and the rulings file with the wave, and cite them by repo path.

## MINORS

- `background_tasks` can now be measured statically. The 2.1.303 binary's Stop schema reads `background_tasks: T(cy()).optional()`, described as "Empty array when nothing is in flight". Upgrade the §6 row, and keep "absent ⇒ refuse".
- `exit_requested_at` is stamped and the marker deleted *before* SIGINT. If the signal does not end the session, a sibling-mode record reads "finished" with no retry left. η should require `exit_requested_at` *and* a dead owner pid.
- A refused attempt leaves `exit-requested` behind. The marker should carry `$CLAUDE_CODE_SESSION_ID`, and the hook should act only on a matching session id (resume safety).
- B5's "second render byte-identical" contradicts H3's "a render can say when the store was read" (`measured_at`). Keep `measured_at` out of the doc body.
- OQ5 says δ reads only `findings[]`, but δ's own signal needs the record's `filings[]` for "Phase 5 filings rows verbatim".
- γ: "`--handoff` writes the four sections §1 lists", but §1 lists none. SKILL.md §Lead handoff's duties (`launched_from`, `--yes`, fable seats) live in the record's method block, not in `program.json`, so γ must read `record_path`.
- "A return under 100 characters is relaunch" catches the designed one-line digest, for example `S10 decompose: 8 filed, 1 held, 0 questions, 2 actions → inbox/S10.decompose.json` at about 81 characters. Key degeneracy on the inbox file, not on message length.
- `stream_of` returns None for Phase 5 mechanical filings (they carry `x_finding_run` but no stream key). FILED needs an explicit "unstreamed" bucket (INV-13).
- `seam_scan.py` reads tasks' `metadata.files` only. `deliberation.md:47` also asks for the Files lists of non-terminal PRDs.
- ρ's module list omits `backends/metadata_lookup.py`, the "one place" `validate_metadata_query` checks `include_value` as a bool. Also, `include_value` is typed `Any` and validated in the body, not declared `bool`.
- Lone-row elision: StoreReader's handling of `truncated_row` is unspecified. On this store, metadata maxes at 46.2 KB and none exceeds 48 KB; reify is unmeasured. Re-read by `get_task` and say so, so that program membership is never dropped silently. One row can carry both elisions when `gate()` passes `include_value` and `include_rows`; with `include_rows`, `matched_value` is redundant.
- The md twin says the patterns are "symbol-anchored (task 7150's rule)". Seven are bare tokens, and the single-line rule is `deliberation.md` step 4(b), not 7150.
- κ's `absent` checks go non-vacuous only when the phrases reach `main` of `project_root` (`GATE_REF='main'`), not at a commit on branch `review-all-amend-20261019`. One vacuous check rejects all 13 leaves. Re-measure at decompose, after D7's fold of `deliberation.md:54`.
- Decision 3 calls this "the third copy of the MCP handshake". At least seven exist (migrate_metadata, census_trigger, mcp_lifecycle, cgl_eta_scheduler_gate, strip_leaked_control_keys, trial_module_tagger_haiku, dashboard data), all on httpx; this would be the first stdlib copy. OQ1's home should be a stdlib sibling in `shared/src/shared/mcp_post.py`, imported via `sys.path` like `orchestrator/src`, not private to a skill.
- κ's INV-10 mirror test is not in κ's module list. A `complexity: simple` leaf that adds a test file should name it.
- The `release` recipe should name the sidecar that `commit_planning` stamps in `project_root`, so the clerk returns it for the wave commit.
- `rulings_in_session[].id` is session-local; `apply_return` must re-mint it into the program's `D<n>` namespace.
- `released[]` claims (dependency added, `x_seam_released` present) are not in the validator's list.

## RULINGS-REFUTED (author decisions the code contradicts)

1. **H3 / B11 / contract 1:** "a server that rejects the argument (one predating ρ)". FastMCP drops it silently (B1).
2. **§6 row 4 / contract 4 / manifest `loop-blocking-ratchet-row`:** "widening the handler changes its content hash". The hash is of the call node (M1).
3. **Decision 6:** "held-leaf shapes … imported by the validator as constants" from `report-format.md`. No such import exists (M7).
4. **§3:** "none imports a workspace package" as a requirement of bare `python3`. Stdlib-only workspace modules import fine (M6). Also, the handshake is not the "third copy".
5. **§6 row 7:** "human_gate depends on the program's high/critical tasks — §3–§4, wired". That is §4 only (M8).
6. **H5:** `--prompt-file` "byte-for-byte". This reverses `skills/spawn/SKILL.md:30` and hits tmux's 16,240-byte limit (B2).
7. **H5:** "after the single write … inside the same single write" is self-contradictory (M3).
8. **G7 walk:** "no-lockstep-duplication: σ1 edits all four" is wrong (M6), and so is "no-silent-fail-soft: absent `background_tasks` is logged" (M4).
9. **md twin:** "grep present `include_rows` in the four fused-memory files". Any one file satisfies the check (M2).
10. **D5 G4:** "pass" (M14).
11. **D5 META:** "one reader of the store". ε's scan needs a reader method α does not define (M12).

## Verdict: **fold-before-commit**

Neither blocker needs a redesign:
- B1 needs a response echo plus a fake that tells the truth.
- B2 needs a pointer prompt in place of the inlined text.

The majors are concentrated in σ2 (M3, M4, M5), the reader and fixture contract (M12, M13), and SPOT (M6, M7). Most are one-paragraph folds.

Because σ2 goes live on every Stop of every session on the host at merge, run the fold-verifier (opus/high) after the fold. M5's opt-in versus default and M14's ρ seams (6581/4390 against Leo's A1(b) "widen find_tasks_by_metadata") are Leo's call, not the lead's.

### Leaves and files each blocker touches

| Blocker | Leaves | Files |
|---|---|---|
| B1 | α, ρ, ω (and manifest contract 1, α block binding) | `skills/review-all/scripts/program/state.py`, `…/fixture/fake_store.py`, `fused-memory/src/fused_memory/server/tools.py` (+ `fused-memory/tests/test_find_tasks_by_metadata_tool.py`), `scripts/check_review_all_program_fixture.py` (B11), PRD §4 H3 / §4 boundary table, md twin contract 1 |
| B2 | σ1, δ, σ3 (and ω's B9) | `skills/spawn/spawn-claude.sh`, `skills/spawn/SKILL.md`, `tests/scripts/test_spawn_claude.py`, `skills/review-all/scripts/program/gen_brief.py`, PRD §4 H5 and the σ3 row, B9 |
