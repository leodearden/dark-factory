# D10: fold-verifier for `plans/review-all-program-tooling-prd.md` (D8 critic → D9 fold)

Fold-verifier seat, 2026-10-19. Nothing in any repo was changed and nothing was committed. Scratch material is under `<scratchpad>/d10/`: a throwaway git repo for the ERE test and `measure.py`.

**Sources measured**
- Code: main `93f478fa3b`, which is the main checkout's HEAD.
- Polarity: the worktree HEAD `3ed84d5c5a`, plus its uncommitted working tree for the skill text.
- Tasks: 6528, 6922, 5916, 6959 and 6105 were read with `get_task`.

## 1. Per-item disposition

| Item | Verdict | Evidence in the folded text |
|---|---|---|
| B1 | **fixed** | H3 l.136: "the reader requires the response to echo `include_rows: true` and every row to carry both `metadata` and `dependencies`, and raises `StoreReaderUnsupported` … also accepts the explicit rejection task 4867 will later add". The fake behaves as FastMCP does (l.170). B11 now reads "accepts the argument, ignores it, no echo, 5-field rows". §2(iv), the §6 row and the 4867 §8 row are all present. I re-verified `ArgModelBase`'s behaviour and confirmed that `check_method_param_wiring.py` passes for `include_value` (rc 0) and fails for `include_rows` (rc 1) on both `tools.py` and `task_interceptor.py`, so ρ's two script checks are well-formed. |
| B2 | **fixed** | H5 l.163: "It does **not** inline the file: it composes a short pointer prompt … `record.prompt` holds the pointer. A relative path is refused". σ1 has the 64 KB tmux-branch test, B9 asserts an absolute path, and σ3 uses a >16 KB brief. `parse_spawn_identity` ignores `prompt`, so the pointer does not change role/slug resolution. |
| M1 | **fixed** | §6 l.222 states the call-node rule. ρ's module list no longer carries the ledger files. The backend uses aiosqlite (`async with self._fresh_read_conn … conn.execute`), so "no blocking call" holds. |
| M2 | **fixed** (minor residual N16) | No multi-path `grep` remains, and `path` checks are conjunctive (`evaluate_path_at_tree`, "CONJUNCTIVE and short-circuiting"). I measured that `git grep -E` treats `\|` as a literal pipe: the trailer pattern matches only the full ordered line, and the pair pattern matches the SKILL.md backticked form `` (`done|…|handed-off|in-progress`) ``. ρ uses two `check_method_param_wiring.py --forwards-to` checks. |
| M3 | **fixed** (two minor residuals, N6/N7) | H5 l.165: "`run_stop` passes a Stop-only `self_exit=` decision into `_run_status_refresh_and_retitle`, which evaluates it **before** the one `write_record`". The code agrees: `may_bind` and `in_launch_window` are locals computed before `session_registry.write_record(record, root=root)` (the one write), and `_stamp_session_pointer` follows it. The action location is underspecified, because the shared body returns only the OSC string. Either the kill runs inside the body after the pointer stamp (a Stop-only callable), or the return type widens. |
| M4 | **fixed** (minor residual N6) | "every evaluation with a marker present stamps `exit_request: {state: refused\|signalled, reason, pid, at}` … with no marker the hook stays silent". |
| M5 | **fixed** (for Leo: N11) | H5 l.164 makes the flag opt-in. `_is_shared_session_host` and `session_crons` are among the preconditions. OQ7 is filed for Leo. |
| M6 | **fixed** (minor residuals N8, N9, N15) | H1 l.61 and H5 l.160 move the vocabulary into `session_registry.py`. The import rule now reads "No script imports a third-party package". σ1's modules cover all four live consumers: `spawn-claude.sh` l.338, `skills/spawn/SKILL.md` l.141, `skills/escalation-watcher/SKILL.md` l.2268 and `scripts/sitting/ownership.py` l.45–46. |
| M7 | **fixed** | `state.py` constants and `clerk_batch.py::CLERK_RULES` are the homes (H1 rows; decision 6). κ re-points the skill text and has checks `CLERK_RULES` and `SEAM_WAIT_KEYS`. |
| M8 | **fixed** | §1 l.16, H3 `completion_gate()`, B7 covers both forms, and 6389 is in §8. The role strings `completion_gate` / `human_gate` match `filing-the-trigger-chain.md` l.53/75. |
| M9 | **fixed** | The form/contradiction classes (l.121–122), `--check`, `--from-store`, `--amend` idempotence, and degeneracy measured on the inbox file are all present. |
| M10 | **fixed in the PRD; contradicted by the skill text committed with it** (N3) | l.94: "and nowhere else — a spawned session's `result.md` names that path … carries no copy". |
| M11 | **fixed** | `sessions.json` is `{title: {stream, pass, spawned_after, record_dir\|null}}`. η is the only writer of `record_dir` (l.80; contract 5). |
| M12 | **partial** (N5) | `tasks_with_delivered_checks` and `completion_gate` were added, along with the fixture inventory and decision 17. α's enumerated surface still omits state.py symbols that H1 places there. |
| M13 | **fixed** | B12–B15 were added. Every script runs under `sys.executable -I -S` from a temp cwd. ω is executable with a shebang. |
| M14 | **fixed** | The §8 rows are present. Edges σ2→6922, σ1→6528 and κ→6959 were added. No cycle (see §2g). |
| M15 | **partial** | Renders go into the home and `--publish` runs at the wave commit (l.66, decision 15). The remaining window is stated but bounded only as "the wave". The justification "they cannot live elsewhere" is true only for the post-decompose stamped sidecar. Author-pass PRDs and manifests (author → critic → fold, hours) could live in the home and be published like the doc. |
| M16 | **fixed** | The rulings, S3, D6 and D8 exist under `plans/review-all-feedback-synthesis-2026-10-17/` in the worktree (untracked) and are cited by those paths. |
| Minors 1–18 | folded as D9 states | Exceptions: minor 9's fold introduced N1, and minor 3's fold introduced N4. |

## 2. NEW defects (introduced or exposed by the fold)

### Majors

**N1. `seam_scan.py` now parses YAML under a stdlib-only, `-I -S` rule.**
- **Claim (fold of minor 9):** ζ reads "committed sidecars' `delivered_check.paths` under `plans_dir`" (§1 l.19, §7 l.255, ζ row l.298).
- **Actually:** sidecars are `.capability-manifest.yaml`, and the stdlib has no YAML reader. Decision 7 already concedes this for writing: "no script here needs a YAML writer (stdlib has none) or a prose parser (heuristic 12)". So ζ either imports PyYAML, which fails B14 and the import rule, or hand-parses YAML, which violates heuristic 12.
- **Fold:** drop the sidecar read. Filed tasks already carry their checks in `metadata.delivered_checks`, which `StoreReader.tasks_with_delivered_checks()` returns. Then say plainly that an authored-but-unfiled PRD is out of `seam_scan`'s reach.

**N2. The brief that is actually spawned (author) is not asserted to carry the return contract; B9 spawns a decompose pass that the skill never spawns.**
- **Claim:** §1 says "A seat or session writes one JSON object to `inbox/<S>.<pass>.json` and runs `apply_return.py --check`".
- **Actually:**
  - B9 (l.182) lists the schema, the example, the inbox path, the `--check` instruction and the spawn-only paragraph for the **decompose** brief only.
  - The author brief row lists the stream row, rulings, statements and filings.
  - `deliberation.md` l.61 has the spawned session run the author pass only: "its critic and fold … and its decompose then run as `agent` seats".
  - §1 l.14 gives `gen_brief.py <S> decompose` with no `--spawn`, yet B9 runs `decompose --spawn` and asserts `CLAUDE_SPAWN_SELF_EXIT=1` on it.
  - So ω would certify a spawned author brief that cannot produce a valid return or self-exit.
- **Fold:**
  - Both briefs carry the schema and example, the absolute inbox path and the `--check` line.
  - Only a `--spawn` brief carries the spawn paragraph.
  - B9 spawns the author pass. Decompose is an agent seat, or `--spawn` is stated as an allowed option for it.

**N3. The skill text committed in the same wave contradicts the folded return contract, and κ's absent check depends on committing that contradiction.**
- **Claim:** "one channel" (H2, decision 16). The md twin says κ's three absent checks "FAIL once the wave commits".
- **Actually:** the worktree's `deliberation.md` l.54 says "written to … `inbox/<S>.<pass>.json`, and, for a spawned session, also fenced at the end of `result.md`". The same line calls `filed`, `held` and `released` "top-level" for both passes. Under the PRD's `validate_return`, an author return carrying `filed: []` is an "extra top-level key" form rejection unless off-pass keys are tolerated, and the PRD does not say which.
  - κ lands last: after α–η, ω and the low-priority 6959.
  - The committed skill text therefore instructs two channels for the whole build.
  - `second-channel-removed` is non-vacuous only because this contradiction gets committed.
- **Evidence:** `grep -cE 'fenced at the end of'` returns 1 on the worktree file and 0 on `main`.
- **Fold:**
  - Fix the D1 sentence now: inbox only, `result.md` names the path. Delete κ's `second-channel-removed` check, since it becomes vacuous.
  - Make the D1 field list say which keys are decompose-only, or cite §Return block without a list.
  - State `validate_return`'s rule for off-pass keys.

**N4. The session-id guard on the marker rests on an unmeasured premise that compaction can break.**
- **Claim (fold of minor 3):** the marker holds `$CLAUDE_CODE_SESSION_ID`, and the hook acts only when it equals the Stop input's `session_id`.
- **Actually:** the codebase records that "Claude Code RE-MINTS `session_id` in place on `/clear` and on automatic compaction" (`session_registry.py::SessionRecord` docstring; `session_hooks.py` l.184).
  - Nothing measures whether the Bash tool's `CLAUDE_CODE_SESSION_ID` tracks the re-mint.
  - If it holds the launch-time id, every long (compacted) review-all session is refused forever. Those are exactly the ~5 h idlers the feature targets.
  - σ3's probe is a short session with no compaction, so it cannot see this. The failure is stamped (`exit_request.state = refused`) but the feature would silently not deliver.
- **Fold:**
  - List the premise as unmeasured in §6.
  - Add a `/compact` step to σ3's probe before the marker is written.
  - Alternatively, accept a marker equal to the hook's `session_id` **or** the record's bound `claude_session_id`, keeping the resume-safety intent.

**N5. α's enumerated surface still omits state.py symbols that H1 assigns, and decision 17 turns each omission into a forced chain or a parallel edit.**
- **Actually:**
  - H1 puts `stream_of`, `waves`, `LEDGER_OPS` and `SEAM_RELEASED_KEYS` in `state.py`, and so does H1 l.88's "Ids are minted by `record.py` and `apply_return.py` in one `D<n>/L<n>` namespace".
  - α's row (l.292) lists only the constants, the validator, home/paths, append and the reader.
  - Decision 17 makes `state.py` read-only for β–η. Then γ (render: `waves`, `stream_of`), δ (`waves`) and β (a shared id minter for two scripts) must either chain after an α amendment or add the code to `state.py` in parallel, which is the M12 hazard again.
  - `append` is specified as a seq check with no lock, and two appliers exist (the lead's `apply_return.py` and η). A mid-apply refusal can leave a partial ledger.
- **Fold:**
  - α's row names `stream_of`, `waves`, `next_ruling_id`, `LEDGER_OPS` and `SEAM_RELEASED_KEYS`, plus `REVIEW_ALL_HOME` / `FUSED_MEMORY_URL` resolution (twin contracts 2–3).
  - `append` takes an `fcntl` lock, and `apply_return` is all-or-nothing.

**N17. κ removes the only instruction for the main handoff, which `--handoff` cannot render.** This is exposed rather than introduced: the fold added handoff duties but not the timing.
- **Actually:**
  - `SKILL.md` §Lead handoff (worktree l.154) has the phases 0–5 lead write `HANDOFF.md` "At the Phase 5 commit". That is before deliberation, so before `record.py init` writes `program.json`.
  - γ's `--handoff` renders "from the state, the run record and the store", and B8's precondition is "B4".
  - κ deletes "until then the lead writes that content by hand". After κ, the prescribed Phase 5→6 handoff has no renderer input.
- **Fold:** `--handoff` runs with no `program.json` (record + store only, saying "no program yet"). Add a B8 variant for it.

### Minors

- **N6.** "Every evaluation with a marker present stamps `exit_request`" includes withheld launch-window events and `may_bind`-false ones. The shared body's contract is that NOTHING such an event carries lands on the record ("not the binding, not the status, not the pending question"). Withheld or unproven events should stamp nothing.
- **N7.** `exit_request.state = signalled` and `exit_requested_at` are written before `os.kill`, and the marker is deleted afterwards. If `os.kill` raises, or SIGINT is swallowed, the record says "signalled" and nothing retries. Stamp the attempt, not the outcome, or re-stamp on a kill error.
- **N8.** The fold invokes heuristic 14 to keep σ2 out of `session_hooks.py` (1,960 lines). It then moves the vocabulary into `session_registry.py` and adds `self_exit`, `exit_request` and `exit_requested_at` there. That file is **5,879** lines. A small stdlib-only `session_result.py` beside it would hold the vocabulary and parser. The record fields must stay on `SessionRecord`: `to_dict`/`from_dict` enumerate fields, and unknown keys are dropped on every read-modify-write.
- **N9.** There is a fifth vocabulary prose site: the amended `deliberation.md` l.61 spells `done|blocked|abandoned|handed-off` and "`in-progress` is this program's convention until the program-tooling PRD adds it to the spawn skill".
  - κ's row does not name this edit.
  - The mirror test covers three sites.
  - OQ2's "revisit if a fourth prose site appears" has already triggered.
- **N10.** κ's test "every backticked `<name>.py` span in the three files resolves to a file under `skills/review-all/scripts/program/`" fails as written. The three files carry 13 path-qualified `.py` spans that resolve elsewhere or nowhere. Examples: `scripts/source_measures.py`; `shared/src/shared/finding_key.py` and `scripts/check_run_completion.py`, both absent on main. Scope the test to bare names.
- **N11 (for Leo, with OQ7).** Only self-exit is opt-in. The new trailer sentences — `/reflect` before ending, and `outcome: in-progress` checkpoints — go to every `/spawn` caller, including escalation-watcher's `/unblock` sessions. σ1 changes only the vocabulary line of watcher §Joining. The join step gains no rule for an `in-progress` outcome at join; it should be treated as "ended at checkpoint", meaning fall back to exploration.
- **N12.** `session_self_exit.py` runs from `stop.sh` with only `orchestrator/src` on `PYTHONPATH`. Its stdlib-only contract, and that of 6922's probe module, is unstated. It is also not added to `test_session_registry.py::TestStdlibOnlySelfContainment`'s entrypoint rows.
- **N13.** Edges onto low-priority tasks:
  - σ2 → 6922 (pending, **low**) and κ → 6959 (pending, **low**).
  - σ1 → 6528, which is in progress with deps 6524/6525, transitively gates δ, η and the signal leaf ω.
  - Decompose should raise 6922 and 6959 to the program's priority, or the PRD should say why it doesn't.
- **N14.** D9's report has wrong numbers, though the twins are right:
  - The sidecar loads **54** checks, not 55.
  - "`lint_delivered_checks` at `HEAD`: no finding" is false. At HEAD the lint rejects κ's three checks with `vacuous_absent`, which the md twin states correctly.
  - The lint also warns `absent_overbroad` on σ1 `vocabulary-copy-removed-from-ownership` when `files=None`. Decompose must set σ1's `metadata.files` to include `scripts/sitting/ownership.py`.
- **N15.** H5 says σ1 "moves … `TERMINAL_RESULT_OUTCOMES` … from `scripts/sitting/ownership.py`". No such symbol exists there (l.45–47 hold `RESULT_OUTCOMES`, `OWNING_OUTCOMES` and `TERMINAL_TASK_STATUSES`), so it is new.
- **N16.** ρ's echo, the server half of B1, has no mechanical delivered check. The script checks prove declaration and forwarding, and the greps prove tokens. Add a check on the echo key in `tools.py`, or accept that the fused-memory boundary rows are its only proof.

## 2a–h. The brief's specific checks

- **(a)** Placement holds. In `_run_status_refresh_and_retitle`, `in_launch_window` and `may_bind` are decided before the single `session_registry.write_record(record, root=root)`, and `_stamp_session_pointer` follows it.
  - The `exit_request` stamp fits the record only as new `SessionRecord` fields with `to_dict`/`from_dict` entries. σ2 lists `session_registry.py`, so that is fine.
  - It also fits `SCHEMA_MINOR`'s rule ("Bump this when a new backward-compatible (optional/defaulted) field is added", currently 3).
  - A residual risk: any long-lived process still on pre-σ1 code that read-modify-writes a record drops the new fields. That is fail-safe (no self-exit).
  - N6 and N7 apply here.
- **(b)** `_run_launching(env)` exists and builds the record from `CLAUDE_SPAWN_*` env, so it can read `CLAUDE_SPAWN_SELF_EXIT`.
  - The child env holds: `sanitize_env='for _v in ${!CLAUDE_SPAWN_@}; do unset "$_v"; done; …'` runs first in `$inner`, followed by exactly five exports (`SESSION_ID`, `PARENT_ID`, `RESULT_FILE`, `WM_TITLE`, `OWNER_PPID`).
  - The trailer is built in the parent before the sanitiser, so it can gate on the flag.
- **(c)** Every live consumer is in σ1's module list. The fifth (D1) site is uncovered (N9).
  - σ1 → 6528 is sound: 6528 is in progress and its description edits `ownership.py`'s `COALESCE_KEY`. It does lengthen the signal path (N13).
  - Behaviour change checked: `parse_result_outcome` returns `in-progress` instead of None. `_judge_session` on an exited record still answers `mentions`, because the value is not in `OWNING_OUTCOMES`, so the answer is unchanged.
- **(d)** Same user and same `$HOME`, so the path is reachable. However:
  - The author brief is not asserted to carry it (N2).
  - It must be absolute, because a tmux-lane child inherits the tmux server's env, not the lead's `REVIEW_ALL_HOME`.
- **(e)** The window is stated but not bounded (M15 partial).
- **(f)** α at M–L (800–1,100 LOC) is inside the overlay band (300–1,500), so it is not too large. β–η are read-only on `state.py` only if N5 is folded.
- **(g)** No cycle. α ∥ ρ ∥ σ1←6528; β, γ, ε, ζ ← α; δ ← α, σ1; η ← α, β, σ1; ω ← α–η, ρ; κ ← α–η, ω, 6959; σ2 ← σ1, 6922; σ3 ← σ1, σ2.
  - No sibling-green check found: every delivered check greps or lists a file in its own leaf's module list.
  - ω is integration-green by design, behind its deps.
  - σ2/σ3 sit off the signal path.
  - 6105's partition candidates (B3 gate, red-on-main, shadow rulings) do not include §Joining, so σ1's watcher check is not at risk from it.
- **(h)** Confirmed. Each κ phrase gives `grep -cE` = 1 in the worktree working tree and 0 on `main`:
  - `by hand until the program-tooling PRD lands` (`deliberation.md`)
  - `fenced at the end of` (`deliberation.md`)
  - `until then the lead writes that content by hand` (`SKILL.md`)

  All three flip to FAIL once the four amended files reach `main`. The four κ `present` patterns are 0 in the working tree, so the wave commit does not make them vacuous.

## 3. Manifest measurement

- **Load:** `load_capability_manifest` via `uv run --directory /home/leo/src/dark-factory`. The worktree sidecar has 13 blocks, `signal_leaves: ['ω']`, and **54** delivered checks: 49 grep/path, 3 script, 2 manual.
- **Evaluate:** I ran `evaluate_grep_at_tree` / `evaluate_path_at_tree` on all 49 grep/path checks, both at the worktree `HEAD` (`3ed84d5c5a`) and at `main` (`93f478fa3b`). In both runs, every one FAILs except κ's three `expect: absent` checks, which PASS (vacuous). **No other PASS-today check.**
- **Scripts:** ρ's two script checks exit 1 today ("function does not declare the parameter"). ω's script is absent.
- **Lint:** `lint_delivered_checks(files=None)` at both refs gives:
  - `reject vacuous_absent` ×3 for κ (expected until the wave commits);
  - `warn absent_overbroad` for σ1 `vocabulary-copy-removed-from-ownership`. This is an artefact of `files=None` and clears once σ1's `metadata.files` declares `ownership.py`.
- **Scripts:** `<scratchpad>/d10/measure.py`; output in `<scratchpad>/d10/at-worktree-head.txt`.

## 4. Consistency with the amended skill text

- **Return block** (`deliberation.md` l.54 vs PRD §Return block): the ops list matches (`cancel|edge|note|stamp|repath|release|amend`). Two disagreements, both N3:
  - the channel ("also fenced at the end of `result.md`" vs "nowhere else");
  - pass scoping (D1 makes `filed`/`held`/`released` top-level for both passes).

  The D1 list omits `touches`, `seam_dependencies`, `commit_planning`, `other_filed`, `notes`, `commits` and `session`, but it defers field shapes to the PRD, which is acceptable.
- **State home** (`SKILL.md` §Lead handoff vs PRD layout): `~/.claude/review-all/<run_id>/HANDOFF.md` matches, and so does the `.state/` snapshot `program.json`, `rulings.jsonl`, `ledger.jsonl`. The timing does not match (N17).
- **`x_seam_wait` / `x_seam_released`:** `report-format.md` l.99 has `{stream, contract, release_owner}` and `{by, at, onto: [<ids>], contract_as_delivered}`, which equal PRD H1 l.53. The PRD's return block names the same field `held[].waits_on` where the store key says `stream`. That is two spellings of one fact; minor, and it should be stated as a mapping in `validate_return`.

## Verdict: **second-fold-needed**

No blockers. Both D8 blockers are fixed.

Items for the second fold:
- **N1:** drop the sidecar YAML read.
- **N2:** author-brief contract; fix B9's spawned pass.
- **N3:** fix the D1 `deliberation.md` sentence and field scoping now; drop κ's `second-channel-removed`.
- **N4:** session-id guard vs re-mint; extend σ3's probe.
- **N5:** complete α's surface; lock `append`.
- **N17:** `--handoff` without `program.json`.
- **M12, M15:** finish the partials.

The minors N6–N16 are one-line text edits. A lead spot-check is enough after the second fold, except N4, which changes σ2/σ3 and should be re-checked against the code.
