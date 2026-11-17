# D9 — fold of the D8 critic into `review-all-program-tooling-prd` (2026-10-19)

Files rewritten in place (worktree only, uncommitted, nothing else touched):
- `/home/leo/src/dark-factory/.claude/worktrees/review-all-amend-20261019/plans/review-all-program-tooling-prd.md`
- `/home/leo/src/dark-factory/.claude/worktrees/review-all-amend-20261019/plans/review-all-program-tooling-prd.capability-manifest.yaml`
- `/home/leo/src/dark-factory/.claude/worktrees/review-all-amend-20261019/plans/review-all-program-tooling-prd.capability-manifest.md`

Every critic item was checked against the code before folding (main `93f478fa3b`; no cited file differs from the PRD's `3ed84d5c5a`). Verified first-hand: `ArgModelBase.model_config` is `{'arbitrary_types_allowed': True}` (no `extra` → pydantic ignore); `test_loop_blocking_gate.py` hashes `ast.unparse(call node)`; `check_method_param_wiring.py` takes `--file --function --param --forwards-to`; `session_hooks.py` is 1,960 lines; `_is_shared_session_host` exists and nothing reads `session_crons`; `orchestrator.session_registry` and `sitting.ownership` both import under `env -i /usr/bin/python3 -I -S` with `sys.path` inserts; `scripts/check_run_completion.py` is absent on main; the synthesis package (S3, D6, D8, rulings) is now under `plans/review-all-feedback-synthesis-2026-10-17/` in the worktree; tasks 4867, 6581, 4390, 6922, 6945, 6528 (in progress), 6611, 6105, 6389, 6080, 6102, 5270 read with `get_task`.

## Blockers

| Item | Disposition |
|---|---|
| B1 FastMCP drops `include_rows` silently | **folded** as directed: ρ echoes `include_rows` in the envelope and always puts `metadata` + `dependencies` on each row; α raises `StoreReaderUnsupported` when the echo or either key is missing, and accepts 4867's future rejection as the same condition; `fake_store.py --no-include-rows` ignores the argument like the real server; B11 asserts the missing echo; §6 row and 4867 seam added. |
| B2 `--prompt-file` inlines the brief past tmux's 16 KB | **folded** as directed: `spawn-claude.sh --prompt-file <abs>` composes a short pointer prompt (per `skills/spawn/SKILL.md`'s existing rule), `record.prompt` holds the pointer, "byte-for-byte" dropped; σ1 test with a 64 KB brief through the tmux branch; relative path refused; B9 asserts an absolute path; σ3 probes with a >16 KB brief. |

## Majors

| Item | Disposition |
|---|---|
| M1 ratchet re-hash | **folded** (critic right): re-hash claim dropped; `loop_blocking_allowlist.py` / `test_shared_governed_lists.py` removed from ρ's modules; §6 states the call-node rule; 5270 named as the ledger owner. |
| M2 hollow/fragile greps | **folded**: ρ proves the parameter with two `check_method_param_wiring.py --forwards-to` script checks (tool → interceptor, interceptor → backend) plus one grep per remaining path; every multi-path grep split into one check per path; the vocabulary line pinned verbatim as `outcome: done\|blocked\|abandoned\|handed-off\|in-progress` (literal pipes) with the order stated in H5. |
| M3 σ2 placement / shared body fork | **folded**: decision and kill in a new `orchestrator/src/orchestrator/session_self_exit.py`; `run_stop` passes a Stop-only decision into the shared body, which evaluates it before the one `write_record` and stamps in that write; signal after `_stamp_session_pointer`; σ2 sequenced after 6922 and imports the probes from its module. |
| M4 no log sink | **folded**: every evaluation with a marker stamps `exit_request: {state: refused\|signalled, reason, pid, at}` on the record in the same write; silent with no marker; σ3 and η read the record field. |
| M5 self-exit default | **folded as opt-in** per the lead: `CLAUDE_SPAWN_SELF_EXIT=1` is a per-launch input consumed by `spawn-claude.sh`; `_run_launching` stamps `record.self_exit` (the child's env stays the five documented vars); set by `gen_brief.py --spawn` and σ3 only; preconditions add `not _is_shared_session_host(owner_pid)` (6945) and empty `session_crons`; **for Leo**: the default after σ3 (PRD OQ7). |
| M6 vocabulary copies | **folded**: σ1 moves `RESULT_OUTCOMES`, `TERMINAL_RESULT_OUTCOMES`, `OWNING_OUTCOMES`, `parse_result_outcome` into `session_registry.py` beside `RESULT_FILENAME`; `ownership.py`, `session_self_exit.py`, `watch_sessions.py` import them; the rule is now "no third-party import" with the measured `-I -S` evidence; σ1 → 6528 edge (in progress on `ownership.py`). |
| M7 contract homes | **folded**: `state.py` constants (`RETURN_SCHEMA`, `validate_return`, `LEDGER_OPS`, `SEAM_WAIT_KEYS`, `SEAM_RELEASED_KEYS`) are the machine homes; `clerk_batch.py::CLERK_RULES` is the recipe home; decision 6 rewritten; κ re-points `deliberation.md` and `report-format.md` at the symbols and deletes the field lists (new κ checks `SEAM_WAIT_KEYS`, `CLERK_RULES`). |
| M8 gate_check interim-only | **folded**: `StoreReader.completion_gate()`; `gate_check.py` detects the form by the sibling, applies §3 or §4, restores the escapee report (program `prd_path` without `x_finding_run`); B7 gains the final-form row; 6389 in §8. |
| M9 no repair path | **folded**: rejections classed form (fix inbox, re-apply) vs store contradiction (stop for lead); `apply_return.py --check` run by every seat before ending (brief carries it); `--from-store <S>` reconstructs filed/held/released; idempotent on `<S>.<pass>`; degeneracy judged on the inbox file, not the digest line. |
| M10 double application | **folded** (one channel): the inbox file is the return; `result.md` names its path and carries no fence; η applies the inbox file once; `apply_return` refuses a second application; κ removes "also fenced at the end of `result.md`" (new absent check). |
| M11 `sessions.json.record_dir` | **folded**: δ writes `{title, stream, pass, spawned_after, record_dir: null}`; η resolves by role prefix, `record.title` and mtime after `spawned_after` and is the only writer of `record_dir`. |
| M12 α's surface | **folded**: full `StoreReader` API enumerated incl. `tasks_with_delivered_checks` and `completion_gate`; per-B-row fixture inventory in α; decision 17: β–η treat `state.py`/`fixture/` as read-only, copy-and-mutate, chain after an α amendment otherwise; α resized M–L. |
| M13 ω gaps and the stdlib premise | **folded**: B12 (supersede render), B13 (clerk ingest), B9 author pass, B10 "ended at checkpoint"; every script run as a subprocess under `sys.executable -I -S` from a temp cwd outside the repo (B14); script executable with a shebang. |
| M14 missing seams | **folded**: §8 rows for 4867, 6581, 4390, 7150 (file overlap stated), 5270, 3364, 6922, 6945, 6080/6102, 6528, 6611, 6105, 6389; edges added where a task reshapes bound symbols or is in progress on the file: **σ2 → 6922, σ1 → 6528, κ → 6959**; hub-file overlaps (`tools.py`: 7150, 4867, 6581, 4390) left to the scheduler's file locks with "re-read landed state", stated so the decompose seat can add edges if the lead rules otherwise. |
| M15 dirty checkout for a wave | **folded**: render into `~/.claude/review-all/<run_id>/render/`; `program_doc.py --publish` copies doc + `.state/` into `plans_dir` right before the wave commit; PRDs/sidecars still go to the checkout (the `commit_planning` stamper locates them under `project_root`) — stated as the one remaining window; `release`/`repath` return `dirtied` paths; decision 15. |
| M16 dangling inputs | **folded**: all design inputs cited by their `plans/review-all-feedback-synthesis-2026-10-17/…` paths (committed with this wave). |

## Minors (18)

1. `background_tasks` static schema → §6 row upgraded to "verified statically"; absent ⇒ refuse kept. **folded**
2. Stamp-before-signal / sibling-mode retry → η requires `exit_requested_at` **and** a dead owner pid; marker deleted after the signal. **folded**
3. Marker carries the session id → marker content = `$CLAUDE_CODE_SESSION_ID`; hook acts only on a match. **folded**
4. `measured_at` vs byte-identical renders → `measured_at` printed and ledgered, never in the doc body. **folded**
5. δ needs `filings[]` → OQ5 and δ read `findings[]` + `filings[]`. **folded**
6. `--handoff`'s four sections and `record_path` → listed in B8/γ; duties read from the record's `method.extra`. **folded**
7. 100-char digest line counted as degenerate → degeneracy keyed on the inbox file. **folded**
8. `stream_of` None → explicit `unstreamed` bucket in FILED (INV-13). **folded**
9. `seam_scan` ignores PRD Files lists → **partially folded**: it also reads committed sidecars' `delivered_check.paths`; prose Files lists of unfiled PRDs stay out (no prose parser, heuristic 12) and κ narrows `deliberation.md`'s sentence to say so. **declined in part, reason stated in §7**
10. `metadata_lookup.py` missing; `include_value` typed `Any` → ρ's modules add it; `include_rows: Any`, validated in the body like `include_value`. **folded**
11. `truncated_row` handling / double elision → re-read by `get_task`, counted and printed; `gate()` passes `include_rows` only, never `include_value` too. **folded**
12. "symbol-anchored (7150)" wording → corrected to "single-line (deliberation.md step 4(b)); anchored on a symbol where one exists". **folded**
13. κ `absent` checks vacuous until the phrases reach `main` → stated in both twins with `GATE_REF`; re-measure at decompose. **folded** (there are three such checks now, not four — the twins say three)
14. "third copy" of the handshake → "first stdlib copy beside at least seven httpx ones"; OQ1 names `shared/src/shared/mcp_post.py` as the home with a `sys.path` insert. **folded**
15. κ's test not in its module list → `tests/scripts/test_review_all_skill_text_cites_scripts.py` named, with a path check. **folded**
16. `release` names the stamped sidecar → recipe returns its path; `dirtied` in the clerk result. **folded**
17. `rulings_in_session[].id` re-minted into `D<n>` → stated in H1 (rulings) and the return block. **folded**
18. `released[]` claims validated → added to the store-contradiction class. **folded**

## Rulings-refuted (11)

1 echo-based detection (B1) · 2 call-node hash (M1) · 3 decision 6 rewritten: `state.py` constants are the home · 4 §3 rule restated as "no third-party import"; "first stdlib copy" · 5 §6 row names §3 and §4 separately · 6 pointer prompt (B2) · 7 decision-before-write, stamp in the one write (M3) · 8 G7 walk rewritten (one machine home + mirror test; `exit_request` stamp as the sink) · 9 md twin: one check per path (M2) · 10 G4 re-walked with the twelve tasks read (M14) · 11 `tasks_with_delivered_checks` added to the reader (M12). All **folded**.

## For Leo

- **OQ7 — self-exit default.** Authored opt-in (`CLAUDE_SPAWN_SELF_EXIT=1`, set only by `gen_brief.py --spawn` and σ3's probe). Leo rules whether escalation-watcher's `/unblock` spawns and other callers get it after σ3 measures.
- **ρ seams with 6581/4390** (the critic names these as Leo's call against A1(b)): folded as "no duplication" — ρ widens `find_tasks_by_metadata` per A1(b); 6581's `fields` projection on `get_tasks` is the sibling tool's answer and OQ4 points at it for any later projection need. No change to Leo's ruling is proposed.
- **Edges on hub files**: the PRD adds edges only where a task reshapes bound symbols or is in progress on the file (σ2→6922, σ1→6528, κ→6959). If Leo wants ρ to wait for 7150/4867 on `tools.py`, the decompose seat adds them; the PRD says so.

## New leaf list and order

13 leaves: α (M–L, 800–1,100: state.py + fixture + fake store), ρ (M, 300–500), β (M, 500–700), γ (M–L, 600–900), δ (M, 400–600), ε (M, 400–600), ζ (S–M, 300–450), η (M, 350–500), σ1 (M, 350–500; after 6528), σ2 (M, 350–500; new `session_self_exit.py`; after σ1 and 6922), σ3 (human-run), κ (docs `complexity: simple` ~150 lines + one test; after 6959), ω (M, 400–600). Order: α ∥ ρ ∥ (σ1 after 6528) → β, γ, ε, ζ ∥ δ (after σ1) → η → ω → κ (after 6959); σ2 (after σ1, 6922) → σ3. Signal leaf ω.

## Re-measurement

- Sidecar loads through `shared.capability_manifest.load_capability_manifest`: 13 blocks, `signal_leaves: [ω]` (one YAML scalar needed quoting after the rewrite; fixed).
- 55 `delivered_check`s: 52 FAIL at the worktree `HEAD` (= main `3ed84d5c5a`) under `evaluate_grep_at_tree` / `evaluate_path_at_tree`; the two ρ script checks exit 1 today (`check_method_param_wiring.py` finds no `include_rows`); ω's script is absent (fails by construction); the two `manual` checks are gate-excluded. The 3 PASS-today checks are κ's `expect: absent` phrases, which live only in the sibling seat's uncommitted working tree (`grep -c` = 1 each there) and flip to FAIL at the wave commit, before decompose — stated in both twins with `GATE_REF`.
- `lint_delivered_checks` at `HEAD`: no finding in my harness (the critic's run at `main` reported `vacuous_absent` for those same κ checks, which is the vacuity documented above; it clears when the phrases reach main).
- No multi-path grep remains; the vocabulary pattern uses literal pipes in the stated order.

Not done, by instruction: no commit; the stray `.venv` in the worktree left in place.
