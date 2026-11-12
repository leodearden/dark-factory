# D11 — second fold of `review-all-program-tooling-prd` from the D10 fold-verifier (2026-10-19)

Files edited in place (worktree only, uncommitted; the skill files were NOT touched — the lead's post-D10 wording was re-read and κ re-pointed at it):
- `/home/leo/src/dark-factory/.claude/worktrees/review-all-amend-20261019/plans/review-all-program-tooling-prd.md`
- `/home/leo/src/dark-factory/.claude/worktrees/review-all-amend-20261019/plans/review-all-program-tooling-prd.capability-manifest.yaml`
- `/home/leo/src/dark-factory/.claude/worktrees/review-all-amend-20261019/plans/review-all-program-tooling-prd.capability-manifest.md`

Verified before folding: `session_registry.py` is 5,879 lines; `session_hooks.py::_OWNER_ONLY_SESSION_START_SOURCES = {'clear','compact'}` and `_bind_claude_session_id(allow_rebind=…)` re-bind the record on a re-mint; `orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment` exists with entrypoint rows; `find_tasks_by_metadata` echoes `key`/`value`/`statuses` and not `include_value`; the lead's `deliberation.md` line 54 now states the one channel and the decompose-only keys, line 61 no longer spells the vocabulary, and `SKILL.md` §Lead handoff says "by hand until then" (present in the working tree, absent on main); `check_review_all…` and `next_ruling_id` patterns absent today.

## Items

| Item | Disposition |
|---|---|
| N1 `seam_scan.py` reads YAML | **folded**: store only (`metadata.files` + `metadata.delivered_checks` via `tasks_with_delivered_checks`); unfiled PRDs out of reach, said in §1, §7, ζ; B2 fixture gains a second out-of-program task anchoring the path in `delivered_checks`; B14 names `seam_scan.py` under `-I -S`. |
| N2 author brief / B9 spawns decompose | **folded**: both briefs carry the schema, example, absolute inbox path and `--check` line; only a `--spawn` brief carries the spawn paragraph; `--spawn` is author-pass only and refused on decompose (δ, §1, B9 rewritten with a third, refused call; contract 5 note). |
| N3 skill-text contradiction / κ `second-channel-removed` | **folded** per the lead: check dropped (now vacuous by design); κ re-pointed at the lines as they stand (op list deletion, "by hand until the program-tooling PRD lands", SKILL.md "by hand until then"); `validate_return`'s off-pass-key rule stated (a pass-specific key on the other pass is a form error even when empty); decision 16 cites the amended sentence. |
| N4 session-id guard vs re-mint | **folded** (record-dir keying chosen): the marker is keyed on the registry record dir — only the session holding that dir's `CLAUDE_SPAWN_RESULT_FILE` reaches it and a resumed session forks onto its own record — its content is stamped as `marker_session_id`, not compared; the unmeasured premise (`CLAUDE_CODE_SESSION_ID` after a re-mint) is a §6 row; σ3 runs `/compact` before the marker and records whether the marker id equals the re-bound `claude_session_id`; B15 asserts re-mint tolerance. |
| N5 α's surface / `append` lock | **folded**: α lists `RETURN_SCHEMA`, `LEDGER_OPS`, `SEAM_WAIT_KEYS`, `SEAM_RELEASED_KEYS`, `validate_return`, `stream_of`, `waves`, `next_ruling_id`, the `REVIEW_ALL_HOME`/`FUSED_MEMORY_URL` resolution; `append` takes `fcntl.flock`; `apply_return` is all-or-nothing (validate and compute, then append under one lock); new α check `^def next_ruling_id\(`. |
| N17 `--handoff` without `program.json` | **folded on the skill side** by the lead: `--handoff` is specified for Phase 6 state only; §1 and κ say the Phase 5 → 6 handoff stays hand-written; B8's precondition unchanged. |
| M12 partial | **finished** with N5 (above). |
| M15 partial | **finished**: author-pass PRDs/manifests/sidecars live in the home under `prds/<S>/` and are published by `--publish` at the wave commit; the only longer window is the decompose-stamped sidecars (and `release`/`repath` edits), bounded from a wave's first decompose to its commit, which the lead makes as soon as the last decompose returns (H1, decision 15, κ). |
| N6 withheld/unproven events stamp | **folded**: a withheld launch-window or `may_bind`-false event stamps nothing; stated in H5 and B15. |
| N7 stamp before kill | **folded**: the one write stamps `attempted` + `exit_requested_at`; the marker is deleted after the kill; a raising `os.kill` stamps `kill_failed` with the errno in a second fail-soft write and keeps the marker, so the next Stop retries; the one-write rule governs the normal path. |
| N8 `session_registry.py` size | **folded as directed**: the vocabulary stays in `session_registry.py` (the T5 owner); H1 states the 5,879-line trade-off honestly — the record fields must be `SessionRecord` fields regardless, and a partition is that file's owner's call with its own measurement. |
| N9 fifth vocabulary site | **resolved by the lead's edit** (line 61 no longer spells it); κ's row does not name it; the mirror test covers the three live prose sites. |
| N10 κ's test fails on path-qualified spans | **folded**: scoped to bare backticked `<name>.py` spans (no `/`); path-qualified spans out of scope, reason stated. |
| N11 `/reflect` and `in-progress` reach every caller; no join rule | **folded**: σ1 adds the join rule to `skills/escalation-watcher/SKILL.md` §Joining (an `in-progress` outcome on a finished record is "ended at checkpoint" → explore); the remaining default question stays OQ7 for Leo. |
| N12 stdlib contract of `session_self_exit.py` | **folded**: stated in H5; registered in `TestStdlibOnlySelfContainment`'s entrypoint rows (new σ2 check `session_self_exit` in `test_session_registry.py`); 6922's probe module carries the same contract. |
| N13 low-priority gates | **folded**: §9 names 6528 (in progress, medium) gating σ1 → δ, η, ω; 6922 and 6959 (**low**) gating σ2 and κ; the decompose seat raises them to the batch's priority when it wires the edges or records why not. |
| N14 D9's numbers | **folded**: no D9 figure had leaked into the PRD; the twins' lint statement was wrong and is corrected (below). |
| N15 `TERMINAL_RESULT_OUTCOMES` | **folded**: stated as new in H1 and H5; `ownership.py` has no such symbol. |
| N16 no mechanical check on the echo | **folded**: ρ gains `include-rows-echoed` (`'include_rows':` present in `tools.py`; fails today, `include_value` is not echoed either). |
| `held[].waits_on` vs `x_seam_wait.stream` | **folded**: stated as one fact under a mapping; `validate_return` compares them (H1, Return block). |

Declined: nothing outright. One partial: N1's "say plainly that an unfiled PRD is out of reach" is said, and the store-only rule is now also the reason (no YAML reader), not only heuristic 12.

## Re-measurement (worktree HEAD = main `3ed84d5c5a`; main at `93f478fa3b` unchanged for every cited file)

- Sidecar loads: 13 blocks, `signal_leaves: [ω]`, **56** delivered checks (51 grep/path, 3 script, 2 manual).
- `evaluate_grep_at_tree` / `evaluate_path_at_tree`: every grep/path check FAILs except κ's two `expect: absent` checks (`by hand until the program-tooling PRD lands` in `deliberation.md`; `by hand until then` in `SKILL.md`), which PASS (vacuous) at HEAD and at main; both phrases are present once in the worktree working tree and flip to FAIL when the wave commits, before decompose. ρ's two `check_method_param_wiring.py` checks exit 1 today; ω's script is absent.
- `lint_delivered_checks` run correctly this time (named dict checks — my D9 harness passed `DeliveredCheck` objects, which the linter ignores silently, so D9's "no finding" was my measurement defect, not the sidecar's state; D10 was right): `reject vacuous_absent` ×2 on κ (expected until the wave commits) and `warn absent_overbroad` on σ1's `vocabulary-copy-removed-from-ownership` with `files=None`, which is empty with `files=['scripts/sitting/ownership.py']` — recorded as decompose contract 7 in the md twin. Both twins now state this.
- No multi-path grep; the vocabulary pattern uses literal pipes in the pinned order.

## Final leaf list and order

13 leaves: α (M–L, 800–1,100: `state.py` full surface + fixture + fake store), ρ (M, 300–500, with the echo), β (M, 500–700), γ (M–L, 600–900), δ (M, 400–600; `--spawn` author-only), ε (M, 400–600), ζ (S–M, 300–450; store-only), η (M, 350–500), σ1 (M, 350–500; after 6528; adds the join rule), σ2 (M, 350–500; `session_self_exit.py`, record-dir marker, `attempted`/`kill_failed`; after σ1 and 6922), σ3 (human; with `/compact`), κ (docs `complexity: simple` ~150 lines + bare-name mirror test; after 6959), ω (M, 400–600). Order: α ∥ ρ ∥ (σ1 after 6528) → β, γ, ε, ζ ∥ δ (after σ1) → η → ω → κ (after 6959); σ2 (after σ1, 6922) → σ3. Signal leaf ω. Decompose notes: σ1's `metadata.files` names `scripts/sitting/ownership.py`; raise 6922 and 6959 to the batch's priority or record why not.

Not done, by instruction: no commit; the skill files untouched; the stray `.venv` left in place.
