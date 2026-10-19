# Capability manifest — review-all program tooling

**PRD:** `plans/review-all-program-tooling-prd.md`. **Machine twin:** `plans/review-all-program-tooling-prd.capability-manifest.yaml` (the bindings below and the sidecar's `delivered_check`s are the same data; `task_id`s are stamped by `commit_planning` at decompose).
**Substrate verified** on main `3ed84d5c5a` (2026-10-19), author mode; re-checked at the D8 critic fold against `93f478fa3b` the same day (no cited file changed). Every cite resolves by symbol; evidence pointers are backticked `grep:<repo-root-relative path>[::<symbol>]` spans (task 6423's grammar).

**Polarity.** Every `grep`/`path` check below was evaluated with `shared.delivered_check_polarity.evaluate_grep_at_tree` / `evaluate_path_at_tree` at the worktree `HEAD` (= main `3ed84d5c5a`) and read FAIL; `lint_delivered_checks` (named dict checks, `files=None`) reports exactly three findings: `warn absent_overbroad` on σ1's `vocabulary-copy-removed-from-ownership`, which clears when σ1's `metadata.files` names `scripts/sitting/ownership.py` (contract 7; verified), and `reject vacuous_absent` on κ's two `expect: absent` checks, whose phrases live in the sibling seat's **uncommitted** working-tree `deliberation.md` and `SKILL.md` (the lead's post-D10 wording): they read PASS (vacuous) against `HEAD` and `main` today and FAIL once the wave commits, which precedes decompose; the polarity gate reads `main` of `project_root` (`GATE_REF`), so re-measure at decompose after that commit. The `script` checks fail by construction (`scripts/check_review_all_program_fixture.py` does not exist; `check_method_param_wiring.py` finds no `include_rows`). Rules applied after D8 M2: no multi-path `grep` (git grep over several paths passes on any one of them — every required path has its own check; `path` checks are conjunctive and may list several); a pipe inside a pattern is the literal `\|`, and the vocabulary line is pinned in one stated order; a widened parameter is proved with `scripts/check_method_param_wiring.py --forwards-to`, not a grep. Patterns are single-line (`deliberation.md` §Launching step 4(b)) and anchored on a symbol where one exists; no `file:line`.

## Decompose-time contracts (not in the PRD's words; the gates depend on them)

1. **Fake store tells the truth (α → ω).** `fixture/fake_store.py` drops unknown arguments and echoes `include_rows` exactly as ρ specifies; `--no-include-rows` accepts the argument, ignores it, sends no echo and returns 5-field rows (FastMCP's `extra='ignore'`, D8 B1). ω's walk is only as true as that shape, so ρ's own boundary rows are the proof of the live server and ω carries ρ as a prerequisite.
2. **Home override (every script → ω).** Every script reads `REVIEW_ALL_HOME` (default `~/.claude/review-all`) so the fixture walk never touches the real home; `program.json.plans_dir` is where `--publish` copies.
3. **Server URL and project root (α → every script).** `state.py` reads `FUSED_MEMORY_URL` (default the main checkout's `.mcp.json` entry) and takes `project_root` from `program.json`; no script takes them as positional arguments.
4. **Isolated interpreter (ω).** ω runs each script with `sys.executable -I -S` from a temp cwd outside the repo; a script's `sys.path` inserts of `orchestrator/src` and `scripts` are computed from its own `__file__`, so they survive `-S`.
5. **Record resolution (δ → η).** `--spawn` is accepted on the author pass only. δ writes `{title, stream, pass, spawned_after, record_dir: null}`; η is the only writer of `record_dir`, resolved by role prefix, `record.title` and an mtime after `spawned_after`.
6. **Self-exit flag (σ1 → σ2, δ).** The marker is keyed on the record dir; its content is provenance.
7. **Decompose metadata.files (σ1).** σ1's `metadata.files` must name `scripts/sitting/ownership.py`, or its `expect: absent` check lints `absent_overbroad`. `CLAUDE_SPAWN_SELF_EXIT=1` is a per-launch input; `_run_launching` stamps `record.self_exit`; the child's environment stays the five documented vars; the trailer carries the marker instruction only when the flag was set.
8. **Script-check timeout (ω, ρ).** `timeout_secs: 90` and `60` < the orchestrator's `delivered_checks.check_timeout_secs` (120); the fake store binds an ephemeral port and the walk is ~10 s locally.

## Known risks (G6)

- `background_tasks` / `session_crons` element shapes are unmeasured (their presence is verified statically in the 2.1.303 binary); σ2 refuses on absence and σ3 measures. Both are in the DAG above the signal, not below it.
- `SIGINT` graceful exit rests on binary strings; σ3 is the measurement, and σ2's failure mode is today's behaviour (the session stays open).
- The numeric bounds in the PRD — 600-character `notes`, 100-byte degenerate inbox file, 48 KB page budget, 16,240-byte tmux command limit — are inherited (S3 §D, R2 §2.4, task 6385, D8 B2) and asserted by nothing here as accuracy claims.

## Bindings

## α — state.py, fixture, fake store

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `lookup-tool-wired` | capability→substrate (wired) — `grep:fused-memory/src/fused_memory/server/tools.py::find_tasks_by_metadata` | PASS | — |
| `unknown-argument-dropped-silently` | capability→substrate (wired, as a hazard) — FastMCP `ArgModelBase` keeps `extra='ignore'` (measured); task 4867 | PASS | — |
| `mcp-wire-shape` | capability→substrate (wired) — `grep:scripts/migrate_metadata_modules_to_files.py::FusedMemoryClient` | PASS | — |
| `held-leaf-key-shapes` | capability→substrate (wired) — `grep:skills/review-all/references/report-format.md` §Artefact 3 Held leaf | PASS | — |
| `state-library-present` | built by this leaf | PASS | path present: `skills/review-all/scripts/program/state.py` |
| `fixture-and-fake-store-present` | built by this leaf | PASS | path present: `fixture/fake_store.py`, `fixture/store.json`, `fixture/program.json` under `skills/review-all/scripts/program/` |
| `return-schema-constant` | built by this leaf | PASS | grep present `^RETURN_SCHEMA = 'review-all-prd-return/1'` in `state.py` |
| `seam-key-constants` | built by this leaf | PASS | grep present `^SEAM_WAIT_KEYS` in `state.py` |
| `ledger-ops-and-ruling-ids` | built by this leaf (decision 17) | PASS | grep present `^def next_ruling_id\(` in `state.py` |
| `reader-refuses-missing-echo` | built by this leaf | PASS | grep present `^class StoreReaderUnsupported\(` in `state.py` |

## ρ — find_tasks_by_metadata(include_rows=True), echoed

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `lookup-layering` | capability→substrate (wired) — `grep:fused-memory/src/fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend.find_tasks_by_metadata`, `grep:fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor.find_tasks_by_metadata`, `grep:fused-memory/src/fused_memory/server/tools.py::find_tasks_by_metadata`, `grep:fused-memory/src/fused_memory/backends/metadata_lookup.py::validate_metadata_query` | PASS | — |
| `dependency-edges-at-backend` | capability→substrate (wired) — `grep:fused-memory/src/fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend.get_dependency_edges` | PASS | — |
| `budget-fitter` | capability→substrate (wired) — `grep:fused-memory/src/fused_memory/server/tools.py::_fit_rows_to_budget` | PASS | — |
| `no-ledger-row-needed` | capability→substrate (wired) — `grep:shared/tests/test_loop_blocking_gate.py` hashes the call node; ρ adds no blocking call | PASS | — |
| `param-wiring-checker` | capability→substrate (wired) — `grep:scripts/check_method_param_wiring.py` | PASS | — |
| `tool-declares-and-forwards-include-rows` | built by this leaf | PASS | script `scripts/check_method_param_wiring.py --file …/server/tools.py --function find_tasks_by_metadata --param include_rows --forwards-to find_tasks_by_metadata`, 60 s |
| `interceptor-declares-and-forwards-include-rows` | built by this leaf | PASS | script `scripts/check_method_param_wiring.py --file …/middleware/task_interceptor.py --function find_tasks_by_metadata --param include_rows --forwards-to find_tasks_by_metadata`, 60 s |
| `backend-include-rows` | built by this leaf | PASS | grep present `include_rows` in `sqlite_task_backend.py` |
| `include-rows-validated-as-bool` | built by this leaf | PASS | grep present `include_rows` in `metadata_lookup.py` |
| `include-rows-echoed` | built by this leaf — the server half of B1 | PASS | grep present `'include_rows':` in `tools.py` |
| `lone-row-elision` | built by this leaf | PASS | grep present `truncated_row` in `tools.py` |
| `live-signal-with-echo` | built by this leaf | PASS | manual — needs the running service restarted onto the landed code |

## β — apply_return.py and record.py

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `state-library` | capability→producer (upstream) — α | PASS | — |
| `scripts-present` | built by this leaf | PASS | path present: `apply_return.py`, `record.py` |
| `check-mode` | built by this leaf | PASS | grep present `'--check'` in `apply_return.py` |
| `from-store-reconstruction` | built by this leaf | PASS | grep present `'--from-store'` in `apply_return.py` |

## γ — program_doc.py

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `state-library` | capability→producer (upstream) — α | PASS | — |
| `section-shape` | capability→substrate (wired) — `grep:skills/hotspot-survey/references/report-format.md` §Artifact 3; `grep:skills/review-all/references/report-format.md` §Artefact 3 | PASS | — |
| `run-record-method-extra` | capability→substrate (wired) — `grep:skills/review-all/references/report-format.md` §Artefact 1 `method.extra` | PASS | — |
| `script-present` | built by this leaf | PASS | path present: `program_doc.py` |
| `heading-self-check` | built by this leaf | PASS | grep present `--allow-drop` in `program_doc.py` |
| `publish-step` | built by this leaf | PASS | grep present `'--publish'` in `program_doc.py` |

## δ — gen_brief.py

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `state-library` | capability→producer (upstream) — α | PASS | — |
| `prompt-file-and-self-exit-inputs` | capability→producer (upstream) — σ1 | PASS | — |
| `run-record-findings-and-filings` | capability→substrate (wired) — `grep:skills/review-all/references/report-format.md` §Artefact 2 | PASS | — |
| `script-present` | built by this leaf | PASS | path present: `gen_brief.py` |
| `spawn-command-uses-prompt-file` | built by this leaf | PASS | grep present `--prompt-file` in `gen_brief.py` |
| `spawn-command-opts-in-to-self-exit` | built by this leaf | PASS | grep present `CLAUDE_SPAWN_SELF_EXIT` in `gen_brief.py` |

## ε — clerk_batch.py

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `state-library` | capability→producer (upstream) — α (`tasks_with_delivered_checks`) | PASS | — |
| `delivered-checks-shape` | capability→substrate (wired) — `grep:docs/task-authoring.md` delivered_checks | PASS | — |
| `commit-planning-hold-kinds` | capability→substrate (wired) — `grep:fused-memory/src/fused_memory/server/tools.py::commit_planning` | PASS | — |
| `script-present` | built by this leaf | PASS | path present: `clerk_batch.py` |
| `clerk-rules-home` | built by this leaf | PASS | grep present `^CLERK_RULES` in `clerk_batch.py` |
| `impact-scan` | built by this leaf | PASS | grep present `^def impact_scan\(` in `clerk_batch.py` |

## ζ — gate_check.py and seam_scan.py

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `state-library` | capability→producer (upstream) — α | PASS | — |
| `gate-identity-both-forms` | capability→substrate (wired) — `grep:skills/_shared/filing-the-trigger-chain.md` §3 and §4; `check_run_completion.py` absent on main (task 6389 pending) | PASS | — |
| `scripts-present` | built by this leaf | PASS | path present: `gate_check.py`, `seam_scan.py` |
| `form-detection` | built by this leaf | PASS | grep present `completion_gate` in `gate_check.py` |

## η — watch_sessions.py

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `state-library` | capability→producer (upstream) — α | PASS | — |
| `apply-return` | capability→producer (upstream) — β | PASS | — |
| `outcome-vocabulary-home` | capability→producer (upstream) — σ1 | PASS | — |
| `registry-importable-under-isolated-python` | capability→substrate (wired) — `grep:orchestrator/src/orchestrator/session_registry.py::TERMINAL_STATUSES`, `grep:orchestrator/src/orchestrator/session_registry.py::RESULT_FILENAME`; loads under `python3 -I -S` (measured) | PASS | — |
| `script-present` | built by this leaf | PASS | path present: `watch_sessions.py` |
| `final-rule` | built by this leaf | PASS | grep present `^def is_final\(` in `watch_sessions.py` |
| `imports-the-registry-not-a-copy` | built by this leaf | PASS | grep present `from orchestrator.session_registry import` in `watch_sessions.py` |

## σ1 — spawn contract

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `trailer-and-payload` | capability→substrate (wired) — `grep:skills/spawn/spawn-claude.sh`, `grep:tests/scripts/test_spawn_claude.py`; tmux's 16,240-byte limit (D8 B2) | PASS | — |
| `pointer-prompt-prescribed` | capability→substrate (wired) — `grep:skills/spawn/SKILL.md` "spawn a short prompt pointing at it" | PASS | — |
| `outcome-parser-today` | capability→substrate (wired) — `grep:scripts/sitting/ownership.py::RESULT_OUTCOMES`, `grep:scripts/sitting/ownership.py::parse_result_outcome`; task 6528 holds the file | PASS | — |
| `launching-record-from-env` | capability→substrate (wired) — `grep:orchestrator/src/orchestrator/session_registry.py::_run_launching` | PASS | — |
| `prompt-file-flag-in-script` | built by this leaf | PASS | grep present `--prompt-file` in `spawn-claude.sh` |
| `prompt-file-flag-documented` | built by this leaf | PASS | grep present `--prompt-file` in `skills/spawn/SKILL.md` |
| `trailer-vocabulary-line` | built by this leaf | PASS | grep present `outcome: done\|blocked\|abandoned\|handed-off\|in-progress` in `spawn-claude.sh` |
| `spawn-skill-vocabulary-line` | built by this leaf | PASS | grep present `handed-off\|in-progress` in `skills/spawn/SKILL.md` |
| `watcher-skill-vocabulary-line` | built by this leaf | PASS | grep present `handed-off\|in-progress` in `skills/escalation-watcher/SKILL.md` |
| `vocabulary-home-in-registry` | built by this leaf | PASS | grep present `^RESULT_OUTCOMES` in `session_registry.py` |
| `vocabulary-copy-removed-from-ownership` | built by this leaf | PASS | grep absent `^RESULT_OUTCOMES` in `scripts/sitting/ownership.py` |
| `self-exit-input` | built by this leaf | PASS | grep present `CLAUDE_SPAWN_SELF_EXIT` in `spawn-claude.sh` |
| `self-exit-record-field` | built by this leaf | PASS | grep present `self_exit` in `session_registry.py` |
| `exit-requested-instruction` | built by this leaf | PASS | grep present `exit-requested` in `spawn-claude.sh` |

## σ2 — opt-in Stop-hook self-termination

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `trailer-and-record-flag` | capability→producer (upstream) — σ1 | PASS | — |
| `probes-extracted` | capability→producer (upstream) — task 6922 | PASS | — |
| `stop-hook-wired` | capability→substrate (wired) — `grep:orchestrator/src/orchestrator/session_hooks.py::run_stop`, `grep:orchestrator/src/orchestrator/session_hooks.py::_run_status_refresh_and_retitle`; `~/.claude/settings.json` Stop → `skills/spawn/hooks/stop.sh` | PASS | — |
| `shared-host-guard` | capability→substrate (wired) — `grep:orchestrator/src/orchestrator/session_registry.py::_is_shared_session_host`, `grep:orchestrator/src/orchestrator/session_registry.py::SessionRecord` | PASS | — |
| `hook-input-lists` | capability→substrate (verified statically) — the 2.1.303 Stop schema's optional `background_tasks` / `session_crons` arrays | PASS | — |
| `self-exit-module` | built by this leaf | PASS | path present: `orchestrator/src/orchestrator/session_self_exit.py` |
| `marker-in-record-dir` | built by this leaf — content stamped, not compared (D10 N4) | PASS | grep present `exit-requested` in `session_self_exit.py` |
| `background-work-guard` | built by this leaf | PASS | grep present `session_crons` in `session_self_exit.py` |
| `exit-request-sink` | built by this leaf | PASS | grep present `exit_requested_at` in `session_registry.py` |
| `stdlib-containment-row` | built by this leaf | PASS | grep present `session_self_exit` in `orchestrator/tests/test_session_registry.py` |
| `unit-tests` | built by this leaf | PASS | path present: `orchestrator/tests/test_session_self_exit.py` |

## σ3 — acceptance probe (human-run)

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `self-termination` | capability→producer (upstream) — σ1, σ2 | PASS | — |
| `sigint-graceful-exit` | capability→substrate (unmeasured) — this leaf measures it | PASS | manual — the human records the observations in the task |

## κ — skill text

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `scripts-landed` | capability→producer (upstream) — α–η, ω | PASS | — |
| `held-leaf-prose-owner` | capability→producer (upstream) — task 6959 | PASS | — |
| `phrases-present-today` | capability→substrate (wired) — the two phrases in this worktree's working tree | PASS | — |
| `by-hand-pointer-removed` | built by this leaf | PASS | grep absent `by hand until the program-tooling PRD lands` in `deliberation.md` |
| `handoff-pointer-removed` | built by this leaf | PASS | grep absent `by hand until then` in `SKILL.md` |
| `scripts-cited` | built by this leaf | PASS | grep present `apply_return\.py` in `deliberation.md` |
| `clerk-rules-cited` | built by this leaf | PASS | grep present `CLERK_RULES` in `deliberation.md` |
| `held-leaves-section` | built by this leaf | PASS | grep present `## Held leaves` in `report-format.md` |
| `seam-keys-cited` | built by this leaf | PASS | grep present `SEAM_WAIT_KEYS` in `report-format.md` |
| `cites-test` | built by this leaf | PASS | path present: `tests/scripts/test_review_all_skill_text_cites_scripts.py` |

## ω — integration gate

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `all-scripts` | capability→producer (upstream) — α, β, γ, δ, ε, ζ, η; ρ | PASS | — |
| `tests-collected` | capability→substrate (wired) — `grep:tests/scripts/test_scripts_module_config.py::test_root_fleet_chain_and_scripts_module_agree_on_the_scripts_suites`; `grep:tests/scripts/test_skills_module_config_decision.py::test_skills_has_no_tests_of_its_own` | PASS | — |
| `fixture-walk-script` | built by this leaf — executable with a shebang | PASS | script `scripts/check_review_all_program_fixture.py`, 90 s |
| `fixture-walk-test` | built by this leaf | PASS | path present: `tests/scripts/test_review_all_program_fixture_walk.py` |
