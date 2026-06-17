# Capability manifest — agent-liveness-telemetry-resume-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Evidence verified on `main`
`d82a1aa732`, 2026-06-17. Line refs drift; symbols are canonical. Empty-value sentinel for
the field-population sub-check: `None` (the `transcript_turns` field is `None` when the
transcript could not be read — a *deliberate* sentinel with a defined conservative meaning,
§5 decision 3, not an unpopulated-field bug).

## α — Transcript-read classifier (Component 1, must-have core)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Orchestrator knows the run's session id without the result JSON | grep:`workflow.py:6207` `session_id_val = str(uuid.uuid4())`; passed `session_id=session_id_val` (:6235); persisted pre-launch `write_agent_session(...)` (:6210, artifacts.py:642) | PASS wired |
| Orchestrator owns `CLAUDE_CONFIG_DIR` (the transcript root) | grep:`config_dir.py:36` `claude-config-{task_id}`; set into agent env `invoke.py:209`; `projects/` kept per-task `config_dir.py:23` | PASS wired |
| Killed CLI leaves a readable transcript jsonl on disk | brief transcript evidence: 4415 `d498f369-….jsonl` (43 turns) and 4360 `590a40bd-….jsonl` (120 turns) both read **post-kill**; the CLI writes the transcript incrementally (it is what `--resume` replays) | PASS (artifact exists post-kill) |
| `is_zero_output_timeout` exists to rewrite | grep:`cli_invoke.py:251-272` `def is_zero_output_timeout(result)` | PASS wired |
| `_parse_claude_output` empty-stdout branch to extend with the field | grep:`cli_invoke.py:967-976` empty-stdout → `subtype='error_empty_output'` | PASS wired |
| `AgentResult`/`_SubprocessResult` dataclasses to add `transcript_turns` to | grep:`cli_invoke.py:235-248` `AgentResult` fields incl. `timed_out`, `proc_tree` | PASS wired |
| `_run_subprocess` timeout path is the place to read the transcript on kill | grep:`cli_invoke.py:1146-1195` `TimeoutError` → SIGTERM/SIGKILL → `_SubprocessResult(timed_out=True, ...)` | PASS wired |
| `transcript_turns` is **populated** with a real int on the kill path (field-population, not declared-only) | α's `count_transcript_turns` returns an `int` count from the on-disk jsonl; `None` only on read failure (§5 decision 3 — defined sentinel, legacy fallback) | PASS populated (non-sentinel on the success-read path) |
| `_capture_zero_output_evidence` exists to extend (currently proc_tree-only) | grep:`workflow.py:3922-3963` writes `proc_tree`/`config_dir_listing`, **no transcript** | PASS wired |
| `steward._is_empty_output` exists with the false "no work done" docstring to fix | grep:`steward.py:808-819` docstring "no real work was done" | PASS wired |
| Consumers route through the predicate (no per-call-site classify) | grep:`workflow.py:3789` `if is_zero_output_timeout(result)`; `cli_invoke.py:680` cap-retry guard | PASS wired |

## β — Two-regime liveness watchdog (Component 2)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Single flat-wall timeout to replace with the watchdog | grep:`cli_invoke.py:1142-1145` `await asyncio.wait_for(proc.communicate(...), timeout=timeout_seconds)` | PASS wired |
| Per-role timeout selection (the post-turn-1 ceiling) | grep:`workflow.py:6139` `timeout_val = getattr(timeouts_cfg, role_key, ...)`; passed `timeout_seconds=timeout_val` (:6234) | PASS wired |
| `TimeoutsConfig` to add `startup_grace_secs` | grep:`config.py:193-216` `class TimeoutsConfig`, `implementer: float = 1200.0` | PASS wired |
| SIGTERM→SIGKILL-group kill path to preserve unchanged | grep:`cli_invoke.py:1152-1194` `proc.terminate()` (5 s flush) → `terminate_process_group(proc, pgid)` | PASS wired |
| Live transcript reader available | producer:task-α (upstream of β) — `count_transcript_turns` | PASS producer upstream |
| `_run_subprocess` callers already hold session id + config dir to thread in | grep:`invoke.py:177-178` `--session-id session_id`; `invoke.py:208-209` `CLAUDE_CONFIG_DIR`; cli_invoke.py:953 `_run_subprocess(cmd, cwd, env, ...)` | PASS wired |
| Numeric floor — `startup_grace_secs`=120 s vs the method "floor" (observed turn-1 latency) | floor:`120 s` ≫ observed turn-1 ≈ 6 s (brief, reify-4415); 20× margin; not an accuracy bound (liveness timeout) — floor branch satisfied | PASS (bound > floor) |

## γ — Resume across the wall (Component 3)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Crash-recovery resume lifecycle to reuse | grep:`workflow.py:6193-6205` `_pending_resume_session_id`/`_pending_resume_role` → `resume_session_id` → `--resume` | PASS wired |
| `--resume` plumbing through to the CLI | grep:`cli_invoke.py:869-871` `cmd.extend(['--resume', resume_session_id])`; invoke.py:168-169 | PASS wired |
| The killed run's session id is recoverable for the resume | grep:`workflow.py:6207` minted up-front; `_invoke` has `session_id_val` in scope at the call; sidecar `read_agent_session` (artifacts.py:658) as backstop | PASS wired |
| `is_timed_out_with_progress` predicate to branch on | producer:task-α (upstream of γ) | PASS producer upstream |
| Cap-retry resume-clear guard is gated by construction (rejection-mechanism / negative assertion) | grep:`cli_invoke.py:679-690` guard fires iff `is_zero_output_timeout(result) and resume_session_id`; after α a progress run has `is_zero_output_timeout==False` ⇒ guard does **not** fire. δ-B4 authors a progress result with `resume_session_id` set and asserts `_reset_for_fresh_retry` is NOT called — i.e. the *non-clear* is observed to hold | PASS (gated; bound as δ-B4) |
| DAG-direction (anti-inversion) | α is upstream of γ; γ is upstream of δ — no owner depends on its consumer | PASS producer upstream |

## δ — End-to-end liveness boundary gate (B+H integration tests)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| All structural capabilities (classifier, watchdog, resume) | producers: α, β, γ — all upstream of δ | PASS producer upstream |
| B6 long-synchronous-tool-not-false-killed producible | β's two-regime contract (§3): post-turn-1 agents are not idle-killed; a fake/real agent emitting turn-1 then a > grace synchronous tool exercises it | PASS (β contract) |
| B7 unreadable-transcript fallback producible | §5 decision 3: `transcript_turns is None` → legacy `turns==0 && cost==0`; β does not early-kill on `None`; a missing transcript path is trivially constructible in a test | PASS (α/β contract) |
| No numeric throughput/accuracy floor asserted in any leaf | only β's `startup_grace_secs` (liveness timeout, floor-checked above) — no accuracy bound anywhere | PASS (floor branch n/a) |

No FAIL bindings. Batch clear to queue.
