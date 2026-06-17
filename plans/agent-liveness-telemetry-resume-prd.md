# PRD: Agent liveness heartbeat + zero-output telemetry classifier fix + work-killed resume

**Date:** 2026-06-17 · **Status:** approved for decomposition · **Scope:** all
load-bearing code is dark-factory orchestrator/shared code; reify is a pure consumer
(its agents stop being mis-killed and mislabelled). **Approach:** B+H (mechanism count
≥ 8, core agent-invocation seam touched).

Cite by symbol; line refs are as-of `main` `d82a1aa732` and drift.

## 1. Consumer + user-observable surface (G1, G2)

**Consumer (the code that changes, all in `/home/leo/src/dark-factory`):**
- `shared/src/shared/cli_invoke.py` — `is_zero_output_timeout` (251-272), `_parse_claude_output`
  (962-976), `_run_subprocess` timeout/SIGTERM loop (1102-1204), the cap-retry resume-clear
  guard (679-690) inside `invoke_with_cap_retry`.
- `orchestrator/src/orchestrator/workflow.py` — `_execute_iterations` zero-output circuit
  breaker (3784-3827), `_capture_zero_output_evidence` (3922-3972), the per-role timeout
  selection + session-resume lifecycle in `_invoke` (6129-6212).
- `orchestrator/src/orchestrator/steward.py` — `_is_empty_output` (808-819).
- `orchestrator/src/orchestrator/config.py` — `TimeoutsConfig` (193-216),
  `max_consecutive_zero_output_timeouts` (922).

**User-observable surface (what an operator sees after this PRD lands):**
1. A task whose agent legitimately runs long work (a 15-min test suite, a slow solver) is
   no longer killed mid-progress and auto-deferred after two timeouts. The work-killed
   iteration is reclassified `TIMED_OUT_WITH_PROGRESS`, does **not** trip the zero-output
   circuit breaker, and the task **resumes the same session** (`--resume`) and completes —
   visible in the journal as `resuming prior session <id>` and in `.task/agent_session.json`.
2. A genuinely wedged agent (0 assistant turns ever — the original from-source-build /
   `uv` / MCP-startup wedge) is killed within ~`startup_grace_secs` (default 120 s), not
   after burning the full 1200 s wall.
3. `zero_output_evidence-*.json` (and any escalation built from it) carries the **transcript
   turn count + last tool + last ~5 records** — so no downstream human or debugger agent can
   repeat the proc-tree-only misdiagnosis that motivated this PRD.

## 2. Premise validation (G6 — the whole reason this PRD exists; validated by transcripts + code, 2026-06-17)

The multi-day "agents hang at startup (0-turn MCP-startup wedge)" investigation was a
**misdiagnosis**. The signature `turns=0 / cost_usd=0.0 / subtype=error_empty_output /
duration≈1200s / SIGTERM` was read as "the CLI never reached turn 1." It is also the exact
signature of **a productive agent SIGTERM-killed at the fixed role wall-clock before it
could emit its end-of-run result JSON**.

1. **The predicate is structurally blind.** `claude --print --output-format json` emits its
   single result object (`num_turns`, `total_cost_usd`, `subtype`) **only at the very end**.
   A SIGTERM at the wall → empty stdout → `_parse_claude_output` (cli_invoke.py:967-976)
   returns `subtype='error_empty_output'` with `turns`/`cost` **defaulting to 0**. Then
   `is_zero_output_timeout` (cli_invoke.py:272) `= timed_out and turns==0 and cost_usd==0.0`
   → True **regardless of how much work happened**. The fields are parsed from an artifact a
   killed session never produces.
2. **Hard transcript evidence (ground truth the diagnosis never consulted).**
   - reify 4415, session `d498f369-5a62-4b90-868e-d43581129b02`: **43 assistant turns over
     1198.86 s**, reached turn-1 in ~6 s, killed mid-`cargo test` (`Exit 137`) at the 1200 s
     wall. Yet its `zero_output_evidence-iter2.json` shows the 0-turn signature.
   - reify 4360, session `590a40bd-71de-4e50-b206-c3ad1c684c92`: **120 assistant turns over
     1081 s**, last tool `Agent`. Its proc_tree carries the exact `uv` + `futex_wait_queue`
     signature treated as the smoking gun of the genuine wedge — proving **the proc_tree is
     not diagnostic** (`uv run` parked in futex idly supervising a healthy `plan_tools` child
     while claude worked).
3. **The blindness is baked into prose.** `steward._is_empty_output` (steward.py:813-817)
   asserts in its docstring "no real work was done"; `_capture_zero_output_evidence`
   (workflow.py:3950-3961) writes **proc_tree only, no transcript** — so a debugger agent
   reads the proc_tree, sees the futex signature, and re-escalates "MCP startup wedge."
4. **The original wedge was real and is already fixed.** The from-source-build / `uv` wedge
   (jcodemunch pin, plan_tools direct-interpreter, `MCP_TIMEOUT=30000` — apply_mcp_startup_env,
   invoke.py:153) genuinely hung pre-turn-1 (reify-4429: 10/10 iterations, 0 turns). A
   *different* failure — slow legitimate work killed at the wall under host oversubscription —
   now produces an *identical* signature and masquerades as the old one. **This PRD
   distinguishes the two by reading the transcript the killed session leaves on disk.**

**Substrate that makes the fix possible (G3 — all verified on `main` this session):** the
orchestrator **pre-mints** the session id (`session_id_val = str(uuid.uuid4())`,
workflow.py:6207) and passes it via `--session-id` (cli_invoke.py:882) **and** persists it to
`.task/agent_session.json` before the subprocess starts (`write_agent_session`,
artifacts.py:642 / workflow.py:6210). It owns `CLAUDE_CONFIG_DIR` (`TaskConfigDir`,
config_dir.py:36; set into the agent env at invoke.py:209). The CLI writes the transcript
under `<CLAUDE_CONFIG_DIR>/projects/<cwd-slug>/<session-id>.jsonl` (config_dir.py:23 keeps
`projects/` per-task). **Therefore the orchestrator can locate and read the transcript of a
just-killed run without the result JSON** — the load-bearing fact the misdiagnosis never
exploited.

## 3. Approach

Three components, sequenced smallest-and-safest first. The transcript-read primitive (α) is
the foundation both other components reuse.

- **Component 1 / α — Telemetry classifier fix (the minimal must-have core).** Classify a
  timed-out run by reading the session transcript, not the absent-JSON defaults. Add
  `transcript_turns: int | None` to `_SubprocessResult` + `AgentResult`. A shared reader
  resolves the transcript by **glob-by-session-id** (`<config>/projects/*/<session-id>.jsonl`)
  — robust to the CLI's cwd-slug convention drifting across versions — and counts
  `type == "assistant"` records. `is_zero_output_timeout` becomes `timed_out and
  (transcript_turns or 0) == 0` (genuine pre-turn-1 wedge); a new `is_timed_out_with_progress`
  is `timed_out and (transcript_turns or 0) > 0`. `_capture_zero_output_evidence` gains the
  turn count, last tool, and last ~5 records. `steward._is_empty_output` stops asserting "no
  work done" for progress runs.
- **Component 2 / β — Two-regime liveness watchdog (replaces the flat wall).** Replace
  `_run_subprocess`'s single `asyncio.wait_for(communicate, timeout=role_timeout)` with a
  progress watchdog that polls the transcript: (a) **startup regime** — if **no assistant
  turn appears within `startup_grace_secs`** (default 120 s, ≫ the observed ~6 s turn-1
  latency), kill fast → this is the genuine pre-turn-1 wedge; (b) **working regime** — once
  ≥ 1 turn is seen, the agent has proven liveness; let it run to a **generous absolute
  ceiling** (the existing per-role timeout). The kill path is unchanged (SIGTERM 5 s flush →
  SIGKILL process-group, cli_invoke.py:1152-1194). The watchdog stamps `transcript_turns`
  onto the result from its live tracking (preserving α's field contract).
- **Component 3 / γ — Resume across the wall.** On an `is_timed_out_with_progress`
  implementer result, set `_pending_resume_session_id = <killed session id>` +
  `_pending_resume_role` so the next iteration's `_invoke` re-dispatches with `--resume`
  (the existing crash-recovery lifecycle, workflow.py:6193-6205), continuing with full
  context instead of discarding ~20 min of work. The cap-retry resume-clear guard
  (cli_invoke.py:679-690) keys on `is_zero_output_timeout`, so once α reclassifies the run it
  no longer fires for work-killed results — the clear is gated by construction.

### The load-bearing design decision: liveness contract is two-regime, not naive idle-kill

The brief's Component 2 sketch (kill on 120-180 s of transcript silence, any time) **cannot
be taken literally**, because it false-kills Signal 1's own example. During a single
**synchronous** long tool call (a 15-min foreground `cargo test`), the agent emits one
`assistant` `tool_use` record and then the transcript is **silent for the tool's entire
duration** until the `tool_result` arrives. A flat 120-180 s idle-kill would SIGKILL that
agent at 120 s — *worse* than the 1200 s wall it replaces, and directly contradicting
"a 15-min test suite … completes."

The resolution confines the aggressive idle-kill to the **pre-turn-1** window, where it is
both safe (legit agents reach turn-1 in ~6 s, so 120 s silence there genuinely means a
wedge) and high-value (it is exactly the original from-source-build wedge). Post-turn-1, we
do **not** try to distinguish "long synchronous tool" from "hung after turn-1" with a short
timer — that distinction is unreliable (4360 proves proc-activity/futex is not diagnostic).
Instead we let the agent run to a generous ceiling and rely on Components 1+3 to make a
ceiling-kill **non-destructive**: it is reclassified and the session resumes. A genuinely
hung-after-turn-1 agent costs at most one ceiling window and then resumes (and a real wedge
that somehow reached turn-1 will re-hang and eventually exhaust `max_execute_iterations` as
today) — an acceptable tail, paid rarely, with no false-kill of productive work.

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Flat 120-180 s transcript-idle kill at any time (brief's literal Component 2) | False-kills a single synchronous long tool call (a 15-min `cargo test`) at 120 s — regresses Signal 1's own example to *worse than* the 1200 s wall. |
| Proc-activity (CPU/IO) liveness as the primary signal | reify-4360 shows `uv` parked in `futex_wait_queue` (low CPU) while the agent worked, and a genuine wedge can also sit low-CPU; proc state is demonstrably not diagnostic. Usable only as an optional post-turn-1 refinement (Open questions). |
| Keep the flat wall; only fix classification (α) + resume (γ), no watchdog (β) | Leaves the genuine pre-turn-1 wedge burning the full 1200 s before the circuit breaker trips — fails Signal 2 ("killed fast"). β is the only component that buys the fast-wedge-kill. |
| Mandate agents background every long op (`run_in_background`) so they keep emitting poll turns | A behavioural contract agents will not reliably honour; the design must be correct when an agent runs a tool synchronously. Backgrounding already works (4415 polled and stayed alive) — but we cannot *depend* on it. |
| Compute the transcript path from the cwd-slug formula | The slug (`/`→`-`) is an undocumented CLI-internal convention that can drift across CLI versions; glob-by-session-id (the id is a unique UUID) is version-robust. |

## 4. Pre-conditions (G3 — verified on `main` `d82a1aa732` this session)

No novel substrate is introduced — every capability the components touch exists today:

- **Session id is orchestrator-known without the JSON:** `session_id_val = str(uuid.uuid4())`
  (workflow.py:6207) → `--session-id` (invoke.py:177-178 / cli_invoke.py:880-882); persisted
  pre-launch via `write_agent_session(session_id, role, started_at)` (artifacts.py:642,
  read back by `read_agent_session` :658).
- **Config dir is orchestrator-owned:** `TaskConfigDir` = `base/claude-config-<task_id>`
  (config_dir.py:36); injected as `CLAUDE_CONFIG_DIR` (invoke.py:209). `projects/` is kept
  per-task (config_dir.py:23) — the transcript root.
- **Classifier substrate:** `_parse_claude_output` empty-stdout branch (cli_invoke.py:967-976);
  `is_zero_output_timeout` (251-272); `AgentResult`/`_SubprocessResult` dataclasses (the
  `transcript_turns` field is the only new member).
- **Timeout/kill substrate:** `_run_subprocess` `asyncio.wait_for(communicate, timeout)` →
  SIGTERM(5 s flush)→`terminate_process_group` (cli_invoke.py:1142-1194); `timed_out=True`
  propagation; per-role `timeout_val` selection (workflow.py:6139); `TimeoutsConfig`
  (config.py:206-216) — the `startup_grace_secs` field is the only new member.
- **Resume substrate (Component 3 reuses it whole):** `_pending_resume_session_id` /
  `_pending_resume_role` crash-recovery lifecycle (workflow.py:6193-6205); `--resume`
  plumbing (cli_invoke.py:869-871, invoke.py:168-169); the cap-retry resume-clear guard
  (cli_invoke.py:679-690) that already keys on `is_zero_output_timeout`.
- **Circuit-breaker + evidence substrate:** `consecutive_zero_output` loop
  (workflow.py:3789-3827); `_capture_zero_output_evidence` (3922-3972, currently proc-tree
  only); `ZERO_OUTPUT_HANG_REASON` (279); `max_consecutive_zero_output_timeouts`
  (config.py:922); `steward._is_empty_output` (808-819).

**New mechanisms:** the shared transcript reader (`transcript_turns` + glob-by-session-id),
produced by **α**; the two-regime watchdog, produced by **β**; the resume-on-progress wiring,
produced by **γ**. All are wired into the existing agent-invocation path; no new dispatch
seam is introduced.

## 5. Resolved design decisions

1. **Liveness = transcript progress (an `assistant` record appearing).** Aggressive idle-kill
   is confined to the **pre-turn-1 startup window** (`startup_grace_secs`); post-turn-1 runs
   to a **generous absolute ceiling** (the existing per-role timeout). A ceiling-kill is made
   non-destructive by α (reclassify) + γ (resume). See §3 for the rejected naive-idle model.
2. **Transcript located by glob-by-session-id**, not a computed cwd-slug — version-robust.
   The session id is a unique UUID; `<config>/projects/*/<session-id>.jsonl` resolves it.
3. **`transcript_turns` is the authoritative classifier input; `None` means "could not
   read".** `is_zero_output_timeout` falls back to the legacy `turns==0 and cost_usd==0.0`
   when `transcript_turns is None`, so an unreadable transcript can never *upgrade* a wedge to
   "progress" silently — it degrades to today's behaviour. The watchdog, symmetrically, never
   early-kills on an unreadable/absent transcript (it cannot prove a wedge), falling through
   to the ceiling. Conservative in both directions.
4. **Classification lives in shared (`cli_invoke`), at/around `_run_subprocess`**, so every
   consumer (`workflow` circuit breaker, the cap-retry guard, `steward`) gets correct
   behaviour for free by routing through the predicates — no per-call-site classification.
5. **The work-killed run resumes the *same* session** (`--resume <killed id>`), not a fresh
   invocation — it continues the ~20 min of context. The orchestrator already wrote that id to
   the sidecar before launch, so it is available even though the result JSON was empty.
6. **No post-turn-1 idle heartbeat in this PRD.** Distinguishing "long synchronous tool" from
   "hung after turn-1" reliably needs proc-activity heuristics shown to be non-diagnostic
   (decision deferred to Open questions); the generous ceiling + resume covers the case
   safely without it.
7. **`startup_grace_secs` and the post-turn-1 ceiling are config, with defensible defaults**
   (120 s grace vs observed ~6 s turn-1; ceiling = today's per-role 1200 s). Exact values are
   calibration (Open questions), not architecture.
8. **No host-load / admission-control scope here.** This PRD makes work *not get killed and
   gets classified correctly*; the concurrent admission-control PRD makes work *fast*. They are
   complementary and must not be merged (§7).

## 6. Out of scope

- **Host-load governance / admission control** (the concurrent separate PRD): reducing the
  oversubscription that makes agents slow enough to hit the wall. This PRD is robust to slow
  agents; it does not make them fast.
- **A post-turn-1 idle heartbeat / proc-activity liveness signal** — deferred refinement
  (Open questions), not needed for correctness given the generous ceiling + resume.
- **An agent-emitted explicit heartbeat for long synchronous waits** — possible future
  refinement if a real workload is found that sits genuinely idle (no turns, not in a tool
  call) yet is alive; no such case is evidenced today.
- **Changing reify** — reify is a pure consumer; nothing in its repo changes.
- **Tuning `max_consecutive_zero_output_timeouts` / the circuit-breaker thresholds** — α makes
  the breaker fire only on true wedges; its threshold is left as-is.

## 7. Cross-PRD seams (G4)

| Other PRD / seam | Direction | Mechanism | Owner | Status |
|---|---|---|---|---|
| Concurrent admission-control / host-load-governance PRD (authored concurrently) | complementary (this PRD tolerates slow agents; that one makes them fast) | none shared — disjoint code (scheduler/admission vs invocation classification) | separate PRD | independent; **do not absorb load-governance scope** |
| Original from-source-build / `uv` / MCP-startup wedge fix (jcodemunch pin, plan_tools direct-interpreter, `MCP_TIMEOUT`) | consumes (this PRD preserves it; the fast pre-turn-1 kill is the safety net *behind* it) | `apply_mcp_startup_env` (invoke.py:153) — untouched | landed; unchanged | preserved |
| Intra-DF `shared/cli_invoke` ↔ `orchestrator` invocation seam | this PRD produces the contract (α) and consumes it (β, γ, δ) | `transcript_turns` field + `is_zero_output_timeout` / `is_timed_out_with_progress` predicates | this PRD (α owns) | wired by this batch |

## 8. Decomposition (G5: B+H — contract = §5 + Appendix A; boundary tests = δ)

- **α — Transcript-read classifier (Component 1, the must-have core)**
  (`shared/cli_invoke.py` + `orchestrator/steward.py` + `orchestrator/workflow.py`
  `_capture_zero_output_evidence`). Add `transcript_turns: int | None` to `_SubprocessResult`
  + `AgentResult`; a glob-by-session-id reader (`<config>/projects/*/<session-id>.jsonl`,
  count `type=="assistant"`); rewrite `is_zero_output_timeout` to
  `timed_out and (transcript_turns or 0)==0` with the legacy fallback when `None`; add
  `is_timed_out_with_progress`; `_run_subprocess` reads the transcript on the kill path and
  stamps `transcript_turns` (session id + config dir threaded in explicitly — Appendix A);
  extend `_capture_zero_output_evidence` with turn count + last tool + last ~5 records; fix
  `steward._is_empty_output` docstring/predicate to not assert "no work done" for progress
  runs. **Signal:** a timed-out result that emitted ≥ 1 assistant turn classifies
  `is_timed_out_with_progress` (not `is_zero_output_timeout`) and does **not** increment
  `consecutive_zero_output` / trip the breaker; a 0-turn timeout still classifies as a wedge;
  `zero_output_evidence-*.json` contains `transcript_turns`/`last_tool`/`last_records`; an
  unreadable transcript degrades to the legacy `turns==0 && cost==0` check. **Consumer:** β
  (live reader), γ (`is_timed_out_with_progress`), the circuit breaker / cap-retry guard /
  steward (corrected), and any human/debugger reading the evidence.
- **β — Two-regime liveness watchdog (Component 2)** (`shared/cli_invoke.py` `_run_subprocess`
  + `orchestrator/config.py` `TimeoutsConfig`). Replace the flat
  `asyncio.wait_for(timeout=role_timeout)` with a poll-the-transcript watchdog: no turn-1
  within `startup_grace_secs` → kill (fast wedge); turn-1 seen → run to the per-role ceiling;
  unchanged SIGTERM→SIGKILL-group kill path; watchdog stamps `transcript_turns` from live
  tracking. Add `TimeoutsConfig.startup_grace_secs` (default 120.0). **Signal:** a 0-turn
  wedge is SIGKILLed within ~`startup_grace_secs` (not the 1200 s ceiling); a turn-emitting
  agent survives past `startup_grace_secs`; an agent that emits turn-1 then makes a single
  synchronous tool call longer than `startup_grace_secs` is **not** killed at the grace bound.
  **Consumer:** the agent-invocation path (every task); δ. **Depends:** α.
- **γ — Resume across the wall (Component 3)** (`orchestrator/workflow.py` `_execute_iterations`).
  On an `is_timed_out_with_progress` implementer result, set `_pending_resume_session_id` =
  the killed session id + `_pending_resume_role = IMPLEMENTER` so the next iteration's
  `_invoke` resumes via `--resume`; confirm the cap-retry resume-clear guard
  (cli_invoke.py:679-690) does not fire for this class (it keys on `is_zero_output_timeout`,
  already False after α). **Signal:** after a work-killed implementer iteration, the next
  `_invoke` is called with `resume_session_id == the killed session id`; the journal logs
  `resuming prior session <id>` and `.task/agent_session.json` reflects it; the iteration
  continues rather than restarting. **Consumer:** δ, the "long task completes" surface.
  **Depends:** α.
- **δ — End-to-end liveness boundary gate (B+H integration tests)** (tests spanning
  `shared` + `orchestrator`). The Appendix B boundary table. **Signal:** all rows green;
  specifically **B6** (long synchronous tool after turn-1 not false-killed) and **B3** (resume
  re-dispatch carries the same session id) exercise the integrated path; **B7** (unreadable
  transcript → legacy fallback, no early-kill) demonstrates the conservative degrade.
  **Consumer:** the user-observable outcome — a long-legit task completes and the
  proc-tree-only misdiagnosis cannot recur. **Depends:** α, β, γ.

**DAG:** α → β; α → γ; β ∥ γ (disjoint files: β = `cli_invoke.py`+`config.py`, γ =
`workflow.py`); {β, γ} → δ. (α must precede both — it co-edits `cli_invoke.py` with β and
`workflow.py` with γ.)

## 9. Open questions (tactical — deferred, not blocking)

1. **Exact `startup_grace_secs`.** Default 120 s (≫ observed ~6 s turn-1). Decide during β;
   a calibration follow-up may lower it once real wedge-vs-startup latencies are measured.
2. **Post-turn-1 ceiling value.** Keep the per-role 1200 s, or raise it now that ceiling-kills
   resume? Default: keep 1200 s. Decide during β.
3. **Post-turn-1 idle-heartbeat refinement** (proc-activity-gated): worth adding later to kill
   a genuinely-hung-after-turn-1 agent faster than the ceiling? Needs a reliable
   alive-but-silent signal first (proc-activity shown non-diagnostic). File as a follow-up if a
   real after-turn-1 hang is observed.
4. **Resume budget interaction.** Should a work-killed→resume cycle count against any
   per-task iteration/cost cap, or is `max_execute_iterations` sufficient backstop? Default:
   rely on `max_execute_iterations` (a resume that keeps getting killed still exhausts it).
   Decide during γ.
5. **Does `transcript_turns` belong on the success path too** (not just timeouts), e.g. for
   richer telemetry? Out of scope for correctness; α populates it only where it drives a
   decision (the timeout/kill path).

---

## Appendix A — Contract (B+H): the `shared` ↔ `orchestrator` seam

**Transcript reader (shared, produced by α):**
```
def count_transcript_turns(config_dir: Path, cwd: Path, session_id: str) -> int | None:
    """Count `type == "assistant"` records in the run's transcript.

    Resolves the transcript by glob-by-session-id:
        <config_dir>/projects/*/<session_id>.jsonl
    Returns the count, or None when the transcript cannot be located/read
    (caller must treat None conservatively — never as a proven wedge).
    """
```
- `cwd` is accepted for forward-compatibility/diagnostics; resolution is by session id glob,
  so a drifting cwd-slug convention does not break it.
- Best-effort: any I/O / JSON error → `None`, logged, never raised (mirrors
  `_capture_zero_output_evidence`'s best-effort contract).

**Result field (shared, produced by α):** `_SubprocessResult.transcript_turns: int | None`
and `AgentResult.transcript_turns: int | None`, propagated by `_parse_claude_output` on every
return path (default `None`). Populated on the timeout/kill path: by α's post-kill read, and
once β lands, by the watchdog's live tracking.

**Predicates (shared, produced by α; the seam every consumer routes through):**
- `is_zero_output_timeout(r) := r.timed_out and ((r.transcript_turns == 0) if r.transcript_turns is not None else (r.turns == 0 and r.cost_usd == 0.0))`
- `is_timed_out_with_progress(r) := r.timed_out and (r.transcript_turns or 0) > 0`
- Invariant: the two are mutually exclusive, and exactly one holds when `r.timed_out` and
  `transcript_turns is not None`.

**Watchdog timeout signature (shared, β):** `_run_subprocess` gains the run's `session_id`
and `config_dir` (threaded explicitly from `invoke_claude_agent` / the sandbox path in
invoke.py — both already hold them) so the watchdog can poll the transcript live; the
SIGTERM→SIGKILL-group kill path and `timed_out=True` semantics are unchanged. Liveness
contract: kill at `startup_grace_secs` iff zero `assistant` records observed; otherwise kill
at the per-role ceiling.

**Resume contract (orchestrator, γ):** on `is_timed_out_with_progress(result)` for the
IMPLEMENTER role, `self._pending_resume_session_id = <session id used for the killed run>`
and `self._pending_resume_role = IMPLEMENTER`; the existing `_invoke` lifecycle
(workflow.py:6193-6205) then resumes. The cap-retry resume-clear (cli_invoke.py:679-690) is
gated by construction (its `is_zero_output_timeout` condition is False for this class).

## Appendix B — Boundary-test sketch (B+H; δ's observable signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Work-killed run reclassified | a run with `timed_out=True` and a transcript of ≥ 1 `assistant` record | `is_timed_out_with_progress` True; `is_zero_output_timeout` False; `consecutive_zero_output` not incremented; breaker not tripped |
| B2 | Genuine pre-turn-1 wedge | `timed_out=True`, transcript has 0 `assistant` records | killed within ~`startup_grace_secs` (β), not the ceiling; `is_zero_output_timeout` True; breaker increments |
| B3 | Resume across the wall | a work-killed IMPLEMENTER iteration | next `_invoke` receives `resume_session_id == killed session id`; `_pending_resume_role == IMPLEMENTER`; journal `resuming prior session <id>` |
| B4 | Cap-retry guard gated | a `TIMED_OUT_WITH_PROGRESS` result with `resume_session_id` set | `_reset_for_fresh_retry` NOT called; resume preserved |
| B5 | Evidence carries transcript | any `zero_output_evidence-*.json` written | JSON contains `transcript_turns`, `last_tool`, `last_records` (not proc_tree only) |
| B6 | Long synchronous tool not false-killed | agent emits turn-1, then a single synchronous tool call longer than `startup_grace_secs` (transcript silent during it) | not killed at the grace bound; survives into the working regime to the ceiling |
| B7 | Unreadable transcript → conservative | `timed_out=True`, transcript path absent/unreadable (`transcript_turns is None`) | classification falls back to legacy `turns==0 && cost_usd==0.0`; β does not early-kill (cannot prove a wedge) |
