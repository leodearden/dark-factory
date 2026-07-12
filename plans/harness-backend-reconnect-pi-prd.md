# PRD: Multi-backend agent dispatch reconnect + codex hardening + pi backend + per-role env + CostStore role telemetry

**Date:** 2026-07-12 · **Status:** approved for decomposition · **Origin:** decision brief
`~/.claude/spawn-briefs/2026-07-12-harness-reconnect-pi.md` (interactive assessment session,
Leo + Claude, 2026-07-12; G1–G6 pre-answered there). **Scope:** all load-bearing code is
dark-factory `orchestrator/` + `shared/`; Claude Code stays the production default and the
`claude` backend path stays byte-identical. **Approach:** B+H (mechanism count ≥ 8; the core
agent-invocation seam `invoke_with_cap_retry → invoke_fn` is touched — the reconnect contract +
its two-way boundary test are Appendix A / task T1).

Cite by symbol; line refs are as-of `main` `7b6c1f829c` and drift.

## 1. Consumer + user-observable surface (G1, G2)

**Consumer (which surfaces consume the mechanisms this PRD introduces):**
- The **eval-revival** PRD (`~/.claude/spawn-briefs/2026-07-12-eval-revival.md`) — its candidate
  bundles (Codex CLI + GPT-5.6 Sol for the Rust implementer; pi+GLM / pi+MiniMax; pi+Sonnet
  harness-isolating control; MiniMax/GLM/DeepSeek/Kimi via official Anthropic-format endpoints)
  are evaluated only once backends actually dispatch (T1), non-claude backends are trialable
  (T3/T4), the price table exists (T-price), and per-role endpoints route (T5). Eval-revival
  declares plain integer deps on this batch's task ids.
- The **production orchestrator config** (`config.backends.<role>` in `defaults.yaml` +
  `BackendsConfig` config.py:255) — dead config today; live after T1. An operator sets a role's
  backend to `codex`/`pi` and it takes effect.
- The **CostStore consumer** (`data/orchestrator/runs.db`, the digest cost roll-ups
  harness.py:6249 / digest.py:387) — gains complete per-role economics after T6.

**User-observable surface (what an operator/eval sees after this PRD lands):**
1. With `config.backends.<role> = codex` (or `pi`), a real role invocation **in eval mode** runs
   on that backend end-to-end and returns a correct `AgentResult` (success, cost, tokens, turns,
   session id) visible in the eval report. With backends left at `claude`, dispatch is
   **byte-identical to today** (T1's boundary test asserts this invariant).
2. A codex-backend implementer run leaves **no `AGENTS.md`** in its committed diff; its recorded
   cost is derived from a **config price table**, not the silent `$2/$8` default; a configured
   budget/turn ceiling is honoured or its native-unsupported status is documented and the
   wall-clock watchdog covers it (T3).
3. A committed **pi-spike findings document** answers the seven empirical questions (exit-code
   semantics, cost aggregation, tool-name format, verdict-tool compliance, liveness signal,
   sandbox wrap, turn/budget cap) with observed evidence — before any pi backend code is written
   (T2 → T4).
4. An operator can assign `api.z.ai/api/anthropic` (or Kimi/MiniMax/DeepSeek Anthropic-format
   endpoints) to a **specific role** via `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` with no
   harness change, and the judge is **not** dragged onto that endpoint unless the judge role is
   explicitly configured (T5 — opt-in per role, preserving the vLLM-burn safety).
5. After a task run, `runs.db` contains cost rows for `role='judge'`, `role='steward'`, and
   `role='triage'` (currently absent) — role economics are complete (T6).

## 2. Premise validation (G6 — verified archaeology, re-checked on `main 7b6c1f829c`)

**The multi-backend abstraction exists and was never deleted; it is severed at one wire.**

1. **Dispatcher exists.** `invoke_agent` (invoke.py:52-123) accepts `backend: str = 'claude'`
   and dispatches to `_invoke_claude_with_sandbox` / `_invoke_codex` (invoke.py:259) /
   `_invoke_gemini` (invoke.py:433), else `raise ValueError(f'Unknown backend: {backend!r}')`
   (invoke.py:123). Added by `d3b14de810` (2026-03-19); codex shelved "pending integration rework."
2. **The wire is cut.** `invoke_with_cap_retry` (cli_invoke.py:700) takes `backend='claude'`
   (cli_invoke.py:715) and consumes it **only** for cap classification —
   `classify_invocation(result, ..., backend=backend)` (cli_invoke.py:936) and
   `slot.detect_cap_hit(..., backend=backend)` (cli_invoke.py:1007) — and **never writes it into
   `invoke_kwargs`** before calling `invoke = invoke_fn or invoke_claude_agent` (cli_invoke.py:884)
   → `await invoke(**invoke_kwargs)` (cli_invoke.py:896, 917). So even when the orchestrator passes
   `invoke_fn=invoke_agent` **with** `backend='codex'`, `invoke_agent` runs with its own default
   `backend='claude'`. The defect is documented in prose at `steward.py:556-561` (W4-eta
   amendment) — "invoke_agent always runs with its own default (backend='claude')" — but **no
   follow-up task was ever filed**.
3. **The whole `backends:` block is dead config.** `defaults.yaml:210-220` and `BackendsConfig`
   (config.py:255-267, hot-reloadable) are read by `_invoke` (workflow.py:7349
   `backends_cfg = self.config.backends`; passed `backend=backend_val` at workflow.py:7456) and
   the steward (steward.py:575 `backend=self.config.backends.steward`) — every value flows into
   `invoke_with_cap_retry`'s `backend` param and dies there.
4. **Codex path is un-hardened** (verified):
   - `AGENTS.md` is written **into the task worktree** `cwd` (invoke.py:274-276) and unlinked only
     in `finally` (invoke.py:304-306); the implementer commit path stages with `git add -A`
     (git_ops.py:591), which excludes only `.task/` (`:!.task/` pathspec + root `.gitignore` :40)
     — so `AGENTS.md` is stage-eligible.
   - `_invoke_codex` (invoke.py:259-306) passes **no `max_turns`** and does not enforce
     `max_budget_usd` (only forwards it to `_run_subprocess_local`); the claude caps are dropped.
   - Cost is a hardcoded 4-entry `_MODEL_COSTS` (invoke.py:42-49) with a silent
     `{'input': 2.0, 'output': 8.0}` default (invoke.py:393) — wrong for any un-listed model.
   - `.codex/config.toml` MCP serializer already exists (`_write_codex_mcp_config` invoke.py:409).
5. **Env forwarding is implementer/debugger-only.** `_build_agent_env` (workflow.py:7304-7333)
   returns `None` for every role except architect/implementer/debugger, and only
   implementer/debugger receive `config.env_overrides` (workflow.py:7327). The global forwarding
   was **narrowed away from the judge** after a vLLM `ServerDisconnectedError` burn
   (`3cd380a079`; comment workflow.py:7470-7474). `config.env_overrides` is a **global**
   `dict[str,str]` (config.py:2505) — there is no per-role env map today.
6. **CostStore role telemetry gap** (empirically verified against `runs.db` by the brief).
   `save_invocation` fires in exactly three places: `_invoke` (workflow.py:7538, the shared
   chokepoint — records architect/implementer/debugger/reviewer/merger), `review_checkpoint.py:218`,
   and inside `invoke_with_cap_retry` **guarded by `if cost_store:`** (cli_invoke.py:1177-1195).
   The judge (`_run_completion_judge`, workflow.py:4776), steward (steward.py:570), and
   steward pre-triage/triage (steward.py:612) call `invoke_with_cap_retry` **without passing
   `cost_store=`** → the guard is skipped → these roles never land a `runs.db` row. Context:
   architect+implementer+debugger ≈ **80–83%** of spend on both projects since June 1
   (~$27k API-equivalent); the missing roles hide the remainder.

No premise here asserts an impossible number/exactness. Every capability the tasks touch is
present on `main` today (§4). The pi *runtime* contracts T4 assumes are the one exception — they
are **not** verified on main; they are queued as an explicit prerequisite (T2, the spike), which
is exactly what G3 requires.

## 3. Approach

Eight tasks. One root **enabler** (T1) unblocks every non-claude backend; one root **spike** (T2)
de-risks pi before any pi code is written; one root **price table** (T-price) is the shared
cost-accounting substrate (also consumed by eval-revival); the rest fan out.

- **T1 — Reconnect (the enabler, `shared/cli_invoke.py`).** Forward `backend` from
  `invoke_with_cap_retry` into the dispatched `invoke_fn` call — **only when a custom `invoke_fn`
  is supplied** (the multi-backend `invoke_agent`); the default `invoke_claude_agent` path (used by
  fused-memory recon) is untouched and never sees a `backend` kwarg. Non-claude backends may bypass
  the claude-specific cap-failover / session-resume machinery for now (single-account acceptable) —
  `classify_invocation` / `detect_cap_hit` already branch on `backend`. **Boundary test (B+H):** a
  spy `invoke_fn` receives `backend='codex'` when set, and `backend='claude'` (default) preserves
  today's call shape byte-for-byte.
- **T2 — pi empirical spike (`plans/` findings doc + throwaway `scripts/` probe; FIRST, no code
  deps).** Pin `@earendil-works/pi-coding-agent` v0.80.x + `pi-mcp-adapter` v2.11.x; run pi headless
  and record observed answers to the seven questions in §Appendix B. Output: a committed
  `plans/pi-spike-findings.md`. This is the G3 verification of pi's runtime substrate; T4 consumes
  it.
- **T-price — Config-driven model price table (`config.py` + `defaults.yaml` + `invoke.py`).**
  Replace the hardcoded `_MODEL_COSTS` (invoke.py:42-49) with a config `prices` map
  (`model → {input_per_1m, output_per_1m}`); cost estimation for backends without native cost
  reporting reads it, with an explicit (logged) fallback rather than a silent `$2/$8`. **Shared
  seam:** eval-revival's "per-config price table for cloud endpoints" consumes this owner (G4).
- **T3 — Codex hardening to trialable (`invoke.py` + `config.py` + `defaults.yaml`; deps: T1,
  T-price).** (a) Guarantee `AGENTS.md` never enters the committed diff — write the instruction
  file **outside** the task worktree (or exclude it from staging); (b) enforce budget/turn ceilings
  on the codex path, or document the native-unsupported ones and rely on the wall-clock watchdog
  (parity with pi); (c) codex cost accounting reads T-price. Structured output for codex verdict
  roles is **not** in scope here — it is owned by mcp-verdict-servers (G4).
- **T4 — pi backend (`invoke.py` + `roles.py` + `config.py` + `defaults.yaml`; deps: T1, T2,
  T-price).** `_invoke_pi` + `_parse_pi_output → AgentResult`; `--tools` allowlist built from the
  role's allowed/disallowed lists (roles.py) + MCP tool names, using the tool-name format the spike
  observed; per-task session resume via `--session`/`--session-dir`; provider/model/thinking mapping
  from role config (pi natively supports Anthropic incl. Claude OAuth, zai/GLM, Kimi, MiniMax,
  DeepSeek, OpenRouter, custom base URLs); add `'pi'` to the `BackendsConfig` docstring +
  `invoke_agent` dispatch (`elif backend == 'pi'`). Cost aggregation and liveness-watchdog wiring
  follow the spike's findings.
- **T5 — Per-role env forwarding (`config.py` + `workflow.py` + `defaults.yaml`; no deps).** Add a
  per-role env map (`role_env_overrides: dict[str, dict[str,str]]`, default empty); widen
  `_build_agent_env` to merge the per-role entry for **every** role (not just
  architect/implementer/debugger). The judge/vLLM burn is avoided **by construction**: forwarding is
  now **opt-in per role**, so the judge receives an endpoint env only if an operator sets one for
  the judge role. Eval-revival's Claude-endpoint candidates consume this (G4).
- **T6 — CostStore role telemetry (`workflow.py` + `steward.py`; no deps).** Thread
  `cost_store=self.cost_store` (+ `run_id`/`task_id`/`project_id`/`role`) into every role-invocation
  path that currently omits it — the judge `_run_completion_judge`, the steward main invoke, and
  the steward pre-triage/triage invoke — so the `if cost_store:` guard (cli_invoke.py:1177) fires
  for those roles. **Signal:** after a run, `runs.db` has rows for `role in {judge, steward,
  triage}`.
- **T-dedupe — Claude-argv dedupe (`invoke.py`; dep: T1; blocks nothing; LOW priority hygiene).**
  Extract one `build_claude_argv(...)` shared by the sandbox fork (invoke.py:174-208) and the
  non-sandbox builder (cli_invoke.py claude-argv assembly) so the two cannot drift. **Signal:** a
  parity test asserts both paths produce identical argv (modulo the sandbox wrapper prefix).

### The load-bearing design decision: per-role env is opt-in, which *is* the vLLM-burn fix

The judge was excluded from env forwarding (workflow.py:7470-7474) because the **global**
`config.env_overrides` dragged every enrolled role — including the judge, which must hit the real
Claude API — onto a vLLM endpoint that `ServerDisconnectedError`'d after two tool rounds. T5 does
**not** re-globalise env; it makes forwarding a **per-role opt-in map**. The judge gets an endpoint
env iff `role_env_overrides['judge']` is set. So the safety property that motivated the exclusion is
preserved without a special-case: no role is routed off-Claude unless an operator names it. This is
why T5 removes the role allow-list gate in `_build_agent_env` rather than adding the judge to it.

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Forward `backend` unconditionally into `invoke_kwargs` (T1) | Breaks the default `invoke_claude_agent(**invoke_kwargs)` path (recon/curator) — it has no `backend` param. Forward only when a custom `invoke_fn` is supplied. |
| Re-globalise env forwarding for all roles (drop the judge exclusion by widening `config.env_overrides`) | Reintroduces the exact vLLM `ServerDisconnectedError` burn (3cd380a079). Per-role opt-in is the only widening that is safe by construction. |
| Fold the price table into T3 (codex) | eval-revival also needs it; a shared owner (T-price) with a clean seam beats duplicating the table or coupling eval's timeline to codex hardening. |
| Write the pi backend (T4) from the brief's research without the spike (T2) | pi's exit-code semantics, cost-aggregation source, tool-name format, and liveness signal are **undocumented**; guessing them produces a fake-done backend (G2 hazard). The spike verifies them first. |
| Give non-claude backends full cap-failover / multi-account resume parity now | Out of scope for the pilot (single-account acceptable, brief §Non-goals); the cap-retry loop tolerates non-claude by branching on `backend` already. |
| Exclude `AGENTS.md` by adding it to the merge-lane `:!.task` pathspec | The clean invariant is "the instruction file never lives in the worktree at commit time" — relocate it outside `cwd` (robust to any staging command) rather than growing a per-file blocklist. |

## 4. Pre-conditions (G3 — verified on `main 7b6c1f829c` this session)

No novel dark-factory substrate is introduced; every capability the tasks touch exists today.

- **Dispatch seam:** `invoke_agent(..., backend='claude', ...)` with codex/gemini branches +
  `Unknown backend` guard (invoke.py:52-123); `invoke_with_cap_retry(..., invoke_fn, backend, **invoke_kwargs)`
  (cli_invoke.py:700-716); `invoke = invoke_fn or invoke_claude_agent` (cli_invoke.py:884);
  `await invoke(**invoke_kwargs)` (cli_invoke.py:896, 917). — T1.
- **Config surface:** `BackendsConfig` (config.py:255, hot-reloadable; leaf paths config.py:2806);
  `backends:` (defaults.yaml:210); `config.env_overrides` global map (config.py:2505). — T1, T5.
- **Codex path:** `_invoke_codex` (invoke.py:259-306); `AGENTS.md` write (invoke.py:274-276) +
  `finally` unlink (:304-306); `_MODEL_COSTS` (invoke.py:42-49) + `$2/$8` fallback (invoke.py:393);
  `_parse_codex_output` (invoke.py:309-406); `_write_codex_mcp_config` (invoke.py:409);
  `_run_subprocess_local` (invoke.py:561). — T3, T-price.
- **Staging invariant to protect:** implementer commit `git add -A` (git_ops.py:591) excludes only
  `.task/` (`:!.task/` git_ops.py:5150,5216; root `.gitignore` :40). — T3.
- **Claude-argv duplication to dedupe:** sandbox fork (invoke.py:174-208) vs the non-sandbox
  builder in cli_invoke.py. — T-dedupe.
- **Env-build seam:** `_build_agent_env(role)` role allow-list + `config.env_overrides` merge
  (workflow.py:7304-7333); consumed at `_invoke` `env_overrides=self._build_agent_env(role)`
  (workflow.py:7475). — T5.
- **CostStore seam:** `CostStore.save_invocation` (cost_store.py:102); the guarded call inside
  `invoke_with_cap_retry` (cli_invoke.py:1177-1195); the recording chokepoint in `_invoke`
  (workflow.py:7538); the omitting call-sites — judge `_run_completion_judge` (workflow.py:4776),
  steward (steward.py:570), pre-triage/triage (steward.py:612). — T6.

**pi runtime substrate (the sole not-on-main dependency):** pi's headless exit codes, cost-event
model, direct-tools name format, verdict-tool convention, session-JSONL liveness path, and
turn/budget-cap surface are **verified by T2 (the spike)**, whose findings gate T4. This is the
explicit-prerequisite branch of G3, not an unverified assumption.

## 5. Resolved design decisions

1. **`backend` is forwarded only when `invoke_fn` is supplied.** The default `invoke_claude_agent`
   path keeps its exact signature; the multi-backend `invoke_agent` is the only consumer of the
   forwarded kwarg. (Tactically: `if invoke_fn is not None: invoke_kwargs.setdefault('backend', backend)`.)
2. **Backends stay at `claude` by default; the `claude` dispatch path is byte-identical.** T1's
   boundary test is the guardrail; any behavioural change to the claude path is a regression.
3. **Non-claude backends run single-account, no cap-failover/session-resume parity this round.**
   The cap-retry loop already tolerates this by branching on `backend`.
4. **The pi spike precedes and gates the pi backend.** T4 hard-depends on T2; T4's tool-name
   mapping, cost aggregation, and liveness wiring are derived from observed spike evidence, not
   guessed.
5. **The model price table is config, owned here, shared with eval-revival.** `prices:
   {model: {input_per_1m, output_per_1m}}`; an un-listed model logs a warning and falls back —
   never a silent default. (T-price.)
6. **Per-role env is an opt-in map; forwarding is widened to all roles.** `role_env_overrides`
   (default empty) merged per-role in `_build_agent_env`. The judge is safe because it is off unless
   named (§3). (T5.)
7. **`AGENTS.md` must not be in the worktree at commit time** — relocate the instruction file
   outside `cwd`, not blocklist it. (T3.)
8. **CostStore records every role.** Threading `cost_store` into the judge/steward/triage paths is
   the whole fix; no schema change (`save_invocation` already accepts `role`). (T6.)
9. **Codex/pi trialability of *verdict-emitting* roles (reviewer/judge/triage/merger) gates on
   mcp-verdict-servers, not on this PRD.** This PRD makes the backends dispatch and run
   implementer/architect roles (git-commit + plan-tools MCP contracts, already portable). The
   verdict-role gate lives in eval-revival's trial tasks (G4).

## 6. Out of scope

- **Gemini hardening** — keep the code, invest nothing this round (brief §Non-goals).
- **Multi-account cap-failover parity for non-claude backends** — single-account pilot.
- **Local/vLLM serving** — cloud APIs only this round (Leo).
- **Structured-output replacement (MCP verdict tools)** — owned by mcp-verdict-servers; this PRD
  does not migrate `--json-schema`.
- **The eval task-set refresh / eval-mode profile / eval report** — owned by eval-revival; this PRD
  only makes the bundles dispatchable and priced.
- **The `~/.claude` sandbox-writable observation** (sandbox.py:96-98, landlock_exec.py:154) — noted
  for a future security pass; not a fix in this batch (Open questions §9.5).
- **Sandbox/bwrap changes** beyond confirming pi wraps like codex (single-process, children inherit
  the namespace) — the spike observes it; no new sandbox mechanism.

## 7. Cross-PRD seams (G4)

| Other PRD / seam | Direction | Mechanism | Owner | Status |
|---|---|---|---|---|
| **eval-revival** (concurrent) | eval **consumes** this PRD | its codex/pi bundles gate on T1+T3+T4; its Claude-endpoint bundles need only T5; its price table is T-price | **this PRD** owns T1/T3/T4/T5/T-price; eval declares plain integer deps on those task ids | independent; eval wires deps after this batch is filed |
| **mcp-verdict-servers** (concurrent) | this PRD **consumes** it for verdict-role trials only | codex/pi have no `--json-schema`; verdict roles need MCP verdict tools | **mcp-verdict-servers** owns the contract; the gate sits in **eval-revival's** verdict-role trial tasks, NOT in this batch | this batch does **not** depend on mcp-verdict — implementer/architect roles are portable already |
| **Per-harness liveness/progress signal** | owned **here** | pi session-JSONL watchdog analog of the Claude transcript glob (`_run_subprocess` cli_invoke.py) | **this PRD** (T2 observes it, T4 wires it) | wired by T4 per T2 findings |
| Intra-DF `shared/cli_invoke` ↔ `orchestrator/agents/invoke` dispatch seam | this PRD produces (T1) + consumes (T3, T4) | `backend` forwarded through `invoke_fn`; Appendix A contract | **this PRD** (T1 owns) | wired + boundary-tested by this batch |

**Seam mechanics:** all three PRDs are `dark_factory` tasks → sibling deps are **plain integer**
`add_dependency`, not the cross-project `project_id:task_id` form (which is for reify↔dark_factory).
The consuming siblings (eval-revival, mcp-verdict-servers) wire their own deps onto **this batch's**
task ids; this batch depends on **neither** sibling. The batch's task ids are surfaced in the
hand-back.

## 8. Decomposition (G5: B+H — contract = Appendix A; two-way boundary test = T1)

DAG (ids are batch-local labels; real fused-memory ids assigned at filing):

```
T1  (reconnect, cli_invoke.py)            ── root enabler
T2  (pi spike, plans/scripts)             ── root (no code deps); FIRST/high
T-price (price table, config/invoke)      ── root
T5  (per-role env, config/workflow)       ── root
T6  (CostStore telemetry, workflow/steward)── root

T3  (codex harden)  ← T1, T-price
T4  (pi backend)    ← T1, T2, T-price
T-dedupe (argv)     ← T1        (blocks nothing; LOW)
```

Parallelism note: T1/T2/T-price/T5/T6 are independent roots and may dispatch concurrently. T3, T4,
T-price, T-dedupe all edit `orchestrator/agents/invoke.py`; T-price, T3, T4, T5 all edit
`config.py` — the narrow-file locks will **serialize** these regardless of the DAG, which is
correct (no integration is starved because every task is filed together with explicit deps). The
DAG edges encode semantic ordering (enabler/spike/price before their consumers), not a parallelism
claim.

Each task's user-observable signal and consumer are in §1 / §3 and restated in the capability
manifest (committed beside this PRD).

## 9. Open questions (tactical — deferred, not blocking; no open *design* questions remain)

1. **T1 forwarding mechanism** — `setdefault('backend', ...)` when `invoke_fn` is set, vs teaching
   `invoke_claude_agent` to accept-and-ignore `backend`. Recommend the former (no signature change
   to the recon path). Decide in T1.
2. **T3 `AGENTS.md` relocation** — write to a temp dir + point codex at it (if codex supports an
   instruction-file-path flag; verify in T3) vs write outside the worktree root. Either satisfies
   "never in the committed diff." Decide in T3.
3. **T3 codex turn/budget cap** — codex CLI has no obvious `--max-turns`; enforce budget via a `-c`
   config knob if one exists, else document native-unsupported and rely on the wall-clock watchdog
   (parity with pi). Decide in T3 from codex CLI capability.
4. **T4 pi turn/budget cap** — brief §3: a tiny pi extension via `agent_end`/turn hooks vs
   external-watchdog-only for the pilot. Recommend external-only for the pilot; revisit if a real
   runaway is observed. Decide in T4 from spike findings.
5. **`~/.claude` sandbox-writable** (sandbox.py:96-98, landlock_exec.py:154) — a follow-up security
   task, not this batch. File separately if the eval trials surface a concrete risk.
6. **DeepSeek legacy model-name retirement 2026-07-24** — an eval-config timing note (brief); the
   T5 mechanism is name-agnostic, so no code impact here.

---

## Appendix A — Contract (B+H): the `shared/cli_invoke` ↔ `orchestrator/agents/invoke` seam

**Reconnect contract (produced by T1, consumed by T3/T4):**
```
invoke_with_cap_retry(..., invoke_fn=invoke_agent, backend=<b>, **invoke_kwargs)
  ⇒ dispatches invoke_agent(**invoke_kwargs, backend=<b>)          # when invoke_fn is not None
invoke_with_cap_retry(..., backend='claude', **invoke_kwargs)      # invoke_fn is None (default)
  ⇒ dispatches invoke_claude_agent(**invoke_kwargs)                # NO backend kwarg — unchanged
```
- Invariant 1 (byte-identical claude default): with `backend='claude'` (or unset) and no
  `invoke_fn`, the call shape to `invoke_claude_agent` is identical to today.
- Invariant 2 (forwarding): with `invoke_fn=<spy>` and `backend='codex'`, the spy is called with
  `backend='codex'`.
- Invariant 3 (recon-path safety): `invoke_claude_agent` never receives a `backend` kwarg.

**Two-way boundary test (T1 authors; the B+H artifact):** a spy `invoke_fn` capturing kwargs
asserts Invariant 2; a claude-default path (no `invoke_fn`) asserts Invariant 1 + 3.

**Price-table contract (produced by T-price, consumed by T3/T4 + eval-revival):**
`config.prices: dict[str, {input_per_1m: float, output_per_1m: float}]`; cost estimation for a
backend without native cost = `(in_tok*input_per_1m + out_tok*output_per_1m)/1e6`; un-listed model
⇒ logged warning + defined fallback (not a silent `$2/$8`).

**Per-role env contract (produced by T5, consumed by eval-revival):**
`config.role_env_overrides: dict[role_name, dict[str,str]]` (default `{}`); `_build_agent_env(role)`
merges `role_env_overrides.get(role.name, {})` for **every** role; a role absent from the map
receives no endpoint env (judge safety by construction).

**CostStore contract (produced by T6):** every role that invokes an agent lands a
`save_invocation(role=<role>, ...)` row in `runs.db`; specifically `role in {judge, steward,
triage}` are present after a run (currently absent).

## Appendix B — pi spike question set (T2's observable signal: all seven answered with evidence)

Pin `@earendil-works/pi-coding-agent` v0.80.x + `pi-mcp-adapter` v2.11.x (maintainer nicobailon,
MIT; direct-tools mode `toolPrefix:"server"`; reads the same `.mcp.json` Claude Code uses; stdio +
streamable-HTTP → covers fused-memory + escalation HTTP servers and plan-tools stdio).

1. **Headless exit-code semantics** (`--mode json` / `-p`) — success vs error vs timeout mapping.
2. **Cost aggregation** — no aggregate terminal result event; choose among summing per-message
   `Usage` events / parsing session JSONL / RPC `get_session_stats`.
3. **Direct-tools generated name format** — normalization/separator — before writing `--tools`
   allowlists.
4. **Verdict-tool compliance rate + absent-report fallback** — structured output on pi is a
   convention (`terminate:true` tool), not decode-forced.
5. **Liveness signal** — append-only session JSONL under `~/.pi/agent/sessions/...` (or
   `--session-dir`) as the watchdog analog of the Claude transcript glob (cli_invoke.py `_run_subprocess`).
6. **bwrap/landlock wrap** — pi is single-process; bash children inherit the namespace (pi
   first-party example uses Anthropic sandbox-runtime/bubblewrap).
7. **Turn/budget cap** — no documented `--max-turns`/budget flag; decide tiny-extension
   (`agent_end`/turn hooks) vs external-watchdog-only for the pilot.

Findings dated 2026-07-12 (sources: pi.dev/docs/latest usage/json/rpc/providers/extensions/
containerization; github.com/earendil-works/pi v0.80.6; github.com/nicobailon/pi-mcp-adapter
v2.11.0). Single-maintainer risk on pi-mcp-adapter accepted for a pilot (MIT, standard MCP SDK,
forkable).
