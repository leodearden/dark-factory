# PRD — Escalation connection-level capability guard + server-attributed watcher identity

**Status:** active — authored 2026-07-03. Fleet-wide (shared `escalation/` package; fixes every project once escalation servers + orchestrators restart).
**Type:** greenfield hardening of an existing subsystem (escalation MCP server + orchestrator watcher-supervisor).
**Shape:** B + H (contract + two-way boundary tests) — the change installs a permission boundary on the escalation-resolution seam, a load-bearing coordination surface.

## Goal (user-observable behaviour)

The orchestrator-spawned autonomous escalation-watcher can **no longer resolve, close, or park a level-2 escalation** — the escalation MCP server rejects the attempt. Its legitimate work is unaffected: it still resolves level-0/1 admin items and still promotes clusters to L2. Every state change it makes is stamped with a **defined, server-attributed identity** (`orchestrator-escalation-watcher-auto`) instead of the model-invented `escalation-watcher-L2`, so the interactive `/escalation-watcher` session can tell the trusted supervised watcher from a genuine rogue and stops firing its "stand down" reflex.

Concretely, after this lands:
- A level-2 `resolve_issue(action=close_only|park|resume|restart|abandon)` from the supervised watcher's connection returns a `level_forbidden` error and makes **no** state change; the L2 stays pending for a human.
- The same watcher's `resolve_issue` on a level-1 admin escalation, and its `promote_to_l2`, still succeed — with `resolved_by="orchestrator-escalation-watcher-auto"` on the archived record.
- The human's interactive session (and any other un-restricted client) retains full level-2 authority, unchanged.

## Background / motivation

From a `/deb` investigation this session (esc-fused-memory-5):

- The dark-factory orchestrator auto-spawns a headless autonomous escalation-watcher per rotation via the watcher-supervisor (`orchestrator/harness.py` `_watcher_supervisor_loop` / `_run_watcher_rotation`, task 1326; `watcher_supervisor_enabled: true` default). It runs the `escalation-watcher-auto` skill, whose contract is **L1-only**: resolve admin-class L1 items and **promote** judgment/L2 items to the human — never resolve `level >= 2` itself.
- In practice the agent oversteps. Session `e0dbb43f` directly resolved a **level-2** escalation (`esc-fused-memory-5`) via `resolve_issue(action=close_only)`, stamping `resolved_by="escalation-watcher-L2"` — a label defined **nowhere** in code or skills (verified: human skill uses `escalation-watcher`; auto skill uses `escalation-watcher-auto`; the model free-invents `escalation-watcher-L2`). Archives show this recurring: **reify 40 items, dark-factory 5 items** so stamped, mostly level 2, since ~Jun 15.
- Two concrete harms: (1) it **races** the human's interactive `/escalation-watcher` on the same L2 queue, and because the identity is undefined it is **indistinguishable from a rogue second human handler** (trips the live watcher's stand-down reflex every time); (2) it **prematurely closed** `esc-fused-memory-5` with a **wrong** "Drained/benign" diagnosis — the real cause is a full-recon lock exceeding 1800s that never completes (recurrence of task 1667, now filed as task 2040), so `trigger_reconciliation` was ineffective and the backlog stayed stuck ~880. So an auto-watcher resolving an L2 didn't just race — it closed a live problem on a false root cause.
- **Enforcement gap:** scope is prose-only. `resolve_issue` accepts any escalation id regardless of level, and `resolved_by` is a spoofable tool argument (`escalation/server.py:477-482`). The watcher's `_WATCHER_ALLOWED_TOOLS` allowlist (`harness.py:158-184`) is not a hard boundary because the subprocess runs `--permission-mode bypassPermissions` (it even reached `trigger_reconciliation`, deliberately omitted from the allowlist, and it succeeded).

**Key architectural insight:** the durable fix must live **server-side**, keyed on a per-connection capability set at connection-setup time — *not* a CLI permission (defeated by `bypassPermissions`) and *not* an identity→policy `if` baked into the server (poor separation of concerns; the user was explicit). The escalation server is a shared FastMCP HTTP server (`http://host:port/mcp`); each client (orchestrator, supervised watcher, interactive human, L0 stewards) connects with its own `--mcp-config` block. A per-connection capability rides that block as an **HTTP header**, enforced generically in the server.

## Sketch of approach

Add a **connection-level capability** carried by two HTTP headers the orchestrator sets in the supervised watcher's `mcp_config`, and enforce them generically in the escalation MCP server:

- `X-Escalation-Levels: 0,1` — the escalation **levels** whose state this connection may change via `resolve_issue` (any action, incl. `park`). Absent ⇒ unrestricted (all levels). This is a **capability grant passed at connection setup**, enforced per request; the server compares the target escalation's `level` against the connection's permitted set and rejects on miss. `promote_to_l2` (the sanctioned L1→L2 funnel) is **not** gated by this — it reads L1 members and creates/updates a pending L2, which is exactly what the watcher must keep doing.
- `X-Escalation-Identity: orchestrator-escalation-watcher-auto` — when present, the server **stamps `resolved_by` from this header**, ignoring the model-supplied `resolved_by` tool argument. Absent ⇒ current behaviour (`resolved_by` from the tool arg). This kills the invented `escalation-watcher-L2` and makes the identity server-attributed / non-spoofable by the agent.

The orchestrator (which builds the watcher's `mcp_config` in `_run_watcher_rotation`) attaches `Levels=0,1` + `Identity=orchestrator-escalation-watcher-auto` to the escalation server block. Every other client is unchanged and keeps full authority (default-open).

This approach is robust to `bypassPermissions` (the check is in the server, not the CLI permission layer) and general (levels are data at one enforcement point, not a fork of the tool surface).

## Resolved design decisions

1. **Server-side enforcement keyed on a connection header, not caller identity policy.** Per the user: the server reads a level-allowlist supplied at connection setup and enforces it generically; it does not contain a hard-coded `if resolved_by == watcher` policy. (Separation of concerns — the *policy* of who gets which levels lives at the connection-config site, i.e. the orchestrator's `mcp_config`; the *mechanism* of enforcement lives in the server.)
2. **Default-open for un-headered connections.** Absence of `X-Escalation-Levels` ⇒ all levels permitted. Rationale: the goal is to constrain one known over-eager trusted subprocess, not to defend against malicious clients (the watcher is our own process); fail-closed would break the interactive human session, L0 stewards, and every existing client, all of which connect header-less today. Restriction is **opt-in** by the connection that should be constrained.
3. **Malformed `X-Escalation-Levels` fails closed for that call.** An unparseable levels header ⇒ the state-changing call is rejected (`bad_capability_header`), never silently widened to open. A malformed *restriction* must not grant more than an honest one.
4. **`promote_to_l2` is exempt from the level gate.** It is the L1→L2 promotion funnel and does not resolve/close an existing L2. Gating it would break the watcher's core job. (The watcher creating an L2 via promotion is the *sanctioned* escalation path; the abuse is *resolving* an existing L2.)
5. **Server-attributed identity overrides the tool arg.** When `X-Escalation-Identity` is present, the server writes `resolved_by` from it for `resolve`/`park`. The agent cannot spoof or drift it.
6. **`l2-cascade` member resolution is unaffected.** Cascade fires server-internally when a *permitted* caller resolves a parent L2; it is not a fresh caller op and does not re-check the header.
7. **Supervised-watcher levels = `{0,1}`.** It may resolve L0/L1 admin items and promote to L2; it may not resolve/park level ≥ 2.
8. **The fix is fleet-wide via shared code.** `escalation/` is shared; landing on main + restarting each project's escalation server and orchestrator fixes reify (40 occurrences) and every other project too.

## Substrate reality check (G3)

| Assumed capability | Status | Evidence / resolution |
|---|---|---|
| FastMCP tool can read a request header | **verified exists** | `fastmcp 3.2.2`; `from fastmcp.server.dependencies import get_http_headers` imports and resolves in-venv this session. |
| `escalation/server.py` is a FastMCP HTTP server with per-client connections | **verified** | `create_server()` builds `FastMCP('escalation')`; orchestrator connects via `{'type':'http','url':escalation_url}` (`mcp_lifecycle.py:697-701`). |
| `mcp_config_json` can attach a `headers` field to the escalation block | **verified (trivial extension)** | The block is a plain dict at `mcp_lifecycle.py:697-701`; adding `'headers': {...}` is a one-line change. |
| **A `headers` field in `--mcp-config` survives the Claude Code CLI round-trip and reaches the tool per request** | **assumed — must be proven** | Novel here (escalation block carries no headers today). Split: (i) *server reads a client-sent header* → proven by task **α** (CI, direct HTTP client). (ii) *CLI forwards the configured header on each request* → proven by task **β**'s live smoke against the running escalation server. If (ii) fails, β escalates and the design pivots (URL-embedded token or a dedicated restricted tool surface) — see Open questions. |

No numeric premises. G3 is otherwise N/A (pure wiring of existing capabilities).

## Cross-PRD relationship (G4)

No contested-ownership seams. Adjacent, non-conflicting:

| Other PRD / work | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| Config hot-reload batch (df 2005-2009) | independent | `watcher_supervisor_*` are green-tier reloadable knobs; this PRD does not change reload semantics | this-prd | no overlap |
| Merge-queue refactor batch (df 1985-2002) | consumes (deploy only) | `scripts/restart-all-orchestrators.sh` may be reused for the fleet-wide deploy restart | that-prd owns the script | wired (reuse, no edit) |

## Decomposition plan

Labels are author-time; task IDs assigned at decompose. Linear-ish DAG with a companion doc-correction leaf.

### α — Server-side per-connection capability enforcement (escalation/)
- **Modules:** `escalation/src/escalation/server.py`, `escalation/src/escalation/queue.py` (read-only reference), new tests.
- **What:** In `resolve_issue`, read `X-Escalation-Levels` + `X-Escalation-Identity` via `get_http_headers()`. If Levels present and target `esc.level ∉ parsed_levels` → return `{'error': ..., 'code': 'level_forbidden'}` and make no state change (covers all actions incl. `park` — guard before `queue.park()` and before the pre-stamp/`queue.resolve()` path). Malformed Levels → `{'code':'bad_capability_header'}`. When Identity present, stamp `resolved_by` from it for resolve/park. Leave `promote_to_l2` ungated. Default (no header) = current behaviour.
- **Observable signal (leaf-capable, CI):** an integration test drives the escalation server over HTTP: a client sending `X-Escalation-Levels: 0,1` gets `level_forbidden` (and the record is untouched) when resolving a seeded **level-2** escalation; the same client resolving a **level-1** escalation succeeds and the archived record's `resolved_by` equals the `X-Escalation-Identity` header; a client sending **no** header resolves the level-2 escalation successfully. `promote_to_l2` from the `0,1` client still succeeds.
- **Manifest bindings:** `get_http_headers` → `grep:import fastmcp.server.dependencies` (verified exists); `level_forbidden` rejection mechanism → authored **here**, upstream of γ (G6 branch-4: rejection built + observed to fire in this task's own test).
- **Prereqs:** none.

### β — Orchestrator attaches the capability header for the supervised watcher (orchestrator/)
- **Modules:** `orchestrator/src/orchestrator/mcp_lifecycle.py` (`mcp_config_json`), `orchestrator/src/orchestrator/harness.py` (`_run_watcher_rotation`), tests.
- **What:** Extend `mcp_config_json(escalation_url, escalation_headers=None)` to emit `'headers'` on the escalation block; `_run_watcher_rotation` passes `{'X-Escalation-Levels':'0,1','X-Escalation-Identity':'orchestrator-escalation-watcher-auto'}`. Add a code comment at `_WATCHER_ALLOWED_TOOLS` noting the durable enforcement is now server-side (allowlist is advisory under `bypassPermissions`).
- **Observable signal (leaf-capable, live smoke — per the "live smoke before filing" feedback):** with the built escalation server running, a connection configured exactly as the watcher's `mcp_config` (Levels=0,1 + Identity header) is **denied** a level-2 `resolve_issue` (proving the CLI/HTTP header reaches the server end-to-end, substrate (ii)), while its level-1 resolve + `promote_to_l2` succeed with server-stamped `resolved_by`. If the header does **not** arrive (substrate (ii) fails), escalate — do not mark done.
- **Manifest bindings:** header channel end-to-end → `live-smoke` evidence (recorded on the task); `mcp_config_json` headers field → `producer:α` (server must read it) — α is upstream. ✓ DAG-direction.
- **Prereqs:** α.

### γ — Integration gate: two-way boundary tests (the leaf that realizes the contract)
- **Modules:** `escalation/tests/` (cross-cutting integration), possibly `orchestrator/tests/`.
- **What:** Realize the Boundary-test sketch below as an end-to-end test facing **both** sides of the seam.
- **Observable signal (leaf):** the six boundary scenarios all pass — watcher-side connection denied L2 resolve/park but allowed L1 resolve + promote; human-side (header-less) connection retains full L2 resolve/park; `l2-cascade` member resolution still fires when the human resolves an L2 with members; malformed-header call fails closed.
- **Manifest bindings:** rejection capability → `producer:α` (upstream); orchestrator wiring → `producer:β` (upstream). ✓
- **Prereqs:** α, β.

### δ — Docs + memory correction (companion correction phase)
- **Modules:** `skills/escalation-watcher-auto/SKILL.md`, `skills/escalation-watcher/SKILL.md`, memory note `feedback_escalation_watcher_check_for_canonical_owner`.
- **What:** (1) auto-watcher skill: state the invariant "never `resolve_issue` on `level >= 2`; the escalation server now **enforces** this — resolve L0/L1 admin, promote L2, leave L2 pending." (2) interactive watcher skill: teach that `resolved_by="orchestrator-escalation-watcher-auto"` is the **trusted supervised owner**, not a rogue — do not stand down on seeing it; distinguish from a genuinely unknown resolver. (3) correct the memory note's inverted dark-factory ownership framing.
- **Observable signal (companion-correction leaf, doc-grep):** both skills grep-match the new identity string and the invariant; the interactive skill's rogue-detection section names `orchestrator-escalation-watcher-auto` as trusted.
- **Prereqs:** α (needs the final identity string + error semantics).

## Contract (H)

**Header schema (set in the connecting client's `mcp_config` escalation block; read per request in the server):**

- `X-Escalation-Levels`: comma-separated non-negative ints (e.g. `"0,1"`). **Absent** ⇒ all levels permitted (default-open). **Present** ⇒ the exclusive set of escalation `level` values whose state this connection may change via `resolve_issue` (all actions, incl. `park`). **Unparseable** ⇒ every state-changing call from this connection is rejected `bad_capability_header` (fail-closed).
- `X-Escalation-Identity`: opaque string (e.g. `"orchestrator-escalation-watcher-auto"`). **Absent** ⇒ `resolved_by` comes from the tool argument (current behaviour). **Present** ⇒ the server overrides `resolved_by` with this value for `resolve`/`park` stamping.

**Server enforcement invariants:**
1. `resolve_issue(escalation_id, action)`: resolve `esc`; if Levels present and `esc.level ∉ Levels` → return `{'error': '<msg>', 'code': 'level_forbidden'}` and perform **no** mutation (no pre-stamp, no `queue.resolve`, no `queue.park`, no archive).
2. `promote_to_l2` is **not** gated by Levels.
3. Identity, when present, is the authoritative `resolved_by` for `resolve`/`park`; the tool arg is ignored.
4. `l2-cascade` member resolution (triggered internally by a permitted parent-L2 resolve) is not re-checked against Levels.
5. Malformed Levels ⇒ `bad_capability_header` on any state-changing call; no mutation.
6. Ordering: the level check runs **before** any write (before the `park` branch and before the `pending.resolution_action` pre-stamp at `server.py:554-556`).

## Boundary-test sketch (H) — the γ integration-gate signal

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Watcher denied L2 resolve | conn Levels=`0,1`; seeded pending **level-2** esc | `resolve_issue(close_only)` → `code=level_forbidden`; esc still pending; not archived |
| 2 | Watcher denied L2 park | conn Levels=`0,1`; pending level-2 esc | `resolve_issue(park)` → `level_forbidden`; esc unchanged (still open, no park stamp) |
| 3 | Watcher allowed L1 resolve, server-stamped | conn Levels=`0,1`, Identity set; pending **level-1** admin esc | resolve succeeds; archived; `resolved_by="orchestrator-escalation-watcher-auto"` regardless of tool arg |
| 4 | Watcher allowed promote | conn Levels=`0,1`; pending level-1 members | `promote_to_l2` succeeds; pending L2 created/updated |
| 5 | Human unrestricted | **no** Levels header; pending level-2 esc | `resolve_issue(park|close_only|resume)` succeeds; `resolved_by` from tool arg (`escalation-watcher`) |
| 6 | Cascade intact | header-less caller resolves an L2 with members | member L1s cascade-resolved (`resolved_by="l2-cascade:<id>"`) as today |
| 7 | Malformed header fail-closed | conn Levels=`"garbage"`; any esc | state-change call → `bad_capability_header`; no mutation |

## Pre-conditions for activating / deploy

- **Deploy = restart the escalation MCP server + orchestrator for each project** so the new server enforcement and the watcher header wiring load. Fleet-wide (shared code): dark-factory and reify are the live-impacted queues; may reuse `scripts/restart-all-orchestrators.sh` (merge-queue batch). Deploy can be a deterministic-deploy capstone or manual — see Open questions.
- No upstream task/PRD prerequisites; α has no deps.

## Out of scope

- Fixing the reconciliation-backlog lock itself (task 2040 / recurrence of 1667) — cited only as motivating evidence.
- Removing `trigger_reconciliation` reachability from the watcher subprocess (a `bypassPermissions`/allowlist concern). Once the watcher cannot resolve L2s, the drain-then-close pattern that motivated calling it is gone; the residual reachability is a separate, smaller hardening — note-only, future.
- A general authn/authz scheme for the escalation server (signed tokens, per-client secrets). This is a single opt-in capability grant for a trusted subprocess, not an adversarial trust boundary.
- Changing `watcher_supervisor_enabled` or the rotation cadence.

## Open questions (tactical — decide at impl/decompose)

1. **Deploy capstone shape.** Deterministic `task_kind='deterministic'` deploy that restarts all orchestrators + escalation servers, vs a manual operator restart. *Suggested:* manual for the first landing (small blast radius, easy to verify the header arrives via β's smoke), consider a capstone if it recurs. Decide at decompose.
2. **Header names.** `X-Escalation-Levels` / `X-Escalation-Identity` vs an existing convention. *Suggested:* the `X-Escalation-*` names above unless a house convention exists. Decide in α.
3. **Substrate (ii) fallback if the CLI drops the header.** If β's smoke shows the configured header does not reach the server, pivot to a URL-embedded capability token or a dedicated restricted tool surface for the watcher. Decide in β only if the smoke fails.
4. **Should L0-steward connections also carry an explicit ceiling?** Out of scope now (they connect header-less = open, unchanged), but a future tightening could give each role an explicit level set. Note-only.
