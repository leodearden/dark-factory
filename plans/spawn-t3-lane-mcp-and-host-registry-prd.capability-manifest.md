# Capability manifest — deepen the `/spawn` T3 lane through T3's MCP server, host-qualify the session registry

PRD: `plans/spawn-t3-lane-mcp-and-host-registry-prd.md` (landed `0fcc13fcd9`). Decomposed
2026-11-01 by the authoring `/team` seat in `/prd` decompose mode. Substrate re-read on main
at `0fcc13fcd9` and on the T3 source worktree `~/src/t3code-v0.1.6` (tag `v0.1.6`, the
installed server). Machine-readable twin:
`plans/spawn-t3-lane-mcp-and-host-registry-prd.capability-manifest.yaml` (stamped by
`commit_planning`). Task ids: α 8122, β 8123, γ 8124, δ 8125, ζ 8126, ε 8127.

Binding vocabulary is `skills/prd/references/gates.md` → *Capability Manifest*. Every
mechanical `delivered_check` below was pre-checked against `main` before filing: each
`expect: present` pattern has zero content matches today and each `kind: path` file is
absent from `main`'s tree; the one `expect: absent` check (`import sqlite3` in
`spawn_t3_lane.py`) matches today. Zero polarity findings.

## Resolutions made at decompose

No binding failed. Decisions the decomposition had to take without a human:

1. **Priority.** α, β, γ, δ, ζ `high` (they are the critical path to the PRD's goal and to
   Leo's two 2026-11-01 scope additions); ε `medium`, as the 7194 gate was.
2. **Substrate flag spelling.** `metadata.substrate_confirmed: true` on every leaf (a Tier-A
   key; this PRD has no grammar, so `grammar_confirmed` would be a misnomer).
3. **Mechanical checks avoid implementer-chosen names.** Checks bind contract-fixed tokens
   (PRD §6.1 verdict words, §6.5 input/field names, the §4 M3 helper name `record_exit`, the
   SKILL.md subsection heading) or new-file paths, never a class name an implementer may
   spell differently. The sibling-mode stdout line and the handoff gate are `manual`: the
   line's tokens (`t3_thread_id=`, `record=`) already occur in `spawn-claude.sh`, so a grep
   would be vacuous (`vacuous_present`).
4. **Out-of-batch edges.** 7651, 7653 and 7617 now depend on γ (8124), as PRD §9 directs;
   their own text is unchanged — the decision to close 7653/7617 as superseded is taken when
   γ lands, not here.

## Per-leaf bindings

### α (8122) — `t3_mcp_client.py`, pairing exchange, credential file, `pair`/`status`

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| client module exists | producer α (new file); absent on main | PASS | path `orchestrator/src/orchestrator/t3_mcp_client.py` present |
| self-pair refuses CLI/server version mismatch | rejection built by α; verdict word `cli-version-mismatch` fixed by PRD §6.1; absent on main | PASS | grep `cli-version-mismatch` in `orchestrator/src/orchestrator/` |
| pairing-code OAuth exchange, 30-day session | T3 substrate present: `packages/contracts/src/environmentHttp.ts` `mcpOAuth` (`register`/`decision`/`token`), `apps/server/src/auth/McpOAuth.ts::approve` pairing-code branch, `EnvironmentAuth.ts` `MCP_CLIENT_SESSION_TTL`; `apps/server/src/cliAuthFormat.ts::formatIssuedPairingCredential` emits `credential`; B1/B2 | PASS | manual |

### β (8123) — launch/await-started/wait over MCP; join writes `exited`

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| launch binds an existing worktree | T3 substrate `project/tools.ts::ThreadLaunchTool` `workspaceStrategy`, `handlers.ts::assertProjectWorktree`; mapping built by β; absent in the lane on main | PASS | grep `existing_worktree` in `spawn_t3_lane.py` |
| worktree refusal named, never parsed | T3 fails with the shared code `invalid_request`; the lane maps it only for an `existing_worktree` request (PRD D6, §7 B4) | PASS | grep `cwd-not-project-worktree` in `spawn_t3_lane.py`, `spawn-claude.sh` |
| private sqlite read removed | `registered_projects` and `import sqlite3` deleted (PRD D1); present today | PASS | grep `import sqlite3` absent in `spawn_t3_lane.py` |
| child join writes `exited` via the `exit` verb | one `record_exit` helper shared by `finish()` and `join_t3_thread` (PRD D4, INV-6); absent on main | PASS | grep `record_exit` in `spawn-claude.sh` |
| runtime mode from `skip_permissions` | T3 `ThreadLaunchTool.runtimeMode`, `providerPolicy.ts::RuntimeMode`; B3 asserts the sent input | PASS | manual |

### γ (8124) — `T3CODE_THREAD_ID` binding, pointer index, indexed env-qualified sweep

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `sessions-by-t3-thread/` index | producer γ in the shape of `session_registry.py::write_claude_session_pointer` (present); name absent on main | PASS | grep `sessions-by-t3-thread` in `orchestrator/src/orchestrator/` |
| `Display.t3_environment_id` populated | stamped by β's launch from the unauthenticated descriptor on the production path; field by γ | PASS | grep `t3_environment_id` in `session_registry.py` |
| hook binds via the thread pointer under adopt-or-fork | `T3CODE_THREAD_ID` set per launch and resume (`PortScanner.ts::AGENT_THREAD_ENV`, `ClaudeAdapterV2.ts`); `session_hooks.py::_pointer_resolution` rules present; B6/B7 | PASS | manual |

### δ (8125) — host-qualified liveness

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `session_liveness.py` exists | producer δ over `shared/src/shared/process_identity.py::liveness` (present; `OTHER_HOST` present); absent on main; homed in `shared` per Leo ruling A, 2026-11-04, esc-8125-3 (PRD D16) | PASS | path `shared/src/shared/session_liveness.py` present |
| records carry `launcher_identity` | stamped by `_run_launching` on the production path | PASS | grep `launcher_identity` in `session_registry.py` |
| self-exit refuses `other-host` | `SelfExitRefusal.OTHER_HOST` ahead of the pid comparison in `_first_refusal` (present) | PASS | grep `OTHER_HOST` in `session_self_exit.py` |
| `OTHER_HOST` branch | declared seam: no production producer on this host; proven by B9 alone; consumer the future multi-host PRD (PRD §3, §9) | PASS | manual |

### ζ (8126) — end-of-work acts and the handoff archive

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `CLAUDE_SPAWN_T3_END` resolved by the script | producer ζ; name fixed by PRD §6.5 | PASS | grep `CLAUDE_SPAWN_T3_END` in `spawn-claude.sh` |
| records carry `t3_end` | stamped by `_run_launching` from the resolved input | PASS | grep `t3_end` in `session_registry.py` |
| "Ending a T3 session" has one home | SKILL.md §T3 lane subsection (INV-9) | PASS | grep `Ending a T3 session` in `skills/spawn/SKILL.md` |
| self-archive and self-settle exist in T3 | `thread/tools.ts::ThreadOrganizeTool` (`archive`, `settle`; `threadId` omitted = caller), `McpToolAccess.ts::writesThreads` skips the caller's thread, `ThreadManagementService.ts::settleThread` `byOwnAgent` → `settlesWhenTurnEnds`, `::settleAfterRun` requires `completed`; archive's detach ends the run `failed` (`ProviderSessionManager.ts::detach` → `::releaseEntry` → `::failSubscribers`) | PASS | manual |
| handoff archive gated on the named successor record | no archive without a stamped record read from the sibling spawn's stdout line (PRD D15); B11 (a)–(d) | PASS | manual |

### ε (8127) — live acceptance probe (human gate)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| twelve live observations recorded | DAG-direction: everything produced upstream (α–ζ); the task is the integration gate, filed as 7194 was (`execution_class: operational`, `task_kind: deterministic`, `always_escalates: true`, `files: []`) | PASS | manual |
