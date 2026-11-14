# PRD: Deepen the `/spawn` T3 lane through T3's MCP server, and host-qualify the session registry

**Status:** active — authored 2026-11-01 (author mode, `/prd`), awaiting commit and decompose.
**Approach:** B+H (contracts §6 + boundary-test sketch §7); the lane is attended-session
infrastructure where a silent defect strands records and spawns (Leo 2026-09-30: a wasted
verify is acceptable, a silent jam is not).
**Code anchors:** cite-by-symbol throughout, never `path:line`. Dark-factory cites are
repo-relative to `/home/leo/src/dark-factory`; T3 cites are repo-relative to the source
worktree `~/src/t3code-v0.1.6` (tag `v0.1.6`, which is the installed workstation server as
of 2026-11-01, `~/.t3/userdata/server-runtime.json` → `serverVersion 0.1.6`). Every T3
tool name and input key named below was re-opened in that tree before saving (§8).

## 1. Goal

The `/spawn` T3 lane (task 7489, landed 2026-10-27) becomes a first-class host for
attended sessions: a T3-hosted spawner can hand a session into a T3 thread that runs in a
`.worktrees/<id>` checkout, under `skip_permissions=false`, with its registry record bound
to the thread by identity rather than by transcript text, and whose child-mode join ends
with the record `exited` — all through T3's `/mcp` server as an outside OAuth client, with
no read of T3's private sqlite and no shell-out to the `t3 thread` CLI. Separately, every
pid-based liveness judgment over the session registry becomes host-qualified, so that no
sweep, watcher or self-exit ever judges a record from another host by a local pid.

User-observable behaviour that ships:

- `CLAUDE_SPAWN_BACKEND=t3` (or `auto` from a T3-hosted spawner) with `cwd=.worktrees/<id>`
  launches a thread bound to that worktree instead of exiting 145.
- `skip_permissions=false` on the T3 lane launches an `approval-required` thread instead of
  exiting 145.
- A child-mode T3 join leaves `record.json` at `status: exited` with the join's code.
- A T3 thread's record is bound at the child's first `SessionStart` hook via
  `T3CODE_THREAD_ID`; a thread opened by hand in the T3 UI gets a `display.kind: t3` record.
- `spawn_t3_lane.py pair` / `status` manage the lane's 30-day MCP credential; an expired or
  missing credential is a loud `no-launcher` (exit 126) that names the fix, never a silent
  fallback.
- Session records carry `launcher_identity` / `claude_owner_identity` (host, boot id, start
  ticks); after a reboot, pre-reboot records are judged dead by boot id; a record from
  another host is never swept, reaped or SIGINTed by a local pid.
- A T3-hosted session can end its work with an explicit act — `settle` (out of Leo's inbox,
  thread kept, `result.md` final) or `archive` (off the active list, record `exited`) —
  asked for per launch by `CLAUDE_SPAWN_T3_END` or chosen by the skill; a refused act is
  reported in `result.md`, never retried blind. A T3-hosted `/escalation-watcher` told to
  "handoff" archives its own thread only after its successor sibling is confirmed launched
  (record with `display.t3_thread_id`), so Leo's sidebar shows one live watcher per queue.

## 2. Background

- The 2026-11-01 `/team` study (`plans/t3-orchestrator-v2-study-2026-11-01/report.md`
  §0, §2, §5.2, §6; seats S2, S3, S4, S6) concluded that T3 Code **complements** `/spawn`
  and never replaces it: T3 is the session host and cockpit; `/spawn` is dark-factory's
  handoff contract (identity, fleet registry record, `result.md`, join semantics, title
  convention, lane policy). Leo ruled the same day: deepen the existing lane, do not add a
  mechanism.
- Today's lane (`orchestrator/src/orchestrator/spawn_t3_lane.py`) shells to the `t3 thread`
  CLI (`::run_t3`) and reads `$T3CODE_HOME/userdata/statev2.sqlite` table
  `projection_projects` (`::registered_projects`). The CLI has no runtime-mode flag, no
  existing-worktree flag and no project list (`apps/server/src/cli/thread.ts::threadStartCommand`,
  `apps/server/src/cli/project.ts::projectCommand`; S3 §3, unchanged in the 0.1.7 nightly),
  so three inputs are refused with exit 145 (`skills/spawn/spawn-claude.sh::refuse_inputs_the_t3_lane_cannot_honour`,
  `spawn_t3_lane.py::launch`). The MCP surface honours all three
  (`apps/server/src/mcp/toolkits/project/tools.ts::ThreadLaunchTool` takes `runtimeMode` and
  `workspaceStrategy`, `::ProjectListTool` lists projects with `workspaceRoot`).
- `registered_projects` selects `WHERE deleted_at IS NULL`; 0.1.6 added an archived project
  state (`apps/server/src/persistence/Migrations/076_ProjectionProjectsPinArchive.ts`), and
  `apps/server/src/orchestration-v2/ProjectStore.ts` lists archived projects alongside live
  ones, so the lane admits an archived project as a launch target (S6 W4).
- Identity: no caller-supplied env reaches a T3 thread, but 0.1.6 sets one per-thread
  variable, `T3CODE_THREAD_ID` (`apps/server/src/preview/PortScanner.ts::AGENT_THREAD_ENV`,
  set in `apps/server/src/orchestration-v2/Adapters/ClaudeAdapterV2.ts::ClaudeAgentSdkQueryRunner`
  `open`, on every launch and every resume). The lane binds records by transcript discovery
  today (`spawn_t3_lane.py::discover_transcript`, `::_launch_markers`). Hooks do fire under
  T3: tasks 7614 and 7617 measured hand-launched T3 threads writing
  `session-<project>-<sid>` records on the live fleet root on 2026-10-27.
- Join: `spawn-claude.sh::join_t3_thread` exits with the lane's code and never calls the
  registry `exit` verb, unlike `::finish` on the terminal lanes (S4 P21, S6 confirmed); a
  finished, unarchived thread's record stays non-terminal until a later spawn's thread
  sweep happens to run (`session_registry.py::mark_t3_sessions_exited`, driven only from
  `spawn_t3_lane.py::_sweep_project_listing` and `::wait`).
- Host locality (S4 §Host-locality, S5): `orchestrator/src/orchestrator/session_registry.py::SessionRecord`
  carries no host; liveness is `::_pid_alive` (`os.kill(pid, 0)`), `::is_shared_session_host`
  and `::_wmctrl_live_window_ids` read `/proc` and X11; the fleet root is `~/.claude/fleet`
  (`::fleet_root`). `shared/src/shared/process_identity.py::liveness` already answers
  `ALIVE | DEAD | OTHER_HOST` from a `ProcessIdentity(pid, start_ticks, boot_id, host)` and
  has no registry caller.
- Standing rulings preserved (§11): a watcher runs on the lane its skill names (T3 when
  operator-launched there, tmux from a terminal — `skills/escalation-watcher/SKILL.md`
  §"Launching this watcher"), and `/do` plan-mode handoffs stay off T3; self-exit default-on
  for tmux/terminal, opt-in for T3 (tasks 7194, 7947, landed `8f6781cce1`); decision A spawner rule for lane selection
  (`spawn-claude.sh::resolve_spawn_backend`, exit 145 on refusal); from a T3 session spawn
  one at a time in the foreground, never under `setsid` (memory 2026-11-01).

## 3. Consumers and user-observable surface (G1)

| Mechanism introduced | Named consumer | Surface where it is observed |
|---|---|---|
| M1 outside-OAuth MCP client (`t3_mcp_client.py`) + credential file + `pair`/`status` verbs | `spawn_t3_lane.py` `launch`/`await-started`/`wait` (every `/spawn` caller on the T3 lane: escalation-watcher `/unblock` spawns, `/prd` sibling decompose, review-all seats, `/team` carries — S4 §Callers) | `spawn_t3_lane.py status` output; `no-launcher reason=t3-mcp-*` lines from `spawn-claude.sh` |
| M2 MCP-driven launch (`existing_worktree`, `runtimeMode`, `t3_project_list`) | the same callers; specifically `/team` hand carries that work in `.worktrees/carry-<id>` and attended `/unblock` sessions that want permission prompts | the thread appears in T3 with `worktreePath` set and `runtimeMode: approval-required`; the three 145 refusals disappear from `skills/spawn/SKILL.md` §T3 lane |
| M3 child-join `exited` write | `skills/review-all/scripts/program/watch_sessions.py::is_over` (terminal status), the fleet cockpit's status column (`plans/fleet-cockpit-prd.md` §1), `session_registry.py::reap_stale_records` (terminal TTL) | `record.json` `status: exited` after a child join |
| M4 `T3CODE_THREAD_ID` binding + `sessions-by-t3-thread/` index + env-qualified thread sweep | `orchestrator/src/orchestrator/session_hooks.py::_resolve_hook_slug` (every hook event in a T3 thread); `session_registry.py::mark_t3_sessions_exited`; task 7651's T3 focus backend (reads `display.t3_thread_id`, `display.t3_environment_id`) | record bound at first `SessionStart`; hand-opened T3 threads get `display.kind: t3`; archived threads flip to `exited` without a full `sessions/` scan |
| M5 host-qualified liveness (`session_liveness.py`, identity fields) | `session_registry.py::reap_stale_records`, `::mark_orphaned_sessions_exited`, `::reap_stale_session_pointers`; `session_hooks.py::_reclaim_dead_owner_pid`; `session_self_exit.py::_owner_refusal`; `watch_sessions.py::is_over` | `record.json` identity fields; post-reboot sweep behaviour; `watch_sessions.py` line shows `host=`; self-exit refusal `other-host` |
| M6 T3 end-of-work acts (`CLAUDE_SPAWN_T3_END`, the "Ending a T3 session" procedure, the watcher's handoff archive) | Leo triaging the T3 inbox/sidebar from the laptop or phone, who must not settle or archive finished agent threads by hand and should see exactly one live watcher thread per queue after a handoff; `session_registry.py::mark_t3_sessions_exited` and the registry `exit` verb, which would otherwise keep the old record non-terminal; `watch_sessions.py::standing` (`FINISHED_NOT_EXITED` for a settled thread) | the thread leaves the inbox (settle) or the active list (archive); `record.json` `exited` after archive; `result.md` `action_needed` naming a refused act |

Honest limits of G1 for M5: the readers are real and named, and the *local* half of the
contract (boot-id death after a reboot, start-ticks immunity to pid reuse) has a production
producer on this host today. The `OTHER_HOST` branch has **no production producer** until a
second host writes into a shared fleet root, which this PRD does not build (§11). The PRD
still lands the branch — one comparison inside `judge`, reusing
`shared/src/shared/process_identity.py::liveness`'s existing `OTHER_HOST` value — because
omitting it would leave a partial predicate that judges a foreign record by a local pid,
the exact failure this part exists to prevent. Its proof is B9 (fake identities) alone; its
consumer is the future multi-host PRD (study §5.3), recorded as a declared seam in §9, and
δ's user-observable signal lists only what this host shows. Every reader must already
render "unqualified (legacy record)" distinctly from "this host" (INV-13
`readers-prove-their-producer`). A host key on the
orchestrator fleet heartbeat (`orchestrator/src/orchestrator/fleet_heartbeat.py::DEFAULT_FLEET_DIR`,
`::build_heartbeat_payload`) has **no consumer that can honestly be named**: every
orchestrator unit runs on the workstation, its readers (`scripts/orchestrator-watchdog.py
--report`, `scripts/drain_check.py`, `scripts/restart-all-orchestrators.sh`) address
heartbeats by unit name, and the only host dimension that exists in production is already
carried per verify in `verifies_in_flight[].host`. It is out of scope (§11), not deferred.

## 4. Sketch of approach

### M1 — the lane becomes an outside OAuth MCP client

A new stdlib-only module `orchestrator/src/orchestrator/t3_mcp_client.py` speaks JSON-RPC
2.0 over MCP streamable HTTP to `<origin>/mcp` (`initialize`, `notifications/initialized`,
`tools/call`; `Authorization: Bearer <token>`, `MCP-Protocol-Version: 2025-06-18`, the
`Mcp-Session-Id` header echoed back, `Accept: application/json, text/event-stream`, and a
decoder for both a JSON body and SSE `message` frames). The origin is
`server-runtime.json`'s `origin` field (`http://127.0.0.1:3773` on this host; plain http is
accepted by T3 only to loopback, `docs/user/outside-agents.md` §"Get the MCP URL").

Sign-in is the pairing-code OAuth exchange with no browser, copied from
`apps/server/src/peer/PeerLinks.ts::signIn`: `POST /oauth/mcp/register` (public client,
`redirect_uris` a loopback URI nothing listens on, `software_id` of the lane's own) →
`POST /oauth/mcp/decision` with `{_tag: "pairing-code", access, code}` plus PKCE S256 →
`POST /oauth/mcp/token` (`packages/contracts/src/environmentHttp.ts` `mcpOAuth` group;
`apps/server/src/auth/McpOAuth.ts::approve` pairing-code branch via
`apps/server/src/auth/EnvironmentAuth.ts::consumeMcpApprovalCode`; `::exchangeCode` →
`::issueMcpClientSession` with `MCP_CLIENT_SESSION_TTL = 30 days` and
`runtimeModeCeiling = access`). Access is `full-access` (`packages/contracts/src/auth.ts::AuthMcpClientAccess`),
the ceiling a thread may be started at.

On the T3 host the lane **self-pairs**: it runs `t3 auth pairing create --json` (`apps/server/src/cli/auth.ts::pairingCreateCommand`;
reads the owner state dir, as `t3 thread` already does to mint its in-process full-access
session, `apps/server/src/cli/thread.ts::withThreadTools`), takes the code from the
`credential` field of that JSON (`apps/server/src/cliAuthFormat.ts::formatIssuedPairingCredential`
emits `{id, credential, label?, scopes, expiresAt, pairUrl?}`) and completes the exchange.
Before minting it compares `t3 --version` with the descriptor's `serverVersion` and refuses
with `pair-failed reason=cli-version-mismatch` when they differ — a GUI server update leaves
`~/.local/bin/t3` on the old version (memory `project_t3_code_setup_2026_10_23`), and a stale
CLI opening the newer state dir would fail confusingly. Off the host,
`spawn_t3_lane.py pair --code <code> [--origin <url>]` takes an operator-minted code. The credential (`<fleet_root>/t3-mcp/<environment_id>.json`, mode 0600, §6.2) is
selected per launch by the environment id from the unauthenticated descriptor
`GET /.well-known/t3/environment` (`packages/contracts/src/environmentHttp.ts` `descriptor`;
`packages/contracts/src/environment.ts::ExecutionEnvironmentDescriptor`). Expiry
(`expires_at` within one hour, or any 401) triggers a self-pair on the host and a loud
`no-launcher reason=t3-mcp-credential-expired` elsewhere.

### M2 — launch, started, wait over MCP

- `launch`: `t3_project_list` (paged, `limit ≤ 100`, follow `nextCursor`; refuse on an
  incomplete page set) → projects with `archivedAt == null` and `deletedAt == null`
  (`packages/contracts/src/project.ts::Project`) → the deepest `workspaceRoot` containing
  `realpath(cwd)` (today's `::containing_project` rule). Then `t3_thread_launch` with
  `projectId`, `title`, `message` (or the spill pointer, `::_message_for` unchanged),
  `modelSelection: {instanceId: "claudeAgent", model}` when `CLAUDE_SPAWN_MODEL` is set,
  `runtimeMode` per §5 D3, `workspaceStrategy: {type: "existing_worktree", worktreePath: cwd}`
  when `cwd` is not the root (else `{type: "root"}`), and `clientRequestId` = the record
  slug (replaces `::_find_launched_after_failure`: a lost reply is retried with the same key
  and T3 replays the same thread, `apps/server/src/mcp/toolkits/project/handlers.ts`
  `launchCommandId`). A worktree that `git worktree list` does not know fails inside T3
  (`handlers.ts::assertProjectWorktree`, `OrchestratorMcpFailure{code: "invalid_request"}`).
  `invalid_request` is a shared code, so the lane maps it to
  `refused reason=cwd-not-project-worktree` only when the request it sent carried
  `workspaceStrategy.type == "existing_worktree"`, and puts T3's `message` verbatim in
  `detail=`; it never parses the message text (heuristic 12). The lane re-implements no
  worktree listing (heuristic 11, SPOT: T3 is the authority).
- `await-started`: thread status from `t3_thread_read` (`OrchestratorMcpThreadDetail.status`,
  `packages/contracts/src/orchestratorMcp.ts`) with today's verdict table; binding per M4
  first, transcript discovery second.
- `wait`: loop on `t3_thread_wait(threadId, timeoutMs)` in slices (default
  `DEFAULT_WAIT_TIMEOUT_MS` 10 min, cap `MAX_WAIT_TIMEOUT_MS` 60 min,
  `apps/server/src/mcp/OrchestratorMcpService.ts`); on return, `JoinTracker.verdict` as
  today over `(status, latestRunId, archived)` from `t3_thread_read`, with `result_is_final`
  read between slices; `CallerGone` checked between slices; the project sweep (M4) runs
  once per slice return, not every 30 s. `SPAWN_T3_POLL_SECS` and `DEFAULT_WAIT_POLL_SECS`
  retire; `ENDED_RUN_GRACE_SECS` and `UNREACHABLE_GRACE_SECS` keep their meaning
  (unreachable = transport failure, not a tool failure).
- `detect` and `self-record` are unchanged: they read the process tree and
  `server-runtime.json`, which is the `t3` CLI's own runtime pointer, not the private store.
- `::run_t3`, `::registered_projects`, `::ProjectStoreUnreadable`, `::thread_detail`,
  `::list_threads` (CLI-shaped) and the `sqlite3` import are deleted, not kept as fallbacks
  (D1).

### M3 — the child join writes `exited`

`spawn-claude.sh::join_t3_thread` records the join's code through the same registry `exit`
verb `::finish` uses, via one factored `record_exit <code>` helper both call. The registry
`exit` verb additionally deletes the record's `sessions-by-t3-thread/<id>` pointer (M4),
so the live-thread index never names a terminal record.

### M4 — identity by `T3CODE_THREAD_ID`, and an index of live T3 records

- New pointer dir `<fleet_root>/sessions-by-t3-thread/<thread_id>` → slug, the shape of
  `session_registry.py::write_claude_session_pointer`. Written by the lane in
  `spawn_t3_lane.py::_stamp_launched_record` (spawn slug) and by
  `session_hooks.py::run_session_start` when the env carries `T3CODE_THREAD_ID` and no
  pointer exists yet (its own hand-launched slug).
- `session_hooks.py::_resolve_hook_slug`: with no `CLAUDE_SPAWN_SESSION_ID` and a
  `T3CODE_THREAD_ID`, the thread pointer is tried as a third pointer kind inside
  `::_pointer_resolution`, under the same adopt-or-fork rules as the sid and pid pointers
  (`docs/session-hook-ownership.md` §One read, one decision, one write): a nested `claude`
  started inside the thread inherits the variable exactly as it inherits
  `CLAUDE_SPAWN_SESSION_ID`, and is told apart by a bound-sid mismatch or, before binding,
  by `EventProbes.owning_claude_pid` against `claude_owner_pid`. A hand-launched record the
  hook creates for a T3 thread is stamped `Display(kind="t3", t3_thread_id=...)`, which puts
  it under the thread sweep and out of the pid sweeps (`::is_t3_record`) — the class task
  7617 measured.
- Lane reconcile: if `_stamp_launched_record` finds the pointer already naming a
  hand-launched record (the hook won the race), it binds that record's `claude_session_id`
  and `claude_owner_pid` into the spawn record, folds the hand record
  (`session_registry.py::mark_session_folded`), and re-points the index. `::bind_session`'s
  transcript path stays as the fallback when no pointer exists (a T3 without the variable,
  or hooks not installed). A resume after idle release carries the same `T3CODE_THREAD_ID`,
  so its `SessionStart` re-binds through the pointer — closing the "failed resume binds a
  fresh sid" gap (`plans/spawn-t3-lane-design-2026-10-25.md` §8 "Idle release").
- `Display` gains `t3_environment_id` (from the descriptor, M1). `mark_t3_sessions_exited`
  is re-keyed: it iterates `sessions-by-t3-thread/*` (hundreds of entries) instead of
  `sessions/*` (134,926 dirs, ~150 s per scan measured by task 7653), takes one
  `t3_thread_list(allProjects: true)` paged listing of the environment, and marks any
  non-terminal record of that environment whose thread is absent; the `cwd`/project-root
  match is dropped (a worktree-bound thread's `cwd` is not the root). A record with
  `t3_environment_id` unset is treated as this environment (legacy).

### M5 — host-qualified liveness

New module `shared/src/shared/session_liveness.py` (stdlib + `shared.process_identity`, a
registered member of `shared/tests/test_pure_stdlib_leaves.py::PURE_STDLIB_LEAVES`), beside
`shared/src/shared/process_identity.py`, which already provides `capture` and `liveness`: the
new module is a thin `judge(identity, pid) -> Liveness` (D16). Bare-python3 consumers reach it
through `session_registry.py`'s existing `shared/src` bootstrap, pinned by
`orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment`; it imports
nothing from `orchestrator`. `SessionRecord` gains `launcher_identity` and
`claude_owner_identity` (`ProcessIdentity` as a dict; additive, default `None`), stamped
where the pids are stamped today (`::_run_launching`; `session_hooks.py::_bind_claude_session_id`,
`::_reclaim_dead_owner_pid`; the new-record path of `::run_session_start`). Every reader that
decides "dead" switches from `_pid_alive(pid)` to `judge(...) is Liveness.DEAD`: `OTHER_HOST`
is neither alive nor dead, so sweeps skip, `is_over` answers not-over, `_reclaim_dead_owner_pid`
does not reclaim, and `session_self_exit.py::_first_refusal` gains `SelfExitRefusal.OTHER_HOST`
ahead of the pid comparison. An identity of `None` falls back to
`session_registry.py::pid_alive` (legacy) — passed into `judge` or kept at the call site,
since the shared module cannot import the registry — and readers that print a record say
`host=unqualified`.

### M6 — T3 end-of-work acts: settle, archive, and the watcher's handoff archive

T3 offers two end states a thread can reach on its own through `t3_thread_organize`
(`apps/server/src/mcp/toolkits/thread/tools.ts::ThreadOrganizeTool`, actions `pin | unpin |
snooze | unsnooze | settle | unsettle | archive | unarchive | mark_unread`, `threadId`
omitted = the calling thread; `apps/server/src/mcp/McpToolAccess.ts::writesThreads` skips the
mode-limit check for the caller's own thread):

- **settle** — inbox triage: the thread leaves the inbox, stays in the sidebar and in the
  active listing (`t3_thread_list` filters on `settled` only when asked,
  `apps/server/src/mcp/OrchestratorMcpService.ts`); the provider session is detached at the
  end of the turn (`apps/server/src/orchestration-v2/ThreadManagementService.ts::settleThread`
  with `byOwnAgent` returns `{settlesWhenTurnEnds: true}` and defers through
  `settleAfterRun` while the run is active); idle terminals close and the project's
  `settle` script runs (S1 §6.5; auto-settle default 3 days,
  `apps/server/src/orchestration-v2/ThreadSettlementService.ts`).
- **archive** — the thread leaves the active list: `thread.archive`
  (`apps/server/src/orchestration-v2/Orchestrator.ts`) refuses only an already-archived
  thread, cancels `queued`/`preparing` runs, detaches every provider session with
  `revokeMcpCredential: true` and runs `terminal.cleanup`. The command itself leaves a
  `running` run alone, but the detach it queues ends it: Claude's adapter runs one provider
  thread per session, so `apps/server/src/orchestration-v2/ProviderSessionManager.ts::detach`
  → `::releaseEntry` (`manual_shutdown`, "Thread archived.") kills the live `claude` process
  group (S1 §4.5) and `::failSubscribers` ends the run `failed`
  (`RunExecutionService.ts::makeFailedTerminalEvent`, the provider turn stamped
  `interrupted`; `::writeFinalRunEvents`), a terminal status for
  `ThreadManagementService.ts::isTerminalRunStatus`, so a `t3_thread_wait` on it wakes promptly. The lane still keys the join on
  `OrchestratorMcpThreadDetail.archived` (`JoinTracker` → 0), not on the run status, and the
  sweep on absence from the active listing (M4).

The lane exposes the choice as one per-launch input, `CLAUDE_SPAWN_T3_END = none | settle |
archive` (default `none`; a `CLAUDE_SPAWN_*` input, wiped from the child like the rest),
stamped on the record as `t3_end` and rendered by the trailer as the last instruction of a
T3-lane spawn. `CLAUDE_SPAWN_SELF_EXIT=1` on the T3 lane keeps its meaning and is the same as
`archive` (the existing archive trailer; `gen_brief.py`'s review-all seats need no change);
`SELF_EXIT=1` together with `T3_END=settle` is bad usage (exit 2). On the tmux and terminal
lanes the input is not applicable (the process exit is the end) and the script says so on
stderr once. The self-exit ruling (default-on tmux/terminal, opt-in T3) is untouched: the
default end on T3 stays `none`; the act is asked for per launch or chosen by the skill.

One procedure, **"Ending a T3 session"**, lives in `skills/spawn/SKILL.md` §T3 lane and is
pointed at by the trailer and by `skills/escalation-watcher/SKILL.md`: (1) `result.md` final
(an outcome other than `in-progress`); (2) `/reflect`; (3) for `archive`: write the own
record `exited` through the registry choke point (`python3 session_registry.py exit --record
<own dir> --code 0`, the dir from `spawn_t3_lane.py self-record`), then call
`t3_thread_organize action=archive` with no `threadId` as the final tool call — success is
unobservable from inside (the process dies with the detach), so nothing may follow; for
`settle`: call `t3_thread_organize action=settle`, read `settlesWhenTurnEnds: true`, and end
the turn with no further tool call and no question pending — the settle applies only if
this turn then **completes** normally (`ThreadManagementService.ts::settleAfterRun`
dispatches `thread.settle` only for `run.status === "completed"`); a turn that ends
`failed`/`interrupted`, or parks on a pending question, leaves the thread in the inbox with
no failure ever returned; (4) a refused act (an `OrchestratorMcpFailure` result) is written
into `result.md`'s `action_needed` with the act and T3's `code`/`message`, once, with no
retry; for archive the record already marked `exited` is corrected by the thread's next
hook event (`session_registry.py::mark_orphaned_sessions_exited` docstring: a hook refresh
flips a wrongly terminal record back). A settle that silently did not apply (3) has the
same registry state as one that did (`FINISHED_NOT_EXITED`) and needs no further action
from the session; Leo sees the thread still in the inbox.

Decision rule the skills apply: **settle** when the thread's own transcript is what Leo reads
(attended `/unblock`, `/prd` author, hand carries, anything that ends `awaiting-input` for
him); **archive** when the output lives elsewhere (review-all seats, `result.md`-only
workers) or after a successful handoff.

Registry consequence: a self-settled thread's record stays `idle` with `result.md` final —
`watch_sessions.py::standing` already names this `FINISHED_NOT_EXITED`; the thread sweep
leaves it (settled threads are listed) until Leo or a later act archives it. A self-archived
thread's record is `exited` (own write), the thread is absent from the active listing (the
sweep is the backstop), and a child-mode `wait` on it returns 0.

**Handoff archive.** `skills/spawn/SKILL.md` §`handoff` (landed `94d33dba75`) gains a final
step for a T3-hosted watcher: only after step 4 has confirmed the successor — exit 0 and
the record the spawn itself named has `display.t3_thread_id` set and `status: running` —
and after step 5's `result.md` (`outcome: handed-off`), the predecessor runs the `archive`
act. To make that confirmation exact, a T3-lane **sibling** spawn prints one stdout line,
`record=<record dir> t3_thread_id=<id>`, before returning 0 (`spawn-claude.sh` already holds
`SESSION_RECORD_DIR` and the lane's `started thread_id=`; `_stamp_launched_record` writes
the display and `RUNNING` before `started` prints, so the line names a stamped record). Step
4 reads that line instead of today's "newest record carrying the title" lookup, which is
ambiguous because predecessor and successor share the title. If step 4 fails (non-zero exit, or no thread id — the
2026-11-01 silent-launch trap: a `t3_thread_id None` record means no thread), the
predecessor does **not** archive, writes `outcome: blocked`, `action_needed: queue
unwatched; successor spawn failed (<exit code / reason>)`, and stops. `skills/escalation-watcher/SKILL.md`
"Stopping" points at the procedure rather than restating it (INV-9).

## 5. Resolved design decisions

- **D1 — one control surface: MCP, hand-rolled in stdlib.** The lane runs under bare
  `python3` with `PYTHONPATH=orchestrator/src` (tier-2 of `TestStdlibOnlySelfContainment`),
  so the `mcp` package is not importable there. *Rejected:* `uv run --project orchestrator`
  (slow, venv path not derivable, breaks the no-venv contract hooks and `spawn-claude.sh`
  depend on); keeping the `t3` CLI for launch and adding MCP only for the missing bits (two
  auth paths, two surfaces to keep in step, heuristic 11); `t3 auth session issue --subject
  mcp-client` (code-read: lacks a `runtimeModeCeiling`, so it caps at `approval-required`,
  S3 §4; moot given the pairing flow, S6 AC3).
- **D2 — self-pair on the host, operator-pair elsewhere; failure is loud.** Same trust level
  as today: `t3 thread` already mints a full-access session with no approval on the host.
  *Rejected:* a browser `claude mcp login` flow (needs a human every 30 days); storing the
  token in env (scrubbed from children anyway, and leaks); reusing T3's secret store (its
  private files, the dependency this PRD retires); silently falling back to the CLI on an
  expired token (INV-11).
- **D3 — `skip_permissions` maps two-valued onto `runtimeMode`:** `true` → `full-access`
  explicitly (not the project default, matching `--dangerously-skip-permissions` on the
  terminal lanes), `false` → `approval-required`. *Rejected:* a new `CLAUDE_SPAWN_T3_RUNTIME_MODE`
  input for `auto`/`auto-accept-edits` (no caller asks; `RuntimeMode` literals exist in
  `packages/contracts/src/providerPolicy.ts` for when one does).
- **D4 — the `exited` write stays in `spawn-claude.sh`, through the registry `exit` verb,
  for every lane.** One choke point for the terminal transition (INV-6). *Rejected:* the
  lane's `wait` writing `exited` itself (a second writer of the same transition).
- **D5 — wait in 10-minute `t3_thread_wait` slices.** Caller-gone is noticed at slice end;
  a stray waiter blocks at most one slice. *Rejected:* keeping the 30 s `thread show` poll
  (the cost model task 7653 measured; MCP long-poll exists for this); 60-minute slices
  (an hour of a stray waiter after its caller died).
- **D6 — T3 remains the authority for "is this a project worktree".** The lane maps
  `assertProjectWorktree`'s failure to exit 145; it does not pre-list worktrees.
  *Rejected:* calling `t3_worktree_list` first (duplicates the check; two answers).
- **D7 — the thread sweep is indexed by the pointer dir and keyed by environment, not
  by cwd/project.** *Rejected:* adding `t3_project_id` to `Display` and matching per
  project (a second key for what one listing already answers); keeping the full `sessions/`
  scan (7653's measured cost).
- **D8 — pid liveness is qualified by a full `ProcessIdentity` per pid, not a record-level
  `host` alone.** One predicate for host, reboot and pid reuse (heuristic 10, uniform
  enforcement), already defined in `shared/src/shared/process_identity.py`. The int pid
  fields stay (slug derivation, pointer dir names, readers). *Rejected:* a bare `host`
  field (no reboot or pid-reuse immunity; a second liveness vocabulary); copying
  `process_identity` into the registry (the bootstrap already reaches stdlib-only `shared`
  leaves; SPOT).
- **D9 — new code lives in new modules.** `t3_mcp_client.py` and
  `shared/src/shared/session_liveness.py` (D16) are added; `session_registry.py` (~6,090 lines) gains only the pointer-dir helpers in the
  shape of the two existing pointer dirs and two record fields. No leaf splits
  `session_registry.py` (heuristics 13 and 14 bind: a split stitched back together with
  imports is the forbidden cheat; a real split is its own design question, out of scope).
- **D10 — what stays refused on T3.** Non-empty `CLAUDE_SPAWN_CLAUDE_ARGS` (raw `claude`
  args, including `--resume` crash recovery) is still exit 145; the five `CLAUDE_SPAWN_*`
  identity variables are still absent in a T3 thread (`self-record` and the trailer path
  remain the child's routes). Both are T3 facts (`launchArgs` are per provider instance;
  no caller-supplied per-thread env), not lane gaps.
- **D11 — `environmentId` is recorded, never used to launch remotely.** `cwd` must be a
  local worktree, `result.md` is a local file, hooks write a local registry; a thread in a
  linked environment (`t3_thread_launch` with `environmentId`) is attributable through
  `display.t3_environment_id` but not launched by this lane (study §3, §5.3).
- **D12 — the live acceptance probe is a human gate**, like task 7194: it needs the running
  T3 server and a human to read the T3 UI; the orchestrator's sandboxed agents cannot be
  assumed to reach `127.0.0.1:3773`.
- **D13 — end state is a per-launch input (`CLAUDE_SPAWN_T3_END`), with `SELF_EXIT=1` on T3
  as the alias for `archive`; default `none`.** *Rejected:* flipping the T3 self-exit default
  (ruling 7194/7947); overloading `CLAUDE_SPAWN_SELF_EXIT` with a third value (`settle` is
  not an exit); a registry-side auto-archive of finished T3 threads (nothing in dark-factory
  may organize Leo's threads for him — the act is the session's own).
- **D14 — the archiving session writes its own `exited` through the registry `exit` verb
  before the archive call.** INV-6: the terminal transition goes through the one choke
  point before the claimant goes away; the sweep stays a crash backstop. *Rejected:* relying
  on the sweep (the successor's launch sweep runs before the predecessor archives, so the
  record would stay non-terminal until a later lane event); archiving first and letting a
  Stop hook write it (no hook runs after the detach kills the process).
- **D15 — the handoff archive is gated on the successor's thread id, read from the record
  the spawn names on stdout, not on the exit code alone.** Sibling mode returns 0 after the
  started grace, and the 2026-11-01 trap produced rc 0 with no thread (the launcher died
  under `setsid` before `launch` ran). The stamp order is race-free (`_stamp_launched_record`
  runs inside `launch`, before `started` is printed), so the gate is exact once the spawn
  names its record. *Rejected:* "newest record carrying the title" (predecessor and
  successor share `escalation-watcher:<project_id> handoff`); retrying the spawn blind (two
  watchers racing one lease).
- **D16 — the liveness predicate lives in `shared`, not `orchestrator`** (Leo ruling,
  option A, 2026-11-04, esc-8125-3). `judge` is `shared/src/shared/session_liveness.py`, a
  registered `PURE_STDLIB_LEAVES` member beside `shared.process_identity`. Two reasons:
  `session_registry.py`'s header NOTE forbids `from orchestrator import <sibling>` (tier-1
  run-by-path containment, `TestStdlibOnlySelfContainment`), and ruling R17
  (`plans/review-all-program-dark_factory-2026-10-10.md`) moves the registry's stdlib-only
  parts to a shared session package. The legacy `None`-identity branch uses
  `session_registry.py::pid_alive`, passed in or kept at the call site. An unreadable
  `/proc` answers `ALIVE`, consistent with task 7283's planned shared
  `process_identity.liveness_or_unread` helper (unread → `None`, which `judge` maps to
  `ALIVE`). *Rejected:* `orchestrator/src/orchestrator/session_liveness.py` (the original
  M5 text: unimportable from the run-by-path registry; the G3 row checked only
  `shared.process_identity`'s reachability, not the new module's).

## 6. Contracts (H)

### 6.1 Lane verdict vocabulary (`spawn_t3_lane.py` ↔ `spawn-claude.sh`)

Unchanged words: `started`, `no-launcher`, `refused`, `launch-failed`, `timeout`,
`failed-to-start`, `finished code=<n>`, `t3`/`not-t3`, `slug=… parent=…`. Added fields and
reasons:

- `no-launcher reason=t3-mcp-unpaired | t3-mcp-credential-expired | t3-mcp-pair-failed |
  server-runtime-missing` (the last exists today). `spawn-claude.sh` exits 126 and names
  `spawn_t3_lane.py pair`.
- `pair-failed reason=pairing_rejected | cli-version-mismatch | descriptor-unreachable | …`
  (`cli-version-mismatch` carries `cli=<v> server=<v>`).
- `spawn-claude.sh` itself, T3 lane, sibling mode only: one stdout line
  `record=<record dir> t3_thread_id=<id>` before `exit 0` (today its stdout is empty; child
  mode stays silent).
- `refused reason=cwd-not-project-worktree detail=<project root>` replaces
  `cwd-not-project-root`; `refused reason=project-archived` for a `cwd` under an archived
  project.
- `started … bound=yes|no via=thread-pointer|transcript|none`.
- New verbs: `pair [--code C] [--origin URL]` → `paired environment_id=… expires_at=…
  access=full-access` or `pair-failed reason=…`; `status` → `paired|expired|unpaired
  environment_id=… expires_at=… origin=…`. `unpaired` and `expired` are distinct words
  (INV-13).

### 6.2 Credential file

`<fleet_root>/t3-mcp/<environment_id>.json`, mode 0600, written atomically
(`shared/src/shared/safe_io.py::atomic_write_text`): `{schema_version, origin,
environment_id, server_version, access, client_id, access_token, expires_at, paired_at,
paired_by: "self-pair" | "operator"}`. Readers: the lane only. A missing or unparsable file
is `unpaired`; `expires_at ≤ now + 1h` is `expired`.

### 6.3 `sessions-by-t3-thread/` pointer dir

`<fleet_root>/sessions-by-t3-thread/<thread_id>` holds one slug, same writer discipline as
`sessions-by-claude-id/`. Invariants: every non-terminal record with `display.kind == "t3"`
and a `t3_thread_id` has a pointer; a terminal record has none (deleted by the registry
`exit` verb, by `mark_t3_sessions_exited`, and by a pointer reap that removes entries whose
record is terminal or missing). Writers: the lane at stamp and reconcile; the hook when none
exists. The index is the sweep's iteration set.

### 6.4 Record fields and the liveness predicate

`SessionRecord` additive fields: `launcher_identity: ProcessIdentity | None`,
`claude_owner_identity: ProcessIdentity | None`; `Display.t3_environment_id: str | None`.
`SCHEMA_VERSION` bumps; old records read with `None`.
`shared.session_liveness.judge(identity, pid) -> Liveness`: `OTHER_HOST` when `identity.host !=
gethostname()`; `DEAD` on boot-id or start-ticks mismatch or an exited pid; `ALIVE`
otherwise; `None` identity → `ALIVE`/`DEAD` from `session_registry.py::pid_alive(pid)`
(passed in or at the call site, D16); an unreadable `/proc`
answers `ALIVE` (never evidence of death, `process_identity.ProcessIdentityUnreadable`).
Readers decide "dead" only on `DEAD`.

### 6.5 End-of-work input, record field and procedure

`CLAUDE_SPAWN_T3_END ∈ {none, settle, archive}` (unset = `none`; any other value exits 2;
`CLAUDE_SPAWN_SELF_EXIT=1` on the T3 lane reads as `archive`; `SELF_EXIT=1` + `settle` exits
2). `SessionRecord.t3_end: "settle" | "archive" | None` (additive), stamped by
`_run_launching` from the input the script resolved; `self_exit` stays `True` iff the
resolved end is `archive` on the T3 lane. The trailer's last paragraph for a T3-lane spawn
names the act and points at `skills/spawn/SKILL.md` §T3 lane "Ending a T3 session", the one
home of the four-step procedure (M6; the T3 tool shapes it relies on are pinned in §6.6). Observable end states: settle → `t3_thread_read.settled
= true`, record `idle`, `result.md` final (`FINISHED_NOT_EXITED`); archive →
`t3_thread_read.archived = true`, record `exited` with `exit_code 0`; refused → record
unchanged, `result.md` `action_needed` names the act and the failure; a settle whose final
turn did not complete → thread still in the inbox, record and `result.md` as for settle.

### 6.6 T3 contract pinned (0.1.6)

Tools and keys the lane sends or reads, each verified in §8: `t3_project_list{cursor, limit}`
→ `{projects: Project[], nextCursor}`; `t3_thread_launch{projectId, title, message,
modelSelection, runtimeMode, workspaceStrategy, clientRequestId}` → `{threadId, link,
projectId, runId, status}`; `t3_thread_read{threadId, limit}` → `thread:
OrchestratorMcpThreadDetail{status, latestRunId, archived, worktreePath, runtimeMode,
pendingRequestCount}`; `t3_thread_wait{threadId, runId?, timeoutMs}` → `{status, runId,
timedOut}`; `t3_thread_list{allProjects, cursor, limit}` → `{threads[].threadId, nextCursor,
total}`; `t3_environment_read{}` → `{environmentId, label, serverVersion}` (used by `pair`
to validate the token); `t3_thread_organize{action: "settle" | "archive"}` with `threadId`
omitted (the calling thread) → `{sequence}` or `{settlesWhenTurnEnds: true}`. Failures
arrive as `OrchestratorMcpFailure` results, never thrown.

## 7. Boundary-test sketch (H)

A stdlib fake T3 server (`http.server`) in `orchestrator/tests/` serves the OAuth endpoints
and `/mcp` with canned 0.1.6 shapes; the live probe (ε) is the only test against the real
server.

| # | Boundary | Test | Pass |
|---|---|---|---|
| B1 | client ↔ `/oauth/mcp/*` | pairing exchange against the fake (code read from the `credential` field of a canned `t3 auth pairing create --json` reply); wrong code; `expires_in` → `expires_at`; a fake `t3 --version` that differs from the descriptor's `serverVersion` | credential file 0600 with the right fields; `pair-failed reason=pairing_rejected`; `pair-failed reason=cli-version-mismatch` with nothing minted |
| B2 | client ↔ `/mcp` | `initialize` + `tools/call` with JSON and with SSE framing; 401 mid-wait | result decoded both ways; 401 → re-pair once, then `credential-expired` loud |
| B3 | `launch` ↔ project list | three pages, one archived, one deleted, `cwd` under a `.worktrees/<id>` of the live one | `existing_worktree` strategy and `runtimeMode` in the sent input; archived → `refused reason=project-archived` |
| B4 | `launch` ↔ `assertProjectWorktree` | fake returns `invalid_request` (a) to a launch that sent `existing_worktree`, (b) to one that sent `root` | (a) `refused reason=cwd-not-project-worktree detail=<T3 message>`, exit 145, nothing stamped; (b) `launch-failed` with the message, exit 127 — no message-text matching in either |
| B5 | `wait` ↔ `t3_thread_wait` | slices: `timedOut` twice, then `completed` with final `result.md`; `failed` with no result for > `ENDED_RUN_GRACE_SECS`; transport down > `UNREACHABLE_GRACE_SECS`; caller gone between slices | codes 0 / 1 / 126; `CallerGone`; the registry `exit` verb called once with the code by the script (`tests/scripts/test_spawn_claude.py`) |
| B6 | hook ↔ pointer dir | SessionStart with `T3CODE_THREAD_ID` and (a) a spawn pointer, (b) no pointer, (c) a nested claude inheriting the variable after binding | (a) adopts and binds the spawn record; (b) creates a `display.kind: t3` hand record and writes the pointer; (c) forks its own record |
| B7 | lane reconcile ↔ hook race | hook wrote the pointer first | spawn record bound to the hand record's sid, hand record folded (`FOLDED_EXIT_CODE`), pointer re-pointed |
| B8 | sweep ↔ index | 3 pointers, listing lacks one, one record has a foreign `t3_environment_id` | exactly one record marked `exited`, the foreign one untouched, no read of `sessions/*` outside the index |
| B9 | liveness ↔ readers | records with identity on this host (alive; dead by start ticks; dead by boot id), on another host, and `None` | sweeps mark only `DEAD`; `is_over` false for `OTHER_HOST`; self-exit refusal `other-host`; `watch_sessions.py` line carries `host=` / `host=unqualified` |
| B10 | script ↔ end-of-work input | T3-lane spawns with `T3_END` unset / `settle` / `archive` / `SELF_EXIT=1` / `SELF_EXIT=1`+`settle` / junk; a tmux-lane spawn with `T3_END=archive` | `record.t3_end` and `self_exit` as §6.5; trailer names the act and the procedure; exit 2 for the conflict and the junk value; one stderr line and no stamp on the tmux lane (`tests/scripts/test_spawn_claude.py`) |
| B11 | handoff ↔ confirmation gate | a scripted handoff rehearsal in `tests/scripts/test_spawn_claude.py`'s fake-lane harness: a T3-lane sibling spawn whose stdout names a record that is (a) stamped with a thread id, (b) `display: null`; (c) spawn exit non-zero; (d) two same-titled records, the older unbound | (a) the predecessor reads the record the spawn named, calls the `exit` verb on its own record, then the archive instruction is the final action; (b) and (c) no archive, `result.md` `outcome: blocked` with `queue unwatched`; (d) the named record, not the newest-by-title, decides |

## 8. Pre-conditions and substrate verification (G3)

All verified 2026-11-01 by re-opening the symbol; none is queued as a prerequisite.

| Assumed capability | Evidence |
|---|---|
| Outside OAuth client may launch threads without a thread scope | `apps/server/src/mcp/McpToolAccess.ts::startsThreads` needs `writingCaller` only; `project/handlers.ts` branches on `caller === undefined` to project defaults |
| `t3_thread_launch` takes `runtimeMode`, `workspaceStrategy.existing_worktree`, `clientRequestId`, `modelSelection` | `apps/server/src/mcp/toolkits/project/tools.ts::ThreadLaunchTool` |
| `.worktrees/<id>` qualifies as a project worktree | `project/handlers.ts::assertProjectWorktree` realpath-compares against `git worktree list` from `workspaceRoot` (S6 "Checked and confirmed") |
| `t3_project_list` returns `workspaceRoot`, `archivedAt`, `deletedAt` | `project/tools.ts::ProjectListTool`; `packages/contracts/src/project.ts::Project` |
| `t3_thread_wait` default 10 min, max 60 min | `apps/server/src/mcp/OrchestratorMcpService.ts` `DEFAULT_WAIT_TIMEOUT_MS`, `MAX_WAIT_TIMEOUT_MS`; input `packages/contracts/src/orchestratorMcp.ts::OrchestratorMcpThreadWaitInput` |
| `t3_thread_read` detail carries `archived`, `worktreePath`, `status`, `latestRunId` | `orchestratorMcp.ts::OrchestratorMcpThreadDetail` |
| `t3_thread_list` lists active threads only, `allProjects` paged | `apps/server/src/orchestration-v2/ThreadManagementService.ts` (`location: "active"`); `orchestratorMcp.ts::OrchestratorMcpThreadListInput` |
| `t3_environment_read` → `environmentId` | `apps/server/src/mcp/toolkits/environment/tools.ts::EnvironmentReadTool` |
| Unauthenticated descriptor with `environmentId` | `packages/contracts/src/environmentHttp.ts` `descriptor` (`/.well-known/t3/environment`) |
| Pairing-code OAuth exchange, no browser | `environmentHttp.ts` `mcpOAuth` group (`register`, `decision`, `token`); `apps/server/src/auth/McpOAuth.ts::approve`, `::exchangeCode`; `apps/server/src/peer/PeerLinks.ts::signIn` |
| 30-day session, ceiling = access | `apps/server/src/auth/EnvironmentAuth.ts` `MCP_CLIENT_SESSION_TTL`, `::issueMcpClientSession` |
| `t3 auth pairing create --json` emits the code as `credential` | `apps/server/src/cli/auth.ts::pairingCreateCommand` (flags `scopes`, `ttl`, `label`, `json`); `apps/server/src/cliAuthFormat.ts::formatIssuedPairingCredential` |
| `assertProjectWorktree` fails with the shared code `invalid_request` | `apps/server/src/mcp/toolkits/project/handlers.ts::assertProjectWorktree` (`OrchestratorMcpFailure{code: "invalid_request"}`; the same code serves other refusals in the file) |
| Self-settle applies only when the run completes | `apps/server/src/orchestration-v2/ThreadManagementService.ts::settleAfterRun` (`run.status === "completed"`) |
| A detached provider session ends the live run `failed` (provider turn `interrupted`) | `apps/server/src/orchestration-v2/ProviderSessionManager.ts::detach` → `::releaseEntry` (`manual_shutdown`) → `::failSubscribers` → `apps/server/src/orchestration-v2/RunExecutionService.ts::makeFailedTerminalEvent` → `::writeFinalRunEvents`; `apps/server/src/orchestration-v2/ThreadManagementService.ts::isTerminalRunStatus` (`completed`, `failed`, `cancelled`, `interrupted`, `rolled_back`) → `::waitForThread` settles |
| `execution_class` vocabulary for the human gate | `docs/task-authoring.md` §"`execution_class` routes to a HUMAN" (`operational`, `decision`); task 7194's filed shape |
| `T3CODE_THREAD_ID` set on every Claude launch and resume | `apps/server/src/preview/PortScanner.ts::AGENT_THREAD_ENV`; `ClaudeAdapterV2.ts::ClaudeAgentSdkQueryRunner` `open` |
| Hooks fire inside T3 threads | measured: tasks 7614 and 7617 (hand-launched T3 records on the live fleet root, 2026-10-27) |
| `server-runtime.json` carries `origin`, `pid`, `startedAt`, `serverVersion` | read on this host 2026-11-01 (`{"origin": "http://127.0.0.1:3773", …}`) |
| `shared.process_identity` reachable from bare-python3 registry consumers | `session_registry.py` module-scope `shared/src` bootstrap; `TestStdlibOnlySelfContainment` tier rules. Corrected 2026-11-04 (esc-8125-3): this row verified only `shared.process_identity`; the original M5 claim that a new `orchestrator/src/orchestrator/session_liveness.py` was reachable the same way was wrong — the bootstrap reaches `shared` leaves, and the header NOTE forbids orchestrator-sibling imports. The new module therefore lives at `shared/src/shared/session_liveness.py` (D16), reachable by this same row |
| `t3_thread_organize` actions `archive`, `settle` (and `pin|unpin|snooze|unsnooze|unsettle|unarchive|mark_unread`), `threadId` omitted = the calling thread | `apps/server/src/mcp/toolkits/thread/tools.ts::ThreadOrganizeTool`; `thread/handlers.ts` `t3_thread_organize` (dispatches `thread.<action>`; `settle` → `settleThread({byOwnAgent})`) |
| A thread may organize itself; the mode-limit check skips the caller's own thread | `apps/server/src/mcp/McpToolAccess.ts::writesThreads` (`threadId === caller.scope.thread?.threadId` → continue) |
| Self-settle with an active run defers to turn end | `apps/server/src/orchestration-v2/ThreadManagementService.ts::settleThread` (`settlesWhenTurnEnds: true` via `settleAfterRun`); "Settlement rejects active or blocked runs" (`Orchestrator.ts` dispatch comment) |
| Archive refuses only an already-archived thread; cancels `queued`/`preparing` runs; detaches provider sessions with `revokeMcpCredential: true`; `terminal.cleanup`; does not finalize a `running` run itself | `apps/server/src/orchestration-v2/Orchestrator.ts` (`thread.archive` branches: "is already archived", `activeRunIds` filter, `detachSessionIds`, `terminal.cleanup` effect) |
| Settled threads stay in the active listing unless filtered | `apps/server/src/mcp/OrchestratorMcpService.ts` (`input.settled === undefined ||` …); `t3_thread_list.settled` optional |
| Registry `exit` verb takes `--record <dir> --code <int>`; a hook refresh revives a wrongly terminal record | `session_registry.py` argparse `exit`; `::mark_orphaned_sessions_exited` docstring ("self-correcting -- its next hook event calls refresh_record and flips status back") |
| `handoff` procedure and its step-4 confirmation | `skills/spawn/SKILL.md` §`handoff` (`94d33dba75`); `skills/escalation-watcher/SKILL.md` "Stopping", "Launching this watcher" |
| MCP tool names in the live 0.1.6 session | this authoring session's own `mcp__t3-code__*` tool list carries `t3_thread_wait`, `t3_project_list`, `t3_environment_read`, `t3_thread_launch`, `t3_thread_list`, `t3_thread_read`, `t3_thread_organize`, `t3_thread_interrupt` — the 0.1.6 names, none of the renamed `t3_thread_fork/merge_back/configuration`, `t3_pending_request_*`, `t3_queue_*` |

Unverified at authoring, resolved by the fake-server tests and ε: the exact streamable-HTTP
framing Effect's MCP transport uses (JSON vs SSE per response) and whether it requires the
`Accept` header. Both are implementation-time facts the client must handle either way.

## 9. Cross-PRD and cross-task seams (G4)

| Seam | Owner | This PRD |
|---|---|---|
| Cockpit T3 focus backend, `DisplayTarget.t3_thread_id`, no silent default-to-wm | task 7651 (pending) | consumer of `display.t3_thread_id` and `display.t3_environment_id` (v2 link `t3-thread://v2/<env>/<thread>`, `packages/shared/src/threadLinks.ts`); decompose adds 7651 → depends on γ |
| Harness startup trigger for the T3 sweep | task 7651 item 2 | γ makes the sweep cheap (indexed) so a startup trigger is affordable; 7651 keeps the trigger |
| `mark_t3_sessions_exited` full-registry scan cost | task 7653 (pending) | γ dissolves it (index + long-poll); decompose adds 7653 → depends on γ and recommends close-as-superseded once γ lands |
| Hand-launched T3 records marked EXITED by the pid sweep | task 7617 (pending) | γ's `display.kind: t3` stamping is 7617's "display-tag the record" direction; decompose adds 7617 → depends on γ, re-evaluate then |
| Pid reuse defeats `_reclaim_dead_owner_pid` | task 7654 (pending) | δ's `claude_owner_identity.start_ticks` closes it for records written after δ; 7654 stays for legacy records |
| Shared `claude agents` host pid collapsing pointers | task 6945 (pending) | untouched; `is_shared_session_host` keeps its role |
| Self-exit lane defaults and trailer | task 7947 (done), `plans/review-all-program-tooling-prd.md` | preserved; the T3 archive trailer unchanged |
| `/spawn` contract doc, callers matrix | `skills/spawn/SKILL.md` (its `handoff` section landed separately, `94d33dba75`, 2026-11-01) | β edits §T3 lane, §Verification rows for 126/145, and §Arguments `skip_permissions`; ζ adds the `CLAUDE_SPAWN_T3_END` argument, the "Ending a T3 session" subsection and `handoff` step 6 + failure path; nothing else in `handoff` changes |
| Watcher lifecycle text ("Stopping", "Launching this watcher") | `skills/escalation-watcher/SKILL.md` | ζ adds one pointer to the "Ending a T3 session" procedure and the handoff archive gate; no restatement |
| Review-all seat spawns (`CLAUDE_SPAWN_SELF_EXIT=1`) | `skills/review-all/scripts/program/gen_brief.py` | unchanged: `SELF_EXIT=1` on T3 reads as `archive` (D13) |
| Watcher leases (`LeaseHolder.pid`, `reap_stale_leases`) | registry (unchanged) | out of scope (§11); the predicate is available to a later leaf |
| Remote headless agent leg, peer-link delegation | future PRD per study §5.3 | none; D11 |
| `OTHER_HOST` liveness branch — its first producer (a second host writing records) | future multi-host PRD (study §5.3) | δ lands the comparison and B9 proves it; the producer, a shared or synced fleet root, is that PRD's |
| fused-memory `0.0.0.0:8002` bind | PRD B, `plans/mcp-network-exposure-prd.md` | none |
| T3 checkpoint refs accumulating in the repo | no task filed (study §4.1, §5.5 asks upstream); task 8076 is the git-maintenance timer only | none |

## 10. Decomposition plan

Sizing follows the overlay bands (300–1,500 LOC, ≤10–12 files; >15 is a review trigger).
Every leaf is reviewed against `docs/code-quality.md`; the heuristics named per leaf are the
ones that bind, cited by name only.

**α — `t3_mcp_client.py`: stdlib MCP client, pairing exchange, credential file, `pair`/`status` verbs.**
Files: `orchestrator/src/orchestrator/t3_mcp_client.py` (new), `orchestrator/src/orchestrator/spawn_t3_lane.py`
(verbs only), `orchestrator/tests/test_t3_mcp_client.py` (new, with the fake server),
`orchestrator/tests/test_spawn_t3_lane.py`. ~500–700 LOC. Deps: none.
Signal: on the T3 host, `spawn_t3_lane.py pair` prints `paired environment_id=<uuid>
expires_at=<iso> access=full-access` and writes `<fleet_root>/t3-mcp/<uuid>.json` mode 0600;
`status` prints `paired`; with the file removed `status` prints `unpaired`; with `expires_at`
in the past it prints `expired`. B1, B2.
Heuristics: 9 (one deep module, narrow `call(tool, args)` interface), 12 (structured
JSON-RPC and credential records, no string parsing of CLI output), 10 (every transport
failure is a typed result), 13.

**β — lane verbs over MCP; join writes `exited`; refusals narrowed.**
Files: `spawn_t3_lane.py`, `skills/spawn/spawn-claude.sh`, `skills/spawn/SKILL.md`,
`plans/spawn-t3-lane-design-2026-10-25.md` (§8 appendix), `orchestrator/tests/test_spawn_t3_lane.py`,
`tests/scripts/test_spawn_claude.py`. ~800–1,100 LOC (net of the deleted CLI/sqlite code).
Deps: α.
Signal: from a T3-hosted spawner, `spawn-claude.sh /home/leo/src/dark-factory/.worktrees/<id>
false '<title>' '<prompt>'` exits without 145; the thread's `t3_thread_read` detail shows
`worktreePath` = that path and `runtimeMode: approval-required`; when the child writes a
final `result.md`, the join returns 0 and `record.json` reads `status: exited, exit_code: 0`;
`CLAUDE_SPAWN_CLAUDE_ARGS=--resume x` still exits 145; `import sqlite3` and `run_t3` are gone
from the module. B3, B4, B5.
Heuristics: 3 (lane input → launch request is one mapping, runtime mode and workspace
strategy orthogonal), 2 (the wait loop is slices + the unchanged `JoinTracker`), 11 (T3
owns worktree validity; the `exit` verb owns the terminal write), 14 (`spawn_t3_lane.py`
must not grow past today's size net; the client is α's module).

**γ — identity by `T3CODE_THREAD_ID`: pointer index, hook resolution, lane reconcile, indexed env-qualified sweep.**
Files: `orchestrator/src/orchestrator/session_registry.py` (pointer-dir helpers, `Display.t3_environment_id`,
`mark_t3_sessions_exited`, `exit` verb pointer delete), `orchestrator/src/orchestrator/session_hooks.py`,
`spawn_t3_lane.py`, `docs/session-hook-ownership.md`, `orchestrator/tests/test_session_hooks_ownership.py`,
`orchestrator/tests/test_session_registry.py`, `orchestrator/tests/test_spawn_t3_lane.py`.
~500–700 LOC. Deps: β, δ (both edit `session_registry.py` and `session_hooks.py`; δ lands
first so γ's sweep uses the predicate).
Signal: a T3-lane spawn's `await-started` verdict reads `bound=yes via=thread-pointer`
before any transcript matches; a thread opened by hand in the T3 UI has a record with
`display.kind: t3` and a `sessions-by-t3-thread/<id>` entry; archiving that thread in the UI
and running any T3-lane `launch` flips the record to `exited` with a log line reporting the
index size scanned (not the `sessions/` count). B6, B7, B8.
Heuristics: 11 (the pointer dir is the one index of live T3 records; the sweep has one key),
5 (the hook reads the record once per event, as today), 13/14 (registry additions follow the
existing pointer-dir shape; no new block of logic in `session_registry.py`; no split).

**δ — host-qualified liveness.**
Files: `shared/src/shared/session_liveness.py` (new, D16), `shared/tests/test_pure_stdlib_leaves.py`
(register it in `PURE_STDLIB_LEAVES`), `session_registry.py`
(fields, stamp in `_run_launching`, three readers), `session_hooks.py` (stamp at bind and
reclaim, new-record path), `orchestrator/src/orchestrator/session_self_exit.py`,
`skills/review-all/scripts/program/watch_sessions.py`, `shared/tests/test_session_liveness.py`
(new), `orchestrator/tests/test_session_registry.py`, `orchestrator/tests/test_session_hooks_ownership.py`,
`tests/scripts/test_review_all_program_watch_sessions.py`. ~500–700 LOC.
Deps: none (parallel with α/β).
Signal: a new spawn's `record.json` carries `launcher_identity: {pid, start_ticks, boot_id,
host: "<this hostname>"}` and, once bound, `claude_owner_identity`; `watch_sessions.py`'s
per-session line shows `host=<hostname>` for new records and `host=unqualified` for old
ones; after the next host reboot, `session_registry.py reap` marks every pre-reboot
non-terminal record `exited` by boot id (observable in the reap's output). The `OTHER_HOST`
branch (sweeps skip with reason `other_host`, `is_over` false, self-exit refusal
`other-host`) has no producer on this host and is proven by B9 alone; its consumer is the
future multi-host PRD (§9 seam). B9.
Heuristics: 10 (one predicate, uniformly applied at every pid decision), 11 (identity is
defined once in `shared.process_identity`), 8 (`ProcessIdentity` is frozen), 9 (one small
module, two functions), 14 (no growth of `session_registry.py` beyond fields and call-site
swaps).

**ζ — T3 end-of-work acts and the watcher's handoff archive.**
Files: `skills/spawn/spawn-claude.sh` (`CLAUDE_SPAWN_T3_END` resolution, conflict check,
trailer, stderr note on other lanes, the sibling-mode `record=… t3_thread_id=…` stdout line),
`orchestrator/src/orchestrator/session_registry.py`
(`t3_end` field, stamp in `_run_launching`), `skills/spawn/SKILL.md` (§Arguments entry,
§T3 lane "Ending a T3 session", §`handoff` step 6 and failure path), `skills/escalation-watcher/SKILL.md`
("Stopping" pointer), `tests/scripts/test_spawn_claude.py`, `orchestrator/tests/test_session_registry.py`.
~300–500 LOC including prose. Deps: β (trailer and §T3 lane text), γ (the `exit` verb's
pointer deletion and the indexed sweep that backstops an archive).
Signal: a T3-lane spawn with `CLAUDE_SPAWN_T3_END=settle` whose session finishes is out of
Leo's inbox with `t3_thread_read.settled = true`, its record `idle`, `result.md` final, and
`watch_sessions.py` standing `finished, not exited`; one with `archive` has
`t3_thread_read.archived = true` and its record `exited`, `exit_code 0`, with no later lane
event needed; after "handoff" from a T3-hosted watcher, `t3_thread_list` shows one
`escalation-watcher:<project_id>` thread for that queue, the old record is `exited` and the
record the spawn named on stdout is `running` with `display.t3_thread_id`; with the
successor spawn broken (no thread id), the old thread is still listed and its `result.md`
reads `outcome: blocked`, `action_needed: queue unwatched …`. B10, B11.
Heuristics: 11 (one procedure in one skill section; the trailer and the watcher point at
it), 12 (the end state is an enum input and a record field, not a phrase parsed from the
trailer), 10 (the conflict and the junk value refuse at the script boundary), 6 (the act is
the session's; the registry only records what was asked and what happened).

**ε — live acceptance probe (human gate, filed as task 7194 was: `metadata.execution_class: "operational"`, `task_kind: "deterministic"`, `always_escalates: true`, `files: []` — `docs/task-authoring.md` §"`execution_class` routes to a HUMAN").**
Deps: ζ (hence α, β, γ, δ). No code. Sequence, recorded in the task's details with timestamps:
(1) `pair` self-pairs; (2) a T3-hosted spawner launches a child-mode spawn into
`.worktrees/<id>` with `skip_permissions=false`; (3) the T3 UI shows the thread in that
worktree asking for approval; (4) `record.json` is bound via `thread-pointer` before the
thread's first reply; (5) the child writes a final `result.md`; the join exits 0 and the
record reads `exited`; (6) the record carries both identities; (7) a hand-opened thread gets
a `display.kind: t3` record and, after archiving, flips to `exited` on the next lane launch;
(8) `status` after deleting the credential prints `unpaired`, and the next launch self-pairs;
(9) a spawn with `CLAUDE_SPAWN_T3_END=settle` ends out of the inbox, record `idle`,
`result.md` final; (10) one with `archive` ends archived with its record `exited` before any
later lane event; (11) a T3-hosted `/escalation-watcher` told "handoff" leaves exactly one
live watcher thread for the queue, its own record `exited`, the successor's `running` with
a thread id; (12) the same with the successor spawn deliberately broken (a `cwd` the lane
refuses) leaves the old thread live and its `result.md` at `outcome: blocked`.
Signal: all twelve observed on the live server and written into the task.

Dependency DAG: α → β → γ → ζ → ε; δ → γ. Decompose adds 7651, 7653, 7617 → depends on γ
(§9). The integration gate is ε (C-as-integration-gate); α–δ and ζ each also carry their own
signal.

## 11. Out of scope

- Lane selection is unchanged. A watcher's lane is the one its skill names — a T3 thread
  when the operator launches it there, the tmux lane from a plain terminal
  (`skills/escalation-watcher/SKILL.md` §"Launching this watcher"); `/do` plan-mode handoffs
  stay off T3 (ExitPlanMode is denied there, study §2).
- Self-exit defaults (default-on tmux/terminal, opt-in T3; task 7947) are unchanged; the T3
  archive trailer becomes the `archive` case of `CLAUDE_SPAWN_T3_END` with the same text.
- T3's own auto-settle (3 days, per project) and any registry-side organizing of threads on
  Leo's behalf: dark-factory never settles or archives a thread the session did not ask to.
- Decision A spawner rule and `CLAUDE_SPAWN_BACKEND=auto|t3|tmux|wm` unchanged; exit 145
  remains the refusal code.
- Raw `claude` args / `--resume` crash recovery and the five identity env vars on T3: still
  refused / absent (D10).
- A host key on the orchestrator fleet heartbeat (§3: no consumer can honestly be named).
- Watcher leases' pid liveness (`LeaseHolder`, `reap_stale_leases`).
- Sharing the fleet root across hosts, launching into a linked environment, peer-link
  delegation, a remote headless agent leg (study §5.3).
- Splitting `session_registry.py`; a cockpit T3 focus backend (task 7651); the T3
  checkpoint refs (study §4.1; no task filed); fused-memory's `0.0.0.0:8002` bind (PRD B,
  `plans/mcp-network-exposure-prd.md`); `delegate_task` for agent-initiated handoffs (study
  §5.6).
- Spawning one at a time in the foreground from a T3 session stays a caller rule
  (memory 2026-11-01); this PRD does not add serialisation to the script.

## 12. Open questions (tactical)

1. Wait-slice length: 10 min (server default) or 5 min to halve a stray waiter's life
   after `CallerGone`; either keeps the design.
2. Credential re-pair horizon: one hour before `expires_at`, or at the first 401 only.
3. The lane's OAuth `software_id` / `client_name` literal (e.g. `dark-factory-spawn-t3-lane`),
   visible in T3's Settings → Sharing & access.
4. Whether `status` also probes the descriptor (reports `server_version` drift) or stays a
   pure file read.
5. The exact SSE/JSON framing Effect's streamable-HTTP transport emits for `tools/call`;
   the fake server mirrors whatever ε observes.

## 13. Gate notes

- **G6.** Numeric claims cite their source constants (10 min / 60 min, 30 days) or a dated
  measurement (134,926 dirs / ~150 s, task 7653, 2026-10-28; ~100 s thread preparation,
  design doc §7 pilot). Rejection claims bind a mechanism: `cwd-not-project-worktree` fires
  from `assertProjectWorktree`'s failure; `project-archived` from the lane's own filter on
  `archivedAt` (T3's listing admits archived projects, so the lane must filter — premise
  verified in `ProjectStore.ts`). Field-population: `display.t3_environment_id` is written
  by the lane from the descriptor on every launch, and `launcher_identity` by
  `_run_launching`, both on the production path. End-of-work claims: "archive is terminal
  for the join" rests on `OrchestratorMcpThreadDetail.archived`, not on the run's status
  (the detach ends the run `failed`, which is terminal, but the lane does not depend
  on it); "settle is not terminal" rests on settled threads staying in the active listing,
  and "settle applies" only when the final turn completes (`settleAfterRun`); "a thread may archive itself" rests on
  `writesThreads` skipping the caller's own thread; the handoff gate's "no thread id means no
  thread" rests on the 2026-11-01 measured trap (memory
  `feedback_spawn_t3_concurrent_or_setsid_launches_silently_fail`).
- **G7 (advisory walk).** INV-6: the terminal write has one choke point (registry `exit`
  verb), used by the child join and by the archiving session itself; sweeps are backstops. INV-7: every hold is owned and bounded (wait slices + caller
  liveness; the credential's `expires_at` with a named re-pair owner). INV-11: expiry, 401,
  an incomplete project listing and transport failures each refuse or surface a named
  reason; nothing falls back to the CLI. INV-13: `status` distinguishes `unpaired` from
  `expired`; readers print `host=unqualified`; the sweep's log reports the index size.
  INV-9/INV-5: runtime-mode mapping, worktree validity and liveness each have one home.
  INV-12: no new suppressions; `shared.process_identity` needs no containment exception.
  INV-1: the fake server pins the exact tool names and input keys, so a future T3 rename
  fails B3/B5 rather than a live launch.
- **META.** No design question is open: transport, auth, lane mapping, identity, sweep,
  liveness, the end-of-work acts and the handoff gate, and their homes are decided with rejected alternatives recorded; the remaining
  items are tactical (§12).
