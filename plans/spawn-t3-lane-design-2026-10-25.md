# /spawn T3 lane — design proposal

Date: 2026-10-25. Status: IMPLEMENTED 2026-10-26 under task 7489 (branch
`task/7489`), decision A; §8 records where the implementation departs from
this text. v2 after one critic pass (21 findings, all anchored in source;
the ones that changed the design are marked `[critic]`). Author: a `/team` lead session running
inside T3 Code (thread 217ba20e). Source of truth for T3 behaviour:
`~/src/t3code` at tag `v0.1.2`, the version installed as `t3code.service`
on the workstation.

## 1. The ask

`/spawn` opens a fresh interactive `claude` in a new terminal window. Leo
now drives sessions through T3 Code from the laptop, remotely operating
the workstation. He wants: when the spawner is a T3 session, the spawn
should be a T3 thread; when the spawner is a plain terminal, a terminal.

## 2. What exists today (`skills/spawn/spawn-claude.sh`)

One axis already varies *where the session lives*: a window-manager
emulator (discovered, or `$CLAUDE_TERMINAL_CMD`), the tmux lane
(`CLAUDE_SPAWN_BACKEND=tmux`), macOS Terminal. Every lane shares one
payload (`bash -c "$inner"`) that exports five `CLAUDE_SPAWN_*` vars and
runs `claude --dangerously-skip-permissions [--model …] "<prompt+trailer>"`.
Everything else keys off that payload:

- registry: `session_registry.py launching` writes the record before
  launch; the child's SessionStart/Notification/Stop hooks adopt it through
  `CLAUDE_SPAWN_SESSION_ID` in the child's environment; liveness sweeps key
  on `launcher_pid` (the script's own pid) and a 1 h heartbeat TTL, and
  `reap_stale_records` removes a terminal record dir after 24 h;
- join: child mode blocks until the process exits (exit code is liveness
  only); sibling mode returns at once and re-parents the child to the
  spawner's parent; `result.md` is the semantic channel;
- started-watchdog (exit 144): a new transcript under
  `~/.claude/projects/<encoded-cwd>/` within grace.

## 3. What T3 offers, measured against the source and one live pilot

| Fact | Where it comes from |
|---|---|
| Two launch surfaces: MCP `t3_thread_launch` (inside a T3 thread) and CLI `t3 thread start [--title] [--provider --model] [--worktree <ref> --branch] [--wait] [--json] <project-id-or-path> <message or ->` | `apps/server/src/cli/thread.ts`, `apps/server/src/mcp/toolkits/project/tools.ts` |
| CLI signs in as an owner-level MCP client (no pairing code involved); a thread it starts gets the **project's default runtime mode**; the CLI has no runtime-mode flag | `thread.ts::withThreadTools`, `toolkits/project/handlers.ts` (`projectDefaults`) |
| MCP launch inherits the caller's modes and may not exceed them; takes `runtimeMode`, `interactionMode`, `modelSelection`, `workspaceStrategy` incl. `existing_worktree` (must be one of the project's git worktrees) | `tools.ts`, `handlers.ts::assertProjectWorktree` |
| Runtime modes are `approval-required`, `auto-accept-edits`, `auto`, `full-access`; full-access → bypassPermissions **unless** Claude settings disable bypass, in which case it is downgraded `[critic F13]` | `packages/contracts/src/providerPolicy.ts::RuntimeMode`, `ClaudeAdapterV2.ts` (`claudeRuntimeQueryPolicyForRuntimePolicy`, `bypassDisabled`) |
| The claude child's environment is the T3 server's `process.env` merged with a **per-provider-instance** overlay, plus `CLAUDE_CONFIG_DIR` when a home path is set. Per-instance `launchArgs` exist too. **Nothing is per thread** `[critic F14]` | `ClaudeAdapterV2.ts::createClaudeAdapterV2` (`mergeProviderInstanceEnvironment`), `provider/Drivers/ClaudeHome.ts::makeClaudeEnvironment`, `packages/contracts/src/settings.ts::ClaudeSettings` |
| Model aliases (`haiku`, `opus`, `sonnet`, `fable`) resolve through the catalog at turn time; unknown ids pass through to claude `[critic F4]` | `provider/ClaudeModelCatalog.ts::resolveClaudeCatalogModel`, `provider/model-manifest.json` |
| T3 allocates the Claude `--session-id` itself at `ensureThread`; it is **not exposed** by `t3_thread_read`, `thread show --json` or the launch result | `ClaudeAdapterV2.ts::ensureThread`, `packages/contracts/src/orchestratorMcp.ts::OrchestratorMcpThreadDetail` |
| `--wait --json` prints `{threadId, runId, status, reply, error}` (or the launch shape `{threadId, projectId, runId}` when no run started); exit 1 if the run failed, 2 if it stopped to ask; polls `t3_thread_wait` in 15 s slices | `thread.ts::finishRun`, `WAIT_SLICE_MS` |
| A thread never exits. Its provider process is released after 30 min idle; release is deferred while background tasks or subagents are pending, up to about 4 h of **silence**, reset by provider activity; background Bash dies with the process `[critic F15]` | `ProviderSessionManager.ts::releaseIfStillIdle`, `ClaudeAdapterV2.ts` (`hasPendingBackgroundWork`, `backgroundTasksEndWithProcess: true`) |
| The Claude session id of a thread is **not stable**: `/clear` and compaction re-mint it in place; T3 compacts idle threads on the user's return by default (`idleCompaction: onReturn`); a failed resume binds a fresh native session with a context handoff `[critic F1]` | `docs/session-hook-ownership.md` §Session-id re-mints, `~/.t3/userdata/settings.json`, `ClaudeAdapterV2.ts::hasEmptyClaudeTranscript` |
| Settlement is a sidebar flag after 3 inactive days (or PR merge); it does **not** archive `[critic F8]` | `ThreadSettlementService.ts::resolveAutoSettlementAt`, `settings.ts::DEFAULT_SIDEBAR_AUTO_SETTLE_AFTER_DAYS` |
| Completion of a top-level launched thread is delivered to nobody that I could find; delivery code exists only for subagent lineage (not proven exhaustively) | `ProviderContinuationService.ts`, `relationshipToParent: "fork" \| "subagent"` |
| Message ceiling 120 000 chars | `t3_thread_launch` schema |

Pilot (2026-10-25 08:04Z, from inside this T3 session, CLI lane, Haiku 5.5):

| Measured | Value |
|---|---|
| `t3 thread start --json --wait` wall time for a one-line turn | 103 s |
| Child saw `CLAUDE_CODE_SESSION_ID` | yes |
| Child saw `CLAUDE_CODE_ENTRYPOINT` | `sdk-ts` |
| Child's claude parent pid | 881190 = `pid` in `~/.t3/userdata/server-runtime.json` |
| Child tty | none |
| Registry record written by the hooks | `session-dark-factory-<sid>`, role `session`, `display: null`, `parent_session_id: null`, `result_file: null` |

So the hooks already run for a T3-hosted session and key it as
hand-launched. What is missing is everything the five env exports carried:
the spawn's identity, parent, result file and display. The spawner, when
it is itself a T3 thread, is in the same position: it has no
`CLAUDE_SPAWN_SESSION_ID` of its own to hand down `[critic F2]`.

## 4. Design

### 4.1 Backend selection

`CLAUDE_SPAWN_BACKEND` becomes the single vocabulary for *where the
session lives* `[critic F10]`: `auto` (the default when unset), `t3`,
`tmux`, `wm`. A file `<fleet_root>/spawn-backend` (resolved through
`session_registry.fleet_root`, so `CLAUDE_FLEET_ROOT` is honoured) may
hold a default for the variable; the variable, when set, wins. `auto`
means: `t3` iff the spawner is T3-hosted, else emulator discovery as today.

T3-hosted means all of `[critic F11]`: some ancestor pid of the script
equals the `pid` in `$T3CODE_HOME/userdata/server-runtime.json` (default
`~/.t3/userdata`); that pid's start time is consistent with the file's
`startedAt` (T3 guards pid reuse the same way in `serverOwnership.ts`);
and the script has no controlling tty. The tty test keeps a plain `claude`
run inside a T3 integrated-terminal pane on the terminal lane.
`CLAUDE_CODE_ENTRYPOINT=sdk-ts` is only a hint. The detector lives in the
script alone (SKILL.md keeps: never re-implement discovery).

The file is a scope decision for Leo, not part of the stated rule: with
`t3` in it, the tmux-hosted escalation-watcher's `/unblock` spawns, which
exist for Leo to drive, appear as T3 threads he can reach from the laptop;
without it, they open terminals on the workstation. Both watcher skills
pin `tmux` explicitly for themselves and keep it, because a loop in a T3
thread dies at the 30 min idle release.

What the `t3` value changes, stated honestly `[critic F9]`: the host (a
thread, not a window), the identity transport (transcript-discovered, not
env), the join (a result file, not a process exit), and the permission and
model vocabularies (runtime mode and catalog ids). The exit-code table
keeps its meanings; 4.3 says what "finished" means on this lane.

### 4.2 The lane itself

Thin bash, logic in Python, as the hooks trio already does (PRD
`plans/session-attention-rail-prd.md` §4 decision 9): the `t3-lane` case
in `spawn-claude.sh` calls a new stdlib-only module by absolute path under
the same import contract as `session_registry.py`. Exit codes stay defined
in bash; the module returns a status word that bash maps `[critic F20]`.
`session_registry.py` is 5 844 lines, past the heuristic-14 alarm, so the
new verbs (4.4) go in the new module, which imports `session_registry`
downward and nothing imports it back.

Positional contract unchanged. Mapping:

| Input | T3 lane |
|---|---|
| `cwd` | project argument to `t3 thread start`. The CLI accepts any path inside a project and **silently runs the thread in the project root**, so the lane compares realpaths itself and refuses a cwd that is not a registered `workspaceRoot` `[critic F16]`. |
| `skip_perms=true` | project default runtime mode (Full access on every registered project today; see the bypass-disabled caveat in §3). |
| `skip_perms=false` | not expressible through the CLI (no `--runtime-mode`). Refused. |
| `title` | `--title`; the convention title `<role>:<project>#<task> slug` is what the T3 sidebar shows. A nonce is not needed: the record dir path in the trailer already identifies the launch (4.4). |
| `prompt` / `--prompt-file` | message on stdin (`-`), trailer appended as today. A message over 120 000 chars is spilled to a file and sent as the existing `--prompt-file` pointer prompt instead of failing `[critic F18]`. |
| `CLAUDE_SPAWN_MODEL` | `--provider claudeAgent --model <value>`, passed through; T3 resolves aliases itself `[critic F4]`. |
| `CLAUDE_SPAWN_CLAUDE_ARGS` | no per-thread args exist. Refused. |
| `CLAUDE_SPAWN_MODE` | unchanged meaning (parent rule and whether to wait); see 4.3 for what waiting is on this lane. |

Refusals (heuristic 10, `[critic F5]`): "this lane cannot honour that
input" is a different invariant from "bad usage", so it gets its own code
from the free 145–199 range, say 145, with a message naming the input and
the lane, whether the lane was explicit or auto-selected. Falling back to
a terminal when Leo is remote would be a window nobody sees, which is the
silent jam the exit-code contract exists to prevent.

Launch and liveness:

- `launching` runs exactly as today, so `result.md`, `record.prompt`, the
  role/project/task ids and the sweeps are unchanged in shape.
- `t3 thread start --json` → `{threadId, projectId, runId}`. The lane
  stamps `set-display --kind t3 --t3-thread-id <id>` (new `DisplayKind.T3`,
  `Display.t3_thread_id`) and flips the record to `running`.
- If the CLI reports the server unreachable **after** the send, the lane
  lists the project's threads by title before declaring 127, because the
  launch has no retry key and may have succeeded `[critic F6]`.
- Started probe: `t3 thread list --json` (one page, has `status`) until
  the thread's status leaves `preparing|queued|starting`; `failed` before
  any run started → 144. The transcript probe is kept for 4.4, not for
  liveness. `thread show` is not used for polling: it issues and revokes an
  auth session and pages the whole timeline each call `[critic F17]`.
- No `t3` on PATH or no `server-runtime.json` → 126; `t3 thread start`
  failing with no thread found afterwards → 127.

### 4.3 Join: what "finished" means for a thread

A thread never exits, so child mode waits on state, not a pid. Terminal
conditions, each enumerated `[critic F8]`:

| Observation | Verdict |
|---|---|
| `result.md` present with `outcome:` other than `in-progress` | finished, exit 0 |
| thread `archived: true` | finished, exit 0 (result.md may be absent) |
| latest run `failed`, `cancelled`, `interrupted` or `rolled_back`, no newer run within a grace, and no final `result.md` | finished, exit 1 |
| thread not found (deleted) | finished, exit 127 |
| server unreachable for longer than a bounded grace | give up, exit 126 |
| `settled: true` | **not** finished: a sidebar flag after 3 idle days `[critic F8]` |
| idle, `waiting`, or `running` | keep waiting |

Exit codes stay liveness-only; the watcher's join step (read `result.md`)
is unchanged. The poll is slow (~30 s) and uses `thread list`.

A spawner that is itself a T3 thread is released after about 4 h of
silence even with a background task pending, and the task dies with it.
So from inside T3 a child-mode wait longer than that is unreliable; the
SKILL.md section says so and recommends sibling mode for hours-long
hand-offs from a T3 spawner. The default is **not** changed to sibling:
that mode also re-parents the child, and "don't wait" and "re-parent" are
two axes the existing `CLAUDE_SPAWN_MODE` already fuses `[critic F2]`;
splitting them is out of this proposal's scope and is noted as such.

### 4.4 Identity: linking the spawn record to the thread's session

The five env exports cannot reach the child, and the child's session id is
not stable (§3). Design, replacing v1's prompt-run `adopt`:

**Discovery, deterministic.** The trailer already embeds the record dir
path in the first user message. The child's transcript lands at
`~/.claude/projects/<encoded-cwd>/<sid>.jsonl` (same encoding the 144 probe
uses; `CLAUDE_CONFIG_DIR` is unset for T3's Claude instance). After
launch, the lane scans new transcripts in that directory for its own
record dir path and so learns `<sid>` and the file exactly, with no
ambiguity between concurrent same-cwd spawns `[critic §6 note]`. No T3
private store is read.

**Binding.** A `bind-t3-session` verb (new module) then: stamps
`claude_session_id` and the live claude pid on the launch record; writes
`<fleet_root>/sessions-by-claude-id/<sid>` → slug, a sibling of the
existing `sessions-by-pid/`; and folds the hand-launched record the
child's first SessionStart already created (marks it exited, pointing at
the adopted slug), so there is one row per spawn.

**Hook keying.** `session_hooks.py::_resolve_hook_slug` gains one step
between the env slug and the hand-launched fallback: resolve the stdin
`session_id` through `sessions-by-claude-id`, and if that misses, the
event's claude pid through `sessions-by-pid`. A pointer-resolved slug goes
through the same ownership probe as an env slug `[critic F12]`, so a
nested `claude --resume <sid>` cannot capture the record. On a re-mint
(`/clear`, compaction: same pid, new sid) the pid pointer finds the record
and the hook re-binds the new sid and writes a new sid pointer; the
existing re-mint rule's `claude_owner_pid` is refreshed whenever the
record is re-bound `[critic F1]`. `docs/session-hook-ownership.md`
§Record keying is updated in the same change `[critic F19]`.

**Residual.** After an idle release, T3 resumes in a new pid. A plain
resume keeps the sid and the sid pointer carries; a failed resume binds a
fresh native session with a context handoff, and nothing links the new sid
or pid to the record. The lane re-runs discovery (a new transcript in the
cwd dir that mentions the record dir path) while it is still waiting;
after the spawner has gone, the record stays bound to the old sid and a
hand-launched row takes over. This is the one known gap and is stated in
SKILL.md.

**Liveness without pids.** A `DisplayKind.T3` record is exempt from the
pid-based orphan sweep and the 24 h reap trigger; its liveness is the
thread's existence and `archived` flag via `t3 thread list --json`, run by
the same opportunistic sweep that drives the pid sweeps `[critic F3]`.
Without this, a thread waiting a day for Leo's answer would be marked
exited after an hour of hook silence and its `result.md` deleted a day
later.

**The spawner's own identity** `[critic F2]`. When the script finds no
`CLAUDE_SPAWN_SESSION_ID` in its environment, it resolves its own record
through `sessions-by-pid/<ancestor claude pid>` and uses that as
`parent_of_child`, so a T3-hosted spawner's children are parented. The
same lookup backs a small `own-record` verb that skills use in place of
the env vars `[critic F7]`: `skills/prd/references/author-mode.md` today
writes `result.md` only when `$CLAUDE_SPAWN_RESULT_FILE` is set and would
silently skip the handoff record on this lane. Every skill that reads a
`CLAUDE_SPAWN_*` var directly is a caller of this change.

The T3 analogue of `CLAUDE_SPAWN_SELF_EXIT=1` is `t3_thread_organize`
`archive` on itself as the final action (not `settle`, which is only a
sidebar flag); the handler's self-target behaviour is not yet verified.
Thread retitling by the hooks (OSC to `/dev/tty`) is a no-op in T3 and
stays one; T3 shows run state natively.

### 4.5 Callers and documents

- `skills/spawn/SKILL.md`: a "T3 lane" section (selection, the 4.2 table,
  the 4.3 join table, the `t3-thread://v1/<id>` link form). No rationale
  from this document migrates there (Comments stance).
- `skills/escalation-watcher/SKILL.md`: one line that a T3-lane background
  task ends on the result file rather than a process exit. Lane stays tmux.
- `skills/prd/references/author-mode.md` and any other direct reader of
  `CLAUDE_SPAWN_RESULT_FILE` / `CLAUDE_SPAWN_SESSION_ID`: use the
  `own-record` verb when the env var is absent.
- `docs/session-hook-ownership.md`: the new keying step and re-bind rule.
- `cockpit` focus backend for `DisplayKind.T3` (open the thread URL): out
  of scope beyond the record field.

### 4.6 Tests

A fake `t3` on PATH emitting the recorded JSON shapes for `thread start`,
`thread list`; a temp `T3CODE_HOME` with a `server-runtime.json` whose pid
and start time match the test's own ancestor, for the detector; a fake
transcript written under a temp `CLAUDE_PROJECTS_DIR` carrying the record
dir path, for discovery; registry tests for `DisplayKind.T3`, the bind
verb, the fold, the two pointer dirs and the t3 liveness sweep; hook tests
for the new keying step and the re-mint re-bind driven through
`run_session_start` with a stdin `session_id` and no env slug.

## 5. Phasing

1. Lane + detector + `CLAUDE_SPAWN_BACKEND` vocabulary + fleet default
   file + `DisplayKind.T3` + started probe + exit 145 + SKILL.md section.
   Root-cwd, full-access spawns work end to end; records are not yet
   linked.
2. Identity and liveness: discovery, bind, fold, pointer dirs, hook keying
   step, re-bind on re-mint, t3 liveness sweep, spawner self-resolution,
   `own-record` verb and its skill callers, ownership doc.
3. Optional upstream CLI flags (§6), then lift the two refusals.

## 6. Upstream asks, all nice-to-have `[critic §6]`

None of these fixes the identity or liveness problems; each maps onto an
existing MCP capability, and every caller found today passes
`skip_permissions=true` with a project-root cwd.

- `t3 thread start --runtime-mode <mode>` → would honour `skip_perms=false`.
- `t3 thread start --existing-worktree <path> [--branch]` → spawns with a
  `.worktrees/<id>` cwd.
- `t3 project list --json` (id, title, workspaceRoot) → would replace the
  lane's read of `projection_projects` (§8).

Exposing the provider session id in the thread detail is no longer needed:
transcript discovery is exact.

## 7. Known limits

- ~100 s to first turn on the pilot (thread preparation dominates).
- The T3 server must be running; laptop app and server must share a
  protocol version.
- Project default runtime mode is Full access everywhere today, subject to
  the bypass-disabled downgrade.
- Child-mode waits from inside T3 are bounded by the spawner's own
  release (~4 h of silence after 30 min idle).
- The context-handoff residual in 4.4.
- Not verified: that no path notifies a launcher thread of a top-level
  thread's completion; `t3_thread_organize archive` on the calling thread;
  the SDK argv of the child (irrelevant after dropping option B of v1).

## 8. As implemented (2026-10-26, task 7489)

- **Decision A.** No `<fleet_root>/spawn-backend` default file. Leo now
  hosts every escalation-watcher session in T3, so the spawner rule alone
  routes `/unblock` spawns into T3 threads; the per-call
  `CLAUDE_SPAWN_BACKEND=t3` override is the escape hatch from a plain
  terminal. The two watcher skills describe T3 hosting as the norm and
  tmux as the plain-terminal lane.
- **Project root (§4.2 cwd rule).** No `t3` command prints a project's
  `workspaceRoot` and there is no `t3 project list`. `launch` reads the
  registered roots read-only from `$T3CODE_HOME/userdata/statev2.sqlite`
  table `projection_projects` (the table `t3_project_list` is served
  from) and refuses, exit 145 `detail=unverifiable`, when it cannot. This
  is the one T3 private store the lane reads; §4.4's discovery still reads
  none. Upstream ask added to §6: `t3 project list --json`.
- **Archived threads (§4.3).** `thread list --json` carries active
  threads only, so `archived: true` never appears in a listing. When a
  thread drops out of the list `wait` makes one `thread show --json` call:
  a successful show that reports `archived` finishes with exit 0; a failed
  show counts as unreachable (the lane does not parse the CLI's error
  text), so a deleted thread ends with exit 126 after the unreachable
  grace rather than the table's 127, which now covers only a successful
  show that reports no thread.
- **Parentage.** A T3 thread has no `CLAUDE_SPAWN_PARENT_ID`, so
  `launch --parent <slug>` stamps `parent_session_id` with the display;
  the script resolves its own slug through `self-record` when it has no
  `CLAUDE_SPAWN_SESSION_ID` (every lane).
- **Idle release (§4.4 residual).** On a sid-proven hook event whose
  recorded `claude_owner_pid` is dead, the hooks reclaim the pid, so a
  resume in a new process followed by a compaction re-mint stays on the
  spawn record. The remaining gap is a failed resume that binds a fresh
  sid in a new process.
- **Fold.** The hand-launched record is marked exited with
  `FOLDED_EXIT_CODE=201`; nothing on it points at the adopted slug.
- **Sweep driver.** `launch` and every `wait` poll sweep the project's T3
  records against a fresh thread listing, so a sibling-mode record whose
  thread vanishes is marked exited by the next spawn in that project. There
  is still no periodic driver.
- **Review fixes (2026-10-26).** A failed/cancelled/rolled_back status seen
  before any transcript is discovered is `failed-to-start`; a missing bound
  transcript is no evidence for a re-bind; the hand-launched record to fold
  is found through its sid pointer, not a re-derived slug; `wait` exits when
  its caller is gone; after-send recovery is skipped when two threads carry
  the title; `$CLAUDE_CONFIG_DIR/projects` is a transcript location.

## 9. As implemented (2026-11, task 8123, β)

The contract is `plans/spawn-t3-lane-mcp-and-host-registry-prd.md`. This
section records what β changed against §4 and §8.

- **Transport.** `launch`, `provider`, `await-started` and `wait` reach T3
  only through α's client (`orchestrator/src/orchestrator/t3_mcp_client.py`)
  on this host's `/mcp` credential, self-paired through the local `t3` CLI on
  the T3 host. The `t3 thread start|list|show` subprocesses, the
  `statev2.sqlite` project read and the title-based recovery of a lost launch
  are deleted. A failed connect is `no-launcher reason=t3-mcp-<unpaired|
  credential-expired|pair-failed>`, or α's `descriptor-unreachable` /
  `descriptor-incompatible`. The script answers the `t3-mcp-*` family with
  exit 126 and names `spawn_t3_lane.py pair`.
- **Projects.** Project ids come from `t3_project_list`, paged with its
  cursor; a listing missing any page counts as unread. Deleted projects are
  dropped, archived ones are set aside, and the deepest live workspace root
  containing `realpath(cwd)` is the thread's project. The `provider` verb
  resolves the project the same way and falls back to environment-wide
  enablement when T3 cannot be read.
- **Workspace and runtime mode.** `workspaceStrategy` is `root` when cwd is
  the project root, else `existing_worktree` with `worktreePath = cwd`.
  `runtimeMode` follows `skip_permissions` alone: `true` is `full-access`,
  `false` is `approval-required`. The §4.2 refusals of `skip_permissions=false`
  and of a non-root cwd are gone. T3 judges whether a worktree belongs to the
  project.
- **Refusal vocabulary.** `cwd-outside-projects` (no live project contains
  cwd), `project-archived` (only an archived one does; detail is its root),
  and `cwd-not-project-worktree`, which is T3's `invalid_request` to an
  `existing_worktree` launch, with T3's message as the detail. The lane never
  parses that message. `provider-disabled` and the `CLAUDE_SPAWN_CLAUDE_ARGS`
  refusal are unchanged.
- **Lost launch reply.** `clientRequestId` is a fresh uuid per `launch`
  invocation, not the record slug the brief named: a slug repeats once its
  launcher pid recycles, and T3 replays any accepted launch whose thread is
  not deleted (archived included), so a slug key could silently hand a new
  spawn an old thread. A launch whose reply is lost (transport unreachable or
  protocol) is sent once more within the same invocation with the same key,
  and T3 replays the same thread rather than starting a second.
- **Call bounds.** `t3_thread_wait` waits one slice plus the client's HTTP
  margin. Every other lane call waits at most `CALL_TIMEOUT_SECS` (60 s) for
  its reply, replacing the deleted CLI seam's 300 s per subprocess.
  `await-started`'s reads are also bounded by its remaining deadline. A
  stalled launch reply counts as lost and is resent with the same key.
- **Join.** Each `wait` pass reads the thread (`t3_thread_read`), sweeps the
  project's t3 records from one complete `t3_thread_list`, maintains the
  binding and asks the unchanged `JoinTracker`. With no verdict it holds one
  `t3_thread_wait` slice of 600 s. T3 0.1.6's `waitForThread` answers at once
  when the latest run is already terminal or there is no run, so after a
  slice that did not time out the pass pauses `SETTLED_RECHECK_SECS` (60 s)
  first; otherwise a settled or failed thread is a busy loop. No pause
  follows once `result.md` is final. `SPAWN_T3_POLL_SECS` is gone.
  `ThreadState.LISTED` is renamed `PRESENT`.
- **Deleted thread.** Only `thread_not_found` means the thread is gone, and
  that is exit 127. §4.3's "a deleted thread ends as 126" is superseded. Every
  other failure to observe it (transport, a failed connect, any other tool
  failure) is unreachable and ends as 126 after `UNREACHABLE_GRACE_SECS`.
- **Exit record.** `skills/spawn/spawn-claude.sh::record_exit` is the single registry
  `exit` write. `finish()` and the t3 child join both call it, so a child
  join leaves `record.json` `exited` with its code. A lost join writes
  nothing.
- **Record.** The launch stamp adds `display.t3_environment_id`, the
  environment the client connected to. β adds the field to
  `session_registry.Display`, defaulting to None, with no schema bump; γ
  builds on it.
