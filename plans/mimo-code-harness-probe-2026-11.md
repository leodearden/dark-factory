# MiMo Code harness probe

Can the eval runner drive MiMo Code headlessly with MCP?

- Task 7993, PRD `plans/live-shadow-eval-prd.md` decision 15 (amended
  2026-10-31). Precedent: task 5399, `plans/zcode-harness-probe-2026-11.md`.
- Measured 2026-11-09 and 2026-11-12 by the task 7993 architect, outside the
  agent sandbox.
- MiMo Code 0.1.16, a Bun single-executable and an OpenCode fork. npm
  `@mimo-ai/cli` is only a launcher for
  `https://mimocode.cnbj1.mi-fds.com/mimocode/mimocode/releases/v0.1.16/mimocode-linux-x64.tar.gz`,
  archive sha256 `3793d8b08ae7c1c6cee707645da1b1c827e0f13faa5d40d076f08ea4ee1d9a5e`.
  No `.sha256` is published.

**Method.** Every model call went to a mock OpenAI chat-completions server on
127.0.0.1 that logged each request's path, headers and body, and scripted a
first-turn tool call. A stdio MCP stub named `plan-tools` exposed
`mark_step_done` and appended each `tools/call` to a marker file. No live MiMo
call was made, because no agent holds `MIMO_API_KEY`.

**Verdict: GATE PASSES.** (a) present; (b) present; (c) present as a step
cap, with the wall clock as backstop; (d) achievable; (e) route established,
live key unexercised; (f) harness mapping established, live endpoint
semantics not. Stage 2 (the `mimo` backend arm and the
`mimo-v2.6-pro-mimocode` / `mimo-v2.6-flash-mimocode` candidates) is
therefore built.

## (a) Headless invocation

Command, prompt on stdin:

    mimo run --title <x> --format json --dir <wt> --dangerously-skip-permissions \
      -m mimo-token-plan/<model> [--variant low|medium|high]

stdout is JSONL: `step_start`, `text`, `tool_use`,
`step_finish{tokens, cost, reason}` and `error`.

Reliability:

- 27 runs, 1 void (a mock port collision).
- 26/26 sessions completed and called the MCP tool.
- 1/26 exited rc 0 with EMPTY stdout although its session completed: the
  shape of upstream issue #2727.
- A 401 exits rc 0 with one `error` event. 429 and 404 are retried until the
  process is killed.

Detection: the exit code is insufficient. The arm reads
`mimo session list --format json` (cwd = the worktree) and then
`mimo export <id>` under the same `MIMOCODE_HOME`. That recovered the lost
run's usage, the 401 as `info.error`, and 33 completed steps after a
wall-clock kill.

Usage is summed over every exported session. The run's outcome and output
come from its root session's last assistant step. The root is the session
named by the first stdout event that carries a `sessionID`. When stdout names
none (the #2727 shape, or a wall-clock kill), the root is the one exported
session without a `parentID`. If the named session has no export, or no
parentless session or more than one exists, the arm logs a WARNING and fails
the run with subtype `error_no_root_session`. It still reports the summed
usage.

Also:

- A non-TTY stdin is read to EOF before the run starts. An inherited open
  pipe hung the run with zero model requests, so the arm always closes stdin
  after writing the prompt.
- `--title` suppresses an extra title-generation request, which the session
  store does not record.
- Measured 2026-11-12: `--variant low|medium|high` is sent as request
  `reasoning_effort`. Any other name (`max`, `xhigh`, `minimal`, an unknown
  word) is silently dropped: rc 0, no `reasoning_effort`, no warning. So the
  arm forwards low/medium/high and maps any other effort to `high` with a
  WARNING.

## (b) MCP server configuration

The `mimocode.json` `mcp` entry is
`{type: 'local', command: [...], environment, enabled}` or
`{type: 'remote', url, headers}`. Tools are lazy: the first request offers
the proxies `mcp_tool_search` and `mcp_tool_use` (17 tools in all), plus a
reminder naming `plan-tools_mark_step_done`. A first-turn `mcp_tool_use`
reached the server in every valid run. The marker line:

    {"name": "mark_step_done", "arguments": {"step_id": "step-1"}, "_meta": {"progressToken": 2}}

## (c) Turn bound

`agent.build.steps = N`: the cap turn says `MAXIMUM STEPS REACHED` with
`tool_choice: 'none'`, and the run ends after N requests. That depends on the
endpoint honouring `tool_choice: 'none'`. No cost or budget flag exists, and
429/404 retry forever. The arm therefore sets `steps = max_turns` AND refuses
an invocation without a wall-clock timeout.

The cap turn offers no tools, so it can finish `stop`, and a capped run would
then read as a success. The arm treats a root session with `max_turns` or more assistant
steps as capped. It reports that as `success=False` with subtype
`error_max_turns`, as the Claude arm does, so `classify_agent_failure` routes
it to `MAX_TURNS`. Subagent sessions run their own loops and do not count
towards the cap.

That timeout bounds `mimo run` only. The store read that follows, one
`session list` plus one `export` per session, is bounded separately: each
call gets
`orchestrator/src/orchestrator/agents/mimo_backend.py::SESSION_READ_TIMEOUT_SECS`
(120 s). So an invocation over N sessions can last up to
`timeout_seconds + (1 + N) × 120 s`.

## (d) Isolation

- `MIMOCODE_HOME` holds `{config, data, cache, state}`; `data` holds the
  sqlite session db. Measured 2026-11-12 with `mimo debug paths`: all four
  stay under `MIMOCODE_HOME` even with every `XDG_*` set elsewhere, nothing is
  written to the XDG dirs, and a `mimocode.json` at
  `$XDG_CONFIG_HOME/mimocode/` is not read (`mimo mcp list`: no servers).
- Under landlock, `TMPDIR` must point inside a writable dir, or the sqlite
  migration fails with `DrizzleError`.
- `~/.claude.json` MCP servers are imported unless
  `MIMOCODE_DISABLE_CLAUDE_CODE_MCP=1` (measured both ways). The worktree
  `.mcp.json` was never imported.
- A self-gitignored `.mimocode/` appears in the worktree; `git status` stays
  clean.
- Deny rules hold under `--dangerously-skip-permissions`, both for bash
  `git push*` and for `<server>_<tool>` called through the `mcp_tool_use`
  proxy.
- Project `AGENTS.md`/`CLAUDE.md` and `<home>/config/AGENTS.md` are delivered
  as `Instructions from:` blocks after MiMo's native prompt. The arm writes the
  role system prompt to `<home>/config/AGENTS.md`.
- `HOME` is redirected to the run home. As in the ZCode precedent, the agent's
  commands keep the operator's git identity (`<home>/.gitconfig` includes the
  operator's `~/.gitconfig`) and tool homes. The measurement in the first
  bullet shows that keeping the operator's `XDG_*` does not leak MiMo's own
  state.

The child env the arm sets, over the inherited env with every secret-named
variable removed (`shared/src/shared/secret_env.py::is_secret_env_name`). Every
inherited `MIMOCODE_*` variable is removed too. MiMo Code is an OpenCode fork,
and OpenCode honours `OPENCODE_CONFIG_CONTENT`- and `OPENCODE_PERMISSION`-style
overrides. A MiMo equivalent left on the host for interactive use could
otherwise add MCP servers or relax a deny.

| variable | value |
|---|---|
| `HOME`, `USERPROFILE`, `MIMOCODE_HOME` | the run home |
| `TMPDIR` | `<run home>/tmp` |
| `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`, `CARGO_HOME`, `RUSTUP_HOME` | the operator's value, or its default under the operator's `HOME` |
| `MIMOCODE_ENABLE_ANALYSIS` | `false` |
| `MIMOCODE_DISABLE_CLAUDE_CODE_MCP`, `MIMOCODE_DISABLE_MODELS_FETCH`, `MIMOCODE_DISABLE_AUTOUPDATE`, `MIMOCODE_DISABLE_CRON`, `MIMOCODE_DISABLE_SHARE` | `1` |

## (e) Credentials and protocol

A literal `options.apiKey` in a 0600 `mimocode.json` reaches the endpoint as
`Authorization: Bearer`. The key never enters the child env, where the
model's bash tool could read it.

MiMo Code forces any model id matching `/(^|[/_-])mimo(?:-|$)/i` onto
`@ai-sdk/openai-compatible`. This was source-read, then measured: with npm
`@ai-sdk/anthropic` configured it still posted `/chat/completions`. So the
arm speaks OpenAI `/v1`, at `https://token-plan-{cn|sgp|ams}.xiaomimimo.com/v1`
and `https://api.xiaomimimo.com/v1`, per MiMo Code's catalog providers
`xiaomi-token-plan-*` and `xiaomi`. The Claude-harness control uses
`/anthropic`, so the arms differ in protocol as well as harness. That is the
vendor harness's native wiring, and it is part of the harness difference.

## (f) Usage semantics

| mock `usage` sent | step `tokens` recorded |
|---|---|
| `prompt_tokens` 1000 | `input` 700 |
| `prompt_tokens_details.cached_tokens` 300 | `cache.read` 300 |
| — | `cache.write` 0 |
| `completion_tokens` 50 | `output` 30 |
| `completion_tokens_details.reasoning_tokens` 20 | `reasoning` 20 |

The source formula agrees: `tokens.input` excludes cache reads and writes,
and `tokens.output` excludes reasoning. Over OpenAI, `cache.write` is always
0. `part.cost` is MiMo's own price-table USD estimate (0 for Token Plan rows),
not a bill, so the arm ignores it.

Credits for task 7994: the arm emits `input_tokens = Σinput`,
`cache_create_tokens = Σcache.write`, `cache_read_tokens = Σcache.read` and
`output_tokens = Σ(output + reasoning)`, summed over every session in the run
home. So 7994's Claude-harness mapping applies unchanged: miss = input +
cache_create, hit = cache_read, out = output. `credits_for_usage` was not on
main when this was written.

## Pairing note

`orchestrator/src/orchestrator/evals/shadow_eval.py::sample_point` hashes the
candidate name, so each arm samples its own tasks and is paired with
production, not with its control. Harness minus control is therefore an
UNPAIRED comparison. No `@one-step` variant is added, because fresh sessions
multiply cache misses.

## Not established (live only)

- Whether `completion_tokens` includes `reasoning_tokens`.
- Whether cache hits arrive as `prompt_tokens_details.cached_tokens`.
- Whether a `tp-` key is accepted on `/v1`.
- The 0.8 off-peak coefficient.
- The quota-exhaustion response.
- `mimo export` size on a long run.
- Checkpoint-writer sessions under large contexts. They would be counted,
  because the arm sums every session in the run home. If one has no
  `parentID` and stdout is lost, the arm sees two roots and fails the run.
- Whether every stdout event's `sessionID` names the root session rather
  than a subagent's. The arm takes the first one as the root.
- Whether, against a live endpoint, the cap's forced turn is the root
  session's `max_turns`-th assistant message. The mock run ended after N
  requests, and the arm reads N root steps as the cap.

All are left to operator task 8955, which installs `mimo` on the orchestrator
host and runs the paired `-mimocode` / `-endpoint` eval cells. Until task 8955
is done, the arm is built and awaiting its live run, not measured.

## 2026-11-13 live run (task 8955)

Operator gate esc-8955-2, run on main `a2377e8699`. Logs, result summaries
and the probe scripts are in `/tmp/gate-8955/`, which is volatile.

**Install.** The archive's sha256 matched
`3793d8b0…ee1d9a5e`. It holds one file, `mimo`, a Bun 1.3.14
single-executable with sha256
`09245de5a046a664ad3ed3b0ebda8a4c684a1b1f6b9b14e81754572c974be8c4`. It is installed as `~/.local/bin/mimo`, and
`mimo --version` prints `0.1.16`, under `env -i` as well.

**.env walk: none.** ZCode (task 9541) loaded the first `.env` above its
cwd. MiMo Code loads no `.env`, neither from its cwd nor from a parent. The
canary was `XDG_CONFIG_HOME` in the `.env`, because `mimo debug paths` echoes
it back:

- `.env` in the parent dir: not read.
- `.env` in the cwd: not read.
- `XDG_CONFIG_HOME` passed explicitly: echoed. This is the positive control.
- An env-dumping `rg` on PATH, which `mimo debug rg files` spawned, received
  mimo's own `MIMOCODE_*` variables but no canary, in both layouts.

The binary carries Bun's `--compile-autoload-dotenv` option text and Effect's
`ConfigProvider.fromDotEnv`, but nothing at runtime acts on either. In the
two live MiMo Code cells, 106 `/proc` samples of `mimo` and its children
(`HOME=/tmp/mimo-home-*`) carried 0 secret names. The eval CLI's own verify
subprocesses still carry the whole `.env`, which tasks 6828 and 7649 track.

**Endpoint facts.** These come from a direct probe of
`https://token-plan-ams.xiaomimimo.com/v1/chat/completions`
(`/tmp/gate-8955/v1-probe.jsonl`), plus the cells.

| question | answer |
|---|---|
| Does the `tp-` key answer on `/v1`? | **Yes.** The ams region returned 200 for both `mimo-v2.6-pro` and `mimo-v2.6-flash`, and both MiMo Code cells completed through it. |
| Does `completion_tokens` include `reasoning_tokens`? | **Yes.** `total_tokens` = `prompt_tokens` + `completion_tokens`. One flash call: completion 22, of which `completion_tokens_details.reasoning_tokens` 17, for the 3-character answer "289". MiMo Code's subtraction and the arm's `output + reasoning` are therefore correct. |
| Do cache hits arrive as `prompt_tokens_details.cached_tokens`? | **Yes.** Three identical pro calls with an 8,550-token prompt read 0, then 8448, then 8448. `prompt_tokens` stayed 8550, so it includes the cached part. Two flash repeats read 0 and 0, so a hit is not guaranteed; the calls were answered by different `MiFE` front ends. |
| Is the 0.8 off-peak coefficient still live? | **Documented, not observable.** The price page, updated 2026-10-21, still lists 0.8x from 00:00 to 08:00 Beijing time (16:00–24:00 UTC). Neither the responses nor their headers carry credits. All cells ran from 18:32 to 23:34Z, inside that window. |
| What does quota exhaustion return? | **Not reached.** No 429 or quota response appeared in 4 cells and 5 direct calls: 272/272 bridged `/v1/messages` calls returned 200. The docs say only that the service is suspended. Task 7994 still needs the real status and body. |

**Paired cells.** Fixture `df_task_12`, `--timeout 180`, run sequentially.
Every cell has `outcome=done`, `role_under_test=implementer`, a scoring
verify that is green on tests, lint and types, `cap_tainted=false` and
`invocation_errors=null`.

| arm | impl wall | impl turns | total turns | in / out | cache read / create | cost_usd | debug | review verdict | composite |
|---|---|---|---|---|---|---|---|---|---|
| pro-mimocode `804dae3d` | 3m45s | 33 | 72 | 221,131 / 22,438 | 3,472,651 / 132,540 | 0.1157 | 2 | PASS, 6 suggestions | 0.90 |
| pro-endpoint `8e83c100` | 14m16s | 31 | 92 | 96,264 / 29,222 | 2,938,992 / 105,505 | 0.0673 | 1 | ISSUES_FOUND, 0 blocking, 5 suggestions | 0.95 |
| flash-mimocode `0dd5e13c` | 3m36s | 37 (+20 in a review fix) | 103 | 168,088 / 50,700 | 6,070,235 / 148,677 | 0.0377 | 1 | cycle 1: 1 blocking (tests patch private `_run_cmd`); cycle 2: ISSUES_FOUND, 0 blocking, 4 suggestions | 0.95 |
| flash-endpoint `9a48a423` | 43m29s | 57 | 101 | 4,924 / 29,534 | 4,196,141 / 133,798 | 0.0090 | 1 | PASS, 5 suggestions | 0.95 |

How to read the table:

- **Metrics are summed over every role**, including the Claude judge
  (sonnet) and the reviewer (opus). That follows
  `orchestrator/src/orchestrator/workflow.py` metrics accumulation and
  `orchestrator/src/orchestrator/evals/runner.py::run_eval`. So `turns`,
  the tokens and `cost_usd` are not the implementer's own figures.
- **The MiMo Code arm's own `cache.write` is 0 over OpenAI**, as section (f)
  says. The non-null `cache_create_tokens` in the two `-mimocode` rows comes
  from the Claude roles. The acceptance check "non-null" passes, but it does
  not exercise the arm.
- **`cost_usd` prices uncached input and output at the MiMo list rate and
  drops cache reads**, the same as gate 8939's finding 1. It also prices the
  Claude roles' tokens at MiMo rates, so it supports no comparison.
- **Every cell's task-gate verify was red** on fixture noise: lint scripts
  missing at `pre_task_commit`, and `mem0`/`graphiti_core` import errors.
  Each cell paid 1 or 2 debugger cycles to work around it, and every arm
  copied about 2.1k lines of lint scripts from main. Turns and diff size
  therefore do not compare across arms.
- **The `-endpoint` control runs Claude Code through `shared.vllm_bridge`.**
  `shared/src/shared/cli_invoke.py` starts the bridge for any
  `ANTHROPIC_BASE_URL`. The bridge forces `stream=False` and clamps
  `max_tokens` from 32000 to 8192, because no `max_model_len` is discovered.
  So the control is not the plain vendor `/anthropic` path. This was
  equally true of gate 8939's GLM control.

**Still not established:** the quota-exhaustion response, the `mimo export`
size on a long run (the largest here was 339 KB), and the checkpoint-writer
case under a large context.
