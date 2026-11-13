# ZCode harness probe: can the eval runner drive ZCode headlessly with MCP?

- Task 5399, PRD `plans/live-shadow-eval-prd.md` leaf ω, manifest capability
  `zcode-is-headlessly-drivable` (`plans/live-shadow-eval-prd.capability-manifest.yaml`).
- Measured 2026-11-09 by the task 5399 architect, outside the agent sandbox.
- Upstream `zai-org/ZCode` commit `a9fe401eede6e6e5e01be9048679c206fe6f02d9`,
  CLI 0.17.1, Node 24.14.0, pnpm 10.33.2.

**Verdict: GATE PASSES.** (a) present, (b) present, (c) present, (d) absent
natively, so the launcher's wall clock bounds each invocation. Stage 2 (the
`zcode` backend arm and the `glm-5.3-zcode` candidate) is therefore built.

## Capabilities

| capability | verdict | evidence |
|---|---|---|
| (a) headless, non-interactive invocation | PRESENT | `zcode --help` lists `-p, --prompt <text>  Run a single prompt without opening the TUI`, `--cwd <path>`, `--mode <mode> ... (default: yolo for --prompt)`, `--disallowed-tools` and `--json`. `--output-format json` prints one object: `sessionId`, `traceId`, `turnId`, `response`, `usage{source, modelRequestCount, inputTokens, outputTokens, totalTokens, cacheReadTokens, cacheWriteTokens, reasoningTokens, ...}`, `eventCount`, `projection{status, turnCount, ...}`. Exit 0 on success; on error it writes `Error: ...` to stderr and exits 1. |
| (b) MCP server configuration | PRESENT, measured | The mock-model run below: ZCode offered `mcp__plan-tools__mark_step_done`, executed it, returned the `tool_result` to the model and exited 0. |
| (c) target directory argument | PRESENT | `--cwd <abs>` works. With `HOME`, `USERPROFILE` and `ZCODE_DATA_BASE_DIR` pointed at a temporary run home, nothing was written into the `--cwd` directory. |
| (d) per-invocation turn or cost bound | ABSENT natively | The parser is strict (`node:util` `parseArgs`, `strict: true`): `--max-turns` exits 1 with `Unknown option '--max-turns'`, and `--model` and `--mcp-config` likewise exit 1 with `Unknown option '--model'` and `Unknown option '--mcp-config'`. `apps/zcode-cli/docs/usage/02-cli-reference.md` leaves runtime limits to the evaluation process manager that launches ZCode. |

### The MCP run (b)

A mock Anthropic-protocol model server on 127.0.0.1 scripted one tool call,
and a FastMCP stdio server named `plan-tools` exposed a `mark_step_done` tool
that writes a marker file. ZCode was wired to them through two files and no
flags. The provider file, named by `ZCODE_PERSONAL_PROVIDER_CONFIG_FILE`, is
`{schemaVersion: 1, config: {providerConfigRules: {providerRules: [{providerId, providerName, config: {group: 'standard-personal', access: {type: 'api-key', apiKey}, api: {type: 'anthropic-messages', baseUrl}, personalModelIds: ['glm-5.3']}}]}, modelConfigRules: {providerModelRules: [], manualProviderModelRules: []}, defaultModelSelection: {providerId, modelId: 'glm-5.3'}}}`.
Upstream's codec (`packages/provider-node/src/provider-config-file-codec.ts`)
parses it strictly, so any other top-level key is rejected. The user-scope
`$HOME/.zcode/cli/config.json` is
`{features: {mcp: true}, mcp: {servers: {plan-tools: {type: 'stdio', command, args, enabled: true}}}}`.

Command:

    zcode --prompt '...' --cwd <wt> --mode yolo --output-format json --no-color \
      --disallowed-tools 'AskUserQuestion EnterPlanMode ExitPlanMode'

The arm passes the prompt as one inline token, `--prompt=<text>`. Strict
`parseArgs` refuses a separate option value that starts with `-`. Measured
2026-11-11 on the host's Node 22.23.4: `--prompt '- x'` and `--prompt '--- x'`
throw `ERR_PARSE_ARGS_INVALID_OPTION_VALUE`, while the inline form parses any
value. This was measured against `node:util` directly, not against ZCode on
Node 24.14.0.

Observed:

- The request log shows two `POST /v1/messages`, each `stream=true` and model
  `glm-5.3`, each offering `mcp__plan-tools__mark_step_done`, the same naming
  as Claude Code.
- The offered tool list was recorded as 17 tools: Agent, AskUserQuestion,
  Bash, Edit, EnterPlanMode, ExitPlanMode, Read, ReadSessionContext,
  SendMessage, Skill, TaskOutput, TaskStop, TodoRead, TodoWrite, WebFetch and
  Write, plus the MCP tool. The record does not say whether that list came
  from a run with or without the `--disallowed-tools` argument above.
- The second request carried the `tool_result`.
- The MCP server's marker file was written.
- rc=0, `response` `probe complete`, `usage.modelRequestCount` 2.

## Distribution

GitHub releases ship only macOS and Windows desktop builds, and there is no npm
package (`@zcode/cli` returns 404). The only Linux distribution is a source
build. It needs Node 24.14.0 (the `engines` field of
`apps/zcode-cli/package.json`; the orchestrator host has 22.23.4):

    npx -y pnpm@10.33.2 install --frozen-lockfile          # 50 s
    npx -y pnpm@10.33.2 --filter '@zcode/cli...' build     # 51 s

That produces `apps/zcode-cli/packages/cli/dist/zcode.cjs`, whose `--version`
prints 0.17.1.

## Isolation

- User-scope MCP servers are not gated by workspace trust
  (`apps/zcode-cli/packages/bootstrap/src/workspace-mcp-trust/CONTRACT.md`),
  while project `zcode.json` servers are denied by default. User scope is
  therefore the arm's only reliable MCP channel.
- The system-prompt channel is `$HOME/.zcode/AGENTS.md`. Measured: a marker
  string written there was found in the request body, delivered as the user's
  private global instructions inside the first user message.
- ZCode has no default OS sandbox. Outside an eval the orchestrator's landlock
  wrap confines it. An eval cell, the arm's only consumer, runs with the
  sandbox disabled (`orchestrator/src/orchestrator/evals/runner.py::build_eval_orch_config`
  sets `SandboxConfig(enabled=False)`). So in `--mode yolo` ZCode runs
  unconfined there, exactly like its Claude-harness control.
- A plain `--prompt` run has no interactive approval client, so asking and
  approval tools fail. ZCode's docs (`04-tools.md`) recommend removing
  AskUserQuestion, EnterPlanMode and ExitPlanMode. Every arm invocation
  removes all three; that removal is an experimental condition of the arm.
  `--disallowed-tools` removes whole tools, so a pattern spec such as
  `Bash(git *)` would remove all of Bash; the arm drops such specs with a
  warning rather than passing them.

## Agent tool environment

- ZCode resolves its own state through Node's `homedir()`. Upstream
  `zai-org/ZCode` a9fe401eed keeps the session db, logs, workflows and the CLI
  config under `~/.zcode/cli`, so the run home must be `HOME`. ZCode passes
  that env to its Bash tool and stdio MCP children, stripping only the proxy,
  CA and telemetry variables
  (`packages/shared/src/runtimeEnv.ts::SANITIZED_RUNTIME_ENV_KEYS`).
- Measured 2026-11-11 with git 2.43.0: with `HOME` redirected alone,
  `git var GIT_AUTHOR_IDENT` exits 128 "Author identity unknown", and
  `git add -A` stages a file that the operator's `~/.config/git/ignore`
  excludes. On this host that ignore file is what keeps `.claude/` artifacts
  out of a commit.
- The arm's remedy. A run-home `.gitconfig` includes the operator's
  `~/.gitconfig`; git skips the include when the file is missing (measured).
  `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`,
  `CARGO_HOME` and `RUSTUP_HOME` are pinned to the operator's values, as
  `orchestrator/src/orchestrator/agents/zcode_backend.py::OPERATOR_TOOL_HOMES`
  defines. The agent's commands therefore resolve the same git identity, git
  excludes, uv cache, uv Pythons and Rust toolchain as the commands of the
  `glm-5.3-endpoint` control.

## Dotenv walk (task 9541)

- Upstream `zai-org/ZCode` a9fe401eed loads one `.env` file into its own
  process env at startup. `apps/zcode-cli/packages/cli/src/env.ts::findDotenv`
  walks up from `--cwd` lexically (`path.resolve`, no realpath) to the first
  regular file named `.env`, and `::loadCliDotenv` loads it with
  `override: false`. `prompt-command.ts` does this unconditionally in
  `--prompt` mode; no flag or env var disables it. What it loads reaches the
  agent's Bash tool and stdio MCP children, as the section above describes.
- Landlock cannot deny that read.
  `orchestrator/src/orchestrator/agents/landlock_exec.py` grants read on `/`,
  and landlock rules only add access.
- Measured 2026-11-13 under landlock with the real CLI 0.17.1: with
  `--cwd <worktree>` and the worktree nested inside a checkout, a canary in
  the checkout's `.env` reached the stdio MCP child's environment.
- The arm's remedy
  (`orchestrator/src/orchestrator/agents/zcode_backend.py::zcode_run_home`,
  `::zcode_argv`). ZCode runs with `--cwd <run home>/workspace`, a symlink to
  the worktree, beside an empty `<run home>/.env`. The walk sees the
  worktree's own `.env`, if any, then stops at the empty file in the run
  home. The sentinel is needed on its own account: the run home lives under
  `/tmp`, which every sandboxed agent can write, so without it a `.env`
  planted in `/tmp` would be loaded. It holds against the agent ZCode runs,
  whose tools start only after the walk; a concurrent peer agent is a
  residual below. The subprocess cwd and the landlock grants stay on the
  real worktree.
- Measured with that layout: the canary was absent, and the MCP child's cwd
  still resolved to the worktree. Upstream's
  `core/src/tool/handlers/bash-cwd-policy.ts` accepts a symlinked session
  root, comparing both the logical and the realpath pairs.
- The regression probe is `orchestrator/tests/test_zcode_dotenv_live.py`. It
  drives the real `zcode` under landlock against a local endpoint that
  answers 401, which ends the run in about 3 s, after ZCode has started its
  MCP children. A control run first shows that a value from a worktree's own
  `.env` does reach the MCP child, so the probe cannot pass merely because
  its observation channel went blind. It is skipped where `zcode` is
  missing, and where landlock is missing unless `DF_REQUIRE_SANDBOX_TESTS=1`
  arms the host, in which case it fails. Measured 2026-11-13: it fails
  against the old `--cwd <worktree>` wiring and passes against the fence.
- Residuals:
  - The worktree's own `.env` is still loaded. Dark-factory worktrees have
    none: `.env` is gitignored and nothing provisions one.
  - Any backend's agent can still read the checkout's `.env` deliberately
    with its own tools, because landlock grants read on `/`.
  - An implementer step committing under the symlinked workspace has not
    been measured; see Not established.
  - The fence does not hold against a concurrent peer agent. The run home
    sits under `/tmp`, which `landlock_exec.py` makes writable for every
    sandboxed agent, and all of them run as one uid, so the home's 0700 mode
    keeps no peer out. Before ZCode's startup walk, a peer could delete the
    sentinel, letting the walk reach a planted `/tmp/.env`, or retarget
    `<run home>/workspace` at the checkout. That adds no capability a peer
    lacks today: it can already rewrite the same home's
    `.zcode/cli/config.json` or `.gitconfig`, and read the checkout's `.env`
    directly (task 9561). Follow-up ticket `tkt_0RWXYYX0E1AHT5BZWS5GVSHGMQ`
    covers moving run homes out of the shared `/tmp` grant.

## Protocol

ZCode's built-in "Z.ai Coding Plan" provider template
(`config/provider/zcode-builtin.json`, templateId `zai-api`) is
`anthropic-messages` at `https://api.z.ai/api/anthropic`. That is
`orchestrator/src/orchestrator/evals/configs.py::GLM_BASE_URL`, the endpoint of
the Claude-harness control `glm-5.3-endpoint`. The official harness and its
control therefore drive the same endpoint over the same protocol, and the pair
differs only in harness. `ZAI_CODING_BASE_URL` (OpenAI protocol) is not used
by the arm.

## Usage semantics

The arm assumes `usage.inputTokens` includes cache read and write tokens,
from a source reading of
`packages/contracts/src/model/index.ts::getModelUsageTotalTokens`, which falls
back to `cacheRead + cacheWrite`. This was not measured: the mock reported no
cache usage. The same helper reads just as naturally the other way, with
`inputTokens` excluding cache tokens, as Anthropic's `usage.input_tokens`
does. The arm subtracts both from `inputTokens`, so its `input_tokens` follows
the shared uncached-input convention, and it clamps a negative difference to 0
with the WARNING "zcode reported cache read+write tokens ... above
inputTokens". It maps `turns` to `usage.modelRequestCount`
(`projection.turnCount` counts user turns and is always 1 for a single
`--prompt`).

## Upstream caveat

`apps/zcode-cli/docs/usage/13-automation.md` states that Linux and real
providers were not verified upstream.

## Not established

- A live GLM run. No agent holds `ZAI_API_KEY`, so no request reached Z.ai.
- Behaviour under the landlock sandbox past startup. Under landlock ZCode
  has been seen to start, spawn its stdio MCP children and exit on a refused
  request (Dotenv walk above); no model turn or tool call has run there.
- Whether an implementer step commits under the symlinked workspace. ZCode's
  session root is now `<run home>/workspace`, a link to the worktree, so its
  file tools, its Bash cwd policy and git all start from a symlinked path.
  Task 8939's live-run commit check below must run with this layout.
- Telemetry defaults. The environment variable
  `ZCODE_MODEL_TELEMETRY_ENABLED` exists; its default was not measured.
- The cache-token arithmetic above, against a real provider. Task 8939's live
  run should compare one ZCode `usage` block with the raw provider usage of
  the same request. Acceptance check: the clamp WARNING does not fire on
  cache-heavy calls. If it does, `inputTokens` excludes cache tokens, the
  arm under-reports uncached input by the cache volume, and the subtraction
  must go before the paired cost comparison is read.
- Whether `--disallowed-tools` honours an MCP tool name. The implementer and
  debugger roles disallow only `mcp__fused-memory__set_task_status`
  (`orchestrator/src/orchestrator/agents/roles.py::_NO_TASK_STATUS_WRITE`),
  and the probe passed only built-in names. If ZCode ignores the name, the
  arm gains a tool its control withholds. If ZCode rejects it at startup,
  every implementer and debugger invocation exits 1. The mock-model rig can settle this
  offline: name `mcp__plan-tools__mark_step_done` in `--disallowed-tools` and
  check whether it leaves the offered tool list.
- Whether ZCode, or a Node dependency, touches the operator state under the
  pinned `XDG_CONFIG_HOME`, `XDG_STATE_HOME` and `XDG_DATA_HOME`. The measured
  "nothing written outside the run home" predates those pins, so it does not
  cover the final child env, and the SEA bullet below shows ZCode does
  consult an XDG directory.
- The agent tool environment under the redirected `HOME`, beyond what the
  remedy pins. These residual differences from the control remain:
  - ZCode's login-shell snapshot sources the run home's rc files, which do
    not exist, instead of the operator's `~/.bashrc` and `~/.profile`. PATH
    entries and variables exported from rc files therefore come only from the
    orchestrator's env.
  - `~` paths inside the operator's git config resolve against the run home.
  - Other `HOME`-relative dotfiles, such as `~/.npmrc`, `~/.docker` and
    `~/go`, are not carried.
  - A single-executable (SEA) ZCode build would cache its assets under the
    pinned `XDG_CACHE_HOME`
    (`apps/zcode-cli/packages/cli/src/tui-runtime-loader.ts`). The arm runs
    the source-built `zcode.cjs`, so this does not arise today.

All eight are left to task 8939 (from ticket `tkt_0RWSZ5GKKD78GK8FKD16G2X67J`),
the operator task that installs the ZCode CLI on the orchestrator host and runs
the paired `glm-5.3-zcode` / `glm-5.3-endpoint` eval cells. Task 8939's live
run should also confirm that an implementer step commits under the arm. Until
task 8939 is done, the arm is built and awaiting its live run, not measured.

## 2026-11-13 re-run (task 8939)

### Probe note from the 2026-11-12 partial run

> **Live run (task 8939, 2026-11-12).** The build at a9fe401eed succeeds on Node 26.13.1 (the engine mismatch only warns); the launcher is `~/.local/bin/zcode`. One `glm-5.3-zcode` implementer cell on `df_task_12` through `run_eval`:
> - The implementer succeeded in 27 turns, committing each of the 3 plan steps and calling `mark_step_done` through ZCode's user-scope MCP. The debugger took one cycle, and verify passed.
> - `role_under_test='implementer'`. Tokens: 118,537 uncached input, 37,509 output, 2,629,312 cache read.
> - **Settled:** the clamp WARNING did not fire with cache read about 22× input, so `usage.inputTokens` includes cache tokens and the arm's subtraction is right.
> - **Settled:** an MCP name in `--disallowed-tools` does not break startup. Whether ZCode honours it is still open.
> - **Settled:** telemetry is off unless an `OTEL_EXPORTER_OTLP_*ENDPOINT` is set.
> - **New risk:** ZCode loads the first `.env` above its working directory (`packages/cli/src/env.ts::loadCliDotenv`, no override). In a `.worktrees/<id>` dispatch that is the main checkout's secrets file, and its values reach the agent's tools. Unverified under landlock.
> - **Still open:** the paired control `glm-5.3-endpoint` could not run. `shared/vllm_bridge.py` forwards the client `Host` header, and Z.ai returns 421. The paired comparison and landlock behaviour stay open.

The 421 was fixed by task 9540 (`31b096f236`) and the dotenv risk by task
9541 (`b0203ead89`, the "Dotenv walk" section above). Both are ancestors of
`2479cdc1fc`, the main HEAD both cells ran from.

### How the cells ran

- Both arms ran concurrently through the CLI:
  `orchestrator eval --config dark-factory-orchestrator.yaml --task …/df_task_12.json --config-name <arm>`.
  The CLI loads `.env`, so the Claude judge and reviewer authenticated: the
  "Not logged in" failure of 11-12 did not recur.
- `run_eval` builds its config with the sandbox off
  (`orchestrator/src/orchestrator/evals/runner.py::build_eval_orch_config`),
  so landlock was not exercised past startup. It remains covered only by
  `orchestrator/tests/test_zcode_dotenv_live.py`, which passed on this HEAD.
- Attempt 1 used the fixture's default 60 min wall clock and both cells were
  cancelled by it before review completed. Attempt 2 used `--timeout 180`.
  Z.ai then answered 429 (`retry-after: 9827`) to both arms from about
  10:18Z, during the debugger, and both cells ended blocked.
- Host load averaged about 50 throughout. Each cold verify took 15 to 34 min.

### Paired results, df_task_12

Cost is the persisted `cost_usd` (`cost_source=price_table`). "Arm log" is
the per-agent figure the workflow logs. Tokens are input (uncached) / output
/ cache read / cache create.

| Cell | Outcome | Impl. turns, wall | Total turns | Tokens | cost_usd | Arm log (impl.) | Debugger | Diff | Scoring verify |
|---|---|---|---|---|---|---|---|---|---|
| zcode, attempt 1 (`8cdc1b35`) | cancelled (60 min) | 43, 9m39s | 67 | 113,526 / 31,638 / 2,888,680 / 39,821 | $0.298 | $2.74 | 1 cycle, ok, 21 turns | 6 files, 2,266 lines | pass |
| endpoint, attempt 1 (`46074e01`) | cancelled (60 min) | 37, 22m35s | 40 | 1,193 / 12,537 / 1,370,402 / 13,702 | $0.057 | $0.50 | cut off by the wall clock | 2 files, 132 lines | pass |
| zcode, attempt 2 (`c5ecc294`) | blocked | 27, 9m17s | 30 | 52,328 / 12,487 / 1,175,101 / 40,803 | $0.128 | $1.67 | 3 failed: Z.ai 429 | 2 files, 153 lines | pass |
| endpoint, attempt 2 (`c04164f6`) | blocked | 47, 20m44s | 68 | 11,443 / 19,516 / 1,976,930 / 39,517 | $0.102 | $0.65 | 2 failed: Z.ai 429 | 2 files, 108 lines | pass |

All four persist `role_under_test='implementer'`, `plan_completion_pct=1.0`
and one judge invocation. In every cell the step judge (Claude Sonnet,
$0.07 to $0.18) signalled completion at iteration 1. No cell produced a
reviewer verdict: the zcode attempt-1 reviewer (Claude Opus) authenticated
and ran for 5 min before the wall clock cancelled it, and no other cell
reached review. `review_blocking_issues=0` in every result therefore means
"not reviewed".

### What the re-run settles

- **A live GLM run under ZCode works end to end.** Both ZCode implementers
  committed all 3 plan steps in TDD order under
  `--cwd <run home>/workspace`, the symlinked workspace from task 9541. An
  implementer step therefore commits under the fenced layout.
- **ZCode's implementer took about 2.3× less wall clock than the
  Claude-harness control.** It ran in 9m39s and 9m17s against 22m35s and
  20m44s, on the same endpoint and model. That is n=2 per arm on one
  fixture.
- **The dotenv fence held in the live cell.** Each run home held a 0-byte
  `.env` beside `workspace -> <eval worktree>`. 930 `/proc` samples of
  the cell's process tree were taken. None of the processes running with
  `HOME=<run home>` (`zcode-cli`, its `plan_tools` stdio MCP child, and the
  agent's bash, uv and pytest children) carried any `.env` variable name.
  The run homes were removed on exit.

### What the re-run found

- **`cost_usd` leaves out cache tokens.** It equals uncached input × $1.40/M
  plus output × $4.40/M to the cent in all four cells; for example zcode
  attempt 1 is 113,526 × 1.4 + 31,638 × 4.4 = $0.298. Cache reads are 92 to
  99 % of input, so the figure is understated. The ZCode arm's logged figure
  prices cache reads at the full input rate, because the eval price table has
  no `cached_input_per_1m` for `glm-5.3`, so it is overstated. The two arms
  split uncached and cached input very differently (113k against 1.2k
  uncached in attempt 1), so neither figure supports a paired cost
  comparison yet. This resolves the "$0.33 against $4.02" question from
  11-12.
- **df_task_12 is red at its own commit under the live task gate.** The
  workflow's file-scoped verify takes the current module configs'
  `lint_command`. That command calls
  `fused-memory/scripts/check_bare_magicmock_config.py` and
  `tests/scripts/raw_envelope_literal_guard.py`, which do not exist at the
  fixture's `pre_task_commit`. It also hits a pyright `proc is possibly
  unbound` error in `verify.py`, an import failure for `graphiti_core` or
  `mem0`, and later a shrink-only debt-baseline ratchet. The scoring verify
  at cell end uses the fixture's own commands and passed in all four cells,
  including immediately after three identical task-gate failures. Every arm
  therefore pays a debugger cycle for fixture noise. ZCode's debugger copied
  about 2,100 lines of current-main scripts to pass that gate, while the
  control's did not finish, so turns, cost and diff size are not comparable
  across arms on this fixture.
- **A provider 429 does not taint the cell.** Both attempt-2 results carry
  `cap_tainted=false` and `invocation_errors=null` even though every
  debugger call failed on Z.ai 429. The ZCode arm exited 1 with empty stdout
  after retrying for about 15 min, and was parsed as 0 turns at $0.
  `composite_score` is 0.85 to 0.95 for cells that were cancelled or blocked.

### Still not established

- Behaviour under landlock past startup: no eval cell runs sandboxed.
- A reviewer verdict for either arm.
- A paired comparison of cost and quality. That needs a fixture whose task
  gate is green at `pre_task_commit`, a cache-aware GLM price entry, and
  Z.ai quota for both arms in the same window.
