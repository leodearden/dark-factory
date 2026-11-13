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
- Behaviour under the landlock sandbox.
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

All seven are left to task 8939 (from ticket `tkt_0RWSZ5GKKD78GK8FKD16G2X67J`),
the operator task that installs the ZCode CLI on the orchestrator host and runs
the paired `glm-5.3-zcode` / `glm-5.3-endpoint` eval cells. Task 8939's live
run should also confirm that an implementer step commits under the arm. Until
task 8939 is done, the arm is built and awaiting its live run, not measured.
