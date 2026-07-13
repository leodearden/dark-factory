# pi backend empirical spike — findings

**Task:** 2458 (T2 of `plans/harness-backend-reconnect-pi-prd.md`). **Consumer:** T4 / task 2463
(the pi backend — `_invoke_pi` + `_parse_pi_output`). **Date observed:** 2026-07-13.

**These are OBSERVED answers** — every claim below was produced by running headless `pi` on this
host and reading its real stdout / exit codes / on-disk session files (not docs). The reproducer is
`scripts/pi_spike_probe.sh`.

## Provenance / method

| | |
|---|---|
| `@earendil-works/pi-coding-agent` | **0.80.6** (`bin: pi`, PRD pin 0.80.x ✓) |
| `pi-mcp-adapter` (nicobailon, MIT) | **2.11.0** (PRD pin 2.11.x ✓) |
| node / npm | v22.22.3 / 10.9.8 (host; DF task sandbox is disabled) |
| provider used for *successful* runs | `google` / `gemini-3-flash` (cheap, provider-agnostic for the 7 questions) |
| Claude OAuth path | **validated** separately: pi accepted the live `~/.claude` OAuth token as an Anthropic bearer (progressed past 401 → a 400 "out of extra usage" error), i.e. auth plumbing works. The 400 is a **separate pay-as-you-go "extra usage" API bucket** hit by passing the OAuth token as a raw bearer from a third-party client — **not** the Claude Code subscription (which is healthy), so it says nothing about account/subscription state. Successful runs simply used Gemini. T4 note: how Anthropic meters OAuth-token API calls made *outside* the official client is worth confirming when wiring pi's Claude-OAuth path. |

Install + all probes are throwaway (a scratch `node_modules`, never committed; `node_modules/` is
already gitignored). The seven questions are PRD Appendix B.

> **Orphan-cause note (for the reconcile record):** the earlier *unattended* attempts at this task
> cut a degenerate zero-commit branch and died pre-commit. Root cause was **not** a sandbox/network
> block — node+npm are on the host, the pi packages install and run fine, and there was even a
> leftover pi session dir (`~/.pi/agent/sessions/--home-leo-src-dark-factory-.worktrees-2458-.task-pi-spike--`)
> proving a prior run *did* invoke pi. The likely cause is the unattended agent hanging on a headless
> pi call (pi drops to interactive/stdin-wait when `-p` is omitted, and has no self-timeout — see Q1/Q7)
> then being idle-reaped. An attended run installs + probes without incident.

---

## Answers to the seven questions

### Q1 — Headless exit-code semantics (`--mode json` / `-p`)

`pi --mode json -p` streams **newline-delimited JSON** to stdout; **each line is one complete JSON
object** (safe to `for line in stdout.splitlines(): json.loads(line)`, exactly like the codex JSONL
path). Exit-code mapping — **observed**:

| Case | exit | stdout | how to detect |
|---|---|---|---|
| success | **0** | full JSON stream, terminal assistant `stopReason:"stop"` | `stopReason != "error"` and no `errorMessage` |
| runtime API error (401 / 400 / 404, out-of-quota, bad model id) | **0** | full JSON stream, terminal assistant `stopReason:"error"` + `errorMessage:"…"`, `agent_end.willRetry:false` | `stopReason == "error"` or `errorMessage` present |
| CLI/usage error (unknown flag, etc.) | **1** | *empty*; `Error: Unknown option: …` on **stderr** | exit != 0 and empty stdout |
| harness timeout | n/a (see below) | partial | `_run_subprocess_local` sets `timed_out=True` |

**The load-bearing rule: exit 0 does NOT mean success.** A hard `401 invalid x-api-key` still exits
0 — the failure lives *in the stream* (`stopReason:"error"`). `_parse_pi_output` MUST decide success
from the terminal assistant message, not the exit code (contrast codex, whose `_parse_codex_output`
leans on `returncode == 0`). Concretely:

```
success = (result.returncode == 0)
          and (a terminal assistant message exists)
          and (terminal.stopReason != "error")
          and (not terminal.errorMessage)
```

**Timeout is harness-owned, not pi's.** pi has no self-timeout flag (Q7). Reuse
`_run_subprocess_local(..., timeout_seconds)` unchanged: its `asyncio.wait_for` path
`terminate_process_group`s and returns `returncode=1, timed_out=True` itself, independent of pi's
own exit code — byte-identical to how codex is timed out today.

### Q2 — Cost aggregation

There is **no** single aggregate "result total" event — **but pi computes cost natively**. Every
assistant message (in the stream and on disk) carries a fully-populated:

```json
"usage": {"input":632,"output":31,"cacheRead":0,"cacheWrite":0,"reasoning":29,"totalTokens":663,
          "cost":{"input":0.000316,"output":0.000093,"cacheRead":0,"cacheWrite":0,"total":0.000409}}
```

`cost.*` is **USD**, computed by pi from its own per-provider/model price data. **usage/cost is
per-message (per-turn), NOT cumulative** — observed a 2-turn run: turn1 `cost.total=0.0009355` +
turn2 `cost.total=0.000887` = **0.0018225** total (both the per-`turn_end` sum and the
`agent_end.messages[]` assistant-sum agree).

**Recommended source (simplest, single event):** the one `agent_end` event carries
`messages:[...]`; sum `usage.cost.total` over `role=="assistant"` messages for `cost_usd`, and
`usage.input/output/totalTokens` for tokens. `turns` = count of `turn_end` events (== assistant
messages). Equivalent alternative: accumulate over `turn_end` events during the stream.

**Implication for T-price:** pi does **not** need the config price table for cost — unlike codex
(tokens-only → `_MODEL_COSTS`/T-price), pi reports `cost.total` directly. Use pi's native cost; fall
back to T-price only if `cost.total` is absent/zero for an exotic `--provider`/custom-base-url combo
pi has no price data for.

### Q3 — Direct-tools generated tool-name format

MCP tools are exposed to pi via **pi-mcp-adapter**. Two modes matter:

* **proxy mode (default):** ONE tool named `mcp`; the agent self-discovers via `mcp({search})` then
  `mcp({tool, args})`. Only `mcp` is allowlistable — no good for a per-tool `--tools` allowlist.
* **direct-tools mode:** set **`"directTools": true`** on the server in the MCP config; each MCP tool
  is registered as a **top-level, individually-allowlistable** pi tool.

**Name formula** (adapter `types.ts::formatToolName`/`getServerPrefix`, and **confirmed by a live
run**): `<serverKey>_<mcpToolName>`, where the **serverKey**'s hyphens become underscores and the
mcp tool's own name is used **verbatim** (MCP names are already snake_case). Prefix mode is
`settings.toolPrefix`, default **`"server"`**:

| prefix mode | server key `fused-memory`, tool `add_memory` → |
|---|---|
| `server` (default) | `fused_memory_add_memory` |
| `short` (strip trailing `-mcp`/`mcp`) | `fused_memory_add_memory` (only differs when key ends in `-mcp`, e.g. `plan-tools-mcp` → `plan_tools_…`) |
| `none` | `add_memory` (bare) |

**Observed:** server key `spike-demo` + tool `echo_it` → generated name **`spike_demo_echo_it`**;
callable as `--tools spike_demo_echo_it`; the tool-result even carries `details:{server:"spike-demo",
tool:"echo_it"}`. For the DF servers T4 will allowlist: `escalation` → `escalation_<tool>` (e.g.
`escalation_merge_request`), `fused-memory` → `fused_memory_<tool>`, a stdio `plan-tools` →
`plan_tools_<tool>`.

**⚠ CRITICAL GOTCHA for T4:** direct-tools registration needs a **pre-built metadata cache**
(`~/.pi/agent/mcp-cache.json`). **Until the cache is populated, the adapter exposes only the proxy
`mcp` tool** and the generated `--tools <name>` don't exist (the call silently falls back to proxy /
fails the allowlist). Both states were observed on the same server. The cache is populated when the
adapter first **connects** to a server — `pi-mcp-adapter init` alone only does config
discovery/import and does **not** reliably populate a project-local server's tools (observed: a
fresh run that only ran `init` still fell back to proxy). T4 MUST warm the cache with **one real
connection** — a proxy `mcp({search})` call, or letting the adapter connect on first use — before
relying on the `--tools` allowlist of direct tool names.

### Q4 — Verdict-tool compliance / structured output

pi has **no decode-forced structured output**: no `--output-schema` / `--json-schema` flag, no
built-in `terminate`/verdict tool (confirmed by `--help` and a full dist grep — the only `terminate`
hits are unrelated process/image utils; the only `budget` hits are context-compaction budget).
Structured output on pi is **convention only** (instruct the agent to call a custom "submit" tool,
or parse the final assistant text).

The final assistant **text** (→ `AgentResult.output`) = the `content[]` entries with `type=="text"`
of the terminal assistant message, joined — available in `agent_end.messages[-1]` or the last
`turn_end.message`.

**Scope:** verdict-emitting roles (reviewer/judge/triage/merger) are **out of scope for T4** — PRD
§6/§9.9 assign structured-output replacement to **mcp-verdict-servers**. T4 dispatches
implementer/architect roles, which need only the final text + git side-effects. So T4 needs **no**
structured-output shim.

### Q5 — Liveness signal (watchdog analog of the Claude transcript glob)

Two surfaces:

1. **stdout `--mode json` stream (recommended liveness source):** fine-grained; during generation it
   emits `message_update` (`text_start`/`text_delta`/`text_end`) deltas, and during tool runs
   `tool_execution_update` deltas. It advances continuously — so the existing `_run_subprocess`
   stdout-advancing watchdog can watch **stdout directly**, no file glob needed (better than Claude,
   whose progress is only in the transcript file).
2. **on-disk session JSONL:** `<session-dir>/<ISO-ts>_<uuid>.jsonl`. Default dir is
   `~/.pi/agent/sessions/--<cwd-path-slug>--/` (slug = cwd with `/`→`-`); override with
   `--session-dir <dir>`. This is a **compact persisted** log — `session`, `model_change`,
   `thinking_level_change`, then one `type:"message"` record per **completed** message (each with full
   `usage`/`cost`). It grows only at message boundaries → **coarser** than stdout (a long single turn
   won't touch the file until the message finishes).

**T4 recommendation:** use the **stdout stream** as the liveness signal (reuse the `_run_subprocess`
idle-watchdog on stdout); point `--session-dir` at a per-task dir for a durable transcript +
post-hoc cost/verdict parsing.

### Q6 — bwrap / landlock wrap

pi is a **single Node process**; its `bash` tool spawns children that **inherit pi's environment and
namespace** — observed: a marker env var (`SPIKE_MARKER=…`) set on pi's parent env was visible inside
the `bash` tool's own subprocess output. `bwrap` 0.9.0 is present on the host.

⇒ Wrapping the whole `pi …` argv with `sandbox_dispatch.wrap_command(cmd, cwd, sandbox_modules)`
(exactly as `_invoke_codex` does) places **pi and all its tool children inside one
bwrap/landlock namespace**. `_run_subprocess_local` already uses `start_new_session=True` for
whole-tree signalling. **No new sandbox mechanism is needed** — pi wraps like codex. (Caveat, per PRD
§6: DF's task sandbox is currently disabled and neither backend network-isolates; this is
filesystem-scoping only.)

### Q7 — Turn / budget cap

**No native cap.** No `--max-turns` / `--max-steps` / `--max-budget` / turn-count flag exists
(confirmed by `--help` and a dist grep; every "budget" symbol is context-compaction budget, not
spend/turns). `max_budget_usd` and `max_turns` are **not natively enforceable** on pi.

**Recommendation (pilot):** external **wall-clock watchdog only** — reuse
`_run_subprocess_local(..., timeout_seconds)` (SIGTERM→SIGKILL the process group), identical to
codex; document `max_budget_usd`/`max_turns` as native-unsupported (PRD §9.4, parity with codex T3).
If a real runaway is later observed, a tiny follow-up can count turns: the stream already delimits
them with `turn_end` events, so a stdout-watching wrapper (or a small pi `agent_end`/turn-hook
extension) can kill at N turns without a pi core change.

---

## `_invoke_pi` / `_parse_pi_output` implementation guide (for T4 / task 2463)

Mirror the codex shape (`orchestrator/agents/invoke.py::_invoke_codex` / `_parse_codex_output`).

**Invocation template**

```
pi --provider <provider> --model <model>[:<thinking>] --mode json -p \
   --session-dir <per-task-dir> [--session-id <uuid> | --session <resume-id>] \
   --tools <csv-allowlist> [--exclude-tools <csv>] \
   --system-prompt <text-or-@file> [--no-context-files] \
   "<user-prompt>"
```

* **System prompt:** pass via `--system-prompt` (or `--append-system-prompt`, repeatable, accepts
  text or a file path). **pi writes NO instruction file into the worktree** (no AGENTS.md/GEMINI.md)
  → the codex `AGENTS.md`-in-diff hazard (T3) simply does not exist for pi. Add `--no-context-files`
  to stop pi auto-discovering the repo's own `AGENTS.md`/`CLAUDE.md` if that's unwanted.
* **Provider/model/thinking ← role config:** `--provider anthropic --model claude-haiku-4-5`
  (or `--model anthropic/claude-…`); map role `effort` → `--thinking {off|minimal|low|medium|high|xhigh|max}`.
* **Auth:** pi reads `ANTHROPIC_OAUTH_TOKEN` (Claude OAuth — validated), `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY` (google is pi's default provider), plus zai/DeepSeek/etc. Map
  the claude backend's `oauth_token` → `ANTHROPIC_OAUTH_TOKEN` for a Claude-provider pi run.
* **Sessions:** `--session-id <uuid>` pre-creates/uses an exact id (← `session_id`); `--session
  <path|id>` resumes (← `resume_session_id`). The id also appears in the first stdout
  `{"type":"session","id":…}` event (→ `AgentResult.session_id`).
* **MCP:** put the servers T4 needs in the MCP config with `"directTools": true`, **warm the cache
  with one connection** (a proxy `mcp({search})` call — not just `pi-mcp-adapter init`; see Q3), then
  build `--tools` from the role's allow-list mapped through the Q3 name formula
  (`<server '-'→'_'>_<tool>`).

**Event model for `_parse_pi_output`** (line-parse JSONL; the events, in order):

```
session          → session_id (present again on every message)
agent_start
turn_start
message_start    (assistant usage here is ZERO — pre-stream; DO NOT read cost here)
message_update*  ← IGNORE (streaming text_start/text_delta/text_end deltas)
tool_execution_start / tool_execution_update* / tool_execution_end   (toolName,args,result.content[],isError)
message_end      (per message)
turn_end         (once per assistant turn) → message.usage {tokens, cost}, message.stopReason
… repeat turn_start…turn_end per turn …
agent_end        (ONCE, authoritative) → messages:[full transcript w/ per-msg usage], willRetry
agent_settled    (final terminal marker)
```

Parse rules:
* **Ignore** `message_update` and `tool_execution_update` (streaming deltas).
* `cost_usd` = Σ `usage.cost.total` over `agent_end.messages[]` where `role=="assistant"`
  (or Σ over `turn_end`). tokens likewise. `turns` = #`turn_end`.
* `output` = join of `content[type=="text"].text` from the terminal assistant message.
* `session_id` = the `session` event's `id`.
* `stopReason` values seen: `"stop"` (natural), `"error"` (API failure). Treat **only** `"error"`
  (or a present `errorMessage`) as failure; other providers may also emit `tool_calls`/`max_tokens`/
  `aborted` — none of those are failures.
* `success` per the Q1 rule (exit0 + terminal `stopReason!="error"` + no `errorMessage`).
* On exit!=0 with empty stdout → CLI/process failure (like codex's `error_empty_output` path);
  propagate `stderr`. `timed_out` and `duration_ms` come from `_run_subprocess_local`'s
  `_SubprocessResult` unchanged.

**`AgentResult` field mapping:** `success`, `output`, `cost_usd` (native pi cost),
`duration_ms` (harness), `turns` (#turn_end), `session_id`, `subtype` (`'success'`|`'error'`),
`stderr`, `timed_out` (harness) — same shape codex fills.

**Dispatch wiring:** add `elif backend == 'pi': return await _invoke_pi(...)` to
`invoke_agent` (invoke.py) and `'pi'` to the `BackendsConfig` docstring, per PRD T4.
