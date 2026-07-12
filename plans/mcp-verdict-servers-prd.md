# PRD: Role-specific MCP verdict servers (replace structured-output + substring contracts)

**Status:** active — authored 2026-07-12 (design session; user AFK-by-design,
recommended defaults adopted for the author's-call items, all noted in
§Resolved). **Project:** dark_factory. **Approach:** B+H (contract +
boundary-test sketch; G5 heuristic hit: ≥5 orchestrator modules, load-bearing
completion-gating seam, 2 cross-PRD consumers). Origin brief:
`~/.claude/spawn-briefs/2026-07-12-mcp-verdict-servers.md` (pre-answers G1–G6;
its recorded decisions are honored, not re-derived). Substrate re-verified this
session against main @ `7b6c1f829c`.

## Goal

Replace the four fragile output contracts by which the orchestrator's
**reviewer**, **judge**, **triage**, and **merger** roles communicate a verdict
back to the workflow — three `--json-schema` structured-output contracts and one
`'BLOCKED' in output.upper()` substring grep — with a small **per-worktree MCP
verdict-tool server**, extending the proven plan-tools MCP side-effect pattern
(`orchestrator/mcp/plan_tools.py`) already used by the architect/implementer/
simple_task roles.

Each verdict role calls a role-specific MCP tool (`submit_review_verdict`,
`submit_completion_verdict`, `submit_triage`, `submit_merge_disposition`); the
tool persists a **verdict artifact** under the task's `.task-meta/<name>/
verdicts/<role>.json`; the workflow reads the artifact after the agent returns —
never parsing transcript text or grepping for a word. An absent verdict yields
the same fail-safe disposition the role's failure path yields today.

User-observable surfaces (per-role, decompose §):
- **merger:** a merger run whose prose *mentions* "BLOCKED" while it calls
  `submit_merge_disposition(blocked=false)` → the workflow **proceeds** (today's
  substring grep false-blocks it — a real latent bug this PRD removes).
- **reviewer/judge/triage:** the verdict arrives as a constrained tool call
  persisted as an artifact; the workflow consumes the artifact and ignores any
  surrounding prose; an absent tool call falls to the role's existing safe
  disposition (ERROR / keep-iterating / steward-inline).
- **eval + non-Claude harnesses:** a contract-agnostic on-disk verdict artifact
  the eval-revival scorer and the codex/pi verdict roles (harness-reconnect-pi)
  can consume without a `--json-schema` equivalent.

## Background — why MCP tools, why now

The four current contracts and their fragility (verified this session, refs on
main @ `7b6c1f829c`):

- **reviewer** (`workflow.py:5527-5577`, `_run_reviewer`): `output_schema=
  review_schema` (`:5533-5555`, `verdict∈{PASS,ISSUES_FOUND}`), consumed as
  `result.structured_output` with a **bare `json.loads(result.output)` fallback**
  (`:5566`) → synthesized `verdict:'ERROR'` on any parse failure (`:5572-5577`).
  Five reviewers run this concurrently (`ALL_REVIEWERS`, `:5486`).
- **judge** (`workflow.py:4960-5030`, `_run_completion_judge`): `output_schema=
  COMPLETION_JUDGE_SCHEMA` (`:4993`; schema `briefing.py:32-44`), consumed as
  `result.structured_output` only (`:5014`), missing keys → `None`. **Safety-
  critical — gates task completion** (`workflow.py:4775-4801`, opt-in via
  `judge_after_each_iteration`, always-on in eval mode).
- **triage** (`agents/triage.py:184-231`, `TRIAGE_OUTPUT_SCHEMA`): consumed by
  `parse_triage_result` (`triage.py:270-282`); invoked in the **steward**, not
  `workflow._invoke` — `steward.py:599-668` (`_pre_triage_escalation`), with a
  graceful **inline-triage fallback** when parse returns `None` (`:644-650`).
- **merger** (`workflow.py:6332-6335`): plain `_invoke(MERGER, …)`, **no schema
  at all** — disposition decided by `'BLOCKED' in merger_result.output.upper()`.
  This **false-trips whenever the model merely mentions the word** (e.g.
  "this is not a case to mark BLOCKED"), on any model including Claude.

All three `--json-schema` contracts ride the CLI's synthetic `StructuredOutput`
tool + deny-list dance (`shared/cli_invoke.py:174-217`, `1256-1278`) — a
CLI-version-fragile surface (the 2.1.168 incident). And the mechanism has **no
portable equivalent** on codex/pi, blocking those harnesses for verdict roles.

The **plan-tools precedent** already solves exactly this shape for the
plan-building roles: the architect's escape-hatch tools (`report_blocking_
dependency`, `report_false_premise`, `report_task_already_done`,
`report_unactionable_task`, `plan_tools.py:392-468`) each **write an artifact
via `TaskArtifacts` and return `{status: ok}`; the workflow reads the artifact
after the agent returns and acts deterministically** (`plan_tools.py:398-405`
states this contract verbatim). Memory: *"MCP tools convert free-form outputs
into constrained function calls, improving reliability"* and *"the architect
plan-MCP precedent prevents the LLM from smuggling a bare/​truncated id into the
report."* This PRD extends that identical pattern to the four verdict roles.

## Resolved design decisions

Author's-call items the brief delegated ("or an extension of plan-tools —
author's call"; artifact format/location; judge transition window; fold-ins).
User AFK-by-design → recommended defaults adopted, each recorded here.

1. **Separate `verdict-tools` per-worktree *stdio* server** (new
   `orchestrator/mcp/verdict_tools.py`), NOT an extension of plan-tools and NOT
   an HTTP server. Rationale: (a) leaves the architect/implementer plan-tools
   **hot path byte-untouched** — zero risk to the highest-value, most-tuned
   invocation path; (b) a per-worktree stdio server binds to the task's
   `--worktree`/`--meta-root` so the tool writes the verdict where the workflow
   reads it (an HTTP server has no clean per-invocation worktree binding);
   (c) mirrors `plan_tools.py` structure line-for-line for reviewability.

2. **One exposed tool per server invocation, selected by `--verdict-role`.**
   The server is launched with `--verdict-role <role>` and registers **exactly
   that role's** `submit_*` tool, writing `verdicts/<role>.json`. The artifact
   filename is therefore **authoritative from the orchestrator**, never chosen
   by the agent — a reviewer cannot misname its artifact onto a sibling's path.
   For the 5-reviewer panel, each reviewer's server gets `--verdict-role
   <reviewer_name>` and writes `verdicts/<reviewer_name>.json` (mirrors the
   existing `reviews/<name>.json` convention, `artifacts.py:492-509`).

3. **Verdict envelope** (the eval-revival seam contract), written to
   `.task-meta/<name>/verdicts/<role>.json`:
   ```json
   {
     "role": "<reviewer-name>|judge|triage|merger",
     "schema_version": 1,
     "session_id": "<uuid of the emitting invocation>",
     "emitted_at": "<ISO-8601 UTC>",
     "verdict": { …role-specific payload matching the EXISTING schema… }
   }
   ```
   `verdict` is the unchanged role payload (reviewer: the `review_schema` object;
   judge: the `COMPLETION_JUDGE_SCHEMA` object; triage: the
   `TRIAGE_OUTPUT_SCHEMA` object; merger: `{blocked: bool, reason: str}` — a new
   minimal schema replacing the substring). `schema_version` lets eval-revival
   evolve; `session_id` lets it correlate to a run.

4. **Freshness via pre-invocation clear (invariant I-FRESH).** The verdict
   artifact lives in `.task-meta/`, which **survives worktree resets and is
   reused across pooled lanes** (`artifacts.py:179-204` documents the stale-lane
   hazard). The judge/reviewer also run **multiple iterations per task**. So the
   workflow **clears `verdicts/<role>.json` immediately before each
   verdict-emitting spawn**; after the agent returns, **present ⇒ parse+use,
   absent ⇒ safe fallback**. This makes cross-iteration and cross-pooled-lane
   staleness impossible and makes "absent verdict ⇒ fallback" a genuine,
   testable rejection mechanism (G6 branch 4).

5. **Tool params ARE the existing schema's fields** — explicit typed parameters
   (`submit_completion_verdict(complete, reasoning, uncovered_plan_steps,
   substantive_work)`), NOT a single opaque `verdict: dict` blob. This preserves
   the constrained-function-call reliability benefit that motivates the PRD; a
   blob wrapper would re-introduce a decode step and lose the constraint.

6. **Fail-safe fallbacks preserved, unchanged in meaning:** reviewer absent →
   `verdict:'ERROR'` (retried, then aggregated as error); judge absent → `None`
   → keep iterating (never a false completion); triage absent → steward inline
   triage; merger absent → **blocked-equivalent** (fail-safe: an unresolved
   merger blocks). "Absent" = the artifact is missing after a cleared
   pre-invocation slot.

7. **Judge migration uses a transition window; merger/reviewer/triage do clean
   swaps.** The judge is completion-gating and this orchestrator **self-hosts the
   tasks that implement this very PRD** — a botched judge cutover that silently
   returned "not complete" would wedge completion for those tasks. So the judge
   migration (ζ) **keeps `output_schema=COMPLETION_JUDGE_SCHEMA` and reads the
   verdict artifact first, falling back to today's `result.structured_output`**;
   a terminal cleanup task (η) removes the fallback only after real-run
   verification. The merger/reviewer/triage swaps are clean because their
   fallbacks are *already* fail-safe (ERROR / steward-inline / blocked-equiv),
   so no transition scaffold is needed.

8. **Shared machinery is OUT OF SCOPE and must keep working.** The cleanup (η)
   removes `output_schema`/substring usage **from the four orchestrator roles
   only**. It does **not** touch the shared `cli_invoke` `--json-schema` /
   `StructuredOutput` / deny-list machinery (`cli_invoke.py:174-217`,
   `1256-1278`), which fused-memory recon / curator / path-scope-adjudicator
   still ride (`task_curator.py:2002,2093`; `reconciliation/stages/base.py:210-
   226`; `cli_stage_runner.py:262,315`; `agent_loop.py`). η ships a regression
   guard asserting that path still functions.

9. **Fold-ins.** (a) The `done_provenance` briefing bug — `briefing.py:863`
   teaches a **kind-less** `done_provenance` shape that fused-memory's
   `_validate_done_provenance` **rejects** (kind required, `task_interceptor.py:
   4199-4213; note-alone rejected :4227-4234`), contradicting the *correct*
   steward system prompt (`roles.py:885-941`) — **INCLUDED** as an independent
   one-line companion task (ι), no dep on the verdict work. (b) The reviewer's
   **inert `mcp__jcodemunch__*` grant** (`roles.py:426`; never materializes
   because the reviewer is in no MCP-config gate) — **REMOVED**, folded into the
   reviewer migration (δ) since it edits the same `allowed_tools` line.

## Contract (B+H)

### Server CLI

```
python -m orchestrator.mcp.verdict_tools \
    --worktree <wt> --meta-root <mr> --verdict-role <role>
```
Mirrors `plan_tools.py:815-859` (`_artifacts_from_args` + `main()`). Constructs
`TaskArtifacts(worktree, meta_root)` — the **identical** root the orchestrator
builds via `_meta_root_for_worktree` (`workflow.py:128-139`), so a verdict
written agent-side is transparently read orchestrator-side. Launch dict built by
`verdict_tools_mcp_server(worktree, meta_root, role)` in `mcp_lifecycle.py`,
modeled on `plan_tools_mcp_server` (`mcp_lifecycle.py:69-144`, no-uv hot path +
uv fallback + `MCP_TIMEOUT` env via `apply_mcp_startup_env`).

### Tools (one registered per invocation, per `--verdict-role`)

| Tool | Params (= existing schema fields) | Writes |
|---|---|---|
| `submit_review_verdict` | `reviewer:str`, `verdict:'PASS'|'ISSUES_FOUND'`, `issues:list[obj]`, `summary:str` (shape = `workflow.py:5533-5555`) | `verdicts/<reviewer>.json` |
| `submit_completion_verdict` | `complete:bool`, `reasoning:str`, `uncovered_plan_steps:list[str]`, `substantive_work:bool` (= `COMPLETION_JUDGE_SCHEMA`) | `verdicts/judge.json` |
| `submit_triage` | `accepted:list[obj]`, `skipped:list[obj]`, `proposed_task_groups:list[obj]` (= `TRIAGE_OUTPUT_SCHEMA`) | `verdicts/triage.json` |
| `submit_merge_disposition` | `blocked:bool`, `reason:str` (NEW minimal schema) | `verdicts/merger.json` |

Each handler = a standalone testable `_impl(artifacts, role, …) -> {status}` +
a thin `@mcp.tool()` wrapper (the `plan_tools.py:126-157` shape); it validates
its params, wraps them in the §Resolved-3 envelope, and `artifacts.write_verdict
(role, envelope)`.

### `TaskArtifacts` additions (`artifacts.py`)

- `write_verdict(role, envelope)` → `_write_json(self.root / 'verdicts' /
  f'{role}.json', envelope)` (mirrors `write_review`, `:492-509`).
- `read_verdict(role) -> dict | None` (mirrors `read_reviews`, `:511-518`;
  `None` on absent/unparseable).
- `clear_verdict(role)` → `(self.root / 'verdicts' / f'{role}.json').unlink
  (missing_ok=True)` (reuses the existing unlink helper, `:397-398`).
- `verdicts/` created in `init()` alongside `reviews/` (`:219`).

### Wiring (`workflow.py`, `mcp_lifecycle.py`, `roles.py`)

- `_VERDICT_TOOLS_ROLES = (<reviewer names…>, 'judge', 'merger')` +
  `_inject_verdict_tools_mcp(mcp_config, cwd, role)` mirroring
  `_PLAN_TOOLS_ROLES` / `_inject_plan_tools_mcp` (`workflow.py:125,142-175`).
  Injection at the `_invoke` spawn site (`workflow.py:7397`-adjacent), gated
  `role.name in _VERDICT_TOOLS_ROLES and cwd`. Because `_inject_*` **creates a
  `{'mcpServers':{}}` skeleton when `mcp_config` is `None`**, reviewer/merger/
  judge acquire the verdict server **without** being added to `_MCP_CONFIG_ROLES`.
- **Triage is bespoke:** its invocation is in `steward.py:599-668`, not
  `workflow._invoke`. ε injects the verdict server + grants `submit_triage` in
  that call's tool list (`steward.py:622-627`) and reads `read_verdict('triage')`
  — the `_inject_verdict_tools_mcp` helper is reused there.
- Per-role tool grants in `roles.py` (mirroring `_PLAN_CREATOR_TOOLS`,
  `roles.py:80-104`): a `_VERDICT_TOOLS = ['mcp__verdict-tools__submit_*']`
  family added to each role's `allowed_tools`.

### Invariants

- **I-FRESH** (§Resolved-4): workflow clears `verdicts/<role>.json` before every
  verdict-emitting spawn; absent-after-return ⇒ safe fallback. No stale verdict
  is ever consumed (cross-iteration or cross-pooled-lane).
- **I-AUTHORITATIVE-PATH** (§Resolved-2): artifact filename derives from the
  server's `--verdict-role` CLI arg, never from an agent-supplied field.
- **I-FAIL-SAFE** (§Resolved-6): absent/malformed verdict ⇒ the role's existing
  worst-case disposition, never a more-permissive one. Specifically: merger
  absent ⇒ blocked; judge absent ⇒ not-complete; reviewer absent ⇒ ERROR;
  triage absent ⇒ steward inline.
- **I-SHARED-INTACT** (§Resolved-8): no edit to `cli_invoke` `--json-schema` /
  `StructuredOutput` / deny-list; fused-memory recon/curator/adjudicator
  `output_schema` callers unaffected — regression-guarded in η.
- **I-CONTRACT-AGNOSTIC-ARTIFACT** (G4): the envelope at `verdicts/<role>.json`
  is the single documented cross-PRD verdict format; eval-revival reads it,
  harness-reconnect-pi's codex/pi roles emit it.

## Pre-conditions for activating

None external. All substrate exists on main @ `7b6c1f829c` (G3 verified this
session): the plan-tools stdio server + `TaskArtifacts` side-effect pattern
(`plan_tools.py`, `artifacts.py`), the `_PLAN_TOOLS_ROLES`/`_inject_plan_tools_
mcp` injection template (`workflow.py:125,142-175`), the four parse sites
(`workflow.py:4993,5014,5557-5577,6334`; `triage.py:270-282`; `steward.py:644-
650`), the role prompts (`roles.py:422,518-519,543-588`), the meta-root
derivation (`artifacts.py:179-204`), and the fused-memory `output_schema`
callers to preserve. No novel syntax, endpoint, schema migration, or flag.

## Cross-PRD relationship (G4)

Two sibling PRDs authored concurrently in this checkout; **this PRD OWNS both
seams** (the tool contract and the artifact envelope). No reciprocal ownership
ambiguity — both siblings' briefs state they CONSUME and this PRD OWNS.

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `~/.claude/spawn-briefs/2026-07-12-harness-reconnect-pi.md` | consumes | the `mcp__verdict-tools__*` tool contract for codex/pi verdict roles (`.codex/config.toml` / pi direct-tools `terminate:true`) | **this PRD** | queued — codex/pi verdict-role trialability deps on α/β; wired from the **sibling's** decomposition (dependent holds the gate) |
| `~/.claude/spawn-briefs/2026-07-12-eval-revival.md` | consumes | the `verdicts/<role>.json` envelope (contract-agnostic scoring — read the artifact, not the transcript) | **this PRD** (owns the envelope format/location) | queued — eval scoring deps on α; wired from the sibling's decomposition |

The `implementer`/`architect` roles do **not** gate on this PRD (their contracts
are git commits + plan-tools MCP side effects already) — harness-reconnect-pi
can proceed on those roles in parallel. This PRD wires **no** dep *toward* the
siblings; each sibling's decompose session adds its `dark_factory:<α|β id>` dep
from its own side.

## Decomposition plan

DAG: **α → β → {γ merger, δ reviewer, ε triage, ζ judge} → η → θ**; **ι**
independent. Greek labels; real ids assigned at decompose. Signals per G2/G6.

- **α — verdict-tools server + artifact contract** (intermediate → unlocks
  β + eval-revival). New `orchestrator/mcp/verdict_tools.py` (FastMCP factory,
  `--worktree/--meta-root/--verdict-role` CLI, four `submit_*` tools + `_impl`
  fns, one-tool-per-role selection) + `TaskArtifacts.write_verdict/read_verdict/
  clear_verdict` + `verdicts/` in `init()` + the §Contract envelope. Modules:
  `orchestrator/mcp/verdict_tools.py` (new), `orchestrator/artifacts.py`.
  *Consumer:* β + the eval-revival scorer (named). Unit signal: each `_impl`
  writes a schema-valid envelope; write/read/clear roundtrip; `_artifacts_from_
  args` parity with plan-tools. (Foundation — roped into θ as its user-facing
  proof.)
- **β — orchestrator wiring** (intermediate → unlocks γ/δ/ζ). `_VERDICT_TOOLS_
  ROLES` (reviewer-names + judge + merger), `_inject_verdict_tools_mcp`,
  `verdict_tools_mcp_server` (`mcp_lifecycle.py`), the `_VERDICT_TOOLS` grant
  family (`roles.py`). No parse-site change yet — servers become *available* to
  the roles. Modules: `orchestrator/workflow.py`, `orchestrator/mcp_lifecycle.py`,
  `orchestrator/agents/roles.py`. *Consumer:* γ/δ/ζ (in-batch). Depends α.
- **γ — migrate merger** (leaf). Grant `submit_merge_disposition` (merger
  `allowed_tools`, `roles.py:590`); merger prompt: call the tool instead of
  emitting "BLOCKED" prose (`roles.py:543,569,588`); clear `verdicts/merger.json`
  before the merger spawn and **replace the substring grep** (`workflow.py:6334`)
  with `read_verdict('merger')` — present ⇒ `blocked` bool; absent ⇒
  blocked-equivalent (I-FAIL-SAFE). **Signal (user-observable + G6 rejection):**
  a merger transcript that *mentions* "BLOCKED" in prose while calling
  `submit_merge_disposition(blocked=false)` → workflow **proceeds** (no
  `_mark_blocked`), where today's substring grep false-blocks it. Depends β.
- **δ — migrate reviewer panel** (leaf). Grant `submit_review_verdict` +
  **remove the inert `mcp__jcodemunch__*` grant** (`roles.py:426`, fold-in κ);
  reviewer prompt: call the tool (`roles.py:422`, drop pure-JSON-fences
  instruction + fenced example `:389-408`); in `_run_reviewer` clear
  `verdicts/<name>.json` before each reviewer spawn and read it back, replacing
  the `structured_output`/`json.loads` cascade (`workflow.py:5561-5577`); absent
  ⇒ `verdict:'ERROR'` (existing retry/aggregate path unchanged, `:5490-5525`).
  **Signal:** a reviewer that emits its verdict via the tool alongside prose →
  the workflow consumes the artifact and ignores the prose; a reviewer that
  never calls the tool → `ERROR` verdict (retried). Depends β.
- **ε — migrate triage** (leaf; steward path). In `steward.py:_pre_triage_
  escalation` (`:599-668`): inject the verdict server + grant `submit_triage`
  (`:622-627`), clear `verdicts/triage.json`, read `read_verdict('triage')`
  instead of `parse_triage_result(structured_output)` (`triage.py:270-282`);
  absent ⇒ steward inline triage (existing fallback, `:644-650`). Triage prompt
  (`build_triage_prompt`, `triage.py:234`) instructs the tool call. **Signal:** a
  triage run emits accepted/skipped/proposed_task_groups via the tool → the
  steward files the follow-up tasks; a triage run that never calls the tool →
  steward falls back to inline triage. Depends β.
- **ζ — migrate judge (transition window)** (leaf). Grant `submit_completion_
  verdict` (`roles.py:521`); judge prompt: call the tool, drop the `--json-
  schema` sentence (`roles.py:518-519`); in `_run_completion_judge` clear
  `verdicts/judge.json`, then **prefer `read_verdict('judge')`, else fall back to
  today's `result.structured_output`** (KEEP `output_schema=COMPLETION_JUDGE_
  SCHEMA`, `workflow.py:4993,5014`); absent-both ⇒ `None` ⇒ keep iterating
  (I-FAIL-SAFE). **Signal:** with `judge_after_each_iteration` on, a completion
  decision (`workflow.py:4775-4801`) gated on a judge verdict that arrived via
  the tool; absent tool ⇒ today's structured-output path (transition) ⇒ `None`.
  Depends β. Land after γ/δ/ε (defense-in-depth for the completion-gating seam).
- **η — terminal cleanup + shared-machinery regression guard** (leaf). After all
  four migrated and verified: remove the **judge transition scaffold** (drop
  `output_schema=COMPLETION_JUDGE_SCHEMA` + the `structured_output` read, leaving
  artifact-read + `None` fallback); confirm the reviewer `json.loads` fallback
  and merger substring are gone. **Do NOT touch shared `cli_invoke`.** **Signal
  (G6 negative + positive):** `git grep` shows none of the four roles pass
  `output_schema` / grep transcript for a disposition; **and** a regression test
  proves the fused-memory recon/curator `output_schema` path
  (`cli_invoke.py:1256-1278`) still functions (I-SHARED-INTACT). Depends
  γ,δ,ε,ζ.
- **θ — integration gate (B+H boundary-test signal)** (leaf). End-to-end
  scenarios through the workflow test rig with a fake agent runner (`invoke_fn`)
  that emits verdict tool-calls (writes the envelope) — the §Boundary-test sketch
  is θ's observable signal. Depends γ,δ,ε,ζ,η.
- **ι — done_provenance briefing fix** (leaf; independent companion, fold-in 9a).
  One-line `briefing.py:863` edit teaching the `kind` field (align with the
  validator + the correct steward system prompt `roles.py:885-941`). **Signal
  (G6 rejection):** a steward escalation-resolution prompt now yields a
  `done_provenance` the fused-memory `_validate_done_provenance` **accepts**
  (`task_interceptor.py:4098-4234`), where the current kind-less shape
  hard-errors (`:4199-4213`). Depends none.

G2 note: α/β are intermediates (in-batch consumers β/γ/δ/ε/ζ); γ/δ/ε/ζ/η/θ/ι
are user-observable leaves. G5: θ is the B+H integration-gate whose signal is the
boundary-test sketch, closing the G2 loop.

## Boundary-test sketch (θ's signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Merger emits `blocked=false` with "BLOCKED" in prose | fake merger runner writes `verdicts/merger.json {blocked:false}` + prose mentioning BLOCKED | workflow **proceeds** (no `_mark_blocked`); the pre-γ substring grep would have blocked (the removed-bug proof) |
| 2 | Merger emits `blocked=true, reason` | writes `{blocked:true, reason}` | `_mark_blocked(reason,…)` fires with the tool-supplied reason |
| 3 | Merger emits **no** verdict | runner returns without calling the tool; slot cleared | blocked-equivalent disposition (I-FAIL-SAFE) |
| 4 | Reviewer emits verdict + prose | writes `verdicts/<name>.json` PASS/ISSUES + prose | `aggregate_reviews` consumes the artifact; prose ignored; `write_review` path unchanged |
| 5 | Reviewer emits no verdict | slot cleared, no tool call | `verdict:'ERROR'` synthesized → retry path (`:5490-5507`) |
| 6 | Judge emits `complete=true, substantive_work=true` | `judge_after_each_iteration` on; writes `verdicts/judge.json` | early-exit completion (`:4785-4801`) |
| 7 | Judge emits `complete=true, substantive_work=false` | writes that shape | verdict **ignored** (existing safety, `:4780-4784`); loop continues |
| 8 | Judge transition fallback | ζ live, η not yet; runner emits **legacy** `structured_output`, no tool | workflow still gates correctly via the structured-output fallback (transition-window proof) |
| 9 | Judge absent-both | slot cleared, no tool, no structured output | `None` ⇒ keep iterating (never false-complete) |
| 10 | Triage emits groups | writes `verdicts/triage.json` | steward files the follow-up tasks |
| 11 | Triage absent | slot cleared, no tool | steward inline-triage fallback (`:644-650`) |
| 12 | **Stale-artifact freshness** (I-FRESH) | `verdicts/judge.json` left by a PRIOR iteration; current judge spawn writes nothing | pre-spawn clear removed it ⇒ current read is absent ⇒ `None` (NOT the prior verdict) |
| 13 | **Pooled-lane staleness** (I-FRESH) | `.task-meta/<name>/verdicts/*` left by a prior task on the reused lane | first verdict spawn on lane acquisition sees a cleared slot, not the prior task's verdict |
| 14 | **Shared machinery intact** (I-SHARED-INTACT) | fused-memory recon/curator `output_schema` call after η | `--json-schema` path still produces `structured_output`; recon/curator unaffected |

## Out of scope

- The shared `cli_invoke` `--json-schema` / `StructuredOutput` / deny-list
  machinery and all fused-memory recon / curator / path-scope-adjudicator
  `output_schema` callers (§Resolved-8; regression-guarded, not modified).
- The `architect`/`implementer`/`simple_task`/`debugger` plan-tools contracts
  (untouched — §Resolved-1).
- Wiring the reviewer's jcodemunch tools (the inert grant is *removed*, not
  wired — jcodemunch enablement is a separate concern).
- Consuming the envelope from eval-revival or emitting it from codex/pi — owned
  by the sibling PRDs (they dep on α/β from their side, §Cross-PRD).
- Recording judge/steward/triage cost in CostStore (owned by
  harness-reconnect-pi).

## Open questions (tactical — deferred, not design-blocking)

1. **Reviewer server per-invocation vs shared.** Five reviewers each spawn a
   verdict-tools server bound to the same worktree writing distinct
   `verdicts/<name>.json`. **Suggested:** per-invocation (matches plan-tools;
   distinct filenames ⇒ no write contention). Decide in δ.
2. **`submit_merge_disposition` reason on `blocked=false`.** Require a non-empty
   `reason` only when `blocked=true`, or always. **Suggested:** required always
   (cheap, aids audit); empty allowed when `blocked=false`. Decide in γ.
3. **Envelope `session_id` source.** The workflow allocates `session_id_val` per
   invocation (`workflow.py:7426`) but the *server* (agent-side) doesn't see it.
   **Suggested:** pass it via a `--session-id` CLI arg to the verdict server
   (or omit `session_id` from the envelope in v1 and let eval-revival correlate
   by `emitted_at` + task id). Decide in α.
4. **Integration-rig home.** Reuse the existing workflow test rig with a fake
   `invoke_fn`, or add a dedicated verdict-tools fixture. **Suggested:** reuse
   the existing workflow test suite's fake-runner pattern. Decide in θ.
