# Capability manifest — escalation store ambiguity ("wrong store" vs "genuine absence")

Binds every leaf signal's asserted capabilities to evidence, mechanizing G3 + G6.
Machine-readable twin: `escalation-store-ambiguity-prd.capability-manifest.yaml`.
PRD: `plans/escalation-store-ambiguity-prd.md` (committed `1da50b070c`).

**Verdict summary: 21 bindings, 21 PASS, 0 blocking.** Nothing in this batch is
declared-only, producer-downstream, or extent-short. The one binding that could
have inverted the DAG — γ4's B6 (`get_task_escalation_history` answers what
`get_pending_escalations` cannot) — is homed on **γ4** with **β upstream by a real
dependency edge**, and both halves of B6 were verified *on disk today* before the
tool exists, so β's own signal carries no unproven premise.

The one premise deliberately **not** tightened: α's signal asserts the stage
"cannot obtain an escalation-read result", **not** that the CLI returns a denial.
Whether `--disallowed-tools` rejects a denied MCP tool on call or omits it from the
listing is unverified (PRD open question 4, decided during α). The signal, B8, and
the binding below all hold under either. Do not reword this into a rejection claim
without first observing a live stage spawn.

## Substrate findings that shaped the bindings

| Capability | Status at decompose (re-verified at `1da50b070c`) | Consequence |
|---|---|---|
| Stage disallow-list hook + argv plumbing | **CONFIRMED** — `stages/base.py:97` `get_disallowed_tools()`, 3 overrides (`memory_consolidator.py:654`, `task_knowledge_sync.py:2627`, `:3656`); `base.py:287` → `cli_stage_runner.py:393` → `shared/cli_invoke.py:1537` `cmd.extend(['--disallowed-tools', *disallowed_tools])` | α is a constants edit against a live wire, not new plumbing |
| `STAGE2_DISALLOWED = DISALLOW_BUILTIN` | **CONFIRMED DEFECT** — `cli_stage_runner.py:78`; Stage 2 denies only built-ins, never `mcp__escalation__*` | that omission is the whole bug; all three lists must cover the reads |
| MCP tool names honored in disallow lists | **CONFIRMED** — `mcp__fused-memory__delete_entity`, `…__submit_task`, … already listed (`cli_stage_runner.py:38-49`) and production-proven | name-matching is not a new capability α must invent |
| `escalate_blocker` is the ONLY sanctioned recon escalation use | **CONFIRMED** — `prompts/stage2.py:623`, `:647-648` (FIX D stale-flag case); `get_pending_escalations` appears in **zero** recon prompts | α denies the two reads and must **not** deny the write |
| `queue.get_by_task(task_id, status=None)` is archive-inclusive | **CONFIRMED** — `queue.py:396-430`: `status != 'pending'` extends the candidate paths with `_iter_archive_paths(...)`; docstring states the two-tier scan explicitly | β is a thin wrapper over a proven primitive, not new scan logic |
| B6 holds on disk **before** the tool exists | **CONFIRMED EMPIRICALLY** — against `/home/leo/src/reify/data/escalations`: `get_by_task('5534', status='pending')` → `[]` while `get_by_task('5534')` → `esc-5534-1`, `status='resolved'`, `level=2`; same for `5557` | β's signal asserts nothing unproven; γ4's B6 is satisfiable the moment the tool lands |
| Pending-only semantics are spec-locked | **CONFIRMED** — `escalation/tests/test_queue.py:376` (`test_get_by_task_status_pending_excludes_archive`), `:386` (`test_get_pending_excludes_archive`) | β must not alter either; bound as a non-regression check |
| `create_server` has **exactly two** production call sites | **CONFIRMED** — `orchestrator/harness.py:9097`, `reconciliation/harness.py:1909` (imported as `create_escalation_server`, `harness.py:100`), plus one test patch point `fused-memory/tests/test_harness.py:605` | γ1's blast radius is exact. `orchestrator/mcp/plan_tools.py:962` and `verdict_tools.py:320` are **different functions sharing the name** — do not touch |
| `config.project_id` exists and is populated | **CONFIRMED** — `orchestrator/config.py:955`, `Field(default='dark_factory')`; populated in all 8 per-project `dark-factory-orchestrator.yaml` | γ1's `kind='project'` identity has a real source |
| `_require_matching_project_root` exists | **CONFIRMED** — `escalation/server.py:115-137`, currently wired to only `claim_warm_worktree` / `release_warm_worktree` (`:1820`, `:1890`); its docstring already carries the generalizing rationale | γ2 reuses it — but it reads `harness.git_ops.project_root` and the recon server is **harness-less**, so γ2 must generalize it to take the `StoreIdentity` |
| Consecutive-streak escape house pattern | **CONFIRMED** — `orchestrator/merge_liveness.py:446-501` (generalized by task 2558) | γ2's INV-4 escape has a pattern to follow, not invent |
| fastmcp `description=` overrides the docstring | **CONFIRMED EMPIRICALLY** — installed fastmcp is **3.2.2** (re-verified at decompose); `@mcp.tool(description=<text>)` makes `(await mcp.get_tool(name)).description` return `<text>`, not the docstring. Note `FastMCP` has `get_tool` (singular) | γ3's whole mechanism is real; 21 `@mcp.tool` sites in `escalation/server.py` |

## Bindings

### α — deny escalation read tools to reconciliation stage agents

| Capability | Binding | Verdict |
|---|---|---|
| `recon-stage-disallow-covers-escalation-reads` | capability→producer (wired) — α edits `cli_stage_runner.py:77-84`; the hook and the full argv path to `--disallowed-tools` are confirmed live above | PASS |
| `escalation-write-stays-sanctioned` | rejection-mechanism, inverted — `escalate_blocker` must **not** appear in any disallow list; it is the sole sanctioned recon use (`prompts/stage2.py:623,647-648`), so over-denying breaks FIX D | PASS |
| `stage-prompt-declares-the-boundary` | capability→producer — α adds the boundary paragraph rendered from **one** `ESCALATION_BOUNDARY*` module constant (INV-5), extending the scope wording already at `prompts/stage2.py:647-648` | PASS |
| `mcp-tool-names-honored-in-disallow-lists` | substrate CONFIRMED — `mcp__fused-memory__*` names already in `DISALLOW_*` and production-proven | PASS |

### β — `get_task_escalation_history`

| Capability | Binding | Verdict |
|---|---|---|
| `get_task_escalation_history-tool-exists` | capability→producer — β registers it via `@mcp.tool()` (21 existing instances in `escalation/server.py`) | PASS |
| `archive-inclusive-per-task-primitive` | substrate CONFIRMED — `queue.py:396-430`; proven in-process at `orchestrator/harness.py:11024` and `deterministic_runner.py:1954`, both of which already pass `status=None` | PASS |
| `b6-both-halves-hold-on-disk` | premise verified **empirically** against the live reify store for tasks 5534 and 5557 (see substrate table) — β's signal asserts nothing that is not already true of the underlying data | PASS |
| `pending-only-semantics-unchanged` | non-regression — `test_queue.py:376` and `:386` are spec-locked; β wraps, never widens, the `status='pending'` fast path | PASS |

### γ1 — thread `StoreIdentity` into `create_server` *(intermediate)*

| Capability | Binding | Verdict |
|---|---|---|
| `StoreIdentity-declared` | capability→producer — γ1 introduces the frozen dataclass (PRD §6.1) | PASS |
| `both-production-call-sites-wired` | capability→producer (**wired**, not merely declared) — exactly two production sites, `orchestrator/harness.py:9097` and `reconciliation/harness.py:1909`; the same-named `create_server` in `orchestrator/mcp/plan_tools.py:962` / `verdict_tools.py:320` are different functions and are out of scope | PASS |
| `config.project_id-populated` | substrate CONFIRMED — `orchestrator/config.py:955`; populated in all 8 per-project configs | PASS |
| `identity-optional-degrades-to-today` | non-regression — `store_identity=None` (tests, standalone) accepts no assertion, renders no identity line, never raises (PRD §6.1); asserted by B3 | PASS |

### γ2 — optional `project_root` assertion + mismatch storm escape

| Capability | Binding | Verdict |
|---|---|---|
| `project_root-assertion-on-read-and-write-tools` | capability→producer, γ1 upstream (wired edge) — the five tools of PRD §6.3 | PASS |
| `recon-endpoint-names-itself-not-a-path-diff` | rejection-mechanism, **producer is self** — γ2 delivers the very rejection its signal asserts, so the binding is satisfied by construction (this is not the false-premise shape). The recon branch must name the reconciliation store rather than emit a path diff, because there is no project path to diff against | PASS |
| `mismatch-streak-escape` | INV-4 — consecutive-streak counter per asserted `project_root`; at threshold (default 5, config-tunable) files **one** `infra_issue` L1 into the server's **own** queue, then re-arms. House pattern `merge_liveness.py:446-501` | PASS |
| `guard-reuse-not-reimplementation` | INV-5 extraction — `_require_matching_project_root` (`server.py:115-137`) is reused and generalized; note it currently reads `harness.git_ops.project_root`, which the harness-less recon server does not have, so the generalization to `StoreIdentity` is load-bearing, not cosmetic | PASS |
| `omitted-assertion-byte-identical` | non-regression (B3) — the fleet-agent path passes no `project_root` and must be unchanged | PASS |

### γ3 — render store identity into every escalation tool description

| Capability | Binding | Verdict |
|---|---|---|
| `fastmcp-description-override` | substrate CONFIRMED **empirically** — fastmcp 3.2.2; `description=` overrides the docstring and is what the model sees | PASS |
| `descriptions-carry-store-identity` | capability→producer, γ1 upstream (wired edge) — the §6.2 block, built at `create_server` time from the identity | PASS |
| `identity-block-single-render-site` | INV-5 — rendered from one helper consumed by all tools, not pasted per tool (PRD §6.2) | PASS |
| `wire-shape-unchanged` | non-regression — `get_pending_escalations` keeps returning `list[dict[str, Any]]`; the response-envelope alternative was rejected in PRD §5.1 precisely because five skill files perform list operations on it (`escalation-watcher-auto/SKILL.md:234-268`) | PASS |

### γ4 — two-way boundary tests for the store-identity seam

| Capability | Binding | Verdict |
|---|---|---|
| `server-side-boundary-tests-b1-b4-b7` | capability→producer, γ2 + γ3 upstream (wired edges) — B1/B2/B3/B4/B7 | PASS |
| `consumer-side-boundary-test-b8` | capability→producer, **α upstream** (wired edge) — the recon-stage argv assertion; α owns the mechanism, γ4 owns the test | PASS |
| `b6-history-vs-pending` | capability→producer, **β upstream** (wired edge). DAG-direction PASS — β is a prerequisite of γ4, never a dependent. Both halves already verified on disk (substrate table) | PASS |
| `b5-agent-side-attribution` | capability→producer, γ3 upstream (wired edge) — an agent on 8103 receives `[]` **and** a description naming the store, sufficient to attribute the emptiness | PASS |

## Notes for the dispatch-time architect

- **Do not touch** pump-web-ui `esc-18-1`, the reify gate tasks
  (5534/5537/5547/5549/5550/5552/5557), or reify task 5597. This batch removes the
  mechanism that produced those false premises; it does not adjudicate them.
- **No in-process consumer changes.** All 71 in-process read sites are out of scope
  (PRD §2.6, §10) — they resolve `queue_dir` against their own `project_root` and
  the two substantive absence-inferring consumers already fail safe. A task that
  starts editing `harness.py` read sites has drifted from the PRD.
- Pre-existing INV-5 instance explicitly out of scope: `_queue_for` duplicated
  verbatim across three fused-memory middleware modules (PRD §9, §10).
