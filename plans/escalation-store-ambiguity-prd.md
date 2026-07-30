# Escalation store ambiguity — "wrong store" vs "genuine absence"

**Status**: active · authored 2026-07-28 · approach **B + H** (contract scoped to γ)
**Root cause verified at** dark-factory HEAD `7658f909fc`

## 1. Goal

`EscalationQueue`'s read API cannot distinguish **"there is genuinely no escalation
for this task"** from **"I am pointed at a store that could never contain it."**
Both return `[]`. Consumers have repeatedly treated `[]` as proof of absence and
acted on it — filing "stranded gate" reports and, in one case, a whole follow-up
task built on a false premise.

When this PRD lands:

- A reconciliation stage agent can no longer ask a question its endpoint cannot
  answer — the escalation read tools are denied to it, so it gets a **denial**
  instead of a misleading `[]`.
- Any MCP caller can ask **"was an escalation ever filed for task X?"** — a
  question that has no answer in today's tool surface.
- Any MCP caller can see, in the tool description itself, **which store the
  endpoint serves**, before it interprets an empty result.

## 2. Background

### 2.1 The primitive

`escalation/src/escalation/queue.py` is a filesystem queue: one JSON file per
escalation in `queue_dir`, resolved records relocated to `archive/YYYY-MM-DD/`.
`escalation/src/escalation/server.py` exposes an MCP server over exactly one such
queue. Eight are live on this host (7 orchestrators on 8100-8108 + the
reconciliation harness on 8103), all named `FastMCP('escalation')`, all exposing
identically-named tools. Nine stores exist on disk.

### 2.2 Four flavors of the ambiguity (all empirically demonstrated)

| # | Flavor | Mechanism |
|---|---|---|
| 1 | Wrong store → empty | `glob` on another project's dir (`queue.py:428`, `:506`) |
| 2 | Resolved reads as never-filed | `status=='pending'` skips `archive/` (`queue.py:429`) |
| 3 | Typo'd path → empty | `__init__` does `mkdir(parents=True)` (`queue.py:152`) — the store is **created**, so no existence check can ever fail |
| 4 | Wrong store → false **positive** | task ids are not namespaced: **53 of 1317 ids collide across ≥2 stores**, so a mis-targeted read can return another project's record |

Flavor 4 is the most dangerous and was not in the original framing. The
`Escalation` model has no project field (`models.py:89-104`), no tool returns
`queue_dir`, and nothing logs the store at startup — so an MCP observer has no way
to tell which store answered.

### 2.3 Verified root cause of the incident chain

A recon stage agent's `escalation` MCP server is wired to the **reconciliation**
queue, not the subject project's:

- `reconciliation/stages/base.py:443-448` — the stage MCP config sets
  `servers['escalation'] = {'type': 'http', 'url': self._escalation_url}`
- `reconciliation/harness.py:1899-1927` — that URL is the reconciliation harness's
  own server, over `config.escalation_queue_dir`
- `fused-memory/config/config.yaml:137` — which is
  `data/reconciliation/escalations`, port 8103

That store contains only `reconciliation-stage1` findings (verified: 8 records).
So `get_pending_escalations(task_id=<any orchestrator task>)` from a recon stage
returns `[]` **categorically, 100% of the time** — not a race, not a timing
window, and *not* the level-cap hypothesis that was invented on esc-5559-1,
propagated into reify task 5597, and cancelled as a verified-false premise on
2026-07-27. Reads are ungated for every connection: `get_pending_escalations`
never calls `get_http_headers()`.

### 2.4 The asymmetry that made it invisible

Recon's **task** reads are correctly project-routed: `known_projects:
dict[project_id, project_root]` is threaded into every stage
(`stages/base.py:69-74`), surfaced to the LLM
(`task_knowledge_sync.py:3564 _format_known_projects_section`), and used to route
`get_task(task_id, root)` (`flag_dedup.py:2411-2463`). Recon's **escalation**
reads are pinned to one hard-wired store. The agent sees a table of N projects,
gets correctly-routed task reads, and gets silently mis-routed escalation reads.

### 2.5 Incident history — three projects, still recurring

- **reify**: esc-5559-1 (2026-07-26) reported 7 stranded gates
  (5534/5537/5547/5549/5550/5552/5557), stating
  `get_pending_escalations(task_id=X)` returned `[]` for every one. The archived
  records for 5534 and 5557 show filing timestamps matching their
  `gate_escalated_at` stamps exactly — they were live and pending at the moment
  they read as absent. Precedents: tasks 5330, 5322.
- **pump-web-ui**: `esc-18-1` — **pending L2 right now**, filed 2026-07-23,
  summary *"Investigate escalation-filing/persistence pipeline gap (pending queue
  empty despite filed escalations)"*, citing 4 recon cycles. Its cited records
  (`esc-7-2` pending L1, `esc-7-3` pending L2, `esc-17-1` pending L2) are all
  present on disk; `esc-17-1`'s filing timestamp matches task 17's
  `gate_escalated_at` to 2 ms. The premise is false; a human is queued on it.
- **interactive sessions**: `skills/recon-escalation-watcher/SKILL.md:295-297`
  already carries a troubleshooting note — *"`get_pending_escalations` errors /
  empty when you expect items → confirm this session's `escalation` server points
  at 8103, not 8102."* A human hit this and the shipped mitigation was **prose in
  a skill file**. That is INV-1's failure shape exactly.

### 2.6 What is NOT broken

The 71 in-process read sites across 18 modules resolve `queue_dir` against their
own `project_root` (`harness.py:9065-9068`, `config.py:984`), so flavors 1/3/4
cannot reach them absent a genuine config error. Their flavor-2 exposure is
narrow and mostly benign (dedup → a duplicate; resolve-scan → a no-op). The two
consumers that infer absence substantively already use the right discriminator:
`harness.py:11024` and `deterministic_runner.py:1954` both pass `status=None`
(archive-inclusive) with an `agent_role` scope, and both fail **safe** on `[]`
(re-file the gate rather than phantom-complete).

**This PRD therefore changes no in-process consumer.** The defect lives entirely
at the MCP boundary handed to LLM agents.

## 3. Consumer + user-observable surface (G1)

| Mechanism | Consumer |
|---|---|
| α — escalation read tools denied to recon stages | The reconciliation stage runner (`cli_stage_runner.py` → `--disallowed-tools`); operator inspecting a live stage spawn |
| β — `get_task_escalation_history` MCP tool | `/unblock`, `escalation-watcher`, `escalation-watcher-auto` skills; any agent asking whether a gate was ever filed |
| γ — store identity + scoping guard | Every MCP caller of every escalation server; `skills/recon-escalation-watcher` (its §295 troubleshooting note becomes obsolete) |

No mechanism here is a producer without a live consumer.

## 4. Sketch of approach

**α — deny, don't route.** Recon stages have a *sanctioned write* need
(`escalate_blocker` for the FIX D stale-flag case, `prompts/stage2.py:623-648`)
and **no sanctioned read need at all** — `get_pending_escalations` appears in zero
recon prompts. The stage agents were reading on their own initiative because
Stage 2 runs a denylist (`STAGE2_DISALLOWED = DISALLOW_BUILTIN`,
`cli_stage_runner.py:78`) that never covered `mcp__escalation__*`. Deny the two
read tools and state the boundary in the stage prompt.

*Road not taken:* per-project escalation routing for recon, mirroring
`known_projects` + the dashboard's `_discover_escalation_urls`
(`dashboard/config.py:92`). Rejected — it would mean N escalation connections per
stage to serve a need no prompt sanctions. Recorded here because if recon ever
*does* need escalation reads, that is the path.

**β — expose the existence question.** `get_escalation(id)` is the only
archive-inclusive MCP read and it requires already knowing the id;
`get_pending_escalations(task_id=…)` is pending-only by construction. So the MCP
surface cannot express *"was an escalation ever filed for task X?"* — the exact
question all three incidents were asking. Add a thin wrapper over
`queue.get_by_task(task_id)` with `status=None`, which is already the proven
in-process discriminator at two call sites.

**γ — declare the store where callers see it.** Plumb an explicit store identity
into `create_server`, render it into the tool **descriptions** at construction,
and accept an optional `project_root` assertion that errors on mismatch.

## 5. Resolved design decisions

1. **Identity is declared in the tool description, not the response envelope.**
   Changing `get_pending_escalations` from `list[dict]` to a dict envelope would
   lock-step five skill files whose prose performs list operations
   (`escalation-watcher-auto/SKILL.md:234-268` filters and set-differences the
   list directly) — an INV-5 violation and a G5 integration-starvation risk.
   `@mcp.tool(description=…)` is supported in fastmcp 3.2.2 and **verified
   empirically** to override the docstring and be what the model sees. The
   description is built at `create_server` time from the store identity, so it
   costs no wire change and is visible before the caller ever interprets an `[]`.
   β is a *new* tool with no legacy callers, so it returns a full envelope.

2. **`project_root` is optional, never required.** Fleet agents get exactly one
   escalation endpoint (`mcp_config_json(escalation_url=…)`,
   `mcp_lifecycle.py:1014-1059`, singular), so they cannot mis-target and a
   required param would be pure migration cost. Optional-and-validated binds the
   population actually at risk — multi-project interactive callers — which is
   verbatim the rationale already recorded for `_require_matching_project_root`
   (`server.py:124-127`).

3. **The guard covers write tools as well as reads.** It is never correct for an
   MCP caller to file into another project's queue. Cross-project filing is real
   but exclusively **in-process** — fused-memory is one shared server (8002) and
   its middleware routes by `project_root` through a per-project queue cache
   (`ticket_janitor.py:200`, `curator_escalator.py:270`,
   `scope_violation_escalator.py:138`, `candidate_key_escalation.py:91`,
   `reconciliation/targeted.py:1358`). None of that goes through MCP, so none of
   it is affected.

4. **The reconciliation server asserts a non-project identity.** It is built
   harness-less (`reconciliation/harness.py:1909`), so there is no
   `harness.git_ops.project_root` to compare against. It declares
   `kind='reconciliation'`; any `project_root` assertion against it fails with a
   message naming what it actually is, rather than a path mismatch.

5. **No in-process consumer changes.** See §2.6.

### Amendment (2026-07-30) — §5.1's β name and envelope, ratified

Recorded because two remediation paths for the `get_task_escalations` /
`get_task_escalation_history` collision were adjudicated independently and
neither cross-referenced the other. **Decision: option (b) — β keeps its own
name and its envelope.** Ratified by the PRD owner 2026-07-30 against
escalation `esc-3230-1` (gate task 3230).

**The collision.** Task 3023, planned 2026-07-24 — four days before this PRD
existed — landed `get_task_escalations(task_id, status=None, level=None,
agent_role=None, compact=False) -> list[dict]` in `escalation/server.py`: a
1:1 delegation to `queue.get_by_task(...)` over the same primitive β targets.
Neither side was at fault; the two plans could not have seen each other.

**The losing path, recorded so it is not re-derived.** Task 3023's steward
adjudicated the collision in `esc-3023-6` (resolved, archived 2026-07-29;
Graphiti episode `8296f70b-1908-49eb-9f0f-03eeca44d0e9`) and recommended
option (a): keep the landed `get_task_escalations`, re-scope β to adopt it,
ship no second tool. Their argument, in their order of weight:

1. **Name semantics.** `..._history` connotes past/archived, but the tool
   returns pending AND archived. In a subsystem whose documented failure mode
   is agents trusting a tool name's implied semantics, a name that
   under-promises its own scope is the wrong choice.
2. **The landed tool exceeds β's spec** — it already answers open question 2
   (`level`: yes), adds `agent_role`, adds a `compact` projection deliberately
   shape-compatible with `get_pending_escalations`, and carries the
   evidence-of-absence contract in its docstring. β is therefore a strict
   *restriction* of it, plus an envelope.
3. Blast radius does **not** decide it — they measured ~13 references to β's
   name against ~8 to the landed name, and called them comparable.

They deliberately did not execute it: option (a) renames a tool this PRD names
in §6.2 and overturns a §5 resolved decision, which needs the owner's assent
rather than a steward's unilateral edit from an adjacent task.

**Why (b) was affirmed.**

1. §5.1's closing sentence and §6.2's rendered block are decisions of record,
   and §6.2 puts the literal string `get_task_escalation_history` into every
   escalation tool's description via γ3 — so the name is a **contract**, not an
   incidental label. (a) requires amending both; (b) requires amending neither.
2. **The steward's decisive argument is one this PRD already answers.** §5.1's
   doctrine is that a tool's true nature is declared in its DESCRIPTION, not
   inferred from its shape — and §6.2 applies exactly that to a *semantic*
   (pending-vs-ever-filed) confusion. β's shipped docstring opens "Every
   escalation ever filed for a task" and "ARCHIVE-INCLUSIVE … Returns pending
   AND resolved/dismissed records". A name that under-promises, corrected by an
   explicit description, is this PRD's own prescribed remedy.
3. **The duplication (a) feared did not materialise.** β as shipped is an
   eight-line delegation to `get_task_escalations` plus a docstring that
   contrasts the two explicitly: `get_task_escalations` is the general
   FILTERABLE form; β is the un-narrowable one. β deliberately has **no**
   `status` parameter, so a caller cannot silently recreate the false-absence
   trap β exists to remove.

Conceded to the steward: two tools where one is a strict subset of the other is
a smell, and the name concern is real. If the name is observed to mislead in
practice, (a) remains a clean rename against landed code — the arguments above
survive that reversal, so re-read this note rather than re-deriving it.

**Standing tax of the second tool, discovered while implementing β.**
`create_server` backs both the orchestrator and the reconciliation queues, and
recon-stage gating is a DENY list only (`DISALLOW_ESCALATION_READS`,
`reconciliation/cli_stage_runner.py`) — there is no allow-list. So **every**
tool accession on that server must be classified there in the same change, or
it is reachable from every recon stage pointed at the wrong store. β is
strictly worse than a bare list if left unclassified: its envelope echoes
`task_id` and `level_filter` back, so a `count: 0` against the wrong store
reads as an *attributable* answer rather than the artefact it is. β denies
itself to all three stages.

### Amendment (2026-07-30) — the deferred-INV-5 concern, closed

The steward observed that §5.1's own reasoning appears to boomerang: it refused
an envelope for `get_pending_escalations` because five skill files perform list
operations on the result, and §6.2 then directs those same skills at the
archive-inclusive tool — so they would inherit the lock-step cost §5.1 avoided,
merely deferred. That does not hold, for three reasons worth pinning down:

- **Nothing forces a migration.** `get_pending_escalations` keeps
  `list[dict[str, Any]]` unchanged, so no existing skill prose breaks. The five
  files are not edited in lock-step with anything.
- **It is a different question at a different call site.** "What is open now?"
  and "was one ever filed?" are separate reads; a skill adopts the second
  additively, when it has that question, independently of the others.
- **The adapter is one subscript.** β returns the delegate's list unmodified
  under `escalations`, so `result['escalations']` is byte-identical to what a
  bare-list tool would have returned. Adoption costs a key lookup, not a shape
  migration.

**Correction to §5.1's framing while we are here.** "Five skill files" reads as
a lock-step block; measured, it is five *independent* consumers of one return
type, each calling the tool differently — `escalation-watcher`
(`level=2, compact=True`, L2-only drain), `escalation-watcher-auto` (two
unfiltered calls, then a level-1/level-2 set-difference), `recon-escalation-watcher`
(no `level` arg — the recon queue is flat — against **8103**), `merge-queue`
(a liveness probe that never reads the contents), and `unblock`
(`task_id=`-scoped). There is no shared passage to hoist into a common
reference; the differing arguments, level filters and *stores* are the whole
content of each usage. The genuinely duplicated thing across them is **store
identity** — restated in `escalation-watcher/SKILL.md:22,995` and throughout
`recon-escalation-watcher/SKILL.md` — and γ3 dedups exactly that, at a strictly
better seam: rendered once from `StoreIdentity`, delivered in the description at
the point of use, reaching interactive and fleet callers the skills do not
cover, and unable to drift because it derives from the live `queue_dir`. The
skills-side follow-up is therefore a **deletion** sweep once γ3 lands (§10
already names `recon-escalation-watcher/SKILL.md:295-297`), not a new include
layer.

INV-5 would bite if a future change made the envelope non-additive — if β
reshaped its ELEMENTS, or if `get_pending_escalations` were later given an
envelope so both had to move together. Neither is in this PRD's scope; a task
proposing either must re-open this note.

## 6. Contract (γ) — B + H

### 6.1 Store identity

```python
@dataclass(frozen=True)
class StoreIdentity:
    kind: Literal['project', 'reconciliation']
    queue_dir: Path          # absolute, resolved
    project_id: str | None   # e.g. 'reify'; None when kind != 'project'
    project_root: Path | None
```

`create_server(queue, ..., store_identity: StoreIdentity | None = None)`
(`escalation/server.py:231`). Exactly two production call sites:

| Caller | Passes |
|---|---|
| `orchestrator/harness.py:9097` | `kind='project'`, `project_id=cfg.project_id` (`config.py:955`), `project_root=cfg.project_root`, `queue_dir=queue.queue_dir` |
| `reconciliation/harness.py:1909` | `kind='reconciliation'`, `project_id=None`, `project_root=None`, `queue_dir=queue.queue_dir` |

`None` (tests, standalone) degrades to today's behaviour: no assertion accepted,
no identity line in descriptions. Never raises.

### 6.2 Description rendering

Every escalation tool's description gains a trailing block, rendered once at
construction:

```
THIS ENDPOINT SERVES: <queue_dir> (kind=<kind>, project=<project_id or n/a>).
It contains ONLY escalations for that store. A task id belonging to any other
project returns an empty result because the record is not HERE — not because it
does not exist. To ask whether an escalation was EVER filed for a task
(including resolved/archived), use get_task_escalation_history.
```

Rendered from **one** helper consumed by all tools (INV-5: one site plus a call).

The tool name in that last sentence is a **contract**, not a label: it is the
reason β's name could not be changed unilaterally when task 3023 landed a
sibling tool over the same primitive. That collision is adjudicated in §5's
"Amendment (2026-07-30) — §5.1's β name and envelope, ratified"; read it before
proposing a rename.

### 6.3 Guard semantics

`project_root: str | None = None` added to `get_pending_escalations`,
`get_escalation`, `get_task_escalation_history`, `escalate_info`,
`escalate_blocker`.

| Condition | Result |
|---|---|
| arg omitted | proceed, unchanged (fleet-agent path) |
| `kind=='project'` and resolved paths equal | proceed |
| `kind=='project'` and paths differ | `{'error': 'project_root mismatch: … does not match this server's wired project (…)'}` — reuse `_require_matching_project_root` |
| `kind=='reconciliation'` | `{'error': "this endpoint serves the reconciliation store …, not a project task queue; it never contains orchestrator task escalations"}` |
| `store_identity is None` | proceed (standalone/test) |

### 6.4 Storm escape (INV-4)

Mismatch rejections are a rejection path, so they carry a counter, not silence.
Per-`(asserted_project_root)` consecutive-mismatch streak; at threshold
(default 5, config-tunable) the server files **one** `infra_issue` L1 into its
**own** queue naming both stores and the asserted value, then re-arms. Follows
the consecutive-streak house pattern (`merge_liveness.py`, generalized by 2558).

### 6.5 Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Mismatched assertion is loud | project server for reify; caller asserts `project_root=/home/leo/src/dark-factory` | error naming both roots; **not** `[]` |
| B2 | Recon server names itself | recon server, `kind='reconciliation'`; any `project_root` asserted | error identifies the reconciliation store; no path-diff message |
| B3 | Omitted assertion unchanged | any server; no `project_root` arg | byte-identical to pre-change result |
| B4 | Description carries identity | recon server built with identity | `list_tools` description contains the queue_dir and `kind=reconciliation` |
| B5 | Agent-side: empty is attributable | agent on 8103 calls `get_pending_escalations(task_id='5534')` | receives `[]` **and** a description naming the store, sufficient to attribute the emptiness |
| B6 | History answers what pending cannot | reify server; task 5534 resolved+archived | `get_pending_escalations(task_id='5534')` → `[]`; `get_task_escalation_history('5534')` → `esc-5534-1`, `status='resolved'` |
| B7 | Storm escape fires | 5 consecutive mismatches, same asserted root | exactly one `infra_issue` L1 in the server's own queue; 6th does not double-file |
| B8 | Recon stage cannot obtain an escalation read | recon stage spawn | `--disallowed-tools` argv contains both read tools; the stage cannot obtain an escalation-read **result** — whether the CLI rejects the call or omits the tool from the listing, no `[]` reaches the agent |

B5, B6 and B8 face the **consumer** side; B1-B4 and B7 face the **server** side.

## 7. Decomposition plan

| Label | Title | Modules | Kind | Observable signal | Deps |
|---|---|---|---|---|---|
| **α** | Deny escalation read tools to reconciliation stage agents | `fused-memory/reconciliation` | leaf | An operator inspecting a live recon stage spawn sees both read tools in the process's `--disallowed-tools` argv and the escalation-boundary paragraph in its `--system-prompt-file`; the stage cannot obtain an escalation-read result, so no misleading `[]` reaches the agent (**B8**) | — |
| **β** | Add `get_task_escalation_history` — archive-inclusive per-task escalation read | `escalation` | leaf | Against reify's server, `get_task_escalation_history('5534')` returns `esc-5534-1` with `status='resolved'` while `get_pending_escalations(task_id='5534')` returns `[]` (**B6**) — both halves already verified on disk | — |
| **γ1** | Thread `StoreIdentity` into `create_server` and both call sites | `escalation`, `orchestrator`, `fused-memory/reconciliation` | intermediate | Unlocks γ2 + γ3 | — |
| **γ2** | Optional `project_root` assertion on escalation read + write tools, with mismatch storm escape | `escalation` | leaf | A caller asserting a mismatched `project_root` gets an error naming both stores instead of `[]`; the recon server names itself as non-project (**B1, B2, B3**); 5 consecutive mismatches file exactly one `infra_issue` L1 (**B7**) | γ1 |
| **γ3** | Render store identity into every escalation tool description | `escalation` | leaf | An agent listing tools on 8103 sees the queue_dir and `kind=reconciliation` in the description of every escalation tool (**B4, B5**) | γ1 |
| **γ4** | Two-way boundary tests for the store-identity seam | `escalation`, `fused-memory/reconciliation` | leaf (integration gate) | All of B1-B8 green in CI | α, β, γ2, γ3 |

Ordering is α → β → γ1 → {γ2, γ3} → γ4. α ships first and alone retires the live
incident class.

## 8. Cross-PRD relationship (G4)

**No cross-PRD seams.** Every dependency is intra-batch. This PRD touches
`escalation/`, `orchestrator/harness.py` (one call site), and
`fused-memory/reconciliation/` but introduces no mechanism another PRD owns or
consumes.

## 9. Design-invariant walk (G7)

| Invariant | Verdict |
|---|---|
| INV-1 `contracts-machine-checked` | **This PRD is the fix.** The endpoint's store envelope is today discovered by failure — literally documented as a troubleshooting note (`recon-escalation-watcher/SKILL.md:295`). γ3 declares it where callers see it. |
| INV-2 `structured-facts-at-failure` | Addressed. `[]` is a fact the emitter knew (its own `queue_dir`) but did not emit, forcing consumers to re-derive a story. γ3 emits it. |
| INV-3 `corroborate-before-acting` | Addressed by β — absence-inferring consumers get an archive-inclusive corroboration read instead of overloading a work-queue read. |
| INV-4 `storm-escape-required` | **Hit, resolved in design.** γ2's mismatch rejection carries a consecutive-streak counter → one `infra_issue` L1 (§6.4). α is a hard denial, not a fail-soft, so it is out of INV-4's scope — but it ships a prompt paragraph so the agent is told the boundary rather than silently degrading. |
| INV-5 `no-lockstep-duplication` | **Hit, resolved in design.** Drove decision 5.1: the response-envelope option would have lock-stepped five skill files; the description approach does not. The identity block is rendered from one helper (§6.2). *Pre-existing, out of scope:* `_queue_for` is duplicated verbatim across three middleware modules. |

No waivers required.

## 10. Out of scope

- Any change to the 71 in-process read sites (§2.6).
- Deduplicating the `_queue_for` middleware triplet (pre-existing INV-5).
- Namespacing task ids across projects — the real cure for flavor 4, but a
  far larger change; the guard plus the description mitigate it.
- Retiring `skills/recon-escalation-watcher/SKILL.md:295-297` once γ3 makes it
  obsolete — a docs follow-up, not load-bearing. **Owned by task 3266**, which
  depends on γ3 and also carries the positive half (point the reader at the
  rendered `THIS ENDPOINT SERVES` block rather than deleting the warning and
  leaving nothing). Measured 2026-07-30: this is the *only* such bullet across
  the five escalation-consuming skills — 3266 pins the two look-alikes that
  must NOT be swept with it.
- Adjudicating pump-web-ui `esc-18-1` or the reify gate tasks. They belong to
  other owners; this PRD only removes the mechanism that produced them.

## 11. Open questions (tactical)

1. **Streak threshold for §6.4.** Default 5 proposed. Decide during γ2; any value
   in 3-10 is defensible and it is config-tunable.
2. **Does `get_task_escalation_history` take a `level` filter?** Symmetry with
   `get_pending_escalations` suggests yes; no consumer needs it today. Decide
   during β — additive either way.
3. **Whether γ3's identity block also lands on the merge/halt tools** served by
   the same server, or only the escalation-semantic ones. Decide during γ3.

   *Widened during α (task 3163): decide the whole degraded-read surface, not only
   the merge/halt identity block.* α denies just the two escalation READ tools
   (`DISALLOW_ESCALATION_READS`) — the ones the three incidents actually misread.
   But the recon-wired server is built by
   `reconciliation/harness.py::_start_escalation_server` (`:2011-2016`) as
   `create_escalation_server(self._escalation_queue)` — queue positional only, so
   `harness`, `merge_queue`, `event_store`, `orch_config` and
   `merge_inflight_registry` all stay `None`. Measured standalone behaviour of the
   still-reachable non-escalation reads:

   | tool | standalone return | shape |
   |---|---|---|
   | `get_task_runtime_state` | `{'offline': False, 'tasks': []}` — `escalation/server.py:1674-1675` returns `TaskRuntimeSnapshot()`, whose defaults are `offline=False`, `tasks=[]` | **silent false absence** — positively asserts "online, nothing running" |
   | `get_merge_queue` | `{'error': 'Merge queue not available — orchestrator not running'}` (`server.py:1693-1694`) | loud, self-describing |
   | `get_merge_halt_status` | `{'wired': False, 'error': 'escalation server running standalone'}` (`server.py:1648-1649`) | loud, self-describing |

   The merge/halt pair therefore already fails loudly and needs nothing beyond the
   identity block. `get_task_runtime_state` is the outlier: it reproduces the exact
   categorical-empty failure mode this PRD exists to close, since a stage reading
   `tasks: []` as "no task is running" is the same inference the three incidents
   made from `[]` — and it is worse than `[]`, because `offline: False` asserts the
   snapshot is live. That stages reach for this server unprompted is not
   hypothetical: OQ4 below records a live Stage-2 transcript calling
   `mcp__escalation__merge_status` on its own initiative. So γ3 should decide
   `get_task_runtime_state` explicitly — either give it the same
   never-silently-empty treatment as the escalation reads (deny it to stages, or
   make the standalone envelope declare itself unwired the way
   `get_merge_halt_status` does), or record why an unconditional empty-but-live
   snapshot is safe here. α covers this case verbally only, via the
   `ESCALATION_BOUNDARY_NOTE` "serves the RECONCILIATION store only" sentence;
   nothing mechanical does.
4. **Does `--disallowed-tools` reject a denied MCP tool on call, or omit it from
   the listing?** **RESOLVED during α (task 3163): OMISSION.** A denied tool is
   simply absent from the agent's visible tool set; there is no rejection event
   and no error message.

   *Method (reproducible).* Recon stages run in deferred-tools mode, so a stage
   transcript carries the agent's visible tool set in its
   `attachment.type == 'deferred_tools_delta'` → `addedNames` payload. Across
   five live recon-stage transcripts under
   `~/.claude/projects/-home-leo-src-dark-factory/`, the denied names are absent
   from that listing and track each stage's disallow list exactly:

   | transcript | stage | tools listed | absent from listing |
   |---|---|---|---|
   | `e8e4fe99`, `aaeeb70c` | 3 | 80 | builtins, `…__delete_entity`, `…__submit_task`, `…__add_memory`, `…__write_entity_standing_decision` |
   | `e22890a3` | 1 | 93 | builtins, `…__submit_task` (memory writes present — matches STAGE1) |
   | `ec23af49`, `43f8155c` | 2 | 100 | builtins, `…__write_entity_standing_decision` only |

   *Consequence.* Because it is omission rather than rejection, the agent gets no
   explanation for the missing tool — it would find the escalation surface simply
   gone and could conclude it does not exist, or route around it. This is exactly
   the branch anticipated above: α's prompt paragraph is therefore **load-bearing
   and ships** (`ESCALATION_BOUNDARY_NOTE`, rendered into all three stage
   prompts). Per G6 branch 4, α's user-observable signal and boundary test B8 are
   NOT reworded into a rejection claim — they were written to hold under either
   outcome, and this verdict is recorded as an annotation, not a rewrite.

   *Two corroborating observations.* Both escalation read tools were PRESENT in
   all five listings pre-change — direct evidence of the live bug α closes. And
   one Stage-2 transcript (`ec23af49`) called `mcp__escalation__merge_status`
   unprompted, evidencing that stages do reach for escalation-server tools on
   their own initiative rather than only when instructed.
