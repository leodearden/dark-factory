# Tool-call markup containment — blanket guard, deterministic repair, retro-sweep

**Status:** active · 2026-08-05 · approach **B + H** (contracts + two-way boundary tests)

**Successor to:** DF **3083** (done, merged `7899eef17b` — root cause, `scan_memory_content`, Mem0 corpus sweep) and DF **3141** (done, merged `da94baf14a` — the write-time tripwire at four fused-memory boundaries). Both succeeded at what they scoped. This PRD owns what neither did: the **blast radius outside fused-memory**, the **deterministic repair** of the corruption, and the **retro-sweep of the stores that were never guarded**.

**Excluded by operator decision (2026-08-05):** model routing changes. The corruption is ~17× more frequent on `opus` than `sonnet`, but re-routing architect-class roles is not part of this PRD.

---

## 1. Goal (G1 consumer + user-observable surface)

> **Every MCP tool call carrying leaked tool-call envelope markup is detected at a single blanket guard, deterministically repaired against the tool's own schema, and either bounced back to the caller with the repaired call in hand or forwarded with a loud structured fact — and the records already corrupted are repaired in place, atomically.**

Named consumers, one per mechanism (G1):

| Mechanism | Consumer |
|---|---|
| `shared.toolcall_markup` (detector + repairer + schema validator) | `MarkupGuardMiddleware` (β), the retro-sweep (δ), plan-tools lazy write-back (ε) |
| `MarkupGuardMiddleware` | the four FastMCP servers registered in γ |
| repair-and-reject error payload | the calling agent — it re-issues the repaired call |
| repair-and-forward + `markup_repaired` fact | the operator (journal/structured fact) and the filed escalation/verdict record |
| residue escalation | the L2 escalation-watcher (existing consumer, `escalation/data/escalations`) |
| retro-sweep script | the operator, run once against terminal state |
| plan-tools lazy write-back | architects and reviewers reading `plan.json` |

User-observable surface: a `submit_task` whose description carries a mis-closed `</description>` is rejected with an error **containing the fully repaired call including the recovered `priority`**, instead of today's bare "strip the fragment and resubmit"; an `escalate_info` with the same defect **lands with its `suggested_action` intact** instead of being lost; and `add_design_decision` stops writing corrupted rationales into `plan.json`.

---

## 2. Background — evidence (why this PRD exists)

Root-caused 2026-08-05 from esc-markup-tripwire-3. Measurements are reproducible from the archived agent transcripts (`data/orchestrator/agent-transcripts/` — ~4,400 `.jsonl.gz` across ~609 task dirs, 468 MB, as of 2026-08-05; the tree grows live, so re-running against a later snapshot will yield slightly larger counts).

Collection predicate for the corpus, stated exactly so the numbers below are reproducible: a `tool_use` record's string-valued parameter matching `\x3c/invoke>\s*$` **or** `\x3c/[A-Za-z_]\w*>\s*\x3cparameter\s+name="[^"]+">`.

### 2.1 The mechanism — one defect, not two

The model emits a parameter's **closing** tag in the wrong dialect — `</description>`, `</content>`, `</rationale>` (echoing the parameter *name*) instead of `</parameter>` — and frequently continues the remaining parameters in that same wrong dialect. The harness parser does not find its expected closer, **over-consumes** to the next available terminator (a later well-formed `</parameter>`, else `</invoke>`), dumps everything into the current parameter's value, and **silently drops every parameter in between**.

Four specimens with the parsed input visible:

| Call | Value tail (as parsed) | Keys received | Dropped |
|---|---|---|---|
| `submit_task` 07-30T16:47Z | `…direction.</description>\n<priority>medium</priority>\n<agent_id>…</agent_id>\n<metadata">{…}</metadata">\n</invoke>` | `project_root, title, description` | **priority, agent_id, metadata** |
| `submit_task` 08-04T12:13Z | `…automatically).</description>\n<parameter name="priority">low` | no `priority` | **priority** (intended `low`) |
| `add_memory` 08-04T16:58Z | `…by design.</content>\n</invoke>` | all | none (`content` was last) |
| `update_memory` 08-02T21:26Z | `…scoped).</parameter>\n<parameter name="agent_id">escalation-watcher-l2` | no `agent_id` | **agent_id** — boundary is **ungated**, so it landed |

Specimen 1's `<metadata">` is a literal dialect blend — the model interpolating between `<parameter name="X">` and `<X>`.

### 2.2 The "second mechanism" was a pattern-coverage artifact

`MCP_MARKUP_PATTERNS = ('</content>', '<parameter name=', '</invoke>')` and `find_markup_pattern` reports the **earliest-position** match. A mis-closed `content` reports its own tag; a mis-closed `description`/`details` cannot, because `</description>` is not in the list — so the guard reports whatever follows: `<parameter name=` on partial drift, `</invoke>` on total drift. `PREFILTER_NEEDLES` in `fused_memory/utils/toolcall_xml_leak.py` *does* list all four closers while `MCP_MARKUP_PATTERNS` lists one; that divergence is what made the diagnostic ambiguous. **INV-5 hit — this PRD consolidates them.**

### 2.3 Blast radius — fused-memory is 29% of it

334 corrupted calls / 128,066 tool calls = **0.26%**. The guard covers only the fused-memory column:

| Server | Corrupted calls | Guarded today |
|---|---|---|
| orchestrator/plan-tools | 198 (59.3%) | **no** |
| fused-memory | 96 (28.7%) | 4 of 6 write tools |
| orchestrator/verdict-tools | 19 (5.7%) | **no** |
| escalation | 18 (5.4%) | **no** |
| Claude Code builtins (`Agent`, `Edit`) | 3 (0.9%) | out of scope — not our servers |

Top victims: `add_design_decision.rationale` (109), `add_memory.content` (90), `add_design_decision.decision` (33), `add_reuse_item.how` (33), `submit_review_verdict.summary` (19), `escalate_info.detail` (17). Within fused-memory, `update_memory` and `add_system_record` are write tools with **no gate at all**.

### 2.4 Landed damage at the ungated boundaries

Re-measured 2026-08-05 **split by sweep lane**, because §5 D4 routes the two lanes to different tasks. The first pass reported a combined "21 of 296 plan.json / 87 strings", which was right in total but wrong in attribution — nearly all of it is live, not terminal:

| Lane | Owner | Files with corruption | Corrupted strings |
|---|---|---|---|
| `.worktrees-orphaned/**/.task/plan.json` (terminal) | **δ** (eager sweep) | 1 of 15 | 2 |
| `.worktrees/**/.task/plan.json` (live) | **ε** (lazy write-back) | 22 of 270 | 91 |
| `data/escalations/**` | **δ** (eager sweep) | 52 of 3,386 | 53 |

Consequences for the decomposition: δ's *plan* work is nearly a no-op — its real payload is the escalation corpus — while **ε carries essentially all the plan.json damage**. This strengthens D4 rather than undermining it: the corruption sits precisely where a global rewrite would have had to contend with running tasks. Escalation counts drift upward as escalations are written live (51 → 52 across a few hours); treat ~50 as the figure and re-measure at sweep time.

Corrupted strings include cases where `rationale` was absorbed into `decision`, so the design rationale a future architect reads is another field's text. Retained plans are a survivor sample; most worktrees are deleted post-merge, so the historical total is larger.

**Sweep hazard pinned for δ:** `docs/task-recovery-2026-05-13/worktree-inventory.json` is git-tracked, legitimately contains predicate matches, and is replicated into every worktree — a loose glob hits it ~47 times. A sloppy sweep would rewrite committed evidence.

### 2.5 Containment *where installed* works — which is why the gap is the story

Exhaustive scan: 21,064 Mem0 records, 41 legacy specimens, newest `2026-07-30T05:07:34Z` — **21 minutes before 3141 merged** (`da94baf14a`, 07-30 05:28:07Z). Zero new specimens in six days. The tripwire is not the problem; its *coverage* is.

### 2.6 The corruption is deterministically repairable

Replaying all 334 specimens through a schema-validated repairer: **308 repair cleanly (92.2%), recovering 194 dropped parameters**; 26 (7.8%) are ambiguous (doubly-corrupted calls) and must escalate. Recovered parameters are exactly the silently-lost ones: `add_memory.category` ×70, `add_memory.project_id` ×32, `add_design_decision.rationale` ×25, `add_memory.agent_id` ×18, `escalate_info.suggested_action` ×13, `submit_review_verdict.issues` ×10, `submit_task.priority` ×5.

This measurement is the reason repair is deterministic rather than LLM-mediated (§5 D2).

---

## 3. Sketch of approach

One module in `shared/`, one middleware, four registrations, two sweeps.

```
shared/src/shared/toolcall_markup.py
  ├─ detect(value)            -> MarkupHit | None       (single pattern source, INV-5)
  ├─ repair(value, param, schema_params, supplied)
  │                           -> Repair(clean_value, recovered:{name:value}) | Unrepairable
  └─ FIXTURES: 334 real specimens committed as the regression pin

shared/src/shared/mcp_markup_middleware.py
  MarkupGuardMiddleware(policy: RepairPolicy, exempt_tools: frozenset)
    on_call_tool:  detect -> repair -> validate -> {reject-with-repair | forward-repair}
                   -> structured fact + storm counter (INV-2, INV-4)

registered on:  fused-memory tools.py:1038 · plan_tools.py:596
                verdict_tools.py:172 · escalation/server.py:307
```

Repair is applied only when it **validates**: every recovered name is a real parameter of *that* tool (read from `context.fastmcp_context.fastmcp.get_tool(name).parameters`), no recovered name collides with an already-supplied argument, and the absorbed tail parses with zero leftover. Validation failure ⇒ never guess (§4 C2).

---

## 4. Contracts (H)

### C1 — Detection and repair contract (`shared.toolcall_markup`)

**Single source of truth.** This module owns the envelope-literal enumeration. `fused_memory.server.markup_tripwire.MCP_MARKUP_PATTERNS` and `fused_memory.utils.toolcall_xml_leak.PREFILTER_NEEDLES` are re-exports of it or are deleted; no third site enumerates the literals (INV-5). The write-time/read-time calibration split documented in those modules is preserved as two *named predicates over one literal set*, not two literal sets.

```python
class Repair(NamedTuple):
    clean_value: str              # the caller's intended text, tail removed
    recovered: dict[str, str]     # dropped params, name -> value
    pattern: str                  # the matched envelope literal
    misclose: str                 # the wrong closing tag, e.g. '</description>'

def detect(value: object) -> str | None: ...
def repair(value: str, param: str, schema_params: Collection[str],
           supplied: Collection[str]) -> Repair | None: ...
```

**Repair algorithm (normative).** Scan candidate mis-close positions left-to-right. A candidate `</X>` qualifies iff `X == param` or `X ∈ schema_params`. For each candidate, parse the remaining tail as a sequence of pseudo-parameters (`<name>value</name>`, `<parameter name="name">value</parameter>`, or a final **unterminated** `<name>value` running to end-of-string — the parser consumed that closer as its terminator), stripping a trailing `</invoke>`. Accept the **earliest** candidate for which: the tail parses with zero leftover text, every recovered name ∈ `schema_params`, and no recovered name ∈ `supplied`. Otherwise return `None` (unrepairable).

**Invariants.**
- `repair` is pure, synchronous, and never raises for any input.
- `clean_value` is always a **prefix** of the input — the repairer never invents or reorders caller text.
- `recovered` values are verbatim substrings of the input — the repairer never synthesises a value.
- Determinism: identical input ⇒ identical output. Pinned by the 334-specimen corpus.

### C2 — Boundary policy contract (`MarkupGuardMiddleware`)

Policy is a **declared enum passed at registration**, never inferred from the tool name and never prose (INV-1):

| Policy | Servers | Behaviour on a validating repair |
|---|---|---|
| `REJECT_WITH_REPAIR` | fused-memory, plan-tools | Reject. The error dict carries `repaired_call` — the complete corrected argument map — so the retry is mechanical and correct. The caller stays the author of its own arguments. |
| `FORWARD_REPAIR` | verdict-tools, escalation | Forward the repaired arguments to the tool. Emit `markup_repaired` and attach a warning to the tool response. Chosen because a lost `submit_review_verdict` strands a review gate and a lost `escalate_info` strands a task (INV-6/INV-7). |

**Unrepairable input is never guessed, under either policy:** reject, and file an escalation carrying the **full raw payload** so nothing is discarded even if the caller never retries. That escalation names its owner and carries the standing L2 age bound (INV-7).

**Override.** `metadata.allow_mcp_markup is True` bypasses the guard for deliberately-quoted markup, preserving today's `markup_tripwire` semantics, and is stripped before the call proceeds.

**Exemptions.** A declared `exempt_tools` frozenset passed at registration — tools whose arguments legitimately contain envelope literals (`scan_memory_content` needles, the sweep tools). Declared at the registration site, machine-checked, not discovered by failure (INV-1).

**Storm escape.** Repair is a fail-soft path, so it carries the rate/streak escalation (INV-4). The existing `MarkupStormCounter` generalises; counts are per `(project, policy_outcome)` so a burst of *repairs* is as visible as a burst of *rejections*.

**Structured facts (INV-2).** Every outcome emits `markup_detected` with `tool`, `param`, `pattern`, `misclose`, `outcome ∈ {repaired, rejected, unrepairable}`, `recovered_params: [names]`, `agent_id`, `project`. No consumer re-derives any of this by log-scraping.

### C3 — Sweep contract

Every rewrite is **atomic**: repair into a temp file in the same directory, verify the result parses (`json.load`), then `os.replace` onto the target. A verification failure leaves the original untouched and reports the path. `--apply` is opt-in; dry-run emitting a full diff is the default.

**Scope split (operator decision, §5 D3):** terminal state is swept now; live worktree plans are repaired lazily on next plan-tools read, under the same atomic contract.

---

## 5. Resolved design decisions

- **D1 — Blanket middleware, not per-boundary call sites.** A `FastMCP.Middleware` in `shared/` registered on all four servers, rather than promoting today's `_markup_gate` and calling it at ~15 boundaries. Per-boundary enumeration is precisely how 11 boundaries ended up unguarded; the middleware covers every tool and every string parameter, including tools added later. The four in-line `_markup_gate` call sites in `fused-memory/server/tools.py` are **retired** in favour of it — one mechanism, not two (INV-5). Cost accepted: first use of FastMCP middleware in this repo.

- **D2 — Deterministic repair, not an LLM repair session.** Considered and rejected. The corruption is a rigid grammar; a schema-validated deterministic repairer fixes 92.2% of the real corpus and recovers 194 dropped parameters (§2.6). Schema validation — recovered names must be real parameters of the invoked tool — is the corroboration that makes repair safe (INV-3), and an LLM cannot perform it. An LLM repairer would also be the same model class that emitted the malformed markup, would sit in the write path, and could plausibly *invent* a `priority` — the silent-wrong-value failure 3083 exists to stop. Deterministic behaviour is pinned by a committed fixture corpus (INV-1); an LLM's is not.

- **D3 — Two-tier boundary policy by retry cost, not a uniform rule.** Where retry is cheap the caller re-issues the repaired call; where a lost write strands the pipeline the middleware forwards the repair. Both tiers preserve all information. The earlier "strip the fragment and accept" option is rejected as lossy now that repair is available.

- **D4 — Terminal state swept eagerly, live plans lazily.** 245 of 296 `plan.json` files are live worktree state read by in-flight tasks; a global rewrite would need a fleet quiesce. Orphaned plans and the 51 escalation records have no live reader and are swept immediately. Live plans are repaired on next plan-tools read with write-back, so the fix arrives without a downtime window and without racing a running task.

- **D5 — `clean_value` is a prefix; `recovered` values are verbatim substrings.** The repairer is forbidden from synthesising text. This is what distinguishes recovery from fabrication and is directly testable.

- **D6 — Claude Code builtin tools are out of scope.** `Agent` (2) and `Edit` (1, itself a false positive — the test fixture in `test_toolcall_xml_leak.py`) are not our servers. The middleware cannot reach them; the upstream report (θ) is the only lever.

- **D7 — The upstream defect is not fixable here.** The originating error is model-side (wrong closing-tag dialect); the *amplification* — silently over-consuming and dropping parameters instead of raising a parse error — is the harness parser's. Neither is in this repo. Everything in this PRD is containment, recovery and repair.

---

## 6. Pre-conditions / substrate (G3 — verified live 2026-08-05)

| Assumed capability | Verification | Result |
|---|---|---|
| `fastmcp.server.middleware.Middleware` with `on_call_tool` | `uv run python -c "import fastmcp; from fastmcp.server.middleware import Middleware"` | ✅ fastmcp **3.2.2**, `on_call_tool` present |
| `FastMCP.add_middleware` | `hasattr(FastMCP,'add_middleware')` | ✅ True |
| Middleware sees tool name + arguments | live probe, `context.message.name` / `.arguments` | ✅ `demo`, `{'content':…,'category':'orig'}` |
| Middleware can read the tool's parameter schema | live probe, `context.fastmcp_context.fastmcp.get_tool(n).parameters['properties']` | ✅ `['category','content','project_id']` — **the validation substrate is real** |
| Mutated arguments reach the tool | live probe, set `arguments['category']='REPAIRED'` | ✅ tool returned `ok:…|REPAIRED|None` — **forward-repair is implementable** |
| `dark-factory-shared` importable from all three packages | `pyproject.toml` workspace dep + live imports | ✅ fused-memory, orchestrator, escalation all import `shared.*` today |
| The four registration sites | grep `FastMCP(` | ✅ `tools.py:1038`, `plan_tools.py:596`, `verdict_tools.py:172`, `escalation/server.py:307` |
| `EscalationQueue` for residue escalations | already used by `markup_tripwire.emit_markup_storm_escalation` | ✅ |
| 334-specimen corpus extractable | `data/orchestrator/agent-transcripts/**/*.jsonl.gz` | ✅ 334 collected, 308 repair, 26 escalate |

**No middleware is registered anywhere in this repo today** — the substrate exists and is proven by probe, but the integration is novel. Recorded as the PRD's principal implementation risk, not as an unverified assumption.

---

## 7. Out of scope

- **Model routing** — excluded by operator decision. For the record, the correlation measured 2026-08-05 over the same task set: **320 of 325 resolvable corruptions were `opus`, 5 were `sonnet`**, against a baseline of 3,277 opus / 853 sonnet invocations — a per-invocation rate of **9.8% vs 0.59%, ≈17×**. By role: architect 239/839 (28.5%), implementer 54/1033 (5.2%), reviewer 19/1231 (1.5%) — confounded, since architects are the heavy plan-tools users. Revisit separately if incidence rises.
- **Claude Code builtin tools** (`Agent`, `Edit`, `Write`) — not our servers (D6).
- **Fixing the harness parser** — outside this repo (D7). θ produces the report only.
- **The Mem0/Graphiti corpus sweep** — done by 3083; §2.5 confirms it holds.
- **Re-litigating the write-time/read-time calibration split** between `markup_tripwire` and `toolcall_xml_leak` — C1 preserves both predicates over one literal set.

---

## 8. Cross-PRD / seam ownership (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/memory-write-path-convergence.md` §9 leaf ο | this PRD supersedes | the write-time tripwire at four fused-memory boundaries | **this PRD** (retires the in-line gates for the middleware) | ι files the paired edit |
| `docs/prds/memory-write-path-convergence.md` §8 row "XML-leak cure" | re-point | row currently reads "DF 3083 (pending)"; 3083 is `done` | **this PRD** | ι files the paired edit |
| DF 3083 (done, `7899eef17b`) | consumes | `toolcall_xml_leak` detector, `scan_memory_content` | 3083 (terminal) — this PRD re-exports its literals from `shared` | C1 |
| DF 3141 (done, `da94baf14a`) | consumes | `markup_tripwire`, `MarkupStormCounter` | 3141 (terminal) — generalised here | C2 |
| `docs/legibility/design-invariants.md` | consumes | INV-1/2/3/4/5/7 gate this batch | that doc (normative) | G7 walk in §9 |

No reciprocal-ownership ambiguity: 3083 and 3141 are both terminal, so this PRD takes the territory uncontested. ι makes that explicit in the owning PRD rather than leaving it inferred.

---

## 9. Decomposition plan (one bullet per task; signals are the G2 gate)

> **`execution_class` on the docs-only leaves (ζ, η, θ, ι) — do NOT "correct" this to `operational`.**
> The valid vocabulary is `EXECUTION_CLASSES = ('code_tdd', 'operational', 'decision')`
> (`fused_memory/reconciliation/recon_self_model.py:238`). These four leaves edit files in the
> repo, so `code_tdd` is correct even though their deliverable is documentation.
> `operational_routing_guard._maybe_coerce` (`:127`) silently rewrites any task with
> `execution_class ∈ {operational, decision}` to `task_kind='deterministic'` +
> `always_escalates=true` — a docs task filed that way **escalates to a human instead of
> editing the documentation**, and the `submit_task` response does not show the rewrite.
> This PRD's first decompose (2026-08-05) hit exactly that and corrected all four in place.
> Original annotation here was an invented `docs` value — inert, but it invited the harmful repair.

**α — `shared.toolcall_markup`: detector, deterministic repairer, committed fixture corpus.** *(intermediate — unlocks β, δ, ε, ζ, θ)*
Modules: `shared/src/shared/toolcall_markup.py`, `shared/tests/`, `shared/tests/fixtures/toolcall_markup_corpus.jsonl`.
Implements C1. Extracts the 334 real specimens from the archived transcripts into a committed corpus (tool, param, supplied keys, raw value) and pins the repairer against it.
*Unlocks:* β (the middleware imports `detect`/`repair`), δ, ε.
*Evidence:* each corpus record carries its **expected outcome** (`repaired` with the expected recovered-parameter names, or `unrepairable`), committed alongside the specimens; replay asserts the repairer matches every committed expectation, that replay is byte-identical across two runs, and that D5 holds for every repaired case (`clean_value` is a prefix of the input; every recovered value is a verbatim substring).
*G6 note — deliberately not a bare threshold.* The reference implementation scores **308 repaired / 26 unrepairable (92.2%)**, and that is the basis for expecting a high rate; but the signal is agreement-with-committed-expectations, not a literal count. A correct implementation that repairs *more* of the 26 ambiguous cases must update the expectation file in the same commit — which is a reviewable improvement, not a RED test. Pinning the literal 308 would make a better repairer look like a regression.

**β — `MarkupGuardMiddleware`: policy enum, structured facts, storm escape.** *(intermediate — unlocks γ)* · depends: α
Modules: `shared/src/shared/mcp_markup_middleware.py`, `shared/tests/`.
Implements C2: `RepairPolicy.{REJECT_WITH_REPAIR,FORWARD_REPAIR}`, `exempt_tools`, `allow_mcp_markup` override, `markup_detected` structured fact, generalised storm counter keyed `(project, outcome)`.
*Unlocks:* γ (registration on the four servers).
*Evidence:* against an in-process `FastMCP` harness, a corrupted call under each policy produces the contracted outcome; a burst of repairs fires the storm escalation (INV-4).

**γ — Register the middleware on all four servers; retire the in-line gates.** *(leaf)* · depends: β
Modules: `fused-memory/src/fused_memory/server/tools.py`, `orchestrator/src/orchestrator/mcp/plan_tools.py`, `orchestrator/src/orchestrator/mcp/verdict_tools.py`, `escalation/src/escalation/server.py`.
Registers with the declared per-server policy from C2 and removes the four `_markup_gate` call sites (D1, INV-5).
*Signal:* a live `submit_task` whose description carries `</description>` + `<parameter name="priority">low` is **rejected with `repaired_call` containing `priority: "low"`**; a live `escalate_info` with the same defect **lands with `suggested_action` populated** and emits `markup_repaired` — both observable through the tool response and the filed escalation record, not by reading storage.

**δ — Retro-sweep of terminal state, atomic.** *(leaf)* · depends: α
Modules: `scripts/sweep_toolcall_markup.py`, `scripts/tests/`.
Implements C3 over `.worktrees-orphaned/**/.task/plan.json` and `data/escalations/**`. Dry-run default; `--apply` writes temp → verify-parse → `os.replace` (D4, operator's atomicity requirement).
*Signal:* `--apply` repairs the 51 escalation records and the orphaned plans; a second run reports **0 remaining**; every rewritten file still parses as JSON and its byte-diff is confined to the corrupted strings.

**ε — plan-tools lazy write-back for live plans.** *(leaf)* · depends: α
Modules: `orchestrator/src/orchestrator/mcp/plan_tools.py`, `orchestrator/tests/`.
On `plan.json` read, if a field is corrupted, repair and write back under C3's atomic contract, emitting a structured fact. No fleet quiesce (D4).
*Signal:* opening a corrupted live plan through plan-tools returns the repaired `rationale`, and the file on disk is repaired atomically — a concurrent reader sees either the old or the new file, never a partial one.

**ζ — Correct `docs/mcp-toolcall-xml-leak.md` §1.** *(leaf)* · depends: α · `execution_class: code_tdd`
3083's §1 states the parser "terminates a string argument **early**" at a quoted closing tag. The specimens show the opposite direction — over-consumption at the *closing position*. Only over-consumption explains the observed signature (fragment **inside** the value **plus** siblings missing), and the stated direction changes the guidance to authors.
*Signal:* the doc states the over-consumption direction, carries the four specimen shapes from §2.1 and the §2.3 blast-radius table, and points at `shared.toolcall_markup` as the single literal source.

**η — Retire the non-canonical evidence-log metadata convention.** *(leaf)* · `execution_class: code_tdd`
Migrate task 3083's `markup_tripwire_rejections_20260730` / `_burst3` to Tier-C `x_`-prefixed keys per `docs/task-authoring.md`, and document `allow_mcp_markup: True` as the correct move for writes that quote the literals — the convention currently manufactures both schema warnings and tripwire rejections, and is self-perpetuating because it is documented inside 3083's own details.
*Signal:* touching task 3083 emits **zero** `task_metadata.schema_warning … code=unknown_key` lines in the fused-memory journal. Baseline measured 2026-08-05 over the journal window 2026-07-01→08-05: **43 such lines** (22 for `…_20260730`, 21 for `…_20260730_burst3`). The escalation record's "17" was a narrower window; 43 is the figure this signal is measured against.

**θ — Upstream harness bug report.** *(leaf)* · depends: α · `execution_class: code_tdd`
A specimen-backed write-up of the parser defect: the four shapes, the over-consumption behaviour, the silent parameter drop, the 0.26% incidence, and the 92.2% deterministic repairability that demonstrates the format is unambiguous enough for the parser to have errored instead.
*Signal:* a committed report at `docs/upstream/toolcall-parser-overconsumption.md`, self-contained enough to file without this repo as context.

**ι — Paired edit to the owning PRD (G4 bookkeeping).** *(leaf)* · depends: γ · `execution_class: code_tdd`
Re-point `docs/prds/memory-write-path-convergence.md` §8's "XML-leak cure" row from "DF 3083 (pending)" to this PRD, and mark §9 leaf ο's in-line tripwire as superseded by the middleware.
*Signal:* the owning PRD's §8 table names this PRD as owner and no longer describes 3083 as pending.

### DAG

```
α ──┬── β ── γ ── ι
    ├── δ
    ├── ε
    ├── ζ
    └── θ
η (independent)
```

### G7 walk (`docs/legibility/design-invariants.md`)

| Invariant | Disposition |
|---|---|
| `contracts-machine-checked` (INV-1) | Policy is a registration-time enum and `exempt_tools` a declared frozenset — not prose, not tool-name heuristics. ✅ |
| `structured-facts-at-failure` (INV-2) | `markup_detected` carries tool/param/pattern/misclose/outcome/recovered_params; no consumer log-scrapes. ✅ |
| `corroborate-before-acting` (INV-3) | Repair is applied only after schema validation against the invoked tool's live parameter set. ✅ |
| `storm-escape-required` (INV-4) | Repair is fail-soft ⇒ storm counter keyed `(project, outcome)`, so a burst of repairs escalates like a burst of rejections. ✅ |
| `no-lockstep-duplication` (INV-5) | One literal set in `shared`; the current `MCP_MARKUP_PATTERNS` / `PREFILTER_NEEDLES` divergence is the defect being fixed; the four in-line gates are retired rather than duplicated. ✅ |
| `status-matches-liveness` (INV-6) | Motivates `FORWARD_REPAIR`: a lost verdict/escalation strands a task in a claimed state. ✅ |
| `holds-owned-and-bounded` (INV-7) | Residue escalations carry the raw payload, a named owner and the standing L2 age bound. ✅ |

No waivers required.

---

## 10. Boundary-test sketch (H) — two-way, producer and consumer sides

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Partial drift, cheap-retry tier | `submit_task` description ends `…</description>\n<parameter name="priority">low`; policy `REJECT_WITH_REPAIR` | Rejected; error carries `repaired_call` with `description` = prefix only and `priority: "low"`; nothing written; `markup_detected outcome=rejected recovered_params=[priority]` |
| B2 | Total drift, cheap-retry tier | `submit_task` tail `</description>\n<priority>medium</priority>\n<agent_id>x</agent_id>\n<metadata">{…}</metadata">\n</invoke>` | Rejected; `repaired_call` recovers all three of `priority`, `agent_id`, `metadata` |
| B3 | Strand-risk tier forwards | `escalate_info` detail ends `…</suggested_action">\n</invoke>`; policy `FORWARD_REPAIR` | Escalation **is filed**; `suggested_action` populated from the recovered value; response carries a repair warning; `markup_detected outcome=repaired` |
| B4 | Last-parameter case, nothing dropped | `add_memory` content ends `…</content>\n</invoke>` | `clean_value` = prefix; `recovered` empty; tier policy applied |
| B5 | Unrepairable residue never guesses | doubly-corrupted `add_reuse_item.how` (one of the 26) | Rejected under **both** policies; escalation filed carrying the **full raw payload**; no partial write |
| B6 | Deliberate quote passes | content quotes `<parameter name=` with `metadata.allow_mcp_markup=True` | Call proceeds unmodified; override stripped before dispatch; no fact emitted |
| B7 | Exempt tool passes | `scan_memory_content(needles=['</content>'])` | Call proceeds unmodified; guard skipped by declared exemption |
| B8 | Schema validation rejects a bad recovery | tail parses but a recovered name is not a parameter of that tool | Treated as unrepairable (B5 path) — never forwarded |
| B9 | Collision is not overwritten | tail recovers `agent_id` but `agent_id` was already supplied | Treated as unrepairable — the middleware never overwrites a caller-supplied argument |
| B10 | Storm escape fires | 3 repairs within the window on one project | Storm escalation filed once, naming outcome `repaired` |
| B11 | Sweep atomicity | δ interrupted between temp-write and replace | Target file unchanged and still parses; no partial JSON |
| B12 | Lazy write-back under concurrency | ε repairs a live plan while a task reads it | Reader observes either the old or the repaired file, never a partial one |
| B13 | Corpus determinism | replay all 334 specimens twice | Byte-identical results both runs; every outcome matches the committed per-specimen expectation (reference: 308 repaired / 26 unrepairable) |

---

## 11. Open questions (tactical, implementation-time)

1. **Middleware ordering.** If a server later adds a second middleware, does the markup guard run first? **Suggested resolution:** register it first and assert its position in a test. Decide during γ.
2. **`add_system_record` / `update_memory` policy tier.** Both are fused-memory writes, so they inherit `REJECT_WITH_REPAIR`; `add_system_record` is recon-stage-only and may not retry. **Suggested resolution:** start with the server default, revisit if the storm counter shows rejections there. Decide during γ.
3. **Fixture corpus size in-repo.** 334 raw values include long text; the committed corpus may be large. **Suggested resolution:** store truncated-but-sufficient values (tail + 200 chars of lead-in) if size is a problem, keeping the 26 unrepairable cases verbatim. Decide during α.
4. **Retention of the archived transcripts.** The corpus is extracted from `agent-transcripts/`, which is retention-bounded. **Suggested resolution:** the committed corpus is the durable artifact; no dependency on the archive after α.
