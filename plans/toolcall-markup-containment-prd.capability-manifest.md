# Capability manifest — toolcall-markup-containment

PRD: `plans/toolcall-markup-containment-prd.md` (committed `52e27ff13f`)
Built at decompose, 2026-08-05, against main tip `37f761f5a4`.
Machine-readable twin: `plans/toolcall-markup-containment-prd.capability-manifest.yaml`.

Mechanizes G3 + G6 over the PRD's 9-label decomposition (α..ι). Every binding below
was re-derived at decompose time against main — the PRD's own §6 table is a **claim**,
and each of its ten rows was independently re-probed here. No fused-memory store,
escalation record or task was mutated while building this manifest; task 3083 was read
via `get_task` only (see Finding 1).

## Operator rulings, 2026-08-05 (post-decompose)

Recorded **alongside** the findings below, which are left as written — their numbers are the
decompose-time claim and stay the historical record.

**Ruling 1 — η released.** Finding 1's hold is discharged: task **3697** was set `pending`.
The manifest's single FAIL binding (`authority-to-mutate-task-3083`) is resolved by operator
authorization, not by re-scoping. η's title and description were rewritten in place to drop
the hold notice.

**Ruling 2 — never permit duplicate tasks.** The three overlaps surfaced under G4 were
resolved by cancelling the standalone follow-up in each pair, with `x_superseded_by` and
`x_cancel_reason` stamped on each:

| Cancelled | Superseded by | Territory |
|---|---|---|
| **3654** | 3688 (α) | the detector/repairer |
| **3662** | 3691 (δ) | the `data/escalations/` sweep |
| **3685** | 3692 (ε) | live `plan.json` repair |

A **fourth** overlap was found while executing this ruling and is the most dangerous of the
set: task **3643** (pending) listed `shared/src/shared/mcp_markup.py` in its `metadata.files`
as a rival "single envelope-literal enumeration", plus a guard on the escalation write
boundary and edits to `markup_tripwire.py` — colliding with α, γ **and** the narrow-file-lock
model. Two pending tasks each building *the* single enumeration in `shared/` would **be** the
INV-5 lockstep duplication this PRD exists to eliminate.

3643 was **narrowed, not cancelled**: its mechanism half is superseded, but its verification
question — *does the landed containment cover the `claude-task-3514-implementer` producer
path?* — is owned by no one else, and PRD §5 D7 explicitly puts the producer/parser defect
outside this repo. Its `metadata.files` was reduced to the escalation specimens and its own
doc, and **`3643 depends_on 3690`** was wired so it runs against the final architecture
instead of racing it.

**Ruling 3 — `code_tdd` is correct for documentation.** Anything that mutates files is
`code_tdd`, and editing documentation mutates files. Finding 2 below correctly identified that
`'docs'` is invalid, but resolved it to `'operational'`, which was **also wrong** — and
measurably so:

> Filing ζ/η/θ/ι with `execution_class='operational'` **silently coerced** them to
> `task_kind='deterministic'` + `always_escalates=true`. All four would have **escalated to a
> human instead of editing the documentation**. This was not visible in the `submit_task`
> response; it was found by reading the stored metadata back.

All four (3693, 3695, 3696, 3697) were corrected in place to `execution_class='code_tdd'`,
`task_kind='normal'`, `always_escalates=false`, and their descriptions rewritten to record the
correction. `x_docs_only` was cleared. **PRD §9 should be amended** — its four
`execution_class: docs` annotations are the upstream source of this error.

Finding 5 (that `execution_class` is itself absent from `_BLESSED_METADATA_KEYS`) stands
unchanged and is now folded into η/3697's docs half as optional scope.

## Substrate re-verification (G3)

All ten of §6's rows hold at the cited locations. Line numbers re-derived, not copied.

| Capability | PRD cites | Observed on `37f761f5a4` | Verdict |
|---|---|---|---|
| `fastmcp` with `Middleware.on_call_tool` | fastmcp 3.2.2 | `fastmcp.__version__ == '3.2.2'`, `hasattr(Middleware,'on_call_tool')` True | PASS |
| `FastMCP.add_middleware` | present | True | PASS |
| `FastMCP.get_tool` (schema read) | `context.fastmcp_context.fastmcp.get_tool` | `hasattr(FastMCP,'get_tool')` True | PASS |
| Registration site — fused-memory | `tools.py:1038` | `fused-memory/src/fused_memory/server/tools.py:1038` | PASS |
| Registration site — plan-tools | `plan_tools.py:596` | `orchestrator/src/orchestrator/mcp/plan_tools.py:596` | PASS |
| Registration site — verdict-tools | `verdict_tools.py:172` | `orchestrator/src/orchestrator/mcp/verdict_tools.py:172` | PASS |
| Registration site — escalation | `escalation/server.py:307` | `escalation/src/escalation/server.py:307` | PASS |
| `_markup_gate` call sites to retire | "four in-line call sites" | def at `tools.py:1121`; calls at `:1828`, `:2020`, `:5446`, `:5990` — **exactly four** | PASS |
| `MCP_MARKUP_PATTERNS` (the 3-literal tuple) | §2.2 quotes it verbatim | `markup_tripwire.py:85` — `('</content>', '<parameter name=', '</invoke>')`, **byte-identical to the PRD's quote** | PASS |
| `PREFILTER_NEEDLES` (the divergent 4-closer list) | `toolcall_xml_leak.py` | `fused-memory/src/fused_memory/utils/toolcall_xml_leak.py:115` | PASS |
| `MarkupStormCounter` / `emit_markup_storm_escalation` | already used | `markup_tripwire.py:294` / `:374` | PASS |
| `shared` importable everywhere | workspace dep | `shared/src/shared/` present with 25+ modules; `shared/tests/` present | PASS |
| Transcript corpus | ~4,400 `.jsonl.gz` | **4,498** — grew as §2 predicted ("the tree grows live") | PASS |

`docs/upstream/` does not exist yet; θ creates it. Not a substrate gap — θ's deliverable
is the directory's first file.

**One stale path in PRD prose (non-blocking).** §1's G1 table routes residue escalations
to `escalation/data/escalations`. That path **does not exist**; the live corpus is
`data/escalations/` at the repo root (3,377 JSON files). δ's own §9 entry names the
correct path, so no contract is affected — recorded so a reader of §1 is not misled.

## Extent re-measurement (the numbers the leaf signals bind)

Scanned with the PRD's own §2 collection predicate
(`</invoke>\s*$` **or** `</[A-Za-z_]\w*>\s*<parameter\s+name="[^"]+">`) over JSON string
values, at decompose time:

| Corpus | Files scanned | Corrupted strings | Files hit | Owner |
|---|---|---|---|---|
| `data/escalations/**/*.json` | 3,377 | **49** | 48 | δ |
| `.worktrees-orphaned/**/.task/plan.json` | 15 | **2** | 1 | δ |
| `.worktrees/**/.task/plan.json` (live) | 265 | **88** | **21** | ε |
| `docs/task-recovery-2026-05-13/worktree-inventory.json` | 1 | 3 | 1 | **nobody — must not be touched** |

This **corroborates** the PRD's §2.4 and D4 split rather than contradicting it: §2.4's
"21 of 296 retained plan.json files, 87 corrupted strings" reproduces almost exactly as
the **live** set (21 files, 88 strings), which D4 assigns to ε. See Findings 3 and 4 for
the two consequences.

## G6 premise re-validation

Five findings. Two blocked and are resolved at filing; two are pinned into task text;
one is recorded for the operator.

### Finding 1 — η's core action collides with the operator's hard constraint (BLOCK → held)

η is "Migrate task **3083**'s `markup_tripwire_rejections_20260730` / `_burst3` to Tier-C
`x_`-prefixed keys". Both keys are confirmed present on task 3083's metadata (read-only
`get_task`, 2026-08-05), so η's premise is **true**. But executing η **is** an
`update_task` against task 3083 — and the operator's standing instruction for this
decompose is that task 3083 and escalation `esc-markup-tripwire-3` are *deliberately left
for him*. A dispatched η would violate that directly.

Compounding it: 3083's own `details` records why the mutation is delicate — "any update
rewrites the whole column, and a curator `combine` verbatim-replaces it" — over a ~6 KB
hand-curated evidence log with a standing DO-NOT-LOSE warning. This is not autonomous work.

**Resolution (G6 (a), do not queue an impossible-to-honour capability):** η is **filed but
deliberately excluded from the `commit_planning` flip**. It stays `deferred` — not
schedulable, nothing dispatched against 3083 — with the operator named in its description
as the owner who releases it. Nothing is lost; the decision is returned to the person who
reserved it. INV-7: the hold names a machine-readable owner and is surfaced here, in the
task text, and in the session hand-back.

### Finding 2 — `execution_class: docs` is not a valid value (BLOCK → corrected)

The PRD (§9) and the decompose instruction both specify `metadata.execution_class=docs`
for ζ, η, θ, ι. The vocabulary does not contain it:

```
fused-memory/src/fused_memory/reconciliation/recon_self_model.py:238
    EXECUTION_CLASSES = ('code_tdd', 'operational', 'decision')

fused-memory/src/fused_memory/middleware/routing_intent_guard.py:88-90
    _EXEMPT_EXECUTION_CLASSES = frozenset(c for c in EXECUTION_CLASSES if c != 'code_tdd')
```

`'docs'` is therefore **not** in the routing-intent lint's exempt set
(`routing_intent_guard.py:229`). Filing these four docs-only leaves with `docs` would
leave them *unexempted* — the precise failure `decompose-mode.md` Step 3 warns about,
where a `task_kind="normal"` leaf whose own text declares a no-code path is flagged or
rejected for carrying no matching declaration. It would also land an out-of-vocabulary
value: the same manufactured-metadata-noise class that **η itself exists to retire**.

**Resolution (G6 (a), rewrite to an existing capability):** ζ, η, θ, ι are filed with
`execution_class='operational'` — the vocabulary's sanctioned non-code class, and the
one the lint actually honours — plus a Tier-C `x_docs_only: true` marker preserving the
operator's docs-only intent losslessly. Deviation from the literal instruction, flagged
here and in the hand-back.

### Finding 3 — δ's "orphaned plans" extent is 1 file, not 21 (extent → pinned)

δ's signal reads "`--apply` repairs the 51 escalation records **and the orphaned plans**".
Measured: the orphaned-plan share is **1 file / 2 strings**; the 21-file pocket is *live*
and belongs to ε (see the extent table). And the escalation corpus measures **49**, not 51
— a small drift, in a corpus that is still being written to.

Pinning literal counts in a signal is the hazard the PRD already names for α ("pinning the
literal 308 would make a better repairer look like a regression"); the same logic applies
here. **Resolution:** the measured extents are pinned into δ's description as *decompose-time
observations*, and δ's binding signal is the predicate-based invariant the PRD already
supplies — **a second `--apply` run reports 0 remaining, and every rewritten file still
parses** — not the literal 51.

### Finding 4 — a committed evidence file sits inside δ's blast radius (NEW hazard → pinned)

`docs/task-recovery-2026-05-13/worktree-inventory.json` is **git-tracked** and legitimately
contains 3 predicate-matching strings — it is an evidence document that *quotes* leak
specimens. Because it is checked in, it is replicated into every worktree checkout: a
δ implementation globbing `.worktrees-orphaned/**/*.json` instead of the exact
`.worktrees-orphaned/**/.task/plan.json` path hits it **47 times across 16 directories**
and would rewrite committed evidence.

This is the file-level analogue of the specimen-loss warning already standing on task 3083.
**Resolution:** δ's scope is pinned to the exact `.task/plan.json` path in its description,
and the inventory file is named as must-not-touch, with a mechanical `expect: present`
delivered-check asserting its markup **survives** the sweep.

### Finding 5 — `execution_class` is not on the Tier-A allowlist (observation, not filed)

`_BLESSED_METADATA_KEYS` (`shared/src/shared/task_metadata.py:725-757`) does not list
`execution_class`; it is neither a typed `TaskMetadata` field nor a registered submodel,
yet `execution_class_guard` and `routing_intent_guard` both read it. So *every* task
carrying it plausibly emits the same `task_metadata.schema_warning … code=unknown_key`
line that η is filed to eliminate on 3083. Pre-existing and systemic — not caused by this
batch and outside this PRD's scope. Recorded for the operator; **not** filed as a task,
per "don't file follow-ups for things outside the PRD".

## G4 — three live pending tasks overlap this batch (surfaced, not mutated)

The PRD's §8 seam table covers 3083, 3141, `memory-write-path-convergence.md` and
`design-invariants.md`. It does **not** mention three tasks that are `pending` right now
and whose scope this batch subsumes:

| Task | Status | Overlap |
|---|---|---|
| **3662** | pending | "No discovery sweep exists over the escalation-queue corpus (`data/escalations/`)" — this is **δ's escalation half**, filed separately on 2026-08-05 |
| **3685** | pending | "Envelope markup leaks into `.task/plan.json` via plan-tools — 4th unswept surface (144 fields / 23 worktrees)" — this is **ε's territory** |
| **3654** | pending | "Add a third content-preserving repairable shape to the toolcall-XML detector" — overlaps **α's** repairer |

None is a reciprocal-ownership ambiguity, so G4 does not block. But all three are
schedulable, so the orchestrator can dispatch duplicate work against δ/ε/α. Deciding
their disposition (cancel as superseded, or re-point onto this batch) is the operator's
call and was **not** actioned here — no task was mutated.

## G5 — B+H: the 13-row boundary sketch has full leaf ownership

There is no single integration-gate task, but every row of §10 has exactly one owner and
no row is orphaned:

| Row | Owner | Row | Owner |
|---|---|---|---|
| B1 partial drift, reject-tier | γ | B8 schema validation rejects bad recovery | α |
| B2 total drift, reject-tier | γ | B9 collision not overwritten | α |
| B3 strand-risk tier forwards | γ | B10 storm escape fires | β |
| B4 last-parameter case | α | B11 sweep atomicity | δ |
| B5 unrepairable never guesses | β | B12 lazy write-back under concurrency | ε |
| B6 deliberate quote passes | β | B13 corpus determinism | α |
| B7 exempt tool passes | β | | |

## Per-leaf capability bindings

Evidence key: `wired` = a production entry path exists on main today or is produced by a
task in this leaf's own transitive dependency closure; `producer:X` = produced upstream by
label X; `measured` = re-derived at decompose by direct probe. Verdict `OPEN` — added to the
vocabulary by task **4471**, after this file was last written, which is why it appears nowhere
above — = the binding is deliberately **undecided** and the decision is homed in that leaf as its
own work product; it does not block queueing and must never be read as a green G3 binding
(`shared/src/shared/capability_manifest.py`; `plans/capability-delivered-checks-prd.md` §Contract).

### α — `shared.toolcall_markup` (detector + repairer + fixture corpus)

| Capability | Binding | Verdict |
|---|---|---|
| detector/repairer module in `shared/` | `wired` — `shared/src/shared/` is a live package imported by all three others; `shared/tests/` exists | PASS |
| committed 334-specimen corpus extractable | `measured` — 4,498 `.jsonl.gz` present under `data/orchestrator/agent-transcripts/` | PASS |
| single literal source (INV-5) | `wired` — both divergent sites exist to consolidate: `markup_tripwire.py:85`, `toolcall_xml_leak.py:115` | PASS |
| expectation-agreement signal (not a bare 308) | `floor` deliberately **absent** — PRD's own G6 note; the signal is agreement-with-committed-expectations | PASS |
| D5 prefix/substring invariant | `wired` — directly testable over the committed corpus | PASS |

### β — `MarkupGuardMiddleware`

| Capability | Binding | Verdict |
|---|---|---|
| `RepairPolicy` enum at registration (INV-1) | `producer:α` + `wired` — `fastmcp 3.2.2` Middleware substrate probed live | PASS |
| tool parameter-schema read | `wired` — `FastMCP.get_tool` present (§6 row re-probed) | PASS |
| `markup_detected` structured fact (INV-2) | `producer:β` | PASS |
| storm counter keyed `(project, outcome)` (INV-4) | `wired` — `MarkupStormCounter` at `markup_tripwire.py:294` generalises | PASS |
| residue escalation carrying raw payload (INV-7) | `wired` — `emit_markup_storm_escalation` at `:374`; `data/escalations/` live | PASS |

### γ — register on four servers; retire the in-line gates — **SPLIT THREE WAYS 2026-08-19**

**Superseded by γ1/γ2/γ3 below.** The decompose-time table is left exactly as written — it is the
2026-08-05 claim and stays the historical record — with the correction recorded **alongside** it,
mirroring how Ruling 1 is recorded above and how the `.yaml` twin preserves η's decompose-time
FAIL in that capability's binding prose rather than deleting it.

| Capability | Binding | Verdict |
|---|---|---|
| four registration sites | `measured` — all four `FastMCP(` sites confirmed at the cited lines | PASS |
| in-flight argument mutation (forward-repair) | `wired` — PRD §6 live probe; `add_middleware` present | PASS |
| four `_markup_gate` call sites to retire (INV-5) | `measured` — exactly four at `:1828 :2020 :5446 :5990` | PASS |
| live `submit_task` rejects with `repaired_call` | `producer:β` — in γ's closure (γ←β←α) | PASS |
| live `escalate_info` lands with `suggested_action` | `producer:β` — in γ's closure | PASS |

**Correction, 2026-08-19** (operator commit `965f3206eb`; recorded authoritatively in PRD §9).
The single γ leaf declared all four servers and was therefore **never dispatched once** —
`data/orchestrator/runs.db` carries zero events of any type for task **3690** — because it
required simultaneous module locks on the two hottest files in the repo. Nothing about the
contracts changed; only the packaging. Two rows above do **not** survive the split intact:

- **`four registration sites` was a FALSE PASS.** §6 probed `add_middleware` against the
  STANDALONE `fastmcp` class and then confirmed "four sites" by grepping the `FastMCP`
  constructor — a class-**name** match that does not distinguish the two implementations.
  fused-memory imports the MCP SDK's *bundled* `mcp.server.fastmcp.FastMCP`, which has neither
  `add_middleware` nor `get_tool`. Re-verdicted `OPEN` under γ3.
- **the `_markup_gate` anchors `:1828 :2020 :5446 :5990` are stale.** The four call sites are at
  `fused-memory/.../tools.py:2701, 2936, 6686, 7239` (def at `:1144`). Re-verdicted `OPEN` under γ3.

### γ1 — register on the escalation and verdict-tools servers (the `FORWARD_REPAIR` pair) — task **3690**

| Capability | Binding | Verdict |
|---|---|---|
| `middleware-registered-on-forward-repair-pair` | capability→producer (`measured` 2026-08-19, live `hasattr` probe) — both sites import the STANDALONE `fastmcp 3.2.2`, where `add_middleware` and `get_tool` are present | PASS |
| `list-typed-evidence-recovery-pinned` | capability→producer (**UNMEASURED**) — `Repair.recovered` is typed `dict[str, str]` and the middleware applies it as a raw `arguments.update` with no type coercion; `escalate_info.evidence` is list-typed, so the esc-3184-2 specimen exercises a case B3 does not | OPEN |
| `live-escalate-info-lands-with-suggested-action` | `producer:β` — in γ1's closure (γ1←β←α); argument mutation reaching the tool confirmed by PRD §6 live probe | PASS |

*Provenance of the split (the `.yaml` twin's γ1 task-level `note:`, rendered here).* **SPLIT
2026-08-19.** The original γ declared all four servers and was never dispatched once — zero
`runs.db` events, ever, in the seven days after β landed — because it required simultaneous
module locks on the two hottest files in the repo. γ2 (task **4457**) carries plan-tools; γ3
(task **4458**) carries fused-memory. γ1 also absorbs task **4180** (cancelled into it).

**Re-measured 2026-09-22 — currency (the `.yaml` twin's γ1 `note:` carries the same clause).**
Task **3690** is `done`; registration landed in `07a967fab0`. The `OPEN` verdict in the table
above is reproduced from the `.yaml` **as authored on 2026-08-19** and is deliberately not
overwritten — it is the authoring-time record, kept the way Ruling 1 and the γ correction above
are kept, with the current reading recorded alongside. Disposition today:
`list-typed-evidence-recovery-pinned` — the row's grep `delivered_check` (`evidence` present in
`escalation/tests/test_markup_middleware_registration.py`) is satisfied, 44 occurrences.

### γ2 — register on plan-tools, and rule on the overlap with ε's read-time path — task **4457**

| Capability | Binding | Verdict |
|---|---|---|
| `middleware-registered-on-plan-tools` | capability→producer (`measured` 2026-08-19) — `plan_tools.py` imports the STANDALONE `fastmcp 3.2.2`; site moved to `:1553`, +957 from the `:596` the PRD cites | PASS |
| `d1-ruling-recorded-against-3692s-read-time-path` | capability→decision — task 3692 (ε, done) landed an independent read-time repairer in the same file (`_with_markup_repairs` at `:982`, ~25 call sites) that cannot reject, memoizes refusals per-process, and deliberately leaves `_create_plan`'s inbound arguments to this middleware. Composing vs superseding must be decided and tested **here**, not left implicit (D1, INV-5). | OPEN |
| `live-plan-tools-write-rejected-with-repaired-call` | `producer:β` — in γ2's closure (γ2←β←α); in-flight argument access proven by PRD §6 live probe | PASS |

**Re-measured 2026-09-22 — currency (the `.yaml` twin's γ2 `note:` carries the same clause).**
Task **4457** is `done`; registration landed in `37eed69c97`. The `OPEN` verdict above is
reproduced from the `.yaml` **as authored on 2026-08-19** and is deliberately not overwritten.
Disposition today: `d1-ruling-recorded-against-3692s-read-time-path` — the ruling it awaited was
made, and it is **compose**, not supersede. The middleware guards INBOUND ARGUMENTS at the
request layer; task 3692's read-time path repairs STORED STATE inside the tool body. The two
populations are disjoint, so registering the guard does not retire 3692. The ruling is prose in
`orchestrator/src/orchestrator/mcp/plan_tools.py::_create_plan` and is pinned by
`orchestrator/tests/test_plan_tools_markup_guard.py::TestComposesWithTheReadTimeRepair` — the
prose-plus-test pair the row's `manual` check calls for.

### γ3 — adapt the guard to fused-memory's **bundled** FastMCP; cover the two ungated write tools; only then retire the in-line gates — task **4458**

| Capability | Binding | Verdict |
|---|---|---|
| `fused-memory-boundary-adapted` | capability→producer — ⚠️ **THE ORIGINAL γ ROW FOR THIS SITE WAS A FALSE PASS.** §6 verified `add_middleware` against the STANDALONE `fastmcp` class only and then confirmed the four sites by grepping the `FastMCP` constructor — a NAME match that does not distinguish the two classes. Measured 2026-08-19: `fused-memory/server/tools.py:17` imports `mcp.server.fastmcp.FastMCP` (the MCP SDK's **BUNDLED** class), where `hasattr add_middleware` is False and `hasattr get_tool` is False. The middleware is unattachable here as written; this leaf must choose between migrating to standalone `fastmcp` or adapting through the existing `_safe_call_tool` chokepoint in `server/main.py`. | OPEN |
| `ungated-write-tools-now-covered` | capability→producer (`measured` 2026-08-19) — `add_system_record` (`tools.py:3102`) and `update_memory` (`tools.py:4316`) have no gate of any kind; both write to Mem0 and both carry optional trailing parameters, which is the shape where a swallow is silent rather than loud | OPEN |
| `inline-markup-gate-call-sites-retired-inv5` | capability→producer (`measured` 2026-08-19) — exactly four call sites at `tools.py:2701, 2936, 6686, 7239` (def at `:1144`); the `1828/2020/5446/5990` anchors in the original row are stale. **MUST NOT** be retired before the replacement is live and demonstrated — this is the one boundary whose containment provably works (Mem0 frozen since 2026-07-30, Graphiti since 2026-07-29). | OPEN |

**Re-measured 2026-09-22 — currency (the `.yaml` twin's γ3 `note:` carries the same clause).**
Task **4458** is `done`. All three `OPEN` verdicts above are reproduced from the `.yaml` **as
authored on 2026-08-19** and are deliberately not overwritten; all three are discharged.

- `fused-memory-boundary-adapted` — the fork the row poses was decided for **adapt**, not
  migrate: the guard attaches through the existing `_safe_call_tool` chokepoint as
  `fused-memory/src/fused_memory/server/markup_guard.py::install_markup_guard`, called from
  `fused-memory/src/fused_memory/server/main.py::_install_tool_dispatch_guards`, landed
  `60293e0d8c` on 2026-08-20. This is the same live path `docs/mcp-toolcall-xml-leak.md` names.
  The row's `manual` `delivered_check` reason still holds under the option taken — an
  `add_middleware` grep remains unsatisfiable at this site by design, so do not re-add one.
- `ungated-write-tools-now-covered` — the row's grep check (`add_system_record` present in
  `fused-memory/tests/test_markup_guard_fused_memory.py`) is satisfied, 57 occurrences.
- `inline-markup-gate-call-sites-retired-inv5` — the **MUST NOT** no longer binds, because the
  replacement it guarded against is live. The four gates were retired in `5d066281c1`, and the
  row's `expect: absent` grep passes: zero definitions or call sites remain under
  `fused-memory/src/`, every surviving mention there being past-tense prose (e.g.
  `markup_guard.py` reads "The retired `_markup_gate`").

### δ — retro-sweep of terminal state, atomic

| Capability | Binding | Verdict |
|---|---|---|
| `data/escalations/` sweep target | `measured` — 3,377 JSON, 49 corrupted strings / 48 files | PASS |
| `.worktrees-orphaned/**/.task/plan.json` target | `measured` — 15 files, 2 corrupted strings / 1 file (**extent pinned, Finding 3**) | PASS |
| atomic temp→verify-parse→`os.replace` (C3) | `producer:δ` — pure stdlib | PASS |
| "second run reports 0 remaining" | `producer:δ` — predicate-based, no literal count | PASS |
| must-not-touch: committed evidence file | `measured` — Finding 4; bound as an `expect: present` survival check | PASS |

### ε — plan-tools lazy write-back

| Capability | Binding | Verdict |
|---|---|---|
| corrupted live plans exist to repair | `measured` — **88 corrupted strings across 21 of 265** live `.task/plan.json` | PASS |
| plan-tools read path to hook | `wired` — `orchestrator/src/orchestrator/mcp/plan_tools.py:596` | PASS |
| atomic write-back under C3 | `producer:α` (repairer) + `producer:ε` (write path) | PASS |

### ζ — correct `docs/mcp-toolcall-xml-leak.md` §1

| Capability | Binding | Verdict |
|---|---|---|
| the incorrect claim is actually present | `measured` — `docs/mcp-toolcall-xml-leak.md:26-27` reads "terminates a string **argument** early". **Premise TRUE** | PASS |
| over-consumption direction is establishable | `producer:α` — the corpus is the evidence | PASS |

### η — retire the non-canonical evidence-log metadata convention

| Capability | Binding | Verdict |
|---|---|---|
| the two non-canonical keys exist on 3083 | `measured` — `markup_tripwire_rejections_20260730` and `…_burst3` both present (read-only `get_task`). **Premise TRUE** | PASS |
| 43-line `unknown_key` baseline | **unverified at decompose** — a closed-window historical journal measurement; the journal scan exceeded its time budget. Non-blocking: the binding signal is *zero going forward*, not the baseline | PASS (noted) |
| authority to mutate task 3083 | `rejection-absent` (authority not held) — reserved to the operator (Finding 1) | **FAIL** |

η's single **FAIL** binding is the one that blocks it from queueing, and is the reason it is
filed `deferred` and excluded from the flip. It is the only FAIL in the manifest; the other
eight tasks clear cleanly.

### θ — upstream harness bug report

| Capability | Binding | Verdict |
|---|---|---|
| specimen-backed content | `producer:α` — the four shapes + repairability figure come from α's corpus | PASS |
| `docs/upstream/` target | new directory created by θ; no substrate dependency | PASS |

### ι — paired edit to the owning PRD

| Capability | Binding | Verdict |
|---|---|---|
| §8 row still reads "DF 3083 (pending)" | `measured` — `docs/prds/memory-write-path-convergence.md:115`, and `:5`. **Premise TRUE** | PASS |
| §9 leaf ο exists to mark superseded | `measured` — `:144` | PASS |
| the superseding mechanism has landed | `producer:γ` — ι depends on γ, so the claim is true when ι runs | PASS |

## G7 walk (`docs/legibility/design-invariants.md`)

Re-walked per task against all seven invariants. The PRD's §9 table holds; **no waivers
required**, and no task carries `metadata.g7_waivers`.

One addition to the PRD's walk: η's `deferred` hold (Finding 1) is an INV-7 subject. It
names a machine-readable owner (the operator), its reason is recorded in the task
description, this manifest and the session hand-back, and its exit is a one-call
`commit_planning` once he releases it. Deliberate operator reservation, not a strand.

## Bindings that had to be resolved before filing

| # | Binding | Resolution |
|---|---|---|
| 1 | η → mutate task 3083 | **Held.** Filed `deferred`, excluded from the flip, operator named as owner |
| 2 | ζ/η/θ/ι → `execution_class: docs` | **Rewritten** to `execution_class: operational` + Tier-C `x_docs_only: true` |
| 3 | δ → "the 51 escalation records and the orphaned plans" | **Extent pinned** (49 / 2); binding signal moved to the predicate invariant |
| 4 | δ → sweep glob blast radius | **Scope pinned** to `.task/plan.json`; committed evidence file named must-not-touch with a survival check |

No binding resolved to `declared-only`, `test-only`, `producer-downstream`,
`producer-absent`, `fixture-ERROR`, `bound≤floor` or `rejection-absent`. The batch clears
the manifest gate.
