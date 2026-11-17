# Capability manifest — plans/consolidation-adjudicator-prd.md

**PRD:** `plans/consolidation-adjudicator-prd.md`, as amended at decompose (§10 splits the check
surfaces into task χ and reshapes gates Γ, φ and ρ2). **Machine twin:**
`plans/consolidation-adjudicator-prd.capability-manifest.yaml`; every Evidence cell below is that
sidecar's `binding` verbatim, and every Delivered check is its `delivered_check`.
**Substrate verified** 2026-11-17 against main `7b31131e51` (the PRD's own commit; its anchors were
taken at `a473533aee`). Each `grep:` span was resolved on main itself: `git cat-file -e
main:<path>`, and for `::<name>` a module-level `def`, `class` or assignment (or `Class.method`) in
`git show main:<path>`.

Batch DAG: α → β → γ → η → ζ → χ → Γ → ε → φ → ρ1 → ρ3; ρ2 depends on φ and χ. ζ is existing
task 5242 (below); it is not a block in this sidecar.

**Polarity.** Every `grep` and `path` check fails on main `7b31131e51`: each `expect: present`
pattern or path is absent today, and each `expect: absent` pattern is present today.
`shared.delivered_check_polarity.lint_delivered_checks` reports no finding for any block. The two
gate `script` checks are not linted; both were run as bound and exit 1 today, because
`scripts/check_auto_consolidation.py` does not exist yet and the wrapper fails on a failing
subcheck.

## Decompose-time contracts (the checks depend on them)

1. **η test names.** Each boundary row A*N* of PRD §5 has a test named `test_a<N>_<slug>` in
   `fused-memory/tests/test_auto_consolidation_e2e.py`. This follows the file's existing
   `test_b<N>_<slug>` convention for the parent's rows. The η table below is the A1–A21 mapping
   the PRD's η signal asks for.
2. **ε spelling.** `consolidation_auto:` starts at column 0 of `fused-memory/config/config.yaml`,
   and the staging line is the flow list `enabled_projects: ['dark_factory']`, on one line, as
   the PRD writes it. A block-style list (`- dark_factory`) would leave ε's check red.
3. **χ argv words.** The three checks are selected by the argv words `code-live`,
   `live-clean` and `rejections`, which is how Γ, φ and ρ2 call them.
4. **Gate arguments.** The Γ and φ `delivered_check`s are the gates' own predicates. Each gate
   task's `before_done` uses the same script with the same args, so a re-base edits both.
   Timeouts: the wrapper's `--subcheck-timeout` is left at its default of 90 s. That is below the
   check's `timeout_secs` of 120, which equals the orchestrator's
   `delivered_checks.check_timeout_secs` default (120).

## α — Predicate and ledger shape (intermediate → β, γ)

K1's class table and K2's waivers in `consolidation_auto.py`; K6's states, outcome fields,
`members_key`, retention and streak rule in `consolidation_proposal.py`.

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `predicate-and-ledger-substrate` | `grep:fused-memory/src/fused_memory/reconciliation/consolidation_auto.py::AutoReasonCode` `grep:fused-memory/src/fused_memory/reconciliation/consolidation_auto.py::evaluate_auto_predicate` `grep:fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py::_evaluate` `grep:fused-memory/src/fused_memory/reconciliation/consolidation_proposal.py::ProposalOutcome.from_payload` wired: the closed reason-code enum K1 classifies, the pure predicate K2 extends and its one executor caller, and the outcome decoder K6 widens | PASS | — |
| `reason-code-class-table` | built here — K1's one table from AutoReasonCode to class (no-op, repair, retry, refuse, adjudicator) in consolidation_auto.py | PASS | grep `(ADJUDICATOR\|[Aa]djudicator)` expect present in `fused-memory/src/fused_memory/reconciliation/consolidation_auto.py` |
| `every-reason-code-classified` | built here — a test pins that every AutoReasonCode member has exactly one class | PASS | manual — Enum completeness is a property of the table's keys against the enum, which a grep cannot express; α's own test pins it and the merge gate runs it. |
| `predicate-takes-waived` | built here — evaluate_auto_predicate gains the keyword argument waived (frozenset of (code, member id)) and raises on a waiver outside the adjudicator class | PASS | grep `waived *:` expect present in `fused-memory/src/fused_memory/reconciliation/consolidation_auto.py` |
| `proposal-state-dropped` | built here — ProposalState gains dropped | PASS | grep `= *['"]dropped['"]` expect present in `fused-memory/src/fused_memory/reconciliation/consolidation_proposal.py` |
| `proposal-state-unresolved` | built here — ProposalState gains unresolved | PASS | grep `= *['"]unresolved['"]` expect present in `fused-memory/src/fused_memory/reconciliation/consolidation_proposal.py` |
| `outcome-members-key` | built here — ProposalOutcome gains members_key (and kept_ids, repaired, effective_claim, effective_category, reused_from, adjudication), each read with a default so pre-change rows decode | PASS | grep `members_key` expect present in `fused-memory/src/fused_memory/reconciliation/consolidation_proposal.py` |

`from_payload` reads list fields through `_str_tuple`, which raises on a missing key. That is why
every new outcome field is read with a default (PRD §7, A20).

## β — Adjudicator, calibration, OpenAI helper, config (intermediate → γ)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `transport-and-config-substrate` | `grep:fused-memory/src/fused_memory/server/write_triage_judge.py::_call_openai_responses` `grep:fused-memory/src/fused_memory/server/write_triage_judge.py::_call_llm` `grep:fused-memory/src/fused_memory/config/schema.py::ConsolidationAutoConfig` `grep:fused-memory/src/fused_memory/config/reload.py::RELOADABLE_FIELDS` wired: the judge reaches the Responses transport through _call_llm (β parameterises it into the shared helper); the config block and the hot-reload allowlist K7 edits | PASS | — |
| `eval-set-committed` | `grep:fused-memory/calibration/consolidation_adjudicator_eval_set.jsonl` committed with the PRD (N6); β pins its sha256 as a module constant and resolves it from the package | PASS | — |
| `k1-class-table-and-ledger-states` | producer: α (upstream: β ← α) — K1's class table (calibration applies the case-decidable rules first) and the ledger kinds the calibration record needs | PASS | — |
| `adjudicator-module` | built here — fused-memory/src/fused_memory/reconciliation/consolidation_adjudicator.py (prompt, schema, validation, identity, keep rule, call); imports no executor code | PASS | path expect present: `fused-memory/src/fused_memory/reconciliation/consolidation_adjudicator.py` |
| `calibration-entry-point` | built here — consolidation_adjudicator_calibration.py::run_calibration with the E7 bounds as module constants | PASS | grep `def run_calibration\(` expect present in `fused-memory/src/fused_memory/reconciliation/consolidation_adjudicator_calibration.py` |
| `adjudications-budget` | built here — ConsolidationAutoConfig.max_adjudications_per_cycle (default 6), one of the eight new hot-reloadable leaves | PASS | grep `^ +max_adjudications_per_cycle *:` expect present in `fused-memory/src/fused_memory/config/schema.py` |

The retirement check is anchored on the field declaration and the allowlist string, not on the
bare name. A docstring or yaml comment that mentions the retired knob therefore cannot keep it
red.

**Settled at filing (2026-11-17).** Removing `max_gate_filings_per_cycle` breaks its readers in
`consolidation_auto_executor.py` and their tests, which γ rewrites. So β only adds the eight new
leaves (the reload test's leaf count goes 14 to 22) and γ removes the old one (22 to 21); the
`gate-filings-budget-retired` row sits in γ's block.

## γ — Executor flow (intermediate → η)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `executor-call-site` | `grep:fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py::MemoryConsolidator._run_auto_consolidation` wired: awaits run_auto_consolidation on full cycles only | PASS | — |
| `subset-execute-and-revert-substrate` | `grep:fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py::apply_auto_verdict` `grep:fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py::revert_auto_consolidation` exist: apply takes claim and verdict.retain_ids; revert has no production caller today | PASS | — |
| `predicate-waivers-ledger-and-adjudicator` | producer: α, β (upstream: γ ← α, β) — K1/K2 table and waivers, K6 states and outcome fields; K3 adjudicator call, keep rule, identity, K5 calibration | PASS | — |
| `hazard-gate-filing-removed` | built here — E3 deletes _gate_hazard, _gate_unclosed and _file_hazard_gate with their tests | PASS | grep `def (_gate_hazard\|_gate_unclosed\|_file_hazard_gate)\(` expect absent in `fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py` |
| `identity-stats-key` | built here — auto_consolidation_initial_stats publishes auto_consolidation_adjudicator_identity on every Stage 1 cycle, staged or not (read by Γ) | PASS | grep `auto_consolidation_adjudicator_identity` expect present in `fused-memory/src/fused_memory` |
| `repair-keep-rule-refusal` | built here — a proposal that fails the keep rule after repair is refused with too_few_members_after_repair (A7) | PASS | grep `too_few_members_after_repair` expect present in `fused-memory/src/fused_memory` |
| `closure-failure-reverts` | built here — K4 step 8, the executor calls revert_auto_consolidation on a closure failure (its first production caller) | PASS | grep `(await\|return\|[=(:,]) *revert_auto_consolidation\(` expect present in `fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py` |
| `gate-filings-budget-retired` | built here — max_gate_filings_per_cycle leaves the schema and the reload allowlist once the executor no longer reads it (K7) | PASS | grep `^ +max_gate_filings_per_cycle *:\|'consolidation_auto\.max_gate_filings_per_cycle'` expect absent in `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/src/fused_memory/config/reload.py` |

Two present checks are scoped to the whole `fused-memory/src/fused_memory` tree, because the PRD
does not fix which module defines the new refusal code. γ lands after α and β, so either home
satisfies them. The revert check's pattern matches a call and not the `async def` line.

## η — Integration gate (signal leaf)

PRD §5's 21 rows run as e2e tests through the real ledger, the real executor and the real
calibration function, with a scripted adjudicator double. *Signal:* the e2e file passes and this
manifest maps A1–A21 to test names.

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `executor-flow` | producer: γ (upstream: η ← γ ← α, β) — the executor flow, records and stats the rows assert | PASS | — |
| `e2e-harness` | `grep:fused-memory/tests/test_auto_consolidation_e2e.py::build_world` exists: the real-ledger World harness the parent's B-rows use; A-rows join it | PASS | — |
| `a2-bannered-members-kept` | built here — A2, test named test_a2_<slug> | PASS | grep `def test_a2_` expect present in `fused-memory/tests/test_auto_consolidation_e2e.py` |
| `a12-adjudicator-down` | built here — A12, test named test_a12_<slug> | PASS | grep `def test_a12_` expect present in `fused-memory/tests/test_auto_consolidation_e2e.py` |
| `a14-calibration-fails-twice` | built here — A14, test named test_a14_<slug> | PASS | grep `def test_a14_` expect present in `fused-memory/tests/test_auto_consolidation_e2e.py` |
| `a21-eval-set-tampered` | built here — A21, test named test_a21_<slug> | PASS | grep `def test_a21_` expect present in `fused-memory/tests/test_auto_consolidation_e2e.py` |
| `a1-to-a21-mapped-to-tests` | built here — every row A1–A21 has a test named test_a<N>_<slug> in fused-memory/tests/test_auto_consolidation_e2e.py (the mapping table in the .md twin), and the file passes | PASS | manual — Twenty-one present-greps would be disproportionate and the four above sample the contract; η's reviewer confirms all twenty-one prefixes and the merge gate runs the file. |

The η mapping table below gives each row's test-name prefix. η picks the `<slug>`.

| Row | Scenario | Test |
|---|---|---|
| A1 | Clean pass | `test_a1_<slug>` |
| A2 | Bannered members kept | `test_a2_<slug>` |
| A3 | Bannered member excluded | `test_a3_<slug>` |
| A4 | Subset rescue refused | `test_a4_<slug>` |
| A5 | Keep 2 of 3 | `test_a5_<slug>` |
| A6 | Deleted member repaired | `test_a6_<slug>` |
| A7 | Too few after repair | `test_a7_<slug>` |
| A8 | Mixed category, twice, then fixed | `test_a8_<slug>` |
| A9 | Store inconsistent | `test_a9_<slug>` |
| A10 | Unresolved, then re-proposed under another topic and claim | `test_a10_<slug>` |
| A11 | Other-topic member | `test_a11_<slug>` |
| A12 | Adjudicator down | `test_a12_<slug>` |
| A13 | Calibration passes | `test_a13_<slug>` |
| A14 | Calibration fails twice | `test_a14_<slug>` |
| A15 | Calibration inconclusive | `test_a15_<slug>` |
| A16 | Dry-run | `test_a16_<slug>` |
| A17 | Closure failure | `test_a17_<slug>` |
| A18 | Execution cap | `test_a18_<slug>` |
| A19 | Stale facts | `test_a19_<slug>` |
| A20 | Old rows decode | `test_a20_<slug>` |
| A21 | Eval set tampered | `test_a21_<slug>` |

The parent's rows B4 and B8 are replaced by A2, A3 and A16. γ deletes the gate functions they
exercise, so γ retires those two tests. No check is bound to their names.

## ζ — Review surfaces (existing task 5242; not a block in this sidecar)

ζ is task 5242. Decompose amends it to this PRD's ζ scope: `scripts/report_auto_consolidations.py`
with adjudication columns, calibration records, `--revert` and `--json` carrying the rows' `shown`
text; the watcher playbook rows; `OPERATIONS.md`. Decompose also adds its dependency on η. It
keeps its binding as label ζ, `task_id: 5242`, in
`plans/memory-auto-consolidation-prd.capability-manifest.yaml`, so it has no block here: a second
block would bind one task in two sidecars. Here ζ appears only as a producer: of the report
`--json` that ρ1 and ρ3 read, and as an ordering dependency of χ.

## χ — Check surfaces (intermediate → Γ, φ, ρ2)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `real-ledger-harness` | producer: η (upstream: χ ← η) — the harness whose real ledgers χ's tests run the checks over; χ also depends on ζ (task 5242), for ordering only | PASS | — |
| `check-script` | built here — scripts/check_auto_consolidation.py, standard library only; each check exits 0 or 1 and prints one JSON line saying why | PASS | path expect present: `scripts/check_auto_consolidation.py` |
| `check-code-live` | built here — the code-live check Γ runs | PASS | grep `code-live` expect present in `scripts/check_auto_consolidation.py` |
| `check-live-clean` | built here — the live-clean check φ runs | PASS | grep `live-clean` expect present in `scripts/check_auto_consolidation.py` |
| `check-rejections` | built here — the rejections check ρ2 runs | PASS | grep `rejections` expect present in `scripts/check_auto_consolidation.py` |
| `review-thresholds-home` | built here — fused-memory/config/consolidation-review.yaml, the alarm thresholds' one home, with the §10 ρ2 filing values | PASS | path expect present: `fused-memory/config/consolidation-review.yaml` |
| `standard-library-only` | built here — the script imports nothing outside the standard library, so the gate's bare python3 subcheck runs it | PASS | manual — An expect-absent grep for third-party imports is vacuous while the file does not exist (polarity lint rejects it); χ's reviewer confirms the imports and Γ's subcheck exercises it under bare python3. |

For χ's design, not a binding: Γ reads dark_factory's *newest* Stage 1 `cycle_summary` row.
`ReconciliationHarness._ensure_stage1_cycle_summary` in
`fused-memory/src/fused_memory/reconciliation/harness.py` writes a backstop row when Stage 1
raised before its own write. Such a row can lack the executor's stats, so a crashed cycle as the
newest row would fail `code-live` even though the new code is running. χ decides whether
`code-live` looks past such rows.

## Γ — Gate: the adjudicator code is the code running (deterministic; delayed 86400 s)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `predicate-wrapper` | `grep:scripts/check_write_triage_readiness_gate.py::check_subcheck` wired: runs the --subcheck command shlex-split with no shell, exit 0 required; any failing subcheck makes the wrapper exit 1 | PASS | — |
| `stats-reach-stage1-summary` | `grep:fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py::run_auto_consolidation` `grep:fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py::write_stage1_cycle_summary` wired: the executor merges its stats into report.stats in a finally (unstaged projects included); Stage 1 writes them to the cycle_summary row with flag_type memory_consolidator | PASS | — |
| `identity-key-published` | producer: γ (upstream: Γ ← χ ← η ← γ) — auto_consolidation_adjudicator_identity in the Stage 1 stats | PASS | — |
| `code-live-check` | producer: χ (upstream: Γ ← χ) — scripts/check_auto_consolidation.py code-live | PASS | — |
| `gamma-adjudicator-code-running` | built here — the gate's own predicate, re-run so a cancelled Γ cannot release ε | PASS | script `scripts/check_write_triage_readiness_gate.py` (args below), timeout 120 s |

Script args, which are also the gate's `before_done` args:
`--gate Gamma --subcheck 'python3 scripts/check_auto_consolidation.py code-live'`
`--on-pass 'epsilon may stage dark_factory: fused-memory runs the adjudicator code'`
`--on-fail 'hold: newest dark_factory Stage 1 cycle_summary lacks auto_consolidation_adjudicator_identity; the new code is not running'`.

## ε — Switch on dark_factory (normal, simple)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `commented-config-block` | `grep:fused-memory/config/config.yaml` holds the consolidation_auto: block commented out (enabled_projects: [] , enabled: false), with the comment 'All fourteen leaves' β updates | PASS | — |
| `staging-leaves-reloadable` | `grep:fused-memory/src/fused_memory/config/schema.py::ConsolidationAutoConfig` `grep:fused-memory/src/fused_memory/config/reload.py::RELOADABLE_FIELDS` wired: enabled and enabled_projects are schema leaves on the hot-reload allowlist | PASS | — |
| `block-uncommented` | built here — consolidation_auto: is a live top-level key in fused-memory/config/config.yaml | PASS | grep `^consolidation_auto:` expect present in `fused-memory/config/config.yaml` |
| `dark-factory-staged` | built here — enabled_projects: ['dark_factory'] (flow-list form, as the PRD writes it) | PASS | grep `^ +enabled_projects:.*dark_factory` expect present in `fused-memory/config/config.yaml` |
| `writes-on` | built here — enabled: true in the same block | PASS | manual — An 'enabled true' line under one block is a multi-line YAML fact git grep cannot anchor (other blocks carry the same line); ε's review confirms it, φ confirms the project is no longer skipped, and ρ1 reads the executed rows. |

`writes-on` has no mechanical gate. φ's `live-clean` passes on a staged project in dry-run as
well as live. So until ρ1 runs, nothing mechanical distinguishes `enabled: true` from
`enabled: false`.

## φ — Gate: calibrated, live and clean (deterministic; delayed 259200 s; signal leaf)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `project-staged` | producer: ε (upstream: φ ← ε) — dark_factory in enabled_projects | PASS | — |
| `live-clean-check` | producer: χ (upstream: φ ← ε ← Γ ← χ) — scripts/check_auto_consolidation.py live-clean | PASS | — |
| `calibration-record-and-findings` | producer: β, γ (upstream: φ ← … ← η ← γ ← β) — the consolidation_adjudicator_calibration ledger row (K5) and the blocking finding records (K6) | PASS | — |
| `predicate-wrapper` | `grep:scripts/check_write_triage_readiness_gate.py::check_subcheck` wired: runs the --subcheck command, exit 0 required | PASS | — |
| `phi-calibrated-live-clean` | built here — the gate's own predicate, re-run so a cancelled φ cannot release ρ1 or ρ2 | PASS | script `scripts/check_write_triage_readiness_gate.py` (args below), timeout 120 s |

Script args, which are also the gate's `before_done` args:
`--gate phi --subcheck 'python3 scripts/check_auto_consolidation.py live-clean'`
`--on-pass 'rho1 and rho2 may proceed: dark_factory is staged, calibrated pass, no blocking adjudicator record open'`
`--on-fail 'hold: the live-clean JSON line names the failing condition; never cancel this gate'`.

## ρ1 — First review (normal; delayed 604800 s)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `report-json-with-shown-text` | producer: ζ (task 5242; upstream: ρ1 ← φ ← … ← χ ← ζ) — scripts/report_auto_consolidations.py --json carrying each row's shown text | PASS | — |
| `thresholds-file` | producer: χ (upstream: ρ1 ← φ ← … ← χ) — fused-memory/config/consolidation-review.yaml with the filing values | PASS | — |
| `review-log` | built here — docs/auto-consolidation-reviews.md with ρ1's dated entry | PASS | path expect present: `docs/auto-consolidation-reviews.md` |
| `thresholds-set-with-reasoning` | built here — consolidation-review.yaml's thresholds replaced by ρ1's values, reasoning in the dated entry | PASS | manual — The new values are ρ1's judgement and may equal the filing values; no pattern distinguishes them. The reviewer reads the diff and the dated entry. |
| `findings-and-follow-ups` | built here — findings written to fused-memory; a task or escalate_info for anything significant, including whether to stage reify | PASS | manual — Memory writes and filed follow-ups are not repository content. |

## ρ3 — Second review (normal; delayed 2592000 s)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `first-review-landed` | producer: ρ1 (upstream: ρ3 ← ρ1) — docs/auto-consolidation-reviews.md and the thresholds ρ1 set | PASS | — |
| `report-json-with-shown-text` | producer: ζ (task 5242; upstream through ρ1) — scripts/report_auto_consolidations.py --json | PASS | — |
| `second-dated-entry` | built here — a second dated entry appended to docs/auto-consolidation-reviews.md | PASS | manual — The file exists once ρ1 lands, so a path check is vacuous; the entry's date heading is not fixed by the PRD. The reviewer reads the appended entry. |
| `next-review-decision` | built here — the entry says whether further agent reviews are warranted and files the next one if so | PASS | manual — A judgement plus an optional filed task; nothing in the repository to check. |

## ρ2 — Monthly rejection alarm (recurring chain `consolidation-rejection-review`)

| Capability | Evidence | Verdict | Delivered check |
|---|---|---|---|
| `rejections-check-and-thresholds` | producer: χ (upstream: ρ2 ← χ) — scripts/check_auto_consolidation.py rejections and fused-memory/config/consolidation-review.yaml | PASS | — |
| `recurring-chain-carrier` | `grep:docs/task-authoring.md` §6.1 Recurring chains (metadata.recurrence), interval_secs 2592000, seed link dated 2027-01-20 | PASS | — |
| `predicate-wrapper` | `grep:scripts/check_write_triage_readiness_gate.py::check_subcheck` wired: runs the --subcheck command, exit 0 required | PASS | — |
| `monthly-link` | built here — each link runs the rejections predicate; a fail reaches Leo as an L2 escalation carrying the digest | PASS | manual — A recurring chain link delivers no repository artifact and has no dependents; its predicate is its before_done, re-run by every link. |
