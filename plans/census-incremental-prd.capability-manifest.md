# Capability Manifest — census-incremental-prd

Mechanizes G3 (substrate exists and is **wired**) and G6 (premise valid) per task
of `plans/census-incremental-prd.md`. One block per task, and each capability is
bound to evidence on main. Any **FAIL** binding blocks queueing.

**Domain flags:** tooling domain, with no grammar or DSL, so grammar-fixture checks
are **N/A**. There are no numeric accuracy bounds, so numeric-floor checks are **N/A**.
The numeric premises (the novelty replay and the index sizes) are bases for design
choices, and no leaf signal asserts them. The live checks are
**capability→producer (wired)**, **DAG-direction**, **field-population** and
**rejection-mechanism**.

Evidence was re-confirmed on 2026-10-05 against main `1e73930085`, with
`path::symbol` cites only. The PRD's own anchors were verified at `92af0716e8`. No
cited file has changed since then. Between the two commits, the only drift under
`scripts/legibility/` and `orchestrator/src/orchestrator/agents/code_quality.py` is one
new file that nothing cites, `scripts/legibility/check_transcript_check_liveness.sh`.
The machine-readable twin is
`plans/census-incremental-prd.capability-manifest.yaml`. Its `task_id`s are null until
`commit_planning` stamps them. Every mechanical `delivered_check` in it was linted
with `shared/src/shared/delivered_check_polarity.py::lint_delivered_checks` at main.
All 33 checks fail on main today and pass once their producer lands, with 0 findings.

External producers appear in two places. `find_tasks_by_metadata` comes from
`plans/task-metadata-lookup-prd.md` L1 and is absent from
`fused-memory/src/fused_memory/server/tools.py` today. `scripts/check_run_completion.py`
comes from `plans/completion-driven-triggers-prd.md` δ and is absent from the tree.
Both are **upstream hard dependencies** wired into L4 and L8b respectively. Neither
batch is filed yet: no task in the store carries either `prd_path`.

---

## L1 — run identity, processed-session ledger, resumable window  *(intermediate → L8a, L9, L3; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Host-local state dir for the ledger | capability→producer (wired) | **confirmed**: `scripts/legibility/trickle_state.py::project_state_dir(project_id)` | PASS |
| Coding success point to hook the ledger write | capability→producer (wired) | **confirmed**: `scripts/legibility/nightly.py::run_nightly`, `::select_scored_records` (production path of the `legibility-trickle@.service` unit, `uv run --no-sync --project shared python scripts/legibility/nightly.py run`) | PASS |
| Zero-signal gate | capability→producer (wired) | **confirmed**: `scripts/legibility/sampling.py::ScoredRecord.score` (= `counts.total_signal`) | PASS |
| Private scorer reached from two modules (heuristic 13 move) | capability→producer | **confirmed**: `sampling._score_and_find_first_turn` is called from `census.py::default_batch_source` and `nightly.py::select_scored_records` | PASS |
| Window, state and report seams to rework | capability→producer (wired) | **confirmed**: `census.py::_census_window_dates`, `::advance_census_state`, `::census_report_sections` (`SECTION_*` keys), `::run_census`, `::main`; `census_trigger.py::load_census_state`; `config.py::Census` | PASS |
| `ledger_retention_days` default 30 | G6 given | `census.py::_DEFAULT_CENSUS_LOOKBACK_DAYS` | PASS |
| Ledger rows written by the real trickle (signal) | field-population / INV-13 | producer = L1 itself, wired into `nightly.py::run_nightly`. A real nightly run is the only evidence, so the check is manual. Precondition: the trickle must be running, and its 09-29/30 account-access failures are out of scope (PRD §7) | PASS (production evidence pending) |

## L11 — report conformance checker  *(intermediate → G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Predicate script exists at G's filing | capability→producer | **lead action**: the decomposer commits the fail-loud stub (PRD §6; precedent `plans/confusion-reduction-prd.md` decision 10). L11 replaces it, and G ← L11 | PASS (stub must land before filing G) |
| Signal: exits 1 naming the 10-03 report's missing `## Method` | rejection-mechanism | **confirmed premise**: `plans/confusion-census-2026-10-03.md` has no `## Method` heading, and no `plans/confusion-census-*.json` exists. L11 itself builds the rejection (row 16) | PASS |

## L8a — relative novelty spike and watermark floor  *(intermediate → L2; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Trigger seams | capability→producer (wired) | **confirmed**: `census_trigger.py::evaluate`, `::CensusConfig.from_mapping` (per-field validate, never-raise), `::decide_for_project`, `::codebook_signal`; `config.py::NoveltySpike`; called by `nightly.py::evaluate_census_step` | PASS |
| Relative rule fires ~3/50 days | G6 basis | **re-derived by this seat** from the committed codebook at `92af0716e8`. The 72 h first_seen counts over 2026-08-15..10-03 are min 0, median 43.5, max 108; the absolute rule (≥4) fires 48/50 days and the relative rule (≥2× trailing-30-day median and ≥4) fires 3/50. This is the same result whether or not the baseline includes the day itself | PASS (provisional per PRD; the test pins the rule, not the count) |
| Floor "since session watermark" | field-population / DAG | `session_watermark` is written by L1 (upstream) through `advance_census_state`, but only after a **real census runs** post-L1. Until then the floor falls back to `last_census_at` and its line says `since last_census_at (no watermark yet)`. The PRD L8a signal and the payload were re-worded to that on 2026-10-05 (G6 resolution b) | PASS (signal re-worded) |

## L9 — invariant slugs and the Definition to the coder; slug check  *(intermediate → L3, L10; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `## Definition` slicer reachable from `scripts/legibility` | capability→producer (wired) | **confirmed**: `orchestrator/src/orchestrator/agents/code_quality.py::section`, `::NORMATIVE_DOC`. Importing through `orchestrator/src` on `sys.path` loads only `orchestrator`, `orchestrator.agents`, `orchestrator.agents.code_quality` and no third-party module (run under `/usr/bin/python3`). The Definition is **138 words** | PASS |
| One prompt for trickle and miner | capability→producer (wired) | **confirmed**: `coder.py::build_prompt` | PASS |
| Write-time slug enforcement point | capability→producer (wired) | **confirmed**: `codebook.py::validate_coding_record` | PASS |
| Heading reader to replace | capability→producer | **confirmed**: `scripts/tests/test_design_invariants_consistency.py::_HEADING_RE` (dark-factory `INV-<n>` form only today) | PASS |
| `invariant slugs: N valid, M rejected` journal line (signal) | field-population | producer = L9 in `nightly.py`, written by the next real trickle run | PASS |

## L2 — typed verdict, judged against the rendered definition  *(intermediate → L3; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Quality block renderer | capability→producer (wired) | **confirmed**: `code_quality.py::guidance`, wired in `orchestrator/src/orchestrator/agents/roles.py` (import at module top); 8,613 chars rendered | PASS |
| Verify seams | capability→producer (wired) | **confirmed**: `census.py::_verify_prompt`, `::_build_default_verify_fn`, `::_in_tree_remediation`, `::promote_candidate` | PASS |
| One severity definition | capability→producer | **confirmed**: `census.py::_VALID_ENTRY_SEVERITIES` duplicates the enum inline in `codebook.py::_ENTRY_SCHEMA`. L2 must read the codebook enum **without editing `codebook.py`**, because L9 and L3 own those edits and L2 is not serialised with L9 | PASS (constraint in payload) |
| Signal: non-`medium` severities with reasons | DAG-direction | observed only through G (upstream producers L2, L3) | PASS |

## L3 — finding keys on records and tickets  *(intermediate → L4, L6, L8b; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Interim key reference for parity | capability→producer | **confirmed**: `skills/hotspot-survey/scripts/findings_artefact.py::finding_key` and its `key` CLI subcommand. **No committed fixture exists**, although PRD L3 says "that script's fixture". Contract §2 requires "a committed fixture", so L3 commits one (`shared/tests/finding_key_parity_fixture.json` in `files`) | PASS (PRD wording drift noted) |
| Area vocabulary | capability→producer | **confirmed**: `review/briefing.yaml` `subprojects` = fused-memory, orchestrator, shared, dashboard, escalation, sampler. Each key names a workspace member directory (`pyproject.toml` `[tool.uv.workspace].members`, which also lists `cockpit`, a member the briefing omits) | PASS |
| Overlay rule for paths under no member | capability→producer | **confirmed**: `skills/review-all/references/project-overlay.md` Path → area: `scripts/legibility/**` → `shared` (and `.claude/skills/review-all/project.md`, CONFIRMED Leo 2026-10-05) | PASS |
| Schema seams | capability→producer (wired) | **confirmed**: `codebook.py::_validate_node`, `::STATUSES`, `::_ENTRY_SCHEMA`, `::_SIGHTING_SCHEMA`, `::validate`; `census.py::build_task_payloads`, `::promote_candidate`, `::retire_entry` | PASS |
| `x_finding_run` populated on filed tasks | field-population | producer = L3 on the production filing path `census.py::build_task_payloads`. The 23 historical `source=legibility_census` tasks prove curator-filed tasks keep their metadata (re-counted 2026-10-05: 23, 19 done / 3 pending / 1 cancelled) | PASS |

## L4 — dedup protocol and back-link  *(intermediate → L5; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `find_tasks_by_metadata` | capability→producer / DAG | **absent on main**. producer = `plans/task-metadata-lookup-prd.md` L1, an **upstream external dep** (that PRD's §Cross-PRD row names this PRD's L4 as consumer, with no inversion) | PASS (queued prerequisite) |
| Store failure shape | rejection-mechanism | Per the lookup contract, store and transport failures return an **`error` payload**, not a raise (that PRD's §Cross-PRD row for this PRD). PRD §4.8 row 8 says "raises", so the payload instructs the leaf to test the payload too | PASS (seam note) |
| MCP transport | capability→producer (wired) | **confirmed**: `census_trigger.py::post_mcp_tool_call` | PASS |
| Step 2/3 and resolve | capability→producer (wired) | **confirmed**: `fused-memory/src/fused_memory/server/tools.py::search_tasks`, `::get_task`, `::resolve_ticket` (statuses created / combined / failed / refused) | PASS |

## L5 — outcome feedback  *(intermediate → L6; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Status read path | capability→producer (wired) | **confirmed**: `tools.py::get_statuses`, reached through `census_trigger.py::default_status_fetcher` | PASS |
| `filed_tickets` | DAG-direction | producer L4, upstream | PASS |
| Re-verify for `fixed` | DAG-direction | verifier L2, upstream (INV-3) | PASS |

## L6 — pre-screen  *(intermediate → L7, L10; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Hook point | capability→producer (wired) | **confirmed**: `census.py::_novel_clusters`, `::_find_pending_candidate_id`, `::run_census` | PASS |
| Keys and back-links | DAG-direction | L3, L5 upstream | PASS |
| Other instruments' JSON records | INV-13 | none in contract shape may exist yet. `inputs_consumed: []` plus `extra.inputs_consumed_note` is the distinct state (C3) | PASS |

## L7 — similarity index and adjudication queue  *(intermediate → L10; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| HTTP client | capability→producer | **confirmed**: `shared/pyproject.toml` `httpx>=0.27` | PASS |
| API key reaches the unit | capability→producer (wired) | **confirmed**: `scripts/legibility-trickle@.service` has `EnvironmentFile=-/home/leo/src/dark-factory/.env`. The key's presence in that file is PRD §6's claim, **not re-read** by this seat (outside the worktree; secret) | PASS (operator confirms) |
| Embedding backlog ≈ 75 k tokens | G6 estimate | re-derived as 1,001 pending × ~291 chars ≈ 73 k tokens at 4 chars/token; reported as `embedding_calls` | PASS (estimate) |
| D3 premise: pending index ≈ 13.5× entry index | G6 basis | re-derived at `92af0716e8` with `coder.py::build_codebook_index`'s line form: entries 21,240 chars, pending 291,759 chars, **13.7×**. PRD: 21,464 / 289,778 = 13.5×. The ~1% gap is immaterial to D3 | PASS |
| Similarity thresholds 0.90 / 0.85 | G6 | **provisional** (D4); no signal asserts a precision, and every attach is audited | PASS (provisional) |

## L10 — structured synthesis  *(intermediate → L8b; G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Synthesis seams | capability→producer (wired) | **confirmed**: `census.py::_synthesis_prompt`, `::_build_default_synthesize_fn`, `::census_stage_specs` (`session_runner.py::CLASSIFIER`) | PASS |
| Corrections op | capability→producer (wired) | **confirmed**: `codebook.py::apply_coding_record` | PASS |
| Screened/attached sets, slug counts | DAG-direction | L6, L7, L9 upstream | PASS |

## L8b — completion condition; retire tasks-landed and the backstop  *(intermediate → G)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `scripts/check_run_completion.py` CLI (0 / 75 / 2) | capability→producer / DAG | **absent on main**. producer = `plans/completion-driven-triggers-prd.md` δ, an **upstream external dep**. Its CLI is `--run R --threshold T --project-root P` with T a fraction, and "the census converts its `completion_threshold_pct` itself". `empty_run` exits 2, which C7 (a) reads as N/A | PASS (queued prerequisite) |
| `x_finding_run` on the previous run's tasks | field-population | producer L3, upstream (`build_task_payloads`) | PASS |
| Retirement targets | capability→producer | **confirmed**: `config.py::Census.max_interval_days`, `.tasks_landed_threshold`, `.tasks_landed_min_days`; `census_trigger.py::compute_tasks_landed`; `docs/legibility/legibility.yaml` `census.max_interval_days: 10`. `config.Census` is `extra='allow'`, so a lingering yaml key is accepted structurally and `CensusConfig.from_mapping` owns the deprecation WARNING | PASS |

## G — live census conformance gate  *(leaf; C-as-integration-gate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Deterministic predicate milestone shape | capability→producer | **validated**: `shared/src/shared/task_metadata.py::Milestone` (`delayed`, `after_secs` 1209600) and `::BeforeDone` (`kind: predicate`, script, args, `timeout_secs` 120); `parse_metadata(direction='write', enforce=True)` clean | PASS |
| Script exists at filing | capability→producer | **not yet**: the stub must be committed on main first (lead action) | PASS (blocking lead action before filing G) |
| Every section produced upstream | DAG-direction | Method L1; Findings L3; Filed Tasks/Structural L4; Dispositions L5; Screened L6; Adjudication L7; Synthesis L10; trigger lines L8a/L8b; slugs L9; checker L11. All are upstream of G (G ← L8b, L9, L11, plus the chain) | PASS |
| Report written by the real producer | INV-13 | G reads the first automatic report after landing. On "no conforming report yet" the remedy is a forced census, then `resume` | PASS |
| 14-day delay | G6 | a fixed choice (PRD §10); the relative spike's replay rate is about one fire per 17 days | PASS (choice) |

---

## Manifest verdict

**All bindings PASS. No FAIL.** Two external upstreams are queued and wired as hard
dependencies (L4 ← lookup L1, L8b ← completion-triggers δ). One lead action blocks
filing **G** only: commit the `scripts/legibility/check_census_report.py` stub. G6
re-words L8a's signal so its floor line names its anchor, and records that L3 commits
its own parity fixture.
