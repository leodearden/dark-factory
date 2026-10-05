# Capability manifest — write-triage link healing

**PRD:** `plans/write-triage-link-healing-prd.md` (landed `b3b4415805`, merge carrier 6180). **Machine twin:** `plans/write-triage-link-healing-prd.capability-manifest.yaml`. Both files are generated from one spec, so the tables below and the sidecar's `delivered_check`s are the same data. Every producer's `metadata.delivered_checks` was set from that spec at `submit_task`.
**Substrate verified** on main `0a9870b604` (2026-10-02), the decompose session's re-walk of PRD §6. Every cite resolved by symbol. The only drift since the PRD's `4dfdccc2ad` is docstring text in `server/tools.py` and `services/memory_service.py`.

**Polarity.** Every `grep`/`path` check below fails on that tree; `shared.delivered_check_polarity.lint_delivered_checks` reported no finding. The three gate `script` checks are not linted, and were run by hand instead:
- on the 2026-10-02 tree, Γ_A and θ fail with "report unreadable", and φ fails because `write_triage.enabled = False`;
- on synthetic reports meeting every bound, Γ_A and θ pass;
- on a copy of `config.yaml` with only the flag flipped, `scripts/check_write_triage_enabled.py` passes.

## Decompose-time contracts (not in the PRD's words; the gates depend on them)

1. **Report key contract (δ → Γ_A, θ).**
   - Every H3 figure is a plain JSON number at its own key, e.g. `selection.misfile_recall`.
   - Its Wilson bounds and counts sit beside it as `<figure>_ci95` `[lo, hi]`, `<figure>_num` and `<figure>_den`.
   - `population.n_pairs` is an integer.
   - Why: the gates read these with `scripts/check_write_triage_readiness_gate.py --require`, which fails closed on a missing key or a non-number.
   - δ carries a seam test that runs Γ_A's six requires over a `--arms fake` report.
2. **Summary key contract (ε → θ).**
   - `fused-memory/data/memory-evals/link-heal/summary.json` carries top-level numbers `runs_7d`, `writing_runs_7d`, `would_escape_7d`, `adjudication_failure_rate_7d`, `actions_planned_7d` and `skipped_cap_7d`.
   - θ requires the first, third and fourth.
3. **λ text keys (6151 → δ → ζ).**
   - Task 6151 (π) does not pin the two text field names in `write_triage_pairs_to_rate.jsonl`, and this batch may not edit it.
   - δ reads the names from 6151's committed file or script if either has landed, and records its assumption in the report's provenance.
   - ζ, a human-run leaf, verifies the names before scoring.
4. **Gate timeouts.**
   - φ: `--subcheck-timeout 60` < `before_done.timeout_secs 100` < the orchestrator's `delivered_checks.check_timeout_secs` (120).
   - Γ_A and θ: 60 s.
   - Each gate's `delivered_checks` entry re-runs its own `before_done` args, so a re-base edits both.
5. **`link_heal.writes` home (ε → ξ).** ε ships `link_heal:` with `  writes: false` in `fused-memory/config/config.yaml`; ξ turns it to `true`. ξ's check reads that line.

## Known risk (G6, provisional bound)

θ requires `population.n_pairs >= 300` on λ's triage pairs. The flip PRD's §11 Q6 expects "a few hundred" distinct pairs, so the bound may be unreachable. If so, θ's escalation re-bases per PRD D9.

## Bindings

## α — task 6181: Link-heal executor, ledger, caps, escapes and the link-heal- prefix

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `update-memory-patch-and-delete-routes` | capability→substrate (wired) — fused-memory/src/fused_memory/server/tools.py::update_memory (metadata_patch, metadata_delete_keys) → services/memory_service.py::MemoryService._apply_metadata_delta → backends/mem0_client.py::Mem0Backend.set_payload / delete_payload; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `mem0-update-prefix-allowlist` | capability→substrate (wired) — fused-memory/src/fused_memory/server/mem0_update_authz.py::resolve_mem0_update_authorization keyed on config/schema.py::Mem0UpdateConfig.metadata_patch_allowed_agent_prefixes (default ['recon-stage-', 'curator-']); the config.yaml block is commented out today; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `script-to-server-transport` | capability→substrate (wired) — shared/src/shared/mcp_post.py::post_mcp_tool_call; tools.py::_extract_causation carries _causation_id; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `child-kinds-and-contested-key` | capability→substrate (wired) — fused-memory/src/fused_memory/server/grouped_read.py::CHILD_KINDS (amendment, sighting), CONTESTED_METADATA_KEY (x_contested), _carve_outs_allow_suppression; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `raw-read-and-child-count` | capability→substrate (wired) — services/memory_service.py::MemoryService.get_memory_by_id, count_memories_by_metadata, _count_children, _apply_memory_metadata_validation (parent liveness, task 3197); verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `folded-escalation-filer` | capability→substrate (wired) — fused-memory/src/fused_memory/middleware/_folded_escalation.py::file_folded_escalation; fused-memory/tests/test_folded_escalation.py::TestNoTwoFilersShareAnAnchor; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `reloadable-config-registry` | capability→substrate (wired) — fused-memory/src/fused_memory/config/reload.py::RELOADABLE_FIELDS; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `hand-link-verdict-corpus` | capability→producer (upstream) — fused-memory/calibration/hand_link_verdicts.jsonl + .summary.json, committed with the PRD (8faca80c27, merge carrier 6180); commit_planning waits for it on main | PASS | — |
| `link-heal-executor-and-cli` | built by this leaf — maintenance/link_heal.py + scripts/link_heal.py (plan/apply/undo/status) | PASS | path present: `fused-memory/src/fused_memory/maintenance/link_heal.py`, `fused-memory/scripts/link_heal.py` |
| `link-heal-prefix-admitted` | built by this leaf — an uncommented config.yaml line admits 'link-heal-' beside recon-stage-/curator- | PASS | grep present `^[^#]*link-heal-` in `fused-memory/config/config.yaml` |
| `link-heal-knobs-reloadable` | built by this leaf — link_heal.* leaves registered in RELOADABLE_FIELDS | PASS | grep present `link_heal\.` in `fused-memory/src/fused_memory/config/reload.py` |
| `status-names-the-no-producer-state` | built by this leaf — `status` on an empty ledger prints "no link-heal run has written here" | PASS | manual — behavioural; asserted by α's own boundary test and observed live by β |

## β — task 6182: Cleanup sitting: plan from the corpus, Leo approves, apply

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `link-heal-cli` | capability→producer (upstream) — α (plan --from-corpus, apply --approved-plan-sha, status) | PASS | — |
| `hand-link-verdict-corpus` | capability→producer (upstream) — fused-memory/calibration/hand_link_verdicts.jsonl + .summary.json, committed with the PRD (8faca80c27, merge carrier 6180); commit_planning waits for it on main | PASS | — |
| `writing-corpus-run-in-ledger` | built by this leaf — a writing corpus run whose applied = approved plan − skipped_stale | PASS | manual — live-store ledger state after Leo's go-ahead; not on the git tree |

## γ — task 6183: add_memory kind guidance and typed inert-link ack

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `add-memory-tool` | capability→substrate (wired) — fused-memory/src/fused_memory/server/tools.py::add_memory; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `child-kinds` | capability→substrate (wired) — fused-memory/src/fused_memory/server/grouped_read.py::CHILD_KINDS; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `add-memory-typed-link-ack` | built by this leaf — ack field link: {status: 'child' \| 'inert', reason} | PASS | grep present `['"]inert['"]` in `fused-memory/src/fused_memory/server/` |

## δ — task 6184: Link adjudicator over the Claude CLI, its escapes, and the confusion-count eval

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `claude-cli-over-oauth` | capability→substrate (wired) — shared/src/shared/cli_invoke.py::invoke_with_cap_retry (output_schema, strips ANTHROPIC_API_KEY); shared/src/shared/neutral_cwd.py::neutral_cli_cwd; precedent fused-memory/src/fused_memory/middleware/path_scope_adjudicator.py::PathScopeAdjudicator; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `shard-failure-storm-counter` | capability→substrate (wired) — shared/src/shared/storm_counter.py::StormCounter; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `rater-brief` | capability→producer (upstream) — fused-memory/calibration/write_triage_rater_brief.md, committed with the PRD (8faca80c27, carrier 6180), byte-identical to the 2026-09-29 brief | PASS | — |
| `heal-executor` | capability→producer (upstream) — α (same files; δ adds the second plan source) | PASS | — |
| `lambda-pairs-shape` | capability→producer (upstream) — 6152 verdicts {entry_id, target_id, verdict, rater, batch}; texts from 6151's write_triage_pairs_to_rate.jsonl, whose text field names 6151 does not pin — δ reads them from 6151's committed file or script if landed and records its assumption; ζ (human) verifies before running | PASS | — |
| `adjudicator-and-eval-modules` | built by this leaf — maintenance/link_adjudicator.py + scripts/eval_link_adjudicator.py | PASS | path present: `fused-memory/src/fused_memory/maintenance/link_adjudicator.py`, `fused-memory/scripts/eval_link_adjudicator.py` |
| `adjudicate-links-entry-point` | built by this leaf — link_adjudicator.py::adjudicate_links(pairs, *, model, shard_size) | PASS | grep present `def adjudicate_links\(` in `fused-memory/src/fused_memory/maintenance/link_adjudicator.py` |
| `link-heal-cli-adjudicator-source` | built by this leaf — scripts/link_heal.py gains --from-adjudicator | PASS | grep present `from-adjudicator` in `fused-memory/scripts/link_heal.py` |
| `report-figures-readable-by-the-gate` | built by this leaf — each H3 figure is a plain number at its own key (Γ_A/θ --require fails closed on a non-number), bounds and counts beside it as <figure>_ci95 / _num / _den | PASS | manual — δ's boundary test runs the readiness-gate script with Γ_A's requires over a fake-arm report |

## δm — task 6185: Measure the adjudicator on the 359 rated hand links

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `eval-script` | capability→producer (upstream) — δ (scripts/eval_link_adjudicator.py --corpus --arms) | PASS | — |
| `hand-link-verdict-corpus` | capability→producer (upstream) — fused-memory/calibration/hand_link_verdicts.jsonl + .summary.json, committed with the PRD (8faca80c27, merge carrier 6180); commit_planning waits for it on main | PASS | — |
| `hand-link-adjudicator-report` | built by this leaf — committed link_adjudicator_report.json (+ .md) | PASS | path present: `fused-memory/calibration/link_adjudicator_report.json` |

## Γ_A — task 6186: Gate: adjudicator fit to drive the sweep

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `adjudicator-report` | capability→producer (upstream) — δm (committed report) with δ's key contract | PASS | — |
| `readiness-gate-predicate` | capability→substrate (wired) — scripts/check_write_triage_readiness_gate.py (--report/--require; exit code is the contract); verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `gamma-a-adjudicator-fit` | built by this leaf — the gate's own predicate, re-run so a cancelled Γ_A cannot release ε (D9) | PASS | script `scripts/check_write_triage_readiness_gate.py` (the gate's own before_done args, timeout 60 s) |

## φ — task 6187: Gate: the write-triage flip is live on main

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `the-flip` | capability→producer (upstream) — task 3169 sets write_triage.enabled: true in fused-memory/config/config.yaml | PASS | — |
| `flag-subcheck` | capability→producer (upstream) — scripts/check_write_triage_enabled.py, committed by this decompose session (lands before 3169 can close); exits non-zero on the 2026-10-02 tree (enabled: false), 0 on a flipped copy | PASS | — |
| `readiness-gate-predicate` | capability→substrate (wired) — scripts/check_write_triage_readiness_gate.py (--subcheck, --subcheck-timeout); verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `phi-write-triage-flip-live` | built by this leaf — the gate's own predicate, re-run so a cancelled 3169 or φ cannot release ε (D9) | PASS | script `scripts/check_write_triage_readiness_gate.py` (the gate's own before_done args, timeout 100 s) |

## ε — task 6188: Nightly link-heal sweep in report mode

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `executor-and-adjudicator` | capability→producer (upstream) — α and δ (upstream through Γ_A ← δm ← δ ← α) | PASS | — |
| `metric-series-writer` | capability→substrate (wired) — shared/src/shared/memory_eval_metrics.py; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `nightly-timer-convention` | capability→substrate (wired) — OPERATIONS.md §12 (wrapper .sh, oneshot .service, .timer Persistent=true, scripts/install-<job>-timer.sh); 03:00–05:30 taken, 02:00–02:30 free; verified on main 0a9870b604 (2026-10-02) | PASS | — |
| `sweep-timer-files` | built by this leaf — wrapper, service, timer and installer | PASS | path present: `scripts/fused-memory-link-heal.sh`, `scripts/fused-memory-link-heal.service`, `scripts/fused-memory-link-heal.timer`, `scripts/install-link-heal-timer.sh` |
| `operations-ladder-row` | built by this leaf — OPERATIONS.md §12 row at 02:15 | PASS | grep present `^\\| 02:15 \\|.*link-heal` in `OPERATIONS.md` |
| `summary-7d-keys` | built by this leaf — summary.json keys runs_7d … skipped_cap_7d that θ reads | PASS | grep present `runs_7d` in `fused-memory/src/fused_memory/maintenance/`, `fused-memory/scripts/link_heal.py` |

## η — task 6189: Install the sweep and observe a night

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `sweep-and-installer` | capability→producer (upstream) — ε | PASS | — |
| `timer-installed-and-ran` | built by this leaf — a non-writing adjudicator run from the timer with adjudicated > 0 | PASS | manual — live systemd + ledger state; θ's runs_7d >= 6 re-checks it a week later |

## ζ — task 6190: Measure the adjudicator on λ's triage pairs

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `eval-lambda-mode` | capability→producer (upstream) — δ (λ reader; see δ lambda-pairs-shape note) | PASS | — |
| `lambda-verdict-corpus` | capability→producer (upstream) — task 6152 (fused-memory/calibration/write_triage_pair_verdicts.jsonl), external to this batch, wired as ζ's dependency | PASS | — |
| `triage-adjudicator-report` | built by this leaf — committed link_adjudicator_report_triage.json | PASS | path present: `fused-memory/calibration/link_adjudicator_report_triage.json` |

## θ — task 6191: Gate: the sweep may write

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `triage-report` | capability→producer (upstream) — ζ | PASS | — |
| `sweep-summary` | capability→producer (upstream) — ε writes fused-memory/data/memory-evals/link-heal/summary.json (gitignored, main checkout) with runs_7d, would_escape_7d, adjudication_failure_rate_7d; η installs it | PASS | — |
| `theta-sweep-may-write` | built by this leaf — the gate's own predicate, re-run so a cancelled θ cannot release ξ (D9) | PASS | script `scripts/check_write_triage_readiness_gate.py` (the gate's own before_done args, timeout 60 s) |

## ξ — task 6192: Let the sweep write

| capability | binding | verdict | delivered_check |
|---|---|---|---|
| `writes-knob` | capability→producer (upstream) — ε ships link_heal.writes: false in fused-memory/config/config.yaml | PASS | — |
| `sweep-writes-on` | built by this leaf — link_heal.writes: true | PASS | grep present `^  writes: true` in `fused-memory/config/config.yaml` |
