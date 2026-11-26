# Capability manifest — metadata-modules-retirement-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). All substrate
evidence verified 2026-08-20 at HEAD `eba215060c` during the authoring
session's 4-agent census. Machine-readable twin:
`metadata-modules-retirement-prd.capability-manifest.yaml`.

## α — Sweep the metadata.modules template sites in skills/

- **template-sites-exist-at-head** → capability→producer (wired):
  `skills/prd/references/decompose-mode.md:93`,
  `skills/orchestrate/SKILL.md:157`, `skills/unblock/SKILL.md:269`,
  `skills/escalation-watcher/SKILL.md:1061`,
  `skills/review/references/phase3-triage.md:97` all carry the
  `"modules"` template key today (grep-verified). **PASS**
- **post-sweep-absence** → delivered_check `expect: absent` on the four
  files with no 3465-owned lines; `skills/escalation-watcher/SKILL.md`
  is excluded from the mechanical check (its :981-988 recipe lines are
  DF 3465's scope and persist until 3465 lands) — manual there. **PASS**
- **files-template-declared** → delivered_check `"files"` present in the
  decompose template. **PASS**

## β — Fix the migration copy branch and run it across the corpora

- **sanitize-predicate-substrate** → capability→producer (wired):
  `module_charter.sanitize_files_for_persist` exists and is the contract
  `orchestrator/tests/test_module_charter.py` asserts for every
  `metadata.files` write path (census-verified; also cited by DF 4507).
  **PASS**
- **directory-reject-gate-exists** (the collision β fixes) →
  `_reject_directory_locks_in_update_metadata` wired in
  `fused-memory/src/fused_memory/middleware/task_interceptor.py`
  (~:4450 region at `eba215060c`). **PASS**
- **branch-logic-tests** → delivered_check: `sanitize` present in
  `tests/scripts/test_migrate_metadata_modules_to_files.py` (today that
  file covers only the HTTP handshake — β adds the branch tests).
  **PASS** (producer is β itself)
- **fleet-run-executed** → `kind: manual`: the live run's evidence is
  the per-project before/after counts in β's PR plus the zero-action
  `--dry-run` re-run; operational, not greppable. **PASS** (recorded,
  excluded from the dispatch gate)

## ε — Boundary retirement (tuple drop + rejection + annotation)

- **attestation-tuple-currently-reads-modules** → capability→producer
  (wired): `task_interceptor.py:1666`
  `for key in ('files', 'files_to_modify', 'modules'):` feeds
  `path_scope_guard.local_attesting_signals` (census-verified, consumed
  at every submit). delivered_check `expect: absent` post-landing.
  **PASS**
- **rejection-mechanism** (G6 branch 4) → `retired_key_modules_error`
  present in both `fused-memory/src/fused_memory/` and
  `fused-memory/tests/` — the test authors a modules-carrying submission
  and observes the rejection fire (same mechanized shape as DF 4524's
  `simple_task_files_error`). **PASS** (producer is ε itself)
- **guard-precedent-substrate** → `lock_charter_error` construction
  pattern exists at the same boundary (`task_interceptor.py` ~:5909-5927
  region; DF 4524 is building the sibling — reuse, don't copy). **PASS**
- **blessing-annotation** → delivered_check: `retired 2026-08` present in
  `shared/src/shared/task_metadata.py` (blessing entry at :872 stays,
  gains the comment — the `origin_finding_id` / tagger-PRD decision-5
  shape). **PASS**
- **DAG-direction** → producers α, β upstream intra-batch; DF 4507
  (deletes the harness `modules` copy-forward so nothing legitimate
  bounces off the new rejection) wired upstream via `add_dependency`.
  **PASS**

No binding resolved to a blocking verdict.
