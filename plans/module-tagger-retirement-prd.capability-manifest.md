# Capability manifest — module-tagger-retirement-prd

Human-readable twin of `module-tagger-retirement-prd.capability-manifest.yaml`.
Evidence anchors re-derived at HEAD `dc47ede940` (2026-08-20). All bindings
PASS; no blockers. Line anchors are authoring-time evidence only — the
sidecar's delivered_checks are pattern-anchored, never file:line.

## β — Delete the LLM module tagger end-to-end

- **tagger-sites-present-for-removal** — capability→producer (wired):
  `harness.py:2477` (startup call), `:2883` (`_tag_task_modules` def),
  `:2906` (skip gate), `:10080`, `:10135` (review-path calls);
  `module_tagger_prompt.py` exists; `cli.py` `--retag-modules`. PASS.
  Delivered when `_tag_task_modules` / `retag_modules` /
  `module_tagger_prompt` no longer appear under `orchestrator/`.
- **upstream-producer-3122** — DAG-direction: DF 3122 (in-progress) is
  upstream of β via a real dependency edge; β deletes the
  `files_tagged_empty` persistence 3122 lands. PASS.
- **synthetic-fallback-survives** — capability→producer (wired):
  `scheduler.py:8810` returns the `task-<id>` fallback from
  `_get_modules`; β must leave it intact (it is the post-retirement
  behavior for undeclared tasks). PASS.
- **vocab-retained-annotated** — `files_tagged_at` blessed Tier-A key at
  `shared/src/shared/task_metadata.py:903`; `'module_tagger'` in
  `KNOWN_ROLE_NAMES` at `:82`. Both are RETAINED and annotated historical
  (PRD decision 5); a delivered β still greps them present. PASS.
- **digest-historical-rows-tolerated** — `digest.py:467,516` NULL-task_id
  `module_tagger` handling stays (PRD decision 6). PASS.

## δ — Hard-gate simple-task submissions on declared files

- **interceptor-boundary-exists** — capability→producer (wired): the γ
  lock-charter guard returns `lock_charter_error` from
  `task_interceptor.py:5926`; δ's gate is a sibling at the same boundary.
  Prescribed error identifier: `simple_task_files_error` (mandated in the
  task description so the delivered_check is meaningful). PASS.
- **complexity-simple-field-real** — substrate:
  `metadata.complexity == "simple"` is read on the production path at
  `workflow.py:2601` and documented at `docs/task-authoring.md:490-513`
  and `plans/author-declared-complexity-prd.md`. PASS.
- **rejection-fires** — rejection-mechanism (G6 branch 4): δ's own tests
  must author a `complexity='simple'` fileless submission and observe
  `simple_task_files_error` fire, plus both allow-cases (simple+files,
  non-simple fileless). Bound as a grep on the test corpus naming the
  error id; rejection *quality* is judged by the tests themselves. PASS.
- **submit-prompts-updated** — roles that file simple tasks state the
  requirement (`agents/roles.py` submit stamps). Manual — prompt prose is
  not mechanically bindable. PASS.

## ε — Tagger-debris census across all six corpora

- **audit-lens-exists** — capability→producer (wired):
  `scripts/audit_wiped_metadata_files.py` models both plan-scope event
  sources; ε reuses (imports/extracts, per INV-5 — never duplicates) its
  lens. PASS.
- **census-script-delivered** — prescribed deliverable
  `scripts/census_tagger_debris.py` containing the `files_tagged_at`
  classification logic. PASS (delivered-state check).
- **artifact-committed** — prescribed deliverable
  `plans/module-tagger-debris-census.md` + `.json` (machine-readable,
  keyed for DF 3113 P4a / DF 3427). PASS (delivered-state check).
- **known-positives-present** — reify 6068/5602/5632 and DF 3113's own
  record must appear in the artifact (positive controls). Manual —
  external corpora are not greppable from this repo. PASS.
