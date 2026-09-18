# Capability manifest — inv12-exceptions-owned-or-ratified-prd

Binds each task's signal to substrate evidence (G3 + G6). Bindings cite symbols,
never lines; verified 2026-09-18 against main `890be05658` by two scout seats and
an adversarial premise seat (positive controls where a rejection is asserted).
Machine-readable twin: `inv12-exceptions-owned-or-ratified-prd.capability-manifest.yaml`.

## α — INV-12 in the pinned sites
- Family is derived, not enumerated:
  `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` (5 sites);
  headings in `docs/legibility/design-invariants.md` are the source of truth —
  wired (the test runs in `scripts/tests/`, a registered gate module). PASS.
- Trigger-shape span exists: `skills/prd/references/gates.md`
  `inv-trigger-shapes:begin/end` markers. PASS.
- `CONTRIBUTING.md` and `docs/code-quality.md` are pinned as absence / partial
  pairing — nothing to add. PASS.

## β — shared vocabulary + kernel
- `shared` is a workspace dependency of every package that will host a
  declarations test module: `orchestrator/pyproject.toml`,
  `fused-memory/pyproject.toml`, `dashboard/pyproject.toml`,
  `escalation/pyproject.toml` list `dark-factory-shared`; `scripts/tests` runs
  under `uv run --project shared`. PASS.
- No existing module to collide with: `shared/src/shared/governed_exceptions.py`
  and `shared/src/shared/ratchet.py` absent. PASS.

## γ1 — inline scanner and ratchet
- Marker grammar survives the consumers (rejection-check, positive controls,
  2026-09-18): ruff `--isolated --select E,F` — control `import re` fires F401,
  `# noqa: F401  # debt: ticket tkt_…` does not, no `Invalid # noqa directive`
  warning; pyright — control error fires, the four marked forms do not;
  coverage.py's default `exclude_lines` is a search, so trailing text is inert.
  PASS.
- Scan budget `≤10 s`: measured full `tokenize` pass 11.9 s over 1,918 files;
  prefilter to the 697 marker-bearing files ⇒ ≈4.3 s. `floor: 10 s > 4.3 s`. PASS.
- Consumer model inputs exist: `select = ["E","F","UP","B","SIM","I"]` verbatim
  in all 8 `pyproject.toml`; first-party pragma codes defined in
  `fused-memory/scripts/check_bare_magicmock_config.py`. PASS.
- `scripts/` → `scripts/` bare-name import precedent
  (`scripts/systemd_unit_parity.py` imported by three `check_*_unit_parity.py`);
  `scripts/tests/conftest.py` inserts `scripts/` on `sys.path`. PASS.

## γ2 — exception register
- Static AST discovery precedent: `scripts/merge_lane_metrics.py::patch_targets`,
  `shared/tests/silent_fallthrough_scan.py`. PASS.
- `producer:β` upstream for the vocabulary it reads. PASS (DAG-direction).

## δ — day-one rulings
- `execution_class: "decision"` → deterministic pure gate, born-at-L2
  (`docs/task-authoring.md` §4/§5). PASS.
- The pure gate closes on `resume`
  (`orchestrator/src/orchestrator/deterministic_runner.py` does not consult
  `delivered_checks`); what holds ζb is δ's `delivered_checks` grep in the
  scheduler's delivered-check cache (`Scheduler._compute_delivered_check_cache`).
  Same for κ1. PASS.

## ε1 / ε2 — declarations
- Every Appendix A part 1 container exists with the stated count (re-counted by
  AST 2026-09-18; `_BLESSED_METADATA_KEYS` 47, `ALLOWLIST_ENTRIES` 14, both
  `PINNED_SITES` 5). PASS.
- Live owners cited exist and are non-terminal as of 2026-09-18: 5149, 5034,
  4354, 4920, 5215 `pending`; 5578 `merge-deferred`. Each leaf re-checks at
  implementation. PASS.
- No production module or bare-`python3` script gains an import: declarations
  live in test modules (D4). Anti-regression of the two hosts that would break:
  `check_bare_magicmock_config.py`, `check_dashboard_unit_parity.py`. PASS.

## η — sweep
- Store resolver: `scripts/_task_db_scan.py::tasks_db_path` (a plain join on
  the project root) and `::connect_ro`; the sweep supplies the main checkout as
  `--project-root`, resolved from `git worktree list --porcelain`. PASS.
- Dead set: `shared/src/shared/task_statuses.py::TERMINAL` =
  `{DONE, CANCELLED}`; `TaskStatus.DEFERRED` composed explicitly. PASS.
- Filing client: `scripts/legibility/census_trigger.py::post_mcp_tool_call`;
  escalation poster precedent `scripts/legibility/census.py::_build_default_escalate_fn`. PASS.
- Tickets persist: `fused-memory/src/fused_memory/middleware/ticket_store.py` has
  no delete path. PASS.
- Tier-C keys accepted: `shared/src/shared/task_metadata.py` admits `x_`-prefixed
  keys. PASS.
- Unit-pair model: `scripts/memory-metadata-coverage-census.{service,timer}`,
  wrapper `.sh` (always exits 0), installer + `scripts/tests/test_install_memory_metadata_coverage_census_timer.py`. PASS.
- Rejection capability "curator is not the dedup mechanism" is avoided, not
  relied on: idempotence is the sweep's own key lookup. PASS.

## κ2 — sweep activation
- `execution_class: "operational"` → deterministic pure gate
  (`docs/task-authoring.md` §4). A `before_done` predicate was declined:
  `fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_before_done`
  rejects a script that does not exist at filing, and η produces it. The proof of
  activation is θ's real-report signal. PASS.

## θ — integration gate
- Every capability its signal needs is upstream: register report (γ2), seeded
  baseline (κ1), ruled dispositions (ζb), live sweep (η, κ2). PASS
  (DAG-direction).
- Planted-branch red is produced by γ1's guard test on the real tree — the
  rejection mechanism is γ1's scenario 1, observed to fire on fixtures before θ
  runs it live. PASS.
