# Capability manifest — S23 sampler collection groups and schema

PRD: `plans/review-all-s23-sampler-metric-groups-prd.md`. YAML twin:
`plans/review-all-s23-sampler-metric-groups-prd.capability-manifest.yaml`.

**Code anchors** verified against main `ae3047ec3a` (2026-10-10). Bindings cite symbols, not lines.

## α — Sampler store: leaf as a column, public schema with additive migration, one connection seam

| Capability | Binding | Verdict |
|---|---|---|
| SQLite `ALTER TABLE … ADD COLUMN` + DDL inside one transaction on the live corpus | measured on a `.backup` copy of `data/load-samples.db` (8,830,537 rows): ADD COLUMN 0.03 s, split UPDATE 22.5 s, CREATE INDEX 18.1 s, DROP INDEX 2.1 s | PASS |
| Feature-detect additive-migration idiom to mirror | `orchestrator/src/orchestrator/run_store.py::RunStore._migrate_task_results_block_context` (`PRAGMA table_info` + `ADD COLUMN`) | PASS |
| Oneshot tick may run ~45–65 s once without being killed | `dashboard/dark-factory-load-sampler.service` is `Type=oneshot` with no `TimeoutStartSec` | PASS |
| Readers keep working during the migration transaction | `shared/src/shared/sqlite_sync_base.py::apply_full_durability_pragmas_sync` sets `journal_mode=WAL`; live DB reports `wal` | PASS |
| Dashboard query keeps an index range on `(metric, ts)` after the index swap | `EXPLAIN QUERY PLAN` on the migrated copy: `SEARCH samples USING INDEX idx_samples_metric_ts_leaf (metric=? AND ts>?)`; anchor `MAX(ts)` covering | PASS |
| Calibration per-metric fetch stays index-ordered | migrated copy: `SEARCH samples USING INDEX idx_samples_metric_ts_leaf (metric=?)`, no temp B-tree; tick `COUNT(*)` covering | PASS |
| Freed index pages are reclaimed by an existing mechanism | `sampler/src/sampler/store.py::_VACUUM_MIN_RECLAIMABLE_FRACTION` = 0.10; probe file 1.012 → 1.257 GB ≈ 19% free | PASS |
| Dashboard env can import `sampler.store` | `uv run --no-sync --project dashboard python -c 'import sampler.store'` succeeds; α declares the dependency in `dashboard/pyproject.toml` | PASS |
| Calibration script tolerates an unmigrated corpus | produced by α itself (D6 `corpus_not_migrated`); boundary row 6 | PASS (α delivers) |
| Producer of live leaf rows under the real trigger (INV-13) | `dashboard/dark-factory-load-sampler.service` `ExecStart=… uv run --no-sync --project sampler python -m sampler`, from the main checkout, every 5 s | PASS |

## β — One MetricGroup table, its projections, and INV-14 `sampler-vocabulary-one-table`

| Capability | Binding | Verdict |
|---|---|---|
| `sampler.metrics` importable by the system python (the ε1/ε2 gate interpreter) | `/usr/bin/python3 -I` 3.12.3 with `sampler/src` and `shared/src` prepended imports `sampler.metrics`; `psutil` not loaded (lazy inside `collect_process_metrics`) | PASS |
| Something puts `sampler/src` and `shared/src` on the system python's `sys.path` for the script | `producer:task-6758` (S25 γ, `scripts/_checkout.py`; binds every workspace member's `src/` from the root pyproject, sampler included; its boundary test B1 asserts `sampler.__file__`). Upstream of β; edge wired at decompose 2026-10-11. Was producer-absent at authoring (`ae3047ec3a`) | PASS |
| `sampler.store` must NOT be imported by the script at runtime | measured: `/usr/bin/python3` import of `sampler.store` raises `ModuleNotFoundError: aiosqlite` via `shared.async_sqlite_base` — design avoids it (D6) | PASS |
| `proc.info` populated by `process_iter(attrs)` | psutil 7.2.2 in the sampler env (`uv run --no-sync --project sampler`); `psutil.process_iter` stores `as_dict(attrs)` as `proc.info` (finding `fk-be1621618dee` verification) | PASS |
| Gate-arm names exist to map onto | `shared/src/shared/psi.py::_ARMS` (owned by shared.psi; table maps onto it, cross-owner parity test kept) | PASS |
| Four-site INV registration guard exists | `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` | PASS |
| Rejection: one group's unadmitted key drops only that group | produced by β (D4); boundary row 9 observes the ERROR and the surviving rows | PASS (β delivers) |
| Rejection: the vocabulary guard flags a literal outside the table | produced by β; its own fixture module test observes the flag firing (INV-10) | PASS (β delivers) |
