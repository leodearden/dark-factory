# Capability Manifest — quality-metrics-snapshot-prd

Mechanizes G3 (the substrate exists and is **wired**) and G6 (the premise is valid) for each
task. There is one block per task, and each capability is bound to evidence on the tree.
Any **FAIL** binding blocks queueing.

**Domain flags:** tooling/measurement domain. There is no grammar or DSL, so
grammar-fixture checks are **N/A**. No numeric accuracy bound appears, so numeric-floor
checks are **N/A**. The two exactness premises (S1's byte-identical report and S1b's
unchanged census columns) are bound under G6 branch 2, each with its identity and its
configuration. Live checks: **capability→producer (wired)**, **DAG-direction**,
**exactness (G6-2)**, **rejection-mechanism**.

Evidence was re-verified at decompose (2026-10-05) on `2bdfa20ce2` (branch
`wip/quality-rulings-1005`, 297 commits after the PRD's `92af0716e8` anchor; main has
nothing this branch lacks). Cites are by symbol. Machine-readable twin:
`plans/quality-metrics-snapshot-prd.capability-manifest.yaml` (`task_id: null` until
`commit_planning` stamps it).

**Decompose-time correction.** Task 5414 was found done (`cdc15a6b56`, 2026-10-05). Its
Part 2 shipped as `scripts/suite_census_pinning.py`, with its own enumeration and with new
seams in `merge_lane_metrics.py`. The lead ruled (within Leo's 2026-10-05 ruling that there
be one enumeration) that the batch re-points the landed code: S1 covers imports and seams,
and the new leaf S1b covers the enumeration. No action is taken on task 5414. The PRD is
amended to match (header, decisions 1 and 3, Cross-PRD row, Decomposition plan).

---

## S1 — Extract `scripts/source_measures.py`; generalise `patch_targets`; add `workspace_domain`  *(intermediate → S1b, S2; also carries its own signal)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The measures to move exist and are wired into the ratchet | capability→producer (wired) | **confirmed** — `scripts/merge_lane_metrics.py::file_size_measures_in_tree`, `::function_local_imports_in_tree`, `::reexport_names_in_tree`, `::file_cognitive_measures`, `::require_complexipy`, `::COMPLEXIPY_MIN`, `::COMPLEXIPY_MAX_EXCLUSIVE`, `::maintainability_index`, `::patch_targets_in_tree`, `::private_reads_in_tree`, `::src_module_name`, `::tracked_python_files`, `::MetricsError`, all reached from `::build_report` / `::sweep_repo` | PASS |
| The seams 5414 added exist (move list widened) | capability→producer (wired) | **confirmed** — `::PatchCall`, `::patch_calls_in_tree`, `::tracked_files` (via `::_git_output`), added by `f44ac85b25` (task 5414) | PASS |
| Every importer is enumerated | capability→producer | **confirmed** by `git grep` for importers of `merge_lane_metrics`: `scripts/check_staged_ratchet_raise.py`, the three `orchestrator/tests/test_merge_lane_*` modules, `tests/scripts/test_pyright_unnecessary_ignore_opt_in.py` (pointer to `::_import_complexipy`), and the six 5414 files (`scripts/suite_census{,_evidence,_pinning,_rust}.py`, `scripts/tests/test_merge_lane_metrics_patch_calls.py`, `scripts/tests/test_suite_census_pinning.py`) | PASS |
| `workspace_domain` can read the member list | capability→producer | **confirmed** — root `pyproject.toml` `[tool.uv.workspace].members` = 7 entries (cockpit, dashboard, escalation, fused-memory, orchestrator, sampler, shared); `tomllib` is stdlib on 3.13 | PASS |
| `patch_targets_in_tree(tree, modules)` | capability→producer | producer = S1 (decision 2). Today `::patch_targets_in_tree` and `::_lane_module_aliases` read module-scope `::LANE_PATCH_MODULES` | PASS (built by S1) |
| `--json` is byte-identical before and after on the same tree | exactness (G6-2) | **identity:** the moved functions are pure and keep their bodies, so the same tree gives the same report. **configuration:** no file S1 edits appears in `orchestrator/tests/merge_lane_ratchet_baseline.json`, and none is lane-importing (`::imports_lane_module_in_tree`). `source_measures.py` names no `::ALIAS_MODULES` path, so the external-importers sweep is unchanged. The signal states this condition. Standing guard: `orchestrator/tests/test_merge_lane_ratchet.py::test_baseline_matches_a_fresh_measurement` | PASS |
| Sizing | overlay band | 14 declared files, which is not over the >15 review trigger; ~1,000–1,500 LOC, mostly moves. The behaviour change is split out as S1b so this leaf stays a pure move | PASS |

## S1b — Re-point the landed 5414 census enumeration onto `workspace_domain`  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The enumeration to replace exists on the production path | capability→producer (wired) | **confirmed** — `scripts/suite_census_pinning.py::measure_python_tree` (`tracked_files` + `::_in_test_tree`) and `::_FirstParty.of` (src paths + `::_SCRIPT_DIRS`), reached from `scripts/suite_census.py::_python_pinning` via `::ECOSYSTEMS['pytest']` | PASS |
| `workspace_domain` with per-entry import names | DAG-direction | producer S1 is **upstream** (S1b depends on S1) | PASS |
| The census's first-party-independent columns cannot move | exactness (G6-2) | **measured at `2bdfa20ce2`:** tracked `.py` files with a `tests` path segment = 1,656 = 1,443 member `tests` + 117 `scripts/tests` + 96 top-level `tests/`, with 0 outside the domain, and the grouping key (first path segment) equals the member names. **configuration:** first-party-dependent columns may move only for tests that import a `scripts/legibility` module by bare name (`import census` after a `sys.path` insert of `scripts/legibility`, as in `scripts/tests/test_legibility_census.py`). The signal requires that delta to be listed, not hidden | PASS |
| A tree with no workspace members is refused loudly | rejection-mechanism | `MetricsError` from `workspace_domain` (S1) is already caught by `scripts/suite_census.py::main`, which exits 2 | PASS |

## S2 — `scripts/quality_metrics_snapshot.py` and the `/review-all` phase-1 wiring  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Measures, enumeration and complexipy pin | DAG-direction | producer S1 is **upstream** (S2 depends on S1) | PASS |
| `complexipy.file_complexity` 6.2.x and radon available | capability→producer | **confirmed** — `/home/leo/src/dark-factory/.venv` has complexipy 6.2.0 (`file_complexity` present) and radon 6.0.1; `orchestrator/pyproject.toml` dev group | PASS |
| Consumer expects the `import_graph` shape | capability→producer (wired) | **confirmed** — `skills/review-all/references/orchestration.md` declares `import_graph_path` as `{edges:[[from,to]], reach_back, deferred, cycles}` (committed), and `skills/review-all/SKILL.md` §"Phase 1 — metrics snapshot" already names `scripts/quality_metrics_snapshot.py` and `plans/quality-metrics/<run_id>.json` | PASS |
| `measured_files` equals the domain count | exactness (G6-2) | **recounted** from `git ls-tree`: 2,255 at `92af0716e8` (531 + 1,426 + 94 + 110 + 94, matching the PRD) and 2,295 at `2bdfa20ce2`. The signal compares against the live count at the measured commit, never a constant | PASS |
| The snapshot emits no ranking, average or exit 1 | rejection-mechanism | producer = S2 (decision 6); boundary rows 6 and 11 observe it | PASS (built by S2) |

---

## G7 record

S1: INV-5 and INV-9 are what it removes. INV-10: the signal runs the instrument, not a text
match. INV-11: tool faults raise `MetricsError`. No waiver.
S1b: it removes the landed duplicate enumeration (INV-5, INV-9), and keeps
`PythonPinningCensus.unreadable` / `.complete` (INV-11). No waiver.
S2: INV-11 (decision 7), INV-13 ("no previous snapshot given; since = none"), INV-4 (the
unreadable list is carried in the result, and the reader is an attended human), INV-10
(boundary rows run on a real fixture repository). No waiver.

## Manifest verdict

**All bindings PASS (S1, S1b, S2).** The DAG is S1b←S1 and S2←S1. There are no external
or cross-PRD edges, and nothing is filed against task 5414. The sidecar loads through
`shared.capability_manifest.load_capability_manifest`. Its mechanical checks (S1: 4,
S1b: 2, S2: 1) are clean under `shared.delivered_check_polarity.lint_delivered_checks` at
both `main` and `HEAD`.
