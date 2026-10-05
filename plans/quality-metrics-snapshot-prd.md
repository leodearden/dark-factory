# Quality metrics snapshot: a whole-repo measurement report

**Status:** authored 2026-10-04 (seat S6 of the quality-skills alignment team), not yet
decomposed.
**Type:** a measurement instrument (report only) plus the extraction of the measures it
shares with the merge-lane ratchet.
**Approach:** B+H, small — one JSON contract read by four consumers (G5 "cross-PRD
consumers ≥ 2").
**Code anchors** verified against `92af0716e8` (worktree `quality-skills-alignment`,
2026-10-04). Cite-by-symbol; re-locate at implementation time.
**Ruling (Leo, 2026-10-04):** the whole-repo metrics snapshot is a REPORT — never a gate,
never a line-count ranking. First consumer `/review-all` phase 1; then `/review` and
`/hotspot-survey` for module selection.

## Goal

One command measures every workspace member's `src/` and `tests/` at one commit and writes
one JSON snapshot; a second invocation says what moved since the previous snapshot.
Observable when it lands:

- `uv run python scripts/quality_metrics_snapshot.py --run-id <run_id> --out <path>` on a
  clean checkout writes a snapshot whose `evidence.complete` is `true` and whose
  `evidence.measured_files` equals the tracked `.py` count under the members' `src/` and
  `tests/` (1,957 at `92af0716e8`).
- `--diff <previous snapshot>` prints, per module, the complexity pair as
  `docs/code-quality.md` §"What to measure" reads it, every file that crossed heuristic
  14's 1,500 or 2,000 raw-line marks (labelled "measure, not fix"), and the import and
  test-coupling measures that changed — in path order, never ranked.
- `/review-all` phase 1 builds its `metrics_summary` and `import_graph_path` arguments
  (`skills/review-all/references/orchestration.md`) from the snapshot instead of from
  ad-hoc `radon`/`complexipy` runs.
- For every merge-lane cluster path, the snapshot's `lines`, `prose_lines` and
  `cognitive_total` equal `scripts/merge_lane_metrics.py --json` at the same commit,
  because both read one set of measures.

## Background

`docs/code-quality.md` §"What to measure" names the instruments (`radon raw`,
`complexipy`, AST counts, the import graph, test patch targets) and §"Do not steer by"
excludes raw line count and average complexity as steering metrics; complexity is read as
a pair, maximum per function against module total. Heuristic 14 sets 1,500 raw lines as a
soft ceiling and 2,000 as an alarm that triggers measurement, not a target.
`docs/quality-findings-contract.md` §10 lets measurements appear in a report as context,
never as a ranking, target or severity input; §5 fixes the header a report pins.

Premises (G6): the first, third and fourth rows were measured 2026-09-30 and carried in
the alignment team's brief of 2026-10-04; the rest were re-measured for this PRD:

| Premise | Value |
|---|---|
| Merge-lane ratchet coverage, 2026-09-30 | 22 of 496 tracked `*/src/*.py` |
| Same, at `92af0716e8` | `merge_lane_metrics.py::CLUSTER_PATHS` resolves to 37 files, 35 of them among 531 member `src/` files |
| Whole-repo complexity, size or import-graph measurement | none exists |
| Mechanical instruments for heuristics 2, 4, 13 (symptoms), 14 and the Tests stance | only inside the merge-lane cluster |
| Domain at `92af0716e8` (members from root `pyproject.toml` `[tool.uv.workspace].members`) | 7 members; 531 `src` files / 442,992 lines; 1,426 `tests` files / 1,379,826 lines |
| Domain files at or over 1,500 / 2,000 raw lines | 279 / 192 |
| Functions complexipy reports | 7,006 in `src`, 60,966 in `tests` |
| `scripts/merge_lane_metrics.py` itself | 2,833 lines — over heuristic 14's alarm; it holds both the measures and the ratchet |

**Runtime, measured** (scratch timer calling `complexipy.file_complexity` plus the
`merge_lane_metrics.py` AST measures per file, serial, workspace venv Python 3.13 with
complexipy 6.2.0, on `92af0716e8`, load average 47–138 on 32 cores):

| Domain | Files | Lines | complexipy | AST measures | Wall |
|---|---|---|---|---|---|
| merge-lane cluster | 37 | 56,828 | 5.9 s | 1.2 s | 7.2 s |
| member `src/` | 531 | 442,992 | 21.0 s | 15.5 s | 37.0 s |
| member `tests/` | 1,426 | 1,379,826 | 48.9 s | 62.9 s | 112.9 s |

Extrapolating the cluster's complexipy cost per line to the whole domain would predict
~190 s; the direct measurement is 70 s, because complexipy's cost grows faster than
linearly in file size (the 21,686-line
`orchestrator/src/orchestrator/merge_lane/worker.py` alone takes 3.3–4.1 s) and the
cluster over-represents the largest files. The AST column is indicative only: the timer
ran tokenize plus four tree walks per file, and the instrument adds the import-graph and
patch-target walks. A whole run is therefore of the order of 2.5 minutes serial under
heavy load, without a cache.
complexipy 7.x would make it unusable (`merge_lane_metrics.py`'s `COMPLEXIPY_*` note:
247 s on one file), which the shared version pin already prevents.

Task **5414** (pending; ruling R5 of `plans/agent-capacity-study-2026-09-09.md`, which
lives untracked in the main checkout) Part 2 asks to "widen the 5021 ratchet enumeration
to the whole first-party tree in a report-only mode" for test-pinning counts. That is the
same enumeration and two of the same measures this snapshot takes; two copies would
violate heuristic 11 (SPOT).

## Sketch of approach

Two layers. Below: `scripts/source_measures.py`, the measures and the domain enumeration,
knowing nothing of the merge lane or of snapshots. Above it, two consumers that import it
downward: `scripts/merge_lane_metrics.py` (the cluster ratchet, a gate) and
`scripts/quality_metrics_snapshot.py` (the whole-repo report, never a gate). Task 5414
becomes a third consumer of the same enumeration and of the snapshot's numbers.

## Resolved design decisions

1. **Extract the measures before adding a consumer.** The parse/tokenize helpers,
   `file_size_measures*`, `function_local_imports*`, `reexport_names*`, the complexipy
   adapter and its version pin (`COMPLEXIPY_MIN`, `COMPLEXIPY_MAX_EXCLUSIVE`,
   `require_complexipy`, `file_cognitive_measures`), `maintainability_index`,
   `private_reads*`, `patch_targets*`, `src_module_name`, `tracked_python_files` and
   `MetricsError` move from `scripts/merge_lane_metrics.py` to
   `scripts/source_measures.py`. The cluster spec, lane-specific predicates
   (`LANE_PATCH_MODULES`, `lane_module_names`, `imports_lane_module*`,
   `test_file_measures*`, the alias-importer sweep), report, baseline, ledger, check and
   CLI stay. This is the heuristic-14 measurement the doc requires before a split: the
   partition is *measures* versus *cluster ratchet*; the only edge is
   `merge_lane_metrics → source_measures`, and `source_measures` imports nothing from
   its consumer, so the split is layering, not stitching (heuristic 13). No moved name
   stays importable from `merge_lane_metrics.py`: every importer
   (`scripts/check_staged_ratchet_raise.py`, `orchestrator/tests/test_merge_lane_ratchet.py`,
   `orchestrator/tests/test_merge_lane_ratchet_commit_gate.py`,
   `orchestrator/tests/test_merge_lane_alias_names.py`) is re-pointed, because a
   compatibility re-export is the heuristic-13 symptom the snapshot itself counts.
2. **`patch_targets` takes its module set as an argument.** Today it reads
   `LANE_PATCH_MODULES` from module scope. It becomes
   `patch_targets_in_tree(tree, modules)` returning `(module, leaf)` pairs: the ratchet
   passes `LANE_PATCH_MODULES` and keeps counting distinct leaves; the snapshot passes the
   members' top-level import packages and keeps the targets whose leaf has a
   single-underscore segment — the Tests stance's "patches a module's private names by
   dotted path". `private_reads_in_tree` is already receiver-agnostic and is used as is.
3. **One enumeration, two consumers** (task 5414). `source_measures.workspace_domain(root)`
   reads the member list from root `pyproject.toml` with `tomllib` (one home for the
   member list), lists `<member>/src/**.py` and `<member>/tests/**.py` from
   `git ls-files -s` (path and blob sha), and is the enumeration both the snapshot and
   5414's Part 2 use. The ratchet keeps its own fixed `CLUSTER_PATHS` manifest, which is a
   different question (a named cluster where a missing path is the finding). At
   decompose, 5414 gains a dependency on S2 and its Part 2 is re-pointed: pinning counts
   per package are aggregated from the snapshot's test-file records, and its extra
   measures (prose-constant assertions, duplicate bodies) iterate `workspace_domain`;
   no report-only mode is added to `merge_lane_metrics.py`. That re-pointing edits a
   task that carries a ruling — see Open questions for Leo.
4. **Measure the checked-out commit, refuse a dirty domain.** `as_of_sha` is
   `git rev-parse HEAD`; if `git status --porcelain` lists any path under the domain the
   run exits 2 naming the paths. Files are read from the work tree and measured with
   `complexipy.file_complexity` exactly as the ratchet does, so cluster numbers agree
   (Goal bullet 4). Blob shas are recorded per file, which lets `--diff` recognise a pure
   rename and leaves a content-addressed cache possible later.
5. **No cache in v1.** A whole run is ~2.5 minutes serial under heavy load (Background).
   That is within the few minutes the team brief allowed before a cache, and a cache is
   state with its own staleness risks. Revisit only if a snapshot's own `cost.wall_clock_s` exceeds 600 s;
   the remedy then is a cache keyed by `(blob sha, complexipy version, schema_version)`,
   never a "skip when nothing changed" rule (`plans/merge-lane-quality-prd.md`
   Correction 7's prohibition applies by analogy).
6. **The snapshot is a report: no verdicts, no ranking, no thresholds of its own.** It
   names exactly the doc's two heuristic-14 marks (1,500 and 2,000 raw lines), held as
   constants whose docstring points at heuristic 14 rather than restating it. It does not
   use the ratchet's `FILE_LINE_CEILING` or `NEW_FUNCTION_COGNITIVE_CEILING`: those are the
   merge-lane PRD's ceilings for its cluster, a different fact that happens to share a
   number. Every list is in path order; nothing is sorted by size or score. No average is
   computed anywhere. Exit codes are 0 (measured, or diff printed) and 2 (instrument
   failure) only — there is no exit 1, because there is nothing to fail.
7. **Whole-tree completeness polarity follows the ratchet's open sweep** (INV-11,
   `merge_lane_metrics.py` module docstring): a domain file that cannot be read, parsed or
   measured is skipped and named in `evidence.unreadable`, and `evidence.complete`
   becomes `false`; tool-level faults (git absent, complexipy missing or outside the pin,
   no workspace members, dirty domain) exit 2. `--diff` against an incomplete snapshot
   prints the unreadable paths first and reports their measures as unknown, never as
   zero.
8. **The script writes a file; the consuming skill commits it.** The script never
   touches the git index. Committing from a measurement script would put a long-running
   commit (and its pre-commit hook holding `.git/index.lock`) wherever the script happened
   to run, including the machine-operated main checkout, where `CLAUDE.md` §"Working in the
   main checkout" requires `git commit --only`, forbids committing while a merge verify is
   in flight, and an uncommitted file left behind can halt the merge queue. The attended
   skill knows which checkout it is in and the lane's state; the script does not. The
   suggested home is `plans/quality-metrics/<run_id>.json` beside the `/review-all`
   report (contract §5 report homes); the skill owns the choice.
9. **Header fields: the ones the script can know.** `run_id` (required argument — a
   snapshot is always some run's artefact), `as_of_sha`, `since` (the previous snapshot's
   `as_of_sha` when `--diff` is given, else `none`), `evidence` (members, domain and
   measured file counts, `unreadable`, `complete`), `cost.wall_clock_s`, and `params`
   (complexipy version, the two heuristic-14 marks, `schema_version`). `verification` and
   `inputs_consumed` belong to the consuming run's report, not to a measurement.
10. **The import graph is emitted in the shape `/review-all` already expects.**
    `skills/review-all/references/orchestration.md` declares `import_graph_path` as
    `{edges:[[from,to]], reach_back:[...], deferred:[...], cycles:[[...]]}`; the snapshot's
    `import_graph` section is exactly that, so phase 1 writes it to scratch unchanged.
    Edges are explicit first-party imports between `src` modules (absolute and resolved
    relative), excluding implicit parent-package imports. A *reach-back* is an import, in a
    module of package P, of a name defined in P's `__init__` or an ancestor's (an import of
    a sibling submodule is not one). *Deferred* is a function-local import (the existing
    `function_local_imports` measure, listed by site). *Cycles* are strongly connected
    components of size > 1 over module-level edges outside `if TYPE_CHECKING:`. Re-export
    names are reported per file with a `package_init` flag, so a façade `__init__` is
    distinguishable from a shim in another module; the snapshot does not judge which is
    which.
11. **Per-function detail for `src` only.** Every file record carries `cognitive_total`,
    `cognitive_max` and the qualname holding the max (the doc's pair). The full
    `path::qualname → score` map is stored for the 7,006 `src` functions, not for the
    60,966 test functions, which keeps a committed snapshot near a megabyte; test files
    keep their pair. One entry per line in sorted key order, as the ratchet's baseline
    does, so two snapshots diff legibly in git.

## Pre-conditions for activating

- complexipy `>=6.2,<7` and radon in the workspace venv (`orchestrator/pyproject.toml`
  dev group; present in the main checkout's venv: complexipy 6.2.0, radon 6.0.1).
- The `/review-all` skill text exists (in authoring 2026-10-04 by another seat); S2's
  skill-wiring part edits it after that text is committed.

## Cross-PRD relationship

| Other PRD / surface | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/merge-lane-quality-prd.md` (ratchet α = task 5021, landed) | shares measures | `scripts/source_measures.py`, imported by `scripts/merge_lane_metrics.py` | **this PRD** (S1) moves the measures; the merge-lane PRD keeps the ratchet, its baseline, ceilings and gate unchanged — S1's signal is a byte-identical `--json` report | S1 queued at decompose |
| task 5414 (R5 test census, pending) | consumes | `source_measures.workspace_domain` + the snapshot's test-file records | **this PRD** owns enumeration and measures; 5414 depends on S2 and stops adding a report-only mode to `merge_lane_metrics.py` | needs Leo's confirmation (Open questions) |
| `skills/review-all/**` (phase 1) | consumes | the snapshot file, its `import_graph` section, the `summary` output | **this PRD** owns the script and JSON contract; the skill owns invocation, the snapshot's committed home and how the summary enters `metrics_summary` | skill in authoring |
| `skills/review/**`, `skills/hotspot-survey/**` | consume | latest committed snapshot for module selection (contract §9) | the skills own the reading rule; this PRD owns the fields | skills in authoring |
| `plans/task-metadata-lookup-prd.md` | none | — | — | independent |

## Contract (B+H)

Command line (`scripts/quality_metrics_snapshot.py`):

```
--run-id RUN_ID --out PATH [--root DIR] [--diff PREVIOUS.json]   measure HEAD, write PATH, then diff if asked
--current SNAPSHOT.json --diff PREVIOUS.json                      diff two snapshots, measure nothing
--summary SNAPSHOT.json                                           per-member aggregate table, path order
exit 0 = done; exit 2 = instrument failure (MetricsError), message names the cause
```

Snapshot (`schema_version: 1`):

```
{ schema_version, instrument: "quality-metrics-snapshot", run_id, as_of_sha, since,
  evidence: {members: [...], domain_files, measured_files, unreadable: [...], complete},
  cost: {wall_clock_s},
  params: {complexipy_version, h14_soft_ceiling_lines: 1500, h14_alarm_lines: 2000},
  files: { "<path>": {member, kind: "src"|"tests", blob, lines, prose_lines, prose_ratio,
                      cognitive_total, cognitive_max, cognitive_max_function, functions,
                      # kind == "src":
                      function_local_imports, reexport_names, package_init,
                      reach_back_imports, fan_out, fan_in_src, fan_in_tests,
                      # kind == "tests":
                      private_patch_targets: [sorted dotted names], private_reads } },
  functions: { "<src path>::<qualname>": score },
  import_graph: { edges: [[from, to]], reach_back: [...], deferred: [...], cycles: [[...]] } }
```

`--diff` prints, in this order and in path order within each section: completeness of
both snapshots; added, removed and renamed (same blob) files; the complexity pair per
changed module, annotated with the doc's reading only where it applies ("max down, total
flat or down: complexity moved, per docs/code-quality.md §What to measure"; "total up:
complexity added"); heuristic-14 crossings of either mark in either direction, labelled
"ALARM — measure per heuristic 14, not a target"; import-graph changes (edges, reach-backs,
deferred imports, cycles added and removed); per test file, private patch targets added and
removed and the private-read delta. A measuring run without `--diff` ends by printing
"no previous snapshot given; since = none" (INV-13), so a first run never reads as a run in
which nothing moved; a `--diff` path that does not exist or does not parse is an exit 2
naming it.

## Boundary-test sketch

Fixture rows build a small real git repository with a root `pyproject.toml` naming two
members; no measure is patched (Tests stance).

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Ratchet unchanged by S1 | the real repository at one commit, measured by the pre-S1 and post-S1 script | `merge_lane_metrics.py --json` byte-identical; `--check` green against the unchanged committed baseline |
| 2 | Domain from members | fixture with 2 members, an untracked `.py`, a `scripts/x.py` | snapshot lists exactly the tracked member `src`/`tests` files |
| 3 | Dirty domain | modify one tracked member file | exit 2 naming the path; no file written |
| 4 | Unparseable member file | commit a file with a syntax error | exit 0; path in `evidence.unreadable`; `complete: false` |
| 5 | Complexity pair | commit 2 splits one function into two, same total | `--diff` shows max down, total flat, with the doc's reading |
| 6 | Alarm crossing | a file grows from 1,490 to 2,010 lines | `--diff` names both crossings with the "measure, not fix" label; nothing ranked |
| 7 | Rename | `git mv` only | reported as a rename, no measure deltas |
| 8 | Reach-back vs sibling | `pkg/a.py` imports a name from `pkg/__init__.py`; `pkg/b.py` imports submodule `pkg.c` | one reach-back (a), no reach-back for b |
| 9 | Cycle | `a` imports `b`, `b` imports `a` at module level; `c`↔`d` only under `TYPE_CHECKING` | `cycles == [["a","b"]]` |
| 10 | Private patch targets | test patches `pkg.mod._x` by string, `pkg.mod.public`, and `patch.object(mod, "_y")` | `private_patch_targets == ["pkg.mod._x", "pkg.mod._y"]` |
| 11 | No previous snapshot | measure without `--diff`; then with `--diff` naming a missing file | first: `since == "none"` and the "no previous snapshot given" line; second: exit 2 naming the path |
| 12 | Cluster agreement | real repo | for every cluster path, snapshot `lines`/`prose_lines`/`cognitive_total` equal the ratchet report's |

## Decomposition plan

Two leaves, both `task_kind='normal'`. S1 is quality-motivated: heuristic 14 (the host
file is over the alarm), heuristic 6 (*well-defined purpose* — measures versus ratchet),
heuristic 13 (the new file must make sense alone), heuristic 11 (one copy of each
measure). The S1 task text cites them by name and requires the heuristic-14 measurement
in the commit message.

- **S1 — Extract `scripts/source_measures.py`; generalise `patch_targets`; add
  `workspace_domain`.** [high; ~1,000–1,500 changed LOC, mostly moves; ~8 files]
  Files: `scripts/source_measures.py` (new), `scripts/merge_lane_metrics.py`,
  `scripts/check_staged_ratchet_raise.py`, `orchestrator/tests/test_merge_lane_ratchet.py`,
  `orchestrator/tests/test_merge_lane_ratchet_commit_gate.py`,
  `orchestrator/tests/test_merge_lane_alias_names.py`, a new test module for the measures
  under `tests/scripts/` (the measure unit tests move with the measures),
  `tests/scripts/test_pyright_unnecessary_ignore_opt_in.py` (its
  `merge_lane_metrics.py::_import_complexipy` pointer). **Signal:** boundary row 1 —
  `python scripts/merge_lane_metrics.py --json` is byte-identical before and after on the
  same tree and `pytest orchestrator/tests/test_merge_lane_ratchet.py` is green against
  the unchanged committed baseline; `git grep -n "^def file_size_measures\|^def patch_targets" scripts/merge_lane_metrics.py`
  is empty; `source_measures.py` imports nothing from `merge_lane_metrics`. Unlocks S2 and
  (re-pointed) 5414.
- **S2 — `scripts/quality_metrics_snapshot.py` (measure, `--diff`, `--summary`) and the
  `/review-all` phase-1 wiring.** [high; ~900–1,300 LOC with fixture tests; ~6 files]
  Files: `scripts/quality_metrics_snapshot.py` (new), its fixture-repo test module under
  `tests/scripts/`, `skills/review-all/references/orchestration.md` and the phase-1 text of
  `skills/review-all/SKILL.md` (invoke the script, write `import_graph` to scratch, build
  `metrics_summary` from `--summary`), and the module-selection sentence of
  `skills/review/SKILL.md` and `skills/hotspot-survey/SKILL.md` (read the latest committed
  snapshot). **Signal:** on main, `uv run python scripts/quality_metrics_snapshot.py
  --run-id review-all-dark_factory-<date> --out /tmp/s.json` exits 0 with
  `evidence.complete == true` and `measured_files` equal to the domain count; boundary rows
  2–12 green; the `/review-all` skill text names the script as phase 1's source. Consumer:
  the next attended `/review-all` run commits its first snapshot (contract §11 human gate).
  Depends on S1.

Decompose-time actions: `add_dependency(5414, S2)`; `update_task(5414)` re-pointing Part 2
(decision 3) — only after Leo confirms.

### Capability bindings (draft for the decompose manifest)

| Leaf | Capability | Evidence at `92af0716e8` |
|---|---|---|
| S1 | measures to move | `scripts/merge_lane_metrics.py::file_size_measures_in_tree`, `::function_local_imports_in_tree`, `::reexport_names_in_tree`, `::file_cognitive_measures`, `::require_complexipy`, `::maintainability_index`, `::patch_targets_in_tree`, `::private_reads_in_tree`, `::src_module_name`, `::tracked_python_files`, `::MetricsError` |
| S1 | importers to re-point | `scripts/check_staged_ratchet_raise.py` (uses `metrics.MetricsError`, `metrics.AppendOnlyViolation`, ledger and compare functions); three `orchestrator/tests/` modules importing `merge_lane_metrics` |
| S2 | complexipy per-file API, version pin | `complexipy.file_complexity` 6.2.0 in the workspace venv; `merge_lane_metrics.py::COMPLEXIPY_MIN` / `::COMPLEXIPY_MAX_EXCLUSIVE` |
| S2 | member list | root `pyproject.toml` `[tool.uv.workspace].members` (7 entries) |
| S2 | consumer's expected graph shape | `skills/review-all/references/orchestration.md` `import_graph_path` description (uncommitted in this worktree on 2026-10-04 — re-verify at decompose) |

G7 walk (advisory at author time): INV-11 `no-silent-fail-soft` — decision 7;
INV-13 `readers-prove-their-producer` — a run with no previous snapshot says so;
INV-5 `no-lockstep-duplication` and INV-9 `one-fact-one-home` — decisions 1, 3 and 6;
INV-10 `guards-exercise-behaviour` — boundary rows run the measures on a real fixture
repository. No waiver.

## Out of scope

- Any gate, ratchet, threshold, severity input or ranking built on the snapshot (ruling;
  contract §10).
- Changing the merge-lane ratchet's cluster, baseline, ceilings or runtime.
- Non-Python projects (reify's cargo-side measures are 5414's Part 2 business).
- Measuring `scripts/`, top-level `tests/` and `hooks/` (see Open questions).
- Mutation scores, coverage, churn and fix-rate history (other rows of the doc's table;
  `/hotspot-survey` owns history).
- Committing the snapshot (decision 8).

## Open questions

For Leo (gates needing a human):

1. **Re-pointing task 5414 Part 2.** R5 says "widen the 5021 ratchet enumeration to the
   whole tree". Decision 3 meets that by sharing the ratchet's measures and one
   whole-tree enumeration, and drops the "report-only mode in
   `scripts/merge_lane_metrics.py`" from 5414's text. Confirm before decompose edits a
   task that carries your ruling.
2. **Domain breadth.** The team brief scoped the domain to members' `src/` and `tests/`;
   `scripts/` (204 tracked `.py`, including `merge_lane_metrics.py` itself) and top-level
   `tests/` (94) are excluded. Include them as a pseudo-member `repo-scripts`?

Tactical (decide in the leaf):

3. Committed snapshot size (~1 MB estimated from 1,957 file records and 7,006 function
   entries): accept, or drop the per-function `src` map in favour of each file's pair.
4. The `--summary` columns, within the rule that every column is a count, total or
   pair, rows are members in name order, and nothing is averaged or ranked.
