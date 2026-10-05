# Merge lane: mutation-score baseline

**Task:** 5035, `plans/merge-lane-quality-prd.md` task ε (Phase 1, measurement).
Scope ruled by Leo on esc-5035-3 (2026-10-04, option D). The run was hand-carried
from an interactive session rather than dispatched.
**Taken against:** main `5d6ca7e880`, with mutmut 3.7.0 and CPython 3.13.9. Run
2026-10-04, 19:24–21:23 UTC, with 8 workers on the shared host (load average
70–410).
**No threshold.** This report measures and does not gate. Whether a mutation score
becomes a ratchet measure, and at what granularity, is PRD Open question 4.
**Follow-up:** task 6327 measures the modules that κ, μ and ο (5040, 5042, 5045)
extract from `merge_lane/worker.py`.

## Results

| Module | Mutants | Measured | Killed | Survived | Timeout | No tests | Score | Code-only score |
|---|---|---|---|---|---|---|---|---|
| `merge_lane/types.py` | 191 | all | 170 | 21 | 0 | 0 | **0.890** | **0.899** (170/189) |
| `merge_lane/gates.py` | 2,496 | random 300 | 175 | 125 | 0 | 0 | **0.622** (0.583–0.661) | **0.868** (145/167, 0.809–0.911) |
| `merge_lane/worker.py` | 11,768 | **INCOMPLETE** | — | — | — | — | — | — |

`worker.py` is INCOMPLETE because its mutated module never got past mutmut's
generation step. See the section below.

How to read it:

- A **mutant** is one small deliberate change to the code, such as `<` → `<=`,
  `and` → `or`, a constant changed, or an argument dropped.
- **Killed** means at least one test failed against the mutant.
- **Survived** means every test that reaches the mutated function still passed.
- **No tests** means no test in the selection reaches the function. Here it is
  zero: every measured mutant is reached by at least one test. No mutant was
  suspicious either.
- **Score** = (killed + timeout) / (killed + timeout + survived).
- **For `gates.py`, the score is a stratified estimate for the whole module, not
  the sample's raw 175/300 = 0.583.**
  - Mutants inside `logger.*(…)` calls are 956 of the module's 2,496 (38%). They
    are 133 of the 300 sampled (44%), so the sample over-represents them.
  - Log mutants are killed far less often (30/133) than code mutants (145/167).
  - Weighting the two strata by their exact population sizes gives 0.622. The
    interval is a stratified 95% interval.
- **Code-only score** drops two kinds of mutant:
  - mutants located inside a `logger.*(…)` call;
  - the two `types.py` mutants that edit an unreachable `assert_never` arm.
  The logger classification is mechanical: it uses the AST spans of logger calls in
  the measured source. `gates.py`'s code-only interval is a 95% Wilson interval for
  the 167 sampled code mutants. `types.py` was measured in full, so it has no
  interval.

**For Open question 4.**

- **The raw score is dominated by log mutants, but not only because log text goes
  unasserted.** The 133 sampled log mutants divide into three groups:
  - Most of the 30 that were killed broke the log call itself, with a format or
    argument mismatch. pytest's log capture (`_pytest/logging.py::LogCaptureHandler.handleError`)
    turns that into a test failure; few were killed by a test asserting log text.
  - Many of the 103 survivors are in log calls on error branches that no test
    reaches. For example, the logger mutants of `_path_existed_in_branch_history`
    survive in the same untested branch as its behavioural survivor.
  - The rest change only the text of log lines that do run.
- **Fencing the logger calls is cheap.** mutmut's
  `do_not_mutate_patterns = ['logger\.\w+']` (its own documented example) skips any
  expression that starts on a matching line, together with everything inside it,
  so multi-line calls are fenced too. Generating `gates.py` with that pattern gives
  exactly the 1,540 code mutants.
- **That also makes a full code-only `gates.py` run affordable.** Code mutants
  averaged 29.7 s each in this run, against 191.7 s for log mutants. A full
  code-only run is therefore about 1.6 h of mutant phase at 8 workers, against
  about 9–10 h for all 2,496.
- **So OQ4 has two real choices:** fence the logs and ratchet on code mutants, or
  keep them and ratchet on a number that partly measures whether error paths are
  reached at all.

## What survived (behavioural survivors)

These are the 41 survivors outside logger calls and unreachable arms: 22 in
`gates.py` and 19 in `types.py`. At least five are equivalent mutants, meaning no
test could tell them apart from the original:

- `gates._ls_tree_object_type__mutmut_10`: `[0]` of a `split` with `maxsplit=2`
  instead of `1`.
- `gates._run_unscoped_typechecks__mutmut_40`: appends an empty string that joins
  to the same detail.
- `types.TerminalOutcomeRetention.record_alias__mutmut_6`: `popitem(last=None)`
  against `last=False`.
- `types.InFlightMergeRegistry._release_if_current__mutmut_5`: the pop only runs
  after `self._slots.get(branch) is entry`, so the key is always present.
- `types.InFlightMergeRegistry.acquire__mutmut_18`: drops `generation=1`, which is
  the field's default.

The clearest genuine gaps:

- **`gates._commit_is_linear`** (5 survivors).
  - Changing `git rev-list --parents` to `--PARENTS` survives.
  - Replacing the whole git call with `None`, which falls into the `except` path,
    also survives.
  - Both `return False` → `return True` mutants survive, as does `<= 2` → `< 2`.
  - No test distinguishes this function's result.
- **`gates._path_existed_in_branch_history`**: `return False` → `return True`
  survives.
- **`gates._run_unscoped_typechecks`**: `continue` → `break` after a timed-out
  subproject survives. The checks themselves ran concurrently, so the change makes
  every later subproject's result go unread.
- **`types.TerminalOutcomeRetention`** (8 survivors).
  - `record` appending `None` instead of the record survives.
  - The alias-cap eviction in `record_alias` is unexercised: `* 4` → `/ 4`, `* 5`,
    `>` → `>=` and `last=True` all survive.
  - `forget` with `v == request_id` → `!=` survives.
- **`types.InFlightMergeRegistry`**.
  - In `eta_seconds`, `- elapsed` → `+ elapsed` and both boundary changes survive.
  - In `attach` and `detach`, `and` → `or` survives.
  - In `acquire`, the slot's `branch` can be dropped or set to `None`, and the first
    waiter's `request_id or ''` fallback can be changed, without a test noticing.
- **`types.MergeBounceRegistry.clear`**: `pop(branch, None)` → `pop(branch)`
  survives, so clearing a branch that never bounced is untested.

<details><summary>All 41 behavioural survivors</summary>

`gates.py`:
`_auto_chain_on_equivalence_blocked__mutmut_30`,
`_commit_is_linear__mutmut_1`, `_11`, `_27`, `_30`, `_32`,
`_disjoint_skip_blockers__mutmut_26`,
`_finalize_advanced_merge__mutmut_1`, `_79`,
`_ls_tree_object_type__mutmut_10`,
`_map_advance_failure__mutmut_62`, `_63`, `_167`,
`_path_existed_in_branch_history__mutmut_39`,
`_rebase_delta_touched_overlap__mutmut_4`,
`_rename_aware_compare_set__mutmut_5`, `_40`,
`_rename_aware_real_drops__mutmut_35`,
`_reverify_rebased_tree__mutmut_1`,
`_run_unscoped_typechecks__mutmut_23`, `_32`, `_40`.

`types.py`:
`InFlightMergeRegistry._release_if_current__mutmut_5`,
`InFlightMergeRegistry.acquire__mutmut_9`, `_16`, `_18`, `_31`,
`InFlightMergeRegistry.attach__mutmut_7`,
`InFlightMergeRegistry.detach__mutmut_9`,
`InFlightMergeRegistry.eta_seconds__mutmut_7`, `_8`, `_9`,
`MergeBounceRegistry.clear__mutmut_3`,
`TerminalOutcomeRetention.__init____mutmut_1`,
`TerminalOutcomeRetention.forget__mutmut_13`,
`TerminalOutcomeRetention.record__mutmut_4`,
`TerminalOutcomeRetention.record_alias__mutmut_3`, `_4`, `_5`, `_6`, `_7`.

To see any of them, regenerate (see Method) and run `mutmut show
orchestrator.merge_lane.<module>.<mangled name>`. Methods mangle as
`xǁClassǁmethod`, functions as `x_function`.

</details>

## Why `merge_lane/worker.py` is INCOMPLETE

mutmut 3 does not patch the file once per mutant. It writes **a complete copy of
the enclosing function for every mutant**, plus a trampoline that picks the copy at
run time. The mutated module therefore grows as Σ(function length × mutants in that
function), which is roughly the square of function size. After writing it, mutmut
`ast.parse`s the whole module before it records any mutant.

Measured with mutmut 3.7.0's own generator on `5d6ca7e880`:

| Module | Source lines | Mutants | Functions | Mutated module |
|---|---|---|---|---|
| `merge_lane/types.py` | 1,699 | 191 | 26 | 0.4 MB |
| `merge_lane/gates.py` | 3,493 | 2,496 | 34 | 19 MB |
| `merge_lane/worker.py` | 21,686 | 11,768 | 216 | **318 MB, 5.22M lines** |

The four worst-churn functions the task originally named account for 2,618 of
worker.py's mutants and 60% of its mutated lines (3.15M):

| Function | Mutants |
|---|---|
| `_run_post_merge_verify` | 898 |
| `SpeculativeMergeWorker._run_inflight_verify` | 738 |
| `SpeculativeMergeWorker._finalize_inflight` | 614 |
| `SpeculativeMergeWorker._verifier_loop` | 368 |

On 2026-10-02, the first 5035 architect ran the same generation on worker.py's
predecessor, `merge_queue.py` (21,707 lines, 11,616 mutants, 319 MB). mutmut's
`ast.parse` of the result had not finished after more than 70 minutes, and a
standalone compile of the file was killed after 43 minutes. No `.meta` was ever
written. That is the evidence behind "INCOMPLETE". The parse was not re-attempted
on `worker.py`, whose mutated module is the same size.

Choosing the four functions by mutant-name glob (`mutmut run
'orchestrator.merge_lane.worker.xǁSpeculativeMergeWorkerǁ_finalize_inflight__mutmut_*'`)
filters which mutants **run**. It does not filter which are **generated**: generation
is whole-file (`only_mutate` / `do_not_mutate` take file patterns) and happens first.
So the glob scope fails the same way.

Even a generated subset would be expensive. The first architect recorded which tests
reach each function, with their durations, at `e935a23116`. Each of the four
functions is reached by 400–608 tests, so a surviving mutant would cost 22–31
CPU-minutes.

κ and μ move these functions into new modules (`verify_dispatch.py`,
`landing.py`), and all three of κ/μ/ο decompose them into pieces of cognitive
complexity 40 or less. ο caps the residual `worker.py` at 1,500 lines.
- Splitting the functions *within* one file could shrink its mutated module by at
  most about 2.5×, because the rest of `worker.py` alone is 2.07M mutated lines.
- The bigger gain comes from the move, which divides that bulk across modules that
  can be measured one at a time.

Fencing logger calls (see Open question 4 above) shrinks every module further.
Task 6327 measures the split modules.

## Method

**Configuration** is `[tool.mutmut]` in `orchestrator/pyproject.toml`, committed
with this report. mutmut is pinned `>=3.7,<3.8` in the orchestrator dev group,
because 3.8.0 requires `click>=8.4.2` and `uv.lock` pins click 8.3.2. mutmut must run
from `orchestrator/`. It names a mutant after its file path with only a leading
`src.` stripped, while its stats collector records hits under the runtime module
name. From any other directory the two never match, and every mutant reports
"no tests".

**Test selection: the migrated suite.**
- **The rule.** Every lane-importing test file under `orchestrator/tests`
  (`scripts/merge_lane_metrics.py::imports_lane_module`: 234 files, 8,786 tests)
  that executes at least one function of `gates.py` or `types.py`. That leaves
  **141 files**, in which 3,192 tests reach a target function.
- **How it was found.** A pre-pass ran the 234 files under xdist with the plugin
  below loaded (`-p lane_hit_recorder`, with `LANE_HIT_DIR` set and the plugin's
  directory on `PYTHONPATH`).
- **Why the other 93 files are out.** mutmut only runs a mutant against tests its
  own stats pass saw reach the mutated function, so those files could not kill
  anything. Leaving them out changes no result; it only shortens the serial stats
  pass.
- **A snapshot, not a live list.** The committed 141-file list will fail to collect
  (pytest exit 4) once a listed file is renamed, so re-derive it with the plugin
  rather than editing it by hand.
- **Checked before the run.** All 5,097 tests in the 141 files passed inside
  mutmut's `mutants/` layout.

```python
# lane_hit_recorder.py: map each test to the gates.py/types.py functions it runs.
import json, os, sys
from collections import defaultdict

_TARGETS = ('/orchestrator/merge_lane/gates.py', '/orchestrator/merge_lane/types.py')
_TOOL = 4
_mon = sys.monitoring
_current: list[str | None] = [None]
_hits: dict[str, set[str]] = defaultdict(set)

def _on_start(code, offset):
    if not code.co_filename.endswith(_TARGETS):
        return _mon.DISABLE
    if _current[0] is not None:
        _hits[_current[0]].add(code.co_filename.rsplit('/', 1)[1] + '::' + code.co_qualname)

def pytest_configure(config):
    _mon.use_tool_id(_TOOL, 'lane_hit_recorder')
    _mon.register_callback(_TOOL, _mon.events.PY_START, _on_start)
    _mon.set_events(_TOOL, _mon.events.PY_START)

def pytest_runtest_logstart(nodeid, location):
    _current[0] = nodeid

def pytest_runtest_logfinish(nodeid, location):
    _current[0] = None

def pytest_unconfigure(config):
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    with open(os.path.join(os.environ['LANE_HIT_DIR'], f'{worker}.json'), 'w') as fh:
        json.dump({k: sorted(v) for k, v in _hits.items()}, fh)
```

**Sample.** `types.py` was measured in full (191 mutants). For `gates.py`, the full
population of 2,496 mutant names came from mutmut's own generator
(`mutmut.mutation.file_mutation.mutate_file_contents`, names via
`mutmut.utils.format_utils.get_mutant_name`). They were sorted, and 300 were drawn
uniformly without replacement with `random.Random(5035).sample(names, 300)`. The
sample covers 30 of the 34 mutated functions. This sampled run took 1 h 59 min in
all: about 45 min for the serial stats and clean-test passes, then about 75 min of
mutants at 0.11 mutants per second.

**Run.**

```
cd orchestrator
ln -s ../dark-factory-orchestrator.yaml dark-factory-orchestrator.yaml   # harness only,
ln -s ../df_pytest_isolation.py df_pytest_isolation.py                   # gitignored
env -u VIRTUAL_ENV uv run mutmut run --max-children 8 <the 491 mutant names>
```

The two symlinks exist because mutmut runs the tests from a copy of the tree in
`orchestrator/mutants/`. `orchestrator/tests/conftest.py::REPO_ROOT`
(`parents[2]`) then resolves to `orchestrator/` instead of the repo root. The
symlinks give the copied suite the operational yaml and the isolation plugin it
loads from there. Results are in `orchestrator/mutants/**/*.meta` (gitignored).
mutmut re-runs every mutant that is named explicitly. Only a bare `mutmut run`
skips mutants that already have a result, and a bare run covers the whole of both
files, not the sample.

**Classifying survivors.** Each measured mutant's `mutmut show` diff was located in
the measured source. It counts as a log mutant when its changed line falls inside
the AST span of a `logger.<level>(…)` call. An independent reviewer recomputed the
classification and agreed on all 300 sampled `gates.py` mutants.
