# Study sources for the 2026-10-04 dark-factory test census

The section "Reading against the 2026-09-10 studies" of
`plans/test-census-2026-10-04-dark-factory.md` compares the census with the
2026-09-10 verify-speed study. The study lives in the main checkout as
`plans/verify-speed-study-df-2026-09-10.md` and
`plans/verify-speed-study-df-2026-09-10/`, and git does not track either. Once
those files are gone, this file is what is left of them. It holds three things,
each copied verbatim:

1. The study passages that state every figure the census compares against.
2. The two study scripts, from `plans/verify-speed-study-df-2026-09-10/scratch-E/`,
   that the census re-ran to produce the middle column of its "Part 2 against
   the study's own scripts" table.
3. What those two scripts printed on the census tree.

Nothing here is a new measurement or a new reading.

## 1. Study passages

### A-baseline.md, §5 "Where the time actually goes — production junit reports", opening

Where the study's per-test times came from: one merge-role junit report per
module.

```markdown
## 5. Where the time actually goes — production junit reports

`verify.py::_prepare_junit_report_path` writes `<worktree>/.df-verify-junit/report.<module>.xml`
for **merge-role, breadth=full** runs. Two such worktrees survive and hold real per-test times:

- `.worktrees/_mainprobe-eee92be0/` — 2026-08-19, complete 7-module set, **arm A (`-n auto`=32)**
- `.worktrees/_merge-db00cdd4/` — 2026-09-10 19:16→, live merge, **arm C (`-n auto`=8)**, first
  4 modules complete

(`scripts` and `tests/scripts` produce no junit in either worktree.)
```

### A-baseline.md, §5b "Concentration — is the time in a few tests?"

The study's p50 per test and top-1% share, per package.

```markdown
### 5b. Concentration — is the time in a few tests?

Share of a module's total per-test time held by its slowest 1 % / 5 % / 10 % of tests:

| module | tests | p50 per test | top 1 % | top 5 % | top 10 % | reading |
|---|---|---|---|---|---|---|
| shared | 4364 | **1 ms** | **69.0 %** | 84.3 % | 91.1 % | pathological tail; ~44 tests are the suite |
| fused-memory (09-10) | 19485 | 8 ms | **54.5 %** | 71.8 % | 82.0 % | strongly concentrated |
| escalation | 1605 | 7 ms | 23.3 % | 46.4 % | 58.3 % | concentrated |
| dashboard | 2229 | 5 ms | 17.7 % | 47.9 % | 70.5 % | concentrated |
| cockpit | 346 | 1 ms | 14.5 % | 46.7 % | 70.4 % | concentrated |
| sampler | 52 | 30 ms | 35.9 % | 44.9 % | 61.0 % | trivial anyway |
| **orchestrator** | **17983** | **1443 ms** | **5.1 %** | **16.4 %** | **26.8 %** | **FLAT — no tail to cut** |

This is the single most important structural finding. Orchestrator's distribution:
p25 = 0.903 s, p50 = 1.443 s, p75 = 2.328 s, p90 = 3.448 s, p99 = 6.66 s, mean 1.84 s.
**67 % of orchestrator tests run in under 2 s and they carry 41 % of the total time.** Its
519 test files are flat too — the top 12 files are only **27 %** of the package.
```

### E-selection.md, §II.5 "Duplicates and near-duplicates"

The passage names its script by its session scratchpad path. The copy kept with
the study is the one in `scratch-E/`, reproduced in §2 below.

```markdown
## II.5 Duplicates and near-duplicates

Method: hash each test function's AST after stripping decorators and normalising the name.
"Structural" additionally erases identifier, attribute, constant and keyword identity, so
only control-flow/call shape survives (`scratchpad/wave1/E/static_dupes.py`).

Population: **44,700 test functions** in source (≈54,700 collected node ids after
parametrisation).

| class | groups | redundant functions | share |
|---|---|---|---|
| **identical** bodies (byte-equal AST) | 57 | **66** | 0.15% |
| **structurally identical** (names/constants normalised) | 2,589 | **5,020** | **11.23%** |
| `@pytest.mark.parametrize` decorators present | — | 2,178 | — |

Worst structural families are one-assertion variants that should be one `@parametrize`:
`orchestrator/tests/test_config.py` (48 in one family), `test_eval_configs.py` (45),
`test_workflow_signature_loop_guard.py` (37), `test_pytest_marker_deselection.py` (32),
`test_verify_classify.py` (28), `test_verify_env_transient.py` (25),
`dashboard/tests/test_tab_orchestrators.py` (23).

**Estimated time recovered: near zero.** Measured per-test durations: `shared` median
**0.0 ms**, p90 30 ms; `escalation` median 10 ms, p90 220 ms. 5,020 trivial tests at the
`shared` median contribute under a second of execution; the recoverable cost is collection
and reporting, not runtime. Collapsing them is a **legibility** change (heuristic 11 SPOT,
heuristic 3 orthogonal dimensions of variability), not a speed change. It should not be sold
as a verify-speed lever.
```

### E-selection.md, §II.6 "Implementation-pinning tests"

```markdown
## II.6 Implementation-pinning tests

`docs/code-quality.md`: tests that reach a module's internals are an interface-design smell.

| package | tests | touches a `_private` attribute | % | `patch()`es a `_private` symbol | % | files with module-level `from X import _y` |
|---|---|---|---|---|---|---|
| orchestrator | 17,266 | 8,073 | 46.8% | **602** | 3.5% | 242 |
| fused-memory | 15,704 | 5,948 | 37.9% | 47 | 0.3% | 81 |
| shared | 3,559 | 1,100 | 30.9% | 25 | 0.7% | 20 |
| escalation | 1,234 | 436 | 35.3% | 10 | 0.8% | 3 |
| dashboard | 2,043 | 328 | 16.1% | 12 | 0.6% | 16 |
| scripts+tests | 4,498 | 430 | 9.6% | 0 | 0.0% | 18 |
| cockpit / sampler | 396 | 34 | 8.6% | 0 | 0.0% | 1 |
| **TOTAL** | **44,700** | **16,349** | **36.6%** | **696** | **1.6%** | **381** |

The 36.6% column is an **upper bound** — the regex also catches a test calling its own
`self._helper()`. The trustworthy figure is the 696 that patch a private symbol of the module
under test.
```

### E-selection.md, §II.8 "Flakes", the flake_occurrence query

```markdown
The live DF ledger is in `data/orchestrator/runs.db`: `flake_occurrence` (append-only
evidence) and `flake_debt` (one row per governed test).

| query (last 30 days) | result |
|---|---|
| `flake_occurrence` rows | **299** |
| distinct `test_id` | **67** |
| verdict `fails_in_isolation` | **295** |
| verdict `unconfirmable` | 4 |
| verdict `passes_in_isolation` | **0** |
| `call_site` | 299/299 `merge_gate` |
| **`flake_debt` rows (all time)** | **0** |
```

## 2. Study scripts

Copied from `plans/verify-speed-study-df-2026-09-10/scratch-E/` as they stood on
2026-10-05:

| file | sha256 |
| --- | --- |
| `static_dupes.py` | `64a0863be64b6c6350959945e542b2543278b7e51455fc159d03650bf77f3cd0` |
| `static_rest.py` | `3ea275ec85e64de395e45cd4a83a0529517cbeb70a140d08ceeba2acfef93b79` |

To re-run them on another tree, save each block below under its file name. Then
follow "Commands used for the middle column" in the census report, copying these
saved files where it copies from `scratch-E/`. The `sed` there rewrites only the
`ROOT=` line. `static_dupes.py` takes the path of its JSON output as its only
argument; `static_rest.py` takes none and prints to stdout.

### static_dupes.py

```python
"""Part II.5 — identical/near-identical test bodies, modulo names."""
import ast, json, sys
from collections import defaultdict, Counter
from pathlib import Path
ROOT=Path('/home/leo/src/dark-factory')
PKGS=['orchestrator','fused-memory','shared','escalation','dashboard','sampler','cockpit']
def tfiles():
    for p in PKGS:
        d=ROOT/p/'tests'
        if d.is_dir(): yield from d.rglob('test_*.py')
    for d in (ROOT/'tests',ROOT/'scripts'/'tests'):
        if d.is_dir(): yield from d.rglob('test_*.py')

class Norm(ast.NodeTransformer):
    """Erase identifier/docstring/constant identity so only STRUCTURE remains."""
    def visit_Name(self,n): return ast.copy_location(ast.Name(id='_',ctx=n.ctx),n)
    def visit_arg(self,n): n.arg='_'; n.annotation=None; return n
    def visit_Attribute(self,n):
        self.generic_visit(n); n.attr='_'; return n
    def visit_Constant(self,n): return ast.copy_location(ast.Constant(value='_'),n)
    def visit_keyword(self,n):
        self.generic_visit(n); n.arg='_'; return n

def body_key(fn, strict):
    f=ast.parse(ast.unparse(fn))
    node=f.body[0]
    node.name='_'; node.decorator_list=[]
    if strict:
        return ast.dump(node)
    return ast.dump(Norm().visit(ast.parse(ast.unparse(node))))

exact=defaultdict(list); norm=defaultdict(list); total=0; params=[]
for f in tfiles():
    try: tree=ast.parse(f.read_text(errors='replace'))
    except SyntaxError: continue
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test'):
            total+=1
            rel=f'{f.relative_to(ROOT)}::{node.name}'
            src=ast.unparse(node)
            if len(src.splitlines())<3: pass
            try:
                exact[body_key(node,True)].append(rel)
                norm[body_key(node,False)].append(rel)
            except Exception: pass
            for d in node.decorator_list:
                s=ast.unparse(d)
                if 'parametrize' in s:
                    params.append((rel,s[:160]))
def rep(m,label):
    groups=[v for v in m.values() if len(v)>1]
    dup=sum(len(v)-1 for v in groups)
    print(f'{label}: {len(groups)} groups, {dup} redundant test functions ({dup/total:.2%} of {total})')
    for v in sorted(groups,key=len,reverse=True)[:8]:
        print(f'   x{len(v)}  {v[0]}')
        print(f'         + {v[1]}')
    return dup
print(f'total test functions: {total}')
d1=rep(exact,'IDENTICAL bodies (byte-equal AST after decorator strip)')
print()
d2=rep(norm,'STRUCTURALLY identical (names, attrs, constants normalised)')
print()
print(f'@pytest.mark.parametrize decorators: {len(params)}')
json.dump({'total':total,'exact':d1,'norm':d2},open(sys.argv[1],'w'))
```

### static_rest.py

```python
"""Part II.6/7/9/10 — implementation-pinning, sleeps, serial floors, test:source ratio."""
import ast, json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
ROOT=Path('/home/leo/src/dark-factory')
PKGS=['orchestrator','fused-memory','shared','escalation','dashboard','sampler','cockpit']
def tfiles():
    for p in PKGS:
        d=ROOT/p/'tests'
        if d.is_dir(): yield from d.rglob('test_*.py')
    for d in (ROOT/'tests',ROOT/'scripts'/'tests'):
        if d.is_dir(): yield from d.rglob('test_*.py')

# ---- 6. implementation-pinning -------------------------------------------------
priv_import=defaultdict(list); priv_patch=defaultdict(list); tests_total=Counter()
patch_re=re.compile(r"""patch(?:\.object)?\(\s*['"]?([A-Za-z0-9_.]*\._[A-Za-z0-9_]+)""")
for f in tfiles():
    pkg=f.relative_to(ROOT).parts[0]
    try: tree=ast.parse(f.read_text(errors='replace'))
    except SyntaxError: continue
    fns=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test')]
    tests_total[pkg]+=len(fns)
    for fn in fns:
        src=ast.unparse(fn)
        rel=f'{f.relative_to(ROOT)}::{fn.name}'
        hit_patch = bool(patch_re.search(src))
        # private symbol imported/used
        hit_import=False
        for n in ast.walk(fn):
            if isinstance(n,ast.ImportFrom) and n.module and not n.module.startswith('.'):
                if any(a.name.startswith('_') and not a.name.startswith('__') for a in n.names):
                    hit_import=True
            if isinstance(n,ast.Attribute) and n.attr.startswith('_') and not n.attr.startswith('__'):
                hit_import=True
        if hit_patch: priv_patch[pkg].append(rel)
        if hit_import: priv_import[pkg].append(rel)
# file-level: from X import _y at module scope
mod_priv=defaultdict(list)
for f in tfiles():
    pkg=f.relative_to(ROOT).parts[0]
    try: tree=ast.parse(f.read_text(errors='replace'))
    except SyntaxError: continue
    for n in tree.body:
        if isinstance(n,ast.ImportFrom) and n.module and any(
                a.name.startswith('_') and not a.name.startswith('__') for a in n.names):
            mod_priv[pkg].append(str(f.relative_to(ROOT)))
print('=== 6. implementation-pinning (private-symbol reach) ===')
print(f"{'pkg':14} {'tests':>7} {'privattr':>9} {'%':>6} {'patch _priv':>12} {'%':>6} {'files w/ module-level from X import _y':>10}")
for pkg in sorted(tests_total,key=lambda p:-tests_total[p]):
    t=tests_total[pkg]
    print(f'{pkg:14} {t:7} {len(priv_import[pkg]):9} {len(priv_import[pkg])/t:6.1%} '
          f'{len(priv_patch[pkg]):12} {len(priv_patch[pkg])/t:6.1%} {len(set(mod_priv[pkg])):10}')
tot=sum(tests_total.values()); pi=sum(len(v) for v in priv_import.values()); pp=sum(len(v) for v in priv_patch.values())
print(f'{"TOTAL":14} {tot:7} {pi:9} {pi/tot:6.1%} {pp:12} {pp/tot:6.1%}')

# ---- 7. sleeps / wall clock ---------------------------------------------------
print('\n=== 7. sleeps and wall-clock waits (>= 1 s) ===')
sleeps=[]; per_file=Counter()
SLEEP=re.compile(r'\b(?:time\.sleep|asyncio\.sleep|await\s+asyncio\.sleep)\(\s*([\d.]+)\s*\)')
WAITFOR=re.compile(r'wait_for\([^)]*timeout\s*=\s*([\d.]+)')
TMO=re.compile(r'\btimeout\s*=\s*([\d.]+)')
for f in tfiles():
    txt=f.read_text(errors='replace')
    try: tree=ast.parse(txt)
    except SyntaxError: continue
    for fn in [n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test')]:
        src=ast.unparse(fn)
        tot=0.0; kinds=[]
        for m in SLEEP.finditer(src):
            v=float(m.group(1)); tot+=v; kinds.append(('sleep',v))
        if tot>0:
            sleeps.append((tot,f'{f.relative_to(ROOT)}::{fn.name}',kinds))
            per_file[str(f.relative_to(ROOT))]+=tot
allsleep=sum(s[0] for s in sleeps)
print(f'test functions containing a literal sleep(): {len(sleeps)}')
print(f'GUARANTEED (unconditional-literal) sleep seconds summed over those tests: {allsleep:.1f}s')
print(f'  of which sleeps >= 1s per test: {sum(1 for s in sleeps if s[0]>=1.0)} tests, '
      f'{sum(s[0] for s in sleeps if s[0]>=1.0):.1f}s')
print('\n10 worst tests by literal sleep seconds:')
for tot,name,kinds in sorted(sleeps,reverse=True)[:10]:
    print(f'  {tot:7.1f}s  {name}   ({len(kinds)} sleep calls)')
n_wf=sum(len(WAITFOR.findall(f.read_text(errors="replace"))) for f in tfiles())
big_tmo=Counter()
for f in tfiles():
    for m in TMO.finditer(f.read_text(errors='replace')):
        if float(m.group(1))>=1.0: big_tmo[str(f.relative_to(ROOT))]+=1
print(f'\nwait_for(..., timeout=) call sites: {n_wf}')
print(f'timeout= literals >= 1 s in test files: {sum(big_tmo.values())} across {len(big_tmo)} files')
for k,v in big_tmo.most_common(8): print(f'  {v:4}  {k}')

# ---- 9. serial floors ----------------------------------------------------------
print('\n=== 9. serial floors ===')
xg=Counter(); ser=Counter()
for f in tfiles():
    t=f.read_text(errors='replace')
    n=len(re.findall(r'xdist_group', t))
    if n: xg[str(f.relative_to(ROOT))]=n
    n2=len(re.findall(r'mark\.serial|@pytest\.mark\.forked|--dist\s+loadgroup', t))
    if n2: ser[str(f.relative_to(ROOT))]=n2
print(f'files using xdist_group: {len(xg)} (total marks {sum(xg.values())})')
for k,v in xg.most_common(10): print(f'  {v:4}  {k}')

# ---- 10. test:source ratio -----------------------------------------------------
print('\n=== 10. test-to-source line ratio, per module-under-test ===')
def loc(p):
    try: return sum(1 for _ in p.open(errors='replace'))
    except Exception: return 0
rows=[]
for pkg in PKGS:
    td=ROOT/pkg/'tests'; sd=ROOT/pkg/'src'
    if not td.is_dir() or not sd.is_dir(): continue
    srcmap={}
    for s in sd.rglob('*.py'):
        srcmap[s.stem]=srcmap.get(s.stem,0)+loc(s)
    for f in td.rglob('test_*.py'):
        stem=f.stem[len('test_'):]
        sl=srcmap.get(stem)
        if sl and sl>50:
            rows.append((loc(f)/sl, str(f.relative_to(ROOT)), loc(f), stem, sl))
rows.sort(reverse=True)
print(f"{'ratio':>7} {'test LOC':>9} {'src LOC':>8}  test file  (module under test)")
for r,tf,tl,stem,sl in rows[:12]:
    print(f'{r:7.2f} {tl:9} {sl:8}  {tf}  ({stem})')
tl=sum(loc(f) for f in tfiles())
sl=sum(loc(s) for p in PKGS for s in (ROOT/p/'src').rglob('*.py') if (ROOT/p/'src').is_dir())
print(f'\nworkspace totals: test LOC {tl}, src LOC {sl}, ratio {tl/sl:.2f}:1')
```

## 3. Output on the census tree

Both scripts ran on 2026-10-05 against the census's measured tree, `6d2dbbfe1f`,
with `ROOT` re-pointed at it.

### static_dupes.py

Standard output:

```text
total test functions: 55576
IDENTICAL bodies (byte-equal AST after decorator strip): 70 groups, 90 redundant test functions (0.16% of 55576)
   x7  scripts/tests/test_install_transcript_check_timer.py::test_script_is_executable
         + scripts/tests/test_check_transcript_check_liveness.py::test_script_is_executable
   x5  fused-memory/tests/test_canonical_labels.py::test_returns_none
         + fused-memory/tests/test_canonical_labels.py::test_local_node_name_with_a_unicode_digit_is_not_a_task_label
   x4  fused-memory/tests/test_cgl_eta_auto_apply_wrapper.py::test_wrapper_is_executable
         + scripts/tests/test_flag_marker_sweep_wrapper.py::test_wrapper_is_executable
   x4  fused-memory/tests/test_completion_claim_gate.py::test_negated_and_aspirational_framing_yields_nothing
         + fused-memory/tests/test_completion_claim_gate.py::test_non_refs_yield_nothing
   x3  orchestrator/tests/test_eval_driver.py::test_teardown_survives_a_failing_cell
         + orchestrator/tests/test_eval_driver.py::test_teardown_survives_a_failing_cell
   x3  orchestrator/tests/test_eval_driver.py::test_teardown_survives_cancellation
         + orchestrator/tests/test_eval_driver.py::test_teardown_survives_cancellation
   x3  orchestrator/tests/test_eval_driver.py::test_a_degraded_campaign_stays_ungated
         + orchestrator/tests/test_eval_driver.py::test_a_degraded_campaign_stays_ungated
   x3  orchestrator/tests/test_streaks.py::test_clear_missing_key_is_a_noop
         + orchestrator/tests/test_streaks.py::test_clear_missing_key_is_a_noop_for_cause_variant

STRUCTURALLY identical (names, attrs, constants normalised): 2974 groups, 5861 redundant test functions (10.55% of 55576)
   x57  orchestrator/tests/test_config.py::test_park_stop_parked_threshold_ge_1_rejects_zero
         + orchestrator/tests/test_config.py::test_park_stop_parked_window_hours_gt_0_rejects_zero
   x50  orchestrator/tests/test_eval_configs.py::test_returns_none_for_unknown
         + orchestrator/tests/test_eval_configs.py::test_qwen25_32b_not_found_by_name_lookup
   x41  orchestrator/tests/test_briefing.py::test_none_renders_nothing
         + orchestrator/tests/test_briefing.py::test_string_value_renders_nothing
   x32  orchestrator/tests/test_pytest_marker_deselection.py::test_none_source_is_none
         + orchestrator/tests/test_pytest_marker_deselection.py::test_syntax_error_is_none_not_a_raise
   x28  orchestrator/tests/test_verify_classify.py::test_compile_error_rustc_code
         + orchestrator/tests/test_verify_classify.py::test_compile_error_compile_error_string
   x27  orchestrator/tests/test_retry_cap.py::test_true_for_5xx
         + orchestrator/tests/test_retry_cap.py::test_false_for_non_transient
   x25  orchestrator/tests/test_fm_retry.py::test_window_is_120_seconds
         + orchestrator/tests/test_fm_retry.py::test_sentinel_constant_matches_documented_d8_value
   x25  orchestrator/tests/test_verify_env_transient.py::test_xdist_usage_error_is_env_transient
         + orchestrator/tests/test_verify_env_transient.py::test_no_module_named_pip_is_env_transient

@pytest.mark.parametrize decorators: 3484
```

The JSON file it wrote:

```json
{"total": 55576, "exact": 90, "norm": 5861}
```

### static_rest.py

Standard output and standard error, as captured. The census uses section 6
only. The script stops in section 7 on a `sleep(...)` literal that it cannot
convert to a number.

```text
=== 6. implementation-pinning (private-symbol reach) ===
pkg              tests  privattr      %  patch _priv      % files w/ module-level from X import _y
orchestrator     19825      8800  44.4%          537   2.7%        266
fused-memory     19696      7209  36.6%           48   0.2%        107
shared            4642      1281  27.6%           25   0.5%         21
scripts           4299       275   6.4%            4   0.1%          7
dashboard         2733       383  14.0%           17   0.6%         20
tests             2059       289  14.0%            0   0.0%         12
escalation        1691       593  35.1%           12   0.7%          6
cockpit            494        73  14.8%            0   0.0%          1
sampler            137        34  24.8%            0   0.0%          0
TOTAL            55576     18937  34.1%          643   1.2%

=== 7. sleeps and wall-clock waits (>= 1 s) ===
Traceback (most recent call last):
  File "/tmp/5414-study/static_rest.py", line 70, in <module>
    v=float(m.group(1)); tot+=v; kinds.append(('sleep',v))
ValueError: could not convert string to float: '...'
```
