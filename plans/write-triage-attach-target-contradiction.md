# The write_triage attach-target contradiction

**Filed by task 4822, from esc-4810-1. Measured at main `753ea8bd1b`.**

Task 4762 is gated by two independent encodings of one demand, and they
disagree with each other and with the code. This file is the durable record:
task-record state is not a repo artifact, so without it the correction would
live only in `metadata` and a description, which a curator combine has already
eaten once (`esc-markup-residue-1`, 2026-08-26 — the reason task 4762's own
details carry a RECOVERY PROVENANCE block).

The executable copy of the measurement is
`fused-memory/tests/server/test_write_triage_flip_gate_invariants.py`. Prefer
re-running it to re-deriving anything below.

## 1. What is wrong

**Half one — the delivered check gates 4762's dependents on a claim 4762 never
makes.** Task 4762's `metadata.delivered_checks` item 1 was
`{name: judge_verdict_carries_candidate_id, kind: grep, pattern: candidate_id,
expect: present, paths: [.../write_triage_judge.py]}` — option (a), encoded as
a git grep against main. Task 4762 chose option (b). Task 3169 depends on 4762,
so `scheduler.py::_deps_satisfied` withholds 3169 while the check fails, and at
`delivered_checks.grace_cycles` files a born-at-L2 `dependency_capability`
escalation and sets 3169 `blocked` — which `docs/task-authoring.md` §3.3 states
is NOT auto-re-pended. That fires strictly BEFORE 3169's own `before_done`
predicate ever runs, so it is a second, earlier gate on the same invariant.

**Half two — 4762's frozen plan step-2 is false.** It read, verbatim:
"restructure `build_judge_prompt` to render `candidates[0]` under a heading
naming it the attach target", and its step-1 pinned that structurally. The
contradiction is internal to 4762's own plan as filed: its step-13 rebuilds the
hoisted test so the evidence child is scored BELOW the cut — producing exactly
the slate on which step-2's claim is false.

## 2. The measurement

Six results at cosine `0.90 - i/100` (the cosine lives in
`metadata['store_score']`, which is what
`near_duplicate_guard::_cosine_of` reads — `relevance_score` is post-RRF and is
NOT the cosine), plus one `child-1` at `0.60` stamped
`{PARENT_ID_KEY: 'parent-1'}`. Then
`select_judge_candidates(results, 3, canonical_id='parent-1')`:

| observation | value |
| --- | --- |
| returned slate | `['m0', 'm1', 'child-1']` |
| `len(selected)` | `3` |
| target is first | `False` |
| target is last | `True` |
| `'parent-1'` in slate | `False` |

Control, `canonical_id='m0'`: `['m0', 'm1', 'm2']`, and the target IS first.

**Mechanism.** `write_triage_judge.py::select_judge_candidates` rescues a
winner that fell outside the top *n* by APPENDING it, not promoting it:
`selected = [*selected[: max(n - 1, 0)], winner]`. So the attach target lands
LAST, and the slate stays exactly *n* long because the rescue EVICTS rather
than widens — a caller cannot detect the rescue from the length. The
`PARENT_ID_KEY` fallback then keeps the evidence CHILD when the hoisted parent
never appeared as a result of its own, which is why the canonical id is absent
from the slate entirely.

The two paths DISAGREE. That is the finding: position is not a sound encoding
of the attach target. It is not that position is always wrong — the control
pins that it is sometimes right, which is exactly what makes it unreliable.

## 3. The corrected step-2 requirement

Supersedes 4762's frozen step-2. Paste-able verbatim; also carried in
`task 4762 metadata.x_4762_option_contradiction.required_step2_shape`.

> Pass the band's `decision.canonical_id` — already in scope at `judge_write`'s
> call site — into `build_judge_prompt`, and mark the attach target BY ID,
> wherever it sits in the slate. Do NOT identify it by position.
>
> The marked candidate is the one satisfying
> `r.id == canonical_id or (r.metadata or {}).get(PARENT_ID_KEY) == canonical_id`.
> A naive `r.id == canonical_id` marker marks NOTHING on the hoisted path,
> because the canonical id is absent from the slate (§2, and assertion (c) of
> `TestAttachTargetIsNotAlwaysFirst`).
>
> Also correct `build_judge_prompt`'s docstring sentence "Candidate ids ARE
> rendered, because the model must be able to say which candidate it means".
> Under option (b) the truth is that ids let the model tell candidates apart
> AND identify the one marked as the attach target.

Scope stays inside `write_triage_judge.py` plus
`fused-memory/tests/server/test_write_triage_judge.py`. It does NOT require
option (a)'s cross-module refactor of `judge_write`'s return type,
`write_triage.py`'s judge protocol, `BandDecision` or `tools.py` — that remains
task 4798 item 7. **Explicitly: do NOT close this as a duplicate of 4798 item
7.** They are different work.

## 4. The descriptor swap

`metadata.delivered_checks` on task 4762. `update_task`'s default metadata
merge overwrites a supplied key WHOLESALE, so items 2 and 4 — which are
CORRECT (`expect: absent` against the exact defective expressions) — must be
carried through byte-for-byte rather than dropped.

BEFORE:

```json
[
  {"name": "judge_verdict_carries_candidate_id", "kind": "grep", "pattern": "candidate_id", "expect": "present", "paths": ["fused-memory/src/fused_memory/server/write_triage_judge.py"]},
  {"name": "eval_outcome_order_is_deterministic", "kind": "grep", "pattern": "dict\\.fromkeys\\(TRIAGE_OUTCOMES|list\\(TRIAGE_OUTCOMES\\)", "expect": "absent", "paths": ["fused-memory/scripts/eval_write_triage_judge.py"]},
  {"name": "report_path_does_not_overwrite_its_own_json", "kind": "grep", "pattern": "report_path\\.with_suffix\\('\\.md'\\)", "expect": "absent", "paths": ["fused-memory/scripts/eval_write_triage_judge.py"]}
]
```

AFTER:

```json
[
  {"name": "write_triage_pre_flip_preconditions_on_main", "kind": "script", "script": "scripts/check_write_triage_flip_preconditions.sh", "args": [], "timeout_secs": 120},
  {"name": "eval_outcome_order_is_deterministic", "kind": "grep", "pattern": "dict\\.fromkeys\\(TRIAGE_OUTCOMES|list\\(TRIAGE_OUTCOMES\\)", "expect": "absent", "paths": ["fused-memory/scripts/eval_write_triage_judge.py"]},
  {"name": "report_path_does_not_overwrite_its_own_json", "kind": "grep", "pattern": "report_path\\.with_suffix\\('\\.md'\\)", "expect": "absent", "paths": ["fused-memory/scripts/eval_write_triage_judge.py"]}
]
```

**Why `kind='script'`, naming that script.** It is byte-for-byte the same
`script`/`args`/`timeout_secs` as task 3169's own `metadata.before_done`
predicate. The invariant then has ONE encoding referenced from TWO enforcement
points instead of two independent encodings that drifted into contradiction.
That drift is the actual disease here, and pointing both gates at the same
committed predicate is the only shape that cannot recur.

**Residual risk, stated plainly.** An ERRORED check — missing script,
non-executable script, git failure, timeout — is a fail-safe wait with NO
streak bump and NO escalation. That is a SILENT, indefinite hold on the
dependent, and it is the one new failure mode this swap introduces. It is
guarded by
`test_write_triage_flip_gate_invariants.py::TestFlipGateDeliveredCheckDescriptor`,
which pins that the descriptor validates as a `DeliveredCheckMeta` and that the
script it names exists and is executable.

## 5. The interlock

Task **4810** (in-progress) rewrites the predicate's item 1 to assert the
invariant behaviourally and mechanism-agnostically, so it passes on EITHER a
correct option (b) — verified by a swap test: the same slate rendered against
two different targets must produce different output — or option (a)
(`parse_judge_verdict` no longer returns a bare `str`). **Marking
`candidates[0]` alone will NOT satisfy it.** 4810 also fixes the secondary
defect that item 1 was a bare `grep -q 'candidate_id'` over module source,
satisfiable by prose that changes no behaviour — the source-text-grep meta-test
class this repo's norm says to delete.

Task **3169** is the operator flip gate, `blocked`, holding
`x_flip_hold_ruling_2026_08_27`. Task **4798 item 7** owns option (a) itself
and is not superseded by any of this.

**Ordering.** Both 4810 and 4822 must land before 4762 executes. Module locks
could not supply that: `lock_depth: 12` makes lock keys effectively
file-granular, and 4822's three files are disjoint from 4762's eight, so
nothing serializes them. The ordering is therefore encoded as real dependency
edges `4762 -> 4810` and `4762 -> 4822`, added before any other mutation.

## APPLIED

Filled in by task 4822 step-5, after the task-record surgery.
