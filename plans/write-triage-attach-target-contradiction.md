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

Task 4822, 2026-08-28, at main `753ea8bd1b`. Observed return values, not
intended ones. **All of 5a-5e are now APPLIED. 5d was blocked for two
attempts and was finally run handler-side — the two APPLIED-5d sections
below are kept as the record of why, and §APPLIED-5d, COMPLETE is the
current state.**

### Premises re-verified before any mutation (pre-1)

| check | expected | observed |
| --- | --- | --- |
| `git show main:...write_triage_judge.py \| grep -c candidate_id` | 0 | **0** |
| `git rev-list --count main..task/4762` | 0 | **0** |
| `bash scripts/check_write_triage_flip_preconditions.sh` | rc=1, items 1/2/4 FAIL | **rc=1, items 1, 2 and 4 FAIL** |
| `select_judge_candidates(results, 3, canonical_id='parent-1')` | §2 | **`['m0','m1','child-1']`, len 3, first False, last True, `'parent-1'` absent** |
| control, `canonical_id='m0'` | target first | **`['m0','m1','m2']`, first True** |

Baseline `tests/server/test_write_triage_judge.py`: **163 passed**, green before
anything was added. A byte-exact capture of task 4762 was saved to
`.worktrees/.task-meta/4822/capture/task-4762.pre1.json` first (description
sha256 `e2e3813f9553bf48af46dc0ac0797fda67bc8230dea0a75fb0cd91f556dd091b`,
details sha256 `756bee1988c081e179b188cba904d2fa34b43b1d45860d101a923ca49425dd8d`).

### pre-2 — the freeze

Two `add_dependency` calls, bare-integer form:

```
add_dependency(id="4762", depends_on="4810") -> {"id":"4762","dependency_id":"4810","message":"Added dependency: 4762 now depends on 4810"}
add_dependency(id="4762", depends_on="4822") -> {"id":"4762","dependency_id":"4822","message":"Added dependency: 4762 now depends on 4822"}
```

Observed afterwards: `dependencies: [4810, 4822]`. No cycle, no rejection.

### 5a — 4762 still idle

`.worktrees/.task-meta/4762/` held `metadata.json`, `plan.json`, `reviews/`,
`verdicts/` — no `plan.lock`, no `agent_session.json`. Task row read
`('pending', None, None)` for `(status, claimant_run_id, heartbeat_at)`.

### 5b — metadata (APPLIED)

`update_task` returned `{"id":"4762","message":"Task 4762 updated","updated":true}`.
Read back, `delivered_checks` is the three-entry list of §4 AFTER: the
`kind='script'` descriptor first, then `eval_outcome_order_is_deterministic`
and `report_path_does_not_overwrite_its_own_json` byte-identical to the
pre-1 capture. `x_4762_option_contradiction` is present with
`measured_slate ['m0','m1','child-1']`, `target_index 2`, `slate_len 3`,
`canonical_id_in_slate false`.

### 5c — description round-trip (APPLIED)

`update_task` returned `updated: true`. Verified against the pre-1 capture,
not merely by sentinel search:

| assertion | result |
| --- | --- |
| new description `.endswith(original)` — byte-exact | **True** |
| original found at offset | **2599** (chars prepended; none altered) |
| length | **4898 -> 7497** |
| `details` sha256 vs capture | **identical — details never passed** |
| sentinels `ORDERING CONSTRAINT`, `reviews-cycle-2/` | **both present** |
| sentinels `RECOVERY PROVENANCE`, payload sha256 | **both present in details** |
| `title` / `status` / `priority` / `test_strategy` | **all unchanged** |

### APPLIED-5d — NOT APPLIED. The plan archive is BLOCKED.

`cp`/`rm` against `.worktrees/.task-meta/4762/` both failed with
`Permission denied`. This is OS sandbox **worktree containment**, not a
filesystem permission: the process runs as `leo` (uid 1000), the directory is
`leo:leo 775` and `plan.json` is `leo:leo 664`, and a `touch` probe into this
task's own `.worktrees/.task-meta/4822/` succeeds while the same probe into
`4762/` is denied. 5b and 5c landed because they go through the fused-memory
MCP server, not through the filesystem.

**CONSEQUENCE, stated plainly.** 4762's corrected description (5c) says the
previous plan "has been SUPERSEDED and archived to
`.worktrees/.task-meta/4762/plan.superseded-by-4822.json`". That sentence is
currently FALSE: `plan.json` is still in place, unarchived, and
`.worktrees/4762/.task/plan.json` still resolves to it. So does
`metadata.x_4762_option_contradiction.plan_superseded`. Until the archive is
performed, 4762's record makes a claim about itself that is not true — the
same class of defect this whole file exists to correct.

**Why this is contained rather than urgent.** pre-2 landed first and by
design: 4762 now depends on 4810 and 4822, so it cannot be dispatched, and
therefore cannot be re-planned or implemented against the stale
`plan.json`, until both land. The freeze is doing exactly the job it was
ordered first to do.

**What remains, verbatim.** From a context that can write
`.worktrees/.task-meta/4762/`:

```
cp .worktrees/.task-meta/4762/plan.json .worktrees/.task-meta/4762/plan.superseded-by-4822.json
cmp .worktrees/.task-meta/4762/plan.json .worktrees/.task-meta/4762/plan.superseded-by-4822.json
rm .worktrees/.task-meta/4762/plan.json
```

Do NOT hand-edit `plan.json`, and do not touch `metadata.json`, `reviews/` or
`verdicts/`. Leave the `.worktrees/4762/.task/plan.json` symlink dangling —
`TaskArtifacts.write_plan` / `single_source_plan` recreates it. With
`plan.json` absent, `read_plan()` returns `{}` and `workflow._plan()` takes
the fresh-planning path, which is what re-plans 4762 from its corrected
record. The archived plan is 45846 bytes, 24 steps, `_finalized_at`
`2026-08-27T14:52:57.460150+00:00`; its step-2 begins "[PRE-FLIP CRITICAL —
cycle-2 item 1, part 2 of 2] In write_triage_judge.py: (1) restructure
build_judge_prompt (:314-362".

Filed as a `scope_violation` blocker by task 4822.

### APPLIED-5d, attempt #2 — 2026-08-28T05:58Z, main `49e9b5c1e6`

Still BLOCKED. Three things are now known that were not known at attempt #1,
and two of them make the archive MORE load-bearing, not less.

**The attempt-#1 blocker was never adjudicated.** `esc-4822-2` reads
`status: dismissed`, `resolved_by: "auto-dismissed"`,
`resolution: "Auto-dismissed: orchestrator restarted — stale from prior run"`,
`resolved_at 2026-08-28T02:22:46Z` — twelve minutes after it was filed. So is
`esc-4822-1`. Neither was read by a handler; both were swept by a restart. The
absence of a handler response is NOT a ruling that the archive should be
skipped.

**Denial reproduced.** `touch .worktrees/.task-meta/4762/.probe-4822` ->
`Permission denied`. Directory listing still succeeds (reads are allowed;
writes are not).

**FINDING 1 — `granted_files` cannot fix this; only a handler-side run can.**
The OS sandbox write set is `write_set.writable_paths()`
(`orchestrator/src/orchestrator/agents/write_set.py::WriteSet.writable_paths`),
a FIXED contract list — worktree, **its own** `task_meta`, the git object/ref/
reflog/admin dirs, uv cache, claude_fleet, tmp, dev — built by
`compute_write_set(cwd)` at `workflow.py` (the `sandbox_extras` construction
site) and derived **entirely from `cwd`**. `granted_files` is a different
mechanism: `_collect_granted_files` folds it into `plan.files` /
`metadata.files` / locks, and touches the sandbox write set not at all. So
there is no escalation-resolution knob, and no operator config knob, that lets
a task lane write a SIBLING task's `.task-meta/`. **Cross-task plan surgery is
architecturally unreachable from any implementer lane** — the attempt-#1
hypothesis, now read off the code rather than inferred from one denial. Any
future plan scheduling it needs an MCP-mediated mechanism or a handler-side
step; the filesystem route will always fail.

**FINDING 2 — the stale plan can dispatch with NO architect pass at all.**
This is the consequence attempt #1 understated, and it is the reason the
archive matters. `workflow.py::_plan` picks its branch from the plan file:

| branch | requires | 4762's plan |
| --- | --- | --- |
| completion pass | `not _finalized_at and not _session_id` | no — both set |
| **revalidation** | `steps and _session_id and _old_plan_base` | **yes** |
| fresh planning | falsy `existing_plan` | only if `plan.json` is ABSENT |

Inside the revalidation branch sits **Lever B**, a short-circuit that stamps
`_revalidated_at`, bumps `base_commit` and returns `PLANNED` *without invoking
the architect* when `revalidation_skip_enabled and not overlap and
_can_skip_revalidation(plan)`. Measured just now, **all three hold**:

| condition | value |
| --- | --- |
| `revalidation_skip_enabled` | **True** (`config.py` default) |
| `_schema_version` vs `PLAN_SCHEMA_VERSION` | **1 == 1** |
| all 8 plan files exist in worktree | **yes** |
| plan age vs `max_revalidation_age_hours` 24.0 | **15.06h** — under the bound until 2026-08-28T14:54:41Z |
| commits on main touching any of the 8 plan files since the plan base | **0, 0, 0, 0, 0, 0, 0, 0** — overlap is EMPTY |

So if 4762 were dispatched right now, the corrected description this task wrote
in 5c would be **read by nobody**, and the superseded 24-step plan — step 2 of
which requires the refuted `candidates[0]` marking — would go straight to an
implementer. Prose corrections to the task record do not mitigate this; only
removing `plan.json` does, because absence is what selects the fresh-planning
branch. The freeze from pre-2 (`dependencies [4810, 4822]`) is the ONLY thing
currently preventing it.

**Recon independently confirmed the false claim.** `metadata.
x_4762_archive_claim_verification` was stamped by
`recon-stage-task_knowledge_sync`: `result: "confirmed_absent"`, "Do not cite
or attempt to read plan.superseded-by-4822.json; it will not be found on disk."
Its `IMPACT: low` assessment rests on the corrected requirement already being
inlined in 4762's description — true, but it addresses only the *readability*
of the archived file and not FINDING 2, where nothing in the record is read.

**Deliberately NOT done this iteration: a second description round-trip.**
4762's description still contains 5c's now-false sentence "That plan has been
SUPERSEDED and archived to ...". Correcting it would mean re-emitting all 7497
characters by hand — `update_task`'s `description` is a column overwrite, and
its `append=True` applies to `details` only — putting the recovered
esc-markup-residue-1 payload at risk to fix prose that FINDING 2 shows is not
read on the failure path that matters. The falsehood is already flagged
adjacent to itself by `x_4762_archive_claim_verification`. `details` was left
untouched per step 5c. The record stays as attempt #1 left it.

**Still the whole of what remains** (unchanged, from a context that can write
`.worktrees/.task-meta/4762/`):

```
cp .worktrees/.task-meta/4762/plan.json .worktrees/.task-meta/4762/plan.superseded-by-4822.json
cmp .worktrees/.task-meta/4762/plan.json .worktrees/.task-meta/4762/plan.superseded-by-4822.json
rm .worktrees/.task-meta/4762/plan.json
```

Re-check 4762 is still unclaimed immediately before the `rm`. At this
measurement it is: `status pending`, `claimant_run_id None`,
`heartbeat_at None`, no `plan.lock`, no `agent_session.json`.

Premises re-verified at `49e9b5c1e6`: `candidate_id` count on
`main:write_triage_judge.py` = **0**; `main..task/4762` = **0** commits;
`test_write_triage_flip_gate_invariants.py` = **8 passed**.

Re-filed as `esc-4822-3` (`scope_violation`, `blocking`, level 0) by task 4822,
attempt #2 — queued, `action: terminate_cleanly`.

FINDING 2 was also filed as a standalone follow-up against the orchestrator,
ticket **`tkt_0RSZHCP45CV2DADKJ76CYMCMMC`** (`agent-followup`, spawned_from
4822, files `orchestrator/src/orchestrator/workflow.py` +
`config.py`): `_can_skip_revalidation` checks schema, plan-file existence and
plan age, but never whether the TASK RECORD was rewritten after the plan was
finalized — and a record-only rewrite is exactly the case that leaves `overlap`
empty, which is what enables the skip. Task 4762 is the worked instance: task
`updatedAt` 2026-08-28T05:50:12Z against plan `_finalized_at`
2026-08-27T14:52:57Z, with every skip condition holding anyway. Suggested fix
there: decline the skip when the record is newer than the plan, consistent with
`_can_skip_revalidation`'s documented "on any uncertainty, return False"
posture. Ticket ids are curator tickets, not task ids — the curator decides
create/combine/drop asynchronously.

Both findings are also in memory: `architect_plan_revalidation` gained a stamped
peer subcase `lever_b_skip_bypasses_architect_entirely`
(`d9652828-09ee-48de-a3c5-46020c40ab1f`), and the granted_files trap is
`478a8029-e285-445e-8b36-46b5cf1386f4`. Recording them here as well because the
lesson this whole file exists to carry is that an escalation's `detail` field
must never be the only copy.

### APPLIED-5d, COMPLETE — 2026-08-28, run handler-side by the steward

The archive was performed **outside this lane** by the steward resolving
`esc-4822-3`, from `/home/leo/src/dark-factory` (the main checkout), because
FINDING 1 above is confirmed: `granted_files` cannot widen a lane's sandbox
write set, so no retry from this worktree could ever have succeeded. The three
commands are the ones §APPLIED-5d attempt #2 specified, unmodified.

| step | observed |
| --- | --- |
| `cp plan.json plan.superseded-by-4822.json` | done |
| `cmp` the two, **before** the `rm` | **rc=0, byte-identical** |
| sha256, both files | **`b76c3c425689dbf9cc51bca0ad7602fe6c673b1f7f6d32aab52d66dd42051e91`** |
| `rm plan.json` | done |

Pre-`rm` safety re-check by the steward: 4762 `status pending`,
`claimant_run_id None`, `heartbeat_at None`, `dependencies [4810, 4822]`, no
`plan.lock`, no `agent_session.json`, `main..task/4762` = 0 commits. The live
runtime snapshot's 4762 entry (phase EXECUTE, lane null, loops 0, attempts 0)
is stale residue from the aborted prior run, not an active claim.

Post-state, re-verified independently from this lane (reads are permitted;
only writes were denied): `.worktrees/.task-meta/4762/` now holds
`metadata.json`, `plan.superseded-by-4822.json`, `reviews/`, `verdicts/` —
`plan.json` is **gone**, and the on-disk archive's sha256 matches the value
above. `metadata.json`, `reviews/` and `verdicts/` were not touched.
`.worktrees/4762/.task/plan.json` is left dangling by design.

**FINDING 2 is thereby closed at the source.** With `plan.json` absent,
`workflow.py::_plan` selects the fresh-planning branch, so Lever B can no
longer return `PLANNED` without an architect pass and the corrected
description IS read. That absence, not the archive copy, is the load-bearing
half.

**The stale recon stamp was corrected too**, which this lane also could not
reach. `metadata.x_4762_archive_claim_verification` had been stamped
`result: "confirmed_absent"` with "Do not cite or attempt to read
plan.superseded-by-4822.json". The steward rewrote that key via `update_task`
(merge mode, nothing else altered) to `result:
"present_since_2026-08-28_steward"`, carrying the sha256, an explicit "do not
restore plan.json from the archive — audit copy only", and a note that the
prior finding's ROOT CAUSE was correct and only its present-tense verdict went
stale. Read back from `get_task(4762)` at the time of this commit: confirmed.

Consequently 4762's description sentence "has been SUPERSEDED and archived to
..." and `x_4762_option_contradiction.plan_superseded` are now TRUE as written,
so the 7497-character description round-trip §APPLIED-5d attempt #2 declined is
**not needed** — the recovered `esc-markup-residue-1` payload was never put at
risk. Verified unchanged in the same read: description sentinels `ORDERING
CONSTRAINT` and `reviews-cycle-2/` both present, `details` sentinels `RECOVERY
PROVENANCE` and payload sha256
`ebbbd9af07698fee05b46082aba64c89a441dc4449f391299ff62f0f9426b898` both
present, `delivered_checks` still the three-entry list of §4 AFTER with the
`kind='script'` descriptor first.

FINDING 1 was filed as a standalone follow-up, ticket
**`tkt_0RSZHH3H7K8KX99FVN7DRX5RPN`** (`agent-followup`, spawned_from 4822,
`escalation_id` esc-4822-3, scoped to `orchestrator/src/orchestrator/agents/
write_set.py` + `workflow.py` + `docs/task-authoring.md`): reject or flag at
plan time any step naming a path under another task's `.task-meta/`, and
provide an MCP-mediated mechanism if cross-task plan surgery is a legitimate
recurring need. Marked not-combinable with 4762/4798/4810/4822. Curator
decision lands asynchronously; ticket ids are not task ids.
