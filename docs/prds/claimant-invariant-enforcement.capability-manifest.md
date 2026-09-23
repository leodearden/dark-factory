# Capability manifest — claimant-invariant-enforcement

PRD: `docs/prds/claimant-invariant-enforcement.md` · verified against main
`d31a125357` · machine-readable twin:
`docs/prds/claimant-invariant-enforcement.capability-manifest.yaml`

Mechanises G3 + G6 per leaf: every capability each task's signal asserts, bound
to evidence, so a dispatch-time architect diffs intent against substrate instead
of re-deriving the check. **33 capabilities across 6 tasks — 33 PASS, 0 FAIL,
0 OPEN.** 15 mechanical `delivered_check`s (copied into producer
`metadata.delivered_checks` at `commit_planning`), 18 `manual` (recorded here,
excluded from the dispatch gate).

## Scoping is load-bearing, not tidiness

Every `grep` check carries an explicit `paths:` scope. This is not style: an
unscoped `git grep -cE 'DEFAULT_CLAIMANT_HEARTBEAT_TTL'` returns **3** hits on
this HEAD, and **all three are the PRD's own prose**. A repo-wide check would be
satisfied by the document that *describes* the work rather than by the work —
the "structural assertion satisfied by a neighbouring name" trap, firing on the
manifest's own subject.

Every pattern was measured on `d31a125357` with a known-positive control:

| Check | today | control |
|---|---|---|
| `def is_stranded_any_status` (shared) | 0 | `def is_stranded` → 2 |
| `def violates_terminal_claimant_invariant` | 0 | ″ |
| `def is_stale_hygiene_tier_claimant` | 0 | ″ |
| `DEFAULT_CLAIMANT_HEARTBEAT_TTL` (`*.py`) | 0 | 3 repo-wide, all PRD prose |
| `claimant_run_id=None` (interceptor) | 0 | `set_task_claimant` (harness) → 2 |
| `is_stranded_any_status` (task_ground_truth) | 0 | ″ |
| `def clear_claim_then_set_status` | 0 | ″ |
| `clear_claim_then_set_status` (harness) | 0 | ″ |
| `violates_terminal_claimant_invariant` (scripts/) | 0 | ″ |
| `claimant-invariant-enforcement` (task-status-authority) | 0 | `claimant` → 21 |
| `so this is inert today` (tools.py) | **1** | *`expect: absent`* — true-after |
| `never freeform text` (tools.py) | **1** | *`expect: absent`* — true-after |

The two `expect: absent` checks are genuine rejection-style bindings: both
comments are present today and falsified on this HEAD, so each is
false-before / true-after rather than vacuous.

## Per-task bindings

Full bindings with their evidence strings are in the YAML twin; this is the
reviewer's summary.

### α — shared predicates + single TTL (intermediate; unlocks γ, ζ)
The liveness **core is already factored out** (`task_claimant.py:63`
`_claimant_liveness_stranded`, delegated to by all three predicates), so α adds
a public wrapper plus the two-line `infra_hold` carve-out — **not** the
"delete the duplicated cores" an earlier draft imagined. α also exports the
C4-E1 predicate and the single TTL, **and repoints the four existing copies**:
exporting a sixth definition while leaving five standing would be a net INV-5
regression. `artifacts.py:1339`'s `600.0` is a `plan.lock` threshold on a
different mechanism — annotated, not repointed. α **re-expresses, never
retires**: `is_stranded` keeps a live consumer at `dashboard/data/tasks.py:315`
and `is_stranded_blocked` at `scheduler.py:6365`.

### β — choke-point clear (intermediate; unlocks ζ, η)
The choke point genuinely covers the orchestrator — chain traced end to end:
`Scheduler.mark_done` → `set_task_status` → `dispatch_tool` → `mcp_call` →
`tools.py:7698` → `_apply_status_transition` → the SQL. Two `expect: absent`
checks bind β's comment corrections. Note the corrected anchors: the sentinel is
at `tools.py:953` (not `:946`), its defaults at `:7566-7567` / `:7677-7678`
(not `:7603-7604` / `:7714-7715`), and the infra-hold comment at `:1328-1329`
(not `:1405-1407`, which is unrelated citation prose) — three anchors the PRD
originally got wrong and two review passes wrongly blessed.

### γ — repoint the reader (leaf)
`is_stranded_any_status` comes from α, γ's declared prereq: DAG direction
correct. The `_maybe_submit_stranded_verified_green` short-circuit is
**structurally unreachable** for γ's shape — it needs `resolve_branch_sha` to
resolve the branch, and `GONE_NO_MARKER` is reached precisely *because* that
same call returned `None`. Exactly **one** executable pin asserts the old
behaviour (`test_task_ground_truth.py:783`); it is **inverted carrying the
rationale**, not deleted, plus two prose assertions to update.

### δ — extract the clear-then-flip helper (intermediate; unlocks ζ)
Redesigned after G7: adding a third hand-written copy would reproduce the very
convention-by-imitation the PRD names as the root cause. There is **no
harness/scheduler seam** — `Harness.scheduler` is a `Scheduler` and δ's site
already calls `self.scheduler.set_task_status`. The extraction also collapses
two rotted slot-release citations into one docstring.

### ζ — census + repair (leaf)
Imports α's predicates and TTL rather than re-expressing them. Per-row re-read
before each write (an aggregate pre-flight cannot catch a row reopened between
census and apply). The zero carries a **named positive control**: task 4028,
re-verified live 2026-08-22. The terminal tier held at exactly **29** across a
day in which the hygiene tier churned 104 → 84 — that stability, not the raw
total, is the achievability basis.

### η — amend the origin contract (leaf, non-code)
All three target sites verified verbatim: C4 `:239`, D4 `:158-159`, A5 `:285`.
D4's named mechanism is itself doubly stale — it cites `scheduler.release`,
which clears **nothing** — which is why a one-line pointer near C4 was
insufficient.

## Detection is absent by design

The alarm leaf was removed from this batch (PRD D8). Its capabilities are **not**
recorded here as OPEN, because the task is not filed: carrying them would imply
a queued producer. The detection PRD inherits four unmet requirements — an
invocation model, a discriminator that discriminates (`infra_issue` is 30.8% of
the dark-factory corpus), a full closer analysis (`authority.py`'s denylist
governs only one of at least three closers), and a re-arm that can distinguish
"clean" from "blind".
