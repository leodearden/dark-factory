# Capability manifest — `plans/task-meta-task-keying-prd.md`

Mechanizes G3 + G6 for the batch. One block per task; every capability the
task's observable signal asserts, bound to evidence on main at `7658f909fc`.
Machine-readable twin: `plans/task-meta-task-keying-prd.capability-manifest.yaml`.

No FAIL bindings. This PRD introduces no novel substrate — every capability is
either present on main today or produced by a task **upstream** in this batch.

---

## α — lane↔task resolution off `plan.json`, onto `metadata.json`

| Capability | Evidence | Verdict |
|---|---|---|
| `metadata.json` carries `task_id` in the lane root | capability→producer (wired) — `artifacts.py:305-313` `init()` writes `{'task_id': task_id, …}` to `self.root/metadata.json` on **every** dispatch; confirmed live on disk (`.task-meta/_lane-36/metadata.json` → `task_id: "5069"`) | PASS |
| The two call sites to rewrite exist and read `plan.json` today | capability→producer (wired) — `git_ops.py:5442` (disk-backstop reuse) and `git_ops.py:6386` (`_find_lane_by_plan_task_id`), both `json.loads(plan_path.read_text())` then `data.get('task_id')` | PASS |
| `TaskArtifacts` is the right home for the accessor | capability→producer (wired) — `artifacts.py:262-287` is the only site joining `TASK_META_DIRNAME` to a per-worktree name | PASS |

## β — `TaskArtifacts` dual-root + routing table + sandbox grant

| Capability | Evidence | Verdict |
|---|---|---|
| A second writable root can be granted to the sandbox | capability→producer (wired) — `agents/write_set.py:152` `task_meta=TaskArtifacts.meta_root_for(...).resolve()`, a single call site in `compute_write_set` | PASS |
| Plan-tools MCP targets an arbitrary root | capability→producer (wired) — `mcp_lifecycle.py:142-167` builds `['--meta-root', str(meta_root)]` and passes it to both plan-tools and verdict-tools | PASS |
| A path-component validation guard shape already exists to copy | capability→producer (wired) — `artifacts.py:191-208` `_validate_verdict_role` + `_VALID_VERDICT_ROLE_RE`, the identical escape-prevention shape `task_root_for` needs | PASS |
| The four small readers to rewire are enumerable | capability→producer (wired) — `steward.py:727`, `task_runtime.py:223`, `task_runtime.py:255`, `stranded_verified_green.py:160`, `worktree_identity.py:83` | PASS |
| Lane resolution no longer needs `plan.json` in the lane root | producer:α **upstream** — β depends on α | PASS |

## γ — self-healing adoption at dispatch

| Capability | Evidence | Verdict |
|---|---|---|
| A lane→task scan over pool lanes exists to adapt | capability→producer (wired) — `git_ops.py:6346-6396` `_find_lane_by_plan_task_id` iterates `worktree_base`, filters `pool.is_lane(entry)`, reads the meta root | PASS |
| `.lane-state/<lane>.json` carries the corroborating occupancy record | capability→producer (wired) — `warm_lane_pool.py:295` `_note_assigned_durable(...)` on the fresh-allocation path; confirmed live on disk (`.lane-state/_lane-20.json` → `{"state":"assigned","task_id":"5702",…}`) | PASS |
| The adoption hook point runs every dispatch with the task id in scope | capability→producer (wired) — `workflow.py:2229-2246` `_setup_worktree_and_artifacts` constructs `TaskArtifacts` and calls `init()` + `ensure_lane_plan_symlink()` | PASS |
| The task root it populates is derivable | producer:β **upstream** — `task_root_for` | PASS |

## δ — plan-carried `_base_commit`

| Capability | Evidence | Verdict |
|---|---|---|
| `plan.json` already carries underscore-prefixed provenance stamps | capability→producer (wired) — live plan on disk carries `_schema_version`, `_finalized_at`, `_revalidated_at`, `_session_id`, `_created_at`; written via `write_plan`/`stamp_plan_provenance` (`artifacts.py:349-352`) | PASS |
| The revalidation branch gated on the base exists | capability→producer (wired) — `workflow.py:3733-3737` `elif existing_plan and … and self._old_plan_base:` — the exact predicate this task widens | PASS |
| `_old_plan_base` is sourced from `metadata.json` pre-`init()` | capability→producer (wired) — `workflow.py:2232-2234` `self._old_plan_base = self.artifacts.read_base_commit()` immediately before `init()` overwrites it | PASS |
| The revalidation-skip fast path that must stay coherent exists | capability→producer (wired) — `workflow.py:3757-3768` `_apply_revalidation_skip` bumps `base_commit` and stamps `_revalidated_at` | PASS |
| Layer-B hygiene's discriminator is the same value | capability→producer (wired) — `workflow.py:2248-2259` gates the `iterations.jsonl` wipe on `_old_plan_base` known **and** different from the fresh base | PASS |

## ε — per-round review staleness

| Capability | Evidence | Verdict |
|---|---|---|
| A single-file stale-review clear precedent exists to generalize | capability→producer (wired) — `workflow.py:5449-5453` unlinks `reviews/merge.json` at the top of `_execute_verify_review_loop` | PASS |
| Verdict reuse is already tree-hash-gated, so no second staleness rule is owed | capability→producer (wired) — `artifacts.py:599` `state['verdicts'][tree_hash] = {…}`; `artifacts.py:609` reads back by the same key | PASS |
| `reviews/` is written per reviewer per round | capability→producer (wired) — `workflow.py:7713` `self.artifacts.write_review(role.name, result)`; `artifacts.py:832` writes `reviews/<name>.json` | PASS |
| The reviews dir lives in the task root by then | producer:β **upstream** | PASS |

## ζ — terminal-task GC sweep *(leaf)*

| Capability | Evidence | Verdict |
|---|---|---|
| Terminal status is queryable in bulk for a sweep | capability→producer (wired) — `get_statuses` is the compact `{id: status}` read path documented in `CLAUDE.md` §Task Routing and exposed by the fused-memory MCP | PASS |
| A consecutive-streak escalation house pattern exists (INV-4) | capability→producer (wired) — `merge_liveness.py`'s consecutive-streak gate, named as the house pattern in `docs/legibility/design-invariants.md` §INV-4 | PASS |
| The sweep can identify task roots by name without a task lookup | producer:β **upstream** — `task_root_for` fixes the `task-<id>` prefix, so the sweep parses ids off directory names | PASS |
| DAG-direction | no batch task depends on ζ; every capability it needs is upstream (β) or on main | PASS |

## η — one path for agents

| Capability | Evidence | Verdict |
|---|---|---|
| The prompt sites that compute the meta path are enumerable | capability→producer (wired) — `agents/roles.py:417-418`, `:433`, `:493-494`, `:505`, `:808`, `:1137`, `:1467`; `agents/briefing.py:528`, `:532`, `:636`, `:651`, `:653`, `:882-883` | PASS |
| The lane symlink mechanism to extend already exists | capability→producer (wired) — `artifacts.py:354-386` `ensure_lane_plan_symlink` (task 2763), idempotent, recreated each dispatch | PASS |
| `iterations.jsonl` is a single file (symlinkable like `plan.json`) | capability→producer (wired) — `artifacts.py` routes `iterations.jsonl` as one path under the root, appended not rewritten | PASS |
| Rejection-style check (`expect: absent`) is satisfiable | rejection-check — the asserted signal is that `task-meta/<worktree-name>` **no longer appears** under `agents/`; the string is present today (14 sites above), so its absence is an observable state change, not a vacuous truth | PASS |

## ω — integration gate *(leaf)*

| Capability | Evidence | Verdict |
|---|---|---|
| Every leg the B1–B9 matrix exercises is produced upstream | DAG-direction — plan travel (β), adoption (γ), revalidation without an architect run (δ), review staleness (ε), prompt surface (η); ω depends on γ, δ, ε, η, each of which depends on β, which depends on α | PASS |
| A simulated multi-lane pool fixture exists to drive it | capability→producer (wired) — `orchestrator/tests/test_warm_lane_pool.py` and `orchestrator/tests/test_lane_lifecycle_gitops.py` already build `_lane-N` pool fixtures and exercise acquire routes | PASS |
| B3's assertion has a real counter to read | capability→producer (wired) — `workflow.py:5455-5459` seeds the loop counters from the persisted task-lifetime totals (task 2749) | PASS |
| B6's assertion has a real sidecar to check absent | capability→producer (wired) — `artifacts.py:1040-1064` `write_agent_session` / `clear_agent_session`, schema v2 (task 2771) | PASS |
| B9's cold-mode path is real | capability→producer (wired) — dark-factory's own `dark-factory-orchestrator.yaml` declares no `warm_lane_pool`, so this repo's suite runs the cold path by default | PASS |
