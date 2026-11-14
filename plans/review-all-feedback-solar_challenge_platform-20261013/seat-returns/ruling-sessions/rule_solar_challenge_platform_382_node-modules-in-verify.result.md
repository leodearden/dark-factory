---
outcome: done
changed:
- commits 294dbb2 + 42dbc4b (PRD note beside F22 and its same-day amendment, docs-only)
- task 382 re-scoped in place (high; depends on 1239; tests drive the install at merge, fail-closed)
- ruling/amendment notes on 382, 1239, 1268; edge 1268 -> 382
- SCP 1298 filed (B: declared-input `frontend` step on the verify-cache per-step tier; deps 382, 1268, external dark_factory:5653)
- dark_factory 7024 filed then cancelled (path selection, superseded)
- note on dark_factory 5652 (tree-tier key must cover role + provisioning)
- decisions_and_rationale memories written (ruled_by human)
action_needed:
1. review-all lead: apply the program-doc amendments listed below.
2. After 382 lands, refresh review/briefing.yaml interactively on main: its frontends decision entry still records task 321's "no npm in verify".
3. Leo/lead: decide on the fallback-scoping caveat (no pytest at all for .py diffs without a test file).
4. dark-factory: the verify-cache β/γ authors must honour the 5652 note.
---
**Ruling (Leo, 2026-10-16), task 382, as amended the same day.** The frontend vitest suites and tsc gate merges at the MERGE verify, fail-closed.
- **The tests drive the install.** There is no orchestrator preprovision. When `DF_VERIFY_ROLE=merge`, `tests/e2e/frontend_npm.py`'s gate runs the root `npm ci` once per session, only when a frontend test actually executes, and fails (never skips) if the install is missing or fails. Other roles skip, as today.
- **Why:** every merge verifies in a fresh worktree, so the install matches the merged lockfile and leaves nothing on disk. Verify is not landlocked. Cost is ~1 min per merge.
- **Why the amendment:** when node_modules presence is the proxy for "run the tests", the policy lives in two places and leaks through agent-installed node_modules. With the tests driving the install, the policy has one home.
- **Option B (later) = SCP 1298.** "Always run the frontend tests; skip them when not needed", via dark-factory's verify-result cache per-step tier (`plans/verify-result-cache-prd.md` § Reify seam contract): a declared `frontend` step with input globs, a per-step verdict sidecar, and `DF_VERIFY_SKIP_STEPS` honoured at merge. 1298 depends on 382, 1268 and dark_factory:5653 (γ).
- **Superseded:** path selection (dark_factory 7024) was cancelled.
- **The per-test cache tier would not cover the frontend tests.** It is scoped to dark-factory's own repo, and a test that runs a subprocess in the repo is unkeyable (repo_state).

**Rejected options:**
- A1 (install in every task verify): stale installs and disk.
- C (warm-lane reflink seeding): feasible on the XFS warm-lanes volume, but dark-factory assumes one warm base per host, and moving the worktrees off ext4 breaks uv's hard-linked venvs. Leo corrected my "host is ext4" framing.
- D (cadence lane): a break would land on main, and an L2 would surface it a day later.

**Cross-project finding, noted on dark_factory 5652 (β).** The verify-cache tree-tier key, as specified, omits the verify role and the preprovision. SCP's task-role and merge-role test commands are identical, and a rebased branch's tree equals its merge tree. So a task-role green with the frontend tests skipped could replay as the merge verdict. The key must cover the role and the provisioning, or a role-sensitivity flag must make that conditional.

**Program-doc amendments for the review-all lead:**
- S11 Q3 becomes: "RULED 2026-10-16: merge-verify frontend gate, tests drive the install, fail-closed; 382 -> 1239; 1268 -> 382; B later = 1298 on dark_factory 5653".
- Close "382 (per-task node_modules)" in the kept-for-Leo list.
- FILED row S11: add 1298.

**Caveat found (not ruled):** per `verify.py::_build_fallback_config` (code reading), a diff whose .py part has no collectable test, conftest or test-data file runs NO pytest in task or merge verify. That disables every pytest-side gate for such a diff: S11's docsgen --check and INV-4, and these frontend lanes. It is noted on 1268.

**Team:** 4 sonnet seats (verify-gate high 210k/38m, nets-sandbox high 214k/40m, warm-seed medium 89k/11m, exposure medium 84k/7m), plus lead measurements and the lead's reading of the verify-cache PRD.
