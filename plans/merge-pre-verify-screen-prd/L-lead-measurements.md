# Lead measurements (2026-10-29/30)

The PRD's figures that no seat produced. All were taken in a fresh worktree of main `07bd68fd99` (`.claude/worktrees/stream1-prd`) on the workstation, with `VIRTUAL_ENV` unset.

## L1. The scanner runs in a tree with no venv

- **With `-I`.** `/usr/bin/python3 -I scripts/inline_suppressions.py --check --root .` (Python 3.12.3, no `.venv`) prints `ModuleNotFoundError: No module named '_checkout'` and exits **1**, 0.19 s. Isolated mode drops the script's own directory from `sys.path`. An import failure then exits with the scanner's FINDING code, which is why decision 5 reads a typed report and never an exit code alone.
- **Without `-I`.** The same command exits **0** in 14.23 s wall: `clean: 6381 suppression sites in 993 of 3420 tracked files; 6371 unowned in 2989 keys; excess 0, slack 19`.
- **`--json`** (mutually exclusive with `--check`) exits 0 with a report whose keys include `schema_version` (2), `status` (`clean`), `violations` (list) and `baseline` {`excess`, `slack`, `path`, `present`}.

The screen does not depend on the no-venv property, because it runs under `uv run` (decision 4).

## L2. Provisioning cost

`uv sync --all-packages --frozen --offline -q` in the venv-less worktree:

| Run | Wall | Notes |
|---|---|---|
| cold | 1.62 s | 65 MB max RSS; the resulting `.venv` is 306 MB apparent size, hardlinked from the uv cache |
| immediate re-run | 0.14 s | |
| critic re-measure (`git clone --shared` at the same sha, different load) | 7.60 s | 66 MB max RSS, `.venv` 258 MB. A single sample each, so the PRD states "2–8 s" |

## L3. The fast seven in parallel

The seven commands were started together (`&` … `wait`), each with `env -u VIRTUAL_ENV uv run …`:

| Check | Command |
|---|---|
| C1 | `uv run --project shared python scripts/inline_suppressions.py --check` |
| C8 | `cd orchestrator && uv run pytest tests/test_remote_gate.py -k wire_fingerprint -n0 -q -p no:cacheprovider` |
| C11 | `uv run pytest tests/scripts/test_orchestrator_watchdog.py -k boundary_fake_systemctl -n0 -q -p no:cacheprovider` |
| C14 | `uv run pytest tests/scripts/test_pytest_workspace_collection.py …` |
| C15 | `uv run pytest tests/scripts/test_uv_run_venv_isolation.py …` |
| C18 | `cd orchestrator && uv run pytest tests/test_recovery_veto_predicate_collapse.py …` |
| C20 | `cd orchestrator && uv run pytest tests/test_config_disjoint_skip_reload.py …` |

- All seven exited 0. Wall clock from launch to the last exit was **9.84 s**. Load average was 120.75 / 103.04 / 81.03 on 32 cores.
- Per-check pytest summaries:

  | Check | Result |
  |---|---|
  | C8 | 6 passed, 105 deselected, 0.74 s |
  | C11 | 2 passed, 536 deselected, 2.52 s |
  | C14 | 8 passed, 5.48 s |
  | C15 | 38 passed, 2.54 s |
  | C18 | 15 passed, 2.08 s |
  | C20 | 10 passed, 1.29 s |

## L4. The fast seven on non-train failures

**Figure.** 48 of 129 non-train failed verifies (37.2%), for 1,201 verify-minutes (20.0 h). The ratchet alone catches 43 of 129 (33.3%, 1,055 min), which matches M1 §5.

**Rule.**
- Population: the rows of M1 Appendix A (147 distinct `merge_verify` ids) whose `kind` is not `train`, which leaves 129.
- A row counts as caught when its `caught` column names any of C1, C8, C11, C14, C15, C18 or C20.
- Minutes are the sum of the `dur` column over the caught rows.

## L5. The ratchet as its pytest gate node

Command: `env -u VIRTUAL_ENV uv run pytest "scripts/tests/test_inline_suppression_ratchet.py::test_the_live_tree_passes_the_gate_once_the_baseline_is_seeded" -n0 -q -p no:cacheprovider`, at the worktree root.

| Run | Pytest's own time | Wall | Load average |
|---|---|---|---|
| first | 8.08 s | 33.92 s | 55 |
| second | 7.62 s | 8.26 s | ~61 |
| third | 7.59 s | 8.34 s | ~61 |
| `--collect-only` | — | 0.87 s | ~61 |

The first run pays about 26 s of cold start.

## L6. The fast seven, cold, in a fresh tree

Setup: a fresh `git clone --shared` of the main repo at `c8308465b2` (`/tmp/stream1-lead/cold`), then `uv sync --all-packages --frozen --offline -q`, which took 2.13 s.

The seven checks were then started together, with C1 as its pytest node (L5) and the other six as in L3. All seven exited 0. **Wall from launch to the last exit: 44.66 s**, at load average 63 / 70 / 95.

This is the cost a per-item screen pays in a freshly checked-out merge worktree. Every new venv compiles its imports to bytecode on first use. L3's 9.8 s is the warm figure.
