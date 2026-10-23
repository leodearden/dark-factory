# Provenance — `verify_xdist_abort` fixtures

These logs pin how verify classifies a **pytest-xdist test leg that was torn
down from outside the tests**, so that it never reached a verdict about the branch
(task **4843**, workstream #5132). Before 4843, every one of them classified
`unknown_test_failure` and went to the debugger, even though the tree was clean.

## The rule

> **If a test that reads these bytes fails, the INFRA_ABORT classification has
> drifted.** Fix the **consumer** (`orchestrator/src/orchestrator/verify.py::classify_leg`
> and its evidence helpers). Do **NOT** edit these logs, and do not hand-write a
> substitute. Their value is that real pytest/xdist/pluggy code produced every
> byte, so the classifier is not tested against our own guess at the format.

## Why reproductions and not the sighting logs

The original sighting logs under `data/verify-logs/{4808,3824,4246,4071,4347}/`
were **pruned by retention before capture**. On 2026-10-07 those directories
were empty. No copy survived in `data/`, `/tmp` or git, and the escalations
(esc-4071-1, esc-4808-1) record no verbatim bytes. The 2026-10-04 plan-time
measurement of the originals found **0 failure lines** and **8 adjacent
signature pairs** in shapes 1 and 3. The reproductions below match both.

So shapes 1 and 3 are **real reproductions**, and shape 2 is a byte-exact copy
of a surviving **real** green log of the same command.

## The signature

Shapes 1 and 3 each carry exactly 8 adjacent pairs, one per `-n 8` worker:

```
Plugin: <numeric id>, Hook: pytest_sessionfinish
OSError: cannot send (already closed?)
```

Each pair's warning header names
`_pytest/main.py:353: PluggyTeardownRaisedWarning` and quotes
`config.hook.pytest_sessionfinish(`. That quoted line is byte-for-byte
esc-4808-1's recorded `cause_hint`.

## Mechanism observed while reproducing

The signature appears **only when the xdist controller ALONE is signalled**.
Its workers outlive it and fail to send at `pytest_sessionfinish`. A SIGTERM to
the whole process **group** leaves no signature at all, only `rc=143` and
partial progress. So no single piece of evidence covers every shape, which is
why INFRA_ABORT has an **rc arm** (shell-reported 128+SIGTERM/SIGINT/SIGHUP)
and a **text arm** (the signature).

Measured under verify's real `uv run` wrapper (`shape1_uv_forwarded_sigterm.log`):
`uv run` forwards a SIGTERM to its direct child, the pytest controller, and
only to it. The workers outlive the controller and emit the signature. uv then
exits 143 itself, so the `/bin/bash -c` wrapper prints no `Terminated`, and the
unterminated last progress line is glued to the first worker warning's path.

Hypothesis: in production, the SIGTERM is delivered to uv.

## Recipe

Tool versions: pytest 9.0.3, pytest-xdist 3.8.0, pluggy 1.6.0, execnet 2.1.2,
CPython 3.13.9 (`/home/leo/src/dark-factory/.venv/bin/python`, written `$PY` below).

- Scratch suite in a fresh `/tmp` dir, never in a worktree. `pytest.ini` is
  exactly `[pytest]\naddopts =\n`.
  - Shape 1 (every variant): `test_slow.py` with 400 tests, each `time.sleep(0.5)`.
  - Shape 3: `test_many.py` with 10000 tests, each `time.sleep(0.05)`.
- Launch: `setsid /bin/bash -c "cd <suite> && $PY -m pytest -n 8 -q --tb=short -p no:cacheprovider <file>; exit \$?" > <log> 2>&1 &`, then `B=$!`.
- Wait until a progress line (`\[ *[0-9]*%\]`) appears. For shape 3, wait for `\[ *0%\]`.
- Controller-only kill: `kill -TERM "$(pgrep -P "$B" | head -1)"`.
  Group kill: `kill -TERM -- "-$B"`. Then `wait "$B"`.

The `Terminated` text glued to the last progress line in shapes 1 and 3 is the
`/bin/bash -c` wrapper reporting its child's death. verify's own `_run_cmd`
interposes the same shell. Under `uv run`, the glued text is a surviving
worker's warning path instead.

## Files

### `shape1_controller_sigterm.log`

| | |
|---|---|
| **Produced** | 2026-10-07, plan-time capture `/tmp/4843-repro/s1_ctrl_2055083.log`, copied with `cp` |
| **How** | Recipe above, 400-test suite, controller-only SIGTERM mid-run |
| **Measured rc** | 143 |
| **sha256** | `b4c1856fc682618670c11f7243a782a5bd5e36d1b07eb54f68800e27bcbba7e5` |
| **Size** | 45 lines / 3568 bytes |
| **Content** | `bringing up nodes...` x2, progress stopped at 18%, 8 signature pairs, 0 `FAILED`/`ERROR`/`INTERNALERROR` lines |
| **Models** | Shape 1: esc-4808-1 (rc=143, timed_out=false, 379.7s) and esc-3824-1/-2 (rc=143, timed_out=false, 920.13s) |

### `shape1_group_sigterm.log`

| | |
|---|---|
| **Produced** | 2026-10-07, plan-time capture `/tmp/4843-repro/s1_group_2055083.log`, copied with `cp` |
| **How** | The same run as above, but SIGTERM to the whole process **group** |
| **Measured rc** | 143 |
| **sha256** | `536fa55845b0b0a0e32b2381fa7904229edf0a7fef7d9078b6e114768ba42427` |
| **Size** | 4 lines / 130 bytes, no trailing newline |
| **Content** | `bringing up nodes...` x2, partial progress, **0 signature pairs**, 0 failure lines |
| **Models** | The shape-1 kill delivered to the group. Only the rc arm can catch it: replayed at rc=1 it must stay `unknown_test_failure` |

### `shape1_uv_forwarded_sigterm.log`

| | |
|---|---|
| **Produced** | 2026-10-07, plan-time capture `/tmp/4843-glue/uv_xdist_term.keep.log`, copied with `cp` |
| **How** | Recipe above, 400-test suite, with three changes: the launch is `setsid /bin/bash -c "cd <suite> && uv run --no-project --python $PY python -m pytest -n 8 -q --tb=short -p no:cacheprovider test_slow.py; exit \$?"`; the kill is `kill -TERM "$(pgrep -P "$B" | head -1)"`, where that pid is **uv**, not pytest; and the wait adds ~0.3s after the first progress line so the partial line is non-empty. uv 0.11.6, bash 5.2.21 |
| **Measured rc** | 143 |
| **sha256** | `dc49df7678d19d13bffa2a7ca63b0bea5a3d5abba7840a26bef5d1067a80fec8` |
| **Size** | 44 lines / 3691 bytes |
| **Content** | `bringing up nodes...` x2, progress stopped at 18%, the unterminated partial line `........` glued to the first worker warning's path, 8 signature pairs, 0 `FAILED`/`ERROR`/`INTERNALERROR` lines |
| **Models** | Shape 1 under verify's real `uv run` wrapper: no `Terminated`, because uv itself exits 143 |

### `shape3_bringup_sigterm.log`

| | |
|---|---|
| **Produced** | 2026-10-07, plan-time capture `/tmp/4843-repro/s3_2175810.log`, copied with `cp` |
| **How** | Recipe above, 10000-test suite, controller-only SIGTERM once `[  0%]` appeared |
| **Measured rc** | 143 |
| **sha256** | `3bcab1ecabf1c5e393672c4d109fdcbf6ba3be9850321abcba304122b8671fd0` |
| **Size** | 46 lines / 3650 bytes |
| **Content** | `bringing up nodes...` x2, progress lines at 0% and 1%, 8 signature pairs, 0 failure lines |
| **Models** | Shape 3: esc-4071-1. **That sighting's rc was never recorded**, so its classification must rest on the text arm |

### `shape2_green_tally.log`

| | |
|---|---|
| **Source path** | `/home/leo/src/dark-factory/.worktrees/4347/.task/verify/attempt-1.fused-memory.test.log` |
| **Source sha256** | `3e25e56ceb41ae5de17ced066850dc3ca9b6ffddfddb8f3b5d0200fdf9f4cf3d` (checked equal after `cp`) |
| **Produced** | 2026-09-06 (the source file's mtime), a real verify leg |
| **Command** | `uv run --directory fused-memory pytest tests/ --tb=short -q --timeout=300`, the same command as the esc-4246-4 sighting |
| **Real rc** | 0. Tests replay it at rc=143 to model esc-4246-4 (rc=143, timed_out=false, 435.78s) |
| **Size** | 334 lines / 26908 bytes |
| **Content** | A complete green tally, `18885 passed, 1 skipped, 249 warnings in 225.85s (0:03:45)`, a real warnings-summary section, **0 signature pairs**, 0 failure lines |
| **Models** | Shape 2: the session finished, then the shell reported a SIGTERM. Only the rc arm can catch it |

## The RED premise, measured on all four files on 2026-10-07

Before task 4843:

```
classify_failure(ToolKind.PYTEST, 143, <file>, False) == 'unknown_test_failure'
classify_failure(ToolKind.PYTEST, 1,   <file>, False) == 'unknown_test_failure'
verify._is_bare_xdist_worker_crash(<file>)            is False
```

`shape1_uv_forwarded_sigterm.log`, added later the same day, was measured the
same way: `classify_failure(PYTEST, 143|1, <file>, False) == 'unknown_test_failure'`.
