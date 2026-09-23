# Provenance — `verify_classify_run_all` fixtures

This fixture exists because `verify_classify` guard 3 classified an **aggregated
`run_all.sh` transcript** as host infrastructure trouble on the strength of
marker lines emitted by suites that **reported zero failures**. The bytes here
are a verbatim excerpt of the real transcript that caused the incident, so a
future regression fails a test instead of silently re-parking a branch fault as
infra.

## The rule

> **If a test that reads these bytes fails, guard 3's scoping has drifted.**
> Re-read the producer (`reify tests/infra/run_all.sh`) and fix the **consumer**
> (`orchestrator/src/orchestrator/verify_classify.py`). Do **NOT** edit this
> fixture, or re-cut its line ranges, to make a test pass — the markers' value
> is precisely that they are byte-identical to production emissions embedded in
> real PASS blocks. A retyped or re-sliced approximation stops being that.

---

## `reify-5623-run-all-test-leg.log`

| | |
|---|---|
| **Grounding** | **Captured run** (strongest) — a verbatim line-slice excerpt, never hand-typed |
| **Source path** | `/home/leo/src/reify/data/verify-logs/5623/attempt-1.test-20260809T125028_871741Z.log` |
| **Source sha256** | `9909d799f7a422e0d135ea68a78fc73664499244ff50407d748d50282cb3f64c` |
| **Source size** | 13826 lines / 922797 bytes |
| **Produced** | 2026-08-09 (the log's own timestamp — authoritative provenance) |
| **reify tip when excerpted** | `c09a26b5b1` (observed 2026-08-20; the log predates it, so treat the 2026-08-09 date as the binding fact) |
| **Excerpted** | 2026-08-20, dark-factory task **4492** |
| **Fixture sha256** | `5dbb063d32c915bc914c3011303df1d08deb88f6a439d11b50dc56580e2d216a` |
| **Fixture size** | 332 lines / 19564 bytes, LF endings, no trailing whitespace on any framing line |

### Why an excerpt, and why it is still authentic

The source log is **922 KB** — too large to commit. It also lives in **another
repo's** `data/verify-logs/` archive, which is reaped on its own schedule, so
this committed excerpt is the **durable** artifact and the sha256 above is what
lets a future reader prove the excerpt came from the log it claims.

The reduction is a pure line slice — every retained byte is reify's, not ours,
and only unrelated suites were dropped. Reproduce it exactly, with `$L` set to
the source path above:

```sh
{ sed -n '1,8p'         "$L";   # verify.sh preamble
  sed -n '5835,5923p'   "$L";   # test_lane_x_flock.sh       PASS block
  sed -n '8314,8382p'   "$L";   # test_reify_audit_ptodo.sh  FAIL block (the real failure)
  sed -n '10834,10990p' "$L";   # test_test_run_semaphore.sh PASS block
  sed -n '13818,13826p' "$L";   # Summary / FAILED / FAILED-DETAIL tail
} > reify-5623-run-all-test-leg.log
```

Ranges are keyed to that **exact byte stream**. If the source sha256 ever
differs, the ranges are meaningless — do not re-cut them by eye, and do not
hand-write a substitute.

### What the excerpt contains, in original order

| Fixture lines | Content |
|---|---|
| 1–9 | `verify.sh` preamble (8 sliced lines + the blank that precedes the first header) |
| 10–97 | `test_lane_x_flock.sh` block — open `--- Running: test_lane_x_flock.sh ---`, close `  RESULT: PASS (test_lane_x_flock.sh)`. Carries **markers 1 and 2**. |
| 99–166 | `test_reify_audit_ptodo.sh` block — close `  RESULT: FAIL (test_reify_audit_ptodo.sh)`. **The real failure.** |
| 168–323 | `test_test_run_semaphore.sh` block — close `  RESULT: PASS (test_test_run_semaphore.sh)`. Carries **marker 3**. |
| 324 | `  RESULT: PASS (test_warm_lane_sizing_lifecycle.sh)` — an **unmatched close** whose open header fell in an elided region. Retained deliberately: it exercises the close-with-no-open path for free. |
| 326–331 | `=== Summary: 145 discovered, 1 failed ===`, `=== FAILED: ... ===`, `FAILED test_reify_audit_ptodo.sh`, and the FAILED-DETAIL block |

### The three markers — the whole point of the fixture

All three sit at **column 0**, are **byte-identical to production emissions**,
and are enclosed in suite blocks that **PASSED**:

```
L58   lib_lane_x_flock.sh: failed to acquire Lane-X lock within 0s (LOCK=/tmp/tmp.gt3JgYzyGW)
L69   lib_lane_x_flock.sh: failed to acquire Lane-X lock within 1s (LOCK=/tmp/tmp.lgYIoAIqOb)
L320  lib_test_semaphore.sh: failed to acquire test slot within 0s (LOCK=/tmp/reify-test-semaphore-1000.lock, N=1))
```

They are genuine because those suites **execute the real emitter** to test it.
That is why line-anchoring could not fix this class: task 3679 already closed
the mid-line assertion-prose vector, and these markers survive it — there is no
*positional* rule that separates them from a host event. Only the surrounding
**PASS attestation** does.

`_SLOT_TIMEOUT_SENTINEL_RE` matches **nothing** in this fixture;
`_SLOT_ACQUIRE_DEADLINE_RE` matches **exactly these three lines**.

### The reproduction control that motivates the fixture

On dark-factory `eba215060c` (unmodified, pre-4492):

```
classify_failure(ToolKind.OPAQUE, 1, <this fixture>, False)  ->  semaphore_timeout
```

…while the true cause is stated unambiguously in the fixture's own tail: a
`test_reify_audit_ptodo.sh` PTODO **ratchet** failure —
`FAIL: live fingerprints are a subset of committed baseline (no ratchet regression)`.
The excerpt reproduces the misclassification **identically to the full 922 KB
log**, which is what makes the reduction sound.

Cost of the defect: reify **5623** held blocked **2026-08-09 → 08-19** under a
false "disk pressure / SEMAPHORE_TIMEOUT" L1 — infra-hold routing instead of
debugfix. This is the **4th recurrence** of the class (siblings **2748**,
**2821**, **3677**+**3679**, **4212**), which is why task 4492 changed the
*layer* rather than adding a 5th shape-specific veto.

---

## Producer contract this fixture pins

Emitted by **both** `run_all.sh` paths — the H2 concurrent pool
(`:1772`, `:1792`–`:1806`) and the legacy serial fallback (`:1840`–`:1850`);
the subset path emits the same shapes at `:1274`/`:1293`–`:1308` and the H9 path
at `:1222`/`:1226`–`:1228`:

```
^--- Running: <name> ---                  (open)
^  RESULT: (PASS|FAIL|SKIP) (<name>)      (close; optional " [flaky: passed on serial retry]" suffix)
```

**Blocks are atomic and never interleaved, even under the concurrent pool.**
Phase 2 buffers each member's output to its own file; Phase 3 replays it in
discovered order under its own header via
`_ra_emit_sanitized "$_H2_WORKDIR/${_h2_i}.out"` (`run_all.sh:1767`–`:1810`).
Read that loop before assuming concurrency can split a block — it cannot.

Retried members archive **both attempts under one header**, delimited by
`--- attempt 1 (concurrent pool) ---` / `--- attempt 2 (serial retry) ---`.
Those delimiters deliberately do **not** match `^--- Running: ` — `run_all.sh`
says so in a comment at `:1776`, precisely to keep the one-header-per-discovered-test
contract intact. They are ordinary interior lines.

Measured on the **full** source log: 146 open headers (`^--- Running: `), 145
`RESULT: PASS`, 1 `RESULT: FAIL`, 0 `RESULT: SKIP`, and **zero** un-indented
nested or echoed occurrences. `--- Running: ` appears on 148 lines in total; the
2 non-header occurrences are both inside indented `  PASS: ...` assertion prose
(source log L8934 and L9254 — suites asserting on run_all's *own* output
contract), which the column-0-anchored open pattern cannot match. 146 + 2 = 148
accounts for every occurrence, so nothing nested is being missed.
