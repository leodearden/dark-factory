# Provenance — `reify_verify_retry` fixtures

These fixtures exist because the shipped D2 retry-subset producer was built from
its own docstring rather than from the bytes its consumer actually emits (PRD
§12 root cause (a)), and shipped three real producer/consumer mismatches as a
result. Every file here is grounded in a **real producer** so a future drift
fails a test instead of silently narrowing a retry.

## The rule

> **If a test that reads these bytes fails, the DF/reify seam has drifted.**
> Re-capture the fixture from a live lane (or re-read the producing source) and
> fix the **consumer**. Do **NOT** edit a fixture to make a test pass — that
> re-creates exactly the failure mode these fixtures exist to prevent.

---

## `reify-verify-attempt.json`

| | |
|---|---|
| **Grounding** | **Captured run** (strongest) |
| **Source path** | `/home/leo/src/warm-lanes/worktrees/_lane-43/target/reify-verify-attempt.json` |
| **Captured** | 2026-07-30 |
| **Producing code site** | `reify` `scripts/verify.sh`, function `add_test_passes()` — the `printf` to `_ATTEMPT_SIDECAR_PATH` (`verify.sh:738` defines `_ATTEMPT_SIDECAR_PATH="${REIFY_VERIFY_ATTEMPT_SIDECAR:-target/reify-verify-attempt.json}"`). Landed by reify task **#5548**. |
| **Bytes** | `{"tree_oid":"39476eca69c4f4c10fb2cca86c4f36fe3aa41a36","profiles":"debug release","timestamp":"2026-07-30T04:56:41Z"}` + a single trailing newline |

Copied **verbatim**: not reformatted, re-indented, key-reordered, or
pretty-printed. The point is that the bytes are reify's, not ours.

Load-bearing facts a reader must not re-derive from prose:

* **`profiles` is a space-delimited STRING, not a JSON list.** DF parses it with
  `.split()`.
* The path is **relative to the worktree root** — `target/reify-verify-attempt.json`,
  not a DF-invented `.reify-verify-retry/attempt0.json` (the phantom this task
  deletes).
* `tree_oid` is reify's `git rev-parse HEAD:` — the same value DF's
  `git_ops.get_head_tree_hash` produces, which is what makes the INV-3
  corroboration two genuinely independent reads.
* `add_test_passes()` stamps the sidecar as its **first** plan line, so it
  survives a RED psi-gate / compile-gate / nextest pole. Consequence: `profiles`
  records what the attempt **planned** to run — it is never proof a profile
  actually executed.

---

## `nextest-list.json`

| | |
|---|---|
| **Grounding** | **Captured run** |
| **Producer** | `cargo nextest list --workspace --message-format json` |
| **Version** | cargo-nextest **0.9.136** (`1d5bf1ec9 2026-05-16`) — the version reify's merge gate runs on this host |
| **Captured** | 2026-07-30, from a throwaway two-crate workspace (no reify lane with warm binaries was available to list without a compile) |

The workspace was synthetic; **the JSON was not** — it is unmodified
cargo-nextest output. Shape that matters, and that the parser is pinned to:

* top-level `rust-suites`, an object **keyed by suite id**;
* each suite value carries `binary-id` and a `testcases` object **keyed by bare
  test name**;
* top-level `test-count` equals the total number of testcases.

Capture covers 3 suites / 5 tests, including a lib suite, an integration-test
suite (`crate-a::integration`), and two crates sharing a leaf test name — so the
parser's suite-qualification and de-duplication paths are exercised by real
bytes rather than a hand-written one-suite stub.

### The empirical result that forced this fixture

Run against the same 0.9.136 binary:

```
cargo nextest list -E 'test(=mymod::mytest)'          -> MATCHES
cargo nextest list -E 'test(=nxprobe mymod::mytest)'  -> MATCHES NOTHING
```

reify wraps every filter-file line as `test(=<line>)` (`verify.sh`
`emit_nextest_pass`). So writing DF's `"<binary-id> <test-name>"` parse key into
a filter file yields a file that is **non-empty** (reify's "retry refused: no
subset" loud fallback therefore never fires) and matches **zero tests** — a
narrowed retry that runs nothing and reports PASS. That is a latent **FALSE
GREEN**, strictly worse than the inertness the task names. Filter files must
carry the **bare test name**; `merge_shadow.nextest_filter_ids` performs that
mapping at the single write boundary.

---

## `run_all-failed-marker.txt`

| | |
|---|---|
| **Grounding** | **Contract-derived from source** — *weaker* than the two above. Read this before trusting it. |
| **Source** | `/home/leo/src/reify/tests/infra/run_all.sh:26-36` (documented contract) and `:1839-1841` (the emitting `echo`/`printf`) |
| **Read** | 2026-07-30 |

This is **not a captured run**. The header block and the emission site are
copied verbatim from run_all.sh; the tail below the marked divider is synthetic
and uses only the two line formats those sites emit. It is grounded in producer
*source* rather than producer *output*, so a drift in run_all.sh's behaviour
that does not touch those lines would not be caught here. If a real failing
run_all log becomes available, prefer re-capturing over keeping this.

Corroboration that the bare `FAILED <names>` line is real and already consumed:
DF's own `verify.py` classifies it today via the `^FAILED\s` regex (pattern
\#7b), which run_all.sh cites by name.

### Known producer variant NOT handled by the current parser

`run_all.sh:684-685` emits a **partial** marker when an outer timeout SIGTERMs
the run mid-flight:

```
    echo "=== FAILED: ${_names} (partial) ==="
    printf 'FAILED %s(partial)\n' "${_names:+$_names }"
```

`parse_failed_run_all_members` implements the documented
`FAILED <space-separated names>` contract only, so against a partial marker it
would return the literal token `(partial)` alongside the real member names.
That degrades **safely** — run_all.sh warns
`REIFY_RUN_ALL_MEMBER_SUBSET member '(partial)' not found in $INFRA_DIR
(ignored)` and skips it — but it is noise, and it is recorded here rather than
silently absorbed. Handling it is out of scope for task 3059 (filed as
follow-up).

---

## Deliberately absent: a gui / vitest fixture

`REIFY_GUI_RETRY_SPECS` ships **empty** in this leaf, which `verify.sh:2127-2158`
treats as "run the full `npm test` suite" — unambiguously safe. No real reify
gui failure log was available to pin a fixture to, and authoring one from prose
is precisely the drift class this task corrects. A follow-up captures real gui
bytes and adds the parser; until then there is no gui subset to get wrong.
