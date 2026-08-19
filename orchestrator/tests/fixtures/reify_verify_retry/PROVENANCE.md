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

## `nextest-list-ignored.json`

| | |
|---|---|
| **Grounding** | **Captured run** |
| **Producer** | `cargo nextest list --workspace --message-format json -E 'not test(=gamma::test_one)'` |
| **Version** | cargo-nextest **0.9.136** (`1d5bf1ec9 2026-05-16`) — same binary as `nextest-list.json` |
| **Captured** | 2026-07-31, from a throwaway two-crate workspace carrying two `#[ignore]`d tests |

Companion to `nextest-list.json`, capturing the shapes that file does not:
`cargo nextest list` lists every **discovered** test, including ones it has
already decided **not to run**, and marks them two different ways. Both appear
here, verbatim:

```json
"alpha::test_ignored": {"kind":"test","ignored":true,
                        "filter-match":{"status":"mismatch","reason":"ignored"}}
"gamma::test_one":     {"kind":"test","ignored":false,
                        "filter-match":{"status":"mismatch","reason":"expression"}}
```

The `-E` expression is what forces the second shape (`reason: "expression"` with
`ignored: false`) — it cannot be produced by `#[ignore]` alone, and it is the
shape reify's own per-pass filtersets emit.

Load-bearing facts:

* **`test-count` is NOT the planned count.** It is `5` here while only **2**
  tests would run. Nothing may re-derive the plan from it.
* **Excluded testcases are not planned.** `parse_per_test_results` deliberately
  drops SKIP/ignored *result* lines, so a skipped test never earns a verdict, is
  annotated `not-started` by `build_fail_fast_map`, and would land in the
  {did-not-pass} subset of **every** narrowed retry. Never *unsafe* (nextest
  still refuses to run it), but it inflates every filter file toward reify's
  `REIFY_VERIFY_RETRY_MAX_SUBSET` ceiling — and tripping that ceiling makes
  reify refuse narrowing for the whole profile, so an ignore-heavy workspace
  would silently lose the capability. `merge_shadow._nextest_case_is_planned`
  drops them.
* **Unrecognised shapes are treated as PLANNED.** The superset bias is
  deliberate and matches the module's `None`-is-never-an-empty-plan rule: a
  future nextest schema change may make DF re-run more tests, never skip one.

---

## `run_all-failed-marker.txt`

| | |
|---|---|
| **Grounding** | **Mixed.** The clean-marker tail is **contract-derived from source** — *weaker* than the two above; read this before trusting it. The partial-marker block is **producer-executed** (see below). |
| **Source** | `/home/leo/src/reify/tests/infra/run_all.sh:26-36` (documented contract), `:1839-1841` (the emitting `echo`/`printf`), and `:683-685` (`_ra_on_term`, the partial-marker producer) |
| **Read** | 2026-07-30; partial-marker block added 2026-07-31 |

This is **not a captured run**. The header block and the emission site are
copied verbatim from run_all.sh; the tail below the marked divider is synthetic
and uses only the two line formats those sites emit. It is grounded in producer
*source* rather than producer *output*, so a drift in run_all.sh's behaviour
that does not touch those lines would not be caught here. If a real failing
run_all log becomes available, prefer re-capturing over keeping this.

Corroboration that the bare `FAILED <names>` line is real and already consumed:
DF's own `verify.py` classifies it today via the `^FAILED\s` regex (pattern
\#7b), which run_all.sh cites by name.

### Second producer: the `(partial)` outer-timeout marker

`run_all.sh:684-685` (inside `_ra_on_term`) emits a **partial** marker when an
outer timeout SIGTERMs the run mid-flight:

```
    echo "=== FAILED: ${_names} (partial) ==="
    printf 'FAILED %s(partial)\n' "${_names:+$_names }"
```

Those two statements were **executed in a shell on 2026-07-31** with `_names=""`
and with `_names="a.sh b.sh"`, and their exact stdout is checked into the
fixture under its `RENDERED PARTIAL MARKERS` heading — so this variant is
grounded in producer *output*, not producer *source*:

```
=== FAILED:  (partial) ===
FAILED (partial)
=== FAILED: a.sh b.sh (partial) ===
FAILED a.sh b.sh (partial)
```

The `${_names:+...}` guard is what makes the empty-`_names` form possible, and
run_all.sh's own `DELIBERATE` comment (:652-663) confirms the marker is emitted
unconditionally in that case rather than carrying a distinct interrupted token.
The rendered block sits ABOVE the synthetic tail so that a whole-file read still
resolves (last-marker-wins) to the clean marker.

**Handled: `parse_failed_run_all_members` refuses to narrow on either form.**
When the `(partial)` token appears among the governing marker's tokens the
parser returns `[]`, routing reify to the full run_all suite.

An earlier revision of this entry called the variant out-of-scope and claimed it
"degrades safely" because run_all.sh would warn
`REIFY_RUN_ALL_MEMBER_SUBSET member '(partial)' not found in $INFRA_DIR
(ignored)` and skip the bogus name. That was wrong, and the entry is corrected
rather than deleted so the mis-analysis stays on the record:

* **Empty `_names` — a total FALSE GREEN, not noise.** The marker is
  `FAILED (partial)` with no real names at all, so the parse is `['(partial)']`
  and the subset is the single bogus token. Non-empty means verify.sh:2545's
  `[ -n "${REIFY_RUN_ALL_MEMBER_SUBSET:-}" ]` gate PASSES, so the safe
  full-suite fallback never fires; run_all then warns-and-ignores the only
  member it was given and runs **zero** members, reporting green with no
  coverage. The warning is the mechanism of the false green, not a mitigation
  of it.
* **Non-empty `_names` — a real coverage hole, under-rated as "noise".** The
  `(partial)` token itself is indeed ignored, but the surviving subset is an
  INTERRUPTED run's failure list: members that had not yet executed when the
  SIGTERM landed are neither passed nor failed, and narrowing to only the named
  failures silently skips them on the retry.

Both are instances of the same rule the rest of this leaf already follows —
never narrow on an incomplete plan (cf. an unparseable `cargo nextest list`
probe routing to a full verify, and the first-profile-only rule).

---

## Deliberately absent: a gui / vitest fixture

`REIFY_GUI_RETRY_SPECS` ships **empty** in this leaf, which `verify.sh:2127-2158`
treats as "run the full `npm test` suite" — unambiguously safe. No real reify
gui failure log was available to pin a fixture to, and authoring one from prose
is precisely the drift class this task corrects. A follow-up captures real gui
bytes and adds the parser; until then there is no gui subset to get wrong.
