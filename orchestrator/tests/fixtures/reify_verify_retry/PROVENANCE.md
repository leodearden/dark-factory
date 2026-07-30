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
