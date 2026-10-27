# Red-on-main recovery (enforce-safe, break-glass)

When a bad merge has turned `main` RED and the ref must be moved to a known-good SHA, use the `recover_main` CLI for a **single atomic move**. This avoids the rewind-then-readvance back-and-forth that produces a net no-op.

**This move is BACKWARD (non-fast-forward):** `<current-main>` is not an ancestor of `<good-sha>`, so it rewrites history. A project whose main-gate hook has an **always-on non-fast-forward guard** (reify) REJECTS a backward ref move *unconditionally* — before the sanction/sentinel check is even reached — so `git.main_gate_mark_command` (the forward-move sanction that `advance_main` uses) is **not** sufficient on its own for recovery. For those projects the CLI instead engages a **durable break-glass bypass** of the non-ff guard for exactly the CAS window (see the prerequisite check). Projects with only a sanction gate (no non-ff guard) still get the mark and land the move as `SANCTIONED`.

## Prerequisite check

- `$DARK_FACTORY_ROOT` names the dark-factory checkout, whose `.venv` runs `orchestrator.recover_main`.
- The watched project's `project_root` working tree must be **clean** (no uncommitted WIP). The CLI does not stash/pop — that is out of scope for a break-glass operation.
- **If the watched project has an always-on non-ff main-gate guard (reify):** confirm its `dark-factory-orchestrator.yaml` `git:` block sets **BOTH** `git.main_gate_bypass_command` (engages the durable bypass) **AND** `git.main_gate_bypass_clear_command` (clears it). When both are set, `recover_main` **AUTO-engages** the durable bypass immediately before its CAS `update-ref` and **AUTO-clears** it immediately after on every path (success, CAS failure, exception), so the recovery move is allowed through without leaking the bypass into later ref moves. These are generic per-project shell commands (run via `sh -c` in `project_root`); they should drive the same gate-side knob the project's hook reads — see the fallback subsection below for the knob forms. When the bypass command is set it **supersedes** `git.main_gate_mark_command`: the CLI skips the mark entirely (running both would leave the one-shot sanction sentinel unconsumed and falsely sanction the next ref move).
- **If the watched project has only a sanction gate (no non-ff guard):** confirm `git.main_gate_mark_command` is set. Leave the bypass commands unset. If the mark is also unset, the move is still atomic but will not be sanctioned by the reference-transaction hook (use the manual fallback below).

## Step 1 — Identify the bad merge and good target

```bash
# In the WATCHED project's repo:
git log --oneline -10 main
# Find the last known-good SHA (pre-bad-merge) and the current bad SHA.
```

- `<good-sha>` = the commit you want to restore `main` to (pre-bad-merge)
- `<current-main>` = current value of `refs/heads/main` (the bad merge; CAS old-value)

## Step 2 — Perform the single atomic recovery move

Run from `$DARK_FACTORY_ROOT`:

```bash
.venv/bin/python -m orchestrator.recover_main \
  --project-root <watched-project-root> \
  --config <watched-project-root>/dark-factory-orchestrator.yaml \
  --target-sha <good-sha> \
  --expected-main <current-main>
```

Parse JSON output: `{"result": "rewound"|"cas_failed"|"error", "target_sha": "..."}` (on an unexpected exception the object also carries a `"detail"` key).

- `rewound` (exit 0): `main` is now at `<good-sha>`; the durable bypass (if configured) has already been cleared. Proceed to step 3.
- `cas_failed` (exit 1): another writer moved the ref first. Re-read current `main` and retry — or escalate to the human if the situation is unclear.
- `error` (exit 1): a SHA failed pre-validation (typo, or not a commit in the watched repo). Fix the SHA and retry — do **not** treat this as a CAS race.

## Step 3 — Re-advance the fix through the normal merge queue

Once `main` is at the good SHA, the fix commit must land through the normal merge queue (already sanctioned by `advance_main`). Do **not** repeat a raw `git update-ref` — use the queue.

## Break-glass bypass fallback (bypass commands NOT configured)

Prefer the automatic path above: set `git.main_gate_bypass_command` / `git.main_gate_bypass_clear_command` in the watched project's `dark-factory-orchestrator.yaml` so `recover_main` engages and clears the bypass for exactly the CAS window. Use this **manual** fallback only when a non-ff-guarded project has **not** yet configured those two commands and you cannot add them right now.

The gate itself honors a project-specific break-glass bypass; the two config commands above are just generic `sh -c` wrappers that toggle one of these gate-side knobs. Check the watched project's config for the knob it reads:

- **Environment variable**: `<PROJ>_MAIN_GATE_BYPASS=1` (e.g. `REIFY_MAIN_GATE_BYPASS=1`)
- **Git config key**: `git config <proj>.mainGate.bypass true` in the watched repo
- **Flag file**: a sentinel file under the repo's git common-dir (path from the watched project's gate config)

These are gate-side controls read by the watched project's reference-transaction hook — they are not dark-factory settings. Consult the watched project's gate documentation for the exact knob. To recover manually: **(1)** engage the bypass (set the knob), **(2)** run a raw CAS `git update-ref refs/heads/main <good-sha> <current-main>` in the watched repo (which the guard now allows through), **(3)** **clear the bypass immediately after** — on both success and failure, since the bypass is durable and would otherwise leak into every later ref move. Once the bypass commands are configured, `recover_main` does steps 1–3 for you; wire them up so the next recovery is one command.
