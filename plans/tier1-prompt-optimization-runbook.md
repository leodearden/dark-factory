# Tier-1 Prompt Deploy Runbook — Pin → Watch → Unpin-on-Regress

Operationalizes `plans/tier1-prompt-optimization-prd.md` §8 step 4: once an
operator has decided to ship an optimized reviewer/curator heuristics block
(§8 steps 1–3 — running the real optimization loop and reading its held-out
**TEST** verdict — is human judgment, out of scope for this doc), this is the
mechanical pin → watch → unpin-on-regress sequence. Rollback is **one
function call** (`PromptArtifactStore.unpin`, D-4) — there is no separate
revert path, so "watch" is the only step that takes real judgment here.

**Applies once T2 (reviewer) or T3 (curator) loader wiring is live** — until
the relevant prompt's call site resolves via `PromptArtifactStore.resolve`,
pinning an artifact has no observable effect on the running pipeline (it just
sits on disk). Both are deps of this PRD's later tasks, not of T8.

Time budget: pinning is a five-minute scripted action; watching spans the
post-deploy window itself (default 7 days — not continuous attention, just a
daily canary re-run).

---

## Pre-flight

1. **Identify the key.** You need the exact `(prompt_id, executor_model,
   harness_version)` triple the candidate was optimized for — `executor_model`
   is the model **resolved at invocation time** (P-4: `adaptive-model-routing`
   makes this dynamic), not a name you pick by hand. Get it from the loop
   report / provenance you're about to pin.

2. **Confirm you're in the MAIN checkout**, not a task worktree:

   ```bash
   cd /home/leo/src/dark-factory && pwd
   ```

   Both halves of this runbook read/write **untracked, checkout-local**
   state, and each is resolved differently — see the two warning boxes below
   (Step 1 and Step 2). Running from `.worktrees/<n>/` silently operates on
   that worktree's own empty copy instead of production's.

3. **Check nothing unexpected is already pinned** for this key:

   ```python
   from shared.prompt_artifact import PromptArtifactStore, default_artifacts_root

   store = PromptArtifactStore(default_artifacts_root())
   print(store.read_provenance("reviewer", "<executor_model>", "<harness_version>"))
   ```

   `None` means nothing pinned (the live pipeline is on the in-code
   constant). A non-`None` result means a prior artifact is live for this
   key — confirm before overwriting it that you intend to replace it.

---

## Step 1 — Pin the accepted artifact (T1)

> **Root gotcha:** `default_artifacts_root()` resolves relative to
> `shared/src/shared/prompt_artifact.py`'s **own on-disk location** (or
> `DARK_FACTORY_PROMPT_ARTIFACTS` if set) — not your shell's `cwd`. Run this
> from the main checkout so it lands in
> `/home/leo/src/dark-factory/data/prompt_artifacts`, the root every T1
> consumer (T2/T3 live wiring, T8 tooling) agrees on. Pinning from inside a
> task worktree resolves a *different* directory
> (`.worktrees/<n>/data/prompt_artifacts`) that the live pipeline never
> reads — the pin would silently do nothing observable.

```python
from datetime import datetime, timezone
from shared.prompt_artifact import ArtifactProvenance, PromptArtifactStore, default_artifacts_root

store = PromptArtifactStore(default_artifacts_root())

deploy_at = datetime.now(timezone.utc).isoformat()  # anchors Step 2's --deploy-at

store.pin(
    "reviewer",                    # prompt_id
    "<executor_model>",            # the ROUTER-RESOLVED model this candidate was scored on (P-4)
    "<harness_version>",           # must equal provenance.harness_version below, or pin() raises
    heuristics=open("<candidate_heuristics.txt>").read(),
    provenance=ArtifactProvenance(
        optimizer_model="<frontier model that proposed edits>",
        corpus_hash="<held-out corpus hash from the loop report>",
        split_seed=42,
        held_out_TEST_score=0.87,       # the loop report's TEST-split score
        accept_delta=0.05,              # paired delta vs. baseline that cleared the repeatability band
        git_sha="<git rev-parse HEAD>",
        date=deploy_at,                 # doubles as this deploy's canary anchor -- see Step 2
        harness_version="<harness_version>",
    ),
)
```

Notes:

- `pin()` writes `heuristics.txt` then `provenance.json` atomically (temp
  file + `os.replace`); a reader never observes a half-written file. On a
  **re-pin** of an already-pinned key the stale `provenance.json` is removed
  first, so a crash mid-write fails safe to "nothing pinned" rather than a
  mismatched heuristics/provenance pair.
- `provenance.date` is read straight back by `deploy_at_from_provenance` in
  Step 2 — it does not have to carry a time-of-day component
  (`datetime.fromisoformat` accepts a bare `YYYY-MM-DD` too), but recording
  the full pin timestamp costs nothing and gives the canary a tighter
  baseline/post split.
- The next `resolve()` call anywhere in the live process picks this up
  immediately — `PromptArtifactStore.resolve` does disk I/O with no
  memoization, so **no service restart is needed** (unlike e.g.
  `plans/sqlite-cutover-runbook.md`'s config-reload step).

---

## Step 2 — Watch: run the canary (T8)

> **Root gotcha (different from Step 1):** `data/orchestrator/runs.db` is a
> **gitignored runtime artifact** the live orchestrator writes to — it only
> exists, and is only current, in the checkout that's actually running
> production traffic. Point `--runs-db` at
> `/home/leo/src/dark-factory/data/orchestrator/runs.db` explicitly (the
> CLI's default) or set `PROMPT_OPT_RUNS_DB`; a task worktree's copy (if one
> exists at all) is stale/empty.

```bash
cd /home/leo/src/dark-factory/orchestrator && uv run python -m orchestrator.evals.prompt_opt canary \
    --deploy-at "$deploy_at" \
    --project-id dark_factory \
    --baseline-days 7 --post-days 7
```

Or skip hand-copying `$deploy_at` out of Step 1 and resolve it straight from
the T1 provenance sidecar you just pinned (the `deploy_at_from_provenance`
T8→T1 seam):

```bash
cd /home/leo/src/dark-factory/orchestrator && uv run python -m orchestrator.evals.prompt_opt canary \
    --prompt-id reviewer --executor-model "<executor_model>" --harness-version "<harness_version>" \
    --project-id dark_factory \
    --baseline-days 7 --post-days 7
```

**The four metrics** (`orchestrator/src/orchestrator/evals/prompt_opt/canary.py`),
all oriented **higher = regression** — this is the D-7 MAS net-negative guard:
a role-locally-better prompt (e.g. a reviewer that blocks more) can still be
net-negative if it shifts cost into debugger/steward cycles downstream.

| Metric | Definition | Window / filter |
|---|---|---|
| `cost_per_done_task` | `sum(cost_usd + steward_cost_usd)` over **every** row / `count(outcome=='done')` | all rows summed, divided by done-row count — captures downstream cost-shift |
| `requeue_rate` | `count(outcome=='requeued') / count(all rows)` | all rows |
| `mean_review_cycles` | mean `review_cycles` | `outcome=='done'` rows only |
| `mean_verify_attempts` | mean `verify_attempts` | `outcome=='done'` rows only |

**`requeue_rate` is a lower bound, not an absolute count** — `save_task_result`
does `INSERT OR REPLACE` keyed on `(run_id, task_id)`, so a task that requeues
and later completes ends up recorded as `'done'`; only the *latest-recorded*
requeue survives as an `outcome=='requeued'` row. This undercounts true
per-task requeue frequency but is consistent and monotonic across both
windows, so a rise between baseline and post is still a reliable churn signal
— just don't read the raw fraction as "N% of tasks requeued".

**Default window lengths and per-metric thresholds** (`CanaryThresholds` /
`--baseline-days`/`--post-days` CLI defaults) are **PRD §9 calibration
starting points** — not asserted numeric guarantees. Tune them against real
`runs.db` baseline variance once you have some; every unit test in
`orchestrator/tests/test_prompt_opt_canary.py` passes explicit thresholds
rather than relying on these:

| Setting | Default |
|---|---|
| `--baseline-days` | 7 |
| `--post-days` | 7 |
| `*_rel_tol` (all four metrics) | 0.2 (flag if post > baseline × 1.2) |
| `cost_per_done_task_abs_floor` | 1.0 |
| `requeue_rate_abs_floor` | 0.05 |
| `mean_review_cycles_abs_floor` | 1.0 |
| `mean_verify_attempts_abs_floor` | 1.0 |
| `min_samples` | 5 (per window) |

The abs-floor value is used **instead of** the relative tolerance whenever a
metric's baseline reads `0` (avoids flagging on a divide-by-near-zero blip).

**Verdict is one of three states**, printed with a per-metric `[PASS]` /
`[REGRESS]` / `[SKIP]` line and a summary; exit code lets a script branch on
it without scraping stdout:

| Verdict | Exit code | Meaning |
|---|---|---|
| `pass` | 0 | every metric within threshold — keep the pin |
| `regress` | 1 | ≥1 metric exceeded threshold — go to Rollback |
| `insufficient_data` | 3 | either window has < `min_samples` rows — **not** evidence either way |
| *(usage error — bad flags, missing `runs.db`)* | 2 | fix the invocation, not a verdict |

`insufficient_data` is common right after a deploy — `run_canary` is a pure
function of `deploy_at`/`baseline_days`/`post_days` and whatever rows already
exist in `runs.db`, so there's no need to wait for the full `post_days`
window to elapse before checking: re-run the **same command** daily as more
post-deploy rows land. The verdict naturally moves from
`insufficient_data` → `pass`/`regress` once both windows clear `min_samples`.
If it's still `insufficient_data` after `post_days` has fully elapsed, either
widen `--baseline-days`/`--post-days` (low pipeline volume) or treat it as
"not enough signal to ship confidence" and keep watching.

---

## Rollback — unpin on regress

If a canary run reports `regress`, unpin **immediately** — this is the sole
rollback lever (D-4); there is no separate revert path to reach for:

```python
from shared.prompt_artifact import PromptArtifactStore, default_artifacts_root

store = PromptArtifactStore(default_artifacts_root())  # same root Step 1 pinned into
store.unpin("reviewer", "<executor_model>", "<harness_version>")
```

`unpin()` returns `True` if a pin was actually removed, `False` if nothing
was pinned for that key (idempotent — safe to call again if you're not sure
it landed). It also prunes now-empty ancestor directories. The **very next**
`resolve()` call anywhere in the live process returns the in-code constant —
again, no restart required.

Re-run the canary command from Step 2 afterward to confirm the regressed
metric(s) recover over a fresh post-unpin window — that's your evidence the
regression tracked the prompt change and not some unrelated pipeline event.

**On the other two verdicts:**

- `pass` — keep the pin. Re-validate (repeat this whole runbook) on every
  model upgrade — an optimized artifact is pinned to
  `(prompt, model, harness)`; a model bump invalidates it (PRD §8 step 5).
- `insufficient_data` — do **not** unpin preemptively. Absence of evidence
  is not evidence of regression; widen the window or wait for more
  post-deploy volume per Step 2, then re-run.

---

## Reference

- PRD: `plans/tier1-prompt-optimization-prd.md` — T8 (§7), D-4 / D-7 (§3),
  operator runbook §8, open calibration questions §9.
- Canary logic: `orchestrator/src/orchestrator/evals/prompt_opt/canary.py`
  (`WindowMetrics`, `CanaryThresholds`, `compare_windows`, `run_canary`,
  `deploy_at_from_provenance`).
- Canary CLI: `orchestrator/src/orchestrator/evals/prompt_opt/__main__.py`
  (`canary` command — `--help` for the full flag list, including per-metric
  threshold overrides not detailed above).
- T1 loader (the rollback lever): `shared/src/shared/prompt_artifact.py`
  (`PromptArtifactStore.pin` / `.unpin` / `.read_provenance`,
  `default_artifacts_root`, `ArtifactProvenance`).
- Tests (synthetic-fixture verdict examples worth reading before your first
  real run): `orchestrator/tests/test_prompt_opt_canary.py`,
  `shared/tests/test_prompt_artifact.py`.
