# Retrospective sweep: capability-manifest stamp coverage for the toolcall-markup-containment PRD

**Task 4590 · swept 2026-08-28 · dark-factory `b2035cf8c6`**

> **This sweep answers a question, and the answer is "nothing was lost".** Task 4590 was filed
> because `commit_planning`'s capability-manifest stamper silently skips *both* halves of its
> work — the `task_id` stamp on the sidecar and the mechanical `delivered_checks` copy onto the
> producer task — for an entire batch whenever the batch's sidecar fails schema validation. The
> sidecar for `plans/toolcall-markup-containment-prd.md` was confirmed invalid on main for
> roughly 21 hours. The task asked: (a) which producer tasks were planned inside that window,
> (b) what did the stamper fail to write, and (c) hand-backfill it. **The measured answer to (c)
> is that there is nothing to backfill.** That null result is this artifact's deliverable, and it
> is committed as raw measurement rather than asserted, so a third party can re-derive it instead
> of re-litigating it.

Artifacts in this directory:

| file | what it is |
|---|---|
| `report.json` | byte-for-byte the sweep's measurement output, unedited |
| `provenance.json` | run metadata the report itself could not know |
| `investigation.md` | this file — the human adjudication |

## Headline

| item | asked | measured |
|---|---|---|
| (a) labels missing a `task_id` stamp | which producer tasks were skipped? | **0 of 11** — `unstamped_labels: []` |
| (b) tasks missing the `delivered_checks` copy | what did the stamper fail to write? | **0 of 11** — every task's `delivered_checks` name set equals its label's mechanical check names |
| (c) hand-backfill | apply the fix | **empty set — nothing to backfill** |

Two real rows *did* survive the sweep, neither of them item (a) or (b):

- **One live, unowned defect** — three tasks carry a transliterated `prd_task_label`
  (`gamma-1` / `gamma-2` / `gamma-3`) where the sidecar declares `γ1` / `γ2` / `γ3`.
  Remediated by this task; see [The one live defect](#the-one-live-defect).
- **One deferred row** — a single field-level descriptor divergence on label δ, already
  owned by task 4545. See [The deferred row](#the-deferred-row).

## The mechanism

`fused-memory/src/fused_memory/server/manifest_stamping.py::_stamp_capability_manifests_impl`
loads and validates the sidecar in its step 3, inside a
`try` / `except (OSError, yaml.YAMLError, ValidationError)`. That handler appends to
`report['errors']` and **returns** — before step 4 (stamp `task_id` onto the matching label
entry and write the sidecar back) and before step 5 (copy that label's mechanical `grep`/`script`
`delivered_check`s into the producer's `metadata.delivered_checks`).

So a sidecar that fails schema validation does not fail *one* label. It skips both writes for the
**whole batch**, and the only trace is an `errors` entry on a report that nothing was, at the
time, obliged to read. That is the blind spot task 4590 exists to audit — and the one task 4628
closes at source by making `missing_labels` / `errors` non-advisory.

## The window

The window is confirmed independently at both ends, from two sources that do not depend on each
other.

**It opened** at commit `965f3206eb` (2026-08-19T18:39:55+01:00, *"docs(markup-prd): split gamma
three ways; correct the substrate row it passed green on"*). That commit added **5** capability
rows carrying `verdict: OPEN` to the sidecar. At that moment the schema in
`shared/src/shared/capability_manifest.py` still declared `verdict: Literal['PASS', 'FAIL']` —
`OPEN` had never been accepted. All 5 rows are still present today.

**It closed** at commit `5f185fdf00` (2026-08-20T15:40:21+01:00, *"Merge task/4471 into main"*),
which widened the literal to `Literal['PASS', 'FAIL', 'OPEN']`. Note what that merge touched:
`plans/capability-delivered-checks-prd.md`, `shared/src/shared/capability_manifest.py`, and
`shared/tests/test_capability_manifest.py` — **not** the sidecar. The fix was made on the schema
side, which is why the sidecar is byte-identical today to its state during the window, and why
this sweep can measure the exact file that was failing.

Independently, task 4457's own `dry_run_proposals` record the failure verbatim from the other
direction: a merge-verify on 2026-08-20 that names the same 5 `OPEN` rows and the same
`Literal['PASS', 'FAIL']` schema, and correctly diagnoses it as a pre-existing main defect
outside 4457's own diff.

## Why the window cost this PRD nothing

A null result invites the wrong conclusion, so state the right one plainly: **the stamper really
was skipped, and it cost this batch nothing only because the batch never depended on it.**

`git show 965f3206eb -- <sidecar>` shows the operator hand-authored the stamps *inside the split
commit itself*, alongside the new labels:

```
-- label: γ
+- label: γ1
   task_id: 3690
+- label: γ2
+  task_id: 4457
+- label: γ3
+  task_id: 4458
```

`γ1` inherited task 3690's already-stamped `task_id` (an unchanged context line — the label was
renamed, the stamp was not re-derived), and `γ2` / `γ3` arrived with `+  task_id: 4457` /
`+  task_id: 4458` written by hand. The `delivered_checks` on the task side were likewise already
in place. So `commit_planning` had nothing left to do for this batch, and its silent skip was a
no-op *in this instance*.

**This is not evidence that the bug is harmless.** It is one batch that happened to be
hand-stamped by an operator who was editing the sidecar directly. A batch that relied on
`commit_planning` — the normal path — would have lost both writes silently. The generalizable
finding here is about the blind spot, not about this batch's luck.

## The one live defect

Tasks **3690**, **4457** and **4458** carry `metadata.prd_task_label` values of `gamma-1`,
`gamma-2` and `gamma-3`. The sidecar declares those same three leaves as `γ1`, `γ2`, `γ3`.

This matters because of how the stamper matches. Step 4 builds
`label_to_task_id` keyed on each batch task's `prd_task_label`, and matches those keys against
`entry.get('label')` read from the sidecar. A transliterated label matches **no** sidecar entry,
so a future `commit_planning` over these three would stamp nothing, copy nothing, and deposit all
three in step 4b's `missing_labels` — the same failure class this task was filed to audit, still
live today, and reached by a different route.

The Greek form is the convention, not an arbitrary pick:

- The other **eight** labels in this same batch (α, β, δ, ε, ζ, η, θ, ι) all match Greek-to-Greek.
- `skills/prd/references/decompose-mode.md` documents the metadata shape with
  `"prd_task_label": "α"`.

So `gamma-1/2/3` are the deviants. **This task remediates them by realigning the task metadata to
the sidecar** (see [What was changed](#what-was-changed)) — never by renaming the sidecar's
labels, which belongs to task 4545. Task **4628** is the source-level fix that stops the class
recurring, by making `missing_labels` non-advisory at `commit_planning` time; this sweep repairs
the three existing rows, which 4628 does not do.

## The deferred row

Label **δ** (task **3691**), capability check **`committed-evidence-file-survives-the-sweep`**,
field `pattern`: the sidecar and the task record hold two different spellings. The task side
carries the **bracket-free `CANONICAL_OPENER_PREFIX` form**; the sidecar still carries the **raw
`INVOKE_CLOSER` sentinel** — a short, bracket-bearing literal.

**Deferred to task 4545, deliberately.** 4545 names this exact check as item (2) of its own
measured 8-descriptor drift set, states that the task-side spelling is the corrected one, and
declares this sidecar in its `metadata.files`. This task holds no lock on that file and does not
declare it. Fixing it here would contend with in-flight work to reach the same end state.

**Neither literal is quoted anywhere in this artifact set, by design.** Writing a raw toolcall
envelope sentinel into a tracked file forces the authoring agent to emit it inside its own tool
call, reproducing the very leak the toolcall-markup PRD exists to contain, and corrupting the
file being written. It is also rejected at the wire (`update_task` / `submit_task` return
`error_type=mcp_markup_detected` unless `allow_mcp_markup` is set). `report.json` therefore
records the divergence as **measured properties** — length, angle-bracket and slash presence, and
a `sha256` prefix for each side — which is reproducible by a third party without either value
ever being written down. Verified after generation by hashing every substring of both artifacts:
neither literal appears.

## What was changed

The one remediation this sweep applied. It is recorded here because a task-store write leaves no
diff — these three files are its only durable trace.

| task | `metadata.prd_task_label` before | after | status |
|---|---|---|---|
| 3690 | `gamma-1` | `γ1` | done |
| 4457 | `gamma-2` | `γ2` | done |
| 4458 | `gamma-3` | `γ3` | done |

Applied as three `update_task` calls passing **only** `{"prd_task_label": "<greek>"}`, inheriting
the default shallow last-write-wins metadata merge so every sibling key survives. The whole
metadata blob was deliberately *not* sent back: this repo has a documented class of whole-blob
metadata writers clobbering sibling keys (tasks 3260, 3933, 4507).

**Verified rather than assumed**, both ways:

- Re-read all three records afterwards and byte-compared *every* metadata key against the
  pre-write capture. `keys_changed` is exactly `["prd_task_label"]` on all three; key counts are
  unchanged (21 / 21 / 20); `delivered_checks` and `prd_path` are byte-identical. Zero sibling
  keys clobbered.
- Re-ran the corpus-wide label sweep. Unmatched labels fell **24 → 21**, a delta of exactly −3,
  clearing exactly `{3690, 4457, 4458}` with **no** row newly appearing. A larger drop would have
  meant something else moved, and would have stopped this step.
- The unrelated findings are unchanged: `unstamped_labels` still `[]`, `missing_delivered_checks`
  still `[]`, and the δ/3691 descriptor divergence still present and still deferred.

All three tasks are `done`, and the change is still safe: `prd_task_label` is provenance-only — a
blessed key in `shared/src/shared/task_metadata.py` whose sole functional consumer is
`manifest_stamping` (verified by grep across every `*.py`). Nothing re-reads it to make a decision
about a completed task.

See `report.json` → `remediation` for the full before/after record.

## Reproducing

Entirely read-only; the measurement makes no writes.

1. Open the authoritative task store read-only:
   `sqlite3.connect('file:/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db?mode=ro', uri=True)`,
   tag `master`. `.taskmaster/tasks/tasks.json` is **stale**; `data/tasks.db` and
   `data/orchestrator/tasks.db` have no `tasks` table. `metadata` is plain JSON text.
2. Build the manifest-bearing set exactly as the stamper's step 1 does: a task qualifies only if
   its metadata carries **both** a non-empty `prd_path` and `prd_task_label`.
3. Derive the sidecar with the stamper's own substitution —
   `re.sub(r'\.md$', '', prd_path) + '.capability-manifest.yaml'`. **Do not** bind via
   `metadata.capability_manifest`: measured, tasks 4457 and 4458 both point that key at the
   `.capability-manifest.md` twin, not the `.yaml` the stamper reads, so binding by metadata
   answers the wrong question.
4. Compare only mechanical checks (`kind` in `grep`/`script`); `manual` is never copied.
5. Normalize `None` against `[]` for the optional list fields (`paths`, `args`) before declaring
   a difference. The task side is a `DeliveredCheckMeta(...).model_dump()` of the sidecar side,
   and that round-trip materializes defaults — a raw comparison reports spelling differences as
   real drift.
6. Enumerate the corpus with `git ls-files '*.capability-manifest.yaml'` (53 tracked sidecars at
   this base).

**This is a one-shot audit, not a detector.** Once task 4545 lands,
`scripts/audit_manifest_descriptor_drift.py` is the standing mechanism for the descriptor half of
this comparison, and should be used in preference to re-running anything here. No script was
added by this task, deliberately: 4545 already implements that sweep over the same corpus, and
task 4782 owns the `scripts/_task_db_scan.py` consolidation that a fourth audit script would
contend with.

## Coverage — what this sweep did not see

- **It audited one PRD's sidecar for items (a) and (b).** The null result is scoped to
  `plans/toolcall-markup-containment-prd.capability-manifest.yaml`. It says nothing about whether
  any *other* batch lost writes to the same blind spot; no other sidecar's invalid window was
  reconstructed.
- **It cannot see task state at any past moment.** Every measurement is of the store *as of the
  run* (2026-08-28). The claim "nothing was lost" is inferred from present-day completeness plus
  the commit evidence that the stamps were hand-authored — not from a snapshot taken during the
  window. Had something been lost and later repaired by hand, this sweep would report the same
  null.
- **The corpus-wide unmatched-label rows are recorded but not adjudicated.**
  `report.json` → `corpus_unmatched_labels` carries all **24** manifest-bearing tasks whose
  `prd_task_label` matches no label in their derived sidecar. Three are this PRD's γ cases, fixed
  above. The other **21** are at least three distinct classes and are *not* one defect: ~15 on
  `plans/eval-framework-revival-prd.capability-manifest.yaml` and 4 on
  `plans/found-on-main-provenance-integrity-prd.capability-manifest.yaml` (early-Greek tasks
  against sidecars declaring only a later wave), plus 2 non-Greek labels (3289 `restore-gate`,
  4172 `kappa-followup`). They need per-class adjudication by someone holding the relevant PRD
  context, not a blanket rename, and this task holds no lock on those sidecars.
- **The δ descriptor divergence is measured but not fixed** — see
  [The deferred row](#the-deferred-row).
