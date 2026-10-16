# Sidecar-orphan reconciliation, 2026-09-29 (task 4935)

This investigation covers the checked-in capability-manifest sidecar capabilities that no task's
`metadata.delivered_checks` ever received. The runtime gate evaluates only what is stamped in
metadata, so an unstamped sidecar check gates nothing and its producer's dependents dispatch
ungated. Task-store writes leave no repo diff, so this directory is their only durable trace. The
per-row record is `ledger.json`. The layout follows task 4590's
`docs/manifest-stamp-coverage-audit-2026-08-28/`.

Measured against dark-factory main `7bb8138c40` and reify main `9663f2a5dc`.

## Headline

| | rows | what happened |
|---|---|---|
| Sidecar-only rows in the baseline audit | 60 | sources: 25 delivered, 14 healthy, 15 no_task, 1 superseded, 5 vacuous_live_gate |
| Stamped onto an open producer | 22 | 8 dark-factory (3131, 3140, 5382, 5387 ×2, 5389, 2890, 2892), 14 reify (7334 ×4, 7335 ×8, 5613, 5614) |
| Stamped after the sidecar descriptor was re-pointed | 1 | 3136 `timer-unit-committed` |
| Sidecar descriptor converted to `kind: manual` | 4 | 3139, ρ1/ρ2/ρ3 preservation and precondition facts |
| Not stamped: producer done | 25 | the design decision below |
| Not stamped: superseded | 1 | 3141 `markup-rejection-with-named-pattern` |
| Not an orphan (audit phantom) | 2 | 2862, 4681: the audit bug is fixed here, and the drift belongs to 6036 |
| Owned by another task | 2 | 5034 → 5446 (write reverted, see below), 5754 → 6036 |
| Deferred, producer in flight | 2 | 5384 ×2 (in-progress) |
| Foreign block pending `external_task_id` | 1 | reify 5616 κ, the follow-up after 4731 |

Every write was checked by a read-only readback of the store it was made to. The readback
confirmed that `delivered_checks` equals the written list and that every other metadata key is
byte-identical to the pre-write capture.

## What the audit could not see, and what changed in it

`scripts/audit_delivered_checks.py` changes in two ways:

1. **Phantom fix.** Sidecar rows were deduplicated against the metadata GREP rows only, so a
   sidecar grep was re-reported as an orphan whenever its producer already carried the same
   capability as a `kind: path` check. The two measured cases were 2862
   `eval-bootstrap-smoke-gate` and 4681 `predicate-variant-script-exists`. Deduplication now keys
   on the stamped `(task_id, name)` set of every kind, read in the same single status read. That
   is what the dedupe docstring always claimed: the metadata copy wins.
2. **`unwired_live_gate`.** Before this change, a sound, failing (forward-looking) sidecar check
   on a live producer that was never stamped was classified `healthy`. The text report never
   prints healthy rows, so the class this task reconciles was invisible to the tool meant to find
   it. It is now its own actionable disposition, with its own LIVE section, exit 1, and
   `open_dependents`. It is set by a pure `stamped` keyword that changes only the live+FAIL cell
   of `classify_descriptor`.

## The seven classes and the adjudication policy

1. **Phantoms** (2862, 4681). The metadata already carries the check as `kind: path`. This is the
   audit bug above; the descriptor drift itself belongs to task 6036.
2. **Foreign-registry producers.** ρ1–ρ3 (reify 7334–7336) and warm-lane η–κ (reify 5613–5616)
   are reify tasks filed against dark-factory PRDs. Reify's `commit_planning` stamper refuses a
   `prd_path` that resolves outside its project root, and these tasks carry an absolute
   dark-factory `prd_path`, so nothing was ever copied. In dark-factory's audit, ρ reads as
   `no_task`. η–κ **misjoin** to unrelated dark-factory tasks 5613–5616, because the ids collided
   once dark-factory passed 5613. That misjoin produces false `vacuous_live_gate` and
   `unwired_live_gate` rows.
3. **Open dark-factory producers with a lint-clean descriptor.** These were stamped. The
   memory-write-path batch (3127–3142) never carried `prd_path`/`prd_task_label`, so the stamper
   could not bind it. The live-shadow and stranding sidecars gained rows after stamping.
4. **Open producers whose descriptor the polarity lint rejects.** These are never stamped by
   hand; `update_task` runs no lint, so doing so would install exactly the gate the task-3500 lint
   exists to refuse. The fix goes in the SIDECAR:
   - 3136 κ (`vacuous_present`: `OnCalendar` over `scripts/` already matched six timers) was
     re-pointed to the timer file 3136's own `metadata.files` declares,
     `scripts/fused-memory-duplicate-audit.timer`. It then lints clean and was stamped.
   - 3139 ν (`filename_shaped`) was converted to `kind: manual`. The task names no validation
     symbol to re-point to.
   - ρ1 `the-detector-block-and-its-census-entry-are-untouched`, ρ2
     `partof-and-the-canary-timer-survive` and ρ3 `held-back-is-observable-for-the-signal`
     (`vacuous_present` at reify main) are preservation or precondition facts that a present-grep
     can never gate. They were converted to `kind: manual` with name, binding, verdict, task_id
     and note unchanged.
   - 5754 lives in a sidecar owned by 6036 and was left. κ 5616 lives in 4731's sidecar and was
     left for the follow-up.
5. **Open reify producers with a descriptor lint-clean at reify main.** 7334, 7335, 5613 and 5614
   were stamped through `update_task(project_root=/home/leo/src/reify)`, after confirming that
   each reify row's `prd_path` names the dark-factory PRD and that its `prd_task_label` equals the
   sidecar label.
6. **Done producers whose check is delivered on main.** These were not stamped (see the next
   section).
7. **Done and superseded** (3141). Not stamped: stamping would hold dependents on a check that
   later work legitimately undid.

Every entry was built exactly as `manifest_stamping.py` step 5 builds it:
`DeliveredCheckMeta(...).model_dump()`, linted with
`lint_delivered_checks([{**meta, 'manifest_path': rel}], files=<the task's metadata.files>,
repo_root=<the producer's project>, ref='main')`. The write was
`update_task(metadata={'delivered_checks': <existing list verbatim + new entries>},
metadata_mode='merge')`.

## Why done producers were not stamped

A done producer has already passed its mark-done gate, so a stamped check has exactly one runtime
effect: `deps_satisfied` re-evaluates it each time an open dependent is dispatched. If later work
legitimately undoes the pattern, that dependent is held and the delivered-check grace streak
escalates to L2. The audit's own SUPERSEDED class shows this happens. Several of these producers
have open dependents today (3128→3131/3169, 3129→4010, 3133→4453/4454, 3135→4493/5881,
3195→3202, 3081→3077), so stamping them would buy regression-wedge exposure and no gating
benefit. They stay visible as `delivered` rows from `source=manifest`.

## Things the plan did not anticipate

- **5034 `reachback-patch-guard-retired` was stamped and then reverted within 47 seconds.** The
  update response showed `x_reachback_guard_check_moved_to: "5446"` in 5034's metadata. Leo's
  ruling esc-5024-2 / esc-5025-2 suspended the guard deletion for 5034 and moved the check to
  5446, which carries it. Stamping it on 5034 would hold 5034 on a deletion it is forbidden to
  perform. 5034's metadata was restored, and the readback shows it string-identical to the
  pre-write capture. The sidecar δ block still binds the capability to 5034, so the audit keeps
  listing it as unwired until that block is re-homed (esc-4935-4). The lesson for the next
  reconciliation: read the producer's `x_*` metadata for a moved-check marker before writing.
- **2890 and 2892 are deferred and coalesced into 5311** (`x_coalesced_into`). Their stamps are
  harmless but moot, and 5311 carries no `delivered_checks` (esc-4935-5).
- **The ρ1–ρ3 `note:` text is now stale.** It says reify copied no checks and that they "are NOT a
  live dispatch gate". ρ1 and ρ2 are now stamped. The plan froze those notes byte-identical, so
  the refresh is part of the follow-up.
- The memory-write-path `.md` twin shows κ's verdict as PASS, while the sidecar has carried FAIL
  since the 2026-08-30 D18 ruling. This drift pre-dates this task and was not touched.

## After-measurement

**dark-factory**, run with the branch script against main's sidecars. The step-5 sidecar edits
become visible only after this branch merges. Orphan rows fell from 60 to 49: 24 delivered,
15 no_task, 5 unwired_live_gate, 4 vacuous_live_gate, 1 superseded. The phantoms are gone. The
nine stamped rows now read from `source=metadata`. What remains:

| row | why it remains | clears when |
|---|---|---|
| ρ1/ρ2/ρ3 ×15 `no_task` | foreign ids | three become manual at merge; the rest when the follow-up rebinds them to `external_task_id` |
| 3139 `unwired_live_gate` | main's sidecar still spells it as a grep | this branch merges |
| 5034 `unwired_live_gate` | the check lives on 5446 | the sidecar δ block is re-homed |
| 5384 ×2 `unwired_live_gate` | the producer is in-progress | the producer leaves its workflow; re-run and stamp |
| 5614 `unwired_live_gate`, 5613/5615/5616 `vacuous_live_gate` | the warm-lane misjoin | 4731 merges |
| 5754 `vacuous_live_gate` | owned by 6036 | 6036 merges |

**reify.** Every row this task stamped reads `source=metadata`, disposition `healthy`. The run
also surfaced reify's OWN sidecar orphans: 14 `unwired_live_gate` rows (5285, 5312, 5313, 5482,
6689, 6691, 6692, 6693, 6701, 6702, 6804), plus 6 broken and 9 vacuous sidecar-only rows. They
are listed in `ledger.json` under `after.reify` and are out of this task's scope. They are for
reify's owner.

## Follow-up

Ticket `tkt_0RV7072RRXN8Z0HTZJTPDSFFBH` (escalation id `agent-followup-4935`, depends on 4731).
Once 4731 lands, it will rebind ρ1–ρ3 and os-sandbox γ7a–e to their registries through
`external_task_id`, refresh the stale ρ notes, and adjudicate warm-lane κ's `vacuous_absent`
descriptor.

## Reproduce

```bash
uv run --project shared python scripts/audit_delivered_checks.py \
    --project-root /home/leo/src/dark-factory --json   # about 5-6 min, exit 1 on pre-existing defects
uv run --project shared python scripts/audit_delivered_checks.py \
    --project-root /home/leo/src/reify --json          # about 5 min
```

The orphan subset is every finding with `source == "manifest"` or `disposition == "no_task"`.
