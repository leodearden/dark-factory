# Entities-omission advice: corpus repair (task 6060)

## What was repaired, and why

`add_memory` rejects some honest `entities` declarations with
`DeclaredReferentConflictRejected`. The cause is a defect in
`fused-memory/src/fused_memory/utils/referent_resolution.py::_conflicting_referents`:
when the content scanner cannot see a declared task, the gate treats that
silence as a contradiction whenever some other task in the content is
scannable. Task 6059 owns the fix.

The gate's hint told rejected agents to drop `entities`. Several agents did so
and then recorded "omit `entities`, which always succeeds" as durable advice in
Mem0. That advice works against `plans/memory-referent-fidelity-prd.md`
decision 4, which tracks the declaration rate as the adoption signal. On a
Graphiti-primary write it also misattributes the memory to whichever task the
scanner could see.

Phase 1 amended each of those records in place. The point id stays the same.
Each record now leads with a dated correction block: the rejection is a defect,
not a rule; spell each declared task as "task NNN" in the content and keep
declaring; the fix is task 6059. Records outside dark_factory cite it only as
`dark_factory:6059`, because the scanner reads a bare task number in a foreign
project as that project's own task. In each body only the omission clause was
replaced, by a pointer to the correction. The rest of the observation is kept.

## The artifacts

- `census.json` is the read-only capture, committed before anything was
  changed. It records the searches that were run in each project, every hit
  that was read and how it was classified, the verbatim pre-image of every
  record, and the auto-memory files that were checked. It is the single list
  of targets: a target is any record whose `disposition` is not
  `reviewed_not_target`.
- `phase1-apply.json` is the mutation record. For each amendment it holds the
  text that was sent, the raw `update_memory` reply and the raw
  `get_memory_by_id` readback. It also holds the census searches re-run
  afterwards, and a readback of every hit those searches returned that the
  census had not seen.
- `fused-memory/tests/test_entities_omission_advice_repair_artifacts.py`
  validates both files and never touches a live store. It checks that every
  target was found by a recorded search, that exactly the census targets were
  amended, and that each amendment landed under the same id. It also runs each
  correction through the production `resolve_referents` to confirm the fix task
  is attributed to dark_factory, and checks that no new record carries the
  correction marker.

## How the amendments were made

A sandboxed implementer has no `update_memory` or `get_memory_by_id` tool.
The driver was a small subclass of
`scripts/migrate_metadata_modules_to_files.py::FusedMemoryClient`, run from the
worktree's gitignored `.task/` directory and never committed. It talked to the
fused-memory server on :8002 as `curator-6060-entities-advice-repair`. A
zero-UUID `update_memory` probe came back `MemoryNotFound`, not a denial, which
confirmed that identity was authorized before any real write. Just before each
amendment was sent, the record was re-read and compared with its census
pre-image.

## Claude Code auto-memory

`census.json` records the auto-memory dispositions under `auto_memory_files`.
The dark-factory files were already corrected. The solar-challenge-platform
file `decompose-filing-gotchas.md` still teaches omission. The sandbox cannot
write `~/.claude`, so that edit was handed to the steward.

## Phase 2

Once task 6059 lands, the corrections should say the defect is fixed. That
work is filed as a follow-up with a dependency on 6059, ticket
`tkt_0RV7E46RG462T68VVYQS0NMDZT`.
