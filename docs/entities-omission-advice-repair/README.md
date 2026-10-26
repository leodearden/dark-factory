# Entities-omission advice: corpus repair (tasks 6060 and 6067)

## What was repaired, and why

`add_memory` rejected some honest `entities` declarations with
`DeclaredReferentConflictRejected`. The cause was a defect in
`fused-memory/src/fused_memory/utils/referent_resolution.py::_conflicting_referents`:
when the content scanner could not see a declared task, the gate treated that
silence as a contradiction whenever some other task in the content was
scannable. Task 6059 fixed it, in merge commit `1ac3835e85` on 2026-10-18.

The gate's hint told rejected agents to drop `entities`. Several agents did so
and then recorded "omit `entities`, which always succeeds" as durable advice in
Mem0. That advice works against `plans/memory-referent-fidelity-prd.md`
decision 4, which tracks the declaration rate as the adoption signal. On a
Graphiti-primary write it also misattributes the memory to whichever task the
scanner could see.

Phase 1 amended each of those records in place. The point id stays the same.
Each record then led with a dated correction block: the rejection is a defect,
not a rule; spell each declared task as "task NNN" in the content and keep
declaring; the fix is task 6059. Records outside dark_factory cite it only as
`dark_factory:6059`, because the scanner reads a bare task number in a foreign
project as that project's own task. In each body only the omission clause was
replaced, by a pointer to the correction. The rest of the observation is kept.

Phase 2 replaced that correction block in the same records, again in place.
The new block says the defect is fixed, cites 6059's merge commit and date,
and names the one residual (see "Phase 2" below). The body is unchanged.

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
- `phase2-apply.json` is the phase-2 mutation record, in the same shape. Each
  amendment also holds the pre-read of its target. The file adds the
  authority probe's arguments and raw reply, and the measured fix commit
  under `fix_task`. It holds the same re-run of the census searches and
  readbacks of their new hits.
- `fused-memory/tests/test_entities_omission_advice_repair_artifacts.py`
  validates both files and never touches a live store. It checks that every
  target was found by a recorded search, that exactly the census targets were
  amended, that each amendment landed under the same id, and that each body
  still carries the original observation. It also runs each correction
  through the production `resolve_referents` to confirm the fix task is
  attributed to dark_factory. Finally it checks that the re-run repeated every
  census search, and that no new record carries either phase's correction
  marker. For phase 2 it also checks that the phase-1 block was replaced and
  the body kept verbatim, and that the cited commit is 6059's merge, read from
  git. The `resolve_referents` checks run only against the current (phase-2)
  text, because the phase-1 text is no longer in the store.

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

Phase 2 used a driver of the same shape, under the identity
`curator-6060-phase2-entities-advice-repair`. Its zero-UUID probe also came back `MemoryNotFound`. Every target was read
before any write, and each had to equal its phase-1 text exactly, or nothing
would have been written.

## Claude Code auto-memory

`census.json` records the auto-memory dispositions under `auto_memory_files`.
The dark-factory files were already corrected. The solar-challenge-platform
file `decompose-filing-gotchas.md` still teaches omission. The sandbox cannot
write `~/.claude`, so that edit was handed to the steward.

In phase 2 the two auto-memory files named by task 6067 were handed to the
steward the same way, via `escalate_blocker(scope_violation)`:
`/home/leo/.claude/projects/-home-leo-src-dark-factory/memory/reference_scripted_fused_memory_write_client.md`
and
`/home/leo/.claude/projects/-home-leo-src-solar-challenge-platform/memory/decompose-filing-gotchas.md`.

## Phase 2

Task 6067 ran after 6059 landed. The phase-1 validator was re-run first, on the
post-6059 scanner, and passed (14 tests, 2026-10-18). Its two
`resolve_referents` checks now run against the current text only.

6059 did not fix one case, and the phase-2 correction says so. Prose of the
form "word: NNNN" is still scanned as a project-qualified referent, so
declaring that same number beside it is still rejected. Records `d3d753ef` and
`2be6b8c3` describe exactly this case. It is tracked as task 6826.

The census searches re-run in phase 2 also found records written after the
census that still teach omission. `census.json` is the single list of targets,
so they are out of phase 2's scope. They were filed as follow-up ticket
`tkt_0RVYG32127ZF9JAN9XCV4ZKFB4`.
