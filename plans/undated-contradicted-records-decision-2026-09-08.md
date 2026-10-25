# Undated-but-contradicted memory records: remit decision (2026-09-08, verified 2026-10-06)

**Task:** 4904 · **Filed by:** task-4610 architect (esc-4610-1) · **Origin hazard:** esc-3578-5
**Verified:** 2026-10-06 against main `12195b4d26`
**Evidence:** `fused-memory/src/fused_memory/services/memory_service.py::MemoryService._search_graphiti`,
`fused-memory/src/fused_memory/models/memory.py::MemoryResult`,
`graphiti_core.edges.EntityEdge` (introspected), `fused-memory/src/fused_memory/services/topic_anchor.py::_UNDATED`,
`orchestrator/src/orchestrator/agents/memory_recall.py::_entry_date`,
`fused-memory/scripts/audit_duplicate_memories.py`, `fused-memory/scripts/read_transform_selection.py`,
`fused-memory/src/fused_memory/server/write_triage.py::triage_write`,
`fused-memory/src/fused_memory/server/write_triage_judge.py::JUDGE_SYSTEM_PROMPT`.

## The question

A memory record that is contradicted by a newer one, but carries no date, cannot
lose a recency tie-break to its successor. Is closing that hazard in the remit of
an existing corpus-hygiene sweep, or does it need new tooling?

## Verdict: neither, and not won't-fix

The premise that this is a corpus-content problem is wrong. The undated property
is a read-path field drop at a single seam. This task closes it with a bounded
read-path correction, shipped alongside this record.

## The mechanism

- `memory_service.py::MemoryService._search_graphiti` built each `MemoryResult`
  without `created_at`, so every Graphiti hit was `created_at: null` by
  construction. `models/memory.py::MemoryResult` documented this as intended.
- `graphiti_core.edges.EntityEdge.created_at` is a required `datetime`. A value
  guaranteed to be present was being discarded.
- The second null is independent. `memory_service.py::_serialize_temporal`
  returns `None` only when an edge has neither `valid_at` nor `invalid_at`.
  Record fb96a8c0 also lacked `valid_at`, which is why it alone presented fully
  undated. Only the Graphiti gap can produce a fully undated record.

## Why it matters

- `topic_anchor.py::_UNDATED` already implements the esc-3578-5 principle (an
  undated record must lose the recency tie-break, not win it by accident), but
  only over Mem0 payloads.
- `memory_recall.py::_entry_date` dated every Graphiti briefing row from
  `temporal.valid_at` alone, so a row lacking it rendered as "undated". That is
  the agent-facing form of the hazard.

## The four candidate hosts

**(a) `fused-memory/scripts/audit_duplicate_memories.py` is not the host**,
because it detects similarity and a contradicting pair is dissimilar by
construction. It is also Mem0-only by its own docstring: a Graphiti edge has no
vector there to cluster on.

**(b) Task 4004's read-transform is not the host**, because that work is
complete and landed as `services/topic_anchor.py` under task 3111. It reads only
`topic` plus `canonical`, and `read_transform_selection.py` states that
`contested` "is not read".

**(c) Write-triage `contested` (`write_triage.py::triage_write`) is not the
host.** Its shipped judge (`write_triage_judge.py::JUDGE_SYSTEM_PROMPT`) does
treat a write saying a candidate is "wrong, outdated or different" as
contesting, under Leo's 2026-09-30 ruling
(`plans/write-triage-flip-readiness-prd.md` §11.3 C1''). It still cannot host
this, because:

- it takes only the NEW incoming `content`, so it cannot adjudicate two
  already-stored records;
- contract C1 forbids it editing a canonical;
- its contested outcome is a Mem0 child record (`PARENT_ID_KEY` + `kind` +
  `CONTESTED_METADATA_KEY`), which cannot attach to a Graphiti edge;
- it is still dark (`fused-memory/config/config.yaml` `write_triage.enabled:
  false`; the flip is owned by task 3169).

Once flipped, it is the right PROSPECTIVE host for a new Mem0 write that
supersedes an older one. That complements this fix and does not replace it.

**(d) New sweep tooling is not the host**, because a sweep would be
retrospective and partial over a symptom. Populating the field at its one seam
is prospective and total over the cause.

## Decision and action taken

`_search_graphiti` now emits `EntityEdge.created_at`, normalized to canonical UTC
through `memory_service.py::_created_at_to_utc_iso`, the helper the episode path
already uses. Canonical UTC keeps lexicographic recency comparison chronological,
which `topic_anchor.py::select_topic_canonical` relies on. `temporal` is left
untouched: no `valid_at` is synthesized from `created_at`. The `MemoryResult`
comment and the `memory_recall.py` placeholder docstring are corrected to match.

No further action. This record launches no corpus-wide sweep and proposes none.

## Honest limitation

`EntityEdge.created_at` is the edge's BIRTH, not its last edit. An edge reworded
in place, as fb96a8c0 was, keeps its original stamp. Briefings now show that
birth date for Graphiti rows where they previously showed `valid_at` or
"undated". It is a tiebreaker, not a last-modified signal.

## What would reopen this

A future need to adjudicate two ALREADY-STORED contradicting records. The
reusable seam is `write_triage_judge.py::build_judge_prompt`,
`write_triage_judge.py::parse_judge_verdict` and
`write_triage_judge.py::JUDGE_VERDICTS`. What is missing is a pair-enumerating
driver and an adjudication surface, both absent today.
