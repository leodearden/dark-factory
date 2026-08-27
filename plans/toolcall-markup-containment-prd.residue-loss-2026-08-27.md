# Residue-loss evidence, 2026-08-27

Companion to `toolcall-markup-containment-prd.residue-loss-2026-08-27.jsonl` —
five refused `add_memory` calls whose payloads the markup guard could **not**
file, recovered from journald before its ~72h retention rolled them out.

## Why this is committed

PRD §4 contract C2 and boundary row **B5** both promise that an unrepairable
call is *"rejected, and an escalation filed carrying the full raw payload so
nothing is discarded even if the caller never retries."* For these five, no
escalation was filed: `fused_memory.server.markup_guard::_file_residue`
resolved `project=None`, so `_resolve_project_root` returned `None`, and the
record went to a log line instead of the queue. These records are the primary
evidence that C2/B5 do not hold for that class, and they lived only in
gitignored `data/` on one machine.

## What the records show

| Observed (local) | Payload identity | Absorbed calls | Filed? |
|---|---|---|---|
| Aug 25 03:03:43 | `dark_factory` / `claude-task-4647-implementer` | 1 | no |
| Aug 25 19:42:21 | `dark_factory` / `claude-architect` | 2 | no |
| Aug 27 00:10:46 | `reify` / `claude-task-6751-architect` | 2 | no |
| Aug 27 00:58:01 | `reify` / `claude-task-6810-architect` | 1 | no |
| Aug 27 09:49:11 | `dark_factory` / `claude-task-4696-steward` | 1 | no |

Five refused calls, **seven** intended memory writes: two payloads each absorbed
a whole second `add_memory` call.

Note the two identity columns in the JSONL. `guard_resolved_identity` is what
the guard saw — `null` in every case. `identity_present_in_payload` is what a
scan of the record's own `raw_value` recovers — a real `project_id` in **5 of
5**, each resolving against the live 9-project registry. The routing key was in
the guard's hand the whole time.

## Cause, in one line

The leak over-consumes *forward* from the leaking parameter's closing tag, so it
absorbs every parameter declared **after** it. `add_memory(content, project_id,
…)` declares `content` first, so `project_id` is always eaten. `submit_task(
project_root, …, description, …)` declares `project_root` first, so it survives
— which is why a `submit_task.description` refusal at 2026-08-27 18:15 preserved
its payload fine as `esc-markup-residue-2` while all five of these were lost.
Whether a refused payload survives currently depends on parameter declaration
order.

## Encoding — read before editing

Every opening angle bracket is written as its six-character JSON unicode escape
(backslash-u-0-0-3-c), matching
`shared/tests/fixtures/toolcall_markup_corpus.jsonl`. The file therefore
contains **zero** literal opening-angle-bracket bytes and cannot trip the
detector, the tripwire, or `scripts/sweep_toolcall_markup.py`. `json.loads`
restores the payloads byte-for-byte (round-trip asserted at generation).

Keep it that way: regenerate through `json.dumps(..., ensure_ascii=True)`
followed by the same escape substitution rather than hand-editing, or this file
becomes a specimen of the very leak it documents.

This is **evidence, not a repairer fixture** — it is deliberately not wired into
the corpus replay in `shared/tests/test_toolcall_markup_corpus.py`, whose
committed expectations are a separate population.

## Disposition of the payloads

All seven writes were re-landed by their own authoring agent, median under a
minute (+6s, +16s, +23s, +51s, +58s, +68s, +11m57s). Nothing was permanently
lost. The one slow recovery came back with roughly a third of its claims gone,
its `agent_id` changed and its category demoted — so the residual harm is
content attrition that scales with retry latency, not outright loss. That says
the agent retry loop, not this preservation path, is what actually saved the
data; the path C2 relies on contributed nothing.
