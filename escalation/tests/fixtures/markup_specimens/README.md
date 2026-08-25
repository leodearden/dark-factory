# `markup_specimens/` — the esc-3514 escalation records

Two REAL persisted escalation records that carry leaked MCP envelope markup in
their `detail`, preserved verbatim so the leak can be replayed against the
landed containment code. Consumed by
`escalation/tests/test_markup_specimen_3514.py`; the verification answer they
support is written up in `docs/escalation-markup-write-boundary.md`.

PRD `plans/toolcall-markup-containment-prd.md`. Preserved by task **3643**.

## Never hand-write these files — and why they contain no `\x3c`

Two rules, both load-bearing, both lifted unchanged from
`shared/tests/fixtures/toolcall_markup_corpus.README.md`:

1. **These files are only ever produced by a script.** They were copied and
   re-encoded by a `python3 -` heredoc that read the source path and wrote the
   destination, so the bytes never entered a tool-call argument. Writing them
   through an agent's own Write/Edit call would put envelope literals inside
   that call's arguments, which reproduces the exact defect these records
   document: the harness parser over-consumes at the literal, truncates the
   argument, and silently drops every sibling argument of the same call. That
   is not hypothetical here — it is precisely what happened to the two calls
   preserved below.

2. **Every `\x3c` in the emitted text is escaped as its `\u003c` JSON escape.**
   That is standard JSON — `json.loads` decodes it transparently and the parsed
   value is byte-identical — but it means the file text carries no literal
   opening angle bracket. PRD G6 anticipates a future agent *hand-editing an
   expectation* when the repairer improves; the escaping is what makes that
   edit safe. `test_committed_text_is_escaped_while_the_parsed_value_is_not`
   enforces it mechanically. This README is written under the same rule.

## Provenance

Both records were copied on **2026-08-25** from the live escalation tree at
`/home/leo/src/dark-factory/data/escalations/`, which `.gitignore` excludes —
so before this directory existed, neither record was in git and either could
vanish. That risk was not theoretical: `esc-3514-3` was `status=pending` in the
LIVE queue when task 3643 was first planned, and had been dismissed and moved
into `archive/2026-08-08/` by the time it was preserved. It moved once already.

| File | Source path (under `data/escalations/`) | Source bytes | `agent_role` |
|---|---|---|---|
| `esc-3514-1.json` | `archive/2026-08-03/esc-3514-1.json` | 3958 | `implementer` |
| `esc-3514-3.json` | `archive/2026-08-08/esc-3514-3.json` | 7740 | `harness-orphan-reaper` |

`esc-3514-1` is the **direct producer** record: the leaking session's own
filing. `esc-3514-3` is the **reaper's re-filing** of that orphaned L0, which
propagated the same corrupted `detail` verbatim — the two `detail` strings
share an identical 2812-character prefix, and `esc-3514-3` differs only by a
trailing `[note] originating worktree may be reaped` line. That propagation is
itself part of the finding: nothing between the two filings noticed the markup.

Digests, taken against the SOURCE files before any re-encoding:

| File | `sha256` of source bytes | `sha256(json.dumps(obj, sort_keys=True))` |
|---|---|---|
| `esc-3514-1.json` | `c0231182da2fab09e8ed2652688b646b5e7bcd77083a2461d2840748639070a7` | `a4fbd90ff0371a88c711b82d63f3cbd6ba3bdcae217540cc74db486eb6023299` |
| `esc-3514-3.json` | `cab14ac7e97f097fa5a3e53fe835d63be1ee513cc4c685131656f320c4581664` | `aeaeb3dca1a345ebfe0afd3e9d97db51d760f754cd2100564a8ac022f6429c56` |

The second column is the **parsed-value digest**, and it is the one that
matters. The committed text was re-encoded (pretty-printed, key-sorted, opening
brackets escaped) so its own `sha256` no longer matches the first column by
design; the parsed value did not move, which is what proves the re-encoding was
lossless. The test pins the second column, never the first.

## Why these are not in the existing corpus

`shared/tests/fixtures/toolcall_markup_corpus.jsonl` holds **transcript
payloads** harvested from archived agent transcripts, one record per corrupted
tool call, keyed by `tool_use_id`. These are **persisted escalation records** —
a completely different shape, and one the corpus batch has none of.

That gap was known and recorded rather than discovered here. The module
docstring of `escalation/tests/test_markup_middleware_registration.py` states
that its motivating specimen `esc-3184-2` "CANNOT be a fixture here: `data/` is
gitignored so it does not exist in this worktree". This directory closes that
gap for a different pair of records with the same shape.

The intended downstream consumer is task **3691** (the escalation-corpus
sweep), which needs real on-disk records to develop a discovery predicate
against. Nothing today sweeps `data/escalations/` for this class.

## Measured corruption signature

Measured on the committed fixtures, against the landed
`shared.toolcall_markup`:

| | `esc-3514-1` | `esc-3514-3` |
|---|---|---|
| `len(detail)` | 2812 | 2873 |
| literal `\x3c` in `detail` | 5 | 5 |
| `MCP_MARKUP_PATTERNS` matching | 2 of 3 | 2 of 3 |
| `detect(detail)` | the `parameter` closer | the `parameter` closer |
| stored `suggested_action` | `''` | `'manual_intervention'` |
| stored `evidence` | `[]` | `[]` |
| `level` / `status` | 0 / `dismissed` | 1 / `dismissed` |

> **Correction to an earlier count.** An earlier write-up of these records said
> "3 markup hits". The measured number is **2 of the 3** patterns in
> `MCP_MARKUP_PATTERNS`, with **5** literal opening brackets in `detail`. The
> "3" appears to have conflated the pattern count with the hit count. Use the
> numbers in the table.

### The swallowed values, still legible in the `detail` tail

Both `suggested_action` and `evidence` were dropped from the arguments map and
absorbed into `detail`. They survive there as inert text that nothing reads:

- a real `suggested_action` of **261 characters**, beginning *"Attach these
  observations to DF task 3083 (root cause + retroactive corpus sweep). No
  action needed on task 3514."* and ending *"content is summarized in the
  evidence entries."*;
- an `evidence` list of **three** `{observation, measured_at, ref}` entries,
  every one pinned to `HEAD=860abb2210110deec67355c12b235b8b38f50c77`,
  recording that all three `add_memory` calls were rejected
  (`error_type=McpEnvelopeMarkupWriteRejected`, `field=content`) and that the
  third tripped the storm threshold (`count=3 threshold=3 window_seconds=3600`,
  `escalation_id=esc-markup-tripwire-2`).

Neither value was ever restored to the live queue. The fixtures are the only
place they are now preserved.

### What blocks recovery — the self-referential blind spot

Replayed through `shared.toolcall_markup.repair` with `escalate_info`'s real
parameter set, **both records return `None`** — unrepairable. Under task 3690's
`RepairPolicy.FORWARD_REPAIR` that routes to `_refuse_unrepairable`, so a
filing of this shape today would be REFUSED rather than repaired.

The cause is isolated by a controlled experiment the test performs. Inside the
swallowed `evidence` value the report quotes the very `matched_pattern` that
tripped the memory tripwire — a `content` closer, in prose. `repair()` rejects
any candidate whose parsed tail contains a second mis-close, so that quotation
defeats the tail parser. Replacing **only** that one quoted literal with an
inert placeholder, leaving every other byte untouched, flips the outcome:

    repair(...) -> Repair(recovered={'evidence', 'suggested_action'})
      suggested_action  261 chars, the real recommendation text
      evidence         1016 / 1077 chars, the three entries
      clean_value      1453 chars, detect(clean_value) is None

So the quote is the SOLE blocker. Generalised: **an escalation that REPORTS a
markup leak is the one payload class the repairer structurally cannot
recover**, because a faithful report quotes the pattern. This independently
confirms, on a second pair of records, the "doubly corrupted" PRD boundary row
B5 shape that `test_markup_middleware_registration.py` describes for
`esc-3184-2` and says could never demonstrate a successful recovery.

## The control caveat — do not read `evidence == []` as corruption

Sibling record `esc-3514-2` (same task, `agent_role=orchestrator`, still only
in `data/escalations/archive/2026-08-03/`) is **clean**: zero matching markup
patterns, `detect(detail)` returns `None`, and its `suggested_action` of
`'await_preexisting_main_hotfix'` is intact. It nevertheless ALSO stores
`evidence == []`.

An empty `evidence` list is therefore **not** a corruption signal on its own —
most escalations simply never pass evidence, and a detector keyed on it would
fire on clean records. The discriminating pair is:

1. `detect()` fires on `detail`; **and**
2. `suggested_action` is empty or a bare default **while its real text is a
   substring of `detail`**.

`esc-3514-3` shows why (2) must allow "a bare default" and not just "empty":
its stored value is `'manual_intervention'`, a plausible-looking default, which
is the same loss wearing a disguise. Any future sweep (task 3691) should use
this pair, not the empty list.

## Do not clean, do not regenerate, do not normalise

For future consolidation, janitor, lint and formatting passes:

- **The corruption IS the payload.** These files are specimens under test. A
  pass that "fixes" the markup, fills in `suggested_action`, populates
  `evidence`, or normalises the text destroys the only preserved copy — the
  sources are in a gitignored tree and one of them has already been archived
  out from under an earlier plan.
- **The `\u003c` escaping must be preserved on any edit**, for the reason in
  rule 2 above.
- **The parsed-value digests are pinned** in
  `escalation/tests/test_markup_specimen_3514.py`. Any edit that changes a
  parsed value fails that test, by design. If an edit is genuinely intended,
  re-measure and update the digest deliberately — do not delete the assertion.
- **`repair()` returning `None` is a recorded verdict, not a bug to fix here.**
  If a future repairer improves and the unrepairable pin starts failing, that
  is the intended signal: revisit `docs/escalation-markup-write-boundary.md`
  and update the recorded answer rather than deleting the test.
