# `decision_pairing_corpus.jsonl`

Real **semantically cross-paired** `design_decisions` entries, harvested from
live task plans, used to pin `shared.decision_pairing.detect_mispairing`
against regression — and, just as load-bearing, a set of hand-picked
**negative controls** that pin its *precision*.

Task 3967. Replayed by `shared/tests/test_decision_pairing.py`
(`TestCommittedCorpus`) and `shared/tests/test_decision_pairing_containment.py`.

## The damage class

A mis-paired entry holds a `decision` and a `rationale` that are each perfectly
well-formed prose — only their *association* is wrong. Nothing is malformed, no
sentinel is present, no parser is upset. `shared.toolcall_markup.detect` returns
`None` on 34 of the 46 strings collected here. That is why this corpus exists
separately from `toolcall_markup_corpus.jsonl`: the two damage classes are
disjoint at the detector, and no literal-keyed predicate can see this one.

The only machine-visible trace is the **correction entry an author appends after
noticing** — a later `design_decisions` entry that opens on a correction header
and says, in words, that a preceding entry was mis-paired. That self-documenting
correction is what the predicate keys on, and it is why every count derived from
it is a **strict lower bound**: a mis-pairing nobody noticed leaves no trace at
all.

## Why this file is committed

Both sources are gitignored — `.worktrees/` (the plans) and `/data/` (the
transcripts). The plans additionally churn: worktree lanes are reset and reused.
So the committed corpus is the durable artifact, exactly as
`plans/toolcall-markup-containment-prd.md` section 11.4 resolved for the
envelope corpus. Nothing in the replay tests reads the live tree.

## Angle brackets — DO NOT "helpfully" un-escape these

Every `\x3c` in the harvested text is stored as its six-character `\u003c` JSON
escape, so the **file text carries no literal opening angle bracket**. That is standard JSON —
`json.loads` decodes it transparently and the parsed value is byte-identical —
and it is what makes this file safe for an agent to hand-edit later. Writing one
verbatim would put an envelope literal inside that agent's own tool-call
arguments, reproducing the defect the sibling corpus documents. The generator
asserts `'\x3c' not in file_text` before it writes, and round-trips the parse.

Three specimens genuinely carry envelope literals in their harvested text
(`3201[1]`, `3382[1]`, `3209[8]`), which is precisely why the rule is
unconditional rather than "if it looks like markup".

## Record schema

One JSON object per line, ten keys, all present on every record:

| Key | Type | Meaning |
|---|---|---|
| `task_id` | str | The task whose plan the entry was harvested from. |
| `index` | int | 0-based index into that plan's `design_decisions`. |
| `field` | str \| null | Which field the PAIRING marker matched in — `decision` or `rationale`. `null` on every negative control, because nothing matched. |
| `text` | str | The entry's `decision` text, possibly truncated (see below). |
| `rationale_text` | str | The entry's `rationale` text, possibly truncated. |
| `expect` | str | `mispaired` or `clean` — the verdict `detect_mispairing` must return. |
| `envelope_clean` | bool | `true` when `toolcall_markup.detect` returns `None` on BOTH stored fields. Consumed by the containment test's detector-blindness assertion. |
| `near_miss_class` | str \| null | On a `clean` record, WHICH near-miss class it controls for. `null` on positives. |
| `truncated` | bool | Whether either stored text is shorter than the harvested one. |
| `original_lengths` | dict | `{"decision": int, "rationale": int}` — harvested lengths, before truncation. |

`(task_id, index)` is the record's identity and its sort key.

### Why a record carries BOTH fields

The task's plan sketched the schema as `{task_id, index, field, text, expect}`.
It is spelled with an extra `rationale_text` because the predicate is a
conjunction *across* the two fields — the pairing conjunct searches `decision`
**or** `rationale` — so a single-field record cannot replay it at all.
`envelope_clean` and `near_miss_class` are likewise present because the
containment test and the four-near-miss-class assertion consume them directly.
Every key the plan named is kept, spelled as it named it.

### Measured: the `rationale` half of the disjunction has NO live specimen

The task's plan asserted that specimen `3727[1]` "splits the evidence" by
carrying its header on `decision` and its pairing language on `rationale`.
**That is refuted by measurement.** All 23 positives carry their pairing marker
in `decision`; `field` is `decision` on every single one. Zero carry it only in
`rationale`, and zero carry it in both. `3727[1]` in particular reads
`CORRECTION of the mis-titled entry above: the preceding rationale belongs to
THIS decision` — header and pairing language are in the same field.

The disjunction is implemented anyway, exactly as the plan specifies, and it is
pinned by a hand-authored specimen in `TestDetectMispairing` rather than by any
corpus record. It is deliberate **recall headroom** for a correction phrased
with a bare header and its pairing language deferred to the rationale — a shape
that is plausible and cheap to accept, but which nobody has yet written. Do not
read the corpus as evidence that it occurs.

## Collection predicate

Stated exactly, so the numbers below are reproducible. An entry matches when
**both** conjuncts hold:

1. its `decision` is **start-anchored** (leading whitespace tolerated,
   case-insensitive) on one of

       CORRECTION | RESTATEMENT | READ THIS INSTEAD | SUPERSEDES | CORRECTED

2. **and** one of these pairing phrases appears anywhere in `decision` **or**
   `rationale`, matched case-insensitively and with each hyphen optional
   (`mispaired` and `mis-paired` both count — two live specimens use the
   unhyphenated spelling):

       mis-paired | cross-paired | mis-titled | mis-attributed
       recorded against the wrong | swapped | belongs to THIS decision

Dropping either conjunct costs real specimens. Without the start-anchor, three
`3209` entries that merely use the word `supersedes` mid-sentence for a metadata
edge kind are swept in. Without the pairing conjunct, `3382[5]` — a genuine
design reversal opening `SUPERSEDES decision #3` — is swept in. Both classes are
committed here as negative controls.

The predicate lives in code exactly once, in `shared/src/shared/decision_pairing.py`,
and this corpus is re-findable with the committed scanner:

    python3 scripts/scan_plan_decision_pairing.py --root .worktrees/.task-meta --json

## Negative controls, and what each one controls for

These are what make this a **precision** pin rather than a recall pin. Every one
is real observed text, stored **verbatim** — never truncated, because the whole
point is the discriminating context.

| Record | `near_miss_class` | Why it must NOT fire |
|---|---|---|
| `3382[5]` | `genuine-supersession-no-pairing-language` | Opens `SUPERSEDES decision #3` and really does supersede it — but it is a design REVERSAL, not a mis-pairing. Header conjunct holds; pairing conjunct does not. |
| `3209[1]` | `incidental-mid-prose-keyword` | Uses `supersedes` mid-sentence as the name of a metadata edge kind. |
| `3209[8]` | `incidental-mid-prose-keyword` | Same, and additionally carries an envelope literal — the one record that is a negative for THIS class while being a positive for the envelope class. |
| `3209[9]` | `incidental-mid-prose-keyword` | Same, via the identifier `normalize_supersedes`. |
| `3298[2]` | `parenthetical-restatement-not-headed` | Says `(restatement of decision #1 with its correct rationale)` — mid-prose, parenthetical, not a header. A start-anchor is the only thing separating it from a positive. |
| `3692[8]` | `meta-prose-about-another-plans-mispairing` | Prose ABOUT task 3567's mis-pairing, containing the phrase `recorded against the wrong` verbatim in its rationale. This is the false positive the originating task description acknowledged; the start-anchor is what sheds it. |

## Truncation rule

Applied unconditionally **by rule**, so the output is deterministic:

* A **negative control** is stored verbatim, always.
* A **positive** field longer than 400 characters is stored as its first 400
  characters, extended if necessary to cover the matched pairing phrase plus 60
  trailing characters. `truncated` is `true` and `original_lengths` records the
  harvested lengths.
* The generator **refuses** a truncation that changes either verdict — the
  pairing verdict, or the `envelope_clean` measurement — and stores the value
  verbatim instead. Otherwise the corpus would pin an outcome the real text
  never produced. This ground is live: several rationales carry their envelope
  literal as a trailing fragment, which truncation would silently remove.

## Provenance (NOT assertions)

Every figure below is a measurement of one snapshot. **No test asserts any of
them**, and none may: `.worktrees/.task-meta` grew from 1,196 to 1,297 plans in
about eight days, new victims are still landing, and pinning a count would make
a *better* predicate read as a regression. Tests assert only agreement with this
file's own `expect` column and against synthetic `tmp_path` trees.

| | |
|---|---|
| Extraction date | 2026-08-16 |
| Source root | `/home/leo/src/dark-factory/.worktrees/.task-meta/*/plan.json` |
| Plans scanned | 1,297 |
| Plans unreadable | 0 |
| Victim plans | 20 |
| Matched entries | 23 |
| Victim task ids | 3042, 3098, 3201, 3209, 3210, 3216, 3298, 3337, 3363, 3382, 3415, 3473, 3567, 3664, 3668, 3727, 3757, 3918, 4030, 4096 |
| Negative controls | 6 |
| Records committed | 29 |
| Records truncated | 12 |
| Records fully envelope-clean | 16 (13 of 23 positives, 3 of 6 negatives) |
| Positive strings with NO envelope literal | 34 of 46 |
| File size | 42,455 bytes |

The 34-of-46 figure is the one that answers the containment question: even once
the write-time markup tripwire is registered on the servers, it decides by
running `detect` over each incoming string, so it admits the majority of these
outright. That is measured, not argued — see
`shared/tests/test_decision_pairing_containment.py`.

Newest victim plan mtimes run to 2026-08-15, and task ids `4030` and `4096` are
both ABOVE task 3967's own, i.e. they were authored after this defect was filed.

## If the predicate legitimately improves

Regenerate or hand-edit **in the same commit** as the change, updating the
`expect` column for any record whose verdict moved and adding a row to the
negative-control table for any new near-miss class. That edit is a reviewable
improvement; a widened predicate that silently drops a specimen is not, which is
why the replay asserts every one of the 20 victim task ids is still present.

**Never add a per-specimen exception to the detector.** Widen or tighten the
marker sets, which live in exactly one place. A per-specimen carve-out is the
drift the single-owner literal set exists to prevent.

## Companion fixture

`decision_pairing_wire_evidence.json` holds task 3727's 7 on-disk
`(decision, rationale)` pairs beside the 7 `add_design_decision` tool-call
inputs recovered from its archived transcripts. All 7 are byte-identical,
including the mis-paired entry `[0]` that entry `[1]` corrects — so plan-tools
wrote exactly what it received, and the cross-pairing was already present in the
arguments the model composed. See that test's docstring for the scope limit:
tasks 3567 and 4096 do NOT reproduce byte-identity on every field and are
inconclusive, so no wire record for either is committed.
