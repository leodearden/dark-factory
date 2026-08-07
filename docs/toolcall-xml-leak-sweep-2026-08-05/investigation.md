# Toolcall-XML leak: live Mem0/Qdrant corpus sweep, 2026-08-05

**Task:** 3567 (operational follow-up to 3083)
**Git sha at run:** `411f453063306a76466134a32e42221d4a9fd9e0` (a pre-rebase
branch sha — a record of the run environment, not a durability pin, and not
expected to be reachable from `main`; see the note in §"the `--apply` run")
**Swept at:** 2026-08-05T10:27:11Z → 10:27:26Z (15s)
**Collection:** `fused_dark_factory` (Qdrant), exact point count 21079 @ 10:14:27Z → 21087 @ 10:45:17Z
**Exit code:** 0

```
cd fused-memory
uv run python scripts/sweep_toolcall_xml_leak.py --exhaustive \
  > ../docs/toolcall-xml-leak-sweep-2026-08-05/dry-run-report.json
```

Raw evidence lives beside this file and is committed verbatim — `dry-run-report.json`
is byte-for-byte the script's stdout, not pretty-printed or hand-edited.
`dry-run-provenance.json` records the run's own metadata out-of-band. The gated
`--apply` run that followed is captured the same way in `apply-report.json` and
`apply-provenance.json`. `recovery-tracking.json` carries one entry per
unrepaired live-corpus mutation — its owning task and the committed report
holding the recoverable payload — and is machine-checked on every test run (§4,
§6).

> **Notation.** Every leak marker below spells `<` as `\x3c`. Writing one verbatim
> makes the authoring tool call terminate early — reproducing the very bug under
> study. Same convention as `fused-memory/tests/test_sweep_toolcall_xml_leak.py`.

---

## 1. The incidence measurement

This is the headline deliverable: the first real incidence rate anyone has had
for this defect in the Mem0 corpus.

| Quantity | Value |
|---|---|
| Points walked (`scanned`) | **21,080** |
| Records carrying a leak (`len(records)`) | **41** |
| Incidence | **~0.19%** (41 / 21,080) |
| `clean` | 0 |
| `repairable_tail` | 0 |
| `repairable_duplicate` | **1** |
| `manual_review` | **40** |
| `truncated` / `limit` / `aborted` | false / null / absent |

`clean` is 0 by construction, not by luck: the sweep only appends a record when
the detector flags it, so clean points are counted in `scanned` and never
materialise as records.

### How far to trust the denominator

**Not very far, and deliberately so.** 21,080 is what the sweep walked; the
collection-level exact counts bracketing the run were 21,079 (13 min before) and
21,087 (18 min after). Three independent reasons forbid reading any of these as a
clean ratio:

1. **The corpus is live.** It took concurrent writes throughout; +8 points drifted
   across the ~31-minute bracket.
2. **It is not monotonic.** Consolidation actively *deletes* entries — that is this
   task's own premise — so a later count can fall *below* `scanned`.
3. **Different populations.** `scan_payload_text` scopes by `group_id`
   (`project_id=dark_factory`); a collection-level count does not. They are not the
   same set.

So "~0.19%" is a sound statement about *what the sweep walked*, and only a rough
one about the collection. No assertion anywhere in
`fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py` relates `scanned`
to the point counts, for exactly these reasons. What the test *does* pin is that
the run was `--exhaustive`, un-`--limit`ed, un-truncated and non-aborted — so a
partial run can never be committed as an incidence rate.

### Why `--exhaustive` is load-bearing

It skips the server-side `MatchText` prefilter and paginates the whole
collection, so the result depends on nothing but the shared Python detector.
Runbook §7 warns that `MatchText` on the un-indexed `data` field would silently
flip to tokenized matching if a payload index were ever added. Measured at
capture time: the collection's `payload_schema` indexes `agent_id`, `run_id`,
`user_id` and `actor_id` — **not** `data`. The hazard was not live for this run,
and the exhaustive walk makes it moot regardless.

---

## 2. Surviving specimens

41 records still carry a real leak. For scale, the two specimens that motivated
task 3083 — mem0 `c759c53b` (the `repairable_tail` shape) and `9f2d2ae6` (the
`repairable_duplicate` shape), both dated 2026-07-27 — were already **lost to
consolidation** before this sweep ran. Neither appears in these results.

That is the evidence-loss risk this task existed to beat, and it is real: the
two best-documented specimens are gone. The 41 records below are now the
corpus of record, and they are committed verbatim in `dry-run-report.json`.

Oldest surviving specimen is 2026-04-25; newest is 2026-07-30. By month:
April 2, May 16, June 7, July 16.

---

## 3. Adjudication of the 40 `manual_review` records

Every record was adjudicated by re-running the production detector
(`_split_at_leak`, `classify_record`) over its own stored content — no eyeballing.
**Nothing was mutated in this step:** no `delete_memory`, no `add_memory`, no
`update_memory`. The sweep itself refuses `manual_review` by construction.

### The single shared shape

All 40 have the *same* structure, and it is not the shape 3083 anticipated:

```
<intended content>  \x3c/TAG>  \n  \x3c parameter name="ARG">  <ARG's value>
```

- **None** is a leak at offset 0 (`before` is non-empty in all 40), so no record is
  unrepairable-for-lack-of-content.
- **All 40** have a `parameter` continuation tag.
- **Measured:** for all 40, dropping everything from the first marker leaves text
  that `classify_record` judges `clean` *and* non-empty. Zero exceptions.

They are `manual_review` for one narrow reason: `repairable_tail` requires
`not remainder.strip()`, and here the remainder is non-empty — it is **the value of
the sibling argument that early termination swallowed**. The detector cannot prove
that value isn't user content, so it fail-safes. Correctly.

This is manifestation #1 from `fused_memory/utils/toolcall_xml_leak.py`'s root-cause
section — *sibling-argument loss, the silent and dangerous one* — showing up in live
data at scale, where 3083 had only inferred it.

### Class 1 — swallowed short argument value (35 records)

The remainder is a short, structured argument value, not prose:

| swallowed arg | value | n |
|---|---|---|
| `priority` | `low` | 14 |
| `priority` | `medium` | 4 |
| `priority` | `high` | 1 |
| `category` | `procedural_knowledge` | 12 |
| `category` | `preferences_and_norms` | 2 |
| `category` | `observations_and_summaries` | 2 |

**Disposition: recommend repair, but via a detector change — not a hand edit.**
Dropping from the marker is provably content-preserving here (verified above), but
authorising it requires a new classification in the shared detector, which is
production code and out of scope for an operational task. Filed as a follow-up
(ticket `tkt_0RS3HSVVY7CGSQSMNM8P5KH184`).

> **Hypothesis (not verified here).** The `priority` values are the intended
> argument, surviving only as text because the call itself received `priority=None`.
> Per that module's account of `sqlite_task_backend`'s `priority or 'medium'`, those
> calls would have silently stored `medium` — making the 14 `low` and 1 `high`
> records candidate silently-mispriced tasks. Corroborating precedent: `submit_task`'s
> own docstring records that "one reify task was filed priority=high and stored as
> medium". **Not checked against the task DB** — that is the
> `scripts/scan_task_toolcall_leaks.py` / task-2939 surface, out of scope here.

### Class 2 — nested double leak (3 records)

`a34bb57b-8d71-4d9e-811b-34d7e5375ce9`, `e36a95b6-639d-4a64-a829-55d90ba130b9`,
`eac4eff7-1683-4c37-b97b-6f0990dd8826`.

These are **`repairable_duplicate` records the detector cannot see as such.** The
content self-duplicated *and* the duplicate carries its own second leak bearing a
`category` value, so `remainder` is `before` + second-marker + value — which fails
the `remainder == before` verbatim test by exactly that trailing residue.

Measured, per record: `remainder[:len(before)] == before` is **True** (`before` =
1046 / 946 / 1686 chars), and `classify_record(before) == 'clean'`.

**Disposition: recommend repair.** A repair to `before` is provably lossless — the
discarded text is a verified verbatim copy plus leak residue. Same follow-up ticket.

### Class 3 — substantial distinct content (2 records) — **needs a human**

These are the only two where a repair would destroy real text that exists nowhere
else.

| memory id | arg | size | evidence it is distinct |
|---|---|---|---|
| `6fc731fb-525f-48ab-9db3-d71be34ffc66` | `details` | 957 chars | similarity to `before` 0.010, common prefix 1 char — a complete "TEST STRATEGY (TDD — RED before, GREEN after)" section |
| `88582b74-edee-4139-9a4d-15fe516b3fa1` | `metadata` | 115 chars | a JSON blob: `{"topic": "lock-charter-extension-drift-main-red", "kind": "investigation_outcome", "source": "steward-esc-3194-3"}` |

**Disposition: leave as `manual_review`. Do not bulk-repair.** Recommend preserving
the swallowed text elsewhere *before* any repair is contemplated. `6fc731fb` in
particular is ~1KB of genuine authored prose that a naive tail-drop would delete.
These two are the calibration counterexample for any future sibling-argument rule.

### Full per-record table

Class `R` on the last row is the one `repairable_duplicate` record, included for
completeness. `kept` = chars a repair would preserve, `dropped` = chars it would
discard.

| # | class | memory id | created | total | kept | dropped | swallowed arg | value |
|---|-------|-----------|---------|-------|------|---------|---------------|-------|
| 1 | C1 | `02daa988-d83f-45af-ba85-0d8861c8d735` | 2026-04-25 | 2861 | 2817 | 4 | `priority` | `high` |
| 2 | C1 | `fdceb81a-d853-41d5-a21c-4104de586263` | 2026-04-26 | 1249 | 1204 | 3 | `priority` | `low` |
| 3 | C1 | `7c7f98a9-3cb2-48cf-8f92-6e3d7ff9938e` | 2026-05-10 | 1541 | 1493 | 6 | `priority` | `medium` |
| 4 | C1 | `fc0b776b-685a-4c9b-9da7-fdbb80f4c2f4` | 2026-05-13 | 1360 | 1312 | 6 | `priority` | `medium` |
| 5 | C1 | `8b01fcc4-fec3-4498-8f7a-8185b22c80fe` | 2026-05-14 | 1178 | 1133 | 3 | `priority` | `low` |
| 6 | C1 | `307ca0d6-2eae-49f7-9470-bfa61144ed89` | 2026-05-15 | 1612 | 1567 | 3 | `priority` | `low` |
| 7 | C1 | `a7cc3013-13e1-45a2-bd64-48f0dd5bc10d` | 2026-05-15 | 1306 | 1261 | 3 | `priority` | `low` |
| 8 | C1 | `f48c8f4e-4f28-41d4-badf-4fb740cb297b` | 2026-05-15 | 1047 | 1002 | 3 | `priority` | `low` |
| 9 | C1 | `053517c6-0e74-4763-9f9b-422e9028f10a` | 2026-05-16 | 2562 | 2514 | 6 | `priority` | `medium` |
| 10 | C1 | `0b4ee60b-d2c9-47d3-9094-e2cffef6861f` | 2026-05-16 | 1592 | 1547 | 3 | `priority` | `low` |
| 11 | C1 | `631d0ecc-179f-4702-b826-2ed691ecb2c8` | 2026-05-16 | 926 | 881 | 3 | `priority` | `low` |
| 12 | C1 | `cdf03586-c79a-4742-89e7-b15e3c1e7753` | 2026-05-16 | 685 | 640 | 3 | `priority` | `low` |
| 13 | C1 | `d041fd76-7bd3-46e3-af17-2f877823d01f` | 2026-05-16 | 2148 | 2103 | 3 | `priority` | `low` |
| 14 | C1 | `dc7fab6c-725a-4a00-b3db-355b8a468ced` | 2026-05-16 | 843 | 798 | 3 | `priority` | `low` |
| 15 | C1 | `dcdbbf91-771b-4238-aa0c-67fe1037e005` | 2026-05-16 | 1974 | 1926 | 6 | `priority` | `medium` |
| 16 | C1 | `e5216cb0-ae7c-4622-b167-25c7806364f2` | 2026-05-16 | 2315 | 2270 | 3 | `priority` | `low` |
| 17 | C1 | `eb6131ab-e926-4c28-845e-a236c93c3e76` | 2026-05-16 | 1414 | 1369 | 3 | `priority` | `low` |
| 18 | C1 | `f5129da2-3dc1-4ce8-a07b-c4bf93a3d4de` | 2026-05-16 | 1174 | 1129 | 3 | `priority` | `low` |
| 19 | C1 | `33e6e736-3665-4cd9-a494-94ab83cd43d2` | 2026-06-18 | 701 | 643 | 20 | `category` | `procedural_knowledge` |
| 20 | C1 | `db561044-e5ca-4c54-b129-ee90b7b8c54b` | 2026-06-18 | 1969 | 1924 | 3 | `priority` | `low` |
| 21 | C1 | `5b17469e-1574-4e3a-94a0-6ed3046707d1` | 2026-07-04 | 911 | 847 | 26 | `category` | `observations_and_summaries` |
| 22 | C1 | `32c8f8d7-5c40-46f9-ae0b-8ea16eeb5257` | 2026-07-07 | 1081 | 1023 | 20 | `category` | `procedural_knowledge` |
| 23 | C1 | `0c947fa8-bce1-4fb1-a768-033f42dcf894` | 2026-07-10 | 906 | 848 | 20 | `category` | `procedural_knowledge` |
| 24 | C1 | `245c5ae9-a3cb-49a6-8c70-8869eca48061` | 2026-07-10 | 801 | 743 | 20 | `category` | `procedural_knowledge` |
| 25 | C1 | `1fe22a5c-d6ea-4182-b4f9-04cc16c4910a` | 2026-07-12 | 1050 | 992 | 20 | `category` | `procedural_knowledge` |
| 26 | C1 | `1b7b4739-551b-414c-bd9a-eccab487f64a` | 2026-07-18 | 791 | 733 | 20 | `category` | `procedural_knowledge` |
| 27 | C1 | `18386d96-3cd2-479b-b079-a0255069a821` | 2026-07-28 | 701 | 643 | 20 | `category` | `procedural_knowledge` |
| 28 | C1 | `4970a431-ad8d-45f5-acbe-7d8743b1862e` | 2026-07-28 | 894 | 830 | 26 | `category` | `observations_and_summaries` |
| 29 | C1 | `6900120b-fb52-4452-bf3e-6ff0514675f3` | 2026-07-28 | 898 | 840 | 20 | `category` | `procedural_knowledge` |
| 30 | C1 | `b3e109c5-1b0c-41f5-9fd0-2b0fe71e5f94` | 2026-07-28 | 956 | 898 | 20 | `category` | `procedural_knowledge` |
| 31 | C1 | `633040e7-207c-4863-84fd-949e46daa7ad` | 2026-07-29 | 652 | 594 | 20 | `category` | `procedural_knowledge` |
| 32 | C1 | `8e527eb5-0c64-4443-8247-7b271ee02546` | 2026-07-29 | 740 | 682 | 20 | `category` | `procedural_knowledge` |
| 33 | C1 | `c73fe5b3-f556-42ba-bb6f-8faead8ba63f` | 2026-07-30 | 1050 | 992 | 20 | `category` | `procedural_knowledge` |
| 34 | C1 | `ce562b7a-f86f-4366-87e4-7053688bfd5b` | 2026-07-30 | 862 | 803 | 21 | `category` | `preferences_and_norms` |
| 35 | C1 | `d370820e-729b-47dd-bdeb-1fa415d51fc6` | 2026-07-30 | 774 | 715 | 21 | `category` | `preferences_and_norms` |
| 36 | C2 | `a34bb57b-8d71-4d9e-811b-34d7e5375ce9` | 2026-06-23 | 2190 | 1046 | 1107 | `content` | `(1107 chars)` |
| 37 | C2 | `e36a95b6-639d-4a64-a829-55d90ba130b9` | 2026-06-23 | 1993 | 946 | 1010 | `content` | `(1010 chars)` |
| 38 | C2 | `eac4eff7-1683-4c37-b97b-6f0990dd8826` | 2026-06-23 | 3473 | 1686 | 1750 | `content` | `(1750 chars)` |
| 39 | C3 | `6fc731fb-525f-48ab-9db3-d71be34ffc66` | 2026-06-22 | 3555 | 2557 | 957 | `details` | `(957 chars)` |
| 40 | C3 | `88582b74-edee-4139-9a4d-15fe516b3fa1` | 2026-07-29 | 1958 | 1805 | 115 | `metadata` | `(115 chars)` |
| 41 | R | `7d073281-4c5d-4ba3-a01c-3a167f4460f4` | 2026-07-09 | 867 | 415 | 415 | `content` | `(415 chars)` |

---

## 4. What was repaired — **nothing, and one record was lost**

> **Status: OPEN, and TRACKED AS TASK 3686.** Escalated as a blocker
> (`esc-3567-2`); that escalation was subsequently AUTO-DISMISSED on timeout —
> "steward did not resolve within the ESCALATED wait window" — not adjudicated
> by a human. That dismissal is precisely why a task had to be filed: an
> escalation that can time out is not an owner. Recovery of `7d073281` has not
> happened; **task 3686** owns it, along with the sandbox-policy decision it
> depends on. The machine-readable half of that tracker is
> `recovery-tracking.json` in this directory, asserted on every test run by
> `fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py`. Do not re-run
> the sweep to "fix" this.

The gate opened (`repairable_tail 0 + repairable_duplicate 1 = 1 ≥ 1`), so
`--apply` ran, with the dry-run report and its sidecar already committed. That
ordering is the only reason this section can be written at all.

> **Why no commit sha is cited for that payload.** Earlier drafts pinned it to
> `a1b77e3265` / `1b7906526b` / `3e926d2744`. Those shas were real when written
> and are unreachable now: the merge lane rebases this branch, so a sha naming
> a same-branch commit is an orphan by the time the work lands on main, and
> `git show <sha>:<path>` fails outright once `git gc` prunes it — taking the
> documented recovery path with it. A durability pin on unmerged work must
> therefore be a **repo-relative path plus a content hash**, never a
> same-branch commit sha. `recovery-tracking.json` records
> `payload_content_sha256` / `payload_repaired_content_sha256`, which the
> artifact suite re-derives from the committed bytes on every run.

```
cd fused-memory
uv run python scripts/sweep_toolcall_xml_leak.py --apply --exhaustive \
  > ../docs/toolcall-xml-leak-sweep-2026-08-05/apply-report.json
# exit 1 — see below; NOT the benign manual_review-only case
```

**Outcome: 0 repaired, 1 record deleted with no re-add.**

| | |
|---|---|
| Record | `7d073281-4c5d-4ba3-a01c-3a167f4460f4` (the sole `repairable_duplicate`) |
| Flag | `record_error` — *"attempt to write a readonly database"* |
| `repaired` | `false`; no `new_id` recorded |
| Qdrant state | **absent** — measured read-only after the run, 0 points returned for that id |
| Point count | 21,089 → 21,088 (net −1) |
| Surviving copy | **git only** — `docs/toolcall-xml-leak-sweep-2026-08-05/dry-run-report.json`: 867-char original (`sha256 f8cf0112…c8a6bd`) + the intended 415-char `repaired_content` (`sha256 e3214e60…a5691b`) |

### Why exit 1 here is *not* the benign case

`resolve_exit_code` returns 1 for two independent reasons on this run. The
first — 40 `manual_review` records left behind — is the expected disclosure the
plan anticipated. The second is not: `record_error` is one of the four
per-record outcomes the runbook says a human must adjudicate. Reading the exit
code alone would have hidden this. The flag scan is what surfaced it.

### What actually happened

The traceback puts the failure **inside `delete_memory`**
(`memory_service.py:4058` → `_journaled_backend_call:1294` →
`sqlite3.OperationalError`), *not* in the re-add. So the Qdrant point removal
had already succeeded and the exception aborted the repair before a re-add was
ever attempted. This is the `content_lost_in_flight` situation in substance,
arriving under the `record_error` flag.

### Root cause — measured, not guessed

**The agent sandbox denies file *creation* in `~/.mem0`.** A probe from the same
environment shows `open('~/.mem0/.write-probe', 'w')` raises
`PermissionError [Errno 13]`, while the existing `history.db` still accepts
`BEGIN IMMEDIATE`. SQLite cannot create its rollback journal in that directory,
so it reports *"attempt to write a readonly database"*. The db file itself is
mode `0644`, uid 1000, `os.access(W_OK)=True` — this is **not** a filesystem
permissions defect.

### The operational lesson

**`--apply` must not be run from inside a sandboxed agent session.** The Qdrant
mutation is a network call to `localhost:6333` and succeeds; mem0's local SQLite
history write is blocked. The sandbox splits delete-then-re-add exactly down the
middle and produces partial mutations. That is a property of the *environment*,
not of the sweep — which behaved correctly throughout: it recorded the error on
the record, kept going rather than discarding the report, and refused all 40
`manual_review` records.

### Recovery

The content is not lost, because it was committed first. To restore
`7d073281`, re-add the 415-char `repaired_content` from `dry-run-report.json`
(the clean, leak-free text) **from a non-sandboxed session**. Note the memory
id will necessarily differ. Per the runbook, do **not** re-run the sweep to
recover it.

**Not yet done — tracked as task 3686.** It was left for a human deliberately —
re-adding text to the shared corpus mints a new id and is a mutation whose
authorisation is not this task's to assume — and the escalation that asked for
that decision (`esc-3567-2`) was auto-dismissed on timeout rather than answered,
which is why the ask was converted into a task rather than left as prose. Task
3686 carries both open decisions together, because they share a root cause and
whoever performs the re-add must already have settled the second one — a
recovery attempted from a still-sandboxed session fails identically:

1. the recovery above; and
2. whether `--apply`-class operations should be blocked from sandboxed sessions
   outright, or the sandbox permit file creation under `~/.mem0`.

Until one lands, no agent session should run this sweep — or any
`delete_memory` — with `--apply`.

When the re-add happens, flip `recovered` to `true` in `recovery-tracking.json`
and record the new memory id there. The artifact test asserts that pairing
(`recovered: false` forbids a `new_id`; `true` requires one), so the tracker
cannot drift out of step with the store.

---

## 5. Caveats and boundaries

- **The report's own `collection` field is blank.** `new_progress()` seeds it to
  `''` and it landed blank in `dry-run-report.json`. That is the defect **task
  3243** (pending) fixes upstream. Worked around here by recording the collection
  out-of-band in `dry-run-provenance.json`, and asserted non-empty by the artifact
  test — so this capture is trustworthy without pre-empting 3243's work or editing
  the sweep script.
- **Graphiti-side discovery is NOT covered here.** This sweep is Mem0/Qdrant only.
  The Graphiti side is **task 3233** (pending).
- **Residual episode `d12b0eb4-f027-4d0c-a26c-096ccd0e75c2` is deliberately left in
  place**, under 3083's redact-never-cascade-delete policy. Not touched.
- **A payload index on `data` would silently change `MatchText` semantics** for any
  future *non*-exhaustive run (runbook §7). Confirmed absent at capture time. Any
  future incidence measurement must pass `--exhaustive` regardless.
- **The two 3083 specimens are gone**, consolidated away before this run. Any
  future sweep should be treated as measuring a moving population.

## 6. Reproducing / reusing this

`fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py` re-derives every
classification in this report by calling the production `classify_record` on each
record's own stored `content`. It is hermetic — pure file reads plus the pure
detector, no live store — so it stays green whether or not Qdrant is up, and it
runs in the normal test suite.

That makes `dry-run-report.json` a **ready-made regression fixture**: 41 real
specimens with verified classifications, usable to validate a detector change
without touching production memory.

## Related

- `docs/mcp-toolcall-xml-leak.md` — the runbook (root cause, remediation table, exit codes)
- Task 3083 — shipped the sweep tooling and the runbook
- Task 3233 — Graphiti-side discovery (pending)
- Task 3242 — `add_system_record` guard (pending)
- Task 3243 — blank `collection` field in the report (pending)
- Task 2939 — the task-DB-side scan (`scripts/scan_task_toolcall_leaks.py`)
- Escalation `esc-3567-1` — this sweep's findings, filed as disclosure
  (**auto-dismissed on timeout**, so the class-3 dispositions were never ruled on)
- Escalation `esc-3567-2` — the partial-mutation blocker of §4
  (**auto-dismissed on timeout**, which is why the ask below was filed as a task)
- **Task 3686 — the live owner of §4**: re-add `7d073281` from a non-sandboxed
  session, and decide whether mutating operations should be blocked from
  sandboxed sessions or `~/.mem0` writes permitted. Filed high priority
- `recovery-tracking.json` (this directory) — the machine-checked half of that
  tracker: one entry per unrepaired live-corpus mutation, naming its owning task
  and the committed report holding the recoverable payload.
  `test_toolcall_xml_leak_sweep_artifacts.py` asserts every entry has a task
  owner and re-runs the production detector over the payload to prove the
  documented recovery is still executable — so deleting the artifact that holds
  the only surviving copy turns the suite RED instead of silently making the
  loss permanent
- Ticket `tkt_0RS3HSVVY7CGSQSMNM8P5KH184` — follow-up: a third repairable shape
