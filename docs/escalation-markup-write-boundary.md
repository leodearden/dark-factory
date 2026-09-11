# The escalation write boundary and the esc-3514 markup leak

**The verification answer produced by task 3643.** Written so the question is
not re-litigated. Every claim below is a first-hand measurement on this
worktree, cited to the symbol that produces it.

The question, as posed: did task **3083** cover the envelope-markup leak from
the `claude-task-3514-implementer` session, and is that path guarded now?

**Short answer.** 3083 did not, and structurally could not, cover it — but the
leak did not reach the memory corpus either, because a different mechanism
stopped it. It landed in the escalation queue instead. That boundary **is**
guarded now, for DETECTION. It is **not** recovered, and the reason is
self-referential and generalises. Details follow in four parts.

Specimens: `escalation/tests/fixtures/markup_specimens/` — two corrupted
records and their clean control. Executable form:
`escalation/tests/test_markup_specimen_3514.py`.

**Where the numbers live.** That fixture directory's `README.md` is the
normative home for every measured figure — provenance, digests, the corruption
signature table, the recovered lengths, the control comparison. This page keeps
the narrative answer and links there rather than restating them, per INV-5
(`no-lockstep-duplication`, `docs/legibility/design-invariants.md`): two copies
of a measurement that must agree byte-for-byte is exactly the shape that drifts
when one is re-measured. Where a figure appears below it is because the
sentence is about that figure, not because the table is being mirrored.

---

## 1. The memory-write path was covered, and behaved correctly

The premise that the session "leaked raw markup into stored content" is half
right, and the wrong half is where the finding lives.

All three `add_memory` calls from that session were **rejected at the write
boundary** by task 3141's tripwire. The evidence is the escalation the session
filed about its own rejections, preserved as `esc-3514-1.json`: `error_type=`
`McpEnvelopeMarkupWriteRejected`, `field=content`, and the third call tripped
the storm threshold (`count=3 threshold=3 window_seconds=3600`, yielding
`esc-markup-tripwire-2`). **Nothing entered the memory corpus.** For the
corpus, containment worked exactly as designed.

## 2. The leak landed in the escalation queue, which 3083 could not reach

The write that *succeeded* was `escalate_info`, on the escalation MCP server.
Two records carry the residue, and both show the silent sibling-argument-loss
shape — `suggested_action` and `evidence` dropped from the arguments map and
absorbed into `detail`, where nothing reads them.

`esc-3514-1` is the direct producer filing; `esc-3514-3` is the orphan reaper's
re-filing, which propagated the identical corrupted `detail` verbatim. Nothing
between the two filings noticed the markup — that propagation is part of the
finding, not incidental to it. Per-record figures — detail lengths, pattern
hits, what each stored in `suggested_action` — are in the fixture README's
signature table, alongside the clean control that anchors what those figures
do and do not prove.

Both were reachable by none of 3083's tooling, for a structural reason rather
than an oversight. 3083 shipped `MemoryService.scan_memory_content`,
`fused-memory/scripts/sweep_toolcall_xml_leak.py` and
`MemoryService.redact_episode_content` — all scoped to Mem0/Qdrant and
Graphiti. **Escalation records are plain JSON files under a gitignored `data/`
tree**, in no vector store and no knowledge graph. And 3141's tripwire guarded
exactly four fused-memory tool bodies in `fused_memory/server/tools.py`;
`escalate_info` / `escalate_blocker` were a fifth boundary, on a different
server, unguarded at the time of the incident.

So the answer to the first half of the question is: **3083 does not cover this
path, and no widening of 3083 would have.** It is a different store behind a
different server.

## 3. The boundary IS guarded now — task 3690

`escalation/src/escalation/server.py::create_server` registers
`shared.mcp_markup_middleware.MarkupGuardMiddleware` with
`policy=RepairPolicy.FORWARD_REPAIR` and `exempt_tools=frozenset()`. Exemptions
match bare in-server tool names and the set is empty, so **both** `escalate_info`
and `escalate_blocker` are intercepted. The residue sink is
`escalation/src/escalation/server.py::_file_markup_residue`, wired because the
queue is in-process there and it is the one place a refused call's payload can
actually be preserved.

**Answer to the second half of the question: YES for detection**, and this is
where. `escalation/tests/test_markup_specimen_3514.py` asserts
`shared.toolcall_markup.detect` fires on both real records, so the coverage
claim is pinned against the specimens rather than asserted.

Since then, task **4458** moved fused-memory's own containment to its dispatch
boundary as well (`fused_memory/server/markup_guard.py`, the same shared
middleware wrapping `ToolManager.call_tool`), retiring 3141's four in-line
gates in favour of coverage of every tool. The fifth-boundary gap that produced
these records is closed on both servers.

## 4. THE FINDING — recovery does not occur for this specimen class

Task 3643's own description anticipated that under `FORWARD_REPAIR` "a
corrupted `escalate_info` LANDS with its `suggested_action` recovered instead
of being lost." **Measured against the real records, that is false for this
shape.**

Replayed through `shared.toolcall_markup::repair` with `escalate_info`'s real
parameter set, both records return `None`. `None` routes to
`shared.mcp_markup_middleware::MarkupGuardMiddleware._refuse_unrepairable`,
which raises `ToolError` and writes nothing through the tool — the escalation
**does not file under its own task id at all**. The payload survives only as a
separate record on the synthetic `mcp-markup-residue` anchor, filed at
`severity='critical'`, `level=2`, owned by `l2-escalation-watcher`. Preserved,
but not where anyone looking for task 3514's escalations would find it.

### Why: the controlled experiment

`repair()` rejects any candidate whose parsed tail contains a second mis-close.
Inside the swallowed `evidence` value, the report **quotes the very
`matched_pattern` the memory tripwire handed back to it** — a `content` closing
tag, reproduced faithfully in prose because the escalation was *reporting a
markup leak*. That quotation is the second mis-close.

Replacing only that one quoted literal with an inert placeholder — every other
byte of `detail` untouched, verified by round-trip — flips the outcome:
`repair()` returns a `Repair` recovering **both** swallowed siblings, the real
`suggested_action` text and the three `{observation, measured_at, ref}`
evidence entries, with a `clean_value` the detector no longer fires on. The
recovered lengths and the exact strings are tabulated in the fixture README;
`test_the_quoted_pattern_is_the_sole_blocker` asserts them against the bytes.

**The quote is the sole blocker.** Generalised, and this is the part worth
carrying forward:

> An escalation that REPORTS a markup leak is the one payload class the
> repairer structurally cannot recover, because a faithful report quotes the
> pattern, and the quote defeats the tail parser's no-second-mis-close
> condition.

This independently confirms, on a second and unrelated pair of records, the
"doubly corrupted" PRD boundary row B5 shape that
`escalation/tests/test_markup_middleware_registration.py` describes for
`esc-3184-2` and says could never demonstrate a successful recovery. It is a
recurring shape, not a one-off property of one record.

### A measurement correction

Earlier write-ups of these records said "3 markup hits", conflating the pattern
count with the hit count. The corrected figures, and the correction itself, are
recorded in the fixture README's signature table — read them there, not from
memory or from any older write-up.

### And a caveat for any future sweep

Sibling record `esc-3514-2` (same task, `agent_role=orchestrator`) is **clean**
— the detector does not fire on it and its `suggested_action` is intact — and
nevertheless stores `evidence == []`. An empty `evidence` list is therefore
**not** a corruption signal on its own; most escalations simply never pass
evidence, so a sweep keyed on it would flag clean records.

That is no longer a warning a sweep author has to read and believe. The control
record is preserved beside the specimens as `esc-3514-2.json`, and
`test_the_control_record_is_clean_yet_shares_the_empty_evidence_list` asserts
both halves: the naive empty-evidence predicate fires on all three records, the
discriminating pair on the two corrupted ones only. The predicate's exact legs
— including why it must accept a plausible-looking default such as
`manual_intervention` and not only an empty string — are stated once, in the
fixture README, and implemented once, in that test.

---

## 5. Scope and residuals

Stated explicitly so nothing here is quietly dropped.

- **No production code was changed by task 3643.** It preserved two corrupted
  specimens and their clean control, made them load-bearing with a regression
  test, and wrote this page.
- **The fix does not belong here.** PRD `plans/toolcall-markup-containment-prd.md`
  D7 puts the originating defect model-side (wrong closing-tag dialect) and the
  amplification harness-side (over-consuming instead of raising a parse error),
  neither in this repo. `shared/src/shared/toolcall_markup.py`,
  `shared/src/shared/mcp_markup_middleware.py` and
  `escalation/src/escalation/server.py` are owned by tasks 3688 / 3689 / 3690;
  a rival guard built here would be exactly the duplicate implementation INV-5
  forbids. Task 3643 reports rather than builds, by its own item 4.
- **The two records' lost values are preserved but NOT restored.** The real
  `suggested_action` and the three evidence entries live in the committed
  fixtures only. Nothing wrote them back to the live queue, and both records
  are dismissed.
- **There is still no discovery sweep for this class in `data/escalations/`.**
  Task 3691 (the escalation-corpus sweep) is the intended consumer of these
  fixtures; until it lands, the size of this population is unmeasured. Two
  records is what was found by hand, not a count. The predicate that sweep
  should use is committed and tested against a clean control (see the caveat
  above); what is missing is the sweep that runs it.
- **The repair gap is filed as a follow-up**, to be coordinated with the
  3688 / 3689 / 3690 owners rather than patched independently.

## Division of labour

| Task | What it owns |
|---|---|
| 3069 / 3083 | Root cause and the retroactive **memory-corpus** sweep: `scan_memory_content`, `sweep_toolcall_xml_leak.py`, `redact_episode_content`. Mem0/Qdrant + Graphiti only. |
| 3141 | The original write-time tripwire at four fused-memory tool bodies. Retired by 4458. |
| 3567 | The task-text vector (the description parser mis-parsing a leaked fragment silently). |
| 3688 | `shared/toolcall_markup.py` — the literal enumeration, `detect`, `repair`, the override lifecycle. |
| 3689 | `shared/mcp_markup_middleware.py` — the reusable dispatch-boundary guard. |
| 3690 | Registering that guard on the **escalation** server, with the residue sink. |
| 4458 | Registering it on the **fused-memory** server, retiring 3141's four gates. |
| 3691 | The escalation-corpus sweep — the intended consumer of these fixtures. |
| **3643** | **This page, the two preserved specimens plus their clean control, and the regression test that pins the verdict.** |

## If the unrepairable pin ever fails

`test_the_specimen_is_unrepairable_as_stored` asserts `repair()` returns
`None`. A failure there is **the intended signal**, not a broken test: a
repairer has improved past this shape. Revisit and update part 4 of this page
rather than deleting the assertion.
