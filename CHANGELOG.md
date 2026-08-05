# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed

#### Leaked tool-call XML: root cause, and the tooling to sweep the corpus (task 3083)

**This task ships no write-time guard.** Live rejection at the MCP write
boundary is delivered by `fused_memory/server/markup_tripwire.py` (task 3141 —
see "MCP writes carrying raw envelope markup are now REJECTED" below), which is
deliberately the only place in the package that enumerates the markup literals.
An earlier revision of this task added a second, parallel guard; it was retired
before merge as a duplicate enumeration of that invariant. If you need to write
content that legitimately quotes a fragment, the override is 3141's
`metadata={'allow_mcp_markup': True}` and the rejection is
`error_type = 'McpEnvelopeMarkupWriteRejected'`.

**What this task establishes: the fragments do not originate in this repo.**
There is no XML parser anywhere in `fused-memory/src/`. They are evidence that
the *harness's* tool-call parser terminated a string argument early at a literal
closing tag inside that argument's value, which also **silently swallows the
sibling arguments that followed**. A leaked fragment in `description` means
`priority` may never have reached the MCP boundary at all, in which case
`sqlite_task_backend.py`'s `priority or 'medium'` substituted a plausible wrong
value with no log. That sibling-argument loss is the part of the failure that is
otherwise invisible — a corrupted record looks merely untidy, not wrong.

**Added:** `fused_memory/utils/toolcall_xml_leak.py`, the shared detector for
*already-stored* content, now the single implementation behind the corpus sweep,
`scan_memory_content`, the redaction path, and `scripts/scan_task_toolcall_leaks.py`
(which previously carried its own inline copy of the pattern).

It is calibrated in the **opposite** direction from 3141's tripwire, on purpose:
the tripwire over-reports to maximise recall at write time, where a false
positive costs one retry; this detector under-reports to stay precise, because
it runs over already-stored content where a false positive would silently
rewrite something a user wrote. That asymmetry is why both exist, and why this
one must not adopt the tripwire's pattern list.

**Also added:** `scan_memory_content` (read-only literal substring scan over
Qdrant payload text — semantic `search` provably cannot find these),
`redact_episode_content` (neutralise a leaked Graphiti episode without
`delete_episode(cascade=True)` destroying its valid extracted edges), and
`fused-memory/scripts/sweep_toolcall_xml_leak.py` (the corpus sweep; dry-run by
default).

**What the sweep's repair preserves.** The memory **text** in full, plus the
**payload metadata** that is the record's only metadata-scoped retrieval axis —
carry-over rule `payload keys - _MEM0_OWNED_KEYS`, with the scope identities
threaded back as `agent_id=` / `session_id=` arguments and anything carried
nowhere *named* per-record in the report's `metadata_dropped`. Without that,
a repaired record would silently vanish from `get_memories_by_metadata` /
`count_memories_by_metadata`, which match payload keys by equality.

**The sweep verifies the re-add persisted** rather than trusting a non-raising
`add_memory`: the service swallows a Mem0 write failure into
`AddMemoryResponse.message` as `[mem0_error: ...]` and returns normally, so a
returned response is not evidence of a write. Three per-record outcomes exit
non-zero and need a human — `content_lost_in_flight` (the delete landed, the
re-add did not persist; the original text now exists only in the printed report,
restore it from there before re-running), `skipped_not_mem0_routed` (a
repairable record whose category does not route to mem0, left entirely untouched
because neither a plain re-add nor `dual_write=True` is safe), and
`record_error` (that record's repair aborted on an unexpected error, so whether
its delete landed is unknown).

**The report always survives an abort**, since for a `content_lost_in_flight`
record it is the only remaining copy of the original text. Each record enters
the report *before* any store mutation is attempted and its repair runs under
its own `try`, so one record's transport failure is recorded as `record_error`
on that record and the sweep continues instead of unwinding and discarding every
earlier entry. If anything escapes anyway, the CLI prints the **partial** report
— same shape, plus `"aborted": true` — before exiting `2`.

**Full root cause, evidence, and operator runbook:**
[`docs/mcp-toolcall-xml-leak.md`](docs/mcp-toolcall-xml-leak.md). Run the sweep
**before** any further large consolidation pass — consolidation deletes
corrupted entries as a merge side effect and destroys the specimens.

#### First live-corpus sweep: ~0.19% incidence, and `--apply` must not run sandboxed (task 3567)

Ran 3083's sweep against the live Mem0/Qdrant corpus. This ships no runtime
behaviour; the deliverable is committed evidence in
[`docs/toolcall-xml-leak-sweep-2026-08-05/`](docs/toolcall-xml-leak-sweep-2026-08-05/)
— verbatim report, provenance sidecar, and `investigation.md`.

**The incidence measurement**, the first anyone has had: an exhaustive walk of
**21,080 points** found **41 leak-carrying records (~0.19%)** — 1
`repairable_duplicate`, 40 `manual_review`, 0 `repairable_tail`. Read it as a
statement about what the sweep walked, not a clean ratio over the collection:
the corpus takes concurrent writes, consolidation *deletes* entries so the count
is not monotonic, and `scan_payload_text` scopes by `group_id` while a
collection count does not. Both 2026-07-27 specimens were already consolidated
away before the run — the evidence-loss risk that motivated running this now was
real, and it had already fired.

**All 40 `manual_review` records were adjudicated by hand** (nothing mutated),
each verdict re-derived by running the production detector over that record's own
content. They share one shape, and it is not the shape 3083 anticipated: a stray
closing tag followed by a `parameter` continuation naming a **sibling argument**
whose value is the swallowed remainder — manifestation #1 (sibling-argument loss,
the silent one) in live data at scale. 35 swallowed a short `priority`/`category`
value; 3 are nested double leaks that are functionally `repairable_duplicate` but
invisible to the detector; **2 need a human**, holding ~1KB of real prose and a
JSON blob that a tail-drop would destroy. A third repairable shape would make 38
of the 40 auto-repairable — filed as follow-up, not done here.

**`--apply` must not be run from a sandboxed agent session.** The gated apply run
repaired 0 records and **deleted one record from Qdrant with no re-add**
(`7d073281`, flagged `record_error`). The sandbox denies file *creation* under
`~/.mem0`, so SQLite cannot create its rollback journal and mem0's history write
fails with *"attempt to write a readonly database"* — while the Qdrant delete,
being a network call, succeeds. That splits delete-then-re-add down the middle.
The record's text survives **only because the dry-run report was committed
first**, which is exactly the ordering constraint the runbook prescribes; that
safety net fired on its first real use. Two lessons now in the runbook: read the
per-record flags rather than the exit code (exit 1 was overdetermined here, and
the exit code alone would have hidden the mutation), and run `--apply` from an
ordinary interactive shell. Recovery of `7d073281` from the committed report is
**still outstanding** — the blocking escalation was auto-dismissed on timeout,
not adjudicated.

`fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py` guards any
committed sweep artifact against fabrication: it re-runs the production
`classify_record` over each record's own stored content and requires a provenance
sidecar naming the collection, sha, argv and bracketing point counts. Hermetic —
no live store — so it stays green whether or not Qdrant is up.

### Changed (BREAKING)

#### MCP writes carrying raw envelope markup are now REJECTED (task 3141)

Four fused-memory MCP write tools — `add_memory`, `add_episode`, `submit_task`,
`update_task` — now **hard-reject** a write whose payload contains a raw MCP
tool-call envelope fragment.  Writes that previously succeeded (and silently
persisted the fragment) now fail.

**Rejected fields:** `content` for the memory tools; `title`, `description`,
`details` and `prompt` for the task tools — the same four-field set
`premise_lint_guard` already lints, since all four reach the same description
parser.

**Response shape** (the write does NOT reach the store or the interceptor):

```
error       : mcp_markup_write_blocked
error_type  : McpEnvelopeMarkupWriteRejected
field       : which field tripped
matched_pattern : the exact literal that matched
content_excerpt : first 200 chars of the offending text
hint        : remediation + the override key + DF 3083
storm       : present ONLY on a rejection burst (count/threshold/window_seconds
              /hint, plus escalation_id when one was filed)
```

**Why:** a harness serialization bug leaks envelope fragments into write
payloads.  Two observed vectors: memory `content` arriving with a closing-tag
tail (permanent specimens now sitting in the mem0 and Graphiti corpora), and
task text arriving with a `<parameter name=`-shaped fragment that the
interceptor's description parser then mis-parsed **silently** — one reify task
was filed `priority=high` and stored as `medium`.  Loud rejection at the
boundary is strictly better than either outcome.  Rejecting is deliberately
scoped to CONTAINMENT: DF task 3083 owns the root cause, the Qdrant payload
text-match read tool and the retroactive corpus sweep.

**Migration note:** if you write text that quotes envelope markup on purpose —
documenting this very leak, for instance — pass
`metadata={'allow_mcp_markup': True}`.  Only a literal boolean `True` counts,
and the flag is write-time-only: it is stripped before persistence, so it never
enters stored memory metadata or the task metadata vocabulary.  An accidental
serialization leak never sets an explicit flag; an author can.  The matcher is
deliberately a bare case-sensitive substring scan and therefore over-reports
relative to the retrospective `scripts/scan_task_toolcall_leaks.py`; the
authoritative pattern list and the reasoning behind the differing calibration
live in `fused-memory/src/fused_memory/server/markup_tripwire.py`.

**Also new:** a per-server rolling-window storm counter.  Three rejections
within an hour emit one greppable `markup_tripwire_storm` ERROR log line and
file a best-effort `mcp_markup_write_storm` escalation (level 1, deduped against
an open one), because a burst means the upstream leak is *active* rather than
that the tripwire is misfiring.  Escalation is purely additive — a queue
failure never changes a rejection's outcome.

#### `reservation_installed` (reason=reserve_now) → `reserve_now_consumed` (task 1230)

The reserve-now short-circuit path in the scheduler **no longer emits**
`reservation_installed` with `data.reason == 'reserve_now'`.  It now emits the
dedicated `reserve_now_consumed` event instead.

**Old behaviour (pre-task-1230):**

```
event_type : reservation_installed
data       : {modules, priority, reason='reserve_now'}
```

**New behaviour:**

```
event_type : reserve_now_consumed
data       : {modules, priority}
```

**Commits:** `4d45eecd9b` (add `reserve_now_armed`/`reserve_now_consumed`
`EventType` members) · `deb8f426ab` (replace `reservation_installed` with
`reserve_now_consumed` at the scheduler short-circuit emit site, steps 5-6).

**Migration note:**  Any downstream consumer (dashboard query, log filter,
reconciliation tooling, external subscriber) that was filtering on
`event_type = 'reservation_installed' AND data->>'reason' = 'reserve_now'`
must migrate to `event_type = 'reserve_now_consumed'`.

**Threshold-based reservation path is UNCHANGED** — the scheduler still emits
`reservation_installed` when the skip-count threshold is exceeded
(`scheduler.py:1593`); that event's data payload has never contained a
`reason` key (`data = {modules, skip_count, priority}`).  The two events are
now discriminated by **event name**, not by a `reason` field:

| Event | Path | Data keys |
|---|---|---|
| `reservation_installed` | threshold (skip_count ≥ threshold) | `modules`, `skip_count`, `priority` |
| `reserve_now_consumed` | reserve-now short-circuit | `modules`, `priority` |

**Dual-emit rejected:** Option (b) — emitting both `reservation_installed` and
`reserve_now_consumed` during a deprecation window — was evaluated and
rejected.  It would protect zero in-repo consumers (see audit below) and would
require reverting the deliberately-added, merged locked-in regression test
`TestReserveNowConsumedShortCircuit` (part b, `test_scheduler_state.py:191-197`)
from task 1230, contradicting a merged decision for no benefit.

**Audit result (task 1333) — in-repo blast radius: ZERO:**

An exhaustive search across all `*.py`, `*.md`, `*.sql`, `*.js`, `*.ts`,
`*.html`, `*.yaml`, `*.yml`, `*.json` files (excluding `.venv/`, `.git/`,
`uv.lock`) found the following `reservation_installed` sites:

- `orchestrator/src/orchestrator/event_store.py:75` — enum definition only
- `orchestrator/src/orchestrator/scheduler.py:1593` — **threshold-path emit**
  (unrelated to reserve_now; no `reason` key; UNCHANGED)
- `orchestrator/src/orchestrator/scheduler.py:2030` — code comment only
- `orchestrator/tests/test_scheduler_state.py:155,191-197` — task-1230
  locked-in regression test (asserts short-circuit emits `reserve_now_consumed`
  and no legacy `reservation_installed` with `reason=='reserve_now'`)
- `orchestrator/tests/test_scheduler.py:2019,2206,2230-2234,3899,3952-3955,4022-4025`
  — tests of the threshold `reservation_installed` path (unaffected)

Dashboard (`dashboard/`): all `reserve_now` references concern the override
*flag* (UI badges, clear-fields allow-list, POST body, scheduler-snapshot
reads) — the dashboard does **not** consume `reservation_installed` events from
the event_store.  No `scripts/`, SQL, JS/TS, or documentation file consumes or
filters the event.

No emit site in the current (post-task-1230) tree sets `data.reason` on
`reservation_installed`.  (Before task 1230 the reserve-now short-circuit did
emit `reservation_installed` with `reason='reserve_now'`; removed in
`deb8f426ab` — historical event-store rows predating task 1230 may still
contain it.)  The renamed event's pre-1230 form (`reservation_installed` +
`data.reason='reserve_now'`) had zero in-repo consumers.
