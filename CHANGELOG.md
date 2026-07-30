# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

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
