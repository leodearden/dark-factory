# Capability manifest — dashboard-availability-prd

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) for
`plans/dashboard-availability-prd.md`. Every binding below was checked
against main on **2026-07-30**; no binding resolved to a FAIL value, so the
batch cleared the gate without re-scoping.

Machine-readable twin: `plans/dashboard-availability-prd.capability-manifest.yaml`.

Evidence conventions: `grep:<file>:<line>` means the symbol is **wired into
the consuming entry path** on main (not merely declared, not test-only).
`measured:` means a figure taken on this host on 2026-07-30 and reproducible
by the command shown.

---

## α — Index `write_ops(created_at)`

| Capability | Binding | Verdict |
|---|---|---|
| `schema-sql-runs-every-start` | `grep:fused-memory/src/fused_memory/services/write_journal.py:104` — `initialize()` calls `executescript(SCHEMA_SQL)` unconditionally on every start, so an `IF NOT EXISTS` index added to `SCHEMA_SQL` applies to the **existing** 6.5 GB DB on next restart | PASS |
| `create-index-if-not-exists-precedent` | `grep:fused-memory/src/fused_memory/services/write_journal.py:34-36,50-52` — the same DDL block already creates 5 indexes idempotently, incl. `idx_bo_created ON backend_ops(created_at)`, the exact sibling shape being added here | PASS |
| `dashboard-queries-filter-created-at` | `grep:dashboard/src/dashboard/data/write_journal.py:56,90,118` — all three consuming queries filter `WHERE created_at >= ?`; 3 occurrences confirmed | PASS |
| `numeric-floor: memory-graphs < 2s` | `measured:` seekable range-scan of the same 24h window = **0.00s** (`EXPLAIN QUERY PLAN` → `SEARCH … (kind=? AND created_at>?)`) vs current `SCAN` = **20.94s**; endpoint baseline **108.14s**. Bound 2s sits ~3 orders of magnitude above the 0.00s floor | PASS — floor stated, bound > floor |

No FAIL. DAG-direction: α has no upstream producer — it is the producer.

## β — Frontend poll flow control

| Capability | Binding | Verdict |
|---|---|---|
| `single-poll-loop-site` | `grep:dashboard/src/dashboard/static/redux/data.js:165` — one `setInterval(... , 3000)` drives every endpoint; the guard has exactly one site to land in | PASS |
| `fanout-is-promise-all` | `grep:dashboard/src/dashboard/static/redux/data.js:157` — `Promise.all(Object.entries(endpointsFor(...)).map(...))`, 13 endpoints, no in-flight flag | PASS |
| `dashboard-tests-harness-exists` | `dashboard/tests/` present with existing frontend-contract tests (task 2662 precedent) | PASS |
| `stacking-is-real-not-theoretical` | `measured:` `memory-graphs` = 108.14s against a 3s interval ⇒ ~36 concurrent in-flight by construction | PASS |

Guard **semantics** are not reliably greppable — bound `manual` in the sidecar.

## γ — Hysteresis watchdog with storm escape

| Capability | Binding | Verdict |
|---|---|---|
| `shallow-liveness-endpoint` | `grep:dashboard/src/dashboard/app.py:386-388` — `@app.get('/api/health')` returning a bare `{'status':'ok'}`, no DB access, wired on the live app object | PASS |
| `script-callable-born-at-L2-writer` | `grep:escalation/pyproject.toml:21` — console script `escalation = "escalation.submit:main"`; module docstring names "a detached systemd OnFailure unit" as the intended caller and states it works **without the MCP server** | PASS |
| `born-at-L2-severity-vocabulary` | `grep:escalation/src/escalation/models.py:65` — `BORN_AT_L2_SEVERITIES = frozenset({'critical','urgent'})`, enforced at the argparse boundary | PASS |
| `watchdog-script-precedent` | `scripts/orchestrator-watchdog.py` — existing stdlib-only oneshot watchdog with per-unit revive; the state-persistence and restart idioms exist to copy | PASS |
| B1–B6 behavioural invariants | `manual` — boundary-test table in the PRD is the signal; not expressible as a grep | PASS (recorded, excluded from dispatch gate) |

## δ — Bounded shutdown drain

| Capability | Binding | Verdict |
|---|---|---|
| `uvicorn-graceful-shutdown-flag` | `measured:` `python -m uvicorn --help` on uvicorn **0.44.0** lists `--timeout-graceful-shutdown INTEGER` and `--timeout-keep-alive INTEGER` | PASS |
| `sigkill-is-the-observed-failure` | `measured:` journal shows `State 'stop-sigterm' timed out. Killing.` → `Killing process … with signal SIGKILL` → `Failed with result 'timeout'` on every restart cycle; `TimeoutStopSec=15` | PASS |

## ε — `/healthz` deliverable deadline

| Capability | Binding | Verdict |
|---|---|---|
| `healthz-handler-wired` | `grep:dashboard/src/dashboard/app.py:395-435` — route registered on the live app | PASS |
| `probe-timeout-constant` | `grep:dashboard/src/dashboard/app.py:392` — `_DB_PROBE_TIMEOUT = 5.0` | PASS |
| `uncovered-execute-is-real` | `grep:dashboard/src/dashboard/app.py:421-422` — `asyncio.wait_for` wraps only `cursor.fetchone()`; the `conn.execute(...)` awaited by `async with` on line 421 carries no deadline | PASS |
| `numeric-floor: 503 within <5s` | `measured:` today the same probe returned **503 at 50.6s** — 10× its nominal budget — because 3 DBs × 5.0s exceeds any 5s caller deadline. A whole-handler bound below 5s is achievable purely by construction (bounding the sum), not by making any query faster | PASS — floor stated, bound achievable |

## ζ — write_journal growth alarm

| Capability | Binding | Verdict |
|---|---|---|
| `journal-path-resolvable` | `grep:dashboard/src/dashboard/config.py:191-192` — `write_journal_db` property resolves the path from `RECONCILIATION_DATA_DIR` | PASS |
| `numeric-basis: thresholds` | `measured:` 6.5 GB / **16,311,786** rows spanning 2026-04-07 → 2026-07-30 (~114 days) ⇒ ~**57 MB/day**. Thresholds anchored on measurement, not guessed | PASS — basis cited |
| `alarm-not-prune` | Scope decision: ζ adds a loud counter only. The prune precedent (`prune_idempotent_ops`, `prune_mem0_intents`, same class) is deliberately **not** invoked — irreversible deletion waits for the alarm | PASS |

## η — Dashboard unit-file parity

| Capability | Binding | Verdict |
|---|---|---|
| `unit-parity-precedent` | `scripts/check_fused_memory_unit_parity.py` exists — the directive-allowlist parity idiom to copy | PASS |
| `repo-unit-files-exist` | `dashboard/dark-factory-dashboard.service`, `-watchdog.service`, `-watchdog.timer` all present in-repo | PASS |
| `drift-is-real-not-hypothetical` | `measured:` installed `~/.config/systemd/user/dark-factory-dashboard.service` carries `DASHBOARD_KNOWN_PROJECT_ROOTS` with **9** roots; the in-repo copy carries **1** | PASS |

## θ — Empty-state legibility

| Capability | Binding | Verdict |
|---|---|---|
| `garbled-site-located` | `grep:dashboard/src/dashboard/static/redux/tabs.jsx:278` — `No {filter === 'all' ? '' : filter + ' '}tasks` | PASS |
| `filter-is-an-object` | `grep:dashboard/src/dashboard/static/redux/tabs.jsx:190,248-250` — `DEFAULT_FILTER = {active,pending,complete}`, read as `filter.active` / `.pending` / `.complete`; line 193 even carries the `// back-compat: ignore old string values` migration comment that line 278 was missed by | PASS |
| `rejection-check: no [object Object]` | Expressible mechanically as an `expect: absent` grep for the stringified-object expression — bound as a `delivered_check` in the sidecar | PASS |

---

## Gate summary

- Capability→producer (anti-orphan): all PASS — every consumed symbol is
  wired on main, none `declared-only` or `test-only`.
- DAG-direction (anti-inversion): no capability is owned by a task that
  *depends on* its consumer. The only intra-batch producer edges are
  α→ζ, δ→γ, {γ,δ}→η, all upstream-correct.
- Numeric floors: three bounds asserted (α < 2s, ε < 5s, ζ thresholds); all
  three carry a measured basis and sit above their floor. No `bound≤floor`.
- Rejection-mechanism: θ's is bound and mechanically checkable; γ's
  behavioural invariants are `manual` by nature and excluded from the
  dispatch gate.
