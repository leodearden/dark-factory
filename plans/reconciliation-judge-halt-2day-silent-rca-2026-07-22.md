# RCA — reconciliation backlog grew for 2 days because a judge halt was never surfaced or cleared

**Date:** 2026-07-22
**Symptom queue:** `data/escalations` (escalation MCP port 8103) — `ReconciliationBacklogExceeded` for `dark_factory`
**Affected category:** `reconciliation` (judge halt state machine + backlog escalation routing + operator probe)
**Task:** 2920

---

## ⚠️ PREMISE CORRECTION — the original framing was an artifact of the wrong probe

An earlier read of this incident framed it as **"9 self-clearing spike-then-drain
cycles → the backlog alert is crying wolf"** and proposed a *sustained-crossing
gate* that would have **suppressed** the alert. That framing was WRONG, and acting
on it would have made the incident strictly worse (even longer silence).

The "self-clearing spikes" were an illusion produced by **probing the wrong
metric**: the auto-watcher and the L2 responder kept calling `get_queue_stats`,
which reported the **durable-write queue** (a different subsystem that idles at
~0), saw ~0, and concluded the backlog had drained. It had not.

**Ground truth.** The judge **halted** `dark_factory` at **2026-07-20T08:34:36Z**
(reason: *"Serious verdict in run e87d8e4a…"*, `cooldown_until` = 2026-07-20T09:04:36Z,
`unhalted_at` = `None`) and **nothing ever cleared the halt**. With reconciliation
halted, the real backlog — `current_backlog` = buffered events + event-queue depth
+ retries — grew **monotonically from ~501 to ~1548 over two days**. There was no
spike-then-drain; there was one halt and a straight-line climb.

Every conclusion downstream of the "crying wolf" framing is void. This RCA
documents the real mechanism and the three fixes shipped under task 2920.

---

## TL;DR

Four independent defects compounded into a two-day silent outage:

1. **A halt has no auto-clear.** `Judge._apply_halt` latches the project into
   `_halted_projects` and records a `cooldown_until`, but `cooldown_until` **only
   gates the trend detector from re-firing** (`_check_error_trends`) — it is *not*
   a mechanism that clears the halt. The halt persists until `unhalt()` is called
   explicitly, and `initialize()` **rehydrates the halt from the journal on every
   restart**, so restarts re-latch it. The cooldown expired 30 minutes after the
   halt and then nothing on the system ever acted on that expiry for two days.
2. **The halt escalation was absorbed into backlog noise.** `_maybe_write_escalation`
   filed *every* reconciliation escalation under the id prefix
   `esc-reconciliation-backlog-` regardless of kind, AND rate-limited all kinds
   through a **single shared** `last_escalation_ts` — so a routine backlog
   escalation within the 900 s window **silently suppressed** the judge-halt
   escalation. The one signal that would have named the halt was mis-labeled and
   then eaten.
3. **The operator probe read the wrong subsystem.** `get_queue_stats` returned the
   durable-write-queue counts (≈0). The reconciliation backlog (`current_backlog`)
   was never exposed by any tool, so responders had no correct number to look at
   and mis-triaged for two days.
4. **No human acted.** Because (1)–(3) hid the halt, the halt sat until an operator
   manually ran `unhalt_reconciliation` + `trigger_reconciliation` at ~13:22Z.

---

## Timeline

| time (UTC, 2026-07-20 unless noted) | event |
|---|---|
| 08:32:55 | `error_during_execution` churn from the 07-20 cap-storm / claude-CLI-failure window feeds the judge a serious verdict for run `e87d8e4a…`. |
| 08:34:36 | **Judge halts `dark_factory`** (`_apply_halt`, reason "Serious verdict in run e87d8e4a…"). `cooldown_until` set to 09:04:36Z. `journal.set_halt` persists it. |
| 08:34:36 | `on_judge_halt` fires — but its escalation is filed as `esc-reconciliation-backlog-…` and (being inside the shared 900 s rate-limit window of a preceding backlog escalation) is **suppressed**. No distinct halt signal reaches the queue. |
| 09:04:36 | `cooldown_until` **expires**. Nothing acts on the expiry — cooldown only ever gated the trend detector, never the halt itself. Reconciliation stays halted. |
| 08:34 → 07-22 | Each cycle: `harness.run_loop` sees `is_halted` → logs "Skipping cycle for halted project", `_notify_judge_halt` (deduped per-process), returns. Backlog `current_backlog` climbs **~501 → ~1548**. |
| 07-20 → 07-22 | Auto-watcher / L2 probe `get_queue_stats`, see durable-write-queue ≈0, conclude "drained." Chronic mis-triage. |
| 07-22 ~13:22 | **Operator** runs `unhalt_reconciliation` + `trigger_reconciliation`. `unhalt()` clears the halt, seeds `halt_grace_cycles` (3) post-unhalt grace, drops `cooldown_until`. Backlog drains. |

---

## File-and-line walkthrough

### 1. `cooldown_until` gates the trend detector, not the halt

`fused-memory/src/fused_memory/reconciliation/judge.py`

- `_apply_halt` (`judge.py:580-598`) latches the halt and records the cooldown:
  ```python
  self._halted_projects.add(project_id)
  self._halt_cooldown_until[project_id] = cooldown_until   # now + halt_cooldown_seconds
  ...
  await self.journal.set_halt(project_id, halted_at=now, cooldown_until=cooldown_until, reason=reason)
  ```
- `cooldown_until` is consumed in **exactly one place** — `_check_error_trends`
  (`judge.py:541-548`) — where it merely suppresses the trend detector from
  *re-firing* while a cooldown is active:
  ```python
  cooldown_until = self._halt_cooldown_until.get(project_id)
  if cooldown_until and cooldown_until > now:
      logger.debug(...); return
  ```
  There is **no** code path anywhere that says "cooldown expired → clear the halt."
- `is_halted` (`judge.py:600-601`) returns `project_id in self._halted_projects`,
  which is only ever removed by `unhalt` (`judge.py:655-665`). `initialize`
  (`judge.py:82-100`) rehydrates `_halted_projects` / `_halt_cooldown_until` from
  `journal.halt_state` on startup, so a restart **re-latches** the halt rather
  than clearing it.

**→ This is the "no-auto-unhalt gap": a halt is permanent until an explicit
`unhalt()`, and the one clock value that expires (`cooldown_until`) was never
wired to trigger that unhalt.**

### 2. The harness silently skips every halted cycle

`fused-memory/src/fused_memory/reconciliation/harness.py:2196-2244` (`run_loop`):
for a halted project the loop logged `"Skipping cycle for halted project"`, called
`_notify_judge_halt` (`harness.py:566`, deduped **per-process** via `_halt_escalated`,
so a restart re-notifies once and then goes quiet again), replayed deferred writes,
marked the run complete, and `return`ed. Correct as a skip; fatal as the *only*
behaviour, because there was no escape hatch when the cooldown had long expired.

### 3. The halt escalation was mis-filed and rate-limited into oblivion

`fused-memory/src/fused_memory/reconciliation/backlog_policy.py`

Pre-fix, `_maybe_write_escalation`:
- hardcoded the id prefix `esc-reconciliation-backlog-` for **every** escalation,
  so a judge halt was literally filed as "backlog"; and
- rate-limited via a **single scalar** `_PolicyState.last_escalation_ts` shared
  across backlog + halt + wedge, so a backlog escalation within the 900 s window
  **suppressed** the judge-halt escalation (`on_judge_halt`, `backlog_policy.py:214-233`).

Two independent failures — *mis-identification* (wrong id) and *loss of signal*
(shared rate-limit bucket). Renaming the id alone would not have fixed the
suppression.

### 4. The operator probe read the wrong subsystem

`fused-memory/src/fused_memory/server/tools.py:2628-2684` (`get_queue_stats`)
returned `durable_queue.get_stats(...)` — the durable **write** queue, a distinct
subsystem that idles at ~0. The reconciliation backlog that actually governs the
backlog escalation is `BacklogPolicy.current_backlog` (`backlog_policy.py:160` =
buffered + queue depth + retries), which **no MCP tool exposed**. Responders had
nothing correct to probe.

---

## The suspect verdict

The halt traces to the **07-20 cap-storm / claude-CLI-failure churn**: an
`error_during_execution` burst around 08:32:55Z produced the serious verdict that
tripped `_apply_halt` for run `e87d8e4a…`. That the *originating* verdict may have
been environmental (a CLI/cap storm, not a genuine data defect) is exactly the case
auto-resume-after-cooldown is designed for: a transient, suspect halt should
self-heal.

> **Note.** Task **2947** already removed the phantom-halt *fabrication* path, so a
> halt raised from here forward reflects a real verdict — a NEW halt is trustworthy
> and worth re-escalating loudly. The two tasks are complementary: 2947 made halts
> honest; 2920 makes them **loud, probeable, and self-healing**.

---

## The three fixes shipped (task 2920)

| # | Defect | Fix | Where |
|---|---|---|---|
| a | Halt escalation mis-filed as "backlog" and suppressed by the shared rate-limit bucket | **Distinct, un-suppressible halt escalation.** A `kind` ∈ {`backlog`,`judge_halt`,`wedge`} is threaded through `_route_over_limit` → `_maybe_write_escalation`; the id prefix is derived from `kind` (`esc-reconciliation-halt-` for a halt), and `_PolicyState.last_escalation_ts` is now a **per-kind dict** so each fault class rate-limits independently. `on_judge_halt` summary now reads `Reconciliation HALTED for {project}: {reason}`. | `backlog_policy.py:51-52` (prefixes), `:99` (per-kind dict), `:214-233` (halt summary), `:317-354` (per-(project,kind) rate-limit + id from kind) |
| b | Responders probed `get_queue_stats` and saw the wrong (durable-write) subsystem | **Expose the right metric where they already look.** `get_queue_stats` now adds `reconciliation_backlog` = `current_backlog(project_id)` **when** a `backlog_policy` is wired and a `project_id` is given; the backlog escalation `detail` now names that exact probe (`get_queue_stats(project_id=…).reconciliation_backlog`) and contrasts it with the durable-write-queue counts. | `server/tools.py:2628-2684`, `backlog_policy.py:203-210` |
| c | A halt never cleared; nobody acted on cooldown expiry for 2 days | **Auto-resume-with-grace after cooldown expiry, config-gated, loud.** New `reconciliation.auto_unhalt_after_cooldown: bool` (default `False`); new `Judge.cooldown_expired(project_id)`; the harness halt-check, when enabled **and** the cooldown has expired, logs a WARNING and calls `judge.unhalt` (seeding the existing post-unhalt grace and clearing the per-process `_halt_escalated` sentinel), then **falls through** to run the cycle. | `config/schema.py` (field), `config/config.yaml` (enacted `true`), `judge.py:603-619` (`cooldown_expired`), `harness.py:2197-2222` (auto-resume branch) |

### DECISION — why auto-resume, not escalate-only

The task asked the architect to choose between **(A) auto-resume-with-grace on
cooldown expiry** and **(B) escalate loudly but keep the halt latched**. We chose
**A**, gated by an opt-in config field defaulting `False` and enacted `true` for
this deployment, with the resume logged at WARNING.

- The incident's *defining* failure was "**nobody acted for two days**." Only
  auto-resume removes the human-in-the-loop dependency that actually failed;
  escalate-only would still be stuck-forever if the human is asleep.
- It is **safe** because the resume seeds the *existing* post-unhalt grace (which
  only suppresses the **trend** detector — `judge.py:541-548`), while a fresh
  serious verdict or the judge-infra-failure threshold **re-halts on the very next
  run** if the pipeline is still sick. So a genuinely-broken pipeline re-latches
  within one cycle and re-fires a now-**distinct, loud** halt escalation (fix a) —
  producing a visible ~30-minute drumbeat instead of either silent-forever or
  stuck-forever.
- Task 2947 already made a NEW halt trustworthy, so resuming-then-re-halting is
  honest signal, not churn.
- **Default `False`** keeps every existing test and every other deployment
  byte-identical; enacting `true` only in this repo's `config.yaml` closes the gap
  for the affected service. The resume is a deliberate, config-gated recovery
  action, so a **WARNING log** (not an escalation) is the right altitude — the
  loud/durable escalation is reserved for the (re-)halt itself.

---

## Clean-drain confirmation

The operator's manual `unhalt_reconciliation` + `trigger_reconciliation` at ~13:22Z
cleared the halt (seeding the 3-cycle post-unhalt grace) and the backlog drained.

**Going forward, verify a drain with the correct probe:**
```
get_queue_stats(project_id='dark_factory').reconciliation_backlog   # → ~0 when drained
```
NOT the top-level durable-write-queue `counts` (a separate subsystem that stays ~0
regardless of the reconciliation backlog). With `auto_unhalt_after_cooldown: true`
now enacted, a future halt of this kind auto-resumes ~30 min after it fires and,
if the pipeline is genuinely sick, re-halts with a distinct `esc-reconciliation-halt-`
escalation — no more two-day silence.

---

## Known bounds (not fixed here)

- **`get_queue_stats(project_id=None)` (global) omits `reconciliation_backlog`.**
  `current_backlog` is per-project, so the field is only emitted when a
  `project_id` is supplied. Watchers must probe **per-project**. (Emitting a
  global aggregate would require summing over registered projects — deferred.)
- **Auto-closing the pending halt escalation on `unhalt_reconciliation` is not done
  here.** Resolving the already-filed `esc-reconciliation-halt-…` escalation when a
  halt clears is the escalation-server / steward's responsibility, not
  `backlog_policy`'s. Possible follow-up: have `unhalt` (or the steward) mark the
  matching halt escalation resolved so the queue reflects the recovery.
- **`auto_unhalt_after_cooldown` is restart-tier.** It is read from config each
  cycle but is deliberately **not** in `RELOADABLE_FIELDS`; changing it requires a
  reconciliation restart.
