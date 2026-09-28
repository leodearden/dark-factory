# Diagnosis: the recurring fused-memory wedge (task 5544)

Two episodes prompted this — 2026-09-15 23:55 and 2026-09-16 12:04 — in which
fused-memory stopped answering every HTTP route while the process stayed up.
Both were previously root-caused as "wedged, cause unexamined".

Measurement says the premise was too narrow. **The wedge happened at least
eight times on 2026-09-16 alone**, and the two that prompted this task are the
minority the watchdog happened to catch. Everything below separates what was
measured from what is inferred; every causal claim is marked `Hypothesis:`.

Re-run any of it with `scripts/fm_wedge_forensics.py`, which exists because
reconstructing one episode by hand costs an afternoon:

```
journalctl --user -u fused-memory.service \
    --since "2026-09-16 12:03" --until "2026-09-16 12:10" \
    -o short-iso --no-pager | scripts/fm_wedge_forensics.py
```

`--user` is mandatory. `fused-memory.service` is a systemd **--user** unit, so
a system-scope `journalctl -u fused-memory.service` prints `-- No entries --`.
Reading that as "no evidence" is the trap this failure hid behind twice; the
script refuses such an input loudly rather than reporting "no stalls".

## Premise correction: what could NOT be measured

Item (a) of the task asks for CPU, memory and open-connection counts across
the two windows. **That data does not exist.** `scripts/install-load-sampler.sh`
is still a stub (`exit 64`, "not implemented until task 3592 lands"), so no
historical resource sampler ever ran, and both processes are long gone.

What survives is systemd's per-invocation `Consumed` line and the log timing.
Those turn out to be decisive, so the absence does not block the diagnosis —
but no claim below rests on a per-window resource metric, because none exists.

## (a) The named root cause

### The mechanism

Reconciliation and the HTTP server are two tasks on **one event loop in one
process**. `fused_memory/server/main.py` creates the harness loop
(`asyncio.create_task(reconciliation_harness.run_loop())`) and, further down,
the uvicorn server (`asyncio.create_task(server.serve(), name='fused_memory_primary')`).
So any *synchronous* work inside a reconciliation cycle stalls every route,
including the zero-I/O `/alive` the watchdog probes.

The cycle's own guard does not bound this.
`fused_memory/reconciliation/harness.py` wraps the cycle in
`asyncio.wait_for(..., cycle_timeout_seconds)`, and **`wait_for` cannot
interrupt synchronous code** — it fires at the next `await`. Each blocking site
below therefore runs to completion no matter what that timeout is set to.

### The decisive observation

In episode 2, fused-memory **never serviced SIGTERM** for the whole 90s
`TimeoutStopSec` and had to be SIGKILLed:

```
12:06:52  systemd: Stopping fused-memory.service...
12:08:22  systemd: State 'stop-sigterm' timed out. Killing.
12:08:22  systemd: Killing process 3655692 (uv) with signal SIGKILL.
12:08:22  systemd: Killing process 3655756 (python3) with signal SIGKILL.
12:08:22  systemd: Killing process 1289425 (git) with signal SIGKILL.
12:08:22  systemd: Consumed 2h 29min 21.259s CPU time, 3.2G memory peak, 934.4M memory swap peak.
```

asyncio installs its signal handlers **on the event loop**, so servicing
SIGTERM is itself proof the loop ran. A process that cannot service it for 90s
therefore has a *blocked* loop — not a merely descheduled one, which the
scheduler would run well inside that budget. This rules out both CPU
starvation and the task's own stated candidate (concurrent reconnect/notify
load from seven orchestrators).

It also explains why the verdict was `wedged` rather than `port-down`: a
blocked loop leaves the listening socket bound, so the kernel keeps accepting
and `probe_port` passes, while no route is ever served.

And `Killing process 1289425 (git)` is a **git child still alive in
fused-memory's own cgroup** at teardown — a synchronous subprocess in flight
on the blocked loop.

### The blocking sites

All three are plain `def` I/O reached from `async def` in the remediation tail
of a reconciliation cycle, with no `to_thread`/`run_in_executor`:

1. **`reconciliation/harness.py::ReconciliationHarness._run_remediation_pass`**
   calls `escalation/queue.py::iter_all_escalation_paths`, which globs the
   queue root *and* `rglob('esc-*.json')`s the entire dated escalation archive
   — which `queue.py` itself documents as "an archive shared across 7+
   projects" — then `read_text()`s and JSON-parses every hit inline. Unbounded;
   minutes-scale on a large archive.
2. The same pass's local `_task_is_live` helper reaches
   `services/live_workflow_detector.py`, which runs up to **three
   `subprocess.run(['git', ...], timeout=_GIT_TIMEOUT)` calls per finding**
   with `_GIT_TIMEOUT = 10`, inline on the loop. This is what the SIGKILLed
   `git` child above is consistent with.
3. **`reconciliation/harness.py::ReconciliationHarness._finding_recently_resolved`**
   repeats the whole-archive scan once per finding on its fallback arm.

`Hypothesis:` these sites are the cause of the stalls. The evidence for it is
circumstantial but consistent on every axis checked — the blocked-loop proof
above, the surviving `git` child, and the pre-stall context of every episode
sitting in the reconciliation path. Nothing directly observed *which* call was
executing during a silence, and no stack was captured; a faulthandler dump
would be needed to close that gap.

### The systemic enabler

The systemd `WATCHDOG=1` heartbeat was deliberately moved onto a dedicated OS
thread (`server/main.py::_watchdog_thread_loop`, task 1731) so a busy loop
could not miss it. The side effect is that a **wedged** loop also keeps the
heartbeat alive, so systemd itself never restarts fused-memory for this. The
external `scripts/orchestrator-watchdog.py` is the only thing that catches this
class of failure at all — and, per the frequency data below, it catches a
minority of it.

### Frequency: the premise was too narrow

Running the analyzer over 2026-09-16 12:10–18:00 (305,465 lines) finds **six
more stalls above 90s**, on the same day:

| stall | duration | outcome |
|---|---|---|
| 12:16:07 → 12:18:47 | 160s | self-recovered |
| 15:14:26 → 15:16:06 | 100s | self-recovered |
| 15:21:13 → 15:23:06 | 113s | self-recovered |
| 16:04:42 → 16:06:28 | 106s | self-recovered |
| 16:08:08 → 16:10:04 | 116s | self-recovered |
| 16:15:05 → 16:17:21 | 136s | restarted by the watchdog |

Five of six resolved with no systemd action whatsoever. **Self-recovery is the
common outcome**; the two "episodes" are the tail of a continuous distribution
that happened to exceed the watchdog's ~121s detection floor. The 100–136s band
sits right at that floor, so most of this is invisible to it by construction.

The two restarts in that window report `Consumed 29min 6.600s CPU time, 2.7G
memory peak` and `20min 30.697s CPU time, 2.3G memory peak` — far below episode
2's `2h 29min / 3.2G / 934.4M swap`. The wedge occurs at ordinary resource
levels, not only in a long-lived process that has accumulated memory.

### Co-factors, and why the count is not a rate

`esc-5544-1` (filed by the architect after the plan froze) raises a caveat this
frequency data needs, and it is a real one.

Memory `a398ae1a` records that on 2026-08-19, **twelve concurrent `journalctl`
processes** scanning this same unit over a 4.5GB journal burned 906% CPU — ~9
of 32 cores, the largest single consumer on the box — several running 7-23
minutes, driving host load 63 -> 94 -> 130. It attributes that storm to
escalation triage by agent sessions independently re-deriving one
investigation. Task 5544's own description notes the 12:04 episode "was
observed live ... during the escalation drain that filed this task".

This session corroborates the cost first-hand: a `--since 2026-09-13` scan of
this unit ran **over 40 minutes at ~80% CPU and had to be abandoned**. The
bounded 6h scan that produced the table above took 2m40s for 305,465 lines.

`Hypothesis:` the blocking sites are filesystem- and subprocess-bound rather
than CPU-bound-in-Python, so host I/O contention lengthens each one, and
concurrent journal forensics during a drain could stretch a normally-survivable
remediation pass into a multi-minute stall. If that holds, **episode count is
not a stationary rate** — it is partly a function of agent triage activity, and
the eight stalls above should be read as "at least eight on a day with triage
activity", not as a baseline. Potentially self-reinforcing, too: diagnosing a
wedge by streaming the unit's journal is itself load that can provoke one.

What this does NOT change: the mechanism. Episode 2's SIGTERM went unserviced
for the full 90s, which is a blocked loop regardless of host load, and a
descheduled process would have been scheduled well inside that budget. Nor does
it change the remedy — an I/O-bound site that is sensitive to host contention
is an *additional* argument for getting it off the event loop.

Worth noting the mitigation is already in the deliverable: the analyzer is
stdin-driven and reads one extract rather than re-streaming the journal per
question, which is exactly what memory `a398ae1a` recommends.

### Hypotheses disconfirmed

- **Seven-orchestrator reconnect/notify load.** Episode 2 had no fleet restart
  in flight, and a SIGTERM unserviced for 90s is a blocked loop, not connection
  pressure.
- **`shared.cli_invoke` / the agent CLI.** Uses `asyncio.create_subprocess_exec`
  throughout and offloads transcript reads via `to_thread`; no
  `subprocess.run`/`Popen` on the loop.
- **The `task_curator ... latency_ms=100819` line.** That is awaited latency
  around an awaited subprocess, not 100s of loop occupancy.
- **The index-drift detection read.** One `CALL db.indexes()` plus a set-diff;
  not a graph scan. Its `index_drift_escalation_suppressed` line is emitted
  *after* its bounded pending-queue glob returns, and appears immediately
  before the silence in both episodes — so the stall begins downstream of it.
- **The MCP retry path (`orchestrator/src/orchestrator/fm_retry.py`).** The
  existing 120s window absorbed the outage, as the task description already
  established. This evidence positively refutes "harden the MCP client" as the
  remedy.

### Aggravating context, not the cause

Swap peak nearly doubled in 12.5h (529.9M → 934.4M) at ~3.2–3.3G RSS.

At 12:01:35, ~2.5 min before the stall, fused-memory logged
`graphiti_core.driver.falkordb_driver - ERROR - Error executing FalkorDB query:
Query timed out`. Note the attribution: that is fused-memory's **client**
giving up, not the server reporting a problem — `docker logs docker-falkordb-1`
contains **zero** error or timeout lines for the whole of 2026-09-16. So it is
evidence of a slow round trip as seen from inside the stalling process, which
is as consistent with a loop that is not getting back to its awaits as it is
with a slow store.

FalkorDB also BGSAVEs on a 300s cycle, with one running 12:03:30–12:03:52 local
(11:03:30–11:03:52 in the container's UTC log) — seconds before the silence
began.

Recorded as context. None of it explains a 90s unserviced SIGTERM, and the
ordering does not establish direction.

## (b) Was the 23:59:30 fleet restart a deploy?

**The merge-coordinator explanation is refuted on ordering**, not merely
unreconciled. Measured:

- The fleet restart ran **23:59:29 → 00:00:43**, all seven units sequentially
  (first `Stopping orchestrator-autopilot-video.service` at 23:59:29).
- The nearest merge, `2e18ca0419 Merge task/5368 into main`, has committer
  timestamp **00:00:50** — 81s *after* the restart began and 7s *after* it
  finished. A merge-landed coordinator cannot have triggered a restart that
  completed before the merge existed.
- The previous merge, `016f3547a8`, landed at 22:20:20 — 99 minutes earlier.
- The fleet-deploy clock (`data/orchestrator/last_redeploy_orchestrator.json`)
  reads `2026-09-14T12:31:42+00:00`, so this sweep did not stamp it.
- **No log line names the invoker.**

Also measured, and relevant to the wedge itself: the fleet restart *followed*
fused-memory's own restart (`Started` at 23:58:42) rather than preceding it, so
it cannot have caused episode 1.

**Unreconciled, needs its own task:** who issued the 23:57:10
`Stopping fused-memory.service` in episode 1. It was **not** the watchdog — its
streak only reached 3/3 at 23:58:03, by which time fused-memory had already
been stopped (`Stopped` at 23:58:00), which is why that third verdict reads
`port-down` rather than `wedged`. Nothing in the user journal for 23:56:55–
23:57:20 names an actor. The same gap covers the 23:59:29 fleet restart.

## (c) What the outage cost

Episode 2, from the two journals, decomposing **exactly** to the 302s total:

| term | interval | cost |
|---|---|---|
| stall before the first failed probe | 12:04:01 → 12:04:51 | 50s |
| watchdog detection streak (1/3 → 3/3) | 12:04:51 → 12:06:52 | **121s** |
| teardown (ignored SIGTERM → SIGKILL) | 12:06:52 → 12:08:22 | 90s |
| startup | 12:08:22 → 12:09:03 | 41s |
| **total silence to recovery** | 12:04:01 → 12:09:03 | **302s** |

Episode 1 does **not** decompose the same way — the terms differ in kind, not
just in size, so it gets its own table rather than a one-line echo of the one
above. Non-overlapping, to a 212s total:

| term | interval | cost |
|---|---|---|
| stall (silence, start to `Stopping`) | 23:55:10 → 23:57:10 | **120s** |
| teardown (SIGTERM serviced, clean stop) | 23:57:10 → 23:58:00 | 50s |
| stopped → restart issued | 23:58:00 → 23:58:03 | 3s |
| startup | 23:58:03 → 23:58:42 | 39s |
| **total silence to recovery** | 23:55:10 → 23:58:42 | **212s** |

Two differences from episode 2, both real rather than presentational:

- **There is no detection term at all.** The watchdog's streak did not drive
  this restart: it reached 3/3 only at 23:58:03, three seconds *after*
  fused-memory was already `Stopped` — which is exactly why that third verdict
  reads `port-down` and not `wedged` (see (b): the actor is unreconciled). The
  whole 120s silence is therefore stall, not stall-plus-detection.
- **The stopped → restart gap is visible here and 0s in episode 2**, where
  systemd's `Starting` lands in the same second as the SIGKILL. That is why
  episode 2's table has four terms and this one has five.

Both tables are checkable against the tool: run
`scripts/fm_wedge_forensics.py` over either window and its teardown and startup
terms are the same numbers (episode 1, 50s and 39s; episode 2, 90s and 41s),
measured from fused-memory's journal alone. Only the detection term needs the
watchdog's journal, which is why the analyzer reports it as unavailable rather
than guessing it.

In **episode 2** the detection streak is the largest single term, at 121s
against the teardown's 90s. (This corrects the plan's own summary, which quoted
the 121s figure while asserting the teardown was largest.) Episode 1 has no
detection term at all, so its largest term is the 120s stall.

### Recommendation: do NOT shorten the port-down streak

In the one episode the streak actually drove, it is the largest term — so
shortening it is the obvious move. The measurements say it is the wrong one.

1. **It would convert self-recovery into restarts.** Five of the six stalls
   above self-recovered, in the 100–136s band — precisely the range a shorter
   streak would begin restarting. Dropping 3 → 2 ticks would have restarted the
   single shared MCP server roughly five extra times in six hours, for stalls
   that resolved on their own.
2. **It re-arms a regression the detector exists to suppress.**
   `scripts/orchestrator-watchdog.py`'s own comments record that a false
   `wedged` verdict previously "got the single shared MCP server restarted for
   nothing" — the defect tasks 3764 (the streak) and 3765 (`/health` → `/alive`)
   were filed to fix — and a controlled 2026-08-06 window where `/health`
   exceeded 15s four times in 2h with peaks of 23.5s and 25s+, *every one
   self-recovered*.
3. **It treats the symptom.** Detection latency only matters because the loop
   blocks for minutes. Removing the blocking is strictly better than noticing
   it faster, and is the follow-up below.

The 90s teardown is also not worth attacking directly: it is `TimeoutStopSec`
elapsing in full precisely *because* the loop is blocked. Unblock the loop and
that term disappears on its own.

## Follow-up

The hardening is deliberately out of scope here (the task instructs that a
hardening task follow only if the diagnosis finds something reproducible, and
be scoped to what it actually shows). Filed as ticket
**task 5550** (ticket `tkt_0RTQQ7PT7P515J3D4FW1668RGH`, which the curator
resolved to a created task).

Scope: offload or bound the three blocking sites named above, using the remedy
shape this codebase already establishes — `await asyncio.to_thread(...)`, as
`reconciliation/targeted.py` and `middleware/task_curator.py` already do for
this same class of work, and as the `NOTE (blocking I/O)` in
`reconciliation/harness.py` already prescribes for a sibling site. It must
*not* touch `orchestrator/src/orchestrator/fm_retry.py`.

## Related escalations

- `esc-5544-2` — three plan steps stated stall magnitudes 1–2s off what the
  journal says; the tests assert the measured values.
- `esc-5544-3` — the six additional stalls, filed when found.
- `esc-5544-1` — the load-amplifier co-factor, weighed in “Co-factors, and why
  the count is not a rate” above.
