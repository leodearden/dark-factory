# Residue-loss evidence, 2026-08-23 → 2026-08-27

Companion to `toolcall-markup-containment-prd.residue-loss-2026-08-27.jsonl` —
**every** refused call whose payload the markup guard could not file, for the
whole life of the guard to date.

## Why this is committed

PRD §4 contract C2 and boundary row **B5** both promise that an unrepairable
call is *"rejected, and an escalation filed carrying the full raw payload so
nothing is discarded even if the caller never retries."* For these eleven, no
escalation was filed: `fused_memory.server.markup_guard::_file_residue`
resolved `project=None`, so `_resolve_project_root` returned `None`, and the
record went to a log line instead of the queue. These records are the primary
evidence that C2/B5 do not hold for that class.

## This is the COMPLETE population, not a sample

The guard first installed at 2026-08-23 04:33:58; `/var/log/syslog` begins
2026-08-23 00:00:37, and every rotated `syslog.N` back to 2026-07-26 contains
zero residue lines **and** zero guard-installed lines — a proper control showing
that zero is non-deployment, not missing logging.

| Path | Events | Share |
|---|---|---|
| Refused, payload **destroyed** (no escalation) | **11** | 73.3% |
| Refused, payload preserved in an escalation | 4 | 26.7% |

So the preservation property that C2 relies on failed for **roughly three out of
four** refusals over the guard's entire deployed life.

## The eleven

All eleven are `add_memory.content`. Not one is any other tool or parameter.

| Observed (BST) | Payload identity | Absorbed calls |
|---|---|---|
| 2026-08-23 16:11:34 | `reify` | 2 |
| 2026-08-23 17:25:55 | `reify` | 1 |
| 2026-08-24 15:20:30 | `dark_factory` | 1 |
| 2026-08-24 16:13:21 | `reify` | 2 |
| 2026-08-24 16:56:41 | `dark_factory` | 2 |
| 2026-08-24 17:43:38 | `reify` | 2 |
| 2026-08-25 03:03:43 | `dark_factory` | 1 |
| 2026-08-25 19:42:21 | `dark_factory` | 2 |
| 2026-08-27 00:10:46 | `reify` | 2 |
| 2026-08-27 00:58:01 | `reify` | 1 |
| 2026-08-27 09:49:11 | `dark_factory` | 1 |

Eleven refused calls, **seventeen** intended memory writes: six payloads each
absorbed a whole second `add_memory` call.

Note the two identity columns in the JSONL. `guard_resolved_identity` is what
the guard saw — `null` in every case. `identity_present_in_payload` is what a
scan of the record's own `raw_value` recovers — a real `project_id` in **11 of
11**, each resolving against the live 9-project registry. The routing key was in
the guard's hand every single time.

## Cause, in one line

The leak over-consumes *forward* from the leaking parameter's closing tag, so it
absorbs every parameter declared **after** it. `add_memory(content, project_id,
…)` declares `content` first, so `project_id` is always eaten — which is why all
eleven losses are that one tool and parameter. `submit_task(project_root, …,
description, …)` declares `project_root` first, so it survives: a
`submit_task.description` refusal at 2026-08-27 18:15 preserved its payload fine
as `esc-markup-residue-2`. Whether a refused payload survives currently depends
on parameter declaration order.

## Where these came from, and the retention trap

The guard's own log line claims *"THIS LOG LINE is the only copy"*. That is
false twice over. journald is not the only sink — `journald.conf`'s
`ForwardToSyslog` feeds rsyslog, and `/var/log/syslog` carries the identical
lines for roughly two to four weeks longer. Six of these eleven had **already
aged out of journald** and were recovered from syslog alone.

The journald floor moves much faster than its configuration suggests: the
`SystemMaxUse=32G` drop-in never binds (actual usage 4.6G) because the cap that
actually fires is `SystemMaxFiles=100` — exactly 100 journal files exist. The
floor advanced several hours during this investigation alone.

Line lengths make the stakes concrete: destroyed-payload lines run 2216–3928
characters because they carry the whole `raw_value` inline, while preserved-path
lines are 383–388 characters (just a pointer to the escalation). For these
eleven, syslog held the only copy of the content — and it ages out at the next
weekly rotations. That is why they are committed here.

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

Checked for the seven writes in the five journald-visible events: all seven were
re-landed by their own authoring agent, median under a minute (+6s, +16s, +23s,
+51s, +58s, +68s, +11m57s). Nothing was permanently lost there. The one slow
recovery came back with roughly a third of its claims gone, its `agent_id`
changed and its category demoted — so the residual harm is content attrition
that scales with retry latency.

**The remaining ten writes, in the six syslog-only events, have not been checked**
against the corpus. If the same pattern holds they were re-landed too; that is an
assumption, not a measurement.

Either way the agent retry loop, not this preservation path, is what saved the
data. The path C2 relies on contributed nothing in 11 of 15 refusals.
