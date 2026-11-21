# Common brief: cross-project weights team, 2026-11-12 (read this first)

## Why this team exists

Leo (the operator) runs several software projects through the Dark Factory orchestrator
(`/home/leo/src/dark-factory`), each with its own orchestrator service, on one 32-thread workstation plus a
16-thread laptop (`ssh leo-laptop`) used for verifies. Nothing arbitrates between projects. On 2026-11-08T16:04Z
he halted the schedulers of reify, solar-challenge and solar-challenge-platform to give Dark Factory the CPU.
He now wants to replace that whole-project pause with **weights between projects, on CPU and on the agent usage
allowance** (the shared pool of Claude accounts that caps out), and asked for an investigation and design
discussion, with a PRD to follow once the open questions are answered.

The lead's brief is `/tmp/team-gates-1112/followups/cross-project-weights-brief.md`. Read it; it lists the seven
open questions. Your seat supplies evidence for some of them. Your output is evidence for the lead, not prose
for Leo. A wrong figure that looks right is the expensive failure; an honest "could not determine" is cheap.

## Prior evidence (claims with dates, not current truth)

`/tmp/team-gates-1112/REPORT.md` is this morning's study; where it and a seat file differ, trust `REPORT.md`.
Beside it: `capacity.md`, `interventions.md`, `admission-policy.md`, `sched-policy.md`, `flow.md`, each with
re-runnable read-only scripts in a subdirectory of the same name (reuse them rather than rewriting).
`/tmp/team-gates-1112/COMMON.md` section "Data sources" describes every database and its traps (event store
`data/orchestrator/runs.db` with `events`, `invocations`, `account_events`; task store
`.taskmaster/tasks/tasks.db`; `data/load-samples.db` whose `ts` is an epoch integer; PCP archives). Read that
section before querying anything. Ask each database for its schema; never guess a column.

Windows, so seats can be joined: **W0** = 2026-10-25T00:00Z .. 2026-11-08T16:00Z (all projects running).
**W1** = 2026-11-09T14:13Z .. now (three projects paused). State the window per figure.

Facts the lead already established today (verify anything you depend on):
- Nine orchestrator units run on the workstation under `systemctl --user`: dark-factory, reify,
  my-solar-challenge, solar-challenge-platform, autopilot-video, know-live, pump-web-ui, scoping-intake,
  stream-touch. All sit directly under `app.slice`; none has `CPUWeight` set.
- `/home/leo/src/reify/config/usage-accounts.yaml` is a symlink to Dark Factory's
  `config/usage-accounts.yaml`; pump-web-ui's config names the same Dark Factory file. solar-challenge and
  solar-challenge-platform have no `config/usage-accounts.yaml` in their checkout.
- Tasks 3394, 5139, 5529, 8261, 8791, 8792, 8793 are all `pending` in Dark Factory's task store.

## Hard rules

- READ-ONLY. Do not edit, commit or create any file under any project checkout. Write scratch and your result
  only under `/tmp/team-weights-1112/` (scratch in a subdirectory named after your seat).
- Do not change config, task status, pins, dependencies, queues, schedulers, cgroups or any service. Do not
  restart anything. Call no MCP tool that writes. Do not file tasks or write memories.
- Nothing CPU-heavy: no tests, builds or type-checks. Bound every query by time; never load a large table whole.
  On the laptop only cheap reads (`uptime`, `/proc/pressure/*`, `systemctl show`, logs).
- Never `git stash`. Never kill a process by name; stop only a pid you launched.
- Do not start subagents. Do the work yourself.
- Never `find .` or Glob from a checkout root (each holds a full tree copy per worktree). Use
  `git -C <checkout> ls-files -- '<pattern>'` and Grep with an ABSOLUTE path.
- Never print the contents of an accounts file, token or credential. Account NAMES (max-b, max-h ...) are fine.
- The running orchestrator can lag `main`: Dark Factory's service started 2026-11-10T23:48Z. Say which code you
  read (main at what commit) and flag where the running service may differ.
- If the brief is under-specified or you hit a blocker, say exactly what and stop that line; do not widen scope.

## Deliverable shape

Write your full result to `/tmp/team-weights-1112/<seat>.md`. Every measured claim uses:

```
F1. <claim in one sentence>
    figure: <number + unit>            rule: <what was counted, what was excluded>
    window: <UTC from..to>             population: <n rows / which projects / which hosts>
    kind: measured total | ratio (of what to what) | bound | projection
    evidence: <the query or command, or path/to/file.py::symbol>
```

One `rule:` per figure, not one per seat. Cite code as `path/to/module.py::symbol`, never a bare line number,
and name only symbols you actually read. Then a `mechanisms` section (how things work, from code), an `options`
section if your brief asks for one, and an `unknowns` section. Six well-evidenced findings beat fifteen thin
ones. Before returning, re-run the query behind your three most important figures and confirm the file matches.

Your final message is a compact summary: the path of your result file, your five most important findings each
with its figure and rule, and your unknowns.
