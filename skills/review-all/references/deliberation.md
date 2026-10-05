# Deliberation — from committed report to filed program

Phase 6 of `/review-all`. The run ends at the report; this protocol turns its structural findings into decisions, a program doc and `/prd` sessions, and closes with the contract §11 trigger chain. It is human-attended throughout; never run it under AFK autonomy.

## What goes to the human

Everything structural (contract §6) is deliberated; mechanical findings were already filed by the run. Order the agenda by the synthesis matrix, then take contested items first:

1. a stream with `open_choices`;
2. a contradiction the synthesis could not resolve;
3. an invariant candidate;
4. a `weakened` finding the matrix still ranks in a cell's top three;
5. a critic join error the lead overruled;
6. any stream sized `L` or risk `high`;
7. any stream whose `deletable` is empty (a remedy that adds without removing is a cost increase until shown otherwise).

Uncontested streams (one option, confirmed findings, size S/M, low risk) are presented as a block for a single yes/no.

## How an item is put

One screen per item, the same shape every time, so decisions are comparable:

- **Finding(s)**: key, anchor, statement, the measurements with their rules, verdict (weakened notes inline).
- **Next change**: the scenario the seat wrote — what an agent does here next and what it costs today.
- **Options**: two to four, each with trade-off (what it costs to build, what it risks breaking) and **long-term consequence** (what the next agent change costs a year on under this option; what becomes deletable; which invariant it would enforce or weaken). Include "accept as standing" as an option when the cost is real but contained — it files an `accepted:<task>` disposition with a reason, never silence.
- **Recommendation**: only when the analysis converges; otherwise say it does not and why.
- **Invariant candidate** (when present): the checkable form and fixture sketch; the existing INV it extends, if any. Promotion is in checkable form only, via the four-site lockstep edit `CONTRIBUTING.md` §6 describes (quality doc §Relationship to the design invariants).

Use `AskUserQuestion` for a choice independent of context; otherwise raise it inline. Push back on an unstated assumption in the framing. Do not relitigate a ruling recorded in a previous program doc unless the finding's evidence changed — cite the ruling.

## Recording

Each ruling is recorded immediately, before the next item:

- in the program doc §Resolved decisions: the ruling, the options considered, the consequence that decided it, the keys it covers;
- as one `decisions_and_rationale` memory with `entities=` naming the tasks or program once they exist (`project_id`, `agent_id` per the overlay); the program doc is the home, the memory is the pointer;
- an "accept as standing" ruling files a task carrying `x_finding_key`, `x_finding_run`, `source="review-all"` and then cancels it with `metadata.x_acceptance_reason = "<the ruling>"` (contract §7–§8); a `deferred` task is postponed work, never an acceptance, and a `cancelled` task without the reason reads as abandoned. The next run's step 1b keys on that field.

## The program doc

Written once the agenda is exhausted, at the path `report-format.md` §Artefact 3 names, with every section it lists. Streams are sized for one `/prd` session each: a design-heavy stream becomes `spawn`, a stream whose remedy is settled and whose tasks are mechanical becomes `agent`. Each stream lists its finding keys; the G4 table gives every shared artefact one owner before any PRD is authored. Commit it with `git commit --only`.

## Launching `/prd` per stream

- **`agent` streams**: one `/team` run, one seat per stream, each seat running `/prd` author + decompose against the program doc with the overlay's conventions; seats are `opus`/`high` for authoring, and the lead verifies filed batches with `get_task`, not `search_tasks`. Concurrent sessions in one checkout commit with `git commit --only <path>`.
- **`spawn` streams**: one interactive session per stream via `skills/spawn/SKILL.md`, with a brief file under the overlay's brief directory that names the program doc, the stream row, the finding keys and the freshness clause. The hotspot program (`plans/bug-hotspot-remediation-program-2026-07-06.md`) is the proven shape.
- Every `/prd` session stamps `metadata.x_finding_key` (list) and `metadata.x_finding_run = <run_id>` on each task it files, runs the contract §8 dedup before filing, and appends its PRD path and task anchors to the program doc's FILED section.

## Closing: the trigger chain

The contract §11 chain is the program's last act, filed exactly as `skills/_shared/filing-the-trigger-chain.md` §§0–5 says, with `skill = "review-all"` (the bare skill name, per its §0) and this run's `run_id`; nothing about its shape is restated here. What is specific to this skill:

- **Timing.** The gate is filed (§4 interim, or §3 final when the §2 probe in the dark-factory checkout passes) when the first `/prd` stream files; every later `/prd` session adds its own `high`/`critical` tasks as dependencies as it files; `commit_planning` happens only once the program doc's FILED table is complete.
- **Supersede first.** An older open `review-all` chain (§1) is superseded per §5 before this run's gate is filed. When this run was launched from that chain's `review_due` escalation — the normal path once `plans/completion-driven-triggers-prd.md` lands, its id having arrived as `--launched-from` (SKILL.md Phase 0 step 5) — §5 step 2 is where it is resolved, with `ran as <run_id>`.
- **No tasks filed** (program filed nothing): no chain, and the report's `## Method` `extra` says so (§0).

Record the gate id(s) in the program doc's FILED section, then write the run's `observations_and_summaries` memory with `entities=` naming the gate task(s). If the store is unreachable, file nothing and say so in the program doc; the next session files the chain.
