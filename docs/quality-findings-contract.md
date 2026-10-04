# Quality findings contract

The single normative contract for every instrument that produces quality
findings about a factory-operated project: `/review`, `/hotspot-survey`,
`/census` and `/review-all`, and any later instrument. Ratified by Leo on
2026-10-04 (rulings recorded in the session that wrote this file). The
skills point here and do not restate it (INV-9).

The instruments answer different questions from different evidence, and
that is deliberate. What this contract makes common is everything that
lets their results be joined, carried forward and not re-derived: how a
finding is identified, graded and located; what a report must pin; how a
finding becomes work without filing a task that already exists; where a
disposition lives; and how one run schedules the next.

## 1. What a finding is

A finding is a claim that a specific part of the code raises the cost or
risk of the next change, in the sense of `docs/code-quality.md`, located
precisely enough that another agent can re-verify it at a later commit.
Everything else a run learns (counts, trends, caveats, method notes) is
report prose, not a finding.

Every finding carries these fields, in the instrument's committed artefact:

| Field | Shape | Rule |
|---|---|---|
| `key` | `fk-<12 hex>` | §2. Stable across runs and instruments. |
| `area` | one area key | §3. |
| `anchor` | `path/to/file.py::symbol`, `path/to/file.py`, or `slug:<kebab>` | Repo-relative. `::symbol` is the enclosing top-level or class-level definition. `slug:` only for a non-code subject (a prompt, a contract, an operating practice). |
| `tags` | list of `h1`..`h14`, `comments`, `tests`, `inv-<n>`, `kind:<instrument kind>` | At least one. `h<n>` are the fourteen heuristics of `docs/code-quality.md`; `comments`/`tests` its two stances; `inv-<n>` a `docs/legibility/design-invariants.md` id. |
| `severity` | `high` \| `medium` \| `low` | §4. |
| `evidence_source` | `present-tree` \| `fix-history` \| `agent-transcripts` \| `metrics` | What the claim rests on. A finding corroborated by two sources lists both. |
| `statement` | one paragraph | The claim, with the measured facts that support it. No proposal in this field. |
| `proposal` | optional paragraph | What would remove the cost, if the instrument has one. Structural proposals are inputs to deliberation (§6), not instructions. |
| `verdict` | `confirmed` \| `weakened` \| `refuted` \| `unverified` | Set by the instrument's verification step; `unverified` only when that step did not run, and the report says why. |
| `disposition` | §7 | The finding's standing after triage. |
| `first_seen` | run id | §5. The run that first recorded the key. |
| `last_seen` | list of run ids | Every later run that re-observed the key appends its own (§9). |
| `supersedes` | list of keys, usually empty | §2. The keys this finding replaces (anchor moved, legacy positional id). |
| `sub_area` | optional free text | A finer location an instrument uses internally (a hotspot cluster); never part of the key. |

## 2. The finding key

```
canonical = f"{area}|{anchor}|{primary_tag}".lower()
key       = "fk-" + sha256(canonical.encode()).hexdigest()[:12]
```

`area` is the plain area key of §3 (never `<area>/<sub>`). `primary_tag`
is the first entry of `tags`. The anchor is normalised before hashing:
repo-relative, forward slashes, a trailing line pin removed (`:N`, `:N-M`
and comma lists of them), surrounding whitespace stripped. Two instruments
that locate the same cost at the same anchor under the same heuristic
therefore mint the same key without talking to each other, and a re-run
reproduces the key of an unchanged finding.

One implementation mints every key: `shared/src/shared/finding_key.py`
(`plans/census-incremental-prd.md` leaf L3). Until it lands,
`skills/hotspot-survey/scripts/findings_artefact.py key` is the interim
reference; the leaf that lands `finding_key.py` carries a parity test
against that subcommand on a committed fixture, and any other code that
computes a key is replaced by an import of `finding_key.py` when it lands.

A key names a *location and a lens*, not a wording, so a reworded
statement keeps its key. A finding whose anchor moves (symbol renamed,
file split) gets a new key; the run that notices records the old key under
`supersedes` so the chain is followable.

Display ids (`F3`, `merge-queue.4`, `R2`) are per-run conveniences and may
appear beside the key, never instead of it.

## 3. Area vocabulary

The project's area keys are the `subprojects` keys of its
`review/briefing.yaml`, plus `repo` for a cross-cutting finding. There is
no second list. An instrument that works at a finer grain (hotspot
clusters) carries that grain in `sub_area` (§1) and still writes a defined
key in `area`, so keys join across instruments. One rule maps a path to an
area — the project's `review/briefing.yaml` subproject whose member
directory contains the path, with the project's `/review-all` overlay
naming the rule for paths under no member — and every instrument uses it.
A project whose briefing omits a workspace member has a briefing defect,
fixed with `/review-briefing`, not a new area list.

## 4. Severity

One scale, equal to task priority: `high` (the next change in this area
is likely to be wrong or expensive without this), `medium` (a real cost,
contained), `low` (worth doing when the area is next touched). `critical`
is reserved for operational breakage (red main, halted lane) and is never
a quality-finding severity. Instruments that previously used
`warning`/`info` or `impact`×`effort` map onto this scale in their own
reference text and emit only these three values.

## 5. What a report must pin

Every instrument run writes one committed report: a machine-readable
record (JSON) and the markdown rendered from it in the same run, sharing
one basename that carries the run's date and `-<n>` suffix (each
instrument's report-format names its exact pattern). The record is the
source; the rendering is never edited by hand. Its method header is a
`## Method` section whose first element is a fenced ```yaml block, so a
script reads it with one YAML load; the block carries these keys and no
instrument-specific ones outside a `extra:` map:

- `run_id`: `<instrument>-<project_id>-<YYYYMMDD>[-<n>]` (`-<n>` for a
  second run on one day; a rerun never overwrites an earlier report).
- `as_of_sha`: the main commit every anchor was verified against. The
  whole run reads one tree.
- `since`: the previous run's `as_of_sha` (or `none`), so "changed since"
  is a `git diff` and never a date.
- `evidence`: the corpora read and their sizes.
- `verification`: counts of `confirmed / weakened / refuted / unverified`.
- `cost`: agents, subagent tokens where the runtime reports them, wall
  clock.
- `inputs_consumed`: the `run_id` of every other instrument's report this
  run read (§9).

Report homes: `/review` → `review/reports/` (tracked, not gitignored);
`/hotspot-survey` and `/review-all` → the project's plans directory;
`/census` → the project's plans directory plus
`docs/legibility/confusion-codebook.yaml`, whose entries carry the
`finding_key` of the finding they record.

## 6. From finding to work

Two routes, chosen by the finding's class, recorded on the finding:

- **Mechanical** (one anchor, no design choice, a competent agent can fix
  it from the statement): the instrument files a curator ticket itself,
  after the dedup protocol in §8.
- **Structural** (spans modules, or chooses between designs, or changes a
  contract): the instrument files nothing. The finding goes to
  deliberation with the human, then a program doc, then `/prd`, which files
  tasks under `planning_mode` and stamps the finding keys (§8) on them.

A finding whose proposal is "split this file" or "reduce this number" is
structural by definition; heuristic 14's measurement protocol in
`docs/code-quality.md` applies before any split is proposed.

## 7. Dispositions

Exactly one of:

| Disposition | Meaning | Where the truth lives |
|---|---|---|
| `open` | confirmed, not yet routed | the report |
| `filed:<task id>` | work exists | the task store |
| `accepted:<task id>` | will not be fixed, with an owner and a reason | a `cancelled` task carrying the key and `x_acceptance_reason` (§8); a `deferred` task is work postponed, never an acceptance |
| `refuted` | verification rejected the claim | the report, with the refutation |
| `fixed:<sha>` | re-verification at `as_of_sha` found it gone | the report |

There is no separate ledger. The task store is the home of every
disposition that is somebody's decision, so `review/briefing.yaml`
`known_gaps` entries keep only a pointer to the accepting task; a
`known_gaps` entry without one is a defect the next `/review` run reports.
An `accepted` disposition is re-verified like any other finding: if the
accepting task is later `done` or the anchor changes, the finding reopens.

## 8. Filed-task metadata and the dedup protocol

Every task filed from a finding carries, under the Tier-C namespace of
`docs/task-authoring.md`:

```
metadata.x_finding_key   = "fk-…"            # one task may carry several, as a list
metadata.x_finding_run   = "<run_id>"         # the run that filed it
metadata.x_supersedes_task = <task id>        # only when §8 step 1c applies
metadata.x_acceptance_reason = "<text>"       # only on an accepting task (§7), set when it is cancelled
metadata.source          = "<instrument>"     # existing convention
```

Before filing, in this order, and the report records the step that
decided each finding:

1. **Key lookup.** `find_tasks_by_metadata(project_root, key="x_finding_key",
   value=<key>)` (fused-memory MCP; until it lands, a read-only forensic
   query per `CLAUDE.md` §"Forensic reads of tasks.db" that tests
   membership with `json_each(metadata, '$.x_finding_key')`, since the key
   may be a scalar or a list).
   a. A task in `pending`, `in-progress`, `blocked`, `deferred` or
      `merge-deferred` → disposition `filed:<id>`; do not file.
   b. A `cancelled` task carrying `x_acceptance_reason` → `accepted:<id>`;
      do not file. A `cancelled` task without it was abandoned, not
      accepted: treat as absent and continue to step 2.
   c. A `done` task → re-verify at `as_of_sha`; if the finding is still
      present, file with `x_supersedes_task` and say so in the title.
2. **Semantic lookup.** `search_tasks` with a paraphrase of the statement at
   `score_threshold ≥ 0.6`; its corpus excludes `deferred` tasks, which is
   why step 1 runs first. A plausible match is read with `get_task` before
   deciding.
3. **Curator.** The ticket path's own `candidate_key` dedup is the last net,
   never the first.

A run that cannot perform step 1 (store unreachable) files nothing and
says so in its report.

## 9. Consumption between instruments

Each instrument reads the others' latest reports as inputs and records
them in `inputs_consumed`:

- `/review` reads the latest hotspot and `/review-all` reports for module
  selection, and the codebook for confusion-dense anchors.
- `/hotspot-survey` seeds its clusters with open findings from every
  instrument that share its areas, and its skeptics re-verify them.
- `/census` matches new sightings against finding keys as well as codebook
  titles, so a confusion at a known anchor attaches instead of minting.
- `/review-all` reads all of the above first and runs an instrument only
  when its `as_of_sha` is stale for the question being asked.

A finding another instrument has already recorded is re-observed, not
re-reported: the later run adds its `run_id` to `last_seen` and, if its
evidence changes the verdict or severity, says which field changed and why.

## 10. Judging against the quality definition

Every prompt that judges code in these instruments carries the quality
definition and tells its agent to tag each finding with the heuristics it
rests on (§1 `tags`). The definition reaches the prompt one of two ways,
never by hand-restating it: an interactive agent in a project that carries
`docs/code-quality.md` is told to `Read` it; a prompt assembled by code, or
one for a project without the file (reify today), embeds the block that
`orchestrator/src/orchestrator/agents/code_quality.py::guidance` renders
from the packaged normative copy. Either way the doc stays the one source.
Measurements the doc lists under "Do not steer by" may appear in a report
as context and never as a ranking, a target or a severity input.

## 11. Scheduling the next run

Cadence is driven by what has landed, not by the calendar, except where a
calendar window is needed to accumulate operational evidence (the census's
transcript window is the one such case today).

Each attended run (`/review`, `/hotspot-survey`, `/review-all`) ends by
filing a two-task trigger chain in the project's task store:

1. **Completion gate** — `task_kind='deterministic'`,
   `before_done.kind='predicate'`, running
   `scripts/check_run_completion.py --run <run_id> --threshold 0.7`
   (project-generic; reads the task store). It computes the weighted share
   of the run's filed tasks that have landed:

   ```
   weight: critical 4, high 3, medium 2, low 1, polish 1; a missing
   priority counts as medium, an unknown value is an error
   landed  = Σ weight(task) for tasks with status done
   pending = Σ weight(task) for tasks in any other status except cancelled
   share   = landed / (landed + pending)        # cancelled tasks count in neither
   ```

   Exit `0` when `share ≥ threshold` (the gate is `done`); exit `75` when
   not yet, which re-arms the gate `before_done.recheck_secs` later
   without escalating; any other exit is an error and escalates. The
   `75` verdict is the one orchestrator change this contract depends on.
2. **Human gate** — `task_kind='deterministic'`, `always_escalates=True`,
   no `before_done`, depending on the completion gate, titled
   `Run /<instrument> on <project_id> (<run_id> landed)`. When it
   dispatches it files an L2 escalation, which the escalation watcher
   routes to a human or spawns as a session. The instrument is never run
   unattended from this gate.

Both tasks carry `metadata.trigger_chain = {role: completion_gate |
human_gate, skill: "<instrument>" (bare name, no slash), run_id: "<run_id>",
superseded_by?: "<run_id>"}` and never `x_finding_run`, so the completion share does not
count the gates themselves. The key is typed and routed on by the runner,
the submit guard and the escalation builder, so it is a Tier-A key rather
than an `x_` annotation; until `plans/completion-driven-triggers-prd.md`
blesses it, writing it costs one `unknown_key` census line per chain task,
which is accepted. One chain per (project, instrument) may be open; a new
run supersedes the old chain before filing its own.

The `75` verdict's presence is probed in the dark-factory checkout (the
factory root, where the predicate script lives), never in the target
project. Until it lands, a deterministic task with neither `before_done`
nor `always_escalates` is rejected at submit, so the chain collapses to
the human gate alone, carrying `milestone: {mode: delayed, after_secs:
259200}` (three days of settling) and depending on the run's `critical` and
`high` tasks (all filed tasks if there are none); it is filed `deferred`
under `planning_mode`, gains dependencies as `/prd` sessions file them, and
is committed once the program's FILED table is complete. A run that files
no tasks files no chain and says so in its report. The exact calls for
both forms, the supersede order and the probe live in
`skills/_shared/filing-the-trigger-chain.md`; skills cite it.

The census is the exception: it stays automated, and its trigger
(`scripts/legibility/census_trigger.py`) fires on the same weighted
completion measure over the tasks its previous run filed, with the
calendar floor kept only as the minimum transcript window.

## 12. Where reasoning lives

Per the Comments stance of `docs/code-quality.md`: the report carries
measurements and claims; a defence of a decision made in deliberation
goes in the program doc or the task record; rationale that must outlive
the run goes in `docs/` with a pointer from the code. A report is never
the home of a rule.
