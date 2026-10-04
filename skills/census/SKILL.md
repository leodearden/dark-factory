---
name: census
description: "Run an operator-initiated LEGIBILITY CENSUS: an attended first census for a newly-enabled project (run with cost caps and review-before-filing, before the nightly trickle auto-fires an unattended one), or an ad-hoc forced census on demand (e.g. after a big remediation wave). ALWAYS use this skill for: '/census <project-root>', 'run a census', 'run a legibility census', 'first census for <project>', 'force a census now'. Recurring census runs need NO skill — the nightly trickle's evaluate_census_step launches scripts/legibility/census.py automatically when census_trigger fires; this skill covers only the cases where a human kicks it off by hand. NOT in scope: editing scripts/legibility/* code, the nightly trickle, or the census trigger logic itself — this is an operator run-and-verify skill, not a dev skill."
argument-hint: "[project-root — absolute path to the project being censused; omit to use the current project]"
---

# /census — operator-initiated legibility census

A legibility census (`scripts/legibility/census.py`) does the following:

1. Mines agent **session transcripts** for confusion sightings, in stratified-random batches, until novelty saturates. The transcripts come from `~/.claude/projects` plus the project's `agent_transcript_roots`, matched by `cwd_prefixes` in `docs/legibility/legibility.yaml`.
2. Verifies each novel cluster against the project's current tree. The verifier is the only stage that reads the code.
3. Synthesizes the verified clusters into a dated report.
4. Updates `docs/legibility/confusion-codebook.yaml`.
5. Files remediation tickets.

What a finding must carry (key, area, anchor, tags, severity, disposition), what a report must pin, and how a run schedules the next are all defined in `docs/quality-findings-contract.md`. This skill points there and does not restate it.

Statements marked *(until Ln)* describe today's code and change when that leaf of `plans/census-incremental-prd.md` lands. The final section lists what changes.

Normally the census runs unattended. Each nightly trickle run ends by evaluating `census_trigger.decide_for_project` and, on FIRE, launches `census.py` with the caps from `legibility.yaml` `census.trickle_caps`. Run it by hand only in these two cases:

1. **Attended first census.** A first census can auto-fire. When `docs/legibility/census-state.json` is missing, `decide_for_project` anchors `last_census_at` on the earliest codebook date and exempts the floor. Condition (c), the novelty spike, needs no anchor. It counts candidates whose `first_seen` falls in the last 72 h against a threshold of 4, and a running trickle exceeds that on almost every night *(until L8a)*. So a project's first census fires unattended within a night or two of its trickle producing candidates. That run is capped by `trickle_caps` but files tickets for real. Run the first census by hand, before installing the trickle timer, when you want it bounded and reviewed (`--dry-run-filing`). A project with no trickle data and no codebook does not fire.
2. **Ad-hoc forced census.** Run one when you want fresh sightings now, regardless of the trigger's verdict, for example right after a remediation wave.

Both cases run the same command and the same checklists. Only the reason differs, so this is one skill.

## Preflight

1. **Is the trickle healthy?** For a project with a running trickle:
   ```bash
   uv run --directory <factory checkout>/shared python <factory checkout>/scripts/legibility/check_trickle_progress.py <project_id> 3
   journalctl --user -u legibility-trickle@<project_id>.service --since "-3 days" | grep -E 'census trigger:|digest|failed' | tail -20
   ```
   Exit 0 means signal is flowing. Any other exit names its own cause. A failed trickle does not block a census, because the census mines transcripts directly, but a forced census on top of a broken trickle is worth knowing about.
2. **What would the trigger say?** Do not skim the codebook: it is over 2 MB.
   ```bash
   PYTHONPATH=<factory checkout>/scripts uv run --directory <factory checkout>/shared python <factory checkout>/scripts/legibility/census_trigger.py evaluate --project-root <target-root>
   ```
   It prints FIRE or NO-FIRE and one reason line per condition: days since the last census, tasks landed, the 72 h candidate count, and the floor. It makes one read-only `get_statuses` call.
3. **Headroom.** The run probes this itself (`preflight_headroom`, one trickle-model call, which defers the whole run when the account pool is capped), and probes again before verification. Mining uses `census_miner` (Sonnet), verification `census_verify` (Sonnet) and synthesis `census_synthesis` (Fable). Avoid starting an uncapped census while other heavy work holds the pool.

## Running

From the factory checkout. `<factory checkout>` is the dark-factory checkout that ships `scripts/legibility/`. `uv run --directory` changes the working directory to `shared/`, so every path below is absolute. Use `--directory`, not `--project`, following the `uv run --directory <member>` convention in `review/briefing.yaml`.

```bash
uv run --directory <factory checkout>/shared python <factory checkout>/scripts/legibility/census.py \
  --project-root <target-root> --force
```

- `<target-root>` is the absolute root of the censused project. It must agree with `project_root` in that project's `legibility.yaml`, or the run refuses.
- `--force` bypasses the trigger. Without it, the CLI prints the NO-FIRE reasons and exits 0.
- `--config <path>` names a non-default `legibility.yaml`. `--date YYYY-MM-DD` stamps the report with another date.
- `--harness-config <path>` names the `legibility.yaml` of the project that owns harness fix surfaces. It defaults to this checkout's own, which is dark_factory.

### Cost-control flags (each optional; omitted means unbounded)

- **`--max-batches N`** stops mining after N batches (N ≥ 1). The report marks coverage as partial and `stop_reason=capped`. *(until L1)* The run still advances `last_census_at`, and the next window starts there, so capped-away sessions are never mined. To sweep them, roll `last_census_at` back first.
- **`--max-verify-clusters N`** hands the verifier at most N novel clusters, in mining order. The rest merge as `pending` candidates. *(until L7)* A pending candidate is re-adjudicated only if a later run reproduces its exact title, so in practice it stays pending until someone adjudicates it by hand.
- **`--dry-run-filing`** writes every would-be `submit_task` payload to `plans/confusion-census-<date>-payloads.json` (or `…-payloads-2.json`, never overwritten) and files nothing. The codebook and census-state still advance, so filing that JSON by hand is the only way those tasks will ever land.

For an attended first census, use all three:

```bash
uv run --directory <factory checkout>/shared python <factory checkout>/scripts/legibility/census.py \
  --project-root <target-root> --force \
  --max-batches 50 --max-verify-clusters 150 --dry-run-filing
```

Against an empty codebook, `dup_rate` only measures "the miner found nothing to match", so saturation cannot bound the run. A bounded first census is a sample, not a sweep. The trickle-launched census passes these same two caps from `census.trickle_caps` (schema default 50/150; `null` on a key means uncapped). Flags typed on a manual run are unaffected by that block.

### What the run does with verified findings

- **Filing is through tickets.** `submit_task` returns a `tkt_*` id. The curator decides create, combine or drop asynchronously, and `resolve_ticket(ticket, project_root)` returns the task id once it has. Each ticket carries `metadata.source=legibility_census` and `origin_project_id`.
- **Harness routing.** When a cluster's descriptive fields name a harness component, the ticket files into the harness project's tree, with the matched markers under `metadata.x_fix_surface`. The rule is in `scripts/legibility/filing_policy.py::resolve_target`. When the verifier proposed an in-tree remediation, the ticket stays in the observed project. The report lists each cross-project ticket with the `project_root` that `resolve_ticket` needs.
- **Singleton filing gate** (`filing_policy.is_fileable`, since 2026-09-27). A verified cluster files only when the verifier proposed an existing in-tree remediation path, or when its title has at least `MIN_UNREMEDIATED_SIGHTINGS` (2) distinct-session sightings. Anything else is promoted into the codebook, marked `filing: withheld`, and listed under **Recorded, Not Filed**. A withheld entry files automatically on a later run once its sightings reach 2.

Watch for the final line:

- `census: done -- report=... filed_tickets=N stop_reason=...`, with `unresolved_verdicts=N` appended only when the run paid for verdicts that a standing adjudication of the same title overrode.
- `census: deferred -- stage=... unverified_clusters=...` when a headroom gate deferred the run. Nothing was persisted, so re-run once capacity returns.

## Post-run checklist

1. **Read the report** at `plans/confusion-census-<date>.md` in the censused project. *(until L1)* A second run on the same date overwrites it, and the earlier one survives only in git history (each run commits its report). Its sections:
   - **Saturation:** batch count, stop reason and per-batch `dup_rate`. A stop at the two-batch minimum is the common case, and it is not evidence of full coverage.
   - **Verification:** present on every run given `--max-verify-clusters`, and therefore on every trickle-launched run. The anomaly to look for is the bullet **"ALL N … were REJECTED and none survived"**. It signals a suspected systemic verifier failure. Check the per-cluster `verify failed` warnings in the journal before accepting the run.
   - **Unresolved Verdicts:** verdicts paid for and dropped against a standing record.
   - **Origin x Manifestation Matrix:** from the verifier's phase stamps.
   - **Synthesis:** prose. *(until L10)* It is not parsed. Its "inputs to the merger" dispositions are not applied, so act on any you agree with by hand.
   - **Filed Tasks:** ticket ids, plus the cross-project list.
   - **Recorded, Not Filed** and **Cost**.
   The report has no per-stratum coverage section.
2. **Resolve the tickets.** Call `resolve_ticket` for each id; for cross-project tickets, use the project root the report names. *(until L4)* The report cannot name task ids, because the curator decides after the report is rendered. Triage or re-prioritise what landed, since every ticket is filed at priority `medium` *(until L3)*.
3. **Check `docs/legibility/census-state.json` advanced:**
   - `last_census_at` is this run's **date**, `YYYY-MM-DD` (date-only, not a timestamp).
   - `last_census_report` names the new report (written as an absolute path *(until L1)*).
   - `last_census_done_count` *(until L8b)* is three-valued:
     - an integer matching the project's done-task count: correct.
     - `0` on a project that has done tasks: wrong. The `get_statuses` call failed and a fabricated baseline was persisted (task 3291). Repair it with a dated `get_statuses` readback and investigate the endpoint.
     - `0` on a project with no done tasks: legitimate.
     - `null`: the count was unobservable. The run stands, and the tasks-landed condition fails safe until the next successful census.
4. **If you ran `--dry-run-filing`,** file the payload JSON by hand. A later census will not re-file it, because those confusions now code as matches and the window has moved.

## Anchor seeding — when an equivalent manual survey already exists

To have the trigger treat an earlier hand-run legibility survey as the first census, seed the state file directly:

```json
{
  "last_census_at": "<survey-date>",
  "last_census_report": "<repo-relative path to the survey report>",
  "last_census_done_count": <done-task count at survey time, from get_statuses>
}
```

`last_census_at` may be a date or an ISO datetime (`datetime.fromisoformat`). `last_census_done_count` *(until L8b)* must be an **unquoted non-negative JSON integer**, or `null` if no readback is possible. `census_trigger.compute_tasks_landed` validates rather than coerces it, so a bad value fails that condition safe and logs a warning naming the value. Every threshold in `legibility.yaml`'s `census:` block must likewise be an unquoted non-negative integer. `CensusConfig.from_mapping` falls back per field and names each rejected value in one warning.

Precedent: dark_factory was seeded this way in commit `0b99cf4ca2`, from `plans/agent-legibility-survey-2026-07-13.md`, with `last_census_done_count: 2321`. Use this only when an equivalent survey really exists.

## What changes when `plans/census-incremental-prd.md` lands

- **L1:** The run writes a JSON record and renders the report from it under one basename (contract §5), with a `run_id` and a `## Method` YAML block (the seven §5 keys plus `extra`). A same-day rerun writes `…-2.json`/`…-2.md` instead of overwriting.
  - The trickle and the census record every coded session in a host-local ledger. The census skips ledgered and zero-signal sessions, and a capped run's unmined sessions are picked up by the next run, so rolling `last_census_at` back is no longer needed.
  - `census-state.json` gains `last_census_run_id`, `last_census_as_of_sha` and `session_watermark`, and `last_census_report` becomes repo-relative. Check the ledger with `session_ledger.py stats --project-id <id>`.
- **L8a:** The novelty spike becomes relative to its trailing baseline, and the floor is measured from the session watermark, so the trigger stops firing every night.
- **L8b:** The tasks-landed condition and `last_census_done_count` are retired. The trigger fires on the weighted completion of the tasks the previous run filed (contract §11), or on a novelty spike. The anchor-seeding JSON then needs only `last_census_at` and `last_census_report`.
  - **Open for Leo:** whether the 10-day `max_interval_days` backstop survives. The contract allows the calendar only as the minimum transcript window. If the backstop is kept, a first census still auto-fires once the earliest codebook date is `max_interval_days` old. If it is dropped, a first census fires only on a novelty spike or by hand.
- **L2, L3:**
  - Verdicts carry anchor, tags, severity with a reason, and route.
  - Entries carry `finding_key`/`finding_area`/`finding_anchor`/`finding_tags`, and the report gains `## Findings` in the contract §1 shape.
  - Tickets carry `x_finding_key`/`x_finding_run`, at a priority equal to severity.
- **L4:** Filing follows the contract §8 dedup protocol, and the report names task ids and the step that decided each finding.
  - Structural findings are listed, not filed.
  - Entries carry `filed_tickets` back-links. A run that cannot reach the task store files nothing and says so.
- **L5:** `## Dispositions` carries earlier tickets forward to their task outcomes. A done task triggers re-verification (`fixed` or re-filed), and a cancelled task with a reason makes its entry `accepted`.
- **L6, L7:**
  - `## Screened` counts clusters attached to existing records with no verify spend.
  - `## Adjudication` drains the pending-candidate backlog through a capped, similarity-clustered queue and lists every attach for audit.
  - `--max-verify-clusters` deferrals then reach adjudication without recurring.
- **L9:** The coder is given the project's invariant slugs, and unknown slugs are dropped and counted.
- **L10:** Synthesis is rendered from validated JSON as new versus re-observed findings, and its phase refinements and corrections are applied rather than left in prose.
- **G:** A deterministic gate checks the first automatic report after the batch lands for all of the above.

## Out of scope

This skill runs and verifies a census. It does not change `scripts/legibility/*`, the trickle units, or the trigger logic. Those changes go through `/prd` (see `plans/census-incremental-prd.md`) or `/do`.
