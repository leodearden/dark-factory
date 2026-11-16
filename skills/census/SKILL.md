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

Design record: `plans/census-incremental-prd.md`.

Normally the census runs unattended. Each nightly trickle run ends by evaluating `census_trigger.decide_for_project` and, on FIRE, launches `census.py` with the caps from `legibility.yaml` `census.trickle_caps`. The trigger fires on either of two conditions, and on neither inside the floor:

- **(a) Completion:** the weighted completion of the tasks the previous run filed under `x_finding_run` (contract §11), read with one `scripts/check_run_completion.py` check against `census.completion_threshold_pct` (default 70). With no previous run id, or a previous run that filed nothing, completion is N/A.
- **(b) Novelty:** a novelty spike relative to its trailing baseline.
- **Floor:** the minimum transcript window, measured from the session watermark. It never fires a census by itself, and elapsed time alone fires nothing. The tasks-landed condition and the 10-day `max_interval_days` backstop are retired, the backstop by decision (Leo, 2026-10-05). A `legibility.yaml` still carrying a retired key loads, and the key is ignored with one deprecation warning.

When the completion check fails on 3 consecutive nightly evaluations (any exit but 0, 75, or 2 for a run that filed nothing), the trickle files one `escalate_info` for the project. If that post does not land, it is retried on each later night until one does. Until the check answers again, the census can fire on the novelty spike alone. The next exit 0 or 75 resets the streak; a night with nothing to check carries it.

Run it by hand only in these two cases:

1. **Attended first census.** A first census can auto-fire. A never-censused project has no previous run id, so completion is N/A. Its floor is measured from the earliest codebook date, and once that floor has passed it fires on a novelty spike. That run is capped by `trickle_caps` but files tickets for real. Run the first census by hand, before installing the trickle timer, when you want it bounded and reviewed (`--dry-run-filing`). A project with no trickle data and no codebook never fires, so running it by hand is the only way it gets a census.
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
   `--config <path>` names a non-default `legibility.yaml`. As with the census, a config whose `project_root` is not `<target-root>` is refused with exit 1. It prints a `config:` line naming the `legibility.yaml` it loaded and the validated thresholds, then FIRE or NO-FIRE and one reason line each for completion (the share of the previous run's filed tasks landed, against 0.70, or a named N/A), the novelty spike and the floor. It runs one read-only `scripts/check_run_completion.py` check and makes no `get_statuses` call.
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

- **`--max-batches N`** stops mining after N batches (N ≥ 1). The report marks coverage as partial and `stop_reason=capped`. Sessions the cap left unmined are not in the session ledger, so the next run picks them up; `last_census_at` needs no rollback.
- **`--max-verify-clusters N`** hands the verifier at most N novel clusters, in mining order. The rest merge as `pending` candidates, which the report's `## Adjudication` queue works through on later runs, a capped number of clusters per run.
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

1. **Read the report** at `plans/confusion-census-<date>.md` in the censused project, alongside its JSON record. Each run has its own identity (`scripts/legibility/census_identity.py`): a same-day rerun writes `…-2.md`/`…-2.json` and never overwrites the first. The newest committed report (`git ls-files 'plans/confusion-census-*.md' | tail -1`) shows the current section list. Its sections:
   - **Method:** the YAML block of the run's parameters, with its `run_id`.
   - **Saturation:** batch count, stop reason and per-batch `dup_rate`. A stop at the two-batch minimum is the common case, and it is not evidence of full coverage.
   - **Screened:** clusters attached to existing records with no verify spend.
   - **Adjudication:** the capped, similarity-clustered queue that works through pending candidates, with every attach listed for audit.
   - **Verification:** present on every run given `--max-verify-clusters`, and therefore on every trickle-launched run. The anomaly to look for is the bullet **"ALL N … were REJECTED and none survived"**. It signals a suspected systemic verifier failure. Check the per-cluster `verify failed` warnings in the journal before accepting the run.
   - **Unresolved Verdicts** (only when present): verdicts paid for and dropped against a standing record.
   - **Origin x Manifestation Matrix:** from the verifier's phase stamps.
   - **Synthesis:** rendered from validated JSON as new versus re-observed findings; its phase refinements and corrections are applied to the codebook and reported (`scripts/legibility/synthesis.py`). To act on a proposal that was not applied and that you agree with, put an entry correction or sighting in a §7.3 coding record and run `scripts/legibility/codebook.py apply`, checking that its printed `applied:` stats show it matched or applied (a record reusing the sighting's own session is a silent no-op); promote or reject a candidate through census adjudication; never hand-edit `docs/legibility/confusion-codebook.yaml`.
   - **Findings:** each finding in the contract §1 shape.
   - **Filed Tasks:** one row per finding with its `task_id`, ticket and disposition, plus the cross-project list.
   - **Recorded, Not Filed** (only when present): withheld entries.
   - **Structural:** structural findings, listed and not filed.
   - **Dispositions:** earlier tickets carried forward to their task outcomes. A done task triggers re-verification (`fixed` or re-filed), and a cancelled task with a reason makes its entry `accepted`.
   - **Cost.**
   The report has no per-stratum coverage section.
2. **Resolve the tickets.** The `## Filed Tasks` table names the `task_id` of every finding the curator had already decided. Call `resolve_ticket` for each row that has only a ticket (its disposition reads `open … pending`; it resolves on the next run); for cross-project tickets, use the project root the report names. Triage or re-prioritise what landed: a ticket's priority is set from its cluster's severity (`scripts/legibility/census_filing.py`), so check that the severity fits.
3. **Check `docs/legibility/census-state.json` advanced:**
   - `last_census_at` is this run's **date**, `YYYY-MM-DD` (date-only, not a timestamp).
   - `last_census_report` names the new report, repo-relative (e.g. `plans/confusion-census-2026-11-07.md`).
   - `last_census_run_id` names this run. It is what the next trigger's completion arm reads.
   - `last_census_as_of_sha` is the commit the run censused, and `session_watermark` is where the next run's transcript window starts (the floor is measured from it).
4. **If you ran `--dry-run-filing`,** file the payload JSON by hand. A later census will not re-file it, because those confusions now code as matches and the window has moved.

## Anchor seeding — when an equivalent manual survey already exists

To have the trigger treat an earlier hand-run legibility survey as the first census, seed the state file directly:

```json
{
  "last_census_at": "<survey-date>",
  "last_census_report": "<repo-relative path to the survey report>"
}
```

`last_census_at` may be a date or an ISO datetime, which is read as its UTC date. Add `"last_census_run_id": "<run id>"` only when the survey filed its tasks under `x_finding_run=<run id>`; without it, completion is N/A and the census fires on a novelty spike alone.

Every threshold in `legibility.yaml`'s `census:` block must be an **unquoted integer**: `completion_threshold_pct` in 1..100, the others non-negative (the `saturation` and `adjudication` similarities are unquoted numbers in 0..1). A malformed `census:` value fails `config.load_config` naming the field and value: the trickle records a failed night, its error naming the field and value, and `census_trigger.py evaluate` exits 1. A retired key (`max_interval_days`, `tasks_landed_threshold`, `tasks_landed_min_days`) is ignored, with one warning naming it and the file it is in.

Precedent: dark_factory was seeded this way in commit `0b99cf4ca2`, from `plans/agent-legibility-survey-2026-07-13.md`. Use this only when an equivalent survey really exists.

## Session ledger and the gate

The trickle and the census record every coded session in a host-local session ledger. The census skips ledgered and zero-signal sessions. Check the ledger with `session_ledger.py stats --project-id <id>`.

Gate leaf G (`scripts/legibility/check_census_report.py`, task 6409) checks the first automatic report after the batch. When it reports no conforming report yet, the remedy is a forced census with this skill.

## Out of scope

This skill runs and verifies a census. It does not change `scripts/legibility/*`, the trickle units, or the trigger logic. Those changes go through `/prd` (see `plans/census-incremental-prd.md`) or `/do`.
