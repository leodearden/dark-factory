---
name: census
description: "Run an operator-initiated LEGIBILITY CENSUS: the first census for a newly-enabled project (never censused, so the trigger fails safe and never auto-fires), or an ad-hoc forced census on demand (e.g. after a big remediation wave). ALWAYS use this skill for: '/census <project-root>', 'run a census', 'run a legibility census', 'first census for <project>', 'force a census now'. Recurring census runs need NO skill — the nightly trickle's evaluate_census_step launches scripts/legibility/census.py automatically when census_trigger fires; this skill covers only the two cases where a human has to kick it off by hand. NOT in scope: editing scripts/legibility/* code, the nightly trickle, or the census trigger logic itself — this is an operator run-and-verify skill, not a dev skill."
argument-hint: "[project-root — absolute path to the project being censused; omit to use the current project]"
---

# /census — operator-initiated legibility census

A LEGIBILITY CENSUS is a saturation-mining sweep (`scripts/legibility/census.py`) that mines the codebase for confusion sightings until novelty saturates, verifies and synthesizes them into a dated report, updates the confusion codebook, and files remediation tasks. Normally this runs unattended: the nightly trickle evaluates `census_trigger` at the end of every run and launches `census.py` itself the moment a fire condition is met. **This skill exists only for the two cases a human has to start it by hand:**

1. **First census for a newly-enabled project.** `census_trigger.load_census_state` reads `docs/legibility/census-state.json` three ways — `"ok"`, `"malformed"`, or `"missing"`. A project that has never been censused has no file, so the read returns `"missing"` → `never_censused=True` → there is no `last_census_at` anchor to measure `days_since` against → every interval-based fire condition evaluates to "N/A", never "FIRE". **A first census can never auto-fire.** This is deliberate fail-safe design (see the module docstring in `scripts/legibility/census_trigger.py`), not a bug — but it means someone has to run it manually, once, per project.
2. **Ad-hoc forced census.** An operator wants a census right now regardless of the trigger's verdict — e.g. right after a large remediation wave, to capture fresh sightings and re-baseline the codebook before the next scheduled interval.

## Why one skill, not two

The recurring case needs no skill at all — it's fully automatic (nightly trickle → `evaluate_census_step` → `census.py`, no human in the loop). The two cases this skill *does* cover (first-run bootstrap, ad-hoc `--force`) both terminate in the exact same command and the exact same pre/post checklist; the only difference is *why* the operator is running it, which is context for the report header, not a different procedure. Splitting this into two skills would duplicate the checklist for no operational benefit.

## Preflight checklist

Before running, confirm there's something worth mining and headroom to mine it:

1. **Trickle data is present and fresh.** The census mines sightings the nightly trickle has already been accumulating in the confusion codebook; check the trickle has actually been running for this project:
   ```bash
   journalctl --user -u legibility-trickle@<project>.service --since "-3 days" | tail -50
   ```
   Look for recent successful runs and a logged census-trigger decision line each time. If the trickle unit isn't installed or hasn't run recently, the codebook may be stale or empty — a census can still run, but expect a smaller/duller sweep (more likely to saturate on near-zero novel sightings).
2. **Fresh codebook sightings exist.** Skim `docs/legibility/confusion-codebook.yaml` for recent `sightings`/`candidates` dates. A census mines fresh ground beyond what's already coded — if the codebook hasn't moved in weeks, mining will spend most of its budget rediscovering what's already known before saturating.
3. **Usage headroom.** The run itself preflights this (`preflight_headroom` — a cheap probe against the lightweight trickle model (haiku) that defers the whole run rather than burning budget mid-sweep on a rate-limited/degraded session), but it's worth a sanity glance yourself first: mining uses Sonnet (`census_miner`), synthesis uses Fable exclusively (`census_synthesis`) — make sure neither is already under heavy load from other concurrent work before kicking off a long saturation-mining sweep.

## Running the census

The canonical operator command, from the dark_factory main checkout:

```bash
cd /home/leo/src/dark-factory && uv run --project shared python scripts/legibility/census.py \
  --project-root <target-root> --force
```

- `<target-root>` is the absolute path to the project being censused (its own checkout, with its own `docs/legibility/` — not necessarily `dark_factory` itself).
- `--force` is what makes this an *operator*-initiated run: it bypasses the `census_trigger.decide_for_project` gate entirely (the same gate the nightly trickle checks before launching), so it works identically whether the trigger would have said NO-FIRE (first-census case — it always would, per above) or you simply don't want to wait for the next scheduled fire (ad-hoc case). Without `--force`, the CLI prints the NO-FIRE reasons and exits 0 without mining anything — expected behavior for a first census, not an error.
- Optional flags: `--config <path>` to point at a non-default `legibility.yaml`, `--date YYYY-MM-DD` to stamp the report with a date other than today (rare — mainly for backfilling a report for a run that started the previous day).

### Operator cost-control flags

Three composable flags bound what one run may spend. Each is optional and defaults to today's unbounded behavior, so omitting them all is exactly the command above.

- **`--max-batches N`** — stop mining after N batches. The report states the cap and that coverage is **partial** (sessions beyond the cap were never mined), and `stop_reason` becomes `capped` rather than `exhausted`.
- **`--max-verify-clusters N`** — hand the verifier at most N novel clusters, taken in mining order. Verification costs one Sonnet call per cluster, and that is the spend being bounded. The rest still merge into the codebook as `pending` candidates for a later census to adjudicate — **deferred, never dropped**.
- **`--dry-run-filing`** — write every would-be `submit_task` payload to `plans/confusion-census-<date>-payloads.json` for human review and file **nothing**. Everything else — codebook update, promotions, report, census-state advance — proceeds normally.

For an attended **first** census, run with all three:

```bash
cd /home/leo/src/dark-factory && uv run --project shared python scripts/legibility/census.py \
  --project-root <target-root> --force \
  --max-batches 50 --max-verify-clusters 150 --dry-run-filing
```

Why: a first census cannot rely on saturation to bound spend — against an empty codebook, a batch's `dup_rate` only measures "the miner found nothing to match", so mining runs to source exhaustion, per-cluster verification scales with however many novel clusters that produces, and filing would bulk-load a live task tree in one shot.

The run mines, verifies, synthesizes, updates the codebook, and files remediation tasks unattended — it can take a while (saturation mining runs until novelty drops below the configured duplicate-rate threshold for several consecutive batches). Watch the terminal for the final `census: done -- report=... filed_tasks=N stop_reason=...` line, or `census: deferred -- <reason>` if the headroom preflight declined to start.

## Post-run checklist

1. **Read the dated report.** `plans/confusion-census-<date>.md` in the *censused* project's own checkout — open it and read the origin × manifestation matrix (where confusions came from vs. how they showed up) plus the narrative sections. This is the actual deliverable; don't just trust the one-line CLI summary.
2. **Sanity-check per-stratum coverage counts.** The report should show mining coverage across the strata the sampler drew from — if one stratum has near-zero sightings while others are dense, that's worth a second look (could be a genuinely clean area, could be a sampling gap).
3. **Confirm `census-state.json` advanced.** Check `docs/legibility/census-state.json` in the censused project — `last_census_at` should now be this run's timestamp and `last_census_report` should point at the new report. This is what makes the *next* census automatic: with a real anchor in place, `census_trigger` can now compute `days_since` and the interval/tasks-landed/novelty-spike conditions become live instead of perpetually "N/A".
4. **Review the filed remediation tasks.** The run submits tasks through the normal curator path (`submit_fn`) — check the project's task tree for what landed and whether it needs triage/re-prioritization.
5. **If you ran `--dry-run-filing`: review the payload JSON and file by hand** (or re-run without the flag) before treating the census as complete. Nothing was filed, so the remediation half of the census is still outstanding — `plans/confusion-census-<date>-payloads.json` is the deliverable to work through.

## Anchor-seeding alternative — when a manual survey already happened

If a project already had an equivalent manual survey *before* this skill/pipeline existed for it — a hand-run agent-legibility survey, not `census.py` itself — you don't need to re-run mining from scratch just to unblock the trigger. Seed `docs/legibility/census-state.json` directly with the survey's own date/report/done-count, so the trigger treats that survey as the first census:

```json
{
  "last_census_at": "<survey-date>T00:00:00+00:00",
  "last_census_report": "<path to the survey report>",
  "last_census_done_count": <done-task count at survey time, from get_statuses>
}
```

Precedent: dark_factory itself was seeded this way — commit `0b99cf4ca2` set `last_census_at` from `plans/agent-legibility-survey-2026-07-13.md` (the 2026-07-13 survey, treated as the de-facto first census) with `last_census_done_count: 2321` (a 2026-07-14 done-count readback, intentionally slightly conservative so the tasks-landed condition doesn't over-fire on a stale baseline). Use this path only when a genuinely equivalent survey already exists — it's a substitute for *running* a first census, not a shortcut around wanting one.

## Out of scope

This skill only runs and verifies a census. It does not cover: modifying `scripts/legibility/*` (census mining/verification/synthesis logic, the codebook merger, etc.), the nightly trickle (`scripts/legibility/nightly.py`, the `legibility-trickle@.service`/`.timer` units), or the census trigger's fire-condition logic (`scripts/legibility/census_trigger.py`). Those are code changes for `/prd`/`/do`, not an operator run.
