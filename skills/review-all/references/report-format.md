# Output contract — report, findings JSON, program doc

Three artefacts. The first two are written by the run (Phase 5): the JSON record is the source and holds everything; the markdown is a human-facing digest rendered from it by the same digest script in the same run, capped near 40 KB, with the per-area finding lists in separate findings files, and none of it is edited by hand (contract §5). Per-finding prose — statement, measurements, next change, proposal, skeptic note — is read from the JSON or from the deliberation briefs, not from the digest; the findings files carry it in full only for `high` findings. All are named by `run_id` and committed together with `git commit --only`. The third is written by the deliberation that follows (Phase 6, `deliberation.md`). Field shapes are those of `docs/quality-findings-contract.md`; this file adds only the run-local fields and the section order.

Measured on the first dark-factory run: a markdown that rendered every one of 203 findings in full, 292 prior rows and 44 heuristic-14 measurements was 1.15 MB — unreadable as a report and too large for a session to load. The old 60 KB cap is why the one-line form exists. Measured on the three runs after it: 430 / 176 / 157 KB against a 60 KB cap — once the `high` findings are in full, one file cannot meet it, so the finding lists left the digest.

## Artefact 1 — `<plans_dir>/review-all-<project_id>-<YYYYMMDD>[-<n>].md` (rendered)

Required sections of the digest, in order, except §5 and §6: those two are rendered once per area into `<plans_dir>/review-all-<project_id>-<YYYYMMDD>[-<n>].findings/<area>.md` — one file per area plus `repo.md` and `cross-area.md`, each holding that area's `## High findings` then `## All findings` — by the same digest script, and `extra.findings_files` lists them. Measurements from the quality doc's `Do not steer by` list may appear in §2 as context and nowhere as a ranking or severity input (contract §10).

1. **`## Method`** — its first element is a fenced ```yaml block, loadable with one YAML load, carrying exactly the contract §5 keys and the instrument-specific ones under `extra:`:
   ```yaml
   run_id: review-all-<project_id>-<YYYYMMDD>
   as_of_sha: <sha>
   since: <sha> | none
   evidence: {source_files: <n>, source_lines_by_area: {<area>: <n>}, test_files: <n>, corpora: [...]}
   verification: {confirmed: <n>, weakened: <n>, refuted: <n>, unverified: <n>, refutation_rate: <refuted / (confirmed + weakened + refuted), 3 dp>}
   cost: {seats: {<model>/<effort>: <n>}, subagent_tokens_phase_2: <n>, subagent_tokens_phases_3_4: <n>, subagent_tokens_total: <n>, wall_clock_phases_1_4: <h:mm>, wall_clock_total: <h:mm>}
   inputs_consumed: [<run_id> | legacy:<path>, ...]
   extra:
     findings_json: <path>
     findings_files: [<plans_dir>/<run_id>.findings/<area>.md, ...]
     worktree: {path: <detached path>, removed: true}
     yellow_flag_sample: <null | {sampled: 10, strata: {high: <n>, random: <n>}, confirmed: n, weakened: n, refuted: n, lost: n, drift: <bool>, outcomes: [{key, severity, outcome: confirmed | weakened | refuted | lost, note}]}>
     refutation_rate_prior_runs: [{run_id, refutation_rate}]   # this instrument's previous runs, for the comparison SKILL.md Phases 3–4 asks for
     instruments: [{run_id, as_of_sha, commits_since, verdict: fresh | re-run | stale-not-run, why}]
     fable_seats: {wanted: [synthesis, critic], run: [...]}
     seats_lost: [...]
     severity_corrections: {applied: <n>, disputed: <n>}
     trigger_chain: <"filed by the program; ids in the program doc FILED table" | "no trigger chain: the run filed no tasks">
     launched_from: <review_due escalation id | none>
     metrics_snapshot: {path: plans/quality-metrics/<run_id>.json, diffed_against: <path | none>}
     key_implementation: <shared/src/shared/finding_key.py | skills/hotspot-survey/scripts/findings_artefact.py>
   ```
   A prior report that predates the contract (no `as_of_sha`) is stale by definition and is listed in `inputs_consumed` as `legacy:<path>`. Prose after the block may expand any of these; nothing in the prose contradicts the block. The prose closes with the **read coverage** table — one row per seat: files read fully and how many of them for the first time (not under `files_read_fully` or `partial` in the previous `/review-all` record; SKILL.md Phase 1 §Rotation), indexed, read partially (`read_coverage.partial`, with lines read verbatim), lines read against `planned_lines` — so a reader can see what was *not* read.
2. **`## Metrics snapshot`** — a report, never a gate, rendered from the committed `plans/quality-metrics/<run_id>.json` (its home; `extra.metrics_snapshot` names it): per area, the §What to measure instruments — cognitive complexity read as the pair (max per function, module total) for the top files, fan-in/fan-out, reach-back imports, deferred imports and how many close a cycle, cycles, hidden cycles and typing cycles, re-export shims, distinct test patch targets into privates, prose share, files over the soft ceiling and the alarm; the whole-repo totals; and a `delta vs since` column diffed against the one previous snapshot nearest `since`, named in the section (same instrument, same rule — say the rule). Line counts appear as the read-budget figure they are, with the doc's caveat cited, never as a ranking.
3. **`## Instrument freshness`** — table: instrument, latest run id, its `as_of_sha`, commits since (`git rev-list --count`), verdict (`fresh` / `stale → re-run as <run_id>` / `stale → not re-run, why`), open findings consumed.
4. **`## Area × heuristic view`** — the synthesis `matrix`: rows are areas, columns the primary tags that have findings; each cell lists keys best payoff first with severity and verdict glyphs. Standing findings are shown in a separate row style and never ranked. One ranking rule, stated once above the table.
5. **`## High findings`** *(findings file, not the digest)* — every `high` finding of the file's area in full, grouped **new / re-observed / standing / fixed**: key, display id, anchor, tags, severity (with `change_note` when the digest corrected it, or the `critic_disputes_severity` note), verdict (weakened marked inline `*(weakened: <skeptic note gist>)*`, and the `calibration_disputes_verdict` note beside it), class, statement, measurements with rules, `next_change`, `proposal` when one exists, the skeptic's note. A `fixed` entry names the confirming evidence.
6. **`## All findings`** *(findings file, not the digest)* — one line per finding that is not `high`, grouped the same way: key, anchor, severity, verdict glyph (with a mark for `calibration_disputes_verdict`), class, observation; `area_health` prose from the seat(s) opens the file. Everything else about these findings lives in the JSON.
7. **`## Heuristic-14 measurements`** — one line per `h14_measurements` entry: file, prose share, prose home ok, partition size, stays-whole verdict (the named partition and per-candidate symptoms are in the JSON). Present even when every file stays whole.
8. **`## Candidate streams (for deliberation)`** — the synthesis `streams`: name, member keys, remedy, deletable code, size, risk, mode, seam candidates, open choices with options / trade-offs / long-term consequence. Nothing here is a decision.
9. **`## Invariant candidates`** — statement, heuristic, checkable form, fixture sketch, or the existing INV it maps onto.
10. **`## Contradictions`** — between, conflict, resolution or open question. Empty section allowed, must be present.
11. **`## Exonerations`** — what the metrics flagged that reading showed healthy, and why.
12. **`## Critic`** — the critic's join errors by kind with the fix applied, whether the synthesis was revised, which severity corrections the digest applied and which it only flagged, and anything the lead overruled with the reason.
13. **`## Dropped and degraded`** — seats lost, findings dropped by validation (with the problem), lanes not run, instruments not re-run, fable seats wanted and not run, synthesis entries left unresolved by the id mapping. Never empty by omission: write `none` explicitly.
14. **`## Mechanical filings`** — one row per mechanical finding: key, source instrument(s) (this run, review, hotspot — candidates are merged by key before dedup), dedup step that decided it (contract §8 step 1a/1b/1c/2/filed), ticket or task id, disposition. Structural findings are listed as `open → deliberation`. This table is what a `/prd` seat is handed verbatim for its stream's keys (`deliberation.md` §Launching).
15. **`## Next`** — the deliberation hand-off: the program doc path expected, and that the contract §11 trigger chain is filed by the program's last act (its ids land in the program doc's FILED section).

Evidence hard rules: every finding has an anchor that resolved at `as_of_sha`; every measurement carries its rule; every proposal names what becomes deletable; weakened never disappears at the markdown layer.

## Artefact 2 — `<plans_dir>/review-all-<project_id>-<YYYYMMDD>[-<n>].json` (the record)

The Phase 5 digest of the workflow result, with keys minted; the markdown above is rendered from this file. `method` is byte-for-byte the yaml block's content.

```
{
  "method": { run_id, as_of_sha, since, evidence: {...}, verification: {confirmed, weakened, refuted, unverified, refutation_rate}, cost: {...}, inputs_consumed: [run_id | legacy:<path> ...], extra: {...} },
  "metrics": { per_area: {<area>: {...}}, totals: {...}, delta_vs_since: {...} | null },
  "findings": [ <contract §1 fields: key, area (plain briefing key), anchor, tags, severity, evidence_source, statement, proposal?, verdict, disposition, first_seen, last_seen (list), supersedes (list, usually empty), sub_area?>
                + run-local: canonical, display_id, seat, class, next_change, measurements [{name, value, rule}], verdict_notes, falsification [{measurement, ran, would_refute, found, reproduced}], observation: new|re-observed|standing|fixed, also_seen_by?, change_note?, critic_disputes_severity?, calibration_disputes_verdict?: {to: weakened | refuted, note} ],
  "refuted": [ {key, canonical, seat, display_id, anchor, why} ],
  "prior_verdicts": [ {canonical, key, sub_area?, status, checked, gone_verdict?} ],
  "unchecked": [ {canonical, key, seat, sub_area?, why} ],
  "h14_measurements": [ ... as emitted ... ],
  "read_coverage": [ {seat, files_read_fully, read_fully_first_time, files_indexed, partial: [{path, lines_read_verbatim, how}], lines_read, planned_lines} ],
  "short_ids": [ {id, canonical} ],
  "synthesis": { matrix, streams, invariant_candidates, contradictions, exonerations, method, unresolved_ids, join_errors_mechanical },
  "critic": { join_errors, verdict, notes, synthesis_revised },
  "severity_corrections": [ {canonical, key, from, to, reason, accepted, how, applied} ],
  "calibration": { refutation_rate, downgraded_unreproduced, yellow_flag_sample, high_stratum_reverified: bool, high_stratum?: {reverified, confirmed, weakened, refuted, lost, outcomes} },
  "dropped": [ ... ],
  "filings": [ {key, sources, step, ticket_or_task, disposition} ]
}
```

`first_seen` is the run that first recorded the key, read from the earliest artefact of any instrument carrying it (hotspot, review, census, review-all); `last_seen` is the prior artefact's list with this `run_id` appended for a re-observed key, `[]` for a new one; `supersedes` lists the keys this finding replaces (the digest fills it from the script's `supersedes_canonical`); `sub_area` is carried through, never hashed. Keys are minted by the one implementation `extra.key_implementation` names. `disposition` at commit time is `open`, `filed:<id>`, `accepted:<id>`, `refuted` or `fixed:<as_of_sha>`; the program's `/prd` sessions change `open` to `filed:<id>` in the task store, not in this file (contract §7: the store is the home; the next run re-derives).

**Short ids.** Inside the workflow the synthesis and critic name findings by `f<n>` / `p<n>`; the script maps the matrix, stream and contradiction entries back to canonicals before returning and lists what it could not map under `synthesis.unresolved_ids`. The critic's `where`, the synthesis `method` and `exonerations` prose keep the ids; `short_ids` is the table the digest uses to render them as keys. The digest never guesses at an abbreviation: an unresolved entry is reported under Dropped and degraded, not expanded by hand.

**Severity corrections.** For every `severity_corrections` entry the script returned (a critic `severity-unsupported` item with `severity_to`), the digest does one of two things and records which in `applied`:
- `accepted: true` (the revised synthesis answered the item as accepted) → the finding's `severity` becomes `to`, `change_note` records `severity <from> → <to>: <critic reason> (critic correction accepted in the revised synthesis)`, and any filing made in Phase 5 uses the corrected severity as its priority. The record then agrees with the matrix the deliberation reads.
- otherwise (`accepted` false or null — the item was declined, the revision was lost, or the critic's `where` resolved to no finding) → the seat's severity stands and the finding carries `critic_disputes_severity: {to, reason}`, which the markdown and the deliberation brief show beside the severity so the human rules on it rather than discovering it at decompose.

Measured on the first dark-factory run: the critic's correction on `fk-3505aba0481d` (high → medium) was accepted in the revision, the record kept `high`, and the lead had to rule at decompose.

**Calibration disputes.** A sampled finding that a calibration skeptic weakens or refutes (SKILL.md §Phases 3–4) keeps its seat skeptic's verdict and carries `calibration_disputes_verdict: {to: weakened | refuted, note}`, which the findings file shows beside the verdict, as it shows `critic_disputes_severity` beside the severity; the human rules. `calibration.high_stratum_reverified` is true when the sample's drift signal re-verified the whole high stratum, and `calibration.high_stratum` carries that step's tally and per-finding outcomes. `synthesis.join_errors_mechanical` lists the script-run set-membership checks (a key in two streams, a key twice in one matrix cell, a high finding in no stream) after the synthesis and again after a revision; the rendered `## Critic` section shows them beside the critic's own join errors. Measured: reify 5 of 10 sampled highs weakened, solar_challenge_platform 2 of 10, and no record carried it.

## Artefact 3 — `<plans_dir>/review-all-program-<project_id>-<YYYYMMDD>.md` (written by deliberation, not by the run)

The coordination contract every `/prd` session of the program reads first. Section shape follows `skills/hotspot-survey/references/report-format.md` §Artifact 3 with these deltas:

- **`## Streams`** gains a `Finding keys` column; every stream lists the keys it acts on, and no key appears in two streams.
- **`## Seam ownership (G4)`** unchanged in shape; seeded from the synthesis `seam_candidates`.
- **No invariant registry.** Candidate invariants that survive deliberation are promoted through the four-site lockstep edit `CONTRIBUTING.md` §6 describes and live in `docs/legibility/design-invariants.md`; the program doc carries a pointer per promoted INV id and the owning stream, never a second list (INV-9).
- **`## Resolved decisions (do not relitigate)`** — the deliberation rulings, each with the options considered and the long-term consequence that decided it, and `ruled_by` (`human`, or `lead (--yes)` when the recommendation was accepted under `/review-all --yes`; `deliberation.md` §Recording).
- **`## Shared conventions`** adds the filing stamps: `metadata.x_finding_key` (list), `metadata.x_finding_run = <run_id>`, `metadata.source = "review-all"`, severity → priority, `planning_mode=True` → `add_dependency` → one `commit_planning`; `metadata.files` file-level; an acceptance is a `cancelled` task with `metadata.x_acceptance_reason` (never `deferred`); the **freshness clause** — anchors were verified at `as_of_sha`; before building a task on one, `git diff <as_of_sha>..main -- <path>` and re-read the symbol. It also carries the two program-doc conventions the first run established:
  - **Held leaf.** A leaf that depends on a later-wave stream's contract is filed in planning mode with `metadata.x_seam_wait = {stream: "<owning stream>", contract: "<the seam or leaf it waits for, as specified>", release_owner: "<who releases it>"}`, wired into its own batch's dependencies, and **excluded from that batch's `commit_planning`**. `release_owner` defaults to the waited-on stream, and is the lead when that stream's session is already running at hold time. The server stamps the leaf's `planning_mode` birth with a `planning` record, which `shared/src/shared/task_deferral.py::needs_human` reports `stale_planning` after 24 h, so the leaf would read as an abandoned planning batch; the session that holds it therefore re-stamps it before that session ends: `set_task_status(id=<leaf>, status="deferred", deferral={"kind": "until_condition", "condition": "<x_seam_wait.stream> lands <x_seam_wait.contract>"}, project_root="<project_root>")`, with no `expires_at` (once the deferral sweep lands, a lapsed one releases the row to `pending` before its seam dependency exists), and confirms the response echoes `deferral` (no echo means the server predates the deferral gate: the hold is unrecorded, so re-stamp it after fused-memory restarts). The leaf stays `deferred` until its release owner releases it in the writes `deliberation.md` §Launching step 3 lists; release is `set_task_status(status="pending")`, because `commit_planning` refuses the re-stamped hold (`not_planning_hold`). Recipe: `docs/task-authoring.md` §9 "Deferring a task". On release it gains `metadata.x_seam_released = {by, at, onto: [<ids>], contract_as_delivered}`. Both keys are `x_`-namespace (`docs/task-authoring.md` §Tier-C), no blessing. Measured on reify: #8741 and #8851 needed their contract reworded on release. The FILED row names every held leaf with its `x_seam_wait`.
  - **One editor per document.** A `/prd` seat never edits the program doc or another stream's PRD. It returns its FILED row and any cross-reference lines (a seam note for another stream, a cancellation of an older task) as a structured block, and the lead applies them.
- **`## Triage briefing`** optional; include when a stream changes what looks stuck or what counts as a regression.
- **`## FILED — program status`** appended by the lead from the rows seats return (`deliberation.md` §Launching): PRD path, task anchors, held leaves, coordinator interventions, the gate id(s) of the trigger chain filed per `skills/_shared/filing-the-trigger-chain.md` (`metadata.trigger_chain.skill == "review-all"`, this `run_id`), the superseded chain's run id and the `review_due` escalation resolved in its §5 step 2 if any, and `none — no tasks filed` when the program filed nothing.
