# Output contract — report, findings JSON, program doc

Three artefacts. The first two are written by the run (Phase 5): the JSON record is the source, the markdown is rendered from it by the same digest script in the same run and never edited by hand (contract §5); both are named by `run_id` and committed together with `git commit --only`. The third is written by the deliberation that follows (Phase 6, `deliberation.md`). Field shapes are those of `docs/quality-findings-contract.md`; this file adds only the run-local fields and the section order.

## Artefact 1 — `<plans_dir>/review-all-<project_id>-<YYYYMMDD>[-<n>].md` (rendered)

Required sections, in order. Measurements from the quality doc's `Do not steer by` list may appear in §2 as context and nowhere as a ranking or severity input (contract §10).

1. **`## Method`** — its first element is a fenced ```yaml block, loadable with one YAML load, carrying exactly the contract §5 keys and the instrument-specific ones under `extra:`:
   ```yaml
   run_id: review-all-<project_id>-<YYYYMMDD>
   as_of_sha: <sha>
   since: <sha> | none
   evidence: {source_files: <n>, source_lines_by_area: {<area>: <n>}, test_files: <n>, corpora: [...]}
   verification: {confirmed: <n>, weakened: <n>, refuted: <n>, unverified: <n>}
   cost: {seats: {<model>/<effort>: <n>}, subagent_tokens: <n>, wall_clock_phases_1_4: <h:mm>, wall_clock_total: <h:mm>}
   inputs_consumed: [<run_id>, ...]
   extra:
     findings_json: <path>
     worktree: {path: <detached path>, removed: true}
     yellow_flag_sample: <null | {sampled: 5, confirmed: n, weakened: n, refuted: n}>
     instruments: [{run_id, as_of_sha, commits_since, verdict: fresh | re-run | stale-not-run, why}]
     fable_seats: {wanted: [synthesis, critic], run: [...]}
     seats_lost: [...]
     trigger_chain: <"filed by the program; ids in the program doc FILED table" | "no trigger chain: the run filed no tasks">
     launched_from: <review_due escalation id | none>
     metrics_snapshot: {path: plans/quality-metrics/<run_id>.json, diffed_against: <path | none>}
     key_implementation: <shared/src/shared/finding_key.py | skills/hotspot-survey/scripts/findings_artefact.py>
   ```
   Prose after the block may expand any of these; nothing in the prose contradicts the block.
2. **`## Metrics snapshot`** — a report, never a gate, rendered from the committed `plans/quality-metrics/<run_id>.json` (its home; `extra.metrics_snapshot` names it): per area, the §What to measure instruments — cognitive complexity read as the pair (max per function, module total) for the top files, fan-in/fan-out, reach-back imports, deferred imports and how many close a cycle, cycles, hidden cycles and typing cycles, re-export shims, distinct test patch targets into privates, prose share, files over the soft ceiling and the alarm; the whole-repo totals; and a `delta vs since` column diffed against the one previous snapshot nearest `since`, named in the section (same instrument, same rule — say the rule). Line counts appear as the read-budget figure they are, with the doc's caveat cited, never as a ranking.
3. **`## Instrument freshness`** — table: instrument, latest run id, its `as_of_sha`, commits since (`git rev-list --count`), verdict (`fresh` / `stale → re-run as <run_id>` / `stale → not re-run, why`), open findings consumed.
4. **`## Area × heuristic view`** — the synthesis `matrix`: rows are areas, columns the primary tags that have findings; each cell lists keys best payoff first with severity and verdict glyphs. Standing findings are shown in a separate row style and never ranked. One ranking rule, stated once above the table.
5. **`## Findings by area`** — one `### <area>` per area (then `### repo`). Within each: `area_health` prose from the seat(s); then findings grouped **new / re-observed / standing / fixed**, each with key, display id, anchor, tags, severity, verdict (weakened marked inline `*(weakened: <skeptic note gist>)*`), class, statement, measurements with rules, and `proposal` when one exists. A `fixed` entry names the confirming evidence. Read coverage per seat (files read fully, indexed, lines) closes the section — a reader must be able to see what was *not* read.
6. **`## Cross-area findings`** — one `### <lens>` per cross lens with the same per-finding shape, plus the lens's `re_observations`.
7. **`## Heuristic-14 measurements`** — every `h14_measurements` entry: file, prose share and whether the prose is in its right home, the named partition with per-candidate heuristic-13 symptom, stays-whole verdict. Present even when every file stays whole.
8. **`## Candidate streams (for deliberation)`** — the synthesis `streams`: name, member keys, remedy, deletable code, size, risk, mode, seam candidates, open choices with options / trade-offs / long-term consequence. Nothing here is a decision.
9. **`## Invariant candidates`** — statement, heuristic, checkable form, fixture sketch, or the existing INV it maps onto.
10. **`## Contradictions`** — between, conflict, resolution or open question. Empty section allowed, must be present.
11. **`## Exonerations`** — what the metrics flagged that reading showed healthy, and why.
12. **`## Critic`** — the critic's join errors by kind with the fix applied, whether the synthesis was revised, and anything the lead overruled with the reason.
13. **`## Dropped and degraded`** — seats lost, findings dropped by validation (with the problem), lanes not run, instruments not re-run, fable seats wanted and not run. Never empty by omission: write `none` explicitly.
14. **`## Mechanical filings`** — one row per mechanical finding: key, dedup step that decided it (contract §8 step 1a/1b/1c/2/filed), ticket or task id, disposition. Structural findings are listed as `open → deliberation`.
15. **`## Next`** — the deliberation hand-off: the program doc path expected, and that the contract §11 trigger chain is filed by the program's last act (its ids land in the program doc's FILED section).

Evidence hard rules: every finding has an anchor that resolved at `as_of_sha`; every measurement carries its rule; every proposal names what becomes deletable; weakened never disappears at the markdown layer.

## Artefact 2 — `<plans_dir>/review-all-<project_id>-<YYYYMMDD>[-<n>].json` (the record)

The Phase 5 digest of the workflow result, with keys minted; the markdown above is rendered from this file. `method` is byte-for-byte the yaml block's content.

```
{
  "method": { run_id, as_of_sha, since, evidence: {...}, verification: {confirmed, weakened, refuted, unverified}, cost: {...}, inputs_consumed: [run_id...], extra: {...} },
  "metrics": { per_area: {<area>: {...}}, totals: {...}, delta_vs_since: {...} | null },
  "findings": [ <contract §1 fields: key, area (plain briefing key), anchor, tags, severity, evidence_source, statement, proposal?, verdict, disposition, first_seen, last_seen (list), supersedes (list, usually empty), sub_area?>
                + run-local: canonical, display_id, seat, class, next_change, measurements [{name, value, rule}], verdict_notes, observation: new|re-observed|standing|fixed, also_seen_by? ],
  "refuted": [ {key, canonical, seat, display_id, anchor, why} ],
  "prior_verdicts": [ {canonical, key, sub_area?, status, checked, gone_verdict?} ],
  "unchecked": [ {canonical, key, seat, sub_area?, why} ],
  "h14_measurements": [ ... as emitted ... ],
  "synthesis": { matrix, streams, invariant_candidates, contradictions, exonerations, method },
  "critic": { join_errors, verdict, notes, synthesis_revised },
  "dropped": [ ... ],
  "filings": [ {key, step, ticket_or_task, disposition} ]
}
```

`first_seen` is the run that first recorded the key, read from the earliest artefact of any instrument carrying it (hotspot, review, census, review-all); `last_seen` is the prior artefact's list with this `run_id` appended for a re-observed key, `[]` for a new one; `supersedes` lists the keys this finding replaces (the digest fills it from the script's `supersedes_canonical`); `sub_area` is carried through, never hashed. Keys are minted by the one implementation `extra.key_implementation` names. `disposition` at commit time is `open`, `filed:<id>`, `accepted:<id>`, `refuted` or `fixed:<as_of_sha>`; the program's `/prd` sessions change `open` to `filed:<id>` in the task store, not in this file (contract §7: the store is the home; the next run re-derives).

## Artefact 3 — `<plans_dir>/review-all-program-<project_id>-<YYYYMMDD>.md` (written by deliberation, not by the run)

The coordination contract every `/prd` session of the program reads first. Section shape follows `skills/hotspot-survey/references/report-format.md` §Artifact 3 with these deltas:

- **`## Streams`** gains a `Finding keys` column; every stream lists the keys it acts on, and no key appears in two streams.
- **`## Seam ownership (G4)`** unchanged in shape; seeded from the synthesis `seam_candidates`.
- **No invariant registry.** Candidate invariants that survive deliberation are promoted through the four-site lockstep edit `CONTRIBUTING.md` §6 describes and live in `docs/legibility/design-invariants.md`; the program doc carries a pointer per promoted INV id and the owning stream, never a second list (INV-9).
- **`## Resolved decisions (do not relitigate)`** — the deliberation rulings, each with the options considered and the long-term consequence that decided it (`deliberation.md` §Recording).
- **`## Shared conventions`** adds the filing stamps: `metadata.x_finding_key` (list), `metadata.x_finding_run = <run_id>`, `metadata.source = "review-all"`, severity → priority, `planning_mode=True` → `add_dependency` → one `commit_planning`; `metadata.files` file-level; an acceptance is a `cancelled` task with `metadata.x_acceptance_reason` (never `deferred`); the **freshness clause** — anchors were verified at `as_of_sha`; before building a task on one, `git diff <as_of_sha>..main -- <path>` and re-read the symbol.
- **`## Triage briefing`** optional; include when a stream changes what looks stuck or what counts as a regression.
- **`## FILED — program status`** appended as streams land: PRD path, task anchors, coordinator interventions, the gate id(s) of the trigger chain filed per `skills/_shared/filing-the-trigger-chain.md` (`metadata.trigger_chain.skill == "review-all"`, this `run_id`), the superseded chain's run id and the `review_due` escalation resolved in its §5 step 2 if any, and `none — no tasks filed` when the program filed nothing.
