# Output contract — survey report, findings JSON, program doc

The normative contract for every quality instrument is `docs/quality-findings-contract.md` (cited below as "contract §n"). This file states only what is specific to this survey: how its review vocabulary maps onto the contract's fields, the report's survey-specific sections, and the program doc the deliberation session writes. It merges the strengths of the two proven runs: dark-factory 2026-07-06 (machine-readable findings, adversarial verdicts, contradiction resolutions) and reify 2026-07-05 (churn exonerations, S/M/L+risk tags, triage briefing). Neither proven run conforms to this file; the first refresh brings the DF baseline into conformance (`SKILL.md` §Refresh mode).

## Survey vocabulary → contract fields

**Area and sub_area** (contract §1, §3): `area` is the plain `review/briefing.yaml` subproject key (or `repo`), e.g. `orchestrator`; the cluster key goes in `sub_area`, e.g. `merge-queue`, and never enters the key. Each Phase 0 cluster entry carries its `area`: the subproject whose member directory contains the cluster's core files (contract §3's path rule), or `repo` when they span subprojects. The cluster key is the survey's fixed subsystem vocabulary, never a second area list.

**Anchor** (contract §1): the reviewer's single `anchor`, `path::symbol` of the enclosing top-level or class-level definition. Supporting locations go in `anchors`, same shape. No `file:line` anywhere in the artefacts.

**Tags.** The reviewer tags every finding with the heuristics it rests on, judged against the quality definition its prompt carries (contract §10). The `kind` enum stays as a survey-local lens and always appears as a `kind:<kind>` tag. Because `tags[0]` enters the key (contract §2), a kind's primary tag is fixed by the block below rather than chosen per finding, so a re-run reproduces the key. A `null` entry takes the reviewer's primary heuristic, which the skeptic confirms. This block is the single home of the mapping; `scripts/findings_artefact.py` loads it from here.

<!-- kind-tags -->
```json
{
  "primary_tag": {
    "missing-invariant": "h10",
    "state-machine-gap": "h10",
    "duplication": "h11",
    "cross-system-patching": "h7",
    "mismatched-abstraction": "h9",
    "redundant-abstraction": "h9",
    "stringly-typed": "h12",
    "porous-boundary": "h13",
    "god-module": "h6",
    "latent-bug": null,
    "other": null
  },
  "usual_secondary": {
    "missing-invariant": ["h11", "inv-2"],
    "state-machine-gap": ["h12", "h8"],
    "duplication": ["inv-5"],
    "cross-system-patching": ["h11"],
    "mismatched-abstraction": ["h13"],
    "redundant-abstraction": ["h11"],
    "porous-boundary": ["tests"],
    "god-module": ["h14"]
  },
  "severity_from_impact": {"high": "high", "medium": "medium", "low": "low"}
}
```
<!-- /kind-tags -->

- `god-module` carries `h14` only when its statement records heuristic 14's measurement protocol from `docs/code-quality.md` (the named partition and the heuristic-13 symptom per candidate); otherwise it is an `h6` purpose finding and its proposal is not a split (contract §6).
- `latent-bug` and `other` still put a heuristic or stance first, never `kind:`. Heuristics 1, 2, 3, 4, 5, 8 and the two stances usually arrive this way.

**Severity** (contract §4): reviewers emit `severity` directly, judged by contract §4's definitions. `effort` (cost of the fix) stays a separate field, is never a severity input, and feeds only the remedy ranking. The 2026-07-06 JSON carries `impact` instead; `severity_from_impact` above is its whole mapping, because impact already measured the cost of leaving the finding, which is what severity is, so no effort value moves it.

**Evidence source** (contract §1): `present-tree` for every reviewed finding, plus `fix-history` when `bug_history_link` names commits or tasks rather than `speculative`. Derived at digest time, never asked of an agent.

**Route** (contract §6): `mechanical` for a `latent-bug` with one anchor and no design choice, `structural` for everything else. A `confirmed` mechanical finding is filed by the survey itself in Phase 5 as a curator ticket after contract §8's dedup protocol, and carries `filed:<id>` (or the `filed:`/`accepted:` the protocol found); everything else goes to deliberation and the survey files none of it (`SKILL.md` §Phase 5, §Hand-off).

## Artifact 1 — `bug-hotspot-survey-<date>[-<n>].md` (the synthesis report)

Required sections, in order:

1. **`## Method`** — directly under the H1. Its first element is a fenced ```yaml block that is the JSON's `method` object rendered with `yaml.safe_dump`: exactly contract §5's seven keys (`run_id`, `as_of_sha`, `since`, `evidence`, `verification`, `cost`, `inputs_consumed`) plus `extra`, which holds everything survey-specific: `mode` (`full` | `refresh`), `prior_run_id`, `findings_json`, `max_task_id` (the task-store high-water mark the next refresh mines above), `fix_signal` (`{chosen: {fix, total}, raw: {fix, total}, amend: {count, total}}`, so the chosen ratio sits beside the raw one, `SKILL.md` §Phase 0 step 2), `lanes_run`, `launched_from` (the launching escalation id, if any), and `verification_skipped_reason` (required when any verdict is `unverified`). After the block, at most three lines of prose: lanes run or lost, team shape, which dedup step decided each mechanical filing (or that the store was unreachable and nothing was filed).
2. **`## Ranked hotspots`** — `# | Hotspot (area, files) | Evidence | Root structural cause`. Evidence is fix history: fix-commit count and chosen ratio, recency, incident and task ids. File sizes may appear as context, never as the ordering (contract §10). The ordering says where the survey looked; it is not a severity.
3. **`## Churn exonerations`** — clusters or files hot by raw count but healthy, with the reason, from the reviewers' `churn_exoneration`. Must be present, may be empty.
4. **`## Latent bugs (route: mechanical)`** — numbered; each gives key, `path::symbol` anchor, **bolded one-sentence consequence**, history cross-ref, severity, and disposition (`filed:<id>` once Phase 5 filed it).
5. **`## Per-hotspot findings`** — one `### N. <area>` per ranked hotspot:
   - **Diagnosis.** Dense prose; every claim cites its finding key.
   - **Proposals (ranked).** Bullets, each: **named mechanism** (type/chokepoint/table/invariant and where it is enforced), what existing code becomes **deletable**, size (S/M/L) + risk, finding key(s), back-refs to latent bugs.
   - Mark weakened findings inline (`*(weakened: <skeptic note gist>)*`).
6. **`## Carried forward`** (refresh only) — per prior key: re-verification outcome (`confirmed` / `weakened` / `refuted` / `fixed:<sha>`), disposition, and `supersedes` where the anchor moved. Findings another instrument already recorded (contract §9) are listed here with the field that changed, not re-reported.
7. **`## Cross-system chains`** — numbered, named chains: member areas, the missing guarantee, the compensation inventory it spawned, the fundamental fix, member finding keys.
8. **`## Ranked priorities (payoff × feasibility)`** — the single canonical ranking of remedies (5–8 items). Other sections point here.
9. **`## Contradiction resolutions`** — must be present, may be empty. These become the program doc's resolved decisions.

**Evidence rules:** every finding has an anchor at `as_of_sha` or an explicit *hypothesis* marker, cites bug history or says `speculative — prophylactic`, and every structural proposal names what becomes deletable.

**Ranking rubric.** Hotspots by fix-history evidence, which orders where to look and nothing else. Remedies by payoff × feasibility: payoff ≈ (findings closed, by key and severity) + (compensations deleted) + (open incident classes closed); feasibility = S/M/L effort + risk. A remedy's historical fix count may describe its reach in prose but never ranks it, and "the hotspot will cool" is never a payoff: `docs/code-quality.md` §What to measure lists the fix rate as lagging.

## Artifact 2 — `bug-hotspot-survey-<date>[-<n>]-full-findings.json`

Top level: `{ method, clusters, findings, themes, cross_system }`.

- `method`: the object the md's `## Method` block renders — contract §5's seven keys plus `extra` (Artifact 1 item 1 lists its keys).
- `clusters[*]`: `{key, area, files, material_change, architecture_notes, churn_exoneration, cross_system_notes}`. `files` is the list the next refresh diffs against; `material_change` records the `git diff --shortstat` figure that selected or skipped its reviewer.
- `findings[*]`: one flat list carrying every contract §1 field — `sub_area` always set to the cluster key, `last_seen` and `supersedes` as lists (`supersedes` holds prior keys, or `positional:clusters[i].findings[j]` for the 2026-07-06 baseline) — plus `id` (display id `<cluster>.<n>`, contract §2), `kind`, `title`, `anchors`, `bug_history_link`, `effort`, `route` and `verdict_notes`. Refuted and fixed findings stay in the list under their disposition; there is no separate `refuted` array.
- `themes`: mining output, `{subsystem, theme, evidence, count_estimate}` per entry.
- `cross_system`: `{chains, top_priorities, contradictions}` from synthesis, chain members cited by key.

**Conformance is machine-checked.** Before committing, the coordinator runs

```bash
python3 <skill>/scripts/findings_artefact.py check --findings <json> --report <md> --repo "$ROOT"
```

and does not commit on a non-zero exit. Each failed assertion prints the finding and the offending values (INV-2). It asserts:

1. every contract §1 field is present with its shape; `tags[0]` is a heuristic or stance, `tags` contains `kind:<kind>`, and `tags[0]` equals the kind's fixed primary tag where the block above fixes one;
2. `key` recomputes from the plain `area`, the normalised anchor and `tags[0]` per contract §2 (no `sub_area` in it), and no two findings share a key;
3. `area` is a `review/briefing.yaml` subproject key or `repo`, and `sub_area` is a `clusters[*].key`;
4. anchors are normalised (repo-relative, no line numbers), and for every finding not `refuted` or `fixed:` the anchor's path exists at `as_of_sha` and its symbol occurs in that file there;
5. `verdict`, `disposition`, `severity`, `route` and `evidence_source` are in their enums; a `refuted` verdict goes with a `refuted` disposition and vice versa; weakened and refuted findings carry `verdict_notes`; `unverified` appears only with `extra.verification_skipped_reason`;
6. `method` has exactly contract §5's seven keys plus `extra`, `as_of_sha` (and `since` unless `none`) resolve to commits, and `verification` equals the tally of verdicts;
7. `first_seen` is a run id; `last_seen` is a list of run ids ending with this run; `supersedes` is a list whose entries are keys or positional refs;
8. the md has a `## Method` section whose first element is a fenced yaml block that loads equal to the JSON's `method`, every `fk-` key the md cites exists in the JSON, and every open, filed or accepted finding's key appears in the md.

## Artifact 3 (successor, not written by the survey) — the remediation program doc

Written by the deliberation session as `bug-hotspot-remediation-program-<date>.md`. Sections:

1. **`## Streams`** — `ID | Slug | Scope | Finding keys | Mode (agent /prd vs spawned interactive /prd) | Wave | Upstream deps`. Streams sized for one /prd session. Confirmed mechanical findings were filed in Phase 5 and appear here only by their task ids; a weakened one the deliberation still wants fixed becomes an agent-mode micro-stream.
2. **`## Seam ownership (G4 — authoritative)`** — `Seam/artifact | Owner stream | Consumers (do NOT redefine)`.
3. **`## Invariants`** — each invariant candidate the deliberation accepted is either an existing `docs/legibility/design-invariants.md` id that a named stream enforces, or a proposal to add one there through the four-site lockstep edit `CONTRIBUTING.md` §6 describes, owned by one stream. The program doc never mints INV-ids of its own; that file is the single list.
4. **`## Resolved design decisions (do not relitigate)`** — contradiction resolutions promoted to rulings, plus deliberation rulings, each defended here or in the task record (contract §12), never in the survey report.
5. **`## Shared conventions (every session)`** — filing mechanics (agent_id tagging, `planning_mode=True` + bulk `commit_planning`, file-level `metadata.files`); every filed task carries `metadata.x_finding_key` (the keys it closes), `metadata.x_finding_run` (the survey's `run_id`) and `metadata.source="hotspot-survey"` (contract §8), after contract §8's dedup protocol per key; findings are cited by key, never by array position or display id; the trigger-chain gate id and the rule that each session adds its `critical`/`high` tasks as its dependencies (`skills/_shared/filing-the-trigger-chain.md` §4); `git commit --only`; and the **freshness clause**: "re-verify any anchor you build a task on — the survey is pinned to `<as_of_sha>` and main moves".
6. **`## Triage briefing`** — postures that change triage, structural changes in flight, things that look stuck but aren't, dup-check guidance.
7. **`## FILED — program status`** (appended during execution) — `Stream | PRD | Task id | Finding keys | Severity`, one row per task, plus the trigger-chain task ids and a coordinator-interventions log.
