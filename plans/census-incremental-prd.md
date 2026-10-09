# PRD: incremental, contract-conformant legibility census

**Project:** dark-factory (`scripts/legibility/`, one helper in `shared/`). **Status:** draft, authored 2026-10-04 in the quality-skills-alignment worktree; not yet committed or decomposed. **Type:** extension of `plans/confusion-reduction-prd.md` (task η, the census runner, and ζ, its trigger; both landed). **Approach:** B+H. The batch introduces more than eight mechanisms, changes the codebook schema that the nightly trickle and the census both write, and produces ticket metadata that two sibling PRDs read. It therefore carries contracts (§4) and a boundary-test sketch (§4.8).
**Origin:** Leo's rulings of 2026-10-04 (§5.1). The normative contract is `docs/quality-findings-contract.md`, cited below by section number and never restated. The quality definition is `docs/code-quality.md`, cited by heuristic name.
**Code anchors** were verified against main `92af0716e8` (2026-10-04). Citations are `path::symbol`. Re-locate them at implementation time.

## 1. Goal (G1 consumer + user-observable surface)

The census stays automated. After this batch, a run spends only on transcripts nobody has coded and on findings nobody has catalogued. Its findings can be joined with the other instruments' findings, it carries its own dispositions forward, and its cadence follows what has landed. Observable on the next automatic census report after the batch lands (`plans/confusion-census-<date>[-<n>].md`), checked mechanically by gate **G** (§9):

- Each run writes a JSON record and the markdown rendered from it (contract §5). The markdown opens with a `## Method` fenced YAML block carrying the seven §5 keys, plus `extra`. `evidence` counts the sessions skipped because they were already coded and those skipped for zero signal. A same-day rerun writes a `-2` record and report and never overwrites the first.
- A `## Findings` table: one row per finding the run touched, with every contract §1 field: `key` `fk-…`, `area`, `sub_area` (null for the census), `anchor`, `tags`, `severity`, `evidence_source`, `statement`, `proposal`, `verdict`, `disposition`, `first_seen`, `last_seen` (list), `supersedes` (list).
- `## Filed Tasks` names task ids as well as tickets, and says which §8 dedup step decided each finding. `## Dispositions` carries the previous runs' tickets forward to tasks, and those tasks' outcomes back to codebook entries.
- `## Screened` counts the clusters attached to an existing record with no verify spend. `## Adjudication` reports the pending-candidate queue's depth, how many entries it processed this run, and an audit line for every similarity attach.
- `## Synthesis` is rendered from validated JSON, split into new findings and re-observed ones, with no re-narrated history.
- `scripts/legibility/census_trigger.py evaluate` prints these reason lines: completion share of the previous run's filed tasks, a relative novelty spike measured against its trailing baseline, and the floor measured against the session watermark. There is no max-interval line: the calendar backstop is dropped (C7).

Consumers: Leo and the operator, through the report and `skills/census/SKILL.md`. The nightly trickle's `scripts/legibility/nightly.py::evaluate_census_step` consumes the trigger. Through the codebook's `finding_key`/`finding_anchor`, contract §9 makes `/review`, `/hotspot-survey` and `/review-all` consumers too. `scripts/check_run_completion.py` (owned by `plans/completion-driven-triggers-prd.md`) reads `x_finding_run`.

## 2. Background: evidence (dated pointers, INV-5)

These facts come from the census study of **2026-09-30**, which was verified against code and data. The PRD rests on them as measured on that date and does not re-anchor them:

- The census mines **session transcripts**, not the codebase (`census.py::default_batch_source`). Only the verifier reads the tree.
- 977 of 1,105 candidates were `pending`. Every candidate has exactly one sighting, because the merger (`codebook.py::apply_coding_record`) groups candidates by exact title. A pending candidate is re-adjudicated only if a later run reproduces its exact title, and the miner's index (`coder.py::build_codebook_index`) shows entries only. Trickle candidates are therefore effectively write-only.
- The verifier does not screen the codebook before it spends a call. In the 09-10, 09-20 and 09-25 reports, 6 of 12 verified findings came from sessions that were already catalogued (09-25's single finding was pending twice).
- 7 of 13 runs stopped at the two-batch saturation minimum: 280 miner calls for 8 verified findings. Two things inflate `dup_rate`. The census re-codes sessions the trickle already coded, because no processed-session ledger exists. It also keeps zero-signal sessions that the trickle's sampler drops.
- Synthesis re-narrates standing conclusions on every run: "no merge/verify-manifested sighting for the Nth consecutive cycle" appears in 12 of 13 reports. Its prose is never parsed, although the reports call its dispositions "inputs to the merger".
- All 89 census-promoted entries have severity `medium`, and every ticket defaults to priority `medium`; no stage assigns either. `invariant_violated` is non-null on 230 of 3,714 sightings, and none of those 230 equals an INV slug. The coder prompt shows the field as `null` and gives no slug list. Ownership of the slug check is circular between `plans/confusion-reduction-prd.md` §10 and `docs/legibility/design-invariants.md` §Census seam.
- Tickets carry only `{source, origin_project_id, x_fix_surface?}`. Entries record no ticket or task id. Task `done`/`cancelled` never flows back to an entry. The default verifier always returns `fixed=[]` (`census.py::_build_default_verify_fn`).
- 11 of 13 reports said "none filed", because `run_census` read `submit_task`'s result as `{"id"}` instead of `{"ticket"}`. Task 4965 fixed this on 09-23. 23 tasks with `source=legibility_census` exist, 19 of them done. The report still cannot name task ids, because the curator decides after the report is rendered.
- The date-only anchor makes each run re-mine the anchor day, and same-day reruns overwrote the 09-20 report. Cadence sits at the 5-day floor (median gap 5.0 days over 12 gaps), because today's condition (c), the absolute novelty spike, always fires: 19–86 candidates per 72 h against a threshold of 4.

Measured for this PRD on **2026-10-04** (committed codebook at `92af0716e8`, read-only):

- 1,132 candidates: 1,001 `pending`, 90 promoted, 41 rejected. All have one sighting each. The miner's entry index is 21,464 chars. A pending-candidate index in the same `- id: title — cause` form would be 289,778 chars, about 13.5× larger.
- Replaying 72 h candidate counts over 2026-08-15..10-03 (50 days, using `first_seen` dates) gives min 0, median 43.5, max 108. The current absolute threshold (≥ 4) fires on 48 of 50 days. A relative rule (≥ 2× the trailing 30-day median and ≥ 4) fires on 3 of 50.
- The 10-03 census (the fourteenth, after the study) verified one finding. Its own synthesis records that the finding was already the pending candidate `cand-20260925-19`, under a different title and session. The run promoted the twin `cand-20260925-21` to `entry-cand-20260925-21` (severity `medium`) and left `cand-20260925-19` pending, so one event now has two records. Its synthesis refined the origin phase to `merge`, but the runner's matrix rendered the verifier's stamp. `census-state.json` holds an absolute `last_census_report` and a date-only `last_census_at`.
- Reify has `review/briefing.yaml` and `docs/legibility/design-invariants.md` (an `INV-SF-<n>`/`INV-AD-<n>` id family) but no `docs/code-quality.md`. The verifier's sandbox is the censused tree (`census.py::census_stage_specs`), so a prompt that tells it to `Read docs/code-quality.md` fails on reify.

## 3. Sketch of approach

Thirteen tasks. **L1** gives every run an identity (`run_id`, `as_of_sha`, the §5 header) and a host-local processed-session ledger. The trickle and the census both write that ledger, and the census mines only un-ledgered, non-zero-signal sessions inside a fixed retention window, so a capped run resumes instead of losing its remainder. **L8a** repairs the trigger's novelty and floor conditions. **L2** makes the verifier return a typed verdict: anchor, tags, severity with a reason, route, and remediation, judged against the rendered quality definition. **L9** gives the coder the project's invariant slugs and the quality definition's `## Definition` section as its minting test, and makes the merger slug-check the slugs. **L3** mints finding keys onto codebook records and tickets. **L4** runs the contract §8 dedup protocol and back-links tickets to tasks. **L5** feeds task outcomes back to entries. **L6** screens novel clusters against the codebook before any verify spend. **L7** makes pending candidates reachable through a similarity index and a capped adjudication queue. **L10** structures synthesis. **L8b** replaces the tasks-landed condition with contract §11 completion. **L11** and gate **G** prove the whole chain on a live report.

New mechanisms land in new modules. `scripts/legibility/census.py` is 4,035 lines, which is past heuristic 14's alarm. Each leaf grows it only by wiring, and no new module imports `census.py` (heuristic 13). Splitting `census.py` is a structural finding under contract §6, out of scope here.

## 4. Contracts (H)

### 4.1 C1: codebook delta (extends `plans/confusion-reduction-prd.md` §7.1; append-only, never-delete unchanged)

All new fields are optional, written only by the census, and validated by `codebook.py::validate` when present. L3 lands the whole schema delta in one edit.

- **Entry:**
  - `finding_key` (matches `^fk-[0-9a-f]{12}$`), `finding_area` (a plain §3 area key), `finding_anchor`, `finding_tags` (a list of at least one tag, contract §1 vocabulary).
  - `severity_reason` (string). `first_seen_run` (a run id) and `last_seen_runs` (a list of run ids, appended per contract §9). `supersedes` (a list of keys, usually empty; contract §2).
  - The entry carries the fields that join it to other records. Every remaining §1 field lives in the run's JSON record (C3), derived as follows:
    - `statement`: the entry's `cause` plus the verifier's `reason`.
    - `proposal`: `remediation.change`, or null.
    - `evidence_source`: always `agent-transcripts`, plus `present-tree` when the verifier confirmed the finding at `as_of_sha`.
    - `sub_area`: null.
    - `verdict` and `disposition`: per C4 and contract §7.
  - `filed_tickets`: a list of `{ticket: str|null, project_root, run_id, finding_key, task_id: int|null, resolution: created|combined|refused|failed|existing|null, step: key|semantic|curator}`.
  - `fixed_at_sha`, `accepted_by` (task id).
  - `status` gains `accepted` in `codebook.py::STATUSES`.
  - The v1 free-form `filed_tasks` (16 legacy entries, strings or lists) is left untouched. `filed_tickets` is the structured back-link.
- **Candidate:**
  - The four `finding_*` fields once verified (promoted or rejected).
  - `attached_by: {method: session|title|key|similarity, score?: float, run_id}` when promoted onto an existing entry without a verify call.
  - `adjudicated_with: <cand id>` for members of an adjudicated similarity cluster.
- **Sighting:** `invariant_violated` is a slug of the observed project's `docs/legibility/design-invariants.md` or absent. It is enforced on write (`codebook.py::validate_coding_record`), not by `validate`, so the 230 legacy free-text values stay valid.
- `codebook.py::_validate_node` gains `pattern`. `census.py::_VALID_ENTRY_SEVERITIES` is deleted in favour of the codebook's own enum (INV-5).

### 4.2 C2: run identity, census state, ledger, window

- `run_id = census-<project_id>-<YYYYMMDD>[-<n>]` (contract §5). `n` is the first free suffix for which neither the report path nor the run id exists. The report path follows it: `plans/confusion-census-<YYYY-MM-DD>[-<n>].md`.
- `docs/legibility/census-state.json` is written atomically by `census.py::advance_census_state` as `{last_census_at: <YYYY-MM-DD>, last_census_run_id, last_census_report: <repo-relative>, last_census_as_of_sha, session_watermark: <ISO datetime UTC, the exclusive upper bound of the run's enumerated window>, last_census_done_count}`. `last_census_done_count` is dropped by L8b. `census_trigger.py::load_census_state` accepts state files missing the new keys (the transition case) and hand-seeded ones.
- The ledger is sqlite at `trickle_state.py::project_state_dir(project_id)/coded-sessions.sqlite`. It is host-local because transcripts are host-local, and a committed ledger would dirty the machine-operated checkout every night (`plans/confusion-reduction-prd.md` decision 7). It holds one table, `coded_sessions(session PRIMARY KEY, instrument_version INT, coded_by ∈ {trickle, census}, run_ref, outcome ∈ {matched, candidate, empty}, coded_at)`.
  - `session` is the same string sightings carry.
  - Only successfully coded digests are recorded. A failed coding is never ledgered, so it is retried.
  - Rows whose session date falls before the window start are pruned at census start. The window is `census.ledger_retention_days` long (default 30, the current `_DEFAULT_CENSUS_LOOKBACK_DAYS`).
  - The public API lives in a new `scripts/legibility/session_ledger.py`, plus a `stats` CLI.
- **Window:** sessions dated in `[today − ledger_retention_days, today)` UTC, minus ledgered sessions, minus zero-signal sessions (the sampler's own `score > 0` gate, `sampling.py::ScoredRecord`).
  - **Transition:** while the ledger holds no `census` rows, the lower bound is `max(that, last_census_at)`, so the first run after L1 does not re-mine the history that predates the ledger.
  - A capped run leaves its unmined sessions un-ledgered, and the next run resumes them.
- Decision 2 of the parent PRD (idempotency through sighting identity) stands for the merge. The ledger is the mining-skip key, a different question.

### 4.3 C3: report

**Record and rendering (contract §5).** Each run first writes a JSON record, `plans/confusion-census-<YYYY-MM-DD>[-<n>].json`. The record holds the method block, every finding the run touched (all §1 fields), the screened, attached and adjudicated sets, the filed tickets and tasks, and the synthesis document (C6). `census.py::census_report_sections` then renders `plans/confusion-census-<YYYY-MM-DD>[-<n>].md` from that record alone, and both files share one basename. The record is the source; the rendering is never edited by hand. The commit carries both files, along with the codebook and state.

The renderer gains stable keys: `SECTION_METHOD` (L1), `SECTION_FINDINGS` (L3), `SECTION_DISPOSITIONS` (L5), `SECTION_SCREENED` (L6), `SECTION_ADJUDICATION` (L7), and `SECTION_STRUCTURAL` (L4: structural findings, not filed, for deliberation per contract §6).

`## Method` holds a fenced `yaml` block of the seven §5 keys plus `extra`, which takes every census-specific key:

- `evidence`: `{window: [start, end), sessions_enumerated, skipped_coded, skipped_zero_signal, mined, ledger_rows}`.
- `verification`: `{confirmed, weakened, refuted, unverified}`.
- `cost`: `{miner_calls, verify_calls, synthesis_calls, probe_calls, embedding_calls, wall_clock_secs}`.
- `inputs_consumed`: the `run_id` of the latest `/review`, `/hotspot-survey` and `/review-all` reports whose finding keys the pre-screen matched sightings against (contract §9; L6). If none exist, the value is `[]` and `extra.inputs_consumed_note` says why.
- `extra`: `{screened, attached, adjudicated, ledger_created_this_run, verify_normalisations, slug_rejections}`.

Every gating rule of the existing NO-SILENT-CAPS sections is unchanged.

### 4.4 C4: verifier verdict (one call per cluster, parsed by a new `scripts/legibility/verdict.py::parse_verdict` into a frozen `Verdict`)

```json
{"verified": true, "reason": "…",
 "anchor": "path/to/file.py::symbol | path/to/file | slug:<kebab>",
 "tags": ["h13", "inv-11", "tests"],
 "severity": "high|medium|low", "severity_reason": "…",
 "route": "mechanical|structural",
 "remediation": {"path": "…", "change": "…"} | null}
```

The parser normalises the anchor (repo-relative, forward slashes, no line numbers) and requires the anchor's path to exist below the root, exactly as `census.py::_in_tree_remediation` already requires of a remediation. If the anchor is invalid, it falls back to the remediation path, and then to `slug:<kebab of title>`.

Tags outside the contract §1 vocabulary are dropped. `kind:confusion` is always appended, so the primary tag is a heuristic or an `inv-` id whenever the verifier names one.

A severity outside the enum becomes `medium`, with `severity_reason` recording the substitution. `critical` is never a finding severity (contract §4). A missing `route` becomes `mechanical` only when `remediation` is present, and `structural` otherwise.

Every normalisation is counted and reported, never silent (INV-11). The verifier and synthesis prompts are assembled by code, so they embed the block rendered by `orchestrator/src/orchestrator/agents/code_quality.py::guidance` (contract §10). This also covers reify, whose tree has no `docs/code-quality.md` (§2).

### 4.5 C5: finding key, area, ticket metadata

- **Key:** a new `shared/src/shared/finding_key.py::finding_key(area, anchor, primary_tag)` implements contract §2 verbatim, with §2's anchor normalisation (trailing `:N`, `:N-M` and comma lists stripped). Contract §2 names it the one key implementation (heuristic 11). Until it lands, `skills/hotspot-survey/scripts/findings_artefact.py::finding_key` is the interim reference, and L3's tests assert agreement with it on a fixture L3 commits (contract §2 parity test). That script is small skill-local tooling, which is exempt from "code via /prd" and stays (Leo, 2026-10-05); `shared.finding_key` supersedes only its key function, and nothing else of it moves (its adoption is out of scope here, §7).
- **Area:** contract §3's single path→area rule, applied to the verdict's anchor path (in practice the verifier's remediation path).
  - The area is the `review/briefing.yaml` subproject whose member directory contains the path.
  - A path under no member takes the rule named by the project's `/review-all` overlay (schema: `skills/review-all/references/project-overlay.md` "Path → area map"; for dark-factory, `scripts/legibility/**` → `shared`).
  - A `slug:` anchor, or a path that neither rule maps, gets `repo`.
  - The area is always a plain key, never `<area>/<sub>`. `sub_area` is null for the census.
  - A project with no briefing makes every area `repo`, and the report says so.
  - Caveat: `review/briefing.yaml` omits the workspace member `cockpit` (a briefing defect for `/review-briefing`, contract §3), so until it is fixed the census keys `cockpit/` anchors under `repo` while `/review-all` reports them as uncovered, and their keys can diverge.
- **Tickets** carry `metadata: {source: "legibility_census", origin_project_id, x_fix_surface?, x_finding_key, x_finding_run, x_supersedes_task?}`. `priority` equals the entry's severity.

### 4.6 C6: synthesis output (validated by a new `scripts/legibility/synthesis.py`; prose rendered from it)

```json
{"new": [{"key": "fk-…", "observation": "…"}],
 "re_observed": [{"key": "fk-…", "sessions": ["…"], "changed": [{"field": "severity|verdict|origin_phase|manifested_phase", "from": "…", "to": "…", "why": "…"}]}],
 "phase_refinements": [{"key": "fk-…", "origin_phase": "…", "manifested_phase": "…", "why": "…"}],
 "corrections": [<§7.3 correction op, entries only>],
 "notes": [{"claim": "…", "evidence": "…", "verified": false}]}
```

The runner computes "standing, no new evidence" from the codebook and renders it as a count. It is never sent to or narrated by the model.

`phase_refinements` are applied to this run's verified clusters before the matrix and promotion. `corrections` are applied through `codebook.py::apply_coding_record`'s existing corrections op. Invalid output renders `synthesis unavailable: <reason>`, files one info escalation, and the run continues, because every data section is complete without synthesis.

### 4.7 C7: trigger (`census_trigger.py::evaluate` / `decide_for_project`)

- **(a) Completion** (L8b). `scripts/check_run_completion.py` is invoked for `last_census_run_id` against the observed project, per its CLI in `plans/completion-driven-triggers-prd.md`, with threshold `census.completion_threshold_pct` (default 70, contract §11). Exit 0 means FIRE and exit 75 means not yet. Any other exit, a state with no `last_census_run_id`, or a previous run that filed no tasks (contract §11) means N/A plus one WARNING (fail safe). An N/A caused by an error exit, as distinct from those two standing states, counts toward a consecutive-error streak in host-local trickle state; at 3 consecutive such evaluations one info escalation is filed through the existing legibility `escalate_info` poster, and the next successful evaluation (exit 0 or 75) resets the streak (INV-4).
- **(b) Novelty spike, relative** (L8a). The 72 h count of candidate `first_seen` must be ≥ `novelty_spike.count`, and also ≥ `novelty_spike.multiple` × the median of the daily 72 h counts over the trailing `novelty_spike.baseline_days`. Defaults are 4, 2 and 30. While fewer than `baseline_days` of history exist, the condition is N/A.
- **No max-interval condition** (decided, R8). Today's calendar backstop is deleted by L8b. The trigger fires on (a) or (b) only, each subject to the floor. A project with no completion signal (no previous run id, or a previous run that filed nothing) therefore fires only on (b) or by hand (`skills/census/SKILL.md`).
- **Floor:** `floor_days` is the minimum transcript window, and the calendar's only role (R2). It never fires a census by itself. No condition fires until `now − session_watermark ≥ floor_days`. It is anchored on the watermark, not the date, and a never-censused project measures it from its earliest codebook date. `tasks_landed_*` and `last_census_done_count` are retired by L8b.
- Every threshold remains an unquoted non-negative int, validated by `census_trigger.py::CensusConfig.from_mapping`.

### 4.8 Boundary-test sketch

All rows run with real sqlite in tmp, real git fixtures for `as_of_sha`, and fake LLM/MCP seams injected through `run_census`'s existing parameters. No test patches a private name (Tests stance).

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | trickle-coded session excluded | ledger row `S` (trickle); window holds `S`, `T` | batch source yields `T` only; `evidence.skipped_coded == 1` |
| 2 | capped run resumes | run 1 `--max-batches 1` over 2 batches of sessions | run 2 mines exactly the unmined remainder; nothing ledgered twice |
| 3 | same-day rerun | report for today exists | run writes `…-2.json` + `…-2.md`, `run_id` ends `-2`; first record and report byte-identical |
| 4 | zero-signal session | session with `score == 0` | never digested; `skipped_zero_signal == 1` |
| 5 | verdict → key → ticket | verdict anchor `orchestrator/x.py::f`, tags `[h13]` | entry `finding_key == finding_key("orchestrator", anchor, "h13")`; ticket `x_finding_key`/`x_finding_run` set; `priority == severity` |
| 6 | malformed verdict | unknown tag, missing anchor path, severity `critical` | normalised per C4, each counted; run completes |
| 7 | dedup step 1 hit | `find_tasks_by_metadata` returns a `pending` task with the key | no `submit_task`; `filed_tickets` row `resolution: existing, step: key` |
| 8 | store unreachable | `find_tasks_by_metadata` returns an error payload (or the transport raises) | files nothing; report states it (contract §8) |
| 9 | outcome: done | back-linked task `done`; re-verify says gone / present | `status: fixed`, `fixed_at_sha = as_of_sha` / re-filed with `x_supersedes_task` |
| 10 | outcome: cancelled | cancelled task with / without acceptance reason | `status: accepted`, `accepted_by` / back to `open` |
| 11 | pre-screen | novel cluster, title equals an adjudicated candidate's | zero verify calls; sighting attached; `screened == 1`; no unresolved verdict |
| 12 | adjudication | pending candidate ≥ attach threshold to an entry; queue over cap; embedding endpoint down | attached with `attached_by.similarity`; verify calls ≤ cap; when down: stage skipped, one escalation, run completes |
| 13 | trigger | 72 h = 43 vs median 40 / 90 vs 40; watermark 3 d ago; completion exit 0 / 75 / 2; no prior run id; config still carrying the deprecated backstop key (L8b), watermark 30 d ago, no spike, no completion | no-spike / spike; floor blocks; FIRE / not / N/A+WARNING; N/A; no fire, one deprecation WARNING |
| 14 | synthesis invalid | model returns non-JSON | "synthesis unavailable" rendered; data sections intact; one info escalation |
| 15 | invariant slug | coder emits a known slug and an unknown string | known persists; unknown stored as absent and counted |
| 16 | checker | newest report lacks `SECTION_SCREENED` / has every key | exit 1 naming it / exit 0 |
| 17 | coder minting test | a digest whose confusion does not meet the definition (no cost or risk to the next change); fake coder seam returns no candidate for it | no candidate minted; ledger `outcome: empty`; the prompt the seam received carries the `## Definition` body and none of the fourteen heuristics |

## 5. Resolved design decisions

### 5.1 Leo's rulings (2026-10-04, binding)

R1 The census stays automated.
R2 Its trigger is task-completion driven, using the contract §11 measure over the tasks its previous run filed. The calendar floor is kept only as the minimum transcript window.
R3 The novelty-spike condition is fixed.
R4 There is no new findings store. Entries carry `finding_key`, tickets carry `x_finding_key`/`x_finding_run`, and an entry↔ticket back-link exists.
R5 The census never re-mines sessions that are already coded and never re-verifies findings that are already catalogued.
R6 Judging prompts read the quality definition by reference.
R7 `find_tasks_by_metadata` is consumed by name from `plans/task-metadata-lookup-prd.md`.

Leo, 2026-10-05 (binding; closes the former Q1 and Q2):

R8 The calendar backstop (today's max-interval condition) is dropped. The trigger conditions are (a) weighted completion of the previous run's filed tasks (contract §11) and (b) the relative novelty spike; the floor stays only as the minimum transcript window. There is no max-interval condition (C7).
R9 The coder (Haiku trickle and Sonnet miner) does not receive the fourteen heuristics. It receives only `docs/code-quality.md`'s `## Definition` section, rendered by the `code_quality.py` slicer, as the test for minting a candidate. All heuristic tagging stays with the verifier (L2). Closes the former Q1 (D5, L9).

### 5.2 Decisions made in this PRD (not in the contract or the rulings; flagged for review in §13)

- **D1 Ledger is host-local sqlite** (C2). Losing it fails toward re-mining, which costs money but stays correct (the merge is idempotent). The report shows `ledger_created_this_run`, so an empty ledger is never read as "nothing coded" (INV-13).
- **D2 Window is retention-bounded, not anchored on a date.** `session_watermark` serves the trigger floor and the report's `since`. It does not serve as the window start, because a window starting at the watermark would strand capped-away sessions.
- **D3 The pending adjudication path is a similarity index plus a capped queue, not pending titles in the miner index.** The index option grows every miner and trickle prompt about 13.5× (289,778 against 21,464 chars, §2) on the standing nightly spend.
  - The chosen path embeds title plus one-line cause through the OpenAI embeddings endpoint, using `httpx` (already a `shared` dependency). `OPENAI_API_KEY` reaches the unit through `EnvironmentFile=` in `scripts/legibility-trickle@.service`.
  - Vectors are cached host-local, keyed by `(record id, sha256(text), model)`.
  - One-time backlog cost is about 1,001 × ~300 chars ≈ 75 k tokens, a fraction of a cent at list price. Steady state is under 100 new candidates per run.
  - The queue spends at most `census.adjudication.max_clusters_per_run` verify calls (default 10). Recurring clusters, by distinct sessions, go first, then the newest. The 1,001 backlog drains over many runs, and the report shows the depth.
- **D4 Thresholds for similarity are provisional** (`attach_min_similarity` 0.90, `cluster_min_similarity` 0.85; G6). Every attach is audited in the report until they are calibrated.
- **D5 The verifier and synthesis get the full rendered quality block (`guidance()`); the coder (trickle and miner) gets only the definition** (R9). The coder judges transcripts, not code. Tags are assigned where code is read (L2), and candidates are not findings until verified (contract §1).
- **D6 Structural findings are recorded, not filed** (contract §6). The verdict's `route` decides. Today such clusters are filed whenever they recur.
- **D7 Tickets resolve in-run, with a bound:** `resolve_ticket` per ticket, 300 s in total, before render. The remainder resolve on the next run.
- **D8 `fixed` requires re-verification.** A back-linked task going `done` queues one verify call at `as_of_sha`, counted against the same verify cap. "Gone" sets `status: fixed` and `fixed_at_sha`. "Present" re-files with `x_supersedes_task` (contract §7, §8 1c). The codebook status is a projection reconciled from the task store on every run, never an independent copy (INV-9).
- **D9 Completion (a) evaluates the observed project only.** Cross-project tickets are reported, but they do not gate.
- **D10 `scripts/legibility/invariants.py` reads `## INV-<id> \`<slug>\`` headings from a project's own design-invariants doc.** It accepts both the `INV-<n>` and the `INV-<FAMILY>-<n>` id forms. `scripts/tests/test_design_invariants_consistency.py::_HEADING_RE` is replaced by an import of it (INV-5).

## 6. Pre-conditions / substrate (G3)

Verified on `92af0716e8`:

- **Mining and filing:** `census.py::default_batch_source`, `::_census_window_dates`, `::mine_to_saturation`, `::_novel_clusters`, `::_find_pending_candidate_id`, `::_split_fileable`, `::_filing_queue`, `::promote_candidate`, `::retire_entry`, `::build_task_payloads`, `::_ticket_id_from_submit_result`.
- **Verify, synthesis and report:** `census.py::_verify_prompt`, `::_build_default_verify_fn`, `::_in_tree_remediation`, `::_synthesis_prompt`, `::_build_default_synthesize_fn`, `::census_report_sections`, `::advance_census_state`, `::run_census`, `::main`.
- **Runtime plumbing:** `census.py::census_stage_specs`, `::_free_payloads_path`, `::_post_mcp_tool_call`, `::_SHARED_SRC`, `::DEFAULT_HARNESS_CONFIG_PATH`, `::_VALID_ENTRY_SEVERITIES`.
- **Trigger:** `census_trigger.py::decide_for_project`, `::evaluate`, `::codebook_signal`, `::load_census_state`, `::CensusConfig`, `::compute_tasks_landed`, `::default_status_fetcher`.
- **Coder and codebook:** `coder.py::build_codebook_index`, `::build_prompt`, `::parse_coder_output`; `codebook.py::apply_coding_record` (its corrections op included), `::validate_coding_record`, `::validate`, `::_validate_node`, `::STATUSES`, `::_ENTRY_SCHEMA`, `::_SIGHTING_SCHEMA`.
- **Filing policy and trickle:** `filing_policy.py::is_fileable`, `::resolve_target`, `::MIN_UNREMEDIATED_SIGHTINGS`; `nightly.py::run_nightly`, `::evaluate_census_step`, `::_default_census_launcher`, `::select_scored_records`.
- **Sampling and config:** `sampling.py::_score_and_find_first_turn` (private, reached by `nightly.py` and `census.py`), `::score_signals`, `::ScoredRecord`, `::stratified_sample`; `trickle_state.py::project_state_dir`; `digest.py::DIGEST_INSTRUMENT_VERSION`; `config.py::Census`, `::NoveltySpike`, `::TrickleCensusCaps`; `session_runner.py::CLASSIFIER`, `::READ_ONLY_EXPLORER`.
- **Outside `scripts/legibility/`:** `orchestrator/src/orchestrator/agents/code_quality.py::guidance` (stdlib-only; the package `__init__`s are empty). The fused-memory tools `resolve_ticket` (statuses `created|combined|failed|refused`), `get_task`, `get_statuses` and `search_tasks` are nested in `fused-memory/src/fused_memory/server/tools.py`. `scripts/tests/test_design_invariants_consistency.py::_HEADING_RE`. The `review/briefing.yaml` `subprojects` keys (fused-memory, orchestrator, shared, dashboard, escalation, sampler). `.env` defines `OPENAI_API_KEY`. Tasks filed through the curator keep their metadata: the 23 `source=legibility_census` tasks.

Not present, queued as prerequisites:

- `find_tasks_by_metadata`: owned by `plans/task-metadata-lookup-prd.md`, a hard dependency of L4.
- `scripts/check_run_completion.py`: owned by `plans/completion-driven-triggers-prd.md`, a hard dependency of L8b.
- `scripts/legibility/check_census_report.py`: the decomposer commits a fail-loud stub with this PRD, because `submit_task` validates `before_done.script` at filing (precedent: `plans/confusion-reduction-prd.md` decision 10). L11 replaces the stub.

## 7. Out of scope

- Splitting `census.py`, or any heuristic 14 measurement of it (structural, contract §6).
- The nightly trickle's 2026-09-29/30 account-access failures, which are owned elsewhere.
- `find_tasks_by_metadata` and `check_run_completion.py` themselves.
- The other instruments' adoption of `shared.finding_key`.
- Healing the 230 legacy `invariant_violated` values.
- Re-wording the 1,001 pending candidates.
- Fixing any confusion cause (that is the filed tasks' work).
- `skills/census/SKILL.md`. It was corrected alongside this PRD and carries a "what changes when this lands" note.

## 8. Cross-PRD / seam ownership (G4)

| Seam / mechanism | Owner | This PRD's edge |
|---|---|---|
| `find_tasks_by_metadata(project_root, key, value)` | `plans/task-metadata-lookup-prd.md` (its L1) | consumes in L4 (§8 step 1 on `x_finding_key`, and the backfill on `source`), through the transport that PRD names, `census_trigger.py::post_mcp_tool_call`; hard dep. That PRD's §Cross-PRD row lists this PRD as a consumer keyed on `x_finding_run`; here that read happens inside `check_run_completion.py` (L8b), not in census code |
| `scripts/check_run_completion.py` (share, exit 0/75) | `plans/completion-driven-triggers-prd.md` | consumes in L8b by CLI; produces `x_finding_run` (L3) that it reads. A previous run that filed nothing makes (a) N/A (contract §11, D9) |
| `shared/src/shared/finding_key.py` | **this PRD (L3)**, named by contract §2 | the other instruments consume it; until it lands, `skills/hotspot-survey/scripts/findings_artefact.py::finding_key` is the interim reference that L3 must agree with |
| Path→area rule for paths under no member | the project's `/review-all` overlay (contract §3) | L3 applies it read-only |
| Other instruments' report records (homes per contract §5; basename patterns per each skill's report-format) | `/review`, `/hotspot-survey`, `/review-all` skills | L6 reads their latest JSON records' keys and lists their `run_id`s in `inputs_consumed` (contract §9) |
| `invariant_violated` population and slug check | **this PRD (L9)**, resolving the circular claim in `plans/confusion-reduction-prd.md` §10 ↔ `docs/legibility/design-invariants.md` §Census seam | decomposer adds a pointer to both (not owned by this seat) |
| Trigger semantics (§6, §7.4, §7.5, §8.5 of the parent PRD) | **this PRD (L8a, L8b)** supersedes them | decomposer adds a supersession line to `plans/confusion-reduction-prd.md` |
| `nightly.py` | parent PRD ε (landed) | L1 adds the ledger write; L8b changes the `decide_for_project` call |
| Quality-definition renderer | `orchestrator/src/orchestrator/agents/code_quality.py::guidance` | consumed read-only (L2, L10, and L9, which slices `## Definition` with `code_quality.py::section`) through a `__file__`-relative src path, as `census.py::_SHARED_SRC` does |

## 9. Decomposition plan

Every leaf edits `census.py` except L8a and L11, so the `census.py` editors form one chain: L1 → L8a → L2 → L3 → L4 → L5 → L6 → L7 → L10 → L8b. L8a sits in the chain because it shares `config.py` with L1, L7 and L8b, and `census_trigger.py` with L1 and L8b. L9's `census.py` edit is wiring only, because C1's "enforced on write" needs the slug list at the merger; L9 enters the chain at L3 through the existing L3 ← L9, and between L9 and L2 there is no dependency, so the scheduler's file lock serialises them. L9 ← L1 serialises `nightly.py`, and L3 ← L9 serialises `codebook.py`. L8b's `nightly.py` edit follows L9 transitively. Externals: L4 ← task-metadata-lookup's `find_tasks_by_metadata` leaf; L8b ← completion-driven-triggers' `check_run_completion.py` leaf. G ← L8b, L9, L11. The graph is acyclic.

L1–L10 are intermediates whose user-observable completion is G's check of a live report, the C-as-integration-gate pattern. Each also names its own observable below. Tests drive public seams only.

- **L1: run identity, processed-session ledger, resumable window** (normal; no deps; ~1,000 LOC).
  - **Files:** new `scripts/legibility/session_ledger.py`; `census.py` (`main`, `default_batch_source`, `advance_census_state`, `census_report_sections`, `run_census` gains `run_id`/`as_of_sha`/`since` values); `nightly.py` (ledger write after coding); `sampling.py` (promote `_score_and_find_first_turn` to a public name; both outside callers move to it, heuristic 13); `census_trigger.py::load_census_state`; `config.py::Census` (`ledger_retention_days`); `scripts/tests/`.
  - **Behaviour:** C2, plus C3's record-then-render split. The run writes the JSON record, renders the markdown from it, and commits both under one basename, with the `## Method` block. It retires `_census_window_dates`.
  - **Tests:** boundary rows 1–4.
  - **Consumers:** L8a (watermark), L3 (`run_id`), operator.
  - **Signal:** after the next nightly run, `session_ledger.py stats --project-id dark_factory` prints non-zero `trickle` rows written by the real trickle.
- **L8a: trigger, relative novelty and watermark floor** (normal; deps L1; ~350 LOC).
  - **Files:** `census_trigger.py` (`evaluate`, `CensusConfig`, `decide_for_project`), `config.py::NoveltySpike`, `docs/legibility/legibility.yaml`, tests.
  - **Behaviour:** C7 (b) and the floor. The tasks-landed condition and today's max-interval condition are untouched here; L8b retires both.
  - **Tests:** row 13, minus completion. A replay test over a fixture codebook with a recorded daily-count series.
  - **Consumer:** `nightly.py::evaluate_census_step`.
  - **Signal:** `census_trigger.py evaluate --project-root /home/leo/src/dark-factory` prints the `novelty-spike: N within 72h (baseline median M, ×2)` line and a `floor:` line naming its anchor. The anchor is `since session watermark` once a census has run after L1. Until then it is the named fallback `since last_census_at (no watermark yet)`, because only a real census run writes the watermark (G6).
- **L2: verifier verdict contract and judging by reference** (normal; deps L8a by chain; ~500 LOC).
  - **Files:** new `scripts/legibility/verdict.py`; `census.py` (`_verify_prompt`, `_build_default_verify_fn`, promotion severity; delete `_VALID_ENTRY_SEVERITIES`); tests.
  - **Behaviour:** C4. Heuristic 12: a typed `Verdict` replaces dict reads. Heuristic 10: the severity enum is checked at parse time and again at `codebook.validate`, and both read one definition.
  - **Tests:** row 6, plus a golden prompt test asserting that the rendered quality block comes from `guidance()`.
  - **Consumers:** L3, L4 (route), ticket priority.
  - **Signal:** G (Findings rows carry severities other than `medium`, each with a reason).
- **L9: invariant slugs to the coder, slug check in the merger** (normal; deps L1 (`nightly.py`); ~350 LOC).
  - **Files:** new `scripts/legibility/invariants.py`; `coder.py::build_prompt` (slug list from the observed project's doc; absent doc → empty list, stated in the prompt); `codebook.py::apply_coding_record` (screens each record through `codebook.py::screen_invariant_slugs` after the `validate_coding_record` schema gate passes: an unknown slug is stored as absent, counted in a new stat, one WARNING per run naming the values; `validate_coding_record` and `validate` are unchanged); `census.py` (wiring only: `run_census` reads `invariants.read_slugs` once, `mine_to_saturation` forwards `invariant_slugs` to `coder.code_digests`, and the mining-record merge passes it to `codebook.apply_coding_record`, with one WARNING per run); `nightly.py` journal summary line; `scripts/tests/test_design_invariants_consistency.py` imports the reader.
  - **Definition as the minting test (R9):** `coder.py::build_prompt` is the one prompt both the trickle and the miner use, so L9 (the leaf that already edits it) adds the `## Definition` section of the normative doc, sliced by `orchestrator/src/orchestrator/agents/code_quality.py::section(text, '## Definition')` over `code_quality.py::NORMATIVE_DOC`, never restated, as the test a confusion must meet before it is minted as a candidate. The prompt carries no heuristic and asks for no tag. Cost: the whole section, 138 words, about 200 tokens per miner and trickle call. Rationale: a heuristic tag is a judgement about code the coder has not read, and the primary tag is part of the finding key, so pre-tagging would churn keys across the trickle→census boundary.
  - **Tests:** rows 15 and 17; reify-form headings.
  - **Consumers:** the report's invariant counts (L10); the design-invariants §Census seam guard-task rule (Leo).
  - **Signal:** the next trickle journal shows `invariant slugs: N valid, M rejected`.
- **L3: finding keys on codebook records and tickets** (normal; deps L2, L9; ~900 LOC).
  - **Files:** new `shared/src/shared/finding_key.py` and its tests in `shared/tests/`; new `scripts/legibility/finding_area.py`; `codebook.py` (the whole C1 schema delta, with `pattern` support); `census.py` (stamping at promotion and rejection, appending this run's id to `last_seen_runs` on every entry sighted this run, `build_task_payloads` metadata and priority, the record's findings with every §1 field, `SECTION_FINDINGS`); tests.
  - **Area:** comes from contract §3's path→area rule (C5), applied to the verifier's anchor or remediation path: the briefing member directory first, then the project's `/review-all` overlay rule, then `repo`. It is never the path's first segment taken by itself.
  - **Behaviour:** C1, C5, contract §1 fields in the report. Heuristic 11: one key function for every instrument.
  - **Tests:** row 5; a key-stability test (reworded statement → same key; renamed symbol → new key, recorded in `supersedes`); agreement with `skills/hotspot-survey/scripts/findings_artefact.py::finding_key` on a fixture L3 commits (contract §2 parity test); area cases (member path, overlay-mapped `scripts/legibility/x.py` → `shared`, unmapped → `repo`).
  - **Consumers:** L4, L6, L8b, and `/review`/`/hotspot-survey` through contract §9.
  - **Signal:** G (Findings rows with `fk-` keys; the filed task's metadata carries `x_finding_key`).
- **L4: dedup protocol and ticket→task back-link** (normal; deps L3, external `find_tasks_by_metadata`; ~700 LOC).
  - **Files:** new `scripts/legibility/disposition.py`; `census.py` (filing loop, in-run `resolve_ticket` under D7, `SECTION_STRUCTURAL`, Filed Tasks with task ids and steps); tests.
  - **Behaviour:** contract §8 steps 1–3 in order.
    - **Step 1, key lookup:** `find_tasks_by_metadata` on `x_finding_key`; a list value matches by membership. A `cancelled` task without `x_acceptance_reason` counts as absent, and the run continues to step 2. The census does not use §8's interim forensic `json_each` read: it is unattended code, so the hard dependency stands.
    - **Step 2:** `search_tasks` at ≥ 0.6, with a `get_task` read-back.
    - **Step 3:** the curator. `filed_tickets` back-links (C1). Structural findings are not filed (D6). A one-shot idempotent backfill links the 23 historical `source=legibility_census` tasks to entries by exact `[legibility census] <title>`.
  - **Tests:** rows 7–8, combined-resolution, refused-resolution. INV-9: the codebook keeps pointers, the task store is the home.
  - **Consumer:** L5.
  - **Signal:** G (Filed Tasks names task ids and the deciding step).
- **L5: outcome feedback** (normal; deps L4; ~600 LOC).
  - **Files:** `disposition.py`; `census.py` (`SECTION_DISPOSITIONS`, re-verify queue under D8 through L2's verifier); tests.
  - **Behaviour:** Every run reads `get_statuses` for back-linked tasks. `done` queues re-verification (D8). `cancelled` is read with `get_task`. With `metadata.x_acceptance_reason` (contract §7, §8 1b) it gives `status: accepted` and `accepted_by`. Without it the task was abandoned, not accepted, so the entry returns to `open` and is eligible to re-file. Contract §7 dispositions are rendered from status plus `filed_tickets`. INV-3: a status snapshot never marks `fixed` without the re-verify.
  - **Tests:** rows 9–10.
  - **Consumers:** report, L6 (a fixed entry re-observed after its fix is surfaced, not screened away).
  - **Signal:** G (`## Dispositions` lists previous tickets → tasks → outcome).
- **L6: verifier pre-screen** (normal; deps L5; ~350 LOC).
  - **Files:** new `scripts/legibility/prescreen.py`; `census.py` (`run_census` between `_novel_clusters` and verify; post-verify key match before promotion; `SECTION_SCREENED`; `extra.screened`; `inputs_consumed`); tests.
  - **Behaviour:**
    - **Title rule:** a cluster whose normalised title equals the title of any entry or candidate, of any disposition, attaches its sighting through the merger's existing routing. No verify call is made. This replaces most of today's `DroppedVerdict` spend.
    - **Session rule:** a cluster whose session already carries a sighting on any record is not verified. It merges as `pending` and goes to L7's queue. This is a second net behind L1's ledger (heuristic 10, *redundantly*).
    - **Key rule (after verify, before promotion):** a verdict whose `(area, anchor)` equals an existing entry's attaches to that entry instead of promoting a duplicate. The entry keeps its standing key. The verifier's tags are recorded on the sighting, and a primary-tag disagreement is reported.
    - **Cross-instrument rule (contract §9):** a minted key, or `(area, anchor)`, that equals a finding in the latest `/review`, `/hotspot-survey` or `/review-all` JSON record is re-observed. The census promotes it with that key and its `supersedes`, rather than minting a new one. The finding is recorded as re-observed, and those reports' `run_id`s go into `inputs_consumed`.
    - Every attach sets `attached_by` and appends to `last_seen_runs`.
  - **Tests:** row 11, plus the 10-03 shape (pending twice) minus similarity, which L7 covers.
  - **Consumers:** L7, L10 (`re_observed`).
  - **Signal:** G (`## Screened` with per-rule counts).
- **L7: similarity index and pending-candidate adjudication queue** (normal; deps L6; ~1,100 LOC).
  - **Files:** new `scripts/legibility/similarity.py` (embed, cache, nearest, greedy single-link clusters); `config.py::Census` (`adjudication` block); `census.py` (a pre-verify similarity screen as L6's fourth rule; the queue after verify; `SECTION_ADJUDICATION`); tests with a fake embedder seam.
  - **Behaviour:** D3 and D4. A pending candidate at or above `attach_min_similarity` to an entry is promoted onto it (`attached_by.similarity`). Pending clusters are verified by representative, up to the cap, and the verdict applies to every member (`adjudicated_with`). The singleton gate counts cluster sessions. An embedding failure skips the stage, files one info escalation and is reported (INV-4, INV-11).
  - **Tests:** row 12.
  - **Consumers:** the singleton filing gate, report.
  - **Signal:** `similarity.py queue --project-root /home/leo/src/dark-factory` (embedding spend only, no verify) prints the queue depth and the top clusters over the live codebook. G (`## Adjudication` with an audit line per attach).
- **L10: structured synthesis** (normal; deps L7; ~700 LOC).
  - **Files:** new `scripts/legibility/synthesis.py` (schema, validation, deterministic renderer); `census.py` (`_synthesis_prompt` input: this run's verified, screened and attached findings plus counts, never prior reports; the guidance block; apply `phase_refinements` and `corrections`); tests.
  - **Behaviour:** C6. Heuristic 12: prose is rendered from data. The report's per-slug invariant counts come from L9's sightings. Reconcile task 5692's interim `census.py::SYNTHESIS_NOT_APPLIED_NOTICE` and `::SYNTHESIS_PROPOSALS_HEADING`: once `corrections` and `phase_refinements` are applied, the notice must name what was applied, not deny it.
  - **Tests:** row 14, plus a golden render.
  - **Consumer:** operator.
  - **Signal:** G (`## Synthesis` has "New" and "Re-observed" sub-blocks; a standing count; the matrix reflects refinements).
- **L8b: completion condition, retire tasks-landed** (normal; deps L10 by chain, L3, external `check_run_completion.py`; ~450 LOC).
  - **Files:** `census_trigger.py`, `config.py::Census`, `legibility.yaml`, `nightly.py::evaluate_census_step`, `census.py` (`main`'s `decide_for_project` call; stop writing `last_census_done_count`); tests.
  - **Behaviour:** C7 (a), D9, and R8: delete the `max_interval_days` condition from `census_trigger.py::evaluate`, the field from `config.py::Census`, and the key from the `census` block of `docs/legibility/legibility.yaml`. `census_trigger.py::CensusConfig.from_mapping` stops reading it: a mapping that still carries `max_interval_days` (a not-yet-migrated project's `legibility.yaml`) is accepted, the key is ignored as deprecated, and one WARNING per evaluation names it, never a validation failure. It also adds C7 (a)'s consecutive-error streak in host-local trickle state: one `escalate_info` at 3 consecutive error-exit N/As, reset by the next successful evaluation (exit 0 or 75; G7 `storm-escape-required`).
  - **Tests:** row 13 completion arms; a config carrying the deprecated key evaluates with no max-interval reason line and one deprecation WARNING; three consecutive error exits file exactly one escalation, while "no previous run id" and "filed nothing" never count toward the streak, and an exit 0 or 75 resets it.
  - **Consumer:** `nightly.py::evaluate_census_step`.
  - **Signal:** `census_trigger.py evaluate` prints `completion: <share> of <run_id> (threshold 0.70)` or a named N/A.
- **L11: report conformance checker** (normal; no deps; ~250 LOC).
  - **Files:** `scripts/legibility/check_census_report.py` (replaces the stub); tests.
  - **Behaviour:** Parses the newest JSON record and its rendered report against C3: the seven §5 keys plus `extra`, every §1 field on each finding, the section set, and the shared basename. Exit 0 when the report conforms. Exit 1 when the newest report predates the header, or lacks a key; the message names the report and the missing keys (INV-2, INV-13: "no conforming report yet" is distinct from "keys missing").
  - **Tests:** row 16.
  - **Consumer:** G.
  - **Signal:** run on main, it exits 1 naming the 10-03 report's missing `## Method`.
- **G: live census conformance gate** (deterministic predicate; deps L8b, L9, L11; `metadata.milestone: {mode: delayed, after_secs: 1209600}`, a fixed fourteen days (§10); `before_done: {kind: predicate, script: scripts/legibility/check_census_report.py, args: ["--project-root", "/home/leo/src/dark-factory"], timeout_secs: 120}`).
  - **On pass:** done. The batch is proven on a report written by the real automatic census.
  - **On fail:** a born-at-L2 `milestone_check_failed` naming the missing keys. Nothing is cancelled. When the cause is that no census has fired since the batch landed ("no conforming report yet", L11), the remedy is a forced census by hand (`skills/census/SKILL.md`), then `resume`.

Sizing (overlay bands): L1, L3, L7 are near the top of the band. L6, L8a, L9 and L11 sit at the floor and above it. G is deterministic and exempt. No leaf declares more than 10 files.

## 10. G6: premise validity

| Asserted number / claim | Basis | Status |
|---|---|---|
| relative spike `multiple` 2, `baseline_days` 30 | replay 2026-08-15..10-03: fires 3/50 days vs 48/50 today (§2, measured 2026-10-04) | provisional; L8a's replay test pins the rule, not the count |
| completion threshold 0.70 | contract §11 | given |
| `ledger_retention_days` 30 | the current `census.py::_DEFAULT_CENSUS_LOOKBACK_DAYS` | given |
| adjudication cap 10/run | spend bound, a choice; the backlog drain time is reported, not asserted | choice |
| similarity 0.90 / 0.85 | none | **provisional (D4)**; no leaf signal asserts a precision |
| embedding backlog ≈ 75 k tokens | 1,001 × ~300 chars (§2) | estimate, reported as `embedding_calls` |
| G's fourteen-day delay | a fixed choice. With the backstop dropped (R8), neither trigger condition has a predictable cadence: the relative spike replays at 3 fires in 50 days (§2, about one per 17 days), and completion (a) is N/A until a run has filed tasks carrying `x_finding_run` (L3) | choice; a fail with "no conforming report yet" is answered by a forced census, not a longer delay |
| "6 of 12 already catalogued" | 09-30 study | motivates L6; no signal asserts a screened ratio |

Each end-to-end signal is produced by its own leaf or an upstream one. G is the only leaf that reads a live report, and it depends on every producer.

## 11. G7: advisory walk (`docs/legibility/design-invariants.md`)

- `structured-facts-at-failure` (INV-2) and `no-silent-fail-soft` (INV-11): every normalisation in C4, every rejected slug and every skipped stage is counted, rendered and named with its values.
- `storm-escape-required` (INV-4): the embedding failure path skips the stage once per run instead of retrying per candidate. The completion arm's error-exit N/A carries a consecutive-error streak that escalates once at 3 and resets on the next successful evaluation (C7 (a), L8b). Without it, a broken completion arm would silently reduce cadence to novelty-only (hit found at decompose, 2026-10-05; redesigned, not waived).
- `no-lockstep-duplication` (INV-5): one key function, one heading reader, one severity enum.
- `holds-owned-and-bounded` (INV-7): the ledger is pruned to the retention window. The embedding cache drops vectors for records that are no longer pending and are not entries. The adjudication cap bounds per-run spend.
- `one-fact-one-home` (INV-9): codebook `status`/`filed_tickets` are pointers, reconciled from the task store every run (D8).
- `corroborate-before-acting` (INV-3): `fixed` needs re-verification. A similarity attach acts only inside the codebook, is reversible because the candidate record is retained, and is audited per attach. Not a hit.
- `readers-prove-their-producer` (INV-13): the ledger reader shows `ledger_created_this_run`. Completion is N/A, not "0 landed", when the previous run has no `run_id`. G reads a report the real producer wrote.

No unwaived hit.

## 12. Capability manifest (draft; the decomposer commits `plans/census-incremental-prd.capability-manifest.{md,yaml}`)

- **L1:** `session_ledger.stats` reads rows written by `nightly.py::run_nightly` (producer: L1 itself, wired on the production path); `sampling` public scorer (producer: L1).
- **L8a:** `census_trigger.py::evaluate` (grep-wired from `decide_for_project` → `nightly.py::evaluate_census_step`); watermark (producer: L1, upstream).
- **L9:** `orchestrator/src/orchestrator/agents/code_quality.py::section` and `::NORMATIVE_DOC` (exist; consumed read-only through the same src path as L2).
- **L2:** `orchestrator/src/orchestrator/agents/code_quality.py::guidance` (exists, wired in `orchestrator/src/orchestrator/agents/roles.py`); verdict fields consumed at promotion (producer: L2).
- **L3:** `run_id` (L1, upstream); verdict anchor/tags (L2, upstream); slug field (L9, upstream); briefing `subprojects` (exists).
- **L4:** `find_tasks_by_metadata` (external producer, upstream dep); `resolve_ticket`/`search_tasks`/`get_task` (exist).
- **L5:** `filed_tickets` (L4, upstream); verifier (L2, upstream); `get_statuses` (exists, wired through `census_trigger.py::default_status_fetcher`).
- **L6:** keys (L3, upstream).
- **L7:** `OPENAI_API_KEY` in the unit env (exists); prescreen hook (L6, upstream).
- **L10:** screened/attached sets (L6, L7, upstream); corrections op (exists, `codebook.py::apply_coding_record`).
- **L8b:** `check_run_completion.py` (external producer, upstream dep); `x_finding_run` on tasks (L3, upstream; field-population: written by `build_task_payloads` on the production filing path).
- **G:** the section keys of C3 (L1, L3–L7, L10 upstream).

## 13. Open questions

Questions for Leo (design-level, defaults stated; work proceeds on the default):

- **Q3** Accept D6? Under it, structural findings stop being auto-filed, a policy change from today.

These are settled by the amended contract and are no longer open:

- The `## Method` block shape (§5).
- The single key implementation (§2: `shared/src/shared/finding_key.py`, this PRD's L3).
- The acceptance-reason field (§7, §8: `x_acceptance_reason`).
- Embedding the `guidance()` block for code-assembled prompts (§10).
- A run that files no tasks (§11: it files no chain; for the census, (a) is then N/A, D9).
- The calendar backstop (former Q2): dropped (R8, Leo, 2026-10-05).
- What the coder receives (former Q1): the `## Definition` section only, never the heuristics (R9, D5, L9).

Tactical (decided at implementation):

- **T1** Embedding model. Default `text-embedding-3-small`; recorded in the cache key and the report.
- **T2** How the verifier's prompt ranks tags so the primary tag is stable across runs. If a re-verdict picks a different primary tag at the same anchor, L6's key rule matches on `(area, anchor)` and keeps the standing key, so no second key is minted. `supersedes` is reserved for anchor moves (contract §2).
- **T3** Whether synthesis runs tool-less. The 10-03 synthesis records reads of `runs.db` and task artefacts, but `census.py::census_stage_specs` gives it `session_runner.py::CLASSIFIER` (`disallowed_tools=("*",)`). L10 confirms which is in force, because C6 needs no tools.
- **T4** On a `combined` ticket resolution, whether to append `x_finding_key` to the target task's metadata. Default: no, because a metadata list write replaces. The entry's back-link carries it.
- **T5** Reach `code_quality.py` by a src path, or move the renderer into `shared`. Default: the src path, as `_SHARED_SRC` does.
