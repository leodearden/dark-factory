# Recon replay of write-triage attaches (task 7161)

Can reconciliation Stage 1 repair write triage's attaches better than the
triage judge filed them, and for which error classes? Every figure below is
from `fused-memory/calibration/recon_triage_replay_scores.md`, which renders
`recon_triage_replay_scores.json`. This report restates only the headline
numbers. The per-class splits in §4 are computed from the same committed
files, by the method stated there.

## 1. Question and design

- **Sample** (`recon_triage_replay_sample.jsonl`, seed 7161). Drawn from the
  rubric v2.2 corpus (`write_triage_v2_soak_items.jsonl`, 673 items), from
  Decided items only.
  - Headline: 160 children. For each of dark_factory, reify, solar_challenge
    and solar_challenge_platform that is contested 20, amended 10 and
    restated 10.
  - Report-only: the 12 children of seat D's touched set that were not drawn
    (14 touched children are in the corpus).
  - Population figures reweight each (outcome, project) stratum to the soak's
    eligible writes (contested 473, amended 1,224, restated 125;
    `write_triage_v2_soak_strata.json`).
- **Context shape.** Per-project worklists of up to 8 children, 23 batches,
  each rendered as a Stage-1 payload:
  - the Stage 1 header;
  - the child and parent records through
    `stage1_payload_format.py::format_memories`, with the parent link from
    `child_memory_link.py::render_parent_link`;
  - the triage outcome and x_contested of each child;
  - the project_root directive.

  The texts are the ones the v2.1/v2.2 raters saw (basis current), as of the
  2026-10-19 soak.
- **Invocation.** Recon's own path, `cli_stage_runner.py::run_stage_via_cli`,
  under recon's usage gate.
  - System prompt: `STAGE1_SYSTEM_PROMPT` plus a replay addendum
    (`eval_recon_triage_replay.py::REPLAY_SYSTEM_PROMPT`).
  - MCP servers: fused-memory and jcodemunch only
    (`stages/base.py::recon_core_mcp_servers`), with no recon-report and no
    escalation server.
- **Read-only.** `REPLAY_DISALLOWED` is `STAGE3_DISALLOWED`, which denies
  every memory write, `propose_consolidation` and journal writes. No store
  was written.
- **Vocabulary.** keep, detach, contest_upheld, not_a_contest,
  relabel→amended, relabel→contested and superseded_parent. The actions each
  outcome allows are set by `recon_triage_replay_scoring.py::allowed_actions`.
  - The correct repair comes from `::expected_action`, which reads rubric
    v2.2's cells.
  - Every repair is priced by re-deriving the repaired item through the same
    rubric `derive` that priced the judge.
- **Truth and weights.** Rubric v2.2 (Opus+Opus; kappa home 0.94, loss 0.87,
  conflict 0.84). Harm weights are
  `score_write_triage_reader.py::PROVISIONAL_HARM_WEIGHTS`, which are
  **provisional**, not ratified.
- **Model.** sonnet (`reconciliation.agent_llm_model`, recon's quiescent
  default).
- **Run.** 2026-11-22, 10:18–10:24Z.
  - All 23 batches parsed on the first pass.
  - Cost $2.29 in all (CLI `total_cost_usd`, OAuth quota), about $0.013 per
    decision.
  - fused-memory `search` was clean before the pilot, after it, and after the
    run. The checks are in the decisions header's `tool_health`.
- **Recon used no tools.** In all 23 sessions its only tool call was
  StructuredOutput. The fused-memory and jcodemunch tools were offered as
  deferred tools, and none was loaded. This replay therefore measures recon
  judging from the payload alone, and every decision's `evidence` says so.

## 2. Recon beside the judge, same rows

| figure | n | sample | Wilson 95% | population | population 95% |
|---|---|---|---|---|---|
| judge wrong home, all attaches | 160 | 0.394 | [0.321, 0.471] | 0.462 | [0.358, 0.568] |
| judge wrong home, contested | 80 | 0.263 | [0.179, 0.368] | 0.260 | [0.167, 0.358] |
| recon misfile detection | 63 | 0.635 | [0.512, 0.743] | 0.442 | [0.291, 0.608] |
| recon wrong-detach rate | 97 | 0.124 | [0.072, 0.204] | 0.075 | [0.039, 0.118] |
| recon contest-resolution accuracy | 80 | 0.563 | [0.453, 0.666] | 0.557 | [0.448, 0.666] |
| recon action accuracy | 160 | 0.619 | [0.542, 0.690] | 0.581 | [0.483, 0.679] |

All 12 wrong detaches were correct contests.

**Harm per write** (provisional weights), paired over the 159 priced
children:

| figure | judge as filed | recon repaired | status quo | recon − judge |
|---|---|---|---|---|
| sample | 0.969 | 0.623 | 0.698 | −0.346 [−0.569, −0.126] |
| population | 0.817 | 0.571 | 0.466 | −0.246 [−0.394, −0.103] |

Recon's repairs lower harm against the judge as filed. On the population,
though, the repaired state (0.571) is still above the status quo of storing
every write flat (0.466).

By project:

| project | judge wrong home | misfile detection | wrong-detach | contest resolution | harm Δ, population |
|---|---|---|---|---|---|
| dark_factory | 0.425 | 8/17, 0.471 [0.262, 0.690] | 4/23, 0.174 [0.070, 0.371] | 11/20, 0.55 [0.342, 0.742] | −0.049 [−0.277, 0.156] |
| reify | 0.225 | 6/9, 0.667 [0.354, 0.879] | 0/31, 0 [0, 0.110] | 13/20, 0.65 [0.433, 0.819] | −0.159 [−0.441, 0.111] |
| solar_challenge | 0.425 | 10/17, 0.588 [0.360, 0.784] | 5/23, 0.217 [0.097, 0.419] | 9/20, 0.45 [0.258, 0.658] | −0.334 [−0.637, −0.056] |
| solar_challenge_platform | 0.500 | 16/20, 0.80 [0.584, 0.919] | 3/20, 0.15 [0.052, 0.360] | 12/20, 0.60 [0.387, 0.781] | −0.543 [−0.924, −0.151] |

The harm interval excludes zero only for the two solar projects. For
dark_factory and reify it straddles zero.

**Context, labelled by rubric.** None of these is on the same rows as above.
- **v1**, the soak correctness review (seat D,
  `data/write-triage-soak-2026-10-19/team/D/report.md`): judge misfile about
  0.31 over all attaches and 0.324 over contested ones.
- **v2.2 baseline** (`write_triage_v2_soak_baseline_v2_2.md`): M4 wrong home
  0.3487 [0.3136, 0.3856] over the sample of 671 decided items, and 0.5422
  [0.4767, 0.6048] on the population.

This sample's judge wrong-home rates (0.394 sample, 0.462 population) are
not the baseline's, because the allocation differs: half of this sample is
contested, against 473 of 671 in the baseline. The intervals overlap on both
figures.

**The link-heal adjudicator** (`link_adjudicator_report_triage.md`, task
6190) was run on 1,216 λ triage pairs, with majority verdicts under the v1
pair brief.

| arm | misfile_recall | false_detach_rate |
|---|---|---|
| adjudicator, sonnet | 0.788 [0.751, 0.820] | 0.158 [0.133, 0.187] |
| adjudicator, opus | 0.857 [0.825, 0.884] | 0.019 [0.011, 0.032] |
| recon (sonnet), this replay | 0.635 [0.512, 0.743] | 0.124 [0.072, 0.204] |

The populations differ in three ways:
- λ is a pre-soak pair set;
- its "false detach" is a detach of a pair the raters judged to belong;
- here it is a detach of any child without wrong_home.

The metrics have the same shape, though. Recon detects fewer misfiles than
the adjudicator's sonnet arm, and the false-detach intervals of those two
overlap. Against the opus arm recon is clearly worse on both counts:
- its wrong-detach lower bound (0.072) is above opus's upper bound (0.032);
- its detection upper bound (0.743) is below opus's lower bound (0.825).

## 3. The touched subset

Of the 14 corpus children touched live (`team/D/touched.json`), 4 were
deleted by their own writing agent minutes after the write. Recon changed the
other 10. Each live action below is read from the stamped and live metadata
in touched.json; the truth column is the expected repair. Rows from
`recon_triage_replay_scores.md`, "Touched children".

| entry | outcome | live action | truth | live right | replay | replay right |
|---|---|---|---|---|---|---|
| 024ca326 | contested | detached (cross-task completion) | contest_upheld | no | detach | no |
| 3df2714b | restated | detached (cross-task) | detach | yes | detach | yes |
| 58959f97 | contested | x_contested cleared, left under parent | detach | no | detach | yes |
| 5b5850ac | contested | re-parented to new index 01ebe34c | detach (from old parent) | yes | detach | yes |
| 81711e8a | contested | contest resolved "upheld", left under parent | detach | no | detach | yes |
| 911ee809 | restated | detached (cross-task completion) | detach | yes | detach | yes |
| de109f51 | contested | detached (cross-task completion) | detach | yes | detach | yes |
| e15ddb6f | contested | x_contested cleared, left under parent | detach | no | detach | yes |
| e511b7ae | contested | detached (cross-task completion) | contest_upheld | no | not_a_contest | no |
| f57607c7 | contested | contest resolved by rewriting parent | contest_upheld | yes | contest_upheld | yes |
| 092c0560 | contested | deleted by its writer | contest_upheld | — | not_a_contest | no |
| 61fe3b21 | contested | deleted by its writer | contest_upheld | — | detach | no |
| 95a00908 | restated | deleted by its writer | relabel→contested | — | detach | no |
| b6b9d450 | contested | deleted by its writer | detach | — | detach | yes |

Live recon touched 8 contested children, and its placement was right on 3:
5b5850ac, de109f51 and f57607c7. That reproduces seat D's 3 of 8 under v2.2
truth. The replay gets 6 of the same 8 right: 0.75 [0.409, 0.929], against
live 0.375 [0.137, 0.694].

The replay fixes the three placements live recon left wrong after settling
the contest's truth (58959f97, 81711e8a, e15ddb6f). Its two misses, 024ca326
and e511b7ae, are both cross-task orchestrator completion children carrying a
correct contest. Live recon detached both as well, so the replay shares the
blind spot of the task-id rule that seat D flagged.

Read this with care. n is 8, the children were selected (recon touched them
because another finding led there), and the replay asked a placement
question that live recon was never asked.

## 4. Recommendation per error class

**Method for the splits.** The 160 headline rows are grouped by the truth
cell or outcome named in each row. They are computed from the committed
sample, decisions, v2.2 items and verdicts through
`recon_triage_replay_scoring.py::derive_item`, `::expected_action` and
`::repaired_item`. Harm deltas are bootstrapped with
`write_triage_stats.py::stratified_bootstrap95` (10,000 resamples, seed
7161), with membership units over the priced rows. The all-rows run of this
method reproduces scores.md exactly: −0.3459 [−0.5692, −0.1258] on the sample
and −0.246 [−0.3935, −0.103] on the population.

Completion records are identified by `meta.source == 'orchestrator_completion'`
in the soak bundle's `correctness/attaches_soak.json`. Intent ids are mapped
to memory ids through `adjudications.jsonl` and the rating scratch's
`key.json` "extension". These splits are not in scores.json.

| class (truth) | n | recon repaired it | Wilson 95% | what it cost |
|---|---|---|---|---|
| wrong home, amended or restated child | 42 | 27 detached | 0.643 [0.492, 0.770] | 0 of 38 other amended/restated detached, [0, 0.092] |
| — amended only | 22 | 9 detached | 0.409 [0.233, 0.613] | 13 kept |
| — restated only | 20 | 18 detached | 0.900 [0.699, 0.972] | 2 relabelled amended |
| wrong home, contested child | 21 | 13 detached | 0.619 [0.409, 0.793] | 12 of 59 other contested detached, all correct contests: 0.203 [0.120, 0.323] |
| false contest (false_conflict) | 25 | 23 un-flagged by detach or not_a_contest | 0.920 [0.750, 0.978] | 25 of 53 correct contests un-flagged: 0.472 [0.344, 0.603] |
| hidden or digest conflict | 13 | 4 relabelled contested | 0.308 [0.127, 0.576] | 3 more relabelled amended, a partial repair |
| hidden detail | 2 | 1 relabelled amended | 0.5 [0.095, 0.906] | — |
| attached supersession | 4 | 1 superseded_parent | 0.25 [0.046, 0.699] | — |
| cross-task completion record | 19 | 15 right | 0.789 [0.567, 0.915] | 2 correct contests detached |

Recon's "this contest is wrong" calls (detach or not_a_contest on a
contested child) were right 23 of 48 times: 0.479 [0.345, 0.617].

| subset | n | harm Δ (recon − judge), sample | harm Δ, population |
|---|---|---|---|
| amended + restated | 80 | −0.588 [−0.744, −0.444] | −0.304 [−0.444, −0.185] |
| contested | 79 | −0.101 [−0.513, 0.317] | −0.078 [−0.488, 0.334] |
| completion records | 19 | −1.132 [−1.643, −0.583] | −1.076 [−1.882, −0.216] |
| everything else | 140 | −0.239 [−0.479, 0.000] | −0.213 [−0.365, −0.067] |

Repairs tallied by recon's action, as harm removed against harm added under
the provisional weights:

| action | harm removed | harm added | net |
|---|---|---|---|
| detach | 69 | 18 | +51 |
| not_a_contest | 20 | 34 | −14 |
| relabel→contested | 12 | 0 | +12 |
| relabel→amended | 5 | 0 | +5 |
| superseded_parent | 1 | 0 | +1 |

keep and contest_upheld moved no harm either way.

The verdicts follow the task's rule: a class is not recommended when its
interval does not beat the judge, or when its wrong-detach cost exceeds its
gain.

- **Wrong home on amended or restated children: recommended, as a detach-only
  repair over a worklist.** This is the one class whose interval clearly
  beats the judge: Δ −0.304 [−0.444, −0.185] on the population. Recon
  detached none of the 38 correctly homed amended and restated children.
  - The gain is mostly sightings: 18 of 20 restated misfiles were detached,
    against 9 of 22 amended ones.
  - The amended strata dominate the population (1,224 writes), which is why
    population misfile detection, 0.442, sits so far below the sample's
    0.635.
  - The opus link-heal adjudicator measured higher recall at lower
    false-detach on its own population. Before recon is handed this class,
    compare the two on these same rows.
- **Wrong home on contested children, and false contests: not recommended.**
  Recon un-flags nearly every false contest (23 of 25), but it also un-flags
  25 of 53 correct contests: 12 detached and 13 called not_a_contest.
  - Its contest-wrong calls are right less than half the time, and
    not_a_contest is net harmful (−14).
  - The contested harm interval, −0.078 [−0.488, 0.334], does not beat the
    judge.
  - This is the class where recon's live strength, settling truth from the
    task store and the code (seat D), should matter. Here it was not
    exercised, because recon made no tool call.
- **Hidden or digest conflict: not recommended.** Recon caught 4 of 13. Each
  of its 4 relabel→contested calls landed on a conflict-carrying child, so
  the calls are precise but rare.
- **Hidden detail: not recommended, unmeasured.** n is 2. Recon's 7
  relabel→amended calls added no harm, but only 1 landed on a hidden-detail
  child.
- **Attached supersession: not recommended.** Recon got 1 of 4.
- **Cross-task completion records: not a standing recon duty.**
  - Triage now force-stores completion records at write time
    (`server/write_triage.py::is_orchestrator_completion_write`, applied in
    `server/tools.py::add_memory`; task 6956), so only legacy attaches
    remain.
  - On those, recon's repairs beat the judge as filed: Δ −1.076
    [−1.882, −0.216] on the population. But it detaches them as a class
    (18 of 19 headline completion children), including the 2 whose contest
    is correct. It also un-flagged both touched completion children carrying
    a correct contest (024ca326 detached, e511b7ae called not_a_contest).
  - A one-off backlog repair could use detach for legacy completion attaches
    only if contested ones are excepted.

**What the prompt and payload would need** for even the recommended class:

- **The parent's text, not just its link.** Both payload paths now render
  parent links: the quiescent path through
  `stage1_payload_format.py::format_memories`, and the backlog path through
  `context_assembler.py::_format_memory_result`. Both use
  `child_memory_link.py::render_parent_link`. The backlog path gained it in
  task 6294 on 2026-10-30, after seat D found none there.
  - The link names only `parent_id` and `kind`, so recon would have to fetch
    each parent.
  - The replay showed child and parent side by side, and recon judged on
    that alone.
- **A deterministic child worklist.** The quiescent path reads
  `mem0.get_all(scope, limit=memory_limit)`, filtered to records newer than
  the watermark (`stages/memory_consolidator.py`). That is an unordered
  sample with no section listing new triage children. A per-project worklist
  of 8 cost about $0.10 per batch with sonnet.
- **x_contested visibility.** `render_parent_link` never renders
  x_contested, so in recon's real payload a contested child reads as a plain
  amendment. The replay stated the outcome and x_contested for each child.
- **A placement vocabulary.** `prompts/stage1.py::CHILD_MEMORY_AUTHORITY_RULE`
  says only never to delete a child to dedupe it. Nothing in the Stage 1
  prompt covers misfiles, detaching or contests; the replay addendum
  supplied all of that.
- **For contests, evidence before un-flagging.** Hypothesis: requiring a
  task-store or code read before any not_a_contest or contested detach
  would cut the correct contests un-flagged. This replay cannot test that,
  because recon never used a tool.

## 5. Limitations

- **Soak-time texts.** The pairs are as of 2026-10-19. Recon could have read
  live records, but made no tool call, so live drift never reached a
  decision.
- **No categories.** The rated texts carry none, so every record renders `(?)`
  (step-11 deviation).
- **Elision.** `format_memories` cuts record bodies at 2,000 characters,
  while the raters saw up to 4,000 (`write_triage_rating_rounds.py::TEXT_CAP`).
  Across the corpus, 68 current-basis children and 98 parents run past 2,000
  characters.
- **A single model arm**, sonnet. The opus backlog tier
  (`harness.py::is_backlog_size`) was not run. In the link-heal eval, opus
  and sonnet differed eightfold on false detach.
- **Payload-only judgement.** Whether tool use changes these rates is not
  measured.
- **Sample sizes.**
  - Each project has 40 children, 20 of them contested, so per-project
    intervals are wide.
  - Classes with n of 13 or less (hidden or digest conflict, hidden detail,
    supersession) cannot support a recommendation either way.
  - The touched comparison rests on 8 selected children.
- **Weights not ratified.** The harm figures move with the weights; the rate
  metrics do not.
- **One unpriced child.** Recon's not_a_contest repair of 9b472ee2 derives
  Incomplete, so harm is priced over 159 children.
- **Transcripts are temporary.** The per-ask transcripts live in this task
  worktree's gitignored `data/recon-triage-replay-7161/` (step-18 deviation:
  the main checkout's `data/` was not writable from the sandbox), and they
  last only as long as the worktree. The decisions, cost, plan hash and tool
  census are committed in `recon_triage_replay_decisions.jsonl`.

## Reproduce

From `fused-memory/`, with the worktree interpreter:

```
python scripts/eval_recon_triage_replay.py score
```

This re-scores from committed files alone and is byte-identical across runs.
Re-asking recon needs `run --asker stage --scratch DIR`, then
`collect --scratch DIR --tool-health JSON`, with
`CONFIG_PATH`, `PROJECT_ROOT` and `DASHBOARD_KNOWN_PROJECT_ROOTS` set as the
fused-memory service sets them.
