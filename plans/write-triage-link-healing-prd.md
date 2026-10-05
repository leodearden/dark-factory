# PRD: write-triage link healing — detach misfiled children, surface hidden ones, flag corrections, complete half-links

**Project:** dark-factory (fused-memory). **Status:** active, 2026-10-01. **Type:** extension of `docs/prds/memory-write-path-convergence.md` (D4: attaches are non-destructive and re-parentable) and the pointer target of `plans/write-triage-flip-readiness-prd.md` §11 D17 ("healing misfiles is out of scope here, pending a separate ruling"). **Approach:** B+H — the executor writes to the memory store under a corroborate-then-verify contract that two plan sources share.
**Origin:** Leo's rulings of 2026-10-01 at this PRD's gates: *clean up now, build the nightly sweep at the flip*; *complete half-links by verdict*; *flag CORRECTS-rated children contested*. Settled earlier and binding: a correct attach is any note about the same claim (09-29); `contested` means the child says the parent's claim is wrong, outdated or different, and nothing adjudicates it (09-30; convergence D3).
**Code anchors** verified against main `4dfdccc2ad` (2026-10-01). Main moves fast — cite-by-symbol; re-locate at implementation time.

## 1. Goal (G1 consumer + user-observable surface)

The consumers are the grouped readers — `fused-memory/src/fused_memory/server/grouped_read.py` behind every `search` and `get_memory_by_id`, and `orchestrator/src/orchestrator/agents/briefing.py::render_memory_results` behind every agent briefing's `# Context` block — and the agents who write links. When this PRD lands:

- No child is grouped under a parent about a different subject. A misfiled amendment's 240-character digest stops appearing under the wrong parent, and a query matching the misfiled child returns the child as its own hit instead of the wrong parent in its slot.
- No extension hides as a sighting: a sighting that adds to its parent becomes an amendment whose digest renders under the parent.
- No correction hides or folds silently: a child that says its parent is wrong, outdated or different carries `x_contested`, the state the write-time contract already gives a contradicting write, so it is never suppressed (`grouped_read.py::_carve_outs_allow_suppression`).
- No inert half-link: a record carrying `parent_id` with a kind other than `sighting`/`amendment` either becomes the child its verdict says it is, or stands alone.
- An agent writing a link the read path will ignore is told so in the `add_memory` ack, and the tool description says which kind to use.
- After the flip (task 3169), the same holds for links triage writes: a nightly sweep adjudicates every link not yet judged and heals under a capped, reversible run.

Operator surface: `fused-memory/scripts/link_heal.py status` prints the last runs, their counts, sources and whether they wrote, and says "no link-heal run has written here" when that is so (INV-13). Every heal is undoable by run id.

## 2. Background — evidence (pointer, INV-5)

Bundles (gitignored, main checkout): `data/write-triage-jev-trial-2026-09-29/`, `data/write-triage-stronger-models-2026-09-30/`. The verdicts this PRD acts on are committed with it (§6). Live census 2026-10-01 ~15:50Z.

- **The links.** `fused_dark_factory`: 324 records carry `parent_id` (189 amendment, 76 sighting, 3 `correction`, 56 no kind). `fused_reify`: 41 (8 amendment, 9 sighting, 9 `correction`, 7 no kind, 8 other: 3 `child_amendment`, 4 `peer`, 1 `extension`). 4 dark_factory links dangle (parent gone), 1 is a two-level chain, 1 carries `x_contested`. Every one was written by hand: `write_triage.enabled` is false, and `write_triage.py::declares_attach_keys` force-stores any write that sets `parent_id`, `kind` or `x_contested`, so a hand link never meets the judge. Hand links are the only live producer of `parent_id` today.
- **Ratings.** 359 links — the whole 2026-09-29 freeze (322 + 41) minus 4 whose parents were already gone — were blind-rated under the rater brief. Two Opus raters agree at κ 0.877 on belongs-vs-misfile (355/359 raw) and κ 0.887 on CORRECTS-vs-not; a third rated the 20 disputes. Majority words: EXTENDS 199, CORRECTS 111, SUBSUMED 32, RELATED 15, SAME 2. Rule: one pair per link (the child against its own parent); misfile = RELATED or UNRELATED; majority per `score_models.py::truth_l2` (recorded in the corpus summary).
- **Misfiles.** 15/359 = 4.2% (Wilson 95% [2.5%, 6.8%]); the first rater flagged 16, the second 18, both 15. The 5.7% (8/140) figure on task 3169 is the 2026-09-29 sample of the same population.
- **Hidden text.** Of 85 sightings, 66 EXTENDS, 14 CORRECTS, 2 SUBSUMED, 3 RELATED. A sighting's text is only counted on a grouped read (`grouped_read.py::_read_grouped_document`), so 80 notes that add to or correct their parent are invisible unless a query matches them directly.
- **Corrections unflagged.** 111 links are rated CORRECTS; 1 carries `x_contested`. 64 are plain amendments (a digest under the parent they contradict), 14 sightings (hidden), 31 half-links, 1 `peer`, 1 already contested.
- **Half-links.** `grouped_read.py::_parent_id_in_meta` and `write_triage.py::_canonical_id_of` treat a record as a child only when `kind ∈ CHILD_KINDS` (`amendment`, `sighting`). The 83 other linked records (80 of them rated) read as ordinary peers, yet `MemoryService._count_children` (the delete gate), `tools.py::consolidate_memories` (which moves a folded parent's children by `parent_id`) and `_read_grouped_document`'s total-children probe all count them by `parent_id` alone. A post-flip triage attach to a half-link folds under it as under any peer; completing a half-link that already has children would turn them into grandchildren that never fold (H1 guards it; none has children today).
- **Rate.** Flag off, dark_factory gained ~12 links/day 09-02→09-24 but 19 in 09-25→10-01 (~2.7/day) on steady writes (~105/day); reify ~1/day. At 4.2% that is ~0.15 new misfiles a day. After the flip the rate is writes × attach fraction, which π (task 6151) measures; ~235 Mem0 writes/day reached the two stores 09-29→10-01 (636 in ~2.7 days).
- **Detection is unmeasured.** The judge's veto (`distinct` on a slate of one) catches 4–10 of the 15 misfiles at precision 0.13–0.44 for every arm measured, Opus and gpt-6.1-sol included — not a detector. The rater question ("does CHILD's central claim belong under PARENT's?") has never been run as a detector by a model that is not one of the raters. Cosine cannot stand in: task 4916 records ANN AUC 0.25 (inverted) between true duplicates and hard negatives.
- **Nothing heals today.** No stage, sweep or audit reads whether a link is right; recon Stage 1 sees children as ordinary memories (`reconciliation/stages/memory_consolidator.py::_format_memories` renders no `parent_id` or `kind`).

## 3. Sketch of approach

One executor, two plan sources, one deferred sweep.

- **α** builds the heal executor: a plan of per-link actions, each corroborated against the live store immediately before one atomic metadata write through the running server's `update_memory`, verified immediately after, and recorded in a ledger that drives undo and idempotence. Its plan source is the committed hand-link verdict corpus.
- **β** is the one-off cleanup sitting: plan from the corpus, Leo's go-ahead on the dry run, apply.
- **γ** closes the hand-link producer side: the `add_memory` description states the kind rule, and the ack names an inert link.
- **δ** builds the adjudicator (the rater question through the Claude CLI) as the second plan source, plus an evaluation over rated pairs; **δm** runs it on the 359 rated links; **Γ_A** decides whether it is fit to drive the sweep.
- After the flip: **φ** proves the flag is really on; **ε** builds the nightly sweep and **η** installs it in report mode; **ζ** re-scores the adjudicator on λ's triage pairs; **θ**, a week after install, decides whether the sweep may write; **ξ** switches it on.

Healing changes placement and label only. It never re-parents to a new target, never deletes a record, never decides who is right in a contest, and never touches a child that already carries `x_contested`.

## 4. Contracts (H)

### H1 — heal plan and executor

> **A heal is one atomic metadata write on one record, made only if the record and its parent still match what the verdict judged when re-read just before the write, and verified just after; every heal is ledgered with its exact pre-image and undoable.**

**Action table** (one home: `fused_memory/maintenance/link_heal.py`; `CHILD_KINDS` and `CONTESTED_METADATA_KEY` imported from `grouped_read.py`, never respelled). Rows are evaluated top to bottom; the first match wins.

| record / basis | verdict | action |
|---|---|---|
| corpus row with `rated_text_matches_live: false` | any | none — `stale_rating` |
| `x_contested` truthy | any | none — `contested_reported` |
| parent absent (an id read in the child's project returns `None`) | — | detach (`basis_source = deterministic`) |
| parent found only in another known project's collection | — | none — `cross_project_reported` |
| parent itself carries `parent_id` (chain) | — | none — `chain_reported` |
| any | RELATED, UNRELATED | detach |
| any | UNCLEAR | none — `unclear` |
| kind `sighting` | EXTENDS | relabel → `amendment` |
| kind `sighting` | CORRECTS | relabel → `amendment` + `x_contested: true` |
| kind `sighting` | SAME, SUBSUMED | none |
| kind `amendment` | CORRECTS | flag `x_contested: true` |
| kind `amendment` | SAME, SUBSUMED, EXTENDS | none |
| kind `peer` (an author-declared relation) | any belongs | none — `peer_reported` |
| other kind ∉ CHILD_KINDS (half-link) with children | any belongs | none — `has_children` |
| other kind ∉ CHILD_KINDS (half-link) | EXTENDS | complete → `amendment` |
| other kind ∉ CHILD_KINDS (half-link) | CORRECTS | complete → `amendment` + `x_contested: true` |
| other kind ∉ CHILD_KINDS (half-link) | SAME, SUBSUMED | complete → `sighting` |

A "parent absent" is a `None` from an id read, never an exception: a timeout or transport error fails the action, it never detaches.

**Writes.** Each action is exactly one call, through the running server's `update_memory` tool over `shared/mcp_post.py::post_mcp_tool_call`, with `agent_id = link-heal-<run8>` (8 hex) and `_causation_id = <run_id>`, so `server/mem0_update_authz.py::resolve_mem0_update_authorization` and the write journal apply unchanged. A detach is delete-only: `metadata_delete_keys=['parent_id']` plus `'kind'` when the kind is in CHILD_KINDS (an agent-invented kind such as `correction` or `extension` stays; it is census-only under `memory_metadata.enforce_kind_registry: false`) — the `delete_payload` route. A relabel, flag or completion is patch-only (`metadata_patch={'kind': …, 'x_contested': True}` as the row says) — the `set_payload` route. A patch and a delete are never combined in one call (that is `MemoryService.update_memory`'s read-modify-overwrite route), and `None` is never patched (`parent_id: None` stores a null). A reply carrying `error_type` is a `failed` write. The journal `reason` is a pointer under 200 characters: `link-heal r=<run8> a=<action> prev_parent=<uuid> prev_kind=<k>`. A run that cannot reach the server refuses to start (INV-11).

**Authorization.** `mem0_update.metadata_patch_allowed_agent_prefixes` gains `link-heal-` beside the schema defaults `recon-stage-` and `curator-` (the block is commented out in `config/config.yaml` today; α writes it with all three). The config comment names this PRD and task α (INV-12).

**Corroborate before, verify after (INV-3).** Immediately before each write the executor re-reads the child, and the parent where the row needs it, from the live store — never from the corpus or the ledger. It requires:
- the child exists, and its `parent_id`, `kind` and `x_contested` equal the plan's pre-image;
- the sha256 of its full stored text equals the basis's `child_sha256`;
- the parent exists with text hash `parent_sha256`, except for a dangling detach, where the parent must still be absent;
- for a half-link completion, the record still has no children (`count_memories_by_metadata({'parent_id': id}) == 0`).

Any mismatch skips the action as `skipped_stale` and names the field that differed. Immediately after the write it re-reads the record and requires the post-image; anything else is `failed`. `update_memory` has no compare-and-set, so a concurrent writer that lands between the re-read and the write — in practice only a `consolidate_memories` re-parent of the same child — is not excluded, only narrowed to one round trip (the house precedent is `fused-memory/scripts/amend_stale_resume_cwd_records.py`). A service-side precondition on `update_memory` was considered and not adopted for v1: the overwrite route's own read leg is not atomic with its write either, and a raced detach is reversible. After a fold, the child's old verdict is stale, and the new (child, canonical) pair is adjudicated afresh.

**Ledger.** It is the one home of heal history and adjudicator verdicts: SQLite `link_heal.db` under the fused-memory data dir, with AUTOINCREMENT keys so that `sqlite_sequence` proves a producer (INV-13).
- `runs(run_id, source ∈ {corpus, adjudicator, undo}, writes bool, started_at, finished_at, counts_json, plan_sha256)`
- `adjudications(project_id, child_id, parent_id, child_sha256, parent_sha256, verdict, reason, model, run_id, at)`. This table holds adjudicator verdicts only. Corpus verdicts stay in the corpus file and are referenced by row key, never copied (INV-9).
- `actions(run_id, project_id, child_id, action, pre_image_json, post_image_json, basis_source, basis_key, state ∈ {planned, applied, skipped_stale, skipped_cap, failed, undone}, detail, at)`

A pair counts as judged when the corpus or the ledger holds a verdict for it at its current text hashes.

**Caps and storm escapes (INV-4).**
- Per unattended run, at most `link_heal.max_actions_per_run` actions are applied, oldest-planned first. The rest stay `planned` and are disclosed as `skipped_cap`, to be drained on later runs.
- When pending planned actions exceed `max_actions_per_run × link_heal.backlog_multiplier`, the run escalates once and keeps draining.
- `link_heal.write_failure_streak` consecutive failed writes stop the run with a non-zero exit.
- An attended apply (β) may exceed the cap only with `--approved-plan-sha <sha256>` equal to the plan file the operator reviewed.
- δ adds the adjudicator's escapes (H2).
- Each escape condition files through `fused-memory/src/fused_memory/middleware/_folded_escalation.py::file_folded_escalation` into dark-factory's queue under its own anchor: `link-heal-backlog`, `link-heal-write-failure`, `link-heal-misfile-share`, `link-heal-corrects-share`, `link-heal-adjudicator`. Each anchor is registered with `fused-memory/tests/test_folded_escalation.py::TestNoTwoFilersShareAnAnchor`.
- A run that does not write evaluates every escape and records it as `would_escape` without filing.

**Disclosure (INV-11).** Every run reports:
- `links_total`, `examined`, `adjudicated`, `unexamined`, `planned`;
- `applied`, `skipped_stale`, `skipped_cap`, `failed`;
- `stale_rating`, `contested_reported`, `chain_reported`, `peer_reported`, `has_children`, `cross_project_reported`, `unclear`;
- `would_escape`;
- each cap that bit.

A partial run is never reported as clean.

**Undo.** `link_heal.py undo --run <id>` writes back exactly each `applied` action's pre-image after corroborating that the record still shows the post-image. Keys the pre-image lacked are removed with a delete-only call. Keys it had are restored with a patch-only call. A record needing both gets two ledgered writes, delete first, each corroborated. `None` is never patched. A re-attached `parent_id` passes through the existing liveness check in `MemoryService._apply_memory_metadata_validation`. An `undone` action suppresses re-planning of the same action for that pair at the same text hashes; a changed text or parent re-opens it. The suppression lives in the `actions` row and is owned by the undo run (INV-12).

**One run at a time.** An exclusive lock enforces it, and the lock records its holder's pid and start time so a crashed run's lock is reclaimed rather than held forever (INV-6).

### H2 — adjudicator

> **One question, one scale, one home: the rater brief.**

- `fused_memory/maintenance/link_adjudicator.py::adjudicate_links(pairs, *, model, shard_size) -> list[LinkVerdict]`. It is pure in (child text, parent text) and returns one verdict per input pair or a counted failure, never a silent default.
- Instructions are rendered at call time from the committed `fused-memory/calibration/write_triage_rater_brief.md`, never transcribed (INV-5).
- Output is schema-bound: the seven words as an enum, plus a short reason. The belongs, misfile and CORRECTS mappings live in code beside the enum. A test mirrors the enum against the brief file's word list (INV-10: the live artifact, not a pinned string).
- Texts are capped at 4,000 characters with the brief's truncation marker, the cap the 359 ratings used (`link_heal.field_chars`, default 4000).
- The route is `shared/cli_invoke.py::invoke_with_cap_retry` with an `output_schema`, no tools, no MCP servers and a neutral cwd (`shared/neutral_cwd.py::neutral_cli_cwd`). This is the `fused-memory/src/fused_memory/middleware/path_scope_adjudicator.py::PathScopeAdjudicator` shape: Claude over OAuth, with the model set by `link_heal.adjudicator_model`.
- Pairs go in shards of `link_heal.shard_size` (default 40). A shard whose output omits or duplicates a pair id fails as a whole and is counted. A streak of failed shards (`shared/storm_counter.py::StormCounter`) escalates under `link-heal-adjudicator`.
- The adjudicator never sees ratings, kinds, `x_contested`, or which source produced the pair.
- Escapes for adjudicator-sourced plans: if more than `link_heal.misfile_share_ceiling` (0.25) of at least 20 adjudicated pairs come back misfile, or more than `link_heal.corrects_share_ceiling` (0.60) come back CORRECTS, the run writes nothing and escalates. Neither a prompt regression nor a model change may mass-detach or mass-flag. Measured shares on the 359: misfile 0.042, CORRECTS 0.31.

### H3 — measurement

> **The adjudicator is scored against blind majority verdicts before it drives anything, on hand links (δm) and again on triage pairs (ζ).**

- `fused-memory/scripts/eval_link_adjudicator.py --corpus <path> --arms <models>` runs each arm over every rated pair whose texts are still the rated texts. A hand-link pair is read live by id and checked against its hashes and `rated_text_matches_live`. A λ pair is read from π's `write_triage_pairs_to_rate.jsonl`. Excluded pairs are counted.
- From (arm verdict, majority verdict) it computes the confusion counts directly:
  - `misfile_recall` = arm-misfile ÷ majority-misfile;
  - `false_detach_rate` = arm-misfile ÷ majority-belongs;
  - `corrects_recall` = arm-CORRECTS ÷ majority-CORRECTS;
  - `false_corrects_rate` = arm-CORRECTS ÷ majority SAME/EXTENDS/SUBSUMED;
  - `parse_failure_rate`.
  
  A failed pair counts as a miss for both recalls. Each figure carries Wilson 95% bounds and its numerator and denominator.
- `kind_agreement`, over sightings and half-links, is reported beside its always-`amendment` baseline. It is a sanity metric, not a gate input.
- The ι scorer (`score_write_triage_pairs.py::score_pairs`, task 6147) is not used. Its metrics are rates over a judge arm's attaches, and these are a detector's confusion counts. Nothing is defined twice.
- Output: `fused-memory/calibration/link_adjudicator_report.json` (and `link_adjudicator_report_triage.json` for λ). It holds a row per arm, `population {n_pairs, n_misfile, n_corrects, n_belongs, excluded}`, and `selection`, which copies one arm's metrics verbatim. The selection rule is threshold-free: highest `misfile_recall`, then lowest `false_detach_rate`, then lowest `false_corrects_rate`, then lower cost. The gates alone judge bounds.

### Boundary-test sketch

All rows run in-process against the fused-memory test harness that already exercises payload writes (`fused-memory/tests/test_consolidate_memories_tool.py`, `fused-memory/tests/test_mem0_qdrant_integration.py`) behind a fake MCP transport — never against the live project collections (unregistered projects are rejected by `tools.py::_known_project_gate`, and merge-verify may run on another host). β is the live smoke.

| scenario | preconditions | postconditions |
|---|---|---|
| detach a misfiled amendment | amendment under P; verdict RELATED; hashes match | no `parent_id`, no `kind`; a grouped `search` matching it returns it as its own hit; P's grouped document no longer digests it; one `applied` row; one journal row from `link-heal-*` |
| detach a misfiled `extension`-kind half-link | verdict RELATED | `parent_id` removed, `kind: extension` kept |
| relabel an extension-sighting | sighting under P; EXTENDS | kind `amendment`; P's grouped document lists its digest; `render_memory_results` shows it nested |
| flag a correcting amendment | amendment under P; CORRECTS | `x_contested: true`; it is never suppressed in a grouped read |
| complete a half-link as sighting | no kind, parent P; SAME | kind `sighting`; P's `sighting_count` +1 |
| half-link with a child | no kind, has a child | no write; `has_children` +1 |
| stale pre-image | child text edited after rating | `skipped_stale` naming `child_sha256`; no write |
| folded since rated | child re-parented to C | `skipped_stale` naming `parent_id` |
| contested child rated RELATED | `x_contested` true | no write; `contested_reported` +1 |
| dangling parent | parent deleted | detach without a verdict; `basis_source = deterministic` |
| parent read times out | id read raises | `failed`, never a detach |
| cap drains oldest first | 40 planned, cap 25 | 25 oldest applied; 15 `skipped_cap`; next run applies them |
| misfile-share escape | 30 adjudicated, 12 misfile | zero writes; one escalation under `link-heal-misfile-share` |
| report mode | `writes` false | zero writes; `would_escape` recorded, nothing filed |
| undo a no-kind completion | half-link completed to `sighting` | delete-only call removes `kind`; record matches its pre-image; no `kind: null` |
| undo a detach | amendment detached | patch restores `parent_id` and `kind`; a second undo is a no-op; the sweep does not re-plan it |
| server unreachable | fused-memory down | refuses to start, non-zero exit, no writing run row |
| prefix not admitted | `link-heal-` absent from allow-list | first write returns `Mem0UpdateNotAuthorized`; `failed` counted; streak stops the run |
| status before any run | empty ledger | "no link-heal run has written here", distinct from a run with 0 actions |

## 5. Resolved design decisions

- **D1 Scope** (Leo, 2026-10-01). Built now:
  - the executor;
  - the cleanup sitting;
  - the guidance and ack;
  - the adjudicator, measured and gated.

  The nightly sweep is built after the flip (3169) and Γ_A. Reason: with the flag off, about 0.15 new misfiles a day do not pay for a sweep, but the store's accumulated damage does pay for one cleanup. The sweep adjudicates every link not yet judged, so building it at the flip loses nothing permanent.
- **D2 Half-links are completed by verdict** (Leo, 2026-10-01), per H1. They then fold as `docs/prds/memory-metadata-vocabulary.md` V2 defines a child. A query that matches one still returns its full text as a matched child. The exceptions are `peer` (an author-declared relation, reported) and a half-link with children (reported, so no grandchild is created).
- **D3 CORRECTS ⇒ contested** (Leo, 2026-10-01). The rater brief's CORRECTS is word for word the 09-30 definition of `contested`, and the write-time contract already turns a contradicting write into an amendment flagged `x_contested` (`tools.py::add_memory`).
  - Healing gives every belongs-rated CORRECTS child that state (109 today).
  - This is detection, not adjudication: nothing decides whether parent or child is right.
  - A child that already carries `x_contested` is never touched; one rated misfile is reported, not detached.
  - The contested consumers (χ 6150's briefing render; 5277 A2) are unchanged and see a larger population.
  - The adjudicator's CORRECTS agreement is gated before the sweep may flag.
- **D4 No re-parenting.** A detached child is never wrong, only unconsolidated. Re-placing it would need a target chosen by something shown to tell subjects apart, but cosine is inverted (AUC 0.25) and the judge places at the same rate it misfiles at. The `update_memory` re-parent path checks only liveness. `t_high` (0.887) was calibrated for "same fact" on write-time slates, not for re-homing a child. A detached child is a standalone record, which recon Stage 1 may later stamp as a topic peer under Option C. No third grouping mechanism is added.
- **D5 Labels move toward visibility, except where Leo ruled a fold.**
  - Sighting → amendment adds a digest.
  - Flagging contested makes a child a full hit.
  - An amendment rated SAME stays an amendment.
  - Only half-links rated SAME or SUBSUMED fold into a sighting count (D2).
- **D6 The adjudicator uses the Claude CLI over OAuth, not the write-triage judge arm.**
  - The judge's question is the wrong detector (§2).
  - `write_triage_judge.py::_call_llm` hardcodes the judge prompt and a 128-token cap.
  - The CLI route needs no ω (6148).
  - δm measures Opus and Sonnet, and the report's `selection` picks between them.
- **D7 Writes go through the server tool under a dedicated `link-heal-` prefix.** Their journal attribution stays separate from human curation (`curator-`) and from recon (`recon-stage-*` writes are counted as recon stage ops by `tools.py::_extract_causation`).
- **D8 Verdicts have one home each.** Human ratings live in the committed corpus, and adjudicator verdicts in the ledger. Records gain no heal stamp, and the journal row carries a pointer.
- **D9 Gates are predicate tasks running `scripts/check_write_triage_readiness_gate.py`** (`--report`, `--require NAME PATH OP VALUE`, `--subcheck`; the exit code is the contract), as in the flip-readiness PRD's D3.
  - Each gate's thresholds live in its `before_done.args`. This document records the filing values, and a re-base edits the args and appends a dated line to §9.
  - Every gate a later task depends on also carries a `delivered_checks` entry of `kind: script` that re-runs its predicate. A cancelled dependency counts as satisfied (`orchestrator/src/orchestrator/scheduler.py::_deps_satisfied`), so cancelling a gate must not open what it holds.
- **D10 Measurement runs and installs are human-run operational leaves** (β, δm, η, ζ). It is unverified whether a dispatched task agent can spawn the Claude CLI inside its sandbox. The blind protocol must keep ratings from the arms. Installing a systemd timer and observing a night's run are operator acts.

## 6. Pre-conditions / substrate (G3)

Verified on main `4dfdccc2ad`:

**Memory update path**
- `server/tools.py::update_memory`:
  - `metadata_patch` sets or changes keys, and `metadata_delete_keys` removes them (`services/memory_service.py::MemoryService._apply_metadata_delta`).
  - A patch alone takes the `set_payload` route and a delete alone the `delete_payload` route. Each is an in-place Qdrant payload write that keeps the point id, `created_at`, embedding and content (`backends/mem0_client.py::Mem0Backend.set_payload` / `delete_payload`).
  - Each call journals a `write_ops` row with the patch, the delete keys and `reason[:200]`. There is no metadata before-image, hence H1's ledger.
- `server/mem0_update_authz.py::resolve_mem0_update_authorization` is keyed on `mem0_update.metadata_patch_allowed_agent_prefixes` (`config/schema.py::Mem0UpdateConfig`, default `['recon-stage-', 'curator-']`, hot-reloadable). A delete-only call is gated the same way.
- `tools.py::_extract_causation` carries `_causation_id`. `shared/mcp_post.py::post_mcp_tool_call` is the script-to-server transport.
- Parent liveness on a changed `parent_id` comes from task 3197. It is warn-only under the shipped `memory_metadata.enforce: false`.
- `MemoryService._count_children` and `count_memories_by_metadata` count children by `parent_id`. `MemoryService.get_memory_by_id(project_id, memory_id) -> dict | None` is a raw point read.

**Grouped reads and triage**
- `server/grouped_read.py`:
  - `CHILD_KINDS`, `_parent_id_in_meta`, `is_contested_child`, `CONTESTED_METADATA_KEY` (`x_contested`);
  - `_carve_outs_allow_suppression` (a contested child is never suppressed);
  - `_DIGEST_CHARS` 240.
- `orchestrator/.../briefing.py::render_memory_results` renders `amendments` and `matched_children` as nested bullets. It does not render counts or the `contested` marker.
- `write_triage.py::_canonical_id_of` hoists CHILD_KINDS only. `declares_attach_keys` force-stores any write carrying `parent_id`, `kind` or `x_contested`.

**Adjudication**
- `shared/cli_invoke.py::invoke_with_cap_retry` runs over OAuth and strips `ANTHROPIC_API_KEY`.
- Precedents: `reconciliation/judge.py::Judge._call_judge_cli`, `middleware/path_scope_adjudicator.py::PathScopeAdjudicator`, and the standalone script `fused-memory/scripts/probe_schema_max_turns.py`.

**Storm escapes and gates**
- `shared/storm_counter.py::StormCounter`.
- `middleware/_folded_escalation.py::file_folded_escalation(project_root, …)` with `test_folded_escalation.py::TestNoTwoFilersShareAnAnchor`.
- `scripts/check_write_triage_readiness_gate.py`.
- `docs/task-authoring.md` provides the `delivered_checks` `kind: script` (run against the working checkout) and the §6 delayed milestone `{"mode": "delayed", "after_secs": …}`, which counts from when the task's own dependencies are satisfied.

**Timer and storage**
- The nightly-timer convention is in `OPERATIONS.md` §12: a wrapper `.sh`, a oneshot `.service`, a `.timer` with `Persistent=true`, and `scripts/install-<job>-timer.sh`. Slots 03:00–05:30 are taken; 02:00–02:30 is free.
- Qdrant payloads are flat: `parent_id`, `kind`, `x_contested` and the text key `data` (`backends/mem0_client.py::_MEM0_TEXT_KEY`) are top-level.
- There is no payload index on them, so a full scroll per project (~0.5 s at 30–37k points) is the candidate query.

Committed with this PRD. The bundles are gitignored, and task worktrees cannot see them:
- `fused-memory/calibration/hand_link_verdicts.jsonl` + `.summary.json` cover the 359 rated links. Each row carries ids, project, kind at rating, per-rater words, the majority word, the primary reason, the sha256 of each live text at export, and `rated_text_matches_live` (true for all 359 at export, 2026-10-01).
  - The summary carries the majority rule, the cap function, the content key, and the source files with their hashes.
  - Applying H1 to this corpus plans 15 detaches, 66 sighting relabels, 14 sighting relabels+flag, 64 amendment flags, and 74 half-link completions (32 amendment, 31 amendment+flag, 11 sighting).
  - It also reports 4 `peer`, 1 contested, and 121 rows needing no action.
  - The live store adds 4 dangling detaches, 1 chain reported, and 3 unrated links. That makes 237 actions.
- `fused-memory/calibration/write_triage_rater_brief.md` is byte-identical to the 2026-09-29 brief (sha256 `7f161b0e…`), at the path λ (6152) names. λ's "commit it verbatim" step will find it present and unchanged.

**Queued as prerequisites**

In this batch:
- the executor, ledger, `link_heal.*` caps and the `link-heal-` prefix (α);
- the ack (γ);
- the adjudicator, its escapes and the eval (δ);
- the sweep and its timer files (ε).

Out of batch:
- λ's corpus (6152 ← 6151), for ζ;
- the flip (3169), behind φ.

Deliberately not used:
- `consolidation_flood_control.py::rank_and_cap`. It does not exist (task 5240 is pending) and ranks by consolidation weights; H1 shares only the knob semantics.
- `write_triage_judge.py::_call_llm` (D6).
- ι's `score_pairs` (H3).
- `audit_duplicate_memories.py`'s delete-only apply path.

## 7. Out of scope

- Re-parenting to a new target (D4).
- Deleting any record.
- Deciding a contest.
- Flipping `write_triage.enabled` or setting `write_triage.judge_*`.
- Editing tasks 5804/5805/5809, 5807/5808/5810, 6147–6152 or 3169. This batch adds dependencies from its own tasks onto 6152 and 3169 only.
- A write-time placement check in triage, i.e. asking the rater question before attaching. δm's report is the evidence such a change would need; whether to build one is the flip-readiness PRD owner's call.
- The two-level chain (1 link) and the 4 `peer` half-links: reported, not repaired.
- Adjacent hazards that destroy children rather than misplace them (owners in §8):
  - Recon Stage 1 may delete a child as a duplicate (`reconciliation/prompts/stage1.py`: "Keep most recent / highest confidence. Delete duplicate.").
  - `audit_duplicate_memories.py` is child-blind, and its `--apply` could delete a sighting as a near-duplicate loser once 3136 schedules it.
- Role-prompt guidance. `orchestrator/src/orchestrator/agents/roles.py::_MEMORY_INSTRUCTIONS` says nothing about `parent_id`; agents learn it from the `add_memory` description, which γ amends. Task 3131 (the convergence PRD's leaf ε) owns `_MEMORY_INSTRUCTIONS` and is gated on 3169.

## 8. Cross-PRD relationship / seam ownership (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/memory-write-path-convergence.md` C1/D3/D4 | consumes | child schema (`parent_id` + kind ∈ amendment\|sighting; contradiction = amendment + `x_contested`); "a wrong attach is re-parentable" | convergence PRD; this PRD acts within D4 and detects per D3 | landed (3127–3129) |
| `docs/prds/memory-metadata-vocabulary.md` V2/V3 | consumes | grouping key strictly `parent_id`; no operation silently orphans a child | vocabulary PRD; no new grouping mechanism (D4) | landed (3129, 3197) |
| `plans/write-triage-flip-readiness-prd.md` λ (6152) | consumes | `write_triage_pair_verdicts.jsonl` + π's pairs file feed ζ; the rater brief path | flip PRD; this PRD commits the brief byte-identical first | pending; ζ depends on 6152 |
| same, ι (6147) | none | not consumed (H3) | flip PRD | pending |
| same, D17 | produces | "healing misfiles … pending a separate ruling" → this PRD | this PRD | — |
| same, χ (6150) / task 5277 A2 | produces for | ~109 more contested children for the briefing render and the contested consumer | χ / 5277; must-not-touch their code | pending |
| task 3169 (flip) | consumes | `write_triage.enabled: true` on main | Leo at 3169; φ proves it | blocked |
| `plans/memory-auto-consolidation-prd.md` 5240 `rank_and_cap` | none | knob semantics only | auto-consolidation PRD | pending; not consumed |
| task 3136 (audit timer) | adjacent hazard | `audit_duplicate_memories.py --apply` must not delete a record carrying `parent_id` | 3136 — decompose appends the hazard to its details | pending |
| recon Stage 1 prompt | adjacent hazard | Stage 1 may delete a child as a duplicate | decompose searches tasks for an owner and files one if none | — |
| task 3131 (convergence leaf ε) | none | `_MEMORY_INSTRUCTIONS` link guidance | 3131; γ covers the `add_memory` description only | pending, gated on 3169 |
| `OPERATIONS.md` §12 | produces | the nightly ladder row for the link-heal timer (02:15) | ε | — |

## 9. Decomposition plan

Dependencies:

| Task | Depends on |
|---|---|
| α | — |
| β | α |
| γ | — |
| δ | α |
| δm | δ |
| Γ_A | δm |
| φ | 3169 |
| ε | Γ_A, φ |
| η | ε |
| ζ | δ, 6152 |
| θ | η, ζ (delayed 7 days) |
| ξ | θ |

The graph is acyclic. Same-file serialization: α, δ and ε share `fused_memory/maintenance/link_heal.py`, `fused-memory/scripts/link_heal.py` and the config schema, and are chained by the edges above. γ alone touches `server/tools.py`.

- **α — heal executor, ledger, caps, prefix** (normal; fused-memory `maintenance/link_heal.py`, `scripts/link_heal.py`, `config/schema.py`, `config/reload.py`, `config/config.yaml`, tests).
  - Scope: H1 in full, with the corpus as plan source (`plan --from-corpus`, `apply`, `undo`, `status`). That covers corroboration, verification, the ledger, the cap and backlog and write-failure escapes, disclosure, the run lock, and the INV-13 status state.
  - Config leaves: `link_heal.max_actions_per_run` 25, `backlog_multiplier` 5, `write_failure_streak` 3. They are registered in `config/reload.py::RELOADABLE_FIELDS` so a config edit never reports `restart_required`; the scripts read config at each run.
  - The `link-heal-` prefix is added with its INV-12 comment.
  - Tests: boundary rows other than the adjudicator escapes, run on the harness named in H3.
  - *Cap basis (G6, provisional):* post-flip actions/day ≈ writes/day (~235) × attach fraction (π measures it; ≤ 0.8 assumed) × action share (misfile 0.04–0.09 + relabel ~0.03 + flag 0–0.3 depending on the judge's labels). That gives ≤ ~19 actions/day without flags. 25 is that with margin and is re-based at θ from the report-mode week.
  - *Signal:* `link_heal.py plan --from-corpus fused-memory/calibration/hand_link_verdicts.jsonl` against the live store prints per-action counts matching §6 (minus anything stale since export), writes a plan file with its sha256, and changes nothing. `status` prints "no link-heal run has written here".
- **β — cleanup sitting** (operational, human-run; deps α).
  - Plan from the corpus for both projects, plus the deterministic dangling detaches.
  - Present the counts, the 15 misfiles with their rater reasons, and the flagged corrections to Leo.
  - On his go-ahead, run `apply --approved-plan-sha`, then `status`.
  - Spot-check four heals with a grouped `search`: one detach, one sighting relabel, one flag, one half-link completion. Record the run id on the task.
  - *Signal:* the ledger holds a writing corpus run whose `applied` count equals the approved plan minus `skipped_stale`. A grouped `search` for a relabelled child's parent shows its digest, and a flagged child surfaces as its own hit.
- **γ — kind guidance and inert-link ack** (normal, `complexity: simple`; `server/tools.py` + tests).
  - The `add_memory` description says:
    - use `kind: amendment` when the note adds to its parent;
    - also set `x_contested: true` when the note says the parent is wrong, outdated or different;
    - use `kind: sighting` only for a pure restatement;
    - any other kind leaves `parent_id` inert.
  - A write carrying `parent_id` returns a typed ack field `link: {status: 'child' | 'inert', reason}`. The status is `child` iff the kind ∈ `grouped_read.CHILD_KINDS`. It is documented in the tool's return shape and tested as data, not prose.
  - Nothing is rejected, and what is stored is unchanged.
  - *Signal:* `add_memory` with `parent_id` and no kind returns `link.status == 'inert'`; with `kind: 'amendment'` it returns `'child'`.
- **δ — adjudicator, escapes and eval** (normal; deps α).
  - Files: `maintenance/link_adjudicator.py`, `scripts/eval_link_adjudicator.py`, and `scripts/link_heal.py`, which gains `--from-adjudicator`.
  - Scope: H2 and H3.
  - Config leaves: `link_heal.adjudicator_model`, `shard_size` 40, `field_chars` 4000, `misfile_share_ceiling` 0.25, `corrects_share_ceiling` 0.60.
  - Tests: the enum-vs-brief mirror test; fake-CLI tests for shard failure, duplicate ids and parse failure; the misfile-share escape row; one live-edge test on a single fixed pair, skipped without the CLI.
  - This is an intermediate task. It unlocks δm, ζ (the eval) and ε (the plan source).
  - *Signal for its consumers:* `eval_link_adjudicator.py --corpus <fixture> --arms fake` writes a report with the H3 keys, `population`, and `selection`.
- **δm — measure the adjudicator on the hand links** (operational, human-run; deps δ).
  - Runs `eval_link_adjudicator.py --corpus fused-memory/calibration/hand_link_verdicts.jsonl --arms opus,sonnet`.
  - Commits `fused-memory/calibration/link_adjudicator_report.json` (+ `.md`).
  - *Signal:* the committed report has a row per arm with the H3 confusion counts and Wilson bounds, a `population` block, and a `selection` block.
- **Γ_A — gate: adjudicator fit to drive the sweep** (deterministic, predicate; deps δm; `delivered_checks` re-runs the same predicate, D9).
  - Reads `--report r fused-memory/calibration/link_adjudicator_report.json`.
  - Requires:
    - `selection.misfile_recall >= 0.60`
    - `selection.false_detach_rate <= 0.02`
    - `selection.corrects_recall >= 0.60`
    - `selection.false_corrects_rate <= 0.10`
    - `selection.parse_failure_rate <= 0.05`
    - `population.n_pairs >= 300`
  - *Basis (G6, provisional):*
    - **Misfile recall and false detaches.** Rater against rater on this corpus, the first rater's 16 misfile flags include 15 the second shares, and the second's 18 include those 15: agreement 0.83–0.94, with 1–3 extra flags among ~344 belongs-rated links (0.003–0.009). Recall against the majority is 1.0 for both raters by construction, so it is not cited.
    - The best judge veto reaches 9/15 = 0.60 at precision ≈ 0.19, and 9/15 has a Wilson interval of [0.36, 0.80]. So `misfile_recall >= 0.60` asks for at least the judge's recall, and `false_detach_rate <= 0.02` (≈ 6 links) holds it to roughly twice the worse rater. Every detach is reversible.
    - **CORRECTS bounds.** They are D15's `contradiction_recall` and `false_contested_rate` bounds from the flip-readiness PRD for the judge. On these same 359 links, Sol and Opus measured 0.86–0.91 and 0.08–0.16 there.
    - **Parse failures.** 0.05 allows one failed shard in twenty.
    - **Caveat.** The raters and the Opus arm share a model family (Opus vs Sonnet κ 0.66 on 40 pairs), so a pass is weaker evidence than its numbers.
  - *On pass:* ε's dependency is satisfied; ε still waits on φ.
  - *On fail:* hold, owned by the gate's escalation. ε waits, and β is unaffected. Re-base per D9 or keep holding.
- **φ — gate: the flip is live** (deterministic, predicate; deps 3169; `delivered_checks` re-runs it, D9).
  - Uses the readiness gate script with a `--subcheck` that exits 0 iff the committed `fused-memory/config/config.yaml` has `write_triage.enabled: true`.
  - It exists so a cancelled 3169 cannot dispatch ε.
  - *On pass:* ε dispatches.
  - *On fail* (3169 closed without a flip): hold, and the escalation owns re-ruling ε.
- **ε — nightly sweep** (normal; deps Γ_A, φ).
  - Files: `maintenance/link_heal.py` (sweep), `scripts/fused-memory-link-heal.{sh,service,timer}`, `scripts/install-link-heal-timer.sh`, and the `OPERATIONS.md` §12 row at 02:15.
  - Each run:
    - fully scrolls the linked records of each project in `link_heal.projects`;
    - takes as candidates the pairs not judged at their current hashes;
    - adjudicates up to `link_heal.max_adjudications_per_run` (200) of them and discloses the rest as `unexamined`;
    - plans by H1 and writes only when `link_heal.writes` is true (ships false);
    - emits a metric series through `shared.memory_eval_metrics` under `fused-memory/data/memory-evals/link-heal/`, plus a rolling `summary.json` for θ with the last 7 days' `runs`, `writing_runs`, `would_escape`, `adjudication_failure_rate`, `actions_planned` and `skipped_cap`.
  - This is an intermediate task. It unlocks η.
  - *Signal for its consumer:* in the harness, a report-mode run over a seeded store writes a non-writing ledger run, the metrics artifact and `summary.json`. The backlog and cap rows pass.
- **η — install the sweep and observe a night** (operational, human-run; deps ε).
  - Run `scripts/install-link-heal-timer.sh`, confirm `systemctl --user list-timers` shows it, and wait for the first nightly run on the live store.
  - *Signal:* the ledger holds a non-writing adjudicator run from the timer with `adjudicated > 0`, and `status` shows it.
- **ζ — measure the adjudicator on λ's triage pairs** (operational, human-run; deps δ, 6152).
  - The same command over `write_triage_pair_verdicts.jsonl`, with texts from π's pairs file.
  - Only rated pairs are scored; sampled-out and seed pairs are counted as excluded.
  - Commits `fused-memory/calibration/link_adjudicator_report_triage.json`.
  - *Signal:* the committed report carries `population` and `selection` with the H3 keys.
- **θ — gate: the sweep may write** (deterministic, predicate; deps η, ζ; `metadata.milestone {"mode": "delayed", "after_secs": 604800}`; `delivered_checks` re-runs it, D9).
  - On the triage report it requires the six Γ_A bound names. Their values start equal to Γ_A's and are an independent home.
  - On `fused-memory/data/memory-evals/link-heal/summary.json` it requires `runs_7d >= 6`, `would_escape_7d <= 0`, and `adjudication_failure_rate_7d <= 0.05`.
  - *Basis:* the brief's "report-only first" week, allowing one missed night, with no escape that would have fired and at most one bad shard in twenty. Provisional (G6).
  - *On pass:* ξ dispatches. *On fail:* the sweep stays non-writing and the escalation owns the hold.
- **ξ — let the sweep write** (normal, `complexity: simple`; deps θ).
  - Set `link_heal.writes: true` in `fused-memory/config/config.yaml`.
  - Re-base `max_actions_per_run` from θ's measured `actions_planned`, if the report-mode week says so, with a dated line in this section.
  - *Signal:* the next nightly ledger run has `writes = true` and `applied > 0` or an explicit zero-plan count.

**G7 walk (advisory, author mode).**
- INV-3: H1 corroborates before and verifies after (α).
- INV-4: per-condition escapes with separate anchors (α, δ, ε).
- INV-5: the action table, CHILD_KINDS, the rater brief and the confusion counts each have one home (α, δ).
- INV-6: a stale run lock is reclaimed (α).
- INV-7: the report-mode hold is owned by θ's delayed milestone and its escalation; Γ_A's and φ's holds by their escalations.
- INV-8: the sweep runs in its own process with adjudication capped per run (ε).
- INV-9: D8.
- INV-10: the enum mirror test reads the live brief (δ).
- INV-11: disclosure counters, refuse-to-start without the server, and a typed ack (α, γ).
- INV-12: the allow-list entry and undo suppressions are owned (α).
- INV-13: `status`'s no-producer state, with `sqlite_sequence` on the ledger (α).

No waivers.

**Sizing (overlay bands).**
- α: ~800–1,200 LOC.
- δ: ~600–900.
- ε: ~400–700.
- γ and ξ: under 100 LOC (`complexity: simple`).
- β, δm, η, ζ: operational.
- Γ_A, φ, θ: deterministic.

**Threshold home and re-bases.** The filing-time values are above, and each value's home is its gate task's `before_done.args` (D9). Re-bases are appended here as dated lines.

## 10. Open questions (tactical)

1. **Ledger file location.** Suggested: beside `write_journal.db` under `reconciliation.data_dir`. Decide in α from how the CLI and timer resolve the fused-memory data dir.
2. **Corroboration read path.** The choice is a direct Qdrant read by id or `MemoryService.get_memory_by_id` through the server; either reads the live store. Decide in α by which returns exactly the `data` string the hashes were taken over.
3. **Which queue receives a reify-link escalation.** Suggested: dark-factory's queue (the fused-memory system's home), with the project named in the detail. Decide in α.
4. **Shard size.** 40 sits between the raters' ≤ 55 and a smaller failure blast radius. Re-measure in δm.
5. **φ's subcheck command.** It must load the YAML with the project's own interpreter (`uv run --project fused-memory …`) and must fail closed on a missing key. Write it at decompose and run it once against today's tree, where it must fail.
