# PRD: legibility pipeline — one detector, one merge result, one alarm, typed run outcomes

**Project:** dark-factory (`scripts/legibility/`; consumes one `shared/` client). **Status:** authored 2026-10-10 by the S21 seat of the `/review-all` remediation program (`plans/review-all-program-dark_factory-2026-10-10.md`, wave 1, agent mode, approved as proposed under R21). **Type:** structural remediation of findings from run `review-all-dark_factory-20261008` and the prior rows it carries (`hotspot-survey-dark_factory-20261009`, `review-dark_factory-20261009`). **Approach:** B with contracts (§5) and per-leaf boundary tests (§6); no separate integration-gate leaf, because every leaf's signal is read off a production artefact (§9).

**Code anchors** were verified against main `ae3047ec3a` (2026-10-10); the findings were pinned at `6551b75aba`. Main moved `census.py`, `codebook.py`, `coder.py`, `nightly.py` (tasks 6400, 6401, 5692) and added `verdict.py`, `invariants.py` since the pin; every anchor below was re-read on `ae3047ec3a`. Citations are `path::symbol`. Re-locate them at implementation time.

The normative quality definition is `docs/code-quality.md`, cited by heuristic name and never restated. The findings contract is `docs/quality-findings-contract.md` (§6–§8). Design invariants are cited by slug from `docs/legibility/design-invariants.md`.

## 1. Goal (G1: consumer and user-observable surface)

The legibility pipeline (inventory → sampling → digest → coder → codebook merger → nightly trickle / census, plus trigger, state and probes) keeps its behaviour while each policy it holds gets exactly one home. After this batch:

- **The sampler and the digest agree on every confusion class.** A session's selection score is computed from the same per-record class hits the digest renders. On the live transcript population, the class-presence disagreement the finding measured (56 of 87 sessions with signal) is 0.
- **A codebook merge returns one frozen result.** The nightly, the census and `codebook.py apply` read the same `MergeResult`; a deletion-shaped coder record is skipped and reported in all three, never raised out of a paid census run.
- **One committer lands legibility writes.** Nightly and census commit through `unlanded.commit_only`, with the same lock-contention retry; a transient `index.lock` no longer quarantines a whole census.
- **A night's outcome is one typed value.** A capped night is recorded as `deferred` (not `productive`), `trickle-state.json` records whether the barren escalation fired, and the health probe reads that fact instead of re-deriving nightly's predicate.
- **Every legibility alarm goes through one poster** over the shared MCP client: it journals the cause before posting, refuses real posts under pytest, and never raises. `census_trigger.py` no longer carries an MCP transport.
- **The census trigger reads the validated config.** A malformed `census:` value is rejected once, at load, naming the field and value; the trigger, nightly and census see the same thresholds.
- **A census verdict resolves to a candidate by id, not by title,** and codebook lifecycle transitions are a table in the codebook's owner with source-state checks.
- **`run_census` and `run_nightly` are compositions of stage functions returning values**; `census.py` splits into layers that each pass heuristic 13, and its tests drive public seams.

**Consumers.** Leo and the operator, through the nightly journal (`legibility-trickle@.service`), `scripts/legibility/check_trickle_progress.py` / `check_trickle_health.py` output, the census report and `census_trigger.py evaluate`. The next `/review-all` run, which re-verifies the sixteen keys (§2). `plans/census-incremental-prd.md`'s remaining leaves and every later legibility change, whose cost this batch lowers (contract §6). The L2 watcher reads the alarm lines.

## 2. Finding keys

Every key below is closed by the named leaves (filed with `metadata.x_finding_key` listing it, `x_finding_run = "review-all-dark_factory-20261008"`). No key is dropped. None of the sixteen is in R23's weakened list; all were re-verified still present on `ae3047ec3a` (§7).

| Key | Sev | Anchor | Closed by |
|---|---|---|---|
| `fk-3a23d0a51e69` | high | `scripts/legibility/sampling.py::scan_transcript` | α |
| `fk-f1314d3b25eb` | medium | `scripts/legibility/codebook.py::apply_coding_record` | β |
| `fk-74142ca0f882` | medium | `scripts/legibility/census.py::_build_default_commit` | β |
| `fk-234e0a63dfd7` | low | `scripts/legibility/census.py` (bare sibling imports) | β |
| `fk-15f40965b6ba` | medium | `scripts/legibility/trickle_state.py::classify_run` | γ |
| `fk-988f412eced2` | low | `scripts/legibility/check_trickle_health.py::_should_escalate` | γ |
| `fk-0531ecaefba5` | medium | `scripts/legibility/trickle_state.py::load_state` | γ (trickle-state), ζ (census-state) |
| `fk-da87d818b1b8` | medium | `scripts/legibility/nightly.py::run_nightly` | δ |
| `fk-ef642b412580` | high | `scripts/legibility/nightly.py::post_escalation` | ε |
| `fk-12f0d08a6ccd` | medium | `scripts/legibility/census_trigger.py::post_mcp_envelope` | ε |
| `fk-e3cc7c87d73e` | medium | `scripts/legibility/census_trigger.py::CensusConfig` | ζ |
| `fk-3ea2b2df348a` | high | `scripts/legibility/census.py::_find_adjudicated_candidate` | η |
| `fk-a0af0f46c1dc` | medium | `scripts/legibility/census.py::_find_pending_candidate_id` | η |
| `fk-c9ab2773a5d9` | medium | `scripts/legibility/census.py::run_census` | θ (stages), ι (layers) |
| `fk-ee9a4bae4e83` | low | `scripts/legibility/census.py::_build_default_verify_fn` | ι |
| `fk-0af74b3620ec` | medium | `scripts/legibility/census_trigger.py` (prose share) | α, γ, δ, ε, θ (per-edit rehoming, §4 D10) |

## 3. Background

The review-all seat's summary of the slice (report `## Findings by area` → `shared/legibility`): the import graph is acyclic and several modules are healthy (inventory, the codebook validator, census_identity, census_window, session_ledger, unlanded, the trickle_state classifier). The cost sits in four places: policies held in two or three drifted copies (scorer and agent-class classifier, merge loop, scoped commit, census config reader, barren predicate, alarm posters); run outcomes spread over status strings and booleans; two very long orchestrating functions (`run_census` 774 lines / 26 parameters, `run_nightly` 461 lines / 7 hand-built `NightlyResult` literals on `ae3047ec3a`); and prose at 44–63% of the core modules. Each finding record carries its measurement and the "next change" it makes expensive; this PRD does not restate them.

`plans/census-incremental-prd.md` (tasks 6397–6409; 6397–6401 done, 6402–6408 pending in one `census.py` chain, 6409 its gate) is mid-flight on the same files. It explicitly left "splitting `census.py`" out of scope as a contract §6 structural finding; this PRD is where that work lands, sequenced after its chain (§8).

## 4. Resolved design decisions

- **D1 — Detector layer.** `digest.py`'s carrier/provenance code and pattern tables become two modules: `scripts/legibility/reingestion.py` (the provenance stratum task 5960 measured: `CODER_JUDGMENT_KEYS`, `is_coder_judgment_payload`, the `HARNESS_*` heading sets and markers, `is_harness_injected_turn`, `is_reingested_content`, `has_non_human_origin`; it imports only stdlib) and `scripts/legibility/detectors.py` (the `SignalClass` and `AgentClass` vocabularies, the per-class pattern tables, the per-record carrier scoping now in `digest.py::_dialogue_text_sources` / `_is_non_dialogue_carrier` / `_strip_decoy_lines`, `record_hits`, and the one `classify_agent_class`). `digest.py` and `sampling.py` import both downward. **The digest's tables are the reference**: they carry the deliberate exclusions (`DF_GUARD_PATTERNS` does not match bare `scope_violation` / `phantom-done`) and task 5685's ruling that the non-dialogue-carrier filter is consulted by every signal bucket. A sampler-only pattern (including the four guard tool names `sampling.py::_has_guard_tool_use` matches) is dropped by default; promoting one into the shared table changes what the digest renders, so it bumps `digest.py::DIGEST_INSTRUMENT_VERSION` and is listed in the task record. α is otherwise behaviour-preserving for the digest (golden digests unchanged; no version bump) and behaviour-changing for the sampler, which is the point.
- **D2 — One agent-class classifier over (path, records).** Precedence: the encoded-dir path shape (today's `sampling.py::_WORKTREE_DIR_MARKERS`) → markers in the first non-sidechain, non-meta user turn → markers anywhere in the raw carriers (today's `digest.py::classify_agent_class` haystack, kept as the last resort) → `interactive`. One marker table holds the union of both classifiers' markers. It classifies by injected markers, so it reads raw carriers, not the dialogue filter (unchanged from both today). The coder-session inheritance defect (task 5785) is not fixed here; 5785 is re-pointed at the one classifier (§8).
- **D3 — `MergeResult`.** `codebook.apply_coding_records(cb, records, *, invariant_slugs) -> MergeResult` is the only batch entry point; `apply_coding_record` stays the per-record step, private to the batch. `MergeResult` is frozen: the new codebook, `mutated: bool` (the rule now hand-summed in `nightly.py::run_nightly` over four stats keys, task 6357), per-op counts, `conflicts`, `skipped_deletions` (record id + reason, one per `NeverDeleteError`), `invalid_records`, the summed `SlugTally`, and (η) one `Resolution` per candidate payload. Nightly, census and `codebook.py apply` consume it; none catches `NeverDeleteError` itself.
- **D4 — One committer.** `unlanded.py` owns both halves of landing: `commit_only(repo, paths, message) -> CommitResult` (nightly's 5-attempt retry on the lock-contention stderr patterns and the one-shot scoped `git add` fallback, moved verbatim from `nightly.py::_git_commit_docs_only`) beside `roll_back`. The census still rolls back on a commit that fails after retries (fail loud); it no longer rolls back on the first transient lock.
- **D5 — `RunOutcome`.** One `StrEnum` in `trickle_state.py` (stdlib-only, on the bare-predicate path): `productive | quiet | barren | failed | deferred`. A capped night is `deferred`: it does not refresh `last_productive_at` and leaves the barren streak unchanged (neither increments nor resets), so a run of deferred nights surfaces through `check_trickle_progress.py`'s existing staleness check. `coder.RunResult.status` becomes a `CodingStatus` enum (`ok | failure`); the five `status == "failure"` comparisons (coder 2, nightly 1, census 2) match on it. The cap fact has one predicate (`coder.py::is_cap_deferral`) feeding `RunOutcome.deferred`.
- **D6 — Barren escalation is a recorded fact.** Nightly posts the barren escalation before `record_run` writes the document and passes `barren_escalated` into it. The streak predicate (`barren_streak_due(state, max_runs)`) and the door→remedy mapping (`BARREN_REMEDIES`) live once in `trickle_state.py`; nightly and `check_trickle_progress.py` call them; `check_trickle_health.py::_should_escalate` reads the recorded `barren_escalated` instead of re-deriving nightly's edge predicate. A document written before γ lacks the field: the probe reads it as **unrecorded**, does not suppress, and says "barren_escalated unrecorded" in its post (INV-13 `readers-prove-their-producer`; at most one extra info escalation, on the first night after deploy). Whether a persistent streak re-escalates stays task 5513's decision, now a one-field read.
- **D7 — Typed state documents.** `scripts/legibility/state_docs.py` (stdlib-only) holds the codec-agnostic pieces: `StateRead[T] = Missing | Malformed(reason) | Ok(value)`, one reader and one writer that route through `shared.safe_io.load_json_or_warn` / `atomic_write_text(mkdir=True)` (with the `_SHARED_SRC` bootstrap `account_pool.py` uses; every unit already runs under `uv run --project shared`). `TrickleState` lives in `trickle_state.py` (γ); `CensusState` and its reader/writer move to a new `census_state.py` (ζ), absorbing `census_trigger.py::load_census_state` and `census.py::advance_census_state`. Each frozen dataclass's `from_json`/`to_json` is its only codec; consumers receive validated values and drop their `isinstance` guards. The racy fixed-name `.tmp` writer in `trickle_state.py::record_run` goes.
- **D8 — Census config: one validated reader.** `config.Census` (and `NoveltySpike`, `TrickleCensusCaps`) become strict: integer thresholds are `StrictInt` with `ge=0` (bools, strings and floats rejected), `dup_rate` stays a bounded float. A malformed value fails `load_config` with a `ValidationError` naming field and value — rejected once, loudly, instead of the trigger's per-field silent fallback (INV-11 `no-silent-fail-soft`). Keys census-incremental L8b (task 6408) deprecates are accepted and ignored with one WARNING, carried from `CensusConfig.from_mapping` into a `config.Census` validator. `census_trigger.py::CensusConfig.from_mapping` and `load_census_config` are deleted; `decide_for_project` takes the loaded `LegibilityConfig` (and so the `config_path` the trickle pinned, task 3269) instead of re-resolving `<project_root>/docs/legibility/legibility.yaml`. The live `legibility.yaml` of both dark_factory and reify validate under the strict model on 2026-10-10 (all integer thresholds are unquoted ints). A nightly that fails at `load_config` is recorded as `failed` by task 6549 (a δ prerequisite).
- **D9 — One layout owner.** `scripts/legibility/layout.py` (stdlib-only) owns the `docs/legibility/` paths: `legibility.yaml`, `confusion-codebook.yaml`, `census-state.json` (absorbing `census_trigger.census_state_path` if task 5783 has added it). The ten literal sites become calls.
- **D10 — Prose is rehomed per edit, not swept** (`fk-0af74b3620ec`'s own proposal, CONTRIBUTING §2). Each leaf that rewrites a function in `sampling`, `digest`, `trickle_state`, `coder`, `nightly`, `census_trigger` or `census` deletes the prose its new structure makes redundant (the "mirrors the merger" notes, the poster-copy rationale, run_census's ordering paragraphs), moves incident history and measurements to its commit message or task record with at most a one-line pointer, and corrects any rotted claim it meets in the code it touches (e.g. `nightly.py::_default_census_launcher`'s "census files its own escalation for the failing stage", false for `census.main`'s prelude; `NightlyResult.capped`'s stale "VOCABULARY RESERVATION for task 4514"). No leaf deletes rationale in code it does not otherwise change. Bare line pins are not swept (CLAUDE.md). The leaf reports each touched module's prose share before/after (the finding's rule) in its task record; there is no numeric target (`docs/code-quality.md` §"Do not steer by").
- **D11 — One alarm poster over the shared client.** A new `scripts/legibility/alarm.py` exposes `raise_alarm(project, source, summary, detail, *, level, poster=None) -> bool` and `fail_loud_boundary(source, project)`. It owns the `escalate_info` tool name, category, severity, the `legibility-<source>-<project>` task-id scheme, the port→URL mapping and the pytest-process guard (`census.py::_refuse_real_post_under_test`'s rule, now for every source). It journals `summary -- detail` at `level` first, then posts, and never raises; it returns whether the post landed. `nightly.post_escalation`, `check_trickle_health.post_health_finding`, `check_transcript_persistence.post_findings` and `census._build_default_escalate_fn` shrink to a source label and message construction; census's three `try: escalate_fn(...) except Exception` wrappers go. `census.main` and `nightly.main` each wrap their whole body in `fail_loud_boundary`, which logs the traceback and raises one alarm on any escaping exception (generalising whatever task 6619 lands for `census.main`'s prelude).
- **D12 — The MCP transport is S14's.** The poster and every other `census_trigger` transport consumer call the shared synchronous MCP client that S14 lands in `shared/` (program seam table: "`shared.mcp_post` as the only MCP client"). ε deletes `census_trigger.py`'s transport (`MCP_STREAMABLE_HTTP_HEADERS`, the SSE decoders, `_raise_on_mcp_error`, `_terminate_mcp_session`, `post_mcp_envelope`, `post_mcp_tool_call`) and re-points every importer present at dispatch, inside and outside `scripts/legibility/` (on `ae3047ec3a`: nightly, census, both `check_*` probes, `scripts/_config_reload_gate.py`). The done-count status fetcher, if task 6408 left it, keeps `StatusFetchUnavailable` at its own boundary by mapping the client's neutral error. No re-export shim (heuristic 13). This PRD builds no transport.
- **D13 — Candidate identity is the merger-minted id.** The census merges mined records **before** verification, so every candidate payload has a codebook id and a `Resolution` (`NewPending(id) | PendingAbsorbed(id) | RoutedToEntry(id) | RoutedToCandidate(id, disposition)`) in `MergeResult`. Verification clusters are built from merged candidate ids; a verdict resolves to its candidate by id; title stays display text and the merger's grouping rule's input, owned once in `codebook.py`. `_find_pending_candidate_id`, `_find_adjudicated_candidate`, `_novel_clusters`' title de-duplication and most of `_dropped_verdict`'s precedence logic are deleted; `DroppedVerdict` survives only for genuine standing conflicts the `Resolution` names.
- **D14 — Lifecycle is a table.** Candidate disposition (`pending → promoted | rejected`), entry status (including census-incremental's `accepted` / `fixed` from `disposition.py`) and the entry `filing` marker (`withheld → filed`) change only through `Lifecycle` transitions in the codebook's owning module, driven by one legal-transition table that checks the source state and raises naming the transition and the record. `codebook.validate` adds the cross-reference checks: a `promoted` candidate's `promoted_to` resolves to an entry, and `filing` ∈ {withheld, filed}. The live codebook passes both on 2026-10-10 (90 promoted candidates, 0 dangling `promoted_to`; no `filing` values outside the set). Census's `promote_candidate`, `reject_candidate`, `retire_entry`, `mark_entry_filed` dict edits move there (subsuming task 6009).
- **D15 — Stages before layers.** `run_census` becomes `mine → merge → adjudicate → plan_filing → persist`, each a function over frozen values; `persist` alone owns the effect order (file tickets, write record and report, dump codebook, advance state, commit, ledger) that the docstring states in prose today; `CensusOutcome` becomes a sum type (`Deferred(stage: CensusStage, counts) | Done(...) | Unlanded(rollback, filed_tickets)`); the injected callables travel as one `CensusEffects` value. Only then does `census.py` split, at the seams the stages expose, re-measured at dispatch by heuristic 14's protocol; the pin's measurement (mining; headroom and pytest guard; report records with `DroppedVerdict`; the default verifier; CLI) is the starting partition. The verifier builder becomes a public function in its own module, and `main` injects the poster instead of tests patching `_post_mcp_tool_call`. Each new file passes heuristic 13; no re-export shim keeps an old dotted path alive; tests move to the public names.
- **D16 — `run_nightly` as stages.** Each stage returns a value; posted alarms accumulate in one list; one `_finish(outcome, alarms, ...)` builds the frozen `NightlyResult`, deriving `escalated` from the list instead of OR-ing flags at each return site; the census launcher and digest builder are injected through `run_nightly` like `invoke`, `status_fetcher`, `poster`, `committer` and `recorder` already are. Nightly prints one final journal line rendered from `NightlyResult`.
- **D17 — Sequencing against census-incremental.** Leaves that only touch files census-incremental's remaining chain does not restructure (α), or touch them additively (β, γ, δ, ε), go now. Leaves that restructure what that chain is editing — the census config reader and state (ζ), candidate identity and lifecycle (η), `run_census`'s stages and the `census.py` partition (θ, ι) — wait for its last `census.py` editor, task 6408.

## 5. Contracts (internal seams)

### 5.1 Detector layer (α)

- `detectors.SignalClass(StrEnum)`: `tool_error, self_correct, not_found, df_guard, interrupt, designed_outcome`. `digest.py::SIGNAL_WEIGHTS`, `digest.signal_counts` keys and `sampling.SignalCounts` fields derive from it.
- `detectors.record_hits(record) -> RecordHits`: frozen; for each class, the matched patterns (with the carrier text index the digest renders). Per record, no cross-record state. `tool_error`/`designed_outcome` keep going through `digest.classify_error_content` (task 3610).
- `digest.signal_counts(records)` sums hits per class (as today). `sampling.scan_transcript(path)` reduces each record's hits to at most one increment per class. **Invariant:** for every transcript, `bool(scan.counts[c]) == bool(digest.signal_counts(load_transcript(path))[c])` for all six classes; both readers already share `inventory.iter_json_lines`'s parse contract.
- `detectors.AgentClass(StrEnum)` (the five strata of `sampling.py::STRATA`) and `detectors.classify_agent_class(path, records)` per D2.

### 5.2 Codebook merge (β, extended by η)

- `codebook.apply_coding_records(cb, records, *, invariant_slugs) -> MergeResult` (D3); never raises `NeverDeleteError`; never mutates `cb`.
- η adds `MergeResult.resolutions: tuple[Resolution, ...]` keyed by candidate id, and `Lifecycle` transitions (D14).

### 5.3 Landing (β)

- `unlanded.commit_only(repo, paths, message) -> CommitResult(ok, sha, stderr, attempts)` per D4; raises nothing for git failures (the caller decides); bounded at 5 attempts (INV-4 `storm-escape-required`).

### 5.4 Run outcome and state documents (γ, ζ)

- `trickle_state.RunOutcome`, `trickle_state.classify_run(exit_code, selected_count, *, capped)`, `record_run(..., outcome, barren_escalated)`, `barren_streak_due`, `BARREN_REMEDIES`, `TrickleState` (D5, D6).
- `state_docs.StateRead`, `state_docs.read_state(path, codec)`, `state_docs.write_state(path, value)` (D7). `census_state.CensusState`, `load`, `advance` (ζ).

### 5.5 Alarm (ε)

- `alarm.raise_alarm(...) -> bool`, `alarm.fail_loud_boundary(source, project)` (D11). The client it calls is S14's (D12).

## 6. Boundary-test sketch

| # | Leaf | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | α | Committed fixture corpus, one transcript per discriminating case (bare `scope_violation`, `phantom-done`, a re-ingested coder JSON answer, a harness-injected briefing, a guard tool_use, each of the six classes positive) | — | `scan_transcript` and `digest.signal_counts` agree on class presence for every file; the re-ingested/injected files score 0 in both |
| 2 | α | Agent class | the five path/first-turn/haystack shapes | one classifier, one answer per shape, used by both the sampler and `build_digest` |
| 3 | β | Deletion-shaped coder record (`matches: [{entry_id, action: delete}]`) through nightly, census and `codebook.py apply` | real codebook fixture | each completes; `MergeResult.skipped_deletions` names the record; census persists its run |
| 4 | β | Commit under contention | real git fixture holding `.git/index.lock`, released after the second attempt | nightly and census both land through `commit_only`; `attempts == 3` |
| 5 | γ | Capped night | coder returns a cap deferral | `trickle-state.json` outcome `deferred`; `last_productive_at` unchanged; streak unchanged; `check_trickle_progress.py` reports deferred |
| 6 | γ | Barren streak at a non-default threshold | `--max-barren-runs 5` | nightly and both probes agree; the health probe reads `barren_escalated` |
| 7 | γ | Pre-γ document (no `barren_escalated`) | old-shape fixture | probe reports "unrecorded", does not suppress |
| 8 | δ | Each nightly exit branch (budget, capped, barren, coder failure, merge, census step) | injected stages, no dotted-path patches | `NightlyResult` from `_finish`; `escalated` true iff an alarm was posted |
| 9 | ε | Every source posts against a real escalation server (`escalation/tests/test_legibility_census_escalation_e2e.py`'s `serve_escalation_mcp`) | live server | the escalation is retrievable by id for census, nightly, trickle-health and transcript-check |
| 10 | ε | Server down | no listener | the journal line names the cause; `raise_alarm` returns False; nothing raises |
| 11 | ε | Under pytest with no injected poster | `PYTEST_CURRENT_TEST` set | every source refuses the real post |
| 12 | ζ | Malformed `census:` values (`max_interval_days: true`, `floor_days: -1`, `count: '9'`) | yaml fixtures | `load_config` raises naming field and value; no reader falls back |
| 13 | ζ | Deprecated key present | 6408's deprecated key | accepted, one WARNING, trigger output unchanged |
| 14 | η | Reworded title across two runs; two causes sharing a title | codebook fixture | verdicts resolve by candidate id; no dropped verdict |
| 15 | η | Illegal transition (`reject` an already-promoted candidate) | fixture | raises naming transition and record; `validate` flags a dangling `promoted_to` |
| 16 | θ | Commit fails after persist filed tickets | fake effects | `Unlanded(rollback, filed_tickets)`; effect order observed equals the `persist` table |
| 17 | ι | Test suite | — | no test reads or patches an underscore name of a census module |

## 7. Pre-conditions / substrate (G3)

Verified on `ae3047ec3a` (2026-10-10):

- **Sampling / digest:** `sampling.py::_NOT_FOUND_PATTERNS`, `_SELF_CORRECT_PATTERNS`, `_DF_GUARD_TEXT_PATTERNS`, `_INTERRUPT_PATTERNS`, `_has_guard_tool_use`, `scan_transcript`, `SignalCounts`, `classify_agent_class`, `STRATA`, `_WORKTREE_DIR_MARKERS`; `digest.py::signal_counts`, `iter_self_corrections`, `iter_not_found`, `iter_df_guards`, `iter_interrupts`, `iter_error_neighborhoods`, `classify_error_content`, `_dialogue_text_sources`, `_signal_text_sources`, `_strip_decoy_lines`, `is_reingested_content`, `has_non_human_origin`, `_is_non_dialogue_carrier`, `classify_agent_class`, `DIGEST_INSTRUMENT_VERSION` (5), `build_digest` (`agent_class_override`); `inventory.py::iter_json_lines`; `census_window.py` imports `sampling`/`digest` only through public names.
- **Codebook / landing:** `codebook.py::apply_coding_record` (returns `tuple[dict, dict]`), `NeverDeleteError`, `SlugTally`, `_cmd_apply`, `validate`, `dump`; three merge loops (`nightly.py::run_nightly`, `census.py::run_census`, `codebook.py::_cmd_apply`), `NeverDeleteError` caught in 2. `nightly.py::_git_commit_docs_only`, `_LOCK_ERROR_PATTERNS`, `GitCommitResult`; `census.py::_build_default_commit` (no retry); `unlanded.py::roll_back`.
- **Outcomes / state:** `trickle_state.py::OUTCOME_*` (4 values, no deferred), `classify_run`, `record_run` (fixed `.tmp`), `load_state`, `DEFAULT_MAX_BARREN_RUNS`; `coder.py::RunResult`, `CodingResult`, `is_cap_deferral`; `nightly.py::NightlyResult`, `_record_trickle_progress`, `_escalate_barren_streak`, `evaluate_census_step`, `_default_census_launcher`; `check_trickle_health.py::_should_escalate`; `check_trickle_progress.py` (`streak >= max_barren_runs`). `shared/src/shared/safe_io.py::atomic_write_text(mkdir=)`, `load_json_or_warn` (stdlib only).
- **Alarm / transport:** `nightly.py::post_escalation`, `check_trickle_health.py::post_health_finding`, `check_transcript_persistence.py::post_findings`, `census.py::_build_default_escalate_fn`, `_post_mcp_tool_call`, `_refuse_real_post_under_test`, `own_endpoint`; `census_trigger.py::post_mcp_envelope`, `post_mcp_tool_call`, `StatusFetchUnavailable`; outside importer `scripts/_config_reload_gate.py`; `escalation/tests/test_legibility_census_escalation_e2e.py` drives a real escalation server for census, nightly and transcript-check. `shared/src/shared/mcp_post.py` is async-only (`post_mcp_tool_call`, `open_mcp_client`, `decode_mcp_response_body`, `MCP_POST_HEADERS`).
- **Config / trigger:** `config.py::Census`, `NoveltySpike`, `load_config` (lax pydantic, `extra='allow'`); `census_trigger.py::CensusConfig.from_mapping`, `load_census_config`, `load_census_state`, `decide_for_project`; both projects' live `census:` blocks are integer-valued.
- **Census:** `census.py::run_census` (774 lines, 26 parameters), `_build_default_verify_fn` (277 lines), `main` (374 lines), `_novel_clusters`, `_find_pending_candidate_id`, `_find_adjudicated_candidate`, `_dropped_verdict`, `DROP_*`, `DroppedVerdict`, `CensusOutcome`, `MiningResult`, `promote_candidate`, `reject_candidate`, `retire_entry`, `mark_entry_filed`, `advance_census_state`; bare `import codebook` / `coder` / `filing_policy` / `config`.

**Not present, queued as prerequisites:**

- **A synchronous shared MCP client** (header pair, SSE decode, initialize handshake for the stateful escalation servers and the stateless fused-memory server, session release, `isError` surfaced, a neutral transport error): owned by S14 (program seam table). Hard dependency of ε (D12). Its required extent is stated in the capability manifest so the producer-extent check can be made against S14's leaf at decompose.
- **Census-incremental's `census.py` chain end (task 6408)**: hard dependency of ζ and so of η, θ, ι (D17).

## 8. Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S14 (`plans/review-all-s14-*-prd.md`) | consumes | shared synchronous MCP client in `shared/` | S14 | queued; ε depends on its leaf (id at decompose) |
| `plans/census-incremental-prd.md` (6402–6408) | sequenced after | `census.py`, `census_trigger.py::CensusConfig`, `config.py::Census`, codebook lifecycle (`disposition.py`) | census-incremental for its semantics; **this PRD** for the reader mechanism, candidate identity, lifecycle table and `census.py` partition | ζ depends on 6408; its C7 sentence "validated by `census_trigger.py::CensusConfig.from_mapping`" is superseded by D8 (the semantics are unchanged) |
| Detector module | produces | `detectors.record_hits`, `classify_agent_class` | **this PRD (α)** | consumers: `sampling`, `digest`, `census_window` (through sampling) |
| Codebook merge result | produces | `codebook.apply_coding_records` / `MergeResult` | **this PRD (β, η)** | consumers: nightly, census, `codebook.py apply`, census-incremental's later readers |
| Shared streak primitive (S14) | not consumed | — | S14 | the barren streak is a persisted counter in `trickle-state.json`, not an in-process streak; D6 keeps it there |
| Other PRDs naming `census_trigger.post_mcp_tool_call` as the script transport (completion-driven-triggers, task-metadata-lookup, deferral-flip-condition, inv12, plan-deviation-recording) | consumes (by name) | the transport ε deletes | S14 owns the replacement | ε re-points every importer present at its dispatch; a task landing afterwards imports the shared client (its verify fails on the old import otherwise) |

**Live tasks this PRD subsumes or orders** (decompose acts on each; none carries one of the sixteen keys — the forensic `json_each(metadata, '$.x_finding_key')` query found no task for any of them on 2026-10-10):

- Cancel with `x_acceptance_reason` naming the subsuming leaf: **5693** (sampler carrier filter → α), **5960** (re-ingestion stratum module → α), **6357** (mutation rule into codebook → β), **6466** (transport out of census_trigger → ε), **5927** (census-state path literals → ζ's layout), **6009** (codebook-lifecycle family out of census → η), **4241** (verdict dropped when no pending candidate resolves; impossible by construction under D13 → η).
- Depend on this batch: **5785** (agent class inherited from an embedded digest) depends on α; **5513** (barren re-escalation policy) depends on γ.
- This batch depends on: γ on **5678**, **6548**, **5783** (in-flight `trickle_state.py`/`nightly.py` semantics); δ on **6549**, **6358**, **5754** (in-flight `run_nightly` edits; 5754 adds a stage); ε on **6619** (in progress, `census.main`'s alarm) and S14's client; ζ on **6408**.
- Left alone (file overlap only; the scheduler's file lock serialises them): 6602, 6603, 5202, 3985, 5959 (`digest.py`); 3416, 5717, 6421 (`census_trigger.py`); 6645 (`coder.py`, the sibling key `fk-b83311fe5621`); 5759, 4118, 4129, 4594, 6467 (`census.py`).

## 9. Decomposition plan

Same-file serialisation: every leaf but α edits `nightly.py` and/or `census.py`, so β → γ → δ → ε → ζ → η → θ → ι is one chain. α is independent. Priority = the highest severity among a leaf's keys. Every leaf is `task_kind=normal`, and every leaf's brief cites heuristics 13 and 14 where it moves code.

- **α — One detector layer for sampler and digest** (high; no deps; ~1,000 LOC).
  - **Files:** new `scripts/legibility/reingestion.py`, new `scripts/legibility/detectors.py`, `digest.py`, `sampling.py`; tests `scripts/tests/test_legibility_detectors.py` (new), `test_legibility_reingestion.py` (new), `test_legibility_sampling.py`, `test_legibility_digest.py`; a committed fixture corpus under `scripts/tests/fixtures/legibility_detectors/`.
  - **Behaviour:** D1, D2, §5.1. Deletes sampling's four pattern tables, `_has_guard_tool_use` (unless promoted per D1), its `classify_agent_class` and `_first_user_turn_text`, and digest's `classify_agent_class`. D10 for the touched functions.
  - **Keys:** `fk-3a23d0a51e69`, `fk-0af74b3620ec`.
  - **Signal:** boundary rows 1–2 in CI over the committed corpus; and, recorded in the task record, the finding's own measurement re-run on the live population (the 400 most recent dark-factory transcripts under 3 MB) printing `0 of N sessions disagree` on class presence (was 56 of 87).
- **β — One codebook merge result and one committer** (medium; no deps; ~700 LOC).
  - **Files:** `codebook.py`, `unlanded.py`, `nightly.py`, `census.py`; tests `test_codebook.py`, `test_legibility_unlanded.py`, `test_legibility_nightly.py`, `test_legibility_census.py`.
  - **Behaviour:** D3, D4, §5.2–5.3. The three merge loops become calls; nightly's four-key sum becomes `MergeResult.mutated`; census handles a deletion directive like the others; census and nightly inject `unlanded.commit_only`; `census.py` and its tests import siblings as `legibility.*` only, so `NeverDeleteError` has one class identity.
  - **Keys:** `fk-f1314d3b25eb`, `fk-74142ca0f882`, `fk-234e0a63dfd7`.
  - **Signal:** `python scripts/legibility/codebook.py apply` prints the `MergeResult` summary line, including skipped deletion directives with reasons; the nightly journal's merge line reads `MergeResult`; boundary rows 3–4 in CI against a real git fixture.
- **γ — Typed run outcome and trickle state; barren escalation recorded** (medium; deps β, 5678, 6548, 5783; ~1,000 LOC).
  - **Files:** `trickle_state.py`, new `state_docs.py`, `coder.py`, `nightly.py`, `census.py` (the two status comparisons), `check_trickle_health.py`, `check_trickle_progress.py`; tests `test_trickle_state.py`, `test_check_trickle_health.py`, `test_check_trickle_progress.py`, `test_legibility_coder.py`, `test_legibility_nightly.py`.
  - **Behaviour:** D5, D6, D7 (trickle half), §5.4. D10.
  - **Keys:** `fk-15f40965b6ba`, `fk-988f412eced2`, `fk-0531ecaefba5`, `fk-0af74b3620ec`.
  - **Signal:** after the next real nightly run, `check_trickle_progress.py dark_factory 3` prints the recorded `RunOutcome` and the `barren_escalated` value read from the `trickle-state.json` that run wrote (INV-13: a production write, not a seeded one); boundary rows 5–7 in CI.
- **δ — `run_nightly` as stage functions with one result builder** (medium; deps γ, 6549, 6358, 5754; ~800 LOC).
  - **Files:** `nightly.py`; `test_legibility_nightly.py`.
  - **Behaviour:** D16. Removes the seven hand-built `NightlyResult` literals and the `HAZARD` comment telling test authors to stub `nightly._default_census_launcher`. D10.
  - **Keys:** `fk-da87d818b1b8`, `fk-0af74b3620ec`.
  - **Signal:** the nightly journal ends each run with one line rendered from `NightlyResult` (`legibility trickle: <date> outcome=<RunOutcome> alarms=<n> ...`); `test_legibility_nightly.py` contains no `setattr(nightly, ...)` of `build_digests`, `evaluate_census_step`, `_default_census_launcher`, `select_digest_sessions` or `build_digest`; boundary row 8.
- **ε — One alarm poster over the shared MCP client** (high; deps δ, S14's client leaf, 6619; ~1,000 LOC).
  - **Files:** new `scripts/legibility/alarm.py`, `census_trigger.py`, `nightly.py`, `census.py`, `check_trickle_health.py`, `check_transcript_persistence.py`, `scripts/_config_reload_gate.py`; tests `test_legibility_alarm.py` (new), `test_census_trigger.py`, `test_config_reload_gate.py`, `scripts/tests/config_reload_script_fakes.py`, `escalation/tests/test_legibility_census_escalation_e2e.py`.
  - **Behaviour:** D11, D12, §5.5. Deletes the four `_ESCALATION_*` constant sets and envelope builders, census's three wrappers, `census_trigger.py`'s transport (~450 lines) and the "by CONVENTION not import" paragraphs. D10.
  - **Keys:** `fk-ef642b412580`, `fk-12f0d08a6ccd`, `fk-0af74b3620ec`.
  - **Signal:** boundary rows 9–11; the e2e suite (real escalation server) passes for all four sources through the shared client; `git grep -E 'census_trigger\.(post_mcp_tool_call|post_mcp_envelope)'` finds nothing.
- **ζ — Census trigger reads validated config and a typed census state** (medium; deps ε, 6408; ~800 LOC).
  - **Files:** new `layout.py`, new `census_state.py`, `config.py`, `census_trigger.py`, `census_identity.py`, `census.py` (`main`, `advance_census_state` removed, inline paths), `nightly.py` (`evaluate_census_step` passes the loaded config), `codebook.py` (path literal); tests `test_legibility_config.py`, `test_census_trigger.py`, `test_legibility_census_state.py` (new).
  - **Behaviour:** D7 (census half), D8, D9.
  - **Keys:** `fk-e3cc7c87d73e`, `fk-0531ecaefba5`.
  - **Signal:** `census_trigger.py evaluate --project-root /home/leo/src/dark-factory` prints its thresholds from the validated `LegibilityConfig.census`; boundary rows 12–13; `git grep -n 'def from_mapping' scripts/legibility/census_trigger.py` is empty.
- **η — Candidate identity by merger-minted id; lifecycle as a transition table** (high; deps ζ; ~1,200 LOC).
  - **Files:** `codebook.py` (or a `codebook_lifecycle.py` it imports downward, if heuristic 14's measurement says so — tactical), `census.py`, `disposition.py` (census-incremental's), `filing_policy.py`; tests `test_codebook.py`, `test_legibility_census.py`, `test_legibility_disposition.py`.
  - **Behaviour:** D13, D14, the `Resolution` half of §5.2. Re-measures at dispatch which title joins remain after 6405–6407 and removes all of them.
  - **Keys:** `fk-3ea2b2df348a`, `fk-a0af0f46c1dc`.
  - **Signal:** the next census report's verify-call count equals its distinct merged candidate ids, and its dropped-verdict section names no title-mismatch drop; `python scripts/legibility/codebook.py validate docs/legibility/confusion-codebook.yaml` runs the new cross-reference checks on the live codebook and exits 0; boundary rows 14–15.
- **θ — `run_census` as stage functions with an explicit persist order** (medium; deps η; ~1,000 LOC).
  - **Files:** `census.py`; `test_legibility_census.py`.
  - **Behaviour:** D15's first half: stages, `CensusOutcome` sum type, `CensusStage` enum, `CensusEffects`; the ordering paragraphs and stage-boundary comment blocks go (D10).
  - **Keys:** `fk-c9ab2773a5d9`, `fk-0af74b3620ec`.
  - **Signal:** `census.py main` exit paths match on `CensusOutcome` variants (no `status ==` string tests); boundary row 16; the next automatic census writes its report through `persist`.
- **ι — `census.py` layers and a public verifier module** (medium; deps θ; ~1,400 LOC, mostly moves).
  - **Files:** `census.py`, new `census_verify.py`, and the further layer modules heuristic 14's re-measurement names (pin starting point: `census_mining.py`, `census_headroom.py`, `census_report.py`); tests `test_legibility_census.py`, `test_legibility_census_verify.py` (new), `test_census_verify_sandbox_cwd.py`, `test_census_harness_config.py`, `escalation/tests/test_legibility_census_escalation_e2e.py`, `scripts/tests/conftest.py`.
  - **Behaviour:** D15's second half; the heuristic-14 measurement goes in the task record with a one-line pointer.
  - **Keys:** `fk-c9ab2773a5d9`, `fk-ee9a4bae4e83`.
  - **Signal:** boundary row 17; `python scripts/legibility/census.py --help` runs from the split modules and the next automatic census (launched by the nightly trickle) writes its report through them; every new file imports only downward (no `import census` from a layer module).

Sizing (overlay bands): ι and η sit at the top of the band, δ near the middle, β at its lower part; no leaf declares more than 12 files. The trigger-chain gate gains α, ε and η (the high leaves) as dependencies at decompose.

## 10. G6 premise validity

- α's `0 disagree` is true by construction: both scorers reduce the same `record_hits` over the same record stream (`iter_json_lines` and `load_transcript` share one parse contract), and presence of a class in a record is the same boolean whichever reduction counts it. A count equality would be false (digest counts one hit per matched pattern), so the signal asserts presence only.
- β's deletion-directive row is reproducible today (the finding's reproduction: `validate_coding_record` accepts the record and `apply_coding_record` raises).
- γ's signal reads a production write; the `deferred` path is exercised in CI because a capped night is not guaranteed to occur within the observation window.
- ζ's rejection signal is backed by a rejection mechanism (`StrictInt`/`ge=0` raise `ValidationError`); the live configs of both projects pass the strict model, so deploying ζ does not break either nightly.
- η's `validate` signal: the live codebook has 0 dangling `promoted_to` and no out-of-set `filing` values, so the new checks pass on deploy.
- ε's e2e signal runs against a real server in CI (the existing `serve_escalation_mcp` fixture).

## 11. G7 advisory walk (`docs/legibility/design-invariants.md`)

No unwaived hit. The batch removes `no-lockstep-duplication` instances (α, β, γ, ε, ζ) and `one-fact-one-home` instances (β's mutation rule, γ's barren fact, ζ's thresholds and paths). `no-silent-fail-soft`: β reports skipped deletions; ζ rejects instead of falling back; ε returns whether a post landed and journals first. `storm-escape-required`: `commit_only` is bounded at 5 attempts; the alarm poster adds no loop. `readers-prove-their-producer`: γ's probe distinguishes an unrecorded `barren_escalated` from false. `contracts-machine-checked`: η's transition table and validate checks, θ's `persist` order. `guards-exercise-behaviour`: α's agreement test runs both scorers on a corpus; ε's suite posts to a real server. No new allow-list or suppression (`exceptions-owned-or-ratified`).

## 12. Out of scope

- The census-incremental semantics (C1–C7 of that PRD), the trigger's conditions, and the similarity/prescreen logic.
- Fixing task 5785's agent-class inheritance (re-pointed at α's one classifier).
- The barren re-escalation policy (task 5513).
- Building any MCP transport (S14).
- `coder.py`'s bare `import codebook` (`fk-b83311fe5621`, task 6645).
- A prose sweep of modules a leaf does not otherwise change (D10), and bare line pins (CLAUDE.md).

## 13. Open questions (tactical)

1. **Empty-transcript agent class.** `digest.classify_agent_class` returns `unknown` for no records; the sampler returns `interactive`. **Suggested:** keep `unknown` as an explicit `AgentClass` member that only the digest renders; the sampler never sees an empty session it would stratify. Decide in α.
2. **Where lifecycle transitions live.** `codebook.py` (1,146 lines on `ae3047ec3a`) or a `codebook_lifecycle.py` it imports downward. **Suggested:** decide by heuristic 14's measurement in η; either passes heuristic 13.
3. **ι's exact partition.** Re-measured at dispatch; census-incremental adds modules (`disposition.py`, `prescreen.py`, `similarity.py`, `synthesis.py`) that may already absorb parts of the pin's partition.
4. **S14 client extent.** Confirm at decompose that S14's client leaf delivers the extent §7 lists (sync call, both server modes, SSE, session release, `isError`, neutral error); if it does not, ε's dependency moves to the S14 leaf that does, or S14 amends its leaf.
