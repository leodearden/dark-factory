# Evaluating the fused-memory subsystem: dimensions, candidate evals, trade-offs

**Status:** proposal (design only — nothing here is filed as tasks). **Date:** 2026-07-29.
**Origin:** spawn brief `df-memory-subsystem-eval-design` (Leo: "What are the correct
dimensions of performance to be measuring, and what are the best evals to measure them
with? Offer some possibilities with trade-offs.")
**Immediate trigger:** the 2026-07-28/29 consolidation-retrieval inversion (DF 3111) and
the fact that four in-flight designs (3111, 3112, 3129, 3136) address the same pathology
at different layers with **no measurement to arbitrate between them** — 3112 and 3129
are being chosen between essentially on argument.

Everything in §4 (substrate) was verified against live source, the live store, or task
state on 2026-07-29; per-item evidence is cited inline.

---

## 1. What the subsystem is *for* — and why that reframes "performance"

fused-memory's job is not to be a good search engine over its own corpus. Its job is to
**change fleet-agent behaviour**: an agent should get the right fact into context at the
moment it needs it, at tolerable token cost, and should not re-derive or re-write what
the store already knows. Classic IR metrics (recall@k, precision) are necessary but not
sufficient — they are *leading indicators* for the behavioural outcomes that actually
matter, and they are only valid to the extent the query distribution they're computed
over resembles what agents actually ask.

The useful mental model is a funnel, and each dimension below is a stage of it:

```
  WRITE  ──► STORE ──► RETRIEVE ──► CONSUME ──► ACT
  (D3:      (D6:       (D1, D7:     (D2, D4:    (D5: behaviour
   write     corpus     findability,  correct,    change — the
   health)   integrity) fusion)       cheap)      actual objective)
```

A failure anywhere upstream shows up downstream with delay and attenuation — which is
exactly what happened with the consolidation pathology: a *storage-shape* decision
(one long canonical) caused a *retrieval* failure (centroid embedding unfindable), which
caused a *behavioural* failure (agents concluded the topic was unrecorded and wrote
entry N+1), which fed back into *write-path* degradation (post-consolidation write rate
measurably doubled). An eval program should instrument the funnel at every stage, and be
explicit about which measurements are leading (fast, cheap, proxy) versus lagging (slow,
expensive, real).

One measurement discipline up front, learned from reproducing the pathology live:
**every retrieval metric in this program must be rank-based, never absolute-score-based.**
Re-running 3111's probe today on the canonical's own topic phrase returned scores of
0.44–0.51 for the same corpus where the task record measured 0.72–0.90 — wording and
embedding/config drift move the score scale wholesale. Ranks and set-membership
(present-in-top-k) survive that; thresholds on raw cosine do not.

---

## 2. Dimensions of performance

### D1. Retrieval effectiveness at the moment of need *(leading — the core axis)*

Does a query an agent would realistically issue surface the entry that would have
changed its behaviour, within the `limit` it realistically uses (agents and the near-dup
guard both live at k=5–10)?

Sub-metrics, in decreasing order of how much they currently hurt:

- **Canonical findability** — for a consolidated topic, does the authoritative entry
  appear in top-k on that topic's own queries? (Live answer today: no. On the reify
  harness-layout topic the canonical is absent from top-10 *on its own topic phrase*,
  while a duplicate written 28h later ranks 1st. Reproduced during this design session,
  matching 3111's measurements.)
- **Rank inversion** — superseded/corrected entries outranking their successors.
- **Claim recall** — for a query targeting one specific claim inside a topic, does an
  entry carrying that claim surface? (This is the metric on which 3112's N-short-peers
  and 3129's grouped-suppression designs genuinely differ.)
- **Contamination** — off-topic results occupying top-k slots. (The live probe put two
  entirely unrelated entries above on-topic cluster members.)

### D2. Correctness & currency of what surfaces *(leading, but label-hungry)*

Conditional on something being retrieved: is it true *now*? The store demonstrably
carries superseded and wrong entries — the reify appendix taught a wrong rule for five
weeks (esc-5712-1) precisely because retrieval kept it unread and unchallenged; a
sibling that *was* retrievable carried the correction. Sub-metrics: superseded-entry
surfacing rate; wrong-outranks-right incidents; staleness (entries whose falsifying
event — a task going terminal, a file changing — is machine-visible but unjoined).

### D3. Write-path health & corpus convergence *(leading)*

Does the corpus converge instead of accreting? Sub-metrics: duplicate-cluster count and
size distribution over time; per-topic accretion rate (the pathology's signature:
~1.5/day → ~3.4/day *after* consolidation); net entry delta per consolidation event
(the Stage-1 ratchet: failed pass ⇒ +1); time-from-write-to-fold; and — once C1 triage
lands — routing accuracy against curator dispositions.

### D4. Cost per useful recall *(orthogonal efficiency axis)*

Context-window burn: tokens returned per search, tokens per fact that plausibly got
used, search latency, LLM calls per write. A 9,000-char canonical returned on every
topic query is a real cost even when findability is fixed — 3129's grouped read and
3112's short peers make *opposite* bets on this axis, which is measurable.

### D5. Agent behaviour change *(lagging — the actual objective)*

- **Re-derivation rate**: agent spends turns rediscovering a fact the store held at
  session start.
- **Repeat-incident rate**: a gotcha with a covering memory bites a second agent anyway.
- **N+1 write rate**: writes that duplicate existing knowledge — the behavioural echo of
  a D1 failure, and the single best single-number proxy for "memory is working",
  because it requires an agent to have searched, missed, and concluded wrongly.
- **Hint efficacy**: task `memory_hints` queries that actually surface something the
  executing agent uses. (Caveat discovered during this design's audit: the hints
  channel is currently *dead as an agent-facing mechanism* — hints are generated by
  recon and consumed only by recon's own Stage-1 context assembly, never delivered to
  task-agent briefings. See §4. Measuring its efficacy first requires deciding whether
  it should exist.)

### D6. Corpus integrity *(mechanical hygiene — cheap to check, embarrassing to skip)*

MCP-envelope leakage, orphan `parent_id`s, dangling `supersedes` UUIDs (already live:
3111's appendix pointer went dangling within a day when curation replaced the target),
fabricated completion claims, malformed metadata. Deterministic lints, no labels needed.

### D7. Fusion-layer correctness *(leading; currently unmeasured and visibly crude)*

The seam that makes fused-memory *fused*: query→store routing and cross-store rank
merging. The current fusion is `(is_primary_store, -relevance_score)` — every
primary-store result outranks every secondary-store result regardless of score — and
Graphiti results carry a *synthetic* score fabricated from rank position
(`1.0 - i*0.05`, `memory_service.py:2916`), so cross-store scores are not commensurable
at all. Nobody knows whether this hurts, because nothing measures it. Also under this
dimension: category-classification accuracy on writes, and dual-write divergence.

**Category blind spot worth naming:** every dedup/consolidation tool in the repo
(near-dup guard, topic guard, audit script, the whole write-path PRD) is scoped to
`procedural_knowledge` — 2.5k/4.0k entries (dark_factory/reify). Meanwhile
`observations_and_summaries` is 16k/24.4k — 6–8× larger, entirely unswept, and included
in every top-k an agent sees. The eval program should measure it even though no fix yet
targets it; otherwise we optimise the small category and call it victory.

---

## 3. Leading vs lagging, and what arbitrates what

| Dimension | Nature | Latency of signal | Best evals (§5) |
|---|---|---|---|
| D1 retrieval-at-need | leading | minutes (probe) | E1, E2, E3 |
| D2 correctness/currency | leading | hours (sweep) | E4, (E1) |
| D3 write health | leading | days (trend) | E5, E6 |
| D4 cost | orthogonal | minutes | E2, E7 |
| D5 behaviour change | **lagging** | weeks | E7 (write-after-miss), E8, E9 |
| D6 integrity | leading | hours | E6 lints |
| D7 fusion | leading | minutes | E11, (E3) |

The four in-flight designs map onto the funnel as: 3111 = retrieve, 3112 = store-shape,
3129 = store-shape + consume, 3136 = write-health detection. §7 states exactly which
eval discriminates between which designs.

---

## 4. Substrate that exists (verified 2026-07-29)

What follows is what an eval program can stand on *today*. Each item was checked against
source, the live store, or task state during this design session.

**α — the calibration harness (DF 3130) is real, in-flight, and is the right
foundation.** Status: `in-progress` with a live claimant (heartbeat 11:16Z today);
nothing on main yet. In its worktree (`.worktrees/3130`):
`fused-memory/tests/fixtures/write_triage_calibration.jsonl` (104 labeled records, not
the reported 89) and `fused-memory/scripts/calibrate_write_triage.py` (~800 lines)
with: fixture loading + pair-set construction, embedding-similarity distributions via
the live embedding path, band derivation (T_high/T_low as order statistics),
**candidate-retrieval recall@k against the real `search` with explicit
canonical-absent tracking**, a markdown report renderer, and a config-block writer. A
live calibration run has already executed ("0 deterministic-band false positives").
Verdict per the brief's question: **yes — build on it, do not parallel it.** Its
components (embed-fn wrapper, recall@k-with-absent-list, report emission) are exactly
E1/E2's bones. Its limitation: fixture scope is write-triage pairs from one curator
session — it calibrates the *write* path, not general moment-of-need retrieval.

**Eval-harness precedent lives in the orchestrator, and the right template is not
`evals/runner.py`.** `orchestrator/src/orchestrator/evals/runner.py` is workflow-shaped
(worktree per run, real agent, judge the diff) — wrong shape for retrieval evals. The
right templates are `evals/prompt_opt/` and `evals/reviewer_trial/`: labeled corpus
(mined from `tickets.db`, frontier-adjudicated + human spot-check), train/dev/test
splits, a `Scorer` protocol, variance machinery, and **canary thresholds with
window-comparison** (`canary.py`) for detecting regression over time. A memory eval
should reuse that structure wholesale.

**Retrieval quality is currently asserted nowhere.** No test in `fused-memory/tests/`
touches ranking, recall, or result identity — all `search` tests are plumbing
(degraded-marker, category filters, routing). `relevance_score` appears in tests only
as hand-set fixture input. The fusion sort and the synthetic Graphiti scores
(§2 D7) have zero coverage.

**Seeded-store probes against the real embedder are cheap and proven.**
`tests/test_recon_dedup_premise.py` is the working recipe: per-worker scoped
`project_id` → derived Qdrant collection → delete before/after, real `Mem0Backend`,
`real_embedder=True` path that drives the actual OpenAI embedding
(`text-embedding-3-small`, 1536-dim), gated behind `-m integration` +
`OPENAI_API_KEY`. ~40 lines to copy. For offline similarity studies without add_memory,
`TaskCurator._get_embedder()` exposes the production embedder object directly
(`middleware/task_curator.py:659`). Caveat: `scripts/cleanup_test_collections.py`
reaps only the `_test_mem0_qdrant_integration_` prefix — extend it for any new one.

**The near-dup guard is replayable offline.** `find_near_duplicate_memory()` and the
topic-cluster matcher are pure, synchronous selectors over already-fetched results
(`near_duplicate_guard.py`) — fetch candidates once, replay the guard at any threshold.
One empirical landmine already on record: `test_recon_dedup_premise.py` documents that
the 0.92-cosine "Mem0 dedups at this threshold" premise is **false** for the
`infer=False` write path — the constant is a guard heuristic, not store behaviour.

**Where memory actually touches an agent — three channels, one of them dead.**
(1) **Briefing-time mechanical injection**: the orchestrator itself fires **four
hardcoded searches at prompt-build time**, `limit=5` each (`briefing.py:978-1013`:
"project overview architecture goals", "coding conventions and project norms",
"recent decisions and rationale", "task {id} context and related decisions"), pasted
into every role's prompt as `# Context`, silently degrading to a one-line notice on
failure. This is the *highest-volume* moment-of-need channel and its four fixed
phrasings dominate the live query log. (2) **Ad-hoc mid-task searches**: roles get
`_MEMORY_INSTRUCTIONS` (`roles.py:218-250`) — write-eager guidance plus "search before
assuming conventions" — but with no session-start mandate and no trigger points. Not
all roles: **reviewers have no memory tools at all** (`REVIEWER_COMPREHENSIVE` gets
read-only file tools only), which is worth knowing before interpreting any
reviewer-behaviour metric. (3) **`memory_hints`**: generated by recon (targeted +
Stage-2) and persisted on tasks — but *consumed by exactly one reader, recon's own
Stage-1 context assembler* (`context_assembler.py:292-313`, 3 queries at limit=3,
unlogged). The orchestrator's 20 references to `memory_hints` are all merge-preservation;
`briefing.py` never reads them. **Task agents never see their hints.** Any eval story
about hints must start from that fact.

**Historical logs hold the real query distribution but no relevance labels.**
`data/reconciliation/write_journal.db` (6.7 GB, never pruned — full history since
2026-04-07) logs **every MCP-surface search unconditionally** at the tool layer
(`tools.py:1441-1459` → `_log_read`), ~394k rows, plus ~59.5k `add_memory` (with
`content[:200]` and ~80% agent attribution) and ~17.5k `delete_memory`. Three hard
gaps: (a) `result_summary` for a search is `{"count": N}` — **no result IDs, no
scores**, so retrospective recall/precision is unreconstructible from the journal;
(b) **attribution is absent** — `agent_id` blank on 99.7% of search rows,
`session_id` on 100%, for a structural reason: `search`'s `agent_id` parameter is
documented as a *result filter* ("filter by authoring agent"), so callers correctly
omit it and the server cannot recover caller identity over HTTP transport; (c) query
text is truncated to 200 chars. The recon ledger's cycle summaries (1,993 rows) carry
per-cycle consolidation stats but only ~18 days survive the 30-day TTL + prune.

**Transcripts are the one place "what the agent was shown" survives.** Full
turn-by-turn agent transcripts are archived durably per task
(`data/orchestrator/agent-transcripts/`, 294 task dirs / 208 MB today, gzipped
Claude Code session JSONL including **every fused-memory search call with its full
result payload**; retention knobs exist but no GC consumer is built yet — task 2731).
Joined with `runs.db` (16.8k invocations: per-role tokens/cost/turns; 2.8k task
results with outcomes), this is a real — if mining-intensive — substrate for both
retrospective query-result replay and re-derivation auditing (E7, E8).

**Scheduled-report plumbing is a solved problem.** The
`fused-memory-flag-marker-sweep.{sh,service,timer}` pattern (env-sourcing shell wrapper
→ `uv run` Python → JSON report; separate read-only `--check` predicate wrapper usable
as a `before_done` gate; installer script; unit-config tests) is the idiom for any
scheduled eval. For persisted artifact series rather than journal lines,
`cgl_eta_auto_apply_impl.py`'s `data/…/report-{STAMP}.json` idiom is the copyable
variant.

**The deterministic dedup tool needs an upgrade before it can cover the corpus.**
`audit_duplicate_memories.py` is `difflib.SequenceMatcher` union-find at 0.85 *lexical*
similarity, `procedural_knowledge` only, O(n²) pairs. It cannot reach the 24k-entry
observations corpus (n² ≈ 3×10⁸ ratio calls), and lexical similarity misses exactly
the paraphrase-with-novel-fragment duplicates that caused the curation crisis (measured
genuine-rediscovery pair: 0.824 cosine, well below lexical near-identity). An
embedding-based variant using Qdrant ANN as the candidate generator is both the fix and
an eval instrument (E6).

**Live-store probe facts an eval must design around:** search results *do* return
`metadata` (topic/kind/supersedes are visible to a probe — λ's missing provenance
fields are agent_id/task_id/created_at, not these), but only 6 entries across both
projects carry `canonical: true` — a findability probe must key on
`kind=procedural_consolidation*` / `metadata.topic` / non-empty `supersedes`, plus a
curated topic registry, not on the canonical flag. And curation *replaces* UUIDs
(417d86d0 → 11ed0e22 within a day of 3111 being filed), so golden fixtures keyed on
raw UUIDs rot within days; key on content hashes with an id-repair step.

---

## 5. Candidate evals

Format per eval: what's measured / fixture & corpus / signal / how it runs / trade-offs.
Costs assume the current corpus scale and `text-embedding-3-small` pricing (embedding
costs are cents; the real costs are always fixture authoring and maintenance).

### E1. Consolidated-topic retrieval probe *(D1, D2 — the flagship monitor)*

**Measures:** canonical-in-top-k rate (k=5 and 10 — 5 because the near-dup guard lives
there), superseded-above-successor count, claim recall on per-facet queries,
off-topic contamination share.

**Fixture:** a topic registry: `topic → {canonical id/content-hash, member ids,
3–5 query phrasings (at least one held out from any tuning), per-facet claim queries}`.
Seed it two ways: auto-derive from consolidation metadata (`kind`, `topic`,
`supersedes`) and curator-gate records (reify has 16+ gates enumerated in 3112 — each
gate is a free labeled cluster); hand-write phrasings. ~20–40 topics across reify +
dark_factory initially. Include the briefing assembler's four fixed queries (§4) in
the probe set — they run against every task's context window and are therefore the
single highest-leverage query surface in the system.

**Signal:** per-topic pass/fail + aggregate rates, trended over runs; canary-threshold
regression alerts (reuse `prompt_opt/canary.py` window comparison).

**Runs:** read-only against the **live** store on a timer (minutes of wall clock,
~50–200 search calls, no writes — the probe run during this session was three MCP
calls); also point-in-time before/after any retrieval-layer change.

**Trade-offs.** Build: small (1–2 days; the prototype was run by hand today, and α's
recall@k/canonical-absent code is directly reusable). Run: trivial. Labels: light —
registry maintenance is the tax, and UUID rot means content-hash keys + an id-repair
step or the fixture decays in days. Drift: deliberate — it probes the live corpus, so
it moves as the corpus moves; that makes it a *monitor*, not a controlled experiment
(pair with E2 for causal claims). Gameability: **high for exactly the fix under
consideration** — 3111's pin makes canonical-in-top-k trivially 100%, at which point
the probe must lean on its other three metrics (claim recall, contamination,
superseded-inversion) and held-out phrasings, or it stops discriminating. Fails to
catch: anything not in the registry; knowledge that should exist but doesn't.

### E2. Storage-shape bake-off on seeded ephemeral stores *(D1, D4 — the arbitration experiment)*

**Measures:** the 3112-vs-3129 question directly, plus whether 3111's pin is needed
under each shape. Metrics per arm: claim recall@k (does the *specific claim* a query
targets surface), canonical/topic discoverability, tokens returned per query (the D4
cost of a grouped read vs N short hits), and near-dup-guard candidate adequacy (replay
the pure guard over each arm's top-5: would the write that became duplicate N+1 have
been matched?).

**Fixture:** take ~10–30 real topics with full cluster content (α's 104-record fixture
+ the curator-gate clusters). Materialise the same knowledge in ephemeral collections,
one per arm: **(a)** status quo (N near-duplicates as they actually existed);
**(b)** 3112's shape — N short single-claim peers sharing `metadata.topic`, one marked
canonical; **(c)** 3129's shape — canonical + child amendment/sighting records with
children suppressed behind a grouped read; **(d)** optionally each of the above with
3111's topic-anchored pin on. Query set: per-topic phrasings + per-claim queries +
realistic paraphrases sampled from the write_journal query log where topical.

**Signal:** a decision table — per arm: claim-recall, discoverability, tokens/query,
guard adequacy. The PRD's choice between δ-as-default and peers-as-default gets made by
reading it.

**Runs:** offline, one-shot (rerunnable), on the `test_recon_dedup_premise.py`
isolation pattern with `real_embedder=True`. ~30 topics × ~10 entries × 4 arms ≈ ~1.2k
embeddings + a few hundred queries: **well under $1 of API and ~an hour of wall clock.**

**Trade-offs.** Build: the largest of the recommended set (2–4 days), and almost all of
it is fixture authoring — decomposing topics into single-claim peers *is* 3112's
editorial work done manually for the fixture, which cuts both ways: it front-runs the
real cost of 3112, and it means **arm quality reflects authoring skill — the
experiment is gameable by authoring one arm well and another lazily.** Mitigate by
having the arms authored blind to the metrics, or by two authors cross-checking.
Run: cheap, repeatable, CI-able in principle (needs `OPENAI_API_KEY` + Qdrant, both
already integration-test dependencies). Drift: none — frozen fixture; that's its
strength (controlled, causal) and its weakness (it will slowly stop resembling the
live corpus). Fails to catch: behavioural effects (whether agents *use* what grouped
reads return), long-tail topic diversity, cross-category interference from the 24k
observations corpus (mitigable: seed each arm's collection with a realistic slab of
distractor entries — worth doing, the contamination result today says distractors
matter).

### E3. Golden moment-of-need query set *(D1, D7 — the regression suite)*

**Measures:** recall@k / MRR over `(query, expected-content, forbidden-content)`
triples spanning **all six categories and both stores** — the only proposed eval that
exercises Graphiti retrieval and cross-store fusion on equal footing.

**Fixture:** mine three sources: task `memory_hints` queries from *resolved* tasks
(hints are written in agent-realistic phrasing and pair naturally with the memory
that in fact mattered — usable as *query material* even though the hints channel
itself never reaches agents, §4); curator/incident postmortems; α's fixture.
Expected/forbidden keyed on content-hash, not UUID. Start ~50 triples, grow
opportunistically — every incident postmortem donates one.

**Signal:** recall@k, MRR, forbidden-in-top-k violations; split train/dev/test if it
ever tunes anything (reuse `prompt_opt` splits).

**Runs:** against the live store (drifts; needs id-repair maintenance) or against a
frozen snapshot collection (stable; snapshot restore from Qdrant is a solved
operation). Recommend: live for the standing monitor, snapshot for pre-merge CI of
retrieval-layer changes.

**Trade-offs.** Build: 1–2 days plus ongoing curation. Labels: human (or
frontier-adjudicated per `prompt_opt` precedent — "decisions ≠ ground truth" caveat
transfers verbatim). Drift: the maintenance tax is real and is the reason golden sets
die; content-hash keys + an explicit quarterly re-adjudication budget or accept decay.
Gameability: low-moderate. Fails to catch: queries nobody thought to gold-label —
i.e. the distribution problem again, which only E7 fixes.

### E4. Superseded-and-stale surfacing sweep *(D2 — deterministic, label-free)*

**Measures:** (a) for every entry with `supersedes`/`corrects` metadata: query with the
*superseded* entry's own content — flag if the superseded entry surfaces without or
above its successor; (b) dangling-pointer census (supersedes/parent/corrects UUIDs that
no longer resolve); (c) staleness joins — entries citing a task as open
blocker/limitation where the task store says terminal (ξ/3135's hook logic run as a
*metric* over the whole corpus, not just as an event trigger).

**Fixture:** none — the corpus's own metadata is the label source. That is the whole
appeal.

**Signal:** counts + worst-offender list per run, trended; feeds curator-gate filing
the same way κ's report does.

**Runs:** scheduled read-only sweep (flag-marker-sweep idiom). Cost: one search per
superseding entry (~hundreds) + metadata scrolls; minutes.

**Trade-offs.** Build: ~1 day. Labels: none. Drift: none (self-referential).
Gameability: low. **Blind spot is structural and must be stated:** it only sees
*declared* supersession — the dangerous wrong entries are precisely the undeclared
ones (the appendix that taught a wrong rule carried no marker saying so). This eval
measures hygiene of the declared graph, not truth. Pairing it with E5's label harvest
(below) is the only path to the undeclared case.

### E5. Write-triage & judge accuracy vs curator labels *(D3 — already in flight; endorse and extend)*

This is PRD leaf α (3130) + leaf γ's judge-accuracy report. Not proposed here —
already being built. Two extensions worth adding cheaply: **(a)** treat every future
curator session as a label harvest — each gate execution yields cluster membership +
dispositions; append to the fixture (the 104-record fixture becomes a growing corpus,
`prompt_opt`-style, instead of a one-shot calibration); **(b)** when triage goes live,
log every band decision + judge verdict with enough context to audit against the next
curation pass — that turns live operation into a continuous labeled stream, i.e.
precision/recall of the triage against curator ground truth, per band.

**Trade-offs:** marginal build cost near zero (riding on committed work); label latency
is curation cadence (weeks); measures the write path only.

### E6. Corpus-health time series *(D3, D6 — 3136 turned into a metric)*

**Measures:** duplicate-cluster count/size distribution; per-topic accretion rate
(entries/day against topic registry); net entry delta around each consolidation event
(ratchet detector); integrity lints (MCP-markup tripwire hits, orphan children,
dangling UUIDs — overlapping E4b); category size trends including
`observations_and_summaries`.

**Fixture:** none (deterministic over the live corpus).

**Signal:** per-run JSON artifact (`report-{STAMP}.json` idiom) + trend deltas; canary
thresholds for "accretion rate doubled post-consolidation" — the pathology's exact
signature, which no instrument would have caught automatically when it actually
happened.

**Runs:** scheduled; this *is* 3136's report with three upgrades: **(1)** persist a
time series, not a point-in-time plan; **(2)** replace SequenceMatcher-only clustering
with embedding-based candidate generation (Qdrant ANN) so it scales past
`procedural_knowledge` to the 24k observations corpus and catches paraphrase
duplicates (the 0.824-cosine class the lexical 0.85 threshold misses); **(3)** cover
all three Mem0 categories.

**Trade-offs.** Build: 1–2 days *on top of* 3136's planned work (the timer/report
plumbing is 3136's; the upgrades are this proposal's). Run: embedding cost for the
ANN variant — but embeddings already exist in Qdrant, so candidate generation is
vector-search over stored vectors: near-free. Labels: none. Drift: it measures drift —
that's the point. Gameability: low. Fails to catch: retrieval quality entirely (a
converged corpus can still be unfindable); duplicate pairs below whatever ANN
candidate threshold is chosen (report the threshold and the dropped-candidate count —
no silent caps).

### E7. Search/write telemetry + shadow replay + write-after-miss *(substrate for D1 realism, D4, and the best D5 proxy)*

**Measures (once the substrate exists):** (a) which agent/task issued which search —
today ~100% unattributed (§4); (b) what each search returned — result IDs, scores,
tokens (D4) — today a bare count; (c) **write-after-miss** — the N+1 loop live: an
`add_memory` whose content is near-duplicate (by the pure guard replayed offline, or
post-hoc by triage bands) of an entry that the *same session's preceding searches
were not shown*. That sequence — searched, missed, wrote a duplicate — is the
pathology's full signature, per-incident, in production, with no labels; (d) shadow
replay: re-run logged queries against a candidate retrieval config and diff the
top-k against what was actually served (offline A/B on the true distribution, no
fleet risk).

**Build — three small, surgical changes** (the audit produced the exact seams):
widen `_log_read`'s search `result_summary` from `{'count': N}` to result IDs +
scores + token counts (`tools.py:1446`); add a *caller* identity field distinct from
`search`'s existing `agent_id` parameter — which is a **result filter**, the design
conflict that explains why 99.7% of rows are unattributed — threaded from the
briefing/MCP client side; and log hint-query executions in
`context_assembler.py:296-303` (currently silent). Log `add_memory` acks likewise
(C1's triage ack schema is about to exist and is the natural payload). All logging,
no behaviour change; volume is ~thousands of rows/day against a table already at
6.7 GB and never pruned — add retention at the same time, and revisit the 200-char
query truncation (full queries are the asset). **A retro-corpus exists before any of
this lands:** the archived transcripts (§4) contain each search *with its full
result payload* for ~294 tasks — mining them yields an immediate, if smaller,
query→shown-results corpus for a first shadow-replay baseline.

**Trade-offs.** Build: ~1 day. Run: negligible. Labels: none. Drift: n/a — it *is* the
ground stream. Gameability: low (production behaviour). Privacy/volume: trivial here.
The catch: **it pays off with latency** — the replay corpus and write-after-miss rates
need weeks of accumulation before they're statistically interesting, which is exactly
why it should land *early*, and why not having landed it a month ago is the reason
today's 3112-vs-3129 debate can't be settled from history. Fails to catch: nothing it
sees, but it only sees the MCP surface — direct-store access (recon internals) needs
its existing causation-id path kept distinct.

### E8. Re-derivation transcript audit *(D5 — sampled, judge-based, honest-but-weak)*

**Measures:** of N sampled completed-task transcripts, in how many did the agent spend
turns deriving a fact the store demonstrably held at session start (and in how many did
a retrieved memory visibly change a decision — the positive case).

**Fixture/corpus:** the durable transcript archive (§4 — 294 tasks today, growing;
crucially it includes the search results each agent actually saw, so the judge can
distinguish "never searched", "searched and wasn't shown it" (a D1 failure), and
"was shown it and re-derived anyway" (a D5 failure) — three different fixes) + a
point-in-time store snapshot or at minimum current-store lookup; an LLM-judge rubric
with a small human-labeled anchor set to estimate judge error.

**Signal:** re-derivation rate ± judge-error bars, with linked exemplar transcripts
(the exemplars are the actionable part; the rate is context).

**Runs:** scheduled sample (say 20 transcripts/week), frontier judge, ~$2–10/week.

**Trade-offs.** Build: 2–3 days (transcript access + rubric + anchor set). Labels:
small anchor set, human. Drift: rubric drift is real; judge model changes move the
metric — anchor set re-scoring on every judge change is mandatory. Gameability: high
if targeted (Goodhart: agents told to "cite memory" will cite it uselessly); keep it a
**diagnostic, never a target**. This is the only direct measure of the actual
objective (D5), and it is noisy, expensive per bit, and unsuitable for CI — say so and
use it as a quarterly reality check, not a dashboard number.

### E9. Planted-canary end-to-end probes *(D5 — controlled behavioural test)*

**Measures:** causal, end-to-end memory efficacy: seed a distinctive, harmless,
verifiably-fake-but-plausible memory (e.g. a project-specific flag or step), dispatch a
controlled task whose efficient solution requires it, check whether the agent's
output/transcript shows recall vs re-derivation vs failure.

**Trade-offs.** Build: 1–2 days for 3–5 canaries + a dedicated project_id/cleanup
discipline (canaries must never pollute real corpora — dedicated scope, or tagged +
swept). Run: real agent invocations, ~$1–5 each; low throughput. Signal per run is
binary and high-variance — this is an *integration smoke test* for the whole funnel
(write→store→retrieve→consume→act in one shot), not a metric. Its unique value:
it's the only eval that fails when retrieval is fine but the *agent ignores what it
retrieved* — a failure class nothing else here can see. Gameability: canaries leak
into agent lore over time; rotate them.

### E10. Fleet A/B on task outcomes *(D5 — named to be declined)*

Variant-on vs variant-off across matched tasks, outcome = success rate / turns / cost /
escalations. **Not recommended at current scale**, stated plainly: the fleet runs tens
of tasks/day with enormous per-task outcome variance and heterogeneous task mix;
detecting anything but a catastrophic effect needs hundreds of matched tasks per arm —
weeks of fleet time per comparison, confounded by concurrent system churn (this repo
merges retrieval-affecting changes weekly). The shadow replay (E7d) captures most of
the decision value at ~zero risk. Revisit if the fleet grows ~10× or if a change is so
contentious that weeks of measurement is cheaper than the argument.

### E11. Router & fusion one-shot measurement *(D7 — measure before building)*

**Measures:** (a) read-router store-selection accuracy on a labeled query set (would
the right store have been queried?); (b) how often cross-store fusion's
primary-first rule inverts an obviously-better secondary hit (replayable offline:
run E3's golden set with `stores` forced both ways and diff); (c) write-classifier
category accuracy against α's labeled categories.

**Trade-offs:** ~1 day, piggybacked on E3's fixture. Deliberately a **one-shot study,
not a recurring eval** — the base rate of misrouting is unknown; if it's ~0 the
recurring cost isn't justified, if it's material it motivates its own fix-PRD.
Building recurring infrastructure before measuring the base rate is how eval programs
bloat.

---

## 6. Trade-off matrix

| Eval | Build | Per-run cost | Labels | Drift/maintenance | Gameable? | CI-able? | Structural blind spot |
|---|---|---|---|---|---|---|---|
| E1 topic probe | 1–2 d | ¢ / minutes | registry (light) | UUID rot → hash keys | **yes, by 3111 itself** — needs held-out phrasings + claim/contamination metrics | scheduled; snapshot-CI possible | unregistered topics; absent knowledge |
| E2 shape bake-off | 2–4 d | <$1 / ~1 h | fixture authoring (heavy, one-shot) | frozen — none | **yes, via arm-authoring quality** | yes (integration-gated) | behavioural effects; long-tail topics |
| E3 golden queries | 1–2 d + curation | ¢ / minutes | human/frontier-adjudicated | the classic golden-set decay | low-mod | yes (snapshot) | unimagined queries |
| E4 superseded sweep | ~1 d | ¢ / minutes | none | none | low | scheduled | **undeclared** wrongness |
| E5 triage accuracy | ~0 (in flight) | ¢ | curator (free, slow) | fixture grows | low | yes | read path entirely |
| E6 health series | 1–2 d over 3136 | ~free | none | none | low | scheduled | retrieval quality |
| E7 telemetry+replay | ~1 d | ~free | none | none | low | n/a (substrate) | pays off with weeks of latency |
| E8 transcript audit | 2–3 d | $2–10/wk | anchor set | rubric/judge drift | **high if targeted** | no | noisy; judge error |
| E9 canaries | 1–2 d | $1–5/run | none | canary leakage | mod | no | low throughput; binary signal |
| E10 fleet A/B | — | weeks of fleet | none | — | low | no | **underpowered at current scale** |
| E11 router study | ~1 d | ¢ | shares E3 | one-shot | low | one-shot | unknown base rate (the point) |

---

## 7. Pragmatic starting subset

Three builds, ordered; total ≈ 4–7 focused days, most components riding on substrate
that exists or is landing anyway.

**First: E2, the storage-shape bake-off** — because it is the only item that directly
discharges the trigger for this brief. It arbitrates:
- **3112 vs 3129** (the mutually-exclusive-defaults question): arm (b) vs arm (c) on
  claim-recall, discoverability, tokens/query, guard adequacy. This is the "settle it
  by running something" the brief asks for.
- **3111**: arm (d) shows what pinning adds *under each* storage shape — including
  whether a good storage shape makes the pin unnecessary, which is a cheaper world.
Fixture authoring is the cost and the bias risk (§5 E2); author arms blind to metrics.

**Second: E1 + E4 as one scheduled "retrieval health" runner** — the standing monitor
that would have caught this pathology the day it happened (canonical absent from
top-10 on its own topic phrase is not subtle). E1 arbitrates nothing by itself but
verifies *whichever* of 3111/3112/3129 ships actually works on the live corpus, and
its claim-recall + contamination metrics keep discriminating after 3111's pin
saturates the canonical-presence metric. E4 rides in the same runner for free and
keeps the declared-supersession graph honest (it would have caught the dangling
417d86d0 pointer within a day). Reuse α's recall@k/canonical-absent code and the
flag-marker-sweep scheduling idiom; add `prompt_opt`-style canary thresholds.

**Third: E7 telemetry** — one day of surgical logging changes (result IDs+scores,
caller attribution, hint-execution logging — exact seams in §5 E7) with weeks-later
payoff, landed early precisely because it back-fills slowly; the transcript archive
provides a smaller retro-corpus immediately. It is the substrate that turns the
*next* 3112-vs-3129-shaped debate into a shadow replay over the real query
distribution instead of an argument, and its write-after-miss metric is the best
cheap proxy for D5 that exists. It also quantifies 3136's value honestly: how often
does the deterministic report surface clusters that write-after-miss telemetry
hadn't already flagged live?

**Deliberately deferred:** E3 (start opportunistically — every postmortem donates a
triple; formalize once E7 shows the real query distribution), E5 extensions (ride on
3130/γ landing), E6 upgrades (ride on 3136), E8/E9 (after the leading indicators
exist — behavioural measurement before retrieval measurement is building the roof
first), E10 (declined at current scale), E11 (one-shot, when E3's fixture exists).

**On 3136 specifically:** none of the first three arbitrate *whether* to build it —
it's detection, orthogonal to the retrieval fixes. E6/E7 instead measure whether it
earns its keep once built (clusters found vs clusters already flagged by live
telemetry; lead time from first duplicate to report to gate). Recommendation: let 3136
proceed (its cost is small, D8 already decided it), but land its report as a
time-series artifact per E6 so it doubles as measurement.

---

## 8. What resists measurement — said plainly

- **Counterfactual utility of a recall.** "The agent read the memory and it changed
  the outcome" has no clean observable: agents don't cite provenance reliably, and the
  counterfactual (what they'd have done without it) doesn't exist. E8's judge is a
  weak proxy and E9's canaries are narrow; both are flagged as such. Anyone proposing
  a "memory ROI" dashboard number is selling something.
- **Coverage — what *should* be in the store but isn't.** No corpus-side measurement
  can see it. The only signals are downstream and lossy: re-derivation mining (E8) and
  repeat incidents. Accept it as unmeasured rather than proxying it badly.
- **Correctness of entries absent declared structure.** E4 audits declared
  supersession; the undeclared-wrong-entry class (the five-week wrong appendix) is
  only ever caught by curation, whose labels arrive weeks late (E5's harvest). There
  is no fast instrument for "this confidently-written memory is false", and
  pretending a similarity metric can find it would be exactly the weak proxy the
  brief asks to avoid.
- **Fleet-level outcome effects** at current scale (E10): underpowered, stated above.
- **Goodhart pressure is structural here**, because the treatments under evaluation
  (3111's pin) directly optimize the headline metric (canonical presence). The
  program's defense is metric plurality (claim recall, contamination, tokens,
  write-after-miss) and held-out phrasings — single-number memory scores should be
  refused on principle.
- **Score-scale non-stationarity**: absolute cosine thresholds embedded in evals rot
  silently when embedding config or query wording drifts (observed first-hand this
  session: same corpus, same topic, 0.72→0.44 scale shift across phrasings/eras).
  Rank/set metrics only; any threshold must be a *calibration output* (α's discipline,
  generalized).

---

## 9. Open questions

1. **Snapshotting.** E3-CI and E7 shadow replay want a frozen store; Qdrant snapshot
   restore is cheap but Graphiti/FalkorDB point-in-time is less so. Is Mem0-only
   snapshotting acceptable for v1? (Probably yes — every measured pathology so far is
   Mem0-side.)
2. **Where fixtures live.** `fused-memory/tests/fixtures/` (α's precedent) vs a
   top-level `evals/` corpus dir mirroring `prompt_opt`. Leaning: keep probe fixtures
   with fused-memory tests; keep any judge/adjudication corpus in the orchestrator
   evals tree where the tooling lives.
3. **Who owns the topic registry** (E1) — auto-derivation catches consolidated topics
   only; curator gates could emit registry entries as a side effect (cheap hook into
   θ/`consolidate_memories`, which holds exactly the right inputs — same argument as
   PRD D5 for guard auto-seeding).
4. **Telemetry retention** (E7): full-query logging at fleet volume into a DB already
   at 6.7 GB (never pruned) needs a retention decision up front, not later.
5. Whether E2 should also test the **combined** shape (3112's peers *plus* 3129's
   child-records for amendments) — the PRD's D4 and 3112's ASKED are not actually
   exclusive for all record kinds (peers for facets, children for sightings), and the
   bake-off is the right place to check whether the hybrid dominates both.
6. **The dead hints channel** (§4) is a finding, not an eval: task agents never
   receive their `memory_hints`, and the briefing assembler's four fixed queries are
   the only mechanical delivery of memory into prompts. Whether hints *should* be
   delivered (and whether the four briefing queries are the right four — nothing has
   ever measured what they return) is a design question this eval program would
   inform (E1 includes those queries; E7 would log their yield) but cannot answer.
