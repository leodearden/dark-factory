# PRD: local embedder cutover — fused-memory embeds with granite-embedding-english-r2, every in-scope vector re-embedded

PRD-MARKER:local-embedder-cutover

**Project:** dark-factory (fused-memory). **Status:** active, 2026-10-10.

**Type:** the named follow-up of `plans/local-memory-models-eval-prd.md`. That PRD's
§"Out of scope" and `plans/local-memory-models-eval-decision-embedding.md` §6 call it "the
embedding backfill/migration PRD".

**Approach:** B+H. The work rewrites every persisted memory vector across two stores, so
the cutover seam carries a contract (§6) and two-way boundary tests (§7). The integration
gate is a full dress rehearsal on scratch copies.

**Origin:** esc-3725-1 (task 3725, LME-μ). Leo ruled per axis on 2026-10-10: LLM axis "Stay
on gpt-4o-mini"; embedding axis "Move to granite-r2"; backfill "Full speed, any time". He
ruled the design decisions in §5 D1–D9 the same day, and D14 (the operator-local config
overlay) on 2026-10-11. D10–D13 are author decisions recorded with their grounds.

**Review:** an adversarial critic seat (opus, 2026-10-10) found 8 blocking issues in the
first draft: the quiesce could be undone by restarters, scope was undefined against 161
live graphs, the census basis was wrong, backup could not restore Qdrant, the Mem0 text
derivation was wrong, no FalkorDB width aggregate exists, the rehearsal would have touched
live units, and one dependency pointed the wrong way. All are folded in here, along with
its should-level findings.

A delta review of the revision (opus) then caught three more, also fixed: a live helper
γ2 would have deleted, leaf signals that depended on downstream work, and no actor for the
post-reboot restore.

**Code anchors** cite symbols, verified against main `ae3047ec3a` (2026-10-10). Main moves
fast; re-locate at implementation time.

## 1. Goal (G1 consumer + user-observable surface)

fused-memory computes every embedding locally, on both the write path and the interactive
query path.

- **Serving:** a permanent vLLM service beside WhisperWriter on the one RTX 3090 serves
  `ibm-granite/granite-embedding-english-r2` (768 dims).
- **Data:** every in-scope vector that fused-memory searches is 768-wide in that one space.
  "In scope" is defined once, by C0.
- **Thresholds:** every embedder-space threshold is re-derived for that space, and each
  carries a provenance stamp that is checked against the live embedder before use.

**Consumers** are fused-memory's own MCP surfaces, which every agent and orchestrator
already calls: `search`, `search_tasks`, `add_memory` (and its near-duplicate guard), and
the Graphiti write path's dedupe search.

**What an operator sees:**

- `get_status` reports an `embedder` block: model `granite-embedding-english-r2`, endpoint
  `http://127.0.0.1:8420/v1`, dimensions 768, `reachable: true`, and `width_guard: ok` for
  each in-scope family.
- `python -m fused_memory.maintenance.reembed census` against the live stores shows one
  width bin, 768, for every in-scope container. Its counts equal the post-quiesce reference
  census, and it lists every excluded container with its reason.
- A paired known-item smoke over live memories holds R@10 at ≥ 0.9× its pre-cutover value
  in each family, with no errors and no empty store legs.
- Query-embed p95 falls from about 263 ms to about 30 ms (ι report §2). The OpenAI
  embeddings dependency is gone; the LLM stays on OpenAI by ruling.
- A band or threshold consumer whose provenance stamp names a different embedder refuses
  loudly instead of applying the stale value.

## 2. Background

**The eval and the ruling.**
- λ's decision record measured granite on with-indices known-item recall (the production
  configuration): R@5 170/199, R@10 179/199, mrr 0.718, query p95 30 ms. The incumbent
  scored 163, 175, 0.681 and 263 ms.
- ι projected the backfill in minutes, not hours (ι report §7).
- Leo ruled the move.

**Residency, measured** (`plans/local-memory-models-eval-embedder-residency-2026-10-10.md`,
SAFE on all five pre-registered criteria):
- The granite engine plateaued at 952 MiB.
- The worst device peak was 12550 MiB, 51% of the card. That figure includes a second
  Whisper large-v3 copy decoding on top of production; the production-equivalent peak is
  about 8.4 GiB.
- No OOMs occurred, and transcripts stayed exact.
- Query-rate load moved dictation latency by 0.96–0.97×, which is noise.
- A full-speed backfill makes dictation about 2.5× slower while it runs. Leo accepted
  this ("Full speed, any time").

**Live census** (read-only, 2026-10-10):

| family | populated containers | vectors | width |
|---|---|---|---|
| Qdrant `fused_*` | 8 of 11 collections | 80,952 points | 1536, Cosine (Qdrant normalises on upsert) |
| Qdrant `task_curator_*` | 12 collections, some unrooted or pollution | 15,826 points | 1536, Cosine |
| FalkorDB, 8 registered-project graphs | 8 | 150,644 (Entity name + RELATES_TO fact) | 1536, stored raw |
| FalkorDB, other graphs (`GRAPH.LIST` returns 161) | ~84 `_probe*`/`_test_*` + `evalmem_lme_ref_incumbent_a` | ~2.8k | 4 (probes), 1536 (eval reference) |

`dark_factory` also holds 7,239 RELATES_TO edges with a NULL `fact_embedding` (out of
scope, §11). No graph carries a VECTOR index. Similarity is brute-force
`vec.cosineDistance` in Cypher (`FalkorEdgeSearch.edge_similarity_search`).

**Substrate facts that shape the design** (survey and critic, 2026-10-10):

1. **One wrong-width vector breaks a whole graph.** `vec.cosineDistance` over mismatched
   widths raises "Vector dimension mismatch" for the whole query. A single wrong-width
   vector therefore fails every cosine search, and every Graphiti dedupe search, on that
   graph.
2. **FalkorDB has no width function.** `size()`, `length()` and `toString()` all fail on a
   Vectorf32. The existing stale-width read (`GraphitiBackend._query_stale_embeddings`)
   pages every vector to the client. Two server limits matter:
   - the module's default query `TIMEOUT` is 1000 ms, which killed whole-graph scans under
     load (an explicit TIMEOUT finished them in 0.12–0.22 s);
   - `RESULTSET_SIZE` is 10,000.
   Use the existing helpers `_paged_ro_query` and `apply_incompleteness_policy`.
3. **Qdrant collections never resize.**
   - Mem0's `QdrantDB.create_col` skips any existing name, though it still re-creates
     filter indexes.
   - A recreate drops the payload indexes (`actor_id`, `agent_id`, `user_id`, `run_id` are
     live on `fused_dark_factory`).
4. **Mem0 will send `dimensions`.** At a non-1536 `embedder.dimensions`, Mem0's
   `OpenAIEmbedding` sends a `dimensions` request parameter, because fused-memory emits
   `embedding_dims` whenever the value differs from `_MEM0_DEFAULT_EMBEDDING_DIMS`. vLLM
   rejects that for granite: a live probe during the survey returned HTTP 400. No other
   client sends it: Graphiti `build_embedder`, `TaskCurator._get_embedder` and
   `calibrate_write_triage.build_embed_fn` already request native width.
5. **The embedder endpoint is already independent.** `embedder.providers.openai.api_url`
   is separate from the LLM's at config level; only the shared `${OPENAI_API_URL}` default
   couples them, so a literal decouples them. `build_embedder` returns None on an empty
   key, `TaskCurator._get_embedder` raises, and `build_embed_fn` falls back to the env key.
   A local endpoint therefore needs a non-empty placeholder key.
6. **Production text derivation, verified per family:**
   - **Mem0** embeds `data.replace('\n', ' ')` (`mem0.embeddings.openai.OpenAIEmbedding.embed`)
     and stores `payload['data']` raw. 373 of 2,000 sampled `fused_dark_factory` points
     contain newlines.
   - **Graphiti's add_episode bulk path** (`create_entity_node_embeddings` /
     `create_entity_edge_embeddings`) embeds `name` / `fact` RAW.
     `EntityNode.generate_name_embedding` and `EntityEdge.generate_embedding` replace
     newlines. No live graph text currently contains a newline, so the two agree today.
   - **The curator** embeds `task_curator.embedding_text(title, description, files_to_modify)`, composed from the task
     store across all tags.
7. **`reindex.py` is unusable.**
   - It embeds a different text (`f'{name} {summary}'`).
   - It covers one graph and never touches Qdrant.
   - It embeds one item at a time.
   - It replays every dead letter.
   - It has no production caller.
   It is the sole production caller of `GraphitiBackend.query_stale_*_embeddings` /
   `_query_stale_embeddings`, `update_edge_embedding` and `drop_vector_indices`.
   `update_node_embedding` also serves the live MCP tools `ensure_entity_node` and
   `rename_entity_node`, so it stays. Tests and `backends/falkor_indices.py` reference the
   reindex-only helpers. `arm_harness/graph_copy.py::reembed_graph` is a second batched re-embed with
   its own derivation.
8. **Writes can restart behind a stop.** Four mechanisms restart fused-memory, and none is
   disabled by stopping the unit:
   - the watchdog's `fused_memory_liveness_pass`, after three port-down ticks (it skips a
     unit only when `is_unit_enabled` is false);
   - every orchestrator's post-merge coordinator, via `scripts/restart-fused-memory.sh`
     (`systemctl --user restart`);
   - the watchdog's fused-memory staleness backstop;
   - `Wants=fused-memory.service` in eight installed `orchestrator-*.service` units.
   `fused-memory.socket` socket-activates the service. Two timers also run:
   `fused-memory-flag-marker-sweep` and `memory-metadata-coverage-census`.
9. **Neither queue carries vectors.** `data/queue/write_queue.db` holds no vector-shaped
   payloads in pending, in-flight or dead rows, and `planned_episodes.db` holds only
   uuid/project rows. Some maintenance tools do write vectors byte-exact:
   `maintenance/cross_graph_move.py::_embedding_set_clause`, and recovery dumps such as
   `scripts/cgl_eta_auto_apply_impl.py`.
10. **Embedder-space thresholds, with their live state:**

    | threshold | value | home | status |
    |---|---|---|---|
    | near-duplicate guard | 0.92 | schema default, absent from `config.yaml` | live; "Mem0's own cited ~0.92", uncalibrated |
    | `search_tasks` | 0.3 | parameter default in three signatures, no config key | live |
    | `search_tasks` as passed by skills | 0.6 | `skills/review/references/phase3-triage.md`, `skills/review-all/SKILL.md` | live |
    | curator zot-duplicate | 0.65 | schema default | live |
    | write-triage `t_high` / `t_low` / procedural | 0.8868 / 0.5230 / 0.8390 | config, calibrated in 1536 space | dormant for triage; LIVE as `audit_duplicate_memories --apply`'s destructive ANN cutoff |
    | `write_triage_project_calibrations.dark_factory` | observations 0.8578 | config | live, also read by the audit |
    | Graphiti cosine floor (`DEFAULT_MIN_SCORE` 0.6, cosine > 0.2) | — | Graphiti | carried through ι's eval, which admitted granite under it |

    The calibration report's provenance already carries `embedder_model` and
    `embedder_dimensions`. `embedder.*` is restart-tier; the near-dup and write-triage keys
    are hot-reloadable (`config/reload.py`).

## 3. Activation status

Active. Every governing ruling is made (Origin), and residency was measured SAFE. Every
leaf builds on landed substrate (§8).

## 4. Sketch of approach

1. **A permanent embedder service** (α1 code, α2 operator install).
   - It is a third service in `fused-memory/docker/docker-compose.yml`: vLLM pooling,
     image pinned by digest, the GPU share bounded as in Phase 1, `127.0.0.1:8420`,
     Hugging Face offline mode, bounded restarts.
   - fused-memory.service's `ExecStartPre` brings it up with a bounded, never-blocking wait.
   - A hold drop-in (`ConditionPathExists=!<cutover lock>`) installed on
     `fused-memory.service` and `.socket` makes the unit unstartable while a cutover holds
     the lock. Otherwise it is inert.
2. **Embedder wiring, scope and the width guard** (β1, β2).
   - **C0**, the single in-scope enumerator.
   - Native-width requests everywhere.
   - A startup probe of the served model and its width.
   - The C2 width guard: a typed probe per container that refuses mismatches loudly.
   - Width assertions at direct vector-write chokepoints.
   - `get_status` surfacing.
   - A loud `search_corpus` failure.
   - β1 adds the operator-local config overlay (D14) and the endpoint and probe pieces;
     β2 adds scope, the guard and the chokepoints.
3. **The re-embed engine** (γ1, γ2): `fused_memory.maintenance.reembed`.
   - γ1: the derivation module, the text-keyed `prepare` cache, `census` and `prepare`.
   - γ2: a record-keyed `backup` with full Qdrant points and params, and `apply`.
   - γ2 also retires `reindex.py` and its orphaned helpers.
4. **The cutover command** (δ): `reembed cutover`.
   - It runs under a transient systemd unit with a target profile, `live` or `scratch`.
   - Its phases are P0 preflight → P1 hold, quiesce, reference census and backup →
     P2 delta → P3 apply → P4 verify → P5 overlay flip → P6 release, start and checks.
   - A failure before P6 auto-restores. A dead owner triggers a bounded restore.
   - Nothing is committed to git inside the window. After the 14-day soak, λ folds the
     overlay into tracked config through the merge queue, and θ removes the overlay and
     the backups.
5. **Thresholds** (ε).
   - Adds a `search_tasks` config key.
   - Adds provenance stamps beside every embedder-space value group, checked by one guard.
   - Adds a quantile mapper with a defined gated population per threshold.
   - The write-triage bands are recalibrated in full after the live cutover (η).
6. **Rehearsal** (ζ). A dress rehearsal with the `scratch` profile, against point-in-time
   copies of the live stores and a scratch fused-memory instance. It produces the committed
   threshold map. Then comes the live window (η), the post-soak fold (λ), the cleanup (θ),
   and documentation corrections (κ1, κ2).

## 5. Resolved design decisions

- **D1 — Arm.** granite-embedding-english-r2, 768 dims, no query prefix. Leo, 2026-10-10.

- **D2 — Cutover shape: pre-embed, then a short window.** Leo chose this over
  stop-the-world and over shadow-and-swap.
  - `prepare` runs live at full speed.
  - The window embeds only texts written since `prepare`, then rewrites vectors from the
    cache. That includes the curator, whose texts `prepare` caches too.
  - The window is estimated at 3–5 min; ζ measures it.

- **D3 — Serving: a compose service.** Leo chose this over a dedicated systemd unit.
  - vLLM is the measured stack (eval PRD Open Q1 resolves to vLLM).
  - α's `lms-arm@` units stay eval-only.

- **D4 — Thresholds: recalibrate where labels exist, quantile-map the rest** (Leo).
  - **Recalibrated in η:** the pooled write-triage bands and the `dark_factory` per-project
    calibration, by two full `calibrate_write_triage.py` runs. η's row in §10 gives the
    commands.
  - **Quantile-mapped:** near-dup 0.92, `search_tasks` 0.3, the skills' 0.6, and zot 0.65.
    Each is mapped on the score population its consumer actually gates, scored in both
    spaces over the same sampled items, with the side kept as `≥`:

    | threshold | gated population |
    |---|---|
    | near-dup | top-1 neighbour score of sampled `procedural_knowledge` memories within their project |
    | `search_tasks` 0.3 / 0.6 | query→task scores for task titles as queries against the project corpus, top-k |
    | zot | new-task→corpus top-k scores |

    The mapped value is the 1536-space threshold's quantile within that population, read in
    the 768 space, reported with a bootstrap 95% CI. A CI wider than 0.05 cosine escalates
    to Leo instead of landing.
  - **Carried:** the Graphiti cosine floor, which ι validated.

- **D5 — Provenance stamps, one guard.** Every embedder-space value group in config carries
  a stamp `{embedder_model, embedder_dimensions}`:
  - `write_triage` bands, written by `calibrate --write-config`. While an overlay is
    active, calibrate writes the band keys into the overlay (ε), so the effective config
    stays one consistent space;
  - each `write_triage_project_calibrations` entry;
  - a new `embedder_thresholds` group (near-dup, `search_tasks`, zot), written by ε's
    mapper and by the P5 overlay.
  One guard compares a stamp with `config.embedder` before any consumer applies the value,
  and refuses with `CalibrationProvenanceMismatch` on a mismatch.
  `audit_duplicate_memories --apply` exits non-zero and deletes nothing. That covers the
  gap between η's window and the post-cutover recalibration.

- **D6 — Sequencing: the cutover goes first** (Leo). The write-triage PRD's 1536-space flip
  evidence becomes stale at η; that PRD owns re-measuring it. κ1
  files the correction. *Update 2026-10-11: the original "3169 stays on HOLD" no longer
  holds. Task 3169 is done and flipped on 2026-10-11 (`x_flip_ruling_2026_10_11`); see
  `plans/write-triage-flip-readiness-prd.md` §13.3, which supersedes this PRD on the point.*

- **D7 — Rollback: keep backups 14 days** (Leo).
  - The backup is record-keyed and complete: full Qdrant points (id, payload, vector) plus
    collection params and payload-index schema, and per-graph `uuid → vector` with each
    record's text hash.
  - Restore never depends on live state.
  - Rolling back is the `cutover` code path aimed at text-embedding-3-small. A backup vector
    is used where the record's text hash still matches; otherwise the record is re-embedded
    with OpenAI.
  - During the soak, the rollback's P5 removes the overlay. The base config is still the
    1536 identity, with its 1536-stamped thresholds and bands.
  - θ deletes the backups after 14 days.

- **D8 — The width guard absorbs 3867 and 4877** (Leo). Both are cancelled at decompose as
  superseded by β2.

- **D9 — Full speed, any time** (Leo). Nothing is throttled for WhisperWriter.

- **D10 — Text derivation is production's, per family.**
  - Mem0: `data.replace('\n', ' ')`.
  - Graph: raw `name` / `fact` (the add_episode bulk path). Live text has no newlines, so
    this equals the `generate_*` path today.
  - Curator: `task_curator.embedding_text` over all tags.
  One derivation module is the home. Behaviour tests pin it at the HTTP layer against
  Mem0's and Graphiti's own code (B1–B3). `arm_harness/graph_copy.py` keeps its frozen eval
  derivation for reproducibility of committed results, and B2 also pins it against
  Graphiti, so drift in either is caught.

- **D11 — No fallback to OpenAI.** An unreachable embedder makes calls fail loudly.
  `get_status` degrades, and `search_corpus` raises a structured error instead of returning
  `[]`. β2 updates its callers, `search_tasks` and the zot sweep, to surface that error
  rather than read it as "no duplicates". Graphiti writes keep their DurableWriteQueue
  retry. A silent fallback would
  write vectors from another space.

- **D12 — Storage form.** FalkorDB stores raw granite vectors (norms ≈ 30). Qdrant Cosine
  collections normalise on upsert. Both search by cosine, so no normalisation is added
  (§11).

- **D13 — Scope is a registry, not a scan.**
  - **In scope:** C0 enumerates in-scope containers from the project registry. That is the
    project ids of fused-memory's known roots, plus an explicit `embedder_scope.extra_projects`
    list, plus every `fused_*` Mem0 collection, since Mem0 points carry their own text and
    need no root.
  - **Excluded:** every other vector-bearing container must match an
    `embedder_scope.excluded` pattern with a reason: `evalmem_*` (frozen eval reference),
    `_probe*`, `_test_*`, curator pollution. Excluded containers are reported by census,
    ignored by the guard, and never touched.
  - **Unrooted:** `task_curator_*` collections with no resolvable root are derived data.
    B backs them up like every other collection P3 touches, and I4 restores them. A
    successful cutover drops them at P3; they self-heal through `_maybe_backfill_corpus` at
    the new width when that project next appears.
  - **Refusal:** P0 refuses on any unclassified container. Deleting pollution is out of
    scope.

- **D14 — The live identity flips through an operator-local overlay; the tracked commit
  follows the soak.** Leo, 2026-10-11, choosing this over a commit to tracked config inside
  the window.
  - **The overlay** is a host-local YAML outside the repository
    (`~/.config/fused-memory/embedder-overlay.yaml`). It is referenced by a
    `fused-memory.service` drop-in setting `FUSED_MEMORY_CONFIG_OVERLAY`, the same pattern
    as `DASHBOARD_KNOWN_PROJECT_ROOTS`.
  - **The loader** (β1) deep-merges it over `config.yaml` for an allowlist of key paths
    only: `embedder.*`, `embedder_thresholds.*`, the `write_triage` band keys plus their
    stamp, and `write_triage_project_calibrations.*`.
  - **Refusals:** a key outside the allowlist refuses start, and so does a set variable
    whose file is missing or unparseable. Each refusal is structured; the loader never
    silently falls back to base.
  - **Visibility:** `get_status.config_overlay` reports the path, the keys and the mtime, so
    the period when identity has two homes is visible.
  - **Bound:** λ ends that period by folding the overlay into tracked config through the
    merge queue, as a dated leaf at η + 14 d. θ then removes the overlay once its values
    equal tracked config.
  - **Failure:** nothing is committed in the window. A P5 failure removes the overlay and
    the drop-in and takes the I4 auto-restore.
  - **Extension (Leo, 2026-10-11, at decompose):** the drop-in goes on
    `fused-memory.service` and on both timer services
    (`fused-memory-flag-marker-sweep.service`, `memory-metadata-coverage-census.service`).
    Their jobs build `MemoryService` / `Mem0Backend` straight from tracked `config.yaml`,
    which keeps the 1536 identity for the soak. Leo chose this over "guard only" (both jobs
    refuse nightly for 14 days) and over pausing the timers. δ's `live` profile lists all
    three units; a rollback, the auto-restore and θ's cleanup remove all three.

## 6. Contract

### C0 — in-scope enumerator (β2)

```
in_scope(config, registry) -> Scope{
  qdrant_mem0: [collection],           # every fused_* collection
  qdrant_curator: [(collection, project_id, project_root)],
  graphs: [graph_name],                # registry project ids that exist as graphs
  excluded: [(container, pattern, reason)],
  unrooted_curator: [collection],
  unclassified: [container]}           # non-empty => P0 refuses; guard reports
```

The registry is structured data, never parsed from a unit file:
- **In the server:** the process env, `PROJECT_ROOT` + `DASHBOARD_KNOWN_PROJECT_ROOTS`, which
  `get_status` exposes as `embedder.scope.registry`.
- **`live` profile:** P0 reads that registry from the running service's `get_status` and
  journals it; later phases use the journaled copy.
- **`scratch` profile:** the registry is a field of the profile file.

Each root resolves to its `project_id` through the same discovery the dashboard uses
(canonical `dark-factory-orchestrator.yaml`, legacy spellings as fallback), and
`embedder_scope.extra_projects` adds more. A root that does not resolve makes its
containers `unclassified`, so P0 refuses. C0 is the only definition of scope; the guard,
census, prepare, backup, apply and P4 all call it.

### C1 — `reembed` CLI and the cutover phases

```
reembed census   [--profile P] [--json]                    # read-only; distinct widths per in-scope container, paged, explicit TIMEOUT
reembed prepare  --target SPEC --cache C [--profile P]     # reads stores and task store; writes only C
reembed backup   --out B [--profile P]                     # record-keyed, complete (D7)
reembed apply    --target SPEC --source C|B [--profile P]  # writes stores at SPEC.dims
reembed cutover  --target SPEC --cache C --backup B --profile live|scratch [--dry-run] [--resume [--restore-only]]
```

- **`SPEC`** is `{model, base_url, dims}`, validated against `/v1/models` and a probe embed
  of native length `dims`.
- **The cache** `C` is keyed by `(model, family, sha256(derived text))`, records `model` and
  `dims`, and refuses a lookup under another model.
- **The profile** names the systemd units to hold, the lock path, the MCP endpoint, the
  config path, the store endpoints, and the overlay and drop-in paths. `live` is the
  installed units, :8002, the tracked config and the host overlay. `scratch` is ζ's
  scratch instance, with scratch config and overlay paths.

`cutover` runs under `systemd-run --user --unit=fused-memory-cutover`. Each phase appends a
journal record `{phase, status, started_at, pid, boot_id, deadline, counts, error}`.

| Phase | Does | Refuses / fails when |
|---|---|---|
| P0 preflight (service live) | Probe `SPEC`; check cache model; check backup path empty; C0 classification; scan both queues for vector-shaped payloads; confirm the active identity (base plus any overlay) equals the source spec; run the **baseline known-item smoke** through the profile's MCP endpoint; run an informational census | Any check fails, or `unclassified` is non-empty. Nothing is touched |
| P1 hold + quiesce + backup | Write the lock; `disable --now` socket and service; stop both timers; confirm inactive. Take the **reference census** after quiesce (the only comparison basis). Write backup B and verify it covers the reference census record-for-record | A unit is still active, or B is short. Release the hold and stop |
| P2 delta | Re-derive every in-scope text; embed cache misses with `SPEC` at full speed (D9) | Endpoint errors past a bounded retry → auto-restore |
| P3 apply | Re-confirm inactive. Qdrant: recreate each in-scope collection at `SPEC.dims` with B's params and payload-index schema, then chunked upsert from C, curator included, using `TaskCurator._point_id`'s convention (`uuid5(NAMESPACE_URL, f'{project_id}/{task_id}')`). Drop unrooted curator collections. Graphs: batched in-place `SET` with directed patterns, paged, explicit TIMEOUT | Any write error → auto-restore |
| P4 verify | Re-confirm inactive. Census: every in-scope vector is `SPEC.dims` wide. Mem0 and graph counts equal the reference census; curator upserted equals tasks-with-title per project, with errors 0 | Any mismatch → auto-restore |
| P5 overlay flip | Write the overlay atomically (temp file plus rename): the target `embedder` block, plus the `embedder_thresholds` group and its stamp from ε's committed C4b map. Write the drop-in and `daemon-reload`. Validate the effective config through fused-memory's own loader in a subprocess. On a rollback whose target is the base identity, remove both instead | Validation fails → remove both, `daemon-reload`, auto-restore |
| P6 release + start + checks | Remove the lock; enable and start socket and service; wait for `get_status` to show `reachable` and the guard `ok`; run the post-cutover smoke | Start or `get_status` fails → the I4 rollback path: re-hold, re-quiesce, `cutover` back with B. A smoke below bound is reported to the attended gate and not auto-rolled back |

**Invariants:**
- **I1.** No running fused-memory observes mixed widths. The unit is unstartable from P1
  to P6 (hold drop-in plus disable). P3 and P4 re-confirm. β2's guard refuses writes if a
  start ever meets a mismatch.
- **I2.** No store write before B is verified complete.
- **I3.** Phases are idempotent. A journal whose `pid`/`boot_id` is dead is a crashed run,
  never "in progress".
  - If the cutover unit fails before P6 completes, `OnFailure=` runs `--resume
    --restore-only` once.
  - P1 also installs a persistent user timer, `fused-memory-cutover-guard.timer`
    (`OnBootSec=2min`, `OnUnitActiveSec=1min`), and P6 removes it. Each run executes
    `reembed cutover --resume --restore-only --if-owner-dead`. When the journal is pre-P6-complete
    and its owner is dead past `deadline` (or `boot_id` changed), it restores, re-enables
    and starts on the old config, then removes itself. That bounds the hold across a dead
    process, a killed user manager or a reboot.
- **I4.** Before P6 starts the unit, auto-restore removes any overlay and drop-in P5 wrote,
  runs `apply(source=B)` at the source width, then releases and starts on the base config.
  Once P6 has started the unit, it may have accepted writes, so its failures do NOT use
  `apply(B)`. That would leave post-start 768 vectors among restored 1536 ones. They take
  the rollback path instead: re-hold, re-quiesce, then `cutover` with `target = source
  spec` and B as the cache, which embeds any record missing from B with the source
  embedder. Both are the operator rollback's code.
- **I5.** Text derivation has one home: the engine's derivation module (D10).
- **I6.** FalkorDB reads page explicitly through `_paged_ro_query` /
  `apply_incompleteness_policy`, with an explicit TIMEOUT. A full page is never taken as
  complete, and edge matches use directed patterns.

### C2 — width guard (β2)

- **Probe:** for each in-scope graph and property, one typed probe:
  `MATCH … WHERE v IS NOT NULL RETURN count(vec.euclideanDistance(v, vecf32(<W zeros>)))`
  with an explicit TIMEOUT. It errors if and only if any non-NULL vector's width ≠ W. For each in-scope
  Qdrant collection, its configured vector size.
- **When:** at startup, off the event loop's hot path. `get_status` reads the cached result.
- **On mismatch:** `EmbedderWidthMismatch{family, container, configured, live: "≠W" | size}`.
  Mem0 and Graphiti writes and searches for that scope are refused with it. The guard never
  auto-repairs.
- **Distinct widths:** enumerated only by the offline census.
- **Write chokepoints:** every direct vector-write site checks `len(v) == W` before the
  `SET`. β2 enumerates them by grepping `vecf32(` SET clauses across `fused-memory/src` and
  `fused-memory/scripts`: at least `GraphitiBackend.update_node_embedding`,
  `cross_graph_move._embedding_set_clause`, and the recovery-dump re-apply path of
  `scripts/cgl_eta_auto_apply_impl.py`.

### C3 — embedder endpoint (β1)

- **No `dimensions` parameter**, from any client: Graphiti `build_embedder`, the Mem0 config
  dict (stop emitting `embedding_dims` to the embedder), `TaskCurator._get_embedder`, and
  `calibrate_write_triage.build_embed_fn`. γ's engine asserts the same.
- **The endpoint** is a literal `embedder.providers.openai.api_url` with a placeholder key.
  It is set in the overlay at P5; β1 adds no URL coupling.
- **Startup probe:** the served model id equals `embedder.model`, and native width equals
  `embedder.dimensions`. A mismatch degrades status with a structured error; it never
  coerces.

### C4 — provenance stamps (ε)

`{embedder_model, embedder_dimensions}` sits beside each value group (D5).
`check_embedder_provenance(stamp, config.embedder)` raises
`CalibrationProvenanceMismatch{group, stamp_model, stamp_dims, config_model, config_dims}`.
The callers are `server/write_triage.py`, `near_duplicate_guard`, `search_tasks`, the zot
sweep and `audit_duplicate_memories`.

**C4b — the threshold-map artifact:**
`fused-memory/calibration/embedder-threshold-map-<date>.json`:

```
{source:{model,dims}, target:{model,dims},
 thresholds:[{name, config_key, source_value,
              population:{rule, n, sample_seed},
              quantile, target_value, ci95:[lo,hi], escalated:bool}],
 computed_on:{profile, census_sha}}
```

P5 reads only non-escalated entries; any escalated entry blocks P5.

## 7. Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Graph name parity | Graphiti's `create_entity_node_embeddings` (bulk) and `generate_name_embedding` run against a fake `/v1/embeddings` recording bodies, for a newline-free name | The engine's derivation equals the recorded input; `graph_copy.reembed_graph` equals it too |
| B2 | Graph fact parity | Same for `create_entity_edge_embeddings` / `generate_embedding` | Equal, the engine and `graph_copy` both |
| B3 | Mem0 text parity | `AsyncMemory.add(infer=False)` with the fake endpoint, for text containing newlines | The recorded body equals the engine's `data.replace('\n', ' ')` for the stored `payload['data']` |
| B4 | No `dimensions` sent | Each of the four existing clients embeds once at dims 768 against the fake (γ adds the engine) | No body contains `dimensions` |
| B5 | Endpoint width or model mismatch | Fake returns 1024 wide, or a different model id | Status degraded, error structured |
| B6 | Qdrant width guard | An in-scope 1536 collection, config 768 | Refused, naming the collection |
| B7 | FalkorDB width guard | An in-scope graph with one 1536 vector among 768s | The typed probe errors → refused, naming the graph |
| B8a | Scope (β2) | An unclassified vector-bearing graph; separately, one matching an excluded pattern | C0 reports it in `unclassified` and the guard reports it; the excluded one is listed with its reason and ignored by the guard |
| B8b | Scope refusal (δ) | Same unclassified graph | P0 refuses, naming it; nothing touched |
| B9 | Dry-run | Fixture stores | No writes; plan, preflight verdicts and delta count printed, under both profiles |
| B10a | Apply round trip (γ2) | Fixture stores at 1536 → `backup` B → `apply` 768 from cache → `apply source=B` | FalkorDB vectors byte-equal; Qdrant vectors within 1e-6 per component (normalised); payloads, ids and payload indexes equal, unrooted curator collections included |
| B10b | Cutover round trip (δ) | Fixture stores at 1536 → `cutover` 768 → `cutover` back with B | Same tolerances as B10a |
| B11 | Auto-restore | Fault injected mid-P3 | Stores restored per B10a tolerances; the service starts on the old config; the journal records it |
| B12a | Dead owner | Cutover process killed in P3 | `OnFailure` restore runs once; the unit is released and started on the old config |
| B12b | Reboot mid-window | Journal pre-P5 with a different `boot_id` (simulated); `OnFailure` never fired | The guard timer's next run restores, re-enables and starts on the old config, then removes itself |
| B13 | Hold (δ fixture: a transient user unit carrying the drop-in property; observed for real in ζ on the scratch instance) | A `systemctl --user start` / `restart` and a socket connection during P3 | The unit does not start (condition failed); P4 proceeds |
| B14 | Traffic between P0 and P1 | Writes and deletes land after P0 | No restore triggered: P4 compares with the post-quiesce reference census |
| B15a | FalkorDB paging, read (γ1) | A fixture graph with more than 10,000 edges, including incoming edges on matched nodes | `census` and `prepare` cover every edge; counts equal |
| B15b | FalkorDB paging, write (γ2) | Same fixture | `apply` re-embeds every edge; post-apply census equals pre |
| B16 | Provenance | Stamp granite/768 with config 3-small/1536, and the reverse | Each consumer refuses; `audit --apply` exits non-zero and deletes nothing |
| B17 | P5 failure | Overlay validation fails (an injected bad key) | Overlay and drop-in removed; auto-restore; the service is back on the base config |
| B19 | Overlay loader (β1) | (a) variable set, file missing; (b) a key outside the allowlist; (c) a valid overlay | (a) and (b) refuse start with structured errors; (c) gives the merged effective values, and `get_status.config_overlay` lists path, keys and mtime |
| B20 | Calibrate under an overlay (ε) | Overlay active; `calibrate --write-config` | Band keys and stamp land in the overlay; tracked `config.yaml` unchanged |
| B18 | Chokepoint | `cross_graph_move` given a 1536 vector, config 768 | Refused before the `SET` |

## 8. Pre-conditions (G3 substrate, verified 2026-10-10)

- **Image and weights.** The vLLM image `vllm/vllm-openai:v0.26.0@sha256:ffb2d59b…` and the
  granite weights are in the local Hugging Face cache. The pooling share 0.062
  (`lms_serve.py::_memory_share_for`) was measured resident (Phase 1). Docker Compose
  v5.1.2 and the NVIDIA runtime are present.
- **Compose and start-up hooks.** `fused-memory/docker/docker-compose.yml` exists, and the
  installed `fused-memory.service` `ExecStartPre` already brings up `falkordb`/`qdrant` and
  waits for their ports (template `scripts/fused-memory.service.template`). Port 8420 is
  free.
- **Stores and engine primitives.**
  - Qdrant is 1.17.1, which supports recreate, chunked upsert and payload-index creation.
  - The FalkorDB functions `vecf32` and `vec.euclideanDistance` exist, and a width
    mismatch raises. `_paged_ro_query` and `apply_incompleteness_policy` exist.
  - `task_curator.embedding_text` exists (module-level), and the task store is readable across all tags
    (the `BackfillManager.prune` pattern).
- **No vectors in either queue** (§2 fact 9). P0 still scans them.
- **Quiesce surface:** the socket, the service, the two timers and the four restarters
  (§2 fact 8). A unit disabled with a failed `ConditionPathExists` cannot be started by any
  of them; B13 proves it.
- **No other vector writer outside fused-memory** (repo grep: only setup/import scripts and
  tests). Tests and probes do write probe graphs and `task_curator_test_*` to the live
  stores, and D13 excludes them.
- **Mem0's `mem0migrations` collection** (1536, 1 point). γ1 verifies whether Mem0 writes
  vectors there at the configured width. If it does, C0 includes it; if not, it is
  excluded with that reason.
- **Calibration provenance** already carries `embedder_model` and `embedder_dimensions`.

## 9. Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/local-memory-models-eval-prd.md` | consumes | λ decision records, ι evidence, Phase 1 residency artifact | eval PRD (complete); κ1 adds a pointer | wired |
| `plans/write-triage-flip-readiness-prd.md` (3169 flipped 2026-10-11; see its §13.3) | produces | write-triage bands and their stamps; the slates and retrievals its flip evidence measured | this PRD: η's recalibration, ε's guard, κ1's correction. Write-triage PRD: re-measuring its flip evidence in the granite space | queued (κ1) |
| `docs/prds/falkordb-index-provisioning.md` | none today | No vector index exists; any future one takes its width from `embedder.dimensions` | falkordb-index-provisioning | note |
| `skills/review*` (quality-findings dedup) | consumes | `search_tasks(score_threshold=0.6)` | this PRD: ε maps, κ2 updates | queued |
| tasks 3867, 4877 | superseded | Mem0 and curator width guards | this PRD (β2) | cancel at decompose |
| task 6406 | independent | own text-embedding-3-small path in `scripts/legibility` | 6406 | out of scope |

## 10. Decomposition plan

Every leaf is authored against `docs/code-quality.md`. Labels map to task ids at decompose.

| | Title | Modules / files | Observable signal | Prereqs |
|---|---|---|---|---|
| **α1** | Define the granite embedder compose service and the fused-memory hold drop-in | `fused-memory/docker/docker-compose.yml`, `scripts/fused-memory.service.template`, new hold drop-in template, `OPERATIONS.md` (embedder section) | `docker compose -f … config` validates; the service binds `127.0.0.1:8420` with `--gpu-memory-utilization 0.062`, the digest pin and offline mode; the template carries the bounded embedder wait; the drop-in carries `ConditionPathExists=!<lock>` | — |
| **β1** | Config overlay loader, native-width requests, startup endpoint probe, `get_status` embedder and overlay blocks | `fused-memory/src/fused_memory/config/` (loader, schema), `backends/mem0_client.py`, `server/` (startup probe, `get_status`) | B4, B5 and B19 pass. Unlocks β2 | — |
| **β2** | C0 scope enumerator, C2 width guard, chokepoint asserts, loud `search_corpus` | new `embedder_scope`/guard module, `backends/graphiti_client.py`, `middleware/{task_curator.py, curator_zot_duplicate_sweep.py}`, `server/tools.py`, `maintenance/cross_graph_move.py`, `scripts/cgl_eta_auto_apply_impl.py` | B6, B7, B8a and B18 pass. Unlocks γ1, ε, δ, α2. Its live signal is observed in α2 | β1 |
| **α2** | Install and verify the embedder on the host (attended operator step) | host: compose up, surgical patch of the installed unit (new directives only, backup and diff) plus the hold drop-in | (1) `curl 127.0.0.1:8420/v1/embeddings` returns a 768-length vector. (2) `ss -ltn` shows loopback only. (3) The Phase 1 probe (committed evidence scripts) re-run on THIS service with WhisperWriter running: engine ≤ 1536 MiB, device peak ≤ 85%, no OOM, appended to the residency artifact. (4) After a fused-memory restart, live `get_status` shows the `embedder` block on the unchanged 1536 deployment with `width_guard: ok` for every in-scope family and every excluded container listed (β1/β2's live signal) | α1, β2 |
| **γ1** | Reembed engine part 1: derivation module, text cache, census, prepare | `fused-memory/src/fused_memory/maintenance/reembed/` (derivation, cache, census, prepare), `arm_harness/graph_copy.py` parity test | B1–B3 and B15a pass. Unlocks γ2, ε. Its live census (in-scope counts matching §2 within drift, all 1536, the excluded list) is observed in ζ's report | β2 |
| **γ2** | Reembed engine part 2: record-keyed backup, apply; retire `reindex.py` and orphaned helpers | `maintenance/reembed/` (backup, apply). Remove `maintenance/reindex.py` and `tests/test_reindex.py`. In `backends/graphiti_client.py` remove `query_stale_*` / `_query_stale_embeddings`, `update_edge_embedding` and `drop_vector_indices`, keeping `update_node_embedding`, a C2 chokepoint with live callers. Also `backends/falkor_indices.py`, and the tests that reference the removed helpers (`test_drop_vector_indices*.py`, `test_graph_read_pagination.py`, `_fm_helpers.py`, `test_fm_helpers.py`, `test_falkor_indices.py`, `test_local_endpoint_base_url_integration.py`) | B10a and B15b pass on fixture stores. No definition or call site of the removed symbols remains in `fused-memory/src` or `fused-memory/tests`. Unlocks δ | γ1 |
| **ε** | `search_tasks` config key, provenance stamps + guard (C4), quantile mapper (D4) | `config/schema.py`, `server/{write_triage.py,near_duplicate_guard.py,tools.py}`, `middleware/{task_curator.py,curator_zot_duplicate_sweep.py}`, `scripts/{calibrate_write_triage.py,audit_duplicate_memories.py}`, new mapper script | B16 and B20 pass. The mapper produces a C4b artifact on a fixture. Live `audit_duplicate_memories` (no `--apply`) under matching stamps reports exactly as before. Unlocks δ, ζ | γ1, β2 |
| **δ** | `reembed cutover`: profiles, phases P0–P6, hold, journal, transient unit with dead-owner restore, auto-restore, P5 semantics, smoke | `maintenance/reembed/cutover.py` (+ smoke module), systemd transient-unit wiring | B8b, B9, B10b, B11, B12a/b, B13 (transient-unit fixture), B14 and B17 pass on fixture stores. Unlocks ζ, which runs the live `--dry-run` and the real-systemd B13 | α1, β2, γ2, ε |
| **ζ** | Dress rehearsal on scratch copies (integration gate) | scratch compose project on alternate ports, loaded from point-in-time copies (FalkorDB RDB, Qdrant snapshots); scratch fused-memory instance as a transient user unit; no product code | Committed `plans/local-embedder-cutover-rehearsal-<date>.md`: the live `reembed census` and live `cutover --profile live --dry-run` P0 verdicts (no side effect); measured window; reference/P4 census; paired smoke per family pre/post; `search_tasks` self-retrieval pre/post, the B10b round trip on real-shaped data, an injected-fault auto-restore. Also commits `fused-memory/calibration/embedder-threshold-map-<date>.json` computed on the copy. **Predicate:** window ≤ 10 min AND widths uniform AND counts equal AND smoke ≥ 0.9× per family AND round trip within B10a tolerance AND no escalated threshold; otherwise escalate to Leo with the numbers | α2, δ |
| **η** | Live cutover (born-at-L2 attended operator gate) | live stores, host overlay (P5), calibration reports | Live `get_status`: granite, 768, reachable, guard ok, with `config_overlay` listing the overlay. Live census: all in-scope containers 768. Two full `calibrate_write_triage.py` runs write granite-stamped bands into the overlay: a pooled `--write-config` run, then a separate `--project-id dark_factory --write-project-config --fixture <...> --report-path <...>` run (without `--project-id`, the second run silently calibrates `reify`). Their reports are committed. Write triage is live during η's window (3169 flipped 2026-10-11), so live writes meet 1536-stamped bands until the recalibration; ε's `CalibrationProvenanceMismatch` guard refuses those bands. Paired live smoke passes. Write-triage owner notified per κ1 | ζ |
| **λ** | Fold the embedder overlay into tracked config after the soak | `fused-memory/config/config.yaml` | Merged through the merge queue: `config.yaml` carries the overlay's values exactly (embedder block, `embedder_thresholds`, bands, project calibrations, stamps). Live `get_status` effective values are unchanged. Dated milestone = η done + 14 d | η |
| **θ** | Remove the overlay and the 1536 backups after the fold | host overlay, drop-in, backup path | Deterministic. It first asserts that every overlay value equals the tracked `config.yaml` value at HEAD, then removes the overlay and drop-in (`daemon-reload`) and deletes the backups. After the next restart, `get_status` shows no `config_overlay` and the same effective values | λ |
| **κ1** | Cross-PRD corrections: write-triage evidence stale at η; eval PRD pointer | `plans/write-triage-flip-readiness-prd.md`, `plans/local-memory-models-eval-prd.md` | Both carry a dated amendment naming this PRD and, for write-triage, who re-measures what | — |
| **κ2** | Docs and secondary config after cutover | `SETUP.md`, `README.md`, `CLAUDE.md` (OPENAI key now LLM-only), `DESIGN.md`, `OPERATIONS.md` (rollback runbook, the overlay and its fold), `fused-memory/config/config-docker.yaml` (embedder → compose `embedder` service), `skills/review/references/phase3-triage.md`, `skills/review-all/SKILL.md` (mapped 0.6) | `git grep -n "text-embedding-3-small"` hits only provenance/history files and the rollback runbook; both skills carry the mapped value with a pointer to the C4b artifact | η |

**Serialisation:** β1 → β2 → γ1 → γ2, β2 → ε, α1 → α2, and η → λ → θ. β2 precedes
every leaf that shares its files (`graphiti_client.py`, `task_curator.py`, `tools.py`). γ2 and ε are siblings
with disjoint file sets: γ2 has `graphiti_client.py` and `falkor_indices.py`, ε has
`task_curator.py`, `tools.py` and the write-triage scripts.

**Sizing:** β1, β2, γ1, γ2, δ and ε fall in the 300–1,500 LOC band. α1, κ1, κ2 and λ are
`complexity='simple'`. α2, ζ and η are operational, with α2 and η attended. θ is
deterministic.

## 11. Out of scope

- **The LLM axis.** It stays on gpt-4o-mini. fused-memory keeps its OpenAI key for the LLM
  and the Graphiti reranker.
- **NULL vectors** (7,239 NULL `fact_embedding` edges and any NULL entity embedding) stay
  NULL; the cutover is like-for-like.
- **Deleting probe, test or eval graphs, or curator pollution** (D13 excludes them).
- **Vector index creation** (falkordb-index-provisioning). **Normalisation** (D12).
- **Re-tuning the Graphiti cosine floor.**
- **Graphiti's internal newline inconsistency** between `generate_*` and the bulk path
  (harmless while no text contains a newline).
- **The legibility census embedder (6406), and α's eval substrate.** `lms_vram`'s consumer
  allowlist will now see the production embedder as a foreign GPU holder; it belongs to
  whoever next reuses α.

## 12. Open questions (tactical)

1. **Cache and backup format and location.** Suggested: SQLite under
   `data/embedding-cutover/`, about 0.8 GB each. Decide in γ1/γ2.
2. **Embedder restart and wait.** Suggested: compose `restart: on-failure:5`; `wait-for-port`
   180 s with a `-` prefix, so fused-memory starts degraded rather than blocking. Decide in
   α1.
3. **Textless Mem0 points.** Suggested: census counts them; cutover refuses while the count
   is non-zero unless `--drop-textless`, and journals them. Decide in γ1/δ.
4. **Orchestrator noise during the window.** Suggested: η halts every orchestrator's
   scheduler for the window and resumes it at P6.
5. **Dead-owner `deadline`.** Suggested: 15 min past the P3 start. Decide in δ against ζ's
   measured window.
6. **Calibration wall-clock** is unmeasured. η schedules it after P6.

## 13. G7 advisory walk (design invariants)

| Invariant | How this PRD meets it |
|---|---|
| `contracts-machine-checked` | C0 scope, the C2 typed width probe, the C3 endpoint probe and C4 stamps are code-checked; B1–B3 pin derivation |
| `structured-facts-at-failure` | `EmbedderWidthMismatch`, `CalibrationProvenanceMismatch`, phase journal fields |
| `corroborate-before-acting` | P1 confirms quiesce; P3 and P4 re-confirm; P4 verifies against the post-quiesce census before P5; P5 validates the effective config through the real loader; θ asserts overlay equals tracked config before removing it |
| `storm-escape-required` | Embedder restart bounded; dead-owner restore runs once; held starts fail a condition instead of looping |
| `no-lockstep-duplication` | One derivation home (I5), with graph_copy pinned by B1/B2; one scope home (C0); `reindex.py` and its helpers retired |
| `status-matches-liveness` | Journal pid/boot_id/deadline; `embedder.reachable` from a live probe |
| `holds-owned-and-bounded` | The window hold is owned by the transient cutover unit, bounded by `OnFailure` plus the persistent guard timer that restores after a dead owner or a reboot (I3) |
| `loop-thread-occupancy-bounded` | The guard is one server-side typed probe per container at startup; `get_status` reads the cache; distinct-width scans are offline |
| `one-fact-one-home` | `embedder.dimensions` (width); C0 (scope); stamps make every threshold's space explicit. Embedder identity has two DECLARED homes during the soak (D14, Leo's ruling): an allowlisted overlay visible in `get_status`, bounded by λ's dated fold and θ's equality-checked removal |
| `guards-exercise-behaviour` | B1–B3 run Graphiti's and Mem0's code at the HTTP layer; B6/B7/B15a/B15b run against real store fixtures; B13 runs against systemd (a transient unit in δ, real units in ζ) |
| `no-silent-fail-soft` | No OpenAI fallback; `search_corpus` raises; truncated reads are incomplete; unclassified containers refuse |
| `readers-prove-their-producer` | β2's guard reports on the live 1536 stores at α2 before any cutover |
| `exceptions-owned-or-ratified` | `embedder_scope.excluded` entries each carry a reason; no suppression is added |

## 14. Decompose record (2026-10-11)

Decomposed by a `/team` + `/prd` session. All 14 leaves were filed with
`planning_mode=True`, wired, and released together.

| Leaf | Task | Execution path | Depends on |
|---|---|---|---|
| α1 | 6797 | simple | — |
| α2 | 6801 | operator gate | α1 6797, β2 6800 |
| β1 | 6798 | normal | — |
| β2 | 6800 | normal | β1 6798 |
| γ1 | 6802 | normal | β2 6800 |
| γ2 | 6803 | normal | γ1 6802 |
| ε | 6804 | normal | γ1 6802, β2 6800 |
| δ | 6805 | normal | α1 6797, β2 6800, γ2 6803, ε 6804 |
| ζ | 6806 | normal (integration gate) | α2 6801, δ 6805 |
| η | 6807 | operator gate | ζ 6806, κ1 6799 |
| λ | 6808 | simple, delayed milestone 14 d | η 6807 |
| θ | 6809 | deterministic | λ 6808 |
| κ1 | 6799 | simple | — |
| κ2 | 6810 | simple | η 6807 |

**Manifest.** `plans/local-embedder-cutover-prd.capability-manifest.md` and its YAML
sidecar bind each leaf's capabilities to evidence. The manifest also records every
resolution made at decompose; this section lists only what changes the PRD's own text.

**Cancelled as superseded by β2 (D8):** tasks 3867 and 4877, and task 4294, the other
half that had been coalesced into 4877. 4877's documentation item (a configured `api_url`
beats an ambient `OPENAI_BASE_URL`) moved into κ2.

**One edge added.** η depends on κ1, because η's signal includes notifying the
write-triage owner "per κ1". The other 17 edges are §10's.

**Statements above that did not hold on main, and what the leaves do instead:**

- **§2 fact 9 and C2.** `fused-memory/scripts/cgl_eta_auto_apply_impl.py` writes no
  vector; its recovery dump only reads. β2's chokepoints are
  `GraphitiBackend.update_node_embedding`, `cross_graph_move._embedding_set_clause` and
  `arm_harness/graph_copy.py::_write_vectors`.
- **C0's registry.** The dashboard's discovery is dashboard-only and maps to escalation
  URLs. C0 builds on `fused_memory/models/scope.py`, which reads the canonical config
  only. D13 leaves "vector-bearing" undefined; β2 defines it and seeds the exclusions.
- **C1 never restarts the two timers it stops at P1.** δ's P6 and every restore path
  start them again.
- **C1 does not say how P5 finds the dated C4b file.** `cutover` takes
  `--threshold-map`, validated at P0; ζ runs the mapper before the scratch cutover.
- **C2's "at startup".** The guard binds wherever a backend initialises, in any process,
  and its probe has a third outcome, `unknown`, for a timeout or connection error.
- **D14's "deep-merges".** `embedder.*` merges per key; each calibration unit in the
  overlay replaces the base unit wholesale, so a stamp never vouches for a base value.
- **§4 item 1.** The compose file already has three services; the embedder is the fourth.
- **§10 θ.** `submit_task` requires θ's script to exist at filing, so a fail-loud stub
  `scripts/embedder-cutover-cleanup.sh` was committed with the manifest and δ replaces it.
- **§10 κ2.** The `text-embedding-3-small` grep is scoped to κ2's own files: tracked
  `config.yaml` keeps the string until λ, and task 6406's path keeps it by design.

**Review.** Two critic passes ran at decompose. The first reviewed the raw drafts; the
second reviewed the integrated package and found two of the first round's fixes wrong.
Both sets of findings are resolved in the filed task texts.

