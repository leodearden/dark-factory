# Capability manifest — local embedder cutover

PRD: `plans/local-embedder-cutover-prd.md` (committed `20eb4e2d04`, with its evidence
artifact `plans/local-memory-models-eval-embedder-residency-2026-10-10.md`). Substrate
re-read on main `5238fc26d8`; `fused-memory/`, `scripts/`, `shared/` and `skills/` are
unchanged since `20eb4e2d04`. Machine-readable twin:
`plans/local-embedder-cutover-prd.capability-manifest.yaml` (task ids are stamped by
`commit_planning`).

Binding vocabulary is `skills/prd/references/gates.md` → *Capability Manifest*. Every
mechanical `delivered_check` was linted against `main` with
`shared.delivered_check_polarity.lint_delivered_checks`. The only finding is δ's
`cleanup-script-real-not-stub` (`vacuous_absent`), which is expected until the decompose
stub `scripts/embedder-cutover-cleanup.sh` lands with this manifest. Against a scratch tree
of main plus that stub, the lint returns zero findings.

## Resolutions made at decompose

The lead's decisions R1–R7 change these bindings:

- **R1 (θ's script).** θ is `task_kind='deterministic'`, with `before_done`
  `{script: scripts/embedder-cutover-cleanup.sh, args: [--profile, live], timeout_secs: 600,
  kind: deploy}`. A fail-loud stub ships with this manifest; it exits 1 and prints the
  marker token `EMBEDDER-CUTOVER-CLEANUP-STUB`. δ replaces it with the real action, which
  delegates to a `reembed` cleanup subcommand that δ names. δ's check
  `cleanup-script-real-not-stub` asserts the marker is gone.
- **R2 (κ2's signal).** The `text-embedding-3-small` grep covers only κ2's own files.
- **R3 (edge η→κ1).** η's "owner notified per κ1" binding is upstream.
- **R4 (λ).** λ carries a delayed milestone (`after_secs: 1209600`, 14 days after η is done). It reads the host overlay and
  writes tracked `config.yaml`.
- **R5 (4877 item 1).** κ2 absorbs it.
- **R6 (execution paths).** α2 and η are `operational` gates, so all their checks are
  manual. ζ is a normal full-path task.
- **R7 (ε stamps).** ε stamps every existing group in tracked `config.yaml`. That is ε's
  `tracked-config-stamped` check, and the precondition for "live audit reports exactly
  as before".

Seven bindings **failed as written** against main. Each is rebound to the truth below. The
lead must carry each rebinding into the named leaf's text, because a manifest note alone
does not reach the implementer.

1. **F1, β2 chokepoints: producer-absent.** C2 and §2 fact 9 name "the recovery-dump
   re-apply path of `scripts/cgl_eta_auto_apply_impl.py`". The file actually lives at
   `fused-memory/scripts/cgl_eta_auto_apply_impl.py`, and it only reads and dumps
   `fact_embedding`; nothing re-applies the dump. Its cross-graph writes go through
   `fused-memory/scripts/migrate_cross_graph_leak.py` into
   `maintenance/cross_graph_move.py::_embedding_set_clause`, which β2 already asserts.
   **Rebind:** drop the cgl_eta file from β2's files and text. The vecf32 SET sites are:
   - `GraphitiBackend.update_node_embedding`
   - `::update_edge_embedding` (γ2 removes it later)
   - `cross_graph_move.py::_embedding_set_clause` / `::format_vecf32_literal`
   - `arm_harness/graph_copy.py::_write_vectors`
2. **F2, β2's C0 registry: producer-absent.** The PRD names "the same discovery the
   dashboard uses (canonical, legacy as fallback)". That function,
   `dashboard/src/dashboard/config.py::_discover_escalation_urls`, cannot be imported from
   fused-memory, and it maps a basename to an escalation URL, not to a `project_id`.
   **Rebind:** use `fused_memory/models/scope.py::build_known_projects_map` and
   `::resolve_project_id_for_root`. `server/main.py` already builds that map from
   `PROJECT_ROOT` + `DASHBOARD_KNOWN_PROJECT_ROOTS`. The "does not resolve" case becomes
   `::is_phantom_project_root`.
   **Semantic effect:** only the canonical yaml is read, with a basename fallback. A root
   whose config uses only a legacy spelling therefore resolves by basename; it does not
   become `unclassified`.
3. **F3, live P0 and guard signals (α2, ζ, η): producer-extent-short.** D13's seeds
   (`evalmem_*`, `_probe*`, `_test_*`, "curator pollution") do not cover live
   `mem0migrations`. That collection holds one point: mem0's telemetry placeholder, which
   mem0 writes only while `MEM0_TELEMETRY` is on (it is pinned false).
   "Vector-bearing" is also undefined. If it is read from configuration, these become
   `unclassified` and the first live P0 refuses:
   - the 0-point Qdrant collections `test`, `task_dedup_dark_factory` and
     `task_dedup_reify`;
   - the 0-embedding graphs `probe_e1_*`, `sweep_e4_*` and `knowlive`.
   **Rebind (resolved by the lead in β2's task text):** a graph is vector-bearing iff it
   holds a non-NULL `name_embedding` or `fact_embedding`; in-scope membership is by name;
   any other Qdrant collection with at least one point must be excluded or unrooted. The
   seeds are `evalmem_*`, `_probe*`, `_test_*`, `task_curator_test_*`, `task_dedup_*`,
   `test` and `mem0migrations`. `task_curator_p`, `_proj` and `_project` are D13
   "unrooted": backed up and dropped at P3. γ1 confirms the `mem0migrations` exclusion.
4. **F4, γ2's removal signal: producer-extent-short.** Three gaps:
   - **File list.** It omits `tests/test_graphiti_group_arg_canonicalization.py` (sweep
     rows and `len(_ALL_GROUP_ARG_SWEEP_CASES) == 43`) and
     `tests/test_timeout_marker_inversion_guard.py`. The latter's `_GRANDFATHERED` entry
     names the deleted integration file, so its no-stale-entries test fails.
   - **Orphans.** These die with `drop_vector_indices` and are not on the removal list:
     `GraphitiBackend.drop_vector_index`, `._await_index_catalog_settled` and
     `falkor_indices.py::IndexCatalogUnsettledError`. (`::unsettled_index_statuses` was
     first listed here too; it has live callers and stays.)
   - **Prose.** Docstrings in six or more files mention the removed names, so a literal
     "no mention remains" grep cannot pass.
   **Rebind:** add both files to γ2, and name the orphans as PRD §4's "orphaned
   helpers". The checks assert definitions and call sites only; prose is reworded but not
   gated. `falkor_indices.py::vector_index_properties` and `::list_indices` stay.
5. **F5, α2 (3), the Phase-1 probe re-run: producer-extent-short.** `probe.py` and
   `run_phase.sh` are committed, but their inputs are not: `audio/jfk.flac` and
   `data/mem0_texts.json` are deliberately uncommitted, and no committed script builds
   the latter. `probe.py` also defaults to port 8415.
   **Rebind:** α2's brief carries both recreation steps:
   - fetch openai/whisper `tests/jfk.flac` and verify its sha256;
   - run a read-only Qdrant scroll of 6000 `payload.data` strings from
     `fused_dark_factory`.
   It also passes `--embed-url http://127.0.0.1:8420/v1/embeddings`.
6. **F6, δ's dead-owner deadline: producer-downstream.** Q5 says δ decides it "against ζ's
   measured window", but ζ depends on δ.
   **Rebind:** δ ships the deadline as a profile field, defaulting to the PRD's suggested
   15 min past the P3 start. ζ reports its measured window against that deadline and
   escalates if the margin is short. ζ's predicate (window ≤ 10 min) already implies a
   margin of at least 5 min.
7. **F7, ζ's injected-fault auto-restore: producer-extent-short.** δ's B11 injects its
   fault in a fixture test, and ζ cannot reuse that on a real scratch run.
   **Rebind:** δ's text adds a fault-injection seam that works only under the scratch
   profile and refuses under `live`.

Drift that changes leaf wording but no binding:
- **γ2:** `update_node_embedding`'s live callers are now `_mint_entity_node` and
  `rename_entity_node`.
- **δ:** `_maybe_backfill_corpus` is on `TaskInterceptor`, not `TaskCurator`.
- **α1:** the compose file already has three services (`fused-mcp`), so the embedder is
  the fourth. α1 adds two new `-`-prefixed ExecStartPre lines (the wait uses the
  host-local `/home/leo/bin/wait-for-port.py`) and leaves the existing two unchanged.
- **β2 and ε:** the Mem0 store class is `Qdrant`, not `QdrantDB`. `calibrate_write_triage.py`
  and `audit_duplicate_memories.py` live in `fused-memory/scripts/`.
- **δ and η:** a naive grep finds nine installed `Wants=fused-memory.service` units, not
  eight, because `df-flip-1543-to-pending.service` also carries it.

Two hazards the leaves should carry:
- **ζ's report marker.** ζ's check greps `plans/` for the marker. This manifest therefore
  writes it only with bracket breaks: PRD[-]MARKER:local-embedder-cutover
  rehearsal[-]report. The lead's decompose record in the PRD must do the same.
- **ε's fixture output.** ε's fixture C4b output must not land at
  `fused-memory/calibration/embedder-threshold-map-*.json`, which is ζ's delivered path.

## Per-leaf bindings

### α1 — embedder compose service and hold drop-in

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| service on `127.0.0.1:8420` | producer α1; compose file exists, port unnamed at main, 8420 free | PASS | grep `127\.0\.0\.1:8420` in compose |
| image digest pin | producer α1; local image `v0.26.0 sha256:ffb2d59b…` | PASS | grep the full digest in compose |
| GPU share 0.062 | `scripts/local-model-serving/lms_serve.py::_memory_share_for` = 0.062 (executed) | PASS | grep `0\.062` in compose |
| bounded embedder wait in the template | producer α1 over `scripts/fused-memory.service.template` ExecStartPre | PASS | grep `127\.0\.0\.1:8420` in template |
| hold drop-in `ConditionPathExists=!` | producer α1; no `scripts/` template uses it at main | PASS | grep `ConditionPathExists=!` in `scripts/` |
| `docker compose config` validates | Compose v5.1.2 + NVIDIA runtime present | PASS | manual |

### β1 — overlay loader, native width, startup probe

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `FUSED_MEMORY_CONFIG_OVERLAY` loader | producer β1; loader substrate in `config/schema.py` (`CONFIG_PATH`) | PASS | grep present in `fused-memory/src/` |
| `get_status.config_overlay` | producer β1 on `server/tools.py::get_status` → `MemoryService.get_status` | PASS | grep `config_overlay` present |
| Mem0 stops emitting `embedding_dims` | removal in `backends/mem0_client.py` (present at main) | PASS | grep `embedder_config\[['"]embedding_dims['"]\]` absent |
| the other three clients send no `dimensions` | `build_embedder`, `TaskCurator._get_embedder` and `build_embed_fn` already comply (D6) | PASS | manual (B4) |
| startup probe degrades on mismatch | built by β1 | PASS | manual (B5) |
| overlay refusals | built by β1 | PASS | manual (B19) |

### β2 — C0, C2 guard, chokepoints, loud `search_corpus`

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `EmbedderWidthMismatch` | producer β2; absent at main | PASS | grep `^class EmbedderWidthMismatch[(:]` |
| `embedder_scope` config section | producer β2 | PASS | grep `embedder_scope` in `config/` |
| C0 registry source | F2 rebinding to `models/scope.py` | PASS (rebound) | manual (B8a) |
| vector-bearing definition and seeds | F3; resolved by the lead in β2's task text | PASS | manual |
| Qdrant and Falkor width guard | built by β2; `vec.euclideanDistance` raises on width mismatch | PASS | manual (B6, B7) |
| chokepoint asserts | F1 rebinding; B18 on `cross_graph_move` | PASS (rebound) | manual (B18) |
| chokepoint scope for `graph_copy._write_vectors` and the `reassign_edge` / `redirect_node_edges` copy sites | resolved by the lead: the assert takes the expected width as a parameter, and the copy sites are width-preserving and named exempt | PASS | manual |
| `search_corpus` raises; callers surface it | `TaskCurator.search_corpus` returns `[]` on failure today | PASS | manual |

### α2 — host install (attended)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| 768 vector on loopback 8420 | producer α1 upstream | PASS | manual |
| Phase-1 probe re-run | F5 rebinding (recreate inputs, `--embed-url` :8420) | PASS (rebound) | manual |
| live `get_status` embedder block, guard ok | producers β1 and β2 upstream; extent per F3 | PASS | manual |

### γ1 — derivation, cache, census, prepare

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `python -m fused_memory.maintenance.reembed` | producer γ1; no `reembed` at main | PASS | path `maintenance/reembed/__main__.py` present |
| derivation parity | mem0 `OpenAIEmbedding.embed`, graphiti bulk paths, `task_curator.embedding_text`, `graph_copy.reembed_graph` (D5, D7, D8, D11) | PASS | manual (B1–B3) |
| paged Falkor reads | `graphiti_client.py::_paged_ro_query` and `::apply_incompleteness_policy` | PASS | manual (B15a) |
| curator texts across all tags | `BackfillManager.prune` pattern; `TaskCurator._point_id` | PASS | manual |
| scope from C0 | producer β2 upstream | PASS | manual |

### γ2 — backup, apply, retire `reindex.py`

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| stale/reindex helpers undefined | removal; defined in `graphiti_client.py` at main | PASS | grep `def (…)\(` absent in src+tests |
| …and uncalled | removal; call sites in `reindex.py` and the deleted tests | PASS | grep `\.(…)\(` absent in src+tests |
| orphaned index helpers undefined | F4 rebinding | PASS (rebound) | grep orphan defs absent in src |
| `maintenance/reindex.py` removed | exists at main | PASS | path absent |
| `fused_memory.maintenance.reindex` unimported | imported by two test files at main | PASS | grep absent in src, tests and scripts |
| backup/apply round trip | producer γ2 over γ1 upstream; Qdrant 1.17.1 | PASS | manual (B10a, B15b) |

### ε — thresholds, stamps, mapper

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `CalibrationProvenanceMismatch` | producer ε | PASS | grep `^class CalibrationProvenanceMismatch[(:]` |
| `check_embedder_provenance` | producer ε | PASS | grep `def check_embedder_provenance\(` |
| `embedder_thresholds` group | producer ε (stamped into tracked `config.yaml`, R7) | PASS | grep `^embedder_thresholds:` in `fused-memory/config/config.yaml` |
| tracked `config.yaml` stamped (R7) | no `embedder_model:` key in `config.yaml` at main | PASS | grep `^[[:space:]]+embedder_model:` in `config.yaml` |
| consumers refuse on mismatch | built by ε at the five consumers (`audit_duplicate_memories.py::resolve_ann_threshold` present) | PASS | manual (B16) |
| calibrate writes the overlay | producer β1 upstream; `--write-config` and `--write-project-config` present | PASS | manual (B20) |
| quantile mapper on a fixture | producer ε; name is the leaf's | PASS | manual |

### δ — `reembed cutover`

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| `maintenance/reembed/cutover.py` | producer δ inside γ1's package | PASS | path present |
| `fused-memory-cutover-guard.timer` | producer δ (I3); the unit name is a profile field | PASS | grep in `fused-memory/`, `scripts/` |
| `--if-owner-dead` | producer δ (I3) | PASS | grep `if-owner-dead` |
| real cleanup script (R1) | stub committed at decompose; δ replaces it | PASS | grep `EMBEDDER-CUTOVER-CLEANUP-STUB` absent in the script |
| engine, guard, overlay, C4b format, drop-in | producers γ1, γ2, β2, β1, ε, α1, all upstream | PASS | manual |
| dead-owner deadline | F6 rebinding | PASS (rebound) | manual |
| restore, hold, P5 behaviour | built by δ; quiesce surface verified (D12) | PASS | manual (B8b–B17) |

### ζ — dress rehearsal (integration gate)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| rehearsal report committed | producer ζ; no report at main | PASS | grep `PRD[-]MARKER:local-embedder-cutover rehearsal[-]report` in `plans/` |
| C4b map committed | producer ζ; format from ε upstream | PASS | grep `"census_sha"` in `fused-memory/calibration/embedder-threshold-map-*.json` |
| live census and dry-run P0 | producers γ1 and δ upstream; extent per F3 | PASS | manual |
| injected-fault auto-restore on a real run | F7 rebinding | PASS (rebound) | manual |
| real-systemd B13, B10b on real-shaped data | producers δ and α1 upstream; precedent task 3722 | PASS | manual |

### η — live cutover (attended)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| cutover and live status | producers δ, β1, β2, γ1, ζ upstream | PASS | manual |
| recalibration into the overlay | producer ε upstream; calibrate flags present | PASS | manual |
| write-triage owner notified | producer κ1 (R3 edge) | PASS | manual |

### λ — post-soak fold

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| tracked `config.yaml` carries granite | producer λ; `config.yaml` names text-embedding-3-small at main | PASS | grep anchored to the embedder block's `model:` line in `config.yaml` |
| effective values unchanged | producer η upstream | PASS | manual |

### θ — overlay and backup cleanup (deterministic)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| cleanup action | producers δ (real script, gated by δ's check) and λ, both upstream | PASS | manual (before_done exit code) |

### κ1 — cross-PRD corrections

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| write-triage PRD amended | producer κ1; no mention at main | PASS | grep `local-embedder-cutover-prd` in that file |
| eval PRD pointer | producer κ1; no mention at main | PASS | grep `local-embedder-cutover-prd` in that file |

### κ2 — docs and secondary config

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| docs, `config-docker.yaml`, skills' mapped 0.6, `api_url` precedence note (R5) | producers ζ (C4b) and α1 (service name) upstream via η; `config/env_precedence.py::warn_if_ambient_base_url_is_overridden` present | PASS | manual (no dependents) |

## Resolutions after the critic pass (2026-10-11)

An adversarial critic reviewed the drafted leaves and this manifest. Its six blocking
findings and the should-level ones are resolved in the filed task texts; the ones that
change a binding are recorded here.

- **Leo's ruling, 2026-10-11 (extends D14).** The `FUSED_MEMORY_CONFIG_OVERLAY` drop-in
  goes on `fused-memory.service` and on both timer services
  (`fused-memory-flag-marker-sweep.service`, `memory-metadata-coverage-census.service`).
  Their jobs build `MemoryService` / `Mem0Backend` straight from tracked `config.yaml`,
  which keeps the 1536 identity for the 14-day soak. δ's `live` profile lists all three
  units; P5 writes them, rollback and auto-restore remove them, and θ's cleanup removes
  them. Chosen over "guard only" and "pause the timers".
- **F3 and the chokepoint scope are no longer OPEN.** β2's text carries the
  vector-bearing definition, the exclusion seeds and the copy-site treatment (both
  verdicts are now PASS in the sidecar).
- **The guard binds at backend initialisation, in any process (β2).** It is the backstop
  for every script that builds a backend outside the service during the soak.
- **P5's map input (δ, ζ, η, ε).** `cutover --threshold-map PATH`, validated at P0. ζ runs
  the mapper between `prepare` and the scratch cutover. ε's mapper takes `--out` and
  commits nothing under `fused-memory/calibration/embedder-threshold-map-*`.
- **Overlay merge granularity (β1, ε, λ, θ).** A stamp and the values it vouches for are
  one unit: `embedder.*` merges per key, and each calibration unit in the overlay
  replaces the base unit wholesale. A per-key merge would leave a base 1536-space
  per-category cutoff under a granite stamp. λ's fold and θ's check compare units in both
  directions.
- **Absent-stamp rule for schema defaults (ε, κ2).** The `embedder_thresholds` schema
  default carries its values with a default stamp equal to the schema-default identity.
  A group set in YAML or an overlay without a stamp refuses. κ2 gives
  `config-docker.yaml` the mapped group with a granite stamp.
- **γ2's orphans.** `falkor_indices.py::unsettled_index_statuses` has live callers and
  stays; the removal check names only `drop_vector_index`,
  `_await_index_catalog_settled` and `IndexCatalogUnsettledError`.
- **Profile owns every unit name and path (δ).** A scratch run cannot stop a live timer
  or install the live guard timer; B9 asserts it.
- **Store writers need a profile (γ2).** `backup` and `apply` refuse without an explicit
  `--profile`, so the registry never comes silently from an operator shell's environment.
- **Check anchors.** ε's group check reads `^embedder_thresholds:` in tracked
  `config.yaml`, and λ's check is anchored to the embedder block's `model:` line, so
  neither goes green before its producer lands.
- **Also cancelled with 3867 and 4877:** task 4294, the other half coalesced into 4877.

## Resolutions after the final critic pass (2026-10-11)

A second critic reviewed the integrated package (the patched leaves and this manifest).
It confirmed 18 of the 20 first-round findings resolved, found two of the lead's fixes
wrong, and added new findings. All are text changes; none changed the DAG or needed a
ruling. The ones that change a binding:

- **β2's live proof needs the service's registry.** An agent's environment carries no
  `DASHBOARD_KNOWN_PROJECT_ROOTS`, so the pre-merge read-only C0 run takes the registry
  from `systemctl --user show fused-memory.service -p Environment`.
- **The guard probe has three outcomes (β2).** `ok`, `mismatch` and `unknown`. A timeout
  or connection error is `unknown`: it degrades status, retries with bounded backoff, and
  refuses only vector writes for that scope. Outside the server the guard is per-scope and
  lazy, so tests on excluded scopes are never refused during the soak.
- **The engine never constructs a guarded backend (γ1, γ2, δ).** It uses raw store
  clients, because P3–P4 and every restore run while the stores and the base identity
  disagree.
- **The profile has one home, created in γ1** (`maintenance/reembed/profiles.py` and
  `fused-memory/config/reembed-profiles/live.yaml`); δ extends both.
- **Timers come back (δ, closes a gap in PRD C1).** P1 stops each timer and waits for its
  service; P6 and every restore path start the timers again.
- **Restore branches on a journalled 'service started' record (δ)**, and the boot-time
  units resolve `uv` through `scripts/lib/resolve_uv.sh`.
- **P0 probes the source spec too (δ)**, and the cutover unit gets the repo `.env`, so a
  rollback can embed with OpenAI.
- **C4b entries (ε, δ).** The skills' 0.6 entry has `config_key: null`; P0 refuses a
  duplicate or missing key; P5 writes only non-null entries.
- **θ's failure path.** The runner never re-runs a deploy action, so the cleanup is run by
  hand and confirmed before the escalation is resumed.
- **λ's early warning (ζ, η).** ζ finds identity-coupled tests by editing the worktree's
  own `config.yaml` (an autouse fixture re-pins `CONFIG_PATH`), and η adds those tests to
  λ's declared files.
- **Soak-period verify (ζ, η).** ζ runs the default test selection against the 768
  scratch stores under the 1536 identity; η runs it once after P6 before resuming the
  queues. A red run would mean red merge verifies for 14 days.
- **The api_url guard test (β1).** β1 relaxes the embedder half of
  `test_config_schema.py::test_shipped_config_yaml_shares_one_openai_api_url_default`,
  so neither κ2 nor λ edits that file.
- **Where the resolutions live.** Each leaf's lead resolutions are in its `details`,
  under a heading that says they are binding.
