# Capability manifest — local-memory-models-eval PRD

Decompose-time G3+G6 mechanization for `plans/local-memory-models-eval-prd.md`
(committed `0cf6de56ac`). One block per task; every binding re-verified against
the working tree / installed wheel on 2026-08-05 at decompose. Machine-readable
twin: `local-memory-models-eval-prd.capability-manifest.yaml` (task_id stamped
by `commit_planning`).

Delivered-check marker convention: committed artifacts of this PRD carry a
literal marker line `PRD-MARKER:local-memory-models-eval <artifact-slug>`
(spelled here and in the sidecar only in self-match-proof bracketed regex form,
`PRD[-]MARKER:…`, so the manifest pair never satisfies its own checks).

## Decompose-time deviations from the PRD's plan (recorded, not silent)

1. **μ added (operator gate split out of λ).** The PRD's λ row says "raise the
   operator gate for Leo's ruling"; its Goal says verdicts are "ruled on by Leo
   through the standard operator gate". The standard operator gate in this
   project is a `task_kind='deterministic'` pure gate (`always_escalates=true`,
   no `before_done`) — the landed 3634/2864 shape. Folding it into a normal λ
   would either trip the submit-time routing-intent lint or leave a merged
   normal task stranded blocked-on-human (INV-6). λ stays normal (synthesis +
   committed decision records); μ is the gate whose resolution is Leo's ruling.
2. **`structured_output_mode` naming drift.** graphiti_core 0.28.2's
   `OpenAIGenericClient` has no `structured_output_mode` knob: mode selection is
   response_model-driven (`json_schema` when a response_model is passed,
   `json_object` otherwise — `openai_generic_client.py:95-128`). The capability
   the design needs (both modes reachable, plus *forcing* `json_object` for the
   MoE arm) is delivered by β (thin wrapper/config flag), upstream of every arm.
3. **α files without `execution_class='operational'`.** The PRD's Kind column
   says "operational", but `operational_routing_guard` coerces any
   `execution_class='operational'` submission into a deterministic pure gate on
   every submit path (planning_mode included) — which would strip α of its
   agent/worktree and its committed scripts. α's deliverables are committed ops
   scripts + host verification: an honest `task_kind='normal'` agent task.

## α — Serving substrate (systemd --user units + health check)

| Capability | Evidence | Verdict |
|---|---|---|
| vLLM ≥0.26 OpenAI-compatible structured outputs (`json_schema`, xgrammar) | research-verified 2026-08-05 against current vLLM docs (PRD appendix); measured conformance 81–100% with guided decoding (SqueezeBits benchmark) | PASS |
| llama.cpp MoE arm constraint honesty | ggml-org/llama.cpp#21228: silent fallback to unconstrained output on `$ref`/`$defs` schemas → arm pinned to `json_object` + client-side validator (validator = ε; forced mode = β) | PASS |
| GPU headroom: ~16.4 GiB free VRAM available to an arm (measured) | 2026-08-05 decompose-time derivation: 24GB − ~4GB whisper-writer resident (Leo, PRD D10) — SUPERSEDED 2026-08-06 by direct measurement: `nvidia-smi` free ≈ 16.4 GiB (`--query-compute-apps` omits the KDE/X11 desktop's graphics contexts, ~3.2–3.3 GB, so the subtraction overstated headroom) | PASS (corrected 2026-08-06 — see PRD D10) |
| Candidate weights exist (incl. **no Qwen3-Embedding-2B** — family is 0.6B/4B/8B) | HF cards / QwenLM GitHub verified 2026-08-05 (PRD appendix) | PASS |

Delivered-check: grep `PRD[-]MARKER:local-memory-models-eval serving` in
`scripts/` (α commits units + health script under `scripts/local-model-serving/`
carrying the resolved marker literal).

## β — Config + client plumbing

| Capability | Evidence | Verdict |
|---|---|---|
| `OpenAIGenericClient` in installed wheel | `graphiti_core 0.28.2` (dist-info METADATA), `llm_client/openai_generic_client.py:37`; chat.completions + `json_schema`/`json_object` response_format `:111-128` | PASS |
| LLM `base_url` gap is real (β delivers the plumb) | `fused-memory/src/fused_memory/backends/graphiti_client.py` LLM construction (~:497-510) builds `GraphitiLLMConfig` without `base_url`; embedder construction *does* pass it (~:533-539) — the asymmetry confirms the seam | PASS |
| Mem0 config-dict seam | `backends/mem0_client.py:138-170` — llm/embedder config dicts exist, no `openai_base_url`/`embedding_model_dims` today | PASS |
| reindex embedder seam | `maintenance/reindex.py:155-166` — `OpenAIEmbedderConfig` built without `base_url`; the config class accepts one (graphiti_client.py embedder path passes it today) | PASS |
| Forced `json_object` mode for MoE arm | **not** a stock 0.28.2 knob (deviation 2) — producer: **β itself**, upstream of ε/η/θ | PASS |
| Restart-tier only (no reload wiring needed) | `llm.*`/`embedder.*` absent from `RELOADABLE_FIELDS` (`config/reload.py`) | PASS |
| Anti-silent-fallback (graphiti #912 class) | β's integration test observes the mock server *receive* the traffic — a plumbed-but-ignored base_url fails the test, never silently hits api.openai.com | PASS (rejection bound by β's test) |

Delivered-check: grep `client_class` in `fused-memory/src/fused_memory/`
(zero hits on main today — the knob's introduction is unambiguous).

## γ — Durable write telemetry

| Capability | Evidence | Verdict |
|---|---|---|
| Journal choke point exists | `_journaled_backend_call`, `services/memory_service.py:1302` (call sites :2206, :2257, :2316, :2688, :2992, :4068) | PASS |
| Token usage reachable | wheel `llm_client/token_tracker.py:55` `TokenUsageTracker`; every `LLMClient` carries `self.token_tracker` (`client.py:84`); `graphiti.py:272` exposes a getter | PASS |
| Field-population (result-field twin) | γ is the producer writing non-sentinel `duration_ms` + token counts on the production write path; signal observed on a **live** write via the documented sqlite query | PASS |

Delivered-check: grep `duration_ms` in `fused-memory/src/fused_memory/services/`
(clean on main today; collisions exist only in `middleware/`/`reconciliation/`).

## δ — Corpus builder

| Capability | Evidence | Verdict |
|---|---|---|
| Episode store readable | ~2,635 dark_factory episodes, measured read-only 2026-08-05 (PRD §Background) | PASS |
| No-outcome-filter constraint | process constraint, hazard-binding on the task; manifest + stratification report must state it | PASS (manual) |

Delivered-check: grep `PRD[-]MARKER:local-memory-models-eval corpus[-]manifest`
in `fused-memory/scripts/` (builder + committed manifest carry the resolved
marker).

## ε — Arm-runner harness (integration gate)

| Capability | Evidence | Verdict |
|---|---|---|
| Metrics schema home | `shared/src/shared/memory_eval_metrics.py` exists (imported, not restated — INV-5) | PASS |
| E1 probe math | `fused-memory/scripts/memory_eval_retrieval_probe.py` exists; **import, don't fork** — a probe-file split belongs to the eval-program lane (G4 seam) | PASS |
| Transcript-query corpus tooling | `fused-memory/scripts/memory_eval_transcript_corpus.py` exists | PASS |
| Real construction path reusable | `GraphitiBackend` construction path in `backends/graphiti_client.py` (INV-5: reused, not re-implemented) | PASS |
| Scratch-guard rejection | **built + bound by ε**: `^evalmem_[a-z0-9_]+$` constructor guard; boundary test observes the typed rejection fire on `group_id="dark_factory"` (G6 branch 4) | PASS |
| Upstream deps deliver inputs | β (clients), γ (telemetry), δ (corpus) all upstream in the DAG | PASS |

Delivered-check: grep `evalmem_` in `fused-memory/` (zero hits in that tree
today; only the PRD in `plans/` mentions it).

## ζ — Controls + pre-registration

| Capability | Evidence | Verdict |
|---|---|---|
| Margins finite by construction | `margin_m = max(2·σ_control(m), floor_m)` over two completed control replays — σ finite when runs complete; INV-4 abort rule prevents silent partial σ | PASS |
| Latency envelope anchor | `config.yaml` queue block: `write_timeout_seconds`/`backend_write_timeout_seconds` = 120 (env-defaulted) — p95 < 120s with stated headroom | PASS |
| Incumbent path live | 10,153 completed writes, 0 dead letters (PRD §Background) — control replays run on the working incumbent | PASS |
| Ordering enforceable | ζ upstream of η/θ/ι + ε's schema rejects candidate-arm artifacts lacking `preregistration_sha` | PASS |

Delivered-check: grep `PRD[-]MARKER:local-memory-models-eval preregistration`
in `plans/` (the committed prereg doc carries the resolved marker; the PRD
does not).

## η — LLM screening

| Capability | Evidence | Verdict |
|---|---|---|
| All four arms servable | α upstream (endpoints); MoE via llama.cpp `json_object` + ε's validator | PASS |
| Phi-4 context-fit check is measurable | screening measures graphiti's longest real prompt vs 16K (Open Q2) — drop-with-margin rule pre-registered in ζ | PASS |
| Survivor rule exists before selection | ζ upstream commits the funnel's survivor rule | PASS |

Delivered-check: grep `PRD[-]MARKER:local-memory-models-eval screening[-]report`
in `plans/`.

## θ — LLM full arm runs + comparison report

| Capability | Evidence | Verdict |
|---|---|---|
| Realistic concurrency bound | graphiti env-derived `SEMAPHORE_LIMIT` default 20; **collision confirmed**: `config.yaml` queue block reads `${SEMAPHORE_LIMIT:3}` — harness must set graphiti concurrency without poisoning the product knob | PASS |
| Client-class parity | every arm incl. incumbent via `OpenAIGenericClient` (β); OpenAIClient↔GenericClient delta measured by ζ's control pair | PASS |
| One-code-SHA validity | ArmSpec carries `code_sha`; instrument check rejects cross-SHA comparisons (referent-fidelity PRD 3666-3676 lands mid-flight) | PASS |
| Artifact schema enforcement | ε's MetricsRecord schema rejects candidate artifacts without matching `preregistration_sha` | PASS |

Delivered-check: grep `PRD[-]MARKER:local-memory-models-eval llm[-]report` in
`plans/`.

## ι — Embedding arms

| Capability | Evidence | Verdict |
|---|---|---|
| Re-embed machinery | `maintenance/reindex.py` exists (β adds its `base_url`) | PASS |
| Frozen reference graph | produced by ζ (one incumbent control-replay graph frozen), upstream | PASS |
| Index-build recipe on scratch graphs | upstream `build_indices_and_constraints()` exists in the wheel (`driver/falkordb_driver.py`); **evalmem_\* only** per hazard — the with-indices config is proven different-in-fact by the boundary sketch's fulltext-leg row | PASS |
| Probes | E1 probe math + transcript corpus (ε bindings) + replay-derived known-item queries from the frozen graph | PASS |
| Native-dims comparison legal | dims are a free variable (PRD D7); mixed-dims cosine hazard is the *cutover* PRD's problem — this eval only measures re-embed throughput | PASS |
| Mem0 replica collection | Qdrant live (get_status: connected; 21,243 dark_factory memories); replica named + torn down via ε's prefix-revalidating helper | PASS |

Delivered-check: grep `PRD[-]MARKER:local-memory-models-eval embedding[-]report`
in `plans/`.

## λ — Synthesis + decision records

| Capability | Evidence | Verdict |
|---|---|---|
| Inputs exist when λ runs | θ + ι reports upstream; ζ's committed decision rules upstream | PASS |
| Decision-record surface | the project's standing decision-record pattern in `plans/` + `add_memory` `decisions_and_rationale` (consumer: μ + operators reading `plans/`) | PASS |
| Follow-up naming only (no filing) | cutover + backfill PRDs are deliberately unfiled until Leo's ruling (PRD packaging decision) — λ names, never files | PASS |

Delivered-checks: grep `PRD[-]MARKER:local-memory-models-eval decision[-]llm`
and `PRD[-]MARKER:local-memory-models-eval decision[-]embedding` in `plans/`
(one per axis record).

## μ — Operator gate (deterministic pure gate)

| Capability | Evidence | Verdict |
|---|---|---|
| Gate mechanism | `task_kind='deterministic'` + `always_escalates=true`, no `before_done` — the landed 3634/2864 shape; born-at-L2 on the age-surfaced queue (INV-7: owner Leo, bound = L2 age surfacing) | PASS |
| Resolution references records | escalation presents λ's two committed decision records; resolution text records the per-axis ruling verbatim | PASS (manual — resolution text is not greppable) |
