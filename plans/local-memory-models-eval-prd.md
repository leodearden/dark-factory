# Local memory-models eval — local LLM and local embedder for fused-memory's memory operations

**Project:** dark-factory (fused-memory Graphiti/Mem0 backends). **Status:** active, authored
2026-08-05. **Approach:** B+H-lite (contract + boundary-test sketch on the eval-harness seam).
**Packaging (Leo, 2026-08-05):** this PRD is *eval-only* — it ends in two committed decision
records. Production cutover (LLM) and the embedding backfill/migration are **follow-up PRDs
authored only after — and gated on — the verdicts.** They are deliberately unfiled today and are
not counted as G1 consumers of anything here.

> **Amendment 2026-08-06 (task 3748)** — the operating VRAM budget is the MEASURED ~16.4 GiB, not
> the nominal 19–20GB this PRD was authored against. See D10.
>
> Applied corrections (2026-08-06, task 3748):
> - Task 3720 (LME-η, pending) — **correction**: VRAM-budget bullet updated to the measured figure;
>   MoE-sizing sentence updated twice — first to record the sizing question the correction opened,
>   then (once α step 22 ruled) to the pinned Gemma QAT arm at 13.27 GiB. See Open Q3.
> - Task 3721 (LME-θ, pending) — **insertion**: no VRAM figure existed; one bullet added.
> - Task 3713 (LME-α; **in-progress** when this amendment was written, **`blocked`** — steward
>   re-escalated to a human, no active claimant — as of this re-check) — its task **record** was
>   deliberately not edited by this task; its **code** no longer needs editing. The code claim below
>   is pinned to `task/3713` @ `b3745f5a5c`
>   (2026-08-06T06:35:56+01:00; locator: `scripts/local-model-serving/lms_vram.py`, function
>   `evaluate_budget` — the README's own section on this was still titled `OPEN: the budget verdict's
>   subject is miscalibrated` at that exact SHA, renamed to `RESOLVED: the budget verdict's subject`
>   only by the branch's later tip, so the function is the stable locator here, not the heading),
>   which is **not yet on `main`** — re-read the branch tip before relying on them.
>   - *Code — resolved.* α step 23 landed the verdict-**subject** correction (esc-3713-6).
>     `lms_vram.evaluate_budget(used_mib, total_mib, *, baseline_mib, baseline_free_mib)` now judges
>     the **arm's own footprint** (`used − baseline`) against the free VRAM measured immediately
>     before that arm started. No nominal-ceiling parameter remains on that path, and
>     `lms_healthcheck.py` passes a live per-arm baseline rather than relying on a default. This
>     matches `arm_fits`, which already gated each arm against measured free VRAM. An earlier
>     revision of this amendment asserted the opposite (that a `NOMINAL_CEILING_GIB = 19.5` default
>     was still enforced against **total** card usage) and instructed α reviewers to disbelieve a
>     healthcheck PASS on that basis. That instruction was wrong and is **withdrawn**: judge α's
>     healthcheck output on its merits.
>   - *Task record — now corrected too, independently of this task.* A fresh re-check (`get_task`,
>     this amendment pass) shows 3713's `description` and `metadata.user_observable_signal` **no
>     longer** read "19–20GB" — both now state the measured ~16.4 GiB figure, and 3713's own metadata
>     cites this task as the source (`vram_budget_correction_source_task: 3748`,
>     `vram_budget_corrected_at: 2026-08-06T06:04:00Z`). This task did not make that edit — it
>     deliberately left 3713's record alone while 3713 was in-progress and live-claimed (see design
>     decisions) — so someone or something else brought it in line first. No follow-up task is filed
>     for this: the fix a follow-up would have requested is already done.

> **Amendment 2026-08-10 (task 3804)** — **the LLM candidate slate is THREE arms, not four.**
> Leo dropped **Mistral-Small-3.2-24B** on 2026-08-06 (esc-3713-10, follow-up from task 3713/LME-α)
> after live measurement proved it unservable on this host. `scripts/local-model-serving/arms.yaml`
> is the **authoritative slate**; this PRD is the narrative record of it. Remaining arms: Qwen3.5-9B,
> Phi-4 14B, and the MoE stretch arm (Gemma-4-26B-A4B-it QAT).
>
> **Consequence, stated at the top so it cannot be missed:** η's screening funnel narrows
> **3 → at most 3**, so the **≤3 cap's selectivity is exactly nil** — three candidates against a cap
> of three eliminate nobody by ranking. The funnel's **absolute gates remain fully live** and can
> still drop arms (conformance smoke, VRAM fit under the measured ~16.4 GiB, Phi-4's 16K context
> fit, ζ's throughput floor), so η may still return fewer than three survivors — it just can no
> longer return fewer *because of the cap*. **Whether that warrants re-opening the slate is Leo's
> call: surfaced here, deliberately not decided.** See the consequence note at the candidate slate.
>
> **Single-sourcing, so a future slate change is a two-site edit and not a six-site one:** the
> canonical statements of the three-arm slate and its funnel consequence are exactly **two** — this
> banner and the consequence note at the candidate slate. Every other site in this document (the
> approach sketch, D6, the Contract survivor-rule constraint, η's decomposition row) now
> **cross-references** them rather than restating them, and carries only what is locally its own (a
> count, or a MUST addressed to ζ). If Leo re-opens the slate, edit those two places. Tasks 3719 and
> 3720 hold their own copies out of necessity — a task record cannot cross-reference a file section —
> and are listed under the dispositions below, which is where a future editor should look for them.
>
> Applied corrections (2026-08-10, task 3804) — all in `plans/`, documentation/record only:
> - **This PRD**: the candidate-slate table (Mistral row struck through and annotated, **kept** as
>   evidence of what was commissioned, plus a footnote closing the already-fixed quant defect and
>   stating the re-admission condition); the funnel prose in §Sketch of approach; D6; D10's
>   Consequence bullet (arm count only); the Contract §Pre-registration clause constraining ζ's
>   survivor rule; α's and η's rows in the decomposition plan; the research appendix.
> - **`plans/local-memory-models-eval-prd.capability-manifest.md`** — η's `All four arms servable …
>   PASS` row: that verdict was **falsified** by α's own gate (one arm proved unservable), and the
>   row now says so rather than being quietly renumbered.
> - **`plans/local-memory-models-eval-prd.capability-manifest.yaml`** — η's `title:` realigned to
>   task 3720's live title; `task_id`, `capabilities`, `binding`, `verdict` and every
>   `delivered_check` left byte-identical (machine-consumed contracts).
>
> Cross-check dispositions (whole 3713–3725 batch re-read at correction time; each states what was
> actually checked):
> - Task 3719 (LME-ζ, pending) — **corrected**: the two four-arm sentences in its reasoning-mode
>   block fixed, and the three-arm constraint on the survivor rule appended to its item 4. Its
>   "gemma-4 extracts all four **probe entities**" / "finds all four **entities**" sentences count
>   probe entities, not arms, and were deliberately left byte-identical.
> - Task 3720 (LME-η, pending) — **already correct**, no edit; its description already carries the
>   drop and the selectivity flag.
> - Task 3721 (LME-θ, pending) — **checked, clean**: zero four-arm or Mistral assertions.
> - Task 3722 (LME-ι, pending) — its "all four" counts the four **embedding** arms, which this
>   LLM-axis ruling does not touch; that sentence was **correctly left alone**. A *different* defect
>   in the same record **was** fixed: its pointer `(PRD line 136)` for the qwen3-embedding-4b
>   batch-job constraint. That pointer was **correct when authored** — at `0cf6de56ac` (2026-08-05)
>   line 136 was exactly the Qwen3-Embedding-4B row — and was already broken by task 3748's banner on
>   2026-08-06 (it landed on a table separator, then on prose); this amendment's banner pushed it
>   further, onto an unrelated `SEMAPHORE_LIMIT` paragraph. Replaced with the stable locator
>   "PRD §Candidate slate, the Qwen3-Embedding-4B row"; the rest of that description was verified
>   **byte-identical** afterwards, including its `PRD-MARKER:` delivered-check string. This is the
>   same defect class as task 3973's item 2 (arms.yaml's "line 127" pointer) and the same lesson task
>   3748 learned when it rejected a README heading as a pin — fixed at source here rather than
>   deferred, since it is a one-field edit.
> - Tasks 3715 (β), 3716 (γ), 3717 (δ), 3718 (ε), 3723 (λ), 3725 (μ) — **checked, clean**.
> - Task 3713 (LME-α, `done`) — **deliberately NOT edited.** Its enumeration of the original four
>   dense arms is the historical record of what α was commissioned to serve — the same evidence
>   argument that keeps the struck-through table row, and the same disposition task 3748 took.
>
> Known-remaining drift, named so it is not mistaken for an oversight: the **"Est. VRAM" column is
> stale** against arms.yaml's measured figures (Qwen3.5-9B listed ~6GB, **measured 11.21 GiB**) —
> marked inline at both points of use (the arm table and D10's Consequence bullet) so the deferral is
> visible where the number is read, not only here. That is the 3748 drift class, not slate
> composition; out of scope here and filed as **task 3973** (from ticket
> `tkt_0RS9PDHN212E4G1ZW924MMCV75`). Cite the **task** id: it resolves via `get_task` like every other
> task reference in this document, whereas a `tkt_` id is a `submit_task` receipt that may resolve to
> `combined` under a different id and is not addressable. Task 3973 is **wired to depend on this
> task** (`dependencies: [3804]`, added 2026-08-10), so the sequencing below is enforced by the
> scheduler rather than only asserted in prose.

## Goal

Two evidence-backed, pre-registered verdicts, each observable as a committed decision record plus
a `decisions_and_rationale` memory write, ruled on by Leo through the standard operator gate:

1. **LLM axis** — should fused-memory's Graphiti write-path LLM calls (entity extraction, node
   dedupe, edge extraction, per-edge dedupe/invalidation, summaries — today `gpt-4o-mini`) move to
   a locally-served model on the RTX 3090?
2. **Embedding axis** — should fused-memory's embedder (today `text-embedding-3-small` @ 1536d,
   serving both Graphiti node/edge embeddings and Mem0/Qdrant vectors, write *and* interactive
   query paths) move to a local embedding model?

An operator reading `plans/` after this PRD completes sees, per axis: the pre-registered decision
rule, the measured comparison against it, and the ruling.

## Background — verified substrate (all file:line checked 2026-08-05)

**Current configuration.** `fused-memory/config/config.yaml:12-15` — LLM `gpt-4o-mini`,
`max_tokens: 4096`, `small_model` pinned to the same model (`graphiti_client.py:505`), temperature
defaults to 0.0 (`graphiti_client.py:506`). `config.yaml:26-29` — embedder
`text-embedding-3-small`, `dimensions: 1536`. Both provider blocks carry an `api_url` leaf but
default to the **same** `${OPENAI_API_URL}` env var (`config.yaml:20,34`) — independent LLM/embedder
endpoints require per-block YAML values, not the env var.

**The LLM swap is a client-class change, not only a base_url change.** fused-memory instantiates
`graphiti_core.llm_client.OpenAIClient` (`graphiti_client.py:28,509`), whose structured calls all go
through the OpenAI **Responses API** (`responses.parse`, `openai_client.py:65-97` in the installed
0.28.2 wheel); startup hard-fails without SDK Responses support (`graphiti_client.py:67-123`). Local
OpenAI-compatible servers implement `/v1/chat/completions`, generally not `/v1/responses`. The swap
path is graphiti's `OpenAIGenericClient` (chat.completions + `json_schema` response_format, with a
`json_object` prompt-injection fallback mode) — upstream's own escape hatch for local servers.
Additionally, `base_url` is **not plumbed** today for: (a) the graphiti LLM client
(`graphiti_client.py:502-509` omits it), (c) Mem0's LLM (`mem0_client.py:138-147`), (d) Mem0's
embedder (`mem0_client.py:160-167`), and the standalone reindex tool (`maintenance/reindex.py:160-165`).
It IS plumbed for (b) the graphiti embedder (`graphiti_client.py:533-539`). All `llm.*`/`embedder.*`
config is restart-tier (absent from `RELOADABLE_FIELDS`, `config/reload.py:29-99`).

**Call profile and concurrency.** With fused-memory's real config (no `entity_types`/`edge_types`
forwarded, so attribute-extraction call sites are dead), an episode costs ~3 fixed LLM calls + 1 per
surviving extracted edge + 0–few batched summary calls ≈ **6–10 calls/episode**. Per-episode
fan-outs inside `node_operations.py`/`edge_operations.py` are bounded by graphiti's env-derived
`SEMAPHORE_LIMIT` (default **20**) — `max_coroutines=5` covers only a few `graphiti.py` call sites.
The `SEMAPHORE_LIMIT` env var name **collides** with fused-memory's queue-worker knob
(`config.yaml:53`); setting one silently sets both. Burst concurrency to a local server: up to ~20.

**Volume and cost (measured, read-only, from the live runtime DBs under
`/home/leo/src/dark-factory/data/`).** `add_memory_graphiti` (the operation that triggers the full
extraction pipeline) runs **~26–96/day** (31-day window). ≈300–900 gpt-4o-mini calls/day ⇒ order
**$15–25/month**. Query-time embedding traffic dwarfs writes: ~414k lifetime `search` ops vs ~63k
memory writes — both stores embed the query inline on the interactive path
(`search.py:105-107` in graphiti_core; `mem0/memory/main.py:2206-2210`). Embedding spend at
$0.02/1M tokens is negligible. **No durable token or latency telemetry exists** — graphiti's
in-process `TokenUsageTracker` is discarded, `backend_ops` has no duration column, and graphiti's
OTel spans are no-ops here (no tracer configured).

**Honest motivation.** The write queue shows 10,153 completed writes, **0 dead letters** — the
OpenAI path is not currently failing. The documented memory-path freezes (account caps, 529 storms)
were on the **Anthropic-side** `claude_cli` recon/consolidator stages, which this PRD does not
touch. The case for local is therefore: **(1) availability/independence** — removing a cloud
dependency, its rate limits, and its outage modes from the memory write+query path entirely
(prophylactic, not remedial); **(2) privacy** (secondary); **(3) cost** (minor, ~$15–25/mo,
quantified precisely by this eval). Cost alone does not justify the work.

**Mem0 LLM scope.** Mem0 write paths pin `infer=False` (`mem0_client.py:207,248`) — zero LLM calls.
The LLM axis is Graphiti-only. The embedding axis covers **both** stores.

**Prior art.** April 2026: first successful vLLM local-model eval in this project (coding-agent
subject, rented 96GB GPU) — vLLM ops experience exists. The memory-eval program
(`docs/prds/memory-eval-program.md`) owns a metrics-artifact schema (`shared/memory_eval_metrics`),
the E1 retrieval probe (`fused-memory/scripts/memory_eval_retrieval_probe.py`), and a mined corpus
of real search queries (`memory_eval_transcript_corpus.py`) — reused here, not reinvented.

## Sketch of approach

Two eval axes, strictly isolated, all replay on **throwaway scratch graphs** (never live graphs),
funnel-shaped to keep Leo's full candidate slate affordable:

**LLM axis** (embedder held fixed at the incumbent): replay a committed episode corpus per arm
through the real `GraphitiBackend` construction path onto scratch graphs. Funnel: all three
remaining candidates pass a cheap **screening** stage (schema-conformance smoke, VRAM fit beside
whisper-writer, prompt-length fit, throughput floor); at most three advance to **full corpus
replay** under realistic concurrency on the busy box (was four candidates — Mistral-Small-3.2-24B
was dropped 2026-08-06; what the narrowing does to the cap is stated once, at the consequence note
at the candidate slate). Client-class parity: **every** arm — incumbent
included — runs via `OpenAIGenericClient`, and a one-off control quantifies the
`OpenAIClient → OpenAIGenericClient` delta on the incumbent itself, so the client-class change is
measured, not confounded.

**Embedding axis** (no LLM in the loop): designate one incumbent control-replay graph as the
**frozen reference graph**; per arm, re-embed its node/edge texts (plus a replica of one Mem0
collection) with the candidate — identical topology, only vectors/dims differ. Retrieval quality is
probed in **two configurations**: *with indices built on the scratch graph* (primary — future
production, per `docs/prds/falkordb-index-provisioning.md`) and *embedding-only* (secondary —
today's production, where the BM25 leg silently returns nothing). Probes use replay-derived
known-item queries plus real queries from the transcript corpus. All embedding arms run in full —
re-embedding ~28k short texts is minutes, not hours.

**Pre-registration before candidate arms.** Incumbent-vs-incumbent control runs measure run-to-run
variance; the non-inferiority margins are **derived from that measured variance by a committed
formula**, never hand-picked. The decision rule (Leo, 2026-08-05: non-inferiority — quality within
margin AND latency inside the timeout envelope ⇒ availability decides) is committed *before* any
candidate arm runs, and every candidate-arm artifact embeds the pre-registration doc's git SHA.

**Instrument validation** (the fable-trial lessons, encoded): control-population run before any arm
comparison; explicit checks for arm-config symmetry, non-empty reference artifacts, and non-zero
token/cost accounting; corpus **never filtered on incumbent success**; all arms of a comparison
pinned to one code SHA (the referent-fidelity PRD, tasks 3666-3676, will change the dedupe call
profile mid-flight — a SHA mismatch between arms invalidates the comparison).

### Candidate slate (researched 2026-08-05; Leo selected all)

LLM arms (serving: vLLM ≥0.26 structured outputs / xgrammar, OpenAI-compatible; MoE arm via
llama.cpp only, see hazard):

| Arm | Size / quant | Est. VRAM | Basis (cited in research appendix) |
|---|---|---|---|
| Qwen3.5-9B | dense, Q4/AWQ | ~6GB | IFEval 91.5, BFCL-V4 66.1 (official card) — best published conformance-adjacent scores; huge KV headroom |
| ~~Mistral-Small-3.2-24B~~ **DROPPED 2026-08-06** | ~~dense, AWQ~~ | ~~`~14GB`~~ | **Dropped by α after live measurement (Leo's ruling, esc-3713-10): a vision-language model whose quantized repo's tokenizer encodes vLLM's startup `[IMG]` probe to zero image tokens against a text count of one, so the engine never reaches weight loading.** Original basis: mature quant ecosystem; release targeted stronger function calling |
| Phi-4 14B | dense, Q4 | ~9GB | SOB Value Accuracy 0.798 (top small model); **16K ctx — screening must verify graphiti's longest prompts fit** |
| MoE stretch: ~~Qwen3.6-35B-A3B or~~ **Gemma-4-26B-A4B-it (QAT)** | ~~GGUF IQ4/Q4~~ **`UD-Q4_K_XL`** | ~~≈17GB (Qwen IQ4 — real, but 16.51 GiB of weights before KV, so it does not fit the measured 16.4 GiB)~~ → **13.27 GiB, fits** (α step 22, Open Q3; `task/3713` @ `a161c2858b`, not yet on `main`) | 115–133 tok/s on a 3090 (6× dense-on-vLLM) — but llama.cpp silently falls back to *unconstrained* output on Pydantic `$ref`/`$defs` schemas (llama.cpp #21228), so this arm runs `json_object` mode + a hard client-side validator; tightest VRAM |

*Reading the **Est. VRAM** column: those are 2026-08-05 research-time **estimates**, and at least one
is stale against measured reality — Qwen3.5-9B is listed `~6GB` here, but
`scripts/local-model-serving/arms.yaml` records a **measured 11.21 GiB** (vLLM-reported, 2026-08-06).
Refreshing this column is **task 3973**, sequenced after this correction (it depends on 3804); this
pass deliberately corrected only slate composition, so read every figure in the column as an
estimate, not a measurement.*

The struck-through row above is **kept** rather than deleted: it is the record of what was
commissioned, and deleting it would erase the fact that the slate narrowed. Two facts about the
dropped arm, so no reader re-litigates a closed defect (authoritative record:
`scripts/local-model-serving/arms.yaml`, the `mistral-small-3.2-24b — DROPPED FROM THE SLATE`
block):

- *The declared quant was wrong, and that defect is **fixed and verified**, separately.* `awq` →
  `compressed-tensors`, read from the downloaded weights' own `config.json`; vLLM 0.26 then accepted
  the model and resolved `max_model_len` 16384. The arm still never reached weight loading, for the
  unrelated tokenizer reason above — the card never moved (7212 → 7221 MiB). Do not reopen the quant
  question; it is closed.
- *Re-admission needs a different quantized repo (or an upstream tokenizer fix) — **not** a flag that
  suppresses the multimodal path.* Suppressing it would mean the eval measured a model configured
  differently from the one this PRD costed.

> **Consequence for η's screening funnel — the LLM slate is THREE arms, not four.** The remaining
> arms are **Qwen3.5-9B**, **Phi-4 14B**, and the **MoE stretch arm (Gemma-4-26B-A4B-it QAT)**.
> Two things follow, and both matter for reading η's report:
>
> 1. **The ≤3 cap's selectivity is now exactly nil.** The funnel narrows 3 → at most 3: three
>    candidates against a cap of three can eliminate nobody *by ranking*. The cap no longer binds.
> 2. **The funnel's absolute gates remain fully live and can still drop arms** — schema-conformance
>    smoke, VRAM fit under the measured ~16.4 GiB (D10), Phi-4's 16K context fit (Open Q2 already
>    reserves "drop the arm if it doesn't fit with margin"), and ζ's pre-registered throughput
>    floor. So η may still legitimately return **fewer than three** survivors; it just can no longer
>    return fewer *because of the cap*. Do not read "3 → at most 3" as "everything survives by
>    construction".
>
> **Whether the nil cap selectivity warrants re-opening the slate — a different Mistral quant repo,
> or another candidate entirely — is Leo's call. It is surfaced here and deliberately not decided.**
> One fact bearing on that call, stated as a fact and **not** as a recommendation: this PRD's own
> research appendix already named **gpt-oss-20b** as the reserve, "screened out only if slate must
> shrink". The slate has now shrunk, so that pre-existing conditional has **fired**. Naming it is
> surfacing, not selecting; no replacement candidate is chosen anywhere in this correction.

Embedding arms (serving: TEI or vLLM-pooling, OpenAI-compatible `/v1/embeddings` — screening picks;
incumbent `text-embedding-3-small` runs as its own arm):

| Arm | Params | Dims | Basis |
|---|---|---|---|
| Qwen3-Embedding-0.6B | 0.6B | 1024 (MRL) | MTEB(eng,v2) 70.70; Apache 2.0; ~1.2GB; query-side instruct prefix required |
| granite-embedding-english-r2 | 149M | 768 (MRL) | IBM 6-benchmark composite 59.5 (top of its tested group); fastest mid-size; no prefix |
| Qwen3-Embedding-4B | 4B | 2560 (MRL, can emit 1536) | MTEB(eng,v2) 74.60 — quality ceiling. **No 2B variant exists** (family is 0.6B/4B/8B, verified 2026-08-05); if 4B squeezes, use its GGUF Q4 (~2.5GB). Eval runs it as a batch job (LLM server stopped) — residency only matters at cutover |
| gte-modernbert-base | 149M | 768 | second small-model pole; no prefix |

External-anchor caveat (carried into the report): the incumbent's 62.3 MTEB is a 2024 v1-era score
not comparable to the candidates' MTEB-v2 numbers, and no apples-to-apples retrieval-only column
exists across candidates — which is exactly why **our own replay-based known-item retrieval eval on
the real corpus is the primary instrument** and public benchmarks are a sanity anchor only.

## Resolved design decisions

1. **Eval-only PRD; follow-ups gated on verdicts** (Leo). No cutover or migration tasks here.
2. **Pre-registered non-inferiority** (Leo). Margins derived from measured control variance by a
   committed formula; latency envelope anchored to the config-verified 120s write timeouts
   (`config.yaml:59,61`) with the arm's p95 required inside it with headroom.
3. **Client-class parity**: all LLM arms via `OpenAIGenericClient`; the client-class delta measured
   on the incumbent (OpenAIClient vs GenericClient control pair). A production cutover would also
   mean adopting GenericClient — the eval measures the production configuration.
4. **Axis isolation**: LLM axis fixes the embedder; embedding axis re-embeds one frozen graph, no
   LLM involved.
5. **Two retrieval configurations** on the embedding axis; *with-indices is primary* (future prod);
   the current-prod embedding-only configuration is reported alongside, with the confound stated.
6. **Funnel**: LLM axis screens all 3 (was 4 — Mistral-Small-3.2-24B dropped 2026-08-06), replays
   ≤3; for what the narrowed slate does to the cap's selectivity see the consequence note at the
   candidate slate (canonical — not restated here). Embedding axis runs all arms in full.
7. **Dims are a free variable**: the backfill is forced regardless, so candidates are compared at
   their native/MRL-best dims, not forced to 1536. (Mixed dims break cosine — but cutover atomicity
   is the follow-up PRD's problem; this eval only quantifies re-embed throughput to inform it.)
8. **Telemetry lands in the product, not just the harness**: `duration_ms` on `backend_ops` and
   surfaced token counts are permanent operator-visible instrumentation added by this PRD.
9. **Scratch-graph guard is a hard rejection**: harness writes only to `evalmem_`-prefixed
   group_ids; anything else raises a typed error, and the boundary test observes the rejection fire.
10. **whisper-writer stays resident** (Leo): all capacity math against ~19–20GB, not 24GB.
    - **Superseded 2026-08-06 — the operative figure is the measured ~16.4 GiB, not the ~19–20GB in
      the ruling above (kept verbatim by design — see the plan's design decisions).** Direct
      measurement (task 3748, pre-1): `nvidia-smi --query-gpu=memory.total,memory.used,memory.free`
      → `24576, 7309, 16813` MiB ⇒ **16.42 GiB free** (consistent with the architect's plan-time
      reading of `24576, 7312, 16811` MiB ⇒ 16.42 GiB, and with the task-3713 steward's 2026-08-05
      reading; small drift across readings is normal desktop jitter, not signal).
    - **Why the nominal figure was wrong (the mechanism):** `nvidia-smi --query-compute-apps`
      enumerates only CUDA *compute* applications — here exactly one row, whisper-writer at
      4050 MiB — and does **not** enumerate the KDE/X11 desktop's graphics contexts at all. The
      remaining **~3.18 GiB (3259 MiB, measured — task 3748 pre-1; other readings across this doc
      land at 3259–3262 MiB, consistent within desktop jitter)** is that desktop. Therefore any
      `24GB − whisper-writer` arithmetic overstates headroom by that same ~3.18 GiB on a host running
      a desktop session; the nominal 19–20GB was an **arithmetic derivation**, never a measurement.
    - The ruling above (whisper-writer stays resident) is **unchanged** — only the capacity number
      it implies is corrected.
    - **Consequence** (not a decision — see Open Q3): on a **weights-only** basis — the same basis
      that disqualifies the MoE arm below — the two remaining dense LLM arms (~6, ~9 GiB — the
      ~14 GiB Mistral-Small-3.2-24B arm was dropped 2026-08-06, see the candidate slate) and all
      embedding arms fit inside ~16.4 GiB; the MoE stretch arm as then specified (~17GB, i.e.
      Qwen3.6-35B-A3B at IQ4) does not. **Caveat on the `~6, ~9` figures, marked here at the point of
      use:** they are the 2026-08-05 research-time estimates, and the `~6` is known-wrong —
      arms.yaml records Qwen3.5-9B at a **measured 11.21 GiB**, ~5 GiB above the estimate this
      sentence's "fits" conclusion leans on. Refreshing the Est. VRAM figures is **task 3973**'s pass,
      deliberately not this one's (both would edit this paragraph); until it lands, treat the
      conclusion as resting on stale inputs. **Runtime fit is stricter and separate**: vLLM's paged KV
      cache can balloon well past the weights figure (α's README, `RESOLVED alongside: the pooling
      arms' KV balloon` — a 0.6B embedding arm declared at 2.0 GiB weights measured 16.2 GiB resident
      before `--kv-cache-memory` / `_memory_share_for` bounded it), so "fits" for the dense arms is
      not established here — it is α's to confirm per-arm under that cap. Surfaced at the arm table
      and Open Q3 rather than decided here — and α step 22 subsequently **resolved** the MoE sizing
      question against this measured figure by pinning Gemma-4-26B-A4B-it QAT `UD-Q4_K_XL` at
      13.27 GiB (also weights-only), which fits the same comparison. See Open Q3.
    - **Subject, made explicit:** ~16.4 GiB above is measured **free** VRAM — the pool an arm's own
      footprint draws from — and is **not** a ceiling on total card usage. Total usage necessarily
      includes the ~7.2 GiB whisper-writer + KDE/X11 desktop baseline, so a `total_used ≤ 16.4 GiB`
      reading is a different, much stricter claim that this correction does **not** make. Which of
      the two readings α's health verdict enforces was an open question when this correction was
      written; α resolved it (esc-3713-6, step 23) in favour of the **arm-footprint** reading —
      `evaluate_budget` judges `used − baseline` against the free VRAM measured just before that arm
      started. Pinned to `task/3713` @ `b3745f5a5c` (locator: `lms_vram.py::evaluate_budget` — see
      the banner above for why the function, not a README heading, is the stable pin), **not yet on
      `main`**; re-read the branch tip rather than treating this line as current.
    - Cites: memory `c01e7d1b-2916-4a8d-8f6e-c5e42692ce3d` (authoritative measurement),
      `38a4fcf2-30ba-4884-82f9-412737ddda13` (contradiction resolution).
11. **Long runs in transient `systemd --user` units**, never bare background shells.
12. **No conflation-rate metric** — that number is owned (and about to be zeroed) by the
    referent-fidelity PRD; using it would confound both directions.

## Hazards (binding on every task in this PRD)

- **Evidence-destroying:** upstream `FalkorDriver.__init__` fire-and-forgets
  `build_indices_and_constraints()` whenever an event loop runs
  (`.venv/.../graphiti_core/driver/falkordb_driver.py:161-169`). Constructing one in an async
  script against a real graph **creates indices and destroys the protected no-index evidence**
  owned by `docs/prds/falkordb-index-provisioning.md`. Never construct upstream drivers against
  real graphs. Never create indices on any real graph. Protected graphs: `dark_factory`, `reify`,
  `know_live`, `solar_challenge_platform`, `autopilot_video`, `pump_web_ui`, `my_solar_challenge`,
  `probe_e1_master`, `_probe`.
- Live-graph probes are read-only via `docker exec docker-falkordb-1 redis-cli GRAPH.RO_QUERY`.
- Scratch graphs use throwaway `evalmem_*` names; indices may be built freely **there only**.
- Corpus sampling must not condition on incumbent outcome (fixture-filtering lesson, 2026-08-04).
- Never `git stash` in any dark-factory checkout.

## Pre-conditions for activating (G3)

Verified-existing substrate: vLLM structured outputs via OpenAI-compatible `json_schema`
(research-verified against current docs, 2026-08-05); `OpenAIGenericClient` with
`structured_output_mode` in installed graphiti_core 0.28.2 wheel; embedder `base_url` plumbing
(`graphiti_client.py:533-539`); `shared/memory_eval_metrics` schema home; transcript-query corpus
tooling; `maintenance/reindex.py` re-embed machinery; episode store (~2,635 dark_factory episodes)
readable; GPU headroom measured directly (~16.4 GiB free with whisper-writer and the desktop
resident — see D10; the earlier 24GB − ~4GB derivation overstated it by ~3.18 GiB, D10's pinned
desktop-baseline figure).

Gaps that are **prerequisite tasks in this batch** (not assumed): LLM `base_url` + client-class
plumbing (β); Mem0 LLM/embedder `base_url` + `embedding_model_dims` plumbing (β); duration/token
telemetry (γ); serving units (α). Nothing else novel is assumed.

## Cross-PRD relationship (G4)

| Other PRD / owner | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/falkordb-index-provisioning.md` | consumes its finding; must not disturb its evidence | index state on real graphs; index-build recipe reused on scratch graphs | index remediation: **that PRD**; scratch-graph index usage + confound framing: **this PRD** | wired (hazard block) |
| `plans/memory-referent-fidelity-prd.md` (3666-3676) | timing interaction only | Graphiti dedupe call profile (its γ removes LLM dedupe for `Task N` refs) | each PRD owns its own tasks; **this PRD pins one code SHA per comparison** to stay valid either side of its landing | wired (D-item 12, instrument checks) |
| `docs/prds/memory-eval-program.md` | consumes | `shared.memory_eval_metrics` M1 artifact schema; transcript-query corpus artifacts; E1 probe metric math (imported, not copied — INV-5) | schema + probe: **that PRD**; this PRD's arm-comparison runner: **this PRD**. If reuse requires splitting the probe file (a split its own docstring already plans), that edit belongs to the eval-program lane — coordinate, don't fork | wired |
| `docs/prds/memory-briefing-and-fusion` (3658-3660) | none | it rewrites *fused* ranking/briefing; this PRD measures *per-store* retrieval below that layer | n/a | noted |
| Orchestrator model routing (`OPERATIONS.md` §7) | none | `routing.allowed_models` governs claude-backend agent roles only; the memory-path model is not routed through it | n/a — admission of local models as *agent* models is explicitly out of scope | verified N/A |

## Contract (B+H-lite)

**ArmSpec** (pydantic, in the harness package; validated at run start):
`{arm_id, axis: llm|embedding, model_id, serving: {stack: vllm|llamacpp|tei|openai, base_url, quant,
unit_name}, client_class: openai|openai_generic, structured_output_mode: json_schema|json_object,
params: {temperature, max_tokens}, code_sha, corpus_sha, preregistration_sha|null,
scratch_group_id}` — `scratch_group_id` MUST match `^evalmem_[a-z0-9_]+$`; the constructor raises
otherwise (the protected-list is unreachable by construction, and the guard is tested by observed
rejection).

**MetricsRecord**: one artifact per (arm, metric), conforming to `shared.memory_eval_metrics`'s M1
series conventions (schema imported, not restated), carrying `arm_id`, `metric_id`, value, n,
`measured_at`, and the three SHAs. Primary metric ids —
LLM axis: `conformance-rate` (schema-valid responses / calls, retries counted),
`episode-failure-rate` (terminal failure or >120s), `episode-latency-p50/p95` (under concurrent
replay), `graph-sameness` (per-episode entity/edge counts + normalized-name Jaccard vs incumbent
reference), `retrieval-utility` (known-item recall@k on the arm's graph, incumbent embedder),
`tokens-per-episode` / `usd-per-episode`.
Embedding axis: `known-item-recall@5/10` and `mrr` (per index configuration), `query-embed-latency-p95`,
`reembed-throughput` (vectors/s → projected full-backfill wall-clock), plus the public-benchmark
anchor row.

**Pre-registration artifact**: `plans/local-memory-models-eval-preregistration.md`, committed by ζ
BEFORE any candidate arm runs. Contains: the margin-derivation formula bound to named control-run
artifacts (e.g. margin_m = max(2·σ_control(m), floor_m) with each floor justified or absent), the
per-metric pass/fail direction, the latency envelope (p95 < 120s with stated headroom factor), the
decision rule per axis, and the survivor rule for the screening funnel. Candidate-arm artifacts
missing a matching `preregistration_sha` are invalid by schema.
**Constraint on that survivor rule (authored 2026-08-10, task 3804; per Leo's 2026-08-06 drop
ruling — the ruling date and the authoring date are different and are kept apart deliberately):** it
MUST be authored against the **three-arm** LLM slate (authoritative: `scripts/local-model-serving/
arms.yaml`; for what the narrowing does to the cap, see the consequence note at the candidate slate
— not restated here), MUST NOT presume a 4 → 3 narrowing, and MUST state explicitly whether the ≤3
cap binds at all given three candidates — so η reports a substantive result rather than a vacuous
"all candidates survived". This constrains *how* ζ writes the rule; it does not pre-empt *what* the
rule says.

**Failure/storm rule (INV-4)**: the replay engine aborts an arm run after 5 consecutive item
failures with a structured error record (arm, item ids, error class) — no silent absorb-and-continue;
partial artifacts carry `incomplete: true`.

**Teardown**: scratch graphs and Qdrant replica collections are deleted only by a harness helper
that re-validates the `evalmem_` prefix at deletion time.

## Boundary-test sketch (integration-gate signals for ε)

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Endpoint conformance smoke | serving unit up for arm X | `json_schema`-constrained request returns schema-valid JSON; a deliberately-invalid schema response path is detected as failure, not silently accepted |
| Scratch-guard rejection | ArmSpec with `group_id="dark_factory"` | typed rejection raised at construction; no driver built, no write occurs |
| Client-class parity control | incumbent key valid | OpenAIClient and GenericClient runs over the same 20-episode subset both complete; delta recorded as a MetricsRecord |
| Control variance → margins | two incumbent GenericClient replays complete | margin formula yields finite margins; instrument checks pass (symmetric params, non-empty reference artifacts, token counts > 0) |
| Frozen-graph re-embed integrity | reference graph frozen | per-arm re-embedded graph has identical topology hash (nodes+edges), differing only in vectors/dims |
| Index-configuration reality | one arm's scratch graph, indices built | fulltext leg returns >0 rows for a seeded query (proving the with-indices config differs from embedding-only in fact, not in name) |
| Telemetry presence | γ landed; one replayed episode | journal row carries `duration_ms > 0` and token counts > 0 |

## Decomposition plan

Greek labels are PRD-local; IDs assigned at decompose. All tasks carry the Hazards block as binding
constraints. Signals are the G2 candidates; capability bindings drafted here get mechanized in the
manifest at decompose time.

| # | Task | Modules | Kind | Signal (user-observable) | Prereqs |
|---|---|---|---|---|---|
| **α** | Serving substrate: candidate endpoints as `systemd --user` units (vLLM structured-outputs for dense LLM arms; llama.cpp for the MoE arm; TEI or vLLM-pooling for embedders), weights on disk, VRAM caps set for whisper-writer coexistence, health-check script | ops scripts (`scripts/`), no product code | operational | health script output lists every candidate endpoint answering a schema-constrained completion (LLM) / an embeddings call (embedder) with valid output, and `nvidia-smi` confirms the arm's own footprint fits the measured ~16.4 GiB of free VRAM (D10). NOTE: α resolved the verdict's subject in favour of arm-footprint-vs-free (esc-3713-6, step 23), pinned to `task/3713` @ `b3745f5a5c` — not yet on `main`, so confirm against the branch tip and α's README when judging this signal. NOTE: α also **dropped the Mistral-Small-3.2-24B arm** on live measurement (Leo, 2026-08-06, esc-3713-10) — the substrate refused an arm it cannot serve, before the eval could attribute numbers to it; see the candidate slate. | — |
| **β** | Config + client plumbing: `llm.client_class` knob (`openai`\|`openai_generic`), LLM `base_url` honored (`graphiti_client.py:502-509`), Mem0 LLM/embedder `openai_base_url` + `embedding_model_dims` plumbed (`mem0_client.py:138-167`), reindex tool `base_url` (`reindex.py:160-165`); default config byte-identical behavior | `fused-memory/src/fused_memory` | normal | integration test: a config naming a local base_url + generic client constructs clients that hit a local mock server; with the shipped config, construction is behaviorally unchanged (existing tests green) | — |
| **γ** | Durable write telemetry: `duration_ms` on `backend_ops` rows (`_journaled_backend_call`, `memory_service.py:1302-1337`) + per-write token usage surfaced from graphiti's `TokenUsageTracker` into the journal `result_summary` | `fused-memory/src/fused_memory` | normal | after any live memory write, the documented read-only sqlite query shows the new row carrying `duration_ms` and token counts — permanent operator observability, consumed by ε and by operators | — |
| **δ** | Corpus builder: stratified sample (~150–300, size finalized in-task from control-variance needs) of real dark_factory episodes across time and payload kind, explicitly **not** conditioned on incumbent outcome; committed manifest (ids + content hashes + stratification report + the no-outcome-filter statement) | `fused-memory/scripts` (read-only against episode store) | normal | committed corpus manifest; a reviewer can re-derive the sample from the manifest's recorded criteria | — |
| **ε** | Arm-runner harness: ArmSpec + scratch-guard, replay engine over the real `GraphitiBackend` construction path onto `evalmem_*` graphs, MetricsRecord emission (importing `shared.memory_eval_metrics`), instrument-validation checks, INV-4 abort rule; boundary tests above | `fused-memory/scripts` + harness module, tests | normal | all boundary-test rows green in CI, including the observed scratch-guard rejection | β, γ, δ |
| **ζ** | Controls + pre-registration: incumbent GenericClient replay ×2 (one graph frozen as the embedding reference), OpenAIClient control pair, margin derivation, commit `plans/local-memory-models-eval-preregistration.md` | harness runs + `plans/` | normal | pre-registration doc committed with finite derived margins AND control MetricsRecords committed; git history shows it predates every candidate-arm artifact | ε |
| **η** | LLM screening: all 3 remaining candidates × small subset (was 4; see the candidate slate for the drop and what it does to the cap) — conformance smoke, VRAM fit, prompt-length fit (Phi-4's 16K vs measured longest graphiti prompt), throughput floor; survivor selection per the pre-registered rule | harness runs | normal | committed screening report naming survivors (≤3) with per-candidate evidence — the report must also name the dropped arm and state whether the ≤3 cap bound at all, so the narrowed slate is legible rather than silent | α, ε, ζ |
| **θ** | LLM full arm runs + comparison report: survivors × full corpus, realistic concurrency (fan-out ≤20) on the busy box in transient units; report vs pre-registered margins incl. cost/availability quantification | harness runs + `plans/` | normal | committed LLM-axis comparison report; every arm artifact embeds preregistration_sha + code_sha (schema-enforced) | ζ, η |
| **ι** | Embedding arms: re-embed the frozen reference graph + one Mem0 replica collection per candidate (and incumbent-as-arm), build with-indices and embedding-only variants, run known-item + transcript-query probes, measure query-latency and re-embed throughput; report | harness runs + `plans/` | normal | committed embedding-axis comparison report covering both index configurations, with the current-prod confound stated | α, β, ζ |
| **λ** | Synthesis: apply pre-registered rules; write two decision records (committed to `plans/` + `add_memory` `decisions_and_rationale`); raise the operator gate for Leo's ruling; name the follow-up PRD(s) the verdicts warrant | `plans/`, memory | normal | **leaf** — two committed decision records; the operator-gate escalation resolved by Leo referencing them | θ, ι |

G1 note: λ's consumer is the operator ruling surface (the project's standing decision-record
pattern); β's and γ's consumers are ε *in this batch* plus a named permanent surface (γ: the
documented operator query; β: the config surface any later cutover uses — but β is justified by ε
alone, no reliance on unfiled PRDs). G7 walk (advisory, author mode): INV-1 → contracts are pydantic
schemas + a schema-enforced preregistration_sha, not prose; INV-2 → structured MetricsRecords and
structured abort records, no log-scraping; INV-3 → scratch-guard revalidates at write and teardown;
INV-4 → the 5-consecutive-failure abort rule; INV-5 → metrics schema and probe math imported from the
eval program, construction path reused from fused-memory; INV-6/7 → runs live in supervised transient
units, the single human hold is the λ operator gate on the standard age-surfaced queue.

## Out of scope

- **Production cutover of the LLM** and **the embedding backfill/migration** (collections rebuild,
  FalkorDB vector-index rebuild, `calibrate_write_triage` re-run — its `t_high`/`t_low` are
  embedder-space-dependent, `config.yaml:249-263`; `maintenance/reindex.py` is the substrate).
  Follow-up PRDs, authored after the verdicts.
- Index remediation on real graphs (`docs/prds/falkordb-index-provisioning.md`).
- The `claude_cli` LLM paths (reconciliation agent/judge, curator, consolidator) — different
  provider, different failure domain.
- The `Task N` conflation defect and its metrics (referent-fidelity PRD).
- Admitting local models as orchestrator *agent* models (`routing.allowed_models`).
- SGLang — revisit only if vLLM fails the pre-registered throughput floor at screening.

## Open questions (tactical, decided in-task)

1. **Embedding serving stack** (TEI vs vLLM-pooling vs in-process for batch re-embeds). Suggested:
   whatever screening shows serving query-latency best; batch re-embeds may run in-process. Decide in α/ι.
2. **Phi-4 context fit** — measure graphiti's longest real prompt at screening; drop the arm if it
   doesn't fit with margin. Decide in η.
3. **MoE stretch-arm engine details AND sizing** (which of the two models, GGUF quant level,
   client-side validator placement). The sizing half became live when D10's budget was corrected:
   the arm's ~17GB estimate does not fit the measured ~16.4 GiB (the comparison had previously been
   made against the stale 19–20GB nominal figure).
   **RESOLVED by α step 22 — `unsloth/gemma-4-26B-A4B-it-qat-GGUF`, quant `UD-Q4_K_XL`, 13.27 GiB**
   (pinned to `task/3713` @ `a161c2858b`, **not yet on `main`** — confirm against the branch tip;
   full table in `scripts/local-model-serving/README.md` §`Open Q3`). Decided from real GGUF file
   sizes read from the HF API, **against the measured 16.4 GiB**, which is precisely what the
   correction in this amendment existed to ensure. The PRD's ~17GB estimate was for Qwen3.6-35B-A3B
   and is accurate — its smallest true 4-bit quant is 16.51 GiB of weights before a single KV byte,
   so it *would* have fitted the nominal 19.5 GiB and does not fit the measured 16.4. Gemma's QAT
   weights are quantization-aware *trained* at 4 bits, so the arm that fits is also the one that
   does not trade quality to fit. This lands as option (ii) of the four recorded when the question
   was surfaced (re-quantize / swap model / require the desktop's ~3.18 GiB be freed / drop the
   arm); nothing needed to be freed and no arm was dropped.
4. **Corpus size N** (~150–300) from control-variance and wall-clock measured in ζ. Decide in δ/ζ.
5. **Qwen3-Embedding-4B production-residency estimate** (quantized footprint next to the winning
   LLM) — only needed if 4B wins on quality. Decide in ι/λ.

## Research appendix (sources, retrieved 2026-08-05)

Serving: vLLM structured outputs (xgrammar/llguidance backends, OpenAI-compatible `json_schema`) —
vllm docs + SqueezeBits guided-decoding benchmark (conformance 81–100% with guided decoding);
llama.cpp silent unconstrained-fallback on `$ref`/`$defs` schemas — ggml-org/llama.cpp#21228 (plus
#25746, #22072); Ollama constrained decoding weaker/less verifiable for concurrent production;
graphiti `OpenAIGenericClient` + `structured_output_mode` — getzep/graphiti PR#1227; graphiti
local-endpoint rough edges — issues #912, #1116 (api_base ignored → silent fallback to
api.openai.com; re-verified against our own plumbing in β), #868, #1074.
LLM candidates: Qwen3.5-9B official HF card (IFEval 91.5, BFCL-V4 66.1); Mistral-Small-3.2 release
notes (retained: this appendix records what was researched on 2026-08-05, not which arms are live —
the Mistral arm was dropped 2026-08-06, see the candidate slate); Phi-4 SOB score arXiv:2604.25359;
gpt-oss-20b model card arXiv:2508.10925 (reserve candidate, screened out only if the slate must
shrink — Leo selected four arms on 2026-08-05; the slate is now three after the 2026-08-06 drop, so
this condition has **fired**. Whether to admit a reserve is Leo's call, surfaced at the candidate
slate and not decided here); 3090 throughput: tfriedel/qwen3.6-rtx3090-lab
(single-source, treated as directional).
Embedders: Qwen3-Embedding family (0.6B/4B/8B — **no 2B**) QwenLM GitHub + HF cards, MTEB(eng,v2)
70.70/74.60; granite-embedding-english-r2 arXiv:2508.21085 (Fig.1 composite 59.5; 144 docs/s H100);
gte-modernbert-base HF card; incumbent anchor: OpenAI announcement (62.3 MTEB v1-era, $0.02/1M).
Unverified-claims lists from both research passes are preserved in the session transcript; every
number above that drives a design decision was either primary-sourced or is re-measured by this
eval before use.
