# Local embedder VRAM residency beside WhisperWriter (LME-μ, task 3725)

PRD-MARKER:local-memory-models-eval embedder-residency

**Measured** 2026-10-10, 17:12Z–17:38Z, on the one RTX 3090 (24576 MiB) that also runs
`whisperwriter.service` (faster-whisper 1.2.1 / CTranslate2 4.8.0, Whisper large-v3,
float16, pid 9024, resident at 4050 MiB). **Why:** esc-3725-1 asked Leo to rule on the
embedding axis, and λ's decision record
(`plans/local-memory-models-eval-decision-embedding.md` §3) said production-residency VRAM
beside WhisperWriter "was not measured". Leo made "iff safe" the precondition for the
cutover PRD. **Verdict: SAFE for both admitted arms, on every pre-registered criterion,
with wide margins.** Raw logs, scripts and the pre-registration live in
`plans/local-memory-models-eval-embedder-residency-2026-10-10/`.

## 1. The bar, written before measuring

`SAFETY-DEFINITION.md` in the evidence directory was written at 17:10Z, before any unit
started. It is quoted here in short:

| # | Criterion | Bound |
|---|---|---|
| S1 | Device `memory.used` peak (100 ms `nvidia-smi` sampling) in the worst phase | ≤ 85% of 24576 = **20889 MiB**. Measured with a SECOND Whisper large-v3 copy decoding on top of the resident one, so it is stricter than production |
| S2 | No CUDA OOM or allocation error (embedder journal, probe, WhisperWriter journal); WhisperWriter pid alive, its `used_memory` unchanged | none |
| S3 | The probe transcribes `jfk.flac` through WhisperWriter's own model path and options, with the embedder loaded and busy | word match ≥ 0.95 |
| S4 | Embedder footprint after all bursts | ≤ post-load + 256 MiB |
| S5 | Median dictation latency with the embedder at fused-memory's query rate | ≤ 1.25× no-embedder |

Anything within 10% of a bound was to count as borderline, and that meant stop and report.
Whisper latency during a backfill was declared reported-only.

## 2. Method

- **Serving.** Each arm ran through α's own path: `lms_ctl.py start <arm>`, then the
  `lms-arm@<arm>.service` unit, then `lms_serve.py`, then the digest-pinned
  `vllm/vllm-openai:v0.26.0` image in pooling mode. The share was bounded to
  `est_vram_gib + 0.5`, giving `--gpu-memory-utilization 0.062` for granite. α's baseline
  records were redirected (`LMS_BASELINE_DIR`) so the host's own records stayed untouched.
- **Probe.** `probe.py` runs a second faster-whisper large-v3 process from WhisperWriter's
  venv, with the same transcribe options as `whisper-writer/src/transcription.py`. Its clips
  are openai/whisper `tests/jfk.flac` (11 s; sha256 `63a4b1e4…949715`) and a 30 s
  concatenation of it, which hits Whisper's full window.
- **Load.** The generator runs inside the probe, with a stdlib HTTP client against
  `/v1/embeddings`:
  - Query paths are Poisson, single-text, using 25-word queries cut from real
    `fused_dark_factory` texts. The target rates were 5 and 50 per second; the achieved
    rates were 3.6–4.6 and 21.7–23.8 per second.
  - Backfill uses 6000 real Mem0 texts at batch 64 × concurrency 4, the setting ι used for
    `reembed-throughput`.
  - The max-length variant sends 8000-token inputs.
  - Write bursts are paced at 80 vectors/s.
- **Rates, from a read-only census of `data/reconciliation/write_journal.db`.** Each search
  op embeds the query at most twice, once per store. Over 7 days: max 15 ops/min, 11 per
  10 s. The 30-day max was 350 ops/min ≈ **11.7 query embeds/s** during the 2026-09-29/30
  recon storm. The write-path proxy max was 4773 embeds/min ≈ 80/s.
- **Latency A/B.** Ten interleaved rounds of {idle, queries 3.6/s, queries 12/s, writes
  80/s}, so host drift hits every condition equally. The host CPU was saturated by the
  factory throughout (load average 176–268, 1.5% idle), which is why Whisper latency
  jitters ±30% even with the engine idle.

## 3. Results

Device peaks include the probe's second Whisper copy (+4242–4338 MiB).

| arm | engine post-load → after bursts | worst device peak | production-equivalent peak* | query-rate dictation latency (A/B, paired median) | write-burst 80/s | full-speed backfill |
|---|---|---|---|---|---|---|
| none (baseline) | — | 11493 MiB (probe only) | 7245 MiB | 0.636 s (11 s clip) | — | — |
| **granite-embedding-english-r2** | **782 → 952 MiB** (plateau: 950 → 952 across a second full pass) | **12550 MiB (51.1%)** | **≈ 8391 MiB (34.1%)** | 0.971× at 4.6/s; 0.964× at 10.1/s | 1.075× | 500–517 vec/s; dictation 2.4–2.5× slower while it runs |
| qwen3-embedding-0.6b | 3800 → 3888 MiB | 15370 MiB (62.5%) | ≈ 11273 MiB (45.9%) | 0.991× at 4.3/s; 0.999× at 9.5/s | 1.186× | 154 vec/s; dictation 2.8× slower while it runs |

\*Production-equivalent = desktop + WhisperWriter + the engine's plateau + WhisperWriter's
own decode transient. That transient was 4050 → 4242 MiB, observed in pid 9024 itself
during Leo's dictation.

Throttled granite backfill (pass 2, compared with the same pass's idle reading of 0.826 s):
- one stream at batch 16 ran at 184 vec/s, with dictation at 0.880 s (1.07×);
- one stream at batch 64 ran at 235 vec/s, with dictation at 1.181 s (1.43×).

Max-length inputs (8000 tokens) pushed no further memory growth past the real-text peak on
either arm. Per-phase tables are in `logs/*/events.jsonl` and come out of `analyse.py`.

## 4. Verdict per criterion

| | granite-r2 | qwen3-0.6b |
|---|---|---|
| S1 peak ≤ 20889 | PASS, 12550 (8339 MiB headroom) | PASS, 15370 (5519 headroom) |
| S2 no OOM, no eviction | PASS: 0 embed errors over 1437 requests (35,115 vectors); no OOM or CUDA error in either journal; pid 9024 alive at 4050 MiB before and after | PASS: 0 errors over 847 requests (7678 vectors); same journal and pid result |
| S3 Whisper functional | PASS: word match 1.0 on all 91 scored 11 s-clip transcriptions, backfill phases included (the 30 s clips are timed, not scored) | PASS: 1.0 on all 65 |
| S4 footprint bounded | PASS, +170 MiB with a plateau shown | PASS, +88 MiB |
| S5 query-rate latency | PASS, 0.96–0.97× | PASS, 0.99–1.00× |

Neither arm is borderline. 12550 MiB is 60% of the S1 bound, and the S5 ratios are at
noise. Item 4 of the brief, both processes growing at once, was exercised in every backfill
phase: the probe's decode and the engine's batch peak landed together, and the totals above
are those concurrent peaks. Both the CTranslate2 and the PyTorch caching allocators keep
their peak reservation, so a 100 ms sampler catches transients that last less than one
sample.

## 5. WhisperWriter's own process, exercised by Leo

Leo dictated through pid 9024 during the run. Timings only:

| UTC | arm state | recording | transcribed in |
|---|---|---|---|
| 17:17:09 | granite engine finishing start-up | 6.99 s | 0.63 s |
| 17:17:16 | granite engine ready | 2.31 s | 0.58 s |
| 17:34:27 | qwen3-0.6b full-speed backfill + probe decoding | 12.24 s | 1.34 s |
| 17:35:03 | qwen3-0.6b max-length backfill + probe decoding | 10.80 s | 2.08 s |
| 17:35:55 | qwen3-0.6b max-length backfill + probe decoding | 3.48 s | 1.14 s |

He usually sees 0.5–1.2 s. The slow rows fell in the worst phases, while a second Whisper
model was also decoding on the card. No errors occurred, and the process's memory went
4050 → 4242 → 4050 MiB.

## 6. Limits

- **Single host state.** The card was measured under the factory's normal CPU load; one
  run per arm, plus a repeat pass for granite.
- **Embedder exercised directly.** fused-memory itself was not pointed at the embedder, and
  this phase touched no fused-memory config. Load came from a generator at measured rates.
- **Throughput is not ι's number.** Backfill throughput here covers Mem0 texts only, and
  graph names and facts embed faster (ι report §7). The figures size dictation impact; they
  do not replace ι's backfill projection.
- **No production unit tested.** The measured serving path is α's eval unit, so the
  permanent service's resident footprint must be re-checked when it lands. The cutover PRD
  carries that check.

## 7. Host left as found

- Both units were stopped. `lms-arm@granite-embedding-english-r2` was reset from the
  `failed` state that `docker stop`'s exit 137 leaves; qwen3-0.6b unloaded cleanly.
- No `lms-arm` container is left, and the device was back to 7162 MiB.
- WhisperWriter is pid 9024 at 4050 MiB, unchanged.
- α's baseline records in `$XDG_RUNTIME_DIR/lms-baselines` are unmodified (Sep 12 / Oct 7
  mtimes).
- fused-memory, its config and every orchestrator were not touched.
