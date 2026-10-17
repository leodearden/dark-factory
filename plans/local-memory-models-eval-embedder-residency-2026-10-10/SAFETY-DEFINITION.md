# Pre-registered "safe" definition: embedder residency beside WhisperWriter

Written 2026-10-10T17:10Z, BEFORE any embedder unit was started or any measurement taken.

Host: one RTX 3090, 24576 MiB. WhisperWriter (`whisperwriter.service`, pid 9024,
faster-whisper 1.2.1 / CTranslate2 4.8.0, large-v3, float16, beam 5 default) is resident
at about 4050 MiB. Desktop baseline brings the device to ~7.3 GiB used.

The arm is SAFE for production residency iff ALL of the following hold for it.

S1 — Combined peak memory. The device `memory.used` peak, sampled every 100 ms by
     `nvidia-smi`, stays ≤ 85% of 24576 MiB = 20889 MiB during the worst phase: the
     embedder under a backfill-rate burst with max-length inputs, while a Whisper large-v3
     decode runs concurrently. The stress phase runs a SECOND Whisper large-v3 copy (the
     probe) on top of the resident WhisperWriter. That makes it stricter than production,
     which has one copy.
S2 — No OOM and no eviction. There is no CUDA out-of-memory or allocation error in the
     embedder unit's journal, in the probe's output, or in the WhisperWriter journal over the
     window. WhisperWriter pid 9024 is still alive afterwards, and its per-process
     `used_memory` is unchanged (still resident).
S3 — Whisper functional. The probe transcribes a known speech clip (openai/whisper
     `tests/jfk.flac`) through the same model path WhisperWriter uses (large-v3, cuda,
     float16, language=en, temperature 0, condition_on_previous_text False, vad off). It must
     do so with the embedder loaded, and also while it is busy at backfill rate. The
     normalised transcript must match the reference (word-level match ≥ 0.95).
S4 — Bounded embedder footprint. The embedder process's resident memory after all bursts is
     ≤ its post-load resident + 256 MiB. Growth beyond that means unbounded growth or
     fragmentation.
S5 — No dictation slowdown at query rate. With the embedder serving fused-memory's measured
     query-path rate, the probe's median transcription latency on the clip is ≤ 1.25× its
     median with no embedder present.

Reported, not gating:
- Whisper latency during the backfill burst. A backfill is a one-off of minutes. If it
  slows dictation, the PRD must schedule or throttle it, and these numbers size that.
- The production-equivalent peak: baseline + embedder peak delta + one Whisper decode
  transient.

Borderline: within 10% of any bound (S1 above 18800 MiB; S5 above 1.15×). Borderline or
unsafe means STOP and report to Leo with the numbers. Do not proceed to the PRD.

Limit stated up front: WhisperWriter's OWN process is not exercised end to end. That
needs Leo's voice and F8, and its output is typed into the focused window. S3 exercises
the identical model path in a second process. S2 checks that the resident process
survives untouched.
