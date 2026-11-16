# Evidence: local embedder VRAM residency beside WhisperWriter (2026-10-10)

The report is `plans/local-memory-models-eval-embedder-residency-2026-10-10.md`.

- `SAFETY-DEFINITION.md`: the pass/fail bar, written at 17:10Z before any measurement.
- `probe.py`: the second faster-whisper process plus the embedding load generator. It runs
  under `/home/leo/src/whisper-writer/.venv` and its phases are `baseline`, `loaded`,
  `loaded2` and `ab`.
- `run_phase.sh`: sets up WhisperWriter's CUDA library path, starts two `nvidia-smi`
  samplers (device every 100 ms, per-process every 250 ms), then runs one probe phase.
- `analyse.py <logs/tag>`: per-phase device peaks, per-pid peaks, Whisper latency, load
  summaries.
- `logs/<tag>/`: `gpu.csv`, `apps.csv`, `events.jsonl` and `probe.log` for each run:
  - `baseline`: no embedder;
  - `granite`, `granite2`, `granite_ab`: granite-embedding-english-r2;
  - `qwen06`, `qwen06_ab`: qwen3-embedding-0.6b.

Two inputs are deliberately not committed:
- `audio/jfk.flac`: https://github.com/openai/whisper/raw/main/tests/jfk.flac, sha256
  `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`.
- `data/mem0_texts.json`: 6000 `payload.data` strings scrolled read-only from Qdrant
  `fused_dark_factory`. They are memory content, so they stay out of git.

To reproduce, recreate both inputs beside `probe.py`.

Timestamps in `gpu.csv`/`apps.csv` are host local time (BST); `events.jsonl` `t` is epoch
seconds.

`probe.py` and `analyse.py` were lint-cleaned for ruff after the runs that produced
`logs/`, and their behaviour is unchanged:
- an unused import was dropped;
- loop and lambda variables were renamed;
- the event log opens its file per write instead of holding one handle;
- the texts are read with `Path.read_text`.
