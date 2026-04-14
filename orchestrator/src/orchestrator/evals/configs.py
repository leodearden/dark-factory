"""Eval configuration matrix: model/backend combinations to test."""

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """One model/backend configuration for evaluation."""

    name: str
    backend: str          # 'claude' | 'codex' | 'gemini'
    model: str            # model name for that backend
    effort: str | None    # backend-specific effort/reasoning level
    max_budget_usd: float = 20.0
    env_overrides: dict[str, str] = field(default_factory=dict)
    # vLLM-only pod fields (None / 0 for non-vllm or workstation configs)
    image: str | None = None              # Docker image tag
    gpu_type: str | None = None           # RunPod GPU type id
    gpu_count: int = 1
    container_disk_gb: int = 50           # ~50 for baked images, 2.5× model size for :latest


def _vllm_config(
    name: str,
    hf_model: str,
    *,
    image: str | None = None,
    gpu_type: str | None = None,
    gpu_count: int = 1,
    container_disk_gb: int = 50,
    quantization: str | None = None,
    tool_call_parser: str = 'hermes',
    effort: str = 'high',
    max_model_len: int | None = None,
    gpu_memory_util: float | None = None,
    enforce_eager: bool = False,
    override_generation_config: str | None = None,
    pp_size: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> EvalConfig:
    """Build an EvalConfig that routes through a vLLM-compatible endpoint."""
    env = {
        'ANTHROPIC_API_KEY': 'dummy',
        'ANTHROPIC_DEFAULT_SONNET_MODEL': hf_model,
        'MODEL_NAME': hf_model,
        'TOOL_CALL_PARSER': tool_call_parser,
    }
    if quantization:
        env['QUANTIZATION'] = quantization
    if gpu_count > 1:
        env['TP_SIZE'] = str(gpu_count)
    if max_model_len is not None:
        env['MAX_MODEL_LEN'] = str(max_model_len)
    if gpu_memory_util is not None:
        env['GPU_MEMORY_UTIL'] = str(gpu_memory_util)
    if enforce_eager:
        # Disables CUDA-graph capture (--enforce-eager) — workaround for the
        # qwen3-coder-next-fp8 startup hang (vLLM #35504). Requires the
        # entrypoint-vllm.sh hook in runpod-toolkit to take runtime effect.
        env['ENFORCE_EAGER'] = '1'
    if override_generation_config:
        # Fixes models with bogus eos_token_id in config.json — e.g.
        # lukealonso REAP NVFP4 has eos_token_id=2 from ModelOpt but
        # the real EOS is 200020. Without this, the model never stops.
        env['OVERRIDE_GENERATION_CONFIG'] = override_generation_config
    if pp_size and pp_size > 1:
        env['PP_SIZE'] = str(pp_size)
    if extra_env:
        env.update(extra_env)
    return EvalConfig(
        name=name, backend='claude', model='sonnet', effort=effort,
        env_overrides=env,
        image=image, gpu_type=gpu_type, gpu_count=gpu_count,
        container_disk_gb=container_disk_gb,
    )


RTX_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
H200 = "NVIDIA H200"
B200 = "NVIDIA B200"


VLLM_EVAL_CONFIGS = [
    # ===== Existing-image configs: OLD baked images on bigger pods =====
    # container_disk_gb on RunPod includes the image, so size = image_size + ~50 GB headroom
    # Qwen3-Coder-Next OLD: 149 GB bf16 baked, vllm 0.18.1, hermes hardcoded — needs 2x H200
    _vllm_config('qwen3-coder-next-fp8',
        hf_model='Qwen/Qwen3-Coder-Next',
        image='leosiriusdawn/runpod-vllm:qwen3-coder-next',
        gpu_type=H200, gpu_count=2, container_disk_gb=250),
    # REAP-139B OLD baked: 131 GB FP8 baked — fits 1x H200
    _vllm_config('reap-139b-nvfp4',
        hf_model='cerebras/MiniMax-M2.5-REAP-139B-A10B',
        image='leosiriusdawn/runpod-vllm:reap-139b',
        gpu_type=H200, gpu_count=1, container_disk_gb=220,
        quantization='fp8'),
    # REAP-172B OLD baked: 162 GB FP8 baked — needs 2x H200 (fits in 282 GB)
    _vllm_config('reap-172b-nvfp4',
        hf_model='cerebras/MiniMax-M2.5-REAP-172B-A10B',
        image='leosiriusdawn/runpod-vllm:reap-172b',
        gpu_type=H200, gpu_count=2, container_disk_gb=260,
        quantization='fp8'),
    # MiniMax-M2.5 OLD: no existing baked image, use :latest with HF download
    # 215 GB FP8 — needs 2x H200, container_disk for HF download (2.5× 215 GB)
    _vllm_config('minimax-m25-fp8',
        hf_model='MiniMaxAI/MiniMax-M2.5',
        image='leosiriusdawn/runpod-vllm:latest',
        gpu_type=H200, gpu_count=2, container_disk_gb=600,
        quantization='fp8'),

    # ===== New-image configs: new HF FP8/NVFP4 model variants =====
    # Qwen3-Coder-Next-FP8: ~75 GB on disk, baked into the image via the
    # shard-per-layer Dockerfile (40 × 2 GB COPY layers — see
    # runpod-toolkit/scripts/bake_model_image.py). The baked layout uses
    # the standard HF cache shape under /models/hub/, so MODEL_NAME stays
    # the HF name and the runtime resolves locally via TRANSFORMERS_OFFLINE.
    # Switched 2026-04-07 PM after the :latest + HF download path hit a
    # pre-download Python init hang that --enforce-eager did not fix.
    # Qwen3-Coder series emits <tool_call><function=name><parameter=name>val</parameter>
    # </function></tool_call> XML, not hermes JSON — must use qwen3_coder parser.
    # Hardware: 2× RTX PRO 6000 (192 GB aggregate); RunPod's per-DC layer
    # cache makes second-pull TTFT ≪ HF download once warm.
    # ENFORCE_EAGER=1 retained as a belt-and-braces against the
    # CUDA-graph-capture hang (vLLM #35504).
    # Qwen3-Coder-Next hangs on RTX PRO 6000 (SM120) even with ENFORCE_EAGER
    # — both TP=1 and TP=2 hang at NCCL init before weight download.
    # Use 1× H200 (SM90, 141 GB) which has no init hang and matches the
    # DGX Spark VRAM class. 75 GB FP8 weights fit comfortably.
    _vllm_config('qwen3-coder-next-fp8-new',
        hf_model='Qwen/Qwen3-Coder-Next-FP8',
        image='leosiriusdawn/runpod-vllm:latest',
        gpu_type=H200, gpu_count=1, container_disk_gb=200,
        tool_call_parser='qwen3_coder',
        enforce_eager=True),
    # REAP-139B NVFP4: ~79 GB on disk, baked into the image. Fits 1× RTX
    # PRO 6000 (96 GB VRAM). KV cache budget after weights + CUDA graphs
    # is tight; GMU 0.97 → ~10 GB KV pool, cap context at 80k for
    # 40k+ tool-turn slack on the 39k-token eval prompt.
    # MiniMax M2.5 emits <minimax:tool_call> XML — must use minimax_m2 parser.
    # REAP NVFP4 models have bogus eos_token_id=2 in config.json injected
    # by NVIDIA ModelOpt v0.39.0. The real EOS is 200020 (generation_config).
    # Without the override, vLLM reads eos=2, the model never generates it,
    # and generation runs to max_model_len producing pad tokens.
    # NOTE: lukealonso NVFP4 confirmed broken 2026-04-09 — outputs pure \x00
    # pad tokens for any input. Root cause: likely SM120 NVFP4 MoE kernel bug
    # (vLLM #35566) AND/OR missing explicit quantization flag. Added
    # quantization='modelopt_fp4' as a potential fix. If still broken, this
    # config is dead — use reap-139b-awq-new instead.
    _vllm_config('reap-139b-nvfp4-new',
        hf_model='lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4',
        image='leosiriusdawn/runpod-vllm:latest',
        gpu_type=RTX_PRO_6000, gpu_count=1, container_disk_gb=180,
        max_model_len=80000, gpu_memory_util=0.97,
        quantization='modelopt_fp4',
        tool_call_parser='minimax_m2',
        override_generation_config='{"eos_token_id": 200020}'),
    # REAP-139B AWQ 4-bit: ~23 GB on disk, ~35-40 GB VRAM. Comfortably fits
    # 1× RTX PRO 6000 (96 GB) with 50+ GB KV cache headroom. AWQ W4A16
    # kernels work on SM120 without the NVFP4 CUDA crash bug (vLLM #35566).
    # Primary alternative for GB10/DGX Spark target after lukealonso NVFP4
    # was confirmed broken. Quality unknown — sparse docs, no published evals.
    # AWQ 4-bit: ~78 GB VRAM after load. Doesn't fit 80k context on 96 GB
    # RTX PRO 6000 even at max_num_seqs=5. Use 1× H200 (141 GB) to match
    # the DGX Spark deployment target (128 GB). Gives full 80k context with
    # concurrency=5 and room to spare — faithful to production scenario.
    _vllm_config('reap-139b-awq-new',
        hf_model='cyankiwi/MiniMax-M2.5-REAP-139B-A10B-AWQ-4bit',
        image='leosiriusdawn/runpod-vllm:latest',
        gpu_type=H200, gpu_count=1, container_disk_gb=100,
        max_model_len=80000, gpu_memory_util=0.95,
        tool_call_parser='minimax_m2'),
    # REAP-139B AWQ on 2× RTX PRO 6000 via pipeline parallelism (PP=2).
    # SM120 TP=2 hangs at NCCL all-reduce init (observed with both AWQ and
    # qwen3). PP splits by layers instead — each GPU gets ~39 GB of AWQ
    # weights, fitting in 96 GB with room for KV cache. TP stays at 1.
    _vllm_config('reap-139b-awq-pp2-rtxpro',
        hf_model='cyankiwi/MiniMax-M2.5-REAP-139B-A10B-AWQ-4bit',
        image='leosiriusdawn/runpod-vllm:pp-test',
        gpu_type=RTX_PRO_6000, gpu_count=2, container_disk_gb=100,
        max_model_len=80000, gpu_memory_util=0.95,
        tool_call_parser='minimax_m2',
        pp_size=2),
    # REAP-139B NVFP4 GB10 (saricles): 75 GB on disk, compressed-tensors
    # format via llm-compressor. Different quant than lukealonso — quantizes
    # ALL Linear layers including self_attn. Originally built for DGX Spark
    # (GB10, SM121a). On RTX PRO 6000 (SM120), MUST use --moe-backend marlin
    # to avoid the NVFP4 MoE CUDA crash (vLLM #35566). Env vars from saricles
    # model card: VLLM_TEST_FORCE_FP8_MARLIN=1, VLLM_USE_FLASHINFER_MOE_FP4=0.
    # Same eos_token_id bug as lukealonso — needs override.
    # 2× RTX PRO 6000 for generous VRAM headroom (192 GB for 75 GB weights).
    # STATUS (2026-04-09): BLOCKED — marlin MoE backend on SM120 causes infinite
    # GPU kernel hang (100% SM, 0% mem, no progress for 13+ min). The workaround
    # for vLLM #35566 doesn't crash but also doesn't complete. Weight download
    # never starts because workers are stuck in initialization. Needs either:
    # (a) vLLM fix for NVFP4 MoE on SM120, or (b) a DGX Spark (SM121a) pod.
    _vllm_config('reap-139b-nvfp4-gb10-new',
        hf_model='saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10',
        image='leosiriusdawn/runpod-vllm:latest',
        gpu_type=RTX_PRO_6000, gpu_count=2, container_disk_gb=200,
        max_model_len=80000, gpu_memory_util=0.95,
        tool_call_parser='minimax_m2',
        override_generation_config='{"eos_token_id": 200020}',
        extra_env={
            'MOE_BACKEND': 'marlin',
            'VLLM_TEST_FORCE_FP8_MARLIN': '1',
            'VLLM_USE_FLASHINFER_MOE_FP4': '0',
        }),
    # REAP-139B FP8 (cerebras source): 131 GB on disk, mixed BF16+F8_E4M3.
    # Expert weights already stored as FP8 in the checkpoint. Too large for
    # single 96 GB GPU — use 2× RTX PRO 6000 (192 GB aggregate, TP=2).
    # This is the canonical source model; if it works well on 2× GPU, it
    # validates REAP-139B quality independent of any quantization issues.
    _vllm_config('reap-139b-fp8-new',
        hf_model='cerebras/MiniMax-M2.5-REAP-139B-A10B',
        image='leosiriusdawn/runpod-vllm:latest',
        gpu_type=RTX_PRO_6000, gpu_count=2, container_disk_gb=350,
        max_model_len=80000, gpu_memory_util=0.95,
        quantization='fp8',
        tool_call_parser='minimax_m2'),
    # REAP-172B NVFP4 GB10 (saricles compressed-tensors): ~93 GB on disk.
    # NVFP4 MoE kernels are broken on SM120 (RTX PRO 6000 Blackwell) —
    # both native and marlin backends hang. Use 1× H200 (SM90, 141 GB)
    # where NVFP4 MoE works. Faithful to DGX Spark (128 GB) deployment.
    # Same eos_token_id bug as reap-139b — needs override.
    _vllm_config('reap-172b-nvfp4-gb10-new',
        hf_model='saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10',
        image='leosiriusdawn/runpod-vllm:latest',
        gpu_type=H200, gpu_count=1, container_disk_gb=250,
        max_model_len=80000, gpu_memory_util=0.95,
        tool_call_parser='minimax_m2',
        override_generation_config='{"eos_token_id": 200020}'),
    # MiniMax-M2.5 NVFP4 (nvidia): ~131 GB on disk, baked. 1× H200.
    # KV cache budget after weights + CUDA graphs is tight on a single H200:
    # at default GMU 0.95 + max_model_len 131072, vLLM 0.19 needs 15.5 GB
    # KV pool but only ~6 GB free → crashes at startup with
    #   ValueError: To serve at least one request with the model's max seq
    #   len (131072), 15.5 GiB KV cache is needed, which is larger than the
    #   available KV cache memory (5.98 GiB).
    # Fix: same recipe as reap-139b-nvfp4-new (similar size + same hardware
    # tier) — bump GMU to 0.97 and cap context at 80k. The eval prompt is
    # ~39k tokens so 80k leaves ~40k of slack for tool turns.
    # KV cache was tight at max_model_len=80000 with max_num_seqs=16.
    # Baked image has old entrypoint (default 16), so override explicitly.
    # (:latest configs get the new default of 5 automatically.)
    _vllm_config('minimax-m25-nvfp4-new',
        hf_model='nvidia/MiniMax-M2.5-NVFP4',
        image='leosiriusdawn/runpod-vllm:minimax-m25-nvfp4-baked',
        gpu_type=H200, gpu_count=1, container_disk_gb=240,
        max_model_len=80000, gpu_memory_util=0.97,
        tool_call_parser='minimax_m2',
        extra_env={'MAX_NUM_SEQS': '5'}),
    # MiniMax-M2.5 FP8 (full): ~215 GB on disk, baked. Needs >192 GB VRAM.
    # Prefers 2× H200 (282 GB); falls back to 4× RTX PRO 6000 (384 GB)
    # when H200 fleet is sold out. Also trying B200 (192 GB single GPU)
    # as a middle option.
    _vllm_config('minimax-m25-fp8-new',
        hf_model='MiniMaxAI/MiniMax-M2.5',
        image='leosiriusdawn/runpod-vllm:minimax-m25-fp8-baked',
        gpu_type=RTX_PRO_6000, gpu_count=4, container_disk_gb=320,
        quantization='fp8',
        tool_call_parser='minimax_m2',
        extra_env={'MAX_NUM_SEQS': '5'}),

    # ===== 3090 tier (workstation) — not RunPod, image/gpu_type left None =====
    # leo-workstation RTX 3090 (24 GB). AWQ 4-bit quantization (compressed-
    # tensors) fits in 24 GB. vLLM auto-detects the quantization format.
    # Eval invoked via ``orchestrator eval --vllm-url http://leo-workstation:8000``.
    _vllm_config('qwen3-coder-30b-q4',
        'stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ',
        tool_call_parser='qwen3_coder'),
    _vllm_config('devstral-small-2505-q6', 'mistralai/Devstral-Small-2505',
        tool_call_parser='mistral'),
    # Dropped 2026-04-06 (Task 453): qwen25-coder-32b-q4 removed because its
    # max_position_embeddings cannot accommodate the Claude Code system prompt.
]

EVAL_CONFIGS = [
    # Cloud baselines
    EvalConfig('claude-opus-high', 'claude', 'opus', 'high'),
    EvalConfig('claude-opus-max', 'claude', 'opus', 'max'),
    EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max'),
    EvalConfig('codex-gpt54-xhigh', 'codex', 'gpt-5.4', 'xhigh'),
    EvalConfig('codex-gpt54mini-xhigh', 'codex', 'gpt-5.4-mini', 'xhigh'),
    EvalConfig('gemini-31-pro-high', 'gemini', 'gemini-3.1-pro-preview', 'high'),
    EvalConfig('gemini-3-flash-high', 'gemini', 'gemini-3-flash-preview', 'high'),
    # Self-hosted vLLM backends (Task 453: spread into single canonical list)
    *VLLM_EVAL_CONFIGS,
]


# ===== Final-run configs: DGX Spark scenario (128 GB unified, SM121a) =====
# Tuned for ideal circumstances: full context, generous timeouts, H200 as
# proxy for DGX Spark.  Only configs that have demonstrated task completions
# or are architecturally sound on SM90/SM121a.  Run with:
#   --task-timeout-min 180 --orch-timeout-min 150
FINAL_RUN_CONFIGS = [
    # ===== GPU compatibility (SM120 = RTX PRO 6000 Blackwell) =====
    # ✅ SM120 works: FP8 quant + any TP (confirmed TP=2, TP=4)
    # ❌ SM120 broken: NVFP4 MoE kernels (vLLM #35566), AWQ TP=2 NCCL hang
    # ✅ SM90 (H200) works: everything — FP8, NVFP4, AWQ
    # ✅ SM102 (B200) works: assumed same as SM90 (Blackwell data-center)
    #
    # Launch with: --task-timeout-min 180 --orch-timeout-min 150 --all-tasks
    # Baked images: leosiriusdawn/runpod-vllm:final-<model-slug>

    # --- REAP-139B AWQ 4-bit (~78 GB VRAM) ---
    # ❌ RTX PRO: AWQ TP=2 hangs at NCCL init; TP=1 can't fit 131k context
    # ✅ H200 (141 GB): 56 GB KV = full 131k context
    _vllm_config('final-reap-139b-awq',
        hf_model='cyankiwi/MiniMax-M2.5-REAP-139B-A10B-AWQ-4bit',
        image='leosiriusdawn/runpod-vllm:final-reap-139b-awq',
        gpu_type=H200, gpu_count=1, container_disk_gb=100,
        max_model_len=131072, gpu_memory_util=0.95,
        tool_call_parser='minimax_m2'),

    # --- REAP-139B FP8 (~131 GB VRAM) ---
    # ✅ RTX PRO: 2× RTX PRO (192 GB), TP=2, FP8 works on SM120
    # Canonical source model — validates REAP-139B quality without quant issues.
    _vllm_config('final-reap-139b-fp8',
        hf_model='cerebras/MiniMax-M2.5-REAP-139B-A10B',
        image='leosiriusdawn/runpod-vllm:final-reap-139b-fp8',
        gpu_type=RTX_PRO_6000, gpu_count=2, container_disk_gb=350,
        max_model_len=131072, gpu_memory_util=0.95,
        quantization='fp8',
        tool_call_parser='minimax_m2'),

    # --- REAP-172B NVFP4 GB10 (~93 GB VRAM) ---
    # ❌ RTX PRO: NVFP4 MoE broken on SM120
    # ✅ B200 (180 GB): 80+ GB KV = full 131k context
    # Falls back to 2× H200 (282 GB) if B200 unavailable.
    _vllm_config('final-reap-172b-nvfp4-gb10',
        hf_model='saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10',
        image='leosiriusdawn/runpod-vllm:final-reap-172b-nvfp4-gb10',
        gpu_type=B200, gpu_count=1, container_disk_gb=250,
        max_model_len=131072, gpu_memory_util=0.95,
        tool_call_parser='minimax_m2',
        override_generation_config='{"eos_token_id": 200020}'),

    # --- MiniMax-M2.5 FP8 (~215 GB VRAM) ---
    # ✅ RTX PRO: 4× RTX PRO (384 GB), TP=4 — best self-hosted in trials
    # Solved reify_task_12 (1531 lines). High availability GPU.
    _vllm_config('final-minimax-m25-fp8',
        hf_model='MiniMaxAI/MiniMax-M2.5',
        image='leosiriusdawn/runpod-vllm:final-minimax-m25-fp8',
        gpu_type=RTX_PRO_6000, gpu_count=4, container_disk_gb=600,
        max_model_len=131072,
        quantization='fp8',
        tool_call_parser='minimax_m2',
        extra_env={'MAX_NUM_SEQS': '5'}),

    # --- MiniMax-M2.5 NVFP4 (~131 GB VRAM) ---
    # ❌ RTX PRO: NVFP4 MoE broken on SM120
    # ✅ B200 (180 GB): 40+ GB KV = full 131k context
    _vllm_config('final-minimax-m25-nvfp4',
        hf_model='nvidia/MiniMax-M2.5-NVFP4',
        image='leosiriusdawn/runpod-vllm:final-minimax-m25-nvfp4',
        gpu_type=B200, gpu_count=1, container_disk_gb=240,
        max_model_len=131072, gpu_memory_util=0.95,
        tool_call_parser='minimax_m2',
        extra_env={'MAX_NUM_SEQS': '5'}),

    # --- MiniMax-M2.7 FP8 (~220 GB VRAM, ships as FP8) ---
    # ✅ RTX PRO: 4× RTX PRO (384 GB), TP=4 — same GPU config as M2.5 FP8
    # 229B/10B MoE, 196K max context, released 2026-04-11.
    # Investigation (2026-04-14): the "CUDA graph hang" was misdiagnosed —
    # M2.7 booted fine in 94s on the initial matrix run (07:28 Apr 13).
    # All final-run results were cap starvation, not a model crash.
    # enforce_eager removed; reasoning-parser added per official deploy guide.
    # Baked image rebaked 2026-04-14 with updated entrypoint (REASONING_PARSER
    # + SAFETENSORS_FAST_GPU support).
    _vllm_config('final-minimax-m27-fp8',
        hf_model='MiniMaxAI/MiniMax-M2.7',
        image='leosiriusdawn/runpod-vllm:final-minimax-m27-fp8',
        gpu_type=RTX_PRO_6000, gpu_count=4, container_disk_gb=600,
        max_model_len=131072,
        tool_call_parser='minimax_m2',
        extra_env={
            'MAX_NUM_SEQS': '5',
            'REASONING_PARSER': 'minimax_m2_append_think',
            'SAFETENSORS_FAST_GPU': '1',
        }),

    # --- Qwen3-Coder-Next FP8 (~75 GB VRAM) ---
    # ❌ RTX PRO: TP=2 hangs at NCCL init (model-specific, not just quant)
    # ✅ H200 (141 GB): 56+ GB KV = full 131k context
    _vllm_config('final-qwen3-coder-next-fp8',
        hf_model='Qwen/Qwen3-Coder-Next-FP8',
        image='leosiriusdawn/runpod-vllm:final-qwen3-coder-next-fp8',
        gpu_type=H200, gpu_count=1, container_disk_gb=200,
        max_model_len=131072,
        tool_call_parser='qwen3_coder',
        enforce_eager=True),

    # --- Cloud baselines (re-run with higher timeout) ---
    EvalConfig('final-claude-sonnet-max', 'claude', 'sonnet', 'max'),
    EvalConfig('final-claude-opus-max', 'claude', 'opus', 'max'),
]

ALL_FINAL_CONFIG_NAMES = [c.name for c in FINAL_RUN_CONFIGS]


def get_config_by_name(name: str) -> EvalConfig | None:
    """Look up an eval config by name."""
    for cfg in [*EVAL_CONFIGS, *FINAL_RUN_CONFIGS]:
        if cfg.name == name:
            return cfg
    return None
