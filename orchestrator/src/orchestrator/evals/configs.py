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
    return EvalConfig(
        name=name, backend='claude', model='sonnet', effort=effort,
        env_overrides=env,
        image=image, gpu_type=gpu_type, gpu_count=gpu_count,
        container_disk_gb=container_disk_gb,
    )


RTX_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
H200 = "NVIDIA H200"


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
    _vllm_config('qwen3-coder-next-fp8-new',
        hf_model='Qwen/Qwen3-Coder-Next-FP8',
        image='leosiriusdawn/runpod-vllm:qwen3-coder-next-fp8-baked',
        gpu_type=RTX_PRO_6000, gpu_count=2, container_disk_gb=180,
        tool_call_parser='qwen3_coder',
        enforce_eager=True),
    # REAP-139B NVFP4: ~79 GB on disk, baked into the image. Fits 1× RTX
    # PRO 6000 (96 GB VRAM). KV cache budget after weights + CUDA graphs
    # is tight; GMU 0.97 → ~10 GB KV pool, cap context at 80k for
    # 40k+ tool-turn slack on the 39k-token eval prompt.
    # MiniMax M2.5 emits <minimax:tool_call> XML — must use minimax_m2 parser.
    _vllm_config('reap-139b-nvfp4-new',
        hf_model='lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4',
        image='leosiriusdawn/runpod-vllm:reap-139b-nvfp4-baked',
        gpu_type=RTX_PRO_6000, gpu_count=1, container_disk_gb=180,
        max_model_len=80000, gpu_memory_util=0.97,
        tool_call_parser='minimax_m2'),
    # REAP-172B NVFP4 GB10: ~93 GB on disk, baked. Prefers 1× H200 (141 GB)
    # but falls back to 2× RTX PRO 6000 (192 GB) when H200 fleet is sold
    # out. TP_SIZE=2 is set automatically by _vllm_config for gpu_count>1.
    _vllm_config('reap-172b-nvfp4-gb10-new',
        hf_model='saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10',
        image='leosiriusdawn/runpod-vllm:reap-172b-nvfp4-gb10-baked',
        gpu_type=RTX_PRO_6000, gpu_count=2, container_disk_gb=200,
        tool_call_parser='minimax_m2'),
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
    _vllm_config('minimax-m25-nvfp4-new',
        hf_model='nvidia/MiniMax-M2.5-NVFP4',
        image='leosiriusdawn/runpod-vllm:minimax-m25-nvfp4-baked',
        gpu_type=H200, gpu_count=1, container_disk_gb=240,
        max_model_len=80000, gpu_memory_util=0.97,
        tool_call_parser='minimax_m2'),
    # MiniMax-M2.5 FP8 (full): ~215 GB on disk, baked. Needs >192 GB VRAM.
    # Prefers 2× H200 (282 GB); falls back to 4× RTX PRO 6000 (384 GB)
    # when H200 fleet is sold out. Also trying B200 (192 GB single GPU)
    # as a middle option.
    _vllm_config('minimax-m25-fp8-new',
        hf_model='MiniMaxAI/MiniMax-M2.5',
        image='leosiriusdawn/runpod-vllm:minimax-m25-fp8-baked',
        gpu_type=RTX_PRO_6000, gpu_count=4, container_disk_gb=320,
        quantization='fp8',
        tool_call_parser='minimax_m2'),

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


def get_config_by_name(name: str) -> EvalConfig | None:
    """Look up an eval config by name."""
    for cfg in EVAL_CONFIGS:
        if cfg.name == name:
            return cfg
    return None
