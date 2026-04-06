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


def _vllm_config(name: str, hf_model: str, effort: str = 'high') -> EvalConfig:
    """Build an EvalConfig that routes through a vLLM-compatible endpoint."""
    return EvalConfig(
        name=name, backend='claude', model='sonnet', effort=effort,
        env_overrides={
            'ANTHROPIC_API_KEY': 'dummy',
            'ANTHROPIC_DEFAULT_SONNET_MODEL': hf_model,
            # ANTHROPIC_BASE_URL injected at runtime via --vllm-url
        },
    )


VLLM_EVAL_CONFIGS = [
    # Spark tier (RunPod H100s)
    _vllm_config('minimax-m25-fp8',       'MiniMaxAI/MiniMax-M2.5'),
    _vllm_config('qwen3-coder-next-fp8-new',  'Qwen/Qwen3-Coder-Next'),  # renamed Task 515: post-fix variant
    _vllm_config('reap-139b-nvfp4',       'cerebras/MiniMax-M2.5-REAP-139B-A10B'),
    _vllm_config('reap-172b-nvfp4',       'cerebras/MiniMax-M2.5-REAP-172B-A10B'),
    # 3090 tier (workstation)
    _vllm_config('qwen3-coder-30b-q4',    'Qwen/Qwen3-Coder-30B-A3B-Instruct'),
    _vllm_config('devstral-small-2505-q6', 'mistralai/Devstral-Small-2505'),
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
    # Self-hosted vLLM backends
    *VLLM_EVAL_CONFIGS,
]


def get_config_by_name(name: str) -> EvalConfig | None:
    """Look up an eval config by name."""
    for cfg in EVAL_CONFIGS:
        if cfg.name == name:
            return cfg
    return None
