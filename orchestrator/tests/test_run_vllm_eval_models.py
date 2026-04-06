"""Tests for scripts/run_vllm_eval.py MODELS, GPU_TYPES, and HF_MODELS literals.

We cannot import run_vllm_eval.py directly because it unconditionally injects
/home/leo/src/runpod-toolkit into sys.path and imports runpod_toolkit.{config,compute},
which are not installed in the test environment. Instead we parse the file with ast
(stdlib, zero deps) and use ast.literal_eval to extract the literal data structures.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from orchestrator.evals.configs import VLLM_EVAL_CONFIGS, get_config_by_name

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_vllm_eval.py"


# ---------------------------------------------------------------------------
# AST-parsing fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _parsed_models() -> dict:
    """Parse MODELS dict from run_vllm_eval.py without importing it."""
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODELS":
                    return ast.literal_eval(node.value)
    raise ValueError("MODELS assignment not found in run_vllm_eval.py")


@pytest.fixture(scope="session")
def _parsed_gpu_types() -> list:
    """Parse GPU_TYPES list from run_vllm_eval.py without importing it."""
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GPU_TYPES":
                    return ast.literal_eval(node.value)
    raise ValueError("GPU_TYPES assignment not found in run_vllm_eval.py")


@pytest.fixture(scope="session")
def _parsed_hf_models() -> dict:
    """Parse HF_MODELS dict from inside run_eval() in run_vllm_eval.py."""
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_eval":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "HF_MODELS":
                            return ast.literal_eval(child.value)
    raise ValueError("HF_MODELS assignment not found inside run_eval() in run_vllm_eval.py")


# ---------------------------------------------------------------------------
# Step-5 test: config_name field uses renamed config
# ---------------------------------------------------------------------------


class TestQwen3CoderNextModelsConfigName:
    """MODELS['qwen3-coder-next'][1] must reference the renamed post-fix config."""

    def test_uses_renamed_config_name(self, _parsed_models):
        """The script's config_name for qwen3-coder-next must be 'qwen3-coder-next-fp8-new'."""
        entry = _parsed_models["qwen3-coder-next"]
        config_name = entry[1]
        assert config_name == "qwen3-coder-next-fp8-new", (
            f"Expected 'qwen3-coder-next-fp8-new', got {config_name!r}. "
            "Update MODELS['qwen3-coder-next'][1] in scripts/run_vllm_eval.py."
        )


# ---------------------------------------------------------------------------
# Step-7 test: TOOL_CALL_PARSER env var in extra_env
# ---------------------------------------------------------------------------


class TestQwen3CoderNextExtraEnv:
    """MODELS['qwen3-coder-next'][4] extra_env must contain the documented vLLM workarounds."""

    def test_tool_call_parser_is_qwen3_coder(self, _parsed_models):
        """extra_env must set TOOL_CALL_PARSER=qwen3_coder (documented fix, graph edge 3095db10).

        The entrypoint-vllm.sh reads TOOL_CALL_PARSER and forwards it as --tool-call-parser
        to vLLM. 'hermes' (the entrypoint default) is wrong for Qwen3-Coder-Next.
        """
        extra_env = _parsed_models["qwen3-coder-next"][4]
        assert extra_env.get("TOOL_CALL_PARSER") == "qwen3_coder", (
            f"TOOL_CALL_PARSER not set to 'qwen3_coder' in extra_env: {extra_env!r}"
        )

    def test_max_model_len_is_reduced(self, _parsed_models):
        """extra_env must set MAX_MODEL_LEN=65536 to halve the default context window.

        The entrypoint default MAX_MODEL_LEN=131072 leaves too little KV-cache headroom
        on the FP8 path and exacerbates CUDA-graph warm-up hangs. 65536 is sufficient
        for code completion and gives the sampler/CUDA-graph profiler room to breathe.
        """
        extra_env = _parsed_models["qwen3-coder-next"][4]
        assert extra_env.get("MAX_MODEL_LEN") == "65536", (
            f"MAX_MODEL_LEN not set to '65536' in extra_env: {extra_env!r}"
        )

    def test_gpu_memory_util_leaves_headroom(self, _parsed_models):
        """extra_env must set GPU_MEMORY_UTIL=0.90 (back off from 0.95 default).

        A lower GPU_MEMORY_UTIL gives the sampler/CUDA-graph profiler more headroom
        on the FP8 path, reducing KV/CUDA-graph warm-up OOM risk.
        """
        extra_env = _parsed_models["qwen3-coder-next"][4]
        assert extra_env.get("GPU_MEMORY_UTIL") == "0.90", (
            f"GPU_MEMORY_UTIL not set to '0.90' in extra_env: {extra_env!r}"
        )
