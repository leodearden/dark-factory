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
