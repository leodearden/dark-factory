"""Contract tests for the `embedder` compose service (task 6797, PRD α1).

They parse the committed compose file and the arms manifest; verify has no docker.
"""

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parent.parent.parent
_COMPOSE = _REPO / "fused-memory" / "docker" / "docker-compose.yml"
_ARMS = _REPO / "scripts" / "local-model-serving" / "arms.yaml"


@pytest.fixture(scope="module")
def embedder() -> dict:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]["embedder"]


@pytest.fixture(scope="module")
def granite_arm() -> dict:
    arms = yaml.safe_load(_ARMS.read_text(encoding="utf-8"))["arms"]
    return next(a for a in arms if a["arm_id"] == "granite-embedding-english-r2")


def _flag(embedder: dict, name: str) -> str:
    command = embedder["command"]
    return command[command.index(name) + 1]


def test_image_is_pinned_by_the_digest_the_arm_manifest_carries(embedder, granite_arm) -> None:
    assert embedder["image"] == f"{granite_arm['image']}@{granite_arm['image_digest']}"


def test_serves_the_granite_model_under_the_name_fused_memory_uses(embedder, granite_arm) -> None:
    assert _flag(embedder, "--model") == granite_arm["model_ref"]
    assert _flag(embedder, "--served-model-name") == granite_arm["served_model_name"]
    assert _flag(embedder, "--runner") == "pooling"
    assert _flag(embedder, "--max-model-len") == str(granite_arm["max_model_len"])
    assert _flag(embedder, "--max-num-seqs") == str(granite_arm["max_num_seqs"])


def test_gpu_share_is_the_bounded_literal(embedder) -> None:
    assert _flag(embedder, "--gpu-memory-utilization") == "0.062"


def test_published_on_loopback_8420_only(embedder) -> None:
    assert embedder["ports"] == ["127.0.0.1:8420:8000"]
    assert _flag(embedder, "--port") == "8000"


def test_hugging_face_runs_offline_from_the_host_cache(embedder) -> None:
    env = set(embedder["environment"])
    assert {"HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_HOME=/root/.cache/huggingface"} <= env
    assert "${HOME}/.cache/huggingface:/root/.cache/huggingface" in embedder["volumes"]


def test_reserves_an_nvidia_gpu(embedder) -> None:
    devices = embedder["deploy"]["resources"]["reservations"]["devices"]
    assert [d["driver"] for d in devices] == ["nvidia"]
    assert "gpu" in devices[0]["capabilities"]


def test_restarts_are_bounded(embedder) -> None:
    assert embedder["restart"] == "on-failure:5"
