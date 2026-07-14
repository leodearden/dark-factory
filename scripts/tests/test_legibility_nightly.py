"""Tests for scripts/legibility/nightly.py — the nightly trickle pipeline
assembly (PRD task epsilon: plans/confusion-reduction-prd.md §5.5, decisions
7/8, boundary test §8.8).

Every stage is exercised behind a dependency-injection seam (invoke for the
LLM, status_fetcher for the census, poster for escalation, committer for
git) plus pytest-monkeypatched module functions — no test here ever touches
a real LLM, the machine-operated main git checkout, a live escalation
server, or real systemd. The one true end-to-end test
(test_run_nightly_happy_path_end_to_end) uses a real temp `git init` repo so
the commit path is genuinely exercised without risking main.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from legibility import nightly


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _write_config(
    root: Path, *, project_id: str, escalation_port: int = 8199, cwd_prefixes=None,
) -> Path:
    """Write a minimal valid docs/legibility/legibility.yaml under *root*."""
    cwd_prefixes = cwd_prefixes if cwd_prefixes is not None else [str(root / "work")]
    legibility_dir = root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    config_path = legibility_dir / "legibility.yaml"
    lines = [
        f"project_id: {project_id}",
        f"project_root: {root}",
        f"escalation_port: {escalation_port}",
        "cwd_prefixes:",
    ]
    lines += [f"  - {prefix}" for prefix in cwd_prefixes]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# step-1/2: resolve_config_path
# ---------------------------------------------------------------------------

def test_resolve_config_path_matches_by_project_id_field(tmp_path):
    config_a = _write_config(tmp_path / "proj-a", project_id="proj_a")
    _write_config(tmp_path / "proj-b", project_id="proj_b")

    resolved = nightly.resolve_config_path("proj_a", search_roots=[tmp_path])

    assert resolved == config_a


def test_resolve_config_path_unknown_id_raises(tmp_path):
    _write_config(tmp_path / "proj-a", project_id="proj_a")

    with pytest.raises(FileNotFoundError):
        nightly.resolve_config_path("does-not-exist", search_roots=[tmp_path])


def test_resolve_config_path_env_override(tmp_path, monkeypatch):
    config_a = _write_config(tmp_path / "proj-a", project_id="proj_a")
    _write_config(tmp_path / "proj-b", project_id="proj_b")
    monkeypatch.setenv("LEGIBILITY_SEARCH_ROOTS", str(tmp_path))

    resolved = nightly.resolve_config_path("proj_a")

    assert resolved == config_a
