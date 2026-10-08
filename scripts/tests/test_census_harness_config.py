"""Tests for scripts/legibility/census.py's --harness-config wiring (task 5931):
main() hands run_census the HARNESS project, the one whose task tree receives
a cluster whose fix surface is the harness's (filing_policy.resolve_target).

The harness identity comes from a legibility.yaml, by default the census
checkout's own docs/legibility/legibility.yaml, and an unloadable one fails
loud before any spend rather than quietly running unrouted.

This file is deliberately SELF-CONTAINED — cross-test-module imports are
fragile under the repo-wide `--import-mode=importlib` addopts (see
scripts/tests/conftest.py) — mirroring test_census_verify_sandbox_cwd.py.
"""
from __future__ import annotations

import subprocess

import census as mod
import filing_policy
from legibility import census_trigger, session_runner

import config as config_mod


def _write_legibility_yaml(config_path, *, project_id="target_project", project_root=None):
    """Write a minimal valid legibility.yaml to *config_path*. Plain-text
    lines, not a yaml.safe_dump round trip — kept independent of the module
    under test's own YAML writer."""
    project_root = project_root if project_root is not None else config_path.parent
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"project_id: {project_id}\n"
        f"project_root: {project_root}\n"
        "escalation_port: 8103\n"
        "cwd_prefixes:\n"
        f"  - {project_root}\n",
        encoding="utf-8",
    )
    return config_path


def _make_fake_main_run_census():
    """Fake `run_census(**kwargs) -> CensusOutcome` seam recording every
    call's kwargs in `.calls`."""
    calls = []

    def fake_run_census(**kwargs):
        calls.append(kwargs)
        return mod.CensusOutcome(
            status="done", report_path="plans/confusion-census-2026-09-27.md",
            filed_ticket_ids=[], stop_reason="exhausted",
        )

    fake_run_census.calls = calls
    return fake_run_census


def _commit_all(repo):
    """Make *repo* a git repo with one commit: main() resolves as_of_sha
    from the censused project's HEAD."""
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test"),
        ("add", "-A"),
        ("commit", "-q", "-m", "fixture"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _poison(name):
    """A seam fake that raises if ever called — proves a path is not taken."""
    def _fn(*args, **kwargs):
        raise AssertionError(f"{name} must never be called on this path")

    return _fn


class _InertRunner:
    """The pooled session runner main() opens, for a run whose stages are
    never invoked (run_census is faked)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def invoker(self, stage):
        return _poison(f"the {stage.name} invoke")


def _setup_main(tmp_path, monkeypatch):
    """A target git repo with its own legibility.yaml, and every side effect
    main() would really perform stubbed out. Returns (target, fake_run_census)."""
    target = tmp_path / "target"
    target.mkdir()
    _write_legibility_yaml(
        target / "docs" / "legibility" / "legibility.yaml", project_root=target,
    )
    _commit_all(target)
    monkeypatch.setattr(
        session_runner, "open_pooled_runner", lambda *_args, **_kwargs: _InertRunner(),
    )
    # Every test here passes --force, which must never reach the gate.
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    return target, fake_run_census


def test_main_passes_the_harness_config_project_to_run_census(tmp_path, monkeypatch):
    target, fake_run_census = _setup_main(tmp_path, monkeypatch)
    harness_root = tmp_path / "harness"
    harness_root.mkdir()
    harness_config = _write_legibility_yaml(
        tmp_path / "harness.yaml", project_id="dark_factory", project_root=harness_root,
    )

    exit_code = mod.main([
        "--project-root", str(target), "--force", "--harness-config", str(harness_config),
    ])

    assert exit_code == 0
    assert fake_run_census.calls[0]["harness_project"] == filing_policy.ProjectRef(
        project_root=str(harness_root), project_id="dark_factory",
    )


def test_main_defaults_the_harness_to_the_census_checkouts_own_config(tmp_path, monkeypatch):
    target, fake_run_census = _setup_main(tmp_path, monkeypatch)

    assert mod.main(["--project-root", str(target), "--force"]) == 0

    harness_cfg = config_mod.load_config(mod.DEFAULT_HARNESS_CONFIG_PATH)
    assert fake_run_census.calls[0]["harness_project"] == filing_policy.ProjectRef(
        project_root=harness_cfg.project_root, project_id=harness_cfg.project_id,
    )


def test_main_fails_loud_on_an_unloadable_harness_config(tmp_path, monkeypatch, capsys):
    target, _ = _setup_main(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "run_census", _poison("run_census"))
    monkeypatch.setattr(
        session_runner, "open_pooled_runner", _poison("session_runner.open_pooled_runner"),
    )
    missing = tmp_path / "no-such-harness.yaml"

    exit_code = mod.main([
        "--project-root", str(target), "--force", "--harness-config", str(missing),
    ])

    assert exit_code == 1
    assert str(missing) in capsys.readouterr().err
