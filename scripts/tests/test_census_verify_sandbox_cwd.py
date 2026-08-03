"""Tests for scripts/legibility/census.py's CWD wiring — that the headless
`claude -p` subprocess every census stage spawns runs inside the CENSUSED
project, not inside whatever directory the operator happened to launch from.

Regression origin: fleet session census-reify-3386101, 2026-08-03
09:23-09:41. A census of /home/leo/src/reify was launched from a
/home/leo/src/dark-factory cwd. `claude -p` sandboxes its tool access to
the cwd tree, so every verifier Read/Bash against /home/leo/src/reify was
permission-denied — and non-interactively there is no prompt to approve.
Because `_build_default_verify_fn` fails CLOSED per cluster (census.py, "an
unverifiable claim rejects, never crashes"), that surfaced not as a crash
but as a SILENT mass rejection of every single cluster. The fail-closed
default is right; what made it dishonest was the subprocess being rooted in
the wrong tree.

This file is deliberately SELF-CONTAINED — cross-test-module imports are
fragile under the repo-wide `--import-mode=importlib` addopts (see
scripts/tests/conftest.py) — and tests only census's OWN wiring at the
main() level. The `_invoke_cli(cwd=...)` primitive itself is unit-tested
beside its three siblings in test_legibility_coder.py, where every other
test of that function lives.
"""
from __future__ import annotations

from pathlib import Path

import census as mod
import coder
import config as config_mod
from legibility import census_trigger


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


def _default_config_path(project_root):
    return Path(project_root) / "docs" / "legibility" / "legibility.yaml"


def _make_fake_main_run_census():
    """Fake `run_census(**kwargs) -> CensusOutcome` seam for main()-level
    tests — records every call's kwargs in `.calls` so a test can drive the
    REAL stage seams main() built and observe what they thread through."""
    calls = []

    def fake_run_census(**kwargs):
        calls.append(kwargs)
        return mod.CensusOutcome(
            status="done", report_path="plans/confusion-census-2026-08-03.md",
            filed_task_ids=[], stop_reason="exhausted",
        )

    fake_run_census.calls = calls
    return fake_run_census


def _poison(name):
    """A seam fake that raises if ever called — proves a path is not taken."""
    def _fn(*args, **kwargs):
        raise AssertionError(f"{name} must never be called on this path")

    return _fn


def _recording_invoke_cli(recorded):
    """A `coder._invoke_cli`-shaped stub recording what each stage threads."""
    def fake_invoke_cli(prompt, model, *, claude_bin=None, timeout=None, cwd=None):
        recorded.append({"prompt": prompt, "model": model, "timeout": timeout, "cwd": cwd})
        return '{"verified": true, "reason": "observed"}'

    return fake_invoke_cli


def _setup_main(tmp_path, monkeypatch):
    """Build a launcher dir and a DIFFERENT target project root, chdir into
    the launcher, and stub out everything main() would really do. Returns
    (target_root, recorded, fake_run_census)."""
    launcher = tmp_path / "launcher"
    target = tmp_path / "target"
    launcher.mkdir()
    target.mkdir()
    _write_legibility_yaml(_default_config_path(target), project_root=target)

    recorded = []
    monkeypatch.setattr(coder, "_invoke_cli", _recording_invoke_cli(recorded))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    # Every test here passes --force, which must never reach the gate.
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    monkeypatch.chdir(launcher)
    return launcher, target, recorded, fake_run_census


def test_main_wires_target_project_root_as_verify_cwd(tmp_path, monkeypatch):
    """THE regression (fleet session census-reify-3386101, 2026-08-03
    09:23-09:41): the verify stage's subprocess must be rooted in the
    censused project, not in the directory the census was launched from."""
    launcher, target, recorded, fake_run_census = _setup_main(tmp_path, monkeypatch)

    exit_code = mod.main(["--project-root", str(target), "--force"])
    assert exit_code == 0

    kwargs = fake_run_census.calls[0]
    kwargs["verify_fn"]([{"title": "x"}], model="sonnet")

    assert Path(recorded[-1]["cwd"]).resolve() == target.resolve()
    assert Path(recorded[-1]["cwd"]).resolve() != launcher.resolve()


def test_main_wires_target_project_root_as_cwd_for_every_stage(tmp_path, monkeypatch):
    """All THREE stages are scoped to the target, not verify alone.

    Mining and synthesis are text-in/text-out and take no tool action, so
    their cwd is unobservable today — which is exactly why an asymmetry
    here would be invisible until it wasn't. Pinning all three keeps "the
    census subprocess runs inside the censused project" a uniform
    invariant, so a later reader cannot "tidy" two of them back to the
    launcher's directory."""
    launcher, target, recorded, fake_run_census = _setup_main(tmp_path, monkeypatch)

    assert mod.main(["--project-root", str(target), "--force"]) == 0
    kwargs = fake_run_census.calls[0]

    kwargs["invoke"]("ping", "haiku")
    kwargs["verify_fn"]([{"title": "x"}], model="sonnet")
    kwargs["synthesize_fn"]([{"title": "x"}], model="fable")

    assert len(recorded) == 3, recorded
    assert [Path(r["cwd"]).resolve() for r in recorded] == [target.resolve()] * 3


def test_verify_prompt_and_subprocess_cwd_name_the_same_root(tmp_path, monkeypatch):
    """The verify prompt tells the model to read *project_root* using
    ABSOLUTE paths only. Guard the prompt text and the sandbox scope
    against drifting apart — an absolute path outside the cwd tree is
    exactly what the headless sandbox denies."""
    launcher, target, recorded, fake_run_census = _setup_main(tmp_path, monkeypatch)

    assert mod.main(["--project-root", str(target), "--force"]) == 0
    fake_run_census.calls[0]["verify_fn"]([{"title": "x"}], model="sonnet")

    assert str(target) in recorded[-1]["prompt"]
    assert Path(recorded[-1]["cwd"]).resolve() == target.resolve()


def test_main_resolves_a_relative_project_root_for_the_verify_cwd(tmp_path, monkeypatch):
    """`--project-root` defaults to "." and is routinely passed relative.
    An unresolved relative root makes the cwd binding vacuous (cwd="." IS
    the launcher cwd — the very bug) and silently falsifies the prompt's
    own absolute-paths contract."""
    launcher, target, recorded, fake_run_census = _setup_main(tmp_path, monkeypatch)
    monkeypatch.chdir(target)

    assert mod.main(["--project-root", ".", "--force"]) == 0
    fake_run_census.calls[0]["verify_fn"]([{"title": "x"}], model="sonnet")

    recorded_cwd = recorded[-1]["cwd"]
    assert Path(recorded_cwd).is_absolute(), recorded_cwd
    assert Path(recorded_cwd).resolve() == target.resolve()


def test_build_stage_invokes_binds_project_root_as_cwd(tmp_path, monkeypatch):
    """Unit test of the seam builder itself: each returned partial threads
    cwd when driven as `invoke(prompt, model)` — two positional args, no
    kwargs, which is how every census stage calls its invoke."""
    recorded = []
    monkeypatch.setattr(coder, "_invoke_cli", _recording_invoke_cli(recorded))

    cfg = config_mod.LegibilityConfig(
        project_id="target_project",
        project_root=str(tmp_path),
        escalation_port=8103,
        cwd_prefixes=[str(tmp_path)],
    )
    mining, verify, synth = mod._build_stage_invokes(cfg, project_root=tmp_path)

    mining("p", "haiku")
    verify("p", "sonnet")
    synth("p", "fable")

    assert [r["cwd"] for r in recorded] == [str(tmp_path)] * 3
