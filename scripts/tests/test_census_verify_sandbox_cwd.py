"""Tests for scripts/legibility/census.py's CWD wiring — that the `claude`
subprocess every census stage spawns runs inside the CENSUSED project, not
inside whatever directory the operator happened to launch from.

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

Since task 6042 every stage runs through the shared session runner, so these
drive a REAL runner over conftest's fake JSON-mode `claude` and a hermetic
roster, and read the cwd, config dir and tool flags off what the fake CLI
recorded.

This file is deliberately SELF-CONTAINED — cross-test-module imports are
fragile under the repo-wide `--import-mode=importlib` addopts (see
scripts/tests/conftest.py) — and tests only census's OWN wiring.
"""
from __future__ import annotations

from pathlib import Path

import census as mod
import pytest
from legibility import account_pool, census_trigger, session_runner

import config as config_mod

pytestmark = pytest.mark.timeout(120)

_VERDICT = '{"verified": true, "reason": "observed"}'
_REAL_OPEN_POOLED_RUNNER = session_runner.open_pooled_runner


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


def _flag_values(argv, flag):
    """The values following *flag* in *argv*, up to the next ``--`` flag."""
    start = argv.index(flag) + 1
    values = []
    for arg in argv[start:]:
        if arg.startswith("--"):
            break
        values.append(arg)
    return values


def _make_seam_driving_run_census():
    """Fake `run_census(**kwargs) -> CensusOutcome` that drives each stage
    seam main() built exactly once — mining, verify, synthesis, in that order
    — while main's runner is still open, recording every call's kwargs."""
    calls = []

    def fake_run_census(**kwargs):
        calls.append(kwargs)
        kwargs["invoke"]("ping", "haiku")
        kwargs["verify_fn"]([{"title": "x"}], model="sonnet")
        kwargs["synthesize_fn"]([{"title": "x"}], model="fable")
        return mod.CensusOutcome(
            status="done", report_path="plans/confusion-census-2026-08-03.md",
            filed_ticket_ids=[], stop_reason="exhausted",
        )

    fake_run_census.calls = calls
    return fake_run_census


def _poison(name):
    """A seam fake that raises if ever called — proves a path is not taken."""
    def _fn(*args, **kwargs):
        raise AssertionError(f"{name} must never be called on this path")

    return _fn


@pytest.fixture
def census_main(tmp_path, monkeypatch, fake_claude_cli, pool_roster, sentinel_login):
    """A launcher dir and a DIFFERENT target project root, cwd in the
    launcher, main()'s runner bound to a one-account tmp roster behind the
    fake CLI, and every other side effect stubbed. Returns
    ``(launcher, target, fake_claude_cli)``."""
    launcher = tmp_path / "launcher"
    target = tmp_path / "target"
    launcher.mkdir()
    target.mkdir()
    _write_legibility_yaml(_default_config_path(target), project_root=target)

    accounts_file, env_file = pool_roster("max-p")
    fake_claude_cli.plan(default={"result": _VERDICT})
    monkeypatch.setattr(
        session_runner, "open_pooled_runner",
        lambda label, **_kwargs: _REAL_OPEN_POOLED_RUNNER(
            label, accounts_file=accounts_file, env_file=env_file,
        ),
    )
    monkeypatch.setattr(mod, "run_census", _make_seam_driving_run_census())
    # Every test here passes --force, which must never reach the gate.
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    monkeypatch.chdir(launcher)
    return launcher, target, fake_claude_cli


def _verify_calls(fake):
    return [call for call in fake.calls() if "periodic-census verifier" in call["stdin"]]


def test_main_wires_target_project_root_as_verify_cwd(census_main):
    """THE regression (fleet session census-reify-3386101, 2026-08-03
    09:23-09:41): the verify stage's subprocess must be rooted in the
    censused project, not in the directory the census was launched from."""
    launcher, target, fake = census_main

    assert mod.main(["--project-root", str(target), "--force"]) == 0

    [verify_call] = _verify_calls(fake)
    assert Path(verify_call["cwd"]).resolve() == target.resolve()
    assert Path(verify_call["cwd"]).resolve() != launcher.resolve()


def test_main_wires_target_project_root_as_cwd_for_every_stage(census_main):
    """All THREE stages are scoped to the target, not verify alone, so a
    later reader cannot "tidy" two of them back to the launcher's directory."""
    _launcher, target, fake = census_main

    assert mod.main(["--project-root", str(target), "--force"]) == 0

    calls = fake.calls()
    assert len(calls) == 3, [call["stdin"][:40] for call in calls]
    assert [Path(call["cwd"]).resolve() for call in calls] == [target.resolve()] * 3


def test_verify_prompt_and_subprocess_cwd_name_the_same_root(census_main):
    """The verify prompt tells the model to read *project_root* using
    ABSOLUTE paths only. Guard the prompt text and the sandbox scope
    against drifting apart — an absolute path outside the cwd tree is
    exactly what the headless sandbox denies."""
    _launcher, target, fake = census_main

    assert mod.main(["--project-root", str(target), "--force"]) == 0

    [verify_call] = _verify_calls(fake)
    assert str(target.resolve()) in verify_call["stdin"]
    assert Path(verify_call["cwd"]).resolve() == target.resolve()


def test_main_resolves_a_relative_project_root_for_the_verify_cwd(census_main, monkeypatch):
    """`--project-root` is routinely passed relative. An unresolved relative
    root makes the cwd binding vacuous (cwd="." IS the launcher cwd — the
    very bug) and silently falsifies the prompt's own absolute-paths
    contract."""
    _launcher, target, fake = census_main
    monkeypatch.chdir(target)

    assert mod.main(["--project-root", ".", "--force"]) == 0

    [verify_call] = _verify_calls(fake)
    assert Path(verify_call["cwd"]).is_absolute(), verify_call["cwd"]
    assert Path(verify_call["cwd"]).resolve() == target.resolve()


def test_build_stage_invokes_runs_each_stage_in_the_project_under_the_runners_config_dir(
    tmp_path, fake_claude_cli, pool_roster, sentinel_login,
):
    """Unit test of the seam builder itself, driven as every census stage
    drives it: `invoke(prompt, model)`, two positional args, no kwargs.

    Verify may read the tree and nothing else; mining and synthesis are pure
    classifiers with no tools at all. The shared runner's own default is
    bypassPermissions with every tool, which would let a verifier write into
    the tree it is censusing."""
    accounts_file, env_file = pool_roster("max-p")
    fake_claude_cli.plan(default={"result": _VERDICT})
    cfg = config_mod.LegibilityConfig(
        project_id="target_project",
        project_root=str(tmp_path),
        escalation_port=8103,
        cwd_prefixes=[str(tmp_path)],
    )

    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    with session_runner.SessionRunner(gate, label="census-test") as runner:
        mining, verify, synth = mod._build_stage_invokes(
            cfg, project_root=tmp_path, runner=runner,
        )
        mining("p", "haiku")
        verify("p", "sonnet")
        synth("p", "fable")

    mining_call, verify_call, synth_call = fake_claude_cli.calls()
    for call in (mining_call, verify_call, synth_call):
        assert Path(call["cwd"]) == tmp_path
        assert not Path(call["env"]["CLAUDE_CONFIG_DIR"]).is_relative_to(sentinel_login)
        assert call["credentials"] == {
            "claudeAiOauth": {"accessToken": pool_roster.token("max-p")},
        }
    assert len({call["env"]["CLAUDE_CONFIG_DIR"] for call in fake_claude_cli.calls()}) == 1, (
        "one runner, one config dir, for every stage"
    )
    assert _flag_values(verify_call["argv"], "--allowed-tools") == ["Read", "Grep", "Glob"]
    assert _flag_values(verify_call["argv"], "--permission-mode") == ["dontAsk"]
    for call in (mining_call, synth_call):
        assert _flag_values(call["argv"], "--disallowed-tools") == ["*"]


def test_main_rejects_a_project_root_that_is_not_a_directory(census_main, tmp_path, capsys):
    """A typo'd --project-root must fail LOUDLY at the CLI boundary — exit
    1, naming the flag — never as a deferral.

    An unguarded bad root reaches the spawn as a missing cwd on the FIRST
    invoke (the headroom probe), and preflight_headroom folds any probe
    exception into HeadroomResult(ok=False). Without this guard the run would
    exit 0 with "census deferred: headroom probe invocation failed: ...",
    wearing the exact costume of a usage-limit defer, on every subsequent
    invocation.
    """
    _launcher, target, fake = census_main
    # A real config elsewhere, so config load cannot be what catches this.
    elsewhere = _write_legibility_yaml(tmp_path / "elsewhere" / "legibility.yaml")
    missing = target / "typo-not-a-real-root"

    exit_code = mod.main(
        ["--project-root", str(missing), "--config", str(elsewhere), "--force"]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "--project-root" in err
    assert str(missing) in err
    assert fake.calls() == []
