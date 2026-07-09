"""Config-integrity test for the FALLBACK full-suite verify command.

Per task 2361: the monorepo FALLBACK full-suite verify path is driven by the
fleet-chain ``test_command`` string in ``orchestrator/config.yaml``, returned
verbatim by ``_build_fallback_config``'s ``__fallback__`` branch whenever
touched files map to no single subproject (e.g. a change confined to the
repo-root ``tests/scripts/`` suite, which is not a workspace member).

Two defects this guards against:

  (1) Coverage gap — the fleet chain ran each of the 5 subprojects' own
      ``tests/`` dirs but never ``tests/scripts/`` itself, so a task that only
      touches ``tests/scripts/`` was gated on unrelated subprojects while its
      own tests never ran in the gating path.

  (2) Host-oversubscription fragility — ``orchestrator`` and ``fused-memory``
      run pytest-xdist with a 60s per-test ``pyproject.toml`` timeout. Under
      host CPU oversubscription an xdist worker can starve past that 60s
      wall-clock ceiling; pytest-timeout's thread method then ``os._exit()``s
      the worker, and ``--max-worker-restart=0`` degrades that into a clean
      per-test "node down: Not properly terminated" failure — tripping on
      whatever test happens to be running at the time, not a real per-test
      defect. Raising the per-test timeout well above 60s (CLI ``--timeout``
      overrides the pyproject default) removes the trigger.

This test loads the *committed* ``orchestrator/config.yaml`` directly (not
through the verify.py fallback-builder code path) — mirrors the pattern in
``test_orchestrator_restart_config_drift.py``.
"""

import pathlib
import re

import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
DF_CONFIG_PATH = REPO_ROOT / "orchestrator" / "config.yaml"


def _fleet_test_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["test_command"]


def _pytest_segments(cmd: str) -> list[str]:
    return [seg for seg in cmd.split("&&") if "pytest" in seg]


def test_fallback_verify_runs_tests_scripts() -> None:
    """The fallback fleet chain must also run the repo-root tests/scripts/ suite.

    Task 2361: tests/scripts/ (e.g. test_spawn_claude.py) is not a workspace
    member and is therefore never covered by any of the 5 subprojects' own
    ``tests/`` dirs. Without this, a task scoped only to tests/scripts/ is
    gated on unrelated subprojects' flakes while its own tests never run in
    the FALLBACK gating path.

    Checked on the pytest-bearing segments (not the raw string) so the
    assertion proves tests/scripts/ is actually invoked by a pytest
    command — not merely mentioned somewhere else in the chain (e.g. inside
    an --ignore flag or an unrelated path fragment).
    """
    cmd = _fleet_test_command()
    segments = _pytest_segments(cmd)
    assert any("tests/scripts" in seg for seg in segments), (
        "orchestrator/config.yaml test_command (FALLBACK full-suite verify) "
        "must have a pytest segment that runs the repo-root tests/scripts/ "
        f"suite (task 2361), not merely mention it elsewhere; got: {cmd!r}"
    )


def test_fanout_includes_sampler_member() -> None:
    """The fallback fleet chain must also run sampler's own tests/ suite.

    Task 2368: ``sampler`` is a workspace member (root pyproject.toml
    ``[tool.uv.workspace].members``) with its own ``pyproject.toml`` and
    ``tests/``, but was never added to the fleet fanout — so a task touching
    sampler alongside a root-level file (which disqualifies the
    subproject-scoped fallback and falls through to this fleet chain) never
    exercised sampler's own tests in the gating path. The segment must be a
    HARD ``cd ../sampler`` (sampler is present on main, unlike cockpit —
    see :func:`test_fanout_includes_cockpit_presence_guarded`) and carry a
    pytest segment with ``--timeout>60`` like every other segment in this
    chain.
    """
    cmd = _fleet_test_command()
    assert re.search(r"cd \.\./sampler\b", cmd), (
        "orchestrator/config.yaml test_command (FALLBACK full-suite verify) "
        "must fan out to sampler/ via 'cd ../sampler' (task 2368) so "
        f"sampler's own tests run; got: {cmd!r}"
    )
    # Matched directly against the raw command (not _pytest_segments' crude
    # &&-split) because the identifying token 'sampler' lives in the 'cd'
    # clause, not the pytest clause that follows it (mirrors the existing
    # per-subproject segments, e.g. 'cd shared && uv run pytest tests/ …').
    match = re.search(
        r"cd \.\./sampler\s*&&\s*uv run pytest tests/\s*--timeout[=\s](\d+)", cmd,
    )
    assert match, (
        "orchestrator/config.yaml test_command (FALLBACK full-suite verify) "
        "must run a pytest segment for sampler's own tests/ suite immediately "
        f"after 'cd ../sampler' (task 2368); got: {cmd!r}"
    )
    timeout_value = int(match.group(1))
    assert timeout_value > 60, (
        f"sampler's pytest segment sets --timeout={timeout_value}, which does "
        "not raise the ceiling above the flaky 60s pyproject default (task 2368)"
    )


def test_fanout_includes_cockpit_presence_guarded() -> None:
    """The fallback fleet chain must run cockpit's tests, guarded by a presence check.

    Task 2368: ``cockpit`` is introduced by the un-merged Fleet Cockpit
    batch (2291-2303) and is absent from main. A HARD ``cd ../cockpit``
    segment would return non-zero and abort this entire ``&&`` chain —
    breaking BOTH the fleet fallback AND merge-queue verify for every task
    until cockpit lands. The segment must instead be guarded by a
    ``[ -d cockpit ]`` presence check that skips cleanly when the directory
    is absent, while still running (and propagating failures from) cockpit's
    own tests/ suite once it exists — carrying a pytest segment with
    ``--timeout>60`` like every other segment in this chain.
    """
    cmd = _fleet_test_command()
    assert "cockpit" in cmd, (
        "orchestrator/config.yaml test_command (FALLBACK full-suite verify) "
        f"must reference cockpit (task 2368); got: {cmd!r}"
    )
    assert re.search(r"\[\s*-d\s+cockpit\s*\]", cmd), (
        "orchestrator/config.yaml test_command (FALLBACK full-suite verify) "
        "references cockpit but is not presence-guarded with a '[ -d cockpit ]' "
        "test (task 2368) — a hard 'cd ../cockpit' would abort the whole "
        f"&&-chain while cockpit is absent from main; got: {cmd!r}"
    )
    # Matched directly against the raw command (not _pytest_segments' crude
    # &&-split, which would split this guarded subshell's own internal &&
    # into a non-pytest guard piece and a pytest piece that no longer
    # contains the word 'cockpit') so the presence guard and its pytest
    # invocation are verified as one associated unit.
    match = re.search(
        r"\[\s*-d\s+cockpit\s*\].*?cd\s+cockpit\s*&&\s*uv run pytest tests/\s*"
        r"--timeout[=\s](\d+)",
        cmd,
    )
    assert match, (
        "orchestrator/config.yaml test_command (FALLBACK full-suite verify) "
        "must run a pytest segment for cockpit's own tests/ suite guarded by "
        f"'[ -d cockpit ]' (task 2368); got: {cmd!r}"
    )
    timeout_value = int(match.group(1))
    assert timeout_value > 60, (
        f"cockpit's pytest segment sets --timeout={timeout_value}, which does "
        "not raise the ceiling above the flaky 60s pyproject default (task 2368)"
    )


def test_fallback_verify_raises_per_test_timeout() -> None:
    """Every pytest segment in the fallback chain must raise --timeout above 60s.

    Task 2361: orchestrator/fused-memory run pytest-xdist with a 60s
    pyproject.toml per-test timeout; under host-oversubscription an xdist
    worker starving past that ceiling gets os._exit()'d by pytest-timeout's
    thread method, and --max-worker-restart=0 turns that into a false-failing
    "node down" for whatever test happens to be running. A CLI --timeout
    (which overrides the pyproject default) above 60s removes the trigger.
    """
    cmd = _fleet_test_command()
    segments = _pytest_segments(cmd)
    assert segments, (
        "orchestrator/config.yaml test_command (FALLBACK full-suite verify) "
        f"has no pytest segments to check (task 2361); got: {cmd!r}"
    )
    for seg in segments:
        match = re.search(r"--timeout[=\s](\d+)", seg)
        assert match, (
            f"pytest segment {seg!r} in orchestrator/config.yaml test_command "
            "(FALLBACK full-suite verify) has no --timeout override (task "
            "2361) — the flaky 60s pyproject default is left in place, so "
            "xdist worker starvation under host oversubscription can still "
            "manufacture a false 'node down' failure"
        )
        timeout_value = int(match.group(1))
        assert timeout_value > 60, (
            f"pytest segment {seg!r} in orchestrator/config.yaml test_command "
            f"sets --timeout={timeout_value}, which does not raise the ceiling "
            "above the flaky 60s pyproject default (task 2361)"
        )
