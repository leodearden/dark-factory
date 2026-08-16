"""Behavioural tests for skills/factory-init/scripts/find_escalation_port.py.

WHY THIS FILE LIVES HERE AND NOT UNDER skills/. A ``skills/**/tests/`` directory
or a ``skills/**/test_*.py`` file would trip ``test_skills_has_no_tests_of_its_own``
in ``test_skills_module_config_decision.py`` — the deliberate REVISIT TRIGGER task
3554 installed, whose six pathspecs cover both the nested ``skills/**/...`` and the
direct-child ``skills/...`` spellings (so even a bare ``skills/test_probe.py`` fires
it). ``tests/scripts/`` is collected by TWO registered module configs' test_commands
(``tests/scripts`` and ``scripts``), so a module placed here actually runs inside
``verify.run_full_verification``'s gather rather than sitting uncollected.

WHY THESE TESTS NEED A MEASURED RED. This is characterization work over code that is
already correct, so every assertion below is green at HEAD BY CONSTRUCTION and a green
run proves nothing about whether the guard bites. Each test therefore records the
failure text observed against a NAMED scratch mutation of
``skills/factory-init/scripts/find_escalation_port.py``, reverted before commit —
the convention ``test_skills_module_config_decision.py`` and
``test_root_lint_covers_nonmember_py.py`` already follow in this same directory.
Every MEASURED RED below was observed at base main ``fc6f048b55``.

MUST NOT SKIP. There is no ``pytest.importorskip`` and no try/except-and-skip
anywhere in this module. The script under test is a tracked file in this repo — the
only tracked ``.py`` under ``skills/`` — so a missing script, a broken loader, or a
module that stops importing is a REAL regression and must FAIL loudly rather than
turn this suite green-by-vacuity.

TWO MACHINE DEPENDENCIES, BOTH MEASURED, BOTH PINNED HERE. ``--df-root`` alone does
NOT make this script hermetic:
  1. ``known_project_roots()`` shells out to ``systemctl --user show
     fused-memory.service -p Environment`` and merges DASHBOARD_KNOWN_PROJECT_ROOTS.
     MEASURED: running the script with ``--df-root`` pointed at a /tmp tree still
     surveyed the real machine's roots (know-live, solar-challenge,
     solar-challenge-platform, pump-web-ui, ...).
  2. ``is_bound()`` binds a REAL 127.0.0.1 socket. MEASURED with systemctl
     neutralised: ``--base 8100`` over a tmp tree returned 8104 rather than 8102, and
     ``--base 8002`` returned 8004 rather than 8003, because live escalation servers
     hold those ports on this host.
So the ``no_systemd`` fixture and the ``is_bound`` rebinding are load-bearing for
hermeticity, not conveniences. The repo-root ``conftest.py``'s session isolation
fixtures (``_df_fleet_dir_redirect``, ``_df_no_leaked_drain_processes``,
``_df_no_synthetic_heartbeats_in_live_fleet``; tasks 3798/3799) govern drain-script
processes, ORCH_FLEET_DIR and fleet heartbeat files — none of them constrains socket
binding or subprocess patching, so no interaction with this module is expected.
"""

import importlib.util
import pathlib
import subprocess
import types

import pytest  # pyright: ignore[reportMissingImports]

REPO_ROOT = pathlib.Path(__file__).parents[2]
SCRIPT_PATH = REPO_ROOT / "skills" / "factory-init" / "scripts" / "find_escalation_port.py"


def _load_find_escalation_port() -> types.ModuleType:
    """Load the script by file path.

    It is not on any package path and its directory has no ``__init__.py``, so a
    plain ``import`` cannot reach it. Mirrors ``_load_watchdog`` in
    ``test_orchestrator_watchdog.py``.
    """
    assert SCRIPT_PATH.is_file(), f"script under test is missing: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("find_escalation_port", SCRIPT_PATH)
    assert spec is not None, f"could not build spec from {SCRIPT_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fep() -> types.ModuleType:
    """A FRESHLY loaded module per test.

    Deliberately not module-scoped: several tests below rebind ``fep.is_bound``
    (main() resolves it as a module global at call time), and a shared instance
    would leak that rebinding into siblings.
    """
    return _load_find_escalation_port()


@pytest.fixture
def no_systemd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``known_project_roots``' systemctl branch.

    MEASURED: without this, running the script with ``--df-root`` pointed at a /tmp
    tree still surveyed the real machine's roots (know-live, solar-challenge, ...),
    because the live unit's DASHBOARD_KNOWN_PROJECT_ROOTS is merged in ahead of the
    sibling glob. Patching the GLOBAL ``subprocess.run`` is the precedent
    ``test_orchestrator_watchdog.py`` already sets; monkeypatch restores it.

    FileNotFoundError is one of the two failure modes the script's bare
    ``except Exception`` swallows (the binary missing); step-3 covers the other
    (``subprocess.TimeoutExpired``, the hung-systemd case the ``timeout=10`` implies).
    """

    def _no_systemctl(*args, **kwargs):
        raise FileNotFoundError("systemctl neutralised for hermetic test")

    monkeypatch.setattr(subprocess, "run", _no_systemctl)


# Verbatim `systemctl --user show fused-memory.service -p Environment` output,
# captured from the live unit on this host at base main fc6f048b55. ONE physical
# line, terminated by a single newline (confirmed with `cat -A`: a lone `$` at the
# end, no trailing spaces). Pinned here in a comment for the same reason
# test_orchestrator_watchdog.py pins its _SS_HEADER iproute2 rows verbatim — the
# helper below must reproduce the shape the REAL command emits, not an invented one,
# or known_project_roots' regex would be exercised against a fiction:
#
# Environment=CONFIG_PATH=/home/leo/src/dark-factory/fused-memory/config/config.yaml PROJECT_ROOT=/home/leo/src/dark-factory TASKMASTER_DIR=/home/leo/src/dark-factory/taskmaster-ai MEM0_TELEMETRY=false DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory,/home/leo/src/reify,/home/leo/src/autopilot-video,/home/leo/src/autotrade,/home/leo/src/know-live,/home/leo/src/solar-challenge,/home/leo/mission-control,/home/leo/src/solar-challenge-platform,/home/leo/src/pump-web-ui "FUSED_MEMORY_PREDONE_HOOK_REIFY=/home/leo/.cargo/bin/reify-audit --task {id} --pre-done"
#
# THREE STRUCTURAL FACTS THAT MAKE THE REGEX r"DASHBOARD_KNOWN_PROJECT_ROOTS=([^\s]+)"
# MATCH FOR THE RIGHT REASON, all reproduced by the helper below:
#   1. a single `Environment=` prefix, then space-separated KEY=value tokens;
#   2. DASHBOARD_KNOWN_PROJECT_ROOTS sits in the MIDDLE with further tokens after it,
#      so `([^\s]+)` genuinely has to stop at whitespace rather than run to EOL;
#   3. a value containing spaces is DOUBLE-QUOTED by systemd — so a quoted token
#      follows the roots list, which is the realistic thing the greedy-until-space
#      capture must not swallow.


def _systemctl_env_stdout(roots) -> subprocess.CompletedProcess:
    """A CompletedProcess whose stdout mirrors the real unit's Environment line."""
    joined = ",".join(str(r) for r in roots)
    stdout = (
        "Environment=CONFIG_PATH=/opt/fake/fused-memory/config/config.yaml "
        "PROJECT_ROOT=/opt/fake/dark-factory "
        "TASKMASTER_DIR=/opt/fake/dark-factory/taskmaster-ai "
        "MEM0_TELEMETRY=false "
        f"DASHBOARD_KNOWN_PROJECT_ROOTS={joined} "
        '"FUSED_MEMORY_PREDONE_HOOK_REIFY=/opt/fake/reify-audit --task {id} --pre-done"\n'
    )
    return subprocess.CompletedProcess(
        args=["systemctl", "--user", "show", "fused-memory.service", "-p", "Environment"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


@pytest.fixture
def fake_systemd(monkeypatch: pytest.MonkeyPatch):
    """Factory: install a fake ``systemctl`` reporting *roots* in the unit env.

    Sibling of ``no_systemd`` — that one makes the branch fail, this one makes it
    succeed. Both patch the GLOBAL ``subprocess.run``; monkeypatch restores it.
    """

    def _install(roots) -> subprocess.CompletedProcess:
        completed = _systemctl_env_stdout(roots)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed)
        return completed

    return _install


def _project_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Build a hermetic parent-of-projects tree; return ``(parent, df_root)``.

    Layout — deliberately falsifiable in both directions:
        <tmp>/dark-factory/        the checkout itself (a project root too)
        <tmp>/proj-a/              a sibling project
        <tmp>/proj-a/nested/       a GRANDCHILD — discovery must not recurse into it
        <tmp>/target/              another sibling (the --exclude-root subject)
        <tmp>/loose.txt            a NON-directory sibling — must be excluded
    """
    parent = tmp_path
    df_root = parent / "dark-factory"
    for d in ("dark-factory", "proj-a", "proj-a/nested", "target"):
        (parent / d).mkdir(parents=True, exist_ok=True)
    (parent / "loose.txt").write_text("not a directory\n")
    return parent, df_root


def test_module_loads_with_its_documented_surface(fep: types.ModuleType) -> None:
    """Non-vacuity: the loader really loaded THIS script, not an empty module.

    Asserts on the KEYS of RESERVED and on callability — never on the reserved
    ports' prose, which is documentation and free to be reworded.
    """
    for name in ("known_project_roots", "find_config", "escalation_port", "is_bound", "main"):
        assert callable(getattr(fep, name, None)), f"{name} missing or not callable"

    assert set(fep.RESERVED) == {8002, 8103}
    assert len(fep.CONFIG_NAMES) > 0
    assert fep.CONFIG_NAMES[0] == "dark-factory-orchestrator.yaml"


# ---------------------------------------------------------------------------
# known_project_roots — sibling-fallback discovery
# ---------------------------------------------------------------------------


def test_known_project_roots_returns_immediate_directory_siblings(
    fep: types.ModuleType, no_systemd: None, tmp_path: pathlib.Path
) -> None:
    """The sibling-glob fallback enumerates the immediate DIRECTORY children of
    ``df_root.parent`` — nothing shallower, nothing deeper, no files.

    RE-MEASURED at base main ``fc6f048b55``: over a parent also containing a
    ``loose.txt`` and a grandchild ``proj-a/nested/``, the result is exactly
    ``['dark-factory', 'proj-a', 'target']``.

    Compared as a SET of resolved paths, never by index: ordering between the
    systemd-env source and this glob is step-3's property, and pinning it here too
    would over-constrain a rule this test does not own.

    MEASURED RED at base ``fc6f048b55``, named scratch mutations of
    find_escalation_port.py, all reverted before commit:

      1. widening the glob (line 70) to ``df_root.parent / '*' / '*'`` — RED::

           E  AssertionError: assert {PosixPath('/...oj-a/nested')} == {...}
           E    Extra items in the left set:
           E    PosixPath('.../test_known_project_roots_retur0/proj-a/nested')
           E    Extra items in the right set:

      2. the non-directory exclusion is guarded by a REDUNDANT PAIR of ``is_dir()``
         checks, and reddens only when BOTH are removed. Recorded as measured, not
         predicted, because the two single-filter mutations were each observed GREEN
         and a docstring claiming otherwise would be false:

           - dropping ONLY ``and r.is_dir()`` from the dedup loop (line 77):
             ``3 passed`` — the glob comprehension's own filter still excludes
             ``loose.txt``. That dedup-loop filter's observable role is the
             SYSTEMD-ENV branch (a stale unit env naming a path that is not a
             directory), which is why step-3 owns its falsification, not this test.
           - dropping ONLY ``if Path(p).is_dir()`` from the glob (line 70):
             ``3 passed`` — the dedup loop's filter still excludes it.
           - dropping BOTH — RED::

               E  AssertionError: assert {PosixPath('/...tur0/target')} == {...}
               E    Extra items in the left set:
               E    PosixPath('.../test_known_project_roots_retur0/loose.txt')
    """
    parent, df_root = _project_tree(tmp_path)

    got = {p.resolve() for p in fep.known_project_roots(df_root)}

    assert got == {
        (parent / "dark-factory").resolve(),
        (parent / "proj-a").resolve(),
        (parent / "target").resolve(),
    }

    # The checkout itself is included: it is a project root too, and its own
    # config is exactly what --exclude-root exists to suppress (step-8).
    assert df_root.resolve() in got
    # A non-directory sibling is excluded (the r.is_dir() filter).
    assert (parent / "loose.txt").resolve() not in got
    # Discovery does not recurse: a grandchild directory is not a project root.
    assert (parent / "proj-a" / "nested").resolve() not in got


def test_known_project_roots_on_empty_parent_returns_empty_list(
    fep: types.ModuleType, no_systemd: None, tmp_path: pathlib.Path
) -> None:
    """An empty parent yields ``[]`` rather than raising.

    The nonexistent ``df_root`` is deliberate: the function only ever touches
    ``df_root.parent``, so a not-yet-created checkout must survey cleanly instead of
    exploding — this is the path a fresh ``factory-init`` run takes.
    """
    empty_parent = tmp_path / "empty-parent"
    empty_parent.mkdir()

    assert fep.known_project_roots(empty_parent / "dark-factory") == []


def test_fake_systemd_helper_round_trips_through_known_project_roots(
    fep: types.ModuleType, fake_systemd, tmp_path: pathlib.Path
) -> None:
    """Verify the helper itself: a root reachable ONLY via the fake unit env is found.

    This asserts nothing about allocation or ordering (steps 3 and 7 own those). It
    exists so that a later red in the systemd-env tests is attributable to the code
    under test rather than to a helper emitting a shape systemd never produces.

    ``elsewhere/`` is created OUTSIDE the sibling parent on purpose: it cannot be
    reached by the glob fallback, so its presence in the result is proof the env
    branch parsed — and proof the regex stopped at whitespace instead of swallowing
    the quoted FUSED_MEMORY_PREDONE_HOOK_REIFY token that follows it.
    """
    parent, df_root = _project_tree(tmp_path)
    elsewhere = tmp_path / "outside" / "elsewhere"
    elsewhere.mkdir(parents=True)

    fake_systemd([elsewhere])
    got = {p.resolve() for p in fep.known_project_roots(df_root)}

    assert elsewhere.resolve() in got
    # The glob fallback still contributes; the env branch adds to it, never replaces.
    assert (parent / "proj-a").resolve() in got
    # Nothing from the trailing quoted token leaked into the parsed root list.
    assert not any("reify-audit" in str(p) for p in got)
