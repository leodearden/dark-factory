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

import contextlib
import importlib.util
import pathlib
import socket
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

    THE STALE-BYTECODE GUARD IS NOT DEFENSIVE PADDING — it was added after this
    module was OBSERVED loading a stale ``__pycache__`` .pyc at base ``fc6f048b55``.
    ``SourceFileLoader`` validates cached bytecode against the source's (mtime, size),
    so a source edit that is SIZE-IDENTICAL and lands inside the same mtime tick —
    exactly what a two-line reordering does, and exactly what this module's scratch
    mutations do — can leave the cache looking valid. Observed concretely: after
    ``git checkout`` restored CONFIG_NAMES, ``test_config_name_precedence_holds_for_every_pair``
    still failed with ``At index 1 diff: 'orchestrator-config.yaml' != 'orchestrator.yaml'``
    while ``git diff`` on the script was EMPTY. Dropping the cache entry makes every
    load source-authoritative, which is what the MEASURED REDs recorded throughout
    this module depend on being true.
    """
    assert SCRIPT_PATH.is_file(), f"script under test is missing: {SCRIPT_PATH}"
    cached = importlib.util.cache_from_source(str(SCRIPT_PATH))
    pathlib.Path(cached).unlink(missing_ok=True)
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


# ---------------------------------------------------------------------------
# known_project_roots — systemd-env branch, ordering, failure swallowing
# ---------------------------------------------------------------------------


def test_env_roots_are_comma_split_and_ordered_before_the_glob(
    fep: types.ModuleType, fake_systemd, tmp_path: pathlib.Path
) -> None:
    """DASHBOARD_KNOWN_PROJECT_ROOTS is comma-split, and its roots come FIRST.

    Ordering is a real contract, not an accident: the live unit env is the
    AUTHORITATIVE source of what projects exist, and the sibling glob is the
    documented fallback for what it misses. main() surveys roots in list order, so
    the env's spelling of a project is the one that reaches the evidence table.

    The two env roots are created OUTSIDE the sibling parent so their presence is
    attributable to the env branch alone — the glob cannot reach them.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    moving the env merge to AFTER the sibling glob (i.e. performing the
    ``roots += [...glob...]`` first) — RED::

        E  AssertionError: env roots must precede glob roots: env at [4, 5], glob at [0, 2, 3]
        E  assert 5 < 0
        E   +  where 5 = max([4, 5])
        E   +  and   0 = min([0, 2, 3])

    That same mutation also reddened
    ``test_a_root_reachable_by_both_sources_appears_exactly_once`` (the retained
    spelling flipped from the env alias to the glob's canonical path) — recorded
    because it is evidence the two tests pin ONE contract from two directions, not
    two independent coincidences.
    """
    parent, df_root = _project_tree(tmp_path)
    env_a = tmp_path / "outside" / "env-a"
    env_b = tmp_path / "outside" / "env-b"
    for d in (env_a, env_b):
        d.mkdir(parents=True)

    fake_systemd([env_a, env_b])
    roots = fep.known_project_roots(df_root)
    resolved = [p.resolve() for p in roots]

    # Comma-split: both listed roots survive as distinct entries.
    assert env_a.resolve() in resolved
    assert env_b.resolve() in resolved

    glob_resolved = {(parent / n).resolve() for n in ("dark-factory", "proj-a", "target")}
    env_at = [i for i, p in enumerate(resolved) if p in {env_a.resolve(), env_b.resolve()}]
    glob_at = [i for i, p in enumerate(resolved) if p in glob_resolved]
    assert glob_at, "sibling glob contributed nothing — fixture is not exercising the merge"
    assert max(env_at) < min(glob_at), (
        f"env roots must precede glob roots: env at {env_at}, glob at {glob_at}"
    )


def test_empty_comma_segments_do_not_become_a_cwd_entry(
    fep: types.ModuleType, fake_systemd, tmp_path: pathlib.Path
) -> None:
    """A trailing or doubled comma is dropped by the ``if p`` filter.

    Without it, ``"".split(",")`` segments become ``Path("")`` — which is
    ``Path('.')``, the CURRENT WORKING DIRECTORY. That is not a harmless no-op: '.'
    is a real directory, so it survives the ``r.is_dir()`` filter, and main() would
    then look for a config in whatever directory the caller happened to run from
    and silently fold its escalation port into the survey.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    dropping the ``if p`` guard from the comma split (line 65) — RED::

        E  AssertionError: an empty comma segment leaked in as Path('.'): [PosixPath('.')]
        E  assert [PosixPath('.')] == []
        E    Left contains one more item: PosixPath('.')
    """
    _parent, df_root = _project_tree(tmp_path)
    env_a = tmp_path / "outside" / "env-a"
    env_a.mkdir(parents=True)

    # "<env_a>,,"  — one doubled comma and one trailing comma.
    fake_systemd([str(env_a), "", ""])
    roots = fep.known_project_roots(df_root)

    cwd_entries = [p for p in roots if p == pathlib.Path(".")]
    assert cwd_entries == [], f"an empty comma segment leaked in as Path('.'): {cwd_entries}"
    assert env_a.resolve() in {p.resolve() for p in roots}


def test_env_roots_that_are_not_existing_directories_are_dropped(
    fep: types.ModuleType, fake_systemd, tmp_path: pathlib.Path
) -> None:
    """A stale unit env cannot inject a phantom project.

    This is the dedup loop's ``r.is_dir()`` filter (line 77) — the one step-2
    measured as unobservable through the glob branch, because the glob comprehension
    filters non-directories of its own. The env branch does NOT, so this is the
    filter's real job: a decommissioned project still listed in the unit env, and a
    listed path that is a FILE rather than a directory, must both drop out rather
    than reach find_config().

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    dropping ``and r.is_dir()`` from the dedup loop (line 77) — RED, and note this is
    the SAME mutation step-2 observed green through the glob branch::

        E  AssertionError: a nonexistent env root became a project root
        E  assert PosixPath('.../outside/decommissioned') not in {PosixPath('.../dark-factory'),
        E    PosixPath('.../outside/still-here'), PosixPath('.../proj-a'), ...}
    """
    parent, df_root = _project_tree(tmp_path)
    phantom = tmp_path / "outside" / "decommissioned"  # never created
    a_file = parent / "loose.txt"  # exists, but is not a directory
    real = tmp_path / "outside" / "still-here"
    real.mkdir(parents=True)

    fake_systemd([phantom, a_file, real])
    resolved = {p.resolve() for p in fep.known_project_roots(df_root)}

    assert phantom.resolve() not in resolved, "a nonexistent env root became a project root"
    assert a_file.resolve() not in resolved, "a non-directory env root became a project root"
    assert real.resolve() in resolved


def test_a_root_reachable_by_both_sources_appears_exactly_once(
    fep: types.ModuleType, fake_systemd, tmp_path: pathlib.Path
) -> None:
    """Dedup keys on ``str(r.resolve())``, and the FIRST-SEEN spelling is retained.

    The realistic collision is not a byte-identical repeat — it is the unit env
    naming a project through a symlink alias while the glob finds its canonical
    sibling path. A duplicate here would be a live bug in main(): the same project's
    claimed port would be surveyed twice, and with --exclude-root only one of the two
    spellings would match, so the excluded repo's own port would still be treated as
    claimed.

    The retained spelling is the ENV's (unresolved) one, because the env branch runs
    first — pinned as observed behaviour, and consistent with the ordering contract
    above.

    MEASURED RED at base ``fc6f048b55``: the same "env merge moved AFTER the glob"
    scratch mutation the ordering test names also reddens this one — RED::

        E  AssertionError: expected the first-seen (env, unresolved) spelling to be retained
        E  assert PosixPath('.../target') == PosixPath('.../outside/target-alias')

    The dedup itself (``len(hits) == 1``) stayed green under that mutation, which is
    the point: dedup keys on the RESOLVED path, so it survives a reordering — only
    the retained spelling moves. Both halves are asserted because only the first
    protects main() from surveying one project twice.
    """
    parent, df_root = _project_tree(tmp_path)
    target = parent / "target"
    alias = tmp_path / "outside" / "target-alias"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(target, target_is_directory=True)

    fake_systemd([alias])
    roots = fep.known_project_roots(df_root)

    hits = [p for p in roots if p.resolve() == target.resolve()]
    assert len(hits) == 1, f"project surveyed twice under two spellings: {hits}"
    assert hits[0] == alias, "expected the first-seen (env, unresolved) spelling to be retained"


@pytest.mark.parametrize(
    "boom",
    [
        pytest.param(FileNotFoundError("no systemctl on PATH"), id="binary-missing"),
        pytest.param(
            subprocess.TimeoutExpired(cmd=["systemctl"], timeout=10), id="hung-systemd"
        ),
    ],
)
def test_systemctl_failure_falls_back_to_the_full_sibling_set(
    fep: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, boom
) -> None:
    """Both failure modes the bare ``except Exception`` swallows leave discovery whole.

    TimeoutExpired is not hypothetical padding: the call passes ``timeout=10``, which
    exists precisely because a hung systemd is a real state, and it raises rather than
    returning a CompletedProcess. If either escaped, ``factory-init`` would crash on a
    box with a sick or absent user manager instead of degrading to the sibling glob.
    """
    parent, df_root = _project_tree(tmp_path)

    def _raise(*args, **kwargs):
        raise boom

    monkeypatch.setattr(subprocess, "run", _raise)

    resolved = {p.resolve() for p in fep.known_project_roots(df_root)}
    assert resolved == {
        (parent / "dark-factory").resolve(),
        (parent / "proj-a").resolve(),
        (parent / "target").resolve(),
    }


# ---------------------------------------------------------------------------
# find_config — candidate-name precedence
# ---------------------------------------------------------------------------


def _write_config(root: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """Write *body* to ``root/name``, creating parents as needed.

    THE single way this module puts a config on disk, so no two tests can drift into
    disagreeing about what a project fixture looks like. Parents are created because
    the nested ``orchestrator/config.yaml`` spelling must work through this same
    helper.
    """
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _touch_config(root: pathlib.Path, name: str) -> pathlib.Path:
    """An EMPTY config at ``root/name``.

    ``find_config`` never reads content — only ``is_file()`` — so the body is
    irrelevant to the precedence tests, and an empty one keeps them honest about
    testing name resolution rather than parsing.
    """
    return _write_config(root, name, "")


def _config_body(escalation_port: int, decoy_port: int = 9999, *, decoy_first: bool = False) -> str:
    """A realistic two-top-level-block config: ``escalation:`` plus a DECOY block.

    The decoy is what makes ``escalation_port``'s block slicing OBSERVABLE rather
    than incidental — a config with only one ``port:`` in it would look identical
    under a correct parser and under one that just greps the whole file.

    Both emission orders are supported because they exercise DIFFERENT branches:
      - decoy AFTER (default): the loop must BREAK on the next column-0 key, so the
        decoy's port is never collected;
      - decoy BEFORE (``decoy_first=True``): collection must not have STARTED yet, so
        the decoy's port is never even considered.
    A parser broken in one direction can still look correct in the other, which is
    why step-5 asserts both.
    """
    escalation_block = f"escalation:\n  host: 127.0.0.1\n  port: {escalation_port}\n"
    decoy_block = f"dashboard:\n  port: {decoy_port}\n"
    return decoy_block + escalation_block if decoy_first else escalation_block + decoy_block


def test_canonical_config_name_wins_over_every_legacy_spelling(
    fep: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """With all four CONFIG_NAMES present at once, the canonical name is returned.

    The module's own comment calls this load-bearing: "a migrated project's real
    config wins over any legacy symlink/spelling still present alongside it". A
    migration that leaves the old file behind is the normal end state, so picking the
    wrong one would read a STALE escalation port and hand the same port to two
    projects.

    RE-MEASURED at base ``fc6f048b55``: canonical wins over a coexisting
    orchestrator.yaml.
    """
    root = tmp_path / "proj-a"
    root.mkdir()
    for name in fep.CONFIG_NAMES:
        _touch_config(root, name)

    assert fep.find_config(root) == root / "dark-factory-orchestrator.yaml"


def test_each_config_name_is_found_when_it_is_the_only_one_present(
    fep: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """Every spelling in CONFIG_NAMES is reachable — including the NESTED one.

    ``orchestrator/config.yaml`` is the entry a naive ``iterdir()``-based
    implementation would miss entirely, which would make such a project invisible to
    the port-collision scan: its claimed port would look free.

    Driven from ``fep.CONFIG_NAMES`` rather than a hand-copied literal, so a name
    added to the production tuple is exercised here without editing this test.
    """
    for i, name in enumerate(fep.CONFIG_NAMES):
        root = tmp_path / f"only-{i}"
        root.mkdir()
        expected = _touch_config(root, name)
        assert fep.find_config(root) == expected, f"{name} unreachable when it is the only config"


def test_config_name_precedence_holds_for_every_pair(
    fep: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """Precedence is checked PAIRWISE in CONFIG_NAMES order, plus an order ratchet.

    TWO ASSERTIONS, AND THE TENSION BETWEEN THEM IS THE POINT — recorded because it
    was MEASURED, not assumed.

    1. MECHANICS, driven FROM ``fep.CONFIG_NAMES``: for every ordered pair (i < j),
       both names are created in their own root and the earlier one must win. This
       tracks the production tuple instead of mirroring it, so a name ADDED to
       CONFIG_NAMES is exercised here with no edit to this test. What it catches is
       an implementation that stops honouring the tuple's order at all — sorting the
       candidates, or switching to an ``iterdir()`` sweep.

    2. ORDER RATCHET, an explicit literal. Assertion 1 CANNOT catch a reordering of
       the tuple itself, because it derives its expectation from that tuple: reorder
       the production names and the expectation reorders with them. MEASURED at base
       ``fc6f048b55``: swapping ONLY the middle two entries ("orchestrator.yaml" and
       "orchestrator-config.yaml"), with the ratchet below temporarily disabled, left
       the whole module GREEN — ``16 passed``. RE-MEASURED after the stale-bytecode
       guard in ``_load_find_escalation_port`` landed, since a size-identical
       reordering is exactly the mutation a .pyc cache can mask; the result was
       unchanged, so this is a property of the assertions and not a loader artifact.
       Since
       ``find_config`` returns the FIRST match, that order is a behavioural
       specification and not a stylistic list, so it gets a literal ratchet. If a
       reorder is ever deliberate, update the literal here and say why: the tuple is
       documented as kept in sync with ``dashboard.config``'s _CANONICAL_CONFIG_NAME
       + _LEGACY_CONFIG_NAMES, and a spelling recognized by one but not the other is
       the latent trap the module's own comment warns about.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    reversing CONFIG_NAMES entirely — RED (this test, on the ratchet)::

        E  AssertionError: assert 'orchestrator/config.yaml' == 'dark-factory...estrator.yaml'
        E    - dark-factory-orchestrator.yaml
        E    + orchestrator/config.yaml

    That same reversal also reddened
    ``test_canonical_config_name_wins_over_every_legacy_spelling`` (2 failed,
    13 passed), which is the canonical-first claim pinned independently.
    """
    names = fep.CONFIG_NAMES
    assert len(names) > 1, "precedence is unfalsifiable with fewer than two candidates"

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            root = tmp_path / f"pair-{i}-{j}"
            root.mkdir()
            winner = _touch_config(root, names[i])
            _touch_config(root, names[j])
            assert fep.find_config(root) == winner, (
                f"{names[i]} must win over {names[j]}"
            )

    assert names == (
        "dark-factory-orchestrator.yaml",
        "orchestrator.yaml",
        "orchestrator-config.yaml",
        "orchestrator/config.yaml",
    )


def test_root_with_no_config_returns_none(fep: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """A directory that is not a dark-factory project is skipped, not guessed at.

    main() relies on this: the sibling glob sweeps up every directory next to the
    checkout, most of which are not projects at all.
    """
    root = tmp_path / "not-a-project"
    root.mkdir()
    (root / "README.md").write_text("# unrelated\n")

    assert fep.find_config(root) is None


def test_a_directory_named_like_a_config_is_not_returned(
    fep: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """The ``is_file()`` guard — a stray DIRECTORY cannot be mistaken for a config.

    ``escalation_port`` would swallow the resulting IsADirectoryError in its bare
    ``except Exception`` and return None, so the project's real claimed port
    (here in the legacy orchestrator.yaml) would silently vanish from the survey and
    its port would be handed to a second project. Falling through to the next
    candidate name is what keeps that from happening.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    swapping ``cand.is_file()`` for ``cand.exists()`` — RED::

        E  AssertionError: a directory was returned as a config file
        E  assert PosixPath('.../proj-a/dark-factory-orchestrator.yaml') ==
        E    PosixPath('.../proj-a/orchestrator.yaml')
    """
    root = tmp_path / "proj-a"
    root.mkdir()
    (root / "dark-factory-orchestrator.yaml").mkdir()  # a DIRECTORY with the canonical name
    legacy = _touch_config(root, "orchestrator.yaml")

    assert fep.find_config(root) == legacy, "a directory was returned as a config file"


def test_config_helpers_round_trip_through_find_config_and_escalation_port(
    fep: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """Verify the fixture builders themselves, before anything depends on them.

    RE-MEASURED at base ``fc6f048b55``: a proj-a config carrying
    ``escalation:\\n  host: 127.0.0.1\\n  port: 8100\\ndashboard:\\n  port: 9999``
    surveys as 8100 — the escalation port, not the decoy.

    This exists so a later red in the parsing or allocation tests is attributable to
    the code under test rather than to a fixture that never contained what it claimed.
    """
    root = tmp_path / "proj-a"
    root.mkdir()
    written = _write_config(root, "dark-factory-orchestrator.yaml", _config_body(8100))

    # The body really is the two-block shape the docstring claims.
    assert "escalation:" in written.read_text()
    assert "dashboard:" in written.read_text()
    assert "9999" in written.read_text()

    cfg = fep.find_config(root)
    assert cfg == written
    assert fep.escalation_port(cfg) == 8100

    # And the decoy-first ordering round-trips to the same answer.
    other = tmp_path / "proj-b"
    other.mkdir()
    _write_config(other, "dark-factory-orchestrator.yaml", _config_body(8101, decoy_first=True))
    assert fep.escalation_port(fep.find_config(other)) == 8101


# ---------------------------------------------------------------------------
# escalation_port — top-level block slicing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decoy_first", [False, True], ids=["decoy-after", "decoy-before"])
def test_only_the_escalation_blocks_port_is_returned(
    fep: types.ModuleType, tmp_path: pathlib.Path, decoy_first: bool
) -> None:
    """A ``port:`` in a DIFFERENT top-level block is never returned — in BOTH orders.

    This is the parse that decides whether a project's claimed port is seen at all,
    and reading the wrong block is worse than reading none: the survey would record a
    port the project does not actually hold, leave the real one looking free, and hand
    it to a second project.

    Both orders are asserted because they pin different code:
      - decoy AFTER pins the ``break`` on the next column-0 key;
      - decoy BEFORE pins that collection only STARTS at the ``escalation:`` line.
    A parser broken in one direction can look correct in the other, and a
    same-valued decoy would hide both — hence 8100 vs 9999.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    searching the whole file instead of the sliced block (``PORT_RE.search(text)``) —
    RED on the decoy-before case, where the decoy is the first ``port:`` in the file::

        E  AssertionError: assert 9999 == 8100

    The decoy-AFTER case stayed green under that same mutation, which is exactly why
    both orders are asserted: whole-file search happens to return the right answer
    when the escalation block comes first. The mutation also reddened
    ``test_config_helpers_round_trip_through_find_config_and_escalation_port``
    (``assert 9999 == 8101``, its decoy-first half) — the fixture verifier catching
    the same defect independently.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = _write_config(
        root, "dark-factory-orchestrator.yaml", _config_body(8100, 9999, decoy_first=decoy_first)
    )

    assert fep.escalation_port(cfg) == 8100


def test_comments_and_blank_lines_do_not_terminate_the_block(
    fep: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """An indented comment, a blank line, and a COLUMN-0 comment all keep the block open.

    The column-0 comment is the interesting one: the break condition is
    ``ln and not ln[0].isspace() and not ln.lstrip().startswith("#")``, so the
    ``#``-carve-out exists precisely so a commented-out key sitting between
    ``escalation:`` and its ``port:`` does not silently truncate the block and hide
    the project's claim. A blank line survives via the leading ``ln and`` guard.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    dropping the ``and not ln.lstrip().startswith("#")`` carve-out — RED::

        E  AssertionError: a column-0 comment truncated the block
        E  assert None == 8100
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = _write_config(
        root,
        "dark-factory-orchestrator.yaml",
        "escalation:\n"
        "  host: 127.0.0.1\n"
        "  # an indented comment\n"
        "\n"
        "# a column-0 comment — NOT a new top-level key\n"
        "  port: 8100\n"
        "dashboard:\n"
        "  port: 9999\n",
    )

    assert fep.escalation_port(cfg) == 8100, "a column-0 comment truncated the block"


def test_first_port_in_the_block_wins(fep: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """With two ``port:`` lines in the block, the FIRST is returned.

    ``PORT_RE.search`` semantics, pinned as OBSERVED behaviour rather than endorsed
    design — a config with two escalation ports is malformed either way, and this
    records which one the survey would believe if one ever appeared.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = _write_config(
        root,
        "dark-factory-orchestrator.yaml",
        "escalation:\n  host: 127.0.0.1\n  port: 8100\n  port: 8200\n",
    )

    assert fep.escalation_port(cfg) == 8100


@pytest.mark.parametrize(
    ("body", "why"),
    [
        pytest.param("dashboard:\n  port: 9999\n", "no escalation: block at all", id="no-block"),
        pytest.param("escalation:\n  host: 127.0.0.1\n", "block with no port:", id="no-port"),
        pytest.param("", "empty file", id="empty"),
    ],
)
def test_configs_without_an_escalation_port_return_none(
    fep: types.ModuleType, tmp_path: pathlib.Path, body: str, why: str
) -> None:
    """None means "claims nothing", which main() folds into the survey as no claim.

    Distinct from the error paths below only in cause, not in effect — but both must
    be None rather than a crash or a zero, because ``used[port]`` would otherwise
    record a phantom claim on port 0.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = _write_config(root, "dark-factory-orchestrator.yaml", body)

    assert fep.escalation_port(cfg) is None, why


def test_unreadable_paths_return_none_instead_of_crashing_the_survey(
    fep: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """The bare ``except Exception`` around ``read_text`` — one malformed project must
    not take down the whole survey.

    main() calls this for EVERY discovered root, so an unreadable path here is the
    difference between "that project is skipped" and "factory-init crashes on a box
    with one odd sibling directory". Both realistic causes are covered: a path that
    does not exist (FileNotFoundError) and a path that is a directory
    (IsADirectoryError).

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    removing the ``try/except Exception`` around ``read_text`` — RED, and note it
    fails as an ERROR escaping the function rather than as a wrong value, which is
    the crash this guard prevents::

        E  FileNotFoundError: [Errno 2] No such file or directory:
        E    '.../test_unreadable_paths_return_n0/nope/missing.yaml'
    """
    assert fep.escalation_port(tmp_path / "nope" / "missing.yaml") is None

    a_dir = tmp_path / "dark-factory-orchestrator.yaml"
    a_dir.mkdir()
    assert fep.escalation_port(a_dir) is None


# ---------------------------------------------------------------------------
# is_bound — the collision-avoidance primitive (real sockets, ephemeral ports)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _held_listener():
    """Hold a real listening socket on an OS-ASSIGNED ephemeral port; yield the port.

    ``bind(("127.0.0.1", 0))`` then ``getsockname()`` — never a hardcoded port. A
    literal would be a coin flip against whatever is actually running on this
    developer box, and the try/finally guarantees a failing assertion cannot leak a
    bound socket into a sibling test.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield sock.getsockname()[1]
    finally:
        sock.close()


def test_is_bound_tracks_a_real_listener(fep: types.ModuleType) -> None:
    """True while a real listener holds the port, False once it is released.

    WHY THIS IS THE ONLY TEST THAT TOUCHES REAL SOCKETS, and why every main() test
    below pins ``is_bound`` instead. MEASURED at base ``fc6f048b55`` with systemctl
    neutralised but ``is_bound`` live: ``--base 8100`` over a tmp tree returned 8104
    rather than 8102, and ``--base 8002`` returned 8004 rather than 8003 — because
    live escalation servers hold those ports on this host. A main() test that let
    the real probe run would therefore assert against the machine's current
    process table, which is the opposite of hermetic.

    WHY THE EPHEMERAL PORT IS REQUIRED: for the same reason. Asking the OS for a free
    port is the only way to get one this host is not already using.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    inverting the return values (``return True`` on successful bind, ``False`` in the
    OSError handler) — RED, and it caught BOTH directions rather than just one
    (``2 failed, 24 deselected``)::

        E  AssertionError: is_bound must report a held port as bound
        E  assert False is True
        E   +  where False = <function is_bound at 0x...>(45259)
        E  AssertionError: assert True is False
        E   +  where True = <function is_bound at 0x...>(34009)
    """
    with _held_listener() as port:
        assert fep.is_bound(port) is True, "is_bound must report a held port as bound"

    # The listener is closed now. SO_REUSEADDR on both sides means the TIME_WAIT
    # left by a never-connected listening socket does not block the re-bind.
    assert fep.is_bound(port) is False, "is_bound must report a released port as free"


def test_is_bound_leaves_nothing_behind(fep: types.ModuleType) -> None:
    """Two consecutive probes of a free port both report False.

    The function binds a socket to find out whether it can. If its own probe socket
    outlived the call, the SECOND probe would see the FIRST one's leftover and report
    the port as bound — and main()'s scan would then skip every port it examined,
    walking the whole 100-port window and returning rc 1 on a machine with nothing
    running. The ``with`` block is what prevents that; this asserts it.

    MEASURED RED at base ``fc6f048b55``, scratch mutation reverted before commit:
    replacing the ``with socket.socket(...)`` context manager with a bare socket held
    in a module-level list (the leak this test exists to catch) — RED on the SECOND
    probe only, exactly as the failure mode predicts::

        E  AssertionError: is_bound saw its own leftover probe socket
        E  assert True is False
        E   +  where True = <function is_bound at 0x...>(34439)
    """
    with _held_listener() as port:
        pass  # released immediately — now a known-free port on this host

    assert fep.is_bound(port) is False
    assert fep.is_bound(port) is False, "is_bound saw its own leftover probe socket"
