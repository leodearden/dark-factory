"""Pytest driver for scripts/tests/test_remove_lms_dropin.sh (task 4200).

That shell self-test was written under task 3750 to exercise the success path
of scripts/remove-lms-arm-worktree-dropin.sh -- a script that runs UNSANDBOXED
and UNATTENDED as a deterministic `before_done` deploy action -- somewhere
other than production.  It was correct, it was green, and NOTHING RAN IT: it
was the only .sh among 61 pytest modules in scripts/tests/, and pytest collects
only `test_*.py`.  A gate nobody runs rots silently, and the first person to
learn it had rotted would have been production.  This module is the collected
caller that closes that gap.

It is deliberately NOT `@pytest.mark.integration`.  The root pyproject.toml
addopts deselect that marker, so an integration-marked gate would be absent
from every default `verify` run -- present in the tree, checked by nobody,
which is verbatim the defect this module exists to fix.  See
scripts/tests/test_lms_verification_artifact.py:25-29 for the same reasoning
recorded for the same reason.  Instead the systemd-dependent tests are gated on
a COMPUTED skip reason, so they run for real wherever a --user manager is
reachable and skip cleanly (never error) in a sandbox or CI container that has
no session bus.
"""
from __future__ import annotations

import contextlib
import fcntl
import functools
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from uuid import uuid4

import pytest

# The repo's session-isolated subprocess runner (task 3798, THIRD DEFENCE), a
# documented drop-in for `subprocess.run(cmd, env=..., capture_output=True,
# text=True, timeout=...)` whose timeout path signals the whole process GROUP.
# Load-bearing here, not stylistic: see _run_selftest for why the direct-child
# kill that stock subprocess.run performs is the wrong one for this .sh.
# Importable because conftest.py appends REPO_ROOT to sys.path for this
# directory.
from df_pytest_isolation import run_in_new_session

SELFTEST_SH = Path(__file__).parent / "test_remove_lms_dropin.sh"


# ---------------------------------------------------------------------------
# The computed skip guard
# ---------------------------------------------------------------------------

def _unit_dir() -> Path:
    """Resolve the systemd --user unit dir EXACTLY as the .sh resolves it.

    The .sh (and the script under test) both spell this
    `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`.  `:-` treats an EMPTY
    value as absent, so this checks truthiness rather than key membership --
    otherwise an exported-but-empty XDG_CONFIG_HOME would resolve a relative
    path under the CWD and probe a directory the .sh will never touch.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "systemd" / "user"


def _systemd_user_manager_skip_reason() -> str | None:
    """Return why the systemd-backed tests must skip, or None to run them.

    Three independent preconditions, checked in order.  Each returned string
    says why the skip is an ENVIRONMENT FACT rather than a defect, so a
    skipped verify log is not mistaken for a silently disabled gate.

    1. ``systemctl`` must be on PATH.
    2. The --user MANAGER must be reachable.  Probed with
       ``systemctl --user show -p Version``, deliberately NOT
       ``is-system-running``: the latter reports non-zero for a merely
       ``degraded`` manager, which the operator host currently is and which
       is perfectly usable for the .sh (it only ever ``daemon-reload``s), so
       gating on it would over-skip on exactly the hosts where the test
       works.  Any non-zero rc, FileNotFoundError/OSError or timeout counts
       as unreachable -- a container with no DBUS_SESSION_BUS_ADDRESS or
       XDG_RUNTIME_DIR produces the first, a stripped image the others.
    3. The unit dir must be genuinely WRITABLE, confirmed by creating and
       unlinking a real probe file rather than by trusting
       ``os.access(..., os.W_OK)``.  os.access reports the mode bits, not a
       sandbox/LSM denial -- and the observed failure in dark-factory task
       worktrees is exactly that shape: files are user-owned and rw, yet the
       write is refused ("install: cannot remove ...: Permission denied").
       A mode-bit check would report writable, the guard would not fire, and
       the wrapper would ERROR instead of skipping.
    """
    if shutil.which("systemctl") is None:
        return (
            "no systemctl on this PATH -- scripts/tests/test_remove_lms_dropin.sh "
            "drives a real systemd --user manager and cannot run here.  This is "
            "an environment fact (sandbox/CI container), not a defect in the "
            "script under test"
        )

    try:
        probe = subprocess.run(
            ["systemctl", "--user", "show", "-p", "Version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        reachable = probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        # FileNotFoundError is an OSError subclass, so a systemctl that
        # vanished between the which() above and here lands here too.
        reachable = False
    if not reachable:
        return (
            "no reachable systemd --user manager (no session bus / "
            "XDG_RUNTIME_DIR, or the probe timed out) -- the .sh installs a "
            "throwaway template and asserts against `systemctl --user show`, "
            "which needs a live user manager.  Environment fact, not a defect"
        )

    unit_dir = _unit_dir()
    try:
        unit_dir.mkdir(parents=True, exist_ok=True)
        # The probe name satisfies _is_prunable_selftest_residue on purpose,
        # so a run SIGKILLed in the microseconds between the write and the
        # unlink strands a file the existing sweep can still reap.  An
        # unprefixed name (or a bare tempfile) would strand residue nothing
        # could ever collect -- the exact accumulation the prune was written
        # to solve.  Deliberately NOT widening the prune's suffix filter to
        # cover a dot-file instead: that filter is what keeps the sweep away
        # from _LOCK_NAME, and a prune that can delete a held lock file would
        # silently destroy the serialization seam below.
        #
        # The forward reference to _SELFTEST_PREFIX (defined below) is safe:
        # globals resolve at CALL time, and this function is now called
        # lazily -- see _skip_reason.
        write_probe = unit_dir / (
            f"{_SELFTEST_PREFIX}writeprobe-{os.getpid()}-{uuid4().hex[:8]}@.service"
        )
        write_probe.write_text("")
        write_probe.unlink()
    except OSError as exc:
        return (
            f"the systemd user unit directory {unit_dir} is not writable "
            f"({exc}) -- the .sh must install its throwaway template there "
            "because the --user manager resolves its unit search path from "
            "its OWN environment, so the dir cannot be redirected to a tmp "
            "path.  Environment fact (agent write-scope), not a defect"
        )

    return None


@functools.cache
def _skip_reason() -> str | None:
    """The skip reason, computed AT MOST ONCE and never at import time.

    Deliberately a cached function rather than the module constant
    ``_SKIP_REASON = _systemd_user_manager_skip_reason()`` this file used to
    carry.  The evaluate-ONCE property that constant existed for is preserved
    exactly by ``functools.cache`` -- what changes is WHEN.

    A module-scope call runs at COLLECTION: merely collecting
    ``scripts/tests/`` -- including ``--collect-only``, and including a ``-k``
    run that deselects every test in this file -- would spawn
    ``systemctl --user show`` with a 15s timeout AND mutate the operator's
    real home, because precondition 3 above ``mkdir -p``s
    ~/.config/systemd/user and writes a probe file inside it.  This repo's
    suite-wide isolation doctrine (df_pytest_isolation.py, whose four defences
    all exist to stop test runs touching live operator state) is squarely
    against that, and collection-time side effects are the worst version of it
    -- they are paid by runs that never intended to touch systemd at all.

    Lazily, the probe runs only when the one systemd-backed test actually
    runs, which is also the only moment its answer is load-bearing.
    """
    return _systemd_user_manager_skip_reason()


def _require_systemd_user_manager() -> None:
    """Skip the calling test unless a usable systemd --user manager is here.

    The in-test spelling of what used to be
    ``@pytest.mark.skipif(_SKIP_REASON is not None, ...)``.  A ``skipif``
    marker must have its condition evaluated at import to build the decorator,
    which is precisely the collection-time side effect ``_skip_reason``
    exists to avoid; calling this on the test's first line reports the same
    skip, with the same reason string, at the same granularity.
    """
    __tracebackhide__ = True  # report the skip against the TEST, not this helper
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)


# ---------------------------------------------------------------------------
# Per-invocation isolation: a unique throwaway template name
# ---------------------------------------------------------------------------

# The single shared stem.  _unique_template() GENERATES names with
# _SELFTEST_PREFIX and _prune_stale_selftest_units() DELETES names with
# _SELFTEST_PREFIX or exactly equal to _DEFAULT_TEMPLATE, so "the prune can
# never touch a real unit" is structural rather than a set of string literals
# free to drift apart.  Neither can prefix-match `lms-arm@` (the real unit the
# script under test targets) or any dark-factory unit.
#
# Two names rather than one widened prefix, on purpose: the generator binds
# ONLY _SELFTEST_PREFIX, so a real unit can never be generated into pruning
# scope; the prune additionally recognises the ONE fixed default name a
# by-hand run falls back to (see the .sh's TEMPLATE= line, pinned above by
# test_default_template_constant_matches_the_shell_default).  Both descend
# from _SELFTEST_STEM so they cannot drift apart from each other.
_SELFTEST_STEM = "lms-dropin-selftest"
_SELFTEST_PREFIX = f"{_SELFTEST_STEM}-"  # what _unique_template() generates
_DEFAULT_TEMPLATE = f"{_SELFTEST_STEM}@"  # what the .sh falls back to by hand


def _unique_template() -> str:
    """Return a fresh throwaway systemd TEMPLATE name for one .sh invocation.

    BOTH components are load-bearing:

    * ``os.getpid()`` disambiguates CONCURRENT pytest processes.  The fleet
      runs ``max_concurrent_tasks: 48`` and all 48 share one $HOME, hence one
      ~/.config/systemd/user.  Measured at plan time with the .sh's old
      hardcoded ``lms-dropin-selftest@``: two concurrent runs BOTH fail
      deterministically, because case 4's ``rm -f "$UNIT"`` tears down the
      other run's fixture mid-test ("drop-in still present after refusal",
      "template survives the re-run", and a WorkingDirectory that resolves to
      the OTHER run's mktemp dir).
    * The uuid suffix disambiguates repeat calls WITHIN one process -- a rerun
      in the same session, or a future second caller -- and survives PID reuse
      after a killed run left same-PID residue behind.

    Note what this does NOT solve.  Distinct names make the unit PATHS
    disjoint; they do nothing about CONTENTION on the single shared
    `systemd --user` manager, whose `daemon-reload` is globally serialized.
    That is _serialized_selftest_slot's job -- see its rationale block below.

    The trailing "@" makes it a template: the .sh appends ".service" and
    builds its probe unit as ``${TEMPLATE}probe.service``.
    """
    return f"{_SELFTEST_PREFIX}{os.getpid()}-{uuid4().hex[:8]}@"


def test_default_template_constant_matches_the_shell_default() -> None:
    """_DEFAULT_TEMPLATE must equal the .sh's actual fallback literal.

    This is a VALUE contract between two files -- the functional default the
    prune's bare-default arm must match -- NOT an assertion on prose,
    comments or docstrings.  Same-shaped precedent:
    tests/scripts/test_dashboard_service_template.py::
    _assert_known_project_roots_comma_separated, which parses a systemd unit
    file's Environment= line the same way: read the file, ``re.search`` an
    ANCHORED ``^...$`` pattern under ``re.MULTILINE``, assert the match
    exists, then assert on the extracted group.

    Anchored rather than a substring/``in`` check on purpose -- MEASURED: the
    literal ``lms-dropin-selftest@`` also appears in the .sh's own header
    comment ("(lms-dropin-selftest@) plus a drop-in") and
    ``LMS_SELFTEST_TEMPLATE=`` appears again in the Usage comment.  A naive
    ``"lms-dropin-selftest@" in sh_text`` check stays True even after the
    default on the ``TEMPLATE=`` line itself is changed to something else
    entirely, so it would silently pass through the exact drift this test
    exists to catch.

    The variable-name half of the pattern is built from ``_TEMPLATE_ENV_VAR``
    (``re.escape``d) rather than re-typed, so the two bind.  The forward
    reference to ``_TEMPLATE_ENV_VAR`` (defined below, near the .sh-driving
    helpers) is safe: globals resolve at CALL time, the same pattern this
    module already documents for the forward reference to ``_SELFTEST_PREFIX``
    in ``_systemd_user_manager_skip_reason``.
    """
    sh_text = SELFTEST_SH.read_text(encoding="utf-8")
    pattern = r'^TEMPLATE="\$\{' + re.escape(_TEMPLATE_ENV_VAR) + r':-([^"}]+)\}"$'
    match = re.search(pattern, sh_text, re.MULTILINE)
    assert match is not None, (
        f"{SELFTEST_SH} no longer has a line of the shape "
        f'TEMPLATE="${{{_TEMPLATE_ENV_VAR}:-<default>}}" -- this test cannot '
        "pin the .sh's default template without it."
    )
    assert match.group(1) == _DEFAULT_TEMPLATE, (
        f"the .sh's default template ({match.group(1)!r}) has drifted from "
        f"_DEFAULT_TEMPLATE ({_DEFAULT_TEMPLATE!r}).  The prune's bare-default "
        "arm keys on _DEFAULT_TEMPLATE, so this drift makes a killed hand-run's "
        "residue permanently unreapable again -- while every other assertion "
        "in this module stays green."
    )


# How long a selftest unit must go untouched before it counts as abandoned.
# Comfortably longer than a run's MEASURED cost (~5.2s solo on the operator
# host, and bounded above by _SELFTEST_TIMEOUT_S + _LOCK_WAIT_S even when the
# fleet is fully contended) and than the suite's own --timeout=300, so a live
# sibling is never mistaken for residue.
_STALE_AFTER_S = 3600.0


def _is_prunable_selftest_residue(name: str) -> bool:
    """Whether *name* is selftest residue that ``_prune_stale_selftest_units``
    may reap, gated by TWO independent properties:

    * THE SUFFIX GATE, evaluated FIRST.  Only a ``.service`` or
      ``.service.d`` name can be residue at all; anything else returns False
      immediately.  This is what keeps the sweep away from ``_LOCK_NAME`` and
      from any future selftest-prefixed bookkeeping file -- it is not
      cosmetic, see the rationale in ``_systemd_user_manager_skip_reason``
      (declining to widen this filter to cover a dot-file) and above
      ``_LOCK_NAME`` itself, and the ``non_units`` half of
      ``test_prune_never_touches_a_non_selftest_unit``.
    * THE NAME GATE.  The remaining stem must either start with
      ``_SELFTEST_PREFIX`` (a unique name ``_unique_template()`` generated)
      or equal ``_DEFAULT_TEMPLATE`` exactly (the ONE fixed name the .sh
      falls back to, pinned to the .sh's actual literal by
      ``test_default_template_constant_matches_the_shell_default``).  Real
      units (``lms-arm@``, ``fused-memory``, ``dark-factory-dashboard``) are
      excluded structurally, at any age, by this gate alone.

      The default arm is what this task adds, and is EXACT EQUALITY rather
      than a widened prefix: ``"lms-dropin-selftest@"`` does not start with
      ``"lms-dropin-selftest-"``, which is why the old prefix-only filter
      could never reach it.  Equality (rather than
      ``startswith(_SELFTEST_STEM)``) is what keeps the near-misses pinned by
      ``test_prune_match_predicate_admits_selftest_residue_and_nothing_else``
      excluded.
    """
    # 1. Suffix gate first.  "x.service.d".endswith(".service") is already
    #    False, so checking ".service.d" ahead of ".service" doesn't change
    #    which names match -- it just makes the two-suffix intent explicit
    #    rather than incidental.
    for suffix in (".service.d", ".service"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break
    else:
        return False
    # 2. Name gate: a unique generated name, or exactly the bare default.
    return stem.startswith(_SELFTEST_PREFIX) or stem == _DEFAULT_TEMPLATE


def _prune_stale_selftest_units(
    unit_dir: Path,
    *,
    now: float,
    max_age_s: float = _STALE_AFTER_S,
) -> None:
    """Best-effort removal of abandoned lms-dropin-selftest-* residue.

    The .sh cleans up after itself on every exit path via `trap cleanup EXIT`,
    but a SIGKILL has no exit path.  With the old fixed template name that was
    self-healing; with per-invocation unique names it is not, so this pays the
    cost explicitly.

    TWO SAFETY PROPERTIES, each with its own test above, because this deletes
    files out of the operator's LIVE unit directory:

    * Only names ``_is_prunable_selftest_residue`` admits are ever
      considered -- a ``_SELFTEST_PREFIX``-generated name (the same constant
      ``_unique_template()`` generates with) PLUS exactly
      ``_DEFAULT_TEMPLATE`` (the one fixed name a by-hand run falls back to),
      both suffix-gated to ``.service``/``.service.d``.  See that function
      for the full inclusion/exclusion table; a real ``lms-arm@`` or fleet
      unit is out of scope structurally, at any age.
    * Anything NEWER than max_age_s is left alone.  Under 48-way concurrency a
      fresh selftest unit is a running sibling's fixture, and deleting it
      would destroy that run -- and since the name gate above now also admits
      the shared ``_DEFAULT_TEMPLATE`` name, this guard is what protects a
      live BY-HAND run too, not just a concurrent pytest sibling.

    ``now`` and ``max_age_s`` are parameters rather than clock reads so the
    tests can drive this deterministically without sleeping or patching time.
    The whole body is best-effort: this is opportunistic hygiene called before
    the real work, and it must never be the thing that turns a green suite red.
    """
    try:
        if not unit_dir.is_dir():
            return
        for path in sorted(unit_dir.iterdir()):
            if not _is_prunable_selftest_residue(path.name):
                continue
            try:
                if now - path.stat().st_mtime <= max_age_s:
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                # One unreadable/undeletable entry must not abort the sweep of
                # the rest -- residue accumulates, so partial progress matters.
                continue
    except OSError:
        return


# ---------------------------------------------------------------------------
# step-1: the computed skip guard
# ---------------------------------------------------------------------------

def test_skip_reason_gates_on_binary_manager_and_writable_unit_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """_systemd_user_manager_skip_reason must gate on ALL THREE preconditions.

    Every case below is host-independent: the binary lookup, the --user
    manager probe and the unit directory are all monkeypatched rather than
    read from this host's real environment, so this test asserts the same
    thing on a live systemd box and in a DBUS-less container.  That is what
    keeps this module from being a total no-op in a sandbox -- the guard
    itself is genuinely exercised everywhere, and only the two subprocess
    tests skip.  Mirrors tests/scripts/test_dashboard_service_template.py::
    test_systemd_analyze_skip_reason_requires_both_binary_and_user_runtime_dir.

      (a) systemctl absent from PATH.
      (b) systemctl present, --user manager unreachable -- in all three ways
          it can be unreachable: non-zero rc, FileNotFoundError, and a
          probe that times out.
      (c) systemctl present and the manager reachable, but the unit dir
          cannot be written.  NOT hypothetical: agent write-scope in
          dark-factory task worktrees has been observed denying writes to
          ~/.config/systemd/user ("install: cannot remove ...: Permission
          denied") even for user-owned files.  Without this case the wrapper
          would ERROR rather than skip in such a sandbox -- the opposite of
          this module's "stay green in sandboxes and CI" contract.
      (d) all three satisfied -- the guarded tests must actually RUN.

    Assertions are on `is None` / `is not None` plus a substring UNIQUE to
    the branch under test, never on exact prose, so the reason strings stay
    free to explain themselves while still identifying WHICH precondition
    fired.
    """
    writable_cfg = tmp_path / "cfg"
    writable_cfg.mkdir()

    def _ok_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["systemctl"], returncode=0, stdout="Version=255.4\n", stderr=""
        )

    # (a) Binary absent -- gated even with a reachable manager and a
    # perfectly writable unit dir alongside it.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(writable_cfg))
    monkeypatch.setattr(subprocess, "run", _ok_probe)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    # "on this PATH", NOT the bare word "systemctl".  Branch 2's
    # manager-unreachable string also contains "systemctl" -- it quotes
    # `systemctl --user show` -- so a bare-word assertion would still pass if
    # this branch were deleted or reordered and control fell through to the
    # manager probe, i.e. it could not tell the two apart.  Cases (b) and (c)
    # below are discriminating for the same reason.
    assert "on this PATH" in reason

    # (b) Binary present, manager unreachable -- three distinct failure
    # modes, because a container with no DBUS_SESSION_BUS_ADDRESS /
    # XDG_RUNTIME_DIR produces the non-zero-rc one while a stripped image
    # produces the others, and all three must skip rather than error.
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/systemctl")

    def _rc1_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["systemctl"],
            returncode=1,
            stdout="",
            stderr="Failed to connect to bus: No medium found\n",
        )

    monkeypatch.setattr(subprocess, "run", _rc1_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "user manager" in reason

    def _missing_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory: 'systemctl'")

    monkeypatch.setattr(subprocess, "run", _missing_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "user manager" in reason

    def _timeout_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=5)

    monkeypatch.setattr(subprocess, "run", _timeout_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "user manager" in reason

    # (c) Binary present and manager reachable, but the unit dir is
    # unwritable.  Rendered here by pointing XDG_CONFIG_HOME at a regular
    # FILE, so "<file>/systemd/user" can never be created -- a denial that
    # holds for any uid, unlike a chmod, which root would sail straight
    # through and turn this case into a silent no-op.
    not_a_dir = tmp_path / "cfg-is-a-file"
    not_a_dir.write_text("")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(not_a_dir))
    monkeypatch.setattr(subprocess, "run", _ok_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "unit dir" in reason.lower() or "unit directory" in reason.lower()

    # (d) All three satisfied: the guarded tests MUST run.  A guard that
    # never returns None would skip everywhere and re-create the very rot
    # this module exists to end.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(writable_cfg))
    reason = _systemd_user_manager_skip_reason()
    assert reason is None


def test_skip_reason_falls_back_to_home_config_like_the_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The unit dir must resolve EXACTLY as the .sh resolves it.

    The .sh uses `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`, whose `:-`
    treats an EMPTY value as absent.  A Python guard checking key membership
    instead of truthiness would resolve `""/systemd/user` -> a relative path
    under the CWD, probe a directory the .sh will never touch, and report the
    wrong answer in both directions.  Pinned host-independently against a
    tmp HOME.
    """
    fake_home = tmp_path / "home"
    (fake_home / ".config").mkdir(parents=True)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/systemctl")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["systemctl"], returncode=0, stdout="Version=255.4\n", stderr=""
        ),
    )
    monkeypatch.setenv("HOME", str(fake_home))

    # Empty XDG_CONFIG_HOME must fall back to $HOME/.config, exactly as `:-`
    # does -- so the writable tmp HOME below is what gets probed, and the
    # guard returns None.
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    assert _systemd_user_manager_skip_reason() is None
    assert (fake_home / ".config" / "systemd" / "user").is_dir()

    # And an UNSET XDG_CONFIG_HOME resolves the same way.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert _systemd_user_manager_skip_reason() is None


def test_importing_this_module_touches_no_unit_directory(tmp_path: Path) -> None:
    """COLLECTION must be inert.  The guard is armed, not merely defined.

    Every other assertion in this file would stay green if the skip reason
    were computed at module scope again -- that spelling works, it is just
    impure.  This is the only test that can tell the difference, and it is the
    difference that matters to every OTHER suite: `pytest scripts/tests/`
    imports this module during collection, so a module-scope probe would spawn
    `systemctl --user show` with a 15s timeout and `mkdir -p` the operator's
    real ~/.config/systemd/user on every run that merely COLLECTS this
    directory -- `--collect-only`, or a `-k` run that deselects everything
    here.  The repo's isolation doctrine (df_pytest_isolation.py) exists to
    stop exactly that.

    Imported in a SUBPROCESS because this module is already imported in the
    running interpreter: by the time any test executes, an import-time side
    effect has long since happened and is unobservable from here.  A fresh
    interpreter with XDG_CONFIG_HOME and HOME redirected at tmp_path is the
    only place the question can still be asked.

    Host-independent -- no systemd, no real unit dir, so this runs everywhere,
    including the sandboxes where test_shell_selftest_passes skips.
    """
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    home = tmp_path / "home"
    home.mkdir()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    # Both, deliberately: XDG_CONFIG_HOME is what _unit_dir reads, and HOME is
    # its `:-` fallback -- so a regression that ignored the first still lands
    # in tmp_path rather than in the operator's real home.
    env["XDG_CONFIG_HOME"] = str(cfg)
    env["HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )

    program = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('_wrapper_import_probe', {str(Path(__file__).resolve())!r})\n"  # noqa: E501
        "assert spec is not None and spec.loader is not None\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
    )
    result = run_in_new_session([sys.executable, "-c", program], env=env, timeout=120)

    assert result.returncode == 0, (
        "importing this module in a fresh interpreter failed.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert not (cfg / "systemd").exists(), (
        f"importing this module created {cfg / 'systemd'} -- the skip probe ran at "
        "IMPORT time.  Compute it lazily (see _skip_reason): collection must not "
        "mkdir or write into the operator's real systemd unit directory."
    )
    assert not (home / ".config").exists(), (
        f"importing this module created {home / '.config'} -- the skip probe ran at "
        "IMPORT time via the $HOME fallback.  See _skip_reason."
    )


# ---------------------------------------------------------------------------
# step-3: RED -- the per-invocation unique template name
# ---------------------------------------------------------------------------

def test_unique_template_is_a_legal_distinct_systemd_template_name() -> None:
    """_unique_template must produce a fresh, legal systemd TEMPLATE name.

    Isolation between concurrent runs comes from the unit NAME and nothing
    else: the systemd --user manager resolves its unit search path from its
    OWN environment, so pointing XDG_CONFIG_HOME at a tmp dir would give the
    fixture a directory the manager never reads.  The .sh must therefore keep
    writing to the real ~/.config/systemd/user, and every property below is
    what makes that safe under `max_concurrent_tasks: 48` sharing one $HOME.

      (a) Trailing "@".  The .sh appends ".service" and builds its probe as
          "${TEMPLATE}probe.service"; a missing "@" silently yields a plain
          unit instead of a template, and the probe unit resolves to nothing.
      (b) Two calls differ.  This is what carries the cross-process isolation
          property STRUCTURALLY: the .sh derives UNIT, DROPIN_DIR and its
          probe unit from `$TEMPLATE` alone, so distinct names imply disjoint
          absolute paths, and no live multi-process test is needed to
          establish it.  (One used to be here.  It cost 71% of the module and
          multiplied the daemon-reload contention it was meant to model, while
          its three threads shared one PID -- so it modelled the SAME-process
          case, not the fleet's cross-process one.  Removed under esc-4200-2.)
      (c) The shared _SELFTEST_STEM.  The generator binds _SELFTEST_PREFIX;
          the prune's predicate binds both _SELFTEST_PREFIX and
          _DEFAULT_TEMPLATE -- and all three descend from the one
          _SELFTEST_STEM constant, so the prune's "never touch a real unit"
          property stays structural rather than a set of string literals
          free to drift, even though it is now delivered by two sibling
          constants instead of one.
      (d) A legal systemd unit-name charset.  Verified viable at plan time:
          PID-suffixed template names resolve correctly against a real
          manager.

    No systemd required -- this runs everywhere.
    """
    name = _unique_template()

    # (a)
    assert name.endswith("@"), (
        f"_unique_template() must return a TEMPLATE name ending in '@'; got {name!r}. "
        "The .sh builds '${TEMPLATE}probe.service' from it."
    )
    # (b)
    assert _unique_template() != _unique_template()
    # (c)
    assert name.startswith(_SELFTEST_PREFIX)
    # (d)
    assert re.fullmatch(r"[A-Za-z0-9:_.\-]+@", name), (
        f"{name!r} is not a legal systemd template unit name; systemd accepts "
        r"only [A-Za-z0-9:_.\-] before the '@'."
    )


# ---------------------------------------------------------------------------
# step-5: RED -- the stale-residue prune
# ---------------------------------------------------------------------------

def test_prune_match_predicate_admits_selftest_residue_and_nothing_else() -> None:
    """_is_prunable_selftest_residue must admit exactly the reapable residue.

    Table-driven, and written BEFORE the predicate exists: the near-miss rows
    below are what FORCE the default arm to be a tight ``stem ==
    _DEFAULT_TEMPLATE`` equality check rather than a widened bare-stem prefix
    match.  Nothing else in this module would catch that sloppier widening --
    the existing test_prune_never_touches_a_non_selftest_unit only plants
    ``lms-arm@``, the fleet units, ``_LOCK_NAME`` and a prefixed ``.conf``,
    none of which are near-misses of the DEFAULT arm this task adds.

    The ``.service``/``.service.d`` suffix gate must be evaluated FIRST
    inside the predicate and must never be weakened: it is the only thing
    standing between this sweep and ``_LOCK_NAME`` (see that case below).
    """
    cases: list[tuple[str, bool, str]] = [
        # MUST MATCH -- today's unique generated names.
        (f"{_SELFTEST_PREFIX}999-deadbeef@.service", True, "a unique generated unit"),
        (f"{_SELFTEST_PREFIX}999-deadbeef@.service.d", True, "a unique generated drop-in dir"),
        # MUST MATCH -- THE DEFECT this task fixes: the bare default a killed
        # hand-run strands, unreapable before this predicate existed.
        (f"{_DEFAULT_TEMPLATE}.service", True, "the bare-default unit a hand-run strands"),
        (f"{_DEFAULT_TEMPLATE}.service.d", True, "the bare-default drop-in dir a hand-run strands"),
        # MUST NOT MATCH -- real units, at any age.
        ("lms-arm@.service", False, "the real unit the script under test targets"),
        ("lms-arm@.service.d", False, "the real unit's drop-in dir"),
        ("fused-memory.service", False, "a real fleet unit"),
        ("dark-factory-dashboard.service", False, "a real fleet unit"),
        ("dark-factory-dashboard.service.d", False, "a real fleet unit's drop-in dir"),
        # MUST NOT MATCH -- the serialization lock: fails the suffix gate.
        # fcntl.flock lives on the open file DESCRIPTION, so unlinking the
        # path releases nothing -- it lets the next session create a fresh
        # inode and take a second "exclusive" slot, silently unserializing
        # the one real-systemd leg (seam added in commit fabf652c83).
        (_LOCK_NAME, False, "the host-wide serialization lock"),
        # MUST NOT MATCH -- selftest-prefixed/stemmed non-unit files.
        (f"{_SELFTEST_PREFIX}x@.conf", False, "a selftest-prefixed non-unit file"),
        (f"{_SELFTEST_STEM}.log", False, "a selftest-stemmed non-unit file"),
        # MUST NOT MATCH -- near-misses of the widened default arm that a
        # sloppy `startswith(_SELFTEST_STEM)` widening would wrongly admit,
        # and which nothing else in the module can catch.
        (f"{_SELFTEST_STEM}.service", False, "no '@' -- not a template at all"),
        (f"{_SELFTEST_STEM}ZZZ@.service", False, "stem is a PREFIX of, not equal to, the default"),
        (f"{_SELFTEST_STEM}@probe.service", False, "a resolved instance name, not the template file"),
        # MUST NOT MATCH -- degenerate.
        (".service", False, "empty stem"),
        ("", False, "empty name"),
    ]
    for name, expected, why in cases:
        assert _is_prunable_selftest_residue(name) is expected, (
            f"{name!r} ({why}): expected _is_prunable_selftest_residue(name) is "
            f"{expected!r}"
        )


def _write_unit(unit_dir: Path, template: str, *, age_s: float, now: float) -> tuple[Path, Path]:
    """Install a <template>.service + <template>.service.d/ pair aged age_s.

    Mirrors the exact on-disk shape the .sh's install_fixture() leaves behind,
    so the prune is driven against real residue rather than a stand-in.
    """
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / f"{template}.service"
    dropin_dir = unit_dir / f"{template}.service.d"
    dropin_dir.mkdir(exist_ok=True)
    unit.write_text("[Service]\nExecStart=/bin/true\n")
    (dropin_dir / "10-worktree-3713.conf").write_text("[Service]\nWorkingDirectory=/x\n")
    stamp = now - age_s
    for path in (unit, dropin_dir / "10-worktree-3713.conf", dropin_dir):
        os.utime(path, (stamp, stamp))
    return unit, dropin_dir


def test_prune_removes_stale_selftest_residue(tmp_path: Path) -> None:
    """(a) An OLD selftest unit and its drop-in dir are both removed.

    Uniquifying the template name costs exactly one thing, and this is it.
    The .sh's old FIXED name self-healed: a run killed before its EXIT trap
    fired left residue the next run simply overwrote and removed.  Unique
    names have no such self-healing -- a SIGKILLed run (verify timeout, task
    cancellation, both routine in this fleet) strands
    lms-dropin-selftest-<pid>-<hex>@.service in the operator's LIVE unit dir
    forever, and those accumulate, slowing every daemon-reload and polluting
    `systemctl --user list-unit-files`.  Driven against tmp_path with
    hand-set mtimes: no systemd, no sleeping, no real unit dir.
    """
    now = 1_000_000.0
    unit_dir = tmp_path / "systemd" / "user"
    unit, dropin_dir = _write_unit(
        unit_dir, f"{_SELFTEST_PREFIX}999-deadbeef@", age_s=7200.0, now=now
    )

    _prune_stale_selftest_units(unit_dir, now=now, max_age_s=3600.0)

    assert not unit.exists(), f"stale {unit.name} should have been pruned"
    assert not dropin_dir.exists(), f"stale {dropin_dir.name}/ should have been pruned"


def test_prune_leaves_a_fresh_selftest_unit_alone(tmp_path: Path) -> None:
    """(b) SAFETY -- a FRESH selftest unit belongs to a RUNNING sibling.

    Under 48-way concurrency another task's .sh may be mid-run right now, and
    its fixture is a selftest-prefixed unit that is only seconds old.
    Deleting it would destroy that run's fixture and turn a green sibling red
    -- reintroducing, from the cleanup side, the very collision the unique
    template name was introduced to eliminate.  Age is the only thing
    separating "my abandoned residue" from "someone else's live fixture".
    """
    now = 1_000_000.0
    unit_dir = tmp_path / "systemd" / "user"
    unit, dropin_dir = _write_unit(
        unit_dir, f"{_SELFTEST_PREFIX}12345-cafebabe@", age_s=5.0, now=now
    )

    _prune_stale_selftest_units(unit_dir, now=now, max_age_s=3600.0)

    assert unit.exists(), "a FRESH selftest unit belongs to a concurrent run and must survive"
    assert dropin_dir.exists(), "a FRESH selftest drop-in dir must survive"


def test_prune_reaps_bare_default_template_residue_but_only_when_stale(tmp_path: Path) -> None:
    """THE DEFECT this task fixes, plus the freshness guard that makes it safe.

    (a) THE DEFECT, expected RED before step-6 wires the predicate in: a
    STALE bare-default unit -- what a hand-run SIGKILLed before its
    ``trap cleanup EXIT`` fires strands in the operator's live unit dir --
    must be reaped.  Measured before this fix: both the unit and its
    drop-in dir survive, because ``"lms-dropin-selftest@.service"`` does not
    start with ``_SELFTEST_PREFIX``.

    (b) SAFETY, expected green before and after -- a regression pin in this
    module's "Expected GREEN on arrival" idiom.  This half is mandatory, not
    decorative: the default template name is SHARED across every hand-run,
    unlike ``_unique_template()``'s per-invocation names, so once the prune
    can match it at all, ``_STALE_AFTER_S`` becomes the ONLY thing
    separating "my abandoned residue" from "an operator's live hand-run
    fixture".  Deleting a fresh one would tear that run down mid-test,
    re-introducing from the cleanup side exactly the collision uniquification
    was introduced to remove.  The margin is large and measured: a solo .sh
    run is 5.2s against ``_STALE_AFTER_S = 3600.0`` (~690x).  This matters
    most on the one path where the sweep runs UNSERIALIZED:
    test_shell_selftest_passes calls ``_prune_stale_selftest_units`` BEFORE
    entering ``_serialized_selftest_slot()``, so the freshness guard is the
    ENTIRE protection there -- which is why this gets its own assertion
    rather than riding on test_prune_leaves_a_fresh_selftest_unit_alone,
    which covers only the unique-name arm.

    Both halves use ``_DEFAULT_TEMPLATE`` -- never a re-typed literal, since
    step-1 pins that constant to the .sh's actual default -- and separate tmp
    unit dirs, so a bug that makes (a) pass by accident cannot also make (b)
    pass by accident.
    """
    now = 1_000_000.0

    stale_unit_dir = tmp_path / "stale" / "systemd" / "user"
    stale_unit, stale_dropin_dir = _write_unit(
        stale_unit_dir, _DEFAULT_TEMPLATE, age_s=7200.0, now=now
    )

    fresh_unit_dir = tmp_path / "fresh" / "systemd" / "user"
    fresh_unit, fresh_dropin_dir = _write_unit(
        fresh_unit_dir, _DEFAULT_TEMPLATE, age_s=5.0, now=now
    )

    _prune_stale_selftest_units(stale_unit_dir, now=now, max_age_s=3600.0)
    _prune_stale_selftest_units(fresh_unit_dir, now=now, max_age_s=3600.0)

    assert not stale_unit.exists(), f"stale {stale_unit.name} should have been pruned"
    assert not stale_dropin_dir.exists(), (
        f"stale {stale_dropin_dir.name}/ should have been pruned"
    )
    assert fresh_unit.exists(), (
        "a FRESH bare-default unit belongs to a live hand-run and must survive"
    )
    assert fresh_dropin_dir.exists(), "a FRESH bare-default drop-in dir must survive"


def test_prune_never_touches_a_non_selftest_unit(tmp_path: Path) -> None:
    """(c) SAFETY -- a real unit is never swept up, at ANY age.

    This prune deletes files out of the operator's live
    ~/.config/systemd/user.  `lms-arm@` is the actual unit the script under
    test targets, and the dark-factory units below are the live fleet.  All
    are aged FAR beyond max_age_s -- age must be irrelevant for anything
    lacking the selftest prefix, or this module becomes the outage it was
    written to prevent.

    Second half: the SUFFIX filter, which neither the prefix case above nor
    the age case in the previous test can reach.  The prune requires BOTH
    _SELFTEST_PREFIX and a .service/.service.d suffix, and the suffix half is
    load-bearing on its own -- see the rationale in
    _systemd_user_manager_skip_reason, which declines to widen it precisely
    because it is what keeps this sweep away from _LOCK_NAME.  Without a test,
    widening the filter to the prefix alone would delete a live lock file with
    every other prune assertion still green.
    """
    now = 1_000_000.0
    unit_dir = tmp_path / "systemd" / "user"
    reals = [
        _write_unit(unit_dir, "lms-arm@", age_s=86_400.0 * 30, now=now),
        _write_unit(unit_dir, "fused-memory", age_s=86_400.0 * 90, now=now),
        _write_unit(unit_dir, "dark-factory-dashboard", age_s=86_400.0 * 90, now=now),
    ]

    # Aged FAR past max_age_s, so ONLY the suffix filter stands between each
    # of these and deletion -- the second one carries _SELFTEST_PREFIX too, so
    # for it the suffix is the only surviving guard of the three.
    #
    #   * _LOCK_NAME is the host-wide serialization lock.  `fcntl.flock` lives
    #     on an open file DESCRIPTION, not on the path, so unlinking it does
    #     not release anything: it makes the next session create a fresh inode
    #     and take a second "exclusive" slot on it, silently unserializing the
    #     one real-systemd leg while every other assertion here stayed green.
    #   * A selftest-prefixed .conf stands in for any future non-unit
    #     bookkeeping file a widened prefix-only filter would sweep up.
    non_units: list[Path] = []
    for name in (_LOCK_NAME, f"{_SELFTEST_PREFIX}x@.conf"):
        path = unit_dir / name
        path.write_text("")
        os.utime(path, (now - 86_400.0, now - 86_400.0))
        non_units.append(path)

    _prune_stale_selftest_units(unit_dir, now=now, max_age_s=3600.0)

    for unit, dropin_dir in reals:
        assert unit.exists(), f"{unit.name} is a REAL unit and must never be pruned"
        assert dropin_dir.exists(), f"{dropin_dir.name}/ is REAL and must never be pruned"

    for path in non_units:
        assert path.exists(), (
            f"{path.name} carries no .service/.service.d suffix and must never be "
            "pruned.  That suffix filter is what keeps this sweep away from the "
            "host-wide serialization lock; a prune that can unlink a held lock file "
            "silently destroys the serialization seam."
        )


def test_prune_is_silent_when_the_unit_dir_does_not_exist(tmp_path: Path) -> None:
    """A fresh checkout or a container with no ~/.config/systemd/user.

    The prune is opportunistic hygiene called before the real work; it must
    never be the thing that turns a green suite red.
    """
    _prune_stale_selftest_units(
        tmp_path / "nope" / "systemd" / "user", now=1_000_000.0, max_age_s=3600.0
    )


# ---------------------------------------------------------------------------
# step-7: cross-PROCESS serialization of the one real-systemd leg
# ---------------------------------------------------------------------------
#
# WHY A LOCK ON TOP OF THE UNIQUE TEMPLATE NAME.  The two seams solve DIFFERENT
# problems and neither substitutes for the other:
#
#   * _unique_template() fixes NAME COLLISION -- two runs writing and `rm`ing
#     the same absolute unit path.  Distinct names make the paths disjoint.
#   * This lock fixes CONTENTION on the single shared `systemd --user` manager,
#     which distinct names do nothing about.  `systemctl --user daemon-reload`
#     is globally serialized inside the manager, and it is essentially ALL of
#     this test's cost: MEASURED on the operator host, daemon-reload is
#     0.85-0.94s against 66 unit files, one .sh run performs ~5 of them (3 in
#     the .sh, 2 more inside the script under test), and a solo run costs
#     5.20s end to end.  Wall clock therefore scales LINEARLY in the number of
#     concurrent runs no matter how unique the names are.
#
# That linear scaling is the whole risk.  The repo root sets
# merge_verify_breadth: "full" and the fleet runs max_concurrent_tasks: 48
# against one $HOME and one user manager, so an unserialized leg would reach
# its subprocess timeout under enough concurrency and go RED on branches with
# no defect -- blocking every merge, review checkpoint and main-tip sweep
# repo-wide.  A test that manufactures a repo-wide merge blocker is strictly
# worse than the rot this module was written to fix.
#
# So: at most ONE process anywhere on this host drives the .sh at a time, and a
# session that cannot get a slot within _LOCK_WAIT_S SKIPS with a reason rather
# than failing.  Skipping is the correct outcome for a contended slot -- the
# holder is running the identical gate against the identical code right now, so
# the gate's coverage is delivered by that run; a red here would report an
# environment fact as a defect.

# The rendezvous file, in ~/.config/systemd/user because that (via $HOME) is
# the one thing all 48 concurrent worktrees demonstrably share -- the same
# reason the collision existed at all.  The leading dot keeps systemd from
# ever parsing it, and _prune_stale_selftest_units cannot reach it either:
# _is_prunable_selftest_residue gates on a .service/.service.d suffix FIRST,
# and this name has none.
_LOCK_NAME = ".lms-dropin-selftest.lock"

# Budget, sized against the suite's per-test --timeout=300
# (scripts/orchestrator.yaml).  The worst case for test_shell_selftest_passes,
# which is the one test that can pay ALL of these on a single invocation, is a
# sum of FOUR terms -- an earlier version of this comment counted only the
# middle two, put the total at 270, and thereby claimed a 30s margin the
# arithmetic did not support:
#
#     15  _require_systemd_user_manager -> _skip_reason: the
#         `systemctl --user show` probe, which runs BEFORE the lock is taken
#         and can burn its full timeout= against a wedged manager
#     90  _LOCK_WAIT_S -- the bounded wait for the host-wide slot
#    150  _SELFTEST_TIMEOUT_S -- the .sh subprocess itself
#     10  run_in_new_session's bounded post-kill drain
#         (df_pytest_isolation.py::_POST_KILL_DRAIN_SECS), charged ON TOP of
#         the timeout above whenever a grandchild still holds the stdout pipe
#    ---
#    265 < 300
#
# leaving ~35s for the prune, the residue sweep and filesystem overhead.  With
# that margin a wedged run always surfaces as this module's own diagnosable
# skip or failure -- naming the probe, the lock or the .sh -- rather than as an
# opaque suite-level timeout that names no cause.
#
# _LOCK_WAIT_S = 90 is ~17x the measured 5.2s solo run, i.e. ~17 fully
# serialized predecessors.  Sizing it for ~20 was conservative to begin with:
# `verify_admission_pytest_n: 8` already caps the fleet at 8 concurrent pytest
# verifies, so fewer than 8 real predecessors can ever be queued ahead of this
# leg and the wait was never the binding constraint -- trimming it to buy the
# margin above costs nothing.  _SELFTEST_TIMEOUT_S = 150 is ~29x the measured
# solo cost, and under the lock the run is EXCLUSIVE, so the only load it now
# has to absorb is unrelated daemon-reload traffic rather than 47 copies of
# itself.
_LOCK_WAIT_S = 90.0
_SELFTEST_TIMEOUT_S = 150
_LOCK_POLL_S = 0.25


@contextlib.contextmanager
def _serialized_selftest_slot(*, wait_s: float = _LOCK_WAIT_S) -> Iterator[Path]:
    """Hold an exclusive host-wide slot for driving the .sh, or skip.

    `fcntl.flock` rather than a lockfile-existence protocol on purpose: the
    kernel drops a flock when the holding file description closes, INCLUDING
    on SIGKILL, so a verify timeout or a task cancellation -- both routine in
    this fleet -- cannot strand the lock and wedge every later session.  A
    hand-rolled "create a file, delete it in a finally" would have exactly
    that failure mode, and would need the same stale-age heuristic
    _prune_stale_selftest_units carries.

    Polled LOCK_NB rather than a blocking LOCK_EX so the wait is BOUNDED: an
    unbounded block would sail past the suite's --timeout=300 and report the
    contention as an opaque suite timeout.

    Both give-up paths call `pytest.skip`, never `fail`: an unwritable unit
    dir and a busy host are environment facts, and the guarded tests are
    already skipif-gated on exactly that kind of fact.
    """
    lock_path = _unit_dir() / _LOCK_NAME
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        pytest.skip(
            f"cannot open the selftest serialization lock {lock_path} ({exc}) -- "
            "environment fact (agent write-scope / read-only $HOME), not a defect"
        )

    deadline = time.monotonic() + wait_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    pytest.skip(
                        f"another process has held {lock_path} for more than {wait_s:g}s "
                        "-- this .sh is serialized host-wide because "
                        "`systemctl --user daemon-reload` is globally serialized, and "
                        "the holder is running this identical gate right now.  "
                        "Environment fact (fleet concurrency), not a defect"
                    )
                time.sleep(_LOCK_POLL_S)
        try:
            yield lock_path
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_serialized_slot_is_exclusive_and_released(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The slot must actually EXCLUDE a concurrent holder, and let go after.

    Host-independent: XDG_CONFIG_HOME is redirected at tmp_path, so no systemd
    and no real unit dir are involved and this runs everywhere -- including
    the sandboxes where the two subprocess tests skip.  A lock that silently
    failed to exclude would restore the unbounded contention this seam exists
    to remove, and nothing else in the module would notice.

    The second handle is a SEPARATE ``os.open``, which the kernel treats as a
    distinct open file description; flock conflicts between two such
    descriptions even inside one process, so this models the cross-process
    case without spawning one.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with _serialized_selftest_slot() as lock_path:
        assert lock_path == tmp_path / "systemd" / "user" / _LOCK_NAME
        rival = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            with pytest.raises(OSError):
                fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(rival)

    # Released on exit -- otherwise the FIRST session on a host would wedge
    # every later one for the life of the process.
    rival = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(rival, fcntl.LOCK_UN)
    finally:
        os.close(rival)


def test_serialized_slot_skips_rather_than_fails_when_contended(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A busy host must SKIP, never go red -- the point of the whole seam.

    ``wait_s=0`` renders "the deadline expired" without sleeping.  If this
    ever raised anything other than Skipped, a contended fleet would report an
    environment fact as a defect and block every merge repo-wide, which is the
    exact outcome this module must not produce.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)

    holder = os.open(unit_dir / _LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        # Combined `with` (ruff SIM117), which is exactly equivalent here:
        # the slot's __enter__ runs INSIDE the raises block, so a Skipped
        # raised during acquisition is still what gets caught.
        with (
            pytest.raises(pytest.skip.Exception, match="fleet concurrency"),
            _serialized_selftest_slot(wait_s=0.0),
        ):
            pytest.fail("the slot must not be entered while a rival holds the lock")
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


# The .sh's side of the seam reads `${LMS_SELFTEST_TEMPLATE:-...}`.  Named
# once here so the single writer below and the guard that protects it bind the
# same string; the .sh's own literal is deliberately NOT derived from this --
# a test that generated the name it then asserts on would assert nothing.
_TEMPLATE_ENV_VAR = "LMS_SELFTEST_TEMPLATE"


def _run_selftest(
    template: str,
    *,
    extra_env: Mapping[str, str] | None = None,
    timeout: float = _SELFTEST_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    """Drive scripts/tests/test_remove_lms_dropin.sh against ONE template name.

    The template is threaded in through the LMS_SELFTEST_TEMPLATE seam, which
    the .sh then forwards to the script under test via its existing
    LMS_UNIT_TEMPLATE seam -- so this single variable propagates through both
    halves and gives the invocation an absolute unit path no concurrent run
    shares.

    EVERY caller goes through here, including
    test_selftest_template_seam_propagates, which is what makes that test
    cover the WRAPPER half of the seam and not just the .sh half.  When it
    built its own env dict inline, a misspelled or dropped key on the
    ``env[...] =`` line below broke nothing visible: the seam test still
    passed (it set the variable itself), test_shell_selftest_passes still
    passed solo (the .sh fell back to the shared default template), and the
    measured concurrency collision quietly re-opened.  ``extra_env`` exists so
    a test can vary the ENVIRONMENT AROUND the seam without owning the seam.

    ``extra_env`` is applied BEFORE the seam and may not contain it: this
    function is the single writer of LMS_SELFTEST_TEMPLATE, and a caller that
    could supply its own would silently restore exactly the blind spot above.

    ``run_in_new_session`` rather than a bare ``subprocess.run(timeout=...)``,
    which is this repo's documented convention (task 3798) and is load-bearing
    on exactly the path _SELFTEST_TIMEOUT_S makes reachable.  stock
    ``subprocess.run``'s POSIX timeout branch calls ``Popen.kill()``, which
    signals the DIRECT CHILD only: the ``bash`` dies without running its
    ``trap cleanup EXIT`` (stranding the template in the live unit dir for
    _prune_stale_selftest_units to reap an hour later), while any in-flight
    ``systemctl``/``python3`` grandchild is reparented to ``systemd --user``
    still holding the stdout/stderr pipe write ends.  Signalling the process
    GROUP reaches them, and the helper's bounded post-kill drain means a
    surviving pipe-holder cannot turn ``timeout=`` into an unbounded block --
    which matters doubly here, because this call runs while holding the
    host-wide slot every other session is waiting on.
    """
    if extra_env is not None and _TEMPLATE_ENV_VAR in extra_env:
        raise ValueError(
            f"{_TEMPLATE_ENV_VAR} must not be supplied via extra_env -- "
            "_run_selftest is its single writer, and a caller that sets it "
            "itself makes test_selftest_template_seam_propagates blind to a "
            "dropped seam.  Pass the name as the `template` argument."
        )
    env = os.environ.copy()
    if extra_env is not None:
        env.update(extra_env)
    env[_TEMPLATE_ENV_VAR] = template
    return run_in_new_session(
        ["bash", str(SELFTEST_SH)],
        env=env,
        timeout=timeout,
    )


def _remove_template_residue(template: str) -> None:
    """Delete one template's unit + drop-in dir from the REAL unit directory.

    The .sh cleans up after itself via `trap cleanup EXIT`; this is the
    belt-and-braces for the path where it cannot -- a mid-test failure or an
    exception raised between launch and assertion.  Best-effort by design.
    """
    unit_dir = _unit_dir()
    try:
        (unit_dir / f"{template}.service").unlink(missing_ok=True)
        shutil.rmtree(unit_dir / f"{template}.service.d", ignore_errors=True)
    except OSError:
        pass


def _assert_selftest_passed(result: subprocess.CompletedProcess[str], template: str) -> None:
    """Assert one run is green, surfacing its full output on failure.

    A future breakage of the script under test must be diagnosable from the
    verify log ALONE -- nobody will have a live systemd box and this worktree
    in hand when it goes red.  The .sh prints a PASS/FAIL line per check, so
    embedding stdout names the exact check that broke.
    """
    detail = (
        f"\n--- template: {template}\n"
        f"--- exit code: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.returncode == 0, f"{SELFTEST_SH.name} exited non-zero.{detail}"
    assert "ALL CHECKS PASSED" in result.stdout, (
        f"{SELFTEST_SH.name} did not report ALL CHECKS PASSED.{detail}"
    )


# ---------------------------------------------------------------------------
# step-9: the primary anti-rot gate
# ---------------------------------------------------------------------------

def test_shell_selftest_is_executable() -> None:
    """Expected GREEN on arrival -- a regression pin.

    The .sh is mode 100755 today and this module drives it through `bash` for
    portability, so the bit is not strictly load-bearing HERE -- but the .sh
    documents itself as directly runnable by an operator
    ("Usage: scripts/tests/test_remove_lms_dropin.sh"), and a lost +x silently
    breaks that contract.  Mirrors
    test_reclaim_orphaned_worktrees_wrapper.py::test_wrapper_is_executable.
    """
    assert SELFTEST_SH.is_file(), f"{SELFTEST_SH} is missing"
    assert os.access(SELFTEST_SH, os.X_OK), (
        f"Expected {SELFTEST_SH} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SELFTEST_SH}"
    )


def test_selftest_template_seam_propagates(tmp_path: Path) -> None:
    """The LMS_SELFTEST_TEMPLATE seam must actually REACH the .sh's paths.

    Nothing else asserts this any more.  Until esc-4200-2 a broken seam would
    have surfaced INDIRECTLY, as test_concurrent_selftests_do_not_collide
    failing because all three runs silently fell back to the shared default;
    that test is gone, so without this one a dropped seam degrades SILENTLY to
    `lms-dropin-selftest@` -- re-opening the measured collision defect, and
    stranding that default unit in the live dir for that run: cleanup's
    _remove_template_residue still keys on the unique, hyphenated name, so it
    will not touch a default-named unit.  _prune_stale_selftest_units now
    also admits _DEFAULT_TEMPLATE, so the strand is bounded to
    _STALE_AFTER_S rather than permanent -- but that is an hour-later mop-up,
    not a substitute for the seam actually propagating.

    Driven through ``_run_selftest``, the SAME helper the real gate uses, so
    both halves of the seam are covered by one assertion.  An earlier version
    built its env dict inline and set LMS_SELFTEST_TEMPLATE itself, which
    covered only the .sh half: a misspelled or deleted key in _run_selftest
    left this test green, left test_shell_selftest_passes green solo (the .sh
    would just fall back to the shared default), and re-opened the measured
    collision in the fleet.  ``extra_env`` therefore varies only the
    environment AROUND the seam -- the unit dir and PATH -- and _run_selftest
    rejects any attempt to pass the seam variable itself.

    HOW, without systemd and without touching the real unit dir.  The seam is
    observable by making the .sh's very first write FAIL at a path only a
    propagated template can name: XDG_CONFIG_HOME is redirected at tmp_path
    and a regular FILE is planted where install_fixture will `mkdir -p` its
    drop-in dir.  If the seam propagates, mkdir refuses and bash prints the
    blocked path -- which contains our unique template -- to stderr.  If the
    seam were dropped, the .sh would build that path from the default template
    instead, find nothing in its way, and never mention our name at all.

    Asserted on the template NAME in stderr, not on the exit code: BOTH cases
    exit non-zero (a dropped seam would proceed and then fail its
    WorkingDirectory checks under the redirect, per design decision 2), so rc
    cannot discriminate and the name is the only signal that can.

    `systemctl` is shimmed to a no-op on PATH so the cleanup trap's
    `daemon-reload` costs nothing and contends with nothing -- this test must
    not add unserialized load to the one shared user manager the slot above
    exists to protect.  Same fake-systemctl-on-PATH idiom as
    scripts/tests/test_lms_ctl.py.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    shim = fake_bin / "systemctl"
    shim.write_text("#!/bin/sh\nexit 0\n")
    shim.chmod(0o755)

    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    template = _unique_template()
    # A regular file exactly where `mkdir -p "$DROPIN_DIR"` wants a directory.
    blocker = unit_dir / f"{template}.service.d"
    blocker.write_text("")

    result = _run_selftest(
        template,
        extra_env={
            "XDG_CONFIG_HOME": str(tmp_path),
            "PATH": os.pathsep.join([str(fake_bin), os.environ.get("PATH", "")]),
        },
        # This run aborts at its first mkdir, so it needs none of
        # _SELFTEST_TIMEOUT_S's budget -- and it holds no serialization slot,
        # so a wedge here must not sit on the suite's --timeout=300.
        timeout=60,
    )

    detail = (
        f"\n--- template: {template}\n"
        f"--- exit code: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert template in result.stderr, (
        f"{SELFTEST_SH.name} never named the template it was handed, so "
        "LMS_SELFTEST_TEMPLATE did not reach its UNIT/DROPIN_DIR paths.  The .sh "
        "has silently fallen back to the shared default 'lms-dropin-selftest@', "
        "which is NOT safe to run concurrently.  Check that line still reads "
        'TEMPLATE="${LMS_SELFTEST_TEMPLATE:-lms-dropin-selftest@}".' + detail
    )
    # The blocker is still a regular file: the run aborted AT the mkdir rather
    # than somehow proceeding past it, so the stderr above is the mkdir refusal
    # and not an unrelated message that happened to quote the template.
    assert blocker.is_file(), f"expected the planted blocker to survive.{detail}"


def test_shell_selftest_passes() -> None:
    """THE assertion this task exists to create.  Expected GREEN on arrival.

    There is no RED to manufacture here and the absence of one is the point:
    scripts/tests/test_remove_lms_dropin.sh was already correct -- 12/12
    checks -- and the defect was that NOTHING RAN IT.  It was the only
    .sh among 61 pytest modules in scripts/tests/, and pytest collects only
    `test_*.py`, so the file sat in the tree checked by nobody.  The fix is
    therefore the EXISTENCE of a collected caller, and this is it.  Same shape
    as tests/scripts/test_know_live_installed_unit_parity.py's
    "Expected GREEN on arrival -- a regression pin".

    What it protects: scripts/remove-lms-arm-worktree-dropin.sh runs
    UNSANDBOXED and UNATTENDED as a deterministic `before_done` deploy action.
    Its own TEST SEAMS comment says such a script "must not have its happy
    path first execute in production" -- a guarantee that, until this module,
    was delivered by an operator remembering to run a file rather than by
    anything mechanical.  From here the verify gate delivers it.

    Safety of running for real in the DEFAULT suite: the .sh only ever
    `daemon-reload`s -- it never starts, stops, enables or restarts any unit
    -- and it operates solely on a uniquely-named throwaway template sharing
    no name with any real unit.

    COST, measured on the operator host rather than estimated (an earlier
    "~2s" here was wrong by ~2.6x and is corrected under esc-4200-2): a solo
    run is 5.20s end to end, and it is essentially ALL `daemon-reload` --
    0.85-0.94s each against 66 unit files, ~5 of them per run.  Against a
    suite measured at 293-931s that is ~0.6-1.8%.  Because the reload is
    globally serialized inside the one shared user manager, that cost would
    otherwise scale LINEARLY with fleet concurrency; the slot below caps the
    host at one run at a time, so this stays 5.2s of exclusive work plus a
    bounded, skip-terminated wait instead.
    """
    _require_systemd_user_manager()
    _prune_stale_selftest_units(_unit_dir(), now=time.time())

    # The slot is acquired BEFORE the template is generated and released only
    # after the residue sweep, so the whole install/drive/assert/clean cycle --
    # not merely the subprocess -- is exclusive host-wide.
    with _serialized_selftest_slot():
        template = _unique_template()
        try:
            _assert_selftest_passed(_run_selftest(template), template)
        finally:
            _remove_template_residue(template)
