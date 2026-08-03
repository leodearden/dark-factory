"""Shipped-script contract for dark-factory's relocated warm-lane scripts.

Task 3072 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` leaf α, Phase 1).

dark-factory now ships its own copies of the seven project-agnostic warm-lane
scripts under ``orchestrator/scripts/warm-lane/`` so a project that does not
carry them still gets warm-lane GC, disk guarding and auditing.  These tests
pin the *shipped* half of that contract — the files exist, are executable, and
are syntactically valid — independently of the *resolution* half
(``test_warm_lane_script_resolution.py``), which pins how ``GitOps`` chooses
between a project override and these copies.

The sibling-wiring class is the load-bearing one.  Three of the seven source a
lib that is not itself one of the seven, so a seven-file-only relocation would
ship three scripts that cannot execute:

* ``warm-lane-gc.sh`` and ``warm-lane-gc-sweep.sh`` source
  ``$SCRIPT_DIR/lib_live_refs.sh`` and deliberately ``exit 2`` when it is
  absent (reify task 5572 made that fail-loud precisely so a silently-missing
  liveness guard cannot recur).
* ``warm-lane-gc.sh`` sources a SECOND lib since dark-factory task 3075 (leaf
  γ): ``$SCRIPT_DIR/lib_lane_state.sh``, behind the same ``exit 2`` guard, so
  it now needs BOTH libs to run at all.  Pass-1 reclaimability is decided from
  the durable ``.lane-state`` record, and a silently-absent reader would
  degrade reclaim back to the ``FREE ≈ flock-free`` approximation γ removed.
* ``warm-lane-audit.sh`` sources TWO: ``$SCRIPT_DIR/lib_portable.sh``, and —
  since dark-factory task 3074 (leaf β) — ``$SCRIPT_DIR/lib_lane_state.sh``.
  Both now sit behind the same ``exit 2`` guard shape copied from
  ``warm-lane-gc.sh``'s; ``lib_portable.sh``'s was added by task 3370 (README
  "Delta 8") and is ordered FIRST, so a copy carrying neither sibling reports
  it rather than the lane-state one.  Unlike the other two libs
  ``lib_lane_state.sh`` is dark-factory-NATIVE rather than a reify relocation:
  it holds the facts only dark-factory owns (the ``.lane-state`` record format
  and ``PROTECTED_PREFIXES``), which is why the audit's ``assigned`` column can
  no longer be produced without it.

Running each with ``--help`` from the new directory is the executable proof
that all three libs actually travelled along.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Resolved from THIS FILE (orchestrator/tests/ -> orchestrator/), never from the
# process CWD: the merge-verify harness runs pytest from the ``orchestrator/``
# cwd while a plain ``pytest orchestrator/tests`` runs from the repo root, and
# this contract must hold identically under both.
WARM_LANE_SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'warm-lane'

#: Resolved ONCE, absolutely, so a case may hand a script a hostile ``PATH``
#: without also hiding the interpreter this harness needs to launch it.  Same
#: discipline as ``test_lane_state_lib.py``'s module-scope ``_BASH``.
_BASH = shutil.which('bash') or '/bin/bash'

#: The seven project-agnostic scripts relocated by this leaf.  ``seed-warm-lane.sh``
#: and ``refresh-warm-base.sh`` are deliberately NOT here — PRD §5 keeps those
#: project-owned primitives in the consuming project.
RELOCATED_SCRIPTS = (
    'warm-lane-gc.sh',
    'warm-lane-gc-sweep.sh',
    'thin-warm-lane.sh',
    'warm-lane-disk-guard.sh',
    'warm-lane-audit.sh',
    'warm-lane-degenerate-ref-check.sh',
    'provision-warm-lane-fs.sh',
)

#: Sourced siblings the seven cannot run without (see module docstring).
SOURCED_LIBS = (
    'lib_live_refs.sh',
    'lib_portable.sh',
    # dark-factory-native, added by task 3074 (leaf β) — not a reify relocation.
    # In SOURCED_LIBS rather than a parallel test class so it inherits the
    # exists / owner-execute-bit / `bash -n` coverage already written here.
    'lib_lane_state.sh',
)

ALL_SHIPPED = RELOCATED_SCRIPTS + SOURCED_LIBS


class TestRelocatedScriptsAreShipped:
    """Every relocated file is present, executable and syntactically valid."""

    @pytest.mark.parametrize('name', ALL_SHIPPED)
    def test_file_exists(self, name: str) -> None:
        path = WARM_LANE_SCRIPT_DIR / name
        assert path.is_file(), (
            f'{name} is not shipped at {path} — dark-factory cannot fall back to '
            f'its own copy for a project that lacks a scripts/{name} override'
        )

    @pytest.mark.parametrize('name', ALL_SHIPPED)
    def test_owner_execute_bit_is_set(self, name: str) -> None:
        path = WARM_LANE_SCRIPT_DIR / name
        assert path.is_file(), f'{name} is not shipped at {path}'
        mode = path.stat().st_mode
        assert mode & 0o100, (
            f'{name} is not owner-executable (mode={mode & 0o777:04o}); '
            f'GitOps spawns these directly, not via `bash <path>`'
        )

    @pytest.mark.parametrize('name', ALL_SHIPPED)
    def test_passes_bash_syntax_check(self, name: str) -> None:
        path = WARM_LANE_SCRIPT_DIR / name
        assert path.is_file(), f'{name} is not shipped at {path}'
        proc = subprocess.run(
            ['bash', '-n', str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f'bash -n rejected {name}: {proc.stderr.strip()!r}'
        )


class TestSiblingLibsTravelledWithTheScripts:
    """The three lib-sourcing scripts actually run from the new directory.

    Each is invoked with ``--help`` (read-only, no mount, no subprocess side
    effects) from ``orchestrator/scripts/warm-lane/``.  Since task 3370 closed
    the last gap there is ONE shape for a missing sibling, and it is what these
    cases assert against: the script's own fail-loud wiring message + ``exit
    2``, for all three libs.
    """

    #: Exit 2 is the wiring/usage sentinel both gc scripts use for
    #: "incomplete deployment" (deliberately NOT 1, which means runtime error).
    WIRING_EXIT = 2

    #: The verbatim fail-loud fragments emitted by the sourcing scripts' guards.
    FAIL_LOUD_FRAGMENTS = (
        'lib_live_refs.sh not found next to',
        # Task 3370 gave lib_portable.sh an explicit guard, so bash's bare
        # ``source`` shape ('lib_portable.sh: No such file') is no longer
        # reachable for the only script that sources it — warm-lane-audit.sh,
        # the sole consumer repo-wide — and the old fragment would now be a pin
        # that can never fire.  The bare shape stays covered generically by the
        # per-line scan below, which trips on any SOURCED_LIBS name appearing
        # with 'No such file' OR 'not found'.
        'lib_portable.sh not found next to',
        # The guard for dark-factory's own lane-state lib, shared VERBATIM by
        # warm-lane-audit.sh (task 3074, which established it) and
        # warm-lane-gc.sh (task 3075, which reused the exact string so this
        # tuple needed no amendment).  Both scripts are already in the
        # parametrize list below, so both are covered by this one entry.
        'lib_lane_state.sh not found next to',
    )

    @pytest.mark.parametrize(
        'name',
        ['warm-lane-gc.sh', 'warm-lane-gc-sweep.sh', 'warm-lane-audit.sh'],
    )
    def test_help_does_not_hit_a_sibling_wiring_error(self, name: str) -> None:
        script = WARM_LANE_SCRIPT_DIR / name
        assert script.is_file(), f'{name} is not shipped at {script}'
        proc = subprocess.run(
            [str(script), '--help'],
            cwd=str(WARM_LANE_SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = proc.stdout + proc.stderr

        for fragment in self.FAIL_LOUD_FRAGMENTS:
            assert fragment not in combined, (
                f'{name} --help hit a sibling-lib wiring error ({fragment!r}) — '
                f'the sourced lib did not travel with the relocation'
            )
        # Generic shape: no line may report either lib as missing/unfound.  Kept
        # per-line and lib-scoped rather than a blanket "No such file" scan
        # because warm-lane-gc.sh's usage text has a pre-existing, unrelated
        # stderr quirk that would false-trip a blanket check.
        for line in combined.splitlines():
            for lib in SOURCED_LIBS:
                if lib in line:
                    assert 'No such file' not in line and 'not found' not in line, (
                        f'{name} --help reported {lib} as missing: {line!r}'
                    )
        assert proc.returncode != self.WIRING_EXIT, (
            f'{name} --help exited {self.WIRING_EXIT} (the wiring/usage sentinel) — '
            f'stderr={proc.stderr.strip()!r}'
        )


class TestAuditFailsLoudOnAMissingLibPortable:
    """``warm-lane-audit.sh``'s two sibling guards, and the order they fire in.

    A sibling lib that did not travel is a WIRING failure — exit 2, not the 1
    bash's own bare ``source`` produces under ``set -e``.  ``lib_portable.sh``
    was the odd one out until task 3370 gave it a guard; README.md "Delta 8" is
    the single home for that rationale and for the base measurement that
    prompted it.  This class is the executable half.

    The two cases below are the two DIRECTIONS of one ordering contract, kept
    in one place so neither half can be broken without the other's reader
    noticing:

    * neither sibling present → the FIRST guard speaks (``lib_portable.sh``)
      and the lane-state message is ABSENT;
    * ``lib_portable.sh`` present, ``lib_lane_state.sh`` withheld → the second
      guard still fires and still names itself.

    The second case is what makes the first's negative assertion an *ordering*
    pin rather than a "there is only one guard left" state, which a deleted or
    silently-degraded lane-state guard would also satisfy.  It is not the
    lane-state guard's primary coverage: ``test_lane_state_lib.py::
    TestAuditReadsThroughTheLib::test_warm_lane_audit_fails_loud_when_the_lib_is_missing``
    (task 3074) pins that guard on its own terms and predates this class.
    """

    def test_a_copy_with_neither_sibling_exits_2_naming_lib_portable(
        self, tmp_path: Path,
    ) -> None:
        staged_dir = tmp_path / 'incomplete-deploy'
        staged_dir.mkdir()
        staged = staged_dir / 'warm-lane-audit.sh'
        staged.write_bytes((WARM_LANE_SCRIPT_DIR / 'warm-lane-audit.sh').read_bytes())
        staged.chmod(0o755)

        # Assert the fixture before asserting on it: if either sibling were
        # quietly present the guards would never fire and every assertion below
        # would pass for the wrong reason.
        for lib in ('lib_portable.sh', 'lib_lane_state.sh'):
            assert not (staged_dir / lib).exists(), (
                f'fixture is not an incomplete deployment: {lib} is present in '
                f'{staged_dir}, so this case would pass vacuously'
            )

        proc = subprocess.run(
            [_BASH, str(staged), '--help'],
            cwd=str(staged_dir),
            capture_output=True,
            text=True,
            timeout=60,
            env=_sanitized_env(),
        )
        combined = proc.stdout + proc.stderr

        # --help normally exits 0, so a 2 additionally proves the guard fires
        # BEFORE argv is parsed (the note warm-lane-gc.sh's Block A9 makes).
        assert proc.returncode == TestSiblingLibsTravelledWithTheScripts.WIRING_EXIT, (
            'an absent lib_portable.sh must be the wiring sentinel exit '
            f'{TestSiblingLibsTravelledWithTheScripts.WIRING_EXIT}, not a runtime '
            f'failure; rc={proc.returncode} stderr={proc.stderr!r}'
        )
        # Split around the message's U+2014 em dash so the pin does not depend
        # on that codepoint surviving an editor.
        assert 'warm-lane-audit.sh: ERROR' in proc.stderr, (
            'the failure must be attributed to the SCRIPT, not to bash; '
            f'stderr={proc.stderr!r}'
        )
        assert 'scripts/lib_portable.sh not found next to warm-lane-audit.sh' in proc.stderr, (
            f'the fail-loud message must name the missing sibling; {proc.stderr!r}'
        )
        assert 'lib_portable.sh: No such file' not in combined, (
            "bash's own bare-`source` failure shape is still reachable — the "
            f'guard is missing or sits AFTER the source.\noutput:\n{combined}'
        )
        # THE ORDERING PIN, negative half: neither sibling is present, so the
        # FIRST guard must be the one that speaks — warm-lane-gc.sh's
        # live-refs-before-lane-state rule.  Read with the positive half below,
        # which proves the lane-state guard still exists and still speaks.
        assert 'lib_lane_state.sh not found next to' not in proc.stderr, (
            'a copy carrying NEITHER sibling reported lib_lane_state.sh first — '
            'the lib_portable.sh guard must be ordered FIRST, above the source '
            f'it protects.\nstderr:\n{proc.stderr}'
        )

    def test_a_copy_with_only_lib_portable_still_reaches_the_lane_state_guard(
        self, tmp_path: Path,
    ) -> None:
        """The POSITIVE half of the ordering pin.

        Same fixture with ``lib_portable.sh`` restored: the first guard must now
        pass silently and control must reach the SECOND one, which names itself.
        Without this, the sibling case's negative assertion would be satisfied
        just as well by a lane-state guard that had been deleted outright or
        softened to a silent degrade.
        """
        staged_dir = tmp_path / 'partial-deploy'
        staged_dir.mkdir()
        staged = staged_dir / 'warm-lane-audit.sh'
        staged.write_bytes((WARM_LANE_SCRIPT_DIR / 'warm-lane-audit.sh').read_bytes())
        staged.chmod(0o755)
        # lib_portable.sh travels — it is the FIRST guard's subject, and this
        # case is about what happens once that guard is satisfied.
        (staged_dir / 'lib_portable.sh').write_bytes(
            (WARM_LANE_SCRIPT_DIR / 'lib_portable.sh').read_bytes(),
        )
        assert not (staged_dir / 'lib_lane_state.sh').exists(), (
            f'fixture is not a partial deployment: lib_lane_state.sh is present '
            f'in {staged_dir}, so this case would pass vacuously'
        )

        proc = subprocess.run(
            [_BASH, str(staged), '--help'],
            cwd=str(staged_dir),
            capture_output=True,
            text=True,
            timeout=60,
            env=_sanitized_env(),
        )

        assert proc.returncode == TestSiblingLibsTravelledWithTheScripts.WIRING_EXIT, (
            'an absent lib_lane_state.sh must be the wiring sentinel exit '
            f'{TestSiblingLibsTravelledWithTheScripts.WIRING_EXIT}; '
            f'rc={proc.returncode} stderr={proc.stderr!r}'
        )
        assert 'lib_lane_state.sh not found next to warm-lane-audit.sh' in proc.stderr, (
            'the SECOND guard must still fire and still name itself once the '
            f'first is satisfied; stderr={proc.stderr!r}'
        )
        # The first guard is satisfied, so it must say nothing at all.
        assert 'lib_portable.sh not found next to' not in proc.stderr, (
            'the lib_portable.sh guard fired even though the lib is present — '
            f'its `[ ! -f ]` test is inverted or mis-pathed.\nstderr:\n{proc.stderr}'
        )


# ---------------------------------------------------------------------------
# provision-warm-lane-fs.sh: repo-root parity at the new nesting depth
# ---------------------------------------------------------------------------

#: The relocation moves provision-warm-lane-fs.sh two levels deeper: reify's
#: ``<repo>/scripts/`` becomes ``<repo>/orchestrator/scripts/warm-lane/``.
_NEW_NESTING = ('orchestrator', 'scripts', 'warm-lane')


def _stage_provision_script(repo: Path, *, git_init: bool) -> Path:
    """Build ``<repo>/orchestrator/scripts/warm-lane/provision-warm-lane-fs.sh``.

    Copies the SHIPPED script (not a fixture of it) into a synthetic repo at
    the real post-relocation depth, so what is under test is the file that
    actually ships.  ``git_init`` makes the synthetic tree a real checkout or
    leaves it without git metadata at all; resolution must land on the same
    root either way, since it is pure path arithmetic (README Delta 1).
    """
    script_dir = repo.joinpath(*_NEW_NESTING)
    script_dir.mkdir(parents=True, exist_ok=True)
    staged = script_dir / 'provision-warm-lane-fs.sh'
    staged.write_bytes((WARM_LANE_SCRIPT_DIR / 'provision-warm-lane-fs.sh').read_bytes())
    staged.chmod(0o755)
    if git_init:
        subprocess.run(
            ['git', 'init', '-q', '-b', 'main'],
            cwd=repo,
            check=True,
            timeout=60,
            env=_sanitized_env(),
        )
    return staged


#: Environment keys that must never leak into a repo-root resolution test.
#: ``REIFY_WARM_LANE_MOUNT`` would override the printed default outright.  The
#: ``GIT_*`` pair is the subtler hazard: pytest's ``--basetemp`` may legally
#: point INSIDE a checkout (an untracked ``.pytest-tmp/`` at this repo's root is
#: evidence that happens), and this suite runs under git hooks and
#: ``git rebase --exec``, which export ``GIT_DIR``.  Any resolution strategy
#: that consulted git would then silently answer from the AMBIENT repo instead
#: of the synthetic one, and the test would report coverage it does not have.
#: Stripping them makes the synthetic repo the only repo in scope.
#:
#: ``REIFY_WARM_LANE_IACT_PREFIX`` joined the list with task 3074's amendment
#: pass: it renames the interactive band ``lane_protect_glob`` renders, so an
#: ambient value would silently change the glob that
#: ``test_lane_state_lib.py``'s bridge cases compare against the value python
#: computes in-process — a green result that only means both sides read the same
#: stray environment.  Stripped here, in the ONE sanitizer both files share,
#: rather than per-case.
_HOSTILE_ENV_KEYS = (
    'REIFY_WARM_LANE_MOUNT',
    'REIFY_WARM_LANE_IACT_PREFIX',
    'GIT_DIR',
    'GIT_WORK_TREE',
)


def _sanitized_env(
    *, ceiling: Path | None = None, extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """A copy of ``os.environ`` with ``_HOSTILE_ENV_KEYS`` removed.

    Used for BOTH the staging ``git init`` and the ``--help`` run.  The staging
    call needs it too: under an inherited ``GIT_DIR`` (git hook,
    ``git rebase --exec``) a bare ``git init`` fails outright with
    ``could not set 'core.repositoryformatversion'``, so without the strip this
    whole class errors out in exactly the environment the merge-verify harness
    may run it in.
    """
    env = {k: v for k, v in os.environ.items() if k not in _HOSTILE_ENV_KEYS}
    if ceiling is not None:
        env['GIT_CEILING_DIRECTORIES'] = str(ceiling)
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# hostile-PATH / decoy-CWD harness (task 3279)
# ---------------------------------------------------------------------------

#: The single assertion vocabulary for "the script reached a file it picked up
#: off the CALLER'S CWD instead of the one next to itself".  Defined once
#: because three separate cases below assert on its absence.
_DECOY_MARKER = 'DECOY'


def _path_hiding(tmp_path: Path, *names: str) -> Path:
    """A stub bin dir that HIDES *names*, to be PREPENDED to ``PATH``.

    Each named binary gets a ``#!/bin/sh`` / ``exit 127`` shim.  A shim that
    exits 127 printing nothing is observationally IDENTICAL to
    ``command not found`` — empty command substitution, status 127 — so
    ``$(dirname ...)`` yields the empty string exactly as it does on a host
    whose ``PATH`` genuinely lacks the binary.

    PREPEND, do not replace.  ``test_lane_state_lib.py``'s
    ``_stub_path_dir(tmp_path, 'nothing-here')`` idiom — an EMPTY directory
    used as the whole ``PATH`` — works there only because *sourcing* a lib
    forks nothing.  It cannot drive these executable scripts: MEASURED,
    ``provision-warm-lane-fs.sh --help`` under an emptied ``PATH`` exits 127
    having printed nothing at all, because its usage heredoc forks ``cat``.  A
    test built on that would fail for a reason unrelated to the defect under
    test — and would go on "failing" after the fix, i.e. it would be a doomed
    RED.  Leaving the rest of ``PATH`` intact lets the script still reach
    ``cat``, ``realpath``, ``git``: everything except the one binary whose
    absence is the subject.
    """
    stub = tmp_path / ('stub-bin-hiding-' + '-'.join(names))
    stub.mkdir(parents=True, exist_ok=True)
    for name in names:
        shim = stub / name
        shim.write_text('#!/bin/sh\nexit 127\n')
        shim.chmod(0o755)
    return stub


def _decoy_dir(tmp_path: Path, *names: str) -> Path:
    """A directory holding same-named decoys for *names*, fit to be the ``cwd``.

    Each decoy announces itself with ``_DECOY_MARKER`` on stderr and does
    nothing else, so it is detectable whether it was *sourced* or *executed*.

    Handed to a subprocess as its ``cwd`` this stands in for the realistic,
    NON-adversarial trigger: invoking a warm-lane script from reify's own
    ``scripts/`` dir, or from another dark-factory checkout's ``warm-lane/``
    dir — both carry precisely these filenames.
    """
    stem = '-'.join(name.rsplit('.', 1)[0] for name in names)
    decoys = tmp_path / f'decoy-cwd-{stem}'
    decoys.mkdir(parents=True, exist_ok=True)
    for name in names:
        decoy = decoys / name
        decoy.write_text(
            '#!/usr/bin/env bash\n'
            f'echo "{_DECOY_MARKER} {name} SOURCED FROM CWD" >&2\n'
        )
        decoy.chmod(0o755)
    return decoys


def _default_mount_in_usage(
    staged: Path,
    *,
    ceiling: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Run the staged script with ``--help`` and return its usage text.

    Args:
        staged: The staged copy of the shipped script.
        ceiling: Sets ``GIT_CEILING_DIRECTORIES`` so that no git invocation
            could ascend out of the synthetic tree even if one were
            reintroduced — the hermeticity backstop for the no-git case.
        extra_env: Keys applied AFTER the hostile-key strip, for the cases that
            deliberately inject a hostile environment.
    """
    proc = subprocess.run(
        [str(staged), '--help'],
        capture_output=True,
        text=True,
        timeout=60,
        env=_sanitized_env(ceiling=ceiling, extra=extra_env),
        cwd=staged.parent,
    )
    return proc.stdout + proc.stderr


#: The advertised default mount, as rendered into the usage text:
#: ``--mount DIR     Mount point (default: ${REIFY_WARM_LANE_MOUNT:-<path>})``.
#: The ``${...:-}`` wrapper is literal in the heredoc (escaped there), so only
#: the interpolated ``_default_mount`` result varies.
_ADVERTISED_MOUNT_RE = re.compile(
    r'Mount point \(default: \$\{REIFY_WARM_LANE_MOUNT:-(?P<mount>[^}]*)\}\)',
)


def _advertised_default_mount(usage: str) -> str:
    """The default ``--mount`` path parsed out of a captured usage text."""
    match = _ADVERTISED_MOUNT_RE.search(usage)
    assert match is not None, (
        'the usage text carries no "--mount DIR ... (default: ...)" line at all '
        f'— the script did not get as far as printing usage.\nusage:\n{usage}'
    )
    return match.group('mount')


class TestProvisionRepoRootParity:
    """The default ``--mount`` is derived from the REPO ROOT, not the script's parent.

    ``provision-warm-lane-fs.sh`` computes ``REPO_ROOT`` from its own location
    and ``_default_mount()`` hangs the operator-facing default mount off it.
    In reify the script sits at ``<repo>/scripts/``, so one ``..`` reached the
    repo root.  Here it sits two levels deeper, at
    ``<repo>/orchestrator/scripts/warm-lane/``.

    A LITERAL byte-copy therefore BREAKS parity rather than preserving it: the
    inherited ``$_SCRIPT_DIR/..`` lands on ``<repo>/orchestrator/scripts`` and
    the advertised default mount silently becomes
    ``<repo>/orchestrator/warm-lanes`` instead of the repo's sibling
    ``warm-lanes`` dir — a wrong path printed to an operator about to
    provision a multi-terabyte volume.  Restoring repo-root resolution is a
    requirement of behaviour parity, not an exception to it.

    Hermetic: a synthetic repo under ``tmp_path``, ``--help`` only — no
    loopback image, no mount, nothing privileged — and every environment key
    that could redirect resolution stripped (``_HOSTILE_ENV_KEYS``).  The strip
    matters: ``--basetemp`` may legally sit inside a checkout and this suite
    runs under git hooks that export ``GIT_DIR``, so without it a passing test
    could not distinguish "resolution is correct" from "the ambient repo
    happened to be the right answer".
    """

    def test_default_mount_is_derived_from_the_repo_root(self, tmp_path: Path) -> None:
        """A checkout at ``<tmp>/repo`` → ``<tmp>/warm-lanes``."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        usage = _default_mount_in_usage(
            _stage_provision_script(repo, git_init=True), ceiling=tmp_path,
        )

        assert str(tmp_path / 'warm-lanes') in usage, (
            f'The advertised default mount must hang off the repo root '
            f'({repo}), giving {tmp_path / "warm-lanes"}.\nusage:\n{usage}'
        )
        assert str(repo / 'orchestrator' / 'warm-lanes') not in usage, (
            'The default mount was derived from the script\'s grandparent '
            '(<repo>/orchestrator/scripts) rather than the repo root — the '
            'inherited "$_SCRIPT_DIR/.." does not survive the two-levels-deeper '
            f'relocation.\nusage:\n{usage}'
        )

    def test_repo_root_resolves_without_git_metadata(self, tmp_path: Path) -> None:
        """No git metadata → resolution still lands on the repo root.

        The script must not depend on being inside a checkout: it is run on a
        fresh host to provision the pool substrate, sometimes from an unpacked
        tree, and ``git`` may be absent entirely.

        ``GIT_CEILING_DIRECTORIES`` pins the case rather than trusting
        ``tmp_path`` to sit outside every checkout — under
        ``--basetemp=<inside a repo>`` an ascending git probe would otherwise
        find the ENCLOSING repo, and the test would silently stop exercising the
        no-git path it is named for.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        usage = _default_mount_in_usage(
            _stage_provision_script(repo, git_init=False), ceiling=tmp_path,
        )

        assert str(tmp_path / 'warm-lanes') in usage, (
            f'With no git metadata resolution must still reach the repo root '
            f'({repo}), giving {tmp_path / "warm-lanes"}.\nusage:\n{usage}'
        )
        assert str(repo / 'orchestrator' / 'warm-lanes') not in usage, (
            f'Resolution landed on the script\'s grandparent instead of '
            f'the repo root.\nusage:\n{usage}'
        )

    def test_repo_root_ignores_an_inherited_git_dir(self, tmp_path: Path) -> None:
        """``GIT_DIR`` in the environment must not redirect the advertised mount.

        A git hook, ``git rebase --exec`` and ``filter-branch`` all export
        ``GIT_DIR`` with no ``GIT_WORK_TREE``.  Under that environment a
        ``git rev-parse --show-toplevel`` probe returns the CWD's own directory
        — here ``<repo>/orchestrator/scripts/warm-lane`` — advertising
        ``<repo>/orchestrator/scripts/warm-lanes``, i.e. worse than no
        resolution at all, and unguardable because that path exists.  Pure path
        arithmetic reads no environment, and this pins that it stays that way.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        staged = _stage_provision_script(repo, git_init=True)
        usage = _default_mount_in_usage(
            staged, ceiling=tmp_path, extra_env={'GIT_DIR': str(repo / '.git')},
        )

        assert str(tmp_path / 'warm-lanes') in usage, (
            f'An inherited GIT_DIR redirected repo-root resolution; the default '
            f'mount must stay {tmp_path / "warm-lanes"}.\nusage:\n{usage}'
        )
        assert str(staged.parent.parent) + '/warm-lanes' not in usage, (
            'Resolution answered from the inherited GIT_DIR rather than the '
            f'script\'s own location.\nusage:\n{usage}'
        )

    def test_repo_root_ignores_an_enclosing_checkout(self, tmp_path: Path) -> None:
        """A tree nested inside an unrelated outer repo resolves to ITS OWN root.

        An unpacked tree can legitimately sit inside some other checkout (a
        dotfiles ``$HOME``, ``/opt/config``).  Anything that ascends parents
        looking for git metadata would answer with the OUTER root; the script's
        depth below its own root is what is actually fixed and known.
        """
        outer = tmp_path / 'outer'
        repo = outer / 'nested' / 'repo'
        repo.mkdir(parents=True)
        subprocess.run(
            ['git', 'init', '-q', '-b', 'main'],
            cwd=outer,
            check=True,
            timeout=60,
            env=_sanitized_env(),
        )
        usage = _default_mount_in_usage(
            _stage_provision_script(repo, git_init=False), ceiling=tmp_path,
        )

        assert str(outer / 'nested' / 'warm-lanes') in usage, (
            f'The tree must resolve its OWN root ({repo}), giving '
            f'{outer / "nested" / "warm-lanes"}, not the enclosing checkout '
            f'({outer}).\nusage:\n{usage}'
        )
        assert str(tmp_path / 'warm-lanes') not in usage, (
            f'Resolution ascended into the enclosing checkout at {outer}.\n'
            f'usage:\n{usage}'
        )

    @pytest.mark.parametrize('worktrees_dir', ['worktrees', '.worktrees'])
    def test_ascend_past_worktrees_is_preserved(
        self, tmp_path: Path, worktrees_dir: str,
    ) -> None:
        """A checkout inside a worktrees dir surfaces the mount one level higher.

        Mirror case for the pre-existing ``_default_mount`` behaviour: the
        warm-lanes dir must live BESIDE the worktrees tree, never inside a
        worktree.  Pinned alongside the depth fix because both are consumers of
        ``REPO_ROOT`` — a fix that got the root right but broke the ascend
        would be just as much a parity break.

        Both spellings are exercised.  reify's copy matched only the literal
        ``worktrees``, but dark-factory's own worktree dir is ``.worktrees``
        (``GitConfig.worktree_dir`` default) — the shape an agent or operator
        in THIS repo actually runs from, and the one the relocation makes newly
        reachable (README Delta 2).
        """
        repo = tmp_path / worktrees_dir / 'repo'
        repo.mkdir(parents=True)
        usage = _default_mount_in_usage(
            _stage_provision_script(repo, git_init=True), ceiling=tmp_path,
        )

        assert str(tmp_path / 'warm-lanes') in usage, (
            f'A repo inside {worktrees_dir}/ must surface the mount beside the '
            f'worktrees tree ({tmp_path / "warm-lanes"}).\nusage:\n{usage}'
        )
        assert str(tmp_path / worktrees_dir / 'warm-lanes') not in usage, (
            f'The ascend-past-{worktrees_dir} behaviour is missing — the '
            f'warm-lanes dir would live inside the worktrees tree.\n'
            f'usage:\n{usage}'
        )

    @pytest.mark.parametrize(
        'nesting',
        [('repo',), ('worktrees', 'repo'), ('.worktrees', 'repo')],
        ids=['plain', 'worktrees', 'dot-worktrees'],
    )
    def test_default_mount_survives_a_path_without_dirname(
        self, tmp_path: Path, nesting: tuple[str, ...],
    ) -> None:
        """A ``PATH`` lacking ``dirname`` must not change the advertised mount.

        See README.md "Delta 7" for the failure mechanism and the base
        measurement (the mount silently became the bare filesystem root).

        Asserted as PARITY against a same-run control rather than a literal
        expected path: the synthetic repo lives under ``tmp_path``, so a
        hardcoded mount would encode the harness's layout and rot the moment
        ``_stage_provision_script`` changes.  The second assertion pins the
        specific regression, so the case cannot pass by both sides being
        equally broken.

        All three nestings are exercised because ``_default_mount()`` reaches a
        DIFFERENT derivation in each: the plain repo uses only the parent-of
        ``REPO_ROOT`` at the top, while a repo inside a worktrees dir also
        takes the ascend branch's second one.

        No assertion on the ``Usage:`` program-name line: ``basename "$0"``
        there is deliberately left forking, so that line renders blank under
        this ``PATH`` by design.
        """
        repo = tmp_path.joinpath(*nesting)
        repo.mkdir(parents=True)
        staged = _stage_provision_script(repo, git_init=False)

        control = _advertised_default_mount(
            _default_mount_in_usage(staged, ceiling=repo),
        )
        hidden = _advertised_default_mount(_default_mount_in_usage(
            staged,
            ceiling=repo,
            extra_env={
                'PATH': (
                    f'{_path_hiding(tmp_path, "dirname", "basename")}'
                    f'{os.pathsep}{os.environ["PATH"]}'
                ),
            },
        ))

        assert hidden == control, (
            f'The advertised default mount depends on PATH: with dirname/'
            f'basename present it is {control!r}, with them hidden it is '
            f'{hidden!r}.  The resolution must be pure parameter expansion '
            f'over builtins, which reads no PATH at all.'
        )
        assert hidden != '/warm-lanes', (
            'The default mount resolved to the bare filesystem root — the '
            'empty-substitution failure this case exists to catch: '
            '`$(dirname ...)` returned nothing, `cd ""` succeeded as a no-op, '
            'and the arithmetic ascended from the caller\'s CWD instead of '
            'the script\'s own location.'
        )


class TestSiblingResolutionIgnoresTheCallersCwd:
    """A script sources the siblings next to ITSELF, never next to the caller.

    ``TestSiblingLibsTravelledWithTheScripts`` above pins that the libs
    travelled with the relocation.  This class pins the other half: that the
    right COPY of them is the one loaded.

    See README.md "Delta 7" — "The corrected hypothesis" — for the mechanism,
    the per-script base measurements, and why these scripts' existing
    ``[ ! -f "$SCRIPT_DIR/lib_*.sh" ]`` guards cannot detect this class.

    Deliberately NO exit-code assertion: the decoys make the run SUCCEED, and
    the defect is *which files were sourced*, not how the process ended.  The
    exit code here is a function of the caller's CWD rather than of the defect
    — which is exactly why the filed expectation of ``exit 2`` was wrong.
    """

    #: Positive control for the two absence-only cases below.  All three
    #: scripts print a ``Usage:`` heredoc on ``--help``, so its presence proves
    #: the run actually REACHED the sibling-resolution code.  Without it a
    #: script that died at line 1 for an unrelated reason (a future ``set -u``
    #: on an unset var, a bad shebang, a binary shimmed away that is needed
    #: earlier) satisfies "the decoy marker is absent" vacuously, and the case
    #: goes green having proved nothing.  Same standard as
    #: ``TestThinSelfClobberGuardDoesNotDependOnPath``'s three-assertion shape.
    USAGE_MARKER = 'Usage:'

    @staticmethod
    def _help_under_a_hidden_dirname(
        name: str, *, cwd: Path, tmp_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``<name> --help`` from *cwd* with ``dirname`` hidden from ``PATH``."""
        script = WARM_LANE_SCRIPT_DIR / name
        assert script.is_file(), f'{name} is not shipped at {script}'
        return subprocess.run(
            [_BASH, str(script), '--help'],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60,
            env=_sanitized_env(extra={
                'PATH': (
                    f'{_path_hiding(tmp_path, "dirname")}'
                    f'{os.pathsep}{os.environ["PATH"]}'
                ),
            }),
        )

    @pytest.mark.parametrize(
        'name',
        ['warm-lane-gc.sh', 'warm-lane-gc-sweep.sh', 'warm-lane-audit.sh'],
    )
    def test_help_does_not_source_a_lib_from_the_callers_cwd(
        self, name: str, tmp_path: Path,
    ) -> None:
        proc = self._help_under_a_hidden_dirname(
            name, cwd=_decoy_dir(tmp_path, *SOURCED_LIBS), tmp_path=tmp_path,
        )
        combined = proc.stdout + proc.stderr

        assert self.USAGE_MARKER in combined, (
            f'{name} --help never printed its usage text, so the absence '
            f'assertion below would hold vacuously — this run proves nothing '
            f'about sibling resolution.\noutput:\n{combined}'
        )
        assert _DECOY_MARKER not in combined, (
            f'{name} --help sourced a sibling lib from the CALLER\'S CWD '
            f'instead of from {WARM_LANE_SCRIPT_DIR}.  Its own directory must '
            f'resolve by parameter expansion, which reads no PATH.\n'
            f'output:\n{combined}'
        )

    def test_a_bare_hostile_cwd_does_not_reach_lib_portable_from_the_cwd(
        self, tmp_path: Path,
    ) -> None:
        """``warm-lane-audit.sh`` resolves its libs beside ITSELF from an EMPTY CWD.

        The audit-specific half: with nothing to pick up in the CWD there is no
        decoy to detect, so the assertion is that no resolved path lies under
        the caller's CWD at all.

        Asserted on the RESOLVED PATH, not the exit code, for the reason in the
        class docstring.  README.md "Delta 7" records what the exit code
        actually was on base and why it did not match the filed expectation;
        the unguarded-``lib_portable.sh`` gap it names was closed separately by
        task 3370 (README "Delta 8").  That does not change what this case
        asserts: the exit code here is a function of the CALLER'S CWD rather
        than of the defect, which is why the resolved path remains the subject.
        """
        proc = self._help_under_a_hidden_dirname(
            'warm-lane-audit.sh', cwd=tmp_path, tmp_path=tmp_path,
        )
        combined = proc.stdout + proc.stderr

        assert self.USAGE_MARKER in combined, (
            'warm-lane-audit.sh --help never printed its usage text, so the '
            'absence assertion below would hold vacuously — this run proves '
            f'nothing about sibling resolution.\noutput:\n{combined}'
        )
        assert str(tmp_path) not in combined, (
            f'warm-lane-audit.sh resolved a sibling lib under the caller\'s '
            f'CWD ({tmp_path}) rather than under {WARM_LANE_SCRIPT_DIR}.\n'
            f'output:\n{combined}'
        )

    def test_reseed_does_not_execute_a_seed_script_from_the_callers_cwd(
        self, tmp_path: Path,
    ) -> None:
        """``thin-warm-lane.sh --reseed`` EXECUTES what it resolves.

        The other three scripts here merely *source* the mis-resolved path;
        this one runs it as a program, with the lane dir as an argument — on
        base it ran the decoy and reported ``[ok] Re-seeded``.

        See README.md "Sibling-seed defaults, and who resolves them" (reached
        from "Delta 3") for the reachability caveat: ``_script_dir`` is
        computed only when ``--seed-script`` was not passed, and dark-factory's
        own caller never passes ``--reseed``.  That is what makes this
        low-priority rather than urgent — and the warning recorded there for a
        future caller is what this closes properly.
        """
        lane = tmp_path / 'lane'
        (lane / 'target').mkdir(parents=True)
        base = tmp_path / 'base-target'
        base.mkdir()

        proc = subprocess.run(
            [
                _BASH, str(WARM_LANE_SCRIPT_DIR / 'thin-warm-lane.sh'),
                str(lane), '--reseed', '--base', str(base),
            ],
            cwd=str(_decoy_dir(tmp_path, 'seed-warm-lane.sh')),
            capture_output=True,
            text=True,
            timeout=60,
            env=_sanitized_env(extra={
                'PATH': (
                    f'{_path_hiding(tmp_path, "dirname")}'
                    f'{os.pathsep}{os.environ["PATH"]}'
                ),
            }),
        )
        combined = proc.stdout + proc.stderr

        assert _DECOY_MARKER not in combined, (
            'thin-warm-lane.sh --reseed EXECUTED a seed script from the '
            f'CALLER\'S CWD instead of resolving one beside itself in '
            f'{WARM_LANE_SCRIPT_DIR}.\noutput:\n{combined}'
        )


class TestThinSelfClobberGuardDoesNotDependOnPath:
    """``thin-warm-lane.sh``'s self-clobber guard must fire without ``basename``.

    A DIFFERENT failure mechanism from every case above, which is why this is
    its own class rather than one more parametrization of
    ``TestSiblingResolutionIgnoresTheCallersCwd``: a failed substitution inside
    ``[ ... ]`` compares FALSE and execution continues, where the same failure
    in an assignment propagates 127 and ``set -e`` aborts.  Same missing
    binary, opposite blast radius.  See README.md "Delta 7" — "The ``[ ... ]``
    vs assignment asymmetry" — for the rule and the measurements.

    The site is ``thin-warm-lane.sh``'s self-clobber guard, the ONLY half that
    fires when ``REIFY_WARM_LANE_MOUNT`` is unset (the two mount-relative
    checks above it are inside ``if [ -n "${REIFY_WARM_LANE_MOUNT:-}" ]``), 33
    lines above ``rm -rf "$LANE_DIR/target"``.

    ``REIFY_WARM_LANE_MOUNT`` is deliberately left UNSET here (``_sanitized_env``
    strips it) — that is the branch under test.  Setting it would let the
    mount-relative checks reach ``_self_clobber=1`` on their own and mask the
    defect entirely.
    """

    @staticmethod
    def _thin_without_basename(
        lane: Path, *, tmp_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``thin-warm-lane.sh <lane>`` with ``basename`` hidden from ``PATH``."""
        script = WARM_LANE_SCRIPT_DIR / 'thin-warm-lane.sh'
        assert script.is_file(), f'thin-warm-lane.sh is not shipped at {script}'
        return subprocess.run(
            [_BASH, str(script), str(lane)],
            capture_output=True,
            text=True,
            timeout=60,
            env=_sanitized_env(extra={
                'PATH': (
                    f'{_path_hiding(tmp_path, "basename")}'
                    f'{os.pathsep}{os.environ["PATH"]}'
                ),
            }),
        )

    def test_self_clobber_guard_survives_a_path_without_basename(
        self, tmp_path: Path,
    ) -> None:
        """A lane literally named ``base`` is refused, ``basename`` or no ``basename``.

        Three assertions, because any ONE of them can pass for the wrong
        reason: a script that died early for an unrelated reason satisfies (a)
        and (b) while proving nothing about the guard, and a script that
        printed the refusal after already deleting would satisfy (b) and (c).
        Together they pin the safety property, the failure, and the DIAGNOSIS.
        """
        pool = tmp_path / 'pool'
        sentinel = pool / 'base' / 'target' / 'seed-source-file'
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text('the pool seed source\n')

        proc = self._thin_without_basename(pool / 'base', tmp_path=tmp_path)
        combined = proc.stdout + proc.stderr

        # (a) the actual safety property.
        assert sentinel.is_file(), (
            'thin-warm-lane.sh DELETED the pool seed source under a PATH '
            'without `basename`.  If the guard has gone back to comparing a '
            '`$(basename ...)` substitution, that is why: a 127 substitution '
            'inside `[ ... ]` yields the empty string WITHOUT tripping '
            '`set -e`, so the guard silently compares false (README.md '
            f'"Delta 7").\noutput:\n{combined}'
        )
        # (b) it must FAIL, not succeed-having-done-nothing.
        assert proc.returncode != 0, (
            'thin-warm-lane.sh reported SUCCESS for a lane named `base`.\n'
            f'output:\n{combined}'
        )
        # (c) and fail for THIS reason.
        assert 'refusing to thin' in combined, (
            'thin-warm-lane.sh failed for some reason other than the '
            f'self-clobber guard.\noutput:\n{combined}'
        )

    def test_a_normally_named_lane_still_thins_without_basename(
        self, tmp_path: Path,
    ) -> None:
        """No-regression companion: the guard must not start refusing everything.

        A guard that fires on every input satisfies the case above while
        breaking the script, so the fix has to be shown to be a CORRECTION of
        the comparison rather than a widening of it.
        """
        pool = tmp_path / 'pool'
        lane = pool / '_lane-1'
        (lane / 'target').mkdir(parents=True)
        (lane / 'target' / 'checkout-file').write_text('reclaimable\n')

        proc = self._thin_without_basename(lane, tmp_path=tmp_path)
        combined = proc.stdout + proc.stderr

        assert proc.returncode == 0, (
            'thin-warm-lane.sh refused a normally-named lane under a PATH '
            'without `basename` — the self-clobber guard must be corrected, '
            f'not widened.\noutput:\n{combined}'
        )
        assert not (lane / 'target').exists(), (
            'thin-warm-lane.sh reported success but left `target/` in place.\n'
            f'output:\n{combined}'
        )


def _warm_lane_pool(tmp_path: Path) -> Path:
    """A minimal pool fixture: ``<tmp>/pool/worktrees`` + a resolvable base target.

    Returns the ``worktrees`` dir, i.e. the value to pass as ``--mount``.

    Holds ONE entry, ``_merge-x``, which is both protected (it matches
    ``warm-lane-gc.sh``'s default ``--protect-glob``) and a real git worktree
    (so ``warm-lane-audit.sh``'s ``_is_git_worktree`` gate admits it and the
    resident walk actually runs).  ``<pool>/base/target`` is a real symlink
    because ``reclaim`` resolves it with ``readlink -f`` and returns early if
    that fails, which would stop the run before the classification loop.
    """
    pool = tmp_path / 'pool'
    worktrees = pool / 'worktrees'
    merge = worktrees / '_merge-x'
    merge.mkdir(parents=True)
    (pool / 'base' / 'gen-1').mkdir(parents=True)
    (pool / 'base' / 'target').symlink_to('gen-1')
    subprocess.run(
        ['git', 'init', '-q', '-b', 'main'],
        cwd=merge, check=True, timeout=60, env=_sanitized_env(),
    )
    return worktrees


class TestLeafExtractionBehaviourWithoutBasename:
    """The two converted LEAF extractions do their real work without ``basename``.

    ``test_only_cosmetic_program_name_forks_remain`` is a static spelling gate:
    it catches a fork reappearing, not a conversion that changed what the
    script DOES.  These two cases pin the semantics for the sites README.md
    "Delta 7" makes its strongest claim about — output byte-identical to the
    full-``PATH`` control — so that claim is enforced by CI rather than by a
    recorded hand measurement.

    ``warm-lane-gc.sh``'s classification loop is the one where the conversion
    also closed a latent bug: an empty ``$name`` matches NEITHER
    ``PROTECT_GLOB`` nor ``LANE_GLOB``, so a protected entry would fall through
    to ``orphan_candidates``.  Only the 127 abort stood between that and an
    orphan-removal pass over a protected worktree — which is exactly why
    "protected entry still recognised as protected" is the assertion.
    """

    @staticmethod
    def _run_without_basename(
        argv: list[str], *, tmp_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run *argv* with ``basename`` shimmed to exit 127."""
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=120,
            env=_sanitized_env(extra={
                'PATH': (
                    f'{_path_hiding(tmp_path, "basename")}'
                    f'{os.pathsep}{os.environ["PATH"]}'
                ),
            }),
        )

    def test_gc_still_recognises_a_protected_entry_without_basename(
        self, tmp_path: Path,
    ) -> None:
        """``reclaim`` classifies ``_merge-x`` as protected, ``basename`` or no ``basename``."""
        mount = _warm_lane_pool(tmp_path)
        proc = self._run_without_basename(
            [
                _BASH, str(WARM_LANE_SCRIPT_DIR / 'warm-lane-gc.sh'),
                'reclaim', '--mount', str(mount),
            ],
            tmp_path=tmp_path,
        )
        combined = proc.stdout + proc.stderr

        assert 'skipping protected: _merge-x' in combined, (
            'warm-lane-gc.sh reclaim did not classify `_merge-x` as protected '
            'under a PATH without `basename`.  An empty $name matches neither '
            '--protect-glob nor --lane-glob, so the entry falls through to the '
            f'DESTRUCTIVE orphan pass (README.md "Delta 7").\n'
            f'output:\n{combined}'
        )
        assert proc.returncode == 0, (
            'warm-lane-gc.sh reclaim failed under a PATH without `basename`; '
            'the leaf extraction must need nothing on PATH.\n'
            f'output:\n{combined}'
        )
        assert (mount / '_merge-x').is_dir(), (
            'warm-lane-gc.sh reclaim REMOVED the protected `_merge-x` '
            f'worktree.\noutput:\n{combined}'
        )

    def test_audit_still_reports_its_rows_without_basename(
        self, tmp_path: Path,
    ) -> None:
        """The resident walk runs and the report rows are emitted, ``basename`` or no ``basename``."""
        mount = _warm_lane_pool(tmp_path)
        proc = self._run_without_basename(
            [
                _BASH, str(WARM_LANE_SCRIPT_DIR / 'warm-lane-audit.sh'),
                '--mount', str(mount),
            ],
            tmp_path=tmp_path,
        )
        combined = proc.stdout + proc.stderr

        assert 'lane=_merge-x' in combined, (
            'warm-lane-audit.sh emitted no per-lane row for `_merge-x` under a '
            'PATH without `basename` — the resident walk did not reach it.\n'
            f'output:\n{combined}'
        )
        assert 'HEADROOM resident=1' in combined, (
            'warm-lane-audit.sh reported no residents under a PATH without '
            f'`basename`.\noutput:\n{combined}'
        )
        assert 'PINNED ' in combined, (
            'warm-lane-audit.sh emitted no PINNED row under a PATH without '
            f'`basename` — the report was cut short.\noutput:\n{combined}'
        )
        assert proc.returncode == 0, (
            'warm-lane-audit.sh failed under a PATH without `basename`.\n'
            f'output:\n{combined}'
        )


#: The info line ``reclaim`` prints before it touches anything:
#: ``warm-lane-gc.sh reclaim: worktrees_dir=<...>  base_target=<...>  main_ref=<...>``.
_REPORTED_BASE_TARGET_RE = re.compile(r'base_target=(?P<target>\S+)\s+main_ref=')


class TestGcBaseTargetMatchesDirname:
    """``BASE_TARGET``'s ``--mount`` derivation is byte-equal to ``dirname``.

    The most intricate expansion in the delta — a ``%/`` trim, TWO separate
    empty-result guards and a ``*/*`` case — and every guard is load-bearing on
    a different input.  Nothing else in the suite pins the table: the shipped
    callers only ever pass a plain absolute ``--mount``, so a "simplification"
    back to a bare ``${MOUNT%/*}`` (the exact naive form the script's comment
    warns about) would misderive a trailing-slash ``--mount`` with every other
    case still green.

    Asserted as PARITY against the real ``dirname`` binary rather than against
    hardcoded strings, and against ``dirname`` rather than
    ``os.path.dirname`` — the two disagree on three of these very inputs
    (``worktrees`` → ``.`` vs ``''``; ``/a/b/wt/`` → ``/a/b`` vs ``/a/b/wt``),
    so the python function would encode the naive behaviour this pins against.

    ``--worktrees-dir`` is passed explicitly at an empty dir so the derivation
    is the ONLY thing under test: without it a ``--mount /`` case would set
    ``WORKTREES_DIR=/`` and the run would go on to walk the filesystem root.
    ``BASE_TARGET`` is still derived from ``--mount`` (the derivation only
    skips a value already set by its own flag), and the info line reports it
    before ``readlink -f`` returns early on the nonexistent target.
    """

    #: The nine inputs README.md "Delta 7" records the parity measurement over,
    #: minus "the real host value" (not reproducible in a test) and plus ``a/``.
    #: ``/`` and ``/worktrees`` are the two that need the empty-result guards;
    #: ``/a/b/wt/`` and ``a/`` are the trailing-slash edge; ``worktrees`` and
    #: ``.`` are the slashless relative edge the ``*/*`` case covers.
    @pytest.mark.parametrize(
        'mount',
        ['/worktrees', 'worktrees', '/a/b/wt', '/a/b/wt/', '/', '.', './x', 'a/'],
    )
    def test_derived_base_target_matches_dirname(
        self, mount: str, tmp_path: Path,
    ) -> None:
        empty_worktrees = tmp_path / 'empty-worktrees'
        empty_worktrees.mkdir()

        control = subprocess.run(
            ['dirname', mount],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()

        proc = subprocess.run(
            [
                _BASH, str(WARM_LANE_SCRIPT_DIR / 'warm-lane-gc.sh'),
                'reclaim', '--mount', mount,
                '--worktrees-dir', str(empty_worktrees),
            ],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=120,
            env=_sanitized_env(),
        )
        combined = proc.stdout + proc.stderr
        match = _REPORTED_BASE_TARGET_RE.search(combined)
        assert match is not None, (
            'warm-lane-gc.sh reclaim printed no `base_target=` info line, so '
            f'the derivation was never reported.\noutput:\n{combined}'
        )

        assert match.group('target') == f'{control}/base/target', (
            f'--mount {mount!r} derived base_target '
            f'{match.group("target")!r}, but `dirname {mount}` is {control!r}, '
            f'giving {control}/base/target.  The parameter-expansion parent-of '
            f'derivation must stay byte-equal to `dirname` (README.md '
            f'"Delta 7") — the two header comment blocks in warm-lane-gc.sh '
            f'document the derived VALUE, and it must not move.'
        )


class TestNoShippedScriptDerivesAPathByForking:
    """Directory-wide drift gate: the forking spelling appears ZERO times.

    The self-directory idiom is necessarily DUPLICATED across five scripts —
    a script's own directory is what tells it where its libs are, so the
    resolution cannot be extracted into a sourceable helper without depending
    on the very thing it resolves.  That rules out the single-definition-site
    guard task 3074 used for the record-scalar ``sed`` idiom.  This gate is its
    inverse and the only such discipline available here: instead of "the idiom
    appears exactly once", assert "the forking spelling appears zero times".

    See README.md "Delta 7" for the measurements behind it.
    """

    #: The ONE fork allowed to remain, by EXACT spelling rather than by a loose
    #: "mentions ``$0``" match — so a future ``$(basename "$0" .sh)`` or
    #: ``$(dirname "$0")`` is caught rather than waved through.  It feeds no
    #: path resolution: worst case under a hidden ``basename`` is a blank
    #: program name in a ``Usage:`` line.
    COSMETIC_PROGRAM_NAME = '$(basename "$0")'

    #: Any fork that derives a path from a value, in either direction, in
    #: either substitution syntax.  Whitespace-tolerant and backtick-aware
    #: deliberately: a gate meant to be load-bearing must not be defeated by
    #: ``$( dirname`` or `` `dirname `` — both are valid bash and neither is a
    #: spelling anyone would think to add to a literal list.
    PATH_DERIVING_FORK_RE = re.compile(r'(?:\$\(|`)\s*(?:dirname|basename)\b')

    #: Sub-classification, used only to ANNOTATE an offender: these two
    #: spellings resolve the script's OWN directory, whose failure mode is
    #: distinct from a path derivation on some other value (see the message).
    FORKING_SELF_DIR = (
        '$(dirname "${BASH_SOURCE',
        '$(dirname "$0"',
    )

    #: The distinct consequence of the self-directory spellings, appended to
    #: the offender line so the message still explains the `cd ""` mechanism
    #: that the narrower predecessor gate used to name on its own.
    SELF_DIR_NOTE = (
        '  <-- resolves the script\'s OWN directory: the substitution is '
        'EMPTY, `cd ""` SUCCEEDS as a no-op, and the directory silently '
        'becomes the CALLER\'S CWD at exit 0'
    )

    def test_only_cosmetic_program_name_forks_remain(self) -> None:
        """No path-deriving fork survives, in either direction, except the cosmetic one.

        ONE gate, not two.  Its predecessor was scoped to ``$(dirname
        "${BASH_SOURCE`` / ``$(dirname "$0"``, and that scope is exactly why
        nothing here flagged ``thin-warm-lane.sh``'s self-clobber guard: a
        ``basename`` fork on a VARIABLE is neither spelling.  A gate that
        cannot see the class it guards is itself a defect, so the scope was
        widened to any ``dirname``/``basename`` substitution — which strictly
        subsumes the old one.  Keeping both meant every future exception had to
        be encoded twice; the specific ``cd ""`` diagnosis the narrow gate
        existed for is preserved as ``SELF_DIR_NOTE``, appended to any offender
        that matches ``FORKING_SELF_DIR``.

        Both the sub-classification and the exception are matched LITERALLY
        while the offender scan is a regex.  That asymmetry is deliberate: the
        scan must be generous (it decides what is caught), the waiver must be
        exact (it decides what is let through).

        Whole-line comments are skipped — ``lib_portable.sh``,
        ``lib_lane_state.sh``, ``warm-lane-gc.sh`` and ``warm-lane-gc-sweep.sh``
        all quote these spellings in header/usage prose, which is documentation
        rather than a fork.  Only WHOLE-line comments, so a real fork with a
        trailing comment is still caught.  ``warm-lane-disk-guard.sh`` and
        ``warm-lane-degenerate-ref-check.sh`` carry only the cosmetic spelling
        and pass trivially.
        """
        offenders = []
        for path in sorted(WARM_LANE_SCRIPT_DIR.glob('*.sh')):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if line.lstrip().startswith('#'):
                    continue
                # Strip the allowed spelling first, then look at what is LEFT —
                # so a line carrying both an allowed and a disallowed fork is
                # still reported.
                residue = line.replace(self.COSMETIC_PROGRAM_NAME, '')
                if not self.PATH_DERIVING_FORK_RE.search(residue):
                    continue
                offender = f'{path.name}:{lineno}: {line.strip()}'
                if any(spelling in line for spelling in self.FORKING_SELF_DIR):
                    offender += self.SELF_DIR_NOTE
                offenders.append(offender)

        assert not offenders, (
            'These shipped scripts still derive a path by forking `basename` '
            'or `dirname`.  Only the cosmetic '
            f'`{self.COSMETIC_PROGRAM_NAME}` may remain: an external binary '
            'missing from PATH yields an EMPTY substitution, which aborts at '
            '127 in an assignment but compares silently FALSE inside '
            '`[ ... ]` (README.md "Delta 7"):\n  '
            + '\n  '.join(offenders)
        )
