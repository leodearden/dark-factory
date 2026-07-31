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
  since dark-factory task 3074 (leaf β) — ``$SCRIPT_DIR/lib_lane_state.sh``,
  behind a guard copied in shape from ``warm-lane-gc.sh``'s.  Unlike the other
  two libs ``lib_lane_state.sh`` is dark-factory-NATIVE rather than a reify
  relocation: it holds the facts only dark-factory owns (the ``.lane-state``
  record format and ``PROTECTED_PREFIXES``), which is why the audit's
  ``assigned`` column can no longer be produced without it.

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
    effects) from ``orchestrator/scripts/warm-lane/``.  A missing sibling lib
    surfaces in one of two shapes, and both are asserted against: the script's
    own fail-loud wiring message + ``exit 2`` (``lib_live_refs.sh`` in the gc
    scripts, ``lib_lane_state.sh`` in the audit), or bash's bare ``source``
    failure (``lib_portable.sh``, which has no explicit guard).
    """

    #: Exit 2 is the wiring/usage sentinel both gc scripts use for
    #: "incomplete deployment" (deliberately NOT 1, which means runtime error).
    WIRING_EXIT = 2

    #: The verbatim fail-loud fragments emitted by the sourcing scripts' guards.
    FAIL_LOUD_FRAGMENTS = (
        'lib_live_refs.sh not found next to',
        'lib_portable.sh: No such file',
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

        The sixth repo-root failure mode, and the only one that is silent end
        to end.  ``dirname`` and ``basename`` are EXTERNAL BINARIES: under a
        ``PATH`` without them ``$(dirname ...)`` yields the empty string,
        ``cd ""`` SUCCEEDS as a no-op, and the resolution lands on the caller's
        CWD — no error, no warning, exit 0.  A systemd unit's ``PATH`` need not
        carry coreutils, and this script is run on a fresh host to provision
        the pool substrate, which is exactly where a minimal ``PATH`` is
        likeliest.

        MEASURED RED on base HEAD 8d276d3c5f: control ``/tmp/<...>/warm-lanes``,
        hidden-``dirname`` ``/warm-lanes`` — the bare FILESYSTEM ROOT,
        advertised to an operator about to provision a multi-terabyte XFS
        volume, at rc=0 with no diagnostic of any kind.  Unlike the three
        lib-sourcing scripts this one has NO ``[ ! -f ]`` check anywhere on the
        path, so a mis-resolution has nothing at all to trip over.

        Asserted as PARITY against a same-run control rather than a literal
        expected path: the synthetic repo lives under ``tmp_path``, so a
        hardcoded mount would encode the harness's layout and rot the moment
        ``_stage_provision_script`` changes.  "the answer with ``dirname``
        hidden equals the answer with ``dirname`` present" IS the contract.
        The second assertion pins the specific regression so the case cannot
        pass by both sides being equally broken.

        All three nestings are exercised because ``_default_mount()`` reaches a
        DIFFERENT fork in each: the plain repo uses only the ``dirname
        "$REPO_ROOT"`` at the top, while a repo inside a worktrees dir also
        takes the ascend branch's second ``dirname``.  ``.worktrees`` is
        dark-factory's own spelling (README Delta 2), i.e. the shape an agent
        or operator in THIS repo actually runs from.

        No assertion on the ``Usage:`` program-name line: ``basename "$0"``
        there is deliberately left forking (README Delta 7), so that line
        renders blank under this ``PATH`` by design.
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
    right COPY of them is the one loaded.  Under a ``PATH`` without ``dirname``
    the ``$(dirname "${BASH_SOURCE[0]}")`` fork yields the empty string,
    ``cd ""`` succeeds as a no-op, and ``SCRIPT_DIR`` silently becomes the
    caller's CWD — so the script sources whatever files with those names happen
    to be sitting there.

    The realistic, non-adversarial trigger is not an attacker: it is invoking
    one of these from reify's own ``scripts/`` dir, or from another
    dark-factory checkout's ``warm-lane/`` dir.  Both carry precisely these
    filenames, at a different (possibly older) version.

    This REFUTES the hypothesis these scripts were filed under — that a
    mis-resolution would land on their existing fail-loud wiring guards and
    ``exit 2``.  The guards are ``[ ! -f "$SCRIPT_DIR/lib_*.sh" ]``, which test
    the MIS-RESOLVED path, so any CWD holding a same-named file satisfies them.
    The loud degrade is therefore contingent on the caller's CWD being empty,
    and cannot detect this class at all.

    MEASURED RED on base HEAD 8d276d3c5f, from a decoy CWD under a
    ``dirname``-less ``PATH``: ``warm-lane-gc.sh`` exit 0 having sourced BOTH
    the decoy ``lib_live_refs.sh`` and the decoy ``lib_lane_state.sh``;
    ``warm-lane-gc-sweep.sh`` exit 0 having sourced the decoy
    ``lib_live_refs.sh``; ``warm-lane-audit.sh`` exit 0 having sourced both the
    decoy ``lib_portable.sh`` and the decoy ``lib_lane_state.sh``.  These are
    scripts whose job is deleting worktrees and reporting pool state.

    Deliberately NO exit-code assertion: the decoys make the run SUCCEED, and
    the defect is *which files were sourced*, not how the process ended.  The
    exit code here is a function of the caller's CWD rather than of the defect
    — which is exactly why the filed expectation of ``exit 2`` was wrong.
    """

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

        assert _DECOY_MARKER not in combined, (
            f'{name} --help sourced a sibling lib from the CALLER\'S CWD '
            f'instead of from {WARM_LANE_SCRIPT_DIR}.  Its own directory must '
            f'resolve by parameter expansion, which reads no PATH.\n'
            f'output:\n{combined}'
        )

    def test_a_bare_hostile_cwd_does_not_reach_lib_portable_from_the_cwd(
        self, tmp_path: Path,
    ) -> None:
        """``warm-lane-audit.sh`` from an EMPTY hostile CWD, and the exit code it uses.

        The audit-specific half, recorded because it corrects the filed
        expectation twice over.  The hypothesis said audit "would land on its
        existing fail-loud ``source`` guard and exit 2".  It does not:

        * The ``[ ! -f ]`` guard task 3074 added covers only
          ``lib_lane_state.sh``, and sits AFTER the ``lib_portable.sh``
          ``source``, which has no guard at all.
        * So with nothing to pick up in the CWD, audit dies on bash's own bare
          ``source`` failure at exit **1** — the RUNTIME code, not the exit-2
          wiring sentinel the exit-code taxonomy assigns this class.  MEASURED
          on base with CWD=/: ``warm-lane-audit.sh: line 145:
          //lib_portable.sh: No such file or directory``, exit 1.  A
          triage-misleading exit code on top of a wrong path.

        Asserted on the RESOLVED PATH, not the exit code, for the reason in the
        class docstring: the exit code here is whatever the caller's CWD makes
        it.  The unguarded-``lib_portable.sh`` gap is real and separately
        actionable, but it is a behaviour change to a reify byte-copy that this
        task's Delta does not cover — and with ``SCRIPT_DIR`` resolved
        correctly the bare ``source`` failure is unreachable in every case
        measured here.
        """
        proc = self._help_under_a_hidden_dirname(
            'warm-lane-audit.sh', cwd=tmp_path, tmp_path=tmp_path,
        )
        combined = proc.stdout + proc.stderr

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
        this one runs it as a program, with the lane dir as an argument.

        MEASURED RED on base HEAD 8d276d3c5f: the decoy ran — stderr carries
        the marker followed by ``[ok] Re-seeded: <lane>/target``, i.e.
        thin-warm-lane reported SUCCESS for a reseed performed by an arbitrary
        file picked up off the caller's CWD.

        Reachability, recorded honestly because it is what makes this
        low-priority rather than urgent: ``_script_dir`` is computed only when
        ``--seed-script`` was NOT passed, and README "Delta 3" states
        dark-factory's own caller never passes ``--reseed`` at all.  The
        exposure is a future caller — and the warning Delta 3 already carries
        ("any future caller that does pass --reseed must also pass
        --seed-script") is exactly the one this closes properly.

        Note also what hiding ``dirname`` CONVERTS: with a full ``PATH`` the
        default seed script resolves to a sibling that deliberately does not
        exist here (``seed-warm-lane.sh`` stayed project-owned, PRD §5), so the
        reseed fails harmlessly and says so.  Hiding ``dirname`` turns that
        safe no-op into an execution.
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


class TestNoShippedScriptResolvesItsOwnDirectoryByForking:
    """Directory-wide drift gate: the forking spelling appears ZERO times.

    D2 forces the correct idiom to be DUPLICATED across five scripts — a
    script's own directory is what tells it where its libs are, so the
    resolution cannot be extracted into a sourceable helper without depending
    on the very thing it resolves.  That rules out the single-definition-site
    guard task 3074 used for the record-scalar ``sed`` idiom.  This is its
    inverse and the only such discipline available here: instead of "the idiom
    appears exactly once", assert "the forking spelling appears zero times".

    Scoped to SELF-directory resolution deliberately, along two axes, because
    a blanket ``dirname`` scan false-trips on things that are not this defect:

    * By SPELLING — ``warm-lane-gc.sh`` legitimately forks
      ``dirname "$MOUNT"`` for ``BASE_TARGET``, which is arithmetic on an
      argument, not on the script's own location.
    * By COMMENT — ``lib_portable.sh:5`` and ``lib_lane_state.sh:6`` both
      document their own ``source "$(dirname "${BASH_SOURCE[0]}")/<lib>"``
      call convention in a usage header.  That is the CALLER's spelling being
      quoted in prose; measured, an unscoped gate reports both as offenders.
      Only whole-line comments are skipped, so a real resolution with a
      trailing comment is still caught.

    ``warm-lane-disk-guard.sh`` and ``warm-lane-degenerate-ref-check.sh``
    carry no self-directory resolution at all and pass trivially.
    """

    #: The two spellings of "resolve my own directory by forking ``dirname``".
    FORKING_SELF_DIR = (
        '$(dirname "${BASH_SOURCE',
        '$(dirname "$0"',
    )

    def test_no_shipped_script_resolves_its_own_directory_by_forking_dirname(
        self,
    ) -> None:
        offenders = [
            f'{path.name}:{lineno}: {line.strip()}'
            for path in sorted(WARM_LANE_SCRIPT_DIR.glob('*.sh'))
            for lineno, line in enumerate(path.read_text().splitlines(), start=1)
            if not line.lstrip().startswith('#')
            and any(spelling in line for spelling in self.FORKING_SELF_DIR)
        ]

        assert not offenders, (
            'These shipped scripts still resolve their own directory by '
            'forking `dirname`, so under a PATH without it the substitution '
            'is EMPTY, `cd ""` succeeds as a no-op, and the directory becomes '
            'the caller\'s CWD (README.md "Delta 7"):\n  '
            + '\n  '.join(offenders)
        )
