"""Shipped-script contract for dark-factory's relocated warm-lane scripts.

Task 3072 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` leaf α, Phase 1).

dark-factory now ships its own copies of the seven project-agnostic warm-lane
scripts under ``orchestrator/scripts/warm-lane/`` so a project that does not
carry them still gets warm-lane GC, disk guarding and auditing.  These tests
pin the *shipped* half of that contract — the files exist, are executable, and
are syntactically valid — independently of the *resolution* half
(``test_warm_lane_script_resolution.py``), which pins how ``GitOps`` chooses
between a project override and these copies.

The sibling-wiring class is the load-bearing one.  ``warm-lane-gc.sh`` and
``warm-lane-gc-sweep.sh`` ``source "$SCRIPT_DIR/lib_live_refs.sh"`` and
deliberately ``exit 2`` when it is absent (reify task 5572 made that fail-loud
precisely so a silently-missing liveness guard cannot recur);
``warm-lane-audit.sh`` sources ``$SCRIPT_DIR/lib_portable.sh``.  Neither lib is
one of the seven named scripts, so a seven-file-only relocation would ship
three scripts that cannot execute.  Running each with ``--help`` from the new
directory is the executable proof that the two libs actually travelled along.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# Resolved from THIS FILE (orchestrator/tests/ -> orchestrator/), never from the
# process CWD: the merge-verify harness runs pytest from the ``orchestrator/``
# cwd while a plain ``pytest orchestrator/tests`` runs from the repo root, and
# this contract must hold identically under both.
WARM_LANE_SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'warm-lane'

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
    surfaces either as the script's own fail-loud wiring message + ``exit 2``
    (``lib_live_refs.sh``) or as bash's ``source`` failure (``lib_portable.sh``),
    so both shapes are asserted against.
    """

    #: Exit 2 is the wiring/usage sentinel both gc scripts use for
    #: "incomplete deployment" (deliberately NOT 1, which means runtime error).
    WIRING_EXIT = 2

    #: The verbatim fail-loud fragments emitted by the sourcing scripts' guards.
    FAIL_LOUD_FRAGMENTS = (
        'lib_live_refs.sh not found next to',
        'lib_portable.sh: No such file',
        # warm-lane-audit.sh's guard for dark-factory's own lane-state lib
        # (task 3074), copied in shape from warm-lane-gc.sh's.
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
_HOSTILE_ENV_KEYS = ('REIFY_WARM_LANE_MOUNT', 'GIT_DIR', 'GIT_WORK_TREE')


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
