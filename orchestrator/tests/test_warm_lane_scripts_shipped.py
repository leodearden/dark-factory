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

    #: The verbatim fail-loud fragments emitted by the two gc scripts' guards.
    FAIL_LOUD_FRAGMENTS = (
        'lib_live_refs.sh not found next to',
        'lib_portable.sh: No such file',
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
    actually ships.  ``git_init`` selects which of the two repo-root resolution
    strategies gets exercised: a real checkout, or a directory tree with no git
    metadata at all.
    """
    script_dir = repo.joinpath(*_NEW_NESTING)
    script_dir.mkdir(parents=True, exist_ok=True)
    staged = script_dir / 'provision-warm-lane-fs.sh'
    staged.write_bytes((WARM_LANE_SCRIPT_DIR / 'provision-warm-lane-fs.sh').read_bytes())
    staged.chmod(0o755)
    if git_init:
        subprocess.run(
            ['git', 'init', '-q', '-b', 'main'], cwd=repo, check=True, timeout=60,
        )
    return staged


def _default_mount_in_usage(staged: Path) -> str:
    """Run the staged script with ``--help`` and return its usage text.

    ``REIFY_WARM_LANE_MOUNT`` is stripped from the environment so the printed
    default is the script's own derivation rather than an inherited override —
    the operator-facing value this test exists to protect.
    """
    env = {k: v for k, v in os.environ.items() if k != 'REIFY_WARM_LANE_MOUNT'}
    proc = subprocess.run(
        [str(staged), '--help'],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
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

    Hermetic: a synthetic repo under ``tmp_path``, and ``--help`` only — no
    loopback image, no mount, nothing privileged.
    """

    def test_default_mount_is_derived_from_the_repo_root(self, tmp_path: Path) -> None:
        """A checkout at ``<tmp>/repo`` → ``<tmp>/warm-lanes``."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        usage = _default_mount_in_usage(_stage_provision_script(repo, git_init=True))

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
        """No git metadata → path-arithmetic fallback still lands on the repo root.

        The script must not depend on being inside a checkout: it is run on a
        fresh host to provision the pool substrate, sometimes from an unpacked
        tree, and ``git`` may be absent entirely.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        usage = _default_mount_in_usage(_stage_provision_script(repo, git_init=False))

        assert str(tmp_path / 'warm-lanes') in usage, (
            f'With no git metadata the fallback must still resolve the repo root '
            f'({repo}), giving {tmp_path / "warm-lanes"}.\nusage:\n{usage}'
        )
        assert str(repo / 'orchestrator' / 'warm-lanes') not in usage, (
            f'Fallback resolution landed on the script\'s grandparent instead of '
            f'the repo root.\nusage:\n{usage}'
        )

    def test_ascend_past_worktrees_is_preserved(self, tmp_path: Path) -> None:
        """A checkout inside ``worktrees/`` still surfaces the mount one level higher.

        Mirror case for the pre-existing ``_default_mount`` behaviour: the
        warm-lanes dir must live BESIDE the worktrees tree, never inside a
        worktree.  Pinned alongside the depth fix because both are consumers of
        ``REPO_ROOT`` — a fix that got the root right but broke the ascend
        would be just as much a parity break.
        """
        repo = tmp_path / 'worktrees' / 'repo'
        repo.mkdir(parents=True)
        usage = _default_mount_in_usage(_stage_provision_script(repo, git_init=True))

        assert str(tmp_path / 'warm-lanes') in usage, (
            f'A repo inside worktrees/ must surface the mount beside the '
            f'worktrees tree ({tmp_path / "warm-lanes"}).\nusage:\n{usage}'
        )
        assert str(tmp_path / 'worktrees' / 'warm-lanes') not in usage, (
            'The ascend-past-worktrees behaviour was lost — the warm-lanes dir '
            f'would live inside the worktrees tree.\nusage:\n{usage}'
        )
