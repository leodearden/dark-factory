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
