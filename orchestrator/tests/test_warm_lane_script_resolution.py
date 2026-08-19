"""Warm-lane script resolution: project override first, dark-factory second.

Task 3072 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` leaf α, Phase 1).

Companion to ``test_warm_lane_scripts_shipped.py``, which pins that
dark-factory's own copies exist under ``orchestrator/scripts/warm-lane/``.
This module pins how :class:`~orchestrator.git_ops.GitOps` *chooses* between
those copies and a project's own ``<project_root>/scripts/<name>`` override,
and pins that the choice is observable:

* **B7** — a project override always wins, and the resolved path + origin is
  logged at INFO on every invocation.
* **B8** — "no implementation at either location" is a WARNING naming both
  tried paths, never a DEBUG line naming one.

The dark-factory fallback root is seamed through ``ORCH_WARM_LANE_SCRIPT_DIR``,
read at call time.  ``orchestrator/tests/conftest.py`` pins it at a
guaranteed-absent directory for the whole suite (autouse) so the ~200 existing
tests that assert "no scripts/ dir → fail-soft sentinel" never execute the real
host scripts against a ``tmp_path``.  Tests here opt back in explicitly.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import orchestrator.git_ops as git_ops_mod
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps

GIT_OPS_LOGGER = 'orchestrator.git_ops'

#: Env seam for the dark-factory fallback root.  Test hermeticity only —
#: production resolution is repo-relative and never reads this.
DF_DIR_ENV = 'ORCH_WARM_LANE_SCRIPT_DIR'


def _pin_df_dir(monkeypatch: pytest.MonkeyPatch, path: Path) -> Path:
    """Point the dark-factory fallback root at ``path`` for this test."""
    monkeypatch.setenv(DF_DIR_ENV, str(path))
    return path


def _make_config(**overrides: Any) -> GitConfig:
    """Canonical warm-lane GitConfig (mirrors test_warm_lane_disk_guard.py)."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_disk_guard=True,
        warm_lane_min_free_gib=50,
        warm_lane_min_free_inodes=500_000,
        **overrides,
    )


def _make_git_ops(project_root: Path, **overrides: Any) -> GitOps:
    """Build a GitOps rooted at ``project_root`` (no real repo needed here)."""
    return GitOps(_make_config(**overrides), project_root, warm_lane_pool_size=1)


def _write_stub(path: Path, body: str = 'exit 0\n', mode: int = 0o755) -> Path:
    """Write an executable stub script at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\n' + body)
    path.chmod(mode)
    return path


# ---------------------------------------------------------------------------
# The six warm-lane script wrappers, as one table
# ---------------------------------------------------------------------------


class Wrapper(NamedTuple):
    """One fail-soft warm-lane wrapper, its script, and its absent-script sentinel."""

    method: str
    script: str
    sentinel: Any

    def invoke(self, git_ops: GitOps) -> Awaitable[Any]:
        """Call the wrapper with whatever arguments it happens to take."""
        fn: Callable[..., Awaitable[Any]] = getattr(git_ops, self.method)
        if self.method == 'warm_lane_ref_is_degenerate':
            return fn('3072')
        if self.method == '_run_thin_warm_lane':
            return fn(git_ops.worktree_base / 'task-3072')
        return fn()


#: Sentinels are the pre-relocation fail-soft contract, preserved verbatim:
#: an absent script must degrade to exactly these, never raise.
WRAPPERS = (
    Wrapper('_run_warm_lane_disk_guard', 'warm-lane-disk-guard.sh', 127),
    Wrapper('_run_warm_lane_soft_guard', 'warm-lane-disk-guard.sh', 127),
    Wrapper('_run_warm_lane_audit', 'warm-lane-audit.sh', None),
    Wrapper('_run_warm_lane_gc_reclaim', 'warm-lane-gc.sh', 127),
    Wrapper('warm_lane_ref_is_degenerate', 'warm-lane-degenerate-ref-check.sh', False),
    Wrapper('_run_thin_warm_lane', 'thin-warm-lane.sh', 127),
)

WRAPPER_PARAMS = [pytest.param(w, id=w.method.lstrip('_')) for w in WRAPPERS]


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Bare project root whose pool storage reads as PRESENT.

    The ``.pool-root`` sentinel keeps the pre-resolution pool-storage guards in
    ``_run_warm_lane_gc_reclaim`` and ``_run_thin_warm_lane`` from
    short-circuiting, so tests here reach the resolution step they mean to
    exercise.  ``TestPreResolutionGuardsStillWinTheRace`` deliberately removes
    it to pin the opposite.
    """
    root = tmp_path / 'proj'
    (root / '.worktrees').mkdir(parents=True)
    (root / '.worktrees' / '.pool-root').touch()
    return root


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    """Formatted messages of every captured record at exactly ``level``."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == level and r.name == GIT_OPS_LOGGER
    ]


class TestResolveWarmLaneScript:
    """``GitOps._resolve_warm_lane_script`` preference order.

    Contract: ``(path, origin)`` where origin is ``'project'`` or
    ``'dark-factory'``, or ``None`` when neither location has the script.
    """

    NAME = 'warm-lane-gc.sh'

    def test_project_copy_wins_when_both_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both present → the project override, tagged 'project' (B7)."""
        project_root = tmp_path / 'proj'
        df_dir = _pin_df_dir(monkeypatch, tmp_path / 'df')
        project_script = _write_stub(project_root / 'scripts' / self.NAME)
        _write_stub(df_dir / self.NAME)

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved == (project_script, 'project'), (
            'A project that carries its own warm-lane tooling must keep it '
            f'(PRD D3); got {resolved!r}'
        )

    def test_dark_factory_copy_used_when_project_lacks_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only the dark-factory copy present → it is used, tagged 'dark-factory'."""
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        df_dir = _pin_df_dir(monkeypatch, tmp_path / 'df')
        df_script = _write_stub(df_dir / self.NAME)

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved == (df_script, 'dark-factory'), (
            'A project with no warm-lane tooling must fall back to '
            f"dark-factory's shipped copy; got {resolved!r}"
        )

    def test_neither_location_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Neither present → None, so callers can emit the both-paths WARNING."""
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        _pin_df_dir(monkeypatch, tmp_path / 'absent-df-dir')

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved is None, (
            f'No implementation at either location must resolve to None; got {resolved!r}'
        )

    def test_project_copy_wins_even_when_not_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Existence, not mode, is the predicate — mirrors today's ``script.exists()``.

        The pre-relocation call sites branched on ``script.exists()`` alone; a
        non-executable project copy therefore reached the subprocess spawn and
        failed there.  Preserving that keeps the failure attributable to the
        project's own broken script rather than silently substituting
        dark-factory's, which would mask it.
        """
        project_root = tmp_path / 'proj'
        df_dir = _pin_df_dir(monkeypatch, tmp_path / 'df')
        project_script = _write_stub(
            project_root / 'scripts' / self.NAME, mode=0o644,
        )
        _write_stub(df_dir / self.NAME)

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved == (project_script, 'project'), (
            'Resolution must key on existence, not the execute bit, so a broken '
            f'project override stays attributable; got {resolved!r}'
        )

    def test_unset_env_falls_back_to_the_repo_relative_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the seam UNSET, the fallback root is orchestrator/scripts/warm-lane.

        This is the case that actually ships: production never sets
        ORCH_WARM_LANE_SCRIPT_DIR, so the repo-relative default is what resolves.
        Pinned explicitly because the autouse hermeticity fixture masks it
        everywhere else in the suite.
        """
        monkeypatch.delenv(DF_DIR_ENV, raising=False)
        project_root = tmp_path / 'proj'
        project_root.mkdir()

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        expected_dir = (
            Path(git_ops_mod.__file__).resolve().parents[2] / 'scripts' / 'warm-lane'
        )
        assert resolved is not None, (
            'With the env seam unset the repo-relative dark-factory copy must '
            'resolve — production depends on it and sets no env var'
        )
        path, origin = resolved
        assert path == expected_dir / self.NAME, (
            f'Expected the repo-relative copy at {expected_dir / self.NAME}; got {path}'
        )
        assert origin == 'dark-factory'

    def test_candidates_are_the_single_searched_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``_warm_lane_script_candidates`` IS the search order, not a copy of it.

        Both halves of resolution consume this one list: the resolver takes the
        first entry that exists, and the not-found WARNING names every entry.
        Pinned so the order an operator reads can never drift from the order
        actually searched — the failure mode of two independently-constructed
        candidate lists.
        """
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        df_dir = _pin_df_dir(monkeypatch, tmp_path / 'df')
        git_ops = _make_git_ops(project_root)

        candidates = git_ops._warm_lane_script_candidates(self.NAME)

        assert candidates == (
            (project_root / 'scripts' / self.NAME, 'project'),
            (df_dir / self.NAME, 'dark-factory'),
        ), f'Search order must be project-then-dark-factory; got {candidates!r}'

        # Each candidate, made to exist in isolation, is what the resolver
        # returns — so the list is genuinely the thing being walked.
        for index, (path, origin) in enumerate(candidates):
            _write_stub(path)
            assert git_ops._resolve_warm_lane_script(self.NAME) == (path, origin), (
                f'Candidate {index} ({origin}) exists but the resolver did not '
                f'return it — the resolver is not walking this list'
            )
            path.unlink()

    def test_env_seam_is_read_at_call_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-pointing the seam takes effect without a module reload.

        The autouse conftest fixture and the opt-in override both rely on
        ``monkeypatch.setenv`` winning over an import-time snapshot.
        """
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        git_ops = _make_git_ops(project_root)
        _pin_df_dir(monkeypatch, tmp_path / 'absent-df-dir')

        assert git_ops._resolve_warm_lane_script(self.NAME) is None

        later_dir = tmp_path / 'df-later'
        df_script = _write_stub(later_dir / self.NAME)
        _pin_df_dir(monkeypatch, later_dir)

        assert git_ops._resolve_warm_lane_script(self.NAME) == (
            df_script, 'dark-factory',
        )


def _write_origin_stub(path: Path, origin: str, log: Path) -> Path:
    """Executable stub that records WHICH copy ran, then succeeds.

    Appends ``<origin> <argv>`` to ``log``.  The log path is absolute and baked
    in at write time rather than derived from ``$0`` (the idiom
    ``test_warm_lane_disk_guard.py`` uses), because here the two stubs live in
    unrelated directories — a project ``scripts/`` dir and a stand-in
    dark-factory dir — and must nonetheless append to ONE ordered log so
    "which copy ran" is answerable by reading it.

    Emits a ``HEADROOM`` line so ``_run_warm_lane_audit`` — the one wrapper
    that parses stdout — reaches its success path like the other five.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        f'echo "{origin} $*" >> "{log}"\n'
        'echo "HEADROOM lanes=1 free=9000G"\n'
    )
    path.chmod(0o755)
    return path


def _origins(log: Path) -> list[str]:
    """First token of each call-log line — which copy ran, in order."""
    if not log.exists():
        return []
    return [line.split()[0] for line in log.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
class TestResolvedPathIsLoggedAtInfo:
    """B7 — the project override wins, and the choice is visible at INFO.

    Two halves, both required by the task's USER-OBSERVABLE SIGNAL:

    * **Which copy ran** is proven by execution, not by inspecting the
      resolver: both stubs append to one shared log, so "the project copy ran
      and the dark-factory copy did not" is a statement about the log's
      contents rather than about a return value.
    * **The resolved path is logged at INFO on every invocation.**  Leaf ζ's
      go/no-go reads this line off a live reclaim pass, so it must not be
      memoised, rate-limited or first-call-only —
      :meth:`test_info_line_is_emitted_on_every_invocation` pins that.

    The origin is asserted in its parenthesised form ``(project)`` /
    ``(dark-factory)``.  A bare substring check would be vacuous: the resolved
    path is interpolated into the same record and ``tmp_path`` embeds the test
    name, so ``'project' in message`` is true for reasons that have nothing to
    do with the origin the resolver reported.
    """

    @pytest.fixture
    def call_log(self, tmp_path: Path) -> Path:
        """Shared ordered log both stubs append to."""
        return tmp_path / 'which-copy-ran.log'

    @pytest.mark.parametrize('wrapper', WRAPPER_PARAMS)
    async def test_project_override_runs_and_is_logged(
        self,
        wrapper: Wrapper,
        project_root: Path,
        df_warm_lane_script_dir,
        call_log: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Both copies present → the PROJECT copy executes and INFO says so."""
        df_dir = df_warm_lane_script_dir()
        project_script = _write_origin_stub(
            project_root / 'scripts' / wrapper.script, 'project', call_log,
        )
        _write_origin_stub(df_dir / wrapper.script, 'dark-factory', call_log)
        git_ops = _make_git_ops(project_root)

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            await wrapper.invoke(git_ops)

        assert _origins(call_log) == ['project'], (
            f'{wrapper.method} must execute the project override and NOT '
            f"dark-factory's copy when both exist (PRD D3: dark-factory's copy "
            f'is the floor, not the ceiling); call log: {_origins(call_log)!r}'
        )
        infos = _records(caplog, logging.INFO)
        matching = [
            m for m in infos if str(project_script) in m and '(project)' in m
        ]
        assert matching, (
            f'{wrapper.method} must log the resolved path and origin at INFO.\n'
            f'  expected path: {project_script}\n'
            f'  expected origin marker: (project)\n'
            f'  captured INFOs: {infos!r}'
        )

    @pytest.mark.parametrize('wrapper', WRAPPER_PARAMS)
    async def test_shipped_copy_runs_when_project_lacks_one(
        self,
        wrapper: Wrapper,
        project_root: Path,
        df_warm_lane_script_dir,
        call_log: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Only dark-factory's copy present → it executes and INFO says so."""
        df_dir = df_warm_lane_script_dir()
        df_script = _write_origin_stub(
            df_dir / wrapper.script, 'dark-factory', call_log,
        )
        git_ops = _make_git_ops(project_root)

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            await wrapper.invoke(git_ops)

        assert _origins(call_log) == ['dark-factory'], (
            f'{wrapper.method} must fall back to the shipped copy when the '
            f'project carries no override; call log: {_origins(call_log)!r}'
        )
        infos = _records(caplog, logging.INFO)
        matching = [
            m for m in infos if str(df_script) in m and '(dark-factory)' in m
        ]
        assert matching, (
            f'{wrapper.method} must log the resolved path and origin at INFO.\n'
            f'  expected path: {df_script}\n'
            f'  expected origin marker: (dark-factory)\n'
            f'  captured INFOs: {infos!r}'
        )

    async def test_info_line_is_emitted_on_every_invocation(
        self,
        project_root: Path,
        df_warm_lane_script_dir,
        call_log: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Two reclaim passes → two INFO records, not one.

        Leaf ζ's go/no-go reads the resolved-path line off a live reclaim pass,
        which may be the hundredth of the process.  A memo or a
        once-per-process guard would make that read silently empty, so the line
        is unconditional per invocation and this test is what holds it there.
        """
        df_dir = df_warm_lane_script_dir()
        df_script = _write_origin_stub(
            df_dir / 'warm-lane-gc.sh', 'dark-factory', call_log,
        )
        git_ops = _make_git_ops(project_root)

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            await git_ops._run_warm_lane_gc_reclaim()
            await git_ops._run_warm_lane_gc_reclaim()

        assert _origins(call_log) == ['dark-factory', 'dark-factory'], (
            f'Both reclaim passes must actually spawn the script; '
            f'call log: {_origins(call_log)!r}'
        )
        resolved = [m for m in _records(caplog, logging.INFO) if str(df_script) in m]
        assert len(resolved) == 2, (
            f'The resolved-path INFO line must be emitted on EVERY invocation '
            f'(no memo, no once-per-process guard) — leaf ζ reads it off an '
            f'arbitrary reclaim pass; got {len(resolved)} record(s): {resolved!r}'
        )


def _argv(log: Path) -> list[list[str]]:
    """Argv of each logged call, with the leading origin token stripped."""
    if not log.exists():
        return []
    return [
        line.split()[1:] for line in log.read_text().splitlines() if line.strip()
    ]


@pytest.mark.asyncio
class TestGcReclaimPassesProjectSeedScript:
    """The PRD §5 seed primitive is named explicitly, not left to a sibling default.

    ``warm-lane-gc.sh`` defaults ``SEED_SCRIPT`` to ``$SCRIPT_DIR/
    seed-warm-lane.sh`` and invokes it UNCONDITIONALLY on the Pass-1 lane-reset
    path.  ``seed-warm-lane.sh`` is one of the two genuinely toolchain-bound
    primitives that PRD §5 deliberately keeps in the project, so it does NOT
    travel to ``orchestrator/scripts/warm-lane/`` with the policy script.

    Left alone, dark-factory's relocated copy would therefore compute a sibling
    path that does not exist and fail EVERY lane reset at cutover (leaf ζ/κ) —
    a silently-stopped GC accreting the pool to ENOSPC, exactly the failure
    class this leaf exists to remove.  ``gc.sh`` already exposes
    ``--seed-script PATH``, so naming the project's primitive explicitly is
    resolution wiring, not a policy change.

    It is also a strict no-op TODAY, which is what these tests pin: for a
    reify-shaped project_root the passed path is byte-identical to the one
    gc.sh's own default computes, and for a project with no seed script the
    flag is omitted so argv is unchanged.  Patching the relocated script's
    default instead would make dark-factory guess at a project-owned path,
    violating PRD invariant C-1.
    """

    @pytest.fixture
    def call_log(self, tmp_path: Path) -> Path:
        """Ordered log of the gc stub's argv."""
        return tmp_path / 'gc-argv.log'

    @pytest.mark.parametrize('origin', ['project', 'dark-factory'])
    async def test_seed_script_is_passed_for_both_origins(
        self,
        origin: str,
        project_root: Path,
        df_warm_lane_script_dir,
        call_log: Path,
    ) -> None:
        """Present project seed script → --seed-script names it, whichever gc ran.

        Parametrised over both resolution origins because the failure this
        prevents is origin-dependent (dark-factory's copy has the wrong
        sibling) while the FIX must not be: a reify-shaped project_root, which
        resolves gc.sh to its own copy, must receive exactly the path that
        copy's sibling default would have computed.  That identity is what
        makes the change a no-op for reify today.
        """
        df_dir = df_warm_lane_script_dir()
        gc_dir = (
            project_root / 'scripts' if origin == 'project' else df_dir
        )
        _write_origin_stub(gc_dir / 'warm-lane-gc.sh', origin, call_log)
        seed = _write_stub(project_root / 'scripts' / 'seed-warm-lane.sh')

        await _make_git_ops(project_root)._run_warm_lane_gc_reclaim()

        assert _origins(call_log) == [origin], (
            f'Expected the {origin} gc copy to run; got {_origins(call_log)!r}'
        )
        argv = _argv(call_log)[0]
        assert '--seed-script' in argv, (
            f"gc.sh must be told where the project's PRD §5 seed primitive "
            f'lives — its own sibling default does not survive the relocation. '
            f'argv: {argv!r}'
        )
        assert argv[argv.index('--seed-script') + 1] == str(seed), (
            f'--seed-script must name <project_root>/scripts/seed-warm-lane.sh '
            f'({seed}); argv: {argv!r}'
        )

    async def test_argv_is_unchanged_when_project_has_no_seed_script(
        self,
        project_root: Path,
        df_warm_lane_script_dir,
        call_log: Path,
    ) -> None:
        """Absent project seed script → argv byte-identical to pre-relocation.

        dark-factory must not invent a path for a primitive the project does
        not have; omitting the flag leaves gc.sh's own default in charge,
        exactly as before this leaf.
        """
        df_dir = df_warm_lane_script_dir()
        _write_origin_stub(df_dir / 'warm-lane-gc.sh', 'dark-factory', call_log)
        git_ops = _make_git_ops(project_root)

        await git_ops._run_warm_lane_gc_reclaim()

        assert _argv(call_log) == [
            ['reclaim', '--mount', str(git_ops.worktree_base)],
        ], (
            'With no project seed script the reclaim argv must be byte-identical '
            f'to today — no invented --seed-script; got {_argv(call_log)!r}'
        )

    async def test_missing_project_seed_script_is_warned_about(
        self,
        project_root: Path,
        df_warm_lane_script_dir,
        call_log: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """dark-factory's copy + no project seed script → a WARNING says so.

        This is the fallback's OWN target case: a project carrying no warm-lane
        tooling is exactly why dark-factory ships these copies.  With no
        ``--seed-script`` to pass, the relocated ``warm-lane-gc.sh`` falls back
        to a sibling default that PRD §5 guarantees will never exist beside it,
        so every non-disk-pressure Pass-1 lane reset fails inside the script and
        counts as *preserved* while reclaiming nothing — the accrete-to-ENOSPC
        class the wrapper exists to prevent, reached from the opposite branch.
        Degraded is acceptable (orphan removal and disk-pressure target rm still
        work); degraded and SILENT is not, so the mode must be attributable from
        dark-factory's own logs rather than inferred from gc.sh's stderr.
        """
        df_dir = df_warm_lane_script_dir()
        df_script = _write_origin_stub(
            df_dir / 'warm-lane-gc.sh', 'dark-factory', call_log,
        )
        git_ops = _make_git_ops(project_root)
        missing_seed = project_root / 'scripts' / 'seed-warm-lane.sh'

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            await git_ops._run_warm_lane_gc_reclaim()

        warnings = _records(caplog, logging.WARNING)
        matching = [
            m for m in warnings
            if str(missing_seed) in m and str(df_script) in m
        ]
        assert matching, (
            'A dark-factory gc copy running against a project with no seed '
            'script must WARN naming the missing path and the script that '
            'cannot reset lanes without it.\n'
            f'  expected missing seed: {missing_seed}\n'
            f'  expected script: {df_script}\n'
            f'  captured WARNINGs: {warnings!r}'
        )
        assert _origins(call_log) == ['dark-factory'], (
            'The warning must not suppress the reclaim — orphan removal and '
            'disk-pressure target rm still work, so this is degraded, not dead; '
            f'call log: {_origins(call_log)!r}'
        )

    async def test_project_origin_with_no_seed_script_is_not_warned_about(
        self,
        project_root: Path,
        call_log: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A PROJECT gc copy with no sibling seed script is the project's business.

        The warning is scoped to the ``'dark-factory'`` origin on purpose.  When
        the project's own ``scripts/warm-lane-gc.sh`` is running, whatever it
        defaults ``SEED_SCRIPT`` to is that project's arrangement — dark-factory
        has no standing to call it broken, and warning on every reclaim of a
        project that deliberately runs GC without lane reseeding would be pure
        noise.  Pins that the new WARNING did not become unconditional.
        """
        _write_origin_stub(
            project_root / 'scripts' / 'warm-lane-gc.sh', 'project', call_log,
        )
        git_ops = _make_git_ops(project_root)

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            await git_ops._run_warm_lane_gc_reclaim()

        seed_warnings = [
            m for m in _records(caplog, logging.WARNING) if 'seed script' in m
        ]
        assert not seed_warnings, (
            'A project-origin gc copy must not be warned about its own '
            f'seed-script arrangement; got {seed_warnings!r}'
        )


@pytest.mark.asyncio
class TestNeitherLocationIsLoud:
    """B8 — "no implementation at either location" is never a silent no-op.

    Pre-relocation each wrapper logged ``logger.debug('<method>: script absent
    at %s — no-op')`` naming ONE path. After the relocation there are two
    candidate locations, so a DEBUG line naming one of them is actively
    misleading: an operator greping for why GC stopped would see nothing at
    default verbosity and no hint that a second location was even searched.
    dark-factory now always ships its own copy, so "neither location" can only
    mean a genuinely broken deployment — rare in production, never routine
    noise, and therefore a WARNING.
    """

    @pytest.mark.parametrize('wrapper', WRAPPER_PARAMS)
    async def test_sentinel_is_unchanged(
        self,
        wrapper: Wrapper,
        project_root: Path,
        df_warm_lane_script_dir,
    ) -> None:
        """Neither location → each wrapper returns its documented sentinel.

        Parametrized rather than looped: the fail-soft sentinel contract is
        per-wrapper (127 / None / False, with the ``type(...) is type(...)``
        check making ``False`` vs ``0`` load-bearing), so a break in one must
        not stop the other five from being checked or hide them from the
        failure report.
        """
        df_warm_lane_script_dir(project_root.parent / 'absent-df-dir')
        git_ops = _make_git_ops(project_root)

        result = await wrapper.invoke(git_ops)

        assert result == wrapper.sentinel and type(result) is type(
            wrapper.sentinel,
        ), (
            f'{wrapper.method} must degrade to {wrapper.sentinel!r} when no '
            f'implementation exists at either location (fail-soft contract '
            f'preserved byte-for-byte); got {result!r}'
        )

    @pytest.mark.parametrize('wrapper', WRAPPER_PARAMS)
    async def test_warning_names_both_tried_paths(
        self,
        wrapper: Wrapper,
        project_root: Path,
        df_warm_lane_script_dir,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The WARNING names the project path AND the dark-factory path."""
        df_dir = df_warm_lane_script_dir(project_root.parent / 'absent-df-dir')
        git_ops = _make_git_ops(project_root)
        project_path = project_root / 'scripts' / wrapper.script
        df_path = df_dir / wrapper.script

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            await wrapper.invoke(git_ops)

        warnings = _records(caplog, logging.WARNING)
        matching = [
            m for m in warnings
            if str(project_path) in m and str(df_path) in m
        ]
        assert matching, (
            f'{wrapper.method} must WARN naming BOTH tried paths so a broken '
            f'deployment is diagnosable from one log line.\n'
            f'  expected both: {project_path}\n'
            f'             and: {df_path}\n'
            f'  captured WARNINGs: {warnings!r}'
        )

    @pytest.mark.parametrize('wrapper', WRAPPER_PARAMS)
    async def test_no_debug_only_absent_line_remains(
        self,
        wrapper: Wrapper,
        project_root: Path,
        df_warm_lane_script_dir,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The old single-path DEBUG line is gone, not merely supplemented."""
        df_warm_lane_script_dir(project_root.parent / 'absent-df-dir')
        git_ops = _make_git_ops(project_root)

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            await wrapper.invoke(git_ops)

        stale = [m for m in _records(caplog, logging.DEBUG) if 'script absent at' in m]
        assert not stale, (
            f'{wrapper.method} still emits the pre-relocation single-path DEBUG '
            f'line, which names only one of two searched locations: {stale!r}'
        )


@pytest.mark.asyncio
class TestPreResolutionGuardsStillWinTheRace:
    """Guards that run BEFORE resolution keep their level, message and ordering.

    Attributability is the point: a skipped reclaim must stay blamed on the
    held merge-verify lease or the unmounted pool, never misreported as a
    missing script. Both guards are asserted to short-circuit *without*
    emitting the new both-paths WARNING.
    """

    async def test_merge_verify_lease_defers_before_resolution(
        self,
        project_root: Path,
        df_warm_lane_script_dir,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A held lease still defers via its INFO line, with no both-paths WARNING."""
        df_dir = df_warm_lane_script_dir(project_root.parent / 'absent-df-dir')
        git_ops = _make_git_ops(project_root)
        monkeypatch.setattr(git_ops, '_merge_verify_lease_active', lambda: True)

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            result = await git_ops._run_warm_lane_gc_reclaim()

        assert result == 127
        infos = _records(caplog, logging.INFO)
        assert any('merge-verify in flight' in m for m in infos), (
            f'The lease deferral must stay attributable at INFO; got {infos!r}'
        )
        warnings = _records(caplog, logging.WARNING)
        assert not [
            m for m in warnings if str(df_dir / 'warm-lane-gc.sh') in m
        ], (
            'A lease-deferred reclaim must NOT be misreported as a missing '
            f'script — the lease guard runs first by design; got {warnings!r}'
        )

    async def test_pool_storage_absent_refuses_before_resolution(
        self,
        tmp_path: Path,
        df_warm_lane_script_dir,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Absent pool storage still refuses thin via its own WARNING."""
        # No .pool-root sentinel: pool_in_use() and not pool_storage_present().
        root = tmp_path / 'proj'
        (root / '.worktrees').mkdir(parents=True)
        df_dir = df_warm_lane_script_dir(tmp_path / 'absent-df-dir')
        git_ops = _make_git_ops(root)

        with caplog.at_level(logging.DEBUG, logger=GIT_OPS_LOGGER):
            result = await git_ops._run_thin_warm_lane(git_ops.worktree_base / 'lane')

        assert result == 127
        warnings = _records(caplog, logging.WARNING)
        assert any('pool storage absent/unmounted' in m for m in warnings), (
            f'The pool-storage refusal must stay attributable; got {warnings!r}'
        )
        assert not [
            m for m in warnings if str(df_dir / 'thin-warm-lane.sh') in m
        ], (
            'An unmounted pool must NOT be misreported as a missing script — '
            f'the pool-storage guard runs first by design; got {warnings!r}'
        )
