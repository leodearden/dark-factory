"""Real-kernel Landlock enforcement-matrix suite (test_sandbox_enforcement_matrix).

Pins the 12 rows of ``plans/os-sandbox-worktree-containment-prd.md``'s
§Enforcement matrix (task alpha4; D9; INV-1 machine-pin of the §Write-set
contract) against the REAL production chain: ``compute_write_set()`` ->
``build_landlock_command()`` -> ``landlock_exec.py``, driven on a real
landlock-capable kernel (skip-gated elsewhere).

This is a pure characterization / machine-pin suite: no production code
changes ship with it. Every row already passes against the landed
alpha2 (``compute_write_set``) / alpha3 (workflow.py call-site wiring) /
2970 (``~/.claude`` grant narrowing) stack — the "implementation under
test" is the existing sandbox stack, not new code. The RED->GREEN here is
driven by the shared test harness (the ``landlock_matrix_scaffold``
fixture + ``_run_sandboxed``/``_assert_denied`` helpers) growing per row
group, one PRD row group per test-file step pair.

Modeled on ``test_landlock.py``'s ``TestLandlockEnforcement`` /
``TestLandlockClaudeHomeNarrowing``: class-level skip-if-no-landlock,
``/var/tmp`` scaffolding, the ``_reset_landlock_probe`` autouse fixture,
and driving ``build_landlock_command`` directly (not the backend-agnostic
``wrap_command`` dispatcher).

Everything is built under ``/var/tmp``, never ``/tmp``: ``landlock_exec.py``
blanket-grants ``/tmp`` (``FS_V1_ALL``) for agent scratch, so a scaffold
placed under ``/tmp`` would be writable regardless of the ruleset under
test — silently nullifying every denial row (see ``test_landlock.py``'s
matching rationale at its ``TestLandlockEnforcement`` docstring).
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from orchestrator.agents.landlock import (
    _reset_probe as _landlock_reset_probe,
)
from orchestrator.agents.landlock import build_landlock_command, is_landlock_available
from orchestrator.agents.write_set import compute_write_set

# subprocess timeout for each sandboxed inner command — comfortably under
# this suite's 60s per-test pytest-timeout even accounting for scaffold
# setup (real git init/worktree-add), matching test_landlock.py's own
# bounded subprocess.run calls.
_RUN_TIMEOUT = 30

# Denial rows accept EITHER EACCES (landlock) or EROFS (bwrap) as a
# permission-error-class signal (PRD D9: denial errno is backend-specific;
# assertions must say "permission error", not a single errno). This suite
# only ever drives the landlock backend directly (build_landlock_command),
# so only EACCES-shaped messages are actually reachable here, but the
# tolerant token list keeps the shared helper's contract aligned with the
# PRD wording rather than hardcoding one backend's errno.
_PERMISSION_SIGNALS = (
    'Permission denied',
    'Read-only file system',
    'Operation not permitted',
    'EACCES',
    'EROFS',
)


@pytest.fixture(autouse=True)
def _reset_landlock_probe():
    """Reset cached probe before and after each test (mirrors test_landlock.py)."""
    _landlock_reset_probe()
    yield
    _landlock_reset_probe()


_VAR_TMP_SKIP_REASON = '/var/tmp not writable in this sandbox'


@functools.cache
def _var_tmp_writable() -> bool:
    """Probe whether /var/tmp is actually usable for scratch dirs in this process.

    Catches ANY OS-level refusal, not just permission denial: agent sandboxes
    deny the write via a syscall filter (EACCES) even though the directory's
    own mode (1777) permits it, minimal containers may not ship /var/tmp at all
    (ENOENT), and a read-only bind mount surfaces as EROFS. All three must
    degrade to a skip — an escaping OSError here would abort collection of the
    whole module, the exact failure mode these guards exist to prevent. Must be
    an actual write attempt rather than an os.access/stat-mode check.

    NOTE: duplicated verbatim in test_landlock.py and in
    fused-memory/tests/reconciliation/test_recon_sandbox_guard.py — no shared
    test-helper module spans both packages, so keep the three copies in sync.
    """
    try:
        probe = tempfile.mkdtemp(dir='/var/tmp')
    except OSError:
        return False
    shutil.rmtree(probe, ignore_errors=True)
    return True


def _skip_var_tmp() -> bool:
    """Whether /var/tmp-dependent tests should be skipped in this environment.

    Under ``DF_REQUIRE_SANDBOX_TESTS=1`` — set by CI jobs known to have a
    writable /var/tmp and a landlock-capable kernel — an unwritable /var/tmp is
    an environment regression, not a reason to skip: quietly dropping all 12
    enforcement-matrix rows would leave the suite green while every denial
    assertion stopped running. Fail loudly there instead (mirrors
    test_landlock.py's copy).
    """
    if _var_tmp_writable():
        return False
    if os.environ.get('DF_REQUIRE_SANDBOX_TESTS') == '1':
        pytest.fail(
            f'DF_REQUIRE_SANDBOX_TESTS=1 but {_VAR_TMP_SKIP_REASON}: refusing to '
            'silently skip the real-kernel sandbox enforcement matrix.',
            pytrace=False,
        )
    return True


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in ``cwd``, returning stripped stdout.

    Raises ``RuntimeError`` (naming the failed command) on non-zero exit —
    every call site here is scaffold setup or a post-run assertion that is
    expected to succeed outside the sandbox (reads/writes as the real
    test-runner user, never itself sandboxed).
    """
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'git {args} failed in {cwd}: {result.stderr}')
    return result.stdout.strip()


@dataclass
class _Scaffold:
    """A realistic linked-worktree environment for one enforcement-matrix row,
    built entirely under ``/var/tmp`` by the ``landlock_matrix_scaffold``
    fixture (grown incrementally alongside the row groups that need each
    field).
    """

    base: Path
    main: Path
    worktree: Path
    name: str
    home: Path
    main_sha: str
    task_meta: Path
    sibling_worktree: Path
    other_task_meta: Path
    plan_symlink: Path
    plan_target: Path
    uv_cache: Path
    uv_data: Path
    claude_fleet: Path
    settings_json: Path


def _run_sandboxed(
    scaffold: _Scaffold, inner_cmd: list[str], *, cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run ``inner_cmd`` landlock-sandboxed, mirroring workflow.py:9911-9914
    exactly: derive the write-set from the scaffold's worktree/home via the
    real ``compute_write_set()``, feed its ``writable_paths()`` into
    ``build_landlock_command()`` as ``writable_extras`` (never a hand-rolled
    path list — INV-5), and run with ``HOME`` redirected to the scaffold's
    hermetic fake home.
    """
    write_set = compute_write_set(scaffold.worktree, home=scaffold.home)
    cmd = build_landlock_command(
        inner_cmd, scaffold.worktree, [],
        writable_extras=[str(p) for p in write_set.writable_paths()],
    )
    return subprocess.run(
        cmd,
        cwd=str(cwd or scaffold.worktree),
        env={**os.environ, 'HOME': str(scaffold.home)},
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT,
    )


def _assert_denied(result: subprocess.CompletedProcess, *, side_effect_verified: bool) -> None:
    """Assert ``result`` represents a denied write.

    Primary signal (mandatory): non-zero exit. Secondary signal: EITHER a
    permission-error-class token appears in stderr, OR the caller has
    already verified the deterministic side effect itself (target absent /
    ref byte-unchanged) and passes ``side_effect_verified=True`` — per PRD
    D9 the denial errno is backend-specific, so the side-effect check is
    the robust primary pin and the stderr token match is a tolerant bonus.
    """
    assert result.returncode != 0, (
        f'expected denial (non-zero exit); got 0. '
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    permission_signal = any(tok in result.stderr for tok in _PERMISSION_SIGNALS)
    assert permission_signal or side_effect_verified, (
        f'expected a permission-error-class signal in stderr or a '
        f'caller-verified side effect; stderr={result.stderr!r}'
    )


@pytest.fixture
def landlock_matrix_scaffold():
    """Build a realistic linked-worktree environment under ``/var/tmp``.

    BASE layout (grown per row group across this suite's paired impl
    steps): a real ``git init`` main repo with one seed commit and
    repo-local identity; a real linked worktree created via
    ``git worktree add -b task/<name> <base>/<name> main`` (so its
    ``.git`` gitdir file is exactly what ``compute_write_set`` parses in
    production); ``<base>/.task-meta/<name>/`` (the writable task-meta
    carve-out, via the same path shape ``TaskArtifacts.meta_root_for``
    owns); a SIBLING linked worktree + its own (non-writable-by-contract)
    ``.task-meta/`` dir, standing in for a second in-flight task on the
    same host (rows 4/5's denial targets — pre-created so their parent
    exists and the denial is a true landlock EACCES, not ENOENT); a
    hermetic fake ``HOME`` with ``~/.cache/uv/`` seeded so the uv-cache
    carve-out is grantable at wrap time; and the git-row setup — auto-gc/
    maintenance disabled on the shared main repo (PRD α5 corollary, mirrors
    ``GitOps.disable_shared_repo_auto_maintenance``) plus one staged
    change in the worktree, ready for a row's own ``git commit``; a
    ``<worktree>/.task/plan.json`` symlink into ``<base>/.task-meta/<name>/
    plan.json`` (the target itself is left ABSENT — row 7's child write is
    what creates it, mirroring ``TaskArtifacts.ensure_lane_plan_symlink``'s
    real production symlink shape); ``~/.local/share/uv/`` pre-created
    as an empty, non-writable-by-contract sibling of the (writable)
    ``~/.cache/uv/``, so row 8's denial is a true EACCES rather than ENOENT;
    ``~/.claude/fleet/`` pre-created so the ``claude_fleet`` carve-out is
    grantable at wrap time (row 10 — an absent fleet/ dir would make
    landlock_exec's ``_add_path`` skip the grant and wrongly deny); and
    ``~/.claude/settings.json`` seeded with known JSON content (row 9 — a
    file directly under ``~/.claude/`` rather than under the ``fleet/``
    carve-out, so the write is denied and the seed content lets the test
    assert it stayed byte-unchanged).

    Built under ``/var/tmp`` (never ``/tmp`` — see module docstring) via
    ``tempfile.mkdtemp``, torn down with ``shutil.rmtree`` regardless of
    test outcome.
    """
    if _skip_var_tmp():
        # A skipif *decorator* can't live on a fixture (pytest treats marks
        # on fixture functions as inert — PytestRemovedIn9Warning, hard
        # error under pytest 9), so the equivalent guard has to run inside
        # the fixture body. This single check covers all 12
        # TestSandboxEnforcementMatrix rows, since every row depends on
        # this fixture.
        pytest.skip(_VAR_TMP_SKIP_REASON)
    base = Path(tempfile.mkdtemp(prefix='landlock-matrix-', dir='/var/tmp'))
    try:
        main = base / 'main'
        main.mkdir()
        _git(['init', '-b', 'main'], main)
        _git(['config', 'user.email', 'landlock-matrix@test.local'], main)
        _git(['config', 'user.name', 'Landlock Matrix Test'], main)
        # PRD α5 corollary (D2): background auto-gc/maintenance under the
        # narrow shared-.git write-set would fail benignly but noisily, so
        # orchestrator-managed shared repos disable it repo-locally. Applied
        # unconditionally here — harmless for every row (none of them come
        # close to gc.auto's loose-object threshold) and is exactly what row
        # 12 pins the absence-of-noise property against.
        _git(['config', 'gc.auto', '0'], main)
        _git(['config', 'maintenance.auto', 'false'], main)
        (main / 'README.md').write_text('seed\n')
        _git(['add', '-A'], main)
        _git(['commit', '-m', 'seed commit'], main)
        main_sha = _git(['rev-parse', 'HEAD'], main)

        name = 'wt-a'
        worktree = base / name
        _git(['worktree', 'add', '-b', f'task/{name}', str(worktree), 'main'], main)

        other_name = 'wt-b'
        sibling_worktree = base / other_name
        _git(
            ['worktree', 'add', '-b', f'task/{other_name}', str(sibling_worktree), 'main'],
            main,
        )

        task_meta = base / '.task-meta' / name
        task_meta.mkdir(parents=True)
        other_task_meta = base / '.task-meta' / other_name
        other_task_meta.mkdir(parents=True)

        home = base / 'home'
        home.mkdir()
        uv_cache = home / '.cache' / 'uv'
        uv_cache.mkdir(parents=True)
        uv_data = home / '.local' / 'share' / 'uv'
        uv_data.mkdir(parents=True)

        # Row 10: ~/.claude/fleet/ must already EXIST at wrap time — landlock_
        # exec's _add_path skips granting a non-existent writable dir, which
        # would silently turn this happy-path row into a wrongful denial.
        claude_fleet = home / '.claude' / 'fleet'
        claude_fleet.mkdir(parents=True)
        # Row 9: ~/.claude/settings.json is a file directly under ~/.claude/
        # (NOT under the fleet/ carve-out), so it is outside the write set —
        # seeded with known content so the test can assert it stays
        # byte-unchanged after the denied write.
        settings_json = home / '.claude' / 'settings.json'
        settings_json.write_text(json.dumps({'orig': True}))

        # One staged change so a row's `git commit` has something to
        # commit (rows 2/12 — each test gets its own fresh scaffold
        # instance, function-scoped, so there is no cross-row contention
        # over this single staged file).
        (worktree / 'staged.txt').write_text('staged content\n')
        _git(['add', 'staged.txt'], worktree)

        # Row 7: <worktree>/.task/plan.json -> <base>/.task-meta/<name>/
        # plan.json. The target's parent (task_meta, above) already exists
        # and is a writable carve-out; the target FILE itself is left
        # absent — row 7's sandboxed child write is what creates it.
        lane_task_dir = worktree / '.task'
        lane_task_dir.mkdir(parents=True, exist_ok=True)
        plan_target = task_meta / 'plan.json'
        plan_symlink = lane_task_dir / 'plan.json'
        os.symlink(plan_target, plan_symlink)

        yield _Scaffold(
            base=base, main=main, worktree=worktree, name=name, home=home,
            main_sha=main_sha, task_meta=task_meta,
            sibling_worktree=sibling_worktree, other_task_meta=other_task_meta,
            plan_symlink=plan_symlink, plan_target=plan_target,
            uv_cache=uv_cache, uv_data=uv_data,
            claude_fleet=claude_fleet, settings_json=settings_json,
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.skipif(
    not is_landlock_available(),
    reason='landlock not supported on this kernel',
)
class TestSandboxEnforcementMatrix:
    """The 12 §Enforcement-matrix rows, each driven against the real
    ``compute_write_set()`` -> ``build_landlock_command()`` ->
    ``landlock_exec`` production chain via the ``landlock_matrix_scaffold``
    fixture and ``_run_sandboxed`` helper (grown in the paired impl steps).
    """

    # -- Group 1: base filesystem (rows 1/3/11) --------------------------

    def test_row01_worktree_write_allowed(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.worktree / 'src' / 'x.py'
        result = _run_sandboxed(
            scaffold,
            ['/bin/sh', '-c', f'mkdir -p {target.parent} && printf X > {target}'],
        )
        assert result.returncode == 0, result.stderr
        assert target.read_text() == 'X'

    def test_row03_main_canary_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.main / 'CANARY'
        result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {target}'])
        assert not target.exists()
        _assert_denied(result, side_effect_verified=True)
        # Representative denial row: additionally pin an actual
        # permission-class token in stderr, so the suite proves landlock
        # itself — not an unrelated wrapper crash — produced the denial.
        # (_assert_denied's stderr branch is otherwise never the deciding
        # signal, since every denial caller passes side_effect_verified=True.)
        assert any(tok in result.stderr for tok in _PERMISSION_SIGNALS), (
            f'expected a permission-error-class token in stderr; stderr={result.stderr!r}'
        )

    def test_row11_tmp_scratch_allowed(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        scratch = Path(f'/tmp/landlock-matrix-row11-{uuid.uuid4().hex}')
        try:
            result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {scratch}'])
            assert result.returncode == 0, result.stderr
            assert scratch.read_text() == 'X'
        finally:
            scratch.unlink(missing_ok=True)

    # -- Group 2: neighbor denial (rows 4/5) ------------------------------

    def test_row04_sibling_worktree_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.sibling_worktree / 'f'
        result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {target}'])
        assert not target.exists()
        _assert_denied(result, side_effect_verified=True)

    def test_row05_other_task_meta_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.other_task_meta / 'x'
        # Pin the carve-out boundary independently of kernel enforcement: the
        # sibling task's meta dir must never appear in the computed write
        # set. Without this, a future regression that widened the task_meta
        # carve-out to the whole `.task-meta/` parent (rather than just
        # `.task-meta/<name>/`) would stay green here purely because of the
        # kernel-level side-effect check below.
        write_set = compute_write_set(scaffold.worktree, home=scaffold.home)
        assert str(scaffold.other_task_meta) not in [
            str(p) for p in write_set.writable_paths()
        ]
        result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {target}'])
        assert not target.exists()
        _assert_denied(result, side_effect_verified=True)

    # -- Group 3: git operations (rows 2/6/12) ----------------------------

    def test_row02_worktree_commit_allowed(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        result = _run_sandboxed(scaffold, ['git', 'commit', '-m', 'row2 test commit'])
        assert result.returncode == 0, result.stderr
        new_head = _git(['rev-parse', 'HEAD'], scaffold.worktree)
        assert new_head != scaffold.main_sha
        branch_ref = _git(['rev-parse', f'refs/heads/task/{scaffold.name}'], scaffold.main)
        assert branch_ref == new_head

    def test_row06_update_ref_main_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        result = _run_sandboxed(
            scaffold, ['git', 'update-ref', 'refs/heads/main', scaffold.main_sha],
        )
        after = _git(['rev-parse', 'refs/heads/main'], scaffold.main)
        assert after == scaffold.main_sha
        _assert_denied(result, side_effect_verified=True)

    def test_row12_commit_clean_with_gc_disabled(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        result = _run_sandboxed(scaffold, ['git', 'commit', '-m', 'row12 test commit'])
        assert result.returncode == 0, result.stderr
        noise_tokens = (
            'maintenance', 'gc.log', 'Permission denied',
            'Read-only file system', 'Operation not permitted',
        )
        for token in noise_tokens:
            assert token not in result.stderr, result.stderr

    # -- Group 4: child-tree + uv-cache (rows 7/8) ------------------------

    def test_row07_child_writes_task_meta_via_symlink(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        inner = [
            '/bin/sh', '-c',
            f"/bin/sh -c 'printf PLAN > {scaffold.plan_symlink}'",
        ]
        result = _run_sandboxed(scaffold, inner)
        assert result.returncode == 0, result.stderr
        assert scaffold.plan_target.read_text() == 'PLAN'

    def test_row08_uv_cache_writable_data_ro(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        probe = uuid.uuid4().hex
        cache_marker = scaffold.uv_cache / probe
        data_marker = scaffold.uv_data / probe
        inner = [
            '/bin/sh', '-c',
            (
                f'(touch {cache_marker} 2>/dev/null && echo cache_ok || echo cache_denied) && '
                f'(touch {data_marker} 2>/dev/null && echo data_ok || echo data_denied) && '
                'echo end'
            ),
        ]
        result = _run_sandboxed(scaffold, inner)
        assert result.returncode == 0, result.stderr
        assert 'end' in result.stdout
        assert 'cache_ok' in result.stdout, result.stdout
        assert cache_marker.exists()
        assert 'data_denied' in result.stdout, result.stdout
        assert not data_marker.exists()

    # -- Group 5: ~/.claude narrowing (rows 9/10) -------------------------

    def test_row09_claude_settings_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        original = json.loads(scaffold.settings_json.read_text())
        poisoned = json.dumps({'pwned': True})
        result = _run_sandboxed(
            scaffold,
            ['/bin/sh', '-c', f"printf '%s' '{poisoned}' > {scaffold.settings_json}"],
        )
        assert json.loads(scaffold.settings_json.read_text()) == original
        _assert_denied(result, side_effect_verified=True)

    def test_row10_claude_fleet_allowed(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.claude_fleet / 'sessions' / 'x'
        result = _run_sandboxed(
            scaffold,
            ['/bin/sh', '-c', f'mkdir -p {target.parent} && printf X > {target}'],
        )
        assert result.returncode == 0, result.stderr
        assert target.read_text() == 'X'
