"""Ruff config/cache resolution must terminate at the worktree boundary.

Task 3922.  ruff resolves BOTH its settings and its cache by walking parent
directories up from each linted file, and a git VCS root does NOT stop that
walk.  A task worktree at ``<parent>/.worktrees/<id>`` whose own root declares
no ``[tool.ruff]`` therefore escapes: the walk reaches the PARENT checkout's
pyproject.toml, so the merge gate reads the parent's rule set (from its
UNCOMMITTED working tree) and shares the parent's ``.ruff_cache`` with every
concurrently-verifying sibling worktree.

The adopted fix has two halves, and this module guards both:

* the CACHE half is FIXED — ``_run_cmd`` threads its ``cwd`` (the worktree
  root, verbatim, for the lint leg) into ``_target_subprocess_env`` so every
  verify spawn gets a worktree-local ``RUFF_CACHE_DIR``.  Measured rule-neutral
  on ruff 0.15.9, hence applied unconditionally.
* the CONFIG half is made LOUD, not fixed — see ``_settings_path_escapes``.  It
  cannot be fixed safely at the gate (a ``--config`` pin aimed at a rule-less
  pyproject silently falls back to ruff's built-in defaults) and it must not be
  fatal (hard-failing would red every stale-based worktree at once).

Sibling guard: ``tests/scripts/test_worktree_ruff_config_boundary.py`` pins the
two ruff measurements the decision rests on.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_verification


@pytest.fixture(autouse=True)
def _clear_escape_latch():
    """The escape latch is MODULE-LEVEL (one record per worktree per process),
    so clear it around every test.

    Each test builds its own tmp_path geometry and therefore its own latch key,
    but pinning that independence here means a future test reusing a path can
    never silently observe another test's suppression.
    """
    verify._RUFF_ESCAPE_REPORTED.clear()
    yield
    verify._RUFF_ESCAPE_REPORTED.clear()



class TestRunCmdThreadsWorktreeIntoRuffCache:
    """``_run_cmd`` must thread its own ``cwd`` into the env builder.

    Step-2's ``worktree=`` parameter is inert until something passes it.  This
    asserts against a REAL spawned subprocess rather than a monkeypatched
    builder on purpose: the defect being fixed is precisely that a value was
    never threaded through, and a mock-based assertion would pass just as
    happily against the un-wired call.
    """

    def test_ruff_cache_dir_visible_to_the_spawned_command(self, tmp_path):
        rc, out, timed_out = asyncio.run(
            verify._run_cmd('echo "$RUFF_CACHE_DIR"', tmp_path, timeout=30)
        )

        assert rc == 0, out
        assert timed_out is False
        assert out.strip() == str(tmp_path / '.ruff_cache')

    def test_caller_env_overlay_still_reaches_the_spawned_command(self, tmp_path):
        # The operator-override contract holds end-to-end, not just in the
        # builder: verify_env wins over the worktree-derived default.
        chosen = str(tmp_path / 'operator-chosen-cache')
        rc, out, _ = asyncio.run(
            verify._run_cmd(
                'echo "$RUFF_CACHE_DIR"',
                tmp_path,
                timeout=30,
                env={'RUFF_CACHE_DIR': chosen},
            )
        )

        assert rc == 0, out
        assert out.strip() == chosen


_RUFF_TABLE = """
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
"""


def _write_project(root: Path, name: str, *, declares_ruff: bool) -> None:
    """Write a pyproject.toml at *root*, with or without a ``[tool.ruff]`` table."""
    body = f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
    if declares_ruff:
        body += _RUFF_TABLE
    (root / 'pyproject.toml').write_text(body)


def _git_init(root: Path) -> None:
    # A VCS root does NOT stop ruff's walk-up — that is the whole point of the
    # defect, so the fixture must be git-initialised for the geometry to be
    # faithful rather than accidentally proving something weaker.
    subprocess.run(
        ['git', 'init', '-q', str(root)],
        check=True, capture_output=True,
    )


@pytest.fixture
def geometry(tmp_path):
    """The M1 geometry: a parent checkout carrying [tool.ruff], plus a task
    worktree at ``<parent>/.worktrees/wt`` holding a REAL .py file.

    Synthetic and per-test on purpose: an assertion bound to the ambient
    checkout would be born green in any worktree whose base already declares
    ``[tool.ruff]`` and would prove nothing.
    """
    parent = (tmp_path / 'parent').resolve()
    worktree = parent / '.worktrees' / 'wt'
    (worktree / 'scripts').mkdir(parents=True)
    _write_project(parent, 'parentproj', declares_ruff=True)
    _git_init(parent)
    _git_init(worktree)
    target = worktree / 'scripts' / 's.py'
    target.write_text('import sys\nimport os\n\nprint(os.getcwd())\n')
    return parent, worktree, target


def _escapes(worktree: Path, target: Path | None = None) -> bool:
    """Compose exactly what production composes: probe, then predicate.

    Deliberately a TEST-LOCAL composition rather than a helper in verify.py.  A
    convenience wrapper living in production code with no production caller can
    drift out of agreement with the real call path — the reporter could change
    its predicate and the wrapper would keep asserting the old behaviour green.
    These two calls are the ones ``_report_ruff_config_escape`` makes.
    """
    return verify._settings_path_escapes(
        verify._ruff_settings_path(worktree, target), worktree,
    )


class TestRuffConfigEscapeDetector:
    """``_ruff_settings_path`` must report WHERE ruff actually resolved its
    settings, and ``_settings_path_escapes`` must say whether that landed
    outside the worktree.

    The config half of the defect is deliberately NOT fixed — it cannot be,
    safely — so the deliverable is that it becomes visible.  These tests pin
    the measurement the diagnostic is built on.

    Probes a REAL file on disk, never ``--stdin-filename``: the task's Details
    warn that stdin input can skip the parent walk-up and false-green.  That
    did not reproduce on ruff 0.15.9, but a real-file probe is correct under
    BOTH behaviours, so the discrepancy never has to be adjudicated here.
    """

    def test_stale_worktree_resolves_the_parent_checkouts_config(self, geometry):
        parent, worktree, target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)

        settings = verify._ruff_settings_path(worktree, target)

        # Not None: a missing ruff must FAIL this guard, never silently skip it
        # (the never-skip doctrine of tests/scripts/test_nonmember_ruff_config.py).
        assert settings is not None, 'ruff did not report a settings path'
        assert settings == parent / 'pyproject.toml'
        assert _escapes(worktree, target) is True
        # and the default probe target (the worktree's own root pyproject)
        # reaches the same verdict, which is the form the detector uses.
        assert _escapes(worktree) is True

    def test_sound_worktree_resolves_its_own_config(self, geometry):
        _parent, worktree, target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=True)

        settings = verify._ruff_settings_path(worktree, target)

        assert settings is not None, 'ruff did not report a settings path'
        assert settings == worktree / 'pyproject.toml'
        assert _escapes(worktree, target) is False
        assert _escapes(worktree) is False

    def test_detector_is_total_and_never_raises(self, tmp_path):
        # A ruff failure, a timeout or an unparseable line must degrade to
        # None/False, never to an exception: the detector is a diagnostic and
        # must be incapable of reddening a verify by itself.
        missing = tmp_path / 'nope'
        assert verify._ruff_settings_path(missing, missing / 'x.py') is None
        assert _escapes(missing) is False


class TestRuffProbeResolution:
    """The probe must resolve ruff deterministically, not via ambient PATH luck."""

    def test_ruff_module_is_importable_for_the_probe(self):
        # Never skip: if this interpreter cannot run ruff the guard above is
        # meaningless, so make that a FAILURE with a legible cause.
        proc = subprocess.run(
            [sys.executable, '-m', 'ruff', '--version'],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f'`{sys.executable} -m ruff` unavailable: {proc.stderr}'
        assert proc.stdout.startswith('ruff ')


def _module_config(**overrides: Any) -> ModuleConfig:
    """A minimal three-leg module config whose lint leg names ruff.

    Module-local rather than imported from a sibling admission test: each of
    those files states the same self-containment rationale, and a conftest.py
    edit would trip verify.py's ``has_conftest``.
    """
    # Annotated dict[str, Any], matching the sibling admission tests: an
    # inferred dict[str, str | bool] makes every ModuleConfig kwarg a pyright
    # error at the ** expansion.
    kwargs: dict[str, Any] = dict(
        prefix='pkg',
        test_command='pytest tests/',
        lint_command='ruff check scripts/',
        type_check_command='pyright',
        concurrent_verify=False,
    )
    kwargs.update(overrides)
    return ModuleConfig(**kwargs)


def _escape_records(caplog):
    # Read the marker EAGERLY, before the filter: inside the comprehension the
    # `and` chain short-circuits whenever no WARNING was emitted, so a missing
    # marker would silently yield [] and let the NEGATIVE test pass vacuously.
    marker = verify._RUFF_ESCAPE_MARKER
    return [
        r for r in caplog.records
        if r.name == 'orchestrator.verify'
        and r.levelno == logging.WARNING
        and marker in r.getMessage()
    ]


async def _verify_with_stubbed_spawn(
    worktree,
    tmp_path,
    *,
    lint_rc: int,
    lint_timed_out: bool = False,
    max_retries: int = 0,
    lint_marker: str = 'ruff',
    **module_overrides: Any,
):
    """Drive the REAL run_verification lint leg with only the SPAWN stubbed.

    The detector itself is left live — it is the thing under test — so this is
    a genuine wiring proof rather than a mock asserting against itself.

    *lint_timed_out* + *max_retries* drive the retry loop, which only re-runs on
    a PURE timeout failure; *module_overrides* reach ``_module_config`` so a
    test can point the lint leg at a non-ruff linter, and *lint_marker* is the
    substring identifying that leg's command to the stub.
    """
    async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        if lint_marker not in cmd:
            return 0, '', False
        return (
            lint_rc,
            ('E501 line too long\n' if lint_rc else ''),
            lint_timed_out,
        )

    config = OrchestratorConfig(
        verify_admission_slots_dir=str(tmp_path / 'slots'),
        verify_admission_task_slots=1,
    )
    with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
        return await run_verification(
            worktree=worktree,
            config=config,
            module_config=_module_config(**module_overrides),
            role='task',
            attempt_id=None,
            max_retries=max_retries,
        )


class TestEscapeIsReportedLoudly:
    """An escaping worktree must produce ONE structured WARNING — and nothing else.

    Deliberately NOT fatal.  Hard-failing an escaping worktree would red every
    stale-based worktree on the host simultaneously; that is the fleet-wide
    outage mode pyproject.toml's ``[tool.ruff]`` block explicitly warns against
    ("a red lint_command … blocks every merge, review checkpoint and main-tip
    sweep repo-wide, on branches carrying no defect at all").  So the escaping
    worktree is still judged on its own lint output, and the diagnostic rides
    alongside as an interpretation layered ON TOP — the same shape as the
    mis-resolved-interpreter record in ``_run_or_skip_timed``.
    """

    @pytest.mark.asyncio
    async def test_escaping_worktree_emits_one_diagnostic(
        self, geometry, tmp_path, caplog,
    ):
        parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            result = await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=0)

        records = _escape_records(caplog)
        assert len(records) == 1, f'expected exactly one record, got {records}'
        message = records[0].getMessage()
        # (i) the worktree root and (ii) the FOREIGN settings path actually
        # resolved — asserted as the measured paths, not as prose.
        assert str(worktree) in message
        assert str(parent / 'pyproject.toml') in message
        # (iii) the remediation — asserted as the named CONSTANT, never as a
        # literal phrase: what has behavioural weight is that the operator is
        # told what to do, not the particular wording, and pinning prose makes
        # a copy edit red for no behavioural reason.
        assert verify._RUFF_ESCAPE_REMEDIATION in message

        # ...and the diagnostic did NOT red the leg: a green lint stays green.
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_diagnostic_does_not_alter_the_lint_verdict(
        self, geometry, tmp_path, caplog,
    ):
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            result = await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=1)

        assert len(_escape_records(caplog)) == 1
        # The escaping worktree is still judged on its OWN lint output.
        assert result.passed is False
        assert 'E501 line too long' in result.lint_output

    @pytest.mark.asyncio
    async def test_sound_worktree_emits_nothing(self, geometry, tmp_path, caplog):
        # The NEGATIVE case is what keeps the signal rare and therefore
        # meaningful: a worktree resolving its own config must be silent.
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=True)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            result = await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=0)

        assert _escape_records(caplog) == []
        assert result.passed is True


@pytest.fixture
def probe_spy():
    """Count probes while leaving the LIVE helper running underneath.

    Module-scoped rather than class-scoped because two classes need it —
    ``TestEscapeProbeIsGated`` (the probe must not fire more often than the
    latch allows) and ``TestLatchIsKeyedOnTheWorktreeBASE`` (it must fire
    AGAIN when the base changes).  pytest resolves module-level fixtures for
    tests in nested classes, so the requesting tests are unchanged.

    Delegates rather than replaces on purpose: the records the surrounding
    assertions read stay genuine measurements, so a test can pin the probe
    COUNT and the emitted RECORD in the same run without either one being a
    mock asserting against itself.
    """
    calls: list[Path] = []
    real = verify._ruff_settings_path

    def counting(worktree, target=None):
        calls.append(Path(worktree))
        return real(worktree, target)

    with patch('orchestrator.verify._ruff_settings_path', new=counting):
        yield calls


class TestEscapeProbeIsGated:
    """The probe must fire on a RUFF LINT leg and nowhere else.

    Both properties below are invisible to the tests above, because the latch
    collapses any number of extra probes into a single record: deleting the
    ``label == 'lint' and 'ruff' in config_cmd`` gate, or re-scoping the latch
    per attempt, leaves them green while the cost (one blocking subprocess per
    extra leg / per retry) is real.  So these assert on the PROBE COUNT, via a
    spy that delegates to the live helper rather than replacing it — the record
    the other tests read stays genuine.
    """

    @pytest.mark.asyncio
    async def test_non_ruff_lint_command_never_probes(
        self, geometry, tmp_path, caplog, probe_spy,
    ):
        # An escaping worktree, but a lint leg that does not run ruff: the
        # question the probe answers ("which ruff rule set ran?") is not even
        # being asked, so asking it would spend a subprocess on nothing and
        # report an escape that no gate is reading.
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            result = await _verify_with_stubbed_spawn(
                worktree, tmp_path, lint_rc=0,
                lint_command='flake8 .', lint_marker='flake8',
            )

        assert probe_spy == [], f'probed on a non-ruff lint leg: {probe_spy}'
        assert _escape_records(caplog) == []
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_retries_do_not_respawn_the_probe(
        self, geometry, tmp_path, caplog, probe_spy,
    ):
        # The stated reason the latch exists. A pure-timeout lint failure with
        # max_retries=1 runs the lint leg TWICE; the probe must still fire once.
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            result = await _verify_with_stubbed_spawn(
                worktree, tmp_path, lint_rc=124, lint_timed_out=True,
                max_retries=1,
            )

        assert result.timed_out is True, 'the retry path was not exercised'
        assert len(probe_spy) == 1, f'probe respawned across retries: {probe_spy}'
        assert len(_escape_records(caplog)) == 1

    @pytest.mark.asyncio
    async def test_second_module_on_one_worktree_does_not_reprobe(
        self, geometry, tmp_path, caplog, probe_spy,
    ):
        # ``verify_all_modules`` gathers one ``run_verification`` per module
        # config against the SAME worktree. A latch scoped to a single call
        # would emit the whole multi-line WARNING once per module.
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=0)
            await _verify_with_stubbed_spawn(
                worktree, tmp_path, lint_rc=0, prefix='other',
            )

        assert len(probe_spy) == 1, f'probed once per module: {probe_spy}'
        assert len(_escape_records(caplog)) == 1


class TestLatchIsKeyedOnTheWorktreeBASE:
    """A recycled worktree PATH must be re-measured when its BASE changes.

    Falsifies the premise the latch was ORIGINALLY built on, quoted verbatim
    from the comment above ``_RUFF_ESCAPE_REPORTED`` as it stood before this
    guard landed: "The answer is a property of the worktree's base, which
    cannot change under a running orchestrator."  It can.  Worktree PATHS are
    recycled across bases within one orchestrator process, by three distinct
    mechanisms:

    * warm lanes — fixed directories ``<worktree_base>/_lane-<k>`` handed out
      task after task by ``warm_lane_pool.py::WarmLanePool.try_acquire`` /
      ``.release``, with ``git_ops.py::GitOps._reset_warm_lane`` re-pointing
      the SAME dir at a different branch and commit via
      ``git checkout -f -B <branch> <commit>``;
    * the single persistent merge-verify worktree
      (``git_ops.py::PERSISTENT_MERGE_WORKTREE_NAME``, a fixed
      ``<worktree_base>/_merge-verify``), reset per merge commit;
    * per-task directories reused when a task id recurs
      (``git_ops.py::GitOps.create_worktree``'s reuse path).

    So the path is not an identity for the question being asked.  Keying the
    latch on it alone means the FIRST task to occupy a lane decides, for the
    rest of the fleet-deploy window, whether every later task on that lane is
    measured at all.  Both directions are pinned below because they fail
    differently.
    """

    @pytest.mark.asyncio
    async def test_sound_then_escaping_base_is_reported(
        self, geometry, tmp_path, caplog,
    ):
        # The fleet-critical direction. A lane whose first task had a sound
        # base latches SILENT, and every escaping task that later lands on
        # that same lane inherits the silence — the diagnostic is suppressed
        # for exactly the geometry it exists to report.
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=True)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=0)
            assert _escape_records(caplog) == [], 'a sound base must stay silent'

            # The lane is re-pointed at a different base: same path, different
            # root config. In production this is _reset_warm_lane's
            # ``git checkout -f -B``; in-place rewrite is its local equivalent.
            _write_project(worktree, 'wtproj', declares_ruff=False)
            await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=0)

        records = _escape_records(caplog)
        assert len(records) == 1, (
            f'the escaping base on the recycled path was not reported: {records}'
        )
        assert str(worktree) in records[0].getMessage()

    @pytest.mark.asyncio
    async def test_escaping_then_sound_base_is_remeasured(
        self, geometry, tmp_path, caplog, probe_spy,
    ):
        # The opposite direction, and the reason this test asserts the PROBE
        # COUNT rather than the record count: a latched key emits nothing
        # either way, so "no second record" passes VACUOUSLY under the
        # path-only key. The observable difference is that the second run must
        # actually MEASURE the new base before concluding it is sound.
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=0)
            assert len(_escape_records(caplog)) == 1

            _write_project(worktree, 'wtproj', declares_ruff=True)
            await _verify_with_stubbed_spawn(worktree, tmp_path, lint_rc=0)

        assert len(probe_spy) == 2, (
            f'the new base was never measured; the latch suppressed it: {probe_spy}'
        )
        # ...and having measured it, it is sound, so no SECOND record appears.
        assert len(_escape_records(caplog)) == 1


class TestEscapeLatchKey:
    """Unit-pin ``_ruff_escape_latch_key``'s four load-bearing properties.

    The end-to-end pair above proves the latch re-measures a recycled path, but
    it cannot localise WHICH property of the key delivered that: a refactor
    could satisfy it while quietly dropping a config spelling, or by keying on
    an mtime stat that re-probes on every untouched run.  These assert the
    properties directly.

    Keys are compared with ``==`` only, never destructured, so the tuple layout
    stays free to change.
    """

    def test_key_is_stable_when_nothing_changes(self, tmp_path):
        # The property that keeps the per-module and per-retry dedup working —
        # and the reason the fingerprint is a CONTENT digest rather than an
        # mtime/size stat: a rewrite with identical bytes is not a base change.
        _write_project(tmp_path, 'wtproj', declares_ruff=False)
        first = verify._ruff_escape_latch_key(tmp_path)

        assert verify._ruff_escape_latch_key(tmp_path) == first

        _write_project(tmp_path, 'wtproj', declares_ruff=False)
        assert verify._ruff_escape_latch_key(tmp_path) == first, (
            'a byte-identical rewrite changed the key; the fingerprint is '
            'reading metadata rather than content'
        )

    @pytest.mark.parametrize('name', ['.ruff.toml', 'ruff.toml', 'pyproject.toml'])
    def test_each_root_config_spelling_discriminates(self, tmp_path, name):
        # Asserted once per SPELLING because ruff's per-directory precedence is
        # .ruff.toml > ruff.toml > pyproject.toml[tool.ruff]: a base carrying
        # only a root ruff.toml halts the walk just as effectively, so a
        # pyproject-only fingerprint would miss that base change entirely.
        absent = verify._ruff_escape_latch_key(tmp_path)
        path = tmp_path / name

        path.write_text('[tool.ruff]\nline-length = 100\n')
        created = verify._ruff_escape_latch_key(tmp_path)
        assert created != absent, f'creating {name} did not change the key'

        path.write_text('[tool.ruff]\nline-length = 120\n')
        edited = verify._ruff_escape_latch_key(tmp_path)
        assert edited != created, f'editing {name} did not change the key'

        path.unlink()
        removed = verify._ruff_escape_latch_key(tmp_path)
        assert removed != edited, f'removing {name} did not change the key'
        assert removed == absent

    def test_key_is_path_sensitive(self, tmp_path):
        # Two lanes can hold byte-identical configs while being different
        # worktrees, so the resolved path stays part of the identity.
        first = tmp_path / 'lane-0'
        second = tmp_path / 'lane-1'
        first.mkdir()
        second.mkdir()
        _write_project(first, 'proj', declares_ruff=True)
        _write_project(second, 'proj', declares_ruff=True)

        assert verify._ruff_escape_latch_key(first) != verify._ruff_escape_latch_key(second)

    def test_missing_worktree_still_yields_a_key(self, tmp_path):
        # TOTAL: this runs on the verify hot path, so it must be structurally
        # incapable of reddening a leg.
        missing = tmp_path / 'nope'

        assert verify._ruff_escape_latch_key(missing) == verify._ruff_escape_latch_key(missing)

    def test_unreadable_root_config_still_yields_a_key(self, tmp_path):
        if os.geteuid() == 0:
            # Explicit rather than an unconditional skip: under a non-root uid
            # this case MUST run, and only root's chmod bypass excuses it.
            pytest.skip('running as root: chmod 000 does not make a file unreadable')
        _write_project(tmp_path, 'wtproj', declares_ruff=True)
        config = tmp_path / 'pyproject.toml'
        config.chmod(0o000)
        try:
            key = verify._ruff_escape_latch_key(tmp_path)
            assert key == verify._ruff_escape_latch_key(tmp_path)
        finally:
            config.chmod(0o644)

        # An unreadable config carries no halt decision, exactly like an absent
        # one, so the two are the same key rather than a spurious re-probe.
        config.unlink()
        assert key == verify._ruff_escape_latch_key(tmp_path)


class TestProbeTargetFallback:
    """A worktree with no root ``pyproject.toml`` must still be measurable.

    That geometry (a project rooted on ruff.toml/setup.cfg, a polyglot repo, a
    branch that deleted the file) is the one MOST likely to escape — no root
    pyproject means certainly no root ``[tool.ruff]`` — and it is exactly where
    a probe aimed at ``<worktree>/pyproject.toml`` gets "No files found under
    the given path", prints no settings line, and goes silent.  Loud over
    silent: fall back to a real .py file, and log the give-up when there is not
    even one.
    """

    def test_falls_back_to_a_real_py_file(self, geometry):
        parent, worktree, _target = geometry
        # No pyproject.toml at the worktree root at all.
        assert not (worktree / 'pyproject.toml').exists()

        assert verify._ruff_probe_target(worktree) == worktree / 'scripts' / 's.py'
        settings = verify._ruff_settings_path(worktree)
        assert settings == parent / 'pyproject.toml'
        assert _escapes(worktree) is True

    def test_root_pyproject_still_wins_when_present(self, geometry):
        _parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=True)

        # The default probe stays the ROOT config question; the .py fallback is
        # a fallback, not a co-equal choice.
        assert verify._ruff_probe_target(worktree) == worktree / 'pyproject.toml'

    def test_fallback_prefers_the_shallowest_file(self, tmp_path):
        # Shallowest-first is load-bearing: a deep file under a workspace
        # MEMBER would resolve that member's own [tool.ruff] and report "no
        # escape" for a reason that has nothing to do with the worktree root.
        (tmp_path / 'member' / 'src').mkdir(parents=True)
        (tmp_path / 'member' / 'src' / 'deep.py').write_text('x = 1\n')
        (tmp_path / 'conftest.py').write_text('x = 1\n')

        assert verify._ruff_probe_target(tmp_path) == tmp_path / 'conftest.py'

    def test_fallback_skips_vendor_and_cache_dirs(self, tmp_path):
        for skipped in ('.venv', '.git', 'node_modules', '.worktrees'):
            (tmp_path / skipped).mkdir()
            (tmp_path / skipped / 'x.py').write_text('x = 1\n')

        assert verify._ruff_probe_target(tmp_path) is None

    def test_give_up_is_logged_at_debug_with_a_reason(self, tmp_path, caplog):
        # Silence must stay DIAGNOSABLE: no settings path, but a DEBUG line
        # naming which of the three give-up reasons fired.
        with caplog.at_level(logging.DEBUG, logger='orchestrator.verify'):
            assert verify._ruff_settings_path(tmp_path) is None

        assert any(
            'probe target missing' in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.DEBUG
        ), [r.getMessage() for r in caplog.records]


class TestProbeBinaryResolution:
    """The probe names WHICH ruff it ran, because it may not be the leg's own.

    The lint leg resolves ruff through the target's toolchain; the cheapest
    probe would be ``sys.executable -m ruff`` — the ORCHESTRATOR's ruff, a
    possibly different build.  The resolution order narrows that gap, and the
    diagnostic states the substitution instead of implying fidelity it lacks.
    """

    def test_prefers_the_worktree_venv_ruff(self, tmp_path):
        local = tmp_path / '.venv' / 'bin'
        local.mkdir(parents=True)
        ruff = local / 'ruff'
        ruff.write_text('#!/bin/sh\nexit 0\n')
        ruff.chmod(0o755)

        assert verify._ruff_probe_binary(tmp_path, {'PATH': '/nonexistent'}) == [str(ruff)]

    def test_falls_back_to_path_then_to_the_orchestrators_module(self, tmp_path):
        on_path = tmp_path / 'bin'
        on_path.mkdir()
        ruff = on_path / 'ruff'
        ruff.write_text('#!/bin/sh\nexit 0\n')
        ruff.chmod(0o755)

        assert verify._ruff_probe_binary(tmp_path, {'PATH': str(on_path)}) == [str(ruff)]
        assert verify._ruff_probe_binary(tmp_path, {'PATH': str(tmp_path / 'empty')}) == [
            sys.executable, '-m', 'ruff',
        ]


class TestProbeDoesNotBlockTheEventLoop:
    """The probe is a blocking ``subprocess.run``; it must not run ON the loop.

    ``verify_all_modules`` gathers one ``run_verification`` per module while
    each verify leg is streaming subprocess output and being wall-clock-timed,
    so a probe executed inline would stall those reads for its whole duration
    (up to ``_RUFF_PROBE_TIMEOUT_S`` = 20s) and could push a concurrent leg over
    its timeout.  Every other potentially-blocking call in verify.py observes
    the same rule; this pins it for the probe.
    """

    @pytest.mark.asyncio
    async def test_the_loop_keeps_running_during_the_probe(self, geometry, tmp_path):
        parent, worktree, _target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=False)
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        def slow_blocking_probe(wt, target=None):
            # Stands in for the real subprocess.run: blocking, and long enough
            # that an on-loop call is unmistakable in the tick count.
            time.sleep(0.25)
            return parent / 'pyproject.toml'

        with patch('orchestrator.verify._ruff_settings_path', new=slow_blocking_probe):
            beat = asyncio.create_task(ticker())
            await asyncio.sleep(0)
            await verify._report_ruff_config_escape(worktree)
            beat.cancel()

        # On-loop the count is ~1; off-loop it is ~50. The threshold is far
        # from both, so this is a structural claim, not a timing race.
        assert ticks > 5, f'event loop was starved during the probe (ticks={ticks})'
