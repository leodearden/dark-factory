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
* the CONFIG half is made LOUD, not fixed — see ``_ruff_config_escapes``.  It
  cannot be fixed safely at the gate (a ``--config`` pin aimed at a rule-less
  pyproject silently falls back to ruff's built-in defaults) and it must not be
  fatal (hard-failing would red every stale-based worktree at once).

Sibling guard: ``tests/scripts/test_worktree_ruff_config_boundary.py`` pins the
two ruff measurements the decision rests on.
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_verification


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


class TestRuffConfigEscapeDetector:
    """``_ruff_settings_path`` must report WHERE ruff actually resolved its
    settings, and ``_ruff_config_escapes`` must say whether that landed outside
    the worktree.

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
        assert verify._ruff_config_escapes(worktree, target) is True
        # and the default probe target (the worktree's own root pyproject)
        # reaches the same verdict, which is the form the detector uses.
        assert verify._ruff_config_escapes(worktree) is True

    def test_sound_worktree_resolves_its_own_config(self, geometry):
        _parent, worktree, target = geometry
        _write_project(worktree, 'wtproj', declares_ruff=True)

        settings = verify._ruff_settings_path(worktree, target)

        assert settings is not None, 'ruff did not report a settings path'
        assert settings == worktree / 'pyproject.toml'
        assert verify._ruff_config_escapes(worktree, target) is False
        assert verify._ruff_config_escapes(worktree) is False

    def test_detector_is_total_and_never_raises(self, tmp_path):
        # A ruff failure, a timeout or an unparseable line must degrade to
        # None/False, never to an exception: the detector is a diagnostic and
        # must be incapable of reddening a verify by itself.
        missing = tmp_path / 'nope'
        assert verify._ruff_settings_path(missing, missing / 'x.py') is None
        assert verify._ruff_config_escapes(missing) is False


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


def _module_config(**overrides):
    """A minimal three-leg module config whose lint leg names ruff.

    Module-local rather than imported from a sibling admission test: each of
    those files states the same self-containment rationale, and a conftest.py
    edit would trip verify.py's ``has_conftest``.
    """
    kwargs = dict(
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


async def _verify_with_stubbed_spawn(worktree, tmp_path, *, lint_rc: int):
    """Drive the REAL run_verification lint leg with only the SPAWN stubbed.

    The detector itself is left live — it is the thing under test — so this is
    a genuine wiring proof rather than a mock asserting against itself.
    """
    async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        rc = lint_rc if 'ruff' in cmd else 0
        return rc, ('E501 line too long\n' if rc else ''), False

    config = OrchestratorConfig(
        verify_admission_slots_dir=str(tmp_path / 'slots'),
        verify_admission_task_slots=1,
    )
    with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
        return await run_verification(
            worktree=worktree,
            config=config,
            module_config=_module_config(),
            role='task',
            attempt_id=None,
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
        # (iii) the remediation, keyed on its stable phrase rather than wording
        assert 'merge main forward' in message

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
