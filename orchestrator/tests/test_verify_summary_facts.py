"""The per-leg facts task 5671 adds to the verify summary record.

These sit on top of task 3353's load stamp (test_verify_load_stamp.py) and
ride the same record through the same path: ``CheckRun`` -> ``to_dict`` ->
``_build_summary_payload``'s key whitelist -> the ``summary.json`` on disk.

Every real-path test here reads the summary actually WRITTEN, never a spy on
``_persist_attempt_logs``' argument: a spy cannot see the whitelist dropping a
field on the way to disk, which is how ``segments`` was once lost (the lesson
recorded in test_verify_load_stamp.py).
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from typing import Literal

import pytest

_TEST_CMD = 'uv run pytest tests/ -q'
_LINT_CMD = 'uv run ruff check src/'


def _module_config(prefix='__fallback__', *, test_command=_TEST_CMD):
    from orchestrator.config import ModuleConfig  # noqa: PLC0415

    return ModuleConfig(
        prefix=prefix,
        test_command=test_command,
        lint_command=_LINT_CMD,
        type_check_command=None,
    )


def _run(label='test', **overrides):
    run = {
        'label': label,
        'cmd': f'uv run {label}',
        'rc': 0,
        'output': '',
        'timed_out': False,
        'started_at': 'ts',
        'duration_secs': 1.5,
        'segments': None,
        'load': None,
    }
    run.update(overrides)
    return run


async def _passing_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **_kw):
    return 0, 'ok', False


async def _run_task_verification(
    tmp_path: Path,
    *,
    role: Literal['merge', 'task', 'background'] = 'task',
    run_cmd=_passing_run_cmd,
    config_overrides=None,
    **kwargs,
):
    """Drive the REAL ``run_verification`` against a tmp_path worktree.

    Only ``_run_cmd`` is stubbed. *config_overrides* are
    ``OrchestratorConfig`` fields; admission is off unless they turn it on.
    """
    from unittest.mock import patch  # noqa: PLC0415

    from orchestrator import verify as verify_mod  # noqa: PLC0415
    from orchestrator.config import OrchestratorConfig  # noqa: PLC0415

    (tmp_path / '.task').mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(
        project_root=tmp_path,
        **{'verify_admission_enabled': False, **(config_overrides or {})},
    )
    kwargs.setdefault('module_config', _module_config())
    kwargs.setdefault('max_retries', 0)
    with patch('orchestrator.verify._run_cmd', side_effect=run_cmd):
        return await verify_mod.run_verification(
            tmp_path,
            config,
            kwargs.pop('module_config'),
            attempt_id=1,
            # Load-bearing: run_verification persists only when attempt_id AND
            # task_id are both set.
            task_id='5671',
            role=role,
            **kwargs,
        )


def _read_worktree_summary(tmp_path: Path, prefix='__fallback__') -> dict:
    path = tmp_path / '.task' / 'verify' / f'attempt-1.{prefix}.summary.json'
    assert path.exists(), (
        f'no summary JSON at {path}; wrote '
        f'{sorted(p.name for p in (tmp_path / ".task" / "verify").glob("*"))}'
    )
    return json.loads(path.read_text(encoding='utf-8'))


def _read_archived_summary(archive_root: Path) -> dict:
    written = sorted((archive_root / '5671').glob('attempt-1*.summary-*.json'))
    assert len(written) == 1, f'expected one archived summary, found {written}'
    return json.loads(written[0].read_text(encoding='utf-8'))


class TestTheSummaryCarriesTheRole:
    """The verify ROLE is a structured field of the record, not a path inference.

    The budget census used to infer it from a ``.worktrees/<lane>`` path, and
    the archive has no such path, so every archived record was role-less.
    """

    @pytest.mark.parametrize('role', ['task', 'merge'])
    def test_the_payload_names_its_role(self, role):
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        payload = _build_summary_payload([_run('test')], 'clean', '', role=role)

        assert payload['role'] == role

    def test_the_role_is_present_even_when_no_command_ran(self):
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        payload = _build_summary_payload(
            [_run('test', cmd=None), _run('lint', cmd=None)], 'clean', '', role='task',
        )

        assert payload['role'] == 'task'

    @pytest.mark.asyncio
    async def test_the_task_path_writes_its_role_to_disk(self, tmp_path):
        await _run_task_verification(tmp_path, role='task')

        assert _read_worktree_summary(tmp_path)['role'] == 'task'

    def test_the_merge_writer_stamps_merge(self, tmp_path):
        """``_archive_merge_verify_logs`` is the merge-path writer by definition.

        It serves both the local merge lane and ``verify_runner.RemoteRunner``,
        so its role cannot be anything but 'merge'.
        """
        from orchestrator.verify import _archive_merge_verify_logs  # noqa: PLC0415

        archive_root = tmp_path / 'archive'
        _archive_merge_verify_logs(
            [_run('test', rc=1)], archive_root, '5671', 1, 'test_failure', '',
        )

        assert _read_archived_summary(archive_root)['role'] == 'merge'


class TestTheTopLevelNamesItsCommand:
    """The top-level rc/cmd/... copy ONE run's fields; 'label' says which.

    Without it a reader assumes the top level is the test leg. The
    2026-09-19 full-suite census over-counted exactly that way: a green test
    leg beside a red lint leg reads, at the top level, as a failing test run
    (plans/orchestrator-suite-slowdown-2026-09-19/C-full-suite-fallback.md §6).
    """

    @staticmethod
    def _payload(runs):
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        payload = _build_summary_payload(runs, 'lint_failure', '', role='task')
        named = [c for c in payload['commands'] if c['label'] == payload['label']]
        if payload['label'] is not None:
            assert [c['cmd'] for c in named] == [payload['cmd']], (
                "payload['label'] must name the commands[] entry whose cmd is "
                "payload['cmd']"
            )
        return payload

    def test_a_red_lint_beside_a_green_test_is_labelled_lint(self):
        payload = self._payload([
            _run('test', cmd=_TEST_CMD, rc=0),
            _run('lint', cmd=_LINT_CMD, rc=1),
        ])

        assert payload['label'] == 'lint'
        assert payload['cmd'] == _LINT_CMD
        assert payload['rc'] == 1

    def test_all_green_names_the_first_active_run(self):
        payload = self._payload([
            _run('test', cmd=_TEST_CMD),
            _run('lint', cmd=_LINT_CMD),
        ])

        assert payload['label'] == 'test'
        assert payload['cmd'] == _TEST_CMD

    def test_a_signal_killed_leg_outranks_a_failing_one_and_says_so(self):
        payload = self._payload([
            _run('test', cmd=_TEST_CMD, rc=1),
            _run('type', cmd='uv run pyright', rc=-9),
        ])

        assert payload['label'] == 'type'
        assert payload['rc'] == -9

    def test_no_command_ran_names_no_label(self):
        payload = self._payload([_run('test', cmd=None), _run('lint', cmd=None)])

        assert payload['label'] is None
        assert payload['cmd'] is None

    def test_every_pre_existing_top_level_key_is_still_present(self):
        payload = self._payload([_run('test', cmd=_TEST_CMD)])

        for key in ('category', 'cause_hint', 'rc', 'timed_out', 'cmd',
                    'started_at', 'duration_secs', 'commands'):
            assert key in payload, f'current readers lost {key!r}'


def _admitting(tmp_path: Path) -> dict:
    """Config for real admission with ONE slot in a private directory."""
    return {
        'verify_admission_enabled': True,
        'verify_admission_task_slots': 1,
        'verify_admission_slots_dir': str(tmp_path / 'slots'),
    }


_HELD_SECS = 0.3


@contextlib.contextmanager
def _another_verify_holds_the_slot(tmp_path: Path):
    """Another task verify holds the only real flock slot, and frees it
    ``_HELD_SECS`` after the verify under test asks for one.

    Timed from the ASK, not from entering this block: under a loaded host
    ``run_verification``'s own setup outlasted a fixed 1.0s hold (measured), so
    a hold timed from here queued nothing. The ask is observed by wrapping
    ``shared.verify_admission.acquire_task_slot`` where verify imports it; the
    real function still grants the slot, on the real flock.
    """
    from unittest.mock import patch  # noqa: PLC0415

    from shared.verify_admission import acquire_task_slot  # noqa: PLC0415

    slots_dir = tmp_path / 'slots'
    slots_dir.mkdir(parents=True, exist_ok=True)
    holder = acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False)
    assert holder.__enter__(), 'precondition: the slot was free to take'
    released = threading.Event()

    def release():
        if not released.is_set():
            released.set()
            holder.__exit__(None, None, None)

    timer = threading.Timer(_HELD_SECS, release)

    def asked(*args, **kwargs):
        if timer.ident is None:
            timer.start()
        return acquire_task_slot(*args, **kwargs)

    try:
        with patch('orchestrator.verify.acquire_task_slot', side_effect=asked):
            yield
    finally:
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        release()


def _entry(summary, label):
    return next(c for c in summary['commands'] if c['label'] == label)


@pytest.mark.real_verify_admission
class TestTheSlotWaitIsRecorded:
    """How long a leg queued for an admission slot, beside how long it ran.

    ``duration_secs`` and ``load`` deliberately start INSIDE the slot, so the
    queueing half of a leg's wall time was recorded nowhere. ``None`` means
    the leg competed for no slot at all, never 0.0.

    Driven through config and the real flock slots: a queue is made by
    another verify holding the only slot.
    """

    @pytest.mark.asyncio
    async def test_a_gated_test_leg_records_its_wait_and_not_as_command_time(
        self, tmp_path,
    ):
        with _another_verify_holds_the_slot(tmp_path):
            await _run_task_verification(tmp_path, config_overrides=_admitting(tmp_path))

        test = _entry(_read_worktree_summary(tmp_path), 'test')
        assert test['slot_wait_secs'] >= _HELD_SECS - 0.05
        assert test['duration_secs'] < test['slot_wait_secs'], (
            'queueing for a slot is not command time'
        )

    @pytest.mark.asyncio
    async def test_the_lint_leg_competes_for_no_slot(self, tmp_path):
        with _another_verify_holds_the_slot(tmp_path):
            await _run_task_verification(tmp_path, config_overrides=_admitting(tmp_path))

        assert _entry(_read_worktree_summary(tmp_path), 'lint')['slot_wait_secs'] is None

    @pytest.mark.asyncio
    async def test_no_admission_means_no_wait_not_a_zero_wait(self, tmp_path):
        await _run_task_verification(
            tmp_path,
            config_overrides={**_admitting(tmp_path), 'verify_admission_enabled': False},
        )

        assert _entry(_read_worktree_summary(tmp_path), 'test')['slot_wait_secs'] is None

    @pytest.mark.asyncio
    async def test_a_merge_leg_never_counts_against_a_slot(self, tmp_path):
        """``is_gated_role('merge')`` is False: merge never queues, even behind
        a task verify holding the only slot.

        A merge run persists no worktree summary, so its test leg is made to
        fail and the ARCHIVED summary is read instead.
        """
        async def failing_pytest(cmd, cwd, timeout, env=None, log_path=None, **_kw):
            if 'pytest' in cmd:
                return 1, 'FAILED tests/test_x.py::test_y - assert 0', False
            return 0, 'ok', False

        archive_root = tmp_path / 'archive'
        with _another_verify_holds_the_slot(tmp_path):
            await _run_task_verification(
                tmp_path,
                role='merge',
                run_cmd=failing_pytest,
                config_overrides=_admitting(tmp_path),
                archive_root=archive_root,
            )

        test = _entry(_read_archived_summary(archive_root), 'test')
        assert test['slot_wait_secs'] is None

    @pytest.mark.asyncio
    async def test_every_entry_carries_the_key_null_or_not(self, tmp_path):
        await _run_task_verification(tmp_path, config_overrides=_admitting(tmp_path))

        entries = _read_worktree_summary(tmp_path)['commands']
        assert {e['label'] for e in entries} == {'test', 'lint'}
        assert all('slot_wait_secs' in e for e in entries)


_GREEN_JUNIT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites>'
    '<testsuite name="pytest" tests="1" failures="0">'
    '<testcase classname="tests.test_x" name="test_ok" time="0.01"/>'
    '</testsuite></testsuites>'
)
_RED_JUNIT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites>'
    '<testsuite name="pytest" tests="1" failures="1">'
    '<testcase classname="tests.test_x" name="test_y" time="0.01">'
    '<failure message="assert 0">assert 0</failure></testcase>'
    '</testsuite></testsuites>'
)


def _junit_target(cmd: str) -> Path | None:
    """The report path a rendered pytest command was told to write, if any.

    ``with_junitxml`` renders ``--junitxml <path>`` as two tokens (verified by
    rendering one); the attached ``=`` form is accepted too so the fake does
    not depend on that choice.
    """
    import shlex  # noqa: PLC0415

    tokens = shlex.split(cmd)
    for index, token in enumerate(tokens):
        if token == '--junitxml' and index + 1 < len(tokens):
            return Path(tokens[index + 1])
        if token.startswith('--junitxml='):
            return Path(token.split('=', 1)[1])
    return None


class _JunitWritingPytest:
    """A fake ``_run_cmd`` that writes a junit report wherever it is told to.

    *passes* scripts each pytest invocation in order as ``(rc, report_xml,
    timed_out)``; ``report_xml=None`` means the run was killed before pytest
    wrote anything. Every spawned command is recorded.
    """

    def __init__(self, *passes):
        self.passes = list(passes) or [(0, _GREEN_JUNIT, False)]
        self.commands: list[str] = []
        self.pytest_runs = 0

    async def __call__(self, cmd, cwd, timeout, env=None, log_path=None, **_kw):
        self.commands.append(cmd)
        if 'pytest' not in cmd:
            return 0, 'ok', False
        rc, report_xml, timed_out = self.passes[min(self.pytest_runs, len(self.passes) - 1)]
        self.pytest_runs += 1
        target = _junit_target(cmd)
        if target is not None and report_xml is not None:
            target.write_text(report_xml, encoding='utf-8')
        output = 'FAILED tests/test_x.py::test_y - assert 0' if rc == 1 else 'ok'
        return rc, output, timed_out

    def command_for(self, tool):
        return next(c for c in self.commands if tool in c)


def _archived_junit(archive_root: Path) -> list[Path]:
    return sorted((archive_root / '5671').glob('attempt-1.orchestrator.junit-*.xml.gz'))


class TestTaskLegsArchiveAJunitReport:
    """A task-path pytest leg leaves a gzipped junit COST record in the archive.

    The merge lane has archived one per module since task 5670; task legs
    archived none, so the most expensive legs (the task-path full suites) had
    no per-test cost record at all. It is a cost record ONLY: task-path
    failure attribution is unchanged.
    """

    @staticmethod
    async def _run(tmp_path, fake, **kwargs):
        kwargs.setdefault('archive_root', tmp_path / 'archive')
        return await _run_task_verification(
            tmp_path,
            # The bound method, not the instance: AsyncMock awaits a side_effect
            # only when it is a coroutine FUNCTION.
            run_cmd=fake.__call__,
            module_config=_module_config('orchestrator'),
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_the_report_goes_to_gitignored_scratch_by_absolute_path(self, tmp_path):
        fake = _JunitWritingPytest()
        await self._run(tmp_path, fake)

        target = _junit_target(fake.command_for('pytest'))
        assert target is not None, 'the task-path test leg was given no --junitxml'
        assert target.is_absolute()
        assert target.is_relative_to((tmp_path / '.task').resolve())
        assert not (tmp_path / '.df-verify-junit').exists(), (
            '.df-verify-junit is not gitignored; it would be an untracked dir '
            'in the task tree'
        )

    @pytest.mark.asyncio
    async def test_the_lint_leg_gets_no_report(self, tmp_path):
        fake = _JunitWritingPytest()
        await self._run(tmp_path, fake)

        assert '--junitxml' not in fake.command_for('ruff')

    @pytest.mark.asyncio
    async def test_a_green_run_archives_exactly_one_gzipped_report(self, tmp_path):
        import gzip  # noqa: PLC0415

        fake = _JunitWritingPytest((0, _GREEN_JUNIT, False))
        result = await self._run(tmp_path, fake)

        assert result.passed
        archived = _archived_junit(tmp_path / 'archive')
        assert len(archived) == 1, archived
        assert gzip.decompress(archived[0].read_bytes()).decode('utf-8') == _GREEN_JUNIT

    @pytest.mark.asyncio
    async def test_task_path_attribution_is_unchanged(self, tmp_path):
        """``verify_failure_is_preexisting_on_main`` branches on this field."""
        fake = _JunitWritingPytest((1, _RED_JUNIT, False))
        result = await self._run(tmp_path, fake)

        assert not result.passed
        assert result.failing_test_ids is None

    @pytest.mark.asyncio
    async def test_no_archive_destination_means_no_report_at_all(self, tmp_path):
        """No destination, no writer: the main probe and review checkpoints stay
        byte-identical."""
        fake = _JunitWritingPytest()
        await self._run(tmp_path, fake, archive_root=None)

        assert '--junitxml' not in fake.command_for('pytest')
        assert not (tmp_path / 'archive').exists()
        assert not list((tmp_path / '.task' / 'verify').glob('*.junit.xml'))

    @pytest.mark.asyncio
    async def test_a_retry_killed_before_writing_archives_nothing(self, tmp_path):
        """The retry clears its predecessor's report at the injection chokepoint."""
        fake = _JunitWritingPytest((124, _GREEN_JUNIT, True), (124, None, True))
        await self._run(tmp_path, fake, max_retries=1)

        assert fake.pytest_runs == 2, 'precondition: the pure timeout was retried'
        assert _archived_junit(tmp_path / 'archive') == []
