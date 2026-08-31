"""Tests for lms_slate_run — the committed slate-run driver (task 4301).

PRD-MARKER:local-memory-models-eval serving

EVERY TEST HERE IS OFFLINE.  The driver shells out through an injected
`runner` seam, so no arm is started, no card is touched, and no artifact is
written.  That is deliberate and load-bearing: the artifact
`scripts/local-model-serving/verification/health-report.json` may only be
produced by a live run on the 3090, and a test that could write it would be
indistinguishable from the fabrication `test_lms_verification_artifact.py`
exists to stop.

No test in this file carries `@pytest.mark.integration` either.  The root
`pyproject.toml` addopts deselect that marker, so an integration-marked gate
would sit in the tree checked by nobody.

PRD hazard 11 -- "long runs in transient `systemd --user` units, never bare
background shells" -- is enforced here as a CHECKED PROPERTY of the built
argv, mirroring `test_lms_fetch_weights.py`'s treatment of hazard 5.
"""
import sys

import lms_fetch_weights
import lms_slate_run
import pytest

_BACKGROUND_SHELL_TOKENS = ('nohup', 'setsid', 'disown', '&', ';', '|')


# ---------------------------------------------------------------------------
# PRD hazard 11 — the submit argv is a transient unit, built from the one
# authored source of that form
# ---------------------------------------------------------------------------


def test_slate_argv_is_built_from_the_shared_transient_unit_prefix(tmp_path):
    """Not merely "looks like" a transient unit: it IS the shared builder's
    output, so the compliant form has one authored source in the repo."""
    parts = tmp_path / 'parts'
    output = tmp_path / 'health-report.json'

    argv = lms_slate_run.slate_argv(parts, output, env={})

    prefix = lms_fetch_weights.transient_unit_prefix(
        lms_slate_run.SLATE_UNIT_NAME, [],
    )
    assert argv[:len(prefix)] == prefix
    assert argv[:2] == ['systemd-run', '--user']
    assert '--collect' in argv
    assert f'--unit={lms_slate_run.SLATE_UNIT_NAME}' in argv
    assert f'--working-directory={lms_fetch_weights.REPO_ROOT}' in argv


def test_the_slate_unit_name_is_the_documented_one():
    """The README tells an operator to follow
    `journalctl --user -u lms-slate-run -f`; a drifted constant would send
    them to a unit that does not exist."""
    assert lms_slate_run.SLATE_UNIT_NAME == 'lms-slate-run'


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def _payload(argv):
    return argv[argv.index('--') + 1:]


def test_the_payload_invokes_an_absolute_interpreter_on_an_absolute_script(tmp_path):
    """`systemd --user` gets a minimal PATH and none of the caller's venv, so
    a bare `python` either misses or resolves to a different interpreter than
    the one that built the argv."""
    payload = _payload(lms_slate_run.slate_argv(
        tmp_path / 'parts', tmp_path / 'out.json', env={},
    ))

    assert payload[0] == sys.executable
    assert payload[0].startswith('/')
    assert payload[1].endswith('lms_slate_run.py')
    assert payload[1].startswith('/')
    assert payload[1] == str(lms_slate_run.MODULE_PATH)


def test_the_payload_re_enters_in_unit_mode(tmp_path):
    """Without `--in-unit` the unit would re-submit itself, recursively."""
    payload = _payload(lms_slate_run.slate_argv(
        tmp_path / 'parts', tmp_path / 'out.json', env={},
    ))

    assert '--in-unit' in payload


def test_parts_dir_and_output_are_passed_as_absolute_resolved_paths(tmp_path):
    """The unit runs with `--working-directory=<REPO_ROOT>`, not the caller's
    cwd, and derives no path of its own -- both layers must name the same
    directory or a resume silently reads somewhere else."""
    payload = _payload(lms_slate_run.slate_argv(
        tmp_path / 'parts' / '..' / 'parts', tmp_path / 'out.json', env={},
    ))

    parts_value = payload[payload.index('--parts-dir') + 1]
    output_value = payload[payload.index('--output') + 1]

    assert parts_value == str((tmp_path / 'parts').resolve())
    assert output_value == str((tmp_path / 'out.json').resolve())
    assert parts_value.startswith('/')
    assert output_value.startswith('/')


def test_the_ready_timeout_is_forwarded(tmp_path):
    payload = _payload(lms_slate_run.slate_argv(
        tmp_path / 'parts', tmp_path / 'out.json', ready_timeout=123.0, env={},
    ))

    assert payload[payload.index('--ready-timeout') + 1] == '123.0'


@pytest.mark.parametrize('token', _BACKGROUND_SHELL_TOKENS)
def test_slate_argv_never_backgrounds_through_a_shell(tmp_path, token):
    """PRD hazard 11.  A bare background shell is unsupervised, unloggable,
    and dies with the invoking session -- which for a ~30 minute slate sweep
    means losing every arm measured so far."""
    argv = lms_slate_run.slate_argv(tmp_path / 'parts', tmp_path / 'out.json', env={})

    for element in argv:
        assert token not in element, f'{token!r} appears in {element!r}'


def test_the_payload_is_not_a_shell(tmp_path):
    payload = _payload(lms_slate_run.slate_argv(
        tmp_path / 'parts', tmp_path / 'out.json', env={},
    ))

    assert payload[0] not in ('sh', 'bash', '/bin/sh', '/bin/bash')
    assert '-c' not in payload


# ---------------------------------------------------------------------------
# the env allowlist — the subtlest hazard of the transient-unit form
#
# `systemd --user` propagates NONE of the caller's environment, so anything
# unstated is silently ABSENT inside the unit: not empty, not inherited.  The
# consequence is specific and quiet.  `lms_ctl start` writes the VRAM baseline
# through `lms_vram.baseline_dir()`, which reads `$LMS_BASELINE_DIR`; the
# healthcheck reads it back through the same function.  If the two disagree the
# healthcheck exits 8 (`EXIT_STALE_BASELINE`) and writes no file at all -- so
# the whole sweep produces nothing, and the reason is a variable nobody
# mentioned.
#
# The other direction matters just as much: the list is a WHITELIST, never a
# copy of os.environ.  A blanket copy would push OPENAI_API_KEY, HF_TOKEN and
# the rest into the unit's recorded systemd properties and the journal.
# ---------------------------------------------------------------------------


def test_a_set_baseline_dir_is_propagated_into_the_unit(tmp_path):
    argv = lms_slate_run.slate_argv(
        tmp_path / 'parts', tmp_path / 'out.json',
        env={'LMS_BASELINE_DIR': '/run/user/1000/lms-baselines'},
    )

    assert '--setenv=LMS_BASELINE_DIR=/run/user/1000/lms-baselines' in argv


def test_an_absent_baseline_dir_emits_no_setenv_at_all(tmp_path):
    """Not an empty one.  `''` is not "unset" to `os.environ.get(...)`'s
    fallback, so an empty setenv would send `baseline_dir()` to Path('') --
    a different directory than the default it was supposed to fall back to."""
    argv = lms_slate_run.slate_argv(tmp_path / 'parts', tmp_path / 'out.json', env={})

    assert not any(a.startswith('--setenv=LMS_BASELINE_DIR') for a in argv)


def test_the_propagated_key_is_the_one_lms_vram_actually_reads():
    """Pinned against `lms_vram`'s own constant rather than a literal, so a
    rename there cannot leave this driver propagating a dead name."""
    import lms_vram

    assert lms_vram.BASELINE_DIR_ENV in lms_slate_run.PROPAGATED_ENV_KEYS


@pytest.mark.parametrize('secret_key', ['OPENAI_API_KEY', 'HF_TOKEN', 'VIRTUAL_ENV'])
def test_unlisted_caller_variables_never_reach_the_unit(tmp_path, secret_key):
    """A whitelist, not a passthrough: `systemd-run --setenv` puts a value in
    the unit's recorded properties and the journal, so a blanket copy of
    os.environ is a secret-leak surface for zero benefit."""
    argv = lms_slate_run.slate_argv(
        tmp_path / 'parts', tmp_path / 'out.json',
        env={
            'LMS_BASELINE_DIR': '/run/user/1000/lms-baselines',
            secret_key: 'sensitive-measured-value',
        },
    )

    assert not any(secret_key in element for element in argv)
    assert not any('sensitive-measured-value' in element for element in argv)


def test_the_allowlist_is_a_whitelist_not_a_copy_of_os_environ(tmp_path, monkeypatch):
    """The default env is the real one; with no injected dict the ONLY
    setenv flags that may appear are allowlisted keys."""
    monkeypatch.setenv('LMS_SLATE_RUN_CANARY', 'must-not-propagate')

    argv = lms_slate_run.slate_argv(tmp_path / 'parts', tmp_path / 'out.json')

    setenvs = [a for a in argv if a.startswith('--setenv=')]
    for flag in setenvs:
        key = flag[len('--setenv='):].split('=', 1)[0]
        assert key in lms_slate_run.PROPAGATED_ENV_KEYS, f'{key} is not allowlisted'
    assert not any('must-not-propagate' in element for element in argv)


# ---------------------------------------------------------------------------
# the in-unit sweep — one arm at a time, strictly serialized
#
# `lms_ctl start` is exclusive BY DEFAULT and REFUSES (exit 4) when a sibling
# arm holds the card; it never evicts.  So overlapping two arms does not
# degrade to a slow sweep, it produces a refusal — which makes the ordering
# below a correctness property rather than a stylistic one.
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Records every argv and returns a scripted returncode.

    Stands in for `subprocess.run`, which is how every test in this file stays
    offline: no arm is started and no card is touched.
    """

    def __init__(self, codes=None, raises=None):
        self.calls: list[list[str]] = []
        self._codes = dict(codes or {})
        self._raises = dict(raises or {})

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        key = _stage_key(argv)
        if key in self._raises:
            raise self._raises[key]
        return _FakeCompleted(self._codes.get(key, 0))

    def stages(self):
        return [_stage_key(argv) for argv in self.calls]


class _FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


def _stage_key(argv):
    """('ctl', verb, arm_id) / ('healthcheck', arm_id) / ('merge',)."""
    if str(lms_slate_run.CTL_PATH) in argv:
        tail = argv[argv.index(str(lms_slate_run.CTL_PATH)) + 1:]
        return ('ctl', tail[0], tail[1] if len(tail) > 1 else None)
    if str(lms_slate_run.HEALTHCHECK_PATH) in argv:
        if '--merge' in argv:
            return ('merge',)
        return ('healthcheck', argv[argv.index('--arm') + 1])
    raise AssertionError(f'unrecognised argv: {argv}')


class _FakeArm:
    def __init__(self, arm_id):
        self.arm_id = arm_id


class _FakeManifest:
    def __init__(self, *arm_ids):
        self.arms = [_FakeArm(a) for a in arm_ids]

    def arm_ids(self):
        return [a.arm_id for a in self.arms]


@pytest.fixture
def two_arms(monkeypatch):
    manifest = _FakeManifest('arm-one', 'arm-two')
    monkeypatch.setattr(lms_slate_run, 'load_arms', lambda: manifest)
    return manifest


def test_each_arm_runs_start_wait_healthcheck_stop_in_that_order(tmp_path, two_arms):
    runner = _FakeRunner()

    lms_slate_run.run_slate(tmp_path / 'parts', tmp_path / 'out.json', runner=runner)

    assert runner.stages()[:8] == [
        ('ctl', 'start', 'arm-one'),
        ('ctl', 'wait-ready', 'arm-one'),
        ('healthcheck', 'arm-one'),
        ('ctl', 'stop', 'arm-one'),
        ('ctl', 'start', 'arm-two'),
        ('ctl', 'wait-ready', 'arm-two'),
        ('healthcheck', 'arm-two'),
        ('ctl', 'stop', 'arm-two'),
    ]


def test_the_arms_are_strictly_serialized(tmp_path, two_arms):
    """Arm two's `start` may appear only AFTER arm one's `stop`.  Overlapping
    them does not merely slow the sweep: `lms_ctl start` refuses rather than
    evicting, so arm two would fail with exit 4 on a healthy card."""
    runner = _FakeRunner()

    lms_slate_run.run_slate(tmp_path / 'parts', tmp_path / 'out.json', runner=runner)

    stages = runner.stages()
    assert stages.index(('ctl', 'start', 'arm-two')) > stages.index(
        ('ctl', 'stop', 'arm-one')
    )


def test_start_is_left_exclusive(tmp_path, two_arms):
    """`--no-exclusive` would let a second arm onto a card that fits one.  The
    sweep never needs it: it stops each arm before starting the next."""
    runner = _FakeRunner()

    lms_slate_run.run_slate(tmp_path / 'parts', tmp_path / 'out.json', runner=runner)

    for argv in runner.calls:
        assert '--no-exclusive' not in argv


def test_wait_ready_carries_the_ready_timeout(tmp_path, two_arms):
    runner = _FakeRunner()

    lms_slate_run.run_slate(
        tmp_path / 'parts', tmp_path / 'out.json', ready_timeout=42.0, runner=runner,
    )

    waits = [a for a in runner.calls if _stage_key(a)[:2] == ('ctl', 'wait-ready')]
    assert waits
    for argv in waits:
        assert argv[argv.index('--timeout') + 1] == '42.0'


def test_the_default_ready_timeout_is_the_one_lms_ctl_documents():
    """Pinned against `lms_ctl`'s constant, not a literal: a driver that
    silently waited less than the CLI's own default would report arms as
    not-ready that were merely slow to load."""
    import lms_ctl

    assert lms_slate_run.DEFAULT_READY_TIMEOUT_S == lms_ctl.DEFAULT_READY_TIMEOUT_S


def test_each_arms_healthcheck_writes_its_own_part_file(tmp_path, two_arms):
    parts_dir = tmp_path / 'parts'
    runner = _FakeRunner()

    lms_slate_run.run_slate(parts_dir, tmp_path / 'out.json', runner=runner)

    for arm_id in ('arm-one', 'arm-two'):
        argv = next(a for a in runner.calls if _stage_key(a) == ('healthcheck', arm_id))
        assert argv[argv.index('--output') + 1] == str(parts_dir / f'{arm_id}.json')


def test_the_parts_directory_is_created(tmp_path, two_arms):
    parts_dir = tmp_path / 'nested' / 'parts'

    lms_slate_run.run_slate(parts_dir, tmp_path / 'out.json', runner=_FakeRunner())

    assert parts_dir.is_dir()


def test_every_helper_script_is_invoked_by_absolute_path_with_sys_executable(
    tmp_path, two_arms,
):
    """The unit has a minimal PATH and none of the caller's venv, so a bare
    `python` or a relative script path resolves to something nobody reviewed --
    or to nothing."""
    runner = _FakeRunner()

    lms_slate_run.run_slate(tmp_path / 'parts', tmp_path / 'out.json', runner=runner)

    assert runner.calls
    for argv in runner.calls:
        assert argv[0] == sys.executable
        assert argv[1].startswith('/')
        assert argv[1] in (str(lms_slate_run.CTL_PATH), str(lms_slate_run.HEALTHCHECK_PATH))


def test_the_helper_script_paths_are_the_sibling_modules():
    assert lms_slate_run.CTL_PATH == lms_slate_run.MODULE_PATH.parent / 'lms_ctl.py'
    assert lms_slate_run.CTL_PATH.exists()
    assert lms_slate_run.HEALTHCHECK_PATH == (
        lms_slate_run.MODULE_PATH.parent / 'lms_healthcheck.py'
    )
    assert lms_slate_run.HEALTHCHECK_PATH.exists()
