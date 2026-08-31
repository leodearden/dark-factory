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
