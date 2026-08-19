"""Tests for scripts/reify-closure-staleness-sweep.sh (task 3102).

The wrapper is the dark-factory half of reify task 5321's cross-repo seam: it
runs reify's ``deterministic-gate-closure-staleness-sweep.sh --emit-requests``
and then drains what that emitted with
``scripts/consume_redispatch_requests.py``, in that order, as ONE nightly
oneshot rather than two independently-timed units.

Driven via subprocess with BOTH ``*_CMD`` seams pointed at fake recorder
executables (each records its argv and an os.environ snapshot into a JSON
state file) -- mirroring test_flag_marker_sweep_wrapper.py. The real reify
sweep, the real task store and systemd are never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WRAPPER = Path(__file__).parent.parent / 'reify-closure-staleness-sweep.sh'


_FAKE_RECORDER_SRC = '''#!/usr/bin/env python3
"""Fake command recorder. Appends {"who", "argv", "env"} to the JSON list at
$FAKE_STATE, then exits with $FAKE_EXIT_<WHO> (default 0)."""
import json
import os
import sys

who = os.environ["FAKE_WHO"]
state_path = os.environ["FAKE_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.append({"who": who, "argv": sys.argv[1:], "env": dict(os.environ)})
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(int(os.environ.get(f"FAKE_EXIT_{who}", "0")))
'''


def _harness(tmp_path, *, sweep_exit=0, consumer_exit=0, extra_env=None,
             reify_repo=None):
    """Build the two fake recorders and the env that points the wrapper at them.

    Returns (env, state_path, fake_reify).
    """
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    state_path = tmp_path / 'state.json'
    state_path.write_text('[]')

    # One recorder binary per half, so `who` is unambiguous in the state file.
    # The interpreter is spelled ABSOLUTELY (sys.executable), never `python3`:
    # one test shims `python3` onto PATH to pin the wrapper's default consumer
    # prefix, and a PATH-resolved shim here would re-enter that shim forever.
    for who in ('SWEEP', 'CONSUMER'):
        shim = bin_dir / f'fake-{who.lower()}'
        shim.write_text(
            '#!/usr/bin/env bash\n'
            f'FAKE_WHO={who} exec {sys.executable} '
            f'{bin_dir / "recorder.py"} "$@"\n'
        )
        shim.chmod(0o755)
    (bin_dir / 'recorder.py').write_text(_FAKE_RECORDER_SRC)

    fake_reify = Path(reify_repo) if reify_repo else (tmp_path / 'fake-reify')
    fake_reify.mkdir(parents=True, exist_ok=True)
    (fake_reify / 'data').mkdir(exist_ok=True)

    env = dict(os.environ)
    env['PATH'] = f'{bin_dir}{os.pathsep}{env["PATH"]}'
    env['REIFY_CLOSURE_SWEEP_CMD'] = 'fake-sweep'
    env['REIFY_REDISPATCH_CONSUMER_CMD'] = 'fake-consumer'
    env['FAKE_STATE'] = str(state_path)
    env['FAKE_EXIT_SWEEP'] = str(sweep_exit)
    env['FAKE_EXIT_CONSUMER'] = str(consumer_exit)
    env['REIFY_REPO'] = str(fake_reify)
    if extra_env:
        env.update(extra_env)
    return env, state_path, fake_reify


def _run(tmp_path, **kw):
    env, state_path, fake_reify = _harness(tmp_path, **kw)
    result = subprocess.run(
        ['bash', str(WRAPPER)], env=env, capture_output=True, text=True,
        timeout=60,
    )
    return result, json.loads(state_path.read_text()), fake_reify


def _by(calls, who):
    return [c for c in calls if c['who'] == who]


def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f'Expected {WRAPPER} to be executable; run: chmod +x {WRAPPER}')


def test_sweep_is_invoked_by_absolute_path_with_emit_requests(tmp_path):
    result, calls, fake_reify = _run(tmp_path)
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    sweeps = _by(calls, 'SWEEP')
    assert len(sweeps) == 1, calls
    argv = sweeps[0]['argv']
    script = argv[0]
    assert os.path.isabs(script), argv
    assert script == str(
        fake_reify / 'scripts' / 'deterministic-gate-closure-staleness-sweep.sh'
    ), argv
    assert '--emit-requests' in argv, argv


def test_requests_dir_defaults_under_reifys_existing_data_dir(tmp_path):
    """The sweep creates the requests dir on demand ONLY when its PARENT
    already exists -- a deliberate guard so a typo'd --emit-requests surfaces
    rather than hiding as "0 requests emitted". reify's data/ is that parent."""
    result, calls, fake_reify = _run(tmp_path)
    argv = _by(calls, 'SWEEP')[0]['argv']
    requests_dir = argv[argv.index('--emit-requests') + 1]
    assert requests_dir == str(fake_reify / 'data' / 'redispatch-requests'), argv
    assert os.path.isdir(os.path.dirname(requests_dir)), requests_dir


def test_consumer_runs_after_the_sweep_against_the_same_requests_dir(tmp_path):
    result, calls, fake_reify = _run(tmp_path)
    assert [c['who'] for c in calls] == ['SWEEP', 'CONSUMER'], calls

    sweep_argv = _by(calls, 'SWEEP')[0]['argv']
    consumer_argv = _by(calls, 'CONSUMER')[0]['argv']
    emitted = sweep_argv[sweep_argv.index('--emit-requests') + 1]
    drained = consumer_argv[consumer_argv.index('--requests-dir') + 1]
    assert emitted == drained, (sweep_argv, consumer_argv)


def test_consumer_is_invoked_with_reifys_project_root(tmp_path):
    result, calls, fake_reify = _run(tmp_path)
    argv = _by(calls, 'CONSUMER')[0]['argv']
    assert argv[0] == str(
        Path(WRAPPER).parent / 'consume_redispatch_requests.py'), argv
    assert argv[argv.index('--project-root') + 1] == str(fake_reify), argv


def test_a_failing_sweep_does_not_prevent_the_consumer_running(tmp_path):
    """The sweep's report and its emission are independent of each other, and
    requests emitted on a previous night are still on disk -- so a sweep fault
    must not also swallow the drain."""
    result, calls, _ = _run(tmp_path, sweep_exit=3)
    assert [c['who'] for c in calls] == ['SWEEP', 'CONSUMER'], calls


def test_a_failing_consumer_does_not_mask_the_sweep(tmp_path):
    result, calls, _ = _run(tmp_path, consumer_exit=4)
    assert [c['who'] for c in calls] == ['SWEEP', 'CONSUMER'], calls
    assert 'consumer' in (result.stdout + result.stderr).lower()


def test_neither_halfs_failure_aborts_the_wrapper(tmp_path):
    """Both halves fail: the wrapper still runs both and still reports.

    The units are `oneshot`; a wrapper that propagated a failure would put the
    service into systemd `failed` state, where it STAYS -- silently stopping
    the whole nightly job (the lesson in fused-memory-flag-marker-sweep.sh).
    """
    result, calls, _ = _run(tmp_path, sweep_exit=3, consumer_exit=4)
    assert [c['who'] for c in calls] == ['SWEEP', 'CONSUMER'], calls
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')


def test_reify_repo_override_flows_through_to_both_halves(tmp_path):
    other = tmp_path / 'elsewhere' / 'reify'
    result, calls, fake_reify = _run(tmp_path, reify_repo=other)
    assert fake_reify == other
    sweep_argv = _by(calls, 'SWEEP')[0]['argv']
    consumer_argv = _by(calls, 'CONSUMER')[0]['argv']
    assert sweep_argv[0].startswith(str(other)), sweep_argv
    assert str(other) in consumer_argv, consumer_argv


def test_requests_dir_is_overridable(tmp_path):
    custom = tmp_path / 'custom-requests'
    result, calls, _ = _run(
        tmp_path, extra_env={'REIFY_GATE_STALENESS_REQUESTS_DIR': str(custom)})
    sweep_argv = _by(calls, 'SWEEP')[0]['argv']
    consumer_argv = _by(calls, 'CONSUMER')[0]['argv']
    assert sweep_argv[sweep_argv.index('--emit-requests') + 1] == str(custom)
    assert consumer_argv[consumer_argv.index('--requests-dir') + 1] == str(custom)


def test_wrapper_needs_no_dotenv_or_config_path(tmp_path):
    """The stdlib-only posture: both halves run without .env sourcing or
    CONFIG_PATH/FALKORDB_URI exports, so a lockfile or venv-resolution problem
    can never wedge the nightly job.

    Asserted on the wrapper's CODE (comments stripped) rather than its text, so
    the header may keep explaining why those things are absent.
    """
    code = '\n'.join(
        ln for ln in WRAPPER.read_text().splitlines()
        if not ln.lstrip().startswith('#')
    )
    for forbidden in ('source ', 'set -a', 'CONFIG_PATH', 'FALKORDB_URI',
                      'uv run', '.env'):
        assert forbidden not in code, f'{forbidden!r} found in:\n{code}'

    env, state_path, _ = _harness(tmp_path)
    for var in ('CONFIG_PATH', 'FALKORDB_URI', 'PROJECT_ROOT'):
        env.pop(var, None)
    result = subprocess.run(['bash', str(WRAPPER)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    calls = json.loads(state_path.read_text())
    assert [c['who'] for c in calls] == ['SWEEP', 'CONSUMER'], calls


def test_default_consumer_prefix_is_a_bare_python3(tmp_path):
    """Pins the default REIFY_REDISPATCH_CONSUMER_CMD prefix, which every other
    test bypasses. `python3 <script>` DIRECTLY -- no uv, no --project -- is the
    whole point of the standalone design (mirrors reclaim-orphaned-worktrees.sh).
    REIFY_REPO carries a space so a dropped-quote regression manifests as a
    word-split argv rather than passing by coincidence."""
    env, state_path, fake_reify = _harness(
        tmp_path, reify_repo=tmp_path / 'reify with spaces')
    env.pop('REIFY_REDISPATCH_CONSUMER_CMD', None)

    # Shim `python3` so the real consumer never runs; it must be found by the
    # bare-`python3` default rather than an absolute interpreter path.
    bin_dir = Path(env['PATH'].split(os.pathsep)[0])
    (bin_dir / 'python3').write_text(
        '#!/usr/bin/env bash\n'
        f'FAKE_WHO=CONSUMER exec {sys.executable} '
        f'{bin_dir / "recorder.py"} "$@"\n'
    )
    (bin_dir / 'python3').chmod(0o755)

    result = subprocess.run(['bash', str(WRAPPER)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    consumer_argv = _by(json.loads(state_path.read_text()), 'CONSUMER')[0]['argv']
    assert consumer_argv[0] == str(
        Path(WRAPPER).parent / 'consume_redispatch_requests.py'), consumer_argv
    assert str(fake_reify) in consumer_argv, consumer_argv
