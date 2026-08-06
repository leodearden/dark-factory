"""Tests for lms_ctl and install-lms-units.sh (task 3713, LME-alpha).

PRD-MARKER:local-memory-models-eval serving

Real systemd is never touched: a fake `systemctl` is shimmed onto PATH and
records every invocation into a JSON state file, mirroring the idiom in
scripts/tests/test_install_flag_marker_sweep_timer.py:34-149.

The load-bearing assertion here is a NEGATIVE one.  `start` on an arm whose
declared footprint exceeds measured free VRAM must refuse having issued NO
systemctl call at all -- the refusal has to come *before* the side effect.  A
refusal after the fact would already have handed the unit to systemd, and on a
single 24 GB card shared with whisper-writer (which Leo requires resident,
PRD D10) that means an OOM that disturbs a process the eval must not disturb.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import lms_ctl
import lms_manifest
import lms_vram

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
LMS_DIR = SCRIPTS_DIR / 'local-model-serving'
INSTALLER = LMS_DIR / 'install-lms-units.sh'
UNIT_TEMPLATE = LMS_DIR / 'lms-arm@.service'

MEASURED_GPU = lms_vram.GpuReading(total_mib=24576, used_mib=7362, free_mib=16761)

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` recording every invocation (minus --user) as JSON.

`list-units` echoes $FAKE_SYSTEMCTL_LIST_UNITS verbatim so a test can pin the
exact column layout it must parse.  Everything else succeeds.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    with open(STATE_PATH) as fh:
        state = json.load(fh)
    state.setdefault("calls", []).append(args)
    with open(STATE_PATH, "w") as fh:
        json.dump(state, fh)

    if args and args[0] == "list-units":
        sys.stdout.write(os.environ.get("FAKE_SYSTEMCTL_LIST_UNITS", ""))
    if args and args[0] == "status":
        return int(os.environ.get("FAKE_SYSTEMCTL_STATUS_RC", "0"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


@pytest.fixture
def baseline_dir(tmp_path, monkeypatch):
    """Point the per-arm baseline store at tmp_path.

    Without this a test would write into $XDG_RUNTIME_DIR and a real arm's
    baseline could be clobbered by the suite.
    """
    root = tmp_path / 'baselines'
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(root))
    return root


@pytest.fixture
def fake_systemctl(tmp_path, monkeypatch):
    """Shim a recording `systemctl` onto PATH; return a calls() accessor."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / 'systemctl'
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / 'systemctl_state.json'
    state_path.write_text(json.dumps({'calls': []}))

    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_SYSTEMCTL_STATE', str(state_path))

    class Shim:
        def __init__(self, bin_dir, state_path):
            self.bin_dir = bin_dir
            self.state_path = state_path

        def calls(self):
            return json.loads(self.state_path.read_text())['calls']

    return Shim(bin_dir, state_path)


def _arm(**overrides):
    fields = {
        'arm_id': 'qwen3.5-9b',
        'axis': 'llm',
        'stack': 'vllm',
        'image': 'vllm/vllm-openai:v0.26.0',
        'model_ref': 'QuantTrio/Qwen3.5-9B-AWQ',
        'quant': 'awq',
        'port': 8410,
        'served_model_name': 'qwen3.5-9b',
        'structured_output_mode': 'json_schema',
        'est_vram_gib': 6.0,
    }
    fields.update(overrides)
    return lms_manifest.ArmEntry(**fields)


# ---------------------------------------------------------------------------
# unit naming
# ---------------------------------------------------------------------------


def test_unit_name_is_the_templated_instance():
    assert lms_ctl.unit_name(_arm()) == 'lms-arm@qwen3.5-9b.service'
    assert lms_ctl.unit_name('moe-stretch') == 'lms-arm@moe-stretch.service'


def test_unit_name_round_trips_through_arm_id_from_unit():
    for arm_id in ('qwen3.5-9b', 'mistral-small-3.2-24b', 'gte-modernbert-base'):
        assert lms_ctl.arm_id_from_unit(lms_ctl.unit_name(arm_id)) == arm_id


# ---------------------------------------------------------------------------
# lifecycle verbs
# ---------------------------------------------------------------------------


def test_start_issues_exactly_one_systemctl_start(fake_systemctl, baseline_dir):
    lms_ctl.start(_arm(), gpu=MEASURED_GPU)

    assert fake_systemctl.calls() == [['start', 'lms-arm@qwen3.5-9b.service']]


def test_start_records_the_pre_start_gpu_reading_as_this_arm_s_baseline(
    fake_systemctl, baseline_dir,
):
    """The budget verdict's subtrahend is produced BY the start event.

    esc-3713-6 made a live per-arm baseline binding: a frozen constant
    misattributes desktop drift to the arm, and does so in the
    fabrication-relevant direction.  Capturing it here -- rather than accepting
    it as a healthcheck flag -- means the number cannot be typed in afterwards
    to make a report fit.
    """
    lms_ctl.start(_arm(), gpu=MEASURED_GPU)

    recorded = lms_vram.read_baseline('qwen3.5-9b')

    assert recorded.used_mib == MEASURED_GPU.used_mib
    assert recorded.free_mib == MEASURED_GPU.free_mib
    assert recorded.total_mib == MEASURED_GPU.total_mib


def test_a_refused_start_records_no_baseline(fake_systemctl, baseline_dir):
    """A baseline with no arm behind it would later be subtracted from some
    OTHER arm's reading, silently discounting it."""
    moe = _arm(arm_id='moe-stretch', stack='llamacpp',
               structured_output_mode='json_object', est_vram_gib=17.0, port=8413)

    with pytest.raises(lms_ctl.ArmPreflightError):
        lms_ctl.start(moe, gpu=MEASURED_GPU)

    assert not lms_vram.baseline_path('moe-stretch').exists()


def test_restarting_an_arm_overwrites_its_previous_baseline(
    fake_systemctl, baseline_dir,
):
    """A stale baseline from an earlier run would charge the new run for memory
    the earlier one had already released."""
    lms_ctl.start(_arm(), gpu=MEASURED_GPU)
    later = lms_vram.GpuReading(total_mib=24576, used_mib=8000, free_mib=16123)

    lms_ctl.start(_arm(), gpu=later)

    assert lms_vram.read_baseline('qwen3.5-9b').used_mib == 8000


def test_stop_issues_exactly_one_systemctl_stop(fake_systemctl):
    lms_ctl.stop(_arm())

    assert fake_systemctl.calls() == [['stop', 'lms-arm@qwen3.5-9b.service']]


def test_status_issues_systemctl_status_and_returns_its_exit_code(
    fake_systemctl, monkeypatch,
):
    """`systemctl status` exits 3 for an inactive unit; swallowing that into 0
    would make a dead arm read as a live one."""
    monkeypatch.setenv('FAKE_SYSTEMCTL_STATUS_RC', '3')

    rc = lms_ctl.status(_arm())

    assert rc == 3
    assert fake_systemctl.calls() == [['status', 'lms-arm@qwen3.5-9b.service']]


# ---------------------------------------------------------------------------
# active-set parsing
# ---------------------------------------------------------------------------


_LIST_UNITS_FIXTURE = (
    'lms-arm@qwen3.5-9b.service loaded active running Local model serving arm qwen3.5-9b\n'
    'lms-arm@phi-4-14b.service loaded failed failed Local model serving arm phi-4-14b\n'
    'lms-arm@moe-stretch.service loaded inactive dead Local model serving arm moe-stretch\n'
    'lms-arm@granite-embedding-english-r2.service loaded active running Local model serving arm granite\n'
)


def test_active_arms_returns_only_running_units(fake_systemctl, monkeypatch):
    monkeypatch.setenv('FAKE_SYSTEMCTL_LIST_UNITS', _LIST_UNITS_FIXTURE)

    active = lms_ctl.active_arms()

    assert active == {'qwen3.5-9b', 'granite-embedding-english-r2'}


def test_active_arms_queries_the_instance_glob(fake_systemctl, monkeypatch):
    monkeypatch.setenv('FAKE_SYSTEMCTL_LIST_UNITS', _LIST_UNITS_FIXTURE)

    lms_ctl.active_arms()

    assert fake_systemctl.calls() == [[
        'list-units', 'lms-arm@*.service', '--no-legend', '--plain',
    ]]


@pytest.mark.parametrize(
    'listing', ['', '\n', '0 loaded units listed.\n'], ids=['empty', 'blank', 'legend'],
)
def test_active_arms_tolerates_an_empty_listing(fake_systemctl, monkeypatch, listing):
    monkeypatch.setenv('FAKE_SYSTEMCTL_LIST_UNITS', listing)

    assert lms_ctl.active_arms() == set()


def test_active_arms_ignores_a_unit_that_is_not_ours(fake_systemctl, monkeypatch):
    monkeypatch.setenv(
        'FAKE_SYSTEMCTL_LIST_UNITS',
        'legibility-trickle@dark_factory.service loaded active running Trickle\n'
        + _LIST_UNITS_FIXTURE,
    )

    assert lms_ctl.active_arms() == {'qwen3.5-9b', 'granite-embedding-english-r2'}


# ---------------------------------------------------------------------------
# stop_all
# ---------------------------------------------------------------------------


def test_stop_all_stops_exactly_the_active_set(fake_systemctl, monkeypatch):
    monkeypatch.setenv('FAKE_SYSTEMCTL_LIST_UNITS', _LIST_UNITS_FIXTURE)

    stopped = lms_ctl.stop_all()

    assert sorted(stopped) == ['granite-embedding-english-r2', 'qwen3.5-9b']
    stop_calls = [c for c in fake_systemctl.calls() if c[0] == 'stop']
    assert sorted(stop_calls) == [
        ['stop', 'lms-arm@granite-embedding-english-r2.service'],
        ['stop', 'lms-arm@qwen3.5-9b.service'],
    ]


def test_stop_all_issues_nothing_when_no_arm_is_active(fake_systemctl, monkeypatch):
    monkeypatch.setenv('FAKE_SYSTEMCTL_LIST_UNITS', '')

    assert lms_ctl.stop_all() == []

    assert [c for c in fake_systemctl.calls() if c[0] == 'stop'] == []


# ---------------------------------------------------------------------------
# VRAM pre-flight — refusal BEFORE the side effect
# ---------------------------------------------------------------------------


def test_start_refuses_an_oversized_arm_without_issuing_any_systemctl_call(
    fake_systemctl,
):
    moe = _arm(arm_id='moe-stretch', stack='llamacpp',
               structured_output_mode='json_object', est_vram_gib=17.0, port=8413)

    with pytest.raises(lms_ctl.ArmPreflightError) as excinfo:
        lms_ctl.start(moe, gpu=MEASURED_GPU)

    message = str(excinfo.value)
    assert 'moe-stretch' in message
    assert '16.37' in message or '16.4' in message
    # The whole point: nothing reached systemd.
    assert fake_systemctl.calls() == []


def test_start_refuses_a_placeholder_arm_without_issuing_any_systemctl_call(
    fake_systemctl,
):
    placeholder = _arm(model_ref='TBD-Q3-pick-a-model', est_vram_gib=2.0)

    with pytest.raises(lms_ctl.ArmPreflightError) as excinfo:
        lms_ctl.start(placeholder, gpu=MEASURED_GPU)

    assert 'TBD' in str(excinfo.value)
    assert fake_systemctl.calls() == []


def test_start_refuses_when_another_arm_already_holds_the_gpu(
    fake_systemctl, monkeypatch,
):
    """Two arms resident at once is how the budget is blown; the PRD's funnel
    explicitly does not run all units simultaneously."""
    monkeypatch.setenv('FAKE_SYSTEMCTL_LIST_UNITS', _LIST_UNITS_FIXTURE)

    with pytest.raises(lms_ctl.ArmPreflightError) as excinfo:
        lms_ctl.start(_arm(arm_id='phi-4-14b', port=8412, est_vram_gib=9.0),
                      gpu=MEASURED_GPU, exclusive=True)

    assert 'qwen3.5-9b' in str(excinfo.value)
    assert [c for c in fake_systemctl.calls() if c[0] == 'start'] == []


# ---------------------------------------------------------------------------
# wait_ready — identity-verified readiness
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _models_payload(*names):
    return {'object': 'list', 'data': [{'id': n, 'object': 'model'} for n in names]}


def test_wait_ready_is_true_when_health_is_200_and_models_lists_the_arm(
    install_fake_httpx,
):
    arm = _arm()
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        if url.endswith('/health'):
            return _Resp(200)
        return _Resp(200, _models_payload('qwen3.5-9b'))

    install_fake_httpx(post=None, get=fake_get)

    assert lms_ctl.wait_ready(arm, timeout_s=0.0, interval_s=0.0) is True
    assert 'http://127.0.0.1:8410/health' in seen
    assert 'http://127.0.0.1:8410/v1/models' in seen
    # 127.0.0.1 explicitly: `localhost` can resolve to ::1 while the server
    # listens on IPv4 only (scripts/run_vllm_eval.py:505-512).
    assert not any('localhost' in url for url in seen)


def test_wait_ready_is_false_when_models_lists_a_different_model(install_fake_httpx):
    """The 2026-04-08 404 bug (scripts/run_vllm_eval.py:541-553): a /health 200
    on a colliding port made a DIFFERENT model look healthy and mis-attributed
    an entire run."""
    def fake_get(url, **kwargs):
        if url.endswith('/health'):
            return _Resp(200)
        return _Resp(200, _models_payload('some-other-model'))

    install_fake_httpx(post=None, get=fake_get)

    assert lms_ctl.wait_ready(_arm(), timeout_s=0.0, interval_s=0.0) is False


def test_wait_ready_returns_false_on_a_transport_error_rather_than_raising(
    install_fake_httpx,
):
    def fake_get(url, **kwargs):
        raise OSError('connection refused')

    install_fake_httpx(post=None, get=fake_get)

    assert lms_ctl.wait_ready(_arm(), timeout_s=0.0, interval_s=0.0) is False


def test_wait_ready_returns_false_when_health_never_goes_green(install_fake_httpx):
    def fake_get(url, **kwargs):
        return _Resp(503)

    install_fake_httpx(post=None, get=fake_get)

    assert lms_ctl.wait_ready(_arm(), timeout_s=0.0, interval_s=0.0) is False


# ---------------------------------------------------------------------------
# install-lms-units.sh
# ---------------------------------------------------------------------------


def _run_installer(tmp_path, shim, extra_env=None):
    env = dict(os.environ)
    env['PATH'] = f'{shim.bin_dir}{os.pathsep}{env["PATH"]}'
    env['FAKE_SYSTEMCTL_STATE'] = str(shim.state_path)
    env['XDG_CONFIG_HOME'] = str(tmp_path / 'config')
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ['bash', str(INSTALLER)],
        env=env, capture_output=True, text=True, timeout=60,
    )


def test_installer_copies_the_unit_template_verbatim(tmp_path, fake_systemctl):
    result = _run_installer(tmp_path, fake_systemctl)

    assert result.returncode == 0, result.stderr
    installed = tmp_path / 'config' / 'systemd' / 'user' / 'lms-arm@.service'
    assert installed.read_bytes() == UNIT_TEMPLATE.read_bytes()


def test_installer_reloads_the_user_daemon(tmp_path, fake_systemctl):
    _run_installer(tmp_path, fake_systemctl)

    assert ['daemon-reload'] in fake_systemctl.calls()


def test_installer_is_idempotent(tmp_path, fake_systemctl):
    first = _run_installer(tmp_path, fake_systemctl)
    second = _run_installer(tmp_path, fake_systemctl)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    installed = tmp_path / 'config' / 'systemd' / 'user' / 'lms-arm@.service'
    assert installed.read_bytes() == UNIT_TEMPLATE.read_bytes()


def test_installer_fails_loudly_when_the_unit_is_absent_after_reload(
    tmp_path, fake_systemctl,
):
    """`daemon-reload` can nominally succeed while the unit is unexpectedly
    absent; installing nothing observable must fail loud, not quietly."""
    result = _run_installer(
        tmp_path, fake_systemctl, extra_env={'LMS_INSTALL_SABOTAGE': '1'},
    )

    assert result.returncode != 0
    assert 'lms-arm@.service' in result.stderr


def test_installer_never_pipes_systemctl_into_grep():
    """Under `set -o pipefail`, `systemctl ... | grep -q` SIGPIPEs systemctl
    the instant grep matches, so the check reports "not found" on a MATCH
    (documented at scripts/legibility/install-trickle-timer.sh:64-69)."""
    source = INSTALLER.read_text()

    assert 'set -euo pipefail' in source
    for line in source.splitlines():
        if line.strip().startswith('#'):
            continue
        assert not ('systemctl' in line and '| grep' in line), line
