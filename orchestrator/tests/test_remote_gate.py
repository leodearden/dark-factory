"""Task 6565: the remote gate contract (orchestrator/remote_gate.py, design D2/D3/D5).

ensure-gate.sh and reap-gates.sh run here against real temporary directories
and git repositories, with a fake ``uv`` (which "builds" a gate binary) and a
fake ``orchestrator`` shim (which honours ``ORCH_GATE_BIN``) on PATH.  The
RemoteRunner tests drive the gate through a scripted transport, or through a
"host" that is this machine.  Nothing here reaches a remote host.

The INV-2 section replaces test_merge_verdict_integrity_inv2.py: a remote
verdict is adoptable only from a gate that answered the dispatcher's identity.
"""

from __future__ import annotations

import asyncio
import dataclasses
import fcntl
import json
import os
import shlex
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

from orchestrator.deployed_sha import DeployedSha
from orchestrator.event_store import EventType
from orchestrator.remote_gate import (
    GATE_ROOT_MARKER,
    EnsureReport,
    GateIdentity,
    GateIdentityPair,
    GateReplyError,
    classify_remote_exit,
    ensure_script_text,
    local_gate_identity,
    parse_ensure_report,
    parse_gate_identity,
    reap_script_text,
    wire_classes,
    wire_fingerprint,
)
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    REMOTE_TOOL_PATH_PRELUDE,
    DriftDetector,
    DriftVerdict,
    MergeVerifySpec,
    RemoteRunner,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    VerifyRunnerPool,
    result_to_json,
)

FAKE_FINGERPRINT = 'a' * 64

# ---------------------------------------------------------------------------
# wire_fingerprint
# ---------------------------------------------------------------------------


def test_wire_fingerprint_is_pinned_for_todays_wire_classes():
    """A changed vector means a wire shape changed: re-pin it deliberately.

    ``fused-memory/scripts/check_bare_magicmock_config.py`` keeps a second,
    exhaustive copy of ``VerifyResult``'s field list for its mock-shape rule;
    a field change must update it too (folding the two is a follow-up).
    """
    assert wire_fingerprint(wire_classes()) == (
        'f0b4dcde23f390891bfb1d7f71f097039f7ce022f94066c01feaedbf0cedfb10'
    )


def _one_class(name: str, *fields_: tuple[str, object]) -> type:
    return dataclasses.make_dataclass(name, list(fields_))


def test_adding_a_field_changes_the_fingerprint():
    before = _one_class('Verdict', ('passed', bool), ('summary', str))
    after = _one_class('Verdict', ('passed', bool), ('summary', str), ('leg', str))
    assert wire_fingerprint([before]) != wire_fingerprint([after])


def test_changing_a_field_type_changes_the_fingerprint():
    before = _one_class('Verdict', ('passed', bool))
    after = _one_class('Verdict', ('passed', 'int'))
    assert wire_fingerprint([before]) != wire_fingerprint([after])


def test_string_and_evaluated_annotations_fingerprint_alike():
    """``from __future__ import annotations`` modules and plain ones agree."""
    evaluated = _one_class('Leg', ('module', str | None), ('ids', tuple[str, ...]))
    stringly = _one_class('Leg', ('module', 'str | None'), ('ids', 'tuple[str, ...]'))
    assert wire_fingerprint([evaluated]) == wire_fingerprint([stringly])


def test_defaults_do_not_enter_the_fingerprint():
    """A default_factory's repr carries an address; it must not reach the hash."""
    with_default = dataclasses.make_dataclass(
        'Spec', [('env', dict, field(default_factory=lambda: {}))],
    )
    bare = _one_class('Spec', ('env', dict))
    assert wire_fingerprint([with_default]) == wire_fingerprint([bare])


def test_class_order_does_not_matter():
    a, b = _one_class('A', ('x', int)), _one_class('B', ('y', str))
    assert wire_fingerprint([a, b]) == wire_fingerprint([b, a])


def test_local_gate_identity_pairs_the_sha_with_this_process_fingerprint():
    identity = local_gate_identity('1' * 40)
    assert identity == GateIdentity('1' * 40, wire_fingerprint(wire_classes()))


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_IDENTITY = {'code_sha': '2' * 40, 'wire_fingerprint': FAKE_FINGERPRINT}


def test_parse_gate_identity_reads_one_identity_object():
    assert parse_gate_identity(json.dumps(_IDENTITY) + '\n') == GateIdentity('2' * 40, FAKE_FINGERPRINT)


@pytest.mark.parametrize('stdout', [
    '',
    'not json',
    json.dumps(_IDENTITY) + '\n' + json.dumps(_IDENTITY),
    json.dumps({**_IDENTITY, 'extra': 1}),
    json.dumps({'code_sha': '2' * 40}),
    json.dumps({**_IDENTITY, 'code_sha': 'HEAD'}),
    json.dumps({**_IDENTITY, 'wire_fingerprint': 'abc'}),
    json.dumps([_IDENTITY]),
])
def test_parse_gate_identity_rejects_anything_else(stdout: str):
    with pytest.raises(GateReplyError):
        parse_gate_identity(stdout)


def test_parse_ensure_report_reads_built_and_identity():
    stdout = json.dumps({'built': True, 'identity': _IDENTITY})
    assert parse_ensure_report(stdout) == EnsureReport(True, GateIdentity('2' * 40, FAKE_FINGERPRINT))


@pytest.mark.parametrize('reply', [
    {'built': 'yes', 'identity': _IDENTITY},
    {'identity': _IDENTITY},
    {'built': False, 'identity': {'code_sha': '2' * 40}},
])
def test_parse_ensure_report_rejects_a_malformed_reply(reply: dict):
    with pytest.raises(GateReplyError):
        parse_ensure_report(json.dumps(reply))


def test_identity_pair_matches_only_when_both_values_agree():
    local = GateIdentity('3' * 40, FAKE_FINGERPRINT)
    assert GateIdentityPair(local, local).matches
    assert not GateIdentityPair(local, GateIdentity('4' * 40, FAKE_FINGERPRINT)).matches
    assert not GateIdentityPair(local, GateIdentity('3' * 40, 'b' * 64)).matches
    assert GateIdentityPair(local, local).to_wire() == {
        'local': local.to_wire(), 'remote': local.to_wire(),
    }


# ---------------------------------------------------------------------------
# classify_remote_exit (design D5 + the 2026-10-20 Q1 ruling)
# ---------------------------------------------------------------------------

_PUSH_REFUSED_CONNECTION = (
    'ssh: connect to host leo-laptop port 22: Connection refused\r\n'
    'fatal: Could not read from remote repository.\n'
)


@pytest.mark.parametrize(('rc', 'stderr', 'channel', 'kind'), [
    (255, '', 'ssh', 'transport'),
    (None, '', 'ssh', 'transport'),
    (1, '', 'ssh', 'gate'),
    (2, "Error: No such command 'gate-identity'.", 'ssh', 'gate'),
    (127, 'orchestrator: command not found', 'ssh', 'gate'),
    (70, 'ensure-gate: rebuilt gate failed its handshake', 'ssh', 'gate'),
    # A remote command's own output cannot pass for the ssh client's.
    (1, 'ssh: connect to host qdrant port 6333: Connection refused', 'ssh', 'gate'),
    (255, '', 'git_push', 'transport'),
    (None, '', 'git_push', 'transport'),
    (128, _PUSH_REFUSED_CONNECTION, 'git_push', 'transport'),
    (128, 'ssh: connect to host leo-laptop port 22: Connection timed out', 'git_push', 'transport'),
    (128, 'ssh: Could not resolve hostname leo-laptop: Name or service not known', 'git_push', 'transport'),
    (128, 'kex_exchange_identification: read: Connection reset by peer', 'git_push', 'transport'),
    (128, 'Connection timed out during banner exchange', 'git_push', 'transport'),
    (128, 'Connection closed by 192.168.1.20 port 22', 'git_push', 'transport'),
    (1, ' ! [remote rejected] abc -> refs/gate/abc (deny updating a hidden ref)', 'git_push', 'gate'),
    (128, "fatal: '/home/leo/src/dark-factory' does not appear to be a git repository", 'git_push', 'gate'),
    (1, ' ! [remote rejected] main -> main (Working directory has unstaged changes)', 'git_push', 'gate'),
    (128, 'remote: error: refusing to update checked out branch', 'git_push', 'gate'),
    (128, 'error: an upstream test said: Connection refused', 'git_push', 'gate'),
])
def test_classify_remote_exit(rc, stderr, channel, kind):
    assert classify_remote_exit(rc, stderr, channel=channel) == kind


def test_classify_remote_exit_refuses_a_success():
    with pytest.raises(ValueError):
        classify_remote_exit(0)


# ---------------------------------------------------------------------------
# gate-identity CLI
# ---------------------------------------------------------------------------


def test_gate_identity_cli_prints_head_and_fingerprint():
    from orchestrator.cli import main
    from orchestrator.deployed_sha import checkout_head, resolve_code_checkout

    result = CliRunner().invoke(main, ['gate-identity'])

    assert result.exit_code == 0, result.output
    checkout = resolve_code_checkout()
    assert checkout is not None
    assert parse_gate_identity(result.output) == local_gate_identity(checkout_head(checkout))


def test_gate_identity_cli_fails_without_a_checkout(monkeypatch):
    from orchestrator.cli import main

    monkeypatch.setattr('orchestrator.deployed_sha.resolve_code_checkout', lambda: None)
    result = CliRunner().invoke(main, ['gate-identity'])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# ensure-gate.sh against a real repository
# ---------------------------------------------------------------------------

# Fake `uv sync`: "builds" a gate binary that answers gate-identity with its own
# checkout's HEAD.  $FAKE_UV_LOG counts builds; $FAKE_UV_BROKEN builds a binary
# that fails; $FAKE_UV_SLEEP widens the race window.
_FAKE_UV = r'''#!/usr/bin/env bash
set -eu
echo "uv $*" >> "$FAKE_UV_LOG"
sleep "${FAKE_UV_SLEEP:-0}"
mkdir -p .venv/bin
if [[ -n ${FAKE_UV_BROKEN:-} ]]; then
	printf '#!/usr/bin/env bash\necho broken >&2\nexit 1\n' > .venv/bin/orchestrator
else
	cat > .venv/bin/orchestrator <<'BIN'
#!/usr/bin/env bash
[[ $1 == gate-identity ]] || exit 2
sha=$(git -C "$(dirname "$0")/../.." rev-parse HEAD)
printf '{"code_sha": "%s", "wire_fingerprint": "%s"}\n' "$sha" "$(printf 'a%.0s' {1..64})"
BIN
fi
chmod +x .venv/bin/orchestrator
'''

# The ORCH_GATE_BIN-honouring host shim (design D3), minus its logging.
_FAKE_SHIM = '#!/usr/bin/env bash\nexec "${ORCH_GATE_BIN:?}" "$@"\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


@dataclass
class _Remote:
    root: Path
    source: Path
    sha: str
    env: dict[str, str]
    uv_log: Path

    @property
    def gate(self) -> Path:
        return self.root / self.sha

    def builds(self) -> int:
        return len(self.uv_log.read_text().splitlines()) if self.uv_log.exists() else 0

    def popen(self, **extra_env: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ['bash', '-c', ensure_script_text(), 'ensure-gate'],
            env={**self.env, **extra_env}, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )

    def ensure(self, **extra_env: str) -> subprocess.CompletedProcess[str]:
        proc = self.popen(**extra_env)
        stdout, stderr = proc.communicate(timeout=60)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

    def break_gate_binary(self) -> None:
        (self.gate / '.venv/bin/orchestrator').write_text('#!/usr/bin/env bash\nexit 1\n')

    def trash(self) -> list[Path]:
        return sorted(self.root.glob('.trash-*'))


@pytest.fixture
def remote(tmp_path: Path) -> _Remote:
    source = tmp_path / 'source'
    source.mkdir()
    _git(source, 'init', '-q')
    (source / 'README').write_text('df\n')
    _git(source, 'add', '-A')
    _git(source, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'one')
    sha = _git(source, 'rev-parse', 'HEAD')
    _git(source, 'update-ref', f'refs/gate/{sha}', sha)
    fakebin = tmp_path / 'bin'
    fakebin.mkdir()
    for name, text in (('uv', _FAKE_UV), ('orchestrator', _FAKE_SHIM)):
        (fakebin / name).write_text(text)
        (fakebin / name).chmod(0o755)
    uv_log = tmp_path / 'uv.log'
    env = {
        'PATH': f'{fakebin}:{os.environ["PATH"]}',
        'HOME': str(tmp_path),
        'GATE_ROOT': str(tmp_path / 'gates'),
        'SHA': sha,
        'SOURCE': str(source),
        'FAKE_UV_LOG': str(uv_log),
    }
    return _Remote(root=tmp_path / 'gates', source=source, sha=sha, env=env, uv_log=uv_log)


def test_ensure_script_text_is_the_packaged_script():
    packaged = Path(__file__).parents[1] / 'src/orchestrator/remote_gate/ensure-gate.sh'
    assert ensure_script_text() == packaged.read_text()


def test_first_ensure_builds_a_ready_gate_and_reports_its_identity(remote: _Remote):
    result = remote.ensure()

    assert result.returncode == 0, result.stderr
    assert parse_ensure_report(result.stdout) == EnsureReport(
        built=True, identity=GateIdentity(remote.sha, FAKE_FINGERPRINT),
    )
    assert (remote.gate / '.ready').exists()
    assert (remote.root / GATE_ROOT_MARKER).exists()
    assert _git(remote.gate, 'rev-parse', 'HEAD') == remote.sha
    assert remote.builds() == 1
    assert 'uv sync --all-packages --frozen' in remote.uv_log.read_text()
    assert _git(remote.source, 'for-each-ref', 'refs/gate') == ''


def test_a_ready_gate_is_reused_not_rebuilt(remote: _Remote):
    assert remote.ensure().returncode == 0
    result = remote.ensure()

    assert result.returncode == 0, result.stderr
    assert parse_ensure_report(result.stdout).built is False
    assert remote.builds() == 1


def test_a_gate_that_fails_its_first_handshake_is_never_marked_ready(remote: _Remote):
    result = remote.ensure(FAKE_UV_BROKEN='1')

    assert result.returncode not in (0, 255)
    assert result.stdout == ''
    assert 'failed its handshake' in result.stderr
    assert remote.gate.exists()
    assert not (remote.gate / '.ready').exists()


def test_a_non_ready_dir_is_trashed_and_rebuilt(remote: _Remote):
    remote.gate.mkdir(parents=True)
    (remote.gate / 'half-built').write_text('x')

    result = remote.ensure()

    assert result.returncode == 0, result.stderr
    assert parse_ensure_report(result.stdout).built is True
    assert not (remote.gate / 'half-built').exists()
    assert remote.trash() == []
    assert (remote.gate / '.ready').exists()


def test_a_build_that_keeps_failing_leaves_no_checkout_behind(remote: _Remote):
    """Each failed build is trashed by the next ensure, and the trash is deleted at once.

    A reprobe re-runs ensure every couple of minutes; keeping each failed
    checkout until the reaper's one-day rule would fill the remote disk.
    """
    for _ in range(3):
        assert remote.ensure(FAKE_UV_BROKEN='1').returncode not in (0, 255)

    assert remote.trash() == []
    assert sorted(p.name for p in remote.root.iterdir() if not p.name.startswith('.')) == [
        remote.sha,
    ]


def test_a_ready_gate_that_fails_its_handshake_is_rebuilt_once(remote: _Remote):
    assert remote.ensure().returncode == 0
    remote.break_gate_binary()

    result = remote.ensure()

    assert result.returncode == 0, result.stderr
    assert parse_ensure_report(result.stdout).built is True
    assert remote.builds() == 2
    assert remote.trash() == []


def test_a_ready_gate_that_fails_again_after_its_rebuild_exits_non_zero(remote: _Remote):
    assert remote.ensure().returncode == 0
    remote.break_gate_binary()

    result = remote.ensure(FAKE_UV_BROKEN='1')

    assert result.returncode not in (0, 255)
    assert 'rebuilt gate' in result.stderr
    assert not (remote.gate / '.ready').exists()
    assert remote.builds() == 2


def test_concurrent_ensures_of_one_sha_build_it_once(remote: _Remote):
    procs = [remote.popen(FAKE_UV_SLEEP='1') for _ in range(2)]
    outputs = [proc.communicate(timeout=60) for proc in procs]

    assert [proc.returncode for proc in procs] == [0, 0], [err for _, err in outputs]
    assert sorted(parse_ensure_report(out).built for out, _ in outputs) == [False, True]
    assert remote.builds() == 1


def test_every_ensure_refreshes_the_gate_dir_mtime(remote: _Remote):
    assert remote.ensure().returncode == 0
    old = time.time() - 30 * 86400
    os.utime(remote.gate, (old, old))

    assert remote.ensure().returncode == 0
    assert remote.gate.stat().st_mtime > old + 29 * 86400


def test_the_build_lock_file_is_left_in_place(remote: _Remote):
    assert remote.ensure().returncode == 0
    assert (remote.root / f'.build-{remote.sha}.lock').is_file()


def test_ensure_never_exits_with_the_ssh_transport_status(remote: _Remote):
    """rc 255 means 'the ssh client failed'; a gate failure must not look like one."""
    (Path(remote.env['PATH'].split(':')[0]) / 'orchestrator').write_text(
        '#!/usr/bin/env bash\nexit 255\n',
    )
    result = remote.ensure()
    assert result.returncode not in (0, 255)


@pytest.mark.parametrize(('name', 'value'), [
    ('SHA', 'main'),
    ('GATE_ROOT', 'relative/gates'),
    ('SOURCE', ''),
])
def test_ensure_rejects_a_malformed_contract(remote: _Remote, name: str, value: str):
    result = remote.ensure(**{name: value})
    assert result.returncode == 64
    assert not remote.root.exists() or not remote.gate.exists()


# ---------------------------------------------------------------------------
# RemoteRunner end to end: real git, the real ensure script, "ssh" run locally
# (acceptance 1 of task 6565: the gate is the injected deployed sha, never
# the code checkout's HEAD)
# ---------------------------------------------------------------------------

_VERDICT = VerifyResult(
    passed=True, test_output='ok', lint_output='', type_output='', summary='gate verified',
)


def _gate_uv(fingerprint: str, verdict_json: str) -> str:
    """A fake ``uv sync`` building a gate binary that answers like a real one.

    ``gate-identity`` prints its own checkout's HEAD with *fingerprint*;
    ``verify-merge`` logs that HEAD and its arguments to ``$GATE_VERIFY_LOG``
    and prints *verdict_json*.
    """
    return (
        '#!/usr/bin/env bash\nset -eu\nmkdir -p .venv/bin\n'
        "cat > .venv/bin/orchestrator <<'BIN'\n"
        '#!/usr/bin/env bash\n'
        'here=$(git -C "$(dirname "$0")/../.." rev-parse HEAD)\n'
        'case $1 in\n'
        'gate-identity) printf \'{"code_sha": "%s", "wire_fingerprint": "%s"}\\n\' '
        f'"$here" {fingerprint} ;;\n'
        'verify-merge) echo "$here $*" >> "$GATE_VERIFY_LOG"; '
        f'printf "%s\\n" {shlex.quote(verdict_json)} ;;\n'
        '*) exit 2 ;;\n'
        'esac\n'
        'BIN\n'
        'chmod +x .venv/bin/orchestrator\n'
    )


class _LocalHost:
    """A RemoteRunner ``run`` for a host that is this machine.

    ``ssh <opts> gate-host <cmd>`` runs ``bash -c <cmd>`` as a non-login ssh
    shell would: a bare system PATH (*sysbin* standing in for /usr/local/bin)
    and HOME, nothing else.  ``git push gate-host:<path> ...`` pushes to
    ``<path>``.  Everything else runs as given.
    """

    def __init__(self, *, sysbin: Path, home: Path, verify_log: Path) -> None:
        self.ssh_env = {
            'PATH': f'{sysbin}:/usr/bin:/bin', 'HOME': str(home),
            'GATE_VERIFY_LOG': str(verify_log),
        }
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], *, cwd: str | Path | None = None) -> tuple[int, str, str]:
        self.calls.append(list(argv))
        env = None
        cmd = list(argv)
        if argv[0] == 'ssh':
            cmd, env = ['bash', '-c', argv[-1]], self.ssh_env
        elif argv[:2] == ['git', 'push'] and argv[2].startswith('gate-host:'):
            cmd[2] = argv[2].removeprefix('gate-host:')
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return proc.returncode or 0, out.decode().strip(), err.decode().strip()


@dataclass
class _GateScene:
    dispatcher: Path
    deployed: str
    head: str
    source: Path
    project: Path
    gates: Path
    verify_log: Path
    host: _LocalHost

    def runner(self) -> RemoteRunner:
        return RemoteRunner(
            'laptop', 'gate-host', str(self.project), str(self.dispatcher),
            df_remote_checkout=str(self.source),
            gate_checkouts_path=str(self.gates),
            deployed_sha=DeployedSha(self.deployed, indeterminate=False, dirty_wire_path=False),
            code_checkout=str(self.dispatcher),
            run=self.host.run,
            id_factory=iter(('rid-1', 'rid-2')).__next__,
        )


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name)
    _git(repo, 'add', '-A')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', name)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def gate_scene(tmp_path: Path) -> _GateScene:
    dispatcher = tmp_path / 'dispatcher'
    dispatcher.mkdir()
    _git(dispatcher, 'init', '-q')
    deployed = _commit(dispatcher, 'deployed')
    head = _commit(dispatcher, 'landed-after-start')
    source = tmp_path / 'laptop-df'
    source.mkdir()
    _git(source, 'init', '-q')
    project = tmp_path / 'project.git'
    _git(tmp_path, 'init', '-q', '--bare', str(project))
    sysbin, home = tmp_path / 'sysbin', tmp_path / 'home'
    (home / '.local/bin').mkdir(parents=True)
    sysbin.mkdir()
    uv = home / '.local/bin/uv'
    uv.write_text(_gate_uv(wire_fingerprint(wire_classes()), result_to_json(_VERDICT)))
    shim = sysbin / 'orchestrator'
    shim.write_text(_FAKE_SHIM)
    for tool in (uv, shim):
        tool.chmod(0o755)
    verify_log = tmp_path / 'gate-verify.log'
    return _GateScene(
        dispatcher=dispatcher, deployed=deployed, head=head, source=source,
        project=project, gates=tmp_path / 'gates', verify_log=verify_log,
        host=_LocalHost(sysbin=sysbin, home=home, verify_log=verify_log),
    )


def _spec() -> MergeVerifySpec:
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )


@pytest.mark.asyncio
async def test_a_dispatch_verifies_in_the_deployed_shas_gate_not_the_checkout_head(
    gate_scene: _GateScene,
):
    """The checkout moved on after the process started; the gate did not.

    Runs the real publish push, the real ensure-gate.sh (uv found on the
    non-login shell's PATH only through REMOTE_TOOL_PATH_PRELUDE) and the
    verify through the host shim with ORCH_GATE_BIN.
    """
    scene = gate_scene
    store = _RecordingEvents()

    result = await scene.runner().run_merge_verify(
        scene.head, _spec(), event_store=store, task_id='t-6565',
    )

    assert result == _VERDICT
    gate = scene.gates / scene.deployed
    assert _git(gate, 'rev-parse', 'HEAD') == scene.deployed
    assert (gate / '.ready').exists()
    assert not (scene.gates / scene.head).exists()
    [ran] = scene.verify_log.read_text().splitlines()
    assert ran.startswith(f'{scene.deployed} verify-merge --sha {scene.head} ')
    assert _git(scene.source, 'for-each-ref', 'refs/gate') == ''
    assert _git(scene.project, 'for-each-ref', 'refs/merge-verify') == ''
    assert store.of(EventType.runner_stale) == [
        {'runner': 'laptop', 'local_head': scene.deployed, 'remote_head': None},
    ]
    assert store.of(EventType.runner_synced) == [{
        'runner': 'laptop', 'kind': 'gate_checkout',
        'from_head': None, 'to_head': scene.deployed, 'forced': False,
    }]


@pytest.mark.asyncio
async def test_a_second_dispatch_reuses_the_gate_silently(gate_scene: _GateScene):
    runner = gate_scene.runner()
    await runner.run_merge_verify(gate_scene.head, _spec())
    store = _RecordingEvents()

    assert await runner.run_merge_verify(gate_scene.head, _spec(), event_store=store) == _VERDICT

    assert store.events == []
    assert len(gate_scene.verify_log.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# RemoteRunner gate failures, scripted transport (design D3/D5)
# ---------------------------------------------------------------------------

_SHA = 'e' * 40
_GATE_BIN = f'/gates/{_SHA}/.venv/bin/orchestrator'


class _RecordingEvents:
    def __init__(self) -> None:
        self.events: list[tuple[Any, dict[str, Any]]] = []

    def emit(self, event_type: Any, *, data: dict[str, Any] | None = None, **_: Any) -> None:
        self.events.append((event_type, dict(data or {})))

    def of(self, event_type: Any) -> list[dict[str, Any]]:
        return [data for kind, data in self.events if kind == event_type]


_Reply = tuple[int, str, str] | BaseException | Callable[[], Awaitable[tuple[int, str, str]]]


class _Scripted:
    """Answers each RemoteRunner command by its role and records it."""

    def __init__(self, **replies: _Reply) -> None:
        identity = local_gate_identity(_SHA).to_wire()
        self.replies: dict[str, _Reply] = {
            'publish': (0, '', ''),
            'ensure': (0, json.dumps({'built': False, 'identity': identity}), ''),
            'verify_push': (0, '', ''),
            'delete': (0, '', ''),
            'verify': (0, result_to_json(_VERDICT), ''),
            'cancel': (0, '', ''),
            'reap': (0, '', ''),
            'other': (0, '', ''),
            **replies,
        }
        self.calls: list[tuple[str, list[str], object]] = []

    @staticmethod
    def role(argv: list[str]) -> str:
        if argv[:2] == ['git', 'push']:
            if '--delete' in argv:
                return 'delete'
            return 'publish' if ':refs/gate/' in argv[3] else 'verify_push'
        if argv[0] == 'ssh':
            if argv[-1].endswith(' ensure-gate'):
                return 'ensure'
            if ' reap-gates ' in argv[-1]:
                return 'reap'
            return 'cancel' if 'cancel-verify' in argv[-1] else 'verify'
        return 'other'

    def roles(self) -> list[str]:
        return [role for role, _, _ in self.calls]

    async def run(self, argv: list[str], *, cwd: object = None) -> tuple[int, str, str]:
        role = self.role(argv)
        self.calls.append((role, list(argv), cwd))
        reply = self.replies[role]
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return await reply()
        return reply


def _scripted_runner(transport: _Scripted, **overrides: Any) -> RemoteRunner:
    kwargs: dict[str, Any] = {
        'df_remote_checkout': '/laptop/dark-factory',
        'gate_checkouts_path': '/gates',
        'deployed_sha': DeployedSha(_SHA, indeterminate=False, dirty_wire_path=False),
        'code_checkout': '/dispatcher/checkout',
        'run': transport.run,
        'id_factory': lambda: 'rid',
        **overrides,
    }
    return RemoteRunner('laptop', 'leo-laptop', 'leo-laptop', '/project', **kwargs)


async def _unavailable(transport: _Scripted) -> RunnerUnavailable:
    with pytest.raises(RunnerUnavailable) as excinfo:
        await _scripted_runner(transport).run_merge_verify('mergesha', _spec())
    return excinfo.value


@pytest.mark.asyncio
async def test_publish_pushes_the_deployed_sha_by_url_from_the_code_checkout():
    transport = _Scripted()

    await _scripted_runner(transport).run_merge_verify('mergesha', _spec())

    assert transport.roles() == ['publish', 'ensure', 'verify_push', 'verify', 'delete']
    _, publish, cwd = transport.calls[0]
    assert publish == ['git', 'push', 'leo-laptop:/laptop/dark-factory', f'{_SHA}:refs/gate/{_SHA}']
    assert cwd == '/dispatcher/checkout'
    ensure = transport.calls[1][1][-1]
    assert ensure.startswith(f'{REMOTE_TOOL_PATH_PRELUDE} env GATE_ROOT=/gates SHA={_SHA} ')
    assert 'SOURCE=/laptop/dark-factory bash -c ' in ensure
    assert gate_argv_for(transport.calls[3][1][-1])[:3] == ['verify-merge', '--sha', 'mergesha']


def gate_argv_for(remote_cmd: str) -> list[str]:
    words = shlex.split(remote_cmd)
    assert words[:3] == ['env', f'ORCH_GATE_BIN={_GATE_BIN}', 'orchestrator'], words
    return words[3:]


@pytest.mark.asyncio
async def test_a_gate_answering_another_commit_is_wire_shape_before_any_verify():
    """An un-updated host shim runs a shared tree that answers with its own HEAD."""
    other = GateIdentity('0' * 40, wire_fingerprint(wire_classes()))
    transport = _Scripted(ensure=(0, json.dumps({'built': False, 'identity': other.to_wire()}), ''))

    exc = await _unavailable(transport)

    assert exc.kind == 'wire_shape'
    assert exc.identity == GateIdentityPair(local=local_gate_identity(_SHA), remote=other)
    assert 'install_laptop_gate_shim.sh' in str(exc)
    assert transport.roles() == ['publish', 'ensure']


@pytest.mark.asyncio
async def test_a_fingerprint_mismatch_is_wire_shape_without_blaming_the_shim():
    other = GateIdentity(_SHA, 'f' * 64)
    transport = _Scripted(ensure=(0, json.dumps({'built': True, 'identity': other.to_wire()}), ''))

    exc = await _unavailable(transport)

    assert exc.kind == 'wire_shape'
    assert exc.identity is not None and exc.identity.remote == other
    assert 'install_laptop_gate_shim.sh' not in str(exc)
    assert transport.roles() == ['publish', 'ensure']


@pytest.mark.asyncio
@pytest.mark.parametrize(('ensure', 'kind', 'shim'), [
    ((255, '', 'ssh: connect to host leo-laptop port 22: No route to host'), 'transport', False),
    (OSError('no ssh binary'), 'transport', False),
    ((70, '', 'ensure-gate: fresh gate failed its handshake'), 'gate', False),
    ((75, '', 'ensure-gate: build lock not acquired within 300s'), 'gate', False),
    ((2, '', "Error: No such command 'gate-identity'.\nensure-gate: fresh gate failed"), 'gate', True),
    ((0, 'not json', ''), 'gate', False),
    ((0, json.dumps({'built': 'yes'}), ''), 'gate', False),
])
async def test_an_ensure_failure_is_classed_and_spends_no_verify(ensure, kind, shim):
    transport = _Scripted(ensure=ensure)

    exc = await _unavailable(transport)

    assert exc.kind == kind
    assert ('install_laptop_gate_shim.sh' in str(exc)) is shim
    assert transport.roles() == ['publish', 'ensure']


@pytest.mark.asyncio
async def test_an_ensure_that_never_answers_is_gate(monkeypatch):
    monkeypatch.setattr('orchestrator.verify_runner.ENSURE_CLIENT_TIMEOUT_SECS', 0.05)

    async def hang() -> tuple[int, str, str]:
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    exc = await _unavailable(_Scripted(ensure=hang))

    assert exc.kind == 'gate'
    assert 'exceeded' in str(exc)


@pytest.mark.asyncio
@pytest.mark.parametrize(('stderr', 'kind'), [
    ('ssh: connect to host leo-laptop port 22: Connection refused\nfatal: Could not read', 'transport'),
    (' ! [remote rejected] e -> refs/gate/e (deny updating a hidden ref)', 'gate'),
])
async def test_a_failed_publish_is_classed_on_the_push_channel_and_runs_no_ensure(stderr, kind):
    transport = _Scripted(publish=(128, '', stderr))

    exc = await _unavailable(transport)

    assert exc.kind == kind
    assert transport.roles() == ['publish']


@pytest.mark.asyncio
@pytest.mark.parametrize(('verify', 'kind'), [
    ((255, '', 'Connection to leo-laptop closed by remote host.'), 'transport'),
    ((1, '', 'Error: config not found'), 'gate'),
    ((127, '', 'orchestrator: command not found'), 'gate'),
])
async def test_a_failed_verify_command_is_classed_by_its_exit(verify, kind):
    exc = await _unavailable(_Scripted(verify=verify))
    assert exc.kind == kind


@pytest.mark.asyncio
async def test_cancel_verify_runs_through_the_gate_binary():
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_verify() -> tuple[int, str, str]:
        entered.set()
        await release.wait()
        return (137, '', 'Killed')

    transport = _Scripted(verify=held_verify)
    runner = _scripted_runner(transport, config_path='/etc/orch.yaml')
    dispatch = asyncio.create_task(runner.run_merge_verify('mergesha', _spec()))
    await entered.wait()

    assert await runner.cancel_verify() == 0
    release.set()
    with pytest.raises(RunnerUnavailable) as excinfo:
        await dispatch

    [(_, cancel, _)] = [call for call in transport.calls if call[0] == 'cancel']
    assert gate_argv_for(cancel[-1]) == [
        'cancel-verify', '--request-id', 'rid', '--config', '/etc/orch.yaml',
    ]
    # Killed by this dispatcher's own cancel: says nothing about the host.
    assert excinfo.value.kind == 'transport'


@pytest.mark.asyncio
async def test_the_same_kill_without_a_cancel_is_gate():
    exc = await _unavailable(_Scripted(verify=(137, '', 'Killed')))
    assert exc.kind == 'gate'


def test_a_relative_gate_root_is_refused():
    with pytest.raises(ValueError, match='absolute'):
        _scripted_runner(_Scripted(), gate_checkouts_path='gates')


def test_a_runner_needs_a_checkout_to_publish_from(monkeypatch):
    monkeypatch.setattr('orchestrator.verify_runner.resolve_code_checkout', lambda: None)
    with pytest.raises(ValueError, match='cannot be published'):
        _scripted_runner(_Scripted(), code_checkout=None)


# ---------------------------------------------------------------------------
# INV-2 (task 2884), fail closed through the gate: the replacement for
# test_merge_verdict_integrity_inv2.py.  A remote verdict is adoptable only
# from a gate that answered this dispatcher's identity; a host that cannot
# produce one is benched before any verify, and the bench is never swallowed.
# ---------------------------------------------------------------------------


class _Local:
    name = 'local'
    is_local: ClassVar[bool] = True

    def __init__(self) -> None:
        self.calls = 0

    async def health(self) -> bool:
        return True

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        self.calls += 1
        return dataclasses.replace(_VERDICT, summary='local verdict')


_BROKEN_GATE = (70, '', 'ensure-gate: fresh gate failed its handshake')


@pytest.mark.asyncio
async def test_inv2_single_runner_pool_raises_and_takes_no_verdict():
    transport = _Scripted(ensure=_BROKEN_GATE)
    store = _RecordingEvents()
    pool = VerifyRunnerPool([_scripted_runner(transport)], event_store=store, task_id='t')

    with pytest.raises(RunnerUnavailable) as excinfo:
        await pool.dispatch('mergesha', _spec())

    assert excinfo.value.kind == 'gate'
    assert 'verify_push' not in transport.roles()
    assert store.of(EventType.merge_verify) == []


@pytest.mark.asyncio
async def test_inv2_two_runner_pool_reports_the_bench_and_adopts_local():
    transport = _Scripted(ensure=_BROKEN_GATE)
    store = _RecordingEvents()
    benched: list[tuple[str, RunnerUnavailable]] = []
    local = _Local()
    pool = VerifyRunnerPool(
        [_scripted_runner(transport), local], event_store=store, task_id='t',
        on_unavailable=lambda name, exc: benched.append((name, exc)),
    )

    result = await pool.dispatch('mergesha', _spec())

    assert result.summary == 'local verdict'
    assert [(name, exc.kind) for name, exc in benched] == [('laptop', 'gate')]
    assert [e['runner'] for e in store.of(EventType.merge_verify)] == ['local']
    assert 'verify_push' not in transport.roles()


@pytest.mark.asyncio
async def test_inv2_a_built_gate_announces_stale_then_synced_and_its_verdict_is_adopted():
    identity = local_gate_identity(_SHA).to_wire()
    transport = _Scripted(ensure=(0, json.dumps({'built': True, 'identity': identity}), ''))
    store = _RecordingEvents()
    pool = VerifyRunnerPool([_scripted_runner(transport), _Local()], event_store=store, task_id='t')

    assert await pool.dispatch('mergesha', _spec()) == _VERDICT

    assert [kind for kind, _ in store.events] == [
        EventType.runner_stale, EventType.runner_synced, EventType.merge_verify,
    ]
    assert store.of(EventType.merge_verify)[0]['runner'] == 'laptop'


@pytest.mark.asyncio
async def test_inv2_a_drift_check_reports_the_unusable_remote_and_stays_inconclusive():
    benched: list[str] = []
    pool = VerifyRunnerPool(
        [_Local(), _scripted_runner(_Scripted(ensure=_BROKEN_GATE))],
        on_unavailable=lambda name, exc: benched.append(f'{name}:{exc.kind}'),
    )

    result = await DriftDetector(pool).check('mergesha', _spec())

    assert result.verdict == DriftVerdict.INCONCLUSIVE
    assert benched == ['laptop:gate']


# ---------------------------------------------------------------------------
# reap-gates.sh against real directories (design D2)
# ---------------------------------------------------------------------------


def _age(path: Path, days: float) -> None:
    then = time.time() - days * 86400
    os.utime(path, (then, then))


def _reap(root: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', '-c', reap_script_text(), 'reap-gates'],
        env={'PATH': os.environ['PATH'], 'GATE_ROOT': str(root), **env},
        capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def gate_root(tmp_path: Path) -> Path:
    root = tmp_path / 'gates'
    root.mkdir()
    (root / GATE_ROOT_MARKER).touch()
    return root


def _dir(root: Path, name: str, days: float) -> Path:
    path = root / name
    path.mkdir()
    (path / 'file').write_text('x')
    _age(path, days)
    return path


def test_the_reaper_deletes_idle_sha_dirs_and_old_trash_only(gate_root: Path):
    idle = _dir(gate_root, '1' * 40, days=8)
    used = _dir(gate_root, '2' * 40, days=6)
    not_a_sha = _dir(gate_root, 'notes', days=30)
    old_trash = _dir(gate_root, '.trash-old', days=2)
    new_trash = _dir(gate_root, '.trash-new', days=0.5)
    lock = gate_root / f'.build-{"3" * 40}.lock'
    lock.touch()
    _age(lock, 30)

    result = _reap(gate_root)

    assert result.returncode == 0, result.stderr
    assert not idle.exists() and not old_trash.exists()
    assert used.exists() and not_a_sha.exists() and new_trash.exists() and lock.exists()
    assert not [p for p in gate_root.glob('.trash-*') if p != new_trash]


def test_the_reaper_leaves_an_unmarked_root_alone(gate_root: Path):
    (gate_root / GATE_ROOT_MARKER).unlink()
    idle = _dir(gate_root, '1' * 40, days=30)

    assert _reap(gate_root).returncode == 0
    assert idle.exists()


def test_the_reaper_skips_a_sha_whose_build_lock_is_held(gate_root: Path):
    sha = '4' * 40
    idle = _dir(gate_root, sha, days=30)
    with open(gate_root / f'.build-{sha}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert _reap(gate_root).returncode == 0
    assert idle.exists()


def test_the_reaper_rejects_a_relative_root(gate_root: Path):
    assert _reap(gate_root, GATE_ROOT='gates').returncode == 64


def test_reap_script_text_is_the_packaged_script():
    packaged = Path(__file__).parents[1] / 'src/orchestrator/remote_gate/reap-gates.sh'
    assert reap_script_text() == packaged.read_text()


@pytest.mark.asyncio
async def test_reap_idle_gates_launches_the_reaper_detached():
    transport = _Scripted()

    await _scripted_runner(transport).reap_idle_gates()

    [(_, argv, _)] = transport.calls
    command = argv[-1]
    assert command.startswith(f'{REMOTE_TOOL_PATH_PRELUDE} nohup env GATE_ROOT=/gates bash -c ')
    assert command.endswith(' reap-gates </dev/null >/dev/null 2>&1 &')


@pytest.mark.asyncio
@pytest.mark.parametrize('reply', [(255, '', 'unreachable'), OSError('no ssh')])
async def test_reap_idle_gates_never_raises(reply):
    await _scripted_runner(_Scripted(reap=reply)).reap_idle_gates()
