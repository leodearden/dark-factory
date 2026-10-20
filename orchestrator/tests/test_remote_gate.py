"""Task 6565: the remote gate contract (orchestrator/remote_gate.py, design D2/D3/D5).

ensure-gate.sh runs here against a real temporary git repository, with a fake
``uv`` (which "builds" a gate binary) and a fake ``orchestrator`` shim (which
honours ``ORCH_GATE_BIN``) on PATH.  Nothing here reaches a remote host.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

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
    wire_classes,
    wire_fingerprint,
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


def test_a_non_ready_dir_is_moved_to_trash_and_rebuilt(remote: _Remote):
    remote.gate.mkdir(parents=True)
    (remote.gate / 'half-built').write_text('x')

    result = remote.ensure()

    assert result.returncode == 0, result.stderr
    assert parse_ensure_report(result.stdout).built is True
    [trashed] = remote.trash()
    assert (trashed / 'half-built').exists()
    assert (remote.gate / '.ready').exists()


def test_a_ready_gate_that_fails_its_handshake_is_rebuilt_once(remote: _Remote):
    assert remote.ensure().returncode == 0
    remote.break_gate_binary()

    result = remote.ensure()

    assert result.returncode == 0, result.stderr
    assert parse_ensure_report(result.stdout).built is True
    assert remote.builds() == 2
    assert len(remote.trash()) == 1


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
