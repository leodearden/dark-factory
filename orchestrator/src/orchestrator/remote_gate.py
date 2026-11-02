"""The remote gate contract: what a per-SHA gate checkout must answer (task 6565, D2/D3/D5).

A remote verify host runs the dispatcher's own code from an immutable checkout
of the dispatcher's deployed sha, built by the package script
``remote_gate/ensure-gate.sh``.  Before any verify is spent on it, the gate
proves who it is with ``orchestrator gate-identity``: the sha it was built from
and a fingerprint of the verify wire shapes.  This module owns that identity,
its one parser, the script, and the one rule that classes every remote failure.

It imports nothing from the wire path at module level, so ``verify_runner`` and
``cli`` may import it; :func:`wire_classes` imports the wire classes lazily.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import typing
from dataclasses import dataclass
from importlib import resources
from typing import Literal

# Why a remote host produced no usable verdict.  The single vocabulary of
# ``verify_runner.RunnerUnavailable.kind``; the design's D6 table gives each
# kind its quarantine class, reprobe and escalation.
RunnerUnavailableKind = Literal['transport', 'gate', 'unparseable', 'no_verdict', 'wire_shape']

# How a remote command reached the host: run over ``ssh``, or a ``git push``
# whose own transport is ssh.  The two report a connect failure differently.
RemoteChannel = Literal['ssh', 'git_push']

# The ssh client's exit status for its own failure (connect, auth, a dropped
# link).  ensure-gate.sh never exits with it, so it always means transport.
SSH_TRANSPORT_FAILURE_RC = 255

# Client-side bound on one ensure round trip: the script's 300 s build-lock
# wait plus a cold ``uv sync``.  Hitting it classes as 'gate' (design D5).
ENSURE_CLIENT_TIMEOUT_SECS = 600

# The marker ensure-gate.sh writes in every gate root; a reaper deletes only
# under a root that carries it.
GATE_ROOT_MARKER = '.dark-factory-gate-root'

_FULL_SHA = re.compile(r'[0-9a-f]{40}')
_SHA256_HEX = re.compile(r'[0-9a-f]{64}')

# Lines the ssh CLIENT writes when it could not reach the host, or lost it
# mid-session (a laptop going to sleep during a push).  A git push over ssh
# exits 128 for these and for every other refusal alike, so its stderr is the
# only place the cause shows.  Anchored at line start so a remote program's
# output (which git prefixes "remote: ") cannot match by mentioning a refused
# connection.
_SSH_CONNECT_FAILURE = re.compile(
    r'^(?:ssh: (?:connect to host |Could not resolve hostname )'
    r'|kex_exchange_identification: '
    r'|Connection (?:closed|reset|timed out) '
    r'|Connection to \S+ closed by remote host'
    r'|Timeout, server \S+ not responding'
    r'|client_loop: send disconnect: '
    r'|ssh_exchange_identification: )',
    re.MULTILINE,
)


@dataclass(frozen=True)
class GateIdentity:
    """Who a gate is: the commit its code comes from and its wire fingerprint."""

    code_sha: str
    wire_fingerprint: str

    def to_wire(self) -> dict[str, str]:
        return {'code_sha': self.code_sha, 'wire_fingerprint': self.wire_fingerprint}


@dataclass(frozen=True)
class GateIdentityPair:
    """The dispatcher's own identity beside the one a remote gate answered."""

    local: GateIdentity
    remote: GateIdentity

    @property
    def matches(self) -> bool:
        return self.local == self.remote

    def to_wire(self) -> dict[str, dict[str, str]]:
        return {'local': self.local.to_wire(), 'remote': self.remote.to_wire()}


@dataclass(frozen=True)
class EnsureReport:
    """What one ensure-gate.sh run reports on stdout.

    ``built`` is True when this run built (or rebuilt) the gate dir, i.e. the
    remote did not have the sha ready; ``identity`` is the gate's handshake.
    """

    built: bool
    identity: GateIdentity


class GateReplyError(ValueError):
    """A gate's stdout is not the documented reply; the caller classes it 'gate'."""


def _annotation_text(annotation: object) -> str:
    return annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)


def _field_shapes(cls: type) -> list[list[str]]:
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}
    return [
        [f.name, _annotation_text(hints.get(f.name, f.type))]
        for f in dataclasses.fields(cls)
    ]


def wire_fingerprint(classes: typing.Iterable[type]) -> str:
    """sha256 over the field shapes of the wire dataclasses *classes*.

    Classes sorted by qualified name; each contributes its fields in
    declaration order as ``(name, annotation text)``, annotations resolved
    through ``typing.get_type_hints`` (whether or not the module uses
    ``from __future__ import annotations``) and rendered by
    ``inspect.formatannotation``, falling back to the raw annotation.
    Defaults are excluded: a ``default_factory`` repr carries an address and
    would differ between processes running identical code.
    """
    shapes = [[cls.__qualname__, _field_shapes(cls)] for cls in classes]
    shapes.sort(key=lambda shape: shape[0])
    payload = json.dumps(shapes, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('ascii')).hexdigest()


def wire_classes() -> tuple[type, ...]:
    """The dataclasses whose shape crosses the dispatcher/remote boundary."""
    from orchestrator.flake_ledger import FlakeSuppression
    from orchestrator.task_verify_spec import TaskVerifySpec
    from orchestrator.verify import VerifyResult
    from orchestrator.verify_admission_record import AdmissionLeg
    from orchestrator.verify_evidence import VerifyUnit
    from orchestrator.verify_runner import MergeVerifySpec

    return (
        VerifyResult,
        MergeVerifySpec,
        TaskVerifySpec,
        AdmissionLeg,
        FlakeSuppression,
        VerifyUnit,
    )


def local_gate_identity(code_sha: str) -> GateIdentity:
    """This process's identity, for a gate built from *code_sha*."""
    return GateIdentity(code_sha=code_sha, wire_fingerprint=wire_fingerprint(wire_classes()))


def _identity_from_mapping(value: object) -> GateIdentity:
    if not isinstance(value, dict) or set(value) != {'code_sha', 'wire_fingerprint'}:
        raise GateReplyError(f'gate identity must be {{code_sha, wire_fingerprint}}, got {value!r}')
    code_sha, fingerprint = value['code_sha'], value['wire_fingerprint']
    if not (isinstance(code_sha, str) and _FULL_SHA.fullmatch(code_sha)):
        raise GateReplyError(f'gate code_sha is not a 40-hex sha: {code_sha!r}')
    if not (isinstance(fingerprint, str) and _SHA256_HEX.fullmatch(fingerprint)):
        raise GateReplyError(f'gate wire_fingerprint is not a sha256 hex digest: {fingerprint!r}')
    return GateIdentity(code_sha=code_sha, wire_fingerprint=fingerprint)


def _single_json_document(stdout: str) -> object:
    try:
        return json.loads(stdout)
    except ValueError as exc:
        raise GateReplyError(f'gate stdout is not one JSON document: {stdout[:200]!r}') from exc


def parse_gate_identity(stdout: str) -> GateIdentity:
    """Parse ``orchestrator gate-identity``'s stdout: exactly one identity object."""
    return _identity_from_mapping(_single_json_document(stdout))


def parse_ensure_report(stdout: str) -> EnsureReport:
    """Parse ensure-gate.sh's stdout: ``{"built": bool, "identity": {...}}``."""
    reply = _single_json_document(stdout)
    if not isinstance(reply, dict) or set(reply) != {'built', 'identity'}:
        raise GateReplyError(f'ensure reply must be {{built, identity}}, got {stdout[:200]!r}')
    if not isinstance(reply['built'], bool):
        raise GateReplyError(f'ensure reply built must be a bool, got {reply["built"]!r}')
    return EnsureReport(built=reply['built'], identity=_identity_from_mapping(reply['identity']))


def classify_remote_exit(
    rc: int | None, stderr: str = '', *, channel: RemoteChannel = 'ssh',
) -> RunnerUnavailableKind:
    """The one rule classing a failed remote command (design D5, Q1 ruling).

    ``rc is None`` means the local spawn itself failed (``OSError``): nothing
    reached the host, so 'transport'.  On the ``'ssh'`` channel rc 255 is the
    ssh client's own failure, 'transport'; any other non-zero rc means the host
    was reached and the command failed there (1, 2, 127, a crashed verify, a
    reaped gate dir), 'gate'.  On the ``'git_push'`` channel git exits 128 for
    every failure, so the cause is read from stderr: an ssh connect failure, or
    the connection lost mid-push, is 'transport' (a sleeping laptop
    self-clears through the ssh-true reprobe);
    any other refusal (ref rejected, repo not found, an ``updateInstead``
    rejection) is 'gate', so a deterministic refusal cannot loop through that
    reprobe.

    A successful command has no class; ``rc == 0`` raises ``ValueError``.
    """
    if rc == 0:
        raise ValueError('classify_remote_exit called for a successful command (rc 0)')
    if rc is None or rc == SSH_TRANSPORT_FAILURE_RC:
        return 'transport'
    if channel == 'git_push' and _SSH_CONNECT_FAILURE.search(stderr):
        return 'transport'
    return 'gate'


def _packaged_script(name: str) -> str:
    return resources.files('orchestrator').joinpath('remote_gate', name).read_text(encoding='utf-8')


def ensure_script_text() -> str:
    """The text of ``remote_gate/ensure-gate.sh``, shipped as package data.

    Its contract: env ``GATE_ROOT`` (absolute), ``SHA`` (40 hex) and
    ``SOURCE`` (the remote repo holding ``refs/gate/<SHA>``), with the remote
    tool PATH already set; ``orchestrator`` on PATH is the host shim, which
    honours ``ORCH_GATE_BIN``.  On success it prints one
    :func:`parse_ensure_report` line; on failure it exits non-zero, never 255,
    with the cause on stderr.
    """
    return _packaged_script('ensure-gate.sh')


def reap_script_text() -> str:
    """The text of ``remote_gate/reap-gates.sh``, shipped as package data.

    Its contract: env ``GATE_ROOT`` (absolute).  Under a root carrying
    :data:`GATE_ROOT_MARKER` it deletes sha dirs idle for 7 days and trash
    older than a day; elsewhere it does nothing.  Run detached: it prints
    nothing and its outcome is not awaited.
    """
    return _packaged_script('reap-gates.sh')


# The client-side tell of a host shim that ignores ORCH_GATE_BIN and runs a
# shared tree too old to know the handshake (click's rc 2 reply).
_GATE_IDENTITY_UNKNOWN = "No such command 'gate-identity'"


def shim_suspected(stderr: str, pair: GateIdentityPair | None = None) -> bool:
    """Whether a failed ensure points at the host shim rather than the gate.

    True when the remote answered with another commit's identity, or did not
    know ``gate-identity`` at all: both mean the shim ran a shared tree
    instead of ``$ORCH_GATE_BIN`` (design D3).
    """
    if pair is not None and pair.remote.code_sha != pair.local.code_sha:
        return True
    return _GATE_IDENTITY_UNKNOWN in stderr


SHIM_REMEDY = (
    'the orchestrator shim on the verify host does not honour ORCH_GATE_BIN; '
    'run scripts/deploy/install_laptop_gate_shim.sh'
)
