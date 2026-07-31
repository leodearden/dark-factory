#!/usr/bin/env python3
"""Startup-completion artifact probe — task 3324 (substrate validation for the
two-regime watchdog startup grace, PRD `plans/server-side-api-error-handling-prd.md`,
consumer task 3326 / contract C5).

WHAT THIS ANSWERS
-----------------
The watchdog's startup regime currently kills an invocation that has produced no
assistant turn by ``startup_grace_secs`` (120s).  C5 wants a SECOND, longer grace
that applies only once the CLI has demonstrably *finished starting up* and is
merely waiting on the server (e.g. a 529 retry cycle).  That needs a predicate
answering: **"has the CLI completed startup, even though turn 1 has not landed?"**

This probe measures which on-disk artifacts actually exist, at which offsets, for
a healthy invocation versus each PRD-named wedge shape — so the predicate is
chosen from observed evidence rather than guessed.

MODES
-----
``healthy``       spawn the real ``claude --print`` (haiku, one-word prompt,
                  ~$0.002) through a production-shaped ``TaskConfigDir``.
``build_wedge``   spawn a stub wrapper that emits from-source-build stderr and
                  never execs the CLI (the "wrapper still compiling" wedge).
``uv_wedge``      spawn a stub wrapper that emits ``uv`` resolution stderr and
                  never execs the CLI.
``mcp_wedge``     spawn the real ``claude`` with ``--mcp-config`` pointing at a
                  stub stdio server that accepts the connection and then never
                  answers ``initialize``.
``replay``        read-only: run the SAME sampler against an existing on-disk
                  ``CLAUDE_CONFIG_DIR`` (the pre-2 fallback when a live spawn is
                  not possible in a given dispatch).

OUTPUT
------
One redacted JSON observation object per sample, JSONL, to stdout (or ``--out``).
Redaction happens at CAPTURE time, not at curation time: file *contents* are
never inlined (only path/kind/size metadata), and transcript records are reduced
to a fixed safe field projection that excludes all prompt/response text.  The
healthy observation is taken from a config dir that really does hold a live OAuth
token in ``.credentials.json``, so this is load-bearing, not hygiene theatre.

USAGE
-----
    uv run --project shared python tests/startup_completion_probe.py \
        --mode healthy --out /tmp/healthy.jsonl

See `docs/startup-completion-artifact-matrix.md` for the resulting matrix and the
chosen predicate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow execution as a bare script (``python tests/startup_completion_probe.py``)
# as well as import from a pytest run, mirroring shared/tests/conftest.py.
_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / 'src'
for _p in (str(_TESTS_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.cli_invoke import (  # noqa: E402
    _resolve_transcript_path,
    count_transcript_turns,
    read_transcript_records,
    transcript_exists,
)
from shared.config_dir import TaskConfigDir  # noqa: E402

MODES = ('healthy', 'build_wedge', 'uv_wedge', 'mcp_wedge', 'replay')

#: Wedge-shape slug recorded on each observation, keyed by probe mode.  ``None``
#: for the healthy/replay regimes.  These slugs are the PRD's names and are the
#: same closed set the corpus rows use.
MODE_WEDGE_SHAPE: dict[str, str | None] = {
    'healthy': None,
    'build_wedge': 'from_source_build',
    'uv_wedge': 'uv_resolving',
    'mcp_wedge': 'mcp_init_hang',
    'replay': None,
}

#: Full-sample offsets (seconds since spawn).  Recorded as PROVENANCE only — no
#: test asserts a wall-clock threshold, because none is achievable (host load,
#: SessionStart hook duration, MCP server count and FS cache all move these).
DEFAULT_SAMPLE_OFFSETS: tuple[float, ...] = (0.25, 1.0, 2.0, 5.0, 15.0, 30.0)

#: Fine polling grid used to catch the pre-first-token boundary sample.
_FINE_TICK_SECS = 0.2

# ---------------------------------------------------------------------------
# Redaction (capture-time gate)
# ---------------------------------------------------------------------------

#: Transcript-record fields the probe is allowed to keep.  Everything else —
#: crucially every prompt/response/tool-payload text field — is dropped.  Keep
#: this an ALLOW-list: a deny-list silently leaks whatever the next CLI version
#: adds.
_RECORD_TYPE_KEYS = ('type', 'subtype', 'operation', 'isMeta', 'isSidechain')

_CREDENTIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ('sk-ant-token', r'sk-ant-'),
    ('oauth-blob', r'claudeAiOauth'),
    ('access-token', r'accessToken'),
    ('refresh-token', r'refreshToken'),
    ('bearer-jwt', r'Bearer\s+eyJ'),
)

#: Filenames whose CONTENT must never be captured, only presence/size metadata.
CREDENTIAL_FILENAMES = frozenset({'.credentials.json'})


def scan_for_credential_material(text: str) -> tuple[str, int] | None:
    """Return ``(pattern_name, offset)`` of the first credential-shaped match, else None.

    Deliberately duplicated in ``startup_completion_fixtures`` as the committed
    assertion form; this copy is the capture-time gate so unredacted material
    never reaches disk in the first place.
    """
    for name, pattern in _CREDENTIAL_PATTERNS:
        match = re.search(pattern, text)
        if match is not None:
            return (name, match.start())
    return None


def _gate(observation: dict[str, Any]) -> dict[str, Any]:
    """Raise if a fully-assembled observation carries credential material."""
    hit = scan_for_credential_material(json.dumps(observation))
    if hit is not None:
        raise RuntimeError(
            f'startup_completion_probe: refusing to emit observation — credential '
            f'pattern {hit[0]!r} matched at offset {hit[1]}'
        )
    return observation


def _redact_argv(argv: list[str]) -> list[str]:
    """Drop any argv element that looks credential-shaped."""
    out: list[str] = []
    for element in argv:
        out.append('<redacted>' if scan_for_credential_material(element) else element)
    return out


def redact_record(record: dict) -> dict:
    """Project a transcript record down to the safe, predicate-relevant fields.

    Keeps the record ``type`` (what ``count_transcript_turns`` and the chosen
    predicate read), the ``queue-operation`` ``operation`` discriminator, the
    ``attachment`` kind/hook name, and ``message.role`` — never any content.
    """
    out: dict[str, Any] = {}
    for key in _RECORD_TYPE_KEYS:
        if key in record:
            out[key] = record[key]
    attachment = record.get('attachment')
    if isinstance(attachment, dict):
        out['attachment'] = {
            k: attachment[k] for k in ('type', 'hookName') if k in attachment
        }
    message = record.get('message')
    if isinstance(message, dict) and 'role' in message:
        out['message'] = {'role': message['role']}
    return out


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def snapshot_config_dir(config_dir: Path, *, epoch: float | None = None) -> list[dict]:
    """Return a sorted, content-free description of every entry under *config_dir*.

    Each entry is ``{relpath, kind, size, mtime_delta_secs}`` where ``kind`` is
    one of ``file`` / ``dir`` / ``symlink``.  Contents are NEVER inlined —
    ``.credentials.json`` in particular is recorded by presence and size only.

    ``mtime_delta_secs`` is the entry's mtime minus *epoch* (the spawn instant),
    rounded, or ``None`` when no epoch is supplied.  It is provenance data, never
    an asserted bound.
    """
    entries: list[dict] = []
    if not config_dir.exists():
        return entries
    for path in sorted(config_dir.rglob('*')):
        try:
            relpath = str(path.relative_to(config_dir))
            if path.is_symlink():
                kind = 'symlink'
                size = None
            elif path.is_dir():
                kind = 'dir'
                size = None
            else:
                kind = 'file'
                size = path.lstat().st_size
            mtime_delta = None
            if epoch is not None:
                mtime_delta = round(path.lstat().st_mtime - epoch, 3)
            entries.append(
                {
                    'relpath': relpath,
                    'kind': kind,
                    'size': size,
                    'mtime_delta_secs': mtime_delta,
                }
            )
        except OSError:
            # A dir that vanished mid-walk (the CLI rotates temp state) is a real
            # observation, not an error — record it as such rather than crashing
            # a capture that cannot be cheaply repeated.
            entries.append(
                {
                    'relpath': str(path),
                    'kind': 'vanished',
                    'size': None,
                    'mtime_delta_secs': None,
                }
            )
    return entries


def sample_proc(pid: int | None) -> dict:
    """Sample ``/proc/<pid>`` — liveness, scheduler state char, argv, direct children.

    Total: every field degrades to ``None``/``[]`` rather than raising, because a
    probe that crashes on a racing exit loses the whole (paid-for) capture.
    """
    out: dict[str, Any] = {
        'pid': pid,
        'alive': False,
        'state': None,
        'comm': None,
        'cmdline': None,
        'children': [],
    }
    if pid is None:
        return out
    proc = Path(f'/proc/{pid}')
    if not proc.exists():
        return out
    out['alive'] = True
    try:
        stat = (proc / 'stat').read_text()
        # comm may contain spaces/parens — split on the LAST ') ' per proc(5).
        out['state'] = stat.rsplit(') ', 1)[1].split()[0]
    except (OSError, IndexError):
        pass
    with contextlib.suppress(OSError):
        out['comm'] = (proc / 'comm').read_text().strip()
    try:
        raw = (proc / 'cmdline').read_bytes().decode('utf-8', 'replace')
        argv = [part for part in raw.split('\0') if part]
        out['cmdline'] = _redact_argv(argv)
    except OSError:
        pass
    try:
        child_pids = (proc / 'task' / str(pid) / 'children').read_text().split()
        for child in child_pids:
            try:
                comm = Path(f'/proc/{child}/comm').read_text().strip()
            except OSError:
                comm = None
            out['children'].append({'pid': int(child), 'comm': comm})
    except (OSError, ValueError):
        pass
    return out


def sample_substrate(config_dir: Path, session_id: str) -> dict:
    """Evaluate the already-committed ``shared.cli_invoke`` transcript readers.

    This is the whole point of the probe: the predicate 3326 ports into production
    must be expressible over substrate that exists on main TODAY.  Recording these
    three returns per sample proves the discrimination without new production code.
    """
    records = read_transcript_records(config_dir, session_id)
    return {
        'transcript_exists': transcript_exists(config_dir, session_id),
        'read_transcript_records_is_none': records is None,
        'record_count': None if records is None else len(records),
        'count_transcript_turns': count_transcript_turns(config_dir, session_id),
    }


def observe(
    *,
    config_dir: Path,
    session_id: str,
    probe_run_id: str,
    mode: str,
    sample_index: int,
    sample_kind: str,
    sample_offset_secs: float,
    cli_version: str,
    capture_method: str,
    pid: int | None,
    epoch: float | None,
    extra: dict | None = None,
) -> dict:
    """Assemble ONE redacted observation object and gate it before returning."""
    transcript_path = _resolve_transcript_path(config_dir, session_id)
    records = read_transcript_records(config_dir, session_id)
    observation: dict[str, Any] = {
        'probe_run_id': probe_run_id,
        'mode': mode,
        'wedge_shape': MODE_WEDGE_SHAPE.get(mode),
        'sample_index': sample_index,
        'sample_kind': sample_kind,
        'sample_offset_secs': round(sample_offset_secs, 3),
        'session_id': session_id,
        'cli_version': cli_version,
        'capture_method': capture_method,
        'captured_at': datetime.now(UTC).isoformat(),
        'config_dir_tree': snapshot_config_dir(config_dir, epoch=epoch),
        'transcript_relpath': (
            str(transcript_path.relative_to(config_dir)) if transcript_path else None
        ),
        'transcript_records': (
            None if records is None else [redact_record(r) for r in records]
        ),
        'substrate_returns': sample_substrate(config_dir, session_id),
        'proc': sample_proc(pid),
    }
    if extra:
        observation.update(extra)
    return _gate(observation)


# ---------------------------------------------------------------------------
# Wedge stubs
# ---------------------------------------------------------------------------

_BUILD_WEDGE_STUB = """#!/bin/sh
# Stub standing in for a wrapper that is building the CLI from source and has
# not yet exec'd it.  Deliberately never touches CLAUDE_CONFIG_DIR.
echo 'Building claude-code from source (this may take a while)...' >&2
echo '   Compiling cli v2.1.220 (/home/build/claude-code)' >&2
sleep %(hold)d
"""

_UV_WEDGE_STUB = """#!/bin/sh
# Stub standing in for `uv` resolving/downloading the environment before the CLI
# is ever launched.  Deliberately never touches CLAUDE_CONFIG_DIR.
echo 'Resolved 214 packages in 1.24s' >&2
echo 'Downloading numpy (18.2MiB)' >&2
sleep %(hold)d
"""

_MCP_HANG_SERVER = '''#!/usr/bin/env python3
"""Stub stdio MCP server: accepts the connection, reads whatever the client
sends, and NEVER writes a response — so the client hangs at `initialize`."""
import sys
import time

while True:
    line = sys.stdin.readline()
    if not line:
        break
time.sleep(3600)
'''


def _write_stub(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body)
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# Spawn shapes
# ---------------------------------------------------------------------------


def _cli_version() -> str:
    try:
        proc = subprocess.run(
            ['claude', '--version'], capture_output=True, text=True, timeout=30
        )
        return proc.stdout.strip() or proc.stderr.strip() or 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def _oauth_token() -> tuple[str, str] | None:
    """Return ``(env_var_name, token)`` for the first available OAuth account.

    Mirrors ``shared/tests/test_cli_invoke_integration.py``'s ``_AVAILABLE_TOKENS``
    discovery so a machine with no accounts degrades to a legible skip.
    """
    for var in [f'CLAUDE_OAUTH_TOKEN_{c}' for c in 'ABCDEFG']:
        token = os.environ.get(var)
        if token:
            return (var, token)
    return None


def _build_argv(
    mode: str,
    *,
    session_id: str,
    prompt: str,
    model: str,
    permission_mode: str,
    stub_dir: Path,
    hold_secs: int,
) -> tuple[list[str], list[Path]]:
    """Assemble the spawn argv for *mode*, mirroring ``build_claude_argv``'s shape.

    Returns ``(argv, temp_paths)``; the caller owns unlinking ``temp_paths``.
    """
    temp_paths: list[Path] = []
    if mode == 'build_wedge':
        return ([str(_write_stub(stub_dir, 'claude-build-wrapper.sh',
                                 _BUILD_WEDGE_STUB % {'hold': hold_secs}))], temp_paths)
    if mode == 'uv_wedge':
        return ([str(_write_stub(stub_dir, 'claude-uv-wrapper.sh',
                                 _UV_WEDGE_STUB % {'hold': hold_secs}))], temp_paths)

    fd, sysprompt_path = tempfile.mkstemp(suffix='.txt', prefix='startup_probe_sysprompt_')
    temp_paths.append(Path(sysprompt_path))
    with open(fd, 'w') as fh:
        fh.write('You are a probe target. Answer in one word.')

    argv = [
        'claude',
        '--print',
        '--output-format',
        'json',
        '--model',
        model,
        '--system-prompt-file',
        sysprompt_path,
        '--session-id',
        session_id,
        '--permission-mode',
        permission_mode,
        '--max-turns',
        '1',
        '--disallowed-tools',
        '*',
    ]

    if mode == 'mcp_wedge':
        server = _write_stub(stub_dir, 'mcp_hang_server.py', _MCP_HANG_SERVER)
        mcp_config = {
            'mcpServers': {
                'hang': {'type': 'stdio', 'command': sys.executable, 'args': [str(server)]}
            }
        }
        fd, mcp_path = tempfile.mkstemp(suffix='.json', prefix='startup_probe_mcp_')
        temp_paths.append(Path(mcp_path))
        with open(fd, 'w') as fh:
            json.dump(mcp_config, fh)
        # --strict-mcp-config scopes the run to ONLY the hanging server, so the
        # ambient project .mcp.json cannot muddy which server the CLI waits on.
        argv.extend(['--mcp-config', mcp_path, '--strict-mcp-config'])

    argv.extend(['-p', prompt])
    return (argv, temp_paths)


def _spawn_env(config_dir: Path, oauth_token: str | None) -> dict[str, str]:
    """Build the subprocess env, mirroring ``cli_invoke._invoke_claude``."""
    env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
    if oauth_token:
        env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token
    env['CLAUDE_CONFIG_DIR'] = str(config_dir)
    return env


# ---------------------------------------------------------------------------
# Probe drivers
# ---------------------------------------------------------------------------


def run_live_probe(
    *,
    mode: str,
    probe_run_id: str,
    cwd: Path,
    prompt: str,
    model: str,
    permission_mode: str,
    offsets: tuple[float, ...],
    max_secs: float,
    hold_secs: int,
    keep_config_dir: bool,
) -> list[dict]:
    """Spawn *mode*'s target and emit one observation per sample."""
    session_id = str(uuid.uuid4())
    cli_version = _cli_version()
    token_pair = _oauth_token()
    config = TaskConfigDir(f'startup-probe-{mode}-{os.getpid()}')
    config_dir = config.path
    if token_pair is not None:
        config.write_credentials(token_pair[1])

    stub_dir = Path(tempfile.mkdtemp(prefix='startup_probe_stubs_'))
    argv, temp_paths = _build_argv(
        mode,
        session_id=session_id,
        prompt=prompt,
        model=model,
        permission_mode=permission_mode,
        stub_dir=stub_dir,
        hold_secs=hold_secs,
    )
    env = _spawn_env(config_dir, token_pair[1] if token_pair else None)

    observations: list[dict] = []
    proc: subprocess.Popen | None = None
    epoch = time.time()
    start = time.monotonic()
    try:
        # start_new_session=True mirrors cli_invoke._run_subprocess's spawn shape,
        # so the observed process-group / children topology is production's.
        proc = subprocess.Popen(  # noqa: S603
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        pending = list(offsets)
        sample_index = 0
        pre_first_token: dict | None = None
        seen_turn = False

        def _take(kind: str) -> dict:
            nonlocal sample_index
            observation = observe(
                config_dir=config_dir,
                session_id=session_id,
                probe_run_id=probe_run_id,
                mode=mode,
                sample_index=sample_index,
                sample_kind=kind,
                sample_offset_secs=time.monotonic() - start,
                cli_version=cli_version,
                capture_method='live_spawn',
                pid=proc.pid if proc else None,
                epoch=epoch,
                extra={'spawn_argv': _redact_argv(argv), 'oauth_env_var': (
                    token_pair[0] if token_pair else None
                )},
            )
            sample_index += 1
            return observation

        while True:
            elapsed = time.monotonic() - start
            if pending and elapsed >= pending[0]:
                pending.pop(0)
                observations.append(_take('scheduled'))
            if not seen_turn:
                turns = count_transcript_turns(config_dir, session_id)
                if turns is not None and turns >= 1:
                    seen_turn = True
                    if pre_first_token is not None:
                        pre_first_token['sample_kind'] = 'pre_first_token'
                        observations.append(pre_first_token)
                    observations.append(_take('first_token'))
                else:
                    # Keep only the most recent pre-turn-1 sample; it is the
                    # incident-shape observation the whole two-regime grace is for.
                    pre_first_token = _take('pre_first_token_candidate')
            if proc.poll() is not None:
                observations.append(_take('after_exit'))
                break
            if elapsed >= max_secs:
                observations.append(_take('deadline'))
                break
            time.sleep(_FINE_TICK_SECS)

        if not seen_turn and pre_first_token is not None:
            pre_first_token['sample_kind'] = 'pre_first_token'
            observations.append(pre_first_token)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        for path in temp_paths:
            path.unlink(missing_ok=True)
        for stub in sorted(stub_dir.glob('*')):
            stub.unlink(missing_ok=True)
        stub_dir.rmdir()
        if not keep_config_dir:
            config.cleanup()
    return observations


def run_replay_probe(
    *,
    probe_run_id: str,
    source_config_dir: Path,
    session_id: str | None,
    offsets: tuple[float, ...],
) -> list[dict]:
    """Run the sampler READ-ONLY against an existing on-disk config dir.

    The pre-2 fallback for a dispatch that cannot spawn a live CLI.  Records
    ``capture_method='replayed_from_live_config_dir'`` plus the source dir so the
    provenance of every derived corpus row stays honest about how it was taken.
    """
    if session_id is None:
        candidates = sorted(source_config_dir.glob('projects/*/*.jsonl'))
        if not candidates:
            raise SystemExit(
                f'replay: no projects/*/*.jsonl transcript under {source_config_dir}'
            )
        session_id = candidates[0].stem
    observation = observe(
        config_dir=source_config_dir,
        session_id=session_id,
        probe_run_id=probe_run_id,
        mode='replay',
        sample_index=0,
        sample_kind='replay',
        sample_offset_secs=offsets[-1] if offsets else 0.0,
        cli_version=_cli_version(),
        capture_method='replayed_from_live_config_dir',
        pid=None,
        epoch=None,
        extra={'source_config_dir': str(source_config_dir)},
    )
    return [observation]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--out', type=Path, default=None, help='JSONL output path (default stdout)')
    parser.add_argument('--probe-run-id', default=None, help='defaults to <mode>-<uuid4 prefix>')
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    parser.add_argument('--prompt', default='ok')
    parser.add_argument('--model', default='haiku')
    parser.add_argument('--permission-mode', default='bypassPermissions')
    parser.add_argument('--max-secs', type=float, default=45.0)
    parser.add_argument('--hold-secs', type=int, default=60, help='wedge stub sleep duration')
    parser.add_argument('--keep-config-dir', action='store_true')
    parser.add_argument(
        '--source-config-dir', type=Path, default=None, help='replay mode: dir to sample'
    )
    parser.add_argument('--session-id', default=None, help='replay mode: session to resolve')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probe_run_id = args.probe_run_id or f'{args.mode}-{uuid.uuid4().hex[:12]}'

    if args.mode == 'replay':
        if args.source_config_dir is None:
            raise SystemExit('--source-config-dir is required for --mode replay')
        observations = run_replay_probe(
            probe_run_id=probe_run_id,
            source_config_dir=args.source_config_dir,
            session_id=args.session_id,
            offsets=DEFAULT_SAMPLE_OFFSETS,
        )
    else:
        observations = run_live_probe(
            mode=args.mode,
            probe_run_id=probe_run_id,
            cwd=args.cwd,
            prompt=args.prompt,
            model=args.model,
            permission_mode=args.permission_mode,
            offsets=DEFAULT_SAMPLE_OFFSETS,
            max_secs=args.max_secs,
            hold_secs=args.hold_secs,
            keep_config_dir=args.keep_config_dir,
        )

    lines = '\n'.join(json.dumps(o, sort_keys=True) for o in observations)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('a', encoding='utf-8') as fh:
            fh.write(lines + '\n')
    else:
        sys.stdout.write(lines + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
