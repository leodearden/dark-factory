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

import startup_completion_fixtures as _scf  # noqa: E402  (isort: after src bootstrap)
from _oauth_accounts import (  # noqa: E402  (isort: after tests-dir bootstrap)
    ALL_TOKEN_LETTERS,
    first_available_token,
)

from shared.cli_invoke import (  # noqa: E402
    _resolve_transcript_path,
    count_transcript_turns,
    read_transcript_records,
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

#: Config-dir subtrees collapsed to a single ``pruned_descendants`` count instead
#: of one entry per file.  ``plugins/marketplaces`` is a full git CLONE (hundreds
#: of loose objects) that the CLI populates on first run and that has nothing to
#: do with startup completion; capturing it verbatim inflated the raw capture to
#: 630 KB of noise.  ``projects/`` — where the transcript lives — is deliberately
#: NEVER pruned.
DEFAULT_PRUNE_PREFIXES: tuple[str, ...] = ('plugins/marketplaces', 'backups')

#: Max scrubbed stderr characters retained per observation (provenance only).
_STDERR_TAIL_CHARS = 600

#: "Never sampled" sentinel, distinct from a real ``None`` observation.
_UNSET = object()

# ---------------------------------------------------------------------------
# Redaction (capture-time gate)
# ---------------------------------------------------------------------------

#: Transcript-record fields the probe is allowed to keep.  Everything else —
#: crucially every prompt/response/tool-payload text field — is dropped.  Keep
#: this an ALLOW-list: a deny-list silently leaks whatever the next CLI version
#: adds.
_RECORD_TYPE_KEYS = ('type', 'subtype', 'operation', 'isMeta', 'isSidechain')

#: The credential pattern set and the ``.credentials.json`` filename set are
#: owned by ``startup_completion_fixtures`` (the committed assertion form) and
#: imported here so the CAPTURE-time gate and the COMMIT-time assertion can
#: never drift apart.  Widening one automatically widens the other.
_CREDENTIAL_PATTERNS = _scf._CREDENTIAL_PATTERNS
_NAMED_CREDENTIAL_PATTERNS = _scf.NAMED_CREDENTIAL_PATTERNS
_GENERIC_CREDENTIAL_PATTERNS = _scf.GENERIC_CREDENTIAL_PATTERNS
CREDENTIAL_FILENAMES = _scf.CREDENTIAL_FILENAMES


def scan_for_credential_material(
    text: str, patterns: tuple[tuple[str, str], ...] = _CREDENTIAL_PATTERNS
) -> tuple[str, int] | None:
    """Return ``(pattern_name, offset)`` of the first credential-shaped match, else None.

    The non-raising form of ``startup_completion_fixtures.assert_no_credential_material``
    over the same pattern set, for call sites that need to substitute rather than
    fail (see :func:`scrub_credential_material`).  *patterns* narrows the scan to
    one pattern class; it defaults to all of them.
    """
    for name, pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return (name, match.start())
    return None


def _scrub_text(text: str, patterns: tuple[tuple[str, str], ...]) -> str:
    """Substitute every *patterns* run in *text* with ``<redacted>``.

    The single substitution definition shared by :func:`_scrub_value`'s string
    LEAF branch and its string KEY branch, so the two can never drift into
    scrubbing the same material differently.
    """
    out = text
    for _name, pattern in patterns:
        out = re.sub(pattern + r'\S*', '<redacted>', out)
    return out


def _scrub_value(value: Any, patterns: tuple[tuple[str, str], ...]) -> Any:
    """Return *value* with every string leaf AND every string dict key scrubbed.

    Keys are rewritten with the same substitution as leaves (non-string keys are
    left untouched).  That is not symmetry for its own sake: ``_gate`` scans the
    JSON encoding of the WHOLE observation, in which a key is just as visible as
    a value, so a generic-pattern hit living in a key would otherwise survive
    this scrub into ``_gate``'s follow-up ``assert_no_credential_material`` and
    turn its documented never-raise branch into a raise — losing the entire
    already-paid-for capture rather than one matched run.
    """
    if isinstance(value, str):
        return _scrub_text(value, patterns)
    if isinstance(value, dict):
        return {
            (_scrub_text(key, patterns) if isinstance(key, str) else key): _scrub_value(
                item, patterns
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_value(item, patterns) for item in value]
    return value


def _gate(observation: dict[str, Any]) -> dict[str, Any]:
    """Refuse — or scrub — an assembled observation carrying credential material.

    The capture-time half of the two-sided guard: unredacted material never
    reaches disk, so a probe run cannot produce a raw capture that the committed
    ``TestCorpusSecretHygiene`` assertion would later have to catch.

    The two pattern classes are handled DIFFERENTLY, because their failure modes
    are different:

    - a NAMED hit (``sk-ant-``, ``accessToken``, ...) is unambiguously a secret,
      so it raises and the run is abandoned;
    - the GENERIC long-run backstop is a heuristic that can fire on a long
      non-path identifier, so it substitutes and WARNS instead.  ``_gate`` is
      called from the sampling loop, and raising there propagates out through
      ``finally`` and destroys the entire live capture — including the
      real-money ``healthy`` / ``mcp_wedge`` runs — for a harness whose whole
      purpose is to be re-run after a CLI bump.  Either way the matched run
      never reaches disk; only the blast radius differs.
    """
    encoded = json.dumps(observation)

    named = scan_for_credential_material(encoded, _NAMED_CREDENTIAL_PATTERNS)
    if named is not None:
        name, offset = named
        raise AssertionError(
            f'credential material in startup_completion_probe:observation: '
            f'pattern {name!r} matched at offset {offset} (match text withheld). '
            f'Redact it — record credential-bearing paths by presence/size only.'
        )

    generic = scan_for_credential_material(encoded, _GENERIC_CREDENTIAL_PATTERNS)
    if generic is None:
        return observation

    name, offset = generic
    print(
        f'startup_completion_probe: WARNING — heuristic pattern {name!r} matched at '
        f'offset {offset} in sample {observation.get("sample_index")} '
        f'({observation.get("sample_kind")}). Scrubbing the matching run rather than '
        f'discarding the capture. Check the emitted observation: if this was a long '
        f'identifier rather than a secret, widen the lookarounds in '
        f'startup_completion_fixtures.GENERIC_CREDENTIAL_PATTERNS.',
        file=sys.stderr,
    )
    scrubbed = _scrub_value(observation, _GENERIC_CREDENTIAL_PATTERNS)
    _scf.assert_no_credential_material(
        json.dumps(scrubbed), source='startup_completion_probe:observation(scrubbed)'
    )
    return scrubbed


def scrub_credential_material(text: str) -> str:
    """Return *text* with every credential-shaped run replaced by ``<redacted>``.

    The substituting counterpart to :func:`scan_for_credential_material`, used on
    free-text provenance (an stderr tail) where raising would throw away a real,
    already-paid-for capture.  Structured observation fields use the raising gate.
    """
    out = text
    for _name, pattern in _CREDENTIAL_PATTERNS:
        out = re.sub(pattern + r'\S*', '<redacted>', out)
    return out


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


def snapshot_config_dir(
    config_dir: Path,
    *,
    epoch: float | None = None,
    prune_prefixes: tuple[str, ...] = DEFAULT_PRUNE_PREFIXES,
) -> list[dict]:
    """Sample *config_dir*, pruning the noisy subtrees by default.

    A thin default-supplying wrapper over
    ``startup_completion_fixtures.snapshot_config_dir`` — the SINGLE sampler
    definition.  Probe output and materialized fixture trees are therefore
    described by exactly one function, so a curated row can never encode a tree
    shape that the sampler would not have produced.  The only difference is the
    default: a live config dir needs the marketplace/backups prune, a
    materialized fixture tree must round-trip verbatim.
    """
    return _scf.snapshot_config_dir(
        config_dir, epoch=epoch, prune_prefixes=prune_prefixes
    )


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


def sample_substrate(
    transcript_path: Path | None, records: list[dict] | None
) -> dict:
    """Project the already-committed ``shared.cli_invoke`` reader returns.

    This is the whole point of the probe: the predicate 3326 ports into production
    must be expressible over substrate that exists on main TODAY.  Recording these
    three returns per sample proves the discrimination without new production code.

    Takes what :func:`observe` has ALREADY read — the resolved path and the parsed
    records — rather than re-globbing and re-reading the transcript three more
    times.  Correctness, not just cost: `transcript_exists` IS
    ``_resolve_transcript_path(...) is not None`` and `count_transcript_turns` IS
    the count of ``type == "assistant"`` records, by their one-line definitions
    in `shared.cli_invoke`, so re-calling them adds no information — but it does
    add three more sample points, and a transcript that lands BETWEEN them yields
    an internally inconsistent observation (``transcript_relpath: null`` beside
    ``transcript_exists: true``).  One read, one instant, one consistent row.

    The equivalence is pinned in the other direction by the committed
    ``TestPredicateDiscrimination::test_committed_substrate_returns_match_the_row``,
    which calls the real committed functions against every materialized row.
    """
    return {
        'transcript_exists': transcript_path is not None,
        'read_transcript_records_is_none': records is None,
        'record_count': None if records is None else len(records),
        'count_transcript_turns': (
            None
            if records is None
            else sum(1 for record in records if record.get('type') == 'assistant')
        ),
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
        'substrate_returns': sample_substrate(transcript_path, records),
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

    ``ALL_TOKEN_LETTERS``, not ``_oauth_accounts``' fleet default: this probe
    spends no fleet capacity and only needs SOME account so a machine with none
    degrades to a legible skip, so the interactive/primary account ``A`` counts.

    ``_oauth_accounts`` is the single home for this scan (task 3700) — see its
    module docstring; the letter set is not restated here.
    """
    return first_available_token(os.environ, ALL_TOKEN_LETTERS)


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


def _drain_exit(
    proc: subprocess.Popen,
    *,
    mode: str,
    stdout_path: Path,
    stderr_path: Path,
) -> dict:
    """Kill (if still running), reap, and return exit provenance from the capture files.

    Reads the child's output from the temp files it was spawned against, NOT from
    pipes.  That is load-bearing rather than incidental: a ``PIPE`` the parent
    does not read fills at ~64 KB and blocks the child on write, and the parent
    here is busy sampling for up to ``--max-secs`` before it could drain.  A
    chatty startup — ``mcp_wedge``, the shape most likely to log connection
    failures, and the run finding F3 rests on — would then wedge *because the
    probe measured it*.  Files cannot backpressure, so the child never blocks and
    the capture is drained once, after the sampling window closes.

    The stderr tail is SCRUBBED (patterns substituted, not raised on) and
    truncated — it is provenance for the report, e.g. whether the CLI logged an
    MCP connection failure while the stub server hung.
    """
    still_running = proc.poll() is None
    if still_running:
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    stdout = stdout_path.read_bytes() if stdout_path.exists() else b''
    stderr = stderr_path.read_bytes() if stderr_path.exists() else b''
    text = stderr.decode('utf-8', 'replace')
    # Scalar-only projection of the CLI's --output-format json envelope: enough
    # to explain a non-zero exit (e.g. `error_max_turns` vs a startup failure)
    # without capturing the model's `result` text.
    envelope: dict[str, Any] = {}
    with contextlib.suppress(ValueError, AttributeError):
        parsed = json.loads(stdout.decode('utf-8', 'replace'))
        if isinstance(parsed, dict):
            envelope = {
                k: parsed[k]
                for k in ('subtype', 'is_error', 'num_turns', 'duration_ms', 'duration_api_ms')
                if k in parsed and isinstance(parsed[k], (str, bool, int, float))
            }
    return {
        'mode': mode,
        'killed_by_probe': still_running,
        'exit_code': proc.returncode,
        'stdout_envelope': envelope,
        'stderr_len': len(text),
        'stderr_tail': scrub_credential_material(text[-_STDERR_TAIL_CHARS:]),
    }


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

    # Capture files, NOT pipes — see _drain_exit: an undrained PIPE blocks the
    # child once ~64 KB of startup chatter fills it, and this parent does not
    # drain until the sampling window closes.
    stdout_path = stub_dir / 'probe-stdout.bin'
    stderr_path = stub_dir / 'probe-stderr.bin'
    stdout_fh = stdout_path.open('wb')
    stderr_fh = stderr_path.open('wb')

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
            stdout=stdout_fh,
            stderr=stderr_fh,
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

        def _transcript_state_key() -> tuple[str, int, int] | None:
            """Cheap change-detector for the watched transcript: one glob + one stat.

            ``_take`` costs a full ``rglob('*')`` of the config dir (which walks
            the several-hundred-file ``plugins/marketplaces`` clone before
            pruning it), plus a transcript read and a JSON credential scan.  At
            the 0.2 s tick that was ~5 tree walks a second, every result but the
            last discarded — and the probe's own FS churn competing, on the same
            filesystem, with the CLI startup whose timing it is measuring.  So
            re-take the candidate only when this key moves.
            """
            path = _resolve_transcript_path(config_dir, session_id)
            if path is None:
                return None
            try:
                stat = path.stat()
            except OSError:
                return None
            return (str(path), stat.st_size, stat.st_mtime_ns)

        # _UNSET, not None: None is the real "no transcript yet" key, and the
        # first tick must always produce a candidate so a run that never reaches
        # session init still carries one.
        candidate_key: Any = _UNSET

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
                    # Re-taken only when the transcript actually changed, so an
                    # unchanging one is not re-sampled ~5x/second for a result
                    # that is discarded.  Nothing is lost: an identical key means
                    # an identical observation, and the late state of a run that
                    # never starts is still captured by the `deadline` /
                    # `after_exit` samples below.
                    #
                    # The turn can land BETWEEN the check above and this sample, so
                    # re-read the candidate's OWN recorded turn count and discard it
                    # if it already shows turn 1 — otherwise the row would be
                    # labelled `pre_first_token` while carrying an assistant record,
                    # which is exactly the mislabel that would let a curated corpus
                    # row lie about the regime it came from.
                    key = _transcript_state_key()
                    if key != candidate_key:
                        candidate_key = key
                        candidate = _take('pre_first_token_candidate')
                        observed = candidate['substrate_returns']['count_transcript_turns']
                        if observed is None or observed < 1:
                            pre_first_token = candidate
            if proc.poll() is not None:
                observations.append(_take('after_exit'))
                break
            if elapsed >= max_secs:
                observations.append(_take('deadline'))
                break
            time.sleep(_FINE_TICK_SECS)

        # Flush the buffered pre-turn-1 sample BEFORE stamping, so it carries the
        # same run_exit provenance as every other observation of this run.
        if not seen_turn and pre_first_token is not None:
            pre_first_token['sample_kind'] = 'pre_first_token'
            observations.append(pre_first_token)

        # Reap the child, read its capture files, and stamp exit provenance onto
        # every observation of this run.
        exit_provenance = _gate(
            _drain_exit(proc, mode=mode, stdout_path=stdout_path, stderr_path=stderr_path)
        )
        for observation in observations:
            observation['run_exit'] = exit_provenance
    finally:
        stdout_fh.close()
        stderr_fh.close()
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
