"""Tests for scripts/memory-metadata-coverage-census.sh, the wrapper the
memory-metadata-coverage-census.service ExecStart names (task 4006). Moved here
from test_install_memory_metadata_coverage_census_timer.py by task 7154, so the
2026-08-31 incident notes below describe this code under its old file name.

The wrapper is driven with BOTH `*_CMD` seams pointed at fake recorder
executables, mirroring test_flag_marker_sweep_wrapper.py, so the
census never runs against live Qdrant and 3201's retro sweep never touches the
live corpus.

WHY THE WRAPPER'S CONTRACT IS PINNED HERE. This job pairs a GAUGE (the census's
topic/canonical coverage block) with the MECHANISM that closes it (3201's
retro_stamp_topics.py), on one cadence, so an operator sees the gap and the
proposed stamping in the same journal entry. The safety property that makes
that pairing legal is that the second half runs in DRY-RUN: bulk `canonical:
true` writes stay an operator decision, never an unattended nightly one. A test
that only checked the units would leave exactly that property unpinned.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _merge_queue_landing import LandingNotConfirmed, land_files
from config_reload_script_fakes import FakeEscalationMcp
from merge_queue_fakes import FakeMergeQueue, assert_target_checkout_clean
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

TEMPLATES_DIR = Path(__file__).parent.parent
WRAPPER = TEMPLATES_DIR / 'memory-metadata-coverage-census.sh'

# Resolved in the PARENT: subprocess looks argv[0] up on the CHILD's PATH, and
# the uv rows below hand the wrapper an empty one.
BASH = shutil.which('bash') or '/bin/bash'


# ── ambient git redirection (incident 2026-08-31) ───────────────────────────
#
# Every git path in this file is pinned to a tmp dir, and that is NOT sufficient.
# `git -C <tmp>` only changes DIRECTORY: GIT_DIR and its siblings skip
# repository discovery outright, so one ambient GIT_DIR sends both this file's
# own `git config user.*` writes and the wrapper's `git -C "$REPO" commit`
# into whatever repository that variable names, with -C and $REPO alike inert.
#
# That is not hypothetical here. On 2026-08-31 a run of this file wrote
# `[user] name=Test email=test@example.com` and `commit.gpgsign = false` -- the
# literal values from _git_repo_harness below -- into the LIVE project_root
# checkout's .git/config, and committed this file's placeholder artifact
# content onto its main. GIT_CEILING_DIRECTORIES did not stop it: a ceiling
# bounds the upward WALK, and an explicit GIT_DIR never walks.
#
# GIT_CEILING_DIRECTORIES is the one GIT_* name deliberately KEPT: it is the
# suite's first-defence containment (df_pytest_isolation._df_git_ceiling_at_basetemp)
# and dropping it here would hand the subprocess LESS protection, not more.
#
# Suite-wide immunity lives in df_pytest_isolation._df_git_env_hermetic. This
# layer is deliberately redundant with it, because this file is also run
# directly (`pytest scripts/tests/test_...py`) from rootdirs whose conftest may
# not wire that fixture, and because the harness -- not the fixture -- is what
# a future editor of this file will read.
_KEEP_GIT_ENV = frozenset({'GIT_CEILING_DIRECTORIES'})


def _scrub_git_env(env):
    """Drop every ambient GIT_* that could retarget git away from the path given."""
    for key in [k for k in env if k.startswith('GIT_') and k not in _KEEP_GIT_ENV]:
        env.pop(key, None)
    return env


# ── the committed wrapper ───────────────────────────────────────────────────

_FAKE_RECORDER_SRC = '''#!/usr/bin/env python3
"""Fake command recorder. Appends {"who", "argv", "env"} to the JSON list at
$FAKE_STATE, then exits with $FAKE_EXIT_<WHO> (default 0).

`env` carries the three service-env roots the wrapper is contracted to export.
Recording them from INSIDE the invoked command is the only way to OBSERVE that
the exports actually ran: all three names also appear in the wrapper's header
comment, so a source-text grep stays green with the real `export` lines deleted.

A CENSUS call also records the checkout it measures -- the root its script
argument resolves to, as census_memory_metadata.py::_REPO_ROOT does -- with
that root's HEAD and git common dir at call time, and writes each
$FAKE_CENSUS_WRITES {repo-relative path: content} entry under that root, as
the real census writes its artifacts beside its own file.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SERVICE_ENV_KEYS = ("CONFIG_PATH", "PROJECT_ROOT", "FALKORDB_URI")


def git(root, *args):
    proc = subprocess.run([os.environ["FAKE_REAL_GIT"], "-C", str(root), *args],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


who = os.environ["FAKE_WHO"]
state_path = os.environ["FAKE_STATE"]
with open(state_path) as f:
    state = json.load(f)
entry = {
    "who": who,
    "argv": sys.argv[1:],
    "env": {k: os.environ.get(k) for k in SERVICE_ENV_KEYS},
}
if who == "CENSUS":
    root = Path(sys.argv[1]).resolve().parents[2]
    entry["root"] = str(root)
    entry["head"] = git(root, "rev-parse", "HEAD")
    entry["common_dir"] = git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    for rel, content in json.loads(os.environ.get("FAKE_CENSUS_WRITES", "{}")).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
state.append(entry)
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(int(os.environ.get(f"FAKE_EXIT_{who}", "0")))
'''

# The wrapper exports these with `${VAR:-default}` semantics, under which an
# AMBIENT value WINS over the wrapper's own root. They are unset in a plain
# login shell (verified) but the fused-memory service env sets them, so they are
# popped explicitly rather than assumed absent -- otherwise the service-env
# assertion below is ambient-dependent: green here, red under the service.
_SERVICE_ENV_KEYS = ('CONFIG_PATH', 'PROJECT_ROOT', 'FALKORDB_URI')


def _git_argv_path(tmp_path):
    """Where the recording `git` shim appends one JSON argv list per call."""
    return tmp_path / 'git_argv.jsonl'


def _wrapper_bin_dir(tmp_path):
    """Where _wrapper_harness writes its shims and the recorder they exec."""
    return tmp_path / 'wbin'


def _scratch_dir(tmp_path):
    """The TMPDIR every wrapper run gets, so a test can assert it is left empty."""
    return tmp_path / 'deploy-tmp'


def _write_recorder_shim(path, who, recorder):
    """An executable that records its argv under `who` via `recorder`.

    Both the shebang and the interpreter are ABSOLUTE, so the shim runs under
    an empty PATH and a PATH-resolved shim can never re-enter itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!{BASH}\n'
        f'FAKE_WHO={who} exec {sys.executable} {recorder} "$@"\n'
    )
    path.chmod(0o755)


def _wrapper_harness(tmp_path, *, census_exit=0, stamp_exit=0, extra_env=None,
                     default_prefix=False, record_git=False, census_writes=None):
    """Point both wrapper seams at fake recorders. Returns (env, state_path).

    `census_writes` ({repo-relative path: content}) is what the fake census
    writes under the checkout it measures: the night's drift.

    With `default_prefix=True` the two `*_CMD` seams are left UNSET instead, so
    the wrapper's own `uv run --frozen --project <FM> python` prefix is the
    thing under test, with a fake `uv` recorder on PATH. Every other wrapper
    test overrides both seams and therefore never exercises that prefix at all.
    """
    bin_dir = _wrapper_bin_dir(tmp_path)
    bin_dir.mkdir(exist_ok=True)
    state_path = tmp_path / 'wrapper_state.json'
    state_path.write_text('[]')

    real_git = shutil.which('git')
    assert real_git, 'no real git on PATH to delegate to'
    shims = {'fake-census': 'CENSUS', 'fake-stamp': 'STAMP'}
    if default_prefix:
        # ONE fake `uv` serves BOTH halves, so it cannot key on FAKE_WHO: every
        # call lands under the same `who`, and the census's is simply the first.
        shims['uv'] = 'UV'
    for name, who in shims.items():
        _write_recorder_shim(bin_dir / name, who, bin_dir / 'recorder.py')
    (bin_dir / 'recorder.py').write_text(_FAKE_RECORDER_SRC)

    if record_git:
        # A `git` shim that RECORDS the real argv and then delegates to the
        # real binary, so the wrapper's git behaviour is unchanged and the
        # landing still genuinely lands. The real git is resolved
        # ABSOLUTELY here -- while bin_dir is not yet on PATH -- so the shim
        # can never re-enter itself.
        git_log = _git_argv_path(tmp_path)
        git_log.write_text('')
        shim = bin_dir / 'git'
        record_snippet = (
            f'{sys.executable} -c '
            '\'import json,sys; open(sys.argv[1],"a").write('
            'json.dumps(sys.argv[2:])+"\\n")\' '
            f'{git_log} "$@"\n'
        )
        shim.write_text(
            '#!/usr/bin/env bash\n'
            + record_snippet +
            f'exec {real_git} "$@"\n'
        )
        shim.chmod(0o755)

    repo = tmp_path / 'fake-repo'
    (repo / 'fused-memory' / 'scripts').mkdir(parents=True, exist_ok=True)

    env = _scrub_git_env(dict(os.environ))
    env['PATH'] = f'{bin_dir}{os.pathsep}{env["PATH"]}'
    env['REPO'] = str(repo)
    for key in _SERVICE_ENV_KEYS:
        env.pop(key, None)
    # An ambient operator pin would steer the default prefix to a REAL uv.
    env.pop('UV_BIN', None)
    env.pop('CENSUS_LANDING_TIMEOUT_SECS', None)
    if default_prefix:
        for key in ('COVERAGE_CENSUS_CMD', 'RETRO_STAMP_CMD'):
            env.pop(key, None)
        # Keeps the commit half out of the prefix assertion.
        env['CENSUS_COMMIT'] = '0'
    else:
        env['COVERAGE_CENSUS_CMD'] = str(bin_dir / 'fake-census')
        env['RETRO_STAMP_CMD'] = str(bin_dir / 'fake-stamp')
    env['FAKE_STATE'] = str(state_path)
    env['FAKE_REAL_GIT'] = real_git
    env['FAKE_CENSUS_WRITES'] = json.dumps(census_writes or {})
    scratch = _scratch_dir(tmp_path)
    scratch.mkdir(exist_ok=True)
    env['TMPDIR'] = str(scratch)
    env['FAKE_EXIT_CENSUS'] = str(census_exit)
    env['FAKE_EXIT_STAMP'] = str(stamp_exit)
    if extra_env:
        env.update(extra_env)
    return env, state_path


def _run_wrapper(tmp_path, **kwargs):
    env, state_path = _wrapper_harness(tmp_path, **kwargs)
    result = subprocess.run(
        [BASH, str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    return result, json.loads(state_path.read_text())


def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f'Expected {WRAPPER} to be executable; run: chmod +x {WRAPPER}')


def test_wrapper_runs_the_census_then_the_retro_stamp_rehearsal(tmp_path):
    """One cadence, two halves, in that order: measure the gap, then show what
    closing it would stamp. The census first, so the rehearsal is read against
    a report generated moments before rather than last night's."""
    result, calls = _run_wrapper(tmp_path)
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    assert [c['who'] for c in calls] == ['CENSUS', 'STAMP'], calls
    assert any('census_memory_metadata.py' in a for a in calls[0]['argv']), calls[0]
    assert any('retro_stamp_topics.py' in a for a in calls[1]['argv']), calls[1]


def test_wrapper_never_applies_the_stamps_unattended(tmp_path):
    """THE safety property of pairing the gauge with its closing mechanism.

    3201's sweep writes `canonical: true` in bulk. Under
    `memory_metadata.enforce: false` (the shipped default) those writes are not
    even guarded by the write-time uniqueness check, so an unattended `--apply`
    could manufacture the very violations this census exists to count. The
    nightly run is a rehearsal; committing it stays an operator decision.
    """
    _, calls = _run_wrapper(tmp_path)
    stamp = next(c for c in calls if c['who'] == 'STAMP')
    assert '--apply' not in stamp['argv'], stamp['argv']


def test_wrapper_always_exits_zero_so_the_oneshot_never_wedges(tmp_path):
    """A recurring `oneshot` that can fail enters systemd `failed` state and
    STAYS there, silently stopping the whole nightly job (the lesson already
    written into fused-memory-flag-marker-sweep.sh's siblings). The census
    exits 1 by design whenever `coverage.complete` is false -- a routine,
    expected outcome on a live corpus -- so propagating it would wedge the
    timer on the first churny night."""
    for census_exit, stamp_exit in ((1, 0), (0, 1), (1, 1)):
        result, calls = _run_wrapper(
            tmp_path, census_exit=census_exit, stamp_exit=stamp_exit)
        assert result.returncode == 0, (
            f'census_exit={census_exit} stamp_exit={stamp_exit} '
            f'-> rc={result.returncode}; stderr={result.stderr!r}')
        # And neither half is allowed to abort the other.
        assert [c['who'] for c in calls] == ['CENSUS', 'STAMP'], calls


def test_wrapper_reports_both_exit_codes_rather_than_swallowing_them(tmp_path):
    """Exiting 0 must not mean the failure is invisible: the journal is the
    only place a shortfall is readable, so both codes are narrated."""
    result, _ = _run_wrapper(tmp_path, census_exit=1, stamp_exit=3)
    combined = result.stdout + result.stderr
    assert 'census=1' in combined, combined
    assert 'stamp=3' in combined, combined


def test_wrapper_runs_under_the_fused_memory_service_env(tmp_path):
    """A fused-memory maintenance action must run under the SERVICE env, not a
    bare shell, or -- in the wrapper's own words -- "the census silently narrows
    to a different config and censuses a different collection than the artifacts
    claim": the cgl_eta_auto_apply.sh runbook lesson already encoded in
    fused-memory-flag-marker-sweep.sh.

    OBSERVED from inside the invoked census rather than grepped out of the
    wrapper's text. `CONFIG_PATH`, `PROJECT_ROOT` and `uv run` all appear in the
    header comment that DESCRIBES this contract, so a source-text check stays
    green after the real `export` lines are deleted -- it tests the comment, not
    the code. The expected values are derived from the tmp fake repo, so they
    can only be right if the exports actually ran.
    """
    repo = tmp_path / 'fake-repo'
    result, calls = _run_wrapper(tmp_path)
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    census = next(c for c in calls if c['who'] == 'CENSUS')
    assert census['env'] == {
        'CONFIG_PATH': f'{repo}/fused-memory/config/config.yaml',
        'PROJECT_ROOT': str(repo),
        # Rooted even though nothing here reaches FalkorDB: the census imports
        # fused_memory.*, whose config resolution reads it at import time.
        'FALKORDB_URI': 'redis://localhost:6379',
    }, census['env']
    # Both halves are fused-memory maintenance; neither may run bare.
    stamp = next(c for c in calls if c['who'] == 'STAMP')
    assert stamp['env'] == census['env'], stamp['env']


def test_wrapper_runs_both_halves_under_uv_so_fused_memory_imports_resolve(
        tmp_path):
    """The DEFAULT interpreter prefix, exercised for real.

    Every other wrapper test overrides COVERAGE_CENSUS_CMD/RETRO_STAMP_CMD and
    so never reaches `uv run --frozen --project <FM> python` at all -- while the
    literal `uv run` appears TWICE in the header comment, leaving a source-text
    check green with both defaults deleted. Here both seams are UNSET and a fake
    `uv` records what it was actually asked to run: without the prefix each
    script would be exec'd bare and its `fused_memory.*` imports would not
    resolve.
    """
    repo = tmp_path / 'fake-repo'
    result, calls = _run_wrapper(tmp_path, default_prefix=True)
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    # One fake `uv` for both halves, so `who` cannot disambiguate: the census is
    # the first call and the rehearsal the second (ordering pinned elsewhere).
    assert [c['who'] for c in calls] == ['UV', 'UV'], calls
    prefix = ['run', '--frozen', '--project', f'{repo}/fused-memory', 'python']
    expected_scripts = (
        f'{repo}/fused-memory/scripts/census_memory_metadata.py',
        f'{repo}/fused-memory/scripts/retro_stamp_topics.py',
    )
    for call, script in zip(calls, expected_scripts, strict=True):
        argv = call['argv']
        assert argv[:len(prefix)] == prefix, argv
        # The script is the prefix's first argument, never a bare argv[0].
        assert argv[len(prefix)] == script, argv


# ── uv is resolved to an absolute path, not trusted to be on PATH ───────────
#
# A Persistent=true boot catch-up run gets the user manager's minimal PATH, so
# a bare `uv` is not found; the wrapper used to narrate census=127 stamp=127
# and exit 0, losing the night's row silently. These rows hand the wrapper an
# EMPTY PATH and HOME, so the only uv it can reach is the one a row plants.

_NO_LAST_RESORT_UV = pytest.mark.skipif(
    os.path.exists('/usr/local/bin/uv'),
    reason=(
        '/usr/local/bin/uv exists on this host, so the ladder\'s last-resort '
        'fallback resolves and the unresolvable-uv path cannot be exercised'
    ),
)


def _empty_dir(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return directory


def _run_wrapper_on_an_empty_path(tmp_path, *, extra_env=None, **kwargs):
    """_run_wrapper with PATH and HOME each an empty dir; `extra_env` wins."""
    env = {
        'PATH': str(_empty_dir(tmp_path, 'empty-path')),
        'HOME': str(_empty_dir(tmp_path, 'empty-home')),
        **(extra_env or {}),
    }
    return _run_wrapper(tmp_path, extra_env=env, **kwargs)


def _halves(calls):
    """Each recorded call as (who, the five args a uv-run prefix occupies)."""
    return [(c['who'], c['argv'][:5]) for c in calls]


def _uv_run_prefix(tmp_path):
    """The args the fake `uv` receives ahead of the script it is asked to run."""
    return ['run', '--frozen', '--project',
            f'{tmp_path / "fake-repo"}/fused-memory', 'python']


def _both_halves_under_uv(tmp_path):
    prefix = _uv_run_prefix(tmp_path)
    return [('UV', prefix), ('UV', prefix)]


def _diag(result):
    return (f'rc={result.returncode} stdout={result.stdout!r} '
            f'stderr={result.stderr!r}')


def test_wrapper_resolves_uv_via_uv_bin_under_an_empty_path(tmp_path):
    fake_uv = _wrapper_bin_dir(tmp_path) / 'uv'

    result, calls = _run_wrapper_on_an_empty_path(
        tmp_path, default_prefix=True, extra_env={'UV_BIN': str(fake_uv)})

    assert result.returncode == 0, _diag(result)
    assert _halves(calls) == _both_halves_under_uv(tmp_path), _diag(result)


def test_wrapper_falls_back_to_home_local_bin_uv_under_an_empty_path(tmp_path):
    """The boot-catch-up branch: no UV_BIN, no uv on PATH, uv installed at
    $HOME/.local/bin/uv."""
    home = tmp_path / 'uv-home'
    _write_recorder_shim(home / '.local' / 'bin' / 'uv', 'UV',
                         _wrapper_bin_dir(tmp_path) / 'recorder.py')

    result, calls = _run_wrapper_on_an_empty_path(
        tmp_path, default_prefix=True, extra_env={'HOME': str(home)})

    assert result.returncode == 0, _diag(result)
    assert _halves(calls) == _both_halves_under_uv(tmp_path), _diag(result)


@_NO_LAST_RESORT_UV
def test_wrapper_refuses_to_run_either_half_when_uv_cannot_be_resolved(
        tmp_path):
    """The one exception to always-exit-0: a configuration fault that would
    fail every night identically refuses to START, loudly, instead of
    narrating two 127s and exiting 0."""
    result, calls = _run_wrapper_on_an_empty_path(tmp_path, default_prefix=True)

    assert result.returncode == 127, _diag(result)
    assert 'memory-metadata-coverage-census.sh: ERROR:' in result.stderr, (
        _diag(result))
    assert 'uv' in result.stderr, _diag(result)
    assert calls == [], _diag(result)
    assert 'done (' not in result.stdout, _diag(result)


def test_wrapper_refuses_a_non_executable_uv_bin_even_with_uv_on_path(
        tmp_path):
    """Falling through to the uv on PATH would run a DIFFERENT uv than the
    one pinned."""
    bad_uv = tmp_path / 'bad-uv'
    bad_uv.write_text('#!/bin/sh\nexit 0\n')
    bad_uv.chmod(0o644)
    path = f'{_wrapper_bin_dir(tmp_path)}{os.pathsep}' + str(
        _empty_dir(tmp_path, 'empty-path'))

    result, calls = _run_wrapper_on_an_empty_path(
        tmp_path, default_prefix=True,
        extra_env={'PATH': path, 'UV_BIN': str(bad_uv)})

    assert result.returncode == 127, _diag(result)
    assert str(bad_uv) in result.stderr, _diag(result)
    assert calls == [], _diag(result)


@_NO_LAST_RESORT_UV
def test_wrapper_resolves_uv_after_sourcing_dotenv(tmp_path):
    """UV_BIN reaches the wrapper ONLY through $REPO/.env, the remedy an
    operator reaches for after a boot-PATH 127. Resolving before the source
    would find nothing and refuse with 127."""
    fake_uv = _wrapper_bin_dir(tmp_path) / 'uv'
    repo = tmp_path / 'fake-repo'
    repo.mkdir(exist_ok=True)
    (repo / '.env').write_text(f'UV_BIN={fake_uv}\n')

    result, calls = _run_wrapper_on_an_empty_path(tmp_path, default_prefix=True)

    assert result.returncode == 0, _diag(result)
    assert _halves(calls) == _both_halves_under_uv(tmp_path), _diag(result)


@_NO_LAST_RESORT_UV
def test_wrapper_needs_no_uv_when_both_seams_are_set(tmp_path):
    """Resolution is conditional on a half actually needing the default
    prefix: with both seams set, a host with no uv at all still runs both."""
    result, calls = _run_wrapper_on_an_empty_path(
        tmp_path, extra_env={'CENSUS_COMMIT': '0'})

    assert result.returncode == 0, _diag(result)
    assert [c['who'] for c in calls] == ['CENSUS', 'STAMP'], _diag(result)
    assert 'ERROR:' not in result.stderr, _diag(result)


@pytest.mark.parametrize('seam, shim, expected_whos', [
    ('COVERAGE_CENSUS_CMD', 'fake-census', ['CENSUS', 'UV']),
    ('RETRO_STAMP_CMD', 'fake-stamp', ['UV', 'STAMP']),
])
def test_wrapper_resolves_uv_only_for_the_half_without_a_seam(
        tmp_path, seam, shim, expected_whos):
    """The mixed case, in both directions: resolving uv for the unset half
    must leave the set seam driving its own half, unprefixed."""
    bin_dir = _wrapper_bin_dir(tmp_path)
    scripts = tmp_path / 'fake-repo' / 'fused-memory' / 'scripts'

    result, calls = _run_wrapper_on_an_empty_path(
        tmp_path, default_prefix=True,
        extra_env={seam: str(bin_dir / shim), 'UV_BIN': str(bin_dir / 'uv')})

    assert result.returncode == 0, _diag(result)
    assert [c['who'] for c in calls] == expected_whos, _diag(result)
    expected_scripts = (f'{scripts}/census_memory_metadata.py',
                        f'{scripts}/retro_stamp_topics.py')
    for call, script in zip(calls, expected_scripts, strict=True):
        uv_args = _uv_run_prefix(tmp_path) if call['who'] == 'UV' else []
        expected_head = [*uv_args, script]
        assert call['argv'][:len(expected_head)] == expected_head, call


# ── the wrapper lands what it regenerates through the merge queue ───────────
#
# The census regenerates three GIT-TRACKED files under plans/, and the history
# file is APPEND-ONLY: it IS the trend this job builds, so a night's row that
# never reaches main is lost for good. The wrapper lands the changed artifacts
# on $REPO's main through $REPO's own merge queue
# (scripts/_merge_queue_landing.py::land_files), never by a direct commit to
# the machine-operated project_root checkout. These run the real wrapper end
# to end against FakeEscalationMcp + FakeMergeQueue, which lands for real.

_ARTIFACTS = (
    'plans/memory-metadata-census-report.json',
    'plans/memory-metadata-census-report.md',
    'plans/memory-metadata-coverage-history.json',
)
_HISTORY = 'plans/memory-metadata-coverage-history.json'

# What main carries, and a night whose census rewrote all three artifacts.
_BASELINE = {rel: '{"baseline": true}\n' for rel in _ARTIFACTS}
_REGENERATED = {rel: '{"regenerated": true}\n' for rel in _ARTIFACTS}

_DEPLOY_BRANCH = re.compile(r'deploy-memory-metadata-census-\d{8}T\d{6}Z')
_ORCHESTRATOR_CONFIG = 'dark-factory-orchestrator.yaml'


def _git(repo, *args):
    # env= is load-bearing, not tidiness: these are the calls that wrote
    # `[user] name=Test` into the live checkout's .git/config on 2026-08-31.
    # Without the scrub they inherit an ambient GIT_DIR and the `-C` below is
    # decorative. See the _scrub_git_env block above.
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True, text=True, check=False,
        env=_scrub_git_env(dict(os.environ)),
    )


def _git_repo_harness(tmp_path, *, tracked=True, initial_branch='main'):
    """Make $REPO a REAL, clean git repo with every file committed.

    The night's drift is what the fake census writes (`census_writes`), so
    what is under test is the wrapper's handling of it, not the census's
    rendering.

    ``tracked=False`` models the FIRST-EVER run: git has never seen the
    artifacts, so whatever the census writes is new.
    """
    repo = tmp_path / 'fake-repo'
    (repo / 'plans').mkdir(parents=True, exist_ok=True)
    (repo / 'fused-memory' / 'scripts').mkdir(parents=True, exist_ok=True)

    _git(repo, 'init', '-q', '-b', initial_branch)
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test')
    # No hooks, no signing -- this exercises the wrapper, not the repo's gates.
    _git(repo, 'config', 'commit.gpgsign', 'false')

    if tracked:
        for rel, content in _BASELINE.items():
            (repo / rel).write_text(content)
        _git(repo, 'add', '--', *_ARTIFACTS)
    else:
        (repo / 'README.md').write_text('unrelated\n')
        _git(repo, 'add', '--', 'README.md')
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'baseline')
    return repo


def _commit_orchestrator_config(repo, port):
    """Commit *repo*'s dark-factory-orchestrator.yaml naming escalation *port*
    (None: an escalation section with no port), so the checkout stays clean."""
    escalation = f'escalation:\n  port: {port}\n' if port is not None else 'escalation: {}\n'
    (repo / _ORCHESTRATOR_CONFIG).write_text(f'project_root: /nowhere\n{escalation}')
    _git(repo, 'add', '--', _ORCHESTRATOR_CONFIG)
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'orchestrator config')


@contextlib.contextmanager
def _census_queue(repo, *, port_in_config=True, **queue_kwargs):
    """*repo*'s merge queue: a FakeMergeQueue(**queue_kwargs) behind a
    loopback escalation MCP whose port *repo*'s committed orchestrator config
    names (none when *port_in_config* is false). Yields (queue, server)."""
    queue = FakeMergeQueue(repo, **queue_kwargs)
    with FakeEscalationMcp(tool_handlers=queue.handlers()) as server:
        _commit_orchestrator_config(repo, server.port if port_in_config else None)
        yield queue, server


def _run_wrapper_in_git_repo(tmp_path, repo, **kwargs):
    env, state_path = _wrapper_harness(tmp_path, **kwargs)
    env['REPO'] = str(repo)
    result = subprocess.run(
        ['bash', str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    return result, json.loads(state_path.read_text())


def _main(repo):
    return _git(repo, 'rev-parse', 'main').stdout.strip()


def _landed_files(repo):
    """The paths the deploy commit main^2 changed."""
    return sorted(_git(repo, 'show', '--name-only', '--format=', 'main^2').stdout.split())


def _assert_left_nothing_behind(tmp_path, repo):
    assert_target_checkout_clean(repo)
    scratch = _scratch_dir(tmp_path)
    assert os.listdir(scratch) == [], f'scratch left behind: {os.listdir(scratch)}'


def _utc_dates_spanning(run):
    """The UTC dates a census commit message made during *run* may carry."""
    before = datetime.now(UTC).strftime('%Y-%m-%d')
    outcome = run()
    return outcome, {before, datetime.now(UTC).strftime('%Y-%m-%d')}


def test_a_drift_night_lands_its_artifacts_through_one_normal_lane_request(tmp_path):
    """THE point of the job: the history row reaches main, gate-verified, as a
    merge of a unique deploy branch, and project_root is never written."""
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo, states=['verifying', 'done']) as (queue, _):
        base = _main(repo)
        (result, _), dates = _utc_dates_spanning(lambda: _run_wrapper_in_git_repo(
            tmp_path, repo, census_writes=_REGENERATED))

    assert result.returncode == 0, _diag(result)
    (submitted,) = queue.calls_to('merge_request')
    assert submitted['lane'] == 'normal' and submitted['verified_green'] is False, submitted
    assert _DEPLOY_BRANCH.fullmatch(submitted['branch']), submitted
    assert repo.resolve() not in Path(submitted['worktree']).resolve().parents, submitted
    assert _git(repo, 'log', '-1', '--format=%s', 'main').stdout.strip() == (
        f'Merge task/{submitted["branch"]} into main')
    subject = _git(repo, 'log', '-1', '--format=%s', 'main^2').stdout.strip()
    assert subject in {f'chore(census): nightly canonical/topic coverage census {d}'
                       for d in dates}, subject
    assert sorted(_git(repo, 'diff', '--name-only', base, 'main').stdout.split()) == sorted(
        _ARTIFACTS)
    for rel, content in _REGENERATED.items():
        assert (repo / rel).read_text() == content, rel
    assert 'commit=0 commit-step=landed:mr-deploy0001)' in _done_line(result.stdout), _diag(result)
    _assert_left_nothing_behind(tmp_path, repo)


def test_the_first_ever_run_lands_the_artifacts_as_new_files(tmp_path):
    repo = _git_repo_harness(tmp_path, tracked=False)
    with _census_queue(repo) as _:
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)

    assert _commit_step(result) == 'landed:mr-deploy0001', _diag(result)
    added = _git(repo, 'diff', '--name-only', '--diff-filter=A', 'main^1', 'main').stdout.split()
    assert sorted(added) == sorted(_ARTIFACTS), added
    _assert_left_nothing_behind(tmp_path, repo)


def test_a_partial_drift_lands_only_the_artifacts_that_changed(tmp_path):
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo) as _:
        result, _ = _run_wrapper_in_git_repo(
            tmp_path, repo, census_writes={_HISTORY: _REGENERATED[_HISTORY]})

    assert _commit_step(result) == 'landed:mr-deploy0001', _diag(result)
    assert _landed_files(repo) == [_HISTORY]


def test_bystander_state_in_repo_survives_and_stays_out_of_the_landing(tmp_path):
    """A concurrent process's WIP in project_root is not the census's to
    land or to touch. No staged bystander: the fake lands with a real
    `git merge`, which aborts on a dirty index the lane's own advance parks."""
    repo = _git_repo_harness(tmp_path)
    tracked_bystander = repo / 'README.md'
    tracked_bystander.write_text('unrelated tracked edit\n')
    _git(repo, 'add', '--', 'README.md')
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'bystander baseline')
    tracked_bystander.write_text('unrelated tracked edit, now dirty\n')
    untracked_bystander = repo / 'plans' / 'someone-elses-wip.md'
    untracked_bystander.write_text('concurrent work, not ours\n')

    with _census_queue(repo) as _:
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)

    assert _commit_step(result) == 'landed:mr-deploy0001', _diag(result)
    assert _landed_files(repo) == sorted(_ARTIFACTS)
    assert tracked_bystander.read_text() == 'unrelated tracked edit, now dirty\n'
    assert untracked_bystander.read_text() == 'concurrent work, not ours\n'


def test_a_night_the_census_reported_a_shortfall_still_lands(tmp_path):
    """The census exits 1 by design when `coverage.complete` is false, and the
    artifacts it wrote carry the evidence of that shortfall."""
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo) as _:
        result, _ = _run_wrapper_in_git_repo(
            tmp_path, repo, census_exit=1, census_writes=_REGENERATED)

    assert result.returncode == 0, _diag(result)
    assert 'census=1 ' in _done_line(result.stdout), _diag(result)
    assert _commit_step(result) == 'landed:mr-deploy0001', _diag(result)
    assert _landed_files(repo) == sorted(_ARTIFACTS)


def test_a_night_without_drift_asks_the_queue_nothing(tmp_path):
    """A quiet night regenerates main's exact bytes."""
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo) as (_, server):
        before = _main(repo)
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_BASELINE)
        calls = server.tool_calls()

    assert result.returncode == 0, _diag(result)
    assert 'commit=0 commit-step=skipped:no-drift)' in _done_line(result.stdout), _diag(result)
    assert calls == []
    assert _main(repo) == before
    _assert_left_nothing_behind(tmp_path, repo)


def test_a_landing_the_queue_refuses_is_narrated_and_still_exits_zero(tmp_path):
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo, states=['verifying', 'conflict']) as _:
        before = _main(repo)
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)

    assert result.returncode == 0, _diag(result)
    assert 'commit=1 commit-step=failed:not_landed)' in _done_line(result.stdout), _diag(result)
    assert _main(repo) == before
    _assert_left_nothing_behind(tmp_path, repo)


def test_the_landing_ceiling_is_the_units_environment_knob(tmp_path):
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo, states=['verifying'], eta_seconds=0.2) as (queue, _):
        result, _ = _run_wrapper_in_git_repo(
            tmp_path, repo, census_writes=_REGENERATED,
            extra_env={'CENSUS_LANDING_TIMEOUT_SECS': '1'})

    assert _commit_step(result) == 'failed:timed_out', _diag(result)
    assert queue.calls_to('merge_cancel') == [{'request_id': queue.request_id}]
    _assert_left_nothing_behind(tmp_path, repo)


def test_a_config_naming_no_escalation_port_is_refused_before_any_call(tmp_path):
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo, port_in_config=False) as (_, server):
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)
        calls = server.tool_calls()

    assert result.returncode == 0, _diag(result)
    assert 'commit=1 commit-step=refused:no-escalation-port)' in _done_line(result.stdout), (
        _diag(result))
    assert calls == []


def test_uncommitted_state_at_an_artifact_path_is_refused_and_left_alone(tmp_path):
    """Landing over it would halt the whole queue (wip_overlap) or overwrite it."""
    repo = _git_repo_harness(tmp_path)
    dirty = repo / _HISTORY
    dirty.write_text('{"an operator is editing this": true}\n')

    with _census_queue(repo) as (queue, _):
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)

    assert _commit_step(result) == 'failed:checkout_dirty', _diag(result)
    assert dirty.read_text() == '{"an operator is editing this": true}\n'
    assert queue.calls_to('merge_request') == []


def test_wrapper_still_exits_zero_when_the_commit_cannot_happen(tmp_path):
    """Same oneshot contract as the other two halves: a landing that cannot
    happen is narrated, never propagated, or the timer wedges in `failed`
    state and the trend quietly ends. Exercised with $REPO not a git repo at
    all -- which is also the state the argv and uv tests run in."""
    result, calls = _run_wrapper(tmp_path)          # plain tmp dir, no .git
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    # Both halves still ran; the landing step aborted neither.
    assert [c['who'] for c in calls] == ['CENSUS', 'STAMP'], calls


def test_wrapper_narrates_the_commit_outcome_alongside_the_other_two(tmp_path):
    """Exiting 0 must never mean the outcome is invisible -- the journal is the
    only place this job is readable."""
    repo = _git_repo_harness(tmp_path)
    with _census_queue(repo) as _:
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)
    line = _done_line(result.stdout)
    assert 'census=0 stamp=0 commit=0 commit-step=' in line, line


def _done_line(output):
    """The wrapper's own final `done (...)` narration line, or '' if absent."""
    for line in output.splitlines():
        if line.startswith('memory-metadata-coverage-census: done ('):
            return line
    return ''


def _parse_commit_step(line):
    """Pull the `commit-step=<token>` field out of one done(...) line, or None.

    Parses the ISOLATED done-line only, never the whole combined output, so a
    token appearing merely in earlier prose (or a stray substring match) can
    never satisfy this.
    """
    marker = 'commit-step='
    idx = line.find(marker)
    if idx == -1:
        return None
    token = line[idx + len(marker):].split(')')[0].strip()
    return token or None


def test_the_wrapper_names_which_commit_step_branch_it_took(tmp_path, monkeypatch):
    """Of the eight commit-step outcomes driven here, FIVE narrate `commit=0`
    on the final done(...) line -- which reads as success while nothing
    landed. In an unattended nightly whose journal is its only readable
    surface, "declined" and "landed" being indistinguishable in the
    summary line is exactly the silent-degradation shape CLAUDE.md's
    loud-over-silent norm warns about.

    Drives the REAL wrapper into all eight branches with the harnesses already
    established in this file (INV-10 `guards-exercise-behaviour`: no
    source-text scanning) and asserts every done(...) line carries a
    `commit-step=<token>` field EQUAL TO the token that branch is contracted
    to emit -- `scenarios`'s own keys double as the expected tokens, so the
    mapping assertion below checks each branch against ITSELF, not merely
    against its neighbours. A pairwise-distinctness check would still pass if
    two branches emitted each other's token under swapped names (verified by
    mutation: renaming `skipped:disabled` <-> `skipped:not-a-git-repo`'s
    tokens left a same-shaped distinctness check green); this catches that.
    """
    scenarios = {}

    for token, states in (('landed:mr-deploy0001', ['verifying', 'done']),
                          ('failed:not_landed', ['verifying', 'conflict'])):
        scene = tmp_path / token.replace(':', '-')
        landing_repo = _git_repo_harness(scene)
        with _census_queue(landing_repo, states=states):
            scenarios[token], _ = _run_wrapper_in_git_repo(
                scene, landing_repo, census_writes=_REGENERATED)

    no_drift_repo = _git_repo_harness(tmp_path / 'no-drift')
    scenarios['skipped:no-drift'], _ = _run_wrapper_in_git_repo(
        tmp_path / 'no-drift', no_drift_repo)

    no_main_repo = _git_repo_harness(tmp_path / 'no-main', initial_branch='trunk')
    scenarios['failed:generation-worktree'], _ = _run_wrapper_in_git_repo(
        tmp_path / 'no-main', no_main_repo)

    in_flight = tmp_path / 'in-flight'
    in_flight_repo = _git_repo_harness(in_flight)
    retained, _ = _retained_census_deploy(in_flight, in_flight_repo, monkeypatch)
    with _census_queue(in_flight_repo, states={retained.removeprefix('task/'): ['verifying']}):
        scenarios['skipped:landing-in-flight'], _ = _run_wrapper_in_git_repo(
            in_flight, in_flight_repo, census_writes=_REGENERATED)

    # _wrapper_harness builds `<dir>/wbin` without `parents=True`, so (unlike
    # the _git_repo_harness-backed scenarios above, whose own `parents=True`
    # mkdir already creates their subdirectory) these two must create theirs
    # first.
    not_a_repo_dir = tmp_path / 'not-a-git-repo'
    not_a_repo_dir.mkdir()
    scenarios['skipped:not-a-git-repo'], _ = _run_wrapper(not_a_repo_dir)

    outer_repo = _git_repo_harness(tmp_path / 'outer')
    inner = outer_repo / 'plans'  # a real subdirectory, not the repo root
    scenarios['refused:repo-not-toplevel'], _ = _run_wrapper_in_git_repo(
        tmp_path / 'outer', inner)

    disabled_dir = tmp_path / 'disabled'
    disabled_dir.mkdir()
    scenarios['skipped:disabled'], _ = _run_wrapper(
        disabled_dir, extra_env={'CENSUS_COMMIT': '0'})

    tokens = {}
    for label, result in scenarios.items():
        assert result.returncode == 0, (
            f'{label}: rc={result.returncode} '
            f'stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}')
        combined = result.stdout + result.stderr
        line = _done_line(combined)
        assert line, (
            f'{label}: no done(...) narration line found; '
            f'stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}')
        token = _parse_commit_step(line)
        assert token is not None, (
            f'{label}: done(...) line carries no commit-step= field: '
            f'{line!r}')
        tokens[label] = token

    # The MAPPING, not merely distinctness: `scenarios`'s keys are the tokens
    # each branch is contracted to emit, so this fails if a branch narrates
    # the wrong (if still distinct) token -- e.g. two branches swapping
    # tokens under different names, which a pairwise-distinctness check
    # cannot see (distinctness follows for free here, from distinct keys).
    assert tokens == {label: label for label in scenarios}, tokens


# The forbidden-git-verb guard, asserted on the argv the wrapper ACTUALLY
# invokes.
#
# HISTORY, so this is not re-litigated in either direction. The original guard
# scanned the wrapper's raw TEXT for bare literals ('git stash', 'git add -A',
# ...), but every git call here is spelled `git -C "$REPO" <verb>`, so 5 of 6
# violations written in the file's own idiom were invisible and the scan passed
# regardless of what the wrapper did. The review of 2026-08-16 then flagged the
# hardened regex successor as a lint-rule-dressed-as-a-test: it stripped
# comments with an admittedly unsound `\s#` split, carried an anti-vacuity
# floor (`code.count('git ') >= 5`) that broke on benign refactors, and needed a
# second test just to prove its own patterns could fire against literals defined
# in this same file -- green by construction, and touching zero production code.
#
# Both problems are the same problem: a TEXT scan cannot tell code from prose.
# Recording argv cannot be fooled by either, and needs no comment stripping, no
# regex over source, and no hand-maintained copy of the wrapper's legitimate
# calls. The scenarios below reach every git call the wrapper makes (rev-parse,
# the generation worktree's add, status and removal) and, through the landing,
# every one land_files makes in $REPO and its deploy worktree, so this observes
# the whole git surface.
_FORBIDDEN_GIT_SUBCOMMANDS = frozenset({'stash', 'reset', 'checkout', 'clean'})


def _forbidden_reason(argv):
    """The offending shape in one recorded git argv, or None.

    Operates on the ARGV LIST, so `-C <dir>` is skipped structurally rather
    than normalised away with a regex, and an option value can never be
    mistaken for a subcommand.
    """
    rest = list(argv)
    while rest[:1] == ['-C'] and len(rest) >= 2:
        rest = rest[2:]
    if not rest:
        return None
    sub, opts = rest[0], rest[1:]
    if sub in _FORBIDDEN_GIT_SUBCOMMANDS:
        return f'`git {sub}` is forbidden in a machine-operated checkout'
    if sub == 'add' and any(o in ('-A', '--all', '.') for o in opts):
        return 'wholesale `git add` sweeps a concurrent process WIP into our commit'
    if sub == 'commit' and any(o in ('-a', '--all') for o in opts):
        return '`git commit -a` stages every tracked modification, ours or not'
    return None


def test_the_forbidden_reason_helper_actually_fires():
    """The guard is only worth its name if it can fire on the wrapper's idiom.

    Unlike the self-referential predecessor this replaces, the synthetic
    violations here are fed through the SAME `_forbidden_reason` the real test
    applies to recorded argv -- so this pins the detector, and the test below
    supplies the subject.
    """
    for argv in (
        ['stash'], ['-C', '/repo', 'stash'], ['-C', '/repo', 'reset', '--hard'],
        ['-C', '/repo', 'checkout', 'main'], ['-C', '/repo', 'clean', '-fd'],
        ['add', '-A'], ['-C', '/repo', 'add', '--all'], ['-C', '/repo', 'add', '.'],
        ['commit', '-a', '-m', 'x'], ['-C', '/repo', 'commit', '--all', '-m', 'x'],
    ):
        assert _forbidden_reason(argv), f'{argv!r} must be caught'
    # And the real calls must never trip it -- notably the landing's scoped
    # `add --`, whose `--` is neither -A, --all nor `.`.
    for argv in (
        ['-C', '/repo', 'rev-parse', '--git-dir'],
        ['-C', '/repo', 'worktree', 'add', '--detach', '--quiet', '/tmp/x/tree', 'abc123'],
        ['-C', '/tmp/x/tree', 'status', '--porcelain', '--', 'a.json'],
        ['-C', '/deploy', 'add', '--', 'a.json'],
        ['-C', '/deploy', 'commit', '-q', '--no-verify', '-m', 'msg', '--', 'a.json'],
    ):
        assert _forbidden_reason(argv) is None, f'{argv!r} false-positived'


def _commit_step(result):
    """The wrapper's own `commit-step=<token>` verdict for one run.

    Parses it out of the wrapper's final `done (...)` narration in
    `result.stdout + result.stderr`, reusing `_done_line` / `_parse_commit_step`
    above. Returns the sentinel '<unreported>' -- NEVER raises -- when the
    done(...) line or the field itself is absent, because this helper's whole
    job is to make a bad run REPORTABLE rather than to die on an unrelated
    exception (an IndexError would give a guard failure that names nothing).
    """
    combined = result.stdout + result.stderr
    line = _done_line(combined)
    token = _parse_commit_step(line) if line else None
    return token if token is not None else '<unreported>'


def test_the_commit_step_reader_reports_a_decline_as_a_decline():
    """Pin the reader before trusting it, in the shape this file already
    established for `_forbidden_reason` just above:
    `test_the_forbidden_reason_helper_actually_fires` pins the detector
    against cheap SYNTHETIC input, and the real test
    (`test_wrapper_never_invokes_a_forbidden_git_verb`, below) supplies the
    subject. `_commit_step` gets the same treatment here -- driven against
    literal done(...) narration, not real wrapper runs.

    Real-wrapper coverage of these three branches already exists elsewhere:
    `test_the_wrapper_names_which_commit_step_branch_it_took`'s
    branch->token mapping assertion, and the honest-failure assertion in
    `test_wrapper_never_invokes_a_forbidden_git_verb`. Re-driving a real git
    repo and subprocess three more times here would prove the same behaviour
    a fourth and fifth time rather than a new one -- `_commit_step` is a
    six-line pure-string helper (`_done_line` + `_parse_commit_step` + a
    sentinel), and synthetic input pins its parsing directly.
    """
    def _stub(stdout):
        return subprocess.CompletedProcess(
            args=['bash'], returncode=0, stdout=stdout, stderr='')

    landed = _stub(
        'memory-metadata-coverage-census: done (census=0 stamp=0 commit=0 '
        'commit-step=landed:mr-deploy0001)\n')
    assert _commit_step(landed) == 'landed:mr-deploy0001'

    refused = _stub(
        'memory-metadata-coverage-census: done (census=0 stamp=0 commit=1 '
        'commit-step=refused:repo-not-toplevel)\n')
    assert _commit_step(refused) == 'refused:repo-not-toplevel'

    no_drift = _stub(
        'memory-metadata-coverage-census: done (census=0 stamp=0 commit=0 '
        'commit-step=skipped:no-drift)\n')
    assert _commit_step(no_drift) == 'skipped:no-drift'

    # A non-wrapper input carrying no narration at all must degrade to the
    # named sentinel instead of raising -- an IndexError here would make the
    # guard's own failure unreadable.
    stub = subprocess.CompletedProcess(args=['x'], returncode=0, stdout='', stderr='')
    assert _commit_step(stub) == '<unreported>'


def test_wrapper_never_invokes_a_forbidden_git_verb(tmp_path):
    """CLAUDE.md's hardest prohibition, on a script that acts UNATTENDED on
    the machine-operated project_root checkout: its generation worktree and
    its landing both run git there.

    `git stash` is the sharpest of these: refs/stash is a SINGLE ref shared by
    every worktree, and the merge worker's advance path also consumes it
    (incident 13674d3c68), so a stash here can be popped out from under an
    unrelated process.

    Each scenario's evidence stays SEPARATE rather than pooled, so a
    precondition decline in one iteration cannot hide behind a healthy run in
    another (esc-3647-3): each iteration asserts its OWN commit-step branch
    and, only once that is confirmed, its OWN argv. The landing's git calls
    run under the same recording shim, which land_files inherits on PATH.
    """
    scenarios = (
        # (label, whether main tracks the artifacts, what the census writes,
        #  expected commit-step,
        #  git-argv substrings that scenario's OWN run must contain)
        ('already-tracked artifacts', True, _REGENERATED, 'landed:mr-deploy0001',
         ('worktree add --detach', ' add -- ')),
        ('first-ever run, untracked', False, _REGENERATED,
         'landed:mr-deploy0001', ('worktree add --detach', ' add -- ')),
        ('a quiet night with no drift', True, None, 'skipped:no-drift',
         ('worktree add --detach',)),
    )
    for label, tracked, census_writes, expected_step, expect_substrings in scenarios:
        repo = _git_repo_harness(tmp_path / label.split()[0], tracked=tracked)
        env, _ = _wrapper_harness(tmp_path, record_git=True, census_writes=census_writes)
        env['REPO'] = str(repo)
        with _census_queue(repo):
            result = subprocess.run(
                ['bash', str(WRAPPER)], env=env,
                capture_output=True, text=True, timeout=120,
            )
        assert result.returncode == 0, f'{label}: {result.stderr!r}'
        recorded = [
            json.loads(line)
            for line in _git_argv_path(tmp_path).read_text().splitlines() if line
        ]
        # ANTI-VACUITY, asserted as OBSERVED BEHAVIOUR rather than as a text
        # shape: if the shim recorded nothing the wrapper never reached git and
        # every "no forbidden verb" assertion below would hold vacuously.
        assert recorded, f'{label}: the wrapper invoked git zero times'
        for argv in recorded:
            reason = _forbidden_reason(argv)
            assert reason is None, f'{label}: `git {" ".join(argv)}` — {reason}'

        observed_step = _commit_step(result)
        assert observed_step == expected_step, (
            f'{label}: expected commit-step={expected_step!r}, observed '
            f'{observed_step!r}; recorded={recorded!r} '
            f'stderr={result.stderr[-2000:]!r}')

        flat = [' '.join(a) for a in recorded]
        for expect_substring in expect_substrings:
            assert any(expect_substring in f'{c} ' for c in flat), (
                f'{label}: no git call recorded containing {expect_substring!r}: '
                f'{flat!r}')


# ── ambient GIT_* can never redirect this job into another repo ──────────────
#
# Regression guard for the 2026-08-31 incident, in which a run of THIS FILE
# committed placeholder content onto main in the live project_root checkout and
# wrote `[user] name=Test email=test@example.com` + `commit.gpgsign = false`
# into its real .git/config.
#
# The mechanism is not a missing $REPO -- _wrapper_harness has pinned REPO to a
# tmp dir since the file was written. It is that `git -C "$REPO"` does not mean
# "act on $REPO": `-C` only changes directory, while GIT_DIR skips repository
# discovery entirely, so an ambient GIT_DIR overrides BOTH the -C and the $REPO
# and every git call lands in the repository it names. GIT_CEILING_DIRECTORIES
# does not cover this -- a ceiling bounds the upward WALK, and GIT_DIR never
# walks.
#
# INV-10 (`guards-exercise-behaviour`): these RUN the real wrapper and the real
# harness against a decoy repository standing in for the live checkout and
# assert the decoy is untouched. Nothing here greps the wrapper's text -- a
# source scan would have stayed green through the entire incident, since the
# wrapper's `git -C "$REPO"` spelling was never what was wrong.


def _decoy_live_repo(tmp_path):
    """A real repo standing in for the live checkout, with its own identity.

    Carries a main and the same tracked artifact paths as the real thing, so
    a redirected generation worktree or landing would SUCCEED here rather
    than failing on an unknown ref or path -- otherwise the guard could pass
    for the wrong reason.
    """
    decoy = tmp_path / 'decoy-live-checkout'
    (decoy / 'plans').mkdir(parents=True)
    _git(decoy, 'init', '-q', '-b', 'main')
    _git(decoy, 'config', 'user.email', 'real-operator@example.invalid')
    _git(decoy, 'config', 'user.name', 'Real Operator')
    for rel in _ARTIFACTS:
        (decoy / rel).write_text('{"real": true}\n')
    _git(decoy, 'add', '--', *_ARTIFACTS)
    _git(decoy, '-c', 'commit.gpgsign=false', 'commit', '-q', '--no-verify',
         '-m', 'real history')
    return decoy


def _decoy_state(decoy):
    """The two things the incident actually damaged: identity, and history."""
    return {
        'user.email': _git(decoy, 'config', '--local', '--get', 'user.email').stdout,
        'user.name': _git(decoy, 'config', '--local', '--get', 'user.name').stdout,
        'gpgsign': _git(decoy, 'config', '--local', '--get', 'commit.gpgsign').stdout,
        'log': _git(decoy, 'log', '--pretty=%H %s').stdout,
    }


def test_wrapper_cannot_commit_into_a_repo_named_only_by_ambient_git_dir(tmp_path):
    """THE incident, reproduced end to end against a decoy.

    GIT_DIR is injected via `extra_env`, which _wrapper_harness applies AFTER
    its own scrub -- deliberately, so this exercises the WRAPPER's own
    `unset GIT_DIR ...` (the production-side defence, which must hold when the
    wrapper is run by systemd, a git hook, or any shell this file never touches)
    rather than the harness's scrub.
    """
    decoy = _decoy_live_repo(tmp_path)
    before = _decoy_state(decoy)

    repo = _git_repo_harness(tmp_path / 'sandbox')
    with _census_queue(repo):
        result, state = _run_wrapper_in_git_repo(
            tmp_path, repo, extra_env={'GIT_DIR': str(decoy / '.git')},
            census_writes=_REGENERATED,
        )

    assert _decoy_state(decoy) == before, (
        'the wrapper wrote into a repository named only by an ambient GIT_DIR; '
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    # POSITIVE CONTROL: the run must have done its real work in the sandbox.
    # Without this the assertion above passes just as well when the wrapper
    # dies on line 1 and touches nothing anywhere.
    assert result.returncode == 0, _diag(result)
    assert [c['who'] for c in state] == ['CENSUS', 'STAMP'], state
    assert _commit_step(result) == 'landed:mr-deploy0001', (
        f'the sandbox landing did not happen, so the guard proved nothing: {_diag(result)}')
    assert _landed_files(repo) == sorted(_ARTIFACTS)


def test_wrapper_commit_runs_no_hook_injected_by_ambient_git_config_parameters(
        tmp_path):
    """GIT_CONFIG_PARAMETERS is git's own `-c` channel, read at command-line
    precedence, so an ambient one can name a core.hooksPath whose hooks the
    generation worktree's checkout (post-checkout) and the landing's deploy
    commit (post-commit, which --no-verify does not skip) would run. Injected
    via `extra_env`, AFTER the harness scrub, so the wrapper's own `unset` is
    what is under test."""
    hooks_dir = tmp_path / 'decoy-hooks'
    hooks_dir.mkdir()
    marker = tmp_path / 'hook-ran'
    for name in ('pre-commit', 'post-commit', 'post-checkout'):
        hook = hooks_dir / name
        hook.write_text(f'#!/bin/sh\ntouch {marker}\nexit 0\n')
        hook.chmod(0o755)
    poisoned = f"'core.hooksPath={hooks_dir}'"

    # PREMISE CONTROL: this host's git really runs a hook injected this way,
    # so the absence asserted below cannot be vacuous.
    premise = _git_repo_harness(tmp_path / 'premise')
    probe = subprocess.run(
        ['git', '-C', str(premise), 'commit', '-q', '--allow-empty', '-m', 'probe'],
        env={**_scrub_git_env(dict(os.environ)), 'GIT_CONFIG_PARAMETERS': poisoned},
        capture_output=True, text=True, check=False,
    )
    assert marker.exists(), (
        f'the injected hook did not run in the premise probe: {probe!r}')
    marker.unlink()

    repo = _git_repo_harness(tmp_path / 'sandbox')
    with _census_queue(repo):
        result, state = _run_wrapper_in_git_repo(
            tmp_path, repo, extra_env={'GIT_CONFIG_PARAMETERS': poisoned},
            census_writes=_REGENERATED,
        )

    assert not marker.exists(), (
        'the wrapper ran a hook injected by an ambient '
        f'GIT_CONFIG_PARAMETERS; stdout={result.stdout!r} stderr={result.stderr!r}')

    # POSITIVE CONTROL: the checkout and the deploy commit really happened, so
    # a hook had its chance.
    assert result.returncode == 0, _diag(result)
    assert [c['who'] for c in state] == ['CENSUS', 'STAMP'], state
    assert _commit_step(result) == 'landed:mr-deploy0001', (
        f'the sandbox landing did not happen, so the guard proved nothing: {_diag(result)}')


def test_harness_git_calls_cannot_be_redirected_by_ambient_git_dir(
        tmp_path, monkeypatch):
    """The OTHER half of the incident: the harness's own identity writes.

    `[user] name=Test email=test@example.com` and `commit.gpgsign = false` are
    _git_repo_harness's literals, and they are what landed in the live
    checkout's .git/config. Exercises _git/_scrub_git_env, not the wrapper.
    """
    decoy = _decoy_live_repo(tmp_path)
    before = _decoy_state(decoy)

    monkeypatch.setenv('GIT_DIR', str(decoy / '.git'))
    repo = _git_repo_harness(tmp_path / 'sandbox')

    assert _decoy_state(decoy) == before, (
        'the harness wrote its fixture identity/commits into a repository '
        'named only by an ambient GIT_DIR')
    # POSITIVE CONTROL: the harness really did build its own repo.
    assert _git(repo, 'config', '--local', '--get', 'user.email').stdout.strip() \
        == 'test@example.com'


def test_wrapper_refuses_to_commit_when_repo_is_not_the_repository_root(tmp_path):
    """Fail-closed backstop, independent of any GIT_* name.

    The `unset` above removes the KNOWN redirection vars; this pins the
    residual-case guard -- $REPO resolving to a repository whose root is
    somewhere else (a nested or symlinked checkout, a $REPO pointed at a
    subdirectory, or a redirection var git adds in a future release). The
    wrapper must decline rather than commit into it.
    """
    outer = _git_repo_harness(tmp_path / 'outer')
    before = _git(outer, 'log', '--pretty=%H %s').stdout

    inner = outer / 'plans'  # a real subdirectory of a real repo, not its root
    result, _ = _run_wrapper_in_git_repo(tmp_path, inner)

    assert 'REFUSING' in result.stderr, result.stderr
    assert _git(outer, 'log', '--pretty=%H %s').stdout == before, (
        'the wrapper committed into the enclosing repository it was not told '
        f'about; stderr={result.stderr!r}')
    # Still exits 0: a refused commit must not wedge the recurring oneshot.
    assert result.returncode == 0, result.stderr


# ── the census measures in a generation worktree, never in $REPO ────────────
#
# A checkout root's census runs on a detached worktree of main's tip, so the
# census writes no checkout's working tree but its own: dirty tracked files at
# the landed paths in project_root are what halt the merge queue (wip_overlap).
# A $REPO inside no checkout still runs in place.

def _census_call(calls):
    (census,) = [c for c in calls if c['who'] == 'CENSUS']
    return census


def _worktree_paths(repo):
    listing = _git(repo, 'worktree', 'list', '--porcelain').stdout
    return [line.removeprefix('worktree ') for line in listing.splitlines()
            if line.startswith('worktree ')]


def _measure_only(tmp_path, repo, **kwargs):
    """A CENSUS_COMMIT=0 run of the wrapper on *repo*: measure and rehearse,
    land nothing."""
    return _run_wrapper_in_git_repo(
        tmp_path, repo, extra_env={'CENSUS_COMMIT': '0'}, **kwargs)


def test_the_census_measures_a_linked_worktree_at_mains_tip(tmp_path):
    repo = _git_repo_harness(tmp_path)
    main = _git(repo, 'rev-parse', 'main').stdout.strip()

    result, calls = _measure_only(tmp_path, repo)

    assert result.returncode == 0, _diag(result)
    census = _census_call(calls)
    root = Path(census['root'])
    assert census['argv'][0].endswith('/fused-memory/scripts/census_memory_metadata.py'), census
    assert root != repo.resolve() and repo.resolve() not in root.parents, census
    assert census['common_dir'] == str((repo / '.git').resolve()), census
    assert census['head'] == main, census


@pytest.mark.parametrize('census_exit', [0, 1])
def test_a_run_leaves_the_checkout_and_tmpdir_as_it_found_them(tmp_path, census_exit):
    repo = _git_repo_harness(tmp_path)

    result, calls = _measure_only(
        tmp_path, repo, census_exit=census_exit, census_writes=_REGENERATED)

    assert result.returncode == 0, _diag(result)
    assert _census_call(calls)['root'], calls
    status = _git(repo, 'status', '--porcelain', '--', *_ARTIFACTS).stdout
    assert status == '', f'the census wrote into $REPO: {status!r}'
    assert _worktree_paths(repo) == [str(repo)], _worktree_paths(repo)
    assert os.listdir(_scratch_dir(tmp_path)) == [], os.listdir(_scratch_dir(tmp_path))


def test_a_repo_inside_a_checkout_but_not_its_root_runs_no_census(tmp_path):
    outer = _git_repo_harness(tmp_path / 'outer')
    before = (_git(outer, 'log', '--pretty=%H').stdout,
              _git(outer, 'status', '--porcelain', '--untracked-files=all').stdout)

    result, calls = _measure_only(
        tmp_path / 'outer', outer / 'plans', census_writes=_REGENERATED)

    assert result.returncode == 0, _diag(result)
    assert [c['who'] for c in calls] == ['STAMP'], calls
    assert 'census=skipped ' in _done_line(result.stdout), _diag(result)
    assert (_git(outer, 'log', '--pretty=%H').stdout,
            _git(outer, 'status', '--porcelain', '--untracked-files=all').stdout) == before


def test_a_checkout_without_main_runs_no_census(tmp_path):
    """No main means no tip to measure, and measuring in place instead would
    write the very checkout the worktree exists to keep clean."""
    repo = _git_repo_harness(tmp_path, initial_branch='trunk')
    before = _git(repo, 'log', '--pretty=%H').stdout

    result, calls = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)

    assert result.returncode == 0, _diag(result)
    assert [c['who'] for c in calls] == ['STAMP'], calls
    line = _done_line(result.stdout)
    assert 'census=skipped ' in line and 'commit=1 ' in line, _diag(result)
    assert _commit_step(result) == 'failed:generation-worktree', _diag(result)
    assert _git(repo, 'status', '--porcelain', '--untracked-files=all').stdout == ''
    assert _git(repo, 'log', '--pretty=%H').stdout == before


# ── one census request at a time: the history file is append-only ───────────
#
# A night whose deploy an earlier run RETAINED (its cancel refused, or the
# orchestrator down at a stop) skips rather than stacking a second request
# that would conflict on the append; a retained deploy that can no longer
# land is reclaimed and the night lands as usual.

_CENSUS_LABEL = 'memory-metadata-census'
_CANCEL_REFUSED_WHILE_LANDING = {
    'cancelled': False,
    'state': 'finalizing',
    'reason': 'Request is landing (its CAS is in progress)',
    'code': 'finalize_window',
}


def _retained_census_deploy(tmp_path, repo, monkeypatch):
    """The (branch, worktree) land_files kept for an earlier night's census
    request, still live at a passed ceiling with its cancel refused."""
    scratch = _scratch_dir(tmp_path)
    scratch.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, 'tempdir', str(scratch))
    queue = FakeMergeQueue(repo, states=['verifying'], cancel_reply=_CANCEL_REFUSED_WHILE_LANDING)
    with FakeEscalationMcp(tool_handlers=queue.handlers()) as server, \
            pytest.raises(LandingNotConfirmed):
        land_files(str(repo), {_HISTORY: b'{"an earlier night": true}\n'}, _main(repo),
                   port=str(server.port), label=_CENSUS_LABEL,
                   message='chore(census): an earlier night', timeout_secs=0.001)
    (submitted,) = queue.calls_to('merge_request')
    return f'task/{submitted["branch"]}', Path(submitted['worktree'])


def _branch_exists(repo, branch):
    return _git(repo, 'show-ref', '--verify', '--quiet', f'refs/heads/{branch}').returncode == 0


def test_a_retained_deploy_still_in_flight_skips_the_night(tmp_path, monkeypatch):
    repo = _git_repo_harness(tmp_path)
    branch, worktree = _retained_census_deploy(tmp_path, repo, monkeypatch)
    bare = branch.removeprefix('task/')

    with _census_queue(repo, states={bare: ['verifying']}) as (queue, _):
        before = _main(repo)
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)

    assert result.returncode == 0, _diag(result)
    assert 'commit=0 commit-step=skipped:landing-in-flight)' in _done_line(result.stdout), (
        _diag(result))
    assert branch in result.stderr, _diag(result)
    assert queue.calls_to('merge_request') == []
    assert _branch_exists(repo, branch) and worktree.is_dir()
    assert _main(repo) == before


def test_a_retained_deploy_that_settled_is_reclaimed_and_the_night_lands(tmp_path, monkeypatch):
    repo = _git_repo_harness(tmp_path)
    branch, worktree = _retained_census_deploy(tmp_path, repo, monkeypatch)
    bare = branch.removeprefix('task/')

    with _census_queue(repo, states={bare: ['abandoned'],
                                     FakeMergeQueue.REQUEST_ID: ['verifying', 'done']}) as (
            queue, _):
        result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_writes=_REGENERATED)

    assert _commit_step(result) == 'landed:mr-deploy0001', _diag(result)
    assert len(queue.calls_to('merge_request')) == 1
    assert not worktree.exists()
    _assert_left_nothing_behind(tmp_path, repo)
