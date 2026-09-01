"""Tests for scripts/install-memory-metadata-coverage-census-timer.sh, the two
systemd unit files it installs, and the wrapper their ExecStart names (task
4006).

Drives the installer via subprocess with a FAKE `systemctl` shimmed onto PATH
(records every invocation, minus `--user`, into a shared JSON state file) --
mirroring test_install_reify_closure_staleness_sweep_timer.py. Real systemd is
never touched.

The wrapper half is driven with BOTH `*_CMD` seams pointed at fake recorder
executables, mirroring test_reify_closure_staleness_sweep_wrapper.py, so the
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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = (Path(__file__).parent.parent
          / 'install-memory-metadata-coverage-census-timer.sh')
TEMPLATES_DIR = Path(__file__).parent.parent
REPO_ROOT = TEMPLATES_DIR.parent

# The units name the PRODUCTION checkout absolutely, since systemd resolves
# ExecStart/WorkingDirectory absolutely and the installed unit is a byte copy
# of the committed one -- so these assertions must not be derived from
# REPO_ROOT, which is a .worktrees/<id> path when the suite runs in a lane.
PRODUCTION_ROOT = '/home/leo/src/dark-factory'

SERVICE_NAME = 'memory-metadata-coverage-census.service'
TIMER_NAME = 'memory-metadata-coverage-census.timer'
WRAPPER = TEMPLATES_DIR / 'memory-metadata-coverage-census.sh'


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing install-memory-metadata-coverage-census-timer.sh.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE. `enable --now <unit>` marks each non-flag arg as an
enabled unit; `start <unit>` is recorded and always succeeds; `list-timers`
echoes one line per *.timer enabled so far THIS RUN, unless
FAKE_SYSTEMCTL_OMIT_LIST_TIMERS=1 -- simulating the self-verify failure where
`enable` nominally succeeded but the unit is absent from `list-timers`.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(args)

    if verb == "daemon-reload":
        _save(state)
        return 0

    if verb == "enable":
        units = [a for a in rest if not a.startswith("-")]
        enabled = state.setdefault("enabled_timers", [])
        for u in units:
            if u not in enabled:
                enabled.append(u)
        _save(state)
        return 0

    if verb == "start":
        _save(state)
        return 0

    if verb == "list-timers":
        _save(state)
        if os.environ.get("FAKE_SYSTEMCTL_OMIT_LIST_TIMERS") == "1":
            print("0 timers listed.")
            return 0
        enabled = state.get("enabled_timers", [])
        for unit in enabled:
            service = unit.replace(".timer", ".service")
            print(f"Mon 2026-08-17 05:00:00 UTC 8h left n/a n/a {unit} {service}")
        print(f"{len(enabled)} timers listed.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _fake_systemctl(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / 'systemctl'
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / 'systemctl_state.json'
    state_path.write_text(json.dumps({'calls': [], 'enabled_timers': []}))
    return bin_dir, state_path


def _systemctl_calls(tmp_path):
    state_path = tmp_path / 'systemctl_state.json'
    if not state_path.is_file():
        return []
    return json.loads(state_path.read_text())['calls']


def _run_script(tmp_path, *, env=None, reset_state=True):
    bin_dir, state_path = _fake_systemctl(tmp_path) if reset_state else (
        tmp_path / 'bin', tmp_path / 'systemctl_state.json')

    full_env = dict(os.environ)
    full_env['PATH'] = f'{bin_dir}{os.pathsep}{full_env["PATH"]}'
    full_env['FAKE_SYSTEMCTL_STATE'] = str(state_path)
    if env:
        full_env.update(env)
    return subprocess.run(
        ['bash', str(SCRIPT)],
        env=full_env, capture_output=True, text=True, timeout=60,
    )


# ── the installer ───────────────────────────────────────────────────────────


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); run: chmod +x {SCRIPT}')


def test_install_copies_both_units_and_enables_the_timer(tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(xdg_config)})
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        installed = unit_dir / name
        assert installed.is_file(), f'Expected {installed} to exist after install'
        assert installed.read_bytes() == (TEMPLATES_DIR / name).read_bytes()

    calls = _systemctl_calls(tmp_path)
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls


def test_install_does_not_kick_an_immediate_run(tmp_path):
    """No surprise run at install time.

    Unlike the flag-marker and reclaim installers, this one never `start`s the
    service. The census is a paginated full scroll of both live collections and
    it APPENDS a row to the committed trend history; an install-time firing
    would put an unreviewed, off-cadence row into the series the nightly job
    owns. An operator who wants a look now runs the wrapper by hand, or the
    census with `--no-history`.
    """
    result = _run_script(tmp_path, env={'XDG_CONFIG_HOME': str(tmp_path / 'xdg')})
    assert result.returncode == 0, result.stderr
    for call in _systemctl_calls(tmp_path):
        assert call[:1] != ['start'], f'unexpected immediate run: {call!r}'


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(tmp_path):
    """The self-verify catches "enable nominally succeeded but the unit is
    absent" -- the case a bare `enable` exit code cannot distinguish."""
    result = _run_script(tmp_path, env={
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg'),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })
    assert result.returncode != 0, (
        f'Expected non-zero on self-verify failure; stdout={result.stdout!r} '
        f'stderr={result.stderr!r}')
    assert TIMER_NAME in result.stderr, result.stderr


def test_install_is_idempotent(tmp_path):
    xdg_config = tmp_path / 'xdg-config'
    env = {'XDG_CONFIG_HOME': str(xdg_config)}
    first = _run_script(tmp_path, env=env)
    second = _run_script(tmp_path, env=env, reset_state=False)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    unit_dir = xdg_config / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        assert (unit_dir / name).read_bytes() == (TEMPLATES_DIR / name).read_bytes()


# ── the committed unit files ────────────────────────────────────────────────


def _directives(name) -> dict[str, list[tuple[str, str]]]:
    """Parse a systemd unit into `{section: [(key, value), ...]}`.

    WHY THIS EXISTS. A raw-substring check on a unit's text cannot tell a live
    DIRECTIVE from the COMMENT that explains it -- and these units comment
    heavily by design. The demonstrated case is `Persistent=true`, which appears
    BOTH in prose at memory-metadata-coverage-census.timer:20 and as the real
    directive at :27, so deleting the directive left the substring assertion
    GREEN. That is the class of finding this closes.

    SCOPED HONESTLY (esc-4006-7). An earlier draft generalized this to "every
    literal these tests care about also appears in prose" and cited
    `census_memory_metadata.py` at memory-metadata-coverage-census.service:7 as a
    second instance. That was measured FALSE and is retracted: the filename
    occurs exactly ONCE in the service unit, at the real `Documentation=` line,
    and the comment at :7 never contains it -- so for that directive the old
    substring form did catch an outright deletion. Routing it through the parser
    anyway is a STRENGTHENING, not a fix; see
    `test_service_documents_where_the_normative_contract_lives` for the
    comment-out edit on which the two forms genuinely diverge.

    Every unit assertion in this file routes through here regardless, so the
    weaker and stronger cases are not left to be told apart by eye; no new
    raw-substring pins.

    WHY IT IS HAND-ROLLED. Duplicate keys are preserved as separate pairs, never
    collapsed -- the service legitimately carries THREE `Documentation=` lines,
    which `configparser` would collapse or reject even with `strict=False`.

    WHY NOT `systemd-analyze verify`. It IS available here (systemd 255) and was
    deliberately not used: it cannot assert that a specific directive is SET,
    and it would make this suite depend on systemd being installed in every
    container/CI runner.

    Accepts a unit filename (resolved under TEMPLATES_DIR) or a Path, so sibling
    `*.timer` units can be parsed for the collision check.
    """
    path = name if isinstance(name, Path) else TEMPLATES_DIR / name
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ''
    for raw in path.read_text().splitlines():
        line = raw.strip()
        # FULL-LINE comments only: systemd treats `#`/`;` as a comment lead-in at
        # the start of a line, and a directive's value may legitimately contain
        # either character (a Documentation= URL fragment, for instance).
        if not line or line[0] in '#;':
            continue
        if line.startswith('[') and line.endswith(']'):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        sections.setdefault(current, []).append((key.strip(), value.strip()))
    return sections


def _values(directives, section, key) -> list[str]:
    """Every value declared for `key` under `[section]`, in file order.

    A list, not a scalar: asserting `== ['x']` pins both the value AND that it
    is declared exactly once, so a second stray `OnCalendar=` (which systemd
    reads as an ADDITIONAL firing) cannot slip in unnoticed.
    """
    return [v for k, v in directives.get(section, []) if k == key]


def test_timer_fires_at_the_next_free_nightly_slot():
    """05:00, after 03:00 legibility-trickle, 03:30 flag-marker-sweep, the
    already-double-booked 04:00 (reclaim-orphaned-worktrees +
    legibility-transcript-check) and 04:30 reify-closure-staleness-sweep. The
    stagger is deliberate: these jobs all touch the same machine and, in
    several cases, the same backing stores -- this one scrolls every point in
    both live Qdrant collections."""
    assert _values(_directives(TIMER_NAME), 'Timer', 'OnCalendar') == [
        '*-*-* 05:00:00']


def test_timer_does_not_collide_with_an_occupied_slot():
    """Guards the ladder itself, not just this unit's own literal: a future
    edit that re-cadences this job onto a taken slot fails here rather than
    silently double-booking a third job.

    Parsed on BOTH sides, so a commented-out slot in a sibling unit can neither
    manufacture a phantom collision nor mask a real one.
    """
    ours = set(_values(_directives(TIMER_NAME), 'Timer', 'OnCalendar'))
    assert ours, 'the timer declares no OnCalendar at all'
    for other in sorted(TEMPLATES_DIR.glob('*.timer')):
        if other.name == TIMER_NAME:
            continue
        clash = ours & set(_values(_directives(other), 'Timer', 'OnCalendar'))
        if clash:
            raise AssertionError(
                f'{TIMER_NAME} shares {sorted(clash)!r} with {other.name} — '
                f'pick a free slot and update the OPERATIONS.md §12 ladder '
                f'table')


def test_timer_catches_up_a_missed_night_and_avoids_a_thundering_herd():
    """A silently skipped night is a hole in the series that no later run can
    reconstruct -- this timer's output is a TREND. `Persistent=true` is what
    makes a night missed to a sleeping laptop get caught up on next login."""
    timer = _directives(TIMER_NAME)
    assert _values(timer, 'Timer', 'Persistent') == ['true']
    assert _values(timer, 'Timer', 'RandomizedDelaySec') == ['300']


def test_timer_is_installed_into_timers_target():
    assert _values(_directives(TIMER_NAME), 'Install', 'WantedBy') == [
        'timers.target']


def test_service_is_a_thin_oneshot_around_the_committed_wrapper():
    """Paths are the PRODUCTION checkout's, not this test run's.

    systemd resolves ExecStart absolutely and the installed unit is a byte copy
    of the committed one, so the unit must name /home/leo/src/dark-factory even
    when these tests run from a worktree under .worktrees/.
    """
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'Type') == ['oneshot']
    assert _values(service, 'Service', 'ExecStart') == [
        f'{PRODUCTION_ROOT}/scripts/memory-metadata-coverage-census.sh']
    assert _values(service, 'Service', 'WorkingDirectory') == [PRODUCTION_ROOT]


def test_service_sends_both_streams_to_the_journal():
    """The report is written to files, but the run's narration -- and any
    shortfall the census exits 1 on -- is only ever readable in the journal."""
    service = _directives(SERVICE_NAME)
    assert _values(service, 'Service', 'StandardOutput') == ['journal']
    assert _values(service, 'Service', 'StandardError') == ['journal']


def test_service_execstart_points_at_a_real_executable_wrapper():
    """The wrapper named by ExecStart exists and is executable.

    Checked against THIS checkout (the production path's scripts/ tail mapped
    onto the tree under test) so the assertion is meaningful from a worktree
    too -- what it pins is that the unit does not name a wrapper that was
    renamed or never committed.
    """
    named, = _values(_directives(SERVICE_NAME), 'Service', 'ExecStart')
    assert named.startswith(f'{PRODUCTION_ROOT}/scripts/'), named
    here = TEMPLATES_DIR / named.split('/scripts/', 1)[1]
    assert here.is_file(), here
    assert os.access(here, os.X_OK), here


def test_service_documents_where_the_normative_contract_lives():
    """The census script and the PRD are normative; the unit points a reader at
    them rather than restating any figure that would drift.

    Asserted as parsed `[Unit] Documentation=` VALUES rather than as a substring
    of the file text. CORRECTION (esc-4006-7): an earlier draft of this docstring
    claimed the bare filename "also appears in the unit's own comment at :7, so
    deleting the real directive left a substring check green". That was measured
    FALSE and is retracted -- `census_memory_metadata.py` occurs exactly ONCE in
    the unit, at the real `Documentation=` line, so the old substring form did
    catch an outright deletion.

    The parsed form is kept as a STRENGTHENING, not a fix, and this is what it
    buys: under the plausible comment-out edit (the directive prefixed with `#`
    rather than removed), the filename is still present in the file text, so a
    substring check stays green while the unit documents nothing. The parsed form
    fails, correctly. Do not "simplify" this back to `in text` on the grounds
    that the two forms are equivalent -- they diverge on exactly that edit.

    Both entries are checked, which additionally exercises the parser's
    duplicate-key preservation -- a dict-collapse would keep only the last of the
    three.
    """
    docs = _values(_directives(SERVICE_NAME), 'Unit', 'Documentation')
    assert docs, 'the unit points a reader at nothing'
    assert any(d.endswith('/census_memory_metadata.py') for d in docs), docs
    assert any(d.endswith('/memory-metadata-vocabulary.md') for d in docs), docs


# DELIBERATELY NOT TESTED HERE (review of 2026-08-16): that no cadence knob
# was added to `dark-factory-orchestrator.yaml`.  That test grepped another
# file's raw TEXT for four tokens, which is not a behaviour this code can
# regress on -- and worse, it opened with a bare `if not config.is_file():
# return`, so any checkout layout that lacks the file (a worktree, a
# differently-rooted run) reported it as a PASSING guard while it inspected
# nothing.  A guard that cannot distinguish "verified" from "never looked" is
# worse than no guard.  The design decision it meant to protect -- the cadence
# lives in the .timer's `OnCalendar`, not in orchestrator config -- is asserted
# directly and positively by the OnCalendar tests in this file.


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
"""
import json
import os
import sys

SERVICE_ENV_KEYS = ("CONFIG_PATH", "PROJECT_ROOT", "FALKORDB_URI")

who = os.environ["FAKE_WHO"]
state_path = os.environ["FAKE_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.append({
    "who": who,
    "argv": sys.argv[1:],
    "env": {k: os.environ.get(k) for k in SERVICE_ENV_KEYS},
})
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


def _wrapper_harness(tmp_path, *, census_exit=0, stamp_exit=0, extra_env=None,
                     default_prefix=False, record_git=False,
                     pad_commit_output=0):
    """Point both wrapper seams at fake recorders. Returns (env, state_path).

    With `default_prefix=True` the two `*_CMD` seams are left UNSET instead, so
    the wrapper's own `uv run --frozen --project <FM> python` prefix is the
    thing under test, with a fake `uv` recorder on PATH. Every other wrapper
    test overrides both seams and therefore never exercises that prefix at all.

    `pad_commit_output` (bytes, default 0/off) is an opt-in knob on the
    `record_git=True` shim: when set, the shim's handling of a `commit`
    invocation captures the REAL git's combined output, replays it VERBATIM
    first, then appends this many padding bytes, and exits with git's REAL
    status. git's own behaviour never changes -- only the volume the wrapper
    must read past does. This is what turns the wrapper's SIGPIPE-under-
    pipefail misread (see the retry-predicate tests below) from a ~0.6% race
    at git's natural output size into a deterministic reproduction.
    """
    bin_dir = tmp_path / 'wbin'
    bin_dir.mkdir(exist_ok=True)
    state_path = tmp_path / 'wrapper_state.json'
    state_path.write_text('[]')

    # The interpreter is spelled ABSOLUTELY (sys.executable), never `python3`,
    # so a PATH-resolved shim can never re-enter itself.
    shims = {'fake-census': 'CENSUS', 'fake-stamp': 'STAMP'}
    if default_prefix:
        # ONE fake `uv` serves BOTH halves, so it cannot key on FAKE_WHO: every
        # call lands under the same `who`, and the census's is simply the first.
        shims['uv'] = 'UV'
    for name, who in shims.items():
        shim = bin_dir / name
        shim.write_text(
            '#!/usr/bin/env bash\n'
            f'FAKE_WHO={who} exec {sys.executable} '
            f'{bin_dir / "recorder.py"} "$@"\n'
        )
        shim.chmod(0o755)
    (bin_dir / 'recorder.py').write_text(_FAKE_RECORDER_SRC)

    if record_git:
        # A `git` shim that RECORDS the real argv and then delegates to the
        # real binary, so the wrapper's git behaviour is unchanged and the
        # commit half still genuinely commits. The real git is resolved
        # ABSOLUTELY here -- while bin_dir is not yet on PATH -- so the shim
        # can never re-enter itself.
        real_git = shutil.which('git')
        assert real_git, 'no real git on PATH to delegate to'
        git_log = _git_argv_path(tmp_path)
        git_log.write_text('')
        shim = bin_dir / 'git'
        record_snippet = (
            f'{sys.executable} -c '
            '\'import json,sys; open(sys.argv[1],"a").write('
            'json.dumps(sys.argv[2:])+"\\n")\' '
            f'{git_log} "$@"\n'
        )
        if pad_commit_output:
            # Only `commit` invocations are padded. Skip any leading `-C
            # <dir>` pair(s) before checking the subcommand, mirroring
            # _forbidden_reason's own argv walk below, so this cannot be
            # fooled by option placement either. Non-commit calls (rev-parse,
            # status, the scoped add --) fall through to the plain `exec`
            # branch, untouched.
            shim.write_text(
                '#!/usr/bin/env bash\n'
                + record_snippet +
                'rest=("$@")\n'
                'while [ "${rest[0]:-}" = "-C" ] && [ "${#rest[@]}" -ge 2 ]; do\n'
                '    rest=("${rest[@]:2}")\n'
                'done\n'
                'if [ "${rest[0]:-}" = "commit" ]; then\n'
                f'    out="$({real_git} "$@" 2>&1)"\n'
                '    rc=$?\n'
                '    printf \'%s\' "$out"\n'
                f'    head -c {pad_commit_output} /dev/zero | tr \'\\0\' x\n'
                '    exit "$rc"\n'
                'fi\n'
                f'exec {real_git} "$@"\n'
            )
        else:
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
    if default_prefix:
        for key in ('COVERAGE_CENSUS_CMD', 'RETRO_STAMP_CMD'):
            env.pop(key, None)
        # Keeps the commit half out of the prefix assertion.
        env['CENSUS_COMMIT'] = '0'
    else:
        env['COVERAGE_CENSUS_CMD'] = 'fake-census'
        env['RETRO_STAMP_CMD'] = 'fake-stamp'
    env['FAKE_STATE'] = str(state_path)
    env['FAKE_EXIT_CENSUS'] = str(census_exit)
    env['FAKE_EXIT_STAMP'] = str(stamp_exit)
    if extra_env:
        env.update(extra_env)
    return env, state_path


def _run_wrapper(tmp_path, **kwargs):
    env, state_path = _wrapper_harness(tmp_path, **kwargs)
    result = subprocess.run(
        ['bash', str(WRAPPER)],
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


# ── the wrapper commits what it regenerates (esc-4006-5) ────────────────────
#
# The census rewrites three GIT-TRACKED files under $REPO/plans/, and $REPO
# defaults to the machine-operated project_root checkout. Leaving them dirty
# every morning is not untidiness: memory-metadata-coverage-history.json is
# APPEND-ONLY and IS the trend this task exists to build, so an uncommitted
# append that the merge worker's advance path resets away takes that night's
# row with it -- permanently, and with no error anywhere.
#
# Precedent for committing rather than not: scripts/legibility/census.py's
# _build_default_commit already commits docs/legibility/census-state.json from
# the 03:00 job, and scripts/legibility/nightly.py::_git_commit_docs_only does
# the same, both via scoped `git commit --only`.

_ARTIFACTS = (
    'plans/memory-metadata-census-report.json',
    'plans/memory-metadata-census-report.md',
    'plans/memory-metadata-coverage-history.json',
)


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


def _git_repo_harness(tmp_path, *, dirty=True, tracked=True):
    """Make $REPO a REAL git repo carrying the three artifacts.

    The fake census recorder does not write artifacts, so the drift a live run
    would produce is staged here directly -- what is under test is the
    wrapper's commit step, not the census's rendering.

    ``tracked=False`` models the FIRST-EVER run: the artifacts exist on disk
    but git has never seen them, which is the only state that reaches the
    wrapper's `did not match any file` retry branch.
    """
    repo = tmp_path / 'fake-repo'
    (repo / 'plans').mkdir(parents=True, exist_ok=True)
    (repo / 'fused-memory' / 'scripts').mkdir(parents=True, exist_ok=True)

    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test')
    # No hooks, no signing -- this exercises the wrapper, not the repo's gates.
    _git(repo, 'config', 'commit.gpgsign', 'false')

    if tracked:
        for rel in _ARTIFACTS:
            (repo / rel).write_text('{"baseline": true}\n')
        _git(repo, 'add', '--', *_ARTIFACTS)
        _git(repo, 'commit', '-q', '--no-verify', '-m', 'baseline')
        if dirty:
            for rel in _ARTIFACTS:
                (repo / rel).write_text('{"regenerated": true}\n')
        return repo

    # An untracked-artifacts repo still needs a root commit, so that `git
    # commit --only <untracked path>` fails the way a real first run fails
    # rather than the way an empty repo does.
    (repo / 'README.md').write_text('unrelated\n')
    _git(repo, 'add', '--', 'README.md')
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'baseline')
    for rel in _ARTIFACTS:
        (repo / rel).write_text('{"regenerated": true}\n')
    return repo


def _run_wrapper_in_git_repo(tmp_path, repo, **kwargs):
    env, state_path = _wrapper_harness(tmp_path, **kwargs)
    env['REPO'] = str(repo)
    result = subprocess.run(
        ['bash', str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    return result, json.loads(state_path.read_text())


def test_wrapper_commits_the_artifacts_it_regenerated(tmp_path):
    """THE point of the fix. The history file is append-only and IS the trend;
    an append left uncommitted in the machine-operated main checkout can be
    reset away by the merge worker, silently losing that night's row."""
    repo = _git_repo_harness(tmp_path)
    result, _ = _run_wrapper_in_git_repo(tmp_path, repo)

    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    status = _git(repo, 'status', '--porcelain', '--', *_ARTIFACTS).stdout
    assert status.strip() == '', (
        f'artifacts left dirty after the run: {status!r}')
    subject = _git(repo, 'log', '-1', '--pretty=%s').stdout.strip()
    assert subject != 'baseline', 'no commit was made'


def test_wrapper_commits_artifacts_git_has_never_tracked(tmp_path):
    """The FIRST-EVER run, and the one path no other test reaches.

    `git commit --only <path>` fails outright for a path git has never seen
    ("did not match any file(s) known to git"), so the wrapper stages those
    paths scoped and retries once. Every other git test here seeds the
    artifacts as already-tracked, so the retry branch never executed and a
    typo in it -- or a future change to git's message -- would silently turn
    the very first night into a narrated non-commit, losing the baseline row
    the whole commit step exists to protect.
    """
    repo = _git_repo_harness(tmp_path, tracked=False)
    result, _ = _run_wrapper_in_git_repo(tmp_path, repo)

    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    # commit=0 in the narration, not merely "the script survived".
    assert 'commit=0' in result.stdout, result.stdout
    assert 'committed the regenerated artifacts' in result.stdout
    status = _git(repo, 'status', '--porcelain', '--', *_ARTIFACTS).stdout
    assert status.strip() == '', f'artifacts left untracked/dirty: {status!r}'
    # All three landed in ONE commit, and the scope held: README.md was
    # already committed, so the retry's `git add --` must not have swept
    # anything else in.
    files = _git(repo, 'show', '--name-only', '--pretty=', 'HEAD').stdout.split()
    assert sorted(files) == sorted(_ARTIFACTS), files


def test_the_untracked_retry_is_reached_only_because_the_plain_commit_fails(
        tmp_path):
    """Reachability, in the shape step-25 established for the git guard.

    Asserts the PRECONDITION the retry branch keys on actually holds for
    real git here: a scoped commit of a never-tracked path fails, and fails
    with the message the wrapper greps for. If git ever reworded it, this
    fails loudly instead of the wrapper silently not committing.
    """
    repo = _git_repo_harness(tmp_path, tracked=False)
    proc = subprocess.run(
        ['git', '-C', str(repo), 'commit', '--only', *_ARTIFACTS, '-m', 'probe'],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    assert 'did not match any file' in (proc.stdout + proc.stderr).lower()


# Trailing bytes appended AFTER git's real (verbatim) commit output, to force
# the retry predicate's SIGPIPE misread deterministically rather than at its
# natural ~0.6%-under-load rate.
#
# Mirrors tests/scripts/test_setup_host_probe_pipelines.py::BULK_BYTES
# (task 4204): that file's 30-trials-per-size measurement records 65536 as
# FLAKY at 26/30 while 262144 and above are 30/30. Do NOT lower this -- a
# too-small pad does not make the test flaky, it silently weakens its power to
# catch a reintroduced `printf | grep -q` pipeline, which is worse because it
# is invisible. (This task separately measured 200/200 at ~200KB on this box;
# 262144 is the repo's already-established, better-sampled floor.)
_PAD_COMMIT_OUTPUT_BYTES = 262144


def test_the_untracked_retry_is_decided_by_gits_message_not_its_volume(
        tmp_path):
    """The untracked-run retry decision must be read from git's MESSAGE, not
    manufactured by how much git printed.

    ROOT CAUSE. The retry predicate (wrapper lines ~199-201) is

        printf '%s' "$commit_out" | grep -qi 'did not match any file'

    under `set -uo pipefail` (wrapper line 68). `grep -q` exits the instant
    it matches, and the match is on line 1 of git's own error message; the
    `printf` writer can then die of SIGPIPE once `commit_out` exceeds the pipe
    buffer, `pipefail` promotes the pipeline's status to 141, and a TRUE
    predicate reads FALSE -- so the scoped `git add --` retry is skipped and
    the first-ever-run commit silently does not happen, even though the
    string the grep looks for was genuinely present in `commit_out`.

    MEASURED (this worktree, 2026-09-01, git 2.43.0, bash 5.2.21, fleet load
    avg ~127/32 cores): 25/4000 (0.6%) misses at git's natural ~270-byte
    message -- an intermittent flake, not a deterministic failure -- and
    200/200 once the payload passes the pipe buffer. This test buys the
    deterministic RED the second way: `pad_commit_output=
    _PAD_COMMIT_OUTPUT_BYTES` inflates ONLY the volume `commit_out` carries
    (git's real message is replayed verbatim first, and git's real exit
    status is preserved) -- never git's message or status -- so a 0.6% race
    becomes a certainty.

    Already-fixed siblings of this exact defect class, adopted here rather
    than rediscovered: scripts/setup-host.sh::_parity_verdict and
    scripts/legibility/install-trickle-timer.sh (task 3527, commit
    5653ccd4f5) -- both replaced `printf | grep -q` with bash's own `[[ ]]`
    for the identical SIGPIPE-under-pipefail reason.

    Asserts on the BRANCH THE WRAPPER TOOK -- the recorded argv, the
    narration, the resulting commit -- never on "did the internal pipeline
    return 141". Measured: that pipeline returns 141 at every payload size
    tried, including 4096 bytes; it is only the VERDICT drawn from it that
    races, so a bare exit-141 check would be green on a healthy wrapper too.
    """
    repo = _git_repo_harness(tmp_path, tracked=False)
    result, _ = _run_wrapper_in_git_repo(
        tmp_path, repo, record_git=True,
        pad_commit_output=_PAD_COMMIT_OUTPUT_BYTES,
    )
    recorded = [
        json.loads(line)
        for line in _git_argv_path(tmp_path).read_text().splitlines() if line
    ]

    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr[:400]!r}')
    # THE HONEST-FAILURE ASSERTION, checked FIRST -- same ordering as the
    # honest-failure assertion in test_wrapper_never_invokes_a_forbidden_git_verb
    # below (commit-step confirmed before argv is trusted). On the regression
    # this test guards against, `commit_out` is padded on EVERY commit
    # attempt, including the failed first one whose message the wrapper
    # echoes to stderr on the `failed` branch -- so a stderr TAIL slice on
    # that failure is a wall of padding, not a diagnostic (confirmed
    # empirically). The commit-step token names the branch directly instead.
    observed_step = _commit_step(result)
    assert observed_step == 'committed', (
        f'expected commit-step=committed, observed {observed_step!r}; '
        f'stdout={result.stdout!r} stderr={result.stderr[:400]!r}')
    # The scoped retry actually fired for THIS run -- the branch under test,
    # not merely "the script survived". Same substring idiom as the pooled
    # guard below (' add -- ' padded on both sides so a bare `add` verb never
    # matches), scoped to this one run's own recorded argv. Checked only once
    # the commit-step assertion above has confirmed this run actually
    # committed, so a decline and a missing substring can never again be
    # conflated into the same failure.
    flat = [' '.join(a) for a in recorded]
    assert any(' add -- ' in f'{c} ' for c in flat), (
        f'the scoped `git add --` retry never fired for this run: {flat!r}')
    assert 'committed the regenerated artifacts' in result.stdout, result.stdout
    assert 'commit=0' in result.stdout, result.stdout
    # All three landed in the one retried commit -- the retry stayed scoped.
    files = _git(repo, 'show', '--name-only', '--pretty=', 'HEAD').stdout.split()
    assert sorted(files) == sorted(_ARTIFACTS), files


def test_wrapper_commit_is_scoped_and_never_sweeps_unrelated_dirty_state(
        tmp_path):
    """CLAUDE.md's rule for this checkout: `git commit --only <paths>`, never a
    bare `git commit` and never `git add -A`. A concurrent process's WIP in the
    same tree must not be swept into the census's nightly commit."""
    repo = _git_repo_harness(tmp_path)
    bystander = repo / 'plans' / 'someone-elses-wip.md'
    bystander.write_text('concurrent work, not ours\n')
    tracked_bystander = repo / 'README.md'
    tracked_bystander.write_text('unrelated tracked edit\n')
    _git(repo, 'add', '--', 'README.md')
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'bystander baseline')
    tracked_bystander.write_text('unrelated tracked edit, now dirty\n')

    result, _ = _run_wrapper_in_git_repo(tmp_path, repo)
    assert result.returncode == 0, result.stderr

    committed = _git(repo, 'show', '--name-only', '--pretty=', 'HEAD').stdout
    assert 'someone-elses-wip.md' not in committed, committed
    assert 'README.md' not in committed, committed
    for rel in _ARTIFACTS:
        assert rel in committed, f'{rel} missing from {committed!r}'
    # And the bystanders are still exactly as the other process left them.
    assert bystander.exists()
    assert 'now dirty' in tracked_bystander.read_text()


def test_wrapper_treats_an_unchanged_artifact_set_as_a_no_op_not_a_failure(
        tmp_path):
    """A night whose corpus did not drift produces byte-identical artifacts.
    `git commit` reports "nothing to commit" with a NON-ZERO code; reading that
    as a fault would narrate a failure every quiet night."""
    repo = _git_repo_harness(tmp_path, dirty=False)
    before = _git(repo, 'rev-parse', 'HEAD').stdout.strip()

    result, _ = _run_wrapper_in_git_repo(tmp_path, repo)
    assert result.returncode == 0, result.stderr
    after = _git(repo, 'rev-parse', 'HEAD').stdout.strip()
    assert after == before, 'an empty commit was created'
    combined = result.stdout + result.stderr
    assert 'commit=0' in combined, (
        f'a no-drift night must not be narrated as a commit failure: '
        f'{combined!r}')


def test_wrapper_still_exits_zero_when_the_commit_cannot_happen(tmp_path):
    """Same oneshot contract as the other two halves: a commit failure is
    narrated, never propagated, or the timer wedges in `failed` state and the
    trend quietly ends. Exercised with $REPO not a git repo at all -- which is
    also the state every other wrapper test runs in."""
    result, calls = _run_wrapper(tmp_path)          # plain tmp dir, no .git
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    # Both halves still ran; the commit step aborted neither.
    assert [c['who'] for c in calls] == ['CENSUS', 'STAMP'], calls


def test_wrapper_commits_even_when_the_census_reported_a_shortfall(tmp_path):
    """The census exits 1 by design when `coverage.complete` is false, and the
    header is explicit that the artifacts still carry the evidence of the
    shortfall. That evidence is exactly what must reach the repo."""
    repo = _git_repo_harness(tmp_path)
    before = _git(repo, 'rev-parse', 'HEAD').stdout.strip()
    result, _ = _run_wrapper_in_git_repo(tmp_path, repo, census_exit=1)
    assert result.returncode == 0, result.stderr

    after = _git(repo, 'rev-parse', 'HEAD').stdout.strip()
    assert after != before, (
        'a shortfall night committed nothing — the evidence never reached '
        'the repo')
    committed = _git(repo, 'show', '--name-only', '--pretty=', 'HEAD').stdout
    for rel in _ARTIFACTS:
        assert rel in committed, f'{rel} missing from {committed!r}'


def test_wrapper_narrates_the_commit_outcome_alongside_the_other_two(tmp_path):
    """Exiting 0 must never mean the outcome is invisible -- the journal is the
    only place this job is readable."""
    repo = _git_repo_harness(tmp_path)
    result, _ = _run_wrapper_in_git_repo(tmp_path, repo)
    combined = result.stdout + result.stderr
    assert 'census=0' in combined, combined
    assert 'stamp=0' in combined, combined
    assert 'commit=0' in combined, combined


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


def test_the_wrapper_names_which_commit_step_branch_it_took(tmp_path):
    """Of the wrapper's five commit-step outcomes, FOUR narrate `commit=0` on
    the final done(...) line -- which reads as success while nothing was
    committed. In an unattended nightly whose journal is its only readable
    surface, "declined" and "committed" being indistinguishable in the
    summary line is exactly the silent-degradation shape CLAUDE.md's
    loud-over-silent norm warns about.

    Drives the REAL wrapper into all five branches with the harnesses already
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

    committed_repo = _git_repo_harness(tmp_path / 'committed')
    scenarios['committed'], _ = _run_wrapper_in_git_repo(
        tmp_path / 'committed', committed_repo)

    no_drift_repo = _git_repo_harness(tmp_path / 'no-drift', dirty=False)
    scenarios['skipped:no-drift'], _ = _run_wrapper_in_git_repo(
        tmp_path / 'no-drift', no_drift_repo)

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
# calls. Every one of the wrapper's five git call sites (rev-parse, status
# --porcelain, commit --only, the scoped `add --`, and the commit retry) is
# reached by the scenarios below, so this observes the whole git surface.
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
    # And the wrapper's real calls must never trip it -- notably the scoped
    # `add --`, whose `--` is neither -A, --all nor `.`.
    for argv in (
        ['-C', '/repo', 'rev-parse', '--git-dir'],
        ['-C', '/repo', 'status', '--porcelain', '--', 'a.json'],
        ['-C', '/repo', 'commit', '--only', 'a.json', '-m', 'msg'],
        ['-C', '/repo', 'add', '--', 'a.json'],
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

    committed = _stub(
        'memory-metadata-coverage-census: done (census=0 stamp=0 commit=0 '
        'commit-step=committed)\n')
    assert _commit_step(committed) == 'committed'

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
    """CLAUDE.md's hardest prohibition, on a script that commits UNATTENDED
    into the machine-operated project_root checkout.

    `git stash` is the sharpest of these: refs/stash is a SINGLE ref shared by
    every worktree, and the merge worker's advance path also consumes it
    (incident 13674d3c68), so a stash here can be popped out from under an
    unrelated process.

    Each scenario's evidence stays SEPARATE rather than pooled. Pooling used
    to let an environmental/precondition decline in one iteration hide behind
    a healthy run in another: the old anti-vacuity check only proved that a
    scoped commit and the add-- retry happened SOMEWHERE across all three
    runs, so a run whose retry silently declined (the printf|grep -q SIGPIPE
    misread pinned elsewhere in this file) could still pass as long as
    ANOTHER iteration produced that substring -- and if it ever did trip, the
    failure dumped a saferepr-truncated pooled list naming neither the
    iteration nor the branch (esc-3647-3). Each iteration now asserts its OWN
    commit-step branch and, only once that is confirmed, its OWN argv -- so a
    precondition decline fails HERE, AS a decline, naming which iteration and
    which branch, instead of surviving to a pooled check it was never meant
    to satisfy. The anti-vacuity INTENT is unchanged: the guard still proves
    the scoped-commit path and the add-- retry path were genuinely exercised.
    """
    scenarios = (
        # (label, _git_repo_harness kwargs, expected commit-step,
        #  a git-argv substring that scenario's OWN run must contain)
        ('already-tracked artifacts', {}, 'committed', 'commit --only'),
        ('first-ever run, untracked (exercises the add-- retry)',
         {'tracked': False}, 'committed', ' add -- '),
        ('a quiet night with no drift', {'dirty': False}, 'skipped:no-drift', None),
    )
    for label, kwargs, expected_step, expect_substring in scenarios:
        repo = _git_repo_harness(tmp_path / label.split()[0], **kwargs)
        env, _ = _wrapper_harness(tmp_path, record_git=True)
        env['REPO'] = str(repo)
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

        # THE HONEST-FAILURE ASSERTION. This is what actually flaked -- the
        # untracked iteration's retry silently skipped under the SIGPIPE
        # misread -- and it now fails HERE, naming the iteration, the branch
        # it was built for, and the branch it actually took, rather than
        # surviving to a pooled check that could not tell a decline in THIS
        # iteration from a healthy run in a different one.
        observed_step = _commit_step(result)
        assert observed_step == expected_step, (
            f'{label}: expected commit-step={expected_step!r}, observed '
            f'{observed_step!r}; recorded={recorded!r} '
            f'stderr={result.stderr[-2000:]!r}')

        if expect_substring is not None:
            # Only checked once the commit-step assertion above has already
            # confirmed this iteration actually committed -- so a missing
            # substring and a declined commit can never again be conflated
            # into the same failure.
            flat = [' '.join(a) for a in recorded]
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

    Carries the same tracked artifact paths as the real thing, so a redirected
    `commit --only` would SUCCEED here rather than failing on an unknown
    pathspec -- otherwise the guard could pass for the wrong reason.
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
    result, state = _run_wrapper_in_git_repo(
        tmp_path, repo, extra_env={'GIT_DIR': str(decoy / '.git')},
    )

    assert _decoy_state(decoy) == before, (
        'the wrapper wrote into a repository named only by an ambient GIT_DIR; '
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    # POSITIVE CONTROL: the run must have done its real work in the sandbox.
    # Without this the assertion above passes just as well when the wrapper
    # dies on line 1 and touches nothing anywhere.
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')
    assert [c['who'] for c in state] == ['CENSUS', 'STAMP'], state
    sandbox_status = _git(repo, 'status', '--porcelain', '--', *_ARTIFACTS).stdout
    assert sandbox_status.strip() == '', (
        f'the sandbox commit did not happen, so the guard proved nothing: '
        f'{sandbox_status!r} stdout={result.stdout!r}')


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
