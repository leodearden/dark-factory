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
import re
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


def test_no_cadence_knob_was_added_to_the_orchestrator_config():
    """Guards the design decision the whole §12 family shares: the cadence is
    the .timer's OnCalendar, NOT a dark-factory-orchestrator.yaml key. A knob
    there would be misplaced (the orchestrator process is not what is being
    scheduled) and would cost a fleet redeploy to change."""
    config = REPO_ROOT / 'dark-factory-orchestrator.yaml'
    if not config.is_file():
        return
    text = config.read_text()
    for token in ('coverage_census', 'coverage-census', 'census_memory_metadata',
                  'retro_stamp_topics'):
        assert token not in text, (
            f'{token!r} found in {config} — the cadence belongs in the .timer '
            f'unit, not in orchestrator config')


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


def _wrapper_harness(tmp_path, *, census_exit=0, stamp_exit=0, extra_env=None,
                     default_prefix=False):
    """Point both wrapper seams at fake recorders. Returns (env, state_path).

    With `default_prefix=True` the two `*_CMD` seams are left UNSET instead, so
    the wrapper's own `uv run --frozen --project <FM> python` prefix is the
    thing under test, with a fake `uv` recorder on PATH. Every other wrapper
    test overrides both seams and therefore never exercises that prefix at all.
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

    repo = tmp_path / 'fake-repo'
    (repo / 'fused-memory' / 'scripts').mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
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
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True, text=True, check=False,
    )


def _git_repo_harness(tmp_path, *, dirty=True):
    """Make $REPO a REAL git repo carrying the three tracked artifacts.

    The fake census recorder does not write artifacts, so the drift a live run
    would produce is staged here directly -- what is under test is the
    wrapper's commit step, not the census's rendering.
    """
    repo = tmp_path / 'fake-repo'
    (repo / 'plans').mkdir(parents=True, exist_ok=True)
    (repo / 'fused-memory' / 'scripts').mkdir(parents=True, exist_ok=True)

    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test')
    # No hooks, no signing -- this exercises the wrapper, not the repo's gates.
    _git(repo, 'config', 'commit.gpgsign', 'false')

    for rel in _ARTIFACTS:
        (repo / rel).write_text('{"baseline": true}\n')
    _git(repo, 'add', '--', *_ARTIFACTS)
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'baseline')

    if dirty:
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


# Forbidden git shapes, matched on the SUBCOMMAND after normalising away any
# number of `-C <dir>` options -- because EVERY git call in this wrapper is
# spelled `git -C "$REPO" <verb>`. The guard this replaces scanned for bare
# literals (`'git stash'`, `'git add -A'`, ...), so a real violation written in
# the file's own idiom contained none of them and most of the pattern set was
# unreachable: the scan passed regardless of what the wrapper did.
#
# Each entry pairs a pattern with synthetic violations, in the wrapper's own
# idiom, that it MUST match. Those synthetics are the load-bearing half -- a
# guard whose reachability is never exercised is what produced the finding, so
# reachability is now itself a test (see the self-check below).
_FORBIDDEN_GIT_PATTERNS = (
    (
        # `git stash` is banned in EVERY dark-factory checkout: refs/stash is a
        # SINGLE ref shared by every worktree and the merge worker's advance
        # path also consumes it (incident 13674d3c68). reset/checkout/clean in
        # the machine-operated main checkout would likewise act on -- and
        # destroy -- another process's state.
        r'\bgit\b(?:\s+-C\s+\S+)*\s+(?:stash|reset|checkout|clean)\b',
        (
            'git stash',
            'git -C "$REPO" stash',
            'git -C "$REPO" reset --hard',
            'git -C "$REPO" checkout main',
            'git -C "$REPO" clean -fd',
        ),
    ),
    (
        # Wholesale staging sweeps a concurrent process's WIP into our commit.
        r'\bgit\b(?:\s+-C\s+\S+)*\s+add\s+(?:-A|--all|\.)(?:\s|$)',
        (
            'git add -A',
            'git add .',
            'git -C "$REPO" add -A',
            'git -C "$REPO" add --all',
            'git -C "$REPO" add . && echo done',
        ),
    ),
    (
        # `commit -a`/`--all` is the un-scoped commit that `--only` exists to
        # avoid; it stages every tracked modification, ours or not.
        r'\bgit\b(?:\s+-C\s+\S+)*\s+commit\b[^\n]*\s(?:-a|--all)\b',
        (
            'git commit -a -m x',
            'git -C "$REPO" commit -a -m x',
            'git -C "$REPO" commit --all -m x',
        ),
    ),
)

# The wrapper's own legitimate git calls. None may match any pattern above --
# notably the scoped `git -C "$REPO" add -- "${ARTIFACTS[@]}"`, whose `--` is
# neither `-A`, `--all` nor `.`.
_LEGITIMATE_GIT_CALLS = (
    'git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1',
    'git -C "$REPO" status --porcelain -- "${ARTIFACTS[@]}" 2>/dev/null',
    'git -C "$REPO" commit --only "${ARTIFACTS[@]}" -m "$commit_msg"',
    'git -C "$REPO" add -- "${ARTIFACTS[@]}"',
)


def _wrapper_code() -> str:
    r"""The wrapper's EXECUTABLE lines, with comments stripped.

    BOTH full-line and TRAILING comments, and trailing matters in both
    directions. The header documents precisely which verbs are forbidden and
    why, so a check that could not tell prose from code would forbid the wrapper
    from explaining itself -- but leaving trailing comments in lets a `# never
    git stash here` note appended to a real command FALSE-POSITIVE the scan.

    KNOWN LIMIT of `\s#`: a `#` preceded by whitespace INSIDE a double-quoted
    string would truncate that line early. No such string exists in this wrapper
    today (verified: no non-comment line matches `\s#` at all, so the trailing
    strip is currently a no-op), and the anti-vacuity guard in the scan below
    fails loudly if one is ever introduced and takes the scan body with it.
    """
    return '\n'.join(
        re.split(r'\s#', line, maxsplit=1)[0]
        for line in WRAPPER.read_text().splitlines()
        if not line.lstrip().startswith('#')
    )


def test_the_forbidden_git_patterns_can_actually_fire():
    """The guard below is only worth its name if its patterns CAN match.

    This is the finding restated as a test: the previous guard's literals could
    never fire against this wrapper's `git -C "$REPO" <verb>` idiom, so it was
    green by construction. Every pattern must match a violation written the way
    this file writes git, and none may match the wrapper's legitimate calls.
    """
    for pattern, synthetics in _FORBIDDEN_GIT_PATTERNS:
        for synthetic in synthetics:
            assert re.search(pattern, synthetic), (
                f'{pattern!r} does not match its own synthetic violation '
                f'{synthetic!r} — the guard cannot fire')
        for legitimate in _LEGITIMATE_GIT_CALLS:
            assert not re.search(pattern, legitimate), (
                f'{pattern!r} false-positives on the wrapper\'s legitimate '
                f'{legitimate!r}')


def test_wrapper_never_reaches_for_the_forbidden_git_verbs():
    """The guard on CLAUDE.md's hardest prohibition, applied to a script that
    commits UNATTENDED into the machine-operated project_root checkout."""
    code = _wrapper_code()

    # ANTI-VACUITY. An empty or over-trimmed scan body satisfies every
    # "does not match" assertion below, so the guard would pass while checking
    # nothing. Measured floor: 6 `git ` and 5 `git -C` invocations survive today.
    assert code.count('git ') >= 5, (
        f'only {code.count("git ")} `git ` invocations survived comment '
        f'stripping — the scan body was over-trimmed and would pass vacuously')
    assert 'commit --only' in code, (
        'the scoped-commit form is the whole point; a bare `git commit` would '
        'sweep up a concurrent process staged work')

    for pattern, _ in _FORBIDDEN_GIT_PATTERNS:
        found = re.search(pattern, code)
        if found is not None:
            raise AssertionError(
                f'{found.group(0)!r} must never appear in a wrapper that runs '
                f'unattended against the machine-operated main checkout')
