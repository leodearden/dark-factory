"""Contract for ``orchestrator/scripts/warm-lane/lib_live_refs.sh``.

Task 8089.  The lib's verdict gates ``rm -rf <lane>/target`` and
``git worktree remove --force`` in ``warm-lane-gc.sh``, yet its behavioural
coverage there runs only on the offline lane.  These cases run at the merge
gate against the SHIPPED lib, never a copy:

* ``TestReferenceDetection`` pins the verdicts against real helper processes:
  a cwd hit, an open-fd hit, an mmap hit, a miss, the ``_lane-1``/``_lane-10``
  name-prefix boundary and the batch subset.
* ``TestScanFootprint`` pins what a negative verdict may read.  The verdict
  needs only ``/proc/<pid>/cwd``, ``/proc/<pid>/fd/*`` and ``/proc/<pid>/maps``,
  so the only directories it may list are ``/proc`` and ``/proc/<pid>/fd``.  It
  observes the scan's ``getdents64`` calls under strace: a structural property,
  so it carries no wall-clock bound to flake under host load.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO

import pytest

# Reused from the sibling shipped-script contract, as test_lane_state_lib.py
# does: the script dir is resolved from __file__, and the env drops the keys
# that would aim the scripts at the ambient checkout.
from test_warm_lane_scripts_shipped import (
    WARM_LANE_SCRIPT_DIR,
    _sanitized_env,
)

LIB = WARM_LANE_SCRIPT_DIR / 'lib_live_refs.sh'

_BASH = shutil.which('bash') or '/bin/bash'

#: Anti-hang backstop for one sourced run, not a timing assertion.
_RUN_TIMEOUT_S = 120

_VERDICT_LINE = re.compile(r'^rc=(\d+)$', re.M)

_FD_DIR = re.compile(r'/proc/\d+/fd')

#: ``mmap.mmap`` keeps a private dup of the fd it maps, so every fd on the file
#: is closed afterwards: the mapping is then the helper's only reference to it.
_MAP_THEN_CLOSE_FDS = """
import mmap, os, sys, time
path = sys.argv[1]
with open(path, 'rb') as mapped:
    mapping = mmap.mmap(mapped.fileno(), 0, prot=mmap.PROT_READ)
for fd in os.listdir('/proc/self/fd'):
    try:
        if os.readlink(f'/proc/self/fd/{fd}') == path:
            os.close(int(fd))
    except OSError:
        pass
time.sleep(300)
"""


def _run_sourced(
    snippet: str, *, prefix: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Source the SHIPPED lib in a fresh bash, optionally under *prefix*."""
    script = f'source {shlex.quote(str(LIB))}\n{snippet}\n'
    return subprocess.run(
        [*prefix, _BASH, '-c', script],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
        cwd=str(WARM_LANE_SCRIPT_DIR),
        env=_sanitized_env(),
    )


def _verdict(proc: subprocess.CompletedProcess[str]) -> int:
    """The ``rc=N`` the snippet echoed; a missing line means the lib never ran."""
    found = _VERDICT_LINE.findall(proc.stdout)
    assert len(found) == 1, (
        f'expected exactly one rc= line; stdout={proc.stdout!r} '
        f'stderr={proc.stderr!r}'
    )
    return int(found[0])


def _live_ref_present(path: Path) -> int:
    return _verdict(
        _run_sourced(f'live_ref_present {shlex.quote(str(path))}; echo "rc=$?"')
    )


def _wait_until(predicate: Callable[[], bool], what: str, deadline_s: float = 30) -> None:
    """Poll *predicate* in 0.05s ticks.

    The deadline is a broken-infra backstop, not a timing assertion: the same
    shape as test_warm_lane_gc.sh::_wait_for_reader_lock.
    """
    deadline = time.monotonic() + deadline_s
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(f'gave up after {deadline_s}s waiting for {what}')
        time.sleep(0.05)


def _link_is(link: str, target: Path) -> Callable[[], bool]:
    def resolved() -> bool:
        try:
            return os.readlink(link) == str(target)
        except OSError:
            return False
    return resolved


def _mapped_with_no_fd_open(pid: int, path: Path) -> Callable[[], bool]:
    def resolved() -> bool:
        fd_dir = f'/proc/{pid}/fd'
        try:
            maps = Path(f'/proc/{pid}/maps').read_text()
            fd_targets = {os.readlink(f'{fd_dir}/{fd}') for fd in os.listdir(fd_dir)}
        except OSError:
            return False
        return str(path) in maps and str(path) not in fd_targets
    return resolved


Spawn = Callable[..., subprocess.Popen[bytes]]


@pytest.fixture
def helpers() -> Iterator[Spawn]:
    """Start long-lived helper processes; kill and reap every one after."""
    started: list[subprocess.Popen[bytes]] = []

    def spawn(
        *,
        cwd: Path,
        stdout: IO[bytes] | int = subprocess.DEVNULL,
        argv: tuple[str, ...] = ('sleep', '300'),
    ) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
        )
        started.append(proc)
        return proc

    yield spawn
    for proc in started:
        proc.kill()
    for proc in started:
        proc.wait(timeout=30)


def _spawn_with_cwd(helpers: Spawn, cwd: Path) -> subprocess.Popen[bytes]:
    proc = helpers(cwd=cwd)
    _wait_until(_link_is(f'/proc/{proc.pid}/cwd', cwd), f'helper cwd at {cwd}')
    return proc


class TestReferenceDetection:
    """The verdicts the scan's producer must not move."""

    def test_a_process_whose_cwd_is_inside_the_dir_is_live(
        self, tmp_path: Path, helpers: Spawn,
    ) -> None:
        lane = tmp_path.resolve() / '_lane-1'
        target = lane / 'target'
        target.mkdir(parents=True)
        _spawn_with_cwd(helpers, target)

        assert _live_ref_present(lane) == 0

    def test_a_process_holding_an_open_fd_inside_the_dir_is_live(
        self, tmp_path: Path, helpers: Spawn,
    ) -> None:
        root = tmp_path.resolve()
        lane = root / '_lane-1'
        (lane / 'target').mkdir(parents=True)
        log = lane / 'target' / 'build.log'
        elsewhere = root / 'elsewhere'
        elsewhere.mkdir()

        with log.open('wb') as handle:
            proc = helpers(cwd=elsewhere, stdout=handle)
        _wait_until(_link_is(f'/proc/{proc.pid}/fd/1', log), f'helper fd 1 on {log}')
        _wait_until(
            _link_is(f'/proc/{proc.pid}/cwd', elsewhere),
            f'helper cwd at {elsewhere}',
        )

        assert _live_ref_present(lane) == 0

    def test_a_process_mapping_a_file_inside_the_dir_is_live(
        self, tmp_path: Path, helpers: Spawn,
    ) -> None:
        root = tmp_path.resolve()
        lane = root / '_lane-1'
        mapped = lane / 'target' / 'libdep.so'
        mapped.parent.mkdir(parents=True)
        mapped.write_bytes(b'\0' * 4096)
        elsewhere = root / 'elsewhere'
        elsewhere.mkdir()

        proc = helpers(
            cwd=elsewhere,
            argv=(sys.executable, '-c', _MAP_THEN_CLOSE_FDS, str(mapped)),
        )
        _wait_until(
            _mapped_with_no_fd_open(proc.pid, mapped),
            f'helper mapping {mapped} with no fd open on it',
        )
        assert os.readlink(f'/proc/{proc.pid}/cwd') == str(elsewhere)

        assert _live_ref_present(lane) == 0

    def test_an_unreferenced_dir_is_not_live(self, tmp_path: Path) -> None:
        lane = tmp_path.resolve() / '_lane-1'
        lane.mkdir()

        assert _live_ref_present(lane) == 1

    def test_a_reference_under_a_longer_sibling_name_does_not_count(
        self, tmp_path: Path, helpers: Spawn,
    ) -> None:
        root = tmp_path.resolve()
        (root / '_lane-1').mkdir()
        sibling_target = root / '_lane-10' / 'target'
        sibling_target.mkdir(parents=True)
        _spawn_with_cwd(helpers, sibling_target)

        assert _live_ref_present(root / '_lane-1') == 1
        assert _live_ref_present(root / '_lane-10') == 0

    def test_the_batch_entry_point_prints_only_the_referenced_subset(
        self, tmp_path: Path, helpers: Spawn,
    ) -> None:
        referenced = tmp_path.resolve() / 'A'
        unreferenced = tmp_path.resolve() / 'B'
        referenced.mkdir()
        unreferenced.mkdir()
        _spawn_with_cwd(helpers, referenced)

        proc = _run_sourced(
            'live_referenced_paths '
            f'{shlex.quote(str(referenced))} {shlex.quote(str(unreferenced))}'
        )

        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert set(proc.stdout.splitlines()) == {str(referenced)}


class TestScanFootprint:
    """What a negative verdict, which runs both scan halves, may list."""

    def test_a_negative_verdict_lists_only_proc_and_per_process_fd_dirs(
        self, tmp_path: Path,
    ) -> None:
        strace = shutil.which('strace')
        if strace is None:
            pytest.skip('strace is an optional host tool and is not installed')

        unreferenced = tmp_path.resolve() / 'unreferenced'
        unreferenced.mkdir()
        trace_file = tmp_path / 'trace'
        proc = _run_sourced(
            f'live_ref_present {shlex.quote(str(unreferenced))}; echo "rc=$?"',
            prefix=(strace, '-f', '--seccomp-bpf', '-qq', '-e', 'trace=getdents64',
                    '-y', '-o', str(trace_file)),
        )

        trace = trace_file.read_text() if trace_file.exists() else ''
        ptrace_refused = re.search(
            r'^strace: .*(ptrace|PTRACE|Operation not permitted)', proc.stderr, re.M,
        )
        if not trace or ptrace_refused:
            pytest.fail(
                'strace is installed but could not trace: this host lacks the '
                f'ptrace capability the footprint check needs; stderr={proc.stderr!r}'
            )
        assert _verdict(proc) == 1

        listed = [
            d for d in re.findall(r'getdents64\(\d+<([^>]*)>', trace)
            if d == '/proc' or d.startswith('/proc/')
        ]
        assert '/proc' in listed and any(_FD_DIR.fullmatch(d) for d in listed), (
            'the trace shows no listing of /proc and /proc/<pid>/fd, so it '
            f'cannot speak for the scan; listed={sorted(set(listed))[:20]}'
        )

        offending = [d for d in listed if d != '/proc' and not _FD_DIR.fullmatch(d)]
        shapes = Counter(
            re.sub(r'/(proc|task)/\d+', r'/\1/<pid>', d) for d in offending
        )
        assert not offending, (
            'a liveness scan lists only /proc and /proc/<pid>/fd — its verdict '
            'reads /proc/<pid>/cwd, /proc/<pid>/fd/* and /proc/<pid>/maps, '
            f'nothing else; it listed {len(offending)} other directories, by '
            f'shape: {shapes.most_common(15)}'
        )
