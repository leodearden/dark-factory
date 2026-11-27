"""The audit-hook recorder B1 and B30 scope effects with: what it records, and what it leaves out."""

import glob
import json
import os
import site
import socket
import sqlite3
import ssl
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from _audit_hook import (
    Connected,
    DatabaseOpened,
    Listed,
    Mutated,
    Opened,
    Spawned,
    ThreadStarted,
    interpreter_roots,
    recording,
    runtime_roots,
)


def test_an_open_records_its_path_and_whether_it_writes(tmp_path: Path) -> None:
    target = tmp_path / 'a'

    with recording() as rec:
        target.write_text('x')
        target.read_text()

    assert rec.opens == (Opened(target, writes=True), Opened(target, writes=False))


def test_a_raw_os_open_for_writing_counts_as_a_write(tmp_path: Path) -> None:
    target = tmp_path / 'raw'

    with recording() as rec:
        os.close(os.open(target, os.O_WRONLY | os.O_CREAT, 0o600))

    assert rec.opens == (Opened(target, writes=True),)


def test_a_mkdir_and_a_replace_are_mutations_of_their_paths(tmp_path: Path) -> None:
    (tmp_path / 'x').write_text('x')

    with recording() as rec:
        os.mkdir(tmp_path / 'd')
        os.replace(tmp_path / 'x', tmp_path / 'y')

    assert rec.mutations == (
        Mutated('os.mkdir', (tmp_path / 'd',)),
        Mutated('os.rename', (tmp_path / 'x', tmp_path / 'y')),
    )


def test_a_connect_records_its_address() -> None:
    with socket.create_server(('127.0.0.1', 0)) as server:
        port = server.getsockname()[1]
        with recording() as rec, socket.socket() as client:
            client.connect(('127.0.0.1', port))

    assert rec.connects == (Connected(('127.0.0.1', port)),)


def test_a_subprocess_is_a_spawn() -> None:
    with recording() as rec:
        subprocess.run([sys.executable, '-c', 'pass'], check=True)

    assert 'subprocess.Popen' in {spawn.event for spawn in rec.spawns}


def test_a_thread_start_is_recorded_with_its_target() -> None:
    def names_this_thread() -> None:
        pass

    with recording() as rec:
        thread = threading.Thread(target=names_this_thread)
        thread.start()
        thread.join()

    assert len(rec.thread_starts) == 1
    (started,) = rec.thread_starts
    assert isinstance(started, ThreadStarted)
    assert 'names_this_thread' in started.target


def test_a_recording_with_no_thread_start_records_none(tmp_path: Path) -> None:
    with recording() as rec:
        (tmp_path / 'a').write_text('x')

    assert rec.thread_starts == ()


def test_opens_of_the_runtime_and_the_ca_bundle_are_left_out(tmp_path: Path) -> None:
    ca_bundles = [root for root in runtime_roots() if root.is_file()]
    assert ca_bundles, 'no CA bundle among the runtime roots'
    target = tmp_path / 'a'
    target.write_text('x')

    with recording() as rec:
        Path(json.__file__).read_bytes()
        ca_bundles[0].read_bytes()
        target.read_text()

    assert len(rec.opens) == 3
    assert rec.opens_outside_runtime() == (Opened(target, writes=False),)


def test_mutations_of_the_runtime_are_left_out(tmp_path: Path) -> None:
    runtime_cache = Path(json.__file__).with_name('no-such-package') / '__pycache__'

    with recording() as rec:
        with pytest.raises(FileNotFoundError):
            os.mkdir(runtime_cache)
        os.mkdir(tmp_path / 'd')

    assert len(rec.mutations) == 2
    assert rec.mutations_outside_runtime() == (Mutated('os.mkdir', (tmp_path / 'd',)),)


def test_a_mutation_of_the_runtime_outside_a_bytecode_cache_is_kept() -> None:
    runtime_data = Path(json.__file__).with_name('no-such-package') / 'data'

    with recording() as rec, pytest.raises(FileNotFoundError):
        os.mkdir(runtime_data)

    assert rec.mutations_outside_runtime() == (Mutated('os.mkdir', (runtime_data,)),)


def test_only_a_write_into_a_runtime_bytecode_cache_is_left_out_of_the_opens() -> None:
    absent = Path(json.__file__).with_name('no-such-package')
    cached, data = absent / '__pycache__' / 'a.cpython.pyc', absent / 'data.txt'

    with recording() as rec:
        for target in (cached, data):
            with pytest.raises(FileNotFoundError):
                target.write_bytes(b'x')

    assert rec.opens == (Opened(cached, writes=True), Opened(data, writes=True))
    assert rec.opens_outside_runtime() == (Opened(data, writes=True),)


def test_only_events_inside_the_block_are_recorded(tmp_path: Path) -> None:
    (tmp_path / 'before').write_text('x')

    with recording() as rec:
        (tmp_path / 'during').write_text('x')
    (tmp_path / 'after').write_text('x')

    assert [opened.path for opened in rec.opens] == [tmp_path / 'during']


def test_sequential_recordings_do_not_see_each_others_events(tmp_path: Path) -> None:
    with recording() as first:
        (tmp_path / 'one').write_text('x')
    with recording() as second:
        (tmp_path / 'two').write_text('x')

    assert [opened.path for opened in first.opens] == [tmp_path / 'one']
    assert [opened.path for opened in second.opens] == [tmp_path / 'two']


def test_a_nested_recording_sees_only_its_own_span(tmp_path: Path) -> None:
    with recording() as outer:
        (tmp_path / 'outer').write_text('x')
        with recording() as inner:
            (tmp_path / 'inner').write_text('x')

    assert [opened.path for opened in inner.opens] == [tmp_path / 'inner']
    assert [opened.path for opened in outer.opens] == [tmp_path / 'outer', tmp_path / 'inner']


def test_a_sqlite_connect_records_its_path(tmp_path: Path) -> None:
    database = tmp_path / 'a.db'

    with recording() as rec:
        sqlite3.connect(database).close()

    assert rec.database_opens == (DatabaseOpened(database),)


def test_a_read_only_uri_connect_records_the_decoded_path(tmp_path: Path) -> None:
    database = tmp_path / 'with space' / 'b.db'
    database.parent.mkdir()
    sqlite3.connect(database).close()

    with recording() as rec:
        sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True).close()

    assert rec.database_opens == (DatabaseOpened(database),)


def test_an_in_memory_connect_records_no_path() -> None:
    with recording() as rec:
        sqlite3.connect(':memory:').close()

    assert rec.database_opens == (DatabaseOpened(None),)


def test_a_subprocess_spawn_records_its_cwd_and_args(tmp_path: Path) -> None:
    with recording() as rec:
        subprocess.run(['git', '--version'], cwd=tmp_path, check=True, capture_output=True)

    popens = [spawn for spawn in rec.spawns if spawn.event == 'subprocess.Popen']
    assert popens == [Spawned('subprocess.Popen', 'git', cwd=tmp_path, args=('git', '--version'))]


def test_a_listing_records_its_directory_however_it_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'a.json').write_text('{}')
    monkeypatch.chdir(tmp_path)

    with recording() as rec:
        os.listdir()
        list(tmp_path.iterdir())
        list(tmp_path.glob('*.json'))
        glob.glob(str(tmp_path / '*.json'))

    assert rec.listings == (
        Listed('os.listdir', tmp_path),
        *(Listed('os.scandir', tmp_path) for _ in range(3)),
    )


def test_listings_outside_given_roots_keep_one_through_a_file_descriptor(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        with recording() as rec:
            os.listdir(Path(json.__file__).parent)
            os.listdir(tmp_path)
            os.listdir(descriptor)
    finally:
        os.close(descriptor)

    assert rec.listings_outside(interpreter_roots()) == (
        Listed('os.listdir', tmp_path), Listed('os.listdir', None),
    )


def _ca_bundles() -> set[Path]:
    paths = ssl.get_default_verify_paths()
    bundles = {Path(path) for path in (paths.cafile, paths.capath) if path}
    try:
        import certifi
    except ImportError:
        return bundles
    return {*bundles, Path(certifi.where())}


def _system_ca_file() -> Path:
    paths = ssl.get_default_verify_paths()
    if paths.cafile:
        return Path(paths.cafile)
    assert paths.capath, 'no system CA store'
    return next(path for path in sorted(Path(paths.capath).iterdir()) if path.is_file())


def test_the_interpreter_roots_are_the_prefixes_and_site_packages_without_a_ca_bundle() -> None:
    roots = interpreter_roots()

    assert Path(sys.prefix) in roots
    assert {Path(path) for path in site.getsitepackages()} <= set(roots)
    assert set(roots).isdisjoint(_ca_bundles())


def test_opens_outside_the_interpreter_roots_keep_a_ca_bundle_read() -> None:
    ca_file = _system_ca_file()

    with recording() as rec:
        Path(json.__file__).read_bytes()
        ca_file.read_bytes()

    assert rec.opens_outside_runtime() == ()
    assert rec.opens_outside(interpreter_roots()) == (Opened(ca_file, writes=False),)


def test_mutations_outside_given_roots_leave_out_only_their_bytecode_caches(tmp_path: Path) -> None:
    runtime_cache = Path(json.__file__).with_name('no-such-package') / '__pycache__'
    runtime_data = Path(json.__file__).with_name('no-such-package') / 'data'

    with recording() as rec:
        for absent in (runtime_cache, runtime_data):
            with pytest.raises(FileNotFoundError):
                os.mkdir(absent)
        os.mkdir(tmp_path / '__pycache__')
        os.mkdir(tmp_path / 'd')

    assert rec.mutations_outside(interpreter_roots()) == (
        Mutated('os.mkdir', (runtime_data,)),
        Mutated('os.mkdir', (tmp_path / '__pycache__',)),
        Mutated('os.mkdir', (tmp_path / 'd',)),
    )
    assert rec.mutations_outside((tmp_path,)) == (
        Mutated('os.mkdir', (runtime_cache,)),
        Mutated('os.mkdir', (runtime_data,)),
        Mutated('os.mkdir', (tmp_path / 'd',)),
    )
