"""The audit-hook recorder B30 scopes the filer's effects with: what it records, and what it leaves out."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from _audit_hook import Connected, Mutated, Opened, recording, runtime_roots


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
