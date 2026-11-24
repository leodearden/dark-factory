"""Contract of ``_tree_pass_workers``: the pass shared across worker processes equals the serial pass.

Every case writes a synthetic guard module and tree under ``tmp_path``; none
touches the real repository, so this module is not a consumer of the shared
pass.  Each worker is a fresh interpreter, so the cases that start workers
keep to a handful of files.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from _tree_pass_workers import TreePassWorkerError, partition_by_size, run_across_workers
from _tree_scan import TreeAnalysis, TreePass, python_files

import orchestrator

GuardFactory = Callable[[str], ModuleType]


@pytest.fixture
def guard_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[GuardFactory]:
    """Writes *source* as a uniquely named guard module, importable by that name in workers too."""
    directory = tmp_path / 'guards'
    directory.mkdir()
    monkeypatch.syspath_prepend(str(directory))
    names: list[str] = []

    def write(source: str) -> ModuleType:
        name = f'synthetic_guard_{uuid.uuid4().hex}'
        (directory / f'{name}.py').write_text(source, encoding='utf-8')
        names.append(name)
        return __import__(name)

    yield write
    for name in names:
        sys.modules.pop(name, None)


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return root.resolve()


def _outcomes(
    scanned: TreePass, analyses: tuple[TreeAnalysis, ...],
) -> dict[str, list[tuple[Path, object]] | str]:
    """Each analysis's facts in order, or the failure its ``facts`` raises."""
    outcomes: dict[str, list[tuple[Path, object]] | str] = {}
    for analysis in analyses:
        try:
            outcomes[analysis.name] = list(scanned.facts(analysis).items())
        except AssertionError as failure:
            outcomes[analysis.name] = str(failure)
    return outcomes


_MIXED_GUARD = '''
from dataclasses import dataclass
from pathlib import Path
from _tree_scan import TreeAnalysis, python_files

ROOT = Path({root!r})


@dataclass(frozen=True)
class Shape:
    name: str
    statements: int


def _shape(module):
    if module.path.name == 'explode.py':
        raise ValueError('boom on purpose')
    return Shape(module.path.name, len(module.tree.body))


def _unlistable():
    raise NotADirectoryError('no such root on purpose')


TREE_ANALYSES = (
    TreeAnalysis(name='shapes', files=lambda: python_files(ROOT), extract=_shape),
    TreeAnalysis(name='sizes', files=lambda: python_files(ROOT / 'sub'), extract=lambda m: len(m.source)),
    TreeAnalysis(name='unlistable', files=_unlistable, extract=lambda m: m.path.name),
)
'''


def test_the_pass_across_workers_equals_the_serial_pass(
    tmp_path: Path, guard_module: GuardFactory,
) -> None:
    root = _tree(tmp_path / 'tree', {
        'a.py': 'x = 1\n',
        'b.py': 'x = 1\ny = 2\nz = 3\n' * 20,
        'explode.py': 'x = 1\n',
        'sub/c.py': 'def f():\n    return 1\n',
        'sub/d.py': 'import os\n',
        'sub/broken.py': 'def broken(:\n',
    })
    module = guard_module(_MIXED_GUARD.format(root=str(root)))

    shapes, sizes, unlistable = module.TREE_ANALYSES
    serial_pass = TreePass.run(module.TREE_ANALYSES)

    serial = _outcomes(serial_pass, module.TREE_ANALYSES)
    across_workers = _outcomes(run_across_workers([module], workers=3), module.TREE_ANALYSES)

    assert across_workers == serial
    assert 'explode.py: ValueError: boom on purpose' in str(serial[shapes.name])
    assert 'broken.py: SyntaxError' in str(serial[shapes.name])
    assert [path.name for path in serial_pass.facts(sizes)] == ['broken.py', 'c.py', 'd.py']
    assert 'NotADirectoryError: no such root on purpose' in str(serial[unlistable.name])


def test_every_file_is_extracted_once_by_one_of_at_most_as_many_workers_as_files(
    tmp_path: Path, guard_module: GuardFactory,
) -> None:
    root = _tree(tmp_path / 'tree', {f'm{index}.py': 'x = 1\n' * index for index in range(3)})
    module = guard_module(
        'import os\nfrom _tree_scan import TreeAnalysis, python_files\n'
        f'TREE_ANALYSES = (TreeAnalysis("pid", lambda: python_files({str(root)!r}), '
        'lambda m: os.getpid()),)\n'
    )

    facts = run_across_workers([module], workers=8).facts(module.TREE_ANALYSES[0])

    assert list(facts) == list(python_files(root))
    assert os.getpid() not in facts.values()
    assert len(set(facts.values())) == min(3, len(os.sched_getaffinity(0)))


class TestPartitionBySize:
    @pytest.mark.parametrize('count', [1, 2, 3, 6, 9])
    def test_every_path_lands_in_exactly_one_non_empty_share(self, tmp_path: Path, count: int) -> None:
        root = _tree(tmp_path, {f'm{index}.py': 'x' * (index * 37 % 11) for index in range(6)})
        paths = python_files(root)

        shares = partition_by_size(paths, count)

        assert len(shares) == min(count, len(paths))
        assert all(shares)
        assert sorted(path for share in shares for path in share) == sorted(paths)
        assert partition_by_size(reversed(paths), count) == shares

    def test_the_largest_files_are_spread_before_the_small_ones_fill_in(self, tmp_path: Path) -> None:
        root = _tree(tmp_path, {'big.py': 'x' * 900, 'mid.py': 'x' * 500, 'a.py': 'x' * 200,
                                'b.py': 'x' * 200, 'c.py': 'x' * 100})

        shares = partition_by_size(python_files(root), 2)

        assert {frozenset(path.name for path in share) for share in shares} == {
            frozenset({'big.py', 'c.py'}), frozenset({'mid.py', 'a.py', 'b.py'}),
        }

    def test_no_shares_is_refused(self) -> None:
        with pytest.raises(ValueError, match='at least one share'):
            partition_by_size([], 0)


_DYING_GUARD = '''
import os, signal, sys
from dataclasses import dataclass
from _tree_scan import TreeAnalysis, python_files


def _die(module):
    if module.path.name == 'die.py':
        print('dying on purpose', file=sys.stderr, flush=True)
        {death}
    return module.path.name


TREE_ANALYSES = (TreeAnalysis('dying', lambda: python_files({root!r}), _die),)
'''

_UNPICKLABLE_FACT = '''@dataclass(frozen=True)
        class Local:
            name: str
        return Local(module.path.name)'''


@pytest.mark.parametrize(
    ('death', 'how_it_ended'),
    [
        pytest.param('os.kill(os.getpid(), signal.SIGKILL)', 'was killed by signal 9', id='killed'),
        pytest.param('os._exit(0)', 'exited 0 but left no readable reply', id='exited-without-reply'),
        pytest.param(_UNPICKLABLE_FACT, 'exited with status 1', id='unpicklable-fact'),
    ],
)
def test_a_worker_that_does_not_deliver_fails_the_pass_naming_it(
    tmp_path: Path, guard_module: GuardFactory, death: str, how_it_ended: str,
) -> None:
    root = _tree(tmp_path / 'tree', {'die.py': 'x = 1\n', 'live.py': 'x = 1\n'})
    module = guard_module(_DYING_GUARD.format(root=str(root), death=death))

    with pytest.raises(TreePassWorkerError) as raised:
        run_across_workers([module], workers=2)

    message = str(raised.value)
    assert how_it_ended in message
    assert 'tree pass worker ' in message and '(pid ' in message
    assert 'dying on purpose' in message


_SPLIT_BRAIN_GUARD = '''
import os
from _tree_scan import TreeAnalysis, python_files


def _named(name):
    return TreeAnalysis(name, lambda: python_files({root!r}), lambda m: m.path.name)


TREE_ANALYSES = (_named('present'),) if os.getpid() == {parent_pid} else {in_worker}
'''


@pytest.mark.parametrize(
    ('in_worker', 'complaint'),
    [
        pytest.param('()', "declares the tree analyses ['present']", id='missing'),
        pytest.param("(_named('present'), _named('present'))", "duplicated: ['present']", id='duplicated'),
    ],
)
def test_a_worker_that_cannot_find_exactly_one_analysis_by_its_name_fails_the_pass(
    tmp_path: Path, guard_module: GuardFactory, in_worker: str, complaint: str,
) -> None:
    root = _tree(tmp_path / 'tree', {'a.py': 'x = 1\n', 'b.py': 'x = 1\n'})
    module = guard_module(
        _SPLIT_BRAIN_GUARD.format(root=str(root), parent_pid=os.getpid(), in_worker=in_worker),
    )

    with pytest.raises(TreePassWorkerError, match=r'exited with status 1;') as raised:
        run_across_workers([module], workers=2)

    assert complaint in str(raised.value)


def test_a_module_not_importable_by_its_name_is_refused_before_any_worker_starts() -> None:
    stray = ModuleType('not_in_sys_modules')

    with pytest.raises(ValueError, match="'not_in_sys_modules'.*cannot re-import"):
        run_across_workers([stray], workers=2)


_ORCHESTRATOR_FILE_GUARD = '''
import os
import orchestrator
from _tree_scan import TreeAnalysis, python_files

TREE_ANALYSES = (TreeAnalysis('orchestrator-file', lambda: python_files({root!r}), lambda m: (os.getpid(), orchestrator.__file__)),)
'''


def test_a_worker_imports_orchestrator_from_the_file_the_parent_did(
    tmp_path: Path, guard_module: GuardFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _tree(tmp_path / 'tree', {'a.py': 'x = 1\n', 'b.py': 'x = 1\n'})
    module = guard_module(_ORCHESTRATOR_FILE_GUARD.format(root=str(root)))
    decoy = _tree(tmp_path / 'decoy', {'orchestrator/__init__.py': ''})
    monkeypatch.syspath_prepend(str(decoy))

    facts = run_across_workers([module], workers=2).facts(module.TREE_ANALYSES[0])

    assert os.getpid() not in {pid for pid, _file in facts.values()}
    assert {Path(file).resolve() for _pid, file in facts.values()} == {
        Path(orchestrator.__file__).resolve(),
    }


def test_a_worker_already_holding_another_copy_of_a_parent_module_fails_the_pass_naming_both_files(
    tmp_path: Path, guard_module: GuardFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _tree(tmp_path / 'tree', {'a.py': 'x = 1\n', 'b.py': 'x = 1\n'})
    module = guard_module(_ORCHESTRATOR_FILE_GUARD.format(root=str(root)))
    decoy = _tree(tmp_path / 'decoy', {'orchestrator/__init__.py': ''})
    startup = _tree(tmp_path / 'startup', {'sitecustomize.py': 'import orchestrator\n'})
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(startup), str(decoy)]))

    with pytest.raises(TreePassWorkerError, match=r'exited with status 1;') as raised:
        run_across_workers([module], workers=2)

    message = str(raised.value)
    assert "'orchestrator' from " in message
    assert str(decoy / 'orchestrator' / '__init__.py') in message
    assert orchestrator.__file__ in message


_HELD_FILE_GUARD = '''
import os, sys
from _tree_scan import TreeAnalysis, python_files

TREE_ANALYSES = (TreeAnalysis('held-file', lambda: python_files({root!r}), lambda m: (os.getpid(), sys.modules[{held!r}].__file__)),)
'''


def test_a_worker_holding_a_parent_module_through_a_symlink_to_the_parent_file_runs(
    tmp_path: Path, guard_module: GuardFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _tree(tmp_path / 'tree', {'a.py': 'x = 1\n', 'b.py': 'x = 1\n'})
    held = guard_module('')
    assert held.__file__ is not None
    held_file = Path(held.__file__)
    module = guard_module(_HELD_FILE_GUARD.format(root=str(root), held=held.__name__))
    link = tmp_path / 'link'
    link.symlink_to(held_file.parent, target_is_directory=True)
    startup = _tree(tmp_path / 'startup', {'sitecustomize.py': f'import {held.__name__}\n'})
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(startup), str(link)]))

    facts = run_across_workers([module], workers=2).facts(module.TREE_ANALYSES[0])

    assert os.getpid() not in {pid for pid, _file in facts.values()}
    assert {file for _pid, file in facts.values()} == {str(link / held_file.name)}


_HANGING_GUARD = '''
import os, time
from pathlib import Path
from _tree_scan import TreeAnalysis, python_files


def _hang(module):
    (Path({started!r}) / str(os.getpid())).touch()
    time.sleep(600)


TREE_ANALYSES = (TreeAnalysis('hanging', lambda: python_files({root!r}), _hang),)
'''


class _Interrupted(Exception):
    pass


def _raise_interrupted(_signum: int, _frame: object) -> None:
    raise _Interrupted


def _interrupt_once_workers_start(started: Path, workers: int, waiter: int, deadline: float) -> None:
    while len(list(started.iterdir())) < workers and time.monotonic() < deadline:
        time.sleep(0.05)
    signal.pthread_kill(waiter, signal.SIGUSR1)


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_an_interrupted_wait_kills_and_reaps_every_worker(
    tmp_path: Path, guard_module: GuardFactory,
) -> None:
    root = _tree(tmp_path / 'tree', {'a.py': 'x = 1\n', 'b.py': 'x = 1\n'})
    started = tmp_path / 'started'
    started.mkdir()
    module = guard_module(_HANGING_GUARD.format(root=str(root), started=str(started)))
    previous_handler = signal.signal(signal.SIGUSR1, _raise_interrupted)
    try:
        threading.Thread(
            target=_interrupt_once_workers_start,
            args=(started, 2, threading.get_ident(), time.monotonic() + 120),
            daemon=True,
        ).start()
        with pytest.raises(_Interrupted):
            run_across_workers([module], workers=2)
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)

    worker_pids = [int(path.name) for path in started.iterdir()]
    assert len(worker_pids) == 2
    assert [pid for pid in worker_pids if _is_running(pid)] == []
