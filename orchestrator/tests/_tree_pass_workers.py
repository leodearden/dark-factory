"""The shared tree pass with its files shared out across a few worker interpreters.

The parent lists every analysis's files once (``_tree_scan.py::list_files``),
partitions the union by file size, and hands each worker the paths it is to
extract, per analysis NAME.  Analyses hold closures that cannot cross a process
boundary, so a worker imports the guard modules that declare them and looks
each one up by name.  Workers are fresh interpreters, never forks: the parent
is a pytest-xdist worker, which runs threads.  A worker that fails, dies, or
replies with anything but exactly its share fails the whole pass, and no worker
outlives the parent's wait.
"""
from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from _tree_scan import (
    AnalysisExtraction,
    FileListing,
    TreeAnalysis,
    TreePass,
    declared_analyses,
    extract_facts,
    list_files,
    modules_requesting_the_pass,
    require_unique_names,
)

__all__ = [
    'WORKER_PROCESSES',
    'TreePass',
    'TreePassWorkerError',
    'partition_by_size',
    'pass_requested_by',
    'run_across_workers',
    'serve',
]

#: At most this many workers, whatever the host: the suite already runs under
#: several xdist workers on a shared, oversubscribed host.
WORKER_PROCESSES = 4

_OUTPUT_TAIL_CHARS = 4000

_WORKER_ENTRY = (
    'import sys; sys.path.insert(0, sys.argv[1]); '
    'import _tree_pass_workers; _tree_pass_workers.serve(sys.argv[2], sys.argv[3])'
)


class TreePassWorkerError(RuntimeError):
    """A worker process did not deliver its share, so the pass has no complete census."""


@dataclass(frozen=True)
class _WorkerRequest:
    sys_path: tuple[str, ...]
    guard_modules: tuple[str, ...]
    paths_by_analysis: dict[str, tuple[Path, ...]]


def pass_requested_by(items: Iterable[object]) -> TreePass:
    """The pass every collected item among *items* that requests the tree pass shares."""
    return run_across_workers(modules_requesting_the_pass(items))


def run_across_workers(
    guard_modules: Iterable[ModuleType], workers: int = WORKER_PROCESSES,
) -> TreePass:
    """The pass over every analysis *guard_modules* declare, shared across up to *workers* processes.

    Never more workers than usable CPUs or files; with fewer than two the pass
    runs in this process.  Equal to ``TreePass.run`` over the same analyses.
    """
    modules = tuple(dict.fromkeys(guard_modules))
    module_names = tuple(_importable_name(module) for module in modules)
    declared = declared_analyses(modules)
    require_unique_names(declared)
    listing = list_files(declared)
    shares = partition_by_size(listing.union, min(workers, _usable_cpus()))
    if len(shares) < 2:
        return TreePass.assembled(listing, [extract_facts(listing.file_sets)])
    requests = [
        _WorkerRequest(tuple(sys.path), module_names, _assignment(listing, share))
        for share in shares
    ]
    return TreePass.assembled(listing, _extract_in_workers(requests))


def partition_by_size(paths: Iterable[Path], count: int) -> tuple[frozenset[Path], ...]:
    """*paths* split into at most *count* non-empty, disjoint shares of near-equal total size.

    Deterministic: largest file first, each to the share with the least so
    far, ties to the share with fewer files, then to the lower index.
    """
    if count < 1:
        raise ValueError(f'a partition needs at least one share, not {count}')
    sizes = {path: _size(path) for path in paths}
    shares: list[list[Path]] = [[] for _ in range(min(count, len(sizes)))]
    totals = [0] * len(shares)
    for path in sorted(sizes, key=lambda p: (-sizes[p], p)):
        lightest = min(range(len(shares)), key=lambda i: (totals[i], len(shares[i]), i))
        shares[lightest].append(path)
        totals[lightest] += sizes[path]
    return tuple(frozenset(share) for share in shares)


def serve(request_path: str, reply_path: str) -> None:
    """One worker's whole life: extract the share at *request_path*, write the reply to *reply_path*."""
    request: _WorkerRequest = pickle.loads(Path(request_path).read_bytes())
    sys.path[:] = request.sys_path
    analyses = _analyses_named(request.paths_by_analysis, request.guard_modules)
    reply = extract_facts({
        analyses[name]: frozenset(paths) for name, paths in request.paths_by_analysis.items()
    })
    Path(reply_path).write_bytes(pickle.dumps(reply))


def _importable_name(module: ModuleType) -> str:
    if sys.modules.get(module.__name__) is not module:
        raise ValueError(
            f'guard module {module.__name__!r} ({getattr(module, "__file__", None)}) is not '
            'what importing that name gives, so a worker process cannot re-import it'
        )
    return module.__name__


def _usable_cpus() -> int:
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _size(path: Path) -> int:
    """*path*'s size for balancing only: a file that cannot be stat'ed weighs nothing, and fails where it is read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _assignment(listing: FileListing, share: frozenset[Path]) -> dict[str, tuple[Path, ...]]:
    return {
        analysis.name: tuple(sorted(files & share))
        for analysis, files in listing.file_sets.items()
        if files & share
    }


def _analyses_named(
    names: Iterable[str], module_names: Sequence[str],
) -> dict[str, TreeAnalysis]:
    declared = declared_analyses(importlib.import_module(name) for name in module_names)
    require_unique_names(declared)
    by_name = {analysis.name: analysis for analysis in declared}
    missing = sorted(set(names) - by_name.keys())
    if missing:
        raise LookupError(
            f'no guard module among {list(module_names)} declares the tree analyses {missing} '
            'in this worker process'
        )
    return {name: by_name[name] for name in names}


def _extract_in_workers(requests: Sequence[_WorkerRequest]) -> list[dict[str, AnalysisExtraction]]:
    with tempfile.TemporaryDirectory(prefix='tree-pass-workers-') as scratch:
        workers: list[_Worker] = []
        try:
            for index, request in enumerate(requests):
                workers.append(_Worker.start(index, request, Path(scratch)))
            for worker in workers:
                worker.process.wait()
        finally:
            for worker in workers:
                worker.stop()
        return [worker.reply() for worker in workers]


@dataclass(frozen=True)
class _Worker:
    index: int
    request: _WorkerRequest
    process: subprocess.Popen[bytes]
    reply_path: Path
    output_path: Path

    @classmethod
    def start(cls, index: int, request: _WorkerRequest, scratch: Path) -> _Worker:
        request_path = scratch / f'worker-{index}.request'
        request_path.write_bytes(pickle.dumps(request))
        reply_path = scratch / f'worker-{index}.reply'
        output_path = scratch / f'worker-{index}.output'
        with output_path.open('wb') as output:
            process = subprocess.Popen(
                [
                    sys.executable, '-c', _WORKER_ENTRY,
                    str(Path(__file__).resolve().parent), str(request_path), str(reply_path),
                ],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            )
        return cls(index, request, process, reply_path, output_path)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()

    def reply(self) -> dict[str, AnalysisExtraction]:
        if self.process.returncode != 0:
            raise self._failure(_how_it_ended(self.process.returncode))
        try:
            reply = pickle.loads(self.reply_path.read_bytes())
        except Exception as error:
            raise self._failure(
                f'exited 0 but left no readable reply ({type(error).__name__}: {error})',
            ) from error
        mismatch = _share_mismatch(self.request.paths_by_analysis, reply)
        if mismatch is not None:
            raise self._failure(f'replied with something other than its share: {mismatch}')
        return reply

    def _failure(self, what_happened: str) -> TreePassWorkerError:
        output = self.output_path.read_bytes().decode('utf-8', errors='replace')
        return TreePassWorkerError(
            f'tree pass worker {self.index} (pid {self.process.pid}) {what_happened}; '
            f'the end of its output:\n{output[-_OUTPUT_TAIL_CHARS:]}'
        )


def _how_it_ended(returncode: int) -> str:
    if returncode < 0:
        return f'was killed by signal {-returncode}'
    return f'exited with status {returncode}'


def _share_mismatch(assigned: Mapping[str, tuple[Path, ...]], reply: object) -> str | None:
    """Why *reply* is not one fact or failure per assigned path for every assigned analysis, or None."""
    if not isinstance(reply, dict) or reply.keys() != assigned.keys():
        return f'a reply for {reply!r:.200}, not for the analyses {sorted(assigned)}'
    for name, paths in assigned.items():
        extraction = reply[name]
        if not isinstance(extraction, AnalysisExtraction):
            return f'{name!r}: a {type(extraction).__name__}, not an AnalysisExtraction'
        covered = sorted([
            *(path for path, _fact in extraction.facts),
            *(failure.path for failure in extraction.failures),
        ])
        if covered != list(paths):
            return f'{name!r}: {len(covered)} paths answered, not exactly its {len(paths)} assigned'
    return None
