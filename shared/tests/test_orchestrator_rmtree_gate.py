"""Gate: no orchestrator coroutine reaches ``shutil.rmtree`` without an offload hop.

A primitive-filtered widening of the INV-8 caller-side scanner
(``loop_blocking_scan.py``) to ``orchestrator/src``, with zero allowed sites
(task 7366). The scanner reports only the FIRST primitive a call site reaches,
so a helper it labels with another primitive is invisible here; the two such
helpers on the warm-lane route, ``git_ops.py::GitOps._scrub_seeded_lane_target``
and ``git_ops.py::GitOps._relocate_foreign_meta_root``, were converted explicitly.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import NamedTuple

import pytest
from loop_blocking_scan import (
    LoopBlockingSite,
    find_loop_blocking_sites,
    find_loop_blocking_sites_in_trees,
)
from silent_fallthrough_scan import ParsedFile

_SCOPE_PREFIX = 'orchestrator/src/'
_PRIMITIVE = 'shutil.rmtree'


def _rmtree_sites(sites: Sequence[LoopBlockingSite]) -> list[LoopBlockingSite]:
    return [
        site for site in sites
        if site.primitive == _PRIMITIVE and site.filename.startswith(_SCOPE_PREFIX)
    ]


class _OrchestratorScan(NamedTuple):
    scanned_files: int
    rmtree_sites: list[LoopBlockingSite]


@pytest.fixture(scope='session')
def orchestrator_scan(first_party_tree: tuple[ParsedFile, ...]) -> _OrchestratorScan:
    in_scope = [r for r in first_party_tree if r.relpath.startswith(_SCOPE_PREFIX)]
    trees = {r.relpath: r.tree for r in in_scope if r.tree is not None}
    return _OrchestratorScan(
        scanned_files=len(in_scope),
        rmtree_sites=_rmtree_sites(find_loop_blocking_sites_in_trees(trees)),
    )


def test_no_orchestrator_coroutine_deletes_a_tree_inline(
    orchestrator_scan: _OrchestratorScan,
) -> None:
    listing = '\n'.join(
        f'  {s.filename}::{s.qualname} -> {s.callee} ~L{s.lineno}'
        for s in orchestrator_scan.rmtree_sites
    )
    assert not orchestrator_scan.rmtree_sites, (
        'these coroutines reach shutil.rmtree on the event loop thread, which '
        'stalls every other task on the loop (task 7366, INV-8):\n'
        f'{listing}\n'
        'Use orchestrator.tree_removal.remove_tree, or asyncio.to_thread for a '
        'sync helper.'
    )


def test_the_sweep_covers_the_orchestrator_tree(
    orchestrator_scan: _OrchestratorScan,
) -> None:
    assert orchestrator_scan.scanned_files >= 150, (
        f'only {orchestrator_scan.scanned_files} files scanned under '
        f'{_SCOPE_PREFIX} (223 at base 0a1cf64a67): the sweep is broken, and '
        'the zero-site assertion would pass vacuously'
    )


_PROBE_MODULE = textwrap.dedent('''\
    import shutil


    def _wipe(p):
        shutil.rmtree(p)


    async def deletes_inline(p):
        shutil.rmtree(p)


    async def deletes_through_a_helper(p):
        _wipe(p)
''')


def test_the_filter_detects_an_inline_rmtree() -> None:
    sites = find_loop_blocking_sites({'orchestrator/src/orchestrator/_probe.py': _PROBE_MODULE})

    found = _rmtree_sites(sites)

    assert sorted(s.qualname for s in found) == [
        'deletes_inline', 'deletes_through_a_helper',
    ], f'the filter does not match what the scanner emits: {sites}'
