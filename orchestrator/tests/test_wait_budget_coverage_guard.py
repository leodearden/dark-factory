"""Tree-wide guard: every pytest item under orchestrator/tests runs under a
timeout that clears its wait bill.

The bill is ``_wait_budget_audit.py``'s static estimate of the wall clock an
item's recognised waits can take. The limit is the closest timeout mark
pytest-timeout applies -- the test's own, its class's, its module's
``pytestmark`` -- else the ambient budget. A breach is not a slow pass:
pytest-timeout's thread method ``os._exit()``s the xdist worker under
``--max-worker-restart=0``, so it surfaces as a worker death blamed on
whichever test shared the worker.

OPT-OUT over every ``orchestrator/tests/test_*.py``: a new module is covered
the moment it exists. Known offenders live in two shrink-only mechanisms
below; a timeout is never widened to pass.

This is not a rule in fused-memory/scripts/check_bare_magicmock_config.py:
that checker holds a stdlib-only contract (hooks/project-checks runs it under
a bare ``python3``), while this guard reads each module's runtime marks and
namespace, which needs the module imported inside this pytest session
(conftest.py puts the tests dir and the repo root on ``sys.path``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT
from _wait_budget_audit import ItemWaitAudit, audit_module, is_module_level_item

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    # One tree scan per worker serves every test that reads it, so keep them
    # on one worker.
    pytest.mark.xdist_group('wait_budget_coverage_scan'),
]

_TESTS_DIR = Path(__file__).resolve().parent

#: Modules whose waits run against an injected clock, so a static wall-clock
#: bill does not describe them. Not debt: an entry goes only when its module
#: no longer yields an offender.
_SYNTHETIC_CLOCK_MODULES: frozenset[str] = frozenset(
    {
        # Its waits run against an injected _StarvationClock or tiny injected
        # caps, never the wall clock its bill describes.
        'test_orch_helpers_wait_responsive.py',
    }
)

#: Shrink-only debt: ``'<file>::<item>'`` -> its EXACT recorded bill. Lower an
#: entry as its bill falls, delete it when the item stops offending, and never
#: add one for new code.
_WAIT_BUDGET_DEBT: Mapping[str, float] = MappingProxyType(
    {
        # Burned down by task 6507.
        'test_merge_lane_restart_drain.py::test_a_train_does_not_park_on_a_predecessor_the_halt_sent_back_to_the_queue': 450.0,
    }
)

# Non-vacuity floors, ~75% of the census measured when this guard landed.
_MIN_BILLED_MODULES = 48
_MIN_BILLED_ITEMS = 311
_MIN_MODULE_LEVEL_BILLED_ITEMS = 27
_MIN_MARK_LIMITED_ITEMS = 72
_MIN_MARK_LIMITED_MODULES = 3

#: Modules whose own per-file timeout-mark self-scans this guard replaced.
_HANDOVER_MODULES = (
    'test_merge_queue_concurrent_verify.py',
    'test_merge_speculation.py',
    'test_multihost_verify_integration.py',
    'test_merge_queue.py',
)


@dataclasses.dataclass(frozen=True)
class _ModuleCensus:
    billed_items: int
    module_level_billed_items: int
    mark_limited_items: int
    covered_mark_limited_items: int


@dataclasses.dataclass(frozen=True)
class _TreeScan:
    offenders: tuple[tuple[str, ItemWaitAudit], ...]
    exempt_offender_counts: Mapping[str, int]
    census: Mapping[str, _ModuleCensus]


def _node_id(module_file: str, offender: ItemWaitAudit) -> str:
    return f'{module_file}::{offender.item}'


def _module_namespace(path: Path) -> Mapping[str, object] | None:
    """The runtime globals of the test module at *path*, or None when it
    skips itself at import (so it never runs and cannot breach)."""
    try:
        module = importlib.import_module(path.stem)
    except pytest.skip.Exception:
        return None
    except Exception as exc:
        raise AssertionError(
            f'{path.name} failed to import ({exc!r}), so its timeout marks '
            f'cannot be read; fix the import rather than exempting the module.'
        ) from exc
    return vars(module)


def _module_census(audits: list[ItemWaitAudit]) -> _ModuleCensus:
    mark_limited = [audit for audit in audits if audit.limit_source.is_mark]
    return _ModuleCensus(
        billed_items=len(audits),
        module_level_billed_items=sum(is_module_level_item(audit.item) for audit in audits),
        mark_limited_items=len(mark_limited),
        covered_mark_limited_items=sum(not audit.offends for audit in mark_limited),
    )


@functools.cache
def _scan_tree(tests_dir: Path) -> _TreeScan:
    """Audit every billed test module in *tests_dir*, each importable by its
    stem; retain extracted facts, never ASTs."""
    offenders: list[tuple[str, ItemWaitAudit]] = []
    exempt_offender_counts: dict[str, int] = {}
    census: dict[str, _ModuleCensus] = {}
    for path in sorted(tests_dir.glob('test_*.py')):
        audits = audit_module(
            path.read_text(encoding='utf-8'), functools.partial(_module_namespace, path)
        )
        if audits is None:
            continue
        found = [audit for audit in audits if audit.offends]
        if path.name in _SYNTHETIC_CLOCK_MODULES:
            exempt_offender_counts[path.name] = len(found)
        else:
            offenders.extend((path.name, offender) for offender in found)
        census[path.name] = _module_census(audits)
    return _TreeScan(tuple(offenders), exempt_offender_counts, census)


def test_no_offender_outside_the_ledger() -> None:
    unledgered = [
        f'  - {module_file}::{offender.describe()}'
        for module_file, offender in _scan_tree(_TESTS_DIR).offenders
        if _node_id(module_file, offender) not in _WAIT_BUDGET_DEBT
    ]

    assert not unledgered, (
        'These pytest items can wait longer than the timeout they run under:\n'
        + '\n'.join(unledgered)
        + '\n\nFix the item, never the ledger:\n'
        '  - bill each wait at the nominal its kind is reserved: a merge '
        'result future at MERGE_RESULT_TIMEOUT, an asyncio.Event or entry '
        'barrier at MERGE_GATE_BARRIER_TIMEOUT (the task-3980 DECISION block '
        'in _orch_helpers.py). Moving a 45s-nominal gate onto wait_responsive '
        'RAISES its bill to 90 (task 5066);\n'
        '  - or carry a @pytest.mark.timeout that clears the bill;\n'
        '  - an unbounded bill is a wait the scan cannot read: spell its bound '
        'as a literal or a module-level constant.\n'
        'Never add new code to _WAIT_BUDGET_DEBT.'
    )


def test_debt_ledger_matches_live_bills() -> None:
    live = {
        _node_id(module_file, offender): offender
        for module_file, offender in _scan_tree(_TESTS_DIR).offenders
    }
    problems: list[str] = []
    for node_id, recorded in sorted(_WAIT_BUDGET_DEBT.items()):
        offender = live.get(node_id)
        if offender is None:
            problems.append(f'{node_id}: no longer an offender -- delete the entry.')
        elif offender.budget < recorded:
            problems.append(
                f'{node_id}: its bill fell -- lower the entry from {recorded} '
                f'to {offender.budget}.'
            )
        elif offender.budget > recorded:
            problems.append(
                f'{node_id}: its bill grew from {recorded} to '
                f'{offender.budget}. A new wait in a ledgered item is '
                f'forbidden; an engine recognition change records the raise '
                f'in the same commit, never silently.'
            )

    assert not problems, (
        '_WAIT_BUDGET_DEBT must record each ledgered bill exactly:\n'
        + '\n'.join(f'  - {problem}' for problem in problems)
    )


def test_synthetic_clock_exemptions_are_still_needed() -> None:
    exempt_offender_counts = _scan_tree(_TESTS_DIR).exempt_offender_counts
    stale = [
        f'  - {module}: no longer exists -- delete the entry.'
        if not (_TESTS_DIR / module).exists()
        else f'  - {module}: yields no offender -- delete the entry.'
        for module in sorted(_SYNTHETIC_CLOCK_MODULES)
        if exempt_offender_counts.get(module, 0) == 0
    ]

    assert not stale, (
        '_SYNTHETIC_CLOCK_MODULES must name only modules that still need '
        'the exemption:\n' + '\n'.join(stale)
    )


def test_the_scan_reaches_sibling_modules() -> None:
    census = _scan_tree(_TESTS_DIR).census
    totals = {
        'billed modules': (len(census), _MIN_BILLED_MODULES),
        'billed items': (
            sum(module.billed_items for module in census.values()),
            _MIN_BILLED_ITEMS,
        ),
        'module-level billed items': (
            sum(module.module_level_billed_items for module in census.values()),
            _MIN_MODULE_LEVEL_BILLED_ITEMS,
        ),
        'mark-limited items': (
            sum(module.mark_limited_items for module in census.values()),
            _MIN_MARK_LIMITED_ITEMS,
        ),
        'modules with a mark-limited item': (
            sum(module.mark_limited_items > 0 for module in census.values()),
            _MIN_MARK_LIMITED_MODULES,
        ),
    }
    below_floor = [
        f'  - {name}: {count} < {floor}'
        for name, (count, floor) in totals.items()
        if count < floor
    ]
    unheld = [
        f'  - {module}'
        for module in _HANDOVER_MODULES
        if module not in census or census[module].covered_mark_limited_items == 0
    ]

    assert not below_floor, (
        'The scan reached less of the tree than it did when this guard '
        'landed, so it may have gone vacuous:\n' + '\n'.join(below_floor)
    )
    assert not unheld, (
        'These modules contribute no covered item whose limit came from a '
        'timeout mark, so this guard no longer holds what their retired '
        'self-scans held:\n' + '\n'.join(unheld)
    )


_PROBE_WAITING_TEST = (
    'import asyncio\n'
    'async def test_waits():\n'
    '    await asyncio.wait_for(asyncio.sleep(0), timeout=10**6)\n'
)


@contextlib.contextmanager
def _probe_tests_dir(directory: Path, modules: Mapping[str, str]) -> Iterator[Path]:
    """*directory* holding *modules* (stem -> source), importable by stem
    while the context lasts."""
    for stem, source in modules.items():
        (directory / f'{stem}.py').write_text(source, encoding='utf-8')
    sys.path.insert(0, str(directory))
    importlib.invalidate_caches()
    try:
        yield directory
    finally:
        sys.path.remove(str(directory))
        for stem in modules:
            sys.modules.pop(stem, None)


def test_the_scan_drops_a_module_that_skips_at_import(tmp_path: Path) -> None:
    modules = {
        'test_wbcg_probe_audited': _PROBE_WAITING_TEST,
        'test_wbcg_probe_skips': (
            "import pytest\npytest.importorskip('wbcg_probe_absent_module')\n"
            + _PROBE_WAITING_TEST
        ),
        'test_wbcg_probe_unbilled': (
            "raise RuntimeError('a module with nothing to bill is never imported')\n"
            'def test_idle():\n    pass\n'
        ),
    }
    with _probe_tests_dir(tmp_path, modules) as tests_dir:
        scan = _scan_tree(tests_dir)

    assert list(scan.census) == ['test_wbcg_probe_audited.py'], (
        'only the module that imported and bills is audited; a module that '
        f'skips at import never runs, so it cannot breach: {list(scan.census)}'
    )
    assert [(module, offender.item) for module, offender in scan.offenders] == [
        ('test_wbcg_probe_audited.py', 'test_waits')
    ]


def test_the_scan_fails_on_a_module_that_cannot_be_imported(tmp_path: Path) -> None:
    """A module whose marks cannot be read must fail the scan, never drop
    out of coverage silently."""
    modules = {
        'test_wbcg_probe_broken': "raise RuntimeError('probe import failure')\n"
        + _PROBE_WAITING_TEST
    }
    with (
        _probe_tests_dir(tmp_path, modules) as tests_dir,
        pytest.raises(AssertionError, match='test_wbcg_probe_broken.py failed to import'),
    ):
        _scan_tree(tests_dir)
