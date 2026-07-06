"""Guard tests for the recon `asyncio.gather(..., return_exceptions=True)` +
`isinstance(r, BaseException)` swallow idiom (PRD
plans/fm-cancellederror-convention-prd.md, task γ).

Two independent tests live here:

1. ``TestCapturedCancelledErrorPropagatesFromSummaryPool`` — a behavioural
   regression test that drives the real
   ``reconciliation.summary_pool.enforce_summary_pool_cap`` sweep with an
   injected ``asyncio.CancelledError`` from one ``delete_memory`` call, and
   asserts it re-raises (rather than being swallowed into a WARNING and an
   under-counted return value).

2. ``TestNoRawGatherReturnExceptionsOutsideHelperOrAllowlist`` — a whole-tree
   AST drift guard: every ``asyncio.gather(..., return_exceptions=True)`` call
   site under ``src/fused_memory/`` must live either inside the helper home
   (``utils/async_utils.py``) or in the explicit drain ALLOWLIST; any other
   site fails the test by file:line:function.

Both tests are new — task γ owns this file exclusively (see the PRD's W5 seam
note); no existing recon test files are touched.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.summary_pool import enforce_summary_pool_cap


class TestCapturedCancelledErrorPropagatesFromSummaryPool:
    """A captured CancelledError from one delete_memory call must re-raise,
    aborting the sweep — NOT be swallowed into a WARNING + under-count.

    RED on current main: enforce_summary_pool_cap's per-item guard is
    ``isinstance(result, BaseException)``, which catches the captured
    CancelledError, logs it as a WARNING ("not counted"), and returns an int
    (success_count) instead of re-raising.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_from_one_delete_reraises(self) -> None:
        members = [
            {'id': 'oldest', 'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'second', 'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'third', 'created_at': '2026-03-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'newest', 'created_at': '2026-04-01T00:00:00+00:00', 'metadata': {}},
        ]

        async def _delete_side_effect(*, memory_id, store, project_id, causation_id, _source):
            if memory_id == 'oldest':
                raise asyncio.CancelledError()
            return None

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(side_effect=_delete_side_effect)

        with pytest.raises(asyncio.CancelledError):
            await enforce_summary_pool_cap(
                memory_service,
                project_id='dark_factory',
                run_id='run-cancel',
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )


SRC_ROOT = pathlib.Path(__file__).parents[1] / 'src' / 'fused_memory'
ASYNC_UTILS_PATH = SRC_ROOT / 'utils' / 'async_utils.py'

# (path-relative-to-src, enclosing-function-name) -> (expected site count,
# one-line justification).
#
# Each entry is a deliberate teardown/drain site: an await on already-
# scheduled asyncio.Task objects where a cancelled child is an EXPECTED drain
# outcome, and the drain must run to completion to guarantee cleanup. These
# are NOT best-effort collect-and-continue sweeps, so they are intentionally
# excluded from the gather_collect conversion (PRD
# plans/fm-cancellederror-convention-prd.md, task γ, resolved design
# decision 3).
#
# The key is (path, enclosing-function) only — it does not distinguish
# between multiple distinct `gather(...)` calls inside the same function. The
# expected-count guards against that: if an allowlisted function later grows
# a SECOND `gather(..., return_exceptions=True)` call (e.g. a best-effort
# collect sweep added alongside a genuine drain), the site count for that key
# exceeds its expected count and the guard fails, forcing that new call to
# get its own explicit review rather than silently inheriting the drain's
# justification.
DRAIN_ALLOWLIST: dict[tuple[str, str], tuple[int, str]] = {
    ('fused_memory/middleware/task_interceptor.py', 'drain'): (
        1,
        'Fire-and-forget background-task shutdown drain; a cancelled child '
        'is an expected drain outcome and the drain must complete to '
        'guarantee cleanup.'
    ),
    ('fused_memory/middleware/task_interceptor.py', 'close'): (
        1,
        'Per-project curator worker drain at interceptor close; same '
        'must-complete teardown semantics as drain().'
    ),
    ('fused_memory/services/durable_queue.py', 'close'): (
        1,
        'Durable queue worker drain at close; cancelled workers are an '
        'expected shutdown outcome, not a failure to propagate.'
    ),
    ('fused_memory/reconciliation/harness.py', 'run_loop'): (
        1,
        'Graceful-shutdown drain of per-project reconciliation loop tasks '
        'in the run_loop finally block.'
    ),
    ('fused_memory/server/main.py', 'run_server'): (
        1,
        'Documented CancelledError-collection drain of the primary/recon '
        'server tasks before re-raising in run_server.'
    ),
}


def _is_true_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


class _GatherSiteVisitor(ast.NodeVisitor):
    """Finds every `gather(..., return_exceptions=True)` call site in a
    module, attributing each to its innermost enclosing FunctionDef /
    AsyncFunctionDef via a def-stack.

    Attributing by (path, enclosing-function-name) rather than by line range
    is drift-stable: it survives line-number churn elsewhere in the file
    (the task's own site census already drifted a few lines since it was
    written), unlike a line-based allowlist would.
    """

    def __init__(self) -> None:
        self._def_stack: list[str] = []
        self.sites: list[tuple[int, str | None]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._def_stack.append(node.name)
        self.generic_visit(node)
        self._def_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_gather = (isinstance(func, ast.Name) and func.id == 'gather') or (
            isinstance(func, ast.Attribute) and func.attr == 'gather'
        )
        if is_gather and any(
            kw.arg == 'return_exceptions' and _is_true_constant(kw.value) for kw in node.keywords
        ):
            enclosing = self._def_stack[-1] if self._def_stack else None
            self.sites.append((node.lineno, enclosing))
        self.generic_visit(node)


def _gather_return_exceptions_sites(path: pathlib.Path) -> list[tuple[int, str | None]]:
    """Every (lineno, enclosing_function_name) for `gather(...,
    return_exceptions=True)` call sites found in the file at *path*."""
    tree = ast.parse(path.read_text(), filename=str(path))
    visitor = _GatherSiteVisitor()
    visitor.visit(tree)
    return visitor.sites


class TestNoRawGatherReturnExceptionsOutsideHelperOrAllowlist:
    """Whole-tree AST drift guard (PRD task γ): every
    `gather(..., return_exceptions=True)` call site under src/fused_memory/
    must live either in the helper home (utils/async_utils.py — where
    gather_collect / gather_or_raise are themselves implemented) or in the
    explicit drain ALLOWLIST above. Any other site — a not-yet-converted
    recon sweep, or a brand new hand-rolled site added later — fails this
    test, listing the offending file:line:function.

    Matches real ``ast.Call`` nodes (not string/grep), so a docstring or
    comment that merely *describes* the idiom (e.g.
    ``services/memory_service.py``'s prose) never false-positives, and a
    plain ``gather(...)`` call without ``return_exceptions=True`` is
    correctly ignored.

    Asserts the PROPERTY (no raw site outside helpers/allowlist), never a
    hardcoded TOTAL site count — this stays correct as sites are converted one
    file at a time (RED now; fully GREEN once the last recon site
    converts) and needs no edits when line numbers drift.

    Each allowlisted (path, function) key does carry a small expected PER-KEY
    site count (currently 1 for every entry, since each is a single-gather
    function today). That count is enforced too: if a second, distinct
    `gather(..., return_exceptions=True)` call later appears inside an
    already-allowlisted function, it silently matches the same key unless the
    count check catches it — so this test also fails in that case, naming the
    function and the unexpected site count.
    """

    def test_no_raw_sites_outside_helper_or_allowlist(self) -> None:
        offenders: list[str] = []
        allowlisted_sites: dict[tuple[str, str], list[int]] = {}

        for path in sorted(SRC_ROOT.rglob('*.py')):
            if path == ASYNC_UTILS_PATH:
                continue
            rel = path.relative_to(SRC_ROOT.parent).as_posix()
            for lineno, func in _gather_return_exceptions_sites(path):
                if func is None:
                    # A module-level gather(...) call can never match an
                    # ALLOWLIST entry (every key names a real enclosing
                    # function), so it is always an offender.
                    offenders.append(f'{rel}:{lineno}:<module>')
                    continue
                key = (rel, func)
                if key in DRAIN_ALLOWLIST:
                    allowlisted_sites.setdefault(key, []).append(lineno)
                else:
                    offenders.append(f'{rel}:{lineno}:{func}')

        assert not offenders, (
            'Found raw gather(..., return_exceptions=True) call site(s) outside '
            f'the helper home ({ASYNC_UTILS_PATH.relative_to(SRC_ROOT.parent).as_posix()}) '
            f'and the drain ALLOWLIST: {offenders}. Convert each to route through '
            'gather_collect / gather_or_raise (fused_memory.utils.async_utils), or '
            '— if it is a genuine teardown/drain site where a cancelled child is an '
            'expected outcome — add a justified ALLOWLIST entry.'
        )

        count_offenders: list[str] = []
        for (rel, func), linenos in allowlisted_sites.items():
            expected_count, _justification = DRAIN_ALLOWLIST[(rel, func)]
            if len(linenos) > expected_count:
                count_offenders.append(
                    f'{rel}::{func} has {len(linenos)} gather(..., return_exceptions=True) '
                    f'call site(s) at lines {linenos}, but the ALLOWLIST only expects '
                    f'{expected_count} for this function'
                )
        assert not count_offenders, (
            'An allowlisted drain function gained an unexpected extra '
            'gather(..., return_exceptions=True) call site: '
            f'{count_offenders}. The (path, function) allowlist key matches on '
            'function name alone, so a new call inside an already-allowlisted '
            'function would otherwise inherit its justification silently. Give '
            'the new site its own review — if it is a best-effort collect sweep '
            'rather than a drain, convert it to gather_collect / gather_or_raise '
            'instead of extending the allowlist.'
        )
