#!/usr/bin/env python3
"""Probe whether the judge-bound candidate is CONSUMED by the attach.

Consumed by ``scripts/check_write_triage_flip_preconditions.sh`` item 5, the
sibling of item 1's ``scripts/check_write_triage_attach_target.py``.

THE INVARIANT
-------------
Item 1 asserts that the judge path can BIND a verdict to a determinate
candidate. It stops there: it does not execute the attach, so nothing it
measures shows that the binding is CONSUMED. A change that only widens the
parse contract therefore opens item 1 while the attach still lands on the
band's top-1 — the very harm item 1 describes, still live.

This probe closes that gap by EXECUTING ``triage_write`` with an injected fake
judge and asking whether the id the judge designated is the id the returned
``BandDecision`` attaches to.

WHY THE JUDGE SEAM. ``triage_write(..., judge=...)`` is a real injection point
the module's own contract tests already use, and ``memory_service`` is
duck-typed all the way down (every config read goes through a
``getattr``-at-every-hop resolver). So no LLM call, no network and no real
service are needed — which is what makes this fit a bounded before_done
predicate at all.

WHY THE SLATE HOISTS A PARENT. The fixture's highest-scoring candidate is an
evidence CHILD whose ``parent_id`` points at a record that never appears in the
slate, so ``_canonical_id_of`` hoists it and the band's canonical is an id NO
candidate carries as its own. That is what makes "the attach followed the
judge's designation" separable from "the attach used the band's winner".

EXIT-CODE CONTRACT
    0  the invariant holds.
    1  it does not hold, OR it could not be verified. An unverifiable
       invariant is not a satisfied one, and this gate protects a production
       flag flip.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

EXIT_OK = 0
EXIT_FAIL = 1

logger = logging.getLogger(__name__)

_MODULE_NAME = 'fused_memory.server.write_triage'

#: The hoisted parent the band picks as its canonical. It deliberately never
#: appears as a candidate of its own, so an attach that lands here is
#: demonstrably following the BAND rather than anything the judge said.
_CANONICAL_ID = 'parent-1'
_CHILD_ID = 'child-1'

#: Six ordinary candidates plus the evidence child. The child scores HIGHEST,
#: which is what makes it the band's winner and therefore the thing
#: ``_canonical_id_of`` hoists.
_ORDINARY_COUNT = 6
_TOP_SCORE = 0.60
_SCORE_STEP = 0.05
_CHILD_SCORE = 0.72

#: Band edges chosen so the child's cosine lands STRICTLY between them and the
#: write routes to the judge. Real floats, never defaults: ``t_low is None``
#: means uncalibrated and short-circuits to ``stored``, and ``t_high is None``
#: is a legitimate empty deterministic band — neither reaches the judge slot on
#: the terms this probe needs.
_T_HIGH = 0.85
_T_LOW = 0.50

_CANDIDATE_K = 20
_PROJECT_ID = 'write-triage-consumption-probe'
_NEW_ENTRY = 'A new memory entry submitted for triage by this probe.'

#: The outcome the fake judge returns. NOT ``stored``: main maps ``stored`` to
#: ``canonical_id = None`` by design (nothing was attached), so a probe that
#: designated a candidate and asked for ``stored`` would be asking the module
#: to contradict itself.
_ATTACH_OUTCOME = 'restated'

_PASS_MARKER = 'PASS  the judge-bound candidate is CONSUMED by the attach'
_FAIL_MARKER = 'FAIL  the judge-bound candidate is NOT CONSUMED by the attach'


class _Unverifiable(Exception):
    """The invariant could not be evaluated. Fails closed, never passes."""


class _Candidate:
    """A duck-typed ``MemoryResult`` stand-in.

    ``_cosine_of`` reads the per-store cosine out of ``metadata['store_score']``
    and ``_canonical_id_of`` reads ``metadata['kind']``/``metadata[PARENT_ID_KEY]``,
    so ``.id``/``.content``/``.metadata`` is the whole contract. Constructing one
    avoids importing ``MemoryResult``, whose module pulls in third-party deps a
    bare extracted tree may not have.
    """

    __slots__ = ('content', 'id', 'metadata')

    def __init__(self, ident: str, content: str, metadata: dict[str, Any]) -> None:
        self.id = ident
        self.content = content
        self.metadata = metadata

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f'<candidate {self.id}>'


class _SearchResults(list):
    """``MemoryService.SearchResults`` stand-in: a list that carries degradation.

    ``triage_write`` reads ``degraded`` before banding — a degraded retrieval is
    a fail-open, not an empty corpus — so the attribute has to exist and be
    False for the probe's run to reach the judge at all.
    """

    degraded = False
    failed_stores = ()


class _Service:
    """A duck-typed ``MemoryService``: a config tree plus an async ``search``."""

    def __init__(self, results: _SearchResults) -> None:
        self.config = SimpleNamespace(
            write_triage=SimpleNamespace(
                enabled=True,
                candidate_k=_CANDIDATE_K,
                t_high=_T_HIGH,
                t_low=_T_LOW,
            ),
        )
        self._results = results
        self.searches: list[dict[str, Any]] = []

    async def search(self, **kwargs: Any) -> _SearchResults:
        self.searches.append(kwargs)
        return self._results


def _fixture_results(parent_key: str, child_kind: str) -> _SearchResults:
    """Six ordinary results plus the top-scoring evidence child of a hoisted parent."""
    results = _SearchResults(
        _Candidate(
            f'm{i}',
            f'ordinary candidate {i}',
            {'store_score': _TOP_SCORE - (i * _SCORE_STEP)},
        )
        for i in range(_ORDINARY_COUNT)
    )
    results.append(
        _Candidate(
            _CHILD_ID,
            'the child record that carries the evidence for the hoisted parent',
            {
                'store_score': _CHILD_SCORE,
                'kind': child_kind,
                parent_key: _CANONICAL_ID,
            },
        ),
    )
    return results


def _is_inside(path: Path, root: Path) -> bool:
    """Is *path* under *root*, comparing PATH COMPONENTS rather than characters?"""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        # An unresolvable path is not demonstrably inside the root, and this
        # gate fails closed.
        return False


def _import_triage(src_root: Path, extra_paths: list[Path]) -> Any:
    """Import the triage module out of *src_root*, shadowing any installed copy.

    *extra_paths* carries first-party trees the module imports but that
    ``--src-root`` does not contain — ``shared/src`` for ``shared.storm_counter``.
    They go on ``sys.path`` BEFORE *src_root*, so *src_root* ends up first and
    the assertion is made against the ref rather than against whatever happens
    to be installed in the interpreter running this probe.
    """
    if not src_root.is_dir():
        raise _Unverifiable(f'--src-root is not a directory: {src_root}')
    for extra in extra_paths:
        sys.path.insert(0, str(extra))
    sys.path.insert(0, str(src_root))
    try:
        module = importlib.import_module(_MODULE_NAME)
    except Exception as exc:  # noqa: BLE001 - any import failure is unverifiable
        raise _Unverifiable(
            f'cannot import {_MODULE_NAME} from {src_root}: {exc!r}',
        ) from exc
    origin = getattr(module, '__file__', None)
    # PATH CONTAINMENT, not a string prefix: `str(a).startswith(str(b))` also
    # accepts a SIBLING whose name extends the root -- `<root>-installed/...`
    # for `--src-root <root>` -- so the probe would report on a module the ref
    # never shipped, which is the substitution this guard exists to catch.
    if origin is None or not _is_inside(Path(origin), src_root):
        raise _Unverifiable(
            f'{_MODULE_NAME} resolved to {origin!r}, which is outside --src-root '
            f'{src_root} — the probe would be testing the wrong tree',
        )
    return module


def _require(module: Any, name: str) -> Any:
    attr = getattr(module, name, None)
    if attr is None:
        raise _Unverifiable(f'{_MODULE_NAME} exposes no {name}')
    return attr


def _child_kind(module: Any) -> str:
    """A ``kind`` the ref's own ``_canonical_id_of`` treats as a child.

    Read from the module rather than spelled here: the hoist only fires for a
    kind in its ``CHILD_KINDS``, and a probe that hardcoded one would silently
    stop hoisting — and therefore stop measuring the case that matters — if the
    vocabulary moved.
    """
    kinds = getattr(module, 'CHILD_KINDS', None)
    for kind in sorted(kinds) if isinstance(kinds, (frozenset, set)) else ():
        if isinstance(kind, str) and kind:
            return kind
    return 'sighting'


def _parent_key(module: Any) -> str:
    key = getattr(module, 'PARENT_ID_KEY', None)
    return key if isinstance(key, str) and key else 'parent_id'


class _VerdictObject:
    """A designating verdict spelled as a small object."""

    __slots__ = ('candidate_id', 'outcome')

    def __init__(self, outcome: str, candidate_id: str) -> None:
        self.outcome = outcome
        self.candidate_id = candidate_id

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f'<verdict {self.outcome} -> {self.candidate_id}>'


class _Spelling(NamedTuple):
    """One plausible wire shape for "the verdict names its own candidate"."""

    label: str
    payload: Callable[[str, str], Any]


#: THE DESIGNATION-SHAPE SEARCH. Option (a) has not landed, so no single
#: spelling may be pinned: requiring one would fail a correct fix that chose
#: another, which is the false-FAIL class this gate family exists to remove.
#: Every shape is tried and the FIRST whose consumption test holds wins. A
#: spelling the implementation cannot consume simply is not the winner and
#: costs nothing; only a module where NO spelling holds fails.
#:
#: The bare outcome str is included DELIBERATELY as a control that must never
#: satisfy on its own — it designates no candidate, so a module that widened
#: nothing must not open the gate on it. A control nobody exercises proves
#: nothing, so it is tried and reported like any other.
_SPELLINGS: tuple[_Spelling, ...] = (
    _Spelling('bare outcome str', lambda outcome, _ident: outcome),
    _Spelling(
        '(outcome, candidate_id) tuple',
        lambda outcome, ident: (outcome, ident),
    ),
    _Spelling(
        '{outcome, candidate_id} mapping',
        lambda outcome, ident: {'outcome': outcome, 'candidate_id': ident},
    ),
    _Spelling(
        'object with .outcome/.candidate_id',
        lambda outcome, ident: _VerdictObject(outcome, ident),
    ),
)


class _FakeJudge:
    """An async judge that records its call and returns a designating verdict.

    Records every keyword argument it is handed, so the probe can measure what
    the module told the judge as well as what it did with the answer.
    """

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._payload


class _Run(NamedTuple):
    """One ``triage_write`` execution: what it returned and what it recorded."""

    decision: Any
    calls: list[dict[str, Any]]
    error: str | None


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _drive(module: Any, judge: _FakeJudge) -> _Run:
    """Execute ``triage_write`` once against the fixture slate."""
    triage_write = _require(module, 'triage_write')
    counter = _make_counter(module)
    results = _fixture_results(_parent_key(module), _child_kind(module))
    service = _Service(results)
    try:
        decision = asyncio.run(
            _await(
                triage_write(
                    service,
                    content=_NEW_ENTRY,
                    project_id=_PROJECT_ID,
                    counter=counter,
                    judge=judge,
                ),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - attributed to this run, not global
        return _Run(None, judge.calls, f'triage_write raised: {exc!r}')
    return _Run(decision, judge.calls, None)


class _CountingCounter:
    """A ``TriageFailOpenCounter`` stand-in for a ref whose class is unusable."""

    def __init__(self) -> None:
        self.records = 0

    def record(self, **_kwargs: Any) -> None:
        self.records += 1
        return None

    def drain_storm(self) -> None:
        return None

    def live_count(self) -> int:
        return self.records


def _make_counter(module: Any) -> Any:
    """A FRESH fail-open counter for one run, preferring the ref's own class."""
    cls = getattr(module, 'TriageFailOpenCounter', None)
    if cls is not None:
        try:
            return cls()
        except Exception:  # noqa: BLE001 - a changed constructor is not fatal
            logger.warning(
                'the ref\'s TriageFailOpenCounter could not be constructed; '
                'using a counting stand-in',
            )
    return _CountingCounter()


def _designated_ids(slate_ids: list[str], band_canonical: Any) -> list[str]:
    """Slate ids usable as a designation: distinguishable from the band's own.

    A designation equal to the band canonical proves nothing — main already
    attaches there — so those are dropped rather than measured. Two are needed
    for the swap; see :func:`_swap_verdict`.
    """
    seen: dict[str, None] = {}
    for ident in slate_ids:
        if isinstance(ident, str) and ident and ident != band_canonical:
            seen.setdefault(ident, None)
    return list(seen)


def _measure(module: Any) -> tuple[_Run, list[str], Any]:
    """A first run with a plain, valid verdict — what the module tells the judge.

    Returns ``(run, slate_ids, band_canonical)``. The verdict is a bare outcome
    str so the run cannot itself be rejected as an unrecognised payload; what is
    being measured here is the module's inputs, not its consumption.
    """
    judge = _FakeJudge(_ATTACH_OUTCOME)
    run = _drive(module, judge)
    if run.error is not None:
        raise _Unverifiable(run.error)
    if not run.calls:
        raise _Unverifiable(
            'the judge slot was never reached — the fixture slate did not route '
            'to the judge band, so nothing here can measure what the attach '
            'consumes',
        )
    call = run.calls[0]
    slate_ids = [getattr(c, 'id', None) for c in call.get('candidates') or ()]
    band_canonical = getattr(call.get('decision'), 'canonical_id', None)
    return run, [i for i in slate_ids if isinstance(i, str)], band_canonical


def _first_few(reasons: list[str], limit: int = 4) -> str:
    """Bounded join. The report shares a 2000-character escalation window."""
    shown = '; '.join(reasons[:limit])
    if len(reasons) > limit:
        shown += f'; …{len(reasons) - limit} more'
    return shown


def _swap_verdict(
    module: Any,
    spelling: _Spelling,
    designations: tuple[str, str],
) -> str | None:
    """None when the attach TRACKED both designations; a reason otherwise.

    THE SWAP, and why one run is not enough. A single run whose attach id
    merely differs from the band's canonical is satisfied by any hard-coded
    position — an implementation that always attaches to the last slate entry
    is not the band's top-1 either. Requiring the attach to follow TWO
    different designations makes the assertion about the DEPENDENCY rather than
    about a value, so it accepts any mechanism that genuinely threads the
    designation and rejects every fixed choice.

    Both halves are necessary: matching one designation alone could be
    coincidence, and differing between runs without matching either means the
    attach is tracking something else entirely.
    """
    for designated in designations:
        judge = _FakeJudge(spelling.payload(_ATTACH_OUTCOME, designated))
        run = _drive(module, judge)
        if run.error is not None:
            return run.error
        observed = getattr(run.decision, 'canonical_id', None)
        if observed != designated:
            return (
                f'the attach landed on {observed!r}; it did not track the '
                f'designated candidate {designated!r}'
            )
    return None


def _search_spellings(
    module: Any,
    designations: tuple[str, str],
) -> tuple[_Spelling | None, list[str]]:
    """Try every designation spelling; return the first the attach CONSUMES.

    Mirrors the discipline of item 1's ``_search_option_b``: a spelling the
    implementation ignores simply is not the winner, and the whole search is
    reported on failure so an operator sees what was tried rather than one
    arbitrary verdict.
    """
    attempts: list[str] = []
    for spelling in _SPELLINGS:
        reason = _swap_verdict(module, spelling, designations)
        if reason is None:
            return spelling, attempts
        attempts.append(f'{spelling.label} — {reason}')
    return None, attempts


def _probe(src_root: Path, extra_paths: list[Path], out: list[str]) -> int:
    out.append(
        f'write_triage attach-consumption probe — src-root={src_root}',
    )
    module = _import_triage(src_root, extra_paths)
    out.append(f'triage module: {getattr(module, "__file__", "<unknown>")}')

    _, slate_ids, band_canonical = _measure(module)
    out.append(
        f'slate: {slate_ids!r} — band canonical {band_canonical!r}',
    )
    usable = _designated_ids(slate_ids, band_canonical)
    if len(usable) < 2:
        # UNVERIFIABLE, never PASS. With fewer than two candidates that are
        # distinguishable from the band's own canonical there is nothing here
        # that could tell a correct fix from a hard-coded position.
        raise _Unverifiable(
            f'the slate {slate_ids!r} carries fewer than two candidates '
            f'distinguishable from the band canonical {band_canonical!r}, so '
            'no swap is possible and nothing here could tell a correct fix '
            'from an attach that simply used a fixed slot',
        )

    # Chosen from the MEASURED slate at runtime rather than by fixed index, and
    # taken from opposite ends so the pair is as far apart as the slate allows.
    designations = (usable[0], usable[-1])
    winner, attempts = _search_spellings(module, designations)
    if winner is not None:
        out.append(
            f'designation channel: {winner.label} — the attach tracked both '
            f'designated candidates {designations!r}',
        )
        out.append(_PASS_MARKER)
        return EXIT_OK

    out.append(f'spellings tried: {_first_few(attempts)}')
    out.append(_FAIL_MARKER)
    out.append(
        f'      band canonical {band_canonical!r}; the judge designated '
        f'{designations!r}; no designation spelling reached the attach.',
    )
    out.append(
        '      A verdict earned by one candidate is filed against another, so the',
    )
    out.append(
        '      judge-side binding item 1 asserts is not carried into the write.',
    )
    return EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--src-root',
        required=True,
        help='directory containing the fused_memory package (a fused-memory/src tree)',
    )
    parser.add_argument(
        '--extra-path',
        action='append',
        default=[],
        help=(
            'an additional sys.path entry for a first-party tree the triage '
            'module imports but --src-root does not contain (shared/src). '
            'Repeatable.'
        ),
    )
    args = parser.parse_args(argv)

    out: list[str] = []
    try:
        rc = _probe(
            Path(args.src_root),
            [Path(p) for p in args.extra_path],
            out,
        )
    except _Unverifiable as exc:
        out.append(f'FAIL  UNVERIFIABLE: {exc}')
        out.append(
            '      Failing closed — an unverifiable invariant is not a satisfied one.',
        )
        rc = EXIT_FAIL
    sys.stdout.write('\n'.join(out) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
