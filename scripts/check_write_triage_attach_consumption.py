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
judge and asking whether the attach id is determined by the candidate the judge
reasoned about.

TWO BRANCHES, BECAUSE TWO REMEDIES ARE OPEN. Under option (a) the JUDGE names
its candidate back and the write honours it; under option (b) the CALLER picks
the attach target and tells the judge which candidate it is reasoning about, so
the judge names nothing back and no judge-side designation exists to track.
Both satisfy the invariant, and the probe PASSes on either — what is asserted
is the invariant, not which remedy landed. Requiring the judge-side branch
alone would fail a correct option (b) and re-block task 3169, which is the
false-FAIL class this gate family was rewritten to remove. Item 1 is structured
the same way and for the same reason.

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
import inspect
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

EXIT_OK = 0
EXIT_FAIL = 1

logger = logging.getLogger(__name__)

#: Prefix for the probe's own degraded-measurement records. They are collected
#: rather than logged straight out because logging's default destination is
#: stderr, which the gate drops.
_WARN_PREFIX = 'WARN  '


class _WarnCollector(logging.Handler):
    """Divert the probe's own warnings into the END of the report.

    A warning about HOW something was measured has to reach the operator
    reading the verdict, and the two channels this probe's output survives are
    narrow: the gate drops stderr, and a report read through a tail keeps only
    its end. So the records go on stdout, and last.
    """

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(_WARN_PREFIX + record.getMessage())


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

_ANNOUNCED_BRANCH = 'announced-target branch'
_ANNOUNCEMENT_IGNORED = 'the attach did not land on the announced target'

#: The kwargs main ALREADY hands the judge. An announcement read out of any of
#: these is one main already makes -- `decision.canonical_id` is precisely the
#: id main already attaches to -- so admitting them would let the branch hold
#: on a codebase where nothing changed at all. Excluding them is what makes the
#: branch evidence rather than decoration.
_JUDGE_KWARGS = frozenset({
    'memory_service',
    'content',
    'project_id',
    'decision',
    'candidates',
})


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
    #: Fail-opens the module recorded on THIS run's own fresh counter.
    fail_opens: int
    error: str | None


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _decision_shape_error(decision: Any) -> str | None:
    """Why *decision* cannot be read as a ``BandDecision``, or None if it can.

    Duck-typed on the two fields this probe reads rather than on the class.
    An ``isinstance`` check against a class imported from the same bare tree
    would add nothing and would fail any ref that renamed the dataclass, which
    is a mechanism this gate may not pin.

    Without this check a returned shape carrying no ``canonical_id`` reads as
    an attach id of None on every run — neither the band's top-1 nor a
    designation — so the probe would report NOT CONSUMED and send an operator
    to fix a defect this run never measured. A wrong diagnosis, not merely a
    wrong verdict.
    """
    missing = [f for f in ('outcome', 'canonical_id') if not hasattr(decision, f)]
    if not missing:
        return None
    return (
        f'triage_write returned {decision!r}, which exposes no '
        f'{"/".join(missing)} — nothing here measured an attach'
    )


def _drive(module: Any, judge: _FakeJudge) -> _Run:
    """Execute ``triage_write`` once against the fixture slate."""
    triage_write = _require(module, 'triage_write')
    counter, read_fail_opens = _make_counter(module)
    results = _fixture_results(_parent_key(module), _child_kind(module))
    service = _Service(results)

    def failed(reason: str) -> _Run:
        return _Run(None, judge.calls, read_fail_opens(), reason)

    try:
        pending = triage_write(
            service,
            content=_NEW_ENTRY,
            project_id=_PROJECT_ID,
            counter=counter,
            judge=judge,
        )
        # Checked BEFORE awaiting: a plain `def` returns its value directly, so
        # `await`ing it raises a TypeError that reads like a defect inside the
        # write path rather than like "the write path never ran".
        if not inspect.isawaitable(pending):
            return failed(
                f'triage_write(...) returned {pending!r}, which is not '
                'awaitable — nothing here executed the write path',
            )
        decision = asyncio.run(_await(pending))
    except Exception as exc:  # noqa: BLE001 - attributed to this run, not global
        return failed(f'triage_write raised: {exc!r}')
    shape_error = _decision_shape_error(decision)
    if shape_error is not None:
        return failed(shape_error)
    return _Run(decision, judge.calls, read_fail_opens(), None)


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


def _make_counter(module: Any) -> tuple[Any, Callable[[], int]]:
    """A FRESH fail-open counter for one run, and a reader for its count.

    THE REF'S OWN CLASS FIRST, so what is measured is the ref's accounting
    rather than a reimplementation of it: ``_record_fail_open`` increments it on
    exactly the paths that swallowed the designation. The counting stand-in is
    the fallback for a ref whose class is absent, whose constructor changed, or
    whose ``live_count`` no longer reports an int — none of which is a reason to
    stop measuring.
    """
    cls = getattr(module, 'TriageFailOpenCounter', None)
    if cls is not None:
        try:
            counter = cls()
            reader = counter.live_count
            if isinstance(reader(), int):
                return counter, reader
        except Exception:  # noqa: BLE001 - a changed shape is not fatal
            pass
    # Outside the branch above, so an ABSENT class warns too. Falling back is
    # not a reason to stop measuring, but it does change what was measured —
    # the stand-in's accounting rather than the ref's — and a gate that
    # authorises a production flag flip may not degrade quietly.
    logger.warning(
        "the ref's TriageFailOpenCounter is absent or unusable, so fail-opens "
        "were counted with a stand-in rather than with the ref's own accounting",
    )
    fallback = _CountingCounter()
    return fallback, fallback.live_count


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
    if run.fail_opens:
        raise _Unverifiable(
            f'the module recorded {run.fail_opens} fail-open(s) on a plain, '
            f'valid {_ATTACH_OUTCOME!r} verdict, so this run measures a '
            'degraded write path rather than what the attach consumes',
        )
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
        # BEFORE the tracking test, and structurally rather than by inferring
        # from the outcome. A fail-open run returns canonical_id=None, which is
        # not the band's top-1 either — so a check that only asked "did the
        # attach avoid the band canonical?" would read main's own
        # `verdict not in TRIAGE_OUTCOMES` arm as CONSUMED. Inferring from the
        # outcome is no better: `stored` is also a legitimate judge verdict.
        if run.fail_opens:
            return (
                f'FAIL-OPEN ({run.fail_opens} recorded) — the module rejected '
                f'this designation and fell open to outcome '
                f'{getattr(run.decision, "outcome", None)!r}, so the '
                'designation was swallowed rather than consumed'
            )
        observed = getattr(run.decision, 'canonical_id', None)
        if observed != designated:
            return (
                f'the attach landed on {observed!r}; it did not track the '
                f'designated candidate {designated!r}'
            )
    return None


def _announced_ids(
    call: dict[str, Any],
    eligible: list[str],
) -> list[tuple[str, str]]:
    """Slate candidates the module NAMED to the judge, as ``(kwarg, id)`` pairs.

    An announcement counts only when it is BOTH beyond :data:`_JUDGE_KWARGS`
    and drawn from *eligible* — the ids that are distinguishable from the
    band's own canonical, the same set the swap draws its designations from and
    for the same reason. Announcing the id main already attaches to says
    nothing about whether the announcement was honoured.

    A candidate may be named as its id or as the object itself; both are read,
    because which one a remedy would pass is a mechanism this probe may not pin.
    """
    found = []
    for name, value in call.items():
        if name in _JUDGE_KWARGS:
            continue
        ident = value if isinstance(value, str) else getattr(value, 'id', None)
        if isinstance(ident, str) and ident in eligible:
            found.append((name, ident))
    return found


def _announced_target_branch(
    run: _Run,
    eligible: list[str],
) -> tuple[bool, str]:
    """Did the attach land on a candidate the module itself announced?

    Returns ``(satisfied, report line)``. The line is emitted whether or not
    the branch holds, so a reader can see the branch was EVALUATED rather than
    skipped — the non-vacuity of "main does not satisfy it" is only legible if
    main's run says so out loud.
    """
    announced = _announced_ids(run.calls[0], eligible)
    observed = getattr(run.decision, 'canonical_id', None)
    if not announced:
        return False, (
            f'{_ANNOUNCED_BRANCH}: the judge was told no slate candidate beyond '
            f'{sorted(_JUDGE_KWARGS)}, so nothing was announced for the attach '
            'to honour'
        )
    honoured = [name for name, ident in announced if ident == observed]
    if honoured:
        return True, (
            f'{_ANNOUNCED_BRANCH}: satisfied — triage_write announced '
            f'{observed!r} to the judge via {honoured[0]!r}, and the attach '
            'landed there'
        )
    return False, (
        f'{_ANNOUNCED_BRANCH}: triage_write announced '
        f'{_first_few([f"{name}={ident!r}" for name, ident in announced])}, but '
        f'the attach landed on {observed!r} — {_ANNOUNCEMENT_IGNORED}'
    )


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


def _pass_scope_note() -> list[str]:
    """What a PASS deliberately does NOT prove.

    Item 5 asserts at ``BandDecision.canonical_id`` — the value
    ``tools.py::add_memory`` consumes verbatim as ``attached_to``. It stops
    there: the remaining hops live inline in that MCP tool body with no
    callable seam, and standing up a real ``memory_service`` is not something a
    bounded before_done predicate can do. So a later change to add_memory's own
    target selection would still pass this gate — an honest smaller claim, said
    out loud rather than left for a reader to infer from the absence of one.

    Emitted on the PASS path only. That is the run an operator acts on to flip
    a production flag, and it is also the report whose window is uncontended: a
    FAIL's window belongs to the remedy.
    """
    return [
        '      NOTE this gate asserts at BandDecision.canonical_id — the value',
        '      tools.py::add_memory consumes verbatim as `attached_to`. It does NOT',
        '      execute the stamp, so it does not show that the write puts that id in',
        "      PARENT_ID_KEY, and a later change to add_memory's own target selection",
        '      would still pass here. Confirm that separately before flipping.',
    ]


def _probe(src_root: Path, extra_paths: list[Path], out: list[str]) -> int:
    out.append(
        f'write_triage attach-consumption probe — src-root={src_root}',
    )
    module = _import_triage(src_root, extra_paths)
    out.append(f'triage module: {getattr(module, "__file__", "<unknown>")}')

    measured, slate_ids, band_canonical = _measure(module)
    out.append(
        f'slate: {slate_ids!r} — band canonical {band_canonical!r}',
    )
    usable = _designated_ids(slate_ids, band_canonical)

    # The option-(b) branch first: it is decided by the run already measured,
    # and a module that satisfies it has no judge-side designation for the swap
    # to find, so searching for one would only spend the report on four
    # spellings none of which could ever have held.
    announced, announced_line = _announced_target_branch(measured, usable)
    out.append(announced_line)
    if announced:
        out.append(_PASS_MARKER)
        out.extend(_pass_scope_note())
        return EXIT_OK

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
        out.extend(_pass_scope_note())
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
    # Argparse stays OUTSIDE the try below: `--help` and a missing --src-root
    # are argparse's own exit codes to own, and reporting a usage error as an
    # UNVERIFIABLE invariant would name a defect in the ref for a defect in the
    # invocation.
    args = parser.parse_args(argv)

    out: list[str] = []
    warnings: list[str] = []
    # propagate=False so the collector is the ONLY destination: logging's
    # lastResort handler would otherwise also write each record to stderr,
    # which the gate drops, leaving a duplicate nobody reads.
    logger.addHandler(_WarnCollector(warnings))
    logger.propagate = False

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
    except BaseException as exc:  # noqa: BLE001 - see below; nothing may escape
        # BaseException, NOT Exception, and this is the whole point of the arm.
        # A `SystemExit` out of the ref's own module body is not an Exception,
        # so an `except Exception` lets it through: the interpreter then exits
        # with the REF's code — 0 for `SystemExit(0)` — having printed nothing
        # at all, and a gate that greps stdout for a marker reads silence plus
        # rc=0 as a PASS. That is a measured escape on the item-1 probe, not a
        # hypothetical one, and it is the worst failure this probe has: it
        # authorises a production flag flip on a run that measured nothing.
        out.append(
            f'FAIL  UNVERIFIABLE: the probe raised {exc!r} while evaluating the '
            'invariant',
        )
        out.append(
            '      Failing closed — an unverifiable invariant is not a satisfied one.',
        )
        rc = EXIT_FAIL

    # LAST, and deduplicated: every run builds its own counter, so one degraded
    # measurement would otherwise repeat itself once per run and crowd the
    # verdict out of a tail-truncated report.
    out.extend(dict.fromkeys(warnings))
    sys.stdout.write('\n'.join(out) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
