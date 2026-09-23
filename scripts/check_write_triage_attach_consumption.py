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

TWO BRANCHES, AND WHAT EACH DOES AND DOES NOT MEASURE. What is asserted is
the INVARIANT, not which remedy landed: requiring either branch alone would
fail a correct fix that took the other route.

    judge-side designation swap (option a). The JUDGE names its candidate back
    and the attach tracks it across two DIFFERENT designations. A measured
    consumption result: the write runs twice and the attach id follows.

    judge-module attach target (option b, as it actually exists here). The
    CALLER picks the target, so ``judge_write`` reads ``decision.canonical_id``
    — which it already holds — and hands it to ``build_judge_prompt``, leaving
    ``triage_write`` unchanged. The judge names nothing back, so no swap is
    possible however correct the module is; consumption holds BY CONSTRUCTION,
    because the announced target and the attach target are one expression. Read
    from the ref's JUDGE module, which is the only place this remedy appears.

The PASS report names the branch that held. "A swap was measured" and "option
(b) held by construction" authorise the production flag flip on different
evidence, and an operator may not be left unable to tell which they have.

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
import ast
import asyncio
import contextlib
import importlib
import inspect
import logging
import re
import sys
import textwrap
from collections.abc import Callable, Iterator
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

_JUDGE_MODULE_NAME = 'fused_memory.server.write_triage_judge'
_JUDGE_TARGET_BRANCH = 'judge-target branch'
_BUILDER_NAME = 'build_judge_prompt'
_WRITER_NAME = 'judge_write'

#: A parameter NAME that reads as designating the attach target: item 1's
#: ``_TARGET_NAME_RE`` in scripts/check_write_triage_attach_target.py, copied
#: word for word until the two probes share a helper module.
_TARGET_NAME_RE = re.compile(r'attach|target|candidate_id', re.IGNORECASE)

#: The attribute a judge-side attach target is read off the decision through,
#: and the kwarg the triage module hands its judge that decision under.
_CANONICAL_ATTR = 'canonical_id'
_DECISION_KWARG = 'decision'

#: The judge module's model-provider seam, stubbed for the drive alongside the
#: renderer (see :func:`_fed_target`).
_PROVIDER_NAME = '_call_llm'

#: Backstop bound on the one ``judge_write`` call this probe makes, for a ref
#: that reaches a model some way neither stand-in in :func:`_fed_target`
#: intercepts. Small against the gate's per-probe budget: nothing correct here
#: takes any time at all.
_JUDGE_DRIVE_TIMEOUT = 5.0

#: THE ONE MACHINE-READABLE LINE saying which branch satisfied item 5, and the
#: two names it can carry. `PASS  item 5` alone cannot tell an operator
#: whether a swap was MEASURED or whether option (b) held BY CONSTRUCTION, and
#: those authorise the production flag flip on different evidence. The gate
#: greps this prefix and quotes the rest into its own report.
#:
#: ASCII only, deliberately: the gate bounds the quoted text with a substring
#: expansion, and a cut through a multi-byte character would emit a broken one.
_BRANCH_PREFIX = 'ITEM5-BRANCH  '


class _Branch(NamedTuple):
    """A way item 5 can hold: its name in the report, and what it cannot see.

    *unchecked* is this branch's own addition to the PASS scope note, for a
    premise the branch's claim rests on and the probe does not execute.
    """

    name: str
    unchecked: tuple[str, ...] = ()


_BRANCH_SWAP = _Branch(
    'judge-side designation swap (option (a)) - a MEASURED consumption result',
)
_BRANCH_JUDGE_TARGET = _Branch(
    'judge-module attach target (option (b)) - holds BY CONSTRUCTION, '
    'not by a measured swap',
    unchecked=(
        '      Nor does it check which judge add_memory injects: the construction',
        "      holds only while tools.py::add_memory passes the judge module's own",
        '      judge_write into triage_write, and the probe drives judge_write itself.',
    ),
)


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


def _import_judge_if_present(src_root: Path) -> Any | None:
    """The ref's judge module, or None. NON-FATAL, unlike :func:`_import_triage`.

    A src-root with no judge module, or one whose import blows up, leaves the
    judge-target branch unsatisfied rather than raising :class:`_Unverifiable`.
    Every fixture repo written before option (b) carries no judge module at
    all, and a branch that ERRORED on them would have invented a new way for
    this gate to report UNVERIFIABLE against a tree it reads perfectly well.

    ``BaseException``, for :func:`main`'s reason: a ``SystemExit`` out of the
    ref's own module body is not an ``Exception``, and letting one through
    here would end the probe mid-report with the REF's exit code.

    *src_root* is already on ``sys.path`` — :func:`_import_triage` put it
    there, and the two modules are siblings in the one tree the gate extracts.
    """
    try:
        module = importlib.import_module(_JUDGE_MODULE_NAME)
    except BaseException:  # noqa: BLE001 - see the docstring; never fatal
        return None
    origin = getattr(module, '__file__', None)
    if origin is None or not _is_inside(Path(origin), src_root):
        return None
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


def _attach_id_for(module: Any, candidate: Any) -> Any:
    """The id an attach to *candidate* MUST land on, with the ref's hoist applied.

    "The attach honoured this candidate" is NOT "the attach id equals this
    candidate's id": for a child it is the PARENT's id. ``_canonical_id_of``
    documents that hoist as mandatory — attaching to a child creates a
    grandchild that can never fold under the true canonical, which reads as
    content loss — so a correct remedy that threads a designation still hoists
    it, and a probe measuring literal equality would report that remedy as
    broken.

    Read from the REF's own ``_canonical_id_of`` where it exposes one, for the
    same reason :func:`_child_kind` is read from the ref: a rule spelled here
    is a second copy of the write side's, and a copy that drifts produces
    exactly the unfoldable children that function exists to prevent. The
    metadata rule below is the fallback for a ref that renamed it, not a second
    opinion.
    """
    hoist = getattr(module, '_canonical_id_of', None)
    if callable(hoist):
        try:
            resolved = hoist(candidate)
        except Exception:  # noqa: BLE001 - a changed shape is not fatal
            resolved = None
        if isinstance(resolved, str) and resolved:
            return resolved
    meta = getattr(candidate, 'metadata', None) or {}
    if meta.get('kind') == _child_kind(module):
        parent_id = meta.get(_parent_key(module))
        if isinstance(parent_id, str) and parent_id:
            return parent_id
    return getattr(candidate, 'id', None)


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

    The recorded kwargs are where :func:`_measure` reads the slate and the
    band's decision the module handed its judge.
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


def _designated_ids(
    module: Any,
    slate: list[Any],
    band_canonical: Any,
) -> list[str]:
    """Slate ids usable as a designation. Two filters, each load-bearing.

    DISTINGUISHABLE FROM THE BAND'S OWN CANONICAL. A designation equal to it
    proves nothing, because main already attaches there.

    THEIR OWN CANONICAL ID. A child's attach target is its PARENT (see
    :func:`_attach_id_for`), so designating one asks a correct remedy for two
    contradictory things at once — honour the designation, and hoist it — and
    the swap's FAIL would read "did not track the designated candidate", which
    is an instruction to delete a mandatory hoist. The child stays on the
    SLATE: it is the band's max-cosine winner, and hoisting it is what makes
    the band canonical an id no candidate carries. It is barred only from being
    designated.

    Two are needed for the swap; see :func:`_swap_verdict`.
    """
    seen: dict[str, None] = {}
    for candidate in slate:
        ident = getattr(candidate, 'id', None)
        if not isinstance(ident, str) or not ident or ident == band_canonical:
            continue
        if _attach_id_for(module, candidate) != ident:
            continue
        seen.setdefault(ident, None)
    return list(seen)


def _measure(module: Any) -> tuple[_Run, list[Any], Any]:
    """A first run with a plain, valid verdict — what the module tells the judge.

    Returns ``(run, slate, band_canonical)``, the slate as the candidate
    OBJECTS the module handed the judge rather than as bare ids: deciding where
    an attach to one of them must land needs its metadata, not just its name.
    The verdict is a bare outcome str so the run cannot itself be rejected as an
    unrecognised payload; what is being measured here is the module's inputs,
    not its consumption.
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
    slate = [
        candidate for candidate in call.get('candidates') or ()
        if isinstance(getattr(candidate, 'id', None), str)
    ]
    band_canonical = getattr(call.get('decision'), 'canonical_id', None)
    return run, slate, band_canonical


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

    EXACT equality is right precisely BECAUSE the pool is filtered. Every
    designation :func:`_designated_ids` yields is already its own canonical id,
    so a remedy that hoists — as ``_canonical_id_of`` obliges it to — lands on
    the designation itself and passes here unaltered. Also accepting the
    hoisted form would therefore be dead code, and it would blur what a FAIL
    means by blessing a module that never hoists at all.
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


class _JudgeAborted(Exception):
    """Raised out of the recorder to stop ``judge_write`` at the prompt."""


class _PromptRecorder:
    """Stands in for ``build_judge_prompt``: records its call, then aborts.

    It never returns, because ``judge_write`` posts whatever it renders to a
    model provider. Raising here unwinds the call before any request exists.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        raise _JudgeAborted


def _refuse_provider_call(*_args: Any, **_kwargs: Any) -> Any:
    """Stands in for the provider seam: no model request, whatever reached it."""
    raise _JudgeAborted


@contextlib.contextmanager
def _replaced(module: Any, name: str, stand_in: Any) -> Iterator[None]:
    """``module.<name>`` is *stand_in* for the duration, if the module has one."""
    if not hasattr(module, name):
        yield
        return
    original = getattr(module, name)
    setattr(module, name, stand_in)
    try:
        yield
    finally:
        setattr(module, name, original)


class _Fed(NamedTuple):
    """What ``judge_write`` handed a target-named parameter of the renderer."""

    name: str
    value: Any
    #: Is *value* the id the write will attach to? A target fed SOMETHING is
    #: not a target fed the RIGHT thing, and the two want different remedies.
    matched: bool


def _usable_parameters(fn: Any) -> list[Any]:
    """*fn*'s parameters that can be supplied by position or by keyword."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return []
    kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    return [p for p in params if p.kind in kinds]


def _target_parameter_names(fn: Any) -> list[str]:
    """Names beyond the first two that read as designating the attach target.

    The first two are the new entry and the slate. ALL of the rest are
    returned rather than just the third, for item 1's measured reason: the
    POSITION of a target parameter is a mechanism, and a fix spelled
    ``build_judge_prompt(content, candidates, *, verdict_words=None,
    attach_target_id=None)`` must not be read as having none.
    """
    return [
        p.name for p in _usable_parameters(fn)[2:] if _TARGET_NAME_RE.search(p.name)
    ]


def _judge_decision(module: Any, canonical_id: str) -> Any:
    """The ref's own ``BandDecision``, carrying a canonical id the probe chose.

    The ref's class rather than a stand-in, so what ``judge_write`` reads is
    the shape it reads in production. The namespace is the fallback for a ref
    that renamed the dataclass — a mechanism this gate may not pin.
    """
    cls = getattr(module, 'BandDecision', None)
    outcome = getattr(module, 'OUTCOME_JUDGE', 'judge')
    if cls is not None:
        try:
            return cls(outcome, canonical_id, _CHILD_SCORE, _T_HIGH, _T_LOW)
        except Exception:  # noqa: BLE001 - a changed shape is not fatal
            pass
    return SimpleNamespace(
        outcome=outcome,
        canonical_id=canonical_id,
        similarity=_CHILD_SCORE,
        t_high=_T_HIGH,
        t_low=_T_LOW,
    )


def _drive_judge_write(writer: Any, decision: Any, slate: list[Any]) -> None:
    """Run ``judge_write`` far enough to see what it tells the renderer.

    Whatever the REF raises is swallowed, the stand-ins' own aborts included:
    what is measured is the recorder's log, and a judge that blows up against
    this probe's fake service has told it nothing either way — which is what
    :func:`_statically_fed_target` then covers. ``SystemExit`` is the ref's
    too. ``KeyboardInterrupt`` is the operator's, and is let through to
    :func:`main`, which fails closed on it.
    """
    service = _Service(_SearchResults())
    with contextlib.suppress(Exception, SystemExit, asyncio.CancelledError):
        asyncio.run(
            asyncio.wait_for(
                writer(
                    memory_service=service,
                    content=_NEW_ENTRY,
                    project_id=_PROJECT_ID,
                    decision=decision,
                    candidates=slate,
                ),
                _JUDGE_DRIVE_TIMEOUT,
            ),
        )


def _fed_in_call(
    builder: Any,
    call: tuple[tuple[Any, ...], dict[str, Any]],
    targets: list[str],
    wanted: str,
) -> _Fed | None:
    """What one recorded render call handed a target parameter, if anything.

    Bound against the REAL builder's signature, so a target supplied by
    POSITION counts as much as one supplied by keyword — which of the two a
    remedy picks is a spelling, and this gate may not pin one.
    """
    args, kwargs = call
    try:
        bound = inspect.signature(builder).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    for name in targets:
        if name in bound.arguments:
            value = bound.arguments[name]
            return _Fed(name, value, value == wanted)
    return None


def _called_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_decision_canonical_id(expr: ast.AST) -> bool:
    """Is *expr* exactly the decision's canonical id, read off ``decision`` itself?

    ``decision.canonical_id``, or ``getattr(decision, 'canonical_id'[, default])``.
    ``decision`` is the kwarg the triage module hands its judge the band's
    decision under. A canonical id read off anything else — a slate candidate
    — is not the id the write attaches to.
    """
    if isinstance(expr, ast.Attribute):
        return expr.attr == _CANONICAL_ATTR and _is_name(expr.value, _DECISION_KWARG)
    if not (
        isinstance(expr, ast.Call)
        and _is_name(expr.func, 'getattr')
        and len(expr.args) in (2, 3)
        and not expr.keywords
    ):
        return False
    owner, attr = expr.args[:2]
    return (
        _is_name(owner, _DECISION_KWARG)
        and isinstance(attr, ast.Constant)
        and attr.value == _CANONICAL_ATTR
    )


def _canonical_id_aliases(tree: ast.AST) -> frozenset[str]:
    """Locals EVERY binding of which is the decision's canonical id.

    main spells it ``attach_target_id = getattr(decision, 'canonical_id',
    None)`` and passes that local one call later. A local bound to anything
    else anywhere in the function may not hold the canonical id where it is
    passed, and working out which binding reaches the call would be a
    dataflow analysis.
    """
    direct: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_decision_canonical_id(node.value):
            direct.update(id(target) for target in node.targets)
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_decision_canonical_id(node.value)
        ):
            direct.add(id(node.target))
    bound_only_directly: dict[str, bool] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound_only_directly[node.id] = (
                bound_only_directly.get(node.id, True) and id(node) in direct
            )
    return frozenset(name for name, only in bound_only_directly.items() if only)


def _statically_fed_target(writer: Any, targets: list[str]) -> str | None:
    """The target parameter *writer*'s SOURCE feeds the decision's canonical id.

    The fallback for a ``judge_write`` that raises before it renders, leaving
    the recorder nothing to see. main's does not: under the probe's fake
    service it renders first, and the dynamic route decides it. Only a keyword
    whose value IS the decision's canonical id, or a local aliased directly
    from it, counts — an expression that merely mentions one is not a feed.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(writer)))
    except (OSError, TypeError, SyntaxError, ValueError):
        return None
    aliases = _canonical_id_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _called_name(node.func) != _BUILDER_NAME:
            continue
        for keyword in node.keywords:
            value = keyword.value
            fed = _is_decision_canonical_id(value) or (
                isinstance(value, ast.Name) and value.id in aliases
            )
            if keyword.arg in targets and fed:
                return keyword.arg
    return None


def _fed_target(
    judge_module: Any,
    builder: Any,
    writer: Any,
    targets: list[str],
    wanted: str,
    decision: Any,
    slate: list[Any],
) -> _Fed | None:
    """What ``judge_write`` feeds the renderer's attach target, or None.

    None means NOTHING was fed: neither a measured render call nor the source
    of one hands a target-named parameter anything at all.

    The DYNAMIC route first, because it measures rather than reads; the static
    one only where the dynamic route never saw a render, so a module that
    demonstrably renders is judged on what it actually passed.

    Both the renderer and the provider seam are replaced for the drive. The
    recorder only intercepts a renderer ``judge_write`` looks up as a module
    global at call time; one bound any other way renders for real, and the
    provider stand-in is what then stops the request.
    """
    recorder = _PromptRecorder()
    with (
        _replaced(judge_module, _BUILDER_NAME, recorder),
        _replaced(judge_module, _PROVIDER_NAME, _refuse_provider_call),
    ):
        _drive_judge_write(writer, decision, slate)
    fed = [
        seen
        for seen in (
            _fed_in_call(builder, call, targets, wanted) for call in recorder.calls
        )
        if seen is not None
    ]
    for seen in fed:
        if seen.matched:
            return seen
    if fed:
        return fed[0]
    if recorder.calls:
        return None
    name = _statically_fed_target(writer, targets)
    return _Fed(name, wanted, True) if name is not None else None


def _judge_target_branch(
    triage_module: Any,
    judge_module: Any,
    band_canonical: Any,
    observed: Any,
    slate: list[Any],
) -> tuple[bool, str]:
    """Option (b) as it exists HERE: does the judge module name the attach target?

    Returns ``(satisfied, report line)``. The line is emitted whether or not
    the branch holds, so a reader can see it was EVALUATED rather than skipped.

    Under option (b) the caller picks the attach target and tells the judge, so
    the judge names nothing back and the judge-side swap cannot hold however
    correct the module is. Requiring the swap would FAIL a correct fix and
    re-block task 3169 — this gate family's false-FAIL disease. Measured before
    this branch existed: with option (b) on main, item 1 PASSed and item 5
    reported a consumption defect the run had never measured.

    THREE conditions, each rejecting a different near-miss: the renderer can be
    TOLD (a signature); ``judge_write`` actually TELLS it the id the write will
    use (consumption, not a widened signature — this is also what keeps item
    1's target-carrying judge fixtures, none of which defines ``judge_write``,
    inert here); and the write LANDS there, without which a triage module that
    ignores everything would ride the judge module's signature to a PASS.
    """
    if judge_module is None:
        return False, (
            f'{_JUDGE_TARGET_BRANCH}: --src-root carries no importable '
            f'{_JUDGE_MODULE_NAME}, so nothing here could read a judge-side '
            'attach target'
        )
    builder = getattr(judge_module, _BUILDER_NAME, None)
    writer = getattr(judge_module, _WRITER_NAME, None)
    if not callable(builder) or not callable(writer):
        return False, (
            f'{_JUDGE_TARGET_BRANCH}: the judge module exposes no callable '
            f'{_BUILDER_NAME}/{_WRITER_NAME} pair, so nothing here could read '
            'a judge-side attach target'
        )
    if not isinstance(band_canonical, str) or not band_canonical:
        # Before the feed test, and load-bearing: `value == wanted` against a
        # `wanted` of None would match every target parameter that simply
        # DEFAULTED to None, i.e. every widened signature nothing ever fed.
        return False, (
            f'{_JUDGE_TARGET_BRANCH}: the band named no canonical id '
            f'({band_canonical!r}), so there was no attach target for the '
            'judge to be told about'
        )
    targets = _target_parameter_names(builder)
    if not targets:
        return False, (
            f'{_JUDGE_TARGET_BRANCH}: {_BUILDER_NAME} takes no attach-target '
            'parameter beyond the new entry and the slate, so the renderer '
            'cannot be told which candidate the attach will touch'
        )
    decision = _judge_decision(triage_module, band_canonical)
    fed = _fed_target(
        judge_module, builder, writer, targets, band_canonical, decision, slate,
    )
    if fed is None:
        return False, (
            f'{_JUDGE_TARGET_BRANCH}: {_BUILDER_NAME} accepts {targets!r}, but '
            f'{_WRITER_NAME} never feeds one the decision canonical '
            f'{band_canonical!r} — a widened signature is not consumption'
        )
    if not fed.matched:
        return False, (
            f'{_JUDGE_TARGET_BRANCH}: {_WRITER_NAME} feeds {fed.name!r} '
            f"{fed.value!r}, which is not the decision's canonical id "
            f'{band_canonical!r} — the model is told to reason about a '
            'candidate the write will not attach to'
        )
    if observed != band_canonical:
        return False, (
            f'{_JUDGE_TARGET_BRANCH}: {_WRITER_NAME} feeds {fed.name!r} the '
            f'decision canonical {band_canonical!r}, but the attach landed on '
            f'{observed!r} — the write did not use the target the judge was '
            'told about'
        )
    return True, (
        f'{_JUDGE_TARGET_BRANCH}: satisfied — {_WRITER_NAME} feeds '
        f"{_BUILDER_NAME}'s {fed.name!r} the decision canonical "
        f'{band_canonical!r}, and the attach landed there'
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


def _pass_scope_note(branch: _Branch) -> list[str]:
    """What a PASS on *branch* deliberately does NOT prove.

    Item 5 asserts at ``BandDecision.canonical_id`` — the value
    ``tools.py::add_memory`` consumes verbatim as ``attached_to``. It stops
    there: the remaining hops live inline in that MCP tool body with no
    callable seam, and standing up a real ``memory_service`` is not something a
    bounded before_done predicate can do.

    Emitted on the PASS path only. That is the run an operator acts on to flip
    a production flag, and it is also the report whose window is uncontended: a
    FAIL's window belongs to the remedy.
    """
    return [
        '      NOTE this gate asserts at BandDecision.canonical_id — the value',
        '      tools.py::add_memory consumes verbatim as `attached_to`. It does NOT',
        '      execute the stamp, so it does not show that the write puts that id in',
        "      PARENT_ID_KEY, and a later change to add_memory's own target selection",
        '      would still pass here.',
        *branch.unchecked,
        '      Confirm that separately before flipping.',
    ]


def _pass(out: list[str], branch: _Branch) -> int:
    """Report a satisfied invariant, NAMING the branch that satisfied it.

    One exit for both branches, so the marker the gate greps, the branch line
    it quotes and the scope note an operator reads can never be emitted by one
    branch and forgotten by the other.
    """
    out.append(_BRANCH_PREFIX + branch.name)
    out.append(_PASS_MARKER)
    out.extend(_pass_scope_note(branch))
    return EXIT_OK


def _probe(src_root: Path, extra_paths: list[Path], out: list[str]) -> int:
    out.append(
        f'write_triage attach-consumption probe — src-root={src_root}',
    )
    module = _import_triage(src_root, extra_paths)
    out.append(f'triage module: {getattr(module, "__file__", "<unknown>")}')
    judge_module = _import_judge_if_present(src_root)

    measured, slate, band_canonical = _measure(module)
    slate_ids = [candidate.id for candidate in slate]
    out.append(
        f'slate: {slate_ids!r} — band canonical {band_canonical!r}',
    )
    observed = getattr(measured.decision, 'canonical_id', None)

    # The judge-target branch first: it is decided from the run already
    # measured plus one judge call, while the swap drives the write eight more
    # times -- and a module satisfying it has no judge-side designation for
    # the swap to find.
    judged, judge_line = _judge_target_branch(
        module, judge_module, band_canonical, observed, slate,
    )
    out.append(judge_line)
    if judged:
        return _pass(out, _BRANCH_JUDGE_TARGET)

    usable = _designated_ids(module, slate, band_canonical)
    if len(usable) < 2:
        # UNVERIFIABLE, never PASS. With fewer than two candidates that are
        # distinguishable from the band's own canonical there is nothing here
        # that could tell a correct fix from a hard-coded position.
        raise _Unverifiable(
            f'the slate {slate_ids!r} carries fewer than two candidates that '
            f'are their own canonical id and distinguishable from the band '
            f'canonical {band_canonical!r}, so no swap is possible and nothing '
            'here could tell a correct fix from an attach that simply used a '
            'fixed slot',
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
        return _pass(out, _BRANCH_SWAP)

    # One reason per spelling, so the count is bounded by _SPELLINGS itself.
    out.append(f'spellings tried: {"; ".join(attempts)}')
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
