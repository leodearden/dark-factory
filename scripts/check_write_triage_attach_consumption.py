#!/usr/bin/env python3
"""Probe whether the judge-bound candidate is CONSUMED by the attach.

Consumed by ``scripts/check_write_triage_flip_preconditions.sh`` item 5, the
sibling of item 1's ``scripts/check_write_triage_attach_target.py``.

THE INVARIANT
-------------
Item 1 asserts that the judge path can BIND a verdict to a determinate
candidate, and never executes the attach. So a change that only widens the
parse contract opens item 1 while the attach still lands on the band's top-1.
This probe EXECUTES ``triage_write`` with an injected fake judge and asks
whether the attach id is the candidate the judge reasoned about.

TWO BRANCHES satisfy it. Requiring either alone would fail a correct fix that
took the other route.

    judge-side designation swap (option a). The JUDGE names its candidate back
    and the attach tracks it across two DIFFERENT designations: MEASURED, by
    running the write twice.

    judge-module attach target (option b, as it exists here). ``judge_write``
    hands ``build_judge_prompt`` the ``decision.canonical_id`` it already
    holds, and ``triage_write`` is unchanged. The judge names nothing back, so
    no swap is possible; consumption holds BY CONSTRUCTION, because announced
    target and attach target are one expression.

The PASS report names the branch that held, since the two authorise the
production flag flip on different evidence.

WHY THE JUDGE SEAM. ``triage_write(..., judge=...)`` is a real injection point,
and ``memory_service`` is duck-typed all the way down, so no LLM call, network
or real service is needed.

WHY THE SLATE HOISTS A PARENT. The top-scoring candidate is an evidence CHILD
whose ``parent_id`` names a record absent from the slate, so the band's
canonical is an id NO candidate carries. That separates "the attach followed
the judge" from "the attach used the band's winner".

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
import inspect
import logging
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

from _write_triage_probe_common import (
    EXIT_FAIL,
    EXIT_OK,
    PROBE_LOGGER_NAME,
    Candidate,
    ImportMode,
    Unverifiable,
    import_from_tree,
    names_a_target,
    parent_key,
    report_fail_closed,
    require,
    usable_parameters,
)

logger = logging.getLogger(PROBE_LOGGER_NAME)

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

#: Band edges the child's cosine lands STRICTLY between, so the write routes to
#: the judge. Real floats: a ``None`` edge means uncalibrated or an empty band,
#: and neither reaches the judge slot.
_T_HIGH = 0.85
_T_LOW = 0.50

_CANDIDATE_K = 20
_PROJECT_ID = 'write-triage-consumption-probe'
_NEW_ENTRY = 'A new memory entry submitted for triage by this probe.'

#: The outcome the fake judge returns. NOT ``stored``, which attaches nothing
#: by design, so a designation paired with it would contradict itself.
_ATTACH_OUTCOME = 'restated'

_PASS_MARKER = 'PASS  the judge-bound candidate is CONSUMED by the attach'
_FAIL_MARKER = 'FAIL  the judge-bound candidate is NOT CONSUMED by the attach'

_JUDGE_MODULE_NAME = 'fused_memory.server.write_triage_judge'
_JUDGE_TARGET_BRANCH = 'judge-target branch'
_BUILDER_NAME = 'build_judge_prompt'
_WRITER_NAME = 'judge_write'

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

#: The one machine-readable line naming the branch that held. The gate greps
#: this prefix and quotes a bounded prefix of the rest, so branch names stay
#: ASCII: a cut through a multi-byte character would emit a broken one.
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


class _SearchResults(list):
    """``MemoryService.SearchResults`` stand-in: a list that carries degradation.

    ``triage_write`` fails open on a degraded retrieval, so ``degraded`` must
    exist and be False for the run to reach the judge.
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


def _fixture_results(parent_field: str, child_kind: str) -> _SearchResults:
    """Six ordinary results plus the top-scoring evidence child of a hoisted parent."""
    results = _SearchResults(
        Candidate(
            f'm{i}',
            f'ordinary candidate {i}',
            {'store_score': _TOP_SCORE - (i * _SCORE_STEP)},
        )
        for i in range(_ORDINARY_COUNT)
    )
    results.append(
        Candidate(
            _CHILD_ID,
            'the child record that carries the evidence for the hoisted parent',
            {
                'store_score': _CHILD_SCORE,
                'kind': child_kind,
                parent_field: _CANONICAL_ID,
            },
        ),
    )
    return results


def _child_kind(module: Any) -> str:
    """A ``kind`` the ref's own ``_canonical_id_of`` treats as a child.

    Read from the ref, because a hardcoded kind would silently stop the hoist
    — and with it the case that matters — if the vocabulary moved.
    """
    kinds = getattr(module, 'CHILD_KINDS', None)
    for kind in sorted(kinds) if isinstance(kinds, (frozenset, set)) else ():
        if isinstance(kind, str) and kind:
            return kind
    return 'sighting'


def _attach_id_for(module: Any, candidate: Any) -> Any:
    """The id an attach to *candidate* MUST land on: its own, or its PARENT's.

    ``_canonical_id_of`` makes hoisting a child to its parent mandatory, so a
    correct remedy lands there. The ref's own function decides, so the probe
    cannot drift from the write side. The metadata rule is only the fallback
    for a ref that renamed it.
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
        parent_id = meta.get(parent_key(module))
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


#: THE DESIGNATION-SHAPE SEARCH. Option (a)'s wire shape has not landed, so
#: none is pinned: each is tried, and the first whose swap holds wins. The bare
#: outcome str is a CONTROL: it designates nothing, so it must never satisfy on
#: its own, and it is tried and reported like the rest.
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

    Duck-typed on the two fields read, so a ref that renamed the dataclass is
    still measured. Without it, a shape with no ``canonical_id`` would read as
    an attach id of None and be reported NOT CONSUMED, a defect this run never
    measured.
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
    triage_write = require(module, 'triage_write')
    counter, read_fail_opens = _make_counter(module)
    results = _fixture_results(parent_key(module), _child_kind(module))
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

    The ref's own ``TriageFailOpenCounter`` where usable, so the ref's
    accounting is what is measured; a counting stand-in where the class is
    absent or its shape changed.
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
    # Outside the branch above, so an ABSENT class warns too: the fallback
    # changes what was measured, and that has to be said.
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
    """Slate ids usable as a designation.

    Each must differ from the band's own canonical, where main already
    attaches. Each must also be its OWN canonical id. A child's attach target
    is its parent (:func:`_attach_id_for`), so designating one would ask a
    correct remedy both to honour the designation and to hoist it. The child
    stays on the slate; it is only never designated.
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

    Returns ``(run, slate, band_canonical)``, the slate as candidate OBJECTS,
    because where an attach to one must land depends on its metadata. The
    verdict is a bare outcome str, which no module can reject as unrecognised.
    """
    judge = _FakeJudge(_ATTACH_OUTCOME)
    run = _drive(module, judge)
    if run.error is not None:
        raise Unverifiable(run.error)
    if run.fail_opens:
        raise Unverifiable(
            f'the module recorded {run.fail_opens} fail-open(s) on a plain, '
            f'valid {_ATTACH_OUTCOME!r} verdict, so this run measures a '
            'degraded write path rather than what the attach consumes',
        )
    if not run.calls:
        raise Unverifiable(
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

    TWO different designations, because any fixed position differs from the
    band's canonical too: only following both makes this a test of the
    DEPENDENCY rather than of a value. Exact equality suffices because
    :func:`_designated_ids` yields only self-canonical ids, so a remedy that
    hoists lands on the designation itself.
    """
    for designated in designations:
        judge = _FakeJudge(spelling.payload(_ATTACH_OUTCOME, designated))
        run = _drive(module, judge)
        if run.error is not None:
            return run.error
        # Before the tracking test, and read off the counter rather than the
        # outcome (`stored` is a legitimate verdict too): a fail-open returns
        # canonical_id=None, which also avoids the band's top-1.
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


def _target_parameter_names(fn: Any) -> list[str]:
    """Names beyond the first two (the entry and the slate) that read as a target.

    All of them, not just the third: a target parameter's position is a
    mechanism, so ``(content, candidates, *, verdict_words=None,
    attach_target_id=None)`` still has one.
    """
    return [p.name for p in usable_parameters(fn)[2:] if names_a_target(p)]


def _judge_decision(module: Any, canonical_id: str) -> Any:
    """The ref's own ``BandDecision``, carrying a canonical id the probe chose.

    The ref's class, so ``judge_write`` reads the shape it reads in
    production; a namespace only for a ref that renamed the dataclass.
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
    ``report_fail_closed``, which fails closed on it.
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

    Bound against the REAL builder's signature, so a target passed by position
    counts as much as one passed by keyword.
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

    Returns ``(satisfied, report line)``; the line is emitted either way, so a
    reader can see the branch was evaluated.

    THREE conditions, each rejecting a different near-miss: the renderer can be
    TOLD (a signature); ``judge_write`` actually TELLS it the id the write will
    use (a widened signature is not consumption); and the write LANDS there,
    or a triage module that ignores everything would ride the judge module's
    signature to a PASS.
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

    Every attempt's reason comes back too, so a FAIL reports the whole search
    rather than one arbitrary verdict.
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

    The probe stops at ``BandDecision.canonical_id``: the remaining hops live
    inline in ``tools.py::add_memory`` with no callable seam. PASS path only —
    the run an operator acts on, and one whose report window is uncontended.
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
    module = import_from_tree(
        _MODULE_NAME, src_root, mode=ImportMode.FATAL, extra_paths=extra_paths,
    )
    out.append(f'triage module: {getattr(module, "__file__", "<unknown>")}')
    judge_module = import_from_tree(
        _JUDGE_MODULE_NAME, src_root, mode=ImportMode.NON_FATAL,
    )

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
        raise Unverifiable(
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
    # Outside report_fail_closed: a usage error is the invocation's defect, not
    # the ref's, and keeps argparse's own exit code.
    args = parser.parse_args(argv)
    return report_fail_closed(
        lambda out: _probe(
            Path(args.src_root), [Path(p) for p in args.extra_path], out,
        ),
    )


if __name__ == '__main__':
    raise SystemExit(main())
