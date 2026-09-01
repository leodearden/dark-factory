#!/usr/bin/env python3
"""Probe the write_triage judge path's ATTACH-TARGET invariant by executing it.

Consumed by ``scripts/check_write_triage_flip_preconditions.sh`` item 1 through
the ``CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY`` env seam (modelled on
``scripts/check_sandbox_soak.sh``'s ``CHECK_SANDBOX_SOAK_PY``).

THE INVARIANT
-------------
The judge is shown several candidates, but the attach touches exactly one of
them. The invariant is that the judge path BINDS A VERDICT TO A DETERMINATE
CANDIDATE — otherwise a verdict reasoned about candidate #3 lands on candidate
#1 and stamps ``x_contested`` on a canonical the entry never contradicted.

WHY THIS EXECUTES RATHER THAN GREPS. Item 1 used to be a source-text grep for
``candidate_id``. A grep asserts which MECHANISM landed, not whether the
invariant holds: it fails a correct fix that establishes the invariant another
way, and it passes prose that changes no behaviour at all. This probe imports
the judge module out of the ref's own extracted tree and CALLS it.

WHY THE SLATE IS BUILT BY THE MODULE'S OWN ``select_judge_candidates``. That
function rescues a band winner from outside the top-n window by APPENDING it
(``selected = [*selected[: max(n - 1, 0)], winner]``). When the winner is a
HOISTED PARENT id that never appeared as a result of its own, the child
carrying the evidence is kept — and lands LAST. Measured against main
f474347580: ``select_judge_candidates(results, 3, canonical_id='parent-1')``
returned ``['m0', 'm1', 'child-1']``, i.e. the attach target is the LAST
element, not the first. Probing on a slate whose target happened to be first
would bless an implementation that marks ``candidates[0]``, which is wrong on
exactly this path.

EXIT-CODE CONTRACT
    0  the invariant holds.
    1  it does not hold, OR it could not be verified. An unverifiable
       invariant is not a satisfied one, and this gate protects a production
       flag flip.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib
import inspect
import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple

EXIT_OK = 0
EXIT_FAIL = 1

logger = logging.getLogger(__name__)

_MODULE_NAME = 'fused_memory.server.write_triage_judge'

#: The hoisted parent id the band picked as its canonical. It deliberately
#: never appears as a result of its own, so the rescue branch has to fall
#: through to the child that points at it.
_CANONICAL_ID = 'parent-1'
_CHILD_ID = 'child-1'

#: Small enough that the six ordinary results overflow the window, which is
#: what forces the rescue branch to run at all.
_SLATE_SIZE = 3

_NEW_ENTRY = 'A new memory entry submitted for triage by this probe.'

#: The two ways an implementation might spell "which candidate is the attach
#: target": by opaque id, or by handing over the candidate object itself.
#: Both are tried, and whichever the implementation accepts is used for every
#: subsequent render, so no calling convention is pinned.
_SPELLINGS = ('id', 'object')

#: Placeholders for a REQUIRED parameter that is not the target but has to be
#: supplied anyway to reach it. Without them the probe could only reach a
#: target parameter when every OTHER parameter beyond the first two was
#: optional -- so a correct fix spelled ``build_judge_prompt(content,
#: candidates, verdict_words, attach_target_id=None)`` raised TypeError on
#: every attempt, fell through to ``verdict_words``, and was reported
#: INDETERMINATE. That is the false-FAIL class this probe exists to remove,
#: one signature shape down.
#:
#: A LADDER, not one value, because the placeholder itself can be rejected:
#: ``None`` covers a parameter that is merely rendered, ``()`` one that is
#: iterated, ``''`` one that is concatenated. Tried in that order, and the
#: FIRST that renders wins -- the same value is then used for BOTH halves of
#: the swap pair, so the comparison stays fair.
_FILLERS: tuple[Any, ...] = (None, (), '')


#: Two ids belonging to NO candidate in the slate. Rendering against them is
#: the ECHO CONTROL: a parameter that MARKS the target matches its argument
#: against the slate, finds nothing, and therefore renders identically for both
#: (and mentions neither); a parameter whose value is merely interpolated into
#: the prompt emits whichever nonce it was handed. See ``_echoes_argument``.
_NONCE_IDS = ('probe-nonce-alpha-9f13', 'probe-nonce-beta-4c07')

#: Names that read as "this parameter designates the attach target". An echoing
#: parameter is forgiven only when it is named for what it designates, because
#: a real option (b) that names the target in a HEADER
#: (``'ATTACH TARGET: ' + attach_target_id``) is structurally indistinguishable
#: from free-text diagnostics that happens to be interpolated — the name is the
#: only thing separating them. Target-named parameters are also tried FIRST, so
#: a module carrying both reaches the real one.
#:
#: DELIBERATELY NARROW, from a measured false PASS. This set used to include
#: ``canonical|winner|chosen|selected|primary`` — the BAND's vocabulary, for
#: things the band already tracks for other reasons. ``canonical`` was the
#: hole: ``build_judge_prompt(content, candidates, canonical_id=None)`` that
#: merely interpolates the band canonical into a header line is an ordinary
#: prompt-legibility change binding no verdict to any candidate, and it PASSED
#: — authorising the production ``write_triage.enabled`` flip on a change that
#: establishes nothing. The ``_echo_forgiven_note`` WARN did not mitigate it:
#: the gate shell greps only for the PASS marker, so no operator ever sees the
#: warning on a passing run.
#:
#: Only ``attach``/``target``/``candidate_id`` survive — spellings that can
#: name the attach target and little else. Note the cost of narrowing is
#: bounded: a parameter that genuinely MARKS the target (matches its argument
#: against the slate) never reaches the echo control at all, so the name
#: matters ONLY for header-style renderings that interpolate the id verbatim.
_TARGET_NAME_RE = re.compile(r'attach|target|candidate_id', re.IGNORECASE)


class _Unverifiable(Exception):
    """The invariant could not be evaluated. Fails closed, never passes."""


class _Candidate:
    """A duck-typed ``MemoryResult`` stand-in.

    ``near_duplicate_guard._cosine_of`` — the reader
    ``select_judge_candidates`` imports — takes the per-store cosine from
    ``metadata['store_score']``, so ``.id``/``.content``/``.metadata`` is the
    whole contract. Constructing one avoids importing ``MemoryResult``, whose
    module pulls in third-party deps the extracted tree may not have.
    """

    __slots__ = ('content', 'id', 'metadata')

    def __init__(self, ident: str, content: str, metadata: dict[str, Any]) -> None:
        self.id = ident
        self.content = content
        self.metadata = metadata

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f'<candidate {self.id}>'


def _fixture_results(parent_key: str) -> list[_Candidate]:
    """Six ordinary results plus one hoisted-parent evidence child.

    The child scores LOWEST, so it falls outside the top-``_SLATE_SIZE``
    window and only reaches the slate via the rescue append.
    """
    results = [
        _Candidate(
            f'm{i}',
            f'ordinary candidate {i}',
            {'store_score': 0.90 - (i * 0.05)},
        )
        for i in range(6)
    ]
    results.append(
        _Candidate(
            _CHILD_ID,
            'the child record that carries the evidence for the hoisted parent',
            {'store_score': 0.40, parent_key: _CANONICAL_ID},
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


def _import_judge(src_root: Path) -> Any:
    """Import the judge module out of *src_root*, shadowing any installed copy.

    ``sys.path.insert(0, ...)`` puts the extracted tree ahead of site-packages,
    so the assertion is made against the REF and not against whatever happens
    to be installed in the interpreter running this probe.
    """
    if not src_root.is_dir():
        raise _Unverifiable(f'--src-root is not a directory: {src_root}')
    sys.path.insert(0, str(src_root))
    try:
        module = importlib.import_module(_MODULE_NAME)
    except BaseException as exc:  # noqa: BLE001 - any import failure is unverifiable
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


def _build_slate(module: Any) -> tuple[list[Any], Any, int]:
    """Return ``(slate, attach_target, index)`` from the module's own selector."""
    select = _require(module, 'select_judge_candidates')
    parent_key = getattr(module, 'PARENT_ID_KEY', None)
    if not isinstance(parent_key, str) or not parent_key:
        parent_key = 'parent_id'
    try:
        slate = list(
            select(_fixture_results(parent_key), _SLATE_SIZE, canonical_id=_CANONICAL_ID),
        )
    except Exception as exc:  # noqa: BLE001 - a raising selector is unverifiable
        raise _Unverifiable(f'select_judge_candidates raised: {exc!r}') from exc
    ids = [getattr(c, 'id', None) for c in slate]
    if _CHILD_ID in ids and ids.index(_CHILD_ID) != 0:
        # The strongest case, and the one main actually produces: the rescued
        # evidence child of a hoisted parent, sitting wherever the selector
        # put it — as long as that is not slate[0].
        #
        # THE `!= 0` IS LOAD-BEARING. A selector that rescues the winner by
        # HOISTING IT TO THE FRONT rather than appending it puts the child at
        # index 0, and _probe then has no second target to swap against: its
        # `other is target` guard fires and a CORRECT option (b) is failed
        # UNVERIFIABLE. Which END the rescue writes to is a MECHANISM, exactly
        # as the append is; falling through to the selection below finds a
        # usable target on such a slate instead of refusing it.
        index = ids.index(_CHILD_ID)
    else:
        # The winner-rescue APPEND is a MECHANISM, and so is its DIRECTION.
        # All this probe needs is an attach target that is not slate[0], so
        # that a candidates[0]-marking implementation is still caught; a
        # selector that reaches the hoisted parent some other way — not at all,
        # or by prepending it — is no less verifiable. Requiring the append
        # would fail a CORRECT fix closed forever for a reason unrelated to
        # which candidate the attach touches — the same false-FAIL defect this
        # probe exists to remove, one level down. Take the LAST candidate that
        # is distinguishable from slate[0].
        index = max(
            (
                position
                for position, ident in enumerate(ids)
                if ident is not None and ident != ids[0]
            ),
            default=-1,
        )
        if index < 0:
            raise _Unverifiable(
                f'select_judge_candidates returned no candidate distinguishable '
                f'from slate[0] (got {ids!r}); there is no attach target to '
                'reason about, so nothing here can tell a correct fix from the '
                'candidates[0] bug',
            )
    return slate, slate[index], index


def _usable_parameters(fn: Any) -> list[Any]:
    """*fn*'s parameters that can be supplied by position or by keyword.

    ``*args``/``**kwargs`` are dropped, so every index used elsewhere in this
    module -- ``_target_parameters``, ``_bind_arguments`` -- is an index into
    THIS list and they cannot drift apart.
    """
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


def _target_parameters(fn: Any) -> list[tuple[int, Any]]:
    """Every ``(index, parameter)`` of *fn* beyond the first two.

    The first two are the new entry and the slate; anything after them is a
    candidate for "which candidate is the attach target". ALL of them are
    returned, not just the third, because the position of that parameter is a
    MECHANISM and the invariant does not depend on it: a correct fix spelled
    ``build_judge_prompt(content, candidates, *, verdict_words=None,
    attach_target_id=None)`` would otherwise have the id fed into
    ``verdict_words`` and be failed with the factually wrong diagnostic that
    the rendering "does not depend on the argument at all".

    Positional-only, positional-or-keyword and keyword-only all count: the
    point is whether the renderer can be TOLD which candidate the attach will
    touch, not how the argument is spelled.
    """
    beyond = list(enumerate(_usable_parameters(fn)))[2:]
    # Target-NAMED parameters first, ties broken by position. A module can
    # carry both an echoing free-text parameter and a real marker; the search
    # takes the FIRST combination that holds, so the real one has to be
    # reached first. ``sorted`` is stable, so declaration order survives
    # within each group.
    return sorted(beyond, key=lambda item: not _names_a_target(item[1]))


def _names_a_target(param: Any) -> bool:
    """True when *param*'s NAME reads as designating the attach target."""
    return bool(_TARGET_NAME_RE.search(param.name))


def _value_for(candidate: Any, spelling: str) -> Any:
    return candidate.id if spelling == 'id' else candidate


def _nonce_candidate(ident: str) -> _Candidate:
    """A candidate that is in NO slate — the echo control's stand-in target."""
    return _Candidate(ident, 'a candidate that is not on the slate', {})


def _filled_parameters(usable: list[Any], index: int) -> list[Any]:
    """The parameters that must be supplied to REACH the one at *index*.

    Two classes, and neither is the target itself:

    * anything beyond the first two with no default. Omitting it raises
      ``TypeError`` before the renderer runs a single line, wherever it sits
      relative to the target -- a required keyword-only parameter declared
      AFTER the target is just as fatal as one declared before it.
    * an optional POSITIONAL-ONLY parameter before the target, which cannot be
      skipped and still leave the target reachable by position.
    """
    filled: list[Any] = []
    for position, param in enumerate(usable):
        if position < 2 or position == index:
            continue
        required = param.default is inspect.Parameter.empty
        blocking = (
            param.kind is inspect.Parameter.POSITIONAL_ONLY and position < index
        )
        if required or blocking:
            filled.append(param)
    return filled


def _fillers_for(usable: list[Any], index: int) -> tuple[Any, ...]:
    """The placeholder ladder to try for parameter *index*, shortest first.

    A single (unused) entry when nothing needs filling, so the ordinary case
    costs exactly one render per (parameter x spelling) as before and the
    failure report does not gain duplicate clauses.
    """
    return _FILLERS if _filled_parameters(usable, index) else _FILLERS[:1]


def _bind_arguments(
    usable: list[Any],
    index: int,
    slate: list[Any],
    value: Any,
    filler: Any,
) -> tuple[tuple[list[Any], dict[str, Any]] | None, str | None]:
    """``((args, kwargs), error)`` calling *fn* with *value* at *index*.

    Everything that can be is passed BY KEYWORD, so a target parameter that is
    not the third one still reaches the slot it was named for. Positional-only
    parameters are passed positionally and therefore have to be CONTIGUOUS:
    ``args`` covers ``usable[:position]`` exactly when ``len(args) ==
    position``, and if a gap has opened the target is genuinely unreachable.
    """
    fill_names = {param.name for param in _filled_parameters(usable, index)}
    args: list[Any] = [_NEW_ENTRY, slate]
    kwargs: dict[str, Any] = {}
    for position, param in enumerate(usable):
        if position < 2:
            continue
        if position == index:
            supplied = value
        elif param.name in fill_names:
            supplied = filler
        else:
            continue
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            if len(args) != position:
                return None, (
                    f'the positional-only parameter {param.name!r} sits at index '
                    f'{position} with only {len(args)} arguments before it, so the '
                    'target cannot be reached without guessing the ones in between'
                )
            args.append(supplied)
        else:
            kwargs[param.name] = supplied
    return (args, kwargs), None


def _render(
    fn: Any,
    usable: list[Any],
    index: int,
    slate: list[Any],
    target: Any,
    spelling: str,
    filler: Any,
) -> tuple[str | None, str | None]:
    """``(rendering, error)`` — render *slate* naming *target* at *index*."""
    bound, error = _bind_arguments(
        usable, index, slate, _value_for(target, spelling), filler,
    )
    if error is not None or bound is None:
        return None, error
    args, kwargs = bound
    try:
        rendered = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any render failure is inconclusive
        return None, f'raised: {exc!r}'
    if not isinstance(rendered, str):
        return None, f'returned {type(rendered).__name__}, not str'
    return rendered, None


def _echoes_argument(
    build: Any,
    usable: list[Any],
    index: int,
    slate: list[Any],
    spelling: str,
    filler: Any,
) -> str | None:
    """The CONTROL for the option-(b) swap test: is the argument just echoed?

    The swap test asks whether the rendering DEPENDS on which candidate was
    named. That is necessary and not sufficient: a parameter whose value is
    merely interpolated into the prompt — free-text diagnostics, a footer note,
    a trace id — satisfies BOTH swap conditions the moment the probe feeds it a
    candidate id. The interpolated line mentions an id (condition 1) and each
    id gains or loses that line (condition 2), while nothing in the module
    binds a verdict to a candidate. Measured: a judge whose only extra
    parameter was ``footer_note`` appended as ``'NOTE: ' + str(footer_note)``,
    with main's flat candidate list and main's bare-str ``parse_judge_verdict``,
    reported option (b) satisfied and exited 0 — authorising the production
    flip this gate exists to hold shut. That is the same false-pass class as
    ``_echoes_payload`` catches in the option-(a) half.

    So render against ids belonging to NO candidate on the slate. An
    implementation that MARKS the target compares its argument against the
    slate, matches nothing, and renders the same prompt for either nonce
    without mentioning either. An echoing parameter emits whichever nonce it
    was handed, or otherwise lets it perturb the output.

    Returns the reason string when the echo is detected, else ``None``.

    FAILS TOWARDS THE PASS, deliberately, on two counts. A renderer that
    REFUSES an off-slate target is a validating marker, not an echo, so a raise
    reports no echo. And an echo is only DISQUALIFYING for a parameter whose
    name does not designate the attach target (see ``_names_a_target``): a real
    option (b) that names the target in a header — ``'ATTACH TARGET: ' +
    attach_target_id`` — echoes its argument in exactly this way, and only the
    parameter's name separates it from the false pass above.
    """
    renders: list[tuple[str, str]] = []
    for nonce in _NONCE_IDS:
        rendered, error = _render(
            build, usable, index, slate, _nonce_candidate(nonce), spelling, filler,
        )
        if error is not None or rendered is None:
            return None
        renders.append((nonce, rendered))
    for nonce, rendered in renders:
        if nonce in rendered:
            return (
                f'the argument is ECHOED into the prompt, not matched against the '
                f'slate — rendering with {nonce!r}, which is no candidate on the '
                f'slate, put {nonce!r} into the prompt verbatim. A parameter that '
                'MARKS the attach target marks nothing for an id it does not '
                'recognise; one that merely interpolates its value satisfies the '
                'swap test without binding any verdict to any candidate'
            )
    if renders[0][1] != renders[1][1]:
        return (
            f'the argument is ECHOED into the prompt, not matched against the '
            f'slate — two ids belonging to no candidate ({_NONCE_IDS[0]!r} and '
            f'{_NONCE_IDS[1]!r}) rendered DIFFERENTLY, so the value perturbs the '
            'prompt on its own rather than selecting among the candidates'
        )
    return None


class _Search(NamedTuple):
    """The outcome of the option-(b) search over (parameter x spelling x filler).

    ``target_named_rendered`` is what separates 'the invariant does not hold'
    from 'nothing measured it'. A signature that carries a target-NAMED
    parameter none of whose calls ever produced a pair of renderings has
    asserted nothing about the attach target, however many OTHER parameters
    rendered happily -- see ``_probe``.
    """

    winner: tuple[int, Any, str, bool] | None
    attempts: list[str]
    rendered_any: bool
    target_named_rendered: bool


def _search_option_b(
    build: Any,
    slate: list[Any],
    target: Any,
    other: Any,
) -> _Search:
    """Search the (parameter x spelling x filler) space for a combination that holds.

    WHY A SEARCH RATHER THAN A CHOICE. The previous shape picked the third
    parameter, then picked the first spelling that did not RAISE. Both choices
    silently asserted a mechanism:

    * "did not raise" is not "was understood". An implementation that marks
      ``if c is attach_target`` accepts the id spelling perfectly happily — it
      simply matches nothing — so the object spelling was dead for exactly the
      implementations that needed it, and a correct object-identity fix
      rendered identically for both targets and FAILED.
    * the third parameter is not the only place a target argument can sit.

    So every combination is tried and the first one whose swap test holds AND
    whose argument survives the echo control (``_echoes_argument``) wins. A
    combination the implementation ignores renders identically and is simply
    not the winner, which costs nothing; only a judge where NO combination
    holds fails. On failure the whole search is reported, so the operator sees
    what was tried rather than one arbitrary verdict.

    ORDER MATTERS, hence ``_target_parameters`` yields target-NAMED parameters
    first: a module can carry both an echoing free-text parameter and a real
    marker, and taking the first combination that holds would otherwise stop at
    whichever came earlier in the signature.

    THE FILLER IS THE INNERMOST LOOP, and it stops at the first value that
    RENDERS rather than at the first that passes: a combination that renders
    has been measured, and re-measuring it with another placeholder would only
    restate the same finding once per placeholder in a report whose window is
    already contended.
    """
    attempts: list[str] = []
    rendered_any = False
    target_named_rendered = False
    usable = _usable_parameters(build)
    for index, param in _target_parameters(build):
        named = _names_a_target(param)
        fillers = _fillers_for(usable, index)
        for spelling in _SPELLINGS:
            for filler in fillers:
                label = (
                    f'{param.name!r} (parameter {index}, candidate {spelling}'
                    f'{_filler_note(fillers, filler)})'
                )
                rendered_target, error = _render(
                    build, usable, index, slate, target, spelling, filler,
                )
                if error is not None:
                    attempts.append(f'{label} — {error}')
                    continue
                rendered_other, error = _render(
                    build, usable, index, slate, other, spelling, filler,
                )
                if error is not None:
                    attempts.append(f'{label} — {error}')
                    continue
                rendered_any = True
                target_named_rendered = target_named_rendered or named
                assert rendered_target is not None and rendered_other is not None
                reason = _swap_verdict(
                    slate, target, other, rendered_target, rendered_other,
                )
                if reason is not None:
                    attempts.append(f'{label} — {reason}')
                    break
                # The CONTROL. The swap test cannot tell a parameter that MARKS
                # the attach target from one that merely echoes whatever it is
                # handed, because feeding a candidate id to an echoing parameter
                # satisfies both swap conditions. Only a parameter NAMED for the
                # target is forgiven an echo, and the report says so out loud.
                echo = _echoes_argument(build, usable, index, slate, spelling, filler)
                if echo is not None and not named:
                    attempts.append(f'{label} — {echo}')
                    break
                return _Search(
                    (index, param, spelling, echo is not None),
                    attempts,
                    rendered_any,
                    target_named_rendered,
                )
    return _Search(None, attempts, rendered_any, target_named_rendered)


def _filler_note(fillers: tuple[Any, ...], filler: Any) -> str:
    """How the report names the placeholder, and nothing at all when unused."""
    return f', other required parameters={filler!r}' if len(fillers) > 1 else ''


# --- option (a): the verdict names its own candidate --------------------------
#
# Evaluated FIRST, and deliberately tolerant. Option (a) — a verdict that
# carries the judged candidate id — makes slate position irrelevant, so it
# closes item 1 with a prompt that marks nothing at all. Task 4798 item 7
# still carries option (a) as the better long-term design; a gate shaped
# around option (b) alone would fail it and re-block task 3169, which is the
# very defect (mirrored) that task 4810 exists to remove.

#: Field names an option-(a) wire contract plausibly uses for the candidate.
#: Every probe payload carries one — a payload naming NO candidate can never be
#: evidence that the verdict binds to one, so the id-less shape is not tried.
_ID_FIELDS = ('candidate_id', 'id')

#: A verdict that is one of these is a bare outcome, not a binding to a
#: candidate. ``bool`` is listed before ``int`` reaches it for clarity only —
#: ``isinstance`` covers the subclass either way.
_NOT_A_CANDIDATE_BINDING = (str, bytes, bool, int, float)

#: A field name no candidate contract can plausibly read as "the candidate this
#: verdict is about". ``_echoes_payload`` hides the id there to tell a parser
#: that READS the contract from one that merely echoes its input back.
_DECOY_FIELD = 'x_probe_field_that_no_contract_reads'

#: How deep ``_recoverable`` descends. Deep enough for any plausible verdict
#: shape (a pair holding a dataclass holding a dict), bounded so a pathological
#: or deeply self-referential structure cannot stall the probe.
_SCAN_DEPTH = 6


def _binds_candidate(result: Any) -> bool:
    """True when *result* is a shape that COULD carry a candidate id.

    Necessary, never sufficient: ``_recoverable`` (plus the ``_echoes_payload``
    control) is what actually decides the branch. ``None`` and the scalars are excluded so a parser that quietly
    returns ``None`` on an unrecognised payload — rather than raising — is not
    mistaken for one that returns an ``(outcome, candidate_id)`` pair.
    """
    return result is not None and not isinstance(result, _NOT_A_CANDIDATE_BINDING)


def _occurrences(result: Any, ident: str, depth: int = _SCAN_DEPTH, seen: set[int] | None = None) -> int:
    """How many times *ident* appears as a DISCRETE string value inside *result*.

    This is the evidence half of the option-(a) branch. A non-scalar return is
    only a SHAPE that could carry a candidate; unless the id actually handed in
    comes back out, the value proves nothing — an ordinary refactor returning a
    dataclass or a ``{'outcome': ...}`` dict for the outcome alone would
    otherwise read as satisfied and authorise a production flag flip.

    TWO DELIBERATE NARROWINGS, both from measured false passes:

    * ONLY EXACT STRING EQUALITY COUNTS. A string that merely CONTAINS the id
      is not evidence — the probe hands the parser a JSON payload naming the
      id, so any parser that returns that raw text alongside its outcome
      (``return (verdict, raw)``) would otherwise "recover" the id it was
      never asked to extract. Measured: that shape reported option (a)
      satisfied and exited 0, authorising the flip.
    * THERE IS NO ``repr`` FALLBACK. ``ident in repr(result)`` is the same
      containment mistake one level up, and it reaches THROUGH any nested echo
      of the input. Measured: ``return {'outcome': v, 'raw_payload':
      json.loads(raw)}`` — a plainly plausible diagnostics refactor that
      discards the candidate entirely — passed on the ``repr`` last resort
      alone. The recursive structural scan below covers every shape the
      fallback was there for (dicts, sequences, ``__dict__``/``__slots__``
      objects, and those nested inside each other), so dropping it costs no
      legitimate implementation.

    Recovering the id from a nested ECHO of the probe's own payload is still
    possible here by construction — a value equal to the payload cannot be
    told from a binding by inspection alone. ``_echoes_payload`` is the
    control that separates them, and it needs the COUNT rather than a bare
    yes/no: a parser that both reads the candidate field and carries the
    payload it read it from surfaces the id once for the binding and once for
    the copy, where a pure echo surfaces it only for the copy.
    """
    if depth < 0:
        return 0
    if isinstance(result, str):
        return 1 if result == ident else 0
    if seen is None:
        seen = set()
    marker = id(result)
    if marker in seen:
        return 0
    seen.add(marker)

    children: list[Any] = []
    if isinstance(result, dict):
        children.extend(result.keys())
        children.extend(result.values())
    elif isinstance(result, (list, tuple, set, frozenset)):
        children.extend(result)
    with contextlib.suppress(Exception):  # a hostile __dict__ is not evidence
        children.extend(getattr(result, '__dict__', {}).values())
    slots = getattr(type(result), '__slots__', ()) or ()
    for name in (slots,) if isinstance(slots, str) else slots:
        try:
            children.append(getattr(result, name, None))
        except Exception:  # noqa: BLE001 - a hostile __getattr__ is not evidence
            continue

    total = 0
    for child in children:
        try:
            total += _occurrences(child, ident, depth - 1, seen)
        except Exception:  # noqa: BLE001 - a hostile container is not evidence
            continue
    return total


def _recoverable(result: Any, ident: str) -> bool:
    """True when *ident* appears at least once inside *result*. See _occurrences."""
    return _occurrences(result, ident) > 0


def _echoes_payload(
    parse: Any, key: str, word: str, ident: str, bound: Any,
) -> str | None:
    """The CONTROL for the option-(a) evidence test: is the parser just echoing?

    Recoverability of an id from a result is only a BINDING if the parser got
    it by reading the candidate field of the contract. A parser that hands the
    whole submitted payload back — ``{'outcome': v, 'raw_payload':
    json.loads(raw)}``, ``{**payload}`` — reproduces whatever it was given, so
    the id comes back out no matter what the field meant. That is an echo, not
    a verdict bound to a candidate, and it passed the pre-fix probe.

    So submit a payload that names the id ONLY under a field the contract
    cannot plausibly read, and NOWHERE a candidate field would be. A parser
    that reads the contract reports no id (or refuses the payload outright); a
    parser that echoes reports *ident* anyway, which is the signature.

    Returns the reason string when the echo is detected, else ``None``.

    A CARRIED PAYLOAD IS NOT AN ECHO ON ITS OWN. A parser that reads the
    candidate field AND hands back the payload it read it from —
    ``(verdict, payload.get('candidate_id'), payload)`` — surfaces the decoy
    too, and no yes/no inspection of the control can tell it from a pure echo,
    because both reproduce their input. What separates them is the COUNT: the
    binder surfaces the id once for the BINDING and once for the copy, where a
    pure echo surfaces it only for the copy. So the echo is only reported when
    the real payload's result carries the id NO MORE OFTEN than the control's
    result carries the decoy — that is, when there is no extra occurrence for
    the binding to live in. Both sides are counted against the SAME id, so a
    parser that duplicates its payload is measured symmetrically and is still
    caught.

    FAILS TOWARDS THE PASS, deliberately: a parser that RAISES on the control
    is a validating one (the stricter option (a) — see the
    ``option_a_rejects_dangling`` fixture — refuses a verdict carrying no
    candidate id at all), and a raise is not evidence of echoing. Only a
    successful parse that surfaces the decoy id counts.

    *bound* is the result of parsing the REAL payload for the same *ident*.
    """
    control = {key: word, _DECOY_FIELD: ident}
    try:
        result = parse(json.dumps(control))
    except Exception as exc:  # noqa: BLE001 - a refusal is a validating parser
        # Not an error, and deliberately not fatal: a parser that REFUSES the
        # decoy is a validating one, which is the stricter option (a). But the
        # control then did NOT run, so a subsequent option-(a) PASS rests on
        # weaker evidence than one where the control executed and came back
        # clean. Say so at WARN+ rather than swallowing it -- this gate
        # authorises a production flag flip, and stderr is folded into the
        # report the operator reads.
        logger.warning(
            'echo control not exercised for verdict word %r under field %r: '
            'parse_judge_verdict refused the decoy payload (%s: %s). Treating '
            'the parser as validating, not echoing.',
            word, key, type(exc).__name__, exc,
        )
        return None
    echoed = _occurrences(result, ident)
    if not echoed:
        return None
    if _occurrences(bound, ident) > echoed:
        # The real result carries the id more often than the control carries
        # the decoy. The surplus cannot come from reproducing the input --
        # the control reproduces just as much -- so it is the binding.
        return None
    return (
        f'the id is ECHOED, not bound — a control payload naming {ident!r} only '
        f'under {_DECOY_FIELD!r} (a field no candidate contract reads, with no '
        f'candidate field present at all) still parsed to {result!r}, which '
        f'carries {ident!r}. The parser reproduces its input rather than '
        'reporting which candidate the verdict is about, so recovering the id '
        'proves nothing about the binding'
    )


def _same_result(left: Any, right: Any) -> bool:
    """``left == right``, treating a raising/ambiguous ``__eq__`` as 'differs'."""
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - numpy-style ambiguity is not sameness
        return False


def _verdict_key(module: Any) -> str:
    key = getattr(module, 'VERDICT_KEY', None)
    return key if isinstance(key, str) and key else 'verdict'


def _verdict_words(module: Any) -> tuple[str, ...]:
    """The judge's own vocabulary, read from the module where it exposes one."""
    words = getattr(module, 'JUDGE_VERDICTS', None)
    if isinstance(words, dict) and words:
        return tuple(str(word) for word in words)
    return ('distinct', 'restates', 'amends', 'contests')


def _parse_callers(parse: Any, slate: list[Any]) -> list[tuple[str, Any]]:
    """Every calling convention an option-(a) ``parse_judge_verdict`` may demand.

    WHY THIS EXISTS. The probe used to call ``parse(raw)`` and nothing else.
    But the STRICTEST option (a) — the one the gate's own failure text
    prescribed, "validate it against the slate in parse_judge_verdict" — takes
    the slate as a REQUIRED positional: ``parse_judge_verdict(raw,
    candidates)``. Every payload then raised ``TypeError`` before the parser
    ran, option (a) reported 'accepted no probe payload', and control fell
    through to option (b), which a correct option-(a) fix has no reason to
    satisfy. Measured: a complete option-(a) fix exited 1 and would have left
    task 3169 permanently re-blocked — the very defect this gate exists to
    remove, reproduced one level down.

    THE SLATE IS PASSED IN BOTH SHAPES because a validating parser may compare
    against ids or against the candidate objects, and there is no way to know
    which from the outside. Positional first, then by each plausible keyword.
    The probe's payloads name the slate's OWN ids (``first``/``last``), so a
    parser that validates membership accepts them.

    Ordered least-assuming first: the single-argument call is tried before any
    slate is supplied, so a parser that never wanted one is never handed one.
    """
    ids = [str(getattr(candidate, 'id', candidate)) for candidate in slate]
    objects = list(slate)
    callers: list[tuple[str, Any]] = [('parse(raw)', lambda raw: parse(raw))]
    for shape, value in (('ids', ids), ('candidates', objects)):
        callers.append((f'parse(raw, <slate {shape}>)', lambda raw, v=value: parse(raw, v)))
        for keyword in ('candidates', 'candidate_ids'):
            callers.append((
                f'parse(raw, {keyword}=<slate {shape}>)',
                lambda raw, k=keyword, v=value: parse(raw, **{k: v}),
            ))
    return callers


def _option_a_verdict(module: Any, slate: list[Any]) -> tuple[bool, str]:
    """``(holds, reason)`` for the option-(a) branch.

    An EVIDENCE test, not a type test. Two payloads differing ONLY in the
    candidate id are parsed, and the branch is satisfied only when all three
    hold: both parses succeed, the two results DIFFER, and each supplied id is
    recoverable from its own result. Anything weaker — including a non-scalar
    return on a payload that named no candidate — is inconclusive and falls
    through to option (b).

    A raise on every CALLING CONVENTION is INCONCLUSIVE, not a failure. The
    conventions themselves are no longer assumed: ``_parse_callers`` supplies
    the slate positionally and by keyword, so a parser that VALIDATES against
    the slate — the strictest option (a), and the one the gate's own text
    prescribed — is exercised rather than rejected at the call boundary.
    """
    parse = getattr(module, 'parse_judge_verdict', None)
    if parse is None:
        return False, 'parse_judge_verdict is absent from the module'
    key = _verdict_key(module)
    first = str(getattr(slate[0], 'id', ''))
    last = str(getattr(slate[-1], 'id', ''))
    if not first or not last or first == last:
        return False, (
            'inconclusive — the slate carries fewer than two distinct candidate '
            'ids, so no two payloads differing only in the id can be built'
        )
    unreached: list[str] = []
    for label, caller in _parse_callers(parse, slate):
        outcome = _option_a_under(module, caller, key, first, last)
        if outcome is None:
            # This convention never got a payload PAST the call boundary, so it
            # asserted nothing. Keep it only as a fallback diagnostic; another
            # convention may still reach the parser.
            unreached.append(label)
            continue
        holds, reason = outcome
        if holds:
            return True, reason
        # A convention that REACHED the parser rendered a real verdict on it.
        # That finding is the actionable one, so it outranks every "the call
        # signature did not match" note from the conventions that did not.
        return False, reason
    return False, (
        f'inconclusive — parse_judge_verdict accepted no probe payload under any '
        f'calling convention ({_first_few(unreached, limit=len(unreached))})'
    )


def _option_a_under(
    module: Any,
    parse: Any,
    key: str,
    first: str,
    last: str,
) -> tuple[bool, str] | None:
    """``(holds, reason)`` for ONE calling convention, or ``None``.

    *parse* is a single-argument adapter from ``_parse_callers``; every call
    below goes through it, so the convention is fixed for the whole evaluation
    (including the ``_echoes_payload`` control).

    A ``None`` RETURN — not a ``None`` reason — means the convention never
    reached the parser, every payload having raised, which is not a verdict
    about the invariant. The caller tries the next convention rather than
    reporting a failure it did not measure.

    WHY THE WHOLE RETURN AND NOT THE REASON. The earlier ``(False, None)``
    spelling made the two fields correlated-but-unchecked: nothing except a
    docstring stopped a later edit returning ``(True, None)``, which the
    driver's ``reason is None`` arm would then have silently filed as
    'this convention asserted nothing' — dropping a SATISFIED verdict on the
    floor and failing the gate closed against a correct fix. That is exactly
    the silent-degradation shape this gate exists to catch, so the state is
    made unrepresentable instead of merely documented: a reason now exists if
    and only if a verdict does, and the type checker enforces it.
    """
    saw_bare_str = False
    inert: list[str] = []
    errors: list[str] = []
    for word in _verdict_words(module):
        for field in _ID_FIELDS:
            parsed: list[tuple[str, Any]] = []
            for ident in (first, last):
                payload = {key: word, field: ident}
                try:
                    result = parse(json.dumps(payload))
                except Exception as exc:  # noqa: BLE001 - a rejection is inconclusive
                    errors.append(f'{payload!r} -> {exc!r}')
                    break
                if isinstance(result, str):
                    # Fully explained by the bare-str branch below. Letting it
                    # fall through to `inert` too restated the same finding
                    # once per (word, field) pair — eight identical clauses on
                    # main — crowding the report's limited window.
                    saw_bare_str = True
                    break
                if not _binds_candidate(result):
                    inert.append(
                        f'{payload!r} -> returned {type(result).__name__}, which '
                        'cannot carry a candidate id',
                    )
                    break
                parsed.append((ident, result))
            if len(parsed) != 2:
                continue
            (id_a, res_a), (id_b, res_b) = parsed
            if _same_result(res_a, res_b):
                inert.append(
                    f'{field!r} — two payloads naming {id_a!r} and {id_b!r} parsed '
                    f'to the SAME value {res_a!r}, so the id is discarded',
                )
                continue
            missing = [
                ident
                for ident, result in parsed
                if not _recoverable(result, ident)
            ]
            if missing:
                inert.append(
                    f'{field!r} — the results differ but the supplied id(s) '
                    f'{missing!r} are not recoverable from them, so the difference '
                    'is not the candidate binding',
                )
                continue
            # The CONTROL. Recoverability alone cannot tell a parser that READS
            # the candidate field from one that hands the whole payload back;
            # both round-trip the id and both make the two results differ.
            echo = _echoes_payload(parse, key, word, id_b, res_b)
            if echo is not None:
                inert.append(f'{field!r} — {echo}')
                continue
            return True, (
                f'satisfied — parse_judge_verdict round-trips the candidate id: '
                f'payloads naming {id_a!r} and {id_b!r} (field {field!r}) parsed to '
                f'the distinct values {res_a!r} and {res_b!r}, each carrying the id '
                'it was given, so the verdict itself names its candidate and the '
                'position of the attach target in the slate stops mattering'
            )
    # ORDER MATTERS. A bare str both sets the flag and fails _binds_candidate,
    # so it ALSO lands in `inert`; consulting `inert` first made this branch
    # dead code and gave main's actual shape — the most common case by far —
    # the least legible diagnostic. The bare-str finding is the specific,
    # actionable one, so it outranks the generic bucket. Nothing is dropped:
    # the rest of the bucket is still reported after it.
    if saw_bare_str:
        rest = ('; also ' + _first_few(inert)) if inert else ''
        return False, (
            'parse_judge_verdict returns a bare str — the verdict names no '
            'candidate, so the attach must guess one from the slate' + rest
        )
    if inert:
        return False, 'inconclusive — ' + _first_few(inert)
    # Every payload raised: this convention asserted nothing. Signal that with
    # a None RETURN so the driver moves on to the next one instead of reporting
    # a failure it did not measure.
    logger.debug('option (a): no payload reached the parser (%s)', _first_few(errors))
    return None


def _first_few(reasons: list[str], limit: int = 3) -> str:
    shown = '; '.join(reasons[:limit])
    if len(reasons) > limit:
        shown += f'; …{len(reasons) - limit} more'
    return shown


# --- the swap test ----------------------------------------------------------
#
# Render the SAME slate twice against two DIFFERENT attach targets. Everything
# below is stated in terms of opaque ids and rendering inequality; no heading
# text is ever matched, so any rewording survives.


def _divergent_lines(left: str, right: str) -> list[str]:
    """Every line either rendering carries that the other does not, IN CONTENT.

    ``SequenceMatcher`` rather than a set difference, because the hoist shape is
    a PERMUTATION: both renderings contain exactly the same SET of lines, so a
    set difference is empty for a correct fix (measured on this probe's own
    ``by_id`` fixture). The opcode blocks keep that reordering visible.
    """
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    divergent: list[str] = []
    matcher = difflib.SequenceMatcher(None, left_lines, right_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        divergent.extend(left_lines[i1:i2])
        divergent.extend(right_lines[j1:j2])
    return divergent


def _id_footprint(rendering: str, ident: str) -> tuple[tuple[int, str], ...]:
    """Where *ident* appears AND what those lines say — its rendered footprint.

    Content is part of the footprint, not just the line index. An index-only
    footprint (which this replaces) rejected the most natural minimal option
    (b) — marking the target INLINE on its own line, e.g.
    ``'- id: ' + c.id + (' <-- ATTACH TARGET' if c.id == target else '')`` —
    because the target id keeps its line index there while the line's text
    changes. That asserted a MECHANISM (the target must be RELOCATED) rather
    than the invariant, the very defect task 4810 exists to remove, and would
    have permanently re-blocked task 3169 against a valid fix.
    """
    return tuple(
        (i, line) for i, line in enumerate(rendering.splitlines()) if ident in line
    )


def _distinguished_by_repetition(rendering: str, slate: list[Any]) -> str | None:
    """The one id mentioned strictly more often than every other, or ``None``.

    An implementation that hard-codes ``candidates[0]`` as the attach target
    renders that id twice — once in whatever marks it, once in the list — while
    every other candidate is mentioned once. That is an ID-LEVEL signal, so it
    separates the positional bug from an argument that is simply ignored
    without reading a word of the prompt.
    """
    counts = {str(c.id): rendering.count(str(c.id)) for c in slate}
    if len(counts) < 2:
        return None
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0] if ranked[0][1] > ranked[1][1] else None


def _swap_verdict(
    slate: list[Any],
    target: Any,
    other: Any,
    rendered_target: str,
    rendered_other: str,
) -> str | None:
    """``None`` when the swap test holds, else the reason it does not.

    Two independent conditions, both necessary. Together they accept either
    natural option-(b) shape — hoisting the target out of the list, or marking
    it inline — while still rejecting a prose-only change, a rendering that
    varies for a reason unrelated to the candidates, and the ``candidates[0]``
    bug.
    """
    if rendered_target == rendered_other:
        marked = _distinguished_by_repetition(rendered_target, slate)
        if marked is not None and marked == str(other.id):
            return (
                f'build_judge_prompt distinguishes slate[0] ({marked!r}) regardless '
                f'of the requested attach target — asking for {target.id!r} and for '
                f'{other.id!r} rendered identically'
            )
        return (
            'build_judge_prompt produced byte-identical prompts for two different '
            'attach targets, so its output does not depend on the argument at all '
            '(a prose-only change renders identically whichever candidate is named)'
        )

    # (1) The divergence must touch the CANDIDATES at all. A rendering that
    #     merely varies run to run (a nonce, a timestamp) differs without any
    #     divergent line ever mentioning a candidate — and an INSERTED such line
    #     shifts every id's index, which is why condition (2) alone is not
    #     enough.
    slate_ids = [str(c.id) for c in slate]
    divergent = _divergent_lines(rendered_target, rendered_other)
    if not any(ident in line for line in divergent for ident in slate_ids):
        return (
            'the two renderings differ, but no differing line mentions any candidate '
            f'id ({slate_ids!r}) — the difference is not attributable to the attach '
            'target at all (a rendering that varies for an unrelated reason differs '
            'without ever naming what it was asked for)'
        )

    # (2) And the divergence must track the NAMED candidate specifically: each
    #     id's footprint has to change when it goes from being the attach target
    #     to being context. Content counts as much as position, so hoisting and
    #     inline marking are both accepted.
    for ident in (str(target.id), str(other.id)):
        if _id_footprint(rendered_target, ident) == _id_footprint(rendered_other, ident):
            return (
                f'the two renderings differ, but the id {ident!r} occupies the same '
                'lines with the same text whether or not it is the named attach '
                'target, so the difference is not attributable to the candidate '
                'being named'
            )
    return None


# --- report -----------------------------------------------------------------

_FAIL_NEITHER = [
    'FAIL  the attach target is INDETERMINATE on the judge path.',
    '      The judge is shown several candidates but the attach touches exactly',
    '      one of them, so a verdict reasoned about one candidate is filed',
    '      against another and x_contested lands on a canonical the entry never',
    '      contradicted.',
    '      EITHER remedy closes this — what is asserted is the INVARIANT, not',
    '      which mechanism landed:',
    '        (a) make parse_judge_verdict return the judged candidate alongside',
    '            the outcome (an (outcome, candidate_id) pair), so the verdict',
    '            names its own candidate and slate position stops mattering; or',
    '        (b) give build_judge_prompt a parameter naming the attach target,',
    '            and make its rendering DEPEND on which candidate that names.',
]


def _rescue_note(slate_ids: list[Any], index: int) -> list[str]:
    """The measured warning that marking ``candidates[0]`` is not sufficient.

    Only cites the winner-rescue APPEND when this run actually OBSERVED it.
    ``_build_slate`` no longer requires that mechanism, so a selector that
    reaches the hoisted parent another way must not be told, in the gate's own
    report, that it does something it does not do.
    """
    head = [
        '      Marking candidates[0] as the attach target is NOT sufficient:',
    ]
    if index < len(slate_ids) and slate_ids[index] == _CHILD_ID:
        head += [
            '      select_judge_candidates rescues a hoisted parent\'s evidence child by',
            '      APPENDING it (selected = [*selected[: max(n - 1, 0)], winner]), so on',
            '      a hoisted-parent slate the attach target is LAST, not first. Measured',
        ]
    else:
        head += [
            '      the attach target is whichever candidate the band picked, which is',
            '      not in general the first one the selector returned. Measured',
        ]
    return head + [
        f'      on this probe\'s own fixture: slate {slate_ids!r}, attach target at',
        f'      index {index}.',
    ]


def _echo_forgiven_note(param: Any) -> list[str]:
    """Say out loud that this pass rests partly on a parameter's NAME.

    The winning parameter echoed an off-slate id straight into the prompt, so
    the swap test alone cannot distinguish it from free-text diagnostics that
    the probe happened to feed a candidate id (see ``_echoes_argument``). It
    was accepted because its name designates the attach target — a real option
    (b) that names the target in a header looks exactly like this. Surfacing it
    keeps the operator from reading a name-assisted pass as a purely behavioural
    one.
    """
    return [
        f'WARN  ACCEPTED ON AN ECHOED, TARGET-NAMED PARAMETER: {param.name!r} put an',
        '      off-slate id into the prompt verbatim rather than matching it against',
        '      the candidates, so the swap test alone cannot separate it from free',
        '      text that merely echoes its argument. It is accepted because its NAME',
        '      designates the attach target (a real option (b) naming the target in a',
        '      header renders exactly this way). Confirm by eye that the judge prompt',
        '      genuinely tells the model which candidate the attach will touch.',
    ]


def _pass_scope_note() -> list[str]:
    """What a PASS deliberately does NOT prove.

    Item 1 asserts that the JUDGE PATH binds a verdict to a determinate
    candidate. It does not execute the attach, so neither remedy is checked
    for being CONSUMED: option (a) does not show the id is validated against
    the slate or threaded through BandDecision, and option (b) does not show
    the attach touches whatever the prompt named. A change that only widens
    the parse contract therefore opens this gate while the attach still lands
    on the band's top-1 — the very harm item 1 describes.

    Emitted on the PASS path only. That is where an operator is about to flip
    a production flag on the strength of this line, and it is also the report
    whose 2000-char window is NOT contended: a FAIL's window belongs to the
    remedy.
    """
    return [
        '      NOTE this gate does NOT assert that the bound candidate is CONSUMED.',
        '      It measures the judge path only: that a verdict can be tied to a',
        '      determinate candidate. It does not execute the attach, so it does not',
        '      show the id is validated against the slate, threaded through',
        '      BandDecision, or that the write the flag enables touches the candidate',
        '      the judge reasoned about. Confirm that separately before flipping.',
    ]


def _probe(src_root: Path, out: list[str]) -> int:
    out.append(f'write_triage attach-target invariant probe — src-root={src_root}')
    module = _import_judge(src_root)
    out.append(f'judge module: {getattr(module, "__file__", "<unknown>")}')

    slate, target, index = _build_slate(module)
    slate_ids = [getattr(c, 'id', None) for c in slate]
    out.append(
        f'slate: {slate_ids!r} — attach target {target.id!r} at index {index} '
        f'of {len(slate)}',
    )

    holds_a, reason_a = _option_a_verdict(module, slate)
    out.append(f'option (a): {reason_a}')
    if holds_a:
        out.append(
            'PASS  the judge path binds a verdict to a determinate candidate '
            '(option (a)).',
        )
        out.extend(_pass_scope_note())
        return EXIT_OK

    build = _require(module, 'build_judge_prompt')
    other = slate[0]
    if other is target:
        # The fixture puts the rescued winner LAST, so this cannot happen —
        # but a swap test that compared a rendering with itself would pass
        # vacuously, which is exactly the failure mode this gate exists to
        # prevent. Refuse rather than "pass".
        raise _Unverifiable(
            'the rescued attach target is also slate[0]; there is no second target '
            'to swap against, so the invariant is unverifiable on this slate',
        )

    if not _target_parameters(build):
        out.append(
            'option (b): build_judge_prompt'
            f'{inspect.signature(build)} takes no attach-target parameter',
        )
        out.extend(_FAIL_NEITHER)
        out.extend(_rescue_note(slate_ids, index))
        return EXIT_FAIL

    search = _search_option_b(build, slate, target, other)
    attempts = list(dict.fromkeys(search.attempts))
    if search.winner is None:
        if not search.rendered_any:
            # Not one combination produced a pair of renderings to compare, so
            # nothing was ever asserted about the invariant. Unverifiable, not
            # failed.
            raise _Unverifiable(
                'build_judge_prompt could not be rendered with an attach target '
                f'({_first_few(attempts)})',
            )
        named = [
            param.name
            for _, param in _target_parameters(build)
            if _names_a_target(param)
        ]
        if named and not search.target_named_rendered:
            # SOME parameter rendered, so the branch above did not fire -- but
            # not one that could have decided this. Reporting the attach target
            # INDETERMINATE here would state a fact this run did not establish
            # (and point the reader at a remedy that may already be present),
            # so say what was actually measured. Still fails closed: an
            # unverifiable invariant is not a satisfied one.
            raise _Unverifiable(
                f'build_judge_prompt takes {named!r}, named for the attach target, '
                'but no call reached it — every attempt failed before a pair of '
                'renderings existed to compare. Only parameters that do NOT name '
                'the target rendered, so nothing here measured whether the '
                f'rendering depends on the named candidate ({_first_few(attempts)})',
            )
        out.append(f'option (b): {_first_few(attempts, limit=len(attempts))}')
        out.extend(_FAIL_NEITHER)
        out.extend(_rescue_note(slate_ids, index))
        return EXIT_FAIL

    param_index, param, spelling, forgiven_echo = search.winner
    out.append(
        f'option (b): satisfied — build_judge_prompt accepts an attach target via '
        f'{param.name!r} (parameter {param_index}, spelled as the candidate '
        f'{spelling}) and its rendering depends on which candidate is the attach '
        'target',
    )
    if forgiven_echo:
        out.extend(_echo_forgiven_note(param))
    out.append(
        'PASS  the judge path binds a verdict to a determinate candidate '
        '(option (b)).',
    )
    out.extend(_pass_scope_note())
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--src-root',
        required=True,
        help='directory containing the fused_memory package (a fused-memory/src tree)',
    )
    args = parser.parse_args(argv)

    out: list[str] = []
    try:
        rc = _probe(Path(args.src_root), out)
    except _Unverifiable as exc:
        out.append(f'FAIL  UNVERIFIABLE: {exc}')
        out.append('      Failing closed — an unverifiable invariant is not a satisfied one.')
        rc = EXIT_FAIL
    except BaseException:  # noqa: BLE001 - the probe itself must never pass by crashing
        # BaseException, not Exception. This probe EXECUTES arbitrary code out
        # of the ref's own tree, and the realistic escape is SystemExit — a
        # lazily-imported dependency's import guard, or an argparse-style bail,
        # calling sys.exit(). SystemExit is not an Exception, so it slipped
        # past every inner handler AND past this arm, and terminated the
        # process with ITS OWN code. Measured: a judge whose
        # select_judge_candidates raises SystemExit(0) exited 0 having printed
        # zero bytes, and the gate read that as PASS and authorised the flip.
        # _import_judge already caught BaseException for exactly this reason;
        # the convention simply was not applied at the outer arm.
        #
        # This is the SINGLE CHOKEPOINT for that class: the inner handlers
        # (_render, _option_a_verdict, _echoes_payload, _build_slate) each
        # catch Exception for DIAGNOSTIC reasons — to attribute a failure to
        # one combination and keep searching — and a SystemExit out of any of
        # them lands here. Widening them individually would be both redundant
        # and worse, since a SystemExit is never evidence about one candidate.
        #
        # KeyboardInterrupt is deliberately included. A gate that reports
        # 'unverifiable' on an interrupted run is correct; one that lets a
        # Ctrl-C read as anything else is not.
        #
        # argparse stays OUTSIDE this try, so --help and a bad --src-root keep
        # their own exit codes instead of being reported as a failed probe.
        out.append('FAIL  UNVERIFIABLE: the probe raised while evaluating the invariant.')
        out.append('      Failing closed — an unverifiable invariant is not a satisfied one.')
        out.extend('      ' + line for line in traceback.format_exc().splitlines())
        rc = EXIT_FAIL
    sys.stdout.write('\n'.join(out) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
