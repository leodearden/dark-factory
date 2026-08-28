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
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_FAIL = 1

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
    if origin is None or not str(Path(origin).resolve()).startswith(str(src_root.resolve())):
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
    if _CHILD_ID not in ids:
        raise _Unverifiable(
            'select_judge_candidates did not rescue the hoisted parent\'s evidence '
            f'child into the slate (got {ids!r}); the probe has no attach target to '
            'reason about',
        )
    index = ids.index(_CHILD_ID)
    return slate, slate[index], index


def _target_parameter(fn: Any) -> Any:
    """The third accepted parameter of *fn*, or ``None``.

    Positional-or-keyword and keyword-only both count: the point is whether
    the renderer can be TOLD which candidate the attach will touch, not how
    the argument is spelled.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None
    kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    usable = [p for p in params if p.kind in kinds]
    if len(usable) < 3:
        return None
    return usable[2]


def _value_for(candidate: Any, spelling: str) -> Any:
    return candidate.id if spelling == 'id' else candidate


def _call_render(fn: Any, param: Any, slate: list[Any], value: Any) -> str:
    if param.kind is inspect.Parameter.KEYWORD_ONLY:
        return fn(_NEW_ENTRY, slate, **{param.name: value})
    return fn(_NEW_ENTRY, slate, value)


def _pick_spelling(fn: Any, param: Any, slate: list[Any], target: Any) -> tuple[str, str]:
    """Render once, returning ``(spelling, rendering)``.

    Tries the id spelling first, then the object spelling. A renderer that
    raises under BOTH is unverifiable — it is not evidence the invariant holds.
    """
    errors: list[str] = []
    for spelling in _SPELLINGS:
        try:
            rendered = _call_render(fn, param, slate, _value_for(target, spelling))
        except Exception as exc:  # noqa: BLE001 - any render failure is unverifiable
            errors.append(f'{spelling}={exc!r}')
            continue
        if not isinstance(rendered, str):
            errors.append(f'{spelling}=returned {type(rendered).__name__}, not str')
            continue
        return spelling, rendered
    raise _Unverifiable(
        'build_judge_prompt could not be rendered with an attach target '
        f'({"; ".join(errors)})',
    )


def _render(fn: Any, param: Any, slate: list[Any], target: Any, spelling: str) -> str:
    """Render *slate* naming *target*, using an already-chosen *spelling*."""
    try:
        rendered = _call_render(fn, param, slate, _value_for(target, spelling))
    except Exception as exc:  # noqa: BLE001 - any render failure is unverifiable
        raise _Unverifiable(
            f'build_judge_prompt raised rendering attach target {target.id!r}: {exc!r}',
        ) from exc
    if not isinstance(rendered, str):
        raise _Unverifiable(
            f'build_judge_prompt returned {type(rendered).__name__}, not str, '
            f'for attach target {target.id!r}',
        )
    return rendered


# --- option (a): the verdict names its own candidate --------------------------
#
# Evaluated FIRST, and deliberately tolerant. Option (a) — a verdict that
# carries the judged candidate id — makes slate position irrelevant, so it
# closes item 1 with a prompt that marks nothing at all. Task 4798 item 7
# still carries option (a) as the better long-term design; a gate shaped
# around option (b) alone would fail it and re-block task 3169, which is the
# very defect (mirrored) that task 4810 exists to remove.

#: Field names an option-(a) wire contract plausibly uses for the candidate.
#: Tried only as EXTRA keys on the probe payload: main ignores them, and a
#: parser that demands one is not thereby failed.
_ID_FIELDS = ('candidate_id', 'id')

#: A verdict that is one of these is a bare outcome, not a binding to a
#: candidate. ``bool`` is listed before ``int`` reaches it for clarity only —
#: ``isinstance`` covers the subclass either way.
_NOT_A_CANDIDATE_BINDING = (str, bytes, bool, int, float)


def _binds_candidate(result: Any) -> bool:
    """True when *result* is something that can carry a candidate id.

    ``None`` and the scalar types are excluded so a parser that quietly
    returns ``None`` on an unrecognised payload — rather than raising — is not
    mistaken for one that returns an ``(outcome, candidate_id)`` pair.
    """
    return result is not None and not isinstance(result, _NOT_A_CANDIDATE_BINDING)


def _verdict_key(module: Any) -> str:
    key = getattr(module, 'VERDICT_KEY', None)
    return key if isinstance(key, str) and key else 'verdict'


def _verdict_words(module: Any) -> tuple[str, ...]:
    """The judge's own vocabulary, read from the module where it exposes one."""
    words = getattr(module, 'JUDGE_VERDICTS', None)
    if isinstance(words, dict) and words:
        return tuple(str(word) for word in words)
    return ('distinct', 'restates', 'amends', 'contests')


def _option_a_verdict(module: Any, slate: list[Any]) -> tuple[bool, str]:
    """``(holds, reason)`` for the option-(a) branch.

    A raise on every payload is INCONCLUSIVE, not a failure: a stricter
    option-(a) parser may demand a slate argument this probe cannot supply.
    The caller falls through to option (b) in that case.
    """
    parse = getattr(module, 'parse_judge_verdict', None)
    if parse is None:
        return False, 'parse_judge_verdict is absent from the module'
    key = _verdict_key(module)
    ident = str(getattr(slate[0], 'id', ''))
    saw_bare_str = False
    errors: list[str] = []
    for word in _verdict_words(module):
        payloads: list[dict[str, str]] = [{key: word}]
        payloads.extend({key: word, field: ident} for field in _ID_FIELDS)
        for payload in payloads:
            try:
                result = parse(json.dumps(payload))
            except Exception as exc:  # noqa: BLE001 - a rejection is inconclusive here
                errors.append(f'{payload!r} -> {exc!r}')
                continue
            if _binds_candidate(result):
                return True, (
                    f'satisfied — parse_judge_verdict returned '
                    f'{type(result).__name__} {result!r} for {payload!r}, so the '
                    'verdict itself names its candidate and the position of the '
                    'attach target in the slate stops mattering'
                )
            if isinstance(result, str):
                saw_bare_str = True
            else:
                errors.append(f'{payload!r} -> returned {type(result).__name__}')
    if saw_bare_str:
        return False, (
            'parse_judge_verdict returns a bare str — the verdict names no '
            'candidate, so the attach must guess one from the slate'
        )
    shown = '; '.join(errors[:3])
    if len(errors) > 3:
        shown += f'; …{len(errors) - 3} more'
    return False, f'inconclusive — parse_judge_verdict accepted no probe payload ({shown})'


# --- the swap test ----------------------------------------------------------
#
# Render the SAME slate twice against two DIFFERENT attach targets. Everything
# below is stated in terms of opaque ids and rendering inequality; no heading
# text is ever matched, so any rewording survives.


def _id_lines(rendering: str, ident: str) -> tuple[int, ...]:
    """The line indices where *ident* appears — its rendered footprint."""
    return tuple(i for i, line in enumerate(rendering.splitlines()) if ident in line)


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
    """``None`` when the swap test holds, else the reason it does not."""
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
    for named, unnamed, ident in (
        (rendered_target, rendered_other, str(target.id)),
        (rendered_other, rendered_target, str(other.id)),
    ):
        if _id_lines(named, ident) == _id_lines(unnamed, ident):
            return (
                'the two renderings differ, but the id ' + repr(ident) + ' occupies '
                'the SAME lines whether or not it is the named attach target, so the '
                'difference is not attributable to the candidate being named'
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
    """The measured warning that marking ``candidates[0]`` is not sufficient."""
    return [
        '      Marking candidates[0] as the attach target is NOT sufficient:',
        '      select_judge_candidates rescues a hoisted parent\'s evidence child by',
        '      APPENDING it (selected = [*selected[: max(n - 1, 0)], winner]), so on',
        '      a hoisted-parent slate the attach target is LAST, not first. Measured',
        f'      on this probe\'s own fixture: slate {slate_ids!r}, attach target at',
        f'      index {index}.',
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
        return EXIT_OK

    build = _require(module, 'build_judge_prompt')
    param = _target_parameter(build)
    if param is None:
        out.append(
            'option (b): build_judge_prompt'
            f'{inspect.signature(build)} takes no attach-target parameter',
        )
        out.extend(_FAIL_NEITHER)
        out.extend(_rescue_note(slate_ids, index))
        return EXIT_FAIL

    spelling, rendered_target = _pick_spelling(build, param, slate, target)
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
    rendered_other = _render(build, param, slate, other, spelling)
    reason = _swap_verdict(slate, target, other, rendered_target, rendered_other)
    if reason is not None:
        out.append(f'option (b): {reason}')
        out.extend(_FAIL_NEITHER)
        out.extend(_rescue_note(slate_ids, index))
        return EXIT_FAIL

    out.append(
        f'option (b): satisfied — build_judge_prompt accepts an attach target via '
        f'{param.name!r} (spelled as the candidate {spelling}) and its rendering '
        f'depends on which candidate is the attach target',
    )
    out.append(
        'PASS  the judge path binds a verdict to a determinate candidate '
        '(option (b)).',
    )
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
    except Exception:  # noqa: BLE001 - the probe itself must never pass by crashing
        out.append('FAIL  UNVERIFIABLE: the probe raised while evaluating the invariant.')
        out.append('      Failing closed — an unverifiable invariant is not a satisfied one.')
        out.extend('      ' + line for line in traceback.format_exc().splitlines())
        rc = EXIT_FAIL
    sys.stdout.write('\n'.join(out) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
