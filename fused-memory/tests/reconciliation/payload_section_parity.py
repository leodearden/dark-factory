"""Shared AST machinery for the recon payload builder-parity guards. (tasks 4708, 5113)

Two guards assert the same structural invariants over two stages: the stage's
``REQUIRED_SECTIONS`` registry is well-formed, and every payload builder of the
stage routes its payload through that stage's ``_render_required_sections``
aggregator, on EVERY payload-returning branch.

* ``test_stage1_payload_section_parity.py`` — ``MemoryConsolidator`` (Stage 1).
* ``test_stage2_payload_section_parity.py`` — ``TaskKnowledgeSync`` (Stage 2).

WHY A MODULE RATHER THAN A COPY, OR A CROSS-SUITE IMPORT. The registry shape
check, the payload predicate and the interpolation check are each one
definition with two consumers. A verbatim
copy in each guard would let them drift: a fix to one guard's predicate — the
bare-``AnnAssign`` and every-branch amendments task 4708 needed are the
precedent — would silently skip the other. Importing the helpers out of the
Stage-1 test module instead is the reach-into-a-test-module smell that
``consolidator_fixtures.py`` documents and moved away from.

Follows the ``plural_enum_shapes.py`` / ``consolidator_fixtures.py`` precedent:
a plain non-test module (never collected), imported as ``from
reconciliation.payload_section_parity import …``, with PUBLIC names because it
exists to be imported. What each guard's predicate selects and rejects in its
own source file is stage-specific and is recorded in that guard, not here.
"""

from __future__ import annotations

import ast
import pathlib
from typing import NamedTuple

from _ast_guard import parse_python_module

from fused_memory.reconciliation.stages.base import RequiredSection

AGGREGATOR = '_render_required_sections'

# Every registered section is a level-3 heading inside the payload.
SECTION_HEADER_PREFIX = '### '

# A payload builder returns a WHOLE payload, which in the recon stage modules
# always means an f-string opening with the top-level markdown header.
PAYLOAD_HEADER_PREFIX = '## '


def registry_shape_violations(stage_cls: type) -> list[str]:
    """Every way ``stage_cls.REQUIRED_SECTIONS`` departs from the registry contract.

    Empty when the registry is a non-empty tuple of ``stages.base.RequiredSection``
    whose headers are level-3 markdown headings and whose renderers NAME
    callable attributes of *stage_cls*. Which sections a stage must register is
    stage-specific and pinned as a value in that stage's guard, not here.
    """
    registry_name = f'{stage_cls.__name__}.REQUIRED_SECTIONS'
    registry = getattr(stage_cls, 'REQUIRED_SECTIONS', None)
    if not isinstance(registry, tuple):
        return [
            f'{registry_name} must be a tuple — an immutable class-level '
            f'declaration read by every payload builder — got {type(registry).__name__}.'
        ]
    if not registry:
        return [
            f'{registry_name} is empty, which makes the aggregator a no-op and '
            f'every parity assertion over that stage vacuous.'
        ]
    violations: list[str] = []
    for section in registry:
        if not isinstance(section, RequiredSection):
            violations.append(
                f'{registry_name} member {section!r} is a {type(section).__name__}, '
                f'not a stages.base.RequiredSection — the one type every stage '
                f'declares its registry with; do not define a per-stage copy.'
            )
            continue
        if not (isinstance(section.header, str) and section.header.startswith(SECTION_HEADER_PREFIX)):
            violations.append(
                f'{registry_name} member {section!r} has header {section.header!r}; '
                f'it must be the exact markdown header the shipped prompt names, '
                f'a level-3 heading ({SECTION_HEADER_PREFIX!r}…).'
            )
        if not isinstance(section.renderer, str):
            violations.append(
                f'{registry_name} member {section!r} has renderer '
                f'{section.renderer!r}; it must be the NAME of a {stage_cls.__name__} '
                f'method (a str the aggregator dispatches via getattr), not the method '
                f'object.'
            )
        elif not callable(getattr(stage_cls, section.renderer, None)):
            violations.append(
                f'{registry_name} member {section.header!r} names renderer '
                f'{section.renderer!r}, which is not a callable attribute of '
                f'{stage_cls.__name__}. A typo in the renderer string must fail in '
                f'the guard, not as an AttributeError at payload-assembly time.'
            )
    return violations


class PayloadBuilder(NamedTuple):
    """One discovered payload builder and every branch of it returning a payload."""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    payload_returns: tuple[ast.Return, ...]


def _is_a_whole_payload(node: ast.Return) -> bool:
    """True when *node* returns a whole stage payload.

    A whole payload is an f-string whose LEADING literal chunk opens with
    ``'## '`` — the top-level markdown header that makes a string an entire
    payload rather than one section of one.
    """
    if not isinstance(node.value, ast.JoinedStr):
        return False
    values = node.value.values
    lead = values[0] if values else None
    return (
        isinstance(lead, ast.Constant)
        and isinstance(lead.value, str)
        and lead.value.startswith(PAYLOAD_HEADER_PREFIX)
    )


def _payload_returns(node: ast.AST) -> list[ast.Return]:
    """EVERY ``Return`` under *node* that returns a whole stage payload.

    ALL of them, not the first one found. A builder may grow a second
    payload-returning branch — an early-return short payload ahead of the full
    one is the obvious shape — and checking only one branch would leave the
    other free to omit the required sections silently, which is precisely the
    drift these guards exist to prevent. Checking the whole set also removes a
    trap for the next reader: ``ast.walk`` is BREADTH-first, so "the first
    payload return" is not source order and which branch got checked would not
    be obvious from reading the file. Sorted by position so failure messages
    name the offending branches in source order.

    Returns the nodes themselves (not a count or a bool) so callers can assert
    over each f-string's interpolations and report its ``lineno``.
    """
    returns = [
        descendant
        for descendant in ast.walk(node)
        if isinstance(descendant, ast.Return) and _is_a_whole_payload(descendant)
    ]
    returns.sort(key=lambda ret: (ret.lineno, ret.col_offset))
    return returns


def discover_payload_builders(source: pathlib.Path, class_name: str) -> dict[str, PayloadBuilder]:
    """Every payload builder on the top-level class *class_name* in *source*.

    DERIVED, not hand-listed — that is the whole point. A new payload builder
    is pulled into scope the day it is added, with no edit to either guard. A
    hand-maintained list of names would pin regression in the builders already
    fixed and stay blind to the next one, which is the defect tasks 4708 and
    5113 close.

    Predicate: a method directly in the class body with at least one
    ``Return`` of an f-string opening with ``'## '``. The class body is
    iterated DIRECTLY rather than via ``ast.walk``: a helper function nested
    inside a builder is not itself a builder.
    """
    tree = parse_python_module(source)
    class_def = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    assert class_def is not None, (
        f'{source.name} no longer defines a top-level class {class_name!r}. '
        f'Builder discovery cannot run; fix the class name in the guard rather '
        f'than letting it check nothing.'
    )
    builders: dict[str, PayloadBuilder] = {}
    for node in class_def.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        payload_returns = _payload_returns(node)
        if payload_returns:
            builders[node.name] = PayloadBuilder(node, tuple(payload_returns))
    return builders


def _aggregator_call(node: ast.AST) -> bool:
    """True when *node* is a call to ``self._render_required_sections(…)``.

    ARGUMENT-AGNOSTIC: only the callee's name is checked, because the stages'
    aggregators take different arguments (Stage 2's takes the payload's
    resolved task tree). The aggregators are coroutine methods (task 3778), so
    the call arrives wrapped in ``await``; the wrapper is unwrapped first.
    """
    if isinstance(node, ast.Await):
        node = node.value
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == AGGREGATOR
    )


def _names_bound_to_the_aggregator(builder: ast.AST) -> set[str]:
    """Locals in *builder* assigned the aggregator's output.

    Lets the parity check accept the indirect spelling
    ``x = self._render_required_sections(…)`` … ``f"…{x}…"`` as well as direct
    interpolation, so the guard proves the rendered text reaches the returned
    payload without pinning one brittle spelling.
    """
    bound: set[str] = set()
    for descendant in ast.walk(builder):
        if isinstance(descendant, ast.Assign) and _aggregator_call(descendant.value):
            bound.update(
                target.id for target in descendant.targets if isinstance(target, ast.Name)
            )
        elif (
            isinstance(descendant, ast.AnnAssign)
            and isinstance(descendant.target, ast.Name)
            # A bare annotation (``x: str``) has ``value=None`` and binds nothing.
            and descendant.value is not None
            and _aggregator_call(descendant.value)
        ):
            bound.add(descendant.target.id)
    return bound


def _payload_interpolates_the_aggregator(builder: ast.AST, payload_return: ast.Return) -> bool:
    """True when the returned payload f-string carries the aggregator's output."""
    bound = _names_bound_to_the_aggregator(builder)
    assert isinstance(payload_return.value, ast.JoinedStr)
    for value in payload_return.value.values:
        if not isinstance(value, ast.FormattedValue):
            continue
        if _aggregator_call(value.value):
            return True
        if isinstance(value.value, ast.Name) and value.value.id in bound:
            return True
    return False


def branches_missing_the_aggregator(builder: PayloadBuilder) -> list[int]:
    """Line numbers of *builder*'s payload branches that omit the aggregator.

    Empty when every payload-returning branch interpolates
    ``self._render_required_sections(…)``, directly or through a local bound
    to it. In source order, so a failure message names the branches as a
    reader would find them.
    """
    return [
        payload_return.lineno
        for payload_return in builder.payload_returns
        if not _payload_interpolates_the_aggregator(builder.node, payload_return)
    ]
