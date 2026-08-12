"""Plan-tools MCP server -- stdio tool server for safe plan.json operations.

Spawned per-agent invocation by the orchestrator.  The architect agent
builds plans via ``create_plan`` / ``add_plan_step`` / etc., and the
implementer marks steps done via ``mark_step_done``.  On revalidation
(blast-radius requeue), the architect uses ``update_plan_metadata``,
``remove_plan_step``, ``replace_plan_step``, and ``confirm_plan`` to
update an existing plan without recreating it from scratch.  All writes
go through ``TaskArtifacts`` methods, preserving ``_session_id`` and
enforcing correct schema.

plan.json is also REPAIRED LAZILY ON READ.  Live plans in the fleet carry
MCP tool-call envelope markup that leaked into their prose fields when the
harness mis-closed an argument — a trailing mis-close, or a whole sibling
parameter absorbed into the value before it.  Every entry point that opens
an existing plan goes through :func:`_read_plan_repaired`, which repairs
what it can, writes the result back through :func:`_atomic_write_plan`
(PRD contract C3, scoped to THAT write: a concurrent reader never
observes a partial *repair* write-back — the tools' own mutation write
still goes through ``TaskArtifacts.write_plan``, which is truncate-then-
write, and closing that window needs a change to ``TaskArtifacts`` that
is out of this module's scope), and reports every repair and every
refusal on the response's ``markup_repairs``
key.  No fleet quiesce is needed: the next tool call an agent makes fixes
its own plan.  ``shared.toolcall_markup`` is the SINGLE owner of the
literal set and of every accept/refuse decision (INV-5) — this module
enumerates only WHICH plan fields are prose (``_REPAIRABLE_PLAN_FIELDS``),
never what corruption looks like.

Usage (stdio transport, spawned by orchestrator):
    # Direct-interpreter no-uv hot path (production, task 1776):
    <sys.executable> -m orchestrator.mcp.plan_tools --worktree /path/to/worktree

    # uv fallback with fast-start flags (task 1775, backward-compat):
    uv run --project <orchestrator-dir> --no-sync --frozen \
        python -m orchestrator.mcp.plan_tools --worktree /path/to/worktree
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import logging
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from fastmcp import FastMCP
from shared.toolcall_markup import detect, repair

from orchestrator.artifacts import PLAN_SCHEMA_VERSION, TaskArtifacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# files-arg recovery helper (task 2428)
# ---------------------------------------------------------------------------


# Bracket/quote characters that, if still present in a delimiter-split
# fallback result, indicate the split did not fully recover clean file
# paths (i.e. the input was corrupted in a shape we don't parse).
_RESIDUAL_CORRUPTION_CHARS = frozenset("[]{}'\"")


def _coerce_files(files: list[str] | str | None) -> list[str]:
    """Coerce a possibly mis-serialized ``files`` MCP arg into a clean list.

    The agent harness intermittently mis-serializes the ``create_plan``
    (and ``update_plan_metadata``) call when a large/complex ``analysis``
    string accompanies the ``files`` array in the same call, corrupting
    ``files`` into a stringified JSON array, a bare JSON string, a
    Python-repr-style single-quoted list, or a comma/newline-delimited
    plain string. This recovers all of those shapes back into a clean
    ``list[str]``; ``None`` maps to ``[]``. Shapes that cannot be
    confidently recovered are logged (rather than silently accepted) so
    the corruption stays visible to operators.
    """
    if files is None:
        return []
    if isinstance(files, list):
        return [str(f).strip() for f in files if str(f).strip()]

    text = files.strip()
    if not text:
        return []

    try:
        decoded = json.loads(text)
    except ValueError:
        decoded = None

    if isinstance(decoded, list):
        return [str(f).strip() for f in decoded if str(f).strip()]
    if isinstance(decoded, str):
        decoded = decoded.strip()
        return [decoded] if decoded else []
    if decoded is not None:
        # Valid JSON, but not a list/str (e.g. a dict or a number) is not a
        # recognized files shape. Fall through to the delimiter split below
        # for a best-effort recovery, but flag it — for a dict that split
        # degrades to the whole brace-string as a single "path" rather than
        # a real recovery.
        logger.warning(
            "_coerce_files: files arg decoded to unrecognized JSON shape "
            "%s (expected list or str); falling back to delimiter-split "
            "of raw text %r",
            type(decoded).__name__,
            text,
        )
    elif text.startswith('[') and text.endswith(']'):
        # Not valid JSON. Try a Python-repr-style stringified list next —
        # e.g. "['a.py', 'b.py']" (single-quoted) is a plausible
        # mis-serialization shape that json.loads cannot parse.
        try:
            literal = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            literal = None
        if isinstance(literal, list):
            return [str(f).strip() for f in literal if str(f).strip()]

    # Last resort: split on commas/newlines, the other corrupted shape
    # observed in practice.
    parts = [part.strip() for part in re.split(r'[,\n]', text) if part.strip()]
    if any(_RESIDUAL_CORRUPTION_CHARS & set(part) for part in parts):
        logger.warning(
            "_coerce_files: delimiter-split fallback produced entries with "
            "residual bracket/quote characters — input may still be "
            "corrupted: %r",
            parts,
        )
    return parts


# ---------------------------------------------------------------------------
# Envelope-markup repair on read (task 3692, PRD contract C3, boundary row B12)
# ---------------------------------------------------------------------------


class _PlanField(NamedTuple):
    """One repairable plan field, and the tool schema it must validate against.

    ``collection`` is the plan key holding a LIST OF DICTS the field lives in,
    or ``None`` for a top-level plan key.

    ``schema_params`` and ``target_keys`` are DELIBERATELY ASYMMETRIC, and the
    asymmetry is the whole point of carrying both.

    ``schema_params`` is the ORIGINATING TOOL'S FULL PARAMETER LIST. ``repair()``
    needs it complete so it can RECOGNISE a tail that names any real parameter
    of that call — a name outside the list is not a leaked argument at all and
    the candidate is refused.

    ``target_keys`` is the strictly smaller subset of those parameters that name
    a PROSE PLAN KEY a recovered value may legally be written to, mapped to the
    key to write it under. The two vocabularies are NOT the same: the tool's
    parameter names and the plan document's key names differ. A parameter absent
    from ``target_keys`` is one of three things, and none of them is a legal
    recovery target:

    * an identifier the plan stores under a different key — ``prereq_id`` and
      ``step_id`` are both written as ``id``;
    * an enum the plan stores under a different key — ``step_type`` is written
      as ``type``;
    * a non-prose value — ``files`` is a list, ``task_id`` an identifier.

    Keying a recovery on the parameter name alone would therefore write a JUNK
    KEY onto the record (``step_type`` beside the real, still-corrupt ``type``)
    or a bare ``str`` over the ``files`` list. Both are silent-wrong-value
    corruption of the plan artifact, which is the damage class this surface
    exists to end.

    ``also_written_by`` names the OTHER tools that write this same field, and
    exists so the structured fact does not overstate its own precision. The
    ``tool`` a fact reports is the collection's SCHEMA OWNER — the call whose
    parameter vocabulary ``repair()`` validates against — which is not always
    the call that produced the corruption: ``replace_plan_step`` also writes
    ``steps[].description`` AND ``prerequisites[].description`` (its lookup loop
    spans both collections), and ``update_plan_metadata`` also writes
    ``analysis``. Naming only the owner would send whoever triages a fact to the
    wrong call site; naming the alternates alongside it keeps the diagnosis
    honest without pretending the fact can identify the writer it cannot see.
    """

    collection: str | None
    field: str
    schema_params: tuple[str, ...]
    target_keys: Mapping[str, str]
    also_written_by: tuple[str, ...]


def _params_of(fn: Callable[..., Any]) -> tuple[str, ...]:
    """The originating tool's parameter names, read off its LIVE signature.

    The leading ``artifacts`` is dropped: it is injected by the server wrapper,
    never a name the harness could leak out of a sibling argument.

    DERIVED, not restated. A hand-copied tuple would be a second spelling of a
    signature this table does not own, and the only way to keep the two in step
    would be a test that pins parameter names by introspection — coverage of the
    copy rather than of any behaviour a caller can observe. Deriving makes the
    drift impossible by construction, which is what INV-1 actually asks for.
    """
    names = tuple(inspect.signature(fn).parameters)
    if names[:1] != ('artifacts',):
        raise RuntimeError(
            f'{fn.__name__} no longer takes artifacts first: _params_of would '
            f'return {names!r}, admitting a non-parameter name as a recovery '
            'candidate. Fix the derivation, do not widen it.'
        )
    return names[1:]

# Which of those parameters name a PROSE PLAN KEY, and which key. Spelled one
# mapping per tool — this half CANNOT be derived, because the plan document's
# key vocabulary is not the tool's — and wrapped in MappingProxyType because a
# DECLARED surface that could be mutated at runtime is not auditable.
#
# Note what is MISSING from each, deliberately. ``task_id`` and ``files`` are
# not prose. ``prereq_id`` and ``step_id`` are stored as ``id``; ``step_type``
# is stored as ``type`` — so keying a recovery on the parameter name would
# create a junk key and leave the real one corrupt. The plan-key values are
# machine-checked against a plan built by these very tools and read back
# (``TestRepairableFieldTable::test_every_target_value_is_a_key_the_real_writers_produce``),
# so this mapping cannot drift from what the writers actually produce.
_CREATE_PLAN_TARGETS: Mapping[str, str] = MappingProxyType(
    {'title': 'title', 'analysis': 'analysis'}
)
_ADD_PREREQUISITE_TARGETS: Mapping[str, str] = MappingProxyType({'description': 'description'})
_ADD_PLAN_STEP_TARGETS: Mapping[str, str] = MappingProxyType({'description': 'description'})
_ADD_DESIGN_DECISION_TARGETS: Mapping[str, str] = MappingProxyType(
    {'decision': 'decision', 'rationale': 'rationale'}
)
_ADD_REUSE_ITEM_TARGETS: Mapping[str, str] = MappingProxyType(
    {'what': 'what', 'where': 'where', 'how': 'how'}
)

#: The OTHER tools that write a given field, per row of the table below. Each
#: entry is machine-checked to name a real plan-tools entry point that actually
#: takes that parameter
#: (``TestRepairableFieldTable::test_every_alternate_writer_is_a_real_tool_taking_that_field``),
#: so an alternate cannot drift into a prose label.
_REPLACE_STEP_ALSO: tuple[str, ...] = ('replace_plan_step',)
_UPDATE_METADATA_ALSO: tuple[str, ...] = ('update_plan_metadata',)

#: THE declared enumeration of the repairable surface. Adding a prose field to
#: the plan schema means ADDING A ROW HERE — nothing infers the surface.
#:
#: Deliberately NOT a recursive walk of the plan dict. A heuristic walk cannot
#: supply the originating tool's parameter names, which is the whole basis on
#: which ``repair()`` decides a recovery is real rather than a guess (PRD D2,
#: INV-3 corroborate-before-acting); it would have to fabricate a schema. It
#: would also sweep in ``files`` (path entries, already recovered by
#: :func:`_coerce_files`) and every future non-prose key, rewriting values this
#: surface has no business touching.
#:
#: Bound ONCE, immediately below the five writer functions it derives its
#: ``schema_params`` from (``_params_of`` needs them to exist). Annotated but
#: deliberately UNBOUND here, so a use before that point raises a loud
#: NameError instead of silently reading an empty table and repairing nothing.
_REPAIRABLE_PLAN_FIELDS: tuple[_PlanField, ...]


def _build_repairable_plan_fields() -> tuple[_PlanField, ...]:
    """Construct :data:`_REPAIRABLE_PLAN_FIELDS`; see its docs above."""
    return (
        _PlanField(None, 'title', _params_of(_create_plan), _CREATE_PLAN_TARGETS, ()),
        _PlanField(
            None,
            'analysis',
            _params_of(_create_plan),
            _CREATE_PLAN_TARGETS,
            _UPDATE_METADATA_ALSO,
        ),
        _PlanField(
            'prerequisites',
            'description',
            _params_of(_add_prerequisite),
            _ADD_PREREQUISITE_TARGETS,
            _REPLACE_STEP_ALSO,
        ),
        _PlanField(
            'steps',
            'description',
            _params_of(_add_plan_step),
            _ADD_PLAN_STEP_TARGETS,
            _REPLACE_STEP_ALSO,
        ),
        _PlanField(
            'design_decisions',
            'decision',
            _params_of(_add_design_decision),
            _ADD_DESIGN_DECISION_TARGETS,
            (),
        ),
        _PlanField(
            'design_decisions',
            'rationale',
            _params_of(_add_design_decision),
            _ADD_DESIGN_DECISION_TARGETS,
            (),
        ),
        _PlanField(
            'reuse', 'what', _params_of(_add_reuse_item), _ADD_REUSE_ITEM_TARGETS, ()
        ),
        _PlanField(
            'reuse', 'where', _params_of(_add_reuse_item), _ADD_REUSE_ITEM_TARGETS, ()
        ),
        _PlanField(
            'reuse', 'how', _params_of(_add_reuse_item), _ADD_REUSE_ITEM_TARGETS, ()
        ),
    )

#: The collection's SCHEMA OWNER — the tool whose parameter vocabulary defines
#: the collection's fields, and which ``repair()`` validates a recovery against.
#: Reported as the structured fact's ``tool`` (contract C2's name, verbatim).
#:
#: NOT "the only tool that writes this collection", which would be false of this
#: module: ``replace_plan_step`` also writes ``description`` on both ``steps``
#: and ``prerequisites`` items, and ``update_plan_metadata`` also writes
#: ``analysis``. A fact therefore carries ``also_written_by`` beside ``tool``
#: (see :class:`_PlanField`) so a triager is not sent to a call site the
#: corruption may never have passed through. Asserted callable via
#: ``getattr(plan_tools, '_' + tool)`` by the repair tests, so the label cannot
#: drift into prose.
_COLLECTION_SCHEMA_TOOL: dict[str | None, str] = {
    None: 'create_plan',
    'prerequisites': 'add_prerequisite',
    'steps': 'add_plan_step',
    'design_decisions': 'add_design_decision',
    'reuse': 'add_reuse_item',
}


def _is_authored(value: object) -> bool:
    """True when *value* already holds real content, i.e. is NOT a hole.

    A blank/whitespace-only str, ``None``, and an empty list/dict are holes a
    recovery may legitimately fill. Anything else is authored content that a
    recovery must never displace — including a NON-str value such as the
    ``files`` list, which the string-only reading would have mistaken for a
    hole and happily overwritten with a recovered string.
    """
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _repair_one_field(
    holder: dict[str, Any],
    index: int | None,
    record: _PlanField,
) -> dict[str, Any] | None:
    """Repair ``holder[record.field]`` IN PLACE; return a fact, or ``None``.

    *holder* is the plan dict itself for a top-level record, or one item of
    ``plan[record.collection]`` otherwise. ``None`` means nothing to report:
    the value is absent, is not a str, or carries no envelope literal at all.

    THE CONTRACT IS BINARY. ``repair()`` validates -> the field becomes
    ``clean_value``. ``repair()`` declines -> the field is left BYTE-IDENTICAL
    and the refusal is reported as an ``'unrepairable'`` fact so the residue
    stays visible. There is deliberately NO fallback that strips the trailing
    bit anyway, and that is not caution for its own sake: measured across the
    28 corrupted live plans, the declining population is dominated by plans
    that legitimately QUOTE the sentinels in prose — worktree 2939's plan is a
    plan ABOUT this leak, so its decision/rationale fields discuss the closing
    and opening tags in ordinary sentences. A trailing-only sanitize would
    silently mutilate that authored text, which is the same class of
    silent-wrong-value damage this whole PRD exists to end. Never partial,
    never guessed.

    A RECOVERY ONLY EVER FILLS A HOLE. ``supplied`` is computed as the sibling
    fields of this same record that already hold authored content, so a
    recovered parameter can land only in an EMPTY or ABSENT sibling and can
    never overwrite real text. That rule is not fussiness: on disk the sibling
    key normally EXISTS (the tool wrote every field), and worktree 3024's live
    plans show the retry fingerprint — a corrupted decision still carrying the
    absorbed rationale while the record's own rationale field holds genuine,
    later-authored prose. At read time those two are indistinguishable, so
    clobbering the authored one would be precisely the silent-wrong-value
    failure this surface exists to prevent.

    The collision check itself is NOT re-implemented here: ``supplied`` is
    handed to ``repair()``, whose accept-time disjointness condition (PRD
    boundary row B9) does the refusing. One mechanism, per INV-5.

    THE ONE HOLE B9 CANNOT COVER IS THE FIELD ITSELF. ``supplied`` must exclude
    ``record.field`` — repair() is repairing that parameter, so declaring it
    already-supplied would refuse every candidate — which means a tail that
    RE-DECLARES the field being repaired passes the disjointness check, and both
    sides of the mis-close become candidates for the one field. WHICH SIDE HOLDS
    THE AUTHORED TEXT DEPENDS ON WHERE THE MIS-CLOSE SITS, so this is decided by
    content and never by position: the fill-a-hole rule above governs the
    field's own key exactly as it governs its siblings, read through the same
    ``_is_authored`` helper. The authored side wins in BOTH directions; only a
    redeclaration that would displace authored text is declined and reported on
    ``declined_params``.

    A RECOVERY MAY ONLY LAND ON A DECLARED PROSE KEY. The repairable surface
    distinguishes the TOOL's parameter vocabulary (``schema_params``) from the
    PLAN's key vocabulary (``target_keys``), and for three parameters they
    differ: ``prereq_id`` and ``step_id`` are stored as ``id``, ``step_type``
    as ``type``. Two more name no prose key at all — ``files`` is a list and
    ``task_id`` an identifier. Writing a recovered prose string into any of
    them is silent-wrong-value corruption of the plan artifact that the lock
    charter (``derive_modules`` / ``files_to_modules``) and the merge gate
    (``plan_files_not_touched``) both consume, which is the exact damage class
    this surface exists to end — so refusing conservatively there is the point,
    not a limitation.

    That rule is enforced through the SAME mechanism, not a second one: a
    parameter with no prose target is placed in ``supplied`` UNCONDITIONALLY,
    so ``repair()``'s existing B9 condition refuses any candidate whose tail
    recovers it and the field is left byte-identical. A parameter WITH a prose
    target has its authorship read from the key the plan actually uses
    (``holder.get('type')``, never the always-absent ``holder.get('step_type')``)
    — without which the fill-a-hole guard was inert for exactly those params,
    because the lookup could never see the authored value it was meant to
    protect.
    """
    value = holder.get(record.field)
    if not isinstance(value, str):
        return None
    pattern = detect(value)
    if pattern is None:
        # The cheap prefilter, and the overwhelmingly common path: no literal
        # anywhere in the value, so repair() is never called.
        return None

    supplied = frozenset(
        name
        for name in record.schema_params
        if name != record.field
        and (
            # No prose key to land on: ALWAYS supplied, so repair() refuses.
            name not in record.target_keys
            # Authorship read from the key the PLAN uses, not the tool's param.
            or _is_authored(holder.get(record.target_keys[name]))
        )
    )
    result = repair(
        value,
        param=record.field,
        schema_params=record.schema_params,
        supplied=supplied,
    )
    if result is None:
        return {
            'tool': _COLLECTION_SCHEMA_TOOL[record.collection],
            'also_written_by': list(record.also_written_by),
            'param': record.field,
            'pattern': pattern,
            # No mis-close was VALIDATED — every candidate was rejected — so
            # naming one here would be a guess dressed up as a diagnostic.
            'misclose': None,
            'outcome': 'unrepairable',
            'recovered_params': [],
            'declined_params': [],
            'collection': record.collection,
            'index': index,
            'field': record.field,
        }

    holder[record.field] = result.clean_value
    # Every recovered value is a verbatim slice of the absorbing field (D5,
    # guaranteed by construction in ``repair``), and every SIBLING target here
    # was proven a hole by the ``supplied`` set above. The field's own key is
    # deliberately NOT in ``supplied``, so it carries no such proof and is
    # decided on content just below.
    written: list[str] = []
    declined: list[str] = []
    for name, recovered_value in result.recovered.items():
        # THE TAIL RE-DECLARED THE FIELD BEING REPAIRED. ``supplied``
        # deliberately excludes ``record.field`` (repair() is repairing it,
        # so calling it already-supplied would refuse every candidate), and
        # repair()'s B9 disjointness check therefore ACCEPTS a tail that
        # recovers that same name. Both sides of the mis-close are then
        # candidates for the one field, and WHICH SIDE IS AUTHORED DEPENDS
        # ENTIRELY ON WHERE THE MIS-CLOSE SITS — so a fixed choice is wrong
        # in one direction whichever way it is fixed:
        #
        #   Mis-close near the END — authored prose, the mis-close, then a
        #   stub re-opener. ``clean_value`` is the prose and the recovered
        #   self-value is the stub. Keep ``clean_value``.
        #
        #   Mis-close at or near the START — the mis-close, then the real
        #   re-opener. ``clean_value`` is the HOLE (empty, or whitespace)
        #   and the recovered self-value is the authored prose. Keep the
        #   RECOVERED value. Declining here would blank the field, and
        #   because the fact still reads ``outcome: 'repaired'``,
        #   ``_read_plan_repaired`` would persist that blank through
        #   ``_atomic_write_plan`` — authored text destroyed on disk,
        #   unrecoverable on the next read. That is the same
        #   silent-authored-text-loss this whole surface exists to end,
        #   merely pointed the other way.
        #
        # So the SAME rule the siblings already follow decides it, through
        # the same ``_is_authored`` helper rather than a second policy
        # (INV-5): A RECOVERY ONLY EVER FILLS A HOLE. Decline when
        # ``clean_value`` is already authored, or when the recovered
        # self-value is not authored; otherwise fall through and let the
        # ordinary ``target_keys`` write path install it — ``record.field``
        # maps to itself for every record, which the repair tests assert, so
        # no separate write path is needed. When BOTH sides are holes the
        # decline keeps ``clean_value``, which is equivalent.
        #
        # A recovered self-value MAY itself still trip ``detect()``: a
        # nested opener with no closing tag parses into the recovered value,
        # since ``_parse_tail`` only refuses on an embedded closer. That is
        # why writing it is RIGHT and not merely tolerable — the prose
        # survives, and the next read re-detects it, finds no mis-close to
        # validate, and reports ``unrepairable``, leaving the residue
        # visible and byte-identical. Declining would have thrown the prose
        # away to reach that same visibility.
        if name == record.field and (
            _is_authored(result.clean_value) or not _is_authored(recovered_value)
        ):
            declined.append(name)
            continue
        key = record.target_keys.get(name)
        if key is None:
            # Unreachable while ``supplied`` holds every non-target: repair()
            # would have refused. Kept as total-safety so a future change to
            # repair() can never make this walk write a junk key — NOT as a
            # second policy, which is why it only declines and reports.
            logger.warning(
                'plan_markup_repair: refusing to write recovered %r — %s.%s '
                'declares no plan key for it',
                name,
                record.collection,
                record.field,
            )
            declined.append(name)
            continue
        holder[key] = recovered_value
        written.append(name)
    return {
        'tool': _COLLECTION_SCHEMA_TOOL[record.collection],
        'also_written_by': list(record.also_written_by),
        'param': record.field,
        'pattern': result.pattern,
        'misclose': result.misclose,
        'outcome': 'repaired',
        # What was recovered AND WRITTEN, versus what the tail declared and this
        # walk refused to write. Splitting them is what keeps a declined
        # recovery visible instead of reading as a successful one.
        'recovered_params': sorted(written),
        'declined_params': sorted(declined),
        'collection': record.collection,
        'index': index,
        'field': record.field,
    }


def _walk_repairable(plan: dict) -> Iterator[tuple[dict[str, Any], int | None, _PlanField]]:
    """Every ``(holder, index, record)`` the declared table addresses in *plan*.

    The surface walked is :data:`_REPAIRABLE_PLAN_FIELDS` — the DECLARED table —
    never an inferred walk of the plan dict. Non-dict items in a collection are
    skipped, mirroring the ``isinstance(item, dict)`` guards the mutation
    helpers already use, so the walk stays total for adversarial plan shapes.

    Shared by the prefilter scan and the repair pass so the two can never
    disagree about WHICH fields are in scope — one enumeration, two readers.
    """
    for record in _REPAIRABLE_PLAN_FIELDS:
        if record.collection is None:
            yield plan, None, record
            continue
        items = plan.get(record.collection)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if isinstance(item, dict):
                yield item, index, record


def _carries_markup(plan: dict) -> bool:
    """True when ANY declared field of *plan* trips :func:`detect`.

    The cheap prefilter over the whole document, run before anything is copied.
    ``detect`` is a substring scan; a deep copy of a plan is tens of KB of
    analysis prose plus every step description, and on the overwhelmingly common
    clean path that copy would be pure waste.
    """
    for holder, _index, record in _walk_repairable(plan):
        value = holder.get(record.field)
        if isinstance(value, str) and detect(value) is not None:
            return True
    return False


def _repair_plan_fields(plan: dict) -> tuple[dict, list[dict[str, Any]]]:
    """Repair every declared prose field of *plan*; return ``(plan, facts)``.

    NEVER MUTATES THE INPUT, and never touches the filesystem. Persisting the
    result is :func:`_read_plan_repaired`'s decision, not this pass's.

    THE COPY IS LAZY. A clean plan is returned AS-IS — the same object, not a
    copy — because nothing was going to change in it. Only once
    :func:`_carries_markup` finds a hit does the document get deep-copied and
    re-walked, so the copy is paid on the rare corrupted read rather than on
    every plan-tools call in the fleet. The no-mutation contract is unaffected:
    on the returned-as-is path this function writes nothing at all. Callers that
    intend to mutate what they get back must therefore own a fresh document —
    which :func:`_read_plan_repaired`'s callers do, since ``read_plan`` parses
    the file anew on every call.

    The prefilter runs ``detect`` a second time on the values that DO trip it
    (once here, once inside :func:`_repair_one_field`). That is deliberate: the
    duplicate pass is paid only on the corrupted path, where the walk is about
    to call ``repair()`` anyway, and it buys skipping the copy on the clean one.
    """
    if not _carries_markup(plan):
        return plan, []

    repaired = copy.deepcopy(plan)
    facts: list[dict[str, Any]] = []
    for holder, index, record in _walk_repairable(repaired):
        fact = _repair_one_field(holder, index, record)
        if fact is not None:
            facts.append(fact)

    return repaired, facts


class PlanWriteError(OSError):
    """An atomic plan write-back failed; the target was left UNTOUCHED.

    Carries the path in its message so the failure is actionable without
    log-scraping, and is chained (``raise ... from``) to the underlying cause.
    """


def _verify_plan_json(path: Path) -> None:
    """Re-read *path* and confirm it parses as JSON. Raises if it does not.

    A named seam, not an inlined ``json.load``: it is the last checkpoint
    before the swap becomes irreversible, so it must be independently
    exercisable by a test that injects a failure there.
    """
    with path.open(encoding='utf-8') as handle:
        json.load(handle)


def _target_file_mode(target: Path) -> int:
    """The permission bits *target* must carry AFTER the replace.

    ``tempfile.mkstemp`` forces 0600 and ``os.replace`` carries that mode onto
    the target, so without this the FIRST repair write-back would silently
    narrow plan.json from whatever ``TaskArtifacts._write_json`` created it as
    (0666 & ~umask, typically 0644) to owner-only — an invisible, permanent
    mutation of a file the lock charter and the merge gate both read, performed
    as a side effect of a READ. Today every consumer runs as the same uid, so it
    would break nothing and go unnoticed until a sandboxed reader with a
    different uid, or a group-readable operator workflow, touched the artifact.

    An existing target's own mode is preserved verbatim. For a target that does
    not exist yet the answer is what ``path.write_text`` would have produced,
    which keeps a fresh atomic write byte- AND mode-identical to
    ``TaskArtifacts.write_plan``. Reading the umask requires setting it (the
    only interface the OS offers) and restoring it immediately; plan-tools is a
    single-threaded stdio subprocess and this runs on the rare repair path, so
    the restored-in-two-statements window is not a real exposure — but it is why
    this is a named helper rather than an inline dance.
    """
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        current = os.umask(0)
        os.umask(current)
        return 0o666 & ~current


def _atomic_write_plan(path: Path, plan: dict) -> None:
    """Write *plan* to *path* atomically. Returns ``None``; raises on failure.

    PRD contract C3 / boundary row B12: a concurrent reader must never observe
    a partially written plan.json. Deliberately NOT routed through
    ``TaskArtifacts._write_json``, which is ``path.write_text`` —
    truncate-then-write, precisely the window B12 forbids. The byte format is
    reproduced exactly (``_schema_version`` stamp, ``json.dumps(indent=2)``
    plus a trailing newline) so a repair write-back cannot churn formatting, and
    the target's existing permission bits are carried across the swap (see
    :func:`_target_file_mode`) so an atomic write is invisible in every respect
    except its content.

    THE GUARANTEE IS SCOPED TO THIS WRITE, and the scope is a real limit rather
    than a formality. A plan-tools CALL typically repairs on the way in through
    :func:`_read_plan_repaired` (atomic, here) and then persists its own mutation
    microseconds later through ``artifacts.write_plan`` — which is still
    ``path.write_text``, i.e. torn-readable. So "no reader ever sees a partial
    plan.json" holds for the repair write-back, NOT for a plan-tools call taken
    as a whole. Closing that second window means making ``TaskArtifacts``
    itself write atomically, which is out of this task's module scope (its
    design decision 4 scopes atomicity to the write no agent asked for — the
    repair — precisely because converting ``_write_json`` would change the
    durability semantics of ~20 unrelated artifact writers); it is filed as
    follow-up work rather than half-done here.

    Order: RESOLVE the target, serialize, write to a temp file in the resolved
    target's own directory, flush, fsync, restore the mode, VERIFY it re-parses,
    and only then ``os.replace``. Any failure unlinks the temp and leaves the
    original byte-identical, then raises :class:`PlanWriteError` naming the path
    — loud, never a silent skip.

    RESOLVING FIRST IS LOAD-BEARING TWICE OVER, which is why it is the first
    statement rather than an incidental normalisation.
    ``TaskArtifacts.ensure_lane_plan_symlink`` makes the lane copy
    ``<worktree>/.task/plan.json`` an ABSOLUTE symlink onto the durable
    meta-root plan, and ``_artifacts_from_args`` still supports
    ``meta_root=None`` where ``self.root`` IS ``<worktree>/.task`` — so the
    path handed here can BE that symlink. ``os.replace`` onto a symlink
    replaces the LINK with a regular file, which would silently re-fork the
    lane and meta-root copies and recreate the esc-5205-9 stale-plan
    divergence the symlink exists to prevent. Resolving also guarantees the
    temp lands on the same filesystem as the real file, which is what makes
    the replace a rename rather than a cross-device error.

    A DANGLING link resolves to a path that does not exist (and whose parent
    may not either). That fails loudly, naming both the original and resolved
    paths, rather than materialising a stray regular file at the link path —
    which would BE the divergence, not a recovery from it.
    """
    target = Path(os.path.realpath(path))
    if not target.parent.is_dir():
        raise PlanWriteError(
            f'atomic plan write-back to {path} failed: its resolved target '
            f'{target} has no existing parent directory (a dangling symlink?) '
            '— refusing to materialise a stray file at the link path'
        )

    plan['_schema_version'] = PLAN_SCHEMA_VERSION
    payload = json.dumps(plan, indent=2) + '\n'

    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix='.plan.json.', suffix='.tmp'
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Before the swap, never after: the replaced file must already carry the
        # right bits, or a reader between the two calls sees the 0600 mkstemp
        # forced — a window with the same shape as the torn read B12 forbids.
        # Looked up HERE, inside the wrapper, so a stat() failure other than
        # FileNotFoundError (EACCES, ELOOP, ENAMETOOLONG — _target_file_mode
        # deliberately swallows only the former) surfaces as PlanWriteError
        # naming the path, per this function's own docstring, instead of
        # escaping bare before the try block ever ran.
        os.chmod(tmp_path, _target_file_mode(target))
        _verify_plan_json(tmp_path)
        os.replace(tmp_path, target)
    except Exception as exc:
        raise PlanWriteError(
            f'atomic plan write-back to {path} (resolved to {target}) failed '
            f'({exc!r}); the existing file was left untouched'
        ) from exc
    finally:
        # os.replace consumed the temp on the success path; missing_ok makes
        # this a no-op there and a guaranteed cleanup on every failure path.
        tmp_path.unlink(missing_ok=True)


#: Stable event name for a structured markup fact (INV-2). plan-tools is a bare
#: stdio subprocess with no event bus reachable, so the log IS the channel — and
#: for that to be usable the WHOLE message must be the payload. One line, one
#: fact, ``json.loads``-able; never a value a consumer has to regex out of prose.
_MARKUP_FACT_EVENT = 'plan_markup_repaired'

#: Stable event name for a repair that could not be PERSISTED. Deliberately
#: distinct from the fact event: the repair itself succeeded and was served, so
#: folding the two would make a delivery failure look like a detection outcome.
_MARKUP_WRITE_FAILED_EVENT = 'plan_markup_write_back_failed'

#: Refusals already reported by THIS process, keyed by locator plus a hash of
#: the refusing value. Process-local, and that is exactly the right scope: the
#: orchestrator spawns one plan-tools subprocess per agent invocation, so "once
#: per process" is "once per agent session".
#:
#: WHY A REPAIRED FACT IS NEVER MEMOIZED AND A REFUSAL ALWAYS IS: a repair
#: CONVERGES — the field is fixed on disk, so the next read emits nothing. A
#: refusal never converges. A plan that legitimately QUOTES the sentinels in
#: prose (worktree 2939, a plan about this very leak) refuses on every single
#: call, forever. An architect making ~50 plan-tool calls against it would
#: otherwise get 50xN identical warning lines and 50xN identical facts fed back
#: into its context, each reading like a NEW detection and none of them
#: actionable. The first report is what keeps the residue visible; the 50th is
#: pure amplification of a fact already delivered.
#:
#: Bounded by the number of DISTINCT refusals a process sees (a handful per
#: plan), and keyed on the value's hash so an EDITED field reports again rather
#: than being suppressed by a stale locator. ONE refusal is never memoized at
#: all: a locator that no longer resolves (see :data:`_UNRESOLVED_VALUE`) never
#: gets a key built for it, so it is reported on every call instead of being
#: suppressed under a key that means nothing.
_REPORTED_REFUSALS: set[tuple[str, str | None, int | None, str, int]] = set()

#: Sentinel returned by :func:`_fact_value` when a fact's locators no longer
#: resolve against the plan (a collection shrank, an index moved). DISTINCT
#: from a resolvable locator whose field is legitimately absent — both used to
#: read back as plain ``None``, so "the locator broke" and "the field is
#: missing" were indistinguishable and shared one memo key. Module-private;
#: never leaves this file.
_UNRESOLVED_VALUE = object()


def _fact_value(plan: dict, fact: dict[str, Any]) -> object:
    """The value a *fact* is about, re-read from *plan* through its locators.

    Kept OUT of the fact itself: a fact is attached to the tool response, and
    echoing a whole corrupted prose field back into the caller's context is the
    amplification this memo exists to avoid. The locators are enough to find it.

    Returns :data:`_UNRESOLVED_VALUE` — never plain ``None`` — when the
    locators themselves no longer resolve, so that case can never share a memo
    key with a locator that resolves to a field that is legitimately absent.
    """
    try:
        collection = fact['collection']
        holder = plan if collection is None else plan[collection][fact['index']]
        return holder.get(fact['field'])
    except (KeyError, IndexError, TypeError, AttributeError):
        # An unresolvable locator is signalled distinctly and is NEVER
        # memoized (see _report_markup_facts), so it is reported every time —
        # rather than collapsing to a constant hash key under which a second,
        # unrelated unresolvable refusal would be silently suppressed.
        return _UNRESOLVED_VALUE


def _report_markup_facts(
    plan_path: Path, plan: dict, facts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Log every fact; return the ones worth surfacing to the caller.

    INV-2: the WHOLE log message is a single-line JSON payload under a stable
    event name, so no consumer regex-scrapes a value out of prose.

    A refusal already reported by this process is logged at DEBUG instead of
    WARNING and is dropped from the returned list — see
    :data:`_REPORTED_REFUSALS` for why a refusal is the one outcome that never
    converges on its own. Dropping it makes a repeat call's response converge to
    the clean shape (no ``markup_repairs`` key at all), which is the same
    convergence a successful repair gets for free.
    """
    reportable: list[dict[str, Any]] = []
    for fact in facts:
        payload = json.dumps({
            'event': _MARKUP_FACT_EVENT,
            'path': str(plan_path),
            **fact,
        })
        if fact['outcome'] == 'unrepairable':
            value = _fact_value(plan, fact)
            if value is not _UNRESOLVED_VALUE:
                key = (
                    str(plan_path),
                    fact['collection'],
                    fact['index'],
                    fact['field'],
                    hash(value),
                )
                if key in _REPORTED_REFUSALS:
                    logger.debug(payload)
                    continue
                _REPORTED_REFUSALS.add(key)
            # else: the locator no longer resolves. No key means anything for
            # it, so nothing is built or inserted — fall through and report
            # this call exactly like a first-time refusal, every time.
        logger.warning(payload)
        reportable.append(fact)
    return reportable


def _read_plan_repaired(artifacts: TaskArtifacts) -> tuple[dict, list[dict[str, Any]]]:
    """Read the plan, repairing envelope-markup corruption on the way through.

    THE ONE READ PATH for every plan-tools mutation helper. Returns
    ``(plan, facts)``: the plan as the caller should see it, and the structured
    facts describing every field this read touched or refused (empty on the
    overwhelmingly common clean path).

    LAZY, NOT SWEPT. The corruption sits in LIVE ``.task/plan.json`` files
    belonging to running tasks, so a fleet-wide sweep would have to quiesce
    every lane first. Repairing at read time needs no quiesce at all: the next
    tool call any agent makes fixes its own plan, in place, atomically.

    WRITTEN BACK ONLY WHEN SOMETHING ACTUALLY CHANGED. A plan whose every fact
    is ``'unrepairable'`` is byte-identical to what is already on disk, so
    rewriting it would churn the file — and its mtime, under every watcher —
    to no effect. Same for a wholly clean plan, which never reaches the write
    at all. Reads stay reads.

    TOTAL BY CONSTRUCTION. Missing plan -> ``artifacts.read_plan()``'s empty
    dict, returned untouched with no facts and no write; an unparseable plan
    raises exactly what ``read_plan`` already raises. This surface adds no new
    failure mode to a path every tool depends on.

    A REFUSAL IS REPORTED ONCE PER PROCESS, not once per call. The returned
    facts are the REPORTABLE ones: a refusal this process has already reported
    for the same field and the same value is logged at DEBUG and left out, so a
    plan that can never be repaired stops re-announcing itself on every call.
    See :data:`_REPORTED_REFUSALS`. Repairs are never suppressed — they converge
    on their own, because the next read finds the field already fixed.

    AND A WRITE-BACK FAILURE NEVER WITHHOLDS THE REPAIR. It is logged loudly
    and the in-memory repaired plan is still returned, because the alternative
    — refusing to serve a plan because its repair could not be persisted —
    hands the caller the CORRUPTED text back, which is strictly worse than
    today's behaviour and is exactly the failure this surface exists to end.
    """
    plan = artifacts.read_plan()
    if not plan:
        return plan, []

    repaired, facts = _repair_plan_fields(plan)
    if not facts:
        return repaired, facts

    plan_path = artifacts.root / 'plan.json'
    reportable = _report_markup_facts(plan_path, repaired, facts)

    if not any(fact['outcome'] == 'repaired' for fact in facts):
        # Every fact was a refusal: nothing in the document changed, so there
        # is nothing to persist. This is what keeps a plan that legitimately
        # QUOTES the sentinels in prose (worktree 2939) byte-stable forever.
        # Decided over ALL facts, never the reported subset — a suppressed
        # repeat must not be able to flip a write decision.
        return repaired, reportable

    try:
        _atomic_write_plan(plan_path, repaired)
    except Exception as exc:
        # Broad on purpose: the caller must receive the repaired plan whatever
        # went wrong on the way to disk. mkstemp and the json.dumps in
        # _atomic_write_plan both raise outside its own PlanWriteError wrapper
        # (OSError / TypeError / ValueError), so narrowing here would turn a
        # persistence failure into a failed tool call.
        logger.warning(
            json.dumps({
                'event': _MARKUP_WRITE_FAILED_EVENT,
                'path': str(plan_path),
                'error': repr(exc),
                'served_from_memory': True,
                'repaired_fields': [
                    fact['field'] for fact in facts if fact['outcome'] == 'repaired'
                ],
            })
        )
    return repaired, reportable


def _with_markup_repairs(
    response: dict[str, Any], facts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach ``markup_repairs`` to *response* — only when there is one to make.

    ABSENT on the clean path, not empty and not null. The overwhelming majority
    of calls repair nothing, and those responses must stay byte-identical to
    what every existing caller and every existing assertion already sees; an
    always-present key would be a silent contract change dressed as a
    diagnostic.

    Purely additive when it does fire, so a documented envelope keeps its
    documented keys — including :func:`_mark_step_committed`'s Contract A1
    ``{ok, step_id, status}``, whose divergence from its siblings is
    contractual and must not be "aligned" by anything passing through here.
    """
    if facts:
        response['markup_repairs'] = facts
    return response


# ---------------------------------------------------------------------------
# Standalone implementation functions (testable without MCP transport)
# ---------------------------------------------------------------------------


def _create_plan(
    artifacts: TaskArtifacts,
    task_id: str,
    title: str,
    analysis: str,
    files: list[str] | str | None,
) -> dict[str, Any]:
    if files is None:
        return {
            'status': 'error',
            'message': (
                'files argument was missing from this create_plan call — a known '
                'agent-harness mis-serialization defect that occurs when a large '
                'or complex analysis string accompanies the files array in the '
                'same call. Workaround: call create_plan again with a SHORT '
                'placeholder analysis plus the full files array, then set the '
                'full analysis via update_plan_metadata(analysis=...).'
            ),
        }
    # DELIBERATELY NOT hooked to _read_plan_repaired: this overwrites plan.json
    # wholesale, so there is no existing document to repair on the way in.
    # Guarding these INBOUND arguments against envelope markup is the write-time
    # middleware's job, not this read-time surface's — don't "fix" the omission.
    files = _coerce_files(files)
    plan = {
        'task_id': task_id,
        'title': title,
        'analysis': analysis,
        'files': files,
        'prerequisites': [],
        'steps': [],
        'design_decisions': [],
        'reuse': [],
    }
    artifacts.write_plan(plan)
    return {'status': 'ok', 'task_id': task_id}


def _add_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
    step_type: str,
    description: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    # Reject duplicate step IDs
    existing_ids = {
        s.get('id')
        for col in ('prerequisites', 'steps')
        for s in plan.get(col, [])
        if isinstance(s, dict)
    }
    if step_id in existing_ids:
        return _with_markup_repairs(
            {'status': 'error', 'message': f'Step {step_id!r} already exists in plan.'},
            markup_facts,
        )

    plan.setdefault('steps', []).append({
        'id': step_id,
        'type': step_type,
        'description': description,
        'status': 'pending',
        'commit': None,
    })
    artifacts.write_plan(plan)
    return _with_markup_repairs(
        {'status': 'ok', 'step_id': step_id, 'total_steps': len(plan['steps'])},
        markup_facts,
    )


def _add_prerequisite(
    artifacts: TaskArtifacts,
    prereq_id: str,
    description: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    existing_ids = {
        s.get('id')
        for col in ('prerequisites', 'steps')
        for s in plan.get(col, [])
        if isinstance(s, dict)
    }
    if prereq_id in existing_ids:
        return _with_markup_repairs(
            {'status': 'error', 'message': f'Prerequisite {prereq_id!r} already exists.'},
            markup_facts,
        )

    plan.setdefault('prerequisites', []).append({
        'id': prereq_id,
        'description': description,
        'status': 'pending',
        'commit': None,
        'tests': [],
    })
    artifacts.write_plan(plan)
    return _with_markup_repairs({'status': 'ok', 'prereq_id': prereq_id}, markup_facts)


def _add_design_decision(
    artifacts: TaskArtifacts,
    decision: str,
    rationale: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    plan.setdefault('design_decisions', []).append({
        'decision': decision,
        'rationale': rationale,
    })
    artifacts.write_plan(plan)
    return _with_markup_repairs(
        {'status': 'ok', 'total_decisions': len(plan['design_decisions'])},
        markup_facts,
    )


def _add_reuse_item(
    artifacts: TaskArtifacts,
    what: str,
    where: str,
    how: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    plan.setdefault('reuse', []).append({
        'what': what,
        'where': where,
        'how': how,
    })
    artifacts.write_plan(plan)
    return _with_markup_repairs(
        {'status': 'ok', 'total_reuse': len(plan['reuse'])}, markup_facts
    )


# The repairable-field table, bound here because ``_params_of`` reads the five
# writer signatures above off the live functions. Declared and documented at
# its annotation further up; nothing between that point and here reads it.
_REPAIRABLE_PLAN_FIELDS = _build_repairable_plan_fields()


def _mark_step_done(
    artifacts: TaskArtifacts,
    step_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    for collection in ('prerequisites', 'steps'):
        for item in plan.get(collection, []):
            if isinstance(item, dict) and item.get('id') == step_id:
                # The repair write-back has already landed, so the re-read
                # inside update_step_status picks up the repaired document.
                artifacts.update_step_status(step_id, 'done', commit=commit_sha)
                return _with_markup_repairs(
                    {
                        'status': 'ok',
                        'step_id': step_id,
                        'new_status': 'done',
                        'commit': commit_sha,
                    },
                    markup_facts,
                )
    return _with_markup_repairs(
        {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'},
        markup_facts,
    )


# ---------------------------------------------------------------------------
# Revalidation helpers
# ---------------------------------------------------------------------------


def _update_plan_metadata(
    artifacts: TaskArtifacts,
    files: list[str] | str | None = None,
    analysis: str | None = None,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    if files is not None:
        plan['files'] = _coerce_files(files)
    if analysis is not None:
        plan['analysis'] = analysis
    artifacts.write_plan(plan)
    return _with_markup_repairs(
        {
            'status': 'ok',
            'files': len(plan.get('files', [])),
        },
        markup_facts,
    )


def _remove_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    for collection in ('prerequisites', 'steps'):
        items = plan.get(collection, [])
        for i, item in enumerate(items):
            if isinstance(item, dict) and item.get('id') == step_id:
                if item.get('status') == 'done':
                    return _with_markup_repairs(
                        {
                            'status': 'error',
                            'message': (
                                f'Step {step_id!r} has status "done" and cannot be removed.'
                            ),
                        },
                        markup_facts,
                    )
                items.pop(i)
                artifacts.write_plan(plan)
                return _with_markup_repairs(
                    {'status': 'ok', 'removed': step_id, 'collection': collection},
                    markup_facts,
                )

    return _with_markup_repairs(
        {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'},
        markup_facts,
    )


def _replace_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
    step_type: str,
    description: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    for collection in ('prerequisites', 'steps'):
        for item in plan.get(collection, []):
            if isinstance(item, dict) and item.get('id') == step_id:
                if item.get('status') == 'done':
                    return _with_markup_repairs(
                        {
                            'status': 'error',
                            'message': (
                                f'Step {step_id!r} has status "done" and cannot be replaced.'
                            ),
                        },
                        markup_facts,
                    )
                item['type'] = step_type
                item['description'] = description
                artifacts.write_plan(plan)
                return _with_markup_repairs(
                    {'status': 'ok', 'replaced': step_id}, markup_facts
                )

    return _with_markup_repairs(
        {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'},
        markup_facts,
    )


def _confirm_plan(
    artifacts: TaskArtifacts,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}
    if not plan.get('steps'):
        return _with_markup_repairs(
            {'status': 'error', 'message': 'Plan has no steps — cannot confirm.'},
            markup_facts,
        )
    if not plan.get('files'):
        return _with_markup_repairs(
            {
                'status': 'error',
                'message': (
                    'Plan has no files — cannot confirm. Call create_plan (or '
                    'update_plan_metadata) with a non-empty files list first.'
                ),
            },
            markup_facts,
        )

    now = datetime.now(UTC).isoformat()
    # ``_finalized_at`` is the durable completeness marker.  Its PRESENCE means
    # the architect declared the plan complete; the workflow requires it before
    # a plan may advance to execution (workflow._plan), and salvages a plan that
    # carries it even when the CLI run failed on a budget/turn cap.  Its ABSENCE
    # on a plan that already has steps means a prior session was interrupted
    # mid-build — the workflow routes that partial to a completion pass instead
    # of discarding it.  ``_revalidated_at`` is retained for revalidation-age
    # checks (see workflow._can_skip_revalidation).
    plan['_finalized_at'] = now
    plan['_revalidated_at'] = now
    artifacts.write_plan(plan)
    return _with_markup_repairs(
        {
            'status': 'ok',
            'finalized': True,
            'steps': len(plan['steps']),
            'files': len(plan.get('files', [])),
        },
        markup_facts,
    )


# ---------------------------------------------------------------------------
# Missing-dependency reporting (architect escape hatch)
# ---------------------------------------------------------------------------


def _report_blocking_dependency(
    artifacts: TaskArtifacts,
    depends_on_task_id: str,
    reason: str,
    main_sha: str,
) -> dict[str, Any]:
    """Write the blocking-dependency artifact for the workflow to act on.

    The architect calls this when it has determined the task cannot be
    planned because it depends on work that has not yet landed on main.
    No Taskmaster mutation happens here — the workflow reads the artifact
    after the architect returns and registers the dependency
    deterministically.
    """
    artifacts.write_blocking_dependency(
        depends_on_task_id=depends_on_task_id,
        reason=reason,
        main_sha_at_report=main_sha,
    )
    return {
        'status': 'ok',
        'depends_on_task_id': depends_on_task_id,
    }


def _report_task_already_done(
    artifacts: TaskArtifacts,
    commit: str,
    evidence: str,
) -> dict[str, Any]:
    """Write the already-done artifact for the workflow to act on.

    The architect calls this when it has determined that the task's work
    is already on main at *commit*.  The workflow validates the claim
    (commit reachable from main) and sets task status to ``done``.
    """
    artifacts.write_already_done(commit=commit, evidence=evidence)
    return {'status': 'ok', 'commit': commit}


def _report_ready_to_merge(
    artifacts: TaskArtifacts,
    commit: str,
    evidence: str,
) -> dict[str, Any]:
    """Write the ready-to-merge artifact for the workflow to act on.

    The architect calls this when the task's work is already complete on
    *this branch* at *commit* and only the physical merge to main is
    missing.  The workflow re-validates the merge-landing-desync predicate
    first-hand and, if it holds, enqueues a deterministic merge request.
    """
    artifacts.write_ready_to_merge(commit=commit, evidence=evidence)
    return {'status': 'ok', 'commit': commit}


def _report_unactionable_task(
    artifacts: TaskArtifacts,
    reason: str,
    evidence: str,
) -> dict[str, Any]:
    """Write the unactionable-task artifact for the workflow to act on.

    The architect calls this when it has determined the task spec itself
    is unworkable — premises are contradictory, the goal is incompatible
    with current main, or no valid plan is possible as written.  The
    workflow short-circuits to a level-1 escalation, bypassing the steward.
    """
    artifacts.write_unactionable_task(reason=reason, evidence=evidence)
    return {'status': 'ok', 'reason': reason}


def _report_false_premise(
    artifacts: TaskArtifacts,
    classification: str,
    premise: str,
    evidence: str,
    proposed_resolution: str,
) -> dict[str, Any]:
    """Write the false-premise artifact for the workflow to act on.

    The architect calls this when it has determined that a RED-test premise
    is false, unreachable, or misattributed — before writing the doomed
    assertion.  The workflow short-circuits to a level-1 design_concern
    escalation, bypassing the steward.
    """
    artifacts.write_false_premise(
        classification=classification,
        premise=premise,
        evidence=evidence,
        proposed_resolution=proposed_resolution,
    )
    return {'status': 'ok', 'classification': classification}


def _resolve_main_sha(worktree: Path) -> str:
    """Return the current ``main`` HEAD SHA visible from *worktree*.

    Falls back to an empty string on git failure — the workflow side
    treats an empty ``main_sha_at_report`` as "advance check skipped".
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'main'],
            cwd=str(worktree),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            'Failed to resolve main SHA for blocking_dependency report: %s', exc
        )
        return ''


def _sha_exists_on_branch(worktree: Path, sha: str) -> bool:
    """Return True iff *sha* is reachable from HEAD in the repo at *worktree*.

    A real ``git merge-base --is-ancestor <sha> HEAD`` reachability check
    (returncode 0 == *sha* is an ancestor of — or equal to — HEAD), run in the
    worktree cwd. This is the machine-checked corroborate-before-acting guard
    for ``mark_step_committed`` (INV-1/INV-3): the architect must be UNABLE to
    pre-satisfy a plan step against a commit that does not resolve on the
    current branch.

    Note we check reachability (``--is-ancestor``), NOT mere object existence
    (``git cat-file -e``): in this repo's shared/multi-worktree object store a
    commit from another task's branch — or a dangling/unreachable object —
    exists in the object DB yet is NOT on this branch. ``--is-ancestor`` returns
    non-zero (exit 1 for a real-but-unreachable commit, 128 for a bad object)
    for both, so the guard matches its advertised on-branch contract. It
    deliberately does NOT prove the step's semantics — VERIFY remains the gate.
    Mirrors :func:`_resolve_main_sha`'s subprocess shape (cwd=worktree,
    timeout=10, OSError/SubprocessError -> safe ``False``).
    """
    try:
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', sha, 'HEAD'],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            'Failed git merge-base --is-ancestor for sha %r: %s', sha, exc
        )
        return False


def _mark_step_committed(
    artifacts: TaskArtifacts,
    step_id: str,
    sha: str,
) -> dict[str, Any]:
    """Pre-satisfy a plan step/prerequisite at AUTHORING time.

    Guards on a real on-branch reachability check for *sha* (corroborate-before-
    acting) before delegating the status/commit/description-tag mutation to
    :meth:`TaskArtifacts.mark_step_committed`. Returns the Contract A1 shapes:
    ``{'ok': True, 'step_id', 'status': 'done'}`` on success (``status`` is the
    step's NEW status), and ``{'ok': False, 'status': 'error', 'message': ...}``
    on a non-resolving sha or an unknown ``step_id``.

    NOTE: this envelope (``ok`` + ``status`` = the step's new status) is the
    Contract A1 shape and DELIBERATELY differs from the sibling
    :func:`_mark_step_done` (``status: 'ok'|'error'`` + ``new_status``). A1
    fixes ``{ok, step_id, status}`` for the re-invoked architect's all-committed
    path; don't "align" the two — the divergence is contractual, not an
    oversight. On error we keep the repo-wide ``{'status':'error','message'}``
    convention (so ``status=='error'``/substring assertions hold) and add
    ``ok:False`` so callers can branch uniformly on ``ok``.
    """
    if not _sha_exists_on_branch(artifacts.worktree, sha):
        return {
            'ok': False,
            'status': 'error',
            'message': (
                f'sha {sha!r} does not resolve on the current branch '
                '(not reachable from HEAD — git merge-base --is-ancestor '
                'failed) — cannot pre-satisfy a step against a commit that '
                'is not on this branch'
            ),
        }
    # This is the one hooked entry point that does NOT read the plan itself —
    # ``artifacts.mark_step_committed`` reaches disk on its own. Repairing here
    # first means the write-back has already landed when the delegate re-reads,
    # so its mutation applies to the repaired document rather than racing it.
    _, markup_facts = _read_plan_repaired(artifacts)
    if not artifacts.mark_step_committed(step_id, sha):
        return _with_markup_repairs(
            {
                'ok': False,
                'status': 'error',
                'message': f'Step {step_id!r} not found in plan.',
            },
            markup_facts,
        )
    return _with_markup_repairs(
        {'ok': True, 'step_id': step_id, 'status': 'done'}, markup_facts
    )


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------


def create_server(artifacts: TaskArtifacts) -> FastMCP:
    """Create the plan-tools MCP server with all tools registered."""
    mcp = FastMCP('plan-tools')

    @mcp.tool()
    def create_plan(
        task_id: str,
        title: str,
        analysis: str,
        files: list[str] | str | None = None,
    ) -> dict[str, Any]:
        """Initialize a new implementation plan with metadata.

        Call this first, before adding steps or prerequisites.
        Creates .task/plan.json with the provided metadata and empty
        step/prerequisite arrays.

        Args:
            task_id: The task identifier (e.g. "df_task_13").
            title: Human-readable task title.
            analysis: Your analysis of the task, existing code, and approach.
            files: List of files (or directory paths) this task will create or modify.
                Drives concurrency locks and the phantom-done gate. Also
                tolerates a stringified JSON array or a comma/newline-delimited
                string, recovering it into a list — a known agent-harness
                mis-serialization when paired with a large/complex analysis.
        """
        return _create_plan(artifacts, task_id, title, analysis, files)

    @mcp.tool()
    def add_plan_step(
        step_id: str,
        step_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a TDD step to the plan. Steps execute in the order added.

        Args:
            step_id: Unique step ID (e.g. "step-1", "step-2").
            step_type: Either "test" or "impl".
            description: What this step does.
        """
        return _add_plan_step(artifacts, step_id, step_type, description)

    @mcp.tool()
    def add_prerequisite(
        prereq_id: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a prerequisite to the plan. Prerequisites run before TDD steps.

        Use for setup work like config files, fixtures, etc.

        Args:
            prereq_id: Unique prerequisite ID (e.g. "pre-1").
            description: What this prerequisite sets up.
        """
        return _add_prerequisite(artifacts, prereq_id, description)

    @mcp.tool()
    def add_design_decision(
        decision: str,
        rationale: str,
    ) -> dict[str, Any]:
        """Record a design decision and its rationale in the plan.

        Args:
            decision: The decision that was made.
            rationale: Why this decision was made over alternatives.
        """
        return _add_design_decision(artifacts, decision, rationale)

    @mcp.tool()
    def add_reuse_item(
        what: str,
        where: str,
        how: str,
    ) -> dict[str, Any]:
        """Record an existing code pattern or utility being reused.

        Args:
            what: What is being reused (e.g. "MergeResult dataclass pattern").
            where: Where it exists (e.g. "orchestrator/src/orchestrator/git_ops.py:14-18").
            how: How it will be reused in this task.
        """
        return _add_reuse_item(artifacts, what, where, how)

    @mcp.tool()
    def mark_step_done(
        step_id: str,
        commit_sha: str,
    ) -> dict[str, Any]:
        """Mark a plan step or prerequisite as done with its commit SHA.

        Use this instead of directly editing .task/plan.json.
        Only the status and commit fields are updated; all other plan
        structure (including provenance metadata) is preserved.

        Args:
            step_id: The step or prerequisite ID (e.g. "step-1", "pre-1").
            commit_sha: The git commit SHA for this step's changes.
        """
        return _mark_step_done(artifacts, step_id, commit_sha)

    @mcp.tool()
    def mark_step_committed(
        step_id: str,
        sha: str,
    ) -> dict[str, Any]:
        """Pre-satisfy a plan step/prerequisite at AUTHORING time.

        Marks the step ``done`` (and prepends a ``[COMMITTED <sha[:12]>]`` tag
        to its description) WITHOUT an implementer turn. Use this ONLY when the
        branch you are (re-)planning ALREADY carries the complete, committed,
        green implementation of this step: a re-invoked architect can then emit
        an all-satisfied plan so the EXECUTE loop becomes a total no-op and the
        branch flows PLAN → VERIFY → REVIEW → MERGE with zero implementer turns.

        The *sha* MUST resolve on the current branch — a real ``git merge-base
        --is-ancestor <sha> HEAD`` reachability guard refuses a commit that is
        not on this branch (corroborate-before-acting; mere existence in a
        shared object store is not enough). This does NOT prove the step's
        semantics: VERIFY remains the gate, and a falsely pre-satisfied step
        surfaces as a VERIFY failure, never a silent skip. Idempotent per
        ``(step_id, sha)``.

        Args:
            step_id: The step or prerequisite ID (e.g. "step-1", "pre-1").
            sha: The git commit SHA that already carries this step's work.
                Must resolve on the current branch.
        """
        return _mark_step_committed(artifacts, step_id, sha)

    # --- Revalidation tools ---

    @mcp.tool()
    def update_plan_metadata(
        files: list[str] | str | None = None,
        analysis: str | None = None,
    ) -> dict[str, Any]:
        """Update the plan's top-level metadata without touching steps or prerequisites.

        Use during plan revalidation to update the file list or analysis
        after reviewing changes on main. All parameters are optional — only
        non-None values are updated.

        Args:
            files: Updated list of files (or directory paths) this task will
                create or modify. Drives concurrency locks and the phantom-done gate.
                Tolerates a stringified array (e.g. a JSON-encoded list or a
                comma/newline-delimited string) and recovers it into a clean
                list; leave unset/None to keep the existing files unchanged.
            analysis: Updated analysis text.
        """
        return _update_plan_metadata(artifacts, files, analysis)

    @mcp.tool()
    def remove_plan_step(
        step_id: str,
    ) -> dict[str, Any]:
        """Remove a pending step or prerequisite from the plan by ID.

        Use during plan revalidation when a step is no longer needed
        due to changes on main. Cannot remove steps with status "done".

        Args:
            step_id: The step or prerequisite ID to remove (e.g. "step-3").
        """
        return _remove_plan_step(artifacts, step_id)

    @mcp.tool()
    def replace_plan_step(
        step_id: str,
        step_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Replace a pending step's type and description in-place.

        Preserves the step's position, status, and commit fields.
        Use during plan revalidation when a step needs revision due
        to changes on main. Cannot replace steps with status "done".

        Args:
            step_id: The step ID to replace (e.g. "step-2").
            step_type: New step type — either "test" or "impl".
            description: New description of what this step does.
        """
        return _replace_plan_step(artifacts, step_id, step_type, description)

    @mcp.tool()
    def confirm_plan() -> dict[str, Any]:
        """Mark the plan COMPLETE. Call this as your final plan-tools action.

        This is the terminal "I am done building (or revising) this plan"
        signal, required in BOTH flows:
          - Initial planning: call it once every step and prerequisite has
            been added, immediately after the last ``add_plan_step``.
          - Revalidation: call it when you have reviewed the changes on main
            and the plan needs no further modification.

        Stamps a durable completeness marker on the plan. A plan WITHOUT this
        marker is treated as incomplete: it will not advance to execution, and
        a later session may resume and finish it. Requires a non-empty steps
        and files list.
        """
        return _confirm_plan(artifacts)

    @mcp.tool()
    def report_blocking_dependency(
        depends_on_task_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Report that this task cannot be planned because it depends on
        another task whose work has not yet landed on main.

        Use this INSTEAD of writing a plan when you discover, during the
        verify-premises phase, that a referenced file/symbol is missing
        because the sibling task that would create it has not yet merged.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will read the report,
        register the Taskmaster dependency, and re-queue this task to
        run again after the named task lands.

        Args:
            depends_on_task_id: The task id this task is blocked on
                (the one whose work is missing). Use the exact task id
                from the briefing or the task tree (e.g. "1042").
            reason: One-line explanation of what's missing — file/symbol
                names and where this task expected to find them.
        """
        # ``artifacts.root`` is a SIBLING of the worktree once a meta_root is
        # supplied (`.task-meta/<name>/`), so `artifacts.root.parent` is the
        # `.task-meta` base, NOT the worktree. `artifacts.worktree` is always
        # correct, legacy or relocated (see TaskArtifacts.__init__).
        worktree = artifacts.worktree
        main_sha = _resolve_main_sha(worktree)
        return _report_blocking_dependency(
            artifacts, depends_on_task_id, reason, main_sha,
        )

    @mcp.tool()
    def report_task_already_done(
        commit: str,
        evidence: str,
    ) -> dict[str, Any]:
        """Report that this task's work is already on main at ``commit``.

        Use this INSTEAD of writing a plan when you discover, during the
        verify-premises phase, that the work this task asks for is already
        present on main — typically because a sibling task direct-merged
        the change, or a prior orchestrator run landed it under a different
        task id.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will validate that ``commit``
        is reachable from main, then set this task to ``done`` with
        provenance pointing at ``commit``.

        Args:
            commit: The git SHA on main that contains this task's work.
                Must be reachable from main; the orchestrator verifies this.
            evidence: Brief description of what you found — file/symbol
                names that match this task's expected output, or why the
                commit appears to subsume this task's scope.
        """
        return _report_task_already_done(artifacts, commit, evidence)

    @mcp.tool()
    def report_ready_to_merge(
        commit: str,
        evidence: str,
    ) -> dict[str, Any]:
        """Report that this task's work is COMPLETE ON THIS BRANCH at
        ``commit`` and only the physical merge to main is missing.

        Use this INSTEAD of writing a plan when you discover, during the
        verify-premises phase, that this task's branch is a *merge-landing
        desync*: every one of the following holds and the ONLY thing left
        to do is land the branch on main.

          - The branch is a clean fast-forward of main — main is an
            ancestor of the branch tip, and the branch is NOT already
            contained in main.
          - Verify already PASSED on this exact branch tip.
          - Review already returned PASS (or suggestions-only) on this
            exact tree.

        Do NOT use this when work is genuinely missing, when verify or
        review never passed on this tip, or when the branch has diverged
        from main (needs a rebase/merge to resolve) — write a plan
        instead.  If the work is already ON MAIN, use
        ``report_task_already_done``.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator re-validates the desync
        predicate FIRST-HAND (it does not trust this report) and, if it
        holds, enqueues a deterministic merge request — no human
        escalation is filed. This tool never lands main itself; the merge
        worker's scoped re-verify remains the sole gate.

        Args:
            commit: The git SHA of THIS BRANCH's tip that holds the
                completed work. The orchestrator resolves the branch tip
                itself and rejects the report if it does not match.
            evidence: Brief description of what you checked — how you
                established the clean fast-forward, and where the passing
                verify and review results came from.
        """
        return _report_ready_to_merge(artifacts, commit, evidence)

    @mcp.tool()
    def report_unactionable_task(
        reason: str,
        evidence: str,
    ) -> dict[str, Any]:
        """Report that this task spec is unworkable as written.

        Use this INSTEAD of writing a plan when you have determined the
        task itself cannot proceed — premises are contradictory, the goal
        is incompatible with current main, or no valid plan exists as
        written. Use this only when a human needs to rewrite or cancel
        the task; for risks/concerns where a plan is still possible, use
        ``escalate_blocker`` instead.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will mark this task BLOCKED
        and submit a level-1 escalation directly — the steward is bypassed
        because no automated handler can fix a broken spec.

        Args:
            reason: One-line summary of why the task is unworkable.
            evidence: Detail — the specific contradiction, false premise,
                or impossibility, with file:line references where possible.
        """
        return _report_unactionable_task(artifacts, reason, evidence)

    @mcp.tool()
    def report_false_premise(
        classification: str,
        premise: str,
        evidence: str,
        proposed_resolution: str,
    ) -> dict[str, Any]:
        """Report that a RED-test premise is false before writing the assertion.

        Use this INSTEAD of writing the test when you have determined that
        a RED test's substantive premise is false, unreachable, or
        misattributed.  The three premise classes are:

        - ``numeric_bound``: The test asserts a numeric threshold or tolerance
          (e.g. "within X%", "<= eps", ">= N dB", "to M digits") with no
          achievability basis.
        - ``exactness``: The test asserts closed-form exactness or reproduction
          (e.g. "exact within 1e-12", "reproduces P(t) exactly") but the
          stated configuration does not satisfy the identity that would make
          it true.
        - ``dependency_capability``: The test asserts that this task produces
          or evaluates a capability that is actually delivered by a task that
          DEPENDS ON this one, not a prerequisite.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will mark this task BLOCKED
        and submit a level-1 design_concern escalation directly — the steward
        is bypassed because re-speccing a test premise requires human/curator
        judgment.

        Args:
            classification: One of ``numeric_bound``, ``exactness``, or
                ``dependency_capability``.
            premise: The exact assertion or claim that is false.
            evidence: The mathematical identity, method-error estimate,
                dependency-graph trace, or other proof that the premise fails.
            proposed_resolution: Suggested fix — move signal to producing task /
                weaken bound + file follow-up / change asserted configuration.
        """
        return _report_false_premise(
            artifacts, classification, premise, evidence, proposed_resolution
        )

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _artifacts_from_args(argv: list[str] | None = None) -> TaskArtifacts:
    """Parse ``--worktree``/``--meta-root`` and return the corresponding
    ``TaskArtifacts``.

    Mirrors workflow.py's orchestrator-side construction
    (``TaskArtifacts(worktree, _meta_root_for_worktree(worktree))``) so both
    sides resolve the IDENTICAL meta_root for a given worktree
    (worktree-lane-lifecycle PRD task ε1). When ``--meta-root`` is omitted,
    behaviour is byte-identical to before this argument existed: the root
    dir checked for existence is the legacy ``<worktree>/.task``.
    """
    parser = argparse.ArgumentParser(description='Plan-tools MCP server (stdio)')
    parser.add_argument(
        '--worktree', type=Path, required=True,
        help='Path to the git worktree containing .task/',
    )
    parser.add_argument(
        '--meta-root', type=Path, default=None,
        help=(
            'Optional `.task-meta` root (sibling of the worktree). When '
            'omitted, plan.json et al. are read/written at the legacy '
            '<worktree>/.task location.'
        ),
    )
    args = parser.parse_args(argv)

    worktree = args.worktree.resolve()
    meta_root = args.meta_root.resolve() if args.meta_root is not None else None
    root_to_check = meta_root if meta_root is not None else worktree / '.task'
    if not root_to_check.is_dir():
        print(f'Error: {root_to_check} does not exist', file=sys.stderr)
        sys.exit(1)

    return TaskArtifacts(worktree, meta_root)


def main() -> None:
    """Parse --worktree/--meta-root and run the stdio MCP server."""
    artifacts = _artifacts_from_args()
    server = create_server(artifacts)
    # show_banner=False suppresses FastMCP's decorative startup banner and its
    # synchronous PyPI update-check (check_for_newer_version -> httpx.get). That
    # check runs on the stdio startup path BEFORE the MCP initialize handshake;
    # eliminating it removes a network call that could delay startup past the
    # CLI's MCP_TIMEOUT and get this server silently dropped (task 2942).
    server.run(transport='stdio', show_banner=False)


if __name__ == '__main__':
    main()
