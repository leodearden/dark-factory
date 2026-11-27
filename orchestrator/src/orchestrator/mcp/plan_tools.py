"""Plan-tools MCP server -- stdio tool server for safe plan.json operations.

Spawned per-agent invocation by the orchestrator.  The architect agent
builds plans via ``create_plan`` / ``add_plan_step`` / etc., and the
implementer marks steps done via ``mark_step_done`` and records where a
step's text proved wrong via ``record_step_deviation``.  On revalidation
(blast-radius requeue), the architect uses ``update_plan_metadata``,
``remove_plan_step``, ``replace_plan_step``, and ``confirm_plan`` to
update an existing plan without recreating it from scratch, and
``record_descoped_step`` when it genuinely drops a done step.  All writes
go through ``TaskArtifacts`` methods, preserving ``_session_id`` and
enforcing correct schema.

plan.json is also repaired lazily on read: every entry point that opens an
existing plan goes through :func:`_read_plan_repaired`, whose mechanism is
``orchestrator/mcp/plan_markup_repair.py``.  That read-time repair COMPOSES
with the write-time ``MarkupGuardMiddleware`` ``create_server`` registers;
the ruling and its two consequences are D8 of
``plans/toolcall-markup-containment-prd.md``.

Heuristic 14's measurement of this file and its split: task 7082's commit.

Usage (stdio transport, spawned by orchestrator):
    # Direct-interpreter no-uv hot path (production, task 1776):
    <sys.executable> -m orchestrator.mcp.plan_tools --worktree /path/to/worktree \
        --grants-enforced [--grant NAME ...]

    # uv fallback with fast-start flags (task 1775, backward-compat):
    uv run --project <orchestrator-dir> --no-sync --frozen \
        python -m orchestrator.mcp.plan_tools --worktree /path/to/worktree \
        --grants-enforced [--grant NAME ...]
"""

from __future__ import annotations

import argparse
import ast
import functools
import inspect
import json
import logging
import re
import sys
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Concatenate, ParamSpec, TypeVar

from fastmcp import FastMCP
from shared.markup_override_param import accepts_markup_override
from shared.mcp_markup_middleware import (
    HostAttribution,
    MarkupGuardMiddleware,
    RepairPolicy,
)
from shared.mcp_missing_arguments import MissingArgumentMiddleware
from shared.plan_deviation import DeviationRecord, UnknownDeviationKind

from orchestrator import plan_descope, quarantine_provenance
from orchestrator.artifacts import Plan, PlanConfirmation, PlanState, TaskArtifacts
from orchestrator.mcp import (
    branch_probes,
    foreign_task_meta,
    markup_journal,
    markup_sink,
    plan_markup_repair,
    plan_markup_stamp,
)
from orchestrator.mcp.branch_probes import Reachability
from orchestrator.mcp.plan_markup_repair import PlanField
from orchestrator.mcp.stored_markup_journal import StoredMarkupJournalMiddleware
from orchestrator.scope_drift import InvalidScopeRequest, ScopeRequest, pending_scope_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# files-arg recovery helper (task 2428)
# ---------------------------------------------------------------------------


# Bracket/quote characters that, if still present in a delimiter-split
# fallback result, indicate the split did not fully recover clean file
# paths (i.e. the input was corrupted in a shape we don't parse).
_RESIDUAL_CORRUPTION_CHARS = frozenset("[]{}'\"")


def _coerce_files(files: object) -> list[str]:
    """Coerce a possibly mis-serialized ``files`` value into a clean list.

    The agent harness intermittently mis-serializes the ``create_plan``
    (and ``update_plan_metadata``) call when a large/complex ``analysis``
    string accompanies the ``files`` array in the same call, corrupting
    ``files`` into a stringified JSON array, a bare JSON string, a
    Python-repr-style single-quoted list, or a comma/newline-delimited
    plain string. This recovers all of those shapes back into a clean
    ``list[str]``; ``None`` maps to ``[]``. Shapes that cannot be
    confidently recovered are logged (rather than silently accepted) so
    the corruption stays visible to operators.

    A ``files`` read back out of a stored plan has passed no pydantic
    boundary, so it may be any JSON type: one that is neither list, str nor
    ``None`` maps to ``[]``, logged, rather than raising.
    """
    if files is None:
        return []
    if isinstance(files, list):
        return [str(f).strip() for f in files if str(f).strip()]
    if not isinstance(files, str):
        logger.warning(
            "_coerce_files: files value has unrecognized type %s (expected "
            "list, str or None); treating it as no files: %r",
            type(files).__name__,
            files,
        )
        return []

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
# ``path`` is deliberately EXCLUDED, for the same reason ``files`` and
# ``task_id`` are: it is not prose. Letting an absorbed tail land there would
# re-point a recorded drop at a different file, turning an honest-drop record
# into a false one.
_DROP_PLAN_FILE_TARGETS: Mapping[str, str] = MappingProxyType({'reason': 'reason'})

#: The OTHER tools that write a given field, per row of the table below. Each
#: entry is machine-checked to name a real plan-tools entry point that is
#: OBSERVED TO ACTUALLY WRITE the field
#: (``TestRepairableFieldTable::test_every_alternate_writer_really_writes_that_field``),
#: so an alternate cannot drift into a prose label. Checked BEHAVIOURALLY, not
#: by signature: a writer need not take the field as a parameter to write it —
#: ``mark_step_committed`` rewrites ``description`` by re-reading and
#: rewriting the plan, never taking a ``description`` argument at all, so a
#: signature-based check would have rejected it as an alternate despite it
#: being a real one. It prepends a ``[COMMITTED <sha12>]`` tag, so it cannot
#: introduce markup, but a triager of a corrupt tagged description must
#: still account for it.
#:
#: That check is SOUNDNESS only — it cannot catch a writer left OUT of this
#: mapping. COMPLETENESS is the converse check,
#: ``TestRepairableFieldTable::test_no_plan_writing_tool_is_an_undeclared_alternate``:
#: it sweeps every row here against a candidate set DERIVED from this module's
#: own plan-mutating surface INTERSECTED WITH the tool surface
#: ``create_server`` actually registers (``_plan_writing_tool_names()`` in the
#: test file, not a hand-maintained list). The intersection is load-bearing:
#: an INTERNAL helper may legitimately rewrite plan.json without being a
#: probeable MCP tool — ``_read_plan_repaired`` does, on its read-repair
#: write-back path — so "writes plan.json" alone was never sufficient to make
#: something a sweep candidate, and adding such a helper does NOT drag it into
#: the behavioural probe. A new plan-writing TOOL, by contrast, is swept
#: automatically and that test fails until its impact on ``also_written_by``
#: is audited. Hand-adding a name here is therefore never required to be
#: *seen* by the sweep — only to make it pass once seen.
#:
#: "Automatically" holds for a tool that follows this module's own conventions
#: — a module-level ``_<tool_name>`` impl that persists through a
#: ``TaskArtifacts`` method — and BOTH conventions are themselves pinned, so
#: a tool that follows neither still cannot pass unnoticed: it fails the
#: non-vacuity floor in
#: ``TestRepairableFieldTable::test_the_derived_candidate_set_cannot_silently_collapse``,
#: which requires every registered non-``report_*`` tool to be a sweep
#: candidate and does not care HOW a tool writes. The outcome is loud in
#: every case; only the remedy differs (audit the row, or teach the
#: derivation about the new shape).
_DESCRIPTION_ALSO: tuple[str, ...] = ('replace_plan_step', 'mark_step_committed')
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
#: Bound ONCE, immediately below the six writer functions it derives its
#: ``schema_params`` from (``_params_of`` needs them to exist). Annotated but
#: deliberately UNBOUND here, so a use before that point raises a loud
#: NameError instead of silently reading an empty table and repairing nothing.
_REPAIRABLE_PLAN_FIELDS: tuple[PlanField, ...]


def _plan_field(
    collection: str | None,
    field: str,
    writer: Callable[..., Any],
    target_keys: Mapping[str, str],
    also_written_by: tuple[str, ...] = (),
) -> PlanField:
    """A row whose ``tool`` and ``schema_params`` are both read off *writer*."""
    return PlanField(
        collection,
        field,
        writer.__name__.removeprefix('_'),
        _params_of(writer),
        target_keys,
        also_written_by,
    )


def _build_repairable_plan_fields() -> tuple[PlanField, ...]:
    """Construct :data:`_REPAIRABLE_PLAN_FIELDS`; see its docs above."""
    return (
        _plan_field(None, 'title', _create_plan, _CREATE_PLAN_TARGETS),
        _plan_field(
            None, 'analysis', _create_plan, _CREATE_PLAN_TARGETS, _UPDATE_METADATA_ALSO
        ),
        _plan_field(
            'prerequisites',
            'description',
            _add_prerequisite,
            _ADD_PREREQUISITE_TARGETS,
            _DESCRIPTION_ALSO,
        ),
        _plan_field(
            'steps', 'description', _add_plan_step, _ADD_PLAN_STEP_TARGETS, _DESCRIPTION_ALSO
        ),
        _plan_field(
            'design_decisions', 'decision', _add_design_decision, _ADD_DESIGN_DECISION_TARGETS
        ),
        _plan_field(
            'design_decisions', 'rationale', _add_design_decision, _ADD_DESIGN_DECISION_TARGETS
        ),
        _plan_field('reuse', 'what', _add_reuse_item, _ADD_REUSE_ITEM_TARGETS),
        _plan_field('reuse', 'where', _add_reuse_item, _ADD_REUSE_ITEM_TARGETS),
        _plan_field('reuse', 'how', _add_reuse_item, _ADD_REUSE_ITEM_TARGETS),
        _plan_field('dropped_files', 'reason', _drop_plan_file, _DROP_PLAN_FILE_TARGETS),
    )


def _read_plan_repaired(artifacts: TaskArtifacts) -> tuple[dict, list[dict[str, Any]]]:
    """Read this server's plan, repaired under :data:`_REPAIRABLE_PLAN_FIELDS`."""
    return plan_markup_repair.read_plan_repaired(artifacts, _REPAIRABLE_PLAN_FIELDS)


_P = ParamSpec('_P')
_R = TypeVar('_R')


def _plan_mutation(
    fn: Callable[Concatenate[TaskArtifacts, _P], _R],
) -> Callable[Concatenate[TaskArtifacts, _P], _R]:
    """The whole mutator, including the repair write-back in ``_read_plan_repaired``
    and any nested TaskArtifacts write, is ONE critical section against every
    other plan.json writer."""

    @functools.wraps(fn)
    def mutation(artifacts: TaskArtifacts, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with artifacts.plan_mutex():
            return fn(artifacts, *args, **kwargs)

    return mutation


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
        response[plan_markup_repair.MARKUP_REPAIRS_KEY] = facts
    return response


def _body_edit(plan: dict[str, Any]) -> Plan:
    """*plan* edited outside its step set: its confirmation is kept."""
    return Plan.from_json(plan)


def _step_set_edit(plan: dict[str, Any]) -> Plan:
    """*plan* after a step-set edit: its confirmation is cleared (``artifacts.py::Plan.restructured``)."""
    return Plan.from_json(plan).restructured()


# ---------------------------------------------------------------------------
# Standalone implementation functions (testable without MCP transport)
# ---------------------------------------------------------------------------


STEP_TYPES: frozenset[str] = frozenset({'test', 'impl'})


def _step_type_refusal(step_type: str) -> dict[str, Any] | None:
    """The one enforcement of :data:`STEP_TYPES`: ``None`` for a member, else the refusal."""
    if step_type in STEP_TYPES:
        return None
    allowed = sorted(STEP_TYPES)
    return {
        'status': 'error',
        'message': (
            f'step_type must be one of {allowed}; got {step_type!r}. '
            "The plan stores this value under the key 'type'."
        ),
        'step_type': step_type,
        'allowed_step_types': allowed,
    }


@_plan_mutation
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
    # DELIBERATELY NOT hooked to _read_plan_repaired: inbound arguments are the
    # write-time middleware's (plans/toolcall-markup-containment-prd.md D8).
    #
    # THE BOOKKEEPING READ BELOW IS NOT A BREACH OF THAT RULE. It goes
    # through a plain `artifacts.read_plan()` for the same reason, and it
    # touches only the guard's own `_markup_rejections` block, the C4
    # descope record (`plan_descope.DESCOPE_PLAN_KEYS`), whose prose already
    # passed the inbound guard on `record_descoped_step`, and the quarantine
    # back-pointer (`quarantine_provenance.QUARANTINED_FROM_KEY`), which the
    # orchestrator writes and which carries no agent prose. Do not "simplify"
    # it onto `_read_plan_repaired`.
    files = _coerce_files(files)
    carried: dict[str, Any] = {}
    replaced = Plan.from_json({})
    try:
        # TWO BOOKKEEPING MOVES, both required and for DIFFERENT reasons
        # (task 4597).
        #
        # CARRY FORWARD, because this function overwrites plan.json WHOLESALE:
        # a re-plan, or a second create_plan in one session, would otherwise
        # erase a counter earned earlier — losing exactly the record it exists
        # to keep, at the moment a reader most wants it.
        #
        # DRAIN, because a create_plan refused before any plan existed had no
        # document to stamp and was buffered instead. This is the plan it was
        # waiting for. Without it, the loudest leak shape on this server — an
        # architect bounced repeatedly before its plan exists — would be the
        # one case the counter could never describe.
        #
        # Both are MERGES, so the carried-forward and buffered blocks compose
        # rather than one clobbering the other. Neither writes a key when there
        # is nothing to record, so the overwhelmingly common clean path
        # produces a document byte-identical to what it produced before.
        #
        # THE CARRY-FORWARD GOES THROUGH THE ALGEBRA, never raw. plan.json is
        # agent-adjacent, and every other consumer of a stored block
        # (`merge_block`, `summary`) already degrades what it finds; copying the
        # on-disk value verbatim would be the one path that launders a mangled
        # block into a brand-new document — and from there into the four
        # architect-facing prompts that embed the plan. `normalize_block`
        # returns None when nothing survives, so an unrecoverable value is
        # DROPPED rather than laundered into a present-and-zero key that would
        # contradict this key's contract that its PRESENCE is the signal.
        existing = artifacts.read_plan()
        block = plan_markup_stamp.normalize_block(
            existing.get(plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY)
            if isinstance(existing, dict) else None
        )
        if block is not None:
            carried[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY] = block
        plan_markup_stamp.drain_pending(carried)
        # The descope record is the one trace of a done step a revalidation
        # dropped; a wholesale overwrite must not erase it.
        if isinstance(existing, dict):
            carried.update(plan_descope.descope_record_of(existing))
            carried.update(quarantine_provenance.quarantined_from_of(existing))
            replaced = Plan.from_json(existing)
    except Exception:
        # BOOKKEEPING CAN NEVER FAIL A create_plan. The counter is a legibility
        # aid; the plan is the work. Losing the count is a cost an operator
        # absorbs, losing the plan is not.
        logger.exception(
            'markup stamp: could not carry the rejection counter into the new '
            'plan for %s; the plan itself is unaffected', task_id,
        )
    recreated = replaced.recreated(
        task_id=task_id, title=title, analysis=analysis, files=files, carried=carried,
    )
    artifacts.write_plan_model(recreated, mutex_held=True)
    return {'status': 'ok', 'task_id': task_id}


@_plan_mutation
def _add_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
    step_type: str,
    description: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    refusal = _step_type_refusal(step_type)
    if refusal is not None:
        return _with_markup_repairs(refusal, markup_facts)

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

    step = {
        'id': step_id,
        'type': step_type,
        'description': description,
        'status': 'pending',
        'commit': None,
    }
    plan.setdefault('steps', []).append(step)
    artifacts.write_plan_model(_step_set_edit(plan), mutex_held=True)
    return _with_markup_repairs(
        foreign_task_meta.annotate(
            {'status': 'ok', 'step_id': step_id, 'total_steps': len(plan['steps'])},
            foreign_task_meta.step_findings(artifacts, [step]),
        ),
        markup_facts,
    )


@_plan_mutation
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

    prerequisite = {
        'id': prereq_id,
        'description': description,
        'status': 'pending',
        'commit': None,
        'tests': [],
    }
    plan.setdefault('prerequisites', []).append(prerequisite)
    artifacts.write_plan_model(_step_set_edit(plan), mutex_held=True)
    return _with_markup_repairs(
        foreign_task_meta.annotate(
            {'status': 'ok', 'prereq_id': prereq_id},
            foreign_task_meta.step_findings(artifacts, [prerequisite]),
        ),
        markup_facts,
    )


@_plan_mutation
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
    artifacts.write_plan_model(_body_edit(plan), mutex_held=True)
    return _with_markup_repairs(
        {'status': 'ok', 'total_decisions': len(plan['design_decisions'])},
        markup_facts,
    )


@_plan_mutation
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
    artifacts.write_plan_model(_body_edit(plan), mutex_held=True)
    return _with_markup_repairs(
        {'status': 'ok', 'total_reuse': len(plan['reuse'])}, markup_facts
    )


@_plan_mutation
def _drop_plan_file(
    artifacts: TaskArtifacts,
    path: str,
    reason: str,
) -> dict[str, Any]:
    """Drop a declared file from ``plan['files']`` AND record why, atomically.

    The honest third exit from the architect narrowing pass. The pre-merge
    plan-files gate names the dilemma twice —
    ``orchestrator/src/orchestrator/merge_gates.py::CROSS_REPO_DELIVERABLE_REASON_PREFIX``
    and ``::ALREADY_LANDED_REASON_PREFIX``: "drop = falsify provenance,
    confirm = mislabel complete work". A bare
    ``update_plan_metadata(files=[narrowed])`` is the falsifying half — the
    entry vanishes with no record that it was ever correctly in scope. Dropping
    and recording the reason in ONE call closes that: the entry leaves the
    gate's re-check surface while the plan keeps why it was declared and why
    the branch legitimately needed no edit to it.

    Atomicity is the invariant, not a convenience. Because the removal and the
    note are the same write, it is structurally impossible to note a file that
    was kept, to drop without a reason, or to ADD a file — so
    ``orchestrator/src/orchestrator/workflow.py::TaskWorkflow._try_narrow_plan``'s
    ``after.issubset(before)`` guard holds by construction whenever the stored
    ``files`` is a list. A stored non-list ``files`` breaks that: this function
    coerces it and writes back a real list, but the guard takes ``before``
    from the raw stored value.
    """
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    # Every refusal below returns BEFORE any artifacts.write_plan_model call — the
    # module-wide convention that a refusal envelope always implies no write.
    if not reason or not reason.strip():
        return _with_markup_repairs(
            {
                'status': 'error',
                'message': (
                    f'Cannot drop {path!r} without a reason. Every drop must '
                    'record WHY the entry was in scope and WHY the branch '
                    'legitimately needed no edit to it — a drop with no '
                    'recorded reason is the falsified provenance this tool '
                    'exists to avoid.'
                ),
            },
            markup_facts,
        )

    current = _coerce_files(plan.get('files'))
    if path not in current:
        return _with_markup_repairs(
            {
                'status': 'error',
                'message': (
                    f'{path!r} is not in the plan files list, so there is '
                    'nothing to drop and no declaration to explain. Current '
                    f'files: {current!r}'
                ),
            },
            markup_facts,
        )

    # Scoped HERE rather than left to _confirm_plan's own empty-files check:
    # the narrowing pass's dropping option never calls confirm_plan() at all,
    # so that check never fires on this route.
    if len(current) <= 1:
        return _with_markup_repairs(
            {
                'status': 'error',
                'message': (
                    f'Refusing to drop {path!r}: it is the last file in the '
                    'plan, and the plan must never be narrowed to an empty '
                    'files list — an empty list is not a narrowed plan, it is '
                    'an unchecked one. If nothing in the plan legitimately '
                    'remains, call confirm_plan() instead and let the '
                    'escalation triage the scope.'
                ),
            },
            markup_facts,
        )

    plan['files'] = [f for f in current if f != path]
    plan.setdefault('dropped_files', []).append({
        'path': path,
        'reason': reason,
    })
    artifacts.write_plan_model(_body_edit(plan), mutex_held=True)
    return _with_markup_repairs(
        {
            'status': 'ok',
            'dropped': path,
            'total_dropped': len(plan['dropped_files']),
            'files': len(plan['files']),
        },
        markup_facts,
    )


# The repairable-field table, bound here because ``_params_of`` reads the six
# writer signatures above off the live functions. Declared and documented at
# its annotation further up; nothing between that point and here reads it.
_REPAIRABLE_PLAN_FIELDS = _build_repairable_plan_fields()


@_plan_mutation
def _mark_step_done(
    artifacts: TaskArtifacts,
    step_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    # CHEAP IN-MEMORY VALIDATION FIRST. The step lookup is a dict walk over an
    # already-parsed plan; the reachability guard below shells out to git
    # twice. Resolving the id first means a typo'd step id reports the
    # actionable 'Step ... not found in plan.' instead of a reachability
    # complaint about a sha that was never going to be written, and no call
    # that cannot write pays for a subprocess.
    if not any(
        isinstance(item, dict) and item.get('id') == step_id
        for collection in ('prerequisites', 'steps')
        for item in plan.get(collection, [])
    ):
        return _with_markup_repairs(
            {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'},
            markup_facts,
        )
    # task 3651 — a plan step must never record a sha the branch does not
    # carry: a bogus commit here silently destroys the step's provenance and
    # feeds workflow.py::_reconcile_done_step_commits an orphan it cannot
    # resolve. Placed AFTER _read_plan_repaired because that helper has already
    # written any repair back to disk, so the reject path must still report it
    # via _with_markup_repairs like every other exit.
    #
    # Fail-OPEN on UNKNOWN, deliberately asymmetric to _mark_step_committed's
    # fail-CLOSED branch_probes.sha_exists_on_branch: this caller is an implementer
    # recording work it JUST committed, so a false reject would leave the step
    # `pending` while its code sits on the branch — inviting a re-implementation
    # on top of a correct one, the exact harm the guard exists to prevent. The
    # degradation is logged at WARNING rather than passing silently.
    reachability = branch_probes.sha_branch_reachability(artifacts.worktree, commit_sha)
    if reachability is Reachability.UNREACHABLE:
        return _with_markup_repairs(
            {
                'status': 'error',
                'message': (
                    f'commit_sha {commit_sha!r} is not reachable from HEAD on '
                    f'this branch (git merge-base --is-ancestor failed) — a '
                    f'plan step must never record a sha the branch does not '
                    f'carry'
                ),
            },
            markup_facts,
        )
    if reachability is Reachability.UNKNOWN:
        logger.warning(
            'mark_step_done(%r, %r): on-branch reachability check could not '
            'run (no usable git repo at %s); recording the sha unverified',
            step_id, commit_sha, artifacts.worktree,
        )
    # The repair write-back has already landed, so the re-read inside
    # update_step_status picks up the repaired document.
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


def _session_identity_in_hand(artifacts: TaskArtifacts, plan: dict) -> str | None:
    """The agent sidecar's session id, else the planner's ``_session_id``, else None."""
    session = artifacts.read_agent_session()
    if session is not None and session.session_id:
        return session.session_id
    plan_session = plan.get('_session_id')
    if isinstance(plan_session, str) and plan_session:
        return plan_session
    return None


@_plan_mutation
def _record_step_deviation(
    artifacts: TaskArtifacts,
    step_id: str,
    kind: str,
    observed: str,
    expected: str,
    action_taken: str,
    prose_only: bool = False,
    note: str = '',
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    try:
        record = DeviationRecord(
            kind=kind,
            observed=observed,
            expected=expected,
            action_taken=action_taken,
            prose_only=prose_only,
            note=note,
            recorded_at=datetime.now(UTC).isoformat(),
            recorded_by=_session_identity_in_hand(artifacts, plan),
        )
    except UnknownDeviationKind as exc:
        return _with_markup_repairs(
            {
                'status': 'error',
                'message': str(exc),
                'kind': exc.kind,
                'allowed_kinds': list(exc.allowed),
            },
            markup_facts,
        )

    count = artifacts.append_step_deviation(step_id, record)
    if count is None:
        return _with_markup_repairs(
            {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'},
            markup_facts,
        )
    return _with_markup_repairs(
        {'status': 'ok', 'step_id': step_id, 'deviations': count}, markup_facts
    )


def _descope_refusal(exc: plan_descope.InvalidDescopeRecord) -> dict[str, Any]:
    refusal: dict[str, Any] = {'status': 'error', 'message': str(exc), 'field': exc.field}
    if exc.field == 'disposition':
        refusal['allowed_dispositions'] = sorted(plan_descope.DESCOPE_DISPOSITIONS)
    return refusal


def _upsert_descoped_step(entries: object, entry: plan_descope.DescopedStep) -> list:
    """*entries* without any record of *entry*'s ``plan_descope.same_done_record``, then *entry*."""
    def same_pair(old: object) -> bool:
        if not isinstance(old, dict):
            return False
        old_id, old_commit = old.get('step_id'), old.get('commit')
        return (
            isinstance(old_id, str)
            and isinstance(old_commit, str)
            and plan_descope.same_done_record((old_id, old_commit), entry.done_record)
        )

    kept = [old for old in entries if not same_pair(old)] if isinstance(entries, list) else []
    return [*kept, entry.to_dict()]


@_plan_mutation
def _record_descoped_step(
    artifacts: TaskArtifacts,
    step_id: str,
    commit: str,
    disposition: str,
    authority: str,
    note: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    try:
        entry = plan_descope.DescopedStep(
            step_id=step_id,
            commit=commit,
            disposition=disposition,
            authority=authority,
            note=note,
        )
    except plan_descope.InvalidDescopeRecord as exc:
        return _with_markup_repairs(_descope_refusal(exc), markup_facts)

    entries = _upsert_descoped_step(plan.get(plan_descope.DESCOPED_STEPS_KEY), entry)
    plan[plan_descope.DESCOPED_STEPS_KEY] = entries
    plan[plan_descope.RESCOPED_AT_KEY] = datetime.now(UTC).isoformat()
    plan[plan_descope.RESCOPED_BY_KEY] = (
        _session_identity_in_hand(artifacts, plan) or plan_descope.RESCOPED_BY_UNKNOWN
    )
    plan[plan_descope.RESCOPED_NOTE_KEY] = note
    artifacts.write_plan_model(_step_set_edit(plan), mutex_held=True)
    return _with_markup_repairs(
        {'status': 'ok', 'step_id': step_id, 'commit': commit, 'descoped_steps': len(entries)},
        markup_facts,
    )


# ---------------------------------------------------------------------------
# Revalidation helpers
# ---------------------------------------------------------------------------


@_plan_mutation
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
        _forget_redeclared_drops(plan)
    if analysis is not None:
        plan['analysis'] = analysis
    artifacts.write_plan_model(_body_edit(plan), mutex_held=True)
    return _with_markup_repairs(
        {
            'status': 'ok',
            'files': len(plan.get('files', [])),
        },
        markup_facts,
    )


def _forget_redeclared_drops(plan: dict) -> None:
    """Keep ``files`` and ``dropped_files`` disjoint: a re-declared path is back
    in scope, so its drop record no longer describes the plan."""
    drops = plan.get('dropped_files')
    if not isinstance(drops, list):
        return
    declared = set(plan['files'])
    plan['dropped_files'] = [
        entry for entry in drops
        if not (isinstance(entry, dict) and entry.get('path') in declared)
    ]


@_plan_mutation
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
                artifacts.write_plan_model(_step_set_edit(plan), mutex_held=True)
                return _with_markup_repairs(
                    {'status': 'ok', 'removed': step_id, 'collection': collection},
                    markup_facts,
                )

    return _with_markup_repairs(
        {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'},
        markup_facts,
    )


@_plan_mutation
def _replace_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
    step_type: str,
    description: str,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    refusal = _step_type_refusal(step_type)
    if refusal is not None:
        return _with_markup_repairs(refusal, markup_facts)

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
                artifacts.write_plan_model(_step_set_edit(plan), mutex_held=True)
                return _with_markup_repairs(
                    foreign_task_meta.annotate(
                        {'status': 'ok', 'replaced': step_id},
                        foreign_task_meta.step_findings(artifacts, [item]),
                    ),
                    markup_facts,
                )

    return _with_markup_repairs(
        {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'},
        markup_facts,
    )


@_plan_mutation
def _confirm_plan(
    artifacts: TaskArtifacts,
) -> dict[str, Any]:
    plan, markup_facts = _read_plan_repaired(artifacts)
    # THE REJECTION VIEW, COMPUTED ONCE AND BEFORE THE FIRST GUARD (task 4597).
    #
    # confirm_plan is the architect's LAST tool result, and therefore the one
    # place a refusal reaches the durable agent transcript while the architect
    # can still act on it — the block is already on disk by then, but plan.json
    # is read by LATER agents. That much is true of every branch; what makes
    # the ERROR branches the important ones is that a LEAKING architect lands
    # on them. Refused add_plan_step calls are what leaves a plan stepless, and
    # a refused create_plan is what leaves it absent — so the three exits that
    # first went without the counter are precisely the exits that need it.
    #
    # `session_summary`, NOT `summary`: on the no-plan branch there is no
    # document to carry a block, and the refused create_plan is sitting in the
    # pending buffer. A plan-only view would stay silent in the one case that
    # most needs explaining.
    #
    # OMIT-WHEN-ABSENT, exactly as `_with_markup_repairs`: absent on the clean
    # path, never present-and-zero, so every existing confirm_plan response
    # stays byte-identical and the key's PRESENCE is an unambiguous signal.
    #
    # A SUMMARY, not the block: `session_summary` returns {count, by_tool}
    # only. The events and the note are already two keys away in the document,
    # and echoing them here would put the block's bulk into the largest
    # response the architect reads, to say what the two numbers already say.
    #
    # COMPOSED WITH `_with_markup_repairs` rather than folded into it: the two
    # diagnostics answer different questions (what this read REPAIRED, versus
    # what this session has REFUSED) and neither should be able to suppress
    # the other.
    rejections = plan_markup_stamp.session_summary(plan)
    findings = foreign_task_meta.step_findings(
        artifacts, [*(plan.get('prerequisites') or []), *(plan.get('steps') or [])],
    ) if plan else {}

    def _respond(payload: dict[str, Any]) -> dict[str, Any]:
        """The ONE place that decides what a confirm_plan response carries.

        Four exits each deciding for themselves is what produced the defect
        this closes — three of them silently omitted the counter. Routing the
        no-plan exit through here also gives it the `_with_markup_repairs`
        wrapping its three siblings already had, which is a harmless
        unification: `_read_plan_repaired` is total on a missing plan and
        returns no facts, so the key stays absent exactly as before.
        """
        if rejections is not None:
            payload['markup_rejections'] = rejections
        return _with_markup_repairs(foreign_task_meta.annotate(payload, findings), markup_facts)

    if not plan:
        return _respond({'status': 'error', 'message': 'No plan exists.'})
    model = Plan.from_json(plan)
    if model.state is PlanState.SHELL:
        return _respond(
            {'status': 'error', 'message': 'Plan has no steps — cannot confirm.'}
        )
    if not plan.get('files'):
        return _respond({
            'status': 'error',
            'message': (
                'Plan has no files — cannot confirm. Call create_plan (or '
                'update_plan_metadata) with a non-empty files list first.'
            ),
        })

    # What confirmation records, and what clears it: ``orchestrator/artifacts.py::Plan.confirmed``.
    confirmed = model.confirmed(at=datetime.now(UTC).isoformat(), by=PlanConfirmation.ARCHITECT)
    artifacts.write_plan_model(confirmed, mutex_held=True)
    return _respond({
        'status': 'ok',
        'finalized': True,
        'steps': len(model.steps),
        'files': len(plan.get('files', [])),
    })


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
    (a full 40-character lowercase hex sha, reachable from main and
    attributable to this task) and sets task status to ``done``.  Nothing
    is validated here, so a malformed *commit* is refused only after the
    session has ended.
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


def _report_missing_plan_file(
    artifacts: TaskArtifacts,
    path: str,
    reason: str,
) -> dict[str, Any]:
    """Record a writer's request to add an existing file to the declared scope.

    The workflow applies it at the next writer boundary
    (``workflow.py::TaskWorkflow._apply_scope_requests``). Reads the declared
    files with ``peek_plan``, which never writes plan.json. A path naming no
    file in the worktree is refused, so a typo never reaches plan.files.
    """
    try:
        request = ScopeRequest(path, reason)
    except InvalidScopeRequest as exc:
        return {'status': 'error', 'field': exc.field, 'message': str(exc)}
    if not (artifacts.worktree / request.path).is_file():
        return {
            'status': 'error',
            'field': 'path',
            'message': (
                f'path {request.path!r} names no file in the worktree; '
                'a file you create needs no report'
            ),
        }
    if not pending_scope_paths([request], artifacts.peek_plan().get('files', [])):
        return {'status': 'already_in_scope', 'path': request.path}
    pending = artifacts.write_scope_request(request)
    return {'status': 'ok', 'path': request.path, 'pending': pending}


def _mark_step_committed(
    artifacts: TaskArtifacts,
    step_id: str,
    sha: str,
) -> dict[str, Any]:
    """Pre-satisfy a plan step/prerequisite at AUTHORING time.

    Guards on a real on-branch reachability check for *sha* (corroborate-before-
    acting) before delegating the status/commit/description-tag mutation to
    :meth:`TaskArtifacts.mark_step_committed`. The probe reads no plan, so it
    runs BEFORE ``plan_mutex()``: no plan writer waits on its git subprocess.
    Returns the Contract A1 shapes:
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
    if not branch_probes.sha_exists_on_branch(artifacts.worktree, sha):
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
    with artifacts.plan_mutex():
        _, markup_facts = _read_plan_repaired(artifacts)
        committed = artifacts.mark_step_committed(step_id, sha)
    if not committed:
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
# The boundary guard's escalation channel (task 4457; promoted to a shared
# module by task 3690)
# ---------------------------------------------------------------------------
#
# The MECHANISM now lives in ``orchestrator.mcp.markup_sink`` — ONE channel
# shared with verdict-tools, rather than a second, weaker one per server
# (INV-5). What stays here is plan-tools' own DECLARATION of it: the spec
# below, and the thin module-level seams the channel is reached through.
#
# The seams are not ceremony. ``_markup_project_root`` and
# ``_escalation_channel`` are resolved through THIS module's globals at call
# time (the lambdas in ``_markup_escalation_sink`` close over the names, not
# the functions), so a test that substitutes either one still steers the sink
# a `create_server` already built.

#: Filed-by attribution. Names the GUARD, not the leaking agent.
_MARKUP_AGENT_ROLE = 'plan-tools-markup-guard'

#: The two queue anchors. Both are NON-TASK ids, and the residue one is
#: load-bearing at that: the middleware declares residue at ``level=2``, and a
#: pending level>=2 record carrying a LIVE task id halts that task. See
#: ``MarkupSinkSpec.residue_anchor_task_id`` for the full reading of the gate.
_MARKUP_RESIDUE_ANCHOR_TASK_ID = 'plan-tools-markup-residue'
_MARKUP_STORM_ANCHOR_TASK_ID = 'plan-tools-markup-storm'

_MARKUP_UNATTRIBUTED_SUBJECT = markup_sink.MARKUP_UNATTRIBUTED_SUBJECT

#: plan-tools' declaration of its own channel. Every field is stated here
#: rather than defaulted in ``markup_sink``: adding a server must be a decision
#: at every axis, not an inheritance of this server's answers.
MARKUP_SINK_SPEC = markup_sink.MarkupSinkSpec(
    server_label='plan-tools',
    agent_role=_MARKUP_AGENT_ROLE,
    residue_anchor_task_id=_MARKUP_RESIDUE_ANCHOR_TASK_ID,
    storm_anchor_task_id=_MARKUP_STORM_ANCHOR_TASK_ID,
    refusal_consequence='NOTHING was written.',
    storm_consequence=(
        "further specimens land permanently in the fleet's plan.json files."
    ),
    #: Composed from ``MARKUP_JOURNAL_DIRNAME`` rather than respelled, so the
    #: constant stays the single owner of the path and the instruction cannot
    #: drift away from the artifact it names (task 4744).
    attribution_source=(
        'every markup fact this server sees is journalled one line per event '
        f'to {markup_journal.MARKUP_JOURNAL_DIRNAME}/plan-tools.jsonl under '
        'the main checkout — each line carries subject_task_id, tool, param, '
        'outcome and a UTC timestamp, so the leaking task is nameable from '
        "the burst window's own lines (jq or grep '\"subject_task_id\"')"
    ),
)


def _markup_project_root(worktree: Path) -> Path | None:
    """plan-tools' seam onto :func:`markup_sink.resolve_project_root`."""
    return markup_sink.resolve_project_root(worktree)


def _escalation_channel(project_root: Path) -> tuple[Any, Any] | None:
    """plan-tools' seam onto :func:`markup_sink.open_escalation_channel`."""
    return markup_sink.open_escalation_channel(project_root)


def _markup_subject_task_id(artifacts: TaskArtifacts) -> str:
    """Which task's call leaked — the record's SUBJECT, not its routing key.

    Deliberately NOT what goes in the escalation's ``task_id`` field: that is
    ``_MARKUP_RESIDUE_ANCHOR_TASK_ID``, because a level-2 record filed under a
    LIVE task id halts that task. This is the attribution that then rides in
    the summary and the detail instead.

    The plan's own ``task_id`` first; the worktree directory name when there is
    no plan yet (a ``create_plan`` refused before any plan exists); an explicit
    "unattributed" only when even that is empty — the last two fallbacks are
    ``markup_sink``'s, applied to whatever this returns. An unreadable plan
    degrades to the same fallback rather than losing the record.
    """
    try:
        plan = artifacts.read_plan()
    except Exception:
        logger.warning(
            'markup guard: could not read the plan for attribution at %s; '
            'falling back to the worktree name', artifacts.artifact_path('plan.json'),
        )
        plan = {}
    task_id = plan.get('task_id') if isinstance(plan, dict) else None
    if isinstance(task_id, str) and task_id:
        return task_id
    return artifacts.worktree.name or _MARKUP_UNATTRIBUTED_SUBJECT


def _markup_escalation_sink(
    artifacts: TaskArtifacts,
    loss_sink: Callable[[dict[str, Any]], Awaitable[object]],
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Build plan-tools' emitter — the shared channel, this server's spec.

    The queue-or-nothing payload floor is ``markup_sink``'s, shared with
    verdict-tools. A record the queue cannot take is handed to *loss_sink*.
    """
    return markup_sink.make_escalation_sink(
        worktree=artifacts.worktree,
        spec=MARKUP_SINK_SPEC,
        subject_task_id=lambda: _markup_subject_task_id(artifacts),
        loss_sink=loss_sink,
        # Resolved through THIS module's globals per call, so a test that
        # substitutes either seam steers an already-built sink.
        resolve_root=lambda worktree: _markup_project_root(worktree),
        open_channel=lambda project_root: _escalation_channel(project_root),
    )


def _markup_fact_journal(
    artifacts: TaskArtifacts,
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Build plan-tools' DURABLE fact channel (task 4744).

    The middleware emits one ``markup_detected`` fact on every outcome, and it
    is the only record anywhere that names WHICH call leaked. Before this it
    reached exactly one place: a ``logger.warning`` in a per-agent stdio
    subprocess whose stderr the CLI agent that spawned it consumes. This gives
    it a consumer that survives the process — see ``markup_journal``.

    ``resolve_root`` is routed through THIS module's ``_markup_project_root``
    global (a lambda closing over the NAME, not the function), exactly as the
    escalation sink's is, so a test that substitutes that seam steers a journal
    ``create_server`` has already built. It is also why one patch steers both
    channels to the same project root — which is the correct coupling: a record
    and the journal line about it belong in the same checkout.
    """
    return markup_journal.make_fact_journal(
        worktree=artifacts.worktree,
        server_label=MARKUP_SINK_SPEC.server_label,
        subject_task_id=lambda: _markup_subject_task_id(artifacts),
        resolve_root=lambda worktree: _markup_project_root(worktree),
    )


def _markup_plan_stamp(
    artifacts: TaskArtifacts,
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Build plan-tools' PLAN-STAMPING fact channel (task 4597, esc-4528-1).

    A THIN SEAM over ``plan_markup_stamp.make_plan_stamp``, spelled beside
    ``_markup_fact_journal`` and reached the same way — through this module's
    globals at call time — so a test that substitutes it steers a sink
    ``create_server`` has already built.

    The journal answers "who leaked, from which tool, into which parameter,
    when" for an OPERATOR reading a file under the main checkout. This answers
    the same question for a later READER OF THE PLAN, which is a different
    audience reached through a different artifact: the implementer briefing
    tells its agent to open ``.task/plan.json`` directly, and the four
    architect-facing prompts embed the document verbatim. Neither audience ever
    sees the journal.
    """
    return plan_markup_stamp.make_plan_stamp(artifacts=artifacts)


def _markup_fact_sink(
    artifacts: TaskArtifacts,
    journal: Callable[[dict[str, Any]], Awaitable[str | None]],
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Fan one markup fact out to BOTH of this server's fact channels.

    ``MarkupGuardMiddleware`` accepts exactly ONE ``fact_sink``, so the
    composition lives here. Teaching the middleware about a list of sinks would
    change a boundary shared with verdict-tools and the escalation server for a
    need only plan-tools has — neither of the others owns a ``plan.json`` to
    stamp — and the registration site is where every other server-specific
    answer on this boundary already lives (``MarkupSinkSpec`` exists precisely
    so nothing is inferred from a server name at call time).

    EACH ARM IS ISOLATED, and that is load-bearing rather than defensive style.
    The middleware's ``_call_sink`` wraps the WHOLE sink in ONE try/except, so
    a composed sink that let the first arm's exception propagate would silently
    skip the second entirely — one channel's outage taking the other down,
    which is the exact fail-soft the containment PRD exists to end.

    THE JOURNAL RUNS FIRST, so a stamp failure can never delay the established
    durable record the storm escalation points an operator at, and the
    composed sink returns THE JOURNAL'S locator — the existing fact-sink return
    contract is unchanged, and the stamp is purely additive.

    Both emitters are built ONCE rather than per record: each memoizes its own
    resolution state, and rebuilding them per call would discard it. The
    journal is built by ``create_server`` and shared with the escalation sink's
    loss channel; the stamp is built here.
    """
    stamp = _markup_plan_stamp(artifacts)

    async def fact_sink(record: dict[str, Any]) -> str | None:
        locator: str | None = None
        try:
            locator = await journal(record)
        except Exception:
            # Both emitters already contain every failure they can see; these
            # are the floors under the ARMS THEMSELVES, so one channel's
            # outage costs only its own record.
            logger.exception(
                'markup guard: the fact journal failed for %r; continuing to '
                'the plan stamp', record.get('fact'),
            )
        try:
            await stamp(record)
        except Exception:
            logger.exception(
                'markup guard: the plan stamp failed for %r; the journal '
                'result stands', record.get('fact'),
            )
        return locator

    return fact_sink



# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------


def create_server(
    artifacts: TaskArtifacts, granted_tools: frozenset[str] | None = None,
    caller_role: str | None = None,
) -> FastMCP:
    """Create the plan-tools MCP server; the one home of its grant policy.

    *granted_tools* ``None`` registers every tool, for in-process callers.
    Otherwise only the granted names stay registered, so an ungranted tool
    neither lists nor dispatches. A granted name this server does not
    register is dropped with a WARNING rather than refused, because a
    running launcher can name a tool this on-disk server has since renamed
    or removed: plans/plan-deviation-recording-prd.md D14.

    *caller_role* names the role this server serves on its markup records;
    ``None`` leaves them attributed from each call's arguments alone.
    """
    mcp, registered = _server_with_every_tool(artifacts, caller_role)
    if granted_tools is not None:
        unknown = sorted(granted_tools - registered)
        if unknown:
            logger.warning(
                'plan-tools grant names tool(s) %s this server does not '
                'register (registered: %s); dropping them',
                unknown, sorted(registered),
            )
        for name in registered - granted_tools:
            mcp.local_provider.remove_tool(name)
    return mcp


def _server_with_every_tool(
    artifacts: TaskArtifacts, caller_role: str | None = None,
) -> tuple[FastMCP, frozenset[str]]:
    mcp = FastMCP('plan-tools')
    registered: list[str] = []

    def tool(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        registered.append(fn.__name__)
        mcp.tool()(fn)
        return fn

    # THE WRITE-TIME BOUNDARY GUARD (task 4457, PRD contract C2). Every INBOUND
    # string argument of every tool on this server is scanned for leaked
    # tool-call envelope markup before pydantic validates it. Each wiring
    # choice below is stated where it is owned:
    #
    # * REJECT_WITH_REPAIR, the empty exempt_tools and a real escalation_sink:
    #   plans/toolcall-markup-containment-prd.md section 4 C2 and section 9 γ2.
    # * The deliberate-quoting override every tool here declares: PRD D12.
    # * Residue filed under a non-task anchor: PRD D9.
    # * The durable fact journal and the measurement that wired it:
    #   orchestrator/mcp/markup_journal.py's docstring.
    # * The plan stamp, and the narrowing of "a refusal leaves plan.json
    #   byte-identical" to "every authored field identical":
    #   orchestrator/mcp/plan_markup_stamp.py's docstring.
    # * The one journal both sinks share:
    #   orchestrator/src/orchestrator/mcp/markup_sink.py::make_escalation_sink.
    # * The caller role, named by the launch because no tool here declares
    #   one: shared/src/shared/mcp_markup_middleware.py::HostAttribution.
    # * The stored-markup facts a response discloses, journalled through the
    #   same journal: orchestrator/src/orchestrator/mcp/stored_markup_journal.py.
    journal = _markup_fact_journal(artifacts)
    mcp.add_middleware(
        MarkupGuardMiddleware(
            RepairPolicy.REJECT_WITH_REPAIR,
            exempt_tools=frozenset(),
            escalation_sink=_markup_escalation_sink(artifacts, journal),
            fact_sink=_markup_fact_sink(artifacts, journal),
            host_attribution=(
                None if caller_role is None
                else lambda: HostAttribution(agent_role=caller_role)
            ),
        )
    )
    mcp.add_middleware(MissingArgumentMiddleware())
    mcp.add_middleware(StoredMarkupJournalMiddleware(journal))

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
    def add_plan_step(
        step_id: str,
        step_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a TDD step to the plan. Steps execute in the order added.

        The response may carry ``foreign_task_meta``, naming
        paths in the description that no lane of this task can write.

        Args:
            step_id: Unique step ID (e.g. "step-1", "step-2").
            step_type: One of ``STEP_TYPES``: "test" or "impl". Anything else
                is refused. The plan stores this value under the key ``type``.
            description: What the step changes and which test it greens. Not
                suite-state forecasts such as exact counts, error types or
                "must stay green".
        """
        return _add_plan_step(artifacts, step_id, step_type, description)

    @tool
    @accepts_markup_override
    def add_prerequisite(
        prereq_id: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a prerequisite to the plan. Prerequisites run before TDD steps.

        Use for setup work like config files, fixtures, etc.
        The response may carry ``foreign_task_meta``, naming
        paths in the description that no lane of this task can write.

        Args:
            prereq_id: Unique prerequisite ID (e.g. "pre-1").
            description: What this prerequisite sets up.
        """
        return _add_prerequisite(artifacts, prereq_id, description)

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
    def drop_plan_file(
        path: str,
        reason: str,
    ) -> dict[str, Any]:
        """Drop a declared file from the plan, recording WHY it needed no edit.

        Use this when a file you declared was CORRECTLY declared and
        CORRECTLY needed no change — the branch delivered the work, and this
        entry simply turned out not to require an edit. The entry leaves the
        ``files`` list so the pre-merge gate's re-check can pass, while
        ``reason`` preserves why it was in scope and why no edit was needed,
        so nothing about the plan's provenance is falsified.

        Call it once per such entry. Dropping NARROWS what the pre-merge
        verify covers (the MergeRequest ``task_files`` scope), which is why
        the reason is mandatory rather than optional.

        Args:
            path: A path currently in the plan's files list.
            reason: One line: why the file was in scope, and why the branch
                legitimately needed no edit to it.
        """
        # Both parameters are flat ``str`` DELIBERATELY, not a map or a list
        # of drops. MarkupGuardMiddleware._markup_arguments skips
        # non-string values (shared/src/shared/mcp_markup_middleware.py::
        # MarkupGuardMiddleware._markup_arguments), so ONLY flat string params get inbound
        # envelope-markup protection — and ``reason`` is agent-authored free
        # prose, exactly the leak class that guard exists for. A later
        # refactor to a batch/map signature would silently drop the guard.
        return _drop_plan_file(artifacts, path, reason)

    @tool
    @accepts_markup_override
    def mark_step_done(
        step_id: str,
        commit_sha: str,
    ) -> dict[str, Any]:
        """Mark a plan step or prerequisite as done with its commit SHA.

        Use this instead of directly editing .task/plan.json.
        Only the status and commit fields are updated; all other plan
        structure (including provenance metadata) is preserved.

        *commit_sha* MUST be reachable from HEAD on the current branch — it is
        checked with a real ``git merge-base --is-ancestor`` before anything is
        written (``orchestrator/mcp/branch_probes.py::sha_branch_reachability``).
        A sha the branch does not carry is REJECTED with
        ``{'status': 'error'}`` and the step is left untouched. On rejection: COMMIT the work first, then call again
        with the sha that commit produced — never retry with a guessed or
        invented one. An ABBREVIATED sha is fine and always has been: the
        already-committed-WIP notice in your briefing shows 12-char short
        forms, ``git merge-base --is-ancestor`` resolves them like any other
        rev, and the harness prefix-matches when it dedups
        (``workflow.py::_detect_tip_wip_commits``) — so keep using the short
        sha that notice gave you. If the check cannot run at all (no usable
        git repo at the worktree) the sha is recorded anyway and a WARNING is
        logged, so an infra fault never blocks recording real work.

        Args:
            step_id: The step or prerequisite ID (e.g. "step-1", "pre-1").
            commit_sha: The git commit SHA for this step's changes. Must be
                reachable from HEAD on the current branch.
        """
        return _mark_step_done(artifacts, step_id, commit_sha)

    @tool
    @accepts_markup_override
    def record_step_deviation(
        step_id: str,
        kind: str,
        observed: str,
        expected: str,
        action_taken: str,
        prose_only: bool = False,
        note: str = '',
    ) -> dict[str, Any]:
        """Record that a plan step's text proved wrong at execution time.

        State what you observed, what the step expected, and what you did.
        Allowed on a step or prerequisite in any status, including after
        ``mark_step_done``. Never changes the step's status, commit or
        description.

        Args:
            step_id: The step or prerequisite ID (e.g. "step-1", "pre-1").
            kind: One of the closed vocabulary
                ``shared.plan_deviation.DEVIATION_KINDS``. An unknown kind is
                refused, and the refusal lists every valid kind.
            observed: What you actually found.
            expected: What the step's text said or assumed.
            action_taken: What you did about it.
            prose_only: True when the claim is destined for a docstring,
                CHANGELOG or SKILL.md.
            note: Anything else a later reader needs.
        """
        # Flat str/bool parameters DELIBERATELY, as drop_plan_file: only flat
        # strings get MarkupGuardMiddleware's inbound coverage.
        return _record_step_deviation(
            artifacts, step_id, kind, observed, expected, action_taken,
            prose_only, note,
        )

    @tool
    @accepts_markup_override
    def record_descoped_step(
        step_id: str,
        commit: str,
        disposition: str,
        authority: str,
        note: str,
    ) -> dict[str, Any]:
        """Record a done step that this revalidation drops from the plan.

        Use it when the revised plan genuinely no longer keeps a done step
        with its commit (for example after ``create_plan`` replaced the
        steps). A revalidation that drops a done step without this record is
        refused. Recording the same ``(step_id, commit)`` again replaces the
        earlier record.

        Args:
            step_id: The dropped step or prerequisite ID (e.g. "step-1").
            commit: The commit the dropped step recorded.
            disposition: One of ``orphan`` (the commit stays on the branch,
                owned by no step), ``revert`` (the commit's change is to be
                undone) or ``ride-along`` (the commit stays and ships as is).
            authority: The escalation id or human ratification marker that
                authorises the drop.
            note: Why the step no longer belongs to the plan.
        """
        # Flat str parameters DELIBERATELY, as drop_plan_file: only flat
        # strings get MarkupGuardMiddleware's inbound coverage.
        return _record_descoped_step(artifacts, step_id, commit, disposition, authority, note)

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
    def replace_plan_step(
        step_id: str,
        step_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Replace a pending step's type and description in-place.

        Preserves the step's position, status, and commit fields.
        Use during plan revalidation when a step needs revision due
        to changes on main. Cannot replace steps with status "done".
        The response may carry ``foreign_task_meta``, naming
        paths in the description that no lane of this task can write.

        Args:
            step_id: The step ID to replace (e.g. "step-2").
            step_type: New step type — either "test" or "impl". An
                out-of-vocabulary value is refused.
            description: New description of what this step does.
        """
        return _replace_plan_step(artifacts, step_id, step_type, description)

    @tool
    @accepts_markup_override
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
        and files list. The stamp records the time of the most recent confirm,
        overwritten by every re-confirm, not when the plan was first frozen.
        """
        return _confirm_plan(artifacts)

    @tool
    @accepts_markup_override
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
        # ``artifacts.lane_root`` is a SIBLING of the worktree once a meta_root
        # is supplied (`.task-meta/<name>/`), so its parent is the
        # `.task-meta` base, NOT the worktree. `artifacts.worktree` is always
        # correct, legacy or relocated (see TaskArtifacts.__init__).
        worktree = artifacts.worktree
        main_sha = branch_probes.resolve_main_sha(worktree)
        return _report_blocking_dependency(
            artifacts, depends_on_task_id, reason, main_sha,
        )

    @tool
    @accepts_markup_override
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
            commit: The full 40-character lowercase hex SHA (e.g. from
                ``git log --format=%H``) of the commit on main that contains
                this task's work; a short hash is refused. Must be reachable
                from main; the orchestrator verifies this.
            evidence: Brief description of what you found — file/symbol
                names that match this task's expected output, or why the
                commit appears to subsume this task's scope.
        """
        return _report_task_already_done(artifacts, commit, evidence)

    @tool
    @accepts_markup_override
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
                completed work, full or an unambiguous abbreviation (at
                least 7 hex characters). A branch or other ref name is
                rejected. The orchestrator resolves the branch tip itself
                and rejects the report if it does not match.
            evidence: Brief description of what you checked — how you
                established the clean fast-forward, and where the passing
                verify and review results came from.
        """
        return _report_ready_to_merge(artifacts, commit, evidence)

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
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

    @tool
    @accepts_markup_override
    def report_missing_plan_file(
        path: str,
        reason: str,
    ) -> dict[str, Any]:
        """Report an EXISTING file you must edit that the plan's ``files`` omits.

        Use this BEFORE editing such a file when the list was merely
        incomplete. You may then make the edit and keep working; there is no
        need to stop. When this session ends the orchestrator tries to take
        the file's lock and adds it to plan.json ``files``. Your next
        briefing reports it granted, or contested with the task holding it.
        If it is contested, revert your edits to it, or call
        ``escalate_blocker(category='scope_violation')`` with the design
        reason it must change.

        A file you CREATE needs no report. A file that is out of scope for a
        DESIGN reason (another owner's module, another PRD) is
        ``escalate_blocker``, not this tool.

        Args:
            path: The repo-relative path of the existing file.
            reason: One line: why this task must edit it.
        """
        return _report_missing_plan_file(artifacts, path, reason)

    return mcp, frozenset(registered)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _artifacts_from_args(argv: list[str] | None = None) -> TaskArtifacts:
    """Parse ``--worktree``/``--meta-root``/``--task-root`` and return the
    corresponding ``TaskArtifacts``.

    Mirrors workflow.py's orchestrator-side construction so both sides
    resolve the IDENTICAL roots for a given dispatch (worktree-lane-lifecycle
    PRD task ε1; the task root, task 3155). Each root must already exist;
    omitting both flags leaves the legacy ``<worktree>/.task`` single root.
    """
    return _artifacts_from_namespace(_cli_parser().parse_args(argv))


def server_from_argv(argv: list[str] | None = None) -> FastMCP:
    """The server a stdio launch with *argv* serves, over its artifacts.

    A launch declaring ``--grants-enforced`` exposes exactly its ``--grant``
    tools; one without it exposes none and logs a WARNING. The grant policy
    itself is :func:`create_server`'s.
    """
    args = _cli_parser().parse_args(argv)
    return create_server(
        _artifacts_from_namespace(args), _launch_grant(args), args.caller_role,
    )


def _launch_grant(args: argparse.Namespace) -> frozenset[str]:
    granted = frozenset(args.grant or ())
    if args.grants_enforced:
        return granted
    logger.warning(
        'plan-tools launched without --grants-enforced: exposing no tool and '
        'ignoring grant(s) %s. orchestrator.mcp_lifecycle.plan_tools_mcp_server '
        'always passes the marker.',
        sorted(granted),
    )
    return frozenset()


def _cli_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        '--task-root', type=Path, default=None,
        help=(
            'Optional task-lifetime `.task-meta/task-<id>` root. When '
            'omitted, every artifact resolves against the lane root.'
        ),
    )
    parser.add_argument(
        '--grant', action='append', metavar='NAME',
        help='A tool this launch exposes; repeatable.',
    )
    parser.add_argument(
        '--grants-enforced', action='store_true',
        help=(
            'Required: without it the server exposes no tool, whatever --grant '
            'names. With it the server exposes exactly the --grant tools.'
        ),
    )
    parser.add_argument(
        '--caller-role', default=None,
        help=(
            'Attribution only: the role this server serves, named on its '
            'markup records. Optional, because older launchers pass none.'
        ),
    )
    return parser


def _artifacts_from_namespace(args: argparse.Namespace) -> TaskArtifacts:
    artifacts = TaskArtifacts(
        args.worktree.resolve(),
        args.meta_root.resolve() if args.meta_root is not None else None,
        args.task_root.resolve() if args.task_root is not None else None,
    )
    for root in (artifacts.lane_root, artifacts.task_root):
        if not root.is_dir():
            print(f'Error: {root} does not exist', file=sys.stderr)
            sys.exit(1)
    return artifacts


def main() -> None:
    """Parse the worktree, artifact roots and grants and run the stdio MCP server."""
    server = server_from_argv()
    # show_banner=False suppresses FastMCP's decorative startup banner and its
    # synchronous PyPI update-check (check_for_newer_version -> httpx.get). That
    # check runs on the stdio startup path BEFORE the MCP initialize handshake;
    # eliminating it removes a network call that could delay startup past the
    # CLI's MCP_TIMEOUT and get this server silently dropped (task 2942).
    server.run(transport='stdio', show_banner=False)


if __name__ == '__main__':
    main()
