"""Read-time repair of envelope markup in plan.json, under a field table the caller injects.

The caller declares WHICH plan fields are prose, and which tool's schema each
validates against, as a tuple of :class:`PlanField` rows; this module walks
exactly those rows and never infers the surface. plan-tools' table is
``orchestrator/src/orchestrator/mcp/plan_tools.py::_REPAIRABLE_PLAN_FIELDS``.
``shared.toolcall_markup`` owns the literal set and every accept/refuse
decision (INV-5).
"""
from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from shared.toolcall_markup import detect_for, only_inert_html_closers, repair

from orchestrator.artifacts import Plan, TaskArtifacts

logger = logging.getLogger(__name__)


class PlanField(NamedTuple):
    """One repairable plan field, and the tool schema it must validate against.

    ``collection`` is the plan key holding a LIST OF DICTS the field lives in,
    or ``None`` for a top-level plan key.

    ``tool`` is the collection's SCHEMA OWNER: the tool whose parameter list
    ``schema_params`` is, reported verbatim as a fact's ``tool`` (contract C2).

    ``schema_params`` and ``target_keys`` are DELIBERATELY ASYMMETRIC, and the
    asymmetry is the whole point of carrying both.

    ``schema_params`` is the ORIGINATING TOOL'S FULL PARAMETER LIST. ``repair()``
    needs it complete so it can RECOGNISE a tail that names any real parameter
    of that call — a name outside the list is not a leaked argument at all and
    the candidate is refused.

    ``target_keys`` is the strictly smaller subset of those parameters that name
    a PROSE PLAN KEY a recovered value may legally be written to, mapped to the
    key to write it under, because a tool's parameter names and the plan
    document's key names can differ. A parameter absent from ``target_keys`` —
    one the plan stores under another key, or one holding no prose at all — is
    never a recovery target. Keying a recovery on the parameter name alone
    would write a JUNK KEY beside the real, still-corrupt one, or a bare ``str``
    over a non-prose value: both silent-wrong-value corruption of the plan
    artifact, the damage class this surface exists to end.

    ``also_written_by`` names the OTHER tools that write this same field, so a
    fact does not overstate its own precision. The schema owner is the call
    whose vocabulary ``repair()`` validates against, not necessarily the call
    that produced the corruption; naming it alone would send whoever triages a
    fact to the wrong call site.

    plan-tools' rows, and the cases behind each column, are
    ``orchestrator/src/orchestrator/mcp/plan_tools.py::_REPAIRABLE_PLAN_FIELDS``
    with its ``*_TARGETS`` maps and ``_DESCRIPTION_ALSO``.
    """

    collection: str | None
    field: str
    tool: str
    schema_params: tuple[str, ...]
    target_keys: Mapping[str, str]
    also_written_by: tuple[str, ...]


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
    record: PlanField,
) -> dict[str, Any] | None:
    """Repair ``holder[record.field]`` IN PLACE; return a fact, or ``None``.

    *holder* is the plan dict itself for a top-level record, or one item of
    ``plan[record.collection]`` otherwise. ``None`` means nothing to report:
    the value is absent, is not a str, carries no envelope literal at all, or
    carries only quoted HTML closers (see below).

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
    never guessed. A declined value whose every needle is a quoted HTML closer
    is no fact at all, by the same predicate the write boundary asks:
    ``shared/src/shared/toolcall_markup.py::only_inert_html_closers``.

    THE ONE ACCEPTED TRUNCATION, AND WHY IT IS ASYMMETRIC. A value ending in
    the field's OWN closer is cut at that tag with nothing recovered, and
    ``read_plan_repaired`` persists the shorter string. That is chosen, not
    overlooked — D10a in ``plans/toolcall-markup-containment-prd.md``: 212 of
    212 invisible specimens are this shape, and the fact's ``misclose`` names
    the deleted span, so the cut is reversible from the record rather than
    silent. The same empty tail under a SIBLING's closer is refused and left
    byte-identical instead, because the discriminator is evidence rather than
    breadth — this value arrived as a named parameter of a known tool, so its
    own closer is evidence about that parameter, while the sweep's
    ``scripts/sweep_toolcall_markup.py::_repair_dict`` qualifies candidates
    against every sibling KEY of the containing object, a far wider vocabulary
    in which the same shape is likelier quotation than leak.

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

    A RECOVERY MAY ONLY LAND ON A DECLARED PROSE KEY. The record distinguishes
    the TOOL's parameter vocabulary (``schema_params``) from the PLAN's key
    vocabulary (``target_keys``), and writing a recovered prose string under a
    parameter with no prose key is silent-wrong-value corruption of the plan
    artifact — so refusing conservatively there is the point, not a limitation.

    That rule is enforced through the SAME mechanism, not a second one: a
    parameter with no prose target is placed in ``supplied`` UNCONDITIONALLY,
    so ``repair()``'s existing B9 condition refuses any candidate whose tail
    recovers it and the field is left byte-identical. A parameter WITH a prose
    target has its authorship read from the key the plan actually uses
    (``holder.get(record.target_keys[name])``, never ``holder.get(name)``,
    which is always absent where the two vocabularies differ) — without which
    the fill-a-hole guard would be inert for exactly those parameters, because
    the lookup could never see the authored value it was meant to protect.
    """
    value = holder.get(record.field)
    if not isinstance(value, str):
        return None
    # PARAMETER-AWARE, and free here (task 4696). Unlike the middleware
    # boundary — which holds only the argument name and will not pay an awaited
    # get_tool to learn more — this site already has the whole schema in hand
    # off the DECLARED table, so it scans the shared enumeration widened by
    # this field's own closer AND every sibling parameter's. The diagnostic
    # below therefore names the tag actually seen instead of whatever fixed
    # literal happens to trail the leak.
    pattern = detect_for(value, record.field, record.schema_params)
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
        if only_inert_html_closers(value, record.field, record.schema_params):
            return None
        return {
            'tool': record.tool,
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
        #   ``read_plan_repaired`` would persist that blank through
        #   ``TaskArtifacts.write_plan_model`` — authored text destroyed on
        #   disk,
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
        'tool': record.tool,
        'also_written_by': list(record.also_written_by),
        'param': record.field,
        # The GATE's pattern, exactly as the unrepairable arm above publishes
        # it, so one field cannot carry two semantics depending on whether the
        # repair happened to succeed. ``result.pattern`` is the same expression
        # on the same inputs since task 5283; naming the local keeps the two
        # arms visibly identical rather than identical by coincidence.
        'pattern': pattern,
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


def _walk_repairable(
    plan: dict, fields: tuple[PlanField, ...]
) -> Iterator[tuple[dict[str, Any], int | None, PlanField]]:
    """Every ``(holder, index, record)`` the declared table addresses in *plan*.

    The surface walked is *fields* — the DECLARED table — never an inferred
    walk of the plan dict. Non-dict items in a collection are skipped,
    mirroring the ``isinstance(item, dict)`` guards the mutation helpers
    already use, so the walk stays total for adversarial plan shapes.

    Shared by the prefilter scan and the repair pass so the two can never
    disagree about WHICH fields are in scope — one enumeration, two readers.
    """
    for record in fields:
        if record.collection is None:
            yield plan, None, record
            continue
        items = plan.get(record.collection)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if isinstance(item, dict):
                yield item, index, record


def carries_markup(plan: dict, fields: tuple[PlanField, ...]) -> bool:
    """True when ANY declared field of *plan* trips :func:`detect_for`.

    The cheap prefilter over the whole document, run before anything is copied.
    Each check is ONE compiled pass over the value; a deep copy of a plan is
    tens of KB of analysis prose plus every step description, and on the
    overwhelmingly common clean path that copy would be pure waste.

    PARAMETER-AWARE AS OF TASK 4696, and this gate is why the task exists. It
    asked the param-free ``detect``, whose fixed literal set spells none of
    plan-tools' own parameter names — so a plan whose only damage was a field
    mis-closed with its OWN tag looked CLEAN, the pass never ran, and the
    corruption survived every read. Measured on 2026-08-25: 296 ``rationale``
    and 129 ``how`` specimens sitting unrepaired on disk for that reason.

    The widening costs this site NOTHING: the walk already yields the
    ``PlanField`` record, so ``record.field`` and ``record.schema_params``
    come straight off the declared table — no lookup, no second enumeration
    (INV-5). It is deliberately WIDER than the middleware's gate, which holds
    only the argument's own name; the asymmetry is a cost decision, recorded at
    ``MarkupGuardMiddleware._markup_arguments``, not a divergence.
    """
    for holder, _index, record in _walk_repairable(plan, fields):
        value = holder.get(record.field)
        if (
            isinstance(value, str)
            and detect_for(value, record.field, record.schema_params) is not None
        ):
            return True
    return False


def repair_plan_fields(
    plan: dict, fields: tuple[PlanField, ...]
) -> tuple[dict, list[dict[str, Any]]]:
    """Repair every declared prose field of *plan*; return ``(plan, facts)``.

    NEVER MUTATES THE INPUT, and never touches the filesystem. Persisting the
    result is :func:`read_plan_repaired`'s decision, not this pass's.

    THE COPY IS LAZY. A clean plan is returned AS-IS — the same object, not a
    copy — because nothing was going to change in it. Only once
    :func:`carries_markup` finds a hit does the document get deep-copied and
    re-walked, so the copy is paid on the rare corrupted read rather than on
    every plan-tools call in the fleet. The no-mutation contract is unaffected:
    on the returned-as-is path this function writes nothing at all. Callers that
    intend to mutate what they get back must therefore own a fresh document —
    which :func:`read_plan_repaired`'s callers do, since ``read_plan`` parses
    the file anew on every call.

    The prefilter runs ``detect_for`` a second time on the values that DO trip
    it (once here, once inside :func:`_repair_one_field`). That is deliberate:
    the duplicate pass is paid only on the corrupted path, where the walk is
    about to call ``repair()`` anyway, and it buys skipping the copy on the
    clean one. Both passes ask the SAME parameter-aware predicate with the SAME
    record, so the prefilter can never wave through a value the walk would then
    decline to look at, nor the reverse.
    """
    if not carries_markup(plan, fields):
        return plan, []

    repaired = copy.deepcopy(plan)
    facts: list[dict[str, Any]] = []
    for holder, index, record in _walk_repairable(repaired, fields):
        fact = _repair_one_field(holder, index, record)
        if fact is not None:
            facts.append(fact)

    return repaired, facts


#: Stable event name for a structured markup fact (INV-2). The WHOLE log message
#: is the payload: one line, one fact, ``json.loads``-able; never a value a
#: consumer has to regex out of prose. The log is a per-process stderr, so the
#: durable record of a disclosed fact is plan-tools' journal line, written by
#: ``orchestrator/src/orchestrator/mcp/stored_markup_journal.py``.
MARKUP_FACT_EVENT = 'plan_markup_repaired'

#: The one response key a host discloses this module's reportable facts under.
MARKUP_REPAIRS_KEY = 'markup_repairs'

#: Stable event name for a repair that could not be PERSISTED. Deliberately
#: distinct from the fact event: the repair itself succeeded and was served, so
#: folding the two would make a delivery failure look like a detection outcome.
MARKUP_WRITE_FAILED_EVENT = 'plan_markup_write_back_failed'

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


def forget_reported_refusals() -> None:
    """Empty this process's refusal memo: every refusal reports afresh."""
    _REPORTED_REFUSALS.clear()

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
            'event': MARKUP_FACT_EVENT,
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


def read_plan_repaired(
    artifacts: TaskArtifacts, fields: tuple[PlanField, ...]
) -> tuple[dict, list[dict[str, Any]]]:
    """Read the plan, repairing envelope-markup corruption on the way through.

    THE ONE READ PATH for every plan-tools mutation helper, under the *fields*
    table that server declares. Returns ``(plan, facts)``: the plan as the
    caller should see it, and the structured facts describing every field this read touched or refused (empty on the
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

    repaired, facts = repair_plan_fields(plan, fields)
    if not facts:
        return repaired, facts

    plan_path = artifacts.artifact_path('plan.json')
    reportable = _report_markup_facts(plan_path, repaired, facts)

    if not any(fact['outcome'] == 'repaired' for fact in facts):
        # Every fact was a refusal: nothing in the document changed, so there
        # is nothing to persist. This is what keeps a plan that legitimately
        # QUOTES the sentinels in prose (worktree 2939) byte-stable forever.
        # Decided over ALL facts, never the reported subset — a suppressed
        # repeat must not be able to flip a write decision.
        return repaired, reportable

    try:
        artifacts.write_plan_model(Plan.from_json(repaired))
    except Exception as exc:
        # Broad on purpose: the caller must receive the repaired plan whatever
        # went wrong on the way to disk. TaskArtifacts.write_plan_model can raise an
        # OSError family member (EACCES, ENOSPC, the ArtifactWriteError
        # dangling-symlink refusal) or a TypeError/ValueError out of its
        # json.dumps, so narrowing here would turn a persistence failure into
        # a failed tool call.
        logger.warning(
            json.dumps({
                'event': MARKUP_WRITE_FAILED_EVENT,
                'path': str(plan_path),
                'error': repr(exc),
                'served_from_memory': True,
                'repaired_fields': [
                    fact['field'] for fact in facts if fact['outcome'] == 'repaired'
                ],
            })
        )
    return repaired, reportable
