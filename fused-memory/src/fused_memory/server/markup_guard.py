"""Attach beta's markup guard to fused-memory's BUNDLED FastMCP (task 4458).

PRD ``plans/toolcall-markup-containment-prd.md``, leaf gamma-3. Task 3689
(beta) delivered :class:`shared.mcp_markup_middleware.MarkupGuardMiddleware` as
a :class:`fastmcp.server.middleware.Middleware`, attached with
``mcp.add_middleware(...)``. fused-memory does not run that FastMCP.

fused-memory's server is the FastMCP BUNDLED inside the ``mcp`` SDK, which has
no ``add_middleware``. :mod:`fused_memory.server.tool_middleware_bridge` runs the
middleware UNMODIFIED at the bundled ``ToolManager.call_tool`` chokepoint, and
says why a rejection RAISES ``ToolError`` and must sit outside
``main._install_safe_tool_wrapper``. INV-5: no second implementation of
detection, repair, policy or payload shape lives here, and nothing is imported
from the middleware but its public names.

Because argument validation runs inside ``Tool.run``, after that chokepoint, a
leak that swallowed a REQUIRED parameter is repaired before pydantic ever sees
the call. Unguarded, that same call fails with a bare ``Field required``; behind
the guard it is rejected with the repair attached.

## Measured parity with the write-time gate this replaces

Before the four in-line ``_markup_gate`` call sites in ``tools.py`` were
retired, this boundary was driven over the retiring gate's ENTIRE scanned
surface — its three hand-authored specimens against ``{'content'}`` on
add_memory / add_episode and ``{title, description, details, prompt}`` on
submit_task / update_task, 30 combinations. All 30 were rejected here, none
reached the service or the interceptor, and each named the same matched literal
the gate named. There was no gap to close.

The rejections use two vocabularies, both this guard's: ``mcp_markup_detected``
carries a repair, ``mcp_markup_unrepairable`` refuses without one. The split is
:func:`shared.toolcall_markup.repair`'s NO SILENT PARTIAL REPAIR contract, not
a hole — ``\x3c/invoke>`` closes the ENVELOPE rather than a parameter, and an
opener with no preceding mis-close leaves the repairer nothing to attribute the
residue to. Either way nothing is written and the residue is escalated rather
than dropped. ``tests/server/test_markup_tripwire_gate.py`` holds the
expectation table and the per-specimen reasoning.

The envelope closer above is spelled with the ``\\x3c`` escape, never a raw
bracket: the "Sentinel-literal hazard" section of
``shared/src/shared/toolcall_markup.py``, the owner of this convention, says
why.
"""

from __future__ import annotations

import logging
from typing import Any

from shared.mcp_markup_middleware import (
    MARKUP_RECURRENCE_ROUTING,
    MarkupGuardMiddleware,
    RepairPolicy,
)

from fused_memory.server.markup_tripwire import (
    _MARKUP_STORM_THRESHOLD,
    _MARKUP_STORM_WINDOW_SECONDS,
    emit_markup_residue_escalation,
    emit_markup_storm_escalation,
)
from fused_memory.server.tool_middleware_bridge import install_tool_middleware

logger = logging.getLogger(__name__)


#: Tools DECLARED to carry envelope literals as ordinary DATA, and therefore not
#: scanned (INV-1: a declared, machine-checkable opt-out, never a heuristic).
#:
#: Names are BARE in-server names. ``context.message.name`` is
#: ``scan_memory_content``, not the agent-facing
#: ``mcp__fused-memory__scan_memory_content`` that the specimen corpus records,
#: so an exemption declared with the prefixed spelling would silently never
#: match — and a machine-checked declaration that fails open is worse than none.
#:
#: One name, with the reason it is here:
#:
#: * ``scan_memory_content`` — PRD boundary row B7. Its whole job is scanning the
#:   corpus for exactly these substrings, so guarding it would make the
#:   retroactive-sweep tool unable to look for the thing it was built to find.
#:   It is the sanctioned tool for that lookup precisely because the fragments
#:   carry almost no semantic signal (a live 2026-07-26 semantic probe for the
#:   known corrupted records returned ZERO), so there is no other path to the
#:   same answer. Load-bearing on a MEASURED shape, not a precaution: ``needles``
#:   is declared ``list[str] | None`` and the bundled dispatcher accepts it as a
#:   JSON STRING and coerces it, which makes it a scanned string argument — and
#:   without this exemption a sweep for the envelope literals is refused with
#:   ``mcp_markup_unrepairable``.
#:
#: Surveyed for other legitimate carriers and deliberately NOT exempted:
#: ``redact_episode_content``, whose ``new_content`` is the CLEAN replacement
#: text for a corrupted episode — markup there is a defect, so rejecting it is
#: correct; and ``search``, whose ``query`` is not a sanctioned carrier because
#: ``scan_memory_content`` is explicitly the tool for literal substring lookup.
EXEMPT_TOOLS: frozenset[str] = frozenset({'scan_memory_content'})


#: The anchor this guard files bursts under. Deliberately NOT markup_tripwire's
#: ``markup-tripwire``: the L1 escalation watcher files its own cluster records
#: under that id and SQUATS it — measured, the tripwire filed nothing
#: 2026-08-16..2026-08-19 while 41 rejections occurred, and all 17 records sat at
#: dedupe_count 0. Deduping against a squatted anchor is suppression that reads
#: as calm, so this guard needs an anchor of its own.
_STORM_ANCHOR_TASK_ID = 'markup-guard'

#: The one record kind the storm filer understands. The middleware sends its
#: per-call RESIDUE records down the same channel, and those are not bursts.
_STORM_ERROR_TYPE = 'mcp_markup_storm'

#: The anchor the per-call RESIDUE records file under — a third series,
#: distinct from both storm anchors because it is a different record KIND: it
#: holds one caller's payload rather than summarising a burst, and it is never
#: deduped (folding two residue records together would destroy one payload).
_RESIDUE_ANCHOR_TASK_ID = 'markup-residue'

_MARKUP_GUARDED_SENTINEL = '_fused_memory_markup_guarded'


def _resolve_project_root(project: Any, known_projects: dict[str, str] | None) -> str | None:
    """Turn the middleware's ``project`` label into a filesystem project_root.

    ``MarkupGuardMiddleware._identity`` resolves ``project_root`` FIRST then
    ``project_id``, so the label arrives in one of two shapes depending on which
    tool leaked: the task tools carry a ``project_root`` and yield a PATH, while
    the memory tools carry only a ``project_id`` and yield a BARE ID like
    ``dark_factory``. Filing against a bare id would open a queue at a relative
    directory named after the project.

    So an absolute path resolves to itself, and anything else is looked up in
    *known_projects* — the same ``_kp.get(project_id)`` translation the
    write-time gate does at its four call sites.

    Either way the answer must be a root the REGISTRY vouches for. The label is
    the caller's RAW ``project_root`` argument, read at the dispatch boundary
    before ``submit_task``/``update_task`` normalise it, and
    ``EscalationQueue.__init__`` does ``mkdir(parents=True)`` — so accepting an
    arbitrary absolute string would let any caller create a
    ``<anything>/data/escalations/`` tree on this host simply by leaking markup
    often enough to fire a burst. The retired ``_markup_gate`` never had that
    exposure: it saw either an already-normalized ``project_root`` or a
    ``_kp.get(project_id)`` lookup, never a raw one.

    An unresolvable project yields ``None`` and files NOTHING. An escalation
    filed against a guessed default is worse than one an operator has to place
    by hand — and an escalation filed against a caller-CHOSEN default is worse
    than both.
    """
    if not isinstance(project, str) or not project:
        return None
    if project.startswith('/') and project in (known_projects or {}).values():
        return project
    if known_projects:
        return known_projects.get(project)
    return None


def install_markup_guard(
    mcp: Any,
    *,
    policy: RepairPolicy,
    known_projects: dict[str, str] | None = None,
    exempt_tools: frozenset[str] = EXEMPT_TOOLS,
) -> None:
    """Wrap ``mcp._tool_manager.call_tool`` with the shared markup guard.

    *policy* is REQUIRED and keyword-only: the declared tier for this server
    belongs at the interception point, never inferred per call from the shape of
    the damage or from a tool's name (INV-1).

    *known_projects* is the ``project_id -> project_root`` registry the storm
    escalation sink uses to place a burst in the right queue. Accepted here and
    threaded to the sink; the guard's rejection behaviour does not depend on it.

    *exempt_tools* names the tools allowed to carry envelope literals as data.

    Installed through
    :func:`fused_memory.server.tool_middleware_bridge.install_tool_middleware`,
    so it covers every registered tool, fails open on its own defects, and is
    idempotent: re-entry leaves the existing wrapping in place, and a second
    guard would count every rejection twice into the storm window.

    Raises:
        ValueError: for any *policy* but ``REJECT_WITH_REPAIR``. Not a
            preference — ``MarkupGuardMiddleware._forward`` returns a fastmcp
            ``ToolResult`` that the bundled dispatcher does not understand, so
            ``FORWARD_REPAIR`` is genuinely unsupported at this site. A
            declarable tier that cannot work must fail loudly at wiring time
            rather than silently at the first repair.
    """
    if policy is not RepairPolicy.REJECT_WITH_REPAIR:
        raise ValueError(
            f'markup guard policy {policy!r} is not supported on the bundled '
            f'FastMCP: MarkupGuardMiddleware._forward returns a fastmcp '
            f'ToolResult that mcp.server.fastmcp\'s dispatcher cannot consume. '
            f'Only {RepairPolicy.REJECT_WITH_REPAIR!r} is installable here.'
        )

    def _file_residue(record: dict[str, Any]) -> str | None:
        """Preserve one refused payload durably, and report where it went.

        The full record — which carries the caller's ``raw_value`` verbatim — is
        logged ONLY when it could not be filed. Logging it on every refusal would
        put the whole payload in the log on every call of the commonest rejection
        shape, i.e. maximum log volume exactly when a leak is bursting, while the
        queued record already holds it.
        """
        project_root = _resolve_project_root(record.get('project'), known_projects)
        escalation_id = None
        if project_root is not None:
            # Purely additive, same as the storm branch below: the refusal is
            # already decided, and emit_markup_residue_escalation is itself built
            # never to raise, so this guards only against a defect in it.
            try:
                escalation_id = emit_markup_residue_escalation(
                    project_root, record, anchor_task_id=_RESIDUE_ANCHOR_TASK_ID
                )
            except Exception:
                logger.exception(
                    'markup_guard: emit_markup_residue_escalation raised for '
                    'project_root=%r; residue of %r not filed',
                    project_root, record.get('tool'),
                )
        if escalation_id is not None:
            logger.error(
                'markup_guard_residue: refused an unrepairable %s.%s and '
                'preserved its payload in %s (%d chars); %s',
                record.get('tool'), record.get('field'), escalation_id,
                len(record.get('raw_value') or ''), MARKUP_RECURRENCE_ROUTING,
            )
            return escalation_id
        logger.error(
            'markup_guard_residue: refused an unrepairable call and could NOT '
            'file its payload (project=%r), so THIS LOG LINE is the only copy '
            'and the caller was handed a null escalation_id: %r; %s',
            record.get('project'), record, MARKUP_RECURRENCE_ROUTING,
        )
        return None

    def _escalation_sink(record: dict[str, Any]) -> str | None:
        """File a fired burst, and never change the rejection that fired it.

        ONE sink serves TWO record kinds — measured, not assumed. The middleware
        routes both its storm summaries (``error_type='mcp_markup_storm'``:
        count, threshold, window_seconds, outcome, project) AND its per-call
        RESIDUE records (``mcp_markup_unrepairable``, carrying the ``raw_value``
        the caller lost) through this one injected callable.

        Each kind has its OWN filer and its own anchor, because they are not the
        same record: a storm summarises a burst and is DEDUPED into one open
        record, while a residue record HOLDS one caller's payload and must never
        be folded into another's. Before this dispatch existed, a residue record
        went to the storm filer, which filed a rejected_writes_in_window=None
        escalation under the STORM anchor and logged 'None None outcome(s) in
        Nones'.

        The residue id is RETURNED, and that return value is load-bearing: the
        middleware folds it into the refusal as ``escalation_id`` beside a hint
        telling the caller its payload is preserved in the escalation named
        there. A sink that returns nothing makes that sentence a lie — and
        unrepairable is the COMMON outcome for the real corpus shapes, not an
        edge case. When filing is genuinely impossible (an unresolvable project,
        no escalation package, a queue outage) the caller does see a null id, and
        THEN — only then — the full record is dumped to the log, because the log
        line really is the only copy.

        The middleware delegates dedup here on purpose — whether an open
        escalation already exists is queue knowledge the shared layer does not
        have and must not guess at.
        """
        if record.get('error_type') != _STORM_ERROR_TYPE:
            return _file_residue(record)

        project_root = _resolve_project_root(record.get('project'), known_projects)
        if project_root is None:
            logger.error(
                'markup_guard_storm: %s %s outcome(s) in %ss but project=%r could '
                'not be resolved to a project_root, so NO escalation was filed — '
                'the serialization leak is ACTIVE; %s',
                record.get('count'), record.get('outcome'),
                record.get('window_seconds'), record.get('project'), MARKUP_RECURRENCE_ROUTING,
            )
            return None

        # ONE greppable operator-facing ERROR line. The storm summary folded
        # into the tool response reaches only the leaking caller — the one party
        # that already knows — so the operator's copy cannot ride on it. Naming
        # the count here is also the visible-counter half of the remedy, without
        # an escalation-schema change.
        logger.error(
            'markup_guard_storm: %s %s outcome(s) in %ss for project_root=%r '
            '(threshold=%s) — the serialization leak is ACTIVE; %s',
            record.get('count'), record.get('outcome'), record.get('window_seconds'),
            project_root, record.get('threshold'), MARKUP_RECURRENCE_ROUTING,
        )

        # Purely additive: the rejection is already decided, so a queue failure
        # costs the operator a heads-up and nothing else. Same reasoning the
        # retiring _markup_gate applies to its own emitter — and
        # emit_markup_storm_escalation is itself built never to raise, so this
        # guards only against a defect in it.
        try:
            emit_markup_storm_escalation(
                project_root, record, anchor_task_id=_STORM_ANCHOR_TASK_ID
            )
        except Exception:
            logger.exception(
                'markup_guard: emit_markup_storm_escalation raised for '
                'project_root=%r; storm %r not escalated',
                project_root, record,
            )
        # The storm id is deliberately not reported back: the middleware folds a
        # returned id into the REFUSAL payload as the record holding that
        # caller's data, and a burst summary holds no payload.
        return None

    guard = MarkupGuardMiddleware(
        policy,
        exempt_tools,
        escalation_sink=_escalation_sink,
        # ONE source for the tuning across both guards, so the in-line gate and
        # the boundary cannot drift to different thresholds while both are live.
        storm_threshold=_MARKUP_STORM_THRESHOLD,
        storm_window_seconds=_MARKUP_STORM_WINDOW_SECONDS,
        # fact_sink is deliberately left unwired: the middleware already emits
        # the structured fact unconditionally as a WARNING (INV-2), and
        # fused-memory has no queue-backed CONSUMER for the stream. This task's
        # consumers are the calling agent (served by repaired_call) and the
        # operator (served by the storm escalation above). Left explicit so a
        # future consumer wires it deliberately rather than inheriting a guess.
        fact_sink=None,
    )
    install_tool_middleware(
        mcp, guard, sentinel=_MARKUP_GUARDED_SENTINEL, label='markup guard'
    )
    logger.info(
        'markup boundary guard installed at ToolManager.call_tool '
        '(policy=%s, exempt=%s)',
        policy, sorted(exempt_tools),
    )
