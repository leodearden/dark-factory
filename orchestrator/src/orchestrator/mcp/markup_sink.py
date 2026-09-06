"""The escalation channel the MCP boundary guards file their records through.

ONE mechanism, shared by every orchestrator-side registration site of
:class:`shared.mcp_markup_middleware.MarkupGuardMiddleware` (PRD
``plans/toolcall-markup-containment-prd.md``, contract C2 / INV-7).

Born as ``plan_tools``' private sink (task 4457) and promoted here by task 3690
when verdict-tools needed the identical thing. The promotion is the point: a
second, weaker preservation mechanism on a sibling boundary is precisely the
INV-5 failure this PRD exists to rule against, and the review that caught it
named the duplication before it shipped. Everything server-specific — the
attribution role, the two queue anchors, and the four sentences of prose a
human reads — arrives in a :class:`MarkupSinkSpec`; nothing here is inferred
from a tool name or a server name at call time.

``MarkupGuardMiddleware`` takes its escalation emitter as an INJECTED callable
because ``shared`` is the base layer every other package imports and so cannot
import ``escalation`` without a cycle. This module is the orchestrator-side
concrete emitter, wired at each registration site against that site's own spec.

Shape inherited from ``fused_memory.server.markup_tripwire.
emit_markup_storm_escalation``, the in-repo precedent for filing from an MCP
server process: same queue dir, same never-raises posture, same
log-and-return-None on any failure.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Where the queue lives under project_root — the same relative path
#: ``markup_tripwire._QUEUE_DIRNAME`` uses, so every guard files into ONE queue.
MARKUP_QUEUE_DIRNAME = 'data/escalations'

#: Rendered as the subject when there is no plan to read it from and the
#: worktree has no name either. Never a guess — an explicit "nobody knows".
MARKUP_UNATTRIBUTED_SUBJECT = 'unattributed'

#: The two record kinds the middleware emits through this one channel. Matched
#: by name rather than by shape: a record kind that grew a key later must not
#: silently start taking the other branch.
MARKUP_RESIDUE_ERROR_TYPE = 'mcp_markup_unrepairable'
MARKUP_STORM_ERROR_TYPE = 'mcp_markup_storm'

#: The burst alarm's OWN category and level, because the middleware's storm
#: record declares neither — unlike the residue record, which carries both and
#: whose vocabulary is therefore never re-decided here (INV-7).
#:
#: ``level=1`` matches ``markup_tripwire``'s storm record, the in-repo
#: precedent for a burst alarm filed from an MCP server process: it is an
#: operator-facing heads-up about a leak that is running now, not a hold on one
#: caller's data.
MARKUP_STORM_CATEGORY = 'mcp_markup_boundary_storm'
MARKUP_STORM_LEVEL = 1

#: Only ever used for a record the middleware grew LATER and this sink does not
#: recognise. Filing it under a visible fallback is the point: silently
#: discarding a record kind is the fail-soft this PRD exists to end.
ESCALATION_FALLBACK_CATEGORY = 'mcp_markup_residue'


@dataclass(frozen=True)
class MarkupSinkSpec:
    """Everything one registration site must DECLARE about its own channel.

    A frozen dataclass rather than a pile of keyword arguments so a new
    registration site cannot half-configure itself: every field below is
    required, so adding a server is a decision at every axis rather than an
    inheritance of plan-tools' answers.
    """

    #: The server's own name, as it appears in a human-facing sentence and in
    #: the two anchors below. ``'plan-tools'``, ``'verdict-tools'``.
    server_label: str

    #: Filed-by attribution. Names the GUARD, not the leaking agent: the
    #: middleware resolves ``agent_id`` from arguments named ``agent_id`` /
    #: ``project_root`` / ``project_id``, and no tool on either of these
    #: servers declares any of the three, so that field is structurally
    #: ``None`` here (it rides in the detail as the null it is, rather than
    #: being guessed at).
    agent_role: str

    #: The queue routing key for EVERY residue record — a NON-TASK anchor,
    #: exactly as ``markup_tripwire`` files under its own ``markup-tripwire``
    #: anchor rather than under whichever task happened to be running.
    #:
    #: THIS IS LOAD-BEARING, not cosmetic. The middleware declares residue at
    #: ``level=2``, and a pending level>=2 record carrying a LIVE task id halts
    #: that task. Preserving one payload must not cost the run that produced
    #: it: the leaking caller is told to resend from its own copy and carry on.
    #: The anchor changes only WHERE the record is filed — its level, owner and
    #: category are still the middleware's, and the subject task is not lost:
    #: it rides in the summary and the detail, which is where a human reader
    #: looks anyway.
    residue_anchor_task_id: str

    #: The dedup anchor for bursts. A burst is a property of the SERVER's
    #: rolling window, not of the task whose call happened to cross the
    #: threshold, so it is filed under a stable id of its own — which is also
    #: what makes the open-record lookup stable enough to dedup against. Kept
    #: DISTINCT from the residue anchor: a residue record is about ONE caller's
    #: payload, and folding it under the server-wide burst anchor would make
    #: the two indistinguishable to a reader.
    storm_anchor_task_id: str

    #: What the refusal COST, in one sentence, rendered into the residue
    #: record's body. Server-specific because the loss is: a refused plan-tools
    #: call writes nothing to plan.json, while a refused verdict-tools call
    #: strands a review gate and destroys a reviewer's findings list.
    refusal_consequence: str

    #: What an ACTIVE burst threatens if the guard is disabled, in one
    #: sentence, rendered into the storm alarm's body.
    storm_consequence: str

    #: WHERE an operator can identify the leaking caller for THIS boundary, as
    #: an instruction they can follow. Rendered into the storm alarm's detail
    #: and its suggested_action.
    #:
    #: REQUIRED, not defaulted, and that is load-bearing (task 4744). This field
    #: replaced a hard-coded "grep the orchestrator logs for 'markup guard:'"
    #: sentence that was measured UNFOLLOWABLE on 2026-08-25::
    #:
    #:     journalctl --user --since 2026-08-22 | grep 'markup guard:'
    #:         ->  0 plan-tools lines
    #:
    #: against 35 real plan-tools rejections in data/orchestrator/
    #: agent-transcripts/ over the same span. These servers are per-agent STDIO
    #: SUBPROCESSES whose stderr the CLI agent that spawned them consumes, so
    #: those lines reach no durable sink; an operator following that instruction
    #: correctly concluded "no evidence" and was wrong. A DEFAULT here would let
    #: the next such server silently inherit the same false sentence, which is
    #: the exact defect being fixed — so each boundary states its own answer.
    attribution_source: str

    @property
    def fallback_summary(self) -> str:
        return (
            f'Unrecognised MCP markup-guard record filed by the '
            f'{self.server_label} boundary guard — see the detail below'
        )


def resolve_project_root(worktree: Path) -> Path | None:
    """The project root whose escalation queue this server files into.

    ``--git-common-dir``, NOT ``--show-toplevel``: from a task worktree the
    toplevel is the WORKTREE, while the common dir is the MAIN checkout's
    ``.git`` — whose parent is project_root, the one queue every lane and every
    watcher already reads. Measured in a live task worktree.

    Never raises: a git failure degrades to a logged ``None``, which costs the
    operator a queued record rather than turning a decided refusal into an
    outage of its own.
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-common-dir'],
            cwd=str(worktree),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            'markup guard: could not resolve project_root from %s (%s); the '
            'residue will not be queued', worktree, exc,
        )
        return None
    common_dir = result.stdout.strip()
    if not common_dir:
        logger.warning(
            'markup guard: git named no common dir for %s; the residue will '
            'not be queued', worktree,
        )
        return None
    git_dir = Path(common_dir)
    if not git_dir.is_absolute():
        # A plain checkout answers with the relative ``.git``; a linked
        # worktree answers with the main checkout's absolute path.
        git_dir = (worktree / git_dir).resolve()
    return git_dir.parent


def resolve_subject(
    subject_task_id: Callable[[], str], worktree: Path, *, what: str
) -> str:
    """Who this record is attributed to: the site's answer, then a ladder.

    ONE ladder, shared by every channel a boundary guard emits through — the
    escalation sink below and ``markup_journal.make_fact_journal``. Kept here
    rather than reimplemented per channel because the two channels' records are
    asserted to name the SAME subject: two copies of an attribution ladder can
    drift in opposite directions on one boundary, and a guard whose escalation
    and whose journal disagree about who leaked is worse than either alone.
    That sibling duplication is the INV-5 shape this module's own header rules
    against.

    The rungs, in order: the site's own answer (plan.json's ``task_id`` for
    plan-tools), then the worktree directory NAME (a task worktree is named for
    its task), then an explicit ``MARKUP_UNATTRIBUTED_SUBJECT`` — never ``None``
    and never absent, because a consumer must be able to tell "nobody knows"
    from "that emitter forgot the key".

    A RAISING thunk falls to the same ladder rather than costing the record.
    Attribution is what these channels are FOR, but a record that says
    "something leaked from add_design_decision at 16:52:34" is still strictly
    more than the nothing that was written before they existed.

    *what* names the record kind in the fallback log line, so an operator
    reading the warning knows which channel degraded.
    """
    try:
        resolved = subject_task_id()
    except Exception:
        logger.warning(
            'markup guard: could not attribute the %s under %s; falling back '
            'to the worktree name', what, worktree,
        )
        resolved = ''
    return resolved or worktree.name or MARKUP_UNATTRIBUTED_SUBJECT


def open_escalation_channel(project_root: Path) -> tuple[Any, Any] | None:
    """Open ``(Escalation, EscalationQueue)`` for *project_root*, or ``None``.

    THE IMPORT IS LAZY, and deliberately not hoisted to module scope. MCP
    server startup latency is load-bearing: the orchestrator spawns one of
    these per agent invocation over stdio, and an import stall past
    ``MCP_TIMEOUT`` gets the server silently dropped, producing 0-turn
    ``error_empty_output`` failures (tasks 1775 / 1776 / 2942). This path runs
    only on the measured 0.27% of calls that carry a leak, so it pays for
    itself the first time it is needed and never before.

    An ImportError degrades to a logged ``None`` exactly as
    ``markup_tripwire``'s ``HAS_ESCALATION`` guard does — ``escalation`` is a
    workspace dependency of orchestrator, so this is a defensive floor rather
    than an expected branch.
    """
    try:
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue
    except ImportError:
        logger.warning(
            'markup guard: the escalation package is unavailable, so the '
            'residue of a refused call will not be preserved anywhere',
        )
        return None
    try:
        return Escalation, EscalationQueue(project_root / MARKUP_QUEUE_DIRNAME)
    except Exception:
        logger.exception(
            'markup guard: failed to open the escalation queue under %s', project_root,
        )
        return None


def residue_detail(
    record: Mapping[str, Any], subject_task_id: str, spec: MarkupSinkSpec
) -> str:
    """The residue record's body — flat fields, then the payload VERBATIM.

    The raw value is rendered last, unquoted and unescaped, because it is the
    ONLY surviving copy of data the caller may never be able to resend: the
    call was refused, so none of it reached the server's store. Everything
    above it is diagnostic and is written with ``!r`` so an empty or ``None``
    field reads as itself rather than as a blank line.

    ``subject_task_id`` is rendered here BECAUSE the record's own ``task_id``
    field is the non-gating anchor — this is the only place a reader learns
    whose call leaked. ``owner`` is rendered for the same class of reason: the
    middleware declares it as INV-7's machine-readable owner and
    :class:`escalation.models.Escalation` has no field for it, so rendering it
    is what keeps a declared contract field from being silently dropped at the
    one sink whose whole argument is against silent drops.
    """
    return '\n'.join([
        f'subject_task_id={subject_task_id!r}',
        f'owner={record.get("owner")!r}',
        f'tool={record.get("tool")!r}',
        f'field={record.get("field")!r}',
        f'matched_pattern={record.get("matched_pattern")!r}',
        f'agent_id={record.get("agent_id")!r}',
        f'project={record.get("project")!r}',
        '',
        f'The {spec.server_label} MCP boundary guard refused this tool call: '
        'the value below carries raw tool-call envelope markup whose own '
        'boundary cannot be determined, so no repair was attempted. '
        + spec.refusal_consequence +
        ' Guessing would have silently dropped whatever arguments hide in the '
        'residue.',
        '',
        'Recover the payload below for the caller if it is still needed, then '
        'chase the harness serialization leak that produced it against '
        'plans/toolcall-markup-containment-prd.md.',
        '',
        'RAW PAYLOAD (verbatim, the only surviving copy):',
        str(record.get('raw_value')),
    ])


def storm_detail(record: Mapping[str, Any], spec: MarkupSinkSpec) -> str:
    """The burst alarm's body — the window's own numbers, then what to do.

    Points at ``plans/toolcall-markup-containment-prd.md`` and NOT at DF 3083,
    which is done and CLOSED to appends: nothing reads what is attached there.

    The "identify the leaking caller" sentence is the SPEC's
    (:attr:`MarkupSinkSpec.attribution_source`), not this module's, because the
    honest answer differs per boundary — and a wrong one here is worse than
    none, since a reader who follows it concludes "no evidence" rather than
    "I looked in the wrong place".
    """
    return '\n'.join([
        f'count={record.get("count")!r}',
        f'threshold={record.get("threshold")!r}',
        f'window_seconds={record.get("window_seconds")!r}',
        f'outcome={record.get("outcome")!r}',
        f'project={record.get("project")!r}',
        '',
        f'The {spec.server_label} MCP boundary guard saw multiple tool calls '
        'carrying raw tool-call envelope markup inside one rolling window. A '
        'BURST means the upstream harness serialization leak is ACTIVE right '
        'now, not that the guard is misfiring — do NOT disable it, or '
        + spec.storm_consequence,
        '',
        'Identify the leaking caller: ' + spec.attribution_source,
        '',
        'Then report it against plans/toolcall-markup-containment-prd.md.',
    ])


def file_residue(
    escalation_cls: Any,
    queue: Any,
    worktree: Path,
    subject_task_id: str,
    record: Mapping[str, Any],
    spec: MarkupSinkSpec,
) -> str | None:
    """File one residue record. NEVER deduped, and that is the point.

    Each record is the ONLY surviving copy of a DIFFERENT caller payload, so
    folding two of them into one open escalation would destroy one. The storm
    anchor's dedup exists because a burst is the same alarm twice; this is the
    opposite case, and the open-record lookup is not even performed.

    Filed under the residue ANCHOR, never under the leaking task's own id: at
    the ``level=2`` the middleware declares, a pending record carrying a live
    task id halts that task. See :attr:`MarkupSinkSpec.residue_anchor_task_id`.
    """
    esc = escalation_cls(
        id=queue.make_id(spec.residue_anchor_task_id),
        task_id=spec.residue_anchor_task_id,
        agent_role=spec.agent_role,
        # 'blocking' even for the level-2 record the middleware declares, and
        # the reason is the SENTINEL-ROLE rule, not squeamishness: a born-at-L2
        # severity (`escalation.models.BORN_AT_L2_SEVERITIES`) is legal only for
        # a role in `escalation.server._HARNESS_SENTINEL_ROLE_PREFIXES`, which
        # `_chokepoint_or_submit` enforces by downgrading and `submit.py` by
        # rejecting. Every spec on this module's callers files under its own
        # server-named role (`plan-tools-markup-guard`,
        # `verdict-tools-markup-guard`) — deliberately, so a reader can tell
        # WHICH boundary leaked — and none of those is a sentinel, so 'blocking'
        # is the only severity these records may legally carry.
        #
        # The guard registered INSIDE the escalation server files the same
        # record class at 'critical' (`escalation.server._MARKUP_RESIDUE_SEVERITY`)
        # because it files as `harness-markup-guard`, which IS a sentinel. That
        # is one rule resolving two ways, not two sites disagreeing; neither may
        # be "made to match" the other by changing the severity alone, because
        # the severity follows the role. Both sites reach the queue through a
        # direct `submit` that bypasses the gate, so each honours the rule by
        # construction rather than by being checked.
        severity='blocking',
        # The middleware OWNS this vocabulary (INV-7). A second site re-deciding
        # a record's category or level is exactly the two-mechanisms-on-one-
        # boundary failure this module exists to rule against.
        category=record.get('category', ESCALATION_FALLBACK_CATEGORY),
        # The subject is PREFIXED rather than folded into the middleware's own
        # sentence: the queue's task_id is the anchor, so a reader scanning
        # summaries alone would otherwise have no way to tell whose call leaked.
        summary=f'[{subject_task_id}] ' + record.get('summary', spec.fallback_summary),
        suggested_action=record.get('suggested_action', ''),
        detail=residue_detail(record, subject_task_id, spec),
        worktree=str(worktree),
        level=record.get('level', 0),
    )
    return queue.submit(esc)


def file_storm(
    escalation_cls: Any,
    queue: Any,
    worktree: Path,
    record: Mapping[str, Any],
    spec: MarkupSinkSpec,
) -> str | None:
    """File one burst alarm, folding into an already-open one if there is one.

    The anchor dedup matters beyond the counter's own per-window rate limit: a
    leak running for hours would otherwise file one escalation per window, so
    those collapse into the single open record until an operator resolves it.

    A queue READ failure falls THROUGH to filing rather than bailing out —
    losing duplicate suppression is strictly better than losing the alarm for
    an actively running leak. Same posture ``markup_tripwire`` takes.
    """
    try:
        existing = queue.get_by_task(spec.storm_anchor_task_id, status='pending')
    except Exception:
        logger.exception(
            'markup guard: could not check for an open burst alarm; filing a '
            'new one rather than losing it',
        )
        existing = []
    open_alarms = [
        esc for esc in existing
        if getattr(esc, 'category', None) == MARKUP_STORM_CATEGORY
    ]
    if open_alarms:
        logger.info(
            'markup guard: %s is already open for this burst (%r now); not '
            'filing a duplicate', open_alarms[0].id, record,
        )
        return open_alarms[0].id

    esc = escalation_cls(
        id=queue.make_id(spec.storm_anchor_task_id),
        task_id=spec.storm_anchor_task_id,
        agent_role=spec.agent_role,
        severity='blocking',
        category=MARKUP_STORM_CATEGORY,
        summary=(
            f'{record.get("count")} {spec.server_label} tool call(s) '
            f'{record.get("outcome")} for leaked envelope markup in '
            f'{record.get("window_seconds")}s — the serialization leak is '
            'ACTIVE (see plans/toolcall-markup-containment-prd.md)'
        ),
        detail=storm_detail(record, spec),
        suggested_action=(
            'identify the leaking caller — ' + spec.attribution_source +
            ' — and report it against plans/toolcall-markup-containment-prd.md '
            '(DF task 3083 is done and closed to appends)'
        ),
        worktree=str(worktree),
        level=MARKUP_STORM_LEVEL,
    )
    esc_id = queue.submit(esc)
    logger.warning('markup guard: queued %s for the burst %r', esc_id, record)
    return esc_id


def make_escalation_sink(
    *,
    worktree: Path,
    spec: MarkupSinkSpec,
    subject_task_id: Callable[[], str],
    resolve_root: Callable[[Path], Path | None] = resolve_project_root,
    open_channel: Callable[[Path], tuple[Any, Any] | None] = open_escalation_channel,
    last_resort: Callable[[dict[str, Any]], str | None] | None = None,
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Build the emitter a boundary guard files its records through.

    Returns the id of the queued record, which the middleware folds into the
    caller-facing refusal so the payload can be looked up — or ``None`` when
    filing was impossible AND no *last_resort* landed either.

    *subject_task_id* is a THUNK, not a value: the middleware cannot resolve
    the leaking task (``MarkupGuardMiddleware._identity`` reads its identity
    off the call's own arguments, and no tool on these servers declares
    ``agent_id`` / ``project_root`` / ``project_id``), and the attribution a
    concrete site can offer may not exist yet at ``create_server`` time — a
    plan-tools ``create_plan`` refused before any plan exists is the case. So
    it is evaluated per record, on the worker thread, never at wiring time.

    *last_resort* is the FLOOR under a queue that cannot be opened at all — an
    unwritable project root, a missing escalation package. It receives the same
    record and returns whatever locator it managed to write, or ``None``. It is
    deliberately not the primary channel: a locator inside a task worktree dies
    with that worktree at ``git worktree remove --force``, so it preserves the
    payload only until teardown and nothing ever proactively reads it. That is
    strictly better than losing the payload immediately and strictly worse than
    a queued escalation, which is exactly what a last resort should be.

    ASYNC, with every blocking step on a worker thread. The middleware calls
    this from inside the server's event loop, and one record can run a
    ``git rev-parse`` subprocess, open a queue directory, take a cross-process
    file lock and do two fsync'd writes.

    NEVER RAISES. The middleware's own ``_call_sink`` already contains a sink
    that throws, but this keeps its own contract so it stays safe under any
    future caller: the call's outcome is decided before either channel runs, so
    a queue outage must cost an operator visibility, never turn a working guard
    into an outage of its own.
    """
    #: Both memos hold SUCCESSES ONLY. A failed git resolution or a failed
    #: queue open is TRANSIENT by nature (a fork failure under load, an EINTR,
    #: a timeout, an index.lock storm), and caching one would permanently
    #: disable residue preservation for every later refused call on this
    #: server — losing the only surviving copy of each of their payloads, which
    #: is the exact data loss this sink exists to prevent.
    project_root: Path | None = None
    channel: tuple[Any, Any] | None = None

    def fall_back(record: dict[str, Any], why: str) -> str | None:
        if last_resort is None:
            return None
        try:
            locator = last_resort(record)
        except Exception:
            logger.exception(
                'markup guard: the last-resort residue writer failed after %s; '
                'the payload for %s.%s is lost',
                why, record.get('tool'), record.get('field'),
            )
            return None
        if locator:
            logger.warning(
                'markup guard: %s, so the payload for %s.%s was written to the '
                'worktree-local %s instead — it dies with this worktree, so '
                'recover it before the lane is reaped',
                why, record.get('tool'), record.get('field'), locator,
            )
        return locator

    def file_record(record: dict[str, Any]) -> str | None:
        """The blocking body, run on a worker thread."""
        nonlocal project_root, channel
        if project_root is None:
            project_root = resolve_root(worktree)
            if project_root is None:
                return fall_back(record, 'the project root could not be resolved')
        if channel is None:
            # Memoised alongside project_root so a queue directory is opened
            # (and mkdir'd) once per server rather than once per record.
            channel = open_channel(project_root)
            if channel is None:
                return fall_back(record, 'the escalation queue could not be opened')
        escalation_cls, queue = channel

        error_type = record.get('error_type')
        try:
            if error_type == MARKUP_STORM_ERROR_TYPE:
                return file_storm(escalation_cls, queue, worktree, record, spec)
            if error_type != MARKUP_RESIDUE_ERROR_TYPE:
                # A kind the middleware grew later. FILE IT rather than
                # dropping it: silently discarding a record kind is the
                # fail-soft this PRD exists to end, and an unfamiliar record in
                # the queue is a question an operator can answer.
                logger.warning(
                    'markup guard: filing an unrecognised record kind %r as '
                    'residue — the middleware emits something this sink does '
                    'not know about', error_type,
                )
            return file_residue(
                escalation_cls, queue, worktree, _subject(), record, spec,
            )
        except Exception:
            logger.exception(
                'markup guard: failed to file the %r record for %s.%s; the '
                'refusal stands',
                record.get('error_type'), record.get('tool'), record.get('field'),
            )
            return fall_back(record, 'the escalation could not be submitted')

    def _subject() -> str:
        return resolve_subject(subject_task_id, worktree, what='record')

    async def sink(record: dict[str, Any]) -> str | None:
        try:
            return await asyncio.to_thread(file_record, record)
        except Exception:
            # ``file_record`` already contains every filing failure; this is the
            # floor under the thread hop itself, so the never-raises contract
            # holds for the whole emitter and not merely for its body.
            logger.exception(
                'markup guard: could not run the escalation sink for %r; the '
                'refusal stands', record.get('error_type'),
            )
            return None

    return sink
