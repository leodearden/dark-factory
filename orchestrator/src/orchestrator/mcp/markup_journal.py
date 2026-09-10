"""The DURABLE journal a boundary guard's per-call markup facts land in.

Task 4744. ``shared.mcp_markup_middleware.MarkupGuardMiddleware`` emits one
``markup_detected`` fact on EVERY outcome (unrepairable, rejected, repaired),
and that record is the only thing anywhere that names WHICH call leaked. On an
orchestrator MCP boundary it reached exactly one place: a ``logger.warning``
inside a per-agent STDIO SUBPROCESS whose stderr the CLI agent that spawned it
consumes. Measured 2026-08-25::

    journalctl --user --since 2026-08-22 | grep 'markup guard:'   ->  0 lines

against 35 REAL plan-tools rejections in ``data/orchestrator/agent-transcripts/``
over the same span. The contrast is not neglect: fused-memory's identical
middleware DOES reach journald because that server runs under
``systemd --user``, while plan-tools does not run under anything.

The consequence is what this module exists to end. A markup STORM files an
escalation telling an operator to identify the leaking caller from the guard's
own log lines — an instruction that cannot be followed, because the lines were
written to a stream nobody retains. Anyone following it correctly concludes "no
evidence" and is wrong.

So the fact channel gets a consumer: one append-only JSONL under
``<project_root>/data/orchestrator/markup-guard/<server_label>.jsonl``, one line
per event, carrying the identity the storm summary structurally cannot.

WHAT IS AND IS NOT IN A LINE. Names, identity and the matched pattern — never
the raw caller payload. The verbatim payload already has exactly one owner, the
residue escalation (``markup_sink.residue_detail``), which is the only surviving
copy of data the caller may never be able to resend. A second, weaker copy of it
here would be the INV-5 duplication this PRD rules against, and would put an
unbounded caller payload inside a file whose concurrency safety depends on a
bounded write. This journal answers "who leaked, from which tool, into which
parameter, when" — the question the storm record asks a human to answer and
gives them no way to.

Shape inherited from ``markup_sink.make_escalation_sink``: never raises, does
its blocking work under ``asyncio.to_thread``, memoizes successes only, returns
a locator string or ``None``. The two injected channels on one boundary behave
identically under failure, so a reader who has read one has read both.

PRD: ``plans/toolcall-markup-containment-prd.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator.mcp import markup_sink

logger = logging.getLogger(__name__)

#: Where the journal lives under project_root. Beside
#: ``data/orchestrator/agent-transcripts``, in the tree an operator already
#: greps, and under the repo-root-gitignored ``/data/`` so a journal line can
#: never contaminate a diff or a verify.
#:
#: THIS CONSTANT IS THE SINGLE OWNER of the path an operator is told to read:
#: the storm escalation's ``attribution_source`` is composed from it rather than
#: respelling it, so the instruction and the artifact cannot drift apart.
MARKUP_JOURNAL_DIRNAME = 'data/orchestrator/markup-guard'

#: The opening angle bracket and the JSON unicode escape it is written as.
#:
#: Built from ``chr()`` rather than written verbatim for the reason
#: ``shared/src/shared/toolcall_markup.py`` records: this module's subject IS
#: envelope markup, so its own bytes stay free of anything a later edit could
#: grow into a literal by appending one character.
_LT = chr(60)
_LT_JSON_ESCAPE = '\\u003c'

#: THE SIZE BOUNDS, which are what make the O_APPEND atomicity premise sound
#: rather than assumed. A single ``os.write`` to an append-only fd is atomic
#: only while it is small; the payloads this guard sees reach tens of KB, so a
#: record is capped rather than trusted.
#:
#: 200 chars per string field mirrors ``build_markup_block``'s existing
#: ``content_excerpt`` convention — this journal answers "who leaked, from which
#: tool, into which parameter", and every one of those answers fits easily. The
#: verbatim payload is deliberately NOT this file's job: it has exactly one
#: owner, the residue escalation.
MARKUP_JOURNAL_MAX_FIELD_CHARS = 200

#: Bounds ``recovered_params`` — the one list field the middleware emits, whose
#: length is a property of the LEAK (how many sibling parameters it ate) and so
#: is not bounded by anything upstream.
MARKUP_JOURNAL_MAX_LIST_ITEMS = 20

#: The line's own ceiling, well under the 4096-byte PIPE_BUF floor times two and
#: far under any filesystem's atomic-append size. Protects the concurrent
#: append: a record that would exceed it is degraded to the identity-only floor
#: below rather than written torn across another process's line.
MARKUP_JOURNAL_MAX_LINE_BYTES = 8192

#: The identity a line must carry even at the floor — who leaked, from which
#: tool, into which parameter, when, and with what outcome. Exactly the question
#: the storm escalation asks a human to answer and gives them no way to.
MARKUP_JOURNAL_FLOOR_KEYS = ('ts', 'server', 'subject_task_id', 'tool', 'param', 'outcome')

#: The two disclosure keys, NAMESPACED under ``journal_`` because everything
#: else in a line except this module's own envelope comes from the middleware's
#: fact record — so a fact that grew a ``truncated`` or an ``overflow`` key
#: later could not collide with, or be mistaken for, the journal's own marker.
#:
#: Both are ABSENT on an untouched record, so their presence means "something
#: was cut" rather than merely "this emitter ran".
MARKUP_JOURNAL_TRUNCATED_KEY = 'journal_truncated'
MARKUP_JOURNAL_OVERFLOW_KEY = 'journal_overflow'


def _encode(entry: dict[str, Any]) -> str:
    """One journal record as a line, with no envelope literal left verbatim.

    Deterministic key order is what makes the file greppable BY FIELD, and
    matches the encoding the existing fact emitters
    (``verdict_tools._emit_markup_fact`` and ``escalation.server``'s twin) use —
    so a journal line and a log line are the same bytes plus this module's
    envelope.

    THE OPENING ANGLE BRACKET IS ESCAPED, blanket. This journal records envelope
    literals by construction: ``pattern`` and ``misclose`` ARE the leaked
    markup. A file holding them verbatim is a file no agent can safely read or
    edit — pulling it into a tool-call argument reproduces the exact
    over-consumption defect the journal exists to record, at the one artifact an
    operator is told to open. The committed specimen corpus
    (``shared/tests/fixtures/toolcall_markup_corpus.jsonl``) escapes every
    literal the same way for the same reason.

    A blanket replacement on the ENCODED line stays valid JSON: in JSON output
    the opening angle bracket can only ever occur inside a string, never as
    structural punctuation, so nothing else can be hit. And the escape
    round-trips through ``json.loads`` back to the original character, so this
    is lossless rather than sanitisation — a mangled ``pattern`` would destroy
    the one field that says which literal the upstream harness bug emitted.

    ``default=str`` because a value ``json`` cannot serialise must cost that
    ONE VALUE its exact type, never the whole line. The middleware emits only
    str / None / list-of-str today, but so was the key-count sprawl the
    overflow floor already defends against: dropping a record because one field
    grew an unexpected type is the exact fail-soft this module exists to end.
    It is not a complete guarantee — a non-string KEY still raises past
    ``default``, which is why the caller keeps the floor under this too.
    """
    return (
        json.dumps(entry, sort_keys=True, default=str).replace(_LT, _LT_JSON_ESCAPE)
        + '\n'
    )


def _fit(entry: dict[str, Any], budget: int) -> str:
    """*entry* encoded into at most *budget* bytes, hard-cutting strings to get there.

    The floor under the floor. ``MARKUP_JOURNAL_FLOOR_KEYS`` bounds a record's
    KEY COUNT but not its VALUE lengths — ``subject_task_id`` comes from an
    injected thunk (for plan-tools, ``plan.json``'s agent-written ``task_id``)
    and ``worktree`` from a path, neither of which anything upstream bounds. So
    the identity-only line is MEASURED after it is built rather than assumed to
    fit, because a floor line that overruns defeats the very bound that makes
    the O_APPEND atomicity premise sound.

    Each pass quarters the per-string budget, so it terminates at zero-length
    values — a line of bare keys, ~100 bytes, still naming the tool and the
    outcome. Ugly, and still strictly more than the nothing that was written
    before this journal existed.
    """
    line = _encode(entry)
    if len(line.encode('utf-8')) <= budget:
        return line
    cut = MARKUP_JOURNAL_MAX_FIELD_CHARS
    while cut > 0:
        cut //= 4
        line = _encode({
            key: value[:cut] if isinstance(value, str) else value
            for key, value in entry.items()
        })
        if len(line.encode('utf-8')) <= budget:
            break
    return line


def _capped(value: Any, trimmed: list[str], name: str) -> Any:
    """*value* bounded in place, appending *name* to *trimmed* if it was cut.

    A trimmed string is a PREFIX of what was sent, never a rewrite — the same
    discipline the middleware's own repair keeps, so an operator comparing a
    journal line against a transcript sees the beginning of the real value
    rather than an elision of it. The disclosure rides in
    ``MARKUP_JOURNAL_TRUNCATED_KEY`` instead of in an in-band suffix for the
    same reason: an in-band marker would corrupt the value it describes.
    """
    if isinstance(value, str) and len(value) > MARKUP_JOURNAL_MAX_FIELD_CHARS:
        trimmed.append(name)
        return value[:MARKUP_JOURNAL_MAX_FIELD_CHARS]
    if isinstance(value, (list, tuple)):
        kept = list(value[:MARKUP_JOURNAL_MAX_LIST_ITEMS])
        items = [
            item[:MARKUP_JOURNAL_MAX_FIELD_CHARS] if isinstance(item, str) else item
            for item in kept
        ]
        # BOTH axes are disclosed: a list is cut by its COUNT and each of its
        # items by its LENGTH, and either cut has to raise the marker. Reporting
        # only the count axis would let a list that fit item-for-item but whose
        # items were rewritten pass as untouched — and the constants above say a
        # missing marker means "nothing was cut", so a consumer would be right
        # to trust it and wrong about the data.
        if len(value) > MARKUP_JOURNAL_MAX_LIST_ITEMS or items != kept:
            trimmed.append(name)
        return items
    return value


def _scalar(value: Any) -> Any:
    """*value* as something ``json`` is guaranteed to render, bounded.

    Used only for the identity-only floor, which is reached BY a record that
    could not be encoded or could not be bounded — so it must not be built out
    of the same values that just failed. A JSON scalar passes through; anything
    else becomes its ``str()`` (the same coercion ``_encode``'s ``default``
    applies, hoisted so the floor cannot depend on it), capped like any other
    field.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)[:MARKUP_JOURNAL_MAX_FIELD_CHARS]


def journal_path(project_root: Path, server_label: str) -> Path:
    """The one file this server's markup facts are appended to.

    ONE file per server, not one per record. The per-record shape
    ``data/orchestrator/retry_cap_reports`` uses would turn "who leaked in this
    window" back into a directory walk, which is the shape of failure this
    module exists to end: an operator gets a single exact path to grep or pipe
    to ``jq``.

    No rotation, deliberately. Measured volume is 35 lines over three days
    across the whole fleet, and a dated or rotated filename would put a moving
    part into the one path an operator is told to open. If volume ever changes,
    adding rotation is a later, separable decision.
    """
    return project_root / MARKUP_JOURNAL_DIRNAME / f'{server_label}.jsonl'


def make_fact_journal(
    *,
    worktree: Path,
    server_label: str,
    subject_task_id: Callable[[], str],
    resolve_root: Callable[[Path], Path | None] = markup_sink.resolve_project_root,
    now: Callable[[], float] = time.time,
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Build the ``fact_sink`` a boundary guard journals its facts through.

    Returns the journal path as a locator string — the same contract
    ``make_escalation_sink`` keeps with the queued record's id, so the two
    injected channels report their result the same way.

    *subject_task_id* is a THUNK, not a value, for the reason ``markup_sink``
    records: the middleware cannot resolve the leaking task
    (``MarkupGuardMiddleware._identity`` reads identity off the call's own
    arguments, and no tool on these servers declares ``agent_id`` /
    ``project_root`` / ``project_id``), and the attribution a concrete site can
    offer may not exist yet at ``create_server`` time — a plan-tools
    ``create_plan`` refused before any plan exists is the case. So it is
    evaluated PER RECORD, on the worker thread, never at wiring time.

    *resolve_root* defaults to ``--git-common-dir`` resolution, which from a
    task worktree names the MAIN checkout. That is the point: a worktree-local
    artifact dies with the lane at ``git worktree remove --force`` and nothing
    ever reads it — the documented reason ``TaskArtifacts.write_markup_residue``
    is wired only as a last resort and never as a channel.

    ASYNC, with the blocking work on a worker thread: the middleware calls this
    from inside the server's event loop, and one record can run a
    ``git rev-parse`` subprocess and a filesystem write.
    """
    #: Holds SUCCESSES ONLY. A failed git resolution is TRANSIENT by nature (a
    #: fork failure under load, an EINTR, a timeout, an index.lock storm), and
    #: caching one would permanently disable this journal for every later event
    #: on this server — losing exactly the records it exists to preserve. Same
    #: asymmetry, for the same reason, as ``markup_sink.make_escalation_sink``'s
    #: two memos.
    project_root: Path | None = None

    def _subject() -> str:
        """The attribution ladder, applied per record.

        THE SAME ladder ``markup_sink.make_escalation_sink`` applies, because it
        is literally the same function: the two channels on one boundary are
        asserted to name the same subject, and two copies of the ladder could
        drift in opposite directions while both kept passing their own tests.
        """
        return markup_sink.resolve_subject(
            subject_task_id, worktree, what='journal line',
        )

    def append(record: dict[str, Any]) -> str | None:
        """The blocking body, run on a worker thread."""
        nonlocal project_root
        if project_root is None:
            try:
                project_root = resolve_root(worktree)
            except Exception:
                # The default resolver never raises — it logs and returns None.
                # An INJECTED one may, and a boundary guard's fact channel is
                # not where that becomes the caller's problem.
                logger.exception(
                    'markup guard: the project-root resolver failed for %s; '
                    'this markup fact will not be journalled', worktree,
                )
                return None
            if project_root is None:
                logger.warning(
                    'markup guard: could not resolve project_root from %s, so '
                    'the markup fact for %s.%s will not be journalled',
                    worktree, record.get('tool'), record.get('param'),
                )
                return None

        path = journal_path(project_root, server_label)

        trimmed: list[str] = []
        # THE ENVELOPE IS CAPPED TOO, not just the fact's fields. ``worktree``
        # is a filesystem path and ``subject_task_id`` comes from the injected
        # thunk — for plan-tools, ``plan.json``'s AGENT-WRITTEN ``task_id`` —
        # so neither is bounded by anything upstream, and both are among the
        # floor keys. An unbounded floor key would push the identity-only line
        # itself past the byte bound, defeating the bound at exactly the point
        # it is the last thing standing.
        envelope = {
            name: _capped(value, trimmed, name)
            for name, value in (
                ('ts', datetime.fromtimestamp(now(), tz=UTC).isoformat()),
                ('server', server_label),
                ('subject_task_id', _subject()),
                ('worktree', str(worktree)),
                ('pid', os.getpid()),
            )
        }
        capped_record = {
            name: _capped(value, trimmed, name) for name, value in record.items()
        }
        # THE ENVELOPE WINS ON COLLISION. Everything but these five keys comes
        # from the middleware's record, and four of the five are floor keys —
        # the identity the floor exists to guarantee. A fact that grew a key
        # named ``server`` or ``ts`` must not be able to overwrite who actually
        # wrote the line and when, so the merge order puts the journal's own
        # answer last. The collision is LOGGED rather than silently resolved:
        # it means the middleware and this envelope have started disagreeing
        # about a name, which is a fact worth an operator's attention.
        for name in envelope.keys() & capped_record.keys():
            logger.warning(
                'markup guard: the markup fact for %s.%s carries %r, which is '
                "the journal's own envelope key; keeping the journal's value "
                'and dropping the fact-supplied one',
                record.get('tool'), record.get('param'), name,
            )
        entry: dict[str, Any] = {**capped_record, **envelope}
        if trimmed:
            entry[MARKUP_JOURNAL_TRUNCATED_KEY] = sorted(set(trimmed))

        try:
            line = _encode(entry)
            degrade = len(line.encode('utf-8')) > MARKUP_JOURNAL_MAX_LINE_BYTES
            if degrade:
                logger.warning(
                    'markup guard: the journal record for %s.%s exceeded %d '
                    'bytes; writing the identity-only floor instead',
                    record.get('tool'), record.get('param'),
                    MARKUP_JOURNAL_MAX_LINE_BYTES,
                )
        except (TypeError, ValueError):
            # ``default=str`` covers an unserialisable VALUE; this covers what
            # it cannot — a non-string KEY, or a key set ``sort_keys`` cannot
            # order. Same disposition as an over-budget record, and for the
            # same reason: degrade the line, never drop it. Dropping it would
            # leave the event exactly where this module found it, in a stack
            # trace on the stderr nobody retains.
            logger.exception(
                'markup guard: the journal record for %s.%s could not be '
                'encoded; writing the identity-only floor instead',
                record.get('tool'), record.get('param'),
            )
            degrade = True
        if degrade:
            # THE OVERFLOW FLOOR. Per-field caps cannot bound a record that is
            # over budget in its KEY COUNT rather than in any one value — a
            # shape a future middleware could grow — so there has to be a floor
            # under them.
            #
            # A shorter honest line beats both a torn line (which would corrupt
            # another process's record in the same file) and a dropped one.
            # Dropping it is the exact fail-soft this PRD exists to end: the
            # whole reason this module exists is that these events reached no
            # durable sink at all.
            #
            # Every value is forced to a JSON scalar first, because this branch
            # is reached BY an unencodable record: a floor built out of the same
            # values that just failed to encode is no floor at all. And the
            # result is MEASURED by ``_fit`` rather than assumed to fit.
            floor: dict[str, Any] = {
                key: _scalar(entry[key])
                for key in MARKUP_JOURNAL_FLOOR_KEYS
                if key in entry
            }
            floor[MARKUP_JOURNAL_OVERFLOW_KEY] = True
            line = _fit(floor, MARKUP_JOURNAL_MAX_LINE_BYTES)

        # ONE ``os.write`` on an O_APPEND fd, and NEVER a read-modify-write.
        #
        # Many of these servers run at once — one stdio subprocess per agent,
        # across concurrent tasks — and all of them append to this one file. A
        # "read the file, concatenate, rewrite" implementation would silently
        # destroy the other processes' lines, which is the same class of data
        # loss the whole markup guard exists to prevent. O_APPEND makes the
        # seek-and-write atomic with no cross-process lock to take, but that
        # atomicity is guaranteed only for a BOUNDED write — so the record is
        # capped rather than trusted to be small.
        payload = line.encode('utf-8')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                # THE RETURN VALUE IS NOT IGNORED. ``write(2)`` may report a
                # SHORT count — classically near ENOSPC, or on a large write
                # interrupted by a signal — and a partial record has no
                # trailing newline, so the next appender concatenates onto it
                # and one unparseable line silently swallows two events. Every
                # consumer of this file splits on newlines, so that is the one
                # corruption shape this format cannot survive. Finishing the
                # remainder is not atomic with the first chunk, but a torn line
                # that is at least COMPLETE and newline-terminated is legible;
                # an unterminated one is not.
                written = 0
                while written < len(payload):
                    chunk = os.write(fd, payload[written:])
                    if chunk <= 0:
                        raise OSError(
                            f'wrote {chunk} bytes of {len(payload)} to {path}'
                        )
                    if written == 0 and chunk < len(payload):
                        logger.warning(
                            'markup guard: short write journalling %s.%s to %s '
                            '(%d of %d bytes); completing the line',
                            record.get('tool'), record.get('param'), path,
                            chunk, len(payload),
                        )
                    written += chunk
            finally:
                os.close(fd)
        except OSError:
            logger.exception(
                'markup guard: could not append the markup fact for %s.%s to '
                '%s; the outcome stands', record.get('tool'),
                record.get('param'), path,
            )
            return None
        return str(path)

    async def journal(record: dict[str, Any]) -> str | None:
        try:
            return await asyncio.to_thread(append, record)
        except Exception:
            # ``append`` already contains every write failure; this is the floor
            # under the thread hop itself, so the never-raises contract holds
            # for the whole emitter and not merely for its body. Same floor
            # ``markup_sink.make_escalation_sink``'s own ``sink`` keeps.
            logger.exception(
                'markup guard: could not run the fact journal for %r; the '
                'outcome stands', record.get('fact'),
            )
            return None

    return journal
