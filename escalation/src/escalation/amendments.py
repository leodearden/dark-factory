"""Size-bounded framing amendments: the caps, and the one append-and-trim
policy every writer of ``Escalation.amendments`` goes through.

A leaf importing only ``escalation.models``, which is what lets
``escalation.queue`` import it downward.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from escalation.models import Amendment, Escalation

logger = logging.getLogger(__name__)


# Hard cap on the NUMBER of Escalation.amendments entries (see
# append_amendment_capped, the single home of the append-and-trim policy for
# every writer of that list).  Worst case is repeated folds of one cluster
# inside a single AFK window, and amendments are deliberately NOT in the server's
# compact projection, so they never inflate a watcher's drain no matter how deep
# the list gets.  Past the cap the OLDEST are shed — the ORIGINAL framing is never
# in this list at all (it lives permanently in the record's own immutable
# root_cause/detail/options/summary), so the oldest amendment is the
# least-informative entry and a rotation triaging NOW needs the most recent
# framing.  Each drop increments the record's `amendments_truncated`, so the loss
# is a durable structured fact rather than log-only.
# SIZED AGAINST THE POST-CANONICALISATION FOLD RATE (task 3998), which raises the
# fold rate BY DESIGN — a cap tuned to today's rate would be immediately wrong.
MAX_AMENDMENTS = 20

# Per-ENTRY character caps.  An entry count alone is NOT a size bound: each
# entry's `detail` is the promote's unbounded free-text `evidence` argument —
# the very field compact mode exists to keep off the wire — so a hot-folding L2
# could stay "inside the cap" while growing by hundreds of KB, and EVERY reader
# of a full record pays that (get_escalation, get_pending_escalations(compact=
# False), every sweep's JSON parse).  These caps are what make the envelope
# ARITHMETIC rather than assertion:
#
#   kept text per entry <= 2*300 (root_cause + summary) + 1200 (detail)
#                          + 6*300 (options)                    = 3600 chars
#   + at most 9 elision markers (~70 chars each) + role/timestamp  ~  800 chars
#   whole list <= MAX_AMENDMENTS * ~4.4 KB                       ~=   88 KB
#
# root_cause and summary are one-LINERS by contract (a dedup key and a one-line
# hypothesis), and each option is an 'A: ...' style choice, so they share one
# line-sized cap; `detail` gets the larger share because it is the only field
# meant to carry prose.  Elision is per-field and marked in-band; the characters
# dropped are counted on the record in `amendments_chars_elided`, exactly as
# shed ENTRIES are counted in `amendments_truncated` — the loss is a durable
# structured fact either way, never log-only.
MAX_AMENDMENT_LINE_CHARS = 300
MAX_AMENDMENT_DETAIL_CHARS = 1200
MAX_AMENDMENT_OPTIONS = 6


def elide(text: str, limit: int, what: str = 'amendment field') -> tuple[str, int]:
    """Return (*text* capped at *limit*, characters dropped).

    The elision is MARKED IN-BAND and names both the count and the cap, so a
    human reading a preserved framing can tell "this is all of it" from "this
    is the head of it" without cross-referencing anything.  Silent truncation
    of decision context would be the loud-over-silent norm inverted.

    *what* names WHICH cap did the eliding, so a reader of a marker on a
    late-resolution entry is not told it hit an "amendment field cap" it has
    nothing to do with.  It defaults to the amendment wording, keeping every
    pre-existing caller's output byte-identical.
    """
    if len(text) <= limit:
        return text, 0
    dropped = len(text) - limit
    return (
        f'{text[:limit]}\n[... {dropped} char(s) elided at the '
        f'{limit}-char {what} cap ...]'
    ), dropped


def _build_amendment(
    *, root_cause: str, summary: str, evidence: str, options: list[str] | None,
    agent_role: str, timestamp: str,
) -> tuple[Amendment, int]:
    """Build one :class:`~escalation.models.Amendment`, size-bounded.

    THE single site that maps the promote's ARGUMENT names onto the Amendment's
    KEY names — notably *evidence* onto ``detail``, the same field the create
    path (``escalation/src/escalation/server.py::promote_to_l2``) writes that
    argument into.  It builds both the entry actually appended AND the
    projection of the record's OWN framing that repeat detection compares
    against (see :func:`_is_repeat_framing`), so those two can never
    drift into comparing differently-shaped dicts — and, because BOTH sides are
    elided by this one function, repeat detection keeps working verbatim on
    framing large enough to be elided.

    Every free-text field is capped (see ``MAX_AMENDMENT_LINE_CHARS`` /
    ``MAX_AMENDMENT_DETAIL_CHARS``) and the options LIST is capped in length
    (``MAX_AMENDMENT_OPTIONS``), which is what turns ``MAX_AMENDMENTS`` from
    an entry count into an actual size bound.  Options past the cap are shed
    whole and their full length counts as elided, so the returned total covers
    everything the entry did not keep.

    Returns ``(amendment, chars_elided)``.
    """
    kept_root_cause, rc_lost = elide(root_cause, MAX_AMENDMENT_LINE_CHARS)
    kept_summary, sm_lost = elide(summary, MAX_AMENDMENT_LINE_CHARS)
    kept_detail, dt_lost = elide(evidence, MAX_AMENDMENT_DETAIL_CHARS)

    incoming_options = list(options or [])
    # Shed the TAIL of an over-long options list, the opposite of the entry-cap
    # policy: options are ordered proposals ('A: ...', 'B: ...'), so the leading
    # ones are the ones a reader is being asked to choose between.
    opt_lost = sum(len(o) for o in incoming_options[MAX_AMENDMENT_OPTIONS:])
    kept_options: list[str] = []
    for option in incoming_options[:MAX_AMENDMENT_OPTIONS]:
        kept, lost = elide(option, MAX_AMENDMENT_LINE_CHARS)
        kept_options.append(kept)
        opt_lost += lost

    amendment: Amendment = {
        'timestamp': timestamp,
        'agent_role': agent_role,
        'root_cause': kept_root_cause,
        'summary': kept_summary,
        'detail': kept_detail,
        'options': kept_options,
    }
    return amendment, rc_lost + sm_lost + dt_lost + opt_lost


def _framing_view(a: Amendment) -> tuple[str, str, str, list[str]]:
    """The comparable framing of *a*: WHAT was said, normalised.

    ``agent_role`` is deliberately excluded — it records WHO said it, not WHAT
    was said, and two roles submitting identical framing is not new framing.
    ``timestamp`` likewise: :func:`append_amendment_capped` stamps it, so it
    always differs and would defeat the comparison entirely.

    ``root_cause`` is compared RAW-STRIPPED — deliberately NOT canonicalised,
    even though
    ``escalation/src/escalation/queue.py::EscalationQueue.find_pending_l2_by_root_cause``
    now matches canonically (task 3998).  Canonicalising here would suppress
    exactly the record the system needs: a fold whose root_cause differs only
    in case/whitespace/punctuation is
    a re-SPELLING of the cause, and that pre-canonical spelling is the ONLY
    evidence that canonicalisation folded it.  It must therefore read as NEW
    framing and be recorded (the entry lands in ``Amendment.root_cause``, which
    is documented as the pre-canonical incoming root cause, and feeds the
    over-fold detector's distinct-variant count).  Treating it as a repeat would
    make an over-fold — distinct causes silently merged under one canonical key —
    unobservable, which is the failure mode canonicalisation introduces.

    The cost is amendment-cap churn from trivial re-spellings; that is bounded by
    ``MAX_AMENDMENTS``, counted in ``amendments_truncated``, and was budgeted:
    that constant is sized against the post-canonicalisation fold rate.

    ``.strip()`` (rather than nothing) is kept because the create path
    (``escalation/src/escalation/server.py::promote_to_l2``) stores
    ``root_cause.strip()`` on the record itself, so comparing unstripped text
    against a stripped field would report new framing for a pure no-op.
    """
    return (
        a.get('root_cause', '').strip(),
        a.get('summary', ''),
        a.get('detail', ''),
        list(a.get('options', [])),
    )


def _is_repeat_framing(esc: Escalation, candidate: Amendment) -> bool:
    """True when *candidate* repeats the framing the record ALREADY carries.

    A watcher rotation re-promoting the same cluster with UNCHANGED text has
    contributed nothing: recording it again would (a) manufacture a spurious
    ``updated_at`` bump and re-trigger the re-assess protocol on a true no-op —
    the contract task 3976 pinned — and (b) burn cap budget, so that
    ``MAX_AMENDMENTS`` worth of identical rows would push genuinely-distinct
    earlier framings out via the drop-oldest policy.  Both are the opposite of
    what preserving framing is for.

    THE BASELINE IS THE RECORD'S CURRENT POSITION, which is the LAST amendment
    when there is one and **the record's OWN framing when there is not**.  The
    empty-list case is not a special case to skip: an L2's first re-promote is
    exactly the one most likely to carry text byte-identical to the create
    (the watcher recomputes the same cluster and re-sends the same framing),
    and treating "no amendments yet" as "nothing to compare against" recorded a
    verbatim copy of the record's own root_cause/detail/options/summary — one
    spurious ``updated_at`` bump per L2, one wasted cap slot, and a re-assess
    re-triggered on a genuine no-op.  The record's own framing IS the implicit
    ``amendments[-1]``, so it is projected through :func:`_build_amendment` and
    compared the same way.

    Compared against the LAST position only, not all of them: an A -> B -> A
    reframing genuinely returns to a previous position and IS new relative to
    what the record currently says.
    """
    if esc.amendments:
        baseline = esc.amendments[-1]
    else:
        # Built through the SAME function as the candidate, so the record's own
        # framing is elided identically — a re-promote of framing long enough to
        # be elided still compares equal instead of reading as new every time.
        baseline, _ = _build_amendment(
            root_cause=esc.root_cause, summary=esc.summary, evidence=esc.detail,
            options=esc.options, agent_role='', timestamp='',
        )
    return _framing_view(baseline) == _framing_view(candidate)


def has_framing(
    *, root_cause: str, summary: str, evidence: str, options: list[str] | None,
) -> bool:
    """True when incoming framing carries anything an amendment could record.

    THE one definition of "empty framing", read by every writer that must tell
    a no-op apart from a write before it decides what to do: an empty amendment
    carries no information and would still burn a ``MAX_AMENDMENTS`` slot.
    """
    return bool(root_cause or summary or evidence or options)


def append_amendment_capped(
    esc: Escalation, *, root_cause: str, summary: str, evidence: str,
    options: list[str] | None, agent_role: str, caller: str,
) -> tuple[bool, int]:
    """Append one framing :class:`~escalation.models.Amendment` to *esc*, capped.

    THE single home of the ``amendments`` append-and-trim policy, whichever
    queue method carries the framing in: build the entry through
    :func:`_build_amendment` (per-field elision), suppress it when
    :func:`_is_repeat_framing`, count elided characters in
    ``amendments_chars_elided``, and shed the OLDEST entries past
    ``MAX_AMENDMENTS`` into ``amendments_truncated``.  Every loss is a durable
    structured fact on the record and a WARNING, never log-only.

    Framing that fails :func:`has_framing` records nothing.

    Mutates *esc* in memory only.  The caller holds
    ``escalation/src/escalation/queue.py::escalation_id_lock``, owns the single
    ``escalation/src/escalation/queue.py::EscalationQueue._rewrite``, and decides
    whether to bump ``updated_at``, so the append stays atomic with whatever
    else that caller writes.  The timestamp is stamped HERE, so the write
    chokepoint owns the clock and no caller can backdate an entry.  *caller*
    only prefixes the WARNING lines.

    Returns ``(recorded, dropped)``: whether an entry was appended, and how many
    oldest entries the cap shed.
    """
    if not has_framing(
        root_cause=root_cause, summary=summary, evidence=evidence, options=options,
    ):
        return False, 0
    candidate, chars_elided = _build_amendment(
        root_cause=root_cause, summary=summary, evidence=evidence,
        options=options, agent_role=agent_role,
        timestamp=datetime.now(UTC).isoformat(),
    )
    # Built BEFORE the repeat check, and the check reads the built entry: the
    # stored form is what a later amendment will be compared against, so
    # comparing anything else would let two entries the record cannot tell
    # apart both be recorded.
    if _is_repeat_framing(esc, candidate):
        return False, 0
    esc.amendments.append(candidate)
    if chars_elided:
        esc.amendments_chars_elided += chars_elided
        logger.warning(
            '%s: %s elided %d char(s) of incoming framing at the per-field '
            "amendment caps (running total elided=%d); the record's own "
            'framing is unaffected',
            caller, esc.id, chars_elided, esc.amendments_chars_elided,
        )
    dropped = 0
    if len(esc.amendments) > MAX_AMENDMENTS:
        dropped = len(esc.amendments) - MAX_AMENDMENTS
        del esc.amendments[:dropped]
        esc.amendments_truncated += dropped
        logger.warning(
            '%s: %s shed %d oldest amendment(s) at the MAX_AMENDMENTS=%d cap '
            "(running total truncated=%d); the record's own original framing "
            'is unaffected',
            caller, esc.id, dropped, MAX_AMENDMENTS, esc.amendments_truncated,
        )
    return True, dropped


def amendment_room(esc: Escalation) -> int:
    """How many more amendments *esc* takes before the cap sheds its oldest one."""
    return max(0, MAX_AMENDMENTS - len(esc.amendments))
