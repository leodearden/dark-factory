"""Filesystem-based escalation queue with atomic writes."""

from __future__ import annotations

import contextlib
import fcntl
import itertools
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from shared.timestamps import parse_timestamp_or_warn

from escalation import archive

# escalation.canonical is a LEAF (zero intra-package imports), which is what
# makes this import legal at all: dedupe.py imports THIS module at module level,
# so queue.py can never import dedupe.py, where the transform used to live.
from escalation.canonical import canonical_root_cause
from escalation.classify import default_resolution_class_for_resolver

# max_severity lives in models.py beside the KNOWN_SEVERITIES vocabulary it must
# stay total over (task 3976), so server.py can share it without reaching for a
# module-private symbol.  Imported under its real, public name: it is a shared
# cross-module helper, and spelling it `_max_severity` here would signal the
# opposite at every use site.
from escalation.models import RESOLUTION_CLASSES, Amendment, Escalation, max_severity

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def escalation_id_lock(queue_dir: Path, escalation_id: str) -> Iterator[None]:
    """Per-escalation-id exclusive advisory lock using a stable sidecar file.

    WHY A SIDECAR (PRD D3 rationale):
    The queue's writers are atomic tmp+rename (_atomic_write_path).  After a
    tmp+rename replace, the data file ``{escalation_id}.json`` is a NEW inode.
    A second writer that flock()s the (new) data-file path binds to a different
    inode and races anyway — the lock is defeated.  The fix is a STABLE lock
    target: ``{escalation_id}.json.lock``, created once via os.open(O_CREAT)
    and NEVER renamed or replaced, so all writers flock the same inode and
    actually serialize.

    EXPORT CONTRACT:
    This helper is module-level and exported so task ε (server-start sweep /
    reaper) can import it as::

        from escalation.queue import escalation_id_lock

    and take the same lock around its root↔archive relocations without
    instantiating an EscalationQueue.

    GLOB INVISIBILITY:
    The sidecar ends in ``.lock`` and does NOT match the ``esc-*.json`` glob
    used by get_pending / get_by_task / make_id / iter_all_escalation_paths.
    It is never RENAMED or replaced — that is the stable-inode contract above
    (task 1609) and the whole reason the sidecar exists.  It IS unlinked, in
    exactly one place: ``sweep.reap_orphan_locks``, the server-start pass that
    stops the root accumulating a sidecar per dead id.  That pass only ever
    touches a DEAD id — one whose record is absent from both the queue root and
    the archive — and ``make_id``'s monotonic durable counter can never re-mint
    such an id, so no future writer can want the inode it removes.

    Usage::

        with escalation_id_lock(queue_dir, escalation_id):
            data = read(); data = mutate(data); atomic_write(data)
    """
    lock_path = Path(queue_dir) / f'{escalation_id}.json.lock'
    # Defensively create queue_dir so standalone callers (e.g. task ε sweep/reaper)
    # can take the lock without having first instantiated an EscalationQueue.
    Path(queue_dir).mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

# Hard cap on EscalationQueue._archive_negative_cache (see _locate_path /
# _cache_archive_negative).  A polling client hammering many distinct
# nonexistent ids (typos, stale references, an adversarial sweep) must not
# grow the negative cache unboundedly; past this size it is reset outright
# rather than evicted piecemeal (simple, and negative-cache staleness is
# already bounded by the next self-archival — see _archive_resolved).
_ARCHIVE_NEGATIVE_CACHE_MAX_SIZE = 10_000

# Suffix marking a durable per-task_id sequence counter, ``esc-{task_id}.seq``.
# TWO coupled consumers, which is why it is a shared constant rather than a
# literal at either site: ``make_id`` NAMES the counter with it, and
# ``sweep.reap_orphan_locks`` uses it as its never-reap guard (a counter has no
# ``.json`` record, so it would otherwise look permanently orphaned).
SEQ_COUNTER_SUFFIX = '.seq'

# Hard cap on the NUMBER of Escalation.amendments entries (see add_members_to_l2,
# the SOLE writer and sole trimmer).  Worst case is repeated folds of one cluster
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
_MAX_AMENDMENTS = 20

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
#   whole list <= _MAX_AMENDMENTS * ~4.4 KB                      ~=   88 KB
#
# root_cause and summary are one-LINERS by contract (a dedup key and a one-line
# hypothesis), and each option is an 'A: ...' style choice, so they share one
# line-sized cap; `detail` gets the larger share because it is the only field
# meant to carry prose.  Elision is per-field and marked in-band; the characters
# dropped are counted on the record in `amendments_chars_elided`, exactly as
# shed ENTRIES are counted in `amendments_truncated` — the loss is a durable
# structured fact either way, never log-only.
_MAX_AMENDMENT_LINE_CHARS = 300
_MAX_AMENDMENT_DETAIL_CHARS = 1200
_MAX_AMENDMENT_OPTIONS = 6

# Hard cap on the NUMBER of Escalation.root_cause_variants entries — the DISTINCT
# pre-canonical root_cause spellings one L2 has been addressed by (task 3998).
# Same sole-writer/sole-trimmer, oldest-shed, count-every-drop policy as
# `_MAX_AMENDMENTS` above, and sized against the same post-canonicalisation fold
# rate; each entry is one elided LINE (a dedup key), so the whole list is bounded
# by 20 * ~370 chars ~= 7 KB.
# DELIBERATELY WELL ABOVE the server's over-fold alarm threshold
# (`_ROOT_CAUSE_OVERFOLD_VARIANT_THRESHOLD = 5`): the cap must never be what
# decides whether the signal fires, or a tighter cap would silently mute the
# very alarm it is meant to feed.  Past the cap the count keeps climbing in
# `root_cause_variants_truncated`, so the TRUE distinct total
# (`len(list) + truncated`) never plateaus.
_MAX_ROOT_CAUSE_VARIANTS = 20

# Hard cap on the NUMBER of Escalation.dedupe_children entries (see
# attach_dedupe_child, the SOLE writer and sole trimmer).
# WHY A BOUND IS NEEDED AT ALL: `DedupeConfig.for_gate_backlog()` sets
# `infra_dedupe_window_secs=float('inf')` BY DESIGN — a 300h-old gate MUST still
# fold — so nothing ever ages a gate-backlog parent out.  It gains one child id
# per Stage-1 cycle for as long as a human has not decided, and that whole record
# is then read and parsed by `get_pending()` on EVERY subsequent dedupe scan, so
# an unbounded list taxes the scan that produced it.
# Sizing is ARITHMETIC, not assertion: each entry is one escalation id
# `esc-{key}-{n}` (the longest live key family is `ticket-janitor`, so <= ~30
# chars, + ~3 bytes of JSON quoting/comma), hence
#
#   whole list <= 200 * ~33 bytes                              ~=  6.6 KB
#
# — the same envelope as `root_cause_variants` (~7 KB) and far under
# `amendments` (~88 KB).
# _MAX_DEDUPE_CHILDREN_HEAD is the HEAD-RETENTION reserve, and it is what makes
# this list's policy DIFFERENT from the two above.  `amendments` can shed purely
# oldest-first because the ORIGINAL framing is not in that list at all (it lives
# in the record's own immutable root_cause/detail/options/summary).
# `dedupe_children` has no such external anchor: the ids that ESTABLISHED the
# fold are the list's only record of where the parent came from.  On the very
# case this cap exists for — a gate-backlog parent folding for weeks — pure
# oldest-shed would erase the fold's origin and leave a window of only-recent
# ids, the least useful provenance possible for a record whose whole problem is
# that it is old.  Head + tail keeps both the origin and the current activity.
# Past the cap the OLDEST NON-HEAD ids are shed, each drop counted in the
# record's `dedupe_children_truncated`, so the TRUE provenance total
# (`len(children) + truncated`) never plateaus and the loss is a durable
# structured fact rather than log-only.
# `dedupe_count` is NOT capped and must never be — it is the recurrence SIGNAL
# (task 3522); `dedupe_children` is PROVENANCE.  The cap separates the two.
_MAX_DEDUPE_CHILDREN = 200
_MAX_DEDUPE_CHILDREN_HEAD = 20


class AmendmentOutcome(TypedDict):
    """What ONE :meth:`EscalationQueue.add_members_to_l2` call did to ``amendments``.

    An OUT-PARAM rather than a widened return value: the ``Escalation`` is that
    method's primary result and every existing call site reads it directly, so
    returning a tuple/NamedTuple would churn all of them into saying
    ``.escalation`` for a fact only one caller wants.

    WHY IT EXISTS: ``server.promote_to_l2`` reports ``amendment_recorded`` to its
    caller and triggers the truncation storm escape.  Both are facts the write
    path already computes INSIDE ``escalation_id_lock``.  Re-deriving them in the
    server from a pre-call ``queue.get`` plus a "did the count grow, or did the
    truncation counter grow" heuristic cost a second full read+parse on every
    fold AND was a real TOCTOU: this queue is explicitly built for cross-process
    mutators (the sidecar flocks), so a concurrent fold landing between the
    pre-read and the call made the reported flag wrong in either direction — this
    call's framing landing but reading False, or another caller's append making a
    suppressed repeat read True.  Reported from inside the lock, it is exact.

    Both keys are populated on EVERY return path — including the not-found and
    no-op early returns — so a caller never reads a stale or missing key.
    """

    recorded: bool  # THIS call appended an amendment
    dropped: int    # entries THIS call shed at the _MAX_AMENDMENTS cap
    # THIS call added a previously-unseen PRE-canonical root_cause spelling
    # (task 3998).  Same TOCTOU argument as `recorded`: "is this spelling new to
    # the record" is only answerable against the record as it is INSIDE the lock.
    variant_added: bool
    # The record's TRUE distinct-spelling total AFTER this call —
    # `len(root_cause_variants) + root_cause_variants_truncated`, NOT the list
    # length, so it keeps climbing past `_MAX_ROOT_CAUSE_VARIANTS` instead of
    # plateauing exactly when a runaway makes it matter.  This is what the
    # server's over-fold threshold crossing is tested against.
    variants: int


def _elide(text: str, limit: int) -> tuple[str, int]:
    """Return (*text* capped at *limit*, characters dropped).

    The elision is MARKED IN-BAND and names both the count and the cap, so a
    human reading a preserved framing can tell "this is all of it" from "this
    is the head of it" without cross-referencing anything.  Silent truncation
    of decision context would be the loud-over-silent norm inverted.
    """
    if len(text) <= limit:
        return text, 0
    dropped = len(text) - limit
    return (
        f'{text[:limit]}\n[... {dropped} char(s) elided at the '
        f'{limit}-char amendment field cap ...]'
    ), dropped


def _build_amendment(
    *, root_cause: str, summary: str, evidence: str, options: list[str] | None,
    agent_role: str, timestamp: str,
) -> tuple[Amendment, int]:
    """Build one :class:`~escalation.models.Amendment`, size-bounded.

    THE single site that maps the promote's ARGUMENT names onto the Amendment's
    KEY names — notably *evidence* onto ``detail``, the same field the create
    path writes that argument into.  It builds both the entry actually appended
    AND the projection of the record's OWN framing that repeat detection
    compares against (see :func:`_is_repeat_framing`), so those two can never
    drift into comparing differently-shaped dicts — and, because BOTH sides are
    elided by this one function, repeat detection keeps working verbatim on
    framing large enough to be elided.

    Every free-text field is capped (see ``_MAX_AMENDMENT_LINE_CHARS`` /
    ``_MAX_AMENDMENT_DETAIL_CHARS``) and the options LIST is capped in length
    (``_MAX_AMENDMENT_OPTIONS``), which is what turns ``_MAX_AMENDMENTS`` from
    an entry count into an actual size bound.  Options past the cap are shed
    whole and their full length counts as elided, so the returned total covers
    everything the entry did not keep.

    Returns ``(amendment, chars_elided)``.
    """
    kept_root_cause, rc_lost = _elide(root_cause, _MAX_AMENDMENT_LINE_CHARS)
    kept_summary, sm_lost = _elide(summary, _MAX_AMENDMENT_LINE_CHARS)
    kept_detail, dt_lost = _elide(evidence, _MAX_AMENDMENT_DETAIL_CHARS)

    incoming_options = list(options or [])
    # Shed the TAIL of an over-long options list, the opposite of the entry-cap
    # policy: options are ordered proposals ('A: ...', 'B: ...'), so the leading
    # ones are the ones a reader is being asked to choose between.
    opt_lost = sum(len(o) for o in incoming_options[_MAX_AMENDMENT_OPTIONS:])
    kept_options: list[str] = []
    for option in incoming_options[:_MAX_AMENDMENT_OPTIONS]:
        kept, lost = _elide(option, _MAX_AMENDMENT_LINE_CHARS)
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
    ``timestamp`` likewise: the queue stamps it, so it always differs and would
    defeat the comparison entirely.

    ``root_cause`` is compared RAW-STRIPPED — deliberately NOT canonicalised,
    even though ``find_pending_l2_by_root_cause`` now matches canonically (task
    3998).  Canonicalising here would suppress exactly the record the system
    needs: a fold whose root_cause differs only in case/whitespace/punctuation is
    a re-SPELLING of the cause, and that pre-canonical spelling is the ONLY
    evidence that canonicalisation folded it.  It must therefore read as NEW
    framing and be recorded (the entry lands in ``Amendment.root_cause``, which
    is documented as the pre-canonical incoming root cause, and feeds the
    over-fold detector's distinct-variant count).  Treating it as a repeat would
    make an over-fold — distinct causes silently merged under one canonical key —
    unobservable, which is the failure mode canonicalisation introduces.

    The cost is amendment-cap churn from trivial re-spellings; that is bounded by
    ``_MAX_AMENDMENTS``, counted in ``amendments_truncated``, and was budgeted:
    that constant is sized against the post-canonicalisation fold rate.

    ``.strip()`` (rather than nothing) is kept because the create path stores
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
    ``_MAX_AMENDMENTS`` worth of identical rows would push genuinely-distinct
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


def iter_all_escalation_paths(escalations_dir: Path) -> Iterator[Path]:
    """Yield all ``esc-*.json`` paths from *escalations_dir* and its archive subtree.

    Two-tier scan with root precedence (mirrors the convention in
    ``EscalationQueue.get_by_task``):

    1. Queue root — ``escalations_dir.glob('esc-*.json')`` is iterated first.
       Each yielded path's stem is added to ``seen``.
    2. Archive subtree — ``escalations_dir/archive/`` (``archive.ARCHIVE_SUBDIR``)
       is scanned with ``rglob('esc-*.json')``.  Paths whose stem is already in
       ``seen`` (i.e. a root copy was found) are skipped so the root copy wins
       on id collisions.  Stems are added to ``seen`` as they are yielded so
       archive-only multi-date duplicates (same id under two dated subdirs) are
       also deduplicated.

    If *escalations_dir* does not exist or is not a directory, the iterator
    yields nothing without raising an exception.  This matches the
    dashboard's read-only usage: a missing or misconfigured path produces an
    empty result rather than a hard failure.

    Primary caller: ``dashboard.data.performance._load_escalations`` — bulk
    read of all escalation files for performance aggregation.
    """
    escalations_dir = Path(escalations_dir)
    if not escalations_dir.is_dir():
        return

    seen: set[str] = set()

    # Tier 1: queue root
    for path in escalations_dir.glob('esc-*.json'):
        seen.add(path.stem)
        yield path

    # Tier 2: archive subtree (root wins on id collisions)
    archive_root = escalations_dir / archive.ARCHIVE_SUBDIR
    if archive_root.exists():
        for path in archive_root.rglob('esc-*.json'):
            if path.stem not in seen:
                seen.add(path.stem)
                yield path


class EscalationQueue:
    """Filesystem queue for escalations.

    Each escalation is a JSON file named {id}.json in the queue directory.
    Writes are atomic (tmp file + rename). Reads tolerate partial writes.
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._notify_callback: Callable[[Escalation], None] | None = None
        self._resolve_callback: Callable[[Escalation], None] | None = None
        # Archive listing cache (serves get()/_locate_path).
        # None = not yet built; dict = {dated-subdir-name: set-of-esc-id-stems}.
        # Lazy-built on first archive lookup; incrementally updated in _archive_resolved.
        self._archive_listing: dict[str, set[str]] | None = None
        # Negative-result cache for _locate_path's cross-process re-probe: an
        # id that a targeted re-probe confirmed genuinely absent, so repeated
        # lookups for it don't re-rglob the archive on every call.  Cleared
        # wholesale in _archive_resolved (this instance's next self-archival)
        # and hard-capped at _ARCHIVE_NEGATIVE_CACHE_MAX_SIZE.  See
        # _locate_path / _cache_archive_negative for the full rationale.
        self._archive_negative_cache: set[str] = set()

    def set_notify_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._notify_callback = callback

    def set_resolve_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._resolve_callback = callback

    def _scan_archive_listing(self) -> dict[str, set[str]]:
        """Build the archive listing from a single rglob scan.

        Returns a dict mapping dated-subdir-name (e.g. '2025-06-15') to a set
        of esc-id stems (e.g. {'esc-1-1', 'esc-1-2'}) for all files under the
        archive root.  Returns an empty dict when the archive root does not exist.
        """
        archive_root = self.queue_dir / archive.ARCHIVE_SUBDIR
        listing: dict[str, set[str]] = {}
        if archive_root.exists():
            for path in archive_root.rglob('esc-*.json'):
                listing.setdefault(path.parent.name, set()).add(path.stem)
        return listing

    def _get_archive_listing(self) -> dict[str, set[str]]:
        """Return the memoised archive listing, building it on first access."""
        if self._archive_listing is None:
            self._archive_listing = self._scan_archive_listing()
        return self._archive_listing

    def _iter_archive_paths(self, pattern: str) -> Iterator[Path]:
        """Yield escalation JSON files from the archive subtree matching *pattern*.

        Returns an empty iterator when the archive root does not exist, avoiding
        a spurious ``rglob`` call on a missing directory.  Centralising the
        existence-check + rglob here means get() and get_by_task() both
        benefit from any future caching or indexing added in one place.
        Note: make_id() uses this only via ``_recover_seq_from_disk`` — i.e.
        when its counter file is absent or unparseable, which includes a
        brand-new task_id's first mint — never while the counter is intact;
        the counter remains authoritative in steady state.
        """
        archive_root = self.queue_dir / archive.ARCHIVE_SUBDIR
        if archive_root.exists():
            yield from archive_root.rglob(pattern)

    def submit(self, escalation: Escalation) -> str:
        """Atomic write: {id}.tmp -> rename to {id}.json.

        The write is serialized per-id by ``escalation_id_lock`` to prevent a
        concurrent same-id submit from clobbering an in-progress RMW mutator
        (add_members_to_l2, attach_dedupe_child, resolve).  The notify callback
        fires OUTSIDE the lock so a callback that re-enters the queue for the
        same id does not deadlock.

        Durability contract: the pending record is written with
        ``durable=True`` — fsync-flushed (temp-file fd, then the containing
        directory fd) before this call returns — so it is at least as
        crash-durable as the per-task seq counter that names it
        (``_write_seq_counter``, also durable=True). This closes the window
        in which a machine-level crash / power-loss / storage fault (e.g.
        during a merge-triggered service restart) could drop a just-filed
        born-at-L2 gate escalation while its durable seq counter and the
        orchestrator's ``gate_escalated_at`` stamp survived.
        """
        with escalation_id_lock(self.queue_dir, escalation.id):
            self._atomic_write(escalation.id, escalation.to_json(), durable=True)
        logger.info(f'Escalation submitted: {escalation.id} [{escalation.severity}]')

        if self._notify_callback:
            try:
                self._notify_callback(escalation)
            except Exception as e:
                logger.warning(f'Notify callback failed for {escalation.id}: {e}')

        return escalation.id

    def _locate_path(self, escalation_id: str) -> Path | None:
        """Return the on-disk path for *escalation_id*, or None if not found.

        Lookup order:
        1. ``queue_dir/{escalation_id}.json`` (queue root).
        2. Archive subtree (``archive/YYYY-MM-DD/{escalation_id}.json``), resolved
           via the memoised per-subdir listing:
           - If exactly one candidate: return it.
           - If multiple candidates (multi-date duplicates): emit a WARNING naming the id
             and return the newest by ``parent.name`` lexicographic order
             (YYYY-MM-DD sorts lexicographically == chronologically; non-date names fall back
             to ``''`` and are treated as oldest).
        3. Cross-process staleness fallback: when the memoised listing yields NO
           candidates, the memo may simply predate another ``EscalationQueue``
           instance (e.g. a long-lived escalation-server process) archiving this
           id — a memo miss is therefore not authoritative on its own.  A single
           TARGETED archive re-probe (``_iter_archive_paths``) is performed
           before concluding the id is genuinely absent (unless the id is
           already negative-cached — see below).  A re-probe HIT is folded
           into the memoised listing (mirroring the incremental update in
           ``_archive_resolved``) so a subsequent lookup for the same
           archived-elsewhere id is a cache hit — bounding the re-probe to at
           most one targeted rglob per id per instance lifetime FOR IDS THAT
           EVENTUALLY RESOLVE TO AN ARCHIVED FILE.

        Negative caching (genuinely-absent ids): a re-probe MISS is recorded
        in ``self._archive_negative_cache`` (see ``_cache_archive_negative``)
        so a repeatedly-polled nonexistent id (a typo, a stale client
        reference, an already-swept id) does not re-trigger a full archive
        rglob on *every* call — this is the path a polling client or a bad
        id amplifies into repeated full-archive scans if left uncached. This
        trades a bounded staleness window (an id negative-cached here that
        some OTHER instance archives afterward will not be found by THIS
        instance until the cache next clears) for that protection: the cache
        is cleared wholesale on this instance's next self-archival
        (``_archive_resolved``) and hard-capped at
        ``_ARCHIVE_NEGATIVE_CACHE_MAX_SIZE``.

        Shared by ``get()`` and ``patch_resolution_metadata()`` so that the archive
        fallback and newest-by-date selection logic live in exactly one place.
        """
        path = self.queue_dir / f'{escalation_id}.json'
        if path.exists():
            return path
        # Fall back to archive using the memoised per-subdir listing.
        # Reconstruct candidate paths from the listing without re-rglob-ing.
        archive_root = self.queue_dir / archive.ARCHIVE_SUBDIR
        listing = self._get_archive_listing()
        candidates = [
            archive_root / subdir / f'{escalation_id}.json'
            for subdir, stems in listing.items()
            if escalation_id in stems
            and (archive_root / subdir / f'{escalation_id}.json').exists()
        ]
        reprobed = False
        if not candidates and escalation_id not in self._archive_negative_cache:
            # The memoised listing can predate another EscalationQueue
            # instance's archival of this id (cross-process staleness — see
            # docstring point 3).  Do a single targeted re-probe of the
            # archive subtree before concluding absence.  Skipped when the
            # id is already known-absent (negative-cached) from a prior
            # re-probe this instance lifetime.
            candidates = list(self._iter_archive_paths(f'{escalation_id}.json'))
            reprobed = True
        if not candidates:
            if reprobed:
                # Re-probe confirmed genuine absence: negative-cache it so a
                # repeated lookup for this id doesn't re-scan the archive.
                self._cache_archive_negative(escalation_id)
            return None
        if len(candidates) > 1:
            logger.warning(
                f'Multiple archive files for {escalation_id}: '
                f'{[str(p) for p in candidates]}; selecting newest by parent dir date'
            )
            # YYYY-MM-DD sorts lexicographically == chronologically.
            # Non-YYYY-MM-DD parent names fall back to '' (treated as oldest),
            # matching the comment and ensuring valid date dirs always win.
            selected = max(
                candidates,
                key=lambda p: (
                    p.parent.name
                    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', p.parent.name)
                    else ''
                ),
            )
        else:
            selected = candidates[0]
        if reprobed:
            # Fold the re-probe hit into the memo so a repeated lookup for
            # this id is a cache hit rather than a re-scan (bounds a polled
            # server to at most one targeted rglob per archived-elsewhere id).
            listing.setdefault(selected.parent.name, set()).add(escalation_id)
        return selected

    def _cache_archive_negative(self, escalation_id: str) -> None:
        """Record *escalation_id* as re-probed-and-genuinely-absent.

        Bounds ``_locate_path``'s re-probe cost for an id that never
        resolves to an archived file (a typo, a stale client reference, an
        already-swept id) to at most one targeted rglob per instance
        lifetime *between clears* — without this, a polled server (or an
        adversarial/buggy caller) requesting the same nonexistent id
        repeatedly re-triggers a full ``archive_root.rglob`` on every call,
        exactly the per-call O(archive) cost the positive memo was built to
        eliminate.

        Cleared wholesale in ``_archive_resolved`` (this instance's next
        self-archival) so a negative-cached id that some OTHER instance
        archives afterward is not masked indefinitely — only until this
        instance's own next archival checkpoint.  Hard-capped at
        ``_ARCHIVE_NEGATIVE_CACHE_MAX_SIZE``: past that size the cache is
        reset outright (simplest bound; avoids unbounded growth from a sweep
        of many distinct nonexistent ids without needing LRU bookkeeping).
        """
        if len(self._archive_negative_cache) >= _ARCHIVE_NEGATIVE_CACHE_MAX_SIZE:
            logger.info(
                f'Archive negative cache exceeded {_ARCHIVE_NEGATIVE_CACHE_MAX_SIZE} '
                'entries; resetting'
            )
            self._archive_negative_cache.clear()
        self._archive_negative_cache.add(escalation_id)

    def get(self, escalation_id: str) -> Escalation | None:
        """Read a single escalation by ID.

        Falls back to the archive directory when the file is not in the
        queue root (i.e. the escalation has been resolved and archived).

        The archive fallback uses a memoised per-dated-subdir listing
        (_archive_listing) so that repeated get() calls for archived ids
        scan the archive at most once per instance lifetime.  The listing
        is lazily built on the first archive miss and incrementally updated
        by _archive_resolved() when this instance archives a new escalation.

        Cross-process freshness: an escalation resolved+archived by another
        EscalationQueue instance (e.g. a concurrent process) after this
        instance's listing was built is still found — a memo miss triggers a
        single targeted archive re-probe in _locate_path (see its docstring)
        rather than being treated as authoritative absence.  This closes the
        window that previously made a just-archived-elsewhere escalation
        appear to vanish from a long-lived instance (e.g. the escalation
        server) until it happened to archive something itself.  A genuinely
        nonexistent id is negative-cached after its first re-probe miss (see
        _locate_path / _cache_archive_negative) — repeated get() calls for
        that same nonexistent id do not re-scan the archive, at the cost of
        a bounded staleness window that clears on this instance's next
        self-archival.
        """
        path = self._locate_path(escalation_id)
        if path is None:
            return None
        try:
            return Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
            return None

    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
        agent_role: str | None = None,
    ) -> list[Escalation]:
        """Scan dir for escalations matching a task ID.

        Two-tier scan:
        - status == 'pending': scan queue root only (fast path, skips archive).
        - status is None or another value: scan queue root PLUS archive/**/ to
          include resolved/dismissed escalations that have been moved out.

        ``agent_role``, when given, restricts results to escalations filed by
        that exact agent_role (None = no filter, full backward compatibility).
        Applied in the same pass as status/level, BEFORE dedup/seen bookkeeping.

        Deduplication: if the same escalation id appears in both the queue root
        and the archive (e.g. crash mid-resolve), only the first occurrence that
        passes all filters is returned and a DEBUG message is logged for any
        later same-id path that also passes.  Filters (task_id / status / level)
        are applied BEFORE adding to ``seen``, so a copy that fails its own
        filter never shadows a matching copy in the other tier.  Iteration order
        is queue root first, archive second, so the queue_dir copy wins when
        both copies pass the filter.

        Pre-scan warning: before applying any filter, if the same escalation id
        appears in BOTH the queue_dir root AND the archive subtree, a single
        WARNING is logged regardless of which (if any) copies pass the caller's
        filter.  Note: the pre-scan covers only the paths that are actually
        scanned — when ``status == 'pending'`` the archive is skipped entirely,
        so a cross-tier duplicate is invisible to the pre-scan in that mode.
        """
        # Build the candidate path list.
        paths: list[Path] = list(self.queue_dir.glob('esc-*.json'))
        if status != 'pending':
            paths.extend(self._iter_archive_paths('esc-*.json'))

        # Pre-scan: detect ids that span both queue_dir root and archive subtree.
        # Uses filename stems only (no JSON parsing) to avoid extra I/O.
        #
        # Tier-membership: classify each path by comparing its parent directory
        # to self.queue_dir.  Files written by queue_dir.glob('esc-*.json') are
        # non-recursive, so their parent IS self.queue_dir (root tier).  Files
        # from _iter_archive_paths come from archive_root.rglob(...) and land
        # inside a dated subdir (e.g. archive/2025-06-15/); their parent is that
        # date dir, NOT self.queue_dir (archive tier).
        #
        # A single id can legitimately appear multiple times within the archive
        # tier (the same stem in two different dated subdirs after a requeue or
        # backup-restore).  That situation is handled by get()'s newest-by-date
        # selection (queue.py:102-117) and is NOT a cross-tier inconsistency.
        # We warn only when an id has members in BOTH tiers — that signals a
        # crash-mid-resolve or backup-restore scenario requiring reconciliation.
        id_to_paths: dict[str, list[Path]] = {}
        for path in paths:
            id_to_paths.setdefault(path.stem, []).append(path)
        for esc_id, esc_paths in id_to_paths.items():
            has_root = any(p.parent == self.queue_dir for p in esc_paths)
            has_archive = any(p.parent != self.queue_dir for p in esc_paths)
            if has_root and has_archive:
                logger.warning(
                    f'Escalation {esc_id!r} exists in both queue_dir and archive: '
                    f'{[str(p) for p in esc_paths]}; reconciliation may be needed'
                )

        seen: set[str] = set()
        results = []
        for path in paths:
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.task_id != task_id:
                    continue
                if status is not None and esc.status != status:
                    continue
                if level is not None and esc.level != level:
                    continue
                if agent_role is not None and esc.agent_role != agent_role:
                    continue
                if esc.id in seen:
                    logger.debug(
                        f'Duplicate escalation id {esc.id!r} at {path}; '
                        'skipping (pre-scan WARNING already logged; queue_dir copy takes precedence)'
                    )
                    continue
                seen.add(esc.id)
                results.append(esc)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse {path}: {e}')
                continue
        return results

    def has_open_l1(self, task_id: str, *, category: str | None = None) -> bool:
        """Return True when the task has at least one pending level-1 escalation.

        Level-1 is the handed-to-human tier: the presence of one signals that
        the workflow must not auto-requeue the task — a human will unblock.

        ``category``, when given, restricts the check to open L1s whose
        ``.category`` matches — a signature filter (task 2757).  This lets a NEW
        root cause (a different failure category) escape being silently
        suppressed by an UNRELATED open L1.  Default None preserves the exact
        prior behavior (any pending L1 counts).
        """
        open_l1s = self.get_by_task(task_id, status='pending', level=1)
        if category is not None:
            open_l1s = [esc for esc in open_l1s if esc.category == category]
        return bool(open_l1s)

    def find_terminal_by_citation(
        self, task_id: str, category: str, citation_sha: str | None,
    ) -> Escalation | None:
        """Newest TERMINAL record matching ``(task_id, category, citation_sha)``, or None.

        A PURE READ with no side effects.  It answers one question: *was this
        exact evidence already adjudicated?*  ``has_open_l1`` above answers the
        complementary one — *is a duplicate still OPEN?* — and reads PENDING
        records only.  The two are deliberately disjoint: a PENDING match is
        NOT reported here, because that case is already ``has_open_l1``'s
        contract and reporting it in both places would blur which guard fired.

        Why a terminal-record read exists at all (task 4499): the
        ``provenance_unattributed`` reject condition is ABSORBING — main only
        moves forward, so evidence that stopped surviving stays gone.  Once the
        auto-watcher resolves the L1, the pending-only guard goes False and the
        next tick refiles the identical finding, forever: a close-then-refile
        ping-pong.  Matching the citation sha carried on the RESOLUTION makes
        "already adjudicated" answerable from the archive, so the refile can be
        suppressed while genuine NEW evidence (a different sha, or none at all)
        still gets through.

        A falsy *citation_sha* short-circuits to None — the
        ``find_dedupe_parent`` falsy-key convention.  An absent identity is not
        an identity, and suppressing on one would collapse unrelated findings.

        Cost — why the TARGETED per-task glob (``esc-{task_id}-*.json`` over
        the queue root plus ``_iter_archive_paths``, the shape
        ``_recover_seq_from_disk`` already uses) rather than
        ``get_by_task``'s full-archive ``esc-*.json`` rglob: this runs on the
        REJECT path, which is exactly the path that recurs every tick in the
        storm case being fixed, over an archive shared across 7+ projects.  An
        O(whole-archive) parse per tick would be the wrong bound.

        The ``.json`` suffix is load-bearing for the same three reasons
        ``_recover_seq_from_disk`` lists: it excludes the
        ``esc-{task_id}{SEQ_COUNTER_SUFFIX}`` counter, the ``{id}.json.lock``
        sidecars, and submit's ``{id}.tmp`` staging files.  The glob
        OVER-matches sibling hyphenated task ids (``esc-1-2-3-9.json`` when
        scanning task ``'1-2'``), so the record's own ``task_id`` field — never
        a filename parse — is the authoritative filter.
        """
        if not citation_sha:
            return None

        pattern = f'esc-{task_id}-*.json'
        newest: Escalation | None = None
        newest_ts = datetime.min.replace(tzinfo=UTC)
        for path in itertools.chain(
            self.queue_dir.glob(pattern), self._iter_archive_paths(pattern),
        ):
            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse {path}: {e}')
                continue
            # task_id is authoritative — the glob over-matches hyphenated siblings.
            if esc.task_id != task_id:
                continue
            if esc.category != category:
                continue
            if esc.citation_sha != citation_sha:
                continue
            if esc.status not in ('resolved', 'dismissed'):
                continue
            # Malformed stamps sort OLDEST (never dropped, never displacing a
            # well-formed newer match) and are logged loudly, not swallowed.
            ts, _ = parse_timestamp_or_warn(
                esc.resolved_at,
                fallback=datetime.min.replace(tzinfo=UTC),
                context='queue.find_terminal_by_citation',
            )
            # Strictly-newer displaces, so a TIE keeps the incumbent — and the
            # queue root is scanned first, matching get_by_task's "queue_dir
            # copy takes precedence" rule for a root/archive duplicate pair.
            if newest is None or ts > newest_ts:
                newest, newest_ts = esc, ts

        return newest

    def get_pending(self) -> list[Escalation]:
        """Get all pending escalations."""
        results = []
        for path in self.queue_dir.glob('esc-*.json'):
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.status == 'pending':
                    results.append(esc)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse {path}: {e}')
                continue
        return results

    def resolve(
        self, escalation_id: str, resolution: str, dismiss: bool = False,
        *, resolved_by: str | None = None, resolution_turns: int | None = None,
        resolution_class: str | None = None,
        granted_files: list[str] | None = None,
    ) -> Escalation | None:
        """Update an escalation's status to resolved or dismissed.

        ``granted_files`` (task 2505): an optional structured scope-expansion
        grant — a list of file-level, project-relative paths — stamped onto
        the record when not ``None``. Distinct from ``resolution``, which
        stays free-text human-readable rationale. Not propagated through the
        member cascade below: grants are L0 ``scope_violation`` escalations,
        which carry empty ``members`` lists.

        ``resolution_class`` (plans/escalation-lifecycle-dashboard-prd.md Seam 1):
        an optional explicit ``'benign' | 'actionable'`` classification stamp,
        validated against ``RESOLUTION_CLASSES`` BEFORE the lock is taken —
        an invalid value raises ``ValueError`` naming the legal values and
        nothing is persisted (INV-1). When ``None``, the stamp defaults per
        the caller's ``resolved_by`` (see ``escalation.classify.
        default_resolution_class_for_resolver``).

        Concurrency: the get → status-check → mutate → _atomic_write →
        _archive_resolved critical section is serialized under
        ``escalation_id_lock``.  Moving the idempotency status-check INSIDE
        the lock ensures that two concurrent resolves for the same id
        serialize and produce exactly one archive copy.

        Callbacks and cascade run AFTER releasing the lock:
        - ``_resolve_callback`` fires after the lock is released, preventing
          re-entrant callback → resolve deadlocks.
        - The member cascade calls ``self.resolve(member_id, …)`` for each L1
          member; each cascaded call takes its OWN distinct sidecar lock
          (different file → different inode → no same-fd re-entrancy/deadlock).

        Cascade: if the escalation is an L2 with a non-empty ``members`` list,
        after archiving the L2 each member id is resolved recursively with the
        same *resolution* text.  The cascade uses
        ``resolved_by='l2-cascade:{escalation_id}'`` so the audit trail
        distinguishes direct resolves from cascades.  *dismiss* is propagated
        so a dismissed L2 dismisses its members too.

        Cascade contract:
        - Recursion terminates because L1 members carry empty ``members`` lists.
        - Per-member exceptions are caught and logged; a single bad member id
          never blocks the remaining cascade.
        - ``resolve()`` is idempotent, so cascading to an already-resolved member
          is a safe no-op.

        Ordering note (for ``_resolve_callback`` consumers):
        ``_resolve_callback`` fires for the L2 **before** the cascade to member
        L1s begins.  Consumers of the callback that inspect member state will see
        members still pending at callback time; they should re-query member state
        rather than assuming terminality.  This ordering is stable — do not rely
        on members being resolved at the moment the L2 callback fires.
        """
        if resolution_class is not None and resolution_class not in RESOLUTION_CLASSES:
            raise ValueError(
                f'invalid resolution_class {resolution_class!r}; '
                f'expected one of {sorted(RESOLUTION_CLASSES)} or None'
            )

        with escalation_id_lock(self.queue_dir, escalation_id):
            esc = self.get(escalation_id)
            if esc is None:
                return None

            if esc.status != 'pending':
                logger.info(
                    f'Escalation {escalation_id} already {esc.status}; resolve() is a no-op'
                )
                return esc

            esc.status = 'dismissed' if dismiss else 'resolved'
            esc.resolution = resolution
            esc.resolved_at = datetime.now(UTC).isoformat()
            if resolved_by is not None:
                esc.resolved_by = resolved_by
            if resolution_turns is not None:
                esc.resolution_turns = resolution_turns
            esc.resolution_class = (
                resolution_class if resolution_class is not None
                else default_resolution_class_for_resolver(resolved_by)
            )
            if granted_files is not None:
                esc.granted_files = granted_files

            self._atomic_write(escalation_id, esc.to_json())
            self._archive_resolved(escalation_id, esc.resolved_at)

        logger.info(f'Escalation {escalation_id} {esc.status}: {resolution[:100]}')

        if self._resolve_callback:
            try:
                self._resolve_callback(esc)
            except Exception as e:
                logger.warning(f'Resolve callback failed for {escalation_id}: {e}')

        # Cascade to member L1s (L2 clusters only — L0/L1 have empty members).
        # Recursion is bounded: L1 members carry empty members[], so no deeper nesting.
        if esc.members:
            cascade_resolved_by = f'l2-cascade:{escalation_id}'
            for member_id in esc.members:
                try:
                    self.resolve(
                        member_id,
                        resolution,
                        dismiss=dismiss,
                        resolved_by=cascade_resolved_by,
                        resolution_class=esc.resolution_class,
                    )
                except Exception as e:
                    logger.warning(
                        'cascade: failed to resolve member %s of L2 %s: %s',
                        member_id, escalation_id, e,
                    )

        return esc

    def park(
        self,
        escalation_id: str,
        resolution: str,
        *,
        resolved_by: str | None = None,
        resolution_turns: int | None = None,
    ) -> Escalation | None:
        """Park an escalation: promote to level=2, stamp resolution_action='park',
        persist resolution text, keep status='pending' (NOT archived).

        Differs from resolve() in three ways:
        - status stays 'pending' (escalation remains OPEN — sweep-quiescence mechanism)
        - level is promoted to 2 (L2 = human-only consumer)
        - _archive_resolved is NOT called (record stays live in queue_dir)
        - resolved_at is NOT set (preserving the resolved_at=None ↔ status='pending' invariant)

        After releasing the per-id lock, fires _resolve_callback(esc) to trigger the harness
        teardown (kill workflow + set task 'blocked') — same callback, different escalation state.

        For cluster L2s: fires _resolve_callback per member L1 with an in-memory
        resolved_by='l2-cascade:<L2-id>' stamp (does NOT persist/archive member records —
        members stay pending L1s covering their tasks).

        Idempotent: returns existing record unchanged if either (a) status != 'pending'
        (already resolved/dismissed) or (b) resolution_action == 'park' AND resolution
        is not None (already parked — resolution text was written by a prior park() call).
        The combined second guard prevents a re-park from re-firing teardown callbacks
        without falsely short-circuiting a first-time park() on an L2 whose
        resolution_action was pre-set to 'park' but resolution is still None.
        The first guard matches the contract of resolve().
        """
        with escalation_id_lock(self.queue_dir, escalation_id):
            esc = self.get(escalation_id)
            if esc is None:
                return None

            if esc.status != 'pending' or (
                esc.resolution_action == 'park' and esc.resolution is not None
            ):
                logger.info(
                    'Escalation %s already %s (resolution_action=%r); park() is a no-op',
                    escalation_id, esc.status, esc.resolution_action,
                )
                return esc

            # Promote to L2, stamp park metadata, keep status='pending'.
            esc.level = 2
            esc.resolution_action = 'park'
            esc.resolution = resolution
            if resolved_by is not None:
                esc.resolved_by = resolved_by
            if resolution_turns is not None:
                esc.resolution_turns = resolution_turns
            # resolved_at stays None — parked escalation is still open/pending.

            # In-place rewrite (no archive — record stays live in queue_dir).
            self._atomic_write(escalation_id, esc.to_json())

        logger.info('Escalation %s parked (kept open at L2): %s', escalation_id, resolution[:100])

        # Fire resolve callback for the L2 itself to trigger harness teardown.
        if self._resolve_callback:
            try:
                self._resolve_callback(esc)
            except Exception as e:
                logger.warning('Park callback failed for %s: %s', escalation_id, e)

        # Cluster L2: fire per-member callback with in-memory l2-cascade stamp.
        # Do NOT persist or resolve member records — members stay pending L1s.
        if esc.members:
            for member_id in esc.members:
                try:
                    member = self.get(member_id)
                    if member is None:
                        logger.warning(
                            'park cascade: member %s of L2 %s not found — skipping',
                            member_id, escalation_id,
                        )
                        continue
                    if member.status != 'pending':
                        # Mirror resolve()'s idempotent-member contract: skip members
                        # that are already resolved/dismissed so that a terminal member
                        # does not have its task driven to 'blocked' after closure.
                        logger.debug(
                            'park cascade: member %s of L2 %s already %s — skipping',
                            member_id, escalation_id, member.status,
                        )
                        continue
                    # In-memory cascade signal only; no disk write.
                    member.resolved_by = f'l2-cascade:{escalation_id}'
                    if self._resolve_callback:
                        try:
                            self._resolve_callback(member)
                        except Exception as e:
                            logger.warning(
                                'Park cascade callback failed for member %s of L2 %s: %s',
                                member_id, escalation_id, e,
                            )
                except Exception as e:
                    logger.warning(
                        'park cascade: error processing member %s of L2 %s: %s',
                        member_id, escalation_id, e,
                    )

        return esc

    def submit_resolved(
        self,
        escalation: Escalation,
        resolution: str,
        *,
        resolved_by: str | None = None,
        resolution_class: str | None = None,
    ) -> Escalation:
        """Atomically write *escalation* directly in resolved state.

        Unlike the two-call ``submit()`` + ``resolve()`` pattern, this helper
        performs a single file write (no transient pending intermediate) and
        fires only ``_resolve_callback`` — never ``_notify_callback``.  This
        eliminates the spurious "pending escalation" wake that the two-call
        path emits before the resolve callback fires.

        ``resolution_class`` mirrors ``resolve()``'s Seam-1 stamp (see its
        docstring): an optional explicit ``'benign' | 'actionable'`` value,
        validated against ``RESOLUTION_CLASSES`` before *escalation* is
        mutated at all — an invalid value raises ``ValueError`` naming the
        legal values and leaves the passed-in object untouched (INV-1).  When
        ``None``, the stamp defaults per *resolved_by* via the same
        ``default_resolution_class_for_resolver`` helper ``resolve()`` uses
        (INV-5 — one classification site for both terminal-write paths).

        Contract:
        - Mutates *escalation* in-place (status/resolution/resolved_at/resolved_by)
          **before** the disk write.  If the write raises (e.g. ENOSPC), the in-memory
          object is already in resolved state — callers must not reuse it after catching
          the exception.  Treat the input as consumed on call.
        - Writes the resolved JSON atomically to ``queue_dir/{id}.json`` using
          the same tmp+rename pattern as ``submit()``.
        - Best-effort archives via ``os.replace`` into the dated archive subdir
          (same two-step design as ``resolve()``): if the archive move fails,
          the resolved file stays in ``queue_dir`` where ``get()`` can find it.
        - Fires ``_resolve_callback`` once (if set); never fires ``_notify_callback``.
        - Returns *escalation* (non-Optional — disk failures raise, no None case).
        """
        if resolution_class is not None and resolution_class not in RESOLUTION_CLASSES:
            raise ValueError(
                f'invalid resolution_class {resolution_class!r}; '
                f'expected one of {sorted(RESOLUTION_CLASSES)} or None'
            )

        # Step 1: mutate in memory
        escalation.status = 'resolved'
        escalation.resolution = resolution
        escalation.resolved_at = datetime.now(UTC).isoformat()
        if resolved_by is not None:
            escalation.resolved_by = resolved_by
        escalation.resolution_class = (
            resolution_class if resolution_class is not None
            else default_resolution_class_for_resolver(resolved_by)
        )

        # Step 2: atomic write + best-effort archive under the per-id lock.
        # The lock prevents a concurrent same-id RMW from clobbering this write.
        # The resolve callback fires OUTSIDE the lock (see lock-scope design decision).
        with escalation_id_lock(self.queue_dir, escalation.id):
            self._atomic_write(escalation.id, escalation.to_json())
            # Step 3: best-effort archive move
            self._archive_resolved(escalation.id, escalation.resolved_at)

        # Step 4: log
        logger.info(
            'Escalation submit_resolved: %s [%s]: %s',
            escalation.id, escalation.severity, resolution[:100],
        )

        # Step 5: fire resolve callback only — never notify callback
        if self._resolve_callback:
            try:
                self._resolve_callback(escalation)
            except Exception as e:
                logger.warning(f'Resolve callback failed for {escalation.id}: {e}')

        return escalation

    def find_pending_l2_by_root_cause(self, root_cause: str) -> str | None:
        """Return the id of the oldest pending L2 escalation whose root_cause matches.

        Matching is on the CANONICAL FORM, not the raw string (task 3998).  Two
        promotes describing one cause but spelled differently — differing only in
        case, in whitespace runs or in punctuation — used to mint two L2s for one
        incident; they now fold into the first.

        THE CALLER CONTRACT WIDENED WITH THAT CHANGE, and a RESOLVE-by-key caller
        must account for it.  This used to answer "the pending L2 filed under
        EXACTLY this key"; it now answers "the OLDEST pending L2 whose key is
        CANONICALLY EQUAL to this one" — which is what the fold path wants, but is
        strictly weaker than identity.  The one resolve-by-root-cause site in the
        repo, ``orchestrator.harness.Harness._resolve_watcher_outage_l2``, feeds
        the returned id straight into ``resolve(..., 'watcher recovered')``, so a
        pending L2 filed under a case- or trailing-punctuation variant of
        ``watcher_supervisor_down`` (``WATCHER_SUPERVISOR_DOWN``,
        ``watcher_supervisor_down.``) would be auto-resolved by a supervisor that
        never filed it.  Unrealised today — that constant has a single spelling in
        the repo and the live corpus holds no variant of it, and the plausible
        hyphenated re-spelling ``watcher-supervisor-down`` does NOT fold anyway
        (``_`` is a word character and survives, ``-`` becomes a separator, so the
        two canonicalise apart) — but the widening is real.  A caller that wants
        IDENTITY rather than a fold must re-check ``esc.root_cause`` against its
        own key on the returned record; this function deliberately grows no
        exact-match mode, because one matcher with one policy is the whole point
        (INV-5).

        Algorithm:
        1. Canonicalise *root_cause* via
           :func:`escalation.canonical.canonical_root_cause` — THE single
           canonicalisation site, shared with dedupe's two normalisation
           consumers.  Return None immediately when the CANONICAL form is empty
           (the falsy-key guard — mirrors ``dedupe.find_dedupe_parent``'s
           convention).  Note this is stricter than ``.strip()``: an
           all-punctuation key like ``'::'`` survives stripping but canonicalises
           to nothing, and such a key can never identify a cluster.
        2. Iterate ``self.get_pending()``, filtering to ``level == 2`` and
           ``canonical_root_cause(esc.root_cause) == candidate``.  BOTH SIDES are
           canonicalised at compare time; the record's own ``root_cause`` is
           stored and displayed VERBATIM and is never overwritten with the
           canonical form — an L2's framing is human-facing and immutable, and a
           persisted derived value could silently desynchronise from the function
           that derives it.
        3. Among matches, return the id of the entry with the OLDEST timestamp
           (ISO 8601 string comparison; malformed timestamps fall back to
           ``datetime.min`` so they sort first — never silently lost).

        PUNCTUATION IS A SEPARATOR, NOT DELETED (``a.b`` -> ``a b``): root_cause
        keys are delimiter-dense identity strings — 69% of the distinct live keys
        carry a digit adjacent to a separator (task ids, SHA prefixes, dates) —
        so deleting the separator would glue ``risk:3184`` onto ``risk:318:4``
        and silently merge two distinct incidents into one L2.  The full
        measurement and the asymmetric-cost argument are in
        :mod:`escalation.canonical`'s docstring.

        Cost: O(N) over pending escalations, where N is the current queue depth.
        Acceptable at current escalation rates; mirrors the existing
        ``find_dedupe_parent`` O(N) pattern in dedupe.py.
        """
        candidate = canonical_root_cause(root_cause)
        if not candidate:
            return None

        oldest_id: str | None = None
        oldest_ts: datetime = datetime.max.replace(tzinfo=UTC)

        for esc in self.get_pending():
            if esc.level != 2:
                continue
            if canonical_root_cause(esc.root_cause) != candidate:
                continue
            # Parse timestamp; fall back to datetime.min on bad input so malformed
            # entries are treated as oldest (never silently dropped). Emits a WARNING
            # (loud-over-silent) via parse_timestamp_or_warn when the timestamp is bad.
            ts, _ = parse_timestamp_or_warn(
                esc.timestamp,
                context='queue.find_pending_l2_by_root_cause',
            )
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_id = esc.id

        return oldest_id

    def add_members_to_l2(
        self, escalation_id: str, new_member_ids: list[str],
        *, severity_floor: str | None = None,
        root_cause: str = '', evidence: str = '', options: list[str] | None = None,
        summary: str = '', agent_role: str = '',
        outcome: AmendmentOutcome | None = None,
    ) -> Escalation | None:
        """Append *new_member_ids* to a pending L2 escalation's ``members`` list.

        **Concurrency contract (sidecar flock).**  The entire read-modify-write
        of ``members`` is serialized per-id by ``escalation_id_lock``, so
        concurrent appends from multiple processes are union-preserving — no
        write clobbers another's additions.  The lock target is the stable
        sidecar ``{escalation_id}.json.lock`` (see ``escalation_id_lock`` for
        the PRD-D3 rationale).

        Loads the L2 directly from ``queue_dir/{escalation_id}.json`` (queue root
        only).  This refuses archived L2s — they were already adjudicated by a
        human; re-opening them via member append would resurrect a closed decision
        (the same Defect-2 class of bug that motivated task 1498).  For the same
        reason this method never falls back to the archive.

        Set-union semantics: member ids already present in ``esc.members`` are
        not added again.  *new_member_ids* is also deduplicated internally via
        ``dict.fromkeys`` so passing ``['a', 'a', 'b']`` adds 'a' exactly once.
        New ids are appended in the order they first appear in *new_member_ids*.

        ``members`` is modified, and — when *severity_floor* is given —
        ``severity`` is additionally promoted UPWARD via ``max_severity``.
        The invariant is that **an L2's severity is monotonically
        non-decreasing after mint**: the floor can raise the record but never
        lower it, so it can only ever add human attention, never suppress it.
        Omitting *severity_floor* leaves ``severity`` untouched (task 3976).
        **Incoming framing is PRESERVED, not discarded (task 3997).**  The
        record's own ``root_cause``, ``options``, ``summary``, ``detail`` and
        ``timestamp`` are still never OVERWRITTEN — the human-facing decision
        context stays the L2's original framing across repeated auto-watcher
        triage passes.  But the framing a fold carries IN is no longer dropped
        on the floor either (measured: 336,875 characters lost): when any of
        *root_cause* / *evidence* / *options* / *summary* is non-empty, one
        :class:`~escalation.models.Amendment` is APPENDED to ``esc.amendments``
        recording it, alongside *agent_role* and a ``timestamp`` this method
        stamps itself (the write chokepoint owns its clock, mirroring
        ``stamp_triage``/``triaged_at``, so a caller cannot backdate one).
        The incoming *evidence* is stored under the amendment's ``detail`` key —
        the same field the create path writes that argument into.  Passing no
        framing appends no amendment, so every existing two-positional-arg
        caller is byte-unchanged.  Framing byte-identical to what the record
        ALREADY says is likewise not re-recorded (see
        :func:`_is_repeat_framing`): a re-promote with unchanged text has
        contributed nothing, so it stays a true no-op and does NOT bump
        ``updated_at``.  "What the record already says" is the LAST amendment
        when there is one and **the record's OWN framing when there is not** —
        so the FIRST re-promote of a freshly-created L2, the one most likely to
        re-send the create's exact text, is a no-op too rather than a verbatim
        copy of the record's own fields.

        **Each entry is size-bounded too.**  Every free-text field is elided to
        its named per-field cap (``_MAX_AMENDMENT_LINE_CHARS`` /
        ``_MAX_AMENDMENT_DETAIL_CHARS``) with an in-band marker, and the options
        list is capped at ``_MAX_AMENDMENT_OPTIONS`` — without that, the entry
        count alone would bound nothing, since *evidence* is unbounded free
        text.  Characters dropped are added to ``amendments_chars_elided``,
        the byte-side counterpart of ``amendments_truncated``.

        **The list is capped at** :data:`_MAX_AMENDMENTS`.  THIS METHOD is the
        trimmer, at write time, in the same critical section and the same single
        ``_rewrite`` as the append — so cap enforcement is atomic with it and no
        over-cap list is ever durable.  It sheds the OLDEST entries, which is
        safe because the ORIGINAL framing is never in this list at all: it lives
        permanently in the record's own immutable ``root_cause``/``detail``/
        ``options``/``summary``.  The oldest amendment is therefore the
        least-informative thing to shed, and a rotation triaging now needs the
        most recent framing.  Every dropped entry increments
        ``amendments_truncated`` and the event logs a WARNING, so the loss is a
        durable structured fact rather than a silent one.

        A severity promotion is a real content change, so it bumps
        ``updated_at`` even when no new member id was appended — the watcher's
        stamp-then-skip protocol keys off ``updated_at > triaged_at``, and a
        record that silently got more severe would otherwise be skipped
        forever.  A recorded amendment bumps it for the identical reason: new
        framing IS the re-assess trigger.  A floor at or below the current
        severity, with no new members and no framing, remains a true no-op and
        does NOT bump.

        Both the member append and the severity bump happen inside the same
        ``escalation_id_lock`` and land in a single ``_rewrite``, so they are
        atomic together — a second write after the fact would be racy against a
        concurrent append.

        Returns the updated ``Escalation`` (or the unchanged escalation when
        there is nothing to append and no floor to apply).  Returns ``None``
        when *escalation_id* is not found in the queue root (unknown id or
        archived).

        **Distinct root-cause spellings are tracked (task 3998).**  Since the
        pending-L2 lookup matches on the CANONICAL form, a fold can arrive
        spelled differently from the record's own ``root_cause``.  Every such
        PRE-canonical spelling is accumulated in ``esc.root_cause_variants``
        (the record's own spelling seeded first), because canonicalisation's
        failure mode is OVER-folding — distinct causes silently merged under one
        canonical key — and the set of mutually-distinct spellings one cluster
        has been addressed by is that failure's only observable signature.  This
        method is the SOLE writer and sole trimmer: the list is oldest-shed at
        ``_MAX_ROOT_CAUSE_VARIANTS`` with every drop counted in
        ``root_cause_variants_truncated``, so the TRUE distinct count
        (``len(variants) + truncated``) stays readable from the record and never
        plateaus.  It lands in the SAME single ``_rewrite`` as everything else
        above.

        Pass *outcome* to learn what this call did to ``amendments`` and to
        ``root_cause_variants`` — whether it recorded an amendment, how many it
        shed, whether it added a new spelling, and the running distinct total —
        as computed HERE, inside the lock, rather than inferred by a caller from
        a racy pre-read.  See :class:`AmendmentOutcome`.  It is filled on every
        return path.
        """
        # Defaults FIRST: the two early returns below leave them as-is, so the
        # caller's dict is complete no matter which path this call takes.
        if outcome is not None:
            outcome['recorded'] = False
            outcome['dropped'] = 0
            outcome['variant_added'] = False
            outcome['variants'] = 0
        with escalation_id_lock(self.queue_dir, escalation_id):
            path = self.queue_dir / f'{escalation_id}.json'
            if not path.exists():
                return None
            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse L2 escalation {escalation_id}: {e}')
                return None

            incoming_framing = bool(root_cause or evidence or options or summary)
            if not new_member_ids and severity_floor is None and not incoming_framing:
                return esc  # no-op

            existing = set(esc.members)
            appended = [m for m in dict.fromkeys(new_member_ids) if m not in existing]

            # Upward-only: max_severity can only return the higher-ranked of
            # the two, so a floor at or below the current severity is inert by
            # construction.  An unrecognised floor falls to rank 0 there (with
            # a WARNING) and likewise leaves the record alone.
            new_severity = (
                max_severity(esc.severity, severity_floor)
                if severity_floor is not None
                else esc.severity
            )
            severity_changed = new_severity != esc.severity

            # Preserve the incoming framing rather than discarding it.  Built
            # inside the SAME escalation_id_lock critical section as the member
            # append, so it lands in the same single _rewrite below — no second
            # write path, no new durability story.
            amendment_recorded = False
            dropped_entries = 0

            # Track the DISTINCT pre-canonical root_cause spellings this cluster
            # has been addressed by (task 3998).  Canonicalising the match makes
            # more promotes fold BY DESIGN, so the failure it introduces is
            # OVER-folding — distinct causes silently merged under one canonical
            # key — and this list is that failure's only observable signature.
            # Written in the SAME critical section and folded into the SAME
            # single _rewrite as the member append and the amendment: no second
            # write path, no new durability story, and cross-process
            # union-preservation is inherited from the lock rather than
            # re-established.
            variant_added = False
            # STRIPPED, exactly as `_framing_view` compares root_cause and as the
            # create path stores it: a key differing only in surrounding
            # whitespace was folded by the lookup even BEFORE canonicalisation,
            # so it is not a distinct SPELLING and carries no over-fold
            # information.  Recording it would also contradict the pinned
            # contract that such a fold is a no-op (no spurious `updated_at`
            # bump, hence no spurious re-assess trigger).
            incoming_variant = root_cause.strip()
            if incoming_variant:
                # Seed the record's OWN spelling first, so the count reflects
                # every spelling the cluster has EVER been addressed by rather
                # than only the post-mint ones.
                if not esc.root_cause_variants and esc.root_cause.strip():
                    seeded, _ = _elide(esc.root_cause.strip(), _MAX_AMENDMENT_LINE_CHARS)
                    esc.root_cause_variants.append(seeded)
                # Elide through the SAME helper as amendment fields, so an
                # over-long spelling is capped with the in-band marker rather
                # than silently truncated — and so the comparison below is
                # against the STORED form, exactly as _is_repeat_framing does.
                incoming, _ = _elide(incoming_variant, _MAX_AMENDMENT_LINE_CHARS)
                if incoming not in esc.root_cause_variants:
                    esc.root_cause_variants.append(incoming)
                    variant_added = True
                    if len(esc.root_cause_variants) > _MAX_ROOT_CAUSE_VARIANTS:
                        shed = len(esc.root_cause_variants) - _MAX_ROOT_CAUSE_VARIANTS
                        del esc.root_cause_variants[:shed]
                        esc.root_cause_variants_truncated += shed
                        logger.warning(
                            'add_members_to_l2: %s shed %d oldest root_cause '
                            'variant(s) at the _MAX_ROOT_CAUSE_VARIANTS=%d cap '
                            '(running total truncated=%d); the TRUE distinct '
                            'count is still len(variants)+truncated=%d',
                            escalation_id, shed, _MAX_ROOT_CAUSE_VARIANTS,
                            esc.root_cause_variants_truncated,
                            len(esc.root_cause_variants)
                            + esc.root_cause_variants_truncated,
                        )

            if incoming_framing:
                candidate, chars_elided = _build_amendment(
                    root_cause=root_cause, summary=summary, evidence=evidence,
                    options=options, agent_role=agent_role,
                    timestamp=datetime.now(UTC).isoformat(),
                )
                # Built BEFORE the repeat check, and the check reads the built
                # entry: the stored form is what a later fold will be compared
                # against, so comparing anything else would let two entries the
                # record cannot tell apart both be recorded.
                if not _is_repeat_framing(esc, candidate):
                    esc.amendments.append(candidate)
                    amendment_recorded = True
                    # Count what the per-field caps dropped on the record, for
                    # the same reason shed ENTRIES are counted below: a reader
                    # must be able to tell a whole framing from the head of one
                    # without scraping logs (INV-8).
                    if chars_elided:
                        esc.amendments_chars_elided += chars_elided
                        logger.warning(
                            'add_members_to_l2: %s elided %d char(s) of incoming '
                            'framing at the per-field amendment caps (running '
                            "total elided=%d); the record's own framing is "
                            'unaffected',
                            escalation_id, chars_elided,
                            esc.amendments_chars_elided,
                        )
                    # Enforce the cap in the SAME critical section, so the trim
                    # lands in the same single _rewrite as the append and there
                    # is no durable window in which an over-cap list exists.
                    if len(esc.amendments) > _MAX_AMENDMENTS:
                        dropped_entries = len(esc.amendments) - _MAX_AMENDMENTS
                        del esc.amendments[:dropped_entries]
                        esc.amendments_truncated += dropped_entries
                        logger.warning(
                            'add_members_to_l2: %s shed %d oldest amendment(s) at '
                            'the _MAX_AMENDMENTS=%d cap (running total '
                            "truncated=%d); the record's own original framing is "
                            'unaffected',
                            escalation_id, dropped_entries, _MAX_AMENDMENTS,
                            esc.amendments_truncated,
                        )

            # A new variant is a real content change, so it joins the existing
            # write condition rather than getting a write of its own — a fold
            # that ONLY re-spells the cause must still be durable.
            if appended or severity_changed or amendment_recorded or variant_added:
                esc.members.extend(appended)
                esc.severity = new_severity
                # Bump the "changed since triaged" signal — a member append, a
                # severity promotion or a new framing amendment is exactly the
                # re-assess trigger the watcher's stamp-then-skip protocol keys
                # off (updated_at > triaged_at).
                esc.updated_at = datetime.now(UTC).isoformat()
                self._rewrite(escalation_id, esc)
                logger.info(
                    'add_members_to_l2: added %d new member(s) to %s '
                    '(total=%d, severity=%s%s, amendments=%d)',
                    len(appended), escalation_id, len(esc.members), esc.severity,
                    ' [promoted]' if severity_changed else '', len(esc.amendments),
                )
            # Reported from INSIDE the lock, so it describes THIS call's write
            # and cannot be invalidated by a concurrent fold.
            if outcome is not None:
                outcome['recorded'] = amendment_recorded
                outcome['dropped'] = dropped_entries
                outcome['variant_added'] = variant_added
                # The TRUE distinct total, not the list length: past the cap the
                # list stops growing, and reporting its length would make the
                # over-fold signal plateau exactly when it matters most.
                outcome['variants'] = (
                    len(esc.root_cause_variants) + esc.root_cause_variants_truncated
                )
            return esc

    def stamp_triage(
        self, escalation_id: str, *, triaged_by: str | None = None, triage_note: str = '',
    ) -> Escalation | None:
        """Stamp a triage-ack ANNOTATION on a pending escalation.

        This is deliberately NOT a resolution — it records that a watcher
        assessed the item (so future rotations can skip re-deriving its
        disposition) without changing ``status``, ``level``, or archiving it.
        A {0,1}-level-capped connection that is forbidden to resolve a pending
        L2 can still stamp triage on it (see server.stamp_triage — ungated).

        **Concurrency contract (sidecar flock).**  Serialized per-id by
        ``escalation_id_lock``, mirroring ``add_members_to_l2`` /
        ``attach_dedupe_child``.

        Loads the record directly from ``queue_dir/{escalation_id}.json``
        (queue root only) — does NOT use ``self.get()``, which falls back to
        the archive. This refuses archived/resolved records: annotating a
        closed decision is meaningless, and loading via the archive fallback
        followed by ``_rewrite`` (which always targets the queue root) would
        resurrect an archived record into the queue root — the same Defect-2
        class of bug that motivated task 1498's ``add_members_to_l2`` guard.

        Sets ``triaged_at`` (now, UTC ISO). ``triaged_by`` is overwritten only
        when not ``None``, and ``triage_note`` is overwritten only when
        non-empty — a ``None``/omitted arg leaves the existing value of each
        field untouched. This symmetry is deliberate: a bare re-stamp (no new
        ``triaged_by``/``triage_note`` supplied) is a harmless freshness bump
        that refreshes ``triaged_at`` without silently wiping a
        previously-recorded predicate/probe. Does NOT touch ``updated_at``:
        an annotation must not masquerade as a content change (see
        ``add_members_to_l2`` for the one mutation that does bump
        ``updated_at``).

        Returns the updated ``Escalation``, or ``None`` when *escalation_id*
        is not found in the queue root, fails to parse, or is not pending
        (archived/resolved/dismissed).
        """
        with escalation_id_lock(self.queue_dir, escalation_id):
            path = self.queue_dir / f'{escalation_id}.json'
            if not path.exists():
                return None
            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
                return None

            if esc.status != 'pending':
                return None

            esc.triaged_at = datetime.now(UTC).isoformat()
            if triaged_by is not None:
                esc.triaged_by = triaged_by
            if triage_note:
                esc.triage_note = triage_note
            self._rewrite(escalation_id, esc)
            logger.info('stamp_triage: stamped triage ack on %s', escalation_id)
            return esc

    def attach_dedupe_child(
        self, parent_id: str, child_id: str, *, child_severity: str = 'info',
    ) -> Escalation | None:
        """Append *child_id* to the pending parent's dedupe_children list.

        **Concurrency contract (sidecar flock).**  All four mutations —
        ``dedupe_count``, ``dedupe_children``, ``severity`` and ``updated_at`` —
        are serialized per-id by ``escalation_id_lock``.  Concurrent attaches from multiple
        processes are therefore safe: no child is lost, no count is reverted,
        and no severity promotion is undone.  The lock target is the stable
        sidecar ``{parent_id}.json.lock`` (see ``escalation_id_lock`` for the
        PRD-D3 rationale; different parent ids take different sidecars —
        no cross-id contention).

        Loads the parent directly from ``queue_dir/{parent_id}.json`` — it does
        NOT fall back to the archive.  This ensures that resolved / dismissed
        parents (which have been moved to the archive) are treated as
        not-eligible and return ``None`` without any mutation.

        On a successful attach:
        - ``parent.dedupe_children`` gains *child_id* (appended), and is capped
          at ``_MAX_DEDUPE_CHILDREN`` entries: past the cap the OLDEST NON-HEAD
          ids are shed (the first ``_MAX_DEDUPE_CHILDREN_HEAD`` are always
          retained), each drop counted in ``parent.dedupe_children_truncated``
          so the TRUE provenance total stays readable from the record.
        - ``parent.dedupe_count`` is incremented by 1.
        - ``parent.severity`` is promoted via
          ``max_severity(parent.severity, child_severity)``; never demoted.
        - ``parent.updated_at`` is stamped — the watcher's "changed since I
          triaged it" signal.  A fold raises the recurrence count and can
          promote severity, so a folded parent that still read
          ``updated_at is None`` would make an L1/L2 rotation applying the
          "newer than triaged_at" re-verify rule trust a stale triage note and
          skip a record that has materially changed.
        - The updated parent is written back to disk via ``_rewrite()``.  Only
          this final file-replace step is atomic (``tempfile.mkstemp`` +
          ``os.rename``); the preceding in-memory mutations are not.
        - The updated ``Escalation`` object is returned.

        Returns ``None`` when:
        - The parent file does not exist in the queue root (unknown id or
          already archived / resolved — refusing to touch archived parents).

        **Recon reuse contract (A7a/A7b):**  ``dedupe.submit_or_dedupe`` calls
        this function for *both* infra and recon dedup.  Two invariants hold:

        1. ``dedupe_count`` doubles as the recon *recurrence count* — each
           successful fold into a recon parent increments it by 1, so the
           steward can read ``dedupe_count`` as "this finding recurred N times".

        2. ``parent.dedupe_fingerprint`` is **preserved** across each
           ``_rewrite()`` call — this function does not touch it.  The
           invariant is that ``submit_or_dedupe`` can match successive folds
           to the same canonical parent by fingerprint without the parent ever
           losing its fingerprint identity after the first write.

        3. ``dedupe_count`` is NEVER capped, trimmed or reset.  It is the
           recurrence SIGNAL; ``dedupe_children`` is PROVENANCE, and only the
           latter is bounded (``_MAX_DEDUPE_CHILDREN``).  The cap separates the
           two deliberately — a drain that sorts the longest-rotting gates by
           ``dedupe_count``, and ``sweep._pick_richer`` which ranks on it, must
           not be silently blunted by a provenance bound.
        """
        with escalation_id_lock(self.queue_dir, parent_id):
            path = self.queue_dir / f'{parent_id}.json'
            if not path.exists():
                return None
            try:
                parent = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse parent escalation {parent_id}: {e}')
                return None
            parent.dedupe_children.append(child_id)
            # Incremented BEFORE the trim below purely so the shed WARNING can
            # report THIS attach's count: a message whose job is to assert
            # "dedupe_count is unaffected" must not quote a value one behind.
            parent.dedupe_count += 1
            # Enforce the cap in the SAME critical section, so the trim lands in
            # the same single _rewrite as the append and there is no durable
            # window in which an over-cap list exists.  HEAD-PRESERVING: the
            # first _MAX_DEDUPE_CHILDREN_HEAD ids established the fold and are
            # the only record of where this parent came from, so unlike
            # `amendments` this list is NOT shed purely oldest-first.  In steady
            # state `shed == 1`, so the del is O(cap) with no whole-list reslice.
            if len(parent.dedupe_children) > _MAX_DEDUPE_CHILDREN:
                shed = len(parent.dedupe_children) - _MAX_DEDUPE_CHILDREN
                del parent.dedupe_children[
                    _MAX_DEDUPE_CHILDREN_HEAD:_MAX_DEDUPE_CHILDREN_HEAD + shed
                ]
                parent.dedupe_children_truncated += shed
                logger.warning(
                    'attach_dedupe_child: %s shed %d oldest non-head child id(s) '
                    'at the _MAX_DEDUPE_CHILDREN=%d cap (running total '
                    'truncated=%d); the TRUE provenance total is still '
                    'len(children)+truncated=%d, and dedupe_count=%d is '
                    'unaffected',
                    parent_id, shed, _MAX_DEDUPE_CHILDREN,
                    parent.dedupe_children_truncated,
                    len(parent.dedupe_children) + parent.dedupe_children_truncated,
                    parent.dedupe_count,
                )
            parent.severity = max_severity(parent.severity, child_severity)
            # Bump the "changed since triaged" signal — a fold is a substantive
            # change (recurrence count up, possible severity promotion), which is
            # exactly the re-assess trigger the watcher's stamp-then-skip
            # protocol keys off (updated_at > triaged_at).
            # UNCONDITIONAL, unlike add_members_to_l2's guarded bump: that
            # function CAN be a genuine no-op (re-appending an already-present
            # member id), this one cannot — every successful call appends a child
            # and increments the count.  A "did anything change?" guard here could
            # only ever evaluate true, while implying to a reader that a silent
            # no-op path exists.
            parent.updated_at = datetime.now(UTC).isoformat()
            self._rewrite(parent_id, parent)
        logger.info(
            f'Dedupe: folded {child_id} into parent {parent_id} '
            f'(dedupe_count={parent.dedupe_count}, severity={parent.severity})'
        )
        return parent

    def patch_resolution_metadata(
        self,
        escalation_id: str,
        *,
        resolved_by: str | None = None,
        resolution_turns: int | None = None,
    ) -> Escalation | None:
        """Patch resolved_by and/or resolution_turns on an already-resolved escalation.

        Updates the existing file in place wherever it lives (root or archive); never
        resurrects an archived file into the queue root.  The tmp file used for the
        atomic write is created in the target file's parent directory, so the rename
        stays within the same filesystem subtree.

        Multi-date duplicates: when multiple archive copies exist for the same id, only
        the newest copy (newest parent directory name in YYYY-MM-DD lexicographic order)
        is patched; staler copies remain untouched.  Callers should clean up duplicate
        archive copies independently.

        Returns None when the escalation is missing OR still pending — caller must have
        already resolved/dismissed it.

        Returns the updated Escalation on success, or the unmodified Escalation when
        called with no patch arguments (both resolved_by and resolution_turns are None).
        """
        # Locate the file (shared helper — same root-first, archive-fallback,
        # newest-by-date logic as get()).
        path = self._locate_path(escalation_id)
        if path is None:
            return None

        # Parse
        try:
            esc = Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
            return None

        # Guard: only patch resolved/dismissed escalations
        if esc.status not in ('resolved', 'dismissed'):
            return None

        # No-op early return: nothing to patch — avoid disk I/O when both args are None.
        if resolved_by is None and resolution_turns is None:
            return esc

        # Apply patches
        if resolved_by is not None:
            esc.resolved_by = resolved_by
        if resolution_turns is not None:
            esc.resolution_turns = resolution_turns

        # Atomically rewrite AT THE SAME PATH (tmp file in path.parent, not queue root).
        self._atomic_write_path(path, esc.to_json())
        return esc

    def note_suppressed_refile(self, escalation_id: str) -> Escalation | None:
        """Record that this record's resolution absorbed one identical refile.

        The INV-4 storm counter for the auto-dismiss path (task 4499).  When
        ``find_terminal_by_citation`` matches, the refile is dropped without a
        record being filed — so without this bump the suppression would be
        LOG-ONLY, and "this adjudication has silently absorbed 900 refiles"
        would be unobservable from the record itself (INV-2
        ``structured-facts-at-failure``).  ``refiles_suppressed`` plays the
        same role for suppression that ``dedupe_count`` plays for the fold
        path: a durable, bounded recurrence signal on the surviving record.

        Deliberately UNTHRESHOLDED — it never escalates at a count.  Escalating
        on repeated suppression would file the very escalation the suppression
        exists to stop, recreating the close-then-refile storm one level up.
        INV-4's applicable house pattern here is the storm COUNTER, not a
        threshold escalation.

        WHO HEARS ABOUT IT — INV-4's other half, answered by name rather than
        left for the next reader to hunt for (amendment pass, task 4499).  The
        counter is readable from the ``get_escalation(escalation_id)`` MCP tool,
        which returns ``Escalation.to_dict()`` — a bare ``asdict``, so both
        ``refiles_suppressed`` and ``citation_sha`` are on that response with no
        whitelist to widen (verified end-to-end, not assumed).  It is
        deliberately NOT on the compact LIST rows: ``_COMPACT_ESCALATION_FIELDS``
        is an exact-key-tested whitelist for triage drains, and a field that is
        non-zero only on already-adjudicated records has no business inflating
        every pending row.

        The residual gap is DISCOVERY, not access: the counter accrues only on
        RESOLVED/DISMISSED records, which no triage surface lists, so reading it
        means knowing the id to ask about.  The suppression WARNING in
        ``file_unattributed_landing_escalation`` names that id on every
        suppression, which is the intended entry point; a listing surface or a
        named operator query for "resolutions absorbing refiles" is genuine
        follow-up work and is filed as such rather than being silently assumed.

        Locking: unlike ``patch_resolution_metadata`` above (a last-write-wins
        field SET) this is a genuine INCREMENT, so the whole read-modify-write
        runs under ``escalation_id_lock`` — a concurrent same-id bump from
        another harness/worker process must not be lost.  The lock sidecar
        lives in ``queue_dir`` and is stable regardless of whether the record
        currently sits in the root or the archive.

        Patches the record IN PLACE wherever it lives, via ``_locate_path`` +
        ``_atomic_write_path`` — NOT ``_rewrite``/``_atomic_write``, which
        always target the queue root and would resurrect an archived record
        out of the archive.

        Returns None (writing nothing) when the record is missing, unparseable,
        or still pending — a pending record has absorbed nothing, since it is
        its own open-L1 veto that suppresses the refile.
        """
        with escalation_id_lock(self.queue_dir, escalation_id):
            path = self._locate_path(escalation_id)
            if path is None:
                return None

            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
                return None

            if esc.status not in ('resolved', 'dismissed'):
                return None

            esc.refiles_suppressed += 1
            self._atomic_write_path(path, esc.to_json())

        logger.info(
            f'Escalation {escalation_id} has now absorbed '
            f'{esc.refiles_suppressed} identical refile(s)'
        )
        return esc

    def _rewrite(self, escalation_id: str, escalation: Escalation) -> None:
        """Atomically rewrite an escalation's JSON file."""
        self._atomic_write(escalation_id, escalation.to_json())

    def _atomic_write(self, escalation_id: str, json_text: str, *, durable: bool = False) -> None:
        """Write *json_text* atomically to ``queue_dir/{escalation_id}.json``.

        Thin wrapper around ``_atomic_write_path`` that constructs the target
        path from *escalation_id* and ``queue_dir``.  Always writes to the queue
        root — not the archive.

        *durable*, when True, is forwarded to ``_atomic_write_path``'s existing
        fsync branch (temp-file fd before rename, containing-directory fd
        after) — the same durability guarantee ``_write_seq_counter`` already
        uses. Defaults to False so existing callers are byte-identical.

        Callers: submit() (durable=True), resolve(), submit_resolved(), _rewrite()
        (the latter three keep the default durable=False).
        """
        self._atomic_write_path(self.queue_dir / f'{escalation_id}.json', json_text, durable=durable)

    def _atomic_write_path(self, path: Path, json_text: str, *, durable: bool = False) -> None:
        """Write *json_text* atomically to *path*.

        Uses the tmp+rename pattern: the tmp file is created in ``path.parent``
        (not hard-coded to ``queue_dir``) so that the ``os.rename`` stays within
        the same directory — required for archive targets where *path* lives in
        a dated subdir.  On failure the tmp file is cleaned up and the exception
        propagates unchanged.

        When *durable* is True, the tmp file is fsync'd before the rename and
        the containing directory is fsync'd after the rename — for callers
        (e.g. ``_write_seq_counter``) that must survive a crash immediately
        after this call returns.  Plain tmp+rename alone only guarantees the
        rename is atomic, not that it is durable yet.

        Callers: _atomic_write() (root targets), patch_resolution_metadata()
        (root or archive targets), _write_seq_counter() (durable=True).
        """
        fd, tmp_path_str = tempfile.mkstemp(
            suffix='.tmp', prefix=path.stem, dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(json_text)
                if durable:
                    f.flush()
                    os.fsync(f.fileno())
            os.rename(tmp_path_str, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path_str)
            raise
        if durable:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def _archive_resolved(self, escalation_id: str, resolved_at: str) -> None:
        """Best-effort move ``queue_dir/{escalation_id}.json`` into the dated archive subdir.

        Two-step design (write-to-root then os.replace to archive) is intentional:
        if the archive move fails, the resolved file remains in ``queue_dir`` so
        ``get()`` can still return it — no data is lost.  Failure logs a warning
        but does not abort the resolution.

        Callers: resolve(), submit_resolved().

        This is also the checkpoint that clears ``_archive_negative_cache``
        (see ``_cache_archive_negative``): a self-archival is a natural
        "state changed" signal for this instance, so it's used to bound how
        long a negative-cached id (re-probed absent earlier) can keep
        masking an id that some OTHER instance archived in the meantime.
        """
        path = self.queue_dir / f'{escalation_id}.json'
        try:
            archive_dir = archive.archive_dir_for_date(self.queue_dir, resolved_at)
            archive_dir.mkdir(parents=True, exist_ok=True)
            os.replace(str(path), str(archive_dir / f'{escalation_id}.json'))
            # Incrementally update the listing cache so get() finds the newly-archived
            # id without a full rescan.  Only update if the cache has already been built.
            if self._archive_listing is not None:
                self._archive_listing.setdefault(archive_dir.name, set()).add(escalation_id)
            # Clear the negative-result cache wholesale — see docstring above.
            self._archive_negative_cache.clear()
        except OSError as exc:
            logger.warning(
                f'Failed to archive escalation {escalation_id}: {exc}; '
                'file remains in queue_dir'
            )

    def dismiss_all_pending(
        self,
        resolution: str,
        *,
        strand_age_secs: float | None = None,
        now: datetime | None = None,
    ) -> int:
        """Dismiss all pending L0 escalations with the given resolution message.

        Level-1 escalations are PRESERVED — they represent steward→human work
        that must survive orchestrator restart and require human attention.
        Only level-0 (agent→steward) escalations are dismissed.

        Every dismissed record's resolution gains a structured
        ``[pending_secs=<n> severity=<s>]`` suffix recording how long it had
        been waiting when the sweep closed it (task 3172).  Before this, one
        fixed string was written to every record, so a level-0 that had been
        pending 20h58m with a workflow parked on it (esc-5189-7) and one
        pending ~90s (esc-5685-1) were indistinguishable afterwards.

        ``strand_age_secs``: opt-in threshold, in seconds.  A level-0 that had
        been pending at least this long is stamped
        ``resolution_class='stale-strand'`` — a distinct, non-benign class
        (models.RESOLUTION_CLASSES) that keeps the strand auditable.  Records
        below the threshold are left unstamped, so
        ``default_resolution_class_for_resolver('auto-dismissed')`` keeps
        stamping them ``'benign'`` exactly as before.  When *None* (the
        default) NO record is stamped ``'stale-strand'``, making the
        classification behaviour identical to the pre-3172 sweep for any
        caller that does not opt in; only the age suffix is new.

        ``now``: injectable clock for tests.  Read ONCE and hoisted out of the
        loop, mirroring ``Harness._reap_orphan_l0_escalations``, so every
        record in one sweep is aged against the same instant.

        Returns the number of escalations where resolve() returned non-None.
        In the common single-writer case this equals the number actually dismissed.
        In a concurrent-write race (another process resolves an escalation between
        get_pending() and resolve()), resolve() is a no-op but still returns the
        existing escalation, so the count may include those no-ops.  The counter
        is best read as "attempted dismissals, including no-ops for concurrent
        resolutions".  This function is single-writer in practice.  The count
        does NOT distinguish the three per-record outcomes — age-annotated,
        strand-stamped, or ``pending_secs=unknown`` for a record whose
        timestamp would not parse; a caller that needs that detail reads the
        resolved records themselves.
        """
        pending = self.get_pending()
        now = now or datetime.now(UTC)
        count = 0
        strands = 0
        for esc in (e for e in pending if e.level == 0):
            try:
                # fallback=datetime.max: an unparseable timestamp sorts as
                # NEWEST, never as infinitely stale (the dedupe.py sentinel
                # rationale).  A corrupt record must not be promoted to
                # 'stale-strand' by the sentinel alone.  parse_timestamp_or_warn
                # stamps a naive timestamp UTC and logs a WARNING naming this
                # escalation on failure — loud, not a silent skip.
                parsed, ok = parse_timestamp_or_warn(
                    esc.timestamp,
                    fallback=datetime.max.replace(tzinfo=UTC),
                    context=f'dismiss_all_pending:{esc.id}',
                )
                # An unknowable age claims no age and is never a strand; the
                # record is still dismissed — clearing stale L0s at startup
                # stays correct regardless of one bad timestamp.
                if ok:
                    age_secs = (now - parsed.astimezone(UTC)).total_seconds()
                    is_strand = strand_age_secs is not None and age_secs >= strand_age_secs
                    per_record = (
                        f'{resolution} [pending_secs={age_secs:.0f} severity={esc.severity}]'
                    )
                else:
                    is_strand = False
                    per_record = f'{resolution} [pending_secs=unknown severity={esc.severity}]'
                resolved = self.resolve(
                    esc.id,
                    per_record,
                    dismiss=True,
                    resolved_by='auto-dismissed',
                    resolution_class='stale-strand' if is_strand else None,
                )
                if resolved is not None:
                    count += 1
                    strands += int(is_strand)
            except Exception as e:
                logger.warning(f'Failed to dismiss escalation {esc.id}: {e}')
        if count:
            logger.info(
                f'Dismissed {count} stale L0 escalation(s) '
                f'({strands} stale-strand): {resolution[:100]}'
            )
        return count

    def _write_seq_counter(self, path: Path, value: int) -> None:
        """Durably persist *value* as the counter contents at *path*.

        Delegates to ``_atomic_write_path(durable=True)`` — the counter must
        survive a crash immediately after this call returns (PRD C9:
        "fsync-incremented").
        """
        self._atomic_write_path(path, str(value), durable=True)

    def _recover_seq_from_disk(self, task_id: str) -> int:
        """Highest sequence observed on disk for *task_id*, or 0 if none.

        Consulted ONLY when the counter file is ABSENT or UNPARSEABLE, never
        while it is present and parseable, so PRD C9's contract — the
        counter, not a directory scan, is the sole source of the next
        sequence — holds in steady state, and task 1879's retired per-mint
        archive glob + ``_archive_max_seq_cache`` stays retired.  ``make_id``
        immediately makes the result durable via ``_write_seq_counter``, so
        this runs at most once per task_id per counter lifetime and never
        gates a steady-state mint.

        Not *purely* a repair path, despite the framing above: the FIRST
        mint for every brand-new task_id also lands on the absent-counter
        branch and pays one scan.  See "Cost" below.

        Scans both halves of the id namespace, because a task_id whose
        escalations were all resolved has evidence ONLY in the archive — and
        that is precisely the case ``get()``'s archive fallback would
        otherwise resolve a re-minted id to (task 3238):

        - queue root: ``esc-{task_id}-*.json``
        - archive subtree: ``_iter_archive_paths('esc-{task_id}-*.json')``

        The ``.json`` suffix in the pattern is load-bearing: it excludes the
        ``esc-{task_id}.seq`` counter itself, the ``{id}.json.lock``
        sidecars, and submit's ``{id}.tmp`` staging files.

        Sequence extraction is PREFIX-ANCHORED: the literal
        ``esc-{task_id}-`` is stripped from the stem and the remainder must
        parse as an int.  The task_id is known, never parsed back out of a
        filename, so this is hyphen-safe by construction — recovering task
        ``'1-2'`` correctly ignores ``esc-1-2-3-9.json`` (task ``'1-2-3'``),
        which the glob over-matches.  Reintroducing the retired
        parse-after-last-hyphen derivation here would resurrect that bug.
        The comparison is NUMERIC (``max`` over parsed ints, not over stem
        suffixes as strings): lexicographically ``'9' > '10'``, which would
        under-report the max and mint a colliding id.

        Cost, and why the archive half is a FRESH targeted rglob rather than
        the memoised ``_get_archive_listing()``: that memo is built once per
        instance and can predate another ``EscalationQueue`` instance's
        archival of records for this task_id.  ``_locate_path`` tolerates
        that staleness because it is an EXISTENCE lookup — it verifies a
        memo hit with ``.exists()`` and re-probes on a memo miss — but a
        MAXIMUM derived from a merely-incomplete memo is wrong in the
        collision-producing direction: it under-reports and mints an id an
        archived record already holds, the exact defect this scan exists to
        prevent.  The freshness is therefore deliberate, and the price is
        one archive rglob per task_id per counter lifetime (so, in a
        long-lived instance, once per new task_id) rather than one per
        instance — paid off the steady-state mint path, under the per-counter
        lock, and bounded by ``prune_archive`` retention.

        Recovery observes SUBMITTED records only.  An id that was minted but
        not yet submitted when the counter was lost is invisible here and
        can therefore still be re-minted — a residual hole inherent to any
        disk-derived recovery, and far narrower than the pre-3238 behaviour
        of restarting from 1.  It is also why ``make_id``'s ``OSError``
        branch is NOT routed here: there the counter still exists and
        remains authoritative over disk.
        """
        prefix = f'esc-{task_id}-'
        pattern = f'{prefix}*.json'
        highest = 0
        for path in itertools.chain(
            self.queue_dir.glob(pattern), self._iter_archive_paths(pattern),
        ):
            stem = path.stem
            if not stem.startswith(prefix):
                continue
            try:
                seq = int(stem[len(prefix):])
            except ValueError:
                # A sibling task_id the glob over-matched (e.g. esc-1-2-3-9
                # while recovering '1-2'), or a non-numeric suffix.
                continue
            highest = max(highest, seq)
        return highest

    def make_id(self, task_id: str) -> str:
        """Generate a unique escalation ID from a durable per-task_id counter.

        The counter file ``queue_dir/esc-{task_id}.seq`` holds the
        last-issued sequence number as plain integer text. In steady state
        this file is the SOLE source of the next sequence: unlike the
        retired directory/archive-scan derivation, make_id() never globs the
        archive to "catch up" — the counter is authoritative (PRD
        task-status-authority-prd.md contract C9 / finding 10.4).

        The four read outcomes, in full:

        - Present and parseable (the steady state): authoritative, taken
          verbatim, with ZERO disk scans of any kind.
        - Unparseable contents (``ValueError``): reconciled from a one-shot
          bounded ``_recover_seq_from_disk`` scan and logged as an ERROR.
          Still ERROR, not WARNING — a corrupt counter is unrecoverable data
          loss an operator must see — but it is now self-healing rather than
          collision-producing.
        - An absent file: reconciled the same way.  0 recovered is the
          common, legitimate case (a brand-new task_id) and stays SILENT;
          a non-zero recovery means the counter was LOST (aggressive
          cleanup, fresh checkout, disk restore, accidental rm of the
          non-.json sidecar) while escalations for this task_id already
          exist, and is logged as an ERROR.  Note this branch is not only a
          repair path — every brand-new task_id's FIRST mint takes it and
          pays one reconciliation scan (see ``_recover_seq_from_disk``
          "Cost"); every mint after that is counter-derived and scan-free.
        - An ``OSError`` while reading an EXISTING file (transient I/O error,
          EINTR, fd exhaustion, permission flap) is NOT reconciled — it is
          allowed to propagate so the mint fails loudly.  A present-but-
          momentarily-unreadable counter is not evidence of loss, and
          reconciling there would derive a maximum from SUBMITTED records
          only, silently rewinding the counter below any id that was minted
          but never submitted.

        Reconciliation returns ``max(observed sequence on disk) + 1``, so a
        recovered id is strictly greater than every root and archived record
        for this task_id — it can never be one that ``get()``'s archive
        fallback resolves to a pre-existing, already-resolved record (task
        3238).  The reconciled value is written durably BEFORE the id is
        returned, which is what bounds the repair scan to once per task_id
        per counter lifetime.  The residual hole: recovery observes
        SUBMITTED records only, so an id minted but not yet submitted when
        the counter was lost is invisible to it and can still be re-minted.

        The read -> increment -> durable write (tmp+fsync+rename+dir-fsync,
        see ``_write_seq_counter``) all happen inside
        ``escalation_id_lock(queue_dir, f'esc-{task_id}.seq')`` so concurrent
        OS processes minting ids for the SAME task_id serialize on the
        stable ``esc-{task_id}.seq.json.lock`` sidecar inode and never
        observe the same counter value.
        """
        counter_id = f'esc-{task_id}{SEQ_COUNTER_SUFFIX}'
        counter_path = self.queue_dir / counter_id
        with escalation_id_lock(self.queue_dir, counter_id):
            current = 0
            if counter_path.exists():
                try:
                    current = int(counter_path.read_text().strip())
                except ValueError as e:
                    current = self._recover_seq_from_disk(task_id)
                    logger.error(
                        f'make_id: could not parse counter file {counter_path}: {e}; '
                        f'reconciled it from a one-shot bounded root+archive scan for '
                        f'task_id {task_id!r} (highest observed sequence {current}) and '
                        f'resuming at {current + 1}, so no already-recorded id is '
                        'reused (an id minted but never submitted is not observable '
                        'on disk and is not covered)'
                    )
            else:
                current = self._recover_seq_from_disk(task_id)
                if current > 0:
                    logger.error(
                        f'make_id: counter file {counter_path} is absent but '
                        f'escalations for task_id {task_id!r} already exist on disk '
                        f'(highest observed sequence {current}); reconciled the '
                        f'counter from a one-shot bounded root+archive scan and '
                        f'resuming at {current + 1} rather than re-minting from 1, '
                        'so no already-recorded id is reused (an id minted but '
                        'never submitted is not observable on disk and is not '
                        'covered)'
                    )
            nxt = current + 1
            self._write_seq_counter(counter_path, nxt)
            return f'esc-{task_id}-{nxt}'


def observed_submit_response(
    queue: EscalationQueue,
    esc_id: str,
    *,
    fallback_level: int | None = None,
) -> dict[str, Any]:
    """Shape a post-write response from OBSERVED state, never from write intent.

    Task 3236.  The submit paths used to return a hardcoded
    ``{'id': ..., 'status': 'queued'}`` with no re-read, so a filing that a
    concurrent resolver/sweep had already dismissed in the same instant was
    still reported to its filer as 'queued'.  The filer then had no way to
    learn its escalation had been swallowed.

    Re-reads the persisted record and returns:
    - still pending → ``{'id', 'status': 'queued', 'level'}`` (as before, plus
      the persisted level so a caller that asked for ``level=1`` can confirm
      it landed);
    - anything else → the auto-resolved shape ``escalate_blocker``'s docstring
      already promises, ``{'id', 'status', 'resolution', 'resolved_by',
      'level'}``, carrying the record's REAL values.  No new status vocabulary
      is invented, so no consumer needs updating.

    FAIL-OPEN by construction: a re-read that returns ``None`` or raises falls
    back to the historical ``'queued'`` response and logs a WARNING rather than
    raising.  A filing must never be lost to a bookkeeping read — that is the
    very failure mode being fixed here.

    ``fallback_level`` is the level the CALLER wrote (i.e. ``esc.level``).  It
    is echoed on the two fail-open branches so ``level`` is present on EVERY
    response this function can return: the ``level`` echo is documented as the
    way a caller confirms its requested level landed, and a caller written to
    that contract must not hit a ``KeyError`` in exactly the degraded case the
    fail-open exists to survive.  It is a fallback, never an override — when
    the re-read succeeds, the PERSISTED level is reported even if it differs
    from what the caller asked for (born-at-L2 severity legitimately overrides
    a requested level=1).

    Lives in ``queue`` rather than ``dedupe`` because it re-reads a persisted
    record and shapes a response — it has nothing to do with fold logic, and
    ``server._submit_or_dedupe``'s born-at-L2 branch needs it precisely because
    that branch does NOT route through dedupe.

    COST NOTE: this adds one ``queue.get()`` per successful submit.  For a
    just-written record that is a cheap queue-root hit, but ``get()`` falls
    through to ``_locate_path``'s archive listing when the record is absent
    from the root — i.e. an O(archive) scan is possible on the submit path in
    exactly the resolve+archive race this targets.  Acceptable at current
    volumes; if it ever shows up in a storm profile (e.g. a 30-task infra
    fan-out), read ``queue_dir/{id}.json`` directly and treat an absent record
    as the fail-open 'queued' case — that is already this function's documented
    behaviour for an unreadable record, though it would forfeit the honest
    observed-state report for a record archived between submit and re-read.
    """
    try:
        persisted = queue.get(esc_id)
    except Exception as exc:
        logger.warning(
            'Post-submit re-read of %s failed (%s); reporting queued (fail-open)',
            esc_id, exc,
        )
        return _fail_open_queued(esc_id, fallback_level)
    if persisted is None:
        logger.warning(
            'Post-submit re-read of %s returned nothing; reporting queued (fail-open)',
            esc_id,
        )
        return _fail_open_queued(esc_id, fallback_level)
    if persisted.status == 'pending':
        return {'id': esc_id, 'status': 'queued', 'level': persisted.level}
    logger.warning(
        'Escalation %s was already %s at filing time (resolved_by=%r); '
        'reporting observed state instead of queued',
        esc_id, persisted.status, persisted.resolved_by,
    )
    return {
        'id': esc_id,
        'status': persisted.status,
        'resolution': persisted.resolution,
        'resolved_by': persisted.resolved_by,
        'level': persisted.level,
    }


def _fail_open_queued(esc_id: str, fallback_level: int | None) -> dict[str, Any]:
    """The fail-open 'queued' response, carrying *fallback_level* when known.

    ``level`` is omitted only when the caller supplied no fallback — legacy
    callers that predate the echo contract — so the key is never fabricated.
    """
    if fallback_level is None:
        return {'id': esc_id, 'status': 'queued'}
    return {'id': esc_id, 'status': 'queued', 'level': fallback_level}
