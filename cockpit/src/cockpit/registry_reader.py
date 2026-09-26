"""cockpit.registry_reader — pure-consumer read path over the C1 session registry.

Fleet Cockpit C5a (plans/fleet-cockpit-prd.md §9). Imports the frozen
orchestrator.session_registry contract (PRD §6 G5: consumers import, never
re-derive the record shape). This module is read-only: it never calls
write_record/write_decision/update_decision_state/set_manual_boost.

It is also the cockpit's PROJECT-TOKEN CANONICALIZATION BOUNDARY (task
3812), and this docstring is the ONE place that argument is written down --
everything else that needs it points here. Read-only is unchanged --
nothing here writes to disk -- but every record that enters the cockpit
through this module leaves it with ``.project`` folded onto the one
canonical spelling (``session_registry.normalize_project_token``, task
3807). The reason the fold has to sit HERE, at the reader, rather than in
the display or scoring adapters downstream: the cockpit unions its two
record kinds -- sessions and decisions -- onto ONE ``project_weights`` key
and ONE weight picker (``priority.score``'s
``project_weights[item.project]`` lookup and
``panes.weight_editor.known_projects``), so the two kinds must fold
together or an operator's weight silently applies to only one of them. The
on-disk records themselves stay raw and unmigrated; this is a read-side
fold, retroactive over every already-written record. Its two entry points
are ``_read_record_soft`` (sessions) and ``scan_decisions`` (decisions),
and both reach the rule itself through the one ``_canonicalize_project``
helper below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from orchestrator import session_registry
from orchestrator.session_registry import normalize_project_token

logger = logging.getLogger(__name__)

# Substantive SessionRecord fields a snapshot is keyed on. Deliberately
# excludes start_ts/age-derived values so a purely-time-passing poll tick
# (nothing on disk actually changed) diffs as a no-op. That rule is about
# values that MOVE ON THEIR OWN, not about timestamps as such: build_snapshot
# appends the record's question.asked_at for exactly the reason spelled out
# in its docstring.
_SNAPSHOT_FIELDS = (
    'status',
    'title',
    'role',
    'project',
    'task_id',
    'escalation_id',
    'parent_session_id',
)


def _canonicalize_project(
    record: session_registry.SessionRecord | session_registry.DecisionRecord,
) -> None:
    """Fold *record*'s ``.project`` onto its canonical spelling, in place.

    The single enforcement point for this module's canonicalization
    boundary (see the module docstring for why the cockpit folds at all).
    Both entry points -- ``_read_record_soft`` for sessions,
    ``scan_decisions`` for decisions -- go through here, so the rule exists
    once instead of twice-and-cross-referenced and the two record kinds
    cannot drift apart. Both are plain mutable dataclasses carrying a
    ``.project``; nothing else about them is touched.

    In place, because both callers hand over a record ``session_registry``
    has just parsed for them and that this module exclusively owns --
    mirroring ``migrate_decision_project_tokens``' own
    ``record.project = ...`` idiom.

    The inequality guard mirrors that same function's already-canonical
    test (``normalize_project_token(p) == p``): a record already spelled
    canonically is left untouched rather than rebound to an equal string.
    What the guard elides is that attribute rebind and nothing more -- the
    fold has already run and already built its stripped/casefolded/
    regex-substituted string by the time the guard is reached. Nor is the
    untouched case the common one here: the dominant on-disk spelling in
    this fleet is ``dark-factory`` (42,026 records measured 2026-09-07,
    against 1,493 ``dark_factory``), which is NOT canonical, so a cold scan
    really does rewrite most of what it reads.
    """
    canonical = normalize_project_token(record.project)
    if canonical != record.project:
        record.project = canonical


def _read_record_soft(
    slug: str, root: Path | str | None, *, caller: str
) -> session_registry.SessionRecord | None:
    """Read *slug*'s record.json, fail-soft: return None instead of raising.

    Shared by scan_sessions and SessionScanner.scan (below) so the fail-soft
    policy -- which exceptions are swallowed, what gets logged -- lives in
    exactly one place: both loops build their per-slug read step on this one
    helper, so they stay behaviorally identical by construction rather than
    by convention (PRD §2; see the parity test
    test_scan_matches_scan_sessions_for_seeded_dir). *caller* is folded into
    the warning message only, preserving each call site's original
    log-message prefix for anyone grepping logs.

    It is also where the returned record's ``.project`` -- and ONLY
    ``.project`` -- is rewritten onto its canonical spelling, via
    ``_canonicalize_project`` (task 3812; the module docstring has the why).
    Being the one shared parse step is exactly why the call belongs here:
    both scan paths inherit the fold by construction rather than by
    convention, the same property they already inherit for fail-soft, and
    SessionScanner's mtime cache therefore stores ALREADY-CANONICAL records
    -- so the fold is paid once per PARSE, not once per poll tick. Every
    other field is left byte-identical to what was written, notably
    ``.title``, the literal terminal title ``.project`` was originally
    parsed out of: rewriting one and not the other is intended, since the
    title is display text while the project token is a join key.
    """
    try:
        record = session_registry.read_record(slug, root=root)
    except (FileNotFoundError, session_registry.CorruptSessionRecord):
        logger.warning('%s: skipping unreadable record for %s', caller, slug, exc_info=True)
        return None
    _canonicalize_project(record)
    return record


def scan_decisions(root: Path | str | None = None) -> list[session_registry.DecisionRecord]:
    """Return every readable DecisionRecord, ``.project`` canonicalized.

    The decision-side twin of scan_sessions (task 3812), applying the same
    rule through the same ``_canonicalize_project`` helper. Why both record
    kinds have to fold together is argued once in the module docstring.

    The whole READ -- and therefore the whole fail-soft policy: absent
    decisions/ dir -> [], a single corrupt or foreign ``*.json`` logged and
    skipped rather than aborting the rest -- is delegated to
    ``session_registry.list_decisions``. This is a fold over that reader,
    never a second implementation of it, so its contract cannot drift away
    from the frozen one (pinned by
    test_returns_the_same_ids_in_the_same_order_as_list_decisions).

    The fold is IDEMPOTENT and therefore a NO-OP for every decision written
    since task 3807 -- the ``write-decision`` verb already stamps the
    canonical token at the write path. What it exists for is the LEGACY rows
    still sitting on disk that ``migrate_decision_project_tokens`` has not
    been run over (measured 2026-09-07: 19 OPEN ``df`` + 2 OPEN
    ``dark-factory``), and, more durably, so the cockpit's guarantee rests on
    a rule it applies itself rather than on a migration having been run
    somewhere else.

    Read-only, like everything in this module: the canonicalized records are
    in-memory only and are never written back (the cockpit never calls
    write_decision).
    """
    decisions = session_registry.list_decisions(root)
    for record in decisions:
        _canonicalize_project(record)
    return decisions


def scan_sessions(root: Path | str | None = None) -> list[session_registry.SessionRecord]:
    """Return every readable SessionRecord under ``sessions_dir(root)``.

    Mirrors reap_stale_records' identity-from-path iterdir loop: an absent
    sessions/ dir returns [] (not an error), and a single corrupt or
    vanished record.json is logged and skipped rather than aborting the
    scan of the remaining slugs (fail-soft, PRD §2).
    """
    base = session_registry.sessions_dir(root)
    if not base.is_dir():
        return []

    records: list[session_registry.SessionRecord] = []
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        record = _read_record_soft(slug_dir.name, root, caller='scan_sessions')
        if record is not None:
            records.append(record)
    return records


class SessionScanner:
    """Stateful, cache-aware counterpart to scan_sessions (above).

    A poll loop calling scan_sessions() every tick pays a full read_text() +
    JSON parse for every session slug, every tick, even when nothing on disk
    changed -- at 10k+ sessions this is the multi-second cost this class
    exists to eliminate (see registry_reader/app.py module docstrings). A
    SessionScanner owns a per-slug cache keyed on record.json's
    st_mtime_ns, so a later scan() call reuses a cached SessionRecord for
    any slug whose record.json mtime is unchanged since the last scan --
    iterdir() is cheap and always runs (still needed to detect added/removed
    slugs), but the parse-heavy read_record() call is skipped for an
    unchanged slug. A stateful, constructed object (rather than a free
    function) gives the cache an obvious owner/lifecycle: each CockpitApp
    instance (or test) gets its own SessionScanner with its own independent
    cache; scan_sessions itself is left untouched as the stateless reader.

    Not safe to call scan() concurrently from more than one thread: reading
    self._cache and reassigning it at the end of scan() are unlocked, so two
    overlapping scan() calls on the same instance could race and one could
    observe a partially-built cache. Nothing inside this class enforces
    that -- it is an invariant upheld by the caller. CockpitApp's own
    SessionScanner is only ever scanned from on_mount's synchronous initial
    refresh_registry() (which completes before self.set_interval starts
    polling) or from the single poll-triggered worker _poll_registry
    launches, itself guarded by the app's _scan_in_flight flag so at most
    one poll-triggered scan() call is ever in flight (see app.py's
    _poll_registry/_scan_registry_worker docstrings). A caller that adds a
    second concurrent path to scan() -- e.g. a hypothetical "force refresh"
    keybinding that could race an in-flight poll worker -- must provide its
    own mutual exclusion (or a lock added to this class) first.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = root
        self._cache: dict[str, tuple[int, int, session_registry.SessionRecord]] = {}

    def scan(self) -> list[session_registry.SessionRecord]:
        """Return every readable SessionRecord under sessions_dir(root).

        Mirrors scan_sessions' fail-soft iterdir + read_record loop (see its
        docstring): an absent sessions/ dir returns [], and a corrupt or
        vanished record.json is logged and skipped rather than aborting the
        scan of the remaining slugs. Additionally: a slug whose record.json
        (st_mtime_ns, st_size) pair is unchanged since the last scan() call
        reuses the cached SessionRecord instead of paying read_record's
        read_text() + JSON parse again.

        The cache is rebuilt from scratch every call from exactly the slugs
        seen THIS scan, and only assigned to self._cache at the end -- a
        slug that vanished since the last scan is dropped from the cache
        even if it never resurfaces, so a later, unrelated record written at
        the same slug is always read fresh rather than risking a match
        against a stale leftover cache entry (e.g. a coincidentally-repeated
        mtime). Pairing st_mtime_ns with st_size costs nothing extra (both
        come from the same stat() call) and closes a gap plain mtime-only
        keying would have: many filesystems' mtime resolution is coarser
        than the nanosecond precision st_mtime_ns reports, so two different
        rewrites of the same slug landing inside one resolution window could
        otherwise share an mtime and the cache would keep serving the first
        rewrite's now-stale record. write_record's full-record JSON
        serialization changes st_size for essentially any substantive field
        change, so a same-mtime rewrite is still caught unless it also
        happens to preserve the exact serialized length -- a narrower,
        lower-probability residual than mtime alone. A stat() failure
        (record.json vanished between iterdir() and stat(), or any other
        OSError) is treated as a cache miss -- it falls back to
        read_record(), which raises FileNotFoundError/CorruptSessionRecord
        in turn and is fail-soft skipped exactly like scan_sessions, rather
        than propagating out of scan() uncaught.
        """
        base = session_registry.sessions_dir(self._root)
        if not base.is_dir():
            self._cache = {}
            return []

        records: list[session_registry.SessionRecord] = []
        new_cache: dict[str, tuple[int, int, session_registry.SessionRecord]] = {}
        for slug_dir in sorted(base.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            record_path = session_registry.record_path_for_slug(slug, root=self._root)
            try:
                record_stat = record_path.stat()
                stat_key: tuple[int, int] | None = (
                    record_stat.st_mtime_ns,
                    record_stat.st_size,
                )
            except OSError:
                stat_key = None

            cached = self._cache.get(slug)
            if stat_key is not None and cached is not None and (cached[0], cached[1]) == stat_key:
                new_cache[slug] = cached
                records.append(cached[2])
                continue

            record = _read_record_soft(slug, self._root, caller='SessionScanner.scan')
            if record is None:
                continue
            if stat_key is not None:
                new_cache[slug] = (*stat_key, record)
            records.append(record)
        self._cache = new_cache
        return records


@runtime_checkable
class SessionScannerProtocol(Protocol):
    """Structural shape of SessionScanner's scan() -- CockpitApp's `scanner` DI seam.

    Mirrors cockpit.backends.base.FocusArrangeBackend's Protocol-for-DI
    pattern: typing the seam structurally (rather than as the concrete
    SessionScanner class) lets a test's fake scanner satisfy it by
    implementing scan() directly, with no need to subclass the real
    SessionScanner -- which would drag in its mtime cache and disk-scanning
    internals for no reason.
    """

    def scan(self) -> list[session_registry.SessionRecord]:
        """Return the current readable SessionRecord set."""
        ...


def build_snapshot(
    records: list[session_registry.SessionRecord],
) -> dict[str, tuple]:
    """Map session_slug -> a hashable tuple of substantive display fields.

    Deliberately excludes start_ts/age so a quiet (nothing-changed) tick
    produces an identical snapshot -- this is what lets the app's poll
    rebuild the table only when something real changed (PRD §5).

    question.asked_at is deliberately INCLUDED, and that is not in tension
    with the exclusion rule above. This snapshot is the WAKE-UP TRIGGER for
    CockpitApp._prune_overlays: _apply_scan short-circuits on an unchanged
    snapshot, so a tick that diffs as a no-op never reaches _rebuild_queue
    and never runs the prune. The trigger must therefore be AT LEAST AS
    STRONG as the identity the prune keys an overlay on
    (CockpitApp._ask_identity: `(question.text, question.asked_at)`) -- a
    weaker trigger lets a real new ask pass the diff unnoticed and pins the
    operator's stale drop forever, silently suppressing a live ask. Do not
    "optimize" asked_at back out of this tuple.

    It does NOT reintroduce the flicker the start_ts exclusion guards
    against: asked_at is an on-disk stamp written only when a new Question
    is stamped, unlike start_ts-derived age which moves on every tick by
    construction. A quiet poll (nothing changed on disk) still diffs as a
    no-op.
    """
    snapshot: dict[str, tuple] = {}
    for record in records:
        values = tuple(getattr(record, field) for field in _SNAPSHOT_FIELDS)
        question_text = record.question.text if record.question is not None else None
        question_asked_at = record.question.asked_at if record.question is not None else None
        snapshot[record.session_slug] = (*values, question_text, question_asked_at)
    return snapshot


def snapshot_changed(old: dict[str, tuple], new: dict[str, tuple]) -> bool:
    """True iff a record was added, removed, or had a substantive field change."""
    return old != new
