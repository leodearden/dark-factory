"""Backend data layer for the Memory-evals dashboard section (task 3215).

Reads the memory-eval program's on-disk artifacts and shapes them for the
``/api/v2/dashboard/memory-evals`` route.  See
``docs/prds/memory-eval-program.md`` (M1 metric series, M2 limits + verdicts,
M3 escalation contract) and ``docs/prds/memory-eval-dashboard.md``.

**Artifacts only, never the module (G6/INV-5).**  This reader uses plain
``json.load`` + dict access and deliberately does NOT import
``shared.memory_eval_metrics`` / ``shared.memory_eval_limits`` — exactly as
``shared/tests/fixtures/memory_eval/README.md`` describes the dashboard-shaped
reader.  The artifact series on disk is the published contract; importing the
producer would couple the dashboard to the producer's in-memory objects and
invite a second implementation of what the producer already decided.  For the
same reason there is **no statistics of any kind** here: the evaluator's
stdlib stats surface is not imported, and neither p-values nor thresholds are
computed.  Verdicts are *read*, never re-derived (INV-1: same file the
evaluator read).  ``TestCommittedExemplarBoundary`` enforces this over the
AST — the module imports no ``statistics`` and no stats surface at all.

**Loud, never silent (DD6/INV-2).**  Every artifact this reader cannot use is
recorded in the payload's structured ``issues`` list — named, with its
``eval_id`` and path — not merely logged and not silently dropped.  Nothing in
here raises: a degraded tree yields a degraded payload, never a 500.  Staleness
is *displayed*, never alarmed on; this module files no escalations.

That never-raises contract is delivered by ONE named boundary, in
``build_memory_evals``, and not by broad per-artifact catches.  Each artifact
reader catches only ``_ARTIFACT_ERRORS`` — what reading and parsing a file off
disk can raise — and degrades one row.  Everything else is a bug in this
module, and a bug is reported AS a bug: ``internal_error``, naming the
exception type, with the traceback in the dashboard log.  The distinction is
the point.  A broad catch around a whole reader reported a typo here to the
operator as ``unreadable_limits``, sending them to inspect an artifact that
was perfectly fine, and left no traceback to fix the real fault from.

Concretely, every artifact reader here distinguishes THREE disposal paths and
names two of them: the file is absent (``missing_*``, and only where absence is
meaningful — a tree where no eval has ever run is healthy, not degraded), it
fails to parse (``unreadable_*``), or it parses but is not the expected type
(``malformed_*``).  **No reader may discard a parsed artifact without appending
to** ``issues``.  The third path is the easy one to omit and the costly one to
lose: a discarded verdicts body leaves every row's verdict absent, which is
indistinguishable from a healthy no-alarm tree unless the discard is recorded.
This does not widen validation into producer territory — the reader still
validates only what it must READ (JSON parses, top level is a dict, ``metrics``
is a list); semantic M1 rules such as a proportion outside [0,1] still pass
through verbatim with no issue.

The same rule governs the reader's INDEXES.  Every ``{key: record}`` build here
(metrics by ``metric_id``, verdicts by ``(eval_id, metric_id, item)``,
escalations by ``dedupe_fingerprint``) would silently keep whichever record came last on a key
collision, so each one instead keeps the FIRST and names the drop
(``duplicate_*``).  A dropped escalation is the sharpest case: last-wins would
make the loser invisible everywhere in the payload — no row link and no
``unmatched_escalations`` entry — which is exactly the silent degradation the
parity view exists to catch.

Index FILTERS are held to the same standard, with one deliberate exception.
Skipping a ``resolved`` / ``dismissed`` escalation, or an escalation from
another category, is the filter doing its job — the record is not wanted, so
its absence is not a loss and earns no issue.  Skipping a record the reader
cannot CLASSIFY is a genuine discard and is named
(``unknown_escalation_status``).  So is skipping one it cannot KEY: an open
escalation with no usable ``dedupe_fingerprint`` is named
(``unfingerprinted_escalation``) and still surfaced in
``unmatched_escalations``, because the index is what that list is built from
and a live alarm must never be absent from both.  The openness test is positive
(``status`` is ``pending``) rather than a blocklist of closed states, so a
status vocabulary that grows a new terminal state fails toward "not shown as
open" rather than toward a closed alarm rendered as live.  A queue file the
escalation corpus could not read as an escalation is neither classifiable nor
keyable, so it takes neither filter path above; it is named at its own path
(``unreadable_escalation_file``), exactly as the Escalations tab lists it.

The derived ``parity`` badge answers to the same rule.  A verdict outside the
closed M2 set has no badge, so it is NAMED (``unknown_verdict``) instead of
falling through to the healthy one — the third instance of the idiom, beside
``unknown_kind`` and ``unknown_escalation_status``.  For the same reason no
verdict class borrows another's label: ``clear`` means "judged, and not
alarming", so an unjudged, an unjudgeable (``insufficient_data``) and a
deliberately-exempted (``grandfathered``) metric each get their own state
rather than all three reading as healthy.  The raw ``verdict`` string is still
passed through the payload unmapped in every case: the reader names the value
it cannot render, it never rewrites it.

**Same-host file reads (DD1).**  The dashboard and the eval runner share a
filesystem; there is no RPC in this path.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from escalation.models import Escalation

from dashboard.data.datum import Datum
from dashboard.data.escalation_corpus import EscalationCorpus, Location, QueueScan
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

# The only escalation category that participates in the fingerprint join.  An
# escalation from any other producer may legitimately carry a colliding
# dedupe_fingerprint; joining it would attribute someone else's alert to a
# memory-eval metric.
_EVAL_REGRESSION_CATEGORY = 'eval_regression'

# Openness is a POSITIVE test on this value, not the absence of a closed one:
# a status vocabulary that grows a new terminal state must not silently start
# rendering as open.  `park()` keeps a genuinely-open L2 at 'pending', so a
# parked escalation still joins.
_OPEN_ESCALATION_STATUS = 'pending'

# The known terminal states.  Skipping one is the filter working as intended,
# so — unlike an unrecognised status — it is not recorded as a discard.
_CLOSED_ESCALATION_STATUSES = frozenset({'resolved', 'dismissed'})

# One screen of trend == one alpha-derivation window.  Matches
# ``runs_per_quarter=90`` in the committed limits artifact, which is what the
# program's derived alpha is computed over, so the chart shows exactly the
# window the limits govern.  When more runs exist on disk the payload says so
# (``truncated`` / ``runs_on_disk``) — a dropped run is never silent.
_TREND_RUN_CAP = 90

# The closed metric-kind vocabulary (M1).  A kind outside this set is a
# RENDERING failure — there is no chart primitive for it — so it earns an
# issue.  Other semantic violations of the M1 schema are the producer's to
# reject at emit time and are passed through verbatim here.
_KNOWN_KINDS = frozenset({'tripwire', 'proportion', 'count', 'scalar'})

# The closed M2 verdict vocabulary.  Like ``_KNOWN_KINDS`` this is a RENDERING
# vocabulary — one badge per state — and NOT a translation table: the raw
# ``verdict`` string is still passed through the payload unmapped, and this set
# is consulted only to decide which parity badge the row earns.  A verdict
# outside it has no badge, so it is named (``unknown_verdict``) rather than
# folded into the healthy label.  Keeping the set closed is what makes the
# pass-through sound: without it, the parity vocabulary would be unbounded and
# the UI would have no single set to switch on — which is the whole reason
# parity is derived here rather than re-derived in the frontend.
_KNOWN_VERDICTS = frozenset({'alarm', 'no_alarm', 'insufficient_data', 'grandfathered'})

# The closed parity vocabulary — every value ``_parity`` can emit, plus the
# ``storm_collapsed`` state ``_verdict_row_fields`` short-circuits to.  PUBLIC on
# purpose: it is the single source of truth the dashboard tab (task 3216) and
# the parity gate (task 3217) should assert against, rather than each
# hardcoding a subset that rots silently the moment a state is added here.
#
# Written out as a literal rather than computed from ``_KNOWN_VERDICTS`` by
# string concatenation, so the vocabulary stays readable and greppable; the
# reachability test in ``TestParityVocabularyIsClosedAndExported`` is what
# keeps the literal and the builder's actual outputs in agreement.
#
# Note the deliberate naming asymmetry: for an ``alarm`` the UNLINKED state is
# the notable one (a filed alarm with no escalation is the parity violation),
# so it carries the distinguishing name; for every other class the LINKED state
# is the notable one, so that is the one suffixed ``_open``.  Every name that
# predates the per-class split was preserved verbatim — ``recovered_open`` and
# ``clear`` were NARROWED to the ``no_alarm`` verdict rather than renamed —
# because an in-flight consumer already branches on them and a rename would
# have left those branches silently dead.
PARITY_STATES: frozenset[str] = frozenset({
    'alarmed_open',
    'alarmed_unlinked',
    'recovered_open',
    'clear',
    'insufficient_data',
    'insufficient_data_open',
    'grandfathered',
    'grandfathered_open',
    'unjudged',
    'unjudged_open',
    'unknown_verdict',
    'unknown_verdict_open',
    'storm_collapsed',
})

# An eval whose newest run is older than this is DISPLAYED as stale.  It is
# never alarmed on: the runner-failure tripwire belongs to the eval program,
# and this module files nothing.  36h gives a nightly cadence one full missed
# run plus slack before the badge appears.
_STALE_AFTER_SECONDS = 36 * 3600.0

# The producer's run-stamp format (zero-padded UTC, which is why filename order
# is chronological order).
_RUN_STAMP_FORMAT = '%Y%m%dT%H%M%SZ'

# What READING AND PARSING one artifact off disk can raise — nothing more.
# Caught around the ``_load_json`` call itself at each artifact boundary, so one
# bad file degrades one row and never the whole payload.
#
# Deliberately narrow.  ``json.JSONDecodeError`` is omitted because it IS a
# ``ValueError`` subclass, and ``TypeError``/``AttributeError`` are omitted
# because they are what a CODE BUG raises, not what a bad file raises: catching
# them here reported a typo in this module to the operator as
# ``unreadable_limits``, sending them to inspect a file that was perfectly
# fine.  Anything outside this tuple is a bug, and is caught once at the
# outermost boundary in ``build_memory_evals`` and named as one.
_ARTIFACT_ERRORS = (OSError, ValueError)

# Issue kinds that CAN mean the walk did not COMPLETE, as opposed to completing
# over a degraded tree.  This is the KIND gate only, not the whole rule:
# ``internal_error`` is emitted at two scopes, so :func:`root_scan_succeeded` —
# still the only reader — applies a further SCOPE test to it, and pins a payload
# whose only ``internal_error`` is per-eval (``eval_id`` set).  Membership here
# is therefore necessary but not sufficient for declining to cache.
_UNCACHEABLE_ISSUE_KINDS = frozenset({'unreadable_root', 'internal_error'})

# The limits artifact carries a great deal more than this — ``alarms``,
# ``alarmed_metric_count``, ``grandfather_set``, ``snapshotted_metric_ids`` and
# its own embedded ``verdicts[]``.  Only this whitelist is surfaced, and it is
# PROVENANCE: what alpha the program derived, over which baseline runs, against
# which grandfather set.  See ``_read_limits`` for why the rest is ignored.
_LIMITS_PROVENANCE_KEYS = (
    'alpha',
    'false_alarm_budget',
    'runs_per_quarter',
    'min_samples',
    'baseline_window',
    'baseline_run_stamps',
    'grandfather_set_hash',
    'run_stamp',
    'generator',
)

# Every value this module interpolates into an `_issue` detail is read out
# of unvalidated data off disk — an escalation queue record, a metrics run,
# a per-eval limits artifact, or the root verdicts artifact — and may be
# arbitrarily large.  Two shapes reach here, and the second is the
# non-obvious one a future reader would otherwise re-litigate: a value with
# no type guard at all (`escalation.id`, `run_kind`, `verdict`,
# `status`) is arbitrary JSON and may be a dict, a list, or a
# multi-megabyte string; a value already `isinstance`-guarded to a
# non-empty `str` (`metric_id`, `eval_id`, `latest_run_stamp`) is
# type-safe but never length-guarded, because the guard proves the value
# is a string, never that it is a short one — this second shape is exactly
# the gap that let the sweep stay incomplete after task 4168.  A detail
# names WHAT was read, so every one of these is capped: an unbounded repr
# would put a multi-megabyte artifact into every poll's payload, making
# the naming of the discard its own degradation.
#
# A THIRD shape reaches a detail without ever going through `_short_repr`:
# `str(exc)` on the `unreadable_*` sites, and
# dashboard/src/dashboard/data/escalation_corpus.py::Unreadable.error on
# `unreadable_escalation_file`.  These stay raw because they are bounded
# BY the exceptions a read/parse boundary raises BEFORE the artifact exists
# as Python data (`_ARTIFACT_ERRORS` here; the corpus' read, `json.loads`
# and `Escalation.from_dict`): a `json.JSONDecodeError` message is one of a
# handful of fixed internal literals plus a line/column/char offset, never
# the document's own bytes; `OSError` carries the path (OS-length-bounded)
# and a fixed errno string; `UnicodeDecodeError` carries one byte value and
# a position; a `from_dict` rejection names a type or a field — none of
# those can embed an artifact-derived string of attacker-chosen length.
# The two `internal_error` sites are the exception: they catch bare
# `Exception` as the last-resort bug boundary (see `build_memory_evals`'s
# docstring), which is NOT bounded by construction the same way — a
# future code path that let a builtin conversion raise on a JSON-derived
# value (`int()`, `datetime.strptime()`, …) would leak it here unbounded.
# Today nothing does: every such conversion in this module already
# catches narrowly and routes the value through `_short_repr` at its own
# point of failure (`unparseable_run_stamp` above is the pattern to
# follow).  That is where a real fix belongs if a future conversion needs
# one — capping `str(exc)` at the wide `internal_error` catch would trade
# a shorter usually-fine bug message for no real protection, since a
# leaked value is not guaranteed to fall within the first 120 characters
# of it either.
#
# The invariant this enforces: no raw `!r` on an artifact-derived value in
# an `_issue` detail — held by the closure test
# `tests/test_memory_evals_data.py::TestAllIssueFieldsAreBounded` for the
# thirteen issue kinds its hostile tree exercises (see `required_kinds`
# there), so an edit that reintroduces a raw `!r` at one of THOSE sites
# fails loudly instead of silently reopening this exposure.  That test
# does not enumerate the module's full issue-kind vocabulary: a brand-new
# `_issue` call, or one for a kind its hostile tree does not already
# trigger, is not covered by it and needs its own per-site bound and
# test.  One further deliberate exception stays uncapped: the
# `seen_kinds` dedup key below (an internal `repr`, never emitted —
# capping it would collide two distinct oversized kinds on their shared
# prefix and silently swallow a real second issue).
#
# `_issue`'s structured `eval_id=` kwarg is handled by rejection at the
# read boundary instead of a cap; see `_MAX_EVAL_ID_LENGTH` below for why.
#
# `_issue`'s structured `path=` kwarg needs no cap at all: every `path=`
# argument in this module is filesystem-derived — a `Path` built from the
# artifact-tree walk, or the corpus walk's `Unreadable.path` — never
# artifact-derived, hence bounded by construction rather than by any knob
# here.
_MAX_DISCARDED_VALUE_REPR = 120

# `_issue`'s structured `eval_id=` kwarg is an IDENTITY field — the
# dashboard groups and links issue rows by it — so `_short_repr` above is
# the wrong tool for it: a truncated identity reads as a real, distinct
# eval_id and is silently WRONG, which is worse than the size exposure it
# would fix.  The remedy is instead rejection at the read boundary, in
# `_read_verdicts` below.
#
# Rejection is lossless: the only `eval_id` that can ever link a row is an
# eval directory's own `name` (`_build_eval` assigns `eval_id =
# eval_dir.name`, and `_RowClaims.verdict_keys` is keyed on it), and a single path
# component is bounded by POSIX `NAME_MAX` on every filesystem this runs
# on.  An `eval_id` longer than that was therefore already unmatchable by
# construction — indexing it could only ever carry the artifact's own
# unbounded bytes into every poll's payload, never link a real row.
#
# The comparison is in CHARACTERS, not encoded bytes: UTF-8 is at least
# one byte per character, so `len(s) > _MAX_EVAL_ID_LENGTH` implies
# `len(s.encode()) > _MAX_EVAL_ID_LENGTH` too — conservative in the safe
# direction — and it never allocates an encoded copy of the hostile
# multi-megabyte string this guard exists to contain.
#
# Held by `tests/test_memory_evals_data.py::TestVerdictEvalIdLengthIsBounded`
# and `tests/test_memory_evals_data.py::TestAllIssueFieldsAreBounded`.
_MAX_EVAL_ID_LENGTH = 255

# ``(eval_id, metric_id) -> {item -> entry}``; a window metric's entry has no
# ``item`` and is keyed ``None``.
_VerdictIndex = dict[tuple[str, str], dict[str | None, dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class _RowClaims:
    """What the payload's rows have accounted for so far in its two joins.

    ``verdict_keys``: the verdict-index keys some row read.  ``linked``: the
    fingerprints some row, or the storm block, links.  ``suppressed``: the
    fingerprints a row claims while a triggered storm suppressed per-metric
    links.
    """

    verdict_keys: frozenset[tuple[str, str]] = frozenset()
    linked: frozenset[str] = frozenset()
    suppressed: frozenset[str] = frozenset()

    def with_verdict_key(self, key: tuple[str, str]) -> _RowClaims:
        return replace(self, verdict_keys=self.verdict_keys | {key})

    def with_link(self, fingerprint: str) -> _RowClaims:
        return replace(self, linked=self.linked | {fingerprint})

    def with_suppressed(self, fingerprint: str) -> _RowClaims:
        return replace(self, suppressed=self.suppressed | {fingerprint})


@dataclass(frozen=True, slots=True)
class _JoinSources:
    """The program-wide inputs, read once per build, that every eval's rows join against or repeat.

    ``storm`` is the projected storm block: each eval's payload repeats it whole,
    while the escalation join reads only whether it is set.
    """

    verdicts: _VerdictIndex
    escalations: Mapping[str, Escalation]
    storm: dict[str, Any] | None


def _load_json(path: Path) -> Any:
    """Parse *path*, or raise for the caller's narrow handler to record."""
    return json.loads(path.read_text())


def _issue(
    issues: list[dict[str, Any]],
    kind: str,
    *,
    eval_id: str | None = None,
    path: Path | str | None = None,
    detail: str = '',
) -> None:
    """Record one degraded artifact — named, located, and counted (DD6/INV-2)."""
    issues.append({
        'kind': kind,
        'eval_id': eval_id,
        'path': str(path) if path is not None else None,
        'detail': detail,
    })


def _short_repr(value: Any) -> str:
    """A length-capped ``repr`` for naming a discarded or unvalidated value in an issue detail.

    Builds the full ``repr`` before capping it, so the OUTPUT is bounded but
    this transient string is not.  Accepted rather than a streaming bound
    (e.g. ``reprlib.Repr``): ``json.loads`` already paid an O(n) cost to parse
    *value* (or the record it came from) off disk into the object graph being
    repr'd here, so this adds no new order of magnitude of work — just one
    extra allocation, the same size as one already paid for, that is
    discarded immediately.  Revisit if that extra allocation itself ever
    becomes the bottleneck.
    """
    text = repr(value)
    if len(text) <= _MAX_DISCARDED_VALUE_REPR:
        return text
    return text[:_MAX_DISCARDED_VALUE_REPR] + '…'


def _verdict_identity(metric_id: Any, item: Any) -> str:
    """How an issue detail names one verdict entry: its metric, then its item when it has one."""
    named = f'metric {_short_repr(metric_id)}'
    return named if item is None else f'{named} item {_short_repr(item)}'


def _read_limits(
    eval_dir: Path,
    latest_run_stamp: str | None,
    issues: list[dict[str, Any]],
    *,
    has_runs: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return ``(provenance_block, metric_id -> limits record)`` for one eval.

    Per-eval, NOT at the root: ``shared.memory_eval_limits.limits_artifact_path``
    returns ``<root>/<eval_id>/limits-current.json`` and the committed exemplar
    sits at ``e1-retrieval-health/limits-current.json``.  The committed
    fixtures are the contract.

    The artifact's embedded ``verdicts[]`` supplies ONE dashboard input —
    ``rule_kind``, the rule that governs each metric.  Its ``status`` /
    ``p_value`` / ``baseline`` / ``alarms`` are deliberately NOT read as verdict
    state: that vocabulary (``baseline_snapshot|ok|alarm|improved|
    insufficient_data``) is not the M2 verdict vocabulary the dashboard
    displays, and these records carry no ``fingerprint``, so they cannot
    participate in the escalation join at all.  Reading both sources would hand
    the UI two disagreeing alarm truths and force a dashboard-side mapping
    between the vocabularies — exactly the re-derivation G6/INV-5 forbids.
    ``verdicts-current.json`` is the sole verdict source.

    All three disposal paths are NAMED: absent (beside runs) is
    ``missing_limits``, a parse failure is ``unreadable_limits`` (recorded
    here, at the read boundary), and an artifact that parses but is not an
    object is ``malformed_limits``.  Without that third path a present-but-wrong-type
    artifact produced no signal at all, since the absent-file issue does not
    fire for a file that exists.

    The embedded ``verdicts[]`` gets the same treatment one level down: a
    ``verdicts`` field that is not a list is ``malformed_limits_verdicts``, and
    records within it that cannot be keyed are counted into one
    ``unidentified_limits_verdicts``.  Both are scoped to what is actually
    lost — the per-metric ``rule_kind`` — and neither discards the provenance
    block, which parsed fine and is still worth displaying.
    """
    path = eval_dir / 'limits-current.json'
    if not path.is_file():
        if has_runs:
            # Metrics on disk with no limits beside them means the evaluator is
            # not completing — a real silent-failure mode, since the UI would
            # otherwise show trends with a blank provenance block and look fine.
            _issue(
                issues,
                'missing_limits',
                eval_id=eval_dir.name,
                path=path,
                detail=f'{eval_dir.name} has metrics runs but no limits artifact',
            )
        return None, {}

    # Scoped to the read+parse itself, exactly as ``_read_run`` already does
    # for each metrics run.  Wrapping the WHOLE function instead (as the caller
    # used to) meant a code bug anywhere below was reported as an unreadable
    # artifact.
    try:
        body = _load_json(path)
    except _ARTIFACT_ERRORS as exc:
        _issue(issues, 'unreadable_limits', eval_id=eval_dir.name, path=path, detail=str(exc))
        return None, {}

    if not isinstance(body, dict):
        # Valid JSON of the wrong shape — `_load_json` does not raise, so the
        # `unreadable_limits` handler above never fires, and the file EXISTS
        # so `missing_limits` above did not fire either.  Discarding it
        # silently would lose the alpha/baseline/grandfather-hash provenance
        # with no signal anywhere.  Carries an `eval_id`: unlike the
        # root-scoped verdicts artifact, this one is per-eval.
        _issue(
            issues, 'malformed_limits', eval_id=eval_dir.name, path=path,
            detail=f'expected an object, got {type(body).__name__}',
        )
        # Provenance stays absent rather than half-populated, and `rule_kind`
        # on every metric row stays None.
        return None, {}

    # ``.get()`` throughout: a schema addition upstream is inert here, never
    # fatal, and an omitted field reads as absent rather than crashing.
    block = {key: body.get(key) for key in _LIMITS_PROVENANCE_KEYS}
    # One string comparison, not a re-derivation.  The committed exemplar
    # exhibits this skew itself (limits stamped 20260704, newest metrics
    # 20260705); displaying the alpha/baseline provenance beside a newer
    # current value without disclosing it would present stale provenance as
    # though it governed the displayed run.
    block['stale_for_latest_run'] = body.get('run_stamp') != latest_run_stamp

    records = body.get('verdicts')
    if records is None:
        records = []
    elif not isinstance(records, list):
        # The body IS an object, so `malformed_limits` above did not fire and
        # the provenance block is perfectly usable.  But iterating a non-list
        # here would walk a dict's KEYS (or a string's characters), fail every
        # record check below, and leave EVERY metric's `rule_kind` None with no
        # signal — the `entries` defect of the verdicts reader, one function
        # earlier.  Scoped to what is actually lost: the rule kinds, not the
        # provenance that parsed fine.
        _issue(
            issues, 'malformed_limits_verdicts', eval_id=eval_dir.name, path=path,
            detail=(
                f'expected a list "verdicts" field, got {type(records).__name__}; '
                'no metric rule_kind could be read'
            ),
        )
        records = []

    by_metric: dict[str, Any] = {}
    unkeyable = 0
    for record in records:
        if not isinstance(record, dict):
            unkeyable += 1
            continue
        metric_id = record.get('metric_id')
        if not isinstance(metric_id, str) or not metric_id:
            unkeyable += 1
            continue
        if metric_id in by_metric:
            # Last-wins would silently change the displayed ``rule_kind``.
            _issue(
                issues, 'duplicate_limits_verdict', eval_id=eval_dir.name, path=path,
                detail=f'metric {_short_repr(metric_id)} has more than one limits verdict; the first is used',
            )
            continue
        by_metric[metric_id] = record
    if unkeyable:
        # Counted once per file rather than one issue apiece, mirroring
        # `_by_metric_id`: a systematically broken artifact should cost one row
        # of the issues list, not flood it into uselessness.
        _issue(
            issues, 'unidentified_limits_verdicts', eval_id=eval_dir.name, path=path,
            detail=(
                f'{unkeyable} limits verdict record(s) carry no usable "metric_id"; '
                'their metrics report no rule_kind'
            ),
        )
    return block, by_metric


def _by_metric_id(
    body: dict[str, Any],
    eval_id: str,
    path: Path,
    issues: list[dict[str, Any]],
) -> dict[str, dict]:
    """Index one run's metric records by ``metric_id``, naming every discard.

    ``metric_id`` is unique within a run (M1 §"Record schema"), so a dict
    comprehension over the array reads as safe — but on a duplicate it keeps
    whichever record happened to come LAST, and the row's ``current_value`` and
    every scalar beside it then silently become one of two candidates with
    nothing in the payload saying so.  A record carrying no usable ``metric_id``
    is likewise unchartable and was previously filtered away without a trace.

    Both discards are NAMED (DD6/INV-2) and the survivor is the FIRST record,
    deterministically: with a duplicate neither candidate is more correct, so
    the payload picks one rule, states it in the issue, and lets the operator
    see that a choice was made at all.  The unidentified records are reported
    once per file with a count rather than one issue apiece, so a systematically
    broken artifact degrades one row instead of flooding the issues list.
    """
    by_id: dict[str, dict] = {}
    unidentified = 0
    for record in body.get('metrics') or []:
        metric_id = record.get('metric_id') if isinstance(record, dict) else None
        if not isinstance(metric_id, str) or not metric_id:
            unidentified += 1
            continue
        if metric_id in by_id:
            _issue(
                issues, 'duplicate_metric_id', eval_id=eval_id, path=path,
                detail=f'metric {_short_repr(metric_id)} appears more than once in this run; the first record is used',
            )
            continue
        by_id[metric_id] = record
    if unidentified:
        _issue(
            issues, 'unidentified_metrics', eval_id=eval_id, path=path,
            detail=f'{unidentified} metric record(s) carry no usable "metric_id" and cannot be charted',
        )
    return by_id


def _read_verdicts(
    root: Path,
    eval_dirs: list[Path],
    issues: list[dict[str, Any]],
) -> tuple[_VerdictIndex, dict[str, Any] | None]:
    """Index the ROOT verdicts artifact by ``(eval_id, metric_id, item)``.

    That triple is the fingerprint identity, so a tripwire contributes one
    entry per item; see ``shared/tests/fixtures/memory_eval_verdicts/README.md``
    §"Entry granularity".

    Root-scoped, not per-eval: ``docs/prds/memory-eval-program.md`` pins
    ``fused-memory/data/memory-evals/verdicts-<STAMP>.json``, and the
    ``storm_escape`` block is per-RUN across the whole program, which only
    makes sense above the eval dirs.

    This artifact is the SOLE verdict source.  Its ``verdict`` strings are
    passed through unmapped — a dashboard-side translation table would be a
    second vocabulary to keep in sync with the evaluator's, and the first
    divergence would silently mislabel an alarm.

    Every disposal path is NAMED: absent is ``missing_verdicts``, a parse
    failure is ``unreadable_verdicts`` (recorded here, at the read boundary), a
    body that parses but is not an object — or is an object whose ``entries``
    is absent or not a list — is ``malformed_verdicts``, an individual
    non-object element is ``malformed_verdict_entry``, and elements that ARE
    objects but carry no usable ``(eval_id, metric_id)`` pair, or an ``item``
    that is present but not a non-empty string, are counted into one
    ``unidentified_verdicts``.  The ``entries`` paths matter most: a
    discarded verdicts body leaves every row's verdict absent, which is
    indistinguishable from a healthy no-alarm tree unless the discard is
    recorded — and the record-level form is the same confusion one row at a
    time, since an unkeyable entry also renders as ``verdict: None``.

    A malformed ``entries`` does NOT suppress ``storm_escape``: the storm block
    is run-scoped and parses independently, so dropping it too would turn one
    named degradation into a second silent one.

    ``missing_verdicts`` is gated on a RUN existing, not on an eval directory
    existing — a registered eval that has never run is empty-but-healthy, the
    same standing the absent root gets, and ``_read_limits`` gates on the same
    signal.
    """
    path = root / 'verdicts-current.json'
    if not path.is_file():
        # Gated on any RUN existing, not on the eval DIRECTORIES existing.  An
        # eval that is registered but has never run is the same
        # empty-but-healthy state as an absent root, which this module
        # deliberately does not flag; firing here would put a standing issue on
        # a tree with nothing wrong and train operators past the list.
        # ``_read_limits`` draws the line in exactly this place (``has_runs``),
        # and two readers disagreeing about what "never run" means would make
        # the two artifacts' absence stories contradict each other.
        if any(next(d.glob('metrics-*.json'), None) is not None for d in eval_dirs):
            # Metrics on disk with nothing judging them: the UI would show
            # trends beside a blank verdict column and look healthy.
            misplaced = [d / 'verdicts-current.json' for d in eval_dirs if (d / 'verdicts-current.json').is_file()]
            detail = 'no verdicts artifact at the memory-evals root'
            if misplaced:
                # Named, never silently fallen back to: if the producer's
                # location drifts, that drift stays visible as an issue rather
                # than being papered over by a reader that guesses.
                detail += (
                    '; a verdicts artifact exists at '
                    + ', '.join(str(p) for p in misplaced)
                    + ' but is NOT read from there'
                )
            _issue(issues, 'missing_verdicts', path=path, detail=detail)
        return {}, None

    # Scoped to the read+parse itself; see ``_read_limits`` for why the
    # whole-function wrapper this replaces was wrong.
    try:
        body = _load_json(path)
    except _ARTIFACT_ERRORS as exc:
        # An unreadable verdicts artifact must never read as "nothing
        # alarmed" — every verdict stays absent and the failure is named.
        _issue(issues, 'unreadable_verdicts', path=path, detail=str(exc))
        return {}, None

    if not isinstance(body, dict):
        # Valid JSON of the wrong shape: `_load_json` does not raise, so the
        # `unreadable_verdicts` handler above never fires.  Discarding it
        # silently here would make a broken verdicts file read exactly like a
        # healthy no-alarm tree.  `eval_id` is omitted because this artifact is
        # root-scoped, matching `missing_verdicts` above.
        _issue(
            issues, 'malformed_verdicts', path=path,
            detail=f'expected an object with an "entries" list, got {type(body).__name__}',
        )
        return {}, None

    entries = body.get('entries')
    if entries is None:
        # Absent entirely — same standing as a body of the wrong type: every
        # row's verdict goes absent, which reads exactly like a healthy
        # no-alarm tree unless the discard is named.
        _issue(
            issues, 'malformed_verdicts', path=path,
            detail='expected an "entries" list, got no such field',
        )
        entries = []
    elif not isinstance(entries, list):
        # The body IS an object but `entries` is not a list.  Iterating it
        # directly would walk a dict's KEYS (or a string's characters), fail
        # every `isinstance(entry, dict)` test below, and yield an empty index
        # with no signal anywhere — the costly silent discard this reader
        # exists to prevent.  Named, then treated as empty.
        _issue(
            issues, 'malformed_verdicts', path=path,
            detail=f'expected a list "entries" field, got {type(entries).__name__}',
        )
        entries = []

    index: _VerdictIndex = {}
    unkeyable = 0
    overlong_eval_id = 0
    for entry in entries:
        if not isinstance(entry, dict):
            # A non-object element cannot carry an (eval_id, metric_id) key, so
            # it can never link a row.  Named rather than dropped, per the
            # module's no-silent-discard invariant.
            _issue(
                issues, 'malformed_verdict_entry', path=path,
                detail=f'entries element is {type(entry).__name__}, not an object; skipped',
            )
            continue
        eval_id = entry.get('eval_id')
        metric_id = entry.get('metric_id')
        item = entry.get('item')
        eval_id_overlong = isinstance(eval_id, str) and len(eval_id) > _MAX_EVAL_ID_LENGTH
        if (
            not isinstance(eval_id, str) or not isinstance(metric_id, str)
            or not eval_id or not metric_id
            or not (item is None or (isinstance(item, str) and item))
            or eval_id_overlong
        ):
            # An object, but one that can never be keyed onto a row.  Dropping
            # it silently leaves that row's verdict absent — indistinguishable
            # from "no entry was ever written for it", which is the same
            # confusion the artifact-level guards above exist to prevent, just
            # one row at a time instead of the whole tree.  Split into two
            # counters below — not a second gate, this entry is unkeyable
            # either way — because the operator's next step differs: a
            # missing/malformed key points at the producer's record shape, an
            # over-length one at a specific runaway `eval_id`.
            if eval_id_overlong:
                overlong_eval_id += 1
            else:
                unkeyable += 1
        else:
            items = index.setdefault((eval_id, metric_id), {})
            if item in items:
                # Two judgements of one fingerprint identity in one run:
                # last-wins would pick the row's verdict AND its fingerprint
                # (hence its escalation link) by array order alone.  First
                # wins, and the drop is named.
                _issue(
                    issues, 'duplicate_verdict_entry', eval_id=eval_id, path=path,
                    detail=(
                        f'{_verdict_identity(metric_id, item)} has more than one verdict '
                        'entry; the first is used'
                    ),
                )
                continue
            items[item] = entry
    if unkeyable or overlong_eval_id:
        # Counted once per file rather than one issue apiece, mirroring
        # `_by_metric_id`: a systematically broken artifact should cost one row
        # of the issues list, not flood it into uselessness.
        _issue(
            issues, 'unidentified_verdicts', path=path,
            detail=(
                f'{unkeyable} verdict entr(ies) carry no usable "eval_id"/"metric_id" pair '
                '(absent, empty, or not a string) or carry an "item" that is not a non-empty '
                f'string, and {overlong_eval_id} carry an "eval_id" longer than '
                f'{_MAX_EVAL_ID_LENGTH} characters; none can be matched to a metric row'
            ),
        )
    _name_shared_fingerprints(index, path, issues)

    # Run-scoped, which is why it lives beside the entries rather than inside
    # them.  Surfaced only while TRIGGERED — an untriggered block is the
    # ordinary case and would otherwise render as a permanent storm banner.
    storm = body.get('storm_escape')
    if not isinstance(storm, dict) or not storm.get('triggered'):
        storm = None
    return index, storm


def _name_shared_fingerprints(index: _VerdictIndex, path: Path, issues: list[dict[str, Any]]) -> None:
    """Name every indexed verdict entry whose fingerprint an earlier one already carries.

    A fingerprint names one ``(eval_id, metric_id, item)`` identity and the
    escalation join is whole-string equality on it, so two entries sharing one
    would both claim the same escalation.  ``_join_escalation`` links it to
    the first row that claims it; this makes the sharing itself visible.
    """
    holders: dict[str, tuple[str, str, str | None]] = {}
    for (eval_id, metric_id), items in index.items():
        for item, entry in items.items():
            fingerprint = entry.get('fingerprint')
            if not isinstance(fingerprint, str) or not fingerprint:
                continue
            holder = holders.setdefault(fingerprint, (eval_id, metric_id, item))
            if holder == (eval_id, metric_id, item):
                continue
            held_eval_id, held_metric_id, held_item = holder
            _issue(
                issues, 'duplicate_verdict_fingerprint', eval_id=eval_id, path=path,
                detail=(
                    f'{_verdict_identity(metric_id, item)} carries fingerprint {_short_repr(fingerprint)}, '
                    f'as does {_verdict_identity(held_metric_id, held_item)} in eval '
                    f'{_short_repr(held_eval_id)}; an escalation carrying it links only whichever '
                    'of their rows comes first in the payload'
                ),
            )


def _escalation_projection(escalation: Escalation) -> dict[str, Any]:
    """The fields a metric row (or ``unmatched_escalations``) carries.

    ``created_at`` is sourced from the queue record's ``timestamp``.  The
    fingerprint is echoed so an operator can see WHY a row linked, without the
    UI having to cross-reference two fields.
    """
    return {
        'id': escalation.id,
        'summary': escalation.summary,
        'severity': escalation.severity,
        'level': escalation.level,
        'created_at': escalation.timestamp,
        'dedupe_fingerprint': escalation.dedupe_fingerprint,
    }


def _unmatched_projection(escalation: Escalation, reason: str) -> dict[str, Any]:
    """An open escalation that no metric row links, plus WHY it carries no link.

    ``no_matching_verdict`` — no row in this payload claims this fingerprint.
    That covers genuine contract drift, escalations no metric row can ever
    explain (the runner-failure self-report is itself an ``eval_regression``)
    and, beside an ``internal_error`` naming the eval, a row lost to a
    dashboard bug.
    ``storm_suppressed`` — a metric row DOES claim this fingerprint, but this
    run's storm escape collapsed per-metric links into the aggregate, so the
    row deliberately renders none.  Per-metric filings from EARLIER runs stay
    open across a storm (program PRD §M3), so this is the ordinary case rather
    than a rare one.
    ``no_fingerprint`` — the record carries no usable ``dedupe_fingerprint``,
    so it could not be joined by ANY row: the failure is upstream of the
    matching question rather than an answer to it.  Distinguished from
    ``no_matching_verdict`` because the operator's next step differs — a
    missing fingerprint is a producer bug, not contract drift between the
    evaluator and the tree.

    Without the discriminator both land in one undifferentiated list that the
    payload calls "no metric row explains this" — which would fire on
    escalations that are in fact fully explained, and train operators to ignore
    the one signal that catches a real parity orphan.
    """
    return {**_escalation_projection(escalation), 'reason': reason}


def _name_unreadable_queue_files(scan: QueueScan, issues: list[dict[str, Any]]) -> None:
    """One ``unreadable_escalation_file`` per live-queue file the corpus could not read.

    The ``detail`` is deliberately CONDITIONAL ("if it held an open
    ``eval_regression`` escalation…"): the queue is shared across every
    category and status, and an unreadable file's category and status are
    exactly what is not known.  Its ``path`` is the FILE, unlike the
    record-level issues, which name the queue dir.  The singular kind stays
    separable from ``unreadable_escalations``, which means no walk was read.
    """
    for entry in scan.unreadable:
        if entry.location is not Location.ROOT:
            continue
        _issue(
            issues, 'unreadable_escalation_file', path=entry.path,
            detail=(
                f'escalation queue file could not be read ({entry.error}); '
                'if it held an open eval_regression escalation it is missing '
                'from this payload, so unmatched_escalations cannot be read '
                'as exhaustive'
            ),
        )


def _index_escalations(
    corpus: Datum[EscalationCorpus],
    escalation_queue: str,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Escalation], list[Escalation]]:
    """Index open ``eval_regression`` escalations by ``dedupe_fingerprint``.

    Returns ``(fingerprint -> escalation, open escalations that carry no usable
    fingerprint)``.  The second list exists because the fingerprint is the
    index KEY: a record without one cannot be represented in the index at all,
    yet it is still an open alarm the payload owes the operator.

    The source is the escalation corpus' ROOT records of *escalation_queue* —
    the live queue, the same population the Escalations tab lists — and its
    root unreadable files are named (:func:`_name_unreadable_queue_files`),
    because a corrupt file holding an open alarm is precisely what makes
    ``unmatched_escalations`` non-exhaustive.  An unknown corpus is one
    ``unreadable_escalations`` issue carrying the corpus' own reason, never
    read as "nothing is escalated".

    Openness is filtered EXPLICITLY on ``status``, not inferred from a record
    lying at the queue root: ``escalation.queue._archive_resolved`` is
    best-effort, and on ``OSError`` leaves the RESOLVED record in the root.
    The test is POSITIVE (``status`` is ``pending``), so a status vocabulary
    that grows a new terminal state does not silently start rendering as open;
    ``park()`` keeps a genuinely-open L2 at ``pending``.  An UNRECOGNISED
    status, a null one included, is skipped and NAMED, unlike the
    resolved/dismissed skip, which is the filter doing its job.
    ``Escalation.from_dict`` validates no field types, so ``status`` and
    ``dedupe_fingerprint`` are type-guarded before use.

    Matching is ``==`` over the WHOLE fingerprint string; nothing here parses
    the producer's private construction of it.

    Two open escalations sharing a fingerprint (a dedupe that did not take) is
    named with BOTH ids; the first in walk order is used.  The loser is not
    carried into ``unmatched_escalations``: it shares the winner's fingerprint,
    so listing it could fire the parity signal on an escalation a metric row
    fully explains.  An OPEN record with no usable fingerprint can never join
    any row and has no sibling carrying its signal, so it alone is both named
    and returned for surfacing.
    """
    if corpus.value is None:
        _issue(
            issues, 'unreadable_escalations',
            detail=f'escalation queue {escalation_queue!r} was not read: {corpus.reason}',
        )
        return {}, []
    scan = corpus.value.scan(escalation_queue)
    _name_unreadable_queue_files(scan, issues)
    queue_dir = scan.queue.directory
    index: dict[str, Escalation] = {}
    unfingerprinted: list[Escalation] = []
    for record in scan.records:
        escalation = record.escalation
        if record.location is not Location.ROOT or escalation.category != _EVAL_REGRESSION_CATEGORY:
            continue
        status = escalation.status
        if status != _OPEN_ESCALATION_STATUS:
            # ``isinstance`` FIRST: an unhashable status (``[]``, ``{}``) would
            # make the set membership test raise ``TypeError``.
            if isinstance(status, str) and status in _CLOSED_ESCALATION_STATUSES:
                continue
            _issue(
                issues, 'unknown_escalation_status', path=queue_dir,
                detail=(
                    f'escalation {_short_repr(escalation.id)} has unrecognised status '
                    f'{_short_repr(status)}; not joined (only pending escalations are '
                    'treated as open)'
                ),
            )
            continue
        fingerprint = escalation.dedupe_fingerprint
        if not (isinstance(fingerprint, str) and fingerprint):
            _issue(
                issues, 'unfingerprinted_escalation', path=queue_dir,
                detail=(
                    f'escalation {_short_repr(escalation.id)} is open but carries no usable '
                    f'dedupe_fingerprint ({_short_repr(fingerprint)}); it can never join '
                    'a metric row'
                ),
            )
            unfingerprinted.append(escalation)
            continue
        if fingerprint in index:
            _issue(
                issues, 'duplicate_escalation_fingerprint', path=queue_dir,
                detail=(
                    f'escalation {_short_repr(escalation.id)} shares dedupe_fingerprint '
                    f'{_short_repr(fingerprint)} with {_short_repr(index[fingerprint].id)}; '
                    'the first is used'
                ),
            )
            continue
        index[fingerprint] = escalation
    return index, unfingerprinted


def _verdict_class(verdict: Any) -> str:
    """Bucket a raw verdict value into the closed rendering vocabulary.

    Returns one of ``alarm``, ``no_alarm``, ``insufficient_data``,
    ``grandfathered``, ``unjudged`` (the value was absent) or
    ``unknown_verdict`` (the value is present but outside the M2 vocabulary,
    including a value that is not a string at all).

    ``isinstance`` FIRST, exactly as the metric-kind loop does: ``verdict`` is
    an unvalidated JSON value, so an unhashable one (``[]``, ``{}``) would make
    ``in _KNOWN_VERDICTS`` raise ``TypeError``.
    """
    if verdict is None:
        return 'unjudged'
    if isinstance(verdict, str) and verdict in _KNOWN_VERDICTS:
        return verdict
    return 'unknown_verdict'


def _parity(verdict_class: str, escalation: dict[str, Any] | None) -> str:
    """The derived display state for one metric row.

    A pure, TOTAL ``(verdict class, linked?)`` lookup — not a statistic.
    Deriving it once here keeps the UI from re-deriving badge state out of
    three separate fields, which is where the two sides would drift apart.
    The full closed output vocabulary is :data:`PARITY_STATES`.

    Every verdict class gets its OWN pair of states rather than being collapsed
    into ``recovered_open``/``clear``.  The collapsed mapping made three
    distinct facts render as one:

    * ``insufficient_data`` — the evaluator lacked the samples to judge at all.
      With a live escalation that is not a recovery; nothing was shown to have
      improved.
    * ``grandfathered`` — a known-bad measurement the ratchet deliberately
      exempts.  It is an outstanding exception, not a healthy metric.
    * an ABSENT verdict — nothing judged this metric.  ``clear`` means "judged,
      and not alarming", which is a materially stronger claim.

    Lending any of them the healthy label is the module's own stated failure
    mode: fail toward "visibly unrenderable", never toward the healthy label.

    The naming asymmetry is deliberate and load-bearing.  For an ``alarm`` the
    UNLINKED state is the notable one — a filed alarm with no escalation behind
    it is the parity violation — so it is the one that gets the distinguishing
    name (``alarmed_unlinked``).  For every other class the LINKED state is the
    notable one, so that is the one carrying the ``_open`` suffix.

    Every pre-existing name is preserved verbatim; the change is purely
    additive for existing consumers.  ``recovered_open`` and ``clear`` are
    NARROWED to the ``no_alarm`` verdict, which only makes them truer:
    ``recovered_open`` now means an escalation filed on a prior alarm with the
    metric reading ``no_alarm`` today, which IS a recovery.

    Reachability of ``unjudged_open``, which is not obvious: the row-level join
    only runs when the verdict ENTRY carries a fingerprint (see
    ``_join_escalation``), so a row no ``(eval_id, metric_id, item)`` entry
    names can never link.  The
    linked variant is reached only by an entry that has a ``fingerprint`` and
    no ``verdict`` field — exactly the case that used to claim a recovery that
    never happened.  The function stays total over both axes anyway, so no
    future call site can fall into an unhandled corner.
    """
    if verdict_class == 'alarm':
        return 'alarmed_open' if escalation else 'alarmed_unlinked'
    if verdict_class == 'no_alarm':
        return 'recovered_open' if escalation else 'clear'
    return f'{verdict_class}_open' if escalation else verdict_class


def _item_order(item: str | None) -> tuple[bool, str]:
    """Row order within one metric: the item-less (window) entry, then items lexically."""
    return item is not None, item or ''


def _join_escalation(
    fingerprint: Any,
    sources: _JoinSources,
    claims: _RowClaims,
) -> tuple[Escalation | None, _RowClaims]:
    """The escalation a row carrying *fingerprint* links, and *claims* extended by that row's claim.

    Whole-string equality, NOT gated on the verdict: a recovered metric whose
    escalation is still open keeps its link.  A triggered storm filed one
    aggregate instead of per-metric escalations, so it suppresses the link and
    records the claim, letting ``unmatched_escalations`` say why.  A
    fingerprint links only its first claimer; ``_name_shared_fingerprints``
    names the sharing.
    """
    if not (isinstance(fingerprint, str) and fingerprint) or fingerprint not in sources.escalations:
        return None, claims
    if sources.storm is not None:
        return None, claims.with_suppressed(fingerprint)
    if fingerprint in claims.linked:
        return None, claims
    return sources.escalations[fingerprint], claims.with_link(fingerprint)


def _verdict_row_fields(
    entry: dict[str, Any] | None,
    metric_id: str,
    eval_dir: Path,
    matched: Escalation | None,
    storm_triggered: bool,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """The verdict half of one metric row: *entry*'s fields, its escalation link and its parity.

    The fields are passed through UNMAPPED from the sole verdict source.  Their
    ``value`` is what the evaluator judged, while the row's ``current_value``
    is what the metrics artifact says: two sources, two fields, never
    conflated.
    """
    # Absent verdict == absent, never defaulted to 'no_alarm'.  The empty
    # dict makes every verdict field read as None below without a second
    # code path.
    judged: dict[str, Any] = entry or {}
    escalation = _escalation_projection(matched) if matched is not None else None

    # Classified ONCE and reused for both the parity badge and the
    # unrenderable-value check below, so the membership rule lives in
    # exactly one place.
    verdict = judged.get('verdict')
    verdict_class = _verdict_class(verdict)
    if verdict_class == 'unknown_verdict':
        # A verdict outside the closed M2 set has no badge, so it is named
        # rather than folded into the healthy label — the same standing as
        # ``unknown_kind`` (no chart primitive) and
        # ``unknown_escalation_status`` (unclassifiable).  An ABSENT
        # verdict is deliberately NOT named: "nothing judged this metric"
        # is a legitimate state the payload already models as ``unjudged``
        # parity, and at the artifact level as
        # ``missing_verdicts``/``orphan_verdict``.
        #
        # ``eval_dir.parent`` IS the memory-evals root, which is where
        # ``_read_verdicts`` locates the sole verdicts artifact — this
        # value came out of that file, not out of the eval's own dir.
        _issue(
            issues, 'unknown_verdict', eval_id=eval_dir.name,
            path=eval_dir.parent / 'verdicts-current.json',
            detail=(
                f'{_verdict_identity(metric_id, judged.get("item"))} has verdict '
                f'{_short_repr(verdict)}, which is outside the M2 vocabulary and has no parity badge'
            ),
        )

    return {
        'verdict': verdict,
        'verdict_detail': judged.get('detail'),
        'value': judged.get('value'),
        'item': judged.get('item'),
        'limit_ref': judged.get('limit_ref'),
        'fingerprint': judged.get('fingerprint'),
        'run_stamp': judged.get('run_stamp'),
        'escalation': escalation,
        'parity': (
            'storm_collapsed'
            if storm_triggered and verdict == 'alarm'
            else _parity(verdict_class, escalation)
        ),
    }


@dataclass(frozen=True, slots=True)
class _Run:
    """One parsed metrics run; ``path`` names its file, since two runs may share a stamp."""

    stamp: Any
    metrics: dict[str, dict[str, Any]]
    corpus: dict[str, Any] | None
    path: Path


@dataclass(frozen=True, slots=True)
class _RunSeries:
    """The parsed trailing window of one eval's runs, oldest first, and how much of the disk it covers."""

    runs: tuple[_Run, ...]
    runs_on_disk: int
    truncated: bool


def _read_run(path: Path, eval_id: str, issues: list[dict[str, Any]]) -> _Run | None:
    """Parse one metrics run, or name why it cannot be charted and return None.

    Validates only what the reader must READ and RENDER; the M1 schema's
    cross-field rules are the producer's to reject at emit time.
    """
    try:
        body = _load_json(path)
    except _ARTIFACT_ERRORS as exc:
        _issue(issues, 'unreadable_metrics', eval_id=eval_id, path=path, detail=str(exc))
        return None

    if not isinstance(body, dict) or not isinstance(body.get('metrics'), list):
        _issue(
            issues, 'malformed_metrics', eval_id=eval_id, path=path,
            detail=f'expected an object with a list "metrics" field, got {type(body).__name__}',
        )
        return None

    stamp = body.get('run_stamp')
    if stamp is None:
        _issue(
            issues, 'missing_run_stamp', eval_id=eval_id, path=path,
            detail='run artifact carries no in-body run_stamp',
        )
    corpus = body.get('corpus')
    return _Run(
        stamp=stamp,
        metrics=_by_metric_id(body, eval_id, path, issues),
        corpus=dict(corpus) if isinstance(corpus, dict) else None,
        path=path,
    )


def _read_run_series(eval_dir: Path, issues: list[dict[str, Any]]) -> _RunSeries:
    """Parse the newest ``_TREND_RUN_CAP`` of *eval_dir*'s metrics runs, ordered by FILENAME.

    Filename order is the producer's own (``shared.memory_eval_metrics.load_series_window``):
    the stamp is zero-padded UTC.  A run the window drops is disclosed as ``truncated``.
    """
    all_paths = sorted(eval_dir.glob('metrics-*.json'))
    window = all_paths[-_TREND_RUN_CAP:] if _TREND_RUN_CAP else all_paths
    parsed = (_read_run(path, eval_dir.name, issues) for path in window)
    return _RunSeries(
        runs=tuple(run for run in parsed if run is not None),
        runs_on_disk=len(all_paths),
        truncated=len(window) < len(all_paths),
    )


def _run_age_seconds(
    eval_dir: Path,
    latest_run_stamp: Any,
    now: datetime,
    issues: list[dict[str, Any]],
) -> float | None:
    """How old the newest run is, for DISPLAY only; an unparseable stamp is named and reads as None."""
    if not isinstance(latest_run_stamp, str):
        return None
    try:
        run_at = datetime.strptime(latest_run_stamp, _RUN_STAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        _issue(
            issues, 'unparseable_run_stamp', eval_id=eval_dir.name, path=eval_dir,
            detail=f'run_stamp {_short_repr(latest_run_stamp)} is not in {_RUN_STAMP_FORMAT}',
        )
        return None
    return (now - run_at).total_seconds()


def _name_unrenderable_kinds(
    eval_id: str,
    metric_id: str,
    runs: tuple[_Run, ...],
    issues: list[dict[str, Any]],
) -> None:
    """Name each distinct absent or unknown kind *metric_id* carries anywhere in the window.

    Deduped on the RAW ``repr``, never capped, so two oversized kinds sharing a
    prefix stay distinct.  A run the metric is absent from is a trend hole, not a defect.
    """
    seen_kinds: set[str] = set()
    for run in runs:
        record = run.metrics.get(metric_id)
        if record is None:
            continue
        run_kind = record.get('kind')
        if isinstance(run_kind, str) and run_kind in _KNOWN_KINDS:
            continue
        if repr(run_kind) in seen_kinds:
            continue
        seen_kinds.add(repr(run_kind))
        if run_kind is None:
            _issue(
                issues, 'missing_kind', eval_id=eval_id, path=run.path,
                detail=(
                    f'metric {_short_repr(metric_id)} carries no "kind" (a required M1 field), '
                    'so it has no chart primitive'
                ),
            )
        else:
            _issue(
                issues, 'unknown_kind', eval_id=eval_id, path=run.path,
                detail=(
                    f'metric {_short_repr(metric_id)} has kind {_short_repr(run_kind)}, '
                    'which has no chart primitive'
                ),
            )


def _metric_fields(metric_id: str, runs: tuple[_Run, ...], rule_kind: Any) -> dict[str, Any]:
    """The metric half of one row: the latest run's fields and a full-width trend.

    A run the metric is absent from is a ``None`` hole, keeping every series
    index-aligned to the shared axis.
    """
    current = runs[-1].metrics.get(metric_id, {})
    return {
        'metric_id': metric_id,
        'kind': current.get('kind'),
        'rule_kind': rule_kind,
        'current_value': current.get('value'),
        'n': current.get('n'),
        'denominator': current.get('denominator'),
        'direction': current.get('direction'),
        'trend': {
            'labels': [run.stamp for run in runs],
            'values': [run.metrics.get(metric_id, {}).get('value') for run in runs],
        },
    }


def _build_eval(
    eval_dir: Path,
    sources: _JoinSources,
    claims: _RowClaims,
    now: datetime,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], _RowClaims]:
    """Fold one eval in: returns the eval's payload row and *claims* extended by what its rows
    claim.  Each verdict entry is its own metric row, so a tripwire's items repeat the
    metric-level fields.
    """
    eval_id = eval_dir.name
    series = _read_run_series(eval_dir, issues)
    run_stamps = [run.stamp for run in series.runs]
    latest_run_stamp = run_stamps[-1] if run_stamps else None
    limits, limits_by_metric = _read_limits(
        eval_dir, latest_run_stamp, issues, has_runs=series.runs_on_disk > 0,
    )
    age_seconds = _run_age_seconds(eval_dir, latest_run_stamp, now, issues)

    metrics: list[dict[str, Any]] = []
    for metric_id in sorted({metric_id for run in series.runs for metric_id in run.metrics}):
        _name_unrenderable_kinds(eval_id, metric_id, series.runs, issues)
        entries = sources.verdicts.get((eval_id, metric_id), {})
        if entries:
            claims = claims.with_verdict_key((eval_id, metric_id))
        rule_kind = limits_by_metric.get(metric_id, {}).get('rule_kind')
        # A metric no entry judges still gets its one (unjudged) row.
        for entry in [entries[item] for item in sorted(entries, key=_item_order)] or [None]:
            matched, claims = _join_escalation((entry or {}).get('fingerprint'), sources, claims)
            metrics.append({
                **_metric_fields(metric_id, series.runs, rule_kind),
                **_verdict_row_fields(
                    entry, metric_id, eval_dir, matched, sources.storm is not None, issues,
                ),
            })

    return {
        'eval_id': eval_id,
        'latest_run_stamp': latest_run_stamp,
        'latest_run_age_seconds': age_seconds,
        'stale': age_seconds is not None and age_seconds > _STALE_AFTER_SECONDS,
        'limits': limits,
        # The same run-scoped block the payload carries at its TOP level (which
        # is the one the program-wide banner reads).  Repeated per row purely so
        # a row showing ``storm_collapsed`` parity carries, in place, the reason
        # its escalation link is absent.
        'storm_escape': sources.storm,
        'run_stamps': run_stamps,
        'run_count': len(run_stamps),
        'runs_on_disk': series.runs_on_disk,
        'truncated': series.truncated,
        'corpus': series.runs[-1].corpus if series.runs else None,
        'metrics': metrics,
    }, claims


def build_memory_evals(
    memory_evals_dir: Path,
    corpus: Datum[EscalationCorpus],
    escalation_queue: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate the memory-eval artifact tree into one dashboard payload.

    Args:
        memory_evals_dir: The memory-eval artifact root
            (``<project_root>/fused-memory/data/memory-evals``); enumerated
            generically — one entry per subdirectory, no per-eval code (DD5).
        corpus: The escalation corpus walk; its *escalation_queue* root
            records are joined by fingerprint.
        escalation_queue: The id of the corpus queue memory-eval
            regressions are filed onto.
        now: Request-scoped reference timestamp, resolved ONCE via
            :func:`~dashboard.data.utils.resolve_now` and threaded through
            every derived time field.  Injectable so staleness is testable
            without freezing the clock.

    Returns:
        The ``MEMORY_EVALS`` payload body.  Never raises.

    The never-raises contract is delivered by ONE named boundary here rather
    than by broad per-artifact catches.  Each artifact reader below catches
    only ``_ARTIFACT_ERRORS`` — what reading and parsing a file can raise — and
    degrades one row.  Anything else is a bug in this module, and is caught
    here, logged WITH its traceback, and reported as ``internal_error`` rather
    than disguised as a broken artifact.  So a bug still cannot 500 the
    dashboard poll, and it also cannot send an operator to inspect a file that
    was never at fault.
    """
    resolved_now = resolve_now(now)
    issues: list[dict[str, Any]] = []
    payload = _empty_payload(resolved_now, issues)

    try:
        _build_payload(
            payload, memory_evals_dir, corpus, escalation_queue, issues, resolved_now,
        )
    except Exception as exc:  # noqa: BLE001 - the never-500 boundary; named, not swallowed
        # Logged with the traceback: the payload names the bug for the
        # operator, the log carries what whoever fixes it actually needs.
        # Without this half, honouring the never-500 contract would mean a bug
        # that degrades the dashboard indefinitely with nothing to debug from.
        logger.exception('memory-evals payload build failed for %s', memory_evals_dir)
        _issue(
            issues, 'internal_error', path=memory_evals_dir,
            detail=(
                f'{type(exc).__name__}: {exc} — this is a dashboard bug, not a broken '
                'artifact; see the dashboard log for the traceback'
            ),
        )

    payload['issue_count'] = len(issues)
    return payload


def root_scan_succeeded(payload: dict[str, Any]) -> bool:
    """Did this scan actually reach the artifact tree?

    The cacheability predicate for the route's TTL cache
    (``cache_ok=root_scan_succeeded``).  False for the payloads produced
    WITHOUT a completed walk:

    * ``root_present`` is falsy — :meth:`Path.is_dir` said no and
      :func:`_build_payload` returned before enumerating.
    * an ``unreadable_root`` issue is present — the root exists but
      :meth:`Path.iterdir` raised, so again nothing was walked.
    * an ``internal_error`` issue with NO ``eval_id`` is present — the
      top-level guard in :func:`build_memory_evals` fired, so a bug aborted
      the build and the payload is missing rows for a reason that has nothing
      to do with the tree.

    The first two are O(1) to recompute — one ``stat``, or one
    immediately-raising ``iterdir`` — so re-deriving them on every poll costs
    nothing and there is no expensive walk behind them to stampede onto.
    Caching them, by contrast, pins a "there is no data here" view for the
    full TTL window past the moment the tree appears: a volume finishing its
    mount, a mode being fixed, the first eval run landing.

    A TOP-LEVEL ``internal_error`` is not free to recompute, and declining it
    is still right: unlike the ordinary-but-incomplete states
    :func:`~dashboard.data.escalation_corpus.acquire_corpus`'s
    ``reached_any`` cache gate deliberately keeps cacheable, a bug is never a
    steady state — it is an emergency with no known trigger — so the walk it
    costs is bounded by how long the bug lives, and pinning a payload that is
    blank for reasons the operator cannot see or fix is the worse trade.  How
    much of the tree that payload is missing is also unknown, which is what
    makes it unlike the case below.

    A PER-EVAL ``internal_error`` (``eval_id`` set to the eval dir basename,
    emitted inside :func:`_build_payload`'s per-eval loop) is a degraded ROW,
    and stays cacheable.  The walk COMPLETED; every other eval row is present
    and correct, and exactly one was lost.  This half is load-bearing: a
    deterministic bug in :func:`_build_eval` raises identically on the next
    pass, so declining to cache would defeat the TTL forever — re-running the
    full artifact walk on every ~3s poll, and firing ``logger.exception`` per
    rebuild, to rediscover a bug that re-scanning cannot fix.

    A degraded ROW is deliberately still cacheable.  An unreadable metrics
    run, an unknown metric kind or an orphan verdict all mean the scan reached
    the tree and did its work; re-running it would rediscover the same
    degradation at the cost of the full walk this cache exists to prevent.
    The predicate keys on named issue KINDS — plus, for ``internal_error``,
    the issue's SCOPE — never on ``issue_count``.

    The issue rides the payload either way, and ``issue_count`` still counts
    it, so narrowing the CACHE predicate costs the operator no visibility: a
    per-eval bug is named on every served response, cached or rebuilt.

    Reads every field through ``.get`` because it runs inside the cache write
    path: a partially-built payload must degrade to "don't cache" rather than
    raise where a raise would surface as a 500 on the dashboard poll.
    """
    if not payload.get('root_present'):
        return False
    return not any(_is_uncacheable(issue) for issue in payload.get('issues') or ())


def _is_uncacheable(issue: dict[str, Any]) -> bool:
    """Does this one issue mean the walk did not complete?

    The scope test is gated on ``internal_error`` SPECIFICALLY rather than
    applied to every member of ``_UNCACHEABLE_ISSUE_KINDS``.  A blanket "any
    scoped issue is cacheable" rule reads cleaner and is a no-op today, which
    is exactly what makes it a trap: ``unreadable_root`` means
    :meth:`Path.iterdir` raised and NOTHING was walked, which must stay
    uncacheable whatever fields the issue later comes to carry.

    ``is None``, not truthiness: :func:`_issue` declares
    ``eval_id: str | None = None`` and writes that default through unchanged,
    so ``None`` IS the "the emitter did not scope this" sentinel rather than a
    proxy for it.  Reads through ``.get`` for the same reason the caller does.
    """
    kind = issue.get('kind')
    if kind not in _UNCACHEABLE_ISSUE_KINDS:
        return False
    return kind != 'internal_error' or issue.get('eval_id') is None


def _empty_payload(resolved_now: datetime, issues: list[dict[str, Any]]) -> dict[str, Any]:
    """The payload skeleton, with every key present on every return path."""
    return {
        'generated_at': resolved_now.isoformat(),
        'root_present': True,
        # Run-scoped across the WHOLE program, so it is carried here rather than
        # only inside the eval rows: the banner has one source instead of the UI
        # arbitrarily electing an eval row to read it from, and it survives a
        # root that enumerates to zero eval dirs — the case where the aggregate
        # escalation exists and every per-eval copy would have vanished.  Set on
        # every return path so the key never has to be probed for.
        'storm_escape': None,
        'evals': [],
        'issues': issues,
        'issue_count': 0,
        'unmatched_escalations': [],
    }


def _resolve_storm(
    storm: dict[str, Any] | None,
    escalation_index: Mapping[str, Escalation],
) -> tuple[dict[str, Any] | None, _RowClaims]:
    """Project the run-scoped storm block, resolving its aggregate against the rows' escalation index.

    A resolved aggregate is claimed as linked: the storm block, not a row,
    explains it, so it is not then reported as unmatched.
    """
    if storm is None:
        return None, _RowClaims()
    aggregate_fingerprint = storm.get('aggregate_fingerprint')
    aggregate: Escalation | None = None
    linked: frozenset[str] = frozenset()
    if isinstance(aggregate_fingerprint, str) and aggregate_fingerprint:
        aggregate = escalation_index.get(aggregate_fingerprint)
        if aggregate is not None:
            linked = frozenset({aggregate_fingerprint})
    block = {
        'triggered': storm.get('triggered'),
        'alarm_count': storm.get('alarm_count'),
        'aggregate_fingerprint': aggregate_fingerprint,
        'escalation': _escalation_projection(aggregate) if aggregate is not None else None,
    }
    return block, _RowClaims(linked=linked)


def _build_payload(
    payload: dict[str, Any],
    memory_evals_dir: Path,
    corpus: Datum[EscalationCorpus],
    escalation_queue: str,
    issues: list[dict[str, Any]],
    resolved_now: datetime,
) -> None:
    """Fill *payload* in place from the artifact tree.

    Mutates rather than returns so that a bug escaping to
    ``build_memory_evals``' guard still yields whatever this got as far as
    setting — a partially-filled payload is more honest than a blank one.
    ``issue_count`` is deliberately NOT maintained here; the caller sets it
    once, on both the clean and the guarded path.
    """
    # "No eval has ever run" is a legitimate pre-deployment state, not a
    # degradation — an explicit empty-but-healthy payload with NO issue.
    # Flagging it would train operators to ignore the issues list.
    if not memory_evals_dir.is_dir():
        payload['root_present'] = False
        return

    try:
        eval_dirs = sorted(p for p in memory_evals_dir.iterdir() if p.is_dir())
    except OSError as exc:
        # ``is_dir()`` above already answered, but the walk itself can still
        # fail — a mode change between the two, or an NFS/permission hiccup.
        # Every other artifact boundary here is guarded; leaving this one open
        # turned a degraded tree into a 500 on the dashboard poll.  The root IS
        # present, so that is what the payload says: it is the enumeration that
        # failed, and the failure is named rather than mistaken for an empty root.
        #
        # ``OSError`` alone, not ``_ARTIFACT_ERRORS``: ``Path.iterdir`` is the
        # only thing that can raise here, and ``Path.is_dir()`` swallows
        # ``OSError`` and returns False rather than raising.  Nothing is being
        # parsed, so there is no ``ValueError`` path to catch.
        _issue(issues, 'unreadable_root', path=memory_evals_dir, detail=str(exc))
        return

    # ``_read_verdicts`` owns its own read boundary and degrades to ``({}, None)``
    # in place, so there is nothing for this caller to catch.
    verdict_index, storm = _read_verdicts(memory_evals_dir, eval_dirs, issues)

    escalation_index, unfingerprinted = _index_escalations(corpus, escalation_queue, issues)

    storm_block, claims = _resolve_storm(storm, escalation_index)
    payload['storm_escape'] = storm_block
    sources = _JoinSources(verdict_index, escalation_index, storm_block)

    # Per-eval bug boundary, not one comprehension.  Artifact errors are
    # already handled inside ``_build_eval``, so anything escaping it is a bug
    # — and as a single comprehension a bug on the fifth of ten dirs discarded
    # the other nine, leaving a present-but-empty root that reads as "no evals
    # have run".  Per-eval it costs that one row and names which one, and
    # ``claims`` advances only past a row that ships.
    evals: list[dict[str, Any]] = []
    for eval_dir in eval_dirs:
        try:
            row, claims = _build_eval(eval_dir, sources, claims, resolved_now, issues)
        except Exception as exc:  # noqa: BLE001 - per-eval half of the never-500 boundary
            logger.exception('memory-evals eval build failed for %s', eval_dir)
            _issue(
                issues, 'internal_error', eval_id=eval_dir.name, path=eval_dir,
                detail=(
                    f'{type(exc).__name__}: {exc} — this is a dashboard bug, not a broken '
                    'artifact; see the dashboard log for the traceback'
                ),
            )
        else:
            evals.append(row)
    payload['evals'] = evals

    # The reverse direction: an open eval_regression escalation that no metric
    # row links.  Surfaced rather than dropped, so parity can be checked both
    # ways — and each one says WHY it is unlinked, so a genuine parity orphan
    # is distinguishable from one a triggered storm merely suppressed.
    payload['unmatched_escalations'] = [
        _unmatched_projection(
            record,
            'storm_suppressed' if fingerprint in claims.suppressed else 'no_matching_verdict',
        )
        for fingerprint, record in escalation_index.items()
        if fingerprint not in claims.linked
    ] + [
        # Open, and unlinkable rather than merely unlinked: with no usable
        # fingerprint these never reached the index, so this is the only place
        # in the payload they can appear at all.
        _unmatched_projection(record, 'no_fingerprint')
        for record in unfingerprinted
    ]

    # A verdict resolving onto nothing is contract drift between the evaluator
    # and the artifact tree — surfaced, not dropped.
    for eval_id, metric_id in verdict_index:
        if (eval_id, metric_id) in claims.verdict_keys:
            continue
        _issue(
            issues,
            'orphan_verdict',
            eval_id=eval_id,
            path=memory_evals_dir / 'verdicts-current.json',
            detail=(
                f'verdict for metric {_short_repr(metric_id)} matches no metric row '
                f'in eval {_short_repr(eval_id)}'
            ),
        )
