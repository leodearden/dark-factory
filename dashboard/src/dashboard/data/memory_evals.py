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

**Same-host file reads (DD1).**  The dashboard and the eval runner share a
filesystem; there is no RPC in this path.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dashboard.data.escalations import load_queue_escalations
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

# The only escalation category that participates in the fingerprint join.  An
# escalation from any other producer may legitimately carry a colliding
# dedupe_fingerprint; joining it would attribute someone else's alert to a
# memory-eval metric.
_EVAL_REGRESSION_CATEGORY = 'eval_regression'

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

# An eval whose newest run is older than this is DISPLAYED as stale.  It is
# never alarmed on: the runner-failure tripwire belongs to the eval program,
# and this module files nothing.  36h gives a nightly cadence one full missed
# run plus slack before the badge appears.
_STALE_AFTER_SECONDS = 36 * 3600.0

# The producer's run-stamp format (zero-padded UTC, which is why filename order
# is chronological order).
_RUN_STAMP_FORMAT = '%Y%m%dT%H%M%SZ'

# Everything a malformed artifact can plausibly raise on the way through
# ``json.loads`` + dict access.  Caught narrowly at each artifact boundary so
# one bad file degrades one row, never the whole payload.
_ARTIFACT_ERRORS = (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError)

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
    ``missing_limits``, a parse failure is ``unreadable_limits`` (recorded by
    the caller's handler), and an artifact that parses but is not an object is
    ``malformed_limits``.  Without that third path a present-but-wrong-type
    artifact produced no signal at all, since the absent-file issue does not
    fire for a file that exists.
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

    body = _load_json(path)
    if not isinstance(body, dict):
        # Valid JSON of the wrong shape — `_load_json` does not raise, so the
        # caller's `unreadable_limits` handler never fires, and the file EXISTS
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

    by_metric: dict[str, Any] = {}
    for record in body.get('verdicts') or []:
        if isinstance(record, dict) and isinstance(record.get('metric_id'), str):
            by_metric[record['metric_id']] = record
    return block, by_metric


def _metric_rows(body: Any) -> list[dict]:
    """The metric records of one parsed run body, or ``[]`` if unusable."""
    if not isinstance(body, dict):
        return []
    metrics = body.get('metrics')
    if not isinstance(metrics, list):
        return []
    return [
        m for m in metrics
        if isinstance(m, dict) and isinstance(m.get('metric_id'), str) and m.get('metric_id')
    ]


def _read_verdicts(
    root: Path,
    eval_dirs: list[Path],
    issues: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any] | None]:
    """Index the ROOT verdicts artifact by ``(eval_id, metric_id)``.

    Root-scoped, not per-eval: ``docs/prds/memory-eval-program.md`` pins
    ``fused-memory/data/memory-evals/verdicts-<STAMP>.json``, and the
    ``storm_escape`` block is per-RUN across the whole program, which only
    makes sense above the eval dirs.

    This artifact is the SOLE verdict source.  Its ``verdict`` strings are
    passed through unmapped — a dashboard-side translation table would be a
    second vocabulary to keep in sync with the evaluator's, and the first
    divergence would silently mislabel an alarm.

    All three disposal paths are NAMED: absent is ``missing_verdicts``, a parse
    failure is ``unreadable_verdicts`` (recorded by the caller's handler), and
    an artifact that parses but is not an object is ``malformed_verdicts``.
    That third path matters most: a discarded verdicts body leaves every row's
    verdict absent, which is indistinguishable from a healthy no-alarm tree
    unless the discard is recorded.
    """
    path = root / 'verdicts-current.json'
    if not path.is_file():
        if eval_dirs:
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

    body = _load_json(path)
    if not isinstance(body, dict):
        # Valid JSON of the wrong shape: `_load_json` does not raise, so the
        # caller's `unreadable_verdicts` handler never fires.  Discarding it
        # silently here would make a broken verdicts file read exactly like a
        # healthy no-alarm tree.  `eval_id` is omitted because this artifact is
        # root-scoped, matching `missing_verdicts` above.
        _issue(
            issues, 'malformed_verdicts', path=path,
            detail=f'expected an object with an "entries" list, got {type(body).__name__}',
        )
        return {}, None

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in body.get('entries') or []:
        if not isinstance(entry, dict):
            continue
        eval_id = entry.get('eval_id')
        metric_id = entry.get('metric_id')
        if isinstance(eval_id, str) and isinstance(metric_id, str):
            index[(eval_id, metric_id)] = entry

    # Run-scoped, which is why it lives beside the entries rather than inside
    # them.  Surfaced only while TRIGGERED — an untriggered block is the
    # ordinary case and would otherwise render as a permanent storm banner.
    storm = body.get('storm_escape')
    if not isinstance(storm, dict) or not storm.get('triggered'):
        storm = None
    return index, storm


def _escalation_projection(record: dict[str, Any]) -> dict[str, Any]:
    """The fields a metric row (or ``unmatched_escalations``) carries.

    ``created_at`` is sourced from the queue record's ``timestamp`` — that is
    the field ``escalation.models.Escalation`` serialises.  The fingerprint is
    echoed so an operator can see WHY a row linked, without the UI having to
    cross-reference two fields.
    """
    return {
        'id': record.get('id'),
        'summary': record.get('summary'),
        'severity': record.get('severity'),
        'level': record.get('level'),
        'created_at': record.get('timestamp'),
        'dedupe_fingerprint': record.get('dedupe_fingerprint'),
    }


def _index_escalations(escalations_dir: Path) -> dict[str, dict[str, Any]]:
    """Index open ``eval_regression`` escalations by ``dedupe_fingerprint``.

    Reuses :func:`dashboard.data.escalations.load_queue_escalations` (INV-5 —
    a call site, never a second reader).  Its existing contracts carry this
    join: a missing dir returns ``[]`` so the no-queue case needs no guard,
    unparseable files are skipped and logged so a corrupt escalation cannot
    crash the join, and the archive subtree is NOT walked — so only PENDING
    escalations are joined, which is exactly what the "still-open" parity
    states mean.

    Matching is ``==`` over the WHOLE fingerprint string.  The fingerprint is
    the producer's private construction; nothing here parses its substructure,
    so the producer can change how it is built without breaking the dashboard.
    """
    index: dict[str, dict[str, Any]] = {}
    for record in load_queue_escalations(escalations_dir):
        if not isinstance(record, dict):
            continue
        if record.get('category') != _EVAL_REGRESSION_CATEGORY:
            continue
        fingerprint = record.get('dedupe_fingerprint')
        if isinstance(fingerprint, str) and fingerprint:
            index[fingerprint] = record
    return index


def _parity(verdict: Any, escalation: dict[str, Any] | None) -> str:
    """The derived display state for one metric row.

    A pure ``(verdict, linked?)`` lookup — not a statistic.  Deriving it once
    here keeps the UI from re-deriving badge state out of three separate
    fields, which is where the two sides would drift apart.
    """
    if verdict == 'alarm':
        return 'alarmed_open' if escalation else 'alarmed_unlinked'
    return 'recovered_open' if escalation else 'clear'


def _build_eval(
    eval_dir: Path,
    issues: list[dict[str, Any]],
    verdict_index: dict[tuple[str, str], dict[str, Any]],
    consumed: set[tuple[str, str]],
    escalation_index: dict[str, dict[str, Any]],
    linked_fingerprints: set[str],
    storm: dict[str, Any] | None,
    now: datetime,
) -> dict[str, Any]:
    """Assemble one eval's trend payload from its ``metrics-*.json`` series.

    Runs are ordered by FILENAME — the producer's own contract
    (``shared.memory_eval_metrics.load_series_window`` sorts the same way,
    because the stamp is a zero-padded UTC string) — so the dashboard never
    invents a second ordering rule.  The x-axis label for each run is the
    artifact's in-body ``run_stamp``.
    """
    eval_id = eval_dir.name
    all_paths = sorted(eval_dir.glob('metrics-*.json'))

    # Trailing window: keep the MOST RECENT runs, drop the oldest.  Dropping
    # the newest instead would leave a trend that ends N runs ago reading as
    # current.  ``_TREND_RUN_CAP`` is looked up in the module namespace here
    # (not captured at import) so it stays one adjustable knob.
    #
    # No silent caps: whenever this window drops anything, the count dropped
    # is disclosed IN THE PAYLOAD (``truncated`` + ``runs_on_disk`` beside
    # ``run_count``), never only in a log the operator will not read.
    runs_on_disk = len(all_paths)
    paths = all_paths[-_TREND_RUN_CAP:] if _TREND_RUN_CAP else all_paths
    truncated = len(paths) < runs_on_disk

    # (in-body stamp, metric_id -> record, source path).  The path is carried
    # so a per-run issue can name the FILE that caused it: the committed
    # ``malformed/`` exemplar's two runs share one in-body run_stamp, so the
    # stamp alone would not identify which artifact is at fault.
    runs: list[tuple[Any, dict[str, dict], Path]] = []
    corpus: dict | None = None
    for path in paths:
        try:
            body = _load_json(path)
        except _ARTIFACT_ERRORS as exc:
            # One unreadable run degrades one run, not the whole eval.
            _issue(issues, 'unreadable_metrics', eval_id=eval_id, path=path, detail=str(exc))
            continue

        # The reader validates only what it needs to READ and RENDER.  The M1
        # schema's cross-field rules (a proportion inside [0,1], a required
        # denominator) are the producer's to reject at emit time — restating
        # them here would be a second implementation of what the producer has
        # already decided.
        if not isinstance(body, dict) or not isinstance(body.get('metrics'), list):
            _issue(
                issues, 'malformed_metrics', eval_id=eval_id, path=path,
                detail=f'expected an object with a list "metrics" field, got {type(body).__name__}',
            )
            continue

        stamp = body.get('run_stamp')
        if stamp is None:
            _issue(
                issues, 'missing_run_stamp', eval_id=eval_id, path=path,
                detail='run artifact carries no in-body run_stamp',
            )
        if isinstance(body.get('corpus'), dict):
            corpus = dict(body['corpus'])
        runs.append((stamp, {row['metric_id']: row for row in _metric_rows(body)}, path))

    run_stamps = [stamp for stamp, _, _ in runs]

    # Metric identity is the metric_id, unioned across the whole window: a
    # metric that appears only in some runs still gets a full-width series
    # (with holes), so every series stays index-aligned to the shared axis.
    metric_ids: list[str] = []
    for _, by_id, _path in runs:
        for metric_id in by_id:
            if metric_id not in metric_ids:
                metric_ids.append(metric_id)

    latest_run_stamp = run_stamps[-1] if run_stamps else None
    try:
        limits, limits_by_metric = _read_limits(
            eval_dir, latest_run_stamp, issues, has_runs=bool(all_paths),
        )
    except _ARTIFACT_ERRORS as exc:
        _issue(
            issues, 'unreadable_limits', eval_id=eval_id,
            path=eval_dir / 'limits-current.json', detail=str(exc),
        )
        limits, limits_by_metric = None, {}

    # Staleness is DISPLAYED, never alarmed on.  An unparseable stamp degrades
    # the age to None (and says so) rather than raising or guessing a date.
    age_seconds: float | None = None
    if isinstance(latest_run_stamp, str):
        try:
            run_at = datetime.strptime(latest_run_stamp, _RUN_STAMP_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            _issue(
                issues, 'unparseable_run_stamp', eval_id=eval_id, path=eval_dir,
                detail=f'run_stamp {latest_run_stamp!r} is not in {_RUN_STAMP_FORMAT}',
            )
        else:
            age_seconds = (now - run_at).total_seconds()

    latest = runs[-1][1] if runs else {}
    metrics: list[dict[str, Any]] = []
    for metric_id in sorted(metric_ids):
        # A metric missing from a run contributes a None hole at that index
        # rather than being dropped — dropping would silently shift this
        # metric's points against its neighbours'.
        values = [by_id.get(metric_id, {}).get('value') for _, by_id, _p in runs]
        current = latest.get(metric_id, {})

        # A kind outside the closed vocabulary is a RENDERING failure: there is
        # no chart primitive for it, so the dashboard genuinely cannot resolve
        # it and says so.  The value itself is still real and is still shown.
        #
        # Checked across the WHOLE window, not just the newest run: a trend
        # line is drawn from every point, so an unrenderable kind anywhere in
        # the series is a rendering failure even when the latest run is clean.
        # The committed ``malformed/`` exemplar is exactly that shape — its
        # bad-kind artifact sorts BEFORE its well-kinded one, so a latest-run
        # check would have declared that fixture healthy.
        #
        # One issue per distinct (metric, kind): a kind that is wrong in every
        # run of a 90-run window is one problem, not ninety.  Deduped on the
        # repr so an unhashable value out of a malformed artifact cannot raise.
        kind = current.get('kind')
        seen_kinds: set[str] = set()
        for _stamp, by_id, run_path in runs:
            run_kind = by_id.get(metric_id, {}).get('kind')
            # ``isinstance`` before the set lookup: an unhashable kind out of a
            # malformed artifact would make ``in _KNOWN_KINDS`` raise, and a
            # non-string kind is unrenderable anyway.
            if run_kind is None or (isinstance(run_kind, str) and run_kind in _KNOWN_KINDS):
                continue
            if repr(run_kind) in seen_kinds:
                continue
            seen_kinds.add(repr(run_kind))
            _issue(
                issues, 'unknown_kind', eval_id=eval_id, path=run_path,
                detail=f'metric {metric_id!r} has kind {run_kind!r}, which has no chart primitive',
            )

        # Absent verdict == absent, never defaulted to 'no_alarm'.  The empty
        # dict makes every verdict field read as None below without a second
        # code path.
        entry = verdict_index.get((eval_id, metric_id))
        if entry is not None:
            consumed.add((eval_id, metric_id))
        judged: dict[str, Any] = entry or {}

        # The join, in full: whole-string equality on the fingerprint.  It is
        # NOT gated on the verdict — a metric that has recovered while its
        # escalation is still open is a real transient state (the watcher has
        # not closed it yet), and hiding the link there would make the
        # escalation look orphaned in exactly the window an operator is most
        # likely to be looking at it.
        #
        # While a storm is triggered the runner files ONE aggregate escalation
        # instead of N per-metric ones, so per-metric links are suppressed
        # here.  Showing them would invent links that were never filed.
        fingerprint = judged.get('fingerprint')
        matched: dict[str, Any] | None = None
        if storm is None and isinstance(fingerprint, str) and fingerprint:
            matched = escalation_index.get(fingerprint)
            if matched is not None:
                linked_fingerprints.add(fingerprint)
        escalation = _escalation_projection(matched) if matched is not None else None

        metrics.append({
            'metric_id': metric_id,
            'kind': kind,
            # Looked up by metric_id, never zipped positionally: the two lists
            # are ordered independently and a metric may be absent from either.
            'rule_kind': limits_by_metric.get(metric_id, {}).get('rule_kind'),
            'current_value': current.get('value'),
            'n': current.get('n'),
            'denominator': current.get('denominator'),
            'direction': current.get('direction'),
            'trend': {'labels': list(run_stamps), 'values': values},
            # Verdict fields, passed through UNMAPPED from the sole verdict
            # source.  `value` is what the evaluator judged; `current_value`
            # above is what the metrics artifact says — two sources, two
            # fields, never conflated.
            'verdict': judged.get('verdict'),
            'verdict_detail': judged.get('detail'),
            'value': judged.get('value'),
            'item': judged.get('item'),
            'limit_ref': judged.get('limit_ref'),
            'fingerprint': judged.get('fingerprint'),
            'run_stamp': judged.get('run_stamp'),
            'escalation': escalation,
            'parity': (
                'storm_collapsed'
                if storm is not None and judged.get('verdict') == 'alarm'
                else _parity(judged.get('verdict'), escalation)
            ),
        })

    return {
        'eval_id': eval_id,
        'latest_run_stamp': latest_run_stamp,
        'latest_run_age_seconds': age_seconds,
        'stale': age_seconds is not None and age_seconds > _STALE_AFTER_SECONDS,
        'limits': limits,
        'storm_escape': storm,
        'run_stamps': run_stamps,
        'run_count': len(run_stamps),
        'runs_on_disk': runs_on_disk,
        'truncated': truncated,
        'corpus': corpus,
        'metrics': metrics,
    }


def build_memory_evals(
    memory_evals_dir: Path,
    escalations_dir: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate the memory-eval artifact tree into one dashboard payload.

    Args:
        memory_evals_dir: The memory-eval artifact root
            (``<project_root>/fused-memory/data/memory-evals``); enumerated
            generically — one entry per subdirectory, no per-eval code (DD5).
        escalations_dir: The recon escalation queue dir, joined by fingerprint.
        now: Request-scoped reference timestamp, resolved ONCE via
            :func:`~dashboard.data.utils.resolve_now` and threaded through
            every derived time field.  Injectable so staleness is testable
            without freezing the clock.

    Returns:
        The ``MEMORY_EVALS`` payload body.  Never raises.
    """
    resolved_now = resolve_now(now)

    issues: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        'generated_at': resolved_now.isoformat(),
        'root_present': True,
        'evals': [],
        'issues': issues,
        'issue_count': 0,
        'unmatched_escalations': [],
    }

    # "No eval has ever run" is a legitimate pre-deployment state, not a
    # degradation — an explicit empty-but-healthy payload with NO issue.
    # Flagging it would train operators to ignore the issues list.
    if not memory_evals_dir.is_dir():
        payload['root_present'] = False
        return payload

    eval_dirs = sorted(p for p in memory_evals_dir.iterdir() if p.is_dir())

    try:
        verdict_index, storm = _read_verdicts(memory_evals_dir, eval_dirs, issues)
    except _ARTIFACT_ERRORS as exc:
        # An unreadable verdicts artifact must never read as "nothing
        # alarmed" — every verdict stays absent and the failure is named.
        _issue(
            issues, 'unreadable_verdicts',
            path=memory_evals_dir / 'verdicts-current.json', detail=str(exc),
        )
        verdict_index, storm = {}, None

    escalation_index = _index_escalations(escalations_dir)

    consumed: set[tuple[str, str]] = set()
    linked_fingerprints: set[str] = set()

    # The storm block is run-scoped across the whole program; resolve its
    # aggregate against the SAME escalation index the metric rows use, and
    # mark that fingerprint consumed so the aggregate is not then reported as
    # unexplained.  It IS explained — by the storm block, not by a row.
    storm_block: dict[str, Any] | None = None
    if storm is not None:
        aggregate_fingerprint = storm.get('aggregate_fingerprint')
        aggregate: dict[str, Any] | None = None
        if isinstance(aggregate_fingerprint, str) and aggregate_fingerprint:
            aggregate = escalation_index.get(aggregate_fingerprint)
            if aggregate is not None:
                linked_fingerprints.add(aggregate_fingerprint)
        storm_block = {
            'triggered': storm.get('triggered'),
            'alarm_count': storm.get('alarm_count'),
            'aggregate_fingerprint': aggregate_fingerprint,
            'escalation': _escalation_projection(aggregate) if aggregate is not None else None,
        }

    payload['evals'] = [
        _build_eval(
            d, issues, verdict_index, consumed, escalation_index,
            linked_fingerprints, storm_block, resolved_now,
        )
        for d in eval_dirs
    ]

    # The reverse direction: an open eval_regression escalation that no metric
    # row explains.  Surfaced rather than dropped, so parity can be checked
    # both ways.
    payload['unmatched_escalations'] = [
        _escalation_projection(record)
        for fingerprint, record in escalation_index.items()
        if fingerprint not in linked_fingerprints
    ]

    # A verdict resolving onto nothing is contract drift between the evaluator
    # and the artifact tree — surfaced, not dropped.
    for eval_id, metric_id in verdict_index:
        if (eval_id, metric_id) in consumed:
            continue
        _issue(
            issues,
            'orphan_verdict',
            eval_id=eval_id,
            path=memory_evals_dir / 'verdicts-current.json',
            detail=f'verdict for metric {metric_id!r} matches no metric row in eval {eval_id!r}',
        )

    payload['issue_count'] = len(issues)
    return payload
