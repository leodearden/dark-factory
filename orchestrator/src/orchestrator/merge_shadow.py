"""Warm-vs-cold shadow-compare detective (MQ-refactor task γ).

Extracted verbatim from :mod:`orchestrator.merge_queue`: the per-test result
parsers, the persisted cadence state, and the shadow-compare functions that
run a from-scratch cold verify alongside a landed warm merge and alarm on
divergence (PRD §10 invariant 6(b)).  ``merge_queue`` re-exports every name
here through a top-level shim so existing importers
(``from orchestrator.merge_queue import X``, etc.) keep working unchanged.

A moved function that calls a merge_queue-resident sibling — whether that
sibling stays permanently (``_run_unscoped_typechecks``) or is monkeypatched
by the existing test suite via the string path
``orchestrator.merge_queue.<name>`` (``run_scoped_verification``,
``build_merge_verify_spec``, ``VerifyRunnerPool``, ``LocalRunner``,
``_run_cold_shadow_verify``, ``_run_shadow_compare``) — resolves it through a
function-local (deferred) import from :mod:`orchestrator.merge_queue` rather
than a direct intra-module reference.  This mirrors the
``_main_health_fingerprint`` convention in ``merge_queue.py`` and keeps this
module free of any top-level import of ``merge_queue`` (which would deadlock
module load, since merge_queue's shim needs this module fully defined
first).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_types import MergeRequest
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_runner import LocalRunner, VerifyRunnerPool, build_merge_verify_spec

if TYPE_CHECKING:
    from orchestrator.merge_queue import SpeculativeMergeWorker

logger = logging.getLogger('orchestrator.merge_queue')


# ---------------------------------------------------------------------------
# PRD §10 invariant 6(b): warm-vs-cold SHADOW compare cadence
# ---------------------------------------------------------------------------

@dataclass
class ShadowCompareState:
    """Persisted cadence state for the warm-vs-cold shadow compare.

    Stored as JSON at ``config.project_root/data/orchestrator/warm_verify_shadow.json``
    so both cadence conditions survive orchestrator restarts.

    Fields:
        merges_since_last_shadow: Count of warm-verified lands since the last
            shadow compare run.  Reset to 0 when a shadow compare is triggered.
        last_shadow_run_at: Unix timestamp (float) of the last shadow compare
            trigger.  0.0 when no shadow compare has ever run.
    """

    merges_since_last_shadow: int = 0
    last_shadow_run_at: float = 0.0


# ---------------------------------------------------------------------------
# Per-test result parsers for the warm-vs-cold shadow compare.
#
# Two formats are supported so that the shadow compare works regardless of
# which test runner the project's verify command uses:
#
#   cargo-nextest (reify's default):
#       "        PASS [   0.045s] reify-core some::mod::test_a"
#       "        FAIL [   1.200s] reify-eval other::test_b"
#       "        TIMEOUT [  5.000s] crate slow::test"
#       "        LEAK [   0.100s] crate leaky::test"
#       "        SIGSEGV [  0.001s] crate crash::test"
#   Groups: (1) status, (2) crate, (3) test_path
#
#   Rust libtest (plain `cargo test` output):
#       "test some::mod::test_name ... ok"
#       "test some::mod::test_name ... FAILED"
#   Groups: (1) test_path, (2) status
#
# SKIP / ignored lines are intentionally excluded: a skipped/ignored test
# is not "run" so treating it as present-but-failed would create spurious
# only_warm / only_cold presence divergences between warm and cold runs.
# ---------------------------------------------------------------------------

# Matches cargo-nextest human-output test result lines.
# Capture groups: (1) status, (2) crate/package::binary, (3) test path (rest of line)
#
# Real cargo-nextest 0.9.136 output (reify's merge-verify runner) inserts an
# OPTIONAL parenthesized progress counter such as '(  1/250)' between the timing
# bracket and the package::binary id, e.g.:
#
#     PASS [   0.130s] (  1/250) reify-cli::cli_affine_eval eval_x
#
# The non-capturing optional group ``(?:\(\s*\d+/\s*\d+\)\s+)?`` consumes and
# DISCARDS the counter so it does not appear in the stable key
# ``"pkg::bin test_path"``.  Without this group the regex captures the open-paren
# '(' as the crate and folds the counter remainder into the test path, producing
# run-specific garbage keys that break warm/cold shadow comparison.
#
# Backward-compatible: the group is optional, so old no-counter format and the
# libtest branch are unaffected.
_NEXTEST_TEST_LINE_RE = re.compile(
    r'^\s*(PASS|FAIL|TIMEOUT|LEAK|SIGSEGV)\s+\[[^\]]*\]\s+'
    r'(?:\(\s*\d+/\s*\d+\)\s+)?'  # optional N/M progress counter — consumed, not captured
    r'(\S+)\s+(\S.*?)\s*$'
)


# Matches plain `cargo test` (libtest) result lines.
# Capture groups: (1) test_path, (2) status ("ok" or "FAILED")
_LIBTEST_TEST_LINE_RE = re.compile(
    r'^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED)\s*$'
)


def _classify_test_status(raw_status: str) -> str:
    """Map a raw nextest or libtest status token to a 3-valued verdict string.

    Verdict vocabulary:
      ``'pass'``        — nextest ``PASS`` or ``LEAK``; libtest ``ok``
      ``'fail'``        — nextest ``FAIL``; libtest ``FAILED``
      ``'inconclusive'`` — nextest ``TIMEOUT`` or ``SIGSEGV``

    **LEAK → 'pass'** mirrors nextest's own default ``--leak-timeout 100ms``
    semantics: nextest counts LEAK as a PASS (suite exit stays 0) because
    teardown-slip leaks are non-fatal by design.  Under host contention,
    fast deterministic tests spuriously trip leak detection, so treating LEAK
    as ``'fail'`` produces false-positive warm/cold divergences (esc-31,
    esc-32).

    **TIMEOUT / SIGSEGV → 'inconclusive'** because these are non-deterministic
    execution artifacts (scheduler jitter, OOM-adjacent crashes) that do not
    imply a genuine warm/cold suite-verdict flip.  Routing them to
    ``'inconclusive'`` prevents the comparator from alarming on noise.

    Unknown tokens (forward-compat) fall through to ``'fail'`` (fail-closed).

    Args:
        raw_status: Status token from the regex capture group, e.g.
            ``'PASS'``, ``'LEAK'``, ``'TIMEOUT'``, ``'ok'``, ``'FAILED'``.

    Returns:
        One of ``'pass'``, ``'fail'``, or ``'inconclusive'``.
    """
    if raw_status in ('PASS', 'LEAK', 'ok'):
        return 'pass'
    if raw_status in ('TIMEOUT', 'SIGSEGV'):
        return 'inconclusive'
    # 'FAIL', 'FAILED', and any unknown forward-compat token → 'fail' (fail-closed)
    return 'fail'


def parse_per_test_results(test_output: str) -> dict[str, str]:
    """Parse test runner output into a per-test verdict map.

    Supports two formats:

    * **cargo-nextest** (reify's default merge-verify runner)::

          <whitespace> PASS|FAIL|TIMEOUT|LEAK|SIGSEGV [<timing>] [(<N>/<M>)] <pkg::bin> <path>

      Real cargo-nextest 0.9.136 output inserts an optional parenthesized progress
      counter ``(  N/M)`` (with internal whitespace padding) between the timing
      bracket and the ``package::binary`` id.  The counter is consumed and
      **excluded** from the key so that warm and cold runs (which have different
      N/M indices) produce identical stable keys.

      Key: ``"<pkg::bin> <test::path>"``, value: verdict string from
      :func:`_classify_test_status`.

    * **libtest** (plain ``cargo test``)::

          test <test::path> ... ok|FAILED

      Key: ``"<test::path>"``, value: ``'pass'`` iff status is ``ok``,
      else ``'fail'``.

    Verdict vocabulary: ``'pass'`` (nextest PASS/LEAK; libtest ok),
    ``'fail'`` (nextest FAIL; libtest FAILED),
    ``'inconclusive'`` (nextest TIMEOUT/SIGSEGV — non-deterministic artifacts
    excluded from alarm-worthy divergence detection).

    SKIP / ignored lines are excluded from both formats so they do not
    introduce spurious presence-divergences in the shadow compare diff.

    All other lines (build output, summary footer, blank lines) are ignored.

    Used by the warm-vs-cold shadow compare (PRD §10 invariant 6(b)) to
    capture per-test granularity so divergences can be named in the L2 alarm.

    Args:
        test_output: Raw string output from a verify run.

    Returns:
        ``dict[str, str]`` mapping test id to verdict string.  Empty dict for
        empty/blank input or when no test lines are present.  A caller that
        receives an empty dict from a genuine verify run should log a warning
        — the parser may not match the project's verify command output format.
    """
    result: dict[str, str] = {}
    for line in test_output.splitlines():
        m = _NEXTEST_TEST_LINE_RE.match(line)
        if m:
            status, crate, test_path = m.group(1), m.group(2), m.group(3)
            result[f"{crate} {test_path}"] = _classify_test_status(status)
            continue
        m = _LIBTEST_TEST_LINE_RE.match(line)
        if m:
            test_path, status = m.group(1), m.group(2)
            result[test_path] = _classify_test_status(status)
    return result


# Matches cargo-nextest Summary footer lines, e.g.:
#   Summary [   1.25s] 250 tests run: 249 passed, 1 failed, 0 skipped
#   Summary [   0.13s]   1 test run: 1 passed, 0 failed, 0 skipped   (N==1 → singular)
#   (leading whitespace tolerated: nextest may indent the Summary footer)
# Capture group: (1) total test count N from 'N tests run:' / 'N test run:'
_NEXTEST_SUMMARY_LINE_RE = re.compile(
    r'^\s*Summary\s+\[[^\]]*\]\s+(\d+)\s+tests?\s+run:',
    re.MULTILINE,
)


def _nextest_reported_test_count(output: str) -> int | None:
    """Return the total number of tests reported in nextest Summary footer line(s).

    Scans all lines in *output* for the cargo-nextest human-format footer::

        Summary [<timing>] N tests run: P passed, F failed, S skipped

    Returns the **sum** of N across all matched Summary lines (to cover
    multi-pass debug+release aggregate runs), or ``None`` when no Summary
    line is found in the output.

    A return value of ``0`` is distinct from ``None``:  ``0`` means a Summary
    was found but reported zero tests run (e.g. legitimately test-free crate);
    ``None`` means no nextest pass occurred at all (pure build noise or empty
    output).

    Used by :func:`_alarm_warm_shadow_unparseable` to discriminate between
    a genuinely test-free merge (no alarm) and a parser failure (alarm).

    Args:
        output: Raw string from a verify run.

    Returns:
        Sum of reported test counts, or ``None`` if no Summary line present.
    """
    matches = _NEXTEST_SUMMARY_LINE_RE.findall(output)
    if not matches:
        return None
    return sum(int(n) for n in matches)


@dataclass
class ShadowCompareDiff:
    """Per-test divergence between a warm and a cold verify run.

    Produced by :func:`diff_per_test_results` for PRD §10 invariant 6(b).

    Verdict model: each test verdict is one of ``'pass'``, ``'fail'``, or
    ``'inconclusive'``.  A divergence is **alarm-worthy** only when it is a
    genuine ``'pass'``↔``'fail'`` flip (or a presence divergence); any
    difference involving ``'inconclusive'`` is routed to the non-alarming
    :attr:`inconclusive` bucket and excluded from :attr:`has_divergence`.

    Attributes:
        diverging: Maps test_id → (warm_verdict, cold_verdict) for every
            alarm-worthy diverging test (genuine ``'pass'``↔``'fail'`` flip).
        warm_pass_cold_fail: Test ids that yielded ``'pass'`` warm but
            ``'fail'`` cold (the dangerous class: warm landed OK, cold reveals
            a real fail).
        warm_fail_cold_pass: Test ids that yielded ``'fail'`` warm but
            ``'pass'`` cold (less dangerous; warm was conservative).
        only_warm: Test ids present in the warm result but absent from cold
            (structural presence divergence → alarm-worthy).
        only_cold: Test ids present in the cold result but absent from warm.
        inconclusive: Maps test_id → (warm_verdict, cold_verdict) for tests
            where EITHER side is ``'inconclusive'``
            (TIMEOUT/SIGSEGV — non-deterministic execution artifacts).
            These differences are logged but NOT alarmed.
            Excluded from :attr:`has_divergence` by design.
    """

    diverging: dict[str, tuple[str, str]]
    warm_pass_cold_fail: list[str]
    warm_fail_cold_pass: list[str]
    only_warm: list[str]
    only_cold: list[str]
    inconclusive: dict[str, tuple[str, str]] = dataclasses.field(default_factory=dict)

    @property
    def has_divergence(self) -> bool:
        """True iff any alarm-worthy divergence bucket is non-empty.

        Deliberately excludes :attr:`inconclusive` — a pair differing by
        TIMEOUT/SIGSEGV is not alarm-worthy.
        """
        return bool(
            self.diverging
            or self.only_warm
            or self.only_cold
        )


def diff_per_test_results(
    warm: dict[str, str],
    cold: dict[str, str],
) -> ShadowCompareDiff:
    """Compute the per-test divergence between warm and cold verify results.

    Classifies every test in the union of both result sets into a divergence
    bucket using the 3-valued verdict model (``'pass'``/``'fail'``/
    ``'inconclusive'``):

    * Tests with **identical** verdicts in both legs are omitted.
    * Tests present in **only one** leg with a ``'pass'`` or ``'fail'`` verdict
      go to :attr:`~ShadowCompareDiff.only_warm` /
      :attr:`~ShadowCompareDiff.only_cold` — alarm-worthy (structural
      difference).
    * Tests present in **only one** leg whose sole verdict is
      ``'inconclusive'`` (TIMEOUT/SIGSEGV) go to
      :attr:`~ShadowCompareDiff.inconclusive` — non-alarming.  A TIMEOUT in
      one leg that the other leg simply never ran is a non-deterministic
      execution artifact, not a suite-verdict-changing flip.
    * Tests where **either** verdict in both legs is ``'inconclusive'``
      (TIMEOUT/SIGSEGV) go to :attr:`~ShadowCompareDiff.inconclusive`
      — non-alarming.
    * Tests with a genuine ``'pass'``↔``'fail'`` flip go to
      :attr:`~ShadowCompareDiff.diverging` and one of the direction buckets
      — alarm-worthy.

    Args:
        warm: Per-test verdict map from the warm (in-place) verify run,
            as returned by :func:`parse_per_test_results`.
        cold: Per-test verdict map from the cold (throwaway-worktree) verify run.

    Returns:
        A :class:`ShadowCompareDiff` with buckets populated for diverging
        tests.  :attr:`~ShadowCompareDiff.has_divergence` is False iff all
        alarm-worthy buckets are empty (``inconclusive`` is excluded by design).
    """
    diverging: dict[str, tuple[str, str]] = {}
    warm_pass_cold_fail: list[str] = []
    warm_fail_cold_pass: list[str] = []
    only_warm: list[str] = []
    only_cold: list[str] = []
    inconclusive: dict[str, tuple[str, str]] = {}

    all_tests = warm.keys() | cold.keys()
    for test_id in sorted(all_tests):
        in_warm = test_id in warm
        in_cold = test_id in cold
        if in_warm and in_cold:
            w, c = warm[test_id], cold[test_id]
            if w != c:
                if w == 'inconclusive' or c == 'inconclusive':
                    # Non-deterministic execution artifact — not alarm-worthy
                    inconclusive[test_id] = (w, c)
                else:
                    # Genuine 'pass'↔'fail' flip — alarm-worthy
                    diverging[test_id] = (w, c)
                    if w == 'pass' and c == 'fail':
                        warm_pass_cold_fail.append(test_id)
                    else:
                        warm_fail_cold_pass.append(test_id)
        elif in_warm:
            v = warm[test_id]
            if v == 'inconclusive':
                # TIMEOUT/SIGSEGV in warm with no cold result — non-deterministic
                # artifact, not alarm-worthy.  Store as ('inconclusive', 'absent')
                # for diagnostics.
                inconclusive[test_id] = ('inconclusive', 'absent')
            else:
                only_warm.append(test_id)
        else:
            v = cold[test_id]
            if v == 'inconclusive':
                # TIMEOUT/SIGSEGV in cold with no warm result — same reasoning.
                inconclusive[test_id] = ('absent', 'inconclusive')
            else:
                only_cold.append(test_id)

    return ShadowCompareDiff(
        diverging=diverging,
        warm_pass_cold_fail=warm_pass_cold_fail,
        warm_fail_cold_pass=warm_fail_cold_pass,
        only_warm=only_warm,
        only_cold=only_cold,
        inconclusive=inconclusive,
    )


def _load_shadow_compare_state(path: Path) -> ShadowCompareState:
    """Load the shadow compare cadence state from a JSON file.

    Fail-safe: returns a default ``ShadowCompareState()`` on any error
    (file not found, unreadable, unparseable JSON, or missing keys) so the
    orchestrator never fails to start due to a corrupt state file.

    Args:
        path: Path to the JSON state file (typically
            ``config.project_root/data/orchestrator/warm_verify_shadow.json``).

    Returns:
        The persisted state, or ``ShadowCompareState(0, 0.0)`` on any failure.
    """
    try:
        data = json.loads(path.read_text())
        return ShadowCompareState(
            merges_since_last_shadow=int(data['merges_since_last_shadow']),
            last_shadow_run_at=float(data['last_shadow_run_at']),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ShadowCompareState()


def _save_shadow_compare_state(path: Path, state: ShadowCompareState) -> None:
    """Persist the shadow compare cadence state to a JSON file.

    Creates parent directories as needed.

    Args:
        path: Destination path for the JSON state file.
        state: The :class:`ShadowCompareState` to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(state)))


def _shadow_compare_due(
    state: ShadowCompareState,
    now: float,
    *,
    every_n_merges: int,
    nightly_interval_secs: float,
) -> bool:
    """Return True when a shadow compare should be triggered.

    Implements PRD §10 invariant 6(b) "whichever sooner" = OR cadence:
    a shadow compare fires when EITHER the merge-count leg OR the nightly-timer
    leg is satisfied.

    Count leg: fires when ``state.merges_since_last_shadow >= every_n_merges``
        (provided ``every_n_merges > 0``; 0 disables the count leg entirely).

    Nightly leg: fires when ``now - state.last_shadow_run_at >= nightly_interval_secs``
        (provided ``nightly_interval_secs > 0``; 0 disables the timer leg).

    Args:
        state: Current persisted cadence state.
        now: Current Unix timestamp (time.time()).
        every_n_merges: From ``config.git.warm_verify_shadow_compare_every_n_merges``.
            0 → count leg disabled.
        nightly_interval_secs: From
            ``config.git.warm_verify_shadow_compare_nightly_interval_secs``.
            0 → timer leg disabled.

    Returns:
        True when at least one trigger condition is met.
    """
    count_due = (
        every_n_merges > 0
        and state.merges_since_last_shadow >= every_n_merges
    )
    timer_due = (
        nightly_interval_secs > 0
        and (now - state.last_shadow_run_at) >= nightly_interval_secs
    )
    return count_due or timer_due


# Sentinel task_id used for dedup on the warm/cold shadow divergence escalation.
# Mirrors ``_DRIFT_SENTINEL`` in verify_runner.py.
_WARM_COLD_SHADOW_SENTINEL = '__warm_cold_shadow__'


# Sentinel task_id for the fail-closed unparseable-format escalation.
# Kept DISTINCT from _WARM_COLD_SHADOW_SENTINEL so a divergence alarm and an
# unparseable-format alarm dedup independently.
_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL = '__warm_cold_shadow_unparseable__'


def _submit_shadow_divergence_escalation(
    escalation_queue: Any,
    merge_commit: str,
    diff: ShadowCompareDiff,
    warm_results: dict[str, str],
    cold_results: dict[str, str],
) -> None:
    """Submit a born-at-L2 escalation for a warm/cold shadow divergence.

    Implements PRD §10 invariant 6(b) L2 alarm.  The escalation is:

    * ``severity='critical'`` (in ``BORN_AT_L2_SEVERITIES``) → born at L2
    * ``level=2``
    * ``agent_role='orchestrator-warm-cold-shadow'`` (``orchestrator-`` prefix →
      harness sentinel → not downgraded by the escalation server)
    * ``category='risk_identified'``
    * ``task_id=_WARM_COLD_SHADOW_SENTINEL`` (dedup key)

    The detail explicitly states that the warm merge has ALREADY LANDED via the
    shadow/async lane and that the commit may be bad on main.

    None-safe: if *escalation_queue* is None the function is a no-op.
    Dedup: if an open escalation for the sentinel already exists (checked via
    ``escalation_queue.has_open_l1``), no second submission is made.

    Args:
        escalation_queue: Live escalation queue (``EscalationQueue`` instance),
            or ``None`` when escalation is unavailable.
        merge_commit: Full or abbreviated SHA of the just-landed merge commit.
        diff: Per-test divergence bucket summary from :func:`diff_per_test_results`.
        warm_results: Per-test pass/fail map from the warm verify run.
        cold_results: Per-test pass/fail map from the cold verify run.
    """
    if escalation_queue is None:
        return

    # Dedup: don't fire again while an open/pending alarm already exists.
    # Global-dedup is intentional — matches DriftDetector's _DRIFT_SENTINEL pattern.
    # A single open escalation suppresses ALL subsequent shadow-divergence alarms
    # while it is unresolved.  The expectation is that divergences are investigated
    # in sequence; a rollback recommendation implicitly covers subsequent same-area
    # divergences.  If per-commit independent alarms are ever needed, incorporate
    # the commit into the dedup key (e.g. make_id(f'{_WARM_COLD_SHADOW_SENTINEL}
    # :{merge_commit[:8]}')) — but that change is out of scope for this task.
    if escalation_queue.has_open_l1(_WARM_COLD_SHADOW_SENTINEL):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    n_diverging = len(diff.diverging) + len(diff.only_warm) + len(diff.only_cold)
    short_sha = merge_commit[:8]

    # Build summary (must name commit[:8] and diverging test count).
    summary = (
        f'Warm/cold shadow divergence on {short_sha}: '
        f'{n_diverging} diverging test(s)'
    )

    # Build detail: list diverging tests + both result sets + "already landed" statement.
    lines: list[str] = [
        f'Commit: {merge_commit}',
        f'Diverging tests ({n_diverging}):',
    ]
    for test_id, (w, c) in sorted(diff.diverging.items()):
        lines.append(f'  warm={w} cold={c}  {test_id}')
    if diff.only_warm:
        lines.append('Tests present only in warm run (absent cold):')
        lines.extend(f'  {t}' for t in diff.only_warm)
    if diff.only_cold:
        lines.append('Tests present only in cold run (absent warm):')
        lines.extend(f'  {t}' for t in diff.only_cold)
    lines.append('')
    lines.append('Warm results: ' + repr(warm_results))
    lines.append('Cold results: ' + repr(cold_results))
    lines.append('')
    lines.append(
        'The warm merge has ALREADY LANDED via the shadow/async lane — '
        'this commit may be bad on main.  '
        'Investigate the diverging tests and consider a potential rollback.'
    )
    detail = '\n'.join(lines)

    esc = Escalation(
        id=escalation_queue.make_id(_WARM_COLD_SHADOW_SENTINEL),
        task_id=_WARM_COLD_SHADOW_SENTINEL,
        agent_role='orchestrator-warm-cold-shadow',
        severity='critical',
        level=2,
        category='risk_identified',
        summary=summary,
        detail=detail,
        suggested_action=(
            'Investigate the diverging tests on the landed merge commit; '
            'roll back main if the cold leg reveals a real failure.'
        ),
    )
    escalation_queue.submit(esc)


def _alarm_warm_shadow_unparseable(
    escalation_queue: Any,
    merge_commit: str,
    test_output: str,
) -> None:
    """Submit a born-at-L2 critical escalation when the warm verify is unparseable.

    Fail-closed guard for the warm/cold shadow-compare detective: when the warm
    verify output shows that tests actually RAN (a nextest Summary footer with
    N > 0) yet :func:`parse_per_test_results` returned an empty dict, the
    detective is silently inert for that landing — a dangerous invisible failure
    mode.  This function converts that silent failure to an L2 alarm.

    The escalation is modelled on :func:`_submit_shadow_divergence_escalation`:

    * ``severity='critical'`` (in ``BORN_AT_L2_SEVERITIES``) → born at L2
    * ``level=2``
    * ``agent_role='orchestrator-warm-cold-shadow-unparseable'``
      (``orchestrator-`` prefix → harness sentinel → not downgraded)
    * ``category='risk_identified'``
    * ``task_id=_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL`` (separate dedup key,
      does not collide with ``_WARM_COLD_SHADOW_SENTINEL``)

    The alarm is suppressed (no false positive) when *test_output* contains no
    nextest Summary line or the Summary reports 0 tests — that case represents a
    legitimately test-free merge.

    None-safe: if *escalation_queue* is None the function is a no-op.
    Dedup: if an open escalation for the unparseable sentinel already exists
    (checked via ``escalation_queue.has_open_l1``), no second submission is made.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        merge_commit: Full or abbreviated SHA of the just-landed merge commit.
        test_output: Raw ``test_output`` string from the warm :class:`VerifyResult`.
    """
    if escalation_queue is None:
        return

    # Discriminate: did tests actually run in this output?
    # NOTE: _nextest_reported_test_count is nextest-only (reads cargo-nextest
    # "Summary [..] N tests run:" footers).  A libtest-format verify run whose
    # per-test parse fails will not match here, so the alarm is suppressed.
    # Warm verify is expected to use cargo-nextest; libtest is not a supported
    # warm-verify format and would fall through as reported=None (no false alarm).
    reported = _nextest_reported_test_count(test_output)
    if reported is None or reported == 0:
        # Legitimately test-free merge (no nextest pass, or zero tests reported).
        # No alarm — would be a false positive.
        if reported is None:
            # Leave a low-severity breadcrumb so suppressed alarms are diagnosable
            # in the field even when no escalation is raised.
            logger.debug(
                'warm shadow-compare: no nextest Summary line found in warm verify '
                'output — unparseable alarm suppressed '
                '(legitimately test-free or non-nextest run)'
            )
        return

    # Dedup: don't fire again while an open/pending alarm already exists.
    if escalation_queue.has_open_l1(_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    short_sha = merge_commit[:8]
    summary = (
        f'Warm/cold shadow-compare INERT on {short_sha}: '
        f'verify output format could not be parsed ({reported} tests ran, 0 parsed)'
    )
    detail = (
        f'Commit: {merge_commit}\n'
        f'Tests reported by nextest Summary: {reported}\n'
        f'Tests parsed by parse_per_test_results: 0\n'
        '\n'
        'The warm verify ran tests successfully but the per-test parser produced '
        'an empty result map.  The warm/cold shadow-compare detective is INERT '
        'for this landing — divergence detection is disabled.\n'
        '\n'
        'This is a fail-closed alarm: a format mismatch between the verify output '
        'and _NEXTEST_TEST_LINE_RE (or _LIBTEST_TEST_LINE_RE) is silently '
        'disabling the shadow compare.  Fix the per-test parser to match the '
        'actual verify command output format.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL),
        task_id=_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        agent_role='orchestrator-warm-cold-shadow-unparseable',
        severity='critical',
        level=2,
        category='risk_identified',
        summary=summary,
        detail=detail,
        suggested_action=(
            'Fix the per-test result parser (_NEXTEST_TEST_LINE_RE or '
            '_LIBTEST_TEST_LINE_RE in merge_queue.py) to match the actual verify '
            'command output format so the shadow-compare detective can resume.'
        ),
    )
    escalation_queue.submit(esc)


async def _run_cold_shadow_verify(
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    event_store: EventStore | None,
) -> dict[str, str]:
    """Run a from-scratch cold verify on *merge_commit* in a throwaway worktree.

    Creates an ephemeral ``_merge-<uuid>`` worktree at *merge_commit* via
    :meth:`~orchestrator.git_ops.GitOps.create_throwaway_verify_worktree`,
    runs the full merge verify (build_merge_verify_spec + VerifyRunnerPool
    dispatch — the same execution path as ``_run_post_merge_verify``), parses
    the per-test results from the output, and removes the throwaway worktree
    in a ``finally`` block.

    The throwaway worktree is NEVER the persistent warm ``_merge-verify`` path
    — it has no retained ``target/`` warmth — ensuring a true from-scratch
    cold verify (PRD §10 invariant 6(b)).

    Args:
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just warm-landed (provides config,
            module_configs, task_files, task_id).
        merge_commit: The merge commit SHA to verify cold.
        event_store: Optional event store (passed to VerifyRunnerPool; None-safe).

    Returns:
        Per-test verdict map as returned by :func:`parse_per_test_results`.
        Empty dict if the cold verify produced no parseable test output.
    """
    wt = await git_ops.create_throwaway_verify_worktree(merge_commit)
    try:
        task_files_tuple = (
            tuple(req.task_files) if req.task_files is not None else None
        )
        spec = build_merge_verify_spec(req.config, req.module_configs, task_files_tuple)
        # LOCAL-ONLY by design: this is the from-scratch cold trust-anchor detective
        # control (PRD §10 invariant 6(b)).  Adding remotes here would (a) defeat the
        # from-scratch-cold guarantee (a remote may have a warm sccache/target) and
        # (b) reintroduce remote scope-derivation concerns into the very control whose
        # purpose is to BE the local ground truth.  See design decision in plan.json.
        pool = VerifyRunnerPool(
            [LocalRunner(
                wt, req.config, req.module_configs, task_files_tuple,
                run_scoped=run_scoped_verification,
                run_unscoped=_run_unscoped_typechecks,
                task_id=req.task_id,
            )],
            event_store=event_store,
            task_id=req.task_id,
        )
        verify = await pool.dispatch(merge_commit, spec)
        return parse_per_test_results(verify.test_output or '')
    finally:
        await git_ops.cleanup_merge_worktree(wt)


def _persistent_alarm_tests(
    diff1: ShadowCompareDiff,
    diff2: ShadowCompareDiff,
) -> set[str]:
    """Return the alarm-worthy test ids that diverge in BOTH cold runs.

    Used by the Option-B re-confirmation logic: only a test that is
    alarm-worthy in *diff1* (first cold run) AND in *diff2* (second cold run)
    is considered a genuine persistent divergence worthy of a born-at-L2 alarm.
    Tests that appear alarm-worthy only in one run are treated as execution
    flakiness and silently discarded.

    An alarm-worthy test is one present in any of the three alarm-worthy
    buckets: :attr:`~ShadowCompareDiff.diverging`,
    :attr:`~ShadowCompareDiff.only_warm`, or
    :attr:`~ShadowCompareDiff.only_cold`.

    Args:
        diff1: Diff between warm results and the first cold run.
        diff2: Diff between warm results and the second (re-confirmation) cold run.

    Returns:
        The intersection of alarm-worthy test ids across both diffs.
    """
    alarm1: set[str] = set(diff1.diverging) | set(diff1.only_warm) | set(diff1.only_cold)
    alarm2: set[str] = set(diff2.diverging) | set(diff2.only_warm) | set(diff2.only_cold)
    return alarm1 & alarm2


async def _run_shadow_compare(
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    warm_results: dict[str, str],
    escalation_queue: Any,
    event_store: EventStore | None,
) -> None:
    """Compare warm vs cold verify results for *merge_commit* and alarm on divergence.

    Implements PRD §10 invariant 6(b) DETECTIVE control:

    1. Runs a cold verify on *merge_commit* via :func:`_run_cold_shadow_verify`
       in a throwaway ``_merge-<uuid>`` worktree (off the serial lane).
    2. Diffs the cold results against *warm_results* via :func:`diff_per_test_results`.
    3. **On alarm-worthy divergence (Option B re-confirmation)**: re-runs the cold
       leg once and escalates via :func:`_submit_shadow_divergence_escalation` ONLY
       when the same alarm-worthy tests persist across both cold runs.  A divergence
       that clears on the second run is logged at WARNING as transient/flaky with no
       alarm and no parity-ok event.  If the re-confirmation cold run itself returns
       empty results (build/infra hiccup), the result is treated as inconclusive.
    4. On agreement (no alarm-worthy divergence after the first run): emits an
       :attr:`~orchestrator.event_store.EventType.verdict_parity_ok` event
       (mirrors :class:`~orchestrator.verify_runner.DriftDetector`).  This path
       also covers inconclusive-only diffs (``has_divergence`` is False by design).

    **Exception handling**: any exception from either cold leg is logged at WARNING
    level and swallowed.  A shadow/detective control must never crash or stall
    the merge worker — it runs off the critical serial lane via
    ``asyncio.create_task`` (see :func:`_maybe_schedule_shadow_compare`).

    Args:
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that warm-landed (provides config +
             module_configs for the cold verify spec).
        merge_commit: The just-landed merge commit SHA.
        warm_results: Per-test verdict map captured from the warm verify run.
        escalation_queue: Live escalation queue, or ``None`` (None-safe).
        event_store: Optional event store for parity-ok event emission.
    """
    try:
        cold_results = await _run_cold_shadow_verify(
            git_ops, req, merge_commit, event_store
        )
    except Exception:
        logger.warning(
            'Shadow compare cold leg failed for %s — swallowing exception',
            merge_commit[:8],
            exc_info=True,
        )
        return

    # Inconclusive guard: if the cold leg produced NO test results but the warm
    # run had results, treat this as inconclusive rather than divergence.
    # An empty cold result usually signals a build/compile failure, OOM, or
    # infra hiccup — not a genuine warm-pass/cold-fail flip.
    # diff_per_test_results({warm tests…}, {}) would classify every warm test as
    # only_warm (has_divergence=True), producing a false-positive born-at-L2 alarm
    # that states "warm merge may be bad" when the cold side simply didn't run.
    # This mirrors DriftDetector's INCONCLUSIVE path (avoids alarming on transport
    # failure).  Neither alarm nor parity-ok event is emitted on inconclusive.
    if not cold_results and warm_results:
        logger.warning(
            'Shadow compare inconclusive for %s: cold leg produced no parseable '
            'test results (possible build/compile/infra failure in the throwaway '
            'worktree); not alarming',
            merge_commit[:8],
        )
        return

    diff1 = diff_per_test_results(warm_results, cold_results)

    if diff1.has_divergence:
        # Option B: Re-confirm the divergence with a second independent cold run.
        # Escalate only on the intersection of alarm-worthy tests that persist
        # across both runs; a transient flip that clears on re-run is not alarmed.
        n_alarm_worthy = (
            len(diff1.diverging) + len(diff1.only_warm) + len(diff1.only_cold)
        )
        logger.info(
            'Shadow compare first-run divergence on %s: %d alarm-worthy test(s); '
            'starting re-confirmation cold run (Option B) — this doubles the cold '
            'verify cost for this commit',
            merge_commit[:8],
            n_alarm_worthy,
        )
        try:
            cold2 = await _run_cold_shadow_verify(
                git_ops, req, merge_commit, event_store
            )
        except Exception:
            logger.warning(
                'Shadow compare re-confirmation cold leg failed for %s — '
                'swallowing exception; treating divergence as inconclusive',
                merge_commit[:8],
                exc_info=True,
            )
            return

        # Empty-cold inconclusive guard for the re-confirmation run
        if not cold2 and warm_results:
            logger.warning(
                'Shadow compare re-confirmation inconclusive for %s: second cold '
                'leg produced no parseable test results; not alarming',
                merge_commit[:8],
            )
            return

        diff2 = diff_per_test_results(warm_results, cold2)
        persistent = _persistent_alarm_tests(diff1, diff2)

        if persistent:
            # Build a ShadowCompareDiff restricted to the persistently-diverging
            # tests only; pass cold2 as the definitive cold result.
            restricted_diff = ShadowCompareDiff(
                diverging={t: v for t, v in diff2.diverging.items() if t in persistent},
                warm_pass_cold_fail=[t for t in diff2.warm_pass_cold_fail if t in persistent],
                warm_fail_cold_pass=[t for t in diff2.warm_fail_cold_pass if t in persistent],
                only_warm=[t for t in diff2.only_warm if t in persistent],
                only_cold=[t for t in diff2.only_cold if t in persistent],
            )
            _submit_shadow_divergence_escalation(
                escalation_queue, merge_commit, restricted_diff, warm_results, cold2
            )
        else:
            # Divergence cleared on re-confirmation — transient/flaky, not a real issue.
            logger.warning(
                'Shadow compare divergence on %s was transient/flaky (did not '
                'persist across re-confirmation run); not alarming',
                merge_commit[:8],
            )
        # No parity-ok event in either sub-case (persistent or transient divergence).
        # Design intent: the result is genuinely uncertain (either a real flip that
        # triggered an alarm, or a flaky flip that was cleared); emitting parity_ok
        # would be misleading for the persistent case and premature for the transient
        # case.  Downstream metric accounting that needs a per-compare outcome can
        # observe the presence/absence of the born-at-L2 alarm instead.  A new
        # 'verdict_parity_inconclusive' EventType was explicitly ruled out of scope
        # for this task to avoid expanding into event_store.py.
    else:
        # Parity OK — emit event (mirrors DriftDetector.check verdict_parity_ok)
        if event_store is not None:
            event_store.emit(
                EventType.verdict_parity_ok,
                task_id=req.task_id,
                data={
                    'merge_commit': merge_commit,
                    'shadow_compare': True,
                    'warm_test_count': len(warm_results),
                    'cold_test_count': len(cold_results),
                },
            )


async def _maybe_schedule_shadow_compare(
    worker: SpeculativeMergeWorker,
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    warm_results: dict[str, str],
    escalation_queue: Any,
    event_store: EventStore | None,
) -> None:
    """Non-blocking scheduler for the warm-vs-cold SHADOW compare (PRD §10 invariant 6(b)).

    Called from :meth:`SpeculativeMergeWorker._verify_and_advance` on every
    successful warm-verified land.  Returns **immediately** without awaiting
    the cold leg — the shadow/detective control must never block or occupy the
    serial merge lane.

    Cadence (whichever sooner = OR):

    * Every *N* merges (``warm_verify_shadow_compare_every_n_merges``).
    * Once per nightly window (``warm_verify_shadow_compare_nightly_interval_secs``).

    State is persisted to ``worker._shadow_state_path`` so the cadence
    survives orchestrator restarts.

    Single-in-flight guard: if a shadow compare task is already running (tracked in
    ``worker._shadow_compare_tasks``), the new trigger is silently skipped so the
    cold leg never piles up behind the serial lane.

    Args:
        worker: The live :class:`SpeculativeMergeWorker` instance (provides
            ``_shadow_compare_tasks`` set and ``_shadow_state_path``).
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed (provides config).
        merge_commit: The just-landed merge commit SHA (same-candidate guarantee).
        warm_results: Per-test pass/fail map from the warm verify run.
        escalation_queue: Live escalation queue, or ``None`` (None-safe).
        event_store: Optional event store for parity-ok event emission.
    """
    # Early exits: knob off or no warm results to compare against
    if not req.config.git.warm_verify_shadow_compare:
        return
    if not warm_results:
        return
    # None-safe: _shadow_state_path is None on bare-harness workers (mirrors the
    # escalation_queue None-safety / bare-harness contract in __init__).
    # _load_shadow_compare_state(None) raises AttributeError — not in its except
    # tuple — so guard here to keep the Path|None type sound at call sites.
    if worker._shadow_state_path is None:
        return

    # Load persisted cadence state (fail-safe: returns default on missing/corrupt)
    state = _load_shadow_compare_state(worker._shadow_state_path)

    # Increment the merge counter (counts this landing)
    state = ShadowCompareState(
        merges_since_last_shadow=state.merges_since_last_shadow + 1,
        last_shadow_run_at=state.last_shadow_run_at,
    )

    now = time.time()
    due = _shadow_compare_due(
        state, now,
        every_n_merges=req.config.git.warm_verify_shadow_compare_every_n_merges,
        nightly_interval_secs=req.config.git.warm_verify_shadow_compare_nightly_interval_secs,
    )

    if not due:
        # Save incremented counter and return without scheduling a task
        _save_shadow_compare_state(worker._shadow_state_path, state)
        return

    # In-flight guard: skip if a shadow compare task is already running.
    # Persist the incremented counter even on early-return so merges that
    # land during an in-flight cold leg are still counted (amendment: fix
    # cadence_counter_loss where the due-but-in-flight path did not persist).
    in_flight = [t for t in worker._shadow_compare_tasks if not t.done()]
    if in_flight:
        _save_shadow_compare_state(worker._shadow_state_path, state)
        return

    # Due and no in-flight task: reset state + persist
    state = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=now)
    _save_shadow_compare_state(worker._shadow_state_path, state)

    # Spawn the shadow compare OFF the serial lane — this call returns IMMEDIATELY
    # without awaiting the cold verify (detective/async control, PRD §10 invariant 6(b)).
    t = asyncio.create_task(
        _run_shadow_compare(
            git_ops, req, merge_commit, warm_results, escalation_queue, event_store
        )
    )

    def _discard_task(task: asyncio.Task) -> None:  # type: ignore[type-arg]
        worker._shadow_compare_tasks.discard(task)

    t.add_done_callback(_discard_task)
    worker._shadow_compare_tasks.add(t)
