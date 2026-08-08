"""Declarative failure-category table (PRD: plans/verify-plan-prd.md task α).

Single source of truth for every category ``verify_classify.classify_failure``
can return. Replaces 4 hand-synced registries in verify.py (``_ARCHIVE_DENY_LIST``,
``_CATEGORY_PRIORITY``, ``PREEXISTING_BREAK_SKIP_CATEGORIES``, the sweep
infra-sentinel set) plus the ``endswith('_error')`` archive heuristic with one
table: ``CATEGORY_POLICY: dict[FailureCategory, CategoryPolicy]``.

``FailureCategory`` is a ``StrEnum`` so every member IS its string value —
``json.dumps`` output and every existing str-keyed comparison/membership check
stay byte-identical to today (verify_runner canonical-JSON Invariant 1).

The two ``verify_runner`` ``UNSCOPED_TYPECHECK_*`` sentinels are a separate,
out-of-band gate namespace injected into ``VerifyResult.category`` — never
produced by the classifier — and are deliberately NOT members here (see
plans/verify-plan-prd.md task α design decisions).

Task 3173 adds a fifth derived registry, ``INDETERMINATE_VERDICT_CATEGORIES``,
naming the categories whose leg produced no completed verdict at all — so that
leg may not veto another host's completed PASS in merge_queue's per-land
cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import cast


class FailureCategory(StrEnum):
    """The closed 15-value output domain of ``verify_classify.classify_failure``."""

    INFRA_TIMEOUT = 'infra_timeout'
    INFRA_KILL = 'infra_kill'
    DISK_FULL = 'disk_full'
    SEMAPHORE_TIMEOUT = 'semaphore_timeout'
    CARGO_CLI_ERROR = 'cargo_cli_error'
    COMPILE_ERROR = 'compile_error'
    TREE_SITTER_GENERATE_ERROR = 'tree_sitter_generate_error'
    FLOCK_ERROR = 'flock_error'
    NPM_ERROR = 'npm_error'
    PYTEST_INTERNALERROR = 'pytest_internalerror'
    ENV_TRANSIENT = 'env_transient'
    TEST_FAILURE = 'test_failure'
    UNKNOWN_TEST_FAILURE = 'unknown_test_failure'
    PASSED = 'passed'
    NONE = ''


class RetryKind(Enum):
    """How ``run_verification`` recovers from a given category, if at all.

    Populated for all 15 ``CATEGORY_POLICY`` rows per the PRD contract
    (plans/verify-plan-prd.md task α item 4: ``CategoryPolicy(severity_rank,
    archive, preexisting_probe, is_infra_transient, retry_kind)``) but NOT
    yet dispatched on. ``run_verification`` still decides retries via two
    pre-existing, independent code paths: a pure-timeout while-loop gated on
    raw per-check ``timed_out`` flags (before any category is even
    classified), and a single bounded env-recovery retry gated on
    ``category == FailureCategory.ENV_TRANSIENT``. Neither path is a second
    hand-synced copy of this table's policy — both predate this module, and
    collapsing them into one computation that dispatches on ``retry_kind``
    is the PRD's task ε (``CheckRun``/``VerifyAttempt``), which lands after
    this task in the linear verify.py spine. Wiring dispatch in here first
    would mean touching the same retry-loop internals ε owns, ahead of the
    ``VerifyAttempt`` abstraction that change depends on.
    """

    NONE = 'none'
    TIMEOUT = 'timeout'
    ENV_SERIAL = 'env_serial'


@dataclass(frozen=True)
class CategoryPolicy:
    """Everything the rest of verify.py needs to know about one category.

    ``verdict_indeterminate`` (task 3173) is a THREE-part predicate, and all
    three must hold for a row to set it True:

    1. the leg produced NO completed verdict — it never reached an exit
       decision about the branch, so nothing it "says" is evidence; AND
    2. the branch under test could not have CAUSED that non-completion —
       the cause is a host condition, not the diff; AND
    3. the classifier's EVIDENCE for this category is a STRUCTURAL,
       out-of-band signal the branch cannot forge in text — equivalently,
       ``verify_classify`` documents no residual by which a branch-caused
       failure can land in this category.

    Part (3) is NOT redundant with (2), and the distinction is the whole
    reason this predicate was re-adjudicated (task 3173 review, blocking
    finding 2).  (2) is about the IDEAL condition the row names; (3) is about
    the PATTERN that decides membership.  The consumer — merge_queue's
    per-land cross-check — never observes ground truth about why a leg
    failed; it only ever sees the string ``classify_failure`` returned.  So a
    row whose ideal condition is genuinely host-only, but whose matcher can
    SWALLOW a branch-caused failure, is still a false-GREEN: the misclassified
    branch break arrives wearing the host-condition label and gets waved
    through.  ``ENV_TRANSIENT`` and ``SEMAPHORE_TIMEOUT`` are exactly that
    shape — see their rows below, and the residuals ``verify_classify``'s own
    docstrings document.

    It has no default on purpose: adding a category forces its author to
    adjudicate (2) and (3) explicitly rather than inherit an answer.  Getting
    it wrong in the True direction is a false-GREEN — unverified code lands —
    which is why ``is_infra_transient`` — which only answers "is retrying
    worthwhile", where a wrong answer costs at most a wasted retry — is NOT
    reused here even though the two sets look similar today.
    """

    severity_rank: int
    archive: bool
    preexisting_probe: bool
    is_infra_transient: bool
    verdict_indeterminate: bool
    retry_kind: RetryKind  # see RetryKind's docstring: populated now, dispatched by task ε


CATEGORY_POLICY: dict[FailureCategory, CategoryPolicy] = {
    # verdict_indeterminate=False DESPITE producing no verdict: a hang is one
    # of the few non-completions a branch can genuinely CAUSE (an infinite
    # loop or a deadlock introduced by the diff), so predicate (2) fails and a
    # timed-out leg keeps its veto over another host's PASS.  Fail CLOSED.
    FailureCategory.INFRA_TIMEOUT: CategoryPolicy(
        severity_rank=0, archive=False, preexisting_probe=False,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.TIMEOUT,
    ),
    # A leg terminated by an EXTERNAL signal (SIGKILL/SIGTERM/SIGINT/SIGHUP —
    # the OOM killer, systemd, verify_cancel.py's sweep, an operator) never
    # produced an exit verdict at all, so no amount of output pattern-matching
    # can make one up. Ranked immediately below INFRA_TIMEOUT: both mean "no
    # verdict was produced", so a kill must dominate any co-occurring
    # output-derived category in ``_worst_category``.
    #
    # ``archive=True`` deliberately DIVERGES from every other infra category
    # (INFRA_TIMEOUT / DISK_FULL / SEMAPHORE_TIMEOUT / PYTEST_INTERNALERROR /
    # ENV_TRANSIENT are all archive=False). A kill is the one infra cause that
    # is genuinely off-box and un-reproducible after the fact, so the durable
    # per-attempt record is its ONLY forensic handle — the incident that
    # motivated this row (merge_sha b1ac2c7f) was diagnosable solely because
    # the killed leg happened to land in archive=True UNKNOWN_TEST_FAILURE.
    # Flipping to the usual infra archive=False would delete the evidence
    # trail and make the whole class invisible again.
    #
    # verdict_indeterminate=True, and the SOLE row that holds it: this is the
    # only category whose evidence is a waitpid status rather than a text
    # match, so predicate (3) holds by construction — a branch cannot make the
    # kernel report a negative returncode by printing anything.
    # ``is_external_kill_rc`` reads the RAW asyncio returncode only.
    #
    # The residual, recorded honestly rather than claimed away: a cgroup-v2
    # ``memory.oom.group`` kill SIGKILLs every process in the group INCLUDING
    # our ``start_new_session=True`` direct child, which is the one
    # branch-reachable path to rc=-9 (a memory-hungry diff triggering the
    # group OOM). It is narrow because the ORDINARY branch-caused OOM kills a
    # memory-hungry GRANDCHILD, which the shell wrapper reports as POSITIVE
    # 137 — a plain exit status ``is_external_kill_rc`` already rejects.
    FailureCategory.INFRA_KILL: CategoryPolicy(
        severity_rank=1, archive=True, preexisting_probe=False,
        is_infra_transient=True, verdict_indeterminate=True,
        retry_kind=RetryKind.NONE,
    ),
    # archive=True (task 3683, generalising task 3679's SEMAPHORE_TIMEOUT
    # adjudication to the rest of the infra-transient family):
    # ``retry_kind=NONE`` means no in-verify serial re-run is attempted, and
    # the bounded windows that DO retry this row are category-AGNOSTIC — they
    # test flat ``in INFRA_TRANSIENT_CATEGORIES`` membership with no
    # per-category branch (workflow.py:9020, merge_queue.py:2761,
    # verify.py:7255) — so exhausting any of them files a BLOCKING level-1
    # ``infra_issue`` escalation with this row's log as the only triage
    # artifact beyond a truncated ``failure_report()``. Archival is decided
    # PER ATTEMPT from the category alone (verify.py:1902), inside the retry
    # loop, with no knowledge of whether this is the exhausting attempt — so
    # archive=False discarded the log on the attempt that hands the incident
    # to a human too. Enforced for the whole family by
    # ``_assert_infra_transient_rows_archive``.
    #
    # verdict_indeterminate=False DESPITE is_infra_transient=True: this row
    # fails predicate (2), NOT (3), and the distinction is worth keeping —
    # its ENOSPC markers ARE reliable evidence that the disk really was full,
    # so the classification is not in doubt. CAUSATION is: a diff that
    # generates very large build artifacts, or whose new test emits a runaway
    # log, can genuinely cause the ENOSPC itself. So the branch may well have
    # caused this non-completion and a local disk_full keeps its veto over
    # another host's PASS. Fail CLOSED; only the retry loop treats it as infra.
    FailureCategory.DISK_FULL: CategoryPolicy(
        severity_rank=2, archive=True, preexisting_probe=False,
        is_infra_transient=True, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    # archive=True (task 3679): this category is infra-transient but NOT
    # self-clearing — retry_kind=NONE means no serial re-run is attempted, so
    # it terminates in a blocking human escalation, and the archived log is
    # then the only triage artifact that human has. Both live incidents
    # blocked on exactly that (reify data/verify-logs/5848 and /5893 were
    # never written). Task 3683 found the same reasoning holds for every
    # is_infra_transient row and generalised it — see the DISK_FULL row above
    # and ``_assert_infra_transient_rows_archive``.
    #
    # verdict_indeterminate=False DESPITE is_infra_transient=True: this row
    # fails predicate (3), on the same residual shape as ENV_TRANSIENT below.
    # ``_classify_environmental``'s own docstring (verify_classify.py:438-453)
    # records the task-2748/2821 "Known gap" verbatim: a deterministic
    # SHELL-script gate assertion — their own example is a manifest-drift
    # check, which is precisely a BRANCH-caused failure — that quotes a lock
    # token together with a timeout token but emits no grounded verdict marker
    # still satisfies the loose ``_LOCK_TOKEN_RE`` + ``_TIMEOUT_TOKEN_RE``
    # co-occurrence, with no veto marker to suppress it, and is still
    # classified SEMAPHORE_TIMEOUT. The row flips back once that heuristic
    # requires a positive wrapper-emitted anchor (a ``lib_slot_acquire.sh`` /
    # ``flock -w`` marker) instead of loose co-occurrence.
    FailureCategory.SEMAPHORE_TIMEOUT: CategoryPolicy(
        severity_rank=3, archive=True, preexisting_probe=False,
        is_infra_transient=True, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.CARGO_CLI_ERROR: CategoryPolicy(
        severity_rank=4, archive=True, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.COMPILE_ERROR: CategoryPolicy(
        severity_rank=5, archive=False, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.TREE_SITTER_GENERATE_ERROR: CategoryPolicy(
        severity_rank=6, archive=True, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.FLOCK_ERROR: CategoryPolicy(
        severity_rank=7, archive=True, preexisting_probe=False,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.NPM_ERROR: CategoryPolicy(
        severity_rank=8, archive=True, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    # archive=True (task 3683): the standing rationale for archive=False was
    # "the sweep already retries this category, so archiving it would create
    # spurious human-triage artifacts". That describes exactly one arm — the
    # FIRST-PASS main-tip sweep (verify.py:7062/:7141), which returns a None
    # sentinel, retries next tick indefinitely and never escalates. It is an
    # ADDITIONAL path, not an exemption from the three bounded windows that
    # terminate in front of a human: workflow.py:9020 (default 5 attempts,
    # then escalate_to_human=True → blocking L1 at :14791),
    # merge_queue.py:2761/:3134, and the isolated-confirm gate at
    # verify.py:7255 (an infra-transient hit CONSUMES one of
    # _SWEEP_CONFIRM_MAX_ATTEMPTS; two convert the result into a red-main L1
    # at harness.py:11325). All three test flat INFRA_TRANSIENT_CATEGORIES
    # membership with no per-category branch, so this row cannot be
    # structurally exempt from any of them. Enforced by
    # ``_assert_infra_transient_rows_archive``.
    #
    # verdict_indeterminate=False DESPITE is_infra_transient=True: retrying an
    # INTERNALERROR is usually worthwhile (predicate 1 holds — collection died
    # before any test ran), but predicate (2) FAILS.  `_PYTEST_INTERNALERROR_RE`
    # matches a bare `^INTERNALERROR>`, which a conftest.py or plugin ADDED BY
    # THE DIFF can raise at collection time — and it may raise only on the
    # local host's interpreter/plugin set while a remote with a cached env
    # collects fine.  Letting that pass for a remote PASS would land a
    # branch-caused break: exactly the false-GREEN class tasks 2822/1700
    # hardened against.  Fail CLOSED; only the retry loop treats it as infra.
    FailureCategory.PYTEST_INTERNALERROR: CategoryPolicy(
        severity_rank=9, archive=True, preexisting_probe=False,
        is_infra_transient=True, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    # archive=True (task 3683): the only row in the infra-transient family
    # with a non-NONE retry_kind, and the retry still does not keep it away
    # from a human. ENV_SERIAL is a SINGLE bounded re-run ("retrying test
    # command once", verify.py:5221-5233), and per task 3367 it fires only
    # when the TEST leg failed (``attempt.test.cmd is not None and
    # attempt.test.rc != 0``, :5220-5223) — so a LINT or TYPE leg classified
    # ENV_TRANSIENT gets ZERO retries and is reported directly. Past that, the
    # same family terminus: workflow.py:9020's default-5-attempt window stamps
    # escalate_to_human=True on exhaustion → blocking L1 at :14791 (also
    # merge_queue.py:3134 and verify.py:7255). Enforced by
    # ``_assert_infra_transient_rows_archive``.
    #
    # verdict_indeterminate=False DESPITE is_infra_transient=True: this row
    # fails predicate (3). It previously read "shared-venv mutations and a
    # broken `_merge-verify` worktree — host conditions the diff cannot reach",
    # which describes the row's IDEAL condition but is contradicted by
    # ``_classify_environmental``'s own docstring as a claim about what the
    # MATCHER actually admits (task 3173 review, blocking finding 2). The
    # accepted residual is recorded at verify_classify.py:498-508: shape-1
    # ``_VERIFY_WORKTREE_COLLATERAL_READ_FAILURE_RE`` matches a bare
    # "<read verb> ... No such file or directory", and the rustc-span veto
    # above it only disambiguates rustc's OWN phrasing — so a non-rustc guard
    # or build script tripping over a file THE DIFF DELETED OR RENAMED is
    # textually indistinguishable from worktree-removal collateral and still
    # classifies ENV_TRANSIENT. That is a branch break wearing a host label,
    # so the leg keeps its veto. The row flips back to True only once shape-1
    # carries a POSITIVE worktree-removal anchor rather than being matched by
    # the absence of a rustc span.
    FailureCategory.ENV_TRANSIENT: CategoryPolicy(
        severity_rank=10, archive=True, preexisting_probe=False,
        is_infra_transient=True, verdict_indeterminate=False,
        retry_kind=RetryKind.ENV_SERIAL,
    ),
    FailureCategory.TEST_FAILURE: CategoryPolicy(
        severity_rank=11, archive=False, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.UNKNOWN_TEST_FAILURE: CategoryPolicy(
        severity_rank=12, archive=True, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.PASSED: CategoryPolicy(
        severity_rank=13, archive=False, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
    FailureCategory.NONE: CategoryPolicy(
        severity_rank=14, archive=False, preexisting_probe=True,
        is_infra_transient=False, verdict_indeterminate=False,
        retry_kind=RetryKind.NONE,
    ),
}


def _validate_exhaustive(enum_cls, policy) -> None:
    """Raise if ``policy`` doesn't have exactly one row per ``enum_cls`` member.

    Reusable, unit-testable form of the F1 exhaustiveness guard: a synthetic
    enum/policy pair can be checked directly instead of only observing the
    failure via a real import crash.
    """
    missing = set(enum_cls) - set(policy)
    extra = set(policy) - set(enum_cls)
    if missing or extra:
        raise AssertionError(
            'CATEGORY_POLICY out of sync with FailureCategory: '
            f'missing rows for {sorted(missing)}, stray rows {sorted(extra)}'
        )


_validate_exhaustive(FailureCategory, CATEGORY_POLICY)


# Ordered from highest to lowest severity; used by ``_worst_category``. Derived
# (not hand-written) so a new category can never drift out of sync with its
# CATEGORY_POLICY row — see bug_history: task 2048 required 4 registry edits
# + 2 inline sets for a single category change.
#
# Declared/cast to list[str] rather than list[FailureCategory]: pyright's
# list is invariant, so a precisely FailureCategory-typed list rejects the
# plain-string .index()/`in` lookups every consumer (verify.py's
# _worst_category included) legitimately makes — FailureCategory IS its str
# value (StrEnum), so this is a static-typing-only relaxation, not a runtime
# change. Casting once here — the single source of truth — means consumers
# import CATEGORY_PRIORITY directly with no per-module re-cast.
CATEGORY_PRIORITY: list[str] = cast(
    'list[str]',
    sorted(FailureCategory, key=lambda c: CATEGORY_POLICY[c].severity_rank),
)

# Categories that must NOT be auto-archived. Derived from CategoryPolicy.archive.
ARCHIVE_DENY_LIST: frozenset[FailureCategory] = frozenset(
    c for c, p in CATEGORY_POLICY.items() if not p.archive
)

# Categories skipped by the preexisting-main-break probe. Derived from
# CategoryPolicy.preexisting_probe.
PREEXISTING_BREAK_SKIP_CATEGORIES: frozenset[FailureCategory] = frozenset(
    c for c, p in CATEGORY_POLICY.items() if not p.preexisting_probe
)

# Categories treated as infra-transient by the main-tip sweep (retried rather
# than reported as drift). Derived from CategoryPolicy.is_infra_transient.
INFRA_TRANSIENT_CATEGORIES: frozenset[FailureCategory] = frozenset(
    c for c, p in CATEGORY_POLICY.items() if p.is_infra_transient
)

# Categories whose leg produced NO completed verdict, whose non-completion the
# branch could not have caused, AND whose evidence the branch cannot forge in
# text — so the leg may not veto another host's completed PASS (merge_queue's
# per-land cross-check). Derived from CategoryPolicy.verdict_indeterminate —
# see that field's docstring for the three-part predicate each row answers.
#
# The set is exactly {infra_kill} BY ADJUDICATION, not by accident: it is the
# only category whose evidence is a waitpid status rather than a text match,
# so it is the only one where a branch-caused failure cannot be MISCLASSIFIED
# into the row. Every other candidate was considered and excluded on its own
# row, with the reasoning recorded there: INFRA_TIMEOUT and
# PYTEST_INTERNALERROR fail (2) — a hang and a collection-time INTERNALERROR
# are both non-completions a diff can genuinely CAUSE; DISK_FULL likewise
# fails (2) — a diff can generate the artifacts that fill the disk;
# SEMAPHORE_TIMEOUT and ENV_TRANSIENT fail (3) on the residuals
# verify_classify's own docstrings document (:438-453 and :498-508). Fail
# CLOSED in every one of those cases.
#
# Deliberately NOT spelled as a subtraction from INFRA_TRANSIENT_CATEGORIES.
# The two sets answer different questions ("is retrying worthwhile" vs "may
# this leg overrule another host"), and a set-arithmetic spelling made the
# exclusions invisible: INFRA_TIMEOUT is already is_infra_transient=False, so
# subtracting it was a no-op that merely LOOKED like an adjudication, and
# PYTEST_INTERNALERROR rode in unadjudicated.
#
# UNRELATED to ``merge_disposition.MergeFailureDisposition.INDETERMINATE``
# (task 3178), which is a DIFFERENT LAYER despite the shared word: that one is
# POST-failure attribution (given a verify that already completed and FAILED,
# was the cause the branch, integration skew, or unknown — feeding runs.db
# stats). This registry is PRE-attribution and one layer down: did this leg
# produce a verdict AT ALL. A signal-killed leg never reaches 3178's
# classifier, because there is no completed failure to attribute.
INDETERMINATE_VERDICT_CATEGORIES: frozenset[FailureCategory] = frozenset(
    c for c, p in CATEGORY_POLICY.items() if p.verdict_indeterminate
)


def should_archive(category: str) -> bool:
    """Return True when *category* warrants durable human-triage archival.

    Pure CATEGORY_POLICY table lookup — no ``endswith('_error')`` heuristic.
    A category outside the known 15 (e.g. a verify_runner UNSCOPED_TYPECHECK_*
    sentinel, or any other unrecognized string) defaults to False. See
    ``verify_classify.classify_failure``'s docstring for the closed-domain
    contract that keeps this default from silently misfiring on a future
    category.
    """
    try:
        member = FailureCategory(category)
    except ValueError:
        return False
    return CATEGORY_POLICY[member].archive


def _assert_sentinels_disjoint(sentinels, enum_cls) -> None:
    """Raise if any value in *sentinels* collides with an *enum_cls* member.

    Reusable, unit-testable guard mirroring ``_validate_exhaustive``: proves
    an out-of-band sentinel namespace (e.g. verify_runner's
    UNSCOPED_TYPECHECK_* gate signals, injected into ``VerifyResult.category``
    but never produced by ``classify_failure``) is provably separate from
    ``FailureCategory``'s closed 15-value output domain, so a future
    accidental collision is caught fail-loud at import time instead of
    silently conflating a gate signal with a real classifier category.
    """
    collisions = set(sentinels) & set(enum_cls)
    if collisions:
        raise AssertionError(
            f'sentinel values collide with {enum_cls.__name__} members: '
            f'{sorted(collisions)}'
        )
