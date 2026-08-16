"""Tool-dispatched failure classifier (PRD: plans/verify-plan-prd.md task δ;
Contract §classify_failure, Invariants C1/C2).

``classify_failure(tool, rc, output, timed_out) -> FailureCategory`` replaces
verify.py's tool-BLIND ``_classify_failure`` regex ladder with a per-tool
dispatch: a tool-T pattern lives ONLY in tool-T's table, so a cargo token can
no longer swallow a pytest/rustc line by construction (C1). Where a tool
offers structured output, the classifier parses it directly instead of
regex-matching human text, falling back to the text table otherwise (C2).

Self-contained: this module defines its own compiled regex copies rather
than importing them from verify.py, both to avoid a verify <->
verify_classify import cycle and to decouple the machine-category
classifier from verify.py's ``_extract_cause_hint`` human-hint ladder — a
separate, deliberately unchanged concern (see that function's docstring in
verify.py).
"""

from __future__ import annotations

import json
import re
import signal

from orchestrator.verify_categories import FailureCategory
from orchestrator.verify_cmd import ToolKind

# ---------------------------------------------------------------------------
# EXTERNAL termination signals (task 3173). ``_run_cmd`` returns
# ``proc.returncode`` verbatim and asyncio sets that to the NEGATIVE signal
# number when a child is terminated by a signal, so ``rc == -9`` means SIGKILL.
#
# This is deliberately an ALLOW-LIST, not the blanket rule "rc < 0 means
# killed". The four signals below are only ever delivered by something OUTSIDE
# the process tree under test — the OOM killer, systemd, verify_cancel.py's
# SIGTERM->grace->SIGKILL sweep, or an operator — so they carry no information
# about the branch.
#
# CRASH signals are excluded on purpose: SIGSEGV (-11), SIGABRT (-6), SIGBUS
# (-7), SIGILL (-4) and SIGFPE (-8) are genuine faults OF THE CODE UNDER TEST
# (a segfaulting native extension, an assertion abort), and they must stay RED.
# Swallowing them as retryable infra would be the false-GREEN inverse of the
# defect this predicate exists to fix — the exact class tasks 2822/1700
# hardened against.
_EXTERNAL_KILL_SIGNALS: frozenset[int] = frozenset({
    signal.SIGKILL,   # 9  — OOM killer, `kill -9`, verify_cancel.py's escalation
    signal.SIGTERM,   # 15 — systemd stop, verify_cancel.py's polite first shot
    signal.SIGINT,    # 2  — operator Ctrl-C / harness interrupt
    signal.SIGHUP,    # 1  — controlling terminal or session went away
})


def is_external_kill_rc(rc: int) -> bool:
    """Return True when *rc* is an asyncio returncode for an EXTERNAL kill.

    See ``_EXTERNAL_KILL_SIGNALS`` for why this is an allow-list rather than
    a blanket ``rc < 0``. Note this reads RAW asyncio returncodes only: a
    shell-reported ``128 + N`` status (e.g. 137 for SIGKILL) is NOT treated as
    a kill, because the predicate must not guess at shell conventions — a
    plain positive rc is a real exit status the process chose to return.
    """
    return rc < 0 and -rc in _EXTERNAL_KILL_SIGNALS

# ---------------------------------------------------------------------------
# Shared regex primitives — patterns that legitimately belong to more than
# one tool's table (e.g. a rustc compile_error diagnostic is meaningful for
# both ToolKind.CARGO_TEST/CARGO_CLIPPY and the ToolKind.OPAQUE generic
# ladder, which must keep classifying byte-identically for any output it
# still receives). Hoisted once here and referenced by every table that
# needs them, rather than duplicated per table — so a future grounding fix
# (in the spirit of the historical cargo_cli_error allowlist tightening
# commits — tasks 1103/1109/1116) lands in one place instead of risking
# re-diverging two copies of "the same" rule, which is the exact
# same-bug-fixed-in-two-places hazard this per-tool-dispatch refactor exists
# to close for the tool-blind ladder itself.
# ---------------------------------------------------------------------------

_PYTEST_INTERNALERROR_RE = re.compile(r'^INTERNALERROR>.+$', re.MULTILINE)
_COMPILE_ERROR_RUSTC_CODE_RE = re.compile(r'error\[E\d+\]:', re.MULTILINE)
_COMPILE_ERROR_STRING_RE = re.compile(r'compile error', re.MULTILINE | re.IGNORECASE)

# cargo CLI errors — narrow allowlist of cargo-only prefixes so rustc
# top-level diagnostics ('error: aborting due to previous errors',
# 'error: could not compile `…`') fall through to unknown_test_failure.
# Intentionally conservative: novel cargo CLI messages not listed here
# (e.g. 'error: unexpected argument', 'error: the manifest-path must be …',
# 'error: manifest path … does not exist') will fall through to
# unknown_test_failure until added to the allowlist. Extend when a new
# cargo CLI failure mode appears in production and needs its own bucket.
#
# Each retained token is grounded in a real observed cargo CLI log line:
#   --              → 'error: --exclude can only be used together with --workspace'
#   no such subcommand
#                   → 'error: no such subcommand: `tset`'
#   failed to (parse|compile|read)
#                   → 'error: failed to parse manifest at `/path/Cargo.toml`'
#                     'error: failed to compile `<name>` (lib), intermediates ...'
#                       (cargo_rustc / compiler job-queue orchestrator)
#                     'error: failed to read `/path/Cargo.toml`'
#   could not find  → 'error: could not find `Cargo.toml` in `/path` or any parent directory'
#
# Dropped tokens — no grounded cargo CLI sample available:
#   `invalid `  — see test_rustc_invalid_diagnostic_not_cargo_cli_error.
#   `package `  — re-add with a tighter suffix (e.g. 'package \`') once a
#                 real cargo log line is observed.
#   `find`      — 'failed to find' (without 'could not' prefix) has no verified
#                 cargo CLI sample; cargo uses 'could not find' for its typical
#                 find-failure case, covered by the top-level alternative above.
#                 See test_failed_to_find_alone_not_cargo_cli_error.
_CARGO_CLI_ERROR_RE = re.compile(
    r'^error: (--|no such subcommand|failed to (parse|compile|read)|could not find)',
    re.MULTILINE,
)

# Rust test runner / pytest FAILED lines — the trailing form ('test NAME ...
# FAILED') is cargo's own test runner shape and also matches a pytest FAILED
# line; the leading form ('FAILED test::name') is pytest-report-shaped only
# and is NOT part of the cargo table (see _CARGO_PATTERNS below).
_TEST_FAILURE_TRAILING_RE = re.compile(r'^.+\s+FAILED\s*$', re.MULTILINE)
_TEST_FAILURE_LEADING_RE = re.compile(r'^FAILED\s', re.MULTILINE)

_NPM_ERROR_RE = re.compile(r'npm\s+(ERR!|error)', re.MULTILINE)
_FLOCK_ERROR_RE = re.compile(r'^flock:', re.MULTILINE)
_TREE_SITTER_GENERATE_RE = re.compile(r'tree-sitter generate', re.MULTILINE)


# ---------------------------------------------------------------------------
# Tool-blind environmental pre-dispatch guard (task 2549) — host-
# infrastructure conditions mean the same thing regardless of which tool's
# command hit them (a full disk is a full disk whether cargo, pytest, or npx
# emitted the line), so DISK_FULL/SEMAPHORE_TIMEOUT are classified in
# `_classify_environmental` BEFORE any per-tool dispatch — mirroring the
# `timed_out` -> `INFRA_TIMEOUT` guard's "the root cause is the environment,
# not the command output" precedent (checked immediately after it, in
# `classify_failure` below). Neither category belongs to any single tool's
# table, so Invariant C1 ("a tool-T pattern lives only in tool-T's table")
# does not apply here.
# ---------------------------------------------------------------------------

# ENOSPC markers reused VERBATIM from merge_queue._ENOSPC_MARKERS (single
# grounded vocabulary shared with merge_queue's _verify_hit_enospc and
# git_ops's disk-pressure detection) — do not invent new ENOSPC strings;
# extend that constant instead if a new grounded sample appears. Matched
# case-insensitively against the whole output, same as _verify_hit_enospc.
_ENOSPC_MARKERS = ('no space left on device', 'os error 28', 'enospc')

# Linker SIGBUS-on-full-disk (git_ops.py: "proceeding into an ENOSPC build
# that SIGBUSes the linker") — a real secondary disk-full mode where the
# human-visible failure is the linker's signal rather than a literal ENOSPC
# string. Requires a linking/collect2 context TOGETHER WITH a signal-7/
# SIGBUS/Bus-error token (both, not either) so a bare 'SIGBUS' mention out of
# context (e.g. a test asserting on signal-handling behavior) cannot
# false-positive into disk_full.
_LINKER_CONTEXT_RE = re.compile(r'linking with|collect2', re.IGNORECASE)
_LINKER_SIGNAL_RE = re.compile(r'signal:?\s*7\b|SIGBUS|Bus error', re.IGNORECASE)

# task 3679: LINE-ANCHORED, producer-grounded slot-timeout markers — the
# POSITIVE detector for the SEMAPHORE_TIMEOUT arm, replacing the whole-output
# token co-occurrence above (deleted in the follow-up step).
#
# ANCHORING CONTRACT: `^` + re.MULTILINE, tolerating LEADING HORIZONTAL
# WHITESPACE ONLY — deliberately NOT an arbitrary leading log/harness prefix.
# This is exactly the discipline `verify._match_clock_marker` (verify.py:3383)
# applies to the `@@REIFY_CLOCK_*@@` family, adopted here for the same
# MEASURED reason (reify task 4998 / esc-4791-52): reify's `run_all.sh
# --include-infra` runs infra tests that capture and re-print these wrapper
# lines, quoting them MID-LINE in assertion prose. A substring-anywhere match
# would classify such a test's output as a slot timeout; a line-anchored one
# cannot.
#
# EMITTER ALLOWLIST — the anchor, carrying `_CARGO_CLI_ERROR_RE`'s
# extend-only-on-a-grounded-sample discipline (see the shared primitives
# above). Each name is grounded in a line reify emits TODAY:
#   lib_test_semaphore    → 'lib_test_semaphore.sh: failed to acquire test slot
#                            within 1800s (LOCK=…, N=8)'
#                           (reify scripts/lib_test_semaphore.sh:173, on
#                            `slot_acquire` rc 75)
#   cargo-test-occt-gated → 'ERROR: cargo-test-occt-gated.sh: failed to acquire
#                            OCCT slot within 1800s (LOCK=…, N=1)'
#                           (reify scripts/cargo-test-occt-gated.sh:182 — the
#                            sole grounded source of the optional `ERROR: `
#                            prefix)
#   lib_lane_x_flock      → 'lib_lane_x_flock.sh: failed to acquire Lane-X lock
#                            within 600s (LOCK=…)'
#                           (reify scripts/lib_lane_x_flock.sh:138)
# Add a fourth name ONLY when a real emitted sample is observed. A generic
# 'failed to acquire … within Ns' shape is deliberately NOT used: any shell
# script anywhere in a verify chain could emit it, which would re-introduce
# precisely the ungrounded looseness this task exists to delete.
#
# `\S+s\b` rather than `\d+s`: reify renders `within unlimiteds` when
# REIFY_TEST_SEMAPHORE_WAIT=unlimited (lib_test_semaphore.sh:170-173), and
# that wart is a real emitted shape, so the pattern must cover it.
#
# `@@REIFY_SLOT_TIMEOUT@@` is NOT emitted by reify today — verified by grep
# over reify `scripts/` on 2026-08-05. It is the forward-compatible anchor for
# reify's companion task (same column-0, first-token `@@REIFY_*@@` emission
# contract as scripts/lib_clock_stop.sh:141), and a harmless no-op until that
# lands. It is therefore a COMPLEMENT to the three grounded anchors and never
# the sole positive — the category is detectable the moment this lands, rather
# than waiting on another repo.
#
# PORTABILITY (task 3679 review). The allowlist above hardcodes three reify
# script basenames into an otherwise project-generic classifier, which makes
# SEMAPHORE_TIMEOUT effectively reify-only detection: a new emitter — in reify
# or in any other project the orchestrator drives — needs a DF source change
# plus a fleet redeploy to be seen. That coupling is accepted here, matching
# the precedent already set by `_MERGE_VERIFY_COLLATERAL`/`_VERIFY_ENV_BROKEN_RE`
# (both encode reify-shaped strings), because three grounded emitters do not
# justify a config surface. The sentinel is the deliberate escape hatch, so
# PREFER IT TO A FOURTH BASENAME: a producer that wants detection without a DF
# release emits `@@REIFY_SLOT_TIMEOUT@@` and is matched by the line above with
# no code change at all. If the allowlist ever does need to grow past three,
# that is the signal to promote the sentinel to the primary contract (or to
# lift the marker list into per-project `verify_env` config) rather than to
# extend the alternation again.
_SLOT_TIMEOUT_SENTINEL_RE = re.compile(r'^[ \t]*@@REIFY_SLOT_TIMEOUT@@', re.MULTILINE)
_SLOT_ACQUIRE_DEADLINE_RE = re.compile(
    r'^[ \t]*(?:ERROR: )?'
    r'(?:lib_test_semaphore|cargo-test-occt-gated|lib_lane_x_flock)\.sh: '
    r'failed to acquire .{1,40}? within \S+s\b',
    re.MULTILINE,
)

# Broken _merge-verify worktree (task 2756) — a guard script (e.g. reify's
# check-manifold-deps.sh) that cannot read the ephemeral merge-verify
# worktree's own Cargo.lock is a broken verify ENVIRONMENT, not a branch
# fault — exactly like DISK_FULL/SEMAPHORE_TIMEOUT above. Line-scoped: the
# read-failure verb and the `Cargo.lock` token must appear on the same line,
# mirroring the single-line grounded sample below.
#
# Grounded in a real observed guard-script failure (reify task 5120 RC-2,
# 2026-07-13):
#   'check-manifold-deps.sh: could not read manifold-csg-sys version from
#    /work/_merge-verify/Cargo.lock'
# Extend the verb alternation only when a new grounded read-failure sample
# appears, mirroring _CARGO_CLI_ERROR_RE's allowlist-tightening discipline.
_VERIFY_ENV_BROKEN_RE = re.compile(
    r"^.*\b(?:could not|cannot|can[’']t|unable to|failed to)\s+read\b.*\bCargo\.lock\b",
    re.MULTILINE | re.IGNORECASE,
)

# Merge-verify restart-collateral shapes (task 2831) — a WIDER class of
# broken-worktree signal than the single Cargo.lock read-failure above.
# Grounded in the reify merge-gate RCA (2026-07-19): when the ephemeral
# _merge-verify worktree is removed out from under a running verify, the
# guard/wrapper/shell emits one of these five shapes. Each carries its own
# worktree-removal-specific anchor (see the per-pattern comments) so a
# genuine branch fault that merely mentions a missing file is not swept into
# ENV_TRANSIENT — mirroring the Cargo.lock pattern's read-verb-adjacent-to-
# lockfile-token discipline above.

# shape 1: a read-failure of some OTHER tracked file (not Cargo.lock).
# Reuses/extends the Cargo.lock pattern's read-verb alternation with the
# "couldn't" contraction (the RCA's own grounded phrasing:
# "couldn't read 'crates/core/src/lib.rs': No such file or directory"),
# requiring the verb immediately adjacent to "read" and "No such file or
# directory" on the same line — a bare application FileNotFoundError with
# no adjacent read verb (e.g. "FileNotFoundError: [Errno 2] No such file
# or directory: 'fixture.json'") does not match. Kept as its own named
# pattern (not folded into the list below) because it alone needs the
# rustc-diagnostic veto immediately below (task 2831 review).
_VERIFY_WORKTREE_COLLATERAL_READ_FAILURE_RE = re.compile(
    r"^.*\b(?:could not|cannot|can[’']t|unable to|failed to|couldn[’']t)\s+read\b.*"
    r"\bNo such file or directory\b",
    re.MULTILINE | re.IGNORECASE,
)

# Veto for shape 1 (task 2831 review): rustc emits this EXACT phrasing —
# "error: couldn't read <path>: No such file or directory" — for a genuine
# branch fault too, e.g. a `#[path = "..."]` attribute pointing at a file the
# branch forgot to add. That diagnostic is textually indistinguishable from
# guard-script worktree-removal collateral on the "couldn't read" line alone,
# but rustc (unlike a shell/guard script) always accompanies ANY
# location-tied diagnostic with its stable span-pointer rendering, e.g.
# " --> src/lib.rs:3:1" — so requiring the ABSENCE of that rendering anywhere
# in *output* lets the genuine compiler fault fall through to per-tool
# dispatch instead of being swallowed as host collateral. Mirrors this
# module's existing clippy::/error[E\d+]: SEMAPHORE_TIMEOUT veto style below.
_RUSTC_DIAGNOSTIC_SPAN_RE = re.compile(r'-->\s*\S+:\d+:\d+', re.MULTILINE)

# Mis-resolved pyright interpreter (task 3367 / esc-3359-1) — the SINGLE
# detection site for this signature. verify.py imports the predicate below
# rather than re-spelling this regex, so there is exactly one pattern to keep
# grounded (Invariant C1 discipline).
_UNRESOLVED_IMPORT_RE = re.compile(
    r'Import "([\w.]+)" could not be resolved \(reportMissingImports\)'
)

# Modules EVERY uv-workspace member declares as a dev dependency, so a HEALTHY
# environment always resolves them. Their absence from the resolved interpreter
# is evidence the INTERPRETER is wrong, not that the branch's dependency set is.
# Keep this to modules that are genuinely universal across members — a module
# only some members depend on would make the sentinel a false-negative source.
_BASELINE_WORKSPACE_MODULES = frozenset({'pytest'})

# Minimum DISTINCT top-level unresolved modules required alongside the sentinel.
# Set far below the observed signature (~40+ distinct modules) and far above the
# realistic branch-defect shape (a branch adds one or two undeclared imports).
_MIN_DISTINCT_UNRESOLVED_IMPORTS = 5


def is_interpreter_missing_workspace_packages(output: str) -> bool:
    """True when *output* shows a Python interpreter holding NONE of the
    workspace's third-party packages — i.e. pyright resolved the wrong
    interpreter, not a branch that is genuinely missing a dependency.

    Task 3367 / esc-3359-1. Measured signature: 509-514 pyright errors, ~40+
    DISTINCT unresolved third-party modules, emitted in a cold merge worktree
    on a DOCS-ONLY diff — a branch that changed no Python at all. Root cause:
    a subproject whose ``[tool.pyright]`` declared no ``venvPath``/``venv``, so
    ``cd <sub> && npx pyright`` resolved its interpreter from the ambient
    ``VIRTUAL_ENV``/``PATH`` — both of which ``verify._target_subprocess_env``
    deliberately strips.

    DISCRIMINATOR CONTRACT — deliberately conservative, and asymmetric on
    purpose. A FALSE POSITIVE is the dangerous direction: it would excuse a
    genuine missing-dependency regression as environmental and let it through
    the merge gate as an infra hold instead of blaming the branch. So BOTH
    conditions must hold, never either alone:

      1. a baseline sentinel module (``_BASELINE_WORKSPACE_MODULES``) is among
         the unresolved imports — every workspace member declares it as a dev
         dependency, so a healthy env ALWAYS resolves it; losing it proves the
         interpreter is wrong rather than the dependency set;
      2. at least ``_MIN_DISTINCT_UNRESOLVED_IMPORTS`` DISTINCT top-level
         modules are unresolved — which rejects the realistic defect shape of a
         branch adding one or two undeclared imports.

    Dotted names collapse to their top-level module, so submodules of a single
    package (``qdrant_client.models``, ``qdrant_client.http``) cannot inflate
    the distinct count on their own.

    Pure and total: any input that grounds neither condition returns ``False``
    rather than raising — a classifier must never be the thing that fails.
    """
    unresolved = unresolved_top_level_modules(output)
    return (
        len(unresolved) >= _MIN_DISTINCT_UNRESOLVED_IMPORTS
        and bool(unresolved & _BASELINE_WORKSPACE_MODULES)
    )


def unresolved_top_level_modules(output: str) -> frozenset[str]:
    """Distinct TOP-LEVEL module names reported unresolved in *output*.

    The one place ``_UNRESOLVED_IMPORT_RE`` is applied — both the predicate
    above and verify.py's loud log (which reports the count) read the signature
    through this helper, so there is never a second copy of the regex.
    """
    return frozenset(
        name.split('.')[0] for name in _UNRESOLVED_IMPORT_RE.findall(output) if name
    )


_VERIFY_WORKTREE_COLLATERAL_PATTERNS: list[re.Pattern[str]] = [
    # shape 2: the verify entrypoint script itself is gone (rc=127 lint
    # stage, <0.4s) — "./scripts/verify.sh: No such file or directory".
    # Anchored on a path-boundary (start-of-line or preceding '/') immediately
    # before 'verify.sh' so an unrelated 'smoke-verify.sh'-style script name
    # cannot false-positive.
    re.compile(r'(?:^|/)verify\.sh: No such file or directory', re.MULTILINE),
    # shape 3: the shell cannot write its captured output back into the
    # (now-removed) _merge-verify worktree — anchored on the `_merge-verify`
    # path token co-occurring with the "could not write output to" phrase so
    # a generic "could not write output to" (some other destination) does not
    # false-positive.
    re.compile(r'could not write output to.*_merge-verify', re.MULTILINE),
    # shape 4: bash's own shell-init cwd probe fails because its cwd (inside
    # the removed worktree) no longer exists — anchored on the full
    # "error retrieving current directory: getcwd:" phrase bash emits, not a
    # bare 'getcwd' mention (which could appear in unrelated code/log text).
    re.compile(
        r'error retrieving current directory:\s*getcwd:',
        re.MULTILINE | re.IGNORECASE,
    ),
    # shape 5: the anticipatory marker reify emits once reify 5261 lands (see
    # this module's classify_failure docstring / plan.json design_decisions
    # for why this is landed ahead of reify 5261). Line-anchored on the
    # literal marker text.
    re.compile(r'^=== INTERRUPTED \(worktree removed\) ===', re.MULTILINE),
]


def _classify_environmental(output: str) -> FailureCategory | None:
    """Tool-blind host-infrastructure guard: DISK_FULL / ENV_TRANSIENT (broken
    ``_merge-verify`` worktree, incl. restart collateral; mis-resolved pyright
    interpreter) / SEMAPHORE_TIMEOUT.

    Checked in ``classify_failure`` immediately after the ``timed_out`` ->
    ``INFRA_TIMEOUT`` guard and before any per-tool dispatch — these
    conditions are properties of the HOST, not of any one tool, so no
    per-tool table is the right home for them. Returns ``None`` (the caller
    proceeds to per-tool dispatch) when none of these conditions is grounded
    in *output*.

    ORDERING: DISK_FULL (ENOSPC/linker) is checked FIRST, then the
    broken-``_merge-verify``-worktree ENV_TRANSIENT checks (task 2756's
    Cargo.lock signature + task 2831's restart-collateral shapes), then the
    pyright-misresolution check, and LAST the SEMAPHORE_TIMEOUT arm.
    DISK_FULL/ENOSPC remains the most-specific root cause and stays first
    (see ``test_enospc_with_unreadable_lock_stays_disk_full`` and
    ``test_enospc_with_collateral_stays_disk_full``).

    SEMAPHORE_TIMEOUT's last position is retained from task 2831's reorder,
    and that ordering REMAINS LOAD-BEARING (task 3679 review). Both signals
    are line-scoped and both can appear in ONE aggregated leg output — verify
    merges stderr into stdout verbatim, and a single ``bash -c`` chain can
    miss a slot deadline in an early step and lose its worktree in a later
    one. Measured on this branch and now pinned as a standing property, not
    a one-off spot-check: every anchored producer line in the grounded
    slot-timeout corpus, concatenated with ANY of the five documented
    collateral shapes (``_VERIFY_WORKTREE_COLLATERAL_READ_FAILURE_RE`` plus
    the four in ``_VERIFY_WORKTREE_COLLATERAL_PATTERNS``), in EITHER
    concatenation order, classifies ENV_TRANSIENT — decided entirely by the
    collateral branch being evaluated first below.

    ENV_TRANSIENT is the intended winner there, for the same reason DISK_FULL
    outranks both: the removed worktree is the more specific root cause, and
    of the two it is the only one with a recovery path
    (``RetryKind.ENV_SERIAL`` versus SEMAPHORE_TIMEOUT's ``RetryKind.NONE``,
    which surfaces to a human). Reading collateral as a slot timeout would
    convert a self-recovering host condition into a blocking escalation.
    Pinned by ``TestAnchoredSlotTimeoutWithCollateralIsEnvTransient``, which
    parametrizes every anchored producer line in the grounded slot-timeout
    corpus against every documented collateral shape, in BOTH concatenation
    orders, at ``ToolKind.OPAQUE`` and one representative non-opaque tool.
    Guard 3's tool-blindness is already cross-producted over every
    ``ToolKind`` by ``TestMergeVerifyCollateralEnvGuard`` above, so
    re-crossing the full axis here would add cases with no added detection
    power. The same corpus separately pins the recovery path itself —
    ``CATEGORY_POLICY[result].retry_kind is RetryKind.ENV_SERIAL`` — the
    consequence named in the paragraph above, not merely the category
    label, at ``ToolKind.OPAQUE``.

    What task 3679 changed is not whether the order matters but WHICH inputs
    reach the tie: before, the arm was a loose co-occurrence that collateral
    output satisfied routinely (any output merely mentioning a lock and a
    timeout), so the ordering was load-bearing for incidental token overlap.
    Now only a genuine anchored producer line can reach the arm, so it is
    load-bearing only for a real co-occurrence of two real host events. Note
    that ``TestReplayedMergeVerifyCollateralWinsOverSemaphore``'s replayed
    5164/5071 fixtures no longer contain any anchored marker, so those two
    now pass structurally and pin nothing about this tie — which is exactly
    why the dedicated co-occurrence test above exists alongside them. That
    disclaimer is itself measured, not inferred: the co-occurrence test's
    mutation-kill (hoisting the SEMAPHORE_TIMEOUT arm above the
    ENV_TRANSIENT branches) fails every one of its cases while the replayed
    5164/5071 fixtures, and the other neighbouring guard classes, stay
    green.

    SEMAPHORE_TIMEOUT — LINE-ANCHORED, producer-grounded marker (task 3679,
    esc-5848-2 / esc-5893-3). The arm fires when *output* contains, AT THE
    START OF A LINE, either the ``@@REIFY_SLOT_TIMEOUT@@`` sentinel or a
    slot-acquisition deadline message from one of three allowlisted reify
    emitter scripts (see ``_SLOT_TIMEOUT_SENTINEL_RE`` /
    ``_SLOT_ACQUIRE_DEADLINE_RE`` above for the per-emitter grounding). The
    event is now detected POSITIVELY, by the producer naming itself, instead
    of being inferred from a whole-output token co-occurrence.

    What this replaced, and why it had to go: the arm previously fired when a
    lock/slot/semaphore token appeared ANYWHERE in *output* together with a
    "timed out" token ANYWHERE else in it. That precondition is satisfied by
    essentially every verify log in the fleet — a repo containing a
    ``package-lock.json``/``uv.lock`` whose runner is invoked under
    ``timeout(1)`` satisfies both halves before any failure has occurred (DF's
    own ``test_command`` carries ``--timeout=300`` seven times). The
    classification was therefore decided almost entirely by SUBTRACTION: three
    successive vetoes (task 2748's clippy/rustc markers, task 2821's test
    verdicts) each removed one shape of false positive without ever making the
    positive correct. Two live escalations got through anyway — rustc BUILT-IN
    lints denied via ``-D warnings``, which carry no ``clippy::`` token, no
    ``error[E\\d+]:`` code and no test verdict — and each held its lane ~5h.
    Meanwhile the arm was BLIND to every real event: none of reify's actual
    deadline lines contains a "timed out" token at all.

    The vetoes are DELETED rather than retained as defence-in-depth, because
    they are no longer load-bearing: a rustc/clippy diagnostic or a
    pytest/cargo verdict cannot begin a line with
    ``lib_test_semaphore.sh: failed to acquire`` or ``@@REIFY_SLOT_TIMEOUT@@``.
    Retaining them would keep ~130 lines of comment justifying a premise
    verified FALSE on both clauses — ``flock -w`` appears nowhere in reify's
    verify path (``lib_slot_acquire.sh`` uses non-blocking ``flock -xn`` in a
    shuffle loop), and this classifier is handed the ENTIRE aggregated verify
    output, not the wrapper's own stream, so "the wrapper's output cannot
    contain a diagnostic" was never the situation being classified.

    The historical "Known gap" (a deterministic shell-script assertion with
    incidental lock+timeout tokens and no verdict marker of any kind) is
    CLOSED, not narrowed further: with no co-occurrence path left, incidental
    tokens cannot reach this category regardless of what else *output*
    contains.

    ENV_TRANSIENT — broken verify worktree (task 2756): a malformed
    ``_merge-verify`` worktree (e.g. an unreadable or missing ``Cargo.lock``)
    is a property of the HOST/environment, not of whichever guard script or
    tool happened to trip over it, so it is checked immediately after the
    more specific ENOSPC/linker DISK_FULL checks — a co-occurring disk-full
    still wins as the more specific root cause (see
    ``test_enospc_with_unreadable_lock_stays_disk_full``) — but BEFORE the
    SEMAPHORE_TIMEOUT arm (task 2831 reorder, now order-independent — see the
    ORDERING note above) — and wins over per-tool dispatch regardless of
    ``ToolKind``. This source is DISTINCT from the pytest-scoped shared-venv
    ``_ENV_TRANSIENT_PATTERNS`` below (different grounded patterns mapping to
    the same ``ENV_TRANSIENT`` category, exactly like how ``DISK_FULL`` above
    is produced by this same tool-blind guard rather than any per-tool
    table) — this does not violate Invariant C1, which forbids duplicating
    the SAME pattern across per-tool tables, not a category being reachable
    from more than one guard.

    ENV_TRANSIENT — merge-verify restart collateral (task 2831): a WIDER
    class of the same broken-``_merge-verify``-worktree host condition,
    grounded in the reify merge-gate RCA (2026-07-19) — five documented
    shapes (``_VERIFY_WORKTREE_COLLATERAL_READ_FAILURE_RE`` plus the four in
    ``_VERIFY_WORKTREE_COLLATERAL_PATTERNS``) emitted when the
    ephemeral worktree is removed out from under a running verify: a
    read-failure of some other tracked file, a missing ``verify.sh``
    entrypoint, a failure to write captured output back into the worktree, a
    shell-init ``getcwd`` failure, and the anticipatory ``=== INTERRUPTED
    (worktree removed) ===`` marker (reify 5261). Checked alongside the
    ``_VERIFY_ENV_BROKEN_RE`` Cargo.lock check immediately above, with the
    identical ordering placement (see the ORDERING note above): after
    DISK_FULL/ENOSPC, before SEMAPHORE_TIMEOUT.

    Shape-1 rustc-diagnostic veto (task 2831 review): the shape-1 read-failure
    text — "couldn't read <path>: No such file or directory" — is also
    rustc's OWN verbatim wording for a genuine branch fault (a
    ``#[path = "..."]`` attribute pointing at a file the branch forgot to
    add), so shape-1 alone additionally requires the ABSENCE of a rustc/
    clippy diagnostic span-pointer line (``_RUSTC_DIAGNOSTIC_SPAN_RE``, e.g.
    " --> src/lib.rs:3:1") anywhere in *output* — rustc always renders that
    span alongside a location-tied diagnostic, but a guard/shell script's
    worktree-removal read failure never does. Shapes 2-5 need no such veto:
    their anchors (the ``verify.sh`` entrypoint token, the ``_merge-verify``
    path token, bash's ``getcwd`` phrase, and the literal INTERRUPTED marker)
    are not plausible rustc/clippy output.

    Known gap (task 2831 review, narrower successor to the task-2748/2821 gap
    above): the rustc-diagnostic veto only disambiguates rustc's OWN
    "couldn't read" phrasing. A non-rustc tool or guard script that happens
    to emit the identical "<read verb> ... No such file or directory" text
    for a genuine branch-caused missing file (not caused by worktree removal)
    is still indistinguishable from host collateral by text alone and would
    still misclassify ENV_TRANSIENT — closing that fully general case would
    need a positive worktree-removal anchor on shape-1 itself, which is not
    grounded in any observed sample today (see plan.json's grounding
    discipline note); left as a residual, accepted gap until a real
    grounded false-positive sample of that shape is observed.

    ENV_TRANSIENT — mis-resolved pyright interpreter (task 3367 / esc-3359-1):
    a THIRD flavour of the same "the host, not the branch" root cause. When the
    checked subproject's ``[tool.pyright]`` pins no ``venvPath``/``venv``,
    pyright resolves its interpreter from the ambient ``VIRTUAL_ENV``/``PATH``
    — both of which ``verify._target_subprocess_env`` deliberately strips — so
    a cold merge worktree type-checks against an environment holding none of
    the workspace's third-party packages and emits hundreds of phantom
    ``reportMissingImports`` errors on a branch with no defect (509-514 errors
    on a DOCS-ONLY diff). Detected via
    ``is_interpreter_missing_workspace_packages``, whose conjunctive
    discriminator (a baseline sentinel module AND >=5 distinct unresolved
    top-level modules) is documented on the predicate itself. Checked after the
    broken-worktree shapes and before the SEMAPHORE_TIMEOUT arm, the same
    placement as those — and, like theirs, that placement stays load-bearing
    on co-occurrence (see the ORDERING note above). Neither signal IMPLIES the
    other, so on the single-signal shapes each is grounded in the order is
    immaterial; on an aggregated output carrying both, this earlier position
    is what makes the recoverable ENV_TRANSIENT label win. Like them, it
    is tool-blind by design — the fleet chain ``cd fused-memory && npx pyright
    && ...`` resolves ``ToolKind.OPAQUE`` while the per-module commands resolve
    ``ToolKind.PYRIGHT``, and one guard-3 branch covers both without any
    per-tool table duplication (Invariant C1 intact — C1 forbids duplicating
    the same pattern across per-tool tables, not a category being reachable
    from more than one guard).
    """
    lower = output.lower()
    if any(marker in lower for marker in _ENOSPC_MARKERS):
        return FailureCategory.DISK_FULL
    if _LINKER_CONTEXT_RE.search(output) and _LINKER_SIGNAL_RE.search(output):
        return FailureCategory.DISK_FULL
    if _VERIFY_ENV_BROKEN_RE.search(output):
        return FailureCategory.ENV_TRANSIENT
    if _VERIFY_WORKTREE_COLLATERAL_READ_FAILURE_RE.search(
        output
    ) and not _RUSTC_DIAGNOSTIC_SPAN_RE.search(output):
        return FailureCategory.ENV_TRANSIENT
    if any(pattern.search(output) for pattern in _VERIFY_WORKTREE_COLLATERAL_PATTERNS):
        return FailureCategory.ENV_TRANSIENT
    if is_interpreter_missing_workspace_packages(output):
        return FailureCategory.ENV_TRANSIENT
    if _SLOT_TIMEOUT_SENTINEL_RE.search(output) or _SLOT_ACQUIRE_DEADLINE_RE.search(output):
        return FailureCategory.SEMAPHORE_TIMEOUT
    return None


def classify_failure(tool: ToolKind, rc: int, output: str, timed_out: bool) -> FailureCategory:
    """Classify a command failure into a ``FailureCategory``, dispatched by tool.

    Guards (checked before any per-tool dispatch, identically for every
    ``ToolKind``):
    1. ``rc == 0``   -> ``FailureCategory.PASSED``
    2. ``timed_out`` -> ``FailureCategory.INFRA_TIMEOUT`` (wins over any
       output pattern — the root cause is the wall-clock limit, not the
       command output)
    2.5. ``is_external_kill_rc(rc)`` -> ``FailureCategory.INFRA_KILL``. A
       process terminated by an EXTERNAL signal never produced an exit
       verdict, so no amount of output pattern-matching can make one up —
       whatever it managed to flush before dying is not a diagnosis. Sits
       ABOVE guard 3 for exactly that reason: a killed process's truncated
       output can carry a misleading environmental marker (a half-written
       ENOSPC line, a lock token) that would otherwise be mined for a root
       cause the run never actually established. Sits BELOW guard 2 because
       our own watchdog SIGKILLing a command it already timed out is a
       timeout, and must keep ``RetryKind.TIMEOUT``. Crash signals
       (SIGSEGV/SIGABRT/SIGBUS/SIGILL/SIGFPE) are deliberately NOT kills —
       see ``_EXTERNAL_KILL_SIGNALS``.
    3. ``_classify_environmental(output)`` -> ``FailureCategory.DISK_FULL``,
       ``FailureCategory.SEMAPHORE_TIMEOUT``, or ``FailureCategory.ENV_TRANSIENT``
       (broken ``_merge-verify`` worktree; or a mis-resolved pyright
       interpreter holding none of the workspace's third-party packages —
       task 3367) when grounded in *output* (wins over any per-tool output
       pattern for the same "root cause is the environment, not the command
       output" reason as guard 2 — a host condition like a full disk, a
       lock/semaphore-slot timeout, an unreadable worktree lockfile, or an
       interpreter resolved from a stripped ambient env is not a property of
       any one tool).

    Then dispatches on *tool* to a per-tool classification table (Invariant
    C1: a tool-T pattern lives ONLY in tool-T's table, so a cargo token can
    no longer swallow a pytest/rustc line by construction):
    - ``ToolKind.PYTEST`` -> ``_classify_pytest`` (notably the env_transient
      shared-venv-mutation signatures, consulted ONLY here — see the
      "BEHAVIORAL NARROWING" note above ``_ENV_TRANSIENT_PATTERNS`` below for
      the resulting constraint: the caller must resolve a command to
      ``ToolKind.PYTEST`` for THIS shared-venv-mutation flavor of
      env_transient auto-recovery to ever fire. Since task 2756,
      ``FailureCategory.ENV_TRANSIENT`` also has a second, ToolKind-
      independent producer — guard 3's ``_classify_environmental`` above,
      via ``_VERIFY_ENV_BROKEN_RE`` — so env_transient auto-recovery overall
      is NOT PYTEST-gated, only this particular pattern source is).
    - ``ToolKind.CARGO_TEST`` / ``ToolKind.CARGO_CLIPPY`` -> a structured
      NDJSON parse (``_parse_cargo_json``) is attempted FIRST (Invariant C2);
      when *output* isn't detected as cargo's ``--message-format json`` NDJSON
      (or carries no error-level compiler-message), falls back to
      ``_classify_cargo``, the text table.
    - ``ToolKind.NPX`` -> ``_classify_npx``.
    - ``ToolKind.PYRIGHT`` -> a structured ``--outputjson`` parse
      (``_parse_pyright_json``) is attempted FIRST (Invariant C2); otherwise
      falls back to ``_classify_opaque`` (no dedicated PYRIGHT text table —
      preserves today's on-the-wire mapping).
    - ``ToolKind.RUFF`` -> a structured ``--output-format json`` parse
      (``_parse_ruff_json``) is attempted FIRST (Invariant C2); otherwise
      falls back to ``_classify_opaque`` (no dedicated RUFF text table —
      preserves today's on-the-wire mapping).
    - ``ToolKind.OPAQUE`` -> ``_classify_opaque``, the FULL legacy generic
      ladder (moved verbatim from verify.py's ``_CLASSIFY_PATTERNS``) — the
      ONLY survivor of the tool-blind ladder.

    Every structured parser is best-effort and exception-safe: *output* that
    isn't valid JSON (or isn't shaped as that tool's schema) yields ``None``
    rather than raising, so the classifier always falls back to the text
    table (Invariant C2's "falls back … does not crash"). The verify COMMANDS
    themselves are unchanged by this — they still emit human text — so the
    human log is unchanged by construction.

    CLOSED DOMAIN: the return value is always a ``FailureCategory`` member —
    see that enum's docstring for the closed 15-value output domain its
    ``CATEGORY_POLICY`` table enforces exhaustively at import time.
    """
    if rc == 0:
        return FailureCategory.PASSED
    if timed_out:
        return FailureCategory.INFRA_TIMEOUT
    if is_external_kill_rc(rc):
        return FailureCategory.INFRA_KILL
    environmental_category = _classify_environmental(output)
    if environmental_category is not None:
        return environmental_category

    if tool is ToolKind.PYTEST:
        return _classify_pytest(output)
    if tool in (ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY):
        # `is not None` (not `or`): FailureCategory.NONE == '' is falsy, so an
        # `or` fallback would wrongly skip a genuine (if currently unused)
        # NONE result from the parser.
        cargo_category = _parse_cargo_json(output)
        return cargo_category if cargo_category is not None else _classify_cargo(output)
    if tool is ToolKind.NPX:
        return _classify_npx(output)
    if tool is ToolKind.PYRIGHT:
        pyright_category = _parse_pyright_json(output)
        return pyright_category if pyright_category is not None else _classify_opaque(output)
    if tool is ToolKind.RUFF:
        ruff_category = _parse_ruff_json(output)
        return ruff_category if ruff_category is not None else _classify_opaque(output)

    return _classify_opaque(output)


# ---------------------------------------------------------------------------
# ToolKind.PYTEST — env_transient (shared-venv-mutation signatures, task
# 2048) is consulted FIRST and ONLY here (Invariant C1's structural win: no
# other tool's table even references these patterns), then INTERNALERROR,
# then FAILED lines, then flock (the test leg is flock-admission-wrapped),
# falling through to UNKNOWN_TEST_FAILURE — which also covers pytest rc=5
# ("no tests ran", kept RED per task 1852 — see _classify_opaque's docstring
# for the same rc=5-stays-RED contract, which applies here identically).
#
# BEHAVIORAL NARROWING vs. the pre-δ tool-blind ladder (intentional, tracked
# here rather than reverted): auto-recovery keyed on THIS pattern table (the
# shared-venv-mutation signatures immediately below) is now reachable ONLY
# when the failing check's config command lexically resolves to
# ToolKind.PYTEST via `parse_config_command` (see verify.py's
# `_tool_for_cmd`) — the tool-blind `_classify_failure` used to consult these
# patterns unconditionally, for every check. A test command the parser
# cannot identify as pytest (invoked indirectly through e.g. `make test`, a
# shell wrapper script, or a bare tox/nox target with no literal `pytest`
# token in the config string) resolves to ToolKind.OPAQUE instead, whose
# ladder deliberately does NOT include these patterns (see
# TestEnvTransientIsPytestScopedC1 in test_verify_classify.py) — so a
# genuine mid-run shared-venv mutation behind such a wrapper would classify
# as UNKNOWN_TEST_FAILURE rather than ENV_TRANSIENT, losing run_verification's
# single-serial-retry auto-recovery (verify.py, gated on
# `category == FailureCategory.ENV_TRANSIENT`) and attributing an infra
# transient to a code regression instead. Today every production test_cmd
# (`uv run pytest ...`, `cd sub && uv run pytest ...`, `python -m pytest`)
# contains a literal `pytest` token and correctly resolves to PYTEST, so this
# is a latent constraint on future config-command shapes, not a currently
# active gap; if a wrapped/indirect test command shape is ever introduced,
# either extend `parse_config_command` to see through the wrapper to PYTEST,
# or revisit whether these patterns belong in the OPAQUE ladder too.
#
# SCOPE OF THE NARROWING (task 2756 update): the "ONLY … ToolKind.PYTEST"
# constraint above binds THIS `_ENV_TRANSIENT_PATTERNS` source only, not
# `FailureCategory.ENV_TRANSIENT` as a whole. Since task 2756,
# `_classify_environmental`'s `_VERIFY_ENV_BROKEN_RE` check (above,
# `classify_failure` guard 3) is a second, ToolKind-independent producer of
# the same category — it runs tool-blind, before per-tool dispatch, so a
# broken-`_merge-verify`-worktree failure trips `ENV_TRANSIENT` (and its
# verify.py auto-recovery retry) regardless of which ToolKind the failing
# check resolves to. verify.py's `_tool_for_cmd` NOTE echoes this same
# now-narrower PYTEST-only claim; read it as scoped identically to this one.
#
# SCOPE OF THE NARROWING (task 3367 update): a THIRD ToolKind-independent
# producer now lives in the same guard 3 —
# `is_interpreter_missing_workspace_packages` (mis-resolved pyright
# interpreter, esc-3359-1). This retires a corollary that USED to hold: it is
# no longer true that "a lint or type-check failure cannot classify as
# env_transient by construction". A TYPE check IS now a possible env_transient
# producer, so any reasoning that relied on env_transient implying the failing
# leg was the TEST leg must be re-derived rather than assumed. The one place
# that reasoning was load-bearing is verify.py's env-recovery retry gate, which
# task 3367 tightens with an explicit `attempt.test.rc != 0` check instead of
# leaning on the retired corollary.
# ---------------------------------------------------------------------------

# Shared-venv mutation signatures (task 2048): a concurrent `uv sync` from
# another orchestrator process on the shared .venv can transiently
# remove-then-readd packages WHILE a consumer is mid-pytest against it (a
# non-atomic install window). Grounded in task 2045's observation: an
# identical `pytest -n auto` that had just passed failed with a pytest usage
# error naming -n/--dist/--max-worker-restart (the xdist plugin vanished),
# `python -c "import xdist"` raised ModuleNotFoundError, and `python -m pip`
# reported "No module named pip". A serial run (`-o addopts=""`) passed,
# confirming it was environmental, not a code regression. Consulted ONLY
# under ToolKind.PYTEST (Invariant C1) — a pytest run's OWN captured output
# is the only place these harness-infrastructure-absence signatures can
# legitimately appear; application/cargo/pyright code does not normally
# emit them, so a genuine code failure in another tool can never be silently
# relabelled environmental just because it happens to mention pip/xdist.
_ENV_TRANSIENT_PATTERNS: list[re.Pattern[str]] = [
    # pytest usage error (rc=4) when the xdist plugin vanished mid-run:
    # "pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0"
    # Anchored to the literal "pytest: error: unrecognized arguments:" prefix
    # that argparse emits for pytest's own CLI (prog='pytest') rather than a
    # bare "unrecognized arguments:" substring, so an unrelated tool's usage
    # error that happens to mention -n/--dist/--max-worker-restart cannot
    # false-positive into env_transient — the same inverse-misattribution
    # hazard the pip pattern below is hardened against with its word-boundary
    # lookahead.
    re.compile(
        r'^.*pytest: error: unrecognized arguments:.*(?:-n\b|--dist\b|--max-worker-restart\b).*$',
        re.MULTILINE,
    ),
    # `python -m pip` when pip itself vanished from the venv. The trailing
    # negative lookahead (?![\w-]) requires 'pip' to be followed by a
    # non-word, non-hyphen boundary so a ModuleNotFoundError whose module
    # name merely STARTS with 'pip' (pipeline, pipx, pipenv, pip_audit,
    # pip-tools) does not false-positive into env_transient — that would be
    # the exact inverse misattribution this feature forbids (a genuine
    # import/code regression silently relabelled environmental, auto-retried,
    # and archive-denied). Grounded positives (task 2045's unquoted runpy
    # line '<executable>: No module named pip' and the quoted 'pip'/"pip"
    # forms) still match since the boundary follows the closing quote (or
    # end of line) in each case.
    re.compile(r'''No module named ['"]?pip['"]?(?![\w-])''', re.MULTILINE),
    # `import xdist` / `import pytest_xdist` when the plugin vanished.
    re.compile(
        r'''ModuleNotFoundError: No module named ['"](xdist|pytest_xdist)['"]''',
        re.MULTILINE,
    ),
]

# Order matters: INTERNALERROR before FAILED so a worker-death run (which has
# both INTERNALERROR> lines and collateral FAILED lines from the dead worker)
# classifies as pytest_internalerror, not test_failure.
_PYTEST_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_PYTEST_INTERNALERROR_RE, FailureCategory.PYTEST_INTERNALERROR),
    (_TEST_FAILURE_TRAILING_RE, FailureCategory.TEST_FAILURE),
    (_TEST_FAILURE_LEADING_RE, FailureCategory.TEST_FAILURE),
    # flock lock failures — the test leg is flock-admission-wrapped.
    (_FLOCK_ERROR_RE, FailureCategory.FLOCK_ERROR),
]


def _classify_pytest(output: str) -> FailureCategory:
    """The PYTEST table: env_transient FIRST (Invariant C1: ONLY here), then
    INTERNALERROR/FAILED/flock, falling through to UNKNOWN_TEST_FAILURE.
    """
    for env_pattern in _ENV_TRANSIENT_PATTERNS:
        if env_pattern.search(output):
            return FailureCategory.ENV_TRANSIENT
    for pattern, category in _PYTEST_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.CARGO_TEST / ToolKind.CARGO_CLIPPY — a structured NDJSON parse
# (Invariant C2) is attempted FIRST; when that finds no error-level
# compiler-message, share one text table: compile_error, then the cargo-CLI
# allowlist, then the (trailing-form only) FAILED line, then tree-sitter
# generate failures, falling through to UNKNOWN_TEST_FAILURE. No
# INTERNALERROR/npm_error/flock_error/leading-FAILED branch here (Invariant
# C1) — those belong to the PYTEST/NPX/OPAQUE tables only, so a stray pytest
# or npm token in cargo output can never be misclassified as a
# cargo-specific category.
# ---------------------------------------------------------------------------


def _parse_cargo_json(output: str) -> FailureCategory | None:
    """Best-effort NDJSON parse of ``cargo … --message-format json`` output.

    cargo's ``--message-format json`` emits one compact JSON object per
    line (NDJSON) — verified against the pinned cargo 1.96.0 binary in this
    worktree, e.g.::

        {"reason":"compiler-message","message":{"level":"error",...}}
        {"reason":"build-finished","success":false}

    Scans line by line for a ``"reason": "compiler-message"`` record whose
    ``message.level == "error"`` -> ``COMPILE_ERROR`` (the structured signal
    Invariant C2 exists to catch — a diagnostic whose rendered human text
    doesn't happen to match any text-table pattern is still correctly
    classified). A line that isn't valid JSON (plain non-JSON text, or a
    JSON-shaped-but-malformed fragment) is skipped rather than raised —
    parsing is best-effort and exception-safe. Returns ``None`` (the caller
    falls back to ``_classify_cargo``, the text table) when no line parses as
    an error-level compiler-message, including when *output* isn't NDJSON at
    all.
    """
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict) or record.get('reason') != 'compiler-message':
            continue
        message = record.get('message')
        if isinstance(message, dict) and message.get('level') == 'error':
            return FailureCategory.COMPILE_ERROR
    return None


_CARGO_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_COMPILE_ERROR_RUSTC_CODE_RE, FailureCategory.COMPILE_ERROR),
    (_COMPILE_ERROR_STRING_RE, FailureCategory.COMPILE_ERROR),
    (_CARGO_CLI_ERROR_RE, FailureCategory.CARGO_CLI_ERROR),
    (_TEST_FAILURE_TRAILING_RE, FailureCategory.TEST_FAILURE),
    (_TREE_SITTER_GENERATE_RE, FailureCategory.TREE_SITTER_GENERATE_ERROR),
]


def _classify_cargo(output: str) -> FailureCategory:
    """The CARGO_TEST/CARGO_CLIPPY table, falling through to UNKNOWN_TEST_FAILURE."""
    for pattern, category in _CARGO_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.NPX — npm_error only, falling through to UNKNOWN_TEST_FAILURE.
# ---------------------------------------------------------------------------

_NPX_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_NPM_ERROR_RE, FailureCategory.NPM_ERROR),
]


def _classify_npx(output: str) -> FailureCategory:
    """The NPX table: npm_error, falling through to UNKNOWN_TEST_FAILURE."""
    for pattern, category in _NPX_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.PYRIGHT — structured JSON only (Invariant C2); no dedicated text
# table exists — non-JSON PYRIGHT output falls through to the OPAQUE
# placeholder ladder (see classify_failure's docstring), preserving today's
# on-the-wire mapping. Adding a dedicated PYRIGHT/type-check FailureCategory
# is out of scope for this task.
# ---------------------------------------------------------------------------


def _parse_pyright_json(output: str) -> FailureCategory | None:
    """Best-effort parse of ``pyright --outputjson`` output.

    pyright's ``--outputjson`` emits one JSON object with a top-level
    ``generalDiagnostics`` array — verified against the pinned pyright
    1.1.408 binary in this worktree, e.g.::

        {"version": "1.1.408", "generalDiagnostics": [
            {"file": "...", "severity": "error", "message": "...", ...}
        ], "summary": {...}}

    A ``generalDiagnostics`` entry with ``severity == "error"`` maps to
    ``UNKNOWN_TEST_FAILURE`` — the preserved on-the-wire category (no
    dedicated PYRIGHT category exists — see module docstring). Exception-safe:
    *output* that isn't valid JSON, or isn't shaped like this schema, returns
    ``None`` rather than raising, so the caller falls back to
    ``_classify_opaque``, the text table.
    """
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    diagnostics = payload.get('generalDiagnostics')
    if not isinstance(diagnostics, list):
        return None
    for diagnostic in diagnostics:
        if isinstance(diagnostic, dict) and diagnostic.get('severity') == 'error':
            return FailureCategory.UNKNOWN_TEST_FAILURE
    return None


# ---------------------------------------------------------------------------
# ToolKind.RUFF — structured JSON only (Invariant C2); no dedicated text
# table exists — non-JSON RUFF output falls through to the OPAQUE placeholder
# ladder (see classify_failure's docstring), preserving today's on-the-wire
# mapping. Adding a dedicated RUFF/lint FailureCategory is out of scope for
# this task.
# ---------------------------------------------------------------------------


def _parse_ruff_json(output: str) -> FailureCategory | None:
    """Best-effort parse of ``ruff check --output-format json`` output.

    ruff's ``--output-format json`` emits a top-level JSON ARRAY of
    violation objects (not wrapped in an object) — verified against the
    pinned ruff binary in this worktree, e.g.::

        [{"cell": null, "code": "F821", "filename": "...",
          "message": "...", "severity": "error", ...}]

    A non-empty violations array maps to ``UNKNOWN_TEST_FAILURE`` — the
    preserved on-the-wire category (no dedicated RUFF category exists — see
    module docstring). Exception-safe: *output* that isn't valid JSON, or
    isn't shaped like this schema, returns ``None`` rather than raising, so
    the caller falls back to ``_classify_opaque``, the text table.
    """
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if not any(isinstance(item, dict) for item in payload):
        return None
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.OPAQUE — the full legacy generic ladder, moved verbatim from
# verify.py's _CLASSIFY_PATTERNS (with its grounding comments, now hoisted
# above as shared primitives). The ONLY surviving dispatch target for the
# tool-blind ladder — every other ToolKind gets its own narrower table so a
# tool-T pattern can only ever match tool-T output (Invariant C1).
# ---------------------------------------------------------------------------

# Compiled regex patterns for the OPAQUE generic ladder — order matters:
# rustc diagnostic codes (error[E0308]) appear before plain 'error:' so
# compile errors are distinguished from cargo CLI errors; INTERNALERROR
# before FAILED so a worker-death run classifies as pytest_internalerror
# rather than test_failure.
_CLASSIFY_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_COMPILE_ERROR_RUSTC_CODE_RE, FailureCategory.COMPILE_ERROR),
    (_COMPILE_ERROR_STRING_RE, FailureCategory.COMPILE_ERROR),
    (_CARGO_CLI_ERROR_RE, FailureCategory.CARGO_CLI_ERROR),
    (_PYTEST_INTERNALERROR_RE, FailureCategory.PYTEST_INTERNALERROR),
    (_TEST_FAILURE_TRAILING_RE, FailureCategory.TEST_FAILURE),
    (_TEST_FAILURE_LEADING_RE, FailureCategory.TEST_FAILURE),
    (_NPM_ERROR_RE, FailureCategory.NPM_ERROR),
    (_FLOCK_ERROR_RE, FailureCategory.FLOCK_ERROR),
    (_TREE_SITTER_GENERATE_RE, FailureCategory.TREE_SITTER_GENERATE_ERROR),
]


def _classify_opaque(output: str) -> FailureCategory:
    """The full legacy generic ladder (verbatim), used for ``ToolKind.OPAQUE``.

    First match wins; a non-matching, non-zero-rc output falls through to
    ``UNKNOWN_TEST_FAILURE`` — this also covers pytest rc=5 ("no tests ran")
    among other non-zero exits. INVARIANT (task 1852): pytest rc=5 is
    intentionally classified RED here, ranked above ``PASSED`` in
    ``CATEGORY_PRIORITY``. A real test target that unexpectedly collects
    zero tests must stay RED so the merge gate catches "tests vanished"
    regressions — the data-module false-RED is fixed in the SCOPING layer
    (verify.py's ``_is_collectable_test_file``), not here.
    """
    for pattern, category in _CLASSIFY_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE
