"""Tests for the failed-only merge-verify retry producer (PRD verify-retry-failed-only D2).

The DF orchestrator PRODUCES the reify retry contract: per-profile nextest
filter files ({did-not-pass} exact ids) written into the shared merge_wt, plus
the brand-new REIFY_VERIFY_RETRY_* / REIFY_RUN_ALL_MEMBER_SUBSET /
REIFY_GUI_RETRY_SPECS env keys threaded through MergeVerifySpec.verify_env.
reify's verify.sh (α/β/γ) is the CONSUMER — out of scope here.

Covers:
  * ``_build_retry_verify_env`` — writes the debug/release nextest filter files
    and returns the REIFY_* env dict.
  * ``_assemble_retry_verify_env`` — INV-3 tree-OID corroboration gate.
  * ``_load_reify_attempt_sidecar`` — the drift tripwire pinned to the REAL
    bytes reify writes (task 3059).
  * ``_run_post_merge_verify`` wiring under ``req.retry_failed_only``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from orchestrator.merge_queue import MergeRequest


def _attempt0(tree_oid: str):
    """An attempt-0 payload shaped exactly as _build_attempt0_payload builds one.

    Ids are in ``parse_per_test_results``' key space (``"<binary-id> <test-name>"``)
    so the filter-file assertions below exercise the real ``nextest_filter_ids``
    mapping.  Only the FIRST profile (debug) is populated; release carries the
    empty subset a later profile always gets.
    """
    from orchestrator.merge_queue import _Attempt0Payload

    return _Attempt0Payload(
        tree_oid=tree_oid,
        profiles=('debug', 'release'),
        # 'alpha::test_two' and 'test_end_to_end' are absent from verdicts —
        # fail-fast cancelled them, so they are 'not-started' and MUST re-run.
        debug_planned=[
            'crate-a alpha::test_one',
            'crate-a alpha::test_two',
            'crate-a beta::test_three',
            'crate-a::integration test_end_to_end',
        ],
        debug_verdicts={
            'crate-a alpha::test_one': 'pass',
            'crate-a beta::test_three': 'fail',
        },
        # A LATER profile is never narrowed on first-profile evidence.
        release_planned=[],
        release_verdicts={},
        run_all_members=['test_skip_ledger.sh'],
        gui_specs=[],
    )


def _git_ops_returning(oid: str | None):
    git_ops = MagicMock()
    git_ops.get_head_tree_hash = AsyncMock(return_value=oid)
    return git_ops


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_subset_is_failed_union_not_started(
    tmp_path: Path,
) -> None:
    """(a) The first profile's subset is {failed ∪ not-started}, NOT {failed}.

    Under nextest fail-fast a failing attempt-0 CANCELS the not-yet-started
    tests, so they are absent from the verdicts.  A failed-only filter would
    silently never re-run them.
    """
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-1', retry_failed_only=True))
    env = await _assemble_retry_verify_env(git_ops, req, tmp_path, _attempt0('abc123'))

    assert env is not None
    debug_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'])
    assert tmp_path in debug_path.parents  # written under merge_wt
    lines = debug_path.read_text().splitlines()

    # 'beta::test_three' FAILED; 'alpha::test_two' and 'test_end_to_end' were
    # fail-fast CANCELLED.  All three re-run.
    assert set(lines) == {'alpha::test_two', 'beta::test_three', 'test_end_to_end'}
    # 'alpha::test_one' PASSED — it must NOT be in the subset.
    assert 'alpha::test_one' not in lines
    git_ops.get_head_tree_hash.assert_awaited_once_with(tmp_path)


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_filter_lines_are_bare_test_names(
    tmp_path: Path,
) -> None:
    """(b) Filter-file lines are BARE nextest names, never DF's parse keys.

    EMPIRICAL BASIS (cargo-nextest 0.9.136, the version reify's merge gate
    runs): `test(=beta::test_three)` MATCHES, `test(=crate-a beta::test_three)`
    matches NOTHING.  reify wraps each line as `test(=<line>)`
    (verify.sh emit_nextest_pass), so a file of full parse keys is non-empty —
    reify's "retry refused: no subset" loud fallback therefore never fires —
    and matches ZERO tests: a narrowed retry that runs nothing and reports PASS.
    A FALSE GREEN, strictly worse than not narrowing at all.
    """
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-b', retry_failed_only=True))
    env = await _assemble_retry_verify_env(git_ops, req, tmp_path, _attempt0('abc123'))

    assert env is not None
    debug_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'])
    for line in debug_path.read_text().splitlines():
        # A parse key is "<binary-id> <test-name>"; a bare name has no space.
        assert ' ' not in line, (
            f'filter line {line!r} still carries the parse-key form — '
            'nextest would match ZERO tests and the retry would FALSE-GREEN'
        )


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_later_profile_file_is_empty(
    tmp_path: Path,
) -> None:
    """(c) The later profile's filter file EXISTS and is EMPTY.

    An empty per-profile filter file is exactly reify's loud per-profile
    "retry refused: no subset" FULL-fallback trigger, so a profile whose
    attempt-0 pass never ran degrades loudly rather than being skipped.
    """
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-c', retry_failed_only=True))
    env = await _assemble_retry_verify_env(git_ops, req, tmp_path, _attempt0('abc123'))

    assert env is not None
    release_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE'])
    assert release_path.exists()
    assert release_path.read_text() == ''


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_tree_oid_is_the_corroborated_oid(
    tmp_path: Path,
) -> None:
    """(e) REIFY_VERIFY_RETRY_TREE_OID is the DOUBLY-corroborated OID.

    Both arms agree here: DF's own `git_ops.get_head_tree_hash` read and
    reify's independent `git rev-parse HEAD:` stamp from the sidecar.
    """
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-e', retry_failed_only=True))
    env = await _assemble_retry_verify_env(git_ops, req, tmp_path, _attempt0('abc123'))

    assert env is not None
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == 'abc123'
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_rebased_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(d) A rebased tree — DF's read disagrees with the SIDECAR's — → None + WARNING.

    The two arms are genuinely independent reads of the same fact, so a
    disagreement means the tree moved under the retry.  Falls back to the
    existing M4 _reverify_rebased_tree FULL re-verify route.
    """
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('different-oid')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-2', retry_failed_only=True))
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        env = await _assemble_retry_verify_env(
            git_ops, req, tmp_path, _attempt0('abc123')
        )

    assert env is None
    warnings = _mq_warnings(caplog)
    assert any('full verify' in m for m in warnings), warnings
    assert any(('rebas' in m or 'does not match' in m) for m in warnings), warnings


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_unknown_tree_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(d) DF's own arm returning None → None (fail-safe full verify).

    An unreadable OID is not a match — it is an absence of corroboration, and
    the retry must not proceed on one arm alone.
    """
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning(None)
    req = cast('MergeRequest', SimpleNamespace(task_id='t-3', retry_failed_only=True))
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        env = await _assemble_retry_verify_env(
            git_ops, req, tmp_path, _attempt0('abc123')
        )

    assert env is None
    assert any('full verify' in m for m in _mq_warnings(caplog))


# ---------------------------------------------------------------------------
# MATERIAL-NARROWING gate.
#
# `narrowed` is what unlocks MAX_POST_MERGE_VERIFY_NARROWED_RETRIES (2), on the
# premise that a narrowed retry re-runs only the {did-not-pass} subset and is
# therefore cheap.  Three reachable subset shapes make reify run the profile in
# FULL anyway — so calling them "narrowed" would make the merge lane pay TWO
# full re-verifies where the legacy max_enospc budget paid one.  Each must be
# refused HERE (returning None), which routes the caller to the legacy budget
# while leaving the retry itself completely unchanged.
# ---------------------------------------------------------------------------


def _payload(
    *, debug_planned: list[str], debug_verdicts: dict[str, str], tree_oid: str = 'abc123',
):
    from orchestrator.merge_queue import _Attempt0Payload

    return _Attempt0Payload(
        tree_oid=tree_oid,
        profiles=('debug', 'release'),
        debug_planned=debug_planned,
        debug_verdicts=debug_verdicts,
        release_planned=[],
        release_verdicts={},
        run_all_members=[],
        gui_specs=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('case', 'planned', 'verdicts'),
    [
        # (a) attempt-0 red BEFORE any test ran — a compile-gate red,
        # env_transient, or semaphore_timeout.  parse_per_test_results returns
        # {}, so every planned test is 'not-started' and the "subset" is the
        # ENTIRE plan.  reify then trips REIFY_VERIFY_RETRY_MAX_SUBSET (or
        # simply re-runs everything) — a FULL verify wearing a narrowed label.
        ('subset is the whole plan', ['c t1', 'c t2', 'c t3'], {}),
        # (b) the probe legitimately returned [] (a test-free workspace, or one
        # where every test is #[ignore]d): both filter files are empty, so
        # reify's per-profile "retry refused: no subset" fires and runs FULL.
        ('empty plan, empty subset', [], {}),
        # (c) everything attempt-0 planned PASSED (the red was elsewhere —
        # clippy, run_all, a gui spec).  Empty subset -> same reify fallback.
        ('every planned test passed', ['c t1'], {'c t1': 'pass'}),
    ],
)
async def test_assemble_refuses_a_subset_that_narrows_nothing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    case: str,
    planned: list[str],
    verdicts: dict[str, str],
) -> None:
    """A retry that narrows NOTHING must not be reported as narrowed."""
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-mn', retry_failed_only=True))
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        env = await _assemble_retry_verify_env(
            git_ops, req, tmp_path,
            _payload(debug_planned=planned, debug_verdicts=verdicts),
        )

    assert env is None, case
    warnings = _mq_warnings(caplog)
    assert any('narrow NOTHING' in m for m in warnings), (case, warnings)
    # The refusal must name the sizes, so the cost decision is auditable.
    assert any('legacy budget' in m for m in warnings), (case, warnings)


@pytest.mark.asyncio
async def test_assemble_refuses_a_subset_at_the_reify_ceiling(
    tmp_path: Path,
) -> None:
    """A subset at/over reify's REIFY_VERIFY_RETRY_MAX_SUBSET is refused wholesale.

    reify's storm escape (verify.sh:1703, default 5000) rejects such a subset
    and runs the profile FULL, so DF must mirror the ceiling rather than charge
    a full re-verify to the narrowed budget.
    """
    from orchestrator.merge_queue import (
        _REIFY_VERIFY_RETRY_MAX_SUBSET,
        _assemble_retry_verify_env,
    )

    n = _REIFY_VERIFY_RETRY_MAX_SUBSET
    planned = [f'c t{i}' for i in range(n + 2)]
    # n+1 did-not-pass — a real reduction of the plan, but over the ceiling.
    verdicts = {'c t0': 'pass'}

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-mn2', retry_failed_only=True))
    env = await _assemble_retry_verify_env(
        git_ops, req, tmp_path,
        _payload(debug_planned=planned, debug_verdicts=verdicts),
    )
    assert env is None

    # …and one BELOW the ceiling on the same plan shape still narrows, so the
    # assertion above is about the ceiling and not about the plan being large.
    verdicts_small = {f'c t{i}': 'pass' for i in range(2)}
    planned_small = [f'c t{i}' for i in range(n)]
    env_ok = await _assemble_retry_verify_env(
        git_ops, req, tmp_path,
        _payload(debug_planned=planned_small, debug_verdicts=verdicts_small),
    )
    assert env_ok is not None


@pytest.mark.asyncio
async def test_assemble_logs_per_suite_subset_sizes_when_it_narrows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The narrowing site logs the sizes behind the larger-budget decision."""
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-mn3', retry_failed_only=True))
    with caplog.at_level(logging.INFO, logger='orchestrator.merge_queue'):
        env = await _assemble_retry_verify_env(
            git_ops, req, tmp_path, _attempt0('abc123')
        )

    assert env is not None
    msgs = [
        r.getMessage() for r in caplog.records
        if r.name == 'orchestrator.merge_queue'
    ]
    # _attempt0(): 4 planned debug tests, 1 passed -> a 3/4 subset.
    assert any('debug 3/4' in m for m in msgs), msgs


def test_build_retry_verify_env_writes_filter_files_and_env(tmp_path: Path) -> None:
    """_build_retry_verify_env writes per-profile filter files + the REIFY_* env.

    The nextest subsets (potentially thousands of ids) ship as newline filter
    FILES; the small run_all-member / gui-spec lists ship as SPACE-delimited
    env VALUES; tree OID + scope ship as env values.

    CONSUMER EVIDENCE for the space delimiter: reify word-splits both values
    (``_mk_ra_toks=(${REIFY_RUN_ALL_MEMBER_SUBSET})`` verify.sh:2579, and
    ``for _gui_retry_tok in $_gui_retry_specs`` :2141), and the gui
    shell-safety allowlist is ``[A-Za-z0-9._/ -]`` — which EXCLUDES ','.  A
    comma-joined multi-spec gui value is therefore rejected outright (loud full
    fallback, :2156), and a comma-joined multi-member run_all value collapses
    into one unmatchable token.
    """
    from orchestrator.merge_queue import _build_retry_verify_env

    debug = ['beta::test_three', 'alpha::test_two']
    release = ['gamma::test_one']
    env = _build_retry_verify_env(
        nextest_subset_debug=debug,
        nextest_subset_release=release,
        run_all_members=['a.sh', 'b.sh'],
        gui_specs=['src/__tests__/x.test.ts', 'src/__tests__/y.test.ts'],
        tree_oid='deadbeef',
        filter_dir=tmp_path,
    )

    # (2) filter-file env keys are absolute paths under filter_dir.
    debug_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'])
    release_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE'])
    assert debug_path.is_absolute()
    assert release_path.is_absolute()
    assert tmp_path in debug_path.parents
    assert tmp_path in release_path.parents

    # (1) the two filter files exist with EXACTLY the newline-joined ids.
    assert debug_path.read_text() == 'beta::test_three\nalpha::test_two'
    assert release_path.read_text() == 'gamma::test_one'

    # (3) run_all members / gui specs ship SPACE-delimited.
    assert env['REIFY_RUN_ALL_MEMBER_SUBSET'] == 'a.sh b.sh'
    assert (
        env['REIFY_GUI_RETRY_SPECS']
        == 'src/__tests__/x.test.ts src/__tests__/y.test.ts'
    )
    # A comma is unusable on BOTH: reify word-splits, and the gui allowlist
    # rejects ',' outright.
    assert ',' not in env['REIFY_RUN_ALL_MEMBER_SUBSET']
    assert ',' not in env['REIFY_GUI_RETRY_SPECS']

    # (4) tree OID + scope.
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == 'deadbeef'
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'


def test_build_retry_verify_env_empty_subsets_still_write_files(tmp_path: Path) -> None:
    """Empty nextest subsets still write (empty) filter files and set env keys.

    The contract is deterministic: reify's verify.sh always finds the filter
    files at the advertised paths, even when a profile has nothing to retry.

    An EMPTY env value is the deliberate SAFE fallback meaning "run this suite
    in FULL", not "run nothing": verify.sh:2545 gates the run_all subset on
    ``[ -n "${REIFY_RUN_ALL_MEMBER_SUBSET:-}" ]`` and :2127 does the same for
    the gui specs.
    """
    from orchestrator.merge_queue import _build_retry_verify_env

    env = _build_retry_verify_env(
        nextest_subset_debug=[],
        nextest_subset_release=[],
        run_all_members=[],
        gui_specs=[],
        tree_oid='cafef00d',
        filter_dir=tmp_path,
    )

    debug_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'])
    release_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE'])
    assert debug_path.exists()
    assert release_path.exists()
    assert debug_path.read_text() == ''
    assert release_path.read_text() == ''
    assert env['REIFY_RUN_ALL_MEMBER_SUBSET'] == ''
    assert env['REIFY_GUI_RETRY_SPECS'] == ''
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == 'cafef00d'
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'


# ---------------------------------------------------------------------------
# _run_post_merge_verify wiring.  Guarded by req.retry_failed_only; the retry env
# is merged into MergeVerifySpec.verify_env inside the classified-infra-transient
# RETRY branch (task 3059) — never before the first dispatch, which would narrow
# attempt-0 itself.  Driven on the LOCAL path (runner=None) so task 2822's
# remote-green cross-check block (runs ONLY when runner is not None) stays inert
# and the wiring is isolated.
# ---------------------------------------------------------------------------


def _make_git_ops_mock() -> MagicMock:
    m = MagicMock()
    m.get_main_sha = AsyncMock(return_value='main-sha')
    m.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
    m.cleanup_merge_worktree = AsyncMock()
    m.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    m.get_head_tree_hash = AsyncMock(return_value='deadbeef')
    return m


# ---------------------------------------------------------------------------
# _load_reify_attempt_sidecar — the DRIFT TRIPWIRE (task 3059, WORK item 5).
#
# READ THIS BEFORE "FIXING" A FAILURE HERE.  These tests are pinned to the
# CHECKED-IN REAL BYTES of the sidecar reify actually writes
# (tests/fixtures/reify_verify_retry/reify-verify-attempt.json, captured from a
# live warm lane on 2026-07-30).  A failure in this class means the DF/reify
# seam has DRIFTED.  The correct response is to RE-CAPTURE the fixture from a
# live lane and fix this consumer.  Do NOT edit the fixture to make a test pass
# — the shipped D2 producer was authored from its own docstring rather than the
# producer's bytes, and that is exactly the failure this task exists to undo.
#
# The prior D2 loader read a DF-invented `.reify-verify-retry/attempt0.json`
# that nothing in reify or DF has ever written.
# ---------------------------------------------------------------------------

_REIFY_FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'reify_verify_retry'
_REIFY_SIDECAR_FIXTURE = _REIFY_FIXTURE_DIR / 'reify-verify-attempt.json'


def _place_real_sidecar(merge_wt: Path, text: str | None = None) -> Path:
    """Copy the fixture bytes VERBATIM to the path reify writes under merge_wt."""
    from orchestrator.merge_queue import _REIFY_ATTEMPT_SIDECAR_RELPATH

    path = merge_wt / _REIFY_ATTEMPT_SIDECAR_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _REIFY_SIDECAR_FIXTURE.read_text() if text is None else text
    )
    return path


def test_reify_attempt_sidecar_relpath_is_reifys_default() -> None:
    """The sidecar path constant equals reify's REIFY_VERIFY_ATTEMPT_SIDECAR default.

    A bare string equality on purpose: reify's verify.sh:738 defines
    ``_ATTEMPT_SIDECAR_PATH="${REIFY_VERIFY_ATTEMPT_SIDECAR:-target/reify-verify-attempt.json}"``,
    so if that default moves this fails LOUDLY at the constant rather than
    silently degrading every retry to a full verify.
    """
    from orchestrator.merge_queue import _REIFY_ATTEMPT_SIDECAR_RELPATH

    assert _REIFY_ATTEMPT_SIDECAR_RELPATH == 'target/reify-verify-attempt.json'


def test_reify_attempt_sidecar_fixture_schema_is_exactly_three_keys() -> None:
    """The real sidecar carries EXACTLY {tree_oid, profiles, timestamp}.

    Drift in EITHER direction fails: a missing key breaks the loader's
    assumptions, an extra key means reify started publishing something DF may
    need to consume.
    """
    assert json.loads(_REIFY_SIDECAR_FIXTURE.read_text()).keys() == {
        'tree_oid',
        'profiles',
        'timestamp',
    }


def test_load_reify_attempt_sidecar_parses_real_bytes(tmp_path: Path) -> None:
    """The loader parses reify's verbatim bytes; `profiles` is a SPACE-DELIMITED STRING.

    Not a JSON list — a reader who assumed a list would get a per-character
    iteration and silently build a nonsense profile set.
    """
    from orchestrator.merge_queue import _load_reify_attempt_sidecar

    _place_real_sidecar(tmp_path)
    sidecar = _load_reify_attempt_sidecar(tmp_path)

    assert sidecar is not None
    expected_oid = json.loads(_REIFY_SIDECAR_FIXTURE.read_text())['tree_oid']
    assert sidecar.tree_oid == expected_oid
    assert sidecar.profiles == ('debug', 'release')


def test_load_reify_attempt_sidecar_absent_file_returns_none(tmp_path: Path) -> None:
    """No sidecar (reify never stamped one) -> None -> full verify."""
    from orchestrator.merge_queue import _load_reify_attempt_sidecar

    assert _load_reify_attempt_sidecar(tmp_path) is None


@pytest.mark.parametrize(
    ('label', 'text'),
    [
        ('non-JSON bytes', 'not json at all'),
        ('a JSON array', '["tree_oid", "profiles"]'),
        ('missing tree_oid', '{"profiles": "debug", "timestamp": "t"}'),
        ('empty profiles', '{"tree_oid": "abc", "profiles": "", "timestamp": "t"}'),
        (
            'whitespace-only profiles',
            '{"tree_oid": "abc", "profiles": "   ", "timestamp": "t"}',
        ),
        (
            'unknown profile name',
            '{"tree_oid": "abc", "profiles": "debug bench", "timestamp": "t"}',
        ),
    ],
)
def test_load_reify_attempt_sidecar_malformed_returns_none(
    tmp_path: Path, label: str, text: str
) -> None:
    """Every malformed/unusable shape returns None WITHOUT raising -> full verify.

    The unknown-profile case is not paranoia: DF has no
    REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_<X> env key for a third profile, so
    it cannot satisfy reify's "set a filter file for EVERY profile named in
    `profiles`, or fall back to a full verify" obligation (verify.sh:219-230).
    Silently ignoring the unknown profile would narrow a profile that never ran.
    """
    from orchestrator.merge_queue import _load_reify_attempt_sidecar

    _place_real_sidecar(tmp_path, text=text)
    assert _load_reify_attempt_sidecar(tmp_path) is None, label


# ---------------------------------------------------------------------------
# _probe_nextest_planned — the DECIDED source of the attempt-0 planned set
# (candidate (a): a `cargo nextest list` probe in the warm merge lane).
#
# Hermetic: never shells out.  The tests patch the module-level `_run_probe_cmd`
# indirection, which exists for exactly this purpose.  The happy path is fed the
# CHECKED-IN REAL cargo-nextest 0.9.136 bytes.
# ---------------------------------------------------------------------------

_NEXTEST_LIST_FIXTURE = _REIFY_FIXTURE_DIR / 'nextest-list.json'


def _mq_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == 'orchestrator.merge_queue' and r.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('profile', 'extra'),
    [('debug', []), ('release', ['--release'])],
)
async def test_probe_nextest_planned_argv(
    tmp_path: Path, profile: str, extra: list[str]
) -> None:
    """The probe lists the workspace as JSON, in the merge lane, per profile.

    `--release` is the ONLY difference between the two profiles: nextest's
    profile selection is a cargo flag, and the JSON message format is identical.
    """
    from orchestrator import merge_queue as mq

    seen: dict[str, object] = {}

    async def _fake(argv, *, cwd, timeout_secs, env=None):
        seen['argv'] = list(argv)
        seen['cwd'] = cwd
        seen['timeout_secs'] = timeout_secs
        seen['env'] = env
        return 0, _NEXTEST_LIST_FIXTURE.read_text(), ''

    with patch.object(mq, '_run_probe_cmd', _fake):
        planned = await mq._probe_nextest_planned(
            tmp_path, profile, timeout_secs=123.0
        )

    assert planned is not None
    assert seen['argv'] == [
        'cargo',
        'nextest',
        'list',
        '--workspace',
        '--message-format',
        'json',
        *extra,
    ]
    assert seen['cwd'] == tmp_path
    assert seen['timeout_secs'] == 123.0


@pytest.mark.asyncio
async def test_probe_nextest_planned_parses_real_bytes(tmp_path: Path) -> None:
    """rc=0 with real nextest JSON -> the parsed planned ids.

    Delegates to parse_nextest_list_planned, so the ids land in
    parse_per_test_results' key space with no translation.
    """
    from orchestrator import merge_queue as mq
    from orchestrator.merge_shadow import parse_nextest_list_planned

    raw = _NEXTEST_LIST_FIXTURE.read_text()

    async def _fake(argv, *, cwd, timeout_secs, env=None):
        return 0, raw, ''

    with patch.object(mq, '_run_probe_cmd', _fake):
        planned = await mq._probe_nextest_planned(tmp_path, 'debug', timeout_secs=5.0)

    assert planned == parse_nextest_list_planned(raw)
    assert planned  # the fixture is non-empty, so this is a real assertion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('label', 'result', 'exc'),
    [
        ('non-zero exit', (101, '{"rust-suites": {}}', 'error: no such command'), None),
        ('empty stdout', (0, '', ''), None),
        ('whitespace-only stdout', (0, '   \n', ''), None),
        ('unparseable stdout', (0, 'error: could not compile', ''), None),
        ('timeout', None, TimeoutError()),
        ('cargo absent', None, FileNotFoundError('cargo')),
        ('spawn OSError', None, OSError('fork failed')),
    ],
)
async def test_probe_nextest_planned_failure_modes_return_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    label: str,
    result: tuple[int, str, str] | None,
    exc: BaseException | None,
) -> None:
    """Every failure mode returns None and WARNs, naming the profile and reason.

    None is NEVER an empty plan — it routes the caller to a FULL verify.  A
    silent empty plan would narrow the retry to nothing: a FALSE GREEN.
    """
    from orchestrator import merge_queue as mq

    async def _fake(argv, *, cwd, timeout_secs, env=None):
        if exc is not None:
            raise exc
        assert result is not None
        return result

    with (
        caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
        patch.object(mq, '_run_probe_cmd', _fake),
    ):
        planned = await mq._probe_nextest_planned(tmp_path, 'release', timeout_secs=5.0)

    assert planned is None, label
    warnings = _mq_warnings(caplog)
    assert any('release' in m for m in warnings), (label, warnings)


@pytest.mark.asyncio
async def test_probe_nextest_planned_nonzero_exit_warning_carries_stderr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-zero exit must say WHY, not just "exited 101".

    Loud-over-silent-degradation (docs/legibility/design-invariants.md,
    structured-facts-at-failure): 'cargo-nextest not installed', 'unknown flag',
    'workspace failed to compile' and 'lockfile out of date' are all rc!=0 and
    are otherwise indistinguishable in the log.  The tail is BOUNDED so a
    runaway build log cannot flood the orchestrator log.
    """
    from orchestrator import merge_queue as mq

    noise = 'x' * 5000
    stderr = f'{noise}\nerror: no such command: `nextest`\n'

    async def _fake(argv, *, cwd, timeout_secs, env=None):
        return 101, '', stderr

    with (
        caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
        patch.object(mq, '_run_probe_cmd', _fake),
    ):
        planned = await mq._probe_nextest_planned(
            Path('/nonexistent'), 'debug', timeout_secs=5.0
        )

    assert planned is None
    hit = [m for m in _mq_warnings(caplog) if 'exited 101' in m]
    assert hit, _mq_warnings(caplog)
    assert 'no such command' in hit[0], hit[0]
    assert len(hit[0]) < len(stderr), (
        'the stderr tail must be bounded, not the whole log'
    )
    assert mq._PROBE_STDERR_TAIL_CHARS <= 2000


@pytest.mark.asyncio
async def test_probe_threads_the_verify_env_into_the_subprocess() -> None:
    """The probe must model the SAME build the verify performs.

    `build_merge_verify_spec` puts `config.effective_verify_env` on the spec —
    the shared sccache backend (RUSTC_WRAPPER, SCCACHE_*) that is *why* a warm
    merge lane is warm.  A probe run under the bare ORCHESTRATOR environment
    compiles without it, so on exactly the lanes this feature targets it burns
    minutes and times out — silently degrading every narrowed retry to a full
    verify while still paying the probe cost.
    """
    from orchestrator import merge_queue as mq

    seen: dict[str, object] = {}

    async def _fake(argv, *, cwd, timeout_secs, env=None):
        seen['env'] = env
        return 0, _NEXTEST_LIST_FIXTURE.read_text(), ''

    verify_env = {'RUSTC_WRAPPER': '/usr/bin/sccache', 'SCCACHE_DIR': '/tmp/sccache'}
    with patch.object(mq, '_run_probe_cmd', _fake):
        planned = await mq._probe_nextest_planned(
            Path('/nonexistent'), 'debug', timeout_secs=5.0, verify_env=verify_env,
        )

    assert planned is not None
    assert seen['env'] == verify_env


@pytest.mark.asyncio
async def test_probe_nextest_planned_timeout_is_bounded_by_a_named_constant() -> None:
    """A module constant bounds the probe so it can never wedge the merge lane."""
    from orchestrator.merge_queue import _NEXTEST_LIST_PROBE_TIMEOUT_SECS

    assert isinstance(_NEXTEST_LIST_PROBE_TIMEOUT_SECS, (int, float))
    assert 0 < _NEXTEST_LIST_PROBE_TIMEOUT_SECS <= 900


# ---------------------------------------------------------------------------
# _run_probe_cmd — the ONE real subprocess seam this change adds.
#
# Every other test in this file patches it out, so without these its argv/cwd/
# env wiring, its TimeoutError -> kill-the-process-GROUP -> reap -> re-raise
# path, its `proc.returncode or 0` normalisation and its errors='replace'
# decode would all ship unexercised.  The timeout path is the load-bearing one:
# a hang there would wedge the merge lane the ceiling exists to protect.
#
# Hermetic: driven against `sh -c`, never cargo.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_probe_cmd_success_returns_rc_stdout_stderr(tmp_path: Path) -> None:
    """rc, stdout and stderr all come back, decoded, from a real subprocess."""
    from orchestrator import merge_queue as mq

    rc, stdout, stderr = await mq._run_probe_cmd(
        ['sh', '-c', 'printf hi; printf oops >&2'], cwd=tmp_path, timeout_secs=30.0,
    )

    assert (rc, stdout, stderr) == (0, 'hi', 'oops')


@pytest.mark.asyncio
async def test_run_probe_cmd_runs_in_cwd_with_the_layered_env(tmp_path: Path) -> None:
    """`cwd` is honoured and `env` layers OVER os.environ (not replacing it)."""
    from orchestrator import merge_queue as mq

    (tmp_path / 'marker.txt').write_text('x')

    rc, stdout, _ = await mq._run_probe_cmd(
        ['sh', '-c', 'ls marker.txt; printf "|%s|%s" "$DF_PROBE_MARKER" "${PATH:+haspath}"'],
        cwd=tmp_path,
        timeout_secs=30.0,
        env={'DF_PROBE_MARKER': 'sccache'},
    )

    assert rc == 0
    assert 'marker.txt' in stdout
    # The override is present AND the inherited PATH survived: env must be
    # {**os.environ, **verify_env}, never a bare replacement (a bare dict would
    # strip PATH and cargo would not even be findable).
    assert '|sccache|haspath' in stdout


@pytest.mark.asyncio
async def test_run_probe_cmd_nonzero_exit_is_returned_not_raised(tmp_path: Path) -> None:
    """A non-zero exit is DATA (the caller warns + falls back), not an exception."""
    from orchestrator import merge_queue as mq

    rc, stdout, _ = await mq._run_probe_cmd(
        ['sh', '-c', 'exit 3'], cwd=tmp_path, timeout_secs=30.0,
    )

    assert rc == 3
    assert stdout == ''


@pytest.mark.asyncio
async def test_run_probe_cmd_timeout_raises_promptly_and_reaps_the_child(
    tmp_path: Path,
) -> None:
    """The timeout path raises TimeoutError FAST and leaves no live child.

    This is the load-bearing path: a hang here would wedge the merge lane the
    _NEXTEST_LIST_PROBE_TIMEOUT_SECS ceiling exists to protect.  Driven against
    a real `sh -c sleep`, so the kill/reap/re-raise sequence is exercised end to
    end rather than mocked.
    """
    import time as _time

    from orchestrator import merge_queue as mq

    spawned: list = []
    real_exec = asyncio.create_subprocess_exec

    async def _spy_exec(*argv, **kwargs):
        proc = await real_exec(*argv, **kwargs)
        spawned.append(proc)
        return proc

    started = _time.monotonic()
    with (
        patch.object(asyncio, 'create_subprocess_exec', _spy_exec),
        pytest.raises(TimeoutError),
    ):
        await mq._run_probe_cmd(
            ['sh', '-c', 'sleep 120'], cwd=tmp_path, timeout_secs=0.5,
        )
    elapsed = _time.monotonic() - started

    assert 0.4 < elapsed < 20.0, (
        f'the timeout path must fire at the ceiling and not hang (took {elapsed:.1f}s)'
    )
    # The reap is synchronous with the raise: the child is dead and collected
    # by the time the caller sees TimeoutError — never left running or zombied.
    assert len(spawned) == 1
    assert spawned[0].returncode is not None, (
        'the probe child must be reaped before TimeoutError propagates'
    )


def test_run_probe_cmd_spawns_its_own_session(tmp_path: Path) -> None:
    """The spawn MUST pass start_new_session=True.

    `cargo nextest list` spawns rustc / build-script children.  Without its own
    session/process group there is no group for the timeout path to kill, so a
    bare proc.kill() would leave those compiles running in the merge lane the
    ceiling exists to protect.  Asserted at the SPAWN because whether an orphan
    happens to die with its parent is OS/environment-dependent — the flag is
    the thing this module actually controls.
    """
    from orchestrator import merge_queue as mq

    seen: dict[str, object] = {}

    async def _fake_exec(*argv, **kwargs):
        seen.update(kwargs)
        raise OSError('spawn intercepted')

    async def _drive() -> None:
        with (
            patch.object(asyncio, 'create_subprocess_exec', _fake_exec),
            pytest.raises(OSError),
        ):
            await mq._run_probe_cmd(
                ['cargo', 'nextest', 'list'], cwd=tmp_path, timeout_secs=1.0,
            )

    asyncio.run(_drive())

    assert seen.get('start_new_session') is True, seen
    assert seen.get('stderr') == asyncio.subprocess.PIPE, (
        'stderr must be CAPTURED, not DEVNULL — a bare exit code cannot tell '
        '"cargo-nextest absent" from "workspace failed to compile"'
    )


@pytest.mark.asyncio
async def test_kill_probe_process_tree_delegates_to_the_shared_group_kill() -> None:
    """The timeout kill reuses shared.proc_group.terminate_process_group.

    That helper is the repo's single blessed SIGTERM->SIGKILL group kill (it
    already backs verify.py), and it is the reason the pgid must be CAPTURED at
    spawn rather than re-read with os.getpgid(): by kill time the process may
    have been reaped and its pid recycled onto an unrelated group, which a
    hand-rolled killpg would then signal.
    """
    from orchestrator import merge_queue as mq

    proc = cast('asyncio.subprocess.Process', SimpleNamespace(pid=4242))
    seen: dict[str, object] = {}

    async def _fake_terminate(p, pgid, *, grace_secs=5.0):
        seen.update(proc=p, pgid=pgid, grace_secs=grace_secs)

    with patch('shared.proc_group.terminate_process_group', _fake_terminate):
        await mq._kill_probe_process_tree(proc, 4242)

    assert seen['proc'] is proc
    assert seen['pgid'] == 4242
    assert seen['grace_secs'] == mq._PROBE_KILL_GRACE_SECS


@pytest.mark.asyncio
async def test_run_probe_cmd_kills_with_the_pgid_captured_at_spawn(
    tmp_path: Path,
) -> None:
    """The pgid handed to the kill is proc.pid, read while the child is alive."""
    from orchestrator import merge_queue as mq

    seen: dict[str, object] = {}

    async def _spy_kill(proc, pgid):
        seen['pgid'] = pgid
        seen['pid'] = proc.pid
        proc.kill()
        await proc.wait()

    with (
        patch.object(mq, '_kill_probe_process_tree', _spy_kill),
        pytest.raises(TimeoutError),
    ):
        await mq._run_probe_cmd(
            ['sh', '-c', 'sleep 120'], cwd=tmp_path, timeout_secs=0.5,
        )

    # start_new_session=True makes pgid == pid by POSIX guarantee.
    assert seen['pgid'] == seen['pid']


# ---------------------------------------------------------------------------
# _build_attempt0_payload — the payload is built from DF-OWNED attempt-0 data
# (task 3059, WORK item 1), not from a phantom reify sidecar.
#
# PROFILE ATTRIBUTION is the subtle part.  VerifyResult.test_output is ONE
# blended blob across both nextest passes and the verdict key
# ("<binary-id> <test-name>") carries no profile, so a `pass` observed in
# profile 1 is indistinguishable from a test that never ran in profile 2.
# verify.sh:219-230 makes "never narrow a profile that never ran" DF's seam
# obligation.  Hence: only the FIRST profile named in the sidecar is narrowed;
# every later profile gets an EMPTY subset, which reify turns into its loud
# per-profile "retry refused: no subset" FULL fallback.
# ---------------------------------------------------------------------------

# A recorded-shape nextest fail-fast blob: PASS lines, one FAIL, and tests that
# never appear at all because fail-fast cancelled them.  The '(  N/M)' progress
# counter is the real cargo-nextest 0.9.136 human-output shape that
# _NEXTEST_TEST_LINE_RE consumes and discards.
_FAIL_FAST_OUTPUT = """\
    Starting 5 tests across 3 binaries
        PASS [   0.130s] (  1/  5) crate-a alpha::test_one
        PASS [   0.021s] (  2/  5) crate-a alpha::test_two
        FAIL [   0.044s] (  3/  5) crate-a beta::test_three
Canceling due to test failure
=== Summary: 3 discovered, 1 failed ===
=== FAILED: test_skip_ledger.sh ===
FAILED test_skip_ledger.sh
"""

# Everything cargo-nextest planned, per the probe (a superset is sound).
_PROBED_PLANNED = [
    'crate-a alpha::test_one',
    'crate-a alpha::test_two',
    'crate-a beta::test_three',
    'crate-a::integration test_end_to_end',
    'crate-b gamma::test_one',
]


def _verify_result(test_output: str):
    from orchestrator.verify import VerifyResult

    return VerifyResult(
        passed=False,
        test_output=test_output,
        lint_output='',
        type_output='',
        summary='attempt-0 red',
        category='disk_pressure',
    )


@pytest.mark.asyncio
async def test_build_attempt0_payload_narrows_only_the_first_profile(
    tmp_path: Path,
) -> None:
    """The FIRST sidecar profile is narrowed; every LATER profile gets an empty set.

    A `pass` seen in the debug pass is NOT evidence the same test ran in the
    release pass, so release must not be narrowed on debug-profile evidence.
    """
    from orchestrator import merge_queue as mq
    from orchestrator.merge_shadow import parse_per_test_results

    _place_real_sidecar(tmp_path)
    calls: list[str] = []

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        calls.append(profile)
        return list(_PROBED_PLANNED)

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is not None
    assert payload.profiles == ('debug', 'release')

    # debug — the FIRST profile — carries the probe's plan and attempt-0 verdicts.
    assert payload.debug_planned == _PROBED_PLANNED
    assert payload.debug_verdicts == parse_per_test_results(_FAIL_FAST_OUTPUT)
    assert payload.debug_verdicts['crate-a alpha::test_one'] == 'pass'
    assert payload.debug_verdicts['crate-a beta::test_three'] == 'fail'

    # release — a LATER profile — is never narrowed on profile-1 evidence.
    assert payload.release_planned == []
    assert payload.release_verdicts == {}

    # The probe is asked ONCE, only for the first profile: a later profile's
    # planned set is never even requested (it could not be used).
    assert calls == ['debug']


@pytest.mark.asyncio
async def test_build_attempt0_payload_fields_come_from_df_owned_sources(
    tmp_path: Path,
) -> None:
    """tree_oid from reify's sidecar; run_all members parsed; gui_specs empty."""
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path)

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        return list(_PROBED_PLANNED)

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is not None
    expected_oid = json.loads(_REIFY_SIDECAR_FIXTURE.read_text())['tree_oid']
    assert payload.tree_oid == expected_oid
    assert payload.run_all_members == ['test_skip_ledger.sh']
    # Deliberately empty: no real reify gui failure log was available to pin a
    # fixture to, and an empty value makes verify.sh run the FULL gui suite.
    assert payload.gui_specs == []


@pytest.mark.asyncio
async def test_build_attempt0_payload_empty_verdicts_still_yields_a_payload(
    tmp_path: Path,
) -> None:
    """An empty verdict map is a sound (if wide) subset: everything not-started.

    attempt-0 can die before any test line is emitted (a compile-gate red).
    planned − {} = every test 'not-started', which re-runs everything in the
    plan — wide, but never skipping.  reify's REIFY_VERIFY_RETRY_MAX_SUBSET
    ceiling is what rejects a runaway subset, not a silent DF narrowing.
    """
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path)

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        return list(_PROBED_PLANNED)

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result('error: could not compile `crate-a`\n')
        )

    assert payload is not None
    assert payload.debug_verdicts == {}
    assert payload.debug_planned == _PROBED_PLANNED


@pytest.mark.asyncio
async def test_build_attempt0_payload_none_when_sidecar_absent(tmp_path: Path) -> None:
    """No sidecar -> None -> full verify. The probe is never even run."""
    from orchestrator import merge_queue as mq

    called = False

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        nonlocal called
        called = True
        return list(_PROBED_PLANNED)

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is None
    assert not called


@pytest.mark.asyncio
async def test_build_attempt0_payload_none_when_sidecar_malformed(
    tmp_path: Path,
) -> None:
    """A malformed sidecar (loader returned None) -> None -> full verify."""
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path, text='{"tree_oid": "abc", "profiles": "bench"}')

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        return list(_PROBED_PLANNED)

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is None


@pytest.mark.asyncio
async def test_build_attempt0_payload_none_when_the_plan_misses_a_real_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A probed plan that does not COVER attempt-0's failures refuses to narrow.

    `build_fail_fast_map` iterates `planned` only and IGNORES verdict keys that
    are not in it.  So a test attempt-0 reported FAIL that the probe never
    listed is silently dropped from the retry subset — the retry then never
    re-runs the actual failure and reports PASS: a FALSE GREEN.

    The module comment asserts the probe returns a SUPERSET of attempt-0's plan
    (`--workspace` ⊇ reify's `-p` selectors), but that holds only by coincidence
    of reify's current flags; an `--all-features` / `--all-targets` /
    `--cargo-profile` change on reify's side would break it with no signal.
    This is the one form of plan-incompleteness DF can detect IN-PROCESS, so the
    module's standing rule — never narrow on an incomplete plan — is ENFORCED
    rather than assumed.
    """
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path)

    # Everything from _FAIL_FAST_OUTPUT except the one test that FAILED.
    partial_plan = [t for t in _PROBED_PLANNED if t != 'crate-a beta::test_three']

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        return list(partial_plan)

    with (
        caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
        patch.object(mq, '_probe_nextest_planned', _fake_probe),
    ):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is None
    warnings = _mq_warnings(caplog)
    assert any('INCOMPLETE' in m for m in warnings), warnings
    assert any('crate-a beta::test_three' in m for m in warnings), warnings


@pytest.mark.asyncio
async def test_build_attempt0_payload_coverage_gate_ignores_passes_and_libtest_ids(
    tmp_path: Path,
) -> None:
    """The coverage gate fires on did-not-pass NEXTEST ids only.

    * a `pass` outside the plan is harmless — it is only ever dropped from a
      subset it would not have been in anyway;
    * `parse_per_test_results` also emits BARE libtest paths (no space), which
      `cargo nextest list` never produces, so they are not evidence of drift.

    Without both carve-outs the gate would refuse to narrow on every mixed
    nextest/libtest workspace — a silent, permanent loss of the capability.
    """
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path)

    # 'crate-a alpha::test_one' PASSED; 'bare::libtest_case' is a libtest id.
    output = (
        '    Starting 2 tests across 1 binaries\n'
        '        PASS [   0.130s] (  1/  2) crate-a alpha::test_one\n'
        '        FAIL [   0.044s] (  2/  2) crate-a beta::test_three\n'
        'test bare::libtest_case ... FAILED\n'
    )
    plan = ['crate-a beta::test_three', 'crate-a::integration test_end_to_end']

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        return list(plan)

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(tmp_path, _verify_result(output))

    assert payload is not None, (
        'a passed-but-unplanned id and a bare libtest id must not trip the gate'
    )
    assert payload.debug_planned == plan


@pytest.mark.asyncio
async def test_build_attempt0_payload_threads_verify_env_into_the_probe(
    tmp_path: Path,
) -> None:
    """attempt-0's own verify_env reaches the probe, so it models the same build."""
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path)
    seen: dict[str, object] = {}

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        seen['verify_env'] = verify_env
        return list(_PROBED_PLANNED)

    env = {'RUSTC_WRAPPER': '/usr/bin/sccache'}
    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT), verify_env=env,
        )

    assert payload is not None
    assert seen['verify_env'] == env


@pytest.mark.asyncio
async def test_build_attempt0_payload_none_when_probe_fails(tmp_path: Path) -> None:
    """The first profile's probe returning None -> None -> full verify.

    Never narrow on a partial plan: an unknown plan cannot distinguish
    'this test passed' from 'this test was never listed'.
    """
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path)

    async def _fake_probe(merge_wt, profile, *, timeout_secs, verify_env=None):
        return None

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is None


# ---------------------------------------------------------------------------
# _run_post_merge_verify wiring — the task's HEADLINE user-observable signal
# (task 3059, step 19).
#
# The shipped D2 built the retry env BEFORE the first dispatch, from a
# DF-invented sidecar nothing has ever written.  That was doubly wrong: the file
# never existed (so the capability was inert), and narrowing before attempt-0
# runs would have narrowed ATTEMPT-0 ITSELF — there is no prior result to narrow
# against at that point.
#
# The re-wire moves the narrowing INTO the classified-infra-transient retry
# branch, built from THIS call's own attempt-0 VerifyResult.  These tests pin
# that causal chain end to end:
#
#     attempt-0 dispatch (FULL) -> infra-transient red -> DF-owned payload
#     -> INV-3 tree-OID corroboration -> narrowed retry dispatch
#
# Driven on the LOCAL path (runner=None) with a patched VerifyRunnerPool that
# records the spec of every dispatch, so the assertions read the exact
# verify_env each attempt was dispatched with.
# ---------------------------------------------------------------------------


def _recording_pool(results: list):
    """A VerifyRunnerPool stand-in recording the spec of every dispatch.

    Returns ``(pool_cls, specs)``; ``specs[i]`` is the ``MergeVerifySpec`` the
    i-th dispatch was called with.  The last result is repeated if the loop
    dispatches more times than there are results.
    """
    specs: list = []

    class _Pool:
        def __init__(self, runners, **kwargs) -> None:
            self.runners = runners

        async def dispatch(self, merge_sha, spec, **kwargs):
            specs.append(spec)
            return results[min(len(specs) - 1, len(results) - 1)]

    return _Pool, specs


def _wiring_req(task_id: str, *, retry_failed_only: bool = True) -> MagicMock:
    """A MergeRequest double whose config keeps every unrelated branch inert."""
    req = MagicMock()
    req.task_id = task_id
    req.task_files = None
    req.module_configs = []
    req.retry_failed_only = retry_failed_only
    req.config.merge_verify_min_free_disk_bytes = 1024
    req.config.merge_verify_workspace = False
    req.config.merge_verify_breadth = 'scoped'
    req.config.verify_env = {}
    req.config.merge_verify_cold_command_timeout_secs = None
    req.config.verify_cold_command_timeout_secs = None
    # Lever C off: no dispatching-host scope derivation, no remote runner.
    req.config.enabled_verify_runners = []
    req.config.git.persistent_merge_worktree = False
    return req


def _infra_transient_attempt0(test_output: str):
    """attempt-0's own failing result: infra-transient category, NOT ENOSPC."""
    from orchestrator.verify import VerifyResult

    return VerifyResult(
        passed=False,
        test_output=test_output,
        lint_output='',
        type_output='',
        summary='attempt-0 infra-transient red',
        category='semaphore_timeout',
    )


def _passing_result():
    from orchestrator.verify import VerifyResult

    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='',
    )


_RETRY_ENV_KEYS = (
    'REIFY_VERIFY_RETRY_SCOPE',
    'REIFY_VERIFY_RETRY_TREE_OID',
    'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG',
    'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE',
    'REIFY_RUN_ALL_MEMBER_SUBSET',
    'REIFY_GUI_RETRY_SPECS',
)


def _no_retry_keys(env: dict) -> bool:
    return not any(k.startswith('REIFY_VERIFY_RETRY') for k in env)


async def _run_wiring(
    merge_wt: Path,
    *,
    req: MagicMock,
    results: list,
    git_ops: MagicMock | None = None,
    probe=None,
    sink: dict[str, str] | None = None,
    max_enospc: int = 1,
    max_narrowed: int = 2,
    narrowed_retries: dict[str, int] | None = None,
    enospc_retries: dict[str, int] | None = None,
):
    """Drive _run_post_merge_verify on the LOCAL path with a recording pool."""
    from orchestrator import merge_queue as mq

    pool_cls, specs = _recording_pool(results)
    git_ops = git_ops if git_ops is not None else _make_git_ops_mock()

    async def _default_probe(_merge_wt, _profile, *, timeout_secs, verify_env=None):
        return list(_PROBED_PLANNED)

    with (
        patch.object(mq, '_ensure_verify_disk_space', AsyncMock(return_value=None)),
        patch.object(mq, 'VerifyRunnerPool', pool_cls),
        patch.object(mq, '_classify_main_health_red', AsyncMock(return_value=None)),
        patch.object(
            mq, '_probe_nextest_planned', probe if probe is not None else _default_probe
        ),
    ):
        outcome = await mq._run_post_merge_verify(
            git_ops, req, merge_wt,
            timeouts={}, enospc_retries=enospc_retries if enospc_retries is not None else {},
            max_timeouts=2, max_enospc=max_enospc,
            max_narrowed=max_narrowed,
            narrowed_retries=narrowed_retries if narrowed_retries is not None else {},
            shadow_baseline_sink=sink,
        )
    return outcome, specs


def _sidecar_oid() -> str:
    return json.loads(_REIFY_SIDECAR_FIXTURE.read_text())['tree_oid']


def _corroborating_git_ops() -> MagicMock:
    """git_ops whose HEAD tree OID AGREES with the checked-in sidecar bytes."""
    git_ops = _make_git_ops_mock()
    git_ops.get_head_tree_hash = AsyncMock(return_value=_sidecar_oid())
    return git_ops


@pytest.mark.asyncio
async def test_attempt0_is_never_narrowed_by_its_own_retry_contract(
    tmp_path: Path,
) -> None:
    """(a) The FIRST dispatch carries NO REIFY_VERIFY_RETRY_* key.

    Attempt-0 IS the evidence the narrowing is derived from — narrowing it
    would scope the very run whose {did-not-pass} set the retry needs.  The
    shipped D2 built the env before this dispatch; this test is the tripwire
    that the narrowing point actually moved.
    """
    _place_real_sidecar(tmp_path)
    req = _wiring_req('t-wire-a')

    outcome, specs = await _run_wiring(
        tmp_path,
        req=req,
        git_ops=_corroborating_git_ops(),
        results=[_infra_transient_attempt0(_FAIL_FAST_OUTPUT), _passing_result()],
    )

    assert outcome is None, f'expected the retry to pass, got: {outcome!r}'
    assert len(specs) >= 1
    assert _no_retry_keys(specs[0].verify_env), (
        f'attempt-0 must be dispatched FULL, got: {specs[0].verify_env!r}'
    )


@pytest.mark.asyncio
async def test_narrowed_retry_dispatch_carries_the_did_not_pass_subset(
    tmp_path: Path,
) -> None:
    """(b) The SECOND dispatch carries the retry env and the {failed ∪ not-started} file."""
    _place_real_sidecar(tmp_path)
    req = _wiring_req('t-wire-b')

    outcome, specs = await _run_wiring(
        tmp_path,
        req=req,
        git_ops=_corroborating_git_ops(),
        results=[_infra_transient_attempt0(_FAIL_FAST_OUTPUT), _passing_result()],
    )

    assert outcome is None
    assert len(specs) == 2, f'expected attempt-0 + one narrowed retry, got {len(specs)}'
    env = specs[1].verify_env
    for key in _RETRY_ENV_KEYS:
        assert key in env, f'{key} missing from the narrowed retry env: {env!r}'
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == _sidecar_oid()

    debug_lines = Path(
        env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG']
    ).read_text().splitlines()
    # 'crate-a::integration test_end_to_end' (and 'crate-b gamma::test_one') are
    # in the PROBED plan but absent from _FAIL_FAST_OUTPUT's verdicts — nextest
    # fail-fast cancelled them, so they are 'not-started' and MUST re-run.
    # ('alpha::test_two' is NOT one of them: _FAIL_FAST_OUTPUT records it PASS.)
    assert 'test_end_to_end' in debug_lines, (
        f'a fail-fast-cancelled test must be in the retry subset: {debug_lines!r}'
    )
    assert 'beta::test_three' in debug_lines, (
        f'the FAILED test must be in the retry subset: {debug_lines!r}'
    )
    # 'alpha::test_one' PASSED in attempt-0 — narrowing it away is the point.
    assert 'alpha::test_one' not in debug_lines, (
        f'a passed test must NOT be re-run: {debug_lines!r}'
    )
    # A LATER profile is never narrowed on first-profile evidence.
    assert Path(
        env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE']
    ).read_text() == ''


@pytest.mark.asyncio
async def test_narrowed_true_unlocks_the_separate_narrowed_budget(
    tmp_path: Path,
) -> None:
    """(c) `narrowed` is observable through the BUDGET selection.

    With ``max_enospc=0`` the legacy budget affords no retry at all, so a
    dispatch beyond attempt-0 happens ONLY if this call actually narrowed and
    therefore switched to ``max_narrowed``.
    """
    _place_real_sidecar(tmp_path)
    req = _wiring_req('t-wire-c')

    outcome, specs = await _run_wiring(
        tmp_path,
        req=req,
        git_ops=_corroborating_git_ops(),
        results=[_infra_transient_attempt0(_FAIL_FAST_OUTPUT), _passing_result()],
        max_enospc=0,
        max_narrowed=2,
        narrowed_retries=(nr := {}),
        enospc_retries=(er := {}),
    )

    assert outcome is None
    assert len(specs) == 2, (
        f'narrowed=True must select max_narrowed=2, not max_enospc=0 — '
        f'got {len(specs)} dispatch(es)'
    )
    assert nr[req.task_id] == 1, f'the narrowed counter must be the one spent: {nr}'
    assert er == {}, f'the legacy ENOSPC counter must stay untouched: {er}'


@pytest.mark.asyncio
async def test_refused_narrowing_keeps_the_legacy_budget(tmp_path: Path) -> None:
    """(c, contrapositive) Narrowing REFUSED -> the legacy max_enospc=0 budget.

    Same scenario, but no sidecar: the payload cannot be built, ``narrowed``
    stays False, and the retry loop gets ZERO budget — proving the extra
    dispatch above was caused by the narrowing, not by max_narrowed alone.
    """
    req = _wiring_req('t-wire-c2')

    outcome, specs = await _run_wiring(
        tmp_path,  # no sidecar placed
        req=req,
        git_ops=_corroborating_git_ops(),
        results=[_infra_transient_attempt0(_FAIL_FAST_OUTPUT), _passing_result()],
        max_enospc=0,
        max_narrowed=2,
        narrowed_retries=(nr := {}),
    )

    assert outcome is not None, 'the un-retried infra red must surface as an outcome'
    assert len(specs) == 1, (
        f'a refused narrowing must NOT unlock max_narrowed, got {len(specs)} dispatches'
    )
    assert nr == {}, f'the narrowed counter must stay untouched: {nr}'


@pytest.mark.asyncio
async def test_a_non_narrowing_payload_keeps_the_legacy_budget(tmp_path: Path) -> None:
    """(c, second contrapositive) A payload that narrows NOTHING is not narrowed.

    The sidecar is present, the tree corroborates and the probe succeeds — every
    gate the earlier tests exercise PASSES.  What fails is the cost premise:
    attempt-0 died before a single test line was emitted (a compile-gate red),
    so `parse_per_test_results` returns {} and the "subset" is the entire probed
    plan.  reify would run those profiles in FULL.

    Charging that to `max_narrowed=2` would make the merge lane pay TWO full
    re-verifies where the legacy `max_enospc=1` paid one — a real worst-case
    regression for exactly the infra-transient class this branch handles.  With
    max_enospc=0 the legacy budget affords nothing, so a second dispatch here
    would BE the regression.
    """
    _place_real_sidecar(tmp_path)
    req = _wiring_req('t-wire-c3')
    sink: dict[str, str] = {}

    outcome, specs = await _run_wiring(
        tmp_path,
        req=req,
        git_ops=_corroborating_git_ops(),
        # A compile-gate red: no per-test lines at all -> verdicts == {}.
        results=[
            _infra_transient_attempt0('error: could not compile `crate-a`\n'),
            _passing_result(),
        ],
        sink=sink,
        max_enospc=0,
        max_narrowed=2,
        narrowed_retries=(nr := {}),
    )

    assert outcome is not None, 'the un-retried infra red must surface as an outcome'
    assert len(specs) == 1, (
        f'a retry that narrows nothing must NOT unlock max_narrowed, '
        f'got {len(specs)} dispatches'
    )
    assert nr == {}, f'the narrowed counter must stay untouched: {nr}'
    assert sink == {}, 'a refused narrowing must not seed the shadow baseline'


@pytest.mark.asyncio
async def test_shadow_baseline_sink_seeded_only_on_the_narrowed_path(
    tmp_path: Path,
) -> None:
    """(d) D4/§5.4 phantom-divergence guard: attempt-0 verdicts reach the sink.

    A narrowed retry re-runs ONLY the {did-not-pass} subset, so its warm output
    alone omits every attempt-0-passed test.  Without this seed a from-scratch
    cold shadow compare would flag them all ``only_cold``.
    """
    _place_real_sidecar(tmp_path)
    req = _wiring_req('t-wire-d')
    sink: dict[str, str] = {}

    outcome, specs = await _run_wiring(
        tmp_path,
        req=req,
        git_ops=_corroborating_git_ops(),
        results=[_infra_transient_attempt0(_FAIL_FAST_OUTPUT), _passing_result()],
        sink=sink,
    )

    assert outcome is None
    assert len(specs) == 2
    assert sink['crate-a alpha::test_one'] == 'pass'
    assert sink['crate-a beta::test_three'] == 'fail'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('case', 'place_sidecar', 'probe_none', 'head_oid'),
    [
        ('probe_failed', True, True, None),
        ('sidecar_absent', False, False, None),
        ('tree_oid_disagrees', True, False, 'a' * 40),
    ],
)
async def test_failsafe_routes_leave_the_retry_dispatch_full(
    tmp_path: Path, case: str, place_sidecar: bool, probe_none: bool, head_oid: str | None,
) -> None:
    """(e) Every fail-safe route leaves the retry FULL and the sink EMPTY.

    A partial plan, an absent sidecar and a disagreeing tree are each an
    ABSENCE of corroboration — never a licence to narrow.  The retry still
    happens (on the legacy budget); it is simply not scoped.
    """
    if place_sidecar:
        _place_real_sidecar(tmp_path)
    req = _wiring_req(f't-wire-e-{case}')
    sink: dict[str, str] = {}

    async def _none_probe(_merge_wt, _profile, *, timeout_secs, verify_env=None):
        return None

    git_ops = _corroborating_git_ops()
    if head_oid is not None:
        git_ops.get_head_tree_hash = AsyncMock(return_value=head_oid)

    outcome, specs = await _run_wiring(
        tmp_path,
        req=req,
        git_ops=git_ops,
        probe=_none_probe if probe_none else None,
        results=[_infra_transient_attempt0(_FAIL_FAST_OUTPUT), _passing_result()],
        sink=sink,
        max_enospc=1,
        max_narrowed=2,
    )

    assert outcome is None
    assert len(specs) == 2, f'the legacy full retry must still run ({case})'
    assert _no_retry_keys(specs[1].verify_env), (
        f'{case}: a fail-safe route must not narrow: {specs[1].verify_env!r}'
    )
    assert sink == {}, f'{case}: an uncorroborated payload must not seed the baseline'


@pytest.mark.asyncio
async def test_flag_off_is_byte_identical_to_the_legacy_path(tmp_path: Path) -> None:
    """(f) D1's strict no-op: retry_failed_only=False never narrows.

    Even with a valid sidecar on disk and a corroborating tree OID, the flag
    off leaves EVERY dispatch full and the sink untouched.
    """
    _place_real_sidecar(tmp_path)
    req = _wiring_req('t-wire-f', retry_failed_only=False)
    sink: dict[str, str] = {}
    probe_calls: list[str] = []

    async def _counting_probe(_merge_wt, profile, *, timeout_secs, verify_env=None):
        probe_calls.append(profile)
        return list(_PROBED_PLANNED)

    outcome, specs = await _run_wiring(
        tmp_path,
        req=req,
        git_ops=_corroborating_git_ops(),
        probe=_counting_probe,
        results=[_infra_transient_attempt0(_FAIL_FAST_OUTPUT), _passing_result()],
        sink=sink,
    )

    assert outcome is None
    assert len(specs) == 2
    assert all(_no_retry_keys(s.verify_env) for s in specs), (
        'the flag-off path must be byte-identical to legacy'
    )
    assert sink == {}
    assert probe_calls == [], 'the flag-off path must not even probe'
