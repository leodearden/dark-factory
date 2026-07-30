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


def test_build_retry_verify_env_writes_filter_files_and_env(tmp_path: Path) -> None:
    """_build_retry_verify_env writes per-profile filter files + the REIFY_* env.

    The nextest subsets (potentially thousands of ids) ship as newline filter
    FILES; the small run_all-member / gui-spec lists ship as comma-delimited
    env VALUES; tree OID + scope ship as env values.
    """
    from orchestrator.merge_queue import _build_retry_verify_env

    debug = ['c a::y', 'c a::z']
    release = ['c b::q']
    env = _build_retry_verify_env(
        nextest_subset_debug=debug,
        nextest_subset_release=release,
        run_all_members=['mem1', 'mem2'],
        gui_specs=['ui/spec_a.ts'],
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
    assert debug_path.read_text() == 'c a::y\nc a::z'
    assert release_path.read_text() == 'c b::q'

    # (3) run_all members / gui specs ship comma-delimited.
    assert env['REIFY_RUN_ALL_MEMBER_SUBSET'] == 'mem1,mem2'
    assert env['REIFY_GUI_RETRY_SPECS'] == 'ui/spec_a.ts'

    # (4) tree OID + scope.
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == 'deadbeef'
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'


def test_build_retry_verify_env_empty_subsets_still_write_files(tmp_path: Path) -> None:
    """Empty nextest subsets still write (empty) filter files and set env keys.

    The contract is deterministic: reify's verify.sh always finds the filter
    files at the advertised paths, even when a profile has nothing to retry.
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

    async def _fake(argv, *, cwd, timeout_secs):
        seen['argv'] = list(argv)
        seen['cwd'] = cwd
        seen['timeout_secs'] = timeout_secs
        return 0, _NEXTEST_LIST_FIXTURE.read_text()

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

    async def _fake(argv, *, cwd, timeout_secs):
        return 0, raw

    with patch.object(mq, '_run_probe_cmd', _fake):
        planned = await mq._probe_nextest_planned(tmp_path, 'debug', timeout_secs=5.0)

    assert planned == parse_nextest_list_planned(raw)
    assert planned  # the fixture is non-empty, so this is a real assertion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('label', 'result', 'exc'),
    [
        ('non-zero exit', (101, '{"rust-suites": {}}'), None),
        ('empty stdout', (0, ''), None),
        ('whitespace-only stdout', (0, '   \n'), None),
        ('unparseable stdout', (0, 'error: could not compile'), None),
        ('timeout', None, TimeoutError()),
        ('cargo absent', None, FileNotFoundError('cargo')),
        ('spawn OSError', None, OSError('fork failed')),
    ],
)
async def test_probe_nextest_planned_failure_modes_return_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    label: str,
    result: tuple[int, str] | None,
    exc: BaseException | None,
) -> None:
    """Every failure mode returns None and WARNs, naming the profile and reason.

    None is NEVER an empty plan — it routes the caller to a FULL verify.  A
    silent empty plan would narrow the retry to nothing: a FALSE GREEN.
    """
    from orchestrator import merge_queue as mq

    async def _fake(argv, *, cwd, timeout_secs):
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
async def test_probe_nextest_planned_timeout_is_bounded_by_a_named_constant() -> None:
    """A module constant bounds the probe so it can never wedge the merge lane."""
    from orchestrator.merge_queue import _NEXTEST_LIST_PROBE_TIMEOUT_SECS

    assert isinstance(_NEXTEST_LIST_PROBE_TIMEOUT_SECS, (int, float))
    assert 0 < _NEXTEST_LIST_PROBE_TIMEOUT_SECS <= 900


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

    async def _fake_probe(merge_wt, profile, *, timeout_secs):
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

    async def _fake_probe(merge_wt, profile, *, timeout_secs):
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

    async def _fake_probe(merge_wt, profile, *, timeout_secs):
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

    async def _fake_probe(merge_wt, profile, *, timeout_secs):
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

    async def _fake_probe(merge_wt, profile, *, timeout_secs):
        return list(_PROBED_PLANNED)

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is None


@pytest.mark.asyncio
async def test_build_attempt0_payload_none_when_probe_fails(tmp_path: Path) -> None:
    """The first profile's probe returning None -> None -> full verify.

    Never narrow on a partial plan: an unknown plan cannot distinguish
    'this test passed' from 'this test was never listed'.
    """
    from orchestrator import merge_queue as mq

    _place_real_sidecar(tmp_path)

    async def _fake_probe(merge_wt, profile, *, timeout_secs):
        return None

    with patch.object(mq, '_probe_nextest_planned', _fake_probe):
        payload = await mq._build_attempt0_payload(
            tmp_path, _verify_result(_FAIL_FAST_OUTPUT)
        )

    assert payload is None
