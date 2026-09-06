"""Guard tests: a test run can never falsify a REAL deploy clock (task 3797).

``scripts/restart-all-orchestrators.sh`` resolves its ``CLOCK_FILE`` from
``$ORCH_FLEET_DEPLOY_CLOCK``, defaulting to
``$REPO_DIR/data/orchestrator/last_redeploy_orchestrator.json`` — the live
checkout the script sits in — and stamps it on its verified-fresh exit-0 path.
``scripts/tests/test_restart_all_orchestrators.py`` drove that path against a
fake ``systemctl`` without setting the env var, so an ordinary green test run
wrote a REAL "the fleet just redeployed" stamp.  Nothing distinguishes it from
a genuine one: ``scripts/orchestrator-watchdog.py`` reads the file and SKIPS
its staleness pass for ``ORCH_RESTART_MIN_INTERVAL_SECS`` (28800s = 8h), so the
fleet-staleness backstop was silently disarmed for the rest of the day.

The fix for that one suite is a conftest redirect.  This module covers the
suite-wide, opt-out-impossible SECOND layer in the root ``df_pytest_isolation``
module — the one that catches the next spawner that forgets:

* two pure helpers, ``deploy_clock_snapshot`` / ``deploy_clock_violation_reason``,
  directly testable without a nested pytest run; and
* a session-scoped autouse fixture that snapshots the protected clocks around
  the whole session and fails loudly if any of them moved.

Shaped like ``test_basetemp_git_isolation.py``, the module covering this repo's
other suite-wide isolation defence, so the two read as one family.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import df_pytest_isolation  # noqa: E402
from df_pytest_isolation import (  # noqa: E402
    CLOCK_PROVENANCE_SESSION_KEY,
    CLOCK_PROVENANCE_SOURCE_KEY,
    PROTECTED_DEPLOY_CLOCK_RELPATHS,
    PYTEST_SESSION_TOKEN_ENV,
    clock_stamp_provenance,
    deploy_clock_change_report,
    deploy_clock_guard_roots,
    deploy_clock_snapshot,
    deploy_clock_violation_reason,
    fixture_marker,
)

# NOT `from df_pytest_isolation import _df_deploy_clocks_unwritten`. Importing a
# fixture into a TEST module binds it as a module-scoped fixture that SHADOWS
# the conftest's — which would make the liveness test below resolve its own
# import and pass even with the conftest wiring removed, i.e. exactly the dead
# defence it exists to detect. Reach it through the module instead.
_GUARD_NAME = '_df_deploy_clocks_unwritten'

_FLEET_RELPATH = 'data/orchestrator/last_redeploy_orchestrator.json'
_FM_RELPATH = 'data/fused-memory/last_redeploy_fused_memory.json'


def _write(root: Path, relpath: str, body: bytes) -> Path:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return target


class TestProtectedRelpaths:
    """The list of clocks the guard watches."""

    def test_it_is_a_non_empty_tuple(self) -> None:
        """A tuple, not a list: a module-level mutable would let one suite
        quietly shrink the protected set for every later one in the process.
        """
        assert isinstance(PROTECTED_DEPLOY_CLOCK_RELPATHS, tuple)
        assert PROTECTED_DEPLOY_CLOCK_RELPATHS

    def test_it_covers_the_orchestrator_fleet_clock(self) -> None:
        assert _FLEET_RELPATH in PROTECTED_DEPLOY_CLOCK_RELPATHS

    def test_it_covers_the_fused_memory_deploy_clock(self) -> None:
        """The identical defect class, not an adjacent one.

        Same watchdog, same min-interval semantics (FM_RESTART_MIN_INTERVAL_SECS,
        8h), and it is stamped from a chained ``--stamp-fm-deploy-clock``
        SUBPROCESS — the same shape (a spawned process resolving a path from the
        environment) that produced the fleet-clock bug.
        """
        assert _FM_RELPATH in PROTECTED_DEPLOY_CLOCK_RELPATHS

    def test_the_relpaths_are_relative(self) -> None:
        """They are joined onto a root by the snapshot; an absolute entry would
        silently escape that root and point at the real checkout.
        """
        for relpath in PROTECTED_DEPLOY_CLOCK_RELPATHS:
            assert not Path(relpath).is_absolute(), relpath


# The gitignored worktree-root vocabulary, .gitignore:15-17.  Spelled out here
# rather than imported so the test pins the values independently of the module.
_WORKTREE_ROOTS = ('.worktrees', '.worktrees-orphaned', '.eval-worktrees')


class TestDeployClockGuardRoots:
    """WHICH checkouts the guard watches — one root is not enough."""

    def test_a_plain_checkout_yields_only_itself(self, tmp_path: Path) -> None:
        """No ``.worktrees`` parent (the main checkout's own shape) — exactly one
        root, so a main-checkout run does not snapshot the same file twice.
        """
        root = tmp_path / 'dark-factory'
        root.mkdir()

        assert deploy_clock_guard_roots(root) == (root.resolve(),)

    @pytest.mark.parametrize('worktree_dir', _WORKTREE_ROOTS)
    def test_a_worktree_also_yields_the_enclosing_main_checkout(
        self, tmp_path: Path, worktree_dir: str,
    ) -> None:
        """The whole point of this helper.

        ``scripts/orchestrator-watchdog.py`` HARDCODES ``REPO_DIR =
        "/home/leo/src/dark-factory"`` (line 96), so its ``FM_DEPLOY_CLOCK_PATH``
        default and the fleet restart script it spawns resolve into the MAIN
        checkout no matter which worktree the suite runs from.  A guard rooted
        only at the worktree would report all-clear while the REAL production
        clock was falsified — the fused-memory entry in the protected set would
        be watching a path nothing under a worktree can even write.
        """
        main = tmp_path / 'dark-factory'
        worktree = main / worktree_dir / '3797'
        worktree.mkdir(parents=True)

        assert deploy_clock_guard_roots(worktree) == (worktree.resolve(), main.resolve())

    def test_the_run_s_own_checkout_comes_first(self, tmp_path: Path) -> None:
        """Ordering is load-bearing for the failure MESSAGE: when a test falsifies
        both, the reader should be pointed at the checkout they are working in.
        """
        main = tmp_path / 'dark-factory'
        worktree = main / '.worktrees' / '3797'
        worktree.mkdir(parents=True)

        assert deploy_clock_guard_roots(worktree)[0] == worktree.resolve()

    def test_the_roots_are_deduped(self, tmp_path: Path) -> None:
        """Snapshotting a root twice would compare it against itself twice and
        double every failure message; it must be impossible by construction.
        """
        worktree = tmp_path / 'dark-factory' / '.worktrees' / '3797'
        worktree.mkdir(parents=True)

        for root in (worktree, tmp_path / 'plain'):
            roots = deploy_clock_guard_roots(root)
            assert len(roots) == len(set(roots)), roots

    def test_a_lookalike_directory_name_is_not_a_worktree(self, tmp_path: Path) -> None:
        """Matching is on the exact parent NAME, never a substring: a directory
        merely containing the word is an ordinary, unrelated checkout, and
        promoting its parent to a guarded root would watch some stranger's tree.
        """
        root = tmp_path / 'my.worktrees-backup' / 'dark-factory'
        root.mkdir(parents=True)

        assert deploy_clock_guard_roots(root) == (root.resolve(),)

    def test_the_roots_are_absolute(self, tmp_path: Path) -> None:
        """The snapshot joins relpaths onto these; a relative root would resolve
        against whatever cwd the session happened to leave behind.
        """
        (tmp_path / 'dark-factory').mkdir()

        for root in deploy_clock_guard_roots(Path('dark-factory')):
            assert root.is_absolute(), root

    def test_this_checkout_is_guarded_in_this_run(self) -> None:
        """Liveness, not just shape: whatever checkout this file lives in — main
        or worktree — must be among the roots the guard would snapshot.
        """
        assert REPO_ROOT.resolve() in deploy_clock_guard_roots(REPO_ROOT)


class TestDeployClockSnapshot:
    """What the guard records at session start."""

    def test_absent_clocks_snapshot_as_none(self, tmp_path: Path) -> None:
        """Absence is a first-class value, not an omitted key.

        The common case IS absence — a fresh worktree has no ``data/`` dir at
        all — so "file created during the run", the exact 3797 shape, is only
        detectable if absence is recorded rather than skipped.
        """
        snapshot = deploy_clock_snapshot(tmp_path)

        assert set(snapshot) == set(PROTECTED_DEPLOY_CLOCK_RELPATHS)
        assert all(value is None for value in snapshot.values())

    def test_a_missing_data_dir_does_not_raise(self, tmp_path: Path) -> None:
        """No ``data/`` at all is the ordinary state of a task worktree."""
        assert not (tmp_path / 'data').exists()

        deploy_clock_snapshot(tmp_path)  # must not raise

    def test_a_present_clock_snapshots_as_bytes_and_mtime(self, tmp_path: Path) -> None:
        clock = _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        snapshot = deploy_clock_snapshot(tmp_path)

        assert snapshot[_FLEET_RELPATH] == (clock.read_bytes(), clock.stat().st_mtime_ns)

    def test_every_protected_path_is_keyed_independently(self, tmp_path: Path) -> None:
        _write(tmp_path, _FLEET_RELPATH, b'fleet')

        snapshot = deploy_clock_snapshot(tmp_path)

        assert snapshot[_FLEET_RELPATH] is not None
        assert snapshot[_FM_RELPATH] is None


def _entry(body: bytes, mtime_ns: int = 1_700_000_000_000_000_000) -> tuple[bytes, int]:
    """A snapshot entry built from a literal body — no I/O, no real clock."""
    return (body, mtime_ns)


# A genuine machine-operated stamp: a provenance-aware writer with no pytest
# ancestor, hence an EMPTY session token. This is the one shape the guard is
# allowed to forgive, so it is spelled out once and reused.
_EXTERNAL_BODY = (
    b'{"ts": 1787849070, "iso": "2026-08-28T09:24:30+00:00", '
    b'"source": "restart-all-orchestrators.sh", "pytest_session": ""}\n'
)
# The pre-4823 shape. MUST stay unattributable: a writer that has not been
# taught provenance cannot buy itself an exemption by omission.
_LEGACY_BODY = b'{"ts": 1787849070, "iso": "2026-08-28T09:24:30+00:00"}\n'


class TestClockStampProvenance:
    """Who wrote the stamp — the whole discriminator, parsed from the body alone.

    Task 4823. The pre-existing guard could see only that the bytes moved, so a
    REAL fleet redeploy straddling a suite was indistinguishable from a test
    falsifying the clock; it failed closed and blocked four innocent branches.
    This parser is the attribution half of the fix. Every case here is built
    from a LITERAL body so the parser's contract is pinned independently of any
    writer, in either direction: a writer that stops emitting provenance must
    fail these, not silently degrade the guard.
    """

    def test_the_env_var_and_key_names_are_the_cross_tier_contract(self) -> None:
        """Named constants, never inlined literals.

        Four tiers must agree on these spellings and none can import another's
        (the module docstring's stdlib+pytest import constraint): the two
        production writers emit them, this parser reads them, and the writer
        tests assert against THESE objects rather than against strings — which
        is what makes those tests drift pins rather than tautologies.
        """
        assert PYTEST_SESSION_TOKEN_ENV == 'DF_PYTEST_SESSION_TOKEN'
        assert CLOCK_PROVENANCE_SOURCE_KEY == 'source'
        assert CLOCK_PROVENANCE_SESSION_KEY == 'pytest_session'

    def test_an_absent_file_has_no_provenance(self) -> None:
        """``None`` in, ``None`` out — a DELETED clock has no body to attribute."""
        assert clock_stamp_provenance(None) is None

    @pytest.mark.parametrize('body', [b'not json', b'{"ts": 1', b'', b'\xff\xfe'])
    def test_an_unparseable_body_is_unattributable_and_never_raises(
        self, body: bytes,
    ) -> None:
        """A parse error must fail CLOSED, not propagate.

        This runs in a session-teardown fixture: an exception here would replace
        the guard's own message with a traceback about JSON, hiding whichever
        clock actually moved.
        """
        assert clock_stamp_provenance(_entry(body)) is None

    @pytest.mark.parametrize('body', [b'[1, 2]', b'"x"', b'null', b'3'])
    def test_a_non_object_body_is_unattributable(self, body: bytes) -> None:
        """Valid JSON is not enough — provenance lives in named keys."""
        assert clock_stamp_provenance(_entry(body)) is None

    def test_the_legacy_ts_iso_body_is_unattributable(self) -> None:
        """The pre-4823 shape stays a violation.

        Any stamp written by a writer that predates (or forgets) provenance is
        indistinguishable from a test's, so it must keep failing the run exactly
        as it does today. This is what makes the change safe to land with no
        coordinated writer rollout.
        """
        assert clock_stamp_provenance(_entry(_LEGACY_BODY)) is None

    def test_a_half_provenance_body_is_unattributable(self) -> None:
        """``pytest_session`` alone does not clear a stamp.

        A writer emitting one key of the pair is a broken writer, and trusting
        it would let a partially-migrated writer grant itself the exemption.
        """
        body = b'{"ts": 1, "iso": "x", "pytest_session": ""}'

        assert clock_stamp_provenance(_entry(body)) is None

    def test_a_source_without_a_session_key_is_unattributable(self) -> None:
        """The mirror case: ``source`` is triage prose, never the discriminator."""
        body = b'{"ts": 1, "iso": "x", "source": "restart-all-orchestrators.sh"}'

        assert clock_stamp_provenance(_entry(body)) is None

    def test_a_full_external_stamp_parses_to_its_two_strings(self) -> None:
        provenance = clock_stamp_provenance(_entry(_EXTERNAL_BODY))

        assert provenance == {
            CLOCK_PROVENANCE_SOURCE_KEY: 'restart-all-orchestrators.sh',
            CLOCK_PROVENANCE_SESSION_KEY: '',
        }

    def test_a_session_token_is_returned_verbatim(self) -> None:
        """Verbatim, because the caller compares it for EQUALITY with its own.

        Any normalisation here (case, strip, truncation) would silently turn a
        foreign token into a match or a match into a miss.
        """
        token = '0f1e2d3c4b5a69788796a5b4c3d2e1f0'
        body = (
            b'{"ts": 1, "iso": "x", "source": "orchestrator-watchdog.py", '
            b'"pytest_session": "' + token.encode() + b'"}'
        )

        provenance = clock_stamp_provenance(_entry(body))

        assert provenance is not None
        assert provenance[CLOCK_PROVENANCE_SESSION_KEY] == token
        assert provenance[CLOCK_PROVENANCE_SOURCE_KEY] == 'orchestrator-watchdog.py'

    @pytest.mark.parametrize(
        'body',
        [
            b'{"ts": 1, "source": 123, "pytest_session": ""}',
            b'{"ts": 1, "source": null, "pytest_session": ""}',
            b'{"ts": 1, "source": "x", "pytest_session": 123}',
            b'{"ts": 1, "source": "x", "pytest_session": null}',
            b'{"ts": 1, "source": "x", "pytest_session": []}',
        ],
    )
    def test_a_wrong_typed_provenance_field_is_unattributable(self, body: bytes) -> None:
        """No coercion. A non-string field is a broken writer, not provenance.

        Coercing ``None`` to ``''`` would be actively dangerous: the empty
        string is the POSITIVE assertion "no pytest session was an ancestor of
        this write", i.e. the one value that forgives a change.
        """
        assert clock_stamp_provenance(_entry(body)) is None


class TestDeployClockViolationReason:
    """What counts as a falsified clock."""

    def test_unchanged_absent_is_clean(self, tmp_path: Path) -> None:
        before = deploy_clock_snapshot(tmp_path)

        assert deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path)) is None

    def test_unchanged_present_is_clean(self, tmp_path: Path) -> None:
        """A suite that merely READS the clock must stay green — the guard
        watches for writes, not for access.
        """
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)

        assert deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path)) is None

    def test_a_created_clock_is_a_violation(self, tmp_path: Path) -> None:
        """The exact 3797 shape: absent before, stamped by a test, present after."""
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1786033966, "iso": "..."}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_changed_bytes_are_a_violation(self, tmp_path: Path) -> None:
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 2, "iso": "y"}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_identical_bytes_with_a_moved_mtime_are_a_violation(self, tmp_path: Path) -> None:
        """NOT pedantry — the case a bytes-only guard would silently miss.

        ``stamp_fleet_deploy_clock`` writes ``{"ts": <int seconds>, "iso": ...}``
        at ONE-SECOND resolution, so two stamps landing inside the same second
        are byte-identical.  A test suite restamping in a tight loop produces
        exactly that, and mtime is the only signal left.
        """
        clock = _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)
        stat = clock.stat()
        os.utime(clock, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        after = deploy_clock_snapshot(tmp_path)
        before_entry, after_entry = before[_FLEET_RELPATH], after[_FLEET_RELPATH]
        # Bound before subscripting: the snapshot value is `tuple | None`, and
        # a None on either side would mean the clock vanished rather than being
        # restamped — a different violation, and one that would make the
        # bytes-identical premise below vacuous rather than false.
        assert before_entry is not None and after_entry is not None
        assert after_entry[0] == before_entry[0], 'bytes must be identical'

        reason = deploy_clock_violation_reason(before, after)

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_a_deleted_clock_is_a_violation(self, tmp_path: Path) -> None:
        """Deletion is as damaging as a stamp, in the other direction: it makes
        the watchdog see "never redeployed" and lose the real last-deploy time.
        """
        clock = _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        before = deploy_clock_snapshot(tmp_path)
        clock.unlink()

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FLEET_RELPATH in reason

    def test_the_fm_clock_is_watched_too(self, tmp_path: Path) -> None:
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FM_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert _FM_RELPATH in reason

    def test_the_reason_names_the_remedy_and_the_benign_alternative(
        self, tmp_path: Path,
    ) -> None:
        """The message has to be actionable at 3am by whoever sees it fail.

        Both readings must be spelled out: a test that forgot the env var (the
        common case, remedy = point it at a tmp file) AND a genuine concurrent
        fleet redeploy in a machine-operated checkout (not a bug at all).  A
        bare "the clock changed" invites the reader to assume whichever one
        they thought of first.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert 'ORCH_FLEET_DEPLOY_CLOCK' in reason
        assert 'redeploy' in reason.lower()

    def test_the_reason_prints_the_observed_bodies(self, tmp_path: Path) -> None:
        """Triage must be possible from the failure OUTPUT alone.

        Telling the reader to "compare the {ts, iso} body against the deploy you
        expect" — the step that separates a genuine concurrent redeploy from a
        test bug — is useless if the bodies are not printed: by the time anyone
        reads the ERROR the file may have moved again, so re-reading it is not the
        same observation.  Both readings are already in hand in the snapshots.
        """
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 111, "iso": "2026-08-06T00:00:00+00:00"}')
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 222, "iso": "2026-08-07T00:00:00+00:00"}')
        after = deploy_clock_snapshot(tmp_path)

        reason = deploy_clock_violation_reason(before, after)

        assert reason is not None
        assert '"ts": 111' in reason, reason
        assert '"ts": 222' in reason, reason
        before_entry, after_entry = before[_FLEET_RELPATH], after[_FLEET_RELPATH]
        # Bound before subscripting: a snapshot value is `tuple | None` by design.
        assert before_entry is not None and after_entry is not None
        assert str(before_entry[1]) in reason, 'the before mtime_ns must be printed'
        assert str(after_entry[1]) in reason, 'the after mtime_ns must be printed'

    def test_the_reason_reports_absence_rather_than_a_bare_none(
        self, tmp_path: Path,
    ) -> None:
        """The created case (the 3797 shape) has no before body to print, and
        ``None`` would read as "the guard failed to look" rather than "the file
        did not exist".
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert 'absent' in reason, reason

    def test_a_root_names_the_absolute_file(self, tmp_path: Path) -> None:
        """A run guards MORE THAN ONE checkout (see deploy_clock_guard_roots), so
        a bare relpath leaves the reader unable to tell which one was falsified.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}')
        after = deploy_clock_snapshot(tmp_path)

        reason = deploy_clock_violation_reason(before, after, root=tmp_path)

        assert reason is not None
        assert str(tmp_path / _FLEET_RELPATH) in reason, reason

    def test_the_detector_is_not_vacuous_end_to_end(self, tmp_path: Path) -> None:
        """Self-test: snapshot and detector agree on a dict shape that WORKS.

        Every case above builds its snapshots through ``deploy_clock_snapshot``
        precisely so the two helpers cannot drift into mutually incompatible
        shapes while each stays green on hand-built dicts.  A guard that
        silently matches nothing is worse than no guard, because it reads as
        coverage.  This asserts both directions in one test: clean stays clean,
        and a single real write flips it.
        """
        clean = deploy_clock_snapshot(tmp_path)
        assert deploy_clock_violation_reason(clean, deploy_clock_snapshot(tmp_path)) is None

        _write(tmp_path, _FLEET_RELPATH, b'stamped by a test\n')

        assert deploy_clock_violation_reason(clean, deploy_clock_snapshot(tmp_path)) is not None


# A plausible per-session token: 32 lowercase hex, the shape uuid4().hex has.
_THIS_SESSION = 'aa11bb22cc33dd44ee55ff6677889900'
_OTHER_SESSION = '00998877ff66ee55dd44cc33bb22aa11'
_FLEET_SOURCE = 'restart-all-orchestrators.sh'
_WATCHDOG_SOURCE = 'orchestrator-watchdog.py'


def _stamp(*, token: str, source: str = _FLEET_SOURCE, ts: int = 1787849070) -> bytes:
    """A provenance-bearing clock body, as the production writers emit it.

    Built with json.dumps rather than a format string so a key-name change in
    the module constants cannot leave this helper writing the old spelling
    while still looking right.
    """
    return (
        json.dumps(
            {
                'ts': ts,
                'iso': '2026-08-28T09:24:30+00:00',
                CLOCK_PROVENANCE_SOURCE_KEY: source,
                CLOCK_PROVENANCE_SESSION_KEY: token,
            }
        ).encode()
        + b'\n'
    )


class TestDeployClockChangeReport:
    """The attributing entry point: WHAT changed, and WHO changed it (task 4823).

    ``deploy_clock_violation_reason`` could only see that the bytes moved, so a
    REAL fleet redeploy straddling a suite was indistinguishable from a test
    falsifying the clock. It failed closed, which is the right default and the
    wrong answer often enough to have blocked four innocent branches. This
    function keeps that default for every unattributable change and downgrades
    exactly one case: a provenance-bearing stamp written with no pytest
    ancestor.
    """

    def test_an_unchanged_absent_clock_reports_nothing(self, tmp_path: Path) -> None:
        before = deploy_clock_snapshot(tmp_path)

        assert deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        ) is None

    def test_an_unchanged_present_clock_reports_nothing(self, tmp_path: Path) -> None:
        """Reading a clock is not writing it — provenance never even gets parsed."""
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=''))
        before = deploy_clock_snapshot(tmp_path)

        assert deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        ) is None

    def test_a_stamp_carrying_this_sessions_token_is_falsified(
        self, tmp_path: Path,
    ) -> None:
        """The 3797 defect, now with POSITIVE proof instead of an inference.

        The write provably descends from this pytest session, so the message
        may say so outright rather than hedging between "a test did it" and "a
        real redeploy did it".
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=_THIS_SESSION))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        verdict, message = report
        assert verdict == 'falsified'
        assert 'falsified a REAL deploy clock' in message, message
        assert _FLEET_SOURCE in message, 'the writer must be named for triage'
        assert 'this run' in message, message

    def test_an_external_stamp_is_a_redeploy_not_a_falsification(
        self, tmp_path: Path,
    ) -> None:
        """THE FIX. An empty token is the positive statement "no pytest ancestor".

        Only a provenance-AWARE writer can make that statement, so this cannot
        be reached by a pre-4823 or half-migrated writer.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=''))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        verdict, message = report
        assert verdict == 'external_redeploy'
        assert 'falsified a REAL deploy clock' not in message, (
            'the benign message must not carry the accusing string — the nested '
            'end-to-end test and the failure-path tests both key on it'
        )
        assert 'not at fault' in message.lower(), message
        assert _FLEET_SOURCE in message, 'the writer must be named for triage'

    def test_a_foreign_session_token_still_fails(self, tmp_path: Path) -> None:
        """Deliberately conservative: this guard never absolves another session.

        A token that is neither empty nor ours means some OTHER pytest session
        wrote the shared main-checkout clock. That session's own guard sees its
        own token and fails, so the signal is never lost — and this run must not
        become the arbiter of another run's bug on the strength of a token it
        cannot verify.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=_OTHER_SESSION))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        verdict, message = report
        assert verdict == 'falsified'
        assert 'another pytest session' in message.lower(), message

    def test_a_legacy_provenance_free_stamp_still_fails(self, tmp_path: Path) -> None:
        """Today's behaviour, preserved: no provenance means no exemption."""
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1787849070, "iso": "..."}\n')

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        assert report[0] == 'falsified'

    def test_a_created_external_stamp_is_a_redeploy(self, tmp_path: Path) -> None:
        """CREATED-from-absent is the 3797 SHAPE but not necessarily its cause:
        a machine-operated checkout with no ``data/`` yet gets its first real
        stamp exactly this way.
        """
        before = deploy_clock_snapshot(tmp_path)
        assert before[_FLEET_RELPATH] is None
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=''))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        assert report[0] == 'external_redeploy'

    def test_a_created_provenance_free_stamp_fails(self, tmp_path: Path) -> None:
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        assert report[0] == 'falsified'

    def test_a_deleted_clock_always_fails(self, tmp_path: Path) -> None:
        """There is no body left to attribute, and nothing legitimate deletes a
        live clock — the watchdog would read "never redeployed" and lose the
        real last-deploy time.
        """
        clock = _write(tmp_path, _FLEET_RELPATH, _stamp(token=''))
        before = deploy_clock_snapshot(tmp_path)
        clock.unlink()

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        assert report[0] == 'falsified'

    def test_a_restamp_with_identical_bytes_is_still_attributed(
        self, tmp_path: Path,
    ) -> None:
        """The mtime-only signal must carry provenance too.

        The clock writes whole seconds, so a genuine redeploy landing inside the
        same second as the snapshot is byte-identical and visible only through
        mtime. If that path skipped attribution it would fail every run it
        straddled — the exact bug being fixed, surviving in its narrowest form.
        """
        body = _stamp(token='')
        clock = _write(tmp_path, _FLEET_RELPATH, body)
        before = deploy_clock_snapshot(tmp_path)
        stat = clock.stat()
        os.utime(clock, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        after = deploy_clock_snapshot(tmp_path)
        before_entry, after_entry = before[_FLEET_RELPATH], after[_FLEET_RELPATH]
        assert before_entry is not None and after_entry is not None
        assert after_entry[0] == before_entry[0], 'bytes must be identical'

        report = deploy_clock_change_report(
            before, after, session_token=_THIS_SESSION,
        )

        assert report is not None
        assert report[0] == 'external_redeploy'

    @pytest.mark.parametrize('token', [None, ''])
    def test_an_unstamped_session_token_fails_closed(
        self, tmp_path: Path, token: str | None,
    ) -> None:
        """A falsy token must never read as "everything is external".

        Mirrors ``leaked_drain_processes``' fail-closed contract on its own
        token: the first time the fixture failed to stamp one, a token-trusting
        guard would silently forgive every write instead of failing loudly.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=''))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=token,
        )

        assert report is not None
        assert report[0] == 'falsified'

    @pytest.mark.parametrize('token,expected', [('', 'external_redeploy'), (_THIS_SESSION, 'falsified')])
    def test_a_root_names_the_absolute_file_in_both_verdicts(
        self, tmp_path: Path, token: str, expected: str,
    ) -> None:
        """A run guards MORE THAN ONE checkout, so a bare relpath cannot say
        which one moved — and that is as true of the benign verdict as of the
        accusing one.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=token))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path),
            session_token=_THIS_SESSION, root=tmp_path,
        )

        assert report is not None
        verdict, message = report
        assert verdict == expected
        assert str(tmp_path / _FLEET_RELPATH) in message, message

    @pytest.mark.parametrize('token,expected', [('', 'external_redeploy'), (_THIS_SESSION, 'falsified')])
    def test_the_fm_clock_is_attributed_identically(
        self, tmp_path: Path, token: str, expected: str,
    ) -> None:
        """The second protected clock, and the one the measured 2026-08-28
        instances actually pointed at — it must not be covered by inheritance
        from the fleet clock's cases alone.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FM_RELPATH, _stamp(token=token, source=_WATCHDOG_SOURCE))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path),
            session_token=_THIS_SESSION, root=tmp_path,
        )

        assert report is not None
        verdict, message = report
        assert verdict == expected
        assert str(tmp_path / _FM_RELPATH) in message, message

    def test_both_observed_bodies_are_printed_in_the_benign_verdict_too(
        self, tmp_path: Path,
    ) -> None:
        """Triage from the OUTPUT alone, exactly as the failing path already
        guarantees: an operator reading a warning in a merge-lane log must be
        able to check the {ts, iso} against the deploy they expect without
        re-reading a file that has since moved again.
        """
        _write(tmp_path, _FLEET_RELPATH, _stamp(token='', ts=111))
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token='', ts=222))

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path), session_token=_THIS_SESSION,
        )

        assert report is not None
        verdict, message = report
        assert verdict == 'external_redeploy'
        assert '"ts": 111' in message, message
        assert '"ts": 222' in message, message

    def test_the_first_offender_in_protected_order_is_reported(
        self, tmp_path: Path,
    ) -> None:
        """Same first-offender contract as ``deploy_clock_violation_reason``:
        one report per root, in PROTECTED_DEPLOY_CLOCK_RELPATHS order.
        """
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')
        _write(tmp_path, _FM_RELPATH, b'{"ts": 2, "iso": "y"}\n')

        report = deploy_clock_change_report(
            before, deploy_clock_snapshot(tmp_path),
            session_token=_THIS_SESSION, root=tmp_path,
        )

        assert report is not None
        assert PROTECTED_DEPLOY_CLOCK_RELPATHS[0] in report[1]


class TestViolationReasonIsTheFalsifiedHalfOfTheReport:
    """The old entry point keeps its signature and narrows to one verdict.

    Everything in ``TestDeployClockViolationReason`` above stays green untouched
    because every body it builds is provenance-free, hence still unattributable
    and still a violation. What is new is the other direction: an attributed
    external stamp must now come back as ``None`` from this function, which is
    what lets the fixture pass the run.
    """

    def test_an_external_stamp_is_not_a_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The token is SET explicitly rather than inherited from the live run.

        The guard fixture does stamp one for the whole session, but a test that
        leaned on that would be pinning the fixture's wiring (which
        ``TestGuardIsLiveInThisRun`` owns) instead of this function's contract,
        and would flip to green for the wrong reason — an absent token fails
        CLOSED here, which is a different pinned behaviour entirely.
        """
        monkeypatch.setenv(PYTEST_SESSION_TOKEN_ENV, _THIS_SESSION)
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=''))

        assert deploy_clock_violation_reason(
            before, deploy_clock_snapshot(tmp_path),
        ) is None

    def test_a_provenance_free_stamp_is_still_a_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-vacuity control for the test above, in the same harness."""
        monkeypatch.setenv(PYTEST_SESSION_TOKEN_ENV, _THIS_SESSION)
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, b'{"ts": 1, "iso": "x"}\n')

        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(tmp_path))

        assert reason is not None
        assert 'falsified a REAL deploy clock' in reason

    def test_an_absent_ambient_token_fails_closed(self, tmp_path: Path, monkeypatch) -> None:
        """The fail-closed default, at the wrapper's own boundary.

        If the guard fixture ever stopped stamping the token, every write would
        read as unattributable and keep failing — noisily wrong, never silently
        permissive. That direction is the one worth pinning.
        """
        monkeypatch.delenv(PYTEST_SESSION_TOKEN_ENV, raising=False)
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=''))

        assert deploy_clock_violation_reason(
            before, deploy_clock_snapshot(tmp_path),
        ) is not None

    def test_it_reads_this_sessions_token_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The three-argument signature is preserved, so the token can only come
        from the ambient environment — which is precisely where every spawner
        picks it up.
        """
        monkeypatch.setenv(PYTEST_SESSION_TOKEN_ENV, _THIS_SESSION)
        before = deploy_clock_snapshot(tmp_path)
        _write(tmp_path, _FLEET_RELPATH, _stamp(token=_THIS_SESSION))

        reason = deploy_clock_violation_reason(
            before, deploy_clock_snapshot(tmp_path), root=tmp_path,
        )

        assert reason is not None
        assert 'this run' in reason, reason


class TestGuardIsLiveInThisRun:
    """The fixture is WIRED, not merely defined.

    Everything above tests pure functions against ``tmp_path`` roots; all of it
    would stay green if the fixture were never loaded by any conftest.  This
    class is the only assertion that the defence is actually armed in the
    process running it — the difference between a wired defence and a dead one.
    """

    def test_the_guard_fixture_exists(self) -> None:
        assert hasattr(df_pytest_isolation, _GUARD_NAME), (
            f'df_pytest_isolation defines no {_GUARD_NAME}; the pure helpers '
            'above protect nothing on their own.'
        )

    def test_the_guard_fixture_is_session_scoped_and_autouse(self) -> None:
        """Both properties pinned STRUCTURALLY, not by inspection of behaviour.

        A function-scoped or non-autouse variant would still import cleanly and
        keep every test above green while protecting nothing: function scope
        would miss writes from module-/session-scoped fixtures (where expensive
        subprocess setup tends to live), and without ``autouse`` nothing would
        ever request it.
        """
        marker = fixture_marker(getattr(df_pytest_isolation, _GUARD_NAME))

        assert marker.scope == 'session', f'scope is {marker.scope!r}, expected session'
        assert marker.autouse is True, 'the guard must be autouse — nothing requests it'

    def test_the_guard_fixture_is_registered_in_this_run(self, request) -> None:
        """The conftest binding is real WIRING, not a dormant definition.

        pytest only collects fixtures bound into a conftest's namespace, which
        is why df_pytest_isolation's fixtures are imported there under
        ``# noqa: F401 — the binding IS the wiring``. Deleting that import
        breaks nothing visible except this assertion.
        """
        try:
            request.getfixturevalue(_GUARD_NAME)
        except pytest.FixtureLookupError:
            pytest.fail(
                f'{_GUARD_NAME} is not registered for this rootdir. Wire the '
                'test-root conftest to import it from df_pytest_isolation '
                '(`# noqa: F401 — the binding IS the wiring`); without that, '
                'this whole suite runs with no deploy-clock guard.',
                pytrace=False,
            )


# ---------------------------------------------------------------------------
# The guard's FAILURE contract, exercised through a real nested pytest run.
#
# Everything above pins the helpers, the marker and the registration; none of it
# pins what a violation actually DOES to the run. A refactor that warned instead
# of failing, or moved the check into a fixture nobody requests, would leave
# every test in this file green while the defence emitted nothing an exit code
# could carry. Only a nested run can observe "the process exits non-zero even
# though every test passed", because a fixture cannot fail its own session.
# ---------------------------------------------------------------------------

# Minimal ini so the nested run's rootdir is the tmp tree and NOT this repo:
# without it pytest walks up looking for an inifile and would inherit this
# repo's addopts (`--import-mode=importlib -m 'not smoke ...'`).
_NESTED_INI = '[pytest]\n'

_NESTED_CONFTEST = '''\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from df_pytest_isolation import _df_deploy_clocks_unwritten  # noqa: F401
'''

_NESTED_STAMP_BODY = '{"ts": 1786033966, "iso": "2026-08-06T16:32:46+00:00"}'


def _nested_test_source(*, writes_clock: bool) -> str:
    """Source for the nested test module — which PASSES either way.

    Built by concatenation rather than ``str.format``: the stamp body is JSON, so
    a template would have to escape its braces, and a mis-escaped one would
    silently write a different file than the guard watches.
    """
    body = (
        (
            '    clock = Path(__file__).resolve().parent / RELPATH\n'
            '    clock.parent.mkdir(parents=True, exist_ok=True)\n'
            f'    clock.write_text({_NESTED_STAMP_BODY!r})\n'
        )
        if writes_clock
        else '    pass\n'
    )
    return (
        'from pathlib import Path\n'
        '\n'
        f'RELPATH = {_FLEET_RELPATH!r}\n'
        '\n'
        '\n'
        'def test_a_forgetful_spawner():\n'
        '    """PASSES. The damage is to the checkout, not to this result."""\n'
        + body
    )


def _nested_run(tmp_path: Path, *, writes_clock: bool) -> subprocess.CompletedProcess[str]:
    """Run a throwaway pytest session wired to the guard, in its own tmp checkout.

    The copied ``df_pytest_isolation.py`` sits at the tmp tree's root, so the
    guard's ``Path(__file__).resolve().parent`` resolves THERE and the protected
    clocks it watches are the tmp ones — the real checkout is never involved.
    """
    root = tmp_path / ('violating' if writes_clock else 'clean')
    root.mkdir()
    shutil.copy2(Path(df_pytest_isolation.__file__), root / 'df_pytest_isolation.py')
    (root / 'pytest.ini').write_text(_NESTED_INI)
    (root / 'conftest.py').write_text(_NESTED_CONFTEST)
    (root / 'test_forgetful.py').write_text(_nested_test_source(writes_clock=writes_clock))
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(root)],
        cwd=root, capture_output=True, text=True, timeout=300,
    )


class TestTheGuardFailsTheRunEndToEnd:
    """A violation must cost the RUN, not merely log something."""

    def test_a_stamped_clock_fails_a_session_whose_tests_all_passed(
        self, tmp_path: Path,
    ) -> None:
        result = _nested_run(tmp_path, writes_clock=True)
        combined = result.stdout + result.stderr

        assert result.returncode != 0, (
            'a run that falsified a deploy clock exited 0 — the guard is inert. '
            f'stdout={result.stdout!r}'
        )
        assert '1 passed' in combined, (
            'the nested TEST must still pass, or this proves nothing about a '
            f'green suite being caught. output={combined!r}'
        )
        assert 'falsified a REAL deploy clock' in combined, combined
        # The BASENAME, not the relpath: the message names the absolute file, and
        # asserting on a 48-char segment of a long tmp path would be hostage to
        # terminal-width wrapping rather than to the guard's behaviour.
        assert Path(_FLEET_RELPATH).name in combined, combined

    def test_the_same_harness_without_the_write_exits_zero(self, tmp_path: Path) -> None:
        """Non-vacuity control: the identical nested tree, minus the clock write.

        Without it the failure above could be any nested-harness breakage — a
        bad ini, an unimportable module, a missing pytest.
        """
        result = _nested_run(tmp_path, writes_clock=False)

        assert result.returncode == 0, (
            f'the control run failed for an unrelated reason: '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert 'falsified a REAL deploy clock' not in result.stdout


# The sibling that used to reach into THIS module for the marker helper. Named
# and located as constants so the guard below fails with a legible message
# rather than a bare ModuleNotFoundError if the file is ever renamed.
_SIBLING_MODULE_NAME = 'test_drain_process_leak_isolation'
_SIBLING_PATH = Path(__file__).with_name(_SIBLING_MODULE_NAME + '.py')


def _sibling_drain_module() -> ModuleType:
    """The live ``test_drain_process_leak_isolation`` module object.

    Preferred from ``sys.modules``, matched by ``__file__`` rather than by name:
    pytest's ``--import-mode=importlib`` (set in the root ``addopts``) registers
    a collected test module under a rootdir-derived dotted key, not under its
    bare filename, so a name lookup would miss the module the session actually
    loaded. Only when this file runs alone — the sibling never collected — is it
    imported by name, which ``tests/scripts/conftest.py`` makes resolvable by
    putting this directory on ``sys.path``.

    A bare import, never an ``importorskip``: if the sibling stops importing,
    this guard must say so loudly rather than skip.
    """
    for module in list(sys.modules.values()):
        origin = getattr(module, '__file__', None)
        if origin and Path(origin).resolve() == _SIBLING_PATH.resolve():
            return module
    return importlib.import_module(_SIBLING_MODULE_NAME)


def test_fixture_marker_is_the_shared_one_not_a_local_copy() -> None:
    """The fixture-marker helper must have exactly ONE definition (task 3960).

    WHY A SECOND COPY IS THE DEFECT AND NOT A STYLE NIT. pytest's fixture marker
    is PRIVATE and has already MOVED once: ``<=8.x`` hangs it off the decorated
    function as ``_pytestfixturefunction``, while ``9.x`` wraps the function in a
    ``FixtureFunctionDefinition`` carrying ``_fixture_function_marker``. Every
    copy of the lookup is one more thing to update the next time it moves — and
    a private-API pin that silently stops finding its target is worse than no
    pin, because it still reads as coverage. The shared helper
    ``df_pytest_isolation.fixture_marker`` is deliberately built to fail loudly
    instead (``pytest.fail(..., pytrace=False)`` when it finds neither
    spelling), so there is exactly one place to fix.

    It also removes a test-module-imports-test-module coupling: the sibling
    ``test_drain_process_leak_isolation`` reached into THIS module for the
    helper, which only resolved because ``tests/scripts/conftest.py`` puts this
    directory on ``sys.path`` — something pytest's ``--import-mode=importlib``
    deliberately does not do.

    MEASURED RED at base main ``23ce883356``: this module defined its own
    ``_fixture_marker`` (with ``_MARKER_ATTRS`` beside it) and bound no
    ``fixture_marker`` at all, and the sibling bound ``_fixture_marker`` via
    ``from test_deploy_clock_isolation import _fixture_marker`` — so all three
    assertions below failed.

    ``tests/scripts/test_fleet_dir_isolation.py`` is the working reference for
    the intended shape: it already imports ``fixture_marker`` from
    ``df_pytest_isolation`` through the same preamble both these modules carry.
    """
    this_module = sys.modules[__name__]

    # (a) This module resolves the SHARED symbol, by identity — not a same-named
    # re-implementation, which would satisfy any weaker "is it callable" test.
    resolved = getattr(this_module, 'fixture_marker', None)
    assert resolved is df_pytest_isolation.fixture_marker, (
        f'{__name__}.fixture_marker is {resolved!r}, not '
        f'df_pytest_isolation.fixture_marker (task 3960) — import the shared '
        f'helper into the existing `from df_pytest_isolation import (...)` '
        f'block rather than re-implementing the private-API lookup here'
    )

    # (b) The two names the private copy occupied are GONE, so it cannot grow
    # back beside the shared one and quietly become the one that gets called.
    for name in ('_fixture_marker', '_MARKER_ATTRS'):
        assert not hasattr(this_module, name), (
            f'{__name__} still defines {name!r} (task 3960) — that is the '
            f'private copy of the fixture-marker lookup; delete it and use '
            f'df_pytest_isolation.fixture_marker / _FIXTURE_MARKER_ATTRS'
        )

    # (c) The sibling does not bind the private copy either — the cross-test-
    # module import must not return. Reached through the live module object
    # rather than by re-importing the name, so this tests what the session
    # actually loaded.
    sibling = _sibling_drain_module()
    assert not hasattr(sibling, '_fixture_marker'), (
        f'{_SIBLING_MODULE_NAME} binds _fixture_marker (task 3960) — that is '
        f'either a second private copy or a `from {__name__} import '
        f'_fixture_marker`, a test-module-imports-test-module coupling that '
        f'only resolves because tests/scripts/conftest.py puts this directory '
        f'on sys.path. Import df_pytest_isolation.fixture_marker there instead.'
    )
