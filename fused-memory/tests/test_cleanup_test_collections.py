"""Tests for cleanup_test_collections.py — the ephemeral-collection reaper.

This script had no test file before task 3199.  It is documented as a cron
job, so the two properties that matter most are the boring ones: it reaps
EVERYTHING an integration test can leave behind, and it never explodes when
the world is not there.

Everything here is pure.  A fake ``qdrant_client`` module is injected into
``sys.modules`` (the script imports it inside ``main()``, which is exactly
what makes that injectable), so no test in this file touches a network,
Qdrant or an API key — the whole file runs in the merge lane.

The prefix-agreement tests are the point of the file.  A collection whose
name is coined in ``bake_off_storage_shape.py`` and reaped by a constant in
THIS module is one rename away from leaking forever: nothing under the
default ``fused`` prefix is reapable by design, so an orphan is permanent,
not slow.
"""
from __future__ import annotations

import errno
import functools
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
SCRIPT_PATH = SCRIPTS_DIR / 'cleanup_test_collections.py'
BAKE_OFF_PATH = SCRIPTS_DIR / 'bake_off_storage_shape.py'


def _load(path: Path, name: str) -> types.ModuleType:
    """Load a standalone script by path, registered under its bare name.

    Registration is required, not cosmetic: ``@dataclass`` and other
    reflection-based decorators look their defining module up in
    ``sys.modules`` on the way in.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@functools.cache
def _mod() -> types.ModuleType:
    return _load(SCRIPT_PATH, 'cleanup_test_collections')


@functools.cache
def _bake_off() -> types.ModuleType:
    return _load(BAKE_OFF_PATH, 'bake_off_storage_shape')


# --- lease-dir isolation (task 4775, prerequisite pre-1) -------------------


@pytest.fixture(autouse=True)
def lease_dir(tmp_path, monkeypatch):
    """Point ``DF_EPHEMERAL_COLLECTION_LEASE_DIR`` at a per-test directory.

    A hard isolation boundary, not a convenience.  The lease directory
    ``cleanup_test_collections.lease_dir()`` returns by default is a
    HARDCODED machine-global absolute path (that is the property the guard's
    correctness rests on — see the design note on that function), and it is
    the very directory the live 6-hourly cron reads.  A test that wrote a
    lease into it would hold a real sweep off this host; a test that reaped
    it would unlink the lease of a live bake-off running in another checkout.

    Autouse, and applied to EVERY test in this module rather than only the
    lease tests, for exactly that reason: a test that forgets to request the
    isolation must not be able to fall through silently to the real
    directory.  Tests that need the path can still request this fixture by
    name; the directory is not created here, because a lease-dir-absent case
    is one of the behaviours under test.
    """
    directory = tmp_path / 'ephemeral-collection-leases'
    monkeypatch.setenv('DF_EPHEMERAL_COLLECTION_LEASE_DIR', str(directory))
    return directory


# --- the fake Qdrant -------------------------------------------------------


class _FakeClient:
    """Records what the reaper asks Qdrant to do."""

    def __init__(self, names, fail_on, url=None, timeout=None):
        self.url = url
        self.timeout = timeout
        self._names = list(names)
        self._fail_on = set(fail_on)
        self.deleted: list[str] = []
        self.listed = 0
        self.closed = False

    def get_collections(self):
        self.listed += 1
        return types.SimpleNamespace(
            collections=[types.SimpleNamespace(name=name) for name in self._names],
        )

    def delete_collection(self, name):
        if name in self._fail_on:
            raise RuntimeError(f'qdrant refused to delete {name}')
        self.deleted.append(name)

    def close(self):
        self.closed = True


def _install_fake_qdrant(
    monkeypatch, names, *, fail_on=(), unreachable=False,
) -> list[_FakeClient]:
    """Inject a fake ``qdrant_client`` module; return the clients constructed."""
    created: list[_FakeClient] = []

    class _Client(_FakeClient):
        def __init__(self, url=None, timeout=None):
            if unreachable:
                raise ConnectionError('connection refused')
            super().__init__(names, fail_on, url=url, timeout=timeout)
            created.append(self)

    module = types.ModuleType('qdrant_client')
    module.QdrantClient = _Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'qdrant_client', module)
    return created


#: Collections a live Qdrant really shows alongside the reapable ones.  Named
#: explicitly rather than as "some other string": these are the production
#: collections, and reaping one would delete real memory.
LIVE_COLLECTIONS = (
    'fused_dark_factory',
    'fused_reify',
    'task_curator_reify',
    'task_dedup_dark_factory',
)


class TestPrefixAgreement:
    """This module owns the prefixes, and the tuple is what the cron deletes.

    The cross-file half — that the bake-off's seeded names really are
    reapable — is asserted where it can actually fail, over the names
    `ephemeral_collections()` builds
    (test_bake_off_storage_shape.py::test_every_arm_collection_starts_with_the_reapable_prefix).
    Comparing `ephemeral_collection_prefix()` against `E2_BAKEOFF_PREFIX`
    here could not: that function's whole body is
    `return load_cleanup_script().E2_BAKEOFF_PREFIX`, so both sides read the
    same attribute of this same module and a rename moves them together.
    """

    def test_the_reaped_prefixes_are_pinned_by_equality(self):
        """By equality, not membership: this tuple is the complete list of
        names this cron job is allowed to delete."""
        mod = _mod()

        assert mod.PREFIXES == (mod.PREFIX, mod.E2_BAKEOFF_PREFIX)

    def test_the_legacy_prefix_is_untouched(self):
        """3199 adds a prefix beside the existing one; it does not retune the
        existing reap."""
        assert _mod().PREFIX == '_test_mem0_qdrant_integration_'


# --- the in-use lease (task 4775) ------------------------------------------


class TestLeaseDirIsEnvironmentIndependent:
    """The guard fires only if the HOLDER and the REAPER resolve one directory.

    They never share an environment.  The reaper is a bare-``python3`` cron
    job with a near-empty env; a holder is a pytest process (usually inside
    a ``.worktrees/<id>`` checkout, under an xdist worker) or a hand-run
    script in a login shell.  Every "portable" temp-directory source moves
    between those two: ``tempfile.gettempdir()`` honours ``TMPDIR``/``TEMP``/
    ``TMP``, which pytest and cron set differently, and ``XDG_RUNTIME_DIR``
    is ``/run/user/<uid>`` in a login session and unset under cron.

    A guard that resolved two different directories would find no lease,
    delete, and leave nothing behind pointing at the reaper — silently never
    firing, which is strictly worse than no guard, because it also stops the
    next person looking.  So environment-independence is not a nicety here;
    it is the whole property, and it is asserted before anything else.
    """

    #: Every variable a "portable temp dir" helper would consult.
    TEMP_VARS = ('TMPDIR', 'TEMP', 'TMP', 'XDG_RUNTIME_DIR')

    def _default(self, monkeypatch):
        """The resolved lease dir with the test override removed."""
        mod = _mod()
        monkeypatch.delenv(mod.LEASE_DIR_ENV, raising=False)
        return mod.lease_dir()

    @pytest.mark.parametrize('var', TEMP_VARS)
    def test_no_single_temp_variable_moves_it(self, monkeypatch, tmp_path, var):
        mod = _mod()
        baseline = self._default(monkeypatch)

        monkeypatch.setenv(var, str(tmp_path / f'{var.lower()}-elsewhere'))

        assert mod.lease_dir() == baseline

    def test_all_four_set_at_once_do_not_move_it(self, monkeypatch, tmp_path):
        """Set together, not merely one at a time: a helper that consulted
        them in priority order would survive every single-variable case."""
        mod = _mod()
        baseline = self._default(monkeypatch)

        for var in self.TEMP_VARS:
            monkeypatch.setenv(var, str(tmp_path / f'{var.lower()}-elsewhere'))

        assert mod.lease_dir() == baseline

    def test_the_working_directory_does_not_move_it(self, monkeypatch, tmp_path):
        """A repo-relative path would give every worktree its own private
        lease dir, while ONE Qdrant at localhost:6333 is shared by all of
        them — so the bake-off in one checkout would be invisible to a cron
        sweep launched from another."""
        mod = _mod()
        baseline = self._default(monkeypatch)
        elsewhere = tmp_path / 'some-other-cwd'
        elsewhere.mkdir()

        monkeypatch.chdir(elsewhere)

        assert mod.lease_dir() == baseline

    def test_it_is_absolute_and_outside_the_repo_working_tree(self, monkeypatch):
        """Absolute for the same reason: the cron's cwd is not this repo, and
        a lease under the tree would also leave the machine-operated
        `project_root` checkout dirty."""
        resolved = self._default(monkeypatch)

        assert resolved.is_absolute()
        repo_root = Path(__file__).resolve().parent.parent.parent
        assert not resolved.resolve().is_relative_to(repo_root)

    def test_the_env_override_is_used_exactly_with_no_suffixing(
        self, monkeypatch, tmp_path,
    ):
        """Exactly, because both sides must be able to agree on it: an
        operator who points one process at a directory has to get THAT
        directory, not a subdirectory of it that the other side may derive
        differently."""
        mod = _mod()
        override = tmp_path / 'operator' / 'chosen' / 'dir'

        monkeypatch.setenv(mod.LEASE_DIR_ENV, str(override))

        assert mod.lease_dir() == override

    def test_the_override_is_read_at_call_time_not_import_time(
        self, monkeypatch, tmp_path,
    ):
        """The module is long since imported by the time this runs, and the
        two calls below straddle a change to the variable — a value captured
        at import (or memoised) could not follow it.  This is what makes the
        pre-1 isolation fixture able to work at all."""
        mod = _mod()
        first, second = tmp_path / 'first', tmp_path / 'second'

        monkeypatch.setenv(mod.LEASE_DIR_ENV, str(first))
        seen_first = mod.lease_dir()
        monkeypatch.setenv(mod.LEASE_DIR_ENV, str(second))
        seen_second = mod.lease_dir()

        assert (seen_first, seen_second) == (first, second)


class TestHoldLease:
    """A holder publishes exactly one file and always takes it away again.

    Publishing is the easy half.  The half that matters is that a holder
    which crashes cannot leave a file behind that holds the 6-hourly cron
    off forever, and that two holders — the xdist case, where ``-n auto``
    workers each seed their own corpus — never share a file, so one
    finishing cannot un-guard the others.
    """

    @staticmethod
    def _names(directory) -> list[str]:
        """Lease filenames currently published, or [] if nothing exists."""
        if not directory.exists():
            return []
        return sorted(path.name for path in directory.iterdir())

    def test_it_yields_true_and_publishes_one_file_naming_the_holder(
        self, lease_dir,
    ):
        """The body is diagnostics for whoever probes next: an operator
        reading cron mail has to be able to attribute a held-off sweep to a
        specific run rather than guess."""
        mod = _mod()

        with mod.hold_lease(owner='e2-bake-off gw3') as held:
            assert held is True
            published = list(lease_dir.iterdir())
            assert len(published) == 1
            body = json.loads(published[0].read_text())

        assert body['owner'] == 'e2-bake-off gw3'
        assert body['pid'] == os.getpid()

    def test_it_removes_the_file_on_a_normal_exit(self, lease_dir):
        mod = _mod()

        with mod.hold_lease(owner='e2-bake-off'):
            assert len(self._names(lease_dir)) == 1

        assert self._names(lease_dir) == []

    def test_it_removes_the_file_and_re_raises_when_the_block_raises(
        self, lease_dir,
    ):
        """A failed bake-off must not hold the cron off.  Re-raised
        UNCHANGED, identity included: a lease that swallowed or wrapped the
        run's own failure would be a far worse bug than the one it guards."""
        mod = _mod()
        boom = RuntimeError('the bake-off fell over mid-seed')

        with pytest.raises(RuntimeError) as caught, mod.hold_lease(owner='e2'):
            assert len(self._names(lease_dir)) == 1
            raise boom

        assert caught.value is boom
        assert self._names(lease_dir) == []

    def test_it_creates_the_lease_directory_when_it_is_absent(self, lease_dir):
        """First holder after a reboot: nothing has made the directory yet."""
        mod = _mod()
        assert not lease_dir.exists()

        with mod.hold_lease(owner='first-after-boot'):
            assert lease_dir.is_dir()
            assert len(self._names(lease_dir)) == 1

    def test_two_holders_get_distinct_files_and_release_independently(
        self, lease_dir,
    ):
        """The xdist case.  Under ``-n auto`` several workers seed at once,
        each holding its own lease; the first to finish must not un-guard
        the ones still running."""
        mod = _mod()

        with mod.hold_lease(owner='worker-gw0'):
            outer = self._names(lease_dir)
            with mod.hold_lease(owner='worker-gw1'):
                both = self._names(lease_dir)
            after_inner_released = self._names(lease_dir)

        assert len(outer) == 1
        assert len(both) == 2
        assert after_inner_released == outer
        assert self._names(lease_dir) == []


def _wait_for_marker(marker_path: Path, timeout: float = 5.0) -> bool:
    """Poll for *marker_path* to appear; return False on timeout.

    Test-side synchronisation only, and bounded, so a child that fails to
    start cannot hang the suite forever.  Mirrors
    ``shared/tests/test_verify_admission.py::_wait_for_marker``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker_path.exists():
            return True
        time.sleep(0.02)
    return False


#: Inline stdlib child: takes a real lease in another process, signals
#: readiness with a marker file, then blocks so the parent can observe it —
#: and, in the SIGKILL case, can kill it without any chance of cleanup
#: running.  The construction mirrors
#: ``shared/tests/test_verify_admission.py::_SELF_HEAL_CHILD_SRC``, which
#: already demonstrates the kernel-release mechanism passing on this
#: toolchain.
_LIVE_HOLDER_CHILD_SRC = (
    'import fcntl, json, os, sys, time\n'
    'fd = os.open(sys.argv[1], os.O_CREAT | os.O_WRONLY, 0o644)\n'
    'fcntl.flock(fd, fcntl.LOCK_EX)\n'
    "os.write(fd, json.dumps({'owner': sys.argv[3], 'pid': os.getpid()}).encode())\n"
    "open(sys.argv[2], 'w').write('ready')\n"
    'time.sleep(60)\n'
)


class TestLiveLeasesSeesOnlyLivingHolders:
    """Liveness is the flock, never the file and never a clock.

    This class is where the no-TTL premise is paid for.  If a lease that
    outlived its holder still read as live, the 6-hourly cron would be
    wedged by the first bake-off that ever crashed, and the design would
    need the expiry it deliberately does not have.
    """

    def test_it_reports_a_lease_held_in_this_process(self, lease_dir):
        mod = _mod()

        with mod.hold_lease(owner='e2-bake-off gw2'):
            live = mod.live_leases()

        assert len(live) == 1
        assert live[0]['owner'] == 'e2-bake-off gw2'
        assert live[0]['pid'] == os.getpid()

    def test_it_reports_nothing_once_that_holder_has_exited(self, lease_dir):
        mod = _mod()

        with mod.hold_lease(owner='e2-bake-off'):
            pass

        assert mod.live_leases() == []

    def test_it_reports_a_lease_held_by_another_live_process(self, lease_dir):
        """The case that actually matters: the cron is a DIFFERENT process
        from the run it must not reap, so an in-process-only probe would
        pass every test here and protect nothing in production."""
        mod = _mod()
        lease_dir.mkdir(parents=True)
        marker = lease_dir.parent / 'ready.marker'
        lease_path = lease_dir / 'other-process.lease'
        proc = subprocess.Popen([
            sys.executable, '-c', _LIVE_HOLDER_CHILD_SRC,
            str(lease_path), str(marker), 'bake-off-in-another-shell',
        ])
        try:
            assert _wait_for_marker(marker), 'child never signalled readiness'

            live = mod.live_leases()

            assert [record['owner'] for record in live] == [
                'bake-off-in-another-shell',
            ]
            assert live[0]['pid'] == proc.pid
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_a_sigkilled_holder_stops_being_reported_with_no_cleanup(
        self, lease_dir,
    ):
        """The single most load-bearing assertion in this file.

        A SIGKILLed holder runs no `finally`, no atexit and no signal
        handler — its lease file is still on disk afterwards, and this
        asserts so, because that is exactly the residue a file-existence
        guard would mistake for a live run and be wedged by forever.  The
        kernel drops the flock on process death, so the probe sees the truth
        with no TTL, no clock and no pid check.
        """
        mod = _mod()
        lease_dir.mkdir(parents=True)
        marker = lease_dir.parent / 'ready.marker'
        lease_path = lease_dir / 'doomed.lease'
        proc = subprocess.Popen([
            sys.executable, '-c', _LIVE_HOLDER_CHILD_SRC,
            str(lease_path), str(marker), 'a-run-that-is-about-to-die',
        ])
        try:
            assert _wait_for_marker(marker), 'child never signalled readiness'
            assert len(mod.live_leases()) == 1

            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
            proc = None

            assert lease_path.exists(), 'a SIGKILLed holder cleans nothing up'
            assert mod.live_leases() == []
        finally:
            if proc is not None:
                proc.kill()
                proc.wait(timeout=5)

    def test_it_returns_empty_and_does_not_raise_when_the_directory_is_absent(
        self, lease_dir,
    ):
        """The state of a machine that has never run an integration test.
        This is called from an unattended cron job whose contract is
        "always exits 0"."""
        mod = _mod()
        assert not lease_dir.exists()

        assert mod.live_leases() == []

    @pytest.mark.parametrize('body', [b'', b'not json at all', b'{"owner":'])
    def test_a_corrupt_body_degrades_the_diagnostic_not_the_guard(
        self, lease_dir, body,
    ):
        """A holder killed between creating its file and writing its record
        is still HOLDING the flock.  Reading its body must not be able to
        decide whether it exists — the filename stands in for the owner and
        the lease is still reported."""
        mod = _mod()
        lease_dir.mkdir(parents=True)
        marker = lease_dir.parent / 'ready.marker'
        lease_path = lease_dir / 'half-written.lease'
        lease_path.write_bytes(body)
        child = (
            'import fcntl, os, sys, time\n'
            'fd = os.open(sys.argv[1], os.O_RDWR)\n'
            'fcntl.flock(fd, fcntl.LOCK_EX)\n'
            "open(sys.argv[2], 'w').write('ready')\n"
            'time.sleep(60)\n'
        )
        proc = subprocess.Popen(
            [sys.executable, '-c', child, str(lease_path), str(marker)],
        )
        try:
            assert _wait_for_marker(marker), 'child never signalled readiness'

            live = mod.live_leases()

            assert len(live) == 1
            assert live[0]['owner'] == 'half-written.lease'
        finally:
            proc.kill()
            proc.wait(timeout=5)


class TestHoldLeaseFailsOpenAndLoud:
    """An unusable lease directory degrades to the status quo, VISIBLY.

    The holder side runs inside integration tests and inside two seeding
    scripts.  If an unwritable or full ``/tmp`` made this raise, a live
    bake-off would abort for a reason with nothing to do with what it
    measures, and an integration test would fail on infrastructure noise.

    What this change improves on is "no guard at all", so degrading back to
    it is not a regression — degrading back to it SILENTLY would be.  Hence
    the yielded ``held: bool`` and the stderr line, mirroring
    ``shared/verify_admission.py::acquire_task_slot`` (clause C-fail-open).
    """

    @staticmethod
    def _unusable_under_a_regular_file(monkeypatch, tmp_path) -> Path:
        """A lease dir whose parent is a regular file: mkdir gives ENOTDIR."""
        blocker = tmp_path / 'not-a-directory'
        blocker.write_text('something else owns this path')
        unusable = blocker / 'leases'
        monkeypatch.setenv(_mod().LEASE_DIR_ENV, str(unusable))
        return unusable

    def test_it_yields_false_and_still_runs_the_block(
        self, monkeypatch, tmp_path, capsys,
    ):
        mod = _mod()
        self._unusable_under_a_regular_file(monkeypatch, tmp_path)
        observed = []

        with mod.hold_lease(owner='e2-bake-off gw0') as held:
            observed.append(held)

        assert observed == [False]
        capsys.readouterr()

    def test_it_reports_the_directory_and_the_error_on_one_stderr_line(
        self, monkeypatch, tmp_path, capsys,
    ):
        """Both facts, because neither alone is actionable: the directory
        says WHERE to look, the OSError says what to fix.  One line, because
        this also fires under an unattended cron-adjacent run."""
        mod = _mod()
        unusable = self._unusable_under_a_regular_file(monkeypatch, tmp_path)

        with mod.hold_lease(owner='e2-bake-off gw0'):
            pass

        captured = capsys.readouterr()
        assert captured.out == ''
        lines = captured.err.strip().splitlines()
        assert len(lines) == 1, captured.err
        assert str(unusable) in lines[0]
        assert os.strerror(errno.ENOTDIR) in lines[0]

    def test_an_exception_in_the_block_still_propagates_unchanged(
        self, monkeypatch, tmp_path, capsys,
    ):
        """Failing open must not change the caller's control flow either —
        the un-held path has no fd and no file, and must still not swallow
        the run's own failure on the way out."""
        mod = _mod()
        self._unusable_under_a_regular_file(monkeypatch, tmp_path)
        boom = RuntimeError('the bake-off fell over anyway')

        with pytest.raises(RuntimeError) as caught, mod.hold_lease(owner='e2'):
            raise boom

        assert caught.value is boom
        capsys.readouterr()

    @pytest.mark.skipif(
        os.geteuid() == 0, reason='root ignores directory permissions',
    )
    def test_it_yields_false_when_the_lease_directory_is_read_only(
        self, monkeypatch, tmp_path, capsys,
    ):
        """The realistic shape of the failure: the directory exists and is
        simply not writable by this uid."""
        mod = _mod()
        parent = tmp_path / 'read-only'
        parent.mkdir()
        parent.chmod(0o500)
        monkeypatch.setenv(mod.LEASE_DIR_ENV, str(parent / 'leases'))
        try:
            with mod.hold_lease(owner='e2-bake-off') as held:
                assert held is False
        finally:
            parent.chmod(0o700)

        assert str(parent / 'leases') in capsys.readouterr().err


class TestSweep:
    """What the reaper deletes, and what it must not."""

    def test_it_reaps_the_legacy_integration_prefix(self, monkeypatch, capsys):
        mod = _mod()
        stale = [f'{mod.PREFIX}dark_factory', f'{mod.PREFIX}probe_e1_gw0']
        clients = _install_fake_qdrant(monkeypatch, [*stale, *LIVE_COLLECTIONS])

        mod.main()

        assert sorted(clients[0].deleted) == sorted(stale)
        capsys.readouterr()

    def test_it_reaps_the_e2_bakeoff_prefix(self, monkeypatch, capsys):
        mod = _mod()
        stale = [f'{mod.E2_BAKEOFF_PREFIX}_e2_bakeoff_c_peers_main']
        clients = _install_fake_qdrant(monkeypatch, [*stale, *LIVE_COLLECTIONS])

        mod.main()

        assert clients[0].deleted == stale
        capsys.readouterr()

    def test_it_reaps_the_names_the_bake_off_actually_creates(
        self, monkeypatch, capsys,
    ):
        """The strongest form of the agreement: not "the prefixes match" but
        "a collection this experiment really seeds is really reaped"."""
        mod = _mod()
        seeded = sorted(_bake_off().ephemeral_collections(suffix='gw3').values())
        clients = _install_fake_qdrant(monkeypatch, [*seeded, *LIVE_COLLECTIONS])

        mod.main()

        assert sorted(clients[0].deleted) == seeded
        capsys.readouterr()

    def test_it_leaves_every_non_matching_collection_alone(self, monkeypatch, capsys):
        """A reaper that over-matched would delete live project memory."""
        mod = _mod()
        clients = _install_fake_qdrant(monkeypatch, LIVE_COLLECTIONS)

        mod.main()

        assert clients[0].deleted == []
        capsys.readouterr()

    def test_both_prefixes_are_reaped_in_one_sweep(self, monkeypatch, capsys):
        """One listing, one client: a per-prefix pass would double the round
        trips and could reap against two different views of the world."""
        mod = _mod()
        stale = [f'{mod.PREFIX}old', f'{mod.E2_BAKEOFF_PREFIX}_new']
        clients = _install_fake_qdrant(monkeypatch, [*stale, *LIVE_COLLECTIONS])

        mod.main()

        assert len(clients) == 1
        assert clients[0].listed == 1
        assert sorted(clients[0].deleted) == sorted(stale)
        capsys.readouterr()

    def test_a_failed_delete_is_reported_and_the_sweep_continues(
        self, monkeypatch, capsys,
    ):
        """Aborting on the first failure would leave the rest of the leak in
        place — and the whole point of the sweep is that nothing survives it."""
        mod = _mod()
        doomed = f'{mod.PREFIX}locked'
        rest = [f'{mod.PREFIX}a', f'{mod.E2_BAKEOFF_PREFIX}_b']
        clients = _install_fake_qdrant(
            monkeypatch, [doomed, *rest, *LIVE_COLLECTIONS], fail_on=[doomed],
        )

        mod.main()

        assert sorted(clients[0].deleted) == sorted(rest)
        assert doomed in capsys.readouterr().err

    def test_the_client_is_closed_when_the_collection_listing_fails(
        self, monkeypatch, capsys,
    ):
        """Constructed, THEN the listing raises — the realistic cron failure.

        A Qdrant that accepts the TCP connection and then times out or 500s on
        `get_collections` leaves a live client object behind, so this is the
        one path where a leak is possible at all: an unreachable Qdrant raises
        in `QdrantClient.__init__` and never produces a client to leak.

        A delete-failure variant of this test would assert nothing — the
        per-collection `except` swallows that, so the close after the loop was
        always reached and the assertion passes with or without the `finally`.
        """
        mod = _mod()
        clients = _install_fake_qdrant(monkeypatch, [f'{mod.PREFIX}a'])

        def _explode(self):
            raise RuntimeError('timed out listing collections')

        monkeypatch.setattr(
            sys.modules['qdrant_client'].QdrantClient,
            'get_collections',
            _explode,
        )

        mod.main()  # a cron job, so still a clean return

        assert len(clients) == 1, 'no client was constructed, so nothing could leak'
        assert clients[0].closed is True
        assert 'unreachable' in capsys.readouterr().err.lower()


class TestStaysIdempotent:
    """It is a cron job: a missing world is a no-op, never a failure."""

    def test_it_returns_cleanly_when_qdrant_is_unreachable(self, monkeypatch, capsys):
        mod = _mod()
        _install_fake_qdrant(monkeypatch, [], unreachable=True)

        # No `main() is None`: it is declared `-> None` with no `return
        # <value>` on any path, so that would imply a return-value contract
        # the script does not have.  Returning at all IS the assertion — a
        # cron job that raised on an unreachable Qdrant would page someone.
        mod.main()

        assert 'unreachable' in capsys.readouterr().err.lower()

    def test_it_returns_cleanly_when_qdrant_client_is_not_installed(
        self, monkeypatch, capsys,
    ):
        """`sys.modules[name] = None` is CPython's own "this import fails"
        marker, so this exercises the real ImportError branch."""
        mod = _mod()
        monkeypatch.setitem(sys.modules, 'qdrant_client', None)

        mod.main()  # returning at all is the assertion; see the sibling above

        assert 'not installed' in capsys.readouterr().err.lower()

    def test_a_second_sweep_over_a_clean_world_deletes_nothing(
        self, monkeypatch, capsys,
    ):
        mod = _mod()
        clients = _install_fake_qdrant(monkeypatch, LIVE_COLLECTIONS)

        mod.main()
        mod.main()

        assert [client.deleted for client in clients] == [[], []]
        capsys.readouterr()


class TestReportingIsQuietWhenThereIsNothingToSay:
    """A cron job that prints on every no-op trains its reader to ignore it."""

    def test_it_says_nothing_on_stdout_when_nothing_was_stale(
        self, monkeypatch, capsys,
    ):
        _install_fake_qdrant(monkeypatch, LIVE_COLLECTIONS)

        _mod().main()

        assert capsys.readouterr().out == ''

    def test_it_reports_the_number_it_actually_deleted(self, monkeypatch, capsys):
        """`'2' in out` would pass on any stdout containing the character —
        a collection name with a 2 in it (`_test_e2_bakeoff_` has one), a
        future `took 2.3s` line, a uuid fragment — so a wrong count would
        very likely still pass.  Two runs of DIFFERENT sizes, each required
        to report its own number and not the other's, cannot.

        Suffixes are letters only and the sizes are 3 and 7 so no digit can
        arrive incidentally from a name.
        """
        import re  # noqa: PLC0415

        mod = _mod()
        letters = 'abcdefghijk'
        reported = {}
        for size in (3, 7):
            stale = [f'{mod.PREFIX}{c}' for c in letters[:size - 1]]
            stale.append(f'{mod.E2_BAKEOFF_PREFIX}z')
            clients = _install_fake_qdrant(
                monkeypatch, [*stale, *LIVE_COLLECTIONS],
            )

            mod.main()

            out = capsys.readouterr().out
            assert sum(len(client.deleted) for client in clients) == size
            reported[size] = {int(tok) for tok in re.findall(r'\d+', out)}

        assert 3 in reported[3] and 7 not in reported[3], reported[3]
        assert 7 in reported[7] and 3 not in reported[7], reported[7]


class TestTheSweepHoldsOffWhileARunIsLive:
    """The hazard this task exists to close.

    A live ``-m integration`` run or an E2 bake-off seeds its corpus under
    ``E2_BAKEOFF_PREFIX``, which is a prefix this cron deletes.  Unguarded,
    a sweep landing between that run's seed and measure phases deletes the
    corpus out from under it, and leaves nothing behind pointing at the
    reaper — the run simply measures a world that quietly emptied.

    A live lease holds off the ENTIRE sweep, not just the collections that
    run happens to own.  A per-name guard would need the lease to enumerate
    what it protects and the reaper to trust that list; it cannot, because
    runs grow their collection set mid-setup and a future driver adds names
    the lease format knows nothing about.  A deferred sweep costs nothing —
    it runs again in six hours over debris nothing depends on — while a
    wrong delete corrupts a live experiment.  With the error directions that
    asymmetric, the guard stays one cell wide and fails toward not deleting.
    """

    STALE = ('{prefix}dark_factory', '{bakeoff}_c_peers_main')

    def _stale(self, mod) -> list[str]:
        return [
            name.format(prefix=mod.PREFIX, bakeoff=mod.E2_BAKEOFF_PREFIX)
            for name in self.STALE
        ]

    def test_it_deletes_nothing_and_never_constructs_a_client(
        self, monkeypatch, capsys,
    ):
        """Both halves in one assertion, and the stronger one is the client:
        a held-off sweep must cost ZERO network, so there is no client for a
        delete to have gone through in the first place."""
        mod = _mod()
        clients = _install_fake_qdrant(
            monkeypatch, [*self._stale(mod), *LIVE_COLLECTIONS],
        )

        with mod.hold_lease(owner='e2-bake-off gw0'):
            mod.main()

        assert clients == []
        capsys.readouterr()

    def test_it_names_the_holder_and_its_pid_on_stderr(self, monkeypatch, capsys):
        """An operator reading cron mail has to be able to attribute the
        hold-off to a specific run rather than guess at one."""
        mod = _mod()
        _install_fake_qdrant(monkeypatch, [*self._stale(mod), *LIVE_COLLECTIONS])

        with mod.hold_lease(owner='e2-bake-off gw0'):
            mod.main()

        captured = capsys.readouterr()
        assert 'e2-bake-off gw0' in captured.err
        assert str(os.getpid()) in captured.err

    def test_it_says_nothing_on_stdout_because_it_deleted_nothing(
        self, monkeypatch, capsys,
    ):
        """stdout stays the deletion report and only that — a cron job that
        prints on every no-op trains its reader to ignore it."""
        mod = _mod()
        _install_fake_qdrant(monkeypatch, [*self._stale(mod), *LIVE_COLLECTIONS])

        with mod.hold_lease(owner='e2-bake-off gw0'):
            mod.main()

        assert capsys.readouterr().out == ''

    def test_it_still_returns_normally(self, monkeypatch, capsys):
        """The standing contract: always exits 0, idempotent.  Returning at
        all IS the assertion — a cron job that raised would page someone."""
        mod = _mod()
        _install_fake_qdrant(monkeypatch, [*self._stale(mod), *LIVE_COLLECTIONS])

        with mod.hold_lease(owner='e2-bake-off gw0'):
            mod.main()

        capsys.readouterr()

    def test_the_same_collections_are_reaped_when_no_lease_is_held(
        self, monkeypatch, capsys,
    ):
        """The complement, and the reason the tests above mean anything: a
        guard that had simply disabled the sweep would pass every one of
        them."""
        mod = _mod()
        stale = self._stale(mod)
        clients = _install_fake_qdrant(monkeypatch, [*stale, *LIVE_COLLECTIONS])

        mod.main()

        assert sorted(clients[0].deleted) == sorted(stale)
        capsys.readouterr()


@pytest.mark.parametrize('suffix', ['main', 'gw0', 'gw11'])
def test_every_worker_suffix_produces_a_reapable_collection(suffix):
    """The per-xdist-worker project id must not be able to dodge the prefix."""
    mod = _mod()

    for name in _bake_off().ephemeral_collections(suffix=suffix).values():
        assert any(name.startswith(prefix) for prefix in mod.PREFIXES)
