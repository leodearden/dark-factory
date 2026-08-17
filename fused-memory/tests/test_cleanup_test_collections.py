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

import functools
import importlib.util
import sys
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


@pytest.mark.parametrize('suffix', ['main', 'gw0', 'gw11'])
def test_every_worker_suffix_produces_a_reapable_collection(suffix):
    """The per-xdist-worker project id must not be able to dodge the prefix."""
    mod = _mod()

    for name in _bake_off().ephemeral_collections(suffix=suffix).values():
        assert any(name.startswith(prefix) for prefix in mod.PREFIXES)
