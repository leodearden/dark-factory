"""Tests for orchestrator.fleet_heartbeat — the fleet-common per-unit merge-idle
heartbeat producer (task 2395, α of the fleet-redeploy PRD).

Covers the pure module functions in isolation (no Harness required):
  - ``DEFAULT_FLEET_DIR`` / ``resolve_fleet_dir`` — fleet-common directory resolution.
  - ``build_heartbeat_payload`` — the five-field on-disk payload shape.
  - ``write_heartbeat`` — the atomic tmp-file + os.replace writer.

This is the SAME module the future reader (γ drain gate, ε --report) will
import, so its on-disk contract is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.fleet_heartbeat import (
    DEFAULT_FLEET_DIR,
    build_heartbeat_payload,
    resolve_fleet_dir,
    write_heartbeat,
)

# ---------------------------------------------------------------------------
# DEFAULT_FLEET_DIR / resolve_fleet_dir
# ---------------------------------------------------------------------------


class TestDefaultFleetDir:
    """DEFAULT_FLEET_DIR matches the watchdog's hardcoded REPO_DIR + /data/fleet."""

    def test_default_fleet_dir_matches_watchdog_repo_dir(self):
        """Pins the constant to scripts/orchestrator-watchdog.py's REPO_DIR (task 2395 analysis)."""
        assert Path('/home/leo/src/dark-factory/data/fleet') == DEFAULT_FLEET_DIR


class TestResolveFleetDir:
    """resolve_fleet_dir(env) — ORCH_FLEET_DIR override with a hardcoded default."""

    def test_returns_default_when_env_unset(self):
        """No ORCH_FLEET_DIR key in the mapping → DEFAULT_FLEET_DIR."""
        assert resolve_fleet_dir({}) == DEFAULT_FLEET_DIR

    def test_returns_default_when_env_empty(self):
        """ORCH_FLEET_DIR present but empty string → falls back to DEFAULT_FLEET_DIR."""
        assert resolve_fleet_dir({'ORCH_FLEET_DIR': ''}) == DEFAULT_FLEET_DIR

    def test_returns_path_of_env_value_when_set(self):
        """ORCH_FLEET_DIR set and non-empty → Path(ORCH_FLEET_DIR), via an explicit env mapping."""
        assert resolve_fleet_dir({'ORCH_FLEET_DIR': '/tmp/custom-fleet-dir'}) == Path(
            '/tmp/custom-fleet-dir'
        )

    def test_defaults_to_os_environ_and_honours_monkeypatch(self, monkeypatch):
        """With no env arg, resolve_fleet_dir reads the real os.environ (monkeypatch-able)."""
        monkeypatch.setenv('ORCH_FLEET_DIR', '/tmp/monkeypatched-fleet-dir')

        assert resolve_fleet_dir() == Path('/tmp/monkeypatched-fleet-dir')

    def test_defaults_to_os_environ_unset(self, monkeypatch):
        """With no env arg and ORCH_FLEET_DIR unset in the real environment → DEFAULT_FLEET_DIR."""
        monkeypatch.delenv('ORCH_FLEET_DIR', raising=False)

        assert resolve_fleet_dir() == DEFAULT_FLEET_DIR


# ---------------------------------------------------------------------------
# build_heartbeat_payload
# ---------------------------------------------------------------------------


class TestBuildHeartbeatPayload:
    """build_heartbeat_payload(...) — the five-field on-disk payload shape."""

    def test_returns_exactly_five_fields_with_type_fidelity(self):
        """Payload has exactly {unit, merge_idle, depth, queue_empty, ts_epoch}, values/types preserved."""
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=1234567890.5,
        )

        assert set(payload.keys()) == {'unit', 'merge_idle', 'depth', 'queue_empty', 'ts_epoch'}
        assert payload['unit'] == 'orchestrator-reify.service'
        assert isinstance(payload['unit'], str)
        assert payload['merge_idle'] is True
        assert isinstance(payload['merge_idle'], bool)
        assert payload['depth'] == 0
        assert isinstance(payload['depth'], int)
        assert payload['queue_empty'] is True
        assert isinstance(payload['queue_empty'], bool)
        assert payload['ts_epoch'] == 1234567890.5
        assert isinstance(payload['ts_epoch'], float)

    def test_busy_values_preserved(self):
        """A busy/non-idle tick's values pass through unchanged (no truthy coercion)."""
        payload = build_heartbeat_payload(
            unit='orchestrator-dark-factory.service',
            merge_idle=False,
            depth=3,
            queue_empty=False,
            ts_epoch=42.0,
        )

        assert payload == {
            'unit': 'orchestrator-dark-factory.service',
            'merge_idle': False,
            'depth': 3,
            'queue_empty': False,
            'ts_epoch': 42.0,
        }


# ---------------------------------------------------------------------------
# write_heartbeat
# ---------------------------------------------------------------------------


class TestWriteHeartbeat:
    """write_heartbeat(fleet_dir, unit, payload) — atomic tmp-file + os.replace writer."""

    def test_writes_unit_json_and_creates_missing_parent_dirs(self, tmp_path):
        """Writes <fleet_dir>/<unit>.json, creating missing nested parent dirs."""
        fleet_dir = tmp_path / 'deep' / 'nested' / 'fleet'
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=111.0,
        )

        result = write_heartbeat(fleet_dir, 'orchestrator-reify.service', payload)

        expected = fleet_dir / 'orchestrator-reify.service.json'
        assert expected.exists(), (
            'Expected heartbeat file to be created, including missing parent dirs'
        )
        assert result == expected

    def test_written_content_round_trips_the_exact_payload(self, tmp_path):
        """File content json.loads back to the exact payload passed in."""
        payload = build_heartbeat_payload(
            unit='orchestrator-dark-factory.service',
            merge_idle=False,
            depth=2,
            queue_empty=False,
            ts_epoch=222.5,
        )

        result = write_heartbeat(tmp_path, 'orchestrator-dark-factory.service', payload)

        assert json.loads(result.read_text()) == payload

    def test_no_leftover_tmp_file_after_write(self, tmp_path):
        """No <unit>.json.tmp remains after the atomic rename."""
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=333.0,
        )

        write_heartbeat(tmp_path, 'orchestrator-reify.service', payload)

        assert not (tmp_path / 'orchestrator-reify.service.json.tmp').exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ['orchestrator-reify.service.json']

    def test_returns_the_final_path(self, tmp_path):
        """Returns the final on-disk Path (not the tmp path)."""
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=444.0,
        )

        result = write_heartbeat(tmp_path, 'orchestrator-reify.service', payload)

        assert result == tmp_path / 'orchestrator-reify.service.json'


class TestMalformedUnitIsRefused:
    """A malformed unit name must never produce a heartbeat file (task 3951).

    The unit-level half of one behaviour; the producer-level half is
    ``test_harness_merge_heartbeat.py``'s ``test_empty_unit_writes_nothing_and_
    is_logged_not_raised``.

    ONE RULE, TWO REJECTIONS.  The unit name is the heartbeat's identity AND is
    interpolated straight into the destination path, so it must be a non-blank,
    single path component:

    BLANK replaces the former ``unknown-unit.json`` fallback, which was written
    to make an unresolved unit produce a deterministic filename rather than a
    file literally named ``.json``.  That trade was wrong in the direction it
    chose: nothing READS ``unknown-unit.json`` (``scripts/drain_check.py``
    addresses heartbeats BY NAME via ``heartbeat_path(fleet_dir, unit)`` and
    never enumerates the directory), so the fallback bought no consumer
    anything while quietly turning "this writer has no unit name" — a real
    misconfiguration — into a plausible-looking file in a machine-global,
    cross-project directory.  Raising makes the next unnamed writer loud at the
    moment it appears.

    NOT-A-BARE-FILENAME is the same defect class one degree worse: the value
    reaches this function from the ambient environment (measured during this
    task as inherited, and sometimes wrong), and a ``..`` component or an
    absolute path does not merely mislabel a file inside the fleet dir — it
    writes clean outside it.

    Matched on ``unit name``, deliberately NOT on ``ORCH_UNIT``:
    ``write_heartbeat`` never reads that variable (its caller does), and this
    module is the shared on-disk contract for future producers that may resolve
    their unit from config instead.

    Every case asserts the fleet dir was never CREATED, not merely left empty —
    the guard has to run before ``safe_io.atomic_write_text(..., mkdir=True)``,
    and directory-absence is the one assertion that catches a guard which moved
    after it.  It also subsumes the final name, the retired
    ``unknown-unit.json``, and any ``.json.tmp`` residue from a partial write.
    """

    @staticmethod
    def _payload(unit: str):
        return build_heartbeat_payload(
            unit=unit,
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=555.0,
        )

    def test_empty_unit_raises_and_writes_nothing(self, tmp_path):
        """unit='' raises ValueError naming the unit name, and creates NO file."""
        fleet_dir = tmp_path / 'fleet'

        with pytest.raises(ValueError, match='unit name'):
            write_heartbeat(fleet_dir, '', self._payload(''))

        assert not fleet_dir.exists()
        assert list(tmp_path.iterdir()) == []

    def test_whitespace_only_unit_raises_and_writes_nothing(self, tmp_path):
        """unit='   ' is the same defect: rejected, never stripped into a name.

        ``unit if unit else …`` treated a whitespace-only unit as truthy and
        would have written a file literally named ``   .json`` — an equally
        corrupt artifact in the same directory, from the same defect class (a
        writer that reached production without a real unit set).  Silently
        repairing it into a plausible name is the silent-degradation this guard
        exists to end; the caller learns its unit name is malformed instead.
        """
        fleet_dir = tmp_path / 'fleet'

        with pytest.raises(ValueError, match='unit name'):
            write_heartbeat(fleet_dir, '   ', self._payload('   '))

        assert not fleet_dir.exists()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        'unit',
        [
            'a/b',
            '../escaped',
            '../../tmp/x',
            '/abs/unit',
            '.',
            '..',
        ],
    )
    def test_unit_that_is_not_a_bare_filename_raises_and_writes_nothing(
        self, tmp_path, unit
    ):
        """A separator, a traversal or an absolute unit escapes the fleet dir.

        ``fleet_dir / f'{unit}.json'`` is plain interpolation, VERIFIED:
        ``'../escaped'`` lands one level ABOVE the fleet dir and ``'/abs/unit'``
        discards *fleet_dir* entirely, yielding ``/abs/unit.json``
        (``Path.__truediv__`` with an absolute right operand).  ``'.'``/``'..'``
        are the degenerate spellings caught by the same rule — both have an
        empty ``Path.name`` — and would write the hidden, unattributable
        ``<fleet_dir>/..json`` / ``<fleet_dir>/...json``.
        """
        fleet_dir = tmp_path / 'fleet'

        with pytest.raises(ValueError, match='unit name'):
            write_heartbeat(fleet_dir, unit, self._payload(unit))

        assert not fleet_dir.exists()
        # Covers the sibling-escape target too: '../escaped' would have landed
        # at tmp_path/'escaped.json', outside fleet_dir but inside tmp_path.
        assert list(tmp_path.iterdir()) == []


class TestDelegatesToSharedAtomicWriter:
    """``fleet_heartbeat.write_heartbeat`` delegates to ``shared.safe_io.atomic_write_text``.

    Task 3223 consolidated the repo's tmp+rename writers into ``shared.safe_io``,
    which also gives this site a unique-per-writer temp name in place of the old
    fixed ``<dest>.json.tmp`` (two concurrent writers used to share it).
    ``mode`` must stay at the umask default: this file is read by other
    processes (the dashboard, the gamma/epsilon watchers, scripts/drain_check.py),
    so narrowing it to 0o600 is the specific silent regression this task avoids.
    """

    @staticmethod
    def _recorder(monkeypatch):
        import shared.safe_io as _safe_io

        calls = []
        real = _safe_io.atomic_write_text

        def recorder(path, text, **kwargs):
            calls.append((path, text, kwargs))
            return real(path, text, **kwargs)

        monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
        return calls

    @staticmethod
    def _assert_common(kwargs):
        assert kwargs.get('mkdir') is True, 'this site created its parent dir'
        assert kwargs.get('encoding') == 'utf-8'
        assert not kwargs.get('fsync'), 'this site never fsynced'
        assert kwargs.get('mode') is None, (
            'umask default, NOT 0o600 — this file is read by other processes'
        )

    def test_delegates_with_preserved_semantics(self, tmp_path: Path, monkeypatch) -> None:
        calls = self._recorder(monkeypatch)
        write_heartbeat(tmp_path / 'fleet', 'orchestrator-df.service', {'ts': 1})

        assert len(calls) == 1, f'expected exactly one delegated call, got {calls}'
        self._assert_common(calls[0][2])

    def test_on_disk_mode_matches_write_text_reference(self, tmp_path: Path) -> None:
        reference = tmp_path / 'reference.json'
        reference.write_text('ref', encoding='utf-8')
        path = write_heartbeat(tmp_path, 'orchestrator-df.service', {'ts': 1})
        assert path.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777

    def test_oserror_still_propagates(self, tmp_path: Path, monkeypatch) -> None:
        """This site propagates — it has no fail-open boundary."""
        import shared.safe_io as _safe_io

        def boom(*_a, **_kw):
            raise OSError('disk full')

        monkeypatch.setattr(_safe_io, 'atomic_write_text', boom)
        with pytest.raises(OSError, match='disk full'):
            write_heartbeat(tmp_path, 'orchestrator-df.service', {'ts': 1})
