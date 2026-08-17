"""Contract tests for ``fused_memory.utils.store_mutation_preflight`` (task 3686).

Background — the measured failure this guard exists to prevent
--------------------------------------------------------------
``scripts/sweep_toolcall_xml_leak.py --apply`` was run from inside a sandboxed
agent session. Its repair is delete-then-re-add, and the two halves live on
different substrates:

  * the Qdrant delete is a NETWORK call to ``localhost:6333`` — landlock does
    not govern the network, so it SUCCEEDED;
  * mem0's history write is a SQLite write under ``~/.mem0`` — a filesystem
    operation, so it was DENIED.

The mutation split down the middle: memory ``7d073281-4c5d-4ba3-a01c-3a167f4460f4``
was removed from the store and never came back. The only safe place to fail is
BEFORE the first mutation, which is what this module's preflight does.

Hermetic by construction: ``tmp_path`` + ``monkeypatch`` only. Never touches a
real ``~/.mem0``, never opens a socket, never imports mem0 (that import costs
~13s and pulls in the whole embeddings/vector-store stack — the resolution
seam is patched instead).
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
    resolve_history_dir,
)

_RESOLVER_SEAM = 'fused_memory.utils.store_mutation_preflight._mem0_configured_history_db_path'
_CREATE_SEAM = 'fused_memory.utils.store_mutation_preflight._create_probe_file'


def _deny_creation(*_args, **_kwargs):
    """Stand in for the measured refusal: EACCES on creating a NEW file.

    This is the exact observed shape — not a stat/mode failure. The real
    ``history.db`` was 0644/uid-1000 with ``os.access(W_OK) is True`` and even
    accepted ``BEGIN IMMEDIATE``, while ``open('~/.mem0/.write-probe', 'w')``
    raised EACCES, because a syscall filter denied the write the directory's
    own mode permitted.
    """
    raise PermissionError(errno.EACCES, 'Permission denied')


class TestResolveHistoryDir:
    """The probed directory is mem0's CONFIGURED one, never a hardcoded path.

    Load-bearing: ``mem0.configs.base`` computes
    ``mem0_dir = os.environ.get('MEM0_DIR') or os.path.join(home_dir, '.mem0')``
    and derives ``history_db_path`` from it. A guard that hardcoded
    ``~/.mem0`` would probe a directory mem0 is not using and pass vacuously —
    reporting "mutation is safe" about the wrong filesystem location.
    """

    def test_follows_mem0_dir_env_var(self, tmp_path, monkeypatch):
        history_dir = tmp_path / 'mem0-home'
        history_dir.mkdir()
        monkeypatch.setenv('MEM0_DIR', str(history_dir))

        assert resolve_history_dir() == history_dir

    def test_follows_the_configured_history_db_path_when_env_is_unset(
        self, tmp_path, monkeypatch
    ):
        """With no ``MEM0_DIR``, resolution defers to mem0's own configured
        ``history_db_path`` — the parent directory of that file, not a guess."""
        monkeypatch.delenv('MEM0_DIR', raising=False)
        configured = tmp_path / 'configured-home' / 'history.db'
        monkeypatch.setattr(_RESOLVER_SEAM, lambda: str(configured))

        assert resolve_history_dir() == configured.parent

    def test_is_not_hardcoded_to_the_home_mem0_directory(self, tmp_path, monkeypatch):
        """The anti-vacuity assertion: under a redirected config the resolved
        directory must NOT be ``~/.mem0``."""
        monkeypatch.setenv('MEM0_DIR', str(tmp_path / 'elsewhere'))

        assert resolve_history_dir() != Path.home() / '.mem0'


class TestAssertStoreMutationAllowed:
    """Fail-closed capability probe: a REAL create attempt, cleaned up after."""

    def test_returns_normally_when_creation_is_permitted(self, tmp_path, monkeypatch):
        monkeypatch.setenv('MEM0_DIR', str(tmp_path))

        assert_store_mutation_allowed(operation='unit-test') is None  # noqa: B015

    def test_leaves_no_probe_file_behind(self, tmp_path, monkeypatch):
        """The probe is scratch, not residue: a guard that littered the mem0
        history directory on every run would be its own operational problem."""
        monkeypatch.setenv('MEM0_DIR', str(tmp_path))
        (tmp_path / 'history.db').write_text('pretend-sqlite')
        before = sorted(p.name for p in tmp_path.iterdir())

        assert_store_mutation_allowed(operation='unit-test')

        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_raises_when_file_creation_is_denied(self, tmp_path, monkeypatch):
        monkeypatch.setenv('MEM0_DIR', str(tmp_path))
        monkeypatch.setattr(_CREATE_SEAM, _deny_creation)

        with pytest.raises(StoreMutationUnavailable):
            assert_store_mutation_allowed(operation='unit-test')

    def test_the_refusal_message_names_the_directory_the_error_and_the_operation(
        self, tmp_path, monkeypatch
    ):
        """An operator must be able to triage from the message alone, without
        re-deriving the root cause from scratch."""
        monkeypatch.setenv('MEM0_DIR', str(tmp_path))
        monkeypatch.setattr(_CREATE_SEAM, _deny_creation)

        with pytest.raises(StoreMutationUnavailable) as excinfo:
            assert_store_mutation_allowed(operation='sweep_toolcall_xml_leak --apply')

        message = str(excinfo.value)
        assert str(tmp_path) in message
        assert 'sweep_toolcall_xml_leak --apply' in message
        assert 'Permission denied' in message
        # The original OSError is chained, not discarded.
        assert isinstance(excinfo.value.__cause__, PermissionError)

    def test_denial_is_about_creation_not_the_db_files_own_mode(
        self, tmp_path, monkeypatch
    ):
        """A directory that exists, lists, and reports ``os.access(W_OK) is
        True`` must STILL raise when creation is refused.

        This pins the exact measured condition and rules out the tempting
        cheap check: an ``os.access``/stat-mode guard would have passed here
        and waved the half-mutation through.
        """
        monkeypatch.setenv('MEM0_DIR', str(tmp_path))
        (tmp_path / 'history.db').write_text('pretend-sqlite')
        assert list(tmp_path.iterdir())  # exists and is listable
        assert os.access(tmp_path, os.W_OK)  # and os.access says "writable"
        monkeypatch.setattr(_CREATE_SEAM, _deny_creation)

        with pytest.raises(StoreMutationUnavailable):
            assert_store_mutation_allowed(operation='unit-test')

    def test_a_missing_history_directory_raises_rather_than_passing_silently(
        self, tmp_path, monkeypatch
    ):
        """Loud over silent. An absent directory is an unknown environment, not
        a permissive one — the inverse of the silent-no-op failure mode."""
        missing = tmp_path / 'does-not-exist'
        monkeypatch.setenv('MEM0_DIR', str(missing))
        assert not missing.exists()

        with pytest.raises(StoreMutationUnavailable) as excinfo:
            assert_store_mutation_allowed(operation='unit-test')

        assert str(missing) in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)


class TestFailClosedPrecedent:
    """The exception mirrors ``RemediationSandboxUnavailable``'s posture."""

    def test_is_a_runtime_error(self):
        """Same base as the in-repo fail-closed precedent
        (``reconciliation.sandbox_guard.RemediationSandboxUnavailable``), so an
        existing broad ``except Exception`` refusal handler catches it too."""
        assert issubclass(StoreMutationUnavailable, RuntimeError)
