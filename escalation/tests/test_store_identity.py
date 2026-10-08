"""Contract tests for ``escalation.store_identity.StoreIdentity``.

Pins the PRD §6.1 dataclass contract from
``plans/escalation-store-ambiguity-prd.md`` (task γ1).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from escalation.store_identity import StoreIdentity


def _project_identity(tmp_path: Path) -> StoreIdentity:
    return StoreIdentity(
        kind='project',
        queue_dir=tmp_path / 'esc',
        project_id='test-project',
        project_root=tmp_path,
    )


def _reconciliation_identity(tmp_path: Path) -> StoreIdentity:
    return StoreIdentity(
        kind='reconciliation',
        queue_dir=tmp_path / 'esc',
        project_id=None,
        project_root=None,
    )


class TestStoreIdentityFields:
    def test_project_identity_exposes_passed_values(self, tmp_path: Path) -> None:
        si = _project_identity(tmp_path)

        assert si.kind == 'project'
        assert si.queue_dir == tmp_path / 'esc'
        assert si.project_id == 'test-project'
        assert si.project_root == tmp_path

    def test_reconciliation_identity_carries_no_project(self, tmp_path: Path) -> None:
        si = _reconciliation_identity(tmp_path)

        assert si.kind == 'reconciliation'
        assert si.queue_dir == tmp_path / 'esc'
        assert si.project_id is None
        assert si.project_root is None


class TestStoreIdentityIsFrozen:
    def test_is_a_dataclass(self, tmp_path: Path) -> None:
        assert dataclasses.is_dataclass(_project_identity(tmp_path))

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('kind', 'reconciliation'),
            ('queue_dir', Path('/elsewhere')),
            ('project_id', 'other-project'),
            ('project_root', Path('/elsewhere')),
        ],
    )
    def test_every_field_rejects_assignment(
        self, tmp_path: Path, field: str, value: object
    ) -> None:
        si = _project_identity(tmp_path)

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(si, field, value)
