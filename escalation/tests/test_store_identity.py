"""Contract tests for ``escalation.store_identity.StoreIdentity``.

Pins the PRD §6.1 dataclass contract from
``plans/escalation-store-ambiguity-prd.md`` (task γ1).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from escalation.queue import EscalationQueue
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


class TestStoreIdentityNormalizesPaths:
    def test_relative_queue_dir_becomes_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        si = StoreIdentity(
            kind='reconciliation', queue_dir=Path('esc'), project_id=None, project_root=None
        )

        assert si.queue_dir.is_absolute()
        assert si.queue_dir == (tmp_path / 'esc').resolve()

    def test_dotdot_in_queue_dir_is_collapsed(self, tmp_path: Path) -> None:
        raw = tmp_path / 'sub' / '..' / 'esc'

        si = StoreIdentity(kind='reconciliation', queue_dir=raw, project_id=None, project_root=None)

        assert si.queue_dir == raw.resolve()
        assert '..' not in si.queue_dir.parts

    def test_symlink_in_queue_dir_is_resolved(self, tmp_path: Path) -> None:
        (tmp_path / 'real').mkdir()
        (tmp_path / 'link').symlink_to(tmp_path / 'real')
        raw = tmp_path / 'link' / 'esc'

        si = StoreIdentity(kind='reconciliation', queue_dir=raw, project_id=None, project_root=None)

        assert si.queue_dir == raw.resolve()
        assert si.queue_dir == (tmp_path / 'real' / 'esc').resolve()

    def test_project_root_is_resolved(self, tmp_path: Path) -> None:
        (tmp_path / 'real').mkdir()
        (tmp_path / 'link').symlink_to(tmp_path / 'real')
        raw_root = tmp_path / 'link' / 'sub' / '..'

        si = StoreIdentity(
            kind='project',
            queue_dir=tmp_path / 'esc',
            project_id='test-project',
            project_root=raw_root,
        )

        assert si.project_root == raw_root.resolve()
        assert si.project_root == (tmp_path / 'real').resolve()

    def test_resolved_absolute_paths_are_preserved(self, tmp_path: Path) -> None:
        queue_dir = (tmp_path / 'esc').resolve()
        project_root = tmp_path.resolve()

        si = StoreIdentity(
            kind='project',
            queue_dir=queue_dir,
            project_id='test-project',
            project_root=project_root,
        )

        assert si.queue_dir == queue_dir
        assert si.project_root == project_root

    def test_queue_dir_off_an_escalation_queue_lands_resolved(self, tmp_path: Path) -> None:
        (tmp_path / 'real').mkdir()
        (tmp_path / 'link').symlink_to(tmp_path / 'real')
        queue = EscalationQueue(tmp_path / 'link' / 'esc')
        assert queue.queue_dir != queue.queue_dir.resolve()

        si = StoreIdentity(
            kind='reconciliation', queue_dir=queue.queue_dir, project_id=None, project_root=None
        )

        assert si.queue_dir == queue.queue_dir.resolve()


class TestStoreIdentityValidatesKindCoherence:
    @pytest.mark.parametrize(
        ('kind', 'project_id', 'project_root', 'offending_field'),
        [
            ('project', None, Path('/repo'), 'project_id'),
            ('project', 'test-project', None, 'project_root'),
            ('reconciliation', 'test-project', None, 'project_id'),
            ('reconciliation', None, Path('/repo'), 'project_root'),
        ],
    )
    def test_incoherent_combination_raises_naming_kind_and_field(
        self,
        tmp_path: Path,
        kind: str,
        project_id: str | None,
        project_root: Path | None,
        offending_field: str,
    ) -> None:
        with pytest.raises(ValueError, match=rf"kind='{kind}'.*\b{offending_field}\b"):
            StoreIdentity(
                kind=kind,  # type: ignore[arg-type]
                queue_dir=tmp_path / 'esc',
                project_id=project_id,
                project_root=project_root,
            )

    def test_coherent_combinations_construct(self, tmp_path: Path) -> None:
        _project_identity(tmp_path)
        _reconciliation_identity(tmp_path)
