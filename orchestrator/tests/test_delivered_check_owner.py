"""Tests for ``orchestrator.delivered_check_owner`` (task 7514, absorbing 4934).

The module answers one question for a ``DEP_CAPABILITY_NOT_DELIVERED`` record:
which task owns the failing ``metadata.delivered_checks`` entry, and where
else is it declared. ``derive_check_owner`` and ``render_check_owner`` are
pure; ``verify_check_owner`` reads the owner's sidecar from ``main`` of a real
git repo.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from escalation.models import DeliveredCheckOwner, SidecarOnMain
from shared.capability_manifest import sidecar_path

from orchestrator.delivered_check_owner import (
    derive_check_owner,
    render_check_owner,
    verify_check_owner,
)

_PRD_PATH = 'docs/prds/v0_6/eradicate-silent-undef.md'
_SIDECAR_PATH = 'docs/prds/v0_6/eradicate-silent-undef.capability-manifest.yaml'
_CHECK_NAME = 'pattern-module-substrate'
_DEP_ID = '5405'


def _git(root: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-b', 'main', str(root)], check=True, capture_output=True)
    _git(root, 'config', 'user.email', 'owner-test@example.com')
    _git(root, 'config', 'user.name', 'Owner Test')
    (root / 'README.md').write_text('# fixture\n', encoding='utf-8')
    _git(root, 'add', 'README.md')
    _git(root, 'commit', '-m', 'initial')
    return root


def _sidecar_doc(
    *,
    task_id: int | None = int(_DEP_ID),
    capability: str = _CHECK_NAME,
    with_check: bool = True,
) -> dict[str, Any]:
    capability_entry: dict[str, Any] = {
        'name': capability,
        'binding': 'grep for the capability token',
        'verdict': 'PASS',
    }
    if with_check:
        capability_entry['delivered_check'] = {
            'kind': 'grep',
            'pattern': 'TOKEN',
            'expect': 'present',
            'paths': ['src/lib.rs'],
        }
    return {
        'prd': _PRD_PATH,
        'schema_version': 1,
        'tasks': [
            {
                'label': 'alpha',
                'task_id': task_id,
                'title': 'Producer alpha',
                'capabilities': [capability_entry],
            },
        ],
    }


def _write_sidecar(root: Path, text: str) -> None:
    target = root / _SIDECAR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def _commit_sidecar(root: Path, text: str) -> None:
    _write_sidecar(root, text)
    _git(root, 'add', _SIDECAR_PATH)
    _git(root, 'commit', '-m', 'sidecar')


def _owner(on_main: SidecarOnMain = 'unverified') -> DeliveredCheckOwner:
    return {
        'task_id': _DEP_ID,
        'check_name': _CHECK_NAME,
        'sidecar': {'path': _SIDECAR_PATH, 'on_main': on_main},
    }


class TestDeriveCheckOwner:
    def test_a_prd_path_derives_the_sidecar_through_the_single_rule(self) -> None:
        owner = derive_check_owner(
            dep_id=_DEP_ID,
            dep_metadata={'prd_path': _PRD_PATH},
            check_name=_CHECK_NAME,
        )

        assert owner == {
            'task_id': _DEP_ID,
            'check_name': _CHECK_NAME,
            'sidecar': {'path': sidecar_path(_PRD_PATH), 'on_main': 'unverified'},
        }

    @pytest.mark.parametrize('metadata', [{}, {'prd_path': ''}, {'prd_path': None}, {'prd_path': 7}])
    def test_a_missing_empty_or_non_str_prd_path_leaves_no_sidecar(self, metadata) -> None:
        owner = derive_check_owner(dep_id=_DEP_ID, dep_metadata=metadata, check_name=_CHECK_NAME)

        assert owner['sidecar'] is None
        assert owner['task_id'] == _DEP_ID

    def test_the_task_id_is_the_dep_id_passed_in_never_the_metadata(self) -> None:
        owner = derive_check_owner(
            dep_id=_DEP_ID,
            dep_metadata={'prd_path': _PRD_PATH, 'task_id': '9999', 'id': '9999'},
            check_name=_CHECK_NAME,
        )

        assert owner['task_id'] == _DEP_ID


class TestVerifyCheckOwner:
    def test_a_committed_block_bound_to_the_owner_declares(self, tmp_path: Path) -> None:
        root = _init_repo(tmp_path / 'repo')
        _commit_sidecar(root, yaml.safe_dump(_sidecar_doc(), sort_keys=False))

        verified = verify_check_owner(_owner(), project_root=root)

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': 'declares'}
        assert verified['task_id'] == _DEP_ID
        assert verified['check_name'] == _CHECK_NAME

    @pytest.mark.parametrize(
        'doc',
        [
            _sidecar_doc(task_id=1234),
            _sidecar_doc(task_id=None),
            _sidecar_doc(capability='another-capability'),
            _sidecar_doc(with_check=False),
        ],
        ids=['other-task', 'unbound-block', 'other-capability', 'no-delivered-check'],
    )
    def test_a_sidecar_that_does_not_carry_the_check_does_not_declare(
        self, tmp_path: Path, doc: dict[str, Any]
    ) -> None:
        root = _init_repo(tmp_path / 'repo')
        _commit_sidecar(root, yaml.safe_dump(doc, sort_keys=False))

        verified = verify_check_owner(_owner(), project_root=root)

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': 'does_not_declare'}

    def test_an_absent_sidecar_does_not_declare(self, tmp_path: Path) -> None:
        root = _init_repo(tmp_path / 'repo')

        verified = verify_check_owner(_owner(), project_root=root)

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': 'does_not_declare'}

    def test_an_uncommitted_sidecar_is_not_read(self, tmp_path: Path) -> None:
        root = _init_repo(tmp_path / 'repo')
        _write_sidecar(root, yaml.safe_dump(_sidecar_doc(), sort_keys=False))

        verified = verify_check_owner(_owner(), project_root=root)

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': 'does_not_declare'}

    @pytest.mark.parametrize(
        ('on_main_doc', 'working_text', 'expected'),
        [
            (_sidecar_doc(), yaml.safe_dump(_sidecar_doc(task_id=1234)), 'declares'),
            (_sidecar_doc(), 'tasks: [unclosed\n', 'declares'),
            (_sidecar_doc(task_id=1234), yaml.safe_dump(_sidecar_doc()), 'does_not_declare'),
        ],
        ids=['declaring-main-dirty-other-task', 'declaring-main-dirty-malformed', 'other-main-dirty-declaring'],
    )
    def test_the_content_is_read_from_main_not_the_dirty_working_tree(
        self, tmp_path: Path, on_main_doc: dict[str, Any], working_text: str, expected: str
    ) -> None:
        root = _init_repo(tmp_path / 'repo')
        _commit_sidecar(root, yaml.safe_dump(on_main_doc, sort_keys=False))
        _write_sidecar(root, working_text)

        verified = verify_check_owner(_owner(), project_root=root)

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': expected}

    @pytest.mark.parametrize(
        'text',
        ['tasks: [unclosed\n', yaml.safe_dump({'prd': 'x', 'schema_version': 2, 'tasks': []})],
        ids=['malformed-yaml', 'schema-invalid'],
    )
    def test_an_unreadable_sidecar_on_main_is_unverified(self, tmp_path: Path, text: str) -> None:
        root = _init_repo(tmp_path / 'repo')
        _commit_sidecar(root, text)

        verified = verify_check_owner(_owner(), project_root=root)

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': 'unverified'}

    def test_a_non_git_root_is_unverified(self, tmp_path: Path) -> None:
        plain = tmp_path / 'not-a-repo'
        plain.mkdir()

        verified = verify_check_owner(_owner(), project_root=plain)

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': 'unverified'}

    def test_an_unresolvable_ref_is_unverified(self, tmp_path: Path) -> None:
        root = _init_repo(tmp_path / 'repo')
        _commit_sidecar(root, yaml.safe_dump(_sidecar_doc(), sort_keys=False))

        verified = verify_check_owner(_owner(), project_root=root, ref='no-such-branch')

        assert verified['sidecar'] == {'path': _SIDECAR_PATH, 'on_main': 'unverified'}

    def test_an_owner_without_a_sidecar_is_returned_unchanged(self, tmp_path: Path) -> None:
        owner: DeliveredCheckOwner = {'task_id': _DEP_ID, 'check_name': _CHECK_NAME, 'sidecar': None}
        before = copy.deepcopy(owner)

        verified = verify_check_owner(owner, project_root=tmp_path)

        assert verified == before
        assert owner == before

    def test_the_input_owner_is_never_mutated(self, tmp_path: Path) -> None:
        root = _init_repo(tmp_path / 'repo')
        _commit_sidecar(root, yaml.safe_dump(_sidecar_doc(), sort_keys=False))
        owner = _owner()
        before = copy.deepcopy(owner)

        verify_check_owner(owner, project_root=root)

        assert owner == before


_GOLDEN_BLOCK = """\
Check owner: task 5405, the dependency. The blocked task does not declare this check.
Declared in: task 5405 metadata.delivered_checks, entry 'pattern-module-substrate'.
Sidecar: docs/prds/v0_6/eradicate-silent-undef.capability-manifest.yaml declares it too on main; a correction edits both.
Remedy: if the capability is already on main, the check is stale or mis-authored; correct it at task 5405's sites above. If the capability is missing from main, the gap is task 5405's.
Blocked dependents are NOT auto-re-pended — set task 5542 back to pending once the check passes on main."""


class TestRenderCheckOwner:
    def test_the_esc_5542_7_owner_renders_the_golden_block(self) -> None:
        owner = _owner('declares')

        assert render_check_owner(owner, dependent_id='5542') == _GOLDEN_BLOCK

    @pytest.mark.parametrize(
        ('sidecar', 'expected_line'),
        [
            (
                {'path': _SIDECAR_PATH, 'on_main': 'declares'},
                f'Sidecar: {_SIDECAR_PATH} declares it too on main; a correction edits both.',
            ),
            (
                {'path': _SIDECAR_PATH, 'on_main': 'does_not_declare'},
                f"Sidecar: {_SIDECAR_PATH} (from task 5405's prd_path) does not declare it on "
                'main; the task metadata is the only site.',
            ),
            (
                {'path': _SIDECAR_PATH, 'on_main': 'unverified'},
                f"Sidecar: {_SIDECAR_PATH} (from task 5405's prd_path) could not be read on "
                'main; check it before editing.',
            ),
            (
                None,
                'Sidecar: none (task 5405 has no prd_path); the task metadata is the only site.',
            ),
        ],
        ids=['declares', 'does_not_declare', 'unverified', 'no-sidecar'],
    )
    def test_each_sidecar_state_renders_its_own_line(self, sidecar, expected_line) -> None:
        owner: DeliveredCheckOwner = {
            'task_id': _DEP_ID,
            'check_name': _CHECK_NAME,
            'sidecar': sidecar,
        }

        lines = render_check_owner(owner, dependent_id='5542').splitlines()

        assert lines[2] == expected_line
        assert len(lines) == 5

    def test_no_owner_renders_only_the_re_pend_line(self) -> None:
        rendered = render_check_owner(None, dependent_id='5542')

        assert rendered == (
            'Blocked dependents are NOT auto-re-pended — set task 5542 back to pending '
            'once the check passes on main.'
        )

    @pytest.mark.parametrize(
        'sidecar',
        [
            {'path': _SIDECAR_PATH, 'on_main': 'declares'},
            {'path': _SIDECAR_PATH, 'on_main': 'does_not_declare'},
            {'path': _SIDECAR_PATH, 'on_main': 'unverified'},
            None,
        ],
    )
    def test_the_dependent_appears_on_exactly_the_last_line(self, sidecar) -> None:
        owner: DeliveredCheckOwner = {
            'task_id': _DEP_ID,
            'check_name': _CHECK_NAME,
            'sidecar': sidecar,
        }

        for rendered in (
            render_check_owner(owner, dependent_id='5542'),
            render_check_owner(None, dependent_id='5542'),
        ):
            lines = rendered.splitlines()
            assert [i for i, line in enumerate(lines) if '5542' in line] == [len(lines) - 1]
