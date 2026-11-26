"""Tests for shared.project_id: the one project-id rule and the tolerant cockpit fold.

Contract: plans/review-all-s14-shared-leaves-prd.md decision 4 and invariant I2.
The fold table is ported from orchestrator/tests/test_session_registry.py
(``_PROJECT_TOKEN_CASES``) so the behaviour matches what session_registry will
repoint onto.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path, PurePath

import pytest

from shared.project_id import (
    PROJECT_ID_ALIASES,
    fold_project_token,
    normalize_project_id,
    project_id_for_root,
)

_CANONICAL_TOKEN = re.compile(r'[a-z0-9]+(_[a-z0-9]+)*')

_NORMALIZE_CASES = [
    ('Dark-Factory', 'dark_factory'),
    ('dark_factory', 'dark_factory'),
    ('Know-Live', 'know_live'),
    ('dark--factory', 'dark__factory'),
    ('REIFY', 'reify'),
    ('', ''),
    ('solar-challenge-platform', 'solar_challenge_platform'),
]

_FOLD_CASES: list[tuple[object, str]] = [
    ('dark_factory', 'dark_factory'),
    ('dark-factory', 'dark_factory'),
    ('Dark-Factory', 'dark_factory'),
    ('Dark_Factory', 'dark_factory'),
    ('DARK_FACTORY', 'dark_factory'),
    ('  dark-factory  ', 'dark_factory'),
    ('dark--factory', 'dark_factory'),
    ('dark-_factory', 'dark_factory'),
    ('df', 'dark_factory'),
    ('DF', 'dark_factory'),
    (' df ', 'dark_factory'),
    ('-df-', 'dark_factory'),
    ('', ''),
    ('   ', ''),
    ('\t\n', ''),
    (None, ''),
    (42, '42'),
    (3.5, '3.5'),
    ('reify', 'reify'),
    ('autopilot-video', 'autopilot_video'),
    ('autopilot_video', 'autopilot_video'),
    ('solar-challenge', 'solar_challenge'),
    ('solar_challenge', 'solar_challenge'),
]

_CANONICAL_CORPUS = [
    'dark_factory',
    'reify',
    'know_live',
    'autopilot_video',
    'solar_challenge',
    'my_solar_challenge',
    'solar_challenge_platform',
    'a1_b2_c3',
    'x',
]


class TestNormalizeProjectId:
    @pytest.mark.parametrize(('raw', 'expected'), _NORMALIZE_CASES)
    def test_table(self, raw: str, expected: str) -> None:
        assert normalize_project_id(raw) == expected

    @pytest.mark.parametrize(
        'raw',
        [raw for raw, _ in _NORMALIZE_CASES]
        + [raw for raw, _ in _FOLD_CASES if isinstance(raw, str)],
    )
    def test_idempotent(self, raw: str) -> None:
        once = normalize_project_id(raw)
        assert normalize_project_id(once) == once


class TestFoldProjectToken:
    @pytest.mark.parametrize(('raw', 'expected'), _FOLD_CASES)
    def test_table(self, raw: object, expected: str) -> None:
        assert fold_project_token(raw) == expected

    def test_never_merges_two_projects_sharing_a_prefix(self) -> None:
        platform = fold_project_token('solar_challenge_platform')
        assert platform == fold_project_token('solar-challenge-platform') == 'solar_challenge_platform'
        assert platform != fold_project_token('solar-challenge')

    @pytest.mark.parametrize('raw', [raw for raw, _ in _FOLD_CASES])
    def test_idempotent(self, raw: object) -> None:
        once = fold_project_token(raw)
        assert fold_project_token(once) == once


class TestFoldAgreesWithNormalizeOnCanonicalIds:
    @pytest.mark.parametrize('project_id', _CANONICAL_CORPUS)
    def test_agreement(self, project_id: str) -> None:
        assert _CANONICAL_TOKEN.fullmatch(project_id)
        assert project_id not in PROJECT_ID_ALIASES
        assert normalize_project_id(project_id) == project_id
        assert fold_project_token(project_id) == project_id

    @pytest.mark.parametrize(
        ('project_id', 'folded'),
        [('dark__factory', 'dark_factory'), ('_x', 'x'), ('x_', 'x'), ('ß', 'ss')],
    )
    def test_divergence_outside_the_canonical_grammar_is_deliberate(
        self, project_id: str, folded: str
    ) -> None:
        assert not _CANONICAL_TOKEN.fullmatch(project_id)
        assert normalize_project_id(project_id) == project_id
        assert fold_project_token(project_id) == folded


def _replace_df(aliases) -> None:
    aliases['df'] = 'other'


def _add_entry(aliases) -> None:
    aliases['new'] = 'other'


def _delete_df(aliases) -> None:
    del aliases['df']


class TestProjectIdAliases:
    def test_df_is_dark_factory(self) -> None:
        assert PROJECT_ID_ALIASES['df'] == 'dark_factory'

    @pytest.mark.parametrize(('alias', 'canonical'), sorted(PROJECT_ID_ALIASES.items()))
    def test_keys_and_values_are_already_folded(self, alias: str, canonical: str) -> None:
        assert _CANONICAL_TOKEN.fullmatch(alias)
        assert _CANONICAL_TOKEN.fullmatch(canonical)
        assert fold_project_token(alias) == canonical
        assert fold_project_token(canonical) == canonical

    def test_is_not_a_mutable_mapping(self) -> None:
        assert not isinstance(PROJECT_ID_ALIASES, MutableMapping)

    @pytest.mark.parametrize('mutate', [_replace_df, _add_entry, _delete_df])
    def test_refuses_mutation(self, mutate) -> None:
        before = dict(PROJECT_ID_ALIASES)
        with pytest.raises(TypeError):
            mutate(PROJECT_ID_ALIASES)
        assert dict(PROJECT_ID_ALIASES) == before


class TestProjectIdForRoot:
    @pytest.mark.parametrize(
        ('root', 'expected'),
        [
            ('/home/leo/src/dark-factory', 'dark_factory'),
            (Path('/srv/Know-Live'), 'know_live'),
            ('/project/', 'project'),
            ('reify-fix2', 'reify_fix2'),
        ],
    )
    def test_basename_under_the_one_rule(self, root: str | Path, expected: str) -> None:
        assert project_id_for_root(root) == expected
        assert project_id_for_root(root) == normalize_project_id(PurePath(root).name)

    @pytest.mark.parametrize('root', ['/', '', '.', '/srv/dark-factory/..', Path('/'), Path('.')])
    def test_a_root_without_a_basename_is_refused_not_the_unset_sentinel(
        self, root: str | Path
    ) -> None:
        with pytest.raises(ValueError) as refused:
            project_id_for_root(root)
        assert repr(str(root)) in str(refused.value)
