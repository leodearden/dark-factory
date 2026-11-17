"""Tests for shared.verify_scope_tag — the verify scope-tag rule's one public home.

PRD ``plans/per-consumer-resource-recording-prd.md`` D2/C2: the load recorder
keys a ``verify`` consumer by the tag a verify scope's unit name carries, and
the hold report computes the same tag from a project root. Both build and
parse go through this module.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from shared.verify_scope_tag import (
    VerifyScopeUnit,
    parse_verify_scope_unit,
    scope_tag_for,
    verify_scope_unit_name,
)

_INSTANCE = '0123456789ab'
_TAG_SHAPE = re.compile(r'^[a-z0-9-]+-[0-9a-f]{8}$')


def _digest(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]


class TestScopeTagFor:
    @pytest.mark.parametrize(
        ('basename', 'slug'),
        [
            ('dark-factory', 'dark-factory'),
            ('reify', 'reify'),
            ('My_Solar Challenge', 'my-solar-challenge'),
            ('a' * 40, 'a' * 32),
            ('---', 'proj'),
            ('__x__', 'x'),
        ],
    )
    def test_slug_then_digest_of_the_resolved_path(self, tmp_path, basename, slug):
        root = tmp_path / basename
        root.mkdir()

        assert scope_tag_for(root) == f'{slug}-{_digest(root)}'

    def test_every_tag_has_the_parseable_shape(self, tmp_path):
        for basename in ('dark-factory', 'My_Solar Challenge', '---', 'Z' * 50):
            assert _TAG_SHAPE.match(scope_tag_for(tmp_path / basename))

    def test_the_same_root_gives_the_same_tag(self, tmp_path):
        assert scope_tag_for(tmp_path / 'p') == scope_tag_for(tmp_path / 'p')

    def test_roots_sharing_a_basename_get_different_tags(self, tmp_path):
        first = scope_tag_for(tmp_path / 'one' / 'reify')
        second = scope_tag_for(tmp_path / 'two' / 'reify')

        assert first != second
        assert first.startswith('reify-') and second.startswith('reify-')

    def test_a_symlink_is_tagged_by_its_target(self, tmp_path):
        target = tmp_path / 'real-project'
        target.mkdir()
        link = tmp_path / 'alias'
        link.symlink_to(target)

        assert scope_tag_for(link) == scope_tag_for(target)


class TestVerifyScopeUnitName:
    def test_builds_the_unit_name(self):
        assert (
            verify_scope_unit_name('dark-factory-c889f090', _INSTANCE)
            == f'df-verify-dark-factory-c889f090-{_INSTANCE}.scope'
        )

    @pytest.mark.parametrize('instance', ['0123456789a', '0123456789abc', 'ABCDEF012345'])
    def test_an_instance_that_would_not_parse_is_refused(self, instance):
        with pytest.raises(ValueError, match=instance):
            verify_scope_unit_name('dark-factory-c889f090', instance)

    @pytest.mark.parametrize('tag', ['dark-factory', 'Dark-c889f090', '-c889f090'])
    def test_a_tag_that_would_not_parse_is_refused(self, tag):
        with pytest.raises(ValueError, match=tag):
            verify_scope_unit_name(tag, _INSTANCE)


class TestParseVerifyScopeUnit:
    @pytest.mark.parametrize(
        'tag',
        [
            'dark-factory-c889f090',
            'x-1234abcd',
            'a--b-1234abcd',
            f'{"a" * 31}--1234abcd',
            'proj-00000000',
        ],
    )
    def test_parses_a_built_name(self, tag):
        unit = parse_verify_scope_unit(verify_scope_unit_name(tag, _INSTANCE))

        assert unit == VerifyScopeUnit(tag=tag, instance=_INSTANCE)

    @pytest.mark.parametrize(
        'name',
        [
            'df-verify-x.scope',
            'df-verify-x-1234abcd-0123456789zz.scope',
            'df-verify-x-1234abcd-0123456789a.scope',
            'df-verify-x-1234abcd-0123456789abc.scope',
            'df-verify-x-1234abcd-0123456789ab.service',
            'df-verify-x-1234abc-0123456789ab.scope',
            'orchestrator-reify.service',
            'app-df-verify-x-1234abcd-0123456789ab.scope',
            '',
        ],
    )
    def test_anything_else_is_not_a_verify_scope(self, name):
        assert parse_verify_scope_unit(name) is None

    def test_round_trip_from_a_project_root(self, tmp_path):
        root = tmp_path / 'My_Solar Challenge'

        unit = parse_verify_scope_unit(
            verify_scope_unit_name(scope_tag_for(root), _INSTANCE)
        )

        assert unit is not None
        assert unit.tag == scope_tag_for(root)
