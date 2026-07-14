"""Tests for fused_memory.maintenance.rehome_scope_tag (task 2452).

Pure functions -- plain dicts, no mocks. scope_tag_for/apply_scope_tag gate
CGL-eta cross_target_rehome entries so a rehomed fact's origin project is
disambiguated inline in its content, not just in metadata.
"""

from __future__ import annotations

from fused_memory.maintenance.rehome_scope_tag import (
    apply_scope_tag,
    scope_tag_for,
)


class TestScopeTagFor:
    def test_src_project_and_src_entity_present(self):
        metadata = {'src_project': 'reify', 'src_entity': 'task 948'}
        assert scope_tag_for(metadata) == '[reify:task 948]'

    def test_src_entity_absent_falls_back_to_dst_entity(self):
        metadata = {'src_project': 'reify', 'dst_entity': 'task 948'}
        assert scope_tag_for(metadata) == '[reify:task 948]'

    def test_neither_entity_present_falls_back_to_bare_project_tag(self):
        metadata = {'src_project': 'reify'}
        assert scope_tag_for(metadata) == '[reify]'

    def test_src_project_missing_returns_none(self):
        metadata = {'src_entity': 'task 948'}
        assert scope_tag_for(metadata) is None

    def test_src_project_empty_string_returns_none(self):
        metadata = {'src_project': '', 'src_entity': 'task 948'}
        assert scope_tag_for(metadata) is None


class TestApplyScopeTag:
    def test_prepends_tag_to_content(self):
        metadata = {'src_project': 'reify', 'src_entity': 'task 948'}
        content = 'Stage 2 should re-escalate task 948 after remediation.'
        assert apply_scope_tag(content, metadata) == (
            '[reify:task 948] Stage 2 should re-escalate task 948 after remediation.'
        )

    def test_idempotent_when_content_already_starts_with_entity_tag(self):
        """Already-tagged content is returned unchanged, even when the entity
        ref baked into the existing tag differs from current metadata -- the
        prefix check (not an exact-match comparison) is what makes re-runs
        of the maintenance script safe."""
        metadata = {'src_project': 'reify', 'src_entity': 'task 999'}
        content = '[reify:task 948] Stage 2 should re-escalate task 948.'
        assert apply_scope_tag(content, metadata) == content

    def test_idempotent_when_content_already_starts_with_bare_project_tag(self):
        metadata = {'src_project': 'reify', 'src_entity': 'task 948'}
        content = '[reify] Stage 2 should re-escalate task 948.'
        assert apply_scope_tag(content, metadata) == content

    def test_src_project_missing_returns_content_unchanged(self):
        metadata = {'src_entity': 'task 948'}
        content = 'Stage 2 should re-escalate task 948 after remediation.'
        assert apply_scope_tag(content, metadata) == content
