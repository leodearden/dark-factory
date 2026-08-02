"""Unit tests for fused_memory.utils.cross_project_refs.find_cross_project_task_refs.

find_cross_project_task_refs is a pure, dependency-free detector for
project-qualified task references ('dark_factory:2500') written in an episode
body. graphiti-core's LLM entity extraction discards the qualifier and mints
(or name-dedupes onto) a bare 'Task 2500' node, silently attaching a foreign
project's facts to the local task entity of the same number. This scanner is
the evidence source for the post-write split hook
(MemoryService._split_cross_project_task_nodes, task 3335), so its precision
matters more than its recall: it drives destructive edge surgery, and a false
positive misattributes a local fact — the same bug in the opposite direction.

Mirrors tests/test_task_naming.py, whose leaf-module shape this module copies.
"""
from __future__ import annotations

import pytest

from fused_memory.utils.cross_project_refs import (
    CrossProjectRef,
    CrossProjectRefScan,
    find_cross_project_task_refs,
)

# The incident that motivated task 3335: a reify-group episode mentioning a
# dark_factory task by qualified id, alongside a bare mention of a DIFFERENT
# local task number.
INCIDENT_CONTENT = (
    'Reify task 5181 was cancelled; its work was rerouted to dark_factory:2500.'
)


class TestFindCrossProjectTaskRefsMatches:
    """A qualified 'proj:N' reference to a project other than the current group
    is reported as exactly one ref, with the canonical entity name to mint and
    the bare node name that extraction would have collapsed onto."""

    def test_incident_content_yields_one_ref(self):
        scan = find_cross_project_task_refs(INCIDENT_CONTENT, group_id='reify')
        assert isinstance(scan, CrossProjectRefScan)
        assert len(scan.refs) == 1
        ref = scan.refs[0]
        assert isinstance(ref, CrossProjectRef)
        assert ref.project_id == 'dark_factory'
        assert ref.task_number == '2500'
        assert ref.entity_name == 'dark_factory:2500'
        assert ref.task_node_name == 'Task 2500'

    def test_bare_mention_of_a_different_number_does_not_suppress_the_ref(self):
        """The bare 'task 5181' in the incident content is a DIFFERENT task
        number, so it must not mark 'dark_factory:2500' ambiguous — only a bare
        mention of the SAME number does (see TestAmbiguousRefs)."""
        scan = find_cross_project_task_refs(INCIDENT_CONTENT, group_id='reify')
        assert scan.ambiguous == []

    @pytest.mark.parametrize(
        ('content', 'expected_project', 'expected_number'),
        [
            ('see dark_factory:2500 for context', 'dark_factory', '2500'),
            # Qualifier canonicalization: hyphens -> underscores, lowercased.
            ('see Dark-Factory:2500 for context', 'dark_factory', '2500'),
            ('see DARK_FACTORY:2500 for context', 'dark_factory', '2500'),
            # Tolerant of whitespace around the separator.
            ('see dark_factory : 2500 for context', 'dark_factory', '2500'),
            # Start of content, and end of content with no trailing punctuation.
            ('dark_factory:2500 is the upstream', 'dark_factory', '2500'),
            ('the upstream is dark_factory:2500', 'dark_factory', '2500'),
        ],
    )
    def test_qualifier_forms(self, content, expected_project, expected_number):
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert [(r.project_id, r.task_number) for r in scan.refs] == [
            (expected_project, expected_number)
        ]

    def test_multiple_distinct_refs_preserve_first_seen_order(self):
        content = 'blocked on dark_factory:2500, then stepping:7, then dark_factory:11'
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert [r.entity_name for r in scan.refs] == [
            'dark_factory:2500',
            'stepping:7',
            'dark_factory:11',
        ]


class TestSelfQualifiedRefsAreDropped:
    """A qualifier naming the CURRENT group is not cross-project — extraction
    collapsing 'reify:5181' onto 'Task 5181' inside the reify graph is correct,
    not a bug, so there is nothing to split."""

    @pytest.mark.parametrize(
        ('content', 'group_id'),
        [
            ('reify:5181 was cancelled', 'reify'),
            # Both sides are canonicalized before the comparison, so case and
            # hyphen/underscore spelling differences still count as SELF.
            ('REIFY:5181 was cancelled', 'reify'),
            ('Reify:5181 was cancelled', 'reify'),
            ('Reify-Factory:5181 was cancelled', 'reify_factory'),
            ('reify_factory:5181 was cancelled', 'Reify-Factory'),
        ],
    )
    def test_self_qualified_ref_yields_nothing(self, content, group_id):
        scan = find_cross_project_task_refs(content, group_id=group_id)
        assert scan.refs == []
        assert scan.ambiguous == []


class TestAmbiguousRefs:
    """When the SAME task number appears both qualified and bare in one episode,
    the content is genuinely ambiguous about which facts belong to which task.
    Refuse to split it (loud-over-silent-degradation) rather than guess."""

    def test_same_number_qualified_and_bare_is_ambiguous(self):
        content = 'dark_factory:2500 blocks task 2500 here'
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert scan.refs == []
        assert [r.entity_name for r in scan.ambiguous] == ['dark_factory:2500']

    @pytest.mark.parametrize(
        'bare_form',
        ['task 2500', 'Task 2500', 'TASK 2500', 'tasks 2500'],
    )
    def test_bare_mention_forms_all_trigger_ambiguity(self, bare_form):
        content = f'dark_factory:2500 relates to {bare_form}'
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert scan.refs == []
        assert len(scan.ambiguous) == 1

    def test_ambiguity_is_per_number_not_per_content(self):
        """One ambiguous ref must not suppress an unrelated clean one."""
        content = 'dark_factory:2500 blocks task 2500; also see dark_factory:99'
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert [r.entity_name for r in scan.refs] == ['dark_factory:99']
        assert [r.entity_name for r in scan.ambiguous] == ['dark_factory:2500']

    @pytest.mark.parametrize(
        'content',
        [
            'dark_factory:2500 relates to subtask 2500',
            'dark_factory:2500 relates to multitask 2500',
            'dark_factory:2500 relates to reify-task 2500',
        ],
    )
    def test_word_glued_task_lookalikes_are_not_bare_mentions(self, content):
        """'subtask 2500' is not a bare mention of task 2500, so it must not
        make the qualified ref ambiguous (mirrors _TASK_NODE_NAME_PATTERN's
        refusal to match 'subtask 5')."""
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert [r.entity_name for r in scan.refs] == ['dark_factory:2500']
        assert scan.ambiguous == []


class TestNoFalsePositives:
    """The scan drives destructive edge surgery, so colon-bearing noise that is
    not a project-qualified task reference must never produce a ref."""

    @pytest.mark.parametrize(
        'content',
        [
            '',
            'no colons at all here',
            # Source locations: the '.py' qualifier is too short AND is
            # preceded by a '.', so the lookbehind rejects it outright.
            'see graphiti_client.py:2091 for the call site',
            'memory_service.py:1077 declares it',
            # Clock times: a qualifier must start with a letter.
            'at 12:30 the run started',
            # URLs.
            'see http://x for details',
            'see https://example.com:8080/path',
            # Too-short qualifiers (<3 chars) are noise, not project ids.
            'w6:2 is a cell reference',
            'py:3 is not a project',
            'a:1 is not a project',
            # A colon-chained token is not a qualifier.
            'foo:bar:3 is not a project ref',
            # Prose colon followed by a number.
            'Total: 42 items',
            # A qualifier glued to a preceding word character.
            'xdark_factory:2500',
        ],
    )
    def test_noise_yields_no_refs(self, content):
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert scan.refs == []
        assert scan.ambiguous == []

    def test_path_shaped_qualifier_is_skipped_not_normalized(self):
        """A path-shaped qualifier must never be silently canonicalized into a
        new, wrong project key (PathShapedProjectIdError / RCA §4) — the
        candidate is dropped instead."""
        scan = find_cross_project_task_refs(
            'see /home/leo/src/dark-factory:2500', group_id='reify'
        )
        assert scan.refs == []
        assert scan.ambiguous == []


class TestKnownProjectIdsAllowlist:
    """An optional fourth narrowing: when a registry of known project ids is
    supplied, shape-valid qualifiers naming an unknown project are dropped."""

    def test_unknown_qualifiers_are_dropped_when_allowlist_supplied(self):
        content = 'blocked on dark_factory:2500, stepping:3 and unknown_proj:7'
        scan = find_cross_project_task_refs(
            content, group_id='reify', known_project_ids={'dark_factory'}
        )
        assert [r.entity_name for r in scan.refs] == ['dark_factory:2500']
        assert scan.ambiguous == []

    def test_allowlist_entries_are_canonicalized_before_comparison(self):
        scan = find_cross_project_task_refs(
            'see dark_factory:2500', group_id='reify', known_project_ids={'Dark-Factory'}
        )
        assert [r.entity_name for r in scan.refs] == ['dark_factory:2500']

    @pytest.mark.parametrize('allowlist', [None, set(), frozenset(), []])
    def test_empty_or_missing_allowlist_is_permissive(self, allowlist):
        """Mirrors validate_known_project_id's documented empty-registry mode:
        an unavailable registry must NOT silently disable the protection."""
        content = 'blocked on dark_factory:2500 and stepping:3'
        scan = find_cross_project_task_refs(
            content, group_id='reify', known_project_ids=allowlist
        )
        assert [r.entity_name for r in scan.refs] == [
            'dark_factory:2500',
            'stepping:3',
        ]

    def test_allowlist_accepts_a_mapping_of_project_id_to_root(self):
        """The caller's registry is MemoryService._known_projects, a
        {project_id: project_root} dict — iterating it yields its keys."""
        scan = find_cross_project_task_refs(
            'blocked on dark_factory:2500 and stepping:3',
            group_id='reify',
            known_project_ids={'dark_factory': '/some/root', 'reify': '/other/root'},
        )
        assert [r.entity_name for r in scan.refs] == ['dark_factory:2500']

    def test_path_shaped_allowlist_entry_is_skipped_not_fatal(self):
        scan = find_cross_project_task_refs(
            'see dark_factory:2500',
            group_id='reify',
            known_project_ids={'dark_factory', '-home-leo-src-oops'},
        )
        assert [r.entity_name for r in scan.refs] == ['dark_factory:2500']


class TestDigitsPreservedVerbatim:
    """Digits are never int-normalized, mirroring canonicalize_task_node_name —
    the scan must never invent or reformat a task number."""

    def test_zero_padded_number_is_preserved(self):
        scan = find_cross_project_task_refs('see dark_factory:0250', group_id='reify')
        assert [r.task_number for r in scan.refs] == ['0250']
        assert scan.refs[0].entity_name == 'dark_factory:0250'
        assert scan.refs[0].task_node_name == 'Task 0250'

    def test_zero_padded_bare_mention_matches_only_the_same_literal_digits(self):
        """'task 250' is a different literal from '0250', so it does not make
        'dark_factory:0250' ambiguous."""
        scan = find_cross_project_task_refs(
            'dark_factory:0250 relates to task 250', group_id='reify'
        )
        assert [r.entity_name for r in scan.refs] == ['dark_factory:0250']
        assert scan.ambiguous == []


class TestDeterministicAndDeduplicated:
    """The same (project, number) pair mentioned repeatedly yields one ref."""

    def test_repeated_ref_is_deduplicated(self):
        content = (
            'dark_factory:2500 is upstream; dark_factory:2500 was cancelled; '
            'dark_factory:2500 again'
        )
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert [r.entity_name for r in scan.refs] == ['dark_factory:2500']

    def test_differently_spelled_qualifiers_dedupe_to_one_canonical_ref(self):
        content = 'dark_factory:2500 and Dark-Factory:2500 and DARK_FACTORY:2500'
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert [r.entity_name for r in scan.refs] == ['dark_factory:2500']

    def test_repeated_scans_are_deterministic(self):
        content = 'dark_factory:2500, stepping:7, dark_factory:2500, stepping:7'
        first = find_cross_project_task_refs(content, group_id='reify')
        second = find_cross_project_task_refs(content, group_id='reify')
        assert [r.entity_name for r in first.refs] == ['dark_factory:2500', 'stepping:7']
        assert [r.entity_name for r in first.refs] == [
            r.entity_name for r in second.refs
        ]

    def test_ambiguous_refs_are_deduplicated_too(self):
        content = 'dark_factory:2500 blocks task 2500; dark_factory:2500 again'
        scan = find_cross_project_task_refs(content, group_id='reify')
        assert scan.refs == []
        assert [r.entity_name for r in scan.ambiguous] == ['dark_factory:2500']


class TestCrossProjectRefIsFrozen:
    """CrossProjectRef is a frozen dataclass — the scan result is evidence for
    destructive surgery and must not be mutated by a consumer."""

    def test_ref_is_immutable(self):
        scan = find_cross_project_task_refs('see dark_factory:2500', group_id='reify')
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError subclasses AttributeError
            scan.refs[0].project_id = 'other'  # type: ignore[misc]

    def test_scan_is_immutable(self):
        scan = find_cross_project_task_refs('see dark_factory:2500', group_id='reify')
        with pytest.raises(Exception):  # noqa: B017
            scan.refs = []  # type: ignore[misc]
