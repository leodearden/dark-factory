"""Unit tests for fused_memory.utils.canonical_labels.

canonical_labels is THE single normative site for the canonical task-label
vocabulary (INV-5 / task 3667): one compiled description of what a task label
looks like, which utils/task_naming.py and utils/cross_project_refs.py both
call rather than each carrying their own lockstep-duplicated copy. Two copies
of a vocabulary drift, and the drift is invisible until a destructive
consumer acts on the half that is behind — 'task #1153' being a task-node
name to a human, but not to the anchored pattern task_naming used to own.

Mirrors tests/test_cross_project_refs.py and tests/test_task_naming.py, whose
leaf-module shape and pytest conventions this module copies.
"""

from __future__ import annotations

import pytest

from fused_memory.utils.canonical_labels import Referent, parse_node_name


class TestReferentNodeName:
    """``Referent.node_name`` renders the graph node name the referent denotes.

    It is a derived property rather than a field so it can never drift out of
    sync with the fields it is computed from (the CrossProjectRef.entity_name
    precedent).
    """

    def test_own_project_referent_renders_the_bare_task_name(self):
        """An EMPTY project_id denotes own-project / unqualified — which is
        what lets node_name discriminate 'Task 132' from 'reify:132' from the
        referent alone."""
        referent = Referent(kind='task', project_id='', number='132')
        assert referent.node_name == 'Task 132'

    def test_qualified_referent_renders_the_qualified_entity_name(self):
        """A project qualifier is a DIFFERENT-project signal and must never be
        normalized away to 'Task 132' — that collapse is precisely the bug
        cross_project_refs exists to detect."""
        referent = Referent(kind='task', project_id='reify', number='132')
        assert referent.node_name == 'reify:132'


class TestDigitsPreservedVerbatim:
    """Digits are stored and rendered verbatim, never int-normalized, mirroring
    canonicalize_task_node_name and CrossProjectRef.task_number."""

    def test_zero_padded_number_survives_rendering(self):
        assert Referent(kind='task', project_id='', number='0132').node_name == 'Task 0132'

    def test_zero_padded_and_unpadded_numbers_are_distinct_referents(self):
        assert Referent(kind='task', project_id='', number='0132') != Referent(
            kind='task', project_id='', number='132'
        )


class TestReferentFieldDefaults:
    """``kind`` defaults to 'task' and ``project_id`` to '' (own-project), so
    the overwhelmingly common referent is spelled with one argument."""

    def test_number_only_construction_is_the_own_project_task_referent(self):
        assert Referent(number='132') == Referent(kind='task', project_id='', number='132')

    def test_number_only_construction_renders_the_bare_task_name(self):
        assert Referent(number='132').node_name == 'Task 132'


class TestReferentIsFrozen:
    """Referent is frozen — a scan result is evidence for destructive graph
    surgery and a consumer must not be able to rewrite which node it names."""

    def test_project_id_is_immutable(self):
        referent = Referent(number='132')
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError subclasses AttributeError
            referent.project_id = 'other'  # type: ignore[misc]

    def test_number_is_immutable(self):
        referent = Referent(number='132')
        with pytest.raises(Exception):  # noqa: B017
            referent.number = '999'  # type: ignore[misc]


class TestParseNodeNameMatchesLocalForms:
    """``parse_node_name`` is ANCHORED (whole-string): it answers "is this
    entity name a task label?", so it must never fire on a name that merely
    mentions a task.

    The separator between the word and the digits is whitespace, '#' or ':'.
    'task #1153' and 'Task: 132' are the forms task_naming's whitespace-only
    pattern could not see — they are two of the PRD's 53 measured variant
    splits, and making them parse is the point of the extraction.
    """

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('task 132', 'Task 132'),
            ('tasks 153', 'Task 153'),
            ('TASK 42', 'Task 42'),
            ('Task  7', 'Task 7'),
            (' tasks 9 ', 'Task 9'),
            ('Task 42', 'Task 42'),  # idempotence: already-canonical maps to itself
            # NEW in task 3667 — the '#' spelling task_naming was structurally
            # unable to match, since '^\s*tasks?\s+(\d+)\s*$' requires whitespace.
            ('task #1153', 'Task 1153'),
            ('task#1153', 'Task 1153'),
            # A task-vocabulary qualifier is LOCAL task vocabulary, never a
            # project id — the same rule cross_project_refs already enforces.
            ('Task: 132', 'Task 132'),
        ],
    )
    def test_parses_to_an_own_project_referent(self, name, expected):
        referent = parse_node_name(name)
        assert referent is not None
        assert referent.kind == 'task'
        assert referent.project_id == ''
        assert referent.node_name == expected

    def test_digits_are_preserved_verbatim(self):
        referent = parse_node_name('task 0132')
        assert referent is not None
        assert referent.number == '0132'
        assert referent != parse_node_name('Task 132')

    def test_is_idempotent_through_node_name(self):
        """Re-parsing a referent's own node_name yields the same name — required
        by the normalization hook's ``canonical != name`` guard, which would
        otherwise rename the same node on every pass."""
        first = parse_node_name('task 132')
        assert first is not None
        second = parse_node_name(first.node_name)
        assert second is not None
        assert second.node_name == 'Task 132'


class TestParseNodeNameMatchesQualifiedForms:
    """A project-qualified node name parses to a FOREIGN referent whose
    qualifier is preserved, never normalized away to the bare 'Task N' form."""

    def test_qualified_name_keeps_its_project(self):
        referent = parse_node_name('reify:132')
        assert referent == Referent(kind='task', project_id='reify', number='132')
        assert referent.node_name == 'reify:132'

    def test_qualifier_is_canonicalized(self):
        referent = parse_node_name('Dark-Factory:2500')
        assert referent is not None
        assert referent.project_id == 'dark_factory'
        assert referent.number == '2500'


class TestParseNodeNameNonMatches:
    """Anchoring plus the precision narrowings: anything that is not itself a
    whole task label returns None, so a caller leaves that node untouched."""

    @pytest.mark.parametrize(
        'name',
        [
            'Alice',
            '',
            'task',  # no number
            'subtask 5',
            'multitask 3',
            'taskforce 9',
            # The anchoring invariant: a name that MENTIONS a task is not a
            # task-node name and must never be renamed to one.
            'Task 3331 dashboard index',
            'Task 42 orchestrator',
            'reify task 12',
            # No separator at all — preserves task_naming's original '\s+'
            # requirement, so the added '#'/':' alternation did not widen the
            # pattern into matching glued digits.
            'task132',
            # Bare-digit names are an explicit PRD blind spot, not a task label.
            '1251',
            # A path-shaped qualifier must be REFUSED, never silently
            # canonicalized into a new, wrong project key (RCA §4).
            '/home/leo/src/dark-factory:2500',
            # Task vocabulary is never a project id, in the qualified form too.
            'subtask: 2500',
            'sub-task: 2500',
        ],
    )
    def test_returns_none(self, name):
        assert parse_node_name(name) is None


class TestUnregisteredKindIsRejectedLoudly:
    """The kind is registry-extensible ('escalation' is the next entry per the
    PRD) and today's registry holds only 'task'.

    Validating at construction makes the extension point concrete and testable,
    and keeps node_name from raising a bare KeyError deep inside a consumer
    (loud-over-silent-degradation / structured-facts-at-failure).
    """

    def test_unregistered_kind_raises_value_error_naming_the_kind(self):
        with pytest.raises(ValueError, match='escalation'):
            Referent(kind='escalation', number='7')

    def test_error_names_the_registered_kinds(self):
        with pytest.raises(ValueError, match='task'):
            Referent(kind='escalation', number='7')
