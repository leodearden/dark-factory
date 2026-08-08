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

from fused_memory.utils.canonical_labels import Referent


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
