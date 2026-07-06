"""Tests for ProjectId/ProjectRoot/ProjectScope — type-level project identity.

Mirrors the class-based style of tests/test_scope.py and the
pytest.raises(InputValidationError) convention of tests/test_validation.py.
"""

import dataclasses

import pytest

from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.utils.validation import InputValidationError


class TestProjectIdAndProjectRoot:
    """ProjectId and ProjectRoot are importable NewTypes wrapping str."""

    def test_project_id_is_a_str_newtype(self):
        """ProjectId.__supertype__ is str, and calling it is the str identity."""
        assert ProjectId.__supertype__ is str
        assert ProjectId('dark_factory') == 'dark_factory'

    def test_project_root_is_a_str_newtype(self):
        """ProjectRoot.__supertype__ is str, and calling it is the str identity."""
        assert ProjectRoot.__supertype__ is str
        assert ProjectRoot('/home/leo/src/dark-factory') == '/home/leo/src/dark-factory'


class TestProjectScopeConstruction:
    """ProjectScope happy path — construction and attribute read-back."""

    def test_constructs_and_reads_back_values(self):
        scope = ProjectScope(
            ProjectId('dark_factory'), ProjectRoot('/home/leo/src/dark-factory')
        )
        assert scope.project_id == 'dark_factory'
        assert scope.project_root == '/home/leo/src/dark-factory'


class TestProjectScopeValidation:
    """__post_init__ rejects empty project_id and non-absolute/empty project_root."""

    def test_empty_project_id_raises(self):
        with pytest.raises(InputValidationError):
            ProjectScope(ProjectId(''), ProjectRoot('/home/leo/src/dark-factory'))

    def test_relative_project_root_raises(self):
        with pytest.raises(InputValidationError):
            ProjectScope(ProjectId('dark_factory'), ProjectRoot('relative/path'))

    def test_empty_project_root_raises(self):
        with pytest.raises(InputValidationError):
            ProjectScope(ProjectId('dark_factory'), ProjectRoot(''))


class TestProjectScopeFrozen:
    """ProjectScope instances are immutable at runtime."""

    def test_mutation_raises_frozen_instance_error(self):
        scope = ProjectScope(
            ProjectId('dark_factory'), ProjectRoot('/home/leo/src/dark-factory')
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(scope, 'project_id', ProjectId('y'))
