"""Tests for orchestrator.verify_cmd — structured VerifyCmd command model.

Task β of the verify-plan PRD (plans/verify-plan-prd.md §Contract·VerifyCmd).
Replaces verify.py's raw-shell-string find/replace-surgery command model with
``parse_config_command`` (raw string -> VerifyCmd), ``render`` (VerifyCmd ->
raw string), and a set of pure VerifyCmd -> VerifyCmd mutators.

No source stub exists yet — every test in this module is RED until
orchestrator/src/orchestrator/verify_cmd.py is created (step-2).
"""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.verify_cmd import ToolKind, VerifyCmd


class TestToolKind:
    """ToolKind is a StrEnum whose members are the JSON-serialisable tool identities."""

    def test_members_present(self):
        names = {member.name for member in ToolKind}
        assert names == {
            'PYTEST', 'RUFF', 'PYRIGHT', 'CARGO_TEST', 'CARGO_CLIPPY', 'NPX', 'OPAQUE',
        }

    def test_str_is_byte_identical_member_value(self):
        """str(ToolKind.PYTEST) == 'pytest' — a StrEnum member IS its string value."""
        assert str(ToolKind.PYTEST) == 'pytest'
        assert ToolKind.PYTEST == 'pytest'


class TestVerifyCmdConstruction:
    """VerifyCmd is a frozen dataclass with 8 fields and sensible empty defaults."""

    def test_constructs_with_all_fields(self):
        cmd = VerifyCmd(
            tool=ToolKind.PYTEST,
            uv_project='shared',
            cwd_rel='fused-memory',
            base_flags=('-v',),
            targets=('tests/test_x.py',),
            env={'FOO': 'bar'},
            wrappers=('npx',),
            raw=None,
        )
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.uv_project == 'shared'
        assert cmd.cwd_rel == 'fused-memory'
        assert cmd.base_flags == ('-v',)
        assert cmd.targets == ('tests/test_x.py',)
        assert cmd.env == {'FOO': 'bar'}
        assert cmd.wrappers == ('npx',)
        assert cmd.raw is None

    def test_empty_defaults(self):
        """Only `tool` is required; every other field defaults to empty/None."""
        cmd = VerifyCmd(tool=ToolKind.OPAQUE)
        assert cmd.uv_project is None
        assert cmd.cwd_rel is None
        assert cmd.base_flags == ()
        assert cmd.targets == ()
        assert cmd.env == {}
        assert cmd.wrappers == ()
        assert cmd.raw is None

    def test_is_frozen(self):
        cmd = VerifyCmd(tool=ToolKind.PYTEST)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cmd.tool = ToolKind.RUFF

    def test_equal_instances_compare_equal(self):
        a = VerifyCmd(tool=ToolKind.PYTEST, targets=('x',))
        b = VerifyCmd(tool=ToolKind.PYTEST, targets=('x',))
        assert a == b
