"""Tests for orchestrator/verify_runner.py — MergeVerifySpec + VerifyResult JSON codec."""

import dataclasses
import json

import pytest

from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    MergeVerifySpec,
    UnscopedTypecheckSpec,
    VerifyCommand,
    result_from_dict,
    result_from_json,
    result_to_dict,
    result_to_json,
    spec_from_json,
    spec_to_json,
)


# ---------------------------------------------------------------------------
# VerifyCommand
# ---------------------------------------------------------------------------


class TestVerifyCommand:
    """VerifyCommand is a frozen dataclass mirroring ModuleConfig command fields."""

    def test_frozen(self):
        vc = VerifyCommand(prefix="src/mymod")
        with pytest.raises(dataclasses.FrozenInstanceError):
            vc.prefix = "other"  # type: ignore[misc]

    def test_fields_all_commands_present(self):
        vc = VerifyCommand(
            prefix="src/mymod",
            test_command="cargo test -p mymod",
            lint_command="cargo clippy -p mymod",
            type_check_command="pyright src/mymod",
        )
        assert vc.prefix == "src/mymod"
        assert vc.test_command == "cargo test -p mymod"
        assert vc.lint_command == "cargo clippy -p mymod"
        assert vc.type_check_command == "pyright src/mymod"

    def test_command_defaults_are_none(self):
        vc = VerifyCommand(prefix="src/mymod")
        assert vc.test_command is None
        assert vc.lint_command is None
        assert vc.type_check_command is None

    def test_roundtrip_fully_populated(self):
        vc = VerifyCommand(
            prefix="src/mymod",
            test_command="cargo test -p mymod",
            lint_command="cargo clippy",
            type_check_command="pyright src",
        )
        assert VerifyCommand.from_dict(vc.to_dict()) == vc

    def test_roundtrip_all_commands_none(self):
        vc = VerifyCommand(prefix="src/only")
        assert VerifyCommand.from_dict(vc.to_dict()) == vc

    def test_to_dict_shape(self):
        vc = VerifyCommand(prefix="src/mymod", test_command="cargo test")
        d = vc.to_dict()
        assert d == {
            "prefix": "src/mymod",
            "test_command": "cargo test",
            "lint_command": None,
            "type_check_command": None,
        }


# ---------------------------------------------------------------------------
# UnscopedTypecheckSpec
# ---------------------------------------------------------------------------


class TestUnscopedTypecheckSpec:
    """UnscopedTypecheckSpec wraps the _run_unscoped_typechecks gate inputs."""

    def test_frozen(self):
        spec = UnscopedTypecheckSpec(commands=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.block_on_timeout = False  # type: ignore[misc]

    def test_block_on_timeout_default_true(self):
        spec = UnscopedTypecheckSpec(commands=())
        assert spec.block_on_timeout is True

    def test_roundtrip_non_empty_commands(self):
        vc1 = VerifyCommand(prefix="src/a", type_check_command="pyright src/a")
        vc2 = VerifyCommand(prefix="src/b", type_check_command="pyright src/b")
        spec = UnscopedTypecheckSpec(commands=(vc1, vc2), block_on_timeout=True)
        restored = UnscopedTypecheckSpec.from_dict(spec.to_dict())
        assert restored == spec

    def test_roundtrip_empty_commands(self):
        spec = UnscopedTypecheckSpec(commands=(), block_on_timeout=False)
        restored = UnscopedTypecheckSpec.from_dict(spec.to_dict())
        assert restored == spec

    def test_roundtrip_restores_tuple_not_list(self):
        vc = VerifyCommand(prefix="src/a", type_check_command="pyright src/a")
        spec = UnscopedTypecheckSpec(commands=(vc,))
        restored = UnscopedTypecheckSpec.from_dict(spec.to_dict())
        assert isinstance(restored.commands, tuple)
        assert isinstance(restored.commands[0], VerifyCommand)

    def test_to_dict_shape(self):
        vc = VerifyCommand(prefix="src/a", type_check_command="pyright src/a")
        spec = UnscopedTypecheckSpec(commands=(vc,))
        d = spec.to_dict()
        assert d == {
            "commands": [vc.to_dict()],
            "block_on_timeout": True,
        }


# ---------------------------------------------------------------------------
# MergeVerifySpec
# ---------------------------------------------------------------------------


class TestMergeVerifySpec:
    """MergeVerifySpec is the full pre-advance merge-verify bundle contract (PRD §A)."""

    def _make_spec(self, task_files=("src/a/mod.py",)):
        vc = VerifyCommand(
            prefix="src/a",
            test_command="pytest src/a",
            lint_command="ruff src/a",
            type_check_command="pyright src/a",
        )
        utc = UnscopedTypecheckSpec(
            commands=(VerifyCommand(prefix="src/a", type_check_command="pyright src/a"),),
            block_on_timeout=True,
        )
        return MergeVerifySpec(
            verify_commands=(vc,),
            unscoped_typecheck=utc,
            task_files=task_files,
            verify_env={
                "RUSTC_WRAPPER": "/usr/bin/sccache",
                "CARGO_INCREMENTAL": "0",
                "SCCACHE_DIR": "/home/user/.cache/sccache",
            },
            cold_timeout_secs=300.0,
            is_merge_verify=True,
        )

    def test_frozen(self):
        spec = self._make_spec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.is_merge_verify = False  # type: ignore[misc]

    def test_is_merge_verify_default_true(self):
        vc = VerifyCommand(prefix="src/a")
        utc = UnscopedTypecheckSpec(commands=())
        spec = MergeVerifySpec(
            verify_commands=(vc,),
            unscoped_typecheck=utc,
            task_files=None,
            verify_env={},
            cold_timeout_secs=60.0,
        )
        assert spec.is_merge_verify is True

    def test_roundtrip_with_task_files(self):
        spec = self._make_spec(task_files=("src/a/mod.py", "src/b/utils.py"))
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert restored == spec

    def test_roundtrip_task_files_none(self):
        spec = self._make_spec(task_files=None)
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert restored == spec
        assert restored.task_files is None

    def test_roundtrip_restores_nested_dataclasses(self):
        spec = self._make_spec()
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert isinstance(restored.verify_commands, tuple)
        assert isinstance(restored.verify_commands[0], VerifyCommand)
        assert isinstance(restored.unscoped_typecheck, UnscopedTypecheckSpec)
        assert isinstance(restored.unscoped_typecheck.commands[0], VerifyCommand)

    def test_roundtrip_task_files_is_tuple(self):
        spec = self._make_spec(task_files=("src/a/mod.py",))
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert isinstance(restored.task_files, tuple)

    def test_verify_env_round_trips(self):
        spec = self._make_spec()
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert restored.verify_env == {
            "RUSTC_WRAPPER": "/usr/bin/sccache",
            "CARGO_INCREMENTAL": "0",
            "SCCACHE_DIR": "/home/user/.cache/sccache",
        }


# ---------------------------------------------------------------------------
# VerifyResult codec  (result_to_dict / result_from_dict)
# ---------------------------------------------------------------------------


class TestVerifyResultCodec:
    """result_to_dict / result_from_dict round-trip the existing VerifyResult dataclass."""

    def test_roundtrip_minimal_defaults(self):
        vr = VerifyResult(
            passed=True,
            test_output="",
            lint_output="",
            type_output="",
            summary="all good",
        )
        assert result_from_dict(result_to_dict(vr)) == vr

    def test_roundtrip_fully_populated(self):
        vr = VerifyResult(
            passed=False,
            test_output="FAILED test_foo",
            lint_output="error: unused import",
            type_output="Type error line 42",
            summary="3 failures",
            timed_out=True,
            cause_hint="test timed out",
            category="test_failure",
            worktree_log_paths=["logs/worktree.txt"],
            archive_log_paths=["logs/archive.txt"],
        )
        assert result_from_dict(result_to_dict(vr)) == vr

    def test_to_dict_json_native_types(self):
        vr = VerifyResult(
            passed=True,
            test_output="ok",
            lint_output="",
            type_output="",
            summary="pass",
            worktree_log_paths=["a/b.txt"],
        )
        d = result_to_dict(vr)
        # Should round-trip through json without error
        json.dumps(d)
        assert isinstance(d["passed"], bool)
        assert isinstance(d["worktree_log_paths"], list)


# ---------------------------------------------------------------------------
# Golden round-trip test — wire codec (spec_to_json / result_to_json)
# ---------------------------------------------------------------------------


class TestGoldenRoundTrip:
    """
    PRD Invariant 1: byte-identical re-serialization after parse.
    - parsed == original (object equality)
    - s == to_json(from_json(s))  (byte-identical)
    - to_json(obj) called twice yields identical bytes (determinism)
    - sort_keys canonicalizes dict key insertion order
    """

    def _make_spec(self) -> MergeVerifySpec:
        vc1 = VerifyCommand(
            prefix="src/core",
            test_command="pytest src/core",
            lint_command="ruff src/core",
            type_check_command="pyright src/core",
        )
        vc2 = VerifyCommand(
            prefix="src/utils",
            test_command="pytest src/utils",
        )
        utc = UnscopedTypecheckSpec(
            commands=(
                VerifyCommand(prefix="src/core", type_check_command="pyright src/core"),
                VerifyCommand(prefix="src/utils", type_check_command="pyright src/utils"),
            ),
            block_on_timeout=True,
        )
        return MergeVerifySpec(
            verify_commands=(vc1, vc2),
            unscoped_typecheck=utc,
            task_files=("src/core/engine.py", "src/utils/helpers.py"),
            verify_env={
                "RUSTC_WRAPPER": "/usr/bin/sccache",
                "CARGO_INCREMENTAL": "0",
                "SCCACHE_DIR": "/home/user/.cache/sccache",
            },
            cold_timeout_secs=120.5,
            is_merge_verify=True,
        )

    def _make_result(self) -> VerifyResult:
        return VerifyResult(
            passed=False,
            test_output="FAILED 3 tests",
            lint_output="warning: unused var",
            type_output="Type error: line 10",
            summary="3 test failures",
            timed_out=False,
            cause_hint="assertion error",
            category="test_failure",
            worktree_log_paths=["logs/wt.txt"],
            archive_log_paths=["logs/arc.txt"],
        )

    # --- spec ---

    def test_spec_parsed_equals_original(self):
        spec = self._make_spec()
        s = spec_to_json(spec)
        assert spec_from_json(s) == spec

    def test_spec_byte_identical_reserialize(self):
        spec = self._make_spec()
        s = spec_to_json(spec)
        assert spec_to_json(spec_from_json(s)) == s

    def test_spec_deterministic(self):
        spec = self._make_spec()
        assert spec_to_json(spec) == spec_to_json(spec)

    def test_spec_sort_keys_canonicalization(self):
        """verify_env built in reverse-sorted order must produce same JSON as sorted."""
        vc = VerifyCommand(prefix="src/a")
        utc = UnscopedTypecheckSpec(commands=())
        env_sorted = {"CARGO_INCREMENTAL": "0", "RUSTC_WRAPPER": "/usr/bin/sccache", "SCCACHE_DIR": "/tmp"}
        env_reversed = dict(reversed(list(env_sorted.items())))
        # keys should differ in insertion order
        assert list(env_sorted.keys()) != list(env_reversed.keys())
        spec_sorted = MergeVerifySpec(
            verify_commands=(vc,),
            unscoped_typecheck=utc,
            task_files=None,
            verify_env=env_sorted,
            cold_timeout_secs=60.0,
        )
        spec_reversed = MergeVerifySpec(
            verify_commands=(vc,),
            unscoped_typecheck=utc,
            task_files=None,
            verify_env=env_reversed,
            cold_timeout_secs=60.0,
        )
        assert spec_to_json(spec_sorted) == spec_to_json(spec_reversed)

    # --- result ---

    def test_result_parsed_equals_original(self):
        vr = self._make_result()
        s = result_to_json(vr)
        assert result_from_json(s) == vr

    def test_result_byte_identical_reserialize(self):
        vr = self._make_result()
        s = result_to_json(vr)
        assert result_to_json(result_from_json(s)) == s

    def test_result_deterministic(self):
        vr = self._make_result()
        assert result_to_json(vr) == result_to_json(vr)
