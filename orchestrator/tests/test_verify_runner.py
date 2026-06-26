"""Tests for orchestrator/verify_runner.py — MergeVerifySpec + VerifyResult JSON codec."""

import dataclasses
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    LocalRunner,
    MergeVerifySpec,
    RemoteRunner,
    UnscopedTypecheckSpec,
    VerifyCommand,
    VerifyRunner,
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

    def test_spec_byte_identical_with_int_cold_timeout(self):
        """A spec constructed with an integer cold_timeout_secs round-trips byte-identically.

        PRD Invariant 1: to_json(from_json(s)) == s must hold even when the
        original spec was built with cold_timeout_secs as an int (e.g. from YAML).
        Without symmetric float() coercion in to_dict(), the first serialization
        would emit ``300`` while the re-serialization would emit ``300.0``.
        """
        vc = VerifyCommand(prefix="src/a")
        utc = UnscopedTypecheckSpec(commands=())
        spec = MergeVerifySpec(
            verify_commands=(vc,),
            unscoped_typecheck=utc,
            task_files=None,
            verify_env={},
            cold_timeout_secs=300,  # int, not float literal
        )
        s = spec_to_json(spec)
        assert spec_to_json(spec_from_json(s)) == s


# ---------------------------------------------------------------------------
# Error-path coverage — codec failure modes for bad wire input
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Codec failure modes when consuming malformed or incomplete bytes from another host."""

    def test_spec_from_json_missing_required_key_raises_key_error(self):
        """spec_from_json with an empty object raises KeyError for 'verify_commands'."""
        with pytest.raises(KeyError):
            spec_from_json("{}")

    def test_spec_from_json_malformed_json_raises_decode_error(self):
        """spec_from_json with invalid JSON raises json.JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            spec_from_json("not { valid } json")

    def test_result_from_dict_unknown_key_raises_type_error(self):
        """result_from_dict with an unrecognised key raises TypeError.

        VerifyResult(**d) fails when d contains keys that don't match any field,
        so the codec has well-defined (not silent) behaviour on unexpected input.
        """
        d = {
            "passed": True,
            "test_output": "",
            "lint_output": "",
            "type_output": "",
            "summary": "ok",
            "unknown_extra_field": "should_fail",
        }
        with pytest.raises(TypeError):
            result_from_dict(d)

    def test_result_from_json_empty_object_raises_type_error(self):
        """result_from_json with an empty JSON object raises TypeError (missing required args)."""
        with pytest.raises(TypeError):
            result_from_json("{}")


# ---------------------------------------------------------------------------
# Step-1: VerifyRunner protocol + LocalRunner identity
# ---------------------------------------------------------------------------


def _make_local_runner(*, run_scoped=None, run_unscoped=None):
    """Build a LocalRunner with injected fake callables."""
    merge_wt = MagicMock()
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.merge_verify_workspace = False
    module_configs = []
    task_files = None
    run_scoped = run_scoped or AsyncMock(return_value=VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='ok',
    ))
    run_unscoped = run_unscoped or AsyncMock(return_value=MagicMock(broken=False, timed_out=False))
    return LocalRunner(
        merge_wt=merge_wt,
        config=config,
        module_configs=module_configs,
        task_files=task_files,
        run_scoped=run_scoped,
        run_unscoped=run_unscoped,
    )


class TestVerifyRunnerProtocol:
    """VerifyRunner is a @runtime_checkable Protocol; LocalRunner satisfies it."""

    def test_local_runner_name_is_local(self):
        runner = _make_local_runner()
        assert runner.name == 'local'

    @pytest.mark.asyncio
    async def test_local_runner_health_returns_true(self):
        runner = _make_local_runner()
        assert await runner.health() is True

    def test_local_runner_is_instance_of_verify_runner_protocol(self):
        runner = _make_local_runner()
        assert isinstance(runner, VerifyRunner)


# ---------------------------------------------------------------------------
# Step-3: LocalRunner.run_merge_verify combined-bundle behaviour
# ---------------------------------------------------------------------------


def _make_pass_result(**kwargs):
    defaults = dict(passed=True, test_output='', lint_output='', type_output='', summary='ok')
    defaults.update(kwargs)
    return VerifyResult(**defaults)  # type: ignore[arg-type]


def _make_fail_result(**kwargs):
    defaults = dict(passed=False, test_output='FAILED', lint_output='', type_output='', summary='test fail')
    defaults.update(kwargs)
    return VerifyResult(**defaults)  # type: ignore[arg-type]


def _make_spec():
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )


@pytest.mark.asyncio
class TestLocalRunnerBundle:
    """LocalRunner.run_merge_verify: combined scoped+unscoped bundle."""

    async def test_pass_path_returns_passed_result_and_invokes_unscoped(self):
        scoped_result = _make_pass_result()
        run_scoped = AsyncMock(return_value=scoped_result)
        unscoped_gate = MagicMock(broken=False, timed_out=False, failing_subprojects=[], timed_out_subprojects=[])
        run_unscoped = AsyncMock(return_value=unscoped_gate)
        runner = _make_local_runner(run_scoped=run_scoped, run_unscoped=run_unscoped)

        result = await runner.run_merge_verify('abc123', _make_spec())

        assert result.passed is True
        run_unscoped.assert_awaited_once()

    async def test_scoped_fail_short_circuits_unscoped(self):
        scoped_result = _make_fail_result()
        run_scoped = AsyncMock(return_value=scoped_result)
        run_unscoped = AsyncMock()
        runner = _make_local_runner(run_scoped=run_scoped, run_unscoped=run_unscoped)

        result = await runner.run_merge_verify('abc123', _make_spec())

        assert result.passed is False
        assert result.summary == 'test fail'
        run_unscoped.assert_not_awaited()

    async def test_scoped_fail_returns_scoped_result_unchanged(self):
        scoped_result = _make_fail_result(category='test_failure', cause_hint='assertion error')
        run_scoped = AsyncMock(return_value=scoped_result)
        runner = _make_local_runner(run_scoped=run_scoped)

        result = await runner.run_merge_verify('abc123', _make_spec())

        assert result is scoped_result

    async def test_unscoped_broken_returns_sentinel_category_result(self):
        from orchestrator.verify_runner import UNSCOPED_TYPECHECK_FAILED_CATEGORY
        scoped_result = _make_pass_result()
        run_scoped = AsyncMock(return_value=scoped_result)
        gate = MagicMock(
            broken=True,
            timed_out=False,
            timed_out_subprojects=[],
            failing_subprojects=['src/a', 'src/b'],
            detail='type error line 10',
        )
        run_unscoped = AsyncMock(return_value=gate)
        runner = _make_local_runner(run_scoped=run_scoped, run_unscoped=run_unscoped)

        result = await runner.run_merge_verify('abc123', _make_spec())

        assert result.passed is False
        assert result.category == UNSCOPED_TYPECHECK_FAILED_CATEGORY
        assert 'src/a' in result.summary or 'src/a' in (result.type_output or '')

    async def test_unscoped_timeout_returns_timeout_sentinel_category(self):
        from orchestrator.verify_runner import UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY
        scoped_result = _make_pass_result()
        run_scoped = AsyncMock(return_value=scoped_result)
        gate = MagicMock(
            broken=True,
            timed_out=True,
            timed_out_subprojects=['src/a'],
            failing_subprojects=['src/a'],
            detail='',
        )
        run_unscoped = AsyncMock(return_value=gate)
        runner = _make_local_runner(run_scoped=run_scoped, run_unscoped=run_unscoped)

        result = await runner.run_merge_verify('abc123', _make_spec())

        assert result.passed is False
        assert result.category == UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY
        assert result.timed_out is True

    async def test_scoped_called_with_correct_kwargs(self):
        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(return_value=MagicMock(broken=False))
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = True
        merge_wt = MagicMock()
        module_configs = [MagicMock()]
        task_files = ('src/a.py',)
        runner = LocalRunner(
            merge_wt=merge_wt,
            config=config,
            module_configs=module_configs,  # type: ignore[arg-type]
            task_files=task_files,
            run_scoped=run_scoped,
            run_unscoped=run_unscoped,
        )
        await runner.run_merge_verify('abc123', _make_spec())

        run_scoped.assert_awaited_once_with(
            merge_wt, config, module_configs,
            task_files=task_files,
            max_retries=0,
            is_merge_verify=True,
            force_workspace=True,
            role='merge',
            task_id=None,
            archive_root=None,
        )


# ---------------------------------------------------------------------------
# Step-5: VerifyRunnerPool.dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyRunnerPool:
    """VerifyRunnerPool.dispatch: routes to single runner + emits merge_verify event."""

    async def test_dispatch_returns_runner_result(self):
        from orchestrator.verify_runner import VerifyRunnerPool
        expected = _make_pass_result()
        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(return_value=expected)
        pool = VerifyRunnerPool([fake_runner])

        result = await pool.dispatch('abc123', _make_spec())

        assert result is expected

    async def test_dispatch_emits_merge_verify_event(self):
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import VerifyRunnerPool
        expected = _make_pass_result()
        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(return_value=expected)

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))

        pool = VerifyRunnerPool([fake_runner], event_store=event_store, task_id='t-42')
        await pool.dispatch('sha999', _make_spec())

        assert len(emitted) == 1
        (event_type,), kwargs = emitted[0]
        assert event_type == EventType.merge_verify
        data = kwargs['data']
        assert data['runner'] == 'local'
        assert data['merge_sha'] == 'sha999'
        assert data['passed'] is True

    async def test_dispatch_without_event_store_does_not_raise(self):
        from orchestrator.verify_runner import VerifyRunnerPool
        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(return_value=_make_pass_result())
        pool = VerifyRunnerPool([fake_runner], event_store=None)

        result = await pool.dispatch('abc123', _make_spec())

        assert result.passed is True


# ---------------------------------------------------------------------------
# Step-7: build_merge_verify_spec
# ---------------------------------------------------------------------------


class TestBuildMergeVerifySpec:
    """build_merge_verify_spec projects config + module_configs into a MergeVerifySpec."""

    def _make_module_config(self, prefix, *, test_cmd=None, lint_cmd=None, type_check_cmd=None):
        mc = MagicMock()
        mc.prefix = prefix
        mc.test_command = test_cmd
        mc.lint_command = lint_cmd
        mc.type_check_command = type_check_cmd
        return mc

    def _make_config(self, *, verify_env=None, cold_timeout=None):
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.verify_env = verify_env or {}
        config.effective_verify_env = verify_env or {}
        config.merge_verify_cold_command_timeout_secs = cold_timeout
        config.verify_cold_command_timeout_secs = None
        return config

    def test_verify_commands_project_module_fields(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        mc = self._make_module_config('src/a', test_cmd='pytest src/a', lint_cmd='ruff src/a')
        spec = build_merge_verify_spec(self._make_config(), [mc], None)

        assert len(spec.verify_commands) == 1
        vc = spec.verify_commands[0]
        assert vc.prefix == 'src/a'
        assert vc.test_command == 'pytest src/a'
        assert vc.lint_command == 'ruff src/a'

    def test_unscoped_typecheck_includes_only_type_check_modules(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        mc_with = self._make_module_config('src/a', type_check_cmd='pyright src/a')
        mc_without = self._make_module_config('src/b')
        spec = build_merge_verify_spec(self._make_config(), [mc_with, mc_without], None)

        prefixes = [vc.prefix for vc in spec.unscoped_typecheck.commands]
        assert 'src/a' in prefixes
        assert 'src/b' not in prefixes

    def test_unscoped_typecheck_block_on_timeout_true(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        mc = self._make_module_config('src/a', type_check_cmd='pyright src/a')
        spec = build_merge_verify_spec(self._make_config(), [mc], None)
        assert spec.unscoped_typecheck.block_on_timeout is True

    def test_task_files_carried_as_tuple(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        task_files = ('src/a/mod.py', 'src/b/utils.py')
        spec = build_merge_verify_spec(self._make_config(), [], task_files)
        assert spec.task_files == task_files

    def test_task_files_none_stays_none(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        spec = build_merge_verify_spec(self._make_config(), [], None)
        assert spec.task_files is None

    def test_verify_env_from_config(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        env = {'RUSTC_WRAPPER': '/usr/bin/sccache', 'CARGO_INCREMENTAL': '0'}
        spec = build_merge_verify_spec(self._make_config(verify_env=env), [], None)
        assert spec.verify_env == env

    def test_cold_timeout_from_merge_verify_specific(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        spec = build_merge_verify_spec(self._make_config(cold_timeout=7200.0), [], None)
        assert spec.cold_timeout_secs == 7200.0

    def test_cold_timeout_falls_back_to_verify_cold(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.verify_env = {}
        config.effective_verify_env = {}
        config.merge_verify_cold_command_timeout_secs = None
        config.verify_cold_command_timeout_secs = 3600.0
        spec = build_merge_verify_spec(config, [], None)
        assert spec.cold_timeout_secs == 3600.0

    def test_cold_timeout_falls_back_to_zero_when_both_none(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.verify_env = {}
        config.effective_verify_env = {}
        config.merge_verify_cold_command_timeout_secs = None
        config.verify_cold_command_timeout_secs = None
        spec = build_merge_verify_spec(config, [], None)
        assert spec.cold_timeout_secs == 0.0

    def test_is_merge_verify_is_true(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        spec = build_merge_verify_spec(self._make_config(), [], None)
        assert spec.is_merge_verify is True

    def test_result_roundtrips_json_codec(self):
        from orchestrator.verify_runner import build_merge_verify_spec, spec_from_json, spec_to_json
        mc = self._make_module_config('src/a', test_cmd='pytest', type_check_cmd='pyright src/a')
        task_files = ('src/a/f.py',)
        spec = build_merge_verify_spec(
            self._make_config(verify_env={'K': 'V'}, cold_timeout=300.0),
            [mc],
            task_files,
        )
        assert spec_from_json(spec_to_json(spec)) == spec

    def test_effective_verify_env_propagated_to_spec(self, monkeypatch, tmp_path):
        """spec.verify_env carries the merged sccache backend from effective_verify_env.

        Uses a real OrchestratorConfig (not MagicMock) so that effective_verify_env
        is computed by the actual property; this exercises the κ wire end-to-end for
        the remote/laptop path (build_merge_verify_spec reads effective_verify_env).
        """
        from orchestrator.config import OrchestratorConfig, SccacheConfig
        from orchestrator.verify_runner import build_merge_verify_spec

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)

        config = OrchestratorConfig(
            verify_env={'RUSTC_WRAPPER': 'sccache'},
            sccache=SccacheConfig(enabled=True, backend_env={'SCCACHE_REDIS': 'redis://orch:6379'}),
        )
        spec = build_merge_verify_spec(config, [], None)
        assert spec.verify_env == {
            'RUSTC_WRAPPER': 'sccache',
            'SCCACHE_REDIS': 'redis://orch:6379',
        }


# ---------------------------------------------------------------------------
# Step-1: _module_config_from_command — inverse of build_merge_verify_spec's projection
# ---------------------------------------------------------------------------


class TestModuleConfigFromCommand:
    """_module_config_from_command(vc, spec) is the inverse of build_merge_verify_spec's projection."""

    def _make_spec(self, *, verify_env=None, cold_timeout_secs=300.0):
        return MergeVerifySpec(
            verify_commands=(),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=None,
            verify_env=verify_env if verify_env is not None else {'K': 'V'},
            cold_timeout_secs=cold_timeout_secs,
        )

    def test_case_a_fully_populated(self):
        from orchestrator.config import ModuleConfig
        from orchestrator.verify_runner import _module_config_from_command
        vc = VerifyCommand(
            prefix='mod',
            test_command='true',
            lint_command='ruff',
            type_check_command='pyright',
        )
        spec = self._make_spec(verify_env={'K': 'V'}, cold_timeout_secs=300.0)
        mc = _module_config_from_command(vc, spec)
        assert isinstance(mc, ModuleConfig)
        assert mc.prefix == 'mod'
        assert mc.test_command == 'true'
        assert mc.lint_command == 'ruff'
        assert mc.type_check_command == 'pyright'
        assert mc.verify_env == {'K': 'V'}
        assert mc.verify_cold_command_timeout_secs == 300.0

    def test_case_b_all_commands_none(self):
        from orchestrator.config import ModuleConfig
        from orchestrator.verify_runner import _module_config_from_command
        vc = VerifyCommand(prefix='mod2')
        spec = self._make_spec(verify_env={'K': 'V'}, cold_timeout_secs=300.0)
        mc = _module_config_from_command(vc, spec)
        assert isinstance(mc, ModuleConfig)
        assert mc.prefix == 'mod2'
        assert mc.test_command is None
        assert mc.lint_command is None
        assert mc.type_check_command is None
        assert mc.verify_env == {'K': 'V'}
        assert mc.verify_cold_command_timeout_secs == 300.0

    def test_cold_timeout_zero_sentinel_maps_to_none(self):
        """build_merge_verify_spec emits cold_timeout_secs=0.0 when neither
        merge_verify_cold_command_timeout_secs nor verify_cold_command_timeout_secs
        is set (the common 'unset' sentinel). The original local ModuleConfig in that
        case has verify_cold_command_timeout_secs=None (config.py default).

        Storing the literal 0.0 breaks fidelity: _resolve_verify_timeout treats 0.0
        as 'not None', returns it, and asyncio.wait_for(..., timeout=0.0) raises
        TimeoutError immediately, spuriously blocking merges.

        The mapper must translate 0.0 → None so the host-side cascade falls through to
        warm/global identically to a real local merge run.

        Regression guard: a positive cold timeout (e.g. 300.0) must still map verbatim.
        """
        from orchestrator.verify_runner import _module_config_from_command
        vc = VerifyCommand(prefix='mod', test_command='true')

        # 0.0 sentinel → None (the fidelity-preserving round-trip value)
        spec_zero = self._make_spec(cold_timeout_secs=0.0)
        mc = _module_config_from_command(vc, spec_zero)
        assert mc.verify_cold_command_timeout_secs is None, (
            f"Expected None for 0.0 sentinel, got {mc.verify_cold_command_timeout_secs!r}"
        )

        # Positive cold timeout still maps verbatim
        spec_pos = self._make_spec(cold_timeout_secs=300.0)
        mc2 = _module_config_from_command(VerifyCommand(prefix='mod'), spec_pos)
        assert mc2.verify_cold_command_timeout_secs == 300.0


# ---------------------------------------------------------------------------
# Step-3: run_merge_verify_on_worktree — host-entry wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunMergeVerifyOnWorktree:
    """run_merge_verify_on_worktree reconstructs module_configs and delegates to LocalRunner."""

    def _make_two_command_spec(self):
        return MergeVerifySpec(
            verify_commands=(
                VerifyCommand('src/a', test_command='true'),
                VerifyCommand('src/b', lint_command='ruff'),
            ),
            unscoped_typecheck=UnscopedTypecheckSpec(
                commands=(VerifyCommand('src/a', type_check_command='true'),),
                block_on_timeout=True,
            ),
            task_files=('src/a/m.py',),
            verify_env={'K': 'V'},
            cold_timeout_secs=123.0,
        )

    async def test_all_pass_returns_pass_result(self):
        from orchestrator.verify_runner import run_merge_verify_on_worktree
        pass_result = _make_pass_result()
        run_scoped = AsyncMock(return_value=pass_result)
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False,
                timed_out=False,
                failing_subprojects=[],
                timed_out_subprojects=[],
            )
        )
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        merge_wt = MagicMock()
        spec = self._make_two_command_spec()

        result = await run_merge_verify_on_worktree(
            merge_wt, config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert result is pass_result

    async def test_module_configs_reconstructed_from_spec(self):
        """run_scoped is called with module_configs reconstructed from spec."""
        from orchestrator.verify_runner import run_merge_verify_on_worktree
        pass_result = _make_pass_result()
        run_scoped = AsyncMock(return_value=pass_result)
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False,
                timed_out=False,
                failing_subprojects=[],
                timed_out_subprojects=[],
            )
        )
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        merge_wt = MagicMock()
        spec = self._make_two_command_spec()

        await run_merge_verify_on_worktree(
            merge_wt, config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        # Inspect positional args: run_scoped(merge_wt, config, module_configs, ...)
        call_args = run_scoped.await_args
        assert call_args is not None
        pos_args = call_args[0]
        module_configs = pos_args[2]
        assert len(module_configs) == 2
        prefixes = [mc.prefix for mc in module_configs]
        assert prefixes == ['src/a', 'src/b']
        assert module_configs[0].test_command == 'true'
        assert module_configs[0].lint_command is None
        assert module_configs[1].test_command is None
        assert module_configs[1].lint_command == 'ruff'

    async def test_task_files_threaded_to_scoped(self):
        """task_files from the spec are passed as the task_files kwarg to run_scoped."""
        from orchestrator.verify_runner import run_merge_verify_on_worktree
        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False,
                timed_out=False,
                failing_subprojects=[],
                timed_out_subprojects=[],
            )
        )
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        spec = self._make_two_command_spec()

        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert run_scoped.await_args is not None
        call_kwargs = run_scoped.await_args[1]
        assert call_kwargs['task_files'] == ('src/a/m.py',)
        assert call_kwargs['max_retries'] == 0
        assert call_kwargs['is_merge_verify'] is True
        assert call_kwargs['force_workspace'] is False
        assert call_kwargs['role'] == 'merge'

    async def test_gate_broken_returns_sentinel_result(self):
        """When run_unscoped returns broken=True, result carries UNSCOPED_TYPECHECK_FAILED_CATEGORY."""
        from orchestrator.verify_runner import (
            UNSCOPED_TYPECHECK_FAILED_CATEGORY,
            run_merge_verify_on_worktree,
        )
        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=True,
                timed_out=False,
                timed_out_subprojects=[],
                failing_subprojects=['src/a'],
                detail='type err',
            )
        )
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        spec = self._make_two_command_spec()

        result = await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert result.passed is False
        assert result.category == UNSCOPED_TYPECHECK_FAILED_CATEGORY
        assert 'src/a' in result.summary  # noqa: SIM910

    async def test_type_check_command_survives_into_unscoped_module_configs(self):
        """type_check_command on verify_commands survives into module_configs passed to run_unscoped.

        build_merge_verify_spec copies type_check_command into both verify_commands and
        unscoped_typecheck.commands, keeping them in sync.  This test mirrors a realistic
        spec produced by that function and asserts that the reconstructed module_configs
        passed to the unscoped gate carry the type_check_command — ensuring the gate is
        not silently a no-op when typecheck commands are present.
        """
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False,
                timed_out=False,
                failing_subprojects=[],
                timed_out_subprojects=[],
            )
        )
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False

        # Spec with type_check_command mirroring what build_merge_verify_spec produces
        # when the module has a real typecheck command
        spec = MergeVerifySpec(
            verify_commands=(
                VerifyCommand('src/a', test_command='true', type_check_command='mypy'),
            ),
            unscoped_typecheck=UnscopedTypecheckSpec(
                commands=(VerifyCommand('src/a', type_check_command='mypy'),),
                block_on_timeout=True,
            ),
            task_files=None,
            verify_env={},
            cold_timeout_secs=123.0,
        )

        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert run_unscoped.await_args is not None, 'run_unscoped must have been called'
        unscoped_pos = run_unscoped.await_args[0]
        # run_unscoped(merge_wt, config, module_configs, ...)  — module_configs is positional arg 2
        unscoped_module_configs = unscoped_pos[2]
        assert len(unscoped_module_configs) == 1
        assert unscoped_module_configs[0].prefix == 'src/a'
        assert unscoped_module_configs[0].type_check_command == 'mypy', (
            "type_check_command must survive spec→module_configs reconstruction so the "
            "unscoped typecheck gate is not silently a no-op"
        )


# ---------------------------------------------------------------------------
# Step-5: run_merge_verify_on_worktree defaults to real merge-path callables
# ---------------------------------------------------------------------------

# NOTE: The sections below are added by task δ (1696) and test RemoteRunner,
# VerifyRunnerPool preference, and fail-safe fallback.


@pytest.mark.asyncio
class TestRunMergeVerifyOnWorktreeDefaults:
    """When no callables are injected, the real merge-path globals are used (production path)."""

    async def test_defaults_to_real_callables(self, monkeypatch):
        import orchestrator.merge_queue as mq_mod
        import orchestrator.verify as verify_mod
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        pass_result = _make_pass_result()
        fake_scoped = AsyncMock(return_value=pass_result)
        fake_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False,
                timed_out=False,
                failing_subprojects=[],
                timed_out_subprojects=[],
            )
        )
        monkeypatch.setattr(verify_mod, 'run_scoped_verification', fake_scoped)
        monkeypatch.setattr(mq_mod, '_run_unscoped_typechecks', fake_unscoped)

        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        spec = MergeVerifySpec(
            verify_commands=(VerifyCommand('mod', test_command='true'),),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=None,
            verify_env={},
            cold_timeout_secs=60.0,
        )

        # Call WITHOUT run_scoped/run_unscoped — should use patched globals
        result = await run_merge_verify_on_worktree(MagicMock(), config, spec)

        fake_scoped.assert_awaited_once()
        fake_unscoped.assert_awaited_once()
        assert result is pass_result


# ---------------------------------------------------------------------------
# δ step-1: RunnerUnavailable — exception class + __all__ presence
# ---------------------------------------------------------------------------


class TestRunnerUnavailable:
    """RunnerUnavailable is an Exception subclass exported in verify_runner.__all__."""

    def test_import_runner_unavailable(self):
        from orchestrator.verify_runner import RunnerUnavailable  # noqa: F401

    def test_is_exception_subclass(self):
        from orchestrator.verify_runner import RunnerUnavailable
        assert issubclass(RunnerUnavailable, Exception)

    def test_constructible_with_message(self):
        from orchestrator.verify_runner import RunnerUnavailable
        exc = RunnerUnavailable("host down")
        assert str(exc) == "host down"

    def test_present_in_dunder_all(self):
        import orchestrator.verify_runner as vr_mod
        assert 'RunnerUnavailable' in vr_mod.__all__


# ---------------------------------------------------------------------------
# δ step-3: RemoteRunner construction + health()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerConstruction:
    """RemoteRunner construction, VerifyRunner conformance, and health() probe."""

    def _make_fake_run(self, responses):
        """Return an async callable that returns successive (rc, stdout, stderr) tuples.

        ``responses`` is a list of (rc, stdout, stderr) tuples consumed in order.
        """
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv)
            return responses.pop(0)

        run: Any = fake_run
        run.calls = calls
        return run

    async def test_name_attribute(self):
        from orchestrator.verify_runner import RemoteRunner
        fake_run = self._make_fake_run([])
        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        assert runner.name == 'laptop'

    async def test_isinstance_verify_runner_protocol(self):
        from orchestrator.verify_runner import RemoteRunner
        fake_run = self._make_fake_run([])
        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        assert isinstance(runner, VerifyRunner)

    async def test_health_true_when_ssh_rc_zero(self):
        from orchestrator.verify_runner import RemoteRunner
        fake_run = self._make_fake_run([(0, '', '')])
        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        result = await runner.health()
        assert result is True
        # health issues `ssh -o BatchMode=yes -o ConnectTimeout=10 <host> true`
        assert fake_run.calls == [
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'laptop.local', 'true']
        ]

    async def test_health_false_when_ssh_rc_nonzero(self):
        from orchestrator.verify_runner import RemoteRunner
        fake_run = self._make_fake_run([(1, '', 'Connection refused')])
        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        result = await runner.health()
        assert result is False

    async def test_health_false_when_run_raises(self):
        from orchestrator.verify_runner import RemoteRunner

        async def raising_run(argv, *, cwd=None):
            raise OSError("ssh not found")

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=raising_run,
        )
        # health() must never raise
        result = await runner.health()
        assert result is False


# ---------------------------------------------------------------------------
# δ step-5: RemoteRunner.run_merge_verify — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerHappyPath:
    """run_merge_verify happy path: git push + ssh + parse stdout."""

    def _make_runner_and_calls(self, expected_result, *, config_path=None):
        """Return (runner, calls_list) where calls_list is appended to on each run()."""
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append((argv, cwd))
            # git push → rc=0
            if argv[0] == 'git':
                return (0, '', '')
            # ssh → rc=0 with VerifyResult JSON stdout
            return (0, result_to_json(expected_result), '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            config_path=config_path,
            run=fake_run,
            id_factory=lambda: 'fixed-id',
        )
        return runner, calls

    async def test_returns_verify_result_equal_to_expected(self):
        expected = VerifyResult(
            passed=True,
            test_output='all green',
            lint_output='',
            type_output='',
            summary='ok',
        )
        runner, _ = self._make_runner_and_calls(expected)
        result = await runner.run_merge_verify('abc123', _make_spec())
        assert result == expected

    async def test_git_push_argv_and_cwd(self):
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected)
        await runner.run_merge_verify('abc123', _make_spec())
        # first call is the git push
        push_argv, push_cwd = calls[0]
        assert push_argv == ['git', 'push', 'origin', 'abc123:refs/merge-verify/fixed-id']
        assert push_cwd == '/repo'

    async def test_ssh_argv_with_shlex_quoted_spec(self):
        """ssh is called as ['ssh', host, remote_cmd] where shlex.split(remote_cmd) round-trips."""
        import shlex as _shlex

        from orchestrator.verify_runner import spec_to_json
        spec = _make_spec()
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected)
        await runner.run_merge_verify('abc123', spec)
        # second call is the ssh (with hardening flags: -o BatchMode=yes -o ConnectTimeout=10)
        ssh_argv, _ = calls[1]
        assert ssh_argv[0] == 'ssh'
        assert ssh_argv[-2] == 'laptop.local'   # host is second-to-last
        remote_cmd = ssh_argv[-1]               # quoted remote command is last
        parsed = _shlex.split(remote_cmd)
        assert parsed[:4] == ['orchestrator', 'verify-merge', '--sha', 'abc123']
        # spec JSON survives as a single token
        spec_idx = parsed.index('--spec') + 1
        assert parsed[spec_idx] == spec_to_json(spec)
        # no --config when not set
        assert '--config' not in parsed

    async def test_ssh_argv_includes_config_path_when_set(self):
        """When config_path is set, ['--config', config_path] appears in the remote cmd."""
        import shlex as _shlex

        spec = _make_spec()
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected, config_path='/etc/orch.yaml')
        await runner.run_merge_verify('abc123', spec)
        ssh_argv, _ = calls[1]
        parsed = _shlex.split(ssh_argv[-1])  # last arg is the quoted remote command
        cfg_idx = parsed.index('--config') + 1
        assert parsed[cfg_idx] == '/etc/orch.yaml'

    async def test_request_id_from_id_factory(self):
        """The pushed ref uses the id_factory's return value."""
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected)
        await runner.run_merge_verify('abc123', _make_spec())
        push_argv, _ = calls[0]
        assert push_argv[3] == 'abc123:refs/merge-verify/fixed-id'


# ---------------------------------------------------------------------------
# δ step-7: RemoteRunner transport vs timeout boundary (PRD Invariant 5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerTransportVsTimeout:
    """Invariant 5: RunnerUnavailable ↔ transport failure only; VerifyResult returned for any verdict."""

    def _make_runner(self, responses, *, raise_on=None):
        """Build a RemoteRunner with a fake `run` that returns successive responses.

        ``responses`` is a list of (rc, stdout, stderr) tuples.
        ``raise_on`` is an optional exception to raise on the Nth call (0-indexed dict).
        """
        calls = []
        call_counter = [0]

        async def fake_run(argv, *, cwd=None):
            n = call_counter[0]
            call_counter[0] += 1
            calls.append(argv[:])
            if raise_on is not None and n in raise_on:
                raise raise_on[n]
            return responses[n]

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'req-id',
        )
        runner._calls = calls
        return runner

    async def test_raises_runner_unavailable_on_push_failure(self):
        """git push rc!=0 → RunnerUnavailable; ssh is never called."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner = self._make_runner([(1, '', 'push error'), (0, '', '')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())
        # ssh must NOT have been attempted
        assert not any(a[0] == 'ssh' for a in runner._calls)

    async def test_raises_runner_unavailable_on_ssh_nonzero(self):
        """ssh rc!=0 (e.g. 255 connection refused) → RunnerUnavailable."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner = self._make_runner([(0, '', ''), (255, '', 'ssh: connect to host laptop.local port 22')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

    async def test_raises_runner_unavailable_on_empty_stdout(self):
        """ssh rc=0 but stdout is empty → RunnerUnavailable (unparseable)."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner = self._make_runner([(0, '', ''), (0, '', '')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

    async def test_raises_runner_unavailable_on_non_json_stdout(self):
        """ssh rc=0 but stdout is non-JSON → RunnerUnavailable."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner = self._make_runner([(0, '', ''), (0, 'not valid json!!!', '')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

    async def test_raises_runner_unavailable_on_wrong_shape_json(self):
        """ssh rc=0 but stdout is valid JSON with wrong schema → RunnerUnavailable (TypeError path).

        This is the most likely real-world malformed-verdict case from a buggy
        remote CLI: the JSON parses but result_from_json raises TypeError because
        the keys don't match VerifyResult's fields.
        """
        from orchestrator.verify_runner import RunnerUnavailable
        # Valid JSON dict but unrecognised keys — triggers TypeError in result_from_json
        runner = self._make_runner([(0, '', ''), (0, '{"unexpected": 1}', '')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())
        # JSON list — also a TypeError because ** unpacking requires a mapping
        runner2 = self._make_runner([(0, '', ''), (0, '[1, 2, 3]', '')])
        with pytest.raises(RunnerUnavailable):
            await runner2.run_merge_verify('abc123', _make_spec())

    async def test_raises_runner_unavailable_when_run_raises_oserror(self):
        """An OSError from the subprocess runner → RunnerUnavailable."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner = self._make_runner([], raise_on={0: FileNotFoundError('git not found')})
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

    async def test_returns_result_when_verify_timed_out(self):
        """ssh rc=0 + stdout=VerifyResult(timed_out=True) → returned unchanged (NOT RunnerUnavailable)."""
        timed_out_result = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='',
            summary='timed out',
            timed_out=True,
        )
        runner = self._make_runner([(0, '', ''), (0, result_to_json(timed_out_result), '')])
        result = await runner.run_merge_verify('abc123', _make_spec())
        assert result.timed_out is True
        assert result == timed_out_result

    async def test_returns_failing_result_unchanged(self):
        """ssh rc=0 + stdout=VerifyResult(passed=False) → returned unchanged (NOT RunnerUnavailable)."""
        fail_result = VerifyResult(
            passed=False,
            test_output='FAILED 2',
            lint_output='',
            type_output='',
            summary='2 failures',
            category='test_failure',
        )
        runner = self._make_runner([(0, '', ''), (0, result_to_json(fail_result), '')])
        result = await runner.run_merge_verify('abc123', _make_spec())
        assert result.passed is False
        assert result == fail_result


# ---------------------------------------------------------------------------
# δ step-9: RemoteRunner ref cleanup (best-effort on return)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerRefCleanup:
    """The pushed ref is deleted best-effort on return (PRD open-Q4)."""

    def _make_tracking_runner(self, responses_by_argv_prefix):
        """Build a RemoteRunner whose fake `run` logs all calls.

        ``responses_by_argv_prefix`` maps an argv[0] to (rc, stdout, stderr).
        The fake always records every call in `runner._calls`.
        """
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            key = tuple(argv[:3])  # e.g. ('git','push','origin') or ('ssh', ...)
            # git delete looks like ['git','push','origin','--delete',...]
            if argv[:2] == ['git', 'push'] and '--delete' in argv:
                return (0, '', '')  # default: delete succeeds
            if key[0] == 'git':
                return responses_by_argv_prefix.get('git', (0, '', ''))
            if key[0] == 'ssh':
                return responses_by_argv_prefix.get('ssh', (0, result_to_json(_make_pass_result()), ''))
            return (0, '', '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'cleanup-id',
        )
        runner._calls = calls
        return runner

    async def test_delete_called_after_success(self):
        """After a successful run, the pushed ref is deleted via git push --delete."""
        runner = self._make_tracking_runner({'git': (0, '', ''), 'ssh': (0, result_to_json(_make_pass_result()), '')})
        await runner.run_merge_verify('abc123', _make_spec())
        delete_calls = [c for c in runner._calls if c[:2] == ['git', 'push'] and '--delete' in c]
        assert len(delete_calls) == 1
        assert 'refs/merge-verify/cleanup-id' in delete_calls[0]

    async def test_delete_called_after_ssh_failure(self):
        """When ssh fails (→ RunnerUnavailable), the ref is still deleted (cleanup in finally)."""
        from orchestrator.verify_runner import RunnerUnavailable

        async def fake_run(argv, *, cwd=None):
            runner._calls.append(argv[:])
            if argv[:2] == ['git', 'push'] and '--delete' in argv:
                return (0, '', '')
            if argv[0] == 'git':
                return (0, '', '')  # push succeeds
            return (255, '', 'ssh: connect refused')  # ssh fails

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'cleanup-id',
        )
        runner._calls = []
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())
        delete_calls = [c for c in runner._calls if c[:2] == ['git', 'push'] and '--delete' in c]
        assert len(delete_calls) == 1

    async def test_no_delete_when_push_failed(self):
        """When the git push itself fails, no delete is attempted (nothing was pushed)."""
        from orchestrator.verify_runner import RunnerUnavailable
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[:2] == ['git', 'push'] and '--delete' in argv:
                return (0, '', '')
            return (1, '', 'push error')  # ALL git push (incl. initial) fail

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'cleanup-id',
        )
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())
        delete_calls = [c for c in calls if c[:2] == ['git', 'push'] and '--delete' in c]
        assert len(delete_calls) == 0

    async def test_cleanup_failure_does_not_mask_result(self):
        """A delete call that raises does NOT change the returned VerifyResult."""
        pass_result = _make_pass_result()
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[:2] == ['git', 'push'] and '--delete' in argv:
                raise OSError('git delete failed')  # cleanup fails
            if argv[0] == 'git':
                return (0, '', '')  # push succeeds
            return (0, result_to_json(pass_result), '')  # ssh succeeds

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'cleanup-id',
        )
        result = await runner.run_merge_verify('abc123', _make_spec())
        assert result == pass_result  # no exception, correct result

    async def test_cleanup_failure_does_not_mask_runner_unavailable(self):
        """A delete that raises does NOT suppress a RunnerUnavailable from the verify path."""
        from orchestrator.verify_runner import RunnerUnavailable
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[:2] == ['git', 'push'] and '--delete' in argv:
                raise OSError('git delete failed')  # cleanup fails
            if argv[0] == 'git':
                return (0, '', '')  # push succeeds
            return (1, '', 'ssh error')  # ssh fails → RunnerUnavailable

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'cleanup-id',
        )
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())


# ---------------------------------------------------------------------------
# δ step-11: VerifyRunnerPool prefers remote runner over local
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyRunnerPoolPreferRemote:
    """VerifyRunnerPool.dispatch prefers the first non-local (remote) runner."""

    def _make_fake_runner(self, name, result):
        fake = MagicMock(spec=VerifyRunner)
        fake.name = name
        fake.is_local = (name == 'local')
        fake.run_merge_verify = AsyncMock(return_value=result)
        return fake

    async def test_dispatch_uses_remote_not_local_when_both_present(self):
        """[local, remote] order forces a real RED against the current runners[0] selection."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_result = _make_pass_result(summary='local result')
        remote_result = _make_pass_result(summary='remote result')
        local_fake = self._make_fake_runner('local', local_result)
        remote_fake = self._make_fake_runner('laptop', remote_result)

        pool = VerifyRunnerPool([local_fake, remote_fake])
        result = await pool.dispatch('sha1', _make_spec())

        assert result is remote_result
        remote_fake.run_merge_verify.assert_awaited_once()
        local_fake.run_merge_verify.assert_not_awaited()

    async def test_dispatch_event_runner_is_remote_name(self):
        """merge_verify event has data['runner'] == 'laptop' when routed to remote."""
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import VerifyRunnerPool
        remote_result = _make_pass_result()
        local_fake = self._make_fake_runner('local', _make_pass_result(summary='local'))
        remote_fake = self._make_fake_runner('laptop', remote_result)

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))

        pool = VerifyRunnerPool([local_fake, remote_fake], event_store=event_store, task_id='t-1')
        await pool.dispatch('sha1', _make_spec())

        assert len(emitted) == 1
        (event_type,), kwargs = emitted[0]
        assert event_type == EventType.merge_verify
        assert kwargs['data']['runner'] == 'laptop'

    async def test_dispatch_single_local_pool_uses_local(self):
        """Regression: single-runner pool [local] still routes to local (β regression guard)."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_result = _make_pass_result()
        local_fake = self._make_fake_runner('local', local_result)

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))

        pool = VerifyRunnerPool([local_fake], event_store=event_store)
        result = await pool.dispatch('sha2', _make_spec())

        assert result is local_result
        (_, kwargs) = emitted[0]
        assert kwargs['data']['runner'] == 'local'


# ---------------------------------------------------------------------------
# δ step-13: VerifyRunnerPool fail-safe fallback (PRD Invariant 2 / D5 / §B B3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyRunnerPoolFailSafe:
    """dispatch() falls back to local when the selected remote raises RunnerUnavailable."""

    def _make_fake_runner(self, name, result=None, raises=None):
        fake = MagicMock(spec=VerifyRunner)
        fake.name = name
        fake.is_local = (name == 'local')
        if raises is not None:
            fake.run_merge_verify = AsyncMock(side_effect=raises)
        else:
            fake.run_merge_verify = AsyncMock(return_value=result)
        return fake

    async def test_fallback_returns_local_result(self):
        """Remote RunnerUnavailable → dispatch returns local result."""
        from orchestrator.verify_runner import RunnerUnavailable, VerifyRunnerPool
        local_result = _make_pass_result(summary='local fallback')
        remote_fake = self._make_fake_runner('laptop', raises=RunnerUnavailable('host down'))
        local_fake = self._make_fake_runner('local', local_result)

        pool = VerifyRunnerPool([local_fake, remote_fake])
        result = await pool.dispatch('sha', _make_spec())

        assert result is local_result

    async def test_fallback_does_not_raise(self):
        """dispatch() does NOT propagate RunnerUnavailable when local fallback exists."""
        from orchestrator.verify_runner import RunnerUnavailable, VerifyRunnerPool
        remote_fake = self._make_fake_runner('laptop', raises=RunnerUnavailable('gone'))
        local_fake = self._make_fake_runner('local', _make_pass_result())
        pool = VerifyRunnerPool([local_fake, remote_fake])
        # must not raise
        await pool.dispatch('sha', _make_spec())

    async def test_fallback_calls_local_run_merge_verify_once(self):
        """local.run_merge_verify is called exactly once as the fallback."""
        from orchestrator.verify_runner import RunnerUnavailable, VerifyRunnerPool
        local_result = _make_pass_result()
        remote_fake = self._make_fake_runner('laptop', raises=RunnerUnavailable('down'))
        local_fake = self._make_fake_runner('local', local_result)

        pool = VerifyRunnerPool([local_fake, remote_fake])
        await pool.dispatch('sha', _make_spec())

        local_fake.run_merge_verify.assert_awaited_once()

    async def test_fallback_event_runner_is_local(self):
        """merge_verify event data['runner'] == 'local' when fallback is used."""
        from orchestrator.verify_runner import RunnerUnavailable, VerifyRunnerPool
        remote_fake = self._make_fake_runner('laptop', raises=RunnerUnavailable('down'))
        local_fake = self._make_fake_runner('local', _make_pass_result())

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))

        pool = VerifyRunnerPool([local_fake, remote_fake], event_store=event_store)
        await pool.dispatch('sha', _make_spec())

        assert len(emitted) == 1
        (_, kwargs) = emitted[0]
        assert kwargs['data']['runner'] == 'local'

    async def test_fallback_logs_one_warning(self, caplog):
        """Exactly one warning is logged identifying the unavailable runner; no escalation."""
        import logging

        from orchestrator.verify_runner import RunnerUnavailable, VerifyRunnerPool
        remote_fake = self._make_fake_runner('laptop', raises=RunnerUnavailable('host down'))
        local_fake = self._make_fake_runner('local', _make_pass_result())

        pool = VerifyRunnerPool([local_fake, remote_fake])
        with caplog.at_level(logging.WARNING):
            await pool.dispatch('sha', _make_spec())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'laptop' in warnings[0].message.lower() or 'laptop' in warnings[0].getMessage().lower()

    async def test_no_fallback_when_remote_returns_timed_out_result(self):
        """A VerifyResult(timed_out=True) from remote is returned unchanged — NOT fallen back (Invariant 5)."""
        from orchestrator.verify_runner import VerifyRunnerPool
        timed_out = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='timeout', timed_out=True,
        )
        remote_fake = self._make_fake_runner('laptop', timed_out)
        local_fake = self._make_fake_runner('local', _make_pass_result())

        pool = VerifyRunnerPool([local_fake, remote_fake])
        result = await pool.dispatch('sha', _make_spec())

        assert result is timed_out
        local_fake.run_merge_verify.assert_not_awaited()

    async def test_no_local_fallback_reraises_runner_unavailable(self):
        """A pool with only a remote runner and no local raises RunnerUnavailable (unsupported config)."""
        from orchestrator.verify_runner import RunnerUnavailable, VerifyRunnerPool
        remote_fake = self._make_fake_runner('laptop', raises=RunnerUnavailable('down'))

        pool = VerifyRunnerPool([remote_fake])
        with pytest.raises(RunnerUnavailable):
            await pool.dispatch('sha', _make_spec())


# ---------------------------------------------------------------------------
# ε step-1: EnvFingerprint frozen dataclass + JSON codec
# ---------------------------------------------------------------------------


class TestEnvFingerprint:
    """EnvFingerprint is a frozen dataclass with canonical JSON codec."""

    def _make_fp(self, **kwargs):
        from orchestrator.verify_runner import EnvFingerprint
        defaults = dict(
            toolchain='rustc 1.80.0 (abc123 2024-07-01)\ncargo 1.80.0',
            verify_env={'SCCACHE_BUCKET': 'builds', 'RUST_LOG': 'info'},
            sccache_reachable=True,
            extra_probes={'python_version': 'Python 3.11.9'},
        )
        defaults.update(kwargs)
        return EnvFingerprint(**defaults)  # type: ignore[arg-type]

    def test_frozen_raises_on_reassignment(self):
        fp = self._make_fp()
        with pytest.raises(dataclasses.FrozenInstanceError):
            fp.toolchain = 'other'  # type: ignore[misc]

    def test_verify_env_frozen_raises_on_reassignment(self):
        fp = self._make_fp()
        with pytest.raises(dataclasses.FrozenInstanceError):
            fp.verify_env = {}  # type: ignore[misc]

    def test_sccache_reachable_bool_field(self):
        fp_true = self._make_fp(sccache_reachable=True)
        fp_false = self._make_fp(sccache_reachable=False)
        assert fp_true.sccache_reachable is True
        assert fp_false.sccache_reachable is False

    def test_to_dict_from_dict_roundtrip(self):
        fp = self._make_fp()
        assert fp.from_dict(fp.to_dict()) == fp

    def test_to_dict_from_dict_empty_maps(self):
        fp = self._make_fp(verify_env={}, extra_probes={})
        assert fp.from_dict(fp.to_dict()) == fp

    def test_json_roundtrip_byte_identical(self):
        """to_json(from_json(s)) == s (byte-identical re-serialisation)."""
        from orchestrator.verify_runner import fingerprint_from_json, fingerprint_to_json
        fp = self._make_fp()
        s = fingerprint_to_json(fp)
        assert fingerprint_to_json(fingerprint_from_json(s)) == s

    def test_json_sort_keys_canonical(self):
        """Env built in reversed insertion order serialises identically (sort_keys)."""
        from orchestrator.verify_runner import fingerprint_to_json
        fp_fwd = self._make_fp(verify_env={'A': '1', 'B': '2', 'C': '3'})
        fp_rev = self._make_fp(verify_env={'C': '3', 'B': '2', 'A': '1'})
        assert fingerprint_to_json(fp_fwd) == fingerprint_to_json(fp_rev)

    def test_json_deterministic(self):
        """Same object serialises to the same bytes on repeated calls."""
        from orchestrator.verify_runner import fingerprint_to_json
        fp = self._make_fp()
        assert fingerprint_to_json(fp) == fingerprint_to_json(fp)

    def test_all_new_epsilon_names_in_dunder_all(self):
        """All ε-added public names are present in __all__ and importable."""
        import orchestrator.verify_runner as vr_mod
        expected = {
            'EnvFingerprint',
            'fingerprint_to_json',
            'fingerprint_from_json',
            'EnvParityVerdict',
            'compare_env_fingerprints',
            'capture_env_fingerprint',
            'ParityRow',
            'VerdictParityReport',
            'parity_report_to_json',
            'parity_report_from_json',
            'run_verdict_parity',
            'render_parity_report',
        }
        missing = expected - set(vr_mod.__all__)
        assert not missing, f"Missing from __all__: {sorted(missing)}"
        # Also verify each name resolves to a real attribute
        for name in expected:
            assert hasattr(vr_mod, name), f"__all__ lists {name!r} but attribute is absent"


# ---------------------------------------------------------------------------
# ε step-3: capture_env_fingerprint — probe commands via injected run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCaptureEnvFingerprint:
    """capture_env_fingerprint is async and probes through an injected run callable."""

    def _make_fake_run(self, responses):
        """Return async callable that returns successive (rc, stdout, stderr) tuples."""
        responses_queue = list(responses)
        issued = []

        async def fake_run(argv, *, cwd=None):
            issued.append(argv)
            return responses_queue.pop(0)

        run: Any = fake_run
        run.issued = issued
        return run

    async def test_toolchain_from_rustc_cargo_stdout(self):
        """toolchain == trimmed rustc + cargo --version stdout joined by newline."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0 (abc 2024-01-01)\n', ''),  # rustc --version
            (0, 'cargo 1.80.0 (def 2024-01-01)\n', ''),  # cargo --version
            (0, 'sccache stats\n', ''),                   # sccache --show-stats
        ])
        fp = await capture_env_fingerprint(fake_run)
        assert fp.toolchain == 'rustc 1.80.0 (abc 2024-01-01)\ncargo 1.80.0 (def 2024-01-01)'

    async def test_sccache_reachable_when_rc_zero(self):
        """sccache_reachable is True when sccache --show-stats returns rc==0."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (0, 'stats here\n', ''),   # sccache rc=0
        ])
        fp = await capture_env_fingerprint(fake_run)
        assert fp.sccache_reachable is True

    async def test_sccache_not_reachable_when_rc_nonzero(self):
        """sccache_reachable is False when sccache --show-stats returns rc!=0."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (1, '', 'connection refused'),  # sccache rc=1
        ])
        fp = await capture_env_fingerprint(fake_run)
        assert fp.sccache_reachable is False

    async def test_verify_env_carried_through_verbatim(self):
        """verify_env is embedded verbatim from the kwarg."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (0, '', ''),
        ])
        env = {'FOO': 'bar', 'SCCACHE_BUCKET': 'builds'}
        fp = await capture_env_fingerprint(fake_run, verify_env=env)
        assert dict(fp.verify_env) == env

    async def test_extra_probe_specs_populate_extra_probes(self):
        """extra_probe_specs (key, argv) pairs populate extra_probes with trimmed stdout."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (0, '', ''),                        # sccache
            (0, 'Python 3.11.9\n', ''),         # python_version probe
            (0, 'Ubuntu 22.04\n', ''),           # os_release probe
        ])
        probes = [
            ('python_version', ['python3', '--version']),
            ('os_release', ['lsb_release', '-d']),
        ]
        fp = await capture_env_fingerprint(fake_run, extra_probe_specs=probes)
        assert dict(fp.extra_probes) == {
            'python_version': 'Python 3.11.9',
            'os_release': 'Ubuntu 22.04',
        }

    async def test_extra_probe_unavailable_on_nonzero_rc(self):
        """When an extra probe exits non-zero, the value is '<unavailable rc=N>'."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (0, '', ''),          # sccache
            (2, '', 'not found'), # extra probe with rc=2
        ])
        probes = [('missing_tool', ['missing_tool', '--version'])]
        fp = await capture_env_fingerprint(fake_run, extra_probe_specs=probes)
        assert fp.extra_probes['missing_tool'] == '<unavailable rc=2>'

    async def test_exact_argv_lists_issued_to_run(self):
        """Assert the exact argv lists issued to the run callable."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (0, 'stats\n', ''),
            (0, 'Python 3.11.9\n', ''),
        ])
        probes = [('python_version', ['python3', '--version'])]
        await capture_env_fingerprint(fake_run, extra_probe_specs=probes)
        assert fake_run.issued == [
            ['rustc', '--version'],
            ['cargo', '--version'],
            ['sccache', '--show-stats'],
            ['python3', '--version'],
        ]

    async def test_no_extra_probes_by_default(self):
        """With no extra_probe_specs, extra_probes is an empty mapping."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (0, '', ''),
        ])
        fp = await capture_env_fingerprint(fake_run)
        assert dict(fp.extra_probes) == {}

    async def test_default_verify_env_empty(self):
        """With no verify_env kwarg, verify_env is an empty mapping."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fake_run = self._make_fake_run([
            (0, 'rustc 1.80.0\n', ''),
            (0, 'cargo 1.80.0\n', ''),
            (0, '', ''),
        ])
        fp = await capture_env_fingerprint(fake_run)
        assert dict(fp.verify_env) == {}


# ---------------------------------------------------------------------------
# ε step-5: compare_env_fingerprints → EnvParityVerdict
# ---------------------------------------------------------------------------


class TestCompareEnvFingerprints:
    """compare_env_fingerprints(local, remote) -> EnvParityVerdict."""

    def _fp(self, **kwargs):
        from orchestrator.verify_runner import EnvFingerprint
        base = dict(
            toolchain='rustc 1.80.0\ncargo 1.80.0',
            verify_env={'KEY': 'val'},
            sccache_reachable=True,
            extra_probes={'python_version': 'Python 3.11.9'},
        )
        base.update(kwargs)
        return EnvFingerprint(**base)  # type: ignore[arg-type]

    def test_identical_fingerprints_is_faithful(self):
        from orchestrator.verify_runner import compare_env_fingerprints
        fp = self._fp()
        verdict = compare_env_fingerprints(fp, fp)
        assert verdict.is_faithful is True
        assert verdict.drift_dimensions == ()

    def test_toolchain_mismatch_not_faithful(self):
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(toolchain='rustc 1.80.0\ncargo 1.80.0')
        remote = self._fp(toolchain='rustc 1.79.0\ncargo 1.79.0')
        verdict = compare_env_fingerprints(local, remote)
        assert verdict.is_faithful is False
        assert 'toolchain' in verdict.drift_dimensions

    def test_verify_env_mismatch_not_faithful(self):
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(verify_env={'FOO': 'bar'})
        remote = self._fp(verify_env={'FOO': 'baz'})
        verdict = compare_env_fingerprints(local, remote)
        assert verdict.is_faithful is False
        assert 'verify_env' in verdict.drift_dimensions

    def test_sccache_reachable_mismatch_not_faithful(self):
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(sccache_reachable=True)
        remote = self._fp(sccache_reachable=False)
        verdict = compare_env_fingerprints(local, remote)
        assert verdict.is_faithful is False
        assert 'sccache_reachable' in verdict.drift_dimensions

    def test_extra_probes_mismatch_not_faithful(self):
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(extra_probes={'python_version': 'Python 3.11.9'})
        remote = self._fp(extra_probes={'python_version': 'Python 3.12.0'})
        verdict = compare_env_fingerprints(local, remote)
        assert verdict.is_faithful is False
        assert 'extra_probes' in verdict.drift_dimensions

    def test_multi_dimension_drift_all_listed(self):
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(toolchain='rustc 1.80.0', sccache_reachable=True)
        remote = self._fp(toolchain='rustc 1.79.0', sccache_reachable=False)
        verdict = compare_env_fingerprints(local, remote)
        assert verdict.is_faithful is False
        assert 'toolchain' in verdict.drift_dimensions
        assert 'sccache_reachable' in verdict.drift_dimensions

    def test_only_differing_dimensions_listed(self):
        """Only the fields that differ appear in drift_dimensions."""
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(toolchain='rustc 1.80.0')
        remote = self._fp(toolchain='rustc 1.79.0')
        verdict = compare_env_fingerprints(local, remote)
        assert 'verify_env' not in verdict.drift_dimensions
        assert 'sccache_reachable' not in verdict.drift_dimensions
        assert 'extra_probes' not in verdict.drift_dimensions


# ---------------------------------------------------------------------------
# ε step-7: run_verdict_parity — all-agree path
# ---------------------------------------------------------------------------


def _make_verify_result(passed=True, category=''):
    return VerifyResult(
        passed=passed,
        test_output='',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category=category,
    )


@pytest.mark.asyncio
class TestRunVerdictParityAllAgree:
    """run_verdict_parity: all-agree path — both runners return identical verdicts."""

    def _make_fake_runner(self, name, results_by_sha):
        """Fake VerifyRunner whose run_merge_verify returns results keyed by sha."""
        fake = MagicMock(spec=VerifyRunner)
        fake.name = name
        fake.is_local = (name == 'local')

        async def run_merge_verify(sha, spec):
            return results_by_sha[sha]

        fake.run_merge_verify = AsyncMock(side_effect=run_merge_verify)
        return fake

    async def test_each_runner_called_once_per_sha(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('aaa', None), ('bbb', None)]
        local_fake = self._make_fake_runner('local', {
            'aaa': _make_verify_result(passed=True),
            'bbb': _make_verify_result(passed=False),
        })
        remote_fake = self._make_fake_runner('laptop', {
            'aaa': _make_verify_result(passed=True),
            'bbb': _make_verify_result(passed=False),
        })
        _report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert local_fake.run_merge_verify.await_count == 2
        assert remote_fake.run_merge_verify.await_count == 2

    async def test_runners_called_with_sha_and_spec(self):
        from orchestrator.verify_runner import run_verdict_parity
        spec = _make_spec()
        corpus = [('sha1', None)]
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result()})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result()})
        await run_verdict_parity(corpus, local_fake, remote_fake, spec)
        local_fake.run_merge_verify.assert_awaited_once_with('sha1', spec)
        remote_fake.run_merge_verify.assert_awaited_once_with('sha1', spec)

    async def test_report_has_one_row_per_sha(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('aaa', None), ('bbb', None), ('ccc', None)]
        results = {s: _make_verify_result() for s in ('aaa', 'bbb', 'ccc')}
        local_fake = self._make_fake_runner('local', results)
        remote_fake = self._make_fake_runner('laptop', results)
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert len(report.rows) == 3
        assert [r.sha for r in report.rows] == ['aaa', 'bbb', 'ccc']

    async def test_row_local_remote_passed_from_results(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', None)]
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=True)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=True)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        row = report.rows[0]
        assert row.local_passed is True
        assert row.remote_passed is True

    async def test_agree_true_when_both_match(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', None)]
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=False)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=False)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert report.rows[0].agree is True

    async def test_all_agree_true_when_every_row_agrees(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('aaa', None), ('bbb', None)]
        local_fake = self._make_fake_runner('local', {
            'aaa': _make_verify_result(passed=True),
            'bbb': _make_verify_result(passed=False),
        })
        remote_fake = self._make_fake_runner('laptop', {
            'aaa': _make_verify_result(passed=True),
            'bbb': _make_verify_result(passed=False),
        })
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert report.all_agree is True
        assert report.divergent_shas == ()

    async def test_empty_corpus_all_agree(self):
        from orchestrator.verify_runner import run_verdict_parity
        local_fake = self._make_fake_runner('local', {})
        remote_fake = self._make_fake_runner('laptop', {})
        report = await run_verdict_parity([], local_fake, remote_fake, _make_spec())
        assert report.all_agree is True
        assert report.rows == ()


# ---------------------------------------------------------------------------
# ε step-9: run_verdict_parity — divergence + expected-class coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunVerdictParityDivergence:
    """run_verdict_parity divergence detection and expected-class checks."""

    def _make_fake_runner(self, name, results_by_sha):
        fake = MagicMock(spec=VerifyRunner)
        fake.name = name
        fake.is_local = (name == 'local')

        async def run_merge_verify(sha, spec):
            return results_by_sha[sha]

        fake.run_merge_verify = AsyncMock(side_effect=run_merge_verify)
        return fake

    async def test_disagreeing_row_has_agree_false(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', None)]
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=True)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=False)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert report.rows[0].agree is False

    async def test_all_agree_false_when_one_disagreement(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', None), ('sha2', None)]
        local_fake = self._make_fake_runner('local', {
            'sha1': _make_verify_result(passed=True),
            'sha2': _make_verify_result(passed=True),
        })
        remote_fake = self._make_fake_runner('laptop', {
            'sha1': _make_verify_result(passed=False),  # disagrees
            'sha2': _make_verify_result(passed=True),   # agrees
        })
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert report.all_agree is False

    async def test_divergent_shas_contains_disagreeing_sha(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', None), ('sha2', None)]
        local_fake = self._make_fake_runner('local', {
            'sha1': _make_verify_result(passed=True),
            'sha2': _make_verify_result(passed=True),
        })
        remote_fake = self._make_fake_runner('laptop', {
            'sha1': _make_verify_result(passed=False),  # disagrees
            'sha2': _make_verify_result(passed=True),   # agrees
        })
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert 'sha1' in report.divergent_shas
        assert 'sha2' not in report.divergent_shas

    async def test_divergent_shas_exactly_the_disagreeing_ones(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('a', None), ('b', None), ('c', None)]
        local_fake = self._make_fake_runner('local', {
            'a': _make_verify_result(passed=True),
            'b': _make_verify_result(passed=False),
            'c': _make_verify_result(passed=True),
        })
        remote_fake = self._make_fake_runner('laptop', {
            'a': _make_verify_result(passed=False),  # disagrees
            'b': _make_verify_result(passed=False),  # agrees
            'c': _make_verify_result(passed=False),  # disagrees
        })
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert set(report.divergent_shas) == {'a', 'c'}

    async def test_expected_pass_none_matches_expected_is_none(self):
        """When expected_pass is None, row.matches_expected is None."""
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', None)]
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=True)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=True)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        assert report.rows[0].matches_expected is None

    async def test_matches_expected_true_when_agreed_verdict_matches(self):
        """matches_expected is True when both agree and verdict == expected_pass."""
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', True)]  # expected pass=True
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=True)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=True)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        row = report.rows[0]
        assert row.agree is True
        assert row.matches_expected is True

    async def test_matches_expected_false_when_verdict_does_not_match(self):
        """matches_expected is False when agreed verdict != expected_pass."""
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', True)]  # expected pass=True but both fail
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=False)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=False)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        row = report.rows[0]
        assert row.agree is True          # both agree on False
        assert row.matches_expected is False  # but expected True

    async def test_agree_independent_of_expected_pass(self):
        """agree is purely local-vs-remote; expected_pass does not affect it."""
        from orchestrator.verify_runner import run_verdict_parity
        # Both pass; expected was False — matches_expected=False but agree=True
        corpus = [('sha1', False)]
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=True)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=True)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        row = report.rows[0]
        assert row.agree is True
        assert row.matches_expected is False

    async def test_matches_expected_none_when_runners_disagree(self):
        """matches_expected is None when agree=False, even if expected_pass is set.

        When runners diverge there is no agreed verdict, so comparing to
        expected_pass would be semantically meaningless.
        """
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('sha1', True)]  # expected pass=True
        local_fake = self._make_fake_runner('local', {'sha1': _make_verify_result(passed=True)})
        remote_fake = self._make_fake_runner('laptop', {'sha1': _make_verify_result(passed=False)})
        report = await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())
        row = report.rows[0]
        assert row.agree is False
        assert row.matches_expected is None  # no agreed verdict to compare

    async def test_runner_error_records_errored_row_not_abort(self):
        """A runner exception for one SHA records an error row; the rest still run."""
        from orchestrator.verify_runner import run_verdict_parity

        class BoomRunner:
            name = 'boom'
            is_local = False

            async def run_merge_verify(self, sha, spec):
                if sha == 'bad':
                    raise RuntimeError("ssh connection refused")
                return _make_verify_result(passed=True)

        local_fake = self._make_fake_runner('local', {
            'bad': _make_verify_result(passed=True),
            'good': _make_verify_result(passed=True),
        })
        remote_boom = BoomRunner()
        corpus = [('bad', None), ('good', None)]
        report = await run_verdict_parity(corpus, local_fake, remote_boom, _make_spec())
        # Both SHAs produce rows — the error did NOT abort the run
        assert len(report.rows) == 2
        shas = {r.sha for r in report.rows}
        assert shas == {'bad', 'good'}
        # The errored row is marked as non-agreeing with error text in category
        bad_row = next(r for r in report.rows if r.sha == 'bad')
        assert bad_row.agree is False
        assert 'runner_error' in bad_row.remote_category
        # The good row is unaffected
        good_row = next(r for r in report.rows if r.sha == 'good')
        assert good_row.agree is True


# ---------------------------------------------------------------------------
# ε: VerdictParityReport JSON codec round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerdictParityReportCodec:
    """parity_report_to_json / parity_report_from_json round-trip byte-identical."""

    def _make_fake_runner(self, name, results_by_sha):
        fake = MagicMock(spec=VerifyRunner)
        fake.name = name
        fake.is_local = (name == 'local')

        async def run_merge_verify(sha, spec):
            return results_by_sha[sha]

        fake.run_merge_verify = AsyncMock(side_effect=run_merge_verify)
        return fake

    async def _build_report(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('abc', True), ('def', None), ('ghi', False)]
        results = {
            'abc': _make_verify_result(passed=True),
            'def': _make_verify_result(passed=False),
            'ghi': _make_verify_result(passed=False),
        }
        local_fake = self._make_fake_runner('local', results)
        remote_fake = self._make_fake_runner('laptop', results)
        return await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())

    async def test_parity_report_json_roundtrip_byte_identical(self):
        """to_json(from_json(s)) == s — byte-identical re-serialisation."""
        from orchestrator.verify_runner import parity_report_from_json, parity_report_to_json
        report = await self._build_report()
        s = parity_report_to_json(report)
        assert parity_report_to_json(parity_report_from_json(s)) == s

    async def test_parity_report_json_roundtrip_equals_original(self):
        """from_json(to_json(report)) == report — structural equality."""
        from orchestrator.verify_runner import parity_report_from_json, parity_report_to_json
        report = await self._build_report()
        assert parity_report_from_json(parity_report_to_json(report)) == report

    async def test_parity_report_to_dict_from_dict_roundtrip(self):
        """to_dict / from_dict round-trip preserves all rows and aggregates."""
        report = await self._build_report()
        restored = report.from_dict(report.to_dict())
        assert restored == report
        assert len(restored.rows) == 3


# ---------------------------------------------------------------------------
# ε step-11: render_parity_report — markdown structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRenderParityReport:
    """render_parity_report(report) -> str — markdown structure assertions."""

    def _make_fake_runner(self, name, results_by_sha):
        fake = MagicMock(spec=VerifyRunner)
        fake.name = name
        fake.is_local = (name == 'local')

        async def run_merge_verify(sha, spec):
            return results_by_sha[sha]

        fake.run_merge_verify = AsyncMock(side_effect=run_merge_verify)
        return fake

    async def _all_agree_report(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('abc123', True), ('def456', False)]
        local_fake = self._make_fake_runner('local', {
            'abc123': _make_verify_result(passed=True),
            'def456': _make_verify_result(passed=False),
        })
        remote_fake = self._make_fake_runner('laptop', {
            'abc123': _make_verify_result(passed=True),
            'def456': _make_verify_result(passed=False),
        })
        return await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())

    async def _diverging_report(self):
        from orchestrator.verify_runner import run_verdict_parity
        corpus = [('abc123', None), ('bad456', None)]
        local_fake = self._make_fake_runner('local', {
            'abc123': _make_verify_result(passed=True),
            'bad456': _make_verify_result(passed=True),
        })
        remote_fake = self._make_fake_runner('laptop', {
            'abc123': _make_verify_result(passed=True),
            'bad456': _make_verify_result(passed=False),  # disagrees
        })
        return await run_verdict_parity(corpus, local_fake, remote_fake, _make_spec())

    async def test_all_agree_headline_contains_pass_marker(self):
        from orchestrator.verify_runner import render_parity_report
        report = await self._all_agree_report()
        text = render_parity_report(report)
        assert 'PASS' in text or 'parity holds' in text.lower()

    async def test_divergence_headline_contains_divergence_marker(self):
        from orchestrator.verify_runner import render_parity_report
        report = await self._diverging_report()
        text = render_parity_report(report)
        assert 'DIVERGENCE' in text or 'diverge' in text.lower()

    async def test_results_table_contains_each_sha(self):
        from orchestrator.verify_runner import render_parity_report
        report = await self._all_agree_report()
        text = render_parity_report(report)
        assert 'abc123' in text
        assert 'def456' in text

    async def test_results_table_has_header_row(self):
        from orchestrator.verify_runner import render_parity_report
        report = await self._all_agree_report()
        text = render_parity_report(report)
        # Table header must mention sha, local, remote, agree
        assert 'sha' in text.lower()
        assert 'local' in text.lower()
        assert 'remote' in text.lower()
        assert 'agree' in text.lower()

    async def test_divergent_shas_listed_when_present(self):
        from orchestrator.verify_runner import render_parity_report
        report = await self._diverging_report()
        text = render_parity_report(report)
        assert 'bad456' in text

    async def test_divergent_shas_section_absent_when_all_agree(self):
        """No divergence callout section when all rows agree."""
        from orchestrator.verify_runner import render_parity_report
        report = await self._all_agree_report()
        text = render_parity_report(report)
        # The divergent SHA from the other corpus must NOT appear
        assert 'bad456' not in text

    async def test_one_table_row_per_corpus_sha(self):
        """Table has exactly one data row per corpus SHA (not counting header)."""
        from orchestrator.verify_runner import render_parity_report
        report = await self._all_agree_report()
        text = render_parity_report(report)
        # Count lines containing 'abc123' and 'def456' — each should appear once
        lines = text.splitlines()
        assert sum(1 for line in lines if 'abc123' in line) == 1
        assert sum(1 for line in lines if 'def456' in line) == 1


# ---------------------------------------------------------------------------
# ι step-1: VerifyRunnerPool quarantine — pool-level runner quarantine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyRunnerPoolQuarantine:
    """VerifyRunnerPool: quarantine(name) drops a runner from eligible dispatch."""

    def _make_fake_runner(self, name, result=None):
        fake = MagicMock(spec=VerifyRunner)
        fake.name = name
        fake.is_local = (name == 'local')
        fake.run_merge_verify = AsyncMock(return_value=result or _make_pass_result())
        return fake

    async def test_quarantine_sets_is_quarantined_true(self):
        from orchestrator.verify_runner import VerifyRunnerPool
        local_fake = self._make_fake_runner('local')
        laptop_fake = self._make_fake_runner('laptop')
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        pool.quarantine('laptop')
        assert pool.is_quarantined('laptop') is True

    async def test_is_quarantined_false_for_non_quarantined(self):
        from orchestrator.verify_runner import VerifyRunnerPool
        local_fake = self._make_fake_runner('local')
        laptop_fake = self._make_fake_runner('laptop')
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        assert pool.is_quarantined('laptop') is False

    async def test_dispatch_routes_to_local_when_remote_quarantined(self):
        """After quarantine('laptop'), dispatch runs local, not laptop."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_result = _make_pass_result(summary='local result')
        local_fake = self._make_fake_runner('local', local_result)
        laptop_fake = self._make_fake_runner('laptop', _make_pass_result(summary='laptop result'))
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        pool.quarantine('laptop')
        result = await pool.dispatch('sha1', _make_spec())
        assert result is local_result
        local_fake.run_merge_verify.assert_awaited_once()
        laptop_fake.run_merge_verify.assert_not_awaited()

    async def test_eligible_remote_returns_none_when_quarantined(self):
        """eligible_remote() returns None when the remote is quarantined."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_fake = self._make_fake_runner('local')
        laptop_fake = self._make_fake_runner('laptop')
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        pool.quarantine('laptop')
        assert pool.eligible_remote() is None

    async def test_eligible_remote_returns_runner_after_clear_quarantine(self):
        """eligible_remote() returns the runner after clear_quarantine."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_fake = self._make_fake_runner('local')
        laptop_fake = self._make_fake_runner('laptop')
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        pool.quarantine('laptop')
        pool.clear_quarantine('laptop')
        assert pool.eligible_remote() is laptop_fake

    async def test_local_runner_property_returns_is_local_runner(self):
        """pool.local_runner returns the runner with is_local=True."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_fake = self._make_fake_runner('local')
        laptop_fake = self._make_fake_runner('laptop')
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        assert pool.local_runner is local_fake

    async def test_quarantine_is_idempotent(self):
        """Quarantining the same runner twice doesn't cause errors."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_fake = self._make_fake_runner('local')
        laptop_fake = self._make_fake_runner('laptop')
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        pool.quarantine('laptop')
        pool.quarantine('laptop')
        assert pool.is_quarantined('laptop') is True

    async def test_clear_quarantine_removes_quarantine(self):
        """clear_quarantine('laptop') makes is_quarantined('laptop') False."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_fake = self._make_fake_runner('local')
        laptop_fake = self._make_fake_runner('laptop')
        pool = VerifyRunnerPool([local_fake, laptop_fake])
        pool.quarantine('laptop')
        pool.clear_quarantine('laptop')
        assert pool.is_quarantined('laptop') is False

    async def test_dispatch_routes_to_local_when_remote_first_and_quarantined(self):
        """[remote, local] ordering + quarantined remote → dispatch uses local (not runners[0])."""
        from orchestrator.verify_runner import VerifyRunnerPool
        local_result = _make_pass_result(summary='local result')
        local_fake = self._make_fake_runner('local', local_result)
        remote_fake = self._make_fake_runner('remote')  # is_local=False (name != 'local')
        # Adversarial ordering: remote is at index 0; old fallback would return runners[0] = remote.
        pool = VerifyRunnerPool([remote_fake, local_fake])
        pool.quarantine('remote')
        result = await pool.dispatch('sha1', _make_spec())
        assert result is local_result
        local_fake.run_merge_verify.assert_awaited_once()
        remote_fake.run_merge_verify.assert_not_awaited()


# ---------------------------------------------------------------------------
# ι step-3: DriftDetector agree path
# ---------------------------------------------------------------------------


def _make_drift_pool(local_result=None, remote_result=None):
    """Return a VerifyRunnerPool with injected fake runners."""
    from orchestrator.verify_runner import VerifyRunnerPool
    local_fake = MagicMock(spec=VerifyRunner)
    local_fake.name = 'local'
    local_fake.is_local = True
    local_fake.run_merge_verify = AsyncMock(return_value=local_result or _make_pass_result())
    remote_fake = MagicMock(spec=VerifyRunner)
    remote_fake.name = 'laptop'
    remote_fake.is_local = False
    remote_fake.run_merge_verify = AsyncMock(return_value=remote_result or _make_pass_result())
    pool = VerifyRunnerPool([local_fake, remote_fake])
    return pool, local_fake, remote_fake


@pytest.mark.asyncio
class TestDriftDetectorAgree:
    """DriftDetector.check(): agree path — both pass or both fail → AGREE + event."""

    async def test_both_pass_returns_agree_verdict(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = _make_drift_pool(
            local_result=_make_pass_result(), remote_result=_make_pass_result()
        )
        event_store = MagicMock()
        escalation_queue = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=escalation_queue, task_id='t')
        result = await detector.check('abc123', _make_spec())
        assert result.verdict == DriftVerdict.AGREE

    async def test_both_fail_returns_agree_verdict(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_fail_result()
        )
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=MagicMock(), task_id='t')
        result = await detector.check('abc123', _make_spec())
        assert result.verdict == DriftVerdict.AGREE

    async def test_agree_result_carries_local_remote_passed(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_pass_result(), remote_result=_make_pass_result()
        )
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=MagicMock())
        result = await detector.check('sha1', _make_spec())
        assert result.local_passed is True
        assert result.remote_passed is True

    async def test_agree_emits_exactly_one_verdict_parity_ok_event(self):
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, task_id='t')
        await detector.check('mysha', _make_spec())
        assert event_store.emit.call_count == 1
        call_args = event_store.emit.call_args
        assert call_args[0][0] == EventType.verdict_parity_ok

    async def test_agree_event_data_contains_merge_sha(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        await detector.check('mysha', _make_spec())
        data = event_store.emit.call_args[1]['data']
        assert data['merge_sha'] == 'mysha'

    async def test_agree_event_data_contains_runner_names(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        await detector.check('sha1', _make_spec())
        data = event_store.emit.call_args[1]['data']
        assert data['local_runner'] == 'local'
        assert data['remote_runner'] == 'laptop'

    async def test_agree_event_data_contains_agreed_passed(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_pass_result(), remote_result=_make_pass_result()
        )
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        await detector.check('sha1', _make_spec())
        data = event_store.emit.call_args[1]['data']
        assert data['passed'] is True

    async def test_both_fail_agree_event_data_passed_false(self):
        """Both FAIL agree path: event data['passed'] is False."""
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_fail_result()
        )
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        await detector.check('sha_both_fail', _make_spec())
        data = event_store.emit.call_args[1]['data']
        assert data['passed'] is False

    async def test_agree_submits_no_escalation(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool()
        escalation_queue = MagicMock()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        escalation_queue.submit.assert_not_called()

    async def test_agree_does_not_quarantine_remote(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, remote_fake = _make_drift_pool()
        detector = DriftDetector(pool)
        await detector.check('sha1', _make_spec())
        assert pool.is_quarantined('laptop') is False


# ---------------------------------------------------------------------------
# ι step-5: DriftDetector diverge path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDriftDetectorDivergence:
    """DriftDetector.check(): diverge path — local FAIL / remote PASS → DIVERGE + escalation + quarantine."""

    async def test_diverge_returns_diverge_verdict(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue, task_id='t')
        result = await detector.check('divergesha', _make_spec())
        assert result.verdict == DriftVerdict.DIVERGE

    async def test_diverge_submits_escalation(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('divergesha', _make_spec())
        escalation_queue.submit.assert_called_once()

    async def test_diverge_escalation_task_id_is_sentinel(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        esc = escalation_queue.submit.call_args[0][0]
        assert esc.task_id == '__drift__'

    async def test_diverge_escalation_level_1_severity_blocking(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        esc = escalation_queue.submit.call_args[0][0]
        assert esc.level == 1
        assert esc.severity == 'blocking'

    async def test_diverge_escalation_category_and_role(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        esc = escalation_queue.submit.call_args[0][0]
        assert esc.category == 'verify_drift_divergence'
        assert esc.agent_role == 'orchestrator-drift-detector'

    async def test_diverge_escalation_summary_mentions_sha(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('mydivergesha', _make_spec())
        esc = escalation_queue.submit.call_args[0][0]
        assert 'mydivergesha' in esc.summary

    async def test_diverge_quarantines_remote(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        assert pool.is_quarantined('laptop') is True

    async def test_diverge_emits_no_verdict_parity_ok_event(self):
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        event_store.emit.assert_not_called()

    async def test_diverge_local_pass_remote_fail_returns_diverge(self):
        """Local PASS / remote FAIL also yields DIVERGE + quarantine + escalation."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = _make_drift_pool(
            local_result=_make_pass_result(), remote_result=_make_fail_result()
        )
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-2')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        result = await detector.check('sha_local_pass', _make_spec())
        assert result.verdict == DriftVerdict.DIVERGE
        assert pool.is_quarantined('laptop') is True
        escalation_queue.submit.assert_called_once()


# ---------------------------------------------------------------------------
# ι step-7: DriftDetector dedup — has_open_l1 guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDriftDetectorDedup:
    """DriftDetector dedup: has_open_l1('__drift__') guard prevents double submission."""

    def _make_diverge_pool(self):
        """Pool where local fails, remote passes (the load-bearing divergence case)."""
        pool, local_fake, remote_fake = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        return pool, local_fake, remote_fake

    async def test_dedup_skips_submit_when_open_l1_exists(self):
        """has_open_l1 True → submit NOT called; quarantine still happens."""
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = self._make_diverge_pool()
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=True)
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        escalation_queue.submit.assert_not_called()

    async def test_dedup_still_quarantines_when_open_l1_exists(self):
        """Even when deduped, the remote runner must be quarantined."""
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = self._make_diverge_pool()
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=True)
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        assert pool.is_quarantined('laptop') is True

    async def test_dedup_still_returns_diverge_when_open_l1_exists(self):
        """Return DIVERGE regardless of dedup state."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = self._make_diverge_pool()
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=True)
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        result = await detector.check('sha1', _make_spec())
        assert result.verdict == DriftVerdict.DIVERGE

    async def test_no_dedup_submits_when_no_open_l1(self):
        """has_open_l1 False → submit called exactly once."""
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = self._make_diverge_pool()
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=False)
        escalation_queue.make_id = MagicMock(return_value='esc-__drift__-1')
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        escalation_queue.submit.assert_called_once()

    async def test_has_open_l1_called_with_drift_sentinel(self):
        """has_open_l1 must be called with the _DRIFT_SENTINEL constant value."""
        from orchestrator.verify_runner import _DRIFT_SENTINEL, DriftDetector
        pool, _, _ = self._make_diverge_pool()
        escalation_queue = MagicMock()
        escalation_queue.has_open_l1 = MagicMock(return_value=True)
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        escalation_queue.has_open_l1.assert_called_with(_DRIFT_SENTINEL)


# ---------------------------------------------------------------------------
# ι step-9: DriftDetector INCONCLUSIVE — Invariant 5 (transport ≠ divergence)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDriftDetectorInconclusive:
    """DriftDetector: transport failure or no eligible remote → INCONCLUSIVE, no side-effects."""

    async def test_runner_unavailable_returns_inconclusive(self):
        """Remote RunnerUnavailable → INCONCLUSIVE (not DIVERGE)."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, RunnerUnavailable
        pool, _, remote_fake = _make_drift_pool()
        remote_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('host down'))
        escalation_queue = MagicMock()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=escalation_queue)
        result = await detector.check('sha1', _make_spec())
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_runner_unavailable_submits_no_escalation(self):
        """Transport failure must NOT raise a drift alarm."""
        from orchestrator.verify_runner import DriftDetector, RunnerUnavailable
        pool, _, remote_fake = _make_drift_pool()
        remote_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('gone'))
        escalation_queue = MagicMock()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        escalation_queue.submit.assert_not_called()

    async def test_runner_unavailable_emits_no_event(self):
        """Transport failure must not emit verdict_parity_ok."""
        from orchestrator.verify_runner import DriftDetector, RunnerUnavailable
        pool, _, remote_fake = _make_drift_pool()
        remote_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('gone'))
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        await detector.check('sha1', _make_spec())
        event_store.emit.assert_not_called()

    async def test_runner_unavailable_does_not_quarantine_remote(self):
        """A closed/flaky laptop must not quarantine itself via transport failure."""
        from orchestrator.verify_runner import DriftDetector, RunnerUnavailable
        pool, _, remote_fake = _make_drift_pool()
        remote_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('flaky'))
        detector = DriftDetector(pool)
        await detector.check('sha1', _make_spec())
        assert pool.is_quarantined('laptop') is False

    async def test_single_local_pool_returns_inconclusive(self):
        """Pool with only local runner (no eligible remote) → INCONCLUSIVE."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, VerifyRunnerPool
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        local_fake.run_merge_verify = AsyncMock(return_value=_make_pass_result())
        pool = VerifyRunnerPool([local_fake])
        escalation_queue = MagicMock()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=escalation_queue)
        result = await detector.check('sha1', _make_spec())
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_single_local_pool_no_escalation(self):
        """No eligible remote → no escalation."""
        from orchestrator.verify_runner import DriftDetector, VerifyRunnerPool
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        local_fake.run_merge_verify = AsyncMock(return_value=_make_pass_result())
        pool = VerifyRunnerPool([local_fake])
        escalation_queue = MagicMock()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        escalation_queue.submit.assert_not_called()

    async def test_quarantined_remote_pool_returns_inconclusive(self):
        """Pool with remote already quarantined → no eligible remote → INCONCLUSIVE."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = _make_drift_pool()
        pool.quarantine('laptop')
        escalation_queue = MagicMock()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        result = await detector.check('sha1', _make_spec())
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_local_runner_unavailable_returns_inconclusive(self):
        """Local RunnerUnavailable → INCONCLUSIVE (symmetric with remote, no false alarm)."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, RunnerUnavailable
        pool, local_fake, _ = _make_drift_pool()
        local_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('local down'))
        escalation_queue = MagicMock()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=escalation_queue)
        result = await detector.check('sha1', _make_spec())
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_local_runner_unavailable_submits_no_escalation(self):
        """Local transport failure must NOT raise a drift alarm."""
        from orchestrator.verify_runner import DriftDetector, RunnerUnavailable
        pool, local_fake, _ = _make_drift_pool()
        local_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('local gone'))
        escalation_queue = MagicMock()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('sha1', _make_spec())
        escalation_queue.submit.assert_not_called()

    async def test_local_runner_unavailable_does_not_quarantine_remote(self):
        """Local transport failure must not quarantine the remote runner."""
        from orchestrator.verify_runner import DriftDetector, RunnerUnavailable
        pool, local_fake, _ = _make_drift_pool()
        local_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('local flaky'))
        detector = DriftDetector(pool)
        await detector.check('sha1', _make_spec())
        assert pool.is_quarantined('laptop') is False


# ---------------------------------------------------------------------------
# ι step-11: DriftDetector cadence predicate — should_sample
# ---------------------------------------------------------------------------


class TestDriftDetectorCadence:
    """DriftDetector.should_sample: every-Nth-land pure predicate (PRD §10 Open Q2)."""

    def _make_detector(self, every_n_lands=20):
        from orchestrator.verify_runner import DriftDetector, VerifyRunnerPool
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        pool = VerifyRunnerPool([local_fake])
        return DriftDetector(pool, every_n_lands=every_n_lands)

    def test_samples_on_multiples_of_20(self):
        detector = self._make_detector(every_n_lands=20)
        assert detector.should_sample(20) is True
        assert detector.should_sample(40) is True
        assert detector.should_sample(60) is True

    def test_does_not_sample_on_non_multiples(self):
        detector = self._make_detector(every_n_lands=20)
        assert detector.should_sample(0) is False
        assert detector.should_sample(1) is False
        assert detector.should_sample(19) is False
        assert detector.should_sample(21) is False

    def test_does_not_sample_on_zero(self):
        detector = self._make_detector(every_n_lands=20)
        assert detector.should_sample(0) is False

    def test_custom_every_n_lands_5(self):
        detector = self._make_detector(every_n_lands=5)
        assert detector.should_sample(5) is True
        assert detector.should_sample(10) is True
        assert detector.should_sample(15) is True
        assert detector.should_sample(1) is False
        assert detector.should_sample(4) is False
        assert detector.should_sample(6) is False

    def test_every_n_lands_zero_raises(self):
        """DriftDetector(pool, every_n_lands=0) raises ValueError at construction time."""
        with pytest.raises(ValueError, match='every_n_lands'):
            self._make_detector(every_n_lands=0)

    def test_every_n_lands_negative_raises(self):
        """DriftDetector(pool, every_n_lands=-1) raises ValueError at construction time."""
        with pytest.raises(ValueError, match='every_n_lands'):
            self._make_detector(every_n_lands=-1)


# ---------------------------------------------------------------------------
# ι step-13: public-surface __all__ — ι additions present and importable
# ---------------------------------------------------------------------------


class TestDriftDetectorPublicSurface:
    """All ι-added public names are present in __all__ and importable."""

    def test_all_new_iota_names_in_dunder_all(self):
        import orchestrator.verify_runner as vr_mod
        expected = {'DriftDetector', 'DriftVerdict', 'DriftCheckResult'}
        missing = expected - set(vr_mod.__all__)
        assert not missing, f"Missing from __all__: {sorted(missing)}"
        for name in expected:
            assert hasattr(vr_mod, name), f"__all__ lists {name!r} but attribute is absent"


# ---------------------------------------------------------------------------
# κ step-9: SccacheStats + parse_sccache_stats
# ---------------------------------------------------------------------------

_SCCACHE_STATS_REDIS = """\
Sccache statistics
    Compile requests    10
    Cache hits          3
    Cache hits (Rust)   3
    Cache misses        1
    Cache timeouts      0
    Cache read errors   0
    Forced recaches     0
    Cache write errors  0
    Cache location      Redis: redis://orch:6379
    Cache size          0 bytes
    Max cache size      10 GiB
"""

_SCCACHE_STATS_LOCAL = """\
Sccache statistics
    Compile requests    5
    Cache hits          5
    Cache hits (Rust)   5
    Cache misses        0
    Cache location      Local disk: /home/u/.cache/sccache
"""


class TestParseSccacheStats:
    """parse_sccache_stats parses sccache --show-stats output into SccacheStats."""

    def test_compile_requests_hits_misses_parsed(self):
        from orchestrator.verify_runner import SccacheStats, parse_sccache_stats
        stats = parse_sccache_stats(_SCCACHE_STATS_REDIS)
        assert isinstance(stats, SccacheStats)
        assert stats.compile_requests == 10
        assert stats.cache_hits == 3
        assert stats.cache_misses == 1

    def test_cache_location_redis(self):
        from orchestrator.verify_runner import parse_sccache_stats
        stats = parse_sccache_stats(_SCCACHE_STATS_REDIS)
        assert stats.cache_location.startswith('Redis')

    def test_hit_rate_exact_arithmetic(self):
        from orchestrator.verify_runner import parse_sccache_stats
        stats = parse_sccache_stats(_SCCACHE_STATS_REDIS)
        # hits=3, misses=1 → 3/(3+1) = 0.75 exactly
        assert stats.hit_rate == 0.75

    def test_is_shared_backend_true_for_redis(self):
        from orchestrator.verify_runner import parse_sccache_stats
        stats = parse_sccache_stats(_SCCACHE_STATS_REDIS)
        assert stats.is_shared_backend is True

    def test_remote_hit_rate_positive_for_redis(self):
        from orchestrator.verify_runner import parse_sccache_stats
        stats = parse_sccache_stats(_SCCACHE_STATS_REDIS)
        assert stats.remote_hit_rate == 0.75
        assert stats.remote_hits == 3

    def test_local_disk_is_not_shared_backend(self):
        from orchestrator.verify_runner import parse_sccache_stats
        stats = parse_sccache_stats(_SCCACHE_STATS_LOCAL)
        assert stats.is_shared_backend is False
        assert stats.remote_hit_rate == 0.0
        assert stats.remote_hits == 0

    def test_local_disk_hit_rate_can_be_one(self):
        """hit_rate for local disk may be 1.0 (no misses); remote_hit_rate is still 0."""
        from orchestrator.verify_runner import parse_sccache_stats
        stats = parse_sccache_stats(_SCCACHE_STATS_LOCAL)
        assert stats.hit_rate == 1.0  # 5/(5+0)

    def test_exact_label_guard_aggregate_not_rust_breakdown(self):
        """cache_hits picks the 'Cache hits' aggregate, NOT 'Cache hits (Rust)'."""
        from orchestrator.verify_runner import parse_sccache_stats
        # blob has Cache hits = 3 but Cache hits (Rust) would also = 3 in this sample;
        # if the parser mistook the Rust line for the aggregate it would give the same
        # value — use a blob where they differ to make the guard meaningful.
        blob = """\
    Compile requests    10
    Cache hits          7
    Cache hits (Rust)   3
    Cache misses        3
    Cache location      Redis: redis://orch:6379
"""
        stats = parse_sccache_stats(blob)
        assert stats.cache_hits == 7, (
            "cache_hits must be 7 (the 'Cache hits' aggregate), not 3 ('Cache hits (Rust)')"
        )

    def test_hit_rate_zero_when_no_hits_no_misses(self):
        """Denominator 0 → hit_rate 0.0 (no division by zero)."""
        from orchestrator.verify_runner import parse_sccache_stats
        blob = """\
    Compile requests    0
    Cache hits          0
    Cache misses        0
    Cache location      Redis: redis://orch:6379
"""
        stats = parse_sccache_stats(blob)
        assert stats.hit_rate == 0.0


# ---------------------------------------------------------------------------
# κ step-11: capture_sccache_stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCaptureSccacheStats:
    """capture_sccache_stats is async and uses the injected run callable."""

    def _make_fake_run(self, responses):
        """Return async callable that returns successive (rc, stdout, stderr) tuples."""
        responses_queue = list(responses)
        issued = []

        async def fake_run(argv, *, cwd=None):
            issued.append(argv)
            return responses_queue.pop(0)

        run: Any = fake_run
        run.issued = issued
        return run

    async def test_parses_redis_stats_blob(self):
        from orchestrator.verify_runner import SccacheStats, capture_sccache_stats
        fake_run = self._make_fake_run([(0, _SCCACHE_STATS_REDIS, '')])
        stats = await capture_sccache_stats(fake_run)
        assert isinstance(stats, SccacheStats)
        assert stats.remote_hit_rate > 0

    async def test_exact_argv_issued(self):
        from orchestrator.verify_runner import capture_sccache_stats
        fake_run = self._make_fake_run([(0, _SCCACHE_STATS_REDIS, '')])
        await capture_sccache_stats(fake_run)
        assert fake_run.issued == [['sccache', '--show-stats']]

    async def test_probe_ok_true_on_zero_rc(self):
        """probe_ok is True when sccache --show-stats exits 0."""
        from orchestrator.verify_runner import capture_sccache_stats
        fake_run = self._make_fake_run([(0, _SCCACHE_STATS_REDIS, '')])
        stats = await capture_sccache_stats(fake_run)
        assert stats.probe_ok is True

    async def test_probe_ok_false_on_nonzero_rc(self):
        """probe_ok is False when sccache --show-stats exits non-zero (daemon absent).

        This distinguishes a probe failure from a legitimately cold shared
        cache (which would also yield all-zero stats but with probe_ok=True).
        """
        from orchestrator.verify_runner import SccacheStats, capture_sccache_stats
        fake_run = self._make_fake_run([(127, '', 'sccache: not found')])
        stats = await capture_sccache_stats(fake_run)
        assert isinstance(stats, SccacheStats)
        assert stats.probe_ok is False


# ---------------------------------------------------------------------------
# κ step-13: ColdWarmVerifyDelta
# ---------------------------------------------------------------------------


class TestColdWarmVerifyDelta:
    """ColdWarmVerifyDelta — frozen value object with JSON codec."""

    def test_speedup_exact_arithmetic(self):
        from orchestrator.verify_runner import ColdWarmVerifyDelta
        d = ColdWarmVerifyDelta(cold_secs=300.0, warm_secs=100.0)
        assert d.speedup == 3.0  # exact: 300/100

    def test_frozen(self):
        from orchestrator.verify_runner import ColdWarmVerifyDelta
        d = ColdWarmVerifyDelta(cold_secs=300.0, warm_secs=100.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.cold_secs = 999.0  # type: ignore[misc]

    def test_to_dict_from_dict_round_trip(self):
        from orchestrator.verify_runner import ColdWarmVerifyDelta
        d = ColdWarmVerifyDelta(cold_secs=300.0, warm_secs=100.0)
        assert ColdWarmVerifyDelta.from_dict(d.to_dict()) == d

    def test_json_codec_round_trip(self):
        from orchestrator.verify_runner import ColdWarmVerifyDelta, delta_from_json, delta_to_json
        d = ColdWarmVerifyDelta(cold_secs=300.0, warm_secs=100.0)
        assert delta_from_json(delta_to_json(d)) == d

    def test_json_sort_keys(self):
        """JSON output must have sort_keys=True (canonical form)."""
        from orchestrator.verify_runner import ColdWarmVerifyDelta, delta_to_json
        d = ColdWarmVerifyDelta(cold_secs=300.0, warm_secs=100.0)
        serialised = delta_to_json(d)
        parsed = json.loads(serialised)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_warm_equal_cold_speedup_one(self):
        """~1× warm expectation: speedup=1.0 when warm==cold (PRD G6, no threshold)."""
        from orchestrator.verify_runner import ColdWarmVerifyDelta
        d = ColdWarmVerifyDelta(cold_secs=200.0, warm_secs=200.0)
        assert d.speedup == 1.0

    def test_warm_zero_speedup_zero(self):
        """warm_secs=0 returns 0.0 (guard, documented in docstring)."""
        from orchestrator.verify_runner import ColdWarmVerifyDelta
        d = ColdWarmVerifyDelta(cold_secs=300.0, warm_secs=0.0)
        assert d.speedup == 0.0


# ---------------------------------------------------------------------------
# κ step-15: public-surface __all__ — κ additions present and importable
# ---------------------------------------------------------------------------


class TestSccacheKappaPublicSurface:
    """All κ-added verify_runner public names are in __all__ and importable."""

    def test_all_new_kappa_names_in_dunder_all(self):
        import orchestrator.verify_runner as vr_mod
        expected = {
            'SccacheStats',
            'parse_sccache_stats',
            'capture_sccache_stats',
            'ColdWarmVerifyDelta',
            'delta_to_json',
            'delta_from_json',
        }
        missing = expected - set(vr_mod.__all__)
        assert not missing, f"Missing from __all__: {sorted(missing)}"
        for name in expected:
            assert hasattr(vr_mod, name), f"__all__ lists {name!r} but attribute is absent"


# ---------------------------------------------------------------------------
# step-3: RemoteRunner.run_merge_verify — main_branch best-effort push
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerMainBranchPush:
    """Tests for RemoteRunner main_branch parameter (opt-in best-effort main push)."""

    def _make_runner_and_calls(self, expected_result, *, main_branch=None, config_path=None):
        """Return (runner, calls) where calls tracks (argv, cwd) pairs."""
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append((argv, cwd))
            if argv[0] == 'git':
                return (0, '', '')
            # ssh
            return (0, result_to_json(expected_result), '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            config_path=config_path,
            main_branch=main_branch,
            run=fake_run,
            id_factory=lambda: 'fixed-id',
        )
        return runner, calls

    async def test_with_main_branch_main_push_is_second(self):
        """With main_branch='main', calls[0] is git rev-parse and calls[1] is the main push.

        β dedup: a rev-parse is now issued BEFORE the best-effort main push so the runner
        can skip the push when the sha is unchanged.  On the first call _last_pushed_main_sha
        is None, so the push always fires; the ordering shifts by one slot.
        """
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected, main_branch='main')
        await runner.run_merge_verify('abc123', _make_spec())
        # calls[0] must be the rev-parse (leading dedup probe)
        rev_argv, rev_cwd = calls[0]
        assert rev_argv == ['git', 'rev-parse', 'main']
        assert rev_cwd == '/repo'
        # calls[1] must be the main-branch push
        push_argv, push_cwd = calls[1]
        assert push_argv == ['git', 'push', 'origin', 'main:refs/heads/main']
        assert push_cwd == '/repo'
        # calls[2] must be the merge-sha push
        merge_push_argv, _ = calls[2]
        assert merge_push_argv == ['git', 'push', 'origin', 'abc123:refs/merge-verify/fixed-id']

    async def test_with_main_branch_ssh_is_fourth(self):
        """With main_branch set, ssh invocation is calls[3] (after rev-parse + two git pushes).

        β dedup: rev-parse is now calls[0], shifting ssh from calls[2] to calls[3].
        """
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected, main_branch='main')
        await runner.run_merge_verify('abc123', _make_spec())
        ssh_argv, _ = calls[3]
        assert ssh_argv[0] == 'ssh'

    async def test_without_main_branch_calls0_is_merge_sha_push(self):
        """main_branch=None (default): calls[0] is the merge-sha push (byte-identical to prior behaviour)."""
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected, main_branch=None)
        await runner.run_merge_verify('abc123', _make_spec())
        push_argv, _ = calls[0]
        assert push_argv == ['git', 'push', 'origin', 'abc123:refs/merge-verify/fixed-id']

    async def test_main_push_failure_is_non_fatal(self):
        """main-push rc!=0 is swallowed; merge-sha push + ssh still succeed and return VerifyResult."""
        expected = VerifyResult(passed=True, test_output='ok', lint_output='', type_output='', summary='ok')
        calls = []
        call_count = [0]

        async def fake_run(argv, *, cwd=None):
            calls.append((argv, cwd))
            call_count[0] += 1
            # First git call = main push → rc=1 (non-fast-forward)
            if argv[0] == 'git' and 'refs/heads/' in (argv[3] if len(argv) > 3 else ''):
                return (1, '', 'rejected: non-fast-forward')
            # git push merge-sha or cleanup → rc=0
            if argv[0] == 'git':
                return (0, '', '')
            # ssh → valid result
            return (0, result_to_json(expected), '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            main_branch='main',
            run=fake_run,
            id_factory=lambda: 'req-id',
        )
        # Must NOT raise RunnerUnavailable — main-push failure is non-fatal
        result = await runner.run_merge_verify('abc123', _make_spec())
        assert result == expected
        # merge-sha push was still issued
        push_argvs = [c[0] for c in calls if c[0][0] == 'git' and 'refs/merge-verify/' in (c[0][3] if len(c[0]) > 3 else '')]
        assert len(push_argvs) >= 1

    async def test_merge_sha_push_failure_still_raises_runner_unavailable(self):
        """Even with main_branch set, a merge-sha push failure raises RunnerUnavailable."""
        from orchestrator.verify_runner import RunnerUnavailable

        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv)
            # main push → ok
            if argv[0] == 'git' and len(argv) > 3 and 'refs/heads/' in argv[3]:
                return (0, '', '')
            # merge-sha push → fail
            if argv[0] == 'git' and len(argv) > 3 and 'refs/merge-verify/' in argv[3]:
                return (1, '', 'rejected')
            return (0, '', '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            main_branch='main',
            run=fake_run,
            id_factory=lambda: 'req-id',
        )
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())


# ---------------------------------------------------------------------------
# β step-11: RemoteRunner --request-id threading + _inflight_request_id lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerRequestId:
    """run_merge_verify passes --request-id <id> appended to the ssh argv."""

    def _make_runner_and_calls(self, *, config_path=None):
        """Return (runner, calls) where calls tracks (argv, cwd) pairs."""
        calls = []
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')

        async def fake_run(argv, *, cwd=None):
            calls.append((argv[:], cwd))
            if argv[0] == 'git':
                return (0, '', '')
            return (0, result_to_json(expected), '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            config_path=config_path,
            run=fake_run,
            id_factory=lambda: 'fixed-id',
        )
        return runner, calls

    async def test_request_id_in_remote_cmd(self):
        """ssh remote command contains --request-id fixed-id (appended)."""
        import shlex as _shlex

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        remote_cmd = ssh_call[0][-1]
        parsed = _shlex.split(remote_cmd)

        rid_idx = parsed.index('--request-id')
        assert parsed[rid_idx + 1] == 'fixed-id'

    async def test_request_id_same_as_push_ref_id(self):
        """--request-id value equals the id in the push ref (abc123:refs/merge-verify/<id>)."""
        import shlex as _shlex

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        push_call = next(
            c for c in calls
            if c[0][0] == 'git' and len(c[0]) > 3 and 'refs/merge-verify/' in c[0][3]
        )
        ref_part = push_call[0][3]
        push_id = ref_part.split('/')[-1]

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = _shlex.split(ssh_call[0][-1])
        rid_idx = parsed.index('--request-id')
        ssh_id = parsed[rid_idx + 1]

        assert push_id == ssh_id == 'fixed-id'

    async def test_existing_positional_args_unchanged(self):
        """parsed[:4] still == ['orchestrator', 'verify-merge', '--sha', 'abc123']."""
        import shlex as _shlex

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = _shlex.split(ssh_call[0][-1])
        assert parsed[:4] == ['orchestrator', 'verify-merge', '--sha', 'abc123']

    async def test_request_id_appended_after_spec(self):
        """--request-id appears after --spec (appended at end)."""
        import shlex as _shlex

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = _shlex.split(ssh_call[0][-1])

        spec_idx = parsed.index('--spec')
        rid_idx = parsed.index('--request-id')
        assert rid_idx > spec_idx, '--request-id must be appended after --spec'

    async def test_request_id_with_config_path(self):
        """--request-id is still appended after --config when config_path is set."""
        import shlex as _shlex

        runner, calls = self._make_runner_and_calls(config_path='/etc/orch.yaml')
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = _shlex.split(ssh_call[0][-1])

        cfg_idx = parsed.index('--config')
        rid_idx = parsed.index('--request-id')
        assert rid_idx > cfg_idx, '--request-id must come after --config'

    async def test_inflight_request_id_cleared_after_return(self):
        """_inflight_request_id is None after run_merge_verify returns."""
        runner, _ = self._make_runner_and_calls()
        assert runner._inflight_request_id is None
        await runner.run_merge_verify('abc123', _make_spec())
        assert runner._inflight_request_id is None

    async def test_inflight_request_id_cleared_after_exception(self):
        """_inflight_request_id is cleared in the finally even on RunnerUnavailable."""
        from orchestrator.verify_runner import RunnerUnavailable

        calls = []

        async def fail_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[0] == 'git' and len(argv) > 3 and 'refs/merge-verify/' in argv[3]:
                return (1, '', 'push rejected')
            return (0, '', '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fail_run,
            id_factory=lambda: 'fail-id',
        )
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

        assert runner._inflight_request_id is None


# ---------------------------------------------------------------------------
# β step-13: RemoteRunner.cancel_verify() and probe_clean()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerCancelVerify:
    """cancel_verify() issues ssh cancel-verify; probe_clean() issues ssh pgrep."""

    def _make_runner(self, *, config_path=None, cancel_rc=0, probe_rc=1):
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[0] == 'ssh':
                if 'pgrep' in argv[-1]:
                    return (probe_rc, '', '')
                return (cancel_rc, '', '')
            return (0, '', '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            config_path=config_path,
            run=fake_run,
            id_factory=lambda: 'req-42',
        )
        runner._calls = calls
        return runner

    async def test_cancel_verify_no_inflight_returns_zero_no_ssh(self):
        """cancel_verify() with _inflight_request_id=None returns 0 without issuing ssh."""
        runner = self._make_runner()
        rc = await runner.cancel_verify()
        assert rc == 0
        ssh_calls = [c for c in runner._calls if c[0] == 'ssh']
        assert len(ssh_calls) == 0

    async def test_cancel_verify_issues_correct_argv(self):
        """cancel_verify() issues ssh BatchMode/ConnectTimeout cancel-verify --request-id."""
        import shlex as _shlex

        runner = self._make_runner(cancel_rc=0)
        runner._inflight_request_id = 'req-42'
        await runner.cancel_verify()

        ssh_calls = [c for c in runner._calls if c[0] == 'ssh']
        assert len(ssh_calls) == 1
        argv = ssh_calls[0]

        assert 'BatchMode=yes' in argv
        assert 'laptop.local' in argv

        remote_cmd = argv[-1]
        parsed = _shlex.split(remote_cmd)
        assert 'cancel-verify' in parsed
        rid_idx = parsed.index('--request-id')
        assert parsed[rid_idx + 1] == 'req-42'

    async def test_cancel_verify_appends_config_when_set(self):
        """cancel_verify() appends --config <path> when config_path is set."""
        import shlex as _shlex

        runner = self._make_runner(config_path='/etc/orch.yaml', cancel_rc=0)
        runner._inflight_request_id = 'req-42'
        await runner.cancel_verify()

        ssh_calls = [c for c in runner._calls if c[0] == 'ssh']
        remote_cmd = ssh_calls[0][-1]
        parsed = _shlex.split(remote_cmd)
        cfg_idx = parsed.index('--config')
        assert parsed[cfg_idx + 1] == '/etc/orch.yaml'

    async def test_cancel_verify_returns_ssh_rc(self):
        """cancel_verify() returns the ssh return code."""
        runner = self._make_runner(cancel_rc=1)
        runner._inflight_request_id = 'req-42'
        rc = await runner.cancel_verify()
        assert rc == 1

    async def test_probe_clean_true_when_pgrep_rc_is_1(self):
        """probe_clean() issues ssh pgrep -f verify-merge; rc==1 (no match) → True."""
        runner = self._make_runner(probe_rc=1)
        result = await runner.probe_clean()
        assert result is True

        ssh_calls = [c for c in runner._calls if c[0] == 'ssh']
        assert len(ssh_calls) == 1
        remote_cmd = ssh_calls[0][-1]
        assert 'pgrep' in remote_cmd
        assert 'verify-merge' in remote_cmd

    async def test_probe_clean_false_when_pgrep_rc_is_0(self):
        """probe_clean() rc==0 (process running) → False."""
        runner = self._make_runner(probe_rc=0)
        result = await runner.probe_clean()
        assert result is False

    async def test_probe_clean_false_when_pgrep_rc_is_2(self):
        """probe_clean() rc>=2 (error) → False (conservative: stay parked)."""
        runner = self._make_runner(probe_rc=2)
        result = await runner.probe_clean()
        assert result is False

    async def test_probe_clean_false_on_oserror(self):
        """probe_clean() on OSError → False (fail-safe)."""
        async def raising_run(argv, *, cwd=None):
            raise OSError('connection refused')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=raising_run,
        )
        result = await runner.probe_clean()
        assert result is False


# ---------------------------------------------------------------------------
# β step-15: RemoteRunner _last_pushed_main_sha dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerMainBranchDedup:
    """_last_pushed_main_sha dedup: skip main push when main sha is unchanged."""

    def _make_runner_with_tracking(self, rev_parse_shas):
        """Return (runner, push_counts) keyed by 'main' and 'merge'."""
        push_counts: dict[str, int] = {'main': 0, 'merge': 0}
        rev_parse_iter = iter(rev_parse_shas)
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')

        async def fake_run(argv, *, cwd=None):
            if argv[0] == 'git' and argv[1] == 'rev-parse':
                sha = next(rev_parse_iter, 'sha-stable')
                return (0, sha, '')
            if argv[0] == 'git' and argv[1] == 'push':
                ref = argv[3] if len(argv) > 3 else ''
                if 'refs/heads/' in ref:
                    push_counts['main'] += 1
                elif 'refs/merge-verify/' in ref:
                    push_counts['merge'] += 1
                return (0, '', '')
            if argv[0] == 'ssh':
                return (0, result_to_json(expected), '')
            return (0, '', '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            main_branch='main',
            run=fake_run,
            id_factory=lambda: 'did',
        )
        return runner, push_counts

    async def test_first_call_fires_main_push(self):
        """First call always fires the main push (no cached sha)."""
        runner, push_counts = self._make_runner_with_tracking(['sha-v1', 'sha-v1'])
        await runner.run_merge_verify('abc123', _make_spec())
        assert push_counts['main'] == 1

    async def test_second_call_same_sha_skips_main_push(self):
        """Second call with unchanged main sha → main push skipped (dedup)."""
        runner, push_counts = self._make_runner_with_tracking(['sha-v1', 'sha-v1'])
        await runner.run_merge_verify('abc123', _make_spec())
        await runner.run_merge_verify('def456', _make_spec())
        assert push_counts['main'] == 1

    async def test_second_call_different_sha_fires_main_push_again(self):
        """When rev-parse returns a new sha, main push fires again."""
        runner, push_counts = self._make_runner_with_tracking(['sha-v1', 'sha-v2'])
        await runner.run_merge_verify('abc123', _make_spec())
        await runner.run_merge_verify('def456', _make_spec())
        assert push_counts['main'] == 2

    async def test_merge_sha_push_always_fires(self):
        """Load-bearing merge-sha push fires on every call regardless."""
        runner, push_counts = self._make_runner_with_tracking(['sha-v1', 'sha-v1'])
        await runner.run_merge_verify('abc123', _make_spec())
        await runner.run_merge_verify('def456', _make_spec())
        assert push_counts['merge'] == 2

    async def test_without_main_branch_no_rev_parse(self):
        """main_branch=None → no rev-parse (existing behaviour unchanged)."""
        rev_parse_called = [0]
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')

        async def fake_run(argv, *, cwd=None):
            if argv[0] == 'git' and argv[1] == 'rev-parse':
                rev_parse_called[0] += 1
            if argv[0] == 'git':
                return (0, '', '')
            return (0, result_to_json(expected), '')

        runner = RemoteRunner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            main_branch=None,
            run=fake_run,
            id_factory=lambda: 'nomain',
        )
        await runner.run_merge_verify('abc123', _make_spec())
        assert rev_parse_called[0] == 0


# ---------------------------------------------------------------------------
# Step-5 (task 1768): LocalRunner archive_root forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLocalRunnerArchiveRoot:
    """LocalRunner forwards archive_root + task_id through run_merge_verify.

    Tests are RED until step-6 adds archive_root to LocalRunner.__init__ and
    wires it through the self._run_scoped(...) call in run_merge_verify.
    """

    def _make_runner_with_spy(
        self,
        *,
        archive_root=None,
        task_id=None,
    ):
        """Build a LocalRunner with a spy run_scoped that records its kwargs."""
        captured: list[dict] = []

        async def spy_run_scoped(*args, **kwargs):
            captured.append(kwargs)
            return _make_pass_result()

        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        run_unscoped = AsyncMock(return_value=MagicMock(broken=False, timed_out=False))

        runner = LocalRunner(
            merge_wt=MagicMock(),
            config=config,
            module_configs=[],
            task_files=None,
            run_scoped=spy_run_scoped,
            run_unscoped=run_unscoped,
            task_id=task_id,
            archive_root=archive_root,
        )
        return runner, captured

    async def test_archive_root_forwarded_to_run_scoped(self, tmp_path):
        """archive_root=tmp_path is forwarded to run_scoped as a kwarg."""
        runner, captured = self._make_runner_with_spy(
            archive_root=tmp_path,
            task_id='1768',
        )
        await runner.run_merge_verify('abc123', _make_spec())

        assert captured, 'run_scoped must have been called'
        kwargs = captured[0]
        assert kwargs.get('archive_root') == tmp_path, (
            f'Expected archive_root={tmp_path!r}, got {kwargs.get("archive_root")!r}'
        )

    async def test_task_id_forwarded_to_run_scoped(self, tmp_path):
        """task_id='1768' is forwarded to run_scoped as a kwarg."""
        runner, captured = self._make_runner_with_spy(
            archive_root=tmp_path,
            task_id='1768',
        )
        await runner.run_merge_verify('abc123', _make_spec())

        assert captured, 'run_scoped must have been called'
        kwargs = captured[0]
        assert kwargs.get('task_id') == '1768', (
            f'Expected task_id="1768", got {kwargs.get("task_id")!r}'
        )

    async def test_role_merge_forwarded_to_run_scoped(self, tmp_path):
        """role='merge' is always forwarded to run_scoped."""
        runner, captured = self._make_runner_with_spy(
            archive_root=tmp_path,
            task_id='1768',
        )
        await runner.run_merge_verify('abc123', _make_spec())

        assert captured, 'run_scoped must have been called'
        kwargs = captured[0]
        assert kwargs.get('role') == 'merge', (
            f'Expected role="merge", got {kwargs.get("role")!r}'
        )

    async def test_default_archive_root_none_forwarded(self):
        """LocalRunner without archive_root (default) forwards archive_root=None."""
        captured: list[dict] = []

        async def spy_run_scoped(*args, **kwargs):
            captured.append(kwargs)
            return _make_pass_result()

        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        run_unscoped = AsyncMock(return_value=MagicMock(broken=False, timed_out=False))

        # Construct without archive_root (uses default)
        runner = LocalRunner(
            merge_wt=MagicMock(),
            config=config,
            module_configs=[],
            task_files=None,
            run_scoped=spy_run_scoped,
            run_unscoped=run_unscoped,
            task_id='1768',
            # archive_root omitted → default None
        )
        await runner.run_merge_verify('abc123', _make_spec())

        assert captured, 'run_scoped must have been called'
        kwargs = captured[0]
        assert kwargs.get('archive_root') is None, (
            f'Expected archive_root=None when not set, got {kwargs.get("archive_root")!r}'
        )


# ---------------------------------------------------------------------------
# task-1920 step-7/9: TestVerifyRunnerPoolArchivalThreading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyRunnerPoolArchivalThreading:
    """VerifyRunnerPool threads archive_root into RemoteRunner.dispatch but not into 2-arg fakes (task 1920)."""

    def _make_remote_runner_with_stderr(self, tmp_path, stderr_text='POOL STDERR'):
        """Build a real RemoteRunner with a fake run that returns a failing result + stderr_text."""
        fail_result = VerifyResult(
            passed=False, test_output='FAILED', lint_output='', type_output='',
            summary='test fail', category='test_failure',
        )
        _it = iter([
            (0, '', ''),                                       # git push
            (0, result_to_json(fail_result), stderr_text),     # ssh
        ])

        async def fake_run(argv, *, cwd=None):
            if argv[0] == 'git' and '--delete' in argv:
                return (0, '', '')
            return next(_it)

        runner = RemoteRunner(
            name='leo-laptop',
            ssh_host='leo-laptop.local',
            git_remote='origin',
            cwd=str(tmp_path),
            run=fake_run,
            id_factory=lambda: 'pool-test-id',
        )
        return runner, fail_result

    async def test_pool_threads_archive_root_to_remote_runner(self, tmp_path):
        """VerifyRunnerPool([remote_runner], archive_root=…) threads archive_root into dispatch."""
        from orchestrator.verify_runner import VerifyRunnerPool

        remote_runner, fail_result = self._make_remote_runner_with_stderr(tmp_path)

        pool = VerifyRunnerPool(
            [remote_runner],
            task_id='1920',
            archive_root=tmp_path,
        )

        result = await pool.dispatch('abc123', _make_spec())

        # Result must be returned UNCHANGED
        assert result == fail_result

        # Exactly one .stderr.log file under tmp_path / '1920'
        task_dir = tmp_path / '1920'
        assert task_dir.is_dir(), f'Expected {task_dir} to exist'
        files = list(task_dir.glob('attempt-1.remote-*-*.stderr.log'))
        assert len(files) == 1, f'Expected 1 stderr log file, got {[f.name for f in files]}'
        assert files[0].read_text(encoding='utf-8') == 'POOL STDERR'

    async def test_local_only_pool_writes_no_remote_stderr_log(self, tmp_path):
        """Local-only pool: 2-arg fake is NOT passed archive_root; no remote stderr file written."""
        from orchestrator.verify_runner import VerifyRunnerPool

        fail_result = VerifyResult(
            passed=False, test_output='FAILED', lint_output='', type_output='',
            summary='test fail', category='test_failure',
        )

        # 2-arg fake (matches existing LocalRunner / test-double signature)
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True

        async def local_run_merge_verify(sha, spec):
            return fail_result

        local_fake.run_merge_verify = AsyncMock(side_effect=local_run_merge_verify)

        pool = VerifyRunnerPool(
            [local_fake],
            task_id='1920',
            archive_root=tmp_path,
        )

        result = await pool.dispatch('abc123', _make_spec())

        assert result == fail_result
        # No remote stderr log files anywhere under tmp_path
        remote_logs = list(tmp_path.rglob('*.remote-*.stderr.log'))
        assert remote_logs == [], f'Unexpected remote logs: {remote_logs}'


# ---------------------------------------------------------------------------
# task-1920 step-1/3/5: TestRemoteRunnerStderrArchival
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerStderrArchival:
    """Failed remote-verify stderr is archived when task_id + archive_root are provided (task 1920)."""

    def _make_fail_result(self):
        return VerifyResult(
            passed=False,
            test_output='FAILED',
            lint_output='',
            type_output='',
            summary='test fail',
            category='test_failure',
        )

    def _make_runner(self, ssh_stderr, *, name='leo-laptop', fail_result=None):
        """Build a RemoteRunner whose fake run returns git→(0,'','') and ssh→(0,json,ssh_stderr)."""
        if fail_result is None:
            fail_result = self._make_fail_result()

        _result_json = result_to_json(fail_result)
        _it = iter([
            (0, '', ''),            # git push (load-bearing)
            (0, _result_json, ssh_stderr),  # ssh verify
        ])

        async def fake_run(argv, *, cwd=None):
            # ref cleanup push (best-effort, inside contextlib.suppress)
            if argv[0] == 'git' and '--delete' in argv:
                return (0, '', '')
            return next(_it)

        runner = RemoteRunner(
            name=name,
            ssh_host=f'{name}.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'test-id',
        )
        return runner, fail_result

    async def test_failure_with_stderr_writes_one_file(self, tmp_path):
        """Failed result + non-empty ssh_stderr → one .stderr.log under <archive_root>/<task_id>.

        task-1921 note: the task_id directory now also contains stream files
        (.test-*.log, summary-*.json) written by _archive_failure_streams.
        This test verifies specifically that the .stderr.log is written correctly
        (not the total file count, which changed in task-1921).
        """
        runner, fail_result = self._make_runner('REMOTE STDERR DETAIL')

        result = await runner.run_merge_verify(
            'abc123', _make_spec(), task_id='1920', archive_root=tmp_path,
        )

        # VerifyResult must be returned UNCHANGED (PRD §A Invariant 5)
        assert result == fail_result

        # task_id directory must exist
        task_dir = tmp_path / '1920'
        assert task_dir.is_dir(), f'Expected {task_dir} to be a directory'

        # Exactly one .stderr.log file with the correct filename shape and content
        stderr_files = list(task_dir.glob('attempt-1.remote-leo-laptop-*.stderr.log'))
        assert len(stderr_files) == 1, f'Expected 1 stderr.log, got {[f.name for f in stderr_files]}'
        fname = stderr_files[0].name
        assert fname.startswith('attempt-1.remote-leo-laptop-'), f'Bad prefix: {fname!r}'
        assert fname.endswith('.stderr.log'), f'Bad suffix: {fname!r}'
        assert stderr_files[0].read_text(encoding='utf-8') == 'REMOTE STDERR DETAIL'

    # step-3 negative cases
    async def test_passing_result_writes_no_file(self, tmp_path):
        """Passing remote verify: ssh_stderr noise is NOT archived (failure-only contract)."""
        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        _it = iter([
            (0, '', ''),
            (0, result_to_json(pass_result), 'some noise'),
        ])

        async def fake_run(argv, *, cwd=None):
            if argv[0] == 'git' and '--delete' in argv:
                return (0, '', '')
            return next(_it)

        runner = RemoteRunner(
            name='leo-laptop',
            ssh_host='leo-laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'test-id',
        )

        await runner.run_merge_verify('abc123', _make_spec(), task_id='1920', archive_root=tmp_path)

        # No file should have been written
        task_dir = tmp_path / '1920'
        assert not task_dir.exists(), f'Expected no archive dir, found {list(task_dir.iterdir()) if task_dir.exists() else "nothing"}'

    async def test_whitespace_only_stderr_writes_no_file(self, tmp_path):
        """Failed verify with whitespace-only ssh_stderr → NO .stderr.log written (strip guard).

        task-1921 note: stream files (.test-*.log, summary-*.json) may still be written
        by _archive_failure_streams when the VerifyResult has non-empty output streams.
        This test verifies specifically that the .stderr.log is NOT written
        (the ssh_stderr strip guard is the contract being tested here).
        """
        runner, _ = self._make_runner('   \n  ')

        await runner.run_merge_verify('abc123', _make_spec(), task_id='1920', archive_root=tmp_path)

        task_dir = tmp_path / '1920'
        # No .stderr.log files (the strip guard must have suppressed stderr archival)
        stderr_files = list(task_dir.glob('*.stderr.log')) if task_dir.exists() else []
        assert stderr_files == [], f'Expected no .stderr.log files, got {stderr_files}'

    # step-5 archival-error-is-swallowed
    async def test_archival_error_is_swallowed_result_unchanged(self, tmp_path):
        """Unwritable archive_root → archival error swallowed; VerifyResult returned unchanged."""
        # Create a regular FILE at the path where the directory would go
        blocker = tmp_path / 'blocker'
        blocker.write_text('x')

        runner, fail_result = self._make_runner('REMOTE STDERR DETAIL')

        # Must NOT raise even though mkdir will fail (NotADirectoryError)
        result = await runner.run_merge_verify(
            'abc123', _make_spec(), task_id='1920', archive_root=blocker,
        )

        # VerifyResult is returned unchanged
        assert result == fail_result


# ---------------------------------------------------------------------------
# task-1921: TestRemoteRunnerStreamArchival
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerStreamArchival:
    """RemoteRunner archives test/lint/type output streams on failure (task 1921).

    Mirrors _archive_merge_verify_logs via a synthetic runs projection.
    Filename pattern: attempt-1.remote-<name>.{label}-<utc_ts>.log
                      attempt-1.remote-<name>.summary-<utc_ts>.json
    """

    def _make_runner(self, fail_result, *, name='leo-laptop', ssh_stderr=''):
        """Build a RemoteRunner whose fake run returns git→(0,'','') and ssh→(0,json,ssh_stderr)."""
        _result_json = result_to_json(fail_result)
        _it = iter([
            (0, '', ''),                             # git push (load-bearing)
            (0, _result_json, ssh_stderr),           # ssh verify
        ])

        async def fake_run(argv, *, cwd=None):
            # ref cleanup push (best-effort, inside contextlib.suppress)
            if argv[0] == 'git' and '--delete' in argv:
                return (0, '', '')
            return next(_it)

        runner = RemoteRunner(
            name=name,
            ssh_host=f'{name}.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'test-id',
        )
        return runner, fail_result

    # (a) single non-empty stream: test_output only
    async def test_single_stream_test_output_writes_one_log_file(self, tmp_path):
        """Failure + non-empty test_output → one .test-*.log file; no lint/type files."""
        fail_result = VerifyResult(
            passed=False, test_output='FAILED TEST OUTPUT', lint_output='', type_output='',
            summary='test fail', category='test_failure',
        )
        runner, _ = self._make_runner(fail_result)

        result = await runner.run_merge_verify(
            'abc123', _make_spec(), task_id='1921', archive_root=tmp_path,
        )

        # VerifyResult must be returned UNCHANGED (PRD §A Invariant 5)
        assert result == fail_result

        task_dir = tmp_path / '1921'
        assert task_dir.is_dir(), f'Expected {task_dir} to exist'

        # Exactly one .test-*.log file
        test_files = list(task_dir.glob('attempt-1.remote-leo-laptop.test-*.log'))
        assert len(test_files) == 1, f'Expected 1 test log, got {[f.name for f in test_files]}'
        assert test_files[0].read_text(encoding='utf-8') == 'FAILED TEST OUTPUT'

        # No lint or type log files
        lint_files = list(task_dir.glob('attempt-1.remote-*.lint-*.log'))
        type_files = list(task_dir.glob('attempt-1.remote-*.type-*.log'))
        assert lint_files == [], f'Unexpected lint files: {lint_files}'
        assert type_files == [], f'Unexpected type files: {type_files}'

    # (a2) single non-empty stream: lint_output only — closes the label-mapping matrix
    async def test_single_stream_lint_output_writes_one_log_file(self, tmp_path):
        """Failure + non-empty lint_output (test/type empty) → one .lint-*.log; no test/type files.

        Guards the per-stream loop against a label-mapping regression on the lint branch.
        """
        fail_result = VerifyResult(
            passed=False, test_output='', lint_output='LINT ERROR: unused import', type_output='',
            summary='lint fail', category='lint_failure',
        )
        runner, _ = self._make_runner(fail_result)

        result = await runner.run_merge_verify(
            'abc123', _make_spec(), task_id='1921', archive_root=tmp_path,
        )

        # VerifyResult must be returned UNCHANGED (PRD §A Invariant 5)
        assert result == fail_result

        task_dir = tmp_path / '1921'
        assert task_dir.is_dir(), f'Expected {task_dir} to exist'

        # Exactly one .lint-*.log file
        lint_files = list(task_dir.glob('attempt-1.remote-leo-laptop.lint-*.log'))
        assert len(lint_files) == 1, f'Expected 1 lint log, got {[f.name for f in lint_files]}'
        assert lint_files[0].read_text(encoding='utf-8') == 'LINT ERROR: unused import'

        # No test or type log files
        test_files = list(task_dir.glob('attempt-1.remote-*.test-*.log'))
        type_files = list(task_dir.glob('attempt-1.remote-*.type-*.log'))
        assert test_files == [], f'Unexpected test files: {test_files}'
        assert type_files == [], f'Unexpected type files: {type_files}'

    # (b) multi-stream: test_output + type_output + summary sidecar
    async def test_multi_stream_test_and_type_output(self, tmp_path):
        """Failure + non-empty test_output and type_output → two .log files + one summary.json."""
        fail_result = VerifyResult(
            passed=False, test_output='FAILED TESTS', lint_output='', type_output='TYPE ERRORS',
            summary='multiple failures', category='test_failure', cause_hint='assertion error',
        )
        runner, _ = self._make_runner(fail_result)

        result = await runner.run_merge_verify(
            'abc123', _make_spec(), task_id='1921', archive_root=tmp_path,
        )

        assert result == fail_result

        task_dir = tmp_path / '1921'
        assert task_dir.is_dir()

        # Two stream files
        test_files = list(task_dir.glob('attempt-1.remote-leo-laptop.test-*.log'))
        type_files = list(task_dir.glob('attempt-1.remote-leo-laptop.type-*.log'))
        assert len(test_files) == 1, f'Expected 1 test log, got {test_files}'
        assert len(type_files) == 1, f'Expected 1 type log, got {type_files}'
        assert test_files[0].read_text(encoding='utf-8') == 'FAILED TESTS'
        assert type_files[0].read_text(encoding='utf-8') == 'TYPE ERRORS'

        # Exactly one summary.json
        summary_files = list(task_dir.glob('attempt-1.remote-leo-laptop.summary-*.json'))
        assert len(summary_files) == 1, f'Expected 1 summary, got {summary_files}'
        import json as _json
        summary = _json.loads(summary_files[0].read_text(encoding='utf-8'))
        assert 'category' in summary, 'Summary missing category'
        assert 'timed_out' in summary, 'Summary missing timed_out'
        assert 'commands' in summary, 'Summary missing commands'

    # (c) passing result → no archive dir
    async def test_passing_result_writes_no_files(self, tmp_path):
        """Passing remote verify (passed=True) → no archive dir written."""
        pass_result = VerifyResult(
            passed=True, test_output='all tests pass', lint_output='clean', type_output='no errors',
            summary='ok', category='',
        )
        _result_json = result_to_json(pass_result)
        _it = iter([
            (0, '', ''),
            (0, _result_json, ''),
        ])

        async def fake_run(argv, *, cwd=None):
            if argv[0] == 'git' and '--delete' in argv:
                return (0, '', '')
            return next(_it)

        runner = RemoteRunner(
            name='leo-laptop', ssh_host='leo-laptop.local', git_remote='origin',
            cwd='/repo', run=fake_run, id_factory=lambda: 'test-id',
        )

        await runner.run_merge_verify('abc123', _make_spec(), task_id='1921', archive_root=tmp_path)

        task_dir = tmp_path / '1921'
        assert not task_dir.exists(), f'Expected no archive dir for passing result, got {task_dir}'

    # (d) failure with all-empty streams → no file and no summary
    async def test_all_empty_streams_failure_writes_nothing(self, tmp_path):
        """Failure with all-empty streams → early-return; no stream file and no summary."""
        fail_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='failed but no output', category='test_failure',
        )
        runner, _ = self._make_runner(fail_result)

        await runner.run_merge_verify('abc123', _make_spec(), task_id='1921', archive_root=tmp_path)

        task_dir = tmp_path / '1921'
        # With all-empty streams AND empty ssh_stderr the dir must never be created.
        # Asserting directly (not under `if exists`) ensures the check runs unconditionally:
        # removing the early-return would create the dir + summary.json, making this fail.
        assert not task_dir.exists(), (
            f'Expected no archive dir for all-empty streams; found: '
            f'{sorted(p.name for p in task_dir.iterdir()) if task_dir.exists() else []}'
        )

    # (e) distinguishability: timeout vs real failure summaries differ
    async def test_distinguishability_timeout_vs_test_failure(self, tmp_path):
        """timed_out=True/infra_timeout and timed_out=False/test_failure produce different summaries."""
        import json as _json

        # timeout result
        timeout_result = VerifyResult(
            passed=False, test_output='TIMED OUT', lint_output='', type_output='',
            summary='timed out', category='infra_timeout', timed_out=True,
        )
        runner_a, _ = self._make_runner(timeout_result, name='leo-laptop')
        await runner_a.run_merge_verify(
            'abc123', _make_spec(), task_id='timeout-task', archive_root=tmp_path,
        )

        # real test failure result (separate task_id to avoid filename collision)
        fail_result = VerifyResult(
            passed=False, test_output='ASSERTION FAILED', lint_output='', type_output='',
            summary='test fail', category='test_failure', timed_out=False,
        )
        runner_b, _ = self._make_runner(fail_result, name='leo-laptop')
        await runner_b.run_merge_verify(
            'abc123', _make_spec(), task_id='fail-task', archive_root=tmp_path,
        )

        timeout_dir = tmp_path / 'timeout-task'
        fail_dir = tmp_path / 'fail-task'

        timeout_summary_files = list(timeout_dir.glob('attempt-1.remote-*.summary-*.json'))
        fail_summary_files = list(fail_dir.glob('attempt-1.remote-*.summary-*.json'))

        assert len(timeout_summary_files) == 1, f'Expected 1 timeout summary, got {timeout_summary_files}'
        assert len(fail_summary_files) == 1, f'Expected 1 fail summary, got {fail_summary_files}'

        timeout_summary = _json.loads(timeout_summary_files[0].read_text(encoding='utf-8'))
        fail_summary = _json.loads(fail_summary_files[0].read_text(encoding='utf-8'))

        # timed_out must differ
        assert timeout_summary['timed_out'] is True, f'Expected timed_out=True, got {timeout_summary["timed_out"]}'
        assert fail_summary['timed_out'] is False, f'Expected timed_out=False, got {fail_summary["timed_out"]}'

        # category must differ
        assert timeout_summary['category'] == 'infra_timeout', f'Unexpected category: {timeout_summary["category"]}'
        assert fail_summary['category'] == 'test_failure', f'Unexpected category: {fail_summary["category"]}'

    # (f) unwritable archive_root → archival swallowed; result unchanged
    async def test_unwritable_archive_root_swallowed_result_unchanged(self, tmp_path):
        """Unwritable archive_root (file where dir should go) → swallowed; result unchanged."""
        fail_result = VerifyResult(
            passed=False, test_output='FAILED', lint_output='', type_output='',
            summary='test fail', category='test_failure',
        )
        runner, _ = self._make_runner(fail_result)

        # Place a regular FILE at the path where the task_id directory would be created
        blocker = tmp_path / 'blocker'
        blocker.write_text('x')

        # Must NOT raise; must return fail_result unchanged
        result = await runner.run_merge_verify(
            'abc123', _make_spec(), task_id='blocker', archive_root=blocker,
        )

        assert result == fail_result

    # (g) local-only pool regression: no remote stream files written
    async def test_local_only_pool_writes_no_remote_stream_logs(self, tmp_path):
        """Local-only pool (2-arg fake, not RemoteRunner): no remote stream files written."""
        from orchestrator.verify_runner import VerifyRunnerPool

        fail_result = VerifyResult(
            passed=False, test_output='FAILED', lint_output='', type_output='',
            summary='test fail', category='test_failure',
        )

        # 2-arg fake (matches existing LocalRunner / test-double signature)
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True

        async def local_run_merge_verify(sha, spec):
            return fail_result

        local_fake.run_merge_verify = AsyncMock(side_effect=local_run_merge_verify)

        pool = VerifyRunnerPool(
            [local_fake],
            task_id='1921',
            archive_root=tmp_path,
        )

        result = await pool.dispatch('abc123', _make_spec())

        assert result == fail_result

        # No remote stream log files anywhere under tmp_path
        remote_stream_logs = list(tmp_path.rglob('attempt-1.remote-*.test-*.log'))
        remote_stream_logs += list(tmp_path.rglob('attempt-1.remote-*.lint-*.log'))
        remote_stream_logs += list(tmp_path.rglob('attempt-1.remote-*.type-*.log'))
        assert remote_stream_logs == [], f'Unexpected remote stream logs: {remote_stream_logs}'

    # step-3: non-OSError archival exception is swallowed; result unchanged
    async def test_stream_archival_exception_is_swallowed_result_unchanged(self, tmp_path, monkeypatch):
        """Non-OSError in _archive_failure_streams → swallowed; VerifyResult returned unchanged.

        Monkeypatches orchestrator.verify_runner._archive_merge_verify_logs to raise
        RuntimeError('boom') — a non-OSError that _archive_merge_verify_logs's internal
        OSError handler does not catch.  Verifies that run_merge_verify:
          (1) does NOT raise;
          (2) returns the VerifyResult unchanged.

        This is RED before step-4 because step-2's _archive_failure_streams has no outer
        exception guard, so the RuntimeError propagates out of run_merge_verify.
        """
        import orchestrator.verify_runner as _vrmod

        def boom(*args, **kwargs):
            raise RuntimeError('boom')

        monkeypatch.setattr(_vrmod, '_archive_merge_verify_logs', boom)

        fail_result = VerifyResult(
            passed=False, test_output='FAILED', lint_output='', type_output='',
            summary='test fail', category='test_failure',
        )
        runner, _ = self._make_runner(fail_result)

        # Must NOT raise; must return fail_result unchanged
        result = await runner.run_merge_verify(
            'abc123', _make_spec(), task_id='1921', archive_root=tmp_path,
        )

        assert result == fail_result
