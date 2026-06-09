"""Tests for orchestrator/verify_runner.py — MergeVerifySpec + VerifyResult JSON codec."""

import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    LocalRunner,
    MergeVerifySpec,
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
    config = MagicMock()
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

    def test_local_runner_has_run_merge_verify_coroutine(self):
        import asyncio
        runner = _make_local_runner()
        assert hasattr(runner, 'run_merge_verify')
        assert asyncio.iscoroutinefunction(runner.run_merge_verify)

    def test_local_runner_has_health_coroutine(self):
        import asyncio
        runner = _make_local_runner()
        assert asyncio.iscoroutinefunction(runner.health)


# ---------------------------------------------------------------------------
# Step-3: LocalRunner.run_merge_verify combined-bundle behaviour
# ---------------------------------------------------------------------------


def _make_pass_result(**kwargs):
    defaults = dict(passed=True, test_output='', lint_output='', type_output='', summary='ok')
    defaults.update(kwargs)
    return VerifyResult(**defaults)


def _make_fail_result(**kwargs):
    defaults = dict(passed=False, test_output='FAILED', lint_output='', type_output='', summary='test fail')
    defaults.update(kwargs)
    return VerifyResult(**defaults)


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
        config = MagicMock()
        config.merge_verify_workspace = True
        merge_wt = MagicMock()
        module_configs = [MagicMock()]
        task_files = ('src/a.py',)
        runner = LocalRunner(
            merge_wt=merge_wt,
            config=config,
            module_configs=module_configs,
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
        config = MagicMock()
        config.verify_env = verify_env or {}
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
        config = MagicMock()
        config.verify_env = {}
        config.merge_verify_cold_command_timeout_secs = None
        config.verify_cold_command_timeout_secs = 3600.0
        spec = build_merge_verify_spec(config, [], None)
        assert spec.cold_timeout_secs == 3600.0

    def test_cold_timeout_falls_back_to_zero_when_both_none(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        config = MagicMock()
        config.verify_env = {}
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
