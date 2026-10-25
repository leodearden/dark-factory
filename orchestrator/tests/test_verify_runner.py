"""Tests for orchestrator/verify_runner.py — MergeVerifySpec + VerifyResult JSON codec."""

import asyncio
import dataclasses
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _gate_fixtures import gate_argv, gated_remote_runner
from _orch_helpers import pydantic_spec, wait_responsive

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    HEARTBEAT_STOP_JOIN_SECS,
    LocalRunner,
    MergeVerifySpec,
    UnscopedTypecheckSpec,
    VerifyCommand,
    VerifyRunner,
    VerifyRunnerPool,
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

    # --- Fix (a): the merge-gate PROFILE now rides the spec (task 2822) ------

    def test_roundtrip_profile_fields(self):
        """merge_verify_workspace + merge_verify_breadth survive to_dict->from_dict."""
        spec = MergeVerifySpec(
            verify_commands=(VerifyCommand(prefix="src/a"),),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=("src/a/mod.py",),
            verify_env={},
            cold_timeout_secs=300.0,
            is_merge_verify=True,
            merge_verify_workspace=True,
            merge_verify_breadth="full",
        )
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert restored.merge_verify_workspace is True
        assert restored.merge_verify_breadth == "full"
        assert restored == spec

    def test_profile_fields_default_narrow(self):
        """A spec built without the profile fields defaults to the NARROW gate."""
        spec = MergeVerifySpec(
            verify_commands=(VerifyCommand(prefix="src/a"),),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=None,
            verify_env={},
            cold_timeout_secs=60.0,
        )
        assert spec.merge_verify_workspace is False
        assert spec.merge_verify_breadth == "scoped"

    def test_from_dict_back_compat_missing_profile_keys(self):
        """BACK-COMPAT: a legacy dict WITHOUT the profile keys deserialises to the
        narrow defaults (mirrors the verify_env / is_merge_verify d.get idiom)."""
        legacy = self._make_spec().to_dict()
        legacy.pop("merge_verify_workspace", None)
        legacy.pop("merge_verify_breadth", None)
        restored = MergeVerifySpec.from_dict(legacy)
        assert restored.merge_verify_workspace is False
        assert restored.merge_verify_breadth == "scoped"

    # --- INV-1, task 2883: global_verify_command back-compat -----------------

    def test_from_dict_back_compat_missing_global_verify_command(self):
        """BACK-COMPAT (d): a legacy dict WITHOUT the 'global_verify_command'
        key deserialises to None (mirrors the profile-keys d.get idiom)."""
        legacy = self._make_spec().to_dict()
        legacy.pop("global_verify_command", None)
        restored = MergeVerifySpec.from_dict(legacy)
        assert restored.global_verify_command is None

    def test_global_verify_command_round_trips_to_dict(self):
        """A directly-set global_verify_command survives to_dict -> from_dict."""
        spec = MergeVerifySpec(
            verify_commands=(),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=("docs/x.md",),
            verify_env={},
            cold_timeout_secs=300.0,
            global_verify_command=VerifyCommand(
                prefix="",
                test_command="cargo test --workspace",
                lint_command="cargo clippy --workspace",
                type_check_command="pyright",
            ),
        )
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert restored.global_verify_command == spec.global_verify_command
        assert restored == spec

    # --- task 6127: the cold-worktree pre-provision command rides the spec ---

    def test_verify_cold_preprovision_command_round_trips(self):
        provision = 'uv sync --all-packages && npm ci --no-audit --no-fund --prefer-offline'
        spec = dataclasses.replace(
            self._make_spec(), verify_cold_preprovision_command=provision,
        )
        restored = MergeVerifySpec.from_dict(spec.to_dict())
        assert restored == spec
        assert restored.verify_cold_preprovision_command == provision
        from_wire = spec_from_json(spec_to_json(spec))
        assert from_wire == spec
        assert from_wire.verify_cold_preprovision_command == provision

    def test_verify_cold_preprovision_command_defaults_empty(self):
        spec = MergeVerifySpec(
            verify_commands=(VerifyCommand(prefix="src/a"),),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=None,
            verify_env={},
            cold_timeout_secs=60.0,
        )
        assert spec.verify_cold_preprovision_command == ''

    def test_from_dict_back_compat_missing_verify_cold_preprovision_command(self):
        """BACK-COMPAT: a pre-6127 dispatcher's dict lacks the key and decodes
        to '', which leaves the consuming host's own value in force."""
        legacy = self._make_spec().to_dict()
        legacy.pop("verify_cold_preprovision_command", None)
        restored = MergeVerifySpec.from_dict(legacy)
        assert restored.verify_cold_preprovision_command == ''


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
# Task 2306 step-1: VerifyResult.contention — machine-readable flock-contention
# payload (JSON-native dict, not a nested dataclass) that round-trips losslessly.
# ---------------------------------------------------------------------------


class TestVerifyResultContention:
    """VerifyResult carries an optional `contention` dict that survives the wire codec."""

    def test_contention_round_trips_via_json_codec(self):
        vr = VerifyResult(
            passed=False,
            test_output="",
            lint_output="",
            type_output="",
            summary="flock contention",
            category="flock_contention",
            contention={"host": "leo-laptop", "holder_pgid": 4242, "waiter_pgid": 4343},
        )
        restored = result_from_json(result_to_json(vr))
        assert restored.contention == {"host": "leo-laptop", "holder_pgid": 4242, "waiter_pgid": 4343}
        assert restored.category == "flock_contention"

    def test_default_contention_is_none(self):
        vr = VerifyResult(
            passed=True,
            test_output="",
            lint_output="",
            type_output="",
            summary="all good",
        )
        assert vr.contention is None


# ---------------------------------------------------------------------------
# Task 2126 step-13: VerifyResult.plan — machine-readable VerifyPlan.to_dict()
# payload (JSON-native dict, not a nested dataclass) that round-trips losslessly.
# Mirrors TestVerifyResultContention above (task 2306 α precedent).
# ---------------------------------------------------------------------------


_SAMPLE_PLAN_DICT = {
    "runs": [
        {
            "module_prefix": "orchestrator",
            "cmd": None,
            "scope_kind": "skipped",
            "reason": "no files under prefix",
        },
    ],
    "needs_pipeline_guard_check": False,
}


class TestVerifyResultPlan:
    """VerifyResult carries an optional `plan` dict that survives the wire codec."""

    def test_plan_round_trips_via_json_codec(self):
        vr = VerifyResult(
            passed=True,
            test_output="",
            lint_output="",
            type_output="",
            summary="all good",
            plan=_SAMPLE_PLAN_DICT,
        )
        restored = result_from_json(result_to_json(vr))
        assert restored.plan == _SAMPLE_PLAN_DICT

    def test_plan_round_trips_via_dict_codec(self):
        vr = VerifyResult(
            passed=True,
            test_output="",
            lint_output="",
            type_output="",
            summary="all good",
            plan=_SAMPLE_PLAN_DICT,
        )
        restored = result_from_dict(result_to_dict(vr))
        assert restored.plan == _SAMPLE_PLAN_DICT
        assert restored == vr

    def test_plan_dict_is_json_dumpable(self):
        vr = VerifyResult(
            passed=True,
            test_output="",
            lint_output="",
            type_output="",
            summary="all good",
            plan={"runs": [], "needs_pipeline_guard_check": True},
        )
        # Should round-trip through json without error — plan must be a plain
        # JSON-native dict, not a nested dataclass (mirrors `contention`).
        json.dumps(result_to_dict(vr))

    def test_default_plan_is_none(self):
        vr = VerifyResult(
            passed=True,
            test_output="",
            lint_output="",
            type_output="",
            summary="all good",
        )
        assert vr.plan is None


# ---------------------------------------------------------------------------
# Task 3789 (ε) step-3: VerifyResult.flake_suppression — the TYPED wire carrier
# that moves a discriminator observation from wherever the worktree is to the
# dispatcher that records it (flake-ledger-prd.md §8, §8.4).
# ---------------------------------------------------------------------------


def _make_suppression(**overrides):
    """A fully-populated FlakeSuppression; override any field by keyword."""
    from orchestrator.flake_ledger import FlakeCallSite, FlakeSuppression, FlakeVerdict

    kwargs = {
        'verdict': FlakeVerdict.passes_in_isolation,
        'test_ids': ('tests/test_a.py::test_one', 'tests/test_b.py::test_two'),
        'observed_at': '2026-08-06T12:00:00+00:00',
        'call_site': FlakeCallSite.merge_gate,
        'runner': 'remote-lab-1',
        'psi_cpu_some10': 12.5,
        'unconfirmable_reason': None,
    }
    kwargs.update(overrides)
    return FlakeSuppression(**kwargs)


class TestVerifyResultFlakeSuppressionWire:
    """VerifyResult carries an optional TYPED `flake_suppression` across the wire.

    Deliberately NOT a plain JSON-native dict like `contention`/`plan`/
    `failing_test_ids`: PRD §8 specifies a typed carrier and the dispatcher-side
    recorder consumes a `FlakeSuppression`, so a bare dict would make the annotation
    a lie on exactly the deserialized path this exists to serve.  `result_to_dict`
    (asdict) already serialises it losslessly; only `result_from_dict` needs the
    reconstruction hook — the same shape `MergeVerifySpec.from_dict` uses for its
    optional nested `global_verify_command`.
    """

    def test_defaults_to_none(self):
        """(a) Every existing hand-built VerifyResult is unaffected."""
        vr = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok'
        )
        assert vr.flake_suppression is None

    def test_round_trips_through_json_as_a_typed_value(self):
        """(b) The headline: what comes back off the wire is a FlakeSuppression with
        its enums and tuple intact, not a bare dict of strings and lists.

        Asserted on the FIELD, never via VerifyResult.__eq__ — the field is
        `compare=False` (see test_differs_only_in_suppression_still_compares_equal),
        so a whole-object assertion would pass VACUOUSLY here.
        """
        from orchestrator.flake_ledger import FlakeCallSite, FlakeSuppression, FlakeVerdict

        s = _make_suppression()
        vr = VerifyResult(
            passed=False,
            test_output='FAILED test_a',
            lint_output='',
            type_output='',
            summary='1 failure',
            flake_suppression=s,
        )
        rt = result_from_json(result_to_json(vr))

        assert isinstance(rt.flake_suppression, FlakeSuppression)
        assert rt.flake_suppression == s
        assert rt.flake_suppression.verdict is FlakeVerdict.passes_in_isolation
        assert rt.flake_suppression.call_site is FlakeCallSite.merge_gate
        assert isinstance(rt.flake_suppression.test_ids, tuple)
        assert rt.flake_suppression.test_ids == (
            'tests/test_a.py::test_one',
            'tests/test_b.py::test_two',
        )
        assert rt.flake_suppression.runner == 'remote-lab-1'
        assert rt.flake_suppression.psi_cpu_some10 == 12.5
        assert rt.flake_suppression.observed_at == '2026-08-06T12:00:00+00:00'

    def test_round_trips_an_unconfirmable_observation(self):
        """(c) §5.5 records the OBSERVATION, not the remedy — an `unconfirmable` rides
        the wire too, and its reason and NULL psi must survive (None means "PSI was
        unreadable", never "the host was idle")."""
        from orchestrator.flake_ledger import FlakeVerdict

        s = _make_suppression(
            verdict=FlakeVerdict.unconfirmable,
            test_ids=(),
            psi_cpu_some10=None,
            unconfirmable_reason='node-ids mapped to no discovered subproject',
        )
        vr = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='',
            summary='red',
            flake_suppression=s,
        )
        rt = result_from_json(result_to_json(vr))

        assert rt.flake_suppression is not None
        assert rt.flake_suppression == s
        assert rt.flake_suppression.verdict is FlakeVerdict.unconfirmable
        assert rt.flake_suppression.test_ids == ()
        assert rt.flake_suppression.psi_cpu_some10 is None
        assert (
            rt.flake_suppression.unconfirmable_reason
            == 'node-ids mapped to no discovered subproject'
        )

    def test_to_dict_needs_no_allowlist_change(self):
        """asdict flattens the nested dataclass and json.dumps flattens its StrEnums —
        the WRITE half of the codec is untouched by this field."""
        vr = VerifyResult(
            passed=True,
            test_output='',
            lint_output='',
            type_output='',
            summary='ok',
            flake_suppression=_make_suppression(),
        )
        d = result_to_dict(vr)
        assert isinstance(d['flake_suppression'], dict)
        payload = json.loads(json.dumps(d, sort_keys=True))
        assert payload['flake_suppression']['verdict'] == 'passes_in_isolation'
        assert payload['flake_suppression']['test_ids'] == [
            'tests/test_a.py::test_one',
            'tests/test_b.py::test_two',
        ]

    # -- (d) B13: NEW dispatcher, OLD remote -------------------------------------

    def test_b13_omitted_key_yields_none(self):
        """B13 — an older remote's payload simply has no such key.  The default
        applies: no suppression, no ledger row, no crash."""
        vr = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='', summary='red'
        )
        d = result_to_dict(vr)
        d.pop('flake_suppression')
        assert result_from_dict(d).flake_suppression is None

    def test_b13_explicit_null_yields_none(self):
        payload = json.dumps(
            {
                'passed': False,
                'test_output': '',
                'lint_output': '',
                'type_output': '',
                'summary': 'red',
                'flake_suppression': None,
            }
        )
        assert result_from_json(payload).flake_suppression is None

    @pytest.mark.parametrize(
        'malformed',
        [
            'passes_in_isolation',
            {'verdict': 'passes_in_isolation'},
            [],
            17,
        ],
        ids=['str', 'missing_keys', 'list', 'int'],
    )
    def test_b13_malformed_payload_degrades_to_none_without_raising(self, malformed):
        """A malformed sub-payload must cost ONE observation, not a whole re-verify:
        anything raising out of result_from_json becomes a RunnerUnavailable in
        orchestrator/src/orchestrator/verify_runner.py::RemoteRunner.run_merge_verify,
        which the pool pays for with a local re-run."""
        payload = json.dumps(
            {
                'passed': False,
                'test_output': '',
                'lint_output': '',
                'type_output': '',
                'summary': 'red',
                'flake_suppression': malformed,
            }
        )
        rt = result_from_json(payload)
        assert rt.flake_suppression is None
        assert rt.passed is False
        assert rt.summary == 'red'

    def test_from_dict_does_not_mutate_the_callers_dict(self):
        """The hook rebuilds into a shallow COPY — a caller inspecting the payload it
        just handed over must not find it silently retyped underneath."""
        vr = VerifyResult(
            passed=True,
            test_output='',
            lint_output='',
            type_output='',
            summary='ok',
            flake_suppression=_make_suppression(),
        )
        d = json.loads(result_to_json(vr))
        before = json.dumps(d, sort_keys=True)
        result_from_dict(d)
        assert json.dumps(d, sort_keys=True) == before

    # -- (e) compare=False -------------------------------------------------------

    def test_differs_only_in_suppression_still_compares_equal(self):
        """`observed_at` is a wall-clock stamp taken by the discriminator, so two
        independent runs of the same logical verification carry different values —
        the identical argument already written out for `duration_secs`.  Without
        compare=False, attaching the field on the NON-suppressed branch too (which
        §5.5 requires) would break test_cli's
        test_verify_merge_cli_wrapper_transparency the moment a failing CLI verify
        produced an `unconfirmable` observation."""
        from orchestrator.flake_ledger import FlakeVerdict

        base: dict[str, Any] = dict(
            passed=False, test_output='', lint_output='', type_output='', summary='red'
        )
        a = VerifyResult(**base, flake_suppression=None)
        b = VerifyResult(**base, flake_suppression=_make_suppression())
        c = VerifyResult(
            **base,
            flake_suppression=_make_suppression(
                verdict=FlakeVerdict.fails_in_isolation,
                observed_at='2099-01-01T00:00:00+00:00',
            ),
        )
        assert a == b == c

    def test_field_is_declared_compare_false(self):
        """Structural pin: a later hand-edit that drops compare=False would break the
        CLI transparency invariant in a distant file, so assert the declaration."""
        f = {f.name: f for f in dataclasses.fields(VerifyResult)}['flake_suppression']
        assert f.compare is False
        assert f.default is None

    # -- (f) the OTHER skew direction — characterization only --------------------

    def test_characterization_unknown_top_level_key_raises_typeerror(self):
        """CHARACTERIZATION PIN of PRE-EXISTING behaviour, not a new capability.

        §8.4 says an unknown key "is ignored" by an OLD dispatcher.  It is not:
        `result_from_dict` is a bare `VerifyResult(**d)`, so an unknown top-level key
        is a TypeError.  This property is shared by every optional field added before
        this one (contention, plan, failing_test_ids, failing_leg_categories,
        trivial), so it is neither introduced nor worsened here — and the OUTCOME
        still holds (see the twin below): degrade to a local re-verify, never a wrong
        verdict.  This task deliberately does NOT change the codec's strictness.
        """
        payload = json.dumps(
            {
                'passed': True,
                'test_output': '',
                'lint_output': '',
                'type_output': '',
                'summary': 'ok',
                'a_field_from_a_newer_remote': 1,
            }
        )
        with pytest.raises(TypeError):
            result_from_json(payload)

    @pytest.mark.asyncio
    async def test_characterization_remote_typeerror_becomes_runner_unavailable(self):
        """The twin of the above, at the boundary that matters: RemoteRunner converts
        that TypeError into RunnerUnavailable
        (orchestrator/src/orchestrator/verify_runner.py::RemoteRunner.run_merge_verify),
        which VerifyRunnerPool.dispatch turns into a LOCAL re-verify.  Degraded, never
        wrong — and never an unhandled crash in the merge path."""
        from orchestrator.verify_runner import RunnerUnavailable

        async def fake_run(argv, *, cwd=None):
            if argv[0] == 'git':
                return (0, '', '')
            return (
                0,
                json.dumps(
                    {
                        'passed': True,
                        'test_output': '',
                        'lint_output': '',
                        'type_output': '',
                        'summary': 'ok',
                        'a_field_from_a_newer_remote': 1,
                    }
                ),
                '',
            )

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'fixed-id',
        )
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())


# ---------------------------------------------------------------------------
# Task 2306 step-3: FLOCK_CONTENTION_CATEGORY + make_flock_contention_result —
# the contention-result builder consumed by task beta (workstation side).
# ---------------------------------------------------------------------------


class TestMakeFlockContentionResult:
    """make_flock_contention_result builds a losslessly-parseable contention VerifyResult."""

    def test_category_constant_value(self):
        from orchestrator.verify_runner import FLOCK_CONTENTION_CATEGORY
        assert FLOCK_CONTENTION_CATEGORY == 'flock_contention'

    def test_builder_shape(self):
        from orchestrator.verify_runner import (
            FLOCK_CONTENTION_CATEGORY,
            make_flock_contention_result,
        )
        result = make_flock_contention_result(host='leo-laptop', holder_pgid=4242, waiter_pgid=4343)
        assert result.passed is False
        assert result.timed_out is False
        assert result.category == FLOCK_CONTENTION_CATEGORY
        assert result.contention == {'host': 'leo-laptop', 'holder_pgid': 4242, 'waiter_pgid': 4343}

    def test_builder_accepts_none_holder_pgid(self):
        from orchestrator.verify_runner import make_flock_contention_result
        result = make_flock_contention_result(host='leo-laptop', holder_pgid=None, waiter_pgid=4343)
        assert result.contention == {'host': 'leo-laptop', 'holder_pgid': None, 'waiter_pgid': 4343}

    def test_builder_result_round_trips(self):
        from orchestrator.verify_runner import make_flock_contention_result
        result = make_flock_contention_result(host='leo-laptop', holder_pgid=4242, waiter_pgid=4343)
        assert result_from_json(result_to_json(result)) == result


# ---------------------------------------------------------------------------
# Task 2307 step-1: is_flock_contention_failure — the discriminant predicate
# consumed by task beta (workstation merge_queue) to recognize alpha's
# distinguished flock-contention VerifyResult.  Mirrors is_unscoped_gate_failure.
# ---------------------------------------------------------------------------


class TestIsFlockContentionFailure:
    """is_flock_contention_failure(vr) is keyed on vr.category == FLOCK_CONTENTION_CATEGORY."""

    def test_true_for_make_flock_contention_result(self):
        from orchestrator.verify_runner import (
            is_flock_contention_failure,
            make_flock_contention_result,
        )
        result = make_flock_contention_result(host='leo-laptop', holder_pgid=4242, waiter_pgid=4343)
        assert is_flock_contention_failure(result) is True

    def test_true_for_hand_built_result_with_matching_category(self):
        from orchestrator.verify_runner import is_flock_contention_failure
        vr = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='',
            summary='flock contention',
            category='flock_contention',
        )
        assert is_flock_contention_failure(vr) is True

    def test_false_for_plain_non_contention_failure_empty_category(self):
        from orchestrator.verify_runner import is_flock_contention_failure
        vr = VerifyResult(
            passed=False,
            test_output='FAILED test_foo',
            lint_output='',
            type_output='',
            summary='1 failure',
        )
        assert is_flock_contention_failure(vr) is False

    def test_false_for_test_failure_category(self):
        from orchestrator.verify_runner import is_flock_contention_failure
        vr = VerifyResult(
            passed=False,
            test_output='FAILED test_foo',
            lint_output='',
            type_output='',
            summary='1 failure',
            category='test_failure',
        )
        assert is_flock_contention_failure(vr) is False

    def test_false_for_passed_true_result(self):
        from orchestrator.verify_runner import is_flock_contention_failure
        vr = VerifyResult(
            passed=True,
            test_output='',
            lint_output='',
            type_output='',
            summary='all good',
        )
        assert is_flock_contention_failure(vr) is False

    def test_importable_and_exported(self):
        from orchestrator import verify_runner
        assert 'is_flock_contention_failure' in verify_runner.__all__
        assert callable(verify_runner.is_flock_contention_failure)


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


def _narrow_config():
    """A real OrchestratorConfig pinned to the NARROW merge-gate profile.

    run_merge_verify_on_worktree does a real ``config.model_copy(update=...)``
    (task 2822 fix a), so its callers need a real config, not a fields-only
    ``pydantic_spec`` mock (which exposes no ``model_copy``).
    """
    return OrchestratorConfig(merge_verify_workspace=False, merge_verify_breadth='scoped')


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
        """The red is returned unchanged — EQUAL, not identical.

        Task 3789 (ε): the merge-flake gate now ATTACHES its observation to the
        result on the non-suppressed branch too (§5.5 records the observation,
        not the remedy), so the returned object is a `replace()` copy. It
        compares EQUAL because `flake_suppression` is `compare=False`, and every
        merge-deciding field is untouched — which is what "unchanged" meant here
        all along.
        """
        scoped_result = _make_fail_result(category='test_failure', cause_hint='assertion error')
        run_scoped = AsyncMock(return_value=scoped_result)
        runner = _make_local_runner(run_scoped=run_scoped)

        result = await runner.run_merge_verify('abc123', _make_spec())

        assert result == scoped_result
        assert result.passed is False
        assert result.category == 'test_failure'
        assert result.cause_hint == 'assertion error'

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
            event_store=None,
        )


# ---------------------------------------------------------------------------
# INV-1 (task 2883): LocalRunner threads event_store into run_scoped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLocalRunnerThreadsEventStore:
    """LocalRunner.run_merge_verify threads its event_store into run_scoped so
    the local merge path emits trivial_pass_escalated (INV-1). The CLI/remote
    in-worktree path constructs the runner with event_store=None and stays
    None-safe (it cannot reach the dispatching host's store)."""

    def _make_runner(self, *, event_store, run_scoped):
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.merge_verify_workspace = False
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False, timed_out=False,
                failing_subprojects=[], timed_out_subprojects=[],
            )
        )
        return LocalRunner(
            merge_wt=MagicMock(),
            config=config,
            module_configs=[],
            task_files=None,
            run_scoped=run_scoped,
            run_unscoped=run_unscoped,
            event_store=event_store,
        )

    async def test_event_store_threaded_into_run_scoped(self):
        sentinel = MagicMock(name='event_store')
        run_scoped = AsyncMock(return_value=_make_pass_result())
        runner = self._make_runner(event_store=sentinel, run_scoped=run_scoped)

        await runner.run_merge_verify('abc123', _make_spec())

        assert run_scoped.await_args is not None
        kwargs = run_scoped.await_args[1]
        assert kwargs['event_store'] is sentinel
        assert kwargs['role'] == 'merge'
        assert kwargs['is_merge_verify'] is True

    async def test_event_store_none_stays_none(self):
        run_scoped = AsyncMock(return_value=_make_pass_result())
        runner = self._make_runner(event_store=None, run_scoped=run_scoped)

        await runner.run_merge_verify('abc123', _make_spec())

        assert run_scoped.await_args is not None
        assert run_scoped.await_args[1]['event_store'] is None


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

    async def test_dispatch_emits_depth_and_speculative_when_provided(self):
        """task 2340: depth/speculative kwargs are threaded into the event
        data dict, alongside the unchanged runner/merge_sha/passed/
        duration_ms/attempt keys."""
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
        await pool.dispatch('sha999', _make_spec(), attempt=0, depth=2, speculative=True)

        assert len(emitted) == 1
        (event_type,), kwargs = emitted[0]
        assert event_type == EventType.merge_verify
        data = kwargs['data']
        assert data['runner'] == 'local'
        assert data['merge_sha'] == 'sha999'
        assert data['passed'] is True
        assert data['attempt'] == 0
        assert 'duration_ms' in data
        assert data['depth'] == 2
        assert data['speculative'] is True

    async def test_dispatch_without_depth_or_speculative_is_none_safe(self):
        """task 2340: the existing call form (no depth/speculative kwargs)
        still emits both keys with value None, and dispatch still does not
        raise with no event_store at all (legacy/bare callers)."""
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import VerifyRunnerPool

        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))
        pool = VerifyRunnerPool([fake_runner], event_store=event_store, task_id='t-42')

        await pool.dispatch('sha999', _make_spec())

        (event_type,), kwargs = emitted[0]
        assert event_type == EventType.merge_verify
        data = kwargs['data']
        assert data['depth'] is None
        assert data['speculative'] is None

        # No event_store at all — still must not raise.
        bare_pool = VerifyRunnerPool([fake_runner], event_store=None)
        result = await bare_pool.dispatch('abc123', _make_spec())
        assert result.passed is True

    async def test_dispatch_emits_retry_scope_for_narrowed_verify(self, tmp_path):
        """task 2837 (PRD D5): a narrowed failed-only post-merge verify carries
        retry_scope='failed_only' + per-suite subset sizes into the merge_verify
        event data (so the survey never miscounts it as a full green gate),
        while a plain full/legacy verify carries None for both keys; the
        pre-existing keys are present/unchanged in either case."""
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import VerifyRunnerPool

        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))
        pool = VerifyRunnerPool([fake_runner], event_store=event_store, task_id='t-42')

        # (a) Narrowed case: two real nextest filter files of 2 and 0 ids.
        debug_file = tmp_path / 'nextest-retry-debug.filter'
        debug_file.write_text('\n'.join(['id::a', 'id::b']))  # 2 ids
        release_file = tmp_path / 'nextest-retry-release.filter'
        release_file.write_text('')  # 0 ids (0-byte)
        narrowed_spec = dataclasses.replace(
            _make_spec(),
            verify_env={
                'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
                'REIFY_RUN_ALL_MEMBER_SUBSET': 'm1',
                'REIFY_GUI_RETRY_SPECS': '',
                'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG': str(debug_file),
                'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE': str(release_file),
            },
        )
        await pool.dispatch('sha-narrow', narrowed_spec, attempt=1, depth=3, speculative=False)

        (event_type,), kwargs = emitted[-1]
        assert event_type == EventType.merge_verify
        data = kwargs['data']
        assert data['retry_scope'] == 'failed_only'
        assert data['retry_subset_sizes'] == {
            'run_all': 1,
            'gui': 0,
            'nextest_debug': 2,
            'nextest_release': 0,
        }
        # Pre-existing keys still present/unchanged.
        assert data['runner'] == 'local'
        assert data['merge_sha'] == 'sha-narrow'
        assert data['passed'] is True
        assert data['attempt'] == 1
        assert 'duration_ms' in data
        assert data['depth'] == 3
        assert data['speculative'] is False

        # (b) Legacy None-safe case: plain _make_spec() (verify_env={}).
        await pool.dispatch('sha-full', _make_spec())
        (event_type,), kwargs = emitted[-1]
        assert event_type == EventType.merge_verify
        data = kwargs['data']
        assert data['retry_scope'] is None
        assert data['retry_subset_sizes'] is None


@pytest.mark.asyncio
class TestDispatchChainItems:
    """``chain_items`` on the merge_verify event (task 3185, PRD γ decision 8).

    A 1-INDEXED count of items in the tree that was actually verified, so the
    smallest truthful value is ``1`` — which is why the kwarg defaults to ``1``
    and NOT to ``None`` the way ``depth``/``speculative`` do.  The PRD contract
    is "``chain_items >= 1`` on EVERY merge verify", and the two merge_shadow.py
    callers (:1254, :1368) thread no telemetry kwargs at all, so a ``None``
    default would emit ``chain_items: null`` there and break both the contract
    and η1's already-committed reader (scripts/merge-deep-canary-predicate.sh:84).

    Template: ``test_dispatch_emits_retry_scope_for_narrowed_verify`` above —
    including its explicit re-assertion of every pre-existing key, so a future
    payload edit cannot silently drop one.
    """

    def _make_local_pool(self, **pool_kwargs):
        """Return ``(pool, emitted)`` over a single passing local runner."""
        from orchestrator.verify_runner import VerifyRunnerPool

        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))
        pool = VerifyRunnerPool(
            [fake_runner], event_store=event_store, task_id='t-chain', **pool_kwargs,
        )
        return pool, emitted

    async def test_bare_call_emits_chain_items_one_not_none(self):
        """No telemetry kwargs → ``chain_items == 1`` (an int, never None).

        This is the merge_shadow.py:1254 / :1368 case: those callers pass
        nothing, and they are exactly the ones that make ">= 1 on EVERY merge
        verify" a real contract rather than a wish.
        """
        from orchestrator.event_store import EventType

        pool, emitted = self._make_local_pool()

        await pool.dispatch('sha-bare', _make_spec())

        (event_type,), kwargs = emitted[-1]
        assert event_type == EventType.merge_verify
        data = kwargs['data']
        assert data['chain_items'] == 1
        assert isinstance(data['chain_items'], int)
        # ``chain_build_ms`` is the nullable companion: a bare dispatch chained
        # nothing, so it paid no build.  Present-but-None, never absent, so a
        # reader does a plain .get() with no per-event schema branching.
        assert 'chain_build_ms' in data
        assert data['chain_build_ms'] is None

    async def test_explicit_chain_build_ms_is_emitted_verbatim(self):
        """The per-round DISPATCH STALL reaches telemetry alongside chain_items.

        Non-None implies a chain was built, so it always rides with
        ``chain_items >= 2``; η1 reads the pair against drain-time so a
        dispatch stall cannot be misattributed to verify time.
        """
        from orchestrator.event_store import EventType

        pool, emitted = self._make_local_pool()

        await pool.dispatch(
            'sha-deep', _make_spec(), chain_items=3, chain_build_ms=1234,
        )

        (event_type,), kwargs = emitted[-1]
        assert event_type == EventType.merge_verify
        assert kwargs['data']['chain_build_ms'] == 1234
        assert kwargs['data']['chain_items'] == 3

    async def test_explicit_chain_items_is_emitted_verbatim(self):
        from orchestrator.event_store import EventType

        pool, emitted = self._make_local_pool()

        await pool.dispatch('sha-deep', _make_spec(), chain_items=4)

        (event_type,), kwargs = emitted[-1]
        assert event_type == EventType.merge_verify
        assert kwargs['data']['chain_items'] == 4

    async def test_preexisting_keys_present_and_unchanged(self, tmp_path):
        """The new key is ADDITIVE — every pre-2340/2837 key still rides along.

        Copied from the retry_scope suite's re-assertion block so a future
        payload edit cannot drop one silently.
        """
        from orchestrator.event_store import EventType

        pool, emitted = self._make_local_pool()

        debug_file = tmp_path / 'nextest-retry-debug.filter'
        debug_file.write_text('\n'.join(['id::a', 'id::b']))  # 2 ids
        narrowed_spec = dataclasses.replace(
            _make_spec(),
            verify_env={
                'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
                'REIFY_RUN_ALL_MEMBER_SUBSET': 'm1',
                'REIFY_GUI_RETRY_SPECS': '',
                'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG': str(debug_file),
            },
        )

        await pool.dispatch(
            'sha-all-keys', narrowed_spec,
            attempt=1, depth=3, speculative=False, chain_items=5,
        )

        (event_type,), kwargs = emitted[-1]
        assert event_type == EventType.merge_verify
        data = kwargs['data']
        assert data['chain_items'] == 5
        # Pre-existing keys still present/unchanged.
        assert data['runner'] == 'local'
        assert data['merge_sha'] == 'sha-all-keys'
        assert data['passed'] is True
        assert 'duration_ms' in data
        assert data['attempt'] == 1
        assert data['depth'] == 3
        assert data['speculative'] is False
        assert data['retry_scope'] == 'failed_only'
        assert data['retry_subset_sizes'] == {
            'run_all': 1,
            'gui': 0,
            'nextest_debug': 2,
            'nextest_release': None,
        }

    async def test_no_event_store_does_not_raise(self):
        """``event_store=None`` still must not raise with the new kwarg."""
        from orchestrator.verify_runner import VerifyRunnerPool

        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        bare_pool = VerifyRunnerPool([fake_runner], event_store=None)
        result = await bare_pool.dispatch('abc123', _make_spec(), chain_items=3)

        assert result.passed is True

    async def test_survives_runner_unavailable_local_fallback(self):
        """The value survives the RunnerUnavailable→local-fallback arm.

        On that arm ``actual_runner`` differs from ``selected``
        (verify_runner.py:2011-2024), and the emit happens AFTER the swap — so
        this pins that the fallback path does not lose the caller's telemetry.
        """
        from orchestrator.verify_runner import RunnerUnavailable, VerifyRunnerPool

        remote_fake = MagicMock(spec=VerifyRunner)
        remote_fake.name = 'laptop'
        remote_fake.is_local = False
        remote_fake.run_merge_verify = AsyncMock(
            side_effect=RunnerUnavailable('host down'),
        )
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        local_fake.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append((a, kw)))
        pool = VerifyRunnerPool(
            [local_fake, remote_fake], event_store=event_store,
        )

        await pool.dispatch('sha-fallback', _make_spec(), chain_items=7)

        assert len(emitted) == 1
        (_, kwargs) = emitted[0]
        data = kwargs['data']
        assert data['runner'] == 'local'      # the fallback ran
        assert data['chain_items'] == 7       # ...and telemetry survived it


@pytest.mark.asyncio
class TestDispatchDurationWiring:
    """The persisted merge_verify payload's ``duration_ms`` is the runner's real wall time.

    It rides in ``data`` only; the events.duration_ms column stays NULL, so
    scripts/merge-deep-canary-predicate.sh reads its verify_p90 from here.
    """

    async def test_persisted_merge_verify_payload_carries_the_runner_wall_time(self, tmp_path):
        async def _slow_pass(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return _make_pass_result()

        fake_runner = MagicMock(spec=VerifyRunner)
        fake_runner.name = 'local'
        fake_runner.is_local = True
        fake_runner.run_merge_verify = AsyncMock(side_effect=_slow_pass)
        db_path = tmp_path / 'runs.db'
        pool = VerifyRunnerPool(
            [fake_runner],
            event_store=EventStore(db_path, 'run-duration'),
            task_id='t-dur',
        )

        t0 = time.monotonic()
        await pool.dispatch('sha-dur', _make_spec(), chain_items=2)
        outer_ms = (time.monotonic() - t0) * 1000

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT data, duration_ms FROM events WHERE event_type = 'merge_verify'"
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 1, rows
        (data, duration_column), = rows
        assert duration_column is None, rows
        duration_ms = json.loads(data)['duration_ms']
        assert isinstance(duration_ms, int) and not isinstance(duration_ms, bool), duration_ms
        assert 150 <= duration_ms <= outer_ms + 1, (duration_ms, outer_ms)


# ---------------------------------------------------------------------------
# Task 3789 (ε) step-13: dispatch re-stamps FlakeSuppression.runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchStampsFlakeSuppressionRunner:
    """``VerifyRunnerPool.dispatch`` re-stamps the carried observation's ``runner``.

    ``FlakeSuppression.runner`` means WHERE the isolated re-run actually executed.
    The discriminator stamps ``'local'`` — honest, but host-RELATIVE: on a remote
    runner it means "local to that remote", and read back on the dispatcher it is
    simply wrong.  Left uncorrected, the ledger's ``runner`` column would read
    ``'local'`` for every observation in the fleet, which is exactly the column θ's
    class-3 systemic check reads to tell a bad HOST from a bad SUITE.

    ``dispatch`` is the only scope that knows which runner really ran, and it knows
    it only AFTER the ``RunnerUnavailable``->local fallback — which is what makes
    ``merge_queue``'s own ``runner`` parameter an unreliable source and puts the
    stamp here rather than at the recorder's call site.
    """

    @staticmethod
    def _runner(name: str, *, is_local: bool, result=None, unavailable: bool = False):
        from orchestrator.verify_runner import RunnerUnavailable

        r = MagicMock(spec=VerifyRunner)
        r.name = name
        r.is_local = is_local
        r.run_merge_verify = AsyncMock(
            side_effect=RunnerUnavailable(f'{name} down') if unavailable else None,
            return_value=result,
        )
        return r

    @staticmethod
    def _result_with(suppression):
        return _make_pass_result(
            category='merge_flake_suppressed', flake_suppression=suppression,
        )

    async def test_remote_dispatch_restamps_the_runner_name(self):
        """(a) The headline: a remote's honest self-report of ``'local'`` becomes the
        remote's NAME, and nothing else about the observation is disturbed."""
        from orchestrator.verify_runner import VerifyRunnerPool

        s = _make_suppression(runner='local')
        remote = self._runner('remote-lab-1', is_local=False, result=self._result_with(s))
        pool = VerifyRunnerPool([remote])

        result = await pool.dispatch('sha1', _make_spec())

        assert result.flake_suppression is not None
        assert result.flake_suppression.runner == 'remote-lab-1'
        # Every other field survives untouched — the stamp is a correction of ONE
        # column, not a re-derivation of the observation.
        assert result.flake_suppression == dataclasses.replace(s, runner='remote-lab-1')

    async def test_local_dispatch_leaves_local(self):
        """(b) On a local dispatch the discriminator's stamp was already right, so
        the correction is a no-op rather than a rename."""
        from orchestrator.verify_runner import VerifyRunnerPool

        s = _make_suppression(runner='local')
        local = self._runner('local', is_local=True, result=self._result_with(s))
        pool = VerifyRunnerPool([local])

        result = await pool.dispatch('sha1', _make_spec())

        assert result.flake_suppression is not None
        assert result.flake_suppression.runner == 'local'

    async def test_runner_unavailable_fallback_stamps_the_runner_that_actually_ran(self):
        """(c) The case that decides WHERE the stamp lives.

        The remote is selected and dies; the LOCAL runner produces the observation.
        The stamp must name the runner that actually ran, never the one that was
        selected — otherwise a fallback verify would file its flakes against an
        innocent host, and that host is precisely what θ's class-3 check indicts.
        """
        from orchestrator.verify_runner import VerifyRunnerPool

        # A deliberately WRONG incoming value, so a missing stamp cannot pass by
        # coincidence and a stamp taken from `selected` fails loudly.
        s = _make_suppression(runner='stale-stamp')
        remote = self._runner('remote-lab-1', is_local=False, unavailable=True)
        local = self._runner('local', is_local=True, result=self._result_with(s))
        pool = VerifyRunnerPool([remote, local])

        result = await pool.dispatch('sha1', _make_spec())

        assert result.flake_suppression is not None
        assert result.flake_suppression.runner == 'local'

    async def test_no_suppression_passes_through_untouched(self):
        """(d) B13 — an old remote carries no observation at all.  The stamp must be
        a no-op on None, not an AttributeError inside the merge path."""
        from orchestrator.verify_runner import VerifyRunnerPool

        remote = self._runner('remote-lab-1', is_local=False, result=_make_pass_result())
        pool = VerifyRunnerPool([remote])

        result = await pool.dispatch('sha1', _make_spec())

        assert result.flake_suppression is None
        assert result.passed is True

    async def test_a_non_dataclass_result_is_not_destroyed_by_the_stamp(self):
        """(d2) BOTH objects ``replace`` touches are guarded, not just the inner one.

        The inner ``isinstance(carried, FlakeSuppression)`` check protects the
        observation, but the OUTER ``dataclasses.replace(result, ...)`` needs
        ``result`` to be a dataclass instance too.  A runner returning a
        Protocol-conformant fake or a test double that happens to carry a real
        ``FlakeSuppression`` would otherwise raise ``TypeError`` out of ``dispatch``
        and into the merge path — which has no ``VerifyInfraError`` handler, so a
        bookkeeping correction would take down a real verdict.

        Same discipline as ``verify._is_attachable``: an observation is evidence
        ABOUT a verdict and must never be able to destroy the verdict it describes.
        The stamp degrades to a no-op and the caller keeps its result.
        """
        from orchestrator.verify_runner import VerifyRunnerPool

        class _NotADataclass:
            passed = True
            category = 'merge_flake_suppressed'
            trivial = False

            def __init__(self, suppression):
                self.flake_suppression = suppression

        carried = _make_suppression(runner='local')
        fake = _NotADataclass(carried)
        remote = self._runner('remote-lab-1', is_local=False, result=fake)
        pool = VerifyRunnerPool([remote])

        result = await pool.dispatch('sha1', _make_spec())

        assert result is fake, 'the caller keeps its own verdict'
        # Un-stamped rather than crashed: the observation is still the one the
        # discriminator produced, just without the runner correction.
        assert result.flake_suppression is carried

    async def test_stamping_is_pure(self):
        """(e) ``dispatch`` stays a TRANSPORT concern: it corrects one field and
        records nothing.  Recording belongs to the dispatcher's merge path, which
        alone holds the project root, merge SHA and task id — and doing it twice
        would double-count every observation in the ledger."""
        from orchestrator import flake_recorder
        from orchestrator.verify_runner import VerifyRunnerPool

        remote = self._runner(
            'remote-lab-1', is_local=False,
            result=self._result_with(_make_suppression(runner='local')),
        )
        emitted = []
        event_store = MagicMock()
        event_store.emit = MagicMock(side_effect=lambda *a, **kw: emitted.append(a[0]))

        record = MagicMock()
        with patch.object(flake_recorder, 'record_merge_flake_suppression', record):
            pool = VerifyRunnerPool([remote], event_store=event_store, task_id='t-1')
            await pool.dispatch('sha1', _make_spec())

        record.assert_not_called()
        # The pre-existing merge_verify telemetry is unaffected; nothing else is emitted.
        assert emitted == [EventType.merge_verify]

# ---------------------------------------------------------------------------
# retry_scope_event_fields — merge_verify event honesty (task 2837, PRD D5)
# ---------------------------------------------------------------------------


class TestRetryScopeEventFields:
    """retry_scope_event_fields derives the merge_verify event's retry_scope +
    per-suite subset sizes from spec.verify_env (the D2 failed-only contract).

    A full/legacy verify (no REIFY_VERIFY_RETRY_SCOPE='failed_only') carries no
    scope signal, so the survey's runtime mining never miscounts a narrowed
    retry as a full green gate (PRD verify-retry-failed-only §4.4/§5.6, INV-2).
    """

    def test_retry_scope_event_fields_absent_when_not_failed_only(self):
        from orchestrator.verify_runner import retry_scope_event_fields

        # (a) empty verify_env — the common full/legacy verify.
        assert retry_scope_event_fields({}) == {
            'retry_scope': None,
            'retry_subset_sizes': None,
        }
        # (b) some other/legacy REIFY_VERIFY_RETRY_SCOPE value — still no signal.
        assert retry_scope_event_fields({'REIFY_VERIFY_RETRY_SCOPE': 'full'}) == {
            'retry_scope': None,
            'retry_subset_sizes': None,
        }

    def test_retry_scope_event_fields_narrowed_sizes(self, tmp_path):
        from orchestrator.verify_runner import retry_scope_event_fields

        # Mirror _build_retry_verify_env's '\n'.join(ids) filter-file format.
        debug_file = tmp_path / 'nextest-retry-debug.filter'
        debug_file.write_text('\n'.join(['id::a', 'id::b', 'id::c']))  # 3 ids
        release_file = tmp_path / 'nextest-retry-release.filter'
        release_file.write_text('\n'.join(['id::z']))  # 1 id

        # SPACE-delimited (task 3059): merge_queue._build_retry_verify_env is the
        # single source of truth for this format, and reify word-splits both
        # values.  Counting with .split(',') would report 1 and 1 here.
        verify_env = {
            'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
            'REIFY_RUN_ALL_MEMBER_SUBSET': 'a.sh b.sh c.sh',
            'REIFY_GUI_RETRY_SPECS': 'x.test.ts y.test.ts',
            'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG': str(debug_file),
            'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE': str(release_file),
        }
        assert retry_scope_event_fields(verify_env) == {
            'retry_scope': 'failed_only',
            'retry_subset_sizes': {
                'run_all': 3,
                'gui': 2,
                'nextest_debug': 3,
                'nextest_release': 1,
            },
        }

        # Empty-subset edge: '' counts 0 tokens (str.split() with no argument
        # already drops empties, dodging the ''.split(',') == [''] pitfall), and
        # a 0-byte filter file counts 0 lines (dodge ''.splitlines() == []).
        empty_file = tmp_path / 'nextest-retry-empty.filter'
        empty_file.write_text('')  # 0 bytes
        empty_env = {
            'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
            'REIFY_RUN_ALL_MEMBER_SUBSET': '',
            'REIFY_GUI_RETRY_SPECS': '',
            'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG': str(empty_file),
            'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE': str(release_file),
        }
        result = retry_scope_event_fields(empty_env)
        assert result['retry_scope'] == 'failed_only'
        assert result['retry_subset_sizes']['run_all'] == 0
        assert result['retry_subset_sizes']['gui'] == 0
        assert result['retry_subset_sizes']['nextest_debug'] == 0

    def test_retry_scope_event_fields_nextest_read_degrades_honestly(self, tmp_path):
        from orchestrator.verify_runner import retry_scope_event_fields

        # Narrowed verify_env, but the debug filter path is MISSING and the
        # release key is entirely ABSENT — INV-2 honest degrade to None (never a
        # crash), while the comma-delimited run_all/gui still compute correctly.
        verify_env = {
            'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
            'REIFY_RUN_ALL_MEMBER_SUBSET': 'm1 m2',
            'REIFY_GUI_RETRY_SPECS': 'ui/x.ts',
            'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG': str(tmp_path / 'missing.filter'),
            # REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE deliberately absent.
        }
        result = retry_scope_event_fields(verify_env)
        assert result['retry_scope'] == 'failed_only'
        sizes = result['retry_subset_sizes']
        assert sizes['nextest_debug'] is None   # unreadable path → None
        assert sizes['nextest_release'] is None  # absent key → None
        assert sizes['run_all'] == 2             # space-values still compute
        assert sizes['gui'] == 1

    def test_retry_scope_event_fields_nextest_non_utf8_degrades_honestly(self, tmp_path):
        from orchestrator.verify_runner import retry_scope_event_fields

        # A filter file with non-UTF-8 bytes makes Path.read_text() raise
        # UnicodeDecodeError — a ValueError subclass, NOT an OSError.  Per INV-2
        # the read must still degrade to None rather than propagate and abort the
        # whole merge_verify event emission (amend: reviewer robustness note).
        bad_file = tmp_path / 'nextest-retry-nonutf8.filter'
        bad_file.write_bytes(b'\x80\x81\x82')  # invalid UTF-8 (lone continuation bytes)
        verify_env = {
            'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
            'REIFY_RUN_ALL_MEMBER_SUBSET': 'm1 m2',
            'REIFY_GUI_RETRY_SPECS': 'ui/x.ts',
            'REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG': str(bad_file),
            # REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE deliberately absent.
        }
        result = retry_scope_event_fields(verify_env)
        assert result['retry_scope'] == 'failed_only'
        sizes = result['retry_subset_sizes']
        assert sizes['nextest_debug'] is None   # undecodable file → None (no crash)
        assert sizes['nextest_release'] is None  # absent key → None
        assert sizes['run_all'] == 2             # space-values still compute
        assert sizes['gui'] == 1


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

    def _make_config(
        self, *, verify_env=None, cold_timeout=None,
        test_cmd=None, lint_cmd=None, type_check_cmd=None,
    ):
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.verify_env = verify_env or {}
        config.effective_verify_env = verify_env or {}
        config.merge_verify_cold_command_timeout_secs = cold_timeout
        config.verify_cold_command_timeout_secs = None
        # Fix (a), task 2822: build_merge_verify_spec now projects these two
        # profile fields from config, so the double must carry real values
        # (a bare MagicMock attr is not JSON-serialisable in spec_to_json).
        config.merge_verify_workspace = False
        config.merge_verify_breadth = 'scoped'
        # INV-1, task 2883: build_merge_verify_spec now reads the global
        # full-gate commands to source spec.global_verify_command when the
        # scope resolves to zero module_configs. Default None (a bare MagicMock
        # attr is truthy and not JSON-serialisable), overridable per-test.
        config.test_command = test_cmd
        config.lint_command = lint_cmd
        config.type_check_command = type_check_cmd
        # Task 6127: projected onto the spec, so it needs a real (JSON-native) value.
        config.verify_cold_preprovision_command = ''
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
        config.test_command = None
        config.lint_command = None
        config.type_check_command = None
        spec = build_merge_verify_spec(config, [], None)
        assert spec.cold_timeout_secs == 3600.0

    def test_cold_timeout_falls_back_to_zero_when_both_none(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.verify_env = {}
        config.effective_verify_env = {}
        config.merge_verify_cold_command_timeout_secs = None
        config.verify_cold_command_timeout_secs = None
        config.test_command = None
        config.lint_command = None
        config.type_check_command = None
        spec = build_merge_verify_spec(config, [], None)
        assert spec.cold_timeout_secs == 0.0

    def test_is_merge_verify_is_true(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        spec = build_merge_verify_spec(self._make_config(), [], None)
        assert spec.is_merge_verify is True

    # --- INV-1, task 2883: ship the global full-gate commands for a
    # zero-module-config project (reify) so the remote runs the SAME gate as
    # local, not its own possibly-stale config (fidelity hole behind 966f23a6).

    def test_global_verify_command_sourced_when_no_module_configs(self):
        """(a) With NO module_configs, spec.global_verify_command carries the
        config's three global full-gate commands (prefix='')."""
        from orchestrator.verify_runner import build_merge_verify_spec
        config = self._make_config(
            test_cmd='cargo test --workspace',
            lint_cmd='cargo clippy --workspace',
            type_check_cmd='pyright',
        )
        spec = build_merge_verify_spec(config, [], ('docs/x.md',))
        assert spec.global_verify_command is not None
        gvc = spec.global_verify_command
        assert gvc.prefix == ''
        assert gvc.test_command == 'cargo test --workspace'
        assert gvc.lint_command == 'cargo clippy --workspace'
        assert gvc.type_check_command == 'pyright'

    def test_global_verify_command_none_when_module_configs_present(self):
        """(b) With a non-empty module_configs the global command is NOT
        sourced — the per-module verify_commands already drive the gate."""
        from orchestrator.verify_runner import build_merge_verify_spec
        config = self._make_config(
            test_cmd='cargo test --workspace',
            lint_cmd='cargo clippy --workspace',
            type_check_cmd='pyright',
        )
        mc = self._make_module_config('src/a', test_cmd='pytest src/a')
        spec = build_merge_verify_spec(config, [mc], ('src/a/mod.py',))
        assert spec.global_verify_command is None

    def test_global_verify_command_none_when_no_global_commands(self):
        """A command-less config (all global commands None) with no
        module_configs sources NO global command (nothing to ship)."""
        from orchestrator.verify_runner import build_merge_verify_spec
        spec = build_merge_verify_spec(self._make_config(), [], ('docs/x.md',))
        assert spec.global_verify_command is None

    def test_global_verify_command_round_trips_json_codec(self):
        """(c) spec_from_json(spec_to_json(spec)) preserves global_verify_command."""
        from orchestrator.verify_runner import (
            build_merge_verify_spec,
            spec_from_json,
            spec_to_json,
        )
        config = self._make_config(
            verify_env={'K': 'V'}, cold_timeout=300.0,
            test_cmd='cargo test --workspace',
            lint_cmd='cargo clippy --workspace',
            type_check_cmd='pyright',
        )
        spec = build_merge_verify_spec(config, [], ('docs/x.md',))
        restored = spec_from_json(spec_to_json(spec))
        assert restored == spec
        assert restored.global_verify_command == spec.global_verify_command

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

    # --- task 6127: the pre-provision command is dispatcher-carried ----------

    def test_verify_cold_preprovision_command_projected_from_config(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        config = OrchestratorConfig(verify_cold_preprovision_command='SENTINEL_PROVISION')
        spec = build_merge_verify_spec(config, [], None)
        assert spec.verify_cold_preprovision_command == 'SENTINEL_PROVISION'

    @pytest.mark.usefixtures('code_default_config')
    def test_unconfigured_preprovision_projects_empty(self):
        from orchestrator.verify_runner import build_merge_verify_spec
        spec = build_merge_verify_spec(OrchestratorConfig(), [], None)
        assert spec.verify_cold_preprovision_command == ''


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
        config = _narrow_config()
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
        config = _narrow_config()
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
        config = _narrow_config()
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

    async def test_spec_profile_overrides_host_config(self):
        """Fix (a): the SPEC's merge-gate profile wins over the (remote) host
        config, so the remote runs the SAME scope/profile as the merge gate
        rather than the laptop config's narrow defaults."""
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
        # The (remote) host config carries the NARROW laptop defaults ...
        # (set explicitly: OrchestratorConfig is a BaseSettings whose bare
        # defaults may be widened by a settings source, so pin them here).
        config = OrchestratorConfig(merge_verify_workspace=False, merge_verify_breadth='scoped')
        assert config.merge_verify_workspace is False
        assert config.merge_verify_breadth == 'scoped'
        # ... but the spec carries the FULL merge-gate profile.
        spec = MergeVerifySpec(
            verify_commands=(VerifyCommand('src/a', test_command='true'),),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('src/a/m.py',),
            verify_env={},
            cold_timeout_secs=60.0,
            merge_verify_workspace=True,
            merge_verify_breadth='full',
        )

        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert run_scoped.await_args is not None
        # force_workspace is read from the (now spec-overridden) config -> True.
        assert run_scoped.await_args[1]['force_workspace'] is True
        # The config object threaded into run_scoped carries the spec's breadth,
        # not the laptop's 'scoped'.
        effective_config = run_scoped.await_args[0][1]
        assert effective_config.merge_verify_breadth == 'full'
        assert effective_config.merge_verify_workspace is True

    async def test_spec_module_set_overrides_host_config_registry(self):
        """Task 4536: the SPEC names the module SET, so the (remote) host's own
        `_discover_module_configs` registry can neither narrow nor widen it.

        The natural extension of fix (a) above: the module set is the third
        spec-supplied element of the merge-deciding profile, alongside
        merge_verify_workspace/merge_verify_breadth (task 2822) and the global
        commands (INV-1, task 2883). Without it the reconstructed set is passed
        positionally but then DISCARDED by
        verify_plan.effective_merge_module_configs, which prefers
        `config.module_configs_or_empty` under breadth='full'.
        """
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False, timed_out=False,
                failing_subprojects=[], timed_out_subprojects=[],
            )
        )
        # The (remote) host config carries a registry of HOST-ONLY modules with
        # stale commands (the blessed direct-assignment idiom — _module_configs
        # is a PrivateAttr, not a model field). Profile fields are pinned
        # explicitly: OrchestratorConfig is a BaseSettings whose bare defaults
        # may be widened by a settings source.
        config = OrchestratorConfig(
            merge_verify_workspace=False, merge_verify_breadth='scoped',
        )
        config._module_configs = {
            'host/only': ModuleConfig(prefix='host/only', test_command='STALE_HOST_TEST'),
            'host/gone': ModuleConfig(prefix='host/gone', test_command='STALE_HOST_TEST_2'),
        }
        # ... but the spec carries a DIFFERENT, larger set (as the merge-request
        # boundary would have widened it before build_merge_verify_spec ran).
        spec = MergeVerifySpec(
            verify_commands=(
                VerifyCommand('src/a', test_command='SPEC_TEST_A', lint_command='SPEC_LINT_A'),
                VerifyCommand('src/b', test_command='SPEC_TEST_B',
                              type_check_command='SPEC_TYPE_B'),
                VerifyCommand('src/c', lint_command='SPEC_LINT_C'),
            ),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('src/a/m.py',),
            verify_env={},
            cold_timeout_secs=60.0,
            merge_verify_workspace=False,
            merge_verify_breadth='full',
        )

        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert run_scoped.await_args is not None
        effective_config = run_scoped.await_args[0][1]
        spec_prefixes = {vc.prefix for vc in spec.verify_commands}
        registry = effective_config.module_configs_or_empty
        assert set(registry) == spec_prefixes, (
            f'the config registry threaded into the merge must be the SPEC\'s set; '
            f'expected {spec_prefixes!r}, got {set(registry)!r}'
        )
        assert not {'host/only', 'host/gone'} & set(registry), (
            'the remote host\'s own discovered modules must not survive into the '
            'merge-deciding registry'
        )
        for vc in spec.verify_commands:
            mc = registry[vc.prefix]
            assert mc.test_command == vc.test_command
            assert mc.lint_command == vc.lint_command
            assert mc.type_check_command == vc.type_check_command

        # The AGREEMENT invariant — the unit-level statement of
        # effective_merge_module_configs' INV-5. The registry and the
        # positionally-passed list are two sources a downstream reader could
        # consult; pinning that they cannot disagree is what makes it impossible
        # to pick the wrong one.
        passed_modules = run_scoped.await_args[0][2]
        assert {mc.prefix for mc in passed_modules} == spec_prefixes
        assert {
            (mc.prefix, mc.test_command, mc.lint_command, mc.type_check_command)
            for mc in passed_modules
        } == {
            (mc.prefix, mc.test_command, mc.lint_command, mc.type_check_command)
            for mc in registry.values()
        }, (
            'config.module_configs_or_empty and the passed module_configs must name '
            'the same modules with the same commands (INV-5)'
        )
        # The unscoped typecheck gate reads the same set.
        assert run_unscoped.await_args is not None
        assert {mc.prefix for mc in run_unscoped.await_args[0][2]} == spec_prefixes

    async def test_caller_config_registry_is_not_mutated(self):
        """Task 4536, constraint 2: the registry install must REBIND the dict,
        never mutate it in place.

        ``model_copy`` rebuilds the ``__pydantic_private__`` MAPPING as a fresh
        dict but carries its VALUES over unchanged — so the copy's
        ``_module_configs`` is initially the SAME dict object as the caller's.
        A ``.clear()``/``.update()`` spelling would therefore reach through that
        shared value and silently corrupt the CALLER's config — the object
        cli.py loaded from disk and may still use — while a rebind cannot. This
        is why the rebind is load-bearing rather than stylistic.
        """
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False, timed_out=False,
                failing_subprojects=[], timed_out_subprojects=[],
            )
        )
        caller_config = OrchestratorConfig(
            merge_verify_workspace=False, merge_verify_breadth='scoped',
        )
        host_modules = {
            'host/only': ModuleConfig(prefix='host/only', test_command='STALE_HOST_TEST'),
            'host/gone': ModuleConfig(prefix='host/gone', test_command='STALE_HOST_TEST_2'),
        }
        caller_config._module_configs = host_modules
        original_dict = caller_config.module_configs_or_empty
        original_items = dict(host_modules)

        spec = MergeVerifySpec(
            verify_commands=(
                VerifyCommand('src/a', test_command='SPEC_TEST_A'),
                VerifyCommand('src/b', test_command='SPEC_TEST_B'),
            ),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('src/a/m.py',),
            verify_env={},
            cold_timeout_secs=60.0,
            merge_verify_workspace=False,
            merge_verify_breadth='full',
        )

        await run_merge_verify_on_worktree(
            MagicMock(), caller_config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert caller_config.module_configs_or_empty is original_dict, (
            'the caller\'s registry dict object must be untouched — the copy\'s '
            '_module_configs starts out as the SAME dict object, so an in-place '
            'mutation of the COPY reaches through that shared value and '
            'corrupts the caller\'s config'
        )
        assert caller_config.module_configs_or_empty == original_items, (
            f'the caller\'s registry contents must be unchanged; got '
            f'{set(caller_config.module_configs_or_empty)!r}'
        )
        # Sanity: the copy really did receive the spec's set, so the identity
        # assertion above is not passing vacuously against a no-op fix.
        call_args = run_scoped.await_args
        assert call_args is not None
        effective_config = call_args[0][1]
        assert set(effective_config.module_configs_or_empty) == {'src/a', 'src/b'}

    async def test_spec_global_verify_command_applied_onto_config(self):
        """INV-1 (task 2883): a spec's global_verify_command overrides the
        (remote) host config's global commands, so a zero-module-config project
        runs the SAME full gate as local — preserving remote↔local scope parity
        without injecting a synthetic module (incident 966f23a6)."""
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False, timed_out=False,
                failing_subprojects=[], timed_out_subprojects=[],
            )
        )
        # The (remote) host config carries STALE global commands ...
        config = OrchestratorConfig(
            test_command='ORIG_TEST', lint_command='ORIG_LINT',
            type_check_command='ORIG_TYPE',
            merge_verify_workspace=False, merge_verify_breadth='scoped',
        )
        # ... but the spec ships the dispatching side's LIVE full gate.
        spec = MergeVerifySpec(
            verify_commands=(),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('docs/x.md',),
            verify_env={},
            cold_timeout_secs=60.0,
            global_verify_command=VerifyCommand(
                prefix='',
                test_command='SENTINEL_TEST',
                lint_command='SENTINEL_LINT',
                type_check_command='SENTINEL_TYPE',
            ),
        )

        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert run_scoped.await_args is not None
        effective_config = run_scoped.await_args[0][1]
        assert effective_config.test_command == 'SENTINEL_TEST'
        assert effective_config.lint_command == 'SENTINEL_LINT'
        assert effective_config.type_check_command == 'SENTINEL_TYPE'

    async def test_none_global_verify_command_leaves_config_globals_unchanged(self):
        """With global_verify_command=None the reconstructed config's global
        commands are left untouched (a normal per-module merge is unaffected)."""
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False, timed_out=False,
                failing_subprojects=[], timed_out_subprojects=[],
            )
        )
        config = OrchestratorConfig(
            test_command='ORIG_TEST', lint_command='ORIG_LINT',
            type_check_command='ORIG_TYPE',
            merge_verify_workspace=False, merge_verify_breadth='scoped',
        )
        spec = MergeVerifySpec(
            verify_commands=(VerifyCommand('src/a', test_command='true'),),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('src/a/m.py',),
            verify_env={},
            cold_timeout_secs=60.0,
            global_verify_command=None,
        )

        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert run_scoped.await_args is not None
        effective_config = run_scoped.await_args[0][1]
        assert effective_config.test_command == 'ORIG_TEST'
        assert effective_config.lint_command == 'ORIG_LINT'
        assert effective_config.type_check_command == 'ORIG_TYPE'

    @staticmethod
    def _host_config(**kwargs):
        """A real (never MagicMock) host config on the merge-verify shape.

        run_merge_verify_on_worktree calls a real ``model_copy``, and the whole
        point of the verify_env merge is what the REAL config properties compute.
        """
        return OrchestratorConfig(
            merge_verify_workspace=False, merge_verify_breadth='scoped', **kwargs
        )

    @staticmethod
    async def _merge_verify_effective_config(config, spec_env, verify_commands=()):
        """Run the merge-verify path; return what was threaded into run_scoped.

        Returns ``(effective_config, module_configs)`` — ``run_scoped.await_args``
        positions 1 and 2, the observation point the sibling 2822/2883/4536
        override tests use.  ``verify_commands=()`` is the zero-module-config
        project (reify), where no ModuleConfig is reconstructed at all.
        """
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False, timed_out=False,
                failing_subprojects=[], timed_out_subprojects=[],
            )
        )
        spec = MergeVerifySpec(
            verify_commands=verify_commands,
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('src/a/m.py',),
            verify_env=dict(spec_env),
            cold_timeout_secs=60.0,
        )

        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )

        assert run_scoped.await_args is not None
        return run_scoped.await_args[0][1], run_scoped.await_args[0][2]

    async def test_host_only_verify_env_key_survives_spec_merge(self):
        """Both sides' keys reach the zero-module remote config at once.

        Without the merge, spec.verify_env has NO consumer on this path (no
        ModuleConfig is reconstructed), so the remote decides the merge under its
        own config's test-selection env — the task-2822 false-green class.  With
        a 'replace' spelling the host's own keys vanish instead.  The live
        host-only class is the remote verify runner's own ``--config`` (reify
        wires one at ``/home/leo/.config/orchestrator/reify-laptop.yaml``): the
        16-thread laptop widens the ``REIFY_VERIFY_*_TIMEOUT`` budgets from
        per-host measurements the workstation-built spec cannot know.
        """
        config = self._host_config(verify_env={'REIFY_VERIFY_TEST_TIMEOUT': '90m'})

        effective_config, _ = await self._merge_verify_effective_config(
            config, {'REIFY_RELEASE_DELTA_SKIP': '1'},
        )

        # Asserted together so neither the 'ignore' nor the 'replace' spelling
        # can pass: both sides' keys must be present at once.
        assert effective_config.verify_env['REIFY_VERIFY_TEST_TIMEOUT'] == '90m', (
            'a host key absent from the spec must survive — the spec is dispatcher-shaped'
        )
        assert effective_config.verify_env['REIFY_RELEASE_DELTA_SKIP'] == '1', (
            "the dispatcher's verify_env must reach the reconstructed remote config "
            'even with zero module configs'
        )

    async def test_sccache_backend_env_reaches_zero_module_remote_config(self):
        """The host side is read through ``effective_verify_env``, the PROPERTY.

        Property and field differ exactly when the config was built directly
        rather than through ``load_config`` (which folds the one into the
        other): the shared sccache backend then lives ONLY in
        ``sccache.backend_env``.  Reading the field would drop it from the
        zero-module remote leg — the same κ fidelity hole
        ``build_merge_verify_spec`` reads the property to close on the
        producing side.
        """
        from orchestrator.config import SccacheConfig

        config = self._host_config(
            sccache=SccacheConfig(
                enabled=True, backend_env={'SCCACHE_REDIS': 'redis://backend:6380'},
            ),
        )
        assert 'SCCACHE_REDIS' not in config.verify_env  # the field read would miss it

        effective_config, _ = await self._merge_verify_effective_config(
            config, {'REIFY_RELEASE_DELTA_SKIP': '1'},
        )

        assert effective_config.verify_env['SCCACHE_REDIS'] == 'redis://backend:6380', (
            'the shared sccache backend must reach the remote leg; it is absent '
            'from both config.verify_env and spec.verify_env, so only the '
            'effective_verify_env property carries it'
        )
        assert effective_config.verify_env['REIFY_RELEASE_DELTA_SKIP'] == '1'

    async def test_caller_config_verify_env_is_not_mutated(self):
        """The merge must REBIND a fresh dict, never mutate in place.

        ``model_copy`` carries the field's VALUE over unchanged, so the copy's
        ``verify_env`` is initially the SAME dict object as the caller's — the
        identical hazard task 4536 records for the module registry. A
        ``config.verify_env.update(...)`` spelling would reach through that
        shared value and corrupt the CALLER's config, the object cli.py loaded
        from disk and may still use.
        """
        caller_config = self._host_config(verify_env={'REIFY_VERIFY_TEST_TIMEOUT': '90m'})
        original_dict = caller_config.verify_env
        original_items = dict(caller_config.verify_env)

        effective_config, _ = await self._merge_verify_effective_config(
            caller_config, {'REIFY_RELEASE_DELTA_SKIP': '1'},
        )

        assert caller_config.verify_env is original_dict, (
            "the caller's verify_env dict object must be untouched — the copy's "
            'field starts out as the SAME dict object, so an in-place mutation '
            "of the COPY reaches through and corrupts the caller's config"
        )
        assert caller_config.verify_env == original_items, (
            f"the caller's verify_env contents must be unchanged; got "
            f'{dict(caller_config.verify_env)!r}'
        )
        # Sanity: the copy really did receive the spec's key, so the identity
        # assertion above is not passing vacuously against a no-op.
        assert effective_config.verify_env['REIFY_RELEASE_DELTA_SKIP'] == '1'

    async def test_zero_module_and_per_module_resolve_the_same_verify_env(self):
        """Close the loop past the config field to the env actually handed to
        verify commands: both paths must resolve the IDENTICAL mapping.

        The defect was a config field populated with no consumer on one path, so
        a test that only inspects the field would repeat the defect's own blind
        spot. ``verify._resolve_verify_env`` is the sole builder of the executed
        env and the real consumer is an AsyncMock here, so this asserts the
        byte-identity claim rather than deriving it.
        """
        from orchestrator.verify import _resolve_verify_env

        host_env = {
            'REIFY_VERIFY_TEST_TIMEOUT': '90m',
            'REIFY_GATE_EXCLUDE_HEAVY': '0',
        }
        spec_env = {
            'REIFY_GATE_EXCLUDE_HEAVY': '1',
            'REIFY_RELEASE_DELTA_SKIP': '1',
        }

        zero_config, zero_modules = await self._merge_verify_effective_config(
            self._host_config(verify_env=dict(host_env)), spec_env,
        )
        assert zero_modules == []
        zero_env = _resolve_verify_env(zero_config, None, role='merge')

        per_config, per_modules = await self._merge_verify_effective_config(
            self._host_config(verify_env=dict(host_env)), spec_env,
            verify_commands=(VerifyCommand('src/a', test_command='true'),),
        )
        assert len(per_modules) == 1
        per_env = _resolve_verify_env(per_config, per_modules[0], role='merge')

        assert zero_env['REIFY_RELEASE_DELTA_SKIP'] == '1'
        assert zero_env['REIFY_GATE_EXCLUDE_HEAVY'] == '1', 'the spec wins the conflict'
        assert zero_env['REIFY_VERIFY_TEST_TIMEOUT'] == '90m', (
            'the host-only key survives into the executed env'
        )
        assert zero_env == per_env, (
            'the two paths must hand verify commands the IDENTICAL env — that '
            'uniformity is what the fix buys, and it is the per-module path '
            'that is unchanged'
        )

    # --- task 6127: the dispatcher's pre-provision command reaches the remote leg

    _PROVISION = 'uv sync --all-packages && npm ci --no-audit --no-fund --prefer-offline'

    @staticmethod
    def _preprovision_spec(verify_cold_preprovision_command):
        return MergeVerifySpec(
            verify_commands=(VerifyCommand('src/a', test_command='true'),),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('src/a/m.py',),
            verify_env={},
            cold_timeout_secs=60.0,
            verify_cold_preprovision_command=verify_cold_preprovision_command,
        )

    @staticmethod
    async def _effective_config(config, spec):
        from orchestrator.verify_runner import run_merge_verify_on_worktree

        run_scoped = AsyncMock(return_value=_make_pass_result())
        run_unscoped = AsyncMock(
            return_value=MagicMock(
                broken=False, timed_out=False,
                failing_subprojects=[], timed_out_subprojects=[],
            )
        )
        await run_merge_verify_on_worktree(
            MagicMock(), config, spec,
            run_scoped=run_scoped, run_unscoped=run_unscoped,
        )
        assert run_scoped.await_args is not None
        return run_scoped.await_args[0][1]

    async def test_spec_preprovision_command_overrides_empty_host_value(self):
        """The 6021 shape: the verifying host's own value is ''."""
        caller_config = self._host_config(verify_cold_preprovision_command='')

        effective_config = await self._effective_config(
            caller_config, self._preprovision_spec(self._PROVISION),
        )

        assert effective_config.verify_cold_preprovision_command == self._PROVISION
        assert caller_config.verify_cold_preprovision_command == ''

    async def test_empty_spec_preprovision_keeps_host_value(self):
        config = self._host_config(verify_cold_preprovision_command='HOST_PROVISION')

        effective_config = await self._effective_config(config, self._preprovision_spec(''))

        assert effective_config.verify_cold_preprovision_command == 'HOST_PROVISION'

    async def test_legacy_spec_without_preprovision_keeps_host_value(self):
        config = self._host_config(verify_cold_preprovision_command='HOST_PROVISION')
        legacy = self._preprovision_spec(self._PROVISION).to_dict()
        legacy.pop('verify_cold_preprovision_command')

        effective_config = await self._effective_config(
            config, MergeVerifySpec.from_dict(legacy),
        )

        assert effective_config.verify_cold_preprovision_command == 'HOST_PROVISION'

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
        config = _narrow_config()
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
        config = _narrow_config()

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
        import orchestrator.merge_lane.worker as mq_mod
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

        config = _narrow_config()
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
        fake_run = self._make_fake_run([])
        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        assert runner.name == 'laptop'

    async def test_isinstance_verify_runner_protocol(self):
        fake_run = self._make_fake_run([])
        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        assert isinstance(runner, VerifyRunner)

    async def test_health_true_when_ssh_rc_zero(self):
        fake_run = self._make_fake_run([(0, '', '')])
        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        result = await runner.health()
        assert result is True
        # health issues `ssh <_SSH_BASE_OPTS> <host> true` (task 2362: keepalive flags included)
        from orchestrator.verify_runner import _SSH_BASE_OPTS
        assert fake_run.calls == [
            ['ssh', *_SSH_BASE_OPTS, 'laptop.local', 'true']
        ]

    async def test_health_false_when_ssh_rc_nonzero(self):
        fake_run = self._make_fake_run([(1, '', 'Connection refused')])
        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='laptop',
            cwd='/repo',
            run=fake_run,
        )
        result = await runner.health()
        assert result is False

    async def test_health_false_when_run_raises(self):

        async def raising_run(argv, *, cwd=None):
            raise OSError("ssh not found")

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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
        parsed = gate_argv(remote_cmd)
        assert parsed[:3] == ['verify-merge', '--sha', 'abc123']
        # spec JSON survives as a single token
        spec_idx = parsed.index('--spec') + 1
        assert parsed[spec_idx] == spec_to_json(spec)
        # no --config when not set
        assert '--config' not in parsed

    async def test_ssh_argv_includes_config_path_when_set(self):
        """When config_path is set, ['--config', config_path] appears in the remote cmd."""

        spec = _make_spec()
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        runner, calls = self._make_runner_and_calls(expected, config_path='/etc/orch.yaml')
        await runner.run_merge_verify('abc123', spec)
        ssh_argv, _ = calls[1]
        parsed = gate_argv(ssh_argv[-1])  # last arg is the quoted remote command
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

    def _make_runner_and_calls(self, responses, *, raise_on=None):
        """Return (runner, calls_list) where calls_list records each run() argv.

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

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'req-id',
        )
        return runner, calls

    async def test_raises_runner_unavailable_on_push_failure(self):
        """git push rc!=0 → RunnerUnavailable; ssh is never called."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner, calls = self._make_runner_and_calls([(1, '', 'push error'), (0, '', '')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())
        # ssh must NOT have been attempted
        assert not any(a[0] == 'ssh' for a in calls)

    async def test_raises_runner_unavailable_on_ssh_nonzero(self):
        """ssh rc!=0 (e.g. 255 connection refused) → RunnerUnavailable.

        task-2362 / incident 5111: this is also the recovery-path counterpart
        of the ssh ServerAlive keepalive hardening — a keepalive-induced dead
        peer detection makes ssh exit non-zero exactly like the connection-
        refused case exercised here, which is what feeds the existing
        RunnerUnavailable → re-dispatch/local-fallback path. See
        TestRemoteRunnerSshKeepalive for the argv-construction side (that the
        keepalive flags are actually present on every ssh site).
        """
        from orchestrator.verify_runner import RunnerUnavailable
        runner, _ = self._make_runner_and_calls([(0, '', ''), (255, '', 'ssh: connect to host laptop.local port 22')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

    async def test_raises_runner_unavailable_on_empty_stdout(self):
        """ssh rc=0 but stdout is empty → RunnerUnavailable (unparseable)."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner, _ = self._make_runner_and_calls([(0, '', ''), (0, '', '')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

    async def test_raises_runner_unavailable_on_non_json_stdout(self):
        """ssh rc=0 but stdout is non-JSON → RunnerUnavailable."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner, _ = self._make_runner_and_calls([(0, '', ''), (0, 'not valid json!!!', '')])
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
        runner, _ = self._make_runner_and_calls([(0, '', ''), (0, '{"unexpected": 1}', '')])
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())
        # JSON list — also a TypeError because ** unpacking requires a mapping
        runner2, _ = self._make_runner_and_calls([(0, '', ''), (0, '[1, 2, 3]', '')])
        with pytest.raises(RunnerUnavailable):
            await runner2.run_merge_verify('abc123', _make_spec())

    async def test_raises_runner_unavailable_when_run_raises_oserror(self):
        """An OSError from the subprocess runner → RunnerUnavailable."""
        from orchestrator.verify_runner import RunnerUnavailable
        runner, _ = self._make_runner_and_calls([], raise_on={0: FileNotFoundError('git not found')})
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
        runner, _ = self._make_runner_and_calls([(0, '', ''), (0, result_to_json(timed_out_result), '')])
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
        runner, _ = self._make_runner_and_calls([(0, '', ''), (0, result_to_json(fail_result), '')])
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
        """Return (runner, calls_list) where calls_list records each run() argv.

        ``responses_by_argv_prefix`` maps an argv[0] to (rc, stdout, stderr).
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

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'cleanup-id',
        )
        return runner, calls

    async def test_delete_called_after_success(self):
        """After a successful run, the pushed ref is deleted via git push --delete."""
        runner, calls = self._make_tracking_runner(
            {'git': (0, '', ''), 'ssh': (0, result_to_json(_make_pass_result()), '')}
        )
        await runner.run_merge_verify('abc123', _make_spec())
        delete_calls = [c for c in calls if c[:2] == ['git', 'push'] and '--delete' in c]
        assert len(delete_calls) == 1
        assert 'refs/merge-verify/cleanup-id' in delete_calls[0]

    async def test_delete_called_after_ssh_failure(self):
        """When ssh fails (→ RunnerUnavailable), the ref is still deleted (cleanup in finally)."""
        from orchestrator.verify_runner import RunnerUnavailable

        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[:2] == ['git', 'push'] and '--delete' in argv:
                return (0, '', '')
            if argv[0] == 'git':
                return (0, '', '')  # push succeeds
            return (255, '', 'ssh: connect refused')  # ssh fails

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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
# task 4537 step-1: capture_env_fingerprint degrades on an ABSENT binary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCaptureEnvFingerprintMissingBinary:
    """An absent probe binary must degrade to rc=127, not raise OSError.

    ``asyncio.create_subprocess_exec`` raises ``FileNotFoundError`` when the
    binary does not exist, whereas an ssh remote shell returns rc=127 for the
    same condition.  Normalising the local side to 127 is what makes a tool
    that is absent on BOTH hosts compare EQUAL instead of exploding locally
    and reporting ``<unavailable rc=127>`` remotely.
    """

    def _make_run(self, *, missing=(), responses=None):
        """Async run callable that raises FileNotFoundError for `missing` argv[0]."""
        queue = list(responses or [])
        issued = []
        missing_set = set(missing)

        async def fake_run(argv, *, cwd=None):
            issued.append(argv)
            if argv[0] in missing_set:
                raise FileNotFoundError(
                    2, 'No such file or directory', argv[0],
                )
            if queue:
                return queue.pop(0)
            return (0, '', '')

        run: Any = fake_run
        run.issued = issued
        return run

    async def test_absent_extra_probe_records_rc_127(self):
        """An absent EXTRA probe binary yields '<unavailable rc=127>', not OSError."""
        from orchestrator.verify_runner import capture_env_fingerprint
        run = self._make_run(
            missing=['nosuchtool'],
            responses=[
                (0, 'rustc 1.80.0\n', ''),
                (0, 'cargo 1.80.0\n', ''),
                (0, 'stats\n', ''),
            ],
        )
        fp = await capture_env_fingerprint(
            run, extra_probe_specs=[('nosuchtool', ['nosuchtool', '--version'])],
        )
        assert fp.extra_probes['nosuchtool'] == '<unavailable rc=127>'

    async def test_absent_rustc_and_cargo_yield_empty_toolchain(self):
        """A rust-less host still produces an EnvFingerprint, with toolchain ''."""
        from orchestrator.verify_runner import EnvFingerprint, capture_env_fingerprint
        run = self._make_run(
            missing=['rustc', 'cargo'],
            responses=[(0, 'stats\n', '')],
        )
        fp = await capture_env_fingerprint(run)
        assert isinstance(fp, EnvFingerprint)
        assert fp.toolchain == ''

    async def test_absent_sccache_not_reachable(self):
        """An absent sccache binary yields sccache_reachable False, not OSError."""
        from orchestrator.verify_runner import capture_env_fingerprint
        run = self._make_run(
            missing=['sccache'],
            responses=[
                (0, 'rustc 1.80.0\n', ''),
                (0, 'cargo 1.80.0\n', ''),
            ],
        )
        fp = await capture_env_fingerprint(run)
        assert fp.sccache_reachable is False

    async def test_uniformly_absent_on_both_hosts_compares_equal(self):
        """Two rust-less hosts both missing the same probe compare FAITHFUL.

        This is the property the detector depends on: a tool absent on both
        sides is not drift.
        """
        from orchestrator.verify_runner import (
            capture_env_fingerprint,
            compare_env_fingerprints,
        )
        probes = [('pyright', ['npx', 'pyright', '--version'])]
        missing = ['rustc', 'cargo', 'sccache', 'npx']

        local_fp = await capture_env_fingerprint(
            self._make_run(missing=missing), extra_probe_specs=probes,
        )
        remote_fp = await capture_env_fingerprint(
            self._make_run(missing=missing), extra_probe_specs=probes,
        )

        verdict = compare_env_fingerprints(local_fp, remote_fp)
        assert verdict.is_faithful is True
        assert verdict.drift_dimensions == ()


@pytest.mark.asyncio
class TestCaptureEnvFingerprintIncludeToolchain:
    """The three hardcoded Rust probes are skippable when nothing compares them.

    On the remote side each probe is a SEPARATE ssh invocation (_SSH_BASE_OPTS
    sets up no ControlMaster/ControlPersist), so issuing rustc/cargo/sccache
    for a comparison scoped to extra_probes costs three extra TCP+auth
    handshakes per host, serially, over a possibly-WAN link.
    """

    def _make_run(self):
        issued = []

        async def fake_run(argv, *, cwd=None):
            issued.append(list(argv))
            return (0, f'{argv[0]} 1.0.0\n', '')

        run: Any = fake_run
        run.issued = issued
        return run

    async def test_default_still_issues_the_toolchain_probes(self):
        """Byte-identical for every existing caller — run_verdict_parity included."""
        from orchestrator.verify_runner import capture_env_fingerprint
        run = self._make_run()
        await capture_env_fingerprint(run, extra_probe_specs=[('uv', ['uv', '--version'])])
        assert run.issued == [
            ['rustc', '--version'],
            ['cargo', '--version'],
            ['sccache', '--show-stats'],
            ['uv', '--version'],
        ]

    async def test_opt_out_issues_only_the_declared_probes(self):
        from orchestrator.verify_runner import capture_env_fingerprint
        run = self._make_run()
        await capture_env_fingerprint(
            run,
            extra_probe_specs=[('uv', ['uv', '--version'])],
            include_toolchain=False,
        )
        assert run.issued == [['uv', '--version']]

    async def test_opt_out_yields_empty_toolchain_and_unreachable_sccache(self):
        """The skipped fields report their not-measured values, not stale ones."""
        from orchestrator.verify_runner import capture_env_fingerprint
        fp = await capture_env_fingerprint(self._make_run(), include_toolchain=False)
        assert fp.toolchain == ''
        assert fp.sccache_reachable is False

    async def test_opt_out_preserves_the_extra_probe_values(self):
        from orchestrator.verify_runner import capture_env_fingerprint
        fp = await capture_env_fingerprint(
            self._make_run(),
            extra_probe_specs=[('uv', ['uv', '--version'])],
            include_toolchain=False,
        )
        assert fp.extra_probes == {'uv': 'uv 1.0.0'}


@pytest.mark.asyncio
class TestCheckEnvSkipsUnreadToolchainProbes:
    """check_env derives include_toolchain from its `dimensions`, not a hardcode."""

    async def test_default_scope_probes_only_the_declared_tools(self):
        """The production path: 2 declared probes -> 2 commands per host, not 5."""
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, remote_fake = _make_env_drift_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)

        expected = [list(argv) for _, argv in _ENV_PROBE_SPECS]
        assert local_run.issued == expected
        assert remote_fake.env_probe_run().issued == expected

    async def test_a_toolchain_scope_still_probes_rustc(self):
        """Derived, not hardcoded off: asking for toolchain must MEASURE it.

        Hardcoding include_toolchain=False would silently make every host
        report toolchain='' and compare faithful — a false green for the one
        caller that explicitly asked about the Rust toolchain.
        """
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = _make_env_drift_pool(
            local_output={**_MATCHING_PROBE_OUTPUT, 'rustc --version': 'rustc 1.96.0\n'},
            remote_output=_MATCHING_PROBE_OUTPUT,
        )
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        result = await detector.check_env(
            _ENV_PROBE_SPECS, local_run=local_run, dimensions=('toolchain',),
        )
        assert ['rustc', '--version'] in local_run.issued
        assert result.verdict == DriftVerdict.DIVERGE

    async def test_sccache_scope_still_probes_sccache(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        await detector.check_env(
            _ENV_PROBE_SPECS, local_run=local_run, dimensions=('sccache_reachable',),
        )
        assert ['sccache', '--show-stats'] in local_run.issued


@pytest.mark.asyncio
class TestCaptureEnvFingerprintProbeTimeout:
    """A HUNG probe must not hold the merge-verify lease and both host slots.

    check_env runs inside _run_drift_check's merge_verify_lease block, after
    both allocator slots are taken, and _maybe_run_drift_check fires it with
    asyncio.create_task and no timeout of its own — so nothing above would ever
    reclaim those resources.  The local seam is a bare create_subprocess_exec +
    communicate(), and the committed probe set includes `npx pyright
    --version`, which on a cold npm cache reaches the npm registry.
    """

    def _hanging_run(self, *, hangs=()):
        """Run callable that never returns for argv[0] in `hangs`."""
        issued = []
        hang_set = set(hangs)

        async def fake_run(argv, *, cwd=None):
            issued.append(list(argv))
            if argv[0] in hang_set:
                await asyncio.sleep(60)
            return (0, f'{argv[0]} 1.0.0\n', '')

        run: Any = fake_run
        run.issued = issued
        return run

    async def test_hung_extra_probe_records_rc_124(self, monkeypatch):
        """A wedged probe degrades to a recorded sentinel, not an unbounded await."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_PROBE_TIMEOUT_SECS', 0.05)
        fp = await vr.capture_env_fingerprint(
            self._hanging_run(hangs=['npx']),
            extra_probe_specs=[('pyright', ['npx', 'pyright', '--version'])],
        )
        assert fp.extra_probes['pyright'] == '<unavailable rc=124>'

    async def test_hung_probe_does_not_block_the_remaining_probes(self, monkeypatch):
        """One wedged probe must not abandon the rest of the capture."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_PROBE_TIMEOUT_SECS', 0.05)
        fp = await vr.capture_env_fingerprint(
            self._hanging_run(hangs=['npx']),
            extra_probe_specs=[
                ('pyright', ['npx', 'pyright', '--version']),
                ('uv', ['uv', '--version']),
            ],
        )
        assert fp.extra_probes['pyright'] == '<unavailable rc=124>'
        assert fp.extra_probes['uv'] == 'uv 1.0.0'

    async def test_capture_returns_promptly_under_a_hung_probe(self, monkeypatch):
        """The whole point: the lease is released rather than parked forever."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_PROBE_TIMEOUT_SECS', 0.05)
        started = time.monotonic()
        await vr.capture_env_fingerprint(
            self._hanging_run(hangs=['npx', 'rustc', 'cargo', 'sccache']),
            extra_probe_specs=[('pyright', ['npx', 'pyright', '--version'])],
        )
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, (
            f'capture took {elapsed:.1f}s under a hung probe — the bound did '
            f'not apply, and in production this would hold the merge-verify '
            f'lease and both verify host slots for that whole time'
        )

    async def test_hung_toolchain_probe_does_not_raise(self, monkeypatch):
        """The bound covers the hardcoded probes too, not just the extra ones."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_PROBE_TIMEOUT_SECS', 0.05)
        fp = await vr.capture_env_fingerprint(
            self._hanging_run(hangs=['rustc', 'sccache']),
        )
        assert fp.toolchain == 'cargo 1.0.0'
        assert fp.sccache_reachable is False

    async def test_uniformly_hung_probe_compares_equal(self, monkeypatch):
        """A probe wedged on BOTH hosts is not drift — same rule as an absent one."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_PROBE_TIMEOUT_SECS', 0.05)
        probes = [('pyright', ['npx', 'pyright', '--version'])]
        local_fp = await vr.capture_env_fingerprint(
            self._hanging_run(hangs=['npx']), extra_probe_specs=probes,
        )
        remote_fp = await vr.capture_env_fingerprint(
            self._hanging_run(hangs=['npx']), extra_probe_specs=probes,
        )
        verdict = vr.compare_env_fingerprints(
            local_fp, remote_fp, dimensions=('extra_probes',),
        )
        assert verdict.is_faithful is True

    async def test_timeout_is_not_misreported_as_a_missing_binary(self, monkeypatch):
        """TimeoutError subclasses OSError — the except order must not swallow it.

        A hung probe and an absent one are different operator findings ("the
        tool is slow/unreachable" vs "the tool is not installed"), so they must
        not collapse onto the same rc.
        """
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_PROBE_TIMEOUT_SECS', 0.05)
        fp = await vr.capture_env_fingerprint(
            self._hanging_run(hangs=['npx']),
            extra_probe_specs=[('pyright', ['npx', 'pyright', '--version'])],
        )
        assert fp.extra_probes['pyright'] != '<unavailable rc=127>'


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

    # -- task 4537 step-3: scoping the comparison to chosen dimensions -------

    def test_scoped_to_extra_probes_ignores_toolchain_and_sccache(self):
        """dimensions=('extra_probes',) is FAITHFUL despite toolchain/sccache skew.

        The non-Rust-project property: this workstation has a stray
        /home/leo/.cargo/bin/rustc that a rust-less laptop does not, and that
        must not manufacture drift for a Python project.
        """
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(toolchain='rustc 1.96.0\ncargo 1.96.0', sccache_reachable=True)
        remote = self._fp(toolchain='', sccache_reachable=False)
        verdict = compare_env_fingerprints(local, remote, dimensions=('extra_probes',))
        assert verdict.is_faithful is True
        assert verdict.drift_dimensions == ()

    def test_same_pair_unscoped_still_reports_both_dimensions(self):
        """Omitting dimensions compares all four — unchanged for every caller."""
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(toolchain='rustc 1.96.0\ncargo 1.96.0', sccache_reachable=True)
        remote = self._fp(toolchain='', sccache_reachable=False)
        verdict = compare_env_fingerprints(local, remote)
        assert verdict.is_faithful is False
        assert 'toolchain' in verdict.drift_dimensions
        assert 'sccache_reachable' in verdict.drift_dimensions

    def test_scoped_still_detects_extra_probe_drift(self):
        """Scoping does not blind the detector to the dimension it DOES watch."""
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(
            toolchain='rustc 1.96.0', extra_probes={'pyright': '1.1.400'},
        )
        remote = self._fp(
            toolchain='', extra_probes={'pyright': '1.1.377'},
        )
        verdict = compare_env_fingerprints(local, remote, dimensions=('extra_probes',))
        assert verdict.is_faithful is False
        assert verdict.drift_dimensions == ('extra_probes',)

    def test_unknown_dimension_raises_value_error(self):
        """An unknown dimension name is a config error, not a silent no-op."""
        from orchestrator.verify_runner import compare_env_fingerprints
        fp = self._fp()
        with pytest.raises(ValueError) as excinfo:
            compare_env_fingerprints(fp, fp, dimensions=('extra_probez',))
        assert 'extra_probez' in str(excinfo.value)

    def test_empty_dimensions_raises_value_error(self):
        """`dimensions=()` is the SAME class of error as a misspelled name.

        Without this guard an empty selection compares nothing and returns
        is_faithful=True — a permanently-green detective control that affirms a
        parity it never measured.  Two fingerprints differing in EVERY field
        are used so a pass here could only mean "compared nothing".
        """
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(
            toolchain='rustc 1.96.0',
            verify_env={'A': '1'},
            sccache_reachable=True,
            extra_probes={'pyright': '1.1.400'},
        )
        remote = self._fp(
            toolchain='',
            verify_env={'A': '2'},
            sccache_reachable=False,
            extra_probes={'pyright': '1.1.377'},
        )
        with pytest.raises(ValueError) as excinfo:
            compare_env_fingerprints(local, remote, dimensions=())
        assert 'at least one' in str(excinfo.value)

    def test_none_is_still_how_a_caller_asks_for_every_dimension(self):
        """Rejecting () must not make the all-dimensions default unreachable."""
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(toolchain='a')
        remote = self._fp(toolchain='b')
        verdict = compare_env_fingerprints(local, remote, dimensions=None)
        assert verdict.drift_dimensions == ('toolchain',)

    def test_scoped_multi_dimension_preserves_canonical_order(self):
        """drift_dimensions keeps the canonical field order, not the caller's."""
        from orchestrator.verify_runner import compare_env_fingerprints
        local = self._fp(toolchain='a', extra_probes={'k': 'v1'})
        remote = self._fp(toolchain='b', extra_probes={'k': 'v2'})
        verdict = compare_env_fingerprints(
            local, remote, dimensions=('extra_probes', 'toolchain'),
        )
        assert verdict.drift_dimensions == ('toolchain', 'extra_probes')


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


def _make_drift_pool(local_result=None, remote_result=None, *, on_unavailable=None):
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
    pool = VerifyRunnerPool([local_fake, remote_fake], on_unavailable=on_unavailable)
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

    # -- task 4188: verify categories surfaced on the AGREE result AND event --

    @pytest.mark.parametrize(
        'local_result, remote_result, expected_local, expected_remote',
        [
            # A suppression-implicated LOCAL arm is named on both artifacts.
            (_make_pass_result(category='merge_flake_suppressed'),
             _make_pass_result(), 'merge_flake_suppressed', ''),
            # The mirror case -- the REMOTE arm's category is threaded too, in
            # its own slot, not just whichever arm happens to be suppressed.
            (_make_pass_result(),
             _make_pass_result(category='merge_flake_suppressed'), '', 'merge_flake_suppressed'),
            # Always-populated, not only-on-divergence.  A NON-EMPTY category on
            # BOTH arms, so this cannot pass vacuously against the '' default.
            (_make_fail_result(category='test_failure'),
             _make_fail_result(category='test_failure'), 'test_failure', 'test_failure'),
            # Uniform shape: a clean, category-less pass still carries both keys.
            (_make_pass_result(), _make_pass_result(), '', ''),
        ],
        ids=['local-suppressed', 'remote-suppressed', 'both-test-failure', 'clean-pass'],
    )
    async def test_agree_carries_both_categories_on_result_and_event(
        self, local_result, remote_result, expected_local, expected_remote
    ):
        """Both arms' categories reach the AGREE result AND the parity payload.

        The two ``*_category`` keys are emitted UNCONDITIONALLY, so a consumer
        reads the same two keys on every drift parity event and never has to
        distinguish an absent key from a clean, sentinel-free result.
        """
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = _make_drift_pool(local_result=local_result, remote_result=remote_result)
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        result = await detector.check('sha1', _make_spec())

        assert result.verdict == DriftVerdict.AGREE
        assert result.local_category == expected_local
        assert result.remote_category == expected_remote

        data = event_store.emit.call_args[1]['data']
        assert 'local_category' in data
        assert 'remote_category' in data
        assert data['local_category'] == expected_local
        assert data['remote_category'] == expected_remote


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

    # -- task 4188: verify categories surfaced on the DIVERGE artifacts --

    def _diverge_queue(self):
        queue = MagicMock()
        queue.has_open_l1 = MagicMock(return_value=False)
        queue.make_id = MagicMock(return_value='esc-__drift__-1')
        return queue

    async def test_diverge_result_carries_both_categories(self):
        """The DIVERGE return carries both arms' categories, not just the AGREE one."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(category='test_failure'),
            remote_result=_make_pass_result(category='merge_flake_suppressed'),
        )
        detector = DriftDetector(pool, escalation_queue=self._diverge_queue())
        result = await detector.check('divergesha', _make_spec())
        assert result.verdict == DriftVerdict.DIVERGE
        assert result.local_category == 'test_failure'
        assert result.remote_category == 'merge_flake_suppressed'

    async def test_diverge_escalation_detail_names_suppressed_arm(self):
        """The suppressed arm is named in the artifact an operator actually rules on.

        Asserts the STRUCTURED, !r-quoted spelling rather than the bare word, so
        it cannot be satisfied vacuously by the operator footnote that also names
        the category.  Also guards that the rewrite stayed ADDITIVE.
        """
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(category='test_failure'),
            remote_result=_make_pass_result(category='merge_flake_suppressed'),
        )
        escalation_queue = self._diverge_queue()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('mydivergesha', _make_spec())
        esc = escalation_queue.submit.call_args[0][0]
        assert "remote_category='merge_flake_suppressed'" in esc.detail
        # Regression guard: the pre-existing detail content survives.
        assert "merge_sha='mydivergesha'" in esc.detail
        assert "local_runner='local'" in esc.detail
        assert "remote_runner='laptop'" in esc.detail
        # The operator footnote fires when an arm IS suppressed.  Pinned by the
        # stable function identifier it names, not by its prose wording.
        assert 'apply_merge_flake_suppression' in esc.detail

    async def test_diverge_escalation_detail_names_local_category(self):
        """The mirror case: both arms are threaded, into the right slots."""
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_pass_result(category='merge_flake_suppressed'),
            remote_result=_make_fail_result(category='test_failure'),
        )
        escalation_queue = self._diverge_queue()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('divergesha', _make_spec())
        esc = escalation_queue.submit.call_args[0][0]
        assert "local_category='merge_flake_suppressed'" in esc.detail
        assert "remote_category='test_failure'" in esc.detail

    async def test_diverge_escalation_detail_carries_categories_when_neither_suppressed(self):
        """Always-populated at the escalation artifact, including the '' arm."""
        from orchestrator.verify_runner import DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(category='test_failure'),
            remote_result=_make_pass_result(),
        )
        escalation_queue = self._diverge_queue()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check('divergesha', _make_spec())
        esc = escalation_queue.submit.call_args[0][0]
        assert "local_category='test_failure'" in esc.detail
        assert "remote_category=''" in esc.detail
        # ...and the suppression footnote is omitted entirely, so it never
        # dilutes the artifact with guidance irrelevant to this divergence.
        assert 'apply_merge_flake_suppression' not in esc.detail


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

    # -- task 4188: categories stay empty when nothing was compared --

    async def test_inconclusive_remote_unavailable_leaves_categories_empty(self):
        """Remote transport failure → categories stay ''.

        Mirrors how ``local_passed`` stays None on this path even though the
        local arm genuinely produced a result — ``verdict`` is the disambiguator.
        """
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, RunnerUnavailable
        pool, _, remote_fake = _make_drift_pool(
            local_result=_make_pass_result(category='merge_flake_suppressed'),
        )
        remote_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('host down'))
        detector = DriftDetector(pool)
        result = await detector.check('sha1', _make_spec())
        assert result.verdict == DriftVerdict.INCONCLUSIVE
        assert result.local_category == ''
        assert result.remote_category == ''


# ---------------------------------------------------------------------------
# task 6648: DriftDetector runs its two legs concurrently
# ---------------------------------------------------------------------------

_LEG_WAIT_SECS = 5


def _gated_verify(gate, result, *, on_finish=None):
    """A run_merge_verify double that parks until *gate* is set, then returns *result*."""
    async def _body(*_args, **_kwargs):
        await gate.wait()
        if on_finish is not None:
            on_finish()
        return result
    return AsyncMock(side_effect=_body)


class _LegDoneRecorder:
    """An ``on_leg_done`` sink: logs ``('done', runner.name)`` and sets that leg's Event."""

    def __init__(self, log=None):
        self.log = [] if log is None else log
        self.ended = {'local': asyncio.Event(), 'laptop': asyncio.Event()}

    async def __call__(self, runner):
        self.log.append(('done', runner.name))
        self.ended[runner.name].set()

    @property
    def done(self):
        return [entry[1] for entry in self.log if entry[0] == 'done']


@pytest.mark.asyncio
class TestDriftDetectorConcurrentLegs:
    """DriftDetector.check(): both legs run at once; each leg's end is reported as it happens."""

    async def test_remote_leg_starts_while_local_leg_is_still_running(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_fake, remote_fake = _make_drift_pool()
        local_gate = asyncio.Event()
        remote_started = asyncio.Event()

        async def _remote(*_args, **_kwargs):
            remote_started.set()
            return _make_pass_result()

        local_fake.run_merge_verify = _gated_verify(local_gate, _make_pass_result())
        remote_fake.run_merge_verify = AsyncMock(side_effect=_remote)
        check = asyncio.create_task(DriftDetector(pool).check('sha1', _make_spec()))
        try:
            await wait_responsive(
                remote_started.wait(), timeout=_LEG_WAIT_SECS, label='remote leg started',
            )
        finally:
            local_gate.set()
            result = await asyncio.wait_for(check, _LEG_WAIT_SECS)
        assert result.verdict == DriftVerdict.AGREE

    async def test_on_leg_done_reports_the_remote_leg_before_the_local_leg_completes(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_fake, _ = _make_drift_pool()
        local_gate = asyncio.Event()
        local_fake.run_merge_verify = _gated_verify(local_gate, _make_pass_result())
        recorder = _LegDoneRecorder()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=MagicMock())
        check = asyncio.create_task(
            detector.check('sha1', _make_spec(), on_leg_done=recorder)
        )
        try:
            await wait_responsive(
                recorder.ended['laptop'].wait(), timeout=_LEG_WAIT_SECS,
                label='remote leg reported done',
            )
            assert recorder.done == ['laptop']
        finally:
            local_gate.set()
            result = await asyncio.wait_for(check, _LEG_WAIT_SECS)
        assert recorder.done == ['laptop', 'local']
        assert result.verdict == DriftVerdict.AGREE
        event_store.emit.assert_called_once()
        (event_type,), kwargs = event_store.emit.call_args
        assert event_type == EventType.verdict_parity_ok
        assert set(kwargs['data']) == {
            'merge_sha', 'local_runner', 'remote_runner', 'passed',
            'local_category', 'remote_category',
        }

    async def test_unavailable_remote_is_benched_before_its_slot_is_reported_done(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, RunnerUnavailable
        log: list[tuple[str, ...]] = []
        pool, local_fake, remote_fake = _make_drift_pool(
            on_unavailable=lambda name, _exc: log.append(('bench', name)),
        )
        local_gate = asyncio.Event()
        local_fake.run_merge_verify = _gated_verify(
            local_gate, _make_pass_result(), on_finish=lambda: log.append(('local finished',)),
        )
        remote_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('down'))
        recorder = _LegDoneRecorder(log)
        event_store, escalation_queue = MagicMock(), MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=escalation_queue)
        check = asyncio.create_task(
            detector.check('sha1', _make_spec(), on_leg_done=recorder)
        )
        try:
            await wait_responsive(
                recorder.ended['laptop'].wait(), timeout=_LEG_WAIT_SECS,
                label='remote leg reported done',
            )
            assert ('local finished',) not in log
        finally:
            local_gate.set()
            result = await asyncio.wait_for(check, _LEG_WAIT_SECS)
        assert log.index(('bench', 'laptop')) < log.index(('done', 'laptop'))
        assert log.index(('local finished',)) < log.index(('done', 'local'))
        assert result.verdict == DriftVerdict.INCONCLUSIVE
        event_store.emit.assert_not_called()
        escalation_queue.submit.assert_not_called()

    async def test_unavailable_local_still_lets_the_remote_leg_finish_and_benches_nothing(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, RunnerUnavailable
        benched: list[str] = []
        pool, local_fake, remote_fake = _make_drift_pool(
            on_unavailable=lambda name, _exc: benched.append(name),
        )
        remote_gate = asyncio.Event()
        local_fake.run_merge_verify = AsyncMock(side_effect=RunnerUnavailable('local down'))
        remote_fake.run_merge_verify = _gated_verify(remote_gate, _make_pass_result())
        recorder = _LegDoneRecorder()
        check = asyncio.create_task(
            DriftDetector(pool).check('sha1', _make_spec(), on_leg_done=recorder)
        )
        try:
            await wait_responsive(
                recorder.ended['local'].wait(), timeout=_LEG_WAIT_SECS,
                label='local leg reported done',
            )
        finally:
            remote_gate.set()
            result = await asyncio.wait_for(check, _LEG_WAIT_SECS)
        remote_fake.run_merge_verify.assert_awaited_once()
        assert benched == []
        assert sorted(recorder.done) == ['laptop', 'local']
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_unexpected_leg_exception_is_raised_after_both_legs_finish(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_fake, remote_fake = _make_drift_pool()
        remote_gate = asyncio.Event()
        local_fake.run_merge_verify = AsyncMock(side_effect=ValueError('boom'))
        remote_fake.run_merge_verify = _gated_verify(remote_gate, _make_pass_result())
        recorder = _LegDoneRecorder()
        check = asyncio.create_task(
            DriftDetector(pool).check('sha1', _make_spec(), on_leg_done=recorder)
        )
        try:
            await wait_responsive(
                recorder.ended['local'].wait(), timeout=_LEG_WAIT_SECS,
                label='local leg reported done',
            )
        finally:
            remote_gate.set()
        with pytest.raises(ValueError, match='boom'):
            await asyncio.wait_for(check, _LEG_WAIT_SECS)
        assert sorted(recorder.done) == ['laptop', 'local']
        remote_fake.run_merge_verify.assert_awaited_once()


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
# task 4537 step-9/11/13: DriftDetector.check_env — host-environment divergence
# ---------------------------------------------------------------------------

_ENV_PROBE_SPECS = [
    ('pyright', ['npx', 'pyright', '--version']),
    ('uv', ['uv', '--version']),
]


def _make_probe_run(outputs, *, raises=None):
    """Return an async (argv, *, cwd=None) run callable driven by a cmdline→stdout map.

    Keys are the space-joined argv (so ['npx','pyright','--version'] resolves
    against 'npx pyright --version'); an unmapped command returns rc=127, the
    same 'command not found' rc capture_env_fingerprint normalises ENOENT to.
    rustc/cargo/sccache raise FileNotFoundError, standing in for a host that
    has no Rust toolchain at all.
    """
    issued = []

    async def run(argv, *, cwd=None):
        issued.append(list(argv))
        if raises is not None:
            raise raises
        head = argv[0]
        joined = ' '.join(argv)
        for probe_argv, stdout in outputs.items():
            if joined == probe_argv:
                return (0, stdout, '')
        if head in ('rustc', 'cargo', 'sccache'):
            raise FileNotFoundError(2, 'No such file or directory', head)
        return (127, '', f'{head}: command not found')

    recording: Any = run
    recording.issued = issued
    return recording


_MATCHING_PROBE_OUTPUT = {
    'npx pyright --version': 'pyright 1.1.377\n',
    'uv --version': 'uv 0.11.6\n',
}


def _make_env_drift_pool(*, local_output=None, remote_output=None, remote_raises=None):
    """Pool with a local runner + a remote exposing env_probe_run()."""
    from orchestrator.verify_runner import VerifyRunnerPool

    local_fake = MagicMock(spec=VerifyRunner)
    local_fake.name = 'local'
    local_fake.is_local = True
    local_fake.run_merge_verify = AsyncMock(return_value=_make_pass_result())

    remote_run = _make_probe_run(
        remote_output if remote_output is not None else _MATCHING_PROBE_OUTPUT,
        raises=remote_raises,
    )

    class _FakeRemote:
        name: str = 'laptop'
        is_local: ClassVar[bool] = False

        def __init__(self):
            self.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        async def health(self) -> bool:
            # Present to satisfy the VerifyRunner Protocol (pyright over tests/
            # is clause 2 of the merge-verify type gate).  Never consulted on
            # the check_env path — HostAllocator.acquire_remote does not probe
            # health, and check_env only calls env_probe_run.
            return True

        def env_probe_run(self):
            return remote_run

    remote_fake = _FakeRemote()

    local_run = _make_probe_run(
        local_output if local_output is not None else _MATCHING_PROBE_OUTPUT
    )
    pool = VerifyRunnerPool([local_fake, remote_fake])
    return pool, local_run, remote_fake


@pytest.mark.asyncio
class TestDriftDetectorCheckEnvFaithful:
    """check_env(): identical probe output on both hosts → AGREE + parity event."""

    async def test_matching_probes_return_agree_and_faithful(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = _make_env_drift_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=MagicMock())
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.AGREE
        assert result.is_faithful is True
        assert result.drift_dimensions == ()

    async def test_matching_probes_emit_exactly_one_parity_event(self):
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, task_id='t')
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert event_store.emit.call_count == 1
        assert event_store.emit.call_args[0][0] == EventType.verify_env_parity_ok

    async def test_parity_event_carries_runner_names_and_probe_keys(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        data = event_store.emit.call_args[1]['data']
        assert data['local_runner'] == 'local'
        assert data['remote_runner'] == 'laptop'
        assert sorted(data['probe_keys']) == ['pyright', 'uv']
        assert tuple(data['dimensions']) == ('extra_probes',)

    async def test_matching_probes_emit_no_divergence_event(self):
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        emitted = [c[0][0] for c in event_store.emit.call_args_list]
        assert EventType.verify_env_divergence not in emitted

    async def test_matching_probes_submit_no_escalation(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool()
        escalation_queue = MagicMock()
        detector = DriftDetector(pool, escalation_queue=escalation_queue)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        escalation_queue.submit.assert_not_called()

    async def test_matching_probes_do_not_quarantine_remote(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool()
        detector = DriftDetector(pool)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert pool.is_quarantined('laptop') is False
        assert pool.eligible_remote() is not None

    async def test_rust_toolchain_absent_on_both_hosts_is_not_drift(self):
        """Neither fake host has rustc/cargo/sccache; that must read as parity."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = _make_env_drift_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=MagicMock())
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.AGREE


# ---------------------------------------------------------------------------
# task 4537 step-11: DriftDetector.check_env — DIVERGENCE arm
# ---------------------------------------------------------------------------

# Same probe set, but the remote reports the uv version measured on leo-laptop
# on 2026-08-20 (0.10.6) against the workstation's 0.11.6.  pyright matches, so
# a detector that merely noticed "some probe output differs" without naming the
# offender would still be useless to an operator.
_PERTURBED_PROBE_OUTPUT = {
    'npx pyright --version': 'pyright 1.1.377\n',
    'uv --version': 'uv 0.10.6\n',
}


class _FakeEscQueue:
    """EscalationQueue stand-in with FAITHFUL has_open_l1 semantics.

    Mirrors escalation.queue::EscalationQueue.has_open_l1's
    ``(task_id, *, category=None)`` filter exactly: category=None counts ANY
    pending L1 on that task_id, a category counts only matching ones.  That
    fidelity is what makes the namespace-isolation test below a real
    behavioural assertion rather than a tautology over a MagicMock.
    """

    def __init__(self):
        self.submitted = []
        self._seq = 0

    def make_id(self, task_id):
        self._seq += 1
        return f'esc-{task_id}-{self._seq}'

    def submit(self, esc):
        self.submitted.append(esc)

    def has_open_l1(self, task_id, *, category=None):
        return any(
            e.task_id == task_id
            and e.level == 1
            and e.status == 'pending'
            and (category is None or e.category == category)
            for e in self.submitted
        )


def _diverging_events(event_store, event_type):
    """Every emit() call of *event_type*, as a list of its data payloads."""
    return [
        c[1]['data']
        for c in event_store.emit.call_args_list
        if c[0][0] == event_type
    ]


@pytest.mark.asyncio
class TestDriftDetectorCheckEnvDivergence:
    """check_env(): one perturbed probe → DIVERGE + named event + one deduped L1."""

    def _diverged(self):
        return _make_env_drift_pool(remote_output=_PERTURBED_PROBE_OUTPUT)

    async def test_perturbed_probe_returns_diverge(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = self._diverged()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.DIVERGE
        assert result.is_faithful is False
        assert result.drift_dimensions == ('extra_probes',)

    async def test_divergence_event_emitted_exactly_once(self):
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = self._diverged()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=_FakeEscQueue())
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert len(_diverging_events(event_store, EventType.verify_env_divergence)) == 1
        assert _diverging_events(event_store, EventType.verify_env_parity_ok) == []

    async def test_divergence_event_names_the_dimension_probe_and_both_values(self):
        """The whole point: the event must be readable without re-probing."""
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = self._diverged()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=_FakeEscQueue())
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        data = _diverging_events(event_store, EventType.verify_env_divergence)[0]
        assert tuple(data['drift_dimensions']) == ('extra_probes',)
        assert data['local_runner'] == 'local'
        assert data['remote_runner'] == 'laptop'
        assert data['diverging_probes'] == {
            'uv': {'local': 'uv 0.11.6', 'remote': 'uv 0.10.6'}
        }

    async def test_matching_probe_is_not_listed_as_diverging(self):
        """pyright agrees; only uv skews.  Naming both would be noise."""
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = self._diverged()
        event_store = MagicMock()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=_FakeEscQueue())
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        data = _diverging_events(event_store, EventType.verify_env_divergence)[0]
        assert 'pyright' not in data['diverging_probes']

    async def test_divergence_submits_exactly_one_escalation(self):
        from orchestrator.verify_runner import _ENV_DRIFT_SENTINEL, DriftDetector
        pool, local_run, _ = self._diverged()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=queue)
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert len(queue.submitted) == 1
        esc = queue.submitted[0]
        assert esc.task_id == _ENV_DRIFT_SENTINEL
        assert esc.level == 1
        assert esc.category == 'verify_env_divergence'
        assert esc.severity == 'blocking'
        assert result.escalated is True

    async def test_escalation_text_names_dimension_probe_and_both_values(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = self._diverged()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=queue)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        esc = queue.submitted[0]
        assert 'uv' in esc.summary
        blob = esc.summary + esc.detail
        assert 'extra_probes' in blob
        assert 'uv 0.11.6' in blob
        assert 'uv 0.10.6' in blob

    async def test_second_check_dedups_to_one_escalation(self):
        """Two consecutive diverging checks file ONE escalation, not two."""
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = self._diverged()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=queue)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        second = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert len(queue.submitted) == 1
        assert second.escalated is False

    async def test_dedup_still_emits_the_divergence_event(self):
        """Dedup suppresses the ESCALATION, never the telemetry."""
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = self._diverged()
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=queue)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        second = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert len(_diverging_events(event_store, EventType.verify_env_divergence)) == 2
        assert second.verdict == DriftVerdict.DIVERGE

    async def test_divergence_does_not_quarantine_the_remote(self):
        """FAIL-SAFE, not fail-closed: env drift alarms, it never benches a host.

        Asserted as an explicit negative because auto-benching a verify host
        over a cosmetic probe difference is the failure mode this design
        deliberately refuses.
        """
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = self._diverged()
        pool.quarantine = MagicMock(side_effect=AssertionError('check_env must not quarantine'))
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert pool.is_quarantined('laptop') is False
        assert pool.eligible_remote() is not None

    async def test_open_verdict_drift_l1_does_not_suppress_the_env_escalation(self):
        """Namespace isolation, direction 1.

        The verdict path dedups on has_open_l1('__drift__') with NO category
        filter, so a shared sentinel would let either alarm silence the other.
        """
        from escalation.models import Escalation

        from orchestrator.verify_runner import _DRIFT_SENTINEL, DriftDetector
        pool, local_run, _ = self._diverged()
        queue = _FakeEscQueue()
        queue.submitted.append(Escalation(
            id='esc-drift-pre', task_id=_DRIFT_SENTINEL, agent_role='orchestrator-drift-detector',
            severity='blocking', category='verify_drift_divergence', summary='pre-existing verdict drift',
            level=1,
        ))
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=queue)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert len(queue.submitted) == 2

    async def test_open_env_drift_l1_does_not_suppress_the_verdict_escalation(self):
        """Namespace isolation, direction 2 — the regression that matters most.

        An open env-drift L1 must not silence a genuine verdict divergence,
        which is the alarm that stops unverified code landing on main.
        """
        from escalation.models import Escalation

        from orchestrator.verify_runner import _DRIFT_SENTINEL, _ENV_DRIFT_SENTINEL, DriftDetector
        pool, _, _ = _make_drift_pool(
            local_result=_make_fail_result(), remote_result=_make_pass_result()
        )
        queue = _FakeEscQueue()
        queue.submitted.append(Escalation(
            id='esc-env-pre', task_id=_ENV_DRIFT_SENTINEL, agent_role='orchestrator-env-drift-detector',
            severity='blocking', category='verify_env_divergence', summary='pre-existing env drift',
            level=1,
        ))
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=queue)
        await detector.check('sha1', _make_spec())
        assert [e.task_id for e in queue.submitted] == [_ENV_DRIFT_SENTINEL, _DRIFT_SENTINEL]


@pytest.mark.asyncio
class TestDriftDetectorCheckEnvNonProbeDimension:
    """A non-default `dimensions` must still produce a LEGIBLE escalation.

    The production path only ever compares ('extra_probes',), but dimensions is
    a public parameter.  Scoped to a dimension with no per-probe detail, the
    diverging-probe map is empty — and an escalation summary that trails off
    after the colon tells an operator nothing about a blocking L1.
    """

    def _toolchain_skewed_pool(self):
        # Only the LOCAL host has rustc, so toolchain differs while every
        # declared probe agrees.
        return _make_env_drift_pool(
            local_output={**_MATCHING_PROBE_OUTPUT, 'rustc --version': 'rustc 1.96.0\n'},
            remote_output=_MATCHING_PROBE_OUTPUT,
        )

    async def test_toolchain_only_scope_diverges(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = self._toolchain_skewed_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        result = await detector.check_env(
            _ENV_PROBE_SPECS, local_run=local_run, dimensions=('toolchain',),
        )
        assert result.verdict == DriftVerdict.DIVERGE
        assert result.drift_dimensions == ('toolchain',)

    async def test_escalation_text_stays_legible_without_probe_detail(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = self._toolchain_skewed_pool()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=queue)
        await detector.check_env(
            _ENV_PROBE_SPECS, local_run=local_run, dimensions=('toolchain',),
        )
        esc = queue.submitted[0]
        assert 'toolchain' in esc.summary
        # No dangling "in toolchain: " with nothing after it.
        assert not esc.summary.rstrip().endswith(':')
        assert esc.summary.split(':', 1)[1].strip()
        assert 'toolchain' in esc.detail

    async def test_default_scope_is_unaffected_by_the_fallback(self):
        """The extra_probes path still names the probe, not the fallback text."""
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool(remote_output=_PERTURBED_PROBE_OUTPUT)
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=queue)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert queue.submitted[0].summary.endswith('uv')


@pytest.mark.asyncio
class TestEnvProbeSshTimeout:
    """A remote probe that hangs after connecting must read as UNREACHABLE.

    ssh's ConnectTimeout=10 (_SSH_BASE_OPTS) bounds only the HANDSHAKE.  A
    connection that establishes and then wedges — a loaded laptop, a stalled
    npx fetching from the npm registry — is bounded by nothing else, and would
    hold the merge-verify lease and both verify host slots for as long as it
    runs.
    """

    def _hanging_runner(self):
        async def fake_run(argv, *, cwd=None):
            await asyncio.sleep(60)
            return (0, '', '')

        return gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
        )

    async def test_hung_ssh_probe_raises_runner_unavailable(self, monkeypatch):
        """Transport failure, NOT a recorded rc: a wedged link is not a finding."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_SSH_PROBE_TIMEOUT_SECS', 0.05)
        adapter = self._hanging_runner().env_probe_run()
        with pytest.raises(vr.RunnerUnavailable) as excinfo:
            await adapter(['uv', '--version'])
        assert 'laptop.local' in str(excinfo.value)

    async def test_hung_remote_probe_is_inconclusive_not_divergence(self, monkeypatch):
        """END TO END: a wedged remote must not file a false env-divergence L1.

        This is the discriminating arm.  Had the ssh hang been normalised to a
        recorded rc=124 like the local side, every declared probe would compare
        a sentinel against a real local version — total apparent env skew — and
        the resulting L1 would dedup on (_ENV_DRIFT_SENTINEL, category) and
        suppress the genuine alarm until a human resolved it.
        """
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_SSH_PROBE_TIMEOUT_SECS', 0.05)

        remote_runner = self._hanging_runner()
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)

        pool = vr.VerifyRunnerPool([local_fake, remote_runner])
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = vr.DriftDetector(
            pool, event_store=event_store, escalation_queue=queue,
        )
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)

        assert result.verdict == vr.DriftVerdict.INCONCLUSIVE
        event_store.emit.assert_not_called()
        assert queue.submitted == []

    async def test_hung_remote_probe_leaves_the_local_host_untouched(self, monkeypatch):
        """check_env captures the remote FIRST, so a wedged link costs nothing local."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_SSH_PROBE_TIMEOUT_SECS', 0.05)

        remote_runner = self._hanging_runner()
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)

        pool = vr.VerifyRunnerPool([local_fake, remote_runner])
        detector = vr.DriftDetector(
            pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue(),
        )
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert local_run.issued == []

    async def test_check_env_returns_promptly_under_a_hung_remote(self, monkeypatch):
        """The lease and both host slots are released, not parked."""
        from orchestrator import verify_runner as vr
        monkeypatch.setattr(vr, '_SSH_PROBE_TIMEOUT_SECS', 0.05)

        remote_runner = self._hanging_runner()
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True

        pool = vr.VerifyRunnerPool([local_fake, remote_runner])
        detector = vr.DriftDetector(pool, event_store=MagicMock())
        started = time.monotonic()
        await detector.check_env(
            _ENV_PROBE_SPECS, local_run=_make_probe_run(_MATCHING_PROBE_OUTPUT),
        )
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, (
            f'check_env took {elapsed:.1f}s against a wedged remote — in '
            f'production that is the merge-verify lease and both verify host '
            f'slots held for the same duration'
        )


@pytest.mark.asyncio
class TestDriftDetectorCheckEnvBadDimensions:
    """A bad `dimensions` selection must be LOUD, never a quiet false green.

    `dimensions` is a public parameter of check_env.  An empty selection would
    take the faithful branch and emit a verify_env_parity_ok event affirming
    that two hosts match — without having compared a single field.  That is
    worse than no detector at all, so it raises instead.
    """

    async def test_empty_dimensions_raises_instead_of_reporting_parity(self):
        from orchestrator.verify_runner import DriftDetector
        # Hosts that genuinely DIVERGE, so a "faithful" answer could only come
        # from having compared nothing.
        pool, local_run, _ = _make_env_drift_pool(remote_output=_PERTURBED_PROBE_OUTPUT)
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=queue)

        with pytest.raises(ValueError):
            await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run, dimensions=())

        event_store.emit.assert_not_called()
        assert queue.submitted == []

    async def test_unknown_dimension_raises_at_check_env_level(self):
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, _ = _make_env_drift_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        with pytest.raises(ValueError) as excinfo:
            await detector.check_env(
                _ENV_PROBE_SPECS, local_run=local_run, dimensions=('extra_probez',),
            )
        assert 'extra_probez' in str(excinfo.value)

    async def test_bad_dimensions_rejected_before_any_host_is_probed(self):
        """The rejection costs no ssh round trips — validated before capture."""
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, remote_fake = _make_env_drift_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        with pytest.raises(ValueError):
            await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run, dimensions=())
        assert local_run.issued == []
        assert remote_fake.env_probe_run().issued == []


# ---------------------------------------------------------------------------
# task 4537 step-13: DriftDetector.check_env — INCONCLUSIVE guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDriftDetectorCheckEnvInconclusive:
    """check_env(): infrastructure conditions degrade quietly — never an alarm.

    Every case here is a reason the check could not be TAKEN.  Reporting any of
    them as divergence would train operators to ignore the alarm, which is the
    only thing that makes a detective control worthless.
    """

    async def test_no_probes_configured_is_inconclusive(self):
        """The shipped default (verify_env_probes: []) leaves the detector off."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = _make_env_drift_pool()
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=queue)
        result = await detector.check_env([], local_run=local_run)
        assert result.verdict == DriftVerdict.INCONCLUSIVE
        assert result.drift_dimensions == ()
        event_store.emit.assert_not_called()
        assert queue.submitted == []

    async def test_no_probes_configured_touches_no_host(self):
        """Off must mean OFF: not one probe is issued on either host."""
        from orchestrator.verify_runner import DriftDetector
        pool, local_run, remote_fake = _make_env_drift_pool()
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        await detector.check_env([], local_run=local_run)
        assert local_run.issued == []
        assert remote_fake.env_probe_run().issued == []

    async def test_no_eligible_remote_is_inconclusive(self):
        """Lever C off, or the remote already quarantined by the verdict path."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, VerifyRunnerPool
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        pool = VerifyRunnerPool([local_fake])
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=queue)
        result = await detector.check_env(_ENV_PROBE_SPECS)
        assert result.verdict == DriftVerdict.INCONCLUSIVE
        event_store.emit.assert_not_called()
        assert queue.submitted == []

    async def test_remote_quarantined_by_verdict_path_is_inconclusive(self):
        """Concrete form of the above: check() benched the remote first."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict
        pool, local_run, _ = _make_env_drift_pool()
        pool.quarantine('laptop')
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_no_local_runner_is_inconclusive(self):
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, VerifyRunnerPool
        _, _, remote_fake = _make_env_drift_pool()
        pool = VerifyRunnerPool([remote_fake])
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        result = await detector.check_env(_ENV_PROBE_SPECS)
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_remote_without_env_probe_run_is_inconclusive(self):
        """A non-RemoteRunner stand-in: duck-typed, so VerifyRunner stays unwidened.

        env_probe_run is NOT on the VerifyRunner Protocol, so any pool member
        that is not a RemoteRunner legitimately lacks it.  That must degrade,
        not raise out of a detective control.
        """
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, VerifyRunnerPool
        local_fake = MagicMock(spec=VerifyRunner)
        local_fake.name = 'local'
        local_fake.is_local = True
        bare_remote = MagicMock(spec=VerifyRunner)
        bare_remote.name = 'laptop'
        bare_remote.is_local = False
        assert not hasattr(bare_remote, 'env_probe_run')
        pool = VerifyRunnerPool([local_fake, bare_remote])
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=queue)
        result = await detector.check_env(_ENV_PROBE_SPECS)
        assert result.verdict == DriftVerdict.INCONCLUSIVE
        event_store.emit.assert_not_called()
        assert queue.submitted == []

    async def test_remote_transport_failure_is_inconclusive_not_divergence(self):
        """Invariant 5: a closed laptop is not a version skew."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, RunnerUnavailable
        pool, local_run, _ = _make_env_drift_pool(
            remote_raises=RunnerUnavailable('laptop unreachable')
        )
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=queue)
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.INCONCLUSIVE
        event_store.emit.assert_not_called()
        assert queue.submitted == []

    async def test_local_transport_failure_is_inconclusive(self):
        """Symmetric with the remote arm."""
        from orchestrator.verify_runner import DriftDetector, DriftVerdict, RunnerUnavailable
        pool, _, _ = _make_env_drift_pool()
        local_run = _make_probe_run({}, raises=RunnerUnavailable('local seam down'))
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.INCONCLUSIVE

    async def test_inconclusive_does_not_quarantine(self):
        from orchestrator.verify_runner import DriftDetector, RunnerUnavailable
        pool, local_run, _ = _make_env_drift_pool(
            remote_raises=RunnerUnavailable('laptop unreachable')
        )
        detector = DriftDetector(pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue())
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert pool.is_quarantined('laptop') is False


# ---------------------------------------------------------------------------
# task 4537: an UNREACHABLE remote is not divergence — through the REAL seam
# ---------------------------------------------------------------------------

# Exit status ssh reserves for its own errors ("or 255 if an error occurred",
# ssh(1) EXIT STATUS).  A sleeping laptop, a refused connection and a
# ConnectTimeout all land here.
_SSH_UNREACHABLE_RC = 255


def _make_ssh_seam(rc_by_probe=None, *, default=None):
    """Return a `self._run` stand-in shaped EXACTLY like _default_subprocess_run.

    That is the point of this helper: the production local seam RETURNS
    ``(rc, stdout, stderr)`` for every outcome including transport failure and
    raises nothing, so a test double that raises cannot reach the code path a
    down host actually takes.  Keys of *rc_by_probe* are matched against the
    probe command the ssh argv carries after REMOTE_TOOL_PATH_PRELUDE.
    """
    from orchestrator.verify_runner import REMOTE_TOOL_PATH_PRELUDE
    if default is None:
        default = (
            _SSH_UNREACHABLE_RC,
            '',
            'ssh: connect to host laptop.local port 22: No route to host',
        )
    issued = []

    async def run(argv, *, cwd=None):
        issued.append(list(argv))
        remote_cmd = argv[-1].removeprefix(f'{REMOTE_TOOL_PATH_PRELUDE} ')
        if rc_by_probe is not None and remote_cmd in rc_by_probe:
            return rc_by_probe[remote_cmd]
        return default

    recording: Any = run
    recording.issued = issued
    return recording


def _make_real_remote_pool(ssh_run):
    """Pool of (local stand-in, REAL RemoteRunner) driven by *ssh_run*.

    A real RemoteRunner — not a fake with an env_probe_run attribute — because
    the defect this class pins lives in the adapter RemoteRunner.env_probe_run
    returns, between ``self._run`` and capture_env_fingerprint.
    """
    from orchestrator.verify_runner import VerifyRunnerPool

    local_fake = MagicMock(spec=VerifyRunner)
    local_fake.name = 'local'
    local_fake.is_local = True
    remote = gated_remote_runner(
        name='laptop',
        ssh_host='laptop.local',
        git_remote='origin',
        cwd='/repo',
        run=ssh_run,
    )
    return VerifyRunnerPool([local_fake, remote]), remote


@pytest.mark.asyncio
class TestDriftDetectorCheckEnvUnreachableRemote:
    """Invariant 5 against the seam production actually has.

    ``TestDriftDetectorCheckEnvInconclusive`` covers a probe seam that RAISES
    RunnerUnavailable — a condition the production path cannot produce.
    ``RemoteRunner.env_probe_run`` dispatches through ``self._run``, which
    defaults to ``_default_subprocess_run``; that RETURNS ``(rc, out, err)``
    and never raises, and ssh reports an unreachable host as rc=255.  Left
    unhandled, every probe renders ``'<unavailable rc=255>'`` on the remote
    side and a real version locally, i.e. a sleeping laptop reads as total
    env skew: a blocking L1 naming every tool, which then DEDUPS AWAY the
    genuine env-divergence alarm until a human resolves it.

    These tests therefore drive a real RemoteRunner over the returning-rc
    shape, which is the only way to exercise the guard against the input the
    closed-laptop case supplies.
    """

    def _detector(self, ssh_run):
        from orchestrator.verify_runner import DriftDetector
        pool, _ = _make_real_remote_pool(ssh_run)
        event_store = MagicMock()
        queue = _FakeEscQueue()
        detector = DriftDetector(pool, event_store=event_store, escalation_queue=queue)
        return detector, pool, event_store, queue

    async def test_unreachable_remote_is_inconclusive_not_divergence(self):
        """rc=255 on every probe is a transport failure, not a version skew."""
        from orchestrator.verify_runner import DriftVerdict
        detector, _, _, _ = self._detector(_make_ssh_seam())
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.INCONCLUSIVE, (
            f'a host that ssh could not reach was reported as {result.verdict} '
            f'with drift_dimensions={result.drift_dimensions!r} — a closed '
            f'laptop must never raise an env-skew alarm'
        )
        assert result.is_faithful is True
        assert result.drift_dimensions == ()

    async def test_unreachable_remote_emits_no_divergence_event(self):
        detector, _, event_store, _ = self._detector(_make_ssh_seam())
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        event_store.emit.assert_not_called()

    async def test_unreachable_remote_files_no_escalation(self):
        """The blocking-L1 half: the bogus alarm would suppress the real one.

        Both escalations dedup on ``(_ENV_DRIFT_SENTINEL,
        category='verify_env_divergence')``, so ONE false L1 filed while the
        laptop is asleep silences the detector for the finding it exists to
        produce.
        """
        detector, _, _, queue = self._detector(_make_ssh_seam())
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert queue.submitted == [], (
            f'unreachable host filed {len(queue.submitted)} escalation(s): '
            f'{[e.summary for e in queue.submitted]}'
        )

    async def test_unreachable_remote_issues_no_local_probe(self):
        """A host we cannot probe must cost ZERO local subprocesses.

        The remote is fingerprinted FIRST precisely so an unreachable host is
        discovered before seven binaries are spawned on the dispatcher — on
        the production path that is every drift check taken while the laptop
        is asleep, and in the test suite it is what keeps a drift test that
        never meant to probe anything off the real machine.
        """
        detector, _, _, _ = self._detector(_make_ssh_seam())
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert local_run.issued == [], (
            f'local host was probed {len(local_run.issued)} times for a '
            f'comparison that could never be made: {local_run.issued!r}'
        )

    async def test_ssh_spawn_failure_is_inconclusive(self):
        """OSError from spawning ssh itself is transport, not a missing tool.

        ``_probe_run_safe`` normalises OSError to rc=127 so an ABSENT LOCAL
        BINARY compares equal against an absent remote one.  Applied to the
        ssh adapter that normalisation is wrong in the same direction as
        rc=255: it would render every remote probe '<unavailable rc=127>'
        against real local versions.
        """
        from orchestrator.verify_runner import DriftVerdict

        async def ssh_missing(argv, *, cwd=None):
            raise FileNotFoundError(2, 'No such file or directory', 'ssh')

        detector, _, _, queue = self._detector(ssh_missing)
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.INCONCLUSIVE
        assert queue.submitted == []

    async def test_remote_binary_absent_is_still_real_divergence(self):
        """The guard must not over-broaden to 'any non-zero rc is transport'.

        A REACHED host that simply lacks the tool exits 127 through the remote
        shell.  That is a genuine, reportable difference between the two
        hosts, and swallowing it would disarm the detector for the most likely
        real finding (a tool installed on one side only).
        """
        from orchestrator.verify_runner import DriftVerdict
        ssh_run = _make_ssh_seam({
            'npx pyright --version': (127, '', 'npx: command not found'),
            'uv --version': (0, 'uv 0.11.6\n', ''),
            'rustc --version': (127, '', 'rustc: command not found'),
            'cargo --version': (127, '', 'cargo: command not found'),
            'sccache --show-stats': (127, '', 'sccache: command not found'),
        })
        detector, _, _, queue = self._detector(ssh_run)
        local_run = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        result = await detector.check_env(_ENV_PROBE_SPECS, local_run=local_run)
        assert result.verdict == DriftVerdict.DIVERGE
        assert result.drift_dimensions == ('extra_probes',)
        assert result.remote_probes['pyright'] == '<unavailable rc=127>'
        assert result.remote_probes['uv'] == 'uv 0.11.6'
        assert len(queue.submitted) == 1
        assert 'pyright' in queue.submitted[0].summary
        assert 'uv' not in queue.submitted[0].summary


# ---------------------------------------------------------------------------
# task 4537 step-21: the LOCAL probe seam must BE the verify seam
# ---------------------------------------------------------------------------


def _make_exec_recorder(*, rc=0, stdout=b'  Python 3.12.3\n', stderr=b'  npm notice \n'):
    """Stand-in for ``asyncio.create_subprocess_exec`` recording every call.

    Returns ``(fake_exec, calls)`` where each entry of *calls* is
    ``{'argv': [...], 'kwargs': {...}}``.  Hermetic by construction: nothing is
    ever spawned, so these tests never touch the machine's real PATH, its real
    toolchain, or the network (`npx pyright --version` reaches the npm registry
    on a cold cache).
    """
    calls: list[dict[str, Any]] = []

    async def fake_exec(*argv, **kwargs):
        calls.append({'argv': list(argv), 'kwargs': kwargs})
        proc = MagicMock()
        proc.returncode = rc
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
        return proc

    return fake_exec, calls


@pytest.mark.asyncio
class TestEnvProbeLocalSeamMatchesVerifySeam:
    """The LOCAL probe must be spawned under the env every VERIFY command gets.

    Reviewer finding (blocking): ``check_env`` passed ``local_run=None``
    straight through to ``capture_env_fingerprint``, whose fallback
    ``_default_subprocess_run`` is a bare ``create_subprocess_exec`` with NO
    ``env=`` — so the local half inherited the orchestrator daemon's own
    environ verbatim.  That daemon is started by systemd as ``uv run --frozen
    --project orchestrator ...``
    (scripts/orchestrator-dark-factory.service), and ``/proc/<pid>/environ`` of
    the live unit carries ``VIRTUAL_ENV=<root>/.venv`` with PATH fronted by
    ``<root>/.venv/bin``.  Under that PATH ``python3`` resolves to
    ``.venv/bin/python3`` (3.13.9) while the remote's non-login ssh shell — and
    every real local verify command, spawned through
    ``verify._target_subprocess_env`` — resolves ``/usr/bin/python3`` (3.12.3).

    The committed ``python3`` probe would therefore have reported a false
    3.13.9-vs-3.12.3 divergence on the first live check, and by ``check_env``'s
    own ``has_open_l1(_ENV_DRIFT_SENTINEL, category=...)`` dedup that false
    blocking L1 would stay open and SUPPRESS the genuine env-divergence alarm —
    exactly the masking failure the INCONCLUSIVE handling exists to prevent.

    Every test here monkeypatches; none spawns.
    """

    def _activate_fake_venv(self, monkeypatch, tmp_path):
        """Synthesise the daemon's own activation, hermetically.

        VIRTUAL_ENV plus a PATH fronted by its bin dir is precisely the shape
        ``uv run`` leaves in the orchestrator process.  HOME is pinned to
        *tmp_path* so the cwd assertions below name a directory this test
        controls rather than the runner's real home.
        """
        monkeypatch.setenv('VIRTUAL_ENV', '/tmp/fakevenv')
        monkeypatch.setenv('PATH', '/tmp/fakevenv/bin:/usr/bin:/bin')
        monkeypatch.setenv('HOME', str(tmp_path))

    async def _spawn_once(self, monkeypatch, tmp_path, **kwargs):
        """Await the seam once and return the single recorded exec call."""
        from orchestrator import verify_runner as vr
        self._activate_fake_venv(monkeypatch, tmp_path)
        fake_exec, calls = _make_exec_recorder()
        monkeypatch.setattr(vr.asyncio, 'create_subprocess_exec', fake_exec)
        result = await vr.env_probe_local_run(['python3', '--version'], **kwargs)
        assert len(calls) == 1
        return calls[0], result

    async def test_seam_drops_the_daemons_virtual_env(self, monkeypatch, tmp_path):
        """VIRTUAL_ENV must not reach the probe — it is what makes uv/python skew."""
        call, _ = await self._spawn_once(monkeypatch, tmp_path)
        env = call['kwargs'].get('env')
        assert env is not None, (
            'the local probe seam spawned with no env= at all, so it inherited '
            "the daemon's own venv activation verbatim"
        )
        assert 'VIRTUAL_ENV' not in env, (
            f"VIRTUAL_ENV={env['VIRTUAL_ENV']!r} leaked into the probe env"
        )

    async def test_seam_strips_the_venv_bin_from_path(self, monkeypatch, tmp_path):
        """The measured defect: .venv/bin first on PATH resolves python3 3.13.9."""
        call, _ = await self._spawn_once(monkeypatch, tmp_path)
        env = call['kwargs'].get('env') or {}
        assert env.get('PATH') == '/usr/bin:/bin', (
            f"probe PATH was {env.get('PATH')!r} — the venv bin dir is still on "
            'it, so a bare `python3`/`uv` resolves the orchestrator venv rather '
            'than the host toolchain the remote will report'
        )

    async def test_seam_defaults_cwd_to_home(self, monkeypatch, tmp_path):
        """Match where `ssh <host> <cmd>` lands, so npx walks the same tree."""
        call, _ = await self._spawn_once(monkeypatch, tmp_path)
        assert call['kwargs'].get('cwd') == os.path.expanduser('~') == str(tmp_path)

    async def test_explicit_cwd_overrides_the_home_default(self, monkeypatch, tmp_path):
        """The seam keeps _default_subprocess_run's cwd= contract."""
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        call, _ = await self._spawn_once(monkeypatch, tmp_path, cwd=str(elsewhere))
        assert call['kwargs'].get('cwd') == str(elsewhere)

    async def test_seam_returns_rc_and_stripped_streams(self, monkeypatch, tmp_path):
        """Satisfies capture_env_fingerprint's run contract exactly like the default."""
        _, result = await self._spawn_once(monkeypatch, tmp_path)
        assert result == (0, 'Python 3.12.3', 'npm notice')

    async def test_check_env_drives_local_capture_through_the_scrubbed_seam(
        self, monkeypatch
    ):
        """ROUTING: with local_run omitted, check_env must use the scrubbed seam.

        The seam existing is not enough — the production call site
        (merge_drift._run_drift_check) is the ONLY caller that relies on the
        ``local_run=None`` default, so a seam the default does not reach fixes
        nothing.
        """
        from orchestrator import verify_runner as vr
        from orchestrator.verify_runner import DriftDetector, DriftVerdict

        scrubbed = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        unscrubbed = _make_probe_run(_MATCHING_PROBE_OUTPUT)
        monkeypatch.setattr(vr, 'env_probe_local_run', scrubbed)
        monkeypatch.setattr(vr, '_default_subprocess_run', unscrubbed)

        pool, _, _ = _make_env_drift_pool()
        detector = DriftDetector(
            pool, event_store=MagicMock(), escalation_queue=_FakeEscQueue()
        )
        result = await detector.check_env(_ENV_PROBE_SPECS)

        assert result.verdict == DriftVerdict.AGREE
        assert ['npx', 'pyright', '--version'] in scrubbed.issued, (
            f'the local probes never reached the scrubbed seam; it saw '
            f'{scrubbed.issued!r}'
        )
        assert unscrubbed.issued == [], (
            f'check_env fell back to _default_subprocess_run — the unscoped '
            f"seam that inherits the daemon's venv — for {unscrubbed.issued!r}"
        )


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

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        remote_cmd = ssh_call[0][-1]
        parsed = gate_argv(remote_cmd)

        rid_idx = parsed.index('--request-id')
        assert parsed[rid_idx + 1] == 'fixed-id'

    async def test_request_id_same_as_push_ref_id(self):
        """--request-id value equals the id in the push ref (abc123:refs/merge-verify/<id>)."""

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        push_call = next(
            c for c in calls
            if c[0][0] == 'git' and len(c[0]) > 3 and 'refs/merge-verify/' in c[0][3]
        )
        ref_part = push_call[0][3]
        push_id = ref_part.split('/')[-1]

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = gate_argv(ssh_call[0][-1])
        rid_idx = parsed.index('--request-id')
        ssh_id = parsed[rid_idx + 1]

        assert push_id == ssh_id == 'fixed-id'

    async def test_existing_positional_args_unchanged(self):
        """The gate binary's arguments still start ['verify-merge', '--sha', 'abc123']."""

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = gate_argv(ssh_call[0][-1])
        assert parsed[:3] == ['verify-merge', '--sha', 'abc123']

    async def test_request_id_appended_after_spec(self):
        """--request-id appears after --spec (appended at end)."""

        runner, calls = self._make_runner_and_calls()
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = gate_argv(ssh_call[0][-1])

        spec_idx = parsed.index('--spec')
        rid_idx = parsed.index('--request-id')
        assert rid_idx > spec_idx, '--request-id must be appended after --spec'

    async def test_request_id_with_config_path(self):
        """--request-id is still appended after --config when config_path is set."""

        runner, calls = self._make_runner_and_calls(config_path='/etc/orch.yaml')
        await runner.run_merge_verify('abc123', _make_spec())

        ssh_call = next(c for c in calls if c[0][0] == 'ssh')
        parsed = gate_argv(ssh_call[0][-1])

        cfg_idx = parsed.index('--config')
        rid_idx = parsed.index('--request-id')
        assert rid_idx > cfg_idx, '--request-id must come after --config'

    async def test_dispatch_not_in_flight_after_exception(self):
        """The dispatch is no longer in flight after RunnerUnavailable (cleared in the finally)."""
        from orchestrator.verify_runner import RunnerUnavailable

        calls = []

        async def fail_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[0] == 'git' and len(argv) > 3 and 'refs/merge-verify/' in argv[3]:
                return (1, '', 'push rejected')
            return (0, '', '')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fail_run,
            id_factory=lambda: 'fail-id',
        )
        with pytest.raises(RunnerUnavailable):
            await runner.run_merge_verify('abc123', _make_spec())

        assert runner.dispatch_in_flight is False

    async def test_dispatch_in_flight_false_before_and_after_return(self):
        """dispatch_in_flight is False before run_merge_verify and False again after it returns."""
        runner, _ = self._make_runner_and_calls()
        assert runner.dispatch_in_flight is False
        await runner.run_merge_verify('abc123', _make_spec())
        assert runner.dispatch_in_flight is False

    async def test_dispatch_in_flight_true_during_dispatch(self):
        """dispatch_in_flight is True while the ssh dispatch is live, False once it returns."""
        recorded = {}
        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')

        async def fake_run(argv, *, cwd=None):
            if argv[0] == 'ssh':
                recorded['during_dispatch'] = runner.dispatch_in_flight
                return (0, result_to_json(expected), '')
            return (0, '', '')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'req-id',
        )
        assert runner.dispatch_in_flight is False
        await runner.run_merge_verify('abc123', _make_spec())

        assert recorded['during_dispatch'] is True
        assert runner.dispatch_in_flight is False


# ---------------------------------------------------------------------------
# β step-13: RemoteRunner.cancel_verify() and probe_clean()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerCancelVerify:
    """cancel_verify() issues ssh cancel-verify; probe_clean() issues ssh pgrep."""

    def _make_runner_and_calls(self, *, probe_rc=1):
        """Return (runner, calls_list) where calls_list records each run() argv."""
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[0] == 'ssh':
                if 'pgrep' in argv[-1]:
                    return (probe_rc, '', '')
                return (0, '', '')
            return (0, '', '')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'req-42',
        )
        return runner, calls

    async def _cancel_mid_verify(self, *, config_path=None, cancel_rc=0):
        """Cancel a real dispatch while its verify ssh is held; return (cancel rc, recorded argvs)."""
        from orchestrator.verify_runner import RunnerUnavailable

        calls: list[list[str]] = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[0] == 'git':
                return (0, '', '')
            if 'cancel-verify' in argv[-1]:
                return (cancel_rc, '', '')
            if 'pgrep' in argv[-1]:
                return (1, '', '')
            entered.set()
            await release.wait()
            return (137, '', 'Killed')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            config_path=config_path,
            run=fake_run,
            id_factory=lambda: 'req-42',
        )
        dispatch = asyncio.create_task(runner.run_merge_verify('abc123', _make_spec()))
        await entered.wait()
        rc = await runner.cancel_verify()
        release.set()
        with pytest.raises(RunnerUnavailable):
            await dispatch
        return rc, calls

    @staticmethod
    def _cancel_ssh_calls(calls):
        return [c for c in calls if c[0] == 'ssh' and 'cancel-verify' in c[-1]]

    async def test_cancel_verify_no_inflight_returns_zero_no_ssh(self):
        """cancel_verify() with no dispatch in flight returns 0 without issuing ssh."""
        runner, calls = self._make_runner_and_calls()
        rc = await runner.cancel_verify()
        assert rc == 0
        ssh_calls = [c for c in calls if c[0] == 'ssh']
        assert len(ssh_calls) == 0

    async def test_cancel_verify_issues_correct_argv(self):
        """cancel_verify() issues ssh BatchMode/ConnectTimeout cancel-verify --request-id."""

        _, calls = await self._cancel_mid_verify(cancel_rc=0)

        ssh_calls = self._cancel_ssh_calls(calls)
        assert len(ssh_calls) == 1
        argv = ssh_calls[0]

        assert 'BatchMode=yes' in argv
        assert 'laptop.local' in argv

        remote_cmd = argv[-1]
        parsed = gate_argv(remote_cmd)
        assert 'cancel-verify' in parsed
        rid_idx = parsed.index('--request-id')
        assert parsed[rid_idx + 1] == 'req-42'

    async def test_cancel_verify_appends_config_when_set(self):
        """cancel_verify() appends --config <path> when config_path is set."""

        _, calls = await self._cancel_mid_verify(config_path='/etc/orch.yaml', cancel_rc=0)

        ssh_calls = self._cancel_ssh_calls(calls)
        remote_cmd = ssh_calls[0][-1]
        parsed = gate_argv(remote_cmd)
        cfg_idx = parsed.index('--config')
        assert parsed[cfg_idx + 1] == '/etc/orch.yaml'

    async def test_cancel_verify_returns_ssh_rc(self):
        """cancel_verify() returns the ssh return code."""
        rc, _ = await self._cancel_mid_verify(cancel_rc=1)
        assert rc == 1

    async def test_probe_clean_true_when_pgrep_rc_is_1(self):
        """probe_clean() issues ssh pgrep -f verify-merge; rc==1 (no match) → True."""
        runner, calls = self._make_runner_and_calls(probe_rc=1)
        result = await runner.probe_clean()
        assert result is True

        ssh_calls = [c for c in calls if c[0] == 'ssh']
        assert len(ssh_calls) == 1
        remote_cmd = ssh_calls[0][-1]
        assert 'pgrep' in remote_cmd
        assert 'verify-merge' in remote_cmd

    async def test_probe_clean_false_when_pgrep_rc_is_0(self):
        """probe_clean() rc==0 (process running) → False."""
        runner, calls = self._make_runner_and_calls(probe_rc=0)
        result = await runner.probe_clean()
        assert result is False

    async def test_probe_clean_false_when_pgrep_rc_is_2(self):
        """probe_clean() rc>=2 (error) → False (conservative: stay parked)."""
        runner, _ = self._make_runner_and_calls(probe_rc=2)
        result = await runner.probe_clean()
        assert result is False

    async def test_probe_clean_false_on_oserror(self):
        """probe_clean() on OSError → False (fail-safe)."""
        async def raising_run(argv, *, cwd=None):
            raise OSError('connection refused')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=raising_run,
        )
        result = await runner.probe_clean()
        assert result is False


# ---------------------------------------------------------------------------
# task 4537 step-7: RemoteRunner.env_probe_run() — the ssh capture adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerEnvProbeRun:
    """env_probe_run() returns a capture_env_fingerprint-compatible adapter
    that dispatches each probe argv over ssh through the plain self._run seam,
    behind REMOTE_TOOL_PATH_PRELUDE (task 7097) so the non-login ssh shell
    finds ~/.local/bin tools the way RemoteRunner's own `uv sync` does."""

    def _make_runner(self, *, responses=None):
        calls = []
        queue = list(responses or [])

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if queue:
                return queue.pop(0)
            return (0, 'ok\n', '')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
        )
        return runner, calls

    async def test_issues_exact_ssh_argv_with_shared_base_opts(self):
        """The adapter reuses _SSH_BASE_OPTS verbatim rather than re-spelling flags."""
        from orchestrator.verify_runner import _SSH_BASE_OPTS, REMOTE_TOOL_PATH_PRELUDE

        runner, calls = self._make_runner()
        adapter = runner.env_probe_run()
        await adapter(['uv', '--version'])

        assert calls == [[
            'ssh', *_SSH_BASE_OPTS, 'laptop.local',
            f'{REMOTE_TOOL_PATH_PRELUDE} uv --version',
        ]]

    async def test_remote_command_leads_with_the_unquoted_path_prelude(self):
        """Task 7097: a non-login ssh shell lacks ~/.local/bin, so a bare
        `uv --version` exits 127 on a host where uv IS installed — a false
        divergence that, deduped, would suppress the real skew alarm. The
        prelude goes first and unquoted ($HOME must expand remotely); the
        probe argv follows, still shlex-quoted."""
        from orchestrator.verify_runner import REMOTE_TOOL_PATH_PRELUDE

        runner, calls = self._make_runner()
        adapter = runner.env_probe_run()
        await adapter(['sh', '-c', 'echo a b'])

        remote_cmd = calls[0][-1]
        assert remote_cmd.startswith(f'{REMOTE_TOOL_PATH_PRELUDE} ')
        assert '$HOME/.local/bin' in remote_cmd
        assert remote_cmd.removeprefix(f'{REMOTE_TOOL_PATH_PRELUDE} ') == "sh -c 'echo a b'"

    async def test_returns_run_result_tuple_unchanged(self):
        """(rc, stdout, stderr) passes through — satisfies capture's run contract."""
        runner, _ = self._make_runner(responses=[(0, 'uv 0.10.6\n', 'warn')])
        adapter = runner.env_probe_run()
        assert await adapter(['uv', '--version']) == (0, 'uv 0.10.6\n', 'warn')

    async def test_argv_elements_are_shlex_quoted(self):
        """An argv element needing quoting is quoted into the remote command string."""
        runner, calls = self._make_runner()
        adapter = runner.env_probe_run()
        await adapter(['sh', '-c', 'echo a b'])

        from orchestrator.verify_runner import REMOTE_TOOL_PATH_PRELUDE

        remote_cmd = calls[0][-1]
        assert remote_cmd == f"{REMOTE_TOOL_PATH_PRELUDE} sh -c 'echo a b'"

    async def test_capture_env_fingerprint_drives_probes_over_ssh(self):
        """End-to-end: capture_env_fingerprint(adapter) populates extra_probes."""
        from orchestrator.verify_runner import capture_env_fingerprint

        runner, calls = self._make_runner(responses=[
            (127, '', 'rustc: command not found'),   # rustc
            (127, '', 'cargo: command not found'),   # cargo
            (127, '', 'sccache: command not found'), # sccache
            (0, '1.1.377\n', ''),                    # pyright probe
            (0, 'uv 0.10.6\n', ''),                  # uv probe
        ])
        adapter = runner.env_probe_run()
        fp = await capture_env_fingerprint(adapter, extra_probe_specs=[
            ('pyright', ['npx', 'pyright', '--version']),
            ('uv', ['uv', '--version']),
        ])

        assert dict(fp.extra_probes) == {'pyright': '1.1.377', 'uv': 'uv 0.10.6'}
        # every probe went over ssh to the configured host
        assert all(c[0] == 'ssh' and 'laptop.local' in c for c in calls)
        from orchestrator.verify_runner import REMOTE_TOOL_PATH_PRELUDE
        assert calls[-2][-1] == f'{REMOTE_TOOL_PATH_PRELUDE} npx pyright --version'
        assert calls[-1][-1] == f'{REMOTE_TOOL_PATH_PRELUDE} uv --version'


# ---------------------------------------------------------------------------
# task-2362: ssh ServerAliveInterval/ServerAliveCountMax keepalive hardening
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerSshKeepalive:
    """All four ssh argv sites (health, run_merge_verify, cancel_verify,
    probe_clean) must carry the ServerAlive keepalive options, and must stay
    identical across sites (shared const, task 2362 / incident 5111)."""

    def _make_runner(self, expected_result, *, verify_entered, verify_release):
        calls = []

        async def fake_run(argv, *, cwd=None):
            calls.append(argv[:])
            if argv[0] == 'git':
                return (0, '', '')
            if argv[0] == 'ssh':
                if 'pgrep' in argv[-1]:
                    return (1, '', '')
                if 'cancel-verify' in argv[-1]:
                    return (0, '', '')
                if 'verify-merge' in argv[-1]:
                    verify_entered.set()
                    await verify_release.wait()
                    return (137, '', 'Killed')
                # health probe
                return (0, result_to_json(expected_result), '')
            return (0, '', '')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'fixed-id',
        )
        return runner, calls

    async def test_all_four_sites_carry_identical_keepalive_flags(self):
        """Every ssh site carries the keepalive flags with their exact
        configured values, and the option block is byte-identical across
        sites (shared `_SSH_BASE_OPTS` const — regression guard against the
        four sites drifting apart from each other).

        This test only asserts the argv surface. The consequence side of the
        change — a non-zero ssh rc (what a dead keepalive session ultimately
        produces) raising RunnerUnavailable — is already covered by
        `TestRemoteRunnerTransportVsTimeout.test_raises_runner_unavailable_on_ssh_nonzero`
        and is deliberately not duplicated here (incident 5111).
        """
        from orchestrator.verify_runner import (
            _SSH_BASE_OPTS,
            SSH_SERVER_ALIVE_COUNT_MAX,
            SSH_SERVER_ALIVE_INTERVAL,
            RunnerUnavailable,
        )

        expected = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
        verify_entered, verify_release = asyncio.Event(), asyncio.Event()
        runner, calls = self._make_runner(
            expected, verify_entered=verify_entered, verify_release=verify_release,
        )

        # Site 1: health()
        await runner.health()
        # Site 2: run_merge_verify() dispatch, its verify ssh held
        dispatch = asyncio.create_task(runner.run_merge_verify('abc123', _make_spec()))
        await verify_entered.wait()
        # Site 3: cancel_verify() of that in-flight verify
        await runner.cancel_verify()
        verify_release.set()
        with pytest.raises(RunnerUnavailable):
            await dispatch
        # Site 4: probe_clean()
        await runner.probe_clean()

        ssh_calls = [c for c in calls if c[0] == 'ssh']
        assert len(ssh_calls) == 4

        alive_interval_opt = f'ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL}'
        alive_count_opt = f'ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}'
        expected_prefix = ['ssh', *_SSH_BASE_OPTS]
        for argv in ssh_calls:
            # Per-flag checks: the keepalive options carry their exact
            # configured values, alongside the still-needed ConnectTimeout
            # (bounds only the initial connect, not a mid-session stall) and
            # BatchMode.
            assert alive_interval_opt in argv, argv
            assert alive_count_opt in argv, argv
            assert 'ConnectTimeout=10' in argv, argv
            assert 'BatchMode=yes' in argv, argv
            # Anti-drift: every site's argv starts with 'ssh' + the exact
            # shared option block, followed by <host> <remote_cmd_or_literal>.
            assert argv[:len(expected_prefix)] == expected_prefix, argv
            assert argv[len(expected_prefix)] == 'laptop.local'


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

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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

        config = _narrow_config()
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

        config = _narrow_config()
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

        runner = gated_remote_runner(
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

        runner = gated_remote_runner(
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
    async def test_passing_result_writes_no_stderr_log(self, tmp_path):
        """Passing remote verify: ssh_stderr noise is NOT archived as a .stderr.log
        (failure-only contract).  Task 2822 fix (c) now writes a pass-summary on the
        pass path, but the failure-specific .stderr.log must never appear."""
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

        runner = gated_remote_runner(
            name='leo-laptop',
            ssh_host='leo-laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'test-id',
        )

        await runner.run_merge_verify('abc123', _make_spec(), task_id='1920', archive_root=tmp_path)

        task_dir = tmp_path / '1920'
        entries = sorted(p.name for p in task_dir.iterdir()) if task_dir.exists() else []
        # The failure-specific .stderr.log must NOT be written on a pass...
        assert not any(n.endswith('.stderr.log') for n in entries), (
            f'Expected no .stderr.log on pass, got {entries}'
        )
        # ...only the task 2822 fix (c) pass-summary appears (exactly one artifact).
        assert entries == [n for n in entries if '.pass-summary-' in n], (
            f'Expected only pass-summary artifacts on pass, got {entries}'
        )
        assert len(entries) == 1, f'Expected exactly 1 pass-summary, got {entries}'

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

        runner = gated_remote_runner(
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
    async def test_passing_result_writes_no_stream_files(self, tmp_path):
        """Passing remote verify (passed=True) → no failure stream logs / summary.
        Task 2822 fix (c) writes a pass-summary on the pass path, but the
        failure-specific stream .log files and summary-*.json must never appear."""
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

        runner = gated_remote_runner(
            name='leo-laptop', ssh_host='leo-laptop.local', git_remote='origin',
            cwd='/repo', run=fake_run, id_factory=lambda: 'test-id',
        )

        await runner.run_merge_verify('abc123', _make_spec(), task_id='1921', archive_root=tmp_path)

        task_dir = tmp_path / '1921'
        entries = sorted(p.name for p in task_dir.iterdir()) if task_dir.exists() else []
        # No failure stream .log files and no failure summary-*.json on a pass...
        assert not any(n.endswith('.log') for n in entries), (
            f'Expected no failure stream .log on pass, got {entries}'
        )
        # ...only the task 2822 fix (c) pass-summary appears (exactly one artifact).
        assert entries == [n for n in entries if '.pass-summary-' in n], (
            f'Expected only pass-summary artifacts on pass, got {entries}'
        )
        assert len(entries) == 1, f'Expected exactly 1 pass-summary, got {entries}'

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


# ---------------------------------------------------------------------------
# task-2822 fix (c): TestRemoteRunnerPassSummaryArchival
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerPassSummaryArchival:
    """RemoteRunner archives a PASSING remote merge-verify as a best-effort JSON
    pass-summary (task 2822 fix c), closing the incident's un-auditable-laptop-scope
    gap.  Filename mirrors the failure archivers:
        attempt-1.remote-<safe_name>.pass-summary-<utc_ts>.json
    capturing SCOPE (task_files + merge_verify_workspace + merge_verify_breadth from
    the SPEC), RESULT (passed, category), and TIMING (duration_ms), plus runner name
    and merge_sha.  A FAILING verify writes NO pass-summary (failure-only archives
    unchanged); a missing archive_root/task_id is a best-effort no-op.
    """

    def _make_runner(self, result, *, name='leo-laptop'):
        """RemoteRunner whose fake run returns git→(0,'','') and ssh→(0,json,'')."""
        _it = iter([
            (0, '', ''),                       # git push (load-bearing)
            (0, result_to_json(result), ''),   # ssh verify
        ])

        async def fake_run(argv, *, cwd=None):
            # ref cleanup delete push (best-effort, inside contextlib.suppress)
            if argv[0] == 'git' and '--delete' in argv:
                return (0, '', '')
            return next(_it)

        return gated_remote_runner(
            name=name,
            ssh_host=f'{name}.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_run,
            id_factory=lambda: 'test-id',
        )

    def _spec_with_scope(self):
        """Spec carrying task_files + the FULL merge-gate profile (fix a fields)."""
        return MergeVerifySpec(
            verify_commands=(),
            unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
            task_files=('src/a/mod.py', 'src/b/mod.py'),
            verify_env={},
            cold_timeout_secs=60.0,
            merge_verify_workspace=True,
            merge_verify_breadth='full',
        )

    async def test_passing_remote_verify_writes_pass_summary(self, tmp_path):
        """A passing remote verify writes one pass-summary JSON with scope+result+timing."""
        pass_result = VerifyResult(
            passed=True, test_output='all green', lint_output='', type_output='',
            summary='ok', category='merge_ok',
        )
        runner = self._make_runner(pass_result)

        result = await runner.run_merge_verify(
            'abc123', self._spec_with_scope(), task_id='2822', archive_root=tmp_path,
        )

        # VerifyResult is returned UNCHANGED (PRD §A Invariant 5)
        assert result == pass_result

        task_dir = tmp_path / '2822'
        assert task_dir.is_dir(), f'Expected {task_dir} to be a directory'

        summaries = list(task_dir.glob('attempt-1.remote-leo-laptop.pass-summary-*.json'))
        assert len(summaries) == 1, f'Expected 1 pass-summary, got {[f.name for f in summaries]}'

        data = json.loads(summaries[0].read_text(encoding='utf-8'))
        # identity
        assert data['merge_sha'] == 'abc123'
        assert data['runner'] == 'leo-laptop'
        # RESULT
        assert data['passed'] is True
        assert data['category'] == 'merge_ok'
        # TIMING
        assert isinstance(data['duration_ms'], (int, float))
        assert data['duration_ms'] >= 0
        # SCOPE — sourced from the SPEC (the merge-gate profile), not the host config
        scope = data['scope']
        assert scope['task_files'] == ['src/a/mod.py', 'src/b/mod.py']
        assert scope['merge_verify_workspace'] is True
        assert scope['merge_verify_breadth'] == 'full'

    async def test_failing_remote_verify_writes_no_pass_summary(self, tmp_path):
        """A FAILING remote verify writes NO pass-summary (only the failure archives)."""
        fail_result = VerifyResult(
            passed=False, test_output='FAILED', lint_output='', type_output='',
            summary='test fail', category='test_failure',
        )
        runner = self._make_runner(fail_result)

        result = await runner.run_merge_verify(
            'abc123', self._spec_with_scope(), task_id='2822', archive_root=tmp_path,
        )
        assert result == fail_result

        task_dir = tmp_path / '2822'
        pass_files = list(task_dir.glob('*.pass-summary-*.json')) if task_dir.exists() else []
        assert pass_files == [], f'Expected no pass-summary on failure, got {pass_files}'

    async def test_no_pass_summary_when_archive_root_none(self, tmp_path):
        """archive_root=None → best-effort no-op; VerifyResult returned unchanged, no raise."""
        pass_result = VerifyResult(
            passed=True, test_output='all green', lint_output='', type_output='', summary='ok',
        )
        runner = self._make_runner(pass_result)

        result = await runner.run_merge_verify('abc123', self._spec_with_scope(), task_id='2822')
        assert result == pass_result

    async def test_no_pass_summary_when_task_id_none(self, tmp_path):
        """task_id=None → best-effort no-op; no files written, no raise."""
        pass_result = VerifyResult(
            passed=True, test_output='all green', lint_output='', type_output='', summary='ok',
        )
        runner = self._make_runner(pass_result)

        result = await runner.run_merge_verify('abc123', self._spec_with_scope(), archive_root=tmp_path)
        assert result == pass_result
        assert list(tmp_path.rglob('*.pass-summary-*.json')) == []


# ---------------------------------------------------------------------------
# γ step-7: _default_ssh_heartbeat_run — stdin heartbeat writer for the ssh
# dispatch child (real tiny subprocesses; no ssh spawned)
# ---------------------------------------------------------------------------


# Task 4195 step-7: sizing for test_beats_survive_a_blocked_event_loop, which
# blocks the dispatching event loop for longer than the child's own heartbeat
# deadline and lets the CHILD return the verdict.
#
# The bound needs a basis, because the flake class it must avoid is measured:
# test_laptop_warm_verify_boundary.py's HeartbeatWriter comment records an
# Event.wait-driven writer showing inter-write gaps of p999 0.386s / MAX 0.845s
# at loadavg 113-178, with the overshoot largely ADDITIVE scheduling delay
# rather than multiplicative.  Five beats at 0.05s is 0.25s nominal; even a
# pathological run costing one full 0.845s gap plus four 0.386s gaps totals
# ~2.4s, inside the 2.5s deadline.  That is why BEATS_REQUIRED is 5 rather than
# 10 and the deadline is 10x nominal rather than 2x.
#
# If this ever measures tight under an xdist storm, raise CHILD_DEADLINE_SECS
# and LOOP_HOG_SECS together (keeping the hog longer than the deadline); do NOT
# lower BEAT_INTERVAL_SECS, which is what re-creates the additive-delay cliff.
# The discriminator survives any such widening, because the pre-4195 producer
# delivers exactly ZERO beats for the whole hog.
#: The three sizings below are all MULTIPLES of production's teardown bound,
#: verify_runner.HEARTBEAT_STOP_JOIN_SECS, rather than independent literals:
#: that constant is the single home of both the number and the argument for it
#: (SPOT, docs/code-quality.md #11), and each relationship these comments claim
#: stays true through any retune of it.
#:
#: How long the wedged-writer thread stays parked: comfortably longer than that
#: bound, so the dispatch can only return by giving up on it rather than by
#: outlasting it.
WEDGED_WRITER_PARK_SECS = 6 * HEARTBEAT_STOP_JOIN_SECS

#: Budget for that test.  A regression to an unbounded join() must fail HERE as
#: a timeout rather than wedging the suite; sized above the bound it is meant to
#: observe and under WEDGED_WRITER_PARK_SECS, so it fires before the parked
#: thread expires on its own.
WEDGED_WRITER_TEST_TIMEOUT_SECS = 4 * HEARTBEAT_STOP_JOIN_SECS

#: Join ceiling for the parked thread's own cleanup once released — a wedge
#: detector, paid only if that thread is already stuck.
HEARTBEAT_STOP_JOIN_CEILING_SECS = HEARTBEAT_STOP_JOIN_SECS

#: Cadence of that test's loop-responsiveness ticker, and how many of its ticks
#: must land inside the teardown join.  The discriminator is EXACT in the
#: failing direction — a synchronous join lets precisely ZERO ticks run between
#: stop() and the dispatch returning, since nothing else runs on a blocked loop
#: — so any positive floor detects it and a low one cannot be flaky.  Two, in a
#: window of HEARTBEAT_STOP_JOIN_SECS: even if every tick cost the ~0.85s worst
#: single-gap scheduling delay measured at loadavg 113-178
#: (test_laptop_warm_verify_boundary.py's HeartbeatWriter comment) rather than
#: this cadence, that window still fits five.
LOOP_TICK_SECS = 0.05
LOOP_TICKS_REQUIRED_DURING_TEARDOWN = 2


BEATS_REQUIRED = 5
BEAT_INTERVAL_SECS = 0.05
CHILD_DEADLINE_SECS = 2.5
LOOP_HOG_SECS = 4.0

#: Child program for that test: counts beats off fd 0 under its own deadline and
#: exits with a verdict — 0 with BEATS_OK on stdout, 3 if it starved, 4 on EOF.
#: Computing the verdict in the child keeps the measurement off the very thread
#: the test deliberately blocks, and off any inter-beat gap measured in the
#: parent (the shape the 0.845s figure above makes flaky).
_BEAT_COUNTING_CHILD = f'''
import os, select, sys, time
deadline = time.monotonic() + {CHILD_DEADLINE_SECS}
beats = 0
while beats < {BEATS_REQUIRED}:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        sys.exit(3)
    ready, _, _ = select.select([0], [], [], remaining)
    if not ready:
        sys.exit(3)
    data = os.read(0, 4096)
    if data == b'':
        sys.exit(4)
    beats += data.count(b'\\n')
print('BEATS_OK')
'''


@pytest.mark.asyncio
class TestDefaultSshHeartbeatRun:
    """_default_ssh_heartbeat_run opens the child on the read end of a pipe it
    owns and writes a heartbeat token every heartbeat_interval seconds, from a
    dedicated OS thread, while awaiting completion.  A heartbeat write against
    an already-closed child stdin (EPIPE) is swallowed and never surfaces as an
    exception."""

    async def test_delivers_heartbeats(self):
        """A child blocked reading 2 stdin lines only prints its marker and exits
        once the heartbeat writer has fed it 2 newlines — proving heartbeats are
        actually written to the child's stdin while the run is in flight."""
        import sys

        from orchestrator.verify_runner import _default_ssh_heartbeat_run

        argv = [
            sys.executable, '-c',
            'import sys\n'
            'sys.stdin.readline()\n'
            'sys.stdin.readline()\n'
            "print('HEARTBEAT_MARKER')\n",
        ]
        rc, stdout, stderr = await _default_ssh_heartbeat_run(argv, heartbeat_interval=0.05)
        assert rc == 0
        assert 'HEARTBEAT_MARKER' in stdout
        assert stderr == ''

    async def test_swallows_epipe_on_immediate_child_exit(self):
        """A child that closes its own stdin (simulating the read end vanishing,
        as happens once ssh/the child is gone) and keeps running briefly gives
        the heartbeat writer several attempts against an already-closed pipe.
        The resulting EPIPE/ConnectionResetError must be swallowed — the call
        returns normally instead of raising.

        A bare `['true']` (immediate exit) is deliberately NOT used here: `true`
        exits so fast that the writer's first scheduled heartbeat routinely never
        fires before `communicate()` completes and the writer is cancelled —
        empirically confirmed to exercise the EPIPE path in 0/30 trials at this
        heartbeat_interval — which would make this test pass without ever
        touching the code it claims to cover. Closing stdin immediately but
        exiting only after a short sleep decouples "channel already closed"
        from "process already reaped", so several heartbeat attempts land on a
        provably closed pipe while `communicate()` is still pending.
        """
        import sys

        from orchestrator.verify_runner import _default_ssh_heartbeat_run

        argv = [sys.executable, '-c', 'import os, time; os.close(0); time.sleep(0.15)']
        rc, stdout, stderr = await _default_ssh_heartbeat_run(argv, heartbeat_interval=0.02)
        assert rc == 0
        assert stdout == ''
        assert stderr == ''

    async def test_beats_survive_a_blocked_event_loop(self):
        """The beat keeps its cadence while the dispatching event loop is blocked.

        THE HEADLINE PROPERTY of task 4195.  The orchestrator's single shared
        loop carries the scheduler, agent dispatch, the merge worker and the
        uvicorn escalation server; a producer scheduled on it stops beating for
        exactly as long as any of them stalls it, and the remote correctly reads
        a silent channel as a dead one.

        The loop is blocked here by a SYNCHRONOUS ``time.sleep`` for longer than
        the child's own heartbeat deadline.  That is an optimistic model of a
        real stall (it releases the GIL, where ``shutil.rmtree`` and sqlite
        commits hold it between syscalls), but modelling GIL contention is not
        this test's job: it discriminates the asyncio-scheduled producer from
        the OS-thread producer, which is the change under test.

        The PARENT asserts no timing at all — only the child's exit code.
        """
        import sys
        import time

        from orchestrator.verify_runner import _default_ssh_heartbeat_run

        argv = [sys.executable, '-c', _BEAT_COUNTING_CHILD]

        async def hog():
            await asyncio.sleep(0)  # let the dispatch coroutine reach its spawn
            time.sleep(LOOP_HOG_SECS)  # synchronous: blocks the whole loop

        (rc, stdout, stderr), _ = await asyncio.gather(
            _default_ssh_heartbeat_run(argv, heartbeat_interval=BEAT_INTERVAL_SECS),
            hog(),
        )

        assert rc == 0, f'child exited {rc} (3=heartbeat starved, 4=EOF): {stderr!r}'
        assert 'BEATS_OK' in stdout

    async def test_writer_does_not_outlive_the_dispatch(self):
        """The writer is stopped and joined before the call returns; its fd closed once.

        Stated against the thread NAME as well as the handle, so it also catches
        a writer leaked by some OTHER concurrently-finished dispatch in the same
        interpreter — which is the production shape, since the orchestrator runs
        many dispatches on one process.
        """
        import os
        import sys
        import threading

        from orchestrator.verify_cancel import HEARTBEAT_THREAD_NAME, start_stdin_heartbeat
        from orchestrator.verify_runner import _default_ssh_heartbeat_run

        handles = []
        closed = []

        def closing_spy(fd):
            closed.append(fd)
            os.close(fd)

        def spy_start(write_fd, **kw):
            handle = start_stdin_heartbeat(write_fd, **{**kw, 'close_fn': closing_spy})
            handles.append((write_fd, handle))
            return handle

        rc, stdout, stderr = await _default_ssh_heartbeat_run(
            [sys.executable, '-c', 'print("DONE")'],
            heartbeat_interval=0.02,
            start_heartbeat=spy_start,
        )

        assert rc == 0
        assert 'DONE' in stdout
        ((write_fd, handle),) = handles
        assert not handle.thread.is_alive()  # joined before the call returned
        assert closed == [write_fd]  # closed exactly once, by its owner
        assert [t for t in threading.enumerate() if t.name == HEARTBEAT_THREAD_NAME] == []

    @pytest.mark.skipif(not Path('/proc/self/fd').exists(), reason='needs procfs')
    async def test_a_failed_spawn_leaks_neither_the_writer_nor_its_pipe(self, tmp_path):
        """The ordering comment above os.pipe(), made executable.

        The writer is started BEFORE the spawn so that exactly one owner exists
        for write_fd on every path — and the failed-spawn path is the only one
        where that claim carries weight, because nothing else there stops the
        thread or closes either end of the pipe. It is also an ordinary
        condition rather than a hypothetical: a missing ssh binary or a worktree
        reaped out from under the cwd raises FileNotFoundError here, which
        callers already convert to RunnerUnavailable. Moving start_heartbeat
        after the spawn, or dropping the inner try/finally that closes read_fd,
        would leak descriptors per failed dispatch with every other test in this
        class — all of which take the happy spawn path — still green.
        """
        import os
        import threading

        from orchestrator.verify_cancel import HEARTBEAT_THREAD_NAME, start_stdin_heartbeat
        from orchestrator.verify_runner import _default_ssh_heartbeat_run

        handles = []
        closed = []

        def closing_spy(fd):
            closed.append(fd)
            os.close(fd)

        def spy_start(write_fd, **kw):
            handle = start_stdin_heartbeat(write_fd, **{**kw, 'close_fn': closing_spy})
            handles.append((write_fd, handle))
            return handle

        argv = [str(tmp_path / 'no-such-ssh-binary'), 'irrelevant']

        async def failed_dispatch():
            with pytest.raises(FileNotFoundError):
                await _default_ssh_heartbeat_run(
                    argv, heartbeat_interval=0.01, start_heartbeat=spy_start
                )

        # The first failed spawn on a loop can allocate asyncio machinery (child
        # watcher, self-pipe) that legitimately outlives it, so the count is
        # taken around the SECOND one: this stays an fd-LEAK assertion instead of
        # an asyncio-warm-up one.
        await failed_dispatch()
        before = len(os.listdir('/proc/self/fd'))
        await failed_dispatch()
        after = len(os.listdir('/proc/self/fd'))

        assert after == before, f'a failed spawn leaked {after - before} descriptor(s)'
        assert closed == [write_fd for write_fd, _ in handles]  # each fd closed once, by its owner
        assert all(not handle.thread.is_alive() for _, handle in handles)
        assert [t for t in threading.enumerate() if t.name == HEARTBEAT_THREAD_NAME] == []

    @pytest.mark.timeout(WEDGED_WRITER_TEST_TIMEOUT_SECS)
    async def test_a_wedged_writer_costs_one_teardown_and_not_the_loop(self):
        """A writer that ignores stop() costs its own dispatch's teardown — and nothing else.

        Two properties, and the bound means what it says only if both hold.

        BOUNDED: the dispatch still returns its rc and stdout. Regressing to a
        bare join() fails this as a timeout — the failure class (one stuck
        dispatch halting the merge lane) this task exists to remove.

        PAID OFF THE LOOP: a synchronous join in the coroutine's finally would
        freeze the orchestrator's shared loop — scheduler, agent dispatch, merge
        worker, uvicorn escalation server — for that whole bound, re-creating on
        the teardown side exactly the stall the beat was moved to a thread to
        survive. The ticker counts loop iterations between ``stop()`` (the last
        thing to run before the join) and the dispatch returning; a blocking
        join admits exactly zero, because nothing else can run on a loop that is
        blocked.
        """
        import os
        import sys
        import threading

        from orchestrator.verify_cancel import HeartbeatHandle
        from orchestrator.verify_runner import _default_ssh_heartbeat_run

        never_set = threading.Event()
        parked = threading.Thread(
            target=never_set.wait, args=(WEDGED_WRITER_PARK_SECS,), daemon=True
        )
        parked.start()

        ticks = 0
        ticks_at_stop: list[int] = []

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(LOOP_TICK_SECS)
                ticks += 1

        ticker_task = asyncio.create_task(ticker())
        write_fds: list[int] = []

        def wedged_start(write_fd, **kw):
            # No writer is started, so nothing else owns write_fd and this test
            # closes it below. Production delegates that close to the writer
            # thread (verify_cancel.run_stdin_heartbeat) — saying so here is how
            # this test states the ownership contract it is standing in for.
            write_fds.append(write_fd)
            return HeartbeatHandle(stop=lambda: ticks_at_stop.append(ticks), thread=parked)

        try:
            rc, stdout, stderr = await _default_ssh_heartbeat_run(
                [sys.executable, '-c', 'print("DONE")'],
                heartbeat_interval=LOOP_TICK_SECS,
                start_heartbeat=wedged_start,
            )
            ticks_at_return = ticks
            still_parked = parked.is_alive()  # the call returned without waiting it out
        finally:
            ticker_task.cancel()
            await asyncio.gather(ticker_task, return_exceptions=True)
            never_set.set()
            parked.join(timeout=HEARTBEAT_STOP_JOIN_CEILING_SECS)
            for fd in write_fds:
                os.close(fd)

        assert rc == 0
        assert 'DONE' in stdout
        assert still_parked

        (stopped_at,) = ticks_at_stop
        assert ticks_at_return - stopped_at >= LOOP_TICKS_REQUIRED_DURING_TEARDOWN, (
            f'the loop ran {ticks_at_return - stopped_at} times while the dispatch waited out '
            f'its {HEARTBEAT_STOP_JOIN_SECS}s join bound. Zero means the join is synchronous '
            f'again and the shared event loop — scheduler, merge worker, escalation server — '
            f'is frozen for that whole bound by every wedged writer.'
        )


# ---------------------------------------------------------------------------
# γ step-9: RemoteRunner ssh_run routing seam — the load-bearing ssh dispatch
# goes through a distinct self._ssh_run, not self._run (git ops)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteRunnerSshRunSeam:
    """The ssh dispatch in run_merge_verify is routed through self._ssh_run, a
    seam parallel to self._run (git ops).  self._ssh_run resolves: injected
    ssh_run -> injected run (back-compat) -> heartbeat default (production)."""

    async def test_ssh_routed_through_ssh_run_happy_path_unchanged(self):
        """run=fake_git only ever sees git argv; ssh_run=fake_ssh receives the
        ['ssh', ..., host, remote_cmd] argv and its VerifyResult JSON is
        returned unchanged — the happy path is byte-identical to today's."""
        git_calls = []
        ssh_calls = []

        async def fake_git(argv, *, cwd=None):
            assert argv[0] == 'git', f'self._run (git-only) received non-git argv: {argv!r}'
            git_calls.append((argv, cwd))
            return (0, '', '')

        expected = VerifyResult(
            passed=True, test_output='all green', lint_output='', type_output='',
            summary='ok',
        )

        async def fake_ssh(argv, *, cwd=None):
            assert argv[0] == 'ssh', f'self._ssh_run received non-ssh argv: {argv!r}'
            ssh_calls.append((argv, cwd))
            return (0, result_to_json(expected), '')

        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
            run=fake_git,
            ssh_run=fake_ssh,
            id_factory=lambda: 'fixed-id',
        )

        result = await runner.run_merge_verify('abc123', _make_spec())

        assert result == expected
        # exactly one ssh dispatch, routed through ssh_run, host + remote_cmd shape intact
        assert len(ssh_calls) == 1
        ssh_argv, _ = ssh_calls[0]
        assert ssh_argv[0] == 'ssh'
        assert ssh_argv[-2] == 'laptop.local'
        # git ops (merge-sha push + best-effort ref cleanup in the finally) went
        # through run, never ssh_run
        assert len(git_calls) == 2
        assert git_calls[0][0] == ['git', 'push', 'origin', 'abc123:refs/merge-verify/fixed-id']
        assert git_calls[1][0] == ['git', 'push', 'origin', '--delete', 'refs/merge-verify/fixed-id']

    async def test_default_construction_wires_distinct_heartbeat_ssh_runner(self):
        """With no run/ssh_run injected, self._ssh_run is a distinct callable
        from self._run (the heartbeat default), asserted via identity only —
        no ssh is ever spawned."""
        runner = gated_remote_runner(
            name='laptop',
            ssh_host='laptop.local',
            git_remote='origin',
            cwd='/repo',
        )
        assert runner._ssh_run is not runner._run


# ---------------------------------------------------------------------------
# Recording event store shared by the event-emitting RemoteRunner tests below
# ---------------------------------------------------------------------------


class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory.

    Mirrors test_merge_verdict_integrity_inv1._RecordingEventStore.
    """

    def __init__(self) -> None:
        self.events: list[tuple[Any, str | None, dict[str, Any]]] = []

    def emit(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        duration_ms: float | None = None,
        **kw: Any,
    ) -> None:
        self.events.append((event_type, task_id, dict(data or {})))

    def events_of(self, event_type: Any) -> list[dict[str, Any]]:
        return [data for (et, _tid, data) in self.events if et == event_type]


# ---------------------------------------------------------------------------
# INV-2 (task 2884): run_merge_verify Step-0 mirror-semantics project-main push
# ---------------------------------------------------------------------------


def _make_mirror_runner(*, ff_rc: int, force_rc: int = 0, resolved_main: str = 'MAINSHA'):
    """RemoteRunner (main_branch='main') + fake_run recorder for the mirror push.

    fake_run routes by argv:
      * ``git rev-parse main``              -> resolved_main (dedup probe)
      * ``git push origin main:refs/heads/main``  (FF)    -> ff_rc
      * ``git push origin +main:refs/heads/main`` (force) -> force_rc
      * ``git push origin <sha>:refs/merge-verify/...``   -> 0 (load-bearing)
      * ``git push origin --delete <ref>``                -> 0 (cleanup)
      * ssh                                               -> canned PASS
    Returns (runner, calls, expected_result).
    """
    calls: list[tuple[list[str], Any]] = []
    expected = VerifyResult(passed=True, test_output='ok', lint_output='', type_output='', summary='ok')

    async def fake_run(argv, *, cwd=None):
        calls.append((list(argv), cwd))
        if argv[:2] == ['git', 'rev-parse'] and len(argv) > 2 and argv[2] == 'main':
            return (0, resolved_main, '')
        if argv[:2] == ['git', 'push'] and len(argv) > 3:
            refspec = argv[3]
            if refspec == 'main:refs/heads/main':
                return (ff_rc, '', '' if ff_rc == 0 else 'rejected: non-fast-forward')
            if refspec == '+main:refs/heads/main':
                return (force_rc, '', '' if force_rc == 0 else 'rejected: hook declined')
            if 'refs/merge-verify/' in refspec:
                return (0, '', '')
            if refspec == '--delete':
                return (0, '', '')
            return (0, '', '')
        return (0, result_to_json(expected), '')

    runner = gated_remote_runner(
        name='laptop',
        ssh_host='laptop.local',
        git_remote='origin',
        cwd='/repo',
        main_branch='main',
        run=fake_run,
        id_factory=lambda: 'fixed-id',
    )
    return runner, calls, expected


def _push_refspecs(calls) -> list[str]:
    """Every `git push` refspec argument seen, in order."""
    return [
        argv[3] for (argv, _cwd) in calls
        if argv[:2] == ['git', 'push'] and len(argv) > 3
    ]


@pytest.mark.asyncio
class TestRemoteRunnerMainPushMirror:
    """run_merge_verify Step-0: on a non-FF main push, force-mirror + emit runner_synced."""

    async def test_ff_failure_retries_with_force_and_emits_project_main_mirror(self):
        """(a) FF main push rc!=0 -> a `+main:refs/heads/main` force push follows and a
        runner_synced(kind='project_main_mirror', forced=True) is emitted; the merge-sha
        push + ssh still run and the VerifyResult is returned."""
        runner, calls, expected = _make_mirror_runner(ff_rc=1, force_rc=0)
        store = _RecordingEventStore()
        result = await runner.run_merge_verify('abc123', _make_spec(), event_store=store)
        assert result == expected

        refspecs = _push_refspecs(calls)
        # FF attempt precedes the force attempt
        assert 'main:refs/heads/main' in refspecs
        assert '+main:refs/heads/main' in refspecs
        assert refspecs.index('main:refs/heads/main') < refspecs.index('+main:refs/heads/main')
        # merge-sha push still issued (load-bearing transport intact)
        assert any('refs/merge-verify/' in r for r in refspecs)

        synced = store.events_of(EventType.runner_synced)
        mirror = [e for e in synced if e.get('kind') == 'project_main_mirror']
        assert len(mirror) == 1
        assert mirror[0]['forced'] is True
        assert mirror[0]['runner'] == 'laptop'

    async def test_ff_success_no_force_no_mirror_event(self):
        """(b) FF main push rc==0 -> no force push and no project_main_mirror event."""
        runner, calls, expected = _make_mirror_runner(ff_rc=0)
        store = _RecordingEventStore()
        result = await runner.run_merge_verify('abc123', _make_spec(), event_store=store)
        assert result == expected
        refspecs = _push_refspecs(calls)
        assert '+main:refs/heads/main' not in refspecs
        mirror = [e for e in store.events_of(EventType.runner_synced) if e.get('kind') == 'project_main_mirror']
        assert mirror == []

    async def test_ff_and_force_both_fail_is_non_fatal_no_event(self):
        """(c) FF rc!=0 AND force rc!=0 -> no raise, no event, best-effort swallow;
        merge-sha push still issued and the result is returned."""
        runner, calls, expected = _make_mirror_runner(ff_rc=1, force_rc=1)
        store = _RecordingEventStore()
        result = await runner.run_merge_verify('abc123', _make_spec(), event_store=store)  # must not raise
        assert result == expected
        refspecs = _push_refspecs(calls)
        assert '+main:refs/heads/main' in refspecs  # force was attempted
        assert any('refs/merge-verify/' in r for r in refspecs)  # merge-sha push still happened
        mirror = [e for e in store.events_of(EventType.runner_synced) if e.get('kind') == 'project_main_mirror']
        assert mirror == []  # force failed -> no success event

    async def test_event_store_none_is_safe_on_force_path(self):
        """event_store=None on the force path must not raise (None-safe emit)."""
        runner, calls, expected = _make_mirror_runner(ff_rc=1, force_rc=0)
        result = await runner.run_merge_verify('abc123', _make_spec(), event_store=None)
        assert result == expected
        assert '+main:refs/heads/main' in _push_refspecs(calls)
