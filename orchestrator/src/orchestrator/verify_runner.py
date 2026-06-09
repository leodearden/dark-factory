"""verify_runner — host-independent MergeVerifySpec + VerifyResult JSON codec.

This module defines the data-contract types for multi-host merge verification
(PRD: plans/merge-throughput-multihost-verify-prd.md §A) and the wire codec
that serialises/deserialises them as byte-identical JSON documents.

Public API
----------
Dataclasses (frozen):
  VerifyCommand          — one module's command-bearing fields (mirrors ModuleConfig)
  UnscopedTypecheckSpec  — _run_unscoped_typechecks gate inputs
  MergeVerifySpec        — full pre-advance merge-verify bundle (PRD §A contract)

Wire codec (module-level functions):
  spec_to_json(spec)     -> str   — canonical JSON for the spec document
  spec_from_json(s)      -> MergeVerifySpec
  result_to_json(vr)     -> str   — canonical JSON for a VerifyResult document
  result_from_json(s)    -> VerifyResult
  result_to_dict(vr)     -> dict  — helper used by result_to_json
  result_from_dict(d)    -> VerifyResult

Byte-identity guarantee (PRD Invariant 1)
-----------------------------------------
All JSON is emitted with ``json.dumps(sort_keys=True, ensure_ascii=False)``.
This canonicalises dict key order so that:
  - ``to_json(from_json(s)) == s``  (re-serialisation is identical)
  - The same object always serialises to the same bytes regardless of dict
    construction order (proved by sort_keys).
Achievability is by construction: every field maps to native JSON scalar /
container types only (str, bool, float, list, dict, None); no float-precision
rounding or numeric tolerance is involved.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass

from orchestrator.verify import VerifyResult

__all__ = [
    "VerifyCommand",
    "UnscopedTypecheckSpec",
    "MergeVerifySpec",
    "spec_to_json",
    "spec_from_json",
    "result_to_json",
    "result_from_json",
    "result_to_dict",
    "result_from_dict",
]


# ---------------------------------------------------------------------------
# VerifyCommand
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyCommand:
    """Frozen projection of a (scoped) ModuleConfig's command-bearing fields.

    Mirrors ``ModuleConfig.prefix`` + the three command fields.  Timeouts and
    environment variables live at ``MergeVerifySpec`` level (shared across the
    bundle), so they are NOT duplicated here.
    """

    prefix: str
    test_command: str | None = None
    lint_command: str | None = None
    type_check_command: str | None = None

    def to_dict(self) -> dict:
        return {
            "prefix": self.prefix,
            "test_command": self.test_command,
            "lint_command": self.lint_command,
            "type_check_command": self.type_check_command,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerifyCommand:
        return cls(
            prefix=d["prefix"],
            test_command=d.get("test_command"),
            lint_command=d.get("lint_command"),
            type_check_command=d.get("type_check_command"),
        )


# ---------------------------------------------------------------------------
# UnscopedTypecheckSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnscopedTypecheckSpec:
    """Inputs for the ``_run_unscoped_typechecks`` pre-advance gate.

    ``commands`` holds one ``VerifyCommand`` per module; only
    ``type_check_command`` is meaningful here (the other fields are carried
    along for consistent structure but are not used by the gate).

    ``block_on_timeout=True`` (the default) mirrors the fail-closed policy
    used at ``merge_queue.py:1565``; set to ``False`` only in tests or when
    the caller explicitly opts out of fail-closed behaviour.
    """

    commands: tuple[VerifyCommand, ...]
    block_on_timeout: bool = True

    def to_dict(self) -> dict:
        return {
            "commands": [vc.to_dict() for vc in self.commands],
            "block_on_timeout": self.block_on_timeout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UnscopedTypecheckSpec:
        return cls(
            commands=tuple(VerifyCommand.from_dict(vc) for vc in d["commands"]),
            block_on_timeout=d.get("block_on_timeout", True),
        )


# ---------------------------------------------------------------------------
# MergeVerifySpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeVerifySpec:
    """Full pre-advance merge-verify bundle — PRD §A contract.

    Fields
    ------
    verify_commands     : one VerifyCommand per module, scoped to task_files
    unscoped_typecheck  : the _run_unscoped_typechecks gate spec
    task_files          : files in the merge commit (None → full verify)
    verify_env          : environment overrides (RUSTC_WRAPPER, CARGO_INCREMENTAL, …)
    cold_timeout_secs   : merge_verify_cold cascade timeout
    is_merge_verify     : always True for merge-path specs (default)

    Note
    ----
    ``verify_env`` is stored as a mutable ``dict`` at runtime.  Although
    ``frozen=True`` prevents attribute reassignment, calling ``hash()`` on a
    ``MergeVerifySpec`` instance raises ``TypeError`` because dicts are
    unhashable.  Specs are serialization value-objects and are not intended to
    be used as dict keys or set members.
    """

    verify_commands: tuple[VerifyCommand, ...]
    unscoped_typecheck: UnscopedTypecheckSpec
    task_files: tuple[str, ...] | None
    verify_env: Mapping[str, str]
    cold_timeout_secs: float
    is_merge_verify: bool = True

    def to_dict(self) -> dict:
        return {
            "verify_commands": [vc.to_dict() for vc in self.verify_commands],
            "unscoped_typecheck": self.unscoped_typecheck.to_dict(),
            "task_files": list(self.task_files) if self.task_files is not None else None,
            "verify_env": dict(self.verify_env),
            "cold_timeout_secs": float(self.cold_timeout_secs),
            "is_merge_verify": self.is_merge_verify,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MergeVerifySpec:
        task_files_raw = d.get("task_files")
        return cls(
            verify_commands=tuple(VerifyCommand.from_dict(vc) for vc in d["verify_commands"]),
            unscoped_typecheck=UnscopedTypecheckSpec.from_dict(d["unscoped_typecheck"]),
            task_files=tuple(task_files_raw) if task_files_raw is not None else None,
            verify_env=dict(d.get("verify_env", {})),
            cold_timeout_secs=float(d["cold_timeout_secs"]),
            is_merge_verify=d.get("is_merge_verify", True),
        )


# ---------------------------------------------------------------------------
# VerifyResult codec
# ---------------------------------------------------------------------------


def result_to_dict(vr: VerifyResult) -> dict:
    """Serialise a VerifyResult to a plain dict of JSON-native types.

    Uses ``dataclasses.asdict`` which recursively converts nested dataclasses
    and preserves all field types (all VerifyResult fields are JSON-native).
    """
    return dataclasses.asdict(vr)


def result_from_dict(d: dict) -> VerifyResult:
    """Reconstruct a VerifyResult from a dict (as produced by result_to_dict)."""
    return VerifyResult(**d)


# ---------------------------------------------------------------------------
# Module-level wire codec
# ---------------------------------------------------------------------------


def spec_to_json(spec: MergeVerifySpec) -> str:
    """Serialise a MergeVerifySpec to canonical JSON (sort_keys=True)."""
    return json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False)


def spec_from_json(s: str) -> MergeVerifySpec:
    """Deserialise a MergeVerifySpec from a JSON string."""
    return MergeVerifySpec.from_dict(json.loads(s))


def result_to_json(vr: VerifyResult) -> str:
    """Serialise a VerifyResult to canonical JSON (sort_keys=True)."""
    return json.dumps(result_to_dict(vr), sort_keys=True, ensure_ascii=False)


def result_from_json(s: str) -> VerifyResult:
    """Deserialise a VerifyResult from a JSON string."""
    return result_from_dict(json.loads(s))
