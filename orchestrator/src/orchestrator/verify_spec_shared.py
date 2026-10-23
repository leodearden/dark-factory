"""The spec-agnostic vocabulary every verify spec is built from and applied with.

``verify_runner`` (the merge spec and the transport) and ``task_verify_spec``
(the task spec) both import this module; it imports neither, so the two specs
share one home for each rule without a cycle between them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from orchestrator.config import ModuleConfig, OrchestratorConfig


@dataclass(frozen=True)
class VerifyCommand:
    """Frozen projection of a (scoped) ModuleConfig's command-bearing fields.

    Mirrors ``ModuleConfig.prefix`` + the three command fields.  Timeouts and
    environment variables live at spec level (shared across the bundle), so
    they are NOT duplicated here.
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

    @classmethod
    def from_module_config(cls, mc: ModuleConfig) -> VerifyCommand:
        return cls(
            prefix=mc.prefix,
            test_command=mc.test_command,
            lint_command=mc.lint_command,
            type_check_command=mc.type_check_command,
        )


def global_verify_command_for(
    config: OrchestratorConfig,
    verify_commands: Sequence[VerifyCommand],
) -> VerifyCommand | None:
    """The dispatching side's global full-gate commands, for a zero-module spec.

    INV-1, task 2883: a spec with no verify_commands would otherwise leave the
    consuming host to fall back to its OWN (possibly stale) config globals.
    Shipped only when the scope resolves to zero modules AND a global command
    exists; non-empty verify_commands drive the gate, so this is None.
    """
    if verify_commands:
        return None
    if not (config.test_command or config.lint_command or config.type_check_command):
        return None
    return VerifyCommand(
        prefix='',
        test_command=config.test_command,
        lint_command=config.lint_command,
        type_check_command=config.type_check_command,
    )


def module_config_from_command(
    vc: VerifyCommand,
    *,
    verify_env: Mapping[str, str] | None = None,
    cold_timeout_secs: float | None = None,
) -> ModuleConfig:
    """Rebuild a ModuleConfig on the consuming host from one VerifyCommand.

    Fields the wire never carries (lock_depth, the warm timeout, ...) stay at
    their dataclass defaults.  *verify_env* is copied, so the ModuleConfig
    never aliases the caller's mapping.
    """
    return ModuleConfig(
        prefix=vc.prefix,
        test_command=vc.test_command,
        lint_command=vc.lint_command,
        type_check_command=vc.type_check_command,
        verify_env=dict(verify_env) if verify_env is not None else None,
        verify_cold_command_timeout_secs=cold_timeout_secs,
    )


def apply_spec_to_host_config(
    config: OrchestratorConfig,
    *,
    verify_env: Mapping[str, str],
    global_verify_command: VerifyCommand | None,
    module_configs: Sequence[ModuleConfig],
    field_overrides: Mapping[str, Any],
    verify_cold_preprovision_command: str,
) -> OrchestratorConfig:
    """Return a copy of the consuming host's *config* with a spec applied onto it.

    The spec is dispatcher-shaped and the host config is host-shaped; the copy
    carries both:

    - verify_env (task 5496): ``{**config.effective_verify_env, **verify_env}``
      — the spec wins on conflict (these keys select tests) and host-only keys
      survive (a remote host's own local necessities, which no dispatcher-built
      spec can know).  Unconditional: an empty spec env leaves every host key.
      It is the rule ``verify.py::_resolve_verify_env`` already applies per
      module, made uniform for the zero-module path.
    - global commands (INV-1, task 2883): overwrite the host's
      test/lint/type_check_command only when *global_verify_command* is set.
    - module registry (task 4536): exactly *module_configs*, ``{}`` included,
      so ``module_configs_or_empty`` and the module list a caller passes on
      cannot disagree and the host's own discovery walk never decides.
      Accepted consequence: ``verify.py::_reverse_dependency_module_configs``
      (task 2607) cannot widen to a dependent outside the spec's set on this
      leg; it logs the unresolvable dependent instead.
    - *field_overrides*: the spec's verdict-deciding profile fields, each
      checked strictly against its declared OrchestratorConfig type first,
      because ``model_copy(update=...)`` validates nothing: an unknown name
      or a mistyped value raises TypeError rather than deciding the run.
    - *verify_cold_preprovision_command*: non-empty replaces the host's, ''
      keeps the host's own, under the same strict check as *field_overrides*.
      '' is both "the dispatcher has none" and what a pre-6127 merge spec
      decodes to, so a host-side value (leo-laptop's 2026-10-01 operational
      fix is the live case) is never wiped.  PRD:
      plans/remote-task-leg-verify-prd.md §4.1.

    Two spelling hazards, both because ``model_copy`` carries field and
    private-attribute VALUES over by reference: verify_env is a FRESH dict
    (never an in-place update), and ``_module_configs`` is REBOUND on the copy
    (never mutated, and never routed through ``model_copy(update=...)``, which
    would shadow the PrivateAttr in ``__dict__``).  Either mistake corrupts
    the caller's config — the one cli.py loaded and may still use.
    """
    overrides = dict(field_overrides)
    if verify_cold_preprovision_command != '':
        overrides['verify_cold_preprovision_command'] = verify_cold_preprovision_command
    update: dict[str, Any] = {
        **_validated_field_overrides(overrides),
        'verify_env': {**config.effective_verify_env, **verify_env},
    }
    if global_verify_command is not None:
        update['test_command'] = global_verify_command.test_command
        update['lint_command'] = global_verify_command.lint_command
        update['type_check_command'] = global_verify_command.type_check_command
    applied = config.model_copy(update=update)
    applied._module_configs = {mc.prefix: mc for mc in module_configs}
    return applied


def _validated_field_overrides(field_overrides: Mapping[str, Any]) -> dict[str, Any]:
    fields = OrchestratorConfig.model_fields
    unknown = sorted(set(field_overrides) - set(fields))
    if unknown:
        raise TypeError(f'field_overrides name no OrchestratorConfig field: {unknown}')
    validated: dict[str, Any] = {}
    for name, value in field_overrides.items():
        adapter = TypeAdapter(fields[name].rebuild_annotation())
        try:
            validated[name] = adapter.validate_python(value, strict=True)
        except ValidationError as exc:
            raise TypeError(
                f'field_overrides[{name!r}] = {value!r} is not a valid '
                f'OrchestratorConfig.{name}'
            ) from exc
    return validated


def canonical_json(payload: Any) -> str:
    """The one wire encoding: sorted keys, non-ASCII kept literal, so equal values encode to equal bytes."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def require_exact_keys(d: object, expected: frozenset[str], type_name: str) -> None:
    """Raise TypeError unless *d* is a dict whose keys are exactly *expected*."""
    if not isinstance(d, dict):
        raise TypeError(
            f'{type_name} wire document must be a JSON object, got {type(d).__name__}'
        )
    unknown = sorted(set(d) - expected)
    missing = sorted(expected - set(d))
    if unknown or missing:
        raise TypeError(
            f'{type_name} wire document has unknown keys {unknown} '
            f'and missing keys {missing}'
        )
