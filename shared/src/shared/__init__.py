"""dark-factory-shared — public API surface."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-only re-exports: this block is for pyright/IDE resolution of
    # `from shared import UsageGate` and for ruff's re-export detection (the
    # static `__all__` literal below is what marks them as re-exports rather
    # than unused imports). TYPE_CHECKING is False at runtime, so none of it
    # executes — the names are served by `__getattr__` instead.
    #
    # Importing them eagerly here would run on EVERY `import shared.<anything>`
    # (Python executes a package's __init__ before any submodule), dragging
    # pydantic, yaml, aiosqlite, dotenv and aiohttp into stdlib-only consumers
    # such as `shared.locking` and `shared.task_claimant`.
    from shared.agent_result import AgentVerdict, extract_agent_verdict
    from shared.async_sqlite_base import (
        AsyncSqliteBase,
        CheckpointResult,
        apply_full_durability_pragmas,
        apply_wal_pragmas,
        connect_daemon,
    )
    from shared.cli_invoke import (
        CAP_HIT_RESUME_PROMPT,
        CRASH_RECOVERY_RESUME_PROMPT,
        AgentFailureClass,
        AgentFailureKind,
        AgentResult,
        AllAccountsCappedException,
        build_failure_message,
        classify_agent_failure,
        count_transcript_turns,
        detect_ended_awaiting_background,
        ended_awaiting_background_for_session,
        invoke_claude_agent,
        invoke_with_cap_retry,
        is_cli_invocation_rejected,
        is_server_error_status,
        is_timed_out_with_progress,
        is_zero_output_timeout,
        read_transcript_records,
        require_non_blank_prompt,
        transcript_exists,
    )
    from shared.config_models import AccountConfig, UsageCapConfig
    from shared.cost_store import CostStore
    from shared.locking import (
        EXTENSIONLESS_FILENAMES,
        FILE_EXTENSIONS,
        directory_locks,
        files_to_modules,
        is_file_path,
        modules_conflict,
        normalize_lock,
        strip_directory_locks,
    )
    from shared.mcp_idempotency import MUTATING_TASK_TOOLS, maybe_inject_client_op_id
    from shared.safe_io import atomic_write_text, load_json_or_warn
    from shared.sqlite_sync_base import apply_full_durability_pragmas_sync
    from shared.usage_gate import (
        AccountPhase,
        AccountState,
        IllegalTransitionError,
        InvokeSlot,
        SessionBudgetExhausted,
        UsageGate,
    )

__version__ = '0.1.0'

#: Single source of truth for lazy resolution: public name -> owning submodule
#: (without the `shared.` prefix). Derived one-for-one from the TYPE_CHECKING
#: block above; the two must stay in step, which
#: shared/tests/test_lazy_public_api.py asserts.
_SYMBOL_MODULE: dict[str, str] = {
    # shared.agent_result
    'AgentVerdict': 'agent_result',
    'extract_agent_verdict': 'agent_result',
    # shared.async_sqlite_base
    'AsyncSqliteBase': 'async_sqlite_base',
    'CheckpointResult': 'async_sqlite_base',
    'apply_full_durability_pragmas': 'async_sqlite_base',
    'apply_wal_pragmas': 'async_sqlite_base',
    'connect_daemon': 'async_sqlite_base',
    # shared.cli_invoke
    'CAP_HIT_RESUME_PROMPT': 'cli_invoke',
    'CRASH_RECOVERY_RESUME_PROMPT': 'cli_invoke',
    'AgentFailureClass': 'cli_invoke',
    'AgentFailureKind': 'cli_invoke',
    'AgentResult': 'cli_invoke',
    'AllAccountsCappedException': 'cli_invoke',
    'build_failure_message': 'cli_invoke',
    'classify_agent_failure': 'cli_invoke',
    'count_transcript_turns': 'cli_invoke',
    'detect_ended_awaiting_background': 'cli_invoke',
    'ended_awaiting_background_for_session': 'cli_invoke',
    'invoke_claude_agent': 'cli_invoke',
    'invoke_with_cap_retry': 'cli_invoke',
    'is_cli_invocation_rejected': 'cli_invoke',
    'is_server_error_status': 'cli_invoke',
    'is_timed_out_with_progress': 'cli_invoke',
    'is_zero_output_timeout': 'cli_invoke',
    'read_transcript_records': 'cli_invoke',
    'require_non_blank_prompt': 'cli_invoke',
    'transcript_exists': 'cli_invoke',
    # shared.config_models
    'AccountConfig': 'config_models',
    'UsageCapConfig': 'config_models',
    # shared.cost_store
    'CostStore': 'cost_store',
    # shared.locking
    'FILE_EXTENSIONS': 'locking',
    'EXTENSIONLESS_FILENAMES': 'locking',
    'directory_locks': 'locking',
    'files_to_modules': 'locking',
    'is_file_path': 'locking',
    'modules_conflict': 'locking',
    'normalize_lock': 'locking',
    'strip_directory_locks': 'locking',
    # shared.mcp_idempotency
    'MUTATING_TASK_TOOLS': 'mcp_idempotency',
    'maybe_inject_client_op_id': 'mcp_idempotency',
    # shared.safe_io
    'load_json_or_warn': 'safe_io',
    'atomic_write_text': 'safe_io',
    # shared.sqlite_sync_base
    'apply_full_durability_pragmas_sync': 'sqlite_sync_base',
    # shared.usage_gate
    'AccountPhase': 'usage_gate',
    'AccountState': 'usage_gate',
    'IllegalTransitionError': 'usage_gate',
    'InvokeSlot': 'usage_gate',
    'SessionBudgetExhausted': 'usage_gate',
    'UsageGate': 'usage_gate',
}


#: Submodules reachable as package attributes (`import shared; shared.locking`).
#: A behaviour-PRESERVATION set, not a wish list: it is precisely the set the
#: pre-lazy eager block bound, so `shared.usage_gate` keeps working while
#: `shared.psi` keeps raising AttributeError exactly as it did before. An
#: unrestricted fallback would silently widen the public surface; an empty one
#: would silently narrow it. Widening it later must be an explicit edit.
_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {
        'agent_result',
        'async_sqlite_base',
        'cli_invoke',
        'config_models',
        'cost_store',
        'locking',
        'mcp_idempotency',
        'safe_io',
        'sqlite_sync_base',
        'usage_gate',
    }
)


def __getattr__(name: str) -> object:
    """PEP 562 lazy attribute access — import the owning submodule on demand.

    Raises AttributeError (never ImportError/KeyError) for an unknown name:
    that is what the protocol requires and what `hasattr`/`getattr` callers,
    and a typo'd `from shared import ...`, expect to see.
    """
    module_name = _SYMBOL_MODULE.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(f'shared.{module_name}'), name)
    elif name in _LAZY_SUBMODULES:
        value = importlib.import_module(f'shared.{name}')
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    # Cache into the module dict so subsequent lookups bypass __getattr__
    # entirely, rather than paying an importlib lookup per access.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Report the whole public surface, not merely what has been resolved yet."""
    return sorted({*globals(), *_SYMBOL_MODULE, *_LAZY_SUBMODULES})


# Static list literal, deliberately NOT derived from _SYMBOL_MODULE: ruff needs
# the literal to treat the TYPE_CHECKING imports as re-exports (it cannot
# evaluate `list(_SYMBOL_MODULE)`, so all 47 would become F401), and
# test_public_api.py::TestInitAllCompleteness compares it to the union of the
# submodule __all__s.
__all__ = [
    'AgentVerdict',
    'extract_agent_verdict',
    'AsyncSqliteBase',
    'CheckpointResult',
    'apply_wal_pragmas',
    'apply_full_durability_pragmas',
    'connect_daemon',
    'apply_full_durability_pragmas_sync',
    'CAP_HIT_RESUME_PROMPT',
    'CRASH_RECOVERY_RESUME_PROMPT',
    'AgentFailureClass',
    'AgentFailureKind',
    'AgentResult',
    'AllAccountsCappedException',
    'build_failure_message',
    'classify_agent_failure',
    'count_transcript_turns',
    'detect_ended_awaiting_background',
    'ended_awaiting_background_for_session',
    'invoke_claude_agent',
    'invoke_with_cap_retry',
    'is_cli_invocation_rejected',
    'is_server_error_status',
    'is_timed_out_with_progress',
    'is_zero_output_timeout',
    'read_transcript_records',
    'require_non_blank_prompt',
    'transcript_exists',
    'AccountConfig',
    'UsageCapConfig',
    'CostStore',
    'UsageGate',
    'AccountState',
    'AccountPhase',
    'InvokeSlot',
    'SessionBudgetExhausted',
    'IllegalTransitionError',
    'normalize_lock',
    'files_to_modules',
    'modules_conflict',
    'FILE_EXTENSIONS',
    'EXTENSIONLESS_FILENAMES',
    'is_file_path',
    'directory_locks',
    'strip_directory_locks',
    'MUTATING_TASK_TOOLS',
    'maybe_inject_client_op_id',
    'load_json_or_warn',
    'atomic_write_text',
]
