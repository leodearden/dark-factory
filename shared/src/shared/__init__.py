"""dark-factory-shared — public API surface."""

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
    invoke_claude_agent,
    invoke_with_cap_retry,
    is_timed_out_with_progress,
    is_zero_output_timeout,
    read_transcript_records,
)
from shared.config_models import AccountConfig, UsageCapConfig
from shared.cost_store import CostStore
from shared.locking import files_to_modules, modules_conflict, normalize_lock
from shared.sqlite_sync_base import apply_full_durability_pragmas_sync
from shared.usage_gate import AccountState, InvokeSlot, SessionBudgetExhausted, UsageGate

__version__ = '0.1.0'

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
    'invoke_claude_agent',
    'invoke_with_cap_retry',
    'is_timed_out_with_progress',
    'is_zero_output_timeout',
    'read_transcript_records',
    'AccountConfig',
    'UsageCapConfig',
    'CostStore',
    'UsageGate',
    'AccountState',
    'InvokeSlot',
    'SessionBudgetExhausted',
    'normalize_lock',
    'files_to_modules',
    'modules_conflict',
]
