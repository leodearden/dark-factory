"""dark-factory-shared — public API surface."""

from shared.cli_invoke import AgentResult, invoke_claude_agent, invoke_with_cap_retry
from shared.config_models import AccountConfig, UsageCapConfig
from shared.cost_store import CostStore
from shared.usage_gate import AccountState, SessionBudgetExhausted, UsageGate

__version__ = '0.1.0'

__all__ = [
    'AgentResult',
    'invoke_claude_agent',
    'invoke_with_cap_retry',
    'AccountConfig',
    'UsageCapConfig',
    'CostStore',
    'UsageGate',
    'AccountState',
    'SessionBudgetExhausted',
]
