"""Build the reconciliation subsystem's usage gate from its config."""

from shared.usage_gate import UsageGate

from fused_memory.config.schema import ReconciliationConfig


def build_reconciliation_usage_gate(config: ReconciliationConfig) -> UsageGate | None:
    return UsageGate(config.usage_cap) if config.usage_cap.enabled else None
