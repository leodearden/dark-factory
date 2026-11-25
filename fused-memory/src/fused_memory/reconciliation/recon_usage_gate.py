"""The reconciliation subsystem's single usage gate, shared by its stages, judge and targeted verifier."""

from shared.usage_gate import UsageGate

from fused_memory.config.schema import ReconciliationConfig


def build_reconciliation_usage_gate(config: ReconciliationConfig) -> UsageGate | None:
    if hasattr(config, 'usage_cap') and config.usage_cap.enabled:
        return UsageGate(config.usage_cap)
    return None
