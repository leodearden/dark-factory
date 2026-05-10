"""Per-project reconciliation policies.

Each sub-module contains project-specific guardrail configuration that would
otherwise pollute shared reconciliation infrastructure.  The policy modules
expose only public names so callers don't need to cross private-name boundaries.
"""
