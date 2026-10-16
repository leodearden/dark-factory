"""The governed exception lists of dashboard/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

from __future__ import annotations

from shared.governed_exceptions import Debt, TaskRef, governed_exceptions

from dashboard.config import _LEGACY_CONFIG_NAMES

LEGACY_CONFIG_NAMES = governed_exceptions(
    'dashboard.config._LEGACY_CONFIG_NAMES',
    'A legacy orchestrator-config filename that dashboard discovery still honours as a '
    'fallback after the canonical dark-factory-orchestrator.yaml, for a project not yet '
    'migrated to the canonical name.',
    list(_LEGACY_CONFIG_NAMES),
    default=Debt(TaskRef(5603)),
    default_covers=3,
)
