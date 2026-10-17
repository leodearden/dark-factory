"""The governed exception lists of fused-memory/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

from __future__ import annotations

from pathlib import Path

from _fm_helpers import load_script_module
from shared.governed_exceptions import Debt, TaskRef, governed_exceptions

from fused_memory.memory_metadata import BLESSED_METADATA_KEYS

_checker = load_script_module(
    Path(__file__).resolve().parents[1] / 'scripts' / 'check_bare_magicmock_config.py',
    mod_name='check_bare_magicmock_config',
)

BLESSED_MEMORY_METADATA_KEYS = governed_exceptions(
    'fused_memory.memory_metadata.BLESSED_METADATA_KEYS',
    'A high-volume conventional memory-metadata key outside the reserved vocabulary that '
    'the metadata census accepts without an unknown-key warning.',
    sorted(BLESSED_METADATA_KEYS),
    default=Debt(TaskRef(5603)),
    default_covers=8,
)

WALL_CLOCK_DEADLINE_BASELINE = governed_exceptions(
    'fused_memory_scripts.check_bare_magicmock_config._WALL_CLOCK_DEADLINE_DEBT',
    'Per-file budgets of wall-clock-deadline (Rule C) violations grandfathered when the '
    'rule was adopted; each budget may only fall.',
    ['fused-memory/scripts/check_bare_magicmock_config.py::_WALL_CLOCK_DEADLINE_DEBT'],
    default=Debt(TaskRef(5603)),
    default_covers=1,
)


def test_the_wall_clock_baseline_unit_still_has_a_baseline() -> None:
    assert getattr(_checker, '_WALL_CLOCK_DEADLINE_DEBT', None), (
        'check_bare_magicmock_config._WALL_CLOCK_DEADLINE_DEBT has been burned down or '
        'deleted, so WALL_CLOCK_DEADLINE_BASELINE (its key and list_id) must be deleted '
        'with it.'
    )
