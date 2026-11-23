"""A calibration of ``_link_heal_harness``'s fake adjudicator, recorded in a link-heal ledger.

A writing sweep is gated on a passing calibration of the adjudicator's exact model
and prompt, so a test that lets a sweep write records one here first.
"""

from __future__ import annotations

from _link_heal_harness import FAKE_ADJUDICATOR_MODEL, FAKE_PROMPT_SHA256

from fused_memory.maintenance.link_heal_ledger import LinkHealLedger, RunSource


def record_calibration(
    ledger: LinkHealLedger, *, passed: bool = True, prompt_sha256: str = FAKE_PROMPT_SHA256,
) -> str:
    """Record a calibration run of the fake adjudicator at *prompt_sha256*; its run id."""
    run_id = ledger.start_run(RunSource.CALIBRATION, writes=False)
    ledger.add_calibration(
        run_id, model=FAKE_ADJUDICATOR_MODEL, prompt_sha256=prompt_sha256,
        population_sha256='0' * 64, passed=passed, report={'passed': passed},
    )
    return run_id
