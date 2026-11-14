"""The committed write-triage calibration corpora, registered once.

reify's corpus is the pooled one. Every other project's was read off its own
live store. Registering a corpus here puts it under every contract that
reads this module: the per-record shape contract in
``test_calibrate_write_triage.py``, and the live-store contract, the
cross-corpus disjointness check and the committed-artifact bindings in
``test_calibrate_write_triage_projects.py``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from fused_memory.models.enums import MEM0_PRIMARY

PACKAGE_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
REIFY_FIXTURE_PATH = FIXTURES_DIR / 'write_triage_calibration.jsonl'
DARK_FACTORY_FIXTURE_PATH = FIXTURES_DIR / 'write_triage_calibration_dark_factory.jsonl'
DARK_FACTORY_ADMITTED_CATEGORIES = frozenset({'preferences_and_norms', 'observations_and_summaries'})


@dataclass(frozen=True)
class LiveStoreFixture:
    """A per-project corpus read off its live store, and the categories it admits."""

    project_id: str
    path: Path
    admitted_categories: frozenset[str]

    @property
    def report_path(self) -> Path:
        return PACKAGE_ROOT / 'calibration' / f'write_triage_calibration_report_{self.project_id}.json'

    @property
    def package_relative_path(self) -> str:
        return self.path.relative_to(PACKAGE_ROOT).as_posix()


LIVE_STORE_FIXTURES = (
    LiveStoreFixture('dark_factory', DARK_FACTORY_FIXTURE_PATH, DARK_FACTORY_ADMITTED_CATEGORIES),
    LiveStoreFixture(
        'solar_challenge',
        FIXTURES_DIR / 'write_triage_calibration_solar_challenge.jsonl',
        MEM0_PRIMARY,
    ),
    LiveStoreFixture(
        'solar_challenge_platform',
        FIXTURES_DIR / 'write_triage_calibration_solar_challenge_platform.jsonl',
        MEM0_PRIMARY,
    ),
)

#: Every committed calibration corpus, keyed on the project it was labelled for.
CALIBRATION_FIXTURES: Mapping[str, Path] = MappingProxyType({
    'reify': REIFY_FIXTURE_PATH,
    **{fixture.project_id: fixture.path for fixture in LIVE_STORE_FIXTURES},
})
