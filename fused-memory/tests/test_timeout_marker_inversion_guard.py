"""The timeout-marker inversion guard, instantiated for fused-memory/tests.

The extractor, the band, the failure messages and their rationale live in
shared/src/shared/testing_timeout_markers.py.  This module wires in this
package's tests root, the verify budget its own orchestrator.yaml passes, its
grandfather allowlist and its anti-vacuity floors.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import yaml
from shared.testing_timeout_markers import (
    DELIBERATE_TIGHT_BOUND_CEILING,
    GrandfatherRatchet,
    TimeoutSite,
    TreeScan,
    grandfather_ratchet,
    inversion_failure_message,
    inverts,
    scan_python_tree,
    stale_grandfather_message,
    timeout_marker_sites,
    verify_cli_timeout,
)

# One xdist worker pays for the cached sweep once, under -n auto --dist loadgroup.
pytestmark = pytest.mark.xdist_group('timeout_marker_inversion_guard')

_TESTS_DIR = Path(__file__).resolve().parent
_ORCH_YAML = _TESTS_DIR.parent / 'orchestrator.yaml'

#: No named timeout constant is shared across this package's modules, so a
#: marker spelled as a NAME reads as "no opinion".  The guard is a floor, with
#: the same stated limit as orchestrator's.
_SANCTIONED: Mapping[str, float] = {}

#: In-band sites that predate this guard: 20 across 13 modules, MEASURED on
#: 2026-10-07 at main 1c6a52dbfd.  Entries may only ever be REMOVED.  Migrating
#: them was deferred on task 5147's reasoning (rewriting 13 modules would take
#: locks across the package); follow-up tkt_0RVDQ479918XDV0C80NXAVT8HX.  Keyed
#: per SITE on the path RELATIVE to tests/, so a nested module cannot inherit a
#: top-level twin's exemption.
_GRANDFATHERED: frozenset[tuple[str, str]] = frozenset({
    # server/test_write_triage_judge.py -- 2 sites at 120s
    ('server/test_write_triage_judge.py', 'TestTheFrontierArmLive'),
    ('server/test_write_triage_judge.py', 'TestTheShippedWordingLive'),
    # test_bm25_serving_integration.py -- 1 site at 120s
    ('test_bm25_serving_integration.py', '<module>'),
    # test_drop_vector_indices_integration.py -- 1 site at 240s
    ('test_drop_vector_indices_integration.py', 'TestDropRebuildWindow'),
    # test_ensure_indices_integration.py -- 1 site at 120s
    ('test_ensure_indices_integration.py', '<module>'),
    # test_falkor_edge_search_integration.py -- 1 site at 120s
    ('test_falkor_edge_search_integration.py', '<module>'),
    # test_index_provisioning_wiring_integration.py -- 1 site at 120s
    ('test_index_provisioning_wiring_integration.py', '<module>'),
    # test_integration_marker_config.py -- 2 sites at 120s
    ('test_integration_marker_config.py', 'TestIntegrationMarkerDeselection'),
    (
        'test_integration_marker_config.py',
        'test_real_embedder_test_gated_by_default_and_selectable_via_marker',
    ),
    # test_integration_marker_real_service.py -- 5 sites at 120s
    ('test_integration_marker_real_service.py', 'test_list_indices_integration_module_gated'),
    ('test_integration_marker_real_service.py', 'test_mem0_client_qdrant_probe_gated'),
    ('test_integration_marker_real_service.py', 'test_merge_entities_live_class_gated'),
    ('test_integration_marker_real_service.py', 'test_refresh_entity_summary_live_class_gated'),
    ('test_integration_marker_real_service.py', 'test_startup_identity_scan_live_classes_gated'),
    # test_local_memory_models_eval_corpus.py -- 1 site at 120s
    ('test_local_memory_models_eval_corpus.py', 'test_live_dark_factory_population_smoke'),
    # test_project_scope.py -- 1 site at 120s
    (
        'test_project_scope.py',
        'TestProjectScopeTypeGateRejection::test_pyright_rejects_transposition_and_frozen_mutation',
    ),
    # test_rrf_cross_store_merge.py -- 1 site at 120s
    ('test_rrf_cross_store_merge.py', '<module>'),
    # test_sqlite_task_backend_crash.py -- 2 sites at 120s and 180s
    ('test_sqlite_task_backend_crash.py', 'test_committed_rows_survive_sigkill'),
    ('test_sqlite_task_backend_crash.py', 'test_repeated_sigkill_cycles_preserve_durability'),
    # test_worker_id_fixture.py -- 1 site at 180s
    ('test_worker_id_fixture.py', 'test_worker_id_survives_the_lanes_serial_confirm_rerun'),
})

#: Floors, not equalities (487 files and 60 sites measured on 2026-10-07): a
#: broken sweep reports zero offenders, which is indistinguishable from a
#: clean tree without them.
_MIN_EXPECTED_TEST_FILES = 300
_MIN_EXPECTED_MARKER_SITES = 40


def _verify_budget() -> int:
    test_command = yaml.safe_load(_ORCH_YAML.read_text(encoding='utf-8'))['test_command']
    budget = verify_cli_timeout(test_command)
    assert budget is not None, (
        f'{_ORCH_YAML} test_command carries no --timeout (got: {test_command!r}), '
        'so there is no verify budget for a marker to invert.'
    )
    return budget


def _sites_by_module(module: str, tree: ast.Module) -> Iterator[tuple[str, TimeoutSite]]:
    for site in timeout_marker_sites(tree, _SANCTIONED):
        yield module, site


@functools.cache
def _scan() -> TreeScan[tuple[str, TimeoutSite]]:
    return scan_python_tree(_TESTS_DIR, _sites_by_module)


def _assert_the_sweep_read_the_tree(scan: TreeScan[tuple[str, TimeoutSite]]) -> None:
    assert scan.examined >= _MIN_EXPECTED_TEST_FILES, (
        f'only {scan.examined} .py files examined under {_TESTS_DIR} (expected '
        f'at least {_MIN_EXPECTED_TEST_FILES}; {len(scan.unreadable)} skipped '
        f'as unreadable: {sorted(scan.unreadable)}) -- the sweep itself is '
        'broken, so this guard would pass vacuously rather than because the '
        'tree is clean.'
    )


def _ratchet() -> GrandfatherRatchet:
    budget = _verify_budget()
    in_band = [pair for pair in _scan().items if inverts(pair[1].seconds, verify_cli_budget=budget)]
    return grandfather_ratchet(in_band, _GRANDFATHERED)


def test_the_verify_budget_is_configured() -> None:
    budget = _verify_budget()

    assert budget > DELIBERATE_TIGHT_BOUND_CEILING, (
        f'{_ORCH_YAML} passes --timeout={budget}, which leaves the inversion '
        f'band ({DELIBERATE_TIGHT_BOUND_CEILING}, {budget}) empty, so this '
        'guard would pass vacuously.'
    )


def test_no_new_timeout_marker_sits_in_the_inversion_band() -> None:
    _assert_the_sweep_read_the_tree(_scan())
    budget = _verify_budget()

    new_offenders = _ratchet().new_offenders

    assert not new_offenders, (
        inversion_failure_message(
            new_offenders,
            verify_cli_budget=budget,
            slow_test_marker=(
                f'@pytest.mark.timeout({budget})   # slow test -- or drop the '
                'marker and take the ambient budget'
            ),
        )
        + '\n\n_GRANDFATHERED only ever shrinks -- do not add yours to it.'
    )


def test_grandfather_allowlist_has_no_stale_entries() -> None:
    _assert_the_sweep_read_the_tree(_scan())

    stale = _ratchet().stale

    assert not stale, stale_grandfather_message(stale)


def test_the_marker_census_is_not_vacuous() -> None:
    scan = _scan()

    assert len(scan.items) >= _MIN_EXPECTED_MARKER_SITES, (
        f'only {len(scan.items)} timeout marker site(s) found across '
        f'{scan.examined} files (expected at least {_MIN_EXPECTED_MARKER_SITES}) '
        '-- timeout_marker_sites has probably stopped matching, so the sweep '
        'would pass vacuously.'
    )
