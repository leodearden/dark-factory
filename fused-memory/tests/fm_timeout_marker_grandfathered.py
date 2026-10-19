"""The fused-memory/tests timeout-marker inversion guard's grandfather allowlist.

Its own module so the INV-12 declaration imports it by a unique name; see
tests/scripts/test_pytest_workspace_collection.py::test_no_governed_list_declaration_imports_a_bare_name_two_dirs_define.
"""

from __future__ import annotations

#: In-band sites that predate this guard: 20 across 13 modules, MEASURED on
#: 2026-10-07 at main 1c6a52dbfd.  Entries may only ever be REMOVED.  Migrating
#: them was deferred on task 5147's reasoning (rewriting 13 modules would take
#: locks across the package); follow-up tkt_0RVDQ479918XDV0C80NXAVT8HX.  Keyed
#: per SITE on the path RELATIVE to tests/, so a nested module cannot inherit a
#: top-level twin's exemption.
INVERSION_BAND_GRANDFATHERED: frozenset[tuple[str, str]] = frozenset({
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
