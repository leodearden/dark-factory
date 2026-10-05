# Test census: reify, 2026-10-04

- Measured tree: `/home/leo/src/reify` at `107c9016fe6042e3005773e90b95c74f870f5586`
- Evidence root: `/home/leo/src/reify`
- Ecosystem: nextest
- Task 5414: a store-only census. No test is retired or changed here.

## Commands used

```
python scripts/suite_census.py --ecosystem nextest --root /home/leo/src/reify --tree /home/leo/src/reify --project reify --date 2026-10-04 --out plans/test-census-2026-10-04-reify.md
```

Evidence read, per source:

- nextest-logs: `data/verify-logs/*/attempt-*.test-*.log`
- flaky-ledger: `data/verify-logs/flaky-ledger.jsonl`
- flake_occurrence: `data/orchestrator/runs.db: SELECT rowid, observed_at, test_id FROM flake_occurrence ORDER BY rowid`

## Part 1: never failed × per-run cost

### Evidence windows

| source | pattern / query | present | artefacts | runs | red runs | red share | first | last | unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nextest-logs | `data/verify-logs/*/attempt-*.test-*.log` | yes | 116 | 78 | 54 | 69.2% | 2026-09-04T23:39:04Z | 2026-10-04T15:51:23Z | 0 |
| flaky-ledger | `data/verify-logs/flaky-ledger.jsonl` | yes | 217 | 202 | 202 | 100.0% | 2026-07-19T22:22:50Z | 2026-10-04T13:49:50Z | 0 |
| flake_occurrence | `data/orchestrator/runs.db: SELECT rowid, observed_at, test_id FROM flake_occurrence ORDER BY rowid` | yes | 77 | 0 | 0 | — | 2026-08-30T15:45:11Z | 2026-10-04T06:04:34Z | 77 |

The failed floor holds 48 tests: every test one of the sources above recorded failing inside its window. Failures outside those windows, or never retained, are unseen, so the true failed set is at least this large. 39 floor members have no cost observation (their failure is known only from a log or ledger line).

### Per package

| package | costed tests | failed (floor) | never failed, costed | never failed, uncosted | p50 of medians s | top 1% share of time |
| --- | --- | --- | --- | --- | --- | --- |
| reify-ast | 89 | 0 | 89 | 0 | 0.062 | 14.9% |
| reify-audit | 854 | 0 | 854 | 0 | 0.381 | 12.9% |
| reify-build-utils | 10 | 0 | 10 | 0 | 0.061 | 86.7% |
| reify-builtins | 73 | 0 | 73 | 0 | 0.060 | 24.3% |
| reify-cli | 666 | 0 | 666 | 0 | 1.282 | 13.4% |
| reify-compiler | 8717 | 3 | 8714 | 0 | 0.389 | 3.8% |
| reify-compute-contract | 38 | 0 | 38 | 0 | 0.282 | 4.7% |
| reify-config | 87 | 0 | 87 | 0 | 0.059 | 5.4% |
| reify-constraints | 627 | 0 | 627 | 0 | 0.342 | 6.3% |
| reify-core | 592 | 0 | 592 | 0 | 0.053 | 1.9% |
| reify-doc | 130 | 0 | 130 | 0 | 0.054 | 2.5% |
| reify-doc-build | 17 | 0 | 17 | 0 | 0.954 | 9.2% |
| reify-eval | 6364 | 5 | 6359 | 0 | 0.886 | 39.6% |
| reify-eval-fea-tests | 130 | 0 | 130 | 0 | 1.446 | 33.6% |
| reify-expr | 1421 | 0 | 1421 | 0 | 0.091 | 7.5% |
| reify-fdm | 87 | 0 | 87 | 0 | 0.088 | 9.5% |
| reify-gcode | 47 | 0 | 47 | 0 | 0.077 | 3.8% |
| reify-geometry | 18 | 0 | 18 | 0 | 0.383 | 7.1% |
| reify-gui | 1233 | 1 | 1232 | 0 | 1.197 | 2.7% |
| reify-ir | 1060 | 0 | 1060 | 0 | 0.146 | 2.2% |
| reify-kernel-conformance | 10 | 0 | 10 | 0 | 1.044 | 71.7% |
| reify-kernel-fidget | 35 | 0 | 35 | 0 | 0.568 | 5.3% |
| reify-kernel-gmsh | 206 | 0 | 206 | 0 | 0.629 | 23.5% |
| reify-kernel-manifold | 66 | 0 | 66 | 0 | 0.601 | 3.0% |
| reify-kernel-occt | 840 | 0 | 840 | 0 | 0.755 | 4.3% |
| reify-kernel-openvdb | 100 | 0 | 100 | 0 | 0.653 | 7.9% |
| reify-lsp | 585 | 0 | 585 | 0 | 1.316 | 2.8% |
| reify-mcp | 110 | 0 | 110 | 0 | 0.099 | 3.8% |
| reify-mesh-morph | 141 | 0 | 141 | 0 | 0.450 | 6.0% |
| reify-runtime | 85 | 0 | 85 | 0 | 0.416 | 1.6% |
| reify-shell-extract | 170 | 0 | 170 | 0 | 0.102 | 32.5% |
| reify-solver-elastic | 1008 | 0 | 1008 | 0 | 0.392 | 11.7% |
| reify-spec-conformance | 7 | 0 | 7 | 0 | 0.246 | 24.5% |
| reify-stdlib | 2395 | 0 | 2395 | 0 | 0.332 | 12.4% |
| reify-syntax | 953 | 0 | 953 | 0 | 0.293 | 3.0% |
| reify-test-support | 700 | 0 | 700 | 0 | 0.318 | 6.3% |
| tests/infra | 0 | 39 | 0 | 152 | — | — |
| tree-sitter-reify | 270 | 0 | 270 | 0 | 0.050 | 10.5% |

### Never failed × per-run cost (top 100 of 29932)

| rank | test | package | runs | median s | max s | total s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `reify-eval::harness_corpus_gates eval_invariant_corpus_sweep::corpus_sweep_shard_20` | reify-eval | 12 | 292.566 | 479.283 | 3155.969 |
| 2 | `reify-eval::solve_elastic_static_body_e2e multi_case_non_prismatic_body_caches_one_realization_for_both_cases` | reify-eval | 30 | 286.793 | 631.250 | 9279.515 |
| 3 | `reify-eval::harness_cache snapshot_cache_divergence_gate::snapshot_cache_sweep_shard_14` | reify-eval | 19 | 275.948 | 459.075 | 3718.872 |
| 4 | `reify-eval::no_stale_undef_invariant_gate broad_corpus_sweep_shard_14` | reify-eval | 21 | 235.887 | 395.679 | 3321.712 |
| 5 | `reify-eval compute_targets::elastic_static::tests::run_adaptive_refinement_over_cantilever_iterates_and_terminates` | reify-eval | 33 | 195.744 | 474.761 | 7410.801 |
| 6 | `reify-eval::harness_corpus_gates eval_invariant_corpus_sweep::corpus_sweep_shard_05` | reify-eval | 12 | 179.613 | 304.536 | 2306.003 |
| 7 | `reify-eval-fea-tests::buckling_multi_case buckling_multi_case_second_eval_reuses_compute_result` | reify-eval-fea-tests | 33 | 152.850 | 336.594 | 5859.515 |
| 8 | `reify-eval-fea-tests::buckling_multi_case two_case_buckling_solve_returns_populated_result` | reify-eval-fea-tests | 33 | 151.216 | 355.818 | 5828.971 |
| 9 | `reify-eval-fea-tests::buckling_multi_case buckling_multi_case_smoke_integration` | reify-eval-fea-tests | 33 | 149.384 | 355.917 | 5783.121 |
| 10 | `reify-eval::harness_cache snapshot_cache_divergence_gate::snapshot_cache_sweep_shard_01` | reify-eval | 19 | 142.138 | 254.608 | 2705.780 |
| 11 | `reify-kernel-conformance::occt_gmsh_volume_conformance occt_fixtures_mesh_to_volume_and_revalidate_through_gmsh` | reify-kernel-conformance | 30 | 141.372 | 261.924 | 4479.720 |
| 12 | `reify-eval::snapshot_cache_divergence_gate snapshot_cache_sweep_shard_10` | reify-eval | 2 | 129.883 | 256.879 | 259.765 |
| 13 | `reify-eval::no_stale_undef_invariant_gate broad_corpus_sweep_shard_01` | reify-eval | 21 | 129.248 | 224.009 | 2815.567 |
| 14 | `reify-kernel-gmsh::volume_fill_fraction interior_nodes_appear_once_resolution_is_finer_than_the_cross_section` | reify-kernel-gmsh | 30 | 129.186 | 229.341 | 4084.147 |
| 15 | `reify-eval::harness_corpus_gates eval_invariant_corpus_sweep::corpus_sweep_shard_02` | reify-eval | 12 | 101.603 | 152.257 | 1233.027 |
| 16 | `reify-kernel-gmsh::volume_fill_fraction tessellated_cylinder_is_completely_tetrahedralized` | reify-kernel-gmsh | 30 | 101.305 | 228.867 | 3330.324 |
| 17 | `reify-eval::snapshot_cache_divergence_gate snapshot_cache_sweep_shard_12` | reify-eval | 2 | 94.960 | 188.939 | 189.921 |
| 18 | `reify-eval::snapshot_cache_divergence_gate snapshot_cache_sweep_shard_00` | reify-eval | 2 | 93.978 | 130.094 | 187.956 |
| 19 | `reify-kernel-gmsh::mesh_to_volume_tests mesh_size_override_increases_tet_count` | reify-kernel-gmsh | 32 | 86.096 | 143.381 | 2853.865 |
| 20 | `reify-eval-fea-tests::buckling_persistent_cache_round_trip buckling_persistent_cache_cross_restart_round_trip` | reify-eval-fea-tests | 5 | 81.215 | 116.110 | 445.821 |
| 21 | `reify-eval::harness_corpus_gates eval_invariant_corpus_sweep::corpus_sweep_shard_09` | reify-eval | 12 | 72.201 | 107.023 | 943.466 |
| 22 | `reify-kernel-gmsh::mesher_poison_recovery a_failed_mesh_to_volume_leaves_the_sibling_meshers_usable` | reify-kernel-gmsh | 10 | 72.108 | 115.678 | 602.765 |
| 23 | `reify-eval::harness_cache snapshot_cache_divergence_gate::snapshot_cache_sweep_shard_00` | reify-eval | 19 | 68.594 | 130.075 | 1115.608 |
| 24 | `reify-kernel-gmsh::mesher_poison_recovery a_failed_mesh_to_volume_leaves_the_mesher_usable_for_the_next_caller` | reify-kernel-gmsh | 10 | 66.516 | 119.679 | 702.671 |
| 25 | `reify-eval::isosurface_iso_option_e2e iso_option_changes_surfaced_mesh` | reify-eval | 32 | 63.751 | 184.381 | 2338.409 |
| 26 | `reify-eval::no_stale_undef_invariant_gate broad_corpus_sweep_shard_00` | reify-eval | 21 | 63.605 | 136.020 | 1337.513 |
| 27 | `reify-kernel-gmsh::mesher_poison_recovery a_failed_sibling_mesher_leaves_mesh_to_volume_usable` | reify-kernel-gmsh | 10 | 61.612 | 96.355 | 652.407 |
| 28 | `reify-eval::harness_fea_solver_e2e process_dfm_thickness_example::example_emits_min_wall_and_min_feature_error` | reify-eval | 1 | 61.546 | 61.546 | 61.546 |
| 29 | `reify-eval::harness_corpus_gates eval_invariant_corpus_sweep::corpus_sweep_shard_14` | reify-eval | 12 | 60.207 | 104.101 | 786.232 |
| 30 | `reify-eval::warm_state_donation warm_state_seeded_modal_solve_matches_cold_baseline` | reify-eval | 3 | 58.426 | 66.060 | 165.960 |
| 31 | `reify-eval::harness_fea_solver_e2e process_dfm_thickness_example::example_emits_min_wall_and_min_feature_warning` | reify-eval | 1 | 57.147 | 57.147 | 57.147 |
| 32 | `reify-eval-fea-tests::modal_analysis_e2e e2e_mode_frequency_is_dimensioned_scalar` | reify-eval-fea-tests | 5 | 56.882 | 72.027 | 284.814 |
| 33 | `reify-eval::harness_process_dfm process_dfm_thickness_example::example_emits_info_thickness_and_conformer_is_silent` | reify-eval | 32 | 56.803 | 114.182 | 1988.675 |
| 34 | `reify-eval::harness_process_dfm process_dfm_thickness_example::example_emits_min_wall_and_min_feature_warning` | reify-eval | 32 | 56.781 | 97.903 | 1954.723 |
| 35 | `reify-eval::solve_elastic_static_body_e2e multi_case_body_solve_shares_one_realization_across_cases` | reify-eval | 32 | 56.656 | 95.721 | 1737.999 |
| 36 | `reify-eval::solve_elastic_static_body_e2e body_solve_runs_on_realized_volume_mesh` | reify-eval | 32 | 56.455 | 101.357 | 1791.918 |
| 37 | `reify-eval::harness_process_dfm process_dfm_thickness_example::example_emits_min_wall_and_min_feature_error` | reify-eval | 32 | 55.585 | 95.413 | 1922.664 |
| 38 | `reify-kernel-gmsh::volume_fill_fraction production_composition_unwelded_plus_repair_is_completely_tetrahedralized` | reify-kernel-gmsh | 30 | 55.090 | 76.163 | 1552.741 |
| 39 | `reify-eval::harness_fea_solver_e2e process_dfm_thickness_example::example_emits_info_thickness_and_conformer_is_silent` | reify-eval | 1 | 54.650 | 54.650 | 54.650 |
| 40 | `reify-eval::volume_mesh_realization_e2e call_edge_writes_volume_mesh_repr_and_gmsh_kernel_for_demanded_body` | reify-eval | 31 | 54.254 | 111.831 | 1776.308 |
| 41 | `reify-eval::solve_elastic_static_body_e2e multi_case_body_solve_survives_a_preceding_template` | reify-eval | 30 | 53.046 | 116.708 | 1750.043 |
| 42 | `reify-eval::volume_mesh_realization_e2e e2e_vm_probe_reads_back_tet_volume_mesh_from_demanded_body` | reify-eval | 31 | 52.927 | 99.022 | 1651.437 |
| 43 | `reify-kernel-gmsh::volume_fill_fraction millimetre_scale_box_is_completely_tetrahedralized` | reify-kernel-gmsh | 30 | 51.888 | 112.453 | 1658.056 |
| 44 | `reify-kernel-gmsh::volume_fill_fraction production_report_surfaces_the_fill_measurement` | reify-kernel-gmsh | 30 | 50.913 | 77.065 | 1494.227 |
| 45 | `reify-eval::harness_cache snapshot_cache_divergence_gate::snapshot_cache_sweep_shard_20` | reify-eval | 19 | 50.839 | 72.608 | 636.066 |
| 46 | `reify-eval-fea-tests::modal_analysis_e2e e2e_simply_supported_modes_match_analytic` | reify-eval-fea-tests | 5 | 50.455 | 66.551 | 271.375 |
| 47 | `reify-kernel-gmsh::volume_fill_fraction realized_production_box_is_completely_tetrahedralized` | reify-kernel-gmsh | 30 | 49.611 | 78.213 | 1541.665 |
| 48 | `reify-kernel-gmsh::volume_fill_fraction unit_cube_is_completely_tetrahedralized` | reify-kernel-gmsh | 30 | 49.304 | 81.168 | 1530.941 |
| 49 | `reify-eval-fea-tests::modal_analysis_e2e e2e_cantilever_first_mode_within_two_percent` | reify-eval-fea-tests | 5 | 49.297 | 62.251 | 259.079 |
| 50 | `reify-eval::snapshot_cache_divergence_gate snapshot_cache_sweep_shard_01` | reify-eval | 2 | 49.283 | 96.043 | 98.566 |
| 51 | `reify-kernel-gmsh::pipeline_integration with_libgmsh::mesh_surface_to_volume_with_diagnostics_all_none_round_trips_unit_cube` | reify-kernel-gmsh | 32 | 48.838 | 82.286 | 1580.515 |
| 52 | `reify-kernel-gmsh::mesh_to_volume_tests mesh_to_volume_leaves_the_gmsh_logger_stopped` | reify-kernel-gmsh | 10 | 48.221 | 83.523 | 499.609 |
| 53 | `reify-kernel-conformance::occt_gmsh_attributed_conformance occt_box_attributed_volume_preserves_face_attribution` | reify-kernel-conformance | 30 | 47.147 | 80.212 | 1099.049 |
| 54 | `reify-eval::snapshot_cache_divergence_gate snapshot_cache_sweep_shard_19` | reify-eval | 2 | 47.002 | 47.234 | 94.005 |
| 55 | `reify-eval::region_resolution_boundary fail_closed_predicate_over_volume_mesh_produces_qns_error_and_undef` | reify-eval | 32 | 47.001 | 109.153 | 1507.901 |
| 56 | `reify-eval compute_targets::elastic_static::tests::adaptive_branch_falls_back_to_uniform_refinement_when_the_gmsh_lane_is_unavailable` | reify-eval | 20 | 46.662 | 92.372 | 1024.607 |
| 57 | `reify-eval::harness_modal modal_material_damping_e2e::material_damping_leaves_nodamping_and_rayleigh_byte_identical` | reify-eval | 21 | 44.609 | 88.374 | 987.898 |
| 58 | `reify-kernel-gmsh::volume_fill_fraction short_box_is_completely_tetrahedralized` | reify-kernel-gmsh | 30 | 44.139 | 75.475 | 1319.724 |
| 59 | `reify-kernel-gmsh::mesh_to_volume_tests trait_mesh_surface_to_volume_then_store_round_trips_through_dyn_kernel` | reify-kernel-gmsh | 32 | 44.010 | 78.199 | 1478.021 |
| 60 | `reify-eval::harness_cache snapshot_cache_divergence_gate::snapshot_cache_sweep_shard_22` | reify-eval | 19 | 43.581 | 70.604 | 673.863 |
| 61 | `reify-kernel-gmsh::pipeline_integration with_libgmsh::caller_mesh_size_wins_over_auto_size_observable_in_tet_count` | reify-kernel-gmsh | 32 | 42.377 | 83.951 | 1448.025 |
| 62 | `reify-kernel-gmsh::mesh_surface_to_volume_attributed gmsh_mesh_surface_to_volume_attributed_threads_boundary_onto_volume_mesh` | reify-kernel-gmsh | 32 | 42.184 | 99.026 | 1221.617 |
| 63 | `reify-eval::harness_modal modal_material_damping_e2e::material_damping_composes_additively_with_its_extra_descriptor` | reify-eval | 21 | 41.526 | 77.677 | 937.975 |
| 64 | `reify-eval::harness_modal modal_material_damping_e2e::material_damping_gives_half_the_loss_factor_for_every_mode` | reify-eval | 21 | 41.235 | 104.736 | 1008.727 |
| 65 | `reify-eval::no_stale_undef_invariant_gate broad_corpus_sweep_shard_20` | reify-eval | 21 | 40.360 | 75.794 | 609.357 |
| 66 | `reify-kernel-gmsh::mesh_to_volume_tests p2_element_order_produces_stride_10_tet_indices` | reify-kernel-gmsh | 32 | 39.344 | 89.684 | 1375.684 |
| 67 | `reify-eval::morph_arm_e2e e2e_structural_tick_remeshes_and_records_ineligible` | reify-eval | 23 | 38.748 | 175.070 | 1445.179 |
| 68 | `reify-eval::voxel_to_mesh_e2e voxel_to_mesh_builds_honest_voxel_operand_and_mesh_terminal` | reify-eval | 31 | 38.741 | 71.612 | 1222.633 |
| 69 | `reify-eval::snapshot_cache_divergence_gate snapshot_cache_sweep_shard_23` | reify-eval | 2 | 38.048 | 65.991 | 76.096 |
| 70 | `reify-kernel-gmsh::mesh_to_volume_tests volume_mesh_store_round_trips_produced_tet_mesh` | reify-kernel-gmsh | 32 | 37.639 | 73.423 | 1323.638 |
| 71 | `reify-kernel-gmsh::mesh_to_volume_tests cube_surface_produces_nonempty_p1_tet_mesh` | reify-kernel-gmsh | 32 | 37.276 | 97.760 | 1314.732 |
| 72 | `reify-kernel-gmsh::mesh_surface_to_volume_attributed mesh_surface_to_volume_attributed_welds_unwelded_surface_and_attributes` | reify-kernel-gmsh | 32 | 36.742 | 84.170 | 1095.731 |
| 73 | `reify-eval::no_stale_undef_invariant_gate broad_corpus_sweep_shard_22` | reify-eval | 21 | 34.371 | 99.535 | 716.281 |
| 74 | `reify-eval::harness_modal modal_material_damping_e2e::the_not_damped_rejection_stays_silent_for_every_valid_combination` | reify-eval | 21 | 33.102 | 77.748 | 851.026 |
| 75 | `reify-cli::harness_cli cli_check::check_constraint_results_come_from_authoritative_check_not_build` | reify-cli | 34 | 28.114 | 76.056 | 1080.394 |
| 76 | `reify-cli::harness_cli cli_dfm_overhang::check_dfm_plus_repr_within_combined_arm` | reify-cli | 36 | 26.574 | 63.840 | 1072.820 |
| 77 | `reify-eval::solver_optimality_unproven example_file_solver_optimality_unproven_emits_warning` | reify-eval | 32 | 26.233 | 57.485 | 898.380 |
| 78 | `reify-eval::snapshot_cache_divergence_gate snapshot_cache_sweep_shard_17` | reify-eval | 2 | 26.202 | 51.598 | 52.404 |
| 79 | `reify-kernel-openvdb::mesh_to_voxel_resolution_tests min_feature_request_produces_a_strictly_finer_grid_than_honest_floor` | reify-kernel-openvdb | 14 | 26.002 | 45.757 | 388.725 |
| 80 | `reify-eval::harness_kernel_realization realization_read_api::ri_box_realizes_with_nonzero_hash_and_shell_extract_consumes_real_openvdb_sdf` | reify-eval | 33 | 25.501 | 58.948 | 903.286 |
| 81 | `reify-eval-fea-tests::modal_analysis_e2e e2e_two_fixed_supports_are_clamped_clamped_not_simply_supported` | reify-eval-fea-tests | 5 | 24.647 | 35.228 | 125.402 |
| 82 | `reify-eval::harness_cache snapshot_cache_divergence_gate::snapshot_cache_sweep_shard_23` | reify-eval | 19 | 24.631 | 128.720 | 800.536 |
| 83 | `reify-eval::solver_optimality_unproven multi_param_objective_emits_solver_optimality_unproven_warning` | reify-eval | 32 | 24.566 | 59.155 | 883.464 |
| 84 | `reify-eval::harness_sweep idler_seat_e2e::idler_sheave_mesh_has_the_declared_seat` | reify-eval | 5 | 24.267 | 64.924 | 159.013 |
| 85 | `reify-stdlib trajectory::tots::tests::line_search_rejects_merit_increase` | reify-stdlib | 29 | 24.112 | 41.971 | 740.345 |
| 86 | `reify-kernel-openvdb::mesh_to_voxel_resolution_tests ingest_mesh_at_resolution_honest_floor_matches_ingest_mesh` | reify-kernel-openvdb | 14 | 23.938 | 47.108 | 360.244 |
| 87 | `reify-eval::isosurface_iso_option_e2e iso_example_fixture_surfaces_nonempty` | reify-eval | 32 | 23.718 | 55.790 | 824.087 |
| 88 | `reify-eval::isosurface_wiring_e2e isosurface_wiring_honors_placement_transform_on_shell_mesh` | reify-eval | 32 | 23.593 | 70.493 | 862.948 |
| 89 | `reify-cli::harness_cli cli_reset_per_build_interleaving::check_both_kinds_yield_real_verdicts_under_occt` | reify-cli | 36 | 23.320 | 53.356 | 915.816 |
| 90 | `reify-eval::no_stale_undef_invariant_gate broad_corpus_sweep_shard_23` | reify-eval | 21 | 22.936 | 141.953 | 887.488 |
| 91 | `reify-eval::isosurface_iso_option_e2e iso_option_out_of_band_surfaces_empty_mesh` | reify-eval | 32 | 22.795 | 51.755 | 782.557 |
| 92 | `reify-stdlib trajectory::tots::tests::sqp_gantry_converges` | reify-stdlib | 29 | 22.733 | 37.432 | 685.829 |
| 93 | `reify-eval::printer_print_envelope_e2e printer_print_envelope_eval_e2e` | reify-eval | 3 | 22.672 | 29.645 | 70.292 |
| 94 | `reify-stdlib trajectory::trampoline::tests::input_shape_value_tots_arm_shapes_profile` | reify-stdlib | 29 | 22.668 | 34.686 | 668.838 |
| 95 | `reify-eval::isosurface_wiring_e2e isosurface_wiring_builds_honest_voxel_operand_and_mesh_terminal` | reify-eval | 32 | 22.262 | 58.963 | 792.703 |
| 96 | `reify-eval::harness_kernel_realization realization_read_api::shell_extract_prefers_real_body_sdf_tracks_real_extents` | reify-eval | 33 | 22.256 | 57.560 | 834.426 |
| 97 | `reify-stdlib trajectory::trampoline::tests::input_shape_value_revolute_tots_arm_shapes_profile` | reify-stdlib | 28 | 21.735 | 41.811 | 678.992 |
| 98 | `reify-kernel-openvdb::ingest_mesh_densify_tests densify_grid_to_sampled_via_trait_object` | reify-kernel-openvdb | 29 | 21.089 | 41.539 | 629.523 |
| 99 | `reify-eval::harness_cache snapshot_cache_divergence_gate::snapshot_cache_sweep_shard_15` | reify-eval | 19 | 20.829 | 385.798 | 2686.591 |
| 100 | `reify-eval::morph_arm_e2e e2e_no_producer_engine_remeshes_volume_mesh` | reify-eval | 32 | 20.678 | 66.519 | 828.760 |

<details><summary>Failed floor (48 tests)</summary>

- `reify-compiler::harness_doc_chunks units_chunk_smoke::documented_eval_only_rejections_are_invisible_to_the_compile_layer` (reify-compiler)
- `reify-compiler::harness_type_checking unresolved_function_tests::struct_constructor_in_a_regular_fn_body_stays_clean_in_both_orders` (reify-compiler)
- `reify-compiler::harness_type_checking unresolved_function_tests::struct_constructor_in_a_trait_static_fn_body_is_not_unresolved` (reify-compiler)
- `reify-eval::harness_corpus_gates units_length_corpus_end_state::no_shipped_example_trips_a_length_gate` (reify-eval)
- `reify-eval::morph_arm_e2e e2e_non_structural_tick_morphs_and_preserves_connectivity` (reify-eval)
- `reify-eval::solve_elastic_static_body_e2e body_adaptive_solve_runs_the_gmsh_realized_localized_lane` (reify-eval)
- `reify-eval::solve_elastic_static_body_e2e non_prismatic_body_solve_runs_on_realized_volume_mesh` (reify-eval)
- `reify-eval::solve_elastic_static_body_e2e non_prismatic_two_case_build_realizes_body_exactly_once` (reify-eval)
- `reify-gui tests::mcp_dispatch_tests::a_write_dispatched_on_the_lane_lands_in_the_callers_engine` (reify-gui)
- `tests/infra/test_affected_crates_lib.sh` (tests/infra)
- `tests/infra/test_check_event_inventory.sh` (tests/infra)
- `tests/infra/test_cited_test_paths_resolve.sh` (tests/infra)
- `tests/infra/test_govtest_slice_reaper.sh` (tests/infra)
- `tests/infra/test_gui_typecheck_test_side.sh` (tests/infra)
- `tests/infra/test_harness_kloc_cap.sh` (tests/infra)
- `tests/infra/test_jobserver_balancer.sh` (tests/infra)
- `tests/infra/test_lane_x_flock.sh` (tests/infra)
- `tests/infra/test_no_new_wallclock_rust_deadlines.sh` (tests/infra)
- `tests/infra/test_occt_deps_preflight.sh` (tests/infra)
- `tests/infra/test_occt_flock_gate.sh` (tests/infra)
- `tests/infra/test_plan_capture_lib.sh` (tests/infra)
- `tests/infra/test_portable_timeout.sh` (tests/infra)
- `tests/infra/test_prd_gate_compiler_type_hygiene.sh` (tests/infra)
- `tests/infra/test_prd_gate_substrate_guard.sh` (tests/infra)
- `tests/infra/test_pre_commit_ts_guard.sh` (tests/infra)
- `tests/infra/test_project_checks_typecheck.sh` (tests/infra)
- `tests/infra/test_reify_audit_pdiag.sh` (tests/infra)
- `tests/infra/test_reify_audit_pdiag_vacuity.sh` (tests/infra)
- `tests/infra/test_reify_audit_ptodo.sh` (tests/infra)
- `tests/infra/test_run_all.sh` (tests/infra)
- `tests/infra/test_run_all_ambient_isolation.sh` (tests/infra)
- `tests/infra/test_run_gui_scripts.sh` (tests/infra)
- `tests/infra/test_seed_warm_lane.sh` (tests/infra)
- `tests/infra/test_slot_timeout_marker.sh` (tests/infra)
- `tests/infra/test_stash_guard.sh` (tests/infra)
- `tests/infra/test_test_helpers.sh` (tests/infra)
- `tests/infra/test_test_run_semaphore.sh` (tests/infra)
- `tests/infra/test_verify_env_ambient_isolation.sh` (tests/infra)
- `tests/infra/test_verify_failfast_order.sh` (tests/infra)
- `tests/infra/test_verify_gate_exclude_heavy.sh` (tests/infra)
- `tests/infra/test_verify_gui_feature_check.sh` (tests/infra)
- `tests/infra/test_verify_gui_retry_specs.sh` (tests/infra)
- `tests/infra/test_verify_ld_library_path_scope.sh` (tests/infra)
- `tests/infra/test_verify_nextest_absent_suites.sh` (tests/infra)
- `tests/infra/test_verify_release_delta_skip.sh` (tests/infra)
- `tests/infra/test_verify_retry_failed_only.sh` (tests/infra)
- `tests/infra/test_verify_scope.sh` (tests/infra)
- `tests/infra/test_warm_lane_sizing_lifecycle.sh` (tests/infra)

</details>

<details><summary>Never failed, uncosted (152 tests)</summary>

- `tests/infra/test_agent_cache_redirect.sh` (tests/infra)
- `tests/infra/test_agent_cargo_shim.sh` (tests/infra)
- `tests/infra/test_amendment_rounds_default_restored.sh` (tests/infra)
- `tests/infra/test_audit_orphan_producers.sh` (tests/infra)
- `tests/infra/test_await_merge_landing.sh` (tests/infra)
- `tests/infra/test_cargo_incremental_lane_decision.sh` (tests/infra)
- `tests/infra/test_cargo_test_tally.sh` (tests/infra)
- `tests/infra/test_compute_trampoline_registration_wired.sh` (tests/infra)
- `tests/infra/test_copy_list_preflight.sh` (tests/infra)
- `tests/infra/test_cpu_admit.sh` (tests/infra)
- `tests/infra/test_cpu_governance_config.sh` (tests/infra)
- `tests/infra/test_cpu_governed_exec.sh` (tests/infra)
- `tests/infra/test_cpu_governed_exec_hostexcl.sh` (tests/infra)
- `tests/infra/test_cpu_load_governance.sh` (tests/infra)
- `tests/infra/test_cpu_load_governance_deflake.sh` (tests/infra)
- `tests/infra/test_engine_hash_closure.sh` (tests/infra)
- `tests/infra/test_ensure_gui_sidecar_placeholder.sh` (tests/infra)
- `tests/infra/test_ensure_warm_base.sh` (tests/infra)
- `tests/infra/test_event_inventory_wired.sh` (tests/infra)
- `tests/infra/test_fd_probe_self_reference.sh` (tests/infra)
- `tests/infra/test_fea_e2e_crate_occt_free.sh` (tests/infra)
- `tests/infra/test_find_uses_smoke_runner.sh` (tests/infra)
- `tests/infra/test_flake_density_report.sh` (tests/infra)
- `tests/infra/test_fleet_load_detector.sh` (tests/infra)
- `tests/infra/test_flock_detached_fork_guard.sh` (tests/infra)
- `tests/infra/test_git_rerere_guard.sh` (tests/infra)
- `tests/infra/test_gui_dist_gitignored.sh` (tests/infra)
- `tests/infra/test_gui_test_script.sh` (tests/infra)
- `tests/infra/test_gui_vitest_rpc_hardening.sh` (tests/infra)
- `tests/infra/test_harness_baseline_registration_gate.sh` (tests/infra)
- `tests/infra/test_heavy_filter_atoms.sh` (tests/infra)
- `tests/infra/test_helpers.sh` (tests/infra)
- `tests/infra/test_hooks_call_verify.sh` (tests/infra)
- `tests/infra/test_host_global_unit_pinning.sh` (tests/infra)
- `tests/infra/test_infra_classification_manifest_gate.sh` (tests/infra)
- `tests/infra/test_infra_git_env_isolation.sh` (tests/infra)
- `tests/infra/test_jcodemunch_index_reify.sh` (tests/infra)
- `tests/infra/test_jcodemunch_index_units.sh` (tests/infra)
- `tests/infra/test_jobserver_acceptance.sh` (tests/infra)
- `tests/infra/test_jobserver_canary.sh` (tests/infra)
- `tests/infra/test_jobserver_role_fifo.sh` (tests/infra)
- `tests/infra/test_jobserver_tuning_harness.sh` (tests/infra)
- `tests/infra/test_land_script.sh` (tests/infra)
- `tests/infra/test_landlock_exec_refer.sh` (tests/infra)
- `tests/infra/test_lane_lock_probe.sh` (tests/infra)
- `tests/infra/test_lane_task_status.sh` (tests/infra)
- `tests/infra/test_lean_debuginfo_profile.sh` (tests/infra)
- `tests/infra/test_lib_task_citation.sh` (tests/infra)
- `tests/infra/test_linker_config.sh` (tests/infra)
- `tests/infra/test_load_tolerance_lib.sh` (tests/infra)
- `tests/infra/test_lock_charter_decompose_guard.sh` (tests/infra)
- `tests/infra/test_lock_charter_guard.sh` (tests/infra)
- `tests/infra/test_lock_charter_lifecycle.sh` (tests/infra)
- `tests/infra/test_main_gate_worktree_config.sh` (tests/infra)
- `tests/infra/test_merge_role_lint_first_seam.sh` (tests/infra)
- `tests/infra/test_merge_verify_cold_outer_timeout.sh` (tests/infra)
- `tests/infra/test_mesh_count_parity_smoke_runner.sh` (tests/infra)
- `tests/infra/test_nan_safe_ordering_guard_wired.sh` (tests/infra)
- `tests/infra/test_nextest_absent_lib.sh` (tests/infra)
- `tests/infra/test_nextest_slow_priority.sh` (tests/infra)
- `tests/infra/test_no_bare_holder_sleep_grace.sh` (tests/infra)
- `tests/infra/test_no_dead_taskmaster_artifacts.sh` (tests/infra)
- `tests/infra/test_no_new_wallclock_upper_bounds.sh` (tests/infra)
- `tests/infra/test_no_prebuilt_reify_bin_spawn.sh` (tests/infra)
- `tests/infra/test_npm_ci_hardening.sh` (tests/infra)
- `tests/infra/test_occt_flock_gate_bounds.sh` (tests/infra)
- `tests/infra/test_occt_gated_scope.sh` (tests/infra)
- `tests/infra/test_orchestrator_config_canonical_path.sh` (tests/infra)
- `tests/infra/test_orchestrator_redeploy_restart.sh` (tests/infra)
- `tests/infra/test_portable_mtime.sh` (tests/infra)
- `tests/infra/test_portable_sha256.sh` (tests/infra)
- `tests/infra/test_prd_capability_check.sh` (tests/infra)
- `tests/infra/test_prd_decompose_verify.sh` (tests/infra)
- `tests/infra/test_prd_gate_corpus.sh` (tests/infra)
- `tests/infra/test_prd_gate_objective_inheritance.sh` (tests/infra)
- `tests/infra/test_prd_gate_struct_ctor_conformance.sh` (tests/infra)
- `tests/infra/test_proc_reaper.sh` (tests/infra)
- `tests/infra/test_provision_warm_lane_fs.sh` (tests/infra)
- `tests/infra/test_pycache_gitignored.sh` (tests/infra)
- `tests/infra/test_queue_db_gitignored.sh` (tests/infra)
- `tests/infra/test_reconciliation_db_gitignored.sh` (tests/infra)
- `tests/infra/test_reference_transaction_gate.sh` (tests/infra)
- `tests/infra/test_refresh_warm_base.sh` (tests/infra)
- `tests/infra/test_reify_audit_freshness.sh` (tests/infra)
- `tests/infra/test_reify_audit_pdiag_budget_skip.sh` (tests/infra)
- `tests/infra/test_reify_audit_pdoccover.sh` (tests/infra)
- `tests/infra/test_reify_audit_pprdstatus_wiring.sh` (tests/infra)
- `tests/infra/test_reify_audit_predone_wrapper.sh` (tests/infra)
- `tests/infra/test_reify_audit_ptodo_budget_skip.sh` (tests/infra)
- `tests/infra/test_reify_audit_ptodo_orphan_hardgate.sh` (tests/infra)
- `tests/infra/test_reify_audit_ptodo_ratchet_superset.sh` (tests/infra)
- `tests/infra/test_reify_audit_ptodo_ratchet_vacuity.sh` (tests/infra)
- `tests/infra/test_reify_bin_freshness.sh` (tests/infra)
- `tests/infra/test_reify_overlap_deploy_smoke.sh` (tests/infra)
- `tests/infra/test_reify_overlap_detector.sh` (tests/infra)
- `tests/infra/test_release_mode_in_test_command.sh` (tests/infra)
- `tests/infra/test_release_scoped_scope.sh` (tests/infra)
- `tests/infra/test_relocate_worktrees_to_warm_lane.sh` (tests/infra)
- `tests/infra/test_review_readme.sh` (tests/infra)
- `tests/infra/test_run_all_ambient_isolation_lib.sh` (tests/infra)
- `tests/infra/test_run_all_classification.sh` (tests/infra)
- `tests/infra/test_run_all_clock_marker_sanitize.sh` (tests/infra)
- `tests/infra/test_run_all_content_skip.sh` (tests/infra)
- `tests/infra/test_run_all_pool_lock_host_global.sh` (tests/infra)
- `tests/infra/test_run_all_tiering.sh` (tests/infra)
- `tests/infra/test_run_offline_deep.sh` (tests/infra)
- `tests/infra/test_sandbox_cache_writability_seam.sh` (tests/infra)
- `tests/infra/test_scope_boundary.sh` (tests/infra)
- `tests/infra/test_seed_lane_lock_release_soak.sh` (tests/infra)
- `tests/infra/test_seed_warm_base_initial.sh` (tests/infra)
- `tests/infra/test_setup_dev_no_ldconfig.sh` (tests/infra)
- `tests/infra/test_setup_worktree_debug_port.sh` (tests/infra)
- `tests/infra/test_sidecar_typecheck_test_path.sh` (tests/infra)
- `tests/infra/test_slot_event_log.sh` (tests/infra)
- `tests/infra/test_slot_holder_handshake_lib.sh` (tests/infra)
- `tests/infra/test_smoke_predone_hook.sh` (tests/infra)
- `tests/infra/test_sn_gate.sh` (tests/infra)
- `tests/infra/test_spec_anchor_lint.sh` (tests/infra)
- `tests/infra/test_sync_comments_grep.sh` (tests/infra)
- `tests/infra/test_target_per_lane_independence.sh` (tests/infra)
- `tests/infra/test_task_branch_contamination_sweep.sh` (tests/infra)
- `tests/infra/test_test_binary_concurrency_sampler.sh` (tests/infra)
- `tests/infra/test_test_pycache_gitignored.sh` (tests/infra)
- `tests/infra/test_thin_warm_lane.sh` (tests/infra)
- `tests/infra/test_tree_sitter_parse_isolation.sh` (tests/infra)
- `tests/infra/test_tree_sitter_pipeline.sh` (tests/infra)
- `tests/infra/test_typecheck_compiles_tests.sh` (tests/infra)
- `tests/infra/test_verify_admission_knob_parity.sh` (tests/infra)
- `tests/infra/test_verify_compile_gate.sh` (tests/infra)
- `tests/infra/test_verify_nextest_probe.sh` (tests/infra)
- `tests/infra/test_verify_offline_partition.sh` (tests/infra)
- `tests/infra/test_verify_pipeline_guard.sh` (tests/infra)
- `tests/infra/test_verify_retry_subset.sh` (tests/infra)
- `tests/infra/test_verify_role_prio.sh` (tests/infra)
- `tests/infra/test_verify_semaphore_e2e.sh` (tests/infra)
- `tests/infra/test_verify_semaphore_wiring.sh` (tests/infra)
- `tests/infra/test_verify_test_threads.sh` (tests/infra)
- `tests/infra/test_verify_throughput.sh` (tests/infra)
- `tests/infra/test_warm_base_coherence.sh` (tests/infra)
- `tests/infra/test_warm_lane_audit.sh` (tests/infra)
- `tests/infra/test_warm_lane_boot_persistence.sh` (tests/infra)
- `tests/infra/test_warm_lane_degenerate_ref.sh` (tests/infra)
- `tests/infra/test_warm_lane_disk_guard.sh` (tests/infra)
- `tests/infra/test_warm_lane_gc.sh` (tests/infra)
- `tests/infra/test_warm_lane_gc_sweep.sh` (tests/infra)
- `tests/infra/test_warm_lane_lock_guard.sh` (tests/infra)
- `tests/infra/test_warm_lane_pool.sh` (tests/infra)
- `tests/infra/test_warm_lane_pool_config.sh` (tests/infra)
- `tests/infra/test_warm_lane_preflight.sh` (tests/infra)
- `tests/infra/test_warm_lane_ref_visibility.sh` (tests/infra)
- `tests/infra/test_warm_lane_source_integrity.sh` (tests/infra)
- `tests/infra/test_with_jcodemunch_serve.sh` (tests/infra)

</details>

<details><summary>Unresolved records per source</summary>

- nextest-logs: 0 unresolved; samples: —
- flaky-ledger: 0 unresolved; samples: —
- flake_occurrence: 77 unresolved; samples: `<unknown>`, `<unknown>`, `<unknown>`, `<unknown>`, `<unknown>`

</details>

### What this ranking is not

- No test is retired by this census.
- Retiring any test needs a planted-defect check first (INV-10
  `guards-exercise-behaviour`): show the test goes red on the defect it guards.
- A slow test that never failed is an offline-lane candidate, not a deletion.
- "Never failed" is weak evidence: it means "never seen failing in the windows
  above", and a guard whose invariant nobody broke also never fails.

## Part 2: pinning and duplication across the whole test tree

### Rust test duplication

A test fn is a `fn` whose attributes include one whose path ends in `test`
(`#[test]`, `#[tokio::test(...)]`), in a tracked `.rs` file. Its crate is the
`[package]` name of the nearest enclosing tracked `Cargo.toml`. Rows are in
crate-name order and are not ranked.

- **non-trivial lines**: test-fn body lines, comments blanked, that contain a
  letter or digit after stripping (so `}` and `});` are trivial).
- **duplicated lines**: non-trivial lines whose stripped text occurs 2+ times
  among the crate's test-fn lines, counting every occurrence; **redundant**
  counts occurrences beyond the first.
- **family members**: test fns in a same-file family of 3+, where the family
  key is the fn name minus its last `_segment`.

| crate | test files | test fns | non-trivial lines | duplicated lines | duplicated share | redundant lines | family members | family share | largest family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reify-ast | 4 | 83 | 999 | 441 | 44.1% | 327 | 0 | 0.0% | — |
| reify-audit | 53 | 970 | 18489 | 10778 | 58.3% | 9357 | 7 | 0.7% | 4: `crates/reify-audit/src/bin/reify-audit.rs` `parse_args_unknown_pattern_lists_*` |
| reify-build-utils | 2 | 10 | 108 | 23 | 21.3% | 15 | 0 | 0.0% | — |
| reify-builtins | 7 | 73 | 1081 | 429 | 39.7% | 335 | 0 | 0.0% | — |
| reify-cli | 96 | 604 | 10600 | 5832 | 55.0% | 4932 | 0 | 0.0% | — |
| reify-compiler | 385 | 5761 | 117130 | 81503 | 69.6% | 72095 | 55 | 1.0% | 10: `crates/reify-compiler/src/type_resolution.rs` `resolve_type_name_recognises_*` |
| reify-compute-contract | 1 | 38 | 744 | 401 | 53.9% | 299 | 0 | 0.0% | — |
| reify-config | 9 | 87 | 776 | 454 | 58.5% | 374 | 0 | 0.0% | — |
| reify-constraints | 27 | 616 | 13976 | 8945 | 64.0% | 7450 | 0 | 0.0% | — |
| reify-core | 13 | 586 | 3734 | 1413 | 37.8% | 1107 | 33 | 5.6% | 3: `crates/reify-core/src/dimension.rs` `dimension_*` |
| reify-doc | 7 | 130 | 3474 | 2215 | 63.8% | 1958 | 0 | 0.0% | — |
| reify-doc-build | 2 | 17 | 696 | 331 | 47.6% | 267 | 0 | 0.0% | — |
| reify-eval | 555 | 5708 | 161558 | 108641 | 67.2% | 93654 | 46 | 0.8% | 5: `crates/reify-eval/src/compute_targets/elastic_static.rs` `validate_all_inputs_gate_rejects_malformed_*` |
| reify-eval-fea-tests | 32 | 137 | 5687 | 3352 | 58.9% | 2738 | 0 | 0.0% | — |
| reify-expr | 49 | 1365 | 21383 | 16074 | 75.2% | 13924 | 20 | 1.5% | 4: `crates/reify-expr/tests/collection_eval_tests.rs` `eval_method_sum_*` |
| reify-fdm | 10 | 87 | 1350 | 656 | 48.6% | 513 | 0 | 0.0% | — |
| reify-gcode | 10 | 47 | 415 | 304 | 73.3% | 243 | 0 | 0.0% | — |
| reify-geometry | 1 | 17 | 250 | 109 | 43.6% | 82 | 0 | 0.0% | — |
| reify-gui | 34 | 1191 | 24246 | 15025 | 62.0% | 12532 | 28 | 2.4% | 16: `gui/src-tauri/src/tests/types_tests.rs` `format_value_*` |
| reify-ir | 24 | 1070 | 11323 | 5465 | 48.3% | 4124 | 45 | 4.2% | 12: `crates/reify-ir/src/value.rs` `value_display_*` |
| reify-kernel-conformance | 5 | 10 | 154 | 74 | 48.1% | 45 | 0 | 0.0% | — |
| reify-kernel-fidget | 4 | 35 | 639 | 355 | 55.6% | 256 | 0 | 0.0% | — |
| reify-kernel-gmsh | 31 | 207 | 3679 | 1762 | 47.9% | 1374 | 0 | 0.0% | — |
| reify-kernel-manifold | 10 | 66 | 1528 | 702 | 45.9% | 544 | 0 | 0.0% | — |
| reify-kernel-occt | 68 | 699 | 14396 | 9133 | 63.4% | 7506 | 3 | 0.4% | 3: `crates/reify-kernel-occt/src/lib.rs` `new_ops_export_*` |
| reify-kernel-openvdb | 13 | 116 | 1821 | 938 | 51.5% | 704 | 0 | 0.0% | — |
| reify-lsp | 18 | 574 | 10539 | 5893 | 55.9% | 4907 | 16 | 2.8% | 9: `crates/reify-lsp/src/analysis.rs` `format_value_*` |
| reify-mcp | 10 | 110 | 1206 | 818 | 67.8% | 655 | 0 | 0.0% | — |
| reify-mesh-morph | 17 | 145 | 3628 | 2155 | 59.4% | 1731 | 0 | 0.0% | — |
| reify-runtime | 5 | 82 | 840 | 481 | 57.3% | 370 | 0 | 0.0% | — |
| reify-shell-extract | 13 | 170 | 3806 | 2143 | 56.3% | 1735 | 0 | 0.0% | — |
| reify-solver-elastic | 90 | 1080 | 19276 | 11078 | 57.5% | 8628 | 3 | 0.3% | 3: `crates/reify-solver-elastic/src/sweep.rs` `sweep_rejects_zero_*` |
| reify-spec-conformance | 1 | 7 | 77 | 22 | 28.6% | 12 | 0 | 0.0% | — |
| reify-stdlib | 58 | 2370 | 29539 | 18458 | 62.5% | 15322 | 15 | 0.6% | 3: `crates/reify-stdlib/src/geometry.rs` `frame_to_frame_*` |
| reify-syntax | 73 | 797 | 11185 | 6814 | 60.9% | 5584 | 33 | 4.1% | 7: `crates/reify-syntax/tests/harness_syntax/annotation_tests.rs` `parse_annotation_on_*` |
| reify-test-support | 46 | 710 | 7658 | 3878 | 50.6% | 2994 | 7 | 1.0% | 4: `crates/reify-test-support/src/values.rs` `snapshot_values_builder_*` |
| tree-sitter-reify | 20 | 269 | 3930 | 2144 | 54.6% | 1800 | 0 | 0.0% | — |
| total | 1803 | 26044 | 511920 | 329239 | 64.3% | 280493 | 311 | 1.2% | 16: `gui/src-tauri/src/tests/types_tests.rs` `format_value_*` |

Unreadable files (0; the census is complete):

- none

## Reading against the 2026-09-10 studies

This section is written by hand; everything above it is generated and unedited.
The study is `plans/verify-speed-study-reify-2026-09-10.md`, an untracked file
that exists only in the reify main checkout.

### Where cost comes from

Reify configures no nextest junit, and none is archived anywhere. Per-test cost
therefore comes from the nextest `PASS` and `LEAK` status lines in the archived
verify logs. The study recorded that those logs are failure-biased: passing
merge logs are not archived.

`run_all.sh` logs no per-member wall time, so every `tests/infra/*.sh` member is
uncosted. That is 191 tests: 152 that never failed and 39 in the floor.

Ranked ids are nextest binary ids plus test paths, and they are not resolved
against the tree. A test renamed or removed since its last logged run can
therefore still be ranked.

Ten rows of the top 100 rest on fewer than 3 runs. Ranks 28, 31 and 39 rest on
one run each, and ranks 12, 17, 18, 50, 54, 69 and 78 on two. The median runs
per row in the top 100 is 29. Each run's time depends on the host load it ran
under, so these ten medians are weak evidence, and each may have pushed a
better-evidenced test out of the top 100. Censuses run after this one print
this list under the ranking
(`scripts/suite_census_outcomes.py::THIN_SAMPLE_RUNS`).

### Failure evidence

| measure | study | census |
| --- | --- | --- |
| failed gates / red runs | 46 merge-gate failures in the 30 d to 2026-09-10 | 54 red runs among the 78 logs that carry nextest status or run_all lines, out of 116 logs from 2026-09-04 to 2026-10-04 |
| flaky ledger | 120 entries (59 in 30 d), all `tests/infra/*.sh` | 217 entries (111 in the 30 d to 2026-10-04), 23 distinct tests, all `tests/infra/*.sh` |
| flake_occurrence | — | 77 rows, every one with test_id `<unknown>` and verdict `unconfirmable`; none reaches the floor |

The study counted merge-gate failures only. The census counts logs of every
role, task leg and merge alike.

The floor holds 48 tests:

- 9 Rust tests, from `FAIL` and `TIMEOUT` status lines;
- 39 infra scripts, from run_all `FAILED` lines and the flaky ledger.

### Duplication

| measure | study | census |
| --- | --- | --- |
| non-trivial test lines that are exact duplicates | 45.5% | 64.3% counting every occurrence (329,239 of 511,920); 54.8% counting occurrences beyond the first (280,493) |
| test fns in same-file name families of 3 or more | 16.9% | 1.2% (311 of 26,044) |

The study's definitions lived in its `trimming.md`, a session scratchpad file
that was not retained. So the gap cannot be split between growth and
definition. What is known about the census side:

- lines count only inside test-fn bodies, with comments blanked;
- duplicates are grouped per crate, not across the workspace;
- the family key is strict: the whole fn name minus its last `_segment`,
  within one file.

### No retirement

No test is retired here. Retiring any listed test needs a planted-defect check
(INV-10) in a follow-up. A slow test that guards something real is an
offline-lane candidate, not a deletion.
