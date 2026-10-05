# Test census: dark-factory, 2026-10-04

- Measured tree: `/home/leo/src/dark-factory/.worktrees/5414` at `6d2dbbfe1f2d50f0b3b326dca6b744d61268b966`
- Evidence root: `/home/leo/src/dark-factory`
- Ecosystem: pytest
- Task 5414: a store-only census. No test is retired or changed here.

## Commands used

```
python scripts/suite_census.py --ecosystem pytest --root /home/leo/src/dark-factory --tree /home/leo/src/dark-factory/.worktrees/5414 --project dark-factory --date 2026-10-04 --out plans/test-census-2026-10-04-dark-factory.md
```

Evidence read, per source:

- archived-junit: `data/verify-logs/*/attempt-*.junit-*.xml.gz`
- live-junit: `.worktrees/*/.df-verify-junit/report*.xml`
- pytest-logs: `data/verify-logs/*/attempt-*.test-*.log`
- flake_occurrence: `data/orchestrator/runs.db: SELECT rowid, observed_at, test_id FROM flake_occurrence ORDER BY rowid`

## Part 1: never failed × per-run cost

### Evidence windows

| source | pattern / query | present | artefacts | runs | red runs | red share | first | last | unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| archived-junit | `data/verify-logs/*/attempt-*.junit-*.xml.gz` | yes | 2807 | 2805 | 57 | 2.0% | 2026-09-21T03:45:23Z | 2026-10-04T23:41:55Z | 26696 |
| live-junit | `.worktrees/*/.df-verify-junit/report*.xml` | yes | 17 | 11 | 1 | 9.1% | 2026-08-19T21:26:49Z | 2026-10-04T23:41:54Z | 24 |
| pytest-logs | `data/verify-logs/*/attempt-*.test-*.log` | yes | 310 | 180 | 180 | 100.0% | 2026-09-04T23:38:57Z | 2026-10-04T19:35:10Z | 21 |
| flake_occurrence | `data/orchestrator/runs.db: SELECT rowid, observed_at, test_id FROM flake_occurrence ORDER BY rowid` | yes | 478 | 472 | 472 | 100.0% | 2026-08-30T16:20:21Z | 2026-10-04T19:13:24Z | 6 |

The failed floor holds 165 tests: every test one of the sources above recorded failing inside its window. Failures outside those windows, or never retained, are unseen, so the true failed set is at least this large. 10 floor members have no cost observation (their failure is known only from a log or ledger line).

### Per package

| package | costed tests | failed (floor) | never failed, costed | never failed, uncosted | p50 of medians s | top 1% share of time |
| --- | --- | --- | --- | --- | --- | --- |
| cockpit | 492 | 4 | 488 | 0 | 0.001 | 17.2% |
| dashboard | 3089 | 15 | 3078 | 0 | 0.022 | 23.3% |
| escalation | 1699 | 2 | 1697 | 0 | 0.026 | 17.2% |
| fused-memory | 19895 | 27 | 19869 | 0 | 0.007 | 49.6% |
| orchestrator | 20281 | 83 | 20198 | 0 | 0.619 | 16.2% |
| sampler | 137 | 0 | 137 | 0 | 0.027 | 21.6% |
| scripts | 4341 | 7 | 4334 | 0 | 0.007 | 40.0% |
| shared | 4689 | 15 | 4678 | 0 | 0.002 | 59.2% |
| tests | 2067 | 12 | 2056 | 0 | 0.003 | 47.2% |

### Never failed × per-run cost (top 100 of 56535)

| rank | test | package | runs | median s | max s | total s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `orchestrator/tests/test_verify_classify.py::TestAnchoredSlotTimeoutWithCollateralIsEnvTransient::test_anchored_slot_timeout_with_collateral_is_env_transient` | orchestrator | 312 | 125.845 | 361.977 | 30437.526 |
| 2 | `orchestrator/tests/test_session_registry.py::test_merge_decision_enrichment_pair_is_always_vouched_for` | orchestrator | 313 | 117.208 | 274.473 | 28159.830 |
| 3 | `orchestrator/tests/test_verify_plan.py::TestPlanRecordScopedTargets::test_module_path_scoped_targets_nonempty_exactly_for_file_scoped_runs` | orchestrator | 312 | 95.575 | 254.440 | 23696.499 |
| 4 | `orchestrator/tests/test_verify_classify.py::TestAnchoredSlotTimeoutWithCollateralIsEnvTransient::test_anchored_slot_timeout_with_collateral_keeps_recovery_path` | orchestrator | 312 | 62.294 | 156.792 | 15315.721 |
| 5 | `orchestrator/tests/test_steward_scaffolding_guards.py::TestNoInlineSandboxedProjectRootAsserts::test_no_module_reimplements_the_block` | orchestrator | 312 | 46.776 | 121.484 | 15109.837 |
| 6 | `orchestrator/tests/test_verify_classify.py::TestGenuineHostEventSurvivesRunAllScoping::test_n1_marker_in_preamble_is_detected` | orchestrator | 311 | 44.922 | 135.225 | 10732.291 |
| 7 | `orchestrator/tests/test_verify_plan.py::TestScoperTrailingClausePreservation::test_lockstep_between_the_two_scopers` | orchestrator | 312 | 44.712 | 105.545 | 11174.997 |
| 8 | `orchestrator/tests/test_verify_classify.py::TestDeterministicLintFailureIsNotSemaphoreTimeout::test_deterministic_fault_is_not_semaphore_timeout` | orchestrator | 313 | 44.449 | 125.857 | 10707.123 |
| 9 | `orchestrator/tests/test_verify_classify.py::TestGenuineHostEventSurvivesRunAllScoping::test_n2_marker_inside_the_failing_block_is_detected` | orchestrator | 311 | 44.148 | 99.884 | 10506.446 |
| 10 | `orchestrator/tests/test_verify_classify.py::TestGenuineHostEventSurvivesRunAllScoping::test_n6_marker_under_mismatched_framing_is_detected` | orchestrator | 311 | 44.123 | 148.716 | 10786.247 |
| 11 | `orchestrator/tests/test_verify_classify.py::TestGenuineHostEventSurvivesRunAllScoping::test_n3_marker_in_an_aborted_run_is_detected` | orchestrator | 311 | 43.785 | 108.855 | 10508.078 |
| 12 | `orchestrator/tests/test_verify_classify.py::TestGenuineHostEventSurvivesRunAllScoping::test_n5_marker_in_a_skipped_block_is_detected` | orchestrator | 311 | 43.520 | 124.884 | 10675.228 |
| 13 | `orchestrator/tests/test_verify_classify.py::TestGroundedSlotTimeoutMarkersAreDetected::test_grounded_slot_timeout_line_is_semaphore_timeout` | orchestrator | 313 | 43.360 | 115.118 | 10623.381 |
| 14 | `orchestrator/tests/test_verify_classify.py::TestDeterministicLintFailureIsNotSemaphoreTimeout::test_deterministic_fault_is_not_infra_transient` | orchestrator | 313 | 42.843 | 128.460 | 10735.553 |
| 15 | `orchestrator/tests/test_verify_classify.py::TestGenuineHostEventSurvivesRunAllScoping::test_n4_marker_in_the_tail_is_detected` | orchestrator | 311 | 42.411 | 118.534 | 10495.216 |
| 16 | `orchestrator/tests/test_steward_scaffolding_guards.py::TestStewardConstructionSitesAreCensused::test_every_steward_construction_site_is_sanctioned` | orchestrator | 312 | 41.108 | 128.249 | 13365.179 |
| 17 | `orchestrator/tests/test_steward_scaffolding_guards.py::TestAbsoluteTmpProjectRootLiteralsAreCensused::test_every_absolute_tmp_project_root_literal_is_adjudicated` | orchestrator | 312 | 40.939 | 126.510 | 12733.511 |
| 18 | `orchestrator/tests/test_hard_v2_fixture_pool.py::TestMintedPool::test_reference_unavailable_only_when_no_landing_merge_exists` | orchestrator | 312 | 39.125 | 111.918 | 13207.510 |
| 19 | `fused-memory/tests/test_check_bare_magicmock_config.py::TestWallClockDeadlineBaselineIntegrity::test_no_scanned_file_outside_the_baseline_carries_a_violation` | fused-memory | 316 | 35.448 | 72.612 | 11992.701 |
| 20 | `fused-memory/tests/test_check_bare_magicmock_config.py::TestAllScannedTestDirsClean::test_every_scanned_tests_directory_exits_zero` | fused-memory | 316 | 34.752 | 98.033 | 11865.219 |
| 21 | `dashboard/tests/test_graph_layout_js.py::test_graph_layout_js_suite_passes` | dashboard | 317 | 31.378 | 39.341 | 9000.893 |
| 22 | `orchestrator/tests/test_verify_classify.py::TestMergeVerifyCollateralEnvGuard::test_collateral_shape_is_infra_transient` | orchestrator | 312 | 30.779 | 118.996 | 7754.264 |
| 23 | `orchestrator/tests/test_verify_classify.py::TestMergeVerifyCollateralEnvGuard::test_collateral_shape_is_env_transient` | orchestrator | 313 | 29.083 | 102.398 | 7586.995 |
| 24 | `orchestrator/tests/test_aiosqlite_leak_isolation.py::test_a_thread_exception_actually_fails_a_test_under_this_projects_inifile` | orchestrator | 314 | 27.373 | 147.609 | 9056.686 |
| 25 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_the_replayed_measurement_is_identical_to_the_live_one` | fused-memory | 316 | 27.158 | 76.339 | 9009.470 |
| 26 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestDumpFetchesFlag::test_a_dumping_run_writes_the_cache_and_the_identical_report` | fused-memory | 316 | 26.907 | 65.576 | 9023.442 |
| 27 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_the_replayed_regrowth_block_and_its_descriptor_agree_with_live` | fused-memory | 316 | 26.693 | 81.409 | 8838.971 |
| 28 | `orchestrator/tests/test_lane_state_lib.py::TestAuditLaneStateBehaviourIsUnchanged::test_assigned_column_and_pin` | orchestrator | 313 | 26.506 | 87.783 | 8779.632 |
| 29 | `orchestrator/tests/test_merge_lane_alias_names.py::test_no_tracked_file_reaches_a_missing_name_through_an_alias` | orchestrator | 56 | 26.423 | 58.908 | 1612.613 |
| 30 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_the_replayed_report_discloses_that_it_was_replayed` | fused-memory | 316 | 26.415 | 68.703 | 9009.801 |
| 31 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_a_replayed_run_makes_zero_backend_calls` | fused-memory | 316 | 26.306 | 80.005 | 8979.069 |
| 32 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_it_never_attaches_to_the_shared_durable_write_queue` | fused-memory | 317 | 26.199 | 60.743 | 8857.371 |
| 33 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_a_replay_reads_the_cache_document_exactly_once` | fused-memory | 316 | 25.488 | 74.337 | 8704.605 |
| 34 | `fused-memory/tests/test_bake_off_storage_shape.py::TestMain::test_the_switch_reaches_run_bake_off_rather_than_stopping_at_the_parser` | fused-memory | 316 | 24.372 | 59.738 | 8346.662 |
| 35 | `orchestrator/tests/test_merge_lane_package.py::test_each_pre_package_name_is_its_submodule` | orchestrator | 56 | 23.996 | 60.002 | 1562.035 |
| 36 | `orchestrator/tests/test_roles_harness_tool_inventory.py::test_prompt_constant_names_no_phantom_harness_tool` | orchestrator | 285 | 23.552 | 93.298 | 5519.893 |
| 37 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffRegrowthWiring::test_the_probe_does_not_disturb_the_six_arm_rows` | fused-memory | 316 | 23.395 | 63.042 | 7896.376 |
| 38 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffRegrowthWiring::test_a_probe_less_run_claims_no_provenance_for_the_injection_fixture` | fused-memory | 316 | 23.361 | 56.790 | 7900.317 |
| 39 | `orchestrator/tests/test_cited_test_class_drift.py::TestCitedTestClassDrift::test_every_cited_test_class_resolves` | orchestrator | 313 | 22.599 | 58.230 | 7653.645 |
| 40 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::TestSanctionedNameMirrors::test_every_mirrored_name_is_really_defined` | orchestrator | 312 | 22.570 | 96.690 | 7473.625 |
| 41 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::TestSanctionedNameMirrors::test_every_mirror_matches_its_real_definition` | orchestrator | 312 | 22.451 | 61.169 | 7386.072 |
| 42 | `scripts/tests/test_check_write_triage_flip_preconditions.py::TestVerdictReadingIsNotRaceProne::test_repeated_runs_against_a_passing_judge_all_agree` | scripts | 307 | 22.371 | 82.955 | 7375.804 |
| 43 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::TestRow7SceneIsGuarded::test_the_row7_marker_is_the_derived_constant_not_a_literal` | orchestrator | 312 | 22.088 | 89.593 | 7296.462 |
| 44 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::TestDeepLandingModuleMarkers::test_the_module_has_no_in_band_marker_sites` | orchestrator | 289 | 21.266 | 72.826 | 6443.715 |
| 45 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::test_grandfather_allowlist_has_no_stale_entries` | orchestrator | 289 | 21.248 | 50.788 | 6242.312 |
| 46 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::test_the_marker_census_is_not_vacuous` | orchestrator | 312 | 21.037 | 70.870 | 6835.690 |
| 47 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::test_no_new_inverting_timeout_marker` | orchestrator | 289 | 20.967 | 65.794 | 6473.791 |
| 48 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::TestDeepLandingModuleMarkers::test_every_marker_site_is_spelled_as_the_named_constant` | orchestrator | 312 | 20.950 | 62.167 | 6911.747 |
| 49 | `orchestrator/tests/test_mcp_post_transport.py::test_no_raw_post_builds_a_trailing_slash_mcp_url` | orchestrator | 110 | 20.444 | 49.488 | 2467.101 |
| 50 | `tests/scripts/test_orchestrator_watchdog.py::test_boundary4_defers_busy_unit_while_others_proceed` | tests | 609 | 20.106 | 20.638 | 8143.782 |
| 51 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::test_the_deep_gate_module_pins_every_timeout_by_name` | orchestrator | 312 | 20.015 | 89.472 | 6245.066 |
| 52 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::TestDeepLandingModuleMarkers::test_the_census_is_not_vacuous` | orchestrator | 312 | 19.917 | 93.716 | 6285.510 |
| 53 | `fused-memory/tests/test_integration_marker_real_service.py::test_refresh_entity_summary_live_class_gated` | fused-memory | 317 | 19.106 | 56.053 | 6471.891 |
| 54 | `orchestrator/tests/test_marker_registration_drift.py::TestMarkerRegistrationDrift::test_every_marker_applied_under_tests_is_registered` | orchestrator | 313 | 18.814 | 57.500 | 6433.517 |
| 55 | `fused-memory/tests/test_store_mutation_preflight_contract.py::TestNoHandRolledNeutraliseFixture::test_no_test_module_defines_its_own_neutralise_fixture` | fused-memory | 297 | 18.712 | 48.331 | 5871.571 |
| 56 | `orchestrator/tests/test_timeout_marker_inversion_guard.py::test_no_timeout_marker_sits_in_the_inversion_band` | orchestrator | 23 | 18.699 | 47.308 | 475.749 |
| 57 | `orchestrator/tests/test_marker_registration_drift.py::TestMarkerRegistrationDrift::test_the_sweep_is_not_vacuous` | orchestrator | 313 | 18.465 | 57.757 | 6316.772 |
| 58 | `orchestrator/tests/test_merge_queue_deep_integration_gate.py::TestDeepGateCapstone::test_a_mixed_run_conserves_and_the_canary_agrees_with_git` | orchestrator | 312 | 18.167 | 68.679 | 6001.601 |
| 59 | `orchestrator/tests/test_verify_categories.py::TestShouldArchiveCategoryDelegatesToTable::test_matches_table_lookup_for_every_known_category` | orchestrator | 1 | 17.980 | 17.980 | 17.980 |
| 60 | `orchestrator/tests/test_verdict_tools_markup_registration.py::TestCorpusReplayAgainstRealServer::test_specimen` | orchestrator | 312 | 17.873 | 57.932 | 5277.287 |
| 61 | `fused-memory/tests/test_store_mutation_preflight_contract.py::TestNoHandRolledDenyRaiser::test_no_test_module_raises_store_mutation_unavailable` | fused-memory | 297 | 17.854 | 79.302 | 5539.896 |
| 62 | `orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment::test_forbidden_import_is_rejected` | orchestrator | 313 | 17.529 | 63.481 | 5787.776 |
| 63 | `fused-memory/tests/test_store_mutation_preflight_contract.py::TestNoInlinedFailClosedMarker::test_no_test_module_spells_the_fail_closed_marker` | fused-memory | 297 | 17.303 | 56.415 | 5236.401 |
| 64 | `orchestrator/tests/test_whole_tree_scan_timeout_guard.py::test_whole_tree_scanners_carry_module_level_timeout_mark` | orchestrator | 311 | 17.240 | 68.322 | 5949.204 |
| 65 | `orchestrator/tests/test_outcome_kind.py::TestOutcomeKindPayloadIdentity::test_payload_identity_through_real_chokepoint` | orchestrator | 313 | 17.160 | 48.268 | 4655.726 |
| 66 | `fused-memory/tests/test_integration_marker_real_service.py::test_mem0_client_qdrant_probe_gated` | fused-memory | 317 | 17.032 | 54.197 | 5840.955 |
| 67 | `fused-memory/tests/test_integration_marker_config.py::test_real_embedder_test_gated_by_default_and_selectable_via_marker` | fused-memory | 317 | 16.900 | 44.358 | 5680.433 |
| 68 | `orchestrator/tests/test_eval_fixture_reference.py::test_backfilled_fixture_reference_diff_materializes` | orchestrator | 313 | 16.767 | 83.207 | 4995.137 |
| 69 | `orchestrator/tests/test_eval_fixture_reference.py::test_declared_diff_stat_matches_the_landed_diff` | orchestrator | 313 | 16.690 | 83.153 | 4971.796 |
| 70 | `orchestrator/tests/test_event_loop_antipattern_guard.py::test_no_get_event_loop_in_orchestrator_tests` | orchestrator | 313 | 16.563 | 53.018 | 5809.643 |
| 71 | `orchestrator/tests/test_merge_queue_invariant_integration_gate.py::TestScenario9GuardMatrixEquivalence::test_merger_and_remerge_paths_agree` | orchestrator | 1 | 16.379 | 16.379 | 16.379 |
| 72 | `orchestrator/tests/test_leaked_task_drain.py::test_without_the_drain_the_leak_hangs_loop_teardown_until_killed` | orchestrator | 231 | 16.184 | 21.264 | 3788.024 |
| 73 | `orchestrator/tests/test_config.py::TestConfigReload::test_representative_reloadable_members_present` | orchestrator | 314 | 15.822 | 44.038 | 3929.164 |
| 74 | `orchestrator/tests/test_reconcile_stranded.py::TestReconcileStrandedInProgress::test_reconcile_lock_format_variants` | orchestrator | 1 | 15.394 | 15.394 | 15.394 |
| 75 | `orchestrator/tests/test_eval_fixture_reference.py::test_fixture_with_a_landed_commit_carries_a_reference_block` | orchestrator | 313 | 15.240 | 50.103 | 3966.945 |
| 76 | `fused-memory/tests/test_check_asyncmock_assertion_style.py::TestRealTestsDirectoryIsClean::test_real_fused_memory_tests_directory_is_clean_under_check` | fused-memory | 317 | 14.933 | 44.773 | 5254.106 |
| 77 | `orchestrator/tests/test_merge_queue_speculative_probe.py::TestSelectProbeDepthByteIdenticalAtZero::test_zero_fraction_always_none` | orchestrator | 313 | 14.363 | 84.577 | 3914.142 |
| 78 | `orchestrator/tests/test_multihost_verify_integration.py::TestUnreachableHostCapstone::test_a_cancel_against_a_down_host_parks_the_slot_and_reprobe_unparks_it` | orchestrator | 312 | 14.088 | 27.171 | 4508.823 |
| 79 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_it_holds_a_live_lease_at_the_moment_it_seeds` | fused-memory | 315 | 13.951 | 37.555 | 4666.315 |
| 80 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_the_reaper_is_resolved_before_any_resource_is_acquired` | fused-memory | 315 | 13.898 | 39.553 | 4673.405 |
| 81 | `orchestrator/tests/test_warm_lane_scripts_shipped.py::TestGcBaseTargetMatchesDirname::test_derived_base_target_matches_dirname` | orchestrator | 312 | 13.835 | 41.199 | 4577.680 |
| 82 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_the_lease_is_released_once_the_run_returns` | fused-memory | 315 | 13.645 | 57.618 | 4675.607 |
| 83 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestDumpFetchesFlag::test_the_dump_covers_every_shape_and_carries_its_fingerprint` | fused-memory | 316 | 13.590 | 34.801 | 4648.720 |
| 84 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_it_clears_the_api_key_so_a_real_embedder_is_used` | fused-memory | 317 | 13.262 | 35.208 | 4557.813 |
| 85 | `tests/scripts/test_pyright_version_pin.py::test_npx_in_a_fresh_worktree_does_not_resolve_the_pin_on_its_own` | tests | 609 | 13.256 | 67.046 | 9528.603 |
| 86 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_a_replay_whose_regrowth_pass_is_missing_a_probe_cluster_exits_three` | fused-memory | 316 | 13.245 | 67.367 | 4612.561 |
| 87 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_a_replay_against_drifted_fixtures_exits_nonzero` | fused-memory | 316 | 13.235 | 30.601 | 4596.344 |
| 88 | `orchestrator/tests/test_session_registry.py::test_normalize_project_token_table` | orchestrator | 312 | 13.216 | 51.518 | 3274.656 |
| 89 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_the_ephemeral_prefix_is_set_on_the_config_not_just_the_project_id` | fused-memory | 317 | 13.149 | 47.314 | 4674.252 |
| 90 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_the_distractor_slab_is_seeded_into_every_arm` | fused-memory | 317 | 13.109 | 36.694 | 4475.926 |
| 91 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_seeding_is_bounded_rather_than_one_unbounded_gather` | fused-memory | 317 | 13.036 | 33.510 | 4482.928 |
| 92 | `orchestrator/tests/test_briefing_progress_spot.py::test_step_status_does_not_change_the_implementer_prompt` | orchestrator | 2 | 13.018 | 16.360 | 26.036 |
| 93 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_the_probing_write_is_never_its_own_guard_match` | fused-memory | 317 | 12.984 | 51.939 | 4509.333 |
| 94 | `orchestrator/tests/test_merge_speculation.py::TestSpecLaneAbortReleasesLane::test_waiter_walking_away_releases_spec_lane` | orchestrator | 312 | 12.973 | 19.640 | 4119.509 |
| 95 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_it_stubs_the_xdist_contended_history_writer` | fused-memory | 317 | 12.948 | 36.435 | 4393.917 |
| 96 | `orchestrator/tests/test_merge_speculation.py::TestSpecLaneAbortReleasesLane::test_operator_halt_releases_spec_lane` | orchestrator | 312 | 12.934 | 20.115 | 4119.024 |
| 97 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_it_seeds_exactly_three_collections_for_six_arms` | fused-memory | 317 | 12.905 | 50.464 | 4432.281 |
| 98 | `fused-memory/tests/test_bake_off_fetch_cache.py::TestReplayFetchesFlag::test_a_replay_against_an_edited_query_set_exits_nonzero` | fused-memory | 316 | 12.898 | 48.855 | 4455.141 |
| 99 | `orchestrator/tests/test_session_registry.py::test_normalize_project_token_is_idempotent` | orchestrator | 312 | 12.890 | 44.070 | 3308.994 |
| 100 | `fused-memory/tests/test_bake_off_storage_shape.py::TestRunBakeOffWiring::test_the_guard_probe_over_fetches_to_cover_its_own_removal` | fused-memory | 317 | 12.887 | 34.020 | 4469.679 |

<details><summary>Failed floor (165 tests)</summary>

- `cockpit/tests/test_app.py::TestBoostReordersAndPersists::test_boost_and_digit_reorder_live_and_persist` (cockpit)
- `cockpit/tests/test_priority.py::TestDroppedBelowOpen::test_answered_scores_below_open` (cockpit)
- `cockpit/tests/test_priority.py::TestDroppedBelowOpen::test_dropped_scores_below_open` (cockpit)
- `cockpit/tests/test_weight_editor.py::TestKnownProjects::test_empty_project_names_excluded_and_never_raise` (cockpit)
- `dashboard/tests/test_api_curator_cancel.py::test_two_url_all_unreachable_invalidates_each_session_in_order` (dashboard)
- `dashboard/tests/test_app.py::TestBoostReordersAndPersists::test_boost_and_digit_reorder_live_and_persist` (dashboard)
- `dashboard/tests/test_config.py::TestOrchestratorConfigSccache::test_effective_verify_env_merges_sccache_backend` (dashboard)
- `dashboard/tests/test_config.py::TestRecoveryEmissionConfig::test_section_is_registered_on_orchestrator_config` (dashboard)
- `dashboard/tests/test_config.py::TestStarvationWatchdogConfig::test_idle_only_secs_default` (dashboard)
- `dashboard/tests/test_dashboard_endpoints_survive_hung_mcp.py::test_every_dashboard_endpoint_survives_a_hung_fetch_tasks` (dashboard)
- `dashboard/tests/test_escalations_data.py::TestFetchPinsRecovery::test_timeout_maps_to_none` (dashboard)
- `dashboard/tests/test_index_html.py::test_redux_cache_buster_is_newer_than_merge_base` (dashboard)
- `dashboard/tests/test_memory.py::TestHandRolledLoopsBoundEachUrl::test_get_queue_stats_skips_the_hung_url_and_aggregates_the_rest` (dashboard)
- `dashboard/tests/test_merge_halt.py::TestGetMergeHaltStatus::test_timeout_yields_offline_entry` (dashboard)
- `dashboard/tests/test_orchestrator.py::TestDiscoverOrchestratorsBudget::test_the_loop_deadline_truncates_a_root_share_not_the_per_root_budget` (dashboard)
- `dashboard/tests/test_orchestrator.py::TestDiscoverOrchestratorsBudget::test_two_pids_sharing_one_root_pay_the_budget_once` (dashboard)
- `dashboard/tests/test_scheduler_page.py::test_override_endpoint_rejects_invalid_body` (dashboard)
- `dashboard/tests/test_tab_burndown.py::TestVelocitySparkWiring::test_completed_window_tile_stays_cumulative` (dashboard)
- `dashboard/tests/test_write_journal.py::TestOperationsBreakdownMissingIndexFallback::test_missing_index_falls_back_to_unhinted_with_error_log` (dashboard)
- `escalation/tests/test_capability_guard_http.py::TestIdentityDerivedCeiling::test_mapped_identity_levels_header_cannot_widen_past_ceiling` (escalation)
- `escalation/tests/test_capability_guard_http.py::TestL2AutoCloseCarveout::test_action_other_than_close_only_still_forbidden` (escalation)
- `fused-memory/tests/test_census_memory_metadata.py::TestCommittedParamsAreCheckoutIndependent::test_run_records_the_default_paths_repo_relative` (fused-memory)
- `fused-memory/tests/test_check_bare_magicmock_config.py::TestHooksIntegration::test_hook_invokes_check_with_python3_not_uv_run` (fused-memory)
- `fused-memory/tests/test_check_bare_magicmock_config.py::TestWallClockDeadlineBaselineIntegrity::test_recorded_budgets_match_the_live_census_exactly` (fused-memory)
- `fused-memory/tests/test_lock_charter_guard.py::test_every_tracked_extension_is_allowlisted` (fused-memory)
- `fused-memory/tests/test_memory_eval_retrieval_probe.py::TestDerivationIsPure::test_payload_is_shaped_like_the_registry` (fused-memory)
- `fused-memory/tests/test_memory_eval_retrieval_probe.py::TestDeriveFromCensus::test_emits_multi_entry_topics_only` (fused-memory)
- `fused-memory/tests/test_memory_eval_retrieval_probe.py::TestDeriveFromCensus::test_skipped_singletons_are_disclosed_not_silently_dropped` (fused-memory)
- `fused-memory/tests/test_memory_eval_retrieval_probe.py::TestDeriveFromGuardClusters::test_emits_one_candidate_per_guard_slug` (fused-memory)
- `fused-memory/tests/test_recon_write_policy.py::TestBothInterceptorCallSitesAwaitCheck::test_update_task_check_does_not_block_the_event_loop` (fused-memory)
- `fused-memory/tests/test_referent_queue_threading.py::TestReferentsSurviveTheRealQueue::test_a_new_format_row_survives_the_sqlite_round_trip` (fused-memory)
- `fused-memory/tests/test_referent_queue_threading.py::TestReferentsSurviveTheRealQueue::test_an_old_format_row_still_executes_end_to_end` (fused-memory)
- `fused-memory/tests/test_scheduler_state_tools.py::TestSnapshotPerformance::test_read_scheduler_state_under_50ms_for_1500_tasks` (fused-memory)
- `fused-memory/tests/test_script_loader_routing_guard.py::test_loads_scripts_through_the_shared_helper` (fused-memory)
- `fused-memory/tests/test_stages.py::TestDisallowedToolLists::test_every_escalation_server_tool_is_classified` (fused-memory)
- `fused-memory/tests/test_store_mutation_conformance.py::TestGuardedScriptCensus::test_guarded_script_census_matches_the_reviewed_column` (fused-memory)
- `fused-memory/tests/test_ticket_worker.py::TestCuratorWorkerBatchDrain::test_curator_lock_held_across_entire_batch` (fused-memory)
- `fused-memory/tests/test_write_journal.py::TestOperationsBreakdownMissingIndexFallback::test_missing_index_falls_back_to_unhinted_with_error_log` (fused-memory)
- `fused-memory/tests/test_write_journal.py::test_causation_id_queries_both_layers` (fused-memory)
- `fused-memory/tests/test_write_journal.py::test_session_id_persists` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestEnsureEntityNode::test_created_at_matches_graphiti_cores_wire_format` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestEnsureEntityNode::test_embedder_failure_is_swallowed_and_mint_still_returns` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestEnsureEntityNode::test_embedding_write_failure_is_swallowed` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestEnsureEntityNode::test_group_id_is_canonicalized` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestEnsureEntityNode::test_idempotent_second_call_mints_nothing` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestEnsureEntityNode::test_mint_regenerates_name_embedding` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestGroupIdScopingAmendment::test_find_duplicate_entity_nodes_filters_by_group_id` (fused-memory)
- `fused-memory/tests/test_write_time_identity.py::TestResolveOrCreateEntityResolve::test_zero_matches_returns_none_without_minting` (fused-memory)
- `orchestrator/tests/test_cli.py::test_cancel_verify_real_impl_dead_pgid` (orchestrator)
- `orchestrator/tests/test_cli.py::test_verify_merge_cancel_end_to_end` (orchestrator)
- `orchestrator/tests/test_coalesce_integration_gate.py::TestScenario2::test_partial_stackability_overlap_keeps_solo` (orchestrator)
- `orchestrator/tests/test_coalesce_integration_gate.py::TestScenario3::test_confidence_gate_excludes_blocked_member` (orchestrator)
- `orchestrator/tests/test_concurrent_verify_boundary.py::TestB1OverlapOrderedAdvance::test_b1_overlapping_spans_and_ordered_advance` (orchestrator)
- `orchestrator/tests/test_concurrent_verify_boundary.py::TestB3HostDownMidOverlap::test_b3_host_down_mid_overlap_zero_stall` (orchestrator)
- `orchestrator/tests/test_concurrent_verify_boundary.py::TestHarnessSmokeTest::test_two_host_harness_runs_single_item_green` (orchestrator)
- `orchestrator/tests/test_config.py::TestOrchestratorConfigSccache::test_effective_verify_env_merges_sccache_backend` (orchestrator)
- `orchestrator/tests/test_config.py::TestRecoveryEmissionConfig::test_section_is_registered_on_orchestrator_config` (orchestrator)
- `orchestrator/tests/test_config.py::TestStarvationWatchdogConfig::test_idle_only_secs_default` (orchestrator)
- `orchestrator/tests/test_fleet_staleness_composition.py::TestBurstCoalescingUnderCommittedConfig::test_burst_of_two_merges_fires_exactly_once` (orchestrator)
- `orchestrator/tests/test_fleet_staleness_composition.py::TestOrchestratorCoordinatorCommittedConfigComposition::test_coordinator_fields_match_committed_fleet_config` (orchestrator)
- `orchestrator/tests/test_fleet_staleness_composition.py::TestOrchestratorCoordinatorCommittedConfigComposition::test_fires_systemd_run_with_fleet_script_via_committed_config` (orchestrator)
- `orchestrator/tests/test_harness_deterministic_recon_sweep.py::TestDeterministicReconStreakRelease::test_a_later_recurrence_files_a_new_alarm` (orchestrator)
- `orchestrator/tests/test_harness_deterministic_recon_sweep.py::TestDeterministicReconStreakRelease::test_a_sustained_hold_files_one_alarm` (orchestrator)
- `orchestrator/tests/test_harness_deterministic_recon_sweep.py::TestDeterministicReconStreakRelease::test_the_next_pass_after_the_hold_clears_resolves_the_alarm` (orchestrator)
- `orchestrator/tests/test_harness_deterministic_recon_sweep.py::TestDeterministicReconStreakReleaseIsSiteScoped::test_a_deterministic_pass_will_not_resolve_a_still_held_alarm` (orchestrator)
- `orchestrator/tests/test_harness_deterministic_recon_sweep.py::TestDeterministicReconStreakReleaseIsSiteScoped::test_a_reconcile_release_will_not_resolve_a_deterministic_hold` (orchestrator)
- `orchestrator/tests/test_harness_plan_step_rederive.py::TestInterIterationRebaseRederivesStepStatus::test_multi_step_log_entry_is_not_trusted_as_per_step_provenance` (orchestrator)
- `orchestrator/tests/test_laptop_warm_verify_boundary.py::test_heartbeat_starved_hard_partition_tree_killed_via_timeout` (orchestrator)
- `orchestrator/tests/test_laptop_warm_verify_boundary.py::test_kill_holder_tree_reaps_a_session_escaped_grandchild` (orchestrator)
- `orchestrator/tests/test_laptop_warm_verify_boundary.py::test_orchestrator_killed_mid_build_tree_killed_via_eof` (orchestrator)
- `orchestrator/tests/test_laptop_warm_verify_boundary.py::test_read_direct_children_sees_a_real_fork_including_off_main_thread` (orchestrator)
- `orchestrator/tests/test_laptop_warm_verify_boundary.py::test_watchdog_timeout_env_override_fires_fast_without_heartbeat` (orchestrator)
- `orchestrator/tests/test_merge_lane_ratchet.py::TestCheckCli::test_check_hands_build_report_the_resolved_repo_root` (orchestrator)
- `orchestrator/tests/test_merge_lane_ratchet.py::TestCheckCli::test_check_is_clean_against_the_committed_baseline` (orchestrator)
- `orchestrator/tests/test_merge_lane_ratchet.py::TestCheckCli::test_check_resolves_its_defaults_and_exits_clean` (orchestrator)
- `orchestrator/tests/test_merge_lane_ratchet.py::TestPatchTargets::test_real_tree_union_anchor` (orchestrator)
- `orchestrator/tests/test_merge_lane_ratchet.py::test_baseline_matches_a_fresh_measurement` (orchestrator)
- `orchestrator/tests/test_merge_lane_ratchet.py::test_merge_lane_ratchet_holds` (orchestrator)
- `orchestrator/tests/test_merge_queue_concurrent_verify.py::TestRedispatchSpeculativeConservation::test_speculative_redispatch_item_stays_counted_through_real_pipeline` (orchestrator)
- `orchestrator/tests/test_merge_queue_deep_integration_gate.py::TestRow11TimeoutMargin::test_a_depth_sixteen_tip_verify_is_handed_the_merge_cold_budget` (orchestrator)
- `orchestrator/tests/test_merge_queue_deep_integration_gate.py::TestRow7KillSwitchByteIdentity::test_a_restarted_worker_inherits_no_halving_suspicion` (orchestrator)
- `orchestrator/tests/test_merge_queue_deep_integration_gate.py::TestRow7KillSwitchByteIdentity::test_the_kill_switched_run_matches_the_golden_transcript` (orchestrator)
- `orchestrator/tests/test_merge_queue_deep_integration_gate.py::TestRow7KillSwitchByteIdentity::test_the_same_sequence_at_cap_six_moves_every_deep_field` (orchestrator)
- `orchestrator/tests/test_merge_queue_deep_landing.py::TestStaleCasAbortLeavesTheRestAlone::test_two_consecutive_tip_fails_render_nothing_for_any_link` (orchestrator)
- `orchestrator/tests/test_merge_queue_request_liveness.py::TestContendedLeaseDefers::test_stale_streak_stamp_does_not_cap_a_fresh_defer` (orchestrator)
- `orchestrator/tests/test_merge_queue_request_liveness.py::TestDeadInflightVerifyAborts::test_remote_lease_dispatch_returning_mid_verify_with_real_merge_wt_writes_is_not_aborted` (orchestrator)
- `orchestrator/tests/test_merge_queue_resolve_release.py::TestCascadeErrorChokepoint::test_cascade_remerge_error_routes_through_chokepoint` (orchestrator)
- `orchestrator/tests/test_merge_speculation.py::TestLateArrivalGuards::test_attached_late_arrival_skip_verify_false` (orchestrator)
- `orchestrator/tests/test_multihost_verify_integration.py::TestTwoHostFalseGreenCapstone::test_b_false_green_is_withheld_quarantined_and_halts_the_lane` (orchestrator)
- `orchestrator/tests/test_offline_lane.py::test_loop_retries_after_red_handling_exception` (orchestrator)
- `orchestrator/tests/test_reify_multi_account.py::TestDarkFactoryProductionPool::test_production_pool_accounts_in_expected_order` (orchestrator)
- `orchestrator/tests/test_routing_integration_gate.py::TestCeilingFallbackDoesNotBlockDispatch::test_ceiling_exhausted_falls_back_without_blocking` (orchestrator)
- `orchestrator/tests/test_session_registry.py::test_main_lease_show_on_a_corrupt_body_is_fail_soft` (orchestrator)
- `orchestrator/tests/test_verify_clock_stop.py::TestRunCmdClockStop::test_happy_exclude_span` (orchestrator)
- `orchestrator/tests/test_workflow_claimant.py::test_heartbeat_loop_refreshes_heartbeat_only` (orchestrator)
- `orchestrator/tests/test_workflow_zero_output_hang.py::TestRecycleConfigDirLoopWiring::test_recycle_called_on_subthreshold_iterations` (orchestrator)
- `orchestrator/tests/test_worktree_namespace_c2.py::test_merge_prefixed_names_classify_merge` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestEmitZeroProgressRequeueAlert::test_below_threshold_files_nothing` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestEmitZeroProgressRequeueAlert::test_distinct_tasks_get_distinct_alerts` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestEmitZeroProgressRequeueAlert::test_emits_zero_progress_requeue_event` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestEmitZeroProgressRequeueAlert::test_fires_at_threshold_with_monitor_alarm_shape` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestEmitZeroProgressRequeueAlert::test_raising_event_store_does_not_unfile_the_escalation` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestEmitZeroProgressRequeueAlert::test_raising_queue_does_not_propagate` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_alert_emitter_failure_does_not_propagate` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_disabled_never_fires` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_disabled_still_tracks_so_progress_resets_the_streak` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_done_between_requeues_resets` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_existing_retry_cap_contract_unchanged` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_fires_exactly_at_threshold` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_fires_for_non_counting_dispositions` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_healthy_dispatches_never_touch_the_queue` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_recovery_resolves_the_alarm_and_rearms_the_detector` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_recovery_runs_even_while_disabled` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_requeue_with_real_work_resets` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_run_id_guard_still_short_circuits` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_short_span_does_not_fire` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestHarnessZeroProgressWiring::test_tracker_failure_does_not_propagate` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestResolveZeroProgressRequeueAlert::test_leaves_unrelated_escalations_alone` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestResolveZeroProgressRequeueAlert::test_long_streak_checks_disk_even_without_a_memo_entry` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestResolveZeroProgressRequeueAlert::test_no_queue_is_a_silent_no_op` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestResolveZeroProgressRequeueAlert::test_nothing_pending_returns_false` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestResolveZeroProgressRequeueAlert::test_raising_queue_does_not_propagate` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestResolveZeroProgressRequeueAlert::test_recurrence_after_recovery_files_again` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueConfig::test_defaults_yaml_block_matches_the_field_defaults` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueConfig::test_leaves_are_green_tier_hot_reloadable` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueConfig::test_min_span_seconds_must_not_be_negative` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueConfig::test_section_is_registered_on_orchestrator_config` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueConfig::test_threshold_must_be_at_least_one` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueTracker::test_every_non_requeue_outcome_resets` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueTrackerSpan::test_default_clock_is_monotonic` (orchestrator)
- `orchestrator/tests/test_zero_progress_requeue.py::TestZeroProgressRequeueTrackerSpan::test_span_of_untracked_task_is_zero` (orchestrator)
- `scripts/tests/test_design_invariants_consistency.py::test_every_enumeration_site_is_pinned` (scripts)
- `scripts/tests/test_install_memory_metadata_coverage_census_timer.py::test_wrapper_never_invokes_a_forbidden_git_verb` (scripts)
- `scripts/tests/test_lms_healthcheck.py::test_a_vllm_shaped_null_prompt_tokens_details_still_produces_both_latencies` (scripts)
- `scripts/tests/test_lms_healthcheck.py::test_the_table_lists_who_else_held_the_card_at_each_reading` (scripts)
- `scripts/tests/test_remove_lms_dropin_wrapper.py::test_shell_selftest_passes` (scripts)
- `scripts/tests/test_restart_all_orchestrators.py::test_busy_unit_that_drains_mid_defer_resumes_and_restarts` (scripts)
- `scripts/tests/test_sitting_nightly_prepare.py::test_builder_temp_files_are_unlinked_after_the_run` (scripts)
- `shared/tests/test_async_sqlite_base.py::TestAtomicConnectionSnapshotPin::test_legacy_multi_hop_read_still_loses_the_race` (shared)
- `shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus::test_checked_in_manifest_prd_field_matches_its_own_filename` (shared)
- `shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus::test_checked_in_manifest_validates` (shared)
- `shared/tests/test_cli_invoke_integration.py::TestCrossAccountResume::test_invoke_returns_session_id` (shared)
- `shared/tests/test_cli_invoke_integration.py::TestCrossAccountResume::test_session_resume_preserves_context_across_accounts` (shared)
- `shared/tests/test_cli_invoke_integration.py::TestCrossAccountResume::test_session_resume_same_account_baseline` (shared)
- `shared/tests/test_loop_blocking_gate.py::TestKnownSiteFloor::test_unfiled_curator_sites_are_found` (shared)
- `shared/tests/test_loop_blocking_gate.py::TestRatchet::test_no_stale_blessings` (shared)
- `shared/tests/test_loop_blocking_gate.py::TestRatchet::test_no_unblessed_findings` (shared)
- `shared/tests/test_proc_group.py::TestTerminateProcessGroup::test_terminate_process_group_kills_real_subprocess` (shared)
- `shared/tests/test_proc_group.py::TestTerminateProcessGroup::test_terminate_process_group_reaps_grandchildren` (shared)
- `shared/tests/test_silent_fallthrough_gate.py::TestAllowlistIntegrity::test_no_stale_entries` (shared)
- `shared/tests/test_silent_fallthrough_gate.py::TestWholeTreeGate::test_no_violations_outside_allowlist` (shared)
- `shared/tests/test_startup_completion_fixtures.py::TestLiveReprobe::test_c_record_type_prefix_still_matches_the_committed_row` (shared)
- `shared/tests/test_vllm_bridge.py::TestVllmBridgeIntegration::test_handles_truncated_json_upstream_response` (shared)
- `tests/scripts/test_atomic_write_regrowth.py::TestNoRegrownAtomicWriters::test_atomic_write_text_helpers_only_delegate` (tests)
- `tests/scripts/test_atomic_write_regrowth.py::TestNoRegrownAtomicWriters::test_no_unapproved_renamers_in_source_trees` (tests)
- `tests/scripts/test_drain_process_leak_isolation.py::TestRunInNewSession::test_a_timeout_kills_the_backgrounded_grandchild_too` (tests)
- `tests/scripts/test_orchestrator_restart_config_drift.py::test_orchestrator_restart_config_round_trips_through_config_model` (tests)
- `tests/scripts/test_orchestrator_restart_config_drift.py::test_orchestrator_restart_on_merge_enabled_is_true` (tests)
- `tests/scripts/test_orchestrator_service_files.py::test_reify_and_df_differ_only_in_config_and_description` (tests)
- `tests/scripts/test_orchestrator_watchdog.py::test_boundary_run_drain_script_timeout_kills_the_whole_process_group` (tests)
- `tests/scripts/test_reify_closure_staleness_sweep_retired.py::test_tree_carries_no_reference_to_the_retired_wiring` (tests)
- `tests/scripts/test_spawn_claude.py::test_tmux_backend_routes_and_runs_session` (tests)
- `tests/scripts/test_spawn_claude.py::test_transcript_appearance_suppresses_flag` (tests)
- `tests/scripts/test_verify_command_invariants.py::test_flag_args_scope_is_the_callers_choice_not_a_default` (tests)
- `tests/scripts/test_worktree_ruff_config_boundary.py::TestCacheRedirectIsRuleNeutral::test_redirect_changes_neither_the_settings_path_nor_the_rules` (tests)

</details>

<details><summary>Never failed, uncosted (0 tests)</summary>



</details>

<details><summary>Unresolved records per source</summary>

- archived-junit: 26696 unresolved; samples: `tests.reconciliation.test_enumeration_guard.TestPassThrough::test_non_recon_callers_are_never_touched[False-claude-interactive]`, `tests.reconciliation.test_enumeration_guard.TestPassThrough::test_non_dict_recon_metadata_passes_through_verbatim[None]`, `tests.reconciliation.test_enumeration_guard.TestPassThrough::test_non_recon_callers_are_never_touched[True-None]`, `tests.reconciliation.test_enumeration_guard.TestPassThrough::test_non_recon_callers_are_never_touched[True-claude-interactive]`, `tests.reconciliation.test_enumeration_guard.TestPassThrough::test_non_recon_callers_are_never_touched[True-orchestrator]`
- live-junit: 24 unresolved; samples: `tests.test_merge_queue_reachback_patch_guard::test_find_merge_queue_private_patches_ignores_docstring_mention`, `tests.test_merge_queue_reachback_patch_guard::test_find_merge_queue_private_patches_flags_object_form_dotted_attribute_chain`, `tests.test_merge_queue_reachback_patch_guard::test_forbidden_reachback_names_from_source_handles_relative_imports`, `tests.test_merge_queue_reachback_patch_guard::test_find_merge_queue_private_patches_ignores_unrelated_merge_queue_attribute`, `tests.test_merge_queue_reachback_patch_guard::test_find_merge_queue_private_patches_flags_monkeypatch_setattr`
- pytest-logs: 21 unresolved; samples: `scripts/tests/test_restart_all_orchestrators.py`, `scripts/tests/test_reviewer_redundancy_diagnostic.py`, `scripts/tests/test_restart_all_orchestrators.py`, `scripts/tests/test_reviewer_redundancy_diagnostic.py`, `scripts/tests/test_restart_all_orchestrators.py`
- flake_occurrence: 6 unresolved; samples: `<unknown>`, `<unknown>`, `<unknown>`, `<unknown>`, `<unknown>`

</details>

### What this ranking is not

- No test is retired by this census.
- Retiring any test needs a planted-defect check first (INV-10
  `guards-exercise-behaviour`): show the test goes red on the defect it guards.
- A slow test that never failed is an offline-lane candidate, not a deletion.
- "Never failed" is weak evidence: it means "never seen failing in the windows
  above", and a guard whose invariant nobody broke also never fails.

## Part 2: pinning and duplication across the whole test tree

### Python pinning and duplication

Every tracked `.py` file with a `tests` directory in its path is a test file;
its package is its first path segment. Rows are in package-name order and are
not ranked.

- **test fns**: module-level `test*` functions and `test*` methods of `Test*`
  classes (nested `Test*` classes included), in `test_*.py` / `*_test.py` files.
- **lines / prose lines**: the merge-lane ratchet's `file_size_measures`
  (prose = docstring or comment lines).
- **private reads**: the ratchet's `private_reads` (single-underscore attribute
  accesses, reads and writes, except on bare `self`/`cls`), over every test file.
- **private-patch tests**: test fns holding a patch call (the ratchet's patch
  shapes) whose string target starts with a first-party name and has a later
  `_private` segment, or whose object-form attribute is `_private` and whose
  receiver is not bound to a non-first-party import. A `Test*` class decorator
  counts for each of its test methods. Receivers are not type-resolved, so the
  object form over-counts.
- **patch sites outside tests**: the same calls in fixtures, helpers and module
  scope, counted here rather than attributed to the tests that use them.
- **distinct targets**: string targets plus object-form targets whose receiver
  is a first-party module import.
- **prose asserts**: `assert` statements comparing (`in`, `not in`, `==`, `!=`)
  a string of 3+ characters that occurs inside a module-level string constant
  of 6+ words in a first-party module the file imports. These are candidates
  for hand sampling, not confirmed prose pins.
- **exact / structural duplicates**: test fns with identical AST bodies
  (name and decorators ignored); structural also erases identifiers, attribute
  names, constant values, keyword and parameter names. Grouped within a
  package; `groups/redundant` where redundant = members beyond the first.

| package | test files | lines | prose lines | test fns | private reads | private-patch tests | patch sites outside tests | distinct targets | prose asserts | tests with prose asserts | exact dups | structural dups | largest structural group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cockpit | 24 | 13297 | 2984 | 494 | 83 | 1 | 0 | 0 | 4 | 4 | 0/0 | 34/43 | 8: `cockpit/tests/test_spawn_bar.py::TestDefaultSkipPerms::test_env_true_resolves_to_true` |
| dashboard | 107 | 88501 | 22803 | 2733 | 426 | 99 | 15 | 27 | 69 | 47 | 3/3 | 112/257 | 23: `dashboard/tests/test_tab_orchestrators.py::TestOrchTabRuntimeColumns::test_lane_column_header_present` |
| escalation | 52 | 49764 | 12787 | 1691 | 98 | 35 | 5 | 6 | 5 | 5 | 0/0 | 97/172 | 17: `escalation/tests/test_classify.py::TestClassifyResolverTierHuman::test_interactive_is_human` |
| fused-memory | 450 | 492607 | 118249 | 19696 | 5376 | 498 | 26 | 41 | 1942 | 1398 | 19/24 | 1073/2074 | 39: `fused-memory/tests/test_config_schema.py::TestServerConfigTransport::test_invalid_transport_raises_validation_error` |
| orchestrator | 656 | 632866 | 172441 | 19824 | 12859 | 931 | 95 | 103 | 505 | 406 | 24/28 | 1026/1910 | 28: `orchestrator/tests/test_verify_classify.py::TestOpaqueReproducesLegacyGenericLadder::test_compile_error_rustc_code` |
| sampler | 4 | 3301 | 754 | 137 | 3 | 1 | 0 | 0 | 2 | 2 | 0/0 | 7/8 | 3: `sampler/tests/test_load_metrics.py::TestParsePressureFile::test_both_lines_extracted` |
| scripts | 117 | 106129 | 26740 | 4299 | 220 | 8 | 0 | 2 | 41 | 36 | 13/20 | 201/331 | 14: `scripts/tests/test_check_transcript_check_liveness.py::test_script_is_executable` |
| shared | 136 | 101281 | 28050 | 4642 | 1796 | 105 | 7 | 14 | 43 | 36 | 5/7 | 306/563 | 16: `shared/tests/test_silent_fallthrough_gate.py::TestSignatureANegatives::test_error_bound_not_discarded` |
| tests | 95 | 82268 | 32687 | 2059 | 287 | 99 | 23 | 1 | 21 | 16 | 2/3 | 115/220 | 15: `tests/scripts/test_orchestrator_watchdog.py::test_staleness_grace_secs_env_override` |
| total | 1641 | 1570014 | 417495 | 55575 | 21148 | 1777 | 171 | 193 | 2632 | 1950 | 66/85 | 2971/5578 | 39: `fused-memory/tests/test_config_schema.py::TestServerConfigTransport::test_invalid_transport_raises_validation_error` |

Unreadable files (0; the census is complete):

- none

## Reading against the 2026-09-10 studies

This section is written by hand; everything above it is generated and unedited.
The study is `plans/verify-speed-study-df-2026-09-10.md` with its
`A-baseline.md` and `E-selection.md`, untracked files that exist only in the
main checkout. The study passages this section quotes, the two study scripts it
re-ran and their output are copied verbatim into the tracked
`plans/test-census-2026-10-04-dark-factory-study-sources.md`, so the comparison
can still be read and re-run once the untracked files are gone.

### Junit is archived green and red

The task brief says junit is archived only for red gates, about 97% red. That
no longer holds: `orchestrator/src/orchestrator/verify.py::_archive_junit_report`
archives every leg's report, green or red. Of the 2,805 archived runs
(2026-09-21 to 2026-10-04), 57 (2.0%) hold a failing testcase.

The archive was re-read about 20 minutes after the census, against a slightly
larger store, by counting the records `suite_census_evidence.pytest_evidence`
yields. It held 21,254,609 testcase records, and 26,706 of them (0.13%) were
unresolved. The unresolved records name test modules that are no longer in the
tree. The largest group, 12,465 records, is
`tests.test_code_quality_guidance_parity`, which task 5738 retired.

### Per-test cost

| package | study p50 per test | census p50 of medians | study top 1% | census top 1% |
| --- | --- | --- | --- | --- |
| orchestrator | 1.443 s | 0.619 s | 5.1% | 16.2% |
| shared | 0.001 s | 0.002 s | 69.0% | 59.2% |
| fused-memory | 0.008 s | 0.007 s | 54.5% | 49.6% |

The two columns measure different things:

- **Unit.** The study's unit is a junit node id, so each parametrised case is
  its own test. The census's unit is a test function, with its parametrised
  cases summed within a run.
- **Runs.** The study read one merge-gate report per module (A-baseline §5).
  The census takes each test's median over every archived and live run, of any
  role, green or red. That is about 312 runs per orchestrator test, each at
  whatever host load and xdist width it ran under.
- **Share.** The study's top-1% share is the share of one run's summed time
  held by its slowest 1% of tests. The census's is the share of the summed
  per-test medians held by the 1% of tests with the largest median.

### flake_occurrence

| | study | store on 2026-10-04 |
| --- | --- | --- |
| rows | 299 | 478, of which 6 have test_id `<unknown>` |
| distinct tests | 67 | 141 distinct test_ids |
| `fails_in_isolation` | 295 | 416 |
| `passes_in_isolation` | 0 | 39 |
| `unconfirmable` | — | 23 |

The store spans 2026-08-30 to 2026-10-04. The census counts every row as a
failure, whatever its isolation verdict.

### Limits of the floor and the ranking

The plan (step-6) specified that test ids resolve at file granularity. A junit
classname or node id resolves when its file is tracked, whether or not the
function still exists. An ambiguous flake_occurrence id, one whose path two
modules both track, is credited to every candidate, which is the conservative
direction for the never-failed claim. Measured against the tree above:

- **Floor.** 10 of the 165 floor members name a function that their file does
  not define. Five are cross-package credits of a test that exists in another
  package (four in dashboard, one in fused-memory). The other five are defined
  nowhere in the tree: they were renamed or removed after they failed. That
  leaves 155 floor members that exist in this tree.
- **Ranking.** In this report's top 100, 7 rows name a function that their
  file no longer defines. A re-run of the census about an hour later, against
  a slightly larger store (56,564 ranked), found 1,236 such rows (2.2%), 58 of
  them in its top 1,000. Each describes a test that was renamed or removed
  inside a file that still exists.
- **Thin samples.** Four rows of the top 100 rest on fewer than 3 runs: ranks
  59, 71 and 74 on one run each, and rank 92 on two. The median runs per row in
  the top 100 is 313. Each run's time depends on the host load and xdist width
  it ran under, so these four medians are weak evidence, and each may have
  pushed a better-evidenced test out of the top 100. Censuses run after this
  one print this list under the ranking
  (`scripts/suite_census_outcomes.py::THIN_SAMPLE_RUNS`).

### Part 2 against the study's own scripts

The study's scripts survive in its `scratch-E/` directory. Re-running them
unmodified on this tree separates growth from definition. Only two changes
were made to the copies: ROOT was re-pointed at the tree, and the JSON output
was written to /tmp.

| measure | study, 2026-09-10 | study script, this tree | census |
| --- | --- | --- | --- |
| test fns | 44,700 | 55,576 | 55,575 |
| private-patch tests | 696 (orchestrator 602) | 643 (orchestrator 537) | 1,777 (orchestrator 931) |
| exact duplicates: redundant fns (groups) | 66 (57) | 90 (70) | 85 (66) |
| structural duplicates: redundant fns (groups) | 5,020 (2,589), 11.2% | 5,861 (2,974), 10.5% | 5,578 (2,971), 10.0% |

- **Private patches.** Measured with the study's own instrument, the count fell
  from 696 to 643, so the census's higher count comes from its definition. The
  study's regex sees only `patch(` or `patch.object(` followed directly by a
  dotted path that holds `._name`. The census differs in three ways:
  - it sees every patch shape the ratchet recognises, including
    `patch.object(mod, '_name')` and the string and object forms of
    `setattr(...)`;
  - it counts a class decorator once for each method of the class;
  - it does not type-resolve object-form receivers, which is the over-count the
    Part 2 definitions state.
- **Duplicates.**
  - The study groups across the whole tree; the census groups within a package.
  - The study takes every `test*` def in `test_*.py` files. The census takes
    module-level `test*` functions and `test*` methods of `Test*` classes, in
    tracked `test_*.py` and `*_test.py` files.

Commands used for the middle column:

```
mkdir -p /tmp/5414-study
cp /home/leo/src/dark-factory/plans/verify-speed-study-df-2026-09-10/scratch-E/static_dupes.py /home/leo/src/dark-factory/plans/verify-speed-study-df-2026-09-10/scratch-E/static_rest.py /tmp/5414-study/
sed -i "s#^ROOT=Path('/home/leo/src/dark-factory')#ROOT=Path('/home/leo/src/dark-factory/.worktrees/5414')#" /tmp/5414-study/static_dupes.py /tmp/5414-study/static_rest.py
python3 /tmp/5414-study/static_dupes.py /tmp/5414-study/dupes.json
python3 /tmp/5414-study/static_rest.py
```

Only section 6 of `static_rest.py` is used here. The script then stops in
section 7 on a `sleep(...)` literal that it cannot parse.

`scratch-E/` is untracked. If it is gone, take the two scripts from
`plans/test-census-2026-10-04-dark-factory-study-sources.md` §2, which also
records their sha256. Their output on this tree is in §3 of the same file.

### No retirement

No test is retired here. Retiring any listed test needs a planted-defect check
(INV-10) in a follow-up. A slow test that guards something real is an
offline-lane candidate, not a deletion.
