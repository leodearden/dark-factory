# Memory-consolidation cluster ledger (relocated from task 3524 `metadata.memory_hints`)
Relocated 2026-08-20 by the `/curate-fused-memories` batch sitting (`curator-sitting-2026-08-19`).

**Why this file exists.** `get_task(3524)` failed with *result exceeds maximum allowed tokens* (~59k payload). The dominant component was `metadata.memory_hints` at 24,537 chars — larger than the description. Per `CLAUDE.md`, `memory_hints` is a reconciliation-internal channel whose only consumer is recon Stage 1, so those chars were unreadable by every consumer actually blocked on 3524.

**What it turned out to be.** `memory_hints.queries` was not a list of queries. Only entry [0] was one. Entries [1]-[25] are a *cluster ledger* — UUID enumerations, do-not-fold warnings, and re-enumeration corrections accumulated by Stage 1 runs. Several clusters here have no gate task of their own, so this content existed nowhere else. It was therefore RELOCATED here verbatim, not deleted, before `memory_hints` was trimmed.

Entries are reproduced **verbatim and unedited**, in their original order.

- Source: `task 3524.metadata.memory_hints.queries` (26 entries, 24508 chars)
- `metadata.memory_hints.entities`: `[]` (empty)

---

## [0] (133 chars)

```text
resolution for: Human gate: batch-rule the 14 open duplicate-consolidation gate tasks in one sitting (per 2026-08-02 policy decision)
```

## [1] (327 chars)

```text
pytest addopts/marker-deselection duplicate procedural_knowledge cluster (4 entries per Stage1 finding 74fb8596-3d22-4edc-b408-a619ae57fef6): 0fe0a7dd-16b9-4c82-bdf2-b652a006ad1e, 0fe0a7dd-16b9-4c82-bdf2-b652a006ad1e, 0fe0a7dd-16b9-4c82-bdf2-b652a006ad1e, 7b0fb970-3db0-4a5b-8f75-d0b9fcd791aa — fold into this batch-ruling pass
```

## [2] (314 chars)

```text
pytest -m single-argparse-value last-wins duplicate procedural_knowledge cluster, 3rd batch (Stage1 finding 35f99b43-abcf-4411-bcdf-6cf93e01130c): 3eab82cd-db9a-4d15-879c-e72cfbac514f, 80573858-a554-4238-be32-a82438bb9f62, c5d7e0cd-6b81-44a7-9284-7f6f729a3f9a — fold into this batch-ruling pass alongside Cluster B
```

## [3] (386 chars)

```text
host-path-writes-blocked-outside-worktree duplicate procedural_knowledge cluster (3 entries per Stage1 finding a2a025e4-0f31-457f-870d-d68c48588693): de3eebb0-655e-4ebc-84f0-4aee1ae061c4 (symptom), 61c820f4-0ed5-467f-83e9-dbe1d379914c (Landlock root cause), 663d8bc2-4153-4141-856f-698eaf10086e (systemd-run --user workaround) — fold into this batch-ruling pass alongside prior clusters
```

## [4] (127 chars)

```text
resolution for: Human gate: consolidate 9-entry duplicate spawn-wrapper-survives-kill procedural_knowledge/observations cluster
```

## [5] (195 chars)

```text
resolution for: Human gate: consolidate 5-entry duplicate metadata_mode='replace' procedural_knowledge cluster (contradiction: unusable on done/merged tasks) into a trigger-tagged canonical entry
```

## [6] (131 chars)

```text
resolution for: Human gate: consolidate 3-entry duplicate guard-tests-grep-code-not-prose (task 3554) preferences_and_norms cluster
```

## [7] (174 chars)

```text
resolution for: Human gate: consolidate 6-entry duplicate OrchestratorConfig-MagicMock/spec_set test-gotcha procedural_knowledge cluster into a trigger-tagged canonical entry
```

## [8] (179 chars)

```text
resolution for: Human gate: consolidate 10-entry duplicate silent-fallthrough-gate procedural_knowledge/observations cluster into a trigger-tagged canonical entry (subcase-tagged)
```

## [9] (656 chars)

```text
host-path-writes-blocked-outside-worktree duplicate procedural_knowledge cluster grew a 4th member this cycle (per Stage1 finding 545babeb-e9d1-46bc-a6cd-55bf927b19f3): a14e9317-4d73-4cc3-90b8-d32ec765fa1b (most detailed/most recent, names task 3642's blocked step as a live use case, folds in de3eebb0's systemctl-exit-code caution) — fold into this batch-ruling pass alongside de3eebb0/61c820f4/663d8bc2; a14e9317 is the natural survivor if judged a safe mechanical merge. Also note: 663d8bc2 is mis-filed under category=observations_and_summaries despite being tracked as part of this procedural_knowledge cluster — correct when the cluster is resolved.
```

## [10] (1196 chars)

```text
symbol/key-not-file:line citation-convention duplicate preferences_and_norms cluster (8 entries, Stage1 findings 979701a5-92cc-45df-b7e0-a2ea2d0b86f5 + 0b9f03d8-17cf-4eca-b35f-6a5a135f4a49, persisted_from_run 5726ecc8-dda7-4c4a-a233-4fb0ed2025e9): 37b6af94-7dcb-4720-b36a-684f4c2f6185 (general orchestrator internals example), a4646dee-5664-4de7-8c21-6d92b1974cd2 (dashboard FORMAT-COUPLING markers example), 1fdd7fe2-ba2c-46c2-a02e-9ffd7b0690ce (warm-lane scripts example), 56338b7e-e7a6-423b-b380-1007955156fb (fused-memory test cross-refs example), b942d348-76b5-48c7-96c1-140bc2de7469 (config-key citation form example), c0fe8cb3-dacb-44ed-8d71-6635c84e447e (caveat: wrong symbol worse than stale line pin), 899d7b8d-d212-4288-bc04-5031812fb2cf (harness.py plan-revalidation anchors on method/closure names not absolute line numbers, task 1881, created 2026-06-23), 6f8bd228-c70d-4b81-8784-a4dfbc27f6f5 (cite cross-file evidence in docstrings/PRDs by SYMBOL name not file:line — task 3508 added ~89 lines and thereby invalidated its own five citations, created 2026-08-07T15:02:13) — content-losing merge, reserved for human curator; fold into this batch-ruling pass alongside prior clusters.
```

## [11] (537 chars)

```text
OrchestratorConfig defaults.yaml layers over pydantic Field defaults duplicate procedural_knowledge cluster (3 entries per Stage1 finding 7a3b1696-6c21-4810-8b4f-338e17a9e6e7): 3303832a-0294-49a7-a1b2-a9da0b24aa23 (2026-07-16), 3f03dd68-2d20-4aa8-8255-3ac3fb22c729 (2026-07-31, lock_depth=4-vs-Field-default=2 + task-3350 postmortem example), 571dc59a-4b8d-42f2-92c9-b52fa3012ba9 (2026-08-07, reuses lock_depth example, adds merge_verify_cold_command_timeout_secs=7200 example) — fold into this batch-ruling pass alongside prior clusters
```

## [12] (513 chars)

```text
npx-pyright-EACCES/npm-cache-workaround cluster tracking MOVED to task 3417 (sole canonical owner) as of 2026-08-07, run c7b5ae8f-7d3f-4503-b720-00692711fcce, per Stage 1 finding cacd3508-33fb-4224-9625-1b37e3f50239 — this task and 3417 had independently diverged (3417 tracked mem0 604ed99f-7c9c-4fb6-9835-df45ea3a880f absent here; this task tracked mem0 3fecb3d6-7364-4d15-a0ef-d3a89db2c0e9 absent from 3417). Do not re-track this cluster's membership here; consult task 3417 for the current authoritative list.
```

## [13] (1355 chars)

```text
commit_planning YAML sidecar comment-loss observations cluster (7 entries, Stage1 finding c7bb4be2-0f60-4433-a0de-4642c09b8977): abe43309-c2c1-427e-9c51-28742fac0d7d, 05975c50-5805-4aa8-b627-799583ce84fa, cb11f26a-3e06-4cd0-81af-c049072d6ac8, 68035bb0-644f-4bf9-8b6e-16b25d1232b5, 8a69ea9a-309b-476c-8fa2-052527c83c09, 03ac65d4-25ac-4a3b-b41f-19bbd5e3feeb, e85723e3-e155-4e2d-96d4-7123266aeec7 — content-losing merge, reserved for human curator; fold into this batch-ruling pass alongside prior clusters. SIBLING TASK 3845 DISPOSITION — RULED 2026-08-08 (run 89f93fa6-b89b-4a7c-96ea-456262edb987, per Stage1 finding 8f406474-fd76-4c4c-bad1-b3a79de5a3f8): KEEP. 3845 is retained as batch-sitting member 30 (it is already wired dependencies=[3524], so it is ruled at THIS sitting, which is what the 2026-08-02 DECIDE-FIRST policy actually asks for; and this cluster is a content-losing merge, squarely inside that policy's own not-a-safe-mechanical-merge carve-out). The earlier CANCEL intent (runs b99c904b / a72982e0 / 0038824d) is SUPERSEDED — do NOT re-attempt set_task_status(3845,'cancelled') in a future cycle, and do NOT remove "3845" from related_tasks or decrement gate_task_count from 30. Full grounds in this task's details field. The human operator at the sitting is of course still free to rule the cluster and dispose of 3845 as they see fit.
```

## [14] (5425 chars)

```text
dashboard redux frontend JS TEST SUBSTRATE duplicate cluster — CORRECTED ENUMERATION as of run 148221d5-4fc9-48ec-bba8-fb12c651fc75 (Stage 1 finding 61d5ff08-db96-4f60-a45c-61a4f7234aa1). 22 members (11 procedural_knowledge + 7 preferences_and_norms + 4 corrective); 17 carry in-place corrective pointers. This SUPERSEDES the prior enumeration (Stage1 finding 88b84fe6-4516-4d2e-81e6-0b697e192f89, run f6183e96-cb1b-495f-8088-b5a9146ac8c7), which was wrong in two ways: its header said '12 entries' but it listed 13 ids, and it enumerated only the procedural_knowledge half, missing the entire preferences_and_norms half.
MISLEADING, procedural_knowledge (11) — [C] = carries an in-place corrective pointer (update_memory; point id and created_at preserved; original text kept verbatim under a dated marker; metadata stamped superseded_by=3182de30-379a-4fd2-a7e8-ef25acf989f8): 7c5cf8b1-44cc-4b34-bd54-82c990d9992d[C], 55bff9dc-8ab2-4727-8bca-54d6221f966d[C], 8fca3563-db38-4c2b-8366-021e120b3e18[C], db99fe6d-2794-4484-97fb-86c95467b8bd[C], cfb5ab0c-3aec-4b81-acb4-114b02e65573[C, run f6183e96], 2eec0cdf-7692-41c1-8587-94acee544691[C], 82e69b7c-c987-49fe-bb10-2ae79d4cc1b8[C], 09668427-88e7-4bb5-99b0-82ee3b1c9647[C, run f6183e96], 643ec874-a7d0-4f8d-b0e3-5b0c0eb80d42[C], f0225474-9b7a-48ff-a634-36ef1824f2a4[C], 2ca4047b-48dd-4ffb-aba7-e897f8db4ca5[C — DISAMBIGUATION note, not a correction: true as written, .jsx-scoped, but ranks #2 on the general query].
MISLEADING, preferences_and_norms (7 — the half the prior enumeration missed entirely): 36d4f62a-f5b7-481c-8abd-80a9915c42dc [deliberately UNCORRECTED: scoped to 'JSX changes', never ranked], 0c86b7ae-2418-4f4a-a825-373250b90a02[C], 0f0250f8-8c61-4f13-ae5e-97488dd94280[C], 7bbca93d-aaea-4a69-8da1-f069924db624[C], 51053862-cc72-4246-aa66-96123ddad345[C], d11fb35f-af67-466e-a5ce-2292b86d0bbc[C], 68d9a9d7-ff2c-4b0b-aeb1-e37feb02562c [deliberately UNCORRECTED: its 'no DOM runner' claim is accurate].
CORRECTIVE (4): 9774f5ed-619b-446f-8421-635da0ffea3f[C — STALENESS note: correct side, but says the suite is only graph_layout.js when it is now ~10 files], 36a3f77e-c82b-4b9c-9bc5-f6e6fd631796, 381354de-7151-4b52-a46e-cdd79f8ee16a, 3182de30-379a-4fd2-a7e8-ef25acf989f8 (CANONICAL SURVIVOR — newest, strictly most complete, and the only member that names the misreading explicitly).
COST IS NOT HYPOTHETICAL: 3182de30 records that task 3543's FIRST PLAN asserted 'there is NO JS test runner', was half wrong, and had to be replanned.
RETRIEVAL STATUS: on a deliberately different probe from the prior cycle's ('can I write a JS unit test for the dashboard frontend, is there a test runner'), canonical 3182de30 now ranks #1 at 0.699 and every other top-6 hit carries a marker — the ranking inversion the prior hint described is gone. Do NOT read this as a general guarantee: the prior hint's claim that Stage 1 had corrected 'the two top-ranked offenders cfb5ab0c and 09668427, so the top hits now carry the correction' has been DELETED from this entry because it named only 2 of the now-17 pointered members and its retrieval claim was query-specific.
GRAFT these distinct details before retiring the others: f0225474's offline esbuild JSX parse-check recipe; 82e69b7c/09668427's node-vm _DRIVER sandbox idiom; 36a3f77e's dual-export (module.exports + window.DF_X) convention.
DO NOT GRAFT: 51053862 prescribes verifying renders via Playwright — Playwright is not used in this repo and is out of scope.
DROPPED: the prefix 'd555aea7' carried by the prior enumeration is NOT a member — it surfaced in none of three semantic probes and its full UUID never resolved. It is most likely mis-transcribed from the sibling cache-buster cluster hint below, where the same prefix also appears. Do not re-add it unresolved.
RESOLVED — NO LIVE COMMAND IS NEEDED AT THE SITTING (settled 2026-08-08 by run 5e55d304-cdbc-40da-9d27-498bd20e5c51; re-verified run 89f93fa6-b89b-4a7c-96ea-456262edb987 per Stage1 finding a48cc1be-0121-46f3-aaf5-1e93ed016d4c). This entry previously carried an 'UNRESOLVED CONTRADICTION ... run the command at the 3524 sitting' instruction; that instruction was FALSE and has been struck. c5faa8c1-9dc5-4494-a29c-c97e4a201f18, canonical 3182de30-379a-4fd2-a7e8-ef25acf989f8 and a859c495-9017-4037-918e-f92427ba385b are JOINTLY SATISFIABLE — they describe three DIFFERENT `node --test` ARGUMENT FORMS, not a disagreement: (1) BARE DIRECTORY `node --test tests/js/` fails MODULE_NOT_FOUND on this repo's Node (c5faa8c1); (2) an EXPLICIT FILE PATH works (c5faa8c1's own recommendation); (3) a NODE-EXPANDED GLOB works from the repo/worktree ROOT and prints '# tests 0' with exit 0 on zero match (a859c495) — the silent-zero-match hazard that the production pytest shim dashboard/tests/test_graph_layout_js.py guards against by parsing the TAP '# tests N' line (canonical 3182de30). c5faa8c1 was disambiguated IN PLACE by run 5e55d304 (point id preserved, created_at 2026-07-14T00:27:34Z preserved, metadata stamped disambiguated_by_run / related_canonical_entry=3182de30 / topic=node-test-invocation-forms), so the memory side needs no further cleanup. CURATOR ACTION: do not run an experiment — pick the form actually being invoked and word the survivor so it names all three forms plus the zero-match-glob hazard.
Content-losing merge for the remaining members — reserved for human curator; fold into this batch-ruling pass alongside prior clusters.
```

## [15] (1862 chars)

```text
dashboard index.html `?v=` CACHE-BUSTER convention duplicate procedural_knowledge cluster (11 entries, Stage1 finding 371e1b11-b7fa-44c9-9d85-685a6ab5f5e9, run f6183e96-cb1b-495f-8088-b5a9146ac8c7): d555aea7 (2026-05-29, floor '>=10'), 4bfcff8e (2026-05-14), 2eec0cdf (2026-06-18), d9e853c0 (2026-07-14, '16 occurrences'), 13cdce63 (2026-07-14, 'version is 28 / floor is 28'), 146d2072 (2026-07-14, SRI carve-out), 762e7bca (2026-07-16), 210292df (2026-07-31, 'floors >=10/19/30/33'), 4c940dff (2026-07-31, '36 across 22 tags'), 67982b6e (2026-08-02, '25 occurrences as of v=41'), e65439d9 (2026-08-08, 'share one version >= 42'). EVERY member pins a version/floor/tag-count that was true only on its write date, so a reader retrieving any single member gets a STALE ABSOLUTE NUMBER presented as current. Two members already recognise the hazard and DISAGREE on the remedy: 210292df says the version is 'a moving target across concurrent dashboard tasks — re-measure at implementation time rather than trusting a number written during planning', while 13cdce63 and 67982b6e simply assert their own snapshot value. RECOMMENDED SURVIVOR SHAPE: keep ONLY the invariant rules (bump uniformly across every /static/redux/*?v= tag INCLUDING the <head> styles.css link, which also matches the guard regex; EXCLUDE the unpkg CDN SRI tags; all guards are FLOORS not exact pins, so any uniform value above the highest floor passes) plus 210292df's explicit RE-MEASURE-at-implementation-time instruction with the grep one-liner. Deliberately DROP every hard-coded version/floor/tag-count from the survivor — those are the only parts that go stale and are recoverable from the file in one command. 4c940dff and 210292df carry the two traps worth preserving verbatim. Content-losing merge, reserved for human curator; fold into this batch-ruling pass alongside prior clusters.
```

## [16] (805 chars)

```text
pytest INTEGRATION-MARKER DESELECTION duplicate procedural_knowledge cluster — RE-ENUMERATED 2026-08-08 (run 9c8a3750-1db9-4e48-b276-13124121d1d9, Stage 1 finding d6378748-5350-4718-8714-e49257289e26). Core lesson: fused-memory's pytest addopts carries -m 'not integration', and the marker is applied PER-TEST, not module-level. 4 members: 6f31a539, 55d165eb, fb41b626, a4628af5. This SUPERSEDES the older 'pytest addopts/marker-deselection' hint entry above, whose id list (0fe0a7dd repeated three times + 7b0fb970) is a citation-tombstone artifact — see metadata.x_memory_citation_tombstones. STILL ACTIVELY GROWING: 6f31a539 was written during the 2026-08-08 15:30Z recon run itself. Content-losing merge (each member differs only in which task verified it); reserved for human curator at this sitting.
```

## [17] (545 chars)

```text
ruff SIM300 YODA-CONDITION on ALL_CAPS constants duplicate procedural_knowledge cluster — NEW, tracked 2026-08-08 (run 9c8a3750-1db9-4e48-b276-13124121d1d9, Stage 1 finding d6378748-5350-4718-8714-e49257289e26). 4 members: 05faadf2, 803626e0, dafc2aab, f963052a. Core lesson restated near-verbatim in each; entries differ only in which task/instance verified it. Content-losing merge, reserved for human curator; fold into this batch-ruling pass alongside prior clusters. Per the 2026-08-02 DECIDE-FIRST policy no standalone gate task was filed.
```

## [18] (807 chars)

```text
os-sandbox BLOCKS HOST-FILE WRITES OUTSIDE THE WORKTREE (e.g. ~/.config/systemd/user) — cluster GROWTH UPDATE 2026-08-08 (run 9c8a3750-1db9-4e48-b276-13124121d1d9, Stage 1 finding d6378748-5350-4718-8714-e49257289e26). Stage 1's window this cycle enumerated 5 members: 44900aa0, a14e9317, 317da35f, 61c820f4, 392911e4 — two of them written this cycle. This EXTENDS, and should be unioned with, the two older 'host-path-writes-blocked-outside-worktree' hint entries above (de3eebb0 symptom / 61c820f4 Landlock root cause / 663d8bc2 systemd-run --user workaround / a14e9317 most-detailed). Take the UNION as the membership list at the sitting: 44900aa0, 317da35f, 392911e4 are the three not previously tracked. a14e9317 remains the natural survivor candidate. Content-losing merge, reserved for human curator.
```

## [19] (499 chars)

```text
subprocess.run(timeout) KILLS ONLY THE DIRECT CHILD — an orphaned grandchild keeps holding the pipe. NEW duplicate procedural_knowledge cluster, tracked 2026-08-08 (run 9c8a3750-1db9-4e48-b276-13124121d1d9, Stage 1 finding d6378748-5350-4718-8714-e49257289e26). 3 members: 666c413e, 33be8677, d725542c. Content-losing merge (each cites a distinct verifying task); reserved for human curator; fold into this batch-ruling pass. Per the 2026-08-02 DECIDE-FIRST policy no standalone gate task was filed.
```

## [20] (449 chars)

```text
escalation-watcher-auto _WATCHER_ALLOWED_TOOLS HARD-CODED LIST duplicate procedural_knowledge cluster — NEW, tracked 2026-08-08 (run 9c8a3750-1db9-4e48-b276-13124121d1d9, Stage 1 finding d6378748-5350-4718-8714-e49257289e26). 3 members: 2e126a5b, acd66f86, 46c7e57a. Content-losing merge, reserved for human curator; fold into this batch-ruling pass alongside prior clusters. Per the 2026-08-02 DECIDE-FIRST policy no standalone gate task was filed.
```

## [21] (419 chars)

```text
SWEEP CAVEAT for the five clusters added 2026-08-08 (run 9c8a3750-1db9-4e48-b276-13124121d1d9): Stage 1 explicitly stated these were observed in that cycle's own related-context window and were NOT an exhaustive sweep. Member ids are recorded at 8-char prefix length as Stage 1 reported them — resolve each to a full UUID before acting on it at the sitting, and expect the true member counts to be floors, not ceilings.
```

## [22] (2461 chars)

```text
pytest SINGLE-INIFILE ROOT-vs-MEMBER DESELECTION duplicate procedural_knowledge cluster — RE-TRACKED 2026-08-08 (run 2cf7c0b3-c326-4650-b800-e86c56799d70, Stage 1 finding e87ae304-e1a2-4f57-a289-d04934acf4c5). RETRACTION FIRST: the 'pytest INTEGRATION-MARKER DESELECTION' hint entry (added run 9c8a3750-1db9-4e48-b276-13124121d1d9) declares that it SUPERSEDES the older 'pytest addopts/marker-deselection' hint entry. That supersession claim is WRONG and is hereby RETRACTED — the two are DIFFERENT clusters, the older entry is NOT fully superseded, and treating it as retired dropped tracking of this distinct root-vs-member sub-cluster. CORE LESSON OF THIS CLUSTER: pytest resolves exactly ONE [tool.pytest.ini_options] (the rootdir inifile) and never merges addopts/markers across workspace members, so a member package's -m 'not X' is silently ignored on a root-bound run. 2 MEMBERS (full UUIDs, both resolved this cycle): 0fe0a7dd-16b9-4c82-bdf2-b652a006ad1e (procedural_knowledge, 2026-08-01, agent claude-task-3444-implementer — CARRIES A TASK-3444-STILL-PENDING CAVEAT plus the 'two -m flags are argparse last-wins, use ONE combined expression' gotcha; BOTH must survive any merge) and 7b0fb970-3db0-4a5b-8f75-d0b9fcd791aa (procedural_knowledge, 2026-07-11, itself already consolidated_from 0b52135a + b2a34f05 — carries the cockpit/smoke X11+tmux live-desktop specifics, the 'an autouse skip fixture is NOT sufficient' point, and the task 2300/2446 lineage). CONTENT-LOSING merge; Stage 1 deliberately did NOT merge it inline; reserved for human curator at this sitting. NOT A MEMBER: 662e1383-3c0d-4db0-9d71-6430fa51ce93 is a THIRD, genuinely distinct topic (unit tests colocated in a module whose CLASS carries @pytest.mark.integration are deselected by shared/'s addopts; task 3483) — do NOT fold it into this cluster or into the integration-marker cluster. DETERMINISTIC RE-DERIVATION (preferred over this prose): Stage 1 tagged both members in place (update_memory metadata_patch; point ids and created_at preserved) with duplicate_cluster='pytest_single_inifile_root_vs_member_deselection' and duplicate_cluster_gate_task='3524'. Verified this cycle via count_memories_by_metadata — exactly 2 records carry each tag. Enumerate membership at the sitting with get_memories_by_metadata on that tag. Does NOT change gate_task_count (still 30) — memory_hints-only tracked cluster, no standalone gate task filed, per the 2026-08-02 DECIDE-FIRST policy.
```

## [23] (1644 chars)

```text
scripts/tests/ vs tests/scripts/ near-homograph test directories duplicate procedural_knowledge/observations cluster — NEW, tracked 2026-08-09 (run 71799bd7-0f73-47f2-bc25-371dc45fcdef, Stage 1 finding fd224aca-7fb0-43b0-8107-06121adb3f54, Mem0 flag marker d069ad2d-4380-489f-a8d2-dc37ef30d9ce). 9 members total, all restating that dark-factory has TWO distinct scripts/-dir test trees (scripts/tests/ and tests/scripts/) that are easy to confuse and cover DISTINCT modules. 3 STALE pre-task-3384/3460-fix members already annotated IN PLACE this cycle (update_memory, point ids preserved, dated [SUPERSEDED] marker prepended, original text kept verbatim, metadata stamped duplicate_cluster=scripts_tests_vs_tests_scripts_homograph/stale_pre_fix_state=true): fd664135-b11c-4682-b1d2-18e9863f72ce, 4b35a45e-8a9d-46cb-b638-4e72ce6ce264, 3492dd9d-fd49-4225-9af4-ba7274cbcf66 (each asserted 'scripts/tests/ is un-gated', now false — task 3384/3460 fixed both trees to be gated together via `pytest tests/scripts/ scripts/tests/`). 6 CURRENT-state members, not annotated, describe the fixed state: d069ad2d-4380-489f-a8d2-dc37ef30d9ce, 12746543-08bc-451e-8060-92ef1df6de2e, 8e5d298e-da0e-47e6-b2ef-ee259679959b, d7da24af-c3fc-40f7-95d6-39ad8c68a90c, 78553ed0-ecb6-407c-8a02-caa34719b7ce, bcf69b92-9bdd-4776-b204-3786271a4ddb. NOT a safe mechanical merge even among the 6 current members (each carries distinct task-citation content); content-losing merge, reserved for human curator at this sitting. Does NOT change gate_task_count (still 30) — memory_hints-only tracked cluster, no standalone gate task filed, per the 2026-08-02 DECIDE-FIRST policy.
```

## [24] (1553 chars)

```text
architect plan.json `files` list must name only files a step actually creates/modifies, never defensively-declared regression-test files — duplicate preferences_and_norms/procedural_knowledge cluster, NEW, tracked 2026-08-09 (run 71799bd7-0f73-47f2-bc25-371dc45fcdef, Stage 1 finding 15a70ebc-46f5-45b9-bd29-afe34b02e190, Mem0 flag marker 81398fcb-b7f2-42a1-a6a3-694fe56e4eef). 10 members, each pairing the shared rule with a distinct worked task example or additive nuance (single-seam defusal vs per-file declaration, runtime-generated artifacts, read-only-import sources, verify-only entries): 81398fcb-b7f2-42a1-a6a3-694fe56e4eef (task 3586, newest), b65a637b-ebb4-4625-b151-f330c288f3cb (task 3302), a2ee35a8-8839-48d5-abb3-d3af84b5b8d6 (task 3072), 587877b4-8d99-41d8-bf67-0d72d9788134 (task 3003), c6554355-667b-4f42-8516-778af963bea2 (payload-compatible exception example), 81398fcb-b7f2-42a1-a6a3-694fe56e4eef (task 3779, verify-only entries nuance), 0a0a5d14-b64c-446c-bef4-85a9b67a4618 (ROOT_CONFTEST_DIFF read-only-import nuance), d2554c1f-6688-45e7-8b1d-6538847be396 (task 2582 runtime-generated-artifact nuance), 4f81c8eb-507e-4f3f-ae1e-7db8da5ebfdf (read-only-import-helper nuance), f0993d85-b291-4a41-a4be-95e332a25665 (task 3424, two legitimately-droppable non-dead-scope cases). Content-losing merge, reserved for human curator judgment on canonical survivor + which nuances to graft. Does NOT change gate_task_count (still 30) — memory_hints-only tracked cluster, no standalone gate task filed, per the 2026-08-02 DECIDE-FIRST policy.
```

## [25] (1440 chars)

```text
orchestrator's automatic 'chore: save WIP before inter-iteration rebase' safety commit can already contain a FULLY COMPLETE implementation for a plan step still marked pending in plan.json — duplicate procedural_knowledge cluster, NEW, tracked 2026-08-09 (run 71799bd7-0f73-47f2-bc25-371dc45fcdef, Stage 1 finding a1e7faff-51ed-4a18-a5c8-d69c7f42bc63, Mem0 flag marker 87d8f11c-b477-4b0e-954a-315386032a1b). 5 live standalone members restating the same core lesson (diff/read current file state and run the step's tests BEFORE re-implementing) with a distinct worked task example each: 87d8f11c-b477-4b0e-954a-315386032a1b (task 3915), 7e953a44-7126-44c2-b752-746b45bb7323 (task 2512), 5a725b24-0489-41e4-96f0-211ace1c2569 (multi-step-satisfied-by-one-commit variant), 98f8cf2d-bb72-4d9a-b5ea-08e22e9f1e2e (already itself consolidated_from 4 earlier entries — tasks 2564/2535/2674 — inline-consolidation precedent exists for this exact cluster), d7210db4-d59c-4be9-bb44-72ab1af230b0 (task 3412). NOT off-topic despite surface similarity to bb99a22e-4978-42c2-a28c-af013c55ad89, which documents a DIFFERENT gotcha (WIP commit capturing a corrupted mid-edit refactor) and is excluded from this cluster. Content-losing merge (each citation distinct/non-redundant); reserved for human curator. Does NOT change gate_task_count (still 30) — memory_hints-only tracked cluster, no standalone gate task filed, per the 2026-08-02 DECIDE-FIRST policy.
```
