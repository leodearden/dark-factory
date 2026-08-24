# Tagger-debris census

Every task record still carrying `metadata.files_tagged_at` — the stamp the
retired module tagger left behind — across all six project corpora, classified
on three axes for the repair pipeline.

Consumers: dark_factory 3113 P4a, dark_factory 3427.

**`module-tagger-debris-census.json` is the complete record.** This markdown is
its readable twin and caps the record table at 60 rows; the JSON
never truncates. Neither file carries a generation timestamp, deliberately, so
re-running the command below and diffing is a meaningful reproducibility check.

## Classification vocabulary

- **status_class** — `terminal` (status in {done, cancelled}) vs `non_terminal`.
- **reconciliation** — `plan_reconciled` if a genuine plan-derived assertion
  (a `set_to_plan` or `phase_skipped` event) postdates the stamp, meaning the
  tagger's guess was superseded; `lock_reconciled` if only a `lock_acquired`
  event does, and that lock named at least one module the record's own
  `metadata.files` cannot explain; `never_reconciled` if neither does, meaning
  the guess is still this record's live scope. Read the caveat below before
  treating `lock_reconciled` as repaired.
- **wipe_signature** — `post_wipe_overwrite` if an authoritative scope event predates
  the stamp (the tagger stamped over it); `no_prior_scope` otherwise.
- **merge_signature** — the audit's own `merge_finalized` verdict
  (`audit_wiped_metadata_files.classify_wipe_signature`), carried as correlating
  evidence in the vocabulary both consumers already speak.

### Why `lock_reconciled` is a weaker signal than `plan_reconciled`

A `lock_acquired` event's module set is **derived from `metadata.files`** by the
scheduler — `Scheduler._get_modules` computes it as
`derive_modules(metadata['files'], depth)` — so a lock is
**not an independent scope derivation**. For a record still carrying the tagger's
guess, the lock is an ECHO of that guess and proves nothing about it.

This census therefore discounts any post-stamp lock whose modules are fully
explained by the record's own `metadata.files`, and reports the rest as
`lock_reconciled` rather than folding them into `plan_reconciled`.

A record counted `lock_reconciled` **may still be carrying the tagger's guess**
as its live scope: all that is known is that some lock named a module the guess
cannot account for. Only `plan_reconciled` reflects a genuine plan-derived assertion.
A consumer must **decide for itself** whether to treat `lock_reconciled` records
as repaired — the class is reported separately precisely so that choice is
available rather than made here.

## Per-project counts

| project | total tasks | stamped | terminal | non-terminal | plan reconciled | lock reconciled | never reconciled | post-wipe overwrite | event log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| autopilot_video | 651 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | read |
| dark_factory | 4630 | 309 | 161 | 148 | 110 | 55 | 144 | 5 | read |
| know_live | 600 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | read |
| pump_web_ui | 19 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | read |
| reify | 6473 | 219 | 87 | 132 | 58 | 14 | 147 | 16 | read |
| solar_challenge_platform | 168 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | read |

## Three-axis cells

Every cell is emitted even at zero: a missing count must never be readable as
a zero. The strict live-victim cell for the repair pipeline is
`non_terminal|never_reconciled|post_wipe_overwrite` — live work whose
scope was overwritten and never superseded.

`non_terminal|lock_reconciled|post_wipe_overwrite` is the SECOND cell a
repair must consider: those records may still be carrying the guess (see the
caveat above). Whether they belong in the population is the consumer's call —
which is why the two cells are counted apart rather than merged.

| project | cell | count |
| --- | --- | ---: |
| autopilot_video | `non_terminal|lock_reconciled|no_prior_scope` | 0 |
| autopilot_video | `non_terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| autopilot_video | `non_terminal|never_reconciled|no_prior_scope` | 0 |
| autopilot_video | `non_terminal|never_reconciled|post_wipe_overwrite` | 0 |
| autopilot_video | `non_terminal|plan_reconciled|no_prior_scope` | 0 |
| autopilot_video | `non_terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| autopilot_video | `terminal|lock_reconciled|no_prior_scope` | 0 |
| autopilot_video | `terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| autopilot_video | `terminal|never_reconciled|no_prior_scope` | 0 |
| autopilot_video | `terminal|never_reconciled|post_wipe_overwrite` | 0 |
| autopilot_video | `terminal|plan_reconciled|no_prior_scope` | 1 |
| autopilot_video | `terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| dark_factory | `non_terminal|lock_reconciled|no_prior_scope` | 29 |
| dark_factory | `non_terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| dark_factory | `non_terminal|never_reconciled|no_prior_scope` | 107 |
| dark_factory | `non_terminal|never_reconciled|post_wipe_overwrite` | 0 |
| dark_factory | `non_terminal|plan_reconciled|no_prior_scope` | 12 |
| dark_factory | `non_terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| dark_factory | `terminal|lock_reconciled|no_prior_scope` | 25 |
| dark_factory | `terminal|lock_reconciled|post_wipe_overwrite` | 1 |
| dark_factory | `terminal|never_reconciled|no_prior_scope` | 36 |
| dark_factory | `terminal|never_reconciled|post_wipe_overwrite` | 1 |
| dark_factory | `terminal|plan_reconciled|no_prior_scope` | 95 |
| dark_factory | `terminal|plan_reconciled|post_wipe_overwrite` | 3 |
| know_live | `non_terminal|lock_reconciled|no_prior_scope` | 0 |
| know_live | `non_terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| know_live | `non_terminal|never_reconciled|no_prior_scope` | 0 |
| know_live | `non_terminal|never_reconciled|post_wipe_overwrite` | 0 |
| know_live | `non_terminal|plan_reconciled|no_prior_scope` | 0 |
| know_live | `non_terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| know_live | `terminal|lock_reconciled|no_prior_scope` | 0 |
| know_live | `terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| know_live | `terminal|never_reconciled|no_prior_scope` | 0 |
| know_live | `terminal|never_reconciled|post_wipe_overwrite` | 0 |
| know_live | `terminal|plan_reconciled|no_prior_scope` | 1 |
| know_live | `terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| pump_web_ui | `non_terminal|lock_reconciled|no_prior_scope` | 0 |
| pump_web_ui | `non_terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| pump_web_ui | `non_terminal|never_reconciled|no_prior_scope` | 0 |
| pump_web_ui | `non_terminal|never_reconciled|post_wipe_overwrite` | 0 |
| pump_web_ui | `non_terminal|plan_reconciled|no_prior_scope` | 0 |
| pump_web_ui | `non_terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| pump_web_ui | `terminal|lock_reconciled|no_prior_scope` | 0 |
| pump_web_ui | `terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| pump_web_ui | `terminal|never_reconciled|no_prior_scope` | 0 |
| pump_web_ui | `terminal|never_reconciled|post_wipe_overwrite` | 0 |
| pump_web_ui | `terminal|plan_reconciled|no_prior_scope` | 1 |
| pump_web_ui | `terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| reify | `non_terminal|lock_reconciled|no_prior_scope` | 10 |
| reify | `non_terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| reify | `non_terminal|never_reconciled|no_prior_scope` | 107 |
| reify | `non_terminal|never_reconciled|post_wipe_overwrite` | 3 |
| reify | `non_terminal|plan_reconciled|no_prior_scope` | 11 |
| reify | `non_terminal|plan_reconciled|post_wipe_overwrite` | 1 |
| reify | `terminal|lock_reconciled|no_prior_scope` | 3 |
| reify | `terminal|lock_reconciled|post_wipe_overwrite` | 1 |
| reify | `terminal|never_reconciled|no_prior_scope` | 31 |
| reify | `terminal|never_reconciled|post_wipe_overwrite` | 6 |
| reify | `terminal|plan_reconciled|no_prior_scope` | 41 |
| reify | `terminal|plan_reconciled|post_wipe_overwrite` | 5 |
| solar_challenge_platform | `non_terminal|lock_reconciled|no_prior_scope` | 0 |
| solar_challenge_platform | `non_terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| solar_challenge_platform | `non_terminal|never_reconciled|no_prior_scope` | 0 |
| solar_challenge_platform | `non_terminal|never_reconciled|post_wipe_overwrite` | 0 |
| solar_challenge_platform | `non_terminal|plan_reconciled|no_prior_scope` | 0 |
| solar_challenge_platform | `non_terminal|plan_reconciled|post_wipe_overwrite` | 0 |
| solar_challenge_platform | `terminal|lock_reconciled|no_prior_scope` | 0 |
| solar_challenge_platform | `terminal|lock_reconciled|post_wipe_overwrite` | 0 |
| solar_challenge_platform | `terminal|never_reconciled|no_prior_scope` | 0 |
| solar_challenge_platform | `terminal|never_reconciled|post_wipe_overwrite` | 0 |
| solar_challenge_platform | `terminal|plan_reconciled|no_prior_scope` | 1 |
| solar_challenge_platform | `terminal|plan_reconciled|post_wipe_overwrite` | 0 |

## Coverage

- projects swept: 6
- tasks examined: 12541
- stamped records: 532
- event log read for every swept project (no coverage shortfall)

## Records (showing 60 of 532)

| project | task | status | status_class | reconciliation | wipe_signature | merge_signature | files_tagged_at |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| autopilot_video | 649 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T19:23:37.951549+00:00 |
| dark_factory | 2763 | done | terminal | plan_reconciled | post_wipe_overwrite | confirmed_null_sha_done_path | 2026-07-20T09:41:11.633249+00:00 |
| dark_factory | 2772 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T01:16:40.810169+00:00 |
| dark_factory | 2773 | done | terminal | lock_reconciled | post_wipe_overwrite | clean_merge_sha | 2026-07-19T01:16:40.810169+00:00 |
| dark_factory | 2774 | done | terminal | lock_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-19T01:16:40.810169+00:00 |
| dark_factory | 2775 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T01:16:40.810169+00:00 |
| dark_factory | 2777 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T01:16:40.810169+00:00 |
| dark_factory | 2802 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T09:20:41.418895+00:00 |
| dark_factory | 2803 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T09:20:41.418895+00:00 |
| dark_factory | 2804 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T09:20:41.418895+00:00 |
| dark_factory | 2805 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T09:20:41.418895+00:00 |
| dark_factory | 2806 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T09:20:41.418895+00:00 |
| dark_factory | 2815 | done | terminal | lock_reconciled | no_prior_scope | clean_merge_sha | 2026-07-19T17:20:52.038659+00:00 |
| dark_factory | 2816 | cancelled | terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-19T17:20:52.038659+00:00 |
| dark_factory | 2818 | done | terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-19T17:20:52.038659+00:00 |
| dark_factory | 2835 | done | terminal | never_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-20T01:23:03.844584+00:00 |
| dark_factory | 2836 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-20T01:23:03.844584+00:00 |
| dark_factory | 2837 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-20T01:23:03.844584+00:00 |
| dark_factory | 2850 | cancelled | terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-20T09:41:11.633249+00:00 |
| dark_factory | 2857 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-20T17:44:34.919588+00:00 |
| dark_factory | 2858 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-20T17:44:34.919588+00:00 |
| dark_factory | 2859 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-20T17:44:34.919588+00:00 |
| dark_factory | 2860 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-20T17:44:34.919588+00:00 |
| dark_factory | 2863 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-20T17:44:34.919588+00:00 |
| dark_factory | 2895 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2896 | merge-deferred | non_terminal | plan_reconciled | no_prior_scope | no_successful_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2897 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2898 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2899 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2900 | pending | non_terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2903 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2904 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2905 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2906 | done | terminal | never_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2907 | done | terminal | never_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2908 | done | terminal | never_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2909 | done | terminal | never_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2910 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2911 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T11:23:11.727792+00:00 |
| dark_factory | 2913 | done | terminal | never_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2915 | pending | non_terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2925 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2927 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2928 | done | terminal | never_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2929 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2930 | in-progress | non_terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2943 | pending | non_terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-22T19:26:12.243534+00:00 |
| dark_factory | 2961 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-22T19:28:31.979856+00:00 |
| dark_factory | 2979 | pending | non_terminal | lock_reconciled | no_prior_scope | no_merge_event | 2026-07-28T21:30:44.781355+00:00 |
| dark_factory | 2985 | pending | non_terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-23T14:02:35.849772+00:00 |
| dark_factory | 2987 | pending | non_terminal | never_reconciled | no_prior_scope | no_merge_event | 2026-07-23T14:02:35.849772+00:00 |
| dark_factory | 2988 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-23T14:02:35.849772+00:00 |
| dark_factory | 2998 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-26T15:05:04.866363+00:00 |
| dark_factory | 3006 | cancelled | terminal | never_reconciled | post_wipe_overwrite | no_merge_event | 2026-07-24T07:26:22.495511+00:00 |
| dark_factory | 3033 | done | terminal | plan_reconciled | no_prior_scope | contradicted_real_merge_sha | 2026-07-24T19:26:16.909747+00:00 |
| dark_factory | 3034 | done | terminal | plan_reconciled | no_prior_scope | clean_merge_sha | 2026-07-24T19:26:16.909747+00:00 |
| dark_factory | 3037 | done | terminal | lock_reconciled | no_prior_scope | clean_merge_sha | 2026-07-24T19:26:16.909747+00:00 |
| dark_factory | 3041 | done | terminal | lock_reconciled | no_prior_scope | clean_merge_sha | 2026-07-25T04:48:24.247157+00:00 |
| dark_factory | 3042 | done | terminal | never_reconciled | no_prior_scope | clean_merge_sha | 2026-07-25T04:48:24.247157+00:00 |
| dark_factory | 3043 | done | terminal | plan_reconciled | no_prior_scope | no_successful_merge_sha | 2026-07-26T15:05:04.866363+00:00 |

## Regenerate

```
python scripts/census_tagger_debris.py --project-root /home/leo/src/autopilot-video --project-root /home/leo/src/dark-factory --project-root /home/leo/src/know-live --project-root /home/leo/src/pump-web-ui --project-root /home/leo/src/reify --project-root /home/leo/src/solar-challenge-platform
```

Read-only: every corpus connection is a `mode=ro` SQLite URI.
