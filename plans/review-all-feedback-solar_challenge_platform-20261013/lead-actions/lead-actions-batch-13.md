# Lead task-store actions, batch 13 (from S16 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-15.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled). Both are currently deferred.
- 803 -> "Superseded by review-all S16-alpha (task 1231: typed minute Band in models/tariff_value.py with a derived wrap flag, Band.half_hours identical to solar_challenge TariffPeriod.matches_time on every half hour) and S16-beta (task 1234: billing and the optimiser adapter both read the typed terms; _parse_hhmm/_tou_rate deleted). Ruling R16; accepted into program leaves; lead cancellation 2026-10-15."
- 811 -> "Superseded by review-all S16-alpha (task 1231: TariffRule.BANDS_OVERLAP / BANDS_LEAVE_GAP, overlap refused rather than first-wins) and S16-epsilon (task 1241: the rule enforced at write on POST /tariffs and /tariffs/{id}/versions as a 422, plus row-local DB CHECKs). Ruling R16; accepted into program leaves; lead cancellation 2026-10-15."

## B. Coordination notes (details append with append=True and NO metadata in that call; no edges)
- 1207: "COORDINATION NOTE 2026-10-15 (review-all S16, lead ruling Q2): the TouTariff type (models/tariff_value.py, S16-alpha task 1231) is the one home of 'is TOU' (R16/C9; fk-c7780f124df0 is S16's). If this task lands before S16-beta (task 1234), it lands planner.is_tou(tariff: TariffConfig) with the rate-count rule and 1234 retypes it to is_tou(tariff: TariffTerms) -> bool = isinstance(tariff, TouTariff). If 1234 lands first, adopt that signature or drop is_tou for the type match; never reintroduce the rate-count rule. This task's delivered_check one-tou-predicate (_is_tou_tariff absent) stays compatible: 1234 deletes _is_tou_tariff too."
- 521: "COORDINATION NOTE 2026-10-15 (review-all S16): after S16-epsilon (task 1241), TariffStore.spans returns TariffSpan.version_id per segment, so the tariff version a segment was priced at can be recorded beside this task's own-use/VAT rates. No edge."

## C. No commit_planning. Never touch 1065. Never edit files. Never write memory.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "notes_appended": [...], "errors": [...]}
