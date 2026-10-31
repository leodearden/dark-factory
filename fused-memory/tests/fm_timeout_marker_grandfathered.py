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
    # test_local_memory_models_eval_corpus.py -- 1 site at 120s
    ('test_local_memory_models_eval_corpus.py', 'test_live_dark_factory_population_smoke'),
})
