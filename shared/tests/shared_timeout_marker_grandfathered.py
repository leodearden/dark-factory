"""The shared/tests timeout-marker inversion guard's grandfather allowlist.

Its own module so the INV-12 declaration imports it by a unique name; see
tests/scripts/test_pytest_workspace_collection.py::test_no_governed_list_declaration_imports_a_bare_name_two_dirs_define.
"""

from __future__ import annotations

#: In-band sites that predate this guard: 1, MEASURED on 2026-10-07 at main
#: 1c6a52dbfd.  Entries may only ever be REMOVED; migration is follow-up
#: tkt_0RVDQ479918XDV0C80NXAVT8HX.  Keyed per SITE on the path RELATIVE to
#: tests/.  The site's own comment still reasons against a 60s ini default that
#: is now 540, which is why 180 reads to its author as a loosening.
INVERSION_BAND_GRANDFATHERED: frozenset[tuple[str, str]] = frozenset({
    # test_root_config_integration_deselection.py -- 1 site at 180s
    ('test_root_config_integration_deselection.py', 'TestRootConfigIntegrationDeselection'),
})
