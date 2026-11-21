"""Rule D ``timing-dependence`` debt baseline: data only, read by _timing_dependence.py."""

# SHRINK-ONLY.  DO NOT ADD ENTRIES, AND DO NOT RAISE A NUMBER.
#
# path -> the number of UNCOVERED timing-dependence VIOLATIONS that file carries.
# A covered site (under a valid @pytest.mark.timing or a virtual clock) is not
# counted.  The count is an EXACT budget, enforced by the lint itself: above it
# the overrun is reported, below it one slack violation names the lowered number.
#
# The baseline is opt-OUT: a NEW file with a timing-dependent test must fail the
# gate, which an opt-in list could never do.
#
# INV-12 owner: task 9513, declared in
# fused-memory/tests/test_fused_memory_governed_lists.py::TIMING_DEPENDENCE_BASELINE.
#
# Provenance: not yet measured; task 8254 populates it when Rule D is wired.
TIMING_DEPENDENCE_DEBT: dict[str, int] = {}
