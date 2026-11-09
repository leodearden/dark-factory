"""Per-unit verify-result cache: a recorded green replays only when every input of
its unit is byte-identical (plans/verify-result-cache-prd.md, § Registry and
§ Contract section). Consumers import the submodules directly; this package
re-exports nothing, so importing the dependency-free ``modes`` leaf costs nothing.
"""
