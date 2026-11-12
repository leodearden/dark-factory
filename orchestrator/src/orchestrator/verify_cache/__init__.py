"""Per-unit verify-result cache: a recorded green replays only when every input its
unit's key records is byte-identical (plans/verify-result-cache-prd.md, § Registry
and § Contract section). Those inputs are derived from the plan, the tree and its
workspace graph, not traced from what a run read, so a test reading a path outside
them can replay stale; the audit lane is the backstop for that. Consumers import
the submodules directly; this package re-exports nothing, so importing the
dependency-free ``modes`` leaf costs nothing.
"""
