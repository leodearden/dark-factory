"""Prompt-artifact loader (PRD plans/tier1-prompt-optimization-prd.md §7 T1 / D-4).

Resolves ``(prompt_id, executor_model, harness_version)`` to an on-disk
heuristics-block artifact, composing the in-code CONTRACT with the artifact's
heuristics at load time, and falling back to the in-code constant when nothing
is pinned. ``executor_model`` is the model resolved *at invocation* (forward
compatible with adaptive-model-routing) so artifacts are per-model; the key
also carries ``harness_version`` so artifacts are per-harness. Every pinned
artifact carries an 8-field provenance sidecar. Unpinning is the sole rollback
lever — there is no separate revert path.

Reachable by both the orchestrator and fused-memory (both declare
``dark-factory-shared`` as a workspace dependency). Like
``shared.task_metadata``, this module is accessed as a submodule
(``shared.prompt_artifact.X``) and is deliberately **not** re-exported from
``shared/__init__.py`` — this keeps ``shared/tests/test_public_api.py``'s
strict ``__all__`` union assertion untouched.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ['ArtifactProvenance']


class ArtifactProvenance(BaseModel):
    """The 8-field provenance sidecar recorded for every pinned prompt artifact.

    All 8 fields are REQUIRED (no defaults) — construction fails unless the
    caller supplies the full sidecar, enforcing "the full provenance sidecar
    recorded" as a schema-level guarantee rather than a convention.
    ``extra='allow'`` matches this repo's forward-compat convention (see
    ``shared.task_metadata``) so a future additional field round-trips without
    a schema bump. ``held_out_TEST_score`` is a machine-contract field name —
    kept verbatim.
    """

    model_config = ConfigDict(extra='allow')

    optimizer_model: str
    corpus_hash: str
    split_seed: int
    held_out_TEST_score: float
    accept_delta: float
    git_sha: str
    date: str
    harness_version: str
