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

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    'ArtifactProvenance',
    'compose_prompt',
    'PromptSpec',
    'ResolvedPrompt',
    'PromptArtifactStore',
]

# Fixed separator between the in-code CONTRACT and the (baseline or pinned)
# heuristics block. Part of compose_prompt's contract: changing this value
# changes every composed prompt's text, so it lives in exactly one place.
_SEPARATOR = '\n\n---\n\n'


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


def compose_prompt(contract: str, heuristics: str) -> str:
    """The single composition chokepoint for every resolved prompt.

    Emits *contract* verbatim, then a fixed separator, then *heuristics*.
    Both the pinned-artifact path and the in-code-fallback path route through
    this one function, so the contract prefix of the returned text is always
    byte-identical to *contract* regardless of what *heuristics* contains
    (D-3: the CONTRACT is un-editable by construction, not by a fallible
    post-edit validator).
    """
    return f'{contract}{_SEPARATOR}{heuristics}'


@dataclass(frozen=True)
class PromptSpec:
    """The in-code definition of a prompt: its id, CONTRACT, and baseline heuristics.

    ``in_code_constant`` is the text every caller treats as "the prompt when
    nothing is pinned" — it is produced by the exact same :func:`compose_prompt`
    rule used for a pinned artifact, so the baseline and a pinned candidate are
    always compared like-for-like (no composition drift).
    """

    prompt_id: str
    contract: str
    baseline_heuristics: str

    @property
    def in_code_constant(self) -> str:
        return compose_prompt(self.contract, self.baseline_heuristics)


@dataclass(frozen=True)
class ResolvedPrompt:
    """The result of :meth:`PromptArtifactStore.resolve`.

    ``source`` names which of the two paths produced ``text``: ``'artifact'``
    when a valid pinned artifact was found (``provenance`` is then non-None),
    ``'in_code'`` for the fallback (``provenance`` is then None).
    """

    text: str
    provenance: ArtifactProvenance | None
    source: Literal['artifact', 'in_code']


_HEURISTICS_FILENAME = 'heuristics.txt'
_PROVENANCE_FILENAME = 'provenance.json'


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via temp-in-dir + os.replace (safe_io.py's pattern).

    A reader never observes a half-written file: either the old contents (if
    any) or the complete new contents, never a truncated partial write.
    """
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class PromptArtifactStore:
    """Resolves a :class:`PromptSpec` to a pinned artifact or the in-code fallback.

    Backed by an on-disk tree rooted at *root*; see :meth:`_key_dir` for the
    per-key layout.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _key_dir(self, prompt_id: str, executor_model: str, harness_version: str) -> Path:
        # NOTE: naive direct segment join for now — hardened to a
        # filesystem-safe, collision-free, traversal-safe encoding once
        # TestKeyIsolationAndPathSafety is in place.
        return self.root / prompt_id / executor_model / harness_version

    def resolve(
        self, spec: PromptSpec, executor_model: str, harness_version: str
    ) -> ResolvedPrompt:
        key_dir = self._key_dir(spec.prompt_id, executor_model, harness_version)
        heuristics_path = key_dir / _HEURISTICS_FILENAME
        provenance_path = key_dir / _PROVENANCE_FILENAME
        if heuristics_path.exists() and provenance_path.exists():
            heuristics = heuristics_path.read_text(encoding='utf-8')
            provenance = ArtifactProvenance.model_validate_json(
                provenance_path.read_text(encoding='utf-8')
            )
            return ResolvedPrompt(
                compose_prompt(spec.contract, heuristics), provenance, 'artifact'
            )
        return ResolvedPrompt(spec.in_code_constant, None, 'in_code')

    def pin(
        self,
        prompt_id: str,
        executor_model: str,
        harness_version: str,
        *,
        heuristics: str,
        provenance: ArtifactProvenance,
    ) -> None:
        """Pin *heuristics* + *provenance* under the (prompt_id, executor_model,
        harness_version) key.

        *provenance.harness_version* must equal *harness_version* — a
        key/provenance mismatch is a caller bug, so this raises rather than
        silently persisting an incoherent artifact.

        Writes atomically, ``provenance.json`` LAST: a crash between the two
        writes leaves at most a heuristics-only directory, which resolve()
        treats as not-pinned (fail-safe).
        """
        if provenance.harness_version != harness_version:
            raise ValueError(
                f'pin: provenance.harness_version={provenance.harness_version!r} does not '
                f'match the harness_version key {harness_version!r}'
            )
        key_dir = self._key_dir(prompt_id, executor_model, harness_version)
        key_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(key_dir / _HEURISTICS_FILENAME, heuristics)
        _atomic_write_text(key_dir / _PROVENANCE_FILENAME, provenance.model_dump_json())

    def read_provenance(
        self, prompt_id: str, executor_model: str, harness_version: str
    ) -> ArtifactProvenance | None:
        key_dir = self._key_dir(prompt_id, executor_model, harness_version)
        provenance_path = key_dir / _PROVENANCE_FILENAME
        if not provenance_path.exists():
            return None
        return ArtifactProvenance.model_validate_json(
            provenance_path.read_text(encoding='utf-8')
        )

    def unpin(self, prompt_id: str, executor_model: str, harness_version: str) -> bool:
        """Remove the pin for this key — the sole rollback lever (no separate revert path).

        Returns ``True`` when a pin was removed, ``False`` when there was
        nothing pinned (idempotent). Either way, the next :meth:`resolve` for
        this key returns the in-code constant.
        """
        key_dir = self._key_dir(prompt_id, executor_model, harness_version)
        if key_dir.exists():
            shutil.rmtree(key_dir)
            return True
        return False
