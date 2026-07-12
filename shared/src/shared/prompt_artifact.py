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

A pinned artifact is only ever trusted when both its heuristics block and a
schema-valid provenance sidecar are present; anything else (nothing pinned,
an orphan heuristics file, or a corrupt/incomplete provenance sidecar)
fails safe to the in-code constant. :func:`default_artifacts_root` gives
every consumer (the T6 optimization loop, T2/T3 call sites, T8 tooling) one
agreed on-disk root, so they never each invent a divergent location for the
same on-disk state.
"""

from __future__ import annotations

import os
import shutil
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from shared.safe_io import load_json_or_warn

__all__ = [
    'ArtifactProvenance',
    'compose_prompt',
    'PromptSpec',
    'ResolvedPrompt',
    'PromptArtifactStore',
    'default_artifacts_root',
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


def _encode_segment(segment: str) -> str:
    """Encode one on-disk key segment: filesystem-safe, injective, traversal-safe.

    ``urllib.parse.quote(segment, safe='')`` alone is not sufficient: per
    RFC 3986, ``.`` (like ``-_~`` and alphanumerics) is *always* left
    unescaped by :func:`urllib.parse.quote` — the ``safe`` argument only
    controls which *additional* characters are spared, it cannot force an
    always-safe character to be escaped. So a segment equal to ``'..'``
    would quote to ``'..'`` unchanged and still resolve as a parent-directory
    reference.

    To close that gap, every literal ``.`` left over from ``quote()`` is
    replaced with the 3-character escape ``%2E`` (the same escape a naive
    percent-encoder would have produced for it). This is still injective:
    ``quote()`` never escapes a ``.`` itself (it is always-safe), so the
    literal substring ``%2E`` never occurs natively in a ``quote()`` output —
    every ``%`` in a ``quote()`` output starts a genuine, non-``%2E`` escape
    triple. That means the position of every ``.``-derived ``%2E`` in the
    final string is unambiguous, so distinct inputs can never collide on the
    same encoded segment. A segment of ``'.'`` or ``'..'`` therefore encodes
    to ``'%2E'`` / ``'%2E%2E'`` — plain directory names that cannot be
    interpreted as "this directory" or "parent directory" — so no segment
    can ever escape the store root.
    """
    return urllib.parse.quote(segment, safe='').replace('.', '%2E')


def _load_valid_provenance(path: Path) -> ArtifactProvenance | None:
    """Load *path* as a schema-valid :class:`ArtifactProvenance`, fail-safe.

    Returns ``None`` — never raises — when *path* is absent, when it is
    present but corrupt/non-JSON, or when it parses as JSON but does not
    satisfy :class:`ArtifactProvenance`'s required fields (e.g. a half-written
    or hand-edited sidecar). Callers treat ``None`` identically to "nothing
    pinned": an unverifiable artifact must never be surfaced as a pinned one.
    """
    parsed, _ok = load_json_or_warn(path, default=None)
    if parsed is None:
        return None
    try:
        return ArtifactProvenance.model_validate(parsed)
    except ValidationError:
        return None


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
        return self.root.joinpath(
            *(_encode_segment(segment) for segment in (prompt_id, executor_model, harness_version))
        )

    def resolve(
        self, spec: PromptSpec, executor_model: str, harness_version: str
    ) -> ResolvedPrompt:
        key_dir = self._key_dir(spec.prompt_id, executor_model, harness_version)
        heuristics_path = key_dir / _HEURISTICS_FILENAME
        provenance_path = key_dir / _PROVENANCE_FILENAME
        if heuristics_path.exists():
            provenance = _load_valid_provenance(provenance_path)
            if provenance is not None:
                heuristics = heuristics_path.read_text(encoding='utf-8')
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
        return _load_valid_provenance(key_dir / _PROVENANCE_FILENAME)

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


def default_artifacts_root() -> Path:
    """The one agreed on-disk root for prompt artifacts.

    Returns ``Path(os.environ['DARK_FACTORY_PROMPT_ARTIFACTS'])`` when that
    env var is set. Otherwise walks up from this file's location looking for
    the monorepo root — a directory containing ``orchestrator/``,
    ``fused-memory/``, and ``shared/`` — and returns
    ``<monorepo_root>/data/prompt_artifacts``. Falls back to
    ``Path.cwd() / 'data' / 'prompt_artifacts'`` when no such marker directory
    is found (e.g. this file was relocated outside the monorepo layout).

    Every T2/T3/T8 consumer should call this instead of hard-coding a path,
    so they never drift onto a different root for the same on-disk state.
    """
    env_override = os.environ.get('DARK_FACTORY_PROMPT_ARTIFACTS')
    if env_override:
        return Path(env_override)

    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / 'orchestrator').is_dir()
            and (candidate / 'fused-memory').is_dir()
            and (candidate / 'shared').is_dir()
        ):
            return candidate / 'data' / 'prompt_artifacts'

    return Path.cwd() / 'data' / 'prompt_artifacts'
