"""The single home for the dispatched-agent briefing's memory queries (D9, INV-5).

This module owns WHAT the briefing asks memory: the scope a query is phrased
from, the area terms derived from it, and (below) the query-spec table itself.
It deliberately owns none of the PRESENTATION — how a result is rendered into
a prompt stays in ``orchestrator/src/orchestrator/agents/briefing.py``, which
also owns the fused-memory payload shape and the cross-project leak filter.
The two dimensions change on different evidence: the query set on retrieval
measurements, the markdown on prompt-budget measurements.

Two importers, and the reason this is a module rather than four string
literals inline:

* ``orchestrator.agents.briefing`` renders these specs into the ``# Context``
  block of every dispatched agent's prompt.
* PRD-γ's E1 registry pinning test imports this module and asserts its
  fixture phrasings equal the rendered templates, so the memory-eval registry
  is keyed by the queries actually fired rather than by hand-copied
  paraphrases that drift the moment a template is reworded.

See ``docs/prds/memory-briefing-and-fusion.md`` (lane β, D1-D3 and D9).
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

AREA_TERM_LIMIT = 8
"""Most area terms a query is phrased from.

A task's ``metadata.files`` can name a dozen paths across four packages,
whose components expand to ~25 distinct words — an unfocused bag that
embeds to nothing in particular. The measured-good probes were 5-6 terms
long, so the first few paths (the ones an author lists first, which are the
ones the task is actually about) decide the phrasing and the tail is cut.
"""

_PATH_NOISE: frozenset[str] = frozenset({'src'})
"""Path components describing LAYOUT rather than subject matter.

Only ``src`` needs naming: this repo's ``<pkg>/src/<pkg>/`` double-nesting
collapses on its own once ``src`` is gone, because the repeated package name
is deduplicated like any other repeated term.
"""

_WORD = re.compile(r'[A-Za-z0-9]+')


def _is_area_word(word: str) -> bool:
    """Reject words that carry no retrievable meaning on their own.

    A bare number is the measured failure mode this whole rescope exists to
    fix: embedding a task id matched OTHER task numbers at 0/5 relevance, and
    a digit lifted out of a filename behaves the same way.
    """
    return len(word) > 1 and not word.isdigit()


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if _is_area_word(w)]


def _path_words(path: str) -> list[str]:
    """Prose words of one repo-relative path, extension and layout dropped."""
    components = [c for c in path.split('/') if c]
    if components:
        components[-1] = components[-1].rsplit('.', 1)[0]
    return [
        word
        for component in components
        if component.lower() not in _PATH_NOISE
        for word in _words(component)
    ]


def _first_seen(words: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(words))


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a wire-read ``files`` value to a tuple of paths.

    A bare string is rejected rather than accepted as a one-element sequence:
    it is iterable, so a permissive ``tuple(...)`` would shred it into single
    characters and phrase the query from the letter ``s``.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


@dataclass(frozen=True)
class BriefingScope:
    """What one dispatch is about, as the query templates need it.

    Frozen because a prompt builder passes the same scope to every spec it
    renders; a mutable one would let a renderer quietly re-point a later
    query at a different task.
    """

    task_id: str | None = None
    title: str = ''
    files: tuple[str, ...] = ()

    @classmethod
    def from_task(cls, task: Mapping[str, Any] | None) -> BriefingScope:
        """Read a queue-time task record (``id`` / ``title`` / ``metadata.files``)."""
        task = task or {}
        raw_id = task.get('id')
        metadata = task.get('metadata')
        return cls(
            task_id=str(raw_id) if raw_id is not None and str(raw_id) else None,
            title=task.get('title') or '',
            files=_string_tuple(metadata.get('files') if isinstance(metadata, Mapping) else None),
        )

    @classmethod
    def from_plan(cls, plan: Mapping[str, Any] | None) -> BriefingScope:
        """Read a frozen plan, whose key spellings differ from a task's.

        ``plan.json`` carries the same three facts at the top level under
        ``task_id`` / ``title`` / ``files``, so the post-planning roles
        (implementer, amender, debugger, judge) scope from the plan they were
        handed rather than re-reading the task record they do not hold.
        """
        plan = plan or {}
        raw_id = plan.get('task_id')
        return cls(
            task_id=str(raw_id) if raw_id is not None and str(raw_id) else None,
            title=plan.get('title') or '',
            files=_string_tuple(plan.get('files')),
        )


def derive_area_terms(scope: BriefingScope) -> tuple[str, ...]:
    """Prose terms naming the area *scope* is about, most specific source first.

    The D2/D3 ladder: the declared file paths, else the task title, else
    nothing at all — the repo-generic last resort, which the query table
    answers with the generic conventions spec (which needs no terms).

    Deriving from ``metadata.files`` at render time, rather than from a
    hand-maintained ``metadata.area`` key, keeps one home for "what area is
    this task about" (INV-9) and cannot drift from the files themselves. The
    files-derived phrasing was re-probed live against the corpus before this
    landed (5/5 relevant conventions hits, cosine 0.564-0.584) per
    ``plans/metadata-modules-retirement-prd.md`` decision 4.
    """
    terms = _first_seen(word for path in scope.files for word in _path_words(path))
    if not terms:
        terms = _first_seen(_words(scope.title))
    return terms[:AREA_TERM_LIMIT]
