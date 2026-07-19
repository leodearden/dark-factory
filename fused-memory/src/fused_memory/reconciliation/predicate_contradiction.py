"""Reusable helper to file a ``task_kind='deterministic'`` predicate task for a
reconciliation-stage contradiction that only LIVE command execution can settle
(task 2779).

## Why this exists

When a recon stage hits a contradiction it cannot resolve from memory alone —
e.g. "memory says test X passes on main, but a finding claims it fails", or
"this pattern was supposedly removed but might still be in the tree" — the only
honest way to settle it is to *run something*: a specific pytest test, a
``git grep``, a ``git log --grep``. Historically a recon stage would instead
write an ad-hoc Mem0 investigation note or suppression record, which silently
ages across cycles without ever being decided (motivating precedent: task 2643,
resolved manually over two investigation cycles).

This module gives recon stages a durable, code-level FILING CONVENTION: build a
one-off deterministic task whose ``before_done.kind='predicate'`` script runs
the check, and let the existing predicate exit-code contract decide it —
exit 0 → task ``done`` (``done_provenance.kind='deterministic-milestone'``);
non-zero → a born-at-L2 ``milestone_check_failed`` escalation. The underlying
predicate mechanism already exists; this is only the helper that files it
correctly.

## Critical interaction — execution_class MUST be 'code_tdd'

``deterministic_task_guard`` invariant 6 FORBIDS ``always_escalates=True`` on a
``before_done.kind='predicate'`` task. But ``execution_class`` in
``{'operational', 'decision'}`` is coerced by ``route_deterministic`` /
``operational_ask_registry`` to ``task_kind='deterministic'`` +
``always_escalates=True`` — which would make a predicate submission illegal.
``'code_tdd'`` is the ONLY execution_class that does not trigger that coercion,
so the helper fixes it to ``'code_tdd'`` and FAILS LOUD on any other value
(matching the project's loud-over-silent-degradation norm) rather than silently
overriding a caller who reached for the wrong class.

## Purity

The builder performs NO filesystem or network I/O: shape validation is
delegated to the shared :class:`shared.task_metadata.BeforeDone` model, and the
script-exists/executable check is the submit-time ``deterministic_task_guard``'s
job (mirroring the existing shape-vs-filesystem split). This keeps the builder
hermetically testable and callable from any stage without side effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from shared.task_metadata import BeforeDone

from fused_memory.reconciliation.recon_self_model import EXECUTION_CLASSES

__all__ = [
    'PROVENANCE_METADATA_KEY',
    'PredicateContradictionTask',
    'build_predicate_contradiction_task',
    'render_predicate_contradiction_section',
]

# The x_-prefixed forward-compat namespace (silently allowed by parse_metadata,
# no code=unknown_key schema-warning census line) under which contradiction
# provenance is stored, linking the predicate task back to the finding/run/cited
# memories durably and greppably.
PROVENANCE_METADATA_KEY = 'x_recon_predicate_contradiction'

# The only execution_class compatible with a before_done.kind='predicate' task —
# see the module docstring's "Critical interaction" section.
_REQUIRED_EXECUTION_CLASS = 'code_tdd'


@dataclass(frozen=True)
class PredicateContradictionTask:
    """An assembled, not-yet-submitted deterministic predicate task.

    ``as_submit_task_kwargs()`` returns a dict splattable directly into
    ``mcp__fused-memory__submit_task`` (the caller adds ``project_root``).
    """

    title: str
    description: str
    priority: str
    task_kind: str
    metadata: dict[str, Any]

    def as_submit_task_kwargs(self) -> dict[str, Any]:
        """Return the submit_task keyword arguments for this task.

        Keys are a subset of ``{title, description, priority, task_kind,
        metadata}`` — the caller supplies ``project_root`` (and any
        ``planning_mode``) at the call site.
        """
        return {
            'title': self.title,
            'description': self.description,
            'priority': self.priority,
            'task_kind': self.task_kind,
            'metadata': self.metadata,
        }


def _default_title(contradiction: str) -> str:
    stripped = contradiction.strip()
    base = 'Predicate check (recon contradiction)'
    if not stripped:
        return base
    snippet = stripped.splitlines()[0][:80]
    return f'{base}: {snippet}'


def _default_description(contradiction: str, script: str, args: Sequence[str]) -> str:
    invocation = ' '.join([script, *(str(a) for a in args)])
    return (
        'A reconciliation-stage contradiction can only be settled by live command '
        'execution (a specific pytest test, a git grep/log check). This '
        "deterministic predicate task runs the check and decides by exit code: "
        'exit 0 -> task done (done_provenance.kind=deterministic-milestone); '
        'non-zero -> born-at-L2 milestone_check_failed escalation + task blocked. '
        'Filed via build_predicate_contradiction_task instead of an ad-hoc Mem0 '
        'investigation note that would silently age across cycles.\n\n'
        f'Contradiction: {contradiction}\n'
        f'Predicate invocation: {invocation}'
    )


def build_predicate_contradiction_task(
    *,
    contradiction: str,
    script: str,
    timeout_secs: int,
    args: Sequence[str] = (),
    finding_id: str | None = None,
    run_id: str | None = None,
    cited_memory_ids: Sequence[str] = (),
    priority: str = 'high',
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    execution_class: str = _REQUIRED_EXECUTION_CLASS,
    title: str | None = None,
    description: str | None = None,
) -> PredicateContradictionTask:
    """Build a ``task_kind='deterministic'`` + ``before_done.kind='predicate'``
    task for a recon-stage contradiction that only live execution can settle.

    PURE — no filesystem/network I/O. Shape validation of ``before_done`` is
    delegated to :class:`shared.task_metadata.BeforeDone` (min-length script,
    positive ``timeout_secs``); the script-exists/executable check is left to
    the submit-time ``deterministic_task_guard``.

    Args:
        contradiction: Human-readable statement of the contradiction being settled.
        script: Project-relative path to a committed, executable predicate runner
            (e.g. ``scripts/recon_predicate_check.sh``).
        timeout_secs: Positive per-check wall-clock timeout.
        args: Arguments passed to the predicate script.
        finding_id / run_id / cited_memory_ids: Provenance linking the predicate
            task back to the recon finding / run / cited memories.
        priority: Task priority (default ``'high'``).
        cwd / env: Optional ``before_done`` working directory / environment.
        execution_class: Must be ``'code_tdd'`` (the only class compatible with a
            predicate — see the module docstring). Any other value FAILS LOUD in
            step-6's validation.
        title / description: Optional overrides; sensible defaults are derived
            from *contradiction* / *script* / *args*.

    Raises:
        ValueError: if *execution_class* is not ``'code_tdd'``, or if the
            ``before_done`` shape is invalid (empty *script*, non-positive
            *timeout_secs*) — the underlying pydantic ``ValidationError`` is
            preserved as ``__cause__``.
    """
    # Fail loud: 'code_tdd' is the ONLY execution_class compatible with a
    # before_done.kind='predicate' task. execution_class in {'operational',
    # 'decision'} is coerced by route_deterministic / operational_ask_registry
    # to task_kind='deterministic' + always_escalates=True, which
    # deterministic_task_guard invariant 6 forbids for a predicate. Reject any
    # other value outright (rather than silently overriding a caller who reached
    # for the wrong class) per the loud-over-silent-degradation norm.
    if execution_class != _REQUIRED_EXECUTION_CLASS:
        raise ValueError(
            f'execution_class={execution_class!r} is invalid for a predicate '
            f'contradiction task: {_REQUIRED_EXECUTION_CLASS!r} is the only '
            "compatible class. A before_done.kind='predicate' task forbids "
            'always_escalates=True (deterministic_task_guard invariant 6), but '
            "execution_class in {'operational', 'decision'} is coerced to "
            "task_kind='deterministic' + always_escalates=True at the submit "
            f'boundary — so only {_REQUIRED_EXECUTION_CLASS!r} can be used here.'
        )
    # Defensive single-source check: the fixed class must be a member of the
    # guard's accepted set, so this helper can never drift from
    # execution_class_guard's allowlist.
    assert _REQUIRED_EXECUTION_CLASS in EXECUTION_CLASSES

    try:
        before_done = BeforeDone(
            kind='predicate',
            script=script,
            args=list(args),
            timeout_secs=timeout_secs,
            cwd=cwd,
            env=dict(env or {}),
        ).model_dump(exclude_none=True)
    except ValidationError as exc:
        # Re-wrap as ValueError with builder context, preserving the pydantic
        # detail as __cause__ so the fail-loud contract is uniformly ValueError.
        raise ValueError(
            'before_done shape validation failed for predicate contradiction '
            f'task (script={script!r}, timeout_secs={timeout_secs!r}): {exc}'
        ) from exc

    metadata: dict[str, Any] = {
        'task_kind': 'deterministic',
        'before_done': before_done,
        'execution_class': execution_class,
        PROVENANCE_METADATA_KEY: {
            'contradiction': contradiction,
            'finding_id': finding_id,
            'run_id': run_id,
            'cited_memory_ids': list(cited_memory_ids),
        },
    }

    return PredicateContradictionTask(
        title=title or _default_title(contradiction),
        description=description or _default_description(contradiction, script, args),
        priority=priority,
        task_kind='deterministic',
        metadata=metadata,
    )


def render_predicate_contradiction_section() -> str:
    """Render the predicate-contradiction filing-convention prompt section
    (task 2779), following the ``render_*`` style in ``recon_self_model.py``.

    Interpolated into the Stage 2 system prompt so recon stages that file tasks
    see WHEN to use this path, HOW to file it, the exit-code contract, and WHY
    ``execution_class`` must be ``'code_tdd'``. The ``code_tdd`` token and the
    forbidden-class list are single-sourced from :data:`EXECUTION_CLASSES` so
    the rendered prose cannot drift from the guard's accepted set.
    """
    forbidden = ', '.join(
        f'`{c}`' for c in EXECUTION_CLASSES if c != _REQUIRED_EXECUTION_CLASS
    )
    return (
        '## Predicate Contradiction Checks\n'
        'When a contradiction can only be settled by LIVE command execution — a '
        'specific pytest test, a `git grep`, a `git log --grep` — do NOT settle '
        'it with an ad-hoc Mem0 investigation note or suppression record. Those '
        'silently age across cycles without ever being decided (the task-2643 '
        'precedent, resolved manually over two investigation cycles). Instead of '
        'that, file a one-off deterministic PREDICATE task that runs the check '
        'and is decided automatically by its exit code.\n\n'
        'HOW: build the submission with '
        '`build_predicate_contradiction_task(...)` (module '
        '`fused_memory.reconciliation.predicate_contradiction`) and submit its '
        "`as_submit_task_kwargs()` — it emits a "
        "`submit_task(task_kind='deterministic', ...)` whose "
        "`metadata.before_done.kind='predicate'` points at a committed, "
        'executable check script (e.g. `scripts/recon_predicate_check.sh`) and '
        f"whose `metadata.execution_class='{_REQUIRED_EXECUTION_CLASS}'`.\n\n"
        'EXIT-CODE CONTRACT (the runner parses the exit code only, never '
        'stdout):\n'
        "  - exit 0 -> task done, done_provenance.kind='deterministic-milestone'\n"
        '  - non-zero -> born-at-L2 `milestone_check_failed` escalation + task '
        'blocked\n\n'
        f"WHY execution_class MUST be `{_REQUIRED_EXECUTION_CLASS}`: a "
        "`before_done.kind='predicate'` task forbids `always_escalates=True` "
        '(deterministic_task_guard invariant 6), but the '
        f'{forbidden} execution classes are coerced at the submit boundary to '
        "`task_kind='deterministic'` + `always_escalates=True` — which would "
        f'make the predicate submission illegal. `{_REQUIRED_EXECUTION_CLASS}` '
        'is the only class that does not trigger that coercion, so it is the '
        'only one compatible with a recon-filed predicate task; the helper '
        'fixes it and rejects any other value.'
    )
