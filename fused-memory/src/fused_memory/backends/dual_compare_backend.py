"""Soak wrapper that drives two task backends in lock-step.

Routes every call to both *primary* and *secondary*, compares the
responses, and logs structured divergences. The caller's return value is
always the *primary*'s — secondary failures are swallowed (logged) so the
wrapper is safe to flip on/off without changing externally-observable
behaviour.

This is the engine for the cutover-soak between Taskmaster (the legacy
default primary) and SqliteTaskBackend (the eventual successor). Once the
divergence log goes silent for a soak window, the operator flips the
``dual_compare_primary`` config field and runs the inverse comparison
before retiring the legacy backend entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

if TYPE_CHECKING:
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

# Threaded by the task interceptor's ``_journal_around`` helper so the
# wrapper can link its per-side ``backend_op`` rows to the parent
# ``write_op``. ``None`` when no journaling is wired.
_current_write_op_id: ContextVar[str | None] = ContextVar(
    'dual_compare_current_write_op_id', default=None,
)


def _resolve_journal_backend_label(backend: Any) -> str:
    """Stable journal label per physical backend ('taskmaster',
    'sqlite_task_backend', or the lowercased class name)."""
    if backend is None:
        return 'unknown'
    cls = type(backend).__name__
    if cls == 'TaskmasterBackend':
        return 'taskmaster'
    if cls == 'SqliteTaskBackend':
        return 'sqlite_task_backend'
    return cls.lower()


def _summarize(value: Any, *, limit: int = 240) -> str:
    """Render *value* as a stable JSON-ish string, clipped at *limit* chars.

    Used in divergence log lines so operators can spot wire-shape drift at
    a glance without flooding the log with multi-KB DTOs.
    """
    try:
        rendered = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    if len(rendered) > limit:
        rendered = rendered[:limit] + '…'
    return rendered


class DualCompareBackend:
    """Wraps two backends; serves the primary, mirrors writes to the secondary.

    Reads compare both responses; writes apply both then compare. The
    secondary's exceptions never propagate — they are logged so a buggy
    secondary can never break the production hot path.
    """

    def __init__(
        self,
        primary: TaskBackendProtocol,
        secondary: TaskBackendProtocol,
        *,
        primary_label: str = 'primary',
        secondary_label: str = 'secondary',
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_label = primary_label
        self.secondary_label = secondary_label
        self.divergence_count = 0
        # Counter incremented when the secondary's ``ensure_connected`` raises;
        # primary's failure still propagates (preserves the existing contract
        # that the wrapper is transparent on the primary's hot path).
        self.secondary_health_failures = 0
        # Counters and bookkeeping for cancellation-safe dispatch (Commit 4):
        # without these, an outer cancel of the request handler tore _dispatch
        # mid-flight — sqlite's local commit often landed while tm's stdio +
        # tasks.json rewrite did not — and the comparator never ran, so the
        # divergence went silently unrecorded.
        self.cancelled_dispatch_count = 0
        self._inflight_drains: set[asyncio.Task] = set()
        # Per-write read-back verification (Commit 5):
        # the input-echo normalisers (``_normalize_set_status``,
        # ``_normalize_id_only``) only confirm both backends agreed on what
        # they were told to do, NOT on the resulting state. Verification
        # re-reads the affected task on both backends and compares the
        # normalised wire shape, surfacing the silent-write-drop case the
        # original comparator was structurally blind to.
        self.verifies_total = 0
        self.verifies_diverged = 0
        self.verifies_by_method: dict[str, dict[str, int]] = {}
        self._inflight_verifies: set[asyncio.Task] = set()
        # Optional write-journal hook: when wired, ``_dispatch_pair`` emits
        # one ``backend_op`` per physical backend (sqlite + taskmaster)
        # linked to the parent ``write_op`` via the contextvar populated by
        # the interceptor's ``_journal_around`` helper. Supports the
        # post-soak directional-attribution query.
        self._write_journal: WriteJournal | None = None
        # Stable per-side labels used by the journal rows; resolved from
        # the underlying backends' class names so they survive renames.
        self._primary_journal_label = _resolve_journal_backend_label(primary)
        self._secondary_journal_label = _resolve_journal_backend_label(secondary)

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self.primary.connected

    @property
    def restart_count(self) -> int:
        return self.primary.restart_count

    @property
    def config(self) -> Any:
        # Belt-and-braces: any caller that still does
        # ``backend.config.project_root`` on the wrapper now gets the
        # primary's config instead of AttributeError.
        return getattr(self.primary, 'config', None)

    def set_write_journal(self, journal: 'WriteJournal') -> None:
        """Attach a write-journal so per-physical-backend ``backend_op`` rows
        are emitted alongside the interceptor's combined ``backend_op``."""
        self._write_journal = journal

    async def start(self) -> None:
        await self.primary.start()
        try:
            await self.secondary.start()
        except Exception as exc:
            logger.warning(
                'dual_compare: secondary.start() failed: %s; comparator disabled',
                exc,
            )

    async def initialize(self) -> None:
        await self.start()

    async def ensure_connected(self) -> None:
        # Mirror the start() pattern: poke both sides so a silently-broken
        # secondary connection can't keep limping unobserved across the
        # whole soak. Primary failures still propagate (callers depend on
        # this); secondary failures are logged and counted only.
        p_task = asyncio.ensure_future(self.primary.ensure_connected())
        s_task = asyncio.ensure_future(self.secondary.ensure_connected())
        p_res, s_res = await asyncio.gather(p_task, s_task, return_exceptions=True)
        if isinstance(p_res, BaseException):
            raise p_res
        if isinstance(s_res, BaseException):
            self.secondary_health_failures += 1
            logger.warning(
                'dual_compare: secondary.ensure_connected failed: %s', s_res,
            )

    async def close(self) -> None:
        # Wait for any in-flight drain tasks to settle so divergence records
        # for cancelled-during-dispatch calls aren't lost on shutdown. Same
        # for read-back verifies — letting them complete keeps the soak
        # signal coherent up to the moment of shutdown.
        pending = list(self._inflight_drains) + list(self._inflight_verifies)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await self.primary.close()
        try:
            await self.secondary.close()
        except Exception as exc:
            logger.warning('dual_compare: secondary.close() failed: %s', exc)

    async def is_alive(self) -> tuple[bool, str | None]:
        return await self.primary.is_alive()

    # ── Comparator core ────────────────────────────────────────────────

    async def _dispatch_pair(
        self, method_name: str, args: tuple, kwargs: dict,
        *, normalize: Callable[[Any], Any] | None = None,
    ) -> tuple[Any, BaseException | None, Any, BaseException | None]:
        """Cancellation-safe dispatch returning both backends' outcomes.

        Both calls run concurrently via ``asyncio.gather(return_exceptions=True)``
        wrapped in ``asyncio.shield`` so an outer cancellation can't tear the
        underlying gather mid-flight (which previously left sqlite-committed +
        tm-not-committed silent drift). On caller cancel we still raise
        ``CancelledError`` to honour the cancellation contract, but a
        background drain task awaits the inner pair to completion and runs
        ``_compare`` so the divergence is logged after the fact. The drain
        is tracked on ``self._inflight_drains`` so :meth:`close` can wait
        for it on shutdown.

        Returns ``(primary_value, primary_exc, secondary_value, secondary_exc)``
        where exactly one of value/exc is non-None on each side. Callers
        that don't need both outcomes go through :meth:`_dispatch`, which
        re-raises ``primary_exc`` and returns ``primary_value``.
        """
        primary_method = getattr(self.primary, method_name)
        secondary_method = getattr(self.secondary, method_name)

        inner = asyncio.ensure_future(asyncio.gather(
            primary_method(*args, **kwargs),
            secondary_method(*args, **kwargs),
            return_exceptions=True,
        ))

        try:
            primary_outcome, secondary_outcome = await asyncio.shield(inner)
        except asyncio.CancelledError:
            self.cancelled_dispatch_count += 1
            self._spawn_drain(method_name, args, kwargs, inner, normalize)
            raise

        primary_exc = primary_outcome if isinstance(primary_outcome, BaseException) else None
        primary_value = None if primary_exc is not None else primary_outcome
        secondary_exc = (
            secondary_outcome if isinstance(secondary_outcome, BaseException) else None
        )
        secondary_value = None if secondary_exc is not None else secondary_outcome

        self._compare(
            method_name, args, kwargs,
            primary_value, primary_exc,
            secondary_value, secondary_exc,
            normalize=normalize,
        )
        # Journal per-physical-backend rows linked to the parent write_op
        # (set by the interceptor). Only fires when the journal is wired
        # AND we're inside an interceptor-issued write — read paths leave
        # the contextvar at None and skip the rows.
        if self._write_journal is not None:
            wo_id = _current_write_op_id.get()
            if wo_id is not None:
                await self._journal_per_side(
                    wo_id, method_name, kwargs,
                    primary_value, primary_exc,
                    secondary_value, secondary_exc,
                )
        return primary_value, primary_exc, secondary_value, secondary_exc

    async def _journal_per_side(
        self,
        write_op_id: str,
        method_name: str,
        kwargs: dict,
        primary_value: Any,
        primary_exc: BaseException | None,
        secondary_value: Any,
        secondary_exc: BaseException | None,
    ) -> None:
        """Emit one ``backend_op`` row per physical backend.

        Failures are logged at debug only — journal misses must never
        disturb the live request path.
        """
        ji = self._write_journal
        if ji is None:
            return
        for label, exc, value in (
            (self._primary_journal_label, primary_exc, primary_value),
            (self._secondary_journal_label, secondary_exc, secondary_value),
        ):
            try:
                await ji.log_backend_op(
                    write_op_id=write_op_id,
                    backend=label,
                    operation=method_name,
                    payload={k: _summarize(v, limit=120) for k, v in kwargs.items()},
                    result_summary=(
                        None if exc is not None
                        else (str(value)[:500] if value is not None else None)
                    ),
                    success=exc is None,
                    error=str(exc)[:500] if exc is not None else None,
                )
            except Exception:
                logger.debug(
                    'dual_compare: per-side log_backend_op failed (%s)',
                    label, exc_info=True,
                )

    async def _dispatch(
        self, method_name: str, args: tuple, kwargs: dict,
        *, normalize: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Backwards-compatible wrapper around :meth:`_dispatch_pair` that
        exposes only the primary's outcome (re-raising on primary failure).
        """
        p_val, p_exc, _s_val, _s_exc = await self._dispatch_pair(
            method_name, args, kwargs, normalize=normalize,
        )
        if p_exc is not None:
            raise p_exc
        return p_val

    def _spawn_drain(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        inner: asyncio.Future,
        normalize: Callable[[Any], Any] | None,
    ) -> None:
        """Schedule a background drain that awaits *inner* and runs the
        comparator, so cancellation-during-dispatch still produces a
        divergence record. Tracked on ``self._inflight_drains``.
        """
        async def _drain() -> None:
            try:
                primary_outcome, secondary_outcome = await inner
            except BaseException as exc:
                # If the gather itself was cancelled (e.g. event loop
                # shutdown), there's nothing to compare; skip silently.
                logger.debug(
                    'dual_compare: drain inner await failed: %s', exc,
                )
                return
            try:
                p_exc = primary_outcome if isinstance(primary_outcome, BaseException) else None
                s_exc = secondary_outcome if isinstance(secondary_outcome, BaseException) else None
                self._compare(
                    method_name, args, kwargs,
                    None if p_exc else primary_outcome, p_exc,
                    None if s_exc else secondary_outcome, s_exc,
                    normalize=normalize,
                )
            except Exception as exc:
                logger.warning('dual_compare: drain compare failed: %s', exc)

        drain = asyncio.create_task(
            _drain(), name=f'dual_compare-drain-{method_name}',
        )
        self._inflight_drains.add(drain)
        drain.add_done_callback(self._inflight_drains.discard)

    def _spawn_verify(
        self,
        method_name: str,
        primary_target: str,
        secondary_target: str,
        project_root: str,
        tag: str | None,
        *,
        strip_ids: bool = False,
        expect_not_found: bool = False,
    ) -> None:
        """Schedule a fire-and-forget read-back verify of a write.

        After a write, both backends should produce equivalent state on
        the affected task. Re-read each side independently, normalise via
        :func:`_normalize_task` (optionally stripping minted ids for
        ``add_task`` / ``add_subtask`` whose ids legitimately differ),
        and log a divergence when the wire shapes drift.

        ``expect_not_found=True`` (used by ``remove_task``) inverts the
        contract: both sides should raise ``TaskmasterError``; either one
        succeeding is the divergence.
        """
        async def _verify() -> None:
            method_stats = self.verifies_by_method.setdefault(
                method_name, {'total': 0, 'diverged': 0},
            )
            self.verifies_total += 1
            method_stats['total'] += 1

            async def _read(backend: TaskBackendProtocol, target: str):
                try:
                    return ('value', await backend.get_task(target, project_root, tag))
                except BaseException as exc:  # noqa: BLE001 — surface to compare
                    return ('exc', exc)

            primary_kind_val, secondary_kind_val = await asyncio.gather(
                _read(self.primary, primary_target),
                _read(self.secondary, secondary_target),
                return_exceptions=False,
            )
            p_kind, p_payload = primary_kind_val
            s_kind, s_payload = secondary_kind_val

            diverged = False
            if expect_not_found:
                # Both should be exceptions. If either is a value, that's a
                # divergence; if both are exceptions, type-equality is enough.
                if p_kind == 'exc' and s_kind == 'exc':
                    return  # both raised — no divergence
                diverged = True
                self._log_verify_presence_divergence(
                    method_name, primary_target, secondary_target,
                    p_kind, p_payload, s_kind, s_payload,
                )
            else:
                if p_kind != s_kind:
                    diverged = True
                    self._log_verify_presence_divergence(
                        method_name, primary_target, secondary_target,
                        p_kind, p_payload, s_kind, s_payload,
                    )
                elif p_kind == 'value':
                    p_norm = _normalize_task(p_payload)
                    s_norm = _normalize_task(s_payload)
                    if strip_ids and isinstance(p_norm, dict) and isinstance(s_norm, dict):
                        p_norm = _strip_ids(p_norm)
                        s_norm = _strip_ids(s_norm)
                    if p_norm != s_norm:
                        diverged = True
                        self._log_verify_field_divergence(
                            method_name, primary_target, secondary_target,
                            p_norm, s_norm,
                        )

            if diverged:
                self.verifies_diverged += 1
                method_stats['diverged'] += 1

        try:
            verify = asyncio.create_task(
                _verify(), name=f'dual_compare-verify-{method_name}',
            )
        except RuntimeError:
            # No running loop (rare; e.g. close() races) — fail safe.
            return
        self._inflight_verifies.add(verify)
        verify.add_done_callback(self._inflight_verifies.discard)

    def _log_verify_presence_divergence(
        self,
        method_name: str,
        primary_target: str,
        secondary_target: str,
        p_kind: str,
        p_payload: Any,
        s_kind: str,
        s_payload: Any,
    ) -> None:
        logger.warning(
            'dual_compare.verify_divergence method=%s primary_target=%s '
            'secondary_target=%s %s=%s(%s) %s=%s(%s)',
            method_name, primary_target, secondary_target,
            self.primary_label, p_kind, _summarize(p_payload, limit=120),
            self.secondary_label, s_kind, _summarize(s_payload, limit=120),
        )

    def _log_verify_field_divergence(
        self,
        method_name: str,
        primary_target: str,
        secondary_target: str,
        p_task: Any,
        s_task: Any,
    ) -> None:
        diff = _field_diff(p_task, s_task)
        if not diff:
            return
        rendered = _format_field_diff(
            diff, primary_label=self.primary_label,
            secondary_label=self.secondary_label,
        )
        logger.warning(
            'dual_compare.verify_divergence method=%s primary_target=%s '
            'secondary_target=%s fields={%s}',
            method_name, primary_target, secondary_target, rendered,
        )

    def _compare(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        primary_value: Any,
        primary_exc: BaseException | None,
        secondary_value: Any,
        secondary_exc: BaseException | None,
        *,
        normalize: Callable[[Any], Any] | None,
    ) -> None:
        """Log a structured divergence line if the two responses disagree.

        Format hardening (Commit 6): the previous "primary={blob}
        secondary={blob}" form clipped at 240 chars mid-content and was
        unreadable for tasks-tree divergences (743 unintelligible lines
        in one storm window). For ``get_task`` / ``get_tasks`` we now
        walk the normalised payload and emit ONE log line per
        differing task, listing only the differing fields. Other
        methods (write input-echos) retain the small summary line.
        """
        primary_kind = type(primary_exc).__name__ if primary_exc else 'value'
        secondary_kind = type(secondary_exc).__name__ if secondary_exc else 'value'

        if primary_exc is not None or secondary_exc is not None:
            if primary_kind != secondary_kind:
                self._log_divergence(
                    method_name, args, kwargs,
                    f'{self.primary_label}={primary_kind}({primary_exc})',
                    f'{self.secondary_label}={secondary_kind}({secondary_exc})',
                )
            return

        p = normalize(primary_value) if normalize else primary_value
        s = normalize(secondary_value) if normalize else secondary_value
        if p == s:
            return

        if method_name == 'get_tasks':
            self._log_tasks_tree_divergence(method_name, args, kwargs, p, s)
        elif method_name == 'get_task':
            self._log_task_field_divergence(method_name, args, kwargs, p, s)
        else:
            self._log_divergence(
                method_name, args, kwargs,
                f'{self.primary_label}={_summarize(p)}',
                f'{self.secondary_label}={_summarize(s)}',
            )

    def _log_tasks_tree_divergence(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        p_norm: Any,
        s_norm: Any,
    ) -> None:
        """Walk both ``{tasks: [...]}`` payloads keyed by id and emit one log
        line per task that differs, listing only the differing fields."""
        p_by_id = _index_tasks_by_id(p_norm)
        s_by_id = _index_tasks_by_id(s_norm)
        all_ids = sorted(set(p_by_id) | set(s_by_id), key=lambda x: str(x))
        emitted = 0
        for task_id in all_ids:
            p_task = p_by_id.get(task_id)
            s_task = s_by_id.get(task_id)
            if p_task == s_task:
                continue
            self._log_per_task_diff(
                method_name, args, kwargs, task_id, p_task, s_task,
            )
            emitted += 1
        if emitted == 0 and p_by_id != s_by_id:
            # Disjoint outer shape (e.g. neither is a tasks list).
            self._log_divergence(
                method_name, args, kwargs,
                f'{self.primary_label}={_summarize(p_norm)}',
                f'{self.secondary_label}={_summarize(s_norm)}',
            )

    def _log_task_field_divergence(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        p_norm: Any,
        s_norm: Any,
    ) -> None:
        """Single-task divergence — emit only the differing fields."""
        task_id = (p_norm.get('id') if isinstance(p_norm, dict) else None) \
                  or (s_norm.get('id') if isinstance(s_norm, dict) else None) \
                  or '?'
        self._log_per_task_diff(
            method_name, args, kwargs, task_id, p_norm, s_norm,
        )

    def _log_per_task_diff(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        task_id: Any,
        p_task: Any,
        s_task: Any,
    ) -> None:
        if p_task is None or s_task is None:
            kind = 'primary_only' if s_task is None else 'secondary_only'
            self.divergence_count += 1
            logger.warning(
                'dual_compare.divergence method=%s task=%s presence=%s '
                '%s=%s %s=%s',
                method_name, task_id, kind,
                self.primary_label, _summarize(p_task),
                self.secondary_label, _summarize(s_task),
            )
            return

        diff = _field_diff(p_task, s_task)
        if not diff:
            return
        self.divergence_count += 1
        rendered = _format_field_diff(
            diff, primary_label=self.primary_label,
            secondary_label=self.secondary_label,
        )
        logger.warning(
            'dual_compare.divergence method=%s task=%s fields={%s}',
            method_name, task_id, rendered,
        )

    def _log_divergence(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        primary_repr: str,
        secondary_repr: str,
    ) -> None:
        self.divergence_count += 1
        logger.warning(
            'dual_compare.divergence method=%s args=%s kwargs=%s %s %s',
            method_name,
            _summarize(list(args)),
            _summarize(kwargs),
            primary_repr,
            secondary_repr,
        )

    # ── Read methods ───────────────────────────────────────────────────

    async def get_tasks(self, project_root: str, tag: str | None = None):
        return await self._dispatch(
            'get_tasks', (project_root,), {'tag': tag},
            normalize=_normalize_tasks_tree,
        )

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None,
    ):
        return await self._dispatch(
            'get_task', (task_id, project_root), {'tag': tag},
            normalize=_normalize_task,
        )

    # ── Mutation methods ───────────────────────────────────────────────

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ):
        p_val, p_exc, _s_val, _s_exc = await self._dispatch_pair(
            'set_task_status',
            (task_id, status, project_root),
            {'tag': tag},
            normalize=_normalize_set_status,
        )
        if p_exc is not None:
            raise p_exc
        # Read-back: both backends should now show the same task state.
        self._spawn_verify(
            'set_task_status', str(task_id), str(task_id),
            project_root, tag,
        )
        return p_val

    async def add_task(self, project_root: str, **kwargs):
        # add_task mints an id — the two backends won't agree on that, so
        # we strip ids out of the comparator and only check that both either
        # succeeded or both raised.
        p_val, p_exc, s_val, s_exc = await self._dispatch_pair(
            'add_task', (project_root,), kwargs,
            normalize=_normalize_id_only,
        )
        if p_exc is not None:
            raise p_exc
        # Verify only when both sides minted ids — no read target otherwise.
        if (
            isinstance(p_val, dict) and 'id' in p_val
            and isinstance(s_val, dict) and 'id' in s_val
        ):
            self._spawn_verify(
                'add_task', str(p_val['id']), str(s_val['id']),
                project_root, kwargs.get('tag'),
                strip_ids=True,
            )
        return p_val

    async def update_task(
        self,
        task_id: str,
        project_root: str,
        prompt: str | None = None,
        metadata: str | None = None,
        append: bool = False,
        tag: str | None = None,
        *,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        dependencies: list[str] | None = None,
    ):
        """Asymmetric per-side dispatch for the sqlite↔taskmaster cutover.

        The legacy ``update_task`` path treats both sides symmetrically
        (same args to both backends). That hides two real issues:

        1. Sqlite's old ``update_task`` only wrote details/metadata/updated_at
           — a structured priority/status change funnelled through the
           prompt-based wire silently dropped.
        2. TM's ``update_task`` is LLM-driven; passing structured fields to
           it would either be ignored or trigger an LLM rewrite, producing
           drift.

        New semantics:

        * **Case 1 — caller passes structured fields**: sqlite gets the
          structured form directly (no prompt). TM is dispatched per-field
          where it has a per-field tool (``set_task_status``); for
          title/description/details/priority TM's only knob is its LLM
          ``update_task``, so we feed it a verbatim-replace prompt that
          constrains the LLM to copy verbatim.
        * **Case 2 — legacy prompt-only call**: TM runs its LLM rewrite
          first; sqlite is teed with the structured fields extracted from
          TM's resolved ``updated_task``. Sqlite never runs an LLM.

        For non-(sqlite+TM) wrappings (test fixtures, future combos) this
        falls back to the prior symmetric ``_dispatch`` behaviour with the
        full kwarg surface.
        """
        sqlite_be = self._side_of_class('SqliteTaskBackend')
        tm_be = self._side_of_class('TaskmasterBackend')
        common_kwargs: dict[str, Any] = {
            'prompt': prompt, 'metadata': metadata, 'append': append, 'tag': tag,
            'title': title, 'description': description, 'details': details,
            'priority': priority, 'status': status, 'dependencies': dependencies,
        }

        if sqlite_be is None or tm_be is None:
            # Symmetric fallback — neither side is the cutover pair we
            # know how to translate for. Pass the full kwarg surface to
            # both and let them raise/ignore as they choose.
            p_val, p_exc, _s_val, _s_exc = await self._dispatch_pair(
                'update_task',
                (task_id, project_root),
                common_kwargs,
                normalize=_normalize_id_only,
            )
            if p_exc is not None:
                raise p_exc
            self._spawn_verify(
                'update_task', str(task_id), str(task_id), project_root, tag,
            )
            return p_val

        primary_is_sqlite = sqlite_be is self.primary
        structured_present = any(
            v is not None
            for v in (title, description, details, priority, status, dependencies)
        )

        if structured_present:
            sqlite_value, sqlite_exc, tm_value, tm_exc = (
                await self._update_task_concurrent(
                    task_id, project_root,
                    sqlite_be=sqlite_be, tm_be=tm_be,
                    prompt=prompt, metadata=metadata, append=append, tag=tag,
                    title=title, description=description, details=details,
                    priority=priority, status=status, dependencies=dependencies,
                )
            )
        else:
            sqlite_value, sqlite_exc, tm_value, tm_exc = (
                await self._update_task_tee_legacy(
                    task_id, project_root,
                    sqlite_be=sqlite_be, tm_be=tm_be,
                    prompt=prompt, metadata=metadata, append=append, tag=tag,
                )
            )

        if primary_is_sqlite:
            primary_value, primary_exc = sqlite_value, sqlite_exc
            secondary_value, secondary_exc = tm_value, tm_exc
        else:
            primary_value, primary_exc = tm_value, tm_exc
            secondary_value, secondary_exc = sqlite_value, sqlite_exc

        self._compare(
            'update_task', (task_id, project_root), common_kwargs,
            primary_value, primary_exc, secondary_value, secondary_exc,
            normalize=_normalize_id_only,
        )
        if self._write_journal is not None:
            wo_id = _current_write_op_id.get()
            if wo_id is not None:
                await self._journal_per_side(
                    wo_id, 'update_task', common_kwargs,
                    primary_value, primary_exc, secondary_value, secondary_exc,
                )

        if primary_exc is not None:
            raise primary_exc
        self._spawn_verify(
            'update_task', str(task_id), str(task_id), project_root, tag,
        )
        return primary_value

    def _side_of_class(self, class_name: str) -> Any:
        """Return whichever side (primary or secondary) is an instance of
        *class_name* (matched on ``type(...).__name__``), or ``None``.

        Class-name string match avoids importing ``SqliteTaskBackend`` /
        ``TaskmasterBackend`` here — keeps this module free of circular
        dependencies on the concrete backends it routes between.
        """
        if type(self.primary).__name__ == class_name:
            return self.primary
        if type(self.secondary).__name__ == class_name:
            return self.secondary
        return None

    async def _update_task_concurrent(
        self,
        task_id: str,
        project_root: str,
        *,
        sqlite_be: Any,
        tm_be: Any,
        prompt: str | None,
        metadata: str | None,
        append: bool,
        tag: str | None,
        title: str | None,
        description: str | None,
        details: str | None,
        priority: str | None,
        status: str | None,
        dependencies: list[str] | None,
    ) -> tuple[Any, BaseException | None, Any, BaseException | None]:
        """Case 1 — both sides run concurrently with per-side translation.

        Returns ``(sqlite_value, sqlite_exc, tm_value, tm_exc)`` mirroring
        the layout of :meth:`_dispatch_pair`. Cancellation is shielded so
        a torn outer await still produces a consistent compare downstream.
        """

        async def _sqlite_call() -> Any:
            return await sqlite_be.update_task(
                task_id=task_id,
                project_root=project_root,
                prompt=None,
                metadata=metadata,
                append=append,
                tag=tag,
                title=title,
                description=description,
                details=details,
                priority=priority,
                status=status,
                dependencies=dependencies,
            )

        async def _tm_call() -> Any:
            # Verbatim prompt covers the fields TM has no per-field tool
            # for. Status is dispatched separately (TM has set_task_status).
            # Dependencies replacement is intentionally skipped on TM —
            # rare in practice, and post-cutover TM's tasks.json is
            # discarded anyway.
            verbatim_pairs: list[tuple[str, str]] = []
            if title is not None:
                verbatim_pairs.append(('TITLE', title))
            if description is not None:
                verbatim_pairs.append(('DESCRIPTION', description))
            if priority is not None:
                verbatim_pairs.append(('PRIORITY', priority))
            if details is not None:
                verbatim_pairs.append(('DETAILS', details))

            tm_result: Any = None
            if verbatim_pairs:
                lines = [
                    'Replace this task with EXACTLY the following fields, '
                    'verbatim. Do not paraphrase, do not merge with existing '
                    'content, do not add commentary.',
                ]
                for label, value in verbatim_pairs:
                    lines.append(f'{label}: {value}')
                tm_prompt = '\n\n'.join(lines)
                tm_result = await tm_be.update_task(
                    task_id=task_id,
                    project_root=project_root,
                    prompt=tm_prompt,
                    metadata=metadata,
                    append=append,
                    tag=tag,
                )
            if status is not None:
                await tm_be.set_task_status(task_id, status, project_root, tag)

            if tm_result is None:
                # Synthesize a deterministic result so the caller-side
                # comparator sees a recognisable shape (status-only or
                # deps-only updates).
                tm_result = {
                    'id': str(task_id),
                    'message': 'TM update applied via per-field tools only',
                    'updated': True,
                    'updated_task': None,
                }
            return tm_result

        inner = asyncio.ensure_future(asyncio.gather(
            _sqlite_call(), _tm_call(), return_exceptions=True,
        ))
        try:
            sqlite_outcome, tm_outcome = await asyncio.shield(inner)
        except asyncio.CancelledError:
            self.cancelled_dispatch_count += 1
            self._spawn_drain_pair('update_task', task_id, project_root, inner)
            raise

        sqlite_exc = sqlite_outcome if isinstance(sqlite_outcome, BaseException) else None
        sqlite_value = None if sqlite_exc is not None else sqlite_outcome
        tm_exc = tm_outcome if isinstance(tm_outcome, BaseException) else None
        tm_value = None if tm_exc is not None else tm_outcome
        return sqlite_value, sqlite_exc, tm_value, tm_exc

    async def _update_task_tee_legacy(
        self,
        task_id: str,
        project_root: str,
        *,
        sqlite_be: Any,
        tm_be: Any,
        prompt: str | None,
        metadata: str | None,
        append: bool,
        tag: str | None,
    ) -> tuple[Any, BaseException | None, Any, BaseException | None]:
        """Case 2 — sequential TM-then-sqlite tee for the legacy prompt path.

        TM's LLM rewrite runs first; whatever it materialised in
        ``updated_task`` is then fed to sqlite as structured fields, so
        sqlite's ``tasks.db`` ends up with the deterministic image of
        TM's rewrite. The tee dies with TM at final cutover — at that
        point any caller still passing ``prompt=`` becomes a fallback or
        a hard error per the cleanup PR.
        """
        tm_value: Any = None
        tm_exc: BaseException | None = None
        try:
            tm_value = await tm_be.update_task(
                task_id=task_id,
                project_root=project_root,
                prompt=prompt,
                metadata=metadata,
                append=append,
                tag=tag,
            )
        except BaseException as exc:  # noqa: BLE001 — surface to compare
            tm_exc = exc

        sqlite_value: Any = None
        sqlite_exc: BaseException | None = None
        if tm_exc is not None:
            # TM blew up — don't fabricate a sqlite write. Mirror the
            # exception type so the comparator sees matching kinds.
            sqlite_exc = tm_exc
        else:
            sqlite_kwargs: dict[str, Any] = {
                'task_id': task_id,
                'project_root': project_root,
                'prompt': None,
                'metadata': metadata,
                # Structured tee always replaces — TM's resolved fields are
                # the deterministic source of truth. ``append`` only made
                # sense for the prompt path.
                'append': False,
                'tag': tag,
            }
            updated_task = (
                tm_value.get('updated_task')
                if isinstance(tm_value, dict) else None
            )
            if isinstance(updated_task, dict):
                for field in ('title', 'description', 'details',
                              'priority', 'status'):
                    val = updated_task.get(field)
                    if val is not None:
                        sqlite_kwargs[field] = val
                deps = updated_task.get('dependencies')
                if isinstance(deps, list):
                    sqlite_kwargs['dependencies'] = [str(d) for d in deps]
            try:
                sqlite_value = await sqlite_be.update_task(**sqlite_kwargs)
            except BaseException as exc:  # noqa: BLE001
                sqlite_exc = exc

        return sqlite_value, sqlite_exc, tm_value, tm_exc

    def _spawn_drain_pair(
        self,
        method_name: str,
        task_id: str,
        project_root: str,
        inner: asyncio.Future,
    ) -> None:
        """Drain helper for the asymmetric concurrent dispatch.

        Mirrors :meth:`_spawn_drain` but skips the comparator (asymmetric
        compare requires per-side context that we don't hold here on a
        cancellation); the goal is just to settle ``inner`` so the
        underlying writes complete and the loop doesn't leak the future.
        """
        async def _drain() -> None:
            try:
                await inner
            except BaseException as exc:
                logger.debug(
                    'dual_compare: %s drain await failed: %s',
                    method_name, exc,
                )

        drain = asyncio.create_task(
            _drain(), name=f'dual_compare-drain-{method_name}',
        )
        self._inflight_drains.add(drain)
        drain.add_done_callback(self._inflight_drains.discard)

    async def add_subtask(self, parent_id: str, project_root: str, **kwargs):
        p_val, p_exc, _s_val, _s_exc = await self._dispatch_pair(
            'add_subtask',
            (parent_id, project_root),
            kwargs,
            normalize=_normalize_id_only,
        )
        if p_exc is not None:
            raise p_exc
        # Read parent on both sides; subtask ids legitimately differ so
        # ``strip_ids=True`` strips ids from the parent and its subtasks
        # before comparing.
        self._spawn_verify(
            'add_subtask', str(parent_id), str(parent_id),
            project_root, kwargs.get('tag'),
            strip_ids=True,
        )
        return p_val

    async def remove_tasks(
        self, ids: list[str], project_root: str, tag: str | None = None,
    ):
        p_val, p_exc, _s_val, _s_exc = await self._dispatch_pair(
            'remove_tasks',
            (ids, project_root),
            {'tag': tag},
            normalize=_normalize_remove,
        )
        if p_exc is not None:
            raise p_exc
        # Each removed id should now be absent on both sides — verify
        # per-id so partial divergence (one side missed a delete) surfaces.
        for raw in ids:
            target = str(raw).strip()
            if not target:
                continue
            self._spawn_verify(
                'remove_tasks', target, target,
                project_root, tag,
                expect_not_found=True,
            )
        return p_val

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ):
        p_val, p_exc, _s_val, _s_exc = await self._dispatch_pair(
            'add_dependency',
            (task_id, depends_on, project_root),
            {'tag': tag},
            normalize=_normalize_id_only,
        )
        if p_exc is not None:
            raise p_exc
        self._spawn_verify(
            'add_dependency', str(task_id), str(task_id),
            project_root, tag,
        )
        return p_val

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ):
        p_val, p_exc, _s_val, _s_exc = await self._dispatch_pair(
            'remove_dependency',
            (task_id, depends_on, project_root),
            {'tag': tag},
            normalize=_normalize_id_only,
        )
        if p_exc is not None:
            raise p_exc
        self._spawn_verify(
            'remove_dependency', str(task_id), str(task_id),
            project_root, tag,
        )
        return p_val

    async def validate_dependencies(
        self, project_root: str, tag: str | None = None,
    ):
        return await self._dispatch(
            'validate_dependencies', (project_root,), {'tag': tag},
            normalize=lambda v: 'ok' if 'success' in v.get('message', '').lower() else 'fail',
        )


# ── Normalisers ───────────────────────────────────────────────────────
#
# The comparator deliberately throws away noisy fields that we know will
# differ between Taskmaster (which records granular timestamps + telemetry)
# and SqliteTaskBackend (which doesn't). The bar is "do the two backends
# agree on the task semantics?" — not "do they emit byte-identical wire."
# Each normaliser returns a stable, comparable shape; mismatches in the
# stripped fields will not surface as divergences.


_VOLATILE_TASK_FIELDS = {'updatedAt', 'metadata'}


def _normalize_task(task: Any) -> Any:
    if not isinstance(task, dict):
        return task
    out = {k: v for k, v in task.items() if k not in _VOLATILE_TASK_FIELDS}
    if 'subtasks' in out and isinstance(out['subtasks'], list):
        out['subtasks'] = [_normalize_task(s) for s in out['subtasks']]
    if 'dependencies' in out and isinstance(out['dependencies'], list):
        out['dependencies'] = sorted(
            int(d) if isinstance(d, (int, str)) and str(d).isdigit() else d
            for d in out['dependencies']
        )
    # id may surface as int (get_task) or str (get_tasks); coerce so the
    # comparator doesn't fire on the asymmetry alone.
    if 'id' in out:
        try:
            out['id'] = int(out['id'])
        except (TypeError, ValueError):
            pass
    return out


def _normalize_tasks_tree(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    tasks = result.get('tasks', [])
    if not isinstance(tasks, list):
        return result
    return {'tasks': [_normalize_task(t) for t in tasks]}


def _normalize_set_status(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    tasks = result.get('tasks') or []
    flags = sorted(
        (str(t.get('taskId', '')), t.get('newStatus', '')) for t in tasks
    )
    return {'count': len(tasks), 'transitions': flags}


def _normalize_id_only(result: Any) -> Any:
    """Strip volatile fields; only meaningful field is "did it succeed"."""
    if not isinstance(result, dict):
        return result
    return {k: True for k in ('id', 'message') if k in result}


def _index_tasks_by_id(payload: Any) -> dict[Any, Any]:
    """Return ``{task_id: task_dict}`` for a normalised tasks-tree payload.
    Empty dict on shape mismatch."""
    if not isinstance(payload, dict):
        return {}
    tasks = payload.get('tasks', [])
    if not isinstance(tasks, list):
        return {}
    out: dict[Any, Any] = {}
    for task in tasks:
        if isinstance(task, dict) and 'id' in task:
            out[task['id']] = task
    return out


def _field_diff(p: Any, s: Any) -> dict[str, tuple[Any, Any]]:
    """Return ``{field: (primary, secondary)}`` for fields that differ.

    Recurses into ``subtasks`` (list of dicts) to surface subtask-level
    drift as ``subtasks[<id>].<field>=tm=X sql=Y`` entries instead of the
    whole subtasks list flagged as one opaque blob.
    """
    if not (isinstance(p, dict) and isinstance(s, dict)):
        return {'<value>': (p, s)} if p != s else {}
    fields = set(p) | set(s)
    diff: dict[str, tuple[Any, Any]] = {}
    for f in sorted(fields):
        pv = p.get(f, _MISSING)
        sv = s.get(f, _MISSING)
        if pv == sv:
            continue
        if (
            f == 'subtasks'
            and isinstance(pv, list) and isinstance(sv, list)
        ):
            sub_diff = _subtasks_diff(pv, sv)
            for sk, vv in sub_diff.items():
                diff[f'subtasks[{sk}]'] = vv
            continue
        diff[f] = (pv, sv)
    return diff


def _subtasks_diff(
    p_subs: list, s_subs: list,
) -> dict[str, tuple[Any, Any]]:
    """Per-subtask field-diff keyed by subtask id."""
    p_by_id = {st.get('id'): st for st in p_subs if isinstance(st, dict)}
    s_by_id = {st.get('id'): st for st in s_subs if isinstance(st, dict)}
    all_ids = sorted(set(p_by_id) | set(s_by_id), key=lambda x: str(x))
    out: dict[str, tuple[Any, Any]] = {}
    for sid in all_ids:
        ps = p_by_id.get(sid)
        ss = s_by_id.get(sid)
        if ps == ss:
            continue
        sub_fields = _field_diff(ps, ss) if (ps is not None and ss is not None) \
            else {'<presence>': (ps, ss)}
        for fk, fv in sub_fields.items():
            out[f'{sid}.{fk}'] = fv
    return out


def _format_field_diff(
    diff: dict[str, tuple[Any, Any]],
    *,
    primary_label: str,
    secondary_label: str,
) -> str:
    """Render a field diff as ``field: pl=X sl=Y`` segments, comma-joined."""
    parts: list[str] = []
    for field, (pv, sv) in diff.items():
        pv_repr = '<missing>' if pv is _MISSING else _summarize(pv, limit=80)
        sv_repr = '<missing>' if sv is _MISSING else _summarize(sv, limit=80)
        parts.append(f'{field}: {primary_label}={pv_repr} {secondary_label}={sv_repr}')
    return ', '.join(parts)


_MISSING = object()


def _strip_ids(task: Any) -> Any:
    """Recursively strip minted ``id`` fields. Used by the read-back
    verifier for ``add_task`` / ``add_subtask`` where the two backends
    legitimately mint different ids for the same logical task.
    """
    if not isinstance(task, dict):
        return task
    out = {k: v for k, v in task.items() if k != 'id'}
    if 'subtasks' in out and isinstance(out['subtasks'], list):
        out['subtasks'] = [_strip_ids(s) for s in out['subtasks']]
    return out


def _normalize_remove(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    return {
        'successful': result.get('successful', 0),
        'failed': result.get('failed', 0),
    }


__all__ = ['DualCompareBackend']
