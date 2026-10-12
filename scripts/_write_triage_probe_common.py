"""Ref-independent plumbing shared by the write_triage flip gate's two probes.

``scripts/check_write_triage_flip_preconditions.sh`` runs two probes against a
ref's extracted ``fused-memory/src`` tree: item 1's
``check_write_triage_attach_target.py`` and item 5's
``check_write_triage_attach_consumption.py``. Each asserts its own invariant.
What lives here is what neither invariant is about: importing the ref's module
out of the right tree, failing closed, and the stand-ins both drive the ref
with. One copy, because the two copies this replaced had already drifted
apart (task 5790).

IMPORT-RESOLUTION CONTRACT — read before moving this file.
This module MUST stay a flat sibling at ``scripts/_write_triage_probe_common.py``.
The probes resolve ``import _write_triage_probe_common`` solely because a
DIRECTLY-EXECUTED script places its own directory at ``sys.path[0]``: the gate
and every test run them by path. Never relocate this module into a package, and
never run a probe with ``python -m``, ``-P`` or ``-I``: each removes that entry,
and the probe dies before it can report a verdict. The gate's hermetic test
repos copy this file beside the probes
(``scripts/tests/test_check_write_triage_flip_preconditions.py::_make_gate_repo``).
In-process pytest resolution is handled by ``scripts/tests/conftest.py``.
"""
from __future__ import annotations

import contextlib
import enum
import importlib
import inspect
import logging
import re
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, overload

EXIT_OK = 0
EXIT_FAIL = 1

#: The logger both probes and this module report through.
PROBE_LOGGER_NAME = 'write_triage_probe'

logger = logging.getLogger(PROBE_LOGGER_NAME)


class Unverifiable(Exception):
    """The invariant could not be evaluated. Fails closed, never passes."""


_FAILING_CLOSED = '      Failing closed — an unverifiable invariant is not a satisfied one.'


def report_fail_closed(evaluate: Callable[[list[str]], int]) -> int:
    """Run *evaluate* into a report on stdout and return its exit code.

    *evaluate* appends the report's lines to the list it is handed and returns
    ``EXIT_OK`` or ``EXIT_FAIL``. Everything after argparse belongs here, so
    the two probes cannot drift on how a crash or a warning is reported.

    NOTHING ESCAPES. Any ``BaseException`` is an UNVERIFIABLE verdict: a probe
    executes the ref's own code, and a ``SystemExit(0)`` out of it once exited
    0 having printed nothing, which the gate read as PASS and authorised the
    flip on. ``KeyboardInterrupt`` too: an interrupted run is unverifiable,
    never anything else.

    WARNINGS go LAST, so they survive a tail-truncated report, and on STDOUT,
    the one channel the gate always reads. It quotes stderr only as a bounded
    tail where stdout carried no verdict, so propagation is off for the run: a
    copy on stderr would only compete for that tail.
    """
    out: list[str] = []
    with _collected_warnings() as warnings:
        try:
            rc = evaluate(out)
        except Unverifiable as exc:
            out.append(f'FAIL  UNVERIFIABLE: {exc}')
            out.append(_FAILING_CLOSED)
            rc = EXIT_FAIL
        except BaseException as exc:
            out.append(
                f'FAIL  UNVERIFIABLE: the probe raised {exc!r} while evaluating '
                'the invariant.',
            )
            out.append(_FAILING_CLOSED)
            out.extend('      ' + line for line in traceback.format_exc().splitlines())
            rc = EXIT_FAIL
    out.extend(f'WARN  {message}' for message in dict.fromkeys(warnings))
    sys.stdout.write('\n'.join(out) + '\n')
    return rc


class _Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _collected_warnings() -> Iterator[list[str]]:
    """The probe logger's WARNING+ messages, collected for the block only."""
    collector = _Collector()
    propagate = logger.propagate
    logger.addHandler(collector)
    logger.propagate = False
    try:
        yield collector.messages
    finally:
        logger.removeHandler(collector)
        logger.propagate = propagate


class Candidate:
    """A duck-typed ``MemoryResult`` stand-in.

    ``.id``/``.content``/``.metadata`` is all the ref's readers use:
    ``near_duplicate_guard._cosine_of`` takes the per-store cosine from
    ``metadata['store_score']``, and ``_canonical_id_of`` a child's parent from
    the metadata too. Constructing one avoids importing ``MemoryResult``, whose
    module pulls in third-party deps a bare extracted tree may lack.
    """

    __slots__ = ('content', 'id', 'metadata')

    def __init__(self, ident: str, content: str, metadata: dict[str, Any]) -> None:
        self.id = ident
        self.content = content
        self.metadata = metadata

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f'<candidate {self.id}>'


def is_inside(path: Path, root: Path) -> bool:
    """Is *path* under *root*, comparing PATH COMPONENTS rather than characters?"""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        # An unresolvable path is not demonstrably inside the root, and this
        # gate fails closed.
        return False


class ImportMode(enum.Enum):
    """What a module the ref's tree cannot yield costs the probe."""

    #: The invariant is about this module, so nothing can be measured without it.
    FATAL = enum.auto()
    #: Only one branch reads this module: it goes unsatisfied, and a WARN says why.
    NON_FATAL = enum.auto()


@overload
def import_from_tree(
    name: str,
    src_root: Path,
    *,
    mode: Literal[ImportMode.FATAL],
    extra_paths: Iterable[Path] = (),
) -> ModuleType: ...


@overload
def import_from_tree(
    name: str,
    src_root: Path,
    *,
    mode: Literal[ImportMode.NON_FATAL],
    extra_paths: Iterable[Path] = (),
) -> ModuleType | None: ...


def import_from_tree(
    name: str,
    src_root: Path,
    *,
    mode: ImportMode,
    extra_paths: Iterable[Path] = (),
) -> ModuleType | None:
    """Import *name* out of *src_root*, shadowing any installed copy.

    The assertion has to be made against the REF, not against whatever the
    probe's own interpreter has installed. *extra_paths* are first-party trees
    the module imports that *src_root* lacks (``shared/src``).

    The ref's escapes are caught, ``SystemExit`` included: an import guard
    calling ``sys.exit()`` is the ref's failure, not a verdict.
    ``KeyboardInterrupt`` is the operator's and propagates in either mode, to
    :func:`report_fail_closed`, which fails closed on it.
    """
    if mode is ImportMode.FATAL:
        return _import_contained(name, src_root, extra_paths)
    try:
        return _import_contained(name, src_root, extra_paths)
    except Unverifiable as exc:
        logger.warning('%s is not importable from %s: %s', name, src_root, exc)
        return None


def _import_contained(
    name: str, src_root: Path, extra_paths: Iterable[Path],
) -> ModuleType:
    if not src_root.is_dir():
        raise Unverifiable(f'--src-root is not a directory: {src_root}')
    _put_first([src_root, *extra_paths])
    try:
        module = importlib.import_module(name)
    except (Exception, SystemExit) as exc:
        raise Unverifiable(f'cannot import {name} from {src_root}: {exc!r}') from exc
    origin = getattr(module, '__file__', None)
    # PATH CONTAINMENT, not a string prefix: a sibling `<root>-installed`
    # extends the root's name without being inside it, and a module resolved
    # there is one the ref never shipped.
    if origin is None or not is_inside(Path(origin), src_root):
        raise Unverifiable(
            f'{name} resolved to {origin!r}, which is outside --src-root '
            f'{src_root} — the probe would be testing the wrong tree',
        )
    return module


def _put_first(entries: list[Path]) -> None:
    """*entries* at the front of ``sys.path`` in order, each exactly once."""
    front = [str(entry) for entry in entries]
    sys.path[:] = [*front, *(entry for entry in sys.path if entry not in front)]


def require(module: ModuleType, name: str) -> Any:
    """``module.<name>``, without which the probe cannot proceed."""
    attr = getattr(module, name, None)
    if attr is None:
        raise Unverifiable(f'{module.__name__} exposes no {name}')
    return attr


def parent_key(module: object) -> str:
    """The metadata key the ref's tree names a child's parent under."""
    key = getattr(module, 'PARENT_ID_KEY', None)
    return key if isinstance(key, str) and key else 'parent_id'


def usable_parameters(fn: Any) -> list[inspect.Parameter]:
    """*fn*'s parameters that can be supplied by position or by keyword.

    ``*args``/``**kwargs`` are dropped, so an index into this list names
    exactly one parameter a caller can supply.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return []
    kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    return [p for p in params if p.kind in kinds]


#: Parameter names that read as "this designates the attach target". A real
#: option (b) that names the target in a HEADER (``'ATTACH TARGET: ' +
#: attach_target_id``) is structurally indistinguishable from free-text
#: diagnostics that happen to be interpolated; the name is the only thing
#: separating them.
#:
#: DELIBERATELY NARROW, from a measured false PASS. This set used to include
#: ``canonical|winner|chosen|selected|primary`` — the BAND's vocabulary, for
#: things the band already tracks for other reasons. ``canonical`` was the
#: hole: ``build_judge_prompt(content, candidates, canonical_id=None)`` that
#: merely interpolates the band canonical into a header line is an ordinary
#: prompt-legibility change binding no verdict to any candidate, and it PASSED,
#: authorising the production ``write_triage.enabled`` flip on a change that
#: establishes nothing. A WARN on that pass mitigated nothing: on a PASS the
#: gate exits 0, files no escalation and forwards its stdout nowhere, so no
#: operator ever reads it.
#:
#: Only ``attach``/``target``/``candidate_id`` survive — spellings that can
#: name the attach target and little else. The cost of narrowing is bounded: a
#: parameter that genuinely MARKS the target (matches its argument against the
#: slate) is never judged by its name, so the name matters ONLY for
#: header-style renderings that interpolate the id verbatim.
TARGET_NAME_RE = re.compile(r'attach|target|candidate_id', re.IGNORECASE)


def names_a_target(param: inspect.Parameter) -> bool:
    """True when *param*'s NAME reads as designating the attach target."""
    return bool(TARGET_NAME_RE.search(param.name))
