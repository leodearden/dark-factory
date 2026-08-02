"""Shared systemd unit parser for the unit-parity checkers.

Holds no CLI and no policy of its own — just the parser
(``_join_continuations`` + ``parse_unit_directives``) that turns a unit file
into ``{section: {key: [value, ...]}}``. It exists because a SECOND consumer
appeared: ``scripts/check_dashboard_unit_parity.py`` wrote the parser, and
``scripts/check_orchestrator_unit_parity.py`` needs exactly it and nothing
else from that module.

Lifting rather than duplicating follows the precedent
``tests/scripts/systemd_unit_invariants.py`` set, which task 3408 moved out of
a single suite the moment a second consumer appeared, with the stated reason
that duplicating it into both is how the two copies drift until one silently
stops catching the defect. That reason binds with extra force here: a second
pasted copy of the parser inside the tooling built to catch silent drift would
reproduce, in the checkers themselves, precisely the failure they exist to
report. ``check_dashboard_unit_parity.py`` re-exports these two names so its
own module surface (and its test suite) stays intact, and
``tests/scripts/test_check_orchestrator_unit_parity.py`` asserts the re-export
is the SAME function object rather than a look-alike.

Import mechanics
----------------
Both consumers do a bare ``import systemd_unit_parity``, which resolves in
both contexts these scripts run in:

- **CLI** — python puts the executed script's own directory (``scripts/``) at
  ``sys.path[0]``, so a sibling module is importable by name.
- **pytest** — ``tests/scripts/conftest.py`` explicitly inserts ``scripts/``
  onto ``sys.path``. That insertion is load-bearing, not belt-and-braces:
  ``pyproject.toml`` sets ``--import-mode=importlib``, and under importlib
  mode pytest deliberately does NOT perform the ``sys.path`` mutation the
  prepend/append modes do.

Stdlib-only and import-free by design, so both checkers stay runnable under a
plain ``python3`` with no environment set up.
"""


def _join_continuations(text: str) -> list[str]:
    """Return *text*'s lines with backslash continuations joined into one line.

    While a line ends in ``\\``, the backslash is dropped and the NEXT line's
    stripped form is appended after a single space.  Mirrors ``_logical_exec_start``
    in tests/scripts/test_dashboard_service_template.py, generalised from "the
    ExecStart line" to "every line".

    Joining happens BEFORE comment classification, which matches systemd's own
    behaviour: a comment line ending in ``\\`` continues, and its continuation
    is part of the comment.  Both real dashboard units rely on this — the
    watchdog service quotes the old inline-shell ExecStart across two ``#``
    lines joined by a backslash, and classifying first would leave the second
    half of that quote looking like a directive.
    """
    joined: list[str] = []
    pending: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        piece = line if pending is None else f"{pending} {line.strip()}"
        if continued:
            pending = piece
            continue
        joined.append(piece)
        pending = None
    if pending is not None:
        # Trailing backslash on the final line — keep what we have rather than
        # silently dropping the directive.
        joined.append(pending)
    return joined


def parse_unit_directives(text: str) -> dict[str, dict[str, list[str]]]:
    """Parse a systemd unit into ``{section: {key: [value, ...]}}``.

    Classification rules are taken verbatim from the precedent's
    ``parse_unit_sections`` (scripts/check_fused_memory_unit_parity.py):

    - ``[X]`` opens section X.
    - Lines starting with ``#`` or ``;`` are comments — skipped.
    - Blank lines are skipped.
    - Lines before the first section header are DROPPED, not attributed.

    Two deliberate divergences from that precedent, each required here:

    1. **key → values LIST, not a flat line list.**  This checker compares
       directives BY KEY, which a flat list of lines cannot express, and it
       needs the several ``Environment=`` lines of a unit addressable as a
       group rather than as unrelated strings.
    2. **Backslash continuations are JOINED.**  ``parse_unit_sections``
       documents that it does not join them, which is harmless for its exact
       whole-line membership checks.  It is fatal here: the dashboard
       ExecStart spans four physical lines, so without joining every uvicorn
       flag task 3306 added lives on a line the parser never associates with
       ``ExecStart`` — the checker would report parity on a command it never
       actually read.

    Each surviving line is split on the FIRST ``=`` only, so
    ``Environment=A=1`` yields key ``Environment`` and value ``A=1``.  A line
    with no ``=`` is skipped (systemd has no valueless directives).
    """
    sections: dict[str, dict[str, list[str]]] = {}
    current: str | None = None
    for joined_line in _join_continuations(text):
        line = joined_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        sections[current].setdefault(key.strip(), []).append(value.strip())
    return sections
