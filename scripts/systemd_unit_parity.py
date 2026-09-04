"""Shared helpers for the unit-parity checkers.

Holds no CLI and no policy of its own — just the pieces the unit tooling
needs: the parser (``_join_continuations`` + ``parse_unit_directives``) that
turns a unit file into ``{section: {key: [value, ...]}}``; ``environment_map``,
which reads a section's ``Environment=`` lines the way systemd does; and
``find_dropins``, which reports the ``.conf`` files systemd would layer over a
unit.

The parser landed first, because a SECOND consumer appeared:
``scripts/check_dashboard_unit_parity.py`` wrote it, and
``scripts/check_orchestrator_unit_parity.py`` needs exactly it and nothing
else from that module.

``find_dropins`` is the SECOND lift, and it arrived here in worse shape than
the parser did: it was ALREADY duplicated, code-identical, in both of those
checkers (only the docstrings' example directives differed). So this move
collapses an existing fork rather than pre-empting a hypothetical one. What
forced it was a THIRD consumer, ``scripts/check_lms_unit_parity.py``, whose
whole subject is a drop-in that survives reinstallation — writing a third
pasted copy to detect silent overrides would have been the same silent
duplication one level up.

``environment_map`` is the THIRD lift, and its second consumer is NOT a fourth
checker: it is ``scripts/render_dashboard_unit.py``, the renderer setup-host.sh
now uses in place of a truncating ``sed`` redirect to install
dark-factory-dashboard.service. That renderer must PRESERVE this host's
host-local ``Environment=DASHBOARD_KNOWN_PROJECT_ROOTS`` value across a
re-render, which means reading the installed unit's ``Environment=`` map — and
it must read it through the SAME code ``check_dashboard_unit_parity.py``
compares those variables with, or a value the checker can see stops being a
value the installer can preserve.

It could not get that by importing the checker, and the obstacle is measured
rather than stylistic: ``tests/scripts/test_check_dashboard_unit_parity.py``'s
section-8 harness builds a tmp repo in which ``write_checker(body=...)``
REPLACES check_dashboard_unit_parity.py with an argparse-usage-error stub (and,
with ``with_checker=False``, omits it and its siblings entirely), precisely to
test that the install still happens when the parity gate did not run. A
renderer importing that module would ImportError under exactly those two tests,
turning them red for a reason unrelated to what they assert. So the shared
dependency went DOWN into this module rather than sideways into the checker.

Lifting rather than duplicating follows the precedent
``tests/scripts/systemd_unit_invariants.py`` set, which task 3408 moved out of
a single suite the moment a second consumer appeared, with the stated reason
that duplicating it into both is how the two copies drift until one silently
stops catching the defect. That reason binds with extra force here: a second
pasted copy of the parser inside the tooling built to catch silent drift would
reproduce, in the checkers themselves, precisely the failure they exist to
report. ``check_dashboard_unit_parity.py`` re-exports these three names so its
own module surface (and its test suite) stays intact, and
``tests/scripts/test_check_orchestrator_unit_parity.py`` (for the parser and
``find_dropins``) and ``tests/scripts/test_render_dashboard_unit.py`` (for
``environment_map``) each assert the re-export is the SAME function object
rather than a look-alike.

What is NOT here yet: the COMPARISON core (read this before assuming it is)
--------------------------------------------------------------------------
Three lifts have happened, and this module deliberately does not yet read as
the home for everything shared.  Still duplicated across the consumers, and NOT
because nobody noticed:

- ``Drift`` — the frozen six-field record — is code-identical in all three
  checkers (``check_dashboard_unit_parity.py``, ``check_orchestrator_unit_parity.py``,
  ``check_lms_unit_parity.py``).
- ``UnitSpec``'s ``compared``/``present_only`` core and ``compare_unit`` are
  duplicated between the dashboard and lms checkers only.  The orchestrator
  checker deliberately has NEITHER: it compares by full symmetric equality over
  the union of sections and keys, with no curated registry, and its own
  docstring argues for that.

``environment_map`` is deliberately NOT a counter-example to this list. It moved
because it had a second consumer that could not reach it any other way (see
above), and — unlike ``compare_unit`` — it is pure parsing with no policy and no
report text, so no consumer's output changed. It was also a FIRST lift rather
than a fork collapse: ``check_lms_unit_parity.py`` never had a copy of it, which
was checked before the move rather than assumed.

What stopped the COMPARISON-CORE lift is not tidiness but a real decision: the
two copies already RENDER differently — the dashboard and orchestrator share a
``_render`` helper (single-value shortcut, then ``" | ".join``), while the lms
checker joins with ``", "`` inline — so one shared ``compare_unit`` changes the
report text of
whichever side loses, and both sides' suites assert on report content.  The
dashboard's ``UnitSpec`` also carries four extra fields and a ``__post_init__``
validator, so the shared part is a CORE the dashboard extends, not the class.
That is a design step, and task 3775 — which wrote the lms copy — held only an
amendment-pass mandate for focused edits.  It is filed as a follow-up (ticket
``tkt_0RSJTASQDQAFHYWCK5VW8BFGC3``, "Lift Drift / UnitSpec / compare_unit into
scripts/systemd_unit_parity.py"), which carries the obstacle above so the work
starts from it.

Recorded here rather than left implicit because the alternative is worse than
the duplication: a module that looks like the shared home while a third pasted
copy of the comparison sits in the consumers invites the next author to assume
the lift is complete and paste a fourth.

Why ``check_fused_memory_unit_parity.parse_unit_sections`` is NOT here
----------------------------------------------------------------------
That third parser stays where it is, deliberately.  It returns
``{section: [line, ...]}`` — whole non-comment LINES, unsplit and with
continuations left unjoined — because its checker asks whole-line membership
questions (``"Environment=MEM0_TELEMETRY=false" in sections["Service"]``) and
synthesizes those same literal lines in its ``--fix`` path.  It is not a
degenerate case of ``parse_unit_directives``: reconstructing a line from a
``{key: [values]}`` mapping cannot reproduce the original spacing that
membership check and that rewriter both depend on.  So folding it in would
mean changing its checker's semantics, not just its imports — a different
change, in a module this task holds no lock on.  What is shared below is the
CLASSIFICATION (section headers, ``#``/``;`` comments, blanks, pre-header
lines), which ``parse_unit_directives`` took from it verbatim and documents as
such.  If that classification is ever changed here it must be changed there
too, and vice versa: two parsers is the tolerated cost of two different return
types, not license for the rules to diverge.

Import mechanics
----------------
Every consumer does a bare ``import systemd_unit_parity``, which resolves in
both contexts these scripts run in:

- **CLI** — python puts the executed script's own directory (``scripts/``) at
  ``sys.path[0]``, so a sibling module is importable by name.
- **pytest** — ``tests/scripts/conftest.py`` explicitly inserts ``scripts/``
  onto ``sys.path``. That insertion is load-bearing, not belt-and-braces:
  ``pyproject.toml`` sets ``--import-mode=importlib``, and under importlib
  mode pytest deliberately does NOT perform the ``sys.path`` mutation the
  prepend/append modes do.

Stdlib-only by design, so every consumer stays runnable under a plain
``python3`` with no environment set up. ``pathlib`` (required by
``find_dropins``) and ``shlex`` (required by ``environment_map`` — see its
docstring for why a naive split is wrong) are the module's ONLY imports, and it
should stay that way: anything needing more than the stdlib's most boring corner
does not belong here.
"""

import pathlib
import shlex


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


def environment_map(
    directives: dict[str, dict[str, list[str]]],
    section: str,
) -> dict[str, str]:
    """Return ``{VAR: value}`` for every ``Environment=`` line in *section*.

    Each LINE is split the way systemd reads it, not the way the simplest
    parser would: ``Environment=`` accepts SEVERAL space-separated assignments
    on one line, and values may be quoted.  ``shlex.split`` handles both, so
    these three spellings all yield ``{A: 1, B: 2}``::

        Environment=A=1 B=2
        Environment="A=1" "B=2"
        Environment=A=1
        Environment=B=2

    Before this used shlex, ``Environment=A=1 B=2`` parsed as the single
    variable ``A`` with value ``1 B=2``, and ``Environment="A=1"`` produced a
    variable literally named ``"A``.  Compared against a copy using the
    one-per-line spelling, either mis-parse invented drift out of a pure
    reformat — and an unexplainable red on a warn-only gate is exactly how the
    gate loses the credibility it exists to have.  Neither committed unit uses
    those forms today; the point is that reformatting one must not fire.

    Each assignment is then split on its FIRST ``=`` — ``A=b=c`` sets A to
    ``b=c``.  A later occurrence of the same variable wins, matching systemd,
    which applies the directives in file order.  A token carrying no ``=`` at
    all is skipped rather than guessed at.

    TWO CONSUMERS, and they must agree. ``check_dashboard_unit_parity.py``
    COMPARES these variables across the committed and installed copies;
    ``render_dashboard_unit.py`` PRESERVES the host-local ones when
    setup-host.sh re-renders the installed unit. A value the checker can see
    has to be exactly a value the installer can preserve, so a second reader
    with its own idea of what an ``Environment=`` line means would let the
    installer silently drop a variable the gate then reported as fine.
    """
    env: dict[str, str] = {}
    for line in directives.get(section, {}).get("Environment", []):
        try:
            tokens = shlex.split(line)
        except ValueError:
            # Unbalanced quotes: systemd would reject the line too. Fall back
            # to the whole line as one token so a malformed value still shows
            # up as a variable rather than vanishing into silent parity.
            tokens = [line]
        for assignment in tokens:
            name, sep, value = assignment.partition("=")
            if not sep:
                continue
            env[name.strip()] = value
    return env

def find_dropins(installed_dir: pathlib.Path, unit_name: str) -> list[pathlib.Path]:
    """Return the ``.conf`` drop-ins systemd would layer over *unit_name*.

    ``systemctl --user edit`` does not modify the unit file; it writes
    ``<unit>.d/override.conf`` beside it, and systemd merges that over the
    unit at load time.  Reading only ``<installed-dir>/<unit>`` is therefore
    blind to it: a drop-in redeclaring a single directive leaves every
    compared directive matching while the RUNNING configuration is not the
    committed one — the precise claim these gates exist to make checkable.

    Not hypothetical on this host: ``~/.config/systemd/user/
    orchestrator-reify.service.d/`` exists, so the mechanism is already in
    live use here.

    Returns the sorted ``.conf`` files, or ``[]`` when the directory is absent
    or holds none (systemd ignores non-``.conf`` files there, so this counts
    exactly what would take effect — counting a stray ``override.conf.bak``
    would report an override that has no effect at all).

    Consulted EVEN WHEN the unit file itself is at parity; see each consumer's
    main().
    """
    dropin_dir = installed_dir / f"{unit_name}.d"
    if not dropin_dir.is_dir():
        return []
    return sorted(p for p in dropin_dir.glob("*.conf") if p.is_file())
