"""Shared pieces of the unit-parity checkers, and the driver they all run on.

- ``parse_unit_directives`` turns a unit file into
  ``{section: {key: [value, ...]}}``, joining backslash continuations first.
- ``environment_map`` reads a section's ``Environment=`` lines the way systemd
  does. The dashboard checker compares with it and scripts/render_systemd_unit.py
  preserves host-local values with it, so the two cannot disagree.
- ``find_dropins`` lists the ``.conf`` drop-ins systemd would merge over a unit.
- ``Drift`` is the record a checker's ``compare`` returns, and ``ABSENT`` is
  the value it renders for a side that does not declare the directive.
- ``ENV_POLICY`` is which ``Environment=`` variables are host-local,
  allowlisted or required, per unit family. The renderer and both checkers
  project it rather than keeping their own lists.

Driver
------
``run_parity(ParityRun(...), out=...)`` owns the per-unit loop every multi-unit
checker used to repeat: it classifies each ``UnitRef`` (vanished, not
installed, unreadable, compared), renders the report blocks in one fixed order
through ``emit``/``emit_block``, prints ``--print-verdicts`` lines when asked,
and returns a ``ParityExit`` — whose docstring is the one statement of what 0, 1
and 2 mean. A checker supplies only its ``compare``, its ``ReportText`` wording
deltas, and any ``Probe`` adding a verdict class of its own.

Constraints
-----------
- It imports no checker, and the driver lives HERE rather than in a new
  sibling: setup-host.sh's section harnesses copy only this file beside a real
  or stubbed checker (tests/scripts/setup_host_sections.py::write_checker), so
  whatever a checker needs from scripts/ must be in it.
- Stdlib-only, so every checker runs under a bare system ``python3`` with no
  environment set up.

Why the family is shaped this way, and what stays per-checker:
plans/review-all-s25-scripts-instruments-prd.md (finding fk-3042baa5983e).

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
"""

import dataclasses
import enum
import pathlib
import shlex
from collections.abc import Callable, Iterable, Sequence
from typing import Final, TextIO

# Rendered in place of a value on whichever side does not declare the
# directive at all.  Deliberately not '' or None: it appears verbatim in the
# operator's report, where "<absent>" reads unambiguously and an empty string
# would look like a directive set to nothing.
ABSENT: Final = "<absent>"


@dataclasses.dataclass(frozen=True)
class Drift:
    """One disagreement between the repo copy and the installed copy."""

    unit: str
    section: str
    key: str
    repo_value: str
    installed_value: str
    reason: str


def _join_continuations(text: str) -> list[str]:
    """Return *text*'s lines with backslash continuations joined into one line.

    While a line ends in ``\\``, the backslash is dropped and the NEXT line's
    stripped form is appended after a single space.  Mirrors
    tests/scripts/systemd_unit_invariants.py::logical_exec_start, generalised
    from "the ExecStart line" to "every line".

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

    Classification rules are shared with
    ``check_fused_memory_unit_parity.parse_unit_sections``; change both or
    neither:

    - ``[X]`` opens section X.
    - Lines starting with ``#`` or ``;`` are comments — skipped.
    - Blank lines are skipped.
    - Lines before the first section header are DROPPED, not attributed.

    Two deliberate divergences from it, each required here:

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
    ``render_systemd_unit.py`` PRESERVES the host-local ones when
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


# ---------------------------------------------------------------------------
# The Environment= policy
# ---------------------------------------------------------------------------

# The section every Environment= line ENV_POLICY governs lives in: the renderer
# preserves from it, the dashboard checker compares in it, the fused-memory
# checker requires in it.
SERVICE_SECTION: Final = "Service"

_POLICY_RATIONALE: Final = "plans/review-all-s25-scripts-instruments-prd.md D5"


class UnitFamily(enum.Enum):
    """A unit whose Environment= policy ENV_POLICY declares.

    The renderer reads every family's host_local column. Each family's checker
    reads one more column, and the two properties below say which.
    """

    DASHBOARD = "dashboard"
    FUSED_MEMORY = "fused-memory"

    @property
    def checker_compares_cross_copy(self) -> bool:
        """Its checker value-compares the copies, so EnvPolicy.compared_cross_copy applies."""
        return self is UnitFamily.DASHBOARD

    @property
    def checker_requires_values(self) -> bool:
        """Its checker requires Environment= lines verbatim, so EnvPolicy.required_value applies."""
        return self is UnitFamily.FUSED_MEMORY


@dataclasses.dataclass(frozen=True)
class EnvPolicy:
    """How one unit family treats one Environment= variable. Why each row is so: PRD D5.

    - host_local, read for every family: the renderer preserves the installed
      value across a re-render.
    - compared_cross_copy, read where ``checker_compares_cross_copy``: the
      dashboard checker value-compares the committed and installed copies;
      False puts the name on its DIVERGENCE_ALLOWLIST.
    - required_value, read where ``checker_requires_values``: the fused-memory
      checker requires ``Environment=NAME=VALUE`` verbatim and --fix appends
      it; None means not required.

    The defaults are the policy of a name with no row. A column its family's
    reader ignores must keep its default, so a setting that does nothing
    cannot be written.
    """

    unit_family: UnitFamily
    name: str
    _: dataclasses.KW_ONLY
    host_local: bool = False
    compared_cross_copy: bool = True
    required_value: str | None = None

    def __post_init__(self) -> None:
        violation = self._violation()
        if violation is not None:
            raise ValueError(
                f"EnvPolicy({self.unit_family.value}, {self.name!r}): {violation} "
                f"(see {_POLICY_RATIONALE})"
            )

    def _violation(self) -> str | None:
        if not self.name or "=" in self.name or any(c.isspace() for c in self.name):
            return (
                f"name={self.name!r} is not an Environment= variable name: it must be "
                "non-empty, with no '=' and no whitespace"
            )
        if self.required_value is not None and "\n" in self.required_value:
            return (
                f"required_value={self.required_value!r} spans lines, so it cannot be "
                "written as one Environment= line"
            )
        if not self.compared_cross_copy and not self.unit_family.checker_compares_cross_copy:
            return (
                "compared_cross_copy=False on a family whose checker compares no values "
                "across copies, so the setting would do nothing"
            )
        if self.required_value is not None and not self.unit_family.checker_requires_values:
            return (
                f"required_value={self.required_value!r} on a family whose checker "
                "requires no Environment= lines, so the setting would do nothing"
            )
        if self.host_local and self.required_value is not None:
            return (
                f"host_local=True with required_value={self.required_value!r} (B5): the "
                "fused-memory checker's --fix would append a last-wins line after the "
                f"last [{SERVICE_SECTION}] line, clobbering the value the renderer preserved"
            )
        if (
            self.host_local
            and self.compared_cross_copy
            and self.unit_family.checker_compares_cross_copy
        ):
            return (
                "host_local=True with compared_cross_copy=True: the dashboard checker "
                "would value-compare a value the renderer deliberately keeps divergent, "
                "so the gate would be red on every configured host"
            )
        return None


@dataclasses.dataclass(frozen=True)
class EnvPolicyTable:
    """The policy rows and their projections, one per consumer.

    A name with no row has the policy of ``EnvPolicy(family, name)``, a row that
    sets no column.
    """

    rows: tuple[EnvPolicy, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[UnitFamily, str]] = set()
        for row in self.rows:
            key = (row.unit_family, row.name)
            if key in seen:
                raise ValueError(
                    f"EnvPolicy({row.unit_family.value}, {row.name!r}) is declared twice: "
                    "one variable of one family has one policy, or the projections "
                    f"disagree about it (see {_POLICY_RATIONALE})"
                )
            seen.add(key)

    def host_local_names(self, family: UnitFamily) -> tuple[str, ...]:
        """The names the renderer preserves for *family*, in table order."""
        return tuple(r.name for r in self.rows if r.unit_family is family and r.host_local)

    def divergence_allowlist(self, family: UnitFamily) -> frozenset[str]:
        """The names whose value the dashboard checker does not compare across copies."""
        return frozenset(
            r.name for r in self.rows if r.unit_family is family and not r.compared_cross_copy
        )

    def required_directives(self, family: UnitFamily) -> tuple[str, ...]:
        """``Environment=NAME=VALUE`` for each row the fused-memory checker requires, in table order."""
        return tuple(
            f"Environment={r.name}={r.required_value}"
            for r in self.rows
            if r.unit_family is family and r.required_value is not None
        )


ENV_POLICY: Final = EnvPolicyTable(
    (
        EnvPolicy(
            UnitFamily.DASHBOARD,
            "DASHBOARD_KNOWN_PROJECT_ROOTS",
            host_local=True,
            compared_cross_copy=False,
        ),
        EnvPolicy(UnitFamily.DASHBOARD, "DASHBOARD_PROJECT_ROOT", compared_cross_copy=False),
        EnvPolicy(UnitFamily.FUSED_MEMORY, "MEM0_TELEMETRY", required_value="false"),
        EnvPolicy(
            UnitFamily.FUSED_MEMORY,
            "PATH",
            required_value=(
                "%h/.local/bin:%h/bin:%h/.cargo/bin:"
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
        ),
        EnvPolicy(UnitFamily.FUSED_MEMORY, "DASHBOARD_KNOWN_PROJECT_ROOTS", host_local=True),
    )
)


# ---------------------------------------------------------------------------
# The parity driver
# ---------------------------------------------------------------------------


class ParityExit(enum.IntEnum):
    """What a parity checker's exit status means: the one statement of the ladder.

    PARITY: at least one unit was compared, and every compared unit matched
    with no drop-in and no probe finding.
    FINDING: drift, OR something that could not be verified (a vanished
    committed copy, an unreadable file, a drop-in, a probe finding, or a run
    that compared nothing). "I could not verify" sits with "I found a
    difference", never with parity. Each cause gets its own report block,
    because they share one code but send the operator to different places.
    NOT_INSTALLED: the only thing found was units not installed on this host.
    Any FINDING outranks it, and the skipped units are still reported.
    setup-host.sh reads this code as a harmless skip, so if it outranked a
    finding, one uninstalled unit would hide that finding.

    A run that compared nothing is never PARITY. The dashboard checker once
    did exactly that: a mistyped ``--repo-root`` compared no units and still
    printed "parity — 3 unit(s) match". That is why the success line counts
    the units it COMPARED, not the units it selected.
    """

    PARITY = 0
    FINDING = 1
    NOT_INSTALLED = 2


# ``--print-verdicts`` kinds in RENDERING PRECEDENCE order, so ``drift,override``
# is a contract rather than an artifact of set iteration. Public because it IS
# the contract: setup-host.sh needs a case arm for every kind, and a test derives
# this tuple and asserts the installer handles each one. ``clean`` and ``absent``
# never appear together, or with any other kind. They are kept apart because an
# absent unit was never compared, so calling it clean would claim a check that
# never ran. A Probe's block tags are extra kinds, and may not reuse these names.
VERDICT_KINDS: Final[tuple[str, ...]] = (
    "clean",  # compared: no directive drift, no drop-in, no probe finding
    "absent",  # no installed copy, so nothing was compared
    "drift",  # at least one directive differs
    "override",  # a <unit>.d/*.conf drop-in is layered over the unit
    "vanished",  # no COMMITTED copy, so there is no source to install from
    "unreadable",  # a file exists but could not be opened or decoded
)

# The kinds that are not a finding: everything else lands on ParityExit.FINDING.
_BENIGN_KINDS: Final = frozenset({"clean", "absent"})


@dataclasses.dataclass(frozen=True)
class UnitRef:
    """One unit to compare; the driver opens nothing but these two paths and their drop-ins."""

    name: str
    repo_path: pathlib.Path
    installed_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class BlockText:
    """A report block: ``[tag] header`` (``{count}`` its only placeholder), entries, ``[tag] tail``."""

    tag: str
    header: str
    tail: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ReportText:
    """A checker's report wording. Only ``drift`` is required: every checker names its own remedy."""

    drift: BlockText
    skip: str = "installed unit not found: {path} (not installed on this host)"
    vanished: BlockText = BlockText(
        "vanished",
        "{count} committed unit(s) not found — nothing was verified for them:",
        (
            "The committed unit is this checker's source of truth. Check --repo-root, "
            "and whether the unit was renamed or moved (the paths live in UNITS in "
            "this script).",
        ),
    )
    override: BlockText = BlockText(
        "override",
        "{count} unit(s) have drop-in overrides — the unit files were compared, but "
        "the EFFECTIVE configuration was NOT verified:",
        (
            "systemd merges these over the unit at load time, so a directive set here "
            "silently wins over the committed value. Inspect with: systemctl --user cat "
            "<unit>  (and remove the drop-in, or move the setting into the committed unit).",
        ),
    )
    repo_label: str = "repo"


_UNREADABLE: Final = BlockText(
    "unreadable",
    "{count} unit file(s) could not be read — nothing was verified for them:",
    (
        "The file exists but could not be decoded as UTF-8 or opened. Check its "
        "permissions and encoding — this is not a directive diff, so there is "
        "nothing to propagate.",
    ),
)


@dataclasses.dataclass(frozen=True)
class ProbeFinding:
    """One entry a probe adds to one of its declared blocks."""

    block: BlockText
    entry: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ProbeOutcome:
    """What a probe established for one compared unit."""

    findings: tuple[ProbeFinding, ...] = ()
    # Drop-ins found beyond the installed directory's glob; merged into [override].
    dropins: tuple[pathlib.Path, ...] = ()
    # An affirmative line printed just before [ok], and only on a parity run.
    confirmation: str | None = None


@dataclasses.dataclass(frozen=True)
class Probe:
    """A checker-specific verdict class, run on each compared unit with its committed text."""

    blocks: tuple[BlockText, ...]
    run: Callable[[UnitRef, str], ProbeOutcome]

    def __post_init__(self) -> None:
        for block in self.blocks:
            if block.tag in VERDICT_KINDS:
                raise ValueError(
                    f"a Probe block is tagged [{block.tag}], a built-in verdict kind; a "
                    "probe's tags are extra verdict kinds, so a finding tagged 'clean' or "
                    "'absent' would never set the exit code to FINDING"
                )


@dataclasses.dataclass(frozen=True)
class ParityRun:
    """Everything a checker hands the driver."""

    tool: str
    units: Sequence[UnitRef]
    compare: Callable[[UnitRef, str, str], list[Drift]]
    report: ReportText
    probes: Sequence[Probe] = ()
    print_verdicts: bool = False


def emit(tool: str, message: str, *, out: TextIO) -> None:
    """Print *message* with ``[tool]`` prefixed onto EVERY physical line.

    setup-host.sh reads the tag's ABSENCE as "the checker did not run", which is
    sound only if no continuation line is ever left untagged.
    """
    for line in message.split("\n"):
        print(f"[{tool}] {line}", file=out)


def emit_block(
    tool: str,
    tag: str,
    header: str,
    entries: Iterable[str],
    tail: Iterable[str],
    *,
    out: TextIO,
) -> None:
    """Render one report block: ``[tag] header``, each entry, then ``[tag] <line>`` per tail line."""
    emit(tool, f"[{tag}] {header}", out=out)
    for entry in entries:
        emit(tool, entry, out=out)
    for line in tail:
        emit(tool, f"[{tag}] {line}", out=out)


@dataclasses.dataclass(frozen=True)
class _Vanished:
    unit: UnitRef


@dataclasses.dataclass(frozen=True)
class _NotInstalled:
    unit: UnitRef


@dataclasses.dataclass(frozen=True)
class _Unreadable:
    unit: UnitRef
    dropins: tuple[pathlib.Path, ...]
    path: pathlib.Path
    error: str


@dataclasses.dataclass(frozen=True)
class _Compared:
    unit: UnitRef
    dropins: tuple[pathlib.Path, ...]
    drifts: tuple[Drift, ...]
    findings: tuple[ProbeFinding, ...]
    confirmations: tuple[str, ...]


_Outcome = _Vanished | _NotInstalled | _Unreadable | _Compared


def _same_file(path: pathlib.Path) -> str:
    """A spelling-independent identity: a symlinked $HOME must not list one drop-in twice."""
    return str(path.resolve())


def _merge_dropins(
    globbed: Sequence[pathlib.Path], reported: Iterable[pathlib.Path]
) -> tuple[pathlib.Path, ...]:
    """The glob's drop-ins, then each probe-reported one not already listed, as its source spelled it."""
    merged = list(globbed)
    seen = {_same_file(path) for path in merged}
    for path in reported:
        if _same_file(path) not in seen:
            seen.add(_same_file(path))
            merged.append(path)
    return tuple(merged)


def _run_probe(probe: Probe, unit: UnitRef, repo_text: str) -> ProbeOutcome:
    outcome = probe.run(unit, repo_text)
    for finding in outcome.findings:
        if finding.block not in probe.blocks:
            raise ValueError(
                f"a probe finding for {unit.name} targets block [{finding.block.tag}], "
                "which its Probe does not declare in blocks — an undeclared block is "
                "never rendered, so the finding would vanish from the report"
            )
    return outcome


def _classify(run: ParityRun, unit: UnitRef) -> _Outcome:
    if not unit.repo_path.is_file():
        return _Vanished(unit)
    if not unit.installed_path.is_file():
        return _NotInstalled(unit)
    # Consulted even at file parity: a drop-in is merged OVER a matching unit file.
    dropins = tuple(find_dropins(unit.installed_path.parent, unit.installed_path.name))
    texts: list[str] = []
    for path in (unit.repo_path, unit.installed_path):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            return _Unreadable(unit, dropins, path, str(exc))
    repo_text, installed_text = texts
    drifts = tuple(run.compare(unit, repo_text, installed_text))
    outcomes = [_run_probe(probe, unit, repo_text) for probe in run.probes]
    return _Compared(
        unit,
        _merge_dropins(dropins, (path for o in outcomes for path in o.dropins)),
        drifts,
        tuple(finding for o in outcomes for finding in o.findings),
        tuple(o.confirmation for o in outcomes if o.confirmation is not None),
    )


def _verdict_kinds(outcome: _Outcome) -> frozenset[str]:
    if isinstance(outcome, _Vanished):
        return frozenset({"vanished"})
    if isinstance(outcome, _NotInstalled):
        return frozenset({"absent"})
    kinds = {"override"} if outcome.dropins else set()
    if isinstance(outcome, _Unreadable):
        return frozenset({*kinds, "unreadable"})
    kinds.update(finding.block.tag for finding in outcome.findings)
    if outcome.drifts:
        kinds.add("drift")
    return frozenset(kinds or {"clean"})


def _render_kinds(kinds: frozenset[str]) -> str:
    """VERDICT_KINDS order; a kind outside it (a probe's block tag) is appended, never dropped."""
    ordered = [kind for kind in VERDICT_KINDS if kind in kinds]
    return ",".join(ordered + sorted(kinds.difference(VERDICT_KINDS)))


def _drift_entry(drift: Drift, unit: UnitRef, repo_label: str) -> tuple[str, ...]:
    return (
        f"  {drift.unit} [{drift.section}] {drift.key}",
        f"      {drift.reason}",
        f"      {repo_label.ljust(len('installed'))} {unit.repo_path}: {drift.repo_value}",
        f"      installed {unit.installed_path}: {drift.installed_value}",
    )


def _emit_bucket(
    tool: str, text: BlockText, items: Sequence[tuple[str, ...]], *, out: TextIO
) -> None:
    """Emit *text*'s block over *items* (each one counted item's entry lines), if there are any."""
    if items:
        header = text.header.format(count=len(items))
        emit_block(tool, text.tag, header, [line for item in items for line in item], text.tail, out=out)


def _declared_blocks(probes: Sequence[Probe]) -> list[BlockText]:
    return list(dict.fromkeys(block for probe in probes for block in probe.blocks))


def _emit_report(run: ParityRun, outcomes: Sequence[_Outcome], out: TextIO) -> None:
    """The human report, in the fixed order skip, vanished, unreadable, override, drift, probes."""
    report, tool = run.report, run.tool
    for o in outcomes:
        if isinstance(o, _NotInstalled):
            skip = report.skip.format(name=o.unit.name, path=o.unit.installed_path)
            emit(tool, f"[skip] {skip}", out=out)
    compared = [o for o in outcomes if isinstance(o, _Compared)]
    vanished = [
        (f"  {o.unit.name}: expected committed copy at {o.unit.repo_path}",)
        for o in outcomes
        if isinstance(o, _Vanished)
    ]
    unreadable = [
        (f"  {o.unit.name}: {o.path}: {o.error}",) for o in outcomes if isinstance(o, _Unreadable)
    ]
    overridden = [
        tuple(f"  {o.unit.name}: {path}" for path in o.dropins)
        for o in outcomes
        if isinstance(o, _Unreadable | _Compared) and o.dropins
    ]
    drifts = [_drift_entry(d, o.unit, report.repo_label) for o in compared for d in o.drifts]
    _emit_bucket(tool, report.vanished, vanished, out=out)
    _emit_bucket(tool, _UNREADABLE, unreadable, out=out)
    _emit_bucket(tool, report.override, overridden, out=out)
    _emit_bucket(tool, report.drift, drifts, out=out)
    for block in _declared_blocks(run.probes):
        found = [f.entry for o in compared for f in o.findings if f.block == block]
        _emit_bucket(tool, block, found, out=out)


def _conclude(kinds: Sequence[frozenset[str]], compared: int) -> ParityExit:
    """The ladder: each unit's verdict kinds and the number of units compared decide the code."""
    if any(unit_kinds - _BENIGN_KINDS for unit_kinds in kinds):
        return ParityExit.FINDING
    if any("absent" in unit_kinds for unit_kinds in kinds):
        return ParityExit.NOT_INSTALLED
    return ParityExit.PARITY if compared else ParityExit.FINDING


def _emit_closing(
    tool: str, outcomes: Sequence[_Outcome], code: ParityExit, *, out: TextIO
) -> None:
    """On PARITY, each confirmation and then [ok]. On an empty run, [error], since no block explains it."""
    compared = [o for o in outcomes if isinstance(o, _Compared)]
    if code is ParityExit.PARITY:
        for confirmation in (line for o in compared for line in o.confirmations):
            emit(tool, confirmation, out=out)
        emit(tool, f"[ok] parity — {len(compared)} unit(s) match their committed copies.", out=out)
    elif not outcomes:
        emit(tool, "[error] no units were compared — nothing was verified.", out=out)


def run_parity(run: ParityRun, *, out: TextIO) -> ParityExit:
    """Compare every unit, print the report (verdict lines first, if asked), return the exit code.

    Each unit's verdict kinds feed BOTH the verdict lines and the ladder, so the
    machine channel and the exit status can never describe different runs.
    """
    outcomes = [_classify(run, unit) for unit in run.units]
    kinds = [_verdict_kinds(outcome) for outcome in outcomes]
    code = _conclude(kinds, sum(isinstance(o, _Compared) for o in outcomes))
    if run.print_verdicts:
        verdicts = sorted(zip((o.unit.name for o in outcomes), kinds, strict=True))
        for name, unit_kinds in verdicts:
            emit(run.tool, f"verdict {name} {_render_kinds(unit_kinds)}", out=out)
    _emit_report(run, outcomes, out)
    _emit_closing(run.tool, outcomes, code, out=out)
    return code
