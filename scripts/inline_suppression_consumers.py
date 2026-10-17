"""Which tool, if any, honours an inline suppression marker (PRD D8).

pyright reads both of its marker spellings wherever they sit; ruff reads a
``noqa`` exactly when the nearest ``pyproject.toml`` selects one of its codes;
a first-party checker reads the codes it defines; nothing reads the rest.
:attr:`Consumer.NONE` is the finding, not an absence of information: a marker
nothing reads protects nothing, so it is deleted rather than dispositioned.
Where this model cannot be sure, it errs in the direction
:data:`_WIDENING_KEYS` describes, and refuses rather than guessing on the
expensive side.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from inline_suppression_kinds import Kind, Site
from inline_suppression_refusal import InstrumentFailure


class Consumer(Enum):
    """Who honours a suppression.

    :attr:`NONE` is the finding D8 exists to surface, not an absence of
    information: a marker nothing reads is not protecting anything, so it is
    deleted rather than dispositioned.
    """

    PYRIGHT = 'pyright'
    RUFF = 'ruff'
    FIRST_PARTY = 'first-party'
    NONE = 'none'


#: Which consumer a kind resolves to WITHOUT consulting the code or any
#: config.  ``noqa`` is deliberately absent — it is the one kind whose answer
#: depends on both.  Totality over :class:`Kind` is enforced at the one read
#: site rather than assumed, so a sixth row added to ``KIND_SPECS`` with no
#: consumer row is a loud exit 2 instead of silently inheriting ``noqa``'s
#: behaviour.
_KIND_CONSUMERS: Mapping[Kind, Consumer] = MappingProxyType(
    {
        # pyright runs over every package as a declared gate, so both of its
        # marker spellings are read wherever they sit.
        Kind.TYPE_IGNORE: Consumer.PYRIGHT,
        Kind.PYRIGHT_IGNORE: Consumer.PYRIGHT,
        # Nothing in this repository reads either one today: no coverage gate
        # is configured, and bandit is not installed.  Both rows are
        # statements about TOOLS, and they change when a tool is adopted, not
        # when code changes.
        Kind.PRAGMA_NO_COVER: Consumer.NONE,
        Kind.NOSEC: Consumer.NONE,
    }
)

_LEADING_LETTERS = re.compile(r'^[A-Za-z]+')
_TRAILING_DIGITS = re.compile(r'[0-9]+$')


@dataclass(frozen=True)
class RuleCode:
    """A ruff rule code or selector, split at the boundary into its two parts.

    Attributes:
        linter: The leading run of letters — ``E``, ``BLE``, ``PLC``.  This is
            the linter ruff resolves the selector against.
        number: The trailing run of digits, which may be empty for a bare
            linter selector like ``B`` and for a first-party kebab-case code
            like ``bare-magicmock``.  An empty number is tolerated rather than
            raising: a code this split cannot read is simply one no selector
            can match, which is the correct answer for it.
    """

    linter: str
    number: str

    def render(self) -> str:
        """The code as the model READ it, for ``--json``.

        Deliberately the parsed form rather than the source string: the report
        exists so a reader can audit what the consumer model used, and a
        selector the model mis-read is exactly what they are looking for.
        """
        return f'{self.linter}{self.number}'


def parse_rule_code(raw: str) -> RuleCode:
    """Split *raw* into its leading letters and trailing digits."""
    linter = _LEADING_LETTERS.match(raw)
    number = _TRAILING_DIGITS.search(raw)
    return RuleCode(
        linter=linter.group() if linter is not None else '',
        number=number.group() if number is not None else '',
    )


def selects(selector: RuleCode, code: RuleCode) -> bool:
    """Whether *selector* selects *code*, ruff's way and never by string prefix.

    A selector is a (linter, code-prefix) PAIR: the linter must be EQUAL —
    ``B`` (flake8-bugbear) never reaches ``BLE`` (flake8-blind-except) — and
    the number is a prefix within it, so ``E4`` selects E402 and not E501.

    A naive ``code.startswith(selector)`` would report every ``BLE001`` marker
    as ruff-consumed under ``select = ["B"]``, defeating D8 for that population.

    KNOWN LIMIT, recorded rather than hidden: ruff's meta-prefix selectors
    (``PL`` covering PLC/PLE/PLR/PLW) read as unselected under this rule, which
    is the expensive direction of error :data:`_WIDENING_KEYS` describes.  No
    ``pyproject.toml`` in this repository uses one, and ``--json`` publishes
    the resolved select lists so a reader can see exactly what the model used.
    """
    return selector.linter == code.linter and code.number.startswith(selector.number)


@dataclass(frozen=True)
class RuffConfig:
    """The ``select``/``ignore`` pair one ``pyproject.toml`` declares.

    Attributes:
        path: The declaring file, repo-relative, so ``--json`` can name it.
        select: The selectors, parsed.
        ignore: The suppressors, parsed.  Read because it is the same tomllib
            call and is strictly more honest: ruff provably never emits an
            ignored code, so every ``noqa`` naming one is dead — which is
            exactly what D8 exists to say.
    """

    path: str
    select: tuple[RuleCode, ...]
    ignore: tuple[RuleCode, ...]

    def consumes(self, code: RuleCode) -> bool:
        """Whether ruff would emit *code* under this config."""
        return any(selects(selector, code) for selector in self.select) and not any(
            selects(suppressor, code) for suppressor in self.ignore
        )


#: Keys this model does not implement that could WIDEN the selected set.  The
#: split is by DIRECTION OF ERROR, which is the only thing that matters for a
#: gate: under-reading the selected set rejects a marker ruff genuinely
#: honours — a false red on a legitimate suppression, the expensive failure —
#: so the scanner refuses to guess.  Keys that only ever SUBTRACT
#: (`extend-ignore`, `per-file-ignores`) are tolerated and deliberately not
#: modelled, because over-reading only grandfathers a dead marker.
#:
#: `extend` is in the list because config INHERITANCE widens in exactly that
#: expensive direction: the inherited file carries its own `select` /
#: `extend-select`, which this model does not follow, so the local list read
#: here would be a subset of the rules ruff actually runs.  Completeness on the
#: expensive side is the whole purpose of the list, so a key belongs here before
#: the first pyproject uses it, not after.
#:
#: An ABSENT `select` belongs in this family for the same reason and is
#: enforced with it: ruff then applies its BUILT-IN default rule set, which is
#: wider than the nothing this model would otherwise infer and which drifts
#: with the ruff version.
_WIDENING_KEYS: tuple[str, ...] = ('extend-select', 'extend')


def _ruff_config_at(location: Path, *, relative: str) -> RuffConfig | None:
    """Read *location*'s ruff config, or ``None`` when it declares none.

    ``None`` covers two cases ruff itself treats identically: the file is not
    there, and the file carries no ``[tool.ruff]`` section at all.  Ruff SKIPS
    such a manifest and keeps looking upward, so a packaging-only
    ``pyproject.toml`` must not shadow the config above it — stated in this
    repository's own root ``pyproject.toml``.

    ``select`` AND ``ignore`` ARE RESOLVED INDEPENDENTLY, each from the last
    table that declares it, because that is what ruff does: a deprecated
    top-level lint key is honoured unless ``[tool.ruff.lint]`` declares that
    same key.  Taking both off the one table that happened to declare
    ``select`` would drop an ``ignore`` sitting beside it, and an ignored code
    is precisely the dead marker :class:`RuffConfig` reads ``ignore`` to name.
    """
    try:
        raw = tomllib.loads(location.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise InstrumentFailure(
            f'{relative}: could not be read as TOML -- {type(exc).__name__}: {exc}'
        ) from exc

    tool = raw.get('tool')
    ruff = tool.get('ruff') if isinstance(tool, dict) else None
    if not isinstance(ruff, dict):
        return None

    lint = ruff.get('lint')
    tables = [('tool.ruff', ruff)]
    if isinstance(lint, dict):
        tables.append(('tool.ruff.lint', lint))
    for name, table in tables:
        for key in _WIDENING_KEYS:
            if key in table:
                raise InstrumentFailure(
                    f'{relative}: [{name}] carries {key!r}, which this consumer '
                    'model does not implement and which could WIDEN the selected rule '
                    'set. Refusing to guess: under-reading the selected set would reject '
                    'a marker ruff genuinely honours.'
                )

    def declared(key: str) -> tuple[RuleCode, ...] | None:
        """*key*'s parsed list from the last table declaring it, else ``None``."""
        for _, table in reversed(tables):
            if key in table:
                return tuple(parse_rule_code(code) for code in table[key])
        return None

    select = declared('select')
    if select is None:
        raise InstrumentFailure(
            f'{relative}: has a [tool.ruff] section but declares no `select`, so ruff '
            "applies its BUILT-IN default rule set. This consumer model does not "
            'implement those defaults — they are wider than the nothing it would '
            'otherwise infer, and they drift with the ruff version. Declare `select` '
            'explicitly.'
        )

    return RuffConfig(path=relative, select=select, ignore=declared('ignore') or ())


#: The rule codes first-party checkers define, each mapped to the checker that
#: defines it.  These are not ruff codes at all — no selector can ever match a
#: kebab-case token — so they are resolved AHEAD of the ruff path.
#:
#: A SECOND COPY OF A PRIVATE CONSTANT, knowingly.  The source of truth is
#: `_RULE_A_CODE` / `_RULE_B_CODE` / `_RULE_C_CODE` in a stdlib-only script in
#: another package, which offers no public seam to import; a copy is
#: unavoidable, so heuristic 11 asks that its DRIFT be made loud rather than
#: that the copy be hidden.  The guard is behavioural and bidirectional:
#: `scripts/tests/test_inline_suppression_consumers.py` runs that checker over one
#: fixture per rule and asserts SET EQUALITY between the codes it emits and this
#: table.  Equality, not containment — a table that merely holds real codes can
#: still MISS one, which is the expensive direction (the scanner would tell an
#: author to delete a marker a live checker reads).
FIRST_PARTY_CODES: Mapping[str, str] = MappingProxyType(
    {
        'bare-magicmock': 'fused-memory/scripts/check_bare_magicmock_config.py',
        'bare-dataclass-double': 'fused-memory/scripts/check_bare_magicmock_config.py',
        'wall-clock-deadline': 'fused-memory/scripts/check_bare_magicmock_config.py',
    }
)


class ConsumerModel:
    """Resolves each site's consumer, reading the nearest ``pyproject.toml``.

    STATEFUL FOR ONE REASON ONLY — the memo.  Resolution walks from a site's
    directory up to the scan root, so without it every site would re-read and
    re-parse the same few manifests.
    The memo is keyed by DIRECTORY rather than by file, so one read serves every
    file in a package, and it is private to one scan.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._by_directory: dict[Path, RuffConfig | None] = {}

    def consumer_for(self, site: Site) -> Consumer:
        """Who honours *site*'s marker — :attr:`Consumer.NONE` if nobody does."""
        fixed = _KIND_CONSUMERS.get(site.kind)
        if fixed is not None:
            return fixed
        if site.kind is not Kind.NOQA:
            raise InstrumentFailure(
                f'{site.path}:{site.line}: kind {site.kind.value!r} is in KIND_SPECS but '
                'has no row in the consumer model, so this scanner cannot say whether '
                'anything reads it. Add its row beside the others.'
            )
        return self._noqa_consumer(site)

    def _noqa_consumer(self, site: Site) -> Consumer:
        """Resolve a ``noqa`` site against the nearest config.

        THE FIRST-PARTY TABLE IS CONSULTED FIRST, because those codes are not
        ruff codes at all and no config could ever answer for them.  A site is
        consumed if ANY of its codes is — the direction that over-reads, which
        :data:`_WIDENING_KEYS` records as the cheap direction of error.

        A CODELESS marker silences whatever ruff would have said, so it is
        consumed exactly when ruff has something to say at all — and a config
        selecting nothing therefore leaves it dead.
        """
        if any(code in FIRST_PARTY_CODES for code in site.codes):
            return Consumer.FIRST_PARTY
        config = self._nearest_config(Path(site.path).parent)
        if config is None:
            return Consumer.NONE
        if not site.codes:
            return Consumer.RUFF if config.select else Consumer.NONE
        if any(config.consumes(parse_rule_code(code)) for code in site.codes):
            return Consumer.RUFF
        return Consumer.NONE

    def resolved(self) -> tuple[RuffConfig, ...]:
        """Every config this model actually consulted, sorted by declaring path.

        THE REPORT'S AUDIT TRAIL, and the reason it is the model that publishes
        it: what a reader needs is not every ``pyproject.toml`` in the tree but
        the ones whose ``select``/``ignore`` decided an answer here — including,
        by their absence, the meta-prefix selectors :func:`selects` records as
        its KNOWN LIMIT.  Deduped by path, because the memo is keyed by directory and one
        manifest serves a whole subtree.
        """
        return tuple(
            sorted(
                {config.path: config for config in self._by_directory.values() if config}.values(),
                key=lambda config: config.path,
            )
        )

    def _nearest_config(self, directory: Path) -> RuffConfig | None:
        """The config governing *directory*, which is relative to the scan root.

        The walk stops AT the root and never climbs out of the tree under
        measurement; otherwise a scan of a fixture tree would silently read
        this repository's own configuration and report a stranger's answer.
        """
        if directory in self._by_directory:
            return self._by_directory[directory]
        relative = str(directory / 'pyproject.toml') if str(directory) != '.' else 'pyproject.toml'
        config = _ruff_config_at(self._root / relative, relative=relative)
        if config is None and str(directory) != '.':
            config = self._nearest_config(directory.parent)
        self._by_directory[directory] = config
        return config
