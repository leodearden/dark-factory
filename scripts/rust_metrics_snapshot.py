#!/usr/bin/env python3
"""Whole-repo quality metrics snapshot for a Cargo workspace (Rust + TypeScript + Python + shell).

The Rust shape of /review-all's Phase 1 instrument (SKILL.md Phase 1), for a project
that is not a Python uv-workspace and so cannot run scripts/quality_metrics_snapshot.py.
First written for reify's first run (review-all-reify-20261010); measured there:
2,901 files / 1.99M lines in 5m25s, 195 files at or over 2,000 lines. That run's
import-graph figures predate the module pre-registration in _cargo_modules() and
under-read every `use` of a file that sorts after its importer.

A report, never a gate: measures and the two heuristic-14 marks only. Mirrors the
shape of dark-factory's scripts/quality_metrics_snapshot.py (file records + import
graph) with Rust-aware instruments:
  - raw/prose counts: tokei (comments / (code+comments) = prose_ratio)
  - per-function complexity: lizard cyclomatic complexity (CCN), read as the pair
    (max per function, file total) -- NOT the quality doc's cognitive complexity,
    which has no working Rust binary on the host (rust-code-analysis-cli does not
    build); the report's metrics section says so
  - import graph: `use` statements resolved to the longest known module prefix;
    reach-back = a child module importing a strict ancestor module; deferred = `use`
    inside a fn body; cycles = SCCs over module-level edges; closes_cycle = a deferred
    edge whose target reaches the importer
  - re-exports: `pub use` leaf names
  - tests reaching internals (Rust): names of `pub(crate)`/`pub(super)`/`#[doc(hidden)] pub`
    items of a src file that appear as tokens in a tests-kind file of the same crate
    (excluding the file's own inline test module); (TS) vi.mock/vi.spyOn targets;
    (Python) patch targets
  - impl-block spread: per type, the src files holding an `impl Type` block -- Rust
    couples files through impl blocks, which the use-graph cannot see
  - fix density: fix-grep'd non-merge commits touching the file since the rules' date

There is no --diff yet: the summary says there was no previous snapshot.

The project's rules (domain excludes, area rules, inferred mappings, test-file pattern,
lizard excludes, fix-density start date) are a TOML file given by --rules, from the
project overlay (references/project-overlay.md, Path -> area map and Metrics
instruments); reify's is scripts/rust_metrics_rules/reify.toml. The snapshot records
them in params.rules so the report can cite them.

Needs tokei, cargo and git on PATH; lizard runs as `python -m lizard` under the measuring
interpreter, which records its version.

    --rules RULES.toml --run-id ID --out FILE --root WORKTREE  measure HEAD into one JSON
                                                               file, the summary on stdout
    --rules RULES.toml --params                                the params a measurement here records

Exit 0 when done; exit 2 on an instrument failure, with the cause named on stderr.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import importlib.metadata
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import _checkout as _checkout
import source_measures

from shared import safe_io

SCHEMA_VERSION = 2
INSTRUMENT = 'rust-metrics-snapshot'

H14_SOFT = 1500
H14_ALARM = 2000

SRC_EXT = {'.rs', '.ts', '.tsx', '.py', '.sh', '.mjs'}


@dataclasses.dataclass(frozen=True)
class AreaRule:
    pattern: re.Pattern[str]
    area: str


@dataclasses.dataclass(frozen=True)
class RustRules:
    """One project's measurement rules, as its --rules TOML file states them."""

    exclude: re.Pattern[str]
    test: re.Pattern[str]
    areas: tuple[AreaRule, ...]
    inferred: Mapping[str, str]
    lizard_excludes: tuple[str, ...]
    fix_density_since: datetime.date

    def area_of(self, path: str) -> str | None:
        """The briefing area of the first area rule matching *path* from its start."""
        return next((rule.area for rule in self.areas if rule.pattern.match(path)), None)

    def excluded(self, path: str) -> bool:
        return self.exclude.match(path) is not None

    def is_test(self, path: str) -> bool:
        return self.test.search(path) is not None

    def to_json(self) -> dict[str, Any]:
        """The rules in the TOML file's own keys, each pattern as its source."""
        return {
            'exclude': self.exclude.pattern,
            'test': self.test.pattern,
            'area': [{'pattern': rule.pattern.pattern, 'area': rule.area} for rule in self.areas],
            'inferred': dict(self.inferred),
            'lizard_excludes': list(self.lizard_excludes),
            'fix_density_since': self.fix_density_since.isoformat(),
        }


_RULE_KEYS = ('exclude', 'test', 'area', 'inferred', 'lizard_excludes', 'fix_density_since')
_AREA_KEYS = ('pattern', 'area')


def _refused(path: Path, key: str, problem: str) -> source_measures.MetricsError:
    return source_measures.MetricsError(f'{path}: rules key {key} {problem}')


def _string(path: Path, key: str, value: object) -> str:
    if not isinstance(value, str):
        raise _refused(path, key, f'must be a string, not {type(value).__name__}')
    return value


def _regex(path: Path, key: str, value: object) -> re.Pattern[str]:
    try:
        return re.compile(_string(path, key, value))
    except re.error as exc:
        raise _refused(path, key, f'is not a regex that compiles: {exc}') from exc


def _table(path: Path, key: str, value: object, keys: tuple[str, ...]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _refused(path, key, f'must be a table, not {type(value).__name__}')
    missing = [name for name in keys if name not in value]
    if missing:
        raise _refused(path, f'{key}.{missing[0]}' if key else missing[0], 'is missing')
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise _refused(path, f'{key}.{unknown[0]}' if key else unknown[0], f'is not one of {", ".join(keys)}')
    return value


def _area_rules(path: Path, value: object) -> tuple[AreaRule, ...]:
    if not isinstance(value, list):
        raise _refused(path, 'area', f'must be an array of tables, not {type(value).__name__}')
    rules = []
    for index, entry in enumerate(value):
        key = f'area[{index}]'
        fields = _table(path, key, entry, _AREA_KEYS)
        rules.append(
            AreaRule(_regex(path, f'{key}.pattern', fields['pattern']), _string(path, f'{key}.area', fields['area']))
        )
    return tuple(rules)


def _inferred(path: Path, value: object) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise _refused(path, 'inferred', f'must be a table, not {type(value).__name__}')
    return MappingProxyType({name: _string(path, f'inferred.{name}', note) for name, note in value.items()})


def _strings(path: Path, key: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _refused(path, key, f'must be an array of strings, not {type(value).__name__}')
    return tuple(_string(path, f'{key}[{index}]', item) for index, item in enumerate(value))


def _date(path: Path, key: str, value: object) -> datetime.date:
    if not isinstance(value, datetime.date) or isinstance(value, datetime.datetime):
        raise _refused(path, key, f'must be a TOML local date (YYYY-MM-DD), not {value!r}')
    return value


def load_rules(path: Path) -> RustRules:
    """The rules file at *path*, every key checked; else a MetricsError naming the file and the key."""
    try:
        parsed = tomllib.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError) as exc:
        raise source_measures.MetricsError(f'{path}: could not be read -- {exc.__class__.__name__}: {exc}') from exc
    except tomllib.TOMLDecodeError as exc:
        raise source_measures.MetricsError(f'{path}: is not valid TOML -- {exc}') from exc
    table = _table(path, '', parsed, _RULE_KEYS)
    return RustRules(
        exclude=_regex(path, 'exclude', table['exclude']),
        test=_regex(path, 'test', table['test']),
        areas=_area_rules(path, table['area']),
        inferred=_inferred(path, table['inferred']),
        lizard_excludes=_strings(path, 'lizard_excludes', table['lizard_excludes']),
        fix_density_since=_date(path, 'fix_density_since', table['fix_density_since']),
    )


def _run(command: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """*command* run to completion; else a MetricsError naming it, its exit code and its stderr."""
    where = f' in {cwd}' if cwd is not None else ''
    try:
        completed = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise source_measures.MetricsError(
            f'could not run `{shlex.join(command)}`{where}: {exc.__class__.__name__}: {exc}'
        ) from exc
    if completed.returncode != 0:
        raise source_measures.MetricsError(
            f'`{shlex.join(command)}` failed{where} (exit {completed.returncode}): '
            f'{completed.stderr.strip() or "no stderr"}'
        )
    return completed


_TOKEI_INSTALL = 'cargo install --locked tokei'


def _tokei_version() -> str:
    try:
        printed = _run(['tokei', '--version']).stdout
    except source_measures.MetricsError as exc:
        raise source_measures.MetricsError(f'{exc} -- tokei counts the prose share; install it with `{_TOKEI_INSTALL}`') from exc
    found = re.match(r'tokei (\S+)', printed)
    if found is None:
        raise source_measures.MetricsError(
            f'`tokei --version` printed {printed.strip()!r}, not `tokei <version>`; reinstall it with `{_TOKEI_INSTALL}`'
        )
    return found.group(1)


def _lizard_version() -> str:
    try:
        return importlib.metadata.version('lizard')
    except importlib.metadata.PackageNotFoundError as exc:
        raise source_measures.MetricsError(
            f'lizard is not installed for {sys.executable}, and it measures per-function complexity; '
            'run `uv sync --all-packages` in the factory checkout'
        ) from exc


@dataclasses.dataclass(frozen=True)
class _Params:
    """A measurement's params: with as_of_sha, its identity (plans/quality-dashboard-prd.md decision 17)."""

    tokei_version: str
    lizard_version: str
    python_version: str
    instrument: dict[str, str]
    h14_soft_ceiling_lines: int
    h14_alarm_lines: int
    rules: dict[str, Any]


_PARAMS_KEYS = tuple(field.name for field in dataclasses.fields(_Params))


def measurement_params(rules: RustRules) -> dict[str, Any]:
    """What a measurement by this interpreter, these tools and these rules records as its params."""
    return dataclasses.asdict(
        _Params(
            tokei_version=_tokei_version(),
            lizard_version=_lizard_version(),
            python_version=platform.python_version(),
            instrument={'rust_metrics_snapshot': source_measures.blob_sha(Path(__file__).resolve())},
            h14_soft_ceiling_lines=H14_SOFT,
            h14_alarm_lines=H14_ALARM,
            rules=rules.to_json(),
        )
    )


def strip_rust_noise(text):
    """Blank out comments, strings and char literals (keeping newlines) so brace scanning is safe."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
            continue
        if text.startswith('/*', i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith('/*', j):
                    depth += 1
                    j += 2
                elif text.startswith('*/', j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            out.append(re.sub(r'[^\n]', ' ', text[i:j]))
            i = j
            continue
        m = re.match(r'b?r(#*)"', text[i:])
        if m:
            close = '"' + m.group(1)
            j = text.find(close, i + m.end())
            j = n if j < 0 else j + len(close)
            out.append(re.sub(r'[^\n]', ' ', text[i:j]))
            i = j
            continue
        if c == '"' or (c == 'b' and text.startswith('b"', i)):
            j = i + (2 if c == 'b' else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(re.sub(r'[^\n]', ' ', text[i:j]))
            i = j
            continue
        if c == "'":
            m2 = re.match(r"'(\\.|\\x[0-9a-fA-F]{2}|\\u\{[0-9a-fA-F]+\}|[^\\'])'", text[i:])
            if m2:
                out.append(' ' * m2.end())
                i += m2.end()
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def split_test_mod(clean, raw):
    """Return (clean_without_test_mods, test_mod_line_count, test_mod_text_raw)."""
    test_lines, test_chunks = 0, []
    spans = []
    for m in re.finditer(r'#\[cfg\(test\)\]\s*(pub(\([^)]*\))?\s+)?mod\s+\w+\s*\{', clean):
        start, j, depth = m.start(), m.end(), 1
        while j < len(clean) and depth:
            if clean[j] == '{':
                depth += 1
            elif clean[j] == '}':
                depth -= 1
            j += 1
        spans.append((start, j))
    if not spans:
        return clean, 0, ''
    kept, last = [], 0
    for s, e in spans:
        kept.append(clean[last:s])
        kept.append(re.sub(r'[^\n]', ' ', clean[s:e]))
        last = e
        test_lines += clean[s:e].count('\n')
        test_chunks.append(raw[s:e])
    kept.append(clean[last:])
    return ''.join(kept), test_lines, '\n'.join(test_chunks)


USE_RE = re.compile(r'(?<![\w:])(pub(?:\([^)]*\))?\s+)?use\s+([^;]+);', re.S)


def expand_use(tree):
    """`a::b::{c, d::{e, f}, self}` -> ['a::b::c', 'a::b::d::e', 'a::b::d::f', 'a::b']"""
    tree = re.sub(r'\s+as\s+\w+', '', tree)
    tree = re.sub(r'\s+', '', tree)

    def rec(s):
        if '{' not in s:
            return [s]
        head, rest = s.split('{', 1)
        inner, depth, parts, cur = rest[:-1], 0, [], ''
        for ch in inner:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            if ch == ',' and depth == 0:
                parts.append(cur)
                cur = ''
            else:
                cur += ch
        if cur:
            parts.append(cur)
        out = []
        for p in parts:
            if p == 'self':
                out.append(head.rstrip(':'))
            else:
                out.extend(rec(head + p))
        return out

    return [p for p in rec(tree) if p]


def scan_uses(clean):
    """Yield (statement_text, is_pub, is_deferred, line) for every `use` in clean text."""
    # brace stack of kinds: 'mod' or 'other'
    stack = []
    uses = []
    for m in USE_RE.finditer(clean):
        uses.append(m)
    use_idx = 0
    pos_kinds = {}
    for j, ch in enumerate(clean):
        if ch == '{':
            before = clean[max(0, j - 80) : j]
            kind = 'mod' if re.search(r'\bmod\s+\w+\s*$', before) else 'other'
            stack.append(kind)
        elif ch == '}':
            if stack:
                stack.pop()
        while use_idx < len(uses) and uses[use_idx].start() <= j:
            pos_kinds[uses[use_idx].start()] = 'other' in stack
            use_idx += 1
    for m in uses:
        deferred = pos_kinds.get(m.start(), False)
        line = clean.count('\n', 0, m.start()) + 1
        yield m.group(0), bool(m.group(1)), deferred, line, m.group(2)


class RustModules:
    def __init__(self, members):
        # members: {crate_name_underscored: (manifest_dir, ...)}
        self.members = members
        self.file_module = {}
        self.module_file = {}

    def register(self, path):
        for crate, d in self.members.items():
            if not path.startswith(d + '/'):
                continue
            rel = path[len(d) + 1 :]
            parts = rel.split('/')
            if parts[0] == 'src':
                segs = parts[1:]
                if segs[-1] in ('lib.rs', 'main.rs', 'mod.rs'):
                    segs = segs[:-1]
                else:
                    segs[-1] = segs[-1][:-3]
                mod = '::'.join([crate] + segs)
            elif parts[0] in ('tests', 'benches', 'examples'):
                segs = parts[1:]
                segs[-1] = segs[-1][:-3] if segs[-1].endswith('.rs') else segs[-1]
                mod = '::'.join([crate, parts[0]] + segs)
            else:
                mod = '::'.join([crate] + [p[:-3] if p.endswith('.rs') else p for p in parts])
            self.file_module[path] = mod
            self.module_file.setdefault(mod, path)
            return mod
        return None

    def resolve(self, from_mod, path):
        segs = path.split('::')
        crate = from_mod.split('::')[0]
        cur = from_mod.split('::')
        if segs[0] == 'crate':
            segs = [crate] + segs[1:]
        elif segs[0] == 'self':
            segs = cur + segs[1:]
        elif segs[0] == 'super':
            k = 0
            while k < len(segs) and segs[k] == 'super':
                k += 1
            base = cur[: max(1, len(cur) - k)]
            segs = base + segs[k:]
        elif segs[0] in self.members:
            pass
        else:
            return None
        for cut in range(len(segs), 0, -1):
            cand = '::'.join(segs[:cut])
            if cand in self.module_file:
                return cand
        return None


@dataclasses.dataclass(frozen=True)
class _Domain:
    paths: tuple[str, ...]
    unmapped: tuple[str, ...]


def _domain(root: Path, rules: RustRules) -> _Domain:
    """The tracked source files the rules keep and map to an area, in git's order; and those left unmapped."""
    kept = []
    for p in _run(['git', 'ls-files'], root).stdout.split('\n'):
        if not p or rules.excluded(p):
            continue
        ext = os.path.splitext(p)[1]
        if ext in SRC_EXT or p.startswith('hooks/'):
            if p.startswith('hooks/') and p.endswith('.sample'):
                continue
            kept.append(p)
    return _Domain(
        paths=tuple(p for p in kept if rules.area_of(p) is not None),
        unmapped=tuple(sorted(p for p in kept if rules.area_of(p) is None)),
    )


_Shape = TypeVar('_Shape')


def _tool_json(command: Sequence[str], root: Path, read: Callable[[Any], _Shape]) -> _Shape:
    """*command*'s stdout in *root* as JSON, taken apart by *read*; else a MetricsError naming the command and what it lacked."""
    printed = _run(command, root).stdout
    try:
        return read(json.loads(printed))
    except json.JSONDecodeError as exc:
        raise source_measures.MetricsError(f'`{shlex.join(command)}` in {root} printed no JSON: {exc}') from exc
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise source_measures.MetricsError(
            f'`{shlex.join(command)}` in {root} printed JSON of another shape: {exc.__class__.__name__}: {exc}'
        ) from exc


def _cargo_modules(root: Path, domain: Sequence[str]) -> RustModules:
    members = _tool_json(
        ['cargo', 'metadata', '--no-deps', '--format-version', '1'],
        root,
        lambda meta: {
            pkg['name'].replace('-', '_'): os.path.relpath(os.path.dirname(pkg['manifest_path']), root)
            for pkg in meta['packages']
        },
    )
    rm = RustModules(members)
    # Register every module before resolving any `use`: resolution matches the longest registered
    # prefix, so a single pass would drop every import of a file that sorts later.
    for p in domain:
        if p.endswith('.rs'):
            rm.register(p)
    return rm


_TokeiStats = tuple[int, int, int, str]


def _tokei_stats(root: Path) -> dict[str, _TokeiStats]:
    """Per file: (code, comments, blanks, language), embedded blobs folded in."""
    return _tool_json(['tokei', '-f', '-o', 'json', '.'], root, _tokei_report)


def _tokei_report(tok: Any) -> dict[str, _TokeiStats]:
    tstats = {}
    for lang, blob in tok.items():
        if lang == 'Total':
            continue
        for r in blob.get('reports', []):
            name = r['name'][2:] if r['name'].startswith('./') else r['name']
            s = r['stats']
            code, com, bl = s['code'], s['comments'], s['blanks']
            for sub in s.get('blobs', {}).values():  # embedded markdown in doc comments etc.
                code += sub['code']
                com += sub['comments']
                bl += sub['blanks']
            tstats[name] = (code, com, bl, lang)
    return tstats


_FunctionRow = tuple[int, str, int]


@dataclasses.dataclass(frozen=True)
class _LizardRun:
    rows: Mapping[str, list[_FunctionRow]]
    stderr: str


def _lizard_rows(root: Path, domain: Sequence[str], rules: RustRules) -> _LizardRun:
    """Per file, each function's (CCN, name, NLOC) in lizard's order; its stderr lines sorted, as -t 8 interleaves them."""
    lz_dirs = sorted({p.split('/')[0] for p in domain})
    lz_cmd = [sys.executable, '-m', 'lizard', '--csv', '-t', '8']
    for pattern in rules.lizard_excludes:
        lz_cmd += ['-x', pattern]
    lz_cmd += ['./' + d for d in lz_dirs]
    lz = _run(lz_cmd, root)
    fn_rows = defaultdict(list)
    for row in csv.reader(io.StringIO(lz.stdout)):
        if len(row) < 9:
            continue
        try:
            nloc, ccn = int(row[0]), int(row[1])
        except ValueError:
            continue
        f = row[6]
        f = f[2:] if f.startswith('./') else f
        fn_rows[f].append((ccn, row[7], nloc))
    return _LizardRun(rows=fn_rows, stderr='\n'.join(sorted(lz.stderr.splitlines())))


def _fix_density(root: Path, head: str, rules: RustRules) -> Counter[str]:
    fixlog = _run(
        [
            'git',
            'log',
            f'--since={rules.fix_density_since.isoformat()}',
            '--no-merges',
            '-i',
            '--grep=fix\\|bug\\|regress\\|repair\\|hotfix',
            '--name-only',
            '--format=',
            head,
        ],
        root,
    ).stdout
    return Counter(name for name in fixlog.split('\n') if name)


_Edge = tuple[str, str, list[str], int, bool, bool]  # (from_mod, to_mod, names, line, deferred, is_pub)


@dataclasses.dataclass
class _Gathered:
    """What the per-file pass collects across files: import edges, and the inputs of Rust test reach."""

    edges: list[_Edge] = dataclasses.field(default_factory=list)
    test_tokens_by_crate: defaultdict[str, set[str]] = dataclasses.field(default_factory=lambda: defaultdict(set))
    internal_pubs: dict[str, set[str]] = dataclasses.field(default_factory=dict)  # path -> names


@dataclasses.dataclass(frozen=True)
class _FileInputs:
    """The per-file measures taken over the whole tree before any file is read."""

    rules: RustRules
    modules: RustModules
    tstats: Mapping[str, _TokeiStats]
    fn_rows: Mapping[str, list[_FunctionRow]]
    fix_density: Counter[str]
    domain: frozenset[str]


def _new_record(p: str, raw: str, inputs: _FileInputs) -> dict[str, Any]:
    lines = raw.count('\n') + (0 if raw.endswith('\n') or not raw else 1)
    code, com, bl, lang = inputs.tstats.get(p, (None, None, None, None))
    prose = (com / (code + com)) if code is not None and com is not None and (code + com) else None
    rows = inputs.fn_rows.get(p, [])
    ccn_max, ccn_fn = max(((c, f) for c, f, _ in rows), default=(0, None))
    return dict(
        path=p,
        area=inputs.rules.area_of(p),
        kind='tests' if inputs.rules.is_test(p) else 'src',
        lang=lang or os.path.splitext(p)[1].lstrip('.'),
        lines=lines,
        code=code,
        comments=com,
        blanks=bl,
        prose_ratio=prose,
        functions=len(rows),
        ccn_max=ccn_max,
        ccn_max_fn=ccn_fn,
        ccn_total=sum(c for c, _, _ in rows),
        fix_density=inputs.fix_density.get(p, 0),
        embedded_test_lines=0,
        module=None,
        reexport_names=[],
        private_patch_targets=[],
        internal_pub_items=0,
        impl_types={},
    )


_INTERNAL_PUB_RE = re.compile(
    r'pub\((?:crate|super)\)\s+(?:unsafe\s+)?(?:fn|struct|enum|type|const|static|trait|mod)\s+(\w+)'
)
_HIDDEN_PUB_RE = re.compile(
    r'#\[doc\(hidden\)\]\s*pub\s+(?:unsafe\s+)?(?:fn|struct|enum|type|const|static|trait|mod)\s+(\w+)'
)
_TOKEN_RE = re.compile(r'\b[a-z_][a-z0-9_]{3,}\b')


def _rust_fields(rec: dict[str, Any], raw: str, modules: RustModules, gathered: _Gathered) -> None:
    p = rec['path']
    mod = modules.register(p)
    rec['module'] = mod
    clean = strip_rust_noise(raw)
    body, tl, test_raw = split_test_mod(clean, raw)
    rec['embedded_test_lines'] = tl
    # impl-block spread: Rust couples files through `impl Type` blocks, not `use`, so the
    # use-graph under-reads it; count impl targets per file and later how many files share one
    rec['impl_types'] = dict(
        Counter(
            m.group(1)
            for m in re.finditer(r'^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?([A-Z]\w*)', body, re.M)
        )
    )
    if rec['kind'] == 'src':
        names = set(_INTERNAL_PUB_RE.findall(body)) | set(_HIDDEN_PUB_RE.findall(body))
        gathered.internal_pubs[p] = names
        rec['internal_pub_items'] = len(names)
    crate = mod.split('::')[0] if mod else None
    if crate:
        toks = set(_TOKEN_RE.findall(test_raw)) if test_raw else set()
        if rec['kind'] == 'tests':
            toks |= set(_TOKEN_RE.findall(raw))
        gathered.test_tokens_by_crate[crate] |= toks
    if mod:
        for _stmt, is_pub, deferred, line, tree in scan_uses(body):
            leaves = expand_use(tree)
            if is_pub:
                rec['reexport_names'].extend(leaf.split('::')[-1] for leaf in leaves)
            targets = defaultdict(list)
            for leaf in leaves:
                t = modules.resolve(mod, leaf)
                if t and t != mod:
                    targets[t].append(leaf.split('::')[-1])
            for t, names_ in targets.items():
                gathered.edges.append((mod, t, names_, line, deferred, is_pub))


def _ts_fields(rec: dict[str, Any], raw: str, domain: frozenset[str], gathered: _Gathered) -> None:
    p = rec['path']
    rec['module'] = p
    for m in re.finditer(
        r'''(?:import|export)\s[^;]*?from\s+['"](\.{1,2}/[^'"]+)['"]|import\(\s*['"](\.{1,2}/[^'"]+)['"]''', raw
    ):
        spec = m.group(1) or m.group(2)
        target = os.path.normpath(os.path.join(os.path.dirname(p), spec))
        cands = [target + e for e in ('', '.ts', '.tsx', '/index.ts', '/index.tsx')]
        t = next((c for c in cands if c in domain), None)
        line = raw.count('\n', 0, m.start()) + 1
        deferred = m.group(2) is not None
        if t and t != p:
            gathered.edges.append((p, t, [], line, deferred, m.group(0).startswith('export')))
    if rec['kind'] == 'tests':
        rec['private_patch_targets'] = sorted(set(re.findall(r'''vi\.(?:mock|spyOn|doMock)\(\s*['"]?([\w./@-]+)''', raw)))


def _py_fields(rec: dict[str, Any], raw: str) -> None:
    rec['module'] = rec['path']
    if rec['kind'] == 'tests':
        rec['private_patch_targets'] = sorted(set(re.findall(r'''patch(?:\.object)?\(\s*['"]([\w.]+)['"]''', raw)))


@dataclasses.dataclass(frozen=True)
class _FileRecords:
    records: dict[str, dict[str, Any]]
    edges: list[_Edge]
    unreadable: tuple[str, ...]  # sorted; their measures are unknown, never zero


def _file_records(root: Path, paths: Sequence[str], inputs: _FileInputs) -> _FileRecords:
    """Per readable domain file, its record; every import edge between the files; the files that could not be read."""
    records = {}
    unreadable = []
    gathered = _Gathered()
    for p in paths:
        try:
            raw = (root / p).read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            unreadable.append(p)
            print(f'unreadable: {p}: {exc.__class__.__name__}: {exc}', file=sys.stderr)
            continue
        rec = _new_record(p, raw, inputs)
        if p.endswith('.rs'):
            _rust_fields(rec, raw, inputs.modules, gathered)
        elif p.endswith(('.ts', '.tsx', '.mjs')):
            _ts_fields(rec, raw, inputs.domain, gathered)
        elif p.endswith('.py'):
            _py_fields(rec, raw)
        records[p] = rec
    # Rust tests-reaching-internals: internal pub names referenced from tests of the same crate
    for p, names in gathered.internal_pubs.items():
        crate = records[p]['module'].split('::')[0]
        reached = sorted(n for n in names if n in gathered.test_tokens_by_crate.get(crate, set()))
        records[p]['private_patch_targets'] = reached
    return _FileRecords(records=records, edges=gathered.edges, unreadable=tuple(sorted(unreadable)))


def _sccs(graph: Mapping[str, set[str]]) -> list[list[str]]:
    """The strongly connected components of more than one node (Tarjan), each sorted and in sorted order.

    Sorting the list matters: the walk's discovery order follows set iteration, which
    hash randomisation varies per process, and a snapshot must not.
    """
    index, low, on, stack, result, idx = {}, {}, set(), [], [], [0]
    sys.setrecursionlimit(100000)

    def strong(v):
        index[v] = low[v] = idx[0]
        idx[0] += 1
        stack.append(v)
        on.add(v)
        for w in graph.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                result.append(sorted(comp))

    nodes = set(graph) | {w for ws in graph.values() for w in ws}
    for v in sorted(nodes):
        if v not in index:
            strong(v)
    return sorted(result)


#: The figures the import graph measures of each file, merged into its record by _measure_tree.
_GRAPH_COUNTS = ('fan_in_src', 'fan_out', 'reach_back_imports', 'deferred_imports', 'cycle_closing_imports')


@dataclasses.dataclass(frozen=True)
class _EdgePass:
    top_edges: Mapping[str, set[str]]
    all_edges: Mapping[str, set[str]]
    reach_back: list[dict[str, Any]]
    deferred: list[_Edge]
    counts: Mapping[str, Counter[str]]  # per module: fan_in_src, fan_out, reach_back_imports, deferred_imports


def _reaches_back(frm: str, to: str) -> bool:
    """Whether *frm*, a module outside a crate's tests, benches and examples, imports its strict ancestor *to*."""
    frm_segs, to_segs = frm.split('::'), to.split('::')
    is_test_crate = len(frm_segs) > 1 and frm_segs[1] in ('tests', 'benches', 'examples')
    return not is_test_crate and len(to_segs) < len(frm_segs) and frm_segs[: len(to_segs)] == to_segs


def _edge_pass(edges: Sequence[_Edge], kind_of: Mapping[str, str]) -> _EdgePass:
    """Top-level and all edges, reach-back entries and deferred edges; per module, its fan-in from src, fan-out and counts."""
    top_edges: defaultdict[str, set[str]] = defaultdict(set)
    all_edges: defaultdict[str, set[str]] = defaultdict(set)
    fan_in: defaultdict[str, set[str]] = defaultdict(set)
    reach_back = []
    deferred = []
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for edge in edges:
        frm, to, names, line, is_def, _is_pub = edge
        all_edges[frm].add(to)
        if not is_def:
            top_edges[frm].add(to)
        if kind_of.get(frm) == 'src':
            fan_in[to].add(frm)
        if _reaches_back(frm, to):
            reach_back.append({'from': frm, 'to': to, 'names': names[:8], 'line': line})
            counts[frm]['reach_back_imports'] += 1
        if is_def:
            deferred.append(edge)
            counts[frm]['deferred_imports'] += 1
    for module, importers in fan_in.items():
        counts[module]['fan_in_src'] = len(importers)
    for module, imported in all_edges.items():
        counts[module]['fan_out'] = len(imported)
    return _EdgePass(top_edges=top_edges, all_edges=all_edges, reach_back=reach_back, deferred=deferred, counts=counts)


@dataclasses.dataclass(frozen=True)
class _ImplSpread:
    widest: list[tuple[int, str, str]]  # (files, crate, type) of each type whose impl blocks span 4+ src files, widest first
    per_path: Mapping[str, tuple[int, str | None]]  # each record's widest-spread impl type: (files, type)


def _impl_spread(records: Mapping[str, Mapping[str, Any]]) -> _ImplSpread:
    """Per (crate, Type), the src files holding an impl block of it; per record, its widest-spread such type."""
    impl_files: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for r in records.values():
        if r['impl_types'] and r['kind'] == 'src' and r['module']:
            crate = r['module'].split('::')[0]
            for t in r['impl_types']:
                impl_files[(crate, t)].add(r['path'])
    per_path = {}
    for path, r in records.items():
        crate = r['module'].split('::')[0] if r['module'] and path.endswith('.rs') else None
        spread = {t: len(impl_files.get((crate, t), ())) for t in r['impl_types']} if crate else {}
        per_path[path] = (max(spread.values(), default=0), max(spread, key=lambda t: spread[t]) if spread else None)
    widest = sorted(((len(v), k[0], k[1]) for k, v in impl_files.items() if len(v) >= 4), reverse=True)
    return _ImplSpread(widest=widest, per_path=per_path)


@dataclasses.dataclass(frozen=True)
class _Graph:
    cycles: list[list[str]]
    hidden: list[list[str]]
    reach_back: list[dict[str, Any]]
    deferred: list[dict[str, Any]]
    impl_spread: list[tuple[int, str, str]]  # (files, crate, type), widest first
    file_figures: Mapping[str, Mapping[str, Any]]  # per path: the _GRAPH_COUNTS, impl_spread and impl_spread_type


def _deferred_entry(edge: _Edge, comp_of: Mapping[str, int]) -> dict[str, Any]:
    frm, to, names, line, _is_def, _is_pub = edge
    closes = comp_of.get(frm) is not None and comp_of.get(frm) == comp_of.get(to)
    return {'from': frm, 'to': to, 'line': line, 'imports': names[:8], 'closes_cycle': closes}


def _file_figures(counts: Counter[str], widest_impl: tuple[int, str | None]) -> dict[str, Any]:
    return {**{key: counts[key] for key in _GRAPH_COUNTS}, 'impl_spread': widest_impl[0], 'impl_spread_type': widest_impl[1]}


def _graph_measures(records: Mapping[str, Mapping[str, Any]], edges: Sequence[_Edge]) -> _Graph:
    """The import graph's cycles and listed edges and the widest impl spreads; per file, what they measure of it.

    Files sharing a module (a crate's lib.rs and main.rs) do not split its counts: the last one holds them.
    """
    owner = {r['module']: path for path, r in records.items() if r['module']}
    passed = _edge_pass(edges, {module: records[path]['kind'] for module, path in owner.items()})
    cycles = _sccs(passed.top_edges)
    full_cycles = _sccs(passed.all_edges)
    top_sets = [set(c) for c in cycles]
    comp_of = {m: number for number, c in enumerate(full_cycles) for m in c}
    deferred = [_deferred_entry(edge, comp_of) for edge in passed.deferred]
    closing = Counter(d['from'] for d in deferred if d['closes_cycle'])
    counts = {
        path: passed.counts.get(module, Counter()) + Counter(cycle_closing_imports=closing[module])
        for module, path in owner.items()
    }
    spread = _impl_spread(records)
    return _Graph(
        cycles=cycles,
        hidden=[c for c in full_cycles if not any(set(c) <= s for s in top_sets)],
        reach_back=passed.reach_back,
        deferred=deferred,
        impl_spread=spread.widest,
        file_figures={path: _file_figures(counts.get(path, Counter()), spread.per_path[path]) for path in records},
    )


def _prose_ratio(counts: Mapping[str, int]) -> float | None:
    return round(counts['comments'] / (counts['code'] + counts['comments']), 3) if (counts['code'] + counts['comments']) else None


def _per_area(records: Mapping[str, Mapping[str, Any]], fn_rows: Mapping[str, list[_FunctionRow]]) -> dict[str, Counter[str]]:
    """Per area, its files' sums; each record must already hold its graph figures (_GRAPH_COUNTS)."""
    per_area: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for r in records.values():
        c = per_area[r['area']]
        c['files'] += 1
        c['lines'] += r['lines']
        c['code'] += r['code'] or 0
        c['comments'] += r['comments'] or 0
        c[f"{r['kind']}_lines"] += r['lines']
        c['embedded_test_lines'] += r['embedded_test_lines']
        if r['lines'] >= H14_ALARM:
            c['alarm_files'] += 1
        elif r['lines'] >= H14_SOFT:
            c['soft_files'] += 1
        c['reach_back_imports'] += r['reach_back_imports']
        c['deferred_imports'] += r['deferred_imports']
        c['cycle_closing_imports'] += r['cycle_closing_imports']
        c['reexports'] += len(r['reexport_names'])
        c['patch_targets'] += len(r['private_patch_targets'])
        c['ccn_total'] += r['ccn_total']
        c['fns_ccn_over_15'] += sum(1 for ccn, _, _ in fn_rows.get(r['path'], []) if ccn > 15)
        c['functions'] += r['functions']
    return per_area


def _cycles_by_area(cycles: Sequence[list[str]], modules: RustModules, records: Mapping[str, dict[str, Any]]) -> Counter[str]:
    return Counter(records[modules.module_file[c[0]]]['area'] for c in cycles if c[0] in modules.module_file)


_IMPORT_GRAPH_NOTE = (
    'nodes are Rust module paths (crate::a::b) or TS/py file paths; Rust intra-crate module cycles '
    'are legal and common, so cycles measure layering not compile errors'
)


@dataclasses.dataclass(frozen=True)
class _Measured:
    evidence: dict[str, Any]
    figures: dict[str, Any]  # files, import_graph, per_area, cycles_by_area, totals, impl_spread


def _measure_tree(root: Path, head: str, rules: RustRules) -> _Measured:
    domain = _domain(root, rules)
    modules = _cargo_modules(root, domain.paths)
    lizard = _lizard_rows(root, domain.paths, rules)
    inputs = _FileInputs(
        rules=rules,
        modules=modules,
        tstats=_tokei_stats(root),
        fn_rows=lizard.rows,
        fix_density=_fix_density(root, head, rules),
        domain=frozenset(domain.paths),
    )
    read = _file_records(root, domain.paths, inputs)
    graph = _graph_measures(read.records, read.edges)
    records = {path: {**record, **graph.file_figures[path]} for path, record in read.records.items()}
    per_area = _per_area(records, lizard.rows)
    totals: Counter[str] = Counter()
    for c in per_area.values():
        totals.update(c)
    evidence = {
        'complete': not read.unreadable,
        'unreadable': list(read.unreadable),
        'domain_files': len(domain.paths),
        'unmapped_paths': list(domain.unmapped),
        'lizard_stderr': lizard.stderr[-2000:],
    }
    return _Measured(
        evidence=evidence,
        figures={
            'files': sorted(records.values(), key=lambda r: r['path']),
            'import_graph': {
                'edges': sorted({(f, t) for f, t, *_ in read.edges}),
                'reach_back': graph.reach_back,
                'deferred': graph.deferred,
                'cycles': graph.cycles,
                'hidden_cycles': graph.hidden,
                'typing_cycles': [],
                'note': _IMPORT_GRAPH_NOTE,
            },
            'per_area': {area: dict(c, prose_ratio=_prose_ratio(c)) for area, c in per_area.items()},
            'cycles_by_area': dict(_cycles_by_area(graph.cycles, modules, records)),
            'totals': dict(totals, prose_ratio=_prose_ratio(totals)),
            'impl_spread': [{'files': n, 'crate': c, 'type': t} for n, c, t in graph.impl_spread],
        },
    )


def _uncommitted_paths(root: Path) -> list[str]:
    listing = _run(['git', 'status', '--porcelain=v1', '-z', '--untracked-files=no', '--no-renames'], root).stdout
    return sorted({record[3:] for record in listing.split('\0') if record})


def _require_clean_tree(root: Path, as_of: str) -> None:
    dirty = _uncommitted_paths(root)
    if dirty:
        raise source_measures.MetricsError(
            f'the tree has uncommitted changes, so HEAD ({as_of}) is not what the work tree holds: '
            f'{", ".join(dirty)}; commit, or measure a clean checkout'
        )


def _require_unmoved_head(root: Path, as_of: str) -> None:
    now = source_measures.head_commit(root)
    if now != as_of:
        raise source_measures.MetricsError(
            f'HEAD moved from {as_of} to {now} during the measurement (a merge '
            'landed?); re-run, or measure a pinned worktree'
        )


def take_snapshot(root: Path, *, run_id: str, rules: RustRules) -> dict[str, Any]:
    """Measure *root*'s HEAD under *rules* as a SCHEMA_VERSION snapshot; refuse a dirty tree or a moved HEAD."""
    started = time.monotonic()
    params = measurement_params(rules)
    as_of = source_measures.head_commit(root)
    _require_clean_tree(root, as_of)
    measured = _measure_tree(root, as_of, rules)
    _require_unmoved_head(root, as_of)
    _require_clean_tree(root, as_of)
    return {
        'schema_version': SCHEMA_VERSION,
        'instrument': INSTRUMENT,
        'run_id': run_id,
        'as_of_sha': as_of,
        'evidence': measured.evidence,
        'cost': {'wall_clock_s': round(time.monotonic() - started, 1)},
        'params': params,
        **measured.figures,
    }


def summary_text(snapshot: Mapping[str, Any]) -> str:
    """The snapshot's headline, per-area table, top CCN files and widest impl spreads."""
    totals = Counter(snapshot['totals'])
    graph = snapshot['import_graph']
    cyc_by_area = snapshot['cycles_by_area']
    s = []
    s.append(
        f"metrics snapshot {snapshot['run_id']} @ {snapshot['as_of_sha'][:10]} — {len(snapshot['files'])} files, {totals['lines']} lines (prose share {totals['prose_ratio']}), "
        f"{totals['functions']} functions, {totals['fns_ccn_over_15']} with CCN>15; files >= {H14_ALARM} lines: {totals['alarm_files']}, {H14_SOFT}-{H14_ALARM}: {totals['soft_files']}; "
        f"module cycles (SCCs>1): {len(graph['cycles'])}, hidden: {len(graph['hidden_cycles'])}; reach-back imports: {totals['reach_back_imports']}; deferred: {totals['deferred_imports']} ({totals['cycle_closing_imports']} close a cycle); re-exports: {totals['reexports']}; test->internal reaches: {totals['patch_targets']}. No previous snapshot; since = none. Instruments: lizard CCN (not cognitive), tokei prose. Line counts set read budgets only (quality doc: Do not steer by)."
    )
    unreadable = snapshot['evidence']['unreadable']
    if unreadable:
        s.append('unreadable (measures unknown, never zero):')
        s.extend(f'  {path}' for path in unreadable)
    s.append(
        'area | files | lines | src/tests/embedded-test lines | prose | fns CCN>15 | alarm/soft files | cycles | reach-back | cycle-closing deferred | re-exports | test reaches'
    )
    for area, row in sorted(snapshot['per_area'].items(), key=lambda kv: -kv[1]['lines']):
        c = Counter(row)
        s.append(
            f"{area} | {c['files']} | {c['lines']} | {c['src_lines']}/{c['tests_lines']}/{c['embedded_test_lines']} | {c['prose_ratio']} | {c['fns_ccn_over_15']} | {c['alarm_files']}/{c['soft_files']} | {cyc_by_area.get(area, 0)} | {c['reach_back_imports']} | {c['cycle_closing_imports']} | {c['reexports']} | {c['patch_targets']}"
        )
    s.append('top files by per-function CCN max (max / file total / fn):')
    for r in sorted(snapshot['files'], key=lambda r: -r['ccn_max'])[:12]:
        s.append(f"  {r['path']} {r['ccn_max']}/{r['ccn_total']} {r['ccn_max_fn']}")
    s.append(
        'types whose impl blocks are spread over the most src files (crate Type files) — Rust couples files through impl blocks, invisible to the use-graph:'
    )
    s.append('  ' + '; '.join(f"{e['crate']} {e['type']} {e['files']}" for e in snapshot['impl_spread'][:12]))
    return '\n'.join(s)


class OtherSchemaError(source_measures.MetricsError):
    """A snapshot this instrument wrote under another SCHEMA_VERSION: it records no identity comparable to this one's."""


def _invalid(path: Path, field: str, found: object, expected: str) -> source_measures.MetricsError:
    return source_measures.MetricsError(f'{path}: {field} is {found!r}, not {expected}')


def load_snapshot(path: Path) -> dict[str, Any]:
    """The snapshot file at *path*, its identity fields checked; else a MetricsError naming the path and the field.

    A file of this instrument's under another schema is refused as an OtherSchemaError.
    """
    try:
        snapshot = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError) as exc:
        raise source_measures.MetricsError(f'{path}: could not be read -- {exc.__class__.__name__}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise source_measures.MetricsError(f'{path}: is not JSON -- {exc}') from exc
    if not isinstance(snapshot, dict):
        raise _invalid(path, 'the top level', type(snapshot).__name__, 'an object')
    if snapshot.get('instrument') != INSTRUMENT:
        raise _invalid(path, 'instrument', snapshot.get('instrument'), repr(INSTRUMENT))
    if snapshot.get('schema_version') != SCHEMA_VERSION:
        found = snapshot.get('schema_version')
        raise OtherSchemaError(f'{path}: schema_version is {found!r}, not {SCHEMA_VERSION}')
    if not isinstance(snapshot.get('as_of_sha'), str):
        raise _invalid(path, 'as_of_sha', snapshot.get('as_of_sha'), 'a string')
    params = snapshot.get('params')
    if not isinstance(params, dict):
        raise _invalid(path, 'params', type(params).__name__, 'an object')
    missing = [key for key in _PARAMS_KEYS if key not in params]
    if missing:
        raise source_measures.MetricsError(f'{path}: params records no {", ".join(missing)}')
    return snapshot


def _write(out: Path, text: str) -> None:
    try:
        safe_io.atomic_write_text(out, text, mkdir=True)
    except OSError as exc:
        raise source_measures.MetricsError(f'{out}: could not be written -- {exc.__class__.__name__}: {exc}') from exc


def _measure(args: argparse.Namespace, rules: RustRules) -> int:
    snapshot = take_snapshot(Path(args.root).resolve(), run_id=args.run_id, rules=rules)
    _write(Path(args.out), json.dumps(snapshot, indent=2) + '\n')
    print(summary_text(snapshot)[:4000])
    unmapped = snapshot['evidence']['unmapped_paths']
    print(f'\nunmapped paths: {len(unmapped)}', file=sys.stderr)
    for u in unmapped[:40]:
        print('  ', u, file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a Cargo workspace's tracked Rust, TypeScript, Python and shell at HEAD "
            'under a project rules file, or print the params such a measurement records. '
            'A report, never a gate.'
        ),
        epilog=(
            'Exit codes: 0 done; 2 instrument failure (a malformed rules file, a missing '
            'tool, a failing git, cargo, tokei or lizard, a dirty tree, a moved HEAD) or a '
            'misused command line.'
        ),
    )
    parser.add_argument('--rules', required=True, help="The project's rules TOML file.")
    parser.add_argument('--run-id', help="The measuring run's id, recorded as run_id.")
    parser.add_argument('--out', help='Where to write the snapshot JSON file.')
    parser.add_argument('--root', help='The checkout to measure.')
    parser.add_argument(
        '--params',
        action='store_true',
        help='Print the params a measurement here would record, as one JSON line; measure nothing.',
    )
    return parser


def _require_one_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    measuring = (args.run_id, args.out, args.root)
    if args.params:
        if measuring != (None, None, None):
            parser.error('--params takes only --rules')
    elif None in measuring:
        parser.error('measuring needs --run-id, --out and --root; otherwise give --params')


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see --help for the exit codes."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _require_one_mode(parser, args)
    try:
        rules = load_rules(Path(args.rules))
        if args.params:
            print(json.dumps(measurement_params(rules)))
            return 0
        return _measure(args, rules)
    except source_measures.MetricsError as exc:
        print(f'rust_metrics_snapshot: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
