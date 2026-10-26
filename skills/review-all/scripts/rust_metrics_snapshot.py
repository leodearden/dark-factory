#!/usr/bin/env python3
"""Whole-repo quality metrics snapshot for a Cargo workspace (Rust + TypeScript + Python + shell).

The Rust shape of /review-all's Phase 1 instrument (SKILL.md Phase 1), for a project
that is not a Python uv-workspace and so cannot run scripts/quality_metrics_snapshot.py.
First written for reify's first run (review-all-reify-20261010); measured there:
2,901 files / 1.99M lines in 5m25s, 195 files at or over 2,000 lines. That run's
import-graph figures predate the module pre-registration in main() and under-read
every `use` of a file that sorts after its importer.

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
  - fix density: fix-grep'd non-merge commits touching the file since SINCE_DATE

Needs tokei, lizard, cargo and git on PATH. There is no --diff yet: the summary says
there was no previous snapshot.

PROJECT RULES (the block between the PROJECT RULES markers below): EXCLUDE_RE,
AREA_RULES, INFERRED, TEST_RE, LIZARD_EXCLUDES and the --since-date default are
supplied by the project overlay (references/project-overlay.md, Path -> area map and
Metrics instruments) -- reify's are the shipped example. Replace them from the
overlay before running on another project; the snapshot records AREA_RULES and
INFERRED in its params so the report can cite them.

Usage: rust_metrics_snapshot.py --root WORKTREE --run-id ID --out DIR [--since-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

H14_SOFT = 1500
H14_ALARM = 2000

SRC_EXT = {'.rs', '.ts', '.tsx', '.py', '.sh', '.mjs'}

# ==== PROJECT RULES: supplied by the overlay -- reify's are the shipped example ====
# Paths outside the measured domain (generated, vendored, docs, fixtures).
EXCLUDE_RE = re.compile(
    r'^(tree-sitter-reify/src/|docs/|prj/|examples/|data/|gui/src/editor/reifyParser(\.terms)?\.js$|'
    r'gui/src-tauri/sandbox/|.*node_modules/|.*\.d\.ts$|gui/src/__tests__/fixtures/)'
)

# One ordered rule list maps a path to a briefing area (contract §3); first match wins, with a
# catch-all per area. The overlay carries it; the snapshot records it in params.area_rules.
AREA_RULES = [
    (r'^crates/reify-eval/src/compute_targets/', 'analysis'),
    (
        r'^crates/reify-eval/src/(compute_persist|shell_extract_compute|morph_stage_\w*|engine_compute|modal_ops|dynamics_ops)\.rs$',
        'analysis',
    ),
    (r'^crates/reify-eval/src/(trajectory_ops)\.rs$', 'kinematics-and-motion'),
    (
        r'^crates/reify-eval/src/(geometry_ops|topology_\w+|primitive_attribute_seed|selector_vocabulary_v2|sweep_classifier|structural_classifier|appearance|kernel_registry|dispatcher|warm_pool|realization_\w+|engine_constraints|relate_solve|tolerance_combine)(\.rs|/)',
        'constraints-and-geometry',
    ),
    (r'^crates/reify-eval/', 'language-pipeline'),
    (
        r'^crates/reify-stdlib/src/(trajectory/|joints\.rs|mechanism\.rs|loop_closure\w*\.rs|orientation\.rs|snapshot\.rs|flexures/)',
        'kinematics-and-motion',
    ),
    (
        r'^crates/reify-stdlib/src/(dynamics/|modal/|fea\.rs|analysis\.rs|loads\.rs|supports\.rs|tensegrity\.rs)',
        'analysis',
    ),
    (r'^crates/reify-stdlib/', 'constraints-and-geometry'),
    (r'^crates/reify-compiler/src/connect\.rs$', 'assemblies'),
    (
        r'^crates/(reify-core|reify-ast|reify-syntax|reify-compiler|reify-ir|reify-expr|reify-builtins|reify-compute-contract|reify-spec-conformance|reify-runtime|reify-test-support)/',
        'language-pipeline',
    ),
    (r'^tree-sitter-reify/', 'language-pipeline'),
    (r'^crates/(reify-constraints|reify-kernel-\w+|reify-geometry)/', 'constraints-and-geometry'),
    (
        r'^crates/(reify-solver-elastic|reify-mesh-morph|reify-shell-extract|reify-fdm|reify-eval-fea-tests)/',
        'analysis',
    ),
    (r'^crates/reify-gcode/', 'kinematics-and-motion'),
    (r'^crates/(reify-cli|reify-lsp|reify-mcp|reify-config|reify-doc|reify-doc-build)/', 'tooling'),
    (r'^crates/reify-audit/', 'reify-audit'),
    (r'^crates/reify-build-utils/', 'factory-infra'),
    (r'^gui/', 'gui'),
    (r'^(scripts/|tests/|hooks/|\.cargo/|dark-factory-orchestrator\.yaml)', 'factory-infra'),
]
# Paths mapped by inference rather than by a briefing statement: briefing defects for /review-briefing.
INFERRED = {
    'crates/reify-test-support/': 'language-pipeline (inferred: compiler/eval test harness; briefing silent)',
    'crates/reify-build-utils/': 'factory-infra (inferred: native-dep link resolution; briefing silent)',
    'crates/reify-geometry/': 'constraints-and-geometry (inferred: kernel holder trait; briefing silent)',
    'crates/reify-eval/src/*': 'sub-crate rules split reify-eval across analysis / kinematics / constraints-and-geometry / language-pipeline by file name',
    'crates/reify-stdlib/src/*': 'sub-crate rules split reify-stdlib across kinematics / analysis / constraints-and-geometry by file name',
    'crates/reify-compiler/src/connect.rs': 'assemblies (inferred: connection graph compile; the assemblies area names no crate)',
}

# Which files are tests (kind 'tests'); everything else in the domain is src.
TEST_RE = re.compile(
    r'(^|/)(tests?|__tests__|benches|examples)/|(_tests?|\.test|\.spec)\.(rs|ts|tsx)$|(^|/)tests\.rs$|(^|/)test_[^/]*\.(py|sh)$|(^|/)test_support\.rs$|(^|/)mocks\.rs$'
)
# Paths lizard must not descend into (it walks directories, not the git domain).
LIZARD_EXCLUDES = ['*/node_modules/*', './target/*', './tree-sitter-reify/src/*', './docs/*', './prj/*', '*/.claude/*']
FIX_DENSITY_SINCE = '2026-07-06'
# ==== end PROJECT RULES ====


def sh(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=check).stdout


def area_of(path):
    for pat, area in AREA_RULES:
        if re.match(pat, path):
            return area
    return None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--since-date', default=FIX_DENSITY_SINCE)
    ap.add_argument('--lizard', default='lizard')
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    head = sh(['git', 'rev-parse', 'HEAD'], cwd=root).strip()
    if sh(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root).strip():
        print('dirty domain: tracked changes in the pinned worktree', file=sys.stderr)
        sys.exit(2)

    tracked = sh(['git', 'ls-files'], cwd=root).split('\n')
    domain = []
    for p in tracked:
        if not p or EXCLUDE_RE.match(p):
            continue
        ext = os.path.splitext(p)[1]
        if ext in SRC_EXT or p.startswith('hooks/'):
            if p.startswith('hooks/') and p.endswith('.sample'):
                continue
            domain.append(p)
    unmapped = sorted(p for p in domain if area_of(p) is None)
    domain = [p for p in domain if area_of(p) is not None]

    # ---- cargo members
    meta = json.loads(sh(['cargo', 'metadata', '--no-deps', '--format-version', '1'], cwd=root))
    members = {}
    for pkg in meta['packages']:
        d = os.path.relpath(os.path.dirname(pkg['manifest_path']), root)
        members[pkg['name'].replace('-', '_')] = d
    rm = RustModules(members)
    # Register every module before resolving any `use`: resolution matches the longest registered
    # prefix, so a single pass would drop every import of a file that sorts later.
    for p in domain:
        if p.endswith('.rs'):
            rm.register(p)

    # ---- tokei
    tok = json.loads(sh(['tokei', '-f', '-o', 'json', '.'], cwd=root))
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

    # ---- lizard
    lz_dirs = sorted({p.split('/')[0] if '/' not in p else p.split('/')[0] for p in domain})
    lz_cmd = [a.lizard, '--csv', '-t', '8']
    for pattern in LIZARD_EXCLUDES:
        lz_cmd += ['-x', pattern]
    lz_cmd += ['./' + d for d in lz_dirs]
    lz = subprocess.run(lz_cmd, cwd=root, text=True, capture_output=True)
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

    # ---- fix density
    fixlog = sh(
        [
            'git',
            'log',
            f'--since={a.since_date}',
            '--no-merges',
            '-i',
            '--grep=fix\\|bug\\|regress\\|repair\\|hotfix',
            '--name-only',
            '--format=',
            head,
        ],
        cwd=root,
    )
    fix_density = Counter(name for name in fixlog.split('\n') if name)

    # ---- per-file records + Rust imports
    records = {}
    edges = []  # (from_mod, to_mod, names, line, deferred, is_pub)
    test_tokens_by_crate = defaultdict(set)
    internal_pubs = {}  # path -> set(names)
    for p in domain:
        full = root / p
        try:
            raw = full.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        lines = raw.count('\n') + (0 if raw.endswith('\n') or not raw else 1)
        code, com, bl, lang = tstats.get(p, (None, None, None, None))
        prose = (com / (code + com)) if code is not None and (code + com) else None
        rows = fn_rows.get(p, [])
        ccn_max, ccn_fn = max(((c, f) for c, f, _ in rows), default=(0, None))
        rec: dict[str, Any] = dict(
            path=p,
            area=area_of(p),
            kind='tests' if TEST_RE.search(p) else 'src',
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
            fix_density=fix_density.get(p, 0),
            embedded_test_lines=0,
            module=None,
            fan_in_src=0,
            fan_out=0,
            reach_back_imports=0,
            deferred_imports=0,
            cycle_closing_imports=0,
            reexport_names=[],
            private_patch_targets=[],
            internal_pub_items=0,
        )
        if p.endswith('.rs'):
            mod = rm.register(p)
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
                names = set(
                    re.findall(
                        r'pub\((?:crate|super)\)\s+(?:unsafe\s+)?(?:fn|struct|enum|type|const|static|trait|mod)\s+(\w+)',
                        body,
                    )
                )
                names |= set(
                    re.findall(
                        r'#\[doc\(hidden\)\]\s*pub\s+(?:unsafe\s+)?(?:fn|struct|enum|type|const|static|trait|mod)\s+(\w+)',
                        body,
                    )
                )
                internal_pubs[p] = names
                rec['internal_pub_items'] = len(names)
            crate = mod.split('::')[0] if mod else None
            if crate:
                toks = set(re.findall(r'\b[a-z_][a-z0-9_]{3,}\b', test_raw)) if test_raw else set()
                if rec['kind'] == 'tests':
                    toks |= set(re.findall(r'\b[a-z_][a-z0-9_]{3,}\b', raw))
                test_tokens_by_crate[crate] |= toks
            if mod:
                for _stmt, is_pub, deferred, line, tree in scan_uses(body):
                    leaves = expand_use(tree)
                    if is_pub:
                        rec['reexport_names'].extend(leaf.split('::')[-1] for leaf in leaves)
                    targets = defaultdict(list)
                    for leaf in leaves:
                        t = rm.resolve(mod, leaf)
                        if t and t != mod:
                            targets[t].append(leaf.split('::')[-1])
                    for t, names_ in targets.items():
                        edges.append((mod, t, names_, line, deferred, is_pub))
        elif p.endswith(('.ts', '.tsx', '.mjs')):
            rec['module'] = p
            for m in re.finditer(
                r'''(?:import|export)\s[^;]*?from\s+['"](\.{1,2}/[^'"]+)['"]|import\(\s*['"](\.{1,2}/[^'"]+)['"]''', raw
            ):
                spec = m.group(1) or m.group(2)
                target = os.path.normpath(os.path.join(os.path.dirname(p), spec))
                cands = [target + e for e in ('', '.ts', '.tsx', '/index.ts', '/index.tsx')]
                t = next((c for c in cands if c in set(domain)), None)
                line = raw.count('\n', 0, m.start()) + 1
                deferred = m.group(2) is not None
                if t and t != p:
                    edges.append((p, t, [], line, deferred, m.group(0).startswith('export')))
            if rec['kind'] == 'tests':
                rec['private_patch_targets'] = sorted(
                    set(re.findall(r'''vi\.(?:mock|spyOn|doMock)\(\s*['"]?([\w./@-]+)''', raw))
                )
        elif p.endswith('.py'):
            rec['module'] = p
            if rec['kind'] == 'tests':
                rec['private_patch_targets'] = sorted(
                    set(re.findall(r'''patch(?:\.object)?\(\s*['"]([\w.]+)['"]''', raw))
                )
        records[p] = rec

    # Rust tests-reaching-internals: internal pub names referenced from tests of the same crate
    for p, names in internal_pubs.items():
        crate = records[p]['module'].split('::')[0]
        reached = sorted(n for n in names if n in test_tokens_by_crate.get(crate, set()))
        records[p]['private_patch_targets'] = reached

    # ---- graph measures
    mod_of = {}
    for r in records.values():
        if r['module']:
            mod_of[r['module']] = r
    top_edges = defaultdict(set)
    all_edges = defaultdict(set)
    reach_back = []
    deferred = []
    fan_in = defaultdict(set)
    fan_out = defaultdict(set)
    for frm, to, names, line, is_def, _is_pub in edges:
        all_edges[frm].add(to)
        if not is_def:
            top_edges[frm].add(to)
        fan_out[frm].add(to)
        if mod_of.get(frm, {}).get('kind') == 'src':
            fan_in[to].add(frm)
        frm_segs, to_segs = frm.split('::'), to.split('::')
        is_test_crate = len(frm_segs) > 1 and frm_segs[1] in ('tests', 'benches', 'examples')
        if not is_test_crate and len(to_segs) < len(frm_segs) and frm_segs[: len(to_segs)] == to_segs and '::' in frm:
            reach_back.append(dict(**{'from': frm, 'to': to, 'names': names[:8], 'line': line}))
            if frm in mod_of:
                mod_of[frm]['reach_back_imports'] += 1
        if is_def:
            deferred.append(dict(**{'from': frm, 'to': to, 'line': line, 'imports': names[:8], 'closes_cycle': False}))
            if frm in mod_of:
                mod_of[frm]['deferred_imports'] += 1

    def sccs(graph):
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
        return result

    cycles = sccs(top_edges)
    full_cycles = sccs(all_edges)
    top_sets = [set(c) for c in cycles]
    hidden = [c for c in full_cycles if not any(set(c) <= s for s in top_sets)]
    comp_of = {}
    for c in full_cycles:
        for m in c:
            comp_of[m] = id(c)
    for d in deferred:
        if comp_of.get(d['from']) is not None and comp_of.get(d['from']) == comp_of.get(d['to']):
            d['closes_cycle'] = True
            if d['from'] in mod_of:
                mod_of[d['from']]['cycle_closing_imports'] += 1
    for m, r in mod_of.items():
        r['fan_in_src'] = len(fan_in.get(m, ()))
        r['fan_out'] = len(fan_out.get(m, ()))
    # impl spread: for each (crate, Type) the set of src files holding an impl block for it
    impl_files = defaultdict(set)
    for r in records.values():
        if r.get('impl_types') and r['kind'] == 'src' and r['module']:
            crate = r['module'].split('::')[0]
            for t in r['impl_types']:
                impl_files[(crate, t)].add(r['path'])
    for r in records.values():
        r.setdefault('impl_types', {})
        crate = r['module'].split('::')[0] if r.get('module') and r['path'].endswith('.rs') else None
        spread = {t: len(impl_files[(crate, t)]) for t in r['impl_types']} if crate else {}
        r['impl_spread'] = max(spread.values(), default=0)
        r['impl_spread_type'] = max(spread, key=lambda t: spread[t]) if spread else None
    snapshot_impl_spread = sorted(((len(v), k[0], k[1]) for k, v in impl_files.items() if len(v) >= 4), reverse=True)

    # ---- aggregates
    per_area: defaultdict[str | None, Counter[str]] = defaultdict(Counter)
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
    totals: Counter[str] = Counter()
    for c in per_area.values():
        totals.update(c)
    prose_by_area = {
        area: round(c['comments'] / (c['code'] + c['comments']), 3) if (c['code'] + c['comments']) else None
        for area, c in per_area.items()
    }
    total_prose = round(totals['comments'] / (totals['code'] + totals['comments']), 3)
    cyc_by_area = Counter(records[rm.module_file[c[0]]]['area'] for c in cycles if c[0] in rm.module_file)

    snapshot = dict(
        schema='review-all-reify-rust-1',
        instrument='rust_metrics_snapshot.py (lead-authored, Phase 1)',
        run_id=a.run_id,
        as_of_sha=head,
        root=str(root),
        params=dict(
            h14_soft_ceiling_lines=H14_SOFT,
            h14_alarm_lines=H14_ALARM,
            complexity_instrument='lizard 1.24.1 cyclomatic complexity (CCN) per function',
            prose_instrument='tokei 15.0.0 comments/(code+comments)',
            fix_density_since=a.since_date,
            fix_density_rule='non-merge commits whose message matches fix|bug|regress|repair|hotfix (case-insensitive) touching the file, since the date, at HEAD',
            area_rules=AREA_RULES,
            inferred_area_mappings=INFERRED,
        ),
        evidence=dict(
            complete=True,
            unreadable=[],
            domain_files=len(records),
            unmapped_paths=unmapped,
            lizard_stderr=lz.stderr[-2000:],
        ),
        files=sorted(records.values(), key=lambda r: r['path']),
        import_graph=dict(
            edges=sorted({(f, t) for f, t, *_ in edges}),
            reach_back=reach_back,
            deferred=deferred,
            cycles=cycles,
            hidden_cycles=hidden,
            typing_cycles=[],
            note='nodes are Rust module paths (crate::a::b) or TS/py file paths; Rust intra-crate module cycles are legal and common, so cycles measure layering not compile errors',
        ),
        per_area={k: dict(v, prose_ratio=prose_by_area[k]) for k, v in per_area.items()},
        cycles_by_area=dict(cyc_by_area),
        totals=dict(totals, prose_ratio=total_prose),
        impl_spread=[dict(files=n, crate=c, type=t) for n, c, t in snapshot_impl_spread],
    )
    (out / 'snapshot.json').write_text(json.dumps(snapshot, indent=1))
    (out / 'import_graph.json').write_text(json.dumps(snapshot['import_graph']))

    # ---- summary (<=4k chars)
    s = []
    s.append(
        f"metrics snapshot {a.run_id} @ {head[:10]} — {len(records)} files, {totals['lines']} lines (prose share {total_prose}), "
        f"{totals['functions']} functions, {totals['fns_ccn_over_15']} with CCN>15; files >= {H14_ALARM} lines: {totals['alarm_files']}, {H14_SOFT}-{H14_ALARM}: {totals['soft_files']}; "
        f"module cycles (SCCs>1): {len(cycles)}, hidden: {len(hidden)}; reach-back imports: {totals['reach_back_imports']}; deferred: {totals['deferred_imports']} ({totals['cycle_closing_imports']} close a cycle); re-exports: {totals['reexports']}; test->internal reaches: {totals['patch_targets']}. No previous snapshot; since = none. Instruments: lizard CCN (not cognitive), tokei prose. Line counts set read budgets only (quality doc: Do not steer by)."
    )
    s.append(
        'area | files | lines | src/tests/embedded-test lines | prose | fns CCN>15 | alarm/soft files | cycles | reach-back | cycle-closing deferred | re-exports | test reaches'
    )
    for area, c in sorted(per_area.items(), key=lambda kv: -kv[1]['lines']):
        s.append(
            f"{area} | {c['files']} | {c['lines']} | {c['src_lines']}/{c['tests_lines']}/{c['embedded_test_lines']} | {prose_by_area[area]} | {c['fns_ccn_over_15']} | {c['alarm_files']}/{c['soft_files']} | {cyc_by_area.get(area, 0)} | {c['reach_back_imports']} | {c['cycle_closing_imports']} | {c['reexports']} | {c['patch_targets']}"
        )
    s.append('top files by per-function CCN max (max / file total / fn):')
    for r in sorted(records.values(), key=lambda r: -r['ccn_max'])[:12]:
        s.append(f"  {r['path']} {r['ccn_max']}/{r['ccn_total']} {r['ccn_max_fn']}")
    s.append(
        'types whose impl blocks are spread over the most src files (crate Type files) — Rust couples files through impl blocks, invisible to the use-graph:'
    )
    s.append('  ' + '; '.join(f"{c} {t} {n}" for n, c, t in snapshot_impl_spread[:12]))
    summary = '\n'.join(s)
    (out / 'metrics_summary.txt').write_text(summary)
    print(summary[:4000])
    print(f"\nunmapped paths: {len(unmapped)}", file=sys.stderr)
    for u in unmapped[:40]:
        print('  ', u, file=sys.stderr)


if __name__ == '__main__':
    main()
