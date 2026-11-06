#!/usr/bin/env python3
"""git's 3-way merge driver for the merge-lane ratchet baseline, and its installer.

``orchestrator/tests/merge_lane_ratchet_baseline.json`` is a byte-exact mirror of
a measurement, so every branch that moves a lane measure rewrites it, and git's
line merge conflicts whenever two branches move one row -- the row holds a
file's five measures on one line. Most such conflicts are not real: the counts
are additive over disjoint edits, so the merged value is computable from the
three images. This driver computes it (task 7298, ruled by Leo 2026-10-21) and
git records a clean merge.

THE RULES, per value, over base O, ours A and theirs B:

* a count in a ``files``/``functions``/``tests`` row: ``A == B`` -> A,
  ``A == O`` -> B, ``B == O`` -> A, else ``A + B - O``;
* ``patch_targets``, a sorted name set: ``(A & B) | (A - O) | (B - O)``, sorted;
* a list leaf of ``params``/``enumeration`` (an ordered path list): both sides'
  removals and insertions applied in place, refused only where both sides
  inserted at one spot -- the one case whose order the images cannot decide;
* a row present on one side only stands; one deleted on one side and changed on
  the other, or added on both sides differently, is refused;
* any other value (a scalar ``params`` entry, ``schema_version``, the README):
  a side that left it alone yields to the other; two different changes are
  refused.

SAFE, NOT MERELY PLAUSIBLE: the gate re-measures the merged tree
(``test_baseline_matches_a_fresh_measurement``), so a composed value that is not
the truth -- two edits to one function's cognitive score, identical hunks git
coalesced -- goes red at verify. The driver can lower nothing silently and
raise nothing silently. A REFUSAL is git's ordinary conflict: the file gets
git's own text merge, markers and all, and the merge stops as it did before this
driver existed. So does a host where the driver is not installed, because the
attribute names a driver git then cannot find.

Output goes through ``merge_lane_metrics.render_baseline`` (SPOT), carrying the
merged images' own README so a checkout whose copy of the instrument differs
from the trees it merges does not substitute its own.

INSTALLING. ``.gitattributes`` (tracked) maps the baseline to ``merge=``
``DRIVER_NAME``; the command lives in the repository's own, untracked git
config, which ``install`` writes. It is dark-factory's own and so is installed
by ``scripts/setup-host.sh``, never by the generic ``hooks/setup.sh``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import merge_lane_metrics as metrics

#: The ``merge=`` attribute value ``.gitattributes`` names, and the
#: ``merge.<name>.*`` config section ``install`` writes.
DRIVER_NAME = 'ratchet-baseline'

#: This file, relative to the repo root. git runs a merge driver from the top
#: of the work tree, so each merging checkout runs ITS OWN copy of the driver.
DRIVER_RELPATH = Path(__file__).resolve().relative_to(metrics.repo_root()).as_posix()

#: The ``merge.<name>.driver`` value: git substitutes %O, %A and %B with the
#: base, ours and theirs images and reads the result back from %A. ``-B``
#: writes no bytecode, so a run under the lane's probe leaves nothing in the
#: checkout it runs from.
DRIVER_COMMAND = f'python3 -B {DRIVER_RELPATH} merge %O %A %B'

_ABSENT = object()


class MergeRefused(Exception):
    """The three images cannot be merged by rule; git must record a conflict."""


def merge_baselines(base: dict, ours: dict, theirs: dict) -> dict:
    """The 3-way merge of three loaded baseline images. Raises ``MergeRefused``.

    Pure. Key order follows ours, then any key only theirs has, because
    ``render_baseline`` emits the small blocks in insertion order.
    """
    merged: dict = {}
    for key in _ordered_keys(ours, theirs):
        values = (base.get(key, _ABSENT), ours.get(key, _ABSENT), theirs.get(key, _ABSENT))
        if key in metrics.PER_PATH_SECTIONS:
            value = _merge_section(key, *(_as_dict(value) for value in values))
        else:
            value = _merge_value(key, *values)
        if value is not _ABSENT:
            merged[key] = value
    return merged


def _ordered_keys(ours: dict, theirs: dict) -> list[str]:
    return [*ours, *(key for key in theirs if key not in ours)]


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _one_side(path: str, base: object, ours: object, theirs: object) -> object:
    """The side that moved, or either when both made the same move; else refuse."""
    if ours == theirs or theirs == base:
        return ours
    if ours == base:
        return theirs
    raise MergeRefused(f'{path}: both sides changed it, differently')


def _merge_value(path: str, base: object, ours: object, theirs: object) -> object:
    """A non-row value: blocks merged key by key, path lists in place, else one side."""
    if ours == theirs or base in (ours, theirs):
        return _one_side(path, base, ours, theirs)
    if isinstance(ours, dict) and isinstance(theirs, dict):
        base_block = _as_dict(base)
        block = {
            key: _merge_value(
                f'{path}.{key}',
                base_block.get(key, _ABSENT),
                ours.get(key, _ABSENT),
                theirs.get(key, _ABSENT),
            )
            for key in _ordered_keys(ours, theirs)
        }
        return {key: value for key, value in block.items() if value is not _ABSENT}
    if isinstance(ours, list) and isinstance(theirs, list):
        return _merge_path_list(path, base if isinstance(base, list) else [], ours, theirs)
    return _one_side(path, base, ours, theirs)


def _merge_section(name: str, base: dict, ours: dict, theirs: dict) -> dict:
    """One per-path section, row by row; a row the merge leaves absent is dropped."""
    merged = {}
    for key in sorted(base.keys() | ours.keys() | theirs.keys()):
        row = _merge_row(
            f'{name}[{key}]',
            base.get(key, _ABSENT),
            ours.get(key, _ABSENT),
            theirs.get(key, _ABSENT),
        )
        if row is not _ABSENT:
            merged[key] = row
    return merged


def _merge_row(path: str, base: object, ours: object, theirs: object) -> object:
    if ours == theirs or base in (ours, theirs):
        return _one_side(path, base, ours, theirs)
    if _ABSENT in (base, ours, theirs):
        raise MergeRefused(
            f'{path}: added on both sides differently, or deleted on one side '
            'and changed on the other'
        )
    if isinstance(ours, dict) and isinstance(theirs, dict) and isinstance(base, dict):
        row = {
            measure: _merge_measure(
                f'{path}.{measure}',
                base.get(measure, _ABSENT),
                ours.get(measure, _ABSENT),
                theirs.get(measure, _ABSENT),
            )
            for measure in _ordered_keys(ours, theirs)
        }
        return {measure: value for measure, value in row.items() if value is not _ABSENT}
    return _merge_measure(path, base, ours, theirs)


def _merge_measure(path: str, base: object, ours: object, theirs: object) -> object:
    if ours == theirs or base in (ours, theirs):
        return _one_side(path, base, ours, theirs)
    if metrics.is_count(base) and metrics.is_count(ours) and metrics.is_count(theirs):
        composed = ours + theirs - base
        if composed < 0:
            raise MergeRefused(f'{path}: the two sides removed more than the base held')
        return composed
    if isinstance(base, list) and isinstance(ours, list) and isinstance(theirs, list):
        return _merge_name_set(base, ours, theirs)
    raise MergeRefused(f'{path}: both sides changed it, differently')


def _merge_name_set(base: list, ours: list, theirs: list) -> list:
    """A sorted name set: what both keep, plus what either side added."""
    base_set, ours_set, theirs_set = set(base), set(ours), set(theirs)
    return sorted((ours_set & theirs_set) | (ours_set - base_set) | (theirs_set - base_set))


def _merge_path_list(path: str, base: list, ours: list, theirs: list) -> list:
    """An ordered list of unique paths, merged the way git merges its lines.

    Each side's insertions are anchored after the nearest base entry that side
    kept; both sides' removals apply. Two different insertions at one anchor,
    or an insertion after an entry the other side removed, cannot be ordered
    from the images and are refused -- where git's text merge also conflicts.
    """
    if not all(isinstance(item, str) for item in (*base, *ours, *theirs)):
        raise MergeRefused(f'{path}: not a list of paths')
    if any(len(set(image)) != len(image) for image in (base, ours, theirs)):
        raise MergeRefused(f'{path}: an image lists one path twice')
    removed = (set(base) - set(ours)) | (set(base) - set(theirs))
    ours_runs, theirs_runs = _insertions(base, ours), _insertions(base, theirs)
    for anchor in ours_runs.keys() & theirs_runs.keys():
        if ours_runs[anchor] != theirs_runs[anchor]:
            raise MergeRefused(f'{path}: both sides inserted after {anchor!r}')
    runs = {**theirs_runs, **ours_runs}
    if any(anchor in removed for anchor in runs if anchor is not None):
        raise MergeRefused(f'{path}: an insertion follows an entry the other side removed')
    merged = list(runs.get(None, []))
    for item in base:
        if item not in removed:
            merged.append(item)
        merged.extend(runs.get(item, []))
    if len(set(merged)) != len(merged):
        raise MergeRefused(f'{path}: both sides added one entry at different places')
    return merged


def _insertions(base: list, side: list) -> dict[str | None, list]:
    """*side*'s entries absent from *base*, grouped by the base entry they follow."""
    known = set(base)
    runs: dict[str | None, list] = {}
    anchor: str | None = None
    for item in side:
        if item in known:
            anchor = item
        else:
            runs.setdefault(anchor, []).append(item)
    return runs


def render_merge(base: Path, ours: Path, theirs: Path) -> str:
    """The merged baseline's exact bytes. Raises ``MergeRefused`` or ``RatchetError``."""
    merged = merge_baselines(*(metrics.load_baseline(path) for path in (base, ours, theirs)))
    readme = merged.pop('_README', None)
    if isinstance(readme, str):
        return metrics.render_baseline(merged, readme=readme)
    return metrics.render_baseline(merged)


def _text_merge(base: Path, ours: Path, theirs: Path) -> str | None:
    """git's own line merge of the three images, conflict markers included."""
    result = subprocess.run(
        ['git', 'merge-file', '-p', '-L', 'ours', '-L', 'base', '-L', 'theirs',
         str(ours), str(base), str(theirs)],
        capture_output=True,
        text=True,
        check=False,
    )
    # merge-file exits with its conflict count, capped at 127; above that is an error.
    return result.stdout if 0 <= result.returncode <= 127 and result.stdout else None


def merge(base: Path, ours: Path, theirs: Path) -> int:
    """The driver: write the merge to *ours* and return 0, or record a conflict and return 1.

    ANY failure is a conflict, not only a refusal by rule: git's contract reads
    every non-zero exit as one, and the resolver should then meet git's own
    markers rather than a silently kept ``ours``.
    """
    try:
        text = render_merge(base, ours, theirs)
    except Exception as exc:
        print(
            f'{DRIVER_NAME} merge driver: {exc}. Recorded as a conflict: resolve '
            'the code, then regenerate with python scripts/merge_lane_metrics.py '
            f'--write-baseline {metrics.BASELINE_RELPATH}.',
            file=sys.stderr,
        )
        conflicted = _text_merge(base, ours, theirs)
        if conflicted is not None:
            ours.write_text(conflicted, encoding='utf-8')
        return 1
    ours.write_text(text, encoding='utf-8')
    return 0


def install(repo: Path) -> int:
    """Write this driver into *repo*'s own git config, which every linked worktree shares.

    ONE KEY, ``merge.<name>.driver``, and deliberately not the optional
    ``.name`` beside it: git DIES (exit 128, "lacks command line") on any
    ``merge.<name>.*`` section without a ``driver``, where a section that is
    wholly absent falls back to its line merge. A second key is one more way to
    leave the section half-written.
    """
    subprocess.run(
        ['git', '-C', str(repo), 'config', '--local', f'merge.{DRIVER_NAME}.driver', DRIVER_COMMAND],
        check=True,
    )
    print(f'{DRIVER_NAME}: installed in {repo} as `{DRIVER_COMMAND}`')
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="git's 3-way merge driver for the merge-lane ratchet baseline."
    )
    verbs = parser.add_subparsers(dest='verb', required=True)
    merging = verbs.add_parser('merge', help='Run as git invokes the driver: merge BASE OURS THEIRS into OURS.')
    for name in ('base', 'ours', 'theirs'):
        merging.add_argument(name, type=Path)
    installing = verbs.add_parser('install', help="Write the driver into a repository's git config.")
    installing.add_argument('--repo', type=Path, default=metrics.repo_root())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.verb == 'install':
        return install(args.repo)
    return merge(args.base, args.ours, args.theirs)


if __name__ == '__main__':
    sys.exit(main())
