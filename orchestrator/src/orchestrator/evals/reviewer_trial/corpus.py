"""Corpus data model for the reviewer panel trial.

Manages synthetic and real-world diffs with ground-truth annotations
for evaluating reviewer panel configurations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GroundTruthIssue:
    """A known issue planted in or present in a corpus diff."""

    id: str                    # unique within diff, e.g. "py_obo_1"
    location: str              # "src/foo.py:42"
    category: str              # "off_by_one", "missing_await", etc.
    severity: str              # "blocking" | "suggestion"
    description: str
    mutation_type: str         # maps to the 11 mutation categories


@dataclass
class CorpusDiff:
    """A single diff in the evaluation corpus."""

    diff_id: str               # "py_missing_await", "rs_off_by_one", etc.
    language: str              # "python" | "rust" | "typescript"
    source: str                # "synthetic" | "real_world"
    diff_text: str
    description: str
    ground_truth: list[GroundTruthIssue]
    project: str | None = None  # for cwd resolution
    cwd: Path | None = None

    def blocking_issues(self) -> list[GroundTruthIssue]:
        return [gt for gt in self.ground_truth if gt.severity == 'blocking']

    def suggestion_issues(self) -> list[GroundTruthIssue]:
        return [gt for gt in self.ground_truth if gt.severity == 'suggestion']


@dataclass
class CorpusManifest:
    """Collection of corpus diffs with load/save and filtering."""

    diffs: list[CorpusDiff] = field(default_factory=list)
    version: str = '1.0'

    def filter_by_language(self, language: str) -> list[CorpusDiff]:
        return [d for d in self.diffs if d.language == language]

    def filter_by_source(self, source: str) -> list[CorpusDiff]:
        return [d for d in self.diffs if d.source == source]

    def get_diff(self, diff_id: str) -> CorpusDiff | None:
        for d in self.diffs:
            if d.diff_id == diff_id:
                return d
        return None

    def save(self, path: Path) -> None:
        """Save manifest and associated diff/annotation files."""
        corpus_dir = path.parent
        manifest_data = {
            'version': self.version,
            'diffs': [],
        }
        for diff in self.diffs:
            # Write diff file
            diff_dir = corpus_dir / diff.source
            diff_dir.mkdir(parents=True, exist_ok=True)
            diff_file = diff_dir / f'{diff.diff_id}.diff'
            diff_file.write_text(diff.diff_text)

            # Write annotation file
            ann_dir = corpus_dir / 'annotations'
            ann_dir.mkdir(parents=True, exist_ok=True)
            ann_file = ann_dir / f'{diff.diff_id}.json'
            ann_data = {
                'diff_id': diff.diff_id,
                'ground_truth': [asdict(gt) for gt in diff.ground_truth],
            }
            ann_file.write_text(json.dumps(ann_data, indent=2))

            # Manifest entry (no diff_text — loaded from file)
            entry = {
                'diff_id': diff.diff_id,
                'language': diff.language,
                'source': diff.source,
                'description': diff.description,
            }
            if diff.project:
                entry['project'] = diff.project
            manifest_data['diffs'].append(entry)

        path.write_text(json.dumps(manifest_data, indent=2))

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        """Load manifest and resolve diff files + annotations."""
        corpus_dir = path.parent
        raw = json.loads(path.read_text())
        diffs = []
        for entry in raw['diffs']:
            diff_id = entry['diff_id']
            source = entry['source']

            # Load diff text
            diff_file = corpus_dir / source / f'{diff_id}.diff'
            diff_text = diff_file.read_text() if diff_file.exists() else ''

            # Load annotations
            ann_file = corpus_dir / 'annotations' / f'{diff_id}.json'
            ground_truth = []
            if ann_file.exists():
                ann_data = json.loads(ann_file.read_text())
                ground_truth = [
                    GroundTruthIssue(**gt) for gt in ann_data['ground_truth']
                ]

            cwd = None
            project = entry.get('project')
            if project:
                cwd = _resolve_project_cwd(project)

            diffs.append(CorpusDiff(
                diff_id=diff_id,
                language=entry['language'],
                source=source,
                diff_text=diff_text,
                description=entry['description'],
                ground_truth=ground_truth,
                project=project,
                cwd=cwd,
            ))

        return cls(diffs=diffs, version=raw.get('version', '1.0'))


# Project → working directory mapping for invoke_agent cwd
_PROJECT_CWD_MAP = {
    'dark-factory': Path('/home/leo/src/dark-factory'),
    'reify': Path('/home/leo/src/reify'),
    'reify-gui': Path('/home/leo/src/reify/gui'),
    'taskmaster-ai': Path('/home/leo/src/dark-factory/taskmaster-ai'),
}


def _resolve_project_cwd(project: str) -> Path | None:
    return _PROJECT_CWD_MAP.get(project)
