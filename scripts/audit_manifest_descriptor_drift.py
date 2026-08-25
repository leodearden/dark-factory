#!/usr/bin/env python3
"""Audit capability-manifest ``delivered_check`` descriptors against their
producer task's ``metadata.delivered_checks`` entry.

READ-ONLY / REPORT-ONLY: this module and its CLI never mutate a task record or
a manifest file. Every database connection it opens is a read-only SQLite URI
(``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``), so the sweep is
structurally incapable of writing to the live WAL database the running
orchestrator holds open. Manifest YAML on disk is only ever read. There is no
``--apply`` flag and no MCP client is ever constructed. RESYNCING A DRIFTED
SIDECAR IS A SEPARATE, REVIEWED EDIT — never done from this report by this
script.

WHAT THE SWEEP IS FOR (task 4545). ``metadata.delivered_checks`` is copied
exactly ONE WAY, sidecar -> task record, at ``commit_planning``
(fused-memory/src/fused_memory/server/manifest_stamping.py step 5, the
``for task in doc.tasks:`` block that builds a ``DeliveredCheckMeta`` per
mechanical capability and writes the list through ``update_task``). NOTHING
EVER SYNCS BACK. So when a delivered_check is repaired by hand on the task
record — because the sidecar's pattern was wrong, or matched the wrong file, or
had its ``expect`` inverted — the sidecar keeps the stale spelling, and the two
descriptors silently disagree from then on.

THAT IS A REGENERATION HAZARD, NOT A LIVE BLOCK, and the distinction is the
whole reason this is an audit rather than a gate. The δ gate
(orchestrator/src/orchestrator/delivered_checks.py) reads the TASK METADATA,
which is already the repaired copy — so no dependent is blocked today. The
exposure is that a re-decompose of the same PRD re-runs the stamper and
re-writes the STALE sidecar spelling over the repair, silently reverting it.
This sweep is the regression check that keeps the two spellings in agreement so
that re-stamp is a no-op.

CORRECTING A DESCRIPTOR FORCES RE-EVALUATION WITHOUT OPERATOR ACTION:
orchestrator/src/orchestrator/scheduler.py::_delivered_checks_descriptor_digest
folds the whole-list digest into the delivered-checks cache key, so an edit at a
fixed main sha is a cache MISS by design. Nothing here has to invalidate
anything.

THERE IS NO COMMIT-ORDERING PREMISE HERE. An earlier framing of this defect
supposed the drift came from sidecars committed before/after their task records;
that half was FALSIFIED and lives on in task 3500. This sweep makes no claim
about when either side was written — only that the two spellings, as they stand
right now, disagree.

AND IT DOES NOT JUDGE WHETHER A CHECK PASSES. A row here means the two
descriptors DISAGREE, nothing more. Both spellings may fail (the transcript
preservation seam's gz-consumer row is exactly that: correct-and-superseded,
task 3578 restored gzip reading after 3618 removed it), both may pass, or one of
each. Whether a check is satisfied on main is the δ gate's question and
``verify_delivered_checks_on_main``'s, not this script's.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Tier 1 (tasks.db discovery) and Tier 3 (audit-script CLI plumbing: the roots
# loop, the warn-and-continue skip, the exit-code ladder and the two reporting
# layout primitives), imported as a flat sibling and shared with
# audit_combine_gate_marker_loss.py / audit_wiped_metadata_files.py. Tier 2 is
# the LEAK-SCANNER skeleton and does not apply here — it sweeps db paths and
# accumulates matches, not one audit per project root.
#
# IMPORT-RESOLUTION CONTRACT: _task_db_scan.py must stay a flat sibling in
# scripts/, and this script must NEVER be invoked via `python -m` — the CLI
# tests shell out to the script path and resolve this import solely because a
# DIRECTLY-EXECUTED script puts its own directory at sys.path[0].
from _task_db_scan import (
    AUDIT_EXIT_FINDINGS,
    AUDIT_EXIT_NO_ROOT,
    AUDIT_EXIT_NOTHING_AUDITED,
    AUDIT_EXIT_OK,
    format_coverage_block,
    format_kv_line,
    run_audit_cli,
    tasks_db_path,
)

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute. An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy of this script running from a worktree would validate manifests using
# the MAIN checkout's schema — and this script is EXPECTED to run from a
# worktree (that is what --manifest-root is for). Same reasoning and same form
# as audit_combine_gate_marker_loss.py (tasks 2881/2882/3329). The
# shared.capability_manifest import below MUST stay after this insert.
_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.capability_manifest import (  # noqa: E402
    DeliveredCheckMeta,
    load_capability_manifest,
)

# The kinds the stamper actually copies. manifest_stamping.py step 5 reads
# `if check is None or check.kind not in ('grep', 'script'): continue`, so a
# 'manual' capability never reaches metadata.delivered_checks at all and can
# never drift. Comparing one would emit a permanent false positive on every
# manual-checked capability in the corpus.
MECHANICAL_CHECK_KINDS = ("grep", "script")

# The sidecar filename suffix, as `git ls-files` matches it.
_MANIFEST_GLOB = "*.capability-manifest.yaml"

_GIT_TIMEOUT_SECS = 30


class ManifestDiscoveryUnavailable(RuntimeError):
    """`git ls-files` could not enumerate the manifest corpus.

    Raised rather than degraded to an empty list on purpose: an empty corpus
    and a clean corpus are INDISTINGUISHABLE in the finding count, and only one
    of them is good news. Swallowing this would let a non-checkout, a broken
    git, or a permissions failure render as a confident zero
    (docs/legibility/design-invariants.md, no-silent-fail-soft).
    """


def _tracked_manifest_paths(manifest_root: str) -> list[str]:
    """Every TRACKED ``*.capability-manifest.yaml`` under *manifest_root*.

    Returns sorted, unique, repo-relative paths — repo-relative because that is
    what a reader can act on against a checkout, whereas an absolute path from
    somebody else's worktree is noise in a report.

    TRACKED rather than globbed, deliberately. The stamper only ever reads what
    is committed, and an untracked scratch copy of a sidecar in a working tree
    is not part of the corpus a re-decompose would re-stamp from. This also
    makes the sweep agree with shared/tests/test_capability_manifest.py's
    checked-in-corpus family, which is likewise git-driven.

    Raises :class:`ManifestDiscoveryUnavailable` on any failure to run git or a
    non-zero return — see that class for why this is never degraded to ``[]``.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(manifest_root), "ls-files", "-z", "--", _MANIFEST_GLOB],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManifestDiscoveryUnavailable(
            f"could not run `git ls-files` in {manifest_root}: {exc}"
        ) from exc

    if completed.returncode != 0:
        raise ManifestDiscoveryUnavailable(
            f"`git ls-files` failed in {manifest_root} (rc={completed.returncode}): "
            f"{completed.stderr.strip() or 'no stderr'}"
        )

    return sorted({p for p in completed.stdout.split("\0") if p})


def _decode_metadata(raw: object) -> dict:
    """Decode a raw ``metadata`` blob into a dict, degrading to ``{}``.

    Copied from :func:`audit_combine_gate_marker_loss._decode_metadata`.
    Degrades for NULL, an empty string, malformed JSON, or a payload that
    decodes to anything other than a dict (a list, a bare scalar, ``null``). A
    corrupt metadata blob is data to be skipped, never a reason to abort a
    sweep over thousands of tasks.
    """
    if not raw or not isinstance(raw, (str, bytes)):
        return {}
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_task_delivered_checks(
    tasks_db_path: str,
) -> tuple[set[int], dict[int, dict[str, dict]]]:
    """Load ``(task ids, task id -> {capability name -> entry})`` from tasks.db.

    Returns BOTH the row-id set and the descriptor mapping because they answer
    two different coverage questions that must not be conflated: a manifest
    binding a task_id with NO ROW AT ALL is a stale/unstamped binding, while a
    task that exists but carries no same-named ``delivered_checks`` entry is an
    un-stamped or metadata-wiped gate. Collapsing them would misattribute 6
    live rows to a population of 32 that a different audit already owns.

    A task absent from the mapping is a COVERAGE row, never a finding: this
    sweep reports descriptors that DISAGREE, and a missing entry is not a
    disagreement — it is an absence, whose dominant live cause is the
    curator-combine ``metadata`` wipe that
    ``scripts/audit_combine_gate_marker_loss.py`` (tasks 3146/3329) owns.

    ``tag`` is pinned to ``'master'`` because that is the tag the stamper writes
    under and the only tag the live store uses; the schema permits the same
    numeric id under a second tag, and a manifest's stamped ``task_id`` carries
    no tag, so an unpinned query would let an unrelated same-id row from another
    tag masquerade as the producer.

    Opens the database via a read-only URI (``mode=ro``) so the load is
    structurally incapable of mutating live task records even while fused-memory
    holds the same file open in WAL mode. Closed in a ``try/finally`` and never
    a ``with`` block — a sqlite3 ``with`` is a TRANSACTION, not a close.
    """
    row_ids: set[int] = set()
    by_task: dict[int, dict[str, dict]] = {}
    conn = sqlite3.connect(f"file:{tasks_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT id, metadata FROM tasks WHERE tag = 'master'")
        for task_id, metadata in cursor:
            try:
                tid = int(task_id)
            except (TypeError, ValueError):
                continue
            row_ids.add(tid)
            checks = _decode_metadata(metadata).get("delivered_checks")
            if not isinstance(checks, list):
                continue
            entries: dict[str, dict] = {}
            for entry in checks:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    entries[entry["name"]] = entry
            if entries:
                by_task[tid] = entries
    finally:
        conn.close()
    return row_ids, by_task


class DescriptorDrift(NamedTuple):
    """One capability whose sidecar and task-record descriptors disagree.

    ``manifest`` is repo-relative; ``differing_fields`` names the normalized
    :class:`DeliveredCheckMeta` fields that differ, sorted; ``sidecar_check``
    and ``task_check`` carry BOTH full normalized descriptors so a reader never
    has to open two files to see what drifted.

    A NamedTuple rather than a dataclass, following the sibling audits' stated
    precedent: ``_asdict()`` feeds the JSON writer.
    """

    manifest: str
    task_id: int
    label: str
    capability: str
    differing_fields: tuple[str, ...]
    sidecar_check: dict
    task_check: dict


def _expected_meta(capability_name: str, check: object) -> dict:
    """The normalized descriptor a re-decompose WOULD stamp for this capability.

    Constructed field-for-field the way manifest_stamping.py step 5 constructs
    it, so the comparison is against exactly what the stamper would write —
    not against an approximation of it. Normalizing through
    :class:`DeliveredCheckMeta` on this side and the task side both is what
    makes an ABBREVIATED task entry (one omitting the defaulted ``script`` /
    ``args`` / ``timeout_secs`` keys) compare equal to a full one, which is the
    difference between the 8 real drift rows and 22 absent-vs-default artifacts.
    """
    return DeliveredCheckMeta(
        name=capability_name,
        kind=check.kind,  # type: ignore[union-attr]
        pattern=check.pattern,  # type: ignore[union-attr]
        expect=check.expect,  # type: ignore[union-attr]
        paths=check.paths,  # type: ignore[union-attr]
        script=check.script,  # type: ignore[union-attr]
        args=check.args,  # type: ignore[union-attr]
        timeout_secs=check.timeout_secs,  # type: ignore[union-attr]
    ).model_dump()


class AuditCoverage(NamedTuple):
    """How much of the corpus the sweep could actually compare.

    ALWAYS reported, including on a zero-finding sweep. The finding list is a
    comparison of MATCHED PAIRS only, and three classes never reach a
    comparison at all: a capability whose producer task carries no same-named
    entry, a manifest binding a task_id with no tasks.db row, and a sidecar that
    would not parse. Presenting the finding list as the whole corpus would be a
    no-silent-fail-soft violation (docs/legibility/design-invariants.md).

    ``manifest_parse_failure_details`` and ``uncomparable_details`` carry the
    strings themselves, not just counts: an operator told only that "2 manifests
    failed to parse" cannot find out WHICH, which swallows the failures at
    exactly the reporting boundary the rule is about.

    ``git_discovery_failed`` marks a run whose manifest corpus could not be
    enumerated at all — the one case where a zero finding count means nothing.
    Every field after the counts is defaulted so it is purely additive to
    positional construction.
    """

    manifests_swept: int
    mechanical_capabilities_compared: int
    capabilities_without_task_entry: int
    manifest_tasks_without_db_row: int
    malformed_task_entries: int
    manifest_parse_failures: int
    manifest_parse_failure_details: tuple[str, ...] = ()
    uncomparable_details: tuple[str, ...] = ()
    git_discovery_failed: bool = False


class ProjectAudit(NamedTuple):
    """One project's audit: what drifted, and what could be seen.

    ``project_root`` owns the tasks.db; ``manifest_root`` owns the sidecars.
    They are the SAME path by default and differ only under ``--manifest-root``
    — which exists because ``.taskmaster/`` is gitignored and lives only in the
    primary checkout, so a sweep of a task WORKTREE's sidecars must still read
    the primary checkout's task store. Both are recorded so a decoupled run is
    unambiguous in the report.
    """

    project_root: str
    manifest_root: str
    findings: list[DescriptorDrift]
    coverage: AuditCoverage


def _drift_sort_key(drift: DescriptorDrift) -> tuple[str, int, str]:
    """Manifest path, then NUMERIC task id, then capability name.

    A report whose row order depends on filesystem or sqlite iteration order
    cannot be diffed between runs.
    """
    return (drift.manifest, drift.task_id, drift.capability)


def audit_project(project_root: str, manifest_root: str | None = None) -> ProjectAudit:
    """Compare every mechanical sidecar descriptor against its task-record twin.

    *manifest_root* defaults to *project_root*, so a single-checkout run and a
    multi-project sweep behave exactly as they would without the flag.

    Raises ``sqlite3.Error`` for an unreadable task store, which
    :func:`_task_db_scan.sweep_project_roots` turns into a warn-and-skip.
    :class:`ManifestDiscoveryUnavailable` is caught HERE and recorded as
    ``git_discovery_failed`` rather than allowed to escape, because that helper
    only catches ``sqlite3.Error`` and an escaping traceback would abort the
    whole multi-root sweep over one bad manifest root.
    """
    root = str(project_root)
    manifests_root = str(manifest_root) if manifest_root is not None else root

    row_ids, task_checks = load_task_delivered_checks(str(tasks_db_path(root)))

    try:
        relpaths = _tracked_manifest_paths(manifests_root)
    except ManifestDiscoveryUnavailable as exc:
        return ProjectAudit(
            project_root=root,
            manifest_root=manifests_root,
            findings=[],
            coverage=AuditCoverage(
                manifests_swept=0,
                mechanical_capabilities_compared=0,
                capabilities_without_task_entry=0,
                manifest_tasks_without_db_row=0,
                malformed_task_entries=0,
                manifest_parse_failures=0,
                uncomparable_details=(str(exc),),
                git_discovery_failed=True,
            ),
        )

    findings: list[DescriptorDrift] = []
    manifests_swept = 0
    compared = 0
    without_entry = 0
    without_db_row = 0
    malformed = 0
    parse_failure_details: list[str] = []
    uncomparable_details: list[str] = []

    for relpath in relpaths:
        try:
            doc = load_capability_manifest(Path(manifests_root) / relpath)
        except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
            # NAMED, not merely counted: a sweep that could not read half the
            # corpus must never read as complete (no-silent-fail-soft).
            parse_failure_details.append(f"{relpath}: {exc}")
            continue

        manifests_swept += 1
        for task in doc.tasks:
            if task.task_id is None:
                # Authoring time, before commit_planning stamps the id. It
                # binds no producer, so there is nothing to compare against.
                continue
            try:
                task_id = int(task.task_id)
            except (TypeError, ValueError):
                continue

            if task_id not in row_ids:
                without_db_row += 1
                continue
            entries = task_checks.get(task_id, {})

            for capability in task.capabilities:
                check = capability.delivered_check
                if check is None or check.kind not in MECHANICAL_CHECK_KINDS:
                    continue
                compared += 1

                entry = entries.get(capability.name)
                if entry is None:
                    without_entry += 1
                    continue

                expected = _expected_meta(capability.name, check)
                try:
                    actual = DeliveredCheckMeta(**entry).model_dump()
                except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
                    # NAMED, never silently dropped. A task-record entry
                    # that will not validate cannot be compared, and is a
                    # different defect from a drifted one — it goes to the
                    # coverage details, not the finding list.
                    malformed += 1
                    uncomparable_details.append(
                        f"task {task_id} capability {capability.name!r}: "
                        f"unvalidatable task-record entry: {exc}"
                    )
                    continue

                if expected == actual:
                    continue

                differing = tuple(sorted(k for k in expected if expected[k] != actual[k]))
                findings.append(DescriptorDrift(
                    manifest=relpath,
                    task_id=task_id,
                    label=task.label,
                    capability=capability.name,
                    differing_fields=differing,
                    sidecar_check=expected,
                    task_check=actual,
                ))

    findings.sort(key=_drift_sort_key)
    return ProjectAudit(
        project_root=root,
        manifest_root=manifests_root,
        findings=findings,
        coverage=AuditCoverage(
            manifests_swept=manifests_swept,
            mechanical_capabilities_compared=compared,
            capabilities_without_task_entry=without_entry,
            manifest_tasks_without_db_row=without_db_row,
            malformed_task_entries=malformed,
            manifest_parse_failures=len(parse_failure_details),
            manifest_parse_failure_details=tuple(parse_failure_details),
            uncomparable_details=tuple(uncomparable_details),
        ),
    )


_COVERAGE_CAVEAT = (
    "  COVERAGE (the findings above are a comparison of MATCHED PAIRS only, "
    "not the whole corpus - a capability with no same-named task-record entry, "
    "a manifest binding a task_id with no tasks.db row, and an unparseable "
    "sidecar are all counted here and are NONE of them drift; the "
    "missing-entry class is owned by audit_combine_gate_marker_loss.py and is "
    "never remediated from this report):"
)

# Printed ABOVE a git-discovery-failed project's rows, because a reader must
# see that the zero below means "nothing was enumerated" before reading it as
# "nothing was wrong".
_DISCOVERY_FAILED_NOTICE = (
    "  WARNING: the manifest corpus could not be enumerated -- this is NOT a "
    "clean result, it is an UNKNOWN one (an empty corpus and a clean corpus "
    "are indistinguishable in the finding count below)"
)


def _format_finding_line(drift: DescriptorDrift) -> str:
    """One drift row, in the sibling audits' ``key=value`` style.

    The field set and order are this script's; format_kv_line supplies only the
    indent and separator. ``fields`` is deliberately shorter than the
    ``differing_fields`` attribute it carries.
    """
    return format_kv_line([
        ("manifest", drift.manifest),
        ("task_id", drift.task_id),
        ("capability", drift.capability),
        ("fields", ",".join(drift.differing_fields)),
    ])


def _format_coverage(coverage: AuditCoverage) -> list[str]:
    """Render the always-printed coverage block.

    Never omitted and never abbreviated when there are no findings: the point
    is that the finding list compares matched pairs only, and a reader must be
    told the size of the unpaired remainder.

    The details NAME the unreadable sidecars and the unvalidatable task
    entries, never just count them. A count alone tells an operator that
    coverage is incomplete but not where to look, which swallows the failure at
    exactly the reporting boundary no-silent-fail-soft is about.

    Only the ALIGNMENT is shared with the sibling audits (via
    format_coverage_block); _COVERAGE_CAVEAT and the labels below are this
    script's, and say what THIS script could not see.
    """
    return format_coverage_block(
        _COVERAGE_CAVEAT,
        [
            ("manifests swept:", coverage.manifests_swept),
            ("mechanical capabilities compared:", coverage.mechanical_capabilities_compared),
            ("capabilities with no task entry:", coverage.capabilities_without_task_entry),
            ("manifest tasks with no db row:", coverage.manifest_tasks_without_db_row),
            ("unvalidatable task entries:", coverage.malformed_task_entries),
            ("manifests that failed to parse:", coverage.manifest_parse_failures),
        ],
        details=(*coverage.manifest_parse_failure_details, *coverage.uncomparable_details),
    )


def format_report(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a human-readable report.

    Every project gets its COVERAGE block, INCLUDING projects with zero
    findings — see :func:`_format_coverage`.

    Both roots are named on the header line so a ``--manifest-root`` run is
    unambiguous: a reader never has to guess which manifest tree was compared
    against which task store.

    Pure: returns the text and never prints. ``main()`` does the single print.
    """
    lines: list[str] = []
    total = 0
    for audit in audits:
        if audit.manifest_root == audit.project_root:
            lines.append(f"{audit.project_root}:")
        else:
            lines.append(
                f"{audit.project_root} (manifests from {audit.manifest_root}):"
            )
        if audit.coverage.git_discovery_failed:
            # ABOVE the rows on purpose — see _DISCOVERY_FAILED_NOTICE.
            lines.append(_DISCOVERY_FAILED_NOTICE)

        total += len(audit.findings)
        lines.append(f"  -- drifted descriptors ({len(audit.findings)}) --")
        for drift in audit.findings:
            lines.append(_format_finding_line(drift))
            for field in drift.differing_fields:
                lines.append(
                    f"      {field}: sidecar={drift.sidecar_check[field]!r} "
                    f"task={drift.task_check[field]!r}"
                )
        lines.extend(_format_coverage(audit.coverage))

    lines.append(
        f"{total} drifted descriptor(s) across {len(audits)} project(s)"
    )
    return "\n".join(lines)


def format_json(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a JSON OBJECT (never a bare array).

    An object because the coverage block must travel WITH the findings: a
    machine consumer handed a bare finding array could read it as the whole
    corpus, which is the exact false-completeness the coverage block exists to
    prevent. The caveat prose ships in the payload for the same reason.
    """
    return json.dumps({
        "caveat": _COVERAGE_CAVEAT,
        "projects": [
            {
                "project_root": audit.project_root,
                "manifest_root": audit.manifest_root,
                "coverage": audit.coverage._asdict(),
                "findings": [drift._asdict() for drift in audit.findings],
            }
            for audit in audits
        ],
    })


def _is_dirty(audits: list[ProjectAudit]) -> bool:
    """Exit 1 keys on findings OR a failed manifest discovery.

    The second disjunct is not belt-and-braces. An empty corpus and a clean
    corpus are INDISTINGUISHABLE in the finding count, and only one of them is
    good news — so a run that could not enumerate the corpus must never exit 0
    (docs/legibility/design-invariants.md, no-silent-fail-soft).
    """
    return any(a.findings or a.coverage.git_discovery_failed for a in audits)


# The per-script NAMES survive because this script's epilog wording is its own,
# but the VALUES have ONE home: the returns live in _task_db_scan.run_audit_cli,
# so a local re-spelling would drift from what actually gets returned.
# test_exit_constants_alias_the_shared_tier_3_codes is what keeps these honest.
EXIT_OK = AUDIT_EXIT_OK                            # swept; no drift
EXIT_DRIFT = AUDIT_EXIT_FINDINGS                   # a drifted descriptor, or a
                                                   # failed manifest discovery
EXIT_NO_ROOT = AUDIT_EXIT_NO_ROOT                  # no project root resolved to
                                                   # a readable tasks.db
EXIT_NOTHING_AUDITED = AUDIT_EXIT_NOTHING_AUDITED  # roots resolved but EVERY
                                                   # one failed to audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY sweep for capability-manifest delivered_check "
            "descriptors that have drifted from the "
            "metadata.delivered_checks entry on their producer task. "
            "delivered_checks is copied one way, sidecar -> task, at "
            "commit_planning and never syncs back, so a hand-repaired task "
            "record sits beside a stale sidecar that any re-decompose would "
            "re-stamp over the repair. Reporting only -- never mutates a task "
            "record or a manifest. Resyncing a drifted sidecar is a separate, "
            "individually-reviewed edit."
        ),
        epilog=(
            "exit codes: 0 = swept, every compared descriptor agrees; 1 = at "
            "least one drifted descriptor, OR the manifest corpus could not be "
            "enumerated (an empty corpus and a clean corpus are "
            "indistinguishable in the finding count, and only one is good "
            "news); 2 = no project root resolved to a readable tasks.db; 3 = "
            "roots resolved but every one failed to audit, so NOTHING was "
            "swept (never treat 3 as a clean run). A drift row means the two "
            "spellings DISAGREE -- it is not a claim that either one passes."
        ),
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help=(
            "Project root to audit (resolves <root>/.taskmaster/tasks/tasks.db, "
            "and its own tracked *.capability-manifest.yaml files unless "
            "--manifest-root says otherwise). May be repeated."
        ),
    )
    parser.add_argument(
        "--manifest-root", dest="manifest_root", default=None,
        help=(
            "Git checkout to read manifest sidecars FROM (default: the project "
            "root itself). Exists because .taskmaster/ is gitignored and lives "
            "only in the primary checkout, so auditing a task WORKTREE's "
            "sidecars means reading manifests from the worktree while reading "
            "the task store from the primary root. INTENDED FOR SINGLE-ROOT "
            "RUNS: this is ONE manifest tree applied to EVERY resolved root, "
            "so combining it with several roots is warned about on stderr."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON object (findings plus coverage) instead of a report.",
    )
    return parser


def _audit_root(root: str, args: argparse.Namespace) -> ProjectAudit:
    """Audit ONE project root under the (possibly overridden) manifest root.

    Raises ``sqlite3.Error`` for an unreadable task store, which
    :func:`_task_db_scan.sweep_project_roots` turns into a warn-and-skip; every
    other exception propagates. Returns exactly one audit, per that function's
    one-audit-per-root contract.
    """
    return audit_project(root, args.manifest_root)


def _render(audits: list[ProjectAudit], args: argparse.Namespace) -> str:
    return format_json(audits) if args.json else format_report(audits)


def _warn_manifest_root_across_roots(
    roots: list[str], args: argparse.Namespace
) -> None:
    """Warn once, up front, that ONE --manifest-root is applied to MANY roots.

    Runs on the RESOLVED root list, before the empty-roots exit-2 return — the
    position run_audit_cli's on_roots hook occupies. Warned, not rejected: one
    manifest tree over several same-corpus task stores is a legitimate use, and
    this script reports rather than gatekeeps.
    """
    if len(roots) > 1 and args.manifest_root:
        print(
            f"warning: --manifest-root {args.manifest_root!r} is applied to ALL "
            f"{len(roots)} resolved project roots, so every root is audited "
            "against the SAME manifest tree rather than its own. Any root whose "
            "sidecars differ will be compared against another checkout's "
            "spellings. Prefer one --manifest-root run per root.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = swept, every compared descriptor agrees; 1 = a drifted
    descriptor OR a manifest corpus that could not be enumerated; 2 = no
    project root resolved to a readable tasks.db; 3 = roots resolved but EVERY
    one failed to audit, so NOTHING was swept.

    3 exists because 0 would otherwise be returned for two opposite outcomes —
    "swept everything, found nothing" and "swept nothing at all" — and a
    CI/cron consumer reading only the exit code would take a total failure for
    a clean run. That is exactly the silent fail-soft this module refuses to do
    (see the module docstring).

    1 covers the failed-discovery case for the SAME reason, one level down: a
    root whose `git ls-files` failed produces zero findings, and zero findings
    from an unenumerable corpus is an UNKNOWN result, not a clean one.

    A single unreadable project (a corrupt/locked tasks.db) does NOT abort the
    sweep: it is logged to stderr and skipped so every other project is still
    audited, and a trailing warning states that the results are incomplete.

    The roots loop, the warn-and-continue skip and the exit-code ladder are
    :func:`_task_db_scan.run_audit_cli` (Tier 3). What stays here is what
    genuinely differs: this script's parser and epilog, its ``--manifest-root``
    handling (:func:`_audit_root` and :func:`_warn_manifest_root_across_roots`),
    its object-shaped JSON, its report and its :func:`_is_dirty` predicate.
    Nothing in this file returns a bare integer.
    """
    return run_audit_cli(
        argv,
        parser=_build_parser(),
        audit_fn=_audit_root,
        render=_render,
        is_dirty=_is_dirty,
        on_roots=_warn_manifest_root_across_roots,
    )


if __name__ == "__main__":
    sys.exit(main())
