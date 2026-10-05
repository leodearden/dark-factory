#!/usr/bin/env python3
"""Finding keys and artefact conformance for /hotspot-survey runs.

Implements docs/quality-findings-contract.md for this survey's artefacts; the
survey-specific field mapping it enforces is references/report-format.md, whose
`kind-tags` JSON block this script loads rather than restating. `key` is the
contract's interim reference implementation of the finding key (contract §2)
until shared/src/shared/finding_key.py lands and replaces it.

    key     print the contract key for one area / anchor / primary tag
    check   assert a findings JSON (and optionally its md report) conforms;
            exit 1 with one line per violation
    legacy  convert a pre-contract findings JSON (the 2026-07-06 shape) into
            contract-shaped seed entries for a first refresh, with provisional
            file-grain anchors, and report the key collisions those leave
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPORT_FORMAT = Path(__file__).resolve().parent.parent / "references" / "report-format.md"

RUN_ID_RE = re.compile(r"^[a-z][a-z-]*?-[a-z0-9_]+-\d{8}(-\d+)?$")
PRIMARY_TAG_RE = re.compile(r"^(h([1-9]|1[0-4])|comments|tests)$")
TAG_RE = re.compile(r"^(h([1-9]|1[0-4])|comments|tests|inv-\d+|kind:[a-z-]+)$")
ANCHOR_RE = re.compile(r"^(slug:[a-z0-9]+(-[a-z0-9]+)*|[\w.-][\w./-]*(::[A-Za-z_][\w.]*)?)$")
LINE_PIN_RE = re.compile(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
DISPOSITION_RE = re.compile(r"^(open|refuted|filed:\d+|accepted:\d+|fixed:[0-9a-f]{7,40})$")
SUPERSEDES_RE = re.compile(r"^(fk-[0-9a-f]{12}|positional:clusters\[\d+\]\.findings\[\d+\])$")
CITED_KEY_RE = re.compile(r"\bfk-[0-9a-f]{12}\b")

SEVERITIES = {"high", "medium", "low"}
VERDICTS = {"confirmed", "weakened", "refuted", "unverified"}
EVIDENCE_SOURCES = {"present-tree", "fix-history", "agent-transcripts", "metrics"}
ROUTES = {"mechanical", "structural"}
FINDING_FIELDS = ("key", "area", "anchor", "tags", "severity", "evidence_source", "statement",
                  "verdict", "disposition", "first_seen", "last_seen", "supersedes", "sub_area",
                  "id", "kind", "title", "route")
METHOD_FIELDS = frozenset({"run_id", "as_of_sha", "since", "evidence", "verification", "cost",
                           "inputs_consumed", "extra"})
CARRIED_DISPOSITIONS = re.compile(r"^(open|filed:\d+|accepted:\d+)$")


def load_kind_tags() -> dict:
    text = REPORT_FORMAT.read_text()
    block = text.split("<!-- kind-tags -->", 1)[1].split("<!-- /kind-tags -->", 1)[0]
    return json.loads(block.split("```json", 1)[1].split("```", 1)[0])


def normalise_anchor(anchor: str) -> str:
    return LINE_PIN_RE.sub("", anchor.strip().replace("\\", "/"))


def finding_key(area: str, anchor: str, primary_tag: str) -> str:
    canonical = f"{area}|{normalise_anchor(anchor)}|{primary_tag}".lower()
    return "fk-" + hashlib.sha256(canonical.encode()).hexdigest()[:12]


def briefing_subprojects(repo: Path) -> set[str]:
    briefing = yaml.safe_load((repo / "review" / "briefing.yaml").read_text())
    return set(briefing["subprojects"])


class Tree:
    """Read-only view of one commit, for anchor checks."""

    def __init__(self, repo: Path, sha: str):
        self.repo, self.sha, self._files = repo, sha, {}

    def is_commit(self, rev: str) -> bool:
        probe = subprocess.run(["git", "-C", str(self.repo), "cat-file", "-e", f"{rev}^{{commit}}"],
                               capture_output=True)
        return probe.returncode == 0

    def text(self, path: str) -> str | None:
        if path not in self._files:
            shown = subprocess.run(["git", "-C", str(self.repo), "show", f"{self.sha}:{path}"],
                                   capture_output=True, text=True)
            self._files[path] = shown.stdout if shown.returncode == 0 else None
        return self._files[path]


def check_finding(f: dict, ctx: dict) -> list[str]:
    where = f"finding {f.get('id') or f.get('key') or '?'}"
    missing = [name for name in FINDING_FIELDS if name not in f]
    if missing:
        return [f"{where}: missing contract fields {missing}"]
    checks = (_check_tags, _check_identity, _check_enums, _check_lineage, _check_anchor_in_tree)
    return [f"{where}: {problem}" for check in checks for problem in check(f, ctx)]


def _check_tags(f: dict, ctx: dict) -> list[str]:
    tags, kind = f["tags"], f["kind"]
    problems = []
    if not tags or not all(isinstance(t, str) and TAG_RE.match(t) for t in tags):
        problems.append(f"tags {tags!r} outside the contract §1 vocabulary")
    elif not PRIMARY_TAG_RE.match(tags[0]):
        problems.append(f"tags[0]={tags[0]!r} must be a heuristic or stance")
    if f"kind:{kind}" not in tags:
        problems.append(f"tags {tags!r} lack 'kind:{kind}'")
    primary_by_kind = ctx["kind_tags"]["primary_tag"]
    if kind not in primary_by_kind:
        problems.append(f"kind={kind!r} is not in report-format.md's kind-tags block")
    elif primary_by_kind[kind] and tags and tags[0] != primary_by_kind[kind]:
        problems.append(f"kind={kind!r} fixes primary tag {primary_by_kind[kind]!r}, got {tags[0]!r}")
    return problems


def _check_identity(f: dict, ctx: dict) -> list[str]:
    anchor, tags = f["anchor"], f["tags"]
    problems = []
    if anchor != normalise_anchor(anchor) or not ANCHOR_RE.match(anchor):
        problems.append(f"anchor {anchor!r} is not normalised (expected {normalise_anchor(anchor)!r})")
    if tags and f["key"] != finding_key(f["area"], anchor, tags[0]):
        problems.append(f"key {f['key']!r} != recomputed {finding_key(f['area'], anchor, tags[0])!r}")
    if f["area"] not in ctx["areas"]:
        problems.append(f"area {f['area']!r} is not a plain area key in {sorted(ctx['areas'])}")
    if f["sub_area"] not in ctx["clusters"]:
        problems.append(f"sub_area {f['sub_area']!r} is not a cluster key in {sorted(ctx['clusters'])}")
    return problems


def _check_enums(f: dict, ctx: dict) -> list[str]:
    problems = [f"{name}={f[name]!r} not in {sorted(allowed)}"
                for name, allowed in (("severity", SEVERITIES), ("verdict", VERDICTS), ("route", ROUTES))
                if f[name] not in allowed]
    sources = f["evidence_source"]
    if not sources or not set(sources) <= EVIDENCE_SOURCES:
        problems.append(f"evidence_source={sources!r} not a non-empty subset of {sorted(EVIDENCE_SOURCES)}")
    disposition, verdict = f["disposition"], f["verdict"]
    if not DISPOSITION_RE.match(disposition):
        problems.append(f"disposition {disposition!r} is not a contract §7 value")
    if (verdict == "refuted") != (disposition == "refuted"):
        problems.append(f"verdict={verdict!r} and disposition={disposition!r} disagree on refuted")
    if verdict in ("weakened", "refuted") and not f.get("verdict_notes"):
        problems.append(f"verdict={verdict!r} without verdict_notes")
    if verdict == "unverified" and not ctx["skip_reason"]:
        problems.append("unverified without method.extra.verification_skipped_reason")
    return problems


def _check_lineage(f: dict, ctx: dict) -> list[str]:
    first, last, supersedes = f["first_seen"], f["last_seen"], f["supersedes"]
    problems = [] if RUN_ID_RE.match(str(first)) else [f"first_seen={first!r} is not a run id"]
    if not isinstance(last, list) or not all(RUN_ID_RE.match(str(r)) for r in last):
        problems.append(f"last_seen={last!r} is not a list of run ids")
    elif not last or last[-1] != ctx["run_id"]:
        problems.append(f"last_seen={last!r} does not end with this run {ctx['run_id']!r}")
    if not isinstance(supersedes, list) or not all(SUPERSEDES_RE.match(str(s)) for s in supersedes):
        problems.append(f"supersedes={supersedes!r} is not a list of keys or positional refs")
    return problems


def _check_anchor_in_tree(f: dict, ctx: dict) -> list[str]:
    anchor, disposition = f["anchor"], f["disposition"]
    if disposition == "refuted" or disposition.startswith("fixed:") or anchor.startswith("slug:"):
        return []
    path, _, symbol = anchor.partition("::")
    body = ctx["tree"].text(path)
    if body is None:
        return [f"anchor path {path!r} absent at as_of_sha {ctx['tree'].sha}"]
    if symbol and not re.search(rf"\b{re.escape(symbol.split('.')[-1])}\b", body):
        return [f"anchor symbol {symbol!r} absent from {path} at as_of_sha"]
    return []


def check_artefact(doc: dict, repo: Path, areas: set[str], report_text: str | None) -> list[str]:
    method = doc.get("method", {})
    if set(method) != METHOD_FIELDS:
        return [f"method: keys must be contract §5's seven plus 'extra'; missing "
                f"{sorted(METHOD_FIELDS - set(method))}, outside extra {sorted(set(method) - METHOD_FIELDS)}"]
    tree = Tree(repo, method["as_of_sha"])
    errors = []
    if not RUN_ID_RE.match(method["run_id"]):
        errors.append(f"method: run_id {method['run_id']!r} is not <instrument>-<project_id>-<YYYYMMDD>[-<n>]")
    for name in ("as_of_sha", "since"):
        if not (name == "since" and method[name] == "none") and not tree.is_commit(method[name]):
            errors.append(f"method: {name}={method[name]!r} does not resolve to a commit in {repo}")
    ctx = {"kind_tags": load_kind_tags(), "areas": areas | {"repo"},
           "clusters": {c["key"] for c in doc.get("clusters", [])}, "run_id": method["run_id"],
           "skip_reason": (method["extra"] or {}).get("verification_skipped_reason"), "tree": tree}
    findings = doc.get("findings", [])
    for finding in findings:
        errors.extend(check_finding(finding, ctx))
    by_key = defaultdict(list)
    for finding in findings:
        by_key[finding.get("key")].append(finding.get("id"))
    errors.extend(f"key {k} shared by findings {ids}" for k, ids in by_key.items() if len(ids) > 1)
    tally = {v: sum(1 for f in findings if f.get("verdict") == v) for v in sorted(VERDICTS)}
    if method["verification"] != tally:
        errors.append(f"method: verification={method['verification']!r} but the findings tally {tally!r}")
    if report_text is not None:
        errors.extend(check_report(report_text, method, findings))
    return errors


def method_block(report_text: str) -> object:
    """The YAML of the fenced block opening the report's `## Method` section, or None."""
    section = report_text.split("\n## Method\n", 1)
    if len(section) != 2:
        return None
    body = section[1].lstrip("\n")
    if not body.startswith("```yaml\n"):
        return None
    return yaml.safe_load(body[len("```yaml\n"):].split("\n```", 1)[0])


def check_report(report_text: str, method: dict, findings: list[dict]) -> list[str]:
    block = method_block(report_text)
    errors = []
    if not isinstance(block, dict):
        errors.append("report: no `## Method` section opening with a fenced yaml mapping (contract §5)")
    elif block != method:
        differing = sorted(k for k in set(block) | set(method) if block.get(k) != method.get(k))
        errors.append(f"report: `## Method` block differs from the JSON method at {differing}")
    known = {f.get("key") for f in findings}
    cited = set(CITED_KEY_RE.findall(report_text))
    errors.extend(f"report: cites {k}, absent from the findings JSON" for k in sorted(cited - known))
    for f in findings:
        if CARRIED_DISPOSITIONS.match(str(f.get("disposition"))) and f.get("key") not in cited:
            errors.append(f"report: {f.get('disposition')} finding {f.get('key')} ({f.get('id')}) is never cited")
    return errors


def legacy_seed(doc: dict, run_id: str, cluster_areas: list[tuple[str, str]]) -> dict:
    kind_tags = load_kind_tags()
    if len(cluster_areas) != len(doc["clusters"]):
        raise SystemExit(f"--clusters names {len(cluster_areas)} clusters, the JSON has {len(doc['clusters'])}")
    seeds, unmintable = [], []
    for i, ((key_name, area_prefix), legacy) in enumerate(zip(cluster_areas, doc["clusters"], strict=True)):
        for j, old in enumerate(legacy["findings"]):
            primary = kind_tags["primary_tag"].get(old["kind"])
            anchor = normalise_anchor(old["files"][0].split()[0])
            history = old.get("bug_history_link", "")
            seed = {
                "key": finding_key(area_prefix, anchor, primary) if primary else None,
                "area": area_prefix, "sub_area": key_name, "anchor": anchor, "provisional_anchor": True,
                "tags": ([primary] if primary else []) + [f"kind:{old['kind']}"],
                "severity": kind_tags["severity_from_impact"][old["impact"]], "effort": old["effort"],
                "evidence_source": ["present-tree"] + ([] if history.lower().startswith("speculative") else ["fix-history"]),
                "statement": old["problem"], "proposal": old["proposal"], "title": old["title"],
                "kind": old["kind"], "files": old["files"], "bug_history_link": history,
                "verdict": old.get("verdict", "unverified"), "verdict_notes": old.get("verdict_notes", ""),
                "disposition": "open", "first_seen": run_id, "last_seen": [run_id],
                "route": "mechanical" if old["kind"] == "latent-bug" else "structural",
                "id": f"{key_name}.{j + 1}", "ref": f"positional:clusters[{i}].findings[{j}]",
            }
            seed["supersedes"] = [seed["ref"]]
            seeds.append(seed)
            if not primary:
                unmintable.append(seed["id"])
    groups = defaultdict(list)
    for seed in seeds:
        if seed["key"]:
            groups[seed["key"]].append(seed["id"])
    collisions = {k: ids for k, ids in groups.items() if len(ids) > 1}
    return {"run_id": run_id, "seeds": seeds,
            "summary": {"findings": len(seeds), "keys_minted": len(groups),
                        "findings_in_colliding_keys": sum(len(v) for v in collisions.values()),
                        "colliding_keys": collisions, "unmintable_needs_reviewer_tag": unmintable}}


def parse_cluster_areas(spec: str) -> list[tuple[str, str]]:
    pairs = [item.split("=", 1) for item in spec.split(",")]
    if any(len(p) != 2 for p in pairs):
        raise SystemExit(f"--clusters wants key=area,... in positional order, got {spec!r}")
    return [(key.strip(), area.strip()) for key, area in pairs]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    key_cmd = sub.add_parser("key")
    key_cmd.add_argument("area")
    key_cmd.add_argument("anchor")
    key_cmd.add_argument("primary_tag")
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--findings", required=True, type=Path)
    check_cmd.add_argument("--report", type=Path)
    check_cmd.add_argument("--repo", required=True, type=Path)
    legacy_cmd = sub.add_parser("legacy")
    legacy_cmd.add_argument("--input", required=True, type=Path)
    legacy_cmd.add_argument("--run-id", required=True)
    legacy_cmd.add_argument("--clusters", required=True, help="key=area,... in the JSON's cluster order")
    legacy_cmd.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "key":
        print(finding_key(args.area, args.anchor, args.primary_tag))
        return 0
    if args.command == "legacy":
        seeded = legacy_seed(json.loads(args.input.read_text()), args.run_id, parse_cluster_areas(args.clusters))
        args.out.write_text(json.dumps(seeded, indent=1))
        print(json.dumps(seeded["summary"], indent=1))
        return 0
    report = args.report.read_text() if args.report else None
    errors = check_artefact(json.loads(args.findings.read_text()), args.repo,
                            briefing_subprojects(args.repo), report)
    for error in errors:
        print(error)
    print(f"{len(errors)} violation(s)", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
