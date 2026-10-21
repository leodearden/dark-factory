"""Every deep merge-ahead chain-telemetry reader agrees with hand-derived truth on one fixture (task 3188 ε).

The four readers are orchestrator/src/orchestrator/chain_telemetry.py::summarize_chain_telemetry
(the operator digest), scripts/analyze_speculation_depth.py (the forensic miner),
scripts/merge_lane_throughput.py::compute_chains (the throughput report) and
scripts/merge-deep-canary-predicate.sh (the η1 canary gate, run here as shipped);
an edit to any of them must keep this file green, or those four surfaces disagree.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import analyze_speculation_depth
import merge_lane_throughput
from cli_subprocess_timeout import cli_timeout_from_env
from orchestrator.chain_telemetry import ChainBucket, summarize_chain_telemetry
from orchestrator.event_store import EventStore

_PREDICATE = Path(__file__).resolve().parents[1] / 'merge-deep-canary-predicate.sh'
_PREDICATE_TIMEOUT = cli_timeout_from_env('CHAIN_TELEMETRY_AGREEMENT_CLI_TIMEOUT_SECS')

# One chain-landing episode: a 4-item chain whose tip verify passed and whose
# four items then landed, a halved 2-item round that failed, an ordinary
# adjacent speculative verify (its depth is frontier height) and a probe-era row.
MERGE_VERIFY = [
    {'chain_items': 4, 'passed': True, 'depth': 1},
    {'chain_items': 2, 'passed': False, 'depth': 1},
    {'chain_items': 1, 'passed': True, 'depth': 2},
    {'depth': 3, 'passed': True},
]
MERGE_FINALIZED = [
    *[{'state': 'done', 'landed_via_chain': 1} for _ in range(4)],
    {'state': 'done', 'landed_via_chain': None},
]

TRUE_HISTOGRAM = {1: ChainBucket(1, 1), 2: ChainBucket(0, 1), 4: ChainBucket(1, 1)}
TRUE_DEEP_VERIFIES = 2
TRUE_DEEP_FAILURES = 1
TRUE_DEEP_FAIL_RATE = 0.5
TRUE_ITEMS_LANDED = 4
TRUE_ITEMS_LANDED_PER_DEEP_VERIFY = 2.0
TRUE_LEGACY_ROWS = 1


def _rows(payloads: list[dict]) -> list[dict]:
    return [{'task_id': f't{i}', 'data': payload} for i, payload in enumerate(payloads)]


class TestReaderAgreement:
    def test_digest_core_reports_the_truth(self):
        telemetry = summarize_chain_telemetry(MERGE_VERIFY, MERGE_FINALIZED)
        assert telemetry.chain_items_histogram == TRUE_HISTOGRAM, f'{telemetry=}'
        assert telemetry.deep_verifies == TRUE_DEEP_VERIFIES, f'{telemetry=}'
        assert telemetry.deep_failures == TRUE_DEEP_FAILURES, f'{telemetry=}'
        assert telemetry.deep_fail_rate == TRUE_DEEP_FAIL_RATE, f'{telemetry=}'
        assert telemetry.items_landed_via_chain == TRUE_ITEMS_LANDED, f'{telemetry=}'
        assert telemetry.items_landed_per_deep_verify == TRUE_ITEMS_LANDED_PER_DEEP_VERIFY, (
            f'{telemetry=}'
        )
        assert telemetry.excluded_legacy_depth == TRUE_LEGACY_ROWS, f'{telemetry=}'

    def test_forensic_miner_reports_the_truth(self):
        per_chain_items = analyze_speculation_depth.compute_per_chain_items(_rows(MERGE_VERIFY))
        assert per_chain_items is not None
        buckets = {k: ChainBucket(v['pass'], v['total']) for k, v in per_chain_items.items()}
        assert buckets == TRUE_HISTOGRAM, f'{per_chain_items=}'
        legacy = sum(analyze_speculation_depth.is_legacy_deep_label(d) for d in MERGE_VERIFY)
        assert legacy == TRUE_LEGACY_ROWS, f'{legacy=}'

    def test_throughput_report_reports_the_truth(self):
        # Its items_per_deep_verify is the mean DISPATCHED chain length, a
        # different quantity, so it is deliberately not compared (task 6323).
        chains = merge_lane_throughput.compute_chains(_rows(MERGE_FINALIZED), _rows(MERGE_VERIFY))
        totals = {k: bucket.total for k, bucket in TRUE_HISTOGRAM.items()}
        assert chains['chain_items']['distribution'] == totals, f'{chains=}'
        assert chains['chain_items']['n_deep'] == TRUE_DEEP_VERIFIES, f'{chains=}'
        assert chains['items_landed_via_chain'] == TRUE_ITEMS_LANDED, f'{chains=}'

    def test_shipped_canary_predicate_reports_the_truth(self, tmp_path: Path):
        db = tmp_path / 'runs.db'
        EventStore(db, 'run-test')
        now = datetime.now(UTC)
        rows = [
            ('merge_verify', payload) for payload in MERGE_VERIFY
        ] + [('merge_finalized', payload) for payload in MERGE_FINALIZED]
        conn = sqlite3.connect(str(db))
        try:
            for i, (event_type, payload) in enumerate(rows):
                conn.execute(
                    'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
                    'VALUES (?, ?, ?, ?, ?)',
                    ((now - timedelta(minutes=i)).isoformat(), 'run-test', f't{i}',
                     event_type, json.dumps(payload)),
                )
            conn.commit()
        finally:
            conn.close()

        result = subprocess.run(
            ['bash', str(_PREDICATE), '--runs-db', str(db)],
            capture_output=True, text=True, timeout=_PREDICATE_TIMEOUT,
        )

        # 1 is expected: the 0.5 deep-fail rate exceeds the default 0.35 bound and the rows carry
        # no duration; 2 is an infra fault.
        assert result.returncode in (0, 1), f'{result.returncode=} {result.stdout=} {result.stderr=}'
        n_deep = re.search(r'n_deep_verifies=(\d+)', result.stdout)
        fail_rate = re.search(r'deep_fail_rate=([^(]+)\(', result.stdout)
        items_per = re.search(r'items_per_verify=([^(]+)\(', result.stdout)
        assert n_deep and fail_rate and items_per, result.stdout
        assert int(n_deep.group(1)) == TRUE_DEEP_VERIFIES, result.stdout
        assert float(fail_rate.group(1)) == round(TRUE_DEEP_FAIL_RATE, 4), result.stdout
        assert float(items_per.group(1)) == round(TRUE_ITEMS_LANDED_PER_DEEP_VERIFY, 4), (
            result.stdout
        )
