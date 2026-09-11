"""Tests for escalation/dedupe.py — dedupe helpers and configuration."""

from __future__ import annotations

import pytest


class TestSummaryDedupeKey:
    """summary_dedupe_key() — pure helper, no I/O."""

    def test_similar_summaries_share_key(self):
        """Two infra-style summaries with the same first 3 words share a key."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('Fused-memory  CONNECTION timeout!')
        k2 = summary_dedupe_key('fused-memory connection timeout on port 8002')
        assert k1 == k2

    def test_short_summary_key_length(self):
        """Fewer than 3 tokens → key length equals token count."""
        from escalation.dedupe import summary_dedupe_key

        key = summary_dedupe_key('lost link')
        assert key == ('lost', 'link')
        assert len(key) == 2

    def test_empty_summary_produces_empty_key(self):
        """Empty or whitespace-only summary → empty tuple."""
        from escalation.dedupe import summary_dedupe_key

        assert summary_dedupe_key('') == ()
        assert summary_dedupe_key('   ') == ()
        assert summary_dedupe_key('\t\n') == ()

    def test_different_first_three_tokens_differ(self):
        """Summaries with different first 3 tokens must NOT share a key."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('fused-memory connection timeout on port 8002')
        k2 = summary_dedupe_key('neo4j connection timeout on port 8002')
        assert k1 != k2

    def test_single_token_key(self):
        """Single-word summary → 1-tuple."""
        from escalation.dedupe import summary_dedupe_key

        assert summary_dedupe_key('oops') == ('oops',)

    def test_punctuation_stripped(self):
        """Punctuation is stripped before tokenizing."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('db! connection? lost.')
        k2 = summary_dedupe_key('db connection lost')
        assert k1 == k2

    def test_casefold_applied(self):
        """Key is case-insensitive."""
        from escalation.dedupe import summary_dedupe_key

        k1 = summary_dedupe_key('UPPER lower MiXeD')
        k2 = summary_dedupe_key('upper lower mixed')
        assert k1 == k2

    def test_unicode_punctuation_stripped(self):
        """En-dash, em-dash, and curly quotes are stripped like ASCII punctuation."""
        from escalation.dedupe import summary_dedupe_key

        # ASCII hyphen (Pd): "fused-memory connection timeout"
        k_ascii = summary_dedupe_key('fused-memory connection timeout')
        # En-dash U+2013 (Pd): "fused–memory connection timeout"
        k_en_dash = summary_dedupe_key('fused–memory connection timeout')
        # Em-dash U+2014 (Pd): "fused—memory connection timeout"
        k_em_dash = summary_dedupe_key('fused—memory connection timeout')
        # Curly double quotes U+201C/U+201D (Pi/Pf): "fused“memory” connection timeout"
        k_curly = summary_dedupe_key('fused“memory” connection timeout')

        # All four variants must produce the same normalised key
        assert k_ascii == k_en_dash == k_em_dash == k_curly
        assert k_ascii == ('fusedmemory', 'connection', 'timeout')

    def test_underscore_preserved_in_word_token(self):
        """Underscore (U+005F, category Pc) is NOT stripped — it is part of \\w.

        Deliberate divergence from the previous _PUNCT_TABLE implementation,
        which stripped all Unicode Pc characters (connector punctuation).
        The regex [^\\w\\s] keeps '_' because \\w includes [a-zA-Z0-9_].
        In practice escalation summaries do not use underscores, so the
        divergence is harmless; this test pins the chosen behaviour so it
        cannot drift silently.
        """
        from escalation.dedupe import summary_dedupe_key

        key = summary_dedupe_key('fused_memory connection timeout')
        # '_' is part of \w, so 'fused_memory' is kept as a single token
        assert key == ('fused_memory', 'connection', 'timeout')

    def test_unicode_symbols_stripped(self):
        """Unicode symbol categories (Sm/Sc/Sk/So) ARE stripped by [^\\w\\s].

        Deliberate divergence from the previous _PUNCT_TABLE implementation,
        which only stripped categories starting with 'P' (Pd/Pc/Pi/Pf/Po/Ps/Pe).
        The regex [^\\w\\s] also removes S* characters because none of Sm/Sc/Sk/So
        belong to \\w or \\s.  For example, the math-plus U+002B (Sm category) in
        'cpu+memory' is stripped, merging the two words into 'cpumemory'.
        This test pins the chosen behaviour across two symbol subcategories
        (Sm, Sc) using two examples so any future narrowing of the regex
        is caught immediately.
        """
        from escalation.dedupe import summary_dedupe_key

        # Sm (math symbol): '+' U+002B — stripped, adjacent words merge
        assert summary_dedupe_key('cpu+memory leak') == ('cpumemory', 'leak')

        # Sc (currency symbol): '$' U+0024 — stripped, no token merge here
        assert summary_dedupe_key('cost$ rises today') == ('cost', 'rises', 'today')


class TestEscalationDedupeFields:
    """Escalation dataclass gains dedupe_count, dedupe_children and
    dedupe_children_truncated fields."""

    def _make_min_escalation(self):
        from escalation.models import Escalation
        return Escalation(
            id='esc-1-1',
            task_id='1',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )

    def test_defaults_are_zero_and_empty(self):
        """New Escalation has dedupe_count==0 and dedupe_children==[]."""
        esc = self._make_min_escalation()
        assert esc.dedupe_count == 0
        assert esc.dedupe_children == []

    def test_round_trips_via_json(self):
        """dedupe_count and dedupe_children survive to_json / from_json."""
        from escalation.models import Escalation
        esc = self._make_min_escalation()
        esc.dedupe_count = 3
        esc.dedupe_children = ['esc-2-1', 'esc-3-1', 'esc-4-1']
        restored = Escalation.from_json(esc.to_json())
        assert restored.dedupe_count == 3
        assert restored.dedupe_children == ['esc-2-1', 'esc-3-1', 'esc-4-1']

    def test_from_dict_without_dedupe_keys_uses_defaults(self):
        """Old JSON on disk (without dedupe keys) loads with default values."""
        from escalation.models import Escalation
        old_dict = {
            'id': 'esc-1-1',
            'task_id': '1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'infra_issue',
            'summary': 'connection lost',
        }
        esc = Escalation.from_dict(old_dict)
        assert esc.dedupe_count == 0
        assert esc.dedupe_children == []

    def test_separate_instances_do_not_share_dedupe_children(self):
        """Two Escalation instances must NOT share the same dedupe_children list."""
        esc_a = self._make_min_escalation()
        esc_b = self._make_min_escalation()
        esc_a.dedupe_children.append('esc-2-1')
        assert esc_b.dedupe_children == [], (
            'dedupe_children must use default_factory, not a shared class-level list'
        )

    # --- dedupe_children_truncated: the growth bound's durable loss counter ---

    def test_dedupe_children_truncated_defaults_to_zero(self):
        """A new Escalation has shed nothing, so the counter starts at 0."""
        esc = self._make_min_escalation()
        assert esc.dedupe_children_truncated == 0

    def test_dedupe_children_truncated_round_trips_via_json(self):
        """The counter survives to_json / from_json.

        Without this the loss would be log-only: the TRUE provenance total is
        ``len(dedupe_children) + dedupe_children_truncated``, so a counter that
        did not persist would make the shed unassertable from the record
        (INV-8 / no-silent-fail-soft).
        """
        from escalation.models import Escalation
        esc = self._make_min_escalation()
        esc.dedupe_children_truncated = 7
        restored = Escalation.from_json(esc.to_json())
        assert restored.dedupe_children_truncated == 7

    def test_from_dict_without_truncated_key_uses_default(self):
        """Legacy on-disk JSON without the key loads with 0 — zero migration.

        Same contract as every field added since: ``from_dict`` filters on
        ``__dataclass_fields__``, so an absent key simply takes its default.
        """
        from escalation.models import Escalation
        old_dict = {
            'id': 'esc-1-1',
            'task_id': '1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'infra_issue',
            'summary': 'connection lost',
        }
        esc = Escalation.from_dict(old_dict)
        assert esc.dedupe_children_truncated == 0


class TestFindDedupeParent:
    """find_dedupe_parent() — scans live queue, returns oldest matching parent id or None."""

    def _make_infra_esc(
        self,
        esc_id: str,
        task_id: str = '1',
        summary: str = 'fused-memory connection timeout on port 8002',
        category: str = 'infra_issue',
    ):
        from escalation.models import Escalation
        return Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='implementer',
            severity='blocking',
            category=category,
            summary=summary,
        )

    def test_matching_parent_returns_parent_id(self, tmp_path):
        """(a) Same category + first-3-words match within window -> returns parent id."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1', task_id='42')
        queue.submit(parent)

        candidate = self._make_infra_esc(
            'esc-1-2',
            task_id='42',
            summary='fused-memory connection timeout on port 9999',
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result == 'esc-1-1'

    def test_different_category_returns_none(self, tmp_path):
        """(b) Different category (risk_identified vs infra_issue) -> None."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1', task_id='42', category='risk_identified')
        queue.submit(parent)

        candidate = self._make_infra_esc('esc-1-2', task_id='42', category='infra_issue')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result is None

    def test_different_summary_tokens_returns_none(self, tmp_path):
        """(c) Different first-3-words summary -> None."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1', summary='neo4j connection timeout on port 8002')
        queue.submit(parent)

        candidate = self._make_infra_esc('esc-1-2', summary='fused-memory connection timeout on port 8002')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result is None

    def test_outside_window_returns_none(self, tmp_path):
        """(d) candidate.timestamp - parent.timestamp > window_secs -> None."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        queue.submit(parent)

        # Move 'now' far into the future, beyond the 600s window
        now = datetime.now(UTC) + timedelta(seconds=700)
        config = DedupeConfig(infra_dedupe_window_secs=600.0)
        candidate = self._make_infra_esc('esc-1-2', summary='fused-memory connection timeout on port 9999')
        result = find_dedupe_parent(queue, candidate, config, now=now)
        assert result is None

    def test_resolved_parent_not_found(self, tmp_path):
        """(e) Already-resolved parent (in archive) -> None, since get_pending() skips archive."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        queue.submit(parent)
        queue.resolve('esc-1-1', 'fixed')

        candidate = self._make_infra_esc('esc-1-2')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result is None

    def test_cross_task_dedupe(self, tmp_path):
        """(f) Cross-task: parent has task_id='42', candidate has task_id='99' -> returns parent id."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-42-1', task_id='42')
        queue.submit(parent)

        candidate = self._make_infra_esc(
            'esc-99-1',
            task_id='99',
            summary='fused-memory connection timeout on port 9999',
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result == 'esc-42-1'

    def test_multiple_matching_returns_oldest(self, tmp_path):
        """(g) Multiple matching pending parents: returns the OLDEST by timestamp."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # Older parent (earlier timestamp)
        older = self._make_infra_esc('esc-1-1', task_id='1')
        older.timestamp = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
        queue.submit(older)

        # Newer parent (later timestamp, still within window)
        newer = self._make_infra_esc('esc-2-1', task_id='2')
        newer.timestamp = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        queue.submit(newer)

        candidate = self._make_infra_esc('esc-3-1', task_id='3',
                                          summary='fused-memory connection timeout on port 9999')
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result == 'esc-1-1'  # oldest

    def test_oldest_match_with_close_timestamps(self, tmp_path):
        """(h2) Three parents with sub-second gaps inserted out of order: oldest wins.

        Pins that selection uses timestamp comparison, not insertion order or
        filesystem iteration order — a gap the existing test_multiple_matching_returns_oldest
        does not unambiguously cover (it inserts in chronological order with 50s gaps).
        """
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        now = datetime.now(UTC)

        # Submit in non-chronological order: middle, oldest, newest
        middle = self._make_infra_esc('esc-2-1', task_id='2')
        middle.timestamp = (now - timedelta(milliseconds=300)).isoformat()
        queue.submit(middle)

        oldest = self._make_infra_esc('esc-1-1', task_id='1')
        oldest.timestamp = (now - timedelta(milliseconds=500)).isoformat()
        queue.submit(oldest)

        newest = self._make_infra_esc('esc-3-1', task_id='3')
        newest.timestamp = (now - timedelta(milliseconds=100)).isoformat()
        queue.submit(newest)

        candidate = self._make_infra_esc(
            'esc-4-1',
            task_id='4',
            summary='fused-memory connection timeout on port 9999',
        )
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now + timedelta(seconds=5))
        assert result == 'esc-1-1'  # oldest by timestamp, not insertion order

    def test_enabled_flag_not_checked_inside_find_dedupe_parent(self, tmp_path):
        """(h) infra_dedupe_enabled=False does NOT affect find_dedupe_parent itself.

        The gate lives in the server callers, not here. This test pins that contract:
        even with enabled=False, find_dedupe_parent still returns a match.
        """
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        queue.submit(parent)

        candidate = self._make_infra_esc('esc-1-2',
                                          summary='fused-memory connection timeout on port 9999')
        config = DedupeConfig(infra_dedupe_enabled=False)  # disabled
        now = datetime.now(UTC) + timedelta(seconds=5)
        # Despite enabled=False, find_dedupe_parent still finds a match
        result = find_dedupe_parent(queue, candidate, config, now=now)
        assert result == 'esc-1-1'

    def test_corrupt_timestamp_parent_folds_with_warning(self, tmp_path, caplog):
        """(i) Corrupt-ts parent + matching candidate → candidate FOLDS (result == parent id).

        Bug in current code: corrupt parent hits `except: continue` → matches empty
        → returns None → candidate is re-filed as a new escalation.

        Fix: parse_timestamp_or_warn with fallback=datetime.max → parent retained;
        a WARNING is emitted (loud-over-silent contract).
        """
        import logging
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1', task_id='42')
        parent.timestamp = 'not-a-timestamp'
        queue.submit(parent)

        candidate = self._make_infra_esc(
            'esc-1-2',
            task_id='43',
            summary='fused-memory connection timeout on port 9999',
        )
        now = datetime.now(UTC) + timedelta(seconds=5)

        with caplog.at_level(logging.WARNING, logger='shared.timestamps'):
            result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)

        # (a) Candidate must fold into the corrupt-ts parent (not re-filed as None)
        assert result == 'esc-1-1', (
            f'Expected corrupt-ts parent to be retained and candidate to fold; got result={result!r}'
        )

        # (b) At least one WARNING must mention the dedupe context
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'dedupe.find_dedupe_parent' in r.message
        ]
        assert warning_records, (
            f"Expected >=1 WARNING mentioning 'dedupe.find_dedupe_parent'; "
            f"got caplog.records: {[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_corrupt_parent_does_not_displace_valid_parent(self, tmp_path):
        """(j) Corrupt-ts parent + valid older parent (both matching) → valid older wins.

        Pins datetime.max for corrupt: corrupt sorts LAST, valid older sorts FIRST.
        If fallback were datetime.min instead, corrupt would sort FIRST and displace
        the valid parent — which is the doc-mandated error for oldest-as-canonical sites.
        """
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # Valid older parent
        valid_older = self._make_infra_esc('esc-valid-1', task_id='10')
        valid_older.timestamp = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
        queue.submit(valid_older)

        # Corrupt-ts parent (same category+key match)
        corrupt = self._make_infra_esc('esc-corrupt-1', task_id='11')
        corrupt.timestamp = 'not-a-timestamp'
        queue.submit(corrupt)

        candidate = self._make_infra_esc(
            'esc-candidate-1',
            task_id='12',
            summary='fused-memory connection timeout on port 9999',
        )
        now = datetime.now(UTC) + timedelta(seconds=5)

        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)

        # Valid older parent must win; corrupt must sort LAST (datetime.max)
        assert result == 'esc-valid-1', (
            f'Expected valid older parent to win; got result={result!r}. '
            f'If corrupt won, fallback is datetime.min (wrong) — must be datetime.max.'
        )


class TestComputeContentFingerprint:
    """compute_content_fingerprint() — pure deterministic fingerprint for recon dedup."""

    def test_identical_inputs_produce_identical_fingerprint(self):
        """(a) Identical inputs => identical fingerprint."""
        from escalation.dedupe import compute_content_fingerprint
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a', 'id-b'])
        fp2 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a', 'id-b'])
        assert fp1 == fp2

    def test_affected_ids_order_independent(self):
        """(b) affected_ids order-independent: [a,b] == [b,a]."""
        from escalation.dedupe import compute_content_fingerprint
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a', 'id-b'])
        fp2 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-b', 'id-a'])
        assert fp1 == fp2

    def test_different_affected_ids_differ(self):
        """(c) Different affected_ids => different fingerprint."""
        from escalation.dedupe import compute_content_fingerprint
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a'])
        fp2 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-b'])
        assert fp1 != fp2

    def test_different_finding_category_differs(self):
        """(d) Different finding_category => different fingerprint."""
        from escalation.dedupe import compute_content_fingerprint
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a'])
        fp2 = compute_content_fingerprint('recon_integrity_issue', 'missing_entity', ['id-a'])
        assert fp1 != fp2

    def test_different_escalation_category_differs(self):
        """(e) Different escalation_category => different fingerprint."""
        from escalation.dedupe import compute_content_fingerprint
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a'])
        fp2 = compute_content_fingerprint('infra_issue', 'entity_mismatch', ['id-a'])
        assert fp1 != fp2

    def test_empty_ids_uses_description_hash(self):
        """(f) Empty affected_ids falls back to normalised description hash."""
        from escalation.dedupe import compute_content_fingerprint
        # Same normalised description => same
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', [], 'entity X is missing')
        fp2 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', [], 'entity X is missing')
        assert fp1 == fp2
        # Different description => different
        fp3 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', [], 'entity Y is missing')
        assert fp1 != fp3

    def test_empty_ids_normalises_description(self):
        """(f cont.) Whitespace/case/punctuation-only differences normalise to same."""
        from escalation.dedupe import compute_content_fingerprint
        # These should all normalise to the same description
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', [], 'Entity X is missing')
        fp2 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', [], 'entity x is missing')
        fp3 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', [], 'entity x  is  missing')
        fp4 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', [], 'entity x is missing!!!')
        assert fp1 == fp2 == fp3 == fp4

    def test_non_empty_ids_ignores_description(self):
        """(g) With NON-empty affected_ids, description is ignored."""
        from escalation.dedupe import compute_content_fingerprint
        fp1 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a'], 'description one')
        fp2 = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-a'], 'COMPLETELY DIFFERENT description')
        assert fp1 == fp2

    def test_determinism_not_builtin_hash(self):
        """(h) Fingerprint is process-stable (sha256-based), not builtin hash().

        This proves cross-process determinism by checking against a hard-coded
        expected value derived from the known sha256 digest.
        The key identity tuple: ('recon_integrity_issue', 'entity_mismatch', 'id-sentinel')
        joined by \\x1f (unit separator).
        """
        import hashlib

        from escalation.dedupe import compute_content_fingerprint
        # Compute what the expected fingerprint should be:
        # identity = escalation_category \x1f finding_category \x1f body
        # body = sorted(['id-sentinel']) joined by \x1f = 'id-sentinel'
        raw = '\x1f'.join(['recon_integrity_issue', 'entity_mismatch', 'id-sentinel'])
        expected = hashlib.sha256(raw.encode()).hexdigest()

        fp = compute_content_fingerprint('recon_integrity_issue', 'entity_mismatch', ['id-sentinel'])
        assert fp == expected, (
            f'Expected sha256-based fingerprint {expected!r}, got {fp!r}. '
            'If you see a different value, the implementation may be using '
            'builtin hash() which is PYTHONHASHSEED-salted and non-deterministic.'
        )


class TestContentFingerprintKey:
    """content_fingerprint_key() adapter and DedupeConfig.key_fn field."""

    def _make_recon_esc(self, esc_id: str, fingerprint: str | None = None):
        from escalation.models import Escalation
        esc = Escalation(
            id=esc_id,
            task_id='42',
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary='Unresolved after remediation: entity mismatch',
        )
        esc.dedupe_fingerprint = fingerprint
        return esc

    def test_content_fingerprint_key_returns_fingerprint(self):
        """(1) content_fingerprint_key(esc) returns esc.dedupe_fingerprint."""
        from escalation.dedupe import content_fingerprint_key
        esc = self._make_recon_esc('esc-1-1', fingerprint='fp-abc123')
        assert content_fingerprint_key(esc) == 'fp-abc123'

    def test_content_fingerprint_key_returns_none_when_unset(self):
        """(1) content_fingerprint_key(esc) returns None when fingerprint is None."""
        from escalation.dedupe import content_fingerprint_key
        esc = self._make_recon_esc('esc-1-1', fingerprint=None)
        assert content_fingerprint_key(esc) is None

    def test_default_config_key_fn_is_none(self):
        """(2) DedupeConfig().key_fn is None (default)."""
        from escalation.dedupe import DedupeConfig
        cfg = DedupeConfig()
        assert cfg.key_fn is None

    def test_default_config_still_folds_infra_by_summary(self, tmp_path):
        """(3) Regression: default config (key_fn=None) folds two infra summaries sharing first 3 tokens."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = Escalation(
            id='esc-1-1', task_id='42', agent_role='impl', severity='blocking',
            category='infra_issue', summary='fused-memory connection timeout on port 8002',
        )
        queue.submit(parent)

        candidate = Escalation(
            id='esc-1-2', task_id='42', agent_role='impl', severity='blocking',
            category='infra_issue', summary='fused-memory connection timeout on port 9999',
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result == 'esc-1-1', 'Default config must still fold by first-3-token summary key'

    def test_content_key_folds_matching_fingerprints(self, tmp_path):
        """(4) find_dedupe_parent with content key folds when fingerprints match."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, content_fingerprint_key, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_recon_esc('esc-1-1', fingerprint='shared-fp')
        queue.submit(parent)

        candidate = self._make_recon_esc('esc-1-2', fingerprint='shared-fp')
        cfg = DedupeConfig(
            key_fn=content_fingerprint_key,
            infra_dedupe_categories=('recon_integrity_issue',),
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, cfg, now=now)
        assert result == 'esc-1-1'

    def test_content_key_does_not_fold_different_fingerprints(self, tmp_path):
        """(4) Different fingerprints do NOT fold."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, content_fingerprint_key, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_recon_esc('esc-1-1', fingerprint='fp-a')
        queue.submit(parent)

        candidate = self._make_recon_esc('esc-1-2', fingerprint='fp-b')
        cfg = DedupeConfig(
            key_fn=content_fingerprint_key,
            infra_dedupe_categories=('recon_integrity_issue',),
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, cfg, now=now)
        assert result is None

    def test_content_key_none_fingerprint_never_folds(self, tmp_path):
        """(4) Candidate with None dedupe_fingerprint never folds (empty-key guard)."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, content_fingerprint_key, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_recon_esc('esc-1-1', fingerprint='some-fp')
        queue.submit(parent)

        candidate = self._make_recon_esc('esc-1-2', fingerprint=None)
        cfg = DedupeConfig(
            key_fn=content_fingerprint_key,
            infra_dedupe_categories=('recon_integrity_issue',),
        )
        now = datetime.now(UTC) + timedelta(seconds=5)
        result = find_dedupe_parent(queue, candidate, cfg, now=now)
        assert result is None, 'None fingerprint must never fold into any parent'


class TestUnboundedWindow:
    """find_dedupe_parent with float('inf') window folds regardless of parent age."""

    def _make_recon_esc(self, esc_id: str, fingerprint: str = 'shared-fp', ts: str | None = None):
        from escalation.models import Escalation
        esc = Escalation(
            id=esc_id,
            task_id='42',
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary='Unresolved after remediation: entity mismatch',
        )
        esc.dedupe_fingerprint = fingerprint
        if ts is not None:
            esc.timestamp = ts
        return esc

    def test_inf_window_folds_regardless_of_age(self, tmp_path):
        """(1) With inf window, parent 10 days old still returns parent id."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, content_fingerprint_key, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # Parent with timestamp 10 days in the past
        ten_days_ago = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        parent = self._make_recon_esc('esc-1-1', ts=ten_days_ago)
        queue.submit(parent)

        candidate = self._make_recon_esc('esc-1-2')
        cfg = DedupeConfig(
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=('recon_integrity_issue',),
            key_fn=content_fingerprint_key,
        )
        result = find_dedupe_parent(queue, candidate, cfg)
        assert result == 'esc-1-1', (
            f'Inf window must fold regardless of age; got: {result}'
        )

    def test_finite_window_out_of_window_returns_none(self, tmp_path):
        """(2) Regression: default 600s window still returns None for out-of-window parent."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = Escalation(
            id='esc-1-1', task_id='42', agent_role='impl', severity='blocking',
            category='infra_issue', summary='fused-memory connection timeout on port 8002',
        )
        queue.submit(parent)

        # Move 'now' 700s into the future (outside the 600s window)
        now = datetime.now(UTC) + timedelta(seconds=700)
        candidate = Escalation(
            id='esc-1-2', task_id='42', agent_role='impl', severity='blocking',
            category='infra_issue', summary='fused-memory connection timeout on port 9999',
        )
        result = find_dedupe_parent(queue, candidate, DedupeConfig(), now=now)
        assert result is None, (
            f'Finite 600s window must return None for out-of-window parent; got: {result}'
        )


class TestDedupeConfigForRecon:
    """DedupeConfig.for_recon() — classmethod for recon integrity dedup config."""

    def _make_recon_esc(self, esc_id: str, fingerprint: str | None = 'shared-fp', ts: str | None = None):
        from escalation.models import Escalation
        esc = Escalation(
            id=esc_id,
            task_id='42',
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary='Unresolved after remediation: entity mismatch',
        )
        esc.dedupe_fingerprint = fingerprint
        if ts is not None:
            esc.timestamp = ts
        return esc

    def test_default_config_unchanged(self):
        """Regression: DedupeConfig() defaults are unchanged by for_recon classmethod."""
        from escalation.dedupe import DedupeConfig
        cfg = DedupeConfig()
        assert cfg.key_fn is None
        assert cfg.infra_dedupe_window_secs == 600.0
        assert cfg.infra_dedupe_categories == ('infra_issue',)
        assert cfg.infra_dedupe_enabled is True

    def test_for_recon_folds_matching_fingerprints_regardless_of_age(self, tmp_path):
        """(1) for_recon(): two recon_integrity_issue escalations with same fingerprint
        fold even when the parent is days old (content key + inf window + recon category)."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # Parent 3 days old
        three_days_ago = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        parent = self._make_recon_esc('esc-1-1', ts=three_days_ago)
        queue.submit(parent)

        candidate = self._make_recon_esc('esc-1-2')  # same fingerprint 'shared-fp'
        cfg = DedupeConfig.for_recon()
        result = find_dedupe_parent(queue, candidate, cfg)
        assert result == 'esc-1-1', (
            f'for_recon() must fold matching fingerprints regardless of age; got: {result}'
        )

    def test_for_recon_does_not_fold_different_fingerprints(self, tmp_path):
        """(2) Different fingerprints do NOT fold under for_recon()."""
        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_recon_esc('esc-1-1', fingerprint='fp-different')
        queue.submit(parent)

        candidate = self._make_recon_esc('esc-1-2', fingerprint='fp-another')
        cfg = DedupeConfig.for_recon()
        result = find_dedupe_parent(queue, candidate, cfg)
        assert result is None, (
            f'Different fingerprints must not fold under for_recon(); got: {result}'
        )

    def test_for_recon_handles_recon_category(self, tmp_path):
        """(3) for_recon() config handles recon_integrity_issue category via find_dedupe_parent."""
        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_recon_esc('esc-1-1')
        queue.submit(parent)

        candidate = self._make_recon_esc('esc-1-2')
        cfg = DedupeConfig.for_recon()
        # find_dedupe_parent should match since category is recon_integrity_issue
        # and fingerprints match
        result = find_dedupe_parent(queue, candidate, cfg)
        assert result == 'esc-1-1'


class TestDedupeConfigForGateBacklog:
    """DedupeConfig.for_gate_backlog() — sibling constructor for stale-gate dedup.

    The gate-backlog path (fused_memory.reconciliation.stage1_stall_detector's
    ``maybe_escalate_stalled_gate_backlog``) files one ``reconciliation_stale_gate_backlog``
    L1 per cycle a gate stays stalled.  Folding those into a single parent is what
    makes ``dedupe_count`` a recurrence signal instead of a constant 0.
    """

    _CATEGORY = 'reconciliation_stale_gate_backlog'

    def _make_gate_esc(
        self,
        esc_id: str,
        fingerprint: str | None = 'gate-fp',
        ts: str | None = None,
    ):
        from escalation.models import Escalation
        esc = Escalation(
            id=esc_id,
            task_id='645',
            agent_role='reconciler',
            severity='blocking',
            category=self._CATEGORY,
            summary='Gate task 645 has awaited a human decision since 2026-08-10T00:00:00+00:00',
            level=1,
        )
        esc.dedupe_fingerprint = fingerprint
        if ts is not None:
            esc.timestamp = ts
        return esc

    def _queue_files(self, queue):
        return sorted(queue.queue_dir.glob('esc-*.json'))

    # --- (a) config shape ---

    def test_for_gate_backlog_config_shape(self):
        """for_gate_backlog() enables dedup on the gate-backlog category with a content key.

        ``key_fn`` is ``gate_backlog_fingerprint_key`` — a superset of
        ``content_fingerprint_key`` that also recovers pre-stamp parents.  See
        that function's docstring for the live-queue measurement showing why the
        plain stamped-only adapter is insufficient here (it would mint a
        duplicate for every legacy record in the backlog).
        """
        from escalation.dedupe import DedupeConfig, gate_backlog_fingerprint_key

        cfg = DedupeConfig.for_gate_backlog()
        assert cfg.infra_dedupe_enabled is True
        assert cfg.infra_dedupe_categories == (self._CATEGORY,)
        assert cfg.key_fn is gate_backlog_fingerprint_key

    def test_for_gate_backlog_window_is_unbounded(self):
        """The window MUST be unbounded — a 300h-old gate must still fold.

        Asserted via ``math.isinf`` rather than an equality against a bounded
        default: any finite window silently mints a duplicate pending record and
        re-pins ``dedupe_count`` at 0, which is the exact bug this config exists
        to prevent.
        """
        import math

        from escalation.dedupe import DedupeConfig

        cfg = DedupeConfig.for_gate_backlog()
        assert math.isinf(cfg.infra_dedupe_window_secs), (
            'gate-backlog window must be unbounded; a bounded window re-pins '
            f'dedupe_count at 0 for long-rotting gates. got: {cfg.infra_dedupe_window_secs}'
        )
        assert cfg.infra_dedupe_window_secs > 0, 'window must be +inf, not -inf'

    # --- (b) regression guard on the sibling ---

    def test_for_recon_categories_unchanged(self):
        """REGRESSION: for_recon() is NOT widened by the new sibling.

        ``fused-memory/scripts/backfill_recon_escalations.py`` derives its
        eligible-collapse set (:168) and the complement defining
        ``blocking_pending`` (:290) from this exact tuple.  Admitting
        ``reconciliation_stale_gate_backlog`` here would silently change that
        one-shot operator script's collapse plan and its report semantics.
        """
        from escalation.dedupe import DedupeConfig

        assert DedupeConfig.for_recon().infra_dedupe_categories == ('recon_integrity_issue',)

    # --- (c) behavioural fold against a real queue ---

    def test_for_gate_backlog_folds_regardless_of_age(self, tmp_path):
        """A 400h-old parent still folds: same id, one pending record, dedupe_count==1."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # Backdate well past any bounded window (400h ≈ 16.7 days).
        old_ts = (datetime.now(UTC) - timedelta(hours=400)).isoformat()
        parent = self._make_gate_esc('esc-645-1', ts=old_ts)
        parent_id = queue.submit(parent)

        child = self._make_gate_esc('esc-645-2')
        child.summary = 'Gate task 645 has awaited a human decision since 2026-08-10T00:00:00+00:00 (452h)'
        result = submit_or_dedupe(queue, child, DedupeConfig.for_gate_backlog())

        assert result['status'] == 'dedup_skipped', (
            f'400h-old gate-backlog parent must still fold; got: {result}'
        )
        assert result['parent_id'] == parent_id
        pending = queue.get_pending()
        assert len(pending) == 1, (
            f'fold must leave exactly ONE pending record; got {len(pending)}'
        )
        reread = queue.get(parent_id)
        assert reread is not None
        assert reread.dedupe_count == 1

    # --- (d) negative ---

    def test_for_gate_backlog_different_fingerprint_does_not_fold(self, tmp_path):
        """A different fingerprint (different gate) gets its own record."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        parent = self._make_gate_esc('esc-645-1', fingerprint='gate-fp-a')
        submit_or_dedupe(queue, parent, DedupeConfig.for_gate_backlog())

        other = self._make_gate_esc('esc-646-1', fingerprint='gate-fp-b')
        result = submit_or_dedupe(queue, other, DedupeConfig.for_gate_backlog())

        assert result['status'] != 'dedup_skipped', (
            f'distinct gates must not fold into each other; got: {result}'
        )
        assert len(queue.get_pending()) == 2


class TestGateBacklogFingerprintKey:
    """gate_backlog_fingerprint_key(esc) — tolerates LEGACY unstamped parents.

    Every ``reconciliation_stale_gate_backlog`` record filed before the stamp
    landed carries ``dedupe_fingerprint: None`` — see the adapter's own
    docstring for the live-queue measurement that motivates this (kept in one
    place, since any point-in-time census goes stale as records fold and
    resolve).  With the plain ``content_fingerprint_key`` those parents key to
    None, ``find_dedupe_parent`` short-circuits, and the very first post-change
    cycle mints a SECOND pending record per stalled gate at ``dedupe_count 0``
    — the exact defect this task exists to remove.  This adapter recovers the
    parent's identity from its own ``detail`` so the backlog migrates itself
    with no operator step.
    """

    _CATEGORY = 'reconciliation_stale_gate_backlog'

    def _legacy_detail(self, project_id: str, task_id: str) -> str:
        """The pre-change detail shape as actually observed on the live queue.

        Note the PRE-3520 ``age_hours:`` key (not ``age_hours_at_filing:``) —
        the parser must not depend on anything past the first line.
        """
        return '\n'.join([
            f'project_id: {project_id}',
            'run_id: 0189ae49-63a8-46a8-a62a-124f1de71180',
            f'task_id: {task_id}',
            'gate_escalated_at: 2026-07-25T21:47:26.573155+00:00',
            'age_hours: 48.7',
            'title: DECISION: establish rotation convention',
        ])

    def _legacy_esc(
        self,
        esc_id: str,
        *,
        task_id: str = '166',
        detail: str | None = None,
        project_id: str = 'dark_factory',
        fingerprint: str | None = None,
        ts: str | None = None,
    ):
        from escalation.models import Escalation
        esc = Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='reconciliation-stage1',
            severity='blocking',
            category=self._CATEGORY,
            # The legacy PRE-3520 relative-age summary, as filed on disk.
            summary=f'Gate task {task_id} has awaited a human decision for 48.7h',
            detail=self._legacy_detail(project_id, str(task_id)) if detail is None else detail,
            level=1,
        )
        esc.dedupe_fingerprint = fingerprint
        if ts is not None:
            esc.timestamp = ts
        return esc

    # --- (1) stamped records take the fast path unchanged ---

    def test_stamped_fingerprint_passthrough(self):
        """A stamped record returns its fingerprint verbatim — no prose parsing."""
        from escalation.dedupe import gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-166-9', fingerprint='abc123')
        assert gate_backlog_fingerprint_key(esc) == 'abc123'

    # --- (2) the core fix: legacy recompute ---

    def test_legacy_unstamped_recomputes_true_identity(self):
        """An unstamped parent recovers exactly the fingerprint a stamp would carry."""
        from escalation.dedupe import compute_content_fingerprint, gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-166-1', task_id='166', project_id='dark_factory')
        expected = compute_content_fingerprint(
            'reconciliation_stale_gate_backlog', '', ['dark_factory:166'], ''
        )
        assert gate_backlog_fingerprint_key(esc) == expected

    # --- (3) mirror-exactness: parent key == child key ---

    def test_legacy_key_mirrors_new_child_stamp(self):
        """The recovered parent key EQUALS the key a new child for that gate carries.

        This is the single property that makes the migration fold work at all:
        the child's stamp is built by stage1_stall_detector as
        ``compute_content_fingerprint(category, '', [f'{project_id}:{task_id}'], '')``.
        """
        from escalation.dedupe import compute_content_fingerprint, gate_backlog_fingerprint_key

        project_id, task_id = 'autopilot_video', '645'
        legacy_parent = self._legacy_esc('esc-645-1', task_id=task_id, project_id=project_id)
        child_stamp = compute_content_fingerprint(
            'reconciliation_stale_gate_backlog', '', [f'{project_id}:{task_id}'], ''
        )
        assert gate_backlog_fingerprint_key(legacy_parent) == child_stamp

    # --- (4) cross-project safety (load-bearing regression guard) ---

    def test_same_task_id_different_projects_do_not_collide(self):
        """REGRESSION: the fallback keys on (category, project_id, task_id).

        The escalation queue is SHARED ACROSS PROJECTS and task ids are small
        per-project integers, so a ``(category, task_id)`` fallback would
        cross-fold two different projects' gates into one record and silently
        discard an escalation a human is waiting on.  See
        ``gate_backlog_fingerprint_key``'s docstring for the live-queue
        project/record census behind that claim; the absence of a collision in
        any given snapshot is a coincidence of that backlog, not an invariant,
        which is why this is pinned as a test rather than left to observation.
        """
        from escalation.dedupe import gate_backlog_fingerprint_key

        a = self._legacy_esc('esc-166-1', task_id='166', project_id='dark_factory')
        b = self._legacy_esc('esc-166-2', task_id='166', project_id='reify')
        assert gate_backlog_fingerprint_key(a) != gate_backlog_fingerprint_key(b), (
            'task_id 166 in dark_factory and in reify are DIFFERENT gates; folding '
            'them together would silently drop one project\'s escalation'
        )

    # --- (5) unparseable -> fail CLOSED ---

    @pytest.mark.parametrize(
        'detail',
        [
            pytest.param('run_id: x\ntask_id: 166', id='no_project_id_first_line'),
            pytest.param('', id='empty_detail'),
            pytest.param('  project_id: dark_factory', id='leading_whitespace_not_prefix'),
            pytest.param('projectid: dark_factory\n', id='misspelled_key'),
        ],
    )
    def test_unparseable_detail_returns_none(self, detail):
        """No recoverable project_id → None → find_dedupe_parent refuses to fold.

        Failing CLOSED (one duplicate record for that gate, visible and
        self-correcting) is strictly safer than failing OPEN (guessing an
        identity and folding unrelated gates, which destroys an escalation).
        """
        from escalation.dedupe import gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-166-1', detail=detail)
        assert gate_backlog_fingerprint_key(esc) is None

    def test_missing_task_id_returns_none(self):
        """A record with no task_id cannot be identified → never fold."""
        from escalation.dedupe import gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-x-1', task_id='166')
        esc.task_id = ''
        assert gate_backlog_fingerprint_key(esc) is None

    def test_recompute_reads_the_records_own_category(self):
        """REGRESSION: the recompute derives the category from the record, not a literal.

        ``submit_or_dedupe`` gates the candidate on
        ``config.infra_dedupe_categories`` and ``find_dedupe_parent`` skips any
        parent whose category differs, so every record reaching the recompute
        already carries the gate-backlog category — reading it off the record is
        correct by construction.  A hardcoded copy of the string would instead
        have to be kept in sync by hand with the emitter's
        ``_GATE_BACKLOG_ESCALATION_CATEGORY`` across a package boundary; a rename
        there would produce a key that can never match a new child's stamp,
        folding would stop, and duplicates would silently reappear.  This test is
        the thing that would catch that drift.
        """
        from escalation.dedupe import compute_content_fingerprint, gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-166-1', task_id='166', project_id='dark_factory')
        esc.category = 'renamed_gate_backlog_category'

        assert gate_backlog_fingerprint_key(esc) == compute_content_fingerprint(
            'renamed_gate_backlog_category', '', ['dark_factory:166'], ''
        )

    def test_none_detail_returns_none(self):
        """detail=None (defensive) must not raise — it fails closed like empty."""
        from escalation.dedupe import gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-166-1')
        # Deliberate type violation: `Escalation.detail` is declared `str`, but a
        # hand-edited or partially-migrated record on disk can deserialise with a
        # null detail, and the adapter must fail CLOSED rather than raise there.
        esc.detail = None  # pyright: ignore[reportAttributeAccessIssue]
        assert gate_backlog_fingerprint_key(esc) is None

    # --- (6) the literal token `None` is NOT special-cased ---

    def test_literal_none_project_is_mirrored_not_special_cased(self):
        """detail `project_id: None` keys on 'None:166', NOT to a None key.

        stage1_stall_detector writes the line as ``f'project_id: {project_id}'``
        and stamps children as ``f'{project_id}:{task_id}'``, so a filing made
        with ``project_id=None`` yields the literal ``'None:166'`` on BOTH
        sides.  Special-casing the token would break folding for exactly that
        case, so the fallback reproduces ``str(project_id)`` byte-for-byte.
        """
        from escalation.dedupe import compute_content_fingerprint, gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-166-1', task_id='166', project_id='None')
        expected = compute_content_fingerprint(
            'reconciliation_stale_gate_backlog', '', ['None:166'], ''
        )
        assert gate_backlog_fingerprint_key(esc) == expected

    def test_project_id_containing_a_space_is_taken_verbatim(self):
        """The line remainder is taken verbatim, NOT via `\\S+`.

        A `\\S+` match would silently truncate `my project` to `my`, producing a
        DIFFERENT key that could collide with another project's — converting a
        parse ambiguity into a wrong fold.  Verbatim can only fail to match.
        """
        from escalation.dedupe import compute_content_fingerprint, gate_backlog_fingerprint_key

        esc = self._legacy_esc('esc-166-1', task_id='166', project_id='my project')
        expected = compute_content_fingerprint(
            'reconciliation_stale_gate_backlog', '', ['my project:166'], ''
        )
        assert gate_backlog_fingerprint_key(esc) == expected

    # --- (7) wiring ---

    def test_for_gate_backlog_uses_the_tolerant_key(self):
        """DedupeConfig.for_gate_backlog() must route through the tolerant adapter."""
        from escalation.dedupe import DedupeConfig, gate_backlog_fingerprint_key

        assert DedupeConfig.for_gate_backlog().key_fn is gate_backlog_fingerprint_key

    # --- (8) end-to-end fold into a LEGACY parent ---

    def test_stamped_child_folds_into_legacy_parent(self, tmp_path):
        """The migration in one assertion: a stamped child folds into an unstamped parent."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, compute_content_fingerprint, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # A legacy parent exactly as filed pre-change: no fingerprint, ~400h old.
        old_ts = (datetime.now(UTC) - timedelta(hours=400)).isoformat()
        legacy = self._legacy_esc(
            'esc-166-1', task_id='166', project_id='dark_factory', ts=old_ts
        )
        assert legacy.dedupe_fingerprint is None
        parent_id = queue.submit(legacy)

        # The child a post-stamp cycle would file for that same gate.
        child = self._legacy_esc(
            'esc-166-2',
            task_id='166',
            project_id='dark_factory',
            fingerprint=compute_content_fingerprint(
                'reconciliation_stale_gate_backlog', '', ['dark_factory:166'], ''
            ),
        )
        result = submit_or_dedupe(queue, child, DedupeConfig.for_gate_backlog())

        assert result['status'] == 'dedup_skipped', (
            'a stamped child MUST fold into its legacy unstamped parent, else the '
            f'first post-change cycle mints a duplicate at dedupe_count 0; got: {result}'
        )
        assert result['parent_id'] == parent_id
        pending = queue.get_pending()
        assert len(pending) == 1, (
            f'fold must leave exactly ONE pending record; got {len(pending)}: '
            f'{[e.id for e in pending]}'
        )
        reread = queue.get(parent_id)
        assert reread is not None
        assert reread.dedupe_count == 1

    # --- (9) the recon path must not inherit gate-backlog prose parsing ---

    def test_for_recon_key_fn_unchanged(self):
        """REGRESSION: for_recon() still uses the plain stamped-only key adapter."""
        from escalation.dedupe import DedupeConfig, content_fingerprint_key

        assert DedupeConfig.for_recon().key_fn is content_fingerprint_key


class TestSubmitOrDedupe:
    """submit_or_dedupe(queue, esc, config, now=None) — gated orchestration wrapper."""

    def _make_recon_esc(self, esc_id: str, fingerprint: str | None = None, task_id: str = '42'):
        from escalation.models import Escalation
        esc = Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary='Unresolved after remediation: entity mismatch',
        )
        esc.dedupe_fingerprint = fingerprint
        return esc

    def _make_infra_esc(self, esc_id: str, task_id: str = '42', summary: str = 'fused-memory connection timeout on port 8002'):
        from escalation.models import Escalation
        return Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary=summary,
        )

    def _queue_files(self, queue):
        return sorted(queue.queue_dir.glob('esc-*.json'))

    # --- RECON config ---

    def test_recon_first_submit_queued(self, tmp_path):
        """(a) First recon_integrity_issue with stamped fingerprint => {'status':'queued'}."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        esc = self._make_recon_esc('esc-1-1', fingerprint='fp-abc')
        result = submit_or_dedupe(queue, esc, DedupeConfig.for_recon())

        assert result['status'] == 'queued'
        assert 'id' in result
        files = self._queue_files(queue)
        assert len(files) == 1

    def test_recon_same_fingerprint_dedupes_regardless_of_age(self, tmp_path):
        """(b) Second escalation with SAME fingerprint + old parent timestamp => dedup_skipped."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        # First: submit parent with old timestamp
        esc1 = self._make_recon_esc('esc-1-1', fingerprint='fp-abc')
        # Give parent an old timestamp (2 days ago)
        esc1.timestamp = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        result1 = submit_or_dedupe(queue, esc1, DedupeConfig.for_recon())
        parent_id = result1['id']
        assert result1['status'] == 'queued'

        # Second: different summary but SAME fingerprint -> should fold
        esc2 = self._make_recon_esc('esc-1-2', fingerprint='fp-abc')
        esc2.summary = 'Non-actionable integrity finding: entity mismatch (different run)'
        result2 = submit_or_dedupe(queue, esc2, DedupeConfig.for_recon())

        assert result2['status'] == 'dedup_skipped'
        assert result2['parent_id'] == parent_id
        assert 'child_id' in result2

        # Still one file
        assert len(self._queue_files(queue)) == 1

        # Parent dedupe_count == 1
        parent = Escalation.from_json(self._queue_files(queue)[0].read_text())
        assert parent.dedupe_count == 1

    def test_recon_triple_fold_preserves_fingerprint(self, tmp_path):
        """(c) Third fold: dedupe_count==2 AND parent.dedupe_fingerprint unchanged."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        esc1 = self._make_recon_esc('esc-1-1', fingerprint='fp-abc')
        submit_or_dedupe(queue, esc1, DedupeConfig.for_recon())

        esc2 = self._make_recon_esc('esc-1-2', fingerprint='fp-abc')
        submit_or_dedupe(queue, esc2, DedupeConfig.for_recon())

        esc3 = self._make_recon_esc('esc-1-3', fingerprint='fp-abc')
        result3 = submit_or_dedupe(queue, esc3, DedupeConfig.for_recon())

        assert result3['status'] == 'dedup_skipped'
        parent = Escalation.from_json(self._queue_files(queue)[0].read_text())
        assert parent.dedupe_count == 2
        # Fingerprint preserved across folds (invariant)
        assert parent.dedupe_fingerprint == 'fp-abc'

    def test_recon_different_fingerprint_queued_separately(self, tmp_path):
        """(d) Candidate with DIFFERENT fingerprint => queued (two files)."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        esc1 = self._make_recon_esc('esc-1-1', fingerprint='fp-abc')
        submit_or_dedupe(queue, esc1, DedupeConfig.for_recon())

        esc2 = self._make_recon_esc('esc-1-2', fingerprint='fp-xyz')
        result2 = submit_or_dedupe(queue, esc2, DedupeConfig.for_recon())

        assert result2['status'] == 'queued'
        assert len(self._queue_files(queue)) == 2

    def test_recon_none_fingerprint_queued(self, tmp_path):
        """(e) Candidate with dedupe_fingerprint=None => queued (never folds)."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        esc1 = self._make_recon_esc('esc-1-1', fingerprint='fp-abc')
        submit_or_dedupe(queue, esc1, DedupeConfig.for_recon())

        esc2 = self._make_recon_esc('esc-1-2', fingerprint=None)
        result2 = submit_or_dedupe(queue, esc2, DedupeConfig.for_recon())

        assert result2['status'] == 'queued', (
            f'None fingerprint must never fold; got: {result2}'
        )
        assert len(self._queue_files(queue)) == 2

    # --- INFRA regression ---

    def test_infra_default_config_dedupes_by_summary(self, tmp_path):
        """INFRA regression: two infra_issue with same first-3-token summary fold."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        esc1 = self._make_infra_esc('esc-1-1')
        result1 = submit_or_dedupe(queue, esc1, DedupeConfig())
        assert result1['status'] == 'queued'

        esc2 = self._make_infra_esc('esc-1-2', summary='Fused-memory  CONNECTION timeout!')
        result2 = submit_or_dedupe(queue, esc2, DedupeConfig())

        assert result2['status'] == 'dedup_skipped'
        assert result2['parent_id'] == result1['id']
        assert len(self._queue_files(queue)) == 1
        parent = Escalation.from_json(self._queue_files(queue)[0].read_text())
        assert parent.dedupe_count == 1

    # --- Gate conditions ---

    def test_gate_disabled_skips_dedup(self, tmp_path):
        """Gate: infra_dedupe_enabled=False => always queued, no fold even with same-key parent."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        cfg = DedupeConfig(infra_dedupe_enabled=False)
        queue = EscalationQueue(tmp_path / 'esc')

        esc1 = self._make_infra_esc('esc-1-1')
        r1 = submit_or_dedupe(queue, esc1, cfg)
        assert r1['status'] == 'queued'

        # Same-key summary — would fold under the default enabled config.
        esc2 = self._make_infra_esc('esc-1-2', summary='Fused-memory  CONNECTION timeout!')
        r2 = submit_or_dedupe(queue, esc2, cfg)

        assert r2['status'] == 'queued', (
            f'Disabled gate must never fold; got: {r2}'
        )
        assert len(self._queue_files(queue)) == 2

    def test_gate_wrong_category_skips_dedup(self, tmp_path):
        """Gate: category not in infra_dedupe_categories => always queued, no fold."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        # Default config gates on 'infra_issue' only; use a design_concern escalation.
        cfg = DedupeConfig()  # infra_dedupe_categories=('infra_issue',)
        queue = EscalationQueue(tmp_path / 'esc')

        def _make_design(esc_id: str):
            return Escalation(
                id=esc_id,
                task_id='42',
                agent_role='implementer',
                severity='info',
                category='design_concern',
                summary='fused-memory connection timeout on port 8002',
            )

        esc1 = _make_design('esc-1-1')
        r1 = submit_or_dedupe(queue, esc1, cfg)
        assert r1['status'] == 'queued'

        # Same summary tokens — would fold if category matched.
        esc2 = _make_design('esc-1-2')
        r2 = submit_or_dedupe(queue, esc2, cfg)

        assert r2['status'] == 'queued', (
            f'Out-of-scope category must never fold; got: {r2}'
        )
        assert len(self._queue_files(queue)) == 2

    # --- TOCTOU ---

    def test_toctou_falls_through_to_submit(self, tmp_path, monkeypatch):
        """TOCTOU: find_dedupe_parent resolves parent before returning => escalation queued."""
        import escalation.dedupe as dedupe_module
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')

        esc1 = self._make_infra_esc('esc-1-1')
        submit_or_dedupe(queue, esc1, DedupeConfig())

        _original_find = dedupe_module.find_dedupe_parent

        def _racing_find(q, esc, cfg, now=None):
            result = _original_find(q, esc, cfg, now=now)
            if result is not None:
                q.resolve(result, resolution='raced')
            return result  # still returns the id — simulating stale read

        monkeypatch.setattr(dedupe_module, 'find_dedupe_parent', _racing_find)

        esc2 = self._make_infra_esc('esc-1-2', summary='Fused-memory  CONNECTION timeout!')
        result2 = submit_or_dedupe(queue, esc2, DedupeConfig())

        assert result2['status'] == 'queued', (
            f'TOCTOU: must fall through to submit; got: {result2}'
        )
        files = self._queue_files(queue)
        assert len(files) == 1  # only the new esc (parent was archived)
        assert result2['id'] == files[0].stem


class TestEscalationDedupeFingerprint:
    """Escalation.dedupe_fingerprint field — added for content-fingerprint dedup (A7a)."""

    def _make_min_escalation(self):
        from escalation.models import Escalation
        return Escalation(
            id='esc-1-1',
            task_id='1',
            agent_role='implementer',
            severity='blocking',
            category='recon_integrity_issue',
            summary='Unresolved after remediation: entity mismatch',
        )

    def test_default_is_none(self):
        """(a) A freshly constructed Escalation has dedupe_fingerprint is None by default."""
        esc = self._make_min_escalation()
        assert esc.dedupe_fingerprint is None

    def test_round_trips_via_json(self):
        """(b) When set to a string, dedupe_fingerprint survives to_json/from_json round-trip."""
        from escalation.models import Escalation
        esc = self._make_min_escalation()
        esc.dedupe_fingerprint = 'abc123deadbeef'
        restored = Escalation.from_json(esc.to_json())
        assert restored.dedupe_fingerprint == 'abc123deadbeef'

    def test_from_dict_without_key_defaults_to_none(self):
        """(c) from_dict on a legacy dict WITHOUT the key defaults to None."""
        from escalation.models import Escalation
        old_dict = {
            'id': 'esc-1-1',
            'task_id': '1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'recon_integrity_issue',
            'summary': 'Unresolved after remediation: entity mismatch',
        }
        esc = Escalation.from_dict(old_dict)
        assert esc.dedupe_fingerprint is None

    def test_separate_instances_do_not_share_state(self):
        """(d) Two separate instances do not share the dedupe_fingerprint field."""
        esc_a = self._make_min_escalation()
        esc_b = self._make_min_escalation()
        esc_a.dedupe_fingerprint = 'fingerprint-for-a'
        assert esc_b.dedupe_fingerprint is None, (
            'Setting dedupe_fingerprint on one instance must not affect another'
        )


class TestCrossLevelDedupeIsolation:
    """find_dedupe_parent never folds a candidate into a parent at a DIFFERENT level.

    Task 3236: once ``escalate_blocker(level=1)`` became filable, a steward's
    level-1 re-escalation of an ``infra_issue`` — the single category in
    ``DedupeConfig.infra_dedupe_categories`` — would otherwise fold straight
    back into the pending level-0 record it was handling, and be swallowed
    again by a new mechanism.  An L1 carries ``severity='blocking'``, so it
    does NOT take the born-at-L2 dedupe bypass in ``server._submit_or_dedupe``
    and does route through this matcher.

    Cross-level folding is never correct regardless of this task: the levels
    have different consumers by contract (models.py module header), so
    collapsing an L1 into an L0 parent hands the record to the wrong consumer.
    """

    def _make_infra_esc(self, esc_id: str, level: int = 0, task_id: str = '42'):
        from escalation.models import Escalation
        return Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='steward',
            severity='blocking',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
            level=level,
        )

    def test_l1_candidate_does_not_fold_into_l0_parent(self, tmp_path):
        """Same category + same summary key, different level -> no parent match."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        queue.submit(self._make_infra_esc('esc-42-1', level=0))

        candidate = self._make_infra_esc('esc-42-2', level=1)
        now = datetime.now(UTC) + timedelta(seconds=5)

        assert find_dedupe_parent(queue, candidate, DedupeConfig(), now=now) is None

    def test_l0_candidate_does_not_fold_into_l1_parent(self, tmp_path):
        """The converse direction is equally isolated (L0 candidate, L1 parent)."""
        from datetime import UTC, datetime, timedelta

        from escalation.dedupe import DedupeConfig, find_dedupe_parent
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        queue.submit(self._make_infra_esc('esc-42-1', level=1))

        candidate = self._make_infra_esc('esc-42-2', level=0)
        now = datetime.now(UTC) + timedelta(seconds=5)

        assert find_dedupe_parent(queue, candidate, DedupeConfig(), now=now) is None

    def test_submit_or_dedupe_queues_l1_beside_pending_l0(self, tmp_path):
        """End-to-end: the L1 gets its OWN record, not a dedup_skipped fold.

        This is the swallow the level=1 fix would otherwise open.
        """
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-42-1', level=0)
        queue.submit(parent)

        candidate = self._make_infra_esc('esc-42-2', level=1)
        result = submit_or_dedupe(queue, candidate, DedupeConfig())

        assert result['status'] == 'queued', f'Expected a fresh record, got: {result}'
        assert result['id'] == 'esc-42-2', f'Expected a fresh id, got: {result}'
        persisted = queue.get('esc-42-2')
        assert persisted is not None
        assert persisted.level == 1
        # The L0 parent must be untouched — no child attached to it.
        l0 = queue.get('esc-42-1')
        assert l0 is not None
        assert l0.dedupe_count == 0, f'L0 parent absorbed the L1: {l0.dedupe_children}'

    def test_same_level_infra_candidates_still_fold(self, tmp_path):
        """CONTROL: existing same-level dedupe behaviour is provably unbroken."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        queue.submit(self._make_infra_esc('esc-42-1', level=0))

        candidate = self._make_infra_esc('esc-42-2', level=0)
        result = submit_or_dedupe(queue, candidate, DedupeConfig())

        assert result['status'] == 'dedup_skipped', f'Expected a fold, got: {result}'
        assert result['parent_id'] == 'esc-42-1'
        assert result['child_id'] == 'esc-42-2'

    def test_same_level_l1_candidates_still_fold(self, tmp_path):
        """CONTROL: the new condition is level EQUALITY, not 'level 0 only'."""
        from escalation.dedupe import DedupeConfig, submit_or_dedupe
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        queue.submit(self._make_infra_esc('esc-42-1', level=1))

        candidate = self._make_infra_esc('esc-42-2', level=1)
        result = submit_or_dedupe(queue, candidate, DedupeConfig())

        assert result['status'] == 'dedup_skipped', f'Expected a fold, got: {result}'
        assert result['parent_id'] == 'esc-42-1'


class TestNormalisationLiftedToCanonical:
    """The casefold/strip/collapse pipeline lives in ONE place: escalation.canonical.

    dedupe.py had THREE uses of the two regexes — ``_normalize_description``
    (feeding ``compute_content_fingerprint``) and ``summary_dedupe_key``'s direct
    ``_NON_WORD_PATTERN.sub`` (feeding ``find_dedupe_parent``).  Both call sites
    now delegate to the lifted helper under an explicitly-pinned
    ``punctuation='strip'`` policy, and the tests below pin that delegation by
    OUTPUT EQUALITY against ``canonical_text`` — not by the absence of any
    particular private name.  The regexes themselves live on quite legitimately
    inside ``escalation.canonical``, their one home, so their presence somewhere
    was never the property worth asserting; DRIFT is.  A second implementation of
    the pipeline is caught the moment it diverges by a single character: the
    delegation-equality assertions and the digest/tuple characterisation pins
    below then fail loudly (INV-5).  A copy that does not diverge is not the risk
    INV-5 exists to catch.

    The digests and key tuples below are CHARACTERISATION pins, not aspirations:
    every value was obtained by RUNNING the pre-lift implementation.  They are
    load-bearing because both outputs are already persisted across the live
    corpus — a fingerprint digest that changes silently un-dedupes every recon
    finding already on disk, and a ``summary_dedupe_key`` tuple that changes
    silently re-partitions every dedupe cluster already keyed fleet-wide.
    """

    # (escalation_category, finding_category, description) -> sha256 hex digest,
    # computed by running the PRE-LIFT dedupe implementation.
    _FINGERPRINT_REFERENCE = {
        ('infra_issue', 'flaky-test', 'Fused-memory  CONNECTION timeout!'):
            '29d2adc399cddb4369e5ad754c054e0f6f595f43ba375846c3f6741e40f59ad4',
        ('risk_identified', 'perf', 'cpu+memory leak in the sweep loop'):
            '393e47310ab5186a41cd3554e656e52c7686aa7bb28e22dd3e61ac95896d52fc',
        ('design_concern', 'coupling', '  Watcher lease STOLEN.  '):
            '26b8b9918444def709c36684a7e1f1edce11a41bec7dab1169c04cf2278d460f',
        ('cleanup_needed', 'dead-code', 'starvation:2370:persistent-lock-contention'):
            '45538de967982d58f3aafd64bcfae927f51a41cbb445c2115134b927cb076dd2',
    }

    def test_normalize_description_delegates_to_the_strip_policy(self):
        """_normalize_description IS canonical_text(..., punctuation='strip')."""
        from escalation.canonical import canonical_text
        from escalation.dedupe import _normalize_description

        for text in [
            'a.b, c',
            'Fused-memory  CONNECTION timeout!',
            '  Watcher lease STOLEN.  ',
            'risk:3184',
            '',
            '::',
        ]:
            assert _normalize_description(text) == canonical_text(
                text, punctuation='strip'
            ), f'delegation diverged for {text!r}'

    def test_normalize_description_keeps_its_measured_outputs(self):
        """Characterisation of the pipeline itself, independent of the delegation."""
        from escalation.dedupe import _normalize_description

        assert _normalize_description('a.b, c') == 'ab c'
        assert _normalize_description('Fused-memory  CONNECTION timeout!') == (
            'fusedmemory connection timeout'
        )
        assert _normalize_description('  Watcher lease STOLEN.  ') == 'watcher lease stolen'
        # STRIP, not separator: the fingerprint policy must stay deletion-flavoured.
        assert _normalize_description('risk:3184') == 'risk3184'

    def test_content_fingerprint_digests_are_byte_identical(self):
        """The digests already on disk must not move.

        If this fails, the strip policy was changed (or the delegation is not
        byte-identical) and every already-fingerprinted recon finding has stopped
        matching its own past self — a large, invisible regression.
        """
        from escalation.dedupe import compute_content_fingerprint

        for (esc_cat, find_cat, description), expected in self._FINGERPRINT_REFERENCE.items():
            actual = compute_content_fingerprint(esc_cat, find_cat, [], description)
            assert actual == expected, (
                f'fingerprint digest changed for {description!r}: '
                f'{actual} != {expected} — the live recon corpus would silently un-dedupe'
            )

    def test_content_fingerprint_ignores_description_when_affected_ids_present(self):
        """Unchanged contract, re-pinned because the lift touched its only helper."""
        from escalation.dedupe import compute_content_fingerprint

        with_ids = compute_content_fingerprint(
            'infra_issue', 'flaky-test', ['task-1', 'task-2'], 'one description'
        )
        other_description = compute_content_fingerprint(
            'infra_issue', 'flaky-test', ['task-1', 'task-2'],
            'a COMPLETELY different description!!',
        )
        assert with_ids == other_description
        assert with_ids == '3d46b1b1c93febe00abf8d28be40c8c30db39d0a6136e04dce8532a5228666f9'

    def test_summary_dedupe_key_matches_its_documented_examples(self):
        """The five docstring doctests, promoted to real assertions.

        ``summary_dedupe_key`` feeds ``find_dedupe_parent`` and its tuples are
        persisted fleet-wide, so the rewire onto the lifted helper has to be
        byte-identical.  These cases already cover the interesting shapes:
        internal punctuation, a symbol join, a doubled space, a trailing '!',
        the empty string, and the >3-token truncation.
        """
        from escalation.dedupe import summary_dedupe_key

        assert summary_dedupe_key('Fused-memory  CONNECTION timeout!') == (
            'fusedmemory', 'connection', 'timeout',
        )
        assert summary_dedupe_key('fused-memory connection timeout on port 8002') == (
            'fusedmemory', 'connection', 'timeout',
        )
        assert summary_dedupe_key('lost link') == ('lost', 'link')
        assert summary_dedupe_key('') == ()
        assert summary_dedupe_key('cpu+memory leak') == ('cpumemory', 'leak')

    def test_summary_dedupe_key_equals_the_strip_policy_expression(self):
        """The rewire is exactly ``canonical_text(s, 'strip').split()[:3]``.

        Verified during planning over all 2796 real summaries in the live queue
        (0 mismatches): the helper's extra whitespace-collapse-and-strip is
        absorbed by the subsequent ``.split()``.
        """
        from escalation.canonical import canonical_text
        from escalation.dedupe import summary_dedupe_key

        for summary in [
            'Fused-memory  CONNECTION timeout!',
            'cpu+memory leak',
            '  leading and trailing  ',
            '\t\n',
            '::',
            'one two three four five',
            'ロック競合 が 発生',
        ]:
            assert summary_dedupe_key(summary) == tuple(
                canonical_text(summary, punctuation='strip').split()[:3]
            ), f'summary_dedupe_key diverged from the lifted helper for {summary!r}'
