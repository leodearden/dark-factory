"""Tests for scripts/run_judge_ofat_pilot.py's ``--results-dir`` loader hardening.

Task 3743: the missing-dir / malformed-JSON hardening that task 3632 added to
``scripts/run_fable_trial_v2_campaign.py``'s ``_load_results_from_dir`` (an
``is_dir()`` check that exits naming the path, and a per-file try/except that
exits naming the offending file rather than raising a bare traceback) now
lives once, in ``scripts/_eval_results_io.py``, and both drivers import it.
This file is the equivalent coverage for THIS driver, mirroring
``scripts/tests/test_run_fable_trial_v2_campaign.py``'s
``test_nonexistent_results_dir_is_diagnosed_as_a_path_problem`` and
``test_malformed_result_json_names_the_offending_file`` /
``test_non_dict_result_json_names_the_offending_file`` — before the
extraction, this driver's copy of the loader had NEITHER of these checks, even
though ``--results-dir`` defaults to the same shared packaged results dir
(``orchestrator/src/orchestrator/evals/results/``) that the campaign driver
reads, where a single truncated or partial write from some other eval run is
the realistic failure.

The bare ``import run_judge_ofat_pilot`` resolves off
``scripts/tests/conftest.py``'s sys.path insertion, matching every other
driver test in this directory (``test_run_fable_trial_v2_campaign.py``).
"""
from __future__ import annotations

import json

import pytest
import run_judge_ofat_pilot as mod


def test_nonexistent_results_dir_is_diagnosed_as_a_path_problem(tmp_path):
    """A typo'd --results-dir must not fall through to a bare/misleading error.

    ``Path.glob`` returns EMPTY for a nonexistent dir rather than raising, so
    pre-extraction this driver's own loader copy (unlike the campaign driver's
    hardened one) had no ``is_dir()`` guard at all and would have proceeded to
    analyze zero results instead of naming the bad path.
    """
    missing = tmp_path / 'no' / 'such' / 'dir'
    out = tmp_path / 'report.md'

    with pytest.raises(SystemExit) as exc:
        mod.main(['--analyze-only', '--results-dir', str(missing), '--out', str(out)])

    message = str(exc.value)
    assert str(missing) in message
    assert 'not found' in message
    assert not out.exists()


def test_malformed_result_json_names_the_offending_file(tmp_path):
    """One truncated cell must name ITSELF, not raise a bare traceback.

    ``--results-dir`` defaults to the shared packaged results dir (many
    campaigns' worth of cells), so a single partial write there is the
    realistic failure this guards against.
    """
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    truncated = results_dir / 'truncated.json'
    truncated.write_text('{"task_id": "fix1", "config_nam')
    out = tmp_path / 'report.md'

    with pytest.raises(SystemExit) as exc:
        mod.main(['--analyze-only', '--results-dir', str(results_dir), '--out', str(out)])

    assert str(truncated) in str(exc.value)


def test_non_dict_result_json_names_the_offending_file(tmp_path):
    """A JSON list/string payload is named too, not surfaced as a bare AttributeError."""
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    stray = results_dir / 'a_list.json'
    stray.write_text(json.dumps([1, 2, 3]))
    out = tmp_path / 'report.md'

    with pytest.raises(SystemExit) as exc:
        mod.main(['--analyze-only', '--results-dir', str(results_dir), '--out', str(out)])

    assert str(stray) in str(exc.value)
