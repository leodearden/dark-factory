#!/usr/bin/env python3
"""Startup-completion artifact probe — task 3324 (substrate validation for the
two-regime watchdog startup grace, PRD `plans/server-side-api-error-handling-prd.md`,
consumer task 3326 / contract C5).

WHAT THIS ANSWERS
-----------------
The watchdog's startup regime currently kills an invocation that has produced no
assistant turn by ``startup_grace_secs`` (120s).  C5 wants a SECOND, longer grace
that applies only once the CLI has demonstrably *finished starting up* and is
merely waiting on the server (e.g. a 529 retry cycle).  That needs a predicate
answering: **"has the CLI completed startup, even though turn 1 has not landed?"**

This probe measures which on-disk artifacts actually exist, at which offsets, for
a healthy invocation versus each PRD-named wedge shape — so the predicate is
chosen from observed evidence rather than guessed.

MODES
-----
``healthy``       spawn the real ``claude --print`` (haiku, one-word prompt,
                  ~$0.002) through a production-shaped ``TaskConfigDir``.
``build_wedge``   spawn a stub wrapper that emits from-source-build stderr and
                  never execs the CLI (the "wrapper still compiling" wedge).
``uv_wedge``      spawn a stub wrapper that emits ``uv`` resolution stderr and
                  never execs the CLI.
``mcp_wedge``     spawn the real ``claude`` with ``--mcp-config`` pointing at a
                  stub stdio server that accepts the connection and then never
                  answers ``initialize``.
``replay``        read-only: run the SAME sampler against an existing on-disk
                  ``CLAUDE_CONFIG_DIR`` (the pre-2 fallback when a live spawn is
                  not possible in a given dispatch).

OUTPUT
------
One redacted JSON observation object per sample, JSONL, to stdout (or ``--out``).
Redaction happens at CAPTURE time, not at curation time: file *contents* are
never inlined (only path/kind/size metadata), and transcript records are reduced
to a fixed safe field projection that excludes all prompt/response text.  The
healthy observation is taken from a config dir that really does hold a live OAuth
token in ``.credentials.json``, so this is load-bearing, not hygiene theatre.

The scrub rewrites dict KEYS as well as values, and guarantees cleanliness of the
JSON-ENCODED form rather than the raw one — the encoding is what the gate scans,
and JSON escaping can manufacture a credential-shaped run that raw scrubbing
never sees.  If a gated value still fails verification after scrubbing, it
degrades to a minimal ``redaction_failed`` row plus a stderr WARNING rather than
raising: losing one value is bounded, whereas raising out of the sampling loop
would discard an entire already-paid-for live capture.  The degraded row is
shaped like whatever was gated — identity fields for an observation, the scalar
exit fields for the run's exit provenance — because that provenance is stamped
onto every observation of the run.  A degraded observation also carries the
probe-authored ``captured_at`` / ``session_id`` (shape-validated), so a row in an
appended-to ``--out`` file can still be attributed to the run that emitted it.

USAGE
-----
    uv run --project shared python tests/startup_completion_probe.py \
        --mode healthy --out /tmp/healthy.jsonl

See `docs/startup-completion-artifact-matrix.md` for the resulting matrix and the
chosen predicate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal, NamedTuple

# Allow execution as a bare script (``python tests/startup_completion_probe.py``)
# as well as import from a pytest run, mirroring shared/tests/conftest.py.
_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / 'src'
for _p in (str(_TESTS_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import startup_completion_fixtures as _scf  # noqa: E402  (isort: after src bootstrap)
from _oauth_accounts import (  # noqa: E402  (isort: after tests-dir bootstrap)
    ALL_TOKEN_LETTERS,
    first_available_token,
)

from shared.cli_invoke import (  # noqa: E402
    _resolve_transcript_path,
    count_transcript_turns,
    read_transcript_records,
)
from shared.config_dir import (  # noqa: E402
    CONFIG_DIR_PREFIX,
    TaskConfigDir,
    sweep_stale_pid_dirs,
)

MODES = ('healthy', 'build_wedge', 'uv_wedge', 'mcp_wedge', 'replay')

#: The closed set of ``sample_kind`` values the samplers actually emit — the
#: four ``_take`` call sites in :func:`run_live_probe`, the two
#: ``pre_first_token`` relabels, and :func:`run_replay_probe`.  Closed because
#: :func:`_poisoned_observation` carries ``sample_kind`` through ONLY when it is
#: a member: a row whose whole claim is "clean by construction" cannot carry an
#: arbitrary string out of an observation that just failed redaction.
SAMPLE_KINDS = (
    'scheduled',
    'first_token',
    'pre_first_token',
    'pre_first_token_candidate',
    'after_exit',
    'deadline',
    'replay',
)

#: Wedge-shape slug recorded on each observation, keyed by probe mode.  ``None``
#: for the healthy/replay regimes.  These slugs are the PRD's names and are the
#: same closed set the corpus rows use.
MODE_WEDGE_SHAPE: dict[str, str | None] = {
    'healthy': None,
    'build_wedge': 'from_source_build',
    'uv_wedge': 'uv_resolving',
    'mcp_wedge': 'mcp_init_hang',
    'replay': None,
}

#: Full-sample offsets (seconds since spawn).  Recorded as PROVENANCE only — no
#: test asserts a wall-clock threshold, because none is achievable (host load,
#: SessionStart hook duration, MCP server count and FS cache all move these).
DEFAULT_SAMPLE_OFFSETS: tuple[float, ...] = (0.25, 1.0, 2.0, 5.0, 15.0, 30.0)

#: Fine polling grid used to catch the pre-first-token boundary sample.
_FINE_TICK_SECS = 0.2

#: Config-dir subtrees collapsed to a single ``pruned_descendants`` count instead
#: of one entry per file.  ``plugins/marketplaces`` is a full git CLONE (hundreds
#: of loose objects) that the CLI populates on first run and that has nothing to
#: do with startup completion; capturing it verbatim inflated the raw capture to
#: 630 KB of noise.  ``projects/`` — where the transcript lives — is deliberately
#: NEVER pruned.
DEFAULT_PRUNE_PREFIXES: tuple[str, ...] = ('plugins/marketplaces', 'backups')

#: Max scrubbed stderr characters retained per observation (provenance only).
_STDERR_TAIL_CHARS = 600

#: "Never sampled" sentinel, distinct from a real ``None`` observation.
_UNSET = object()

# ---------------------------------------------------------------------------
# Redaction (capture-time gate)
# ---------------------------------------------------------------------------

#: Transcript-record fields the probe is allowed to keep.  Everything else —
#: crucially every prompt/response/tool-payload text field — is dropped.  Keep
#: this an ALLOW-list: a deny-list silently leaks whatever the next CLI version
#: adds.
_RECORD_TYPE_KEYS = ('type', 'subtype', 'operation', 'isMeta', 'isSidechain')

#: The credential pattern set and the ``.credentials.json`` filename set are
#: owned by ``startup_completion_fixtures`` (the committed assertion form) and
#: imported here so the CAPTURE-time gate and the COMMIT-time assertion can
#: never drift apart.  Widening one automatically widens the other.
_CREDENTIAL_PATTERNS = _scf._CREDENTIAL_PATTERNS
_NAMED_CREDENTIAL_PATTERNS = _scf.NAMED_CREDENTIAL_PATTERNS
_GENERIC_CREDENTIAL_PATTERNS = _scf.GENERIC_CREDENTIAL_PATTERNS
CREDENTIAL_FILENAMES = _scf.CREDENTIAL_FILENAMES


def scan_for_credential_material(
    text: str, patterns: tuple[tuple[str, str], ...] = _CREDENTIAL_PATTERNS
) -> tuple[str, int] | None:
    """Return ``(pattern_name, offset)`` of the first credential-shaped match, else None.

    The non-raising form of ``startup_completion_fixtures.assert_no_credential_material``
    over the same pattern set, for call sites that need to substitute rather than
    fail (see :func:`scrub_credential_material`).  *patterns* narrows the scan to
    one pattern class; it defaults to all of them.
    """
    for name, pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return (name, match.start())
    return None


def _scrub_text(text: str, patterns: tuple[tuple[str, str], ...]) -> str:
    """Substitute every *patterns* run in *text* with ``<redacted>``.

    The single substitution definition shared by :func:`_scrub_value`'s string
    LEAF branch and its string KEY branch, so the two can never drift into
    scrubbing the same material differently.
    """
    out = text
    for _name, pattern in patterns:
        out = re.sub(pattern + r'\S*', '<redacted>', out)
    return out


def _encodes_clean(value: Any, patterns: tuple[tuple[str, str], ...]) -> bool:
    """True when *value*'s JSON ENCODING carries no *patterns* match.

    The scan domain and the scrub domain must agree.  ``_gate`` scans
    ``json.dumps(observation)``, so cleanliness of the RAW value is the wrong
    question: ``'\\t' + 'A' * 63`` carries a 63-character run raw (below the
    threshold) but a 64-character one once ``json.dumps`` renders the tab as
    ``\\t`` and the escape's literal ``t`` extends it.

    An unencodable value answers True: it can never appear in the encoded
    document at all, so it cannot contribute a match there.  (It also cannot
    reach here through ``_gate``, which encodes the whole observation first.)
    """
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return True
    return scan_for_credential_material(encoded, patterns) is None


def _scrub_value(value: Any, patterns: tuple[tuple[str, str], ...]) -> Any:
    """Return *value* scrubbed so that its JSON ENCODING carries no *patterns* match.

    Two gaps between the scan domain and the scrub domain are closed here.

    KEYS.  ``_gate`` scans the JSON encoding of the WHOLE observation, in which a
    dict key is just as visible as a value, so keys are rewritten with the same
    substitution as leaves (non-string keys are left untouched).  Recursing into
    values alone let a generic-pattern hit in a key survive into ``_gate``'s
    follow-up ``assert_no_credential_material``.

    ENCODED FORM.  Every leaf, scalar and key is additionally checked in its
    ENCODED form (:func:`_encodes_clean`) and degraded wholesale to
    ``<redacted>`` if the raw substitution did not suffice — JSON escaping can
    manufacture a run that raw scrubbing cannot see.

    Either gap turned ``_gate``'s documented never-raise branch into a raise,
    losing the entire already-paid-for capture rather than one matched run.

    Per-part cleanliness suffices for the WHOLE document because a
    ``[A-Za-z0-9_-]{64,}`` run can never span JSON punctuation (``"``, ``:``,
    ``,``, ``{``, ``[``): none of those characters is in the run's class, so no
    match can straddle two parts, and each part's own encoding is delimited by
    exactly the same punctuation it will sit between in the document.

    Cost: the extra per-part ``json.dumps`` is off the sampling hot loop —
    ``_scrub_value`` runs only after ``_gate`` has already detected a generic hit.
    """
    if isinstance(value, str):
        scrubbed = _scrub_text(value, patterns)
        # The raw substitution is tried FIRST because it preserves the maximum
        # surrounding text; the wholesale fallback is only for what it missed.
        return scrubbed if _encodes_clean(scrubbed, patterns) else '<redacted>'
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            scrubbed_key = _scrub_text(key, patterns) if isinstance(key, str) else key
            if not _encodes_clean({scrubbed_key: 0}, patterns):
                # Encoded as ``{"<key>": 0}`` — the same ``"`` delimiters the key
                # will carry in the document, so this asks exactly the question
                # _gate's scan will ask.  Covers a non-string key too, which no
                # substitution branch would otherwise see (json renders it as a
                # string).
                scrubbed_key = '<redacted>'
            # Two distinct keys can scrub to the same '<redacted>' token (or
            # collide with a literal one already present), and letting the last
            # write win would silently DROP an entry — precisely the silent
            # degradation this gate exists to prevent.  Disambiguate instead,
            # counting up from 2 in insertion order so a re-run of the probe over
            # the same observation stays byte-reproducible.  '#' is outside the
            # `[A-Za-z0-9_-]` credential-run character class, so the suffix can
            # neither extend an adjacent run nor re-form a match of its own.
            if scrubbed_key in out:
                suffix = 2
                while f'{scrubbed_key}#{suffix}' in out:
                    suffix += 1
                scrubbed_key = f'{scrubbed_key}#{suffix}'
            out[scrubbed_key] = _scrub_value(item, patterns)
        return out
    if isinstance(value, list):
        return [_scrub_value(item, patterns) for item in value]
    # A non-str, non-container scalar: no substitution branch sees it, but its
    # ENCODING is plain text in the document (a 70-digit int renders as 70
    # run-class characters), so it needs the encoded check too.
    return value if _encodes_clean(value, patterns) else '<redacted>'


#: Scalar-only projection of the CLI's ``--output-format json`` envelope that
#: :func:`_drain_exit` records and :func:`_poisoned_exit_provenance` may carry.
#: ``subtype`` is the one arbitrary STRING among them, so it is listed separately:
#: the degraded row drops it (a clean-by-construction row cannot carry a string
#: the probe did not author), while the healthy path keeps it.
_EXIT_ENVELOPE_SCALAR_KEYS = ('is_error', 'num_turns', 'duration_ms', 'duration_api_ms')
_EXIT_ENVELOPE_KEYS = ('subtype', *_EXIT_ENVELOPE_SCALAR_KEYS)

#: ``substrate_returns`` keys a poisoned row must carry, with the scalar types
#: each is allowed to keep.  ``bool`` is a subclass of ``int``, so ``int`` covers
#: both flags and counts; anything else degrades to ``None``.
_SUBSTRATE_KEYS = (
    'transcript_exists',
    'read_transcript_records_is_none',
    'record_count',
    'count_transcript_turns',
)


def _int_or_none(value: Any) -> int | None:
    """*value* if it is a genuine ``int``, else ``None``.

    The ``not isinstance(value, bool)`` half is the whole reason this exists as a
    helper: ``bool`` is a subclass of ``int``, so ``True`` would otherwise pass
    through as the int ``1`` and make a nonsense ``sample_index`` (or a nonsense
    ``exit_code``) read as a real one.  Stated once here rather than re-argued at
    each of the call sites that need it.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number_or_none(value: Any) -> int | float | None:
    """*value* if it is a genuine ``int``/``float``, else ``None``.

    The :func:`_int_or_none` rationale, widened to the fields that are legitimately
    fractional (``sample_offset_secs``).  ``bool`` is still excluded.
    """
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _bool_or_none(value: Any) -> bool | None:
    """*value* if it is a genuine ``bool``, else ``None``.

    The converse filter: here an int ``1`` must NOT read as ``True``, because a
    degraded row's flags claim to be observations, not truthiness.
    """
    return value if isinstance(value, bool) else None


#: Anchored, length-bounded shapes for the two PROBE-AUTHORED identity strings a
#: degraded row is allowed to carry (see :func:`_poisoned_observation`).  Matching
#: one is what makes the value clean BY CONSTRUCTION rather than by a scan: both
#: alphabets exclude every named marker (``sk-ant-``, ``accessToken``, ...), and
#: the longest ``[A-Za-z0-9_-]`` stretch either shape admits is 36 characters (a
#: whole UUID), well under the 64-character generic run threshold.  ``fullmatch``
#: is load-bearing — an unanchored match would let arbitrary text ride along.
_ISO_TIMESTAMP_RE = re.compile(
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:[+-]\d{2}:\d{2}|Z)'
)
_UUID_RE = re.compile(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def _shaped_or_none(value: Any, shape: re.Pattern[str]) -> str | None:
    """*value* if it is a string FULLY matching *shape*, else ``None``."""
    return value if isinstance(value, str) and shape.fullmatch(value) else None


def _poisoned_observation(
    observation: dict[str, Any], pattern_name: str
) -> dict[str, Any]:
    """Return a minimal row standing in for an observation redaction could not clean.

    Observation-shaped ONLY — :func:`_gate` selects it via ``kind='observation'``.
    Exit provenance degrades through :func:`_poisoned_exit_provenance` instead.

    Every value here is either a probe-owned literal, a member of a closed set
    the probe itself defines, or a non-string scalar — so the row cannot itself
    carry credential material, whatever was in *observation*.  That is the whole
    point: it is clean BY CONSTRUCTION rather than by a scan, which is what lets
    :func:`_gate` promise never to raise on the heuristic branch.

    ``substrate_returns`` is carried (scalar-filtered, all four keys always
    present) because ``run_live_probe`` subscripts
    ``candidate['substrate_returns']['count_transcript_turns']`` on the
    pre-first-token path — a placeholder without it would merely trade the
    AssertionError for a KeyError at exactly the same blast radius.

    ATTRIBUTION.  ``captured_at`` and ``session_id`` are carried too, because
    ``main()`` APPENDS to ``--out``: one JSONL file routinely holds several runs
    and several modes, and a row saying only ``mode='healthy', sample_index=2``
    cannot be traced back to the run that produced it — which makes the stderr
    warning's "fix ``_scrub_value``, then re-run the probe" advice unactionable.
    Both are probe-authored (``datetime.now(UTC).isoformat()`` and a ``uuid4``)
    and both are validated against an anchored shape before being carried, so
    they keep the row's clean-by-construction property rather than trusting their
    provenance: a value the probe did not author simply degrades to ``None``.

    Nothing else survives.  Dropping ``transcript_records`` / ``config_dir_tree``
    / ``run_exit`` / ``spawn_argv`` is what costs this ONE sample its analytical
    value — the deliberate price of not losing the other N.  ``cli_version`` and
    ``probe_run_id`` are dropped with them: the first is ``claude --version``
    output and the second can come straight from ``--probe-run-id``, so neither
    is probe-authored and neither has a shape that could be validated.
    """
    substrate = observation.get('substrate_returns')
    if not isinstance(substrate, dict):
        substrate = {}
    mode = observation.get('mode')
    sample_kind = observation.get('sample_kind')
    return {
        'redaction_failed': True,
        'redaction_failure_pattern': pattern_name,
        'mode': mode if mode in MODES else None,
        'sample_kind': sample_kind if sample_kind in SAMPLE_KINDS else None,
        'sample_index': _int_or_none(observation.get('sample_index')),
        'sample_offset_secs': _number_or_none(observation.get('sample_offset_secs')),
        'captured_at': _shaped_or_none(observation.get('captured_at'), _ISO_TIMESTAMP_RE),
        'session_id': _shaped_or_none(observation.get('session_id'), _UUID_RE),
        'substrate_returns': {
            key: (substrate.get(key) if isinstance(substrate.get(key), int) else None)
            for key in _SUBSTRATE_KEYS
        },
    }


def _poisoned_exit_provenance(
    provenance: dict[str, Any], pattern_name: str
) -> dict[str, Any]:
    """Return a minimal row standing in for exit provenance redaction could not clean.

    The :func:`_drain_exit`-shaped sibling of :func:`_poisoned_observation`, and it
    exists because the two :func:`_gate` call sites gate DIFFERENT shapes.  A
    single observation-shaped fallback applied to exit provenance would delete
    ``exit_code`` / ``killed_by_probe`` / ``stdout_envelope`` / ``stderr_len`` and
    substitute observation identity fields that mean nothing here — and because
    the gated value is stamped onto ``run_exit`` for EVERY observation of the run
    (not one sample), that is precisely the whole-capture blast radius the
    heuristic branch exists to avoid, plus a ``KeyError`` for any consumer reading
    ``run_exit['exit_code']``.

    So the degraded shape matches the gated shape: every ``_drain_exit`` key is
    present, ``mode`` is MODES-checked, the flags/counts are scalar-filtered, and
    the two fields that carry CLI-authored text — ``stderr_tail`` and the
    envelope's ``subtype`` — are dropped.  That is what costs this row its
    analytical value; the exit code, the kill flag and the stderr LENGTH (a
    number, not text) are what survive.
    """
    mode = provenance.get('mode')
    envelope = provenance.get('stdout_envelope')
    if not isinstance(envelope, dict):
        envelope = {}
    return {
        'redaction_failed': True,
        'redaction_failure_pattern': pattern_name,
        'mode': mode if mode in MODES else None,
        'killed_by_probe': _bool_or_none(provenance.get('killed_by_probe')),
        'exit_code': _int_or_none(provenance.get('exit_code')),
        # Numbers and flags only — `subtype` is CLI-authored text and is dropped.
        'stdout_envelope': {
            key: envelope[key]
            for key in _EXIT_ENVELOPE_SCALAR_KEYS
            if isinstance(envelope.get(key), (bool, int, float))
        },
        'stderr_len': _int_or_none(provenance.get('stderr_len')),
        # The field the residual hit most plausibly lived in: arbitrary CLI stderr.
        'stderr_tail': None,
    }


def _observation_label(observation: dict[str, Any]) -> str:
    """Name the gated SAMPLE for a stderr warning, echoing no input text.

    A warning that quoted the value it just matched would print the very material
    the gate exists to keep off the terminal, so the label is built only from
    fields the probe can vouch for: an int index (:func:`_int_or_none` — an
    arbitrary string in that field would otherwise be echoed verbatim, and this
    branch fires precisely when the observation just matched a credential
    pattern) and a ``sample_kind`` checked against the closed set the probe owns.
    """
    index = _int_or_none(observation.get('sample_index'))
    kind = observation.get('sample_kind')
    return (
        f'sample {index if index is not None else "unknown index"} '
        f'({kind if kind in SAMPLE_KINDS else "unknown kind"})'
    )


def _exit_label(provenance: dict[str, Any]) -> str:
    """Name the gated EXIT PROVENANCE, under :func:`_observation_label`'s rule."""
    mode = provenance.get('mode')
    return f'run_exit ({mode if mode in MODES else "unknown mode"})'


class _GatedShape(NamedTuple):
    """How :func:`_gate` handles ONE gated shape: how to name it, how to degrade it."""

    label: Callable[[dict[str, Any]], str]
    degrade: Callable[[dict[str, Any], str], dict[str, Any]]


#: ONE dispatch table per gated shape, selected by :func:`_gate`'s ``kind``.
#: Labelling and degrading are kept together deliberately: they are two answers to
#: the same question (what shape is this?), and splitting them across two
#: independent branches is how a new shape gets one of them and not the other.
_GATED_SHAPES: dict[str, _GatedShape] = {
    'observation': _GatedShape(_observation_label, _poisoned_observation),
    'run_exit': _GatedShape(_exit_label, _poisoned_exit_provenance),
}


def _gate(
    observation: dict[str, Any],
    *,
    kind: Literal['observation', 'run_exit'] = 'observation',
) -> dict[str, Any]:
    """Refuse — or scrub — an assembled observation carrying credential material.

    The capture-time half of the two-sided guard: unredacted material never
    reaches disk, so a probe run cannot produce a raw capture that the committed
    ``TestCorpusSecretHygiene`` assertion would later have to catch.

    The two pattern classes are handled DIFFERENTLY, because their failure modes
    are different:

    - a NAMED hit (``sk-ant-``, ``accessToken``, ...) is unambiguously a secret,
      so it raises and the run is abandoned;
    - the GENERIC long-run backstop is a heuristic that can fire on a long
      non-path identifier, so it substitutes and WARNS instead.  ``_gate`` is
      called from the sampling loop, and raising there propagates out through
      ``finally`` and destroys the entire live capture — including the
      real-money ``healthy`` / ``mcp_wedge`` runs — for a harness whose whole
      purpose is to be re-run after a CLI bump.  Either way the matched run
      never reaches disk; only the blast radius differs.

    The generic branch's never-raise guarantee is enforced BY CONSTRUCTION, not
    by trusting :func:`_scrub_value` to be exhaustive.  Two ways it was not have
    already been found and fixed (a credential-shaped dict KEY; a run that only
    exists in the JSON-ENCODED form), and both times the guarantee rested on a
    composition argument that turned out to be false.  So a still-dirty scrub is
    now handled rather than asserted away: the value degrades to a minimal row
    flagged ``redaction_failed`` plus a stderr WARNING.  That is loud and bounded
    — one value reduced to its identity/scalar fields, against a raise's cost of
    every sample of an already-paid-for live run — and the unclean material still
    never reaches disk, because the dirty object is dropped entirely rather than
    written.

    *kind* selects WHICH degraded row, and it is not cosmetic.  ``_gate`` has two
    call sites gating two different shapes: an assembled observation
    (``kind='observation'``, :func:`_poisoned_observation`) and ``_drain_exit``
    provenance (``kind='run_exit'``, :func:`_poisoned_exit_provenance`).  The
    degraded row must match the shape it stands in for; an observation-shaped
    fallback substituted for exit provenance would silently drop ``exit_code`` and
    friends from EVERY observation of the run, converting the bounded degradation
    into the whole-capture loss this branch exists to prevent.  It is typed as a
    ``Literal`` so pyright catches a wrong spelling at the call site (this repo
    type-checks every staged Python change); the runtime ``ValueError`` below
    stays as the backstop for an untyped caller.
    """
    shape = _GATED_SHAPES.get(kind)
    if shape is None:
        # A wrong literal is a programming error, not a property of the captured
        # data: it is data-INDEPENDENT, so it fires on the first gated value of
        # the first run rather than lurking until a heuristic happens to hit.
        raise ValueError(
            f'_gate: unknown kind {kind!r}; expected one of {tuple(_GATED_SHAPES)}'
        )

    encoded = json.dumps(observation)

    named = scan_for_credential_material(encoded, _NAMED_CREDENTIAL_PATTERNS)
    if named is not None:
        name, offset = named
        raise AssertionError(
            f'credential material in startup_completion_probe:observation: '
            f'pattern {name!r} matched at offset {offset} (match text withheld). '
            f'Redact it — record credential-bearing paths by presence/size only.'
        )

    generic = scan_for_credential_material(encoded, _GENERIC_CREDENTIAL_PATTERNS)
    if generic is None:
        return observation

    name, offset = generic
    # Identify the gated value WITHOUT echoing input text to stderr — see
    # _observation_label.  Built from the SAME dispatch entry that will build the
    # degraded row, so a new gated shape cannot arrive with one and not the other.
    where = shape.label(observation)
    print(
        f'startup_completion_probe: WARNING — heuristic pattern {name!r} matched at '
        f'offset {offset} in {where}. Scrubbing the matching run rather than '
        f'discarding the capture. Check the emitted observation: if this was a long '
        f'identifier rather than a secret, widen the lookarounds in '
        f'startup_completion_fixtures.GENERIC_CREDENTIAL_PATTERNS.',
        file=sys.stderr,
    )
    scrubbed = _scrub_value(observation, _GENERIC_CREDENTIAL_PATTERNS)
    encoded_scrubbed = json.dumps(scrubbed)
    try:
        _scf.assert_no_credential_material(
            encoded_scrubbed, source='startup_completion_probe:observation(scrubbed)'
        )
    except AssertionError:
        # The scrub did not fully clean the sample.  Do NOT re-raise: this is the
        # heuristic branch, and its blast radius is the whole capture.  Re-scan to
        # name the RESIDUAL pattern (which need not be the one that opened this
        # branch) so the warning points at what actually survived, falling back to
        # the triggering name if the scan and the assertion ever disagree.
        residual = scan_for_credential_material(encoded_scrubbed, _CREDENTIAL_PATTERNS)
        residual_name = residual[0] if residual is not None else name
        print(
            f'startup_completion_probe: WARNING — redaction FAILED for pattern '
            f'{residual_name!r} in {where}; emitting a minimal redaction_failed '
            f'row instead. This value is lost for analysis but the run is not: '
            f'fix _scrub_value so it cleans this shape, then re-run the probe.',
            file=sys.stderr,
        )
        return shape.degrade(observation, residual_name)
    return scrubbed


def scrub_credential_material(text: str) -> str:
    """Return *text* with every credential-shaped run replaced by ``<redacted>``.

    The substituting counterpart to :func:`scan_for_credential_material`, used on
    free-text provenance (an stderr tail) where raising would throw away a real,
    already-paid-for capture.  Structured observation fields use the raising gate.
    """
    out = text
    for _name, pattern in _CREDENTIAL_PATTERNS:
        out = re.sub(pattern + r'\S*', '<redacted>', out)
    return out


def _redact_argv(argv: list[str]) -> list[str]:
    """Drop any argv element that looks credential-shaped."""
    out: list[str] = []
    for element in argv:
        out.append('<redacted>' if scan_for_credential_material(element) else element)
    return out


def redact_record(record: dict) -> dict:
    """Project a transcript record down to the safe, predicate-relevant fields.

    Keeps the record ``type`` (what ``count_transcript_turns`` and the chosen
    predicate read), the ``queue-operation`` ``operation`` discriminator, the
    ``attachment`` kind/hook name, and ``message.role`` — never any content.
    """
    out: dict[str, Any] = {}
    for key in _RECORD_TYPE_KEYS:
        if key in record:
            out[key] = record[key]
    attachment = record.get('attachment')
    if isinstance(attachment, dict):
        out['attachment'] = {
            k: attachment[k] for k in ('type', 'hookName') if k in attachment
        }
    message = record.get('message')
    if isinstance(message, dict) and 'role' in message:
        out['message'] = {'role': message['role']}
    return out


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def snapshot_config_dir(
    config_dir: Path,
    *,
    epoch: float | None = None,
    prune_prefixes: tuple[str, ...] = DEFAULT_PRUNE_PREFIXES,
) -> list[dict]:
    """Sample *config_dir*, pruning the noisy subtrees by default.

    A thin default-supplying wrapper over
    ``startup_completion_fixtures.snapshot_config_dir`` — the SINGLE sampler
    definition.  Probe output and materialized fixture trees are therefore
    described by exactly one function, so a curated row can never encode a tree
    shape that the sampler would not have produced.  The only difference is the
    default: a live config dir needs the marketplace/backups prune, a
    materialized fixture tree must round-trip verbatim.
    """
    return _scf.snapshot_config_dir(
        config_dir, epoch=epoch, prune_prefixes=prune_prefixes
    )


def sample_proc(pid: int | None) -> dict:
    """Sample ``/proc/<pid>`` — liveness, scheduler state char, argv, direct children.

    Total: every field degrades to ``None``/``[]`` rather than raising, because a
    probe that crashes on a racing exit loses the whole (paid-for) capture.
    """
    out: dict[str, Any] = {
        'pid': pid,
        'alive': False,
        'state': None,
        'comm': None,
        'cmdline': None,
        'children': [],
    }
    if pid is None:
        return out
    proc = Path(f'/proc/{pid}')
    if not proc.exists():
        return out
    out['alive'] = True
    try:
        stat = (proc / 'stat').read_text()
        # comm may contain spaces/parens — split on the LAST ') ' per proc(5).
        out['state'] = stat.rsplit(') ', 1)[1].split()[0]
    except (OSError, IndexError):
        pass
    with contextlib.suppress(OSError):
        out['comm'] = (proc / 'comm').read_text().strip()
    try:
        raw = (proc / 'cmdline').read_bytes().decode('utf-8', 'replace')
        argv = [part for part in raw.split('\0') if part]
        out['cmdline'] = _redact_argv(argv)
    except OSError:
        pass
    try:
        child_pids = (proc / 'task' / str(pid) / 'children').read_text().split()
        for child in child_pids:
            try:
                comm = Path(f'/proc/{child}/comm').read_text().strip()
            except OSError:
                comm = None
            out['children'].append({'pid': int(child), 'comm': comm})
    except (OSError, ValueError):
        pass
    return out


def sample_substrate(
    transcript_path: Path | None, records: list[dict] | None
) -> dict:
    """Project the already-committed ``shared.cli_invoke`` reader returns.

    This is the whole point of the probe: the predicate 3326 ports into production
    must be expressible over substrate that exists on main TODAY.  Recording these
    three returns per sample proves the discrimination without new production code.

    Takes what :func:`observe` has ALREADY read — the resolved path and the parsed
    records — rather than re-globbing and re-reading the transcript three more
    times.  Correctness, not just cost: `transcript_exists` IS
    ``_resolve_transcript_path(...) is not None`` and `count_transcript_turns` IS
    the count of ``type == "assistant"`` records, by their one-line definitions
    in `shared.cli_invoke`, so re-calling them adds no information — but it does
    add three more sample points, and a transcript that lands BETWEEN them yields
    an internally inconsistent observation (``transcript_relpath: null`` beside
    ``transcript_exists: true``).  One read, one instant, one consistent row.

    The equivalence is pinned in the other direction by the committed
    ``TestPredicateDiscrimination::test_committed_substrate_returns_match_the_row``,
    which calls the real committed functions against every materialized row.
    """
    return {
        'transcript_exists': transcript_path is not None,
        'read_transcript_records_is_none': records is None,
        'record_count': None if records is None else len(records),
        'count_transcript_turns': (
            None
            if records is None
            else sum(1 for record in records if record.get('type') == 'assistant')
        ),
    }


def observe(
    *,
    config_dir: Path,
    session_id: str,
    probe_run_id: str,
    mode: str,
    sample_index: int,
    sample_kind: str,
    sample_offset_secs: float,
    cli_version: str,
    capture_method: str,
    pid: int | None,
    epoch: float | None,
    extra: dict | None = None,
) -> dict:
    """Assemble ONE redacted observation object and gate it before returning."""
    transcript_path = _resolve_transcript_path(config_dir, session_id)
    records = read_transcript_records(config_dir, session_id)
    observation: dict[str, Any] = {
        'probe_run_id': probe_run_id,
        'mode': mode,
        'wedge_shape': MODE_WEDGE_SHAPE.get(mode),
        'sample_index': sample_index,
        'sample_kind': sample_kind,
        'sample_offset_secs': round(sample_offset_secs, 3),
        'session_id': session_id,
        'cli_version': cli_version,
        'capture_method': capture_method,
        'captured_at': datetime.now(UTC).isoformat(),
        'config_dir_tree': snapshot_config_dir(config_dir, epoch=epoch),
        'transcript_relpath': (
            str(transcript_path.relative_to(config_dir)) if transcript_path else None
        ),
        'transcript_records': (
            None if records is None else [redact_record(r) for r in records]
        ),
        'substrate_returns': sample_substrate(transcript_path, records),
        'proc': sample_proc(pid),
    }
    if extra:
        observation.update(extra)
    return _gate(observation)


# ---------------------------------------------------------------------------
# Wedge stubs
# ---------------------------------------------------------------------------

_BUILD_WEDGE_STUB = """#!/bin/sh
# Stub standing in for a wrapper that is building the CLI from source and has
# not yet exec'd it.  Deliberately never touches CLAUDE_CONFIG_DIR.
echo 'Building claude-code from source (this may take a while)...' >&2
echo '   Compiling cli v2.1.220 (/home/build/claude-code)' >&2
sleep %(hold)d
"""

_UV_WEDGE_STUB = """#!/bin/sh
# Stub standing in for `uv` resolving/downloading the environment before the CLI
# is ever launched.  Deliberately never touches CLAUDE_CONFIG_DIR.
echo 'Resolved 214 packages in 1.24s' >&2
echo 'Downloading numpy (18.2MiB)' >&2
sleep %(hold)d
"""

_MCP_HANG_SERVER = '''#!/usr/bin/env python3
"""Stub stdio MCP server: accepts the connection, reads whatever the client
sends, and NEVER writes a response — so the client hangs at `initialize`."""
import sys
import time

while True:
    line = sys.stdin.readline()
    if not line:
        break
time.sleep(3600)
'''


def _write_stub(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body)
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# Spawn shapes
# ---------------------------------------------------------------------------


def _cli_version() -> str:
    try:
        proc = subprocess.run(
            ['claude', '--version'], capture_output=True, text=True, timeout=30
        )
        return proc.stdout.strip() or proc.stderr.strip() or 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def _oauth_token() -> tuple[str, str] | None:
    """Return ``(env_var_name, token)`` for the first available OAuth account.

    ``ALL_TOKEN_LETTERS``, not ``_oauth_accounts``' fleet default: this probe
    spends no fleet capacity and only needs SOME account so a machine with none
    degrades to a legible skip, so the interactive/primary account ``A`` counts.

    ``_oauth_accounts`` is the single home for this scan (task 3700) — see its
    module docstring; the letter set is not restated here.
    """
    return first_available_token(os.environ, ALL_TOKEN_LETTERS)


def _build_argv(
    mode: str,
    *,
    session_id: str,
    prompt: str,
    model: str,
    permission_mode: str,
    stub_dir: Path,
    hold_secs: int,
) -> tuple[list[str], list[Path]]:
    """Assemble the spawn argv for *mode*, mirroring ``build_claude_argv``'s shape.

    Returns ``(argv, temp_paths)``; the caller owns unlinking ``temp_paths``.
    """
    temp_paths: list[Path] = []
    if mode == 'build_wedge':
        return ([str(_write_stub(stub_dir, 'claude-build-wrapper.sh',
                                 _BUILD_WEDGE_STUB % {'hold': hold_secs}))], temp_paths)
    if mode == 'uv_wedge':
        return ([str(_write_stub(stub_dir, 'claude-uv-wrapper.sh',
                                 _UV_WEDGE_STUB % {'hold': hold_secs}))], temp_paths)

    fd, sysprompt_path = tempfile.mkstemp(suffix='.txt', prefix='startup_probe_sysprompt_')
    temp_paths.append(Path(sysprompt_path))
    with open(fd, 'w') as fh:
        fh.write('You are a probe target. Answer in one word.')

    argv = [
        'claude',
        '--print',
        '--output-format',
        'json',
        '--model',
        model,
        '--system-prompt-file',
        sysprompt_path,
        '--session-id',
        session_id,
        '--permission-mode',
        permission_mode,
        '--max-turns',
        '1',
        '--disallowed-tools',
        '*',
    ]

    if mode == 'mcp_wedge':
        server = _write_stub(stub_dir, 'mcp_hang_server.py', _MCP_HANG_SERVER)
        mcp_config = {
            'mcpServers': {
                'hang': {'type': 'stdio', 'command': sys.executable, 'args': [str(server)]}
            }
        }
        fd, mcp_path = tempfile.mkstemp(suffix='.json', prefix='startup_probe_mcp_')
        temp_paths.append(Path(mcp_path))
        with open(fd, 'w') as fh:
            json.dump(mcp_config, fh)
        # --strict-mcp-config scopes the run to ONLY the hanging server, so the
        # ambient project .mcp.json cannot muddy which server the CLI waits on.
        argv.extend(['--mcp-config', mcp_path, '--strict-mcp-config'])

    argv.extend(['-p', prompt])
    return (argv, temp_paths)


def _spawn_env(config_dir: Path, oauth_token: str | None) -> dict[str, str]:
    """Build the subprocess env, mirroring ``cli_invoke._invoke_claude``."""
    env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
    if oauth_token:
        env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token
    env['CLAUDE_CONFIG_DIR'] = str(config_dir)
    return env


# ---------------------------------------------------------------------------
# Probe drivers
# ---------------------------------------------------------------------------


#: Probe config-dir naming, mirroring ``usage_gate``'s pair (task 3086).
#: ``_PROBE_TASK_ID_PREFIX`` is the task-id stem handed to ``TaskConfigDir``;
#: ``_PROBE_DIR_PREFIX`` is the resulting on-disk prefix the sweep keys off.
#: Both the construction site and the sweep build from the same constant, so
#: the swept prefix and the created names cannot drift apart.  The resulting
#: name is byte-identical to the one this probe has always written, so nothing
#: that recorded provenance from an earlier run shifts.
_PROBE_TASK_ID_PREFIX = 'startup-probe-'
_PROBE_DIR_PREFIX = CONFIG_DIR_PREFIX + _PROBE_TASK_ID_PREFIX

#: Appended to the task-id stem under ``--keep-config-dir``, which makes the name
#: end in ``-keep`` rather than ``-<pid>``.  That is the whole mechanism: the
#: sweep only removes a dir it can ATTRIBUTE to a dead process via a parseable
#: trailing ``-<digits>`` (``config_dir._PID_SUFFIX_RE``), and its third safety
#: guard leaves an unattributable dir alone forever.  Without this the flag's two
#: halves disagree — ``cleanup_at_exit`` honours ``--keep-config-dir`` while the
#: next probe run (>``min_age_secs`` later) rmtree's the very dir the operator
#: asked to keep, destroying an artifact that costs a real-money live run to
#: retake.  It also means a kept dir is never auto-reclaimed: that IS the
#: request, and the name says so out loud on an operator's `ls /tmp`.
_PROBE_KEEP_SUFFIX = '-keep'

#: Set once the stale-probe-dir sweep has run in this process.  The sweep
#: reclaims OTHER (dead) processes' leftovers, so it is a process-wide one-shot:
#: re-running it per probe re-scans /tmp for no benefit.
_probe_dir_sweep_done: bool = False


def _sweep_stale_probe_dirs_once() -> int:
    """Reclaim dead-PID probe config dirs left by earlier processes.

    Runs at most once per process.  Returns the number of dirs removed (0 when
    already swept this process, or on failure).

    The SIGKILL half of config-dir reclamation, and the only half that covers a
    hard kill: ``cleanup_at_exit`` handles a clean exit and ``run_live_probe``'s
    ``finally`` handles a raise, but no teardown hook survives ``SIGKILL`` — and
    what is stranded is a 0600 ``.credentials.json`` holding a live OAuth access
    token.  Only reclaiming other processes' dead-PID leftovers bounds the
    on-disk population, which is why ``usage_gate`` pairs the same two
    mechanisms for its own probe dirs.

    Never raises, for ANY exception class: tmp hygiene must not be able to fail
    a capture that costs real money to retake.  The probe has no logger and
    prints its warnings to stderr (see :func:`_gate`), so this does too.
    """
    global _probe_dir_sweep_done
    if _probe_dir_sweep_done:
        return 0
    # Set BEFORE the call, not after, so a raising sweep still cannot re-run on
    # every subsequent probe.
    _probe_dir_sweep_done = True
    try:
        reclaimed = sweep_stale_pid_dirs(_PROBE_DIR_PREFIX)
        if reclaimed:
            # Silent on the zero case so the steady state stays quiet; loud when
            # there is something to say, so an operator can see the /tmp
            # population draining rather than rebuilding.
            print(
                f'startup_completion_probe: reclaimed {reclaimed} stale probe config '
                f'dir(s) under {_PROBE_DIR_PREFIX} (dead-PID sweep).',
                file=sys.stderr,
            )
        return reclaimed
    except Exception as exc:  # noqa: BLE001  (deliberately broad — see docstring)
        # sweep_stale_pid_dirs already contains OSError internally, so anything
        # reaching here is UNFORESEEN.  Letting it escape would abort a probe run
        # over a stale /tmp dir, which is strictly worse than leaving one behind.
        print(
            f'startup_completion_probe: WARNING — stale probe-dir sweep of '
            f'{_PROBE_DIR_PREFIX} failed ({exc!r}); continuing without it (the next '
            f'process start retries).',
            file=sys.stderr,
        )
        return 0


def _drain_exit(
    proc: subprocess.Popen,
    *,
    mode: str,
    stdout_path: Path,
    stderr_path: Path,
) -> dict:
    """Kill (if still running), reap, and return exit provenance from the capture files.

    Reads the child's output from the temp files it was spawned against, NOT from
    pipes.  That is load-bearing rather than incidental: a ``PIPE`` the parent
    does not read fills at ~64 KB and blocks the child on write, and the parent
    here is busy sampling for up to ``--max-secs`` before it could drain.  A
    chatty startup — ``mcp_wedge``, the shape most likely to log connection
    failures, and the run finding F3 rests on — would then wedge *because the
    probe measured it*.  Files cannot backpressure, so the child never blocks and
    the capture is drained once, after the sampling window closes.

    The stderr tail is SCRUBBED (patterns substituted, not raised on) and
    truncated — it is provenance for the report, e.g. whether the CLI logged an
    MCP connection failure while the stub server hung.
    """
    still_running = proc.poll() is None
    if still_running:
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    stdout = stdout_path.read_bytes() if stdout_path.exists() else b''
    stderr = stderr_path.read_bytes() if stderr_path.exists() else b''
    text = stderr.decode('utf-8', 'replace')
    # Scalar-only projection of the CLI's --output-format json envelope: enough
    # to explain a non-zero exit (e.g. `error_max_turns` vs a startup failure)
    # without capturing the model's `result` text.
    envelope: dict[str, Any] = {}
    with contextlib.suppress(ValueError, AttributeError):
        parsed = json.loads(stdout.decode('utf-8', 'replace'))
        if isinstance(parsed, dict):
            envelope = {
                k: parsed[k]
                for k in _EXIT_ENVELOPE_KEYS
                if k in parsed and isinstance(parsed[k], (str, bool, int, float))
            }
    return {
        'mode': mode,
        'killed_by_probe': still_running,
        'exit_code': proc.returncode,
        'stdout_envelope': envelope,
        'stderr_len': len(text),
        'stderr_tail': scrub_credential_material(text[-_STDERR_TAIL_CHARS:]),
    }


@contextlib.contextmanager
def _teardown_step(label: str) -> Iterator[None]:
    """Run ONE teardown step, reporting rather than propagating its failure.

    ``run_live_probe``'s ``finally`` runs while an exception may already be in
    flight, so an unguarded raise from any step there does two harms at once: it
    REPLACES the exception being unwound (the caller is then told about tidy-up
    instead of the actual cause) and it SKIPS every remaining step — including
    ``config.cleanup()``, the only thing that removes a 0600
    ``.credentials.json`` holding a live OAuth access token, and ``proc.kill()``,
    the only reaper left once the ``try`` body has raised.  Isolating each step
    individually buys both back, and is why the ordering comment in that
    ``finally`` can now describe a mechanism rather than an intention.

    LOUD, not silent.  ``contextlib.suppress`` would leave a probe that failed to
    tidy up indistinguishable from one that succeeded — the silent fail-soft this
    repo rejects — and it is also too narrow (closing a detached buffer raises
    ``ValueError``, not ``OSError``).  The failure is printed to stderr, naming
    the step, matching :func:`_gate` and :func:`_sweep_stale_probe_dirs_once`;
    the module has no logger.

    ``Exception``, deliberately NOT ``BaseException``: a ``KeyboardInterrupt`` or
    ``SystemExit`` arriving during teardown must still propagate.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001  (deliberately broad — see docstring)
        print(
            f'startup_completion_probe: WARNING — teardown step {label!r} failed '
            f'({exc!r}); continuing with the remaining teardown.',
            file=sys.stderr,
        )


def run_live_probe(
    *,
    mode: str,
    probe_run_id: str,
    cwd: Path,
    prompt: str,
    model: str,
    permission_mode: str,
    offsets: tuple[float, ...],
    max_secs: float,
    hold_secs: int,
    keep_config_dir: bool,
) -> list[dict]:
    """Spawn *mode*'s target and emit one observation per sample."""
    _sweep_stale_probe_dirs_once()
    session_id = str(uuid.uuid4())
    cli_version = _cli_version()
    token_pair = _oauth_token()
    # --keep-config-dir has to opt OUT of both reclamation mechanisms, not one.
    # The `-keep` suffix takes the name out of the swept namespace (see
    # _PROBE_KEEP_SUFFIX); the cleanup_at_exit negation is the other half and is
    # equally load-bearing, not cosmetic: TaskConfigDir.__init__ registers
    # atexit.register(shutil.rmtree, ...) when it is True, so an unconditional
    # True would silently destroy at interpreter exit exactly the dir the flag
    # exists to preserve for post-run inspection.  The DEFAULT name is unchanged
    # byte for byte, so nothing that recorded provenance from an earlier run
    # shifts; only the deliberately-kept dir is renamed.
    #
    # cleanup_at_exit covers a clean exit, including a raise that escapes
    # run_live_probe entirely (SystemExit, KeyboardInterrupt).  Nothing survives
    # SIGKILL, which is what the dead-PID sweep above is for.
    task_id = f'{_PROBE_TASK_ID_PREFIX}{mode}-{os.getpid()}'
    if keep_config_dir:
        task_id += _PROBE_KEEP_SUFFIX
    config = TaskConfigDir(task_id, cleanup_at_exit=not keep_config_dir)
    config_dir = config.path

    # Every resource the `finally` touches is initialized to a sentinel FIRST,
    # because the `try` below now opens before any of them exists and the
    # teardown must be reachable from every point inside it.
    stub_dir: Path | None = None
    temp_paths: list[Path] = []
    stdout_fh: IO[bytes] | None = None
    stderr_fh: IO[bytes] | None = None
    observations: list[dict] = []
    proc: subprocess.Popen | None = None

    # The `try` deliberately starts at the FIRST credential-bearing operation.
    # write_credentials puts a live OAuth access token in
    # <tmp>/claude-config-startup-probe-<mode>-<pid>/.credentials.json (0600),
    # and config.cleanup() in the `finally` is the only thing that removes it —
    # so a raise anywhere after the write and before the try (a missing binary,
    # a full /tmp, an error building argv or env) used to strand a real token on
    # disk indefinitely.  TaskConfigDir CONSTRUCTION stays outside: no token
    # exists until write_credentials, and `config` must be bound before the
    # `finally` that cleans it can refer to it.
    try:
        if token_pair is not None:
            config.write_credentials(token_pair[1])

        stub_dir = Path(tempfile.mkdtemp(prefix='startup_probe_stubs_'))
        argv, temp_paths = _build_argv(
            mode,
            session_id=session_id,
            prompt=prompt,
            model=model,
            permission_mode=permission_mode,
            stub_dir=stub_dir,
            hold_secs=hold_secs,
        )
        env = _spawn_env(config_dir, token_pair[1] if token_pair else None)

        # Capture files, NOT pipes — see _drain_exit: an undrained PIPE blocks
        # the child once ~64 KB of startup chatter fills it, and this parent
        # does not drain until the sampling window closes.
        stdout_path = stub_dir / 'probe-stdout.bin'
        stderr_path = stub_dir / 'probe-stderr.bin'
        stdout_fh = stdout_path.open('wb')
        stderr_fh = stderr_path.open('wb')

        # Stamped last, immediately before the spawn, so every sample offset
        # still measures from the spawn rather than from setup.
        epoch = time.time()
        start = time.monotonic()

        # start_new_session=True mirrors cli_invoke._run_subprocess's spawn shape,
        # so the observed process-group / children topology is production's.
        proc = subprocess.Popen(  # noqa: S603
            argv,
            cwd=str(cwd),
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,
        )

        pending = list(offsets)
        sample_index = 0
        pre_first_token: dict | None = None
        seen_turn = False

        def _take(kind: str) -> dict:
            nonlocal sample_index
            observation = observe(
                config_dir=config_dir,
                session_id=session_id,
                probe_run_id=probe_run_id,
                mode=mode,
                sample_index=sample_index,
                sample_kind=kind,
                sample_offset_secs=time.monotonic() - start,
                cli_version=cli_version,
                capture_method='live_spawn',
                pid=proc.pid if proc else None,
                epoch=epoch,
                extra={'spawn_argv': _redact_argv(argv), 'oauth_env_var': (
                    token_pair[0] if token_pair else None
                )},
            )
            sample_index += 1
            return observation

        def _transcript_state_key() -> tuple[str, int, int] | None:
            """Cheap change-detector for the watched transcript: one glob + one stat.

            ``_take`` costs a full ``rglob('*')`` of the config dir (which walks
            the several-hundred-file ``plugins/marketplaces`` clone before
            pruning it), plus a transcript read and a JSON credential scan.  At
            the 0.2 s tick that was ~5 tree walks a second, every result but the
            last discarded — and the probe's own FS churn competing, on the same
            filesystem, with the CLI startup whose timing it is measuring.  So
            re-take the candidate only when this key moves.
            """
            path = _resolve_transcript_path(config_dir, session_id)
            if path is None:
                return None
            try:
                stat = path.stat()
            except OSError:
                return None
            return (str(path), stat.st_size, stat.st_mtime_ns)

        # _UNSET, not None: None is the real "no transcript yet" key, and the
        # first tick must always produce a candidate so a run that never reaches
        # session init still carries one.
        candidate_key: Any = _UNSET

        while True:
            elapsed = time.monotonic() - start
            if pending and elapsed >= pending[0]:
                pending.pop(0)
                observations.append(_take('scheduled'))
            if not seen_turn:
                turns = count_transcript_turns(config_dir, session_id)
                if turns is not None and turns >= 1:
                    seen_turn = True
                    if pre_first_token is not None:
                        pre_first_token['sample_kind'] = 'pre_first_token'
                        observations.append(pre_first_token)
                    observations.append(_take('first_token'))
                else:
                    # Keep only the most recent pre-turn-1 sample; it is the
                    # incident-shape observation the whole two-regime grace is for.
                    # Re-taken only when the transcript actually changed, so an
                    # unchanging one is not re-sampled ~5x/second for a result
                    # that is discarded.  Nothing is lost: an identical key means
                    # an identical observation, and the late state of a run that
                    # never starts is still captured by the `deadline` /
                    # `after_exit` samples below.
                    #
                    # The turn can land BETWEEN the check above and this sample, so
                    # re-read the candidate's OWN recorded turn count and discard it
                    # if it already shows turn 1 — otherwise the row would be
                    # labelled `pre_first_token` while carrying an assistant record,
                    # which is exactly the mislabel that would let a curated corpus
                    # row lie about the regime it came from.
                    key = _transcript_state_key()
                    if key != candidate_key:
                        candidate_key = key
                        candidate = _take('pre_first_token_candidate')
                        observed = candidate['substrate_returns']['count_transcript_turns']
                        if observed is None or observed < 1:
                            pre_first_token = candidate
            if proc.poll() is not None:
                observations.append(_take('after_exit'))
                break
            if elapsed >= max_secs:
                observations.append(_take('deadline'))
                break
            time.sleep(_FINE_TICK_SECS)

        # Flush the buffered pre-turn-1 sample BEFORE stamping, so it carries the
        # same run_exit provenance as every other observation of this run.
        if not seen_turn and pre_first_token is not None:
            pre_first_token['sample_kind'] = 'pre_first_token'
            observations.append(pre_first_token)

        # Reap the child, read its capture files, and stamp exit provenance onto
        # every observation of this run.
        # kind='run_exit': this value is NOT observation-shaped, and it is stamped
        # onto every observation below — a mismatched degraded row here would cost
        # the whole capture its exit provenance, not one sample's.
        exit_provenance = _gate(
            _drain_exit(proc, mode=mode, stdout_path=stdout_path, stderr_path=stderr_path),
            kind='run_exit',
        )
        for observation in observations:
            observation['run_exit'] = exit_provenance
    finally:
        # Every step below is INDEPENDENTLY failure-isolated by _teardown_step:
        # one that raises is reported to stderr and the rest still run.  That is
        # the MECHANISM, stated rather than assumed — this block runs with an
        # exception potentially in flight, so an unguarded raise here would both
        # replace that exception with a misleading tidy-up error and skip
        # everything after it.  (It did, before this block was restructured: a
        # subdirectory in the stub dir made the then-per-entry unlink raise
        # IsADirectoryError, which masked the cause, orphaned the child and
        # skipped config.cleanup() — while only the rmdir one statement later
        # was guarded.)
        #
        # Still None-safe throughout: the try opens before these exist, so any of
        # them can still be its sentinel.  Still ordered least-to-most important,
        # with the credential dir LAST — now genuinely unconditionally reached,
        # because no earlier step can abort the block.
        if stdout_fh is not None:
            with _teardown_step('close stdout capture'):
                stdout_fh.close()
        if stderr_fh is not None:
            # Isolated SEPARATELY from the stdout close: a failing flush on one
            # capture file (a full /tmp surfaces as ENOSPC at close, not at
            # write) must not leave the other handle to the GC.
            with _teardown_step('close stderr capture'):
                stderr_fh.close()
        if proc is not None:
            # The only reaper left when the try body raised before _drain_exit:
            # skipping this orphans a live CLI child holding the OAuth token in
            # its environment.
            with _teardown_step('kill child process'):
                if proc.poll() is None:
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=10)
        for path in temp_paths:
            # Per PATH, not per loop: one unlinkable temp file must not strand
            # the others.
            with _teardown_step(f'unlink temp file {path}'):
                path.unlink(missing_ok=True)
        if stub_dir is not None:
            # rmtree, not glob + unlink + rmdir.  Path.unlink cannot remove a
            # SUBDIRECTORY (IsADirectoryError), so the per-entry loop this
            # replaces could only ever REPORT a stub dir containing one and then
            # abandon it in the system tempdir — where nothing reclaims it,
            # because the dead-PID sweep is scoped to the config-dir prefix and
            # never sees `startup_probe_stubs_*`.  One step, one warning, and a
            # dir that is actually reclaimed whatever the child left inside it.
            with _teardown_step(f'remove stub dir {stub_dir}'):
                shutil.rmtree(stub_dir)
        if not keep_config_dir:
            # Isolated too.  This does not weaken the security guarantee: the
            # guarantee is that cleanup is always ATTEMPTED, no code inside this
            # finally could do better if the rmtree itself fails, and letting it
            # raise here would only mask the in-flight exception.  The two
            # remaining backstops — the cleanup_at_exit=not keep_config_dir
            # atexit hook and the dead-PID sweep — do not make this optional:
            # atexit is disabled under --keep-config-dir and does not run under
            # SIGKILL.
            with _teardown_step(f'remove config dir {config_dir}'):
                config.cleanup()
                # TaskConfigDir.cleanup() is rmtree(ignore_errors=True), so it
                # CANNOT raise and the step above can never report a failure of
                # its own — which would make a genuinely failed removal (EACCES
                # on a root-owned child, EBUSY, an immutable inode) completely
                # silent, in the one teardown step with security consequences.
                # So check, and be loud: what survives is a 0600
                # .credentials.json holding a live OAuth access token, and this
                # WARNING naming the path is the operator's only signal.
                if config_dir.exists():
                    print(
                        f'startup_completion_probe: WARNING — the OAuth-bearing config '
                        f'dir {config_dir} SURVIVED cleanup and still holds a 0600 '
                        f'.credentials.json with a live access token. Remove it by hand: '
                        f'nothing else will (atexit rmtree ignores errors too).',
                        file=sys.stderr,
                    )
    return observations


def run_replay_probe(
    *,
    probe_run_id: str,
    source_config_dir: Path,
    session_id: str | None,
    offsets: tuple[float, ...],
) -> list[dict]:
    """Run the sampler READ-ONLY against an existing on-disk config dir.

    The pre-2 fallback for a dispatch that cannot spawn a live CLI.  Records
    ``capture_method='replayed_from_live_config_dir'`` plus the source dir so the
    provenance of every derived corpus row stays honest about how it was taken.
    """
    if session_id is None:
        candidates = sorted(source_config_dir.glob('projects/*/*.jsonl'))
        if not candidates:
            raise SystemExit(
                f'replay: no projects/*/*.jsonl transcript under {source_config_dir}'
            )
        session_id = candidates[0].stem
    observation = observe(
        config_dir=source_config_dir,
        session_id=session_id,
        probe_run_id=probe_run_id,
        mode='replay',
        sample_index=0,
        sample_kind='replay',
        sample_offset_secs=offsets[-1] if offsets else 0.0,
        cli_version=_cli_version(),
        capture_method='replayed_from_live_config_dir',
        pid=None,
        epoch=None,
        extra={'source_config_dir': str(source_config_dir)},
    )
    return [observation]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--out', type=Path, default=None, help='JSONL output path (default stdout)')
    parser.add_argument('--probe-run-id', default=None, help='defaults to <mode>-<uuid4 prefix>')
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    parser.add_argument('--prompt', default='ok')
    parser.add_argument('--model', default='haiku')
    parser.add_argument('--permission-mode', default='bypassPermissions')
    parser.add_argument('--max-secs', type=float, default=45.0)
    parser.add_argument('--hold-secs', type=int, default=60, help='wedge stub sleep duration')
    parser.add_argument('--keep-config-dir', action='store_true')
    parser.add_argument(
        '--source-config-dir', type=Path, default=None, help='replay mode: dir to sample'
    )
    parser.add_argument('--session-id', default=None, help='replay mode: session to resolve')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probe_run_id = args.probe_run_id or f'{args.mode}-{uuid.uuid4().hex[:12]}'

    if args.mode == 'replay':
        if args.source_config_dir is None:
            raise SystemExit('--source-config-dir is required for --mode replay')
        observations = run_replay_probe(
            probe_run_id=probe_run_id,
            source_config_dir=args.source_config_dir,
            session_id=args.session_id,
            offsets=DEFAULT_SAMPLE_OFFSETS,
        )
    else:
        observations = run_live_probe(
            mode=args.mode,
            probe_run_id=probe_run_id,
            cwd=args.cwd,
            prompt=args.prompt,
            model=args.model,
            permission_mode=args.permission_mode,
            offsets=DEFAULT_SAMPLE_OFFSETS,
            max_secs=args.max_secs,
            hold_secs=args.hold_secs,
            keep_config_dir=args.keep_config_dir,
        )

    lines = '\n'.join(json.dumps(o, sort_keys=True) for o in observations)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('a', encoding='utf-8') as fh:
            fh.write(lines + '\n')
    else:
        sys.stdout.write(lines + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
