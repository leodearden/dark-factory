"""The ONE canonicalisation transform for escalation text keys.

Two call-site policies, one implementation
==========================================

Every key-matching site in this package that needs "the same string modulo
case, whitespace and punctuation" calls :func:`canonical_text` here.  There is
deliberately no second implementation anywhere — a lifted helper plus a
surviving inline copy of the same regex pipeline is exactly the lockstep
duplication that makes a transform drift silently.

What differs between call sites is not the transform but ONE named policy: how
punctuation is treated.  It is an explicit keyword argument rather than two
functions, so the pipeline itself (casefold -> map non-word chars -> collapse
whitespace -> strip) is still edited in exactly one place:

``punctuation='separator'`` (the default, used for ROOT CAUSES)
    ``a.b`` -> ``a b``.  Root-cause keys are delimiter-dense IDENTITY strings:
    69% of the distinct non-empty root_cause keys in the live queue (measured
    2026-08-19 over ``data/escalations`` root + archive, 398 keys) carry a digit
    adjacent to a separator, and those digits ARE the identity — task ids
    (``cleanup_needed:3087:...``), SHA prefixes (``bad-merge-to-main:65b011ed8c-...``),
    dates (``curator-failure:account-cap-2026-08-05``).  Deleting the separator
    glues them: ``risk:3184`` and ``risk:318:4`` become one key, and two
    distinct incidents are silently absorbed into one L2.  Separator semantics
    instead over-merges only ``a.b`` with ``a b``, whose exposed surface on the
    same corpus is 7 keys (1.8%), all pytest-output tails rather than identity.
    Under the asymmetric-cost rule — an under-fold leaves a noisy but SAFE
    duplicate L2, an over-fold silently merges distinct incidents — 69% versus
    1.8% picks the separator.

``punctuation='strip'`` (the LEGACY policy, pinned at dedupe's call sites)
    ``a.b`` -> ``ab``.  This reproduces the pre-lift behaviour of
    ``dedupe._normalize_description`` and ``dedupe.summary_dedupe_key``
    byte-for-byte, and it must keep doing so: ``compute_content_fingerprint``'s
    sha256 digests and ``summary_dedupe_key``'s tuples are already persisted
    across the live corpus, so switching those sites to separator semantics
    would silently un-dedupe every recon finding and re-partition every dedupe
    cluster already keyed on disk.  Do not "unify" the policies.

Underscore (U+005F, Unicode category Pc) is a word character and is therefore
PRESERVED by BOTH policies — ``fused_memory`` stays ``fused_memory``.  This is
the same deliberate divergence ``dedupe.summary_dedupe_key`` already documents.

This module is a LEAF
=====================

It has ZERO intra-package imports, and that is load-bearing rather than tidy:
``dedupe.py`` imports ``escalation.queue`` at module level, so ``queue.py``
cannot import ``dedupe.py`` — the transform could not be shared between them at
all if it lived in either.  Any intra-package import added here risks completing
that cycle, and the failure would surface as an ImportError far from the edit
that caused it.  ``tests/test_canonical.py`` pins the property BEHAVIOURALLY
with a fresh-interpreter ``sys.modules``-delta probe, because a source grep for
``import escalation`` is defeated by relative imports, ``importlib.import_module``
and line-broken statements — and falsely tripped by prose like this paragraph.
"""

from __future__ import annotations

import re
from typing import Literal

__all__ = ['canonical_root_cause', 'canonical_text']

# Lifted verbatim from dedupe.py, which is now a caller rather than an owner.
_NON_WORD_PATTERN = re.compile(r'[^\w\s]', flags=re.UNICODE)  # punctuation, symbols, control; keeps word chars and whitespace
_WHITESPACE_PATTERN = re.compile(r'\s+')  # collapse runs of whitespace

# The replacement each policy substitutes for a non-word character.  Kept as a
# mapping so an unknown policy is a KeyError-shaped lookup miss rather than a
# silent fall-through to a default — see the ValueError below.
_PUNCTUATION_REPLACEMENT = {'separator': ' ', 'strip': ''}


def canonical_text(text: str, *, punctuation: Literal['separator', 'strip'] = 'separator') -> str:
    """Return the canonical form of *text* under the given *punctuation* policy.

    Pipeline (the single implementation; see the module docstring for why the
    policy is a parameter rather than a second function):

    1. Casefold (Unicode-aware lower-case).
    2. Map every non-word, non-whitespace character to ``' '`` (``separator``)
       or to ``''`` (``strip``).
    3. Collapse whitespace runs to a single space.
    4. Strip the edges.

    Idempotent under both policies: the output contains only word characters and
    single interior spaces, which step 2 leaves untouched.  That matters because
    both sides of a root-cause match are canonicalised at compare time and a
    stored key may itself already be in canonical form.

    An unknown *punctuation* value raises ``ValueError`` rather than falling back
    to a default: a typo'd policy at a call site that believed it had pinned one
    would silently apply the WRONG matching rule, which is the silent
    degradation this module exists to prevent.
    """
    try:
        replacement = _PUNCTUATION_REPLACEMENT[punctuation]
    except KeyError:
        raise ValueError(
            f'unknown punctuation policy {punctuation!r}; '
            f'expected one of {sorted(_PUNCTUATION_REPLACEMENT)}'
        ) from None

    mapped = _NON_WORD_PATTERN.sub(replacement, text.casefold())
    return _WHITESPACE_PATTERN.sub(' ', mapped).strip()


def canonical_root_cause(text: str) -> str:
    """Return the canonical dedup key for a root_cause string.

    A thin alias that PINS ``punctuation='separator'`` once, in the open, so the
    policy decision lives at a name rather than being re-argued at every call
    site.  This is the form ``EscalationQueue.find_pending_l2_by_root_cause``
    compares on BOTH sides; the record's own ``root_cause`` is stored and
    displayed VERBATIM and is never overwritten with this value.

    Returns ``''`` for input carrying no word characters at all (``'::'``,
    ``'--'``).  Callers treat an empty canonical form as "not a usable dedup
    key" — such a key can never match anything, so minting an L2 under it would
    create a permanently unfoldable duplicate source.  Only purely
    punctuation/symbol keys are affected: ``\\w`` is Unicode-aware, so CJK and
    other non-Latin keys survive canonicalisation non-empty.
    """
    return canonical_text(text, punctuation='separator')
