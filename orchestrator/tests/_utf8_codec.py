"""The utf-8 codec check shared by the task-3387 encoding pins of the fleet-root writers and readers."""

from __future__ import annotations

import codecs


def names_the_utf8_codec(name: str) -> bool:
    """Does ``name`` name the utf-8 codec, however this host spells it?

    ``codecs.lookup`` canonicalises every alias CPython accepts — ``'UTF-8'``,
    ``'utf8'``, ``'utf_8'``, ``'U8'`` — onto the single name ``'utf-8'``, so
    the task-3387 encoding pins assert the SEMANTIC contract (*this is the utf-8 codec*)
    rather than one spelling of it.

    That distinction is the whole reason this helper exists.  Task 3387's
    first attempt compared against the literal ``'utf-8'``, and it only
    discriminated by ACCIDENT: this host's ``locale.getpreferredencoding()``
    returns uppercase ``'UTF-8'``, so the assertion failed on a case mismatch.
    On a host spelling it ``'utf-8'`` that pin would have gone silently
    vacuous.  Normalising here removes the accident; the discriminating power
    is supplied deliberately by each pin instead.

    An unknown name is simply not the utf-8 codec: report ``False`` rather
    than letting ``LookupError`` escape, where it would read as an
    infrastructure error rather than the contract violation it is.
    """
    try:
        return codecs.lookup(name).name == 'utf-8'
    except LookupError:
        return False
