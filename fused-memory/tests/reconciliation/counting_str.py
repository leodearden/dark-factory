"""A ``str`` subclass that counts characters touched and extraction-pipeline
entries, shared by the extraction and sweep suites."""


class CountingStr(str):
    """``str`` subclass that tallies characters TOUCHED by ``rfind``/indexing.

    ``_last_clause_break``'s documented cost property is "no slicing,
    O(len(prefix)) total scan work" (see its docstring) — a claim about how
    many characters the walk examines, not about wall-clock time. A prior
    version of this test asserted the property via ``time.perf_counter()``
    instead, and a re-measurement showed that assertion could not actually
    detect a quadratic regression at the input size it shipped with (the
    regression's extra cost is C-level slice memcpy, cheap enough to stay
    under budget until the input is enormous — see
    ``tests/reconciliation/test_status_snapshot_extraction.py::TestPluralEnumerationPerformance``
    for the numbers). Counting ``rfind``'s backward-scan distance (``stop -
    result``, or ``stop - start`` when not found) and every single-character
    ``__getitem__`` read observes the cost property directly, so a future
    edit that reintroduces slicing or a non-resuming ``rfind`` is caught by
    the touched-character count regardless of how fast the underlying C call
    happens to run on the machine executing the test. (task 4149)
    """

    def __new__(cls, value):
        self = super().__new__(cls, value)
        self.touched = 0
        self.reads = 0
        return self

    def __len__(self):
        """Tally EXTRACTION-PIPELINE ENTRIES (task 3037).

        The extraction pipeline normalises its input exactly once, in
        ``extract_snapshot_edge_task_ids_by_marker_class`` (``fact = fact or
        ''``; the two public accessors delegate to it rather than
        re-normalising). That truthiness test is the one and only
        Python-level ``__len__`` call the pipeline makes on the fact — the
        ``re`` module reads a str subclass's buffer at C level and never
        calls this. So ``reads`` counts exactly
        one per entry into the extraction pipeline, which is the property
        ``test_extraction_runs_exactly_once_per_edge`` needs and the
        ``touched`` character counter structurally cannot express (running
        two marker families costs the same character traversals whether they
        are reached through one entry point or two).
        """
        self.reads += 1
        return str.__len__(self)

    def rfind(self, sub, start=0, end=None):
        stop = len(self) if end is None else end
        result = str.rfind(self, sub, start, stop)
        self.touched += stop - (result if result >= 0 else start)
        return result

    def __getitem__(self, key):
        if isinstance(key, slice):
            out = str.__getitem__(self, key)
            self.touched += len(out)
            return out
        self.touched += 1
        return str.__getitem__(self, key)
