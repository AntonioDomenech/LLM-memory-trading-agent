# AAPL SEC/Gemma shared baseline repair v1

## Status

The shared baseline test repair is preserved at commit
`29e7ecb4f3e42f3d50329e9d2337cd3eb91a49ee` on branch
`codex/aapl-sec-gemma-baseline-suite-repair-v1`.

This branch changes four test files and no production source. It repairs the
four unrelated baseline checks that prevented the frozen v3.9 draft from
passing its repository-wide offline gate. It does not revive or promote v3.9,
and it is not trading evidence.

No SEC, Yahoo, Ollama, model-generation, market-value, prediction,
performance, broker, paid-API, confirmation, final, or trading effect was
made. The private SEC contact was not printed or published.

## Repairs

1. The historical parent-verifier success test now adapts only the known
   `.gitattributes` drift while pinning the historical and current blobs. The
   production verifier and its other dependency checks remain fail closed.
2. The SEC cutoff fixture now uses the allowed Eastern label
   `20241231193000` instead of a deliberately forbidden `+00:00` spelling.
3. The Requests-object identity assertion now runs in a fresh local Python
   process. This tests the production identity rule without inheriting module
   objects imported by unrelated tests earlier in the collector process.
4. The frozen online-risk-overlay v2.2 source pins are verified against their
   exact preregistration revision, `a849b9d704ffd98547e570735a221b2b75f7db86`,
   rather than against later source revisions that the old contract never
   claimed to bind.

## Direct verification evidence

The four repaired modules passed together:

```text
92 passed in 44.88s
```

The exact four formerly failing cases passed together:

```text
4 passed in 13.76s
```

The clean pushed repair commit then collected exactly 5,825 tests under:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

The process started at 2026-07-18 15:04:53 Europe/Madrid and ended at
approximately 17:09:51, about 2 hours 5 minutes later. Files produced by the
last collected case,
`tests/test_warehouse.py::test_validate_empty_warehouse_reports_symbols`,
were written at 17:09:51. Pytest's cache was finalized at the same time.

The tool channel discarded the final human-readable pytest summary because
its output exceeded the remaining context window. Therefore this record does
not invent or quote an unobserved `passed/skipped` sentence.

## Cache audit and bounded conclusion

After completion, `.pytest_cache/v/cache/lastfailed` contained 34 names. An
exact comparison against the 5,825 currently collected node IDs found:

```text
exact current node IDs: 0
stale or renamed node IDs: 34
```

The entries came from other branches or earlier revisions. For example, the
cache included a v3.9 test even though this baseline branch contains no v3.9
test files; other entries named superseded v2.1 overlay cases, renamed
parameters, or removed test semantics. Three independent read-only audits
confirmed that all 34 exact cached names are absent from the current
collection. Twenty-seven mapped current equivalents were rerun and all 27
passed.

The defensible conclusion is that the complete run reached its last test and
left no recorded failure belonging to the code under test. That is strong
evidence that the repaired shared baseline is clean. It is not a substitute
for the dropped terminal sentence, so the exact final pytest count and exit
line remain explicitly unavailable.

The complete legacy suite is maintenance evidence, not the scientific
authorization gate for the next approach. The successor must instead require
its latest tests, the shared production dependencies it actually uses, and
the exact pinned parent-authority checks. Historical branch tests should be
replayed at their own pinned revisions, not forced to reinterpret newer HEAD
sources.
