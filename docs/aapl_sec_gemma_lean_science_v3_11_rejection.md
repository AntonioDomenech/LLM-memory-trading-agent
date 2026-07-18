# AAPL SEC/Gemma lean science v3.11 preregistration rejection

## Decision

V3.11 is permanently rejected before an implementation commit, one-shot
preflight, private source replay, Yahoo request, Ollama request, Gemma
generation, market-value read, prediction, action, return, performance
calculation, paid API, broker action, or real-money action.

The immutable V3.11 preregistration contains an internally inconsistent shared
qualification rule. Its exact 14-file collection has 655 node IDs, 655 unique
node IDs, and ordered node-list SHA-256
`1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c`.
The preregistration instead says that the same 655-entry list has 654 unique
IDs and contains this ID twice:

`tests/test_sec_gemma_lean_v38_transport.py::test_noncanonical_urls_are_rejected[HTTPS://data.sec.gov/submissions/CIK0000320193.json]`

The ID occurs once. The adjacent distinct ID is:

`tests/test_sec_gemma_lean_v38_transport.py::test_noncanonical_urls_are_rejected[https://DATA.sec.gov/submissions/CIK0000320193.json]`

The two strings differ by case. Python's case-sensitive `Counter` reports 655
unique IDs and no duplicate. PowerShell's default case-insensitive grouping
collapses those two strings and reproduces the mistaken 654-unique result.

## Reproduction

The offline reproduction used the 14 selectors frozen in the V3.11 document,
each exactly once, under the frozen environment with bytecode and pytest cache
disabled. The collection command profile was:

```text
python -B -m pytest -q -p no:cacheprovider --collect-only <14 frozen selectors>
```

The exact result was:

| Evidence | Value |
|---|---|
| process exit | `0` |
| collected node IDs | `655` |
| case-sensitive unique node IDs | `655` |
| duplicate IDs | none |
| ordered node-list SHA-256 | `1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c` |
| uppercase-scheme ID occurrences | `1` |
| uppercase-host ID occurrences | `1` |

The pinned hash therefore authenticates the real 655-distinct-node list, not
the contradictory multiplicity statement. Faking a duplicate, comparing IDs
case-insensitively, dropping a node, or changing the frozen hash would weaken
the preregistration after observing the result and is forbidden.

## Preserved state

The immutable preregistration is commit
`1d0975e8e6439335d1e3c24ac8632083af7e4a8a`, tree
`9bbf988bb47db253b9b84f72c259038a8c71cd95`, document blob
`9c9930da6350d445155499032c513788e9e9c90f`, and document literal SHA-256
`893a3f3a704ab7c118f4b6ac60a9c0eb51083fbd4ac31d9862b00b7b1f4fd3b1`.
It says the document is immutable and that a correction requires a successor.

Twelve uncommitted V3.11 implementation-draft paths were under construction
when the contradiction was confirmed. They are not a V3.11 implementation
commit, qualification receipt, preflight, or scientific result. Focused draft
checks that had completed were 40/40 contract/bridge tests and 11/11 journal
tests. They do not override the impossible preregistration.

## Successor boundary

A successor may keep every V3.11 scientific rule, source row, body, nullable
filename/URL rule, prompt, model, feature, market series, order, threshold,
cost, gate, stage boundary, and effect budget unchanged. Its process-only
correction must freeze the actual shared collection as 655 entries, 655 unique
IDs, and zero duplicates under the same ordered node-list hash. It must also
execute the already-preregistered isolated Python bootstrap rather than calling
`python -s -S -B -m pytest` without making pinned site-packages importable.
