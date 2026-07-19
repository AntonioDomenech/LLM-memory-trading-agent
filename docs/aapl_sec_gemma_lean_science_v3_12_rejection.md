# AAPL SEC/Gemma lean science v3.12 preregistration rejection

## Decision

V3.12 is permanently rejected before an implementation commit, official
one-shot preflight, Yahoo request, Ollama request, Gemma generation,
market-value read, prediction, action, return, performance calculation, paid
API, broker action, or real-money action.

The immutable V3.12 preregistration inherits a loaded-code encoder that cannot
represent its own exact 44-module runtime. Accepting the compatibility adapters
from the uncommitted implementation draft would weaken the preregistration
after observing the real runtime and is therefore forbidden.

## Frozen contradiction

V3.12 incorporates every normative V3.11 byte except six explicitly listed
replacements. It says there are no other replacements, weakening by inference
is forbidden, and all inherited callable fingerprints, loaded-code identities,
and hard falsifiers remain exact.

The inherited encoder fingerprints closure values and referenced globals, but
permits `identity_sentinel` only for a literal `builtins.object` held by a
module global. Its only supported synchronization values are module-global
instances of exact `_thread.lock` or `_thread.RLock`. Unsupported semantic
values or conflicting qualified references reject.

One of the exact inherited shared modules,
`agent_benchmark/sec_filing_gemma_market_acquirer.py`, constructs four
module-global capability functions through a factory. Recursive closure replay
from those functions reaches five unique stateful values that are not module
globals:

| Closure value | Unique objects | Closure aliases |
|---|---:|---:|
| literal `builtins.object` sentinels (`issuer`, `missing`) | 2 | 3 and 3 |
| `weakref.WeakKeyDictionary` registries | 2 | 2 and 3 |
| exact `_thread.lock` (`registry_lock`) | 1 | 4 |

The registries and closure-owned lock have no inherited supported encoding.
The two sentinels are not eligible for the module-global-only
`identity_sentinel` rule.

There is a second independent collision. The exact frozen
`sec_filing_gemma_ollama.OllamaExtractionReceipt` is a frozen
`slots=True` dataclass. Its generated `__setattr__` and `__delattr__` close over
the original pre-slots class. That class is distinct from the exported
replacement class but has the same owner module and qualified name. The frozen
conflicting-qualified-reference rule therefore rejects literal replay.

## Reproduction

An isolated Python 3.12 authority probe authenticated the scientific process
and produced these execution-dependency counts before loaded-code derivation:

| Dependency family | Rows |
|---|---:|
| built-in or frozen modules | 63 |
| module files | 365 |
| distributions | 7 |
| loaded binaries | 58 |
| timezone files | 2 |

The literal inherited encoder cannot continue over the real closure values and
slots-dataclass class alias described above. A draft extension could encode
them and reached all 44 repository modules, but that extension is precisely the
unpreregistered rule change and is not V3.12 evidence.

Read-only draft diagnostics separately authenticated the preserved V3.8 store
over 1,414 files and 612,602,718 bytes and built the 75-row nullable projection:
73 rows have a real filename/official-primary URL and two have honest null/null
source identity. This reused local preserved evidence and made no SEC or other
network request. Focused uncommitted draft checks reached 118/118
contract/bridge/journal tests and 93/93 runner/store tests at their respective
checkpoints. They do not override the impossible preregistration and are not a
strategy result.

## Preserved state

The immutable V3.12 preregistration is commit
`f9122a4db5b4cbe27516cd03a8a86933fe9c2472`, tree
`2136f781d804a9579ed78551d25d5d86c07a74d6`, document blob
`7195936f87b66e34fb7e4119d5b9d4f6f949df74`, and document literal SHA-256
`2bf92ba7f29836094c3204dd4a33f897eec316fa50272afab0ee8b98f15168ba`.

Twelve uncommitted V3.12 implementation-draft paths were under construction
when the contradiction was confirmed. They are not an implementation commit,
qualification receipt, preflight receipt, or scientific result. The official
one-shot preflight was not consumed.

## Successor boundary

A successor may keep every V3.12 scientific rule, source row, nullable source
rule, body, prompt, model, feature, market series, execution order, threshold,
cost, gate, stage boundary, and effect budget unchanged. Its process-only
replacement must explicitly and deterministically define closure-slot ownership
for the two sentinels, two weak registries, and one lock; define their allowed
initial and checkpoint state; resolve the frozen-slots dataclass predecessor
class without merging unrelated same-qualified callables; and bind all such
objects to process-local identity continuity. No V3.12 strategy experiment may
be run or reinterpreted.
