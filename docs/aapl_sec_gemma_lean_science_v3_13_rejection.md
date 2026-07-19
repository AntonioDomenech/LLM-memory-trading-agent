# AAPL SEC/Gemma lean science v3.13 implementation rejection

## Decision

V3.13 is permanently rejected at the pushed-implementation inspection gate,
before the private one-shot preflight reservation, official qualification,
source replay, SEC request, Yahoo request, Ollama request, Gemma generation,
market-value read, prediction, action, return, performance calculation, paid
API, broker action, or real-money action.

The pushed implementation contains two variable-driven `importlib.import_module`
calls while its own strict local-import audit accepts only a literal string
written directly inside each recognized dynamic-import call. The contradiction
is deterministic in the immutable pushed source. Rewriting or force-replacing
that source would hide the failure and is forbidden.

## Preserved authority

The immutable V3.13 preregistration is commit
`679cebcfc8c86124be12c1a7ebbce2171735712d`, tree
`f6880ecc0c6fa878b8c3798ae782183afa896113`, document blob
`8c2a32ae424d820ec73d2d053b588c1e3dda4a96`, and document literal SHA-256
`600806541de00846e13f2a1b6a7b4b0bdff09e8d8d3e0432635bfc9b3dab00b2`.

Its exact twelve-addition implementation child is commit
`2f996bc963807f4a901676589b0b66d93855c184`, tree
`7fe69f3cd715e27a6c3eb09fed402419591ba118`, with the preregistration as its
sole parent. Local HEAD, upstream, the cached remote ref, and the live GitHub
branch all authenticated that same implementation before inspection.

## Frozen contradiction

`_audit_v313_local_import_upper_bound` parses the six exact V3.13 production
blobs from Git. For calls whose visible function name is `__import__`,
`import_module`, or `run_module`, it requires a positional first argument that
is an exact Python string literal. Its regression test deliberately proves
that assigning a local module name to a variable and then calling
`importlib.import_module(module_name)` must reject with
`preflight_local_import_dynamic_unresolved`.

The production preloader violates the same rule twice:

1. `agent_benchmark/sec_gemma_lean_science_v313_preflight.py:2912` calls
   `importlib.import_module(verifier_name)`; its first AST argument is a
   variable name.
2. line 2917 calls `importlib.import_module(name)` inside the exact-module
   dictionary comprehension; its first AST argument is also a variable name.

The later `importlib.import_module("numpy")` call is a direct string literal and
passes. No matching unresolved dynamic-import call exists in the other five
production files. Literalizing only the first call would expose the second, so
both are independently terminal for V3.13.

The test suite missed this contradiction because its upper-bound integration
test audits a small synthetic repository rather than the real pushed V3.13
implementation. Local Phase 1 nevertheless passed 322 of 322 tests with 322
case-sensitive unique node IDs, zero duplicates, and all 21 mandatory node IDs.
Those local checks are preserved implementation diagnostics, not an official
qualification receipt and not a strategy result.

## Pre-trust attempts and zero effects

The first launcher invocation stopped in 1.3 seconds with
`creation_pretrust_attestation_invalid` because the wrapper supplied Git for
Windows' `cmd` shim while the frozen validator authenticated the distinct
`bin/git.exe` file. It created no private or public path and was corrected only
through the explicitly retryable pre-trust launcher route.

The corrected launcher authenticated the pushed topology and stopped in 6.6
seconds with `preflight_local_import_dynamic_unresolved`. This happened inside
read-only `inspect_pushed_implementation`, before `_reserve_once`. The private
V3.13 namespace and public preflight artifact remained absent, the worktree
remained clean, and the official one-shot preflight was never consumed.

No SEC, Yahoo, Ollama, Gemma, market-value, prediction, performance, paid-API,
confirmation, final, broker, or real-money effect occurred. No V3.13 strategy
experiment ran, and no historical strategy version was rerun.

## Successor boundary

A successor may keep every V3.13 scientific rule, source row, nullable source
rule, body, prompt, model, feature, market series, execution order, threshold,
cost, gate, stage boundary, effect budget, runtime-v2 owner graph, closure-state
adapter, weak-registry adapter, relative-Path adapter, typing adapter, and
pseudo-module rule unchanged.

Its only process correction is to preserve verifier-first preload ordering while
using direct literal `importlib.import_module("...")` callsites for every exact
bounded local module. The strict dynamic-import scanner must remain unchanged;
aliasing, variable tracing, bypassing, or weakening it is forbidden. The
successor must add a regression that audits the actual successor production
sources with the same rule before its pushed implementation can authorize
preflight. V3.13 may never be retried, resumed, or reinterpreted as a strategy
result.
