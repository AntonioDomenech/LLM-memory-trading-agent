# AAPL SEC/Gemma lean science v3.19 exceptional zero-effect rejection

## Decision

V3.19 is permanently rejected as an execution-protocol implementation. It is
not rejected for strategy underperformance: no Yahoo market-value request,
Ollama identity request, Gemma generation, prediction, action, return,
development score, or buy-and-hold comparison ever opened. Confirmation and
live-style data remained closed. Every V3.19 performance field is therefore
not applicable, not zero.

This document and the matching comparison-table row are an explicit
evidence-preservation exception, not `R319` and not formal `X319`. The frozen
P319 topology did not define the state that actually occurred: a passing
`F319`, followed by a candidate-less zero-effect development rejection, while
the production publication-recovery route was itself unreachable. Creating a
hand-written `DEVELOPMENT_RESULT.json` would falsely claim terminal scientific
authority. Changing only the comparison table would leave the development
failure unpreserved because `F319` is a passing preflight artifact, not a
failure artifact.

## Immutable pushed authority

The V3.19 preregistration is commit
`299a1458f2021be4ea8e38dbb2048b80fac64a18`, tree
`163a88727f45082d69a0c7b80b134e585bb9dbce`. The exact twelve-file
implementation is commit `f048e1fdc007007c5563aad238e2646c0cbe069d`, tree
`cb6911f9898e39b836ac565c72a089fffb8d82d3`. The passing one-file preflight is
commit `dcdb4a78beb1bb954701ed012c829d5040918166`, tree
`e1fafd365341abdf72d15add68553ffab9bec26b`.

The committed public preflight artifact is 9,014 bytes with literal SHA-256
`3f7e228458c88f3dd9f9fb9d3c7a9e793fb0b950219583ca054ac23a090e31c4`
and internal public-artifact SHA-256
`a00e05f7750b10b6aaa9633bdbfa7e79217a5928647c7cd5e6619e75c64caa71`.
It records 333/333 V3.19 tests, 655/655 shared-dependency tests, and 1/1
Requests-identity test passed. It also records all 75 canonical requests, the
exact seven requests and eight sentences affected by the frozen sanitizer,
zero empty required partitions, and zero external effects.

## Exact development stop

The official development command was launched through the frozen scientific
bootstrap from clean pushed `F319`. Its durable journal contains exactly two
events:

1. `attempt_intent`, event SHA-256
   `14a122d1e31342ff989fd385f17703f766d153ebcdd1f28f435e20368a165509`;
2. `runtime_binding_failed`, event SHA-256
   `bf6c11d34114f754366e5b1dc36b82b7d7e1f8780aa06c8a940451ef75338063`.

The second event records checkpoint `attempt_open`, failure code
`runtime_binding_mismatch`, and `external_intent_count=0`. There is no
checkpoint, payload, terminal candidate, terminal receipt, or recovery-journal
event. No SEC request, Yahoo request, Ollama request, model generation, paid
call, broker action, or real-money action occurred during the attempt.

## Exact runtime-binding cause

A separate read-only, no-effect fingerprint replay exited successfully after
1,204.7 seconds. The baseline and the state after loading the private contact
matched the frozen repository, dependency, and loaded-code manifests exactly.
The first and only divergence appeared immediately after authenticating the
preserved V3.8 source.

The source journal uses `datetime.strptime`. Python therefore loaded the
standard-library `_strptime` source module lazily after the V3.19 runtime
baseline had already been sealed. The dependency module-file count changed
from 365 to 366. The sole extra row was:

```text
module_name=_strptime
origin_kind=stdlib_source
resolved_path_sha256=d17608dec44f5f81db8923c0b33768a54a86529350d366ed4026f5465b507086
member_name=null
distribution_name=null
byte_count=25180
literal_sha256=8961aa3bf0fe6d677d26f83ea25c72685a6c7dde51d3bbd2ec4a2a405676a320
```

The dependency-manifest SHA-256 changed from
`e80e27212b6ef1b80408893776f35911912e1a7a516f680f24466b384ef53f13`
to
`40086a3ccff53fc8ff761802ac9aa924a794d309aa932145cd925ec2786c8e42`.
The loaded-code-manifest SHA-256 changed from
`09ab578da7942d49ef16e31c037ca5575221e16044f464753f30d60c8aa39472`
to
`5495dc6ba2481d7c1878639c1287ddf7eeef2deed4b4fd47e373deb5e698e21e`
only because it binds the dependency-manifest hash. All 44 repository module
rows and all 3,242 protected callable rows remained identical. Projection,
request-commitment, and model-plan construction introduced no further change.

A focused read-only proof imported `_strptime` before the baseline and then
ran representative `datetime.strptime` calls. Both manifests remained exactly
stable. The minimal scientific correction is therefore a deterministic
`_strptime` preload before both preflight dependency capture and development
session-baseline capture. No checkpoint reorder or strategy change is needed.

## Why normal recovery could not preserve R319

The official `recover-publication` bootstrap failed before mutation with the
redacted code `runtime_launch_proof_missing`. The attempt journal remained at
two events and the recovery journal remained empty. Read-only production-path
inspection found two independent contradictions:

1. recovery enters the shared authority callback, which first tries the
   development-only loader; that loader consumes the one-use
   `publication_recovery` launch proof before rejecting its invocation kind,
   so the intended recovery fallback sees a missing proof; and
2. even with direct routing, the recovery authority loader accepts only an
   already-paused store when no terminal receipt exists, although the recovery
   worker explicitly supports converting this exact zero-intent
   `rejected/runtime_binding_mismatch` state into the terminal
   `zero_effect_binding_rejection` outcome.

The first defect makes the valid proof unreachable. The second makes the exact
candidate-less state circularly unreachable. V3.19 code and authority are
immutable, so neither defect can be repaired inside V3.19 and the attempt may
not be retried.

## Successor boundary

A separately preregistered successor may change only the execution process
needed to preload `_strptime`, route publication recovery directly without
making its proof reusable, and authorize recovery of the exact clean
zero-intent binding-rejection state. It must add full production-composition
tests for these paths. The authenticated source, sanitizer, canonical 75
requests, local Gemma model, prompt, Yahoo series, features, learner, policy,
thresholds, transaction costs, chronological stages, scientific gates, and
effect budgets remain unchanged.

V3.19 has no strategy result and supplies no evidence that the system beats or
fails to beat buy-and-hold.
