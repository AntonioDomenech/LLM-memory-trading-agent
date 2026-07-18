# AAPL SEC/Gemma lean science v3.9 offline-readiness rejection

## Status

The v3.9 approach stopped at the offline implementation-readiness gate on
2026-07-18 Europe/Madrid. This is not a trading result and it is not a v3.9
preflight result. The one-shot v3.9 preflight was never invoked.

The preregistration is commit
`d50c33515ed9597b2fc07bb40c922a3f3166fbdd`. It requires all applicable
repository tests to pass before implementation publication and binds the
implementation to exactly twelve newly added v3.9 paths. The frozen complete-
suite command, run against the exact twelve-file draft, did not pass, and every
failing repair lies outside that twelve-path boundary. Changing an older file
inside v3.9, excluding a failure from the exact test command, or installing a
collection-time global patch would therefore violate the frozen rules.

No SEC, Yahoo, Ollama, model-generation, market-value, prediction,
performance, broker, paid-API, confirmation, final, or trading effect was
made. The readable private SEC contact was not printed or published.

## Complete-suite evidence

The offline rehearsal used the same command profile frozen for preflight:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

It completed in 7,763.96 seconds (2:09:23):

```text
4 failed, 5937 passed, 14 skipped
```

The failures were:

1. `tests/test_contextual_expert_aggregation_audit_parent.py::test_exact_rejected_parent_returns_typed_lock_evidence_without_future_input`
2. `tests/test_sec_audit_plan.py::test_cutoff_requires_both_filing_and_eastern_acceptance_by_2024`
3. `tests/test_sec_gemma_lean_runner.py::test_runtime_modules_share_exact_verified_requests_identity`
4. `tests/test_sec_gemma_online_risk_overlay_contract.py::test_bound_source_files_match_literal_sha256_pins`

The six focused v3.9 test files themselves passed within the complete run. A
separate focused run before the complete suite had also passed all 130 v3.9
tests. This does not override the complete-suite failure.

## Failure diagnosis

### Historical parent verifier is branch-bound

The first test also failed alone. The only live semantic dependency that
differs from its rejected historical parent is `.gitattributes`: commit
`3c59666` later added `*.py text eol=lf`. Waiving only that mismatch in a
read-only diagnostic made the test pass. The frozen historical verifier must
not be weakened; its success-path test instead needs explicit compatible-
revision scoping while a drift-rejection test remains active.

### SEC cutoff fixture uses a now-forbidden spelling

The second test also failed alone. Its old fixture uses
`2025-01-01T00:30:00+00:00`. Commit `281aec0` intentionally restricted the
parser to SEC 14-digit Eastern wall-clock labels or the exact SEC `Z`
spelling, and dedicated parser tests reject `+00:00`. The equivalent allowed
Eastern label `20241231193000` preserves the intended 2024 cutoff and the
expected bounded count of 44. Production parsing should not be loosened.

### Requests identity depends on collection order

The third test passed alone and with all lean-acquisition tests, but failed in
the complete collection order. An earlier feature-assembly-store test imports
`sec_filing_gemma_ollama` while it holds ambient Requests object A. Lean
acquisition later calls `load_allowed_requests()`, installs audited Requests
object B in `sys.modules`, and production binds B. The runner then correctly
fails closed because Ollama still holds A. Importing all six v3.9 test modules
does not change any of these object identities. The repair belongs in shared
runtime/test setup, not in v3.9 and not in the runner's safety check.

### Five source pins predate later committed source changes

The fourth test also failed alone. It reports the first mismatch, but a full
read-only pin audit found five:

| Role | Frozen pin | Current committed source hash |
|---|---|---|
| `sec_filing_gemma_contract` | `d5a268da138862b510b0f12b139a04fa91a91d7deba4fe2b28666609c03318e4` | `a51a96a9b763fd3f16cb6461b7dfca58d88cafd5862cff7cc5d18d990994a68e` |
| `sec_filing_gemma_preprocessor` | `febdbaa2fa5f6ecd528fdc9642614b0f8fd8df79f0f9c7ea5c398d89d1be2544` | `0f3499424250d45e2371ab93333cdf0d0af898d67f82eb2fe118847aab4d8d20` |
| `sec_filing_gemma_corpus` | `74831feadcae050eee497da0a3405d65a5c4649a59830bf3f768ab4d35f9164b` | `b0dcb9780f159b655b0a9708b5485ac02c141daac25b418a311fa380bfb4aa83` |
| `sec_point_in_time` | `ccdfc7514fc94223fc9b4f1f57975ac17bcd0f8e5c2e82b10c9f970d11248854` | `ae3041328417ffd0fdf3489332ebfda5b5615ebf01108628de27390872a37153` |
| `sec_filing_content` | `70c969c8eee82e82c0ea1a8a5e424178122fdba8ac1b12ed5a12e207177e10bf` | `c56f874c8aa0c0af6a5c03e6fe0e4c65d42c83f19384b189f3e06ac8c12ffd99` |

The overlay pins were last committed before v3.1 changed the first four
sources and v3.4 changed filing content again. Blindly replacing frozen
historical pins would rewrite the meaning of the old approach. Its current-
source success test needs compatible-revision scoping or a separately reviewed
successor contract.

## Preservation and decision

The exact twelve-file v3.9 code and tests are preserved, but not promoted, at
commit `030e30849364cd7f842c612b66ba68771dcfb7b1` on branch
`codex/aapl-sec-gemma-lean-science-v3-9-offline-draft`. That commit is an
offline draft, not the preregistered passing implementation commit.

V3.9 is rejected before promotion as the preregistered passing implementation.
Running its one-shot preflight after learning that the exact embedded suite
must fail would waste more than two hours and could not authorize development.
The correct next step is to repair and verify the shared baseline on a separate
branch, then create a new preregistration that may reuse the audited v3.9
design without claiming that v3.9 passed.
