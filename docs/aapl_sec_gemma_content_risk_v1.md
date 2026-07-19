# AAPL SEC/Gemma corroborated-content risk v1

Status: **preregistered; no Gemma filing output and no content-rule return has
been opened**.

Branch: `codex/aapl-sec-gemma-content-risk-v1`

Parent result commit: `ca599a7bf6569ee03472a3b22c994e9aad95dcb2`

## Question

Can a small local filing reader avoid only the Apple post-filing periods that
contain corroborated business warnings, instead of making the failed decision
to leave AAPL after every filing?

Gemma is an information extractor, not the trader. It never receives Apple,
AAPL, dates, prices, returns, labels, actions, thresholds, or later results.
The deterministic rule below owns every trading decision.

## Frozen source and prompt set

Use the already authenticated private V3.8 SEC receipts and blobs. Make no new
SEC request and never publish the readable SEC contact. The 75 development
events are the exact sequences 124 through 198 used by the calendar baseline.
For the first 2000 filings only, sequences 122 and 123 supply the immediately
prior 1999 same-form text; they can provide context but can never become a
trade or outcome row.

For each event:

1. verify the complete-response bytes, selected-text slice, and normalized-text
   hash against its receipt;
2. use only the current filing and the latest strictly earlier filing of the
   same form;
3. run the existing deterministic identity/date/number/market redactor;
4. remove the existing frozen security-direction sentences; and
5. send only anonymous `C####` and `P####` sentence IDs and text to Gemma.

The offline reconstruction already completed without model or market-outcome
access. It produced:

- 77 authenticated normalized documents;
- 75 model requests, sequences 124 through 198;
- request sizes from 14,576 through 23,920 canonical JSON bytes;
- 70 through 72 retained sentences per request;
- four current and four prior security-direction sentences removed in total;
- canonical commitment bytes: `29101`;
- canonical commitment SHA-256:
  `6ce69886950b42ec21cb9cf4ac281456fae6e89fce5d3883d126513f832d4175`;
- ordered model-payload-hash SHA-256:
  `79bfd807b23312e81134417f1ddc9351d42e44b9af30354e5722e87068a63533`.

The model and request are frozen:

| Item | Value |
|---|---|
| Model | local `gemma4:12b` |
| Installed model manifest SHA-256 | `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c` |
| Existing semantic runtime fingerprint | `816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77` |
| System-prompt SHA-256 | `9ed8496ed101c138cdbee162bdf6dfd53434f6f1e0c64fc93d844405d6eae9f7` |
| Output-schema SHA-256 | `1707ae581abb1a256dfb1ee8f51efd9dd67df5d9b3e8f7c22ee2d92b67b6f82b` |
| Temperature / seed | `0 / 0` |
| Context / output cap | `6144 / 512` tokens |
| Thinking / streaming | `false / false` |
| Retry / repair / pull / fallback | `0 / 0 / 0 / 0` |

Record the local Ollama version and require the exact installed manifest digest
before and after the batch. Do not publish raw `/api/show` bytes or local model
paths.

The exact extractor schema contains ten business dimensions and six flags.
Every claimed dimension or flag may cite only supplied sentence IDs. Invalid
JSON, invalid schema, invalid evidence IDs, an unusable document, or a failed
local call creates one permanent unavailable row and means **stay LONG**. It is
not retried.

## Frozen trading rule

At an eligible filing's conservative availability-session close, schedule one
20-session CASH episode if and only if all three conditions are true:

1. `document_quality` is `usable` or `thin`;
2. at least one of these five adverse flags is present:
   `new_material_risk`, `guidance_withdrawn`, `liquidity_stress`,
   `restructuring_or_impairment`, or `internal_control_weakness`; and
3. at least one of the ten dimensions has either
   `current_impact == "unfavorable"` or
   `change_vs_prior == "deteriorating"`.

`management_transition` alone is not adverse. The conjunction requires two
different kinds of warning: an explicit flag and a negative business
assessment. There is no score, fitted model, threshold search, or manual
interpretation.

A passing close `t` sells at adjusted open `t+1`, stays cash for exactly 20
open-to-open intervals, and buys AAPL at adjusted open `t+21`. A filing while
an episode is scheduled or active is audited but cannot extend it. Otherwise
the account remains long AAPL. Cash earns no interest.

## Operational pilot and checkpoints

The six fixed pilot ordinals are `1, 15, 30, 45, 60, 75`. They are ordinary
experiment calls and are reused, never called twice. The pilot tests only
local completion, schema validity, and runtime; it may not inspect prices or
change the trading rule after seeing semantics.

Continue through all 75 calls when at least five of six pilot outputs are
valid. A schema-invalid pilot is sealed, mapped to LONG, and is not a reason to
invent a successor. A local transport failure is logged without retry and the
process continues with later uncaptured calls when possible. The final 90%
valid-output gate decides whether the reader was usable.

Save one atomic checkpoint after every call. On resume, completed valid or
invalid calls are never repeated; only missing ordinals run. Print after every
call: ordinal, valid/invalid status, seconds, completed count, rolling median,
and estimated remaining time. Runtime estimates are warnings, never trading
gates. There is no total hard timeout.

Offline prompt construction measured 215 seconds in the simple sequential
check. The six-call pilot is expected to take about 2-3 minutes. If healthy,
the complete 75-call batch is expected to take about 25-35 minutes on the
current RTX 3080, updated from observed pilot times.

## Fair development comparison

This is repeated historical development, not a pristine unseen test. Gemma's
reported pretraining knowledge reaches into 2025, so even anonymous filing
text may be recognizable; the model is therefore not proof of a truly unseen
historical reader.

Run one continuous account from `2000-01-03` through `2018-12-31` at 5 and 10
basis points per changing leg. Compare with:

- same-ledger always-long AAPL; and
- the already rejected calendar control that enters cash after every filing.

Use the same EUR 1,000 start, adjusted-open fills, corporate-action adjustment,
dates, final valuation, ledger, costs, no-interest cash, and 0/1 exposure.
The physically bounded market input remains the 4,986-row file ending
2018-12-31 with literal SHA-256
`9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1`.
Do not open any 2019-or-later filing or price.

## Development gates

Open 2019-2023 only if every gate passes:

1. at least 68 of 75 Gemma outputs are schema-valid;
2. at least eight complete CASH episodes occur across at least five entry
   years;
3. cumulative relative wealth versus AAPL is positive at both 5 and 10 bps;
4. content-rule active log edge minus the all-filings calendar control is
   positive at both costs;
5. at 10 bps, winning calendar years outnumber losing years and both mean and
   median annual excess are positive;
6. at 10 bps, mean and median completed-episode edge are positive;
7. at 10 bps, active log edge is positive separately in `2000-2008` and
   `2009-2018`;
8. at 10 bps, active log edge remains positive after removing the single best
   active calendar year;
9. at 10 bps, aggregate edge across negative-AAPL calendar years is positive;
10. same-ledger, chronology, input-hash, no-short, no-leverage, no-borrowing,
    and nonnegative-cash checks all pass.

Ties are separate. Report all required returns, annual results, drawdowns,
trades, cash time, turnover, costs, negative-AAPL years, output validity,
dimension/flag frequency, episode results, model timings, hashes, and safety
proofs.

Failure preserves this branch and stops before 2019. Do not make a cosmetic
v1.1 or tune this conjunction after seeing its returns. A successor is allowed
only for a genuinely different trading hypothesis.

Nothing authorizes real-money trading.
