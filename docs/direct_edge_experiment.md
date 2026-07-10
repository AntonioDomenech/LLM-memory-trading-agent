# Direct CASH-edge development experiment

This experiment asks a narrow question: using only information available by a
completed market close, can a fixed model identify five-session periods when
holding cash should beat holding AAPL after realistic trading costs?

It is development evidence, not a test result. All fitting, candidate
comparison, and selection stop at 31 December 2018. The runner has no command
that can load or evaluate 2019+ market observations.

## Frozen contract

The system evaluates eight candidates: four fixed probability/expected-edge
gates for a price-only model and the same four gates for a price plus market
sentiment model. Decisions are LONG 100% or CASH 100%; shorting, leverage,
margin, fractional exposure, and cash interest are disabled. A CASH trigger is
held for five decision rows and fills at the next adjusted AAPL open.
Every accepted block start is stored separately, so adjacent five-row CASH
blocks remain two scored predictions even though their exposure is continuous.

Seven expanding, purged, two-year out-of-fold blocks cover 2005-2018. A block
may train only on labels whose AAPL open-to-open outcome matured strictly
before that block. The model and all preprocessing remain frozen throughout
the block.

Every candidate is simulated on the same ledger as buy-and-hold at both 5 bps
and 10 bps of slippage per trade leg. A candidate can be selected only if its
full development gates pass at both costs. The scorer proves the declared
cost directly from each reference price, buy/sell fill, share delta, and
slippage value, and reconciles daily returns to ledger equity. Ranking is
fixed in advance:

1. highest 10 bps weakest-fold active log edge;
2. highest 10 bps total active log edge;
3. fewest 10 bps CASH episodes;
4. lexical candidate ID.

If no candidate passes, selection is `null` and final refitting is skipped.
The known current preflight result is therefore preserved honestly rather than
promoting the least-bad failure.

## Inputs and point-in-time sentiment

The public runner accepts:

- AAPL/SPY/QQQ prices replayed from the committed, checksum-verified CFTC
  through-2018 artifact. The prior CFTC strategy result is not reused.
- A local `context_daily.parquet`. One fixed DuckDB query reads only IWM, VIX,
  and TNX rows dated no later than 2018-12-31. It performs no network or API
  call.

Injected price and context frames are supported for tests. Unlike the bounded
public Parquet loader, injected inputs are rejected if they contain any
post-2018 row.

Sentiment context is joined only on the exact AAPL session date. Missing or
nonfinite context is never forward-filled, interpolated, or changed to zero;
the augmented model falls back exactly to the price-only prediction.

The market-sentiment family has an additional selection hurdle. On identical
sentiment-ready rows it must, at both costs:

- have strictly lower Brier score;
- have strictly lower expected-edge mean absolute error;
- have strictly higher total active log edge; and
- not worsen the weakest fold.

Macro data is excluded because stored placeholders and historical-vintage
safety are not proven. News is excluded because event availability and
publication-time semantics are not point-in-time safe. These exclusions are
recorded in every sealed manifest.

## Sealed evidence

A completed run writes a flat immutable artifact containing:

- bounded price and context CSVs;
- feature and input provenance;
- every out-of-fold prediction and candidate target;
- every candidate and buy-and-hold ledger at both costs, including the
  common-support ablation ledgers;
- all candidate metrics, gates, and the deterministic selection;
- final model state only when a candidate passes, fitted twice and required to
  be byte-identical;
- selection manifest, report, byte-preservation rule, and exact checksums.

The verifier recomputes checksums, manifest hashes, every trigger and CASH
block, both cost ledgers, predictive and economic metrics, gates,
common-support ablations, candidate ordering, and deterministic selection. A
committed verification also proves that the recorded clean source commit and
source-file hashes are ancestors of the artifact commit. The complete public
workflow, including input loading and verification, is subject to the
one-hour limit; a temporary seal is promoted atomically only after it passes.

## Commands

The public run requires clean committed source:

```powershell
python -m agent_benchmark.direct_edge_experiment run `
  --repo-root . `
  --price-artifact e/cftc_cot_v1/<committed-run> `
  --context-parquet data/warehouse/parquet/context_daily.parquet `
  --output-dir e/direct_edge_v1
```

Verification is offline:

```powershell
python -m agent_benchmark.direct_edge_experiment verify `
  --development-artifact e/direct_edge_v1/<run-id>
```

Exit code `0` means a candidate passed; exit code `2` means the complete run
was sealed correctly but selection was null.
