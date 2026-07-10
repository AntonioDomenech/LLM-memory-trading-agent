# Frozen pre-2019 one-session regime-consensus experiment

This experiment is a development and model-selection gate, not unseen-test
evidence.  It uses only decisions whose AAPL entry fill is in 2005-2018 and
only labels that had matured before each fixed walk-forward fold.  No 2019 or
later market row is accepted, including a future row whose values are empty.

The model has three independently fitted two-state HMM heads: AAPL state,
broad-market state, and fear/breadth state.  A CASH decision requires at least
two heads to agree, all three heads to be ready, and one of two frozen gates:

* `p55_e5`: consensus CASH probability at least 0.55 and predicted 10-bps
  active-log edge at least 0.0005.
* `p60_e5`: consensus CASH probability at least 0.60 and predicted 10-bps
  active-log edge at least 0.0005.

Each signal moves from 100% AAPL to cash for one next-open-to-following-open
session.  There is no shorting, borrowing, leverage, cash interest, or partial
position.  The same adjusted-open simulator evaluates the strategy and
buy-and-hold at both 5 bps and 10 bps per executed leg.

For each gate the runner also executes an AAPL-only diagnostic on exactly the
same out-of-fold dates.  A full three-head candidate can pass only if its own
economic and predictive gates pass at both costs and the paired ablation shows
strictly lower Brier score, no worse edge MAE, strictly higher total
active edge, and no worse weakest-fold edge at both costs.  Passing candidates
are ranked deterministically by 10-bps weakest-fold edge, 10-bps total edge,
10-bps Brier improvement, fewer cash days, and finally candidate ID.  If none
passes, selection is null and no final model is fitted.

## Inputs and exclusions

The public runner replays the already committed, checksum-verified AAPL/SPY/QQQ
price input and performs one bounded local Parquet query for only IWM and VIX
rows through 2018-12-31.  The raw context file hash, exact query, and bounded
result hash are recorded.  The runner performs no network, API, or LLM call.
Its predictions do use 21 local deterministic HMM fits (plus six repeated
through-2018 refits only if a candidate passes); these are numeric local
models, not pretrained language models.

Macro data, news/events, CFTC positioning, and TNX are frozen exclusions.  They
cannot enter a feature, candidate, model state, or selection result.  This
keeps unavailable-vintage, unsafe-publication-time, sparse-event, and unused
rate inputs out of the evidence.

## Sealed evidence

The run is built in a temporary directory, checksummed, fully replay-verified,
and only then atomically promoted.  The artifact contains the bounded inputs,
feature provenance, exact out-of-fold predictions, all 21 HMM fit states,
candidate targets, every strategy and buy-and-hold ledger at both costs,
candidate metrics/gates, the ablation, and the deterministic selection
manifest.  Verification rebuilds the feature frame, all folds and predictions,
targets, ledgers, scores, gates, ablation, and selection from the sealed inputs
and requires exact bytes.  With a repository path it additionally verifies the
recorded clean source commit and Git-blob hashes.

The complete run, including verification, has a hard one-hour wall-clock
deadline.  A timeout or verification failure removes the temporary artifact.

```powershell
python -m agent_benchmark.regime_consensus_experiment run `
  --repo-root . `
  --price-artifact e\cftc_cot_v1\<committed-run> `
  --context-parquet data\warehouse\parquet\context_daily.parquet `
  --output-dir e\regime_consensus_v1

python -m agent_benchmark.regime_consensus_experiment verify `
  --repo-root . `
  --development-artifact e\regime_consensus_v1\<run-id>
```

Exit code 0 means a candidate passed and was selected.  Exit code 2 means the
experiment completed correctly but rejected every candidate.
