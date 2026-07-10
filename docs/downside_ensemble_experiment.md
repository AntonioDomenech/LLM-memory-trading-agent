# Frozen downside-ensemble development contract

This experiment has one purpose: use only information ending in 2018 to choose
at most one AAPL LONG/CASH downside candidate. Its results are internal
development and selection evidence. They are not an unseen test result.

The runner has no command that can evaluate a later period. It reads a
checksum-verified copy of the earlier CFTC development artifact, independently
revalidates its exact price and CFTC boundaries, and removes any CFTC report
whose release date falls after 2018 before constructing features.

## Chronology

The model is evaluated in seven fixed two-year blocks: 2005-2006, 2007-2008,
2009-2010, 2011-2012, 2013-2014, 2015-2016, and 2017-2018. Before each block it
fits one price-only forest and, when enough causally available records exist,
one price-plus-CFTC forest. A training label must mature strictly before the
block's first decision. Neither model changes inside its two-year block.

The first block can legitimately have too little complete CFTC history for the
fixed forest. In that case the augmented candidate is the exact price-only
candidate for the complete block. The runner does not weaken the forest,
impute sentiment, or manufacture a CFTC model.

## Fixed candidates

There are two model families and four risk gates, making eight candidates:

- price-only;
- price plus the three causal CFTC z-scores, with exact price fallback;
- downside-risk multiples 1.10, 1.20, 1.30, and 1.40.

For a row handled by a given model, its causal baseline is that fold model's
training crash prevalence. A new CASH episode is triggered when:

```text
predicted downside probability
    >= fixed risk multiple * causal fold-training crash prevalence
```

The predicted mean five-session return remains an audited diagnostic. It is
not a trading gate.

An earlier structural preflight tried absolute probabilities 0.25, 0.30, 0.35,
and 0.40 together with predicted mean return at most -0.005. It produced zero
CASH targets: the mean prediction never reached that cutoff and the maximum
probability was only about 0.26. That nonfunctional trigger is recorded as
rejected in every manifest. It was replaced before sealing using pre-2019
development information only.

A trigger starts five consecutive decision rows in CASH. Triggers during an
active episode are ignored. A trigger on the first row after expiry may begin
another episode. Missing price features cannot trigger CASH. Missing, stale,
incomplete, or structurally unavailable CFTC information copies the price
model's prediction and causal baseline exactly.

## Evaluation and selection

Every candidate runs through the same next-adjusted-open, continuous-account
ledger at both 5 and 10 basis points. The benchmark uses the identical fill
sessions and cost assumption. Both ledgers must prove exact binary exposure,
no leverage, no short position, no negative cash, and no margin interest.

The development scorer requires material total and median edge, positive
behavior across years, rolling windows and at least five of seven folds,
sufficient but bounded CASH activity, performance not dominated by one year,
positive behavior in negative buy-and-hold years, and a Brier score better than
the causal training-prevalence baseline. Every named gate must pass at both
costs.

An augmented candidate has an additional ablation. The price and augmented
models are allowed to start episodes only on the same CFTC-ready rows; an
already active episode continues normally. At both costs the augmented policy
must have strictly greater total active-log edge and no worse weakest-fold
edge. Otherwise the augmented candidate is ineligible.

Passing candidates are ranked by:

1. highest weakest-fold active-log edge at 10 bps;
2. highest total active-log edge at 10 bps;
3. greatest causal Brier improvement;
4. fewer CASH days;
5. price-only before augmented;
6. lexical candidate id.

If a candidate is selected, its required model or models are fitted twice on
all labels causally mature by 2018-12-31. Their serialized states must be byte
identical. No gate, feature, risk multiple, or forest parameter may change
afterward.

## Evidence and runtime

The artifact contains normalized bounded inputs, all out-of-fold predictions,
causal baselines and fallback reasons, actual and common-support targets, every
candidate and benchmark ledger at both costs, complete results, the selected
model state when one exists, the selection manifest, and an exhaustive SHA-256
checksum file. Files are written as exact LF/binary payloads and immediately
read back and verified. The verifier can also read committed Git blobs.

The hard runtime budget is 3,600 seconds. The runner checks it after input
validation, feature construction, walk-forward fitting, every simulation,
refitting, sealing, and verification. A timeout rejects the run; it never
reduces the tree count, drops a fold, skips a candidate, or changes a gate.

Example after the source branch and input artifact are committed:

```powershell
python -m agent_benchmark.downside_ensemble_experiment run `
  --repo-root . `
  --input-artifact e/cftc_cot_v1/<bounded-development-run> `
  --output-dir e/downside_ensemble_v1
```

Verify a completed artifact locally or, with `--repo-root`, from committed Git
blobs:

```powershell
python -m agent_benchmark.downside_ensemble_experiment verify `
  --development-artifact e/downside_ensemble_v1/<run>
```
