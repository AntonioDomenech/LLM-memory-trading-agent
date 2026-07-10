# Frozen downside ensemble v1 - rejected in development

Official run: `downside-development-20260710T210039Z-3bd40789`

This approach was rejected using only the chronological development period
ending `2018-12-31`. It did not access the reserved 2019-2023 validation
period or any 2024+ outcome, and no candidate was selected or refitted.

The forest slightly improved five-session crash-probability calibration over
its causal training-prevalence baseline:

- price only: Brier `0.109261` versus `0.113256`;
- price plus CFTC positioning: Brier `0.109996` versus `0.112461`.

That statistical improvement did not become a reliable trading edge. Most
candidates lost to buy-and-hold. The best cumulative candidate,
`price_only_r140`, produced active log edge `+0.230798` at 5 bps (`+25.96%`
relative wealth), but it made only three CASH episodes, won only 2 of 14
calendar years and 2 of 7 development folds, and had 252/756-session rolling
win rates of only `8.33%`/`28.03%`. More than 70% of its positive-year edge
came from one year. It therefore failed the minimum activity, annual,
fold-level, rolling, median-year, and concentration gates at both 5 and 10
bps. The price-plus-CFTC variants all had negative cumulative edge and also
failed their common-support ablation.

An earlier structural preflight combined absolute probability gates with a
predicted-mean-return gate and generated zero CASH decisions. That
nonfunctional trigger is recorded in the sealed manifest; it was replaced
before the official run using only pre-2019 development information.

Integrity summary:

- runtime before sealing: `11.5238` seconds, below the one-hour limit;
- model/API calls: `0`; external cost: `$0.00`;
- actions: unleveraged 100% AAPL or 100% CASH only;
- both strategy and benchmark ledgers passed the exact no-leverage checks;
- selected candidate: `null`; final refit: not performed;
- sealed files: `44`; checksum entries: `43`;
- checksum-manifest SHA-256:
  `0fb53d6bdf20374583331cebc0d0ee995a14fb1c9cbf9efd49f2f0873a305aa7`.

The complete predictions, targets, same-ledger simulations, candidate gates,
input provenance, manifest, and checksums are preserved in the official run
directory. This family must not proceed to 2019-2023 confirmation.
