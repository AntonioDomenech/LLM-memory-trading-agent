# Post-rejection sparse-tail diagnostic

This read-only diagnostic asked whether the rejected sector-breadth model had
useful information only in its most confident development predictions. It did
not reopen 2019-2023 or any 2024+ outcomes, and it is not a newly selected
policy because the tail thresholds were examined after seeing the original
development failure.

## Best observed tail

The least-bad diagnostic used a fold-relative 99th-percentile expected-edge
threshold plus predicted CASH probability at least 0.50. Each fold was refit
using only decisions and labels that had matured before that fold began. Its
threshold was calculated only from that fitted model's eligible training
predictions. Accepted starts created non-overlapping five-session blocks.

| Development metric, 2005-2018 | Result |
|---|---:|
| Accepted five-session blocks | 17 |
| Cash days | 85 |
| Contiguous ledger episodes | 14 |
| Label wins | 8 / 17 (47.1%) |
| Mean episode active log edge at 10 bps | +0.00516 |
| Total active log edge at 5 bps | +0.10769 |
| Relative wealth at 5 bps | +11.37% |
| Total active log edge at 10 bps | +0.09369 |
| Relative wealth at 10 bps | +9.82% |
| Positive two-year folds | 3 / 7 |
| Positive calendar years | 4 / 14 |
| Weakest fold active log edge at 10 bps | -0.06958 |

The seven 10-bps fold edges were approximately `+0.07310`, `+0.05397`,
`-0.06958`, `0`, `+0.03620`, `0`, and `0`. Eight of fourteen years had no
effective trade. A 99th-percentile edge-only variant was also positive
(`+0.05474` total active log edge at 10 bps), but the 95th- and 97.5th-
percentile variants were decisively negative.

## Integrity and decision

- Refitted out-of-fold predictions matched the committed development
  predictions (maximum probability difference `2.88e-9`; maximum expected-
  edge difference `1.47e-13`).
- No label maturing after 2018-12-31 was loaded. Starts stopped on 2018-12-20,
  and six incomplete tail rows remained redacted.
- The signal is too concentrated and unstable to promote: fewer than half of
  accepted labels won, only three folds were positive, and small threshold
  relaxations destroyed the result.

Preserve the 99th-percentile tail as a possible future ensemble component,
but do not treat it as an independent successful strategy or use it to justify
opening later outcomes.
