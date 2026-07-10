# Development attempt 2: pre-tolerance diagnostic only

Run `cftc-cot-development-20260710T174142Z-b4c7e39a` is internally intact and
rejected all four candidates without opening 2019-2023. Its exact 29-file
artifact and checksums are preserved.

It must not be treated as the final current-code evidence because the source
was subsequently corrected in two ways:

1. sealed price CSVs now use pandas round-trip float parsing, which reproduces
   all saved policy and ledger files byte-for-byte;
2. annual, fold, and rolling wins now require active-log edge greater than
   `1e-12`, so floating-point dust cannot count as success.

The fixes do not rescue a candidate: replay still gives `selected_variant_id =
null`. They do change the candidate-results hash and several diagnostic win
rates, so one clean bounded development reproduction is required before the
branch is finalized.
