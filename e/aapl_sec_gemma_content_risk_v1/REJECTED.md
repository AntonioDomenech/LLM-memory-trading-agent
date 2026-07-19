# SEC/Gemma corroborated-content risk v1: rejected before market

Status: **terminally rejected at the corrected local-model pilot**. This is a
real model-usability failure, not a timeout, test-count, import, or bookkeeping
failure. It is not a trading-performance result because no price was opened.

## What happened

The original request used a 6,144-token context and a 512-token output ceiling.
A non-SEC synthetic diagnostic received HTTP 200 and a normal Ollama response,
but ended with `done_reason == "length"` and exactly 512 evaluated output
tokens. That proved mechanical truncation. The old directory remains preserved
with 44 completed invalid checkpoints, one sealed in-progress marker, and 30
missing calls; none of its completed rows contains a raw-output hash, byte
count, timing, or usable extractor object.

One fresh pilot then changed only the mechanical allowances to an 8,192-token
context and a 1,024-token output ceiling. The model, prompt, schema, anonymous
filing sentences, seed, trading rule, dates, costs, and gates stayed fixed. The
new requests have distinct hashes and a fresh checkpoint directory.

All six fixed pilot calls completed without a transport or output-cap failure:

| Ordinal | Sequence | Status | Seconds | Output bytes |
|---:|---:|---|---:|---:|
| 1 | 124 | Invalid schema/evidence | 22.259392 | 2,236 |
| 15 | 138 | Invalid schema/evidence | 20.854458 | 2,497 |
| 30 | 153 | Invalid schema/evidence | 20.651650 | 2,431 |
| 45 | 168 | Invalid schema/evidence | 21.654574 | 2,527 |
| 60 | 183 | Invalid schema/evidence | 20.779318 | 2,486 |
| 75 | 198 | Invalid schema/evidence | 17.331759 | 1,845 |

Each call has a distinct raw-output SHA-256 and timing record, but none has a
valid extractor-output hash. In plain language, Gemma answered normally, yet
every answer failed the exact JSON/schema/evidence contract needed to turn it
into a trusted trading input.

## Exact gate and decision

The frozen pilot required at least five valid outputs out of six. The result was
**0/6**, so the gate failed. The remaining 69 calls were not opened, and the
68-of-75 full-batch usability gate was not evaluated.

No market data, price row, target exposure, trade, return, cost, drawdown, or
buy-and-hold comparison was opened or computed. No 2019-or-later data, paid
API, broker, or real-money action was used.

This does not prove that filing contents can never help trading. It proves that
this exact local reader cannot reliably create the frozen structured input. The
SEC/Gemma correction loop therefore ends here. The next experiment must test a
materially different trading hypothesis instead of creating another cosmetic
version.

The machine-readable safe record is `PILOT_REJECTION.json`. It contains hashes,
counts, timings, and statuses only - no raw SEC text, raw model output, or SEC
contact information.

Nothing here authorizes real-money trading.
