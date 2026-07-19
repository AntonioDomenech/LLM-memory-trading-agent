# AAPL SEC/Gemma content-risk v1 execution log

## 2026-07-19 pilot transport deviation

This record was written before any market price or return was opened.

The first Python worker continued after its parent shell was terminated. It
completed the six fixed pilot calls once, using the exact frozen prompts and
the exact pinned `gemma4:12b` manifest. The runtime identity was unchanged
before and after those calls.

All six checkpoints were marked invalid by the old exact-key Ollama envelope
parser before extractor JSON was read. Each has no raw-output hash, output byte
count, model timing, or semantic extractor object. This uniform pre-output
failure is consistent with the independently identified wrapper-metadata bug.
No pilot call will be repeated.

The corrected implementation tolerates inert wrapper metadata while retaining
the frozen request, schema, evidence, model, and one-call rules. The remaining
69 requests will continue. The six pilot rows remain permanently unavailable
and therefore LONG. This is an explicit deviation from the preregistered
five-valid-of-six operational continuation rule, not a change to the trading
rule. The final requirement of at least 68 valid outputs among all 75 remains
unchanged, so at least 68 of the remaining 69 calls must be valid.

Any result artifact must report both facts: the ordinary pilot gate did not
pass, and the sealed pre-output parser continuation was used. No result from
this run may be described as perfectly preregistration-compliant.

## Mechanical completion-budget correction

The continuation was stopped after 44 old-request calls plus one sealed
in-progress marker because every completed call again failed before output
extraction. A separate non-SEC diagnostic request then recorded the normal
Ollama envelope and `done_reason == "length"`, with exactly 512 evaluated
tokens. This proved the problem was the frozen completion ceiling, not filing
semantics or wrapper metadata. The old checkpoints remain preserved and are
not inputs to the corrected run.

Before any corrected filing output or market value was opened, the same 75
anonymous sentence payloads were rebuilt with only these mechanical option
changes:

- context allowance: `6144 -> 8192` tokens;
- completion allowance: `512 -> 1024` tokens.

The corrected request set has:

- request sizes `14,577` through `23,921` bytes;
- commitment bytes `29,101`;
- commitment SHA-256
  `dda7adb2fbd5f662b9abadc8122d7ae03fb19128951351369ba40087a67672c8`;
- ordered payload-hash SHA-256
  `aed8bd8695622132a4c53a04c514c04a00f72835c69de2828256c99a0d3899dd`.

The SEC text, anonymized sentences, model digest, prompt, output schema,
temperature, seed, trading rule, dates, costs, and gates are unchanged. The
corrected request hashes make these new requests distinct from the truncated
old requests. They use a fresh checkpoint directory and the ordinary fixed
five-valid-of-six pilot gate.

## Terminal corrected-pilot rejection

The earlier plan to continue the remaining 69 old-request calls is superseded
by the completion-budget diagnosis and the fresh corrected pilot below. The old
512-token checkpoints remain sealed and were not reused or repeated.

The corrected 8,192-context / 1,024-output pilot made exactly the six fixed
calls at ordinals `1, 15, 30, 45, 60, 75`. All six completed normally in
17.331759 through 22.259392 seconds. Each recorded a raw-output hash, output
byte count, and model timing; the before/after Ollama version, model manifest,
and semantic runtime fingerprint matched exactly. There were zero transport
failures and zero output-cap failures.

Nevertheless, all six outputs returned
`invalid_json_schema_or_evidence_no_retry_no_repair`. None produced a valid
extractor-output hash. The frozen pilot required at least five valid outputs,
so the observed `0/6` is a terminal model-usability rejection. The remaining 69
calls were not opened.

No market data or price row was opened. Trading actions, returns, costs,
drawdowns, and buy-and-hold performance were not computed. No 2019-or-later
data, paid API, broker, or real-money action was used. This branch will not
create another SEC/Gemma correction version; research moves to a materially
different trading hypothesis.

The privacy-safe terminal evidence is preserved in
`e/aapl_sec_gemma_content_risk_v1/PILOT_REJECTION.json` and `REJECTED.md`. Those
artifacts contain no raw SEC text, raw model output, or SEC contact information.
