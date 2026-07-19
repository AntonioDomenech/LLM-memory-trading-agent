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
