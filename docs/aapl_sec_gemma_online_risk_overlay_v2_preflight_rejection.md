# AAPL SEC/Gemma online risk overlay v2 preflight rejection

## Status

The preregistered `v2` branch is rejected at local-runtime preflight. No
official SEC document or Yahoo market snapshot was acquired for this approach,
Gemma performed no filing extraction, no return was opened, and no stage was
scored.

This is not a trading result. It is a fail-closed implementation finding that
must remain separate from any successor approach.

## What matched

Read-only inspection of the installed `gemma4:12b` runtime reproduced the
frozen:

- model manifest SHA-256;
- config digest;
- ordered four layer digests and the content hash of every layer;
- two active `FROM` blobs in the generated Modelfile;
- Ollama version `0.32.0` and its raw version-response hash; and
- canonical `model_info` SHA-256.

These checks establish that the installed model content is the intended local
model.

## What failed

The frozen contract requires byte-for-byte equality for the raw Ollama
`/api/show` response.

| Item | SHA-256 |
| --- | --- |
| Preregistered raw show response | `8ab2bd35bfd63bc37b9dd7e932ee38f3ad4dfa773d0767baf4a7d08eea7428e0` |
| Repeated current raw show response | `5f56fb0fb2214ddcb9fa21c66aa31e37297f553e8758aeda5958f0f287d70893` |

Repeated non-generative loopback probes reproduced the current hash. Request
body variants also reproduced it. Because the literal runtime fingerprint
includes the preregistered show-response hash, its identity cannot pass.

## Consequence

The `v2` contract says that changing any frozen field creates a new approach
and requires a new branch before semantic extraction or scoring. Therefore:

1. this branch remains preserved as a preflight rejection;
2. none of its four one-shot effectful attempts is consumed by this read-only
   inspection; and
3. a successor branch must preregister the corrected exact runtime identity
   before any SEC, market-value, or Gemma effect.

The successor may reuse the offline implementation work, but it must have its
own branch, contract hash, attempt identities, preregistration commit, and
pushed implementation binding.
