# AAPL SEC/Gemma lean science v3.10 preflight rejection

## Status

V3.10 stopped at its one-shot zero-effect preflight on 2026-07-18
Europe/Madrid. The redacted terminal code is `legacy_record_invalid`.

This is not a trading result. Development science, Yahoo market data, Ollama
identity checks, Gemma generation, prediction, performance calculation,
confirmation, live-style evaluation, broker interaction, and real-money
execution were never authorized or opened.

The immutable V3.10 identities are:

| Item | Value |
|---|---|
| branch | `codex/aapl-sec-gemma-lean-science-v3-10` |
| preregistration commit | `bf66bb95f1ab44bb36b98523f6246761c23bcca3` |
| implementation commit | `c2776bae573faa7963bd31f89f2e570ce704a207` |
| implementation tree | `37ffbd7992ce658a86b6fa917ad843bc80c6d0db` |
| implementation parent count | `1` |
| implementation parent | `bf66bb95f1ab44bb36b98523f6246761c23bcca3` |
| implementation changed paths | exactly the 12 preregistered additions |
| implementation source-inventory SHA-256 | `e26a3029c814996f878ea0a3a4901bfc6e6c3bd83e01e322c6d6c942fed2587d` |
| public failure artifact | `e/aapl_sec_gemma_lean_science_v3_10/DEVELOPMENT_PREFLIGHT.json` |
| public artifact self-hash | `b0410eeefaacfdcfbb9a414bc969c6df89e2b6b69315fb4febf9853a1a507254` |
| public artifact literal SHA-256 | `2de61b0fca9b1696d96c0eefb3ea76892534e0ab8a7e1a066ffa1de7f850b9ee` |

The local branch, remote branch, commit, tree, only parent, clean worktree,
exact changed paths, source blobs, preregistration blob, predecessor equality,
and local-import closure all authenticated before the one-shot preflight.

## Offline implementation evidence

The final latest-approach collection contained 167 unique V3.10 nodes. Every
node passed:

```text
167 passed in 188.65s
```

The collected V3.10 node-list SHA-256 was
`8e4618e6118d2bb1e5c464160cc45bd37834a66b82e1c2fccd6e0ef7effaff61`.
The frozen shared-dependency collection still reproduced 655 nodes and its
preregistered SHA-256
`1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c`.
The isolated Requests-identity sentinel reproduced exactly one node and its
preregistered SHA-256
`465bbb7fb1bd0633502006db2b84f6adab6ca7542d8c4e3d75b13d6cb7e73229`.

Those last two checks confirmed selector identity only in this final pass. The
one-shot preflight did not execute the qualification phases because the source
bridge failed first, as required by the frozen gate order.

## Source authentication and bridge findings

Two production-only bridge defects were found and corrected before the
one-shot attempt:

1. On Windows, path and open-handle stat calls expose different meanings for
   `ctime`. The stable-read proof now compares `ctime` only between like path
   snapshots while retaining descriptor identity, size, modification-time,
   and link-count binding.
2. The frozen V3.8 inventory hash was originally serialized with the literal
   member names `sha256` and `bytes`. The inherited V3.9 draft had reconstructed
   those values under different member names. Restoring the historical names
   reproduced the frozen inventory SHA-256 exactly without changing any
   preserved V3.8 evidence file.

After those corrections, read-only production authentication passed over all
1,410 preserved evidence files and 612,601,642 bytes. It authenticated all 199
role manifests, the logical checkpoint
`704a4a9554444201ec9468e74f8c77abe82a3e3320acce29cac0c7ace11dd6fc`,
and the compact replay
`64a7b7776206b008c0dffe26d4a14be9a092281a7834d878332d4f741db05822`.
No source reacquisition was needed.

## Terminal contradiction

The authenticated development cohort contains 75 selected filing rows. For 73
rows, the V3.8 source can be represented by the inherited legacy universe. Two
valid historical rows are different:

- their authenticated Submissions `primaryDocument` value is empty;
- their sealed Submissions filename, SGML filename, and SEC filename are all
  absent; and
- V3.8 therefore uses the internal identity
  `legacy-sequence-1-no-filename` while explicitly not claiming that it is an
  SEC filename.

The V3.10 inherited downstream schema simultaneously requires every universe
record to contain a nonempty safe primary-document filename and constructs an
official primary-document URL from that filename. V3.10 also requires all 75
rows, exact selected-filename parity, a lossless bridge, and no mapping repair.

There is no honest value the bridge can supply for those two rows:

- using the internal fallback as a filename would fabricate an SEC filename
  and URL;
- inventing another filename would fabricate source evidence;
- dropping the two rows would violate the frozen 75-row cohort; and
- changing the schema or mapping after observing the rows would violate the
  pushed preregistration.

The bridge therefore correctly stopped with `legacy_record_invalid`. The
one-shot attempt took about 191 seconds, published only the 613-byte redacted
failure artifact, consumed the attempt, forbade a rerun, and left development
unauthorized.

## External-effect accounting

The V3.10 attempt made no fresh external request. In particular, it made:

- zero official SEC requests;
- zero Yahoo requests;
- zero Ollama identity requests;
- zero Gemma generations;
- zero paid-API calls;
- zero broker or real-money actions; and
- zero confirmation or live-style accesses.

The readable SEC contact was used only to authenticate the already-preserved
private source and was not printed or published. The public failure artifact
passed its schema, self-hash, and private-token scans.

## Additional readiness finding

A separate adversarial review found that the V3.10 development runner would
need a stronger repeated binding between the loaded Python modules, current Git
state, and the authenticated implementation before and after long-running
effects. This path was never reached and was not expanded after the terminal
source contradiction. Any runnable successor must address it before external
development effects.

## Decision

V3.10 is permanently rejected at zero-effect preflight. It must not be rerun,
patched into a passing result, or described as development evidence.

Any successor must use a new `codex/` branch and pushed preregistration. It may
reuse the authenticated 75-filing source without reacquisition, but it must
freeze an explicit nullable filename representation before rebuilding private
rows. A filename-less row must retain its internal selected-document identity,
authenticated complete-submission URL, and selected embedded TEXT bytes without
claiming a nonexistent primary-document filename or constructing a fabricated
primary-document URL. The successor must also bind its executing code at every
external-effect and publication boundary.
