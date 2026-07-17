# AAPL SEC/Gemma lean evidence v3.1

## Status, prior failure, and design-only evidence

This is a new, separately preregistered approach on
`codex/aapl-sec-gemma-lean-evidence-v3-1`. This document must be committed and
pushed before any programmatic v3.1 SEC source-data request, Yahoo value, Gemma
generation, performance calculation, confirmation source, or final source is
opened.

V3 remains terminally rejected at commit
`698a3cf70ead47ce077a3ccb45cffe9ddfbfe7f5`. Its four exact HTTP-200 SEC
catalogue requests exposed a mismatch between a main Submissions `filingTo`
claim and the maximum filing date observed in the referenced historical file.
V3 opened no Yahoo value, model output, performance result, confirmation or
final source, and made no trade. V3.1 does not reinterpret, erase, or retry the
v3 result under changed rules.

Before this preregistration, read-only SEC guidance, search-index snippets, and
the public page/raw body for one already-known legacy filing were inspected only
to understand public formats and endpoint history. They were not acquired
through the controlled v3.1 source pipeline, are inadmissible as experimental
source evidence, and cannot enter a source set, prompt, feature, target,
threshold, ordering rule, or result. This design review established four format
facts:

- `filingFrom` and `filingTo` describe inclusive date ranges, not necessarily
  occupied endpoints;
- the first ten accession digits identify the submitter or filing agent, not
  necessarily the subject company;
- the SEC's displayed acceptance time is an Eastern wall-clock label even
  where Submissions serializes the same digits with a trailing `Z`; and
- a known pre-EDGAR-7.0 Apple 10-Q has no separately linked sequence-1 primary
  document, while its official complete-submission `.txt` remains available.

The general authorities used for these decisions are:

- <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>
  for the Submissions files array and date-range description;
- <https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data>
  for quarterly indexes, archive paths, legacy layout, and index rebuilding;
  and
- <https://www.sec.gov/about/webmaster-frequently-asked-questions> for EDGAR
  acceptance-time and accession-number semantics; and
- <https://www.sec.gov/Archives/edgar/data/320193/000091205700023442/0000912057-00-023442.txt>
  plus its official filing page solely for the known missing-`FILENAME` format.

No Apple Submissions payload, quarterly master, or complete submission has been
opened by the controlled v3.1 source pipeline before this document. No inspected
design-only body is reusable by that pipeline. No Yahoo value, model generation,
performance result, confirmation source, or final source has been opened.

## Frozen scientific question and parent

The scientific question is copied literally from v3:

> Can a fixed local Gemma reader extract point-in-time deterioration evidence
> from Apple 10-K and 10-Q filings that helps a chronological long-or-cash
> learner avoid enough harmful AAPL exposure to add after-cost value over the
> frozen exhaustion baseline and a no-filing-meaning control?

The scientific parent remains v2.2 implementation commit
`efbc481c57e480d48303763163676e64e87df49d`, with v2.2 preregistration commit
`a849b9d704ffd98547e570735a221b2b75f7db86`, literal contract-manifest SHA-256
`64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d`, and
dependency-closure SHA-256
`d37bb45d1d792efa88cb834e1b6a0cabde5ea3e435466a00905002b29152e1bd`.

V3.1 inherits without scientific change the model identity, prompt, schema,
seed, temperature, context and output limits; twelve filing-derived values;
market features and missingness flags; learner and fitting rules; thresholds;
20-session cash interval; non-overlap and next-open rules; inherited exhaustion
baseline; semantic and no-meaning controls; frozen-learning controls;
development, confirmation and final gates; 5-bps and 10-bps ledgers; and every
no-short, no-leverage, no-borrowing, nonnegative-cash and AAPL-or-cash rule.

The inherited corpus still requires the complete non-amended Apple 10-K/10-Q
universe, at least 95% exact acceptance timestamps, the same conservative
filing/change-date fallback for the remainder, and the same stage call caps.
No threshold search, row deletion, model swap, prompt rewrite, feature change,
class rebalance, alternate horizon, or performance-informed source repair is
allowed.

Only these source/operational corrections are authorized:

1. `filingFrom` and `filingTo` become inclusive bounds while `filingCount`
   remains exact.
2. Canonical ten-two-six accession syntax plus independently authenticated
   Apple subject CIK replaces the false Apple-accession-prefix rule. The real
   accession is never rewritten.
3. Acceptance normalization becomes exact SEC wall-clock lexical
   reconciliation rather than UTC conversion of a misleading `Z` suffix.
4. Every stage's Submissions set is independently exact-set reconciled against
   the complete official quarterly-master history through that stage.
5. Every possible stage filing is read from the exact complete-submission
   `.txt` path supplied by the master index, and only its uniquely reconciled
   sequence-1 primary 10-K/10-Q text enters the inherited preprocessor. The
   frozen pre-EDGAR-7 missing-filename identity below is part of this rule. This
   uniform source path covers both legacy and modern archive layouts without a
   link or endpoint fallback.

All affected source and lexical-identity code must be separately hash-bound.
Every unrelated scientific dependency must remain byte-identical to the
parent. Any other source or scientific change requires a new preregistration
before source effects. After the first Yahoo value, Gemma generation, or
performance calculation, a later source-authority failure terminally rejects
this entire scientific family; it may not be repaired in a successor and then
retested against already seen results.

## Zero-effect implementation and preflight gate

The mandatory sequence is:

1. Commit and push this document alone.
2. Implement a new pure v3.1 acquisition/replay adapter and focused tests.
3. Commit and push the exact implementation.
4. Run a zero-source-effect preflight, publish its redacted receipt, then commit
   and push that receipt before the first v3.1 SEC source request.

The implementation-delta manifest must bind every changed file, named symbol,
literal before/after hash, and test. New files may implement only the pure
acquisition, journal, replay, and redacted-evidence rules frozen here. Existing
files have this exhaustive change budget:

- in `sec_point_in_time.py`, only SEC acceptance lexical normalization;
- in `sec_filing_content.py`, only complete-header change-date parsing, the
  frozen legacy missing-`FILENAME` representation/selection, and exact
  complete-response-to-embedded-`TEXT` provenance. `normalize_filing_text`, its
  thresholds, and every text transform remain byte-identical;
- in `sec_filing_gemma_corpus.py`, only inclusive range/count/duplicate rules,
  generic accession identity, master exact-set proof, cumulative v3.1
  complete-submission acquisition/reuse, sequence-1 extraction, and the source
  receipt/provenance schema needed to keep full response bytes distinct from
  extracted primary bytes;
- in `sec_filing_gemma_stage_access.py`, only generic accession identity and
  replacement of direct-primary compact URLs/receipts with the exact root
  complete-submission plan and derived-primary provenance frozen here; all
  stage authorization, ordering, and scientific budgets remain unchanged; and
- in `sec_filing_gemma_contract.py`, only generic accession identity, the
  nullable SEC-filename/internal-legacy identity distinction, cumulative source
  continuity, and source-provenance validation needed by the preceding rules.

In the following downstream files, the only permitted inherited changes are
mechanical replacement of an Apple-prefix lexical guard with generic
`[0-9]{10}-[0-9]{2}-[0-9]{6}` syntax and mechanical propagation of the frozen
complete-response/extracted-primary provenance fields and hashes. Every Apple
`subject_cik == 0000320193` check and every scientific value remains intact:

- `sec_filing_gemma_preprocessor.py`;
- `sec_filing_gemma_features.py`;
- `sec_filing_gemma_prediction_evidence.py`;
- `sec_filing_gemma_stage_authorization.py`;
- `sec_gemma_online_risk_overlay_acquisition.py`;
- `sec_gemma_online_risk_overlay_features.py`; and
- `sec_gemma_online_risk_overlay_market_verifier.py`.

No submitter accession may be converted to an Apple-looking accession at any
layer. The preflight must compare the implementation with the frozen parent,
recompute the approved delta manifest, and reject any unlisted change.

Focused tests must cover inclusive unoccupied bounds, exact counts, duplicates,
third-party submitter accessions, lexical acceptance normalization, zero-only
fractions, DST gaps/ambiguity, optional change dates, legacy and modern master
paths, strict single-member gzip, malformed master rows, complete-submission
identity, the pre-EDGAR-7 all-missing-`FILENAME` case, unique sequence-1
extraction, separate response/extracted hashes, prior-seal carry, every kind of
cross-stage prefix drift, privacy echo rejection, HTTP framing, the `L + 1`
sentinel, rejection of a loopback self-signed/unverifiable TLS certificate,
byte/request/time caps, intent/seal bijection, clean stop/resume, crash-open
rejection, and detached replay. A synthetic full-size rehearsal must exercise
all three stage configurations without network, Yahoo, or model generation.

The preflight requires a clean branch whose local HEAD exactly equals its
upstream, this preregistration as an ancestor, exact implementation and tree
hashes, no conflicting active run, a writable ignored checkpoint root, and the
private authorized SEC contact without printing it. Exactly two loopback-only
local model identity probes (`/api/tags` and `/api/show`) remain allowed; they
generate no text. `/api/chat` and every other generation or market/performance
effect remain forbidden before the complete development source seal.

## Frozen stage configurations

The same committed implementation and rules apply to every stage. Only the
frozen configuration row changes:

| Stage | Availability window | Master coverage | `Q` | `I` cap | `P` cap | Admitted `D` |
|---|---|---:|---:|---:|---:|---:|
| Development | 2000-01-01 through 2018-12-31 | 1994 Q3 through 2018 Q4 | 98 | 128 | 128 | 72-80 |
| Intermediate | 2019-01-01 through 2023-12-31 | 1994 Q3 through 2023 Q4 | 118 | 160 | 24 | 19-20 |
| Final | 2024-01-01 through 2026-07-09 | 1994 Q3 through 2026 Q3, filtered at the cutoff | 129 | 176 | 16 | at most 12 plus every inherited completed-year coverage rule |

`H` is the number of safe unique historical filenames in that stage's main
Submissions response and is capped at 16. `I` is the exact non-amended Apple
10-K/10-Q master/Submissions set inside the listed master coverage and fixed
stage cutoff. `P` is the new source-document request plan: every `I` member that
lacks an exact passing complete-submission seal from an earlier v3.1 stage.
Development has no earlier seal, so `P == I`; a later stage uses
`P == I - prior_sealed_accessions`. No v3, cache, other branch, or unsealed
artifact can satisfy this subtraction. `U` is the inherited cumulative corpus:
all reconciled `I` members whose conservative availability is inside the frozen
2000-01-01 through 2026-07-09 availability window. `D` is the current stage's
slice of `U`. Every `I` member, including a pre-2000 filing that could be delayed
into `U`, is reconciled against exactly one immutable prior v3.1 seal or one new
`P` response before membership is finalized. Thus `P` may exceed `D`, but it
cannot omit a filing that could enter `D`.

The final-stage 2026-Q3 master is response-time evidence. Rows with
`filingDate` after 2026-07-09 are excluded by the fixed cutoff and cannot affect
the candidate set, counts, ordering, or any later rule.

Development source acquisition runs first. Intermediate source acquisition is
forbidden unless development passes its scientific gate. Final source
acquisition is forbidden unless intermediate passes. All later source rules are
already frozen here; a later format failure rejects the family instead of
authorizing a performance-informed repair.

## Canonical source sequence

Each catalogue/master role and each newly required complete submission uses
one-shot, response-time official evidence. A complete submission is fetched
once at its first v3.1 stage and then carried by its exact seal. V3.1
deliberately does not claim that mutable SEC indexes are immutable or that
responses fetched minutes apart form a simultaneous snapshot. Each exact
response body and receipt is sealed once. The fetched versions must reconcile
exactly; any disagreement rejects the stage. No closing re-fetch or merge is
allowed.

The canonical order is:

1. Fetch the main Apple Submissions JSON once. Authenticate CIK 0000320193,
   freeze the exact safe historical reference array, and require `H <= 16`.
2. Fetch every referenced historical Submissions file once in canonical
   filename order. A reference cannot be skipped because of a count, range,
   form, date, or later result. Before any master request, build the
   Submissions-only candidate and source-document upper bounds and reject an
   `I` or `P` cap violation; this early check can reject but cannot authorize a
   member or replace the later master proof.
3. Fetch all `Q` official `master.gz` files in year/quarter order from
   `https://www.sec.gov/Archives/edgar/full-index/<year>/QTR<q>/master.gz`.
4. Exact-set reconcile `I`, enforce the cross-stage prefix invariant below,
   authenticate every prior v3.1 complete-submission seal, and freeze `P` in
   `(filing_date, accession)` order.
5. Fetch the one complete-submission body for every `P` member from the exact
   master filename by prefixing it once with
   `https://www.sec.gov/Archives/`. The required result is exactly
   `https://www.sec.gov/Archives/edgar/data/320193/<accession>.txt` for every
   era. The official post-EDGAR-7 compact-directory alternative is deliberately
   unused. There is no fallback, redirect, directory probe, guessed layout, or
   alternate extension.
6. Reconcile every carried or new complete header and primary document,
   recompute `D`, enforce all inherited corpus gates, and rebuild the entire
   stage from sealed raw bytes. Exact detached replay is required before the
   next data class opens.

At intermediate and final, the complete canonical target-row projection at or
before the preceding stage cutoff must equal the preceding stage seal exactly.
For every accession this comparison includes all typed Submissions values and
missingness, master CIK/form/filed-date/path identity, normalized acceptance and
change date, computed availability session, immutable stage assignment, full
complete-response hash, selected-primary identity, extracted-`TEXT` hash, and
normalized-text hash. Migration of an identical Submissions row between SEC's
main and historical files is allowed and is sealed as layout-only provenance;
any new, missing, duplicated, changed, or reassigned prior-prefix row is a
terminal family rejection. Prior complete submissions are reused byte-for-byte
from their v3.1 seals and are never refetched.

Master indexes and SGML wrapper metadata are verification evidence only. Only
the extracted primary filing text can enter the inherited preprocessor.

## Submissions and quarterly-master proof

Every historical reference name must be canonical, safe, unique, and derive
one fixed `https://data.sec.gov/submissions/<name>` URL. Each reference must
contain exactly `name`, `filingCount`, `filingFrom`, and `filingTo` with their
canonical types.

For the main `recent` table and every historical table:

- every column is a complete row-aligned array;
- mandatory text arrays include `accessionNumber`, `acceptanceDateTime`,
  `form`, `primaryDocument`, `items`, `filingDate`, and `reportDate`;
- optional `dateOfFilingDateChange`, when present, is a complete aligned array
  of canonical ISO dates or empty strings; an explicit absent marker is part of
  each typed row identity;
- raw row count equals unique-accession count, so no duplicate is silently
  deduplicated; and
- no accession may repeat anywhere across the main/historical union, even with
  byte-identical metadata.

For every historical file, `filingCount` equals both counts exactly,
`filingFrom <= filingTo`, and every `filingDate` lies inside those inclusive
bounds. Observed minima/maxima, endpoint-attained flags, bound slack, row/order
hashes, and typed identities are sealed. An unoccupied endpoint is nonfatal;
an out-of-bounds row is fatal.

Each `master.gz` must be one and only one valid gzip member with correct magic,
CRC, ISIZE and EOF, with no trailing or concatenated bytes. Decompression is
streamed under its caps. The complete canonical header must occur once. Every
parsed row has five fields and a canonical filed date inside the requested
quarter. No raw row may repeat. Inside the Apple CIK 320193 exact non-amended
10-K/10-Q projection, every accession occurs exactly once and its filename is
exactly `edgar/data/320193/<accession>.txt`. No global accession-uniqueness
claim is made about unrelated issuers or co-registrants.

Every Submissions candidate inside the master coverage/cutoff must equal one
master candidate on accession, Apple subject CIK, exact form, and filed date,
with zero missing, unexpected, substituted, duplicated, or conflicting rows.
Every non-amended 10-K/10-Q Submissions row is also boundary-scanned. Any row
filed before the 1994-Q3 master start rejects as `master_boundary_incomplete`,
regardless of preliminary availability, because an unseen header-only delay
could otherwise move it into `U`. A row filed after the fixed stage cutoff is
excluded: acceptance must not be after filing and a change date can only delay,
so such a row cannot enter an earlier stage. Any other candidate outside the
declared master coverage/cutoff also rejects rather than being silently omitted.

## Acceptance, change-date, and complete-submission rules

Acceptance normalization is lexical, not UTC arithmetic. Accept only exact
`YYYYMMDDhhmmss` or
`YYYY-MM-DDThh:mm:ss(?:\.0{1,9})?Z` spellings. Validate the displayed fields,
require every fractional digit to be zero, remove punctuation/fraction/`Z`,
and obtain the identical 14 digits. Treat those displayed digits as SEC
Eastern wall-clock digits and never convert a `Z` value from UTC. The localized
`America/New_York` time must not be a DST gap or ambiguity. Availability uses
its first eight digits.

The complete submission must contain one unambiguous SEC header. Every nonempty
Submissions/header value for accession, exact form, filed date, acceptance, and
Apple subject CIK must agree. A header acceptance date after its filing date is
fatal. Every nonempty Submissions `dateOfFilingDateChange` and header `DATE AS
OF CHANGE` must agree when both exist; a sole authenticated value is preserved
and can delay availability but can never move it earlier. Missingness in each
source is sealed explicitly.

An exact acceptance exists when at least one authenticated Submissions/header
value normalizes successfully and all supplied values agree. The inherited
exact-acceptance rate is computed across the current cumulative `U`, not the
verification members of `I` whose availability is outside the corpus window,
and must remain at least 95%. A member with no exact acceptance uses the
inherited conservative filing/change-date fallback and never gains same-day
use. Availability is the first pinned AAPL session strictly after the latest
defensible acceptance, filing, or change date.

The complete submission must contain exactly one `<DOCUMENT>` whose sequence is
1 and whose type exactly equals the non-amended filing form. A second sequence-1
or exact-form candidate is fatal. For a filing dated on or before 2000-05-26,
an SGML `FILENAME` may be absent on any document; the absence remains explicit,
document uniqueness is then proven by `SEQUENCE`, and every supplied filename
must still be safe and unique. From 2000-05-27 onward every document must have a
safe unique SGML filename.

For the selected sequence-1 document, Submissions `primaryDocument` and SGML
`FILENAME` are each preserved as either an exact safe basename or explicit
missingness. If both are supplied, they must be byte-identical. If exactly one
is supplied, that value is the SEC filename and the other source's missingness
is sealed. If both are missing, the filing must be dated on or before
2000-05-26 and receives the internal identity
`legacy-sequence-1-no-filename`; that label is never represented as an SEC
filename. Both missing on or after 2000-05-27, conflicting supplied names, or a
reserved-label collision is fatal.

The design-known accession `0000912057-00-023442` is governed by that general
date-and-missingness rule and receives no accession-specific exception. Any
observed field that fails the rule rejects like any other filing.

A missing/ambiguous `<TEXT>` boundary, mismatched header, or unusable extracted
text rejects the stage. The receipt separately seals the full complete-response
body hash/length; selected document ordinal, sequence, type, filename values and
missingness; exact one-to-one Latin-1 byte offsets; and the exact embedded
`<TEXT>` byte hash/length. A response-body hash may never stand in for an
extracted-primary hash. Only the selected embedded `<TEXT>` bytes, normalized by
the byte-identical inherited rules, can enter a model request. All other
documents remain quarantined.

After header reconciliation, recompute every `I` member from exactly one carried
or new seal, then derive `U` and `D`. The recomputed set must satisfy the frozen
table, `D` must be a subset of `U`, and every `D` member must be backed by
exactly one prior seal or current-`P` seal. No member is added, removed,
repaired, sampled, or reordered by human judgment.

## Request, byte, privacy, journal, and runtime caps

A passing stage has exactly

`1 + H + Q + P`

successful official SEC requests: one main Submissions response, `H`
historical responses, `Q` quarterly masters, and `P` complete submissions. The
maximum is 243 for development, 159 for intermediate, and 162 for final.
Lifetime intents are capped at 256 per stage.

For a pass, lifetime intent count, final HTTP-200 response count, role-seal
count, and the formula must all be equal. Every phase-qualified role has one
exact URL, one intent, one complete HTTP-200 body, and one seal. Any extra or
duplicate intent, non-200 response, redirect, transport exception, hard-timeout,
interruption after intent, or open intent permanently prevents that attempt
from passing and may not be re-dispatched. Resume is allowed only between fully
sealed roles after a clean invocation close.

There are no retries, transport cache reads/writes, proxies, redirects,
environment-derived contacts, or parallel requests. Authenticated prior-stage
v3.1 carry is mandatory source evidence, not a transport cache. Requests send
`Accept-Encoding: identity`; `Content-Encoding` must be absent or exactly one
`identity` token.

Every official request must use HTTPS with certificate-chain verification and
hostname checking enabled through the runtime's normal default trust store.
`verify=False`, an insecure/custom SSL context, custom CA bundle, environment-
derived certificate override, disabled hostname checking, or certificate error
is terminal. The sealed transport capability attests certificate-required,
hostname-checking, default-trust-store, no-custom-CA, no-proxy, no-redirect,
no-retry, and no-cache settings before the first dispatch.

`Transfer-Encoding` must be absent or exactly one `chunked` token. Duplicate,
comma-joined, signed, nondecimal, or conflicting `Content-Length`, multiple
transfer codings, transfer-encoding plus content-length, a malformed/unterminated
chunk stream, trailers, premature EOF, length mismatch, or bytes beyond a
declared length rejects. With neither transfer-encoding nor content-length, only
a clean connection EOF completes the body. The capped quantity is the exact
response-body bytes after valid HTTP framing, not an unverifiable encoded-wire
estimate. Aggregate successful response-body bytes are capped at 18.5625 GiB
(19,931,332,608 bytes), the exact largest stage sum of every per-role body
limit. Conservative lifetime received bytes, including retained reservations,
are capped at 20 GiB (21,474,836,480 bytes).

Allowed body limits are 32 MiB for main Submissions, 64 MiB for a historical
file, 16 MiB compressed/128 MiB decompressed for each master, and 128 MiB for a
complete submission. Aggregate decompressed masters are capped at 16.125 GiB
(17,314,086,912 bytes), the exact final-stage sum of all 129 per-master limits.
For an allowed body limit `L`, an `L + 1` overflow-detection reservation is
fsynced before dispatch and the streaming reader requests no more than the one
remaining sentinel byte. Observing byte `L + 1` rejects and counts it in
lifetime received bytes. A successful seal replaces the reservation with the
exact body count; a failed/open role conservatively retains the greater of its
reservation and observed count. Storage deduplication can share an identical
blob but never a request role, receipt, reservation, or byte count.

The readable SEC contact exists only in the ignored authorized private config
and an in-memory request-header closure. Before any body, receipt, exception, or
journal payload is persisted, scan in memory for the exact UTF-8 contact and
its JSON-escaped, HTML-escaped, and percent-encoded forms. An echo causes a
redacted terminal event without storing the body. Third-party exceptions and
headers are replaced by fixed safe error codes. Public evidence contains only
the contact fingerprint.

The append-only ignored journal fsyncs a unique role intent before dispatch.
After a response passes status, URL, encoding, size, privacy, and parse checks,
its temporary blob is fsynced and atomically renamed, then its role seal is
fsynced. A crash-open intent or orphan blob is never adopted and makes the
attempt non-passable. Every invocation has explicit fsynced open/clean-close
events; an unclosed invocation is terminal. Checkpoints live under
`data/aapl_sec_gemma_lean_evidence_v3_1/<stage>` and public redacted receipts
under `e/aapl_sec_gemma_lean_evidence_v3_1/<stage>`. No v3 byte, journal,
checkpoint, or terminal authority may be reused.

All SEC dispatch starts are globally separated by at least 0.5 monotonic
seconds across purposes. Every new or resumed invocation waits one full second
before its first dispatch. A request has a hard 30-second total monotonic
deadline, not merely a socket-inactivity timeout. A fetch-through-seal role has
a ten-minute limit, an active invocation has a four-hour limit, and cumulative
active source time per stage has a twelve-hour limit. Role and invocation
durations, including failures, are journaled. An unclean/open span is terminal
rather than silently omitted. Clean stopped time is excluded.

Expected source time is 15-90 minutes for development and 10-60 minutes for
each later stage. These are planning estimates, not pass criteria.

## Hard source falsifiers

Before Yahoo, Gemma generation, or performance, the stage rejects on any:

- unsafe, duplicate, skipped, malformed, changed, or unverifiable Submissions
  reference, row, type, array, count, range, or identity;
- missing quarter; multiple-member/trailing/corrupt/oversized gzip; wrong
  header, quarter, row, target CIK/form/date/path; or exact-set difference;
- accession rewrite or submitter-prefix identity rule in place of authenticated
  Apple subject identity;
- malformed/conflicting acceptance, DST ambiguity, change date, header,
  sequence-1 document, primary filename, or extracted text;
- incomplete boundary scan, `H`/`I`/`P`/`D` cap, 95% acceptance, completed-year,
  request-formula, byte, privacy, replay, runtime, clean-branch, or checkpoint
  failure; or
- any forbidden Yahoo, `/api/chat`, performance, later-stage, paid-API, broker,
  real-money, short, leverage, borrowing, or negative-cash effect.

No mismatch is repaired by adding, removing, rewriting, substituting, or
manually choosing a row or document. The exact terminal rejection and redacted
diagnostics are preserved, the central comparison is updated, and later data
stays closed.

## Gemma pilot and later scientific gates

If and only if the development source seal passes, the existing five largest
canonical UTF-8 development requests run as the real Gemma pilot in the exact
inherited order. All five must authenticate the pinned local model and produce
schema-valid sealed outputs. The full-development estimate remains
`sum(first five elapsed) + remaining calls * slowest pilot call`; a projection
above twelve hours pauses before more generations and requires a separately
committed justification that cannot change the science or source rules.

Development, intermediate, and final scoring retain every v3/v2.2 scientific
gate. A failed stage is preserved in `e/APPROACH_COMPARISON.md`, committed and
pushed, with later stages unopened. Historical results never authorize real
capital. Any survivor must first produce append-only prospective paper
decisions, and broker execution requires separate fresh user authorization.
