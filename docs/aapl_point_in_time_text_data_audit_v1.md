# AAPL point-in-time text data audit v1

This document freezes the data-admissibility audit that follows the
price/market-only research stop. It is not a trading experiment and cannot
train, select, or evaluate a model.

## Why this audit exists

The predeclared rare-loss forest was the final permitted price/market-only
approach. If neither of its two full-model candidates passes every frozen gate
and core-ablation check, that research family stops. Its failed result must be
preserved without inversion, threshold tuning, or another price-derived model.

The only permitted pivot is to genuinely new, point-in-time information:
timestamp-safe SEC filings and, only if independently proven viable, real news
text. This audit must establish the source material before Gemma, another LLM,
lesson memory, embeddings, sentiment labels, market outcomes, or trading rules
may be applied.

## Existing local news verdict: unusable

The local file is
`data/warehouse/parquet/news_articles.parquet`. It is 47,068,670,999 bytes and
contains 182,899,080 rows in 1,485 Parquet row groups. Every row has
`source = gdelt_events`.

This is not an article corpus:

- All 182,899,080 titles begin with `GDELT event`; the title is a synthetic
  actor/event summary rather than a publisher headline.
- There is no article-body column.
- `published_at` values are day-at-midnight timestamps without a timezone, not
  observed publication times.
- Symbol association is a raw substring match in Actor1, Actor2, or URL rather
  than an entity-resolved Apple-news label.

The AAPL slice contains 1,369,228 rows:

- Only 29,756 rows, or 2.17%, contain the term in the synthetic actor title.
- 1,339,472 rows, or 97.83%, are URL-only matches.
- At least 658,577 rows, or 48.1%, are demonstrable false positives whose URLs
  contain `pineapple`, `grapple`, `littleapple`, `applevalley`, or `rappler`.
- `rappler` alone accounts for 517,020 rows.
- Of 1,360,933 nonblank URLs, only 362,718 are distinct.
- There are 240,383 repeated date/title/URL triples.
- 40,625 AAPL rows, or 2.97%, fall outside their claimed archive bucket; the
  minimum represented date is in 1920.
- Before 2013 the coverage is only annual or monthly; approximately daily
  coverage begins in April 2013.

Verdict: the local file provides zero usable historical AAPL news text. It
cannot support headline sentiment, document memory, novelty, event timing, or
a FinMem-style experiment. Its row count must not be presented as news
coverage.

## Existing local SEC assets

### Numeric Companyfacts

The retained SEC dataset is
`data/warehouse/parquet/sec_facts.parquet`:

- size: 4,296,766 bytes;
- SHA-256:
  `E6A1397C13ECFBFD229EFA103D026048864E6C7632B1C962AACAF44210FEAD67`;
- 91,405 rows across 50 symbols; and
- ignored under `data/`, not Git-tracked.

Its schema is:

`fact_id, symbol, cik, concept, unit, value, period_start, period_end,
filed_date, fiscal_year, fiscal_period, form, source`.

For AAPL/Apple, CIK `0000320193`, it contains 2,325 unique numeric facts across
13 selected XBRL concepts. Filing dates range from 2009-07-22 through
2026-05-01; represented period ends range from 2006-09-30 through 2026-03-28.

The date-bounded coverage is:

- through 2018: 1,363 fact rows across 38 distinct filing dates;
- 2019-2024: 782 fact rows across 24 distinct filing dates; and
- after 2024: 180 fact rows across 6 distinct filing dates.

The form counts are fact rows, not filing counts: 757 `10-K`, 27 `10-K/A`,
1,442 `10-Q`, and 99 `8-K`. The 99 8-K facts come from only two filing dates.
There are 804 concept/unit/period/form groups with later-filed versions,
representing 1,080 additional versions. Accession identity was discarded, so
those versions cannot be reliably tied to an original filing, amendment, or
restatement.

This table is also present in `data/warehouse/warehouse.duckdb`. The separate
`data/benchmark.db` contains 13,364 AAPL daily decision snapshots with
nonempty derived fundamentals from 2009-07-22 through 2025-12-30. Those are
repeated calculated ratios and latest-fact summaries, not source documents.

### Ingestion behavior and timing gaps

`agent_benchmark/warehouse/sec.py` downloads the SEC ticker map, Companyfacts,
and Submissions. It stores only selected numeric Companyfacts. For AAPL, the
download log records two successful runs on 2026-06-13, each reporting
`2325 selected facts; submissions=1000`. The 1,000 submission records were
counted and then discarded.

No local asset contains:

- a submission catalogue;
- accession numbers;
- `acceptanceDateTime`;
- primary-document names or archive URLs;
- raw Companyfacts or Submissions JSON;
- complete `.txt`/SGML submissions;
- 10-K, 10-Q, 8-K, or DEF 14A HTML/text;
- filing items, exhibits, risk factors, MD&A, or earnings-release text; or
- immutable source-document hashes.

`data/cache/sec` does not exist, and no SEC/EDGAR filing archive was found
elsewhere under the local Documents filesystem.

The warehouse benchmark query correctly filters `filed_date <= decision_date`
at day precision. The older `agent_benchmark/fundamentals.py` path is not
point-in-time safe: `_latest_fact` filters `period_end <= as_of_date` but does
not require `filed <= as_of_date`. It can therefore expose a fact before the
filing that disclosed it. Neither path has an acceptance time, so same-day
availability cannot be proven.

Verdict: the numeric facts may eventually be auxiliary features from July 2009
onward, after a separate causality repair. They are not a filing-text corpus
and do not cover the 2000-2008 training period.

## Official SEC sources and caveat

The audit is restricted to official SEC material for Apple CIK `0000320193`:

- SEC Submissions API documentation:
  <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>
- current Apple submissions metadata:
  <https://data.sec.gov/submissions/CIK0000320193.json>
- historical submission files named by `filings.files` under:
  <https://data.sec.gov/submissions/>
- quarterly full-index template:
  `https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx`
- accession directory JSON template:
  `https://www.sec.gov/Archives/edgar/data/320193/{accession_without_dashes}/index.json`
- complete submission and primary documents under the exact `filename` and
  accession paths returned by SEC catalogues.

`master.idx` establishes catalogue membership, CIK, form, filing date, and
archive filename, but does not establish an exact public timestamp. The raw
complete `.txt` submission's SGML header is the required source for the
14-digit SEC acceptance timestamp. Submissions metadata and SEC archive
indexes are mutable catalogue views retrieved retrospectively; an acceptance
timestamp is evidence of SEC receipt, not a guarantee that every public system
made the document available at precisely that instant.

For that reason, this audit cannot claim perfect historical tradability from
acceptance time alone. Every admitted filing receives a conservative
availability session: the first complete AAPL trading session strictly after
the latest defensible acceptance, filing, or filing-date-change date. The filing may first enter
a decision after that session's close and may affect execution only at the
following open. A missing, malformed, conflicting, or corrected timestamp
never permits same-day use. If only a filing date is defensible, availability
is no earlier than the next AAPL session. Those sessions must match the frozen
full-session NYSE calendar from 2000-01-01 through 2025-01-10 exactly;
arbitrary caller-supplied dates are not admissible.

## Deterministic 24-accession audit sample

The sample is fixed before any document text is inspected. It contains 24
distinct accessions. Its eligible metadata universe is physically bounded from
2000-01-01 through 2024-12-31. A filing is outside that universe if its filing
date, acceptance date, or preserved filing-date-change evidence falls outside
those dates; pre-2000 and post-2024 records cannot fill an edge slot.

The deterministic core contains 18 accessions. For each of the years 2000,
2005, 2009, 2014, 2019, and 2024, select by exact acceptance order:

1. the first accepted `10-K` or `10-Q`;
2. the first accepted `8-K`; and
3. the first accepted `DEF 14A`.

Add six distinct edge-case accessions that are not already in the core:

1. the earliest eligible filing in 2000;
2. Apple's first XBRL filing;
3. the first amendment form ending in `/A`;
4. the first filing accepted after 17:30 US Eastern time;
5. one Apple-related accession whose submitter/filer CIK differs from Apple's
   subject CIK; and
6. one post-acceptance change or timestamp/catalogue anomaly.

Selection uses metadata only. A missing required year/form or edge case is an
explicit failed gate and coverage gap. It cannot be replaced with a nearby
year, different form, hand-selected interesting filing, or another threshold.

## Cross-source reconciliation

For every selected accession, compare and preserve:

- current and historical Submissions metadata;
- the relevant quarterly `master.idx` row;
- the raw complete `.txt` file and its SGML header;
- accession-directory `index.json`;
- form, filing date, report date, and primary-document name;
- Apple subject CIK and submitter/filer CIK;
- accession number with and without dashes;
- exact 14-digit acceptance timestamp when present;
- conservative availability session;
- raw complete-submission and primary-document byte lengths, content types,
  retrieval URLs, and SHA-256 hashes; and
- deterministically normalized primary-document text plus its SHA-256 hash.

Corrections and amendments remain distinct accessions. Nothing is silently
deduplicated by form/date, and no later document overwrites an earlier version.

## Hard resource and behavior limits

The complete audit is capped at:

- 24 distinct selected accessions;
- 100 total HTTP requests, including redirects and retries;
- 250 MB of response bytes before decompression/normalization; and
- 30 wall-clock minutes.

It performs no LLM call, embedding, sentiment analysis, lesson generation,
return calculation, model fitting, candidate selection, or trading simulation.
It uses no paid API. All requests must identify the real user with an SEC-
appropriate `User-Agent` containing a name/organization and reachable contact
email. The current local configuration does not contain an acceptable real
SEC user-agent; the download phase is therefore blocked. The placeholder
`contact@example.com` fallback is not admissible.

The implementation can enforce only exact syntax, identity text, one plausible
non-placeholder email/domain, length, and control-character rules. It cannot
prove that the address is reachable, owned by the operator, or otherwise real;
that remains an operator requirement before live execution.

An injected or mocked transport may validate the offline pipeline but cannot
set the production audit's `overall_pass`. Only the live entrypoint, which
constructs the bounded official-SEC transport internally after validating the
contact's syntax and non-placeholder form, may produce a production pass.

The command is deliberately preflight-only unless live access is explicit:

```powershell
python -m agent_benchmark.sec_audit_cli --preflight
python -m agent_benchmark.sec_audit_cli --execute-live
```

Both commands read only `secrets.sec_user_agent` from the ignored
`data/local_config.json`; unrelated API keys are neither loaded into the audit
nor printed. Preflight performs zero network calls. Live execution is bounded
to official SEC HTTPS hosts and writes only under
`e/sec_point_in_time_audit_v1/`. The production path performs no cache reads or
writes; `data/cache/sec_point_in_time_audit_v1/` is reserved for offline
diagnostics only.

The zero-network preflight disables repository Git hooks and file-system
monitors, ignores global/system Git configuration, disables interactive
credentials and all Git protocols, and rejects partial/promisor repositories,
Git configuration includes, object alternates, Git-directory indirection, UNC
paths, device paths, and Windows remote mapped drives.
It rejects symbolic links, Windows junctions, and other reparse points in every
configured path component. Caller-provided artifact names are restricted to one
safe direct child of the frozen artifact root and are represented in command
output only by a SHA-256 reference, never echoed verbatim. Invalid arguments
and runtime failures emit fixed JSON reason codes without reproducing raw
command-line values or private contact text.

On Windows, sealing pins the exact local artifact directory chain with open
handles and recorded volume/file identities. Payload files are created
exclusively without following reparse points, the temporary directory cannot be
renamed while it is written or verified, and final promotion renames that exact
open directory handle without replacing an existing destination. Verification
uses the same no-follow reads, preventing a checked path from being swapped to a
junction between validation and use.

A production pass requires fresh official retrieval for every admitted
response. Mutable cache entries are ignored by the live path and cannot support
`overall_pass`; the cache is diagnostic only. The artifact also records the
executing Python, Requests, urllib3, certificate bundle, OpenSSL, operating
system, and timezone-data versions.
Preflight fails closed unless Python is 3.11-3.13, Requests is 2.31 or newer
within major version 2, urllib3 is major version 2, the certificate bundle is
installed, and Windows junction detection is available.

The audit must stop before exceeding any cap. A timeout, request/byte-limit
breach, HTTP ambiguity, or missing required sample is a failed audit, not
permission to enlarge the budget, substitute documents, or relax a gate.

## Frozen pass gates

The audit passes only if all of the following are true:

- Every calendar year from 2000 through 2024 has catalogue coverage for at
  least one periodic report (`10-K` or `10-Q`).
- All 24 required accessions exist and agree across Submissions metadata,
  quarterly `master.idx`, raw `.txt`/SGML header, and accession `index.json` on
  every mutually represented identity field.
- All 24 raw submissions and selected primary documents are hashable and
  attributable to Apple as subject or explicitly documented Apple-related
  filer/subject edge cases.
- At least 95% of the intended corpus has an exact valid 14-digit acceptance
  timestamp. With 24 documents this requires at least 23. Every exception uses
  conservative next-session availability and is explicitly flagged.
- Amendments, corrections, filer/subject differences, and post-acceptance
  catalogue changes remain separately identified rather than collapsed.
- At least 95% of selected primary documents yield usable deterministic text.
  With 24 documents this requires at least 23.
- The sealed audit artifact contains no accession accepted after
  2024-12-31. This is an audit-corpus boundary, not permission to train on
  2019-2024. Any later trading experiment remains physically bounded through
  2018 for development, may open 2019-2023 only as its frozen intermediate
  reveal, and keeps 2024 onward untouched as final test data. The 2019 and
  2024 documents in this audit are structural parser/coverage checks only;
  their semantics and market outcomes cannot tune a model or prompt.
- The 100-request, 250-MB, and 30-minute limits are respected.
- The complete audit uses zero LLM/model calls and performs no outcome-guided
  substitution or tuning.

## Required output and next decision

A successful audit must seal the bounded catalogue, exact 24-accession sample,
raw response hashes, normalized text hashes, reconciliation results,
timestamp/availability decisions, request ledger, byte ledger, coverage table,
all gate results, and explicit exclusions. The source files may remain outside
Git, but the immutable audit artifact must checksum every admitted byte and
record the clean implementation commit. That commit must contain the
byte-equivalent audit implementation actually executing. Artifact verification
also requires the externally retained SHA-256 of `checksums.json`; recomputing
both a payload and its local manifest is not verification against the original
seal.

Passing this audit only authorizes a separately predeclared text-feature
experiment. It does not show that filings predict AAPL or that an LLM can make
money. Failing any gate means the local point-in-time filing corpus is not
ready; no text model may run until a new data-audit contract is proposed and
committed.
