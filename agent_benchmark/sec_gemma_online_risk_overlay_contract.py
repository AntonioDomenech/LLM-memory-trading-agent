"""Pure preregistration for the SEC/Gemma online risk-overlay experiment.

This module performs no filesystem, network, SEC, market, model, or clock I/O.
The exact manifest is intentionally strict: changing any field creates a new
approach and requires a new branch before any semantic extraction or scoring.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_extractor_prompt import (
    EXTRACTOR_SYSTEM_PROMPT,
)
from agent_benchmark.sec_filing_gemma_extractor_schema import (
    EXTRACTOR_SCHEMA_VERSION,
    build_extractor_json_schema,
)


CONTRACT_VERSION: Final[str] = "aapl-sec-gemma-online-risk-overlay-v2"
BRANCH_NAME: Final[str] = "codex/aapl-sec-gemma-online-risk-overlay-v2"
BASELINE_POLICY_ID: Final[str] = "fixed-contextual-plus-weak-trend-union-v1"
BASELINE_SOURCE_FILE: Final[str] = (
    "agent_benchmark/chronological_exhaustion_expert.py"
)
BASELINE_SOURCE_SHA256: Final[str] = (
    "a30224763c9858aed905b76215c2c5a66eddd58f107182d751d8eb6a32688c6e"
)
MODEL_NAME: Final[str] = "gemma4:12b"
MODEL_MANIFEST_SHA256: Final[str] = (
    "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
)
MODEL_CONFIG_DIGEST: Final[str] = (
    "c805f5b265d8e695c44f4065dfc368206cd8026447604925fef8db57ee32ee23"
)
MODEL_LAYER_DIGESTS: Final[tuple[str, ...]] = (
    "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
    "675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842",
    "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594",
    "56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3",
)
OLLAMA_VERSION: Final[str] = "0.32.0"
RUNTIME_FINGERPRINT_SHA256: Final[str] = (
    "f92e21f67100de148c03e222ae9232826e69b2f4ca058f2a2c3f560026ccb92b"
)
FINAL_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-final-attempt-001"
)
CONFIRMATION_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-confirmation-attempt-001"
)
DEVELOPMENT_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-development-attempt-001"
)
DEVELOPMENT_ACQUISITION_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-development-acquisition-001"
)
POSITIVE_EDGE_TOLERANCE: Final[float] = 1e-12
HORIZON_SESSIONS: Final[int] = 20
LABEL_MATURITY_OFFSET: Final[int] = 21
PROBABILITY_GATE: Final[float] = 0.55
EXPECTED_EDGE_GATE: Final[float] = 0.0025
MINIMUM_TRAINING_ROWS: Final[int] = 20
MINIMUM_CLASS_ROWS: Final[int] = 4
MAX_TOTAL_RUNTIME_SECONDS: Final[int] = 3_600
MAX_SEC_SECONDS: Final[int] = 720
MAX_MODEL_SECONDS: Final[int] = 2_160
MAX_DETERMINISTIC_SECONDS: Final[int] = 480
MAX_SEC_REQUESTS: Final[int] = 1_000
MAX_SEC_BYTES: Final[int] = 1_610_612_736
MAX_SEC_REQUESTS_PER_SECOND: Final[int] = 2
MAX_MARKET_REQUESTS_PER_STAGE: Final[int] = 6
MAX_MARKET_SECONDS: Final[int] = 210

SOURCE_PINS: Final[dict[str, str]] = {
    "sec_filing_gemma_contract": (
        "d5a268da138862b510b0f12b139a04fa91a91d7deba4fe2b28666609c03318e4"
    ),
    "sec_filing_gemma_preprocessor": (
        "febdbaa2fa5f6ecd528fdc9642614b0f8fd8df79f0f9c7ea5c398d89d1be2544"
    ),
    "sec_filing_gemma_extractor_prompt": (
        "312d2aad202ece6b9e80f4508127f702fc965e92aa6bcb30217e4a62ec8e8d7d"
    ),
    "sec_filing_gemma_extractor_schema": (
        "ee9b5804a3816799cd26549b56ca852586d5effe57897c5d74cf06ce8db52c6e"
    ),
    "sec_filing_gemma_ollama": (
        "854fa9184658549ef72e6d618eab018d07161adddf0b8268f0f167dc655557d4"
    ),
    "sec_filing_gemma_corpus": (
        "74831feadcae050eee497da0a3405d65a5c4649a59830bf3f768ab4d35f9164b"
    ),
    "sec_filing_content": (
        "70c969c8eee82e82c0ea1a8a5e424178122fdba8ac1b12ed5a12e207177e10bf"
    ),
    "sec_session_calendar": (
        "9a463fa0453440dc777a850e7af934dfc0f2740243e7e8e05ff6ebc430262f71"
    ),
    "sec_filing_gemma_market_source_bytes": (
        "4cf885352769b4b48fd0a7c90c8cd50dbb4330ae6a1f144ff17aed6eb9b9030d"
    ),
    "sec_filing_gemma_market_acquirer": (
        "79ac6275fa5a10b2805a52af7748e81ca3a6a8582413a2077b43a448ab410bc0"
    ),
    "sec_filing_gemma_learner": (
        "b948820036454f76fb606068b54060916535f3721e29564340565c890113fc92"
    ),
    "chronological_exhaustion_expert": BASELINE_SOURCE_SHA256,
}
SOURCE_PIN_FILES: Final[dict[str, str]] = {
    "sec_filing_gemma_contract": "agent_benchmark/sec_filing_gemma_contract.py",
    "sec_filing_gemma_preprocessor": (
        "agent_benchmark/sec_filing_gemma_preprocessor.py"
    ),
    "sec_filing_gemma_extractor_prompt": (
        "agent_benchmark/sec_filing_gemma_extractor_prompt.py"
    ),
    "sec_filing_gemma_extractor_schema": (
        "agent_benchmark/sec_filing_gemma_extractor_schema.py"
    ),
    "sec_filing_gemma_ollama": "agent_benchmark/sec_filing_gemma_ollama.py",
    "sec_filing_gemma_corpus": "agent_benchmark/sec_filing_gemma_corpus.py",
    "sec_filing_content": "agent_benchmark/sec_filing_content.py",
    "sec_session_calendar": "agent_benchmark/sec_session_calendar.py",
    "sec_filing_gemma_market_source_bytes": (
        "agent_benchmark/sec_filing_gemma_market_source_bytes.py"
    ),
    "sec_filing_gemma_market_acquirer": (
        "agent_benchmark/sec_filing_gemma_market_acquirer.py"
    ),
    "sec_filing_gemma_learner": "agent_benchmark/sec_filing_gemma_learner.py",
    "chronological_exhaustion_expert": BASELINE_SOURCE_FILE,
}

DEVELOPMENT_BLOCKS: Final[tuple[tuple[str, str, str], ...]] = (
    ("block_1", "2005-01-03", "2007-12-31"),
    ("block_2", "2008-01-02", "2010-12-31"),
    ("block_3", "2011-01-03", "2013-12-31"),
    ("block_4", "2014-01-02", "2016-12-30"),
    ("block_5", "2017-01-03", "2018-12-31"),
)

MARKET_FEATURES: Final[tuple[str, ...]] = (
    "aapl_minus_qqq_log_return_20",
    "aapl_drawdown_63",
    "aapl_realized_volatility_20",
    "spy_log_return_20",
    "iwm_log_return_20",
    "vix_log_change_20",
)
MEANING_FEATURES: Final[tuple[str, ...]] = (
    "commercial_deterioration",
    "financial_deterioration",
    "risk_outlook_deterioration",
    "adverse_flag_fraction",
)
QUALITY_FEATURE: Final[str] = "semantic_quality_risk"
SEMANTIC_FEATURES: Final[tuple[str, ...]] = (
    *MEANING_FEATURES,
    QUALITY_FEATURE,
)
CONTROL_FEATURES: Final[tuple[str, ...]] = (
    "form_10k",
)
FEATURES: Final[tuple[str, ...]] = (
    *MARKET_FEATURES,
    *SEMANTIC_FEATURES,
    *CONTROL_FEATURES,
)


class SecGemmaOnlineRiskOverlayContractError(ValueError):
    """Raised when a purported contract differs from the preregistration."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 representation used for identity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _prompt_sha256() -> str:
    return hashlib.sha256(EXTRACTOR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _schema_sha256() -> str:
    return canonical_sha256(build_extractor_json_schema())


def build_runtime_fingerprint_material() -> dict[str, Any]:
    """Return the exact local-model metadata whose literal hash is pinned."""

    return {
        "schema_version": "sec-gemma-v2-local-runtime-pin-v1",
        "model_name": MODEL_NAME,
        "ollama_version": OLLAMA_VERSION,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "model_config_digest": MODEL_CONFIG_DIGEST,
        "model_layer_digests": list(MODEL_LAYER_DIGESTS),
        "version_response_sha256": (
            "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
        ),
        "show_response_sha256": (
            "8ab2bd35bfd63bc37b9dd7e932ee38f3ad4dfa773d0767baf4a7d08eea7428e0"
        ),
        "model_info_sha256": (
            "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
        ),
    }


def build_contract_manifest() -> dict[str, Any]:
    """Build a detached copy of the exact frozen experiment contract."""

    manifest: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "branch": BRANCH_NAME,
        "status_at_preregistration": "unrun",
        "objective": {
            "asset": "AAPL",
            "benchmark": "same-ledger AAPL buy-and-hold",
            "allowed_target_exposures": [0, 1],
            "maximum_target_exposure": 1,
            "maximum_realized_exposure": 1,
            "shorting": False,
            "leverage": False,
            "borrowing": False,
            "negative_cash": False,
            "cash_interest": False,
            "paid_api_calls": 0,
        },
        "evidence_classification": {
            "globally_pristine": False,
            "reason": (
                "The repository has already inspected 2024 onward and the "
                "foundation model may contain pretraining knowledge through 2024; "
                "the inherited baseline also includes a retrospectively selected "
                "expert."
            ),
            "honest_claim": "candidate-specific chronological retrospective replay",
            "prospective_proof_required": True,
        },
        "data": {
            "sec": {
                "issuer_cik": "0000320193",
                "forms": ["10-K", "10-Q"],
                "amendments": False,
                "source": "official SEC submissions and archive primary documents",
                "corpus_stages": {
                    "development": ["2000-01-01", "2018-12-31"],
                    "intermediate_confirmation": ["2019-01-01", "2023-12-31"],
                    "final_live_style": ["2024-01-01", "2026-07-09"],
                },
                "availability_rule": (
                    "first complete NYSE session strictly after the latest "
                    "defensible acceptance, filing, or filing-change date"
                ),
                "minimum_development_corpus_filings_2000_2018": 72,
                "minimum_confirmation_filings": 19,
                "complete_metadata_eligible_universe_required": True,
            },
            "market": {
                "evidence_symbols": ["AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX"],
                "feature_symbols": ["AAPL", "QQQ", "SPY", "IWM", "VIX"],
                "unused_but_bound_evidence_symbols": ["TNX"],
                "provider_symbol_mapping": {
                    "AAPL": "AAPL",
                    "SPY": "SPY",
                    "QQQ": "QQQ",
                    "IWM": "IWM",
                    "VIX": "^VIX",
                    "TNX": "^TNX",
                },
                "source_acquisition": {
                    "schema_version": (
                        "aapl-sec-gemma-online-overlay-yahoo-chart-v8-v1"
                    ),
                    "provider_family": (
                        "yahoo-finance-chart-v8-public-unauthenticated"
                    ),
                    "endpoint": (
                        "https://query1.finance.yahoo.com/v8/finance/chart"
                    ),
                    "transport": (
                        "owned HTTPS GET with normal certificate and hostname "
                        "verification; no proxy, cookie, authentication, redirect, "
                        "compression, alternate host, provider, or fallback"
                    ),
                    "fixed_user_agent": (
                        "LLM-memory-trading-agent/1.0 market-evidence "
                        "(no-auth; one-shot)"
                    ),
                    "request_order": ["AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX"],
                    "request_count_each_stage": MAX_MARKET_REQUESTS_PER_STAGE,
                    "request_path": (
                        "endpoint + '/' + percent-encoded provider symbol"
                    ),
                    "query_items_in_order": [
                        ["period1", "stage_period1_utc"],
                        ["period2", "stage_period2_utc"],
                        ["interval", "1d"],
                        ["includePrePost", "false"],
                        ["includeAdjustedClose", "true"],
                        ["events", "div,splits"],
                    ],
                    "request_windows": {
                        "development": {
                            "start": "1998-01-01",
                            "end_exclusive": "2019-01-01",
                            "period1_utc": 883612800,
                            "period2_utc": 1546300800,
                            "last_eligible_session": "2018-12-31",
                        },
                        "confirmation": {
                            "start": "1998-01-01",
                            "end_exclusive": "2024-01-01",
                            "period1_utc": 883612800,
                            "period2_utc": 1704067200,
                            "last_eligible_session": "2023-12-29",
                        },
                        "final": {
                            "start": "1998-01-01",
                            "end_exclusive": "2026-07-11",
                            "period1_utc": 883612800,
                            "period2_utc": 1783728000,
                            "last_transport_session_quarantined": "2026-07-10",
                            "last_exposed_market_value_session": "2026-07-09",
                            "last_scored_fill_origin_decision_session": "2026-07-08",
                            "last_pending_prediction_session": "2026-07-09",
                        },
                    },
                    "provider_timezones": {
                        "AAPL": "America/New_York",
                        "SPY": "America/New_York",
                        "QQQ": "America/New_York",
                        "IWM": "America/New_York",
                        "VIX": "America/Chicago",
                        "TNX": "America/Chicago",
                    },
                    "timestamp_to_session": (
                        "convert each integer Unix timestamp to the frozen expected "
                        "provider timezone, require that its local date equals its UTC "
                        "date, then use that ISO date"
                    ),
                    "canonical_prefix_rule": (
                        "confirmation must reproduce every development canonical "
                        "session, explicit-absence marker, and float bit pattern; final "
                        "must reproduce the complete confirmation prefix identically. "
                        "Any back-adjustment, revision, omission, or addition inside a "
                        "sealed prefix terminally fails the branch"
                    ),
                    "private_raw_metadata_rule": (
                        "exact provider bytes enter a private quarantine because Yahoo "
                        "metadata can contain current quote fields; no raw metadata or "
                        "row beyond the stage value boundary is returned to the "
                        "experiment. The final 2026-07-10 transport row remains private"
                    ),
                    "visibility_and_lock_rule": (
                        "development raw bytes may be acquired into quarantine before "
                        "the development lock but no price value may be exposed; "
                        "confirmation and final acquisition occur only after their "
                        "durable stage locks"
                    ),
                    "retry_and_selection_rule": (
                        "no HTTP retry and no alternate snapshot; the first complete "
                        "authenticated stage batch is sealed, while failed transport "
                        "receipts expose no values and cannot be used to choose a batch"
                    ),
                },
                "required_prefix_sessions": 253,
                "canonical_provider_fields": [
                    "raw_open",
                    "raw_high",
                    "raw_low",
                    "raw_close",
                    "raw_volume",
                    "adjusted_close",
                ],
                "ledger_price_fields": [
                    "raw_open",
                    "raw_close",
                    "adjusted_close",
                ],
                "ledger_price_validation": (
                    "on every exposed AAPL ledger session, raw_open, raw_close, and "
                    "adjusted_close must each exist exactly once and be finite and "
                    "strictly positive; adjusted_open = raw_open*adjusted_close/raw_close "
                    "must also be finite and strictly positive. Any failure terminally "
                    "fails the stage rather than making only one filing unavailable"
                ),
                "price_field": "adjusted_close",
                "feature_cutoff": "completed decision-session close",
                "fill": "next adjusted open",
                "horizon": {
                    "decision_close_index": "t",
                    "entry_open_index": "t+1",
                    "exit_open_index": "t+21",
                    "held_open_to_open_intervals": 20,
                    "label_maturity_session_index": "t+21",
                },
                "calendar": {
                    "calendar_id": "nyse_trading_session_dates_2000_01_01_2026_07_10_v2",
                    "calendar_dates_sha256": (
                        "e0550f12f98d7e0cf38d6797ae0d0e410bb3f9006793830ed75e0f753ccc5cf9"
                    ),
                    "market_calendar_id": (
                        "nyse_trading_session_dates_1998_01_01_2026_07_10_v1"
                    ),
                    "market_calendar_dates_sha256": (
                        "e9d37d63d158f8a3b6de58ef81970b27bcac9edb6a9c7b9e2f3ffe477d2f3032"
                    ),
                    "official_source_evidence_required_before_effects": True,
                },
            },
            "forbidden": [
                "GDELT event rows described as news",
                "future prices or outcomes in features",
                "post-decision filing revisions",
                "paid data or model APIs",
            ],
        },
        "gemma": {
            "role": "fixed evidence-grounded filing reader, never trader",
            "model_name": MODEL_NAME,
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "model_config_digest": MODEL_CONFIG_DIGEST,
            "model_layer_digests": list(MODEL_LAYER_DIGESTS),
            "ollama_version": OLLAMA_VERSION,
            "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
            "runtime_fingerprint_material": build_runtime_fingerprint_material(),
            "runtime_version_response_sha256": (
                "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
            ),
            "runtime_show_response_sha256": (
                "8ab2bd35bfd63bc37b9dd7e932ee38f3ad4dfa773d0767baf4a7d08eea7428e0"
            ),
            "runtime_model_info_sha256": (
                "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
            ),
            "runtime_note": (
                "Gemma 4 exposes two active FROM blobs; v2 must verify the exact "
                "manifest and all four layer digests rather than v1's one-FROM parser"
            ),
            "pre_call_identity_gate": (
                "before every semantic batch, hash the installed manifest bytes, "
                "verify the config and ordered layer digests, hash every layer's "
                "content, query exact version/show bytes, rebuild the fingerprint, "
                "and require byte-for-byte equality with these pins"
            ),
            "endpoint": "http://127.0.0.1:11434/api/chat",
            "loopback_only": True,
            "model_pull": False,
            "temperature": 0,
            "seed": 0,
            "context_tokens": 6144,
            "output_tokens": 512,
            "retries": 0,
            "repairs": 0,
            "prompt_sha256": _prompt_sha256(),
            "schema_version": EXTRACTOR_SCHEMA_VERSION,
            "schema_sha256": _schema_sha256(),
            "input": (
                "issuer/date/market-blinded current and prior same-form filing "
                "sentences only"
            ),
            "forbidden_inputs": [
                "ticker or issuer identity",
                "exact dates",
                "market prices or returns",
                "labels, actions, scores, or benchmark results",
            ],
            "invalid_output_policy": "neutral semantics plus quality risk; no retry",
            "source_pins": {
                role: {"file": SOURCE_PIN_FILES[role], "sha256": digest}
                for role, digest in sorted(SOURCE_PINS.items())
            },
            "new_v2_sources": (
                "feature, runtime, online-replay, ledger, store, and verifier source "
                "hashes must be bound to the preregistration commit in the effectful "
                "attempt manifest before SEC, model, confirmation, or final access"
            ),
        },
        "features": {
            "ordered_names": list(FEATURES),
            "count": len(FEATURES),
            "market_names": list(MARKET_FEATURES),
            "semantic_names": list(SEMANTIC_FEATURES),
            "meaning_names": list(MEANING_FEATURES),
            "quality_name": QUALITY_FEATURE,
            "control_names": list(CONTROL_FEATURES),
            "market_formulas": {
                "aapl_minus_qqq_log_return_20": (
                    "log(AAPL_adj_close[t]/AAPL_adj_close[t-20]) - "
                    "log(QQQ_adj_close[t]/QQQ_adj_close[t-20])"
                ),
                "aapl_drawdown_63": (
                    "AAPL_adj_close[t]/max(AAPL_adj_close[t-62:t inclusive])-1"
                ),
                "aapl_realized_volatility_20": (
                    "sample_std_ddof_1(log(AAPL_adj_close[i]/AAPL_adj_close[i-1]) "
                    "for i=t-19..t)*sqrt(252)"
                ),
                "spy_log_return_20": (
                    "log(SPY_adj_close[t]/SPY_adj_close[t-20])"
                ),
                "iwm_log_return_20": (
                    "log(IWM_adj_close[t]/IWM_adj_close[t-20])"
                ),
                "vix_log_change_20": (
                    "log(VIX_adj_close[t]/VIX_adj_close[t-20])"
                ),
            },
            "market_missingness": (
                "any absent, duplicate, nonfinite, or nonpositive required adjusted "
                "close makes the event unavailable; no imputation or row drop"
            ),
            "semantic_groups": {
                "commercial": ["demand", "pricing_power", "supply_chain"],
                "financial": [
                    "gross_margin",
                    "operating_cost_pressure",
                    "capital_allocation",
                    "liquidity",
                ],
                "risk_outlook": [
                    "forward_guidance",
                    "legal_regulatory",
                    "management_uncertainty",
                ],
            },
            "semantic_encoding": {
                "current_impact": {
                    "favorable": -1,
                    "neutral": 0,
                    "unfavorable": 1,
                    "mixed": 0,
                    "not_stated": 0,
                },
                "change_vs_prior": {
                    "improving": -1,
                    "stable": 0,
                    "deteriorating": 1,
                    "mixed": 0,
                    "not_comparable": 0,
                    "not_stated": 0,
                },
                "dimension_score": "mean(current_impact, change_vs_prior)",
                "group_score": "fixed-denominator mean of member dimension scores",
                "adverse_flags": [
                    "new_material_risk",
                    "guidance_withdrawn",
                    "liquidity_stress",
                    "restructuring_or_impairment",
                    "internal_control_weakness",
                ],
                "semantic_quality_risk": {
                    "usable": 0,
                    "thin": 0.5,
                    "unusable_or_invalid": 1,
                },
                "adverse_flag_fraction": "sum(present for five adverse flags)/5",
            },
            "no_filing_meaning_ablation": (
                "a separate causal arm with identical market, form, extraction-quality, "
                "event, and label rows; set only the four filing-meaning features to "
                "zero and preserve semantic_quality_risk exactly"
            ),
            "no_gemma_channel_diagnostic": (
                "a separately reported diagnostic arm that zeros all five Gemma-derived "
                "features; it cannot satisfy or rescue any filing-meaning success gate"
            ),
            "schema_valid_extraction_rate": (
                "authenticated exact-schema-valid Gemma outputs divided by every "
                "metadata-eligible event requiring a call; missing, unauthenticated, "
                "and schema-invalid outputs remain in the denominator"
            ),
            "extraction_coverage_stage_assignment": (
                "coverage and nonzero-meaning counts use only events whose filing "
                "decision session lies in that development, confirmation, or final "
                "stage; earlier cumulative rows cannot satisfy a later-stage gate"
            ),
            "nonzero_meaning_row": (
                "an authenticated schema-valid row with absolute value above 1e-12 in "
                "at least one of the four filing-meaning features"
            ),
        },
        "event_availability": {
            "eligible_universe_rule": (
                "every metadata-eligible filing receives exactly one chronological "
                "audit row; availability status may not remove or reorder the row"
            ),
            "missing_primary_document_before_attempt": (
                "failure of the acquisition prerequisite: no scored attempt begins "
                "until every eligible primary document and provenance record for the "
                "stage is authenticated and sealed"
            ),
            "first_same_form_filing": (
                "an authenticated current filing with no earlier same-form filing is "
                "available; prior-change fields are not_comparable and encode as zero, "
                "while current-impact fields and current extraction quality remain live"
            ),
            "authenticated_schema_invalid_output": (
                "available and trainable when market inputs exist: set the four meaning "
                "features to zero, semantic_quality_risk to 1, preserve form_10k, seal "
                "the invalid-response receipt, and do not retry or repair. The primary "
                "meaning arm and quality-preserving no-meaning ablation are therefore "
                "identical on this row"
            ),
            "missing_or_unauthenticated_model_output": (
                "event unavailable: seal an audit row, take no SEC-overlay action, make "
                "no prediction, and exclude it from learner membership, Brier support, "
                "episodes, and action-difference counts; no retry"
            ),
            "missing_or_unauthenticated_feature_provenance": (
                "event unavailable under the same exclusions; a whole-batch model or "
                "runtime identity failure terminally fails the consumed stage"
            ),
            "market_unavailable": (
                "event unavailable under the same exclusions when any required market "
                "input is absent, duplicate, nonfinite, or nonpositive; no imputation"
            ),
            "unavailable_label_rule": (
                "a counterfactual label may still mature and is retained as audit-only, "
                "but it can never train either semantic or ablation learner without the "
                "complete immutable decision-time feature row"
            ),
            "learner_unready": (
                "an otherwise available event is retained for later training but emits "
                "no fitted prediction and schedules no overlay; it is outside Brier and "
                "action-difference support until both heads satisfy readiness"
            ),
            "active_overlay_event": (
                "compute and seal its available prediction normally, then force the "
                "effective schedule flag false under non-overlap; its label still "
                "matures and can train future decisions"
            ),
            "undefined_or_nonfinite_metric": "terminal stage failure",
        },
        "learner": {
            "training_mode": "continuous expanding causal refit before each filing",
            "training_rows": (
                "all and only earlier filing labels with maturity_session <= "
                "current_decision_session; the t+21 exit open on the current "
                "session is known before that session's close"
            ),
            "same_session_matured_label_admission": "maturity_session <= decision_session",
            "counterfactual_lessons": True,
            "minimum_training_rows": MINIMUM_TRAINING_ROWS,
            "minimum_rows_per_binary_class": MINIMUM_CLASS_ROWS,
            "unready_action": "no SEC overlay; inherited baseline still trades",
            "label": (
                "counterfactual incremental 10bps log edge of activating the "
                "20-session overlay versus the fixed baseline-only ledger"
            ),
            "probability_head": (
                "robust-scaled ridge logistic for overlay beating fixed baseline"
            ),
            "edge_head": (
                "robust-scaled ridge Huber for incremental edge versus baseline"
            ),
            "learner_config": {
                "raw_mad_multiplier": 1.4826,
                "raw_scale_floor": 1e-6,
                "raw_z_clip": 4,
                "ridge_lambda": 0.1,
                "intercept_regularized": False,
                "logistic_initial_coefficients": (
                    "intercept=logit(training prevalence), all slopes=0"
                ),
                "logistic_max_iterations": 50,
                "logistic_tolerance": 1e-10,
                "newton_line_search": (
                    "Armijo constant 1e-4, halving from step 1"
                ),
                "newton_line_search_max_steps": 50,
                "huber_delta": 1.5,
                "huber_initial_coefficients": (
                    "intercept=median(standardized clipped edge), all slopes=0"
                ),
                "huber_max_iterations": 50,
                "huber_tolerance": 1e-10,
                "target_mad_multiplier": 1.4826,
                "target_scale_floor": 1e-6,
                "edge_clip": [-0.5, 0.5],
                "persisted_floats": "canonical float.hex",
            },
            "action_gate": {
                "probability_at_least": PROBABILITY_GATE,
                "expected_incremental_10bps_log_edge_at_least": EXPECTED_EDGE_GATE,
                "candidate_grid": False,
            },
        },
        "policy": {
            "baseline_policy_id": BASELINE_POLICY_ID,
            "baseline_source_file": BASELINE_SOURCE_FILE,
            "baseline_source_sha256": BASELINE_SOURCE_SHA256,
            "baseline_kind": "fixed raw expert union, not binary-regime selector",
            "baseline_signal_column": "unfiltered_union_signal",
            "baseline_forbidden_column": "unfiltered_union_target_exposure",
            "baseline_parameters": {
                "intraday_return": "AAPL_close/AAPL_open-1",
                "intraday_prior_percentile_lookback": 126,
                "contextual_percentile": 0.90,
                "contextual_spy_qqq_return_lookback": 10,
                "weak_trend_percentile": 0.925,
                "weak_trend_spy_qqq_return_lookback": 20,
                "weak_trend_aapl_sma_lookback": 20,
                "contextual_condition": (
                    "intraday above shifted prior percentile AND SPY and QQQ "
                    "lookback returns both below zero"
                ),
                "weak_trend_condition": (
                    "intraday above shifted prior percentile AND SPY and QQQ "
                    "lookback returns both below zero AND AAPL adjusted close "
                    "below its 20-session simple moving average"
                ),
                "composition": (
                    "canonicalize each expert's one-session signals; OR them; "
                    "canonicalize the union once over the continuous prefix"
                ),
            },
            "baseline_prefix_invariance": (
                "a completed-close raw union signal is persisted even when t+1 or "
                "t+2 lies beyond the current physical price prefix; appending rows "
                "may fill its pending action but may not rewrite the decision"
            ),
            "baseline_action": (
                "signal at close t schedules CASH at open t+1 and LONG at open t+2"
            ),
            "baseline_cooldown_independence": (
                "baseline canonicalization and cooldown evolve from baseline signals "
                "only; an SEC overlay never suppresses, resets, or creates a baseline "
                "signal"
            ),
            "overlay_horizon_sessions": HORIZON_SESSIONS,
            "overlay_entry": "next adjusted open after the filing decision close",
            "overlay_exit": "t+21 adjusted open after 20 open-to-open intervals",
            "active_overlay_extension": False,
            "overlapping_overlay": False,
            "ignored_signals_still_mature_as_lessons": True,
            "combined_cash": "baseline_cash OR active_sec_overlay",
            "overlap_execution": (
                "target exposure is recomputed at every open; a fill and its cost "
                "occur only when combined target exposure changes. A baseline signal "
                "inside an active overlay has no fill but still advances the independent "
                "baseline state. At overlay exit, remain CASH without a fill when the "
                "baseline is CASH at that open; otherwise buy AAPL once."
            ),
            "lesson_counterfactual": (
                "for every eligible filing, fork the exact baseline-only cash/share/"
                "cooldown state immediately before open t+1 into baseline-only and "
                "forced-overlay arms. Ignore all other SEC overlays in both arms, "
                "continue identical future baseline signals, apply actual changing-leg "
                "costs, and compare normalized wealth at open t+21."
            ),
            "costs_bps_per_changing_leg": [5, 10],
            "same_action_stream_at_both_costs": True,
        },
        "ledger": {
            "initial_capital_usd": 1000,
            "initial_exposure_before_genesis_open": 0,
            "genesis_target_for_strategy_and_benchmark": 1,
            "adjusted_open_formula": "raw_open * adjusted_close / raw_close",
            "held_long_return_factor": "adjusted_open[current]/adjusted_open[prior]",
            "held_cash_return_factor": 1,
            "buy_cost_factor": "1/(1+cost_bps/10000)",
            "sell_cost_factor": "1-cost_bps/10000",
            "buy_fill_arithmetic": (
                "shares = pre_fill_cash / (adjusted_open * "
                "(1+cost_bps/10000)); cash = 0"
            ),
            "sell_fill_arithmetic": (
                "cash = pre_fill_shares * adjusted_open * "
                "(1-cost_bps/10000); shares = 0"
            ),
            "fill_order": (
                "earn the prior target's open-to-open return first, then execute the "
                "current open's target-changing fill and cost"
            ),
            "same_prices_and_initial_purchase_as_benchmark": True,
            "stage_boundaries": (
                "carry exact cash, shares, exposure, prior adjusted open, pending fills, "
                "and active overlay; no synthetic trade or capitalization reset"
            ),
            "terminal_valuation": ["adjusted_open", "terminal_adjusted_close"],
            "adjusted_open_valuation": "cash + shares * adjusted_open",
            "terminal_adjusted_close_valuation": (
                "cash + shares * terminal_adjusted_close; no synthetic liquidation "
                "or terminal transaction cost"
            ),
            "realized_exposure_formula": (
                "0 when shares=0; otherwise shares*valuation_price/wealth = 1 within "
                "1e-12, with finite positive wealth and cash >= 0"
            ),
            "cash_interest": False,
            "fractional_shares": True,
            "debt": False,
        },
        "chronology": {
            "portfolio_genesis": "2000-01-03",
            "warmup_and_learning": ["2000-01-03", "2004-12-31"],
            "development_corpus": ["2000-01-01", "2018-12-31"],
            "development_qualification": ["2005-01-03", "2018-12-31"],
            "development_blocks": [
                {"id": item[0], "first": item[1], "last": item[2]}
                for item in DEVELOPMENT_BLOCKS
            ],
            "confirmation": ["2019-01-01", "2023-12-31"],
            "final_live_style_replay": ["2024-01-01", "2026-07-09"],
            "final_cutoff_rule": (
                "2026 YTD performance and both terminal valuations end on 2026-07-09. "
                "Only decisions through 2026-07-08 can create scored fills. A filing "
                "decision after the 2026-07-09 close is sealed with its pending t+1 "
                "action and lesson but contributes no 2026-07-10 price, fill, return, "
                "episode, or action difference to this audit"
            ),
            "updates_inside_every_period": (
                "permitted only after the complete 20-session outcome matures"
            ),
            "every_2025_decision_state": (
                "immediately before each filing, admit every earlier eligible label "
                "with maturity_session <= that filing's decision_session; no annual "
                "cutoff freezes learning"
            ),
            "same_session_event_order": (
                "on one decision session, first admit all newly matured labels once, "
                "then process filings by exact SEC acceptance timestamp ascending and "
                "accession ascending; no outcome can enter between same-session filings"
            ),
            "same_close_pending_overlay_rule": (
                "the first ordered filing that passes may reserve the next-open overlay; "
                "that scheduled overlay blocks every later same-close filing from "
                "scheduling or extending another overlay, although each still receives "
                "an audit prediction and later lesson"
            ),
            "controls": {
                "semantic_arm_continuity": (
                    "the full semantic, quality-preserving no-meaning, and all-five-zero "
                    "diagnostic arms each start at portfolio genesis with an independent "
                    "causal learner, cash/share account, baseline cooldown, pending "
                    "fills, and overlay non-overlap state; all are carried without reset "
                    "through every block and stage"
                ),
                "semantic_arm_labels": (
                    "all semantic arms receive the same matured counterfactual labels; "
                    "only their frozen decision-time feature transforms differ"
                ),
                "frozen_state_contents": (
                    "exact fitted coefficients, scalers, training membership and "
                    "counts; pending pre-fork labels remain audit-only and never refit "
                    "the frozen control after the fork"
                ),
                "fork_boundary_order": (
                    "fork immediately after all labels with maturity_session on or "
                    "before the preceding session have been admitted, and before any "
                    "label maturing on the boundary session or any boundary-session "
                    "filing decision is processed"
                ),
                "development": (
                    "at the start of each block's first session under fork_boundary_order, "
                    "fork the primary account, active/scheduled overlay, baseline state, "
                    "and learner state. The "
                    "control sees later causal features but never admits another label; "
                    "it is discarded after that block and never alters the primary."
                ),
                "confirmation": (
                    "fork the exact continuous primary account and through-2018 learner "
                    "state before any 2019-session admission or decision; preserve any "
                    "scheduled/active overlay; "
                    "never update the control during confirmation"
                ),
                "final": (
                    "fork the exact continuous primary account and through-2023 learner "
                    "state before any 2024-session admission or decision; preserve any "
                    "scheduled/active overlay; "
                    "never update the control during the final replay"
                ),
            },
            "primary_account_continuity": (
                "one account from genesis; no cash/share/position/model reset and no "
                "synthetic boundary fill"
            ),
            "no_reset_at_stage_or_year_boundary": True,
            "design_changes_after_development": False,
        },
        "metric_definitions": {
            "positive_tolerance": POSITIVE_EDGE_TOLERANCE,
            "positive_edge": "active log edge > 1e-12",
            "negative_edge": "active log edge < -1e-12",
            "tie": "absolute active log edge <= 1e-12",
            "reporting_interval_assignment": (
                "an adjusted-open return interval and any target-changing fill cost at "
                "its destination open belong to that destination session; the common "
                "genesis purchase belongs to 2000-01-03. The terminal-close factor, "
                "when reported, belongs to the final evidence session"
            ),
            "reporting_support": {
                "development_years": list(range(2005, 2019)),
                "confirmation_years": list(range(2019, 2024)),
                "final_periods": ["2024", "2025", "2026_ytd_through_2026-07-09"],
                "warmup_excluded_from_qualification_gates": list(range(2000, 2005)),
            },
            "adjusted_open_report_variant": (
                "normalize each continuous arm at the boundary immediately before the "
                "first destination-open interval assigned to the report; apply every "
                "open-to-open return and fill cost assigned through the report's final "
                "session open, with no reset or liquidation"
            ),
            "terminal_adjusted_close_report_variant": (
                "start from the adjusted-open report variant and append exactly one "
                "same-session factor at that report window's final date: adjusted_close/"
                "adjusted_open for an arm long after the final-open fill, or 1 for cash. "
                "Do not carry this diagnostic close valuation into the continuous "
                "account or the next report window and do not charge a liquidation cost"
            ),
            "maximum_drawdown": (
                "within the exact scored report window, begin with normalized wealth 1 "
                "immediately before its first assigned interval; sample wealth after "
                "each destination-open return and fill. For the terminal-close variant "
                "append only its one final-close observation. At each observation peak "
                "is the maximum wealth seen including the initial 1; MDD = min(wealth/"
                "running_peak - 1), so MDD is finite and nonpositive"
            ),
            "drawdown_comparison": (
                "at both 5bps and 10bps and under each terminal variant, strategy_MDD "
                ">= same-ledger_AAPL_MDD - 0.01 over the continuous final window"
            ),
            "active_log_edge": (
                "sum(log(strategy_net_return_factor) - "
                "log(same-ledger_AAPL_net_return_factor)) over exact open intervals"
            ),
            "incremental_log_edge": (
                "sum(log(left_policy_net_return_factor) - "
                "log(right_control_net_return_factor)) over identical intervals"
            ),
            "negative_aapl_year": (
                "same-ledger AAPL calendar-year log return < -1e-12; a strategy win "
                "requires calendar-year active log edge > 1e-12"
            ),
            "action_difference": (
                "one eligible filing where effective schedule_overlay booleans differ "
                "after readiness, threshold, and each arm's non-overlap state; exclude "
                "a final-cutoff prediction whose next-open action remains pending"
            ),
            "difference_block": (
                "one declared development block containing at least one action "
                "difference assigned by filing decision session; count each block once"
            ),
            "semantic_difference_year": (
                "one calendar year containing at least one full-semantic versus "
                "quality-preserving no-meaning action difference assigned by filing "
                "decision session; count each year once"
            ),
            "xor_interval": (
                "one maximal contiguous adjusted-open ledger interval where the two "
                "actual compared target exposures differ; its contribution is the sum "
                "of their daily net log-return difference including changing-leg costs"
            ),
            "complete_xor_interval": (
                "an XOR interval whose unequal exposure starts and returns to equality "
                "inside the scored physical window"
            ),
            "overlay_episode": (
                "one accepted semantic filing overlay from its t+1 entry open through "
                "its t+21 exit open; incremental contribution is combined-policy minus "
                "baseline-only net log return over that exact interval"
            ),
            "episode_win_rate": (
                "strictly positive complete contributions divided by all complete "
                "contributions; zero and negative contributions are non-wins, and an "
                "empty denominator is undefined and fails every dependent gate"
            ),
            "open_boundary_episode_rule": (
                "carried or still-open contributions enter aggregate ledger edge, but "
                "only episodes/intervals with both fill boundaries inside available "
                "data enter complete-count, win-rate, median, and concentration gates"
            ),
            "best_block_removal": (
                "total edge minus max(block edge); block edges partition the scored "
                "open intervals exactly, and a removed block is never refit or replayed"
            ),
            "best_episode_removal": (
                "total incremental edge minus the largest strictly positive complete "
                "episode or XOR contribution; absence of a positive complete item fails"
            ),
            "positive_concentration": (
                "largest strictly positive complete contribution divided by the sum "
                "of all strictly positive complete contributions"
            ),
            "brier_support": (
                "identical eligible filing decisions with available semantic and "
                "no-filing-meaning predictions and a t+21 label matured by the report "
                "cutoff; binary target is incremental overlay edge > 1e-12"
            ),
            "brier_relative_improvement": (
                "(ablation_brier - semantic_brier) / max(ablation_brier, 1e-12)"
            ),
            "comparison_fork_rule": (
                "event-label counterfactual arms fork immediately before that event's "
                "t+1 open; block/stage frozen-learning controls fork only at their exact "
                "declared chronology boundary. Semantic arms never fork after genesis, "
                "and reporting windows only normalize carried wealth rather than "
                "creating a new account"
            ),
        },
        "gates": {
            "development": {
                "combined_total_active_log_edge_10bps_at_least": 0.02,
                "combined_edge_without_best_block_10bps_at_least": 0.005,
                "positive_combined_blocks_10bps_at_least": 4,
                "annual_win_rate_10bps_at_least": 0.55,
                "negative_aapl_year_win_rate_10bps_at_least": 0.60,
                "complete_sec_overlay_episodes_at_least": 12,
                "overlay_episode_win_rate_10bps_at_least": 0.55,
                "overlay_median_edge_10bps_strictly_positive": True,
                "largest_positive_episode_share_10bps_at_most": 0.35,
                "schema_valid_extraction_rate_at_least": 0.90,
                "nonzero_filing_meaning_rows_at_least": 24,
                "incremental_vs_baseline_10bps_at_least": 0.005,
                "incremental_vs_baseline_without_best_block_10bps_strictly_positive": True,
                "online_vs_block_frozen_action_differences_at_least": 5,
                "online_vs_block_frozen_difference_blocks_at_least": 3,
                "online_vs_block_frozen_10bps_edge_strictly_positive": True,
                "semantic_vs_no_filing_meaning_action_differences_at_least": 5,
                "semantic_vs_no_filing_meaning_difference_blocks_at_least": 3,
                "semantic_vs_no_filing_meaning_complete_xor_intervals_at_least": 4,
                "semantic_vs_no_filing_meaning_10bps_edge_at_least": 0.005,
                "semantic_edge_without_best_xor_10bps_strictly_positive": True,
                "semantic_brier_relative_improvement_at_least": 0.01,
            },
            "confirmation": {
                "combined_active_log_edge_positive_at_5_and_10bps": True,
                "combined_positive_years_at_least_3_at_both_5_and_10bps": True,
                "schema_valid_extraction_rate_at_least": 0.90,
                "nonzero_filing_meaning_rows_at_least": 6,
                "incremental_vs_baseline_10bps_at_least": 0.0025,
                "incremental_without_best_episode_10bps_strictly_positive": True,
                "semantic_vs_no_filing_meaning_action_differences_at_least": 3,
                "semantic_difference_years_at_least": 2,
                "semantic_vs_no_filing_meaning_10bps_edge_at_least": 0.0025,
                "online_vs_frozen_action_differences_at_least": 2,
                "online_vs_frozen_10bps_edge_strictly_positive": True,
            },
            "final": {
                "active_edge_5bps_each_2024_2025_2026_ytd_at_least": 0.005,
                "continuous_active_edge_5bps_at_least": 0.02,
                "active_edge_positive_each_period_at_10bps": True,
                "schema_valid_extraction_rate_at_least": 0.90,
                "nonzero_filing_meaning_rows_at_least": 3,
                "incremental_vs_baseline_continuous_10bps_strictly_positive": True,
                "incremental_vs_baseline_positive_periods_10bps_at_least": 2,
                "semantic_vs_no_filing_meaning_action_differences_at_least": 2,
                "semantic_vs_no_filing_meaning_continuous_10bps_strictly_positive": True,
                "online_vs_frozen_continuous_10bps_strictly_positive": True,
                "post_2023_online_vs_frozen_action_differences_at_least": 3,
                "complete_sec_overlay_episodes_at_least": 6,
                "overlay_episode_win_rate_10bps_at_least": 0.55,
                "largest_positive_episode_share_10bps_at_most": 0.50,
                "max_drawdown_not_worse_than_aapl_by_more_than_at_5_and_10bps": 0.01,
                "terminal_valuation_methods_required": [
                    "adjusted_open",
                    "terminal_adjusted_close",
                ],
                "all_return_edge_and_drawdown_gates_pass_under_both_terminal_valuations": True,
                "undefined_metric_fails": True,
            },
            "zero_action_difference_is_rejection": True,
            "undefined_or_nonfinite_metric_fails_every_dependent_gate": True,
            "failure_blocks_next_stage": True,
        },
        "runtime": {
            "scope": (
                "each effectful acquisition attempt and each scored stage attempt is "
                "independently strictly below 3600 seconds; no individual approach "
                "test may combine multiple attempts to evade this ceiling"
            ),
            "cumulative_lifecycle_rule": (
                "report the sum of all acquisition and stage attempts separately; do "
                "not describe the multi-stage research lifecycle as a sub-hour run"
            ),
            "acquisition_seconds_at_most": MAX_SEC_SECONDS,
            "sec_seconds_at_most": MAX_SEC_SECONDS,
            "sec_requests_at_most": MAX_SEC_REQUESTS,
            "sec_transport_bytes_at_most": MAX_SEC_BYTES,
            "sec_requests_per_second_at_most": MAX_SEC_REQUESTS_PER_SECOND,
            "market_seconds_at_most_within_acquisition": MAX_MARKET_SECONDS,
            "market_requests_each_stage": MAX_MARKET_REQUESTS_PER_STAGE,
            "sec_plus_market_combined_seconds_at_most": MAX_SEC_SECONDS,
            "gemma_seconds_at_most": MAX_MODEL_SECONDS,
            "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
            "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
            "contingency_seconds": 239,
            "maximum_phase_caps_plus_contingency_seconds": 3599,
            "per_attempt_phase_budgets": {
                "development_acquisition": {
                    "acquisition_seconds_at_most": MAX_SEC_SECONDS,
                    "gemma_seconds_at_most": 0,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
                "development_scored": {
                    "acquisition_seconds_at_most": 0,
                    "gemma_seconds_at_most": MAX_MODEL_SECONDS,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
                "confirmation_scored": {
                    "acquisition_seconds_at_most": MAX_SEC_SECONDS,
                    "gemma_seconds_at_most": MAX_MODEL_SECONDS,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
                "final_scored": {
                    "acquisition_seconds_at_most": MAX_SEC_SECONDS,
                    "gemma_seconds_at_most": MAX_MODEL_SECONDS,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
            },
            "model_call_caps": {
                "development_2000_2018": 80,
                "confirmation_2019_2023": 20,
                "final_2024_2026_ytd": 12,
            },
            "latency_preflight": {
                "selection": (
                    "after deterministic preprocessing, select the five development "
                    "events with greatest canonical UTF-8 model-request byte length; "
                    "break ties by accession ascending"
                ),
                "execution_order": (
                    "those five execute first in descending request-byte length then "
                    "accession ascending; remaining events execute by availability "
                    "session then acceptance timestamp then accession"
                ),
                "calls_are_sealed_batch_calls_not_duplicates": True,
                "projection": (
                    "sum(first_five_elapsed_seconds) + remaining_call_count * "
                    "max(first_five_elapsed_seconds)"
                ),
                "projection_must_not_exceed_gemma_seconds": MAX_MODEL_SECONDS,
                "minimum_preflight_calls": 5,
            },
            "clock_and_process": (
                "an external parent process samples monotonic elapsed time before and "
                "after each phase and the complete current attempt, enforces shrinking "
                "subprocess deadlines, and terminates a worker on budget overrun"
            ),
            "failure_on_budget_overrun": True,
        },
        "stage_access": {
            "preregistration_binding": (
                "the first implementation commit must record the exact commit that "
                "contains this literal contract hash; every effectful attempt binds "
                "that commit, the implementation commit, clean-tree identity, and all "
                "new-v2 source hashes before access"
            ),
            "development_acquisition": {
                "attempt_id": DEVELOPMENT_ACQUISITION_ID,
                "one_shot": True,
                "consume_before_first_official_sec_or_market_network_request": True,
                "no_model_or_canonical_market_value_access": True,
                "private_quarantine_outputs_only": (
                    "exact raw bytes, authentication receipts, hashes, counts, and "
                    "deterministic blinded model requests; no semantic output, price "
                    "value, return, label, action, or score"
                ),
                "failed_or_indeterminate_acquisition_is_terminal": True,
            },
            "development": {
                "attempt_id": DEVELOPMENT_ATTEMPT_ID,
                "one_shot": True,
                "acquisition_pass_required": True,
                "consume_before_first_real_gemma_call_or_first_canonical_market_value_read": True,
                "latency_preflight_calls_occur_after_consumption": True,
                "indeterminate_execution_is_terminal": True,
            },
            "confirmation": {
                "attempt_id": CONFIRMATION_ATTEMPT_ID,
                "one_shot": True,
                "consume_before_first_stage_network_request": True,
                "consume_before_first_2019_feature_or_outcome_read": True,
                "indeterminate_execution_is_terminal": True,
                "development_pass_required": True,
            },
            "final": {
                "attempt_id": FINAL_ATTEMPT_ID,
                "one_shot": True,
                "consume_before_first_stage_network_request": True,
                "consume_before_first_2024_feature_or_outcome_read": True,
                "indeterminate_execution_is_terminal": True,
                "confirmation_pass_required": True,
                "predecessor_registry_pin_file": (
                    "docs/protocol_evidence/sec_gemma_reveal_registry_initial_pin.json"
                ),
                "predecessor_registry_pin_file_sha256": (
                    "85fe468ddad10a7fa1d226ea769a65cf91a359047075b9567f967fe4f776fdde"
                ),
                "predecessor_registry_sha256": (
                    "5853fcfc8f9ddb651981cb4b1c9f9426e57b214898ab6f080c0485eb1b40fe61"
                ),
                "predecessor_registry_tip_sha256": (
                    "10b0ecce18437a9c0e7f03f27882a3d7ba8836c1bcc48a75c3a64d0623de879a"
                ),
                "historical_final_reveal_count_lower_bound": 10,
                "register_and_externally_pin_before_consumption": True,
            },
            "result_release": (
                "a stage emits no semantic extraction, prediction, partial metric, "
                "action count, return, gate, or direction before its sealed joint "
                "report; failure never authorizes another attempt or a sibling chosen "
                "from the revealed result"
            ),
        },
        "artifacts": {
            "predictions_and_actions_sealed_before_outcomes": True,
            "append_only_lessons": True,
            "same_ledger_benchmark": True,
            "independent_no_leverage_verification": True,
            "complete_metrics": True,
            "checksums": True,
            "failed_attempts_preserved": True,
            "later_stage_joint_release_only": True,
            "input_source_hashes_bound_before_stage_access": True,
            "pending_actions_and_lessons_preserved_across_boundaries": True,
        },
    }
    return copy.deepcopy(manifest)


CONTRACT_SHA256: Final[str] = (
    "57b325b25ae53f650538a0265622a6c662e7fb1894a78e805705f7bae3c55d5d"
)
if canonical_sha256(build_contract_manifest()) != CONTRACT_SHA256:
    raise RuntimeError(
        "SEC/Gemma online-overlay manifest no longer matches its literal preregistration"
    )


def validate_contract_manifest(value: Any) -> dict[str, Any]:
    """Accept only the exact preregistered manifest and return a detached copy."""

    if not isinstance(value, dict):
        raise SecGemmaOnlineRiskOverlayContractError(
            "SEC/Gemma online-overlay contract must be a mapping"
        )
    expected = build_contract_manifest()
    try:
        observed_bytes = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise SecGemmaOnlineRiskOverlayContractError(
            "SEC/Gemma online-overlay contract is not canonical JSON"
        ) from exc
    if observed_bytes != canonical_json_bytes(expected):
        raise SecGemmaOnlineRiskOverlayContractError(
            "SEC/Gemma online-overlay contract differs from preregistration"
        )
    return copy.deepcopy(expected)


__all__ = [
    "BASELINE_POLICY_ID",
    "BASELINE_SOURCE_FILE",
    "BASELINE_SOURCE_SHA256",
    "BRANCH_NAME",
    "CONTRACT_SHA256",
    "CONTRACT_VERSION",
    "CONFIRMATION_ATTEMPT_ID",
    "CONTROL_FEATURES",
    "DEVELOPMENT_ACQUISITION_ID",
    "DEVELOPMENT_ATTEMPT_ID",
    "DEVELOPMENT_BLOCKS",
    "EXPECTED_EDGE_GATE",
    "FEATURES",
    "FINAL_ATTEMPT_ID",
    "HORIZON_SESSIONS",
    "MARKET_FEATURES",
    "MEANING_FEATURES",
    "MAX_MARKET_REQUESTS_PER_STAGE",
    "MAX_MARKET_SECONDS",
    "MAX_TOTAL_RUNTIME_SECONDS",
    "MINIMUM_CLASS_ROWS",
    "MINIMUM_TRAINING_ROWS",
    "MODEL_CONFIG_DIGEST",
    "MODEL_LAYER_DIGESTS",
    "MODEL_MANIFEST_SHA256",
    "MODEL_NAME",
    "OLLAMA_VERSION",
    "PROBABILITY_GATE",
    "QUALITY_FEATURE",
    "RUNTIME_FINGERPRINT_SHA256",
    "SEMANTIC_FEATURES",
    "SOURCE_PIN_FILES",
    "SOURCE_PINS",
    "SecGemmaOnlineRiskOverlayContractError",
    "build_contract_manifest",
    "build_runtime_fingerprint_material",
    "canonical_json_bytes",
    "canonical_sha256",
    "validate_contract_manifest",
]
