from __future__ import annotations

import json

import pytest

from agent_benchmark import llm_client
from agent_benchmark.historical_blinding import (
    HISTORICAL_BLINDING_CONTRACT,
    blind_historical_prompt,
    restore_historical_output,
    should_blind_historical_prompt,
)
from agent_benchmark.local_provider import (
    local_gemma_aapl_causal_replay_config,
    local_gemma_aapl_online_config,
    local_gemma_secret_config,
    validate_no_paid_api_mode,
)


def test_frozen_aapl_preset_requires_parametric_lookahead_blinding():
    config = local_gemma_aapl_online_config()

    assert config.historical_prompt_blinding is True
    assert config.historical_prompt_blinding_contract == HISTORICAL_BLINDING_CONTRACT
    assert config.model_training_data_cutoff == "2025-01-31"
    assert config.historical_decision_authority == "precutoff_quantitative_policy"
    assert should_blind_historical_prompt(config) is True
    validate_no_paid_api_mode(config, local_gemma_secret_config())

    unsafe = local_gemma_aapl_online_config(historical_prompt_blinding=False)
    with pytest.raises(ValueError, match="historical_prompt_blinding"):
        validate_no_paid_api_mode(unsafe, local_gemma_secret_config())

    missing_cutoff = local_gemma_aapl_online_config(model_training_data_cutoff="")
    with pytest.raises(ValueError, match="model_training_data_cutoff"):
        validate_no_paid_api_mode(missing_cutoff, local_gemma_secret_config())

    wrong_contract = local_gemma_aapl_online_config(
        historical_prompt_blinding_contract="old-contract",
    )
    with pytest.raises(ValueError, match="requires blinding contract"):
        validate_no_paid_api_mode(wrong_contract, local_gemma_secret_config())

    news_exposes_identity = local_gemma_aapl_online_config(
        max_news_per_symbol=1,
    )
    with pytest.raises(ValueError, match="forbids news text"):
        validate_no_paid_api_mode(news_exposes_identity, local_gemma_secret_config())

    causal_unblinded = local_gemma_aapl_causal_replay_config(
        historical_prompt_blinding=False,
    )
    assert should_blind_historical_prompt(causal_unblinded) is False
    with pytest.raises(ValueError, match="causal_online_replay requires historical_prompt_blinding"):
        validate_no_paid_api_mode(causal_unblinded, local_gemma_secret_config())

    wrong_company = local_gemma_aapl_online_config(company_name="")
    with pytest.raises(ValueError, match="requires symbol='AAPL' and company_name='Apple'"):
        validate_no_paid_api_mode(wrong_company, local_gemma_secret_config())

    llm_authority = local_gemma_aapl_online_config(historical_decision_authority="llm")
    with pytest.raises(ValueError, match="precutoff_quantitative_policy decision authority"):
        validate_no_paid_api_mode(llm_authority, local_gemma_secret_config())


def test_historical_prompt_blinding_removes_identity_dates_and_raw_amounts():
    config = local_gemma_aapl_online_config()
    payload = {
        "task": "Decide whether Apple/AAPL should beat AAPL buy and hold.",
        "input_bundle": {
            "decision_date": "2024-03-15",
            "fill_date": "2024-03-18",
            "candidate_universe": [
                {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"}
            ],
            "portfolio_state": {
                "cash": 1000.0,
                "equity": 1000.0,
                "positions": {"AAPL": 4.25},
            },
            "market_snapshots": {
                "AAPL": {
                    "as_of_date": "2024-03-15",
                    "open": 171.0,
                    "high": 173.0,
                    "low": 170.0,
                    "close": 172.62,
                    "adj_close": 172.10,
                    "volume": 121_000_000,
                    "return_20d": -0.043,
                    "volatility_20d": 0.22,
                    "drawdown_60d": -0.09,
                }
            },
            "index_context": [
                {
                    "symbol": "^VIX",
                    "date": "2024-03-15",
                    "close": 14.41,
                    "return_20d": 0.12,
                },
                {"symbol": "SPY", "date": "2024-03-15", "return_20d": 0.03},
            ],
            "fundamentals": {
                "AAPL": {"Revenue": {"value": 119_575_000_000, "filed_date": "2024-02-02"}}
            },
            "memory": [
                {
                    "symbol": "AAPL",
                    "decision_date": "2018-10-01",
                    "entry_price": 55.0,
                    "outcome_20": -0.08,
                }
            ],
        },
    }

    blinded = blind_historical_prompt(
        config,
        "You trade Apple stock AAPL using SPY, QQQ, VIX, and TNX.",
        json.dumps(payload),
    )
    combined = f"{blinded.system}\n{blinded.user}"
    lower = combined.lower()

    for forbidden in (
        "aapl",
        "apple",
        "2024-03-15",
        "2024-03-18",
        "2018-10-01",
        "172.62",
        "121000000",
        "119575000000",
        "entry_price",
        '"fundamentals"',
    ):
        assert forbidden not in lower

    assert "ASSET_1" in combined
    assert "ANONYMOUS_COMPANY" in combined
    assert "MARKET_1" in combined
    assert "SENTIMENT_1" in combined
    assert "RATES_1" in combined
    assert "T0" in combined
    assert "T+3D" in combined
    assert '"return_20d": -0.043' in combined
    assert '"volatility_20d": 0.22' in combined
    assert '"outcome_20": -0.08' in combined
    assert '"positions": {"ASSET_1": "LONG"}' in combined

    metadata = blinded.metadata()
    assert metadata["asset_identity_exposed"] is False
    assert metadata["anchor_exposed"] is False
    assert metadata["absolute_market_values_exposed"] is False
    assert metadata["raw_fundamental_amounts_exposed"] is False


def test_historical_output_restores_only_engine_identifiers():
    config = local_gemma_aapl_online_config()
    blinded = blind_historical_prompt(
        config,
        "Return JSON for AAPL.",
        json.dumps({"decision_date": "2024-01-02", "symbol": "AAPL"}),
    )

    restored = restore_historical_output(
        {
            "analyses": [{"symbol": "ASSET_1", "key_evidence": ["MARKET_1 weak at T-5D"]}],
            "note": "ANONYMOUS_COMPANY and SENTIMENT_1",
        },
        blinded.aliases_to_real,
    )

    assert restored["analyses"][0]["symbol"] == "AAPL"
    assert restored["analyses"][0]["key_evidence"] == ["HISTORICAL_TEXT_REDACTED"]
    assert restored["note"] == "HISTORICAL_TEXT_REDACTED"
    assert "2024" not in json.dumps(restored)

    lowercase = restore_historical_output(
        {"analyses": [{"symbol": "asset_1"}]},
        blinded.aliases_to_real,
    )
    assert lowercase["analyses"][0]["symbol"] == "AAPL"

    with pytest.raises(ValueError, match="real identity"):
        restore_historical_output({"note": "VIX and TNX"}, blinded.aliases_to_real)
    with pytest.raises(ValueError, match="Unknown historical output alias"):
        restore_historical_output({"note": "ASSET_2"}, blinded.aliases_to_real)
    with pytest.raises(ValueError, match="alias collision"):
        restore_historical_output(
            {"ASSET_1": True, "asset_1": False},
            blinded.aliases_to_real,
        )


def test_blinding_strips_execution_aliases_and_large_numbers_in_free_text():
    config = local_gemma_aapl_causal_replay_config()
    payload = {
        "decision_date": "2024-03-15",
        "execution": {
            "trades": [
                {
                    "symbol": "AAPL",
                    "reference_price": 172.62,
                    "fill_price": 172.70631,
                    "signed_delta": 5.793,
                    "shares": 5.793,
                    "commission": 1.25,
                }
            ],
            "fees": 1.25,
            "slippage_cost": 0.5,
        },
        "stage1_outputs": [
            {
                "symbol": "AAPL",
                "key_evidence": [
                    "Apple close 172.62 on 2024-03-15 with 121000000 volume; return -0.043"
                ],
            }
        ],
    }

    blinded = blind_historical_prompt(config, "Reflect on AAPL.", json.dumps(payload))
    combined = f"{blinded.system}\n{blinded.user}"

    for forbidden in (
        "AAPL",
        "Apple",
        "2024-03-15",
        "172.62",
        "172.70631",
        "5.793",
        "121000000",
        "reference_price",
        "fill_price",
        "signed_delta",
        '"shares"',
        '"commission"',
        '"fees"',
        '"slippage_cost"',
    ):
        assert forbidden not in combined
    assert "SCALE_REDACTED" in combined
    assert "-0.043" not in combined


def test_blinding_handles_identifier_tickers_scientific_numbers_and_unknown_names():
    config = local_gemma_aapl_online_config()
    payload = {
        "decision_date": "2024-03-15",
        "candidate_universe": [{"symbol": "AAPL", "name": "Apple Incorporated"}],
        "state_features": {
            "aapl_return_20d": -0.04,
            "spy_return_20d": 0.03,
            "qqq_return_20d": 0.05,
        },
        "memo": (
            "SPY20 QQQ20 AAPL20 price 1.7262e2 volume 1.21e8 alternate "
            "17262e-2 .17262e3 iPhone Cupertino Tim Cook"
        ),
    }

    blinded = blind_historical_prompt(
        config,
        "Compare SPY20, QQQ20, and AAPL20.",
        json.dumps(payload),
    )
    combined = f"{blinded.system}\n{blinded.user}"

    for forbidden in (
        "SPY20",
        "QQQ20",
        "AAPL20",
        "spy_return_20d",
        "qqq_return_20d",
        "aapl_return_20d",
        "Apple Incorporated",
        "1.7262e2",
        "1.21e8",
        "17262e-2",
        ".17262e3",
        "iPhone",
        "Cupertino",
        "Tim Cook",
    ):
        assert forbidden not in combined
    assert "MARKET_1_20" in combined
    assert "MARKET_2_20" in combined
    assert "ASSET_1_20" in combined
    assert "market_1_return_20d" in combined.lower()
    assert "ANONYMOUS_COMPANY" in combined
    assert "ISSUER_TERM_REDACTED" in combined


def test_historical_output_rejects_unknown_numerical_fields():
    config = local_gemma_aapl_online_config()
    blinded = blind_historical_prompt(
        config,
        "Analyze AAPL.",
        json.dumps({"decision_date": "2024-03-15", "symbol": "AAPL"}),
    )

    with pytest.raises(ValueError, match="Unknown numerical field"):
        restore_historical_output(
            {
                "analyses": [
                    {
                        "symbol": "ASSET_1",
                        "stance": "neutral",
                        "confidence": 0.5,
                        "expected_return_bps": 0,
                        "horizon_days": 20,
                        "proposed_target_weight": 1.0,
                        "leaked_return": 172.62,
                    }
                ]
            },
            blinded.aliases_to_real,
        )

    critic = restore_historical_output(
        {
            "recommended_exposure_band": [0.0, 1.0],
            "bull_exposure_case": "generic",
            "defensive_case": "generic",
            "cash_drag_risk": "generic",
            "key_disagreement": "generic",
        },
        blinded.aliases_to_real,
    )
    assert critic["recommended_exposure_band"] == [0.0, 1.0]
    assert critic["bull_exposure_case"] == "HISTORICAL_TEXT_REDACTED"


def test_call_json_model_sends_only_blinded_prompt_and_restores_symbol(monkeypatch):
    config = local_gemma_aapl_online_config(use_cached_llm=False)
    captured = {}

    def fake_call(active_config, secrets, system, user, *, cache_namespace):
        captured.update({"system": system, "user": user, "namespace": cache_namespace})
        return {
            "message": {
                "content": json.dumps(
                    {
                        "analyses": [
                            {
                                "symbol": "ASSET_1",
                                "stance": "neutral",
                                "confidence": 0.5,
                                "expected_return_bps": 0,
                                "horizon_days": 20,
                                "key_evidence": ["MARKET_1 at T0"],
                                "memory_refs": [],
                                "uncertainty": [],
                                "proposed_target_weight": 1.0,
                            }
                        ],
                        "market_regime_notes": "SENTIMENT_1",
                        "data_quality_notes": [],
                    }
                )
            }
        }

    monkeypatch.setattr(llm_client, "_call_ollama_native_chat", fake_call)
    result = llm_client.call_json_model(
        config,
        local_gemma_secret_config(),
        "Analyze AAPL and Apple using SPY on 2024-03-15.",
        json.dumps(
            {
                "decision_date": "2024-03-15",
                "symbol": "AAPL",
                "close": 172.62,
                "return_20d": -0.043,
            }
        ),
        cache_namespace="stage1",
    )

    sent = f"{captured['system']}\n{captured['user']}".lower()
    assert "aapl" not in sent
    assert "apple" not in sent
    assert "2024-03-15" not in sent
    assert "172.62" not in sent
    assert "asset_1" in sent
    assert '"return_20d": -0.043' in sent
    assert result["analyses"][0]["symbol"] == "AAPL"
    assert result["analyses"][0]["key_evidence"] == ["HISTORICAL_TEXT_REDACTED"]
    assert result["_historical_prompt_blinding"]["applied"] is True
    # Preserve the exact pseudonymous response for the audit trail.
    assert "ASSET_1" in result["_raw_text"]


def test_historical_json_repair_does_not_echo_unblinded_model_text(monkeypatch):
    config = local_gemma_aapl_online_config(use_cached_llm=False)
    calls = []

    def fake_call(active_config, secrets, system, user, *, cache_namespace):
        calls.append({"system": system, "user": user})
        if len(calls) == 1:
            return {
                "message": {
                    "content": "AAPL Apple 2024-03-15 close 172.62 volume 121000000; not JSON"
                }
            }
        return {
            "message": {
                "content": json.dumps(
                    {
                        "analyses": [],
                        "market_regime_notes": "repaired",
                        "data_quality_notes": [],
                    }
                )
            }
        }

    monkeypatch.setattr(llm_client, "_call_ollama_native_chat", fake_call)
    monkeypatch.setattr(llm_client, "_write_malformed_response", lambda *args, **kwargs: None)
    result = llm_client.call_json_model(
        config,
        local_gemma_secret_config(),
        "Analyze AAPL on 2024-03-15.",
        json.dumps({"decision_date": "2024-03-15", "symbol": "AAPL"}),
        cache_namespace="stage1",
    )

    assert len(calls) == 2
    retry = calls[1]["user"]
    for forbidden in ("AAPL", "Apple", "2024-03-15", "172.62", "121000000"):
        assert forbidden not in retry
    assert "Omitted by historical look-ahead guard" in retry
    assert result["_historical_prompt_blinding"]["applied"] is True


def test_historical_semantic_validation_retry_uses_generic_error(monkeypatch):
    config = local_gemma_aapl_online_config(use_cached_llm=False)
    calls = []

    def fake_call(active_config, secrets, system, user, *, cache_namespace):
        calls.append(user)
        if len(calls) == 1:
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "analyses": [
                                {
                                    "symbol": "AAPL",
                                    "stance": "neutral",
                                    "confidence": 0.5,
                                    "expected_return_bps": 0,
                                    "horizon_days": 20,
                                    "key_evidence": ["iPhone"],
                                    "memory_refs": [],
                                    "uncertainty": [],
                                    "proposed_target_weight": 1.0,
                                }
                            ],
                            "market_regime_notes": "iPhone",
                            "data_quality_notes": [],
                        }
                    )
                }
            }
        return {
            "message": {
                "content": json.dumps(
                    {
                        "analyses": [],
                        "market_regime_notes": "generic",
                        "data_quality_notes": [],
                    }
                )
            }
        }

    monkeypatch.setattr(llm_client, "_call_ollama_native_chat", fake_call)
    monkeypatch.setattr(llm_client, "_write_malformed_response", lambda *args, **kwargs: None)
    llm_client.call_json_model(
        config,
        local_gemma_secret_config(),
        "Analyze AAPL.",
        json.dumps({"decision_date": "2024-03-15", "symbol": "AAPL"}),
        cache_namespace="stage1",
    )

    assert len(calls) == 2
    retry = calls[1]
    assert "AAPL" not in retry
    assert "iPhone" not in retry
    assert "historical_output_validation_failed" in retry


def test_historical_cache_is_forbidden():
    config = local_gemma_aapl_online_config(use_cached_llm=True)
    with pytest.raises(ValueError, match="forbids cached LLM decisions"):
        validate_no_paid_api_mode(config, local_gemma_secret_config())
