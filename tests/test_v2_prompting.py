import json

from agent_benchmark.prompting import build_decision_prompt


def test_prompt_contains_model_owned_decision_contract():
    bundle = {
        "symbol": "AAPL",
        "as_of_date": "2025-01-02",
        "portfolio_state": {"cash": 1000, "position_shares": 0, "equity": 1000},
        "market": {"close": 100},
        "news": {"items": []},
    }

    system, user = build_decision_prompt(bundle)
    payload = json.loads(user)

    assert "sole AI portfolio manager" in system
    assert "simulator will only apply market mechanics" in system
    assert payload["input_bundle"]["symbol"] == "AAPL"
    assert "target_exposure" in payload["decision_output_contract"]
