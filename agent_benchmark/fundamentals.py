from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config_store import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "sec"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _headers(user_agent: str) -> Dict[str, str]:
    return {
        "User-Agent": user_agent or "LLM-memory-agent local benchmark contact@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def _ticker_cache_path() -> Path:
    return CACHE_DIR / "company_tickers.json"


def _companyfacts_path(cik: str) -> Path:
    return CACHE_DIR / f"CIK{int(cik):010d}_companyfacts.json"


def ticker_to_cik(symbol: str, user_agent: str) -> Optional[str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _ticker_cache_path()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        resp = requests.get(TICKERS_URL, headers={"User-Agent": user_agent or "LLM-memory-agent local benchmark contact@example.com"}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        path.write_text(json.dumps(data), encoding="utf-8")
    symbol = symbol.upper()
    for item in data.values():
        if str(item.get("ticker", "")).upper() == symbol:
            return str(item.get("cik_str"))
    return None


def company_facts(symbol: str, user_agent: str) -> Dict[str, Any]:
    cik = ticker_to_cik(symbol, user_agent)
    if not cik:
        return {"source": "sec_companyfacts", "status": "missing_cik", "symbol": symbol.upper()}
    path = _companyfacts_path(cik)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    resp = requests.get(url, headers=_headers(user_agent), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _latest_fact(raw: Dict[str, Any], names: list[str], units: list[str], as_of_date: str) -> Optional[Dict[str, Any]]:
    facts = ((raw.get("facts") or {}).get("us-gaap") or {})
    candidates = []
    cutoff = _parse_date(as_of_date)
    for name in names:
        fact = facts.get(name) or {}
        unit_map = fact.get("units") or {}
        for unit in units:
            for item in unit_map.get(unit, []):
                end = item.get("end")
                if not end:
                    continue
                try:
                    end_dt = _parse_date(end)
                except Exception:
                    continue
                if end_dt <= cutoff and item.get("val") is not None:
                    candidates.append(
                        {
                            "concept": name,
                            "unit": unit,
                            "value": item.get("val"),
                            "end": end,
                            "filed": item.get("filed"),
                            "form": item.get("form"),
                            "fy": item.get("fy"),
                            "fp": item.get("fp"),
                        }
                    )
    candidates.sort(key=lambda item: (item.get("end") or "", item.get("filed") or ""), reverse=True)
    return candidates[0] if candidates else None


def sec_fundamentals_snapshot(symbol: str, user_agent: str, as_of_date: str) -> Dict[str, Any]:
    if not user_agent:
        return {
            "source": "sec_companyfacts",
            "status": "needs_user_agent",
            "message": "Add an SEC user agent in local configuration for polite EDGAR access.",
        }
    try:
        raw = company_facts(symbol, user_agent)
    except Exception as exc:
        return {"source": "sec_companyfacts", "status": "error", "message": str(exc)}

    if raw.get("status"):
        return raw

    facts = {
        "revenue": _latest_fact(
            raw,
            ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
            ["USD"],
            as_of_date,
        ),
        "net_income": _latest_fact(raw, ["NetIncomeLoss"], ["USD"], as_of_date),
        "assets": _latest_fact(raw, ["Assets"], ["USD"], as_of_date),
        "liabilities": _latest_fact(raw, ["Liabilities"], ["USD"], as_of_date),
        "equity": _latest_fact(raw, ["StockholdersEquity"], ["USD"], as_of_date),
        "eps_diluted": _latest_fact(raw, ["EarningsPerShareDiluted"], ["USD/shares"], as_of_date),
        "shares_outstanding": _latest_fact(raw, ["EntityCommonStockSharesOutstanding"], ["shares"], as_of_date),
    }

    ratios: Dict[str, Any] = {}
    assets = facts.get("assets") or {}
    liabilities = facts.get("liabilities") or {}
    equity = facts.get("equity") or {}
    if assets.get("value") and liabilities.get("value"):
        ratios["liabilities_to_assets"] = float(liabilities["value"]) / max(1.0, float(assets["value"]))
    if equity.get("value") and liabilities.get("value"):
        ratios["debt_to_equity_proxy"] = float(liabilities["value"]) / max(1.0, float(equity["value"]))

    return {
        "source": "sec_companyfacts",
        "status": "ok",
        "symbol": symbol.upper(),
        "cik": raw.get("cik"),
        "entity_name": raw.get("entityName"),
        "as_of_date": as_of_date,
        "facts": facts,
        "ratios": ratios,
    }
