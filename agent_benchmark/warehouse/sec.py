from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests

from .store import Warehouse
from .universe import SymbolMeta, STOCKS

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT_FALLBACK = "LLM-memory-trading-agent local research contact@example.com"

CONCEPTS = {
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "EarningsPerShareDiluted",
    "EntityCommonStockSharesOutstanding",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebtNoncurrent",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
}


def _headers(user_agent: str) -> Dict[str, str]:
    return {"User-Agent": user_agent or USER_AGENT_FALLBACK, "Accept-Encoding": "gzip, deflate"}


def fetch_ticker_map(user_agent: str) -> Dict[str, str]:
    resp = requests.get(TICKERS_URL, headers=_headers(user_agent), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    out = {}
    for item in data.values():
        ticker = str(item.get("ticker", "")).upper()
        cik = item.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = f"{int(cik):010d}"
    return out


def _sec_ticker(symbol: str) -> str:
    return symbol.replace("-", ".").upper()


def fetch_companyfacts(cik: str, user_agent: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    resp = requests.get(url, headers=_headers(user_agent), timeout=45)
    resp.raise_for_status()
    return resp.json()


def fetch_submissions(cik: str, user_agent: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    resp = requests.get(url, headers=_headers(user_agent), timeout=45)
    resp.raise_for_status()
    return resp.json()


def _to_date(value: Any):
    if not value:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _fact_rows(symbol: str, cik: str, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    us_gaap = ((facts.get("facts") or {}).get("us-gaap") or {})
    for concept, payload in us_gaap.items():
        if concept not in CONCEPTS:
            continue
        for unit, items in (payload.get("units") or {}).items():
            for item in items:
                filed_date = _to_date(item.get("filed"))
                period_end = _to_date(item.get("end"))
                if filed_date is None or period_end is None:
                    continue
                try:
                    value = float(item.get("val"))
                except Exception:
                    continue
                raw_id = f"{symbol}|{cik}|{concept}|{unit}|{item.get('filed')}|{item.get('end')}|{item.get('form')}|{value}"
                rows.append(
                    {
                        "fact_id": hashlib.sha1(raw_id.encode("utf-8")).hexdigest(),
                        "symbol": symbol,
                        "cik": cik,
                        "concept": concept,
                        "unit": unit,
                        "value": value,
                        "period_start": _to_date(item.get("start")),
                        "period_end": period_end,
                        "filed_date": filed_date,
                        "fiscal_year": item.get("fy"),
                        "fiscal_period": item.get("fp"),
                        "form": item.get("form"),
                        "source": "sec_companyfacts",
                    }
                )
    return rows


def download_sec(warehouse: Warehouse, symbols: Iterable[SymbolMeta] | None = None, *, user_agent: str = "") -> Dict[str, Any]:
    symbols = [item for item in list(symbols or STOCKS) if item.kind == "stock"]
    summary = {"symbols": len(symbols), "ok": 0, "missing_cik": 0, "error": 0, "facts": 0, "errors": []}
    try:
        ticker_map = fetch_ticker_map(user_agent)
        warehouse.log("sec", "sec_ticker_map", "ok", message=f"{len(ticker_map)} tickers")
    except Exception as exc:
        warehouse.log("sec", "sec_ticker_map", "error", message=str(exc))
        return {**summary, "error": len(symbols), "errors": [{"symbol": "*", "message": str(exc)}]}

    for meta in symbols:
        cik = ticker_map.get(_sec_ticker(meta.symbol)) or ticker_map.get(meta.symbol.upper())
        if not cik:
            warehouse.log("sec", "companyfacts", "missing_cik", symbol=meta.symbol, message="No CIK in SEC ticker map")
            summary["missing_cik"] += 1
            continue
        try:
            facts = fetch_companyfacts(cik, user_agent)
            submissions = fetch_submissions(cik, user_agent)
            rows = _fact_rows(meta.symbol, cik, facts)
            if rows:
                warehouse.upsert_frame("sec_facts", pd.DataFrame(rows).drop_duplicates("fact_id", keep="last"), ["fact_id"])
            warehouse.update_symbol_price_dates(meta.symbol, None, None, cik)
            warehouse.log("sec", "companyfacts", "ok", symbol=meta.symbol, message=f"{len(rows)} selected facts; submissions={len((submissions.get('filings') or {}).get('recent', {}).get('accessionNumber', []))}")
            summary["ok"] += 1
            summary["facts"] += len(rows)
        except Exception as exc:
            warehouse.log("sec", "companyfacts", "error", symbol=meta.symbol, message=str(exc))
            summary["error"] += 1
            summary["errors"].append({"symbol": meta.symbol, "message": str(exc)})
    warehouse.export_parquet(["symbols", "sec_facts", "download_log"])
    return summary
