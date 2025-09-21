
import os, json, hashlib, re
from typing import List, Dict


def _canon(s: str) -> str:
    """Normalise text for consistent capsule hashing."""

    return re.sub(r"\s+", " ", (s or "").strip().lower())


def capsule_path(symbol: str, start: str, end: str):
    """Return the path to the capsule archive for the symbol/date range."""

    key = hashlib.md5(f"{symbol}_{start}_{end}".encode()).hexdigest()
    path = os.path.join("data", f"capsules_{key}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def build_capsule(
    date_iso: str,
    symbol: str,
    price_row: dict,
    topk: List[Dict],
    macro: List[Dict],
    regime: Dict,
) -> Dict:
    """Build a compact dictionary summarising the trading day context."""

    headlines = [
        {"t": _canon(a.get("title", ""))[:160], "s": a.get("source", "")[:40]}
        for a in topk
        if a.get("title")
    ]
    macro_h = [
        {"t": _canon(a.get("title", ""))[:160], "s": a.get("source", "")[:40]}
        for a in macro
        if a.get("title")
    ]
    cap = {
        "date": date_iso,
        "symbol": symbol,
        "price": round(float(price_row.get("close", 0)), 4),
        "rsi": round(float(price_row.get("rsi", 0)), 3),
        "macd_hist": round(float(price_row.get("macd_hist", 0)), 5),
        "atr": round(float(price_row.get("atr", 0)), 5),
        "trend_up": int(price_row.get("trend_up", 0)),
        "sma200": round(float(price_row.get("sma200", 0) or 0), 4),
        "sma100": round(float(price_row.get("sma100", 0) or 0), 4),
        "regime": regime,
        "headlines": headlines[:5],
        "macro": macro_h[:3],
    }
    cap["hash"] = hashlib.sha256(json.dumps(cap, sort_keys=True).encode()).hexdigest()
    return cap
