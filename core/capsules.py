
import math
import os, json, hashlib, re
from typing import Any, List, Dict, Optional


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
    price = _rounded_indicator(price_row.get("close", 0), 4, default=0.0)
    rsi = _rounded_indicator(price_row.get("rsi", 0), 3, default=0.0)
    macd_hist = _rounded_indicator(price_row.get("macd_hist", 0), 5, default=0.0)
    atr = _rounded_indicator(price_row.get("atr", 0), 5, default=0.0)
    sma200 = _rounded_indicator(price_row.get("sma200", 0), 4, default=0.0)
    sma100 = _rounded_indicator(price_row.get("sma100", 0), 4, default=0.0)

    cap = {
        "date": date_iso,
        "symbol": symbol,
        "price": price,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "atr": atr,
        "trend_up": int(price_row.get("trend_up", 0)),
        "sma200": sma200,
        "sma100": sma100,
        "regime": regime,
        "headlines": headlines[:5],
        "macro": macro_h[:3],
    }
    cap["hash"] = hashlib.sha256(json.dumps(cap, sort_keys=True).encode()).hexdigest()
    return cap


def _finite_or_default(value: Any, *, default: Optional[float] = 0.0) -> Optional[float]:
    """Return a finite float or ``default`` when the input is not usable."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric is None:
        return None
    if not math.isfinite(numeric):
        numeric = default
    if numeric is None:
        return None
    return numeric


def _rounded_indicator(
    value: Any, digits: int, *, default: Optional[float] = 0.0
) -> Optional[float]:
    """Round indicator values after ensuring they are finite numbers."""

    numeric = _finite_or_default(value, default=default)
    if numeric is None:
        return None
    return round(numeric, digits)
