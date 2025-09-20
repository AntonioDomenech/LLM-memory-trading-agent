
from dataclasses import dataclass, field, asdict
import json, os, re
from typing import Dict, Any

@dataclass
class RetrievalCfg:
    k_shallow: int = 3
    k_intermediate: int = 0
    k_deep: int = 0
    Q_shallow: int = 10
    Q_intermediate: int = 20
    Q_deep: int = 30
    alpha_shallow: float = 0.4
    alpha_intermediate: float = 0.3
    alpha_deep: float = 0.3

@dataclass
class RiskCfg:
    risk_per_trade: float = 0.01
    max_position: int = 1000
    stop_loss_atr_mult: float = 3.0
    trailing_stop_atr_mult: float = 8.0
    take_profit_atr_mult: float = 6.0
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    slippage_bps: float = 8.0
    min_trade_value: float = 0.0
    min_trade_shares: int = 1
    allow_short: bool = False

@dataclass
class Config:
    symbol: str = "AAPL"
    train_start: str = "2022-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2023-08-31"
    news_source: str = "NewsAPI"
    K_news_per_day: int = 5
    embedding_model: str = "text-embedding-3-small"
    decision_model: str = "gpt-4o-mini"
    memory_path: str = "data/memory_bank.json"
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    symbol = raw.get("symbol", "AAPL")
    memory_path = raw.get("memory_path") or _find_preferred_memory_path(symbol)
    cfg = Config(
        symbol = symbol,
        train_start = raw.get("train_start", "2022-01-01"),
        train_end   = raw.get("train_end", "2022-12-31"),
        test_start  = raw.get("test_start", "2023-01-01"),
        test_end    = raw.get("test_end", "2023-08-31"),
        news_source = raw.get("news_source", "NewsAPI"),
        K_news_per_day = int(raw.get("K_news_per_day", 5)),
        embedding_model = raw.get("embedding_model","text-embedding-3-small"),
        decision_model  = raw.get("decision_model","gpt-4o-mini"),
        memory_path = memory_path,
        retrieval = RetrievalCfg(**raw.get("retrieval", {})),
        risk = RiskCfg(**raw.get("risk", {})),
    )
    resolve_memory_path(cfg, prefer_existing=True)
    return cfg

def save_config(path: str, cfg: Config):
    resolve_memory_path(cfg, prefer_existing=False, ensure_parent=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

def get_alpaca_keys():
    key = os.environ.get("ALPACA_API_KEY_ID") or os.environ.get("ALPACA_KEY")
    secret = os.environ.get("ALPACA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET")
    return key, secret


def _normalize_path(path: str) -> str:
    expanded = os.path.expanduser(path or "")
    normalized = os.path.normpath(expanded)
    return normalized


def _symbol_slug(symbol: str) -> str:
    clean = (symbol or "default").strip().upper()
    return re.sub(r"[^A-Z0-9_-]+", "_", clean) or "DEFAULT"


def default_memory_path(symbol: str) -> str:
    slug = _symbol_slug(symbol)
    return os.path.join("data", "memory", f"{slug}_memory.json")


def _legacy_memory_candidates() -> Dict[str, str]:
    return {
        "legacy": "data/memory_bank.json",
    }


def _find_preferred_memory_path(symbol: str) -> str:
    default_path = default_memory_path(symbol)
    for path in [default_path, *_legacy_memory_candidates().values()]:
        if os.path.exists(_normalize_path(path)):
            return path
    return default_path


def resolve_memory_path(cfg: Config, prefer_existing: bool = True, ensure_parent: bool = False) -> str:
    symbol = getattr(cfg, "symbol", "AAPL")
    explicit = getattr(cfg, "memory_path", None)
    candidates = []
    if explicit:
        candidates.append(explicit)
    default_path = default_memory_path(symbol)
    if default_path not in candidates:
        candidates.append(default_path)
    for legacy in _legacy_memory_candidates().values():
        if legacy not in candidates:
            candidates.append(legacy)

    chosen = None
    if prefer_existing:
        for cand in candidates:
            norm = _normalize_path(cand)
            if os.path.exists(norm):
                chosen = cand
                break

    if chosen is None:
        chosen = candidates[0]

    normalized = _normalize_path(chosen)
    if ensure_parent:
        parent = os.path.dirname(normalized)
        if parent:
            os.makedirs(parent, exist_ok=True)

    cfg.memory_path = normalized
    return normalized
