from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


START_DATE = "2000-01-01"
END_DATE = "2025-12-31"


@dataclass(frozen=True)
class SymbolMeta:
    symbol: str
    name: str
    kind: str
    sector: str = ""
    aliases: List[str] = field(default_factory=list)

    @property
    def yahoo_symbol(self) -> str:
        return self.symbol


STOCKS: List[SymbolMeta] = [
    SymbolMeta("AAPL", "Apple", "stock", "Technology", ["Apple Inc"]),
    SymbolMeta("MSFT", "Microsoft", "stock", "Technology", ["Microsoft Corporation"]),
    SymbolMeta("NVDA", "NVIDIA", "stock", "Technology", ["Nvidia"]),
    SymbolMeta("AMD", "Advanced Micro Devices", "stock", "Technology", ["AMD"]),
    SymbolMeta("INTC", "Intel", "stock", "Technology", ["Intel Corporation"]),
    SymbolMeta("AVGO", "Broadcom", "stock", "Technology", ["Broadcom Inc"]),
    SymbolMeta("QCOM", "Qualcomm", "stock", "Technology", ["QUALCOMM"]),
    SymbolMeta("ORCL", "Oracle", "stock", "Technology", ["Oracle Corporation"]),
    SymbolMeta("CSCO", "Cisco", "stock", "Technology", ["Cisco Systems"]),
    SymbolMeta("CRM", "Salesforce", "stock", "Technology", ["Salesforce.com"]),
    SymbolMeta("GOOGL", "Alphabet", "stock", "Communication Services", ["Google", "Alphabet Inc"]),
    SymbolMeta("META", "Meta Platforms", "stock", "Communication Services", ["Facebook", "Meta"]),
    SymbolMeta("NFLX", "Netflix", "stock", "Communication Services", ["Netflix Inc"]),
    SymbolMeta("DIS", "Disney", "stock", "Communication Services", ["Walt Disney"]),
    SymbolMeta("CMCSA", "Comcast", "stock", "Communication Services", ["Comcast Corporation"]),
    SymbolMeta("AMZN", "Amazon", "stock", "Consumer Discretionary", ["Amazon.com"]),
    SymbolMeta("TSLA", "Tesla", "stock", "Consumer Discretionary", ["Tesla Inc"]),
    SymbolMeta("HD", "Home Depot", "stock", "Consumer Discretionary", ["The Home Depot"]),
    SymbolMeta("MCD", "McDonald's", "stock", "Consumer Discretionary", ["McDonalds"]),
    SymbolMeta("NKE", "Nike", "stock", "Consumer Discretionary", ["NIKE"]),
    SymbolMeta("SBUX", "Starbucks", "stock", "Consumer Discretionary", ["Starbucks Corporation"]),
    SymbolMeta("WMT", "Walmart", "stock", "Consumer Staples", ["Wal-Mart"]),
    SymbolMeta("COST", "Costco", "stock", "Consumer Staples", ["Costco Wholesale"]),
    SymbolMeta("PG", "Procter & Gamble", "stock", "Consumer Staples", ["P&G"]),
    SymbolMeta("KO", "Coca-Cola", "stock", "Consumer Staples", ["The Coca-Cola Company"]),
    SymbolMeta("PEP", "PepsiCo", "stock", "Consumer Staples", ["Pepsi"]),
    SymbolMeta("JPM", "JPMorgan Chase", "stock", "Financials", ["JP Morgan"]),
    SymbolMeta("BAC", "Bank of America", "stock", "Financials", ["BofA"]),
    SymbolMeta("WFC", "Wells Fargo", "stock", "Financials", ["Wells Fargo & Company"]),
    SymbolMeta("GS", "Goldman Sachs", "stock", "Financials", ["Goldman"]),
    SymbolMeta("MS", "Morgan Stanley", "stock", "Financials", ["Morgan Stanley"]),
    SymbolMeta("V", "Visa", "stock", "Financials", ["Visa Inc"]),
    SymbolMeta("MA", "Mastercard", "stock", "Financials", ["MasterCard"]),
    SymbolMeta("BRK-B", "Berkshire Hathaway", "stock", "Financials", ["Berkshire", "Berkshire Hathaway B"]),
    SymbolMeta("UNH", "UnitedHealth", "stock", "Healthcare", ["UnitedHealth Group"]),
    SymbolMeta("JNJ", "Johnson & Johnson", "stock", "Healthcare", ["J&J"]),
    SymbolMeta("PFE", "Pfizer", "stock", "Healthcare", ["Pfizer Inc"]),
    SymbolMeta("MRK", "Merck", "stock", "Healthcare", ["Merck & Co"]),
    SymbolMeta("LLY", "Eli Lilly", "stock", "Healthcare", ["Eli Lilly and Company"]),
    SymbolMeta("ABBV", "AbbVie", "stock", "Healthcare", ["AbbVie Inc"]),
    SymbolMeta("XOM", "Exxon Mobil", "stock", "Energy", ["ExxonMobil"]),
    SymbolMeta("CVX", "Chevron", "stock", "Energy", ["Chevron Corporation"]),
    SymbolMeta("COP", "ConocoPhillips", "stock", "Energy", ["Conoco Phillips"]),
    SymbolMeta("CAT", "Caterpillar", "stock", "Industrials", ["Caterpillar Inc"]),
    SymbolMeta("BA", "Boeing", "stock", "Industrials", ["The Boeing Company"]),
    SymbolMeta("HON", "Honeywell", "stock", "Industrials", ["Honeywell International"]),
    SymbolMeta("LIN", "Linde", "stock", "Materials", ["Linde plc"]),
    SymbolMeta("NEM", "Newmont", "stock", "Materials", ["Newmont Corporation"]),
    SymbolMeta("NEE", "NextEra Energy", "stock", "Utilities", ["NextEra"]),
    SymbolMeta("SO", "Southern Company", "stock", "Utilities", ["The Southern Company"]),
]

CONTEXT: List[SymbolMeta] = [
    SymbolMeta("SPY", "SPDR S&P 500 ETF", "context", "Index ETF", ["S&P 500 ETF"]),
    SymbolMeta("QQQ", "Invesco QQQ Trust", "context", "Index ETF", ["Nasdaq 100 ETF"]),
    SymbolMeta("IWM", "iShares Russell 2000 ETF", "context", "Index ETF", ["Russell 2000 ETF"]),
    SymbolMeta("DIA", "SPDR Dow Jones Industrial Average ETF", "context", "Index ETF", ["Dow ETF"]),
    SymbolMeta("XLK", "Technology Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLF", "Financial Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLV", "Health Care Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLE", "Energy Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLY", "Consumer Discretionary Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLP", "Consumer Staples Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLI", "Industrial Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLB", "Materials Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("XLU", "Utilities Select Sector SPDR", "context", "Sector ETF"),
    SymbolMeta("^GSPC", "S&P 500 Index", "context", "Index", ["S&P 500"]),
    SymbolMeta("^IXIC", "Nasdaq Composite", "context", "Index", ["NASDAQ Composite"]),
    SymbolMeta("^DJI", "Dow Jones Industrial Average", "context", "Index", ["DJIA"]),
    SymbolMeta("^RUT", "Russell 2000 Index", "context", "Index", ["Russell 2000"]),
    SymbolMeta("^VIX", "CBOE Volatility Index", "context", "Volatility", ["VIX"]),
    SymbolMeta("^TNX", "CBOE 10-Year Treasury Yield", "context", "Rates", ["10-year yield"]),
]

STOCK_SYMBOLS = [item.symbol for item in STOCKS]
CONTEXT_SYMBOLS = [item.symbol for item in CONTEXT]


def all_symbols() -> List[SymbolMeta]:
    return [*STOCKS, *CONTEXT]


def select_symbols(symbols: Iterable[str] | None = None, *, include_context: bool = True) -> List[SymbolMeta]:
    if symbols is None:
        return all_symbols() if include_context else list(STOCKS)
    wanted = {symbol.upper() for symbol in symbols}
    return [item for item in all_symbols() if item.symbol.upper() in wanted]
