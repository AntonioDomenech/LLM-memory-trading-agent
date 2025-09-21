import numpy as np
import pandas as pd
from typing import Optional


def compute_metrics(
    equity: pd.Series,
    bh_equity: pd.Series,
    contributions: Optional[pd.Series] = None,
) -> dict:
    """Compute risk/return analytics for the strategy versus buy & hold."""

    equity = equity.dropna()
    bh_equity = bh_equity.reindex_like(equity).ffill().dropna()
    if contributions is None:
        contributions = pd.Series(0.0, index=equity.index)
    else:
        contributions = contributions.reindex_like(equity).fillna(0.0)

    contributions_cum = contributions.cumsum()
    equity_adj = (equity - contributions_cum).clip(lower=1e-6)
    bh_adj = (bh_equity - contributions_cum).clip(lower=1e-6)
    if len(equity_adj) < 2:
        return {}

    daily = equity_adj.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    bh_daily = bh_adj.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def sharpe(series: pd.Series) -> float:
        """Return the annualised Sharpe ratio for ``series``."""

        std = series.std()
        if std == 0:
            return 0.0
        return float((series.mean() / std) * np.sqrt(252))

    def sortino(series: pd.Series) -> float:
        """Return the annualised Sortino ratio for ``series``."""

        downside = series[series < 0].std()
        if downside in (None, 0):
            return 0.0
        return float((series.mean() / downside) * np.sqrt(252))

    cagr = (equity_adj.iloc[-1] / equity_adj.iloc[0]) ** (252 / len(equity_adj)) - 1
    dd = (equity_adj / equity_adj.cummax() - 1.0).min()

    return {
        "CAGR": float(cagr),
        "Sharpe": sharpe(daily),
        "Sortino": sortino(daily),
        "MaxDrawdown": float(dd),
        "Volatilidad": float(daily.std() * np.sqrt(252)),
        "BH_CAGR": float((bh_adj.iloc[-1] / bh_adj.iloc[0]) ** (252 / len(bh_adj)) - 1),
        "ActiveReturn": float(
            (equity_adj.iloc[-1] / equity_adj.iloc[0]) - (bh_adj.iloc[-1] / bh_adj.iloc[0])
        ),
    }
