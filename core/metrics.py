
import numpy as np, pandas as pd
from typing import Optional


def compute_metrics(equity: pd.Series, bh_equity: pd.Series, contributions: Optional[pd.Series] = None):
    equity = equity.dropna(); bh_equity = bh_equity.reindex_like(equity).ffill().dropna()
    if contributions is None:
        contributions = pd.Series(0.0, index=equity.index)
    else:
        contributions = contributions.reindex_like(equity).fillna(0.0)
    contributions_cum = contributions.cumsum()
    equity_adj = (equity - contributions_cum).clip(lower=1e-6)
    bh_adj = (bh_equity - contributions_cum).clip(lower=1e-6)
    if len(equity_adj) < 2: return {}
    daily = equity_adj.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0); bh_daily = bh_adj.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    def sharpe(x): return 0.0 if x.std()==0 else (x.mean()/x.std())*np.sqrt(252)
    def sortino(x):
        downside = x[x<0].std()
        return 0.0 if downside is None or downside==0 else (x.mean()/downside)*np.sqrt(252)
    cagr = (equity_adj.iloc[-1]/equity_adj.iloc[0])**(252/len(equity_adj)) - 1
    dd = (equity_adj / equity_adj.cummax() - 1.0).min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe(daily)), "Sortino": float(sortino(daily)),
            "MaxDrawdown": float(dd), "Volatilidad": float(daily.std()*np.sqrt(252)),
            "BH_CAGR": float((bh_adj.iloc[-1]/bh_adj.iloc[0])**(252/len(bh_adj)) - 1),
            "ActiveReturn": float((equity_adj.iloc[-1]/equity_adj.iloc[0]) - (bh_adj.iloc[-1]/bh_adj.iloc[0]))}
