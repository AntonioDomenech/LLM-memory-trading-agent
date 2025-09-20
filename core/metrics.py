
import numpy as np, pandas as pd
def compute_metrics(equity: pd.Series, bh_equity: pd.Series):
    equity = equity.dropna(); bh_equity = bh_equity.reindex_like(equity).ffill().dropna()
    if len(equity) < 2: return {}
    daily = equity.pct_change().fillna(0.0); bh_daily = bh_equity.pct_change().fillna(0.0)
    def sharpe(x): return 0.0 if x.std()==0 else (x.mean()/x.std())*np.sqrt(252)
    def sortino(x):
        downside = x[x<0].std()
        return 0.0 if downside is None or downside==0 else (x.mean()/downside)*np.sqrt(252)
    cagr = (equity.iloc[-1]/equity.iloc[0])**(252/len(equity)) - 1
    dd = (equity / equity.cummax() - 1.0).min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe(daily)), "Sortino": float(sortino(daily)),
            "MaxDrawdown": float(dd), "Volatilidad": float(daily.std()*np.sqrt(252)),
            "BH_CAGR": float((bh_equity.iloc[-1]/bh_equity.iloc[0])**(252/len(bh_equity)) - 1),
            "ActiveReturn": float((equity.iloc[-1]/equity.iloc[0]) - (bh_equity.iloc[-1]/bh_equity.iloc[0]))}
