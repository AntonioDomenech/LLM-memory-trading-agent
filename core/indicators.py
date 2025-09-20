
import pandas as pd
import ta
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["high","low","close"])
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_hist"] = macd.macd_diff()
    atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()
    df["sma200"] = df["close"].rolling(200).mean()
    df["sma100"] = df["close"].rolling(100).mean()
    df["trend_up"] = (df["close"] > df["sma200"]).astype(int)
    return df
