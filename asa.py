
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import jarque_bera

# 1) Fetch S&P500 tickers from Wikipedia
wiki = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
sp500 = wiki[0]["Symbol"].tolist()

# 2) Download daily adjusted close prices
start_date = "2022-01-01"
end_date   = "2025-04-18"
px = yf.download(sp500, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]

# 3) Compute daily returns
rets = px.pct_change().dropna(how="all")

# 4) Run JB on each series, collect p‑values
jb_results = []
for ticker in rets.columns:
    series = rets[ticker].dropna()
    if len(series) < 50:
        continue  # skip too‑short series
    stat, p = jarque_bera(series)
    if p > 0.05:
        print(f"{ticker} is normal")
    jb_results.append((ticker, stat, p))

# 5) Filter for “normal” (p > 0.05) and sort by highest p‑value
jb_df = pd.DataFrame(jb_results, columns=["Ticker","JB_stat","p_value"])
normal_df = jb_df[jb_df["p_value"] > 0.05].sort_values("p_value", ascending=False)

# 6) Show the top two tickers
print("Top 2 S&P 500 assets whose returns do NOT reject normality (p > 0.05):")
print(normal_df.head(2).to_string(index=False))
