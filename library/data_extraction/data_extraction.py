
import yfinance as yf
import pandas as pd
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataExtractor:
      def __init__(self):
            self.list_of_tickers = None
            self.start_date = None
            self.end_date = None
            self.MU = None
            self.SIGMA = None
            self.SIGMA_INV = None
            self.ONE_VECTOR = None
            
      def _get_return_series(self, ticker, start="2020-01-01", end=None):
            """
            Download adjusted price data for a single ticker and return its daily return Series.
            """
            print(f"=> Downloading data for {ticker} from {start} to {end}")
            df = yf.download(ticker, start=start, end=end, auto_adjust=False) 
            if df.empty:
                  raise ValueError(f"No data for {ticker} in {start}–{end}")
            prices = df["Adj Close"] 
            metrics = prices.describe()
            ret = prices.pct_change().dropna()
            ret.name = ticker
            ret.reset_index(inplace=True)
            ret.rename(columns={ticker: "Adj_Close_Change_(%)"}, inplace=True)
            return ret, metrics
      

      def build_securities(self):
            """
            Returns a dict mapping ticker → dict with keys:
                  - "returns": pd.Series of daily returns
                  - "mean":    float mean(return)
                  - "std":     float std(return)
            """
            R = {}
            for t in self.list_of_tickers:
                  sr, metrics = self._get_return_series(t, self.start_date, self.end_date)
                  R[t] = {
                        "returns": sr,
                        "mean": sr["Adj_Close_Change_(%)"].mean(),
                        "variance": sr["Adj_Close_Change_(%)"].var(ddof=0),
                        "metrics": metrics
                  }
            self.securities = R
            return R
      
      def _compute_mean_vector(self):
            """
            Computes the mean vector for the whole portfolio
            """
            MU = pd.Series({t: info["mean"] for t, info in self.securities.items()})
            MU = MU.values.reshape(len(MU), 1)
            self.ONE_VECTOR = np.ones((MU.shape[0], 1))
            self.MU = MU
            return MU
      
      def _compute_sigma_matrix(self):
            """
            Build the variance–covariance matrix Σ (for the whole portfolio)
            """

            returns_df = pd.concat(
            [
                  info["returns"]
                        .set_index("Date")["Adj_Close_Change_(%)"]
                        .rename(ticker)
                  for ticker, info in self.securities.items()
            ],
            axis=1
            )

            SIGMA = returns_df.cov()
            SIGMA = SIGMA.to_numpy()
            self.SIGMA = SIGMA
            self.SIGMA_INV = np.linalg.inv(SIGMA)
            return SIGMA
      
      def analayze_single_security_returns(self):
            for security in self.securities:
                  print(f"Security: {security}")
                  print(f"Mean expected return: {self.securities[security]['mean']}")
                  print(f"Risk: {np.sqrt(self.securities[security]['variance'])}")
                  print("\n")

      def compute_matrices(self):
            self._compute_mean_vector()
            self._compute_sigma_matrix()
            
            


                  
