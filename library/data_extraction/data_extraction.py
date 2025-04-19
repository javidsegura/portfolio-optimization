
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
            try:
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
            except Exception as e:
                  print(f"Error downloading data for {ticker}: {e}")
                  return None, None
      

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
                  if sr is None:
                        continue
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

      def pca_shrinkage(self, n_components=None, var_threshold=None):
            """
            Perform PCA-based spectral filtering on the covariance matrix by shrinking small eigenvalues.

            Mathematical Basis
            ------------------
            Given the covariance matrix Σ, its eigendecomposition is:
                  Σ = Q Λ Qᵀ,
            where Q = [q₁, …, qₙ] are the orthonormal eigenvectors and Λ = diag(λ₁, …, λₙ) are eigenvalues,
            sorted so that λ₁ ≥ λ₂ ≥ … ≥ λₙ ≥ 0.

            To filter noise, we keep the top k eigenvalues corresponding to dominant risk factors,
            and shrink the remaining (tail) eigenvalues to their average:
                  λᵢ' = λᵢ for i = 1…k,
                  λⱼ' = (1/(n-k)) * Σ_{j=k+1}ⁿ λⱼ  for j = k+1…n.

            The filtered covariance is reconstructed as:
                  Σ_filtered = Q Λ' Qᵀ,
            which preserves main variance directions while damping noisy dimensions.

            Parameters
            ----------
            Sigma : np.ndarray, shape (n_assets, n_assets)
                  Sample covariance matrix.
            n_components : int, optional
                  Number of top eigenvalues to keep (k).
            var_threshold : float, optional
                  Fraction of total variance to preserve (0 < var_threshold <= 1).
                  If provided, determines k s.t. cumulative variance ≥ threshold.

            Returns
            -------
            Sigma_filtered : np.ndarray
                  Covariance matrix reconstructed from filtered eigenvalues.

            Notes
            -----
            - If both n_components and var_threshold are None, raises ValueError.
            - This reduces the condition number of Σ, improving numerical stability of Σ⁻¹.
            """
            # Eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(self.SIGMA)
            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals_sorted = eigvals[idx]
            eigvecs_sorted = eigvecs[:, idx]

            # Determine number of components k
            if var_threshold is not None:
                  total_var = eigvals_sorted.sum()
                  frac = np.cumsum(eigvals_sorted) / total_var
                  k = np.searchsorted(frac, var_threshold) + 1
            elif n_components is not None:
                  k = n_components
            else:
                  raise ValueError("Specify n_components or var_threshold for spectral shrinkage.")

            # Shrink tail eigenvalues to their average
            if k < len(eigvals_sorted):
                  avg_tail = eigvals_sorted[k:].mean()
                  new_eigvals = np.concatenate([
                        eigvals_sorted[:k],
                        np.full(len(eigvals_sorted) - k, avg_tail)
                  ])
            else:
                  new_eigvals = eigvals_sorted.copy()

            # Reconstruct filtered covariance
            Sigma_filtered = eigvecs_sorted @ np.diag(new_eigvals) @ eigvecs_sorted.T
            self.SIGMA = Sigma_filtered
            self.SIGMA_INV = np.linalg.inv(Sigma_filtered)
            return self.SIGMA, self.SIGMA_INV
 
      
      def analayze_single_security_returns(self):
            for security in self.securities:
                  print(f"Security: {security}")
                  print(f"Mean expected return: {self.securities[security]['mean']}")
                  print(f"Risk: {np.sqrt(self.securities[security]['variance'])}")
                  print("\n")

      def compute_matrices(self):
            self._compute_mean_vector()
            self._compute_sigma_matrix()
            
            


                  
