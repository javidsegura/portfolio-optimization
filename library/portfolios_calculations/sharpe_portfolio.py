
from library.portfolios_calculations.base import BasePortfolio

import numpy as np

class SharpePortfolio(BasePortfolio):
      def __init__(self, MU, SIGMA, SIGMA_INV, ONE_VECTOR):
            super().__init__(MU, SIGMA, SIGMA_INV, ONE_VECTOR)
      
      def _compute_portfolio_weights(self):
            self.excess = self.MU - self.RISK_FREE_RATE * self.ONE_VECTOR
            unordered_weights = self.SIGMA_INV @ self.excess
            norm = float(self.ONE_VECTOR.T @ unordered_weights)
            self.W = unordered_weights / norm
            assert (round(self.W.sum(), 4) == 1), "The sum of the weights is not 1" # We need to round becuase of floating point precision issues in python
            return self.W

      def _compute_portfolio_return(self):
            """
            Computes the portfolio return
            """
            self.R = float(self.W.T @ self.MU)
            return self.R
      
      def _compute_portfolio_variance(self):
            """
            Computes the portfolio variance
            """
            self.VAR = float(self.W.T @ self.SIGMA @ self.W)
            return self.VAR
      
      def _compute_sharpe_ratio(self):
            """
            Computes the sharpe ratio
            """
            self.SHARPE = float((self.R - self.RISK_FREE_RATE) / np.sqrt(self.VAR))
            return self.SHARPE
      
      def compute_portfolio(self, risk_free_rate):
            """
            Computes the sharpe portfolio
            """
            self.RISK_FREE_RATE = risk_free_rate
            self._compute_portfolio_weights()
            self._compute_portfolio_return()
            self._compute_portfolio_variance()
            self._compute_sharpe_ratio()
            return {
                  "weights": self.W,
                  "expected_return": self.R,
                  "expected_variance": self.VAR,
                  "sharpe_ratio": self.SHARPE
            }