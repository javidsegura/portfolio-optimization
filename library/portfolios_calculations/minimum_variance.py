
import numpy as np
from library.portfolios_calculations.base import BasePortfolio

class GlobalMinimumVariancePortfolio(BasePortfolio):
      def __init__(self, MU, SIGMA, SIGMA_INV, ONE_VECTOR):
            super().__init__(MU, SIGMA, SIGMA_INV, ONE_VECTOR)
      
      def _compute_portfolio_weights(self):
            """
            Computes the portfolio weights
            """
            self.W =  (self.SIGMA_INV @ self.ONE_VECTOR) / (self.ONE_VECTOR.T @ self.SIGMA_INV @ self.ONE_VECTOR)
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
      

      def compute_portfolio(self):
            """
            Computes the global minimum variance portfolio
            """
            self._compute_portfolio_weights()
            self._compute_portfolio_return()
            self._compute_portfolio_variance()
            return {
                  "weights": self.W,
                  "expected_return": self.R,
                  "expected_variance": self.VAR
            }
            
            