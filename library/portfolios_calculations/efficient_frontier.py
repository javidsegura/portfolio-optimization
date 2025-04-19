

import numpy as np
import pandas as pd

from library.portfolios_calculations.base import BasePortfolio


class EfficientFrontierPortfolio(BasePortfolio):
      def __init__(self, MU, SIGMA, SIGMA_INV, ONE_VECTOR):
            super().__init__(MU, SIGMA, SIGMA_INV, ONE_VECTOR)
      
      def _compute_components(self):
            """
            Computes the components of the efficient frontier
            """
            self.A = self.ONE_VECTOR.T @ self.SIGMA_INV @  self.ONE_VECTOR
            self.B = self.ONE_VECTOR.T @ self.SIGMA_INV @ self.MU
            self.C = self.MU.T @ self.SIGMA_INV @ self.MU
            self.D = self.A * self.C - self.B**2
      
      def _compute_portfolio_weights(self, **kwargs):
            """
            Computes the portfolio weights
            """
            Rp = kwargs.get("Rp", None)
            assert Rp is not None, "Rp must be provided"
            self.W =  self.SIGMA_INV @ ((((self.A*Rp) - self.B)*self.MU + (self.C-self.B*Rp)*self.ONE_VECTOR)/self.D)
            assert (round(self.W.sum(), 4) == 1), "The sum of the weights is not 1" # We need to round becuase of floating point precision issues in python
            return self.W

      def _compute_portfolio_variance(self, **kwargs):
            """
            Computes the portfolio variance
            """
            Rp = kwargs.get("Rp", None)
            assert Rp is not None, "Rp must be provided"
            self.VAR = float((self.A * Rp**2 - 2*self.B*Rp + self.C) / self.D)
            return self.VAR
      

      def compute_portfolio(self):
            """
            Returns corresponding weights and variance for the effiuceint protfolios
            """
            self._compute_components()
            min_u = self.MU.min()
            max_u = self.MU.max()
            print(f"Min u: {min_u}, Max u: {max_u}")
            u_range = np.linspace(min_u, max_u, 100)
            weights = []
            variances = []
            sigmas = []
            expected_returns = []
            for u in u_range:
                  weights.append(self._compute_portfolio_weights(Rp=u))
                  var = self._compute_portfolio_variance(Rp=u)
                  variances.append(var)
                  sigmas.append(np.sqrt(var))
                  expected_returns.append(u)
            efficient_frontier =  pd.DataFrame({"weights": weights, "variance": variances, "sigma":sigmas,"expected_return": expected_returns})
            self.efficient_frontier = efficient_frontier
            return efficient_frontier
      
      
            
            