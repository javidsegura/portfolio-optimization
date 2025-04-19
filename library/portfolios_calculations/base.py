

from abc import ABC, abstractmethod

class BasePortfolio(ABC):
      def __init__(self, MU, SIGMA, SIGMA_INV, ONE_VECTOR):
            self.MU = MU
            self.SIGMA = SIGMA
            self.SIGMA_INV = SIGMA_INV
            self.ONE_VECTOR = ONE_VECTOR

      @abstractmethod
      def _compute_portfolio_weights(self, **kwargs):
            pass
      
      @abstractmethod
      def _compute_portfolio_variance(self, **kwargs):
            pass

      @abstractmethod
      def compute_portfolio(self):
            pass