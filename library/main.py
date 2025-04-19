import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from library.data_extraction.data_extraction import DataExtractor
from library.portfolios_calculations.minimum_variance import GlobalMinimumVariancePortfolio
from library.portfolios_calculations.efficient_frontier import EfficientFrontierPortfolio
from library.portfolios_calculations.sharpe_portfolio import SharpePortfolio


from library.plots.plots import Plots


class Portfolio:
      def __init__(self):
            self.data_extractor = DataExtractor()
            self.plots = Plots()
            self.global_minimum_variance_portfolio = GlobalMinimumVariancePortfolio(self.data_extractor.MU, self.data_extractor.SIGMA, self.data_extractor.SIGMA_INV, self.data_extractor.ONE_VECTOR)
            self.efficient_frontier_portfolio = EfficientFrontierPortfolio(self.data_extractor.MU, self.data_extractor.SIGMA, self.data_extractor.SIGMA_INV, self.data_extractor.ONE_VECTOR)
            self.sharpe_portfolio = SharpePortfolio(self.data_extractor.MU, self.data_extractor.SIGMA, self.data_extractor.SIGMA_INV, self.data_extractor.ONE_VECTOR)
      
      def extract_tickers(self):
            wiki = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            return wiki[0]
      
      def update_tickers(self, list_of_tickers: list, start_date: str, end_date: str):
            """
            Updates the tickers
            """
            self.data_extractor.list_of_tickers = list_of_tickers
            self.data_extractor.start_date = start_date
            self.data_extractor.end_date = end_date
            self.securities = self.data_extractor.build_securities()
            self.plots.securities = self.securities
      
      def analyze_securities(self):
            """
            Analyzes the securities
            """
            self.plots.plot_securities_distribution()


      def update_data(self):
            portfolios = [self.global_minimum_variance_portfolio, self.efficient_frontier_portfolio, self.sharpe_portfolio]
            for portfolio in portfolios:
                  portfolio.MU = self.data_extractor.MU
                  portfolio.SIGMA = self.data_extractor.SIGMA
                  portfolio.SIGMA_INV = self.data_extractor.SIGMA_INV
                  portfolio.ONE_VECTOR = self.data_extractor.ONE_VECTOR
      
      def compute_global_minimum_variance_portfolio(self):
            gmvp = self.global_minimum_variance_portfolio.compute_portfolio()
            self.plots.gmvp = {
                  "expected_return": gmvp["expected_return"],
                  "expected_variance": gmvp["expected_variance"]
            }
            return gmvp
      
      def compute_efficient_frontier(self):
            efficient_frontier = self.efficient_frontier_portfolio.compute_portfolio()
            self.plots.efficient_frontier = efficient_frontier
            return efficient_frontier
      
      def compute_sharpe_portfolio(self, risk_free_rate):

            sharpe_portfolio = self.sharpe_portfolio.compute_portfolio(risk_free_rate)
            self.plots.msr = {
                  "expected_return": sharpe_portfolio["expected_return"],
                  "expected_variance": sharpe_portfolio["expected_variance"],
                  "sharpe_ratio": sharpe_portfolio["sharpe_ratio"]
            }
            self.plots.risk_free_rate = risk_free_rate
            return sharpe_portfolio




