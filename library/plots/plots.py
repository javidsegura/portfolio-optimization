import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class Plots:
      def __init__(self):
            self.gmvp = {
                  "expected_return": None,
                  "expected_variance": None
            }
            self.efficient_frontier = None
            self.msr = {
                  "expected_return": None,
                  "expected_variance": None,
                  "sharpe_ratio": None
            }
            self.risk_free_rate = None
            self.securities = None
      
      def _plot_gmvp(self, ax):
            """
            Plots the global minimum variance portfolio
            """
            expected_variance_mgvp = float(self.gmvp["expected_variance"])
            expected_return_mgvp = float(self.gmvp["expected_return"])
            ax.scatter(
                  np.sqrt(self.gmvp["expected_variance"]),
                  self.gmvp["expected_return"],
                  color='purple',
                  label='Global Minimum Variance Portfolio',
                  zorder=5
                  )
            ax.annotate(
                  'GMVP',
                  (np.sqrt(expected_variance_mgvp), expected_return_mgvp),
                  textcoords="offset points",
                  xytext=(-15,10),
                  ha='center',
                  color='purple'
                  )
      
      def _plot_efficient_frontier(self, ax):
            """
            Plots the efficient frontier
            """
            ax.plot(
                  (self.efficient_frontier['sigma']),  # sqrt to convert to std dev if you want
                  self.efficient_frontier['expected_return'],
                  label='Efficient Frontier',
                  color='blue',
                  lw=2
                  )
      
      def _plot_sharpe(self, ax):
            """
            Plots the sharpe portfolio
            """
            std_range = np.linspace(0, np.sqrt(self.msr["expected_variance"]) * 1.5, 100)
            cml_returns = self.risk_free_rate + (self.msr["expected_return"] - self.risk_free_rate) * std_range / np.sqrt(self.msr["expected_variance"])
            # Captial market line
            ax.plot(
                  std_range,
                  cml_returns,
                  label='CML / CAL',
                  color='lightblue',
                  lw=2,
                  linestyle='--'
                  )
            # Market/Tangency portfolio
            ax.scatter(
                  np.sqrt(self.msr["expected_variance"]),
                  self.msr["expected_return"],
                  color='green',
                  label='Tangency/Market/Maximum Sharpe Ratio Portfolio',
                  zorder=5
                  )
            ax.annotate(
                  f'MSR',
                  (np.sqrt(self.msr["expected_variance"]), self.msr["expected_return"]),
                  textcoords="offset points",
                  xytext=(-15,10),
                  ha='center',
                  color='green'
                  )
            # Risk free rate
            ax.scatter(
                  0,
                  self.risk_free_rate,
                  color='yellow',
                  label='Risk Free Rate',
                  zorder=5
                  )
      def _plot_individual_securities(self, ax):
            """
            Plots the individual securities with annotations
            """
            for security_name, security_data in self.securities.items():
                  # Extract mean (expected return) and std (risk) for each security
                  mean = security_data['mean']
                  std = np.sqrt(security_data['variance'])  # convert variance to std dev
                  
                  # Plot the point
                  ax.scatter(std, mean, color='red')
                  
                  # Add annotation with security name
                  # Slightly offset the text to avoid overlapping with the point
                  ax.annotate(security_name, 
                             (std, mean),
                             xytext=(5, 5),  # 5 points offset
                             textcoords='offset points',
                             fontsize=10)

      def plot_results(self, 
                       include_gmvp:bool=False, 
                       include_efficient_frontier:bool=False, 
                       include_sharpe:bool=False, 
                       include_individual_securities:bool=False
                       ):
            """
            Plots the final results
            """
            fig, ax = plt.subplots(figsize=(10, 6))
            if include_gmvp:
                  self._plot_gmvp(ax)
            if include_efficient_frontier:
                  self._plot_efficient_frontier(ax)
            if include_sharpe:
                  self._plot_sharpe(ax)
            if include_individual_securities:
                  self._plot_individual_securities(ax)



            ax.legend()
            ax.grid(True)
            ax.set_xlabel('Portfolio Standard Deviation')
            ax.set_ylabel('Expected Return')
            ax.set_title('Portfolio Optimization')
            plt.show()
      


      def plot_securities_distribution(self):
            """
            Plots for each security:
                  • Left:  time‑series line of daily returns
                  • Right: histogram (with KDE) of daily returns
            """
            n = len(self.securities)
            fig, axes = plt.subplots(
                  nrows=n,
                  ncols=3,
                  figsize=(20, n * 3),
                  sharex=False
            )
            fig.subplots_adjust(hspace=0.6)   

            if n == 1:
                  axes = axes.reshape(1, 2)

            for i, (security, info) in enumerate(self.securities.items()):
                  df = info["returns"].copy()
                  df["Date"] = pd.to_datetime(df["Date"])
                  df.sort_values("Date", inplace=True)

                  ax_trend, ax_hist, ax_metrics = axes[i, 0], axes[i, 1], axes[i, 2]

                  # Plot 1: true trend line of returns
                  ax_trend.plot(
                        df["Date"],
                        df["Adj_Close_Change_(%)"],
                        label=security,
                        linewidth=1.5
                  )
                  ax_trend.set_title(f"Trend of {security} Daily Returns")
                  ax_trend.set_xlabel("Date")
                  ax_trend.set_ylabel("Return (%)")
                  ax_trend.legend()
                  ax_trend.grid(True)

                  # Plot 2: histogram +
                  sns.histplot(
                        df["Adj_Close_Change_(%)"],
                        kde=True,
                        bins=30,
                        stat="density",
                        ax=ax_hist
                  )
                  ax_hist.set_title(f"Distribution of {security} Daily Returns")
                  ax_hist.set_xlabel("Return (%)")
                  ax_hist.set_ylabel("Density")
                  ax_hist.grid(True)

                  # Plot 3: metrics
                  mean = float(info['metrics'].loc["mean"])
                  std = float(info['metrics'].loc["std"])
                  min = float(info['metrics'].loc["min"])
                  max = float(info['metrics'].loc["max"])
                  first_quartile = float(info['metrics'].loc["25%"])
                  third_quartile = float(info['metrics'].loc["75%"])

                  # Turn off axes
                  ax_metrics.axis('off')

                  # Center the text by using the middle of the plot (0.5, 0.5) and center alignment
                  ax_metrics.text(0.5, 0.5, 
                        f"Mean: {mean:.2f}\nStd: {std:.2f}\nMin: {min:.2f}\nMax: {max:.2f}\n1st Quartile: {first_quartile:.2f}\n3rd Quartile: {third_quartile:.2f}", 
                        ha='center', 
                        va='center',
                        transform=ax_metrics.transAxes  # This makes coordinates relative to the axes (0-1 range)
                  )
                  ax_metrics.set_title(f"Metrics of {security}")

            plt.tight_layout()
            plt.show()
