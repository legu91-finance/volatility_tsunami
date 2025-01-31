# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:11:56 2023

@author: legu9
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

class VolatilityTsunamiAnalyzer:
    def __init__(self, start_date, end_date, 
                 std_window=20,
                 std_percentile_threshold=15,
                 spike_threshold=None):
        self.start_date = start_date
        self.end_date = end_date
        self.std_window = std_window
        self.std_percentile_threshold = std_percentile_threshold
        self.spike_threshold = spike_threshold
        self.tickers = "^VVIX ^VIX ^GSPC ^IRX ^TNX"
        
    def fetch_data(self):
        """Fetch data from Yahoo Finance"""
        # Download all tickers data
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='ticker')
        
        # Create a new DataFrame to store the closing prices
        prices = pd.DataFrame()
        
        # Extract the closing prices for each ticker
        for ticker in self.tickers.split():
            price_column = 'Adj Close' if f'{ticker} Adj Close' in data.columns else 'Close'
            prices[ticker] = data[ticker][price_column]
        
        # Rename columns to be more descriptive
        prices.columns = ["VVIX", "VIX", "SPX", "13w_yield", "10year_yield"]
        
        return prices
    
    def calculate_metrics(self, data):
        """Calculate metrics as per the paper's methodology"""
        df = data.copy()
        
        # Core volatility dispersion metrics
        df['VIX Std'] = df['VIX'].rolling(window=self.std_window).std()
        df['VVIX Std'] = df['VVIX'].rolling(window=self.std_window).std()
        df['yield curve spread'] = df["10year_yield"] - df["13w_yield"]
        
        # Calculate percentile ranks
        df['VIX_std_percentile'] = df['VIX Std'].rank(pct=True) * 100
        df['VVIX_std_percentile'] = df['VVIX Std'].rank(pct=True) * 100
        
        # Signal generation
        df['low_dispersion_signal'] = (
            (df['VIX_std_percentile'] < self.std_percentile_threshold) & 
            (df['VVIX_std_percentile'] < self.std_percentile_threshold)
        )
        
        # Calculate forward returns
        df['VIX_fwd_5d_return'] = df['VIX'].pct_change(periods=5).shift(-5)
        df['VIX_fwd_10d_return'] = df['VIX'].pct_change(periods=10).shift(-10)
        df['VIX_fwd_20d_return'] = df['VIX'].pct_change(periods=20).shift(-20)
        
        return df
    
    def calculate_rolling_lows(self, data):
        # VIX rolling lows
        rolling_low_vix = data['VIX'].rolling(window="28D").min()
        is_rolling_low_vix = (data['VIX'] == rolling_low_vix)
        no_rolling_low_vix_last_4_weeks_mask = (is_rolling_low_vix) & ~(is_rolling_low_vix.rolling(window=20, min_periods=1).sum().shift(1).fillna(0).astype(bool))
        rolling_low_mask_vix = (data['VIX'] == rolling_low_vix) & no_rolling_low_vix_last_4_weeks_mask
        
        # VVIX rolling lows
        rolling_low_vvix = data['VVIX'].rolling(window="28D").min()
        is_rolling_low_vvix = (data['VVIX'] == rolling_low_vvix)
        no_rolling_low_vvix_last_4_weeks_mask = (is_rolling_low_vvix) & ~(is_rolling_low_vvix.rolling(window=20, min_periods=1).sum().shift(1).fillna(0).astype(bool))
        rolling_low_mask_vvix = (data['VVIX'] == rolling_low_vvix) & no_rolling_low_vvix_last_4_weeks_mask
        
        return rolling_low_vix, rolling_low_mask_vix, rolling_low_vvix, rolling_low_mask_vvix
    
    def analyze_signals(self, df):
        """Analyze signal effectiveness"""
        signal_stats = {
            '5d': {
                'mean_return': df[df['low_dispersion_signal']]['VIX_fwd_5d_return'].mean(),
                'median_return': df[df['low_dispersion_signal']]['VIX_fwd_5d_return'].median(),
                'positive_signals': (df[df['low_dispersion_signal']]['VIX_fwd_5d_return'] > 0).mean()
            },
            '10d': {
                'mean_return': df[df['low_dispersion_signal']]['VIX_fwd_10d_return'].mean(),
                'median_return': df[df['low_dispersion_signal']]['VIX_fwd_10d_return'].median(),
                'positive_signals': (df[df['low_dispersion_signal']]['VIX_fwd_10d_return'] > 0).mean()
            },
            '20d': {
                'mean_return': df[df['low_dispersion_signal']]['VIX_fwd_20d_return'].mean(),
                'median_return': df[df['low_dispersion_signal']]['VIX_fwd_20d_return'].median(),
                'positive_signals': (df[df['low_dispersion_signal']]['VIX_fwd_20d_return'] > 0).mean()
            }
        }
        return signal_stats
    
    def create_plots(self, data):
        """Create and return matplotlib plots"""
        # Set modern style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: S&P 500 Close Price with signals
        ax1.plot(data.index, data['SPX'], label='S&P 500 close', color='#2E86C1', linewidth=1.5)
        # Add signal dots
        signal_dates = data[data['low_dispersion_signal'] == True].index
        ax1.scatter(signal_dates, data.loc[signal_dates, 'SPX'], 
                   color='red', s=50, label='Low Dispersion Signal (15th percentile)')
        ax1.set_title('S&P 500 Close Price', fontsize=12, pad=10)
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: VIX and VVIX with std markers
        ax2.plot(data.index, data['VIX'], label='VIX', color='#E67E22', linewidth=1.5)
        ax2.plot(data.index, data['VVIX'], label='VVIX', color='#27AE60', linewidth=1.5)
        # Add VIX std markers
        vix_std_low = data[data['VIX_std_percentile'] < 67].index
        vvix_std_low = data[data['VVIX_std_percentile'] < 3.46].index
        ax2.scatter(vix_std_low, data.loc[vix_std_low, 'VIX'], 
                   color='red', s=50, label='VIX std below 0.67')
        ax2.scatter(vvix_std_low, data.loc[vvix_std_low, 'VVIX'], 
                   color='blue', s=50, label='VVIX std below 3.46')
        ax2.set_title('VIX and VVIX', fontsize=12, pad=10)
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Yield Spread
        spread = data['10year_yield'] - data['13w_yield']
        ax3.plot(data.index, spread, label='10y-13w Spread', 
                 color='#C0392B', linewidth=1.5)
        ax3.set_title('10-Year minus 13-Week Yield Spread', fontsize=12, pad=10)
        ax3.legend(frameon=True)
        ax3.grid(True, alpha=0.3)
        
        # Style improvements
        for ax in [ax1, ax2, ax3]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=10)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        return fig

    def backtest_parameters(self, data, 
                           std_windows_range=(5, 50, 5),  # (start, end, step)
                           percentile_thresholds_range=(5, 30, 5)):
        """
        Perform grid search to find optimal parameters
        Returns DataFrame with results for each parameter combination
        """
        results = []
        
        # Create parameter grid
        std_windows = range(std_windows_range[0], std_windows_range[1], std_windows_range[2])
        percentiles = range(percentile_thresholds_range[0], percentile_thresholds_range[1], percentile_thresholds_range[2])
        
        for window in std_windows:
            for percentile in percentiles:
                # Set parameters
                self.std_window = window
                self.std_percentile_threshold = percentile
                
                # Calculate metrics with current parameters
                processed_data = self.calculate_metrics(data)
                signal_stats = self.analyze_signals(processed_data)
                
                # Count total signals
                total_signals = processed_data['low_dispersion_signal'].sum()
                
                # Store results
                results.append({
                    'std_window': window,
                    'percentile': percentile,
                    'total_signals': total_signals,
                    'win_rate_5d': signal_stats['5d']['positive_signals'],
                    'win_rate_10d': signal_stats['10d']['positive_signals'],
                    'win_rate_20d': signal_stats['20d']['positive_signals'],
                    'mean_return_5d': signal_stats['5d']['mean_return'],
                    'mean_return_10d': signal_stats['10d']['mean_return'],
                    'mean_return_20d': signal_stats['20d']['mean_return'],
                    'sharpe_5d': signal_stats['5d']['mean_return'] / processed_data['VIX_fwd_5d_return'][processed_data['low_dispersion_signal']].std() if total_signals > 0 else 0,
                    'sharpe_10d': signal_stats['10d']['mean_return'] / processed_data['VIX_fwd_10d_return'][processed_data['low_dispersion_signal']].std() if total_signals > 0 else 0,
                    'sharpe_20d': signal_stats['20d']['mean_return'] / processed_data['VIX_fwd_20d_return'][processed_data['low_dispersion_signal']].std() if total_signals > 0 else 0
                })
        
        return pd.DataFrame(results)

    def get_optimal_parameters(self, results_df, metric='win_rate_10d', min_signals=10):
        """
        Find optimal parameters based on specified metric
        """
        # Filter for minimum number of signals
        filtered_results = results_df[results_df['total_signals'] >= min_signals]
        
        # Find best parameters
        best_result = filtered_results.loc[filtered_results[metric].idxmax()]
        
        return {
            'std_window': int(best_result['std_window']),
            'percentile': int(best_result['percentile']),
            'total_signals': best_result['total_signals'],
            'win_rate_5d': best_result['win_rate_5d'],
            'win_rate_10d': best_result['win_rate_10d'],
            'win_rate_20d': best_result['win_rate_20d']
        }

def fetch_and_process_data(start_date, end_date):
    analyzer = VolatilityTsunamiAnalyzer(start_date, end_date)
    data = analyzer.fetch_data()
    return analyzer.calculate_metrics(data)

def create_plots(data_close):
    analyzer = VolatilityTsunamiAnalyzer(None, None)  # Dates not needed for plotting
    return analyzer.create_plots(data_close)


