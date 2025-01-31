# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:11:56 2023

@author: legu9
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt
        
        # Create figure with original size
        fig, ax = plt.subplots(figsize=(16, 8))  # Reverted to original size
        
        rolling_low_vix, rolling_low_mask_vix, rolling_low_vvix, rolling_low_mask_vvix = self.calculate_rolling_lows(data)
        vix_percentile = np.percentile(data['VIX Std'].dropna(), self.std_percentile_threshold)
        vvix_percentile = np.percentile(data['VVIX Std'].dropna(), self.std_percentile_threshold)
        
        # Plot 1: S&P 500
        ax.plot(data.index, data['SPX'], label='S&P 500 close')
        ax.scatter(data.index[data['low_dispersion_signal']], 
                   data['SPX'][data['low_dispersion_signal']], 
                   c='red', label=f'Low Dispersion Signal ({self.std_percentile_threshold}th percentile)')
        ax.legend()
        ax.set_ylabel('closing price')
        ax.set_title('S&P 500 close')
        
        # Plot 2: VIX
        ax.plot(data.index, data['VIX'], label='VIX')
        ax.scatter(data['VIX'].index[rolling_low_mask_vix], 
                    rolling_low_vix[rolling_low_mask_vix], c="red", marker="o")
        ax.set_ylabel('VIX close with 4-weeks low')
        ax.legend()
        
        # Plot 3: VVIX
        ax.plot(data.index, data['VVIX'], label='VVIX')
        ax.scatter(data['VVIX'].index[rolling_low_mask_vvix], 
                    rolling_low_vvix[rolling_low_mask_vvix], c="red", marker="o")
        ax.set_ylabel('VVIX close with 4-weeks low')
        ax.legend()
        
        # Plot 4: Standard Deviations
        ax.plot(data.index, data['VIX Std'], label='VIX Std')
        ax.scatter(data.index[data['VIX Std'] < vix_percentile], 
                    data['VIX Std'][data['VIX Std'] < vix_percentile], 
                    c='red', label=f'VIX std below {vix_percentile:.2f}')
        ax.scatter(data.index[data['VVIX Std'] < vvix_percentile], 
                    data['VVIX Std'][data['VVIX Std'] < vvix_percentile], 
                    c='blue', label=f'VVIX std below {vvix_percentile:.2f}')
        ax.plot(data.index, data['VVIX Std'], label='VVIX Std')
        ax.legend()
        ax.set_ylabel('Standard Deviation')
        
        # Plot 5: Yield Curve Spread
        ax.plot(data.index, data['yield curve spread'], label='10y-13w Spread')
        ax.set_ylabel('YC Spread')
        ax.legend()
        
        # Adjust layout
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


