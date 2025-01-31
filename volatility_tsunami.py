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
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        data_close = data["Adj Close"]
        data_close.columns = ["SPX", "13w_yield", "10year_yield", "VIX", "VVIX"]
        return data_close
    
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
    
    def create_plots(self, data_close):
        rolling_low_vix, rolling_low_mask_vix, rolling_low_vvix, rolling_low_mask_vvix = self.calculate_rolling_lows(data_close)
        vix_percentile = np.percentile(data_close['VIX Std'].dropna(), self.std_percentile_threshold)
        vvix_percentile = np.percentile(data_close['VVIX Std'].dropna(), self.std_percentile_threshold)
        
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16))
        
        # Plot 1: S&P 500
        ax1.plot(data_close.index, data_close['SPX'], label='S&P 500 close')
        ax1.scatter(data_close.index[data_close['low_dispersion_signal']], 
                   data_close['SPX'][data_close['low_dispersion_signal']], 
                   c='red', label=f'Low Dispersion Signal ({self.std_percentile_threshold}th percentile)')
        ax1.legend()
        ax1.set_ylabel('closing price')
        ax1.set_title('S&P 500 close')
        
        # Plot 2: VIX
        ax2.plot(data_close.index, data_close['VIX'], label='VIX')
        ax2.scatter(data_close['VIX'].index[rolling_low_mask_vix], 
                    rolling_low_vix[rolling_low_mask_vix], c="red", marker="o")
        ax2.set_ylabel('VIX close with 4-weeks low')
        ax2.legend()
        
        # Plot 3: VVIX
        ax3.plot(data_close.index, data_close['VVIX'], label='VVIX')
        ax3.scatter(data_close['VVIX'].index[rolling_low_mask_vvix], 
                    rolling_low_vvix[rolling_low_mask_vvix], c="red", marker="o")
        ax3.set_ylabel('VVIX close with 4-weeks low')
        ax3.legend()
        
        # Plot 4: Standard Deviations
        ax4.plot(data_close.index, data_close['VIX Std'], label='VIX Std')
        ax4.scatter(data_close.index[data_close['VIX Std'] < vix_percentile], 
                    data_close['VIX Std'][data_close['VIX Std'] < vix_percentile], 
                    c='red', label=f'VIX std below {vix_percentile:.2f}')
        ax4.scatter(data_close.index[data_close['VVIX Std'] < vvix_percentile], 
                    data_close['VVIX Std'][data_close['VVIX Std'] < vvix_percentile], 
                    c='blue', label=f'VVIX std below {vvix_percentile:.2f}')
        ax4.plot(data_close.index, data_close['VVIX Std'], label='VVIX Std')
        ax4.legend()
        ax4.set_ylabel('Standard Deviation')
        
        # Plot 5: Yield Curve Spread
        ax5.plot(data_close.index, data_close['yield curve spread'], label='10y-13w Spread')
        ax5.set_ylabel('YC Spread')
        ax5.legend()
        
        plt.tight_layout()
        return fig

def fetch_and_process_data(start_date, end_date):
    analyzer = VolatilityTsunamiAnalyzer(start_date, end_date)
    data = analyzer.fetch_data()
    return analyzer.calculate_metrics(data)

def create_plots(data_close):
    analyzer = VolatilityTsunamiAnalyzer(None, None)  # Dates not needed for plotting
    return analyzer.create_plots(data_close)


