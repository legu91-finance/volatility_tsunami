# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:11:56 2023

@author: legu9
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = "^VVIX ^VIX ^GSPC ^IRX ^TNX"

data = yf.download(ticker, start="2023-1-1", end="2025-01-31")

data_close = data["Adj Close"]
data_close.columns = ["SPX", "13w_yield", "10year_yield", "VIX", "VVIX"]

# Add the calculated standard deviation to the DataFrame
data_close['VIX Std'] = data_close['VIX'].rolling(window=20).std()
data_close['VVIX Std'] = data_close['VVIX'].rolling(window=20).std()
data_close['yield curve spread'] = data_close["10year_yield"] - data_close["13w_yield"]


# Compute the 4-week rolling low for VIX
rolling_low_vix = data_close['VIX'].rolling(window="28D").min()
# Create a boolean mask for the occurrences of 4-week lows
is_rolling_low_vix = (data_close['VIX'] == rolling_low_vix)
# Compute the 4-week rolling low occurrence mask with no occurrence in the last 4 weeks
no_rolling_low_vix_last_4_weeks_mask = (is_rolling_low_vix) & ~(is_rolling_low_vix.rolling(window=20, min_periods=1).sum().shift(1).fillna(0).astype(bool))
# Create a scatter plot of the 4-week lows with red dots and no occurrence in the last 4 weeks
rolling_low_mask_vix = (data_close['VIX'] == rolling_low_vix) & no_rolling_low_vix_last_4_weeks_mask

# Compute the 4-week rolling low for VVIX
rolling_low_vvix = data_close['VVIX'].rolling(window="28D").min()
# Create a boolean mask for the occurrences of 4-week lows
is_rolling_low_vvix = (data_close['VVIX'] == rolling_low_vvix)
# Compute the 4-week rolling low occurrence mask with no occurrence in the last 4 weeks
no_rolling_low_vvix_last_4_weeks_mask = (is_rolling_low_vvix) & ~(is_rolling_low_vvix.rolling(window=20, min_periods=1).sum().shift(1).fillna(0).astype(bool))
# Create a scatter plot of the 4-week lows with red dots and no occurrence in the last 4 weeks
rolling_low_mask_vvix = (data_close['VVIX'] == rolling_low_vvix) & no_rolling_low_vvix_last_4_weeks_mask

vix_percentile = np.percentile(data_close['VIX Std'].dropna(), 15)
vvix_percentile = np.percentile(data_close['VVIX Std'].dropna(), 15)


# Create subplots
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16))

# Plot the S&P 500 closing price
ax1.plot(data_close.index, data_close['SPX'], label='S&P 500 close')
ax1.legend()
ax1.set_ylabel('closing price')
ax1.set_title('S&P 500 close')
ax1.scatter(data_close.index[(data_close['VIX Std'] <vix_percentile) & 
                             (data_close['VVIX Std'] <vvix_percentile) 
                             ], 
            data_close['SPX'][(data_close['VIX Std'] <vix_percentile) & 
                              (data_close['VVIX Std'] <vvix_percentile)
                              ], 
                              c='red', label='Below 20')

# Plot the VIX with 4-week lows and no occurrence in the last 4 weeks
ax2.plot(data_close.index, data_close['VIX'], label='VIX')
ax2.scatter(data_close['VIX'].index[rolling_low_mask_vix], rolling_low_vix[rolling_low_mask_vix], c="red", marker="o")
ax2.set_ylabel('VIX close with 4-weeks low')
ax2.legend()

# Plot the VVIX with 4-week lows and no occurrence in the last 4 weeks
ax3.plot(data_close.index, data_close['VVIX'], label='VVIX')
ax3.scatter(data_close['VVIX'].index[rolling_low_mask_vvix], rolling_low_vvix[rolling_low_mask_vvix], c="red", marker="o")
ax3.set_ylabel('VVIX close with 4-weeks low')

ax3.legend()

# Plot the standard deviation of VIX and VVIX

ax4.plot(data_close.index, data_close['VIX Std'], label='VIX Std')
ax4.scatter(data_close.index[data_close['VIX Std'] <vix_percentile], data_close['VIX Std'][data_close['VIX Std'] <vix_percentile], c='red', label='VIX std below 0.86')
ax4.scatter(data_close.index[data_close['VVIX Std'] <vvix_percentile], data_close['VVIX Std'][data_close['VVIX Std'] <vvix_percentile], c='blue', label='VVIX std below 3.42')


ax4.plot(data_close.index, data_close['VVIX Std'], label='VVIX Std')
ax4.legend()
ax4.set_ylabel('Standard Deviation')


# Plot the yield curve spread
ax5.plot(data_close.index, data_close['yield curve spread'], label='10y-13w Spread')
ax5.set_ylabel('YC Spread')
ax5.legend()

# Show the plot
plt.show()


