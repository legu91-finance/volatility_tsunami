# Volatility Tsunami Dashboard

A Streamlit dashboard for analyzing volatility dispersion signals in the market, based on VIX and VVIX behavior. This tool helps identify potential market opportunities when both VIX and VVIX show unusually low dispersion.

## Features

- Real-time data fetching from Yahoo Finance
- Interactive date range selection
- Configurable analysis parameters:
  - Standard deviation window (5-50 days)
  - Percentile threshold (1-50%)
- Signal analysis metrics:
  - Forward returns analysis (5, 10, 20 days)
  - Win rate calculations
- Comprehensive visualizations:
  - S&P 500 price with signal overlays
  - VIX and VVIX indicators
  - Rolling standard deviations
  - Yield curve spread
- Parameter Optimization:
  - Automated backtesting functionality
  - Win rate optimization
  - Mean return analysis
  - Sharpe ratio calculations
  - Interactive heatmaps for parameter selection

## Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/volatility-tsunami.git
cd volatility-tsunami

2. Install required packages:
pip install -r requirements.txt

3. Run the dashboard:
streamlit run volatility_tsunami.py

The dashboard will open in your default web browser. You can:
1. Select your desired date range
2. Adjust analysis parameters in the sidebar
3. View signal analysis metrics
4. Analyze the various charts and indicators
5. Run parameter optimization:
   - Click "Run Backtest" in the sidebar
   - View heatmaps for different metrics
   - Find optimal parameters based on win rate
   - Analyze performance across different time horizons

## Dependencies

- streamlit
- yfinance
- pandas
- numpy
- matplotlib
- plotly

## Project Structure
volatility-tsunami/
├── app.py # Streamlit dashboard interface
├── volatility_tsunami.py # Core analysis logic
├── requirements.txt # Project dependencies
└── README.md # This file

## Analysis Methodology

The dashboard implements a volatility dispersion analysis strategy that looks for:
- Low rolling standard deviation in VIX
- Low rolling standard deviation in VVIX
- Coincident signals in both metrics below specified percentile thresholds

### Parameter Optimization

The backtesting functionality helps optimize two key parameters:
1. Standard Deviation Window (5-50 days)
2. Percentile Threshold (5-30%)

Metrics calculated for each parameter combination:
- Win rates for 5, 10, and 20-day periods
- Mean returns for each period
- Sharpe ratios to assess risk-adjusted returns
- Total number of signals generated

The optimization process:
1. Performs grid search across parameter space
2. Filters results by minimum signal count
3. Identifies optimal parameters based on chosen metric
4. Visualizes results through interactive heatmaps

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License

## Disclaimer

This tool is for informational purposes only. It is not financial advice, and you should not make investment decisions based solely on this analysis. Past performance does not guarantee future results.