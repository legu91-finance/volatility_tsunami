import streamlit as st
import datetime
from volatility_tsunami import VolatilityTsunamiAnalyzer
import plotly.express as px

st.set_page_config(layout="wide", page_title="Volatility Tsunami Dashboard")

st.title("Volatility Tsunami Dashboard")

# Settings in sidebar
st.sidebar.header("Analysis Settings")
std_window = st.sidebar.slider("Standard Deviation Window", 5, 50, 20)
std_percentile = st.sidebar.slider("Percentile Threshold", 1, 50, 15)

# Date input widgets
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        datetime.date(2023, 1, 1)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime.date.today()
    )

# Fetch and display data
if start_date and end_date:
    if start_date < end_date:
        try:
            # Validate dates
            today = datetime.date.today()
            if end_date > today:
                st.error(f"End date cannot be in the future. Please select a date before {today}")
                st.stop()
                
            with st.spinner('Fetching data...'):
                analyzer = VolatilityTsunamiAnalyzer(
                    start_date, 
                    end_date,
                    std_window=std_window,
                    std_percentile_threshold=std_percentile
                )
                
                data = analyzer.fetch_data()
                
                if data.empty:
                    st.error("No data available for the selected date range. Please try a different date range.")
                    st.stop()
                
                processed_data = analyzer.calculate_metrics(data)
                
                # Display signal analysis
                signal_stats = analyzer.analyze_signals(processed_data)
                
                st.subheader("Signal Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("5-Day Mean Return", f"{signal_stats['5d']['mean_return']:.2%}")
                    st.metric("5-Day Win Rate", f"{signal_stats['5d']['positive_signals']:.2%}")
                
                with col2:
                    st.metric("10-Day Mean Return", f"{signal_stats['10d']['mean_return']:.2%}")
                    st.metric("10-Day Win Rate", f"{signal_stats['10d']['positive_signals']:.2%}")
                
                with col3:
                    st.metric("20-Day Mean Return", f"{signal_stats['20d']['mean_return']:.2%}")
                    st.metric("20-Day Win Rate", f"{signal_stats['20d']['positive_signals']:.2%}")
                
                # Display plots
                fig = analyzer.create_plots(processed_data)
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
                })
                
                # Backtesting section
                st.sidebar.markdown("---")
                st.sidebar.header("Backtesting")
                if st.sidebar.button("Run Backtest"):
                    with st.spinner('Running backtest...'):
                        try:
                            # Run backtest
                            results = analyzer.backtest_parameters(
                                data,
                                std_windows_range=(5, 50, 5),
                                percentile_thresholds_range=(5, 30, 5)
                            )
                            
                            if results.empty:
                                st.error("No backtest results available. Please try different parameters.")
                                st.stop()
                            
                            # Display results
                            st.subheader("Backtest Results")
                            
                            # Create tabs for different metrics
                            tab1, tab2, tab3 = st.tabs(["Win Rate", "Mean Return", "Sharpe Ratio"])
                            
                            with tab1:
                                # Win rate heatmaps
                                metrics = ['win_rate_5d', 'win_rate_10d', 'win_rate_20d']
                                periods = [5, 10, 20]
                                
                                for metric, period in zip(metrics, periods):
                                    fig = px.density_heatmap(
                                        results,
                                        x='std_window',
                                        y='percentile',
                                        z=metric,
                                        title=f'{period}-Day Win Rate',
                                        labels={metric: 'Win Rate', 'std_window': 'STD Window', 'percentile': 'Percentile'}
                                    )
                                    st.plotly_chart(fig)
                                
                                # Show optimal parameters for 10-day win rate
                                optimal = analyzer.get_optimal_parameters(
                                    results, 
                                    metric='win_rate_10d',
                                    min_signals=10
                                )
                                st.write("Optimal parameters (based on 10-day win rate):")
                                st.write(optimal)
                                
                        except Exception as e:
                            st.error(f"Error during backtesting: {str(e)}")
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try adjusting the date range or parameters.")
    else:
        st.error("End date must be after start date") 