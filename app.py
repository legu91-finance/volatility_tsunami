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
            with st.spinner('Fetching data...'):
                analyzer = VolatilityTsunamiAnalyzer(
                    start_date, 
                    end_date,
                    std_window=std_window,
                    std_percentile_threshold=std_percentile
                )
                
                data = analyzer.fetch_data()
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
                st.plotly_chart(fig, use_container_width=True)
                
                # Add this after the existing imports
                import plotly.express as px

                # Add this after the sidebar settings
                st.sidebar.markdown("---")
                st.sidebar.header("Backtesting")
                if st.sidebar.button("Run Backtest"):
                    with st.spinner('Running backtest...'):
                        # Fetch data for backtesting
                        data = analyzer.fetch_data()
                        
                        # Run backtest
                        results = analyzer.backtest_parameters(
                            data,
                            std_windows_range=(5, 50, 5),
                            percentile_thresholds_range=(5, 30, 5)
                        )
                        
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
                        
                        with tab2:
                            # Mean return heatmaps
                            metrics = ['mean_return_5d', 'mean_return_10d', 'mean_return_20d']
                            periods = [5, 10, 20]
                            
                            for metric, period in zip(metrics, periods):
                                fig = px.density_heatmap(
                                    results,
                                    x='std_window',
                                    y='percentile',
                                    z=metric,
                                    title=f'{period}-Day Mean Return',
                                    labels={metric: 'Mean Return', 'std_window': 'STD Window', 'percentile': 'Percentile'},
                                )
                                st.plotly_chart(fig)
                        
                        with tab3:
                            # Sharpe ratio heatmaps
                            metrics = ['sharpe_5d', 'sharpe_10d', 'sharpe_20d']
                            periods = [5, 10, 20]
                            
                            for metric, period in zip(metrics, periods):
                                fig = px.density_heatmap(
                                    results,
                                    x='std_window',
                                    y='percentile',
                                    z=metric,
                                    title=f'{period}-Day Sharpe Ratio',
                                    labels={metric: 'Sharpe Ratio', 'std_window': 'STD Window', 'percentile': 'Percentile'},
                                )
                                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("End date must be after start date") 