import streamlit as st
import datetime
from volatility_tsunami import VolatilityTsunamiAnalyzer

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
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("End date must be after start date") 