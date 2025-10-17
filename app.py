import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from analysis import load_data, clean_data, get_data_summary, calculate_air_quality_metrics
from visualize import (plot_co_over_time, plot_temperature_vs_humidity, 
                      plot_pollutant_distribution, plot_correlation_heatmap,
                      plot_nox_vs_sensor, create_summary_metrics_display)

# Configure the page
st.set_page_config(
    page_title="Air Quality Dashboard",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2E8B57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E8B57;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    try:
        # Fetch UCI Air Quality dataset
        air_quality = fetch_ucirepo(id=360)
        df = air_quality.data.features.copy()
        
        # Robust Date parsing: let pandas infer common formats (handles MM/DD/YYYY and DD/MM/YYYY)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(
                df['Date'],
                dayfirst=False,
                errors='coerce',
                infer_datetime_format=True
            )
        
        # Robust Time parsing: normalize separators then try multiple parses, fallback to inference
        time_col_saved = None
        if 'Time' in df.columns:
            time_series = df['Time'].astype(str).str.strip()
            # Normalize common separators (e.g. "18.00.00" -> "18:00:00")
            time_norm = time_series.str.replace('.', ':', regex=False)
            
            # Try common explicit formats first
            parsed = pd.to_datetime(time_norm, format='%H:%M:%S', errors='coerce')
            parsed = parsed.combine_first(pd.to_datetime(time_norm, format='%H:%M', errors='coerce'))
            # Final fallback: let pandas infer mixed formats
            parsed = parsed.combine_first(pd.to_datetime(time_series, errors='coerce', infer_datetime_format=True))
            
            # Keep only the time portion (may be NaT if unparsed)
            parsed_time = parsed.dt.time
            # Save parsed time separately and remove Time from df to avoid downstream parsing errors
            time_col_saved = parsed_time
            df = df.drop(columns=['Time'])
        
        # Combine Date and Time into a single Datetime column when both are available
        if 'Date' in df.columns and time_col_saved is not None:
            df['Time'] = time_col_saved.astype(str)  # add back as stable string (HH:MM:SS or 'None')
            df['Datetime'] = pd.to_datetime(
                df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'],
                errors='coerce',
                infer_datetime_format=True
            )
            df = df.sort_values('Datetime').reset_index(drop=True)
        
        # Call clean_data on dataframe without the original raw Time values that may trigger strict parsing
        df = clean_data(df)
        
        # If clean_data removed or altered Time, ensure stable Time/Datetime columns are present
        if 'Datetime' not in df.columns and 'Date' in df.columns and time_col_saved is not None:
            df['Time'] = time_col_saved.astype(str)
            df['Datetime'] = pd.to_datetime(
                df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'],
                errors='coerce',
                infer_datetime_format=True
            )
            df = df.sort_values('Datetime').reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">Air Quality Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Welcome to the Air Quality Analysis Dashboard!** 
    
    This dashboard analyzes air quality data from an Italian city monitoring station. 
    The dataset includes various air pollutants and weather variables collected over time.
    """)
    
    # Load data
    with st.spinner("Loading air quality data..."):
        df = load_and_clean_data()
    
    if df is None:
        st.error("Failed to load data. Please check your connection and try again.")
        return
    
    # Key Metrics - KPI Boxes
    st.markdown('<h2 class="section-header">Key Metrics</h2>', unsafe_allow_html=True)
    
    # Calculate metrics
    metrics = calculate_air_quality_metrics(df)
    display_metrics = create_summary_metrics_display(metrics)
    
    if display_metrics:
        # Display metrics in cards
        cols = st.columns(len(display_metrics))
        
        for i, (metric_name, metric_values) in enumerate(display_metrics.items()):
            with cols[i]:
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.subheader(metric_name)
                for key, value in metric_values.items():
                    st.metric(key, value)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Data preview table
    st.markdown('<h2 class="section-header">Data Preview</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    # Visualizations
    st.markdown('<h2 class="section-header">Data Visualizations</h2>', unsafe_allow_html=True)
    
    # Base charts as required
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CO Concentration Over Time")
        co_plot = plot_co_over_time(df)
        if co_plot:
            st.pyplot(co_plot)
        else:
            st.warning("No CO data available for plotting")
    
    with col2:
        st.subheader("Temperature vs Absolute Humidity")
        temp_humidity_plot = plot_temperature_vs_humidity(df)
        if temp_humidity_plot:
            st.pyplot(temp_humidity_plot)
        else:
            st.warning("No temperature/humidity data available for plotting")
    
    # Additional visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("CO Distribution")
        co_dist_plot = plot_pollutant_distribution(df, 'CO(GT)')
        if co_dist_plot:
            st.pyplot(co_dist_plot)
    
    with col4:
        st.subheader("NOx(GT) vs Sensor Value")
        nox_plot = plot_nox_vs_sensor(df)
        if nox_plot:
            st.pyplot(nox_plot)
        else:
            st.warning("No NOx data available for plotting")
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_plot = plot_correlation_heatmap(df)
    if corr_plot:
        st.pyplot(corr_plot)

if __name__ == "__main__":
    main()
