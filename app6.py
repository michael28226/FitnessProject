
'''
in anaconda prompt, type: 
    conda activate fitness_env
    
    
    streamlit run app4y.py

'''

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Wellness Data Explorer", layout="wide")

@st.cache_data
def load_wellness():
    # Load the parquet file
    df = pd.read_parquet('wellness_clean_df.parquet')
    
    # Standardize missing values
    df = df.replace(['None', 'nan', '', 'NaN'], np.nan)
    
    # Ensure the index is a datetime object for proper plotting
    df.index = pd.to_datetime(df.index)
    
    return df

df = load_wellness()

st.title("📈 Wellness Metric Explorer")

# --- SIDEBAR: SELECT METRICS ---
st.sidebar.header("Settings")

# Get list of all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

selected_metrics = st.sidebar.multiselect(
    "Pick metrics to plot:",
    options=numeric_cols,
    default=['hrv', 'weight'] if 'weight' in numeric_cols else [numeric_cols[0]]
)

# --- TREND SETTINGS ---
st.sidebar.markdown("---")
st.sidebar.header("Trends & Smoothing")
show_trends = st.sidebar.checkbox("Show 7-Day Moving Averages", value=True)
split_charts = st.sidebar.checkbox("Separate Y-Axes (Facet)", value=True)

# --- THE PLOT ---
if selected_metrics:
    # 1. Reset index FIRST so 'date' is a column we can use for plotting
    df_plot = df.reset_index()
    
    metrics_to_melt = []
    
    # 2. CALCULATE TRENDS on the reset dataframe
    for metric in selected_metrics:
        metrics_to_melt.append(metric) # Add the raw data
        
        if show_trends:
            trend_col_name = f"{metric} (7d Trend)"
            # Calculate rolling average on the specific metric
            df_plot[trend_col_name] = df_plot[metric].rolling(window=7, min_periods=1).mean()
            metrics_to_melt.append(trend_col_name)

    # 3. Prepare data for Plotly using the newly created columns
    # We use the 'date' column created by reset_index()
    df_melted = df_plot.melt(id_vars=['date'], value_vars=metrics_to_melt)
    
    if split_charts:
        # Group raw data and its trend line into the same row
        df_melted['group'] = df_melted['variable'].str.replace(" (7d Trend)", "", regex=False)
        
        fig = px.line(
            df_melted, 
            x='date', 
            y='value', 
            facet_row='group',
            color='variable',  
            height=300 * len(selected_metrics),
            labels={'value': 'Value', 'date': 'Date', 'variable': 'Type'}
        )
        
        # STYLING: Thick trend lines, thin dashed raw lines
        fig.for_each_trace(lambda t: t.update(line=dict(width=4)) if "Trend" in t.name 
                           else t.update(line=dict(width=1.5, dash='dot')))
        
        # Ensure Y-axes are independent so eFTP doesn't squash HRV
        fig.update_yaxes(matches=None)
        
    else:
        # Simple combined chart
        fig = px.line(df_melted, x='date', y='value', color='variable')

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select one or more metrics from the sidebar to start plotting.")

# --- DATA TABLE ---
with st.expander("🔍 View Raw Wellness Table"):
    st.dataframe(df, use_container_width=True)