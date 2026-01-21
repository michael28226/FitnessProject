
'''
in anaconda prompt, type: 
    conda activate fitness_env
    
    cd /d "g:\My Drive\projects\FitnessProject"

    streamlit run app7.py

'''
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION / DEFAULTS ---
# Simply change these lists to update the app's behavior
DEFAULT_METRICS = ['weight_ewma', 'ctl_ewma', 'atl_ewma']
DEFAULT_LOOKBACK_MONTHS = 18

st.set_page_config(page_title="Wellness Data Explorer", layout="wide")

@st.cache_data
def load_wellness():
    df = pd.read_parquet('wellness_clean_df.parquet')
    df = df.replace(['None', 'nan', '', 'NaN'], np.nan)
    df.index = pd.to_datetime(df.index)
    return df

df = load_wellness()

st.title("📈 Intervals Metrics Explorer")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("Settings")

# 1. Get available numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 2. Logic to handle default selection safely if columns are missing
safe_defaults = [m for m in DEFAULT_METRICS if m in numeric_cols]
if not safe_defaults:
    safe_defaults = [numeric_cols[0]] if numeric_cols else []

selected_metrics = st.sidebar.multiselect(
    "Pick metrics to plot:",
    options=numeric_cols,
    default=safe_defaults
)

# 3. Time Range Selector (Defaulted to last 12 months)
st.sidebar.markdown("---")
max_date = df.index.max()
default_start = max_date - pd.DateOffset(months=DEFAULT_LOOKBACK_MONTHS)

date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(default_start.date(), max_date.date()),
    min_value=df.index.min().date(),
    max_value=max_date.date()
)

# 4. Chart Display Mode
st.sidebar.markdown("---")
display_mode = st.sidebar.radio(
    "Chart Layout:",
    options=["Overlay (Dual Y-Axes)", "Stacked (Separate Charts)"]
)

# --- DATA FILTERING ---
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    df_filtered = df.loc[mask].copy()
else:
    df_filtered = df.copy()

# --- THE PLOT ---
if selected_metrics:
    df_plot = df_filtered.reset_index()

    if display_mode == "Stacked (Separate Charts)":
        # Create a subplot for each metric
        fig = make_subplots(rows=len(selected_metrics), cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(
                go.Scatter(x=df_plot['date'], y=df_plot[metric], name=metric, mode='lines'),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=metric, row=i+1, col=1)
        
        fig.update_layout(height=300 * len(selected_metrics), showlegend=False)

    else:
        # Dual Y-Axis Overlay Mode
        # Note: Plotly standard supports 2 y-axes easily. More than 2 gets very cluttered.
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for i, metric in enumerate(selected_metrics):
            # Assign first metric to primary axis, others to secondary if they differ in scale
            # Or you can split them 50/50. Here we put the first metric on Axis 1, the rest on Axis 2.
            is_secondary = i > 0
            
            fig.add_trace(
                go.Scatter(x=df_plot['date'], y=df_plot[metric], name=metric, mode='lines'),
                secondary_y=is_secondary
            )
            
        fig.update_layout(
            title_text="Overlay View",
            hovermode="x unified"
        )
        
        # Set axis titles based on selection
        fig.update_yaxes(title_text=selected_metrics[0], secondary_y=False)
        if len(selected_metrics) > 1:
            fig.update_yaxes(title_text=" / ".join(selected_metrics[1:]), secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select metrics from the sidebar to visualize data.")

# --- DATA TABLE ---
with st.expander("🔍 View Raw Data Table"):
    st.dataframe(df_filtered, use_container_width=True)