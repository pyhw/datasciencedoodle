import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="NYC Subway Ridership Analysis - Comprehensive Dashboard",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for light and dark mode support
st.markdown("""
<style>
    /* Main header with dynamic gradient */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1f77b4, #ff7f0e, #2ca02c, #d62728);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 8s ease-in-out infinite;
        background-size: 400% 400%;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Light mode styles */
    [data-theme="light"] .metric-container {
        background: linear-gradient(135deg, #f0f2f6, #e8ecf0);
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    [data-theme="light"] .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    [data-theme="light"] .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
        background: linear-gradient(90deg, #3498db, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Dark mode styles */
    [data-theme="dark"] .metric-container,
    .metric-container {
        background: linear-gradient(135deg, #2d3748, #4a5568);
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #63b3ed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        color: #e2e8f0;
    }
    
    [data-theme="dark"] .metric-container:hover,
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, #4a5568, #2d3748);
    }
    
    [data-theme="dark"] .section-header,
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #63b3ed;
        background: linear-gradient(90deg, #63b3ed, #4299e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    
    [data-theme="dark"] .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748, #1a202c);
    }
    
    /* Form elements dark mode support */
    [data-theme="dark"] .stSelectbox > div > div,
    [data-theme="dark"] .stSlider > div > div > div {
        background-color: #4a5568 !important;
        color: #e2e8f0 !important;
        border-color: #63b3ed !important;
    }
    
    /* Metrics styling for both modes */
    .stMetric {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.1), rgba(255, 127, 14, 0.1));
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(31, 119, 180, 0.2);
    }
    
    [data-theme="dark"] .stMetric {
        background: linear-gradient(135deg, rgba(99, 179, 237, 0.1), rgba(255, 127, 14, 0.1));
        border: 1px solid rgba(99, 179, 237, 0.3);
    }
    
    /* Success/Warning/Error message styling */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    
    /* Plotly chart container enhancement */
    .js-plotly-plot {
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    [data-theme="dark"] .js-plotly-plot {
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    /* Loading spinner enhancement */
    .stSpinner {
        border-color: #1f77b4 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
    }
    
    [data-theme="dark"] .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #4a5568, #2d3748);
        border: 1px solid #63b3ed;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the NYC subway ridership data"""
    import os
    
    # # Temporary debug to see what's in your repository
    # st.info("🔍 **Debug: Checking file locations...**")
    
    # # Show current directory contents
    # try:
    #     root_files = [f for f in os.listdir('.') if f.endswith('.csv') or os.path.isdir(f)]
    #     st.write(f"📁 **Root directory contents:** {root_files}")
    # except:
    #     st.write("❌ Could not list root directory")
    
    # # Check content folder
    # if os.path.exists('content'):
    #     try:
    #         content_files = os.listdir('content')
    #         st.write(f"📁 **Content folder:** {content_files}")
    #     except:
    #         st.write("❌ Could not list content folder")
    # else:
    #     st.write("❌ Content folder doesn't exist")
    
    # st.markdown("---")
    
    try:
        # Try multiple file paths for different environments
        possible_paths = [
            'content/MTA_Daily_Ridership_and_Traffic__Beginning_2020_20250803.csv',  # Linux/Mac path
            r'content\MTA_Daily_Ridership_and_Traffic__Beginning_2020_20250803.csv',  # Windows path
            'MTA_Daily_Ridership_and_Traffic__Beginning_2020_20250803.csv',  # Root directory
            'data/MTA_Daily_Ridership_and_Traffic__Beginning_2020_20250803.csv'  # Alternative data folder
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"✅ Data loaded successfully from: {path}")
                break
            except FileNotFoundError:
                st.write(f"❌ Not found: {path}")
                continue
        
        if df is None:
            st.error("❌ Data file not found in expected locations.")
            st.info("Please ensure the MTA ridership CSV file is uploaded to the repository.")
            return None, None, None
        
        # Filter for subway data and clean
        df_subway = df[df['Mode'] == 'Subway'].copy()
        df_subway.rename(columns={'Date': 'Date', 'Count': 'Subway_Ridership'}, inplace=True)
        df_subway['Date'] = pd.to_datetime(df_subway['Date'], errors='coerce')
        df_subway['Subway_Ridership'] = pd.to_numeric(df_subway['Subway_Ridership'], errors='coerce')
        df_subway.dropna(subset=['Date', 'Subway_Ridership'], inplace=True)
        df_subway.drop('Mode', axis=1, inplace=True)
        df_subway.sort_values(by='Date', inplace=True)
        
        # Filter for last 5 years
        latest_date = df_subway['Date'].max()
        start_date = latest_date - pd.DateOffset(years=5)
        df_recent = df_subway[df_subway['Date'] >= start_date].copy()
        
        # Feature engineering
        df_recent['Year'] = df_recent['Date'].dt.year
        df_recent['Month'] = df_recent['Date'].dt.month
        df_recent['Day'] = df_recent['Date'].dt.day
        df_recent['Dayofweek'] = df_recent['Date'].dt.dayofweek
        df_recent['Dayofyear'] = df_recent['Date'].dt.dayofyear
        df_recent['Weekofyear'] = df_recent['Date'].dt.isocalendar().week
        df_recent['Quarter'] = df_recent['Date'].dt.quarter
        
        return df_recent, df_subway, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def train_prophet_model(df_recent):
    """Train Prophet model on the data"""
    try:
        # Prepare data for Prophet
        df_prophet = df_recent.rename(columns={'Date': 'ds', 'Subway_Ridership': 'y'})
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(subset=['y'], inplace=True)
        
        # Train model
        model = Prophet()
        model.fit(df_prophet[['ds', 'y']])
        
        # Generate predictions for entire dataset
        future = model.make_future_dataframe(periods=365)  # Include future predictions
        forecast = model.predict(future)
        
        return model, forecast, df_prophet
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

def create_comparison_data(df_subway, forecast, threshold=0.20):
    """Create comparison data with unusual activity detection"""
    # Merge actual and predicted data for the full dataset
    full_comparison_df = pd.merge(
        df_subway, 
        forecast[['ds', 'yhat']], 
        left_on='Date', 
        right_on='ds', 
        how='inner'
    )
    full_comparison_df.rename(columns={'yhat': 'Predicted_Ridership'}, inplace=True)
    full_comparison_df.drop('ds', axis=1, inplace=True)
    full_comparison_df['Difference'] = full_comparison_df['Subway_Ridership'] - full_comparison_df['Predicted_Ridership']
    
    # Unusual activity detection
    full_comparison_df['Unusual_Activity'] = full_comparison_df.apply(
        lambda row: 'Unusually High' if row['Difference'] > row['Predicted_Ridership'] * threshold else (
            'Unusually Low' if row['Difference'] < -row['Predicted_Ridership'] * threshold else 'Normal'
        ), axis=1
    )
    
    # Add temporal features
    full_comparison_df['Year'] = full_comparison_df['Date'].dt.year
    full_comparison_df['Month'] = full_comparison_df['Date'].dt.month
    full_comparison_df['Quarter'] = full_comparison_df['Date'].dt.quarter
    
    return full_comparison_df

def create_enhanced_forecast_visualization(df_prophet, forecast, template="plotly_dark"):
    """Create enhanced time series visualization with interactive features"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('NYC Subway Ridership: Actual vs Predicted', 'Prediction Uncertainty'),
        vertical_spacing=0.08
    )

    # Main time series plot
    fig.add_trace(
        go.Scatter(
            x=df_prophet['ds'], 
            y=df_prophet['y'],
            mode='lines',
            name='Actual Ridership',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Actual:</b> %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat'],
            mode='lines',
            name='Predicted Ridership',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            name='95% Confidence Interval',
            hovertemplate='<b>Date:</b> %{x}<br><b>Lower:</b> %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add uncertainty visualization
    uncertainty = forecast['yhat_upper'] - forecast['yhat_lower']
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=uncertainty,
            mode='lines',
            name='Prediction Uncertainty',
            line=dict(color='#d62728', width=1),
            hovertemplate='<b>Date:</b> %{x}<br><b>Uncertainty:</b> %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title={
            'text': 'NYC Subway Daily Ridership Forecast Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Add range selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30D", step="day", stepmode="backward"),
                dict(count=90, label="3M", step="day", stepmode="backward"),
                dict(count=180, label="6M", step="day", stepmode="backward"),
                dict(count=365, label="1Y", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        rangeslider=dict(visible=True, thickness=0.05),
        type="date",
        row=1
    )

    fig.update_yaxes(title_text="Daily Ridership", row=1, col=1)
    fig.update_yaxes(title_text="Uncertainty Range", row=2, col=1)
    
    return fig

def create_unusual_activity_analysis(full_comparison_df, template="plotly_dark"):
    """Create comprehensive unusual activity visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Ridership Timeline with Unusual Activity Highlighting',
            'Unusual Activity Distribution by Year',
            'Monthly Pattern of Unusual Activity',
            'Day of Week Pattern'
        ),
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # Main timeline plot
    colors = {
        'Normal': '#1f77b4',
        'Unusually Low': '#d62728', 
        'Unusually High': '#2ca02c'
    }

    for activity_type in ['Normal', 'Unusually Low', 'Unusually High']:
        subset = full_comparison_df[full_comparison_df['Unusual_Activity'] == activity_type]
        
        fig.add_trace(
            go.Scatter(
                x=subset['Date'],
                y=subset['Subway_Ridership'],
                mode='markers',
                name=activity_type,
                marker=dict(
                    color=colors[activity_type],
                    size=4 if activity_type == 'Normal' else 6,
                    opacity=0.6 if activity_type == 'Normal' else 0.8
                ),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Ridership:</b> %{y:,.0f}<br>' +
                             '<b>Status:</b> ' + activity_type + '<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )

    # Yearly distribution
    yearly_counts = full_comparison_df[full_comparison_df['Unusual_Activity'] != 'Normal'].groupby(['Year', 'Unusual_Activity']).size().unstack(fill_value=0)

    if 'Unusually Low' in yearly_counts.columns:
        fig.add_trace(
            go.Bar(
                x=yearly_counts.index,
                y=yearly_counts['Unusually Low'],
                name='Unusually Low (Yearly)',
                marker_color='#d62728',
                showlegend=False,
                hovertemplate='<b>Year:</b> %{x}<br><b>Unusually Low Days:</b> %{y}<extra></extra>'
            ),
            row=2, col=1
        )

    if 'Unusually High' in yearly_counts.columns:
        fig.add_trace(
            go.Bar(
                x=yearly_counts.index,
                y=yearly_counts['Unusually High'],
                name='Unusually High (Yearly)',
                marker_color='#2ca02c',
                showlegend=False,
                hovertemplate='<b>Year:</b> %{x}<br><b>Unusually High Days:</b> %{y}<extra></extra>'
            ),
            row=2, col=1
        )

    # Monthly pattern
    monthly_counts = full_comparison_df[full_comparison_df['Unusual_Activity'] != 'Normal'].groupby(['Month', 'Unusual_Activity']).size().unstack(fill_value=0)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if 'Unusually Low' in monthly_counts.columns:
        fig.add_trace(
            go.Bar(
                x=[month_names[i-1] for i in monthly_counts.index],
                y=monthly_counts['Unusually Low'],
                name='Unusually Low (Monthly)',
                marker_color='#d62728',
                opacity=0.7,
                showlegend=False,
                hovertemplate='<b>Month:</b> %{x}<br><b>Unusually Low Days:</b> %{y}<extra></extra>'
            ),
            row=2, col=2
        )

    if 'Unusually High' in monthly_counts.columns:
        fig.add_trace(
            go.Bar(
                x=[month_names[i-1] for i in monthly_counts.index],
                y=monthly_counts['Unusually High'],
                name='Unusually High (Monthly)',
                marker_color='#2ca02c',
                opacity=0.7,
                showlegend=False,
                hovertemplate='<b>Month:</b> %{x}<br><b>Unusually High Days:</b> %{y}<extra></extra>'
            ),
            row=2, col=2
        )

    fig.update_layout(
        title={
            'text': 'Comprehensive Unusual Activity Analysis Dashboard',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        hovermode='closest',
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Add range selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        row=1, col=1
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Daily Ridership", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Number of Days", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="Number of Days", row=2, col=2)
    
    return fig

def create_threshold_analysis(full_comparison_df, template="plotly_dark"):
    """Create threshold sensitivity analysis"""
    thresholds = [0.1, 0.2, 0.3]
    all_threshold_results = []
    
    for threshold in thresholds:
        df_temp = full_comparison_df.copy()
        df_temp['Threshold_Activity'] = df_temp.apply(
            lambda row: 'Unusually High' if row['Difference'] > row['Predicted_Ridership'] * threshold else (
                'Unusually Low' if row['Difference'] < -row['Predicted_Ridership'] * threshold else 'Normal'
            ), axis=1
        )
        df_temp['Threshold'] = threshold
        all_threshold_results.append(df_temp)
    
    combined_results = pd.concat(all_threshold_results, ignore_index=True)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Threshold Impact on Total Unusual Activity',
            'Quarterly Trends by Threshold',
            'Activity Distribution Heatmap',
            'Threshold Sensitivity Analysis'
        ),
        specs=[[{}, {}],
               [{"type": "heatmap"}, {}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Threshold comparison
    threshold_summary = combined_results[
        combined_results['Threshold_Activity'] != 'Normal'
    ].groupby(['Threshold', 'Threshold_Activity']).size().unstack(fill_value=0)

    for activity_type in ['Unusually Low', 'Unusually High']:
        if activity_type in threshold_summary.columns:
            fig.add_trace(
                go.Bar(
                    x=threshold_summary.index,
                    y=threshold_summary[activity_type],
                    name=activity_type,
                    marker_color='#d62728' if activity_type == 'Unusually Low' else '#2ca02c',
                    hovertemplate=f'<b>Threshold:</b> %{{x}}<br><b>{activity_type}:</b> %{{y}} days<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Sensitivity analysis
    sensitivity_data = []
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        temp_df = full_comparison_df.copy()
        temp_df['Temp_Unusual'] = temp_df.apply(
            lambda row: 'Unusual' if (abs(row['Difference']) > abs(row['Predicted_Ridership']) * threshold) else 'Normal',
            axis=1
        )
        unusual_count = (temp_df['Temp_Unusual'] == 'Unusual').sum()
        unusual_pct = unusual_count / len(temp_df) * 100
        sensitivity_data.append({'Threshold': threshold, 'Unusual_Percentage': unusual_pct})

    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df['Threshold'],
            y=sensitivity_df['Unusual_Percentage'],
            mode='lines+markers',
            name='Sensitivity Curve',
            line=dict(color='#9467bd', width=4),
            marker=dict(size=8),
            hovertemplate='<b>Threshold:</b> %{x}<br><b>Unusual Activity:</b> %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': 'Threshold Analysis Dashboard: Impact on Unusual Activity Detection',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=800,
        showlegend=True,
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig, sensitivity_df

def create_executive_dashboard(full_comparison_df, forecast, template="plotly_dark"):
    """Create the redesigned executive dashboard with 5 focused charts"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Current Week: Forecast vs Realized (with Confidence)',
            'YTD Ridership vs Last Year (YoY Change)', 
            'Current Month: Unusual Activities Daily',
            'Monthly Seasonality Patterns',
            'Current Month: Historical vs Predicted (Interactive)',
            'Key Performance Summary'
        ),
        specs=[[{}, {}, {}],
               [{}, {}, {"type": "indicator"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.06
    )
    
    # Get current date info for calculations
    latest_date = full_comparison_df['Date'].max()
    current_year = latest_date.year
    last_year = current_year - 1
    current_month = latest_date.month
    
    # 1. Current Week: Forecast vs Realized (with Confidence)
    current_week_start = latest_date - pd.DateOffset(days=latest_date.dayofweek)
    current_week_end = current_week_start + pd.DateOffset(days=6)
    
    current_week_data = full_comparison_df[
        (full_comparison_df['Date'] >= current_week_start) & 
        (full_comparison_df['Date'] <= current_week_end)
    ].sort_values('Date')
    
    if len(current_week_data) > 0:
        # Merge with forecast for confidence intervals
        current_week_forecast = forecast[
            (forecast['ds'] >= current_week_start) & 
            (forecast['ds'] <= current_week_end)
        ]
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=current_week_forecast['ds'],
                y=current_week_forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=current_week_forecast['ds'],
                y=current_week_forecast['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(99, 179, 237, 0.2)',
                name='95% Confidence',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Actual ridership
        fig.add_trace(
            go.Scatter(
                x=current_week_data['Date'],
                y=current_week_data['Subway_Ridership'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Actual:</b> %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Predicted ridership
        fig.add_trace(
            go.Scatter(
                x=current_week_forecast['ds'],
                y=current_week_forecast['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. YTD Ridership vs Last Year (YoY Change)
    current_ytd = full_comparison_df[
        (full_comparison_df['Date'].dt.year == current_year)
    ]['Subway_Ridership'].sum()
    
    last_year_ytd = full_comparison_df[
        (full_comparison_df['Date'].dt.year == last_year) &
        (full_comparison_df['Date'].dt.dayofyear <= latest_date.dayofyear)
    ]['Subway_Ridership'].sum()
    
    yoy_change = ((current_ytd - last_year_ytd) / last_year_ytd * 100) if last_year_ytd > 0 else 0
    
    fig.add_trace(
        go.Bar(
            x=[f'{last_year} YTD', f'{current_year} YTD'],
            y=[last_year_ytd/1000000, current_ytd/1000000],
            marker_color=['#ff7f0e', '#2ca02c' if yoy_change > 0 else '#d62728'],
            text=[f'{last_year_ytd/1000000:.1f}M', f'{current_ytd/1000000:.1f}M<br>({yoy_change:+.1f}%)'],
            textposition='auto',
            showlegend=False,
            hovertemplate='<b>Period:</b> %{x}<br><b>Ridership:</b> %{y:.1f}M<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Current Month: Unusual Activities Daily
    current_month_start = latest_date.replace(day=1)
    current_month_data = full_comparison_df[
        (full_comparison_df['Date'] >= current_month_start) & 
        (full_comparison_df['Date'] <= latest_date)
    ].sort_values('Date')
    
    # Color code by unusual activity
    colors_map = {'Normal': '#1f77b4', 'Unusually Low': '#d62728', 'Unusually High': '#2ca02c'}
    marker_colors = [colors_map.get(activity, '#1f77b4') for activity in current_month_data['Unusual_Activity']]
    
    fig.add_trace(
        go.Scatter(
            x=current_month_data['Date'],
            y=current_month_data['Subway_Ridership'],
            mode='markers',
            marker=dict(
                color=marker_colors,
                size=8,
                line=dict(width=1, color='white')
            ),
            name='Daily Activity',
            showlegend=False,
            hovertemplate='<b>Date:</b> %{x}<br><b>Ridership:</b> %{y:,.0f}<br><b>Status:</b> %{text}<extra></extra>',
            text=current_month_data['Unusual_Activity']
        ),
        row=1, col=3
    )
    
    # 4. Monthly Seasonality Patterns
    monthly_avg = full_comparison_df.groupby(full_comparison_df['Date'].dt.month)['Subway_Ridership'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Highlight current month
    colors = ['#63b3ed' if i+1 == current_month else '#a0aec0' for i in range(12)]
    
    fig.add_trace(
        go.Bar(
            x=month_names,
            y=monthly_avg.values/1000000,
            marker_color=colors,
            name='Monthly Average',
            showlegend=False,
            text=[f'{v:.1f}M' for v in monthly_avg.values/1000000],
            textposition='auto',
            hovertemplate='<b>Month:</b> %{x}<br><b>Avg Ridership:</b> %{y:.1f}M<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 5. Current Month: Historical vs Predicted (Interactive)
    fig.add_trace(
        go.Scatter(
            x=current_month_data['Date'],
            y=current_month_data['Subway_Ridership'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Actual:</b> %{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=current_month_data['Date'],
            y=current_month_data['Predicted_Ridership'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 6. Key Performance Summary (Gauge)
    mae = np.mean(np.abs(full_comparison_df['Difference']))
    mape = np.mean(np.abs(full_comparison_df['Difference'] / full_comparison_df['Subway_Ridership']) * 100)
    model_score = max(0, min(100, (100 - mape)))  # Convert MAPE to a 0-100 score
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=model_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Model Accuracy Score"},
            delta={'reference': 90, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2ca02c" if model_score > 85 else "#ff7f0e" if model_score > 70 else "#d62728"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        title={
            'text': 'NYC Subway Ridership Analysis - Executive Dashboard',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update axes
    fig.update_yaxes(title_text="Daily Ridership", row=1, col=1)
    fig.update_yaxes(title_text="YTD Ridership (M)", row=1, col=2)
    fig.update_yaxes(title_text="Daily Ridership", row=1, col=3)
    fig.update_yaxes(title_text="Avg Ridership (M)", row=2, col=1)
    fig.update_yaxes(title_text="Daily Ridership", row=2, col=2)
    
    # Add range selectors for interactive charts
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="7D", step="day", stepmode="backward"),
                dict(count=14, label="14D", step="day", stepmode="backward"),
                dict(count=30, label="30D", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        row=2, col=2
    )
    
    # Calculate return metrics
    current_ytd = full_comparison_df[
        (full_comparison_df['Date'].dt.year == current_year)
    ]['Subway_Ridership'].sum()
    
    last_year_ytd = full_comparison_df[
        (full_comparison_df['Date'].dt.year == last_year) &
        (full_comparison_df['Date'].dt.dayofyear <= latest_date.dayofyear)
    ]['Subway_Ridership'].sum()
    
    yoy_change = ((current_ytd - last_year_ytd) / last_year_ytd * 100) if last_year_ytd > 0 else 0
    
    return fig, mae, mape, model_score, yoy_change, current_ytd, last_year_ytd, latest_date

def main():
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🚇 NYC Subway Ridership Analysis - Comprehensive Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar with enhanced controls
    st.sidebar.markdown("## 📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Theme selector
    st.sidebar.markdown("### 🎨 Display Settings")
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True, help="Toggle between light and dark theme")
    
    # Apply theme-based styling
    if dark_mode:
        st.markdown('<div data-theme="dark">', unsafe_allow_html=True)
        chart_template = "plotly_dark"
    else:
        st.markdown('<div data-theme="light">', unsafe_allow_html=True)
        chart_template = "plotly_white"
    
    # Dashboard section selector
    dashboard_section = st.sidebar.selectbox(
        "Choose Dashboard Section:",
        ["📈 Executive Summary", "🔮 Forecast Analysis", "⚠️ Unusual Activity Analysis", "🎯 Threshold Analysis", "📋 Current Week Analysis"]
    )
    
    # Threshold selector
    st.sidebar.markdown("### 🎛️ Analysis Parameters")
    threshold = st.sidebar.slider(
        "Unusual Activity Threshold", 
        min_value=0.05, 
        max_value=0.50, 
        value=0.20, 
        step=0.05,
        help="Threshold for detecting unusual activity (higher = more conservative)"
    )
    
    # Load data with progress indicator
    with st.spinner("🔄 Loading and processing data..."):
        df_recent, df_subway, df_original = load_and_process_data()
    
    if df_recent is None:
        st.error("❌ Failed to load data. Please check the data file path.")
        return
    
    # Train model with progress indicator
    with st.spinner("🤖 Training Prophet model..."):
        model, forecast, df_prophet = train_prophet_model(df_recent)
    
    if model is None:
        st.error("❌ Failed to train model.")
        return
    
    # Create comparison data
    full_comparison_df = create_comparison_data(df_subway, forecast, threshold)
    
    # Sidebar metrics
    st.sidebar.markdown("### 📊 Key Metrics")
    total_days = len(full_comparison_df)
    avg_ridership = full_comparison_df['Subway_Ridership'].mean()
    unusual_stats = full_comparison_df['Unusual_Activity'].value_counts()
    
    st.sidebar.metric("📅 Total Days Analyzed", f"{total_days:,}")
    st.sidebar.metric("🚇 Average Daily Ridership", f"{avg_ridership:,.0f}")
    st.sidebar.metric("📉 Unusual Low Days", f"{unusual_stats.get('Unusually Low', 0):,}")
    st.sidebar.metric("📈 Unusual High Days", f"{unusual_stats.get('Unusually High', 0):,}")
    
    # Main dashboard content based on selection
    if dashboard_section == "📈 Executive Summary":
        st.markdown('<div class="section-header">📈 Executive Summary Dashboard</div>', unsafe_allow_html=True)
        
        # Create and display executive dashboard
        fig, mae, mape, model_score, yoy_change, current_ytd, last_year_ytd, latest_date = create_executive_dashboard(full_comparison_df, forecast, chart_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics in columns
        st.markdown("### 🎯 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label="Model Accuracy Score",
                value=f"{model_score:.1f}%",
                delta=f"{'Excellent' if model_score > 85 else 'Good' if model_score > 70 else 'Needs Improvement'}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label="YoY Ridership Change",
                value=f"{yoy_change:+.1f}%",
                delta=f"vs {latest_date.year - 1} YTD"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label="Current YTD Ridership",
                value=f"{current_ytd/1000000:.1f}M",
                delta=f"{(current_ytd - last_year_ytd)/1000000:+.1f}M vs Last Year"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            current_month_unusual = len(full_comparison_df[
                (full_comparison_df['Date'].dt.year == latest_date.year) &
                (full_comparison_df['Date'].dt.month == latest_date.month) &
                (full_comparison_df['Unusual_Activity'] != 'Normal')
            ])
            st.metric(
                label="Current Month Unusual Days",
                value=f"{current_month_unusual}",
                delta="days detected"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed findings
        st.markdown("### 📋 Executive Insights & Action Items")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Current Performance")
            current_week_data = full_comparison_df[
                (full_comparison_df['Date'] >= latest_date - pd.DateOffset(days=latest_date.dayofweek)) & 
                (full_comparison_df['Date'] <= latest_date)
            ]
            current_week_avg = current_week_data['Subway_Ridership'].mean() if len(current_week_data) > 0 else 0
            
            st.write(f"• **Current Week Average**: {current_week_avg:,.0f} daily passengers")
            st.write(f"• **Model Accuracy**: {model_score:.1f}% - {'Excellent performance' if model_score > 85 else 'Good performance' if model_score > 70 else 'Needs improvement'}")
            st.write(f"• **YoY Growth**: {yoy_change:+.1f}% change vs {latest_date.year - 1}")
            st.write(f"• **Current Month Anomalies**: {current_month_unusual} unusual activity days detected")
        
        with col2:
            st.markdown("#### 🎯 Strategic Recommendations")
            monthly_avg = full_comparison_df.groupby(full_comparison_df['Date'].dt.month)['Subway_Ridership'].mean()
            peak_month = monthly_avg.idxmax()
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            st.write(f"• **Capacity Planning**: Peak season in {month_names[peak_month]} ({monthly_avg.iloc[peak_month-1]/1000000:.1f}M avg)")
            st.write(f"• **Alert Threshold**: Current {threshold*100:.0f}% threshold working effectively")
            st.write(f"• **Growth Trend**: {'Positive momentum' if yoy_change > 0 else 'Declining trend - investigate causes'}")
            st.write(f"• **Forecast Confidence**: Monitor weekly actuals vs predictions for accuracy")
        
        # Real-time Analysis Summary
        st.markdown("### 🕐 Real-Time Analysis Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📅 This Week")
            current_week_unusual = len(current_week_data[current_week_data['Unusual_Activity'] != 'Normal']) if len(current_week_data) > 0 else 0
            st.metric(
                label="Unusual Activity Days",
                value=f"{current_week_unusual}",
                delta=f"out of {len(current_week_data)} days" if len(current_week_data) > 0 else "No data"
            )
        
        with col2:
            st.markdown("#### 📊 This Month")
            current_month_data = full_comparison_df[
                (full_comparison_df['Date'].dt.year == latest_date.year) &
                (full_comparison_df['Date'].dt.month == latest_date.month)
            ]
            current_month_avg = current_month_data['Subway_Ridership'].mean() if len(current_month_data) > 0 else 0
            st.metric(
                label="Average Daily Ridership",
                value=f"{current_month_avg:,.0f}",
                delta=f"{len(current_month_data)} days recorded" if len(current_month_data) > 0 else "No data"
            )
            
        with col3:
            st.markdown("#### 📈 YTD Performance")
            st.metric(
                label=f"{latest_date.year} vs {latest_date.year - 1}",
                value=f"{current_ytd/1000000:.1f}M",
                delta=f"{yoy_change:+.1f}% change"
            )
    
    elif dashboard_section == "🔮 Forecast Analysis":
        st.markdown('<div class="section-header">🔮 Time Series Forecast Analysis</div>', unsafe_allow_html=True)
        
        # Enhanced forecast visualization
        fig = create_enhanced_forecast_visualization(df_prophet, forecast, chart_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary statistics
        st.markdown("### 📊 Forecast Summary Statistics")
        uncertainty = forecast['yhat_upper'] - forecast['yhat_lower']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Daily Ridership", f"{df_prophet['y'].mean():,.0f}")
        with col2:
            st.metric("Average Prediction Uncertainty", f"{uncertainty.mean():,.0f}")
        with col3:
            st.metric("Maximum Ridership", f"{df_prophet['y'].max():,.0f}")
        with col4:
            st.metric("Minimum Ridership", f"{df_prophet['y'].min():,.0f}")
    
    elif dashboard_section == "⚠️ Unusual Activity Analysis":
        st.markdown('<div class="section-header">⚠️ Comprehensive Unusual Activity Analysis</div>', unsafe_allow_html=True)
        
        # Unusual activity visualization
        fig = create_unusual_activity_analysis(full_comparison_df, chart_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        st.markdown("### 🔍 Unusual Activity Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Overall Statistics")
            unusual_low_count = len(full_comparison_df[full_comparison_df['Unusual_Activity'] == 'Unusually Low'])
            unusual_high_count = len(full_comparison_df[full_comparison_df['Unusual_Activity'] == 'Unusually High'])
            normal_count = len(full_comparison_df[full_comparison_df['Unusual_Activity'] == 'Normal'])
            
            st.write(f"• **Total days analyzed**: {total_days:,}")
            st.write(f"• **Normal activity**: {normal_count:,} days ({normal_count/total_days*100:.1f}%)")
            st.write(f"• **Unusually low**: {unusual_low_count:,} days ({unusual_low_count/total_days*100:.1f}%)")
            st.write(f"• **Unusually high**: {unusual_high_count:,} days ({unusual_high_count/total_days*100:.1f}%)")
        
        with col2:
            st.markdown("#### 📅 Year-over-Year Trends")
            for year in sorted(full_comparison_df['Year'].unique()):
                year_data = full_comparison_df[full_comparison_df['Year'] == year]
                low_pct = len(year_data[year_data['Unusual_Activity'] == 'Unusually Low']) / len(year_data) * 100
                high_pct = len(year_data[year_data['Unusual_Activity'] == 'Unusually High']) / len(year_data) * 100
                st.write(f"• **{year}**: {low_pct:.1f}% low, {high_pct:.1f}% high activity")
        
        # Most extreme days
        st.markdown("### 🎯 Most Extreme Days")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📉 Lowest Ridership Days")
            extreme_low = full_comparison_df[full_comparison_df['Unusual_Activity'] == 'Unusually Low'].nsmallest(5, 'Subway_Ridership')
            for _, row in extreme_low.iterrows():
                st.write(f"• **{row['Date'].strftime('%Y-%m-%d')}**: {row['Subway_Ridership']:,.0f} passengers")
        
        with col2:
            st.markdown("#### 📈 Highest Ridership Days")
            extreme_high = full_comparison_df[full_comparison_df['Unusual_Activity'] == 'Unusually High'].nlargest(5, 'Subway_Ridership')
            for _, row in extreme_high.iterrows():
                st.write(f"• **{row['Date'].strftime('%Y-%m-%d')}**: {row['Subway_Ridership']:,.0f} passengers")
    
    elif dashboard_section == "🎯 Threshold Analysis":
        st.markdown('<div class="section-header">🎯 Threshold Sensitivity Analysis</div>', unsafe_allow_html=True)
        
        # Threshold analysis
        fig, sensitivity_df = create_threshold_analysis(full_comparison_df, chart_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Threshold recommendations
        st.markdown("### 💡 Threshold Selection Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Threshold Comparison")
            for thresh in [0.1, 0.2, 0.3]:
                threshold_data = create_comparison_data(df_subway, forecast, thresh)
                total_unusual = len(threshold_data[threshold_data['Unusual_Activity'] != 'Normal'])
                pct_unusual = total_unusual / len(threshold_data) * 100
                
                st.write(f"**Threshold {thresh}**: {total_unusual:,} days ({pct_unusual:.1f}%)")
        
        with col2:
            st.markdown("#### 🔍 Sensitivity Insights")
            optimal_threshold = sensitivity_df.loc[
                (sensitivity_df['Unusual_Percentage'] >= 10) & 
                (sensitivity_df['Unusual_Percentage'] <= 25)
            ]['Threshold'].median()
            
            st.write(f"• **Recommended range**: 0.15 - 0.25")
            st.write(f"• **Optimal balance**: ~{optimal_threshold:.2f}")
            st.write(f"• **Current threshold ({threshold})**: {sensitivity_df[sensitivity_df['Threshold']==threshold]['Unusual_Percentage'].iloc[0]:.1f}% unusual")
    
    elif dashboard_section == "📋 Current Week Analysis":
        st.markdown('<div class="section-header">📋 Current Week Analysis</div>', unsafe_allow_html=True)
        
        # Get current week data
        latest_date = full_comparison_df['Date'].max()
        current_week_start = latest_date - pd.DateOffset(days=latest_date.dayofweek)
        current_week_end = current_week_start + pd.DateOffset(days=6)
        
        current_week_data = full_comparison_df[
            (full_comparison_df['Date'] >= current_week_start) & 
            (full_comparison_df['Date'] <= current_week_end)
        ].sort_values('Date')
        
        if len(current_week_data) > 0:
            # Current week visualization
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(
                    'Current Week: Actual vs Predicted Ridership',
                    'Prediction Error (Actual - Predicted)',
                    'Error Percentage'
                ),
                vertical_spacing=0.08
            )
            
            current_week_data['Error_Percentage'] = (current_week_data['Difference'] / current_week_data['Subway_Ridership']) * 100
            current_week_data['Day_Name'] = current_week_data['Date'].dt.day_name()
            
            # Main comparison plot
            fig.add_trace(
                go.Scatter(
                    x=current_week_data['Date'],
                    y=current_week_data['Subway_Ridership'],
                    mode='lines+markers',
                    name='Actual Ridership',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=8, symbol='circle')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=current_week_data['Date'],
                    y=current_week_data['Predicted_Ridership'],
                    mode='lines+markers',
                    name='Predicted Ridership',
                    line=dict(color='#F18F01', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ),
                row=1, col=1
            )
            
            # Error visualization
            colors = ['red' if x < 0 else 'green' for x in current_week_data['Difference']]
            fig.add_trace(
                go.Bar(
                    x=current_week_data['Date'],
                    y=current_week_data['Difference'],
                    name='Prediction Error',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Error percentage
            fig.add_trace(
                go.Bar(
                    x=current_week_data['Date'],
                    y=current_week_data['Error_Percentage'],
                    name='Error Percentage',
                    marker_color=['darkred' if x < -10 else 'red' if x < 0 else 'lightgreen' if x < 10 else 'darkgreen' 
                                 for x in current_week_data['Error_Percentage']],
                    opacity=0.7
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                title={
                    'text': 'Current Week Ridership Analysis: Model Performance Deep Dive',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                height=900,
                showlegend=True,
                template=chart_template,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            fig.update_yaxes(title_text="Daily Ridership", row=1, col=1)
            fig.update_yaxes(title_text="Error (Actual - Predicted)", row=2, col=1)
            fig.update_yaxes(title_text="Error Percentage (%)", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current week metrics
            st.markdown("### 📊 Current Week Performance Metrics")
            mae_week = np.mean(np.abs(current_week_data['Difference']))
            mape_week = np.mean(np.abs(current_week_data['Error_Percentage']))
            rmse_week = np.sqrt(np.mean(current_week_data['Difference']**2))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{mae_week:,.0f}")
            with col2:
                st.metric("Mean Absolute Percentage Error", f"{mape_week:.1f}%")
            with col3:
                st.metric("Root Mean Square Error", f"{rmse_week:,.0f}")
            
            # Day-by-day analysis
            st.markdown("### 📅 Day-by-Day Analysis")
            for _, row in current_week_data.iterrows():
                status = "✅ Good" if abs(row['Error_Percentage']) < 10 else "⚠️ Moderate" if abs(row['Error_Percentage']) < 20 else "❌ Poor"
                direction = "↗️" if row['Difference'] > 0 else "↘️"
                st.write(f"• **{row['Day_Name']}**: {direction} {row['Error_Percentage']:+.1f}% error - {status}")
        
        else:
            st.warning("No data available for the current week.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"**📊 Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**📈 Data Source**: MTA Daily Ridership and Traffic Data")
    st.markdown("**🤖 Model**: Facebook Prophet Time Series Forecasting")
    
    # Close theme wrapper
    if dark_mode or not dark_mode:  # Always close the div
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()