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
    page_title="NYC Subway Ridership Analysis Dashboard",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric > label {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the NYC subway ridership data"""
    try:
        # Try to load the data
        df = pd.read_csv(r'content\MTA_Daily_Ridership_and_Traffic__Beginning_2020_20250803.csv')
        
        # Filter for subway data
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
        
        return df_recent, df_subway
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

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
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        return model, forecast, df_prophet
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

def create_comparison_data(df_recent, forecast):
    """Create comparison data with unusual activity detection"""
    # Merge actual and predicted data
    full_comparison_df = pd.merge(
        df_recent, 
        forecast[['ds', 'yhat']], 
        left_on='Date', 
        right_on='ds', 
        how='inner'
    )
    full_comparison_df.rename(columns={'yhat': 'Predicted_Ridership'}, inplace=True)
    full_comparison_df.drop('ds', axis=1, inplace=True)
    full_comparison_df['Difference'] = full_comparison_df['Subway_Ridership'] - full_comparison_df['Predicted_Ridership']
    
    # Unusual activity detection
    threshold = 0.20
    full_comparison_df['Unusual_Activity'] = full_comparison_df.apply(
        lambda row: 'Unusually High' if row['Difference'] > row['Predicted_Ridership'] * threshold else (
            'Unusually Low' if row['Difference'] < -row['Predicted_Ridership'] * threshold else 'Normal'
        ), axis=1
    )
    
    return full_comparison_df

def create_executive_dashboard(full_comparison_df):
    """Create the executive dashboard visualization"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Historical vs Predicted Ridership',
            'Model Performance Metrics',
            'Unusual Activity Timeline',
            'Monthly Seasonality Patterns',
            'COVID-19 Impact Analysis',
            'Key Performance Indicators'
        ),
        specs=[[{}, {}, {}],
               [{}, {}, {"type": "indicator"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.06
    )
    
    # 1. Historical vs Predicted (last 365 days)
    recent_data = full_comparison_df.tail(365)
    fig.add_trace(
        go.Scatter(
            x=recent_data['Date'],
            y=recent_data['Subway_Ridership'],
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2),
            opacity=0.8
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=recent_data['Date'],
            y=recent_data['Predicted_Ridership'],
            mode='lines',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # 2. Model Performance Metrics
    mae = np.mean(np.abs(full_comparison_df['Difference']))
    mape = np.mean(np.abs(full_comparison_df['Difference'] / full_comparison_df['Subway_Ridership']) * 100)
    r2 = np.corrcoef(full_comparison_df['Subway_Ridership'], full_comparison_df['Predicted_Ridership'])[0,1]**2
    
    metrics = ['MAE', 'MAPE (%)', 'R²']
    values = [mae/1000, mape, r2*100]
    colors = ['#d62728' if v < 50 else '#ff7f0e' if v < 80 else '#2ca02c' for v in [mae/10000, 100-mape, r2*100]]
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}K' if i==0 else f'{v:.1f}%' if i==1 else f'{v:.1f}%' for i,v in enumerate(values)],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Unusual Activity Timeline
    unusual_monthly = full_comparison_df.groupby([full_comparison_df['Date'].dt.to_period('M'), 'Unusual_Activity']).size().unstack(fill_value=0)
    last_24_months = unusual_monthly.tail(24)
    months = [str(m) for m in last_24_months.index]
    
    for activity_type in ['Unusually Low', 'Unusually High']:
        if activity_type in last_24_months.columns:
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=last_24_months[activity_type],
                    mode='lines+markers',
                    name=activity_type,
                    line=dict(width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>Month:</b> %{{x}}<br><b>{activity_type}:</b> %{{y}} days<extra></extra>'
                ),
                row=1, col=3
            )
    
    # 4. Monthly Seasonality
    monthly_avg = full_comparison_df.groupby(full_comparison_df['Date'].dt.month)['Subway_Ridership'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.add_trace(
        go.Bar(
            x=month_names,
            y=monthly_avg.values/1000000,
            marker_color='lightblue',
            name='Avg Ridership',
            showlegend=False,
            text=[f'{v:.1f}M' for v in monthly_avg.values/1000000],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 5. COVID Impact Analysis
    covid_start = pd.to_datetime('2020-03-15')
    covid_recovery = pd.to_datetime('2021-06-01')
    
    pre_covid = full_comparison_df[full_comparison_df['Date'] < covid_start]['Subway_Ridership'].mean()
    during_covid = full_comparison_df[
        (full_comparison_df['Date'] >= covid_start) & 
        (full_comparison_df['Date'] < covid_recovery)
    ]['Subway_Ridership'].mean()
    post_covid = full_comparison_df[full_comparison_df['Date'] >= covid_recovery]['Subway_Ridership'].mean()
    
    periods = ['Pre-COVID', 'During COVID', 'Recovery']
    ridership = [pre_covid/1000000, during_covid/1000000, post_covid/1000000]
    
    fig.add_trace(
        go.Bar(
            x=periods,
            y=ridership,
            marker_color=['#2ca02c', '#d62728', '#ff7f0e'],
            text=[f'{v:.1f}M' for v in ridership],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 6. KPI Indicators
    current_ridership = full_comparison_df['Subway_Ridership'].iloc[-1]
    recovery_rate = (current_ridership / pre_covid) * 100
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=recovery_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "COVID Recovery %"},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 120]},
                'bar': {'color': "#2ca02c" if recovery_rate > 80 else "#ff7f0e" if recovery_rate > 60 else "#d62728"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"},
                    {'range': [100, 120], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'NYC Subway Ridership Analysis - Executive Dashboard',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'darkblue'}
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
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    # Update axes
    fig.update_yaxes(title_text="Ridership (Millions)", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Days Count", row=1, col=3)
    fig.update_xaxes(title_text="Month", tickangle=45, row=1, col=3)
    fig.update_yaxes(title_text="Avg Ridership (M)", row=2, col=1)
    fig.update_yaxes(title_text="Avg Ridership (M)", row=2, col=2)
    
    return fig, mae, mape, r2, recovery_rate, pre_covid, during_covid, post_covid

def main():
    # Header
    st.markdown('<h1 class="main-header">🚇 NYC Subway Ridership Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    with st.spinner("Loading and processing data..."):
        df_recent, df_subway = load_and_process_data()
    
    if df_recent is None:
        st.error("Failed to load data. Please check the data file path.")
        return
    
    # Train model
    with st.spinner("Training Prophet model..."):
        model, forecast, df_prophet = train_prophet_model(df_recent)
    
    if model is None:
        st.error("Failed to train model.")
        return
    
    # Create comparison data
    full_comparison_df = create_comparison_data(df_recent, forecast)
    
    # Sidebar metrics
    st.sidebar.markdown("### 📈 Key Metrics")
    total_days = len(full_comparison_df)
    avg_ridership = full_comparison_df['Subway_Ridership'].mean()
    unusual_stats = full_comparison_df['Unusual_Activity'].value_counts()
    
    st.sidebar.metric("Total Days Analyzed", f"{total_days:,}")
    st.sidebar.metric("Average Daily Ridership", f"{avg_ridership:,.0f}")
    st.sidebar.metric("Unusual Low Days", f"{unusual_stats.get('Unusually Low', 0):,}")
    st.sidebar.metric("Unusual High Days", f"{unusual_stats.get('Unusually High', 0):,}")
    
    # Main dashboard
    st.markdown("### Executive Dashboard")
    
    # Create and display the dashboard
    fig, mae, mape, r2, recovery_rate, pre_covid, during_covid, post_covid = create_executive_dashboard(full_comparison_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics in columns
    st.markdown("### 🎯 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Model Accuracy (MAPE)",
            value=f"{mape:.1f}%",
            delta=f"{'Excellent' if mape < 5 else 'Good' if mape < 10 else 'Acceptable'}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="COVID Recovery Rate",
            value=f"{recovery_rate:.1f}%",
            delta=f"{recovery_rate-100:+.1f}% vs Pre-COVID"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Current Daily Average",
            value=f"{full_comparison_df['Subway_Ridership'].tail(30).mean():,.0f}",
            delta=f"{((full_comparison_df['Subway_Ridership'].tail(30).mean()/avg_ridership-1)*100):+.1f}% vs Overall Avg"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Unusual Activity Rate",
            value=f"{((unusual_stats.get('Unusually Low', 0) + unusual_stats.get('Unusually High', 0))/total_days*100):.1f}%",
            delta="of total days"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed findings
    st.markdown("### 📋 Key Findings & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Data Insights")
        st.write(f"• **Dataset Coverage**: {total_days:,} days from {full_comparison_df['Date'].min().strftime('%Y-%m-%d')} to {full_comparison_df['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"• **Average Daily Ridership**: {avg_ridership:,.0f} passengers")
        st.write(f"• **Model Performance**: {mape:.1f}% MAPE indicates {'excellent' if mape < 5 else 'good' if mape < 10 else 'acceptable'} accuracy")
        st.write(f"• **Normal Activity**: {unusual_stats.get('Normal', 0):,} days ({unusual_stats.get('Normal', 0)/total_days*100:.1f}%)")
    
    with col2:
        st.markdown("#### 🎯 Business Recommendations")
        monthly_avg = full_comparison_df.groupby(full_comparison_df['Date'].dt.month)['Subway_Ridership'].mean()
        st.write(f"• **Seasonal Monitoring**: Month {monthly_avg.idxmax()} shows highest ridership")
        st.write(f"• **Alert System**: Use 20% threshold for unusual activity detection")
        st.write(f"• **Recovery Tracking**: {recovery_rate:.0f}% recovery indicates {'strong' if recovery_rate > 90 else 'moderate' if recovery_rate > 75 else 'slow'} recovery")
        st.write(f"• **Investigation Needed**: {unusual_stats.get('Unusually Low', 0)} days with unusual low activity")
    
    # COVID Analysis
    st.markdown("### 🦠 COVID-19 Impact Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Pre-COVID Average",
            value=f"{pre_covid:,.0f}",
            delta="Baseline"
        )
    
    with col2:
        st.metric(
            label="During COVID Average",
            value=f"{during_covid:,.0f}",
            delta=f"{((during_covid/pre_covid-1)*100):+.1f}%"
        )
        
    with col3:
        st.metric(
            label="Recovery Period Average",
            value=f"{post_covid:,.0f}",
            delta=f"{((post_covid/pre_covid-1)*100):+.1f}%"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Data Source**: MTA Daily Ridership and Traffic Data")

if __name__ == "__main__":
    main()