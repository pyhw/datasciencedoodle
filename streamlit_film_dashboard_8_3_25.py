import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import st_folium

# Configure Streamlit page
st.set_page_config(
    page_title="NYC Film Permits Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the film permits data"""
    import os
    
    # Try different possible paths for the CSV file
    possible_paths = [
        'content/Film_Permits_20250801.csv',
        './content/Film_Permits_20250801.csv',
        'Film_Permits_20250801.csv',
        os.path.join(os.path.dirname(__file__), 'content', 'Film_Permits_20250801.csv'),
        os.path.join(os.getcwd(), 'content', 'Film_Permits_20250801.csv')
    ]
    
    film = None
    used_path = None
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                film = pd.read_csv(path)
                used_path = path
                st.success(f"✅ Data loaded successfully from: {path}")
                break
        except Exception as e:
            continue
    
    if film is None:
        st.error("❌ Could not find Film_Permits_20250801.csv")
        st.error("📁 Current working directory: " + os.getcwd())
        st.error("📋 Tried these paths:")
        for path in possible_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            st.error(f"   {exists} {path}")
        return None
    
    try:
        
        # Process dates
        film['StartDate'] = pd.to_datetime(film['StartDateTime']).dt.strftime('%Y-%m-%d')
        film['EndDate'] = pd.to_datetime(film['EndDateTime']).dt.strftime('%Y-%m-%d')
        film['StartDate_dt'] = pd.to_datetime(film['StartDate'])
        film['EndDate_dt'] = pd.to_datetime(film['EndDate'])
        film['Year'] = film['StartDate_dt'].dt.year
        
        # Rename column for consistency
        if 'SubCategory' in film.columns:
            film = film.rename(columns={'SubCategory': 'SubCategoryName'})
        
        return film
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_borough_coords():
    """NYC Borough coordinates for mapping"""
    return {
        'Manhattan': [40.7831, -73.9712],
        'Brooklyn': [40.6782, -73.9442],
        'Queens': [40.7282, -73.7949],
        'Bronx': [40.8448, -73.8648],
        'Staten Island': [40.5795, -74.1502]
    }

def create_heatmap(borough_counts, filtered_data):
    """Create a Folium heatmap with filming activity"""
    from folium.plugins import HeatMap
    borough_coords = get_borough_coords()
    
    # Create base map with dark theme for better heatmap visibility
    m = folium.Map(
        location=[40.7589, -73.9851],
        zoom_start=11,
        tiles='CartoDB dark_matter'
    )
    
    # Create heatmap data points
    heat_data = []
    
    for borough, count in borough_counts.items():
        if borough in borough_coords:
            lat, lon = borough_coords[borough]
            
            # Create multiple points around each borough center based on activity level
            # More points = more heat intensity
            num_points = min(count, 100)  # Cap at 100 points per borough
            
            for i in range(num_points):
                # Add random offset to create natural distribution
                lat_offset = np.random.normal(0, 0.008)  # ~0.5 mile radius
                lon_offset = np.random.normal(0, 0.008)
                
                # Weight the points based on filming activity
                weight = max(count / 50, 0.5)  # Minimum weight of 0.5
                
                heat_data.append([
                    lat + lat_offset, 
                    lon + lon_offset, 
                    weight
                ])
    
    # Add heatmap layer with custom styling
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=18,
        radius=20,
        blur=15,
        gradient={
            0.2: '#0066ff',    # Blue
            0.4: '#00ff66',    # Green  
            0.6: '#ffff00',    # Yellow
            0.8: '#ff6600',    # Orange
            1.0: '#ff0000'     # Red
        }
    ).add_to(m)
    
    # Add borough labels for reference
    for borough, count in borough_counts.items():
        if borough in borough_coords:
            folium.Marker(
                location=borough_coords[borough],
                popup=f"""
                <div style="font-family: Arial; font-size: 12px; text-align: center;">
                    <h4 style="margin: 5px 0; color: #333;">{borough}</h4>
                    <p style="margin: 2px 0;"><b>{count}</b> filming events</p>
                    <p style="margin: 2px 0;">{(count/len(filtered_data))*100:.1f}% of total</p>
                </div>
                """,
                tooltip=f"{borough}: {count} events",
                icon=folium.Icon(
                    color='white',
                    icon_color='black',
                    icon='video',
                    prefix='fa'
                )
            ).add_to(m)
    
    return m

def create_choropleth_map(borough_counts, filtered_data):
    """Create an alternative choropleth-style map"""
    borough_coords = get_borough_coords()
    
    # Create base map
    m = folium.Map(
        location=[40.7589, -73.9851],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Get max count for color scaling
    max_count = max(borough_counts.values()) if borough_counts.values() else 1
    
    # Create large circles for each borough (choropleth effect)
    for borough, count in borough_counts.items():
        if borough in borough_coords:
            # Color intensity based on count
            intensity = count / max_count
            
            # Color scale from light blue to red
            if intensity > 0.8:
                color = '#d73027'
            elif intensity > 0.6:
                color = '#fc8d59'
            elif intensity > 0.4:
                color = '#fee08b'
            elif intensity > 0.2:
                color = '#e0f3f8'
            else:
                color = '#91bfdb'
            
            # Create large circle
            folium.Circle(
                location=borough_coords[borough],
                radius=count * 50,  # Scale radius by count
                popup=f"""
                <div style="font-family: Arial; font-size: 14px; text-align: center; min-width: 150px;">
                    <h3 style="margin: 5px 0; color: #333;">{borough}</h3>
                    <hr style="margin: 5px 0;">
                    <p style="margin: 5px 0; font-size: 16px;"><b>{count}</b> events</p>
                    <p style="margin: 5px 0;">{(count/len(filtered_data))*100:.1f}% of total</p>
                    <p style="margin: 5px 0; font-size: 12px; color: #666;">
                        Rank: #{list(borough_counts.index).index(borough) + 1}
                    </p>
                </div>
                """,
                tooltip=f"{borough}: {count} events ({(count/len(filtered_data))*100:.1f}%)",
                color='white',
                weight=2,
                fillColor=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    
    return m

def main():
    # Title
    st.markdown('<h1 class="main-header">🎬 NYC Film Permits Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    film = load_data()
    if film is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">📊 Filters & Options</div>', unsafe_allow_html=True)
    
    # Year filter
    years = sorted(film['Year'].dropna().unique())
    selected_year = st.sidebar.selectbox(
        "Select Year",
        years,
        index=len(years)-1 if years else 0,
        help="Choose the year to analyze"
    )
    
    # Month filter for selected year
    year_data = film[film['Year'] == selected_year]
    months = sorted(year_data['StartDate_dt'].dt.month.unique())
    month_names = ['All'] + [datetime(2000, m, 1).strftime('%B') for m in months]
    month_numbers = [0] + months
    
    selected_month_idx = st.sidebar.selectbox(
        "Select Month",
        range(len(month_names)),
        format_func=lambda x: month_names[x],
        help="Choose a specific month or 'All' for the entire year"
    )
    
    selected_month = month_numbers[selected_month_idx]
    
    # Filter data based on selection
    if selected_month == 0:
        filtered_data = year_data
        period_name = f"{selected_year}"
    else:
        filtered_data = year_data[year_data['StartDate_dt'].dt.month == selected_month]
        period_name = f"{month_names[selected_month_idx]} {selected_year}"
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🗺️ Geographic Map", "📈 Analytics", "📋 Data Table", "🎯 Insights"])
    
    with tab1:
        st.header(f"📊 Overview - {period_name}")
        
        if len(filtered_data) > 0:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Events",
                    value=len(filtered_data),
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Unique Projects",
                    value=filtered_data['EventID'].nunique(),
                    delta=None
                )
            
            with col3:
                if 'Category' in filtered_data.columns:
                    top_category = filtered_data['Category'].value_counts().index[0]
                    st.metric(
                        label="Top Category",
                        value=top_category,
                        delta=None
                    )
            
            with col4:
                if 'Borough' in filtered_data.columns:
                    top_borough = filtered_data['Borough'].value_counts().index[0]
                    st.metric(
                        label="Top Borough",
                        value=top_borough,
                        delta=None
                    )
            
            # Charts Row 1
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Category' in filtered_data.columns:
                    category_counts = filtered_data['Category'].value_counts()
                    fig_cat = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title=f"Events by Category - {period_name}"
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                if 'Borough' in filtered_data.columns:
                    borough_counts = filtered_data['Borough'].value_counts()
                    fig_borough = px.bar(
                        x=borough_counts.index,
                        y=borough_counts.values,
                        title=f"Events by Borough - {period_name}",
                        labels={'x': 'Borough', 'y': 'Number of Events'}
                    )
                    st.plotly_chart(fig_borough, use_container_width=True)
            
            # Charts Row 2
            col3, col4 = st.columns(2)
            
            with col3:
                # Multi-Year Monthly Trends (only for full year view)
                if selected_month == 0:
                    st.subheader("Multi-Year Monthly Trends")
                    
                    # Get all available years
                    all_years = sorted(film['Year'].dropna().unique())
                    
                    # Create figure
                    fig_overview_trend = go.Figure()
                    
                    # Month labels
                    month_labels_overview = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    # Color palette for different years - matching Plotly default scheme
                    colors_overview = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                                     '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
                    
                    # Add trend line for each year
                    for i, year in enumerate(all_years):
                        year_data_trend = film[film['Year'] == year]
                        monthly_data = year_data_trend.groupby(year_data_trend['StartDate_dt'].dt.month).size()
                        
                        # Only include months that have data (don't fill with 0)
                        month_names_filtered = []
                        monthly_values = []
                        
                        for month in range(1, 13):
                            if month in monthly_data.index:
                                monthly_values.append(monthly_data[month])
                                month_names_filtered.append(month_labels_overview[month-1])
                        
                        # Determine line style and width
                        is_selected_year = (year == selected_year)
                        line_width = 4 if is_selected_year else 2
                        line_color = colors_overview[i % len(colors_overview)]
                        
                        # Only add trace if there's data for this year
                        if monthly_values:
                            fig_overview_trend.add_trace(go.Scatter(
                                x=month_names_filtered,
                                y=monthly_values,
                                mode='lines+markers',
                                name=f"{year}" + (" (Selected)" if is_selected_year else ""),
                                line=dict(
                                    color=line_color,
                                    width=line_width
                                ),
                                marker=dict(
                                    size=6 if is_selected_year else 4,
                                    color=line_color
                                ),
                                hovertemplate=f"<b>{year}</b><br>" +
                                            "Month: %{x}<br>" +
                                            "Events: %{y}<br>" +
                                            "<extra></extra>",
                                connectgaps=False  # Don't connect lines across missing data
                            ))
                    
                    # Update layout
                    fig_overview_trend.update_layout(
                        title=f"Monthly Trends - All Years (Selected: <b>{selected_year}</b>)",
                        xaxis_title="Month",
                        yaxis_title="Number of Events",
                        height=400,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    # Set x-axis to show all months properly
                    fig_overview_trend.update_xaxes(
                        categoryorder='array',
                        categoryarray=month_labels_overview,
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='lightgray'
                    )
                    fig_overview_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    
                    st.plotly_chart(fig_overview_trend, use_container_width=True)
                
                else:
                    # For specific month selection, show year comparison
                    st.subheader(f"{month_names[selected_month_idx]} Trends Across Years")
                    
                    # Get data for the selected month across all years
                    month_year_data = []
                    for year in sorted(film['Year'].dropna().unique()):
                        year_month_data = film[(film['Year'] == year) & 
                                             (film['StartDate_dt'].dt.month == selected_month)]
                        count = len(year_month_data)
                        if count > 0:  # Only include years with data for this month
                            month_year_data.append({'Year': year, 'Events': count})
                    
                    if month_year_data:
                        month_df = pd.DataFrame(month_year_data)
                        
                        # Create colors for bars, highlighting selected year
                        colors_bars = ['#EF553B' if year == selected_year else '#636EFA' 
                                     for year in month_df['Year']]
                        
                        fig_month_overview = px.bar(
                            month_df,
                            x='Year',
                            y='Events',
                            title=f"{month_names[selected_month_idx]} Filming Events by Year",
                            labels={'Events': 'Number of Events'},
                            color='Events',
                            color_continuous_scale='viridis'
                        )
                        
                        # Update bar colors to highlight selected year
                        fig_month_overview.update_traces(
                            marker_color=colors_bars,
                            marker_line_color='white',
                            marker_line_width=2
                        )
                        
                        fig_month_overview.update_layout(height=400)
                        st.plotly_chart(fig_month_overview, use_container_width=True)
            
            with col4:
                # Selected Year Film Location Map (Borough-based)
                if 'Borough' in filtered_data.columns and len(filtered_data) > 0:
                    st.subheader("📍 Filming Locations")
                    
                    borough_coords = get_borough_coords()
                    borough_counts_map = filtered_data['Borough'].value_counts()
                    
                    # Create map data
                    map_data_overview = []
                    for borough, count in borough_counts_map.items():
                        if borough in borough_coords:
                            map_data_overview.append({
                                'Borough': borough,
                                'Count': count,
                                'Lat': borough_coords[borough][0],
                                'Lon': borough_coords[borough][1],
                                'Size': max(count * 2, 8)
                            })
                    
                    if map_data_overview:
                        map_df_overview = pd.DataFrame(map_data_overview)
                        
                        fig_map_overview = px.scatter_mapbox(
                            map_df_overview,
                            lat='Lat',
                            lon='Lon',
                            size='Size',
                            color='Count',
                            hover_name='Borough',
                            hover_data={'Count': True, 'Lat': False, 'Lon': False, 'Size': False},
                            color_continuous_scale='viridis',
                            title=f'Filming Locations - {period_name}',
                            zoom=9,
                            height=400
                        )
                        
                        fig_map_overview.update_layout(
                            mapbox_style="open-street-map",
                            margin={"r":0,"t":30,"l":0,"b":0}
                        )
                        
                        st.plotly_chart(fig_map_overview, use_container_width=True)
                    
                    # Borough stats summary
                    st.markdown("**Top Locations:**")
                    for i, (borough, count) in enumerate(borough_counts_map.head(3).items(), 1):
                        percentage = (count / len(filtered_data)) * 100
                        st.write(f"{i}. **{borough}**: {count} ({percentage:.1f}%)")
                else:
                    st.info("No location data available for mapping")
        else:
            st.warning(f"No data available for {period_name}")
    
    with tab2:
        st.header(f"🗺️ Geographic Distribution - {period_name}")
        
        if len(filtered_data) > 0 and 'Borough' in filtered_data.columns:
            borough_counts = filtered_data['Borough'].value_counts()
            borough_coords = get_borough_coords()
            
            # Create two columns for different map types
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Interactive Borough Map")
                
                # Plotly map
                map_data = []
                for borough, count in borough_counts.items():
                    if borough in borough_coords:
                        map_data.append({
                            'Borough': borough,
                            'Count': count,
                            'Lat': borough_coords[borough][0],
                            'Lon': borough_coords[borough][1],
                            'Size': max(count * 3, 15)
                        })
                
                if map_data:
                    map_df = pd.DataFrame(map_data)
                    
                    fig_map = px.scatter_mapbox(
                        map_df,
                        lat='Lat',
                        lon='Lon',
                        size='Size',
                        color='Count',
                        hover_name='Borough',
                        hover_data={'Count': True, 'Lat': False, 'Lon': False, 'Size': False},
                        color_continuous_scale='viridis',
                        title=f'Filming Events by Borough - {period_name}',
                        zoom=10,
                        height=500
                    )
                    
                    fig_map.update_layout(
                        mapbox_style="open-street-map",
                        margin={"r":0,"t":30,"l":0,"b":0}
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                
            
            with col2:
                st.subheader("Borough Statistics")
                
                for borough, count in borough_counts.items():
                    percentage = (count / len(filtered_data)) * 100
                    st.metric(
                        label=borough,
                        value=f"{count} events",
                        delta=f"{percentage:.1f}%"
                    )
                
                # Zipcode analysis
                if 'ZipCode(s)' in filtered_data.columns:
                    st.subheader("Top Zip Codes")
                    zipcode_data = []
                    for idx, row in filtered_data.iterrows():
                        if pd.notna(row['ZipCode(s)']):
                            zips = str(row['ZipCode(s)']).split(',')
                            for zip_code in zips:
                                cleaned_zip = zip_code.strip()
                                if cleaned_zip and len(cleaned_zip) == 5 and cleaned_zip.isdigit():
                                    zipcode_data.append(cleaned_zip)
                    
                    if zipcode_data:
                        zipcode_counts = pd.Series(zipcode_data).value_counts().head(10)
                        for zipcode, count in zipcode_counts.items():
                            st.write(f"**{zipcode}**: {count} events")
        else:
            st.warning("No geographic data available for mapping")
    
    with tab3:
        st.header(f"📈 Detailed Analytics - {period_name}")
        
        if len(filtered_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Event types
                if 'EventType' in filtered_data.columns:
                    event_counts = filtered_data['EventType'].value_counts()
                    fig_events = px.bar(
                        x=event_counts.values,
                        y=event_counts.index,
                        orientation='h',
                        title="Events by Type",
                        labels={'x': 'Number of Events', 'y': 'Event Type'}
                    )
                    st.plotly_chart(fig_events, use_container_width=True)
                
                # Multi-year monthly trend (if full year selected)
                if selected_month == 0:
                    st.subheader("Multi-Year Monthly Trends")
                    
                    # Get all available years
                    all_years = sorted(film['Year'].dropna().unique())
                    
                    # Create figure
                    fig_trend = go.Figure()
                    
                    # Month labels
                    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    # Color palette for different years
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    # Add trend line for each year
                    for i, year in enumerate(all_years):
                        year_data_trend = film[film['Year'] == year]
                        monthly_data = year_data_trend.groupby(year_data_trend['StartDate_dt'].dt.month).size()
                        
                        # Only include months that have data (don't fill with 0)
                        month_indices = []
                        monthly_values = []
                        month_names_filtered = []
                        
                        for month in range(1, 13):
                            if month in monthly_data.index:
                                month_indices.append(month)
                                monthly_values.append(monthly_data[month])
                                month_names_filtered.append(month_labels[month-1])
                        
                        # Determine line style and width
                        is_selected_year = (year == selected_year)
                        line_width = 4 if is_selected_year else 2
                        line_color = colors[i % len(colors)]
                        
                        # Only add trace if there's data for this year
                        if monthly_values:
                            fig_trend.add_trace(go.Scatter(
                                x=month_names_filtered,
                                y=monthly_values,
                                mode='lines+markers',
                                name=f"{year}" + (" (Selected)" if is_selected_year else ""),
                                line=dict(
                                    color=line_color,
                                    width=line_width
                                ),
                                marker=dict(
                                    size=8 if is_selected_year else 6,
                                    color=line_color
                                ),
                                hovertemplate=f"<b>{year}</b><br>" +
                                            "Month: %{x}<br>" +
                                            "Events: %{y}<br>" +
                                            "<extra></extra>",
                                connectgaps=False  # Don't connect lines across missing data
                            ))
                    
                    # Update layout
                    fig_trend.update_layout(
                        title=f"Monthly Filming Trends Across Years (Selected: <b>{selected_year}</b>)",
                        xaxis_title="Month",
                        yaxis_title="Number of Events",
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Set x-axis to show all months properly
                    fig_trend.update_xaxes(
                        categoryorder='array',
                        categoryarray=month_labels,
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='lightgray'
                    )
                    fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Add insights below the chart
                    with st.expander("📊 Trend Analysis Insights"):
                        selected_year_data = film[film['Year'] == selected_year]
                        selected_monthly = selected_year_data.groupby(selected_year_data['StartDate_dt'].dt.month).size()
                        
                        if len(selected_monthly) > 0:
                            peak_month = selected_monthly.idxmax()
                            peak_value = selected_monthly.max()
                            low_month = selected_monthly.idxmin()
                            low_value = selected_monthly.min()
                            
                            st.write(f"**{selected_year} Highlights:**")
                            st.write(f"• **Peak Month**: {month_labels[peak_month-1]} with {peak_value} events")
                            st.write(f"• **Lowest Month**: {month_labels[low_month-1]} with {low_value} events")
                            st.write(f"• **Total Events**: {selected_monthly.sum()} for the year")
                            
                            # Year-over-year comparison
                            if len(all_years) > 1:
                                prev_years = [y for y in all_years if y < selected_year]
                                if prev_years:
                                    prev_year = max(prev_years)
                                    prev_year_data = film[film['Year'] == prev_year]
                                    prev_total = len(prev_year_data)
                                    current_total = len(selected_year_data)
                                    
                                    if prev_total > 0:
                                        change = ((current_total - prev_total) / prev_total) * 100
                                        direction = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
                                        st.write(f"• **vs {prev_year}**: {direction} {change:+.1f}% change ({current_total} vs {prev_total} events)")
                else:
                    # Single month view - show comparison across years for that month
                    st.subheader(f"{month_names[selected_month_idx]} Trends Across Years")
                    
                    # Get data for the selected month across all years
                    month_year_data = []
                    for year in sorted(film['Year'].dropna().unique()):
                        year_month_data = film[(film['Year'] == year) & 
                                             (film['StartDate_dt'].dt.month == selected_month)]
                        count = len(year_month_data)
                        month_year_data.append({'Year': year, 'Events': count})
                    
                    if month_year_data:
                        month_df = pd.DataFrame(month_year_data)
                        
                        fig_month_trend = px.bar(
                            month_df,
                            x='Year',
                            y='Events',
                            title=f"{month_names[selected_month_idx]} Filming Events by Year",
                            labels={'Events': 'Number of Events'},
                            color='Events',
                            color_continuous_scale='viridis'
                        )
                        
                        # Highlight selected year
                        fig_month_trend.update_traces(
                            marker_line_color='red',
                            marker_line_width=3,
                            selector=dict(type='bar')
                        )
                        
                        st.plotly_chart(fig_month_trend, use_container_width=True)
            
            with col2:
                # Sub-categories
                if 'SubCategoryName' in filtered_data.columns:
                    subcat_counts = filtered_data['SubCategoryName'].value_counts().head(10)
                    fig_subcat = px.bar(
                        x=subcat_counts.values,
                        y=subcat_counts.index,
                        orientation='h',
                        title="Top 10 Sub-Categories",
                        labels={'x': 'Number of Events', 'y': 'Sub-Category'}
                    )
                    st.plotly_chart(fig_subcat, use_container_width=True)
                
                # Day of week analysis
                filtered_data['DayOfWeek'] = filtered_data['StartDate_dt'].dt.day_name()
                dow_counts = filtered_data['DayOfWeek'].value_counts()
                dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_counts = dow_counts.reindex(dow_order, fill_value=0)
                
                fig_dow = px.bar(
                    x=dow_counts.index,
                    y=dow_counts.values,
                    title="Events by Day of Week",
                    labels={'x': 'Day of Week', 'y': 'Number of Events'}
                )
                st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.warning("No data available for analytics")
    
    with tab4:
        st.header(f"📋 Data Table - {period_name}")
        
        if len(filtered_data) > 0:
            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_rows = st.selectbox("Rows to show", [10, 25, 50, 100], index=1)
            with col2:
                if 'Borough' in filtered_data.columns:
                    borough_filter = st.selectbox(
                        "Filter by Borough",
                        ['All'] + list(filtered_data['Borough'].unique())
                    )
                else:
                    borough_filter = 'All'
            with col3:
                if 'Category' in filtered_data.columns:
                    category_filter = st.selectbox(
                        "Filter by Category",
                        ['All'] + list(filtered_data['Category'].unique())
                    )
                else:
                    category_filter = 'All'
            
            # Apply filters
            display_data = filtered_data.copy()
            if borough_filter != 'All':
                display_data = display_data[display_data['Borough'] == borough_filter]
            if category_filter != 'All':
                display_data = display_data[display_data['Category'] == category_filter]
            
            # Select columns to display
            columns_to_show = ['EventID', 'EventType', 'StartDate', 'EndDate', 'Borough', 'Category']
            if 'SubCategoryName' in display_data.columns:
                columns_to_show.append('SubCategoryName')
            
            # Display table
            st.dataframe(
                display_data[columns_to_show].head(show_rows),
                use_container_width=True
            )
            
            # Download button
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"nyc_filming_{period_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data available to display")
    
    with tab5:
        st.header(f"🎯 Key Insights - {period_name}")
        
        if len(filtered_data) > 0:
            st.subheader("📈 Highlights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎬 Production Activity")
                if 'Category' in filtered_data.columns:
                    category_counts = filtered_data['Category'].value_counts()
                    for category, count in category_counts.items():
                        percentage = (count / len(filtered_data)) * 100
                        st.write(f"**{category}**: {count} events ({percentage:.1f}%)")
            
            with col2:
                st.markdown("### 📍 Location Distribution")
                if 'Borough' in filtered_data.columns:
                    borough_counts = filtered_data['Borough'].value_counts()
                    for borough, count in borough_counts.items():
                        percentage = (count / len(filtered_data)) * 100
                        st.write(f"**{borough}**: {count} events ({percentage:.1f}%)")
            
            # Notable trends
            st.subheader("🌟 Notable Trends")
            
            insights = []
            
            if 'Category' in filtered_data.columns:
                top_category = filtered_data['Category'].value_counts().index[0]
                insights.append(f"📺 **{top_category}** dominates production activity")
            
            if 'Borough' in filtered_data.columns:
                top_borough = filtered_data['Borough'].value_counts().index[0]
                borough_pct = (filtered_data['Borough'].value_counts().iloc[0] / len(filtered_data)) * 100
                insights.append(f"🏙️ **{top_borough}** leads with {borough_pct:.1f}% of all filming")
            
            if 'EventType' in filtered_data.columns:
                top_event = filtered_data['EventType'].value_counts().index[0]
                insights.append(f"🎯 **{top_event}** is the most common event type")
            
            for insight in insights:
                st.write(f"• {insight}")
            
            # Summary stats
            st.subheader("📊 Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Events", len(filtered_data))
                st.metric("Unique Projects", filtered_data['EventID'].nunique())
            
            with col2:
                if 'Category' in filtered_data.columns:
                    st.metric("Categories", filtered_data['Category'].nunique())
                if 'Borough' in filtered_data.columns:
                    st.metric("Boroughs", filtered_data['Borough'].nunique())
            
            with col3:
                if 'SubCategoryName' in filtered_data.columns:
                    st.metric("Sub-Categories", filtered_data['SubCategoryName'].nunique())
                if 'EventType' in filtered_data.columns:
                    st.metric("Event Types", filtered_data['EventType'].nunique())
        else:
            st.warning("No data available for insights")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Source**: NYC Film Permits")
    st.sidebar.markdown("**Last Updated**: August 2025")
    st.sidebar.markdown("🎬 Built with Streamlit")

if __name__ == "__main__":
    main()