import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from datetime import datetime, timedelta

def plot_occupancy_trend(historical_data, days=1):
    """
    Create a line chart showing occupancy trends for the specified number of days.
    
    Parameters:
    - historical_data: DataFrame with historical parking data
    - days: Number of days to display (default: 1)
    
    Returns:
    - Plotly figure object
    """
    # Filter data for the last 'days' days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    mask = (historical_data['timestamp'] >= start_date) & (historical_data['timestamp'] <= end_date)
    filtered_data = historical_data.loc[mask].copy()
    
    # If no data in the filtered range, return an empty figure with a message
    if len(filtered_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected time period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Calculate occupancy percentage
    filtered_data['occupancy_pct'] = (filtered_data['occupancy'] / filtered_data['total_spaces']) * 100
    
    # Create line chart
    fig = px.line(
        filtered_data, 
        x='timestamp', 
        y='occupancy_pct',
        labels={'timestamp': 'Time', 'occupancy_pct': 'Occupancy (%)'},
        title=f'Parking Occupancy Trend for the Last {days} Day(s)'
    )
    
    # Add a horizontal line for high occupancy threshold (80%)
    fig.add_shape(
        type="line",
        x0=filtered_data['timestamp'].min(),
        y0=80,
        x1=filtered_data['timestamp'].max(),
        y1=80,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        )
    )
    
    # Add a horizontal line for moderate occupancy threshold (50%)
    fig.add_shape(
        type="line",
        x0=filtered_data['timestamp'].min(),
        y0=50,
        x1=filtered_data['timestamp'].max(),
        y1=50,
        line=dict(
            color="Orange",
            width=2,
            dash="dash",
        )
    )
    
    # Add annotation for thresholds
    fig.add_annotation(
        x=filtered_data['timestamp'].max(),
        y=82,
        text="High Occupancy",
        showarrow=False,
        yshift=10,
        font=dict(color="Red")
    )
    
    fig.add_annotation(
        x=filtered_data['timestamp'].max(),
        y=52,
        text="Moderate Occupancy",
        showarrow=False,
        yshift=10,
        font=dict(color="Orange")
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Occupancy (%)",
        yaxis_range=[0, 100],
        hovermode="x unified"
    )
    
    return fig

def plot_hourly_average(historical_data):
    """
    Create a grouped bar chart showing average occupancy by hour for each day of the week.
    
    Parameters:
    - historical_data: DataFrame with historical parking data
    
    Returns:
    - Plotly figure object
    """
    # Group by day of week and hour
    grouped_data = historical_data.groupby(['day_of_week', 'hour']).agg({
        'occupancy': 'mean',
        'total_spaces': 'first'
    }).reset_index()
    
    # Calculate occupancy percentage
    grouped_data['occupancy_pct'] = (grouped_data['occupancy'] / grouped_data['total_spaces']) * 100
    
    # Map day of week to day names
    day_map = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    
    grouped_data['day_name'] = grouped_data['day_of_week'].map(day_map)
    
    # Create grouped bar chart
    fig = px.bar(
        grouped_data, 
        x='hour', 
        y='occupancy_pct', 
        color='day_name',
        barmode='group',
        labels={'hour': 'Hour of Day', 'occupancy_pct': 'Average Occupancy (%)', 'day_name': 'Day of Week'},
        title='Average Hourly Occupancy by Day of Week'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Occupancy (%)",
        yaxis_range=[0, 100],
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2
        )
    )
    
    return fig

def create_parking_map(occupancy_data, center_lat=37.7749, center_lng=-122.4194):
    """
    Create an interactive map showing parking areas and their occupancy.
    
    Parameters:
    - occupancy_data: Dictionary with current occupancy data
    - center_lat: Center latitude for the map
    - center_lng: Center longitude for the map
    
    Returns:
    - Folium map object
    """
    # Create base map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=16)
    
    # Define parking areas with simulated coordinates
    areas = occupancy_data['areas']
    
    # Slightly adjust coordinates for each area to spread them out
    coordinates = {
        "Area A - Main": (center_lat + 0.002, center_lng + 0.002),
        "Area B - North": (center_lat + 0.002, center_lng - 0.002),
        "Area C - South": (center_lat - 0.002, center_lng + 0.002),
        "Area D - VIP": (center_lat - 0.002, center_lng - 0.002)
    }
    
    # Add markers for each area
    for area_name, area_data in areas.items():
        lat, lng = coordinates[area_name]
        
        # Determine color based on occupancy
        occupancy_pct = area_data['occupancy_pct']
        if occupancy_pct < 50:
            color = 'green'
        elif occupancy_pct < 80:
            color = 'orange'
        else:
            color = 'red'
        
        # Create popup content
        popup_content = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4>{area_name}</h4>
            <p><b>Occupancy:</b> {area_data['occupied']}/{area_data['total']} spaces</p>
            <p><b>Available:</b> {area_data['available']} spaces</p>
            <p><b>Occupancy Rate:</b> {area_data['occupancy_pct']:.1f}%</p>
        </div>
        """
        
        # Add marker
        folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=color, icon='car', prefix='fa')
        ).add_to(m)
        
        # Add circle to represent parking area size and occupancy
        folium.Circle(
            location=[lat, lng],
            radius=30 + (area_data['total'] * 0.5),  # Scale circle size based on number of spaces
            color=color,
            fill=True,
            fill_opacity=min(0.2 + (area_data['occupancy_pct'] / 100 * 0.6), 0.8),  # Opacity based on occupancy
            tooltip=f"{area_name}: {area_data['occupied']}/{area_data['total']} spaces occupied"
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 150px; height: 120px; 
        background-color: white; border:2px solid grey; z-index:9999; padding: 10px;
        font-size: 12px; font-family: Arial;">
        <p><i class="fa fa-circle" style="color:green;"></i> Low Occupancy (<50%)</p>
        <p><i class="fa fa-circle" style="color:orange;"></i> Moderate Occupancy (50-80%)</p>
        <p><i class="fa fa-circle" style="color:red;"></i> High Occupancy (>80%)</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

if __name__ == "__main__":
    # Test visualization functions with sample data
    from data_generator import generate_parking_data, get_current_occupancy
    
    # Generate sample data
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    historical_data = generate_parking_data(start_time, end_time)
    
    # Test occupancy trend plot
    fig = plot_occupancy_trend(historical_data)
    print("Occupancy trend plot created successfully")
    
    # Test hourly average plot
    fig = plot_hourly_average(historical_data)
    print("Hourly average plot created successfully")
    
    # Test parking map
    occupancy_data = get_current_occupancy()
    m = create_parking_map(occupancy_data)
    print("Parking map created successfully")
