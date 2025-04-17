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

def create_parking_map(occupancy_data, center_lat=28.0609, center_lng=-82.4131):
    """
    Create an interactive map showing USF parking garages and their occupancy.
    
    Parameters:
    - occupancy_data: Dictionary with current occupancy data
    - center_lat: Center latitude for the map (default: USF Tampa campus)
    - center_lng: Center longitude for the map (default: USF Tampa campus)
    
    Returns:
    - Folium map object
    """
    # Create base map centered on USF Tampa campus
    m = folium.Map(location=[center_lat, center_lng], zoom_start=15)
    
    # Add a satellite layer as an option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite'
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add USF marker at the center
    folium.Marker(
        [center_lat, center_lng],
        popup="USF Tampa Campus",
        icon=folium.Icon(color='green', icon='graduation-cap', prefix='fa')
    ).add_to(m)
    
    # Define parking areas with USF garage coordinates
    areas = occupancy_data['areas']
    
    # Create garage-specific markers first
    garages = [
        {
            "name": "Collins Garage",
            "location": [28.0587, -82.4139],
            "total_spaces": 1800
        },
        {
            "name": "Beard Garage",
            "location": [28.0650, -82.4144],
            "total_spaces": 1500
        },
        {
            "name": "Laurel Garage",
            "location": [28.0622, -82.4099],
            "total_spaces": 1700
        },
        {
            "name": "Crescent Hill Garage",
            "location": [28.0643, -82.4119],
            "total_spaces": 1600
        }
    ]
    
    # Add markers for each garage
    for garage in garages:
        folium.Marker(
            location=garage["location"],
            popup=f"<b>{garage['name']}</b><br>Total Spaces: {garage['total_spaces']}",
            icon=folium.Icon(color='blue', icon='building', prefix='fa')
        ).add_to(m)
    
    # USF parking zone coordinates - use more specific names to avoid duplicate keys
    coordinates = {
        # Collins Garage areas
        "Collins - Gold Zone": (28.0587, -82.4139),
        "Collins - Green Zone": (28.0582, -82.4134),
        "Collins - Resident Zone": (28.0592, -82.4144),
        "Collins - Non-Resident Zone": (28.0592, -82.4134),
        
        # Beard Garage areas
        "Beard - Staff Zone": (28.0650, -82.4149),
        "Beard - Student Zone": (28.0650, -82.4139),
        "Beard - Visitor Zone": (28.0645, -82.4144),
        "Beard - Reserved Zone": (28.0645, -82.4139),
        
        # Laurel Garage areas
        "Laurel - Gold Zone": (28.0622, -82.4104),
        "Laurel - Green Zone": (28.0622, -82.4094),
        "Laurel - Visitor Zone": (28.0627, -82.4099),
        
        # Crescent Hill Garage areas
        "Crescent - Staff Zone": (28.0643, -82.4119),
        "Crescent - Student Zone": (28.0643, -82.4114),
        "Crescent - Visitor Zone": (28.0638, -82.4119)
    }
    
    # If we need to map old area names to the new ones
    area_name_mapping = {
        "Gold Zone": "Collins - Gold Zone",
        "Green Zone": "Collins - Green Zone",
        "Resident Zone": "Collins - Resident Zone",
        "Non-Resident Zone": "Collins - Non-Resident Zone",
        "Staff Zone": "Beard - Staff Zone",
        "Student Zone": "Beard - Student Zone",
        "Visitor Zone": "Beard - Visitor Zone",
        "Reserved Zone": "Beard - Reserved Zone"
    }
    
    # Add markers for each area
    for area_name, area_data in areas.items():
        # Map old area names to new USF area names if needed
        usf_area_name = area_name_mapping.get(area_name, area_name)
        
        # Skip if we don't have coordinates for this area
        if usf_area_name not in coordinates:
            continue
            
        lat, lng = coordinates[usf_area_name]
        
        # Determine color based on occupancy
        occupancy_pct = area_data['occupancy_pct']
        if occupancy_pct < 50:
            color = 'green'
        elif occupancy_pct < 80:
            color = 'orange'
        else:
            color = 'red'
        
        # Create popup content with USF branding
        popup_content = f"""
        <div style="font-family: Arial; width: 220px; border-top: 4px solid #006747;">
            <h4 style="color: #006747;">{usf_area_name}</h4>
            <p><b>Occupancy:</b> {area_data['occupied']}/{area_data['total']} spaces</p>
            <p><b>Available:</b> {area_data['available']} spaces</p>
            <p><b>Occupancy Rate:</b> {area_data['occupancy_pct']:.1f}%</p>
            <p style="color: #006747; font-size: 11px; text-align: right;">USF Parking System</p>
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
            tooltip=f"{usf_area_name}: {area_data['occupied']}/{area_data['total']} spaces occupied"
        ).add_to(m)
    
    # Add USF branded legend
    legend_html = """
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 180px; height: 150px; 
        background-color: white; border:2px solid #006747; z-index:9999; padding: 12px;
        font-size: 12px; font-family: Arial; border-radius: 5px;">
        <div style="border-bottom: 2px solid #CFC493; margin-bottom: 8px; padding-bottom: 5px;">
            <strong style="color: #006747;">USF Parking Status</strong>
        </div>
        <p><i class="fa fa-circle" style="color:green;"></i> Low Occupancy (<50%)</p>
        <p><i class="fa fa-circle" style="color:orange;"></i> Moderate Occupancy (50-80%)</p>
        <p><i class="fa fa-circle" style="color:red;"></i> High Occupancy (>80%)</p>
        <p><i class="fa fa-building" style="color:blue;"></i> Parking Garage</p>
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
