import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64

def load_svg(svg_file):
    """
    Load an SVG file and return it as HTML for display in Streamlit.
    
    Parameters:
    - svg_file: Path to the SVG file
    
    Returns:
    - HTML string with the SVG content
    """
    try:
        with open(svg_file, 'r') as f:
            svg_content = f.read()
            return svg_content
    except Exception as e:
        # If file can't be loaded, return a simple SVG
        default_svg = """
        <svg width="50" height="50" xmlns="http://www.w3.org/2000/svg">
            <rect width="50" height="50" style="fill:blue;stroke-width:3;stroke:rgb(0,0,0)" />
            <text x="25" y="25" font-family="Arial" font-size="20" text-anchor="middle" fill="white">P</text>
        </svg>
        """
        return default_svg

def calculate_statistics(historical_data):
    """
    Calculate various statistics from historical parking data.
    
    Parameters:
    - historical_data: DataFrame with historical parking data
    
    Returns:
    - Dictionary with calculated statistics
    """
    # Get today's data
    today = datetime.now().date()
    today_data = historical_data[historical_data['timestamp'].dt.date == today]
    
    # Get last 7 days data
    last_week = today - timedelta(days=7)
    last_week_data = historical_data[historical_data['timestamp'].dt.date >= last_week]
    
    # Calculate statistics
    stats = {}
    
    # Calculate today's statistics if data exists
    if len(today_data) > 0:
        stats['avg_today'] = today_data['occupancy'].mean()
        stats['peak_today'] = today_data['occupancy'].max()
        stats['min_today'] = today_data['occupancy'].min()
        
        # Get total spaces (assuming consistent throughout the day)
        total_spaces = today_data['total_spaces'].iloc[0]
        stats['avg_today_pct'] = (stats['avg_today'] / total_spaces) * 100
        stats['peak_today_pct'] = (stats['peak_today'] / total_spaces) * 100
        stats['min_today_pct'] = (stats['min_today'] / total_spaces) * 100
    else:
        # Default values if no data for today
        total_spaces = historical_data['total_spaces'].iloc[0] if len(historical_data) > 0 else 200
        stats['avg_today'] = 0
        stats['peak_today'] = 0
        stats['min_today'] = 0
        stats['avg_today_pct'] = 0
        stats['peak_today_pct'] = 0
        stats['min_today_pct'] = 0
    
    # Calculate weekly statistics
    if len(last_week_data) > 0:
        stats['avg_week'] = last_week_data['occupancy'].mean()
        stats['peak_week'] = last_week_data['occupancy'].max()
        stats['min_week'] = last_week_data['occupancy'].min()
        
        stats['avg_pct'] = (stats['avg_week'] / total_spaces) * 100
        stats['peak_pct'] = (stats['peak_week'] / total_spaces) * 100
        stats['min_pct'] = (stats['min_week'] / total_spaces) * 100
    else:
        stats['avg_week'] = 0
        stats['peak_week'] = 0
        stats['min_week'] = 0
        stats['avg_pct'] = 0
        stats['peak_pct'] = 0
        stats['min_pct'] = 0
    
    # Calculate busy times
    if len(last_week_data) > 0:
        # Group by hour and calculate average occupancy
        hourly_avg = last_week_data.groupby('hour')['occupancy'].mean()
        stats['busiest_hour'] = hourly_avg.idxmax()
        stats['quietest_hour'] = hourly_avg.idxmin()
        
        # Group by day of week and calculate average occupancy
        daily_avg = last_week_data.groupby('day_of_week')['occupancy'].mean()
        stats['busiest_day'] = daily_avg.idxmax()
        stats['quietest_day'] = daily_avg.idxmin()
        
        # Convert day of week to day name
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        stats['busiest_day_name'] = day_names[stats['busiest_day']]
        stats['quietest_day_name'] = day_names[stats['quietest_day']]
    else:
        stats['busiest_hour'] = 12  # Default to noon
        stats['quietest_hour'] = 3   # Default to 3 AM
        stats['busiest_day'] = 2     # Default to Wednesday
        stats['quietest_day'] = 6    # Default to Sunday
        stats['busiest_day_name'] = 'Wednesday'
        stats['quietest_day_name'] = 'Sunday'
    
    return stats

def generate_recommendations(current_occupancy, predicted_occupancy, total_spaces):
    """
    Generate parking recommendations based on current and predicted occupancy.
    
    Parameters:
    - current_occupancy: Current number of occupied parking spaces
    - predicted_occupancy: Predicted number of occupied parking spaces
    - total_spaces: Total number of parking spaces
    
    Returns:
    - Dictionary with recommendations
    """
    current_pct = (current_occupancy / total_spaces) * 100
    predicted_pct = (predicted_occupancy / total_spaces) * 100
    
    recommendations = {}
    
    # Determine current status
    if current_pct < 50:
        recommendations['current_status'] = "Low occupancy - Plenty of parking available"
        recommendations['current_color'] = "green"
    elif current_pct < 80:
        recommendations['current_status'] = "Moderate occupancy - Parking available but filling up"
        recommendations['current_color'] = "orange"
    else:
        recommendations['current_status'] = "High occupancy - Limited parking available"
        recommendations['current_color'] = "red"
    
    # Determine trend
    if predicted_pct - current_pct > 10:
        recommendations['trend'] = "Occupancy increasing significantly"
        recommendations['trend_icon'] = "↑↑"
    elif predicted_pct - current_pct > 5:
        recommendations['trend'] = "Occupancy increasing"
        recommendations['trend_icon'] = "↑"
    elif predicted_pct - current_pct < -10:
        recommendations['trend'] = "Occupancy decreasing significantly"
        recommendations['trend_icon'] = "↓↓"
    elif predicted_pct - current_pct < -5:
        recommendations['trend'] = "Occupancy decreasing"
        recommendations['trend_icon'] = "↓"
    else:
        recommendations['trend'] = "Occupancy stable"
        recommendations['trend_icon'] = "→"
    
    # Generate recommendation
    if predicted_pct >= 90:
        recommendations['recommendation'] = "Parking will be very limited. Consider alternative transportation or arriving early."
    elif predicted_pct >= 75:
        recommendations['recommendation'] = "Parking may be difficult to find during peak hours. Plan accordingly."
    elif predicted_pct >= 50:
        recommendations['recommendation'] = "Moderate parking availability expected. Some searching may be required."
    else:
        recommendations['recommendation'] = "Good parking availability expected. No special measures needed."
    
    return recommendations

if __name__ == "__main__":
    # Test SVG loading
    svg_content = load_svg("assets/parking_icon.svg")
    print("SVG content loaded successfully" if svg_content else "SVG loading failed")
    
    # Test statistics calculation with sample data
    from data_generator import generate_parking_data
    
    # Generate sample data
    start_time = datetime.now() - timedelta(days=10)
    end_time = datetime.now()
    historical_data = generate_parking_data(start_time, end_time)
    
    # Calculate statistics
    stats = calculate_statistics(historical_data)
    print("\nCalculated statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test recommendations
    recommendations = generate_recommendations(150, 180, 200)
    print("\nRecommendations:")
    for key, value in recommendations.items():
        print(f"{key}: {value}")
