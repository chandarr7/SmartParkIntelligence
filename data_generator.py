import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_parking_data(start_time, end_time, interval_minutes=15, total_spaces=200):
    """
    Generate simulated parking data for a given time period.
    
    Parameters:
    - start_time: datetime object for the start of the data generation
    - end_time: datetime object for the end of the data generation
    - interval_minutes: interval between data points in minutes
    - total_spaces: total number of parking spaces
    
    Returns:
    - DataFrame with simulated parking data
    """
    # Calculate number of time points
    time_diff = end_time - start_time
    total_minutes = time_diff.total_seconds() / 60
    num_points = int(total_minutes / interval_minutes) + 1
    
    # Create time points
    time_points = [start_time + timedelta(minutes=i*interval_minutes) for i in range(num_points)]
    
    # Base occupancy patterns
    # Weekday pattern: peak at 9-10 AM and 4-6 PM
    # Weekend pattern: peak at 11 AM - 3 PM
    
    data = []
    for t in time_points:
        day_of_week = t.weekday()  # 0-6 where 0 is Monday
        hour = t.hour
        minute = t.minute
        
        # Base occupancy percentage (30-40%)
        base_occupancy_pct = np.random.uniform(0.3, 0.4)
        
        # Add time-based patterns
        if day_of_week < 5:  # Weekday
            # Morning peak (8-10 AM)
            if 8 <= hour < 10:
                base_occupancy_pct += np.random.uniform(0.3, 0.5)
            # Lunch peak (12-2 PM)
            elif 12 <= hour < 14:
                base_occupancy_pct += np.random.uniform(0.15, 0.3)
            # Evening peak (4-6 PM)
            elif 16 <= hour < 18:
                base_occupancy_pct += np.random.uniform(0.25, 0.45)
            # Late night (10 PM - 6 AM)
            elif hour >= 22 or hour < 6:
                base_occupancy_pct -= np.random.uniform(0.15, 0.25)
        else:  # Weekend
            # Mid-day peak (11 AM - 3 PM)
            if 11 <= hour < 15:
                base_occupancy_pct += np.random.uniform(0.3, 0.5)
            # Evening (7-9 PM)
            elif 19 <= hour < 21:
                base_occupancy_pct += np.random.uniform(0.2, 0.4)
            # Late night (11 PM - 8 AM)
            elif hour >= 23 or hour < 8:
                base_occupancy_pct -= np.random.uniform(0.2, 0.3)
        
        # Ensure occupancy is within bounds (5-95%)
        occupancy_pct = max(0.05, min(0.95, base_occupancy_pct))
        
        # Calculate actual number of occupied spaces
        occupancy = int(occupancy_pct * total_spaces)
        
        data.append({
            'timestamp': t,
            'occupancy': occupancy,
            'total_spaces': total_spaces,
            'day_of_week': day_of_week,
            'hour': hour,
            'minute': minute
        })
    
    return pd.DataFrame(data)

def get_current_occupancy(total_spaces=200):
    """
    Generate current parking occupancy data for different areas.
    
    Parameters:
    - total_spaces: total number of parking spaces
    
    Returns:
    - Dictionary with current occupancy data
    """
    # Current time
    current_time = datetime.now()
    day_of_week = current_time.weekday()
    hour = current_time.hour
    
    # Define parking areas
    areas = {
        "Area A - Main": {"total": 80},
        "Area B - North": {"total": 60},
        "Area C - South": {"total": 40},
        "Area D - VIP": {"total": 20}
    }
    
    # Calculate base occupancy based on time patterns
    if day_of_week < 5:  # Weekday
        if 8 <= hour < 10:  # Morning peak
            base_occupancy_pct = np.random.uniform(0.7, 0.9)
        elif 12 <= hour < 14:  # Lunch peak
            base_occupancy_pct = np.random.uniform(0.5, 0.7)
        elif 16 <= hour < 18:  # Evening peak
            base_occupancy_pct = np.random.uniform(0.65, 0.85)
        elif hour >= 22 or hour < 6:  # Late night
            base_occupancy_pct = np.random.uniform(0.1, 0.25)
        else:  # Regular hours
            base_occupancy_pct = np.random.uniform(0.4, 0.6)
    else:  # Weekend
        if 11 <= hour < 15:  # Mid-day peak
            base_occupancy_pct = np.random.uniform(0.6, 0.8)
        elif 19 <= hour < 21:  # Evening
            base_occupancy_pct = np.random.uniform(0.5, 0.7)
        elif hour >= 23 or hour < 8:  # Late night
            base_occupancy_pct = np.random.uniform(0.1, 0.2)
        else:  # Regular hours
            base_occupancy_pct = np.random.uniform(0.3, 0.5)
    
    # Calculate occupancy for each area with some variability
    total_occupied = 0
    for area_name, area_data in areas.items():
        # Add variability to each area
        area_modifier = np.random.uniform(0.8, 1.2)
        area_occupancy_pct = base_occupancy_pct * area_modifier
        
        # Ensure within bounds
        area_occupancy_pct = max(0.05, min(0.95, area_occupancy_pct))
        
        # Special case for VIP area (typically less occupied)
        if area_name == "Area D - VIP":
            area_occupancy_pct *= 0.7
        
        occupied = int(area_occupancy_pct * area_data["total"])
        areas[area_name]["occupied"] = occupied
        areas[area_name]["available"] = area_data["total"] - occupied
        areas[area_name]["occupancy_pct"] = area_occupancy_pct * 100
        
        total_occupied += occupied
    
    # Create return object
    return {
        "timestamp": current_time,
        "total_spaces": total_spaces,
        "total_occupied": total_occupied,
        "total_available": total_spaces - total_occupied,
        "occupancy_pct": (total_occupied / total_spaces) * 100,
        "areas": areas
    }

if __name__ == "__main__":
    # Test data generation
    start = datetime.now() - timedelta(days=7)
    end = datetime.now()
    data = generate_parking_data(start, end)
    print(f"Generated {len(data)} data points")
    print(data.head())
    
    # Test current occupancy
    current = get_current_occupancy()
    print("\nCurrent occupancy:")
    print(f"Total: {current['total_occupied']}/{current['total_spaces']} ({current['occupancy_pct']:.1f}%)")
    for area, info in current['areas'].items():
        print(f"{area}: {info['occupied']}/{info['total']} ({info['occupancy_pct']:.1f}%)")
