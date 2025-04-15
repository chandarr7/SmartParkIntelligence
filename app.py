import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import base64

# Import custom modules
from data_generator import generate_parking_data, get_current_occupancy
from prediction_model import train_prediction_model, predict_parking_availability
from visualization import plot_occupancy_trend, plot_hourly_average, create_parking_map
from utils import load_svg, calculate_statistics

# Set page configuration
st.set_page_config(
    page_title="Smart Parking System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create a sidebar
st.sidebar.title("Smart Parking System")
parking_icon = load_svg("assets/parking_icon.svg")
st.sidebar.markdown(parking_icon, unsafe_allow_html=True)
st.sidebar.markdown("---")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Historical Data", "About"],
)

# Initialize session state for data persistence
if 'historical_data' not in st.session_state:
    # Generate initial historical data for the past 7 days
    current_time = datetime.now()
    days_back = 7
    start_time = current_time - timedelta(days=days_back)
    
    # Generate data for past 7 days with 15-minute intervals
    st.session_state.historical_data = generate_parking_data(
        start_time, 
        current_time,
        interval_minutes=15
    )
    
    # Train the prediction model with historical data
    st.session_state.model = train_prediction_model(st.session_state.historical_data)

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
    
if 'real_time_data' not in st.session_state:
    # Initialize real-time data
    st.session_state.real_time_data = get_current_occupancy()

# Function to update real-time data
def update_data():
    current_time = datetime.now()
    
    # Update real-time data
    st.session_state.real_time_data = get_current_occupancy()
    
    # Add new data to historical dataset every 15 minutes
    time_diff = current_time - st.session_state.last_update
    if time_diff.total_seconds() >= 900:  # 15 minutes in seconds
        new_data = pd.DataFrame({
            'timestamp': [current_time],
            'occupancy': [st.session_state.real_time_data['total_occupied']],
            'total_spaces': [st.session_state.real_time_data['total_spaces']],
            'day_of_week': [current_time.weekday()],
            'hour': [current_time.hour],
            'minute': [current_time.minute]
        })
        
        st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_data], ignore_index=True)
        st.session_state.last_update = current_time
        
        # Retrain model with updated data
        st.session_state.model = train_prediction_model(st.session_state.historical_data)

# Dashboard Page
if page == "Dashboard":
    st.title("Parking Availability Dashboard")
    st.markdown("Real-time monitoring and visualization of parking spaces")
    
    # Update data
    update_data()
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        available = st.session_state.real_time_data['total_spaces'] - st.session_state.real_time_data['total_occupied']
        occupancy_pct = (st.session_state.real_time_data['total_occupied'] / st.session_state.real_time_data['total_spaces']) * 100
        st.metric(
            label="Available Spaces", 
            value=f"{available}/{st.session_state.real_time_data['total_spaces']}",
            delta="-" if occupancy_pct > 75 else "+"
        )
    
    with col2:
        st.metric(
            label="Occupancy Rate", 
            value=f"{occupancy_pct:.1f}%",
            delta=f"{occupancy_pct - 70:.1f}%" if occupancy_pct > 70 else f"{70 - occupancy_pct:.1f}%"
        )
    
    with col3:
        # Get prediction for next hour
        next_hour = datetime.now() + timedelta(hours=1)
        predicted_occupancy = predict_parking_availability(
            st.session_state.model, 
            next_hour.weekday(),
            next_hour.hour,
            0  # minute = 0 for the start of the hour
        )
        predicted_pct = (predicted_occupancy / st.session_state.real_time_data['total_spaces']) * 100
        st.metric(
            label="Predicted (Next Hour)", 
            value=f"{predicted_pct:.1f}%",
            delta=f"{predicted_pct - occupancy_pct:.1f}%"
        )
    
    with col4:
        stats = calculate_statistics(st.session_state.historical_data)
        st.metric(
            label="Peak Occupancy Today", 
            value=f"{stats['peak_today_pct']:.1f}%",
            delta=f"{stats['peak_today_pct'] - stats['avg_pct']:.1f}%"
        )
    
    st.markdown("---")
    
    # Interactive map and current status
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Parking Map")
        map_data = create_parking_map(st.session_state.real_time_data)
        folium_static(map_data, width=700)
    
    with col2:
        st.subheader("Current Status")
        
        # Create a visual representation of parking areas
        areas = st.session_state.real_time_data['areas']
        for area_name, area_data in areas.items():
            occupied = area_data['occupied']
            total = area_data['total']
            occupied_pct = (occupied / total) * 100
            
            st.markdown(f"**{area_name}** - {occupied}/{total} spaces occupied")
            
            # Create a progress bar for occupancy
            st.progress(occupied_pct / 100)
            
            # Determine status text and color
            if occupied_pct < 50:
                st.markdown("🟢 **Low occupancy**")
            elif occupied_pct < 80:
                st.markdown("🟠 **Moderate occupancy**")
            else:
                st.markdown("🔴 **High occupancy**")
    
    st.markdown("---")
    
    # Occupancy trends
    st.subheader("Today's Occupancy Trend")
    fig = plot_occupancy_trend(st.session_state.historical_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh the data every 60 seconds
    time.sleep(60)
    st.rerun()

# Prediction Page
elif page == "Prediction":
    st.title("Parking Availability Prediction")
    st.markdown("ML-powered predictions for future parking availability")
    
    # Update data
    update_data()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Time selector
        prediction_date = st.date_input(
            "Select date for prediction",
            datetime.now().date()
        )
        
        prediction_hour = st.slider(
            "Select hour (24-hour format)",
            0, 23, datetime.now().hour
        )
        
        # Generate prediction
        if st.button("Generate Prediction"):
            predicted_date = datetime.combine(prediction_date, datetime.min.time()) + timedelta(hours=prediction_hour)
            
            # Only predict if date is within next 7 days
            max_date = datetime.now().date() + timedelta(days=7)
            
            if prediction_date > max_date:
                st.error("Predictions are only available for the next 7 days")
            else:
                # Get prediction
                predicted_occupancy = predict_parking_availability(
                    st.session_state.model, 
                    predicted_date.weekday(),
                    predicted_date.hour,
                    0  # minute = 0 for the start of the hour
                )
                
                predicted_pct = (predicted_occupancy / st.session_state.real_time_data['total_spaces']) * 100
                
                st.markdown(f"### Predicted Occupancy: {predicted_pct:.1f}%")
                st.markdown(f"Estimated available spaces: {st.session_state.real_time_data['total_spaces'] - predicted_occupancy} out of {st.session_state.real_time_data['total_spaces']}")
                
                # Display prediction confidence and recommendation
                if predicted_pct < 50:
                    st.markdown("🟢 **Low occupancy expected** - Parking should be readily available.")
                elif predicted_pct < 80:
                    st.markdown("🟠 **Moderate occupancy expected** - You may need to search briefly for parking.")
                else:
                    st.markdown("🔴 **High occupancy expected** - Parking will be limited. Consider arriving early or using alternative transportation.")
    
    with col2:
        st.subheader("Weekly Prediction Pattern")
        # Create hourly average occupancy plot
        fig = plot_hourly_average(st.session_state.historical_data)
        st.plotly_chart(fig)
        
        st.markdown("""
        **How to use this prediction feature:**
        
        1. Select a date and time to predict parking availability
        2. The model uses historical patterns and current trends to estimate occupancy
        3. The confidence level indicates reliability of the prediction
        4. Recommendations are provided based on predicted occupancy levels
        """)
    
    st.markdown("---")
    
    st.subheader("Next 24 Hours Forecast")
    
    # Generate 24-hour forecast
    current_time = datetime.now()
    forecast_hours = 24
    forecast_times = [current_time + timedelta(hours=i) for i in range(1, forecast_hours + 1)]
    
    forecast_data = []
    for t in forecast_times:
        pred = predict_parking_availability(
            st.session_state.model, 
            t.weekday(),
            t.hour,
            0  # minute = 0 for the start of the hour
        )
        forecast_data.append({
            'time': t,
            'predicted_occupancy': pred,
            'occupancy_pct': (pred / st.session_state.real_time_data['total_spaces']) * 100
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Create a line chart for the forecast
    fig = px.line(
        forecast_df, 
        x='time', 
        y='occupancy_pct',
        labels={'time': 'Time', 'occupancy_pct': 'Occupancy %'},
        title='24-Hour Occupancy Forecast'
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Predicted Occupancy (%)",
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Historical Data Page
elif page == "Historical Data":
    st.title("Historical Parking Data")
    st.markdown("Analysis of past parking patterns and usage")
    
    # Update data
    update_data()
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            datetime.now().date() - timedelta(days=7)
        )
    
    with col2:
        end_date = st.date_input(
            "End date",
            datetime.now().date()
        )
    
    # Filter data based on date range
    mask = (st.session_state.historical_data['timestamp'].dt.date >= start_date) & \
           (st.session_state.historical_data['timestamp'].dt.date <= end_date)
    filtered_data = st.session_state.historical_data.loc[mask]
    
    if len(filtered_data) == 0:
        st.warning("No data available for the selected date range.")
    else:
        # Calculate daily statistics
        daily_stats = filtered_data.groupby(filtered_data['timestamp'].dt.date).agg({
            'occupancy': ['mean', 'max', 'min'],
            'total_spaces': 'first'
        }).reset_index()
        
        daily_stats.columns = ['date', 'avg_occupancy', 'max_occupancy', 'min_occupancy', 'total_spaces']
        daily_stats['avg_pct'] = (daily_stats['avg_occupancy'] / daily_stats['total_spaces']) * 100
        daily_stats['max_pct'] = (daily_stats['max_occupancy'] / daily_stats['total_spaces']) * 100
        daily_stats['min_pct'] = (daily_stats['min_occupancy'] / daily_stats['total_spaces']) * 100
        
        # Plot daily statistics
        st.subheader("Daily Occupancy Statistics")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_stats['date'], 
            y=daily_stats['max_pct'],
            mode='lines',
            name='Max Occupancy',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_stats['date'], 
            y=daily_stats['avg_pct'],
            mode='lines',
            name='Average Occupancy',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_stats['date'], 
            y=daily_stats['min_pct'],
            mode='lines',
            name='Min Occupancy',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='Daily Occupancy Trends',
            xaxis_title='Date',
            yaxis_title='Occupancy (%)',
            yaxis_range=[0, 100],
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of week patterns
        st.subheader("Day of Week Patterns")
        
        # Calculate average occupancy by day of week and hour
        dow_hourly = filtered_data.groupby(['day_of_week', 'hour']).agg({
            'occupancy': 'mean',
            'total_spaces': 'first'
        }).reset_index()
        
        dow_hourly['occupancy_pct'] = (dow_hourly['occupancy'] / dow_hourly['total_spaces']) * 100
        
        # Create a pivot table for heatmap
        pivot_data = dow_hourly.pivot(index='hour', columns='day_of_week', values='occupancy_pct')
        
        # Create a heatmap using Plotly
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Day of Week", y="Hour of Day", color="Occupancy %"),
            x=[days[i] for i in pivot_data.columns],
            y=pivot_data.index,
            aspect="auto",
            color_continuous_scale="RdYlGn_r"
        )
        
        fig.update_layout(
            title='Occupancy Heatmap by Day and Hour',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            coloraxis_colorbar=dict(title='Occupancy %')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trends if data spans multiple months
        if (filtered_data['timestamp'].max() - filtered_data['timestamp'].min()).days > 30:
            st.subheader("Monthly Trends")
            
            monthly_data = filtered_data.groupby(pd.Grouper(key='timestamp', freq='M')).agg({
                'occupancy': 'mean',
                'total_spaces': 'first'
            }).reset_index()
            
            monthly_data['month'] = monthly_data['timestamp'].dt.strftime('%B %Y')
            monthly_data['occupancy_pct'] = (monthly_data['occupancy'] / monthly_data['total_spaces']) * 100
            
            fig = px.bar(
                monthly_data,
                x='month',
                y='occupancy_pct',
                labels={'month': 'Month', 'occupancy_pct': 'Average Occupancy (%)'},
                title='Monthly Average Occupancy'
            )
            
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title='Average Occupancy (%)',
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig, use_container_width=True)

# About Page
elif page == "About":
    st.title("About Smart Parking System")
    
    st.markdown("""
    ## Overview
    
    The Smart Parking System is an AI-powered application that provides real-time monitoring and prediction of parking availability. The system aims to reduce congestion, improve user experience, and provide valuable insights for parking management.
    
    ### Key Features
    
    - **Real-time Monitoring**: Track current parking occupancy across different areas
    - **Interactive Visualization**: Visual representation of parking spaces with status indicators
    - **Predictive Analytics**: Machine learning algorithms to forecast future parking availability
    - **Historical Analysis**: Explore past usage patterns to identify trends and peak periods
    
    ### Technology Stack
    
    - **Frontend**: Streamlit for interactive web interface
    - **Data Processing**: Pandas and NumPy for data manipulation
    - **Machine Learning**: Scikit-learn for predictive models
    - **Visualization**: Plotly and Folium for interactive charts and maps
    
    ### Data Sources
    
    In a production environment, this system would integrate with:
    
    - Smart parking sensors
    - Camera-based occupancy detection
    - Ticket systems and parking gates
    - Weather APIs and event calendars for improved predictions
    
    For this demonstration, we use simulated data that mimics real-world parking patterns.
    
    ### Future Enhancements
    
    - Integration with mobile apps for navigation
    - Advanced prediction models incorporating weather and events
    - Reservation systems and payment integration
    - Multi-site parking management
    """)

    st.markdown("---")
    
    st.markdown("""
    ## How It Works
    
    ### Data Collection
    
    The system continuously collects occupancy data from parking sensors or cameras. This data is processed and stored to provide real-time updates and historical analysis.
    
    ### Prediction Algorithm
    
    Our machine learning model analyzes historical patterns to predict future parking availability. The model considers:
    
    - Day of the week
    - Time of day
    - Historical occupancy patterns
    - Recent trends
    
    ### Visualization Engine
    
    The dashboard displays real-time and historical data through interactive charts and maps. Users can:
    
    - View current availability by parking area
    - Analyze historical trends and patterns
    - Predict future availability for planning
    """)
