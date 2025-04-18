import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import base64

# Import custom modules
from data_generator import generate_parking_data
from prediction_model import train_prediction_model, predict_parking_availability
from visualization import plot_occupancy_trend, plot_hourly_average, create_parking_map
from utils import load_svg, calculate_statistics
import database as db

# Initialize database
db.init_db()
db.seed_database()

# Set page configuration
st.set_page_config(
    page_title="USF Parking System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create a sidebar
st.sidebar.title("USF Parking System")
# Display USF colors
st.sidebar.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #006747; /* USF Green */
        color: #CFC493; /* USF Gold */
    }
    .stButton button {
        background-color: #006747;
        color: white;
    }
    h1, h2, h3 {
        color: #006747;
    }
    .stProgress > div > div {
        background-color: #006747;
    }
</style>
""", unsafe_allow_html=True)
parking_icon = load_svg("assets/parking_icon.svg")
st.sidebar.markdown(parking_icon, unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**University of South Florida**")
st.sidebar.markdown("Tampa Campus Parking")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Historical Data", "Management", "Student Resources", "About"],
)

# Initialize session state for data persistence
if 'historical_data' not in st.session_state:
    # Get historical data from database
    st.session_state.historical_data = db.get_historical_data(days=7)
    
    # Train the prediction model with historical data
    st.session_state.model = train_prediction_model(st.session_state.historical_data)

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
    
if 'real_time_data' not in st.session_state:
    # Initialize real-time data from database
    st.session_state.real_time_data = db.get_current_occupancy()

# Function to update real-time data
def update_data():
    current_time = datetime.now()
    
    # Update real-time data from database
    st.session_state.real_time_data = db.get_current_occupancy()
    
    # Add new data to historical dataset every 15 minutes
    time_diff = current_time - st.session_state.last_update
    if time_diff.total_seconds() >= 900:  # 15 minutes in seconds
        # Get the most recent data from the database
        st.session_state.historical_data = db.get_historical_data(days=7)
        st.session_state.last_update = current_time
        
        # Get main parking lot
        lots = db.get_parking_lots()
        if lots:
            main_lot = lots[0]
            
            # Add a new occupancy record to the database
            db.add_occupancy_record(
                lot_id=main_lot.id,
                occupied_spaces=st.session_state.real_time_data['total_occupied'],
                timestamp=current_time
            )
            
            # Add records for each area
            areas = db.get_parking_areas(main_lot.id)
            for area in areas:
                area_data = st.session_state.real_time_data['areas'].get(area.name)
                if area_data:
                    db.add_occupancy_record(
                        lot_id=main_lot.id,
                        area_id=area.id,
                        occupied_spaces=area_data['occupied'],
                        timestamp=current_time
                    )
        
        # Retrain model with updated data
        st.session_state.model = train_prediction_model(st.session_state.historical_data)

# Dashboard Page
if page == "Dashboard":
    st.title("USF Parking Availability Dashboard")
    st.markdown("Real-time monitoring and visualization of USF Tampa campus parking garages")
    
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
        st_folium(map_data, width=700, returned_objects=[])
    
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

# Management Page
elif page == "Management":
    st.title("Parking System Management")
    st.markdown("Add and manage parking lots, areas, and view database statistics")
    
    # Create tabs for different management functions
    tabs = st.tabs(["Add Parking Lot", "Add Parking Area", "Database Statistics"])
    
    # Add Parking Lot tab
    with tabs[0]:
        st.subheader("Add New Parking Lot")
        st.markdown("Create a new parking facility in the system")
        
        # Form for adding a new parking lot
        with st.form("add_lot_form"):
            lot_name = st.text_input("Parking Lot Name", placeholder="e.g., Downtown Parking Complex")
            total_spaces = st.number_input("Total Parking Spaces", min_value=1, value=100)
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude (optional)", value=37.7749)
            with col2:
                longitude = st.number_input("Longitude (optional)", value=-122.4194)
            
            submit_button = st.form_submit_button("Add Parking Lot")
            
            if submit_button:
                if lot_name.strip() == "":
                    st.error("Please enter a name for the parking lot")
                else:
                    # Add the new parking lot to the database
                    new_lot = db.add_parking_lot(lot_name, total_spaces, latitude, longitude)
                    st.success(f"Parking lot '{lot_name}' added successfully with ID: {new_lot.id}")
        
        # Display existing parking lots
        st.subheader("Existing Parking Lots")
        lots = db.get_parking_lots()
        if lots:
            lot_data = []
            for lot in lots:
                lot_data.append({
                    "ID": lot.id,
                    "Name": lot.name,
                    "Total Spaces": lot.total_spaces,
                    "Location": f"({lot.latitude}, {lot.longitude})"
                })
            st.dataframe(lot_data)
        else:
            st.info("No parking lots found in the database")
    
    # Add Parking Area tab
    with tabs[1]:
        st.subheader("Add New Parking Area")
        st.markdown("Add a specific area/section to an existing parking lot")
        
        # Get parking lots for selection
        lots = db.get_parking_lots()
        if not lots:
            st.warning("No parking lots available. Please add a parking lot first.")
        else:
            # Form for adding a new parking area
            with st.form("add_area_form"):
                lot_options = {lot.name: lot.id for lot in lots}
                selected_lot = st.selectbox("Select Parking Lot", options=list(lot_options.keys()))
                area_name = st.text_input("Area Name", placeholder="e.g., Level 1 - North")
                area_spaces = st.number_input("Number of Spaces in Area", min_value=1, value=50)
                
                submit_button = st.form_submit_button("Add Area")
                
                if submit_button:
                    if area_name.strip() == "":
                        st.error("Please enter a name for the parking area")
                    else:
                        # Add the new parking area to the database
                        lot_id = lot_options[selected_lot]
                        new_area = db.add_parking_area(area_name, area_spaces, lot_id)
                        st.success(f"Parking area '{area_name}' added successfully to '{selected_lot}'")
            
            # Display existing areas for each lot
            for lot in lots:
                st.subheader(f"Areas in {lot.name}")
                areas = db.get_parking_areas(lot.id)
                if areas:
                    area_data = []
                    for area in areas:
                        area_data.append({
                            "ID": area.id,
                            "Name": area.name,
                            "Spaces": area.total_spaces
                        })
                    st.dataframe(area_data)
                else:
                    st.info(f"No areas defined for {lot.name}")
    
    # Database Statistics tab
    with tabs[2]:
        st.subheader("Database Statistics")
        st.markdown("Overview of the parking system database")
        
        # Get database statistics
        stats = db.get_database_stats()
        
        # Display statistics in metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parking Lots", stats.get('total_lots', 0))
        with col2:
            st.metric("Total Parking Areas", stats.get('total_areas', 0))
        with col3:
            st.metric("Total Parking Spaces", stats.get('total_spaces', 0))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Occupancy Records", f"{stats.get('total_records', 0):,}")
        with col2:
            if 'days_of_data' in stats:
                st.metric("Days of Historical Data", stats['days_of_data'])
        
        # Display time range of data
        if 'earliest_timestamp' in stats and 'latest_timestamp' in stats:
            st.subheader("Data Time Range")
            st.markdown(f"**Earliest record:** {stats['earliest_timestamp']}")
            st.markdown(f"**Latest record:** {stats['latest_timestamp']}")
        
        # Option to reset the database (with confirmation)
        st.subheader("Database Management")
        if st.button("Reinitialize Database"):
            confirm = st.checkbox("I understand this will reset all data")
            if confirm:
                # Re-initialize the database
                db.init_db()
                db.seed_database()
                st.success("Database reinitialized successfully")
                st.rerun()

# Student Resources Page
elif page == "Student Resources":
    st.title("USF Student Parking Resources")
    
    # USF Branding
    st.markdown("""
    <div style="background-color: #006747; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #CFC493; text-align: center;">USF Bull Parking Resources</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different resources
    tabs = st.tabs(["Permit Information", "Campus Map", "Bull Runner", "FAQs"])
    
    # Permit Information tab
    with tabs[0]:
        st.header("Parking Permit Information")
        
        st.subheader("Student Permit Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Resident Student Permits
            
            **S** - Resident Student Permit
            - For students living in campus residence halls
            - Valid in all S and D designated lots/garages
            - Cost: $226.00 per year or $113.00 per semester
            
            **Y** - Resident Park-n-Ride Permit
            - For resident students in specific halls
            - Valid in Lot 43 and Park-n-Ride lots
            - Free Bull Runner shuttle service
            - Cost: $65.00 per year
            """)
        
        with col2:
            st.markdown("""
            ### Non-Resident Student Permits
            
            **D** - Non-Resident Student Permit
            - For commuter students
            - Valid in D designated lots/garages
            - Cost: $226.00 per year or $113.00 per semester
            
            **W** - Park-n-Ride Permit
            - Economy option for commuter students
            - Valid in Park-n-Ride lots only
            - Free Bull Runner shuttle service
            - Cost: $65.00 per year
            """)
        
        st.markdown("---")
        
        st.subheader("How to Purchase a Permit")
        st.markdown("""
        1. Visit [parking.usf.edu](https://parking.usf.edu)
        2. Log in with your USF NetID
        3. Select "Purchase Permit"
        4. Choose your permit type
        5. Provide vehicle information
        6. Complete payment information
        7. Receive confirmation and permit instructions
        """)
        
        # Add a fake permit purchase button for demonstration
        if st.button("Purchase Permit Online"):
            st.info("Redirecting to USF Parking Portal... (This is a demonstration)")
    
    # Campus Map tab
    with tabs[1]:
        st.header("USF Tampa Campus Parking Map")
        
        st.markdown("""
        ### Parking Garages
        
        | Garage | Location | Capacity | Permit Types |
        |--------|----------|----------|--------------|
        | Collins Garage | Near Collins Blvd | 1,800 | S, D, R, E |
        | Beard Garage | Near USF Library | 1,500 | S, D, GZ, E |
        | Laurel Garage | Near College of Medicine | 1,700 | Gold, S, D |
        | Crescent Hill Garage | Near The Village | 1,600 | S, D, R |
        """)
        
        st.markdown("### Legend")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **S** - Student")
        with col2:
            st.markdown("🔵 **D** - Daily")
        with col3:
            st.markdown("🟠 **R** - Reserved")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟣 **E** - Evening")
        with col2:
            st.markdown("🟡 **Gold** - Gold")
        with col3:
            st.markdown("⚪ **GZ** - Green")
        
        # Display campus map with interactive elements
        st.markdown("### Interactive Campus Map")
        map_data = create_parking_map(st.session_state.real_time_data)
        st_folium(map_data, width=700, height=500, returned_objects=[])
    
    # Bull Runner tab
    with tabs[2]:
        st.header("Bull Runner Shuttle Service")
        
        st.image("https://www.usf.edu/administrative-services/parking/documents/bullrunnerroutes.png", 
                caption="Bull Runner Routes (Example image - not real-time)")
        
        st.markdown("""
        ### Bull Runner Schedule
        
        The Bull Runner provides free transportation across campus for USF students, faculty, staff, and visitors.
        
        **Operating Hours:**
        - Monday - Thursday: 7:00 AM - 10:00 PM
        - Friday: 7:00 AM - 6:00 PM
        - Saturday - Sunday: Limited service
        
        **Routes:**
        - **A Route**: Collins Garage to Marshall Center
        - **B Route**: Library to The Village
        - **C Route**: Engineering to College of Medicine
        - **D Route**: Park-n-Ride to Campus Core
        - **E Route**: Off-campus apartments
        - **F Route**: Shopping and entertainment
        """)
        
        st.markdown("### Track the Bull Runner")
        st.markdown("Download the USF Mobile app to track Bull Runner shuttles in real-time.")
        
        # Add fake tracker button for demonstration
        if st.button("Launch Bull Runner Tracker"):
            st.info("Launching Bull Runner Tracker... (This is a demonstration)")
    
    # FAQs tab
    with tabs[3]:
        st.header("Frequently Asked Questions")
        
        # Using expanders for FAQ items
        with st.expander("What times are parking rules enforced?"):
            st.markdown("""
            Parking rules are enforced 24 hours a day, 7 days a week, including weekends and holidays.
            
            Some lots have specific enforcement hours:
            - Reserved spaces: 24/7
            - Resident spaces: 24/7
            - Non-resident spaces: 7:00 AM - 5:30 PM, Monday - Friday
            - Evening permits: Valid after 5:30 PM
            """)
        
        with st.expander("What do I do if I receive a citation?"):
            st.markdown("""
            If you receive a parking citation, you have the following options:
            
            1. Pay the citation online at [parking.usf.edu](https://parking.usf.edu)
            2. Appeal the citation within 14 calendar days
            3. Submit an appeal online with supporting documentation
            
            Unpaid citations may result in registration holds or vehicle immobilization.
            """)
        
        with st.expander("Can I get a temporary permit?"):
            st.markdown("""
            Yes, temporary permits are available for:
            
            - Visitors: Daily permits available for $5.00
            - Students with temporary needs: Weekly permits available
            - Medical conditions: Special accommodation permits
            
            Visit the Parking & Transportation Services office or call (813) 974-3990 for assistance.
            """)
        
        with st.expander("How does the parking waitlist work?"):
            st.markdown("""
            When demand exceeds availability for certain permit types (like Gold or Reserved):
            
            1. Join the waitlist through your parking account
            2. Waitlist position is based on first-come, first-served
            3. Receive notification when a permit becomes available
            4. Purchase within 72 hours or forfeit your position
            
            You must have a valid permit while on the waitlist.
            """)
        
        with st.expander("What are the most common parking violations?"):
            st.markdown("""
            The most common parking violations are:
            
            1. Parking without a valid permit ($30)
            2. Parking in the wrong lot for your permit type ($30)
            3. Parking in a reserved space ($75)
            4. Blocking traffic or parking in a fire lane ($75)
            5. Parking in a handicap space without proper credentials ($275)
            
            Repeat violations may result in increased fines.
            """)
        
        # Contact information
        st.subheader("Contact Parking & Transportation Services")
        st.markdown("""
        **Location:** Parking & Transportation Services Building (PSB)
        **Phone:** (813) 974-3990
        **Email:** psweb@usf.edu
        **Hours:** Monday - Friday, 8:00 AM - 5:00 PM
        """)

# About Page
elif page == "About":
    st.title("About USF Parking System")
    
    # Display USF logo
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://www.usf.edu/identity/images/logos-and-brand-assets/usf-logo-color-rgb.png" width="400">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## Overview
    
    The USF Parking System is an AI-powered application that provides real-time monitoring and prediction of parking availability across the University of South Florida's Tampa campus. The system aims to help students, faculty, staff, and visitors find parking more efficiently, reduce congestion, and improve the overall parking experience at USF.
    
    ### Key Features
    
    - **Real-time Monitoring**: Track current parking occupancy across different areas
    - **Interactive Visualization**: Visual representation of parking spaces with status indicators
    - **Predictive Analytics**: Machine learning algorithms to forecast future parking availability
    - **Historical Analysis**: Explore past usage patterns to identify trends and peak periods
    
    ### Technology Stack
    
    - **Frontend**: Streamlit for interactive web interface
    - **Database**: PostgreSQL for data storage and retrieval
    - **Data Processing**: Pandas and NumPy for data manipulation
    - **ORM**: SQLAlchemy for database interactions
    - **Machine Learning**: Scikit-learn for predictive models
    - **Visualization**: Plotly and Folium for interactive charts and maps
    
    ### Data Sources
    
    In a production environment, this system would integrate with:
    
    - Smart parking sensors
    - Camera-based occupancy detection
    - Ticket systems and parking gates
    - Weather APIs and event calendars for improved predictions
    
    For this demonstration, we use a PostgreSQL database with simulated data that mimics real-world parking patterns.
    
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
