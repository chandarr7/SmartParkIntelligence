import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define database models
class ParkingLot(Base):
    __tablename__ = 'parking_lots'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    total_spaces = Column(Integer, nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Relationship
    areas = relationship("ParkingArea", back_populates="lot", cascade="all, delete-orphan")
    occupancy_records = relationship("OccupancyRecord", back_populates="lot", cascade="all, delete-orphan")

class ParkingArea(Base):
    __tablename__ = 'parking_areas'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    total_spaces = Column(Integer, nullable=False)
    lot_id = Column(Integer, ForeignKey('parking_lots.id'), nullable=False)
    permit_type = Column(String, default="All")  # USF-specific permit type (S, D, R, Gold, etc.)
    
    # Relationship
    lot = relationship("ParkingLot", back_populates="areas")
    occupancy_records = relationship("OccupancyRecord", back_populates="area", cascade="all, delete-orphan")

class OccupancyRecord(Base):
    __tablename__ = 'occupancy_records'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    occupied_spaces = Column(Integer, nullable=False)
    lot_id = Column(Integer, ForeignKey('parking_lots.id'), nullable=False)
    area_id = Column(Integer, ForeignKey('parking_areas.id'))
    day_of_week = Column(Integer)  # 0-6, where 0 is Monday
    hour = Column(Integer)
    minute = Column(Integer)
    
    # Relationships
    lot = relationship("ParkingLot", back_populates="occupancy_records")
    area = relationship("ParkingArea", back_populates="occupancy_records")

class USFPermit(Base):
    __tablename__ = 'usf_permits'
    
    id = Column(Integer, primary_key=True)
    permit_type = Column(String, nullable=False)  # S, D, R, Gold, GZ, E, Y, W
    description = Column(String, nullable=False)  # Description of the permit
    annual_price = Column(Float, nullable=False)  # Annual price
    semester_price = Column(Float)  # Semester price (if available)
    valid_areas = Column(String, nullable=False)  # Areas where permit is valid
    user_type = Column(String, nullable=False)  # Student, Faculty, Staff, etc.
    
    def __repr__(self):
        return f"<USFPermit(type='{self.permit_type}', for='{self.user_type}')>"

def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(engine)

def get_parking_lots():
    """Get all parking lots from the database."""
    session = Session()
    try:
        return session.query(ParkingLot).all()
    finally:
        session.close()

def get_parking_areas(lot_id=None):
    """Get all parking areas, optionally filtered by lot_id."""
    session = Session()
    try:
        query = session.query(ParkingArea)
        if lot_id:
            query = query.filter(ParkingArea.lot_id == lot_id)
        return query.all()
    finally:
        session.close()

def get_current_occupancy():
    """
    Get current occupancy data from the database.
    If no recent data (within last 15 minutes) exists, returns a simulated occupancy.
    """
    session = Session()
    try:
        # Get all parking lots
        lots = session.query(ParkingLot).all()
        
        if not lots:
            # If no lots in database, return empty data
            return {
                "timestamp": datetime.now(),
                "total_spaces": 0,
                "total_occupied": 0,
                "total_available": 0,
                "occupancy_pct": 0,
                "areas": {}
            }
        
        # Get the most recent occupancy record for each area
        recent_time = datetime.now() - timedelta(minutes=15)
        recent_records = {}
        
        total_spaces = 0
        total_occupied = 0
        areas_data = {}
        
        for lot in lots:
            total_spaces += lot.total_spaces
            
            # Get areas for this lot
            lot_areas = session.query(ParkingArea).filter(ParkingArea.lot_id == lot.id).all()
            
            for area in lot_areas:
                # Get most recent record for this area
                recent_record = session.query(OccupancyRecord).filter(
                    OccupancyRecord.area_id == area.id,
                    OccupancyRecord.timestamp > recent_time
                ).order_by(OccupancyRecord.timestamp.desc()).first()
                
                if recent_record:
                    occupied = recent_record.occupied_spaces
                else:
                    # If no recent record, simulate one (50% occupancy with some randomness)
                    occupied = int(area.total_spaces * np.random.uniform(0.3, 0.7))
                
                total_occupied += occupied
                
                # Create area data
                areas_data[area.name] = {
                    "total": area.total_spaces,
                    "occupied": occupied,
                    "available": area.total_spaces - occupied,
                    "occupancy_pct": (occupied / area.total_spaces) * 100 if area.total_spaces > 0 else 0
                }
        
        return {
            "timestamp": datetime.now(),
            "total_spaces": total_spaces,
            "total_occupied": total_occupied,
            "total_available": total_spaces - total_occupied,
            "occupancy_pct": (total_occupied / total_spaces) * 100 if total_spaces > 0 else 0,
            "areas": areas_data
        }
    finally:
        session.close()

def get_historical_data(days=7):
    """
    Get historical occupancy data from the database.
    
    Parameters:
    - days: Number of days to look back
    
    Returns:
    - DataFrame with historical data
    """
    session = Session()
    try:
        # Calculate start date
        start_date = datetime.now() - timedelta(days=days)
        
        # Get records
        records = session.query(OccupancyRecord).filter(
            OccupancyRecord.timestamp >= start_date
        ).order_by(OccupancyRecord.timestamp).all()
        
        if not records:
            # If no records, return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'timestamp', 'occupancy', 'total_spaces', 
                'day_of_week', 'hour', 'minute'
            ])
        
        # Convert to DataFrame
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'occupancy': record.occupied_spaces,
                'total_spaces': record.lot.total_spaces,
                'day_of_week': record.day_of_week,
                'hour': record.hour,
                'minute': record.minute,
                'lot_id': record.lot_id,
                'area_id': record.area_id
            })
        
        return pd.DataFrame(data)
    finally:
        session.close()

def add_occupancy_record(lot_id, occupied_spaces, area_id=None, timestamp=None):
    """
    Add a new occupancy record to the database.
    
    Parameters:
    - lot_id: ID of the parking lot
    - occupied_spaces: Number of occupied spaces
    - area_id: ID of the parking area (optional)
    - timestamp: Record timestamp (defaults to current time)
    
    Returns:
    - The created record
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    session = Session()
    try:
        # Create new record
        record = OccupancyRecord(
            timestamp=timestamp,
            occupied_spaces=occupied_spaces,
            lot_id=lot_id,
            area_id=area_id,
            day_of_week=timestamp.weekday(),
            hour=timestamp.hour,
            minute=timestamp.minute
        )
        
        # Add to database
        session.add(record)
        session.commit()
        
        return record
    finally:
        session.close()

def add_parking_lot(name, total_spaces, latitude=None, longitude=None):
    """
    Add a new parking lot to the database.
    
    Parameters:
    - name: Name of the parking lot
    - total_spaces: Total number of parking spaces
    - latitude: Optional latitude coordinate
    - longitude: Optional longitude coordinate
    
    Returns:
    - The created parking lot
    """
    session = Session()
    try:
        # Create new parking lot
        lot = ParkingLot(
            name=name,
            total_spaces=total_spaces,
            latitude=latitude,
            longitude=longitude
        )
        
        # Add to database
        session.add(lot)
        session.commit()
        
        return lot
    finally:
        session.close()

def add_parking_area(name, total_spaces, lot_id, permit_type="All"):
    """
    Add a new parking area to the database.
    
    Parameters:
    - name: Name of the parking area
    - total_spaces: Total number of parking spaces
    - lot_id: ID of the parking lot this area belongs to
    - permit_type: Type of permit required for this area (default: "All")
    
    Returns:
    - The created parking area
    """
    session = Session()
    try:
        # Create new parking area
        area = ParkingArea(
            name=name,
            total_spaces=total_spaces,
            lot_id=lot_id,
            permit_type=permit_type
        )
        
        # Add to database
        session.add(area)
        session.commit()
        
        return area
    finally:
        session.close()

def get_database_stats():
    """
    Get statistics about the database.
    
    Returns:
    - Dictionary with database statistics
    """
    session = Session()
    try:
        stats = {}
        
        # Count parking lots
        stats['total_lots'] = session.query(ParkingLot).count()
        
        # Count parking areas
        stats['total_areas'] = session.query(ParkingArea).count()
        
        # Count occupancy records
        stats['total_records'] = session.query(OccupancyRecord).count()
        
        # Count USF permits
        stats['total_permits'] = session.query(USFPermit).count()
        
        # Get earliest and latest timestamps
        earliest_record = session.query(OccupancyRecord).order_by(OccupancyRecord.timestamp).first()
        latest_record = session.query(OccupancyRecord).order_by(OccupancyRecord.timestamp.desc()).first()
        
        if earliest_record and latest_record:
            stats['earliest_timestamp'] = earliest_record.timestamp
            stats['latest_timestamp'] = latest_record.timestamp
            stats['days_of_data'] = (latest_record.timestamp - earliest_record.timestamp).days
        
        # Count total parking spaces
        total_spaces = session.query(ParkingLot.total_spaces).all()
        stats['total_spaces'] = sum([spaces[0] for spaces in total_spaces])
        
        return stats
    finally:
        session.close()

def get_usf_permits():
    """
    Get all USF parking permits from the database.
    
    Returns:
    - List of USFPermit objects
    """
    session = Session()
    try:
        return session.query(USFPermit).all()
    finally:
        session.close()

def add_usf_permit(permit_type, description, annual_price, valid_areas, user_type, semester_price=None):
    """
    Add a new USF parking permit to the database.
    
    Parameters:
    - permit_type: Permit type code (S, D, R, Gold, etc.)
    - description: Description of the permit
    - annual_price: Annual price of the permit
    - valid_areas: Areas where the permit is valid
    - user_type: Type of user (Student, Faculty, Staff, etc.)
    - semester_price: Semester price (optional)
    
    Returns:
    - The created permit
    """
    session = Session()
    try:
        # Create new permit
        permit = USFPermit(
            permit_type=permit_type,
            description=description,
            annual_price=annual_price,
            semester_price=semester_price,
            valid_areas=valid_areas,
            user_type=user_type
        )
        
        # Add to database
        session.add(permit)
        session.commit()
        
        return permit
    finally:
        session.close()

def seed_usf_permits():
    """
    Seed the database with USF parking permit data.
    """
    session = Session()
    try:
        # Check if we already have permit data
        existing_permits = session.query(USFPermit).count()
        if existing_permits > 0:
            return
            
        # Create USF permits
        permits = [
            # Student permits
            USFPermit(
                permit_type="S",
                description="Resident Student Permit",
                annual_price=226.00,
                semester_price=113.00,
                valid_areas="S and D designated lots/garages",
                user_type="Resident Student"
            ),
            USFPermit(
                permit_type="D",
                description="Non-Resident Student Permit",
                annual_price=226.00,
                semester_price=113.00,
                valid_areas="D designated lots/garages",
                user_type="Non-Resident Student"
            ),
            USFPermit(
                permit_type="Y",
                description="Resident Park-n-Ride Permit",
                annual_price=65.00,
                semester_price=None,
                valid_areas="Lot 43 and Park-n-Ride lots",
                user_type="Resident Student"
            ),
            USFPermit(
                permit_type="W",
                description="Park-n-Ride Permit",
                annual_price=65.00,
                semester_price=None,
                valid_areas="Park-n-Ride lots only",
                user_type="Non-Resident Student"
            ),
            
            # Staff/Faculty permits
            USFPermit(
                permit_type="Gold",
                description="Gold Staff Permit",
                annual_price=1022.00,
                semester_price=511.00,
                valid_areas="Gold zones and all other non-reserved areas",
                user_type="Faculty/Staff"
            ),
            USFPermit(
                permit_type="GZ",
                description="Green Staff Permit",
                annual_price=428.00,
                semester_price=214.00,
                valid_areas="Green zones and student areas",
                user_type="Faculty/Staff"
            ),
            USFPermit(
                permit_type="E",
                description="Evening Staff Permit",
                annual_price=219.00,
                semester_price=109.50,
                valid_areas="Valid after 5:30 PM in any non-reserved space",
                user_type="Faculty/Staff"
            ),
            USFPermit(
                permit_type="R",
                description="Reserved Permit",
                annual_price=1603.00,
                semester_price=801.50,
                valid_areas="Reserved spaces and all non-reserved areas",
                user_type="Faculty/Staff"
            ),
            
            # Other permits
            USFPermit(
                permit_type="DV",
                description="Daily Visitor Permit",
                annual_price=5.00,
                semester_price=None,
                valid_areas="Visitor areas and student areas",
                user_type="Visitor"
            ),
            USFPermit(
                permit_type="MC",
                description="Motorcycle Permit",
                annual_price=219.00,
                semester_price=109.50,
                valid_areas="Motorcycle spaces only",
                user_type="Any"
            )
        ]
        
        session.add_all(permits)
        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"Error seeding USF permits: {e}")
        raise
    finally:
        session.close()

def seed_database():
    """
    Seed the database with initial data if it's empty.
    Creates a parking lot with areas and generates historical data.
    """
    session = Session()
    try:
        # Check if we already have data
        existing_lots = session.query(ParkingLot).count()
        if existing_lots > 0:
            return
        
        # Create USF parking lots
        collins_garage = ParkingLot(
            name="Collins Garage",
            total_spaces=1800,
            latitude=28.0587,
            longitude=-82.4139
        )
        session.add(collins_garage)
        
        beard_garage = ParkingLot(
            name="Beard Garage",
            total_spaces=1500,
            latitude=28.0650,
            longitude=-82.4144
        )
        session.add(beard_garage)
        
        laurel_garage = ParkingLot(
            name="Laurel Garage",
            total_spaces=1700,
            latitude=28.0622,
            longitude=-82.4099
        )
        session.add(laurel_garage)
        
        crescent_garage = ParkingLot(
            name="Crescent Hill Garage",
            total_spaces=1600,
            latitude=28.0643,
            longitude=-82.4119
        )
        session.add(crescent_garage)
        
        session.flush()  # To get the IDs
        
        # Create parking areas (by permit type)
        areas = [
            # Collins Garage areas
            ParkingArea(name="Gold Zone", total_spaces=200, lot_id=collins_garage.id, permit_type="Gold"),
            ParkingArea(name="Green Zone", total_spaces=900, lot_id=collins_garage.id, permit_type="GZ"),
            ParkingArea(name="Resident Zone", total_spaces=500, lot_id=collins_garage.id, permit_type="S"),
            ParkingArea(name="Non-Resident Zone", total_spaces=200, lot_id=collins_garage.id, permit_type="D"),
            
            # Beard Garage areas
            ParkingArea(name="Staff Zone", total_spaces=400, lot_id=beard_garage.id, permit_type="GZ"),
            ParkingArea(name="Student Zone", total_spaces=800, lot_id=beard_garage.id, permit_type="D"),
            ParkingArea(name="Visitor Zone", total_spaces=200, lot_id=beard_garage.id, permit_type="DV"),
            ParkingArea(name="Reserved Zone", total_spaces=100, lot_id=beard_garage.id, permit_type="R"),
            
            # Laurel Garage areas
            ParkingArea(name="Gold Zone", total_spaces=300, lot_id=laurel_garage.id, permit_type="Gold"),
            ParkingArea(name="Green Zone", total_spaces=1000, lot_id=laurel_garage.id, permit_type="GZ"),
            ParkingArea(name="Visitor Zone", total_spaces=400, lot_id=laurel_garage.id, permit_type="DV"),
            
            # Crescent Hill Garage areas
            ParkingArea(name="Staff Zone", total_spaces=500, lot_id=crescent_garage.id, permit_type="GZ"),
            ParkingArea(name="Student Zone", total_spaces=900, lot_id=crescent_garage.id, permit_type="S"),
            ParkingArea(name="Visitor Zone", total_spaces=200, lot_id=crescent_garage.id, permit_type="DV")
        ]
        session.add_all(areas)
        session.commit()
        
        # Generate historical data
        from data_generator import generate_parking_data
        
        # Generate data for past 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        historical_data = generate_parking_data(start_time, end_time)
        
        # Add historical data to database
        areas_dict = {area.name: area for area in areas}
        
        # Get all parking lots
        lots = [collins_garage, beard_garage, laurel_garage, crescent_garage]
        
        for _, row in historical_data.iterrows():
            # Process each lot with its areas
            for lot in lots:
                # Get areas for this lot
                lot_areas = [area for area in areas if area.lot_id == lot.id]
                
                # Calculate total capacity for this lot
                total_capacity = sum(area.total_spaces for area in lot_areas)
                
                # Calculate occupancy rate (adjust by time of day, day of week)
                # Simulate busier hours during weekdays, especially morning and afternoon
                base_rate = 0.4  # Base occupancy rate
                
                # Higher traffic during day hours on weekdays
                if 0 <= row['day_of_week'] <= 4:  # Weekday
                    if 8 <= row['hour'] <= 15:  # Morning to afternoon classes
                        base_rate = 0.75
                    elif 16 <= row['hour'] <= 18:  # Evening classes
                        base_rate = 0.60
                else:  # Weekend
                    base_rate = 0.25
                    
                # Random variation
                import random
                variation = random.uniform(-0.1, 0.1)
                occupancy_rate = min(0.95, max(0.1, base_rate + variation))
                
                # Calculate occupancy for this lot
                lot_occupancy = int(lot.total_spaces * occupancy_rate)
                
                # Create record for the lot
                lot_record = OccupancyRecord(
                    timestamp=row['timestamp'],
                    occupied_spaces=lot_occupancy,
                    lot_id=lot.id,
                    day_of_week=row['day_of_week'],
                    hour=row['hour'],
                    minute=row['minute']
                )
                session.add(lot_record)
                
                # Create records for each area in this lot
                remaining_occupancy = lot_occupancy
                for i, area in enumerate(lot_areas):
                    # Last area gets any remaining spots to ensure total matches
                    if i == len(lot_areas) - 1:
                        area_occupancy = remaining_occupancy
                    else:
                        # Distribute proportionally to area size
                        area_ratio = area.total_spaces / total_capacity
                        area_occupancy = int(lot_occupancy * area_ratio)
                        remaining_occupancy -= area_occupancy
                    
                    # Ensure we don't exceed the area's capacity
                    area_occupancy = min(area_occupancy, area.total_spaces)
                    
                    area_record = OccupancyRecord(
                        timestamp=row['timestamp'],
                        occupied_spaces=area_occupancy,
                        lot_id=lot.id,
                        area_id=area.id,
                        day_of_week=row['day_of_week'],
                        hour=row['hour'],
                        minute=row['minute']
                    )
                    session.add(area_record)
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        session.close()

# Initialize and seed the database if this file is run directly
if __name__ == "__main__":
    init_db()
    seed_usf_permits()
    seed_database()