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
        
        # Create a main parking lot
        main_lot = ParkingLot(
            name="Downtown Parking Complex",
            total_spaces=200,
            latitude=37.7749,
            longitude=-122.4194
        )
        session.add(main_lot)
        session.flush()  # To get the ID
        
        # Create parking areas
        areas = [
            ParkingArea(name="Area A - Main", total_spaces=80, lot_id=main_lot.id),
            ParkingArea(name="Area B - North", total_spaces=60, lot_id=main_lot.id),
            ParkingArea(name="Area C - South", total_spaces=40, lot_id=main_lot.id),
            ParkingArea(name="Area D - VIP", total_spaces=20, lot_id=main_lot.id)
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
        
        for _, row in historical_data.iterrows():
            # Distribute occupancy across areas based on their size
            area_a_spaces = int(row['occupancy'] * 0.4)  # 40% in Area A
            area_b_spaces = int(row['occupancy'] * 0.3)  # 30% in Area B
            area_c_spaces = int(row['occupancy'] * 0.2)  # 20% in Area C
            area_d_spaces = int(row['occupancy'] * 0.1)  # 10% in Area D (VIP)
            
            # Add records for each area
            area_records = [
                OccupancyRecord(
                    timestamp=row['timestamp'],
                    occupied_spaces=area_a_spaces,
                    lot_id=main_lot.id,
                    area_id=areas_dict["Area A - Main"].id,
                    day_of_week=row['day_of_week'],
                    hour=row['hour'],
                    minute=row['minute']
                ),
                OccupancyRecord(
                    timestamp=row['timestamp'],
                    occupied_spaces=area_b_spaces,
                    lot_id=main_lot.id,
                    area_id=areas_dict["Area B - North"].id,
                    day_of_week=row['day_of_week'],
                    hour=row['hour'],
                    minute=row['minute']
                ),
                OccupancyRecord(
                    timestamp=row['timestamp'],
                    occupied_spaces=area_c_spaces,
                    lot_id=main_lot.id,
                    area_id=areas_dict["Area C - South"].id,
                    day_of_week=row['day_of_week'],
                    hour=row['hour'],
                    minute=row['minute']
                ),
                OccupancyRecord(
                    timestamp=row['timestamp'],
                    occupied_spaces=area_d_spaces,
                    lot_id=main_lot.id,
                    area_id=areas_dict["Area D - VIP"].id,
                    day_of_week=row['day_of_week'],
                    hour=row['hour'],
                    minute=row['minute']
                )
            ]
            session.add_all(area_records)
            
            # Also add a record for the overall lot
            lot_record = OccupancyRecord(
                timestamp=row['timestamp'],
                occupied_spaces=row['occupancy'],
                lot_id=main_lot.id,
                day_of_week=row['day_of_week'],
                hour=row['hour'],
                minute=row['minute']
            )
            session.add(lot_record)
        
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
    seed_database()