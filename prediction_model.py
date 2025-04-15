import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

def train_prediction_model(historical_data):
    """
    Train a machine learning model to predict parking occupancy.
    
    Parameters:
    - historical_data: DataFrame with historical parking data
    
    Returns:
    - Trained model
    """
    # Create features and target
    X = historical_data[['day_of_week', 'hour', 'minute']]
    y = historical_data['occupancy']
    
    # Create preprocessing pipeline with one-hot encoding for day of week
    preprocessor = ColumnTransformer(
        transformers=[
            ('day_encoder', OneHotEncoder(), ['day_of_week'])
        ],
        remainder='passthrough'
    )
    
    # Create model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        ))
    ])
    
    # Train the model
    model.fit(X, y)
    
    return model

def predict_parking_availability(model, day_of_week, hour, minute):
    """
    Predict parking occupancy for a given time.
    
    Parameters:
    - model: Trained prediction model
    - day_of_week: Day of the week (0-6, where 0 is Monday)
    - hour: Hour of the day (0-23)
    - minute: Minute of the hour (0-59)
    
    Returns:
    - Predicted occupancy
    """
    # Create input data
    input_data = pd.DataFrame({
        'day_of_week': [day_of_week],
        'hour': [hour],
        'minute': [minute]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Ensure prediction is reasonable (non-negative integer)
    return max(0, int(round(prediction[0])))

def predict_next_day(model, total_spaces=200):
    """
    Generate predictions for the next 24 hours at hourly intervals.
    
    Parameters:
    - model: Trained prediction model
    - total_spaces: Total number of parking spaces
    
    Returns:
    - DataFrame with hourly predictions
    """
    current_time = datetime.now()
    predictions = []
    
    # Predict for the next 24 hours at hourly intervals
    for i in range(1, 25):
        predict_time = current_time + timedelta(hours=i)
        
        # Make prediction
        predicted_occupancy = predict_parking_availability(
            model, 
            predict_time.weekday(),
            predict_time.hour,
            0  # minute = 0 for the start of the hour
        )
        
        # Calculate occupancy percentage
        occupancy_pct = (predicted_occupancy / total_spaces) * 100
        
        predictions.append({
            'timestamp': predict_time,
            'predicted_occupancy': predicted_occupancy,
            'occupancy_pct': occupancy_pct,
            'available_spaces': total_spaces - predicted_occupancy
        })
    
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    # For testing purposes
    from data_generator import generate_parking_data
    
    # Generate sample data
    start_time = datetime.now() - timedelta(days=30)
    end_time = datetime.now()
    historical_data = generate_parking_data(start_time, end_time)
    
    # Train model
    model = train_prediction_model(historical_data)
    
    # Test prediction
    current_time = datetime.now()
    prediction = predict_parking_availability(
        model, 
        current_time.weekday(),
        current_time.hour,
        current_time.minute
    )
    
    print(f"Current time: {current_time}")
    print(f"Predicted occupancy: {prediction}")
    
    # Test next day prediction
    next_day = predict_next_day(model)
    print("\nNext 24 hours prediction:")
    print(next_day[['timestamp', 'predicted_occupancy', 'occupancy_pct']])
