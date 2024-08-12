# models/prediction.py

from backend.models import model

def get_realtime_traffic_info(lat, lon):
    # Logic to fetch real-time traffic data for specific coordinates
    return model.get_current_traffic(lat, lon)

def get_all_realtime_traffic():
    # Logic to fetch all real-time traffic data
    return model.get_all_current_traffic()

def check_congestion_level(location):
    # Logic to check congestion level at specific location
    return model.get_congestion_level(location)

def suggest_alternative_routes(start_lat, start_lon, end_lat, end_lon):
    # Logic to suggest alternative routes
    return model.get_alternative_routes(model.G, (start_lat, start_lon), (end_lat, end_lon))

def get_directions_with_traffic(start_lat, start_lon, end_lat, end_lon):
    # Logic to get directions with real-time traffic considerations
    return model.get_directions(model.G, (start_lat, start_lon), (end_lat, end_lon))

def get_prediction_traffic_info(lat, lon):
    # Fetch real-time data and prepare features for prediction
    real_time_data = model.preprocess_realtime_data(lat, lon)
    
    if real_time_data is None:
        return {'error': 'Failed to fetch real-time data'}

    # Define forecast horizons
    horizons = [15, 30, 45, 60, 120, 180]  # in minutes
    
    # Fetch predictions
    predictions = {}
    for horizon in horizons:
        prediction_result = model.predict_congestion(lat, lon)
        if 'error' in prediction_result:
            return prediction_result
        
        predictions[horizon] = prediction_result.get(f'{horizon} mins')
    
    return predictions  
    
