# models/model.py

import pandas as pd
from sqlalchemy import create_engine
import geopandas as gpd
from backend.services.traffic import get_current_traffic_data, get_traffic_data_at_point
from backend.services.weather import get_current_weather_data
from shapely.geometry import Point
import osmnx as ox
import networkx as nx
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import holidays
from datetime import datetime, timedelta
from pytz import timezone  # Add this import
from sklearn.preprocessing import StandardScaler
import pytz

# Database connection details
# Fetch environment variables
postgres_db_username = os.getenv('POSTGRES_DB_USERNAME')
postgres_db_password = os.getenv('POSTGRES_DB_PASSWORD')
postgres_db_host = os.getenv('POSTGRES_DB_HOST')
postgres_db_port = os.getenv('POSTGRES_DB_PORT')
postgres_db_name = os.getenv('POSTGRES_DB_NAME')

# Construct the database URI using the fetched environment variables
postgres_db_uri = f"postgresql+psycopg2://{postgres_db_username}:{postgres_db_password}@{postgres_db_host}:{postgres_db_port}/{postgres_db_name}"


# Connect to the database
engine = create_engine(postgres_db_uri)
final_merged_data = pd.read_sql_table('final_merged_data', engine)

# Load the trained model and scaler
model = load_model('models/traffic_model.keras')
# scalers = joblib.load(r'C:\Users\User\OneDrive\Documents\Honours Project\traffic-prediction-api\backend\models\scalers.pkl')

# Define the bounding box for Lagos
north, south, east, west = 6.615057, 6.295058, 3.554180, 3.234180

# Enable caching
ox.settings.use_cache = True
ox.settings.log_console = False

# Global variables to hold the graph and edges GeoDataFrame
G = None
edges_gdf = None

def load_graph():
    global G
    if G is None:
        cache_file = 'cache/lagos_graph.graphml'
        if os.path.exists(cache_file):
            G = ox.load_graphml(cache_file)
        else:
            bbox = (north, south, east, west)
            G = ox.graph_from_bbox(*bbox, network_type='drive')
            ox.save_graphml(G, cache_file)

        for index, row in final_merged_data.iterrows():
            u, v, key = ox.nearest_edges(G, row['longitude__left'], row['latitude__left'])
            G[u][v][key]['current_speed'] = row['current_speed']
            G[u][v][key]['free_flow_speed'] = row['free_flow_speed']
            G[u][v][key]['congestion_level'] = row['congestion_level']

def get_edges_gdf():
    global edges_gdf
    if edges_gdf is None:
        load_graph()
        edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    return edges_gdf

def calculate_congestion_level(current_speed, free_flow_speed):
    if current_speed >= free_flow_speed:
        return 0  # No congestion
    elif current_speed >= 0.75 * free_flow_speed:
        return 1  # Low congestion
    elif current_speed >= 0.5 * free_flow_speed:
        return 2  # Medium congestion
    elif current_speed >= 0.25 * free_flow_speed:
        return 3  # High congestion
    else:
        return 4  # Severe congestion

def get_all_current_traffic():
    traffic_data_json = get_current_traffic_data()
    print("Raw traffic data JSON:", traffic_data_json)  # Debugging line

    try:
        traffic_data = json.loads(traffic_data_json)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return {"error": "Failed to decode traffic data"}

    if not isinstance(traffic_data, list):
        return {"error": "Expected a list of traffic data"}

    all_traffic_info = []

    for data in traffic_data:
        if data.get('currentSpeed') and data.get('freeFlowSpeed'):
            current_speed = float(data['currentSpeed'])
            free_flow_speed = float(data['freeFlowSpeed'])
            congestion_level = calculate_congestion_level(current_speed, free_flow_speed)
            current_travel_time = float(data.get('currentTravelTime', 0))  # Default to 0 if not present
            road_closure = data.get('roadClosure', 'unknown')  # Default to 'unknown' if not present

            traffic_info = {
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'current_speed': current_speed,
                'congestion_level': congestion_level,
                'current_travel_time': current_travel_time,
                'road_closure': road_closure
            }
            all_traffic_info.append(traffic_info)

    return all_traffic_info

def get_current_traffic(lat, lon):
    try:
        # Debugging log to check if the function is called
        print(f"Calling get_current_traffic with latitude {lat} and longitude {lon}")
        
        traffic_data = get_traffic_data_at_point(lat, lon)
        
        # Debugging log to check if traffic data is successfully fetched
        print("Traffic data fetched:", traffic_data)
        
        if traffic_data:
            # Parse JSON string to a dictionary if necessary
            if isinstance(traffic_data, str):
                traffic_data = json.loads(traffic_data)
            
            # Convert string values to numeric values
            current_speed = float(traffic_data['currentSpeed'])
            free_flow_speed = float(traffic_data['freeFlowSpeed'])
            congestion_level = calculate_congestion_level(current_speed, free_flow_speed)
            road_closure = traffic_data['roadClosure'] == 'true'
            
            result = {
                'latitude': lat,
                'longitude': lon,
                'current_speed': current_speed,
                'free_flow_speed': free_flow_speed,
                'congestion_level': congestion_level,
                'road_closure': road_closure
            }
            
            # Debugging log to show the final result
            print("Final result:", result)
            
            return result
        
        else:
            # Debugging log to indicate no traffic data was available
            print("No traffic data available")
            return {'error': 'No traffic data available'}
    
    except Exception as e:
        # Debugging log to catch and display exceptions
        print(f"Exception occurred: {str(e)}")
        return {'error': str(e)}  

def get_congestion_level(location):
    point = Point(location[1], location[0])
    road_info = final_merged_data[final_merged_data['geometry'].apply(lambda x: point.intersects(Point(x)))]
    if not road_info.empty:
        lat = road_info.iloc[0]['latitude__left']
        lon = road_info.iloc[0]['longitude__left']
        traffic_info = get_current_traffic(lat, lon)
        weather_info = get_current_weather_data()
        return {**traffic_info, **weather_info}
    return None

def get_alternative_routes(G, start_point, end_point):
    load_graph()
    start_node = ox.nearest_nodes(G, start_point[1], start_point[0])
    end_node = ox.nearest_nodes(G, end_point[1], end_point[0])

    def weight_func(u, v, d):
        return d['length'] / d.get('current_speed', 1)

    route = nx.shortest_path(G, start_node, end_node, weight=weight_func)

    low_congestion_routes = []
    for path in nx.all_simple_paths(G, start_node, end_node, cutoff=10):
        congestion_levels = [G[u][v][0].get('congestion_level', 0) for u, v in zip(path[:-1], path[1:])]
        if all(cl < 2 for cl in congestion_levels):
            low_congestion_routes.append(path)

    return low_congestion_routes if low_congestion_routes else [route]

def get_directions(G, start_point, end_point):
    load_graph()
    start_node = ox.nearest_nodes(G, start_point[1], start_point[0])
    end_node = ox.nearest_nodes(G, end_point[1], end_point[0])

    def weight_func(u, v, d):
        return d['length'] / d.get('current_speed', 1)

    route = nx.shortest_path(G, start_node, end_node, weight=weight_func)
    return route

# Define the look-back window size
LOOK_BACK = 12

# Set up timezones
mauritius_tz = timezone('Indian/Mauritius')
nigeria_tz = timezone('Africa/Lagos')

scalers = {
    'congestion_level': StandardScaler(),
    'temperature': StandardScaler(),
    'precipitation': StandardScaler(),
    'wind_speed': StandardScaler(),
    'visibility': StandardScaler(),
    'humidity': StandardScaler(),
    'pressure': StandardScaler(),
    'hour': StandardScaler(),
    'day_of_week': StandardScaler(),
    'latitude': StandardScaler(),
    'longitude': StandardScaler(),
    'timestamp': StandardScaler(),  # Add scaler for Unix timestamp
}

# Load historical data
def load_historical_data(file_path):
    df = pd.read_csv(file_path)
    columns_to_keep = [
        'timestamp', 'latitude', 'longitude', 'day_of_week', 'hour', 
        'temperature', 'precipitation', 'wind_speed', 'visibility', 
        'humidity', 'pressure', 'congestion_level'
    ]
    df = df[columns_to_keep]
    
    # Convert timestamp to Unix timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
    
    return df

historical_df = load_historical_data('./data/merged_data.csv')

# Fit the scalers on the historical data
for feature in scalers.keys():
    feature_data = historical_df[feature].values.reshape(-1, 1)
    scalers[feature].fit(feature_data)

def preprocess_realtime_data(lat, lon):
    # Fetch traffic and weather data
    traffic_data_str = get_traffic_data_at_point(lat, lon)
    weather_data = get_current_weather_data()

    print(f"Traffic data fetched: {traffic_data_str}")  # Debugging line

    # Initialize traffic_data to an empty dictionary if it's None
    if traffic_data_str is None:
        traffic_data = {}
    else:
        try:
            traffic_data = json.loads(traffic_data_str)
        except json.JSONDecodeError:
            print("Error decoding JSON for traffic data.")
            return None
    
    # Check if traffic_data contains the required keys
    if not traffic_data or not weather_data:
        print("Missing data: traffic_data or weather_data")
        return None

    # Extract traffic data
    current_speed = float(traffic_data.get('currentSpeed', 0))
    free_flow_speed = float(traffic_data.get('freeFlowSpeed', 0))
    congestion_level = calculate_congestion_level(current_speed, free_flow_speed)

    # Extract weather data
    temperature = weather_data.get('temperature', 0)
    precipitation = weather_data.get('precipitation', 0)
    wind_speed = weather_data.get('wind_speed', 0)
    visibility = weather_data.get('visibility', 0)
    humidity = weather_data.get('humidity', 0)
    pressure = weather_data.get('pressure', 0)
    
    # Current time in Lagos timezone
    current_time_mut = datetime.now(mauritius_tz)
    current_time_ngn = current_time_mut.astimezone(nigeria_tz)
    
    # Convert current time to Unix timestamp
    unix_timestamp = int(current_time_ngn.timestamp())
    
    features = {
        'congestion_level': congestion_level,
        'temperature': temperature,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'visibility': visibility,
        'humidity': humidity,
        'pressure': pressure,
        'hour': current_time_ngn.hour,
        'day_of_week': current_time_ngn.weekday(),
        'latitude': lat,
        'longitude': lon,
        'timestamp': unix_timestamp
    }

    df = pd.DataFrame([features])

    # Append the new data to the historical data
    global historical_df
    historical_df = pd.concat([historical_df, df], ignore_index=True)
    
    # Ensure we have enough data to create sequences
    if len(historical_df) < LOOK_BACK:
        print("Not enough historical data to create sequences")
        return None

    # Keep only the last `LOOK_BACK` timesteps
    historical_df = historical_df[-LOOK_BACK:]
    
    # Scale features
    return scale_features(historical_df)

def scale_features(features_df):
    for column in features_df.columns:
        if column in scalers:
            features_df[column] = scalers[column].transform(features_df[[column]])
    
    # Reshape the DataFrame to match the LSTM input shape
    features_array = features_df.values.reshape(1, LOOK_BACK, -1)
    
    # Ensure the features are of type float32
    features_array = features_array.astype(np.float32)
    
    return features_array

def predict_congestion(lat, lon):
    horizons = [15, 30, 45, 60, 120, 180]  # in minutes
    data = preprocess_realtime_data(lat, lon)
    if data is None:
        return {'error': 'Failed to fetch real-time data'}
    
    predictions = {}
    for horizon in horizons:
        prediction = model.predict(data)
        congestion_levels = ['No congestion', 'Low congestion', 'Medium congestion', 'High congestion', 'Severe congestion']
        
        # Assuming the model outputs a probability distribution across the congestion levels for each horizon
        # Adjust the indexing according to your model's output shape
        predicted_congestion = congestion_levels[np.argmax(prediction)]
        
        future_time = datetime.now(mauritius_tz) + timedelta(minutes=horizon)
        future_time = future_time.astimezone(nigeria_tz)
        
        predictions[f'{horizon} mins'] = {
            'latitude': lat,
            'longitude': lon,
            'predicted_congestion_level': predicted_congestion,
            'prediction_time': future_time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    return predictions