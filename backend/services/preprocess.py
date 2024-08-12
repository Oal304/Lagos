# services/preprocess.py

import pytz
import json
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from shapely.wkb import loads as wkb_loads
from geopy.distance import distance
from scipy.stats import mode

# Function to establish SQLAlchemy engine
def create_sqlalchemy_engine():
    engine = create_engine('mysql+mysqlconnector://root@localhost/traffic_data')
    return engine

# Function to fetch data from MySQL using SQLAlchemy
def fetch_data_from_mysql(query, engine):
    try:
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        print(f"Error reading data from MySQL: {e}")
        return None

# Function to calculate congestion level
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

# Function to preprocess historical traffic data
def preprocess_historical_data(df):
    df['timestamp'] = df['timestamp'].apply(lambda x: x - pd.Timedelta(hours=3))

    def extract_coordinates(coord_json):
        coordinates = json.loads(coord_json)['coordinate']
        latitudes = [point['latitude'] for point in coordinates]
        longitudes = [point['longitude'] for point in coordinates]
        return latitudes, longitudes

    df['latitudes'], df['longitudes'] = zip(*df['coordinates'].apply(extract_coordinates))

    max_points = max(df['latitudes'].apply(len).max(), df['longitudes'].apply(len).max())
    latitude_cols = [f'latitude_{i}' for i in range(max_points)]
    longitude_cols = [f'longitude_{i}' for i in range(max_points)]

    df = pd.concat([df, pd.DataFrame(df['latitudes'].tolist(), columns=latitude_cols, index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(df['longitudes'].tolist(), columns=longitude_cols, index=df.index)], axis=1)

    df.drop(columns=['coordinates', 'latitudes', 'longitudes'], inplace=True)

    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['unix_timestamp'] = df['timestamp'].astype('int64') // 10**9

    df['congestion_level'] = df.apply(lambda row: calculate_congestion_level(row['current_speed'], row['free_flow_speed']), axis=1)

    for column in df.columns:
        if df[column].isnull().any():
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)

    numerical_features = ['current_speed', 'free_flow_speed', 'current_travel_time', 'free_flow_travel_time']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df[['timestamp', 'latitude_0', 'longitude_0', 'day_of_week', 'hour_of_day', 'unix_timestamp', 'congestion_level']]

# Function to interpolate data to 15-minute intervals
def interpolate_to_15_min(df, time_col='timestamp'):
    df.set_index(time_col, inplace=True)
    
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    df = df.resample('15min').interpolate(method='linear')  # Updated to '15min'
    df.reset_index(inplace=True)
    return df

# Function to preprocess weather data
def preprocess_weather_data(weather_df):
    try:
        weather_df['datetime'] = weather_df['datetime'].apply(lambda x: x - pd.Timedelta(hours=3))
        weather_df['day_of_week'] = weather_df['datetime'].dt.dayofweek
        weather_df['hour'] = weather_df['datetime'].dt.hour

        numerical_cols = ['temperature', 'precipitation', 'wind_speed', 'visibility', 'humidity', 'pressure']
        for column in numerical_cols:
            if weather_df[column].isnull().any():
                mode_value = weather_df[column].mode()[0]
                weather_df[column].fillna(mode_value, inplace=True)

        weather_df[numerical_cols] = (weather_df[numerical_cols] - weather_df[numerical_cols].mean()) / weather_df[numerical_cols].std()

        return weather_df
    except Exception as e:
        print(f"Error in preprocessing weather data: {e}")
        return None

# Function to preprocess OSM data
def preprocess_osm_data(df):
    if 'traffic_signal' in df.columns:
        df.drop(['traffic_signal'], axis=1, inplace=True)
    
    df['road_length'] = df.apply(lambda row: calculate_road_length(row['way_geometry']), axis=1)
    df = pd.get_dummies(df, columns=['highway'], prefix='highway')
    
    scaler = MinMaxScaler()
    df[['latitude', 'longitude', 'length', 'road_length']] = scaler.fit_transform(df[['latitude', 'longitude', 'length', 'road_length']])
    
    return df

def calculate_road_length(way_geometry):
    try:
        geom = wkb_loads(bytes.fromhex(way_geometry))
        total_length = 0.0
        previous_point = None
        for current_point in geom.coords:
            if previous_point is not None:
                dist = distance(previous_point, current_point).meters
                total_length += dist
            previous_point = current_point
        return total_length
    except Exception as e:
        print(f"Error calculating road length: {e}")
        return 0.0

def fetch_and_preprocess_osm_data(table_name, db_uri):
    engine = create_engine(db_uri)
    try:
        query = f"SELECT road_id, latitude, longitude, highway, way_geometry, length FROM {table_name}"
        osm_df = pd.read_sql(query, con=engine)
    except Exception as e:
        print(f"Error fetching data from PostgreSQL: {e}")
        return None
    finally:
        engine.dispose()
    
    if osm_df is not None:
        osm_df_cleaned = preprocess_osm_data(osm_df)
        return osm_df_cleaned
    else:
        print("No data available for preprocessing.")
        return None
    
# Function to merge historical, weather, and OSM data
def merge_data(preprocessed_historical_data, preprocessed_weather_data, preprocessed_osm_data):
    # Ensure timestamp is in datetime format
    preprocessed_historical_data['timestamp'] = pd.to_datetime(preprocessed_historical_data['timestamp'])
    preprocessed_weather_data['datetime'] = pd.to_datetime(preprocessed_weather_data['datetime'])

    # Merge historical traffic data with weather data based on closest timestamp
    merged_df = pd.merge_asof(
        preprocessed_historical_data.sort_values('timestamp'),
        preprocessed_weather_data.sort_values('datetime'),
        left_on='timestamp',
        right_on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta('15min')  # Tolerance set to 15 minutes
    )

    # Extract day_of_week and hour_of_day from timestamp
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    merged_df['hour_of_day'] = merged_df['timestamp'].dt.hour  # Keep hour_of_day from historical data
    
    # Merge with OSM data based on location (latitude and longitude)
    def find_nearest_osm(row, osm_df):
        lat = row['latitude_0']
        lon = row['longitude_0']
        distances = ((osm_df['latitude'] - lat) ** 2 + (osm_df['longitude'] - lon) ** 2)
        nearest_index = distances.idxmin()
        return osm_df.loc[nearest_index]

    nearest_osm_df = merged_df.apply(lambda row: find_nearest_osm(row, preprocessed_osm_data), axis=1)
    nearest_osm_df.reset_index(drop=True, inplace=True)  # Reset index for proper concatenation

    merged_df = pd.concat([merged_df, nearest_osm_df], axis=1)  # Concatenate along columns

    return merged_df

# Example usage to fetch, preprocess, and merge data
if __name__ == "__main__":
    engine = create_sqlalchemy_engine()
    
    historical_data = fetch_data_from_mysql("SELECT * FROM historical_traffic_data", engine)
    if historical_data is not None:
        preprocessed_historical_data = preprocess_historical_data(historical_data)
        preprocessed_historical_data = interpolate_to_15_min(preprocessed_historical_data, time_col='timestamp')

        for column in preprocessed_historical_data.columns:
            if preprocessed_historical_data[column].isnull().any():
                mode_value = preprocessed_historical_data[column].mode()[0]
                preprocessed_historical_data[column] = preprocessed_historical_data[column].fillna(mode_value)

        print("Preprocessed Historical Data:")
        print(preprocessed_historical_data.head())

    weather_df = fetch_data_from_mysql("SELECT * FROM weather", engine)
    if weather_df is not None:
        preprocessed_weather_data = preprocess_weather_data(weather_df)

        for column in preprocessed_weather_data.columns:
            if preprocessed_weather_data[column].isnull().any():
                mode_value = preprocessed_weather_data[column].mode()[0]
                preprocessed_weather_data[column] = preprocessed_weather_data[column].fillna(mode_value)

        print("\nPreprocessed Weather Data:")
        print(preprocessed_weather_data.head())

    osm_data = fetch_and_preprocess_osm_data("osm_road_data", "postgresql+psycopg2://postgres:password@localhost:5432/postgres")
    if osm_data is not None:
        print("\nPreprocessed OSM Data:")
        print(osm_data.head())

    if all(data is not None for data in [preprocessed_historical_data, preprocessed_weather_data, osm_data]):
        merged_data = merge_data(preprocessed_historical_data, preprocessed_weather_data, osm_data)
        print("\nMerged Data:")
        print(merged_data.tail())

        # Fill missing values and drop specific dates
        columns_to_fill = ['id', 'datetime', 'temperature', 'precipitation', 'wind_speed', 
                            'visibility', 'humidity', 'pressure', 'weather_description', 'day_of_week', 'day_of_week_y', 'hour']
        for column in columns_to_fill:
            col_mode = merged_data[column].mode()[0]
            merged_data[column] = merged_data[column].fillna(col_mode)
        merged_data = merged_data[merged_data['timestamp'].dt.date != pd.Timestamp('2024-07-04').date()]

        # Save the merged_data DataFrame to a CSV file
        merged_data.to_csv('../data/merged_data.csv', index=False)

    engine.dispose()


