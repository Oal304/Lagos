# services/traffic.py

import requests
import pandas as pd
import xmltodict
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch environment variables
postgres_db_username = os.getenv('POSTGRES_DB_USERNAME')
postgres_db_password = os.getenv('POSTGRES_DB_PASSWORD')
postgres_db_host = os.getenv('POSTGRES_DB_HOST')
postgres_db_port = os.getenv('POSTGRES_DB_PORT')
postgres_db_name = os.getenv('POSTGRES_DB_NAME')

# Construct the database URI using the fetched environment variables
postgres_db_uri = f"postgresql+psycopg2://{postgres_db_username}:{postgres_db_password}@{postgres_db_host}:{postgres_db_port}/{postgres_db_name}"


# Create a database engine
# Connect to the database
engine = create_engine(postgres_db_uri)

# Function to load OSM data from PostgreSQL
def load_osm_data():
    # Assuming the table name is "osm_road_data"
    table_name = "osm_road_data"
    try:
        # Specify columns to load, excluding 'geometry'
        columns_to_load = ['road_id', 'road_name', 'latitude', 'longitude', 'length']  # Adjust based on your actual schema
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", engine).loc[:, columns_to_load]
        return df
    except SQLAlchemyError as e:
        print(f"Error loading OSM data: {e}")
        return None

# Function to get current traffic data for all points
def get_current_traffic_data():
    # Load the OSM data into a DataFrame
    df_osm = load_osm_data()
    if df_osm is None:
        return json.dumps({"error": "Failed to load OSM data"})

    # Filter the DataFrame for roads within Lagos
    bbox = [6.2950575, 3.2341795, 6.6150575, 3.5541795]
    df_lagos_roads = df_osm[(df_osm['latitude'].between(bbox[0], bbox[2])) &
                            (df_osm['longitude'].between(bbox[1], bbox[3]))]

    # Extract points of interest (latitude and longitude) from the filtered DataFrame
    points_of_interest = [{'lat': row['latitude'], 'lon': row['longitude']} for _, row in df_lagos_roads.iterrows()]

    # TomTom API credentials
    API_KEY = os.getenv('TOMTOM_API_KEY')  # Fetch API key from environment variables

    # Prepare the batch request payload
    batch_payload = {
        "batchItems": [
            {
                "query": f"/traffic/services/4/flowSegmentData/absolute/10/xml?key={API_KEY}&point={point['lat']},{point['lon']}"
            } for point in points_of_interest
        ]
    }

    # Make the batch POST request to the TomTom API
    url = f'https://api.tomtom.com/search/2/batch.json?key={API_KEY}'
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(batch_payload))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        all_traffic_data = []

        # Iterate through each batch item result
        for item in response_data['batchItems']:
            if 'response' in item and item['response']['statusCode'] == 200:
                # Convert the XML response to a Python dictionary
                data_dict = xmltodict.parse(item['response']['body'])

                # Remove coordinates from the response data if present
                if 'flowSegmentData' in data_dict and 'coordinates' in data_dict['flowSegmentData']:
                    del data_dict['flowSegmentData']['coordinates']

                # Extract latitude and longitude from the request query
                query_params = item['request']['query'].split('&')
                lat_lon = {param.split('=')[0]: param.split('=')[1] for param in query_params if 'point=' in param}
                lat, lon = map(float, lat_lon['point'].split(','))

                # Add the latitude and longitude to the data
                data_dict['flowSegmentData']['latitude'] = lat
                data_dict['flowSegmentData']['longitude'] = lon

                # Add the traffic data to the dictionary
                all_traffic_data.append(data_dict['flowSegmentData'])
            else:
                print(f"Failed to retrieve data for a point. Status code: {item['response']['statusCode']}")

        return json.dumps(all_traffic_data, indent=4)
    else:
        print(f"Batch request failed. Status code: {response.status_code}")
        return json.dumps({"error": "Batch request failed"})

# Function to get current traffic data for a specific point
def get_traffic_data_at_point(latitude, longitude):
    # TomTom API credentials
    API_KEY = os.getenv('TOMTOM_API_KEY')  # Fetch API key from environment variables
    
    # Construct the API request URL
    url = f'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/xml?key={API_KEY}&point={latitude},{longitude}'

    # Make the GET request to the TomTom API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the XML response to a Python dictionary
        data_dict = xmltodict.parse(response.text)
        
        # Remove coordinates from the response data if present
        if 'flowSegmentData' in data_dict and 'coordinates' in data_dict['flowSegmentData']:
            del data_dict['flowSegmentData']['coordinates']
        
        # Add the latitude and longitude to the data
        data_dict['flowSegmentData']['latitude'] = latitude
        data_dict['flowSegmentData']['longitude'] = longitude
        
        # Convert the data dictionary to a JSON string
        traffic_data_json = json.dumps(data_dict['flowSegmentData'], indent=4)
        
        # Print the fetched data to the console
        print(traffic_data_json)
        
        return traffic_data_json
    else:
        print(f"Request failed. Status code: {response.status_code}")
        return json.dumps({"error": "Request failed"})
