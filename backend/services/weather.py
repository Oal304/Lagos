# services/weather.py

import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_current_weather_data():
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')  # Fetch API key from environment variables

    url = f'https://api.openweathermap.org/data/2.5/weather?q=Lagos,NG&appid={api_key}'

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_data = {
            'temperature': data['main']['temp'] - 273.15,
            'precipitation': data['rain']['1h'] if 'rain' in data else 0,
            'wind_speed': data['wind']['speed'],
            'visibility': data['visibility'] / 1000,
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure']
        }

        print(weather_data)
        return weather_data
    else:
        print("Failed to fetch data from OpenWeatherMap API")
        return None
