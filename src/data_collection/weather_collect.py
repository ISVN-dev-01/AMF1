import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv('OPENWEATHER_API_KEY')

def get_weather(lat, lon, dt):
    """
    Get historical weather data for specific coordinates and time
    
    Args:
        lat: Latitude
        lon: Longitude  
        dt: Unix timestamp
    """
    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not found in environment variables")
    
    url = 'https://api.openweathermap.org/data/2.5/onecall/timemachine'
    params = {
        'lat': lat,
        'lon': lon,
        'dt': dt,
        'appid': API_KEY,
        'units': 'metric'  # Get temperature in Celsius
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_circuit_coordinates():
    """
    Return coordinates for F1 circuits
    """
    return {
        'bahrain': (26.0325, 50.5106),
        'saudi_arabia': (24.4672, 39.1042),
        'australia': (-37.8497, 144.9697),
        'azerbaijan': (40.3725, 49.8533),
        'miami': (25.9581, -80.2389),
        'monaco': (43.7347, 7.4206),
        'spain': (41.5700, 2.2611),
        'canada': (45.5050, -73.5228),
        'austria': (47.2197, 14.7647),
        'great_britain': (52.0786, -1.0169),
        'hungary': (47.5789, 19.2486),
        'belgium': (50.4372, 5.9714),
        'netherlands': (52.3888, 4.5403),
        'italy': (45.6156, 9.2811),
        'singapore': (1.2914, 103.8641),
        'japan': (34.8431, 136.5397),
        'united_states': (30.1328, -97.6411),
        'mexico': (19.4042, -99.0907),
        'brazil': (-23.7036, -46.6997),
        'abu_dhabi': (24.4672, 54.6031)
    }

def collect_race_weather(race_data):
    """
    Collect weather data for race events
    
    Args:
        race_data: DataFrame with race information including date and circuit
    """
    coordinates = get_circuit_coordinates()
    weather_data = []
    
    for _, race in race_data.iterrows():
        # Get circuit coordinates
        circuit_name = race.get('Circuit.circuitId', '').lower().replace(' ', '_')
        
        if circuit_name not in coordinates:
            print(f"No coordinates found for circuit: {circuit_name}")
            continue
        
        lat, lon = coordinates[circuit_name]
        
        # Convert race date to unix timestamp
        try:
            race_date = pd.to_datetime(race['date'])
            # Set to race time (usually afternoon local time, use UTC noon as approximation)
            race_datetime = race_date.replace(hour=12, minute=0, second=0, microsecond=0)
            dt = int(race_datetime.timestamp())
        except:
            print(f"Error parsing date for {race['raceName']}")
            continue
        
        print(f"Fetching weather for {race['raceName']} {race['season']}")
        
        weather = get_weather(lat, lon, dt)
        if weather and 'hourly' in weather:
            # Get weather data closest to race time
            hourly_data = weather['hourly'][0] if weather['hourly'] else {}
            
            weather_record = {
                'season': race['season'],
                'round': race['round'],
                'raceName': race['raceName'],
                'circuitId': race.get('Circuit.circuitId', ''),
                'date': race['date'],
                'lat': lat,
                'lon': lon,
                'temperature': hourly_data.get('temp'),
                'feels_like': hourly_data.get('feels_like'),
                'humidity': hourly_data.get('humidity'),
                'pressure': hourly_data.get('pressure'),
                'wind_speed': hourly_data.get('wind_speed'),
                'wind_deg': hourly_data.get('wind_deg'),
                'weather_main': hourly_data.get('weather', [{}])[0].get('main'),
                'weather_description': hourly_data.get('weather', [{}])[0].get('description'),
                'clouds': hourly_data.get('clouds'),
                'visibility': hourly_data.get('visibility')
            }
            
            weather_data.append(weather_record)
        
        # Rate limiting - OpenWeatherMap allows 1000 calls/day for free tier
        time.sleep(1)  # Wait 1 second between calls
    
    return pd.DataFrame(weather_data)

def main():
    """Main function to collect weather data"""
    # Create output directory
    output_dir = Path('data/raw/weather')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if races data exists
    races_file = Path('data/raw/races.parquet')
    if not races_file.exists():
        print("races.parquet not found. Run ergast_collect.py first.")
        return
    
    # Load races data
    races_df = pd.read_parquet(races_file)
    
    # Sample: Get weather for 2024 races only (to avoid hitting API limits)
    races_2024 = races_df[races_df['season'] == 2024].copy()
    
    print(f"Collecting weather data for {len(races_2024)} races in 2024...")
    weather_df = collect_race_weather(races_2024)
    
    if not weather_df.empty:
        weather_df.to_parquet('data/raw/weather/weather_2024.parquet', index=False)
        print(f"Saved weather data for {len(weather_df)} races")
    else:
        print("No weather data collected")

if __name__ == "__main__":
    main()