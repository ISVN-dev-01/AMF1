#!/usr/bin/env python3
"""
F1 Race Weekend Simulator
Simulate a complete race weekend with qualifying and race predictions
"""

import requests
import json
import random
import time
from datetime import datetime

# Real F1 data for 2024 season
DRIVERS = {
    1: {"name": "Max Verstappen", "team": "Red Bull Racing"},
    11: {"name": "Sergio Pérez", "team": "Red Bull Racing"},
    16: {"name": "Charles Leclerc", "team": "Ferrari"},
    55: {"name": "Carlos Sainz", "team": "Ferrari"},
    44: {"name": "Lewis Hamilton", "team": "Mercedes"},
    63: {"name": "George Russell", "team": "Mercedes"},
    4: {"name": "Lando Norris", "team": "McLaren"},
    81: {"name": "Oscar Piastri", "team": "McLaren"},
    14: {"name": "Fernando Alonso", "team": "Aston Martin"},
    18: {"name": "Lance Stroll", "team": "Aston Martin"}
}

CIRCUITS = {
    1: "Monaco GP",
    2: "Silverstone GP", 
    3: "Spa-Francorchamps GP",
    4: "Monza GP",
    5: "Suzuka GP",
    6: "Interlagos GP",
    7: "Bahrain GP",
    8: "Abu Dhabi GP"
}

WEATHER_SCENARIOS = {
    "dry": {"temp": (20, 30), "humidity": (30, 50), "rain": (0, 10)},
    "hot": {"temp": (30, 40), "humidity": (20, 40), "rain": (0, 5)},
    "cool": {"temp": (15, 22), "humidity": (40, 70), "rain": (5, 20)},
    "wet": {"temp": (15, 25), "humidity": (70, 90), "rain": (60, 100)},
    "mixed": {"temp": (18, 26), "humidity": (50, 75), "rain": (30, 60)}
}

def generate_race_weekend_scenario():
    """Generate a realistic race weekend scenario"""
    
    # Pick random circuit and weather
    circuit_id = random.choice(list(CIRCUITS.keys()))
    circuit_name = CIRCUITS[circuit_id]
    weather_type = random.choice(list(WEATHER_SCENARIOS.keys()))
    weather_data = WEATHER_SCENARIOS[weather_type]
    
    # Generate weather conditions
    temperature = round(random.uniform(*weather_data["temp"]), 1)
    humidity = round(random.uniform(*weather_data["humidity"]), 1)
    rain_probability = round(random.uniform(*weather_data["rain"]), 1)
    
    # Select 8-10 drivers for the race
    selected_drivers = random.sample(list(DRIVERS.keys()), random.randint(8, 10))
    
    drivers_data = []
    for driver_id in selected_drivers:
        # Add some variation in conditions per driver (tire strategy, setup, etc.)
        temp_variation = random.uniform(-2, 2)
        humidity_variation = random.uniform(-5, 5)
        
        driver_data = {
            "driver_id": driver_id,
            "team_id": driver_id % 10 + 1,  # Simple team mapping
            "circuit_id": circuit_id,
            "weather_condition": weather_type,
            "temperature": round(temperature + temp_variation, 1),
            "humidity": max(0, min(100, round(humidity + humidity_variation, 1))),
            "rain_probability": rain_probability,
            "track_temperature": round(temperature + random.uniform(5, 15), 1),
            "wind_speed": round(random.uniform(0, 12), 1),
            "air_pressure": round(random.uniform(1010, 1020), 2)
        }
        drivers_data.append(driver_data)
    
    return {
        "circuit_name": circuit_name,
        "circuit_id": circuit_id,
        "weather_type": weather_type,
        "conditions": {
            "temperature": temperature,
            "humidity": humidity,
            "rain_probability": rain_probability
        },
        "drivers": drivers_data
    }

def make_api_request(endpoint, data, base_url="http://localhost:8080"):
    """Make API request with error handling"""
    try:
        response = requests.post(f"{base_url}/{endpoint}", 
                               json={"drivers": data},
                               headers={"Content-Type": "application/json"},
                               timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return None

def simulate_race_weekend():
    """Simulate a complete F1 race weekend"""
    
    print("🏎️  F1 RACE WEEKEND SIMULATOR")
    print("=" * 60)
    
    # Check API availability
    try:
        health_check = requests.get("http://localhost:8080/health", timeout=5)
        if health_check.status_code != 200:
            print("❌ F1 Prediction API is not running!")
            print("   Start it with: python quick_start.py")
            return
    except:
        print("❌ Cannot connect to F1 Prediction API!")
        print("   Start it with: python quick_start.py")
        return
    
    # Generate race weekend scenario
    scenario = generate_race_weekend_scenario()
    
    print(f"🏁 {scenario['circuit_name']}")
    print(f"🌤️  Weather: {scenario['weather_type'].title()} conditions")
    print(f"🌡️  Temperature: {scenario['conditions']['temperature']}°C")
    print(f"💧 Humidity: {scenario['conditions']['humidity']}%")
    print(f"🌧️  Rain Probability: {scenario['conditions']['rain_probability']}%")
    print(f"👥 Drivers: {len(scenario['drivers'])} competing")
    print()
    
    # List participating drivers
    print("🏎️  Grid:")
    for i, driver_data in enumerate(scenario['drivers']):
        driver_id = driver_data['driver_id']
        driver_info = DRIVERS[driver_id]
        print(f"   {i+1:2d}. {driver_info['name']} ({driver_info['team']})")
    print()
    
    # QUALIFYING SESSION
    print("🏁 QUALIFYING SESSION")
    print("-" * 40)
    
    quali_result = make_api_request("predict_quali", scenario['drivers'])
    
    if quali_result:
        # Sort by predicted time
        quali_sorted = sorted(quali_result, key=lambda x: x['predicted_time'])
        
        print("📊 Qualifying Results:")
        for i, pred in enumerate(quali_sorted):
            driver_info = DRIVERS[pred['driver_id']]
            print(f"   P{i+1:2d}: {driver_info['name']:<20} {pred['predicted_time']:.3f}s "
                  f"({pred['probability_score']:.2f})")
        
        pole_sitter = quali_sorted[0]
        pole_driver = DRIVERS[pole_sitter['driver_id']]
        print(f"\n🥇 POLE POSITION: {pole_driver['name']} - {pole_sitter['predicted_time']:.3f}s")
    else:
        print("❌ Qualifying prediction failed")
        return
    
    print("\n" + "="*60)
    
    # RACE SIMULATION
    print("🏆 RACE PREDICTION")
    print("-" * 40)
    
    race_result = make_api_request("predict_race", scenario['drivers'])
    
    if race_result:
        # Sort by win probability
        race_sorted = sorted(race_result, key=lambda x: x['win_probability'], reverse=True)
        
        print("📊 Race Winner Predictions:")
        for i, pred in enumerate(race_sorted):
            driver_info = DRIVERS[pred['driver_id']]
            win_pct = pred['win_probability'] * 100
            print(f"   {i+1:2d}. {driver_info['name']:<20} {win_pct:5.1f}% "
                  f"(confidence: {pred['confidence_score']:.2f})")
        
        race_favorite = race_sorted[0]
        favorite_driver = DRIVERS[race_favorite['driver_id']]
        print(f"\n🏆 RACE FAVORITE: {favorite_driver['name']} - "
              f"{race_favorite['win_probability']*100:.1f}% win chance")
    else:
        print("❌ Race prediction failed")
        return
    
    print("\n" + "="*60)
    
    # WEEKEND SUMMARY
    print("📋 RACE WEEKEND SUMMARY")
    print("-" * 40)
    
    if quali_result and race_result:
        pole_driver_name = DRIVERS[pole_sitter['driver_id']]['name']
        favorite_driver_name = DRIVERS[race_favorite['driver_id']]['name']
        
        print(f"🏁 Circuit: {scenario['circuit_name']}")
        print(f"🌤️  Conditions: {scenario['weather_type'].title()}")
        print(f"🥇 Pole Position: {pole_driver_name}")
        print(f"🏆 Race Favorite: {favorite_driver_name}")
        
        if pole_sitter['driver_id'] == race_favorite['driver_id']:
            print(f"⭐ {pole_driver_name} is both pole-sitter AND race favorite!")
        else:
            print(f"🔀 Different drivers for pole and race win prediction")
        
        # Weather impact analysis
        if scenario['conditions']['rain_probability'] > 50:
            print("🌧️  High rain chance - expect strategy surprises!")
        elif scenario['weather_type'] == 'hot':
            print("🔥 Hot conditions - tire management will be crucial!")
        elif scenario['weather_type'] == 'cool':
            print("❄️  Cool conditions - good for tire performance!")
    
    print("\n🎉 Race weekend simulation complete!")
    print("💡 Start the API with 'python quick_start.py' to run your own simulations!")

if __name__ == "__main__":
    simulate_race_weekend()