#!/usr/bin/env python3
"""
Random F1 Prediction Generator
Make predictions with randomized driver data
"""

import requests
import json
import random
from typing import List, Dict

def generate_random_driver_data(num_drivers: int = 5) -> List[Dict]:
    """Generate random driver data for predictions"""
    
    # Real F1 driver IDs (2024 season)
    driver_ids = [1, 11, 16, 55, 4, 81, 14, 18, 63, 31, 44, 77, 23, 24, 10, 27, 22, 3, 20, 2]
    team_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Team IDs
    circuit_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Circuit IDs
    
    weather_conditions = ["dry", "wet", "mixed", "cloudy"]
    
    drivers = []
    selected_drivers = random.sample(driver_ids, min(num_drivers, len(driver_ids)))
    
    for driver_id in selected_drivers:
        driver_data = {
            "driver_id": driver_id,
            "team_id": random.choice(team_ids),
            "circuit_id": random.choice(circuit_ids),
            "quali_best_time": round(random.uniform(88.0, 95.0), 3),  # Realistic F1 times
            "weather_condition": random.choice(weather_conditions),
            "temperature": round(random.uniform(15.0, 35.0), 1),
            "humidity": round(random.uniform(30.0, 80.0), 1),
            "wind_speed": round(random.uniform(0.0, 15.0), 1),
            "rain_probability": round(random.uniform(0.0, 100.0), 1),
            "track_temperature": round(random.uniform(20.0, 45.0), 1),
            "air_pressure": round(random.uniform(1005.0, 1025.0), 2)
        }
        drivers.append(driver_data)
    
    return drivers

def make_prediction(endpoint: str, drivers: List[Dict], base_url: str = "http://localhost:8080"):
    """Make a prediction request"""
    
    request_data = {
        "drivers": drivers,
        "session_type": "qualifying" if "quali" in endpoint else "race"
    }
    
    try:
        print(f"🚀 Making request to: {base_url}/{endpoint}")
        print(f"📊 Data: {len(drivers)} drivers")
        
        response = requests.post(f"{base_url}/{endpoint}", 
                               json=request_data,
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response received")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Message: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    """Run random predictions"""
    
    print("🏎️  F1 ML PREDICTION SYSTEM - RANDOM PREDICTION GENERATOR")
    print("=" * 60)
    
    # Check if API is running
    try:
        health_response = requests.get("http://localhost:8080/health")
        if health_response.status_code != 200:
            print("❌ API is not running! Please start the server first:")
            print("   python -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8080")
            return
    except:
        print("❌ Cannot connect to API! Please start the server first:")
        print("   python -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8080")
        return
    
    print("✅ API is running!")
    print()
    
    # Generate random drivers
    num_drivers = random.randint(3, 8)
    drivers = generate_random_driver_data(num_drivers)
    
    print(f"🎲 Generated {len(drivers)} random drivers:")
    for driver in drivers:
        print(f"   Driver {driver['driver_id']}: {driver['weather_condition']} conditions, "
              f"temp {driver['temperature']}°C")
    print()
    
    # Test all endpoints
    endpoints = ["predict_quali", "predict_race", "predict_full"]
    
    for endpoint in endpoints:
        print(f"🔄 Testing: {endpoint}")
        print("-" * 40)
        
        result = make_prediction(endpoint, drivers)
        
        if result:
            if endpoint == "predict_quali":
                predictions = result
                print(f"🏁 Qualifying Predictions:")
                for pred in predictions:
                    print(f"   Driver {pred['driver_id']}: {pred['predicted_time']:.3f}s "
                          f"(Grid P{pred['grid_position_estimate']})")
                          
            elif endpoint == "predict_race":
                predictions = result
                print(f"🏆 Race Winner Predictions:")
                for pred in predictions:
                    print(f"   Driver {pred['driver_id']}: {pred['win_probability']:.1%} "
                          f"win chance (Rank {pred['ranking']})")
                          
            elif endpoint == "predict_full":
                print(f"🎯 Full Predictions:")
                if 'qualifying_predictions' in result:
                    print(f"   Qualifying: {len(result['qualifying_predictions'])} predictions")
                if 'race_winner_predictions' in result:
                    print(f"   Race: {len(result['race_winner_predictions'])} predictions")
        
        print()
    
    print("🎉 Random prediction testing complete!")

if __name__ == "__main__":
    main()