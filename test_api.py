#!/usr/bin/env python3
"""
PHASE 10.2: API Testing Script
Test the FastAPI F1 prediction endpoints
"""

import requests
import json
import sys
from typing import Dict, Any

def test_f1_api(base_url: str = "http://localhost:8080"):
    """Test all F1 API endpoints"""
    
    print("🔄 Testing F1 Prediction API")
    print(f"   Base URL: {base_url}")
    
    # Test 1: Health check
    print("\n📊 Test 1: Health Check")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 2: Detailed health check
    print("\n🔍 Test 2: Detailed Health Check")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        health_data = response.json()
        print(f"   Models loaded: {health_data.get('models_loaded', False)}")
        print(f"   Model version: {health_data.get('model_version', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 3: Model info
    print("\n📋 Test 3: Model Information")
    try:
        response = requests.get(f"{base_url}/model_info")
        print(f"   Status: {response.status_code}")
        model_info = response.json()
        print(f"   Model type: {model_info.get('model_type', 'unknown')}")
        print(f"   Feature count: {model_info.get('feature_count', 0)}")
        print(f"   Training date: {model_info.get('training_date', 'unknown')}")
    except Exception as e:
        print(f"   ⚠️  Model info failed: {e}")
    
    # Test data for predictions
    test_drivers = [
        {
            "driver_id": 1,
            "team_id": 1,
            "circuit_id": 1,
            "temperature": 25.0,
            "humidity": 60.0,
            "wind_speed": 10.0,
            "rain_probability": 0.1,
            "track_temperature": 35.0,
            "additional_features": {
                "practice_best_time": 78.5,
                "driver_experience": 5.2
            }
        },
        {
            "driver_id": 2,
            "team_id": 2,
            "circuit_id": 1,
            "temperature": 25.0,
            "humidity": 60.0,
            "wind_speed": 10.0,
            "rain_probability": 0.1,
            "track_temperature": 35.0,
            "additional_features": {
                "practice_best_time": 79.1,
                "driver_experience": 3.8
            }
        },
        {
            "driver_id": 3,
            "team_id": 3,
            "circuit_id": 1,
            "temperature": 25.0,
            "humidity": 60.0,
            "wind_speed": 10.0,
            "rain_probability": 0.1,
            "track_temperature": 35.0,
            "additional_features": {
                "practice_best_time": 78.9,
                "driver_experience": 7.1
            }
        }
    ]
    
    prediction_request = {
        "drivers": test_drivers,
        "session_type": "qualifying",
        "race_id": 1
    }
    
    # Test 4: Qualifying predictions
    print("\n🏁 Test 4: Qualifying Predictions")
    try:
        response = requests.post(
            f"{base_url}/predict_quali",
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            quali_preds = response.json()
            print(f"   Predictions count: {len(quali_preds)}")
            for pred in quali_preds[:3]:  # Show first 3
                print(f"   Driver {pred['driver_id']}: {pred['predicted_time']:.3f}s (P{pred['grid_position_estimate']})")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Qualifying prediction failed: {e}")
    
    # Test 5: Race winner predictions
    print("\n🏆 Test 5: Race Winner Predictions")
    try:
        response = requests.post(
            f"{base_url}/predict_race",
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            race_preds = response.json()
            print(f"   Predictions count: {len(race_preds)}")
            for pred in race_preds[:3]:  # Show first 3
                print(f"   Driver {pred['driver_id']}: {pred['win_probability']:.3f} prob (#{pred['ranking']})")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Race winner prediction failed: {e}")
    
    # Test 6: Full prediction pipeline
    print("\n🚀 Test 6: Full Prediction Pipeline")
    try:
        response = requests.post(
            f"{base_url}/predict_full",
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            full_preds = response.json()
            print(f"   Qualifying predictions: {len(full_preds['qualifying_predictions'])}")
            print(f"   Race winner predictions: {len(full_preds['race_winner_predictions'])}")
            print(f"   Model version: {full_preds['metadata']['model_version']}")
            print(f"   Feature count: {full_preds['metadata']['feature_count']}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Full prediction failed: {e}")
    
    print(f"\n✅ API Testing Complete!")
    return True

if __name__ == "__main__":
    import time
    
    # Check if server is running
    base_url = "http://localhost:8080"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print("🏁 F1 API Testing Script")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    try:
        test_f1_api(base_url)
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        sys.exit(1)