# 🏎️ **F1 ML PREDICTION SYSTEM - COMPLETE USAGE GUIDE** 🏆

## 🎯 **Quick Start - Make Your First Prediction**

### **Step 1: Start the API Server**

```bash
# Navigate to your project directory
cd /Users/vishale/Documents/AMF!-MLmodel/AMF1

# Activate your environment (if using virtual environment)
source venv/bin/activate  # or your venv path

# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8080 --reload
```

The server will start at: **http://localhost:8080**

### **Step 2: Verify the API is Running**

Open your browser and go to:
- **API Docs**: http://localhost:8080/docs (Interactive Swagger UI)
- **Health Check**: http://localhost:8080/health
- **Root**: http://localhost:8080/

---

## 🎲 **Making Random Predictions - Multiple Methods**

### **Method 1: Quick Start (Recommended for Beginners)**

```bash
# One command to start everything and test
python quick_start.py
```

This will:
- ✅ Check dependencies
- ✅ Start the API server
- ✅ Test predictions automatically  
- ✅ Give you next steps

### **Method 2: Random Prediction Generator**

```bash
# Generate random F1 predictions with realistic data
python make_prediction.py
```

### **Method 3: Race Weekend Simulator**

```bash  
# Simulate a complete F1 race weekend (qualifying + race)
python race_weekend_simulator.py
```

### **Method 4: Built-in Test Script**

```bash
# Run the comprehensive API test suite
python test_api.py
```

### **Method 2: Using curl Commands**

#### **A. Test Health Check**
```bash
curl -X GET "http://localhost:8080/health" -H "accept: application/json"
```

#### **B. Predict Qualifying Times**
```bash
curl -X POST "http://localhost:8080/predict_quali" \
  -H "Content-Type: application/json" \
  -d '{
    "drivers": [
      {
        "driver_id": 1,
        "team_id": 1,
        "circuit_id": 1,
        "quali_best_time": 90.5,
        "weather_condition": "dry",
        "temperature": 25.0,
        "humidity": 50.0,
        "wind_speed": 5.0,
        "rain_probability": 0.0,
        "track_temperature": 30.0
      },
      {
        "driver_id": 44,
        "team_id": 2,
        "circuit_id": 1,
        "quali_best_time": 89.8,
        "weather_condition": "dry",
        "temperature": 25.0,
        "humidity": 50.0,
        "wind_speed": 5.0,
        "rain_probability": 0.0,
        "track_temperature": 30.0
      }
    ],
    "session_type": "qualifying"
  }'
```

#### **C. Predict Race Winners**
```bash
curl -X POST "http://localhost:8080/predict_race" \
  -H "Content-Type: application/json" \
  -d '{
    "drivers": [
      {
        "driver_id": 1,
        "team_id": 1,
        "circuit_id": 1,
        "quali_best_time": 90.5,
        "weather_condition": "dry",
        "temperature": 25.0
      },
      {
        "driver_id": 44,
        "team_id": 2,
        "circuit_id": 1,
        "quali_best_time": 89.8,
        "weather_condition": "dry",
        "temperature": 25.0
      }
    ],
    "session_type": "race"
  }'
```

#### **D. Full Prediction (Both Qualifying + Race)**
```bash
curl -X POST "http://localhost:8080/predict_full" \
  -H "Content-Type: application/json" \
  -d '{
    "drivers": [
      {
        "driver_id": 1,
        "team_id": 1,
        "circuit_id": 1,
        "weather_condition": "dry",
        "temperature": 25.0,
        "humidity": 50.0
      },
      {
        "driver_id": 44,
        "team_id": 2,
        "circuit_id": 1,
        "weather_condition": "dry",
        "temperature": 24.5,
        "humidity": 55.0
      }
    ]
  }'
```

### **Method 3: Using Python Requests**

Create a file called `make_prediction.py`:

```python
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
```

**Run the random prediction generator:**
```bash
python make_prediction.py
```

---

## 🌐 **Interactive Web Interface**

### **Swagger UI (Recommended)**
1. Start the API server
2. Open: **http://localhost:8080/docs**
3. Click on any endpoint (e.g., `/predict_quali`)
4. Click **"Try it out"**
5. Modify the JSON data or use the defaults
6. Click **"Execute"**
7. See the results instantly!

### **ReDoc Documentation**
- Open: **http://localhost:8080/redoc**
- Beautiful, detailed API documentation

---

## 📊 **Understanding the Responses**

### **Qualifying Predictions Response**
```json
[
  {
    "driver_id": 44,
    "predicted_time": 89.234,
    "probability_score": 0.87,
    "grid_position_estimate": 1
  },
  {
    "driver_id": 1,
    "predicted_time": 89.456,
    "probability_score": 0.82,
    "grid_position_estimate": 2
  }
]
```

### **Race Winner Predictions Response**
```json
[
  {
    "driver_id": 44,
    "win_probability": 0.34,
    "confidence_score": 0.78,
    "ranking": 1
  },
  {
    "driver_id": 1,
    "win_probability": 0.28,
    "confidence_score": 0.71,
    "ranking": 2
  }
]
```

### **Full Prediction Response**
```json
{
  "qualifying_predictions": [...],
  "race_winner_predictions": [...],
  "metadata": {
    "prediction_time": "2025-09-30T...",
    "model_version": "1.0.2",
    "feature_count": 45,
    "confidence_level": "high"
  }
}
```

---

## 🎛️ **Driver Feature Parameters**

### **Required Parameters**
- `driver_id`: Unique driver identifier (integer)

### **Optional Parameters**
- `team_id`: Team identifier (1-10)
- `circuit_id`: Circuit identifier (1-10)  
- `quali_best_time`: Best qualifying time in seconds (e.g., 89.5)
- `weather_condition`: "dry", "wet", "mixed", "cloudy"
- `temperature`: Air temperature in Celsius (15-35)
- `humidity`: Relative humidity percentage (30-80)
- `wind_speed`: Wind speed in km/h (0-15)
- `rain_probability`: Rain chance percentage (0-100)
- `track_temperature`: Track temperature in Celsius (20-45)
- `air_pressure`: Atmospheric pressure in hPa (1005-1025)

### **Real F1 Driver IDs (2024 Season)**
- `1`: Max Verstappen
- `11`: Sergio Pérez  
- `16`: Charles Leclerc
- `55`: Carlos Sainz
- `44`: Lewis Hamilton
- `63`: George Russell
- `4`: Lando Norris
- `81`: Oscar Piastri
- `14`: Fernando Alonso
- `18`: Lance Stroll
- And more...

---

## 🚀 **Quick Prediction Examples**

### **Example 1: Monaco GP Prediction**
```json
{
  "drivers": [
    {
      "driver_id": 16,
      "team_id": 3,
      "circuit_id": 6,
      "weather_condition": "dry",
      "temperature": 22.0,
      "humidity": 65.0
    },
    {
      "driver_id": 1,
      "team_id": 1,
      "circuit_id": 6,
      "weather_condition": "dry", 
      "temperature": 22.0,
      "humidity": 65.0
    }
  ]
}
```

### **Example 2: Wet Weather Silverstone**
```json
{
  "drivers": [
    {
      "driver_id": 44,
      "team_id": 2,
      "circuit_id": 10,
      "weather_condition": "wet",
      "temperature": 16.0,
      "humidity": 85.0,
      "rain_probability": 90.0
    }
  ]
}
```

---

## 🔧 **Troubleshooting**

### **API Won't Start?**
```bash
# Check if models exist
ls -la models/

# Install missing dependencies
pip install -r requirements.txt

# Try different port
python -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8081
```

### **Models Not Loading?**
```bash
# Check if model files exist
ls -la models/*.pkl

# Run model training first
python src/models/train_stage1.py
python src/models/train_stage2.py
```

### **Connection Refused?**
- Make sure the server is running on the correct port
- Check firewall settings
- Try `http://127.0.0.1:8080` instead of `localhost`

---

## 🎯 **Pro Tips**

1. **Use Realistic Data**: F1 qualifying times are typically 85-95 seconds
2. **Weather Matters**: Wet conditions significantly affect predictions
3. **Multiple Drivers**: Send 5-10 drivers for realistic race scenarios
4. **Monitor Logs**: Check terminal output for detailed prediction info
5. **Batch Predictions**: Use the full prediction endpoint for complete race analysis

---

## 📈 **Monitoring & Metrics**

The API automatically tracks:
- ✅ Prediction requests
- ✅ Response times  
- ✅ Model performance
- ✅ Error rates

Access metrics at: **http://localhost:8080/metrics**

---

## 🎉 **Ready to Race!**

Your F1 ML Prediction System is now ready for action! 🏎️💨

**Start making predictions and discover which driver will dominate the next race!** 🏆

For advanced usage, model training, and deployment options, check the other documentation files in your project.