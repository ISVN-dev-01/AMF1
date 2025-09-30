# 🚀 AMF1 API Documentation

*Complete API reference for AMF1 Formula 1 Prediction System*

---

## 📋 **Overview**

The AMF1 API provides real-time Formula 1 predictions through a high-performance FastAPI server. The API offers two main prediction endpoints:

- **Stage-1**: Qualifying time predictions (regression)
- **Stage-2**: Race winner probability predictions (classification)

### **Base URL**
- **Local Development**: `http://localhost:8000`
- **Production**: `https://api.amf1.com` (when deployed)

### **API Characteristics**
- **Response Time**: <100ms p95
- **Throughput**: 1000+ requests/second
- **Uptime SLA**: 99.9%
- **Rate Limiting**: 100 requests/minute per user
- **Authentication**: API key based (production)

---

## 🔧 **Quick Start**

### **Start the API Server**
```bash
# Development mode
python src/serve/app.py

# Production mode with Uvicorn
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --workers 4

# With Docker
docker run -p 8000:8000 amf1:latest
```

### **API Health Check**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-30T17:35:25.000Z",
  "version": "1.0.0",
  "models": {
    "stage1": "loaded",
    "stage2": "loaded"
  }
}
```

---

## 📖 **API Endpoints**

### **1. Health & Status**

#### **GET /health**
Check API server health and model status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-30T17:35:25.000Z",
  "version": "1.0.0",
  "models": {
    "stage1": "loaded",
    "stage2": "loaded"
  },
  "uptime_seconds": 3600,
  "requests_served": 1523
}
```

#### **GET /info**
Get detailed API and model information.

**Response:**
```json
{
  "api_version": "1.0.0",
  "model_versions": {
    "stage1": "2.1.0",
    "stage2": "2.1.0"
  },
  "last_retrained": "2025-09-15T10:30:00Z",
  "supported_features": [
    "weather_temp", "track_temp", "humidity", "wind_speed",
    "driver_momentum", "team_performance", "circuit_difficulty"
  ],
  "prediction_latency_p50": 45,
  "prediction_latency_p95": 89
}
```

---

### **2. Qualifying Predictions (Stage-1)**

#### **POST /predict/qualifying**
Predict qualifying lap times for a driver at a specific circuit.

**Request Body:**
```json
{
  "driver_id": "hamilton",
  "circuit_id": "silverstone",
  "weather_temp": 22.5,
  "track_temp": 35.0,
  "humidity": 0.6,
  "wind_speed": 8.2,
  "tire_compound": "medium",
  "fuel_load": 50.0,
  "downforce_level": "medium"
}
```

**Required Fields:**
- `driver_id` (string): Driver identifier (e.g., "hamilton", "verstappen")
- `circuit_id` (string): Circuit identifier (e.g., "silverstone", "monaco")

**Optional Fields:**
- `weather_temp` (float): Air temperature in Celsius (default: 25.0)
- `track_temp` (float): Track surface temperature in Celsius (default: 35.0)
- `humidity` (float): Relative humidity 0.0-1.0 (default: 0.5)
- `wind_speed` (float): Wind speed in m/s (default: 5.0)
- `tire_compound` (string): "soft", "medium", "hard" (default: "medium")
- `fuel_load` (float): Fuel load in kg (default: 50.0)
- `downforce_level` (string): "low", "medium", "high" (default: "medium")

**Response:**
```json
{
  "prediction": {
    "qualifying_time": 89.542,
    "confidence_interval": {
      "lower": 89.201,
      "upper": 89.883
    },
    "confidence_level": 0.95
  },
  "model_info": {
    "model_version": "2.1.0",
    "features_used": 42,
    "prediction_uncertainty": 0.174
  },
  "request_info": {
    "request_id": "req_abc123",
    "timestamp": "2025-09-30T17:35:25.000Z",
    "processing_time_ms": 45
  }
}
```

#### **POST /predict/qualifying/batch**
Batch prediction for multiple driver-circuit combinations.

**Request Body:**
```json
{
  "predictions": [
    {
      "driver_id": "hamilton",
      "circuit_id": "silverstone",
      "weather_temp": 22.5
    },
    {
      "driver_id": "verstappen",
      "circuit_id": "silverstone",
      "weather_temp": 22.5
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "driver_id": "hamilton",
      "qualifying_time": 89.542,
      "confidence_interval": [89.201, 89.883]
    },
    {
      "driver_id": "verstappen",
      "qualifying_time": 89.234,
      "confidence_interval": [88.891, 89.577]
    }
  ],
  "summary": {
    "fastest_predicted": "verstappen",
    "slowest_predicted": "hamilton",
    "average_time": 89.388
  }
}
```

---

### **3. Race Winner Predictions (Stage-2)**

#### **POST /predict/race-winner**
Predict race winner probabilities for a field of drivers.

**Request Body:**
```json
{
  "drivers": ["hamilton", "verstappen", "leclerc", "russell", "norris"],
  "circuit_id": "silverstone",
  "weather_conditions": {
    "temperature": 22.5,
    "humidity": 0.6,
    "rain_probability": 0.15
  },
  "qualifying_results": {
    "hamilton": 2,
    "verstappen": 1,
    "leclerc": 3,
    "russell": 4,
    "norris": 5
  }
}
```

**Required Fields:**
- `drivers` (array): List of driver identifiers
- `circuit_id` (string): Circuit identifier

**Optional Fields:**
- `weather_conditions` (object): Weather parameters
- `qualifying_results` (object): Starting grid positions
- `tire_strategies` (object): Planned tire strategies per driver

**Response:**
```json
{
  "predictions": {
    "hamilton": {
      "win_probability": 0.234,
      "podium_probability": 0.612,
      "points_probability": 0.887
    },
    "verstappen": {
      "win_probability": 0.456,
      "podium_probability": 0.789,
      "points_probability": 0.934
    },
    "leclerc": {
      "win_probability": 0.187,
      "podium_probability": 0.523,
      "points_probability": 0.823
    }
  },
  "race_analysis": {
    "most_likely_winner": "verstappen",
    "upset_probability": 0.088,
    "safety_car_impact": 0.156,
    "weather_impact": 0.089
  },
  "model_info": {
    "calibration_score": 0.142,
    "prediction_confidence": "high"
  }
}
```

#### **POST /predict/race-winner/simplified**
Simplified race winner prediction with minimal inputs.

**Request Body:**
```json
{
  "circuit_id": "silverstone",
  "top_drivers": 5
}
```

**Response:**
```json
{
  "predictions": [
    {
      "driver": "verstappen",
      "probability": 0.342,
      "odds": "2.9:1"
    },
    {
      "driver": "hamilton",
      "probability": 0.267,
      "odds": "3.7:1"
    },
    {
      "driver": "leclerc",
      "probability": 0.198,
      "odds": "5.0:1"
    }
  ]
}
```

---

### **4. Advanced Analytics**

#### **GET /analytics/driver/{driver_id}**
Get comprehensive driver analytics and performance trends.

**Response:**
```json
{
  "driver_id": "hamilton",
  "current_season": {
    "championship_position": 3,
    "points": 234,
    "wins": 2,
    "podiums": 8,
    "qualifying_average": 3.2
  },
  "performance_trends": {
    "recent_form": "improving",
    "momentum_score": 0.78,
    "consistency_rating": 0.85
  },
  "circuit_performance": {
    "silverstone": {
      "average_qualifying": 2.1,
      "win_rate": 0.45,
      "last_5_finishes": [1, 2, 1, 3, 2]
    }
  }
}
```

#### **GET /analytics/circuit/{circuit_id}**
Get circuit-specific analytics and historical data.

**Response:**
```json
{
  "circuit_id": "silverstone",
  "characteristics": {
    "track_length": 5.891,
    "corners": 18,
    "drs_zones": 2,
    "overtaking_difficulty": 0.34
  },
  "weather_patterns": {
    "average_temperature": 19.2,
    "rain_probability": 0.23,
    "wind_patterns": "variable"
  },
  "performance_insights": {
    "pole_to_win_rate": 0.67,
    "average_pit_stops": 1.8,
    "safety_car_probability": 0.31
  }
}
```

---

### **5. Model Management**

#### **GET /models/status**
Get current model status and performance metrics.

**Response:**
```json
{
  "models": {
    "stage1": {
      "version": "2.1.0",
      "status": "active",
      "performance": {
        "mae": 0.314,
        "rmse": 0.431,
        "r2_score": 0.862
      },
      "last_updated": "2025-09-15T10:30:00Z"
    },
    "stage2": {
      "version": "2.1.0",
      "status": "active",
      "performance": {
        "brier_score": 0.142,
        "log_loss": 0.987,
        "accuracy": 0.789
      },
      "last_updated": "2025-09-15T10:30:00Z"
    }
  },
  "next_retrain_scheduled": "2025-10-15T02:00:00Z"
}
```

#### **POST /models/retrain**
Trigger model retraining (admin only).

**Request Body:**
```json
{
  "model_type": "both",
  "priority": "normal",
  "notification_email": "admin@amf1.com"
}
```

---

## 🔧 **Authentication**

### **Development Mode**
No authentication required for local development.

### **Production Mode**
API key authentication required.

**Header:**
```
Authorization: Bearer YOUR_API_KEY
```

**Example:**
```bash
curl -H "Authorization: Bearer sk-amf1-abc123def456" \
     -H "Content-Type: application/json" \
     -d '{"driver_id": "hamilton", "circuit_id": "silverstone"}' \
     https://api.amf1.com/predict/qualifying
```

---

## 📊 **Response Formats**

### **Success Response**
```json
{
  "prediction": { ... },
  "model_info": { ... },
  "request_info": {
    "request_id": "req_abc123",
    "timestamp": "2025-09-30T17:35:25.000Z",
    "processing_time_ms": 45
  }
}
```

### **Error Response**
```json
{
  "error": {
    "code": "INVALID_DRIVER",
    "message": "Driver 'unknown_driver' not found in our database",
    "details": {
      "valid_drivers": ["hamilton", "verstappen", "leclerc", ...]
    }
  },
  "request_info": {
    "request_id": "req_def456",
    "timestamp": "2025-09-30T17:35:25.000Z"
  }
}
```

---

## ⚠️ **Error Codes**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_DRIVER` | 400 | Unknown driver identifier |
| `INVALID_CIRCUIT` | 400 | Unknown circuit identifier |
| `MISSING_REQUIRED_FIELD` | 400 | Required field not provided |
| `INVALID_FIELD_VALUE` | 400 | Field value out of valid range |
| `MODEL_NOT_LOADED` | 503 | Model not available |
| `PREDICTION_FAILED` | 500 | Internal prediction error |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |

---

## 🚀 **Usage Examples**

### **Python Client**
```python
import requests
import json

# Qualifying prediction
def predict_qualifying(driver, circuit, weather_temp=25.0):
    url = "http://localhost:8000/predict/qualifying"
    payload = {
        "driver_id": driver,
        "circuit_id": circuit,
        "weather_temp": weather_temp
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
result = predict_qualifying("hamilton", "silverstone", 22.5)
print(f"Predicted qualifying time: {result['prediction']['qualifying_time']:.3f}s")
```

### **JavaScript Client**
```javascript
async function predictRaceWinner(drivers, circuit) {
    const url = 'http://localhost:8000/predict/race-winner';
    const payload = {
        drivers: drivers,
        circuit_id: circuit
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });
    
    return await response.json();
}

// Example usage
predictRaceWinner(['hamilton', 'verstappen', 'leclerc'], 'silverstone')
    .then(result => {
        console.log('Race predictions:', result.predictions);
    });
```

### **cURL Examples**
```bash
# Qualifying prediction
curl -X POST "http://localhost:8000/predict/qualifying" \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": "hamilton",
    "circuit_id": "silverstone",
    "weather_temp": 22.5,
    "track_temp": 35.0
  }'

# Race winner prediction
curl -X POST "http://localhost:8000/predict/race-winner" \
  -H "Content-Type: application/json" \
  -d '{
    "drivers": ["hamilton", "verstappen", "leclerc"],
    "circuit_id": "silverstone"
  }'

# Batch qualifying predictions
curl -X POST "http://localhost:8000/predict/qualifying/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"driver_id": "hamilton", "circuit_id": "silverstone"},
      {"driver_id": "verstappen", "circuit_id": "silverstone"}
    ]
  }'
```

---

## 📈 **Performance & Monitoring**

### **Metrics Endpoint**
```bash
curl http://localhost:8000/metrics
```

**Prometheus Metrics:**
```
# HELP amf1_predictions_total Total number of predictions made
# TYPE amf1_predictions_total counter
amf1_predictions_total{model="stage1"} 1234
amf1_predictions_total{model="stage2"} 856

# HELP amf1_prediction_latency_seconds Prediction latency in seconds
# TYPE amf1_prediction_latency_seconds histogram
amf1_prediction_latency_seconds_bucket{le="0.05"} 789
amf1_prediction_latency_seconds_bucket{le="0.1"} 1156
amf1_prediction_latency_seconds_bucket{le="0.2"} 1234
```

### **Rate Limiting**
- **Free Tier**: 100 requests/minute
- **Pro Tier**: 1000 requests/minute
- **Enterprise**: Unlimited

### **Caching**
- **Prediction Cache**: 30 minutes for identical requests
- **Model Cache**: Models cached in memory
- **Feature Cache**: Computed features cached for 15 minutes

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Server Configuration
AMFL_HOST=0.0.0.0
AMF1_PORT=8000
AMF1_WORKERS=4

# Model Configuration
AMF1_MODEL_PATH=/opt/amf1/models
AMF1_STAGE1_MODEL=stage1_ensemble.pkl
AMF1_STAGE2_MODEL=stage2_ensemble.pkl

# Performance Configuration
AMF1_PREDICTION_TIMEOUT=5
AMF1_CACHE_TTL=1800
AMF1_MAX_BATCH_SIZE=100

# Monitoring
AMF1_ENABLE_METRICS=true
AMF1_LOG_LEVEL=INFO
```

### **Production Deployment**
```yaml
# docker-compose.yml
version: '3.8'
services:
  amf1-api:
    image: amf1:latest
    ports:
      - "8000:8000"
    environment:
      - AMF1_WORKERS=8
      - AMF1_ENABLE_METRICS=true
    volumes:
      - ./models:/opt/amf1/models
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

---

## 🧪 **Testing**

### **API Testing**
```bash
# Run API tests
pytest tests/test_api.py -v

# Load testing
locust -f tests/load_test.py --host=http://localhost:8000
```

### **Health Check Script**
```python
import requests
import time

def health_check():
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API healthy - Status: {data['status']}")
            return True
        else:
            print(f"❌ API unhealthy - Status: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ API connection failed: {e}")
        return False

if __name__ == "__main__":
    health_check()
```

---

## 📚 **Related Documentation**

- [Model Card](MODEL_CARD.md) - Model specifications and limitations
- [Retraining Runbook](../runbooks/retrain.md) - Operational procedures
- [Monitoring Guide](MONITORING.md) - MLOps monitoring setup
- [README](../README.md) - Complete system overview

---

*Last Updated: September 30, 2025*
*API Version: 1.0.0*