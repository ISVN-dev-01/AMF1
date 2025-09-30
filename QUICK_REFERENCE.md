# 🏎️ **F1 ML PREDICTION SYSTEM - QUICK REFERENCE** 

## 🚀 **One-Command Quick Start**

```bash
python quick_start.py
```
This will:
- ✅ Check dependencies
- ✅ Start the API server  
- ✅ Test predictions
- ✅ Give you next steps

## 🎲 **Make Random Predictions**

### **Method 1: Easy Random Generator**
```bash
python make_prediction.py
```

### **Method 2: Built-in Test Script**
```bash
python test_api.py
```

### **Method 3: Manual API Start**
```bash
# Terminal 1: Start server
python -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8080

# Terminal 2: Make predictions
curl -X POST "http://localhost:8080/predict_quali" \
  -H "Content-Type: application/json" \
  -d '{"drivers": [{"driver_id": 44, "weather_condition": "dry"}]}'
```

## 🌐 **Interactive Web Interface**

1. Start server: `python quick_start.py`
2. Open browser: **http://localhost:8080/docs**
3. Click any endpoint → "Try it out" → "Execute"

## 📖 **Full Documentation**

- **Complete Guide**: `F1_PREDICTION_USAGE_GUIDE.md`
- **API Endpoints**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

## 🎯 **Sample Prediction Data**

```json
{
  "drivers": [
    {
      "driver_id": 44,
      "weather_condition": "dry",
      "temperature": 25.0
    },
    {
      "driver_id": 1, 
      "weather_condition": "dry",
      "temperature": 25.0
    }
  ]
}
```

## 🏆 **Driver IDs (2024 F1 Season)**

- `1`: Max Verstappen  
- `44`: Lewis Hamilton
- `16`: Charles Leclerc
- `4`: Lando Norris
- `11`: Sergio Pérez
- `55`: Carlos Sainz
- `63`: George Russell
- And more...

## 🔧 **Troubleshooting**

**Server won't start?**
```bash
pip install -r requirements.txt
python quick_start.py
```

**Models missing?**
```bash
# Check if models exist
ls models/

# If missing, you may need to train models first
```

**Connection issues?**
- Try `http://127.0.0.1:8080` instead of `localhost`
- Check if port 8080 is free: `lsof -i :8080`

---

🎉 **Ready to predict F1 race outcomes!** Start with `python quick_start.py` 🏁