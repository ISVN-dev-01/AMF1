# 🏎️ **F1 ML PREDICTION SYSTEM - COMPLETE USAGE SUMMARY** 

## 🚀 **FASTEST WAY TO GET STARTED**

```bash
python quick_start.py
```

## 🎯 **4 WAYS TO MAKE PREDICTIONS**

### **1. 🎬 Complete Race Weekend Experience**
```bash
python race_weekend_simulator.py
```
- Simulates qualifying + race
- Realistic F1 driver data  
- Weather scenarios
- Complete race analysis

### **2. 🎲 Random Prediction Generator**
```bash
python make_prediction.py
```
- Generates random driver combinations
- Tests all API endpoints
- Quick and easy predictions

### **3. 🔧 Manual API Control**
```bash
# Terminal 1: Start server
python -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8080

# Terminal 2: Use API
curl -X POST "http://localhost:8080/predict_full" \
  -H "Content-Type: application/json" \
  -d '{"drivers": [{"driver_id": 44, "weather_condition": "dry"}]}'
```

### **4. 🌐 Interactive Web Interface**
1. Run: `python quick_start.py`
2. Open: http://localhost:8080/docs
3. Click "Try it out" on any endpoint
4. Make predictions with the web UI

## 📊 **AVAILABLE PREDICTIONS**

- **🏁 Qualifying Times**: `/predict_quali` 
- **🏆 Race Winners**: `/predict_race`
- **🎯 Full Analysis**: `/predict_full` (both quali + race)

## 🏎️ **REAL F1 DRIVERS (2024)**

- `1`: Max Verstappen
- `44`: Lewis Hamilton  
- `16`: Charles Leclerc
- `4`: Lando Norris
- `11`: Sergio Pérez
- And 15+ more!

## 📖 **DOCUMENTATION**

- **Quick Reference**: `QUICK_REFERENCE.md`
- **Complete Guide**: `F1_PREDICTION_USAGE_GUIDE.md`
- **API Docs**: http://localhost:8080/docs (when server running)

## 🎉 **READY TO RACE!**

Start with: `python quick_start.py` 🏁

**Your F1 ML system can predict:**
- ✅ Qualifying lap times
- ✅ Grid positions  
- ✅ Race winner probabilities
- ✅ Weather impact analysis
- ✅ Complete race weekend scenarios

**Make your first prediction in under 30 seconds!** 🚀