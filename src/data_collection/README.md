# Data Collection Module

This module handles Phase 1 data collection for the F1 AMF project.

## Overview

The data collection system gathers data from multiple sources:
- **Ergast API**: Historical F1 data (races, qualifying, results)
- **FastF1**: Practice session telemetry and lap data
- **OpenWeatherMap**: Weather conditions during races

## Scripts

### Individual Collection Scripts

1. **`ergast_collect.py`** - Collects historical F1 data from Ergast API
   - Outputs: `races.parquet`, `qualifying.parquet`, `results.parquet`
   - Covers seasons 2014-2024

2. **`fastf1_collect.py`** - Collects practice and qualifying telemetry
   - Outputs: Session-specific parquet files (e.g., `fastf1_q_monaco_2024.parquet`)
   - Uses caching for faster subsequent loads

3. **`weather_collect.py`** - Collects historical weather data
   - Requires OpenWeatherMap API key in `.env` file
   - Outputs: `data/raw/weather/weather_YYYY.parquet`

4. **`create_master_dataset.py`** - Merges all data sources
   - Output: `master_dataset.parquet`
   - Creates canonical keys for joining datasets

### Main Orchestrator

**`run_phase1.py`** - Runs all collection scripts in sequence
```bash
python src/data_collection/run_phase1.py
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenWeatherMap API key in `.env`:
```
OPENWEATHER_API_KEY=your_actual_key_here
```

3. Run data collection:
```bash
python src/data_collection/run_phase1.py
```

## Output Structure

```
data/raw/
├── races.parquet              # Race calendar and circuit info
├── qualifying.parquet         # Qualifying results
├── results.parquet           # Race results
├── fastf1_*.parquet          # Session telemetry data
├── weather/
│   └── weather_2024.parquet  # Weather conditions
└── master_dataset.parquet    # Combined dataset
```

## Data Schema

### Canonical Keys
- `season`: Year (int)
- `race_id`: "{season}_{round}" (str)
- `driver_id`: Driver identifier (str)
- `team_id`: Team/constructor identifier (str)
- `circuit_id`: Circuit identifier (str)
- `date_utc`: Race date in UTC (datetime)

## Notes

- FastF1 data collection can be slow due to telemetry download size
- OpenWeatherMap free tier has rate limits (1000 calls/day)
- Weather collection includes 1-second delays between API calls
- Cache directory speeds up repeated FastF1 loads