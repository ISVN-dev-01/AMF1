# Phase 3 - Label Generation

## Overview

Phase 3 creates the target variables for machine learning models by extracting key performance indicators from qualifying and race data. These labels serve as the ground truth for prediction tasks in Formula 1 analytics.

## Goals

- ✅ Create binary classification targets (`is_pole`, `is_race_winner`)
- ✅ Generate regression targets (`quali_best_time`, `race_position`)  
- ✅ Ensure data consistency and validation
- ✅ Provide comprehensive label statistics

## Target Variables

### Qualifying Labels
- **`is_pole`**: Binary flag for pole position (fastest in qualifying)
- **`quali_best_time`**: Best qualifying lap time in seconds
- **`quali_rank`**: Qualifying position (1 = fastest)

### Race Labels  
- **`race_position`**: Final finishing position in the race
- **`is_race_winner`**: Binary flag for race winner (1st place finish)

## Scripts

### Main Label Generation

**`src/features/create_labels.py`** - Core label generation pipeline

Key functions:
- `create_qualifying_labels()`: Extract pole positions and qualifying times
- `create_race_labels()`: Extract race results and winners
- `merge_labels()`: Combine qualifying and race data
- `validate_labels()`: Comprehensive data validation
- `create_summary_stats()`: Generate performance statistics

### Orchestrator

**`src/features/run_phase3.py`** - Complete Phase 3 pipeline

```bash
python src/features/run_phase3.py
```

### Validation & Testing

**`tests/test_labels.py`** - Label validation test suite

Validation checks:
- ✅ Required columns present
- ✅ No null critical fields
- ✅ Binary flags contain only 0/1 values
- ✅ One pole position per race
- ✅ One winner per race  
- ✅ Reasonable qualifying times (60-180 seconds)
- ✅ Valid race positions (1-30)
- ✅ Unique race-driver combinations
- ✅ Consistent winners and positions

## Algorithm Details

### Qualifying Label Creation

```python
# Filter qualifying sessions
quali = df[df.session_type == 'Q'].copy()

# Get best lap time per driver per race
best = quali.groupby(['race_id', 'driver_id'])['lap_time_sec'].min().reset_index()

# Calculate qualifying rank (1 = fastest)
best['quali_rank'] = best.groupby('race_id')['lap_time_sec'].rank(method='min')

# Create pole position flag
best['is_pole'] = (best['quali_rank'] == 1).astype(int)
```

### Race Label Creation

```python
# Filter race sessions  
race = df[df.session_type == 'R']

# Get race positions
race_pos = race[['race_id', 'driver_id', 'race_position']].drop_duplicates()

# Create race winner flag
race_pos['is_race_winner'] = (race_pos['race_position'] == 1).astype(int)
```

## Data Handling

### Flexible Column Detection

The pipeline automatically detects appropriate columns across different data schemas:

- **Session types**: `session_type`, `sessionType`, `Session`
- **Driver IDs**: `driver_id`, `Driver_clean`, `Driver`, `driverId`
- **Lap times**: `lap_time_sec`, `LapTime_seconds`, `Q3_seconds`, etc.
- **Positions**: `race_position`, `position`, `Position`, `positionOrder`

### Missing Data Handling

- Gracefully handles missing qualifying or race data
- Uses best available time source (Q3 > Q2 > Q1 for qualifying)
- Fills missing binary flags with 0
- Provides warnings for data quality issues

## Usage

### Run Complete Phase 3

```bash
python src/features/run_phase3.py
```

### Run Individual Components

```bash
# Label generation only
python src/features/create_labels.py

# Validation tests only  
python tests/test_labels.py --test

# Label validation report only
python tests/test_labels.py
```

## Input/Output

### Input
- `data/processed/master_dataset.parquet` (from Phase 2)

### Output
- `data/processed/labels.parquet` (target variables)

## Label Statistics

Expected label distributions for F1 data:

- **Pole positions**: ~5% of driver-race combinations (1 per race)
- **Race wins**: ~5% of driver-race combinations (1 per race)
- **Qualifying times**: 60-180 seconds (track and conditions dependent)
- **Race positions**: 1-22 (typical F1 field size)

## Validation Results

The pipeline performs comprehensive validation:

### Data Integrity
- ✅ No null race IDs or driver IDs
- ✅ Unique race-driver combinations
- ✅ Binary flags contain only 0/1 values

### Logical Consistency  
- ✅ Exactly one pole position per race
- ✅ Exactly one race winner per race
- ✅ Race winners have position 1
- ✅ Position 1 finishers marked as winners

### Value Ranges
- ✅ Qualifying times in reasonable range
- ✅ Race positions are positive integers
- ✅ No unreasonably high positions

## Performance Statistics

Example output for a complete dataset:

```
Label Summary Statistics:
Total driver-race combinations: 8,456
Unique races: 264
Unique drivers: 45

Top pole position holders:
  verstappen: 28
  hamilton: 22  
  leclerc: 18

Top race winners:
  verstappen: 31
  hamilton: 19
  leclerc: 12

Average qualifying best time: 78.456 seconds
```

## Error Handling

The pipeline handles common data issues:

- Multiple session type formats
- Missing qualifying or race data
- Inconsistent column naming
- Invalid time formats
- Missing position data

## Next Steps

After Phase 3 completion:
1. **Phase 4**: Feature Engineering - Create predictive features
2. **Phase 5**: Model Training - Train ML models using these labels
3. **Phase 6**: Model Evaluation - Assess prediction performance

## Troubleshooting

### Common Issues

1. **No qualifying data found**: Check session type column values
2. **Multiple poles per race**: Data quality issue - investigate source data
3. **Missing position data**: Verify race results are properly loaded
4. **Time conversion errors**: Check time column formats and values

### Performance Tips

- Labels are lightweight compared to features
- Consider caching for repeated analysis
- Use validation report to identify data quality issues early