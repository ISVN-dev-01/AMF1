# Phase 2 - Data Cleaning & Validation

## Overview

Phase 2 transforms the raw master dataset into a clean, normalized dataset ready for machine learning. This phase focuses on data quality, standardization, and validation.

## Goals

- ✅ Normalize schemas and data formats
- ✅ Standardize naming conventions
- ✅ Handle missing data and outliers
- ✅ Create reliability features from DNF/DNS/DSQ data
- ✅ Validate data integrity with automated tests

## Scripts

### Main Processing Script

**`src/data_collection/clean_master.py`** - Main data cleaning pipeline

Key transformations:
- **Time Conversion**: Convert time strings (`"1:23.456"`) to seconds using `time_to_seconds()`
- **Name Standardization**: Normalize driver and team names using mapping dictionaries
- **Tyre Normalization**: Standardize tyre compounds to `{soft, medium, hard, inter, wet}`
- **Status Marking**: Identify DNF/DNS/DSQ records and preserve for reliability analysis
- **Timezone Handling**: Ensure all datetime columns are UTC timezone-aware
- **Reliability Features**: Create completion rates and average positions per driver

### Validation & Testing

**`tests/test_clean_master.py`** - Comprehensive validation suite with pytest

Validation checks:
- ✅ No null values in critical fields (`race_id`, `driver_id`, `date_utc`)
- ✅ Unique combinations of (`race_id`, `driver_id`, `session_type`)
- ✅ Reasonable driver counts per race (10-25 drivers)
- ✅ Timezone-aware datetime columns
- ✅ Valid status values (`finished`, `dnf`, `dns`, `dsq`)
- ✅ Properly normalized tyre compounds

### Orchestrator

**`src/data_collection/run_phase2.py`** - Runs complete Phase 2 pipeline

```bash
python src/data_collection/run_phase2.py
```

## Data Transformations

### 1. Time String Conversion

```python
def time_to_seconds(ts):
    """Convert '1:23.456' -> 83.456 seconds"""
    if ':' in ts:
        m, s = ts.split(':')
        return int(m) * 60 + float(s)
    return float(ts)
```

### 2. Driver Name Standardization

```python
driver_mapping = {
    'VER': 'verstappen',
    'HAM': 'hamilton', 
    'LEC': 'leclerc',
    # ... comprehensive mapping
}
```

### 3. Tyre Compound Normalization

```python
tyre_mapping = {
    'soft': 'soft', 'S': 'soft', 'C5': 'soft',
    'medium': 'medium', 'M': 'medium', 'C3': 'medium',
    'hard': 'hard', 'H': 'hard', 'C1': 'hard',
    'intermediate': 'inter', 'I': 'inter',
    'wet': 'wet', 'W': 'wet'
}
```

### 4. Status Classification

- **Finished**: Completed the race
- **DNF**: Did Not Finish (mechanical, accident, etc.)
- **DNS**: Did Not Start
- **DSQ**: Disqualified

## Usage

### Run Complete Phase 2

```bash
python src/data_collection/run_phase2.py
```

### Run Individual Components

```bash
# Data cleaning only
python src/data_collection/clean_master.py

# Validation tests only
python tests/test_clean_master.py --test

# Validation report only
python tests/test_clean_master.py
```

## Input/Output

### Input
- `data/raw/master_dataset.parquet` (from Phase 1)

### Output
- `data/processed/master_dataset.parquet` (cleaned dataset)

## Data Quality Metrics

The validation suite provides comprehensive quality metrics:

- **Completeness**: Missing data analysis per column
- **Uniqueness**: Duplicate detection and key constraint validation
- **Consistency**: Format standardization and value range validation
- **Accuracy**: Logical relationship validation (e.g., reasonable driver counts)

## Reliability Features

Phase 2 creates derived features for reliability analysis:

- **`completion_rate`**: Percentage of races finished per driver
- **`avg_finish_position`**: Average finishing position (excluding DNFs)
- **`status`**: Standardized race completion status

## Error Handling

The pipeline handles common data quality issues:

- Missing values in time columns
- Inconsistent naming conventions
- Various date/time formats
- Mixed case and special characters
- Historical data variations

## Validation Results

Expected validation outcomes:
- ✅ 0 null values in critical fields
- ✅ Unique race-driver-session combinations
- ✅ 18-22 drivers per race (typical F1 field size)
- ✅ UTC timezone for all timestamps
- ✅ Standardized categorical values

## Next Steps

After Phase 2 completion:
1. **Phase 3**: Feature Engineering - Create ML-ready features
2. **Phase 4**: Model Development - Train prediction models
3. **Phase 5**: Model Serving - Deploy via FastAPI

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all packages installed with `pip install -r requirements.txt`
2. **Missing input data**: Run Phase 1 first (`python src/data_collection/run_phase1.py`)
3. **Test failures**: Check data quality issues in validation report

### Performance Tips

- Large datasets may require chunked processing
- Consider using Dask for datasets > 1GB
- Monitor memory usage during processing