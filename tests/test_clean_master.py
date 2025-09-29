import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data_collection'))

from clean_master import time_to_seconds, normalize_tyre_compound, mark_dnf_status

class TestDataValidation:
    """Test suite for data cleaning and validation"""
    
    @classmethod
    def setup_class(cls):
        """Setup test data"""
        cls.processed_file = Path('data/processed/master_dataset.parquet')
        
        if cls.processed_file.exists():
            cls.df = pd.read_parquet(cls.processed_file)
        else:
            # Create mock data for testing if file doesn't exist
            cls.df = pd.DataFrame({
                'race_id': ['2024_1', '2024_1', '2024_2'],
                'driver_id': ['hamilton', 'verstappen', 'hamilton'],  
                'date_utc': pd.to_datetime(['2024-03-01', '2024-03-01', '2024-03-15']),
                'session_type': ['R', 'R', 'Q'],
                'position': [1, 2, 3],
                'status': ['finished', 'finished', 'finished']
            })
    
    def test_no_null_critical_fields(self):
        """Test: No null race_id, driver_id, date_utc"""
        if self.df.empty:
            pytest.skip("No data to test")
            
        # Check race_id
        null_race_ids = self.df['race_id'].isnull().sum()
        assert null_race_ids == 0, f"Found {null_race_ids} null race_id values"
        
        # Check driver_id (check multiple possible column names)
        driver_columns = ['driver_id', 'Driver_clean', 'Driver']
        driver_col = None
        for col in driver_columns:
            if col in self.df.columns:
                driver_col = col
                break
        
        assert driver_col is not None, "No driver ID column found"
        null_drivers = self.df[driver_col].isnull().sum()
        assert null_drivers == 0, f"Found {null_drivers} null driver values in {driver_col}"
        
        # Check date_utc
        null_dates = self.df['date_utc'].isnull().sum()
        assert null_dates == 0, f"Found {null_dates} null date_utc values"
    
    def test_unique_race_driver_session_pairs(self):
        """Test: Unique (race_id, driver_id, session_type) pairs"""
        if self.df.empty:
            pytest.skip("No data to test")
        
        # Find appropriate driver column
        driver_columns = ['driver_id', 'Driver_clean', 'Driver']
        driver_col = None
        for col in driver_columns:
            if col in self.df.columns:
                driver_col = col
                break
        
        if driver_col is None:
            pytest.skip("No driver column found for uniqueness test")
        
        # Check if session_type column exists
        if 'session_type' not in self.df.columns:
            # If no session_type, just check race_id + driver uniqueness
            key_columns = ['race_id', driver_col]
        else:
            key_columns = ['race_id', driver_col, 'session_type']
        
        # Count duplicates
        duplicates = self.df.duplicated(subset=key_columns).sum()
        
        if duplicates > 0:
            # Show some duplicate examples
            duplicate_rows = self.df[self.df.duplicated(subset=key_columns, keep=False)]
            print(f"\nDuplicate examples:")
            print(duplicate_rows[key_columns].head())
        
        assert duplicates == 0, f"Found {duplicates} duplicate combinations of {key_columns}"
    
    def test_reasonable_drivers_per_race(self):
        """Test: Count rows per race ≈ number of drivers (typically 18-22 in F1)"""
        if self.df.empty:
            pytest.skip("No data to test")
        
        # Find appropriate driver column
        driver_columns = ['driver_id', 'Driver_clean', 'Driver']
        driver_col = None
        for col in driver_columns:
            if col in self.df.columns:
                driver_col = col
                break
        
        if driver_col is None:
            pytest.skip("No driver column found for drivers per race test")
        
        # Count unique drivers per race
        drivers_per_race = self.df.groupby('race_id')[driver_col].nunique()
        
        # F1 typically has 18-22 drivers (accounting for reserves, changes, etc.)
        min_drivers = 10  # Minimum reasonable number
        max_drivers = 25  # Maximum reasonable number
        
        races_with_few_drivers = (drivers_per_race < min_drivers).sum()
        races_with_many_drivers = (drivers_per_race > max_drivers).sum()
        
        # Print some statistics
        print(f"\nDrivers per race statistics:")
        print(f"Mean: {drivers_per_race.mean():.1f}")
        print(f"Min: {drivers_per_race.min()}")
        print(f"Max: {drivers_per_race.max()}")
        
        # Warnings for unusual counts
        if races_with_few_drivers > 0:
            print(f"Warning: {races_with_few_drivers} races with < {min_drivers} drivers")
            print("Races with few drivers:", drivers_per_race[drivers_per_race < min_drivers].head())
        
        if races_with_many_drivers > 0:
            print(f"Warning: {races_with_many_drivers} races with > {max_drivers} drivers")
        
        # Allow some flexibility - shouldn't be too many outliers
        total_races = len(drivers_per_race)
        outlier_races = races_with_few_drivers + races_with_many_drivers
        outlier_percentage = outlier_races / total_races * 100
        
        assert outlier_percentage < 20, f"Too many races ({outlier_percentage:.1f}%) with unusual driver counts"
    
    def test_datetime_timezone_aware(self):
        """Test: datetime columns are timezone-aware (UTC)"""
        if self.df.empty:
            pytest.skip("No data to test")
        
        datetime_columns = ['date_utc', 'date']
        
        for col in datetime_columns:
            if col in self.df.columns:
                if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    # Check if timezone aware
                    has_timezone = self.df[col].dt.tz is not None
                    assert has_timezone, f"Column {col} should be timezone-aware"
                    
                    # Check if UTC
                    if has_timezone:
                        timezone_str = str(self.df[col].dt.tz)
                        assert 'UTC' in timezone_str or '+00:00' in timezone_str, f"Column {col} should be in UTC timezone"
    
    def test_status_column_values(self):
        """Test: Status column has valid values"""
        if self.df.empty or 'status' not in self.df.columns:
            pytest.skip("No status column to test")
        
        valid_statuses = {'finished', 'dnf', 'dns', 'dsq'}
        unique_statuses = set(self.df['status'].unique())
        
        invalid_statuses = unique_statuses - valid_statuses
        assert len(invalid_statuses) == 0, f"Found invalid status values: {invalid_statuses}"
    
    def test_normalized_tyre_compounds(self):
        """Test: Tyre compounds are properly normalized"""
        compound_columns = ['Compound_normalized', 'compound_normalized', 'tyre_normalized']
        
        for col in compound_columns:
            if col in self.df.columns:
                valid_compounds = {'soft', 'medium', 'hard', 'inter', 'wet', None}
                unique_compounds = set(self.df[col].dropna().unique())
                
                invalid_compounds = unique_compounds - valid_compounds
                if len(invalid_compounds) > 0:
                    print(f"Warning: Found non-standard tyre compounds in {col}: {invalid_compounds}")
                
                # Allow some flexibility for historical data
                assert len(invalid_compounds) < len(unique_compounds) * 0.1, f"Too many invalid compounds in {col}"

class TestHelperFunctions:
    """Test suite for helper functions"""
    
    def test_time_to_seconds(self):
        """Test time conversion function"""
        # Test various formats
        assert time_to_seconds('1:23.456') == pytest.approx(83.456)
        assert time_to_seconds('83.456') == pytest.approx(83.456)
        assert time_to_seconds('0:45.123') == pytest.approx(45.123)
        assert time_to_seconds('2:00.000') == pytest.approx(120.0)
        
        # Test edge cases
        assert time_to_seconds(None) is None
        assert time_to_seconds(np.nan) is None
        assert time_to_seconds('') is None
        assert time_to_seconds('invalid') is None
    
    def test_normalize_tyre_compound(self):
        """Test tyre compound normalization"""
        # Test standard mappings
        assert normalize_tyre_compound('soft') == 'soft'
        assert normalize_tyre_compound('SOFT') == 'soft'
        assert normalize_tyre_compound('S') == 'soft'
        assert normalize_tyre_compound('C5') == 'soft'
        
        assert normalize_tyre_compound('medium') == 'medium'
        assert normalize_tyre_compound('M') == 'medium'
        assert normalize_tyre_compound('C3') == 'medium'
        
        assert normalize_tyre_compound('hard') == 'hard'
        assert normalize_tyre_compound('H') == 'hard'
        assert normalize_tyre_compound('C1') == 'hard'
        
        assert normalize_tyre_compound('intermediate') == 'inter'
        assert normalize_tyre_compound('I') == 'inter'
        
        assert normalize_tyre_compound('wet') == 'wet'
        assert normalize_tyre_compound('W') == 'wet'
        
        # Test edge cases
        assert normalize_tyre_compound(None) is None
        assert normalize_tyre_compound(np.nan) is None
    
    def test_mark_dnf_status(self):
        """Test DNF status marking"""
        # Create test dataframe
        test_df = pd.DataFrame({
            'position': [1, None, 3, None],
            'status': ['Finished', 'Accident', 'Finished', '+1 Lap'],
            'laps': [58, 23, 58, 57]
        })
        
        result_df = mark_dnf_status(test_df)
        
        # Check that status column is created
        assert 'status' in result_df.columns
        
        # Check specific cases (this is a basic test - actual implementation may vary)
        assert 'dnf' in result_df['status'].values or 'finished' in result_df['status'].values

def run_validation_report():
    """Generate a comprehensive validation report"""
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    processed_file = Path('data/processed/master_dataset.parquet')
    
    if not processed_file.exists():
        print("❌ Processed dataset not found. Run clean_master.py first.")
        return False
    
    df = pd.read_parquet(processed_file)
    
    print(f"Dataset overview:")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Data quality metrics
    print(f"\nData Quality Metrics:")
    
    # Missing data
    missing_data = df.isnull().sum()
    columns_with_missing = missing_data[missing_data > 0]
    print(f"  Columns with missing data: {len(columns_with_missing)}/{len(df.columns)}")
    
    if len(columns_with_missing) > 0:
        print(f"  Top missing data columns:")
        for col, count in columns_with_missing.head().items():
            percentage = count / len(df) * 100
            print(f"    {col}: {count:,} ({percentage:.1f}%)")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"  Duplicate rows: {duplicates:,}")
    
    # Status distribution
    if 'status' in df.columns:
        print(f"\nStatus Distribution:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            percentage = count / len(df) * 100
            print(f"  {status}: {count:,} ({percentage:.1f}%)")
    
    return True

if __name__ == "__main__":
    # Run validation report
    run_validation_report()
    
    # Run pytest if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        pytest.main([__file__, '-v'])