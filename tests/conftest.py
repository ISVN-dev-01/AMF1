# tests/conftest.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

@pytest.fixture(autouse=True)
def disable_network(monkeypatch):
    """
    Global fixture to prevent network calls in CI tests.
    This automatically runs for all tests and mocks external API calls.
    """
    
    # Mock requests.get globally
    class FakeResponse:
        status_code = 200
        
        def json(self):
            return {
                "MRData": {
                    "RaceTable": {
                        "Races": [
                            {
                                "season": "2024",
                                "round": "1",
                                "raceName": "Test Grand Prix",
                                "Circuit": {
                                    "circuitId": "test_circuit",
                                    "circuitName": "Test Circuit"
                                },
                                "date": "2024-03-10",
                                "time": "15:00:00Z"
                            }
                        ]
                    }
                }
            }
        
        def raise_for_status(self):
            pass
    
    monkeypatch.setattr('requests.get', lambda *args, **kwargs: FakeResponse())
    
    # Mock FastF1 session to return minimal object with required properties
    class FakeSession:
        def __init__(self, *args, **kwargs):
            self.weekend = MagicMock()
            self.weekend.name = "Test Grand Prix"
            self.event = MagicMock()
            self.event.year = 2024
            self.session_info = {"Name": "Race"}
            
        def load(self, *args, **kwargs):
            """Mock the load method"""
            pass
            
        @property
        def laps(self):
            """Return a mock DataFrame with F1 lap data structure"""
            return pd.DataFrame({
                'Time': [pd.Timedelta(seconds=90), pd.Timedelta(seconds=91)],
                'Driver': ['HAM', 'VER'],
                'LapTime': [pd.Timedelta(seconds=90), pd.Timedelta(seconds=91)],
                'Sector1Time': [pd.Timedelta(seconds=30), pd.Timedelta(seconds=29)],
                'Sector2Time': [pd.Timedelta(seconds=30), pd.Timedelta(seconds=31)],
                'Sector3Time': [pd.Timedelta(seconds=30), pd.Timedelta(seconds=31)],
                'SpeedI1': [320.5, 318.2],
                'SpeedI2': [195.8, 197.1],
                'SpeedFL': [310.2, 312.5],
                'SpeedST': [285.6, 287.3],
                'IsPersonalBest': [True, False],
                'Compound': ['SOFT', 'MEDIUM'],
                'TyreLife': [5, 12],
                'FreshTyre': [True, False],
                'Team': ['Mercedes', 'Red Bull Racing'],
                'LapNumber': [1, 2],
                'Stint': [1, 1],
                'PitOutTime': [pd.NaT, pd.NaT],
                'PitInTime': [pd.NaT, pd.NaT],
                'Position': [1, 2],
                'Deleted': [False, False],
                'DeletedReason': [None, None],
                'FastF1Generated': [False, False],
                'IsAccurate': [True, True]
            })
        
        @property  
        def results(self):
            """Return mock race results"""
            return pd.DataFrame({
                'DriverNumber': ['44', '1'],
                'BroadcastName': ['L HAMILTON', 'M VERSTAPPEN'],
                'Abbreviation': ['HAM', 'VER'],
                'DriverId': ['hamilton', 'max_verstappen'],
                'TeamName': ['Mercedes', 'Red Bull Racing'],
                'TeamColor': ['00D2BE', '0600EF'],
                'TeamId': ['mercedes', 'red_bull'],
                'FirstName': ['Lewis', 'Max'],
                'LastName': ['Hamilton', 'Verstappen'],
                'FullName': ['Lewis Hamilton', 'Max Verstappen'],
                'HeadshotUrl': ['https://example.com/hamilton.png', 'https://example.com/verstappen.png'],
                'CountryCode': ['GBR', 'NED'],
                'Position': [1, 2],
                'ClassifiedPosition': ['1', '2'],
                'GridPosition': [2, 1],
                'Q1': [pd.Timedelta(seconds=78.5), pd.Timedelta(seconds=78.2)],
                'Q2': [pd.Timedelta(seconds=77.8), pd.Timedelta(seconds=77.5)],
                'Q3': [pd.Timedelta(seconds=76.9), pd.Timedelta(seconds=76.6)],
                'Time': [pd.Timedelta(seconds=5400), pd.Timedelta(seconds=5420)],
                'Status': ['Finished', 'Finished'],
                'Points': [25.0, 18.0]
            })
    
    # Mock fastf1.get_session
    try:
        import fastf1
        monkeypatch.setattr('fastf1.get_session', lambda *args, **kwargs: FakeSession())
    except ImportError:
        pass
    
    # Mock other potential network calls
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: MagicMock())
    
    # Mock matplotlib to prevent display issues in CI
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        pass


@pytest.fixture
def sample_lap_data():
    """Fixture providing sample F1 lap data for testing"""
    return pd.DataFrame({
        'Time': [pd.Timedelta(seconds=90), pd.Timedelta(seconds=91)],
        'Driver': ['HAM', 'VER'],
        'LapTime': [pd.Timedelta(seconds=90), pd.Timedelta(seconds=91)],
        'Sector1Time': [pd.Timedelta(seconds=30), pd.Timedelta(seconds=29)],
        'Sector2Time': [pd.Timedelta(seconds=30), pd.Timedelta(seconds=31)],
        'Sector3Time': [pd.Timedelta(seconds=30), pd.Timedelta(seconds=31)],
        'SpeedI1': [320.5, 318.2],
        'SpeedI2': [195.8, 197.1],
        'SpeedFL': [310.2, 312.5],
        'SpeedST': [285.6, 287.3],
        'IsPersonalBest': [True, False],
        'Compound': ['SOFT', 'MEDIUM'],
        'TyreLife': [5, 12],
        'Team': ['Mercedes', 'Red Bull Racing'],
        'LapNumber': [1, 2],
        'Position': [1, 2]
    })


@pytest.fixture
def mock_f1_models():
    """Fixture providing mock trained models for testing"""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    import tempfile
    import os
    
    # Create simple mock models
    models = {}
    temp_dir = tempfile.mkdtemp()
    
    # Create and save mock models
    for model_name in ['linear_regression', 'random_forest', 'xgboost']:
        if model_name == 'linear_regression':
            model = LinearRegression()
            # Fit with dummy data
            X_dummy = np.array([[1, 2], [3, 4], [5, 6]])
            y_dummy = np.array([1, 2, 3])
            model.fit(X_dummy, y_dummy)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            X_dummy = np.array([[1, 2], [3, 4], [5, 6]])
            y_dummy = np.array([1, 2, 3])
            model.fit(X_dummy, y_dummy)
        
        model_path = os.path.join(temp_dir, f'{model_name}.joblib')
        joblib.dump(model, model_path)
        models[model_name] = model_path
    
    yield models
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)