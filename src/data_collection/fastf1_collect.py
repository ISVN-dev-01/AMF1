import fastf1
import pandas as pd
from pathlib import Path
import warnings

# Suppress FastF1 warnings
warnings.filterwarnings('ignore')

# Enable cache for faster data loading
fastf1.Cache.enable_cache('cache/fastf1')

def fetch_session(year, event, session_type):
    """
    Fetch session data from FastF1
    
    Args:
        year: Season year
        event: Event name or number
        session_type: 'FP1', 'FP2', 'FP3', 'Q', 'S', 'R'
    """
    try:
        sess = fastf1.get_session(year, event, session_type)
        sess.load()
        
        if sess.laps.empty:
            print(f"No lap data for {year} {event} {session_type}")
            return pd.DataFrame()
        
        # Get lap data with key columns
        laps = sess.laps
        
        # Create session identifier
        session_id = f'{year}_{event}_{session_type}'
        
        # Select relevant columns (handle missing columns gracefully)
        columns_to_keep = ['Driver', 'Team', 'LapNumber', 'LapTime', 'Compound']
        optional_columns = ['Sector1Time', 'Sector2Time', 'Sector3Time', 'SpeedI1', 'SpeedI2', 'SpeedST']
        
        # Add columns that exist
        for col in optional_columns:
            if col in laps.columns:
                columns_to_keep.append(col)
        
        # Filter to existing columns
        existing_columns = [col for col in columns_to_keep if col in laps.columns]
        laps_filtered = laps[existing_columns].copy()
        
        # Add session metadata
        laps_filtered['session'] = session_id
        laps_filtered['year'] = year
        laps_filtered['event'] = event
        laps_filtered['session_type'] = session_type
        
        return laps_filtered
        
    except Exception as e:
        print(f"Error fetching {year} {event} {session_type}: {e}")
        return pd.DataFrame()

def collect_season_data(year, events=None):
    """
    Collect data for all events in a season
    
    Args:
        year: Season year
        events: List of events (if None, fetch all)
    """
    if events is None:
        # Get schedule for the year
        try:
            schedule = fastf1.get_event_schedule(year)
            events = schedule['EventName'].tolist()
        except:
            # Fallback to common events
            events = ['Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami', 
                     'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain',
                     'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Singapore',
                     'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi']
    
    session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']  # Practice, Qualifying, Race
    all_sessions = []
    
    for event in events:
        print(f"Processing {year} {event}")
        for session_type in session_types:
            session_data = fetch_session(year, event, session_type)
            if not session_data.empty:
                all_sessions.append(session_data)
    
    if all_sessions:
        return pd.concat(all_sessions, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    """Main function to collect FastF1 data"""
    # Create output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache directory
    cache_dir = Path('cache/fastf1')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Example: Collect specific session
    print("Collecting sample FastF1 data (Monaco 2024 Qualifying)...")
    laps = fetch_session(2024, 'Monaco', 'Q')
    
    if not laps.empty:
        laps.to_parquet('data/raw/fastf1_q_monaco_2024.parquet', index=False)
        print(f"Saved {len(laps)} lap records to fastf1_q_monaco_2024.parquet")
    
    # Uncomment below to collect full season (WARNING: This takes a long time!)
    # print("Collecting full 2024 season data...")
    # season_data = collect_season_data(2024)
    # if not season_data.empty:
    #     season_data.to_parquet('data/raw/fastf1_2024_full.parquet', index=False)
    #     print(f"Saved {len(season_data)} session records")

if __name__ == "__main__":
    main()