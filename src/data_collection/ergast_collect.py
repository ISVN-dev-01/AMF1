import requests
import pandas as pd
from pathlib import Path

def fetch_json(url):
    """Fetch JSON data from URL with error handling"""
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def collect_races(year):
    """Collect race data for a given year"""
    url = f'http://ergast.com/api/f1/{year}.json?limit=1000'
    data = fetch_json(url)
    races = data['MRData']['RaceTable']['Races']
    df = pd.json_normalize(races)
    df['season'] = year
    return df

def collect_qualifying(year):
    """Collect qualifying data for a given year"""
    url = f'http://ergast.com/api/f1/{year}/qualifying.json?limit=1000'
    data = fetch_json(url)
    races = data['MRData']['RaceTable']['Races']
    
    qualifying_data = []
    for race in races:
        if 'QualifyingResults' in race:
            for result in race['QualifyingResults']:
                result['season'] = year
                result['round'] = race['round']
                result['raceName'] = race['raceName']
                result['date'] = race['date']
                result['circuitId'] = race['Circuit']['circuitId']
                qualifying_data.append(result)
    
    return pd.json_normalize(qualifying_data)

def collect_results(year):
    """Collect race results for a given year"""
    url = f'http://ergast.com/api/f1/{year}/results.json?limit=1000'
    data = fetch_json(url)
    races = data['MRData']['RaceTable']['Races']
    
    results_data = []
    for race in races:
        if 'Results' in race:
            for result in race['Results']:
                result['season'] = year
                result['round'] = race['round']
                result['raceName'] = race['raceName']
                result['date'] = race['date']
                result['circuitId'] = race['Circuit']['circuitId']
                results_data.append(result)
    
    return pd.json_normalize(results_data)

def main():
    """Main function to collect all Ergast data"""
    # Create output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Years to collect
    years = range(2014, 2025)
    
    # Collect races data
    print("Collecting races data...")
    races_data = []
    for year in years:
        print(f"  Fetching races for {year}")
        df = collect_races(year)
        races_data.append(df)
    
    races_df = pd.concat(races_data, ignore_index=True)
    races_df.to_parquet('data/raw/races.parquet', index=False)
    print(f"Saved {len(races_df)} race records to races.parquet")
    
    # Collect qualifying data
    print("Collecting qualifying data...")
    qualifying_data = []
    for year in years:
        print(f"  Fetching qualifying for {year}")
        df = collect_qualifying(year)
        if not df.empty:
            qualifying_data.append(df)
    
    if qualifying_data:
        qualifying_df = pd.concat(qualifying_data, ignore_index=True)
        qualifying_df.to_parquet('data/raw/qualifying.parquet', index=False)
        print(f"Saved {len(qualifying_df)} qualifying records to qualifying.parquet")
    
    # Collect results data
    print("Collecting results data...")
    results_data = []
    for year in years:
        print(f"  Fetching results for {year}")
        df = collect_results(year)
        if not df.empty:
            results_data.append(df)
    
    if results_data:
        results_df = pd.concat(results_data, ignore_index=True)
        results_df.to_parquet('data/raw/results.parquet', index=False)
        print(f"Saved {len(results_df)} result records to results.parquet")

if __name__ == "__main__":
    main()