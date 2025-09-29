# AMF1 - ML Model Project

## Project Structure

```
AMF1/
├── data/                   # Data storage
│   ├── raw/               # Raw, immutable data
│   ├── processed/         # Cleaned and preprocessed data
│   └── features/          # Engineered features
├── cache/                 # Cached data and intermediate results
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/                   # Source code
│   ├── data_collection/   # Data collection scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   ├── serve/             # Model serving and API
│   └── utils/             # Utility functions
├── models/                # Trained models
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the FastAPI application:
```bash
uvicorn main:app --reload
```

## Project Description

This is an ML model project with FastAPI integration for serving predictions.