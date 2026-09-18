import os
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dow_jones.csv')

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), f'Data file not found at {DATA_PATH}'

def test_data_integrity():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 1000, 'Dataset should contain historical trading rows'
    
    # Check required columns
    required_cols = {'DATE', 'Close'}
    df.columns = df.columns.str.strip()
    assert required_cols.issubset(set(df.columns)), f'Missing required columns: {required_cols - set(df.columns)}'
    
    # Clean Close prices
    clean_close = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce').dropna()
    assert len(clean_close) > 0, 'No valid Close prices found'
    assert (clean_close > 0).all(), 'Close prices should be positive'
