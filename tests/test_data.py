import os
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORICAL_PATH = os.path.join(BASE_DIR, 'data', 'dow_jones.csv')
MODERN_PATH = os.path.join(BASE_DIR, 'data', 'dow_jones_modern_2015_2025.csv')

def test_historical_dataset_dates_and_counts():
    """Verify authentic historical Dow Jones archive covers 1901 to 2012 with 27,824 valid trading days."""
    assert os.path.exists(HISTORICAL_PATH), f"Historical data file missing: {HISTORICAL_PATH}"
    df = pd.read_csv(HISTORICAL_PATH)
    df.columns = df.columns.str.strip()
    assert 'DATE' in df.columns and 'Close' in df.columns

    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    clean_close = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce')
    valid_df = df.dropna(subset=['DATE']).copy()
    valid_df = valid_df[clean_close > 0].reset_index(drop=True)

    # Must match authentic historical record exactly
    assert len(valid_df) == 27824, f"Expected 27,824 valid historical observations, found {len(valid_df)}"
    assert valid_df['DATE'].nunique() == 27824, "Every historical trading session must have a unique date (0 duplicates)"
    assert valid_df['DATE'].min().strftime('%Y-%m-%d') == '1901-01-04'
    assert valid_df['DATE'].max().strftime('%Y-%m-%d') == '2012-12-31'

def test_modern_dataset_dates_and_counts():
    """Verify modern Dow Jones benchmark covers 2015 to 2025 with 2,765 daily sessions."""
    assert os.path.exists(MODERN_PATH), f"Modern data file missing: {MODERN_PATH}"
    df = pd.read_csv(MODERN_PATH)
    df.columns = df.columns.str.strip()
    assert 'DATE' in df.columns and 'Close' in df.columns

    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    valid_df = df.dropna(subset=['DATE', 'Close']).reset_index(drop=True)

    assert len(valid_df) == 2765, f"Expected 2,765 modern trading sessions, found {len(valid_df)}"
    assert valid_df['DATE'].nunique() == 2765, "Zero duplicate dates in modern dataset"
    assert valid_df['DATE'].min().strftime('%Y-%m-%d') == '2015-01-02'
    assert valid_df['DATE'].max().strftime('%Y-%m-%d') == '2025-12-30'
    assert (valid_df['Close'] > 15000).all(), "Dow Jones closing prices in 2015-2025 must reflect authentic levels (>15,000 pts)"
