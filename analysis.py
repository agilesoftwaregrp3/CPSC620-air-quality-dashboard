"""
Data analysis and cleaning functions for the UCI Air Quality Dataset.

This module contains functions for loading, cleaning, and analyzing
air quality data from an Italian city monitoring station.
"""

import pandas as pd
import numpy as np


def load_data(file_path="data/AirQualityUCI.csv"):
    """
    Load the air quality dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Load data with semicolon separator
        df = pd.read_csv(file_path, sep=';')
        
        # Remove empty columns (the dataset has trailing semicolons)
        df = df.dropna(axis=1, how='all')
        
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(df):
    """
    Clean the air quality dataset by handling missing values and data types.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace -200 values (missing data indicator) with NaN
    df_clean = df_clean.replace(-200, np.nan)
    
    # Robust Date parsing: infer common formats (handles MM/DD/YYYY and DD/MM/YYYY)
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(
            df_clean['Date'],
            dayfirst=False,
            errors='coerce',
            infer_datetime_format=True
        )
    
    # Robust Time parsing: accept "HH:MM:SS", "HH.MM.SS", "HH:MM" and mixed formats
    if 'Time' in df_clean.columns:
        time_series = df_clean['Time'].astype(str).str.strip()
        # Normalize dots to colons (e.g. "18.00.00" -> "18:00:00")
        time_norm = time_series.str.replace('.', ':', regex=False)
        
        parsed = pd.to_datetime(time_norm, format='%H:%M:%S', errors='coerce')
        parsed = parsed.combine_first(pd.to_datetime(time_norm, format='%H:%M', errors='coerce'))
        # Final fallback: let pandas infer mixed formats
        parsed = parsed.combine_first(pd.to_datetime(time_series, errors='coerce', infer_datetime_format=True))
        
        # Keep only the time portion (may be NaT if unparsed)
        df_clean['Time'] = parsed.dt.time
    
    # Create datetime column for easier time series analysis
    # Populate DateTime (capital T) and also create alias 'Datetime' for other code paths
    df_clean['DateTime'] = pd.NaT
    if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
        valid_datetime_mask = df_clean['Date'].notna() & df_clean['Time'].notna()
        if valid_datetime_mask.any():
            df_clean.loc[valid_datetime_mask, 'DateTime'] = pd.to_datetime(
                df_clean.loc[valid_datetime_mask, 'Date'].dt.strftime('%Y-%m-%d') + ' ' +
                df_clean.loc[valid_datetime_mask, 'Time'].astype(str),
                errors='coerce',
                infer_datetime_format=True
            )
    # alias used elsewhere in repo
    df_clean['Datetime'] = df_clean['DateTime']
    
    # Convert numeric columns, handling comma as decimal separator
    numeric_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 
                      'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 
                      'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            # Replace comma with dot for decimal separator
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '.')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean


def get_data_summary(df):
    """
    Get basic summary statistics for the dataset.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        dict: Summary statistics
    """
    if df is None:
        return {}
    
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['Date'].min(),
            'end': df['Date'].max()
        },
        'missing_data_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
    }
    
    return summary


def calculate_air_quality_metrics(df):
    """
    Calculate key air quality metrics and statistics.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        dict: Air quality metrics
    """
    if df is None:
        return {}
    
    metrics = {}
    
    # CO (Carbon Monoxide) metrics
    if 'CO(GT)' in df.columns:
        co_data = df['CO(GT)'].dropna()
        metrics['co'] = {
            'mean': co_data.mean(),
            'median': co_data.median(),
            'max': co_data.max(),
            'min': co_data.min(),
            'std': co_data.std()
        }
    
    # Temperature metrics
    if 'T' in df.columns:
        temp_data = df['T'].dropna()
        metrics['temperature'] = {
            'mean': temp_data.mean(),
            'median': temp_data.median(),
            'max': temp_data.max(),
            'min': temp_data.min(),
            'std': temp_data.std()
        }
    
    # Humidity metrics
    if 'RH' in df.columns:
        rh_data = df['RH'].dropna()
        metrics['humidity'] = {
            'mean': rh_data.mean(),
            'median': rh_data.median(),
            'max': rh_data.max(),
            'min': rh_data.min(),
            'std': rh_data.std()
        }
    
    # Absolute Humidity metrics
    if 'AH' in df.columns:
        ah_data = df['AH'].dropna()
        metrics['absolute_humidity'] = {
            'mean': ah_data.mean(),
            'median': ah_data.median(),
            'max': ah_data.max(),
            'min': ah_data.min(),
            'std': ah_data.std()
        }
    
    return metrics


def filter_by_date_range(df, start_date=None, end_date=None):
    """
    Filter dataset by date range.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    if df is None:
        return None
    
    df_filtered = df.copy()
    
    if start_date:
        df_filtered = df_filtered[df_filtered['Date'] >= pd.to_datetime(start_date)]
    
    if end_date:
        df_filtered = df_filtered[df_filtered['Date'] <= pd.to_datetime(end_date)]
    
    return df_filtered


def get_daily_averages(df):
    """
    Calculate daily averages for all numeric columns.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        pd.DataFrame: Daily averages
    """
    if df is None:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    daily_avg = df.groupby('Date')[numeric_cols].mean().reset_index()
    
    return daily_avg
