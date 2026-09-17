"""
Data loading and preprocessing utilities.

This module handles the initial data loading, cleaning, and preparation
steps that happen before feature engineering.
"""

import pandas as pd
import numpy as np


def load_data(filepath="data/raw/DataCoSupplyChainDataset.csv"):
    """
    Load the DataCo Supply Chain dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file (default: data/raw/DataCoSupplyChainDataset.csv)
    
    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns
    """
    df = pd.read_csv(filepath, encoding="latin1")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def remove_leakage_columns(df):
    """
    Remove columns that would cause data leakage.
    
    These columns contain information that is only available AFTER delivery,
    so they cannot be used to predict delivery risk at order time.
    
    Removed columns:
    - Days for shipping (real): actual shipping time — only known after delivery
    - Delivery Status: final delivery outcome — this IS the target in another form
    - shipping date (DateOrders): actual shipping date — only known after shipment
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain leakage columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with leakage columns removed
    """
    leakage_cols = [
        "Days for shipping (real)",
        "Delivery Status",
        "shipping date (DateOrders)"
    ]
    
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Removed leakage columns: {cols_to_drop}")
    
    return df


def clean_data(df):
    """
    Clean the dataset by removing unnecessary columns and handling data types.
    
    Steps:
    1. Remove PII and identifier columns (emails, names, passwords, IDs)
    2. Remove columns with excessive missing values (Order Zipcode: 86% missing)
    3. Remove duplicate rows
    4. Convert date columns to datetime
    5. Extract date features (year, month, day, day of week)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    initial_shape = df.shape
    
    # --- Remove PII and identifier columns ---
    # These don't carry predictive information and shouldn't be used
    drop_cols = [
        "Customer Email",
        "Customer Fname",
        "Customer Lname",
        "Customer Password",
        "Customer Street",
        "Product Description",   # 100% missing
        "Product Image",
        "Order Customer Id",
        "Order Id",
        "Order Item Id",
        "Customer Id",
        "Product Card Id",
        "Category Id",
        "Department Id",
        "Product Category Id",
        "Latitude",
        "Longitude",
        "Order Zipcode",         # 86% missing — not usable
    ]
    
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} PII/ID/unusable columns")
    
    # --- Remove duplicates ---
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"Removed {n_dupes} duplicate rows")
    
    # --- Parse dates and extract features ---
    if "order date (DateOrders)" in df.columns:
        df["order date (DateOrders)"] = pd.to_datetime(
            df["order date (DateOrders)"]
        )
        df["Order_Year"] = df["order date (DateOrders)"].dt.year
        df["Order_Month"] = df["order date (DateOrders)"].dt.month
        df["Order_Day"] = df["order date (DateOrders)"].dt.day
        df["Order_DayOfWeek"] = df["order date (DateOrders)"].dt.day_name()
        df["Order_Hour"] = df["order date (DateOrders)"].dt.hour
        
        # Keep the date column for potential time-series analysis later
        print("Extracted date features: Year, Month, Day, DayOfWeek, Hour")
    
    print(f"Cleaning complete: {initial_shape} → {df.shape}")
    
    return df


def get_column_summary(df):
    """
    Get a summary of column types and missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        Summary with dtype, non-null count, missing count and percentage
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notnull().sum(),
        "missing": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "n_unique": df.nunique()
    })
    return summary.sort_values("missing_pct", ascending=False)
