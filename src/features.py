"""
Feature engineering utilities.

This module creates derived features from the cleaned dataset.
All features are designed to be available at ORDER TIME — meaning
we only use information that would realistically be known before delivery.

Important: Aggregated features (like category averages) must be computed
from TRAINING data only and then mapped to both train and test sets
to avoid data leakage.
"""

import pandas as pd
import numpy as np


def create_financial_features(df):
    """
    Create features related to order financials.
    
    Features created:
    - High_Value_Order: 1 if order sales above median, else 0
    - Discount_Percentage: discount as % of total item value
    - Sales_Per_Item: average sales value per item in order
    - Profit_Margin: benefit as % of sales
    - Order_Size: quantity × product price (total order value proxy)
    - Price_Difference: gap between product price and item product price
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new financial features added
    """
    df = df.copy()
    
    # Binary flag for high-value orders
    median_sales = df["Sales"].median()
    df["High_Value_Order"] = (df["Sales"] > median_sales).astype(int)
    
    # Discount as percentage of item value
    item_value = df["Order Item Product Price"] * df["Order Item Quantity"]
    df["Discount_Percentage"] = np.where(
        item_value > 0,
        (df["Order Item Discount"] / item_value) * 100,
        0
    )
    df["Discount_Percentage"] = df["Discount_Percentage"].fillna(0)
    
    # Average sales per item
    df["Sales_Per_Item"] = np.where(
        df["Order Item Quantity"] > 0,
        df["Sales"] / df["Order Item Quantity"],
        0
    )
    
    # Profit margin percentage
    df["Profit_Margin"] = np.where(
        df["Sales"] != 0,
        (df["Benefit per order"] / df["Sales"]) * 100,
        0
    )
    df["Profit_Margin"] = df["Profit_Margin"].fillna(0)
    
    # Total order size (value proxy)
    df["Order_Size"] = df["Order Item Quantity"] * df["Product Price"]
    
    # Price difference
    df["Price_Difference"] = df["Product Price"] - df["Order Item Product Price"]
    
    print("Created financial features: High_Value_Order, Discount_Percentage, "
          "Sales_Per_Item, Profit_Margin, Order_Size, Price_Difference")
    
    return df


def create_shipping_features(df):
    """
    Create features related to shipping and logistics.
    
    Features created:
    - Urgent_Shipment: 1 if scheduled shipping ≤ 2 days, else 0
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    
    df["Urgent_Shipment"] = (
        df["Days for shipment (scheduled)"] <= 2
    ).astype(int)
    
    print("Created shipping features: Urgent_Shipment")
    
    return df


def create_aggregated_features(df_train, df_test=None):
    """
    Create aggregated features computed FROM TRAINING DATA ONLY.
    
    This is important to prevent data leakage — we compute category
    averages and frequencies from the training set, then map them
    to both train and test.
    
    Features created:
    - Category_Avg_Sales: average sales for the product category (from train)
    - Category_Frequency: order count for the product category (from train)
    - Market_Frequency: order count for the market (from train)
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data (used to compute aggregations)
    df_test : pd.DataFrame, optional
        Test data (aggregations from train are mapped here)
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame) or pd.DataFrame
        Modified train (and test if provided)
    """
    df_train = df_train.copy()
    
    # Compute aggregations from training data only
    category_avg_sales = df_train.groupby("Category Name")["Sales"].mean()
    category_freq = df_train["Category Name"].value_counts()
    market_freq = df_train["Market"].value_counts()
    
    # Map to training data
    df_train["Category_Avg_Sales"] = df_train["Category Name"].map(category_avg_sales)
    df_train["Category_Frequency"] = df_train["Category Name"].map(category_freq)
    df_train["Market_Frequency"] = df_train["Market"].map(market_freq)
    
    print("Created aggregated features (from training data): "
          "Category_Avg_Sales, Category_Frequency, Market_Frequency")
    
    if df_test is not None:
        df_test = df_test.copy()
        
        # Map training aggregations to test data
        # Unknown categories in test get NaN, which we fill with overall mean/median
        df_test["Category_Avg_Sales"] = df_test["Category Name"].map(category_avg_sales)
        df_test["Category_Frequency"] = df_test["Category Name"].map(category_freq)
        df_test["Market_Frequency"] = df_test["Market"].map(market_freq)
        
        # Fill any NaN from unseen categories in test
        df_test["Category_Avg_Sales"] = df_test["Category_Avg_Sales"].fillna(
            category_avg_sales.mean()
        )
        df_test["Category_Frequency"] = df_test["Category_Frequency"].fillna(0)
        df_test["Market_Frequency"] = df_test["Market_Frequency"].fillna(0)
        
        return df_train, df_test
    
    return df_train


def create_order_features(df):
    """
    Convenience function to apply all non-aggregated feature engineering.
    
    Applies financial + shipping features. Aggregated features should be
    created separately using create_aggregated_features() after train/test
    split to prevent leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    df = create_financial_features(df)
    df = create_shipping_features(df)
    return df
