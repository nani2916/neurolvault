import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Logging — acts like a diary, prints what's happening at each step
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths — tells Python exactly where files are located
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "Loan_Default.csv"

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Dataset path: {RAW_DATA_PATH}")


def load_data():
    """Load raw CSV into a DataFrame"""
    logger.info("Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def inspect_data(df):
    """Check data quality before cleaning"""
    logger.info("--- Missing values per column ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)

    logger.info(f"--- Duplicate rows: {df.duplicated().sum()} ---")
    logger.info(f"--- Target column 'Status' distribution ---")
    print(df['Status'].value_counts())


def clean_data(df):
    """Handle missing values and basic cleaning"""
    logger.info("Cleaning data...")

    # Numeric columns — fill missing with median (robust to outliers)
    numeric_cols = ['Upfront_charges', 'Interest_rate_spread', 'rate_of_interest',
                     'dtir1', 'property_value', 'LTV', 'income']
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        logger.info(f"Filled '{col}' missing values with median: {median_val:.2f}")

    # Categorical columns — fill missing with 'Unknown'
    categorical_cols = ['loan_limit', 'approv_in_adv', 'submission_of_application',
                         'age', 'loan_purpose', 'Neg_ammortization']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        logger.info(f"Filled '{col}' missing values with 'Unknown'")

    # term has very few missing (41) — safe to drop those rows
    before = df.shape[0]
    df = df.dropna(subset=['term'])
    logger.info(f"Dropped {before - df.shape[0]} rows with missing 'term'")

    logger.info(f"Remaining missing values: {df.isnull().sum().sum()}")
    return df


def save_cleaned_data(df):
    """Save cleaned dataset for model training stage"""
    output_path = DATA_DIR / "Loan_Default_Cleaned.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to: {output_path}")
    logger.info(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    df = clean_data(df)
    save_cleaned_data(df)