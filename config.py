"""
config.py

Configuration file for forecasting project

This file defines default variables used across multiple scripts to ensure consistency.
These values are used as defaults but can be overridden via command-line arguments.
"""

# =============================================================================
# M5 Dataset Configuration (Default)
# =============================================================================

# Data source paths
SOURCE_PATH = "data/m5/sales_train_validation.csv"
AZURE_BLOB_PATH = "/mnt/m5/sales_train_validation.csv"  # For Databricks

# Column mappings for M5 dataset
DATE_COLUMN = "OrderDate"
PRODUCT_ID_COLUMN = "item_id"
QUANTITY_COLUMN = "Quantity"
MONTH_END_COLUMN = "MonthEndDate"

# Target column (created by feature engineering)
TARGET_COLUMN = "lead_month_1"

# =============================================================================
# Feature Engineering Configuration
# =============================================================================

# Moving average windows (in months)
MA_WINDOWS = [4, 8, 12]

# Lag features (in months)
LAG_FEATURES = [1, 2, 3, 4, 5, 6, 11, 12]

# Minimum number of orders required for training
MIN_ORDERS = 5

# =============================================================================
# Training Configuration
# =============================================================================

# Train/test split ratio (0.8 = 80% train, 20% test)
TRAIN_SPLIT_RATIO = 0.8

# Maximum number of distinct items to train on (for computational efficiency)
MAX_DISTINCT_ITEMS = 100

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# Model Configuration
# =============================================================================

# Spark ML Model Parameters
SPARK_ML_PARAMS = {
    "RandomForest": {
        "maxDepth": [2, 3, 5],
        "numTrees": [10, 50, 100]
    },
    "GradientBoostedTrees": {
        "maxDepth": [2, 3, 5],
        "maxIter": [10, 20]
    }
}

# Statistical Model Parameters
STATS_MODEL_PARAMS = {
    "SeasonalExponentialSmoothing": {
        "season_length": 12  # Monthly data with yearly seasonality
    }
}

# =============================================================================
# MLflow Configuration
# =============================================================================

# MLflow tracking URI (use 'databricks' when running in Databricks)
MLFLOW_TRACKING_URI = None  # None uses default local tracking

# MLflow experiment base path
MLFLOW_EXPERIMENT_BASE = "/Users"

# Default experiment name
DEFAULT_EXPERIMENT_NAME = "M5_Forecasting"

# =============================================================================
# Custom Dataset Configuration
# =============================================================================

# When using custom datasets, they must match this schema
REQUIRED_COLUMNS = [
    "date",        # Date column (will be mapped to DATE_COLUMN)
    "item_id",     # Product/item identifier
    "quantity"     # Quantity/demand value
]

# Optional columns that enhance the model
OPTIONAL_COLUMNS = [
    "store_id",    # Store identifier
    "cat_id",      # Category identifier
    "dept_id",     # Department identifier
    "state_id"     # State/region identifier
]

# =============================================================================
# Helper Functions
# =============================================================================

def get_feature_columns():
    """
    Returns the list of feature columns used in training.
    
    Returns:
        list: List of feature column names
    """
    return [
        QUANTITY_COLUMN,
        "ma_4_month",
        "ma_8_month",
        "ma_12_month",
        "avg_demand_quantity",
        "avg_demand_interval",
        "months_since_last_order",
        "last_order_quantity",
        "month",
        "year"
    ]


def get_column_mappings():
    """
    Returns a dictionary of column mappings.
    
    Returns:
        dict: Column name mappings
    """
    return {
        "date_column": DATE_COLUMN,
        "product_id_column": PRODUCT_ID_COLUMN,
        "quantity_column": QUANTITY_COLUMN,
        "month_end_column": MONTH_END_COLUMN,
        "target_column": TARGET_COLUMN
    }


def get_training_config():
    """
    Returns training configuration as a dictionary.
    
    Returns:
        dict: Training configuration
    """
    return {
        "train_split_ratio": TRAIN_SPLIT_RATIO,
        "max_distinct_items": MAX_DISTINCT_ITEMS,
        "random_seed": RANDOM_SEED,
        "min_orders": MIN_ORDERS
    }


def validate_custom_dataset_columns(df_columns):
    """
    Validates that a custom dataset has required columns.
    
    Args:
        df_columns: List of column names in the dataset
        
    Returns:
        tuple: (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df_columns]
    is_valid = len(missing) == 0
    return is_valid, missing


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    print("Configuration Summary:")
    print("=" * 60)
    print(f"Default Data Source: {SOURCE_PATH}")
    print(f"Date Column: {DATE_COLUMN}")
    print(f"Product ID Column: {PRODUCT_ID_COLUMN}")
    print(f"Quantity Column: {QUANTITY_COLUMN}")
    print(f"Target Column: {TARGET_COLUMN}")
    print(f"Train/Test Split: {int(TRAIN_SPLIT_RATIO * 100)}% / {int((1 - TRAIN_SPLIT_RATIO) * 100)}%")
    print(f"Max Items for Training: {MAX_DISTINCT_ITEMS}")
    print(f"Min Orders Required: {MIN_ORDERS}")
    print("\nFeature Columns:")
    for feature in get_feature_columns():
        print(f"  - {feature}")
