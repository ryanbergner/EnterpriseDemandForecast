"""
Azure Configuration for Databricks and Blob Storage

This module provides configuration for connecting to Azure services.
Users can either:
1. Set environment variables
2. Modify the values directly in this file
3. Use Azure Key Vault (recommended for production)

For security, it's recommended to use environment variables or Azure Key Vault
rather than hardcoding credentials.
"""

import os
from typing import Optional

# ============================================================================
# Azure Blob Storage Configuration
# ============================================================================

# Storage account name
AZURE_STORAGE_ACCOUNT_NAME: Optional[str] = os.getenv(
    "AZURE_STORAGE_ACCOUNT_NAME",
    None  # Set your storage account name here or via environment variable
)

# Storage account key (for authentication)
AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = os.getenv(
    "AZURE_STORAGE_ACCOUNT_KEY",
    None  # Set your storage account key here or via environment variable
)

# Storage account connection string (alternative to account name + key)
AZURE_STORAGE_CONNECTION_STRING: Optional[str] = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING",
    None  # Set your connection string here or via environment variable
)

# Container name where M5 dataset is stored
AZURE_STORAGE_CONTAINER_NAME: Optional[str] = os.getenv(
    "AZURE_STORAGE_CONTAINER_NAME",
    "m5-data"  # Default container name
)

# Blob path to M5 dataset files
AZURE_BLOB_PATH: Optional[str] = os.getenv(
    "AZURE_BLOB_PATH",
    "sales_train_validation.csv"  # Path within container
)

# ============================================================================
# Azure Databricks Configuration
# ============================================================================

# Databricks workspace URL
DATABRICKS_WORKSPACE_URL: Optional[str] = os.getenv(
    "DATABRICKS_WORKSPACE_URL",
    None  # e.g., "https://adb-1234567890123456.7.azuredatabricks.net"
)

# Databricks personal access token
DATABRICKS_TOKEN: Optional[str] = os.getenv(
    "DATABRICKS_TOKEN",
    None  # Set your Databricks token here or via environment variable
)

# Databricks cluster ID (optional, for job execution)
DATABRICKS_CLUSTER_ID: Optional[str] = os.getenv(
    "DATABRICKS_CLUSTER_ID",
    None
)

# ============================================================================
# MLflow Configuration
# ============================================================================

# MLflow tracking URI (for Databricks, this is usually auto-configured)
MLFLOW_TRACKING_URI: Optional[str] = os.getenv(
    "MLFLOW_TRACKING_URI",
    None  # Auto-configured in Databricks, set manually for local execution
)

# MLflow experiment name prefix
MLFLOW_EXPERIMENT_PREFIX: str = os.getenv(
    "MLFLOW_EXPERIMENT_PREFIX",
    "/M5_Forecasting"
)

# ============================================================================
# Helper Functions
# ============================================================================

def get_blob_storage_path() -> str:
    """
    Returns the full blob storage path for the M5 dataset.
    
    Returns:
        str: Full path in format: abfss://container@account.dfs.core.windows.net/path
    """
    if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_CONTAINER_NAME:
        raise ValueError(
            "Azure storage account name and container name must be set. "
            "Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_CONTAINER_NAME "
            "environment variables or update azure_config.py"
        )
    
    return f"abfss://{AZURE_STORAGE_CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/{AZURE_BLOB_PATH}"


def get_blob_storage_mount_path() -> str:
    """
    Returns the mount path for blob storage (for Databricks).
    
    Returns:
        str: Mount path like /mnt/m5
    """
    return "/mnt/m5"


def validate_azure_config() -> bool:
    """
    Validates that required Azure configuration is present.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_vars = [
        AZURE_STORAGE_ACCOUNT_NAME,
        AZURE_STORAGE_CONTAINER_NAME,
    ]
    
    # Check if at least one authentication method is provided
    has_auth = (
        AZURE_STORAGE_ACCOUNT_KEY is not None or
        AZURE_STORAGE_CONNECTION_STRING is not None
    )
    
    if not all(required_vars) or not has_auth:
        return False
    
    return True


def print_config_status():
    """Prints the current configuration status."""
    print("=" * 60)
    print("Azure Configuration Status")
    print("=" * 60)
    
    print(f"\nBlob Storage:")
    print(f"  Account Name: {'✓ Set' if AZURE_STORAGE_ACCOUNT_NAME else '✗ Not Set'}")
    print(f"  Container: {'✓ Set' if AZURE_STORAGE_CONTAINER_NAME else '✗ Not Set'}")
    print(f"  Blob Path: {AZURE_BLOB_PATH}")
    print(f"  Account Key: {'✓ Set' if AZURE_STORAGE_ACCOUNT_KEY else '✗ Not Set'}")
    print(f"  Connection String: {'✓ Set' if AZURE_STORAGE_CONNECTION_STRING else '✗ Not Set'}")
    
    print(f"\nDatabricks:")
    print(f"  Workspace URL: {'✓ Set' if DATABRICKS_WORKSPACE_URL else '✗ Not Set'}")
    print(f"  Token: {'✓ Set' if DATABRICKS_TOKEN else '✗ Not Set'}")
    
    print(f"\nMLflow:")
    print(f"  Tracking URI: {MLFLOW_TRACKING_URI or 'Auto-configured'}")
    print(f"  Experiment Prefix: {MLFLOW_EXPERIMENT_PREFIX}")
    
    print("\n" + "=" * 60)
    
    if validate_azure_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration is incomplete. Please set required variables.")
    print("=" * 60)


if __name__ == "__main__":
    # Print configuration status when run directly
    print_config_status()
