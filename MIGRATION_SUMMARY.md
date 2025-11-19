# Migration Summary: Gale Pacific → M5 Dataset

This document summarizes the changes made to migrate the codebase from Gale Pacific dataset to M5 dataset defaults.

## Changes Made

### 1. EDA Notebooks Consolidation
- **Removed:**
  - `EDA_Spark.py` - Less comprehensive EDA notebook
  - `ForecastingAnalysisNotebook.py` - Less comprehensive analysis notebook
  - `ForecastingWithNeuralNet.py` - Neural network exploration notebook
- **Created:**
  - `EDA_M5.py` - Comprehensive consolidated EDA notebook for M5 dataset

### 2. Training Scripts Consolidation
- **Removed:**
  - `main_train_sklearn.py` - Separate sklearn/TabPFN training script
  - `main_train_spark.py` - Separate Spark ML training script
  - `main_train_m5.py` - Old M5 training script
- **Updated:**
  - `main_train.py` - Now includes all model types (Spark ML, StatsForecast, TabPFN) in one script
- **Created:**
  - `main_train_m5_local.py` - Training script for local M5 dataset ingestion
  - `main_train_m5_databricks.py` - Training script for Azure Databricks with blob storage

### 3. Azure Configuration
- **Created:**
  - `azure_config.py` - Configuration file for Azure credentials and blob storage settings
  - Supports environment variables and direct configuration
  - Includes helper functions for blob storage paths and validation

### 4. Data Upload Mechanism
- **Created:**
  - `upload_data.py` - Script for users to upload their own data matching M5 schema
  - Validates schema matches M5 requirements
  - Supports CSV and Parquet input formats

### 5. Configuration Updates
- **Updated:**
  - `config.py` - Changed default column mappings to M5 schema:
    - `product_id_column`: "SalesInvoiceProductKey" → "item_id"
    - Other columns remain compatible

### 6. Dataset References
- **Removed:** All references to "gale_pacific" dataset paths
- **Updated:** Default dataset is now M5 with schema:
  - `item_id` (product identifier)
  - `OrderDate` (date column)
  - `Quantity` (quantity/demand column)

## M5 Dataset Schema

The M5 dataset uses the following schema:
- **item_id**: Product identifier (String)
- **OrderDate**: Date of order/sale (Date)
- **Quantity**: Quantity/demand (Double)

## Usage

### Local Training
```bash
python main_train_m5_local.py --data-path data/m5/sales_train_validation.csv
```

### Databricks Training
Use the `main_train_m5_databricks.py` notebook in Databricks. Ensure Azure credentials are configured in `azure_config.py`.

### Upload Custom Data
```bash
python upload_data.py your_data.csv --output-path validated_data.parquet
```

### General Training (with custom data)
```bash
python main_train.py path/to/your/data.csv
```

## Azure Configuration

To use Azure Databricks and Blob Storage:

1. Set environment variables or update `azure_config.py`:
   ```python
   AZURE_STORAGE_ACCOUNT_NAME = "your_account"
   AZURE_STORAGE_ACCOUNT_KEY = "your_key"
   AZURE_STORAGE_CONTAINER_NAME = "m5-data"
   ```

2. Or use environment variables:
   ```bash
   export AZURE_STORAGE_ACCOUNT_NAME="your_account"
   export AZURE_STORAGE_ACCOUNT_KEY="your_key"
   ```

3. Check configuration:
   ```bash
   python azure_config.py
   ```

## Model Types Supported

All training scripts now support:
1. **Spark ML Models:**
   - Linear Regression
   - Random Forest
   - Gradient Boosted Trees

2. **Statistical Models (StatsForecast):**
   - Seasonal Exponential Smoothing
   - Croston Classic

3. **TabPFN:**
   - Transformer-based tabular model (if installed)

## Notes

- Product categories (Smooth, Erratic, Intermittent, Lumpy) are automatically generated during feature engineering
- All models are trained per category
- MLflow is used for experiment tracking
- Train/test split is based on 80th percentile of dates
