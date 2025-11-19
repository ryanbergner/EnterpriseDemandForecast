# M5 Forecasting Pipeline

A comprehensive forecasting pipeline for time series demand prediction using both statistical and machine learning models. Built for the M5 forecasting dataset with support for custom datasets.

## 🎯 Overview

This project provides a complete end-to-end pipeline for demand forecasting:

- **Data Ingestion**: Support for M5 dataset (local and Azure Blob Storage)
- **Feature Engineering**: Automated feature creation with moving averages, lags, and more
- **Model Training**: Spark ML, StatsForecast, TabPFN, and Neural Networks
- **Experiment Tracking**: MLflow integration for all experiments
- **Custom Data Support**: Upload and validate your own datasets

## 📁 Project Structure

```
.
├── EDA.py                          # Comprehensive EDA notebook
├── main_train_m5.py                # Training script for local M5 dataset
├── main_train_databricks.py        # Training script for Azure Databricks
├── main_train.py                   # Generic training script
├── upload_custom_data.py           # Custom data upload and validation
├── config.py                       # Configuration settings
├── azure_config.py                 # Azure credentials (create from template)
├── cli.py                          # Command-line interface
├── requirements.txt                # Python dependencies
├── src/
│   ├── feature_engineering/        # Feature engineering modules
│   ├── inference/                  # Inference and prediction
│   ├── model_training/             # Model training utilities
│   ├── preprocessing/              # Data preprocessing
│   ├── validation/                 # Data validation
│   └── visualization/              # Visualization tools
└── data/                           # Data directory (not in git)
    └── m5/                         # M5 dataset location
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download M5 Dataset

Download the M5 forecasting dataset from [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy/data):

```bash
# Create data directory
mkdir -p data/m5

# Place the following files in data/m5/:
# - sales_train_validation.csv
# - calendar.csv (optional)
# - sell_prices.csv (optional)
```

### 3. Run EDA

```bash
# Launch Databricks notebook or Jupyter with PySpark
jupyter notebook EDA.py
```

### 4. Train Models (Local)

```bash
# Train on M5 dataset
python main_train_m5.py \
  --data-path data/m5/sales_train_validation.csv \
  --experiment-name M5_Experiment \
  --max-items 100

# View results in MLflow UI
mlflow ui
```

### 5. Train Models (Azure Databricks)

First, configure Azure credentials:

```bash
# Copy the template
cp azure_config.py.template azure_config.py

# Edit azure_config.py and add your credentials
# OR set environment variables:
export AZURE_STORAGE_ACCOUNT_NAME="your_account"
export AZURE_STORAGE_ACCOUNT_KEY="your_key"
export AZURE_STORAGE_CONTAINER_NAME="m5-forecasting"
```

Then run training:

```bash
# For Databricks notebook, import and run
%run /Workspace/path/to/main_train_databricks

# Or as a job
python main_train_databricks.py \
  --experiment-name M5_Production \
  --max-items 500 \
  --user-email your.email@company.com
```

## 📊 Using Custom Data

You can use your own dataset by following these steps:

### 1. Prepare Your Data

Your CSV must have these columns:
- **date**: Date column (any standard date format)
- **item_id**: Product/item identifier
- **quantity**: Demand/sales quantity

Optional columns:
- store_id, cat_id, dept_id, state_id

### 2. Upload and Validate

```bash
# Auto-detect columns
python upload_custom_data.py \
  --input-path my_sales_data.csv \
  --output-path data/custom/validated_data.csv

# Or specify column names
python upload_custom_data.py \
  --input-path my_sales_data.csv \
  --output-path data/custom/validated_data.csv \
  --date-col OrderDate \
  --item-id-col SKU \
  --quantity-col SalesQty
```

### 3. Train Models

```bash
python main_train.py \
  --data-path data/custom/validated_data.csv \
  --experiment-name Custom_Dataset_Experiment
```

## 🔧 Configuration

Edit `config.py` to customize:

- Feature engineering parameters (MA windows, lag features)
- Train/test split ratio
- Maximum number of items to train
- Model hyperparameters
- MLflow settings

## 🤖 Available Models

### Statistical Models
- **Seasonal Exponential Smoothing**: Handles seasonality
- **Croston Classic**: For intermittent demand

### Machine Learning Models
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble tree-based model
- **Gradient Boosted Trees**: Advanced gradient boosting
- **TabPFN**: Foundation model for tabular data (if available)
- **Neural Networks**: Deep learning models with Keras/TensorFlow

## 📈 Model Evaluation

All models are tracked with MLflow and evaluated on:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

View results:
```bash
mlflow ui
# Open browser to http://localhost:5000
```

## 🔐 Azure Databricks Setup

### Option 1: Using azure_config.py

1. Create `azure_config.py` from the provided template
2. Fill in your Azure credentials:
   ```python
   AZURE_STORAGE_ACCOUNT_NAME = "mystorageaccount"
   AZURE_STORAGE_ACCOUNT_KEY = "your_key_here"
   AZURE_STORAGE_CONTAINER_NAME = "m5-forecasting"
   ```

### Option 2: Using Environment Variables

```bash
export AZURE_STORAGE_ACCOUNT_NAME="mystorageaccount"
export AZURE_STORAGE_ACCOUNT_KEY="your_key_here"
export AZURE_STORAGE_CONTAINER_NAME="m5-forecasting"
```

### Option 3: Using Databricks Secrets

```python
# In Databricks notebook
dbutils.secrets.get(scope="azure-storage", key="account-name")
```

### Upload M5 Data to Blob Storage

```bash
# Using Azure CLI
az storage blob upload-batch \
  --account-name mystorageaccount \
  --destination m5-forecasting \
  --source data/m5/ \
  --pattern "*.csv"
```

### Mount Blob Storage in Databricks

```python
# In Databricks notebook
dbutils.fs.mount(
  source = "wasbs://m5-forecasting@mystorageaccount.blob.core.windows.net",
  mount_point = "/mnt/m5",
  extra_configs = {
    "fs.azure.account.key.mystorageaccount.blob.core.windows.net": 
      dbutils.secrets.get(scope="azure-storage", key="account-key")
  }
)
```

## 📚 Feature Engineering

The pipeline automatically creates:

- **Moving Averages**: 4, 8, 12-month windows
- **Lag Features**: Previous 1-12 months
- **Statistical Features**:
  - Average demand quantity
  - Average demand interval
  - Coefficient of variation
  - Months since last order
  - Last order quantity
- **Time Features**:
  - Month
  - Year
  - Day of week
- **Target Variable**: lead_month_1 (next month's demand)

## 🛠️ Advanced Usage

### Custom Model Configuration

Edit `config.py` to modify model parameters:

```python
SPARK_ML_PARAMS = {
    "RandomForest": {
        "maxDepth": [2, 3, 5, 7],
        "numTrees": [50, 100, 200]
    },
    "GradientBoostedTrees": {
        "maxDepth": [3, 5, 7],
        "maxIter": [10, 20, 50]
    }
}
```

### Running Specific Models Only

Modify the model lists in training scripts:

```python
# In main_train_m5.py
spark_ml_models = [
    {"alias": "RandomForest", "model": rf_model, "param_grid": rf_param_grid}
    # Comment out models you don't want to train
]
```

### Batch Processing Multiple Datasets

```bash
# Create a script to loop through datasets
for dataset in data/custom/*.csv; do
  python upload_custom_data.py --input-path $dataset --output-path data/validated/
  python main_train.py --data-path data/validated/$(basename $dataset)
done
```

## 🐛 Troubleshooting

### Issue: "Cannot find M5 dataset"
**Solution**: Ensure `sales_train_validation.csv` is in `data/m5/` directory

### Issue: "Azure Blob Storage access denied"
**Solution**: Verify credentials in `azure_config.py` or environment variables

### Issue: "OutOfMemoryError in Spark"
**Solution**: Reduce `--max-items` parameter or increase Spark memory:
```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
```

### Issue: "TabPFN not available"
**Solution**: TabPFN is optional. Install with:
```bash
pip install tabpfn
```

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

[Add contact information]

## 🔗 Resources

- [M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [StatsForecast Documentation](https://nixtla.github.io/statsforecast/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Azure Databricks Documentation](https://docs.microsoft.com/en-us/azure/databricks/)

---

**Note**: This project has been refactored from the original Gale Pacific dataset implementation to use the M5 forecasting dataset as the default. All training scripts now support both Spark ML and StatsForecast models in a unified pipeline.
