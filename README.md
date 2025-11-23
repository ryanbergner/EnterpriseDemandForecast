# Demand Forecasting Pipeline

A comprehensive, production-ready time series forecasting system for demand prediction using both statistical and machine learning models. Built with Apache Spark for scalability and MLflow for experiment tracking.

## 🎯 Overview

This project provides a complete end-to-end pipeline for demand forecasting that:

- **Handles Multiple Data Sources**: M5 competition dataset, custom datasets, and Azure Blob Storage
- **Trains Diverse Models**: Spark ML (Linear Regression, Random Forest, GBT), statistical models (Seasonal Exponential Smoothing, Croston), and optional TabPFN
- **Automates Feature Engineering**: Creates 20+ features including moving averages, lags, statistical features, and product categorization
- **Tracks Experiments**: Full MLflow integration for model versioning, comparison, and deployment
- **Supports Production Inference**: Generate predictions with champion/challenger model patterns
- **Validates Data Quality**: Automated data validation and quality checks
- **Scales to Big Data**: Built on Apache Spark for distributed processing

## 📁 Project Structure

```
.
├── main_train_m5.py              # Training script for M5 dataset (local)
├── main_train_databricks.py      # Training script for Azure Databricks
├── main_train.py                 # Generic training script for custom datasets
├── main_inference.py             # Production inference script
├── upload_custom_data.py         # Custom data upload and validation tool
├── cli.py                        # Unified command-line interface
├── config.py                     # Configuration settings
├── azure_config.py.template      # Azure credentials template
├── requirements.txt              # Python dependencies
├── EDA.py                        # Exploratory data analysis notebook
│
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py         # Data aggregation and preprocessing
│   │   └── imputation.py         # Missing value handling
│   │
│   ├── feature_engineering/
│   │   ├── feature_engineering.py  # Core feature engineering
│   │   ├── trend_features.py       # Trend-based features
│   │   ├── ewma_features.py        # Exponential weighted moving averages
│   │   └── interaction_features.py # Feature interactions
│   │
│   ├── model_training/
│   │   ├── ml_models.py          # Spark ML model training
│   │   ├── stats_models.py       # Statistical model training
│   │   ├── ensemble.py           # Model ensembling
│   │   ├── early_stopping.py     # Early stopping utilities
│   │   └── feature_importance.py # Feature importance analysis
│   │
│   ├── inference/
│   │   ├── inference.py          # Prediction generation
│   │   └── confidence_intervals.py # Uncertainty quantification
│   │
│   ├── validation/
│   │   ├── data_quality.py       # Data quality checks
│   │   └── time_series_cv.py     # Time series cross-validation
│   │
│   └── visualization/
│       └── model_dashboard.py    # Model performance dashboards
│
└── data/                         # Data directory (not in git)
    ├── m5/                       # M5 dataset location
    └── custom/                   # Custom datasets
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

**Note**: This project requires Java 8+ for Spark. Install Java if not already installed.

### 2. Download M5 Dataset (Optional)

If you want to use the M5 dataset:

```bash
# Create data directory
mkdir -p data/m5

# Download from Kaggle: https://www.kaggle.com/c/m5-forecasting-accuracy/data
# Place sales_train_validation.csv in data/m5/
```

### 3. Train Models on M5 Dataset

```bash
# Train on M5 dataset (local)
python main_train_m5.py \
  --data-path data/m5/sales_train_validation.csv \
  --experiment-name M5_Experiment \
  --max-items 100

# View results in MLflow UI
mlflow ui
# Open browser to http://localhost:5000
```

### 4. Use Your Own Data

```bash
# Step 1: Upload and validate your data
python upload_custom_data.py \
  --input-path my_sales_data.csv \
  --output-path data/custom/validated_data.csv

# Step 2: Train models
python main_train.py \
  --data-path data/custom/validated_data.csv \
  --experiment-name Custom_Experiment \
  --max-items 50
```

## 📊 Key Features

### Automated Feature Engineering

The pipeline automatically creates comprehensive features:

**Time-based Features:**
- Month, year, day of week
- Months since last order
- Last order quantity

**Lag Features:**
- Previous 1-12 months' quantities
- Target variable: `lead_month_1` (next month's demand)

**Moving Averages:**
- 4, 8, 12-month moving averages
- 3, 5, 7-month moving averages for non-zero sales periods

**Statistical Features:**
- Average demand quantity
- Average demand interval
- Coefficient of variation
- Total orders count

**Product Categorization:**
Automatically categorizes products into:
- **Smooth**: Low variation, regular intervals
- **Erratic**: High variation, regular intervals
- **Intermittent**: Low variation, irregular intervals
- **Lumpy**: High variation, irregular intervals

### Model Training

**Spark ML Models:**
- Linear Regression (baseline)
- Random Forest (with hyperparameter tuning)
- Gradient Boosted Trees (with hyperparameter tuning)

**Statistical Models:**
- Seasonal Exponential Smoothing (handles seasonality)
- Croston Classic (for intermittent demand)

**Optional Models:**
- TabPFN (transformer-based foundation model)

All models are trained with:
- Automatic train/test splitting (80/20 by default)
- Hyperparameter tuning via cross-validation
- MLflow experiment tracking
- Model versioning and registry

### Experiment Tracking

Every training run is tracked in MLflow with:
- Model parameters and hyperparameters
- Evaluation metrics (RMSE, MAE, MAPE, R²)
- Model artifacts
- Training metadata
- Model lineage

View results:
```bash
mlflow ui
```

### Production Inference

Generate predictions using champion/challenger model patterns:

```python
from main_inference import main_inference
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Inference").getOrCreate()
df = spark.read.csv("data.csv", header=True)

predictions = main_inference(
    df=df,
    date_column="OrderDate",
    product_id_column="item_id",
    quantity_column="Quantity",
    month_end_column="MonthEndDate",
    target_path="predictions.parquet",
    ind_full_history=0  # 0 = next month only, 1 = full history
)
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Feature engineering
MA_WINDOWS = [4, 8, 12]           # Moving average windows
LAG_FEATURES = [1, 2, 3, 4, 5, 6, 11, 12]  # Lag periods
MIN_ORDERS = 5                    # Minimum orders required

# Training
TRAIN_SPLIT_RATIO = 0.8           # Train/test split
MAX_DISTINCT_ITEMS = 100          # Max items to train on
RANDOM_SEED = 42                  # Reproducibility

# Model hyperparameters
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
```

## 📖 Usage Examples

### Using the CLI

The project includes a unified CLI for all operations:

```bash
# Train models
python cli.py train \
  --data data/m5_sales.csv \
  --models rf,gbt,stats \
  --categories Smooth,Erratic \
  --experiment-name MyExperiment

# Evaluate a model
python cli.py evaluate \
  --model-path models:/MyModel/1 \
  --data test_data.csv \
  --metrics rmse,mae,r2

# Generate predictions
python cli.py predict \
  --model-path models:/MyModel/1 \
  --data input_data.csv \
  --horizon 12 \
  --output predictions.csv

# Validate data quality
python cli.py validate \
  --data data.csv \
  --report-path quality_report.json

# Compare models
python cli.py compare \
  --experiment MyExperiment \
  --metric rmse \
  --top-k 5
```

### Custom Dataset Format

Your CSV must have these columns:
- **date**: Date column (any standard date format)
- **item_id**: Product/item identifier
- **quantity**: Demand/sales quantity

Optional columns:
- `store_id`, `cat_id`, `dept_id`, `state_id`

Example:
```csv
date,item_id,quantity,store_id
2023-01-01,ITEM_001,150,STORE_A
2023-01-02,ITEM_001,200,STORE_A
2023-01-01,ITEM_002,75,STORE_B
```

### Training on Custom Data

```bash
# 1. Upload and validate
python upload_custom_data.py \
  --input-path my_data.csv \
  --output-path data/custom/validated.csv \
  --date-col OrderDate \
  --item-id-col SKU \
  --quantity-col SalesQty

# 2. Train models
python main_train.py \
  --data-path data/custom/validated.csv \
  --date-column OrderDate \
  --product-id-column SKU \
  --quantity-column SalesQty \
  --experiment-name Custom_Experiment \
  --max-items 200
```

## 🔐 Azure Databricks Setup

### Option 1: Using azure_config.py

1. Copy the template:
```bash
cp azure_config.py.template azure_config.py
```

2. Edit `azure_config.py`:
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

### Running on Databricks

```python
# In Databricks notebook
%run /Workspace/path/to/main_train_databricks

# Or as a job
python main_train_databricks.py \
  --experiment-name M5_Production \
  --max-items 500 \
  --user-email your.email@company.com
```

## 📈 Model Evaluation

All models are evaluated on:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

Metrics are automatically logged to MLflow. View them:
```bash
mlflow ui
# Navigate to your experiment and compare runs
```

## 🛠️ Advanced Features

### Cross-Validation

Use time series cross-validation for robust evaluation:

```python
# Via CLI
python cli.py train \
  --data data.csv \
  --cv-folds 5 \
  --cv-strategy expanding

# Or modify training scripts to use TimeSeriesCV
from src.validation.time_series_cv import TimeSeriesCV
cv = TimeSeriesCV(n_splits=5, strategy='expanding')
```

### Early Stopping

Enable early stopping for tree models:

```bash
python cli.py train \
  --data data.csv \
  --models rf,gbt \
  --early-stopping \
  --patience 5
```

### Feature Selection

Automatically select best features:

```bash
python cli.py train \
  --data data.csv \
  --feature-selection
```

### Ensemble Predictions

Generate ensemble predictions:

```bash
python cli.py predict \
  --model-path models:/Model1/1 \
  --data input.csv \
  --ensemble \
  --ensemble-strategy weighted_average
```

### Confidence Intervals

Add uncertainty quantification to predictions:

```bash
python cli.py evaluate \
  --model-path models:/Model/1 \
  --data test.csv \
  --confidence-intervals
```

## 🐛 Troubleshooting

### Issue: "Cannot find M5 dataset"
**Solution**: Ensure `sales_train_validation.csv` is in `data/m5/` directory, or use `--data-path` to specify location.

### Issue: "OutOfMemoryError in Spark"
**Solution**: Increase Spark memory or reduce dataset size:
```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

Or reduce items:
```bash
python main_train_m5.py --max-items 50
```

### Issue: "Azure Blob Storage access denied"
**Solution**: Verify credentials in `azure_config.py` or environment variables. Check Azure storage account permissions.

### Issue: "TabPFN not available"
**Solution**: TabPFN is optional. Install with:
```bash
pip install tabpfn
```

### Issue: "Column not found" errors
**Solution**: Use `upload_custom_data.py` to validate and map your columns, or specify column names explicitly:
```bash
python main_train.py \
  --date-column YourDateCol \
  --product-id-column YourItemCol \
  --quantity-column YourQtyCol
```

### Issue: "Empty train or test set"
**Solution**: Check your date range. Ensure you have sufficient historical data (at least 5 months per product).

## 📚 API Documentation

For detailed API documentation, see the inline documentation in each module:

- **Preprocessing**: `src/preprocessing/preprocess.py`
- **Feature Engineering**: `src/feature_engineering/feature_engineering.py`
- **Model Training**: `src/model_training/ml_models.py`, `src/model_training/stats_models.py`
- **Inference**: `src/inference/inference.py`
- **Validation**: `src/validation/data_quality.py`

## 🔗 Resources

- [M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [StatsForecast Documentation](https://nixtla.github.io/statsforecast/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Azure Databricks Documentation](https://docs.microsoft.com/en-us/azure/databricks/)

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

[Add contact information]

---

**Note**: This project has been refactored to support both the M5 forecasting dataset and custom datasets. All training scripts support Spark ML and StatsForecast models in a unified pipeline with comprehensive MLflow integration.
