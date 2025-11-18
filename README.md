# Comprehensive API Documentation

This document provides comprehensive documentation for all public APIs, functions, and components in the Demand Forecasting System.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Preprocessing Module](#preprocessing-module)
4. [Feature Engineering Module](#feature-engineering-module)
5. [Model Training Module](#model-training-module)
6. [Statistical Models Module](#statistical-models-module)
7. [Inference Module](#inference-module)
8. [Main Scripts](#main-scripts)
9. [Usage Examples](#usage-examples)
10. [Dependencies](#dependencies)

## Overview

The Demand Forecasting System is a comprehensive machine learning pipeline for sales forecasting using multiple modeling approaches:

- **Spark ML Models**: Linear Regression, Random Forest, Gradient Boosted Trees
- **Statistical Models**: ARIMA, Seasonal Exponential Smoothing, Croston's Method
- **TabPFN**: Transformer-based neural network for tabular data
- **Neural Networks**: TensorFlow-based forecasting models

The system processes sales data through preprocessing, feature engineering, model training, and inference stages, with full MLflow integration for experiment tracking and model management.

## Configuration

### `config.py`

Configuration variables used across the system.

```python
# Column mappings
source_path = ""  # Path to data source
date_column = "OrderDate"  # Date column name
product_id_column = "SalesInvoiceProductKey"  # Product identifier column
quantity_column = "Quantity"  # Quantity sold column
month_end_column = "MonthEndDate"  # Month-end date column
```

**Usage:**
```python
from config import date_column, product_id_column, quantity_column, month_end_column
```

## Preprocessing Module

### `src/preprocessing/preprocess.py`

#### `retrieve_sales_data(table_path: str) -> DataFrame`

Retrieves and preprocesses sales data from a Delta table.

**Parameters:**
- `table_path` (str): Path to the Delta table containing sales data

**Returns:**
- `DataFrame`: Preprocessed sales data with applied filters and type conversions

**Features:**
- Filters out invalid product keys (containing '-9999')
- Filters for USD currency only
- Excludes return items
- Filters for positive quantities and non-null prices/costs
- Converts DecimalType columns to DoubleType for ML compatibility

**Example:**
```python
from src.preprocessing.preprocess import retrieve_sales_data

# Load data from Delta table
df = retrieve_sales_data("/path/to/sales/delta/table")
print(f"Loaded {df.count()} sales records")
```

#### `aggregate_sales_data(df: DataFrame, date_column: str, product_id_column: str, quantity_column: str, month_end_column: str = "MonthEndDate") -> DataFrame`

Aggregates daily sales data to monthly level and computes product-level statistics.

**Parameters:**
- `df` (DataFrame): Input sales data
- `date_column` (str): Name of the date column
- `product_id_column` (str): Name of the product ID column
- `quantity_column` (str): Name of the quantity column
- `month_end_column` (str): Name for the month-end date column (default: "MonthEndDate")

**Returns:**
- `DataFrame`: Aggregated data with columns:
  - `product_id_column`: Product identifier
  - `month_end_column`: Month-end date
  - `quantity_column`: Summed quantity for the month
  - `min_date`: Earliest month for the product
  - `max_date`: Latest month for the product

**Example:**
```python
from src.preprocessing.preprocess import aggregate_sales_data

# Aggregate daily sales to monthly
df_agg = aggregate_sales_data(
    df=raw_sales_df,
    date_column="OrderDate",
    product_id_column="ProductID",
    quantity_column="Quantity"
)
```

## Feature Engineering Module

### `src/feature_engineering/feature_engineering.py`

#### `add_features(df_agg: DataFrame, month_end_column: str = "MonthEndDate", product_id_column: str = "ItemNumber", quantity_column: str = "DemandQuantity") -> DataFrame`

Comprehensive feature engineering function that creates time-series features for demand forecasting.

**Parameters:**
- `df_agg` (DataFrame): Aggregated sales data from `aggregate_sales_data()`
- `month_end_column` (str): Month-end date column name
- `product_id_column` (str): Product identifier column name
- `quantity_column` (str): Quantity column name

**Returns:**
- `DataFrame`: Enhanced dataset with engineered features

**Generated Features:**

1. **Time-based Features:**
   - `month`: Month as float (1-12)
   - `year`: Year as float
   - `months_since_last_order`: Months since last non-zero order
   - `last_order_quantity`: Quantity of last non-zero order

2. **Lag Features:**
   - `lag_1` to `lag_5`: Previous 1-5 months' quantities
   - `lead_month_1`: Next month's quantity (target variable)

3. **Statistical Features:**
   - `total_orders`: Count of months with non-zero orders
   - `cov_quantity`: Coefficient of variation
   - `avg_demand_interval`: Average interval between orders
   - `product_category`: Categorization based on demand patterns:
     - "Smooth": Low variation, regular intervals
     - "Erratic": High variation, regular intervals
     - "Intermittent": Low variation, irregular intervals
     - "Lumpy": High variation, irregular intervals

4. **Moving Averages:**
   - `ma_4_month`, `ma_8_month`, `ma_12_month`: Moving averages over 4, 8, 12 months
   - `ma_3_sales`, `ma_5_sales`, `ma_7_sales`: Moving averages for non-zero sales periods

5. **Demand Buckets:**
   - `bucket`: B1-B10 buckets based on average demand quantity deciles

**Example:**
```python
from src.feature_engineering.feature_engineering import add_features

# Add comprehensive features
df_features = add_features(
    df_agg=monthly_aggregated_df,
    month_end_column="MonthEndDate",
    product_id_column="ProductID",
    quantity_column="DemandQuantity"
)

# Check product categories
df_features.groupBy("product_category").count().show()
```

## Model Training Module

### `src/model_training/ml_models.py`

#### `train_sparkML_models(model, train_df: DataFrame, featuresCols: list, labelCol: str, paramGrid=None) -> PipelineModel`

Trains Spark ML models with automatic feature vectorization and optional hyperparameter tuning.

**Parameters:**
- `model`: Spark ML model instance (e.g., `LinearRegression()`, `RandomForestRegressor()`)
- `train_df` (DataFrame): Training data
- `featuresCols` (list): List of feature column names
- `labelCol` (str): Target variable column name
- `paramGrid` (list, optional): Parameter grid for cross-validation

**Returns:**
- `PipelineModel`: Trained pipeline model

**Supported Models:**
- `LinearRegression`
- `RandomForestRegressor`
- `GBTRegressor`

**Example:**
```python
from src.model_training.ml_models import train_sparkML_models
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder

# Define model
rf_model = RandomForestRegressor(labelCol="lead_month_1")

# Define parameter grid
param_grid = (ParamGridBuilder()
    .addGrid(rf_model.maxDepth, [2, 5])
    .addGrid(rf_model.numTrees, [10, 50])
    .build())

# Train model
pipeline_model = train_sparkML_models(
    model=rf_model,
    train_df=training_data,
    featuresCols=["DemandQuantity", "month", "year", "lag_1"],
    labelCol="lead_month_1",
    paramGrid=param_grid
)
```

#### `evaluate_sparkML_models(model: PipelineModel, test_df: DataFrame, features_cols: List[str], label_col: str, requirements_path: Optional[str] = None, model_alias: Optional[str] = None) -> dict`

Evaluates Spark ML models and logs metrics to MLflow.

**Parameters:**
- `model` (PipelineModel): Trained pipeline model
- `test_df` (DataFrame): Test data
- `features_cols` (List[str]): Feature column names
- `label_col` (str): Target variable column name
- `requirements_path` (str, optional): Path to requirements.txt for model logging
- `model_alias` (str, optional): Model alias for MLflow logging

**Returns:**
- `dict`: Dictionary containing evaluation metrics:
  - `rmse`: Root Mean Square Error
  - `r2`: R-squared score
  - `mae`: Mean Absolute Error
  - `mape`: Mean Absolute Percentage Error

**Example:**
```python
from src.model_training.ml_models import evaluate_sparkML_models

# Evaluate model
metrics = evaluate_sparkML_models(
    model=trained_pipeline,
    test_df=test_data,
    features_cols=["DemandQuantity", "month", "year"],
    label_col="lead_month_1",
    model_alias="RandomForest_v1"
)

print(f"RMSE: {metrics['rmse']:.3f}")
print(f"R²: {metrics['r2']:.3f}")
```

## Statistical Models Module

### `src/model_training/stats_models.py`

#### `train_stats_models(models: List[Any], train_df: DataFrame, month_end_column: str, product_id_column: str, target_column: str) -> StatsForecast`

Trains statistical forecasting models using the StatsForecast library.

**Parameters:**
- `models` (List[Any]): List of StatsForecast model instances
- `train_df` (DataFrame): Training data
- `month_end_column` (str): Date column name
- `product_id_column` (str): Product identifier column name
- `target_column` (str): Target variable column name

**Returns:**
- `StatsForecast`: Trained StatsForecast instance

**Supported Models:**
- `AutoARIMA`
- `HoltWinters`
- `ARIMA`
- `CrostonClassic`
- `SeasonalExponentialSmoothingOptimized`

**Example:**
```python
from src.model_training.stats_models import train_stats_models
from statsforecast.models import AutoARIMA, CrostonClassic

# Define models
models = [
    AutoARIMA(),
    CrostonClassic()
]

# Train models
stats_forecast = train_stats_models(
    models=models,
    train_df=training_data,
    month_end_column="MonthEndDate",
    product_id_column="ProductID",
    target_column="DemandQuantity"
)
```

#### `evaluate_stats_models(stats_model, test_df: DataFrame, month_end_column: str, product_id_column: str, target_column: str, experiment_id: str, artifact_location: str, model_name: str) -> None`

Evaluates statistical forecasting models and logs metrics to MLflow.

**Parameters:**
- `stats_model`: Trained StatsForecast instance
- `test_df` (DataFrame): Test data
- `month_end_column` (str): Date column name
- `product_id_column` (str): Product identifier column name
- `target_column` (str): Target variable column name
- `experiment_id` (str): MLflow experiment ID
- `artifact_location` (str): Artifact storage location
- `model_name` (str): Model name for logging

**Example:**
```python
from src.model_training.stats_models import evaluate_stats_models

# Evaluate statistical models
evaluate_stats_models(
    stats_model=trained_stats_forecast,
    test_df=test_data,
    month_end_column="MonthEndDate",
    product_id_column="ProductID",
    target_column="DemandQuantity",
    experiment_id="exp_123",
    artifact_location="/path/to/artifacts",
    model_name="AutoARIMA"
)
```

#### `return_sf_model(model_name: str, statsforecast_instance: StatsForecast) -> Optional[Any]`

Retrieves a specific sub-model from a StatsForecast instance by alias.

**Parameters:**
- `model_name` (str): Model alias to retrieve
- `statsforecast_instance` (StatsForecast): StatsForecast instance

**Returns:**
- `Optional[Any]`: The requested sub-model or None if not found

## Inference Module

### `src/inference/inference.py`

#### `generate_predictions(model_uri, model_name, sales_pattern, df_inference, month_end_column, product_id_column) -> DataFrame`

Generates predictions using trained models (both Spark ML and StatsForecast).

**Parameters:**
- `model_uri`: MLflow model URI
- `model_name` (str): Model name/alias
- `sales_pattern` (str): Product category pattern
- `df_inference` (DataFrame): Inference data
- `month_end_column` (str): Date column name
- `product_id_column` (str): Product identifier column name

**Returns:**
- `DataFrame`: Predictions with columns:
  - `product_id_column`: Product identifier
  - `month_end_column`: Prediction date
  - `prediction`: Forecasted quantity
  - `SalesPattern`: Product category
  - `Model`: Model name

**Example:**
```python
from src.inference.inference import generate_predictions

# Generate predictions
predictions = generate_predictions(
    model_uri="models:/my_model/1",
    model_name="RandomForest",
    sales_pattern="Smooth",
    df_inference=inference_data,
    month_end_column="MonthEndDate",
    product_id_column="ProductID"
)
```

## Main Scripts

### `main_train.py`

Comprehensive training script that trains multiple model types across product categories.

**Features:**
- Trains both Spark ML and StatsForecast models
- Creates separate MLflow experiments for each product category
- Supports hyperparameter tuning
- Handles data filtering and validation

**Usage:**
```bash
python main_train.py
```

### `main_train_spark.py`

Focused training script for Spark ML and StatsForecast models.

**Usage:**
```bash
python main_train_spark.py
```

### `main_train_sklearn.py`

Training script for TabPFN and other scikit-learn models.

**Features:**
- Enforces 10k row limit for TabPFN
- Automatic train/test splitting
- MLflow integration for scikit-learn models

**Usage:**
```bash
python main_train_sklearn.py
```

### `main_inference.py`

Inference script for generating predictions using champion and challenger models.

**Features:**
- Supports both full history and next-month predictions
- Handles multiple product categories
- MLflow model registry integration
- Optional output to Parquet files

**Usage:**
```python
from main_inference import main_inference

# Generate next-month predictions
predictions = main_inference(
    df=sales_data,
    date_column="OrderDate",
    product_id_column="ProductID",
    quantity_column="Quantity",
    month_end_column="MonthEndDate",
    target_path="/path/to/output",
    ind_full_history=0
)
```

## Usage Examples

### Complete Training Pipeline

```python
from pyspark.sql import SparkSession
from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import train_sparkML_models, evaluate_sparkML_models
from pyspark.ml.regression import RandomForestRegressor

# Initialize Spark
spark = SparkSession.builder.appName("DemandForecasting").getOrCreate()

# Load and preprocess data
df = spark.read.csv("sales_data.csv", header=True, inferSchema=True)
df_agg = aggregate_sales_data(df, "OrderDate", "ProductID", "Quantity")
df_features = add_features(df_agg, "MonthEndDate", "ProductID", "Quantity")

# Filter for specific category
df_smooth = df_features.filter(df_features.product_category == "Smooth")

# Define features and target
feature_cols = ["DemandQuantity", "month", "year", "lag_1", "lag_2"]
target_col = "lead_month_1"

# Train model
rf_model = RandomForestRegressor(labelCol=target_col)
pipeline_model = train_sparkML_models(
    model=rf_model,
    train_df=df_smooth.select(*feature_cols, target_col),
    featuresCols=feature_cols,
    labelCol=target_col
)

# Evaluate model
metrics = evaluate_sparkML_models(
    model=pipeline_model,
    test_df=df_smooth.select(*feature_cols, target_col),
    features_cols=feature_cols,
    label_col=target_col,
    model_alias="RandomForest_Smooth"
)

print(f"Model Performance: {metrics}")
```

### Statistical Model Training

```python
from statsforecast.models import AutoARIMA, CrostonClassic
from src.model_training.stats_models import train_stats_models, evaluate_stats_models

# Define statistical models
models = [
    AutoARIMA(),
    CrostonClassic()
]

# Train models
stats_forecast = train_stats_models(
    models=models,
    train_df=training_data,
    month_end_column="MonthEndDate",
    product_id_column="ProductID",
    target_column="DemandQuantity"
)

# Evaluate models
evaluate_stats_models(
    stats_model=stats_forecast,
    test_df=test_data,
    month_end_column="MonthEndDate",
    product_id_column="ProductID",
    target_column="DemandQuantity",
    experiment_id="stats_experiment",
    artifact_location="/artifacts",
    model_name="AutoARIMA"
)
```

### TabPFN Training

```python
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

# Prepare data (convert to Pandas)
df_pandas = df_features.toPandas()
X = df_pandas[feature_cols].values
y = df_pandas[target_col].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train TabPFN
with mlflow.start_run():
    model = TabPFNRegressor(device="auto")
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "TabPFN_model")
```

## Dependencies

### Core Dependencies

```txt
pyspark>=3.5.4          # Apache Spark for big data processing
numpy>=1.21             # Numerical computing
pandas>=2.2.2           # Data manipulation
scikit-learn>=1.5.2     # Machine learning algorithms
scipy>=1.15.1           # Scientific computing
torch>=2.0.0            # PyTorch for TabPFN
mlflow>=2.15.1          # ML lifecycle management
statsforecast>=1.7.0    # Statistical forecasting models
```

### Visualization Dependencies

```txt
seaborn>=0.12.2         # Statistical visualization
matplotlib>=3.4.3       # Plotting library
plotly>=5.3.1           # Interactive visualizations
```

### Development Dependencies

```txt
jupyter>=1.0.0          # Jupyter notebooks
jupyterlab>=3.2.0       # JupyterLab interface
```

### Optional Dependencies

```txt
tabpfn>=2.0.1           # TabPFN transformer model
mlflavors>=1.0.0        # MLflow model flavors
```

## Best Practices

1. **Data Quality**: Always validate input data and handle missing values appropriately
2. **Feature Engineering**: Use domain knowledge to create meaningful features
3. **Model Selection**: Test multiple models and select based on business metrics
4. **Experiment Tracking**: Use MLflow for comprehensive experiment management
5. **Model Validation**: Implement proper train/validation/test splits
6. **Performance Monitoring**: Track model performance over time
7. **Documentation**: Maintain clear documentation for model decisions and parameters

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use appropriate Spark configurations for large datasets
2. **Feature Alignment**: Ensure feature columns match between training and inference
3. **Model Compatibility**: Verify model versions and dependencies
4. **Data Types**: Check column types match expected formats
5. **MLflow Connectivity**: Ensure proper MLflow server configuration

### Performance Tips

1. **Caching**: Cache frequently used DataFrames
2. **Partitioning**: Use appropriate data partitioning strategies
3. **Resource Allocation**: Allocate sufficient memory and CPU resources
4. **Batch Processing**: Process data in appropriate batch sizes
5. **Model Optimization**: Use appropriate hyperparameter tuning strategies

