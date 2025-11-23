# Demand Forecasting Platform

An end-to-end forecasting pipeline that automates data preparation, feature engineering, model training, inference, and reporting for retail and supply-chain demand signals. The toolkit targets the M5 dataset out of the box and can be retargeted to any SKU-level sales history.

## Highlights

- Support for both the Kaggle M5 dataset and custom CSV/Parquet datasets through a unified schema.
- Automated feature engineering (lags, moving averages, seasonality indicators, demand pattern labeling, EWMA/trend enrichments).
- Hybrid model stack: Spark ML regressors (Linear Regression, Random Forest, Gradient Boosting) plus StatsForecast statistical models (Seasonal Exponential Smoothing, Croston).
- MLflow-first execution: every training run logs parameters, metrics, and artifacts; champion/challenger inference pulls straight from the Model Registry.
- Unified CLI (`cli.py`) for training, evaluation, prediction, data quality checks, run comparison, and dashboard generation.
- Production-friendly validation utilities, cross-validation helpers, early stopping, and optional ensemble inference.
- Azure Databricks ready: template for credential configuration, blob upload instructions, and notebook/job entry points.

## Repository Layout

```
.  (project root)
├── cli.py                        # Unified command line entry point
├── main_train.py                 # Generic Spark + StatsForecast trainer
├── main_train_m5.py              # Convenience script for the M5 dataset
├── main_train_databricks.py      # Databricks job/notebook entry point
├── main_inference.py             # Champion/challenger batch inference
├── upload_custom_data.py         # Schema validation and reformatting tool
├── src/
│   ├── preprocessing/            # Data aggregation and cleansing
│   ├── feature_engineering/      # Time-series feature generators
│   ├── model_training/           # Spark/StatsForecast training helpers
│   ├── inference/                # Prediction utilities & confidence bands
│   ├── validation/               # Data and CV utilities
│   └── visualization/            # Interactive dashboards
└── docs & guides                 # README variants, improvements, summaries
```

## Prerequisites

- Python 3.10 or newer.
- Java 8+ (required by PySpark). On macOS/Linux ensure `JAVA_HOME` is set.
- pip or another Python package manager.
- Optional: access to an MLflow tracking server/Databricks workspace for registry-backed inference.

## Installation

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure Azure (optional)

```bash
cp azure_config.py.template azure_config.py
# Populate the constants or set env vars:
export AZURE_STORAGE_ACCOUNT_NAME="..."
export AZURE_STORAGE_ACCOUNT_KEY="..."
export AZURE_STORAGE_CONTAINER_NAME="..."
```

## Preparing Data

### M5 Dataset (default path `data/m5/`)

1. Download `sales_train_validation.csv` (and optionally `calendar.csv`, `sell_prices.csv`) from Kaggle.
2. Create the data directory: `mkdir -p data/m5`.
3. Place the CSV files in `data/m5/`.

### Custom Datasets

Minimum required columns:

- `date`: order date (any ISO-friendly format)
- `item_id`: product identifier
- `quantity`: demand/sales units

Optional columns: `store_id`, `cat_id`, `dept_id`, `state_id`, or any numeric drivers.

To validate and map a custom schema:

```bash
python upload_custom_data.py \
  --input-path raw_sales.csv \
  --output-path data/custom/validated.csv \
  --date-col OrderDate \
  --item-id-col SKU \
  --quantity-col Units
```

The helper enforces basic checks (non-empty dates, numeric quantity, monotonic time) and emits a clean CSV ready for the training scripts or CLI.

## Running the Pipeline

### Option 1: Unified CLI

The CLI wraps the most common workflows. Activate your virtual environment, then:

```bash
python cli.py train \
  --data data/m5/sales_train_validation.csv \
  --models rf,gbt,stats \
  --categories all \
  --experiment-name M5_Baseline
```

Available commands:

- `train`: spark/statistical training with optional cross-validation (`--cv-folds`, `--cv-strategy`) and early stopping for tree ensembles.
- `evaluate`: load a saved Spark pipeline (local path or `models:/registry/name`) and compute metrics (`rmse`, `mae`, `mape`, `r2`). The evaluation dataset must expose the label column expected by the model (default `target`).
- `predict`: batch inference with optional ensemble averaging (`--ensemble --ensemble-strategy weighted_average`).
- `validate`: column-level and temporal data quality checks via `DataQualityValidator`.
- `compare`: MLflow experiment leaderboard by metric.
- `dashboard`: build an interactive HTML report from saved results using `ModelDashboard`.

Run `python cli.py --help` or `python cli.py <command> --help` for the full option set.

### Option 2: Python Entry Points

**Local M5 baseline**

```bash
python main_train_m5.py \
  --data-path data/m5/sales_train_validation.csv \
  --experiment-name M5_Local \
  --max-items 100
```

**Generic training for custom data**

```bash
python main_train.py \
  --data-path data/custom/validated.csv \
  --date-column date \
  --product-id-column item_id \
  --quantity-column quantity \
  --experiment-name Custom_Forecast \
  --max-items 250
```

The script will:

1. Create a Spark session.
2. Aggregate daily data to month-end granularity (`aggregate_sales_data`).
3. Generate features (`add_features`) including demand pattern classification.
4. Split on time (80/20).
5. Train StatsForecast (Seasonal Exponential Smoothing, Croston) and Spark ML models (Linear Regression, Random Forest, Gradient Boosted Trees).
6. Log metrics and models under a timestamped MLflow experiment.

**Inference**

```bash
python - <<'PY'
from pyspark.sql import SparkSession
from main_inference import main_inference

spark = SparkSession.builder.appName("ForecastingInference").getOrCreate()
df = spark.read.csv("data/custom/validated.csv", header=True, inferSchema=True)

preds = main_inference(
    df=df,
    date_column="date",
    product_id_column="item_id",
    quantity_column="quantity",
    month_end_column="MonthEndDate",
    target_path="output/predictions",
    ind_full_history=0
)

preds.show(5)
spark.stop()
PY
```

`main_inference.py` pulls champion and challenger model versions for each demand pattern from the MLflow registry (`champion`/`challenger` aliases) and writes predictions to Parquet when `target_path` is supplied.

### Option 3: Databricks

`main_train_databricks.py` mirrors the local training pipeline but reads credentials from `azure_config.py` or environment variables. You can `%run` the script in a Databricks notebook or submit it as a job. Use `upload_custom_data.py` or Azure CLI to stage data in blob storage:

```bash
az storage blob upload-batch \
  --account-name <account> \
  --destination m5-forecasting \
  --source data/m5 \
  --pattern "*.csv"
```

Mount the container inside Databricks with `dbutils.fs.mount(...)` for direct access.

## Experiment Tracking and Registry

- Set the MLflow tracking URI before running scripts if you are not on Databricks: `export MLFLOW_TRACKING_URI=http://localhost:5000`.
- Launch the UI locally with `mlflow ui`.
- `main_train.py` creates a timestamped experiment and nests child runs by model type.
- `main_inference.py` expects models registered under `main.default.<sales_pattern>` with `champion` and `challenger` aliases. Adapt `catalog`, `schema`, and alias names inside the script to match your registry structure.

## Feature Engineering Overview

`src/feature_engineering/feature_engineering.py` derives:

- Recency statistics (`months_since_last_order`, `last_order_quantity`, `total_orders`).
- Lagged targets (`lag_1`-`lag_12`) and forward-looking labels (`lead_month_1`).
- Moving averages (`ma_3`, `ma_4`, `ma_8`, `ma_12`) plus EWMA/trend enrichments when invoked explicitly.
- Demand variability metrics (coefficient of variation, average demand interval).
- Demand pattern classification into Smooth, Erratic, Intermittent, or Lumpy segments used to route models.

Extend the feature set by creating additional modules inside `src/feature_engineering/` and importing them in your training workflow.

## Validation and Cross-Validation

- `DataQualityValidator` (in `src/validation/data_quality.py`) checks for missing periods, duplicate timestamps, negative quantities, and schema mismatches. Invoke it directly or through `cli.py validate`.
- `TimeSeriesCV` (in `src/validation/time_series_cv.py`) implements expanding or sliding window splits; the CLI exposes it via `--cv-folds`.
- `EarlyStopping` (in `src/model_training/early_stopping.py`) provides patience-based stopping for iterative estimators; the CLI activates it with `--early-stopping --patience N`.

## Visualization

`src/visualization/model_dashboard.py` builds Plotly dashboards comparing model performance across segments and metrics. Feed it a serialized JSON produced during evaluation or by aggregating MLflow runs:

```bash
python cli.py dashboard \
  --results results/performance.json \
  --output reports/model_dashboard.html \
  --theme plotly_dark
```

## Configuration

- `config.py` centralizes default column names, feature toggles, train/test splits, and MLflow settings.
- Override settings via environment variables or command-line arguments when running the scripts.
- Sensitive credentials (Azure, Databricks) should be stored in `azure_config.py`, environment variables, or Databricks secrets rather than source control.

## Troubleshooting

- **Spark cannot find Java**: ensure `JAVA_HOME` points to a supported JDK.
- **Memory pressure**: reduce `--max-items`, limit the forecast horizon, or tune Spark executor memory (`spark.driver.memory`).
- **Missing MLflow metrics**: confirm the tracking URI/experiment permissions; local runs default to a new timestamped experiment.
- **Evaluation failures in CLI**: the evaluate command expects the label column to be named `target`; rename columns or extend the command to match your schema.
- **Model registry lookups fail**: adjust `catalog`, `schema`, or alias definitions inside `main_inference.py` to match your registry naming convention.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Add tests or notebooks when introducing new modelling approaches.
4. Submit a pull request summarizing the change and expected impact.

## License

Add project licensing information in this section (e.g., MIT, Apache 2.0).

