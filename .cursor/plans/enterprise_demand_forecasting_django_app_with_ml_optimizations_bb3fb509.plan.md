---
name: Enterprise Demand Forecasting Django App with ML Optimizations
overview: Transform the existing PySpark-based demand forecasting system into a production-ready Django web application with research-backed ML improvements, PySpark optimizations for batch training, and performance enhancements including Mojo language integration options.
todos:
  - id: pyspark-optimizations
    content: Replace math package functions with PySpark equivalents across all modules (sqrt, pow, log, exp, mean, std)
    status: pending
  - id: batch-training-optimization
    content: "Implement batch training optimizations: DataFrame caching, broadcast variables, partitioning strategy, vectorized operations"
    status: pending
  - id: multiscale-features
    content: Implement multiscale feature engineering (SCINet-inspired) with multiple temporal resolutions
    status: pending
  - id: advanced-lag-features
    content: Add extended lag features (lag_1 to lag_24) with automatic lag selection based on research
    status: pending
  - id: seasonal-decomposition
    content: Implement STL seasonal decomposition features (trend, seasonal, residual components)
    status: pending
  - id: probabilistic-forecasting
    content: Add quantile regression models for uncertainty quantification
    status: pending
  - id: ensemble-stacking
    content: Implement meta-learner stacking for ensemble models
    status: pending
  - id: tabpfn-v2-upgrade
    content: Upgrade to TabPFN-v2 for improved zero-shot performance
    status: pending
  - id: django-project-setup
    content: "Create Django project structure with apps: api, forecasting, ml_pipeline"
    status: pending
  - id: django-models
    content: "Implement Django ORM models: ForecastJob, Product, ForecastResult, ModelVersion, FeatureConfig"
    status: pending
  - id: rest-api-endpoints
    content: Create REST API endpoints for training, prediction, model management using Django REST Framework
    status: pending
  - id: celery-integration
    content: Set up Celery for async model training and batch prediction tasks
    status: pending
  - id: mlflow-django-integration
    content: Integrate MLflow model registry with Django models and API
    status: pending
  - id: spark-session-management
    content: Implement singleton SparkSession management with connection pooling and graceful shutdown
    status: pending
  - id: validation-framework
    content: Enhance time series cross-validation with concept drift detection and backtesting
    status: pending
  - id: mojo-poc
    content: Create proof-of-concept Mojo implementation for critical feature engineering functions
    status: pending
  - id: local-data-ingestion
    content: Create enhanced main_train_local.py with M5 dataset support and data folder ingestion
    status: pending
  - id: monthly-predictions
    content: Implement monthly prediction generation utilities and ensure all outputs are monthly-level
    status: pending
  - id: documentation-consolidation
    content: Consolidate all markdown files into comprehensive README.md and remove unnecessary files
    status: pending
---

# Enterprise Demand Forecasting Django Application - Implementation Plan

## Research Summary

Based on research from arXiv papers, Brave search, and Perplexity, key findings for enterprise demand forecasting include:

1. **Foundation Models**: Lag-Llama and TabPFN-v2 show strong zero-shot generalization for time series
2. **Feature Engineering**: Multiscale decomposition, lag features, and hierarchical features are critical
3. **Model Architecture**: SCINet (sample convolution and interaction) and LLM-Mixer show promise for complex temporal patterns
4. **Validation**: Time series cross-validation with expanding/sliding windows is essential
5. **Performance**: Distributed batch training with PySpark optimizations significantly reduces training time

## Phase 1: Model & Feature Engineering Improvements

### 1.1 Advanced Feature Engineering (Research-Based)

**Location**: `src/feature_engineering/`

**Improvements**:

- **Multiscale Features**: Implement SCINet-inspired downsampling features at multiple resolutions (daily, weekly, monthly aggregations)
- **Hierarchical Features**: Add product hierarchy features (category, subcategory, brand) for hierarchical forecasting
- **Lag-Llama Inspired Features**: Extended lag features (lag_1 to lag_24) with automatic lag selection
- **Seasonal Decomposition**: STL decomposition features (trend, seasonal, residual components)
- **Fourier Features**: Cyclical encoding of time features using Fourier transforms
- **Interaction Features**: Cross-product features between lag values and moving averages
- **Regime Detection**: Volatility regime features (low/medium/high volatility periods)

**Files to Modify**:

- `src/feature_engineering/feature_engineering.py` - Add new feature functions
- `src/feature_engineering/trend_features.py` - Enhance trend detection
- `src/feature_engineering/interaction_features.py` - Add cross-feature interactions
- Create `src/feature_engineering/multiscale_features.py` - New file for multiscale features
- Create `src/feature_engineering/seasonal_decomposition.py` - New file for STL decomposition

### 1.2 Model Architecture Enhancements

**Location**: `src/model_training/`

**Improvements**:

- **Ensemble Methods**: Implement stacking with meta-learner (currently basic ensemble exists)
- **TabPFN-v2 Integration**: Upgrade to TabPFN-v2 for better zero-shot performance
- **Probabilistic Forecasting**: Add quantile regression models for uncertainty quantification
- **Hierarchical Coherence**: Implement Deep Poisson Mixture Network (DPMN) for hierarchical forecasting
- **AutoML Integration**: Add Optuna for automated hyperparameter optimization

**Files to Modify**:

- `src/model_training/ml_models.py` - Add ensemble stacking
- `src/model_training/ensemble.py` - Enhance with meta-learning
- Create `src/model_training/probabilistic_models.py` - Quantile regression
- Create `src/model_training/hierarchical_models.py` - DPMN implementation

### 1.3 Validation Framework

**Location**: `src/validation/`

**Improvements**:

- **Walk-Forward Validation**: Enhance existing walk-forward validator with refit strategies
- **Backtesting Framework**: Add comprehensive backtesting with multiple metrics
- **Concept Drift Detection**: Implement drift detection for model retraining triggers
- **Cross-Validation Metrics**: Add time-aware metrics (MASE, sMAPE, CRPS for probabilistic)

**Files to Modify**:

- `src/validation/time_series_cv.py` - Enhance existing CV framework
- Create `src/validation/drift_detection.py` - Concept drift detection
- Create `src/validation/backtesting.py` - Backtesting framework

## Phase 2: PySpark Performance Optimizations

### 2.1 Replace Math Package with PySpark Functions

**Location**: All feature engineering and model training files

**Key Replacements**:

- `math.sqrt()` → `pyspark.sql.functions.sqrt()`
- `math.pow()` → `pyspark.sql.functions.pow()`
- `math.log()` → `pyspark.sql.functions.log()`
- `math.exp()` → `pyspark.sql.functions.exp()`
- `numpy.sqrt()` → `pyspark.sql.functions.sqrt()` (in UDFs)
- `numpy.mean()` → `pyspark.sql.functions.avg()`
- `numpy.std()` → `pyspark.sql.functions.stddev_samp()`

**Files to Modify**:

- `src/preprocessing/preprocess.py` - Line 38: Replace `from math import sqrt`
- `src/feature_engineering/feature_engineering.py` - Replace all math operations
- `src/feature_engineering/ewma_features.py` - Replace math operations
- `src/model_training/ml_models.py` - Replace numpy operations in evaluation
- `src/model_training/stats_models.py` - Replace numpy.sqrt in line 204

### 2.2 Batch Training Optimizations

**Location**: `src/model_training/`

**Improvements**:

- **Broadcast Variables**: Cache feature metadata and lookup tables
- **DataFrame Caching**: Strategic caching of intermediate DataFrames
- **Partitioning Strategy**: Optimize data partitioning by product_id for window functions
- **Vectorized Operations**: Use PySpark UDFs with pandas_udf for vectorized computations
- **Broadcast Joins**: Force broadcast joins for small lookup tables
- **Column Pruning**: Early selection of required columns to reduce data shuffling

**Files to Modify**:

- `main_train.py` - Add caching and partitioning strategies
- `src/model_training/ml_models.py` - Optimize training pipeline
- Create `src/model_training/batch_optimizer.py` - Batch training utilities

### 2.3 Spark Configuration Tuning

**Location**: Configuration files

**Improvements**:

- **Dynamic Allocation**: Enable dynamic executor allocation
- **Memory Management**: Optimize executor and driver memory
- **Shuffle Optimization**: Configure optimal shuffle partitions
- **Broadcast Threshold**: Adjust broadcast join threshold for feature metadata

**Files to Create**:

- `spark_config.py` - Spark configuration presets for different cluster sizes

## Phase 3: Additional Performance Improvements

### 3.1 Data Processing Optimizations

- **Delta Lake Integration**: Use Delta Lake for ACID transactions and time travel
- **Incremental Processing**: Implement incremental feature computation
- **Feature Store**: Create feature store for reusable features (Feast or custom)
- **Data Quality Checks**: Add automated data quality validation before training

### 3.2 Model Serving Optimizations

- **Model Caching**: Cache loaded models in memory
- **Batch Inference**: Process predictions in batches
- **Async Processing**: Use Celery for async model training jobs
- **Model Versioning**: Enhanced MLflow model registry integration

### 3.3 Mojo Programming Language Integration

**Options for Mojo Integration**:

1. **Critical Path Functions**: Port computationally intensive feature engineering functions to Mojo

   - EWMA calculations
   - Moving average computations
   - Statistical aggregations
   - Use Mojo's Python compatibility to call from PySpark UDFs

2. **Model Inference**: Create Mojo-based inference server for low-latency predictions

   - Port model inference logic to Mojo
   - Use Mojo's performance (up to 35,000x faster than Python for some operations)
   - REST API wrapper in Django to call Mojo inference service

3. **Hybrid Approach**: 

   - Keep PySpark for distributed training
   - Use Mojo for single-product real-time inference
   - Django API routes requests to appropriate backend (Mojo for single, PySpark for batch)

**Implementation Strategy**:

- Start with feature engineering functions (highest ROI)
- Create Mojo modules for: `ewma_calculations.mojo`, `moving_averages.mojo`
- Use Mojo's `@python_callable` decorator for PySpark UDF integration
- Benchmark performance improvements

## Phase 4: Local Data Ingestion & Training Script

### 4.1 Enhanced Local Training Script

**Location**: `main_train_local.py` (new, consolidates `main_train_m5_local.py`)

**Features**:

- **Flexible Data Ingestion**: Support multiple data formats (CSV, Parquet, M5 format)
- **M5 Dataset Support**: Automatic detection and transformation of M5 format (unpivot day columns)
- **Local Data Folder**: Read from `data/` folder with automatic format detection
- **Monthly Aggregation**: Automatically aggregate daily data to monthly for forecasting
- **Monthly Predictions**: Generate monthly-level predictions (not daily)
- **Data Validation**: Schema validation before training
- **Error Handling**: Comprehensive error handling and logging

**M5 Dataset Format Handling**:

- Detects M5 format (columns: `item_id`, `d_1`, `d_2`, ..., `d_1913`)
- Unpivots day columns to time series format
- Converts to standard schema: `item_id`, `OrderDate`, `Quantity`
- Aggregates to monthly level automatically

**Files to Create/Modify**:

- Create `main_train_local.py` - Enhanced local training script (replaces/enhances `main_train_m5_local.py`)
- Update `main_train.py` - Accept DataFrame input (not just file path) for flexibility
- Create `src/data_ingestion/local_data_loader.py` - Data loading utilities with format detection
- Create `src/data_ingestion/m5_transformer.py` - M5 format transformation (unpivot day columns)

**M5 Dataset Transformation Logic**:

```python
# Detect M5 format: columns like item_id, d_1, d_2, ..., d_1913
# Unpivot to: item_id, OrderDate, Quantity
# Aggregate daily → monthly automatically
# Generate monthly predictions
```

**Usage Examples**:

```bash
# M5 dataset from data folder (auto-detect format)
python main_train_local.py --data-folder data/m5

# Explicit M5 format
python main_train_local.py --data-folder data/m5 --format m5

# Standard CSV/Parquet from folder
python main_train_local.py --data-folder data/sales --format csv

# Single file
python main_train_local.py --data-path data/sales.csv

# With custom output
python main_train_local.py --data-folder data/m5 --output-dir models/ --experiment-prefix /M5_Local
```

**Integration with Django**:

- Django API endpoint can trigger local training
- Celery task wraps `main_train_local.py` for async execution
- Training results stored in Django database and MLflow

### 4.2 Monthly Prediction Generation

**Location**: `src/inference/monthly_predictions.py` (new)

**Features**:

- **Monthly Aggregation**: Ensure predictions are at monthly level
- **Forecast Horizon**: Configurable months ahead (1, 3, 6, 12)
- **Batch Processing**: Process multiple products efficiently
- **Output Format**: Standardized monthly prediction format

**Files to Create**:

- `src/inference/monthly_predictions.py` - Monthly prediction utilities
- Update `main_inference.py` - Add monthly prediction wrapper
- Update `main_train_local.py` - Ensure training outputs monthly-level models

**Monthly Prediction Features**:

- Automatic aggregation of daily predictions to monthly
- Support for multi-month forecast horizons (1, 3, 6, 12 months)
- Monthly-level confidence intervals
- Output format: `product_id`, `MonthEndDate`, `prediction`, `lower_bound`, `upper_bound`

## Phase 5: Django Application Architecture

### 5.1 Django Project Structure

```
demand_forecast_app/
├── manage.py
├── demand_forecast/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── api/
│   ├── views.py          # REST API endpoints
│   ├── serializers.py   # DRF serializers
│   ├── urls.py
│   └── permissions.py
├── forecasting/
│   ├── models.py        # Django ORM models
│   ├── tasks.py         # Celery tasks for async jobs
│   ├── services.py       # Business logic
│   └── admin.py
├── ml_pipeline/
│   ├── spark_service.py # PySpark session management
│   ├── training_service.py
│   ├── inference_service.py
│   └── model_registry.py # MLflow integration
├── frontend/
│   └── static/          # React/Vue frontend (optional)
└── tests/
```

### 5.2 Core Django Components

**Models** (`forecasting/models.py`):

- `ForecastJob`: Track training/inference jobs
- `Product`: Product metadata
- `ForecastResult`: Store predictions
- `ModelVersion`: Track model versions from MLflow
- `FeatureConfig`: Store feature engineering configurations

**API Endpoints** (`api/views.py`):

- `POST /api/v1/train/` - Trigger model training
- `GET /api/v1/train/{job_id}/` - Get training status
- `POST /api/v1/predict/` - Generate predictions
- `GET /api/v1/models/` - List available models
- `GET /api/v1/features/` - Get feature importance
- `GET /api/v1/metrics/` - Get model performance metrics

**Services** (`forecasting/services.py`):

- `TrainingService`: Orchestrate model training
- `InferenceService`: Handle prediction requests
- `FeatureEngineeringService`: Manage feature computation
- `ValidationService`: Run cross-validation

**Celery Tasks** (`forecasting/tasks.py`):

- `train_model_async`: Async model training
- `generate_predictions_async`: Batch prediction generation
- `validate_model_async`: Model validation jobs

### 5.3 Integration Points

**MLflow Integration**:

- Django models sync with MLflow model registry
- Automatic model versioning on training completion
- Model metadata stored in Django database

**PySpark Integration**:

- Singleton SparkSession managed by Django
- Connection pooling for Spark sessions
- Graceful shutdown on Django app termination

**Database**:

- PostgreSQL for Django ORM
- Delta Lake for time series data storage
- Redis for Celery task queue and caching

### 5.4 API Design

**REST API** (Django REST Framework):

```python
# Training
POST /api/v1/train/
{
  "product_categories": ["Smooth", "Erratic"],
  "model_types": ["RandomForest", "GBT"],
  "hyperparameter_tuning": true
}

# Prediction
POST /api/v1/predict/
{
  "product_ids": ["P001", "P002"],
  "forecast_horizon": 3,
  "model_version": "latest"
}

# Model Management
GET /api/v1/models/
GET /api/v1/models/{model_id}/metrics/
```

## Phase 6: Documentation Consolidation

### 6.1 Consolidate All Documentation into README.md

**Location**: `README.md`

**Action**: Merge all relevant information from:

- `IMPROVEMENTS.md` - Feature documentation
- `MIGRATION_SUMMARY.md` - Migration notes (keep key points)
- `NEW_FEATURES_SUMMARY.md` - Feature summary (merge into README)
- `PR_DESCRIPTION.md` - PR info (remove, not needed in README)
- `CREATE_PR_INSTRUCTIONS.md` - Remove (temporary instructions)

**README.md Structure**:

1. **Overview** - Project description, capabilities, and architecture
2. **Quick Start** - Installation, basic usage, and example workflows
3. **Data Ingestion** - How to load data:

   - Local data folder structure
   - M5 dataset format and transformation
   - Custom data formats (CSV, Parquet)
   - Data validation and schema requirements

4. **Training** - Training models:

   - Local training with `main_train_local.py`
   - Databricks training with `main_train_m5_databricks.py`
   - Training options and configurations
   - MLflow experiment tracking

5. **Inference** - Generating predictions:

   - Monthly prediction generation
   - Batch vs single predictions
   - Confidence intervals

6. **Feature Engineering** - Available features:

   - Basic features (lags, moving averages)
   - Advanced features (trend, EWMA, interactions)
   - Feature importance analysis

7. **Model Training** - Model types and training:

   - Spark ML models (LR, RF, GBT)
   - Statistical models (ARIMA, Exponential Smoothing)
   - TabPFN transformer model
   - Ensemble methods

8. **Validation** - Data quality and cross-validation:

   - Data quality validators
   - Time series cross-validation
   - Walk-forward validation

9. **Django Application** - Web application:

   - Setup and installation
   - API endpoints
   - Async job processing
   - Model management

10. **Performance Optimizations**:

    - PySpark optimizations (math → PySpark functions)
    - Batch training strategies
    - Mojo language integration options

11. **API Reference** - Complete API documentation
12. **Examples** - Usage examples and tutorials
13. **Troubleshooting** - Common issues and solutions
14. **Contributing** - Development guidelines

**Files to Remove**:

- `CREATE_PR_INSTRUCTIONS.md` - Temporary PR instructions (not needed in repo)
- `IMPROVEMENTS.md` - Content merged into README (all 12 improvements documented)
- `MIGRATION_SUMMARY.md` - Key points merged into README (M5 migration notes)
- `NEW_FEATURES_SUMMARY.md` - Content merged into README (feature summary)
- `PR_DESCRIPTION.md` - PR-specific, not needed in repo

**Files to Keep**:

- `README.md` - Single source of truth with all documentation
- Any technical architecture diagrams (if exist)
- `requirements.txt` - Dependencies (essential)
- `config.py` - Configuration (essential)

**Consolidation Strategy**:

1. Start with existing README.md structure
2. Add sections from IMPROVEMENTS.md (all 12 improvements)
3. Add M5 dataset information from MIGRATION_SUMMARY.md
4. Add feature summaries from NEW_FEATURES_SUMMARY.md
5. Ensure all code examples and usage patterns are included
6. Add local data ingestion examples
7. Add Django application setup instructions
8. Remove redundant information
9. Ensure single, comprehensive source of truth

## Phase 7: Implementation Steps

### Step 1: PySpark Optimizations (Week 1-2)

1. Replace all `math` package imports with PySpark functions
2. Implement batch training optimizations
3. Add Spark configuration tuning
4. Benchmark performance improvements

### Step 2: Feature Engineering Enhancements (Week 3-4)

1. Implement multiscale features
2. Add seasonal decomposition
3. Enhance lag features
4. Add interaction features
5. Validate feature importance

### Step 3: Model Improvements (Week 5-6)

1. Upgrade to TabPFN-v2
2. Implement probabilistic forecasting
3. Add ensemble stacking
4. Integrate Optuna for hyperparameter tuning

### Step 4: Local Data Ingestion (Week 7)

1. Create enhanced `main_train_local.py` with M5 support
2. Implement data folder scanning and format detection
3. Add M5 transformation utilities
4. Test with M5 dataset from `data/` folder
5. Ensure monthly prediction generation works

### Step 5: Documentation Consolidation (Week 7)

1. Read all existing markdown files
2. Consolidate into comprehensive README.md
3. Remove unnecessary markdown files
4. Ensure README.md is complete and well-organized

### Step 6: Django Application Setup (Week 8-9)

1. Create Django project structure
2. Implement core models
3. Set up REST API endpoints
4. Integrate Celery for async tasks
5. Connect to MLflow and PySpark

### Step 5: Mojo Integration (Optional, Week 9-10)

1. Port critical functions to Mojo
2. Create Mojo inference service
3. Integrate with Django API
4. Benchmark performance gains

### Step 8: Testing & Validation (Week 12-13)

1. Comprehensive unit tests
2. Integration tests
3. Performance benchmarking
4. Load testing
5. Documentation

## Success Metrics

- **Training Time**: 50-70% reduction in batch training time
- **Model Accuracy**: 10-15% improvement in forecast accuracy (MAPE reduction)
- **API Latency**: < 200ms for single prediction, < 5s for batch (100 products)
- **Scalability**: Support 10,000+ products, 100+ concurrent users
- **Code Quality**: 80%+ test coverage, type hints throughout

## Dependencies

**New Python Packages**:

- `django>=5.0`
- `djangorestframework>=3.14`
- `celery>=5.3`
- `redis>=5.0`
- `optuna>=3.0`
- `feast>=0.36` (optional, for feature store)
- `delta-spark>=3.0` (for Delta Lake)

**Mojo** (if implementing):

- Mojo SDK for Python interop
- Mojo compiler

## Risk Mitigation

1. **Spark Session Management**: Use connection pooling and graceful shutdown
2. **Memory Issues**: Implement data sampling for large datasets
3. **Model Versioning**: Comprehensive MLflow integration with rollback capability
4. **Mojo Adoption**: Start with proof-of-concept, measure ROI before full migration