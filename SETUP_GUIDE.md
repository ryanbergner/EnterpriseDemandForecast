# Setup Guide

Welcome! This guide will help you get started with the M5 Forecasting Pipeline.

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download M5 Dataset

Option A: **From Kaggle** (Recommended)
```bash
# Install kaggle CLI
pip install kaggle

# Set up Kaggle API credentials (see: https://www.kaggle.com/docs/api)
# Place kaggle.json in ~/.kaggle/

# Download M5 dataset
mkdir -p data/m5
kaggle competitions download -c m5-forecasting-accuracy -p data/m5/
unzip data/m5/sales_train_validation.csv.zip -d data/m5/
```

Option B: **Manual Download**
1. Go to https://www.kaggle.com/c/m5-forecasting-accuracy/data
2. Download `sales_train_validation.csv`
3. Place in `data/m5/` directory

### Step 3: Run Your First Training

```bash
# Quick test (5 items, should take ~2 minutes)
python main_train_m5.py --limit-products 5 --max-items 5

# View results
mlflow ui
# Open browser to http://localhost:5000
```

**That's it! You're ready to go! 🎉**

---

## 📊 Next Steps

### Option 1: Run Full M5 Training

```bash
# Train on 100 items (takes ~30 minutes)
python main_train_m5.py --max-items 100 --experiment-name M5_Experiment_100

# Train on 500 items (takes ~2 hours)
python main_train_m5.py --max-items 500 --experiment-name M5_Experiment_500
```

### Option 2: Explore Data with EDA

```bash
# Launch Jupyter notebook
jupyter notebook EDA.py

# Or in Databricks
# Upload EDA.py to your Databricks workspace and run
```

### Option 3: Use Your Own Data

```bash
# Validate and prepare your data
python upload_custom_data.py \
  --input-path your_sales_data.csv \
  --output-path data/custom/validated.csv

# Train on your data
python main_train.py \
  --data-path data/custom/validated.csv \
  --experiment-name My_Custom_Experiment
```

---

## ☁️ Azure Databricks Setup (Optional)

### Prerequisites
- Azure subscription
- Databricks workspace
- Azure Storage account with M5 dataset

### Step 1: Configure Azure Credentials

```bash
# Copy template
cp azure_config.py.template azure_config.py

# Edit azure_config.py and add:
# - AZURE_STORAGE_ACCOUNT_NAME
# - AZURE_STORAGE_ACCOUNT_KEY (or SAS token)
# - AZURE_STORAGE_CONTAINER_NAME
# - DATABRICKS_HOST
# - DATABRICKS_TOKEN
```

Or use environment variables:

```bash
export AZURE_STORAGE_ACCOUNT_NAME="mystorageaccount"
export AZURE_STORAGE_ACCOUNT_KEY="your_key_here"
export AZURE_STORAGE_CONTAINER_NAME="m5-forecasting"
export DATABRICKS_HOST="https://adb-xxxxx.azuredatabricks.net"
export DATABRICKS_TOKEN="dapi..."
```

### Step 2: Upload M5 Data to Blob Storage

```bash
# Using Azure CLI
az storage blob upload-batch \
  --account-name mystorageaccount \
  --destination m5-forecasting \
  --source data/m5/ \
  --pattern "*.csv"
```

### Step 3: Run Training in Databricks

```python
# In Databricks notebook
%run /Workspace/Repos/your-repo/main_train_databricks

# Or submit as a job
dbx execute --job=m5-training
```

---

## 🔧 Configuration

### Customize Training Parameters

Edit `config.py`:

```python
# Change maximum items to train
MAX_DISTINCT_ITEMS = 200

# Adjust train/test split
TRAIN_SPLIT_RATIO = 0.8  # 80% train, 20% test

# Modify feature engineering
MA_WINDOWS = [3, 6, 12]  # Moving average windows
LAG_FEATURES = [1, 2, 3, 6, 12]  # Lag features
```

### Add Custom Models

Edit any training script:

```python
# Add XGBoost
from xgboost.spark import SparkXGBRegressor
xgb_model = SparkXGBRegressor(labelCol=target_column)
spark_ml_models.append({
    "alias": "XGBoost",
    "model": xgb_model,
    "param_grid": None
})
```

---

## 📚 Common Tasks

### Task 1: Compare Model Performance

```bash
# Train all models
python main_train_m5.py --max-items 100

# View in MLflow
mlflow ui

# Or query programmatically
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    for run in runs:
        print(f'{run.data.tags.get(\"mlflow.runName\")}: {run.data.metrics}')
"
```

### Task 2: Make Predictions on New Data

```python
from main_inference import load_model, make_predictions

# Load best model from MLflow
model = load_model("runs:/abc123/model")

# Make predictions
predictions = make_predictions(model, new_data)
```

### Task 3: Export Model for Production

```python
import mlflow

# Register model
mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="M5_Production_Model"
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="M5_Production_Model",
    version=1,
    stage="Production"
)
```

### Task 4: Schedule Regular Training

Create a cron job:

```bash
# Edit crontab
crontab -e

# Add weekly training (every Monday at 2 AM)
0 2 * * 1 cd /path/to/project && /path/to/venv/bin/python main_train_m5.py --max-items 500 >> /var/log/m5_training.log 2>&1
```

Or use Databricks Jobs:

```json
{
  "name": "Weekly M5 Training",
  "schedule": {
    "quartz_cron_expression": "0 0 2 ? * MON",
    "timezone_id": "UTC"
  },
  "tasks": [{
    "task_key": "train",
    "python_wheel_task": {
      "package_name": "m5_forecasting",
      "entry_point": "main_train_databricks"
    }
  }]
}
```

---

## ❓ FAQ

### Q: How much memory do I need?
**A:** For local training:
- 5-10 items: 2GB RAM
- 50-100 items: 4-8GB RAM
- 500+ items: 16GB+ RAM or use Databricks

### Q: Can I use other datasets?
**A:** Yes! Use `upload_custom_data.py` to validate any dataset with:
- date column
- item_id column
- quantity column

### Q: How long does training take?
**A:** Approximate times (local machine):
- 5 items: 2-5 minutes
- 50 items: 15-30 minutes
- 100 items: 30-60 minutes
- 500 items: 2-4 hours

### Q: Which model should I use?
**A:** Start with:
1. Seasonal Exponential Smoothing (fast, good baseline)
2. Random Forest (robust, handles nonlinearity)
3. Gradient Boosted Trees (often best performance)
4. TabPFN (great for small datasets < 10k rows)

### Q: How do I improve model performance?
**A:**
1. Add more features (see `src/feature_engineering/`)
2. Tune hyperparameters (edit param grids)
3. Increase training data (more items/history)
4. Try ensemble methods
5. Add domain-specific features

### Q: Can I use GPU for training?
**A:** 
- Spark ML: Uses CPU by default
- TabPFN: Supports GPU (`device="cuda"`)
- Neural Networks: Supports GPU (TensorFlow/PyTorch)
- For Databricks: Use GPU clusters

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pyspark'`
```bash
pip install pyspark==3.4.0
```

### Issue: `FileNotFoundError: data/m5/sales_train_validation.csv`
```bash
# Download M5 dataset (see Step 2 above)
mkdir -p data/m5
# Download from Kaggle
```

### Issue: `OutOfMemoryError`
```bash
# Reduce number of items
python main_train_m5.py --max-items 10

# Or increase Spark memory
export PYSPARK_DRIVER_MEMORY=8g
```

### Issue: `Azure Blob Storage access denied`
```bash
# Verify credentials
python -c "from azure_config import validate_azure_config; print(validate_azure_config())"

# Check environment variables
echo $AZURE_STORAGE_ACCOUNT_NAME
echo $AZURE_STORAGE_ACCOUNT_KEY
```

### Issue: `TabPFN not available`
```bash
# TabPFN is optional - install if needed
pip install tabpfn

# Or skip TabPFN (it will be automatically skipped if not installed)
```

---

## 📞 Getting Help

1. **Check Documentation**: Read `README_NEW.md`
2. **Review Changes**: See `REFACTORING_SUMMARY.md`
3. **Check Configuration**: Review `config.py`
4. **Search Issues**: Look for similar problems in GitHub issues
5. **Ask for Help**: Create a new issue with:
   - Error message
   - Full command you ran
   - Python version (`python --version`)
   - Package versions (`pip list`)

---

## ✅ Verification Checklist

Before running production training, verify:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] M5 dataset downloaded and in `data/m5/`
- [ ] Test training completed successfully (5 items)
- [ ] MLflow UI accessible
- [ ] Azure credentials configured (if using Databricks)
- [ ] Sufficient disk space for MLflow artifacts (~1GB per experiment)
- [ ] Sufficient memory for desired number of items

---

## 🎓 Learning Resources

- **M5 Competition**: https://www.kaggle.com/c/m5-forecasting-accuracy
- **Time Series Forecasting**: https://otexts.com/fpp3/
- **PySpark Tutorial**: https://spark.apache.org/docs/latest/api/python/getting_started/quickstart.html
- **MLflow Guide**: https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
- **StatsForecast**: https://nixtla.github.io/statsforecast/
- **Azure Databricks**: https://docs.microsoft.com/en-us/azure/databricks/

---

**Ready to start? Run this:**

```bash
python main_train_m5.py --limit-products 5 --max-items 5
```

Good luck! 🚀
