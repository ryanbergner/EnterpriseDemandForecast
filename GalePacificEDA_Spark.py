from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    HoltWinters,
    ARIMA,
    CrostonClassic,
    SeasonalExponentialSmoothingOptimized
)
from pyspark.sql.functions import (
    col,
    when,
    lag,
    lead,
    sum as spark_sum,
    stddev_samp,
    month,
    year,
    lit,
    to_date,
    unix_timestamp,
    avg as spark_avg,
    rand,
    coalesce,
    abs as spark_abs,
    broadcast
)
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    DoubleType,
    IntegerType,
    DecimalType
)

from typing import Optional, List, Any, Dict, Tuple
from math import sqrt as math_sqrt, isclose

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml
from mlflow.models.signature import infer_signature
import mlflavors
import mlflavors.statsforecast  # Ensure this package is installed
import torch

# Temporary
import warnings
warnings.filterwarnings("ignore")

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MetaAlgorithmReadWrite,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter
)
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasCollectSubModels, HasParallelism, HasSeed

from src.preprocessing.preprocess import aggregate_sales_data, retrieve_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import (
    train_sparkML_models,
    evaluate_sparkML_models
)
from src.model_training.stats_models import (
    train_stats_models,
    evaluate_stats_models
)

# TabPFN-related
import tabpfn
import tabpfn_extensions
# import tabpfn_client
# import tabpfn_time_series

# token = tabpfn_client.get_access_token()
# tabpfn_client.set_access_token(token)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

requirements_path = "/Workspace/Users/ryan.bergner@mcaconnect.com/InspireDatabricks/DemandForecast/requirements.txt"

# COMMAND ----------
# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------
df = (
    spark.read.format("csv")
    .option("header", True)
    .load("/Volumes/main/default/demo_data/gale_pacific_hist_sales (1).csv")
)

df = df.withColumn("DemandDate", to_date(col("DemandDate"), "MM-d-yyyy HH:mm:ss")) \
       .withColumn("DemandQuantity", col("DemandQuantity").cast(DoubleType()))

display(df)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------
# 1) Read in the data
df = df.withColumn("DemandDate", to_date(col("DemandDate"), "MM-d-yyyy HH:mm:ss")) \
       .withColumn("DemandQuantity", col("DemandQuantity").cast(DoubleType()))

# 2) Run monthly aggregation
df_agg = aggregate_sales_data(
    df=df,
    date_column="DemandDate",
    product_id_column="ItemNumber",
    quantity_column="DemandQuantity",
    month_end_column="MonthEndDate"
)

# 3) Add feature-engineered columns
df_feat = add_features(
    df_agg,
    month_end_column="MonthEndDate",
    product_id_column="ItemNumber",
    quantity_column="DemandQuantity"
)

display(df_feat)
df_feat.columns

# COMMAND ----------
print("rows: " + str(df_feat.count()), "Columns: " + str(len(df_feat.columns)))

# COMMAND ----------
# MAGIC %md
# MAGIC - 'https://news.ycombinator.com/item?id=42647343'
# MAGIC - I am excited to announce the release of TabPFN v2, a tabular foundation model
# MAGIC   that delivers state-of-the-art predictions on small datasets in just 2.8
# MAGIC   seconds for classification and 4.8 seconds for regression compared to strong
# MAGIC   baselines tuned for 4 hours. Published in Nature, this model outperforms
# MAGIC   traditional methods on datasets with up to 10,000 samples and 500 features.
# MAGIC ...
# MAGIC (additional TabPFN context/notes)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. HTH Implementation
# MAGIC ### HTH Function

# COMMAND ----------
import datetime
import mlflow.sklearn
import mlflow.spark
import pandas as pd
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, unix_timestamp, broadcast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
from typing import List, Dict, Any

def main_train_no_cat_v2(
    df: DataFrame,
    date_column: str,
    product_id_column: str,
    quantity_column: str,
    month_end_column: str,
    list_of_features: List[str],
    tabpfn_list_of_features: List[str],
    target_column: str,
    list_of_ml_models: List[Dict[str, Any]],
    list_of_stats_models: List[Dict[str, Any]],
    experiment_name: str,
    max_distinct_pids: int,
    requirements_path: str
):
    """
    A 'no category' function that:
      1) aggregator => add_features => df_feat
      2) Removes any raw string columns (like 'bucket'/'month') from numeric pipeline
      3) Samples up to 'max_distinct_pids' product IDs => keep only them
      4) Does 80% date-based train/test split
      5) Trains:
          (A) TabPFN (using tabpfn_list_of_features)
          (B) each StatsForecast model
          (C) each Spark ML (using list_of_features)
      6) Logs each in MLflow nested runs
      7) DISABLES dataset logging
      8) Logs metrics in lower case
    """

    # Turn off dataset logging => set log_datasets=False
    mlflow.sklearn.autolog(disable=True, log_models=False, log_datasets=False)
    mlflow.pyspark.ml.autolog(disable=True, log_models=False, log_datasets=False)

    # Create or retrieve MLflow experiment
    experiment_name_full = f"/Workspace/Users/ryan.bergner@mcaconnect.com/{experiment_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    try:
        experiment_id = mlflow.create_experiment(experiment_name_full)
    except mlflow.exceptions.MlflowException:
        exp_info = mlflow.get_experiment_by_name(experiment_name_full)
        if exp_info:
            experiment_id = exp_info.experiment_id
        else:
            raise

    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"[INFO] Created/Retrieved experiment: '{experiment_name_full}' => ID={experiment_id}\n")

    #################################################
    # 1) aggregator => add_features => df_feat
    #################################################
    print("[INFO] Aggregating + adding features ...")
    df_agg = aggregate_sales_data(
        df=df,
        date_column=date_column,
        product_id_column=product_id_column,
        quantity_column=quantity_column,
        month_end_column=month_end_column
    )
    df_feat = add_features(df_agg, month_end_column, product_id_column, quantity_column)

    # Possibly filter total_orders >= 5
    df_feat = df_feat.filter(col("total_orders") >= 5)

    # 2) Remove string columns 'bucket' or 'month' if present in the feature lists
    for col_to_remove in ["bucket", "month"]:
        if col_to_remove in list_of_features:
            list_of_features.remove(col_to_remove)
        if col_to_remove in tabpfn_list_of_features:
            tabpfn_list_of_features.remove(col_to_remove)

    # 3) Drop null rows
    df_feat = df_feat.dropna()
    final_count = df_feat.count()
    if final_count == 0:
        print("[INFO] => 0 rows remain after dropna => skipping.")
        return
    print(f"[INFO] => {final_count} rows remain.\n")

    # Distinct product IDs => sample
    distinct_pids = df_feat.select(product_id_column).distinct()
    total_distinct = distinct_pids.count()
    print(f"[INFO] Found {total_distinct} distinct product IDs overall.")

    if total_distinct > max_distinct_pids:
        fraction = max_distinct_pids / float(total_distinct)
        distinct_pids_sampled = (
            distinct_pids
            .sample(withReplacement=True, fraction=fraction, seed=42)
            .limit(max_distinct_pids)
        )
    else:
        distinct_pids_sampled = distinct_pids

    sample_count = distinct_pids_sampled.count()
    print(f"[INFO] Sampled {sample_count} distinct product IDs (with replacement).")

    df_feat = df_feat.join(distinct_pids_sampled, on=product_id_column, how="inner")
    new_count = df_feat.count()
    if new_count == 0:
        print("[INFO] => 0 rows remain after sampling => skipping training.")
        return
    print(f"[INFO] => row count after sampling => {new_count}")

    # 4) 80% date-based split
    df_feat = df_feat.withColumn("unix_time", unix_timestamp(col(month_end_column)))
    df_feat = df_feat.filter(col("unix_time").isNotNull())
    if df_feat.count() == 0:
        print("[INFO] => no data after ensuring 'unix_time' => skipping training.")
        return

    approx_list = df_feat.approxQuantile("unix_time", [0.8], 0)
    if not approx_list:
        print("[INFO] => no quantile => skipping training.")
        return
    cutoff_ts = approx_list[0]
    cutoff_date = datetime.datetime.fromtimestamp(cutoff_ts)
    print(f"[INFO] Train/Test split cutoff => {cutoff_date}\n")

    train_spark = df_feat.filter(col("unix_time") < cutoff_ts)
    test_spark = df_feat.filter(col("unix_time") > cutoff_ts)
    train_count = train_spark.count()
    test_count = test_spark.count()
    print(f"[INFO] => train={train_count}, test={test_count}\n")

    if train_count == 0 or test_count == 0:
        print("[INFO] => empty train or test => skipping.")
        return

    # 5) Single MLflow run => nested sub-runs
    with mlflow.start_run(experiment_id=experiment_id, run_name="AllModelsNoCat") as parent_run:
        print("=== Training multiple models on single dataset (no cat) ===\n")

        # (A) TabPFN
        with mlflow.start_run(run_name="TabPFN", nested=True):
            print("[TabPFN] => training ...")
            train_pdf = (
                train_spark
                .select(*tabpfn_list_of_features, target_column)
                .toPandas()
                .dropna()
            )
            test_pdf = (
                test_spark
                .select(*tabpfn_list_of_features, target_column)
                .toPandas()
                .dropna()
            )

            if len(train_pdf) == 0 or len(test_pdf) == 0:
                print("[TabPFN] => no data => skipping.")
            else:
                from tabpfn import TabPFNRegressor

                X_train = train_pdf[tabpfn_list_of_features].values
                y_train = train_pdf[target_column].values
                X_test = test_pdf[tabpfn_list_of_features].values
                y_test = test_pdf[target_column].values

                reg = TabPFNRegressor(device="auto")
                reg.fit(X_train, y_train)

                preds = reg.predict(X_test)
                mse_val = mean_squared_error(y_test, preds)
                rmse_val = float(np.sqrt(mse_val))
                mae_val = float(mean_absolute_error(y_test, preds))
                r2_val = float(r2_score(y_test, preds))

                mlflow.log_metric("mse", mse_val)
                mlflow.log_metric("rmse", rmse_val)
                mlflow.log_metric("mae", mae_val)
                mlflow.log_metric("r2", r2_val)
                print(f"[TabPFN] => mse={mse_val:.3f}, rmse={rmse_val:.3f}, mae={mae_val:.3f}, r2={r2_val:.3f}")

                # Log model
                pdf_sample = train_spark.select(*tabpfn_list_of_features, target_column).limit(5).toPandas()
                signature = infer_signature(
                    pdf_sample[tabpfn_list_of_features],
                    pdf_sample[target_column]
                )
                mlflow.sklearn.log_model(
                    sk_model=reg,
                    artifact_path="TabPFN_model",
                    pip_requirements=requirements_path,
                    signature=signature,
                    input_example=pdf_sample[tabpfn_list_of_features]
                )
                print("[TabPFN] => model logged.\n")

        # (B) StatsForecast => each model
        for stats_info in list_of_stats_models:
            alias_stats = stats_info["alias"]
            model_stats = stats_info["model"]

            with mlflow.start_run(run_name=f"Stats_{alias_stats}", nested=True):
                print(f"[Stats => {alias_stats}] => training ...")
                stats_trained = train_stats_models(
                    models=[model_stats],
                    train_df=train_spark,
                    month_end_column=month_end_column,
                    product_id_column=product_id_column,
                    target_column=target_column
                )
                evaluate_stats_models(
                    stats_model=stats_trained,
                    test_df=test_spark,
                    month_end_column=month_end_column,
                    product_id_column=product_id_column,
                    target_column=target_column,
                    experiment_id=experiment_id,
                    artifact_location=None,
                    model_name=alias_stats
                )
                print(f"[Stats => {alias_stats}] => done.\n")

        # (C) Spark ML => each
        for ml_model_dict in list_of_ml_models:
            model_alias = ml_model_dict["alias"]
            model_obj = ml_model_dict["model"]
            param_grid = ml_model_dict.get("param_grid", None)

            with mlflow.start_run(run_name=f"Spark_{model_alias}", nested=True):
                print(f"[Spark ML => {model_alias}] => training ...")

                pipeline_model = train_sparkML_models(
                    model=model_obj,
                    train_df=train_spark.select(*list_of_features, target_column),
                    featuresCols=list_of_features,
                    labelCol=target_column,
                    paramGrid=param_grid
                )
                eval_metrics = evaluate_sparkML_models(
                    model=pipeline_model,
                    test_df=test_spark.select(*list_of_features, target_column),
                    features_cols=list_of_features,
                    label_col=target_column,
                    requirements_path=requirements_path,
                    model_alias=model_alias
                )
                print(f"[Spark ML => {model_alias}] => {eval_metrics}\n")

    print("[INFO] All training complete! Check MLflow UI for runs & metrics.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example Usage / HTH Execution

# COMMAND ----------
date_column = "DemandDate"
product_id_column = "ItemNumber"
quantity_column = "DemandQuantity"
month_end_column = "MonthEndDate"
target_column = "lead_month_1"

list_of_features = [
    "DemandQuantity",
    "ma_4_month",
    "ma_8_month",
    "ma_12_month",
    "ma_3_sales",
    "ma_5_sales",
    "ma_7_sales",
    "cov_quantity",
    "avg_demand_quantity",
    "avg_demand_interval",
    "months_since_last_order",
    "last_order_quantity",
    "month",
    "bucket"
]

tabpfn_list_of_features = list_of_features[:]  # Copy

# Define Spark ML models
gbt = GBTRegressor(labelCol=target_column)
GBTParamGrid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [2, 3, 5])
    .addGrid(gbt.maxIter, [10])
    .build()
)
gbt_model = {"alias": "GBT_model", "model": gbt, "param_grid": GBTParamGrid}

rf = RandomForestRegressor(labelCol=target_column)
rfParamGrid = (
    ParamGridBuilder()
    .addGrid(rf.maxDepth, [2, 3, 5])
    .addGrid(rf.numTrees, [10])
    .build()
)
rf_model = {"alias": "RF_model", "model": rf, "param_grid": rfParamGrid}

lr = LinearRegression(labelCol=target_column)
lr_model = {"alias": "LR_model", "model": lr, "param_grid": None}

list_of_ml_models = [lr_model, rf_model, gbt_model]

list_of_stats_models = [
    {
        "alias": "SeasonalExponentialSmoothingOptimized",
        "model": SeasonalExponentialSmoothingOptimized(season_length=12)
    },
    {
        "alias": "CrostonClassic",
        "model": CrostonClassic()
    }
]

# COMMAND ----------
# MAGIC %md
# MAGIC ### Run: No Categories (Sample 50 Product IDs)

# COMMAND ----------
main_train_no_cat_v2(
    df,
    date_column,
    product_id_column,
    quantity_column,
    month_end_column,
    list_of_features,
    tabpfn_list_of_features,
    target_column,
    list_of_ml_models,       # 3 Spark ML definitions
    list_of_stats_models,    # 2 StatsForecast models
    experiment_name="ALL_HTH_NOCAT_50PIDS",
    max_distinct_pids=50,
    requirements_path="requirements.txt"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Run: No Categories (Sample 20 Product IDs)

# COMMAND ----------
main_train_no_cat_v2(
    df,
    date_column,
    product_id_column,
    quantity_column,
    month_end_column,
    list_of_features,
    tabpfn_list_of_features,
    target_column,
    list_of_ml_models,
    list_of_stats_models,
    experiment_name="ALL_HTH_NOCAT_20PIDS",
    max_distinct_pids=20,
    requirements_path="requirements.txt"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Run: No Categories (Sample 150 Product IDs)

# COMMAND ----------
main_train_no_cat_v2(
    df,
    date_column,
    product_id_column,
    quantity_column,
    month_end_column,
    list_of_features,
    tabpfn_list_of_features,
    target_column,
    list_of_ml_models,
    list_of_stats_models,
    experiment_name="ALL_HTH_NOCAT_150PIDS",
    max_distinct_pids=150,
    requirements_path="requirements.txt"
)
