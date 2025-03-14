from pyspark.sql import SparkSession
from datetime import datetime
from dateutil.relativedelta import relativedelta
import mlflow
import mlflavors
import pandas as pd
from pyspark.sql.functions import lit


# Initialize MLflow client
client = mlflow.tracking.MlflowClient()


def generate_predictions(
    model_uri,
    model_name,
    sales_pattern,
    df_inference,
    month_end_column,
    product_id_column
):
    try:
        # Try loading the model with mlflow
        loaded_model = mlflow.spark.load_model(model_uri)
        flavor = "sparkml"
    except Exception:
        loaded_model = mlflavors.statsforecast.load_model(model_uri)
        flavor = "statsforecast"

    if flavor == "statsforecast":
        # Find the last month in df_inference and calculate the month for which we want to do inference
        last_month = df_inference.select(month_end_column).distinct().collect()[0][0]
        next_month = (
            last_month
            + relativedelta(days=1)
            + relativedelta(months=1)
            + relativedelta(days=-1)
        )

        # As a Statsforecast model is trained up until time = t,
        # it will automatically predict for the month after t.
        # We first calculate the first month of Predict quantities for a short horizon
        # using the stats model, then calculate how many months ahead we need to forecast.
        min_ds = loaded_model.predict(h=1)["ds"].min()
        new_horizon = (
            pd.to_datetime(next_month) - pd.to_datetime(min_ds)
        ).days // 30 + 1  # Assuming 30 days in a month

        # Run predictions for new_horizon
        predicted_quantities_pdf = loaded_model.predict(h=new_horizon)

        # Keep only the values where "ds" is equal to "next_month"
        predicted_quantities_pdf["ds"] = pd.to_datetime(predicted_quantities_pdf["ds"]).dt.date
        predicted_quantities_pdf = predicted_quantities_pdf[
            predicted_quantities_pdf["ds"] == next_month
        ]

        # Rename and format the prediction dataframe
        print("---------- columns in predicted_df: ", predicted_quantities_pdf.columns)
        predicted_quantities_pdf = predicted_quantities_pdf.reset_index(drop=True)

        # Some Pandas versions might trip on iteritems, but we'll keep as is:
        predicted_quantities_pdf.iteritems = predicted_quantities_pdf.items
        predicted_quantities_pdf.columns = [
            product_id_column,
            month_end_column,
            "prediction",
        ]

        predicted_df = (
            spark.createDataFrame(predicted_quantities_pdf)
            .withColumn("SalesPattern", lit(sales_pattern))
            .withColumn("Model", lit(model_name))
        )

        predicted_df = predicted_df.select(
            [product_id_column, month_end_column, "prediction", "SalesPattern", "Model"]
        )

    else:
        predicted_df = (
            loaded_model.transform(
                df_inference.filter(df_inference["product_category"] == sales_pattern)
            )
            .withColumn("SalesPattern", lit(sales_pattern))
            .withColumn("Model", lit(model_name))
        )

        predicted_df = predicted_df.select(
            [product_id_column, month_end_column, "prediction", "SalesPattern", "Model"]
        )

    return predicted_df


# 20250109 Jeroen: This logic is no longer used
# # Define a function to filter models based on tags
# def get_model_by_tag(model_name, tag_key, tag_value):
#     # Search for registered models with the specified tag
#     registered_models = client.search_model_versions(
#         filter_string=f"name='{model_name}' AND tags.`{tag_key}`='{tag_value}'"
#     )
#     return registered_models

# # Function to retrieve a model by alias
# def get_model_by_alias(model_name: str, alias: str):
#     """
#     Retrieve the model version associated with a specific alias.

#     Args:
#         model_name (str): The name of the model in MLflow.
#         alias (str): The alias of the model to retrieve (e.g., 'champion' or 'challenger').

#     Returns:
#         str: The model URI for the specified alias.
#     """
#     model_version_info = client.get_model_version_by_alias(name=model_name, alias=alias)
#     model_uri = f"models:/{model_name}/{model_version_info.version}"
#     return model_uri
