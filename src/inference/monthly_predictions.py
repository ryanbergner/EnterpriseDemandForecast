"""
Monthly prediction helpers with uncertainty quantification.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, max as spark_max

import mlflow
import mlflavors

from .inference import generate_predictions, _get_spark_session
from .confidence_intervals import UncertaintyQuantifier


class MonthlyPredictionGenerator:
    def __init__(
        self,
        model_uri: str,
        model_name: str = "forecast_model",
        spark: Optional[SparkSession] = None,
        confidence_level: float = 0.95,
        date_col: str = "MonthEndDate",
        product_id_col: str = "item_id"
    ):
        self.model_uri = model_uri
        self.model_name = model_name
        self.spark = spark or _get_spark_session()
        self.confidence_level = confidence_level
        self.date_col = date_col
        self.product_id_col = product_id_col
        self.uncertainty = UncertaintyQuantifier(confidence_level)
        self._model = None
        self._model_flavor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._model = mlflow.spark.load_model(self.model_uri)
            self._model_flavor = "sparkml"
        except Exception:
            self._model = mlflavors.statsforecast.load_model(self.model_uri)
            self._model_flavor = "statsforecast"

    def get_next_month_date(self, df: DataFrame) -> datetime:
        max_date = df.agg(spark_max(col(self.date_col))).collect()[0][0]
        if max_date is None:
            raise ValueError(f"No valid dates found in column '{self.date_col}'")
        if isinstance(max_date, str):
            max_date = datetime.strptime(max_date, "%Y-%m-%d")

        next_month = max_date + relativedelta(months=1)
        next_month_end = next_month.replace(day=1) + relativedelta(months=1) - timedelta(days=1)
        return next_month_end

    def generate(
        self,
        df_inference: DataFrame,
        sales_pattern: Optional[str] = None,
        include_intervals: bool = True,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "lead_month_1"
    ) -> DataFrame:
        self._load_model()

        predictions_df = generate_predictions(
            model_uri=self.model_uri,
            model_name=self.model_name,
            sales_pattern=sales_pattern or "All",
            df_inference=df_inference,
            month_end_column=self.date_col,
            product_id_column=self.product_id_col
        )

        if (
            include_intervals
            and feature_cols
            and self._model_flavor == "sparkml"
            and target_col in df_inference.columns
        ):
            predictions_df = self.uncertainty.pi.add_intervals(
                model=self._model,
                test_df=df_inference,
                feature_cols=feature_cols,
                label_col=target_col
            )

        predictions_df = predictions_df.withColumn(
            "prediction_date",
            lit(datetime.now().strftime("%Y-%m-%d"))
        ).withColumn(
            "confidence_level",
            lit(self.confidence_level)
        )

        return predictions_df


def generate_monthly_predictions(
    model_uri: str,
    df_inference: DataFrame,
    spark: Optional[SparkSession] = None,
    date_col: str = "MonthEndDate",
    product_id_col: str = "item_id",
    sales_pattern: Optional[str] = None,
    include_intervals: bool = True,
    confidence_level: float = 0.95
) -> DataFrame:
    generator = MonthlyPredictionGenerator(
        model_uri=model_uri,
        spark=spark,
        confidence_level=confidence_level,
        date_col=date_col,
        product_id_col=product_id_col
    )
    return generator.generate(
        df_inference=df_inference,
        sales_pattern=sales_pattern,
        include_intervals=include_intervals
    )


def batch_predict_next_month(
    model_uris: Dict[str, str],
    df_inference: DataFrame,
    spark: Optional[SparkSession] = None,
    date_col: str = "MonthEndDate",
    product_id_col: str = "item_id"
) -> DataFrame:
    spark = spark or _get_spark_session()
    all_predictions = None

    for model_name, model_uri in model_uris.items():
        generator = MonthlyPredictionGenerator(
            model_uri=model_uri,
            model_name=model_name,
            spark=spark,
            date_col=date_col,
            product_id_col=product_id_col
        )
        preds = generator.generate(df_inference=df_inference, include_intervals=True)
        preds = preds.withColumn("model_version", lit(model_name))
        if all_predictions is None:
            all_predictions = preds
        else:
            all_predictions = all_predictions.unionByName(preds, allowMissingColumns=True)

    if all_predictions is None:
        raise ValueError("No predictions generated from any model")

    return all_predictions
