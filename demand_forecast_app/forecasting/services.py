"""
Service layer for training, inference, and validation.
"""

from typing import Dict, Any
from pathlib import Path
import sys

from .models import ForecastJob
from ml_pipeline.training_service import train_with_spark
from ml_pipeline.inference_service import predict_with_spark
from ml_pipeline.model_registry import sync_model_versions
from ml_pipeline.spark_service import get_spark_session


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


class TrainingService:
    @staticmethod
    def start_training(**params) -> ForecastJob:
        job = ForecastJob.objects.create(
            job_type=ForecastJob.JobType.TRAIN,
            status=ForecastJob.Status.QUEUED,
            parameters=params,
        )
        from .tasks import train_model_async
        train_model_async.delay(job.id, params)
        return job

    @staticmethod
    def run_training_job(job: ForecastJob, params: Dict[str, Any]) -> Dict[str, Any]:
        metrics = train_with_spark(params)
        sync_model_versions()
        return metrics or {}


class InferenceService:
    @staticmethod
    def start_prediction(**params) -> ForecastJob:
        job = ForecastJob.objects.create(
            job_type=ForecastJob.JobType.PREDICT,
            status=ForecastJob.Status.QUEUED,
            parameters=params,
        )
        from .tasks import generate_predictions_async
        generate_predictions_async.delay(job.id, params)
        return job

    @staticmethod
    def run_prediction_job(job: ForecastJob, params: Dict[str, Any]) -> None:
        predict_with_spark(job, params)


class ValidationService:
    @staticmethod
    def check_drift(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic drift check using prediction error monitor.
        Requires params: actual_col, predicted_col, date_col, data_path
        """
        _ensure_repo_on_path()
        from src.validation.drift_detection import PredictionErrorMonitor
        data_path = params.get("data_path")
        actual_col = params.get("actual_col", "Quantity")
        predicted_col = params.get("predicted_col", "prediction")
        date_col = params.get("date_col", "MonthEndDate")

        spark = get_spark_session()
        df = spark.read.parquet(data_path)
        monitor = PredictionErrorMonitor()
        result = monitor.monitor(df, actual_col, predicted_col, date_col)
        return {
            "drift_detected": result.drift_detected,
            "drift_score": result.drift_score,
            "confidence": result.confidence,
            "details": result.details,
        }
