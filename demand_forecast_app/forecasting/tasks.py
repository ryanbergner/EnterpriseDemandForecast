"""
Celery tasks for training, prediction, and drift checks.
"""

from celery import shared_task
from django.utils import timezone

from .models import ForecastJob
from .services import TrainingService, InferenceService, ValidationService


@shared_task
def train_model_async(job_id: int, params: dict) -> None:
    job = ForecastJob.objects.get(id=job_id)
    job.status = ForecastJob.Status.RUNNING
    job.started_at = timezone.now()
    job.save(update_fields=["status", "started_at"])
    try:
        metrics = TrainingService.run_training_job(job, params)
        job.status = ForecastJob.Status.COMPLETED
        job.metrics = metrics or {}
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "metrics", "completed_at"])
    except Exception as exc:
        job.status = ForecastJob.Status.FAILED
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])


@shared_task
def generate_predictions_async(job_id: int, params: dict) -> None:
    job = ForecastJob.objects.get(id=job_id)
    job.status = ForecastJob.Status.RUNNING
    job.started_at = timezone.now()
    job.save(update_fields=["status", "started_at"])
    try:
        InferenceService.run_prediction_job(job, params)
        job.status = ForecastJob.Status.COMPLETED
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "completed_at"])
    except Exception as exc:
        job.status = ForecastJob.Status.FAILED
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])


@shared_task
def check_drift_async(job_id: int, params: dict) -> None:
    job = ForecastJob.objects.get(id=job_id)
    job.status = ForecastJob.Status.RUNNING
    job.started_at = timezone.now()
    job.save(update_fields=["status", "started_at"])
    try:
        result = ValidationService.check_drift(params)
        job.status = ForecastJob.Status.COMPLETED
        job.metrics = result or {}
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "metrics", "completed_at"])
    except Exception as exc:
        job.status = ForecastJob.Status.FAILED
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])
