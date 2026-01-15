"""
MLflow registry integration for Django.
"""

from typing import List

import mlflow
from django.conf import settings

from forecasting.models import ModelVersion


def _get_client() -> mlflow.tracking.MlflowClient:
    if settings.MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(settings.MLFLOW_REGISTRY_URI)
    if settings.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


def sync_model_versions() -> List[ModelVersion]:
    """
    Sync MLflow registered models into the Django database.
    """
    client = _get_client()
    versions = []
    for model in client.search_registered_models():
        for mv in client.search_model_versions(f"name='{model.name}'"):
            version, _ = ModelVersion.objects.update_or_create(
                name=model.name,
                version=mv.version,
                defaults={
                    "stage": mv.current_stage or "",
                    "mlflow_uri": f"models:/{model.name}/{mv.version}",
                    "run_id": mv.run_id or "",
                    "metrics": mv.tags or {},
                },
            )
            versions.append(version)
    return versions
