from django.urls import path

from .views import (
    TrainView,
    TrainStatusView,
    PredictView,
    ModelsView,
    FeaturesView,
    MetricsView,
    DriftCheckView,
)


urlpatterns = [
    path("train/", TrainView.as_view(), name="train"),
    path("train/<int:job_id>/", TrainStatusView.as_view(), name="train-status"),
    path("predict/", PredictView.as_view(), name="predict"),
    path("models/", ModelsView.as_view(), name="models"),
    path("features/", FeaturesView.as_view(), name="features"),
    path("metrics/", MetricsView.as_view(), name="metrics"),
    path("drift/check/", DriftCheckView.as_view(), name="drift-check"),
]
