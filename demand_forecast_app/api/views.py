"""
API views for training, prediction, model management, and drift checks.
"""

from django.db.models import Count
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from forecasting.models import ForecastJob, FeatureConfig, ModelVersion, Product
from forecasting.services import (
    TrainingService,
    InferenceService,
    ValidationService,
)
from .serializers import (
    TrainRequestSerializer,
    PredictRequestSerializer,
    ForecastJobSerializer,
    ModelVersionSerializer,
)


class TrainView(APIView):
    def post(self, request):
        serializer = TrainRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = TrainingService.start_training(**serializer.validated_data)
        return Response(ForecastJobSerializer(job).data, status=status.HTTP_202_ACCEPTED)


class TrainStatusView(APIView):
    def get(self, request, job_id: int):
        try:
            job = ForecastJob.objects.get(id=job_id)
        except ForecastJob.DoesNotExist:
            return Response({"detail": "Job not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(ForecastJobSerializer(job).data, status=status.HTTP_200_OK)


class PredictView(APIView):
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = InferenceService.start_prediction(**serializer.validated_data)
        return Response(ForecastJobSerializer(job).data, status=status.HTTP_202_ACCEPTED)


class ModelsView(APIView):
    def get(self, request):
        models = ModelVersion.objects.all().order_by("-created_at")
        return Response(ModelVersionSerializer(models, many=True).data)


class FeaturesView(APIView):
    def get(self, request):
        features = FeatureConfig.objects.all().order_by("name")
        data = [{"id": f.id, "name": f.name, "description": f.description, "config": f.config} for f in features]
        return Response(data)


class MetricsView(APIView):
    def get(self, request):
        models = ModelVersion.objects.all().order_by("-created_at")
        data = [
            {
                "name": m.name,
                "version": m.version,
                "stage": m.stage,
                "metrics": m.metrics,
            }
            for m in models
        ]
        return Response(data)


class DriftCheckView(APIView):
    def post(self, request):
        result = ValidationService.check_drift(request.data)
        return Response(result, status=status.HTTP_200_OK)


class HierarchyView(APIView):
    def get(self, request):
        rows = (
            Product.objects.values("category", "subcategory")
            .annotate(product_count=Count("id"))
            .order_by("category", "subcategory")
        )
        data = [
            {
                "category": row["category"] or "Uncategorized",
                "subcategory": row["subcategory"] or "Uncategorized",
                "product_count": row["product_count"],
            }
            for row in rows
        ]
        return Response(data, status=status.HTTP_200_OK)
