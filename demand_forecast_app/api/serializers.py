"""
DRF serializers for API requests/responses.
"""

from rest_framework import serializers

from forecasting.models import ForecastJob, ForecastResult, ModelVersion


class TrainRequestSerializer(serializers.Serializer):
    data_path = serializers.CharField()
    experiment_prefix = serializers.CharField(required=False, default="/Django_Forecasting")
    date_col = serializers.CharField(required=False, default="OrderDate")
    product_id_col = serializers.CharField(required=False, default="item_id")
    quantity_col = serializers.CharField(required=False, default="Quantity")
    month_end_col = serializers.CharField(required=False, default="MonthEndDate")
    use_advanced_features = serializers.BooleanField(required=False, default=False)


class PredictRequestSerializer(serializers.Serializer):
    data_path = serializers.CharField()
    model_name = serializers.CharField()
    model_alias = serializers.CharField(required=False, default="champion")
    date_col = serializers.CharField(required=False, default="OrderDate")
    product_id_col = serializers.CharField(required=False, default="item_id")
    quantity_col = serializers.CharField(required=False, default="Quantity")
    month_end_col = serializers.CharField(required=False, default="MonthEndDate")
    include_intervals = serializers.BooleanField(required=False, default=False)


class ForecastJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastJob
        fields = [
            "id",
            "job_type",
            "status",
            "created_at",
            "started_at",
            "completed_at",
            "parameters",
            "metrics",
            "error_message",
        ]


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = [
            "id",
            "name",
            "version",
            "stage",
            "mlflow_uri",
            "run_id",
            "metrics",
            "created_at",
        ]


class ForecastResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastResult
        fields = [
            "id",
            "product",
            "forecast_date",
            "prediction",
            "lower_bound",
            "upper_bound",
            "model_version",
            "created_at",
        ]
