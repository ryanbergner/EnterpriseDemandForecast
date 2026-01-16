from django.db import models


class Product(models.Model):
    product_id = models.CharField(max_length=128, unique=True)
    name = models.CharField(max_length=255, blank=True)
    category = models.CharField(max_length=128, blank=True)
    subcategory = models.CharField(max_length=128, blank=True)
    brand = models.CharField(max_length=128, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.product_id


class ModelVersion(models.Model):
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=64)
    stage = models.CharField(max_length=64, blank=True)
    mlflow_uri = models.CharField(max_length=512, blank=True)
    run_id = models.CharField(max_length=128, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("name", "version")

    def __str__(self) -> str:
        return f"{self.name}:{self.version}"


class FeatureConfig(models.Model):
    name = models.CharField(max_length=128, unique=True)
    description = models.TextField(blank=True)
    config = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class ForecastJob(models.Model):
    class JobType(models.TextChoices):
        TRAIN = "train", "Train"
        PREDICT = "predict", "Predict"
        DRIFT = "drift", "Drift"

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    job_type = models.CharField(max_length=32, choices=JobType.choices)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.QUEUED)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    parameters = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    model_version = models.ForeignKey(ModelVersion, null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self) -> str:
        return f"{self.job_type}:{self.id} ({self.status})"


class ForecastResult(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    model_version = models.ForeignKey(ModelVersion, null=True, blank=True, on_delete=models.SET_NULL)
    job = models.ForeignKey(ForecastJob, null=True, blank=True, on_delete=models.SET_NULL)
    forecast_date = models.DateField()
    prediction = models.FloatField()
    lower_bound = models.FloatField(null=True, blank=True)
    upper_bound = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["forecast_date"]),
        ]

    def __str__(self) -> str:
        return f"{self.product.product_id} @ {self.forecast_date}"
