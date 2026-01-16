from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Product",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("product_id", models.CharField(max_length=128, unique=True)),
                ("name", models.CharField(blank=True, max_length=255)),
                ("category", models.CharField(blank=True, max_length=128)),
                ("subcategory", models.CharField(blank=True, max_length=128)),
                ("brand", models.CharField(blank=True, max_length=128)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="ModelVersion",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=255)),
                ("version", models.CharField(max_length=64)),
                ("stage", models.CharField(blank=True, max_length=64)),
                ("mlflow_uri", models.CharField(blank=True, max_length=512)),
                ("run_id", models.CharField(blank=True, max_length=128)),
                ("metrics", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "unique_together": {("name", "version")},
            },
        ),
        migrations.CreateModel(
            name="FeatureConfig",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=128, unique=True)),
                ("description", models.TextField(blank=True)),
                ("config", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="ForecastJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("job_type", models.CharField(choices=[("train", "Train"), ("predict", "Predict"), ("drift", "Drift")], max_length=32)),
                ("status", models.CharField(choices=[("queued", "Queued"), ("running", "Running"), ("completed", "Completed"), ("failed", "Failed")], default="queued", max_length=32)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("parameters", models.JSONField(blank=True, default=dict)),
                ("metrics", models.JSONField(blank=True, default=dict)),
                ("error_message", models.TextField(blank=True)),
                ("model_version", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="forecasting.modelversion")),
            ],
        ),
        migrations.CreateModel(
            name="ForecastResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("forecast_date", models.DateField()),
                ("prediction", models.FloatField()),
                ("lower_bound", models.FloatField(blank=True, null=True)),
                ("upper_bound", models.FloatField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("job", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="forecasting.forecastjob")),
                ("model_version", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="forecasting.modelversion")),
                ("product", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="forecasting.product")),
            ],
            options={
                "indexes": [models.Index(fields=["forecast_date"], name="forecastin_forecast_94d0c8_idx")],
            },
        ),
    ]
