from django.contrib import admin

from .models import ForecastJob, Product, ForecastResult, ModelVersion, FeatureConfig


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("product_id", "name", "category", "subcategory", "brand", "created_at")
    search_fields = ("product_id", "name", "category", "brand")


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("name", "version", "stage", "created_at")
    search_fields = ("name", "version", "run_id")


@admin.register(FeatureConfig)
class FeatureConfigAdmin(admin.ModelAdmin):
    list_display = ("name", "created_at")
    search_fields = ("name",)


@admin.register(ForecastJob)
class ForecastJobAdmin(admin.ModelAdmin):
    list_display = ("id", "job_type", "status", "created_at", "completed_at")
    list_filter = ("job_type", "status")
    search_fields = ("id",)


@admin.register(ForecastResult)
class ForecastResultAdmin(admin.ModelAdmin):
    list_display = ("product", "forecast_date", "prediction", "created_at")
    list_filter = ("forecast_date",)