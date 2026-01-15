from typing import Dict, List, Optional

from django.db.models import Count
from django.views.generic import TemplateView

from forecasting.models import ForecastJob, ForecastResult, ModelVersion, Product, FeatureConfig


class DashboardView(TemplateView):
    template_name = "dashboard/index.html"

    def get_context_data(self, **kwargs) -> Dict[str, object]:
        context = super().get_context_data(**kwargs)

        products = list(Product.objects.order_by("product_id").values_list("product_id", flat=True))
        selected_product_id = self.request.GET.get("product_id") or (products[0] if products else None)
        selected_product = (
            Product.objects.filter(product_id=selected_product_id).first()
            if selected_product_id
            else None
        )

        recent_results = self._get_recent_results(selected_product)
        forecast_chart = {
            "labels": [r.forecast_date.strftime("%Y-%m-%d") for r in recent_results],
            "prediction": [r.prediction for r in recent_results],
            "lower_bound": [r.lower_bound for r in recent_results],
            "upper_bound": [r.upper_bound for r in recent_results],
        }

        job_status_chart = self._get_job_status_chart()

        context.update(
            {
                "summary": {
                    "products": Product.objects.count(),
                    "models": ModelVersion.objects.count(),
                    "jobs": ForecastJob.objects.count(),
                    "forecasts": ForecastResult.objects.count(),
                },
                "products": products,
                "selected_product_id": selected_product_id,
                "selected_product": selected_product,
                "latest_jobs": ForecastJob.objects.order_by("-created_at")[:6],
                "latest_models": ModelVersion.objects.order_by("-created_at")[:5],
                "feature_configs": FeatureConfig.objects.order_by("name")[:6],
                "forecast_chart": forecast_chart,
                "job_status_chart": job_status_chart,
                "has_forecast_data": bool(recent_results),
            }
        )
        return context

    @staticmethod
    def _get_recent_results(product: Optional[Product]) -> List[ForecastResult]:
        if not product:
            return []
        results = list(
            ForecastResult.objects.filter(product=product)
            .order_by("-forecast_date")[:24]
        )
        results.reverse()
        return results

    @staticmethod
    def _get_job_status_chart() -> Dict[str, List[object]]:
        job_counts = {status: 0 for status, _ in ForecastJob.Status.choices}
        for row in ForecastJob.objects.values("status").annotate(count=Count("id")):
            job_counts[row["status"]] = row["count"]
        labels = [label for _, label in ForecastJob.Status.choices]
        values = [job_counts[status] for status, _ in ForecastJob.Status.choices]
        return {"labels": labels, "values": values}
