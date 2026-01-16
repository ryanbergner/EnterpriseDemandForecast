"""
Inference service bridging Django and Spark inference.
"""

from pathlib import Path
import sys
from typing import Dict, Any

from django.utils import timezone

from .spark_service import get_spark_session
from forecasting.models import ForecastJob, ForecastResult, Product


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


def predict_with_spark(job: ForecastJob, params: Dict[str, Any]) -> None:
    _ensure_repo_on_path()

    from main_inference import main_inference
    from src.data_ingestion import LocalDataLoader

    data_path = params["data_path"]
    date_col = params.get("date_col", "OrderDate")
    product_id_col = params.get("product_id_col", "item_id")
    quantity_col = params.get("quantity_col", "Quantity")
    month_end_col = params.get("month_end_col", "MonthEndDate")
    include_intervals = params.get("include_intervals", False)

    spark = get_spark_session()
    loader = LocalDataLoader(
        spark=spark,
        date_col=date_col,
        product_id_col=product_id_col,
        quantity_col=quantity_col,
        auto_transform_m5=True,
    )
    df = loader.load(data_path)

    predictions = main_inference(
        df=df,
        date_column=date_col,
        product_id_column=product_id_col,
        quantity_column=quantity_col,
        month_end_column=month_end_col,
        target_path=None,
        ind_full_history=0,
        include_intervals=include_intervals,
    )

    # Persist results to DB
    for row in predictions.toLocalIterator():
        product, _ = Product.objects.get_or_create(product_id=getattr(row, product_id_col))
        ForecastResult.objects.create(
            product=product,
            model_version=None,
            job=job,
            forecast_date=getattr(row, month_end_col),
            prediction=float(row.prediction),
            lower_bound=float(getattr(row, "lower_bound", 0.0)) if include_intervals else None,
            upper_bound=float(getattr(row, "upper_bound", 0.0)) if include_intervals else None,
            created_at=timezone.now(),
        )
