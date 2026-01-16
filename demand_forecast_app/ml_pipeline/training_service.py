"""
Training service that bridges Django and the core training pipeline.
"""

from pathlib import Path
import sys
from typing import Dict, Any

from .spark_service import get_spark_session


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


def train_with_spark(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train models using the main training pipeline.
    """
    _ensure_repo_on_path()

    from main_train import main_train
    from src.data_ingestion import LocalDataLoader

    data_path = params["data_path"]
    date_col = params.get("date_col", "OrderDate")
    product_id_col = params.get("product_id_col", "item_id")
    quantity_col = params.get("quantity_col", "Quantity")
    month_end_col = params.get("month_end_col", "MonthEndDate")
    experiment_prefix = params.get("experiment_prefix", "/Django_Forecasting")
    use_advanced_features = params.get("use_advanced_features", False)

    spark = get_spark_session()
    loader = LocalDataLoader(
        spark=spark,
        date_col=date_col,
        product_id_col=product_id_col,
        quantity_col=quantity_col,
        auto_transform_m5=True,
    )
    df = loader.load(data_path)

    result = main_train(
        data_df=df,
        date_col=date_col,
        product_id_col=product_id_col,
        quantity_col=quantity_col,
        month_end_col=month_end_col,
        experiment_prefix=experiment_prefix,
        spark=spark,
        use_advanced_features=use_advanced_features,
    )
    return result
