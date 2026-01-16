"""
Spark session management for Django services.
"""

from threading import Lock
from typing import Optional

from pyspark.sql import SparkSession


_lock = Lock()
_spark: Optional[SparkSession] = None


def get_spark_session(app_name: str = "DemandForecastService") -> SparkSession:
    """
    Get or create a singleton SparkSession.
    """
    global _spark
    if _spark is None:
        with _lock:
            if _spark is None:
                _spark = (
                    SparkSession.builder
                    .appName(app_name)
                    .config("spark.sql.adaptive.enabled", "true")
                    .getOrCreate()
                )
    return _spark


def stop_spark_session() -> None:
    """
    Stop and clear the singleton SparkSession.
    """
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
