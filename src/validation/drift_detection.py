"""
Concept Drift Detection Module

Implements drift detection methods for time series forecasting:
- Statistical drift tests (Page-Hinkley, ADWIN-inspired)
- Feature distribution monitoring
- Prediction error drift detection
- Model retraining triggers

Concept drift occurs when the statistical properties of the target variable
change over time, requiring model retraining or adaptation.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, lit, when, coalesce,
    avg as spark_avg, stddev_samp, variance,
    min as spark_min, max as spark_max,
    count as spark_count, sum as spark_sum,
    lag, abs as spark_abs, sqrt,
    row_number, dense_rank
)
from pyspark.sql.window import Window
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of concept drift."""
    SUDDEN = "sudden"       # Abrupt change
    GRADUAL = "gradual"     # Slow transition
    INCREMENTAL = "incremental"  # Continuous small changes
    RECURRING = "recurring"  # Cyclic/seasonal changes


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_detected: bool
    drift_type: Optional[DriftType]
    drift_score: float
    drift_location: Optional[str]  # Date or period where drift occurred
    confidence: float
    details: Dict[str, Any]


class PageHinkleyDriftDetector:
    """
    Page-Hinkley test for concept drift detection.
    
    Detects changes in the mean of a sequence by monitoring
    cumulative deviations from the overall mean.
    
    Example:
        detector = PageHinkleyDriftDetector(threshold=50.0, alpha=0.005)
        result = detector.detect(df, value_col='prediction_error', date_col='date')
    """
    
    def __init__(
        self,
        threshold: float = 50.0,
        alpha: float = 0.005,
        min_samples: int = 30
    ):
        """
        Initialize Page-Hinkley detector.
        
        Args:
            threshold: Detection threshold (lambda)
            alpha: Magnitude tolerance
            min_samples: Minimum samples before detection starts
        """
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples
    
    def detect(
        self,
        df: DataFrame,
        value_col: str,
        date_col: str,
        product_col: Optional[str] = None
    ) -> DriftResult:
        """
        Detect drift using Page-Hinkley test.
        
        Args:
            df: DataFrame with time series data
            value_col: Column to monitor for drift
            date_col: Date column
            product_col: Optional product column for per-product detection
        
        Returns:
            DriftResult with detection outcome
        """
        logger.info("Running Page-Hinkley drift detection...")
        
        # Define partition
        if product_col:
            w = Window.partitionBy(product_col).orderBy(date_col)
            w_all = Window.partitionBy(product_col)
        else:
            w = Window.orderBy(date_col)
            w_all = Window.partitionBy(lit(1))
        
        # Calculate cumulative mean
        df = df.withColumn("_row_num", row_number().over(w))
        df = df.withColumn(
            "_cumulative_mean",
            spark_avg(col(value_col)).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            )
        )
        
        # Calculate Page-Hinkley statistic
        # m_t = sum(x_i - x_bar - alpha) for i=1 to t
        df = df.withColumn(
            "_deviation",
            col(value_col) - col("_cumulative_mean") - lit(self.alpha)
        )
        
        df = df.withColumn(
            "_cumsum",
            spark_sum(col("_deviation")).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            )
        )
        
        # Track minimum cumulative sum
        df = df.withColumn(
            "_cumsum_min",
            spark_min(col("_cumsum")).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            )
        )
        
        # Page-Hinkley statistic: PH_t = m_t - min(m_i)
        df = df.withColumn(
            "_ph_stat",
            col("_cumsum") - col("_cumsum_min")
        )
        
        # Find maximum PH statistic and corresponding date
        result_df = df.filter(col("_row_num") >= self.min_samples)
        
        max_ph = result_df.agg(
            spark_max("_ph_stat").alias("max_ph"),
            spark_avg("_ph_stat").alias("avg_ph")
        ).collect()[0]
        
        max_ph_value = max_ph["max_ph"] or 0.0
        avg_ph_value = max_ph["avg_ph"] or 0.0
        
        # Determine if drift detected
        drift_detected = max_ph_value > self.threshold
        
        # Find drift location
        drift_location = None
        if drift_detected:
            drift_row = result_df.filter(
                col("_ph_stat") == lit(max_ph_value)
            ).select(date_col).first()
            if drift_row:
                drift_location = str(drift_row[0])
        
        # Clean up
        df = df.drop(
            "_row_num", "_cumulative_mean", "_deviation",
            "_cumsum", "_cumsum_min", "_ph_stat"
        )
        
        result = DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.SUDDEN if drift_detected else None,
            drift_score=max_ph_value,
            drift_location=drift_location,
            confidence=min(1.0, max_ph_value / (self.threshold * 2)),
            details={
                "threshold": self.threshold,
                "max_ph_statistic": max_ph_value,
                "avg_ph_statistic": avg_ph_value,
            }
        )
        
        logger.info(f"Drift detection result: {result.drift_detected} (score: {result.drift_score:.2f})")
        return result


class FeatureDistributionMonitor:
    """
    Monitor feature distributions for drift detection.
    
    Compares recent feature distributions to historical baseline
    using statistical tests.
    """
    
    def __init__(
        self,
        baseline_window: int = 90,
        monitoring_window: int = 30,
        threshold: float = 0.1
    ):
        """
        Initialize feature distribution monitor.
        
        Args:
            baseline_window: Number of periods for baseline (in days/months)
            monitoring_window: Recent window for comparison
            threshold: Threshold for detecting significant drift
        """
        self.baseline_window = baseline_window
        self.monitoring_window = monitoring_window
        self.threshold = threshold
    
    def monitor(
        self,
        df: DataFrame,
        feature_cols: List[str],
        date_col: str,
        product_col: Optional[str] = None
    ) -> Dict[str, DriftResult]:
        """
        Monitor multiple features for distribution drift.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to monitor
            date_col: Date column
            product_col: Optional product column
        
        Returns:
            Dictionary mapping feature names to drift results
        """
        logger.info(f"Monitoring {len(feature_cols)} features for distribution drift...")
        
        results = {}
        
        # Determine partition
        if product_col:
            w = Window.partitionBy(product_col).orderBy(date_col)
        else:
            w = Window.orderBy(date_col)
        
        # Add row numbers for windowing
        df = df.withColumn("_row_num", row_number().over(w))
        max_row = df.agg(spark_max("_row_num")).collect()[0][0]
        
        if max_row is None or max_row < self.baseline_window + self.monitoring_window:
            logger.warning("Insufficient data for drift detection")
            return {}
        
        # Split into baseline and monitoring periods
        baseline_cutoff = max_row - self.monitoring_window - self.baseline_window
        monitoring_start = max_row - self.monitoring_window
        
        baseline_df = df.filter(
            (col("_row_num") > baseline_cutoff) &
            (col("_row_num") <= monitoring_start)
        )
        
        monitoring_df = df.filter(col("_row_num") > monitoring_start)
        
        for feature in feature_cols:
            if feature not in df.columns:
                continue
            
            # Calculate baseline statistics
            baseline_stats = baseline_df.agg(
                spark_avg(col(feature)).alias("mean"),
                stddev_samp(col(feature)).alias("std"),
                spark_min(col(feature)).alias("min"),
                spark_max(col(feature)).alias("max")
            ).collect()[0]
            
            # Calculate monitoring statistics
            monitoring_stats = monitoring_df.agg(
                spark_avg(col(feature)).alias("mean"),
                stddev_samp(col(feature)).alias("std"),
                spark_min(col(feature)).alias("min"),
                spark_max(col(feature)).alias("max")
            ).collect()[0]
            
            # Calculate drift score (normalized difference in means)
            baseline_std = baseline_stats["std"] or 1.0
            mean_diff = abs(
                (monitoring_stats["mean"] or 0) - (baseline_stats["mean"] or 0)
            )
            drift_score = mean_diff / baseline_std if baseline_std > 0 else 0.0
            
            # Check for drift
            drift_detected = drift_score > self.threshold
            
            results[feature] = DriftResult(
                drift_detected=drift_detected,
                drift_type=DriftType.GRADUAL if drift_detected else None,
                drift_score=drift_score,
                drift_location=None,
                confidence=min(1.0, drift_score / (self.threshold * 2)),
                details={
                    "baseline_mean": baseline_stats["mean"],
                    "baseline_std": baseline_stats["std"],
                    "monitoring_mean": monitoring_stats["mean"],
                    "monitoring_std": monitoring_stats["std"],
                }
            )
            
            if drift_detected:
                logger.warning(f"Drift detected in feature '{feature}' (score: {drift_score:.3f})")
        
        # Clean up
        df = df.drop("_row_num")
        
        return results


class PredictionErrorMonitor:
    """
    Monitor prediction errors for model degradation.
    
    Detects when model performance degrades significantly,
    triggering retraining recommendations.
    """
    
    def __init__(
        self,
        baseline_error: Optional[float] = None,
        degradation_threshold: float = 0.2,
        window_size: int = 30
    ):
        """
        Initialize prediction error monitor.
        
        Args:
            baseline_error: Expected baseline RMSE (None = auto-calculate)
            degradation_threshold: Relative increase threshold for degradation
            window_size: Rolling window for error calculation
        """
        self.baseline_error = baseline_error
        self.degradation_threshold = degradation_threshold
        self.window_size = window_size
    
    def monitor(
        self,
        df: DataFrame,
        actual_col: str,
        predicted_col: str,
        date_col: str
    ) -> DriftResult:
        """
        Monitor prediction errors for drift.
        
        Args:
            df: DataFrame with predictions
            actual_col: Actual values column
            predicted_col: Predicted values column
            date_col: Date column
        
        Returns:
            DriftResult indicating if model is degrading
        """
        logger.info("Monitoring prediction errors...")
        
        w = Window.orderBy(date_col)
        w_rolling = w.rowsBetween(-self.window_size + 1, 0)
        
        # Calculate squared errors
        df = df.withColumn(
            "_squared_error",
            spark_pow(col(actual_col) - col(predicted_col), lit(2.0))
        )
        
        # Calculate rolling RMSE
        df = df.withColumn(
            "_rolling_mse",
            spark_avg(col("_squared_error")).over(w_rolling)
        )
        
        df = df.withColumn(
            "_rolling_rmse",
            sqrt(col("_rolling_mse"))
        )
        
        # Get baseline RMSE (first window)
        if self.baseline_error is None:
            baseline_row = df.orderBy(date_col).limit(self.window_size).agg(
                sqrt(spark_avg("_squared_error")).alias("baseline_rmse")
            ).collect()[0]
            baseline_rmse = baseline_row["baseline_rmse"]
        else:
            baseline_rmse = self.baseline_error
        
        # Get recent RMSE
        recent_row = df.orderBy(col(date_col).desc()).limit(1).select(
            "_rolling_rmse"
        ).collect()[0]
        recent_rmse = recent_row["_rolling_rmse"]
        
        # Calculate degradation
        if baseline_rmse and baseline_rmse > 0:
            degradation = (recent_rmse - baseline_rmse) / baseline_rmse
        else:
            degradation = 0.0
        
        # Determine if degraded
        degraded = degradation > self.degradation_threshold
        
        # Clean up
        df = df.drop("_squared_error", "_rolling_mse", "_rolling_rmse")
        
        result = DriftResult(
            drift_detected=degraded,
            drift_type=DriftType.GRADUAL if degraded else None,
            drift_score=degradation,
            drift_location=None,
            confidence=min(1.0, degradation / (self.degradation_threshold * 2)),
            details={
                "baseline_rmse": baseline_rmse,
                "recent_rmse": recent_rmse,
                "degradation_pct": degradation * 100,
                "threshold_pct": self.degradation_threshold * 100,
            }
        )
        
        if degraded:
            logger.warning(
                f"Model degradation detected! "
                f"RMSE increased by {degradation*100:.1f}% "
                f"(threshold: {self.degradation_threshold*100:.0f}%)"
            )
        
        return result


def from_pyspark_sql_functions(col, spark_pow, lit):
    """Import helper for cleaner code."""
    pass


def recommend_retraining(
    drift_results: List[DriftResult],
    min_confidence: float = 0.7
) -> Tuple[bool, str]:
    """
    Recommend whether model retraining is needed based on drift detection.
    
    Args:
        drift_results: List of drift detection results
        min_confidence: Minimum confidence to consider drift significant
    
    Returns:
        Tuple of (should_retrain, reason)
    """
    significant_drifts = [
        r for r in drift_results
        if r.drift_detected and r.confidence >= min_confidence
    ]
    
    if not significant_drifts:
        return False, "No significant drift detected"
    
    # Check for sudden drift (high priority)
    sudden_drifts = [
        r for r in significant_drifts
        if r.drift_type == DriftType.SUDDEN
    ]
    
    if sudden_drifts:
        return True, f"Sudden drift detected in {len(sudden_drifts)} indicator(s)"
    
    # Check for multiple gradual drifts
    if len(significant_drifts) >= 3:
        return True, f"Multiple drifts detected ({len(significant_drifts)} indicators)"
    
    # Check for high-confidence drift
    high_confidence = [r for r in significant_drifts if r.confidence > 0.9]
    if high_confidence:
        return True, f"High-confidence drift detected (confidence: {high_confidence[0].confidence:.2f})"
    
    return False, f"Drift detected but below retraining threshold ({len(significant_drifts)} indicators)"
