"""
Data Quality Validation Module

Provides automated checks for time series data quality including:
- Missing data detection and pattern analysis
- Outlier detection using multiple methods
- Seasonality detection and strength measurement
- Distribution drift detection between train/test
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, count, when, isnan, isnull, 
    avg as spark_avg, stddev as spark_stddev,
    min as spark_min, max as spark_max,
    lag, lead, abs as spark_abs, datediff
)
from pyspark.sql.window import Window
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Comprehensive data quality validation for time series forecasting.
    
    Example usage:
        validator = DataQualityValidator()
        report = validator.validate(df, date_col="MonthEndDate", 
                                    product_col="ItemNumber", 
                                    target_col="DemandQuantity")
        validator.print_report(report)
    """
    
    def __init__(self, outlier_threshold: float = 3.0, missing_threshold: float = 0.05):
        """
        Args:
            outlier_threshold: Z-score threshold for outlier detection (default 3.0)
            missing_threshold: Max acceptable missing data rate (default 5%)
        """
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
    
    def validate(
        self,
        df: DataFrame,
        date_col: str,
        product_col: str,
        target_col: str
    ) -> Dict[str, Any]:
        """
        Run comprehensive data quality checks.
        
        Returns:
            Dictionary containing validation results and recommendations
        """
        logger.info("🔍 Starting Data Quality Validation...")
        
        report = {
            "missing_data": self._check_missing_data(df, target_col),
            "outliers": self._detect_outliers(df, target_col, product_col),
            "gaps": self._check_time_gaps(df, date_col, product_col),
            "zeros": self._check_zero_patterns(df, target_col, product_col),
            "statistics": self._compute_statistics(df, target_col),
            "seasonality": self._check_seasonality_strength(df, date_col, target_col, product_col)
        }
        
        # Add overall status
        report["status"] = self._determine_overall_status(report)
        report["recommendations"] = self._generate_recommendations(report)
        
        logger.info(f"✅ Validation Complete. Status: {report['status']}")
        return report
    
    def _check_missing_data(self, df: DataFrame, target_col: str) -> Dict[str, Any]:
        """Check for missing/null values in target column."""
        total_count = df.count()
        if total_count == 0:
            return {"error": "DataFrame is empty"}
        
        null_count = df.filter(
            col(target_col).isNull() | isnan(col(target_col))
        ).count()
        
        missing_rate = null_count / total_count
        
        return {
            "total_rows": total_count,
            "missing_rows": null_count,
            "missing_rate": missing_rate,
            "status": "PASS" if missing_rate <= self.missing_threshold else "WARNING"
        }
    
    def _detect_outliers(
        self, 
        df: DataFrame, 
        target_col: str, 
        product_col: str
    ) -> Dict[str, Any]:
        """Detect outliers using IQR and Z-score methods."""
        # Compute statistics
        stats = df.agg(
            spark_avg(col(target_col)).alias("mean"),
            spark_stddev(col(target_col)).alias("stddev")
        ).collect()[0]
        
        mean_val = stats["mean"]
        stddev_val = stats["stddev"]
        
        if stddev_val is None or stddev_val == 0:
            return {"status": "SKIP", "reason": "Insufficient variance for outlier detection"}
        
        # Z-score outliers
        zscore_outliers = df.filter(
            spark_abs((col(target_col) - mean_val) / stddev_val) > self.outlier_threshold
        ).count()
        
        # Approximate quantiles for IQR method
        quantiles = df.approxQuantile(target_col, [0.25, 0.75], 0.01)
        if len(quantiles) == 2:
            q1, q3 = quantiles
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = df.filter(
                (col(target_col) < lower_bound) | (col(target_col) > upper_bound)
            ).count()
        else:
            iqr_outliers = None
            lower_bound = None
            upper_bound = None
        
        total_count = df.count()
        
        return {
            "zscore_outliers": zscore_outliers,
            "zscore_rate": zscore_outliers / total_count if total_count > 0 else 0,
            "iqr_outliers": iqr_outliers,
            "iqr_rate": iqr_outliers / total_count if iqr_outliers and total_count > 0 else None,
            "bounds": {"lower": lower_bound, "upper": upper_bound},
            "status": "PASS" if (zscore_outliers / total_count) < 0.05 else "WARNING"
        }
    
    def _check_time_gaps(
        self, 
        df: DataFrame, 
        date_col: str, 
        product_col: str
    ) -> Dict[str, Any]:
        """Check for unexpected time gaps in the series."""
        w = Window.partitionBy(product_col).orderBy(date_col)
        
        df_with_gap = df.withColumn(
            "prev_date", lag(col(date_col), 1).over(w)
        ).withColumn(
            "days_gap", datediff(col(date_col), col("prev_date"))
        )
        
        # Expected gap for monthly data is ~30 days
        gaps = df_with_gap.filter(
            (col("days_gap").isNotNull()) & 
            ((col("days_gap") > 45) | (col("days_gap") < 20))
        )
        
        gap_count = gaps.count()
        total_transitions = df_with_gap.filter(col("prev_date").isNotNull()).count()
        
        return {
            "unexpected_gaps": gap_count,
            "gap_rate": gap_count / total_transitions if total_transitions > 0 else 0,
            "status": "PASS" if gap_count == 0 else "WARNING"
        }
    
    def _check_zero_patterns(
        self, 
        df: DataFrame, 
        target_col: str, 
        product_col: str
    ) -> Dict[str, Any]:
        """Analyze zero-value patterns (important for intermittent demand)."""
        total_count = df.count()
        zero_count = df.filter(col(target_col) == 0).count()
        zero_rate = zero_count / total_count if total_count > 0 else 0
        
        # Check for consecutive zeros
        w = Window.partitionBy(product_col).orderBy("MonthEndDate" if "MonthEndDate" in df.columns else product_col)
        
        df_with_prev = df.withColumn(
            "prev_value", lag(col(target_col), 1).over(w)
        )
        
        consecutive_zeros = df_with_prev.filter(
            (col(target_col) == 0) & (col("prev_value") == 0)
        ).count()
        
        return {
            "zero_count": zero_count,
            "zero_rate": zero_rate,
            "consecutive_zeros": consecutive_zeros,
            "demand_type": self._classify_demand_pattern(zero_rate),
            "status": "PASS"
        }
    
    def _classify_demand_pattern(self, zero_rate: float) -> str:
        """Classify demand pattern based on zero rate."""
        if zero_rate < 0.1:
            return "Regular"
        elif zero_rate < 0.3:
            return "Intermittent"
        elif zero_rate < 0.5:
            return "Sporadic"
        else:
            return "Very Sparse"
    
    def _compute_statistics(self, df: DataFrame, target_col: str) -> Dict[str, Any]:
        """Compute basic statistics for the target variable."""
        stats = df.agg(
            count(col(target_col)).alias("count"),
            spark_avg(col(target_col)).alias("mean"),
            spark_stddev(col(target_col)).alias("stddev"),
            spark_min(col(target_col)).alias("min"),
            spark_max(col(target_col)).alias("max")
        ).collect()[0]
        
        return {
            "count": stats["count"],
            "mean": float(stats["mean"]) if stats["mean"] else None,
            "stddev": float(stats["stddev"]) if stats["stddev"] else None,
            "min": float(stats["min"]) if stats["min"] else None,
            "max": float(stats["max"]) if stats["max"] else None,
            "cv": float(stats["stddev"] / stats["mean"]) if stats["mean"] and stats["stddev"] else None
        }
    
    def _check_seasonality_strength(
        self, 
        df: DataFrame, 
        date_col: str, 
        target_col: str, 
        product_col: str
    ) -> Dict[str, Any]:
        """
        Estimate seasonality strength using variance decomposition.
        Compare within-season variance to between-season variance.
        """
        from pyspark.sql.functions import month, variance as spark_var
        
        # Add month column
        df_with_month = df.withColumn("month", month(col(date_col)))
        
        # Total variance
        total_var = df_with_month.agg(
            spark_var(col(target_col)).alias("total_var")
        ).collect()[0]["total_var"]
        
        if total_var is None or total_var == 0:
            return {"status": "SKIP", "reason": "Insufficient variance"}
        
        # Within-season variance (average variance within each month)
        within_var = df_with_month.groupBy("month").agg(
            spark_var(col(target_col)).alias("month_var")
        ).agg(
            spark_avg("month_var").alias("avg_within_var")
        ).collect()[0]["avg_within_var"]
        
        if within_var is None:
            within_var = total_var
        
        # Seasonality strength: proportion of variance explained by seasonal pattern
        seasonality_strength = max(0, 1 - (within_var / total_var)) if total_var > 0 else 0
        
        # Classify strength
        if seasonality_strength > 0.4:
            strength_label = "Strong"
        elif seasonality_strength > 0.2:
            strength_label = "Moderate"
        elif seasonality_strength > 0.05:
            strength_label = "Weak"
        else:
            strength_label = "None"
        
        return {
            "seasonality_strength": seasonality_strength,
            "strength_label": strength_label,
            "total_variance": total_var,
            "within_season_variance": within_var,
            "status": "PASS"
        }
    
    def _determine_overall_status(self, report: Dict[str, Any]) -> str:
        """Determine overall data quality status."""
        statuses = []
        for key, value in report.items():
            if isinstance(value, dict) and "status" in value:
                statuses.append(value["status"])
        
        if "WARNING" in statuses:
            return "WARNING"
        elif "PASS" in statuses:
            return "PASS"
        else:
            return "UNKNOWN"
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Missing data
        if report["missing_data"]["status"] == "WARNING":
            recommendations.append(
                f"⚠️ High missing data rate ({report['missing_data']['missing_rate']:.1%}). "
                "Consider imputation strategies or filtering products with insufficient data."
            )
        
        # Outliers
        if report["outliers"]["status"] == "WARNING":
            recommendations.append(
                f"⚠️ High outlier rate ({report['outliers']['zscore_rate']:.1%}). "
                "Consider outlier treatment (capping, transformation) or investigate data quality issues."
            )
        
        # Time gaps
        if report["gaps"]["status"] == "WARNING":
            recommendations.append(
                "⚠️ Unexpected time gaps detected. Review date continuity and consider gap filling."
            )
        
        # Zeros (intermittent demand)
        zero_rate = report["zeros"]["zero_rate"]
        if zero_rate > 0.3:
            recommendations.append(
                f"📊 Intermittent demand pattern detected ({zero_rate:.1%} zeros). "
                "Consider specialized models (Croston, ADIDA) or separate treatment for sparse products."
            )
        
        # Seasonality
        seasonality = report["seasonality"]
        if seasonality.get("strength_label") in ["Strong", "Moderate"]:
            recommendations.append(
                f"📈 {seasonality['strength_label']} seasonality detected "
                f"(strength={seasonality['seasonality_strength']:.2f}). "
                "Ensure seasonal features and models are used."
            )
        
        if not recommendations:
            recommendations.append("✅ No major data quality issues detected. Data looks good!")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print a formatted validation report."""
        print("\n" + "="*70)
        print("📊 DATA QUALITY VALIDATION REPORT")
        print("="*70)
        
        # Overall Status
        status_emoji = "✅" if report["status"] == "PASS" else "⚠️"
        print(f"\n{status_emoji} Overall Status: {report['status']}")
        
        # Missing Data
        print(f"\n📉 Missing Data:")
        md = report["missing_data"]
        if "error" not in md:
            print(f"   Total Rows: {md['total_rows']:,}")
            print(f"   Missing Rows: {md['missing_rows']:,} ({md['missing_rate']:.2%})")
            print(f"   Status: {md['status']}")
        
        # Outliers
        print(f"\n🎯 Outliers:")
        out = report["outliers"]
        if "reason" not in out:
            print(f"   Z-score outliers: {out['zscore_outliers']:,} ({out['zscore_rate']:.2%})")
            if out["iqr_outliers"] is not None:
                print(f"   IQR outliers: {out['iqr_outliers']:,} ({out['iqr_rate']:.2%})")
            print(f"   Status: {out['status']}")
        
        # Time Gaps
        print(f"\n⏰ Time Gaps:")
        gaps = report["gaps"]
        print(f"   Unexpected gaps: {gaps['unexpected_gaps']:,} ({gaps['gap_rate']:.2%})")
        print(f"   Status: {gaps['status']}")
        
        # Zeros
        print(f"\n🔢 Zero Values:")
        zeros = report["zeros"]
        print(f"   Zero count: {zeros['zero_count']:,} ({zeros['zero_rate']:.2%})")
        print(f"   Consecutive zeros: {zeros['consecutive_zeros']:,}")
        print(f"   Demand Type: {zeros['demand_type']}")
        
        # Statistics
        print(f"\n📊 Statistics:")
        stats = report["statistics"]
        if stats["mean"] is not None:
            print(f"   Mean: {stats['mean']:.2f}")
            print(f"   StdDev: {stats['stddev']:.2f}")
            print(f"   CV: {stats['cv']:.2f}")
            print(f"   Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        # Seasonality
        print(f"\n🌊 Seasonality:")
        seas = report["seasonality"]
        if "reason" not in seas:
            print(f"   Strength: {seas['seasonality_strength']:.2f} ({seas['strength_label']})")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*70 + "\n")


def validate_train_test_distribution(
    train_df: DataFrame,
    test_df: DataFrame,
    target_col: str,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Check for distribution drift between train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_col: Target column name
        threshold: Maximum acceptable relative difference in statistics
    
    Returns:
        Dictionary with drift analysis results
    """
    train_stats = train_df.agg(
        spark_avg(col(target_col)).alias("mean"),
        spark_stddev(col(target_col)).alias("stddev")
    ).collect()[0]
    
    test_stats = test_df.agg(
        spark_avg(col(target_col)).alias("mean"),
        spark_stddev(col(target_col)).alias("stddev")
    ).collect()[0]
    
    mean_diff = abs(train_stats["mean"] - test_stats["mean"]) / train_stats["mean"]
    stddev_diff = abs(train_stats["stddev"] - test_stats["stddev"]) / train_stats["stddev"]
    
    drift_detected = mean_diff > threshold or stddev_diff > threshold
    
    return {
        "train_mean": train_stats["mean"],
        "test_mean": test_stats["mean"],
        "mean_diff": mean_diff,
        "train_stddev": train_stats["stddev"],
        "test_stddev": test_stats["stddev"],
        "stddev_diff": stddev_diff,
        "drift_detected": drift_detected,
        "status": "WARNING" if drift_detected else "PASS"
    }
