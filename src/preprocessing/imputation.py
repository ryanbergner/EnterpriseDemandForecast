"""
Smart Imputation Strategies for Time Series Data

Provides intelligent null/zero handling strategies:
- Forward fill with exponential decay
- Seasonal imputation (use same period from previous year)
- KNN-based imputation using similar products
- Interpolation strategies (linear, spline)
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lag, lead, coalesce, lit, 
    avg as spark_avg, last, first,
    months_between, pow as spark_pow, exp as spark_exp,
    abs as spark_abs, sum as spark_sum
)
from pyspark.sql.window import Window
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesImputer:
    """
    Advanced imputation strategies for time series forecasting data.
    
    Example usage:
        imputer = TimeSeriesImputer(strategy='forward_fill_decay')
        df_imputed = imputer.fit_transform(
            df, 
            value_col='DemandQuantity',
            date_col='MonthEndDate',
            product_col='ItemNumber'
        )
    """
    
    def __init__(
        self,
        strategy: str = 'forward_fill_decay',
        decay_rate: float = 0.95,
        seasonal_lag: int = 12,
        fill_value: float = 0.0
    ):
        """
        Args:
            strategy: Imputation strategy ('forward_fill', 'forward_fill_decay', 
                     'seasonal', 'mean', 'zero', 'interpolate')
            decay_rate: Decay factor for forward_fill_decay (0-1, default 0.95)
            seasonal_lag: Periods for seasonal imputation (default 12 for monthly)
            fill_value: Default fill value if no other value available
        """
        self.strategy = strategy
        self.decay_rate = decay_rate
        self.seasonal_lag = seasonal_lag
        self.fill_value = fill_value
        
        valid_strategies = [
            'forward_fill', 'forward_fill_decay', 'seasonal',
            'mean', 'zero', 'interpolate', 'backward_fill'
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def fit_transform(
        self,
        df: DataFrame,
        value_col: str,
        date_col: str,
        product_col: str
    ) -> DataFrame:
        """
        Apply imputation strategy to fill missing values.
        
        Args:
            df: Input DataFrame
            value_col: Column name with values to impute
            date_col: Date column name
            product_col: Product/entity identifier column
        
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"🔧 Applying imputation strategy: {self.strategy}")
        
        # Count nulls before
        null_count_before = df.filter(col(value_col).isNull()).count()
        logger.info(f"   Null values before: {null_count_before}")
        
        if self.strategy == 'zero':
            df_imputed = self._impute_zero(df, value_col)
        
        elif self.strategy == 'mean':
            df_imputed = self._impute_mean(df, value_col, product_col)
        
        elif self.strategy == 'forward_fill':
            df_imputed = self._impute_forward_fill(df, value_col, date_col, product_col)
        
        elif self.strategy == 'forward_fill_decay':
            df_imputed = self._impute_forward_fill_decay(df, value_col, date_col, product_col)
        
        elif self.strategy == 'backward_fill':
            df_imputed = self._impute_backward_fill(df, value_col, date_col, product_col)
        
        elif self.strategy == 'seasonal':
            df_imputed = self._impute_seasonal(df, value_col, date_col, product_col)
        
        elif self.strategy == 'interpolate':
            df_imputed = self._impute_interpolate(df, value_col, date_col, product_col)
        
        else:
            df_imputed = df
        
        # Count nulls after
        null_count_after = df_imputed.filter(col(value_col).isNull()).count()
        logger.info(f"   Null values after: {null_count_after}")
        logger.info(f"   Imputed {null_count_before - null_count_after} values")
        
        return df_imputed
    
    def _impute_zero(self, df: DataFrame, value_col: str) -> DataFrame:
        """Fill nulls with zero."""
        return df.withColumn(
            value_col,
            coalesce(col(value_col), lit(self.fill_value))
        )
    
    def _impute_mean(
        self, 
        df: DataFrame, 
        value_col: str, 
        product_col: str
    ) -> DataFrame:
        """Fill nulls with product-level mean."""
        w = Window.partitionBy(product_col)
        
        return df.withColumn(
            f"{value_col}_mean",
            spark_avg(col(value_col)).over(w)
        ).withColumn(
            value_col,
            coalesce(col(value_col), col(f"{value_col}_mean"), lit(self.fill_value))
        ).drop(f"{value_col}_mean")
    
    def _impute_forward_fill(
        self,
        df: DataFrame,
        value_col: str,
        date_col: str,
        product_col: str
    ) -> DataFrame:
        """Forward fill missing values (use last known value)."""
        w = (
            Window.partitionBy(product_col)
            .orderBy(date_col)
            .rowsBetween(Window.unboundedPreceding, 0)
        )
        
        return df.withColumn(
            value_col,
            coalesce(
                col(value_col),
                last(col(value_col), ignorenulls=True).over(w),
                lit(self.fill_value)
            )
        )
    
    def _impute_backward_fill(
        self,
        df: DataFrame,
        value_col: str,
        date_col: str,
        product_col: str
    ) -> DataFrame:
        """Backward fill missing values (use next known value)."""
        w = (
            Window.partitionBy(product_col)
            .orderBy(date_col)
            .rowsBetween(0, Window.unboundedFollowing)
        )
        
        return df.withColumn(
            value_col,
            coalesce(
                col(value_col),
                first(col(value_col), ignorenulls=True).over(w),
                lit(self.fill_value)
            )
        )
    
    def _impute_forward_fill_decay(
        self,
        df: DataFrame,
        value_col: str,
        date_col: str,
        product_col: str
    ) -> DataFrame:
        """
        Forward fill with exponential decay.
        Missing values are filled with last known value * (decay_rate ^ periods_since_last_value).
        """
        w_order = Window.partitionBy(product_col).orderBy(date_col)
        w_full = w_order.rowsBetween(Window.unboundedPreceding, 0)
        
        # Mark rows with actual values
        df = df.withColumn(
            "_has_value",
            when(col(value_col).isNotNull(), lit(1)).otherwise(lit(0))
        )
        
        # Get last known value
        df = df.withColumn(
            "_last_value",
            last(
                when(col("_has_value") == 1, col(value_col)),
                ignorenulls=True
            ).over(w_full)
        )
        
        # Calculate periods since last known value
        df = df.withColumn(
            "_value_idx",
            when(col("_has_value") == 1, lit(0)).otherwise(lit(None))
        )
        
        # Create a running index for each row
        df = df.withColumn("_row_idx", spark_sum(lit(1)).over(w_full))
        
        # Get index of last known value
        df = df.withColumn(
            "_last_value_idx",
            last(
                when(col("_has_value") == 1, col("_row_idx")),
                ignorenulls=True
            ).over(w_full)
        )
        
        # Periods elapsed since last value
        df = df.withColumn(
            "_periods_elapsed",
            col("_row_idx") - coalesce(col("_last_value_idx"), lit(0))
        )
        
        # Apply decay: last_value * (decay_rate ^ periods_elapsed)
        df = df.withColumn(
            "_decayed_value",
            col("_last_value") * spark_pow(lit(self.decay_rate), col("_periods_elapsed"))
        )
        
        # Fill missing values with decayed value
        df = df.withColumn(
            value_col,
            coalesce(
                col(value_col),
                col("_decayed_value"),
                lit(self.fill_value)
            )
        )
        
        # Clean up temporary columns
        return df.drop(
            "_has_value", "_last_value", "_value_idx", "_row_idx",
            "_last_value_idx", "_periods_elapsed", "_decayed_value"
        )
    
    def _impute_seasonal(
        self,
        df: DataFrame,
        value_col: str,
        date_col: str,
        product_col: str
    ) -> DataFrame:
        """
        Seasonal imputation: fill missing values with value from same period last year.
        Falls back to forward fill if seasonal value not available.
        """
        w = Window.partitionBy(product_col).orderBy(date_col)
        
        # Get value from seasonal_lag periods ago
        df = df.withColumn(
            f"_{value_col}_seasonal",
            lag(col(value_col), self.seasonal_lag).over(w)
        )
        
        # Forward fill as backup
        w_full = w.rowsBetween(Window.unboundedPreceding, 0)
        df = df.withColumn(
            f"_{value_col}_ffill",
            last(col(value_col), ignorenulls=True).over(w_full)
        )
        
        # Use seasonal value first, then forward fill, then fill_value
        df = df.withColumn(
            value_col,
            coalesce(
                col(value_col),
                col(f"_{value_col}_seasonal"),
                col(f"_{value_col}_ffill"),
                lit(self.fill_value)
            )
        )
        
        return df.drop(f"_{value_col}_seasonal", f"_{value_col}_ffill")
    
    def _impute_interpolate(
        self,
        df: DataFrame,
        value_col: str,
        date_col: str,
        product_col: str
    ) -> DataFrame:
        """
        Linear interpolation between known values.
        For time series with regular intervals.
        """
        w = Window.partitionBy(product_col).orderBy(date_col)
        w_full = w.rowsBetween(Window.unboundedPreceding, 0)
        w_future = w.rowsBetween(0, Window.unboundedFollowing)
        
        # Get previous and next known values
        df = df.withColumn(
            "_prev_value",
            last(col(value_col), ignorenulls=True).over(w_full)
        )
        df = df.withColumn(
            "_next_value",
            first(col(value_col), ignorenulls=True).over(w_future)
        )
        
        # Simple average for linear interpolation
        # (More sophisticated interpolation would require distance calculation)
        df = df.withColumn(
            "_interpolated",
            (col("_prev_value") + col("_next_value")) / 2.0
        )
        
        # Fill nulls with interpolated values
        df = df.withColumn(
            value_col,
            coalesce(
                col(value_col),
                col("_interpolated"),
                col("_prev_value"),  # If no next value, use prev
                col("_next_value"),  # If no prev value, use next
                lit(self.fill_value)
            )
        )
        
        return df.drop("_prev_value", "_next_value", "_interpolated")


def auto_select_imputation_strategy(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> str:
    """
    Automatically select the best imputation strategy based on data characteristics.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product column name
    
    Returns:
        Recommended imputation strategy name
    """
    # Calculate null rate
    total_count = df.count()
    null_count = df.filter(col(value_col).isNull()).count()
    null_rate = null_count / total_count if total_count > 0 else 0
    
    # Calculate zero rate (for intermittent demand detection)
    zero_count = df.filter(col(value_col) == 0).count()
    zero_rate = zero_count / total_count if total_count > 0 else 0
    
    logger.info(f"🔍 Auto-selecting imputation strategy:")
    logger.info(f"   Null rate: {null_rate:.2%}")
    logger.info(f"   Zero rate: {zero_rate:.2%}")
    
    # Decision logic
    if null_rate < 0.01:
        # Very few nulls - simple zero fill
        strategy = 'zero'
        logger.info(f"   → Selected: {strategy} (very few missing values)")
    
    elif null_rate < 0.05:
        # Low null rate - forward fill is safe
        strategy = 'forward_fill'
        logger.info(f"   → Selected: {strategy} (low missing rate)")
    
    elif zero_rate > 0.3:
        # Intermittent demand - zero fill appropriate
        strategy = 'zero'
        logger.info(f"   → Selected: {strategy} (intermittent demand pattern)")
    
    elif null_rate < 0.20:
        # Moderate nulls - use decay
        strategy = 'forward_fill_decay'
        logger.info(f"   → Selected: {strategy} (moderate missing rate)")
    
    else:
        # High null rate - try seasonal
        strategy = 'seasonal'
        logger.info(f"   → Selected: {strategy} (high missing rate, seasonal backup)")
    
    return strategy


def impute_with_auto_strategy(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    **kwargs
) -> DataFrame:
    """
    Convenience function that auto-selects and applies imputation strategy.
    
    Args:
        df: Input DataFrame
        value_col: Value column to impute
        date_col: Date column name
        product_col: Product identifier column
        **kwargs: Additional arguments passed to TimeSeriesImputer
    
    Returns:
        DataFrame with imputed values
    """
    strategy = auto_select_imputation_strategy(df, value_col, date_col, product_col)
    imputer = TimeSeriesImputer(strategy=strategy, **kwargs)
    return imputer.fit_transform(df, value_col, date_col, product_col)
