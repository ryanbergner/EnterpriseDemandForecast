"""
Exponentially Weighted Moving Average (EWMA) Features

Provides adaptive features that give more weight to recent observations:
- EWMA with different decay rates (alpha values)
- Exponentially weighted standard deviation (volatility)
- EWMA-based momentum indicators
- Adaptive trend following features
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lag, when, lit, coalesce,
    pow as spark_pow, sqrt, abs as spark_abs,
    avg as spark_avg, sum as spark_sum
)
from pyspark.sql.window import Window
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_ewma_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    alpha_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
) -> DataFrame:
    """
    Add Exponentially Weighted Moving Average features.
    
    EWMA formula: EWMA_t = alpha * value_t + (1 - alpha) * EWMA_(t-1)
    
    - Lower alpha (e.g., 0.1): Slower response, smoother, more historical memory
    - Higher alpha (e.g., 0.9): Faster response, tracks recent changes closely
    
    Args:
        df: Input DataFrame
        value_col: Column name for values
        date_col: Date column name
        product_col: Product identifier column
        alpha_values: List of alpha (smoothing) parameters to use
    
    Returns:
        DataFrame with EWMA features
    
    Example:
        df_with_ewma = add_ewma_features(
            df,
            value_col='DemandQuantity',
            date_col='MonthEndDate',
            product_col='ItemNumber',
            alpha_values=[0.3, 0.7]
        )
    """
    logger.info(f"📊 Adding EWMA features with alpha values: {alpha_values}")
    
    for alpha in alpha_values:
        df = _add_single_ewma(df, value_col, date_col, product_col, alpha)
        logger.info(f"   ✓ Added EWMA with alpha={alpha}")
    
    logger.info(f"✅ EWMA features complete! Added {len(alpha_values)} EWMA series")
    
    return df


def _add_single_ewma(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    alpha: float
) -> DataFrame:
    """
    Add a single EWMA feature with specified alpha.
    
    Uses iterative computation: EWMA_t = alpha * X_t + (1-alpha) * EWMA_(t-1)
    """
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    ewma_col = f"ewma_{int(alpha*100)}"
    prev_ewma_col = f"_prev_{ewma_col}"
    
    # Initialize: first value is just the actual value
    # Subsequent values use the EWMA formula
    df = df.withColumn(
        prev_ewma_col,
        lag(col(ewma_col) if ewma_col in df.columns else col(value_col), 1).over(w)
    )
    
    # EWMA calculation
    df = df.withColumn(
        ewma_col,
        when(
            col(prev_ewma_col).isNull(),
            col(value_col)  # First observation
        ).otherwise(
            alpha * col(value_col) + (1 - alpha) * col(prev_ewma_col)
        )
    )
    
    df = df.drop(prev_ewma_col)
    
    return df


def add_ewma_volatility_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    alpha_values: List[float] = [0.3, 0.7]
) -> DataFrame:
    """
    Add exponentially weighted volatility (standard deviation) features.
    
    Volatility is computed as EWMA of squared deviations from EWMA mean.
    
    Args:
        df: Input DataFrame
        value_col: Column name for values
        date_col: Date column name
        product_col: Product identifier
        alpha_values: List of alpha parameters
    
    Returns:
        DataFrame with EWMA volatility features
    """
    logger.info(f"📈 Adding EWMA volatility features with alpha values: {alpha_values}")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    for alpha in alpha_values:
        ewma_col = f"ewma_{int(alpha*100)}"
        ewm_vol_col = f"ewm_volatility_{int(alpha*100)}"
        
        # Ensure EWMA exists
        if ewma_col not in df.columns:
            df = _add_single_ewma(df, value_col, date_col, product_col, alpha)
        
        # Calculate squared deviation from EWMA
        df = df.withColumn(
            "_squared_dev",
            spark_pow(col(value_col) - col(ewma_col), lit(2.0))
        )
        
        # EWMA of squared deviations
        df = df.withColumn(
            "_prev_var",
            lag(col(ewm_vol_col) if ewm_vol_col in df.columns else col("_squared_dev"), 1).over(w)
        )
        
        df = df.withColumn(
            "_ewm_variance",
            when(
                col("_prev_var").isNull(),
                col("_squared_dev")
            ).otherwise(
                alpha * col("_squared_dev") + (1 - alpha) * col("_prev_var")
            )
        )
        
        # Volatility is square root of variance
        df = df.withColumn(
            ewm_vol_col,
            sqrt(col("_ewm_variance"))
        )
        
        df = df.drop("_squared_dev", "_prev_var", "_ewm_variance")
        
        logger.info(f"   ✓ Added {ewm_vol_col}")
    
    logger.info(f"✅ EWMA volatility features complete!")
    
    return df


def add_ewma_momentum_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    fast_alpha: float = 0.7,
    slow_alpha: float = 0.3
) -> DataFrame:
    """
    Add EWMA-based momentum indicators (similar to MACD in finance).
    
    Features:
    - ewma_momentum: fast_ewma - slow_ewma (trend direction and strength)
    - ewma_signal: EWMA of momentum (smoothed momentum)
    - ewma_divergence: momentum - signal (momentum acceleration)
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
        fast_alpha: Alpha for fast EWMA (default 0.7)
        slow_alpha: Alpha for slow EWMA (default 0.3)
    
    Returns:
        DataFrame with momentum features
    """
    logger.info(f"🚀 Adding EWMA momentum features (fast={fast_alpha}, slow={slow_alpha})")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    fast_col = f"ewma_{int(fast_alpha*100)}"
    slow_col = f"ewma_{int(slow_alpha*100)}"
    
    # Ensure both EWMAs exist
    if fast_col not in df.columns:
        df = _add_single_ewma(df, value_col, date_col, product_col, fast_alpha)
    if slow_col not in df.columns:
        df = _add_single_ewma(df, value_col, date_col, product_col, slow_alpha)
    
    # Momentum = fast_ewma - slow_ewma
    df = df.withColumn(
        "ewma_momentum",
        col(fast_col) - col(slow_col)
    )
    
    # Signal = EWMA of momentum (using medium alpha = 0.5)
    signal_alpha = 0.5
    df = df.withColumn(
        "_prev_signal",
        lag(col("ewma_signal") if "ewma_signal" in df.columns else col("ewma_momentum"), 1).over(w)
    )
    
    df = df.withColumn(
        "ewma_signal",
        when(
            col("_prev_signal").isNull(),
            col("ewma_momentum")
        ).otherwise(
            signal_alpha * col("ewma_momentum") + (1 - signal_alpha) * col("_prev_signal")
        )
    )
    
    # Divergence = momentum - signal (histogram)
    df = df.withColumn(
        "ewma_divergence",
        col("ewma_momentum") - col("ewma_signal")
    )
    
    # Momentum direction (binary: increasing=1, decreasing=-1)
    df = df.withColumn(
        "ewma_momentum_direction",
        when(col("ewma_momentum") > 0, lit(1))
        .when(col("ewma_momentum") < 0, lit(-1))
        .otherwise(lit(0))
    )
    
    df = df.drop("_prev_signal")
    
    logger.info("   ✓ Added ewma_momentum, ewma_signal, ewma_divergence, ewma_momentum_direction")
    logger.info("✅ EWMA momentum features complete!")
    
    return df


def add_ewma_ratio_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    alpha: float = 0.5
) -> DataFrame:
    """
    Add ratio-based EWMA features.
    
    Features:
    - value_to_ewma_ratio: current_value / ewma (deviation from trend)
    - ewma_growth_rate: (ewma_t - ewma_(t-1)) / ewma_(t-1)
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
        alpha: Alpha for EWMA calculation
    
    Returns:
        DataFrame with ratio features
    """
    logger.info(f"📊 Adding EWMA ratio features (alpha={alpha})")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    ewma_col = f"ewma_{int(alpha*100)}"
    
    # Ensure EWMA exists
    if ewma_col not in df.columns:
        df = _add_single_ewma(df, value_col, date_col, product_col, alpha)
    
    # Value to EWMA ratio
    df = df.withColumn(
        "value_to_ewma_ratio",
        when(
            col(ewma_col) != 0,
            col(value_col) / col(ewma_col)
        ).otherwise(lit(1.0))
    )
    
    # EWMA growth rate
    df = df.withColumn(
        "_prev_ewma",
        lag(col(ewma_col), 1).over(w)
    )
    
    df = df.withColumn(
        "ewma_growth_rate",
        when(
            (col("_prev_ewma").isNotNull()) & (col("_prev_ewma") != 0),
            (col(ewma_col) - col("_prev_ewma")) / col("_prev_ewma")
        ).otherwise(None)
    )
    
    df = df.drop("_prev_ewma")
    
    logger.info("   ✓ Added value_to_ewma_ratio, ewma_growth_rate")
    logger.info("✅ EWMA ratio features complete!")
    
    return df


def add_adaptive_ewma_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Add adaptive EWMA where alpha adjusts based on recent volatility.
    High volatility → higher alpha (faster adaptation)
    Low volatility → lower alpha (smoother)
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
    
    Returns:
        DataFrame with adaptive EWMA
    """
    logger.info("🎯 Adding adaptive EWMA features")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    w_lookback = w.rowsBetween(-5, 0)
    
    # Calculate recent volatility (coefficient of variation over last 6 periods)
    df = df.withColumn(
        "_recent_mean",
        spark_avg(col(value_col)).over(w_lookback)
    )
    
    df = df.withColumn(
        "_recent_std",
        sqrt(
            spark_avg(
                spark_pow(col(value_col) - col("_recent_mean"), lit(2.0))
            ).over(w_lookback)
        )
    )
    
    df = df.withColumn(
        "_cv",
        when(
            col("_recent_mean") != 0,
            col("_recent_std") / col("_recent_mean")
        ).otherwise(lit(0.5))
    )
    
    # Adaptive alpha: higher CV → higher alpha (0.3 to 0.9 range)
    df = df.withColumn(
        "_adaptive_alpha",
        when(col("_cv") < 0.2, lit(0.3))
        .when(col("_cv") < 0.5, lit(0.5))
        .when(col("_cv") < 1.0, lit(0.7))
        .otherwise(lit(0.9))
    )
    
    # Compute adaptive EWMA
    df = df.withColumn(
        "_prev_adaptive_ewma",
        lag(col("adaptive_ewma") if "adaptive_ewma" in df.columns else col(value_col), 1).over(w)
    )
    
    df = df.withColumn(
        "adaptive_ewma",
        when(
            col("_prev_adaptive_ewma").isNull(),
            col(value_col)
        ).otherwise(
            col("_adaptive_alpha") * col(value_col) + 
            (lit(1.0) - col("_adaptive_alpha")) * col("_prev_adaptive_ewma")
        )
    )
    
    # Keep the adaptive alpha as a feature (indicates volatility regime)
    df = df.withColumn("adaptive_alpha_value", col("_adaptive_alpha"))
    
    df = df.drop("_recent_mean", "_recent_std", "_cv", "_adaptive_alpha", "_prev_adaptive_ewma")
    
    logger.info("   ✓ Added adaptive_ewma, adaptive_alpha_value")
    logger.info("✅ Adaptive EWMA features complete!")
    
    return df
