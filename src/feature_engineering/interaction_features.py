"""
Feature Interaction Engineering Module

Creates non-linear feature combinations to capture complex relationships:
- Multiplicative interactions (lag × seasonality, volatility × trend)
- Polynomial features
- Category-specific interactions
- Time-based modulation features
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, coalesce,
    pow as spark_pow, sqrt, abs as spark_abs,
    sin, cos, month, dayofyear
)
from pyspark.sql.window import Window
from typing import List, Optional
import logging

# Mathematical constant for cyclical encoding
_TWO_PI = 6.283185307179586  # 2 * pi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_interaction_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    interaction_pairs: Optional[List[tuple]] = None
) -> DataFrame:
    """
    Add feature interaction terms to capture non-linear relationships.
    
    Default interactions include:
    - lag features × seasonal features (e.g., lag_1 × month_sin)
    - volatility × trend strength
    - demand × category indicators
    - lag features × lag features (quadratic terms)
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
        interaction_pairs: Optional list of (col1, col2) tuples to interact
    
    Returns:
        DataFrame with interaction features
    
    Example:
        df_interact = add_interaction_features(
            df,
            value_col='DemandQuantity',
            date_col='MonthEndDate',
            product_col='ItemNumber',
            interaction_pairs=[('lag_1', 'month_sin'), ('lag_1', 'rolling_std_3')]
        )
    """
    logger.info("🔄 Adding feature interaction terms")
    
    # Ensure seasonal features exist
    if 'month_sin' not in df.columns or 'month_cos' not in df.columns:
        df = _add_seasonal_encoding(df, date_col)
        logger.info("   ✓ Added seasonal encoding (month_sin, month_cos)")
    
    # Default interaction pairs if not specified
    if interaction_pairs is None:
        interaction_pairs = _get_default_interactions(df)
    
    # Create interactions
    created_count = 0
    for col1, col2 in interaction_pairs:
        if col1 in df.columns and col2 in df.columns:
            interaction_name = f"{col1}_x_{col2}"
            df = df.withColumn(
                interaction_name,
                col(col1) * col(col2)
            )
            created_count += 1
            logger.info(f"   ✓ Created {interaction_name}")
    
    logger.info(f"✅ Interaction features complete! Added {created_count} interactions")
    
    return df


def _get_default_interactions(df: DataFrame) -> List[tuple]:
    """Generate sensible default interaction pairs based on available columns."""
    interactions = []
    
    # Lag × Seasonality interactions
    lag_cols = [c for c in df.columns if c.startswith('lag_')]
    if 'month_sin' in df.columns:
        for lag_col in lag_cols[:3]:  # First 3 lags
            interactions.append((lag_col, 'month_sin'))
            interactions.append((lag_col, 'month_cos'))
    
    # Lag × Lag (quadratic/polynomial)
    if 'lag_1' in df.columns and 'lag_2' in df.columns:
        interactions.append(('lag_1', 'lag_2'))
    if 'lag_1' in df.columns and 'lag_12' in df.columns:
        interactions.append(('lag_1', 'lag_12'))
    
    # Volatility × Trend
    if 'rolling_std_3' in df.columns and 'trend_strength_3m' in df.columns:
        interactions.append(('rolling_std_3', 'trend_strength_3m'))
    
    # Recent demand × Seasonality
    if 'ma_3_month' in df.columns and 'month_sin' in df.columns:
        interactions.append(('ma_3_month', 'month_sin'))
    
    # EWMA × Growth rate
    if 'ewma_50' in df.columns and 'growth_rate_3m' in df.columns:
        interactions.append(('ewma_50', 'growth_rate_3m'))
    
    # Volatility × Product Category (if exists as numeric indicator)
    if 'cov_quantity' in df.columns and 'avg_demand_interval' in df.columns:
        interactions.append(('cov_quantity', 'avg_demand_interval'))
    
    return interactions


def _add_seasonal_encoding(df: DataFrame, date_col: str) -> DataFrame:
    """Add sin/cos encoding for month (if not already present)."""
    # Use pre-computed constant for better performance
    period_factor = _TWO_PI / 12.0
    
    df = df.withColumn(
        "month_sin",
        sin(col("month") * lit(period_factor)) if "month" in df.columns 
        else sin(month(col(date_col)) * lit(period_factor))
    )
    df = df.withColumn(
        "month_cos",
        cos(col("month") * lit(period_factor)) if "month" in df.columns
        else cos(month(col(date_col)) * lit(period_factor))
    )
    return df


def add_polynomial_features(
    df: DataFrame,
    feature_cols: List[str],
    degree: int = 2
) -> DataFrame:
    """
    Add polynomial features (squared, cubed terms).
    
    Args:
        df: Input DataFrame
        feature_cols: List of column names to create polynomial features for
        degree: Polynomial degree (2 = squared, 3 = cubed)
    
    Returns:
        DataFrame with polynomial features
    """
    logger.info(f"📐 Adding polynomial features (degree={degree})")
    
    created_count = 0
    for feat_col in feature_cols:
        if feat_col not in df.columns:
            continue
        
        for d in range(2, degree + 1):
            poly_name = f"{feat_col}_pow{d}"
            df = df.withColumn(
                poly_name,
                spark_pow(col(feat_col), lit(float(d)))
            )
            created_count += 1
            logger.info(f"   ✓ Created {poly_name}")
    
    logger.info(f"✅ Polynomial features complete! Added {created_count} features")
    
    return df


def add_ratio_features(
    df: DataFrame,
    numerator_cols: List[str],
    denominator_cols: List[str]
) -> DataFrame:
    """
    Add ratio features (numerator / denominator).
    
    Useful for:
    - lag_1 / lag_12 (current vs year-ago comparison)
    - value / ma_12 (current vs long-term average)
    - rolling_std / rolling_mean (coefficient of variation)
    
    Args:
        df: Input DataFrame
        numerator_cols: List of numerator column names
        denominator_cols: List of denominator column names
    
    Returns:
        DataFrame with ratio features
    """
    logger.info(f"➗ Adding ratio features")
    
    created_count = 0
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col in df.columns and den_col in df.columns:
                ratio_name = f"{num_col}_over_{den_col}"
                df = df.withColumn(
                    ratio_name,
                    when(
                        col(den_col) != 0,
                        col(num_col) / col(den_col)
                    ).otherwise(lit(0.0))
                )
                created_count += 1
                logger.info(f"   ✓ Created {ratio_name}")
    
    logger.info(f"✅ Ratio features complete! Added {created_count} ratios")
    
    return df


def add_category_interaction_features(
    df: DataFrame,
    numeric_cols: List[str],
    category_col: str = 'product_category'
) -> DataFrame:
    """
    Add category-specific feature transformations.
    Creates separate features for each category level.
    
    Example: lag_1_Smooth, lag_1_Erratic, etc.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns to create category interactions for
        category_col: Category column name
    
    Returns:
        DataFrame with category interaction features
    """
    logger.info(f"🏷️ Adding category interaction features for '{category_col}'")
    
    if category_col not in df.columns:
        logger.warning(f"   Category column '{category_col}' not found, skipping")
        return df
    
    # Get distinct categories
    categories = [row[0] for row in df.select(category_col).distinct().collect()]
    
    created_count = 0
    for num_col in numeric_cols:
        if num_col not in df.columns:
            continue
        
        for category in categories:
            if category:  # Skip null categories
                interaction_name = f"{num_col}_{category}"
                df = df.withColumn(
                    interaction_name,
                    when(
                        col(category_col) == category,
                        col(num_col)
                    ).otherwise(lit(0.0))
                )
                created_count += 1
    
    logger.info(f"   ✓ Created {created_count} category-specific features")
    logger.info(f"✅ Category interaction features complete!")
    
    return df


def add_temporal_modulation_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Add time-modulated features that capture how patterns change over time.
    
    Features:
    - demand_x_time_index: Value weighted by time position
    - seasonal_strength_x_lifecycle: Seasonality modulated by lifecycle stage
    - trend_x_season: Trend strength modulated by season
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
    
    Returns:
        DataFrame with temporal modulation features
    """
    logger.info("⏰ Adding temporal modulation features")
    
    # Ensure time_index exists
    if 'time_index' not in df.columns:
        w = Window.partitionBy(product_col).orderBy(date_col)
        from pyspark.sql.functions import row_number
        df = df.withColumn("time_index", row_number().over(w))
    
    # Value × Time Index (captures growth/decline over time)
    df = df.withColumn(
        "value_x_time_index",
        col(value_col) * col("time_index")
    )
    logger.info("   ✓ Added value_x_time_index")
    
    # Seasonal × Trend interactions
    if 'month_sin' in df.columns and 'trend_strength_3m' in df.columns:
        df = df.withColumn(
            "season_x_trend",
            col("month_sin") * col("trend_strength_3m")
        )
        logger.info("   ✓ Added season_x_trend")
    
    # Lag × Time position (how recent lags relate to time)
    if 'lag_1' in df.columns:
        df = df.withColumn(
            "lag1_x_time",
            col("lag_1") * col("time_index")
        )
        logger.info("   ✓ Added lag1_x_time")
    
    # Volatility × Recency (is volatility increasing or decreasing over time?)
    if 'rolling_std_3' in df.columns:
        df = df.withColumn(
            "volatility_x_time",
            col("rolling_std_3") * col("time_index")
        )
        logger.info("   ✓ Added volatility_x_time")
    
    logger.info("✅ Temporal modulation features complete!")
    
    return df


def add_lag_interaction_features(
    df: DataFrame,
    max_lag: int = 5
) -> DataFrame:
    """
    Add interactions between different lag features.
    Captures relationships between recent and distant past.
    
    Args:
        df: Input DataFrame
        max_lag: Maximum lag to consider
    
    Returns:
        DataFrame with lag interaction features
    """
    logger.info(f"🔗 Adding lag interaction features (max_lag={max_lag})")
    
    # Find available lag columns
    lag_cols = [f"lag_{i}" for i in range(1, max_lag + 1) if f"lag_{i}" in df.columns]
    
    if len(lag_cols) < 2:
        logger.warning("   Not enough lag features found, skipping")
        return df
    
    created_count = 0
    
    # Adjacent lag products (lag_1 × lag_2, lag_2 × lag_3, etc.)
    for i in range(len(lag_cols) - 1):
        interaction_name = f"{lag_cols[i]}_x_{lag_cols[i+1]}"
        df = df.withColumn(
            interaction_name,
            col(lag_cols[i]) * col(lag_cols[i+1])
        )
        created_count += 1
        logger.info(f"   ✓ Created {interaction_name}")
    
    # Recent × Distant (lag_1 × lag_12 if lag_12 exists)
    if 'lag_1' in df.columns and 'lag_12' in df.columns:
        df = df.withColumn(
            "lag1_x_lag12",
            col("lag_1") * col("lag_12")
        )
        created_count += 1
        logger.info(f"   ✓ Created lag1_x_lag12")
    
    logger.info(f"✅ Lag interaction features complete! Added {created_count} interactions")
    
    return df


def add_all_interaction_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    include_polynomial: bool = True,
    include_category: bool = True
) -> DataFrame:
    """
    Convenience function to add all interaction feature types.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
        include_polynomial: Whether to add polynomial features
        include_category: Whether to add category interactions
    
    Returns:
        DataFrame with all interaction features
    """
    logger.info("🚀 Adding ALL interaction features")
    
    # Basic interactions
    df = add_interaction_features(df, value_col, date_col, product_col)
    
    # Temporal modulation
    df = add_temporal_modulation_features(df, value_col, date_col, product_col)
    
    # Lag interactions
    df = add_lag_interaction_features(df, max_lag=5)
    
    # Polynomial features for key lags
    if include_polynomial:
        poly_cols = ['lag_1', 'lag_2', 'lag_3', value_col]
        poly_cols = [c for c in poly_cols if c in df.columns]
        if poly_cols:
            df = add_polynomial_features(df, poly_cols, degree=2)
    
    # Category interactions
    if include_category and 'product_category' in df.columns:
        category_cols = ['lag_1', 'rolling_std_3', 'ma_3_month']
        category_cols = [c for c in category_cols if c in df.columns]
        if category_cols:
            df = add_category_interaction_features(df, category_cols, 'product_category')
    
    logger.info("✅ ALL interaction features complete!")
    
    return df
